import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import glob
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import deepspeed
import argparse
from tqdm import tqdm
import json


# 设置随机种子确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 自定义数据集类
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, train_split=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 获取所有图片路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_path in glob.glob(os.path.join(class_dir, '*.*')):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((img_path, class_idx))

        # 划分训练集和验证集
        random.Random(seed).shuffle(self.samples)
        split_idx = int(len(self.samples) * train_split)
        if is_train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义模型 - 添加混合精度支持
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        # 使用新的 weights 参数而不是 pretrained
        from torchvision.models import EfficientNet_B6_Weights
        self.base_model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)

        # 冻结底层参数
        for i, param in enumerate(self.base_model.parameters()):
            if i < 200:
                param.requires_grad = False

        # 获取最后一层的输入特征数
        in_features = self.base_model.classifier[1].in_features

        # 替换分类器
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


# 学习率调度函数
def get_lr(epoch, initial_lr=1e-4):
    """更激进的学习率调度以快速收敛"""
    warmup_epochs = 3
    decay_start = 10

    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    elif epoch < decay_start:
        return initial_lr
    else:
        return initial_lr * np.exp(0.1 * (decay_start - epoch))


# 训练函数 - 使用 PyTorch AMP (与 DeepSpeed 兼容)
def train_epoch(model, train_loader, criterion, optimizer, epoch, device, local_rank, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 使用自动混合精度 - DeepSpeed 会自动处理梯度缩放
        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # DeepSpeed backward and step (无需手动 scaler)
        model.backward(loss)
        model.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if local_rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100*correct/total})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# 验证函数
def validate(model, val_loader, criterion, device, local_rank, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 关键修复: 跨GPU聚合验证指标
    # 将损失和总数转换为 tensor 以便同步
    loss_tensor = torch.tensor([running_loss], device=device)
    total_tensor = torch.tensor([total], device=device)
    correct_tensor = torch.tensor([correct], device=device)

    # AllReduce 聚合所有 GPU 的统计量
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)

    # 计算全局平均指标(所有GPU看到的值相同)
    val_loss = loss_tensor.item() / len(val_loader) / torch.distributed.get_world_size()
    val_acc = 100 * correct_tensor.item() / total_tensor.item()

    return val_loss, val_acc, all_preds, all_labels


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Animal Classification with DeepSpeed')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--data_dir', type=str, default='Animal', help='数据集路径')
    parser.add_argument('--num_classes', type=int, default=100, help='类别数量')
    parser.add_argument('--img_size', type=int, default=456, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=12, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--initial_lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout比率')
    parser.add_argument('--patience', type=int, default=5, help='早停等待轮数')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练')

    # DeepSpeed 会添加额外的参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = AnimalDataset(args.data_dir, transform=train_transform, is_train=True)
    val_dataset = AnimalDataset(args.data_dir, transform=val_transform, is_train=False)

    # 使用分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    if args.local_rank == 0:
        print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
        print(f"类别数: {args.num_classes}")
        print(f"混合精度训练: {'启用 (BF16)' if args.use_amp else '禁用 (FP32)'}")

    # 创建模型
    model = EfficientNetClassifier(num_classes=args.num_classes, dropout_rate=args.dropout_rate)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)

    # 初始化 DeepSpeed - 对于 AMP,禁用 DeepSpeed 的 FP16/BF16
    # 创建临时配置
    if args.use_amp:
        # 读取配置文件并禁用其 FP16/BF16
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        ds_config['fp16']['enabled'] = False
        ds_config['bf16']['enabled'] = False
        args.deepspeed_config = ds_config

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # 训练循环
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # 调整学习率
        current_lr = get_lr(epoch, args.initial_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        if args.local_rank == 0:
            print(f'\nEpoch {epoch+1}/{args.epochs}, Learning Rate: {current_lr:.6f}')

        # 训练
        train_loss, train_acc = train_epoch(
            model_engine, train_loader, criterion, optimizer,
            epoch+1, device, args.local_rank, args.use_amp
        )

        # 验证
        val_loss, val_acc, val_preds, val_labels = validate(
            model_engine, val_loader, criterion, device, args.local_rank, args.use_amp
        )

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if args.local_rank == 0:
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型 - DeepSpeed checkpoint 需要所有 rank 参与
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 所有 rank 都需要调用 save_checkpoint (DeepSpeed 要求)
            model_engine.save_checkpoint('checkpoints', 'best_model')
            if args.local_rank == 0:
                print(f'模型已保存,验证损失: {val_loss:.4f}')
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= args.patience:
            if args.local_rank == 0:
                print(f'\n早停触发,已等待 {args.patience} 轮无改善')
            break

    # 保存训练历史(仅在主进程)
    if args.local_rank == 0:
        history_df = pd.DataFrame(history)
        history_df.to_csv('training_history.csv', index=False)

        # 绘制训练曲线
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.title('Training & Validation Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='train acc')
        plt.plot(history['val_acc'], label='val acc')
        plt.title('Training & Validation Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(history['lr'], label='learning rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.bar(['train', 'val'], [len(train_dataset), len(val_dataset)],
                color=['blue', 'orange'])
        plt.title('Dataset Distribution')

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=150)
        plt.close()

        # 生成分类报告
        class_names = train_dataset.classes
        cm = confusion_matrix(val_labels, val_preds)
        class_report = classification_report(
            val_labels, val_preds,
            target_names=class_names,
            output_dict=True
        )

        # 保存分类报告
        class_metrics = []
        for i, class_name in enumerate(class_names):
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            support = class_report[class_name]['support']

            class_errors = cm[i].copy()
            class_errors[i] = 0
            if np.sum(class_errors) > 0:
                main_error_idx = np.argmax(class_errors)
                main_error_class = class_names[main_error_idx]
                error_count = class_errors[main_error_idx]
                error_percent = error_count / np.sum(class_errors) * 100
            else:
                main_error_class = "无错误"
                error_count = 0
                error_percent = 0

            class_metrics.append({
                '类别': class_name,
                '精确率': precision,
                '召回率': recall,
                'F1分数': f1,
                '样本数': support,
                '主要误判类别': main_error_class,
                '误判数量': error_count,
                '误判占比(%)': error_percent
            })

        metrics_df = pd.DataFrame(class_metrics)
        metrics_df.to_csv('class_accuracy_report.csv', index=False)

        print("\n" + "="*50)
        print("训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"训练历史已保存为: training_history.csv")
        print(f"类别准确率报告已保存为: class_accuracy_report.csv")
        print("="*50)


if __name__ == '__main__':
    main()

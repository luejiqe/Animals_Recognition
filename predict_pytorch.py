import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# 配置参数
model_path = './pytorch_model.pt'  # PyTorch 模型权重路径
image_dir = 'test'  # 图片目录路径
output_csv = 'predictions.csv'  # 预测结果保存路径
img_size = (456, 456)  # 与训练时相同的尺寸
class_names_path = 'class.txt'  # 类别名称文件路径
num_classes = 100  # 类别数量
use_gpu = True  # 设置为 True 使用 GPU，False 使用 CPU

# 自动检测设备
if use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("使用 CPU 进行推理")


# 定义模型(与训练时相同)
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
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


# 加载模型
print("正在加载模型...")
try:
    # 创建模型
    model = EfficientNetClassifier(num_classes=num_classes)

    # 加载标准 PyTorch 模型权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        print("✅ 模型加载成功!")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 convert_final.py 转换 DeepSpeed 检查点")
        exit(1)

    # 将模型移到目标设备
    model = model.to(device)
    model.eval()  # 设置为评估模式
    print("模型加载成功!")

except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 加载类别名称
if os.path.exists(class_names_path):
    with open(class_names_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"已加载 {len(class_names)} 个类别名称")
else:
    class_names = None
    print("未找到类别名称文件,将使用数字标签")

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 支持的图片格式
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 获取图片文件列表
image_files = []
for root, _, files in os.walk(image_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(root, file))

if not image_files:
    print(f"在目录 {image_dir} 中未找到图片文件")
    exit(1)

print(f"找到 {len(image_files)} 张待预测图片")

# 创建结果列表
results = []

# 批量处理图片
with torch.no_grad():  # 关闭梯度计算
    for i, img_path in enumerate(image_files):
        try:
            # 加载和预处理图片
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)  # 添加批次维度并移到设备 [1, 3, 456, 456]

            # 预测
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]  # 转换为概率
            predictions = probabilities.cpu().numpy()  # 移回 CPU 用于后续处理

            top_idx = np.argmax(predictions)
            confidence = predictions[top_idx]

            # 获取类别名称
            if class_names and len(class_names) > top_idx:
                predicted_class = class_names[top_idx]
            else:
                predicted_class = str(top_idx)

            # 获取top3预测结果
            top3_indices = np.argsort(predictions)[::-1][:3]
            top3_classes = []
            top3_confidences = []

            for idx in top3_indices:
                if class_names and len(class_names) > idx:
                    top3_classes.append(class_names[idx])
                else:
                    top3_classes.append(str(idx))
                top3_confidences.append(float(predictions[idx]))

            # 添加到结果
            result = {
                'file_path': img_path,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'top1_class': top3_classes[0],
                'top1_confidence': top3_confidences[0],
                'top2_class': top3_classes[1] if len(top3_classes) > 1 else '',
                'top2_confidence': top3_confidences[1] if len(top3_confidences) > 1 else 0.0,
                'top3_class': top3_classes[2] if len(top3_classes) > 2 else '',
                'top3_confidence': top3_confidences[2] if len(top3_confidences) > 2 else 0.0
            }

            results.append(result)

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
            continue

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
            print(f"已处理 {i+1}/{len(image_files)} 张图片")

# 保存结果到CSV
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文

    print(f"\n预测完成!结果已保存至: {output_csv}")
    print("="*50)
    print("结果预览:")
    print(df[['file_path', 'predicted_class', 'confidence']].head())
else:
    print("没有成功处理任何图片")

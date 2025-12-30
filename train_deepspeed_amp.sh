#!/bin/bash

# 双卡A800训练启动脚本 - 使用 PyTorch AMP + DeepSpeed ZeRO-2
# 这个脚本使用 PyTorch 原生的自动混合精度,避免 DeepSpeed FP16/BF16 的兼容性问题

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网卡名称调整

# 训练参数
DATA_DIR="Animal"
NUM_CLASSES=100
IMG_SIZE=456
BATCH_SIZE=12  # 每个GPU的batch size
EPOCHS=20
INITIAL_LR=1e-4
DROPOUT_RATE=0.3
PATIENCE=5

# DeepSpeed 配置文件 (使用 FP32 配置,AMP 由 PyTorch 处理)
DS_CONFIG="ds_config_zero2_fp32.json"

# 创建检查点目录
mkdir -p checkpoints

# 启动分布式训练 - 带 AMP
deepspeed --num_gpus=2 \
    train_pytorch_amp.py \
    --deepspeed \
    --deepspeed_config ${DS_CONFIG} \
    --data_dir ${DATA_DIR} \
    --num_classes ${NUM_CLASSES} \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --initial_lr ${INITIAL_LR} \
    --dropout_rate ${DROPOUT_RATE} \
    --patience ${PATIENCE} \
    --use_amp

echo "训练完成!"

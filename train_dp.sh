#!/bin/bash
# PyTorch DataParallel 训练启动脚本 (混合精度)

echo "=================================================="
echo "启动 PyTorch DataParallel 训练 (BF16)"
echo "=================================================="

# 默认参数
DATA_DIR="${DATA_DIR:-Animal}"
NUM_CLASSES="${NUM_CLASSES:-100}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GPU_IDS="${GPU_IDS:-0,1}"

# 计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
PER_GPU_BATCH=$((BATCH_SIZE / NUM_GPUS))

echo "配置:"
echo "  数据目录: $DATA_DIR"
echo "  类别数: $NUM_CLASSES"
echo "  训练轮数: $EPOCHS"
echo "  GPU IDs: $GPU_IDS"
echo "  GPU 数量: $NUM_GPUS"
echo "  总 Batch Size: $BATCH_SIZE"
echo "  每GPU Batch Size: $PER_GPU_BATCH (DP会自动分割)"
echo "  ⚠️  注意: DP的batch_size是总batch,会被自动分割到各GPU"
echo "=================================================="

python train_pytorch_dp.py \
    --data_dir $DATA_DIR \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --gpu_ids $GPU_IDS \
    --use_amp

echo ""
echo "训练完成! 模型已保存到 checkpoints/pytorch_model_dp.pt"

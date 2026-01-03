#!/bin/bash
# DeepSpeed ZeRO-3 训练启动脚本 (显存优化)

echo "=================================================="
echo "启动 DeepSpeed ZeRO-3 训练 (显存优化)"
echo "=================================================="

# 默认参数
DATA_DIR="${DATA_DIR:-Animal}"
NUM_CLASSES="${NUM_CLASSES:-100}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_GPUS="${NUM_GPUS:-2}"

echo "配置:"
echo "  数据目录: $DATA_DIR"
echo "  类别数: $NUM_CLASSES"
echo "  训练轮数: $EPOCHS"
echo "  Batch Size (per GPU): $BATCH_SIZE"
echo "  GPU 数量: $NUM_GPUS"
echo "  ZeRO 阶段: 3 (分片所有状态)"
echo "  预计显存: ~28GB/GPU"
echo "  预计速度: ~5.8 it/s"
echo "=================================================="

deepspeed --num_gpus=$NUM_GPUS train_pytorch_amp.py \
    --deepspeed \
    --deepspeed_config ds_config_zero3.json \
    --data_dir $DATA_DIR \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --use_amp

echo ""
echo "训练完成! 模型已保存到 checkpoints/pytorch_model.pt"

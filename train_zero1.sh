#!/bin/bash
# DeepSpeed ZeRO-1 训练启动脚本 (性能优先)

echo "=================================================="
echo "启动 DeepSpeed ZeRO-1 训练 (性能优先)"
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
echo "  ZeRO 阶段: 1 (仅分片优化器)"
echo "  预计显存: ~38GB/GPU"
echo "  预计速度: ~7.5 it/s"
echo "=================================================="

deepspeed --num_gpus=$NUM_GPUS train_pytorch_amp.py \
    --deepspeed \
    --deepspeed_config ds_config_zero1.json \
    --data_dir $DATA_DIR \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --use_amp

echo ""
echo "训练完成! 模型已保存到 checkpoints/pytorch_model.pt"

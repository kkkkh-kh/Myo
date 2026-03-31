export CUDA_VISIBLE_DEVICES=1

export ROOT_DIR=$PWD
export DATA_DIR=$PWD/datasets
export TRAIN_FILE=train_augmented.tsv
export OUTPUT_DIR=$PWD/checkpoints/word_order_mix
export CONFIG_PATH=$PWD/configs/default.yaml

# ✅ 续训：指定 checkpoint 路径
export RESUME_CKPT=$PWD/checkpoints/word_order_mix/best_model.pt

# ✅ 是否恢复 optimizer/scheduler 状态（续训中断用 1，微调新数据用 0）
export RESUME_OPTIMIZER=1

nohup python scripts/train.py \
    --resume "$RESUME_CKPT" \
    --resume_optimizer \
    > train_mix_gpu1_resume.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志：tail -f train_mix_gpu1_resume.log"

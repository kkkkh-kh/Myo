#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export ROOT_DIR="${ROOT_DIR:-$PWD}"
export DATA_DIR="${DATA_DIR:-$ROOT_DIR/datasets}"
export TRAIN_FILE="${TRAIN_FILE:-train_augmented.tsv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/word_order_mix}"
export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/default.yaml}"

# Resume settings:
# 1) explicit checkpoint path via RESUME_FROM
# 2) or auto resume via AUTO_RESUME=1 from OUTPUT_DIR/latest_model.pt
export RESUME_FROM="${RESUME_FROM:-}"
export AUTO_RESUME="${AUTO_RESUME:-0}"

# Keep vocab rebuild aligned with current preprocess by default.
export REUSE_CHECKPOINT_VOCAB="${REUSE_CHECKPOINT_VOCAB:-0}"

# Data augmentation stage control for scripts/train.sh
export SKIP_AUGMENT="${SKIP_AUGMENT:-0}"

LOG_FILE="${LOG_FILE:-train_word_order_mix.log}"

nohup bash scripts/train.sh > "$LOG_FILE" 2>&1 &

echo "Training started in background. PID: $!"
echo "Log: tail -f $LOG_FILE"

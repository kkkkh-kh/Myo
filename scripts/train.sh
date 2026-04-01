#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints}"
AUGMENT_RATIO="${AUGMENT_RATIO:-4.0}"
TRAIN_FILE="${TRAIN_FILE:-train_augmented.tsv}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-0}"
SKIP_AUGMENT="${SKIP_AUGMENT:-0}"
REUSE_CHECKPOINT_VOCAB="${REUSE_CHECKPOINT_VOCAB:-0}"

if [[ ! -f "$DATA_DIR/train.tsv" && ! -f "$DATA_DIR/train.csv" ]]; then
  echo "未找到训练数据（train.tsv/train.csv）。" >&2
  exit 1
fi

INPUT_TRAIN="$DATA_DIR/train.tsv"
if [[ ! -f "$INPUT_TRAIN" ]]; then
  INPUT_TRAIN="$DATA_DIR/train.csv"
fi

VAL_FILE="$DATA_DIR/val.tsv"
if [[ ! -f "$VAL_FILE" ]]; then
  if [[ -f "$DATA_DIR/dev.tsv" ]]; then
    VAL_FILE="$DATA_DIR/dev.tsv"
  elif [[ -f "$DATA_DIR/val.csv" ]]; then
    VAL_FILE="$DATA_DIR/val.csv"
  else
    VAL_FILE="$DATA_DIR/dev.csv"
  fi
fi

if [[ "$SKIP_AUGMENT" == "1" ]]; then
  echo "===== Step 0: skip data augmentation (SKIP_AUGMENT=1) ====="
else
  echo "===== Step 0: data augmentation preprocess ====="
  cd "$ROOT_DIR"
  "$PYTHON_BIN" datasets/preprocess_augment.py \
    --input "$INPUT_TRAIN" \
    --output "$DATA_DIR/$TRAIN_FILE" \
    --augment_ratio "$AUGMENT_RATIO" \
    --seed 42 \
    --config "$CONFIG_PATH"

  echo "augmentation stats:"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
stats_path = Path("$DATA_DIR") / "augment_stats.json"
if not stats_path.exists():
    print("  augment_stats.json not found")
else:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    print(f"  original_count: {stats.get('original_count', 0)}")
    print(f"  augmented_count: {stats.get('augmented_count', 0)}")
    for k, v in stats.get("strategy_counts", {}).items():
        print(f"  {k}: +{v}")
PY
fi

echo "===== Step 1: 训练（使用增强数据集） ====="
ROOT_DIR="$ROOT_DIR" \
DATA_DIR="$DATA_DIR" \
TRAIN_FILE="$TRAIN_FILE" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
RESUME_FROM="$RESUME_FROM" \
AUTO_RESUME="$AUTO_RESUME" \
REUSE_CHECKPOINT_VOCAB="$REUSE_CHECKPOINT_VOCAB" \
"$PYTHON_BIN" scripts/train.py

EVAL_INPUT="$VAL_FILE"
if [[ "$VAL_FILE" == *.csv ]]; then
  EVAL_INPUT="$OUTPUT_DIR/val_gloss.txt"
  "$PYTHON_BIN" -c "
import csv
from pathlib import Path
src = Path('$VAL_FILE')
out = Path('$EVAL_INPUT')
with src.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    gloss_col = None
    for name in (reader.fieldnames or []):
        normalized = ''.join((name or '').strip().lower().split())
        if normalized in {'gloss', 'glosssequence', 'glosssentence'}:
            gloss_col = name
            break
    if gloss_col is None:
        raise RuntimeError(f'Cannot find Gloss column in {reader.fieldnames}')
    lines = []
    for row in reader:
        gloss = (row.get(gloss_col) or '').strip()
        if gloss:
            lines.append(gloss)
out.write_text('\\n'.join(lines) + ('\\n' if lines else ''), encoding='utf-8')
print(f'生成评估输入: {out} ({len(lines)} 条)')
"
fi

echo "===== Step 2: 评估（使用原始验证集，不增强） ====="
"$PYTHON_BIN" inference/translate.py \
  --file "$EVAL_INPUT" \
  --output "$OUTPUT_DIR/val_predictions.txt" \
  --model_dir "$OUTPUT_DIR" \
  --config "$CONFIG_PATH"

echo "===== 完成 ====="

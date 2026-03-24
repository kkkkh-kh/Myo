#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-$MODEL_DIR/runtime_config.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$ROOT_DIR/configs/default.yaml"
fi

PYTHON_CMD=()

resolve_python_cmd() {
  local conda_prefix_unix=""
  local candidate=""
  local candidates=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
    return 0
  fi

  if [[ -n "${CONDA_PREFIX:-}" ]] && command -v cygpath >/dev/null 2>&1; then
    conda_prefix_unix="$(cygpath -u "$CONDA_PREFIX" 2>/dev/null || true)"
  fi

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/python.exe" "${CONDA_PREFIX}/bin/python")
  fi

  if [[ -n "$conda_prefix_unix" ]]; then
    candidates+=("${conda_prefix_unix}/python.exe" "${conda_prefix_unix}/bin/python")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      PYTHON_CMD=("$candidate")
      return 0
    fi
  done

  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python)")
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("$(command -v python3)")
    return 0
  fi

  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD=("py" "-3")
    return 0
  fi

  echo "未找到可用的 Python 解释器。请先激活 conda 环境，或显式设置 PYTHON_BIN。" >&2
  echo "示例：PYTHON_BIN=\"/c/Users/你的用户名/.conda/envs/sign-language-translator/python.exe\" bash scripts/deploy.sh" >&2
  return 1
}

resolve_python_cmd
echo "使用 Python: ${PYTHON_CMD[*]}"

ROOT_DIR="$ROOT_DIR" MODEL_DIR="$MODEL_DIR" CONFIG_PATH="$CONFIG_PATH" PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
"${PYTHON_CMD[@]}" - <<'PY'
import os
import sys
from pathlib import Path

ROOT_DIR = Path(os.environ["ROOT_DIR"])
MODEL_DIR = Path(os.environ["MODEL_DIR"])
CONFIG_PATH = Path(os.environ["CONFIG_PATH"])

if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from deploy.export_onnx import export_to_onnx, load_model_for_export
from deploy.quantize import quantize_models

if not (MODEL_DIR / "gloss_vocab.json").exists() or not (MODEL_DIR / "zh_vocab.json").exists():
    raise FileNotFoundError("未找到词表文件，请先运行训练脚本。")

model, checkpoint_path, _ = load_model_for_export(
    save_dir=MODEL_DIR.as_posix(),
    config_path=CONFIG_PATH.as_posix(),
)
model.eval()
export_to_onnx(model, MODEL_DIR.as_posix(), checkpoint_path=checkpoint_path.as_posix())
quantize_models(MODEL_DIR.as_posix(), MODEL_DIR.as_posix())
print(f"部署文件已生成到：{MODEL_DIR}")
PY

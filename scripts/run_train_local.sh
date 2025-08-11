#!/usr/bin/env bash
set -euox pipefail

# Usage: bash scripts/run_train_local.sh [CONFIG_PATH]
# Default config: configs/llama2-7b-gsm8k.yaml

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/failed-llama2-7b-gsm8k.yaml}"

# Environment tuning for H100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Weights & Biases project (can be overridden by env)
export WANDB_PROJECT="${WANDB_PROJECT:-llama2-7b-gsm8k-grpo}"

echo "Using config: ${CONFIG_PATH}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU info:"
  nvidia-smi || true
fi

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# Single-GPU run
python train.py --config "${CONFIG_PATH}"
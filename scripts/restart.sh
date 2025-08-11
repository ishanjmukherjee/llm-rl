#!/usr/bin/env bash
set -euo pipefail

# Repo root for consistent paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

ENV_PATH="/home/ubuntu/anaconda3/envs/grpo"
CFG_PATH="configs/failed-llama2-7b-gsm8k.yaml"

# Stop any running training job (ignore if none running)
pkill -f "python train.py --config ${CFG_PATH}" || true

# Wait for process to terminate
sleep 3

# Create logs directory and log file with timestamp
mkdir -p logs
LOG="logs/qwen3b_$(date +%s)_restart.log"

{
  echo "==== Restart at $(date -Is) ===="
  echo "Env: ${ENV_PATH}"
  echo "Config: ${CFG_PATH}"
} >> "$LOG"

# Launch training using the conda environment (append output to log)
( PATH="${ENV_PATH}/bin:$PATH" bash scripts/run_train_local.sh "${CFG_PATH}" >> "$LOG" 2>&1 ) & disown

# Optionally follow the logs (default off to avoid blocking CI)
TAIL_LOGS="${TAIL_LOGS:-0}"
if [ "$TAIL_LOGS" = "1" ]; then
  echo "Tailing logs: $LOG"
  exec tail -n +1 -F "$LOG"
else
  echo "Launched training. Logs: $LOG"
fi
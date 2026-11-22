#!/usr/bin/env bash
# train_unit Evo adapter — hyperparameters live in the adapter JSON.
# Runs in background (nohup). Edit the JSON before starting.
#
#   bash scripts/run_train_unit_evo.sh
#   tail -f logs/train_unit_evo.log
#   kill $(cat logs/train_unit_evo.pid)  

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

mkdir -p logs
LOG="logs/train_unit_evo.log"
PID_FILE="logs/train_unit_evo.pid"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PID_FILE")  log=$LOG" >&2
  exit 1
fi

# Need --use_deepspeed so accelerate sets RANK/WORLD_SIZE (avoids mpi4py).
nohup .venv/bin/accelerate launch \
  --use_deepspeed \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file Evo_depth/ds_config.json \
  --main_process_port 29519 \
  -m Evo_depth.train_unit.cli \
  --adapter evo \
  --adapter-config Evo_depth/train_unit/adapters/evo_adapter.example.json \
  >>"$LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "started pid=$(cat "$PID_FILE")  log=$LOG"
echo "  tail -f $LOG"

#!/usr/bin/env bash
# LiteGauss LIBERO training (true batch_size=16). Does not touch Evo train.sh.
#
#   bash scripts/run_litegauss_train.sh --selfcheck
#   bash scripts/run_litegauss_train.sh --name run2-5k --steps 5000
#   bash scripts/run_litegauss_train.sh --name smoke --steps 100 --batch-size 16
#   bash scripts/run_litegauss_train.sh --name auto --batch-size 64 --probe-max-batch
# --probe-max-batch: find largest true B <= --batch-size fitting CUDA VRAM before training.
#   DATASET_CONFIG=/path/to/config.libero.yaml bash scripts/run_litegauss_train.sh --name x --steps 100
#
# Checkpoint dir: checkpoints/litegauss/<name>/{config.json,norm_stats.json,model.pt}
# Needs: setup_train_server.sh (.venv), setup_dataset.sh (config + data) for real runs.
# See scripts/litegauss_train.md for full command recipes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LITEGAUSS_PY:-$ROOT/.venv/bin/python}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero.yaml}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
OUTPUT="${OUTPUT:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
RUN_NAME="${RUN_NAME:-${SWANLAB_RUN:-}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-litegauss}"
SELFCHECK=0
NO_SWANLAB=0
PROBE_MAX_BATCH=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
# swanlab 0.10 + some wandb installs; harmless if unused
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

log() { printf '\n==> %s\n' "$*"; }
# CompShare boxes often set a dead cluster proxy; strip it for HF/github pulls.
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

# Safe checkpoint / SwanLab slug: keep alnum, dash, underscore, dot.
sanitize_name() {
  local raw="$1"
  local out
  out="$(printf '%s' "$raw" | tr -c 'A-Za-z0-9._-' '_')"
  out="${out##_}"
  out="${out%%_}"
  [[ -n "$out" ]] || out="litegauss"
  printf '%s' "$out"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --selfcheck) SELFCHECK=1; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --probe-max-batch) PROBE_MAX_BATCH=1; shift ;;
    --lr) LR="$2"; shift 2 ;;
    --warmup-steps|--warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --name|--run-name) RUN_NAME="$2"; shift 2 ;;
    --swanlab-project) SWANLAB_PROJECT="$2"; shift 2 ;;
    --no-swanlab) NO_SWANLAB=1; shift ;;
    --dataset_config_path) DATASET_CONFIG="$2"; shift 2 ;;
    --config) CONFIG_JSON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/Evo_depth"

if [[ "$SELFCHECK" -eq 1 ]]; then
  log "LiteGauss selfcheck (no data / no GPU weights required if using system torch)"
  if [[ -x "$PY" ]]; then
    domestic "$PY" Evo_depth/scripts/train_litegauss_libero.py --selfcheck
  else
    domestic python Evo_depth/scripts/train_litegauss_libero.py --selfcheck
  fi
  exit 0
fi

[[ -x "$PY" ]] || { echo "missing $PY. Run scripts/setup_train_server.sh or set LITEGAUSS_PY." >&2; exit 1; }
[[ -f "$DATASET_CONFIG" ]] || {
  echo "missing $DATASET_CONFIG. Run scripts/setup_dataset.sh or set DATASET_CONFIG." >&2
  exit 1
}

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="litegauss-steps${STEPS}"
fi
RUN_NAME="$(sanitize_name "$RUN_NAME")"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$ROOT/checkpoints/litegauss/${RUN_NAME}"
fi
if [[ -e "$OUTPUT" ]]; then
  echo "checkpoint path already exists (refusing overwrite): $OUTPUT" >&2
  exit 1
fi

cmd=(
  domestic
  HF_ENDPOINT="$HF_ENDPOINT"
  HF_HUB_DISABLE_XET=1
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  SWANLAB_MODE="$SWANLAB_MODE"
  "$PY" Evo_depth/scripts/train_litegauss_libero.py
  --dataset_config_path "$DATASET_CONFIG"
  --steps "$STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --warmup_steps "$WARMUP_STEPS"
  --output "$OUTPUT"
  --swanlab-project "$SWANLAB_PROJECT"
  --swanlab-run "$RUN_NAME"
)
[[ -n "$CONFIG_JSON" ]] && cmd+=(--config "$CONFIG_JSON")
[[ "$NO_SWANLAB" -eq 1 ]] && cmd+=(--no-swanlab)
[[ "$PROBE_MAX_BATCH" -eq 1 ]] && cmd+=(--probe-max-batch)

mkdir -p "$(dirname "$OUTPUT")"

log "LiteGauss train name=$RUN_NAME steps=$STEPS batch_size=$BATCH_SIZE lr=$LR warmup_steps=$WARMUP_STEPS output=$OUTPUT project=$SWANLAB_PROJECT"
"${cmd[@]}"
log "Done. checkpoint dir=$OUTPUT (config.json, norm_stats.json, model.pt)"

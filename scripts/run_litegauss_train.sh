#!/usr/bin/env bash
# LiteGauss LIBERO training (batch size 1). Does not touch Evo train.sh.
#
#   bash scripts/run_litegauss_train.sh --selfcheck
#   bash scripts/run_litegauss_train.sh
#   bash scripts/run_litegauss_train.sh --steps 100 --output checkpoints/litegauss/run1.pt
#   DATASET_CONFIG=/path/to/config.libero.yaml bash scripts/run_litegauss_train.sh
#
# Needs: setup_train_server.sh (.venv), setup_dataset.sh (config + data) for real runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LITEGAUSS_PY:-$ROOT/.venv/bin/python}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero.yaml}"
STEPS="${STEPS:-100}"
OUTPUT="${OUTPUT:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
SELFCHECK=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

log() { printf '\n==> %s\n' "$*"; }
# CompShare boxes often set a dead cluster proxy; strip it for HF/github pulls.
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --selfcheck) SELFCHECK=1; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --dataset_config_path) DATASET_CONFIG="$2"; shift 2 ;;
    --config) CONFIG_JSON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
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

cmd=(
  domestic
  HF_ENDPOINT="$HF_ENDPOINT"
  HF_HUB_DISABLE_XET=1
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  "$PY" Evo_depth/scripts/train_litegauss_libero.py
  --dataset_config_path "$DATASET_CONFIG"
  --steps "$STEPS"
)
[[ -n "$CONFIG_JSON" ]] && cmd+=(--config "$CONFIG_JSON")
if [[ -n "$OUTPUT" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  cmd+=(--output "$OUTPUT")
fi

log "LiteGauss train steps=$STEPS config=$DATASET_CONFIG HF_ENDPOINT=$HF_ENDPOINT"
"${cmd[@]}"
log "Done."

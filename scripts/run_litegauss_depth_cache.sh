#!/usr/bin/env bash
# Build/resume the mono DA3-base cache before LiteGauss training.
#
#   bash scripts/run_litegauss_depth_cache.sh
#   bash scripts/run_litegauss_depth_cache.sh --action-horizon 16
#   bash scripts/run_litegauss_depth_cache.sh --selfcheck
#
# Activates repo .venv (if present), sets DATASET_CONFIG / LITEGAUSS_DEPTH_CACHE /
# HF mirror defaults. Extra args go to cache_litegauss_depth.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PY="${LITEGAUSS_PY:-$ROOT/.venv/bin/python}"
export DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero.yaml}"
export LITEGAUSS_DEPTH_CACHE="${LITEGAUSS_DEPTH_CACHE:-$ROOT/data/litegauss_da3_depth_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/Evo_depth"

domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

log() { printf '\n==> %s\n' "$*"; }

if [[ "${1:-}" == "--selfcheck" && ! -x "$PY" ]]; then
  PY=python
fi
[[ -x "$PY" || "$PY" == python ]] || {
  echo "missing $PY. Run scripts/setup_train_server.sh or set LITEGAUSS_PY." >&2
  exit 1
}
[[ "${1:-}" == "--selfcheck" || -f "$DATASET_CONFIG" ]] || {
  echo "missing DATASET_CONFIG=$DATASET_CONFIG" >&2
  exit 1
}

mkdir -p "$LITEGAUSS_DEPTH_CACHE"
log "LiteGauss depth cache root=$LITEGAUSS_DEPTH_CACHE dataset=$DATASET_CONFIG py=$PY"
domestic "$PY" Evo_depth/scripts/cache_litegauss_depth.py \
  --dataset_config_path "$DATASET_CONFIG" \
  --depth-cache-root "$LITEGAUSS_DEPTH_CACHE" \
  "$@"
log "Done. Cache root=$LITEGAUSS_DEPTH_CACHE"

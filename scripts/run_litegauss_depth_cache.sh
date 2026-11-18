#!/usr/bin/env bash
# Build/resume the mono DA3-base cache before LiteGauss training.
#   bash scripts/run_litegauss_depth_cache.sh
#   bash scripts/run_litegauss_depth_cache.sh --selfcheck
# Same DATASET_CONFIG / LITEGAUSS_PY / LITEGAUSS_DEPTH_CACHE as training.
# Extra arguments go to cache_litegauss_depth.py (e.g. --action-horizon 16).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LITEGAUSS_PY:-$ROOT/.venv/bin/python}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero.yaml}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/Evo_depth"
if [[ "${1:-}" == "--selfcheck" && ! -x "$PY" ]]; then
  PY=python
fi
domestic "$PY" Evo_depth/scripts/cache_litegauss_depth.py \
  --dataset_config_path "$DATASET_CONFIG" "$@"

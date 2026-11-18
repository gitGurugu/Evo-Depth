#!/usr/bin/env bash
# DA3METRIC-Large vs simulator depth (probe_da3_metric.py).
# Does not load da3-base, FiLM, or Evo checkpoints. Does not start the server.
#
#   bash scripts/run_metric_probe.sh
#   bash scripts/run_metric_probe.sh libero_spatial camera 8
#   SKIP_DUMP=1 bash scripts/run_metric_probe.sh   # reuse geom_out/geom_probe.npz
#
# dump: LIBERO-PLUS-evaluation/.venv-libero-plus
# eval: repo-root .venv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-libero_spatial}"
CATEGORY="${2:-camera}"
N="${3:-8}"
PORT_PY="$ROOT/.venv/bin/python"
PLUS_PY="$ROOT/LIBERO-PLUS-evaluation/.venv-libero-plus/bin/python"
PLUS="$ROOT/LIBERO-PLUS-evaluation/LIBERO-plus"
JSON="$PLUS/libero/libero/benchmark/task_classification.json"
NPZ="${NPZ:-$ROOT/geom_out/geom_probe.npz}"
OUT="${OUT:-$ROOT/geom_out/geom_metric.json}"
PROBE="$ROOT/Evo_depth/scripts/probe_da3_metric.py"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero-plus}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

case "$SUITE" in
  libero_spatial|libero_goal|libero_object|libero_10) ;;
  *) echo "suite must be libero_spatial, libero_goal, libero_object, or libero_10" >&2; exit 1 ;;
esac
[[ -x "$PORT_PY" ]] || { echo "Run scripts/setup_train_server.sh first." >&2; exit 1; }
[[ -x "$PLUS_PY" ]] || { echo "Run scripts/setup_libero_plus.sh first." >&2; exit 1; }
[[ -f "$JSON" ]] || { echo "missing $JSON. Run scripts/setup_libero_plus.sh first." >&2; exit 1; }
[[ -f "$LIBERO_CONFIG_PATH/config.yaml" ]] || {
  echo "missing $LIBERO_CONFIG_PATH/config.yaml. Run scripts/setup_libero_plus.sh first." >&2
  exit 1
}
[[ -f "$PROBE" ]] || { echo "missing $PROBE" >&2; exit 1; }

if [[ "${SKIP_DUMP:-0}" != "1" ]]; then
  log "dump ${SUITE}/${CATEGORY} n=${N}  MUJOCO_GL=${MUJOCO_GL}"
  cd "$ROOT"
  domestic \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    PYTHONPATH="$PLUS${PYTHONPATH:+:$PYTHONPATH}" \
    MUJOCO_GL="$MUJOCO_GL" \
    "$PLUS_PY" "$PROBE" dump \
      --suite "$SUITE" \
      --category "$CATEGORY" \
      --n "$N" \
      --json "$JSON" \
      --out "$NPZ"
else
  log "SKIP_DUMP=1, reuse ${NPZ}"
  [[ -f "$NPZ" ]] || { echo "missing $NPZ" >&2; exit 1; }
fi

log "eval DA3METRIC-Large vs simulator depth"
cd "$ROOT"
domestic \
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
  "$PORT_PY" "$PROBE" eval \
    --npz "$NPZ" \
    --out "$OUT"

log "Done. ${OUT}, depth tensors (.npz/.pt/_depth_tensors/*.npy), PNGs in ${OUT%.json}_depth_vis/"

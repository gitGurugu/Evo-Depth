#!/usr/bin/env bash
# Dump LIBERO-PLUS camera frames, then A/B DA3 geometry (probe_da3_geom.py).
# Does not start the websocket server.
#
#   bash scripts/run_geom_probe.sh
#   bash scripts/run_geom_probe.sh libero_spatial camera 8
#   SKIP_DUMP=1 bash scripts/run_geom_probe.sh   # reuse geom_probe.npz
#
# dump: LIBERO-PLUS-evaluation/.venv-libero-plus
# eval: repo-root .venv
# checkpoint: checkpoints/libero/<suite>
# Need setup_train_server.sh, setup_libero_ckpt.sh, setup_libero_plus.sh first.
# MUJOCO_GL defaults to osmesa (same as run_libero.sh). Set MUJOCO_GL=egl if dump fails.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${1:-libero_spatial}"
CATEGORY="${2:-camera}"
N="${3:-8}"
PORT_PY="$ROOT/.venv/bin/python"
PLUS_PY="$ROOT/LIBERO-PLUS-evaluation/.venv-libero-plus/bin/python"
PLUS="$ROOT/LIBERO-PLUS-evaluation/LIBERO-plus"
JSON="$PLUS/libero/libero/benchmark/task_classification.json"
CKPT="${CKPT:-$ROOT/checkpoints/libero/$SUITE}"
NPZ="${NPZ:-$ROOT/geom_probe.npz}"
OUT="${OUT:-$ROOT/geom_ab.json}"
PROBE="$ROOT/Evo_depth/scripts/probe_da3_geom.py"
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
for rel in config.json norm_stats.json mp_rank_00_model_states.pt; do
  [[ -f "$CKPT/$rel" ]] || { echo "missing $CKPT/$rel. Run scripts/setup_libero_ckpt.sh first." >&2; exit 1; }
done
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

log "simulator cameras in ${NPZ}"
"$PLUS_PY" - "$NPZ" <<'PY'
import sys
import numpy as np
d = np.load(sys.argv[1])
dep = d["depths"]
print(f"tasks {d['task_ids'].tolist()}")
print(f"depth_m min {dep.min():.3f} median {np.median(dep):.3f} max {dep.max():.3f}")
for i, tid in enumerate(d["task_ids"]):
    K, E = d["intrinsics"][i], d["extrinsics"][i]
    rel = E[1] @ np.linalg.inv(E[0])
    c = np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)
    ang = float(np.degrees(np.arccos(c)))
    base = float(np.linalg.norm(rel[:3, 3]))
    print(
        f"task {int(tid):4d}  fx {K[0,0,0]:6.1f} / {K[1,0,0]:6.1f}  "
        f"rel_rot {ang:6.1f} deg  baseline {base:.3f} m  "
        f"depth_med {np.median(dep[i]):.3f}"
    )
PY

log "eval A/B with ${CKPT}"
cd "$ROOT"
domestic \
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
  "$PORT_PY" "$PROBE" eval \
    --npz "$NPZ" \
    --ckpt "$CKPT" \
    --out "$OUT"

log "Done. ${OUT}"

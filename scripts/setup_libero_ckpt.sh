#!/usr/bin/env bash
# Download the four Evo-Depth LIBERO checkpoints, plus InternVL3-1B and DA3-BASE.
# Each suite directory contains config.json, norm_stats.json, and mp_rank_00_model_states.pt.
# The two pretrained models go to the default Hugging Face cache, which the server reads.
#
#   bash scripts/setup_libero_ckpt.sh
#   export EVO_DEPTH_CKPT_DIR=/root/Evo-Depth/checkpoints/libero/libero_spatial

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/libero}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-$CKPT_DIR/.hf_home}"

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

[[ -x "$VENV/bin/python" ]] || { echo "Run scripts/setup_train_server.sh first." >&2; exit 1; }
mkdir -p "$CKPT_DIR"

log "Download MINT-SJTU/EVO-Depth-LIBERO"
domestic "$VENV/bin/python" - "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="MINT-SJTU/EVO-Depth-LIBERO",
    repo_type="model",
    local_dir=sys.argv[1],
)
PY

log "Download OpenGVLab/InternVL3-1B and depth-anything/da3-base"
domestic env HF_HOME="${HOME}/.cache/huggingface" \
  "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("OpenGVLab/InternVL3-1B", "depth-anything/da3-base"):
    path = snapshot_download(repo_id=repo)
    print(repo, path)
PY

log "Check the checkpoints"
suites=(libero_spatial libero_goal libero_object libero_10)
for suite in "${suites[@]}"; do
  root="$CKPT_DIR/$suite"
  for rel in config.json norm_stats.json mp_rank_00_model_states.pt; do
    [[ -f "$root/$rel" ]] || { echo "missing $root/$rel" >&2; exit 1; }
  done
  echo "ok $suite"
done
log "Done. Point the server at one suite, for example:"
log "export EVO_DEPTH_CKPT_DIR=${CKPT_DIR}/libero_spatial"

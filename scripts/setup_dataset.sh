#!/usr/bin/env bash
# Download the four LIBERO LeRobot v2.1 datasets and write dataset/config.libero.yaml.
# The training loader reads meta/*.jsonl, data/*/*.parquet, and videos/<chunk>/<camera>/*.mp4.
#
#   bash scripts/setup_dataset.sh
#   # train with --dataset_config_path dataset/config.libero.yaml

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
DATA_DIR="${DATA_DIR:-$ROOT/data/libero}"
CONFIG="$ROOT/Evo_depth/dataset/config.libero.yaml"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-$DATA_DIR/.hf_home}"

log() { printf '\n==> %s\n' "$*"; }
domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

[[ -x "$VENV/bin/python" ]] || { echo "Run scripts/setup_train_server.sh first." >&2; exit 1; }
mkdir -p "$DATA_DIR"

datasets=(
  libero_spatial_no_noops_1.0.0_lerobot
  libero_object_no_noops_1.0.0_lerobot
  libero_goal_no_noops_1.0.0_lerobot
  libero_10_no_noops_1.0.0_lerobot
)

for name in "${datasets[@]}"; do
  log "Download ${name}"
  domestic "$VENV/bin/python" - "$DATA_DIR" "$name" <<'PY'
import sys
from huggingface_hub import snapshot_download
data_dir, name = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=f"IPEC-COMMUNITY/{name}",
    repo_type="dataset",
    local_dir=f"{data_dir}/{name}",
)
PY
done

log "Write ${CONFIG}"
{
  echo "# Written by setup_dataset.sh. Do not change max_action_dim, max_state_dim, max_views."
  echo "max_action_dim: 24"
  echo "max_state_dim: 24"
  echo "max_views: 3"
  echo
  echo "data_groups:"
  echo "  libero_franka:"
  for name in "${datasets[@]}"; do
    cat <<EOF
    ${name}:
      path: ${DATA_DIR}/${name}
      view_map:
        image_1: observation.images.image
        image_2: observation.images.wrist_image
EOF
  done
} > "$CONFIG"

log "Check the datasets"
for name in "${datasets[@]}"; do
  root="$DATA_DIR/$name"
  for rel in meta/tasks.jsonl meta/episodes.jsonl meta/episodes_stats.jsonl; do
    [[ -f "$root/$rel" ]] || { echo "missing $root/$rel" >&2; exit 1; }
  done
  find "$root/data" -name '*.parquet' -print -quit | grep -q .
  find "$root/videos" -name '*.mp4' -print -quit | grep -q .
  echo "ok $name"
done
log "Done. Train with --dataset_config_path dataset/config.libero.yaml"

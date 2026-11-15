#!/usr/bin/env bash
# Build the Evo-Depth environment used to train, serve, and evaluate on LIBERO.
#
#   bash scripts/setup.sh
#
# Steps, in order:
#   1. scripts/setup_train_server.sh   repo-root .venv (Python 3.10, torch 2.5.1+cu124)
#   2. scripts/setup_dataset.sh        four LIBERO LeRobot datasets and config.libero.yaml
#   3. scripts/setup_libero_ckpt.sh    LIBERO checkpoints, InternVL3-1B, and DA3-BASE
#   4. scripts/setup_libero.sh         LIBERO-evaluation/.venv-libero
#   5. scripts/setup_libero_plus.sh    LIBERO-PLUS-evaluation/.venv-libero-plus
#
# Each step can still be run on its own. This script stops at the first failure.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"

log() { printf '\n======== %s ========\n' "$*"; }

run_step() {
  local name="$1"
  log "$name"
  bash "$SCRIPTS/$name"
}

run_step setup_train_server.sh
run_step setup_dataset.sh
run_step setup_libero_ckpt.sh
run_step setup_libero.sh
run_step setup_libero_plus.sh

log "Environment is ready"
cat <<EOF
Train and serve:  source ${ROOT}/.venv/bin/activate
LIBERO client:    source ${ROOT}/LIBERO-evaluation/.venv-libero/bin/activate
LIBERO-PLUS:      source ${ROOT}/LIBERO-PLUS-evaluation/.venv-libero-plus/bin/activate
Dataset config:   ${ROOT}/Evo_depth/dataset/config.libero.yaml
Checkpoints:      ${ROOT}/checkpoints/libero/<suite>
Run one suite:    bash scripts/run_libero.sh libero_spatial
EOF

#!/usr/bin/env bash
# LiteGauss 单场景过拟合（仅 libero_spatial）。顶层一条命令开跑，参数写死在此。
# 用法（仓库根目录）：
#   bash scripts/train_litegauss_overfit_spatial.sh
#
# 实现仍走 scripts/run_litegauss_train.sh；改超参只改本文件即可。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero_spatial_overfit.yaml}"

exec bash "$ROOT/scripts/run_litegauss_train.sh" \
  --swanlab-project Evo-LiteGauss \
  --name overfit-spatial \
  --steps 2000 \
  --batch-size 16 \
  --warmup-steps 50 \
  --lr 1e-4 \
  "$@"

#!/usr/bin/env bash
# LiteGauss 单场景过拟合（仅 libero_spatial）。顶层一条命令开跑。
# 用法（仓库根目录）：
#   bash scripts/train_litegauss_overfit_spatial.sh
#
# 数据：Evo_depth/dataset/config.libero_spatial_overfit.yaml
# 当前训练默认在线 DA3（按图 inference），无离线缓存开关。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/Evo_depth/dataset/config.libero_spatial_overfit.yaml}"

exec bash "$ROOT/scripts/run_litegauss_train.sh" \
  --swanlab-project Evo-LiteGauss \
  --name overfit-spatial \
  --steps 5000 \
  --batch-size 128 \
  --warmup-steps 100 \
  --lr 1e-4 \
  "$@"

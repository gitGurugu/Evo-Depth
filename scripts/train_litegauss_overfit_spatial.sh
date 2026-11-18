#!/usr/bin/env bash
# LiteGauss 单场景过拟合（仅 libero_spatial）。顶层一条命令开跑。
# 用法（仓库根目录）：
#   bash scripts/train_litegauss_overfit_spatial.sh
#
# 不用 DA3 离线缓存：--allow-online-da3（训练步内按图在线推理）。
# 数据：Evo_depth/dataset/config.libero_spatial_overfit.yaml

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
  --allow-online-da3 \
  "$@"

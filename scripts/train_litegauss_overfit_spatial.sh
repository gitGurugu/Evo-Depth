#!/usr/bin/env bash
# LiteGauss 单场景过拟合（仅 libero_spatial）。顶层一条命令开跑。
#
# 用法（仓库根目录）：
#   bash scripts/train_litegauss_overfit_spatial.sh
#   bash scripts/train_litegauss_overfit_spatial.sh --name overfit-b196-1k \
#     --steps 1000 --batch-size 196 --warmup-steps 50 --ckpt-interval 200
#   # 断点续训（--steps 仍是总步数）：
#   bash scripts/train_litegauss_overfit_spatial.sh --name overfit-b196-1k \
#     --steps 1000 --batch-size 196 --warmup-steps 50 --ckpt-interval 200 \
#     --resume checkpoints/litegauss/overfit-b196-1k/step_400
#
# 默认：--name overfit-spatial --steps 5000 --batch-size 128 --warmup-steps 100 --lr 1e-4
# 后面的参数会覆盖上面的默认（见 "$@"）。
#
# Checkpoint 目录（--ckpt-interval N，N>0；N=0 则只有跑完后的根目录三件套）：
#   checkpoints/litegauss/<name>/
#     step_200/        # 中途：每 N 步；--resume 指这里
#       config.json
#       norm_stats.json
#       model.pt       # gst + action_head + optimizer + step
#     step_400/
#       ...            # 同上三件套
#       config.json      # 全部 steps 跑完后写到根目录
#       norm_stats.json
#       model.pt         # 最终权重；评测用这个根目录
# --resume 指向某个 step_<K>/；--steps 仍是总步数（不是再加 N 步）。
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

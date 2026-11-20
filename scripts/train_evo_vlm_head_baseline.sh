#!/usr/bin/env bash
# Evo baseline: InternVL (frozen) + flow action head only.
# No --use_da3 => no FiLM / no DA3. Does not touch Evo_depth/train.sh.
#
# 用法（仓库根目录）：
#   bash scripts/train_evo_vlm_head_baseline.sh
#   bash scripts/train_evo_vlm_head_baseline.sh --name evo-vlm-head-5k --max_steps 5000
#
# 默认：spatial-only 数据、horizon=50、只训 action head、5k step。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVO="$ROOT/Evo_depth"
cd "$EVO"

DATASET_CONFIG="${DATASET_CONFIG:-$EVO/dataset/config.libero_spatial_overfit.yaml}"
SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints/evo-vlm-head-5k}"
RUN_NAME="${RUN_NAME:-evo-vlm-head-5k}"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-2500}"
MAIN_PORT="${MAIN_PROCESS_PORT:-29519}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

domestic() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY "$@"
}

[[ -f "$DATASET_CONFIG" ]] || {
  echo "missing dataset config: $DATASET_CONFIG" >&2
  exit 1
}
[[ -f "$EVO/ds_config.json" ]] || {
  echo "missing $EVO/ds_config.json" >&2
  exit 1
}

mkdir -p "$SAVE_DIR"

echo "==> Evo VLM+head baseline (no DA3/FiLM)"
echo "    dataset=$DATASET_CONFIG"
echo "    save_dir=$SAVE_DIR run_name=$RUN_NAME steps=$MAX_STEPS"

domestic accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port "$MAIN_PORT" \
  scripts/train.py \
  --run_name "$RUN_NAME" \
  --action_head flowmatching \
  --use_augmentation \
  --lr "$LR" \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size "$BATCH_SIZE" \
  --image_size 448 \
  --max_steps "$MAX_STEPS" \
  --log_interval 10 \
  --ckpt_interval "$CKPT_INTERVAL" \
  --warmup_steps "$WARMUP_STEPS" \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path "$DATASET_CONFIG" \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir "$SAVE_DIR" \
  "$@"

echo "==> Done. checkpoints under $SAVE_DIR"

# Two RTX A5000 (24GB each). Do not use train.sh's batch 16 / 1 GPU on these cards.
# Each GPU holds its own activations. ZeRO-2 only shards optimizer state, so stage 3
# (VLM unfrozen, 448, 3 views, no vision checkpointing) is sized at batch 1 per GPU.
# Optimizer states are offloaded to CPU. A5000 supports bf16; keep it.
# train.py calls zero_grad every step, so gradient_accumulation_steps stays 1.
# Effective batch: stage1 = 4*2 = 8, stage2 = 2*2 = 4, stage3 = 1*2 = 2.
#
# Dataset config on the server already points at the four LIBERO sets.
# Checkpoints: /home/kjl/checkpoint/train_libero/stage{1,2,3}/
# Launch from Evo_depth/:  bash train_a5000.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
# This driver rejects expandable_segments. Leaving it on SIGSEGVs in the CUDA caching allocator.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export SWANLAB_MODE=cloud
export WANDB_MODE=disabled
# swanlab 0.10 needs protobuf 6; wandb 0.13.1 still imports the old generated files.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

# ========== Stage 1: action head + FiLM. VLM and DA3 frozen, graph is not kept. ==========
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --gpu_ids 0,1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config_a5000.json \
  --main_process_port 29521 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage1_a5000 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 4 \
  --num_workers 0 \
  --image_size 448 \
  --max_steps 5000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /home/kjl/checkpoint/train_libero/stage1/

# ========== Stage 2: DA3 unfrozen. Checkpointed, but one graph per sample is kept. ==========
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --gpu_ids 0,1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config_a5000.json \
  --main_process_port 29521 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage2_a5000 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 2 \
  --num_workers 0 \
  --image_size 448 \
  --max_steps 10000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /home/kjl/checkpoint/train_libero/stage2/ \
  --resume \
  --resume_pretrain \
  --resume_path /home/kjl/checkpoint/train_libero/stage1/step_5000

# ========== Stage 3: full VLM + DA3 + action head. One sample's graph fills a 24GB card. ==========
accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --gpu_ids 0,1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config_a5000.json \
  --main_process_port 29521 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage3_a5000 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 1 \
  --num_workers 0 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --use_da3 \
  --finetune_vlm \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /home/kjl/checkpoint/train_libero/stage3/ \
  --resume \
  --resume_pretrain \
  --resume_path /home/kjl/checkpoint/train_libero/stage2/step_10000

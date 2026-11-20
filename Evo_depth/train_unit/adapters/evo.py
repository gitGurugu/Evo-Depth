import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..protocol import ModelAdapter


class Adapter(ModelAdapter):
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            raise ValueError("Evo requires --adapter-config PATH")
        with open(config_path, encoding="utf-8") as stream:
            self.config = json.load(stream)
        if not isinstance(self.config, dict):
            raise ValueError("Evo adapter configuration must be a JSON object")
        self.config.pop("train", None)
        dataset_path = self.config.get("dataset_config_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError("dataset_config_path is required in Evo adapter configuration")

    def build_model(self, device: str) -> nn.Module:
        # Delay heavy imports so importing the adapter never loads InternVL weights.
        from Evo_depth.scripts.Evo_depth import Evo_depth

        self.config["device"] = device
        model = Evo_depth(self.config).to(device)
        if model.horizon == 1:
            # Evo creates this lazily; register it before freezing and optimizer construction.
            model.action_head.single_action_proj = nn.Linear(
                model.per_action_dim, model.action_head.embed_dim
            ).to(device)
        model.set_finetune_flags()
        return model

    def set_train_mode(self, model: nn.Module) -> None:
        model.train()
        for module in model.modules():
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()

    def inspect_modules(self, model: nn.Module) -> dict[str, nn.Module]:
        # Same groupings as Evo_depth/scripts/train.py after set_finetune_flags.
        named = {
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head,
        }
        if getattr(model.embedder, "use_da3", False):
            named["da3_embedder"] = model.embedder.dpt_embedder
            named["film_fuser"] = model.embedder.dpt_linear_film
        return named

    def build_loader(self, batch_size: int, num_workers: int) -> DataLoader:
        import yaml
        from Evo_depth.dataset.lerobot_dataset_pretrain_mp import (
            LeRobotDataset, custom_collate_fn, worker_init_fn,
        )

        config = self.config
        with open(config["dataset_config_path"], encoding="utf-8") as stream:
            dataset_config = yaml.safe_load(stream)
        dataset = LeRobotDataset(
            config=dataset_config,
            image_size=config.get("image_size", 448),
            max_samples_per_file=config.get("max_samples_per_file"),
            action_horizon=config.get("action_horizon", config.get("horizon", 16)),
            binarize_gripper=config.get("binarize_gripper", False),
            cache_dir=config.get("cache_dir"),
            use_augmentation=config.get("use_augmentation", False),
        )
        self.norm_stats = dataset.arm2stats_dict
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
            pin_memory=True, drop_last=True, collate_fn=custom_collate_fn,
            persistent_workers=num_workers > 0,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

    def forward_loss(self, model: nn.Module, batch) -> torch.Tensor:
        policy = getattr(model, "module", model)
        device = next(model.parameters()).device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        states = batch["states"].to(device=device, dtype=dtype)
        actions_gt = batch["actions"].to(device=device, dtype=dtype)
        action_mask = batch["action_mask"].to(device)
        if action_mask.sum() == 0:
            raise ValueError("action_mask.sum() is 0: all actions are masked")

        fused_tokens_list = []
        for prompt, images, image_mask in zip(batch["prompts"], batch["images"], batch["image_masks"]):
            fused = policy.get_vl_embeddings(
                images=images.to(device), image_mask=image_mask.to(device),
                prompt=prompt, return_cls_only=False,
            )
            fused_tokens_list.append(fused.to(device=device, dtype=dtype))
        fused_tokens = torch.cat(fused_tokens_list, dim=0)

        if policy.horizon == 1:
            actions_gt = actions_gt.squeeze(1)
            action_mask = action_mask.squeeze(1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred_velocity, noise = model(
                fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask,
            )

        target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
        if pred_velocity.shape != target_velocity.shape:
            raise ValueError(f"Velocity shape mismatch: {pred_velocity.shape} != {target_velocity.shape}")
        action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
        # Match train.py: only mask predictions; LeRobot padding and Evo noise mask zero the target.
        loss = nn.functional.mse_loss(pred_velocity * action_mask, target_velocity)
        return loss * (action_mask.numel() / (action_mask.sum() + 1e-8))

    def export_state(self, model: nn.Module) -> dict:
        return model.state_dict()

    def load_state(self, model: nn.Module, state: dict) -> None:
        model.load_state_dict(state)

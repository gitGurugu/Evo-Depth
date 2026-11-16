"""Frozen InternVL/DA3 + trainable GST/flow policy, one multi-view sample per call.

RGBs are PIL images; each K is in its original image's pixels. State/actions
must already use the dataset's training normalization. No FiLM is constructed.
"""

from types import SimpleNamespace

import torch
from torch import nn

from ..action_head.flow_matching import FlowmatchingActionHead
from .gst import GaussianSpatialTokenizer
from .litegauss import LiteGauss


class LiteGaussPolicy(nn.Module):
    def __init__(self, config=None, *, embedder=None, depth=None):
        super().__init__()
        self.config = dict(config or {})
        cfg = self.config
        device = cfg.get("device", "cuda")
        if embedder is None:
            # Keep heavyweight imports/downloads out of the CPU smoke path.
            from model.internvl3.internvl3_embedder_da3 import InternVL3Embedder

            embedder = InternVL3Embedder(
                model_name=cfg.get("vlm_name", "OpenGVLab/InternVL3-1B"),
                image_size=cfg.get("image_size", 448), device=device, use_da3=False,
            )
        if depth is None:
            from model.depth_anything_3.depth_anything_v3 import DAV3Module

            depth = DAV3Module().to(device)

        llm_dim = embedder.model.language_model.config.hidden_size
        embed_dim = cfg.get("embed_dim", llm_dim)
        if embed_dim != llm_dim:
            raise ValueError(f"embed_dim={embed_dim} must match LLM hidden_size={llm_dim}")
        self.horizon = cfg.get("action_horizon", cfg.get("horizon", 16))
        self.per_action_dim = cfg.get("per_action_dim", 7)
        self.state_dim = cfg.get("state_dim", 7)
        if self.horizon < 1 or self.per_action_dim < 1:
            raise ValueError("horizon and per_action_dim must be positive")
        if cfg.get("num_inference_timesteps", 20) < 1:
            raise ValueError("num_inference_timesteps must be positive")
        gst = GaussianSpatialTokenizer(
            feat_dim=llm_dim, out_dim=embed_dim,
            num_queries=cfg.get("num_queries", 64),
            fourier_L=cfg.get("fourier_L", 10),
            workspace_rho=cfg.get("workspace_rho", 0.8),
            pool_dim=cfg.get("pool_dim", 256),
            num_heads=cfg.get("gst_num_heads", 4),
            learn_affine=cfg.get("learn_affine", False),
        ).to(device)
        head = FlowmatchingActionHead(config=SimpleNamespace(
            embed_dim=embed_dim, hidden_dim=cfg.get("hidden_dim", 1024),
            horizon=self.horizon, per_action_dim=self.per_action_dim,
            action_dim=self.horizon * self.per_action_dim,
            state_dim=self.state_dim, state_hidden_dim=cfg.get("state_hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 8), num_layers=cfg.get("num_layers", 8),
            dropout=cfg.get("dropout", 0.0), num_categories=cfg.get("num_categories", 1),
            num_inference_timesteps=cfg.get("num_inference_timesteps", 20),
        )).to(device)
        # The existing head otherwise creates this after the optimizer is built.
        if self.horizon == 1:
            head.single_action_proj = nn.Linear(self.per_action_dim, embed_dim).to(device)
        self.litegauss = LiteGauss(
            embedder.to(device), depth.to(device), gst, action_head=head,
            max_text_len=cfg.get("max_text_len", 128),
        )
        self.freeze_backbones()

    def freeze_backbones(self):
        self.litegauss.embedder.requires_grad_(False).eval()
        self.litegauss.depth.requires_grad_(False).eval()
        self.litegauss.gst.requires_grad_(True)
        self.litegauss.action_head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.litegauss.embedder.eval()
        self.litegauss.depth.eval()
        return self

    def _tensor(self, value):
        ref = self.litegauss.gst.queries
        return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)

    def _state(self, state):
        if state is None:
            return None
        state = self._tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (1, self.state_dim) or not torch.isfinite(state).all():
            raise ValueError(f"state must be finite with shape (1,{self.state_dim})")
        return state

    def forward(self, rgb, K, prompt="", state=None, actions_gt=None,
                action_mask=None, embodiment_id=None):
        """Return scalar masked flow loss, or predicted (1,horizon,action_dim)."""
        if actions_gt is None:
            return self.predict(rgb, K, prompt, state, action_mask, embodiment_id)
        actions = self._tensor(actions_gt)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        shape = (1, self.horizon, self.per_action_dim)
        if actions.shape != shape or not torch.isfinite(actions).all():
            raise ValueError(f"actions_gt must be finite with shape {shape}")
        mask = torch.ones_like(actions) if action_mask is None else self._tensor(action_mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if (mask.shape != shape or not torch.isfinite(mask).all()
                or not ((mask >= 0) & (mask <= 1)).all() or mask.sum() == 0):
            raise ValueError(f"action_mask must have shape {shape}, values in [0,1], and nonzero sum")
        ids = None if embodiment_id is None else torch.as_tensor(
            embodiment_id, device=actions.device, dtype=torch.long,
        ).reshape(1)
        # Zero padded targets before interpolation as well as masking their loss.
        actions = actions * mask
        head_actions = actions if self.horizon > 1 else actions[:, 0]
        head_mask = mask if self.horizon > 1 else mask[:, 0]
        velocity, noise = self.litegauss(
            rgb, K, prompt, state=self._state(state), actions_gt=head_actions,
            action_mask=head_mask, embodiment_id=ids,
        )
        target = actions.flatten(1) - noise.flatten(1)
        return ((velocity - target).square() * mask.flatten(1)).sum() / mask.sum()

    def train_step(self, rgb, K, prompt, state, actions_gt, **kwargs):
        """Compute loss; caller owns zero_grad(), backward(), and optimizer.step()."""
        return self(rgb, K, prompt, state, actions_gt, **kwargs)

    @torch.no_grad()
    def predict(self, rgb, K, prompt="", state=None, action_mask=None, embodiment_id=None):
        """Inference mask is (1,per_action_dim), shared across the action horizon."""
        mask = self._tensor(torch.ones(1, self.per_action_dim) if action_mask is None else action_mask)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if (mask.shape != (1, self.per_action_dim) or not torch.isfinite(mask).all()
                or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError("inference action_mask must be binary with shape (1,per_action_dim)")
        was_training = self.training
        self.eval()
        try:
            if not isinstance(rgb, (list, tuple)):
                rgb, K = [rgb], [K]
            if not rgb or not isinstance(K, (list, tuple)) or len(rgb) != len(K):
                raise ValueError("RGB and K sequences must have the same nonzero length")
            core = self.litegauss
            tokens = torch.cat([core.encode_gst(im, k)["tokens"] for im, k in zip(rgb, K)], dim=1)
            context = core.build_context(tokens, core.encode_text(prompt))
            ids = None if embodiment_id is None else torch.as_tensor(
                embodiment_id, device=mask.device, dtype=torch.long,
            ).reshape(1)
            # LiteGauss.forward does not pass the inference mask to get_action.
            actions = core.action_head.get_action(
                context, state=self._state(state), action_mask=mask, embodiment_id=ids,
            )
            return actions.reshape(1, self.horizon, self.per_action_dim) * mask.unsqueeze(1)
        finally:
            self.train(was_training)

    run_inference = predict

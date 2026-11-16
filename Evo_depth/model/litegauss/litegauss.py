"""InternVL ViT + DA3 depth -> GST tokens; InternVL LM text-only; cat -> optional flow action."""

from __future__ import annotations

import math

import torch
from torch import nn

from .gst import align_depth_to_patches, scale_intrinsics


class LiteGauss(nn.Module):
    """Vision: InternVL ViT patches + DA3 depth -> GST.
    Language: InternVL LM on text only (no image tokens / no FiLM).
    Inputs: one PIL RGB and K, or nonempty lists/tuples of RGBs and Ks of equal length.
    Each view uses the shared embedder, depth module and GST independently.
    Context: cat([gst_cam0, gst_cam1, ..., lang_tokens]) along the token axis.
    Flow action is provided by an injected action_head (FlowmatchingActionHead),
    not implemented inside LiteGauss. Without it, returns concatenated GST outputs.
    """

    def __init__(self, embedder, depth, gst, action_head=None, max_text_len: int = 128):
        super().__init__()
        self.embedder = embedder
        self.depth = depth
        self.gst = gst
        self.action_head = action_head
        self.max_text_len = max_text_len

    def encode_gst(self, rgb, K) -> dict:
        rgb = rgb.convert("RGB")
        ref = self.gst.queries
        K = torch.as_tensor(K, device=ref.device, dtype=ref.dtype)
        if K.shape != (3, 3) or not torch.isfinite(K).all():
            raise ValueError("K must be a finite (3,3) matrix in original RGB pixels")
        pixels, tiles = self.embedder._preprocess_images([rgb])
        if tiles != [1]:
            raise ValueError("LiteGauss requires one InternVL tile per image")
        feat = self.embedder.model.extract_feature(pixels).to(device=ref.device, dtype=ref.dtype)
        grid = math.isqrt(feat.shape[1])
        if feat.shape[0] != 1 or grid * grid != feat.shape[1]:
            raise ValueError("Expected a single square grid of InternVL patch tokens")

        prediction = self.depth.inference_images([rgb])
        depth = torch.as_tensor(prediction.depth, device=feat.device, dtype=feat.dtype)
        if depth.ndim != 3 or depth.shape[0] != 1 or not torch.isfinite(depth).all():
            raise ValueError("Expected one finite DA3 depth map (1,H,W)")
        K_depth = scale_intrinsics(K, (rgb.height, rgb.width), depth.shape[-2:])
        size = self.embedder.image_size
        aligned = align_depth_to_patches(depth, K_depth, (size, size), (grid, grid))
        return self.gst(feat, aligned["depth"], aligned["K"], aligned["uv"])

    def encode_text(self, prompt: str) -> torch.Tensor:
        """InternVL language_model only; returns (1, L, C)."""
        device = self.gst.queries.device
        tok = self.embedder.tokenizer(
            prompt or "",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        out = self.embedder.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1].to(dtype=self.gst.queries.dtype)
        # drop pads so action context stays short
        keep = attention_mask[0].bool()
        return hidden[:, keep, :]

    def build_context(self, gst_tokens: torch.Tensor, lang_tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([gst_tokens, lang_tokens.to(gst_tokens)], dim=1)

    def forward(
        self,
        rgb,
        K,
        prompt: str = "",
        state: torch.Tensor | None = None,
        actions_gt: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
    ):
        if isinstance(rgb, (list, tuple)):
            if not rgb or not isinstance(K, (list, tuple)) or len(rgb) != len(K):
                raise ValueError("RGB and K sequences must have the same nonzero length")
            views = [self.encode_gst(image, intrinsics) for image, intrinsics in zip(rgb, K)]
            gst_out = {key: torch.cat([view[key] for view in views], dim=1) for key in views[0]}
        else:
            gst_out = self.encode_gst(rgb, K)
        lang = self.encode_text(prompt)
        context = self.build_context(gst_out["tokens"], lang)
        if self.action_head is None:
            return {**gst_out, "lang": lang, "context": context}
        return self.action_head(
            context,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
        )

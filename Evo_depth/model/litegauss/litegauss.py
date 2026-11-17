"""InternVL ViT + DA3 depth -> GST tokens; InternVL LM text-only; cat -> optional flow action."""

from __future__ import annotations

import math

import torch
from torch import nn

from .gst import align_depth_to_patches, scale_intrinsics


class LiteGauss(nn.Module):
    """Vision: InternVL ViT patches + DA3 depth -> GST.
    Language: InternVL LM on text only (no image tokens / no FiLM).
    Inputs: one PIL RGB and K, a list of V views, or RGB [B][V] with K [B,V,3,3].
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
        return self.encode_gst_images([rgb], [K])

    def encode_gst_images(self, images, intrinsics) -> dict:
        """One ViT/GST call over images; DA3 always sees exactly one image."""
        images = [image.convert("RGB") for image in images]
        ref = self.gst.queries
        matrices = [torch.as_tensor(k, device=ref.device, dtype=ref.dtype) for k in intrinsics]
        if len(matrices) != len(images) or any(
                k.shape != (3, 3) or not torch.isfinite(k).all() for k in matrices):
            raise ValueError("Each K must be a finite (3,3) matrix in original RGB pixels")
        pixels, tiles = self.embedder._preprocess_images(images)
        if tiles != [1] * len(images):
            raise ValueError("LiteGauss requires one InternVL tile per image")
        feat = self.embedder.model.extract_feature(pixels).to(device=ref.device, dtype=ref.dtype)
        grid = math.isqrt(feat.shape[1])
        if feat.shape[0] != len(images) or grid * grid != feat.shape[1]:
            raise ValueError("Expected one square InternVL patch grid per image")

        aligned_images = []
        for rgb, K in zip(images, matrices):
            # DA3's image list is a scene/view axis, never an independent-sample batch.
            prediction = self.depth.inference_images([rgb])
            depth = torch.as_tensor(prediction.depth, device=feat.device, dtype=feat.dtype)
            if depth.ndim != 3 or depth.shape[0] != 1 or not torch.isfinite(depth).all():
                raise ValueError("Expected one finite DA3 depth map (1,H,W)")
            K_depth = scale_intrinsics(K, (rgb.height, rgb.width), depth.shape[-2:])
            size = self.embedder.image_size
            aligned_images.append(align_depth_to_patches(depth, K_depth, (size, size), (grid, grid)))
        aligned = {key: torch.cat([image[key] for image in aligned_images]) for key in aligned_images[0]}
        return self.gst(feat, aligned["depth"], aligned["K"], aligned["uv"])

    def encode_text(self, prompt, return_mask=False):
        """InternVL language_model once for str or list[str]; returns (B,L,C)."""
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
            **({"position_ids": (attention_mask.cumsum(-1) - 1).clamp_min(0)}
               if isinstance(prompt, (list, tuple)) else {}),
        )
        hidden = out.hidden_states[-1].to(dtype=self.gst.queries.dtype)
        # drop pads so action context stays short
        keep = attention_mask.bool().any(dim=0)
        hidden, mask = hidden[:, keep, :], attention_mask[:, keep].bool()
        return (hidden, mask) if return_mask else hidden

    def build_context(self, gst_tokens: torch.Tensor, lang_tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([gst_tokens, lang_tokens.to(gst_tokens)], dim=1)

    @staticmethod
    def batch_size(rgb):
        return len(rgb) if isinstance(rgb, (list, tuple)) and rgb and isinstance(rgb[0], (list, tuple)) else 1

    def encode_context(self, rgb, K, prompt=""):
        nested = isinstance(rgb, (list, tuple)) and rgb and isinstance(rgb[0], (list, tuple))
        if nested:
            B, V = len(rgb), len(rgb[0])
            if (not V or len(K) != B or any(len(row) != V for row in rgb)
                    or any(len(row) != V for row in K)
                    or not isinstance(prompt, (list, tuple)) or len(prompt) != B
                    or any(not isinstance(text, str) for text in prompt)):
                raise ValueError("Expected RGB [B][V], K [B,V,3,3], and B text prompts")
            if B > 1:
                out = self.encode_gst_images([im for row in rgb for im in row],
                                             [k for row in K for k in row])
                gst_out = {key: value.reshape(B, V * value.shape[1], *value.shape[2:])
                           for key, value in out.items()}
                lang, lang_mask = self.encode_text(list(prompt), return_mask=True)
                tokens = gst_out["tokens"]
                context_mask = torch.cat([torch.ones(tokens.shape[:2], dtype=torch.bool,
                                                      device=tokens.device), lang_mask], dim=1)
                return gst_out, lang, self.build_context(tokens, lang), context_mask
            # Preserve the production B=1 path, including its per-camera forwards.
            rgb, K, prompt = rgb[0], list(K[0]), prompt[0]
        if isinstance(rgb, (list, tuple)):
            if not rgb or not isinstance(K, (list, tuple)) or len(rgb) != len(K):
                raise ValueError("RGB and K sequences must have the same nonzero length")
            views = [self.encode_gst(image, k) for image, k in zip(rgb, K)]
            gst_out = {key: torch.cat([view[key] for view in views], dim=1) for key in views[0]}
        else:
            gst_out = self.encode_gst(rgb, K)
        lang = self.encode_text(prompt)
        return gst_out, lang, self.build_context(gst_out["tokens"], lang), None

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
        gst_out, lang, context, context_mask = self.encode_context(rgb, K, prompt)
        if self.action_head is None:
            return {**gst_out, "lang": lang, "context": context,
                    **({"context_mask": context_mask} if context_mask is not None else {})}
        return self.action_head(
            context,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
            **({"context_mask": context_mask} if context_mask is not None else {}),
        )

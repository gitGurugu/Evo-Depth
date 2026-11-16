"""LiteGauss GST: relative depth + semantic patches -> pooled Gaussian tokens.

Paper recipe follows GaussVLA GST (arXiv:2608.24959 §3.3). That code is not
released; back-project / Fourier PE pattern is the same idea as open SpatialVLA
(https://github.com/SpatialVLA/SpatialVLA). Depth backbone stays outside —
pass DA3-base relative depth (or sim depth) in; do not fork depth_anything_3.

Affine scale (λ, β) is optional; default identity for the first probe.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSpatialTokenizer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_queries: int = 64,
        fourier_L: int = 10,
        workspace_rho: float = 0.8,
        pool_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        learn_affine: bool = False,
    ):
        super().__init__()
        self.fourier_L = fourier_L
        self.workspace_rho = workspace_rho
        self.fourier_dim = 6 * fourier_L
        self.gauss_dim = feat_dim + self.fourier_dim + 3

        self.lambda_d = nn.Parameter(torch.ones(())) if learn_affine else None
        self.beta_d = nn.Parameter(torch.zeros(())) if learn_affine else None

        self.param_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3 + 3 + 1),
        )
        self.num_heads = num_heads
        self.W_in = nn.Linear(self.gauss_dim, pool_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, pool_dim) * 0.02)
        self.pool_ln_q = nn.LayerNorm(pool_dim)
        self.pool_attn = nn.MultiheadAttention(pool_dim, num_heads, batch_first=True)
        self.pool_ln = nn.LayerNorm(pool_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, pool_dim * 4),
            nn.GELU(),
            nn.Linear(pool_dim * 4, pool_dim),
        )
        self.out = nn.Linear(pool_dim, out_dim) if out_dim != pool_dim else nn.Identity()

    def correct_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if self.lambda_d is None:
            return depth
        return self.lambda_d * depth + self.beta_d

    @staticmethod
    def backproject(depth: torch.Tensor, K: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """depth (B,P), K (B,3,3), uv (B,P,2) pixel coords -> cam xyz (B,P,3)."""
        B, P = depth.shape
        ones = torch.ones(B, P, 1, device=depth.device, dtype=depth.dtype)
        pix = torch.cat([uv, ones], dim=-1)
        rays = torch.linalg.solve(K, pix.transpose(1, 2)).transpose(1, 2)
        return rays * depth.unsqueeze(-1)

    def fourier(self, mu: torch.Tensor) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.fourier_L, device=mu.device, dtype=mu.dtype)) * math.pi
        ang = mu.unsqueeze(-1) * freqs
        return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)

    def forward(
        self,
        feat: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        uv: torch.Tensor,
    ) -> dict:
        """
        feat  (B, P, C) semantic patch features
        depth (B, P)    relative / metric-ish depth at patch centers
        K     (B, 3, 3) intrinsics for the feat resolution
        uv    (B, P, 2) patch-center pixel coords
        """
        d = self.correct_depth(depth)
        c = self.backproject(d, K, uv)
        raw = self.param_head(feat)
        dmu, eta, alpha_logit = raw.split([3, 3, 1], dim=-1)
        alpha = torch.sigmoid(alpha_logit).squeeze(-1)
        mu = c + self.workspace_rho * torch.tanh(dmu)
        gamma = self.fourier(mu)
        g = torch.cat([feat, gamma, eta], dim=-1)

        keys = self.W_in(g)
        B, Nq = feat.size(0), self.queries.size(0)
        q = self.pool_ln_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        # GaussVLA Eq.10: add log α to attn logits (per key). MHA wants (B*H, Nq, P).
        bias = torch.log(alpha.clamp_min(1e-4)).view(B, 1, 1, -1)
        bias = bias.expand(-1, self.num_heads, Nq, -1).reshape(B * self.num_heads, Nq, -1)
        pooled, _ = self.pool_attn(q, keys, keys, attn_mask=bias)
        h = self.pool_ln(q + pooled)
        tokens = self.out(h + self.ffn(h))
        return {"tokens": tokens, "mu": mu, "eta": eta, "alpha": alpha, "c": c}


def patch_centers(h: int, w: int, gh: int, gw: int, device=None, dtype=None) -> torch.Tensor:
    """Centers of a gh×gw grid over an h×w image -> (P, 2) = (x, y)."""
    ys = (torch.arange(gh, device=device, dtype=dtype) + 0.5) * (h / gh)
    xs = (torch.arange(gw, device=device, dtype=dtype) + 0.5) * (w / gw)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def scale_intrinsics(K: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """K for src_hw -> K for dst_hw. Accepts (3,3) or (B,3,3)."""
    sh, sw = src_hw
    dh, dw = dst_hw
    out = K.clone()
    out[..., 0, :] = out[..., 0, :] * (dw / sw)
    out[..., 1, :] = out[..., 1, :] * (dh / sh)
    return out


def align_depth_to_patches(
    depth: torch.Tensor,
    K: torch.Tensor,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Align DA3 depth (native res) to InternVL image plane + patch grid.

    Keeps DA3/InternVL pretrained input sizes outside; only resamples outputs.
    depth (B,Hd,Wd) or (B,1,Hd,Wd): relative depth at DA3 resolution
    K     (B,3,3) or (3,3): intrinsics for that depth map
    image_hw (H,W): InternVL image plane (e.g. 448,448)
    grid_hw  (gh,gw): feat patch grid (P=gh*gw)

    Returns depth (B,P), K (B,3,3) on image_hw, uv (B,P,2) in image pixels.
    """
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    if depth.ndim != 3:
        raise ValueError(f"depth must be (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(depth.size(0), -1, -1)
    Hd, Wd = depth.shape[-2:]
    H, W = image_hw
    gh, gw = grid_hw
    # bilinear to InternVL plane, then one value per patch (avg over patch = area downsample)
    depth_img = F.interpolate(depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
    depth_grid = F.interpolate(depth_img, size=(gh, gw), mode="area").flatten(2)  # (B,1,P)
    depth_p = depth_grid.squeeze(1)
    K_img = scale_intrinsics(K, (Hd, Wd), (H, W))
    uv = patch_centers(H, W, gh, gw, device=depth.device, dtype=depth.dtype)
    uv = uv.unsqueeze(0).expand(depth.size(0), -1, -1)
    return {"depth": depth_p, "K": K_img, "uv": uv}


def _selfcheck():
    B, gh, gw, C = 2, 8, 8, 32
    P = gh * gw
    m = GaussianSpatialTokenizer(feat_dim=C, num_queries=16, fourier_L=4, pool_dim=64, out_dim=64)
    feat = torch.randn(B, P, C)
    # DA3-sized depth + K -> InternVL 128 plane, 8x8 patches
    depth_map = torch.rand(B, 64, 80) * 0.5 + 0.2
    K_da3 = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    K_da3[:, 0, 0] = K_da3[:, 1, 1] = 100.0
    K_da3[:, 0, 2], K_da3[:, 1, 2] = 40.0, 32.0
    aligned = align_depth_to_patches(depth_map, K_da3, image_hw=(128, 128), grid_hw=(gh, gw))
    assert aligned["depth"].shape == (B, P)
    assert aligned["uv"].shape == (B, P, 2)
    assert torch.allclose(aligned["K"][:, 0, 0], torch.tensor(100.0 * 128 / 80))
    out = m(feat, aligned["depth"], aligned["K"], aligned["uv"])
    assert out["tokens"].shape == (B, 16, 64)
    assert out["mu"].shape == (B, P, 3)
    # identity affine: z ≈ depth when principal ray
    c0 = GaussianSpatialTokenizer.backproject(
        torch.ones(1, 1) * 2.0,
        torch.tensor([[[200.0, 0, 64.0], [0, 200.0, 64.0], [0, 0, 1.0]]]),
        torch.tensor([[[64.0, 64.0]]]),
    )
    assert torch.allclose(c0[0, 0], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)
    print("litegauss gst selfcheck ok", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    _selfcheck()

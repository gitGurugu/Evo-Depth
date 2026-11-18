"""Run from repo root: python Evo_depth/scripts/probe_litegauss.py

Default: CPU single/multi-view wiring and fake action selfcheck, no checkpoints.
Uses geom_probe.npz first view when present, otherwise dummy RGB. Real CUDA forward:
  python Evo_depth/scripts/probe_litegauss.py --real
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.litegauss import GaussianSpatialTokenizer, LiteGauss, patch_centers, scale_intrinsics


class _FakeLM(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.emb = nn.Embedding(32, dim)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, return_dict=True):
        h = self.emb(input_ids)
        return SimpleNamespace(hidden_states=(h, h))


def selfcheck(rgb, K):
    rgb2 = Image.new("RGB", (rgb.width + 32, rgb.height + 16), (192, 64, 128))
    K2 = np.array(K, copy=True)
    K2[0, 0] *= 1.5
    depth_calls = []

    def preprocess(images):
        assert len(images) == 1
        assert images[0].size in (rgb.size, rgb2.size)
        return torch.full((1, 3, 448, 448), float(images[0].width)), [1]

    def extract(pixels):
        assert pixels.shape == (1, 3, 448, 448)
        return torch.ones(1, 256, 32, dtype=torch.bfloat16) * pixels[0, 0, 0, 0]

    def infer(images):
        assert len(images) == 1
        depth_calls.append(images[0].size)
        depth = 2.0 if images[0].size == rgb.size else 3.0
        return SimpleNamespace(depth=np.full((1, 126, 252), depth, np.float32))

    def tokenize(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128):
        ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    lm = _FakeLM(64)
    embedder = SimpleNamespace(
        image_size=448,
        _preprocess_images=preprocess,
        tokenizer=tokenize,
        model=SimpleNamespace(extract_feature=extract, language_model=lm),
    )
    gst = GaussianSpatialTokenizer(32, num_queries=16, pool_dim=64, out_dim=64)
    model = LiteGauss(embedder, SimpleNamespace(inference_images=infer), gst).eval()
    out = model(rgb, K, prompt="pick up the cup")
    assert out["tokens"].shape == (1, 16, 64)
    assert out["lang"].shape == (1, 3, 64)  # pad dropped
    assert out["context"].shape == (1, 19, 64)
    assert out["mu"].shape == (1, 256, 3)
    assert all(torch.isfinite(out[k]).all() for k in ("tokens", "lang", "context", "mu", "c", "eta", "alpha"))
    K_img = scale_intrinsics(torch.as_tensor(K), (rgb.height, rgb.width), (448, 448))
    uv = patch_centers(448, 448, 16, 16)
    rays = torch.linalg.solve(K_img, torch.cat([uv, torch.ones(256, 1)], -1).T).T
    torch.testing.assert_close(out["c"][0], rays * 2)

    second = model(rgb2, K2, prompt="pick up the cup")
    depth_calls.clear()
    multi = model([rgb, rgb2], [K, K2], prompt="pick up the cup")
    assert depth_calls == [rgb.size, rgb2.size]
    assert multi["tokens"].shape == (1, 32, 64)
    assert multi["context"].shape == (1, 35, 64)
    for key in ("tokens", "mu", "c", "eta", "alpha"):
        torch.testing.assert_close(multi[key], torch.cat([out[key], second[key]], dim=1))
    torch.testing.assert_close(multi["context"], torch.cat([out["tokens"], second["tokens"], out["lang"]], dim=1))
    K2_img = scale_intrinsics(torch.as_tensor(K2), (rgb2.height, rgb2.width), (448, 448))
    rays2 = torch.linalg.solve(K2_img, torch.cat([uv, torch.ones(256, 1)], -1).T).T
    torch.testing.assert_close(multi["c"][0, 256:], rays2 * 3)
    for images, intrinsics, expected in (((rgb,), (K,), out), ((rgb, rgb2), (K, K2), multi)):
        actual = model(images, intrinsics, prompt="pick up the cup")
        torch.testing.assert_close(actual["context"], expected["context"])
    for images, intrinsics in (([], []), ([rgb, rgb2], [K]), ([rgb], K)):
        try:
            model(images, intrinsics)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid view sequences must be rejected")

    # optional action stub: context -> fixed action
    action_state = torch.zeros(1, 7)
    target = torch.ones(1, 2, 7)
    mask = torch.ones_like(target)
    embodiment = torch.zeros(1, dtype=torch.long)

    class _FakeAction(nn.Module):
        def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_id=None):
            assert state is action_state
            expected = multi if fused_tokens.shape[1] == 35 else out
            torch.testing.assert_close(fused_tokens, expected["context"])
            if actions_gt is not None:
                assert actions_gt is target and action_mask is mask and embodiment_id is embodiment
                return fused_tokens.mean(), actions_gt
            return torch.zeros(fused_tokens.size(0), 7)

    act_model = LiteGauss(
        embedder, SimpleNamespace(inference_images=infer), gst, action_head=_FakeAction()
    ).eval()
    for images, intrinsics in ((rgb, K), ([rgb, rgb2], [K, K2])):
        actions = act_model(images, intrinsics, prompt="go", state=action_state)
        assert actions.shape == (1, 7)
        loss, returned_target = act_model(
            images, intrinsics, prompt="go", state=action_state, actions_gt=target,
            action_mask=mask, embodiment_id=embodiment,
        )
        assert loss.ndim == 0 and torch.isfinite(loss) and returned_target is target
    print(
        "PASS CPU wiring selfcheck",
        {"1-view context": tuple(out["context"].shape), "2-view context": tuple(multi["context"].shape),
         "action_head": "PASS (inference + training)"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Load InternVL3-1B and DA3-base on CUDA")
    args = parser.parse_args()
    path = ROOT.parent / "geom_out" / "geom_probe.npz"
    if path.exists():
        with np.load(path) as data:
            rgb = Image.fromarray(data["images"][0, 0])
            K = data["intrinsics"][0, 0].copy()
        print("RGB/K: geom_probe.npz first frame, first view")
    else:
        rgb = Image.new("RGB", (320, 240), (64, 128, 192))
        K = np.array([[200, 0, 160], [0, 200, 120], [0, 0, 1]], np.float32)
        print("RGB/K: dummy")
    if not args.real:
        selfcheck(rgb, K)
        return

    from model.internvl3.internvl3_embedder import InternVL3Embedder
    from model.depth_anything_3.depth_anything_v3 import DAV3Module

    embedder = InternVL3Embedder().eval()
    depth = DAV3Module().cuda().eval()
    dim = embedder.model.config.llm_config.hidden_size
    gst = GaussianSpatialTokenizer(dim, out_dim=dim).cuda().eval()
    out = LiteGauss(embedder, depth, gst).eval()(rgb, K, prompt="pick up the bowl")
    assert all(torch.isfinite(out[k]).all() for k in ("tokens", "lang", "context"))
    print("PASS real forward", {k: tuple(out[k].shape) for k in ("tokens", "lang", "context")})


if __name__ == "__main__":
    main()

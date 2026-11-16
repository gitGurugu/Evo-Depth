"""Thin GST/action trainer, with no downloads or GPU needed for --smoke.

CPU self-check: D:\\anaconda\\python.exe Evo_depth/scripts/train_litegauss.py --smoke
GPU wiring check (downloads InternVL3/DA3 weights): train_litegauss.py --real
Real data: --real --batch sample.pt --steps 100 --output new_checkpoint.pt
sample.pt is a weights_only-compatible dict: rgb = list of uint8 CHW tensors,
optional K = list of 3x3 matrices in original pixels, prompt = str, state = (1,S),
actions_gt = (1,H,A), optional action_mask = (1,H,A), embodiment_id = (1,).
State/actions must already be normalized. --config accepts a policy JSON dict.
Without --batch, --real checks synthetic inputs with pretrained backbones only.
shortcut: batch size one/repeated sample, import LiteGaussPolicy in a data loop for full training.
Missing K uses fixed LIBERO intrinsics: one view = agent; two = agent, wrist.
LeRobot data-loop wiring is omitted.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from model.litegauss.policy import LiteGaussPolicy
from model.litegauss.libero_intrinsics import default_libero_Ks


def fill_default_intrinsics(batch):
    """Fill missing LIBERO K in input RGB pixels, preserving explicit intrinsics."""
    if batch.get("K") is None:
        images = batch["rgb"]
        if len(images) not in (1, 2):
            raise ValueError("Default LIBERO intrinsics require one or two views (agent, wrist)")
        batch["K"] = [default_libero_Ks((image.height, image.width), (view,))[0]
                      for image, view in zip(images, ("agent", "wrist"))]
    return batch


class MockLanguage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.embedding = nn.Embedding(16, dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=[self.dropout(self.embedding(input_ids))])


class MockVLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.language_model = MockLanguage(dim)
        self.vision = nn.Linear(3, dim)

    def extract_feature(self, pixels):
        patches = F.adaptive_avg_pool2d(pixels, (2, 2)).flatten(2).transpose(1, 2)
        return self.vision(patches)


class MockEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.image_size = 16
        self.model = MockVLM(dim)

    def _preprocess_images(self, images):
        pixels = [torch.tensor(list(im.resize((16, 16)).getdata()), dtype=torch.float32)
                  .reshape(16, 16, 3).permute(2, 0, 1) / 255 for im in images]
        return torch.stack(pixels), [1] * len(images)

    def tokenizer(self, prompt, **kwargs):
        ids = torch.tensor([[len(word) % 16 for word in prompt.split()] or [0]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class MockDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def inference_images(self, images):
        return SimpleNamespace(depth=self.scale * torch.ones(len(images), 8, 10))


def dummy_batch(horizon=4, action_dim=3, state_dim=3):
    images = [Image.frombytes("RGB", (20, 16), torch.randint(
        0, 256, (16 * 20 * 3,), dtype=torch.uint8).numpy().tobytes()) for _ in range(2)]
    mask = torch.ones(1, horizon, action_dim)
    if action_dim > 1:
        mask[..., -1] = 0
    return dict(rgb=images, prompt="pick up cube",
                state=None if state_dim is None else torch.randn(1, state_dim),
                actions_gt=torch.rand(1, horizon, action_dim) * 2 - 1, action_mask=mask)


def smoke():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = dict(device="cpu", embed_dim=16, hidden_dim=32, state_hidden_dim=16,
                  state_dim=3, horizon=4, per_action_dim=3, num_layers=1,
                  num_heads=2, pool_dim=16, gst_num_heads=2, num_queries=4,
                  fourier_L=2, num_inference_timesteps=2)
    policy = LiteGaussPolicy(config, embedder=MockEmbedder(16), depth=MockDepth()).train()
    frozen = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    assert before and frozen
    assert all(n.startswith(("litegauss.gst.", "litegauss.action_head.")) for n in before)
    assert not policy.litegauss.embedder.training and not policy.litegauss.depth.training
    batch = fill_default_intrinsics(dummy_batch())
    expected_Ks = default_libero_Ks((16, 20))
    assert all(torch.equal(k, expected) for k, expected in zip(batch["K"], expected_Ks))
    explicit_K = [torch.eye(3), torch.eye(3)]
    assert fill_default_intrinsics({**batch, "K": explicit_K})["K"] is explicit_K
    assert all(torch.equal(k, expected) for k, expected in zip(
        fill_default_intrinsics({**batch, "K": None})["K"], expected_Ks))
    for images in ([], batch["rgb"] * 2):
        try:
            fill_default_intrinsics({"rgb": images})
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported default view count accepted")
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-3)
    views = []

    def retain_tokens(module, inputs, output):
        output["tokens"].retain_grad()
        views.append(output["tokens"])

    hook = policy.litegauss.gst.register_forward_hook(retain_tokens)
    loss = policy.train_step(**batch)
    hook.remove()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert len(views) == 2 and all(t.grad is not None and t.grad.abs().sum() > 0 for t in views)
    for name, param in policy.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    optimizer.step()
    params = dict(policy.named_parameters())
    assert all(torch.equal(params[n], value) for n, value in frozen.items())
    for prefix in ("litegauss.gst.", "litegauss.action_head."):
        assert any(not torch.equal(params[n], v) for n, v in before.items() if n.startswith(prefix))
    optimizer.zero_grad(set_to_none=True)
    # Invalid masks and dimension mismatches fail before entering the action head.
    for bad_mask in (torch.zeros_like(batch["action_mask"]), torch.ones(1, 1)):
        try:
            policy(**{**batch, "action_mask": bad_mask})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action mask accepted")
    try:
        LiteGaussPolicy({**config, "embed_dim": 32}, embedder=MockEmbedder(16), depth=MockDepth())
    except ValueError:
        pass
    else:
        raise AssertionError("LLM/GST/action dimension mismatch accepted")
    inputs = {k: batch[k] for k in ("rgb", "K", "prompt", "state")}
    actions = policy(**inputs, action_mask=torch.tensor([1, 1, 0]))
    assert actions.shape == (1, 4, 3) and torch.isfinite(actions).all()
    assert not actions.requires_grad and torch.equal(actions[..., -1], torch.zeros(1, 4))
    assert policy.training and not policy.litegauss.embedder.training
    single = LiteGaussPolicy({**config, "horizon": 1, "state_dim": None, "num_categories": 2},
                             embedder=MockEmbedder(16), depth=MockDepth())
    assert "litegauss.action_head.single_action_proj.weight" in dict(single.named_parameters())
    single_batch = dummy_batch(horizon=1, state_dim=None)
    single_batch["rgb"] = single_batch["rgb"][:1]
    fill_default_intrinsics(single_batch)
    assert len(single_batch["K"]) == 1 and torch.equal(single_batch["K"][0], expected_Ks[0])
    single_loss = single(**single_batch, embodiment_id=torch.tensor([1]))
    assert torch.isfinite(single_loss)
    single_loss.backward()
    assert single.litegauss.action_head.single_action_proj.weight.grad is not None
    assert dummy_batch(action_dim=1)["action_mask"].sum() > 0
    assert "model.internvl3.internvl3_embedder_da3" not in sys.modules
    assert "model.depth_anything_3.depth_anything_v3" not in sys.modules
    print(f"PASS: CPU smoke loss={loss.item():.6f}; one/two-view default LIBERO K; explicit K preserved; GST/action backward + optimizer; frozen backbones; masked inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="CPU mocks; no weights/downloads (default)")
    mode.add_argument("--real", action="store_true", help="pretrained InternVL/DA3 on CUDA")
    parser.add_argument("--config", type=Path, help="policy JSON config")
    parser.add_argument("--batch", type=Path, help="prepared tensor batch .pt (see above)")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, help="save a NEW checkpoint; refuses to overwrite")
    args = parser.parse_args()
    if not args.real:
        if args.config or args.batch or args.output or args.steps != 1:
            parser.error("--config/--batch/--steps/--output require --real")
        smoke()
        return
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output and args.output.exists():
        parser.error("--output already exists; only new files may be written")
    if not torch.cuda.is_available():
        parser.error("--real requires CUDA and installed InternVL/DA3 dependencies")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    policy = LiteGaussPolicy({**config, "device": "cuda"}).train()
    if args.batch:
        batch = torch.load(args.batch, map_location="cpu", weights_only=True)
        batch["rgb"] = [Image.fromarray(image.permute(1, 2, 0).numpy()) for image in batch["rgb"]]
    else:
        print("Synthetic batch: pretrained-backbone wiring check only.")
        batch = dummy_batch(policy.horizon, policy.per_action_dim, policy.state_dim)
    fill_default_intrinsics(batch)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = policy.train_step(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0,
                                      error_if_nonfinite=True)
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f}")
    if args.output:
        with args.output.open("xb") as handle:
            torch.save(dict(config=policy.config, gst=policy.litegauss.gst.state_dict(),
                            action_head=policy.litegauss.action_head.state_dict(),
                            optimizer=optimizer.state_dict()), handle)


if __name__ == "__main__":
    main()

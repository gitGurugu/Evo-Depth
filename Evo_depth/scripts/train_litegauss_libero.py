"""Batch-one LiteGauss LIBERO training; --selfcheck needs no data or weights.

Real example (paths must exist):
  python Evo_depth/scripts/train_litegauss_libero.py --dataset_config_path DATA.yaml --steps 2
Optional --config is a policy JSON; state/action dimensions remain 24.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dataset.litegauss_loader import LiteGaussDataset, selfcheck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--dataset_config_path", type=Path)
    parser.add_argument("--config", type=Path, help="Optional policy JSON")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional NEW checkpoint; refuses overwrite")
    args = parser.parse_args()
    if args.selfcheck:
        if args.dataset_config_path or args.config or args.output or args.steps != 1:
            parser.error("--selfcheck cannot be combined with training arguments")
        batch = selfcheck()
        # Reuse the existing mocks, exercising the real GST/action head at 24 dims.
        from train_litegauss import MockDepth, MockEmbedder
        from model.litegauss.policy import LiteGaussPolicy

        policy = LiteGaussPolicy(dict(
            device="cpu", embed_dim=16, hidden_dim=32, state_hidden_dim=16,
            state_dim=24, per_action_dim=24, action_horizon=4, num_layers=1,
            num_heads=2, pool_dim=16, gst_num_heads=2, num_queries=4, fourier_L=2,
        ), embedder=MockEmbedder(16), depth=MockDepth()).train()
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad))
        before = policy.litegauss.gst.queries.detach().clone()
        loss = policy.train_step(**batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        for parameter in policy.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            else:
                assert parameter.grad is None
        optimizer.step()
        assert not torch.equal(before, policy.litegauss.gst.queries)
        print(f"PASS: loader -> 24-dim policy loss/backward/optimizer; loss={loss.item():.6f}")
        return
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.dataset_config_path is None or not args.dataset_config_path.is_file():
        parser.error("--dataset_config_path must name an existing file")
    if args.config and not args.config.is_file():
        parser.error("--config must name an existing file")
    if args.output and args.output.exists():
        parser.error("--output already exists; only new checkpoints may be written")
    if not torch.cuda.is_available():
        parser.error("Real training requires CUDA and installed InternVL/DA3 dependencies")
    import yaml
    from model.litegauss.policy import LiteGaussPolicy

    dataset_config = yaml.safe_load(args.dataset_config_path.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    horizon = config.get("action_horizon", config.get("horizon", 16))
    dataset = LiteGaussDataset.from_config(dataset_config, action_horizon=horizon)
    if not len(dataset):
        raise ValueError("LIBERO dataset is empty")
    policy = LiteGaussPolicy({**config, "device": "cuda", "state_dim": 24, "per_action_dim": 24,
                              "action_horizon": horizon,
                              "num_categories": len(dataset_config["data_groups"])}).train()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    for step in range(args.steps):
        batch = dataset[step % len(dataset)]
        optimizer.zero_grad(set_to_none=True)
        loss = policy.train_step(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad),
                                      1.0, error_if_nonfinite=True)
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f}")
    if args.output:
        with args.output.open("xb") as handle:
            torch.save(dict(config=policy.config, gst=policy.litegauss.gst.state_dict(),
                            action_head=policy.litegauss.action_head.state_dict(),
                            optimizer=optimizer.state_dict()), handle)


if __name__ == "__main__":
    main()

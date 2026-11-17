"""Batch-one LiteGauss LIBERO training; --selfcheck needs no data or weights.

Real example (paths must exist):
  python Evo_depth/scripts/train_litegauss_libero.py --dataset_config_path DATA.yaml --steps 2
Optional --config is a policy JSON; state/action dimensions remain 24.

Checkpoint directory layout (aligned with Evo server sidecars):
  <output>/config.json
  <output>/norm_stats.json
  <output>/model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dataset.litegauss_loader import LiteGaussDataset, selfcheck

CKPT_FILES = ("config.json", "norm_stats.json", "model.pt")


def _norm_stats_from_dataset(dataset):
    stats = getattr(getattr(dataset, "base", None), "arm2stats_dict", None)
    if not isinstance(stats, dict) or not stats:
        raise ValueError("LiteGauss dataset is missing arm2stats_dict for norm_stats.json")
    if len(stats) != 1:
        raise ValueError(
            f"norm_stats.json needs exactly one robot key for the server, got {list(stats.keys())}"
        )
    return stats


def save_checkpoint_dir(
    output_dir,
    *,
    norm_stats,
    run_name,
    steps,
    lr,
    dataset_config_path,
    camera_order,
    policy_config,
    gst_state,
    action_head_state,
    optimizer_state=None,
):
    """Write Evo-style sidecars + LiteGauss weights; refuses overwrite."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        existing = [name for name in CKPT_FILES if (output_dir / name).exists()]
        if existing or any(output_dir.iterdir()):
            raise FileExistsError(
                f"checkpoint dir already exists or is non-empty: {output_dir} ({existing or 'files present'})"
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    export_config = {
        **dict(policy_config or {}),
        "model_type": "litegauss",
        "run_name": run_name,
        "steps": steps,
        "lr": lr,
        "dataset_config_path": str(dataset_config_path),
        "camera_order": list(camera_order),
    }
    (output_dir / "config.json").write_text(
        json.dumps(export_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = dict(
        config=export_config,
        norm_stats=norm_stats,
        gst=gst_state,
        action_head=action_head_state,
        run_name=run_name,
        steps=steps,
    )
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    with (output_dir / "model.pt").open("xb") as handle:
        torch.save(payload, handle)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--dataset_config_path", type=Path)
    parser.add_argument("--config", type=Path, help="Optional policy JSON")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        help="NEW checkpoint directory (writes config.json, norm_stats.json, model.pt)",
    )
    parser.add_argument("--swanlab-project", default=os.environ.get("SWANLAB_PROJECT", "litegauss"))
    parser.add_argument("--swanlab-run", default=os.environ.get("SWANLAB_RUN", None),
                        help="SwanLab run name; default litegauss-stepsN")
    parser.add_argument("--no-swanlab", action="store_true", help="Disable SwanLab logging")
    parser.add_argument(
        "--export-from-pt",
        type=Path,
        help="One-shot: convert a legacy single .pt into an Evo-style checkpoint directory",
    )
    args = parser.parse_args()
    if args.selfcheck:
        if (args.dataset_config_path or args.config or args.output or args.steps != 1
                or args.export_from_pt):
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

    if args.export_from_pt is not None:
        if args.steps != 1 or args.config or args.swanlab_run or args.no_swanlab:
            parser.error("--export-from-pt only uses --dataset_config_path and --output")
        if args.dataset_config_path is None or not args.dataset_config_path.is_file():
            parser.error("--dataset_config_path must name an existing file")
        if args.output is None:
            parser.error("--output checkpoint directory is required with --export-from-pt")
        if not args.export_from_pt.is_file():
            parser.error(f"--export-from-pt not found: {args.export_from_pt}")
        import yaml

        payload = torch.load(args.export_from_pt, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "gst" not in payload or "action_head" not in payload:
            parser.error("legacy .pt must contain gst and action_head state_dicts")
        dataset_config = yaml.safe_load(args.dataset_config_path.read_text(encoding="utf-8"))
        policy_cfg = dict(payload.get("config") or {})
        horizon = int(policy_cfg.get("action_horizon", policy_cfg.get("horizon", 16)))
        dataset = LiteGaussDataset.from_config(dataset_config, action_horizon=horizon)
        norm_stats = _norm_stats_from_dataset(dataset)
        out = save_checkpoint_dir(
            args.output,
            norm_stats=norm_stats,
            run_name=payload.get("run_name") or args.output.name,
            steps=int(payload.get("steps") or 0),
            lr=float(policy_cfg.get("lr", 1e-4)),
            dataset_config_path=args.dataset_config_path,
            camera_order=dataset.camera_order,
            policy_config=policy_cfg,
            gst_state=payload["gst"],
            action_head_state=payload["action_head"],
            optimizer_state=payload.get("optimizer"),
        )
        print(f"exported checkpoint dir={out}")
        for name in CKPT_FILES:
            print(f"  - {out / name}")
        return

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.dataset_config_path is None or not args.dataset_config_path.is_file():
        parser.error("--dataset_config_path must name an existing file")
    if args.config and not args.config.is_file():
        parser.error("--config must name an existing file")
    if args.output and args.output.exists():
        parser.error("--output already exists; only new checkpoint directories may be written")
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
    norm_stats = _norm_stats_from_dataset(dataset)
    policy = LiteGaussPolicy({**config, "device": "cuda", "state_dim": 24, "per_action_dim": 24,
                              "action_horizon": horizon,
                              "num_categories": len(dataset_config["data_groups"])}).train()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    lr = 1e-4
    run_name = args.swanlab_run or f"litegauss-steps{args.steps}"
    use_swanlab = (
        not args.no_swanlab
        and os.environ.get("SWANLAB_MODE", "cloud") != "disabled"
    )
    if use_swanlab:
        import swanlab
        swanlab.init(
            project=args.swanlab_project,
            name=run_name,
            config={
                "steps": args.steps,
                "lr": lr,
                "batch_size": 1,
                "action_horizon": horizon,
                "dataset_config_path": str(args.dataset_config_path),
                "output": str(args.output) if args.output else None,
                **{k: config[k] for k in config},
            },
        )
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
        step_i = step + 1
        loss_v = loss.item()
        print(f"step={step_i} loss={loss_v:.6f}")
        if use_swanlab:
            swanlab.log({"step": step_i, "loss": loss_v, "learning_rate": lr})
    if args.output:
        out = save_checkpoint_dir(
            args.output,
            norm_stats=norm_stats,
            run_name=run_name,
            steps=args.steps,
            lr=lr,
            dataset_config_path=args.dataset_config_path,
            camera_order=dataset.camera_order,
            policy_config=policy.config,
            gst_state=policy.litegauss.gst.state_dict(),
            action_head_state=policy.litegauss.action_head.state_dict(),
            optimizer_state=optimizer.state_dict(),
        )
        print(f"saved checkpoint dir={out}")
        for name in CKPT_FILES:
            print(f"  - {out / name}")
    if use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()

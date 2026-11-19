"""LiteGauss LIBERO training (Evo-aligned optimizer steps).

Each update uses one true sample batch: one policy forward/backward/optimizer step.
--batch-size (alias --batch_size) defaults to 16. --probe-max-batch searches for the
largest true batch up to that limit with full-length text and forward/backward/
AdamW workspace, lowering B on CUDA OOM. DA3 remains image-independent.

Real example (paths must exist):
  python Evo_depth/scripts/train_litegauss_libero.py --dataset_config_path DATA.yaml --steps 2
Optional --config is a policy JSON; state/action dimensions remain 24.

Checkpoint directory layout (aligned with Evo server sidecars):
  <output>/config.json
  <output>/norm_stats.json
  <output>/model.pt
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.litegauss_loader import LiteGaussDataset, collate_litegauss, selfcheck


def probe_max_batch(policy, dataset, max_batch):
    """Bounded integer binary search; full true batches, no parameter updates."""
    ref = policy.litegauss.gst.queries
    sample = dict(dataset[0])
    # Fixed cameras/resolution; reserve the tokenizer's maximum text length.
    sample["prompt"] = "probe " * policy.litegauss.max_text_len
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.0, weight_decay=0.0)

    def attempt(size):
        batch = collate_litegauss([sample] * size)
        loss = policy.train_step(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        # Allocate moments and include optimizer temporaries; lr=0 preserves weights.
        optimizer.step()
        torch.cuda.synchronize(ref.device)

    high = min(max_batch, len(dataset))
    candidate = high
    try:
        with torch.random.fork_rng(devices=[ref.device]):
            try:
                attempt(1)
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError("True batch B=1 does not fit; reduce model/image/horizon configuration") from None
            finally:
                optimizer.zero_grad(set_to_none=True)
            low = 1
            while low < high:
                torch.cuda.reset_peak_memory_stats(ref.device)
                try:
                    attempt(candidate)
                except torch.cuda.OutOfMemoryError:
                    fits = False
                else:
                    fits = True
                finally:
                    optimizer.zero_grad(set_to_none=True)
                print(f"probe: B={candidate} {'fits' if fits else 'OOM'} "
                      f"peak={torch.cuda.max_memory_allocated(ref.device) / 2**30:.3f} GiB")
                gc.collect()
                torch.cuda.empty_cache()
                if fits:
                    low = candidate
                else:
                    high = candidate - 1
                candidate = (low + high + 1) // 2
        print(f"probe: selected true batch_size={low} (requested upper bound={max_batch})")
        return low
    finally:
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


def check_probe(policy, sample):
    """CPU simulation of CUDA OOM boundaries, using real batch forward/backward."""
    from contextlib import nullcontext
    from unittest.mock import patch

    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    real_step = policy.train_step
    attempted = []

    def limited_step(**batch):
        size = batch["actions_gt"].shape[0]
        attempted.append(size)
        if size > 3:
            raise torch.cuda.OutOfMemoryError("simulated capacity")
        return real_step(**batch)

    with torch.random.fork_rng(devices=[]), \
         patch("torch.random.fork_rng", return_value=nullcontext()), \
         patch.multiple(torch.cuda, synchronize=lambda *args: None,
                        reset_peak_memory_stats=lambda *args: None,
                        max_memory_allocated=lambda *args: 0, empty_cache=lambda: None), \
         patch.object(policy, "train_step", side_effect=limited_step):
        assert probe_max_batch(policy, [sample] * 8, 8) == 3
        assert attempted == [1, 8, 4, 2, 3]
        assert probe_max_batch(policy, [sample] * 8, 2) == 2
        assert probe_max_batch(policy, [sample], 8) == 1
        with patch.object(policy, "train_step", side_effect=torch.cuda.OutOfMemoryError("simulated")):
            try:
                probe_max_batch(policy, [sample], 1)
            except RuntimeError as error:
                assert "B=1 does not fit" in str(error)
            else:
                raise AssertionError("B=1 OOM must fail clearly")
        with patch.object(policy, "train_step", side_effect=ValueError("bad input")):
            try:
                probe_max_batch(policy, [sample], 1)
            except ValueError:
                pass
            else:
                raise AssertionError("Probe must propagate non-OOM errors")
    assert all(torch.equal(before[name], p) for name, p in policy.named_parameters())
    assert all(p.grad is None for p in policy.parameters())
    print("PASS: probe OOM search/top-fit/dataset cap/B=1 failure/non-OOM propagation; weights unchanged")


def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    """Same schedule as Evo train.py: linear warmup, then cosine to 0."""

    def lr_lambda(current_step):
        current_step = current_step + resume_step
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda

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
    step=None,
):
    """Write Evo-style sidecars + LiteGauss weights; refuses overwrite of CKPT_FILES."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in CKPT_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"checkpoint dir already has {existing}: {output_dir}"
        )
    export_config = {
        **dict(policy_config or {}),
        "model_type": "litegauss",
        "run_name": run_name,
        "steps": steps,
        "lr": lr,
        "dataset_config_path": str(dataset_config_path),
        "camera_order": list(camera_order),
    }
    if step is not None:
        export_config["step"] = int(step)
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
    if step is not None:
        payload["step"] = int(step)
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    with (output_dir / "model.pt").open("xb") as handle:
        torch.save(payload, handle)
    return output_dir


def load_trainable_checkpoint(policy, optimizer, ckpt_dir):
    """Load gst / action_head / optimizer from a LiteGauss checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)
    model_path = ckpt_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing {model_path}")
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "gst" not in payload or "action_head" not in payload:
        raise ValueError(f"invalid LiteGauss checkpoint: {model_path}")
    policy.litegauss.gst.load_state_dict(payload["gst"])
    policy.litegauss.action_head.load_state_dict(payload["action_head"])
    if "optimizer" in payload and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    step = int(payload.get("step") or 0)
    return step, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--dataset_config_path", type=Path)
    parser.add_argument("--config", type=Path, help="Optional policy JSON")
    parser.add_argument("--steps", type=int, default=1, help="Optimizer steps (each uses batch_size samples)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16,
                        help="True samples per forward/backward/optimizer step (default: 16)")
    parser.add_argument("--probe-max-batch", action="store_true",
                        help="Before training, find largest true B <= --batch-size fitting CUDA VRAM; OOM lowers B")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak AdamW lr after warmup")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Linear warmup steps (Evo train.sh default); cosine decay afterward",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 is safest with video+CUDA)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Checkpoint directory (final files + optional step_N/ mid-saves)",
    )
    parser.add_argument(
        "--ckpt-interval",
        "--ckpt_interval",
        type=int,
        default=0,
        help="Save mid-run checkpoint every N steps under output/step_N/ (0=only final)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a LiteGauss checkpoint dir (loads gst/action_head/optimizer + step)",
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
                or args.export_from_pt or args.batch_size != 16 or args.num_workers != 0
                or args.lr != 1e-4 or args.warmup_steps != 1000 or args.probe_max_batch
                or args.ckpt_interval != 0 or args.resume):
            parser.error("--selfcheck cannot be combined with training arguments")
        batch = selfcheck()
        # Reuse the existing mocks, exercising the real GST/action head at 24 dims.
        from train_litegauss import MockDepth, MockEmbedder, check_true_batch
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
        check_true_batch(policy, batch, {**batch, "state": batch["state"] + 0.25})
        check_probe(policy, batch)
        return

    if args.export_from_pt is not None:
        if (args.steps != 1 or args.config or args.swanlab_run or args.no_swanlab
                or args.batch_size != 16 or args.lr != 1e-4 or args.warmup_steps != 1000
                or args.probe_max_batch or args.ckpt_interval != 0 or args.resume):
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
            step=payload.get("step"),
        )
        print(f"exported checkpoint dir={out}")
        for name in CKPT_FILES:
            print(f"  - {out / name}")
        return

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup_steps must be >= 0")
    if args.ckpt_interval < 0:
        parser.error("--ckpt-interval must be >= 0")
    if args.num_workers < 0:
        parser.error("--num_workers must be >= 0")
    if args.dataset_config_path is None or not args.dataset_config_path.is_file():
        parser.error("--dataset_config_path must name an existing file")
    if args.config and not args.config.is_file():
        parser.error("--config must name an existing file")
    if args.resume is not None and not (args.resume / "model.pt").is_file():
        parser.error(f"--resume missing model.pt: {args.resume}")
    if args.ckpt_interval > 0 and args.output is None:
        parser.error("--ckpt-interval requires --output")
    if args.output and (args.output / "model.pt").exists() and args.resume is None:
        parser.error("--output already has model.pt; use a new --name or --resume")
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
    if len(dataset) < args.batch_size and not args.probe_max_batch:
        raise ValueError(
            f"dataset length {len(dataset)} < batch_size {args.batch_size}; lower --batch_size"
        )
    norm_stats = _norm_stats_from_dataset(dataset)
    policy = LiteGaussPolicy({**config, "device": "cuda", "state_dim": 24, "per_action_dim": 24,
                              "action_horizon": horizon,
                              "num_categories": len(dataset_config["data_groups"])}).train()
    lr = args.lr
    warmup_steps = args.warmup_steps
    batch_size = (probe_max_batch(policy, dataset, args.batch_size)
                  if args.probe_max_batch else args.batch_size)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr)
    step = 0
    if args.resume is not None:
        step, _ = load_trainable_checkpoint(policy, optimizer, args.resume)
        print(f"resume: loaded {args.resume} at step={step}")
    if step >= args.steps:
        parser.error(f"--resume step {step} >= --steps {args.steps}; raise --steps")
    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, args.steps, resume_step=step))
    run_name = args.swanlab_run or f"litegauss-steps{args.steps}"
    use_swanlab = (
        not args.no_swanlab
        and os.environ.get("SWANLAB_MODE", "cloud") != "disabled"
    )
    policy_config = {
        **policy.config,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "lr_schedule": "warmup_cosine",
    }
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    def write_checkpoint(target_dir, current_step):
        out = save_checkpoint_dir(
            target_dir,
            norm_stats=norm_stats,
            run_name=run_name,
            steps=args.steps,
            lr=lr,
            dataset_config_path=args.dataset_config_path,
            camera_order=dataset.camera_order,
            policy_config=policy_config,
            gst_state=policy.litegauss.gst.state_dict(),
            action_head_state=policy.litegauss.action_head.state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=current_step,
        )
        print(f"saved checkpoint dir={out} step={current_step}")
        return out

    if use_swanlab:
        import swanlab
        swanlab.init(
            project=args.swanlab_project,
            name=run_name,
            config={
                "steps": args.steps,
                "lr": lr,
                "warmup_steps": warmup_steps,
                "lr_schedule": "warmup_cosine",
                "batch_size": batch_size,
                "samples_per_step": batch_size,
                "action_horizon": horizon,
                "dataset_config_path": str(args.dataset_config_path),
                "output": str(args.output) if args.output else None,
                "ckpt_interval": args.ckpt_interval,
                "resume": str(args.resume) if args.resume else None,
                "resume_step": step,
                "dataset_len": len(dataset),
                **{k: config[k] for k in config},
            },
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_litegauss,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"train: steps={args.steps} start_step={step} batch_size={batch_size} lr={lr} "
        f"warmup_steps={warmup_steps} ckpt_interval={args.ckpt_interval} "
        f"schedule=warmup_cosine "
        f"dataset_len={len(dataset)} (~{args.steps * batch_size / len(dataset):.4f} epoch)"
    )
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = policy.train_step(**batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad),
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            loss_v = loss.item()
            lr_now = scheduler.get_last_lr()[0]
            print(f"step={step} loss={loss_v:.6f} lr={lr_now:.6e} batch_size={batch_size}")
            if use_swanlab:
                swanlab.log({
                    "step": step,
                    "loss": loss_v,
                    "learning_rate": lr_now,
                    "batch_size": batch_size,
                    "samples_seen": step * batch_size,
                })
            if (
                args.output
                and args.ckpt_interval > 0
                and step % args.ckpt_interval == 0
                and step < args.steps
            ):
                write_checkpoint(args.output / f"step_{step}", step)
    if args.output:
        # Resume into the same run dir may already have a prior final model.pt.
        if args.resume is not None:
            for name in CKPT_FILES:
                path = args.output / name
                if path.exists():
                    path.unlink()
        out = write_checkpoint(args.output, step)
        for name in CKPT_FILES:
            print(f"  - {out / name}")
    if use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    main()

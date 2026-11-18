"""Build mono DA3-base depth cache using the exact LiteGauss training RGB path."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.litegauss_depth_cache import DEFAULT_CACHE_ROOT, cache_depth, load_cached_depth
from dataset.litegauss_loader import LiteGaussDataset


def build_cache(dataset, root, depth_factory, log_every=100):
    depth_provider = None
    written = skipped = 0
    started = time.monotonic()
    for index in range(len(dataset)):
        sample = dataset[index]
        for rgb, camera in zip(sample["rgb"], dataset.camera_order):
            if load_cached_depth(root, rgb, camera, allow_online_da3=True) is not None:
                skipped += 1
                continue
            if depth_provider is None:
                depth_provider = depth_factory()
            written += int(cache_depth(root, rgb, camera, depth_provider))
        if (index + 1) % log_every == 0 or index + 1 == len(dataset):
            elapsed = time.monotonic() - started
            eta = elapsed / (index + 1) * (len(dataset) - index - 1)
            print(f"cache: samples={index + 1}/{len(dataset)} written={written} skipped={skipped} "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)
    return written, skipped


def selfcheck():
    import tempfile
    from unittest.mock import patch

    import numpy as np
    import torch

    from dataset.litegauss_depth_cache import depth_cache_path
    from dataset.litegauss_loader import collate_litegauss, selfcheck as loader_selfcheck
    from model.litegauss.policy import LiteGaussPolicy
    from train_litegauss import MockDepth, MockEmbedder

    sample = loader_selfcheck()
    config = dict(max_state_dim=24, max_action_dim=24, max_views=3,
                  data_groups={"libero_franka": {"fake": {"view_map": {
                      "image_1": "observation.images.image",
                      "image_2": "observation.images.wrist_image"}}}})
    raw = dict(images=torch.stack([
        torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        for image in sample["rgb"]] + [torch.zeros(3, 448, 448)]),
        image_mask=torch.tensor([True, True, False]), prompt=sample["prompt"],
        state=sample["state"][0], state_mask=sample["action_mask"][0, 0],
        action=sample["actions_gt"][0], action_mask=sample["action_mask"][0],
        embodiment_id=sample["embodiment_id"][0])

    class FakeBase:
        def __init__(self):
            self.config, self.action_horizon = config, 4

        def __len__(self):
            return 2  # Reused frame: same key even at a different sample index.

        def __getitem__(self, index):
            return raw

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = LiteGaussDataset(FakeBase())
        depth = MockDepth()
        with patch.object(depth, "inference_images", wraps=depth.inference_images) as infer:
            assert build_cache(dataset, root, lambda: depth, log_every=2) == (2, 2)
            assert infer.call_count == 2 and all(len(call.args[0]) == 1 for call in infer.call_args_list)
        def no_weights():
            raise AssertionError("Resume must not load DA3")
        assert build_cache(dataset, root, no_weights, log_every=2) == (0, 4)
        rgb = sample["rgb"][0]
        path = depth_cache_path(root, rgb, "agent")
        assert path == depth_cache_path(root, rgb.copy(), "agent")
        assert path != depth_cache_path(root, rgb, "wrist")
        changed = rgb.copy()
        changed.putpixel((0, 0), (0, 0, 0))
        assert path != depth_cache_path(root, changed, "agent")
        interrupted_path = depth_cache_path(root, changed, "agent")
        with patch("dataset.litegauss_depth_cache.os.replace", side_effect=OSError("interrupted")):
            try:
                cache_depth(root, changed, "agent", depth)
            except OSError:
                pass
            else:
                raise AssertionError("Write error must propagate")
        assert not interrupted_path.exists() and not list(root.rglob("*.tmp"))
        cached = load_cached_depth(root, rgb, "agent")
        assert cached.dtype == np.float32 and cached.shape == (8, 10)

        loaded = LiteGaussDataset(FakeBase(), root)
        batch = collate_litegauss([loaded[0], loaded[1]])
        for invalid in ({**loaded[1], "depths": []}, sample):
            try:
                collate_litegauss([loaded[0], invalid])
            except ValueError:
                pass
            else:
                raise AssertionError("Ambiguous depth batch accepted")
        policy = LiteGaussPolicy(dict(
            device="cpu", embed_dim=16, hidden_dim=32, state_hidden_dim=16,
            state_dim=24, per_action_dim=24, action_horizon=4, num_layers=1,
            num_heads=2, pool_dim=16, gst_num_heads=2, num_queries=4, fourier_L=2,
        ), embedder=MockEmbedder(16), allow_online_da3=False).train()
        core = policy.litegauss
        with patch.object(core.depth, "forward", side_effect=AssertionError("No online DA3")):
            loss = policy.train_step(**batch)
            loss.backward()
            assert torch.isfinite(loss) and core.gst.queries.grad.abs().sum() > 0
        with torch.no_grad():
            _, _, context, mask = core.encode_context(batch["rgb"], batch["K"], batch["prompt"], batch["depths"])
            _, _, single, _ = core.encode_context(sample["rgb"], sample["K"], sample["prompt"], loaded[0]["depths"])
            torch.testing.assert_close(context[0, mask[0]], single[0], atol=2e-5, rtol=2e-5)
            with patch.object(core, "depth", depth), patch.object(core, "allow_online_da3", True):
                _, _, online, _ = core.encode_context(batch["rgb"], batch["K"], batch["prompt"])
            torch.testing.assert_close(context, online, rtol=0, atol=0)
            # Distinct depths verify the sample/view flatten order, not just shapes.
            distinct_depths = [[cached * value for value in row] for row in ((1, 2), (3, 4))]
            from model.litegauss.gst import align_depth_to_patches
            with patch("model.litegauss.litegauss.align_depth_to_patches", wraps=align_depth_to_patches) as align:
                _, _, distinct_context, distinct_mask = core.encode_context(
                    batch["rgb"], batch["K"], batch["prompt"], distinct_depths)
                assert [call.args[0].mean().item() for call in align.call_args_list] == [1, 2, 3, 4]
            for index in range(2):
                _, _, separate, _ = core.encode_context(
                    batch["rgb"][index], list(batch["K"][index]), batch["prompt"][index], distinct_depths[index])
                torch.testing.assert_close(distinct_context[index, distinct_mask[index]], separate[0],
                                           atol=2e-5, rtol=2e-5)

        path.unlink()
        try:
            loaded[0]
        except FileNotFoundError as error:
            assert "--allow-online-da3" in str(error)
        else:
            raise AssertionError("Required cache miss must fail")
        fallback = LiteGaussDataset(FakeBase(), root, allow_online_da3=True)
        mixed = collate_litegauss([fallback[0], fallback[1]])
        assert mixed["depths"][0][0] is None and mixed["depths"][0][1] is not None
        try:
            policy.train_step(**mixed)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Model must enforce cache-required mode too")
        with patch.object(core, "depth", depth), patch.object(core, "allow_online_da3", True), \
             patch.object(depth, "inference_images", wraps=depth.inference_images) as infer:
            assert torch.isfinite(policy.train_step(**mixed))
            assert infer.call_count == 2 and all(len(call.args[0]) == 1 for call in infer.call_args_list)

        # Corrupt entries fail clearly, even when online fallback is allowed.
        np.savez(path, depth=np.array([[np.nan]], dtype=np.float32))
        try:
            load_cached_depth(root, rgb, "agent", allow_online_da3=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid cache depth accepted")
        assert "model.depth_anything_3.depth_anything_v3" not in sys.modules
    print("PASS: cache keys/camera/pixels; float32; mono builder; dedupe/resume; loader -> B=2 GST/backward; "
          "per-sample/view ordering; cached/online and B=1 parity; "
          "required miss/mixed fallback/corruption/atomic write; no DA3 weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config_path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dataset/config.libero.yaml")
    parser.add_argument("--depth-cache-root", type=Path,
                        default=Path(os.environ.get("LITEGAUSS_DEPTH_CACHE", DEFAULT_CACHE_ROOT)))
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.log_every < 1 or args.action_horizon < 1:
        parser.error("--log-every and --action-horizon must be positive")
    import yaml
    import torch

    config = yaml.safe_load(args.dataset_config_path.read_text(encoding="utf-8"))
    dataset = LiteGaussDataset.from_config(config, action_horizon=args.action_horizon)
    if not len(dataset):
        raise ValueError("LIBERO dataset is empty")

    def load_depth():
        from model.depth_anything_3.depth_anything_v3 import DAV3Module
        return DAV3Module().to(args.device).eval()

    print(f"cache root={args.depth_cache_root} dataset={args.dataset_config_path}", flush=True)
    with torch.inference_mode():
        build_cache(dataset, args.depth_cache_root, load_depth, args.log_every)


if __name__ == "__main__":
    main()

"""Minimal LiteGauss websocket server (same JSON protocol as Evo_depth_server).

Checkpoint directory (LiteGauss train export):
  config.json
  norm_stats.json
  model.pt          # {gst, action_head, ...}

Env:
  EVO_DEPTH_CKPT_DIR / LITEGAUSS_CKPT_DIR  checkpoint directory
  EVO_DEPTH_SERVER_PORT                    default 9000

LIBERO client sends 3 images + image_mask; only masked-in views are used, in
camera_order from config (default agent, wrist).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import websockets
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.litegauss.libero_intrinsics import default_libero_Ks
from model.litegauss.policy import LiteGaussPolicy


class Normalizer:
    """Same min/max state/action scaling as Evo_depth_server.Normalizer."""

    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r", encoding="utf-8") as handle:
                stats = json.load(handle)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                x = torch.cat([x, torch.zeros(24 - x.shape[0], dtype=torch.float32)], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")
        robot_stats = stats[list(stats.keys())[0]]
        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def decode_image_to_pil(img_list, size=448) -> Image.Image:
    """Client HWC uint8 list -> RGB PIL at `size` (match LiteGauss train RGB)."""
    img = np.asarray(img_list, dtype=np.uint8)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image list, got shape {getattr(img, 'shape', None)}")
    img = cv2.resize(img, (size, size))
    return Image.fromarray(img)


def load_policy_and_normalizer(ckpt_dir: str | Path):
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    stats_path = ckpt_dir / "norm_stats.json"
    weights_path = ckpt_dir / "model.pt"
    for path in (config_path, stats_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") not in (None, "litegauss"):
        raise ValueError(f"expected LiteGauss config, got model_type={config.get('model_type')!r}")

    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "gst" not in payload or "action_head" not in payload:
        raise ValueError("model.pt must contain gst and action_head state_dicts")

    policy_cfg = {
        **config,
        "device": "cuda",
        "state_dim": int(config.get("state_dim", 24)),
        "per_action_dim": int(config.get("per_action_dim", 24)),
        "action_horizon": int(config.get("action_horizon", config.get("horizon", 16))),
        "num_categories": int(config.get("num_categories", 1)),
        "num_inference_timesteps": int(config.get("num_inference_timesteps", 20)),
    }
    policy = LiteGaussPolicy(policy_cfg).eval()
    policy.litegauss.gst.load_state_dict(payload["gst"])
    policy.litegauss.action_head.load_state_dict(payload["action_head"])
    policy.freeze_backbones()
    policy.eval()

    camera_order = tuple(config.get("camera_order") or ("agent", "wrist"))
    normalizer = Normalizer(json.loads(stats_path.read_text(encoding="utf-8")))
    return policy, normalizer, camera_order


def infer_from_json_dict(data: dict, policy: LiteGaussPolicy, normalizer: Normalizer, camera_order):
    device = next(policy.parameters()).device
    images = data["image"]
    image_mask = data["image_mask"]
    if len(images) != len(image_mask):
        raise ValueError("image and image_mask length mismatch")

    selected = [decode_image_to_pil(img) for img, keep in zip(images, image_mask) if int(keep)]
    if len(selected) != len(camera_order):
        raise ValueError(
            f"expected {len(camera_order)} active views {camera_order}, got {len(selected)} from image_mask"
        )
    Ks = default_libero_Ks((selected[0].height, selected[0].width), camera_order)

    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros(1, 24 - state.shape[1], device=device)], dim=1)
    elif state.shape[1] > 24:
        state = state[:, :24]
    norm_state = normalizer.normalize_state(state)

    action_mask = torch.tensor(data["action_mask"], dtype=torch.float32, device=device)
    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)
    if action_mask.shape[1] < policy.per_action_dim:
        pad = torch.zeros(1, policy.per_action_dim - action_mask.shape[1], device=device)
        action_mask = torch.cat([action_mask, pad], dim=1)
    action_mask = action_mask[:, : policy.per_action_dim]

    with torch.no_grad():
        actions = policy.predict(
            rgb=selected,
            K=Ks,
            prompt=data.get("prompt", ""),
            state=norm_state,
            action_mask=action_mask,
            embodiment_id=torch.zeros(1, dtype=torch.long, device=device),
        )
        actions = normalizer.denormalize_action(actions[0])
        return actions.detach().float().cpu().numpy().tolist()


async def handle_request(websocket, policy, normalizer, camera_order):
    print("Client connected")
    try:
        async for message in websocket:
            json_data = json.loads(message)
            print("Received JSON observation")
            actions = infer_from_json_dict(json_data, policy, normalizer, camera_order)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


if __name__ == "__main__":
    ckpt_dir = os.environ.get("LITEGAUSS_CKPT_DIR") or os.environ.get("EVO_DEPTH_CKPT_DIR")
    if not ckpt_dir:
        raise SystemExit("Set LITEGAUSS_CKPT_DIR or EVO_DEPTH_CKPT_DIR to a LiteGauss checkpoint directory")
    port = int(os.environ.get("EVO_DEPTH_SERVER_PORT", "9000"))

    print(f"Loading LiteGauss from {ckpt_dir} ...")
    policy, normalizer, camera_order = load_policy_and_normalizer(ckpt_dir)
    print(f"camera_order={camera_order} horizon={policy.horizon}")

    async def main():
        # Same ready string family as Evo so wrappers can grep either.
        print(f"LiteGauss server running at ws://0.0.0.0:{port}")
        print(f"Evo_depth server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, policy, normalizer, camera_order),
            "0.0.0.0",
            port,
            max_size=100_000_000,
        ):
            await asyncio.Future()

    asyncio.run(main())

"""LIBERO adapter: padded LeRobot tensors -> LiteGaussPolicy kwargs.

Run this file directly for a CPU fake-data selfcheck, without videos or weights.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torch.utils.data import Dataset

from model.litegauss.libero_intrinsics import default_libero_Ks


def collate_litegauss(items):
    """RGB [B][V], K [B,V,3,3], prompt [B], tensors [B,...]."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    views = len(items[0]["rgb"])
    sizes = [image.size for image in items[0]["rgb"]]
    if not views or any(len(item["rgb"]) != views or len(item["K"]) != views
                        or [image.size for image in item["rgb"]] != sizes for item in items):
        raise ValueError("Batch samples must have the same cameras and image sizes")
    return dict(
        rgb=[item["rgb"] for item in items],
        K=torch.stack([torch.stack([torch.as_tensor(k) for k in item["K"]]) for item in items]),
        prompt=[item["prompt"] for item in items],
        **{key: torch.cat([item[key] for item in items], dim=0)
           for key in ("state", "actions_gt", "action_mask", "embodiment_id")},
    )


def _camera_order(config):
    if config["max_state_dim"] != 24 or config["max_action_dim"] != 24:
        raise ValueError("LIBERO adapter requires max_state_dim=max_action_dim=24")
    cameras = {"image_1": ("observation.images.image", "agent"),
               "image_2": ("observation.images.wrist_image", "wrist")}
    orders = []
    for group in config["data_groups"].values():
        for dataset in group.values():
            view_map = dataset.get("view_map", {})
            if not view_map or any(key not in cameras or cameras[key][0] != value
                                   for key, value in view_map.items()):
                raise ValueError("Explicit LIBERO image_1=agent/image_2=wrist view_map required")
            orders.append(tuple(cameras[key][1] for key in view_map))
    if not orders or any(order != orders[0] for order in orders):
        raise ValueError("All datasets must use the same explicit camera order")
    if config["max_views"] < len(orders[0]):
        raise ValueError("max_views cannot be smaller than the configured camera count")
    return orders[0]


class LiteGaussDataset(Dataset):
    """Wrap an existing LeRobotDataset; each item can go to policy.train_step(**item)."""

    def __init__(self, base_dataset):
        self.camera_order = _camera_order(base_dataset.config)
        self.base = base_dataset
        # Fixed K is valid only for the base loader's plain Resize, without crops.
        self.base.use_augmentation = False

    @classmethod
    def from_config(cls, config, action_horizon=16):
        _camera_order(config)
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        # Keep pandas/torchvision/video dependencies out of the fake selfcheck.
        from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset

        return cls(LeRobotDataset(config, image_size=448, action_horizon=action_horizon,
                                 use_augmentation=False))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if self.base.use_augmentation:
            raise ValueError("Augmentation must remain disabled for fixed LIBERO K")
        sample = self.base[index]
        images, image_mask = sample["images"], sample["image_mask"]
        if (images.shape != (self.base.config["max_views"], 3, 448, 448)
                or image_mask.shape != images.shape[:1] or image_mask.dtype != torch.bool):
            raise ValueError("Expected images (V,3,448,448) and bool image_mask (V,)")
        real = image_mask.nonzero(as_tuple=True)[0].tolist()
        # The base loader compacts missing views, losing their camera identity.
        if real != list(range(len(self.camera_order))):
            raise ValueError("Missing/ambiguous cameras: all configured views must be present")
        rgb = []
        for view in real:
            image = images[view].detach().cpu()
            if (not image.is_floating_point() or not torch.isfinite(image).all()
                    or image.min() < 0 or image.max() > 1):
                raise ValueError("Real RGB views must be finite float CHW tensors in [0,1]")
            pixels = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            rgb.append(Image.fromarray(pixels))
        state, state_mask = sample["state"], sample["state_mask"]
        actions, action_mask = sample["action"], sample["action_mask"]
        if state.shape != (24,) or state_mask.shape != state.shape:
            raise ValueError("Expected padded state/state_mask (24,)")
        if actions.shape != (self.base.action_horizon, 24) or action_mask.shape != actions.shape:
            raise ValueError("Expected stacked action/action_mask (horizon,24), without flattening")
        if (state_mask.dtype != torch.bool or action_mask.dtype != torch.bool
                or not state_mask.any() or not action_mask.any()
                or not torch.isfinite(state).all() or not torch.isfinite(actions).all()):
            raise ValueError("State/actions must be finite with nonempty bool masks")
        return dict(rgb=rgb, K=default_libero_Ks(images.shape[-2:], self.camera_order),
                    prompt=sample["prompt"], state=state.masked_fill(~state_mask, 0).unsqueeze(0),
                    actions_gt=actions.unsqueeze(0), action_mask=action_mask.unsqueeze(0),
                    embodiment_id=sample["embodiment_id"].reshape(1))


def selfcheck():
    """Exercise padding, view identity, pixels and shapes with an in-memory base."""
    torch.set_num_threads(1)
    config = dict(max_state_dim=24, max_action_dim=24, max_views=3,
                  data_groups={"libero_franka": {"fake": {"view_map": {
                      "image_1": "observation.images.image",
                      "image_2": "observation.images.wrist_image"}}}})
    actions = torch.zeros(4, 24, dtype=torch.bfloat16)
    actions[:, :7] = torch.arange(28).reshape(4, 7) / 28
    action_mask = torch.zeros(4, 24, dtype=torch.bool)
    action_mask[:, :7] = True
    state_mask = action_mask[0].clone()
    images = torch.zeros(3, 3, 448, 448)
    images[0, 0] = 1
    images[1, 1] = 128 / 255
    images[2] = float("nan")  # Dummy pixels must never reach PIL or K.
    sample = dict(images=images, image_mask=torch.tensor([True, True, False]),
                  prompt="pick up cube", state=torch.ones(24, dtype=torch.bfloat16),
                  state_mask=state_mask, action=actions, action_mask=action_mask,
                  embodiment_id=torch.tensor(0))

    class FakeDataset(Dataset):
        def __init__(self):
            self.config = config
            self.action_horizon = 4
            self.use_augmentation = True

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0 and not self.use_augmentation
            return sample

    loader = LiteGaussDataset(FakeDataset())
    batch = loader[0]
    assert len(loader) == 1 and len(batch["rgb"]) == len(batch["K"]) == 2
    assert all(image.mode == "RGB" and image.size == (448, 448) for image in batch["rgb"])
    assert batch["rgb"][0].getpixel((0, 0)) == (255, 0, 0)
    assert batch["rgb"][1].getpixel((0, 0)) == (0, 128, 0)
    assert all(torch.equal(k, expected) for k, expected in zip(batch["K"], default_libero_Ks()))
    assert batch["actions_gt"].shape == batch["action_mask"].shape == (1, 4, 24)
    assert torch.equal(batch["actions_gt"][0], actions)
    assert batch["state"].shape == (1, 24) and batch["state"][0, 7:].count_nonzero() == 0
    assert torch.equal(batch["action_mask"][0], action_mask) and action_mask.sum() == 28
    assert batch["prompt"] == sample["prompt"] and batch["embodiment_id"].tolist() == [0]

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("Invalid sample/config accepted")

    collated = collate_litegauss([batch, batch])
    assert collated["state"].shape == (2, 24)
    assert collated["actions_gt"].shape == collated["action_mask"].shape == (2, 4, 24)
    assert collated["K"].shape == (2, 2, 3, 3)
    assert len(collated["rgb"]) == len(collated["prompt"]) == 2
    rejects(lambda: collate_litegauss([]))
    rejects(lambda: collate_litegauss([batch, {**batch, "rgb": batch["rgb"][:1]}]))
    rejects(lambda: collate_litegauss([batch, {**batch, "rgb": [Image.new("RGB", (32, 32))] * 2}]))

    for key, bad in (("image_mask", torch.tensor([True, False, False])),
                     ("image_mask", torch.tensor([False, True, False])),
                     ("image_mask", torch.tensor([True, True, True])),
                     ("image_mask", torch.zeros(3, dtype=torch.bool)),
                     ("action", actions.flatten()),
                     ("action_mask", torch.zeros_like(action_mask))):
        original = sample[key]
        sample[key] = bad
        rejects(lambda: loader[0])
        sample[key] = original
    images[0, 0, 0, 0] = 2
    rejects(lambda: loader[0])
    images[0, 0, 0, 0] = 1
    loader.base.use_augmentation = True
    rejects(lambda: loader[0])
    loader.base.use_augmentation = False
    rejects(lambda: _camera_order({**config, "max_action_dim": 7}))

    # A single wrist view is safe when the configuration explicitly names it.
    single_config = {**config, "data_groups": {"libero_franka": {"fake": {"view_map": {
        "image_2": "observation.images.wrist_image"}}}}}
    single_base = FakeDataset()
    single_base.config = single_config
    sample["image_mask"] = torch.tensor([True, False, False])
    single = LiteGaussDataset(single_base)[0]
    assert len(single["rgb"]) == 1
    assert torch.equal(single["K"][0], default_libero_Ks(order=("wrist",))[0])
    sample["image_mask"] = torch.tensor([True, True, False])
    assert "dataset.lerobot_dataset_pretrain_mp" not in sys.modules
    print("PASS: fake LIBERO loader; real RGB/K only; camera identity; (1,H,24) masks; no augmentation")
    return batch


if __name__ == "__main__":
    selfcheck()

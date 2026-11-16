"""Fixed LIBERO intrinsics per camera/resolution; no live extrinsics are needed.

shortcut: RandomResizedCrop is ignored; update K with crop parameters for exact augmentation geometry.
"""

import torch

from .gst import scale_intrinsics as scale_K


K_agent_256 = torch.tensor([[309.01935, 0, 128], [0, 309.01935, 128], [0, 0, 1]],
                           dtype=torch.float32)
K_wrist_256 = torch.tensor([[166.81285, 0, 128], [0, 166.81285, 128], [0, 0, 1]],
                           dtype=torch.float32)


def default_libero_Ks(image_hw=(448, 448), order=("agent", "wrist")):
    """Return (3,3) tensors in view order, scaled from the 256x256 reference."""
    cameras = {"agent": K_agent_256, "wrist": K_wrist_256}
    if len(image_hw) != 2 or any(size <= 0 for size in image_hw):
        raise ValueError("image_hw must contain positive height and width")
    if any(view not in cameras for view in order):
        raise ValueError("LIBERO cameras must be 'agent' or 'wrist'")
    return [scale_K(cameras[view], (256, 256), image_hw) for view in order]


if __name__ == "__main__":
    agent, wrist = default_libero_Ks()
    for ref, scaled in zip((K_agent_256, K_wrist_256), (agent, wrist)):
        assert scaled.dtype == torch.float32 and scaled.shape == (3, 3)
        assert torch.allclose(scaled[:2], ref[:2] * (448 / 256))
        assert torch.equal(scaled[2], ref[2])
    assert torch.equal(default_libero_Ks((256, 256), ("wrist",))[0], K_wrist_256)
    rectangular = default_libero_Ks((128, 512), ("agent",))[0]
    assert torch.allclose(rectangular[0], K_agent_256[0] * 2)
    assert torch.allclose(rectangular[1], K_agent_256[1] / 2)
    print(f"PASS: LIBERO K_448 fx agent={agent[0, 0]:.5f}, wrist={wrist[0, 0]:.5f}")

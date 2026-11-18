"""Content-addressed mono DA3-base depths for unaugmented LiteGauss RGBs."""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data/litegauss_da3_depth_cache"
CACHE_VERSION = "da3-base-mono-rgb448-v1"


def depth_cache_path(root, rgb, camera):
    if camera not in ("agent", "wrist") or rgb.mode != "RGB" or rgb.size != (448, 448):
        raise ValueError("Depth cache requires agent/wrist RGB at 448x448, without augmentation")
    # The base loader can substitute a random sample on decode failure and exposes no
    # frame ID. Hash the actual training pixels, never the requested dataset index.
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    return Path(root) / CACHE_VERSION / camera / digest[:2] / f"{digest}.npz"


def validate_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not all(depth.shape) or not np.isfinite(depth).all():
        raise ValueError("Expected a nonempty finite DA3 depth map (H,W)")
    return depth


def load_cached_depth(root, rgb, camera, *, allow_online_da3=False):
    path = depth_cache_path(root, rgb, camera)
    if not path.is_file():
        if allow_online_da3:
            return None
        raise FileNotFoundError(f"Missing DA3 depth: {path}. Build the cache first or use --allow-online-da3")
    with np.load(path, allow_pickle=False) as cached:
        return validate_depth(cached["depth"])


def cache_depth(root, rgb, camera, depth_provider):
    """Return True when written, False on a valid hit; call DA3 only on a miss."""
    if load_cached_depth(root, rgb, camera, allow_online_da3=True) is not None:
        return False
    prediction = depth_provider.inference_images([rgb])
    depth = prediction.depth
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().float().numpy()
    if np.ndim(depth) != 3 or depth.shape[0] != 1:
        raise ValueError("Mono DA3 must return exactly one depth map (1,H,W)")
    depth = validate_depth(depth[0])
    path = depth_cache_path(root, rgb, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete files so an interrupted run can safely resume.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, depth=depth)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True

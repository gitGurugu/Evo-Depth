import torch
import torch.nn as nn
import torch.nn.functional as F
from .api import DepthAnything3
import glob
import os

    
class DAV3Module(nn.Module):
    def __init__(self, config=None):
        super(DAV3Module, self).__init__()
        
        self.model = DepthAnything3.from_pretrained("depth-anything/da3-base", )
        
        

    def inference_images(self, images):
        # images: List of image paths
        return self.model.inference(images)
    
    def extract_features(self, image_tensor,image_mask = None):
        feats = self.model._get_dpt_embeddings(image_tensor)
        assert image_mask is not None, "Image mask must be provided for feature extraction."
        final_feats = []
        for i in range(len(image_mask)):
            if image_mask[i] == True:
                final_feats.append(feats[:, i, :, :])
        
        return torch.stack(final_feats, dim=1) if final_feats else feats


    def forward(self, image_tensor, image_mask):
        
        feats = self.extract_features(image_tensor, image_mask)
        return feats


class DAV3MetricModule(nn.Module):
    """Monocular metric depth. Parallel to DAV3Module; not used by FiLM.

    Cameras never enter the network. Real K is applied after:
    Z = ((fx + fy) / 2) * out / 300.
    Each image is inferred alone so two LIBERO views do not mix.
    """

    hf_id = "depth-anything/da3metric-large"
    focal_norm = 300.0

    def __init__(self, model_id=None):
        super().__init__()
        self.model = DepthAnything3.from_pretrained(model_id or self.hf_id)

    @staticmethod
    def scale_intrinsics(K, src_hw, dst_hw):
        import numpy as np

        K = np.asarray(K, np.float64).copy()
        if K.ndim == 2:
            K = K[None]
        sh, sw = src_hw
        dh, dw = dst_hw
        K[..., 0, :] *= dw / sw
        K[..., 1, :] *= dh / sh
        return K

    @classmethod
    def focal_to_metric(cls, depth, K):
        import numpy as np

        depth = np.asarray(depth, np.float64)
        K = np.asarray(K, np.float64)
        if depth.ndim == 2:
            depth = depth[None]
        f = 0.5 * (K[..., 0, 0] + K[..., 1, 1])
        return depth * (f.reshape(-1, 1, 1) / cls.focal_norm)

    def infer_metric(self, images, K, src_hw=None, process_res=504):
        import numpy as np

        K = np.asarray(K, np.float64)
        if K.ndim == 2:
            K = K[None]
        if src_hw is None:
            src_hw = np.asarray(images[0]).shape[:2]
        maps = []
        for i, image in enumerate(images):
            pred = self.model.inference([image], process_res=process_res)
            ph, pw = pred.depth.shape[-2:]
            Kp = self.scale_intrinsics(K[i], src_hw, (ph, pw))
            maps.append(self.focal_to_metric(pred.depth[0], Kp)[0])
        return np.stack(maps).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    model = DAV3Module().to("cuda")
    example_path = os.environ.get("DA3_EXAMPLE_IMAGE_DIR", "./assets/examples/SOH/")
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    prediction = model.inference_images(images)

    depth = prediction.depth
    d = depth[0]
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

    plt.figure(figsize=(6, 6))
    plt.imshow(d_norm, cmap="magma")
    plt.colorbar(label="normalized depth")
    plt.title("Depth map")
    plt.axis("off")
    plt.show()

"""A vs B geometry probe.

A: pretrained da3-base. B: same heads, Evo checkpoint backbone only.
Simulator intrinsics and extrinsics are the comparison target and the lift
for the point cloud. They are not passed into the network.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BB_PREFIX = "embedder.dpt_embedder.model.model.backbone."
CAMS = ("agentview", "robot0_eye_in_hand")


class GeomProbe:
    process_res = 504

    def dump(self, suite, category, n, json_path, out, resolution=256, seed=0):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import pathlib

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)[suite]
        cats = [c.strip().lower() for c in category.split(",") if c.strip()]
        ids = [
            t["id"] - 1
            for t in rows
            if not cats or any(c in t["category"].lower() for c in cats)
        ][:n]
        if not ids:
            raise SystemExit(f"no tasks for {suite}/{category}")

        suite_obj = benchmark.get_benchmark_dict()[suite]()
        images, depths, intrinsics, extrinsics, task_ids = [], [], [], [], []
        for task_id in ids:
            task = suite_obj.get_task(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=resolution,
                camera_widths=resolution,
                camera_depths=True,
            )
            env.seed(seed)
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            frame_i, frame_d, frame_k, frame_e = [], [], [], []
            for cam in CAMS:
                img = np.ascontiguousarray(obs[f"{cam}_image"])
                if img.shape[0] != img.shape[1]:
                    raise SystemExit(f"{cam} image is not square: {img.shape}")
                dep = np.asarray(obs[f"{cam}_depth"])
                if dep.ndim == 3:
                    dep = dep[..., 0]
                frame_i.append(img)
                frame_d.append(get_real_depth_map(env.sim, dep).astype(np.float32))
                frame_k.append(get_camera_intrinsic_matrix(env.sim, cam, resolution, resolution))
                frame_e.append(get_camera_extrinsic_matrix(env.sim, cam))
            images.append(np.stack(frame_i))
            depths.append(np.stack(frame_d))
            intrinsics.append(np.stack(frame_k))
            extrinsics.append(np.stack(frame_e))
            task_ids.append(task_id)
            env.close()

        np.savez_compressed(
            out,
            images=np.stack(images),
            depths=np.stack(depths),
            intrinsics=np.stack(intrinsics).astype(np.float32),
            extrinsics=np.stack(extrinsics).astype(np.float32),
            task_ids=np.asarray(task_ids, np.int32),
        )
        print(f"wrote {out} tasks={task_ids}")

    def eval(self, npz_path, ckpt, out):
        sys.path.insert(0, ROOT)
        import cv2
        import torch
        from model.depth_anything_3.api import DepthAnything3

        data = np.load(npz_path)
        A = DepthAnything3.from_pretrained("depth-anything/da3-base").cuda().eval()
        B = DepthAnything3.from_pretrained("depth-anything/da3-base").cuda().eval()
        n_loaded = self._load_backbone(B, ckpt)
        print(f"B backbone tensors loaded: {n_loaded}")

        rows = {"A": [], "B": []}
        for i in range(len(data["images"])):
            pair = [data["images"][i, 0], data["images"][i, 1]]
            gt_d = data["depths"][i]
            gt_k = data["intrinsics"][i]
            gt_e = data["extrinsics"][i]
            for name, model in (("A", A), ("B", B)):
                pred = model.inference(pair, process_res=self.process_res)
                ph, pw = pred.depth.shape[-2:]
                oh, ow = gt_d.shape[-2:]
                # ponytail: square dump, so one resize matches upper_bound_resize onto 504
                dmetrics = [
                    self.depth_metrics(
                        pred.depth[v],
                        cv2.resize(gt_d[v], (pw, ph), interpolation=cv2.INTER_NEAREST),
                    )
                    for v in range(2)
                ]
                cmetrics = self.camera_metrics(pred.intrinsics, pred.extrinsics, gt_k, gt_e, (ph, pw), (oh, ow))
                rows[name].append(self._merge(dmetrics, cmetrics))
        summary = {name: self._mean_rows(rows[name]) for name in ("A", "B")}
        summary["B_minus_A"] = {k: summary["B"][k] - summary["A"][k] for k in summary["A"]}
        print(json.dumps(summary, indent=2))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out}")

    def _load_backbone(self, model, ckpt):
        import torch

        path = ckpt
        if os.path.isdir(ckpt):
            path = os.path.join(ckpt, "mp_rank_00_model_states.pt")
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        sd = blob["module"] if isinstance(blob, dict) and "module" in blob else blob
        bb = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            if k.startswith(BB_PREFIX):
                bb[k[len(BB_PREFIX) :]] = v
        if not bb:
            sample = next(iter(sd))
            raise SystemExit(f"no backbone keys under {BB_PREFIX}; example key: {sample}")
        missing, unexpected = model.model.backbone.load_state_dict(bb, strict=True)
        if missing or unexpected:
            raise SystemExit(f"backbone mismatch missing={missing} unexpected={unexpected}")
        return len(bb)

    @staticmethod
    def depth_metrics(pred, gt):
        pred = np.asarray(pred, np.float64)
        gt = np.asarray(gt, np.float64)
        m = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-3)
        if m.sum() < 10:
            raise SystemExit("too few valid depth pixels")
        scale = np.median(gt[m]) / np.median(pred[m])
        aligned = pred * scale
        ratio = np.maximum(aligned[m] / gt[m], gt[m] / aligned[m])
        return {
            "absrel": float(np.mean(np.abs(aligned[m] - gt[m]) / gt[m])),
            "delta1": float(np.mean(ratio < 1.25)),
            "scale": float(scale),
        }

    @staticmethod
    def camera_metrics(pred_k, pred_e, gt_k, gt_e, proc_hw, orig_hw):
        ph, pw = proc_hw
        _, ow = orig_hw
        sx = pw / ow
        fx = []
        for v in range(2):
            fx.append(abs(pred_k[v, 0, 0] - gt_k[v, 0, 0] * sx) / (abs(gt_k[v, 0, 0] * sx) + 1e-8))
        rel_p = GeomProbe._relative(GeomProbe._as_h44(pred_e))
        rel_g = GeomProbe._relative(GeomProbe._as_h44(gt_e))
        return {
            "fx_err": float(np.mean(fx)),
            "rot_deg": GeomProbe._rot_deg(rel_p[:3, :3], rel_g[:3, :3]),
            "trans_deg": GeomProbe._trans_deg(rel_p[:3, 3], rel_g[:3, 3]),
        }

    @staticmethod
    def _as_h44(ext):
        ext = np.asarray(ext, np.float64)
        if ext.shape[-2:] == (4, 4):
            return ext
        if ext.shape[-2:] != (3, 4):
            raise SystemExit(f"extrinsics shape {ext.shape}")
        out = np.repeat(np.eye(4)[None], ext.shape[0], axis=0)
        out[:, :3, :] = ext
        return out

    @staticmethod
    def _relative(w2c):
        return w2c[1] @ np.linalg.inv(w2c[0])

    @staticmethod
    def _rot_deg(ra, rb):
        c = (np.trace(ra.T @ rb) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    @staticmethod
    def _trans_deg(ta, tb):
        na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
        if na < 1e-8 or nb < 1e-8:
            raise SystemExit("relative translation is degenerate")
        c = np.dot(ta / na, tb / nb)
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    @staticmethod
    def _merge(dmetrics, cmetrics):
        row = {k: float(np.mean([d[k] for d in dmetrics])) for k in ("absrel", "delta1")}
        row.update(cmetrics)
        return row

    @staticmethod
    def _mean_rows(rows):
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    def selfcheck(self):
        gt = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        m = self.depth_metrics(gt / 2, gt)
        assert m["absrel"] < 1e-6 and abs(m["delta1"] - 1) < 1e-6 and abs(m["scale"] - 2) < 1e-6

        k = np.eye(3)
        k[0, 0] = k[1, 1] = 100.0
        kp = k.copy()
        kp[0, 0] = kp[1, 1] = 100.0 * 504 / 256
        e0 = np.eye(4)
        e1 = np.eye(4)
        e1[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float64)
        e1[:3, 3] = [1, 0, 0]
        gt_e = np.stack([e0, e1])
        same = self.camera_metrics(np.stack([kp, kp]), gt_e, np.stack([k, k]), gt_e, (504, 504), (256, 256))
        assert same["fx_err"] < 1e-6 and same["rot_deg"] < 1e-4 and same["trans_deg"] < 1e-3

        bad = gt_e.copy()
        bad[1, :3, 3] = [0, 1, 0]
        moved = self.camera_metrics(np.stack([kp, kp]), bad, np.stack([k, k]), gt_e, (504, 504), (256, 256))
        assert abs(moved["trans_deg"] - 90) < 1e-3
        print("selfcheck ok")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--suite", default="libero_spatial")
    d.add_argument("--category", default="camera")
    d.add_argument("--n", type=int, default=8)
    d.add_argument("--json", dest="json_path", default="./LIBERO-plus/libero/libero/benchmark/task_classification.json")
    d.add_argument("--out", default="geom_probe.npz")
    d.add_argument("--resolution", type=int, default=256)
    d.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--npz", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", default="geom_ab.json")
    sub.add_parser("selfcheck")
    args = p.parse_args()
    probe = GeomProbe()
    if args.cmd == "dump":
        probe.dump(args.suite, args.category, args.n, args.json_path, args.out, args.resolution, args.seed)
    elif args.cmd == "eval":
        probe.eval(args.npz, args.ckpt, args.out)
    else:
        probe.selfcheck()


if __name__ == "__main__":
    main()

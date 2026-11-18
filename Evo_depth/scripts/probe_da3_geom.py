"""A vs B geometry probe.

A: pretrained da3-base. B: same heads, Evo checkpoint backbone only.
Network never sees simulator cameras (no cam_enc token).
Simulator K/E are only used after inference to Sim(3)-scale relative depth to metres.
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

        modes = ("relative_median", "pose_scale_metric")
        rows = {mode: {"A": [], "B": []} for mode in modes}
        pred_depth = {mode: {"A": [], "B": []} for mode in modes}
        gt_depth = []
        for i in range(len(data["images"])):
            pair = [data["images"][i, 0], data["images"][i, 1]]
            gt_d = data["depths"][i]
            gt_k = np.asarray(data["intrinsics"][i], np.float32)
            w2c = self._c2w_to_w2c(data["extrinsics"][i])
            gt_resized = None
            for name, model in (("A", A), ("B", B)):
                pred = model.inference(pair, process_res=self.process_res)
                ph, pw = pred.depth.shape[-2:]
                oh, ow = gt_d.shape[-2:]
                if gt_resized is None:
                    gt_resized = np.stack(
                        [cv2.resize(gt_d[v], (pw, ph), interpolation=cv2.INTER_NEAREST) for v in range(2)]
                    )
                    gt_depth.append(gt_resized)
                metric_depth, pose_s = self._relative_to_metric(pred.depth, pred.extrinsics, w2c)
                cmetrics = self.camera_metrics(
                    pred.intrinsics, pred.extrinsics, gt_k, w2c, (ph, pw), (oh, ow)
                )
                packed = (
                    ("relative_median", pred.depth, "median"),
                    ("pose_scale_metric", metric_depth, "none"),
                )
                for mode, depth, align in packed:
                    dmetrics = []
                    maps = []
                    for v in range(2):
                        m = self.depth_metrics(depth[v], gt_resized[v], align=align)
                        m["pose_scale"] = pose_s
                        dmetrics.append(m)
                        maps.append(np.asarray(depth[v], np.float32) * np.float32(m["scale"]))
                    pred_depth[mode][name].append(np.stack(maps))
                    rows[mode][name].append(self._merge(dmetrics, cmetrics))
        summary = {}
        for mode in modes:
            summary[mode] = {name: self._mean_rows(rows[mode][name]) for name in ("A", "B")}
            summary[mode]["B_minus_A"] = {
                k: summary[mode]["B"][k] - summary[mode]["A"][k] for k in summary[mode]["A"]
            }
            summary[mode]["frames"] = rows[mode]
        print(self._format_table(summary["relative_median"], "relative_median: no cam token; median vs GT"))
        print()
        print(
            self._format_table(
                summary["pose_scale_metric"],
                "pose_scale_metric: still no cam token; Umeyama vs simulator E then metres vs GT",
            )
        )
        slim = {mode: {k: summary[mode][k] for k in ("A", "B", "B_minus_A")} for mode in modes}
        print(json.dumps(slim, indent=2))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        tensors = {
            "gt": np.stack(gt_depth).astype(np.float32),
            "relative_A": np.stack(pred_depth["relative_median"]["A"]).astype(np.float32),
            "relative_B": np.stack(pred_depth["relative_median"]["B"]).astype(np.float32),
            "metric_A": np.stack(pred_depth["pose_scale_metric"]["A"]).astype(np.float32),
            "metric_B": np.stack(pred_depth["pose_scale_metric"]["B"]).astype(np.float32),
            "task_ids": np.asarray(data["task_ids"], np.int32),
        }
        stem = os.path.splitext(out)[0]
        depth_out = stem + "_depth.npz"
        depth_pt = stem + "_depth.pt"
        tensor_dir = stem + "_depth_tensors"
        np.savez_compressed(depth_out, **tensors)
        os.makedirs(tensor_dir, exist_ok=True)
        torch.save({k: torch.from_numpy(v.copy()) for k, v in tensors.items()}, depth_pt)
        print("depth tensors (N, 2, H, W) metres except task_ids")
        for key, arr in tensors.items():
            np.save(os.path.join(tensor_dir, f"{key}.npy"), arr)
            if key == "task_ids":
                print(f"  {key:12} {arr.shape} {arr}")
                continue
            print(
                f"  {key:12} {tuple(arr.shape)} {arr.dtype}  "
                f"min {arr.min():.4f}  med {np.median(arr):.4f}  max {arr.max():.4f}"
            )
        vis_dir = stem + "_depth_vis"
        self._write_depth_vis(
            vis_dir,
            data["images"],
            data["task_ids"],
            gt_depth,
            pred_depth,
        )
        print(f"wrote {out}")
        print(f"wrote {depth_out}")
        print(f"wrote {depth_pt}")
        print(f"wrote npy tensors under {tensor_dir}")
        print(f"wrote depth maps under {vis_dir}")

    @staticmethod
    def _colorize_depth(depth, vmin, vmax):
        import cv2

        x = np.clip((np.asarray(depth, np.float32) - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    @staticmethod
    def _label(img, text):
        import cv2

        out = img.copy()
        cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _write_depth_vis(self, vis_dir, images, task_ids, gt_depth, pred_depth):
        import cv2

        os.makedirs(vis_dir, exist_ok=True)
        rel_a = pred_depth["relative_median"]["A"]
        rel_b = pred_depth["relative_median"]["B"]
        met_a = pred_depth["pose_scale_metric"]["A"]
        met_b = pred_depth["pose_scale_metric"]["B"]
        names = ("rgb", "gt_m", "A_rel_m", "B_rel_m", "A_pose_m", "B_pose_m")
        for i, tid in enumerate(task_ids):
            rows = []
            for v, cam in enumerate(("agent", "wrist")):
                gt = gt_depth[i][v]
                vmin, vmax = np.percentile(gt, 2), np.percentile(gt, 98)
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                rgb = cv2.resize(np.ascontiguousarray(images[i, v]), (gt.shape[1], gt.shape[0]))
                if rgb.ndim == 2:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
                else:
                    rgb = rgb[..., ::-1].copy()
                cells = [
                    rgb,
                    self._colorize_depth(gt, vmin, vmax),
                    self._colorize_depth(rel_a[i][v], vmin, vmax),
                    self._colorize_depth(rel_b[i][v], vmin, vmax),
                    self._colorize_depth(met_a[i][v], vmin, vmax),
                    self._colorize_depth(met_b[i][v], vmin, vmax),
                ]
                cells = [self._label(c, f"{cam} {n}") for c, n in zip(cells, names)]
                rows.append(np.concatenate(cells, axis=1))
            panel = np.concatenate(rows, axis=0)
            path = os.path.join(vis_dir, f"task_{int(tid):04d}.png")
            cv2.imwrite(path, panel)
            print(f"depth map {path}")

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
    def _c2w_to_w2c(ext):
        return np.linalg.inv(GeomProbe._as_h44(ext)).astype(np.float32)

    @staticmethod
    def _camera_centers(w2c):
        c2w = np.linalg.inv(GeomProbe._as_h44(w2c))
        return c2w[:, :3, 3]

    @staticmethod
    def _relative_to_metric(pred_depth, pred_ext, gt_w2c):
        # Two views: Umeyama Sim(3) is rank-deficient. Scale by baseline length.
        pred_c = GeomProbe._camera_centers(pred_ext)
        gt_c = GeomProbe._camera_centers(gt_w2c)
        pred_base = float(np.linalg.norm(pred_c[1] - pred_c[0]))
        gt_base = float(np.linalg.norm(gt_c[1] - gt_c[0]))
        if pred_base < 1e-8 or gt_base < 1e-8:
            print("warn: degenerate two-view baseline, skip pose scale (s=1)")
            return np.asarray(pred_depth, np.float64), 1.0
        scale = pred_base / gt_base
        return np.asarray(pred_depth, np.float64) / scale, scale

    @staticmethod
    def depth_metrics(pred, gt, align="median"):
        pred = np.asarray(pred, np.float64)
        gt = np.asarray(gt, np.float64)
        m = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-3)
        if m.sum() < 10:
            raise SystemExit("too few valid depth pixels")
        median_factor = float(np.median(gt[m]) / np.median(pred[m]))
        if align == "median":
            scale = median_factor
        elif align == "none":
            scale = 1.0
        else:
            raise SystemExit(f"unknown depth align {align}")
        aligned = pred * scale
        ratio = np.maximum(aligned[m] / gt[m], gt[m] / aligned[m])
        err = aligned[m] - gt[m]
        return {
            "absrel": float(np.mean(np.abs(err) / gt[m])),
            "delta1": float(np.mean(ratio < 1.25)),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "scale": float(scale),
            "median_factor": median_factor,
        }

    @staticmethod
    def camera_metrics(pred_k, pred_e, gt_k, gt_e, proc_hw, orig_hw):
        ph, pw = proc_hw
        _, ow = orig_hw
        sx = pw / ow
        out = {}
        rel_err = []
        for v, cam in enumerate(("agent", "wrist")):
            pred_fx = float(pred_k[v, 0, 0])
            gt_fx = float(gt_k[v, 0, 0] * sx)
            out[f"pred_fx_{cam}"] = pred_fx
            out[f"gt_fx_{cam}"] = gt_fx
            rel_err.append(abs(pred_fx - gt_fx) / (abs(gt_fx) + 1e-8))
        rel_p = GeomProbe._relative(GeomProbe._as_h44(pred_e))
        rel_g = GeomProbe._relative(GeomProbe._as_h44(gt_e))
        out["fx_err"] = float(np.mean(rel_err))
        out["rot_deg"] = GeomProbe._rot_deg(rel_p[:3, :3], rel_g[:3, :3])
        out["trans_deg"] = GeomProbe._trans_deg(rel_p[:3, 3], rel_g[:3, 3])
        return out

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
        keys = ("absrel", "delta1", "rmse", "scale", "median_factor", "pose_scale")
        row = {k: float(np.mean([d[k] for d in dmetrics])) for k in keys}
        for cam, d in zip(("agent", "wrist"), dmetrics):
            for k in keys:
                row[f"{cam}_{k}"] = float(d[k])
        row.update(cmetrics)
        return row

    @staticmethod
    def _format_table(summary, title):
        a, b = summary["A"], summary["B"]
        lines = [
            title,
            f"{'':12} {'absrel':>8} {'delta1':>8} {'rmse_m':>8} {'med_s':>8}",
        ]
        for tag, row in (("A mean", a), ("A agent", a), ("A wrist", a), ("B mean", b), ("B agent", b), ("B wrist", b)):
            if "mean" in tag:
                keys = ("absrel", "delta1", "rmse", "median_factor")
            elif "agent" in tag:
                keys = ("agent_absrel", "agent_delta1", "agent_rmse", "agent_median_factor")
            else:
                keys = ("wrist_absrel", "wrist_delta1", "wrist_rmse", "wrist_median_factor")
            lines.append(
                f"{tag:12} {row[keys[0]]:8.3f} {row[keys[1]]:8.3f} {row[keys[2]]:8.3f} {row[keys[3]]:8.3f}"
            )
        lines += [
            "",
            "focal length px at 504 (GT is 256 fx * 504/256)",
            f"{'':12} {'gt':>8} {'A':>8} {'B':>8}",
            f"{'agent':12} {a['gt_fx_agent']:8.1f} {a['pred_fx_agent']:8.1f} {b['pred_fx_agent']:8.1f}",
            f"{'wrist':12} {a['gt_fx_wrist']:8.1f} {a['pred_fx_wrist']:8.1f} {b['pred_fx_wrist']:8.1f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _mean_rows(rows):
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    def selfcheck(self):
        gt = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        m = self.depth_metrics(gt / 2, gt)
        assert m["absrel"] < 1e-6 and abs(m["delta1"] - 1) < 1e-6 and abs(m["scale"] - 2) < 1e-6
        assert m["rmse"] < 1e-6
        raw = self.depth_metrics(gt, gt, align="none")
        assert raw["absrel"] < 1e-6 and abs(raw["scale"] - 1) < 1e-6

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
    d.add_argument("--out", default="geom_out/geom_probe.npz")
    d.add_argument("--resolution", type=int, default=256)
    d.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--npz", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", default="geom_out/geom_ab.json")
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

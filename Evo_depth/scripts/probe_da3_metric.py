"""DA3METRIC-Large vs simulator depth.

Each view is inferred alone. Simulator K is used only after inference (f/300).
Reuse dump from probe_da3_geom.py (geom_probe.npz).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MetricProbe:
    process_res = 504

    def dump(self, *args, **kwargs):
        sys.path.insert(0, os.path.dirname(__file__))
        from probe_da3_geom import GeomProbe

        return GeomProbe().dump(*args, **kwargs)

    def eval(self, npz_path, out):
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.dirname(__file__))
        import cv2
        import torch
        from probe_da3_geom import GeomProbe
        from model.depth_anything_3.depth_anything_v3 import DAV3MetricModule

        data = np.load(npz_path)
        model = DAV3MetricModule().cuda().eval()
        rows = []
        pred_maps = []
        gt_depth = []
        for i in range(len(data["images"])):
            gt_d = data["depths"][i]
            gt_k = np.asarray(data["intrinsics"][i], np.float32)
            oh, ow = gt_d.shape[-2:]
            views = [data["images"][i, v] for v in range(2)]
            depth_m = model.infer_metric(views, gt_k, src_hw=(oh, ow), process_res=self.process_res)
            ph, pw = depth_m.shape[-2:]
            gt_resized = np.stack(
                [cv2.resize(gt_d[v], (pw, ph), interpolation=cv2.INTER_NEAREST) for v in range(2)]
            )
            gt_depth.append(gt_resized)
            dmetrics = [
                GeomProbe.depth_metrics(depth_m[v], gt_resized[v], align="none") for v in range(2)
            ]
            sx = pw / ow
            row = {
                "gt_fx_agent": float(gt_k[0, 0, 0] * sx),
                "gt_fx_wrist": float(gt_k[1, 0, 0] * sx),
            }
            keys = ("absrel", "delta1", "rmse", "scale", "median_factor")
            for k in keys:
                row[k] = float(np.mean([d[k] for d in dmetrics]))
            for cam, d in zip(("agent", "wrist"), dmetrics):
                for k in keys:
                    row[f"{cam}_{k}"] = float(d[k])
            rows.append(row)
            pred_maps.append(np.asarray(depth_m, np.float32))

        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(self._format_table(mean, len(rows)))
        summary = {"metric_vs_sim": mean, "frames": rows}
        print(json.dumps({"metric_vs_sim": mean}, indent=2))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        tensors = {
            "gt": np.stack(gt_depth).astype(np.float32),
            "metric": np.stack(pred_maps).astype(np.float32),
            "task_ids": np.asarray(data["task_ids"], np.int32),
        }
        stem = os.path.splitext(out)[0]
        depth_out = stem + "_depth.npz"
        depth_pt = stem + "_depth.pt"
        tensor_dir = stem + "_depth_tensors"
        vis_dir = stem + "_depth_vis"
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
        self._write_vis(vis_dir, data["images"], data["task_ids"], gt_depth, pred_maps)
        print(f"wrote {out}")
        print(f"wrote {depth_out}")
        print(f"wrote {depth_pt}")
        print(f"wrote npy tensors under {tensor_dir}")
        print(f"wrote depth maps under {vis_dir}")

    @staticmethod
    def _format_table(mean, n):
        lines = [
            "DA3METRIC-Large vs simulator depth (metres, no median align)",
            "each view inferred alone; Z = f*out/300 with that view's K",
            f"{'':12} {'absrel':>8} {'delta1':>8} {'rmse_m':>8} {'med_s':>8}",
        ]
        packs = (
            ("mean", "absrel", "delta1", "rmse", "median_factor"),
            ("agent", "agent_absrel", "agent_delta1", "agent_rmse", "agent_median_factor"),
            ("wrist", "wrist_absrel", "wrist_delta1", "wrist_rmse", "wrist_median_factor"),
        )
        for tag, *keys in packs:
            lines.append(
                f"{tag:12} {mean[keys[0]]:8.3f} {mean[keys[1]]:8.3f} "
                f"{mean[keys[2]]:8.3f} {mean[keys[3]]:8.3f}"
            )
        lines += [
            "",
            "focal px at processed res (dump fx * W_pred/W_dump)",
            f"{'agent':12} {mean['gt_fx_agent']:8.1f}",
            f"{'wrist':12} {mean['gt_fx_wrist']:8.1f}",
            f"frames {n}",
        ]
        return "\n".join(lines)

    def _write_vis(self, vis_dir, images, task_ids, gt_depth, pred_maps):
        sys.path.insert(0, os.path.dirname(__file__))
        import cv2
        from probe_da3_geom import GeomProbe

        os.makedirs(vis_dir, exist_ok=True)
        names = ("rgb", "sim_m", "metric_m")
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
                    GeomProbe._colorize_depth(gt, vmin, vmax),
                    GeomProbe._colorize_depth(pred_maps[i][v], vmin, vmax),
                ]
                cells = [GeomProbe._label(c, f"{cam} {n}") for c, n in zip(cells, names)]
                rows.append(np.concatenate(cells, axis=1))
            path = os.path.join(vis_dir, f"task_{int(tid):04d}.png")
            cv2.imwrite(path, np.concatenate(rows, axis=0))
            print(f"depth map {path}")

    def selfcheck(self):
        sys.path.insert(0, ROOT)
        sys.path.insert(0, os.path.dirname(__file__))
        from model.depth_anything_3.depth_anything_v3 import DAV3MetricModule
        from probe_da3_geom import GeomProbe

        k = np.eye(3, dtype=np.float64)
        k[0, 0] = k[1, 1] = 100.0
        kp = DAV3MetricModule.scale_intrinsics(k, (256, 256), (504, 504))
        assert abs(kp[0, 0, 0] - 100.0 * 504 / 256) < 1e-6
        z = DAV3MetricModule.focal_to_metric(np.ones((1, 2, 2)), kp)
        assert abs(float(z[0, 0, 0]) - (100.0 * 504 / 256) / 300.0) < 1e-6
        gt = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        raw = GeomProbe.depth_metrics(gt, gt, align="none")
        assert raw["absrel"] < 1e-6
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
    e.add_argument("--out", default="geom_metric.json")
    sub.add_parser("selfcheck")
    args = p.parse_args()
    probe = MetricProbe()
    if args.cmd == "dump":
        probe.dump(args.suite, args.category, args.n, args.json_path, args.out, args.resolution, args.seed)
    elif args.cmd == "eval":
        probe.eval(args.npz, args.out)
    else:
        probe.selfcheck()


if __name__ == "__main__":
    main()

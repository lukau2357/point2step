import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from point2step.inr_fitting import (
    INR_MAX_STEPS, INR_NETWORK_PARAMETERS, INRNetwork, inr_fit_kwargs,
)
from point2step.occ_surfaces import inr_to_occ
from point2step.surface_types import SURFACE_INR

import mesh_pipeline


def _parse_cid(inr_path):
    base = os.path.basename(inr_path)
    return int(base.replace("surface_inr_", "").replace(".pt", ""))


def _payload_from_fit(result):
    model = result["params"]["model"]
    return {
        "error": float(result["error"]),
        "network_parameters": result["params"]["network_parameters"],
        "is_u_closed": bool(model.is_u_closed),
        "is_v_closed": bool(model.is_v_closed),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cluster_mean": np.asarray(result["params"]["cluster_mean"]),
        "cluster_scale": float(result["params"]["cluster_scale"]),
        "uv_bb_min": np.asarray(result["params"]["uv_bb_min"]),
        "uv_bb_max": np.asarray(result["params"]["uv_bb_max"]),
    }


def fit_freeform_segments(xyzc_path, device, seed=41,
                          max_steps=INR_MAX_STEPS,
                          uv_resolution=mesh_pipeline.DEFAULT_UV_RESOLUTION,
                          threshold_multiplier=mesh_pipeline.DEFAULT_THRESHOLD_MULTIPLIER,
                          spacing_percentile=100.0,
                          inr_uv_margin=mesh_pipeline.DEFAULT_INR_UV_MARGIN):
    from point2step.mesh_clipping import build_cluster_trees
    from point2step.primitive_fitting_utils import rotation_matrix_a_to_b
    from point2step.surface_fitter import MIN_CLUSTER_PTS, fit_surface

    np_rng = np.random.default_rng(seed)
    tm = threshold_multiplier
    abs_th = mesh_pipeline.DEFAULT_ABSOLUTE_THRESHOLD
    R = uv_resolution

    data = np.loadtxt(xyzc_path)
    pts = data[:, :3]
    mean = pts.mean(axis=0)
    centered = pts - mean
    S, U = np.linalg.eigh(centered.T @ centered)
    rot = rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
    rotated = (rot @ centered.T).T
    scale = float((rotated.max(axis=0) - rotated.min(axis=0)).max()) + 1e-7
    data[:, :3] = (rotated / scale).astype(np.float32)

    unique_cids = np.unique(data[:, -1].astype(int))
    segments, cids = [], []
    for cid in unique_cids:
        seg = data[data[:, -1] == cid][:, :3].astype(np.float32)
        if len(seg) < MIN_CLUSTER_PTS:
            continue
        segments.append(seg)
        cids.append(int(cid))

    if not segments:
        return []

    _, spacings = build_cluster_trees(segments, spacing_percentile)

    out = []
    for idx, cid in enumerate(cids):
        segment = segments[idx]
        res = fit_surface(
            segment,
            INR_NETWORK_PARAMETERS,
            np_rng, device,
            inr_fit_kwargs=inr_fit_kwargs(max_steps, seed=seed),
            plane_mesh_kwargs={"mesh_dim": R, "threshold_multiplier": tm,
                               "plane_sampling_deviation": 2, "spacing": spacings[idx],
                               "absolute_threshold": abs_th},
            sphere_mesh_kwargs={"dim_theta": R, "dim_lambda": R,
                                "threshold_multiplier": tm, "spacing": spacings[idx],
                                "absolute_threshold": abs_th},
            cylinder_mesh_kwargs={"dim_theta": R, "dim_height": R // 2,
                                  "threshold_multiplier": tm, "cylinder_height_margin": 0.5,
                                  "spacing": spacings[idx], "absolute_threshold": abs_th},
            cone_mesh_kwargs={"dim_theta": R, "dim_height": R,
                              "threshold_multiplier": tm, "cone_height_margin": 0.5,
                              "spacing": spacings[idx], "absolute_threshold": abs_th},
            inr_mesh_kwargs={"mesh_dim": R, "uv_margin": inr_uv_margin,
                             "threshold_multiplier": tm, "spacing": spacings[idx]},
            radius_inflation=mesh_pipeline.DEFAULT_RADIUS_INFLATION,
        )
        if res["surface_id"] != SURFACE_INR:
            continue
        out.append((cid, segment, _payload_from_fit(res["result"])))

    return out


def _sample_inr_grid(model, payload, K, device):
    cm = np.asarray(payload["cluster_mean"])
    cs = float(payload["cluster_scale"])
    uv_bb_min = np.asarray(payload["uv_bb_min"])
    uv_bb_max = np.asarray(payload["uv_bb_max"])
    pts = model.sample_points(
        K, uv_bb_min.copy(), uv_bb_max.copy(), cm, cs, uv_margin=0,
    ).cpu().numpy().astype(np.float64)
    return pts


def _sample_bspline_grid(bspline, K):
    u1 = bspline.UKnot(1)
    u2 = bspline.UKnot(bspline.NbUKnots())
    v1 = bspline.VKnot(1)
    v2 = bspline.VKnot(bspline.NbVKnots())
    us = np.linspace(u1, u2, K)
    vs = np.linspace(v1, v2, K)
    pts = np.empty((K * K, 3), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            p = bspline.Value(float(u), float(v))
            pts[i * K + j] = (p.X(), p.Y(), p.Z())
    return pts


def _residual_mean_cluster_to_surface(surface_pts, cluster):
    tree = cKDTree(surface_pts)
    d, _ = tree.query(cluster, k=1)
    return float(d.mean())


def _eval_inr_cluster(cluster, payload, device, K_eval=100, K_fit=50):
    cluster = cluster.astype(np.float64)
    N_cluster = len(cluster)

    model = INRNetwork(
        **payload["network_parameters"],
        is_u_closed=bool(payload["is_u_closed"]),
        is_v_closed=bool(payload["is_v_closed"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    inr_pts = _sample_inr_grid(model, payload, K_eval, device)
    inr_residual_mean = _residual_mean_cluster_to_surface(inr_pts, cluster)

    record = {
        "n_cluster_points":          int(N_cluster),
        "K_eval":                    int(K_eval),
        "K_fit":                     int(K_fit),
        "inr_residual_mean":         inr_residual_mean,
        "bspline_residual_mean":     None,
        "bspline_conversion_failed": False,
        "inr_full_cluster_error":    float(payload.get("error", float("nan"))),
    }

    try:
        bspline = inr_to_occ({
            "model":         model,
            "cluster_mean":  np.asarray(payload["cluster_mean"]),
            "cluster_scale": float(payload["cluster_scale"]),
            "uv_bb_min":     np.asarray(payload["uv_bb_min"]),
            "uv_bb_max":     np.asarray(payload["uv_bb_max"]),
        }, grid_resolution=K_fit)
    except Exception as e:
        record["bspline_conversion_failed"] = True
        record["bspline_conversion_error"]  = repr(e)
        return record

    bspline_pts = _sample_bspline_grid(bspline, K_eval)
    record["bspline_residual_mean"] = _residual_mean_cluster_to_surface(
        bspline_pts, cluster)
    return record


def _eval_from_scratch(xyzc_path, device, K_eval, K_fit, max_steps=INR_MAX_STEPS):
    records = []
    for cid, segment, payload in fit_freeform_segments(xyzc_path, device,
                                                       max_steps=max_steps):
        record = _eval_inr_cluster(segment, payload, device,
                                   K_eval=K_eval, K_fit=K_fit)
        record["cid"] = cid
        record["from_cache"] = False
        records.append(record)
        bs = record["bspline_residual_mean"]
        bs_str = f"{bs:.6f}" if bs is not None else "FAIL"
        print(f"  {os.path.basename(xyzc_path)}/cid={cid} (refit): "
              f"inr_residual_mean={record['inr_residual_mean']:.6f}  "
              f"bspline_residual_mean={bs_str}", flush=True)
    return records


def _eval_part_dir(src_part_dir, device, K_eval, K_fit=50, xyzc_path=None,
                   max_steps=INR_MAX_STEPS):
    inr_paths = sorted(glob.glob(os.path.join(src_part_dir, "surface_inr_*.pt")),
                       key=_parse_cid)
    if not inr_paths:
        if xyzc_path is not None and os.path.isfile(xyzc_path):
            print(f"  [refit] no cached INR under {src_part_dir}; "
                  f"fitting from {xyzc_path}", flush=True)
            return _eval_from_scratch(xyzc_path, device, K_eval, K_fit, max_steps)
        return []

    metadata_cids = None
    meta_path = os.path.join(src_part_dir, "metadata.npz")
    if os.path.isfile(meta_path):
        try:
            meta = np.load(meta_path, allow_pickle=True)
            if "cluster_ids" in meta:
                metadata_cids = set(int(c) for c in meta["cluster_ids"])
        except Exception:
            pass

    clusters = []
    for inr_path in inr_paths:
        cid = _parse_cid(inr_path)
        if metadata_cids is not None and cid not in metadata_cids:
            print(f"  [warn] {src_part_dir}: cid {cid} not in metadata cluster_ids — skipping",
                  flush=True)
            continue

        cluster_path = os.path.join(src_part_dir, f"cluster_{cid}.npy")
        if not os.path.isfile(cluster_path):
            print(f"  [warn] {src_part_dir}: cluster_{cid}.npy missing — skipping cid {cid}",
                  flush=True)
            continue

        cluster = np.load(cluster_path).astype(np.float32)

        try:
            payload = torch.load(inr_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  [warn] {src_part_dir}: failed to load {inr_path}: {e!r}",
                  flush=True)
            continue

        record = _eval_inr_cluster(cluster, payload, device,
                                   K_eval=K_eval, K_fit=K_fit)
        record["cid"] = cid
        clusters.append(record)

        bs = record["bspline_residual_mean"]
        bs_str = f"{bs:.6f}" if bs is not None else "FAIL"
        print(f"  {os.path.basename(src_part_dir)}/cid={cid}: "
              f"inr_residual_mean={record['inr_residual_mean']:.6f}  "
              f"bspline_residual_mean={bs_str}  "
              f"(n_cluster_pts={record['n_cluster_points']})",
              flush=True)

    return clusters


def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc per-segment INR vs B-Spline residual_mean evaluation. "
                    "Reads cached INR payloads from --mesh_dir (read-only) and "
                    "refits from --input_dir on a cache miss, writing "
                    "freeform_eval.json under --output_dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mesh_dir", default="output_mesh",
                    help="Read-only source of cached INRs / segments. Never written to.")
    ap.add_argument("--input_dir", default="sample_clouds_abc_parts",
                    help="Point cloud root. Used only on a cache miss, where the "
                         "INR is fitted from scratch with the mesh-pipeline defaults.")
    ap.add_argument("--inr_max_steps", type=int, default=INR_MAX_STEPS,
                    help="Training steps per INR when fitting from scratch.")
    ap.add_argument("--output_dir", required=True,
                    help="Destination for freeform_eval.json sidecars; writes go to "
                         "{output_dir}/{model_id}/part_*/freeform_eval.json. Required "
                         "to force an explicit choice and avoid writing into mesh_dir.")
    ap.add_argument("--model_id", default=None,
                    help="Process only this model (subdir of mesh_dir). "
                         "Default: process every model in mesh_dir.")
    ap.add_argument("--K_eval", type=int, default=100,
                    help="Per-axis UV grid resolution for the eval sampling, "
                         "same on both surfaces. K_eval=100 yields 10000 samples "
                         "per surface; controls metric discretization noise.")
    ap.add_argument("--K_fit", type=int, default=50,
                    help="Per-axis UV grid resolution for the BSpline fit. "
                         "Default 50 matches the production conversion.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-evaluate parts that already have a freeform_eval.json "
                         "under output_dir.")
    args = ap.parse_args()

    abs_out  = os.path.abspath(args.output_dir)
    abs_mesh = os.path.abspath(args.mesh_dir)
    if abs_out == abs_mesh or abs_out.startswith(abs_mesh + os.sep):
        ap.error(f"--output_dir must not be --mesh_dir or any subdirectory of it "
                 f"(refusing to write into the mesh-pipeline source tree). "
                 f"output_dir={abs_out!r}  mesh_dir={abs_mesh!r}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.model_id is not None:
        target = os.path.join(args.mesh_dir, args.model_id)
        if not os.path.isdir(target):
            ap.error(f"model dir not found in mesh_dir: {target}")
        src_model_dirs = [target]
    else:
        src_model_dirs = sorted(
            d for d in glob.glob(os.path.join(args.mesh_dir, "*"))
            if os.path.isdir(d)
        )

    t0 = time.perf_counter()
    n_models = 0
    n_parts_processed = 0
    n_parts_skipped = 0
    n_inr_clusters = 0

    for src_model_dir in src_model_dirs:
        model_id = os.path.basename(src_model_dir)
        src_part_dirs = sorted(
            d for d in glob.glob(os.path.join(src_model_dir, "part_*"))
            if os.path.isdir(d)
        )
        if not src_part_dirs:
            continue
        n_models += 1
        for src_part_dir in src_part_dirs:
            part_name = os.path.basename(src_part_dir)
            try:
                part_idx = int(part_name.split("_", 1)[1])
            except (IndexError, ValueError):
                print(f"  [warn] cannot parse part index from {part_name} — skipping",
                      flush=True)
                continue

            dst_part_dir = os.path.join(args.output_dir, model_id, part_name)
            out_path = os.path.join(dst_part_dir, "freeform_eval.json")
            if os.path.isfile(out_path) and not args.overwrite:
                n_parts_skipped += 1
                continue

            os.makedirs(dst_part_dir, exist_ok=True)
            clusters = _eval_part_dir(
                src_part_dir, device,
                K_eval=args.K_eval, K_fit=args.K_fit,
                xyzc_path=os.path.join(args.input_dir, model_id, f"{part_idx}.xyzc"),
                max_steps=args.inr_max_steps)
            sidecar = {
                "model_id":  model_id,
                "part_idx":  part_idx,
                "K_eval":    int(args.K_eval),
                "K_fit":     int(args.K_fit),
                "clusters":  clusters,
            }
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sidecar, f, indent=2)
            os.replace(tmp, out_path)

            n_parts_processed += 1
            n_inr_clusters += len(clusters)

    dt = time.perf_counter() - t0
    print(f"\n[done] {n_models} model(s), {n_parts_processed} part(s) processed "
          f"({n_inr_clusters} INR clusters), {n_parts_skipped} part(s) skipped "
          f"(existing sidecar) in {dt:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()

import argparse
import glob as _glob
import multiprocessing
import numpy as np
import os
import time
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from point2cad.evaluation import compute_part_metrics
from point2cad.fitting_one_surface import process_one_surface
from point2cad.io_utils import save_unclipped_meshes, save_clipped_meshes, save_topology
from point2cad.utils import seed_everything, continuous_labels, normalize_points, make_colormap_optimal


def process_multiprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = {}
    with ProcessPoolExecutor(max_workers=cfg.max_parallel_surfaces) as executor:
        futures = {
            executor.submit(process_one_surface, idx, points, labels, cfg, device): idx
            for idx in uniq_labels
        }

        for future in tqdm(
            as_completed(futures), total=len(uniq_labels), desc="Fitting surfaces"
        ):
            idx = futures[future]
            out_meshes[idx] = future.result()
    kept_labels = [int(idx) for idx in uniq_labels if out_meshes[idx] is not None]
    out_meshes = [out_meshes[idx] for idx in uniq_labels if out_meshes[idx] is not None]
    return out_meshes, kept_labels


def process_singleprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = []
    kept_labels = []
    for idx in tqdm(uniq_labels, total=len(uniq_labels), desc="Fitting surfaces"):
        surface = process_one_surface(idx, points, labels, cfg, device)
        if surface is not None:
            out_meshes.append(surface)
            kept_labels.append(int(idx))
    return out_meshes, kept_labels


def _process_one_part(cfg, sample_path, part_idx, device, color_list, fn_process):
    """Run the original Point2CAD pipeline on a single .xyzc part file."""
    out_dir = os.path.join(cfg.output_dir, cfg.model_id, f"part_{part_idx}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/unclipped", exist_ok=True)
    os.makedirs(f"{out_dir}/clipped", exist_ok=True)
    os.makedirs(f"{out_dir}/topo", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Part {part_idx}: {os.path.basename(sample_path)}  ->  {out_dir}")
    print(f"{'='*60}")

    # ============================ load points ============================
    points_labels = np.loadtxt(sample_path).astype(np.float32)
    assert (
        points_labels.shape[1] == 4
    ), "This pipeline expects annotated point clouds (4 values per point). Refer to README for further instructions"
    points = points_labels[:, :3]
    labels = points_labels[:, 3].astype(np.int32)
    labels = continuous_labels(labels)

    points, norm_mean, norm_R, norm_scale = normalize_points(points)
    np.savez(os.path.join(out_dir, "normalization.npz"),
             mean=norm_mean, R=norm_R, scale=np.array(norm_scale))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    uniq_labels = np.unique(labels)

    # ============================ fit surfaces ============================
    t_fit = time.perf_counter()
    out_meshes, kept_labels = fn_process(cfg, uniq_labels, points, labels, device)
    fit_time = time.perf_counter() - t_fit

    # ============================ save unclipped meshes ============================
    print("Saving unclipped meshes...")
    pm_meshes = save_unclipped_meshes(
        out_meshes, color_list, f"{out_dir}/unclipped/mesh.ply"
    )

    # ============================ save clipped meshes ==============================
    # Time only the clipping work itself (matches mesh_pipeline's clip_time scope).
    print("Saving clipped meshes...")
    t_clip = time.perf_counter()
    clipped_meshes = save_clipped_meshes(
        pm_meshes, out_meshes, color_list, f"{out_dir}/clipped/mesh.ply"
    )
    clip_time = time.perf_counter() - t_clip

    print(f"[timing]   fit:  {fit_time:.2f}s")
    print(f"[timing]   clip: {clip_time:.2f}s")
    print(f"[timing]   sum:  {fit_time + clip_time:.2f}s")

    import json as _json
    timing_dict = {"fit_time": fit_time,
                   "clip_time": clip_time,
                   "total_time": fit_time + clip_time}
    with open(os.path.join(out_dir, "timing.json"), "w") as _f:
        _json.dump(timing_dict, _f, indent=2)

    # ============================ evaluation metrics ============================
    # `clipped_meshes` is a list[trimesh.Trimesh] aligned with `out_meshes` /
    # `kept_labels` (order preserved by save_clipped_meshes). `points` and
    # `labels` are already in normalized space at this point.
    cluster_meshes_dict = {
        kept_labels[i]: clipped_meshes[i] for i in range(len(clipped_meshes))
    }
    # Mirror io_utils._normalize_type: open_spline -> inr
    surface_types_dict = {
        kept_labels[i]: ("inr" if out_meshes[i]["info"]["type"] == "open_spline"
                         else out_meshes[i]["info"]["type"])
        for i in range(len(out_meshes))
    }
    metrics = compute_part_metrics(
        input_points=points,
        input_labels=labels,
        cluster_meshes=cluster_meshes_dict,
        surface_types=surface_types_dict,
        timing=timing_dict,
        model_id=cfg.model_id,
        part_idx=part_idx,
        seed=cfg.seed,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as _f:
        _json.dump(metrics, _f, indent=2)
    m = metrics["metrics"]
    print(f"[eval] p_cov_p2m={m['p_coverage']:.4f}  "
          f"p_cov_m2p={m['p_coverage_mesh_to_pc']:.4f}  "
          f"resid_mean={m['residual_mean']}  "
          f"chamfer_sym={m['chamfer_sym']}  "
          f"primitive_only={metrics['is_primitive_only']}")

    # ============================ get edges and corners ============================
    # Topology export disabled — not used downstream and skews timing comparisons.
    # print("Saving topology (edges and corners)...")
    # save_topology(clipped_meshes, f"{out_dir}/topo/topo.json")

    print(f"Part {part_idx} done.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    color_list = make_colormap_optimal()

    parser = argparse.ArgumentParser(description="Point2CAD pipeline")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model ID (subdirectory under input_dir / output_dir)")
    parser.add_argument("--input_dir", type=str, default="../point2cad_repr/sample_clouds_abc_parts",
                        help="Root directory for input point cloud subdirs (one per model)")
    parser.add_argument("--output_dir", type=str, default="output_p2cad_orig",
                        help="Root directory for outputs")
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part index (0-based). Default: all parts")
    parser.add_argument("--validate_checkpoint_path", type=str, default=None)
    parser.add_argument("--silent", default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_parallel_surfaces", type=int, default=4)
    parser.add_argument("--num_inr_fit_attempts", type=int, default=1)
    parser.add_argument("--surfaces_multiprocessing", type=int, default=0,
                        help="Run surface fits in parallel. Disabled by default: "
                             "process_one_surface writes tmp.obj to CWD per cluster, "
                             "which races under multiprocessing.")
    cfg = parser.parse_args()

    seed_everything(cfg.seed)

    fn_process = process_singleprocessing
    if cfg.surfaces_multiprocessing:
        multiprocessing.set_start_method("spawn", force=True)
        fn_process = process_multiprocessing

    # ============================ resolve parts ============================
    input_pattern = os.path.join(cfg.input_dir, cfg.model_id, "*.xyzc")
    xyzc_files = sorted(
        _glob.glob(input_pattern),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not xyzc_files:
        raise FileNotFoundError(f"No .xyzc files found matching: {input_pattern}")

    if cfg.part is not None:
        if cfg.part >= len(xyzc_files):
            raise IndexError(
                f"Part {cfg.part} out of range — model has {len(xyzc_files)} part(s)"
            )
        part_indices = [cfg.part]
    else:
        part_indices = list(range(len(xyzc_files)))

    print(f"Model {cfg.model_id}: {len(xyzc_files)} part(s), processing {len(part_indices)}")

    os.makedirs(os.path.join(cfg.output_dir, cfg.model_id), exist_ok=True)

    part_times = []
    t_total = time.perf_counter()
    for part_idx in part_indices:
        t_part = time.perf_counter()
        _process_one_part(cfg, xyzc_files[part_idx], part_idx, device, color_list, fn_process)
        dt_part = time.perf_counter() - t_part
        part_times.append((part_idx, dt_part))
        print(f"[timing] part {part_idx}: {dt_part:.2f}s")

    dt_total = time.perf_counter() - t_total
    print(f"\n[timing] === point2cad orig summary for {cfg.model_id} ===")
    for pi, dt in part_times:
        print(f"[timing]   part {pi}: {dt:.2f}s")
    print(f"[timing]   total ({len(part_times)} parts): {dt_total:.2f}s")
    print("Done")

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
import glob as _glob

import numpy as np
import open3d as o3d

from point2step.surface_types import SURFACE_NAMES
from point2step.inr_fitting import (
    INR_MAX_STEPS, INR_NETWORK_PARAMETERS, inr_fit_kwargs,
)
from point2step.surface_fitter import MIN_CLUSTER_PTS

DEFAULT_FITNESS_THRESHOLD   = 10
DEFAULT_VERTEX_THRESHOLD    = 1e-3
DEFAULT_STEP_SAMPLE_COUNT   = 30000
DEFAULT_SELECTION_FRAME     = "normalized"

# Evaluation regime: a bare run reproduces what the evaluation wrapper drives.
DEFAULT_SPACING_FACTOR_MIN  = 0.5
DEFAULT_SPACING_FACTOR_MAX  = 1.5
DEFAULT_SPACING_FACTOR_STEP = 0.2

CURVE_TYPE_COLORS = {
    "line":    [1.0, 0.2, 0.2],
    "circle":  [0.2, 0.85, 0.2],
    "ellipse": [0.2, 0.4,  1.0],
    "conic":   [0.8, 0.2,  0.8],
    "bspline": [1.0, 0.6,  0.0],
    "curve":   [0.8, 0.8,  0.0],
    "tangent": [1.0, 1.0,  1.0],
}


def _write_part_status(part_dir, status):
    os.makedirs(part_dir, exist_ok=True)
    path = os.path.join(part_dir, "wrapper_status.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"status": status}, f)
    os.replace(tmp, path)


def _denorm(pts, mean, R, scale):
    pts = np.asarray(pts, dtype=np.float64)
    return (scale * (pts @ R) + mean).astype(np.float32)


def _merge_part_dirs(part_dirs, unified_dir):
    os.makedirs(unified_dir, exist_ok=True)

    all_sids, all_snames, all_colors = [], [], []
    per_part_adj = []
    for dir_path, offset, n in part_dirs:
        meta = np.load(os.path.join(dir_path, "metadata.npz"), allow_pickle=True)
        all_sids.extend(meta["surface_ids"].tolist())
        all_snames.extend(meta["surface_names"].tolist())
        all_colors.extend(meta["cluster_colors"].tolist())
        if "adjacency_matrix" in meta.files:
            per_part_adj.append((offset, n, np.asarray(meta["adjacency_matrix"], dtype=bool)))
        else:
            per_part_adj.append((offset, n, np.zeros((n, n), dtype=bool)))

    total = len(all_sids)
    adj_full = np.zeros((total, total), dtype=bool)
    for offset, n, a in per_part_adj:
        adj_full[offset:offset + n, offset:offset + n] = a

    np.savez(os.path.join(unified_dir, "metadata.npz"),
             n_clusters       = total,
             surface_ids      = np.array(all_sids),
             surface_names    = np.array(all_snames),
             cluster_colors   = np.array(all_colors),
             adjacency_matrix = adj_full)

    for vfile in ("vertices.npz", "vertices_pre_filter.npz"):
        all_verts = []
        for dir_path, offset, n in part_dirs:
            vpath = os.path.join(dir_path, vfile)
            if os.path.exists(vpath):
                v = np.load(vpath)["vertices"]
                if len(v):
                    all_verts.append(v)
        np.savez(os.path.join(unified_dir, vfile),
                 vertices=np.concatenate(all_verts) if all_verts else np.zeros((0, 3)))

    for dir_path, offset, n in part_dirs:
        for i in range(n):
            for tmpl in (f"cluster_{i}.npy", f"surface_mesh_{i}.npz"):
                src = os.path.join(dir_path, tmpl)
                if not os.path.exists(src):
                    continue
                dst_name = tmpl.replace(f"_{i}.", f"_{i + offset}.")
                shutil.copy2(src, os.path.join(unified_dir, dst_name))

    for dir_path, offset, n in part_dirs:
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            if fname.startswith("inter_") and fname.endswith(".npz"):
                d = dict(np.load(fpath, allow_pickle=True))
                i, j = int(d["cluster_i"]), int(d["cluster_j"])
                d["cluster_i"] = i + offset
                d["cluster_j"] = j + offset
                np.savez(os.path.join(unified_dir,
                                      f"inter_{i+offset}_{j+offset}.npz"), **d)
            elif fname.startswith("arcs_") and fname.endswith(".npz"):
                d = dict(np.load(fpath, allow_pickle=True))
                ei, ej = int(d["edge_i"]), int(d["edge_j"])
                d["edge_i"] = ei + offset
                d["edge_j"] = ej + offset
                if fname.startswith("arcs_pre_filter_"):
                    dst = f"arcs_pre_filter_{ei+offset}_{ej+offset}.npz"
                else:
                    dst = f"arcs_{ei+offset}_{ej+offset}.npz"
                np.savez(os.path.join(unified_dir, dst), **d)
            elif fname.startswith("boundary_strip_") and fname.endswith(".npy"):
                base = fname.replace("boundary_strip_", "").replace(".npy", "")
                bi, bj = (int(x) for x in base.split("_"))
                dst = f"boundary_strip_{bi+offset}_{bj+offset}.npy"
                shutil.copy2(fpath, os.path.join(unified_dir, dst))


def _visualize_input_only(args):
    pattern = os.path.join(args.input_dir, args.model_id, "*.xyzc")
    paths = sorted(_glob.glob(pattern))
    if not paths:
        print(f"[visualize] no .xyzc files matching {pattern}", file=sys.stderr)
        return
    rng = np.random.default_rng(0)
    pcds = []
    for path in paths:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pts = arr[:, :3]
        labels = (arr[:, -1].astype(int) if arr.shape[1] >= 4
                  else np.zeros(len(arr), dtype=int))
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[mask])
            pcd.paint_uniform_color(rng.random(3).tolist())
            pcds.append(pcd)
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Input point cloud — {args.model_id} (no intermediates)",
                      width=1280, height=960)
    for pcd in pcds:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()


def run_visualize(args):
    if args.part is not None:
        out_dir = os.path.join(args.output_dir, f"{args.model_id}",
                               f"part_{args.part}")
    else:
        out_dir = os.path.join(args.output_dir, f"{args.model_id}", "unified")
    if not os.path.isfile(os.path.join(out_dir, "metadata.npz")):
        return _visualize_input_only(args)

    meta         = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle=True)
    n_clusters   = int(meta["n_clusters"])
    clust_colors = meta["cluster_colors"]
    surf_names   = meta["surface_names"] if "surface_names" in meta else [None] * n_clusters

    cluster_pcds = []
    all_cluster_pts = []
    for i in range(n_clusters):
        pts = np.load(os.path.join(out_dir, f"cluster_{i}.npy"))
        all_cluster_pts.append(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(clust_colors[i].tolist())
        cluster_pcds.append(pcd)

    _all_pts = np.concatenate(all_cluster_pts, axis=0)
    _bbox_min = _all_pts.min(axis=0)
    _bbox_max = _all_pts.max(axis=0)
    _bbox_diag = np.linalg.norm(_bbox_max - _bbox_min)
    _bbox_margin = 0.5 * _bbox_diag
    _vis_min = _bbox_min - _bbox_margin
    _vis_max = _bbox_max + _bbox_margin

    def _lineset(pts, color):
        lines = [[m, m + 1] for m in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        return ls

    def _clip_to_bbox(pts):
        inside = np.all((pts >= _vis_min) & (pts <= _vis_max), axis=1)
        if not np.any(inside):
            return None
        idxs = np.where(inside)[0]
        best_start, best_len = idxs[0], 1
        cur_start, cur_len = idxs[0], 1
        for i in range(1, len(idxs)):
            if idxs[i] == idxs[i-1] + 1:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_start, cur_len = idxs[i], 1
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len
        if best_len < 2:
            return None
        return pts[best_start:best_start + best_len]

    arc_linesets = []
    for arc_path in sorted(_glob.glob(os.path.join(out_dir, "arcs_*.npz"))):
        if "pre_filter" in os.path.basename(arc_path):
            continue
        d      = np.load(arc_path, allow_pickle=True)
        n_arcs = int(d["n_arcs"])
        for k in range(n_arcs):
            pts   = d[f"arc_points_{k}"]
            color = d[f"arc_color_{k}"].tolist()
            arc_linesets.append(_lineset(pts, color))

    pre_filter_arc_linesets = []
    for arc_path in sorted(_glob.glob(os.path.join(out_dir, "arcs_pre_filter_*.npz"))):
        d      = np.load(arc_path, allow_pickle=True)
        n_arcs = int(d["n_arcs"])
        for k in range(n_arcs):
            pts   = d[f"arc_points_{k}"]
            color = d[f"arc_color_{k}"].tolist()
            pre_filter_arc_linesets.append(_lineset(pts, color))

    trimmed_linesets   = []
    untrimmed_linesets = []
    polyline_linesets  = []
    detected_method    = None
    boundary_pcds      = []
    for inter_path in sorted(_glob.glob(os.path.join(out_dir, "inter_*.npz"))):
        d          = np.load(inter_path, allow_pickle=True)
        curve_type = str(d["curve_type"])
        n_curves   = int(d["n_curves"])
        n_raw      = int(d["n_untrimmed_curves"])
        ci, cj     = int(d["cluster_i"]), int(d["cluster_j"])
        method     = str(d["method"]) if "method" in d else "analytical"
        if detected_method is None:
            detected_method = method
        print(f"({ci}, {cj})  {d['surface_i_name']} ∩ {d['surface_j_name']}"
              f"  type={curve_type}  method={method}  trimmed={n_curves}  raw={n_raw}")
        if method in ("mesh", "msi"):
            color = [1.0, 0.0, 0.8]  # magenta for mesh-derived curves
        else:
            color = CURVE_TYPE_COLORS.get(curve_type, [0.8, 0.8, 0.8])
        for k in range(n_curves):
            pts = _clip_to_bbox(d[f"curve_points_{k}"])
            if pts is not None:
                trimmed_linesets.append(_lineset(pts, color))
        for k in range(n_raw):
            pts = _clip_to_bbox(d[f"untrimmed_curve_points_{k}"])
            if pts is not None:
                untrimmed_linesets.append(_lineset(pts, color))
        n_polys = int(d["n_polylines"]) if "n_polylines" in d else 0
        for k in range(n_polys):
            poly_pts = d[f"polyline_points_{k}"]
            print(f"  polyline ({ci},{cj})[{k}]: {poly_pts.shape} "
                  f"range=[{poly_pts.min(axis=0)}, {poly_pts.max(axis=0)}]")
            polyline_linesets.append(
                _lineset(poly_pts, [0.0, 0.9, 0.9]))
        if "boundary_pts" in d:
            bpts = d["boundary_pts"]
            if len(bpts) > 0:
                bpcd = o3d.geometry.PointCloud()
                bpcd.points = o3d.utility.Vector3dVector(bpts)
                bpcd.paint_uniform_color([0.5, 0.5, 0.5])
                boundary_pcds.append(bpcd)

    surface_meshes = []
    surface_mesh_cids = []   # cluster id for each mesh (parallel to surface_meshes)
    for mesh_path in sorted(_glob.glob(os.path.join(out_dir, "surface_mesh_*.npz"))):
        ci   = int(os.path.basename(mesh_path).replace("surface_mesh_", "").replace(".npz", ""))
        d    = np.load(mesh_path)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(d["vertices"])
        mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(clust_colors[ci].tolist())
        surface_meshes.append(mesh)
        surface_mesh_cids.append(ci)

    _highlight = {"idx": -1}  # -1 = show all normally

    def _update_highlight(vis_obj):
        idx = _highlight["idx"]
        for k, mesh in enumerate(surface_meshes):
            ci = surface_mesh_cids[k]
            if idx == -1:
                mesh.paint_uniform_color(clust_colors[ci].tolist())
            elif k == idx:
                mesh.paint_uniform_color(clust_colors[ci].tolist())
            else:
                mesh.paint_uniform_color([0.3, 0.3, 0.3])
            mesh.compute_vertex_normals()
            vis_obj.update_geometry(mesh)
        vis_obj.update_renderer()
        if idx == -1:
            print("[surfaces] showing ALL clusters")
        else:
            ci = surface_mesh_cids[idx]
            sname = str(surf_names[ci]) if ci < len(surf_names) else "?"
            print(f"[surfaces] highlighting cluster {ci} ({sname})")

    def _on_key_next(vis_obj):
        n = len(surface_meshes)
        _highlight["idx"] = (_highlight["idx"] + 1) % (n + 1)
        if _highlight["idx"] == n:
            _highlight["idx"] = -1
        _update_highlight(vis_obj)
        return False

    def _on_key_prev(vis_obj):
        n = len(surface_meshes)
        _highlight["idx"] = (_highlight["idx"] - 1)
        if _highlight["idx"] < -1:
            _highlight["idx"] = n - 1
        _update_highlight(vis_obj)
        return False

    vertex_pcd  = None
    vertex_path = os.path.join(out_dir, "vertices.npz")
    if os.path.exists(vertex_path):
        verts = np.load(vertex_path)["vertices"]
        if len(verts) > 0:
            vertex_pcd = o3d.geometry.PointCloud()
            vertex_pcd.points = o3d.utility.Vector3dVector(verts)
            vertex_pcd.paint_uniform_color([1.0, 1.0, 0.0])
            print(f"Loaded {len(verts)} vertices (post-filter)")

    pre_filter_vertex_pcd = None
    pre_filter_vertex_path = os.path.join(out_dir, "vertices_pre_filter.npz")
    if os.path.exists(pre_filter_vertex_path):
        verts_pre = np.load(pre_filter_vertex_path)["vertices"]
        if len(verts_pre) > 0:
            pre_filter_vertex_pcd = o3d.geometry.PointCloud()
            pre_filter_vertex_pcd.points = o3d.utility.Vector3dVector(verts_pre)
            pre_filter_vertex_pcd.paint_uniform_color([1.0, 0.5, 0.0])
            print(f"Loaded {len(verts_pre)} vertices (pre-filter)")

    W, H = 640, 490

    vis1 = o3d.visualization.Visualizer()
    if polyline_linesets:
        polyline_title = "Raw polylines (MSI)" if detected_method == "msi" else "Raw polylines (mesh)"
        vis1.create_window(polyline_title, width=W, height=H, left=0, top=50)
        for ls in polyline_linesets:
            vis1.add_geometry(ls)
    else:
        vis1.create_window("Untrimmed curves", width=W, height=H, left=0, top=50)
        for ls in untrimmed_linesets:
            vis1.add_geometry(ls)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Pre-filter arcs + vertices", width=W, height=H, left=W, top=50)
    for ls in (pre_filter_arc_linesets if pre_filter_arc_linesets else trimmed_linesets):
        vis2.add_geometry(ls)
    if pre_filter_vertex_pcd is not None:
        vis2.add_geometry(pre_filter_vertex_pcd)
    vis2.get_render_option().point_size = 8.0

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window("Post-filter arcs + vertices", width=W, height=H, left=2*W, top=50)
    for ls in (arc_linesets if arc_linesets else trimmed_linesets):
        vis3.add_geometry(ls)
    if vertex_pcd is not None:
        vis3.add_geometry(vertex_pcd)
    vis3.get_render_option().point_size = 8.0

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window("Point clouds",
                       width=W, height=H, left=0, top=50 + H + 40)
    for pcd in cluster_pcds:
        vis4.add_geometry(pcd)
    vis4.get_render_option().point_size = 2.0

    vis6 = o3d.visualization.Visualizer()
    vis6.create_window("Boundary strips",
                       width=W, height=H, left=W, top=50 + H + 40)
    for bpcd in boundary_pcds:
        vis6.add_geometry(bpcd)
    vis6.get_render_option().point_size = 10.0

    vis5 = o3d.visualization.VisualizerWithKeyCallback()
    vis5.create_window("Fitted surfaces (N/P to cycle clusters)",
                       width=W, height=H,
                       left=2 * W, top=50 + H + 40)
    for mesh in surface_meshes:
        vis5.add_geometry(mesh)
    vis5.register_key_callback(ord("N"), _on_key_next)
    vis5.register_key_callback(ord("P"), _on_key_prev)
    print("[surfaces] Press N/P in 'Fitted surfaces' window to cycle clusters")

    vis4.get_render_option().mesh_show_back_face = True
    vis5.get_render_option().mesh_show_back_face = True
    visualizers = [vis1, vis2, vis3, vis4, vis5]
    if vis6 is not None:
        vis6.get_render_option().mesh_show_back_face = True
        visualizers.append(vis6)
    running     = [True] * len(visualizers)
    while all(running):
        for i, vis in enumerate(visualizers):
            if running[i]:
                running[i] = vis.poll_events()
                vis.update_renderer()
        time.sleep(0.01)
    for vis in visualizers:
        vis.destroy_window()


def _print_sweep_summaries(summaries, selection_frame):
    def _fmt(v, w, p):
        return f"{v:{w}.{p}f}" if v is not None else "-".rjust(w)

    print(f"\n[summary] === sweep results ===")
    print(f"[summary] tau={summaries[0]['tau_norm']:g}  selection_frame={selection_frame}")
    for s in summaries:
        print(f"[summary] part {s['part_idx']}")
        print((f"[summary] {'':>14}  {'':>5}  {'':>9}  {'OCD':^21}").rstrip())
        print(f"[summary] {'spacing factor':>14}  {'valid':>5}  {'P2S':>9}  "
              f"{'normalized':>10} {'world':>10}  best")
        for rec in s["records"]:
            tags = ",".join(t for t, hit in (("overall", rec is s["best_overall"]),
                                             ("valid", rec is s["best_valid"])) if hit)
            row = (f"[summary] {rec.spacing:>14}  {str(rec.valid):>5}  "
                   f"{_fmt(rec.coverage_norm, 9, 5)}  "
                   f"{_fmt(rec.fidelity_norm, 10, 6)} {_fmt(rec.fidelity_world, 10, 6)}  "
                   f"{tags}")
            print(row.rstrip())
        if s["best_overall"] is None:
            print("[summary] no spacing factor produced a STEP file")
        elif s["best_valid"] is None:
            print("[summary] no spacing factor produced a kernel-valid STEP file")


def run_compute(args):
    import torch

    from point2step.surface_fitter       import fit_surface
    from point2step.occ_surfaces         import to_occ_surface
    from point2step.cluster_adjacency    import (
        compute_adjacency_matrix, adjacency_pairs, build_cluster_proximity,
    )
    from point2step.color_config         import get_surface_color
    from point2step.surface_intersection import (
        compute_all_intersections,
        trim_by_vertices,
        compute_vertices_extrema,
        sample_curve,
        _as_safe_curve,
    )
    from point2step.topology import (
        build_edge_arcs,
        ilp_topology_filter,
        _score_vertex, _score_arc,
        print_edge_arcs_summary,
        face_arc_incidence, print_face_arcs_summary,
        assemble_wires, print_face_wires_summary,
        export_step,
        apply_inverse_normalization,
        make_uv_bounded_face,
    )
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    import point2step.primitive_fitting_utils as pfu

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def normalize_points(pts):
        mean    = np.mean(pts, axis=0)
        pts     = pts - mean
        S, U    = np.linalg.eigh(pts.T @ pts)
        R       = pfu.rotation_matrix_a_to_b(U[:, np.argmin(S)], np.array([1, 0, 0]))
        pts     = (R @ pts.T).T
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale   = float(np.max(extents) + 1e-7)
        return (pts / scale).astype(np.float32), mean, R, scale

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    input_pattern = os.path.join(args.input_dir, f"{args.model_id}", "*.xyzc")
    part_files    = sorted(_glob.glob(input_pattern),
                           key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not part_files:
        print(f"No part files found matching: {input_pattern}")
        return
    model_out_dir = os.path.join(args.output_dir, f"{args.model_id}")
    if not args.no_clean_output:
        if os.path.exists(model_out_dir):
            shutil.rmtree(model_out_dir)
            print(f"Removed old results: {model_out_dir}")
        os.makedirs(model_out_dir)
    else:
        os.makedirs(model_out_dir, exist_ok=True)
    print(f"Model {args.model_id}: {len(part_files)} part(s)")

    part_dirs      = []   # (out_dir, cluster_offset, n_clusters) — for final merge
    sweep_summaries = []  # per-part sweep records, printed as the final digest
    cluster_offset = 0

    for part_idx, sample_path in enumerate(part_files):
        if args.part is not None and part_idx != args.part:
            continue
        part_seed = args.seed + part_idx
        np_rng = np.random.default_rng(part_seed)
        torch.manual_seed(part_seed)
        torch.cuda.manual_seed(part_seed)
        step_stem = f"part_{part_idx}"
        out_dir   = os.path.join(model_out_dir, f"part_{part_idx}")

        os.makedirs(out_dir, exist_ok=True)
        failed = False
        try:
            print(f"\n{'='*60}")
            print(f"Part {part_idx}: {os.path.basename(sample_path)}  →  {out_dir}")
            print(f"{'='*60}")

            data = np.loadtxt(sample_path)
            data[:, :3], part_mean, part_R, part_scale = normalize_points(data[:, :3])
            unique_clusters, cluster_counts = np.unique(data[:, -1].astype(int), return_counts=True)
            os.makedirs(out_dir, exist_ok=True)

            clusters_master = []
            for cid in unique_clusters:
                cluster = data[data[:, -1].astype(int) == cid, :3].astype(np.float32)
                clusters_master.append(cluster)

            print(f"[preprocess] {len(unique_clusters)} cluster(s) in input, "
                  f"counts={cluster_counts.tolist()}")

            # HPNet/Point2CAD floor of 20 points, enough evidence for a 6-DoF cone
            keep_mask = [len(c) >= MIN_CLUSTER_PTS for c in clusters_master]
            n_dropped = sum(1 for k in keep_mask if not k)
            if n_dropped > 0:
                dropped_info = [(int(cid), len(clusters_master[i]))
                                for i, cid in enumerate(unique_clusters) if not keep_mask[i]]
                print(f"[preprocess] dropped {n_dropped} cluster(s) "
                      f"with < {MIN_CLUSTER_PTS} pts: {dropped_info}")
                clusters_master = [c for c, k in zip(clusters_master, keep_mask) if k]
                unique_clusters = np.array([cid for cid, k in zip(unique_clusters, keep_mask) if k])
                cluster_counts  = np.array([cnt for cnt, k in zip(cluster_counts, keep_mask) if k])
            print(f"[preprocess] {len(clusters_master)} cluster(s) after MIN_CLUSTER_PTS filter")

            cluster_trees_master, cluster_nn_percentiles_master = build_cluster_proximity(
                clusters_master, percentile=args.spacing_percentile
            )

            surface_ids_master, fit_results_master, fit_meshes_master, occ_surfs_master = [], [], [], []
            for idx, (cid, c_count) in enumerate(zip(unique_clusters, cluster_counts)):
                cluster = clusters_master[idx]
                print(f"[surface fitter] Cluster {cid} ({c_count} pts) fitting ...")
                _spacing = cluster_nn_percentiles_master[idx]
                _plane_kw    = {"mesh_dim": 100, "plane_sampling_deviation": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 5,
                                "absolute_threshold": 0.1}
                _sphere_kw   = {"dim_theta": 100, "dim_lambda": 100,
                                "spacing": _spacing,
                                "threshold_multiplier": 3,
                                "absolute_threshold": 0.1}
                _cylinder_kw = {"dim_theta": 100, "dim_height": 50,
                                "cylinder_height_margin": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 3,
                                "absolute_threshold": 0.1}
                _cone_kw     = {"dim_theta": 100, "dim_height": 100,
                                "cone_height_margin": 0.5,
                                "spacing": _spacing,
                                "threshold_multiplier": 3,
                                "absolute_threshold": 0.1}

                res = fit_surface(
                    cluster,
                    INR_NETWORK_PARAMETERS,
                    np_rng, DEVICE,
                    inr_fit_kwargs=inr_fit_kwargs(args.inr_max_steps),
                    inr_mesh_kwargs={
                        "mesh_dim": 200,
                        "uv_margin": 0.1,
                        "threshold_multiplier": 1,
                    },
                    plane_mesh_kwargs=_plane_kw,
                    sphere_mesh_kwargs=_sphere_kw,
                    cylinder_mesh_kwargs=_cylinder_kw,
                    cone_mesh_kwargs=_cone_kw,
                    radius_inflation=0,
                    angle_inflation_deg=0
                )
                sid = res["surface_id"]
                surface_ids_master.append(sid)
                fit_results_master.append(res["result"])
                fit_meshes_master.append(res["mesh"])
                occ_surfs_master.append(
                    to_occ_surface(sid, res["result"], cluster=cluster, uv_margin = 0.05, grid_resolution = 50, periodic = True)
                )
                chosen_err = res["result"]["error"]
                all_errors = res.get("all_errors", {})
                errors_str = "  ".join(f"{name}={err:.6f}" for name, err in all_errors.items())
                print(f"[surface fitter] Cluster {cid} ({c_count} pts) → "
                      f"{SURFACE_NAMES[sid]}  residual={chosen_err:.6f}")
                if errors_str:
                    print(f"  all errors: {errors_str}")

            clusters_master_concat_norm  = np.concatenate(clusters_master, axis=0).astype(np.float64)
            clusters_master_concat_world = _denorm(
                clusters_master_concat_norm, part_mean, part_R, part_scale)

            for _stale in _glob.glob(os.path.join(out_dir, "*")):
                os.remove(_stale)

            def _run_variant(spacing_factor, write_intermediates):
                clusters       = list(clusters_master)
                surface_ids    = list(surface_ids_master)
                fit_results    = list(fit_results_master)
                fit_meshes     = list(fit_meshes_master)
                occ_surfs      = list(occ_surfs_master)
                cluster_trees  = cluster_trees_master
                cluster_nn_percentiles = cluster_nn_percentiles_master

                adj, _, spacing, boundary_strips, per_pair_thresholds, boundary_strip_trees = compute_adjacency_matrix(
                    clusters, threshold_factor=spacing_factor,
                    spacing_percentile=args.spacing_percentile,
                    local_spacings=cluster_nn_percentiles,
                )
                inter_adj = adj
                print(f"[adjacency] spacing={spacing:.5f}  threshold={spacing_factor * spacing:.5f}")
                print(f"[adjacency] adjacent pairs: {adjacency_pairs(adj)}")
                for (i, j), bpts in sorted(boundary_strips.items()):
                    print(f"  boundary ({i}, {j}): {len(bpts)} points")

                if write_intermediates:
                    np.savez(
                        os.path.join(out_dir, "metadata.npz"),
                        n_clusters       = len(clusters),
                        surface_ids      = np.array(surface_ids),
                        surface_names    = np.array([SURFACE_NAMES[s] for s in surface_ids]),
                        cluster_colors   = np.array(
                            [get_surface_color(SURFACE_NAMES[s]).tolist() for s in surface_ids]
                        ),
                        adjacency_matrix = inter_adj.astype(bool),
                    )
                    for i, cluster in enumerate(clusters):
                        np.save(os.path.join(out_dir, f"cluster_{i}.npy"),
                                _denorm(cluster, part_mean, part_R, part_scale))
                    for (i, j), bpts in boundary_strips.items():
                        np.save(os.path.join(out_dir, f"boundary_strip_{i}_{j}.npy"),
                                _denorm(bpts.astype(np.float64), part_mean, part_R, part_scale))
                    for i, mesh in enumerate(fit_meshes):
                        np.savez(
                            os.path.join(out_dir, f"surface_mesh_{i}.npz"),
                            vertices  = _denorm(np.asarray(mesh.vertices), part_mean, part_R, part_scale),
                            triangles = np.asarray(mesh.triangles),
                        )
                    print(f"[surface fitter] cluster files saved to {out_dir}/")

                raw_intersections = compute_all_intersections(
                    inter_adj, surface_ids, fit_results, occ_surfs,
                )

                vertices, vertex_edges = compute_vertices_extrema(
                    inter_adj, raw_intersections,
                    threshold=args.vertex_threshold,
                )
                print(f"[vertices] found {len(vertices)} vertices "
                      f"(ExtremaCurveCurve, threshold={args.vertex_threshold})")

                vertex_scores = []
                for v_idx, (vpos, edges) in enumerate(zip(vertices, vertex_edges)):
                    involved = set()
                    for edge in edges:
                        involved.update(edge)
                    score = _score_vertex(vpos, involved, cluster_trees,
                                          cluster_nn_percentiles)
                    vertex_scores.append(score)

                fitness_threshold = args.fitness_threshold
                scores_arr = np.array(vertex_scores)

                sorted_indices = np.argsort(scores_arr)
                print(f"[fitness filter] vertices: {len(scores_arr)}, "
                      f"score range [{0 if scores_arr.size == 0 else scores_arr.min():.4f}, "
                      f"{0 if scores_arr.size == 0 else scores_arr.max():.4f}]")
                for rank, idx in enumerate(sorted_indices):
                    edges_str = " ".join(
                        f"({min(e)}, {max(e)})" for e in vertex_edges[idx])
                    print(f"  v{idx:3d}  score={scores_arr[idx]:10.4f}  "
                          f"edges=[{edges_str}]")

                keep_v = scores_arr <= fitness_threshold
                n_drop = int(np.sum(~keep_v))
                n_keep = int(np.sum(keep_v))
                if n_drop > 0 and n_keep >= 2:
                    print(f"[fitness filter] vertex threshold: {fitness_threshold:.2f}  "
                          f"keeping {n_keep}, dropping {n_drop}/{len(scores_arr)}")
                    vertices = vertices[keep_v]
                    vertex_edges = [vertex_edges[i]
                                    for i in range(len(keep_v)) if keep_v[i]]
                else:
                    print(f"[fitness filter] vertex threshold {fitness_threshold:.2f} "
                          f"would keep {n_keep} / drop {n_drop} — skipping")
                print(f"[fitness filter] after vertex filter: {len(vertices)} vertices")

                trim_intersections_ = trim_by_vertices(
                    raw_intersections, vertices, vertex_edges,
                    extension_factor=0.05,
                )
                for (i, j) in raw_intersections:
                    inter_raw  = raw_intersections[(i, j)]
                    inter_trim = trim_intersections_[(i, j)]
                    si = SURFACE_NAMES[surface_ids[i]]
                    sj = SURFACE_NAMES[surface_ids[j]]
                    print(f"[intersect] ({i}, {j})  {si} ∩ {sj}  type={inter_raw['type']}"
                          f"  method={inter_raw['method']}"
                          f"  raw={len(inter_raw['curves'])}  trimmed={len(inter_trim['curves'])}")
                    if write_intermediates:
                        kw = dict(
                            cluster_i          = i,
                            cluster_j          = j,
                            surface_i_name     = si,
                            surface_j_name     = sj,
                            curve_type         = inter_raw["type"],
                            method             = inter_raw["method"],
                            n_curves           = len(inter_trim["curves"]),
                            n_untrimmed_curves = len(inter_raw["curves"]),
                        )
                        boundary_pts = boundary_strips.get((i, j), np.empty((0, 3)))
                        kw["n_boundary_pts"] = len(boundary_pts)
                        if len(boundary_pts) > 0:
                            kw["boundary_pts"] = _denorm(boundary_pts, part_mean, part_R, part_scale)
                        for k, curve in enumerate(inter_trim["curves"]):
                            t0, t1 = curve.FirstParameter(), curve.LastParameter()
                            p0, p1 = curve.Value(t0), curve.Value(t1)
                            endpoint_dist = math.sqrt(
                                (p1.X() - p0.X()) ** 2 +
                                (p1.Y() - p0.Y()) ** 2 +
                                (p1.Z() - p0.Z()) ** 2
                            )
                            print(f"  trimmed  curve[{k}] [{t0:.6f}, {t1:.6f}]"
                                  f"  endpoint_dist={endpoint_dist:.6e}")
                            kw[f"curve_points_{k}"] = _denorm(
                                sample_curve(curve, n_points=200), part_mean, part_R, part_scale)
                        raw_saved = 0
                        for k, curve in enumerate(inter_raw["curves"]):
                            t0_raw, t1_raw = curve.FirstParameter(), curve.LastParameter()
                            if abs(t0_raw) > 1e50 or abs(t1_raw) > 1e50:
                                curve = _as_safe_curve(curve)
                                t0_raw, t1_raw = curve.FirstParameter(), curve.LastParameter()
                                print(f"  raw      curve[{k}] [trimmed to {t0_raw:.6f}, {t1_raw:.6f}]")
                            else:
                                print(f"  raw      curve[{k}] [{t0_raw:.6f}, {t1_raw:.6f}]")
                            kw[f"untrimmed_curve_points_{raw_saved}"] = _denorm(
                                sample_curve(curve, n_points=200),
                                part_mean, part_R, part_scale)
                            raw_saved += 1
                        kw["n_untrimmed_curves"] = raw_saved
                        np.savez(os.path.join(out_dir, f"inter_{i}_{j}.npz"), **kw)

                trim_curves_dict = {k: v["curves"] for k, v in trim_intersections_.items()}
                edge_arcs, vertices, vertex_edges = build_edge_arcs(
                    trim_curves_dict, vertices, vertex_edges, threshold=1e-3
                )
                print_edge_arcs_summary(edge_arcs)

                fitness_threshold = args.fitness_threshold
                n_before = sum(len(a) for a in edge_arcs.values())
                n_dropped = 0
                for edge_key in list(edge_arcs.keys()):
                    i, j = edge_key
                    kept = []
                    for arc in edge_arcs[edge_key]:
                        s = _score_arc(arc, i, j, cluster_trees, cluster_nn_percentiles)
                        if s <= fitness_threshold:
                            kept.append(arc)
                        else:
                            print(f"[fitness filter] dropping arc {edge_key}[{arc['arc_idx']}] "
                                  f"score={s:.4f}")
                            n_dropped += 1
                    if kept:
                        edge_arcs[edge_key] = kept
                    else:
                        del edge_arcs[edge_key]
                print(f"[fitness filter] arcs: keeping {n_before - n_dropped}/{n_before}, "
                      f"dropped {n_dropped} (threshold={fitness_threshold:.2f})")

                if write_intermediates:
                    np.savez(os.path.join(out_dir, "vertices_pre_filter.npz"),
                             vertices=_denorm(vertices, part_mean, part_R, part_scale))
                    for (ei, ej), arcs in edge_arcs.items():
                        kw_pre = {"n_arcs": len(arcs), "edge_i": ei, "edge_j": ej}
                        for k, arc in enumerate(arcs):
                            kw_pre[f"arc_points_{k}"] = _denorm(
                                sample_curve(arc["curve"], n_points=100), part_mean, part_R, part_scale)
                            kw_pre[f"arc_color_{k}"] = [0.2, 0.85, 0.2]
                        np.savez(os.path.join(out_dir, f"arcs_pre_filter_{ei}_{ej}.npz"), **kw_pre)

                edge_arcs, vertices, vertex_edges, shape, brep_info = ilp_topology_filter(
                    edge_arcs, vertices, vertex_edges,
                    clusters, cluster_trees, cluster_nn_percentiles,
                    occ_surfaces=occ_surfs, surface_ids=surface_ids,
                    tolerance=1e-3,
                    omit_sewing=True,
                )
                if shape is not None and not shape.IsNull():
                    print_edge_arcs_summary(edge_arcs)
                    face_arcs = face_arc_incidence(edge_arcs)
                    print_face_arcs_summary(face_arcs)
                    if write_intermediates:
                        face_wires = assemble_wires(face_arcs)
                        print_face_wires_summary(face_wires)

                        arc_id_to_key = {}
                        for (ei_, ej_), arcs_list in edge_arcs.items():
                            for k_, arc_ in enumerate(arcs_list):
                                arc_id_to_key[id(arc_)] = (ei_, ej_, k_)
                        for p_, wires_p in face_wires.items():
                            wire_lengths = []
                            for wire in wires_p:
                                L = 0.0
                                for (arc_, _fwd) in wire:
                                    pts = sample_curve(arc_["curve"], n_points=100)
                                    L += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                                wire_lengths.append(L)
                            outer_idx = int(np.argmax(wire_lengths)) if wire_lengths else -1
                            kw_w = {"n_wires": len(wires_p), "outer_wire_idx": outer_idx}
                            for w_idx, wire in enumerate(wires_p):
                                kw_w[f"wire_{w_idx}_edge_i"]  = np.array(
                                    [arc_id_to_key[id(a)][0] for (a, _) in wire], dtype=np.int32)
                                kw_w[f"wire_{w_idx}_edge_j"]  = np.array(
                                    [arc_id_to_key[id(a)][1] for (a, _) in wire], dtype=np.int32)
                                kw_w[f"wire_{w_idx}_arc_idx"] = np.array(
                                    [arc_id_to_key[id(a)][2] for (a, _) in wire], dtype=np.int32)
                                kw_w[f"wire_{w_idx}_forward"] = np.array(
                                    [fwd for (_, fwd) in wire], dtype=bool)
                            np.savez(os.path.join(out_dir, f"wires_face_{p_}.npz"), **kw_w)

                if write_intermediates:
                    np.savez(os.path.join(out_dir, "vertices.npz"),
                             vertices=_denorm(vertices, part_mean, part_R, part_scale))

                    _ARC_COLORS = [[0.2, 0.85, 0.2], [1.0, 0.2, 0.2],
                                   [0.2, 0.4,  1.0], [0.8, 0.2,  0.8]]
                    for (ei, ej), arcs in edge_arcs.items():
                        kw_arcs = {"n_arcs": len(arcs), "edge_i": ei, "edge_j": ej}
                        for k, arc in enumerate(arcs):
                            kw_arcs[f"arc_points_{k}"] = _denorm(
                                sample_curve(arc["curve"], n_points=100), part_mean, part_R, part_scale)
                            kw_arcs[f"arc_color_{k}"]  = _ARC_COLORS[0]
                        np.savez(os.path.join(out_dir, f"arcs_{ei}_{ej}.npz"), **kw_arcs)

                return shape, len(clusters)

            if args.sweep_mode:
                from point2step.brep_sweep_eval import (
                    VariantRecord, shape_metrics, select_best,
                    is_brep_valid, write_sweep_sidecar,
                    sample_shape_pointcloud, export_pointcloud_ply,
                    COVERAGE_TAU,
                )

                def _fmt(v, w=8, p=5):
                    return f"{v:{w}.{p}f}" if v is not None else "-".rjust(w)

                def _summary(rec):
                    if rec is None:
                        return "None"
                    return (f"spacing_factor={rec.spacing}  valid={rec.valid}"
                            f"  cov_n={_fmt(rec.coverage_norm)}"
                            f"  fid_n={_fmt(rec.fidelity_norm)}"
                            f"  cov_w={_fmt(rec.coverage_world)}"
                            f"  fid_w={_fmt(rec.fidelity_world)}")

                # tau is in normalized units; the world metric uses tau * part_scale so coverage is equivalent across frames
                tau_norm  = COVERAGE_TAU
                tau_world = COVERAGE_TAU * float(part_scale)

                records       = []
                shapes_by_sf  = {}
                for sf in args.sweep_values:
                    print(f"\n[sweep] ----- spacing_factor = {sf} -----")
                    shape_v, _n_clusters_v = _run_variant(
                        sf, write_intermediates=False,
                    )
                    cov_n, fid_n = (None, None)
                    if shape_v is not None and not shape_v.IsNull():
                        cov_n, fid_n = shape_metrics(
                            shape_v, clusters_master_concat_norm,
                            coverage_tau=tau_norm,
                            n_samples=args.step_sample_count,
                            rng=np_rng)
                    valid_v = is_brep_valid(shape_v)
                    rec = VariantRecord(
                        spacing       = sf,
                        sampled_ok       = (cov_n is not None),
                        valid         = valid_v,
                        coverage_norm = cov_n,
                        fidelity_norm = fid_n,
                    )
                    if rec.sampled_ok:
                        try:
                            sw = apply_inverse_normalization(
                                shape_v, part_mean, part_R, part_scale)
                            rec.coverage_world, rec.fidelity_world = shape_metrics(
                                sw, clusters_master_concat_world,
                                coverage_tau=tau_world,
                                n_samples=args.step_sample_count,
                                rng=np_rng)
                        except Exception as e:
                            print(f"[sweep] world-frame eval failed at sf={sf}: {e}")
                    records.append(rec)
                    shapes_by_sf[sf] = shape_v

                    print(f"[sweep] spacing_factor={sf}  valid={valid_v}  sampled_ok={rec.sampled_ok}"
                          f"  cov_n={_fmt(rec.coverage_norm)}  fid_n={_fmt(rec.fidelity_norm)}"
                          f"  cov_w={_fmt(rec.coverage_world)}  fid_w={_fmt(rec.fidelity_world)}")

                if len(clusters_master) == 1 and not any(r.sampled_ok for r in records):
                    print("[sweep] single-cluster fallback: building UV-bounded "
                          "face from the fitted surface")
                    fb_face = make_uv_bounded_face(
                        occ_surfs_master[0], clusters_master[0],
                        uv_margin=0.05, tolerance=1e-3,
                    )
                    if fb_face is not None and not fb_face.IsNull():
                        bb_fb = BRep_Builder()
                        fb_shape = TopoDS_Compound()
                        bb_fb.MakeCompound(fb_shape)
                        bb_fb.Add(fb_shape, fb_face)
                        cov_n_fb, fid_n_fb = shape_metrics(
                            fb_shape, clusters_master_concat_norm,
                            coverage_tau=tau_norm,
                            n_samples=args.step_sample_count,
                            rng=np_rng)
                        valid_fb = is_brep_valid(fb_shape)
                        fb_rec = VariantRecord(
                            spacing       = 1.0,
                            sampled_ok    = (cov_n_fb is not None),
                            valid         = valid_fb,
                            coverage_norm = cov_n_fb,
                            fidelity_norm = fid_n_fb,
                        )
                        if fb_rec.sampled_ok:
                            try:
                                sw_fb = apply_inverse_normalization(
                                    fb_shape, part_mean, part_R, part_scale)
                                fb_rec.coverage_world, fb_rec.fidelity_world = shape_metrics(
                                    sw_fb, clusters_master_concat_world,
                                    coverage_tau=tau_world,
                                    n_samples=args.step_sample_count,
                                    rng=np_rng)
                            except Exception as e:
                                print(f"[sweep] fallback world-frame eval failed: {e}")
                        records      = [fb_rec]
                        shapes_by_sf = {1.0: fb_shape}
                        print(f"[sweep] fallback: sampled_ok={fb_rec.sampled_ok}  "
                              f"valid={valid_fb}  cov_n={_fmt(fb_rec.coverage_norm)}  "
                              f"fid_n={_fmt(fb_rec.fidelity_norm)}")
                    else:
                        print("[sweep] single-cluster fallback: MakeFace failed — "
                              "no STEP will be written")

                best_overall = select_best(records, frame=args.selection_frame,
                                           require_valid=False)
                best_valid   = select_best(records, frame=args.selection_frame,
                                           require_valid=True)
                sweep_summaries.append({
                    "part_idx":     part_idx,
                    "records":      list(records),
                    "best_overall": best_overall,
                    "best_valid":   best_valid,
                    "tau_norm":     tau_norm,
                })
                print(f"\n[sweep] best_overall: {_summary(best_overall)}")
                print(f"[sweep] best_valid:   {_summary(best_valid)}")

                if best_overall is not None and not args.no_intermediates:
                    print(f"[sweep] persisting intermediates for best_overall "
                          f"(spacing_factor={best_overall.spacing})")
                    _run_variant(best_overall.spacing, write_intermediates=True)

                shape_world_overall = None
                shape_world_valid   = None
                for rec, slot in ((best_overall, "overall"), (best_valid, "valid")):
                    if rec is None:
                        continue
                    shp_norm = shapes_by_sf[rec.spacing]
                    try:
                        shp_world = apply_inverse_normalization(
                            shp_norm, part_mean, part_R, part_scale)
                    except Exception as e:
                        print(f"[sweep] apply_inverse_normalization failed for {slot}: {e}")
                        shp_world = shp_norm
                    if rec.coverage_world is None and rec.sampled_ok:
                        rec.coverage_world, rec.fidelity_world = shape_metrics(
                            shp_world, clusters_master_concat_world,
                            coverage_tau=tau_world,
                            n_samples=args.step_sample_count,
                            rng=np_rng)
                    if slot == "overall":
                        shape_world_overall = shp_world
                    else:
                        shape_world_valid = shp_world

                step_path_overall = os.path.join(out_dir, f"{step_stem}.step")
                if shape_world_overall is not None and not shape_world_overall.IsNull():
                    export_step(shape_world_overall, step_path_overall)
                if args.save_norm_step and best_overall is not None:
                    shp_norm = shapes_by_sf[best_overall.spacing]
                    if shp_norm is not None and not shp_norm.IsNull():
                        export_step(shp_norm, os.path.join(out_dir, f"{step_stem}_norm.step"))
                if best_valid is not None and best_overall is not None \
                        and best_valid.spacing != best_overall.spacing:
                    if shape_world_valid is not None and not shape_world_valid.IsNull():
                        export_step(shape_world_valid, os.path.join(out_dir, f"{step_stem}_valid.step"))
                    if args.save_norm_step:
                        shp_norm = shapes_by_sf[best_valid.spacing]
                        if shp_norm is not None and not shp_norm.IsNull():
                            export_step(shp_norm, os.path.join(out_dir, f"{step_stem}_valid_norm.step"))

                write_sweep_sidecar(
                    os.path.join(out_dir, "sweep_eval.json"),
                    spacing_grid    = [r.spacing for r in records],
                    selection_frame = args.selection_frame,
                    records         = records,
                    best_overall    = best_overall,
                    best_valid      = best_valid,
                )

                if args.debug_save_step_pc:
                    if best_overall is not None and shape_world_overall is not None \
                            and not shape_world_overall.IsNull():
                        pts = sample_shape_pointcloud(
                            shape_world_overall, n_total=args.step_sample_count,
                            rng=np_rng)
                        if pts is not None and len(pts) > 0:
                            export_pointcloud_ply(
                                pts, os.path.join(out_dir, f"{step_stem}_step_pc.ply"))
                            print(f"[sweep] debug: wrote {len(pts)} step-pc points → "
                                  f"{step_stem}_step_pc.ply")
                    if best_valid is not None and best_overall is not None \
                            and best_valid.spacing != best_overall.spacing \
                            and shape_world_valid is not None \
                            and not shape_world_valid.IsNull():
                        pts = sample_shape_pointcloud(
                            shape_world_valid, n_total=args.step_sample_count,
                            rng=np_rng)
                        if pts is not None and len(pts) > 0:
                            export_pointcloud_ply(
                                pts, os.path.join(out_dir, f"{step_stem}_valid_step_pc.ply"))

                shape_world = shape_world_overall
                n_clusters_for_offset = len(clusters_master)
            else:
                shape, n_clusters_v = _run_variant(
                    args.spacing_factor,
                    write_intermediates=not args.no_intermediates,
                )

                step_path = os.path.join(out_dir, f"{step_stem}.step")
                shape_world = None
                if shape is None or shape.IsNull():
                    print(f"[brep] build failed — 0 faces produced, skipping STEP export")
                else:
                    try:
                        shape_world = apply_inverse_normalization(shape, part_mean, part_R, part_scale)
                    except Exception as e:
                        print(f"[brep] apply_inverse_normalization failed: {e} — exporting normalized shape")
                        shape_world = shape
                    export_step(shape_world, step_path)
                    if args.save_norm_step:
                        export_step(shape, os.path.join(out_dir, f"{step_stem}_norm.step"))

                n_clusters_for_offset = n_clusters_v

            print(f"\n[part] all results saved to {out_dir}/")

            if args.model_id:
                part_dirs.append((out_dir, cluster_offset, n_clusters_for_offset, shape_world))
                cluster_offset += n_clusters_for_offset
        except Exception:
            failed = True
            print(f"[brep] part {part_idx} FAILED (see stderr for traceback)",
                  flush=True)
            traceback.print_exc()
        if args.write_part_status:
            _write_part_status(out_dir, "failed" if failed else "ok")

    if part_dirs:
        unified_dir = os.path.join(model_out_dir, "unified")
        os.makedirs(unified_dir, exist_ok=True)
        suppress_intermediates = args.no_intermediates
        if not suppress_intermediates:
            _merge_part_dirs(
                [(d, o, n) for d, o, n, _ in part_dirs],
                unified_dir,
            )
        from OCC.Core.BRep import BRep_Builder as _BB
        from OCC.Core.TopoDS import TopoDS_Compound as _TC
        builder = _BB()
        compound = _TC()
        builder.MakeCompound(compound)
        for _, _, _, sw in part_dirs:
            if sw is not None and not sw.IsNull():
                builder.Add(compound, sw)
        export_step(compound, os.path.join(unified_dir, "unified.step"))

    if sweep_summaries:
        _print_sweep_summaries(sweep_summaries, args.selection_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-Rep reconstruction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Load saved results and visualize (host, no OCC needed)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Model ID for multi-part mode (compute: globs "
                             "input_dir/model_id/*_part*.xyzc; visualize: loads "
                             "output_dir/model_id/unified/)")
    parser.add_argument("--input_dir", type=str, default="sample_clouds_abc_parts",
                        help="Root directory for per-model point cloud subdirs "
                             "(multi-part compute mode)")
    parser.add_argument("--output_dir", type=str, default="output_step",
                        help="Directory for saved results")
    parser.add_argument("--seed", type=int, default=41,
                        help="Reproducibility seed")
    parser.add_argument("--inr_max_steps", type=int, default=INR_MAX_STEPS,
                        help="Number of training steps per INR (u,v) closedness combo")
    parser.add_argument("--spacing_percentile", type=float, default=100.0,
                    help="Percentile of intra-cluster NN distance distribution "
                            "used for local spacing in adjacency computation "
                            "(default 100.0 = max NN distance).")
    parser.add_argument("--spacing_factor", type=float, default=None,
                        help="Adjacency detection threshold = spacing factor * "
                             "per-pair local spacing. Passing it runs this single "
                             "value instead of the default sweep.")
    parser.add_argument("--fitness_threshold", type=float, default=DEFAULT_FITNESS_THRESHOLD,
                        help="Max fitness score (d/p) for vertices and arcs in "
                             "pre-filter. Elements with score above this are "
                             "discarded. Score is scale-invariant: 1.0 = cluster "
                             "boundary.")
    parser.add_argument("--vertex_threshold", type=float, default=DEFAULT_VERTEX_THRESHOLD,
                        help="Absolute distance threshold for "
                             "GeomAPI_ExtremaCurveCurve vertex detection. "
                             "Used both as the extremum acceptance gate "
                             "(d ≤ threshold counts as an intersection) and "
                             "as the spatial dedup radius. In normalized "
                             "unit-cube coordinates.")
    parser.add_argument("--part", type=int, default=None,
                        help="Process only this part index (0-based). "
                             "Default: process all parts.")
    parser.add_argument("--spacing_factor_min", type=float, default=DEFAULT_SPACING_FACTOR_MIN,
                        help="Lower bound of the spacing-factor sweep (inclusive). "
                             "The sweep is the default mode and reproduces the "
                             "evaluation regime; pass --spacing_factor to run a "
                             "single value instead.")
    parser.add_argument("--spacing_factor_max", type=float, default=DEFAULT_SPACING_FACTOR_MAX,
                        help="Upper bound of the spacing-factor sweep (inclusive). "
                             "See --spacing_factor_min.")
    parser.add_argument("--spacing_factor_step", type=float, default=DEFAULT_SPACING_FACTOR_STEP,
                        help="Decrement step for the spacing-factor sweep. "
                             "Sweep iterates from max down to min, inclusive.")
    parser.add_argument("--selection_frame", type=str, default=DEFAULT_SELECTION_FRAME,
                        choices=["normalized", "world"],
                        help="Frame used for the (coverage, fidelity) selection "
                             "in sweep mode. 'normalized' (default) is the "
                             "evaluation default. 'world' selects the canonical "
                             "STEP by metrics against the un-normalized cloud, "
                             "appropriate for production use. Both frames are "
                             "computed and recorded in sweep_eval.json regardless.")
    parser.add_argument("--save_norm_step", action="store_true",
                        help="Also write part_{i}_norm.step (pre-inverse-"
                             "normalization, eval-only) alongside the canonical "
                             "world-space STEP. Active in sweep mode and single-"
                             "value mode alike.")
    parser.add_argument("--no_intermediates", action="store_true",
                        help="Suppress per-part scratch (arcs_*.npz, cluster_*.npy, "
                             "wires_face_*.npz, inter_*.npz, metadata.npz). STEP "
                             "files and sweep_eval.json are still written. Active "
                             "in sweep mode and single-value mode alike.")
    parser.add_argument("--step_sample_count", type=int, default=DEFAULT_STEP_SAMPLE_COUNT,
                        help="Number of points sampled from the assembled BRep "
                             "for sweep-mode metric computation (area-weighted "
                             "across faces, respecting trimmed boundaries). "
                             "Matches the mesh-pipeline N=30000 convention. "
                             "Used only in sweep mode.")
    parser.add_argument("--debug_save_step_pc", action="store_true",
                        help="Sweep mode only: also write a PLY of the sampled "
                             "STEP-side point cloud for best_overall (and "
                             "best_valid when different) so it can be loaded "
                             "alongside the canonical STEP for visual "
                             "verification. Temporary diagnostic; off by default.")
    parser.add_argument("--no_clean_output", action="store_true",
                        help="Skip wiping the per-model output directory at "
                             "startup. Use when an external invoker (e.g. a "
                             "wrapper script) has already prepared a clean "
                             "model_dir and placed files (such as log files) "
                             "that should be preserved.")
    parser.add_argument("--write_part_status", action="store_true",
                        help="Write per-part wrapper_status.json ('ok' or "
                             "'failed') after each part is processed. Used by "
                             "run_step_eval.py for resume bookkeeping. Default "
                             "off so direct invocations don't litter the output "
                             "tree with wrapper-only files.")
    args = parser.parse_args()

    if args.spacing_factor is not None:
        args.sweep_values = [args.spacing_factor]
        args.sweep_mode = False
    else:
        if args.spacing_factor_step <= 0:
            parser.error("--spacing_factor_step must be positive.")
        if args.spacing_factor_max < args.spacing_factor_min:
            parser.error(
                "--spacing_factor_max must be >= --spacing_factor_min."
            )
        eps = 1e-9
        vals = []
        v = float(args.spacing_factor_max)
        while v >= float(args.spacing_factor_min) - eps:
            vals.append(round(v, 6))
            v -= float(args.spacing_factor_step)
        args.sweep_values = vals
        args.sweep_mode = True

    if args.model_id is None:
        parser.error("--model_id is required")

    if args.visualize:
        run_visualize(args)
    else:
        run_compute(args)

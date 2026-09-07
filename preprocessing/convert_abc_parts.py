import argparse
import os
import glob
import time

import json
import h5py
import numpy as np
from collections import Counter

# HPNet / ParseNet primitive type codes.
# Analytical types (T_param filled by the ABCParts authors):
#   1 = Plane,  3 = Cone,  4 = Cylinder,  5 = Sphere
# Non-analytical types (skipped, T_param stays zero):
#   0, 6, 7 → closed spline;  2, 8 → open spline;  9 → closed spline
PRIM_NAMES = {
    0: "Spline",
    1: "Plane",
    2: "Spline",
    3: "Cone",
    4: "Cylinder",
    5: "Sphere",
    6: "Spline",
    7: "Spline",
    8: "Spline",
    9: "Spline",
}

# T_param layout (22 dims per point, slots filled only for analytical types):
#   [ 0: 4] Sphere(5)   — center(3) + radius(1)
#   [ 4: 8] Plane(1)    — normal(3) + offset(1)
#   [ 8:15] Cylinder(4) — axis(3) + center(3) + radius(1)
#   [15:22] Cone(3)     — axis(3) + center(3) + half_angle(1)


def convert_one(h5_path, output_dir):
    model_id = os.path.splitext(os.path.basename(h5_path))[0]
    model_dir = os.path.join(output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        points = np.array(f["points"], dtype=np.float64)       # (N, 3)
        labels = np.array(f["labels"], dtype=np.float64)       # (N,)
        normals = np.array(f["normals"], dtype=np.float64)     # (N, 3)
        prim = np.array(f["prim"], dtype=np.int32)             # (N,)
        T_param = np.array(f["T_param"], dtype=np.float64)     # (N, 22)

    data = np.column_stack([points, labels])
    np.savetxt(os.path.join(model_dir, "0.xyzc"), data)

    np.savez_compressed(
        os.path.join(model_dir, "metadata.npz"),
        normals=normals,
        prim=prim,
        T_param=T_param,
    )

    labels_int = labels.astype(int)
    unique_labels, counts = np.unique(labels_int, return_counts=True)
    n_clusters = len(unique_labels)

    clusters = []
    for lbl, cnt in zip(unique_labels, counts):
        mask = labels_int == lbl
        prim_counts = Counter(prim[mask].tolist())
        dominant_id = prim_counts.most_common(1)[0][0]
        clusters.append({
            "cluster_id": int(lbl),
            "n_points": int(cnt),
            "primitive_type": PRIM_NAMES.get(dominant_id, f"Unknown({dominant_id})"),
            "primitive_type_id": int(dominant_id),
        })

    prim_summary = Counter(c["primitive_type"] for c in clusters)

    info = {
        "model_id": model_id,
        "n_points": len(points),
        "n_clusters": n_clusters,
        "pts_per_cluster": {
            "mean": round(float(np.mean(counts)), 1),
            "median": round(float(np.median(counts)), 1),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        },
        "primitive_summary": dict(prim_summary),
        "clusters": clusters,
    }

    with open(os.path.join(model_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


def visualize(model_id, output_dir):
    import open3d as o3d

    model_dir = os.path.join(output_dir, model_id)
    xyzc_path = os.path.join(model_dir, "0.xyzc")
    info_path = os.path.join(model_dir, "info.json")

    if not os.path.exists(xyzc_path):
        print(f"File not found: {xyzc_path}")
        return

    data = np.loadtxt(xyzc_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    points = data[:, :3]
    labels = data[:, 3].astype(int)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    rng = np.random.default_rng(42)
    palette = rng.random((n_clusters, 3))
    label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    colors = np.array([label_to_color[lbl] for lbl in labels])

    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        stats = info["pts_per_cluster"]
        print(f"Model {model_id}: {info['n_points']} points, "
              f"{info['n_clusters']} clusters")
        print(f"  Primitives: {info['primitive_summary']}")
        print(f"  Points/cluster: mean={stats['mean']}  "
              f"median={stats['median']}  "
              f"min={stats['min']}  max={stats['max']}")
        print(f"  {'Cluster':>8}  {'Points':>7}  {'Type':<12}")
        print(f"  {'─'*8}  {'─'*7}  {'─'*12}")
        for c in info["clusters"]:
            print(f"  {c['cluster_id']:>8}  {c['n_points']:>7}  "
                  f"{c['primitive_type']:<12}")
    else:
        print(f"Model {model_id}: {len(points)} points, {n_clusters} clusters")
        print(f"  (no info.json found, run conversion first)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"ABCParts {model_id}", width=900, height=720)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0

    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Convert ABCParts .h5 files to .xyzc for step_pipeline.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str,
                        default="../abc_parts/ABC_final",
                        help="Directory containing *.h5 files")
    parser.add_argument("--output_dir", type=str,
                        default="sample_clouds_abc_parts",
                        help="Output directory for .xyzc files")
    parser.add_argument("--visualize", type=str, default=None, metavar="MODEL_ID",
                        help="Visualize an already-converted model (e.g. 00010)")
    args = parser.parse_args()

    if args.visualize is not None:
        visualize(args.visualize, args.output_dir)
        return

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Found {len(h5_files)} .h5 files in {args.data_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_infos = []
    for i, h5_path in enumerate(h5_files):
        info = convert_one(h5_path, args.output_dir)
        all_infos.append(info)
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{i+1}/{len(h5_files)}] {info['model_id']}: "
                  f"{info['n_points']} pts, {info['n_clusters']} clusters, "
                  f"prims={info['primitive_summary']}")

    all_n_clusters = [d["n_clusters"] for d in all_infos]
    all_mins = [d["pts_per_cluster"]["min"] for d in all_infos]
    all_maxs = [d["pts_per_cluster"]["max"] for d in all_infos]
    all_means = [d["pts_per_cluster"]["mean"] for d in all_infos]
    total_prims = Counter()
    for d in all_infos:
        total_prims.update(d["primitive_summary"])

    print(f"\n{'='*60}")
    print(f"Converted {len(h5_files)} models → {args.output_dir}")
    print(f"{'='*60}")
    print(f"Clusters per model : mean={np.mean(all_n_clusters):.1f}  "
          f"median={np.median(all_n_clusters):.0f}  "
          f"min={np.min(all_n_clusters)}  max={np.max(all_n_clusters)}")
    print(f"Points per cluster : mean={np.mean(all_means):.0f}  "
          f"median(of mins)={np.median(all_mins):.0f}  "
          f"min={np.min(all_mins)}  max={np.max(all_maxs)}")
    print(f"Primitive types (total clusters):")
    for ptype, count in total_prims.most_common():
        print(f"  {ptype:<20} {count:>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

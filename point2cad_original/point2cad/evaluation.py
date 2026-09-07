"""
Mesh-only fidelity metrics for comparing two CAD-reconstruction pipelines on
the same input point cloud. See notes/cad_metrics.md for the formal definitions
and rationale.

Pure numpy / scipy / trimesh. No project-local imports — this file is intended
to be copy-pasted verbatim into the original Point2CAD repo and invoked from
its main pipeline.

Metrics (all computed in normalized space; both pipelines apply their own
per-part PCA + unit-cube normalization):
  - P-coverage (global, union-of-meshes)
  - Per-cluster residual error (point -> clipped cluster mesh)
  - Symmetric Chamfer distance (PC <-> union of clipped meshes)

Timing / classification metadata is passed through as-is.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PRIMITIVE_TYPES = frozenset({"plane", "sphere", "cylinder", "cone"})


def _build_union(cluster_meshes):
    """Concatenate all non-empty cluster meshes into one trimesh.Trimesh.

    Returns None if every cluster mesh is empty / missing.
    """
    Vs, Fs, offset = [], [], 0
    for m in cluster_meshes.values():
        if m is None:
            continue
        V = np.asarray(m.vertices)
        F = np.asarray(m.faces)
        if len(F) == 0 or len(V) == 0:
            continue
        Vs.append(V)
        Fs.append(F + offset)
        offset += len(V)
    if not Vs:
        return None
    V_union = np.vstack(Vs)
    F_union = np.vstack(Fs)
    return trimesh.Trimesh(V_union, F_union, process=False)


def _dist_pts_to_mesh(points, mesh):
    """Euclidean distance from each point to the surface of `mesh`.

    Uses trimesh.proximity.closest_point (AABB-accelerated, exact point-to-
    triangle). Returns None if either argument is empty.
    """
    if mesh is None or len(mesh.faces) == 0 or len(points) == 0:
        return None
    _, d, _ = trimesh.proximity.closest_point(mesh, points)
    return d


def _per_cluster_residual(input_points, input_labels, cluster_meshes):
    """Mean distance from each labelled cluster's input points to that
    cluster's clipped mesh.  Missing / empty meshes -> None.
    """
    out = {}
    for cid, mesh in cluster_meshes.items():
        pts = input_points[input_labels == cid]
        d = _dist_pts_to_mesh(pts, mesh)
        out[cid] = float(d.mean()) if d is not None else None
    return out


def _mesh_to_pc_distances(union_mesh, input_points, n_samples, rng):
    """Per-sample distance from uniform-area mesh samples to the input cloud.

    Shared primitive for Chamfer(Mesh->PC) and reverse P-coverage. `sample_-
    surface_even` is the Poisson-disk variant (lower variance) but may return
    fewer than `n_samples` after rejection — fine for aggregate estimators.
    """
    if union_mesh is None or len(union_mesh.faces) == 0 or len(input_points) == 0:
        return None
    seed = int(rng.integers(np.iinfo(np.int32).max))
    try:
        samples, _ = trimesh.sample.sample_surface_even(
            union_mesh, n_samples, seed=seed
        )
        if len(samples) == 0:
            samples, _ = trimesh.sample.sample_surface(union_mesh, n_samples, seed=seed)
    except TypeError:
        samples, _ = trimesh.sample.sample_surface_even(union_mesh, n_samples)
        if len(samples) == 0:
            samples, _ = trimesh.sample.sample_surface(union_mesh, n_samples)
    tree = cKDTree(input_points)
    d, _ = tree.query(samples, k=1)
    return d


def compute_part_metrics(
    input_points,
    input_labels,
    cluster_meshes,
    surface_types=None,
    timing=None,
    n_chamfer_samples=30_000,
    coverage_radius=0.01,
    model_id="",
    part_idx=0,
    seed=0,
):
    """Compute all mesh-only metrics for one part.

    Parameters
    ----------
    input_points : (N, 3) float array, normalized space.
    input_labels : (N,) int array of cluster ids.
    cluster_meshes : dict[int, trimesh.Trimesh]
        Clipped cluster meshes in normalized space. Clusters that failed to
        produce a mesh may be omitted or mapped to None.
    surface_types : dict[int, str] | None
        Per-cluster surface label ("plane", "sphere", "cylinder", "cone", or
        anything else -> freeform). Pass None if the pipeline doesn't track
        types (e.g. original Point2CAD).
    timing : dict[str, float] | None
        Pipeline-level timing (fit_time, clip_time, etc.).  Passed through
        verbatim into the output JSON.
    n_chamfer_samples : int
        Number of uniform-area samples on the union mesh for Chamfer Mesh->PC.
    coverage_radius : float
        P-coverage threshold r.
    model_id, part_idx : identifiers stored in the output.
    seed : int — for reproducible surface sampling.

    Returns
    -------
    dict — JSON-ready. Caller is responsible for writing to disk.
    """
    rng = np.random.default_rng(seed)
    input_points = np.asarray(input_points, dtype=np.float64)
    input_labels = np.asarray(input_labels, dtype=np.int64)

    union_mesh = _build_union(cluster_meshes)

    # PC -> Mesh: shared distance vector for P-coverage (threshold) and
    # Chamfer PC->Mesh (mean).
    d_pc_to_mesh = _dist_pts_to_mesh(input_points, union_mesh)
    if d_pc_to_mesh is not None:
        p_coverage = float((d_pc_to_mesh <= coverage_radius).mean())
        chamfer_pc_to_mesh = float(d_pc_to_mesh.mean())
    else:
        p_coverage = 0.0
        chamfer_pc_to_mesh = None

    # Mesh -> PC: reverse direction. Same threshold for the hallucination-rate
    # analog (fraction of mesh surface samples within r of the input cloud);
    # the complement, 1 - p_coverage_mesh_to_pc, is the share of mesh area
    # that lives far from any input point (spurious lobes / overfitting bumps).
    d_mesh_to_pc = _mesh_to_pc_distances(
        union_mesh, input_points, n_chamfer_samples, rng
    )
    if d_mesh_to_pc is not None:
        chamfer_mesh_to_pc = float(d_mesh_to_pc.mean())
        p_coverage_mesh_to_pc = float((d_mesh_to_pc <= coverage_radius).mean())
    else:
        chamfer_mesh_to_pc = None
        p_coverage_mesh_to_pc = None

    if chamfer_pc_to_mesh is not None and chamfer_mesh_to_pc is not None:
        chamfer_sym = 0.5 * (chamfer_pc_to_mesh + chamfer_mesh_to_pc)
    else:
        chamfer_sym = None

    residuals = _per_cluster_residual(input_points, input_labels, cluster_meshes)

    # Classification: primitive-only iff every tracked cluster has a primitive
    # surface type. Unknown types (anything outside PRIMITIVE_TYPES) count as
    # freeform. None -> we don't know (Point2CAD side).
    if surface_types is None:
        is_primitive_only = None
    else:
        is_primitive_only = bool(
            len(surface_types) > 0
            and all(s in PRIMITIVE_TYPES for s in surface_types.values())
        )

    cluster_records = []
    for cid in sorted(cluster_meshes.keys()):
        mesh = cluster_meshes.get(cid)
        cluster_records.append({
            "id": int(cid),
            "n_points": int((input_labels == cid).sum()),
            "surface_type": (surface_types.get(cid)
                             if surface_types is not None else None),
            "residual": residuals.get(cid),
            "mesh_missing": mesh is None or len(mesh.faces) == 0,
        })

    valid_residuals = [r for r in residuals.values() if r is not None]
    residual_mean = float(np.mean(valid_residuals)) if valid_residuals else None

    return {
        "model_id": str(model_id),
        "part_idx": int(part_idx),
        "n_points": int(len(input_points)),
        "n_clusters": len(cluster_meshes),
        "clusters": cluster_records,
        "is_primitive_only": is_primitive_only,
        "timing": dict(timing) if timing is not None else {},
        "metrics": {
            "p_coverage": p_coverage,
            "p_coverage_mesh_to_pc": p_coverage_mesh_to_pc,
            "coverage_radius": float(coverage_radius),
            "residual_mean": residual_mean,
            "chamfer_pc_to_mesh": chamfer_pc_to_mesh,
            "chamfer_mesh_to_pc": chamfer_mesh_to_pc,
            "chamfer_sym": chamfer_sym,
            "n_chamfer_samples": int(n_chamfer_samples),
        },
    }

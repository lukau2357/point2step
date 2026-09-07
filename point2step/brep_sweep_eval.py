from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


COVERAGE_TAU       = 0.01     # PC -> STEP coverage threshold, normalized units
DEFAULT_N_SAMPLES  = 30000    # target STEP-side point count (area-weighted across faces)
DEFAULT_GRID_RES   = 50       # per-face UV grid resolution (cells per axis)


@dataclass
class VariantRecord:
    spacing: float
    sampled_ok: bool
    valid: bool
    coverage_norm: Optional[float] = None
    fidelity_norm: Optional[float] = None
    coverage_world: Optional[float] = None
    fidelity_world: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "spacing": self.spacing,
            "sampled_ok": self.sampled_ok,
            "valid": self.valid,
            "coverage_norm": self.coverage_norm,
            "fidelity_norm": self.fidelity_norm,
            "coverage_world": self.coverage_world,
            "fidelity_world": self.fidelity_world,
        }


def _evaluate_face_grid(geom_surf, us, vs):
    pts = np.zeros((len(us), len(vs), 3), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            p = geom_surf.Value(float(u), float(v))
            pts[i, j] = (p.X(), p.Y(), p.Z())
    return pts


def _face_triangle_areas(pts3d):
    A = pts3d[:-1, :-1]
    B = pts3d[1:,  :-1]
    C = pts3d[:-1, 1:]
    D = pts3d[1:,  1:]
    area1 = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=-1)
    area2 = 0.5 * np.linalg.norm(np.cross(D - B, C - B), axis=-1)
    return area1, area2


def _make_cell_mask(face, us, vs, tol=1e-7):
    clf  = BRepClass_FaceClassifier()
    mask = np.zeros((len(us) - 1, len(vs) - 1), dtype=bool)
    for i in range(len(us) - 1):
        u_mid = 0.5 * (us[i] + us[i + 1])
        for j in range(len(vs) - 1):
            v_mid = 0.5 * (vs[j] + vs[j + 1])
            clf.Perform(face, gp_Pnt2d(u_mid, v_mid), tol)
            mask[i, j] = clf.State() in (TopAbs_IN, TopAbs_ON)
    return mask


def _sample_from_grid(pts3d, area1, area2, n_pts, rng):
    if n_pts == 0:
        return np.zeros((0, 3))
    all_areas = np.concatenate([area1.ravel(), area2.ravel()])
    total = all_areas.sum()
    if total < 1e-12:
        return np.zeros((0, 3))

    probs    = all_areas / total
    n_cells  = area1.size
    tri_ids  = rng.choice(len(all_areas), size=n_pts, p=probs)
    is_type1 = tri_ids < n_cells
    cell_ids = np.where(is_type1, tri_ids, tri_ids - n_cells)
    ci       = cell_ids // area1.shape[1]
    cj       = cell_ids %  area1.shape[1]

    r1      = rng.random(n_pts)
    sqrt_r1 = np.sqrt(r1)
    r2      = rng.random(n_pts)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2

    vA = pts3d[ci,     cj    ]
    vB = pts3d[ci + 1, cj    ]
    vC = pts3d[ci,     cj + 1]
    vD = pts3d[ci + 1, cj + 1]

    # Triangle 1: (A, B, C);  Triangle 2: (B, D, C)
    mask  = is_type1[:, None]
    vert0 = np.where(mask, vA, vB)
    vert1 = np.where(mask, vB, vD)
    return w0[:, None] * vert0 + w1[:, None] * vert1 + w2[:, None] * vC


def _face_passes_brepcheck(face) -> bool:
    return bool(BRepCheck_Analyzer(face).IsValid())


def sample_shape_pointcloud(shape,
                            n_total: int = DEFAULT_N_SAMPLES,
                            grid_res: int = DEFAULT_GRID_RES,
                            rng: Optional[np.random.Generator] = None,
                            ) -> Optional[np.ndarray]:
    if shape is None or shape.IsNull():
        return None
    if rng is None:
        rng = np.random.default_rng()

    face_data = []   # (pts3d, a1, a2, total_area)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        try:
            if _face_passes_brepcheck(face):
                adaptor = BRepAdaptor_Surface(face)
                u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
                v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
                geom_surf = BRep_Tool.Surface(face)
                us = np.linspace(u0, u1, grid_res + 1)
                vs = np.linspace(v0, v1, grid_res + 1)
                pts3d = _evaluate_face_grid(geom_surf, us, vs)
                a1, a2 = _face_triangle_areas(pts3d)
                mask = _make_cell_mask(face, us, vs)
                a1m = a1 * mask
                a2m = a2 * mask
                total_m = float(a1m.sum() + a2m.sum())
                if total_m >= 1e-12:
                    face_data.append((pts3d, a1m, a2m, total_m))
        except Exception:
            pass
        exp.Next()

    if not face_data:
        return None

    A_total = sum(fd[3] for fd in face_data)
    if A_total < 1e-12:
        return None

    chunks = []
    for pts3d, a1, a2, total in face_data:
        n_face = int(round(n_total * total / A_total))
        if n_face <= 0:
            continue
        try:
            pts = _sample_from_grid(pts3d, a1, a2, n_face, rng)
            if len(pts):
                chunks.append(pts)
        except Exception:
            pass

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def export_pointcloud_ply(points: np.ndarray, out_path: str) -> None:
    n = int(len(points))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def shape_metrics(shape, cluster_points: np.ndarray,
                  coverage_tau: float = COVERAGE_TAU,
                  n_samples: int = DEFAULT_N_SAMPLES,
                  grid_res: int = DEFAULT_GRID_RES,
                  rng: Optional[np.random.Generator] = None,
                  ) -> tuple[Optional[float], Optional[float]]:
    step_pts = sample_shape_pointcloud(shape, n_total=n_samples,
                                       grid_res=grid_res, rng=rng)
    if step_pts is None or len(step_pts) == 0:
        return None, None

    tree_step = cKDTree(step_pts)
    d_pc, _ = tree_step.query(cluster_points, k=1)
    coverage = float(np.mean(d_pc <= coverage_tau))

    tree_cluster = cKDTree(cluster_points)
    d_step, _ = tree_cluster.query(step_pts, k=1)
    fidelity = float(np.mean(d_step))

    return coverage, fidelity


def select_best(records: list[VariantRecord], frame: str,
                require_valid: bool = False) -> Optional[VariantRecord]:
    if frame == "normalized":
        cov_key, fid_key = "coverage_norm", "fidelity_norm"
    elif frame == "world":
        cov_key, fid_key = "coverage_world", "fidelity_world"
    else:
        raise ValueError(f"unknown selection frame: {frame!r}")

    pool = []
    for r in records:
        if not r.sampled_ok:
            continue
        if require_valid and not r.valid:
            continue
        cov = getattr(r, cov_key)
        fid = getattr(r, fid_key)
        if cov is None or fid is None:
            continue
        pool.append(r)
    if not pool:
        return None
    # Strict lexicographic: maximize coverage, then minimize fidelity.
    pool.sort(key=lambda r: (-getattr(r, cov_key), getattr(r, fid_key)))
    return pool[0]


def write_sweep_sidecar(out_path: str,
                        spacing_grid: list[float],
                        selection_frame: str,
                        records: list[VariantRecord],
                        best_overall: Optional[VariantRecord],
                        best_valid: Optional[VariantRecord]) -> None:
    payload = {
        "spacing_grid":     list(spacing_grid),
        "selection_frame":  selection_frame,
        "variants":         [r.to_dict() for r in records],
        "best_overall":     best_overall.to_dict() if best_overall is not None else None,
        "best_valid":       best_valid.to_dict()   if best_valid   is not None else None,
    }
    tmp_path = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)


def is_brep_valid(shape) -> bool:
    if shape is None or shape.IsNull():
        return False
    try:
        return bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception:
        return False

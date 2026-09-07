import time
import numpy as np

from .surface_types import SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR
from OCC.Core.gp      import gp_Ax3, gp_Pnt, gp_Dir, gp_Pln, gp_Sphere, gp_Cylinder, gp_Cone
from OCC.Core.Geom    import Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface, Geom_ConicalSurface
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.TColgp  import TColgp_Array2OfPnt

def _make_ax3(origin, main_dir):
    main_dir = np.asarray(main_dir, dtype = np.float64)
    main_dir = main_dir / np.linalg.norm(main_dir)
    return gp_Ax3(
        gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
        gp_Dir(float(main_dir[0]), float(main_dir[1]), float(main_dir[2]))
    )

def _grid_axis_closed(xyz_grid, axis, tol = 1e-6):
    if axis == 0:
        return bool(np.allclose(xyz_grid[0], xyz_grid[-1], atol = tol))
    return bool(np.allclose(xyz_grid[:, 0], xyz_grid[:, -1], atol = tol))

def _apply_periodicity(surface, u_periodic, v_periodic):
    if u_periodic:
        try:
            surface.SetUPeriodic()
        except Exception as exc:
            print(f"[inr->occ] SetUPeriodic failed: {exc}")
    if v_periodic:
        try:
            surface.SetVPeriodic()
        except Exception as exc:
            print(f"[inr->occ] SetVPeriodic failed: {exc}")

def fit_bspline_surface(xyz_grid, degree_min = 3, degree_max = 8, continuity = 2, tol3d = 1e-3,
                        u_periodic = False, v_periodic = False):
    continuity_map = {0: GeomAbs_C0, 1: GeomAbs_C1, 2: GeomAbs_C2, 3: GeomAbs_C3}

    M, N, _ = xyz_grid.shape
    points = TColgp_Array2OfPnt(1, M, 1, N)
    for i in range(M):
        for j in range(N):
            x, y, z = xyz_grid[i, j]
            points.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))

    t0 = time.time()
    approx = GeomAPI_PointsToBSplineSurface(points, degree_min, degree_max, continuity_map[continuity], tol3d)
    fitting_time = time.time() - t0

    if not approx.IsDone():
        raise RuntimeError(f"GeomAPI_PointsToBSplineSurface failed (grid={M}x{N}, deg=[{degree_min},{degree_max}], tol={tol3d})")

    surface = approx.Surface()
    _apply_periodicity(surface, u_periodic, v_periodic)
    return surface, fitting_time

def plane_to_occ(params):
    a = np.asarray(params["a"], dtype = np.float64)
    a = a / np.linalg.norm(a)
    d = float(params["d"])

    # Point on plane: p = d * a (from a . x = d with unit normal a)
    p = d * a
    return Geom_Plane(gp_Pln(
        gp_Pnt(float(p[0]), float(p[1]), float(p[2])),
        gp_Dir(float(a[0]), float(a[1]), float(a[2]))
    ))


def sphere_to_occ(params):
    center = np.asarray(params["center"], dtype = np.float64)
    # Ax3 orientation is arbitrary for a sphere
    ax3 = _make_ax3(center, np.array([0.0, 0.0, 1.0]))
    return Geom_SphericalSurface(gp_Sphere(ax3, float(params["radius"])))


def cylinder_to_occ(params):
    ax3 = _make_ax3(np.asarray(params["center"], dtype = np.float64),
                    np.asarray(params["a"], dtype = np.float64))
    return Geom_CylindricalSurface(gp_Cylinder(ax3, float(params["radius"])))


def cone_to_occ(params, cluster = None):
    axis   = np.asarray(params["a"], dtype = np.float64)
    vertex = np.asarray(params["v"], dtype = np.float64)
    theta  = float(params["theta"])

    axis = axis / np.linalg.norm(axis)

    # fit_cone solves a double-cone equation, so orient the axis toward the majority of cluster points; Geom_ConicalSurface covers only the positive-Z nappe
    if cluster is not None:
        proj = (cluster - vertex) @ axis
        if np.sum(proj < 0) > np.sum(proj > 0):
            axis = -axis

    ax3 = _make_ax3(vertex, axis)
    return Geom_ConicalSurface(gp_Cone(ax3, theta, 0.0))

def inr_to_occ(params, grid_resolution = 50, degree_min = 3, degree_max = 8, continuity = 2, tol3d = 1e-3, uv_margin = 0, periodic = False):
    model = params["model"]
    xyz_grid = model.sample_points(
        grid_resolution,
        params["uv_bb_min"].copy(),
        params["uv_bb_max"].copy(),
        params["cluster_mean"],
        params["cluster_scale"],
        uv_margin = uv_margin
    ).cpu().numpy().reshape(grid_resolution, grid_resolution, 3)

    u_periodic = v_periodic = False
    if periodic:
        u_periodic = bool(getattr(model, "is_u_closed", False)) and _grid_axis_closed(xyz_grid, 0)
        v_periodic = bool(getattr(model, "is_v_closed", False)) and _grid_axis_closed(xyz_grid, 1)
        print(f"[inr->occ] closed axes: u={bool(getattr(model, 'is_u_closed', False))} "
              f"v={bool(getattr(model, 'is_v_closed', False))}  "
              f"periodic fit: u={u_periodic} v={v_periodic}")

    surface, _ = fit_bspline_surface(xyz_grid, degree_min = degree_min, degree_max = degree_max,
                                     continuity = continuity, tol3d = tol3d,
                                     u_periodic = u_periodic, v_periodic = v_periodic)
    return surface

def to_occ_surface(surface_id, result, cluster = None, **kwargs):
    params = result["params"]

    if surface_id == SURFACE_PLANE:
        return plane_to_occ(params)

    elif surface_id == SURFACE_SPHERE:
        return sphere_to_occ(params)

    elif surface_id == SURFACE_CYLINDER:
        return cylinder_to_occ(params)

    elif surface_id == SURFACE_CONE:
        return cone_to_occ(params, cluster = cluster)

    elif surface_id == SURFACE_INR:
        return inr_to_occ(params, **kwargs)
    else:
        raise ValueError(f"Unknown surface_id: {surface_id}")

import time

import numpy as np
import open3d as o3d

from .inr_fitting import fit_inr
from .primitive_fitting import fit_plane_numpy, fit_sphere_numpy, fit_cylinder_optimized, fit_cone
from .primitive_fitting_utils import generate_plane_mesh, generate_sphere_mesh, generate_cylinder_mesh, generate_cone_mesh
from .surface_types import (
    SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER, SURFACE_CONE, SURFACE_INR,
    SURFACE_NAMES,
)

# HPNet/Point2CAD floor of 20 points, enough evidence for a 6-DoF cone
MIN_CLUSTER_PTS = 20

# Keys must match surface_types.py and be contiguous from 0 so errors[sid] indexing stays valid
PRIMITIVE_FITTERS = {
    SURFACE_PLANE:    fit_plane_numpy,
    SURFACE_SPHERE:   fit_sphere_numpy,
    SURFACE_CYLINDER: fit_cylinder_optimized,
    SURFACE_CONE:     fit_cone,
}

def ratio(x, y, eps = 1e-8):
    return (x + eps) / (y + eps)

def _inflate_mesh(o3d_mesh, trimesh_mesh, surface_id, params,
                  radius_inflation, angle_inflation_deg):
    if surface_id == SURFACE_SPHERE and radius_inflation != 0.0:
        center = params["center"].reshape(1, 3)
        verts = np.asarray(o3d_mesh.vertices)
        dirs = verts - center
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        verts += dirs / norms * radius_inflation
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        trimesh_mesh.vertices += (dirs / norms * radius_inflation)[:len(trimesh_mesh.vertices)]

    elif surface_id == SURFACE_CYLINDER and radius_inflation != 0.0:
        center = params["center"].reshape(1, 3)
        axis = params["a"].reshape(3)
        verts = np.asarray(o3d_mesh.vertices)
        shifted = verts - center
        along = (shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        radial = shifted - along
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_norm = np.maximum(radial_norm, 1e-10)
        offset = radial / radial_norm * radius_inflation
        verts += offset
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        trimesh_mesh.vertices += offset[:len(trimesh_mesh.vertices)]

    elif surface_id == SURFACE_CONE and angle_inflation_deg != 0.0:
        vertex = params["v"].reshape(1, 3)
        axis = params["a"].reshape(3)
        theta = params["theta"]
        theta_new = theta + np.radians(angle_inflation_deg)
        scale = np.tan(theta_new) / (np.tan(theta) + 1e-10)

        verts = np.asarray(o3d_mesh.vertices)
        shifted = verts - vertex
        along = (shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        radial = shifted - along
        new_verts = vertex + along + radial * scale
        o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
        tv = trimesh_mesh.vertices
        t_shifted = tv - vertex
        t_along = (t_shifted @ axis).reshape(-1, 1) * axis.reshape(1, 3)
        t_radial = t_shifted - t_along
        trimesh_mesh.vertices = vertex + t_along + t_radial * scale


def resolve_mesh(surface_id,
                 result,
                 cluster,
                 np_rng,
                 device,
                 plane_mesh_kwargs,
                 sphere_mesh_kwargs,
                 cylinder_mesh_kwargs,
                 cone_mesh_kwargs,
                 inr_mesh_kwargs,
                 radius_inflation=0.0,
                 angle_inflation_deg=0.0):

    params = result["params"]

    if surface_id == SURFACE_PLANE:
        return generate_plane_mesh(
            a = params["a"],
            d = params["d"],
            cluster = cluster,
            np_rng = np_rng,
            device = device,
**plane_mesh_kwargs
        )

    if surface_id == SURFACE_SPHERE:
        mesh = generate_sphere_mesh(
            radius = params["radius"],
            center = params["center"],
            cluster = cluster,
            device = device,
**sphere_mesh_kwargs
        )
        if radius_inflation != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_CYLINDER:
        mesh = generate_cylinder_mesh(
            radius = params["radius"],
            center = params["center"],
            axis = params["a"],
            cluster = cluster,
            device = device,
**cylinder_mesh_kwargs
        )
        if radius_inflation != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_CONE:
        mesh = generate_cone_mesh(
            vertex = params["v"],
            axis = params["a"],
            theta = params["theta"],
            cluster_points = cluster,
            device = device,
**cone_mesh_kwargs
        )
        if angle_inflation_deg != 0.0:
            _inflate_mesh(mesh[0], mesh[1], surface_id, params, radius_inflation, angle_inflation_deg)
        return mesh

    if surface_id == SURFACE_INR:
        model = params["model"]
        return model.sample_mesh(
            uv_bb_min = params["uv_bb_min"],
            uv_bb_max = params["uv_bb_max"],
            cluster = cluster,
            cluster_mean = params["cluster_mean"],
            cluster_scale = params["cluster_scale"],
            uv_points = params.get("uv_points"),
            **inr_mesh_kwargs
        )

def cone_special_handling(results, errors, simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees):
    if ratio(errors[SURFACE_PLANE], errors[SURFACE_CONE]) < plane_cone_ratio_threshold:
        return SURFACE_PLANE

    cone_angle = results[SURFACE_CONE]["params"]["theta"]
    cone_theta_tolerance_rad = cone_theta_tolerance_degrees * (np.pi / 180)

    cone_diff_pi2 = abs(cone_angle - np.pi / 2)
    cone_diff_0 = cone_angle

    if cone_diff_pi2 < cone_theta_tolerance_rad or cone_diff_0 < cone_theta_tolerance_rad:
        simple_min = np.argmin(errors[:-1])

        return simple_min if errors[simple_min] < simple_error_threshold else -1

    return SURFACE_CONE


def plane_sphere_arbitration(errors, plane_sphere_ratio_threshold):
    if ratio(errors[SURFACE_PLANE], errors[SURFACE_SPHERE]) >= plane_sphere_ratio_threshold:
        return SURFACE_SPHERE
    return SURFACE_PLANE


def fit_surface(cluster,
                inr_network_parameters,
                np_rng,
                device,
                simple_error_threshold = 8e-3,
                simple_inr_ratio_threshold = 1.5,
                plane_cone_ratio_threshold = 1.5,
                plane_sphere_ratio_threshold = 2.5,
                cone_theta_tolerance_degrees = 5,
                sphere_fit_kwargs = None,
                cylinder_fit_kwargs = None,
                cone_fit_kwargs = None,
                inr_fit_kwargs = None,
                plane_mesh_kwargs = None,
                sphere_mesh_kwargs = None,
                cylinder_mesh_kwargs = None,
                cone_mesh_kwargs = None,
                inr_mesh_kwargs = None,
                radius_inflation = 0.0,
                angle_inflation_deg = 0.0,
                classify_only = False):

    sphere_fit_kwargs = sphere_fit_kwargs or {}
    cylinder_fit_kwargs = cylinder_fit_kwargs or {}
    cone_fit_kwargs = cone_fit_kwargs or {}
    inr_fit_kwargs = inr_fit_kwargs or {}
    plane_mesh_kwargs = plane_mesh_kwargs or {"mesh_dim": 100}
    sphere_mesh_kwargs = sphere_mesh_kwargs or {"dim_theta": 100, "dim_lambda": 100}
    cylinder_mesh_kwargs = cylinder_mesh_kwargs or {"dim_theta": 100, "dim_height": 50}
    cone_mesh_kwargs = cone_mesh_kwargs or {"dim_theta": 100, "dim_height": 100}
    inr_mesh_kwargs = inr_mesh_kwargs or {"mesh_dim": 100}

    fitter_kwargs = {
        SURFACE_PLANE:    {},
        SURFACE_SPHERE:   sphere_fit_kwargs,
        SURFACE_CYLINDER: cylinder_fit_kwargs,
        SURFACE_CONE:     cone_fit_kwargs,
    }

    _t_prim = time.perf_counter()
    results = {
        sid: PRIMITIVE_FITTERS[sid](cluster, **fitter_kwargs[sid])
        for sid in sorted(PRIMITIVE_FITTERS)
    }
    primitive_fit_time = time.perf_counter() - _t_prim
    errors = np.array([results[sid]["error"] for sid in range(len(PRIMITIVE_FITTERS))])
    _all_errors = {SURFACE_NAMES[sid]: float(results[sid]["error"])
                   for sid in sorted(PRIMITIVE_FITTERS)}
    simple_min = np.argmin(errors)

    if classify_only:
        # A cluster counts as primitive iff the pipeline would not invoke INR: best primitive below threshold, and cone_special_handling accepting a cone winner
        if errors[simple_min] < simple_error_threshold:
            if simple_min == SURFACE_CONE:
                cone_results = cone_special_handling(
                    results, errors, simple_error_threshold,
                    plane_cone_ratio_threshold, cone_theta_tolerance_degrees,
                )
                if cone_results == -1:
                    return "freeform"
            return "primitive"
        return "freeform"

    if errors[simple_min] < simple_error_threshold:
        if simple_min == SURFACE_CONE:
            cone_results = cone_special_handling(results, errors, simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees)

            if cone_results != -1:
                mesh = resolve_mesh(cone_results, results[cone_results], cluster, np_rng, device,
                            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
                            radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)

                return {"surface_id": cone_results, "result": results[cone_results], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors,
                        "primitive_fit_time": primitive_fit_time, "freeform_fit_time": 0.0}

        else:
            mesh = resolve_mesh(simple_min, results[simple_min], cluster, np_rng, device,
            plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
            radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)
            return {"surface_id": simple_min, "result": results[simple_min], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors,
                    "primitive_fit_time": primitive_fit_time, "freeform_fit_time": 0.0}

    errors_str = "  ".join(f"{SURFACE_NAMES[sid]}={errors[sid]:.6f}" for sid in range(len(PRIMITIVE_FITTERS)))
    print(f"  [surface fitter] no primitive below threshold ({simple_error_threshold:.4f}), "
          f"best={SURFACE_NAMES[simple_min]} ({errors[simple_min]:.6f})")
    print(f"  [surface fitter] primitive errors: {errors_str}")

    _t_freeform = time.perf_counter()

    print(f"  [surface fitter] fitting INR ...")
    inr_result = fit_inr(cluster, inr_network_parameters, device = device, **inr_fit_kwargs)
    results[SURFACE_INR] = inr_result
    errors = np.append(errors, inr_result["error"])
    _all_errors[SURFACE_NAMES[SURFACE_INR]] = float(inr_result["error"])

    global_min = np.argmin(errors)
    resulting_min = global_min

    if global_min == SURFACE_INR and ratio(errors[simple_min] , errors[SURFACE_INR]) < simple_inr_ratio_threshold:
        resulting_min = simple_min
        if resulting_min == SURFACE_CONE:
            resulting_min = cone_special_handling(results, errors[:-1], simple_error_threshold, plane_cone_ratio_threshold, cone_theta_tolerance_degrees)
            # cone_special_handling returns -1 when the second-best simple surface misses the threshold, in which case use INR
            if resulting_min == -1:
                resulting_min = global_min

            elif resulting_min == SURFACE_PLANE:
                resulting_min = plane_sphere_arbitration(errors[:-1], plane_sphere_ratio_threshold)
        elif resulting_min == SURFACE_PLANE or resulting_min == SURFACE_SPHERE:
            resulting_min = plane_sphere_arbitration(errors[:-1], plane_sphere_ratio_threshold)

    freeform_fit_time = time.perf_counter() - _t_freeform

    mesh = resolve_mesh(resulting_min, results[resulting_min], cluster, np_rng, device,
                        plane_mesh_kwargs, sphere_mesh_kwargs, cylinder_mesh_kwargs, cone_mesh_kwargs, inr_mesh_kwargs,
                        radius_inflation=radius_inflation, angle_inflation_deg=angle_inflation_deg)

    return {"surface_id": resulting_min, "result": results[resulting_min], "mesh": mesh[0], "trimesh_mesh": mesh[1], "all_errors": _all_errors,
            "primitive_fit_time": primitive_fit_time, "freeform_fit_time": freeform_fit_time}

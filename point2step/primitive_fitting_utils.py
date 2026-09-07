import numpy as np
import itertools
import open3d as o3d
import torch
import trimesh
from .color_config import get_surface_color

def rotation_matrix_a_to_b(A, B):
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    # A parallel or antiparallel to B gives NaN; the identity is correct for A == B and a harmless stand-in for A == -B
    if sin < 1e-12:
        return np.eye(3, dtype=np.float32)
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v))
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w))
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    try:
        R = F @ G @ F.T

    except:
        R = np.eye(3, dtype=np.float32)

    return R

def grid_trimming(cluster, vertices, size_u, size_v, device,
                   threshold_multiplier=3.0, spacing_percentile=90,
                   spacing=None, absolute_threshold=0.1):
    grid = vertices.reshape(size_u, size_v, -1)
    grid = grid[:, :, np.newaxis, :]
    grid = grid.transpose((3, 2, 0, 1))
    grid = torch.tensor(grid, device = device)
    filter = torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype = torch.float32, device = device).unsqueeze(0).unsqueeze(0)

    cell_means = torch.nn.functional.conv2d(grid, filter)
    cell_means = cell_means.permute(2, 3, 1, 0).squeeze(-2)
    cluster = torch.tensor(cluster, dtype = torch.float32, device = device)

    threshold = min(absolute_threshold, threshold_multiplier * spacing)

    cell_means = cell_means.reshape((-1, 3))
    D = torch.cdist(cell_means, cluster)
    min_dists = D.min(dim = -1).values
    mask = min_dists < threshold
    mask = mask.reshape((size_u - 1, size_v - 1))
    return mask

def triangulate_and_mesh(vertices, size_u, size_v, surface_type, mask = None):
    def index_to_id(i, j, size_v):
        return i * size_v + j

    triangles = []

    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None and not mask[i, j]:
                continue

            v0 = index_to_id(i, j, size_v)
            v1 = index_to_id(i + 1, j, size_v)
            v2 = index_to_id(i + 1, j + 1, size_v)
            v3 = index_to_id(i, j + 1, size_v)

            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])

    triangles = np.array(triangles) if triangles else np.zeros((0, 3), dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    color = get_surface_color(surface_type)
    mesh.paint_uniform_color(color)

    # Build trimesh from o3d's post-prune buffers so both copies index the same vertices
    trimesh_mesh = trimesh.Trimesh(np.asarray(mesh.vertices),
                                   np.asarray(mesh.triangles))
    return mesh, trimesh_mesh

def plane_error(points, a, d):
    assert np.allclose(np.linalg.norm(a), 1, rtol = 1e-6)

    plane_projections = points - np.dot(points, a)[:, np.newaxis] * a
    plane_projections += a * d
    return np.linalg.norm(plane_projections - points, axis = 1).mean()

def sphere_error(points, center, radius):
    return np.abs(np.linalg.norm(points - center, axis = 1) - radius).mean()

def cylinder_error(points, center, axis, radius):
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)

    shifted = points - center
    plane_projection = shifted - np.dot(shifted, axis)[:, np.newaxis] * axis
    orth_distance = np.linalg.norm(plane_projection, axis = 1)
    return np.abs(orth_distance - radius).mean()

def cone_error(points : np.ndarray, vertex: np.ndarray, axis: np.ndarray, theta: float):  
    assert np.allclose(np.linalg.norm(axis), 1, rtol = 1e-6)

    z = points - vertex
    scalar_projections = z @ axis
    h = np.abs(scalar_projections)
    r = z - scalar_projections[:, np.newaxis] * axis
    r = np.linalg.norm(r, axis = 1)

    errors = np.abs(h * np.sin(theta) - r * np.cos(theta))
    return np.mean(errors)

def sample_plane(mesh_dim, a, d, cluster, np_rng, sampling_deviation = 0):

    mean = cluster.mean(axis = 0)
    mean_projected = mean - (np.dot(a, mean) - d) * a
    r1, r2 = np_rng.uniform(low = 0, high = 1, size = (2,)).astype(np.float32)
    x = ((d - a[1] * r1 - a[2] * r2) / (a[0] + 1e-8)).item()
    x = np.array([x, r1, r2], dtype = np.float32)
    x = x - d * a
    x = x / np.linalg.norm(x)

    y = np.cross(a, x)
    y = y / np.linalg.norm(y)

    x_cords = (cluster - mean) @ x[:, np.newaxis]
    y_cords = (cluster - mean) @ y[:, np.newaxis]

    x_min = x_cords.min()
    x_min -= abs(x_min) * sampling_deviation
    x_max = x_cords.max()
    x_max += abs(x_max) * sampling_deviation
    y_min = y_cords.min()
    y_min -= abs(y_min) * sampling_deviation
    y_max = y_cords.max()
    y_max += abs(y_max) * sampling_deviation

    x_grid = np.linspace(x_min, x_max, mesh_dim)
    y_grid = np.linspace(y_min, y_max, mesh_dim)
    grid = np.array(list(itertools.product(x_grid, y_grid)))
    return (x * grid[:, 0:1] + y * grid[:, 1:2] + mean_projected).astype(np.float32)

def generate_plane_mesh(mesh_dim, a, d, cluster, np_rng, device,
                        plane_sampling_deviation = 0.1,
                        threshold_multiplier = 3.0,
                        spacing_percentile = 90,
                        spacing = None,
                        absolute_threshold = None):
    vertices = sample_plane(mesh_dim, a, d, cluster, np_rng, sampling_deviation = plane_sampling_deviation)
    mask = grid_trimming(cluster, vertices, mesh_dim, mesh_dim, device,
                         threshold_multiplier = threshold_multiplier,
                         spacing_percentile = spacing_percentile,
                         spacing = spacing,
                         absolute_threshold = absolute_threshold)
    return triangulate_and_mesh(vertices, mesh_dim, mesh_dim, "plane", mask = mask)

def sample_sphere(dim_theta, dim_lambda, radius, center):
    center = center.reshape((1, 3))
    theta = np.arange(dim_theta - 1) * np.pi * 2 / dim_theta
    theta = np.concatenate([theta, np.array([2 * np.pi])])
    circle = np.stack([np.cos(theta), np.sin(theta)], 1)
    lam = np.linspace(
        -radius + 1e-7, radius - 1e-7, dim_lambda
    )

    radii = np.sqrt(radius ** 2 - lam ** 2)
    circle = np.concatenate([circle] * lam.shape[0], 0)
    spread_radii = np.repeat(radii, dim_theta, 0)
    new_circle = circle * spread_radii.reshape((-1, 1))
    height = np.repeat(lam, dim_theta, 0)
    points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
    points = points + center
    return points.astype(np.float32)

def generate_sphere_mesh(dim_theta, dim_lambda, radius, center, cluster, device,
                         threshold_multiplier = 3.0,
                         spacing_percentile = 90,
                         spacing = None,
                         absolute_threshold = None):
    vertices = sample_sphere(dim_theta, dim_lambda, radius, center)
    mask = grid_trimming(cluster, vertices, dim_theta, dim_lambda, device,
                         threshold_multiplier = threshold_multiplier,
                         spacing_percentile = spacing_percentile,
                         spacing = spacing,
                         absolute_threshold = absolute_threshold)
    return triangulate_and_mesh(vertices, dim_theta, dim_lambda, "sphere", mask = mask)

def sample_cylinder(dim_theta, dim_height, radius, center, axis, points, height_margin = 0):
    # The cluster is needed to determine the minimum and maximum height for cylinder sampling
    center = center.reshape((1, 3))
    axis = axis.reshape((3, 1))

    R = rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

    points = points - center

    projection = points @ axis
    arg_min_proj = np.argmin(projection)
    arg_max_proj = np.argmax(projection)

    min_proj = np.squeeze(projection[arg_min_proj])
    max_proj = np.squeeze(projection[arg_max_proj])
    min_proj -= abs(min_proj) * height_margin
    max_proj += abs(max_proj) * height_margin

    theta = np.arange(dim_theta - 1, dtype = np.float32) * np.pi * 2 / dim_theta
    theta = np.concatenate([theta, np.array([2 * np.pi])])
    circle = np.stack([np.cos(theta), np.sin(theta)], 1, dtype = np.float32)
    circle = np.concatenate([circle] * 2 * dim_height, 0) * radius

    normals = np.concatenate([circle, np.zeros((circle.shape[0], 1), dtype = np.float32)], 1)
    normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)

    height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * dim_height), 1)
    height = np.repeat(height, dim_theta, axis = 0)
    points = np.concatenate([circle, height], 1)
    points = R @ points.T
    points = points.T + center

    return points.astype(np.float32)

def generate_cylinder_mesh(dim_theta, dim_height, radius, center, axis, cluster, device,
                           threshold_multiplier = 3.0,
                           cylinder_height_margin = 0.1,
                           spacing_percentile = 90,
                           spacing = None,
                           absolute_threshold = None):
    vertices = sample_cylinder(dim_theta, dim_height, radius, center, axis, cluster,
                               height_margin = cylinder_height_margin)
    mask = grid_trimming(cluster, vertices, dim_theta, 2 * dim_height, device,
                         threshold_multiplier = threshold_multiplier,
                         spacing_percentile = spacing_percentile,
                         spacing = spacing,
                         absolute_threshold = absolute_threshold)
    return triangulate_and_mesh(vertices, dim_theta, 2 * dim_height, "cylinder", mask = mask)

def sample_cone(dim_theta, dim_height, vertex, axis, theta, cluster_points, height_margin = 0, single_sided = True):
    vertex = vertex.reshape(3).astype(np.float32)
    axis = axis.reshape(3).astype(np.float32)
    axis = axis / np.linalg.norm(axis)

    proj = (cluster_points - vertex) @ axis
    proj_min = np.min(proj)
    proj_max = np.max(proj)
    proj_min -= abs(proj_min) * height_margin
    proj_max += abs(proj_max) * height_margin

    if single_sided:
        n_positive = np.sum(proj > 0)
        n_negative = np.sum(proj < 0)

        if n_positive >= n_negative:
            proj_min = max(proj_min, 1e-7)
        else:
            proj_max = min(proj_max, -1e-7)

    if np.abs(axis[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])

    r_hat = np.cross(axis, temp)
    r_hat = r_hat / np.linalg.norm(r_hat)

    g_hat = np.cos(theta) * axis + np.sin(theta) * r_hat
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    heights = np.linspace(proj_min, proj_max, dim_height)
    angles = np.arange(dim_theta - 1, dtype = np.float32) * 2 * np.pi / dim_theta
    angles = np.concatenate([angles, np.array([2 * np.pi])])
    points = []

    for h in heights:
        t = h / (np.cos(theta) + 1e-10)
        base_vec = t * g_hat

        for phi in angles:
            R = np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * (K @ K)
            rotated_vec = R @ base_vec
            point = vertex + rotated_vec
            points.append(point)

    points = np.array(points, dtype = np.float32)
    return points

def generate_cone_mesh(dim_theta, dim_height, vertex, axis, theta, cluster_points, device,
                       threshold_multiplier = 3.0,
                       cone_height_margin = 0.1,
                       cone_single_sided = True,
                       spacing_percentile = 90,
                       spacing = None,
                       absolute_threshold = None):
    vertices = sample_cone(dim_theta, dim_height, vertex, axis, theta, cluster_points, height_margin = cone_height_margin, single_sided = cone_single_sided)
    mask = grid_trimming(cluster_points, vertices, dim_theta, dim_height, device,
                         threshold_multiplier = threshold_multiplier,
                         spacing_percentile = spacing_percentile,
                         spacing = spacing,
                         absolute_threshold = absolute_threshold)
    return triangulate_and_mesh(vertices, dim_theta, dim_height, "cone", mask = mask)

if __name__ == "__main__":
    np_rng = np.random.default_rng(41)

    cluster_points = np_rng.standard_normal((1000, 3)).astype(np.float32)
    vertex = np.array([2, 1, 0], dtype=np.float32)
    cone_axis = np.array([0, 0, 1], dtype=np.float32)
    half_angle = np.pi / 4
    dummy_cluster = np_rng.standard_normal((500, 3))
    mesh = generate_cone_mesh(100, 100, vertex, cone_axis, half_angle, cluster_points, "cuda:0")
    o3d.visualization.draw_geometries([mesh])

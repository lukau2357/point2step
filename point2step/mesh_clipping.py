import numpy as np
import open3d as o3d
import trimesh
from collections import Counter
from scipy.spatial import cKDTree

from .color_config import get_surface_color


def o3d_mesh_to_numpy(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    return vertices, triangles


def _merge_meshes(mesh_list):
    all_V = []
    all_F = []
    face_sources = []
    vertex_offset = 0

    for idx, (V, F) in enumerate(mesh_list):
        all_V.append(np.asarray(V, dtype=np.float64))
        all_F.append(np.asarray(F, dtype=np.int64) + vertex_offset)
        face_sources.append(np.full(len(F), idx, dtype=np.int32))
        vertex_offset += len(V)

    V = np.vstack(all_V) if all_V else np.empty((0, 3), dtype=np.float64)
    F = np.vstack(all_F) if all_F else np.empty((0, 3), dtype=np.int64)
    face_sources = np.concatenate(face_sources) if face_sources else np.empty(0, dtype=np.int32)

    return V, F, face_sources


def _build_edge_face_map(F):
    edge_to_faces = {}
    for fi in range(len(F)):
        for k in range(3):
            v0 = int(F[fi, k])
            v1 = int(F[fi, (k + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)
    return edge_to_faces


def _find_intersection_edges(FF, face_provenance, n_vertices):
    vertex_provs = [set() for _ in range(n_vertices)]
    for fi in range(len(FF)):
        prov = int(face_provenance[fi])
        for vi in FF[fi]:
            vertex_provs[int(vi)].add(prov)

    boundary_verts = set(vi for vi in range(n_vertices) if len(vertex_provs[vi]) > 1)

    intersection_edges = set()
    for fi in range(len(FF)):
        for k in range(3):
            v0 = int(FF[fi, k])
            v1 = int(FF[fi, (k + 1) % 3])
            if v0 in boundary_verts and v1 in boundary_verts:
                intersection_edges.add((min(v0, v1), max(v0, v1)))

    print(f"[intersection] {len(boundary_verts)} boundary vertices, "
          f"{len(intersection_edges)} intersection edges")
    return intersection_edges


def _upsample_points_knn(points, times=3, k=5):
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(times):
        if len(pts) < k + 1:
            break
        # Query k+1 and drop the self-match so the centroid averages k true neighbours
        _, nn_idx = cKDTree(pts).query(pts, k=k + 1)
        centers = pts[nn_idx[:, 1:]].mean(axis=1)
        pts = np.concatenate([pts, centers], axis=0)
    return pts


def build_cluster_trees(clusters, spacing_percentile=100.0, upsample_passes=3):
    trees, spacings = [], []
    for c in clusters:
        d, _ = cKDTree(c).query(c, k=2)   # spacing: original-cluster NN
        spacings.append(float(np.percentile(d[:, 1], spacing_percentile)))

        c_dense = _upsample_points_knn(c, times=upsample_passes) \
                  if upsample_passes > 0 else c
        trees.append(cKDTree(c_dense))
    return trees, spacings


def _has_igl():
    try:
        import igl
        import igl.copyleft.cgal
        return True
    except ImportError:
        return False


def _resolve_intersections(o3d_meshes, tag="clip"):
    import igl
    import igl.copyleft.cgal

    mesh_list = []
    for m in o3d_meshes:
        V_i, F_i = o3d_mesh_to_numpy(m)
        mesh_list.append((V_i.astype(np.float64), F_i.astype(np.int32)))
    V_merged, F_merged, face_sources_merged = _merge_meshes(mesh_list)
    print(f"[{tag}] Merged: {len(V_merged)} vertices, {len(F_merged)} faces")

    # J[i] = merged face that resolved face i came from (one-level provenance)
    # IM[i] = canonical vertex index for vertex i (deduplication map)
    VV_raw, FF_raw, _IF, J, IM = igl.copyleft.cgal.remesh_self_intersections(
        V_merged, F_merged.astype(np.int32)
    )
    face_provenance = face_sources_merged[J].astype(np.int32)

    FF_dedup = IM[FF_raw]
    VV, FF_clean, _I, _J2 = igl.remove_unreferenced(VV_raw, FF_dedup)
    FF = FF_clean.astype(np.int64)
    print(f"[{tag}] Resolved: {len(VV)} vertices, {len(FF)} faces")
    return VV, FF, face_provenance


def clip_meshes_p2cad(o3d_meshes, clusters, surface_types,
                      cluster_trees=None, spacings=None,
                      area_multiplier=2.0,
                      post_filter_threshold=None,
                      cluster_ids=None):
    if cluster_ids is None:
        cluster_ids = list(range(len(clusters)))

    if post_filter_threshold is not None and (cluster_trees is None or spacings is None):
        raise ValueError("post_filter_threshold requires cluster_trees and spacings")

    if _has_igl():
        VV, FF, face_provenance = _resolve_intersections(o3d_meshes, tag="p2cad-clip")

        # CGAL shares vertices but not edges across surfaces, so intersection boundary edges must be detected explicitly
        edge_to_faces = _build_edge_face_map(FF)
        intersection_edges = _find_intersection_edges(FF, face_provenance, len(VV))
        adj_pairs = []
        for edge, face_list in edge_to_faces.items():
            if len(face_list) == 2 and edge not in intersection_edges:
                adj_pairs.append(face_list)
        adj_edges = np.array(adj_pairs, dtype=np.int64) if adj_pairs else np.empty((0, 2), dtype=np.int64)

        connected_labels = trimesh.graph.connected_component_labels(
            edges=adj_edges,
            node_count=len(FF)
        )
    else:
        raise RuntimeError("mesh_clipping requires igl with igl.copyleft.cgal")

    unique_labels = [item[0] for item in Counter(connected_labels).most_common()]
    print(f"[p2cad-clip] {len(unique_labels)} connected components")

    submeshes = []
    submesh_surface = []
    n_tiny_dropped = 0
    for lbl in unique_labels:
        face_idx = np.where(connected_labels == lbl)[0]
        if len(face_idx) <= 2:
            n_tiny_dropped += 1
            continue
        sub = trimesh.Trimesh(vertices=VV,
                              faces=FF[face_idx],
                              process=False)
        submeshes.append(sub)
        submesh_surface.append(int(face_provenance[face_idx[0]]))

    print(f"[p2cad-clip] dropped {n_tiny_dropped} tiny components (<=2 faces); "
          f"{len(submeshes)} retained for per-surface attribution")

    clipped = []
    for s in range(len(clusters)):
        cluster_pts = clusters[s].astype(np.float64)
        subs = [sub for sub, sid in zip(submeshes, submesh_surface) if sid == s]

        if len(subs) == 0:
            print(f"[p2cad-clip] cluster {cluster_ids[s]} ({surface_types[s]}): no components")
            clipped.append(o3d.geometry.TriangleMesh())
            continue

        nearest = np.argmin(
            np.array([trimesh.proximity.closest_point(sub, cluster_pts)[1]
                      for sub in subs]).T,
            axis=1
        )
        counter = Counter(nearest).most_common()

        area_per_point = np.array([subs[idx].area / count
                                   for idx, count in counter])

        nonzero = np.nonzero(area_per_point)[0]
        if len(nonzero) == 0:
            print(f"[p2cad-clip] cluster {cluster_ids[s]} ({surface_types[s]}): all zero-area components")
            clipped.append(o3d.geometry.TriangleMesh())
            continue

        best = area_per_point[nonzero[0]]
        keep_idx = np.array(counter)[:, 0][
            (area_per_point < best * area_multiplier) & (area_per_point != 0)
        ]

        # Post-APP vote filter: drop survivors with < 5% of the top vote count
        votes_by_idx = {idx: count for idx, count in counter}
        max_votes = max(votes_by_idx[i] for i in keep_idx)
        vote_min = 0.05 * max_votes
        dropped = [i for i in keep_idx if votes_by_idx[i] < vote_min]
        if dropped:
            for i in dropped:
                print(f"[p2cad-clip] cluster {cluster_ids[s]}: dropping cc {i} "
                      f"(votes={votes_by_idx[i]}, < 5% of max={max_votes})")
        keep_idx = np.array([i for i in keep_idx if votes_by_idx[i] >= vote_min])

        kept = trimesh.util.concatenate([subs[i] for i in keep_idx])
        print(f"[p2cad-clip] cluster {cluster_ids[s]} ({surface_types[s]}): "
              f"{len(keep_idx)}/{len(subs)} components kept (best_app={best:.6f})")
        if len(keep_idx) <= 20:
            app_by_idx = {idx: subs[idx].area / count for idx, count in counter}
            for i in keep_idx:
                print(f"    cc {i}: app={app_by_idx[i]:.6f}, votes={votes_by_idx[i]}")

        kept_vertices = np.array(kept.vertices)
        kept_faces    = np.array(kept.faces)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(kept_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(kept_faces)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(get_surface_color(surface_types[s]))
        clipped.append(mesh)

    return clipped


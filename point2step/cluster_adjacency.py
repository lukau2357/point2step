import numpy as np
from scipy.spatial import KDTree


def _reference_spacing(clusters, percentile=100.0):
    nn_dists = []
    for cluster in clusters:
        tree = KDTree(cluster)
        d, _ = tree.query(cluster, k=2)
        nn_dists.append(d[:, 1])
    return float(np.percentile(np.concatenate(nn_dists), percentile))


def _local_spacing(cluster, percentile=100.0):
    tree = KDTree(cluster)
    d, _ = tree.query(cluster, k=2)
    return float(np.percentile(d[:, 1], percentile))


def compute_adjacency_matrix(clusters, threshold_factor=1.5, spacing=None,
                              spacing_percentile=100.0,
                              local_spacings=None):
    n = len(clusters)

    if local_spacings is None:
        local_spacings = [_local_spacing(c, percentile=spacing_percentile) for c in clusters]

    if spacing is None:
        spacing = _reference_spacing(clusters, percentile=spacing_percentile)
    global_threshold = threshold_factor * spacing

    adj = np.zeros((n, n), dtype=bool)
    boundary_strips = {}
    per_pair_thresholds = {}
    boundary_strip_trees = {}

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = clusters[i], clusters[j]
            larger, smaller = (ci, cj) if len(ci) >= len(cj) else (cj, ci)

            tree = KDTree(larger)
            nn_dists, nn_idx = tree.query(smaller, k=1)

            threshold_ij = threshold_factor * max(local_spacings[i], local_spacings[j])
            if nn_dists.min() <= threshold_ij:
                adj[i, j] = adj[j, i] = True
                per_pair_thresholds[(i, j)] = threshold_ij

                mask = nn_dists <= threshold_ij
                strip_smaller = smaller[mask]
                strip_larger = larger[nn_idx[mask]]
                strip_pts = np.vstack(
                    [strip_smaller, strip_larger]
                ).astype(np.float32)
                boundary_strips[(i, j)] = strip_pts
                boundary_strip_trees[(i, j)] = KDTree(strip_pts)

    return (adj, global_threshold, spacing, boundary_strips,
            per_pair_thresholds, boundary_strip_trees)

def build_cluster_proximity(clusters, percentile=99.0):
    trees, thresholds = [], []
    for cluster in clusters:
        tree = KDTree(cluster)
        d, _ = tree.query(cluster, k=2)
        thresholds.append(float(np.percentile(d[:, 1], percentile)))
        trees.append(tree)
    return trees, thresholds


def adjacency_pairs(adj):
    n = adj.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]

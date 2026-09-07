import math
import numpy as np

try:
    from .surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER,
        SURFACE_NAMES,
    )
    from .cluster_adjacency import adjacency_pairs
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from point2step.surface_types import (
        SURFACE_PLANE, SURFACE_SPHERE, SURFACE_CYLINDER,
        SURFACE_NAMES,
    )
    from point2step.cluster_adjacency import adjacency_pairs

try:
    from OCC.Core.gp          import gp_Pnt, gp_Dir, gp_Lin, gp_Circ, gp_Ax2
    from OCC.Core.Geom        import Geom_Line, Geom_Circle, Geom_TrimmedCurve
    from OCC.Core.GeomAPI     import (GeomAPI_IntSS,
                                      GeomAPI_ExtremaCurveCurve,
                                      GeomAPI_ProjectPointOnCurve)
    from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
    from OCC.Core.GeomAbs     import (GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse,
                                       GeomAbs_Hyperbola, GeomAbs_Parabola,
                                       GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
                                       GeomAbs_OtherCurve)
    from OCC.Core.Precision   import precision
    _OCC_INF = precision.Infinite()
    _GEOMABS_NAMES = {
        GeomAbs_Line: "Line", GeomAbs_Circle: "Circle", GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola", GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "Bezier", GeomAbs_BSplineCurve: "BSpline",
        GeomAbs_OtherCurve: "Other",
    }
    OCC_AVAILABLE = True
except ImportError as err:
    _OCC_INF       = float("inf")
    _GEOMABS_NAMES = {}
    OCC_AVAILABLE  = False


def _unit(v):
    v = np.asarray(v, dtype = np.float64).ravel()
    return v / np.linalg.norm(v)

def _perp_to(n):
    n = _unit(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(n, t)
    return v / np.linalg.norm(v)

def _gp_pnt(p):
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))

def _gp_dir(v):
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))

def _result(curves, points, curve_type, method):
    return {"curves": curves, "points": points, "type": curve_type, "method": method}


def _as_safe_curve(curve, model_extent=2.0):
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if abs(t0) < _OCC_INF and abs(t1) < _OCC_INF:
        return curve   # already finite — no wrapping needed

    L       = model_extent
    adaptor = GeomAdaptor_Curve(curve)
    ctype   = adaptor.GetType()

    if ctype == GeomAbs_Hyperbola:
        # Flat conservative bound: cosh(10) ~ 11013, safe and independent of model size
        t_bound = 10.0
    elif ctype == GeomAbs_Parabola:
        f       = max(adaptor.Parabola().Focal(), 1e-10)
        t_bound = max(L, 2.0 * math.sqrt(f * L))
    elif ctype == GeomAbs_Line:
        t_bound = L
    else:
        t_bound = 10.0   # conservative fallback for unknown infinite-domain types

    return Geom_TrimmedCurve(curve, -t_bound, t_bound)


def _intersect_plane_plane(pi, pj):
    n1 = _unit(pi["a"]);  d1 = float(pi["d"])
    n2 = _unit(pj["a"]);  d2 = float(pj["d"])

    dir_vec = np.cross(n1, n2)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm < 1e-10:
        return _result([], [], "empty", "analytical")   # parallel planes
    dir_vec /= dir_norm

    # Minimum-norm solution of the 2x3 system A p = b with A = [n1; n2]: p0 = A^T (A A^T)^-1 b
    A  = np.stack([n1, n2])          # (2, 3)
    b  = np.array([d1, d2])
    p0, _, _, _ = np.linalg.lstsq(A, b, rcond = None)

    line = Geom_Line(gp_Lin(_gp_pnt(p0), _gp_dir(dir_vec)))
    return _result([line], [], "line", "analytical")


def _intersect_plane_sphere(pi, pj):
    n    = _unit(pi["a"]);  d = float(pi["d"])
    c    = np.asarray(pj["center"], dtype = np.float64)
    r    = float(pj["radius"])

    dist = float(n @ c) - d          # signed distance from sphere centre to plane

    if abs(dist) > r + 1e-10:
        return _result([], [], "empty", "analytical")

    if abs(abs(dist) - r) < 1e-10:
        pt = c - dist * n            # tangency point
        return _result([], [_gp_pnt(pt)], "tangent", "analytical")

    center   = c - dist * n          # foot of perpendicular from c onto the plane
    r_circle = math.sqrt(max(r * r - dist * dist, 0.0))
    x_dir    = _perp_to(n)
    ax2      = gp_Ax2(_gp_pnt(center), _gp_dir(n), _gp_dir(x_dir))
    circle   = Geom_Circle(gp_Circ(ax2, r_circle))
    return _result([circle], [], "circle", "analytical")


def _intersect_sphere_sphere(pi, pj):
    c1 = np.asarray(pi["center"], dtype = np.float64);  r1 = float(pi["radius"])
    c2 = np.asarray(pj["center"], dtype = np.float64);  r2 = float(pj["radius"])

    axis = c2 - c1
    d    = float(np.linalg.norm(axis))
    if d < 1e-10:
        return _result([], [], "empty", "analytical")

    # Spheres completely separated or one inside the other
    if d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10:
        return _result([], [], "empty", "analytical")

    n  = axis / d
    # Signed distance from c1 to the radical plane along n:
    #   h = (d^2 + r1^2 - r2^2) / (2d)
    h  = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    p0 = c1 + h * n                  # centre of the intersection circle
    d_p = float(n @ p0)              # radical-plane offset (n . x = d_p)

    # Reuse plane-sphere: the radical plane cuts sphere 1 in the intersection circle
    return _intersect_plane_sphere({"a": n, "d": d_p}, pi)


def _intersect_plane_cylinder_tangent(pi_plane, pi_cyl,
                                       tol_parallel=0.001, tol_tangent=1e-3):
    n = _unit(pi_plane["a"])
    d = float(pi_plane["d"])
    a = _unit(pi_cyl["a"])
    c = np.asarray(pi_cyl["center"], dtype=np.float64)
    r = float(pi_cyl["radius"])

    # Plane must be (nearly) parallel to the cylinder axis
    alpha = float(np.dot(n, a))
    if abs(alpha) >= tol_parallel:
        return _result([], [], "empty", "analytical")

    n_perp    = n - alpha * a
    n_perp_sq = float(np.dot(n_perp, n_perp))
    if n_perp_sq < 1e-12:
        return _result([], [], "empty", "analytical")

    D    = d - float(np.dot(n, c))          # d − n·c
    dist = abs(D) / math.sqrt(n_perp_sq)    # δ = |D| / |n_⊥|

    # Tangency: δ ≈ r  (secant δ < r is non-degenerate and handled by OCC)
    if abs(dist - r) > tol_tangent:
        return _result([], [], "empty", "analytical")

    # Tangent foot on the cylinder surface (lies on both the plane and cylinder)
    q = c + (D / n_perp_sq) * n_perp

    line = Geom_Line(gp_Lin(_gp_pnt(q), _gp_dir(a)))
    return _result([line], [], "line", "analytical")


def _intersect_cylinder_cylinder_parallel(pi, pj,
                                           tol_parallel=0.001, tol_tangent=1e-3):
    a1 = _unit(pi["a"])
    a2 = _unit(pj["a"])
    c1 = np.asarray(pi["center"], dtype=np.float64)
    c2 = np.asarray(pj["center"], dtype=np.float64)
    r1 = float(pi["radius"])
    r2 = float(pj["radius"])

    dot = abs(float(np.dot(a1, a2)))
    if dot < 1.0 - tol_parallel:
        return None   # not parallel — caller should use OCC

    a = a1

    # q_i = c_i − (c_i · a) a   (perpendicular component)
    q1 = c1 - float(np.dot(c1, a)) * a
    q2 = c2 - float(np.dot(c2, a)) * a

    diff = q2 - q1
    d = float(np.linalg.norm(diff))

    if d < 1e-12:
        # Coaxial cylinders — intersection is empty (different r) or degenerate (same r, handled by equiv).
        return _result([], [], "empty", "analytical")

    e = diff / d   # unit vector from q1 to q2 in the perp plane
    x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    h_sq = r1 * r1 - x * x

    if h_sq < -tol_tangent * max(r1, r2):
        return _result([], [], "empty", "analytical")

    # Build the perpendicular direction in the plane ⊥ a e is in the perp plane; need a vector ⊥ both a and e
    f = np.cross(a, e)
    f = f / np.linalg.norm(f)

    # Base point in the perp plane (relative to c1)
    base = q1 + x * e

    if h_sq <= tol_tangent * max(r1, r2):
        # Tangent case: single generator line, adding the a-component from c1 to the perp-plane base
        p3d = base + float(np.dot(c1, a)) * a
        line = Geom_Line(gp_Lin(_gp_pnt(p3d), _gp_dir(a)))
        return _result([line], [], "line", "analytical")

    # Secant case — two generator lines
    h = math.sqrt(h_sq)
    a_comp = float(np.dot(c1, a)) * a

    p1_3d = base + h * f + a_comp
    p2_3d = base - h * f + a_comp

    line1 = Geom_Line(gp_Lin(_gp_pnt(p1_3d), _gp_dir(a)))
    line2 = Geom_Line(gp_Lin(_gp_pnt(p2_3d), _gp_dir(a)))
    return _result([line1, line2], [], "line", "analytical")


def _intersect_occ(occ_surf_i, occ_surf_j, tol, label=""):
    # https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___int_s_s.html#details
    try:
        inter = GeomAPI_IntSS(occ_surf_i, occ_surf_j, tol)
    except Exception as e:
        if label:
            print(f"  [intersect] {label}: FAILED (exception: {e})")
        return _result([], [], "failed", "occ")

    if not inter.IsDone():
        if label:
            print(f"  [intersect] {label}: FAILED (IsDone=False)")
        return _result([], [], "failed", "occ")

    curves = [inter.Line(k) for k in range(1, inter.NbLines() + 1)]

    if not curves:
        # NbLines()==0 after IsDone() means no intersection or point tangency, indistinguishable at this API level
        if label:
            print(f"  [intersect] {label}: empty (NbLines=0)")
        return _result([], [], "empty", "occ")

    prefix = f"[intersect] {label}: " if label else "[intersect] "
    for c in curves:
        t0, t1 = c.FirstParameter(), c.LastParameter()
        gtype  = _GEOMABS_NAMES.get(GeomAdaptor_Curve(c).GetType(), "?")
        t0_str = f"{t0:.4g}" if abs(t0) < _OCC_INF else "-inf"
        t1_str = f"{t1:.4g}" if abs(t1) < _OCC_INF else "+inf"
        print(f"  {prefix}{gtype}  t=[{t0_str}, {t1_str}]")

    return _result(curves, [], "curve", "occ")


def intersect_surfaces(surface_id_i, result_i, occ_surf_i,
                       surface_id_j, result_j, occ_surf_j,
                       tol=1e-6, label=""):
    si, sj = surface_id_i, surface_id_j
    pi, pj = result_i["params"], result_j["params"]
    oi, oj = occ_surf_i, occ_surf_j

    # Normalise so that si <= sj (surfaces are type-indexed 0..4)
    if si > sj:
        si, sj = sj, si
        pi, pj = pj, pi
        oi, oj = oj, oi

    if si == SURFACE_PLANE and sj == SURFACE_PLANE:
        return _intersect_plane_plane(pi, pj)
    if si == SURFACE_PLANE and sj == SURFACE_SPHERE:
        return _intersect_plane_sphere(pi, pj)
    if si == SURFACE_SPHERE and sj == SURFACE_SPHERE:
        return _intersect_sphere_sphere(pi, pj)

    # Plane ∩ Cylinder: try OCC first (handles circle / ellipse correctly); fall back to analytical tangent-line if OCC returns empty.
    if si == SURFACE_PLANE and sj == SURFACE_CYLINDER:
        occ = _intersect_occ(oi, oj, tol, label=label)
        if occ["type"] == "empty":
            return _intersect_plane_cylinder_tangent(pi, pj)
        return occ

    # Cylinder ∩ Cylinder: try OCC first; fall back to analytical parallel- axis formula if OCC fails (IsDone=False) or returns empty.
    if si == SURFACE_CYLINDER and sj == SURFACE_CYLINDER:
        occ = _intersect_occ(oi, oj, tol, label=label)
        if occ["type"] in ("failed", "empty"):
            analytical = _intersect_cylinder_cylinder_parallel(pi, pj)
            if analytical is not None and analytical["type"] != "empty":
                if label:
                    for c in analytical["curves"]:
                        t0, t1 = c.FirstParameter(), c.LastParameter()
                        t0s = f"{t0:.4g}" if abs(t0) < _OCC_INF else "-inf"
                        t1s = f"{t1:.4g}" if abs(t1) < _OCC_INF else "+inf"
                        print(f"  [intersect] {label}: Line (analytical)  t=[{t0s}, {t1s}]")
                return analytical
        return occ

    return _intersect_occ(oi, oj, tol, label=label)


def compute_all_intersections(adj, surface_ids, results, occ_surfaces,
                              tol=1e-6):
    out = {}
    for i, j in adjacency_pairs(adj):
        out[(i, j)] = intersect_surfaces(
            surface_ids[i], results[i], occ_surfaces[i],
            surface_ids[j], results[j], occ_surfaces[j],
            tol   = tol,
            label = f"({i}, {j}) {SURFACE_NAMES[surface_ids[i]]}∩{SURFACE_NAMES[surface_ids[j]]}",
        )
    return out


def sample_curve(curve, boundary_pts=None, threshold=None,
                 n_points=200, extension_factor=0.15, line_extent=1.0):
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    is_inf = abs(t0) >= _OCC_INF or abs(t1) >= _OCC_INF

    if boundary_pts is not None and threshold is not None and len(boundary_pts) >= 2:
        params = []
        for pt in boundary_pts:
            proj = GeomAPI_ProjectPointOnCurve(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])), curve
            )
            if proj.NbPoints() > 0:
                params.append(proj.LowerDistanceParameter())

        if len(params) >= 2:
            ta   = float(min(params))
            tb   = float(max(params))
            span = tb - ta
            ta  -= extension_factor * span
            tb  += extension_factor * span
            if not is_inf:
                ta = max(ta, t0)
                tb = min(tb, t1)
            pts = np.zeros((n_points, 3))
            for i, t in enumerate(np.linspace(ta, tb, n_points)):
                p = curve.Value(t)
                pts[i] = (p.X(), p.Y(), p.Z())
            return pts

    # Fallback: full parameter domain (boundary projection unavailable or yielded fewer than 2 projections within threshold).
    if is_inf:
        t0, t1 = -line_extent, line_extent
    pts = np.zeros((n_points, 3))
    for i, t in enumerate(np.linspace(t0, t1, n_points)):
        p = curve.Value(t)
        pts[i] = (p.X(), p.Y(), p.Z())
    return pts


_CLOSURE_TOL = 1e-4   # must match topology.CLOSURE_TOL — OCC cone∩cylinder
                      # boundary BSplines have endpoint_dist ~9e-6 which is closed at the 1e-4 threshold but open at 1e-7.


def _curve_is_closed(curve):
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    if abs(t0) >= _OCC_INF or abs(t1) >= _OCC_INF:
        return False
    p0 = curve.Value(t0)
    p1 = curve.Value(t1)
    return math.sqrt(
        (p1.X() - p0.X()) ** 2 +
        (p1.Y() - p0.Y()) ** 2 +
        (p1.Z() - p0.Z()) ** 2
    ) < _CLOSURE_TOL


def trim_by_vertices(raw_intersections, vertices, vertex_edges,
                     extension_factor=0.05,
                     phantom_vertex_dist=1e-3):
    trimmed = {}

    edge_to_vpos = {}
    for v_idx in range(len(vertices)):
        for edge in vertex_edges[v_idx]:
            edge_to_vpos.setdefault(edge, []).append(vertices[v_idx])

    for (i, j), inter in raw_intersections.items():
        vpositions = edge_to_vpos.get((i, j), [])

        closed_curves = []
        open_curves   = []
        for c in inter["curves"]:
            (closed_curves if _curve_is_closed(c) else open_curves).append(c)

        if len(open_curves) > 1 and len(vpositions) >= 2:
            def _min_vertex_dist(c):
                best  = float("inf")
                c_s   = _as_safe_curve(c)
                for vpos in vpositions:
                    proj = GeomAPI_ProjectPointOnCurve(
                        gp_Pnt(float(vpos[0]), float(vpos[1]), float(vpos[2])), c_s
                    )
                    if proj.NbPoints() > 0:
                        best = min(best, float(proj.LowerDistance()))
                return best

            dists = [_min_vertex_dist(c) for c in open_curves]
            kept  = [c for c, d in zip(open_curves, dists) if d <= phantom_vertex_dist]
            open_curves = kept if kept else open_curves  # fail-safe

        new_curves = list(closed_curves)

        for c in open_curves:
            t0_orig     = c.FirstParameter()
            t1_orig     = c.LastParameter()
            is_infinite = abs(t0_orig) >= _OCC_INF or abs(t1_orig) >= _OCC_INF

            trimmed_c = None
            if len(vpositions) >= 2:
                params = []
                c_safe = _as_safe_curve(c)
                for vpos in vpositions:
                    proj = GeomAPI_ProjectPointOnCurve(
                        gp_Pnt(float(vpos[0]), float(vpos[1]), float(vpos[2])), c_safe
                    )
                    if proj.NbPoints() > 0:
                        params.append(float(proj.LowerDistanceParameter()))

                if len(params) >= 2:
                    t_min = min(params)
                    t_max = max(params)
                    if t_max > t_min:
                        span   = t_max - t_min
                        t_min -= extension_factor * span
                        t_max += extension_factor * span
                        try:
                            trimmed_c = Geom_TrimmedCurve(c, t_min, t_max)
                        except Exception:
                            pass

            if trimmed_c is None:
                # Open curves without two incident vertices are dropped; closed loops pass through to build_edge_arcs, which emits the V(e) = empty candidate edge
                if is_infinite or not _curve_is_closed(c):
                    continue
                trimmed_c = c   # closed loop with V(e)=∅ — keep as-is

            if (abs(trimmed_c.FirstParameter()) >= _OCC_INF or
                    abs(trimmed_c.LastParameter()) >= _OCC_INF):
                continue

            new_curves.append(trimmed_c)

        trimmed[(i, j)] = {
            **inter,
            "curves":       new_curves,
            "boundary_pts": np.empty((0, 3), dtype=np.float32),
        }

    return trimmed


def compute_vertices_extrema(adj, intersections, threshold=1e-3):
    edge_keys = [k for k, v in intersections.items() if v.get("curves")]
    all_edge_keys = set(edge_keys)

    candidates        = []   # (3,) positions
    candidate_triples = []   # frozenset({i, j, k}), parallel to candidates

    n_pairs_examined    = 0
    n_pairs_skipped_3rd = 0
    n_extrema_total     = 0
    n_extrema_accepted  = 0

    for idx_a in range(len(edge_keys)):
        edge_a   = edge_keys[idx_a]
        curves_a = intersections[edge_a].get("curves", [])
        if not curves_a:
            continue
        for idx_b in range(idx_a + 1, len(edge_keys)):
            edge_b = edge_keys[idx_b]
            common = set(edge_a) & set(edge_b)
            if len(common) != 1:
                continue
            # The third edge between the two unshared surfaces must also be adjacent, or the triple cannot yield a vertex with consistent per-face Eulerian degree
            union      = set(edge_a) | set(edge_b)
            unshared   = sorted(union - common)
            third_edge = (unshared[0], unshared[1])
            if third_edge not in all_edge_keys:
                n_pairs_skipped_3rd += 1
                continue
            triple   = frozenset(union)
            curves_b = intersections[edge_b].get("curves", [])
            if not curves_b:
                continue
            n_pairs_examined += 1
            for ca in curves_a:
                ca_safe = _as_safe_curve(ca)
                for cb in curves_b:
                    cb_safe = _as_safe_curve(cb)
                    try:
                        ext = GeomAPI_ExtremaCurveCurve(ca_safe, cb_safe)
                    except Exception as err:
                        print(f"[compute_vertices_extrema] "
                              f"ExtremaCurveCurve({edge_a}, {edge_b}) "
                              f"exception: {err}")
                        continue
                    try:
                        n_ext = ext.NbExtrema()
                    except Exception:
                        continue
                    n_extrema_total += n_ext
                    for k in range(1, n_ext + 1):
                        try:
                            d = ext.Distance(k)
                        except Exception:
                            continue
                        if d > threshold:
                            continue
                        p1 = gp_Pnt()
                        p2 = gp_Pnt()
                        try:
                            ext.Points(k, p1, p2)
                        except Exception:
                            continue
                        mid = np.array([
                            (p1.X() + p2.X()) * 0.5,
                            (p1.Y() + p2.Y()) * 0.5,
                            (p1.Z() + p2.Z()) * 0.5,
                        ])
                        candidates.append(mid)
                        candidate_triples.append(triple)
                        n_extrema_accepted += 1

    print(f"[compute_vertices_extrema] examined {n_pairs_examined} "
          f"curve pairs (triangles in adj); "
          f"skipped {n_pairs_skipped_3rd} pairs with missing 3rd edge")
    print(f"[compute_vertices_extrema] {n_extrema_total} total extrema, "
          f"{n_extrema_accepted} accepted (d ≤ {threshold})")
    for i, (pos, tri) in enumerate(zip(candidates, candidate_triples)):
        print(f"  cand {i}: ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})  "
              f"triple={sorted(tri)}")

    if not candidates:
        return np.empty((0, 3), dtype=np.float64), []

    candidates_arr = np.array(candidates, dtype=np.float64)
    used = np.zeros(len(candidates_arr), dtype=bool)
    merged_positions   = []
    merged_triple_sets = []  # list of set-of-frozensets
    for idx in range(len(candidates_arr)):
        if used[idx]:
            continue
        dists = np.linalg.norm(candidates_arr - candidates_arr[idx], axis=1)
        close = (dists < threshold) & ~used
        merged_pos      = candidates_arr[close].mean(axis=0)
        triple_set      = set()
        merged_indices  = np.where(close)[0]
        for ci in merged_indices:
            triple_set.add(candidate_triples[ci])
        v_idx     = len(merged_positions)
        n_merged  = int(close.sum())
        print(f"  v{v_idx}: merged {n_merged} candidates "
              f"[{', '.join(str(int(i)) for i in merged_indices)}] → "
              f"({merged_pos[0]:.6f}, {merged_pos[1]:.6f}, "
              f"{merged_pos[2]:.6f})  "
              f"triples={[sorted(t) for t in triple_set]}")
        merged_positions.append(merged_pos)
        merged_triple_sets.append(triple_set)
        used[close] = True

    print(f"[compute_vertices_extrema] {len(merged_positions)} vertices "
          f"after dedup (threshold={threshold})")

    # For each merged vertex, edges = union over triples of the pairwise edges within the triple that exist in adj
    merged_edge_sets = []
    for v_idx, (pos, triple_set) in enumerate(
            zip(merged_positions, merged_triple_sets)):
        edge_set = set()
        for triple in triple_set:
            surfs = sorted(triple)
            for ii in range(len(surfs)):
                for jj in range(ii + 1, len(surfs)):
                    e = (surfs[ii], surfs[jj])
                    if e in all_edge_keys:
                        edge_set.add(e)
        if edge_set:
            print(f"  v{v_idx}: edges={sorted(edge_set)}  "
                  f"(from {len(triple_set)} triple(s))")
        else:
            print(f"  v{v_idx}: NO valid edges — vertex will be dropped")
        merged_edge_sets.append(edge_set)

    keep = [i for i, es in enumerate(merged_edge_sets) if es]
    if len(keep) < len(merged_positions):
        n_drop = len(merged_positions) - len(keep)
        print(f"[compute_vertices_extrema] dropping {n_drop} vertices "
              f"with no valid edges")
        merged_positions = [merged_positions[i] for i in keep]
        merged_edge_sets = [merged_edge_sets[i] for i in keep]

    print(f"[compute_vertices_extrema] {len(merged_positions)} vertices "
          f"after attribution")

    return (np.array(merged_positions, dtype=np.float64) if merged_positions
            else np.empty((0, 3), dtype=np.float64)), merged_edge_sets


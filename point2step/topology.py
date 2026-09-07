import math
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

try:
    from OCC.Core.Geom           import Geom_TrimmedCurve
    from OCC.Core.GeomAPI        import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp             import gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Mat, gp_Vec, gp_Quaternion
    from OCC.Core.BRep           import BRep_Builder, BRep_Tool
    from OCC.Core.TopExp         import TopExp_Explorer
    from OCC.Core.TopAbs         import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_IN
    from OCC.Core.TopoDS         import topods, TopoDS_Wire, TopoDS_Compound
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_Transform,
    )
    from OCC.Core.ShapeFix       import ShapeFix_Wire, ShapeFix_Shape
    from OCC.Core.BRepLib        import breplib
    from OCC.Core.BRepTools      import breptools
    from OCC.Core.STEPControl    import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect       import IFSelect_RetDone
    from OCC.Core.BRepAlgoAPI    import BRepAlgoAPI_Splitter
    from OCC.Core.BRepClass      import BRepClass_FaceClassifier
    from OCC.Core.TopTools       import TopTools_ListOfShape
    from OCC.Core.BRepCheck      import BRepCheck_Analyzer, BRepCheck_NoError
    from OCC.Core.BRepGProp      import brepgprop
    from OCC.Core.GProp          import GProp_GProps
    from OCC.Core.Message        import Message_ProgressRange
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False

try:
    from point2step.surface_types import SURFACE_INR
except ImportError:
    SURFACE_INR = None

# Endpoint distance below this means the curve is a closed loop (cylinder-plane circles give ~1e-17, lines ~1e-1)
CLOSURE_TOL = 1e-4


_BREP_CHECK_STATUS_NAMES = {
    0: "NoError",
    1: "InvalidPointOnCurve",
    2: "InvalidPointOnCurveOnSurface",
    3: "InvalidPointOnSurface",
    4: "No3DCurve",
    5: "Multiple3DCurve",
    6: "Invalid3DCurve",
    7: "NoCurveOnSurface",
    8: "InvalidCurveOnSurface",
    9: "InvalidCurveOnClosedSurface",
    10: "InvalidSameRangeFlag",
    11: "InvalidSameParameterFlag",
    12: "InvalidDegeneratedFlag",
    13: "FreeEdge",
    14: "InvalidMultiConnexity",
    15: "InvalidRange",
    16: "EmptyWire",
    17: "RedundantEdge",
    18: "SelfIntersectingWire",
    19: "NoSurface",
    20: "InvalidWire",
    21: "RedundantWire",
    22: "IntersectingWires",
    23: "InvalidImbricationOfWires",
    24: "EmptyShell",
    25: "RedundantFace",
    26: "InvalidToleranceValue",
    27: "UnorientableShape",
    28: "NotClosed",
    29: "NotConnected",
    30: "SubshapeNotInShape",
    31: "BadOrientation",
    32: "BadOrientationOfSubshape",
    33: "InvalidPolygonOnTriangulation",
    34: "InvalidToleranceValue",
    35: "EnclosedRegion",
    36: "CheckFail",
}


def _status_name(code):
    if isinstance(code, int):
        return _BREP_CHECK_STATUS_NAMES.get(code, f"Unknown({code})")
    try:
        val = int(code)
        return _BREP_CHECK_STATUS_NAMES.get(val, f"Unknown({val})")
    except (TypeError, ValueError):
        return str(code)


def _extract_status_errors(status_list):
    errors = []
    try:
        it = status_list.begin()
        end = status_list.end()
        while it != end:
            s = it.Value()
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
            it.Next()
        return errors
    except Exception:
        pass
    try:
        for s in status_list:
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
        return errors
    except Exception:
        pass
    try:
        for k in range(status_list.Length()):
            s = status_list.Value(k + 1)
            if s != BRepCheck_NoError:
                errors.append(_status_name(s))
        return errors
    except Exception:
        pass
    return [f"(could not iterate: {type(status_list).__name__})"]


def _print_brep_check_details(analyzer, shape):
    _shape_type_names = {
        TopAbs_VERTEX: "Vertex", TopAbs_EDGE: "Edge", TopAbs_WIRE: "Wire",
        TopAbs_FACE: "Face", TopAbs_SHELL: "Shell", TopAbs_SOLID: "Solid",
    }
    any_errors = False
    for stype in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE,
                  TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID):
        exp = TopExp_Explorer(shape, stype)
        idx = 0
        while exp.More():
            sub = exp.Current()
            name = _shape_type_names.get(stype, str(stype))
            try:
                result = analyzer.Result(sub)
                if result is None:
                    idx += 1
                    exp.Next()
                    continue
                errors = _extract_status_errors(result.Status())
                ctx_errors = []
                try:
                    ctx_list = result.StatusOnShape(shape)
                    ctx_errors = _extract_status_errors(ctx_list)
                except Exception:
                    pass
                for ptype in (TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID):
                    if ptype == stype:
                        continue
                    pexp = TopExp_Explorer(shape, ptype)
                    pidx = 0
                    while pexp.More():
                        try:
                            pctx = result.StatusOnShape(pexp.Current())
                            perrs = _extract_status_errors(pctx)
                            if perrs:
                                pname = _shape_type_names.get(ptype, str(ptype))
                                ctx_errors.extend(
                                    f"{e} (in {pname} {pidx})" for e in perrs)
                        except Exception:
                            pass
                        pidx += 1
                        pexp.Next()

                all_errors = errors + ctx_errors
                if all_errors:
                    any_errors = True
                    print(f"  [BRepCheck] {name} {idx}: {all_errors}")
            except Exception as exc:
                any_errors = True
                print(f"  [BRepCheck] {name} {idx}: error reading status: {exc}")
            idx += 1
            exp.Next()
    if not any_errors:
        print("  [BRepCheck] analyzer reported invalid but no specific errors found")


def _endpoint_dist(curve):
    t0 = curve.FirstParameter()
    t1 = curve.LastParameter()
    p0 = curve.Value(t0)
    p1 = curve.Value(t1)
    return math.sqrt(
        (p1.X() - p0.X()) ** 2 +
        (p1.Y() - p0.Y()) ** 2 +
        (p1.Z() - p0.Z()) ** 2
    )


def curve_is_closed(curve):
    return _endpoint_dist(curve) < CLOSURE_TOL


def _basis_curve(curve):
    if not hasattr(curve, "BasisCurve"):
        return None
    try:
        basis = curve.BasisCurve()
        return basis  # may be None if the handle is null
    except Exception:
        return None


def _arc_key(arc):
    ei, ej = arc["edge_key"]
    return (ei, ej, arc["v_start"], arc["v_end"], id(arc["curve"]))


def _make_arc(source_curve, t_start, t_end):
    return Geom_TrimmedCurve(source_curve, t_start, t_end)


def _project_vertex_on_curve(vertex_pos, curve, t_min, t_max):
    pnt  = gp_Pnt(float(vertex_pos[0]), float(vertex_pos[1]), float(vertex_pos[2]))
    proj = GeomAPI_ProjectPointOnCurve(pnt, curve, t_min, t_max)
    if proj.NbPoints() == 0:
        return None, None
    t_star = proj.LowerDistanceParameter()
    return t_star, proj.LowerDistance()


def _pnt_to_np(pnt):
    return np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64)


def build_edge_arcs(intersections, vertices, vertex_edges, threshold=1e-4):
    # Work on mutable copies so seam vertices can be appended when needed.
    verts      = list(vertices)
    vedge_sets = [s.copy() for s in vertex_edges]

    edge_arcs = {}

    for edge_key, curves in intersections.items():
        arcs_for_edge = []

        for curve in curves:
            t_min  = curve.FirstParameter()
            t_max  = curve.LastParameter()
            closed = curve_is_closed(curve)

            incident_params = []   # list of (t*, vertex_index)

            for v_idx, v_edge_set in enumerate(vedge_sets):
                if edge_key not in v_edge_set:
                    continue
                t_star, dist = _project_vertex_on_curve(
                    verts[v_idx], curve, t_min, t_max
                )
                if t_star is None:
                    continue
                if dist > threshold:
                    continue
                incident_params.append((t_star, v_idx))

            incident_params.sort(key=lambda x: x[0])
            k = len(incident_params)

            if closed:
                if k == 0:
                    arcs_for_edge.append({
                        "curve":   _make_arc(curve, t_min, t_max),
                        "v_start": None,
                        "v_end":   None,
                        "t_start": t_min,
                        "t_end":   t_max,
                        "closed":  True,
                        "edge_key": edge_key,
                    })
                else:
                    span = t_max - t_min

                    for m in range(k - 1):
                        t_a, v_a = incident_params[m]
                        t_b, v_b = incident_params[m + 1]
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_a, t_b),
                            "v_start": v_a,
                            "v_end":   v_b,
                            "t_start": t_a,
                            "t_end":   t_b,
                            "closed":  False,
                            "edge_key": edge_key,
                        })

                    # Wrap-around arc: [t_k, t_1 + span] on the periodic basis.
                    t_last,  v_last  = incident_params[-1]
                    t_first, v_first = incident_params[0]
                    t_wrap_end = t_first + span

                    # The wrap-around arc needs a curve accepting parameters beyond [t_min, t_max]: a Geom_TrimmedCurve's periodic basis, a raw closed curve directly, or a seam split for a non-periodic closed curve
                    basis = _basis_curve(curve)
                    if basis is None and curve_is_closed(curve):
                        basis = curve
                    wrap_ok = False
                    if basis is not None:
                        try:
                            arcs_for_edge.append({
                                "curve":   _make_arc(basis, t_last, t_wrap_end),
                                "v_start": v_last,
                                "v_end":   v_first,
                                "t_start": t_last,
                                "t_end":   t_wrap_end,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            wrap_ok = True
                        except Exception:
                            pass  # OCC rejected wrap parameters (e.g. BSpline)
                    if not wrap_ok and curve_is_closed(curve):
                        # Seam-split fallback: emit two sub-arcs joined at a seam vertex placed at the curve start/end point.
                        seam_pos = _pnt_to_np(curve.Value(t_min))
                        seam_idx = len(verts)
                        verts.append(seam_pos)
                        vedge_sets.append({edge_key})
                        try:
                            arcs_for_edge.append({
                                "curve":   _make_arc(curve, t_last, t_max),
                                "v_start": v_last,
                                "v_end":   seam_idx,
                                "t_start": t_last,
                                "t_end":   t_max,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            arcs_for_edge.append({
                                "curve":   _make_arc(curve, t_min, t_first),
                                "v_start": seam_idx,
                                "v_end":   v_first,
                                "t_start": t_min,
                                "t_end":   t_first,
                                "closed":  False,
                                "edge_key": edge_key,
                            })
                            wrap_ok = True
                        except Exception:
                            pass

            else:
                if k == 0:
                    # An open curve with no incident vertices would give a V(e) = empty open candidate edge, which is topologically inconsistent; closed loops with V(e) = empty come from the branch above
                    pass
                else:
                    for m in range(k - 1):
                        t_a, v_a = incident_params[m]
                        t_b, v_b = incident_params[m + 1]
                        arcs_for_edge.append({
                            "curve":   _make_arc(curve, t_a, t_b),
                            "v_start": v_a,
                            "v_end":   v_b,
                            "t_start": t_a,
                            "t_end":   t_b,
                            "closed":  False,
                            "edge_key": edge_key,
                        })

        for arc_idx, arc in enumerate(arcs_for_edge):
            arc["arc_idx"] = arc_idx
        edge_arcs[edge_key] = arcs_for_edge

    out_vertices = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    return edge_arcs, out_vertices, vedge_sets


def _score_vertex(vpos, involved_clusters, cluster_trees, cluster_nn_percentiles):
    ratios = []
    for k in involved_clusters:
        d, _ = cluster_trees[k].query(vpos, k=1)
        p = cluster_nn_percentiles[k]
        if p > 0:
            ratios.append(d / p)
        else:
            ratios.append(float('inf'))
    if not ratios:
        return float('inf')
    ratios.sort()
    # Use the three best ratios, the genuine triangle's clusters; with <= 3 clusters this is the max over all
    return ratios[min(2, len(ratios) - 1)]


def _score_arc(arc, cluster_i, cluster_j, cluster_trees, cluster_nn_percentiles,
               n_samples=10, sample_fraction=0.5):
    t0, t1 = arc["t_start"], arc["t_end"]
    t_mid = (t0 + t1) / 2
    half_span = (t1 - t0) * sample_fraction / 2
    t_start = t_mid - half_span
    t_end = t_mid + half_span

    tree_i = cluster_trees[cluster_i]
    tree_j = cluster_trees[cluster_j]
    p_i = cluster_nn_percentiles[cluster_i]
    p_j = cluster_nn_percentiles[cluster_j]

    ratios = []
    for k in range(n_samples):
        t = t_start + (t_end - t_start) * k / max(n_samples - 1, 1)
        try:
            p = arc["curve"].Value(t)
            pt = np.array([p.X(), p.Y(), p.Z()])
            d_i, _ = tree_i.query(pt, k=1)
            d_j, _ = tree_j.query(pt, k=1)
            r_i = d_i / p_i if p_i > 0 else float('inf')
            r_j = d_j / p_j if p_j > 0 else float('inf')
            ratios.append(max(r_i, r_j))
        except Exception:
            pass
    return np.mean(ratios) if ratios else float('inf')


def _vertex_degree(v_idx, work_arcs):
    deg = 0
    for arcs in work_arcs.values():
        for arc in arcs:
            if arc.get("closed"):
                continue
            deg += (arc["v_start"] == v_idx) + (arc["v_end"] == v_idx)
    return deg


def _non_eulerian_faces_direct(work_arcs):
    face_degree = defaultdict(lambda: defaultdict(int))
    for (i, j), arcs in work_arcs.items():
        for arc in arcs:
            if arc.get("closed"):
                continue
            vs, ve = arc["v_start"], arc["v_end"]
            face_degree[i][vs] += 1
            face_degree[i][ve] += 1
            face_degree[j][vs] += 1
            face_degree[j][ve] += 1
    bad = set()
    for face_idx, vdeg in face_degree.items():
        for v, deg in vdeg.items():
            if deg % 2 != 0:
                bad.add(face_idx)
                break
    return bad


def _apply_removals(edge_arcs, vertices, vertex_edges, removed_vertices,
                    removed_arc_keys):
    work_arcs = {}
    for edge_key, arcs in edge_arcs.items():
        kept = []
        for arc_idx, arc in enumerate(arcs):
            if (edge_key, arc_idx) in removed_arc_keys:
                continue
            kept.append(dict(arc))
        work_arcs[edge_key] = kept

    for edge_key in list(work_arcs.keys()):
        work_arcs[edge_key] = [
            arc for arc in work_arcs[edge_key]
            if arc.get("closed") or
            (arc["v_start"] not in removed_vertices and
             arc["v_end"] not in removed_vertices)
        ]

    all_removed = set(removed_vertices)
    for v_idx in range(len(vertices)):
        if v_idx in all_removed:
            continue
        if _vertex_degree(v_idx, work_arcs) == 0:
            all_removed.add(v_idx)

    surviving_v = sorted(set(range(len(vertices))) - all_removed)
    v_remap = {old: new for new, old in enumerate(surviving_v)}

    new_vertices = vertices[surviving_v]
    new_vertex_edges = [vertex_edges[i] for i in surviving_v]

    new_edge_arcs = {}
    for edge_key, arcs in work_arcs.items():
        kept = []
        for arc in arcs:
            if arc.get("closed"):
                kept.append(arc)
                continue
            vs, ve = arc["v_start"], arc["v_end"]
            if vs in v_remap and ve in v_remap:
                arc = dict(arc)
                arc["v_start"] = v_remap[vs]
                arc["v_end"] = v_remap[ve]
                kept.append(arc)
        new_edge_arcs[edge_key] = kept

    return new_edge_arcs, new_vertices, new_vertex_edges


def ilp_topology_filter(edge_arcs, vertices, vertex_edges,
                        clusters, cluster_trees, cluster_nn_percentiles,
                        occ_surfaces, surface_ids=None,
                        tolerance=1e-3,
                        lam=20, omit_sewing=False):
    n_v = len(vertices)
    n_a_total = sum(len(a) for a in edge_arcs.values())
    print(f"[ILP filter] input: {n_v} vertices, {n_a_total} arcs")

    arc_list = []       # (edge_key, arc_idx, arc_dict)
    for edge_key, arcs in edge_arcs.items():
        for arc_idx, arc in enumerate(arcs):
            arc_list.append((edge_key, arc_idx, arc))
    n_arcs = len(arc_list)
    n_verts = len(vertices)

    if n_arcs == 0:
        print("[ILP filter] no arcs — nothing to optimize")
        return edge_arcs, vertices, vertex_edges, None, {"valid": False, "n_faces": 0}

    arc_scores = []
    for edge_key, arc_idx, arc in arc_list:
        i, j = edge_key
        s = _score_arc(arc, i, j, cluster_trees, cluster_nn_percentiles)
        arc_scores.append(s)
        print(f"  arc {edge_key}[{arc_idx}] score={s:.4f}")

    vertex_scores = []
    for v_idx in range(n_verts):
        involved = set()
        for edge in vertex_edges[v_idx]:
            involved.update(edge)
        s = _score_vertex(vertices[v_idx], involved, cluster_trees,
                          cluster_nn_percentiles)
        vertex_scores.append(s)
        print(f"  v{v_idx:>3d} score={s:.4f}")

    bad_faces = _non_eulerian_faces_direct(edge_arcs)
    if not bad_faces:
        print("[ILP filter] all faces Eulerian — skipping ILP, building directly")
        fa = face_arc_incidence(edge_arcs)
        fw = assemble_wires(fa)
        shape, info = build_brep_shape_direct(
            fa, occ_surfaces, vertices, surface_ids=surface_ids,
            face_wires=fw, tolerance=tolerance, clusters=clusters,
            cluster_trees=cluster_trees, omit_sewing=omit_sewing,
        )
        valid = info.get("valid", False) and info.get("n_faces", 0) > 0
        print(f"[ILP filter] BRep: {info.get('n_faces', 0)} faces, valid={valid}")
        return edge_arcs, vertices, vertex_edges, shape, info

    print(f"[ILP filter] non-Eulerian faces: {sorted(bad_faces)} — running ILP")

    face_vertex_arcs = defaultdict(list)  # (face_idx, vertex_idx) -> [arc indices in arc_list]
    for k, (edge_key, arc_idx, arc) in enumerate(arc_list):
        if arc.get("closed"):
            continue
        vs, ve = arc["v_start"], arc["v_end"]
        if vs is None or ve is None:
            # Open arc with no attributed vertices — will be dropped by _apply_removals anyway; exclude from topology constraints.
            continue
        i, j = edge_key
        face_vertex_arcs[(i, vs)].append(k)
        face_vertex_arcs[(i, ve)].append(k)
        face_vertex_arcs[(j, vs)].append(k)
        face_vertex_arcs[(j, ve)].append(k)

    # One z variable per (face, vertex) pair
    fv_pairs = list(face_vertex_arcs.keys())
    n_z = len(fv_pairs)
    # Variable layout: [a_0 .. a_{n_arcs-1}, v_0 .. v_{n_verts-1}, z_0 .. z_{n_z-1}], a and v binary, z non-negative integer
    n_vars = n_arcs + n_verts + n_z

    print(f"[ILP filter] variables: {n_arcs} arcs + {n_verts} vertices + "
          f"{n_z} parity aux = {n_vars} total")

    c = np.zeros(n_vars)
    for k in range(n_arcs):
        c[k] = arc_scores[k] - lam
    for m in range(n_verts):
        c[n_arcs + m] = vertex_scores[m] - lam

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for idx in range(n_z):
        ub[n_arcs + n_verts + idx] = n_arcs  # generous upper bound
    integrality = np.ones(n_vars)  # all integer

    A_rows = []
    b_lb = []
    b_ub = []

    # C1: Eulerian parity  —  sum_{k in A_f(m)} a_k  -  2 * z_{f,m} = 0
    for z_idx, fv_pair in enumerate(fv_pairs):
        row = np.zeros(n_vars)
        for k in face_vertex_arcs[fv_pair]:
            row[k] += 1.0
        row[n_arcs + n_verts + z_idx] = -2.0
        A_rows.append(row)
        b_lb.append(0.0)
        b_ub.append(0.0)

    # C2: Vertex implication  —  v_m - a_k >= 0  for each open arc's endpoints
    for k, (edge_key, arc_idx, arc) in enumerate(arc_list):
        if arc.get("closed"):
            continue
        vs, ve = arc["v_start"], arc["v_end"]
        if vs is None or ve is None:
            continue
        for v_idx in (vs, ve):
            row = np.zeros(n_vars)
            row[n_arcs + v_idx] = 1.0
            row[k] = -1.0
            A_rows.append(row)
            b_lb.append(0.0)
            b_ub.append(np.inf)

    if A_rows:
        A = np.array(A_rows)
        constraints = LinearConstraint(A, b_lb, b_ub)
    else:
        constraints = None

    print(f"[ILP filter] constraints: {len(A_rows)} rows "
          f"({len(fv_pairs)} parity + {len(A_rows) - len(fv_pairs)} vertex impl.)")

    bounds = Bounds(lb, ub)
    result = milp(c, constraints=constraints, integrality=integrality,
                  bounds=bounds)

    if not result.success:
        print(f"[ILP filter] WARNING: solver failed — {result.message}")
        print("[ILP filter] falling back to keeping all arcs/vertices")
        keep_arcs = set(range(n_arcs))
        keep_verts = set(range(n_verts))
    else:
        x = result.x
        # Verify binary enforcement: all arc/vertex vars must round to {0,1}
        binary_tol = 1e-6
        nonbinary = []
        for i in range(n_arcs + n_verts):
            xi = float(x[i])
            if min(abs(xi), abs(xi - 1.0)) > binary_tol:
                nonbinary.append((i, xi))
        if nonbinary:
            print(f"[ILP filter] WARNING: {len(nonbinary)} non-binary vars "
                  f"(first 5: {nonbinary[:5]})")
        else:
            print(f"[ILP filter] binary check OK: all {n_arcs + n_verts} "
                  f"arc/vertex vars in {{0,1}} (tol={binary_tol})")

        arc_x = [float(x[k]) for k in range(n_arcs)]
        vert_x = [float(x[n_arcs + m]) for m in range(n_verts)]
        print(f"[ILP filter] arc vars x: {arc_x}")
        print(f"[ILP filter] vert vars x: {vert_x}")

        keep_arcs = {k for k in range(n_arcs) if x[k] > 0.5}
        keep_verts = {m for m in range(n_verts) if x[n_arcs + m] > 0.5}
        obj_val = result.fun
        print(f"[ILP filter] solved — objective={obj_val:.4f}, "
              f"keeping {len(keep_arcs)}/{n_arcs} arcs, "
              f"{len(keep_verts)}/{n_verts} vertices")

    removed_arcs = set(range(n_arcs)) - keep_arcs
    removed_verts = set(range(n_verts)) - keep_verts
    for k in sorted(removed_arcs):
        edge_key, arc_idx, arc = arc_list[k]
        print(f"[ILP filter] removed arc {edge_key}[{arc_idx}] "
              f"t=[{arc['t_start']:.4f},{arc['t_end']:.4f}] "
              f"score={arc_scores[k]:.4f}")
    for m in sorted(removed_verts):
        print(f"[ILP filter] removed vertex v{m} score={vertex_scores[m]:.4f}")

    removed_vertex_set = set(range(n_verts)) - keep_verts
    removed_arc_keys = set()
    for k in removed_arcs:
        edge_key, arc_idx, _ = arc_list[k]
        removed_arc_keys.add((edge_key, arc_idx))

    ea, verts, ve = _apply_removals(edge_arcs, vertices, vertex_edges,
                                    removed_vertex_set, removed_arc_keys)

    # C1 forbids degree-1 vertices globally, so any occurrence indicates a constraint-generation bug
    deg_hist = defaultdict(int)
    deg1_details = []
    partial_attrib = []
    anomalous = []  # (v_idx, degree, per_edge_counts, attrib_edges)
    for v_idx in range(len(verts)):
        per_edge = defaultdict(int)
        incident = []
        for ek, arcs in ea.items():
            for ai, a in enumerate(arcs):
                if a.get("closed"):
                    continue
                touches = (a.get("v_start") == v_idx) + (a.get("v_end") == v_idx)
                if touches:
                    per_edge[ek] += touches
                    incident.append((ek, ai, a.get("v_start"), a.get("v_end")))
        d = sum(per_edge.values())
        deg_hist[d] += 1
        if d == 1:
            deg1_details.append((v_idx, incident[0]))
        if d not in (0, 3):
            attrib_edges = (set(ve[v_idx])
                            if v_idx < len(ve) and ve[v_idx] is not None
                            else set())
            anomalous.append((v_idx, d, dict(per_edge), attrib_edges))
    for ek, arcs in ea.items():
        for ai, a in enumerate(arcs):
            if a.get("closed"):
                continue
            vs, ve_ = a.get("v_start"), a.get("v_end")
            if (vs is None) != (ve_ is None):
                partial_attrib.append((ek, ai, vs, ve_))
    print(f"[ILP filter] post-removal vertex degree histogram: "
          f"{dict(sorted(deg_hist.items()))}")
    if partial_attrib:
        print(f"[ILP filter] WARNING: {len(partial_attrib)} partial-attribution "
              f"arc(s) (one endpoint int, other None) — skipped by C1:")
        for ek, ai, vs, ve_ in partial_attrib[:10]:
            print(f"  arc {ek}[{ai}]  v_start={vs}  v_end={ve_}")
    if deg1_details:
        print(f"[ILP filter] WARNING: {len(deg1_details)} degree-1 vertex(es) "
              f"(violates per-face Eulerian guarantee):")
        for v_idx, (ek, ai, vs, ve_) in deg1_details[:10]:
            print(f"  v{v_idx}  touched only by arc {ek}[{ai}]  "
                  f"(arc v_start={vs} v_end={ve_})")
    if anomalous:
        print(f"[ILP filter] anomalous-degree vertex breakdown "
              f"(degree not in {{0, 3}}):")
        for v_idx, d, per_edge, attrib_edges in sorted(
                anomalous, key=lambda x: (-x[1], x[0])):
            attrib_faces = sorted({f for ek in attrib_edges for f in ek})
            print(f"  v{v_idx}  degree={d}  attrib_triple={attrib_faces}  "
                  f"attrib_edges={sorted(attrib_edges)}")
            for ek in sorted(per_edge.keys()):
                tag = "" if ek in attrib_edges else "  [NON-ATTRIB]"
                print(f"    edge {ek}: {per_edge[ek]} arc(s){tag}")
            missing = sorted(attrib_edges - set(per_edge.keys()))
            for ek in missing:
                print(f"    edge {ek}: 0 arc(s)  [attrib edge, no surviving arcs]")

    bad_faces = _non_eulerian_faces_direct(ea)
    if bad_faces:
        print(f"[ILP filter] WARNING: non-Eulerian faces after ILP: {sorted(bad_faces)}")
    fa = face_arc_incidence(ea)
    fw = assemble_wires(fa)
    shape, info = build_brep_shape_direct(
        fa, occ_surfaces, verts, surface_ids=surface_ids,
        face_wires=fw, tolerance=tolerance, clusters=clusters,
        cluster_trees=cluster_trees, omit_sewing=omit_sewing,
    )

    valid = info.get("valid", False) and info.get("n_faces", 0) > 0
    print(f"[ILP filter] BRep: {info.get('n_faces', 0)} faces, valid={valid}")

    return ea, verts, ve, shape, info


def face_arc_incidence(edge_arcs):
    face_arcs = {}

    for edge_key, arcs in edge_arcs.items():
        fi, fj = edge_key
        for arc in arcs:
            face_arcs.setdefault(fi, []).append(arc)
            face_arcs.setdefault(fj, []).append(arc)

    return face_arcs


def print_face_arcs_summary(face_arcs):
    print(f"[face arcs] {len(face_arcs)} faces")
    for face_idx in sorted(face_arcs):
        arcs   = face_arcs[face_idx]
        closed = sum(1 for a in arcs if a["closed"])
        open_  = len(arcs) - closed
        print(f"  face {face_idx:2d}  arcs={len(arcs)}  (open={open_}  closed={closed})")


def assemble_wires(face_arcs):
    face_wires = {}

    for face_idx, arcs in face_arcs.items():
        wires = []

        # Trivial wires: each closed arc is its own single-arc wire.
        open_arcs = []
        for arc in arcs:
            if arc["closed"]:
                wires.append([(arc, True)])
            else:
                open_arcs.append(arc)

        if open_arcs:
            adj = {}   # v -> list of (arc, exit_v, forward)
            for arc in open_arcs:
                vs, ve = arc["v_start"], arc["v_end"]
                adj.setdefault(vs, []).append((arc, ve,  True))
                adj.setdefault(ve, []).append((arc, vs, False))

            for v, neighbours in adj.items():
                deg = len(neighbours)
                if deg % 2 != 0:
                    print(
                        f"[topology] face {face_idx}: vertex {v} has odd degree {deg}"
                    )

            arc_index    = {_arc_key(a): idx for idx, a in enumerate(open_arcs)}
            visited_arcs = set()

            for start_idx, start_arc in enumerate(open_arcs):
                if start_idx in visited_arcs:
                    continue

                wire     = []
                v_target = start_arc["v_start"]
                arc      = start_arc
                forward  = True
                v_cur    = start_arc["v_end"]
                prev_arc = start_arc

                wire.append((arc, forward))
                visited_arcs.add(start_idx)

                broken = False
                while v_cur != v_target:
                    candidates = [
                        (a, exit_v, fwd)
                        for (a, exit_v, fwd) in adj.get(v_cur, [])
                        if arc_index[_arc_key(a)] not in visited_arcs
                    ]
                    if not candidates:
                        print(
                            f"[topology] face {face_idx}: open chain at vertex "
                            f"{v_cur} — wire left incomplete"
                        )
                        broken = True
                        break

                    if len(candidates) > 1:
                        # Edge continuity: prefer arcs from the same edge
                        prev_edge = prev_arc.get("edge_key")
                        same_edge = [c for c in candidates
                                     if c[0].get("edge_key") == prev_edge]
                        if same_edge:
                            arc, v_cur, forward = same_edge[0]
                        else:
                            arc, v_cur, forward = candidates[0]
                    else:
                        arc, v_cur, forward = candidates[0]

                    wire.append((arc, forward))
                    visited_arcs.add(arc_index[_arc_key(arc)])
                    prev_arc = arc

                if not broken:
                    wires.append(wire)

        face_wires[face_idx] = wires

    return face_wires


def print_face_wires_summary(face_wires):
    print(f"[wire assembly] {len(face_wires)} faces")
    for face_idx in sorted(face_wires):
        wires = face_wires[face_idx]
        print(f"  face {face_idx:2d}  wires={len(wires)}")
        for w_idx, wire in enumerate(wires):
            arc_descs = []
            for arc, fwd in wire:
                vs = arc["v_start"]
                ve = arc["v_end"]
                ek = arc.get("edge_key", None)
                ai = arc.get("arc_idx", "?")
                cl = "closed" if arc["closed"] else ("fwd" if fwd else "rev")
                edge_str = f"e({ek[0]}, {ek[1]})[{ai}]" if ek else "e?"
                arc_descs.append(f"({vs}→{ve}, {cl}, {edge_str})")
            print(f"    wire[{w_idx}]  arcs={len(wire)}  " + "  ".join(arc_descs))


def _wire_length(wire):
    props = GProp_GProps()
    brepgprop.LinearProperties(wire, props)
    return props.Mass()


def make_uv_bounded_face(occ_surface, cluster, uv_margin=0.05, tolerance=1e-3):
    u_vals, v_vals = [], []
    for pt in cluster:
        try:
            proj = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])),
                occ_surface,
            )
            if proj.NbPoints() > 0:
                u, v = proj.LowerDistanceParameters()
                u_vals.append(u)
                v_vals.append(v)
        except Exception:
            pass

    if not u_vals:
        return None

    umin, umax = min(u_vals), max(u_vals)
    vmin, vmax = min(v_vals), max(v_vals)
    du = uv_margin * max(umax - umin, 1e-6)
    dv = uv_margin * max(vmax - vmin, 1e-6)

    su1, su2, sv1, sv2 = occ_surface.Bounds()
    u1 = umin - du if math.isinf(su1) else max(umin - du, su1)
    u2 = umax + du if math.isinf(su2) else min(umax + du, su2)
    v1 = vmin - dv if math.isinf(sv1) else max(vmin - dv, sv1)
    v2 = vmax + dv if math.isinf(sv2) else min(vmax + dv, sv2)

    maker = BRepBuilderAPI_MakeFace(occ_surface, u1, u2, v1, v2, tolerance)
    return maker.Face() if maker.IsDone() else None


def _is_closed_surface(surface):
    u_closed = surface.IsUPeriodic() or surface.IsUClosed()
    v_closed = surface.IsVPeriodic() or surface.IsVClosed()
    if u_closed and v_closed:
        return True
    if u_closed and not v_closed:
        _, _, v1, v2 = surface.Bounds()
        # OCC uses ~1e100 for infinite bounds; finite means compact.
        if abs(v2 - v1) < 1e10:
            return True
    if v_closed and not u_closed:
        u1, u2, _, _ = surface.Bounds()
        if abs(u2 - u1) < 1e10:
            return True
    return False


def _build_face_standard(face_idx, surface, wire_items, tolerance):
    occ_wires = [w for w, _ in wire_items]

    if len(occ_wires) == 1:
        outer_wire = occ_wires[0]
        inner_wires = []
        outer_idx = 0
    else:
        lengths = [_wire_length(w) for w in occ_wires]
        length_log = "  ".join(f"w{i}={lengths[i]:.4f}"
                               for i in range(len(lengths)))
        print(f"[brep-direct] face {face_idx}: wire lengths: {length_log}")
        outer_idx = max(range(len(lengths)), key=lambda i: lengths[i])
        outer_wire = occ_wires[outer_idx]
        inner_wires = [w for i, w in enumerate(occ_wires) if i != outer_idx]

    print(f"[brep-direct] face {face_idx}: {len(occ_wires)} wire(s)")
    face_maker = BRepBuilderAPI_MakeFace(surface, outer_wire)
    for iw in inner_wires:
        face_maker.Add(iw)
    if not face_maker.IsDone():
        print(f"[brep-direct] face {face_idx}: MakeFace not done")
        return None

    face = face_maker.Face()
    n_wires = sum(1 for _ in _iter_explorer(face, TopAbs_WIRE))
    print(f"[brep-direct] face {face_idx}: {len(occ_wires)} wires "
          f"(outer=w{outer_idx}) → {n_wires} after MakeFace")
    return face


def _build_face_splitter(face_idx, surface, wire_items, tolerance,
                         cluster_tree):
    if cluster_tree is None:
        print(f"[brep-direct] face {face_idx}: closed surface but no "
              f"cluster_tree — falling back to standard MakeFace")
        return _build_face_standard(face_idx, surface, wire_items, tolerance)

    full_face_maker = BRepBuilderAPI_MakeFace(surface, tolerance)
    if not full_face_maker.IsDone():
        print(f"[brep-direct] face {face_idx}: full-domain MakeFace failed")
        return _build_face_standard(face_idx, surface, wire_items, tolerance)
    full_face = full_face_maker.Face()

    tools = TopTools_ListOfShape()
    n_edges = 0
    for wire, _ in wire_items:
        for edge in _iter_explorer(wire, TopAbs_EDGE):
            tools.Append(edge)
            n_edges += 1

    if n_edges == 0:
        print(f"[brep-direct] face {face_idx}: no edges for Splitter")
        return _build_face_standard(face_idx, surface, wire_items, tolerance)

    splitter = BRepAlgoAPI_Splitter()
    args = TopTools_ListOfShape()
    args.Append(full_face)
    splitter.SetArguments(args)
    splitter.SetTools(tools)
    splitter.Build()
    if not splitter.IsDone():
        print(f"[brep-direct] face {face_idx}: Splitter failed")
        return _build_face_standard(face_idx, surface, wire_items, tolerance)

    result = splitter.Shape()

    regions = list(_iter_explorer(result, TopAbs_FACE))
    print(f"[brep-direct] face {face_idx}: Splitter produced "
          f"{len(regions)} region(s) from {n_edges} edge(s)")

    if len(regions) == 0:
        return _build_face_standard(face_idx, surface, wire_items, tolerance)
    if len(regions) == 1:
        return topods.Face(regions[0])

    best_face = None
    best_score = float("inf")
    n_samples = 1024  # 32x32 UV grid per region
    for ri, region_shape in enumerate(regions):
        region_face = topods.Face(region_shape)
        # Get face-level UV bounds (respects trimming, unlike surface Bounds).
        u_lo2, u_hi2, v_lo2, v_hi2 = breptools.UVBounds(region_face)
        surf_handle = BRep_Tool.Surface(region_face)

        pts_inside = []
        n_u = max(int(math.sqrt(n_samples)), 4)
        n_v = max(n_samples // n_u, 4)
        for iu in range(n_u):
            u = u_lo2 + (u_hi2 - u_lo2) * (iu + 0.5) / n_u
            for iv in range(n_v):
                v = v_lo2 + (v_hi2 - v_lo2) * (iv + 0.5) / n_v
                classifier = BRepClass_FaceClassifier(
                    region_face, gp_Pnt2d(u, v), tolerance)
                if classifier.State() == TopAbs_IN:
                    p = surf_handle.Value(u, v)
                    pts_inside.append([p.X(), p.Y(), p.Z()])

        if len(pts_inside) == 0:
            print(f"[brep-direct] face {face_idx}: region {ri} — "
                  f"0 interior samples")
            continue

        pts_arr = np.array(pts_inside)
        dists, _ = cluster_tree.query(pts_arr)
        mean_dist = float(np.mean(dists))
        print(f"[brep-direct] face {face_idx}: region {ri} — "
              f"{len(pts_inside)} samples, mean_dist={mean_dist:.6f}")

        if mean_dist < best_score:
            best_score = mean_dist
            best_face = region_face

    if best_face is None:
        print(f"[brep-direct] face {face_idx}: no scorable Splitter region")
        return _build_face_standard(face_idx, surface, wire_items, tolerance)

    return best_face


def _iter_explorer(shape, shape_type):
    exp = TopExp_Explorer(shape, shape_type)
    while exp.More():
        yield exp.Current()
        exp.Next()


def build_brep_shape_direct(face_arcs, occ_surfaces, vertices, surface_ids=None,
                            face_wires=None, tolerance=1e-3, clusters=None,
                            cluster_trees=None, omit_sewing=False):
    bb = BRep_Builder()

    arc_to_edge = {}
    for arcs in face_arcs.values():
        for arc in arcs:
            key = _arc_key(arc)
            if key in arc_to_edge:
                continue
            try:
                arc_to_edge[key] = BRepBuilderAPI_MakeEdge(
                    arc["curve"]).Edge()
            except Exception as exc:
                print(f"[brep-direct] MakeEdge failed for arc on "
                      f"{arc.get('edge_key')}: {exc}")

    sewing = BRepBuilderAPI_Sewing(tolerance) if not omit_sewing else None
    direct_faces = []     # non-splitter faces when omit_sewing is True
    splitter_faces = []   # closed-surface faces bypass sewing

    for face_idx, arcs in face_arcs.items():
        if face_idx >= len(occ_surfaces) or occ_surfaces[face_idx] is None:
            continue
        surface = occ_surfaces[face_idx]

        wire_items = []   # list of (TopoDS_Wire, wire_arcs_list)
        for wire_arcs in face_wires.get(face_idx, []):
            wire = TopoDS_Wire()
            bb.MakeWire(wire)
            n_added = 0
            for arc, forward in wire_arcs:
                key = _arc_key(arc)
                if key not in arc_to_edge:
                    continue
                edge = arc_to_edge[key]
                bb.Add(wire, edge if forward else edge.Reversed())
                n_added += 1
            if n_added == 0:
                continue
            fix_w = ShapeFix_Wire()
            fix_w.Load(wire)
            fix_w.SetPrecision(tolerance)
            fix_w.FixConnected()
            healed = fix_w.Wire()
            if healed.IsNull():
                print(f"[brep-direct] face {face_idx}: ShapeFix_Wire "
                      f"produced null wire — skipping")
                continue
            wire_items.append((healed, wire_arcs))

        if not wire_items:
            print(f"[brep-direct] face {face_idx}: no wires — skipping")
            continue

        try:
            if _is_closed_surface(surface):
                ct = cluster_trees[face_idx] if cluster_trees else None
                print(f"[brep-direct] face {face_idx}: closed surface "
                      f"→ using Splitter")
                built_face = _build_face_splitter(
                    face_idx, surface, wire_items, tolerance, ct)
                is_splitter = True
            else:
                built_face = _build_face_standard(
                    face_idx, surface, wire_items, tolerance)
                is_splitter = False
        except Exception as exc:
            print(f"[brep-direct] face {face_idx}: exception: {exc}")
            built_face = None
            is_splitter = False
        if built_face is None:
            print(f"[brep-direct] face {face_idx}: face construction failed")
        elif is_splitter:
            splitter_faces.append(built_face)
        elif omit_sewing:
            direct_faces.append(built_face)
        else:
            sewing.Add(built_face)

    sewn_shape = None
    if not omit_sewing:
        print("[brep-direct] Sewing faces ...")
        sewing.Perform()
        sewn_shape = sewing.SewedShape()
    else:
        print(f"[brep-direct] omit_sewing: {len(direct_faces)} face(s) "
              f"added directly to Compound (no Sewing)")

    compound = TopoDS_Compound()
    bb.MakeCompound(compound)
    if sewn_shape is not None and not sewn_shape.IsNull():
        bb.Add(compound, sewn_shape)
    for f in direct_faces:
        bb.Add(compound, f)
    for sf in splitter_faces:
        bb.Add(compound, sf)
    print(f"[brep-direct] {len(splitter_faces)} Splitter face(s) added "
          f"outside sewing")
    shape = compound

    n_input_faces = len(face_arcs)
    if shape is None or shape.IsNull():
        print("[brep-direct] sewing produced no shape")
        return shape, {"valid": False, "n_faces": 0,
                       "n_input_faces": n_input_faces}

    print("[brep-direct] ShapeFix_Shape ...")
    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(tolerance)
    fixer.Perform()
    shape = fixer.Shape()

    try:
        breplib.SameParameter(shape, True)
        print("[brep-direct] SameParameter done")
    except Exception as exc:
        print(f"[brep-direct] SameParameter failed: {exc}")

    analyzer = BRepCheck_Analyzer(shape)
    eval_results = analyzer.IsValid()

    n_output_faces = 0
    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        n_output_faces += 1
        face_exp.Next()

    print(f"[brep-direct] BRepCheck valid: {eval_results}")
    if not eval_results:
        _print_brep_check_details(analyzer, shape)
    print(f"[brep-direct] Output faces: {n_output_faces}/{n_input_faces}")

    return shape, {
        "valid": eval_results,
        "n_faces": n_output_faces,
        "n_input_faces": n_input_faces,
    }


def export_step(shape, path):
    if shape is None or shape.IsNull():
        print(f"[step] export skipped — shape is null")
        return False
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs, True, Message_ProgressRange())
    ok = writer.Write(path) == IFSelect_RetDone
    if ok:
        print(f"STEP written to {path}")
    else:
        print(f"STEP export failed")
    return ok


def apply_inverse_normalization(shape, mean, R, scale):
    if shape is None or shape.IsNull():
        return shape

    Rt = np.asarray(R, dtype=np.float64).T          # 3x3 rotation
    mean = np.asarray(mean, dtype=np.float64)
    s = float(scale)

    # gp_Trsf is p' = ScaleFactor * RotationMatrix * p + Translation; our inverse normalization is p' = scale * R^T * p + mean
    rot_mat = gp_Mat(
        Rt[0, 0], Rt[0, 1], Rt[0, 2],
        Rt[1, 0], Rt[1, 1], Rt[1, 2],
        Rt[2, 0], Rt[2, 1], Rt[2, 2],
    )
    quat = gp_Quaternion(rot_mat)

    trsf = gp_Trsf()
    trsf.SetRotation(quat)
    trsf.SetScaleFactor(s)
    trsf.SetTranslationPart(gp_Vec(float(mean[0]), float(mean[1]), float(mean[2])))

    # BRepBuilderAPI_Transform (not GTransform) handles BSpline curves by transforming control points only — knot vectors stay unchanged.
    result = BRepBuilderAPI_Transform(shape, trsf, True)
    if not result.IsDone():
        print("[brep] apply_inverse_normalization: transform failed, returning shape as-is")
        return shape
    return result.Shape()


def print_edge_arcs_summary(edge_arcs):
    total_arcs = sum(len(v) for v in edge_arcs.values())
    print(f"[edge arcs] {len(edge_arcs)} edges → {total_arcs} arcs total")
    for edge_key, arcs in sorted(edge_arcs.items()):
        for idx, arc in enumerate(arcs):
            vs = arc["v_start"]
            ve = arc["v_end"]
            cl = "closed" if arc["closed"] else "open"
            print(
                f"  edge {edge_key}  arc[{idx}]  [{arc['t_start']:.4f}, {arc['t_end']:.4f}]"
                f"  v_start={vs}  v_end={ve}  {cl}"
            )

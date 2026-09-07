from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


METRIC_ROWS = [
    ("p_coverage",            "PC→mesh cov.",  "higher_is_better"),
    ("p_coverage_mesh_to_pc", "Mesh→PC cov.",  "higher_is_better"),
    ("residual_mean",         "Residual mean",      "lower_is_better"),
    ("chamfer_sym",           "Chamfer sym.",       "lower_is_better"),
]

TIMING_ROWS = [
    ("fit_time",   "Surface fitting"),
    ("clip_time",  "Mesh generation"),
    ("total_time", "Total"),
]

SPLITS = [
    ("primitive_only", "over the primitive class"),
    ("has_freeform",   "over the freeform class"),
    ("all",            "globally"),
]

ARROW = {"higher_is_better": "↑", "lower_is_better": "↓"}


def _safe_read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] could not read {path}: {e}")
        return None


def load_side(root):
    if root is None or not os.path.isdir(root):
        return {}
    out = {}
    for part_dir in sorted(glob.glob(os.path.join(root, "*", "part_*"))):
        parent = os.path.basename(os.path.dirname(part_dir))
        leaf = os.path.basename(part_dir)
        if not leaf.startswith("part_"):
            continue
        try:
            part_idx = int(leaf[len("part_"):])
        except ValueError:
            continue
        key = (parent, part_idx)

        metrics_path = os.path.join(part_dir, "metrics.json")
        wrapper_path = os.path.join(part_dir, "wrapper_status.json")

        metrics = _safe_read_json(metrics_path) if os.path.isfile(metrics_path) else None
        if metrics is not None and "metrics" not in metrics:
            metrics = None

        wrapper = _safe_read_json(wrapper_path) if os.path.isfile(wrapper_path) else None

        if wrapper is None:
            status = "ok" if metrics is not None else "fail"
        else:
            status = "ok" if (wrapper.get("status") == "ok" and metrics is not None) else "fail"

        out[key] = {"status": status, "metrics": metrics, "wrapper": wrapper}
    return out


def classify_from_mine(side_entry):
    if side_entry is None:
        return None
    m = side_entry.get("metrics")
    if m is None:
        return None
    flag = m.get("is_primitive_only")
    if flag is True:
        return "primitive_only"
    if flag is False:
        return "has_freeform"
    return None


def _ci95_normal(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    se = float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return m, m - 1.96 * se, m + 1.96 * se


def _collect_paired(mine, orig, keys, metric_field):
    mine_vals, orig_vals = [], []
    for k in keys:
        me = mine.get(k, {})
        oe = orig.get(k, {})
        mm = (me.get("metrics") or {}).get("metrics", {})
        om = (oe.get("metrics") or {}).get("metrics", {})
        a = mm.get(metric_field)
        b = om.get(metric_field)
        if a is None or b is None:
            continue
        mine_vals.append(a)
        orig_vals.append(b)
    return np.asarray(mine_vals, dtype=np.float64), np.asarray(orig_vals, dtype=np.float64)


def _collect_paired_timing(mine, orig, keys, timing_field):
    mine_vals, orig_vals = [], []
    for k in keys:
        me = mine.get(k, {})
        oe = orig.get(k, {})
        mt = (me.get("metrics") or {}).get("timing", {})
        ot = (oe.get("metrics") or {}).get("timing", {})
        a = mt.get(timing_field)
        b = ot.get(timing_field)
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        mine_vals.append(a)
        orig_vals.append(b)
    return np.asarray(mine_vals, dtype=np.float64), np.asarray(orig_vals, dtype=np.float64)


def metric_row(mine, orig, keys, field, direction):
    mv, ov = _collect_paired(mine, orig, keys, field)
    if mv.size == 0:
        return {"n": 0, "ours": None, "orig": None,
                "delta": None, "delta_ci95_half": None, "direction": direction}
    delta, lo, hi = _ci95_normal(mv - ov)
    return {
        "n":               int(mv.size),
        "ours":            float(mv.mean()),
        "orig":            float(ov.mean()),
        "delta":           delta,
        "delta_ci95_half": (hi - lo) / 2.0,
        "direction":       direction,
    }


def speedup_row(mine, orig, keys, field):
    mt, ot = _collect_paired_timing(mine, orig, keys, field)
    if mt.size == 0:
        return {"n": 0, "speedup": None, "ci95_half": None}
    m, lo, hi = _ci95_normal(ot / mt)
    return {"n": int(mt.size), "speedup": m, "ci95_half": (hi - lo) / 2.0}


def build_table(mine, orig, keys):
    return {
        "n_parts": len(keys),
        "metrics": {f: metric_row(mine, orig, keys, f, d)
                    for f, _, d in METRIC_ROWS},
        "speedup": {f: speedup_row(mine, orig, keys, f)
                    for f, _ in TIMING_ROWS},
    }


def _fmt(v, prec=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{prec}f}"


def _fmt_delta(v, half, prec=4):
    if v is None or half is None:
        return "-"
    return f"{v:+.{prec}f} ± {half:.{prec}f}"


def _fmt_speedup(v, half):
    if v is None or half is None:
        return "-"
    return f"×{v:.2f} ± {half:.2f}"


def print_table(title, table):
    print(f"\nMesh quality and execution time, averaged {title} "
          f"(N = {table['n_parts']})")
    print(f"  {'Metric':<22}{'Ours':>10}{'Point2CAD':>12}"
          f"       Δ = Ours − Point2CAD")
    print("  " + "-" * 66)
    for field, label, direction in METRIC_ROWS:
        r = table["metrics"][field]
        print(f"  {label + ' (' + ARROW[direction] + ')':<22}"
              f"{_fmt(r['ours']):>10}{_fmt(r['orig']):>12}"
              f"       {_fmt_delta(r['delta'], r['delta_ci95_half'])}")
    print()
    print(f"  {'Stage':<22}{'Speedup (t_orig / t_ours)':>24}")
    print("  " + "-" * 46)
    for field, label in TIMING_ROWS:
        r = table["speedup"][field]
        print(f"  {label:<22}{_fmt_speedup(r['speedup'], r['ci95_half']):>24}")


def print_reliability(rel):
    print("\nPipeline reliability over the point clouds both pipelines attempted "
          f"(N = {rel['n_attempted']})")
    print(f"  {'Pipeline':<22}{'Failures':>10}{'Success rate':>16}")
    print("  " + "-" * 48)
    for label, key in (("Ours", "ours"), ("Point2CAD", "orig")):
        s = rel[key]
        rate = "-" if s["success_rate"] is None else f"{s['success_rate']:.1%}"
        print(f"  {label:<22}{s['n_failed']:>10}{rate:>16}")


def print_scope(scope):
    n = scope["n_common_ok"]
    print(f"\nBoth pipelines completed on {n} point clouds: "
          f"{scope['n_primitive_only']} primitive class "
          f"({100.0 * scope['n_primitive_only'] / n:.1f}%), "
          f"{scope['n_has_freeform']} freeform class "
          f"({100.0 * scope['n_has_freeform'] / n:.1f}%).")
    if scope["n_unclassified"]:
        print(f"  {scope['n_unclassified']} unclassified point clouds are "
              f"counted in the global table only.")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dir_ours", default="output_mesh",
                    help="Root output dir of mesh_pipeline.py "
                         "(contains {model_id}/part_*/metrics.json)")
    ap.add_argument("--dir_orig", default=None,
                    help="Root output dir of patched original Point2CAD (optional)")
    ap.add_argument("--out_dir", default="aggregator_mesh_output",
                    help="Output directory for summary.json")
    args = ap.parse_args()

    mine = load_side(args.dir_ours)
    orig = load_side(args.dir_orig)

    mine_attempted = set(mine)
    orig_attempted = set(orig)
    common_attempted = mine_attempted & orig_attempted

    mine_ok = {k for k in mine_attempted if mine[k]["status"] == "ok"}
    orig_ok = {k for k in orig_attempted if orig[k]["status"] == "ok"}
    common_ok = sorted(mine_ok & orig_ok)

    n_attempted = len(common_attempted)
    n_mine_failed = len(common_attempted - mine_ok)
    n_orig_failed = len(common_attempted - orig_ok)
    reliability = {
        "n_attempted": n_attempted,
        "ours": {
            "n_failed":     n_mine_failed,
            "success_rate": (1.0 - n_mine_failed / n_attempted) if n_attempted else None,
        },
        "orig": {
            "n_failed":     n_orig_failed,
            "success_rate": (1.0 - n_orig_failed / n_attempted) if n_attempted else None,
        },
    }

    class_of = {k: (classify_from_mine(mine.get(k)) or classify_from_mine(orig.get(k)))
                for k in common_ok}
    keys_of = {
        "primitive_only": [k for k in common_ok if class_of[k] == "primitive_only"],
        "has_freeform":   [k for k in common_ok if class_of[k] == "has_freeform"],
        "all":            common_ok,
    }
    scope = {
        "n_common_ok":      len(common_ok),
        "n_primitive_only": len(keys_of["primitive_only"]),
        "n_has_freeform":   len(keys_of["has_freeform"]),
        "n_unclassified":   sum(1 for k in common_ok if class_of[k] is None),
    }

    tables = {name: build_table(mine, orig, keys_of[name]) for name, _ in SPLITS}

    summary = {
        "dir_ours":    args.dir_ours,
        "dir_orig":    args.dir_orig,
        "reliability": reliability,
        "scope":       scope,
        "tables":      tables,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_reliability(reliability)
    print_scope(scope)
    for name, title in SPLITS:
        print_table(title, tables[name])
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

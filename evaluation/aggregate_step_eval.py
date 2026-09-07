from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("ggplot")
matplotlib.rcParams.update({
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


DEFAULT_OUTPUT_DIR    = "output_step"
DEFAULT_INPUT_DIR     = "sample_clouds_abc_parts"
DEFAULT_PRIMITIVE_IDS = "splits/val_primitives.txt"
DEFAULT_FREEFORM_IDS  = "splits/val_freeform.txt"
DEFAULT_OUT_DIR       = "aggregator_step_output"

DEFAULT_DPI     = 600
DEFAULT_FIGSIZE = (4.0, 3.0)
DEFAULT_FORMAT  = "png"

SELECTIONS = [
    ("best_overall", "best overall"),
    ("kernel_valid", "kernel valid"),
]

BOX_COLORS = ["#5778a4", "#e49444"]


def _safe_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def _part_indices_for(model_id, input_dir):
    pattern = os.path.join(input_dir, model_id, "*.xyzc")
    return sorted(int(os.path.splitext(os.path.basename(p))[0])
                  for p in glob.glob(pattern))


def _ci95_normal(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(arr.mean())
    if arr.size < 2:
        return m, m, m
    se = float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return m, m - 1.96 * se, m + 1.96 * se


def _iqr_outlier_counts(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0, 0
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.sum(arr < lo)), int(np.sum(arr > hi))


def _distribution(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None,
                "ci95_half": None, "n_outliers_low": 0, "n_outliers_high": 0}
    mean, lo, hi = _ci95_normal(arr)
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    n_lo, n_hi = _iqr_outlier_counts(arr)
    return {
        "n":               int(arr.size),
        "mean":            mean,
        "median":          float(np.median(arr)),
        "q1":              float(q1),
        "q3":              float(q3),
        "ci95_half":       (hi - lo) / 2.0,
        "n_outliers_low":  n_lo,
        "n_outliers_high": n_hi,
    }


def collect_primitive(model_ids, output_dir, input_dir):
    records = []
    for mid in model_ids:
        for pi in _part_indices_for(mid, input_dir):
            part_dir = os.path.join(output_dir, mid, f"part_{pi}")
            ev = _safe_json(os.path.join(part_dir, "brep_eval.json"))
            records.append({
                "best_overall": ev.get("best_overall") if ev else None,
                "best_valid":   ev.get("best_valid")   if ev else None,
            })
    return records


def primitive_funnel(records):
    n_total      = len(records)
    n_step_sampl = sum(1 for r in records if r["best_overall"] is not None)
    n_step_valid = sum(1 for r in records if r["best_valid"]   is not None)
    return {
        "n_total":        n_total,
        "n_step_sampled": n_step_sampl,
        "n_empty_brep":   n_total - n_step_sampl,
        "n_step_valid":   n_step_valid,
    }


def primitive_metric_arrays(records):
    def _pick(key, field):
        return np.asarray([r[key][field] for r in records if r[key] is not None],
                          dtype=np.float64)
    return {
        "coverage": {
            "best_overall": _pick("best_overall", "coverage_norm"),
            "kernel_valid": _pick("best_valid",   "coverage_norm"),
        },
        "chamfer": {
            "best_overall": _pick("best_overall", "fidelity_norm"),
            "kernel_valid": _pick("best_valid",   "fidelity_norm"),
        },
    }


def collect_freeform(model_ids, output_dir, input_dir):
    clusters = []
    for mid in model_ids:
        for pi in _part_indices_for(mid, input_dir):
            part_dir = os.path.join(output_dir, mid, f"part_{pi}")
            ev = _safe_json(os.path.join(part_dir, "freeform_eval.json"))
            if ev is None:
                continue
            clusters.extend(ev.get("clusters", []))
    return clusters


def freeform_stats(clusters):
    paired = [(c["inr_residual_mean"], c["bspline_residual_mean"])
              for c in clusters
              if c.get("bspline_conversion_failed") is False
              and c.get("inr_residual_mean") is not None
              and c.get("bspline_residual_mean") is not None]

    n_total     = len(clusters)
    n_bs_failed = sum(1 for c in clusters
                      if c.get("bspline_conversion_failed") is True)
    inr = np.asarray([p[0] for p in paired], dtype=np.float64)
    bs  = np.asarray([p[1] for p in paired], dtype=np.float64)

    def _mean_ci(arr):
        if arr.size == 0:
            return {"n": 0, "mean": None, "ci95_half": None}
        m, lo, hi = _ci95_normal(arr)
        return {"n": int(arr.size), "mean": m, "ci95_half": (hi - lo) / 2.0}

    return {
        "n_clusters_total":     n_total,
        "n_bspline_failed":     n_bs_failed,
        "bspline_failure_rate": (n_bs_failed / n_total) if n_total else None,
        "n_paired":             len(paired),
        "inr":                  _mean_ci(inr),
        "bspline":              _mean_ci(bs),
        "delta":                _mean_ci(bs - inr),
    }


def _styled_boxplot(ax, data, tick_labels, showfliers=True):
    bp = ax.boxplot(
        data, tick_labels=tick_labels, widths=0.55,
        patch_artist=True, showfliers=showfliers,
        medianprops={"linewidth": 1.4, "color": "black"},
        boxprops={"linewidth": 0.8, "edgecolor": "black"},
        whiskerprops={"linewidth": 0.8, "color": "black"},
        capprops={"linewidth": 0.8, "color": "black"},
        flierprops={"marker": ".", "markersize": 2.5,
                    "markeredgecolor": "gray", "alpha": 0.4},
    )
    for patch, color in zip(bp['boxes'], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    return bp


def _tick_labels(arrays):
    return [f"{label}\nN={arrays[key].size}" for key, label in SELECTIONS]


def plot_coverage_boxplot(arrays, out_path, dpi, figsize, fmt):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _styled_boxplot(ax, [arrays[k] for k, _ in SELECTIONS], _tick_labels(arrays))
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.5, 2.85)
    ax.set_title("Coverage - primitive class")
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)


def plot_chamfer_boxplot(arrays, out_path, dpi, figsize, fmt):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    data = [arrays[k] for k, _ in SELECTIONS]
    _styled_boxplot(ax, data, _tick_labels(arrays), showfliers=False)

    rng = np.random.default_rng(42)
    for i, (arr, color) in enumerate(zip(data, BOX_COLORS), start=1):
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = arr[(arr < lo) | (arr > hi)]
        if out.size:
            jitter = rng.uniform(-0.13, 0.13, out.size)
            ax.scatter(i + jitter, out, s=4, c=color, alpha=0.30,
                       marker=".", linewidths=0)

    ax.set_yscale("log")
    ax.set_ylabel("One-sided Chamfer distance (log scale)")
    ax.set_xlim(0.5, 2.85)
    ax.set_title("One-sided Chamfer distance")
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)


def _fmt_pct(num, denom):
    return "-" if denom == 0 else f"{100.0 * num / denom:.1f}%"


def _fmt_sci_pm(mean, half, prec=2):
    if mean is None or half is None:
        return "-"
    mant_str, exp_str = f"{mean:.{prec}e}".split("e")
    exp = int(exp_str)
    return (f"({float(mant_str):.{prec}f} ± {half / 10.0 ** exp:.{prec}f})"
            f" × 10^{exp}")


def print_funnel(funnel):
    n = funnel["n_total"]
    rows = [
        ("Primitive class",            funnel["n_total"],        "/"),
        ("Non-empty BRep",             funnel["n_step_sampled"], _fmt_pct(funnel["n_step_sampled"], n)),
        ("Empty BRep",                 funnel["n_empty_brep"],   _fmt_pct(funnel["n_empty_brep"],   n)),
        ("Kernel-valid STEP produced", funnel["n_step_valid"],   _fmt_pct(funnel["n_step_valid"],   n)),
    ]
    print("\nBreakdown of model types for the primitive class evaluation protocol")
    print(f"  {'Model type':<30}{'Count':>8}{'Percentage':>14}")
    print("  " + "-" * 52)
    for name, count, pct in rows:
        print(f"  {name:<30}{count:>8}{pct:>14}")


def print_distribution(title, dists, fmt):
    print(f"\n{title}, over the primitive class")
    print(f"  {'Selection':<16}{'N':>8}{'Median':>14}{'Q1':>14}{'Q3':>14}"
          f"{'IQR outliers':>14}")
    print("  " + "-" * 80)
    for key, label in SELECTIONS:
        d = dists[key]
        n_out = d["n_outliers_low"] + d["n_outliers_high"]
        print(f"  {label:<16}{d['n']:>8}{fmt(d['median']):>14}{fmt(d['q1']):>14}"
              f"{fmt(d['q3']):>14}{n_out:>14}")


def print_inr_bspline(stats):
    rows = [
        ("INR residual d_INR",        stats["inr"]),
        ("B-Spline residual d_BS",    stats["bspline"]),
        ("Δ = d_BS − d_INR",          stats["delta"]),
    ]
    print(f"\nINR to B-Spline conversion over the freeform class "
          f"(N = {stats['n_paired']} clusters, "
          f"{stats['n_bspline_failed']} conversion failures)")
    print(f"  {'Quantity':<26}{'Mean ± 95% CI':>28}")
    print("  " + "-" * 54)
    for name, s in rows:
        print(f"  {name:<26}{_fmt_sci_pm(s['mean'], s['ci95_half']):>28}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--input_dir",     default=DEFAULT_INPUT_DIR)
    ap.add_argument("--primitive_ids", default=DEFAULT_PRIMITIVE_IDS)
    ap.add_argument("--freeform_ids",  default=DEFAULT_FREEFORM_IDS)
    ap.add_argument("--out_dir",       default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi",           type=int, default=DEFAULT_DPI)
    ap.add_argument("--figsize",       type=float, nargs=2,
                    default=list(DEFAULT_FIGSIZE), metavar=("W", "H"))
    ap.add_argument("--format",        default=DEFAULT_FORMAT,
                    choices=["png", "pdf"])
    args = ap.parse_args()

    prim_records = collect_primitive(_read_ids(args.primitive_ids),
                                     args.output_dir, args.input_dir)
    funnel = primitive_funnel(prim_records)
    arrays = primitive_metric_arrays(prim_records)
    coverage = {k: _distribution(arrays["coverage"][k]) for k, _ in SELECTIONS}
    chamfer  = {k: _distribution(arrays["chamfer"][k])  for k, _ in SELECTIONS}

    clusters = collect_freeform(_read_ids(args.freeform_ids),
                                args.output_dir, args.input_dir)
    free_stats = freeform_stats(clusters)

    figures_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    cov_fig = os.path.join(figures_dir, f"coverage_boxplot.{args.format}")
    ch_fig  = os.path.join(figures_dir, f"chamfer_boxplot.{args.format}")
    plot_coverage_boxplot(arrays["coverage"], cov_fig,
                          args.dpi, tuple(args.figsize), args.format)
    plot_chamfer_boxplot(arrays["chamfer"], ch_fig,
                         args.dpi, tuple(args.figsize), args.format)

    summary = {
        "output_dir":         args.output_dir,
        "primitive_ids_file": args.primitive_ids,
        "freeform_ids_file":  args.freeform_ids,
        "funnel":             funnel,
        "coverage":           coverage,
        "chamfer":            chamfer,
        "inr_bspline":        free_stats,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_funnel(funnel)
    print_distribution("Coverage", coverage, lambda v: f"{v:.4f}")
    print_distribution("One-sided Chamfer distance", chamfer,
                       lambda v: f"{v:.2e}")
    print_inr_bspline(free_stats)
    print(f"\nWrote {summary_path}")
    print(f"Wrote figures under {figures_dir}/")


if __name__ == "__main__":
    main()

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_inr_bspline import _eval_inr_cluster, _parse_cid, fit_freeform_segments


_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from step_pipeline import (
    DEFAULT_SELECTION_FRAME, DEFAULT_SPACING_FACTOR_MAX,
    DEFAULT_SPACING_FACTOR_MIN, DEFAULT_SPACING_FACTOR_STEP,
)
from point2step.inr_fitting import INR_MAX_STEPS

PIPELINE_CMD    = ["python", "-u", os.path.join(_REPO_ROOT, "step_pipeline.py")]
OUTPUT_DIR      = "output_step"
MESH_OUTPUT_DIR = "output_mesh"
INPUT_DIR       = "sample_clouds_abc_parts"
MODEL_TIMEOUT_S = 600

SPACING_FACTOR_MIN  = DEFAULT_SPACING_FACTOR_MIN
SPACING_FACTOR_MAX  = DEFAULT_SPACING_FACTOR_MAX
SPACING_FACTOR_STEP = DEFAULT_SPACING_FACTOR_STEP
SELECTION_FRAME     = DEFAULT_SELECTION_FRAME

K_EVAL_DEFAULT = 100
K_FIT_DEFAULT  = 50

LOCKED_REGIME_ARGS = [
    "--no_intermediates",
    "--save_norm_step",
    "--no_clean_output",
    "--write_part_status",
    "--debug_save_step_pc"
]

def read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def part_indices_for(model_id, input_dir):
    pattern = os.path.join(input_dir, model_id, "*.xyzc")
    paths = glob.glob(pattern)
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


TRANSIENT_CUDA_MARKERS = (
    "CUDA not available",
    "Device index must not be negative",
    "No CUDA GPUs are available",
    "CUDA-capable device(s) is/are busy or unavailable",
)


def _cleanup_cuda_unavailable(output_dir):
    if not os.path.isdir(output_dir):
        return []
    cleaned = []
    for mid in sorted(os.listdir(output_dir)):
        mdir = os.path.join(output_dir, mid)
        if not os.path.isdir(mdir):
            continue
        stderr_path = os.path.join(mdir, "stderr.log")
        if not os.path.isfile(stderr_path):
            continue
        try:
            with open(stderr_path, errors="replace") as f:
                text = f.read()
            hit = any(m in text for m in TRANSIENT_CUDA_MARKERS)
        except OSError:
            continue
        if hit:
            shutil.rmtree(mdir)
            cleaned.append(mid)
    return cleaned


def _check_transient_cuda_and_halt(mid, bar, output_dir):
    stderr_path = os.path.join(output_dir, mid, "stderr.log")
    try:
        with open(stderr_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return
    if any(m in text for m in TRANSIENT_CUDA_MARKERS):
        tqdm.tqdm.write(
            f"[wrapper] transient-CUDA marker in {stderr_path} — "
            f"exiting 42; restart container, then re-invoke")
        bar.close()
        sys.exit(42)


def _classify_model_states(output_dir, input_dir):
    done, failed = set(), set()
    if not os.path.isdir(output_dir):
        return done, failed
    for mid in os.listdir(output_dir):
        mdir = os.path.join(output_dir, mid)
        if not os.path.isdir(mdir):
            continue
        indices = part_indices_for(mid, input_dir)
        if not indices:
            continue
        all_ok = True
        for pi in indices:
            wp = os.path.join(mdir, f"part_{pi}", "wrapper_status.json")
            if not os.path.isfile(wp):
                all_ok = False
                break
            try:
                with open(wp) as f:
                    if json.load(f).get("status") != "ok":
                        all_ok = False
                        break
            except (OSError, json.JSONDecodeError):
                all_ok = False
                break
        (done if all_ok else failed).add(mid)
    return done, failed


def _kill_and_reap(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait()
    except Exception:
        pass


def _write_part_status(part_dir, status):
    os.makedirs(part_dir, exist_ok=True)
    path = os.path.join(part_dir, "wrapper_status.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"status": status}, f)
    os.replace(tmp, path)


def _fmt_variant(v):
    if v is None:
        return "—"
    return (f"sf={v['spacing']:.1f} cov={v['coverage_norm']:.3f} "
            f"fid={v['fidelity_norm']:.4f}")


def _summarize_primitive_parts(model_dir):
    lines = []
    for part_dir in sorted(glob.glob(os.path.join(model_dir, "part_*"))):
        part = os.path.basename(part_dir)
        sidecar = os.path.join(part_dir, "brep_eval.json")
        if not os.path.isfile(sidecar):
            lines.append(f"  {part}: sidecar missing")
            continue
        try:
            with open(sidecar) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            lines.append(f"  {part}: sidecar unreadable")
            continue
        bo = data.get("best_overall")
        bv = data.get("best_valid")
        if bo is None and bv is None:
            lines.append(f"  {part}: no variant produced a sampleable STEP")
            continue
        lines.append(
            f"  {part}: best_overall {_fmt_variant(bo)}  "
            f"best_valid {_fmt_variant(bv)}"
        )
    return "\n".join(lines)


def run_primitive_model(model_id, input_dir, output_dir, model_timeout_s,
                        spacing_min, spacing_max, spacing_step,
                        selection_frame):
    model_dir = os.path.join(output_dir, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    stdout_log = os.path.join(model_dir, "stdout.log")
    stderr_log = os.path.join(model_dir, "stderr.log")

    cmd = PIPELINE_CMD + [
        "--model_id",  model_id,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--spacing_factor_min",  str(spacing_min),
        "--spacing_factor_max",  str(spacing_max),
        "--spacing_factor_step", str(spacing_step),
        "--selection_frame",     selection_frame,
    ] + LOCKED_REGIME_ARGS

    t0 = time.perf_counter()
    timed_out = False
    proc = None
    rc = -1
    try:
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=model_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_and_reap(proc)
                rc = proc.returncode if proc.returncode is not None else -1
    except BaseException:
        if proc is not None and proc.poll() is None:
            _kill_and_reap(proc)
        raise
    dt = time.perf_counter() - t0

    summary = ""
    if rc == 0 and not timed_out:
        for part_dir in sorted(glob.glob(os.path.join(model_dir, "part_*"))):
            src = os.path.join(part_dir, "sweep_eval.json")
            dst = os.path.join(part_dir, "brep_eval.json")
            if os.path.isfile(src):
                shutil.move(src, dst)
        summary = _summarize_primitive_parts(model_dir)

    outcome = "ok" if (not timed_out and rc == 0) else "failed"
    return {"outcome": outcome, "rc": rc, "duration_s": dt,
            "timed_out": timed_out, "summary": summary}


def run_freeform_model(model_id, input_dir, mesh_output_dir, output_dir,
                       K_eval, K_fit, device, inr_max_steps=INR_MAX_STEPS):
    model_dir   = os.path.join(output_dir, model_id)
    src_model   = os.path.join(mesh_output_dir, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    stdout_log = os.path.join(model_dir, "stdout.log")
    stderr_log = os.path.join(model_dir, "stderr.log")

    t0 = time.perf_counter()
    indices = part_indices_for(model_id, input_dir)

    out_f = open(stdout_log, "w")
    err_f = open(stderr_log, "w")
    any_failed = False
    try:
        if not os.path.isdir(src_model):
            print(f"[freeform] no cached mesh results at {src_model}; "
                  f"every segment will be refitted", file=out_f, flush=True)

        for pi in indices:
            part_dir = os.path.join(model_dir, f"part_{pi}")
            os.makedirs(part_dir, exist_ok=True)
            src_part = os.path.join(src_model, f"part_{pi}")

            part_failed = False
            clusters = []

            inr_paths = sorted(
                glob.glob(os.path.join(src_part, "surface_inr_*.pt")),
                key=_parse_cid,
            )

            if not inr_paths:
                xyzc_path = os.path.join(input_dir, model_id, f"{pi}.xyzc")
                try:
                    print(f"[freeform] {model_id}/part_{pi}: no cached INR, "
                          f"fitting from {xyzc_path}", file=out_f, flush=True)
                    for cid, segment, payload in fit_freeform_segments(
                            xyzc_path, device, max_steps=inr_max_steps):
                        record = _eval_inr_cluster(segment, payload, device,
                                                   K_eval=K_eval, K_fit=K_fit)
                        record["cid"] = cid
                        record["from_cache"] = False
                        clusters.append(record)
                        bs = record.get("bspline_residual_mean")
                        bs_s = f"{bs:.6f}" if bs is not None else "FAIL"
                        print(f"[freeform] {model_id}/part_{pi}/cid={cid} (refit): "
                              f"inr_res={record['inr_residual_mean']:.6f}  "
                              f"bspline_res={bs_s}", file=out_f, flush=True)
                except Exception as e:
                    print(f"[freeform] {model_id}/part_{pi} refit FAILED: {e!r}",
                          file=err_f, flush=True)
                    traceback.print_exc(file=err_f)
                    part_failed = True

            for inr_path in inr_paths:
                cid = _parse_cid(inr_path)
                cluster_path = os.path.join(src_part, f"cluster_{cid}.npy")
                if not os.path.isfile(cluster_path):
                    print(f"[freeform] {model_id}/part_{pi}: cluster_{cid}.npy missing",
                          file=err_f, flush=True)
                    continue
                try:
                    cluster = np.load(cluster_path).astype(np.float32)
                    payload = torch.load(inr_path, map_location=device,
                                         weights_only=False)
                    record = _eval_inr_cluster(cluster, payload, device,
                                               K_eval=K_eval, K_fit=K_fit)
                    record["cid"] = cid
                    clusters.append(record)
                    bs = record.get("bspline_residual_mean")
                    bs_s = f"{bs:.6f}" if bs is not None else "FAIL"
                    print(f"[freeform] {model_id}/part_{pi}/cid={cid}: "
                          f"inr_res={record['inr_residual_mean']:.6f}  "
                          f"bspline_res={bs_s}",
                          file=out_f, flush=True)
                except Exception as e:
                    print(f"[freeform] {model_id}/part_{pi}/cid={cid} FAILED: {e!r}",
                          file=err_f, flush=True)
                    traceback.print_exc(file=err_f)
                    part_failed = True

            sidecar = {
                "model_id": model_id,
                "part_idx": pi,
                "K_eval":   int(K_eval),
                "K_fit":    int(K_fit),
                "clusters": clusters,
            }
            sidecar_path = os.path.join(part_dir, "freeform_eval.json")
            tmp = sidecar_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sidecar, f, indent=2)
            os.replace(tmp, sidecar_path)

            _write_part_status(part_dir, "failed" if part_failed else "ok")
            if part_failed:
                any_failed = True
    finally:
        out_f.close()
        err_f.close()

    dt = time.perf_counter() - t0
    return {"outcome": "ok" if not any_failed else "failed", "rc": 0,
            "duration_s": dt, "timed_out": False}


def _summarize_dry_run(prim_ids_path, free_ids_path, input_dir, output_dir,
                       skip_failed):
    prim_ids = read_ids(prim_ids_path) if prim_ids_path else []
    free_ids = read_ids(free_ids_path) if free_ids_path else []
    all_ids  = prim_ids + free_ids

    done, failed = _classify_model_states(output_dir, input_dir)

    n_done = n_clean = n_failed_skip = n_failed_redo = n_fresh = 0
    for mid in all_ids:
        mdir = os.path.join(output_dir, mid)
        stderr_path = os.path.join(mdir, "stderr.log")
        has_marker = False
        if os.path.isfile(stderr_path):
            try:
                with open(stderr_path, errors="replace") as f:
                    text = f.read()
                has_marker = any(m in text for m in TRANSIENT_CUDA_MARKERS)
            except OSError:
                pass
        if has_marker:
            n_clean += 1
        elif mid in done:
            n_done += 1
        elif mid in failed:
            if skip_failed:
                n_failed_skip += 1
            else:
                n_failed_redo += 1
        else:
            n_fresh += 1

    n_to_run = n_clean + n_failed_redo + n_fresh
    print("[wrapper] DRY RUN — no files modified, no subprocesses launched",
          flush=True)
    print(f"[wrapper] primitive: {len(prim_ids)} IDs  freeform: {len(free_ids)} IDs",
          flush=True)
    print(f"[wrapper]   already complete (skip):                     {n_done}",
          flush=True)
    print(f"[wrapper]   transient marker (rmtree + redo):            {n_clean}",
          flush=True)
    if skip_failed:
        print(f"[wrapper]   previously failed (--skip_failed, skip):     {n_failed_skip}",
              flush=True)
    else:
        print(f"[wrapper]   previously failed (rmtree + redo):           {n_failed_redo}",
              flush=True)
    print(f"[wrapper]   never started (no output dir):               {n_fresh}",
          flush=True)
    print(f"[wrapper] models main() will process this run: {n_to_run}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Sequential dispatcher for the BRep evaluation cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--primitive_ids", default=None,
                    help="Path to .txt of primitive-only model IDs")
    ap.add_argument("--freeform_ids",  default=None,
                    help="Path to .txt of freeform model IDs")
    ap.add_argument("--input_dir",     default=INPUT_DIR,
                    help="Root of per-part .xyzc inputs")
    ap.add_argument("--output_dir",    default=OUTPUT_DIR,
                    help="Root of the per-model BRep eval outputs.")
    ap.add_argument("--mesh_output_dir", default=MESH_OUTPUT_DIR,
                    help="Root of mesh_pipeline outputs (source of cached INRs)")
    ap.add_argument("--model_timeout_s", type=int, default=MODEL_TIMEOUT_S,
                    help="Wall-clock timeout per primitive subprocess.")
    ap.add_argument("--spacing_factor_min", type=float,
                    default=SPACING_FACTOR_MIN,
                    help="Lower bound of the spacing-factor sweep grid.")
    ap.add_argument("--spacing_factor_max", type=float,
                    default=SPACING_FACTOR_MAX,
                    help="Upper bound of the spacing-factor sweep grid.")
    ap.add_argument("--spacing_factor_step", type=float,
                    default=SPACING_FACTOR_STEP,
                    help="Step of the spacing-factor sweep grid.")
    ap.add_argument("--selection_frame", default=SELECTION_FRAME,
                    choices=["normalized", "world"],
                    help="Frame the per-part argmax is taken in.")
    ap.add_argument("--K_eval", type=int, default=K_EVAL_DEFAULT,
                    help="Per-axis UV grid resolution for the eval sampling, "
                         "same on both surfaces.")
    ap.add_argument("--K_fit",  type=int, default=K_FIT_DEFAULT,
                    help="Per-axis UV grid resolution for the BSpline fit. "
                         "Default 50 matches the production algorithm.")
    ap.add_argument("--inr_max_steps", type=int, default=INR_MAX_STEPS,
                    help="Training steps per INR when a freeform segment has no "
                         "cached fit under --mesh_output_dir and must be refitted.")
    ap.add_argument("--skip_failed",   action="store_true",
                    help="Skip previously-failed models (default: rmtree + redo).")
    ap.add_argument("--dry_run",       action="store_true",
                    help="Report what would run, take no action.")
    args = ap.parse_args()

    if args.primitive_ids is None and args.freeform_ids is None:
        ap.error("at least one of --primitive_ids or --freeform_ids must be given")

    if args.dry_run:
        _summarize_dry_run(args.primitive_ids, args.freeform_ids,
                           args.input_dir, args.output_dir, args.skip_failed)
        return

    prim_ids = read_ids(args.primitive_ids) if args.primitive_ids else []
    free_ids = read_ids(args.freeform_ids)  if args.freeform_ids  else []
    print(f"[wrapper] primitive: {len(prim_ids)} IDs  freeform: {len(free_ids)} IDs",
          flush=True)

    done, failed = _classify_model_states(args.output_dir, args.input_dir)
    cleaned = _cleanup_cuda_unavailable(args.output_dir)
    if cleaned:
        failed -= set(cleaned)
        print(f"[wrapper] removed {len(cleaned)} model dir(s) with transient-CUDA "
              f"marker — will retry: {cleaned}", flush=True)
    skip = done | (failed if args.skip_failed else set())
    prim_to_run = [m for m in prim_ids if m not in skip]
    free_to_run = [m for m in free_ids if m not in skip]
    n_done    = sum(1 for m in prim_ids + free_ids if m in done)
    n_failed  = sum(1 for m in prim_ids + free_ids if m in failed)
    n_skipped = (len(prim_ids) - len(prim_to_run)) + (len(free_ids) - len(free_to_run))
    if n_done:
        print(f"[wrapper] {n_done} models already done — skipping", flush=True)
    if n_failed:
        if args.skip_failed:
            print(f"[wrapper] --skip_failed: skipping {n_failed} previously-failed models",
                  flush=True)
        else:
            print(f"[wrapper] {n_failed} previously-failed models will be re-run "
                  f"from scratch", flush=True)

    total        = len(prim_ids) + len(free_ids)
    total_to_run = len(prim_to_run) + len(free_to_run)
    bar = tqdm.tqdm(total=total, initial=n_skipped, desc="models",
                    unit="model", dynamic_ncols=True)
    counts = {"ok": 0, "failed": 0, "skipped": n_skipped}

    def _log(mid, result):
        outcome = result["outcome"]
        counts[outcome] += 1
        rc  = result["rc"]
        dur = result["duration_s"]
        done_n = counts["ok"] + counts["failed"]
        progress = (f"[{done_n}/{total_to_run}  "
                    f"ok={counts['ok']}  failed={counts['failed']}]")
        if outcome == "ok":
            msg = f"[wrapper] {mid}: ok rc={rc} ({dur:.1f}s)  {progress}"
        elif result.get("timed_out"):
            msg = (f"[wrapper] {mid}: TIMEOUT after {dur:.1f}s "
                   f"(limit {args.model_timeout_s}s)  {progress}")
        else:
            sig = -rc if rc < 0 else None
            sig_str = f" (signal {sig})" if sig is not None else ""
            stderr_log = os.path.join(args.output_dir, mid, "stderr.log")
            msg = (f"[wrapper] {mid}: failed rc={rc}{sig_str} "
                   f"({dur:.1f}s) — see {stderr_log}  {progress}")
        tqdm.tqdm.write(msg)
        summary = result.get("summary", "")
        if summary:
            tqdm.tqdm.write(summary)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        if prim_to_run:
            tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_to_run)} models "
                            f"(sequential)")
            for mid in prim_to_run:
                bar.set_postfix_str(mid, refresh=True)
                tqdm.tqdm.write(f"[wrapper] {mid}: starting (primitive)")
                try:
                    result = run_primitive_model(
                        mid, args.input_dir, args.output_dir,
                        args.model_timeout_s,
                        args.spacing_factor_min, args.spacing_factor_max,
                        args.spacing_factor_step, args.selection_frame,
                    )
                except Exception as e:
                    tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                    counts["failed"] += 1
                else:
                    _log(mid, result)
                bar.update(1)

        if free_to_run:
            tqdm.tqdm.write(f"[wrapper] freeform phase: {len(free_to_run)} models "
                            f"(sequential, inline INR eval)")
            for mid in free_to_run:
                bar.set_postfix_str(mid, refresh=True)
                tqdm.tqdm.write(f"[wrapper] {mid}: starting (freeform)")
                try:
                    result = run_freeform_model(
                        mid, args.input_dir, args.mesh_output_dir,
                        args.output_dir, args.K_eval, args.K_fit, device,
                        inr_max_steps=args.inr_max_steps,
                    )
                except Exception as e:
                    tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                    counts["failed"] += 1
                else:
                    _log(mid, result)
                bar.update(1)
                _check_transient_cuda_and_halt(mid, bar, args.output_dir)
    finally:
        bar.close()
        print(f"[wrapper] done — ok={counts['ok']}  failed={counts['failed']}  "
              f"skipped={counts['skipped']}", flush=True)


if __name__ == "__main__":
    main()

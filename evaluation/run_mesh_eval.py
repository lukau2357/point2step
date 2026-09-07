import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import tqdm


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_CMD = ["python", "-u", os.path.join(_REPO_ROOT, "mesh_pipeline.py")]
OUTPUT_DIR = "output_mesh"
MODEL_TIMEOUT_S = 1800


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
)


def _cleanup_cuda_unavailable(output_dir, done):
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


def _check_transient_cuda_and_halt(mid, bar):
    stderr_path = os.path.join(OUTPUT_DIR, mid, "stderr.log")
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


def run_one_model(model_id, input_dir):
    model_dir = os.path.join(OUTPUT_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    stdout_log = os.path.join(model_dir, "stdout.log")
    stderr_log = os.path.join(model_dir, "stderr.log")

    cmd = PIPELINE_CMD + [
        "--model_id", model_id,
        "--input_dir", input_dir,
        "--output_dir", OUTPUT_DIR,
        "--no_clean_output",
        "--write_part_status",
    ]

    t0 = time.perf_counter()
    timed_out = False
    proc = None
    rc = -1
    try:
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=MODEL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_and_reap(proc)
                rc = proc.returncode if proc.returncode is not None else -1
    except BaseException:
        if proc is not None and proc.poll() is None:
            _kill_and_reap(proc)
        raise
    dt = time.perf_counter() - t0

    outcome = "ok" if (not timed_out and rc == 0) else "failed"
    return {"outcome": outcome, "rc": rc, "duration_s": dt,
            "timed_out": timed_out}


def _summarize_dry_run(prim_ids_path, free_ids_path, input_dir, skip_failed):
    prim_ids = read_ids(prim_ids_path) if prim_ids_path else []
    free_ids = read_ids(free_ids_path) if free_ids_path else []
    all_ids = prim_ids + free_ids

    done, failed = _classify_model_states(OUTPUT_DIR, input_dir)

    n_done = n_clean = n_failed_skip = n_failed_redo = n_fresh = 0
    for mid in all_ids:
        mdir = os.path.join(OUTPUT_DIR, mid)
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
        description="Per-model wrapper for the mesh pipeline, split primitive / freeform phases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--primitive_ids", default=None,
                    help="Path to .txt of primitive-only model IDs (run in parallel)")
    ap.add_argument("--freeform_ids", default=None,
                    help="Path to .txt of freeform model IDs (run sequentially)")
    ap.add_argument("--skip_failed", action="store_true",
                    help="Skip models whose previous run did not complete "
                         "(model dir on disk but missing wrapper_status.json "
                         "or status != 'ok'). Default: such models are rmtree'd "
                         "and re-run from scratch. Useful when failures are "
                         "deterministic OOMs and retrying just wastes time.")
    ap.add_argument("--input_dir", default="sample_clouds_abc_parts",
                    help="Root of per-part .xyzc inputs "
                         "(expects {input_dir}/{model_id}/*.xyzc)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Process pool size for the primitive phase. Raise with "
                         "care: each worker holds a full model in memory and on "
                         "the GPU, so a large pool can exhaust either.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what _cleanup_cuda_unavailable + main() would "
                         "do (rmtree/run/skip counts) and exit; performs no "
                         "rmtree and launches no subprocesses.")
    args = ap.parse_args()

    if args.primitive_ids is None and args.freeform_ids is None:
        ap.error("at least one of --primitive_ids or --freeform_ids must be given")

    if args.dry_run:
        _summarize_dry_run(args.primitive_ids, args.freeform_ids,
                           args.input_dir, args.skip_failed)
        return

    prim_ids = read_ids(args.primitive_ids) if args.primitive_ids else []
    free_ids = read_ids(args.freeform_ids)  if args.freeform_ids  else []
    print(f"[wrapper] primitive: {len(prim_ids)} IDs  freeform: {len(free_ids)} IDs",
          flush=True)

    done, failed = _classify_model_states(OUTPUT_DIR, args.input_dir)
    cleaned = _cleanup_cuda_unavailable(OUTPUT_DIR, done)
    if cleaned:
        failed -= set(cleaned)
        print(f"[wrapper] removed {len(cleaned)} model dir(s) with "
              f"'CUDA not available' in stderr — will retry: {cleaned}",
              flush=True)
    skip = done | (failed if args.skip_failed else set())
    prim_to_run = [m for m in prim_ids if m not in skip]
    free_to_run = [m for m in free_ids if m not in skip]
    n_done   = sum(1 for m in prim_ids + free_ids if m in done)
    n_failed = sum(1 for m in prim_ids + free_ids if m in failed)
    n_skipped = (len(prim_ids) - len(prim_to_run)) + (len(free_ids) - len(free_to_run))
    if n_done:
        print(f"[wrapper] {n_done} models already done on disk — skipping",
              flush=True)
    if n_failed:
        if args.skip_failed:
            print(f"[wrapper] --skip_failed: skipping {n_failed} previously-failed "
                  f"models", flush=True)
        else:
            print(f"[wrapper] {n_failed} previously-failed models will be re-run "
                  f"from scratch", flush=True)

    total = len(prim_ids) + len(free_ids)
    total_to_run = len(prim_to_run) + len(free_to_run)
    bar = tqdm.tqdm(total=total, initial=n_skipped, desc="models",
                    unit="model", dynamic_ncols=True)
    counts = {"ok": 0, "failed": 0, "skipped": n_skipped}

    def _log(mid, result):
        outcome = result["outcome"]
        counts[outcome] += 1
        rc = result["rc"]
        dur = result["duration_s"]
        done_n = counts["ok"] + counts["failed"]
        progress = (f"[{done_n}/{total_to_run}  "
                    f"ok={counts['ok']}  failed={counts['failed']}]")
        if outcome == "ok":
            msg = f"[wrapper] {mid}: ok rc={rc} ({dur:.1f}s)  {progress}"
        elif result.get("timed_out"):
            msg = (f"[wrapper] {mid}: TIMEOUT after {dur:.1f}s "
                   f"(limit {MODEL_TIMEOUT_S}s)  {progress}")
        else:
            sig = -rc if rc < 0 else None
            sig_str = f" (signal {sig})" if sig is not None else ""
            stderr_log = os.path.join(OUTPUT_DIR, mid, "stderr.log")
            msg = (f"[wrapper] {mid}: failed rc={rc}{sig_str} "
                   f"({dur:.1f}s) — see {stderr_log}  {progress}")
        tqdm.tqdm.write(msg)

    try:
        if prim_to_run:
            if args.workers == 1:
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_to_run)} models "
                                f"(sequential, no pool)")
                for mid in prim_to_run:
                    bar.set_postfix_str(mid, refresh=True)
                    tqdm.tqdm.write(f"[wrapper] {mid}: starting")
                    try:
                        result = run_one_model(mid, args.input_dir)
                    except Exception as e:
                        tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                        counts["failed"] += 1
                    else:
                        _log(mid, result)
                    bar.update(1)
                    _check_transient_cuda_and_halt(mid, bar)
            else:
                tqdm.tqdm.write(f"[wrapper] primitive phase: {len(prim_to_run)} models "
                                f"across {args.workers} worker(s)")
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    try:
                        futs = {ex.submit(run_one_model, mid, args.input_dir): mid
                                for mid in prim_to_run}
                        for fut in as_completed(futs):
                            mid = futs[fut]
                            try:
                                result = fut.result()
                            except Exception as e:
                                tqdm.tqdm.write(
                                    f"[wrapper] {mid}: worker raised {e!r}")
                                counts["failed"] += 1
                            else:
                                _log(mid, result)
                            bar.update(1)
                            _check_transient_cuda_and_halt(mid, bar)
                    except (KeyboardInterrupt, SystemExit):
                        tqdm.tqdm.write("[wrapper] halting — cancelling pending jobs")
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise

        if free_to_run:
            tqdm.tqdm.write(f"[wrapper] freeform phase: {len(free_to_run)} models "
                            f"(sequential)")
            for mid in free_to_run:
                bar.set_postfix_str(mid, refresh=True)
                tqdm.tqdm.write(f"[wrapper] {mid}: starting")
                try:
                    result = run_one_model(mid, args.input_dir)
                except Exception as e:
                    tqdm.tqdm.write(f"[wrapper] {mid}: raised {e!r}")
                    counts["failed"] += 1
                else:
                    _log(mid, result)
                bar.update(1)
                _check_transient_cuda_and_halt(mid, bar)
    finally:
        bar.close()
        print(f"[wrapper] done — ok={counts['ok']}  failed={counts['failed']}  "
              f"skipped={counts['skipped']}", flush=True)


if __name__ == "__main__":
    main()

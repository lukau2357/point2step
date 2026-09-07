"""
Minimal per-part wrapper for the mesh pipeline.

Reads a .txt file of ABC model IDs (one per line, comments start with '#'),
globs the parts under `{input_dir}/{model_id}/*.xyzc`, and for each
(model_id, part_idx) spawns a subprocess invoking the pipeline with
`--part part_idx`. Stdout/stderr are captured to per-part log files, and a
`wrapper_status.json` is written at the end regardless of outcome. Existence
of that file is the skip signal on re-runs — status is not checked.

This wrapper is intentionally minimal and should be copy-pasted into the
original Point2CAD repo with only PIPELINE_CMD and the --input_dir default
changed (see the comment at the top of main()).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import tqdm


# Substrings whose presence in stderr.log marks a part as a transient
# CUDA failure rather than a genuine algorithm/OOM failure:
#   "CUDA not available" — torch warning when the subprocess saw no GPU.
#   "Device index must not be negative" — fitting_utils.py:56 calls
#       torch.eye(cols, device=A.get_device()), and get_device() returns
#       -1 on a CPU tensor; this fires when the subprocess fell back to
#       CPU after a prior part's OOM left the CUDA context broken.
# Both indicate "the GPU went away mid-batch" and re-running typically
# succeeds. Honest CUDA-OOMs from create_grid lack these markers and stay
# marked failed so the aggregator records them faithfully.
TRANSIENT_CUDA_MARKERS = (
    "CUDA not available",
    "Device index must not be negative",
)


# ---------------------------------------------------------------------------
# The only lines that differ between the two copies of this wrapper.
#   mesh_pipeline (this repo):  ["python", "mesh_pipeline.py"]     + "output_mesh"
#   point2cad orig:             ["python", "-m", "point2cad.main"] + "output_p2cad_orig"
# OUTPUT_DIR must match the pipeline's own --output_dir default so paths
# constructed here line up with what the subprocess writes on disk.
PIPELINE_CMD = ["python", "-u", "-m", "point2cad.main"]
OUTPUT_DIR = "output_p2cad_orig"
# ---------------------------------------------------------------------------


def read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def cleanup_transient_cuda(output_dir):
    """rmtree any part dir whose stderr.log contains a TRANSIENT_CUDA_MARKERS
    substring, regardless of wrapper_status.json. Even if the part finished
    with status='ok', its CUDA context died at some point during the run
    (the marker proves it), so the recorded duration_s reflects CPU-fallback
    fitting and pollutes the runtime benchmark. The part has to be re-run on
    a healthy GPU. Genuine create_grid CUDA-OOMs lack these markers and stay
    marked failed so the aggregator records them honestly. Parts with no
    wrapper_status.json (e.g. KeyboardInterrupt mid-run) are picked up by
    main()'s to_run filter automatically — no rmtree needed for them."""
    if not os.path.isdir(output_dir):
        return []
    cleaned = []
    pattern = os.path.join(output_dir, "*", "part_*", "stderr.log")
    for stderr_path in sorted(glob.glob(pattern)):
        part_dir = os.path.dirname(stderr_path)
        try:
            with open(stderr_path, errors="replace") as f:
                text = f.read()
            if not any(m in text for m in TRANSIENT_CUDA_MARKERS):
                continue
        except OSError:
            continue
        try:
            shutil.rmtree(part_dir)
        except OSError:
            continue
        cleaned.append(os.path.relpath(part_dir, output_dir))
    return cleaned


def classify_result(returncode, part_dir):
    """Return (status, reason) given the subprocess exit and on-disk state."""
    if returncode != 0:
        sig = -returncode if returncode < 0 else None
        reason = f"exit {returncode}"
        if sig is not None:
            reason += f" (signal {sig})"
        return "fail", reason
    metrics_path = os.path.join(part_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return "fail", "metrics.json missing despite exit 0"
    try:
        with open(metrics_path) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return "fail", f"metrics.json unreadable: {e}"
    if "metrics" not in m:
        return "fail", "metrics.json has no 'metrics' key"
    return "ok", None


def run_one_part(model_id, part_idx, input_dir):
    part_dir = os.path.join(OUTPUT_DIR, model_id, f"part_{part_idx}")
    status_path = os.path.join(part_dir, "wrapper_status.json")
    # Skip is now done in main() so the tqdm bar can pre-seed `initial`.
    # if os.path.isfile(status_path):
    #     tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: already processed, skipping")
    #     return

    os.makedirs(part_dir, exist_ok=True)
    stdout_log = os.path.join(part_dir, "stdout.log")
    stderr_log = os.path.join(part_dir, "stderr.log")

    # Pass --output_dir explicitly so the subprocess uses the same directory
    # we used to build `part_dir` — guards against future drift in the
    # pipeline's argparse default.
    cmd = PIPELINE_CMD + [
        "--model_id", model_id,
        "--input_dir", input_dir,
        "--output_dir", OUTPUT_DIR,
        "--part", str(part_idx),
    ]
    tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: launching {' '.join(cmd)}")

    t0 = time.perf_counter()
    returncode = None
    try:
        # `with` on the log file handles + subprocess.run guarantees the
        # child has exited before we leave this block: run() only returns
        # after the child is reaped, and the file objects flush+close on
        # block exit. No lingering FDs or zombie children.
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.run(cmd, stdout=out, stderr=err)
        returncode = proc.returncode
    finally:
        duration_s = time.perf_counter() - t0

    # If a KeyboardInterrupt fired inside subprocess.run, we never reach this
    # point — the exception propagates out and wrapper_status.json is not
    # written, so the part is retried on the next run.
    status, reason = classify_result(returncode, part_dir)
    payload = {
        "model_id": model_id,
        "part_idx": part_idx,
        "status": status,
        "returncode": returncode,
        "signal": -returncode if returncode is not None and returncode < 0 else None,
        "duration_s": duration_s,
        "reason": reason,
    }
    tmp_path = status_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, status_path)

    tqdm.tqdm.write(f"[wrapper] {model_id} part {part_idx}: {status} "
                    f"(rc={returncode}, {duration_s:.1f}s)"
                    + (f"  reason={reason}" if reason else ""))

    # If this part's stderr contains any transient-CUDA marker, the GPU is
    # broken at the driver level for the rest of this process tree — every
    # subsequent part will at best run slowly on CPU and at worst crash on
    # the same get_device()==-1 path. Bail out with a unique exit code so
    # the operator can restart the container and re-invoke; on the next
    # run, cleanup_transient_cuda will rmtree this part dir and retry it.
    try:
        with open(stderr_log, errors="replace") as f:
            text = f.read()
        if any(m in text for m in TRANSIENT_CUDA_MARKERS):
            tqdm.tqdm.write(
                f"[wrapper] transient-CUDA marker in {stderr_log} — "
                f"exiting 42; restart the Docker container, then re-invoke"
            )
            sys.exit(42)
    except OSError:
        pass


def _summarize_dry_run(input_dir, ids_file):
    """Walk the output tree and report what the next non-dry invocation
    of this wrapper would do: how many parts cleanup_transient_cuda would
    rmtree, how many main() would queue for (re-)running (split into
    never-started vs interrupted-mid-run), and how many would be skipped
    because they already carry a non-transient wrapper_status.json.
    Performs no rmtree and launches no subprocesses."""
    ids = read_ids(ids_file)
    jobs = []
    for model_id in ids:
        model_in_dir = os.path.join(input_dir, model_id)
        part_files = sorted(
            glob.glob(os.path.join(model_in_dir, "*.xyzc")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        for p in part_files:
            jobs.append((model_id, int(os.path.splitext(os.path.basename(p))[0])))

    n_delete = n_fresh = n_resume = n_skip = 0
    for m, p in jobs:
        part_dir = os.path.join(OUTPUT_DIR, m, f"part_{p}")
        stderr_path = os.path.join(part_dir, "stderr.log")
        status_path = os.path.join(part_dir, "wrapper_status.json")
        has_marker = False
        if os.path.isfile(stderr_path):
            try:
                text = open(stderr_path, errors="replace").read()
                has_marker = any(mk in text for mk in TRANSIENT_CUDA_MARKERS)
            except OSError:
                pass
        if has_marker:
            n_delete += 1
        elif os.path.isfile(status_path):
            n_skip += 1
        elif os.path.isdir(part_dir):
            n_resume += 1
        else:
            n_fresh += 1

    n_models = len({j[0] for j in jobs})
    print("[wrapper] DRY RUN — no files modified, no subprocesses launched",
          flush=True)
    print(f"[wrapper] {len(ids)} model IDs from {ids_file}", flush=True)
    print(f"[wrapper] {len(jobs)} parts queued across {n_models} models",
          flush=True)
    print(f"[wrapper]   already complete (skip):                     {n_skip}",
          flush=True)
    print(f"[wrapper]   transient marker (rmtree + redo):            {n_delete}",
          flush=True)
    print(f"[wrapper]   never started (no output dir):               {n_fresh}",
          flush=True)
    print(f"[wrapper]   interrupted mid-run (output dir, no status): {n_resume}",
          flush=True)
    print(f"[wrapper] parts main() will process this run: "
          f"{n_delete + n_fresh + n_resume}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ids_file", required=True,
                    help="Path to a .txt file with one ABC model ID per line")
    ap.add_argument("--input_dir", default="../point2cad_repr/sample_clouds_abc_parts",
                    help="Root of per-part .xyzc inputs "
                         "(expects {input_dir}/{model_id}/*.xyzc)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report what cleanup_transient_cuda + main() would "
                         "do (rmtree/run/skip counts) and exit; performs no "
                         "rmtree and launches no subprocesses.")
    args = ap.parse_args()

    if args.dry_run:
        _summarize_dry_run(args.input_dir, args.ids_file)
        return

    ids = read_ids(args.ids_file)
    print(f"[wrapper] {len(ids)} model IDs from {args.ids_file}", flush=True)

    cleaned = cleanup_transient_cuda(OUTPUT_DIR)
    if cleaned:
        print(f"[wrapper] removed {len(cleaned)} part dir(s) with a "
              f"transient-CUDA marker in stderr — will retry", flush=True)
        for c in cleaned:
            print(f"  {c}", flush=True)

    # Pre-scan: build the full (model_id, part_idx) job list up front so the
    # tqdm bar has a known total and reports ETA. Missing models are reported
    # immediately, not at their would-be turn in the loop.
    jobs = []
    for model_id in ids:
        model_in_dir = os.path.join(args.input_dir, model_id)
        part_files = sorted(
            glob.glob(os.path.join(model_in_dir, "*.xyzc")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        )
        if not part_files:
            print(f"[wrapper] {model_id}: no .xyzc files under {model_in_dir}, skipping",
                  flush=True)
            continue
        for p in part_files:
            jobs.append((model_id, int(os.path.splitext(os.path.basename(p))[0])))

    n_models = len({j[0] for j in jobs})
    print(f"[wrapper] {len(jobs)} parts queued across {n_models} models", flush=True)

    to_run = [(m, p) for m, p in jobs
              if not os.path.isfile(os.path.join(
                  OUTPUT_DIR, m, f"part_{p}", "wrapper_status.json"))]
    n_skipped = len(jobs) - len(to_run)
    if n_skipped:
        print(f"[wrapper] {n_skipped} parts already processed — will skip",
              flush=True)

    bar = tqdm.tqdm(to_run, total=len(jobs), initial=n_skipped,
                    desc="parts", unit="part", dynamic_ncols=True)
    for model_id, part_idx in bar:
        bar.set_postfix_str(f"{model_id} part {part_idx}", refresh=True)
        run_one_part(model_id, part_idx, args.input_dir)
    bar.close()

    print("[wrapper] done", flush=True)


if __name__ == "__main__":
    main()

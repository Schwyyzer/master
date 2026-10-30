# -*- coding: utf-8 -*-
"""
Compare ART search runs (different TRUST_REGION_EPS/DELTA_LM/etc. settings)
from their results.csv files: success rate, and -- for GENUINELY successful
attempts only -- average/median/std barrier energy, iteration count, step
size (from each attempt's trajectory_file), elapsed time, and mep_proximity,
plus a breakdown of exactly how the rest of the attempts ended.

Why "genuinely successful" means converged & is_saddle, not the raw
`success` column: `success=True` in results.csv only means an attempt
wasn't discarded by the degenerate-mode/second-eigenvalue guard -- it does
NOT mean the search actually converged to a saddle (see
art_core_fortran.jl's `success` docstring). A real example from this
project: seed 7 had success=True, converged=False, is_saddle=False,
force_magnitude=5.05 -- a plain timeout (ran out of MAX_ITERATIONS with a
positive, no-longer-unstable eigenvalue) that still counted as "success".
Filtering barrier/iteration/step-size stats on the raw `success` column
would let exactly that kind of false positive drag them around; `converged
& is_saddle` is the real "found a saddle" criterion. The raw success rate
is still reported on its own, since that IS the quantity
TARGET_SUCCESSES/TARGET_ATTEMPTS actually optimizes for.

Every run's metrics are also appended as one row to COMPARISON_LOG_PATH, a
plain CSV (opens directly in Excel -- File > Open, no extra dependency
needed) -- a running log across however many times you invoke this,
comparing however many different settings, instead of only ever seeing one
run's numbers in the console at a time. A results.csv already logged (by
absolute path) is skipped on a later run, not re-appended as a duplicate
row.

Usage:
    Edit RUNS below and just hit Run (Spyder/any IDE -- no command-line
    args needed), or override via command-line args from a terminal:
        python3 compare_runs.py "label1:path/to/results1.csv" "label2:path/to/results2.csv" ...
    A single CSV can skip the label:
        python3 compare_runs.py path/to/results.csv
"""
import csv
import os
import statistics
import sys
from datetime import datetime

# ============================================================
# Settings -- edit these, or override via command-line args
# ============================================================
RUNS = [
    # ("label shown in the output", "path/to/results.csv"),
    # ("eps_0.1", "/path/to/run1/results.csv"),
    # ("eps_nothing", "/path/to/run2/results.csv"),
]

COMPARISON_LOG_PATH = "run_comparisons.csv"   # set to None to disable logging entirely

# Fixed column set for COMPARISON_LOG_PATH, so every appended row has the
# same columns regardless of what a particular run had data for (e.g. an
# older results.csv missing elapsed_time/trajectory_file, or a run with
# zero genuinely successful attempts) -- summarize() always returns a dict
# with exactly these keys, blank ("") for anything it couldn't compute.
LOG_FIELDNAMES = [
    "label", "path", "timestamp", "n_total",
    "n_success_raw", "success_rate_pct",
    "n_genuine", "genuine_rate_pct", "genuine_per_1000",
    "n_degenerate", "n_second_eig", "n_exception",
    "barrier_mean", "barrier_median", "barrier_std",
    "iterations_mean", "iterations_median", "iterations_std",
    "mep_proximity_mean",
    "elapsed_time_mean", "total_cpu_seconds",
    "step_size_mean", "step_size_n_runs",
]


def parse_bool(s):
    return s.strip().lower() == "true"


def parse_float(s, default=float("nan")):
    if s is None or s == "":
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_results_csv(path):
    """Reads every column run_parallel_fortran.jl's results.csv has had
    across this project's history. The original columns (seed, success,
    iterations, converged, is_saddle, force_magnitude, mep_proximity,
    initial_energy, final_energy, degenerate_mode_aborted,
    second_eigenvalue_aborted) are read directly -- they've always been
    there. elapsed_time/trajectory_file are newer (added for comparing
    TRUST_REGION_EPS settings); read with a fallback so an older CSV
    doesn't crash this, just loses those specific stats (see summarize()'s
    warning when that happens)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_elapsed_time = "elapsed_time" in (reader.fieldnames or [])
        has_trajectory = "trajectory_file" in (reader.fieldnames or [])
        for row in reader:
            if not row.get("seed"):
                continue
            rows.append(dict(
                seed=int(row["seed"]),
                success=parse_bool(row["success"]),
                iterations=int(row["iterations"]),
                converged=parse_bool(row["converged"]),
                is_saddle=parse_bool(row["is_saddle"]),
                force_magnitude=parse_float(row["force_magnitude"]),
                mep_proximity=parse_float(row["mep_proximity"]),
                initial_energy=parse_float(row["initial_energy"]),
                final_energy=parse_float(row["final_energy"]),
                degenerate_mode_aborted=parse_bool(row["degenerate_mode_aborted"]),
                second_eigenvalue_aborted=parse_bool(row["second_eigenvalue_aborted"]),
                elapsed_time=parse_float(row.get("elapsed_time")) if has_elapsed_time else None,
                trajectory_file=(row.get("trajectory_file") or None) if has_trajectory else None,
            ))
    return rows, has_elapsed_time, has_trajectory


def load_step_sizes(trajectory_path):
    values = []
    with open(trajectory_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["step_norm"]))
    return values


def compute_stat(values):
    """Returns (mean, median, std, n); all-NaN (n=0) if `values` is empty
    after dropping NaN entries."""
    values = [v for v in values if v == v]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return mean, median, std, n


def print_stat(name, mean, median, std, n, unit=""):
    if n == 0:
        print(f"    {name}: no data")
    else:
        print(f"    {name}: mean={mean:.5g}{unit}  median={median:.5g}{unit}  "
              f"std={std:.5g}{unit}  (n={n})")


def summarize(label, path):
    """Prints the full report for one results.csv and returns a dict with
    exactly LOG_FIELDNAMES' keys (blank "" for anything not computed),
    ready to append to COMPARISON_LOG_PATH."""
    metrics = {k: "" for k in LOG_FIELDNAMES}
    metrics["label"] = label
    metrics["path"] = path
    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")

    rows, has_elapsed_time, has_trajectory = parse_results_csv(path)
    n_total = len(rows)
    metrics["n_total"] = n_total
    print(f"\n=== {label}  ({path}, n={n_total} attempts) ===")
    if n_total == 0:
        print("  no rows parsed, skipping")
        return metrics

    n_success_raw = sum(r["success"] for r in rows)
    genuine = [r for r in rows if r["converged"] and r["is_saddle"]]
    n_genuine = len(genuine)
    n_degenerate = sum(r["degenerate_mode_aborted"] for r in rows)
    n_second_eig = sum(r["second_eigenvalue_aborted"] for r in rows)
    n_exception = n_total - n_success_raw - n_degenerate - n_second_eig

    metrics.update(
        n_success_raw=n_success_raw, success_rate_pct=100 * n_success_raw / n_total,
        n_genuine=n_genuine, genuine_rate_pct=100 * n_genuine / n_total,
        genuine_per_1000=1000 * n_genuine / n_total,
        n_degenerate=n_degenerate, n_second_eig=n_second_eig, n_exception=n_exception,
    )

    print(f"  success rate (raw 'success' column):          {n_success_raw}/{n_total}  "
          f"({100*n_success_raw/n_total:.1f}%)")
    print(f"  genuinely successful (converged & is_saddle):  {n_genuine}/{n_total}  "
          f"({100*n_genuine/n_total:.1f}%)  -- {1000*n_genuine/n_total:.1f} per 1000 attempts")

    print(f"\n  how every attempt ended (mutually exclusive, sums to {n_total}):")
    print(f"    genuinely successful (converged & is_saddle):     {n_genuine:5d}  ({100*n_genuine/n_total:.1f}%)")
    print(f"    success=True but NOT genuine (timeout/non-saddle): {n_success_raw-n_genuine:5d}  "
          f"({100*(n_success_raw-n_genuine)/n_total:.1f}%)")
    print(f"    degenerate-mode aborted:                           {n_degenerate:5d}  ({100*n_degenerate/n_total:.1f}%)")
    print(f"    second-eigenvalue aborted:                         {n_second_eig:5d}  ({100*n_second_eig/n_total:.1f}%)")
    if n_exception > 0:
        print(f"    neither success nor aborted (crashed mid-attempt): {n_exception:5d}  "
              f"({100*n_exception/n_total:.1f}%)")

    if n_genuine == 0:
        print("\n  no genuinely successful attempts -- nothing further to report")
        return metrics

    print(f"\n  --- genuinely successful attempts only (n={n_genuine}) ---")

    barriers = [r["final_energy"] - r["initial_energy"] for r in genuine]
    b_mean, b_median, b_std, _ = compute_stat(barriers)
    print_stat("barrier energy (final - initial)", b_mean, b_median, b_std, len(barriers))
    metrics.update(barrier_mean=b_mean, barrier_median=b_median, barrier_std=b_std)

    it_mean, it_median, it_std, it_n = compute_stat([r["iterations"] for r in genuine])
    print_stat("iterations", it_mean, it_median, it_std, it_n)
    metrics.update(iterations_mean=it_mean, iterations_median=it_median, iterations_std=it_std)

    mep_mean, mep_median, mep_std, mep_n = compute_stat([r["mep_proximity"] for r in genuine])
    print_stat("mep_proximity", mep_mean, mep_median, mep_std, mep_n)
    metrics.update(mep_proximity_mean=mep_mean)

    if has_elapsed_time:
        et_mean, et_median, et_std, et_n = compute_stat([r["elapsed_time"] for r in genuine])
        print_stat("elapsed_time", et_mean, et_median, et_std, et_n, " s")
        total_cpu = sum(r["elapsed_time"] for r in rows if r["elapsed_time"] == r["elapsed_time"])
        print(f"    total CPU-seconds across all {n_total} attempts (sum of elapsed_time -- NOT "
              f"real wall-clock, since attempts run in parallel across workers): {total_cpu:.1f}s")
        metrics.update(elapsed_time_mean=et_mean, total_cpu_seconds=total_cpu)
    else:
        print("    elapsed_time: column not present in this CSV (predates that instrumentation)")

    if has_trajectory:
        step_size_means = []
        for r in genuine:
            if r["trajectory_file"] and os.path.exists(r["trajectory_file"]):
                steps = load_step_sizes(r["trajectory_file"])
                if steps:
                    step_size_means.append(statistics.mean(steps))
        if step_size_means:
            ss_mean, ss_median, ss_std, ss_n = compute_stat(step_size_means)
            print_stat(f"avg step size per run (n={ss_n}/{n_genuine} had trajectory data)",
                       ss_mean, ss_median, ss_std, ss_n)
            metrics.update(step_size_mean=ss_mean, step_size_n_runs=ss_n)
        else:
            print("    avg step size: trajectory_file column present, but no trajectory files could be "
                  "read (MINIMAL_OUTPUT run? files moved?)")
    else:
        print("    avg step size: trajectory_file column not present in this CSV (predates that instrumentation)")

    return metrics


def append_to_log(log_path, metrics):
    """Appends one row to log_path (plain CSV, opens directly in Excel).
    Skips (returns False, no write) if this exact results.csv (by absolute
    path) already has a row in the log -- running compare_runs.py again on
    the same file won't create a duplicate row."""
    abs_path = os.path.abspath(metrics["path"])
    row = dict(metrics, path=abs_path)

    file_exists = os.path.exists(log_path)
    if file_exists:
        with open(log_path, newline="") as f:
            existing_paths = {r.get("path") for r in csv.DictReader(f)}
        if abs_path in existing_paths:
            print(f"  ({os.path.basename(abs_path)} already in {log_path} -- skipping append)")
            return False

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  appended to {log_path}")
    return True


def main():
    args = sys.argv[1:]

    if args:
        runs = []
        for spec in args:
            if ":" in spec:
                label, path = spec.split(":", 1)
            elif len(args) == 1:
                label, path = spec, spec
            else:
                sys.exit(f"expected label:path, got: {spec}")
            runs.append((label, path))
    else:
        runs = RUNS

    if not runs:
        sys.exit("no runs given -- edit RUNS at the top of this file, or pass "
                  '"label:path/to/results.csv" arguments on the command line')

    for label, path in runs:
        metrics = summarize(label, path)
        if COMPARISON_LOG_PATH is not None:
            append_to_log(COMPARISON_LOG_PATH, metrics)


if __name__ == "__main__":
    main()

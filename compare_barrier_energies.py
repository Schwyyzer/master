# -*- coding: utf-8 -*-
"""
Compare barrier-energy distributions across multiple results.csv files
(e.g. different DELTA_LM/TRUST_REGION_EPS/trust-region settings) on one
plot.

"Barrier" = final_energy - initial_energy for a GENUINELY successful
attempt -- converged AND is_saddle, not just the raw `success` column.
success=True in results.csv only means an attempt wasn't discarded by the
degenerate-mode/second-eigenvalue guard; it does NOT mean the search
actually reached a converged, first-order saddle (see art_core_fortran.jl's
`success` docstring). Concrete example, from compare_runs.py's own
docstring: seed 7 had success=True, converged=False, is_saddle=False,
force_magnitude=5.05 -- a plain timeout. Pooling that kind of row into a
"barrier" would mix in an energy difference that isn't a barrier at all.
Set REQUIRE_GENUINE=False below (or pass --allow-non-genuine) to fall back
to raw `success`, e.g. for an old results.csv that predates the
converged/is_saddle columns.

Parses results.csv itself rather than importing validate_saddles.py's
parse_results_csv -- this script only ever needs a handful of scalar
columns, and a local copy means its behavior can't shift out from under it
if that function's return shape changes for some other script's needs.

Usage:
    Edit RUNS below to point at your CSVs, or pass LABEL:PATH pairs on
    the command line (":" rather than "=", since labels like "delta_lm=0.1"
    naturally contain their own "="):
        python3 compare_barrier_energies.py "delta_lm=0.1:run1/results.csv" "delta_lm=0.025:run2/results.csv"
"""
import argparse
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Settings -- edit these, or override via command-line args
# ============================================================
RUNS = [
    # ("label shown in the legend", "path/to/results.csv"),
    # ("delta_lm=0.1", "/path/to/run1/results.csv"),
    # ("delta_lm=0.025", "/path/to/run2/results.csv"),
]

# Require converged & is_saddle rather than just the raw `success` column
# (see module docstring). Overridden by --allow-non-genuine.
REQUIRE_GENUINE = True

COLORS = ["#2a78d6", "#e34948", "#3fa34d", "#c47f17", "#8858c4", "#1f9aa3"]
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def parse_bool(s):
    return s.strip().lower() == "true"


def load_rows(path):
    """Reads seed/success/converged/is_saddle/initial_energy/final_energy
    from a results.csv. converged/is_saddle come back as None (not False)
    when the column is absent, so load_barriers can tell "genuinely not a
    saddle" apart from "this CSV predates the column and we can't tell"."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_converged = "converged" in fieldnames
        has_is_saddle = "is_saddle" in fieldnames
        for row in reader:
            if not row.get("seed"):
                continue
            rows.append(dict(
                seed=int(row["seed"]),
                success=parse_bool(row["success"]),
                converged=parse_bool(row["converged"]) if has_converged else None,
                is_saddle=parse_bool(row["is_saddle"]) if has_is_saddle else None,
                initial_energy=float(row["initial_energy"]),
                final_energy=float(row["final_energy"]),
            ))
    return rows, has_converged, has_is_saddle


def load_barriers(path, require_genuine):
    rows, has_converged, has_is_saddle = load_rows(path)
    if not rows:
        raise SystemExit(f"no rows parsed from {path}")

    if require_genuine and has_converged and has_is_saddle:
        selected = [r for r in rows if r["converged"] and r["is_saddle"]]
        n_raw_success = sum(1 for r in rows if r["success"])
        if len(selected) < n_raw_success:
            print(f"    {path}: {n_raw_success} raw successes, {len(selected)} genuine "
                  f"(converged & is_saddle) -- excluded {n_raw_success - len(selected)} "
                  f"timeout/non-saddle 'successes'")
    else:
        if require_genuine:
            missing = ("converged" if not has_converged else "") + \
                      (" and " if not has_converged and not has_is_saddle else "") + \
                      ("is_saddle" if not has_is_saddle else "")
            print(f"*** WARNING: {path} has no {missing} column -- can't confirm genuine "
                  f"(converged & is_saddle) saddles, falling back to raw `success` for this file.")
        selected = [r for r in rows if r["success"]]

    if not selected:
        raise SystemExit(f"no {'genuine' if require_genuine else 'successful'} rows in {path}")

    final_energies = np.array([r["final_energy"] for r in selected])
    initial_energies = np.array([r["initial_energy"] for r in selected])
    if initial_energies.std() > 1e-6:
        print(f"*** WARNING: {path}'s initial_energy is not constant across rows "
              f"(std={initial_energies.std():.6g}) -- may mix more than one run.")
    return final_energies - initial_energies


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", metavar="LABEL:PATH",
                         help="one or more label:path.csv pairs; overrides RUNS above if given")
    parser.add_argument("--bins", type=int, default=25, help="histogram bin count (default 25)")
    parser.add_argument("--out", type=str, default="barrier_energy_comparison.png",
                         help="output image path")
    parser.add_argument("--allow-non-genuine", action="store_true",
                         help="count any success=True row as a success instead of requiring "
                              "converged & is_saddle (see module docstring)")
    args = parser.parse_args()

    if args.runs:
        runs = []
        for spec in args.runs:
            if ":" not in spec:
                sys.exit(f"expected LABEL:PATH, got: {spec}")
            label, path = spec.split(":", 1)
            runs.append((label, path))
    else:
        runs = RUNS

    if not runs:
        sys.exit("no runs given -- edit RUNS at the top of this file, or pass LABEL:PATH "
                  "arguments on the command line")

    require_genuine = REQUIRE_GENUINE and not args.allow_non_genuine
    data = [(label, load_barriers(path, require_genuine)) for label, path in runs]

    # Shared bin edges across all runs so the histograms are directly
    # comparable (same bin widths/positions), spanning the full range
    # any run touches.
    all_values = np.concatenate([vals for _, vals in data])
    bin_edges = np.linspace(all_values.min(), all_values.max(), args.bins + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    print(f"\n{'label':<20} {'n':>6} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}")
    for (label, values), color in zip(data, COLORS):
        ax.hist(values, bins=bin_edges, color=color, edgecolor=COLOR_INK, linewidth=0.5,
                 alpha=0.55, density=True, zorder=3,
                 label=f"{label}  (n={len(values)}, mean={values.mean():.4f})")
        ax.axvline(values.mean(), color=color, linewidth=1.8, linestyle="--", zorder=4)
        print(f"{label:<20} {len(values):>6} {values.mean():>10.4f} {np.median(values):>10.4f} "
              f"{values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")

    genuine_note = "genuine (converged & first-order)" if require_genuine else "raw-success"
    ax.set_title(f"Barrier height distribution comparison -- {genuine_note} attempts",
                 color=COLOR_INK, fontsize=12, pad=10)
    ax.set_xlabel("barrier = final_energy − initial_energy  (reduced units)", color=COLOR_INK, fontsize=10)
    ax.set_ylabel("density", color=COLOR_INK, fontsize=10)
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_INK, loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, facecolor=COLOR_SURFACE)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

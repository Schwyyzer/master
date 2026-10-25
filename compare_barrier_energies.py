# -*- coding: utf-8 -*-
"""
Compare barrier-energy distributions across multiple results.csv files
(e.g. different DELTA_LM/TRUST_REGION_EPS settings) on one plot.

Reuses validate_saddles.py's parse_results_csv (same lightweight csv+numpy
dependency, no Julia/scipy needed) -- same CSV format plot_saddle_energies.py
reads, just comparing several files instead of one.

"Barrier" = final_energy - initial_energy per successful row (the
activation energy relative to that run's shared starting minimum).

Usage:
    Edit RUNS below to point at your CSVs, or pass LABEL:PATH pairs on
    the command line (":" rather than "=", since labels like "delta_lm=0.1"
    naturally contain their own "="):
        python3 compare_barrier_energies.py "delta_lm=0.1:run1/results.csv" "delta_lm=0.025:run2/results.csv"
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import validate_saddles as vs

# ============================================================
# Settings -- edit these, or override via command-line args
# ============================================================
RUNS = [
    # ("label shown in the legend", "path/to/results.csv"),
    # ("delta_lm=0.1", "/path/to/run1/results.csv"),
    # ("delta_lm=0.025", "/path/to/run2/results.csv"),
]

COLORS = ["#2a78d6", "#e34948", "#3fa34d", "#c47f17", "#8858c4", "#1f9aa3"]
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def load_barriers(path):
    rows = vs.parse_results_csv(path)
    successes = [r for r in rows if r["success"]]
    if not successes:
        raise SystemExit(f"no successful rows in {path}")
    final_energies = np.array([r["final_energy"] for r in successes])
    initial_energies = np.array([r["initial_energy"] for r in successes])
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

    data = [(label, load_barriers(path)) for label, path in runs]

    # Shared bin edges across all runs so the histograms are directly
    # comparable (same bin widths/positions), spanning the full range
    # any run touches.
    all_values = np.concatenate([vals for _, vals in data])
    bin_edges = np.linspace(all_values.min(), all_values.max(), args.bins + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    print(f"{'label':<20} {'n':>6} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}")
    for (label, values), color in zip(data, COLORS):
        ax.hist(values, bins=bin_edges, color=color, edgecolor=COLOR_INK, linewidth=0.5,
                 alpha=0.55, density=True, zorder=3,
                 label=f"{label}  (n={len(values)}, mean={values.mean():.4f})")
        ax.axvline(values.mean(), color=color, linewidth=1.8, linestyle="--", zorder=4)
        print(f"{label:<20} {len(values):>6} {values.mean():>10.4f} {np.median(values):>10.4f} "
              f"{values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}")

    ax.set_title("Barrier height distribution comparison", color=COLOR_INK, fontsize=13, pad=10)
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

# -*- coding: utf-8 -*-
"""
Plot the distribution of saddle-point energies (and barrier heights) from a
results.csv, for inclusion in a write-up.

Reuses validate_saddles.py's config (RESULTS_CSV) and CSV parsing
(parse_results_csv) unchanged -- edit validate_saddles.py's top config
block, not this file, to point at your run.

"Saddle energy" here means final_energy (the row's absolute potential
energy at the saddle), and "barrier" means final_energy - initial_energy
(the activation energy of that transition, relative to the shared starting
minimum every attempt was kicked from) -- the two distributions have
identical shape, just shifted by the constant initial_energy, so this
plots barrier as the primary panel (the physically meaningful quantity:
"how hard is this transition to reach") and final_energy as a secondary
panel for reference.

Usage:
    python3 plot_saddle_energies.py [--bins 20] [--out saddle_energy_distribution.png]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import validate_saddles as vs

COLOR_FILL = "#2a78d6"
COLOR_EDGE = "#0b0b0b"
COLOR_MEAN = "#e34948"
COLOR_MEDIAN = "#898781"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def plot_distribution(ax, values, title, xlabel, bins, unit_note=""):
    ax.set_facecolor(COLOR_SURFACE)
    ax.hist(values, bins=bins, color=COLOR_FILL, edgecolor=COLOR_EDGE,
             linewidth=0.6, alpha=0.9, zorder=3)

    mean = np.mean(values)
    median = np.median(values)
    ax.axvline(mean, color=COLOR_MEAN, linewidth=1.6, linestyle="-", zorder=4,
               label=f"mean = {mean:.4f}")
    ax.axvline(median, color=COLOR_MEDIAN, linewidth=1.6, linestyle="--", zorder=4,
               label=f"median = {median:.4f}")

    ax.set_title(title, color=COLOR_INK, fontsize=12, pad=8)
    ax.set_xlabel(xlabel + (f"  ({unit_note})" if unit_note else ""), color=COLOR_INK, fontsize=10)
    ax.set_ylabel("count", color=COLOR_INK, fontsize=10)
    ax.tick_params(colors=COLOR_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=20, help="histogram bin count (default 20)")
    parser.add_argument("--out", type=str, default="saddle_energy_distribution.png",
                         help="output image path")
    args = parser.parse_args()

    rows = vs.parse_results_csv(vs.RESULTS_CSV)
    successes = [r for r in rows if r["success"]]
    if not successes:
        raise SystemExit(f"no successful rows in {vs.RESULTS_CSV}")

    final_energies = np.array([r["final_energy"] for r in successes])
    initial_energies = np.array([r["initial_energy"] for r in successes])
    barriers = final_energies - initial_energies

    bins = min(args.bins, max(5, len(successes) // 2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=COLOR_SURFACE)
    plot_distribution(axes[0], barriers, "Barrier height distribution",
                       "barrier = final_energy − initial_energy", bins, "reduced units")
    plot_distribution(axes[1], final_energies, "Saddle energy distribution",
                       "final_energy", bins, "reduced units")

    fig.suptitle(f"n = {len(successes)} saddle points (from {vs.RESULTS_CSV})",
                 color=COLOR_MUTED, fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=200, facecolor=COLOR_SURFACE)
    print(f"saved {args.out}")

    print(f"\nn = {len(successes)}")
    print(f"barrier:        mean={barriers.mean():.6f}  median={np.median(barriers):.6f}  "
          f"std={barriers.std():.6f}  min={barriers.min():.6f}  max={barriers.max():.6f}")
    print(f"final_energy:   mean={final_energies.mean():.6f}  median={np.median(final_energies):.6f}  "
          f"std={final_energies.std():.6f}  min={final_energies.min():.6f}  max={final_energies.max():.6f}")
    if initial_energies.std() > 1e-6:
        print(f"*** WARNING: initial_energy is not constant across rows (std={initial_energies.std():.6g}) "
              f"-- this CSV may mix more than one run; barrier values may not be directly comparable.")


if __name__ == "__main__":
    main()

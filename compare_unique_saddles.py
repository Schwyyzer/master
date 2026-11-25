# -*- coding: utf-8 -*-
"""
Compare two results.csv files from the SAME starting system and plot the
energies of saddles each run found that the OTHER run never caught.

"Found" means a genuine saddle -- converged AND is_saddle, not just the raw
`success` column (see compare_barrier_energies.py's docstring for why: a
plain timeout can read success=True). "The other run never caught it" is
judged on the SADDLE POINT ITSELF (each dump's atom positions), not its
push/relax endpoints -- the same way validate_saddles.py's own
"Saddle-point diversity" section judges distinctness within a single run:
max_atom_displacement between two dumps' positions below
MAX_ATOM_DISP_SAME_THRESHOLD means "the same saddle".

Saddles are deduplicated WITHIN each run too, not just across runs. A run
can rediscover the same saddle many times over (this project's own
trust-region notes mention one saddle found 19 times in 300 attempts) --
without deduping first, that one saddle's energy would count once per seed
that found it in the "unique to this run" plot just because the other run
happens to lack it, which would badly overweight it. Concretely: every
genuine saddle from BOTH files goes into one combined pool and gets
clustered together in a single pass (reusing validate_saddles.py's own
cluster_by_max_disp -- the same clustering this project already uses to
define "distinct saddle", not a new definition invented here). A cluster
containing dumps from only one run is "unique to" that run (one energy per
cluster, regardless of how many times that run found it); a cluster with
dumps from both runs is shared and excluded from the unique-energy plot
(but still counted in the printed summary).

Usage:
    Edit LAMMPS_PATH/RUN_A/RUN_B below, or pass them on the command line:
        python3 compare_unique_saddles.py --lammps path/to/config.lammps \\
            "trust_region:run_a/results.csv" "old_settings:run_b/results.csv"
"""
import argparse
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validate_saddles import (
    parse_lammps_data, parse_dump_frame, build_neighbor_pairs, compute_energy,
    max_atom_displacement, cluster_by_max_disp, RC,
)

# ============================================================
# Settings -- edit these, or override via command-line args
# ============================================================
LAMMPS_PATH = "FILL_ME_IN"

RUN_A = ("run_a", "path/to/results_a.csv")
RUN_B = ("run_b", "path/to/results_b.csv")

# Require converged & is_saddle rather than just raw `success` -- see
# module docstring, same standard compare_barrier_energies.py uses.
REQUIRE_GENUINE = True

# Same "is this the same saddle" threshold used throughout this project
# (validate_saddles.py's MAX_ATOM_DISP_SAME_THRESHOLD).
MAX_ATOM_DISP_SAME_THRESHOLD = 0.05

OUTPUT_CSV = "unique_saddles.csv"
OUTPUT_PNG = "unique_saddle_energies.png"
BINS = 25

COLORS = ["#2a78d6", "#e34948"]
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def parse_bool(s):
    return s.strip().lower() == "true"


def load_results_csv(path):
    """Self-contained parser, deliberately not validate_saddles.py's own
    parse_results_csv -- see compare_barrier_energies.py's docstring for
    why: that function's return shape is tuned for validate_saddles.py's
    own main(), and depending on it elsewhere broke once already when it
    changed."""
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
                dump_file=row["dump_file"],
                initial_energy=float(row["initial_energy"]),
                final_energy=float(row["final_energy"]),
            ))
    return rows, has_converged, has_is_saddle


def select_genuine(rows, has_converged, has_is_saddle, label, require_genuine):
    if require_genuine and has_converged and has_is_saddle:
        selected = [r for r in rows if r["converged"] and r["is_saddle"]]
        n_raw_success = sum(1 for r in rows if r["success"])
        print(f"{label}: {n_raw_success} raw successes, {len(selected)} genuine "
              f"(converged & is_saddle)")
    else:
        if require_genuine:
            print(f"*** WARNING: {label} has no converged/is_saddle column -- falling back to "
                  f"raw `success` (can't confirm these are genuine saddles).")
        selected = [r for r in rows if r["success"]]
        print(f"{label}: {len(selected)} success=True row(s)")
    return selected


def load_saddle_positions(rows, data0, label):
    """Reads each row's dump file and attaches its positions_saddle array,
    dropping (with a warning) any that don't parse or whose atom ids/types
    don't match LAMMPS_PATH -- same consistency check validate_saddles.py's
    own main() loop makes for every dump it reads."""
    kept = []
    for r in rows:
        try:
            positions_saddle, move, ids, types = parse_dump_frame(r["dump_file"])
        except (OSError, ValueError, StopIteration) as e:
            print(f"*** {label} seed={r['seed']}: SKIPPED, couldn't read {r['dump_file']}: {e}")
            continue
        if not np.array_equal(ids, data0["id"]) or not np.array_equal(types, data0["cid"]):
            print(f"*** {label} seed={r['seed']}: SKIPPED, {r['dump_file']}'s atom ids/types "
                  f"don't match LAMMPS_PATH")
            continue
        r = dict(r)
        r["positions_saddle"] = positions_saddle
        kept.append(r)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", metavar="LABEL:PATH",
                         help="exactly two label:path.csv pairs; overrides RUN_A/RUN_B above if given")
    parser.add_argument("--lammps", type=str, default=None, help="overrides LAMMPS_PATH above")
    parser.add_argument("--out", type=str, default=OUTPUT_PNG, help="output image path")
    parser.add_argument("--csv-out", type=str, default=OUTPUT_CSV, help="output CSV path")
    parser.add_argument("--bins", type=int, default=BINS, help="histogram bin count")
    args = parser.parse_args()

    lammps_path = args.lammps or LAMMPS_PATH
    if lammps_path == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of this script, or pass --lammps")

    if args.runs:
        if len(args.runs) != 2:
            sys.exit("expected exactly two LABEL:PATH arguments (this script compares a pair)")
        runs = []
        for spec in args.runs:
            if ":" not in spec:
                sys.exit(f"expected LABEL:PATH, got: {spec}")
            label, path = spec.split(":", 1)
            runs.append((label, path))
    else:
        runs = [RUN_A, RUN_B]

    (label_a, path_a), (label_b, path_b) = runs

    data0 = parse_lammps_data(lammps_path)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)
    pairs0 = build_neighbor_pairs(positions0, RC, box)
    initial_energy = compute_energy(positions0, data0["cid"], box, pairs0)

    all_selected = {}
    for label, path in runs:
        rows, has_converged, has_is_saddle = load_results_csv(path)
        if not rows:
            raise SystemExit(f"no rows parsed from {path}")
        if abs(rows[0]["initial_energy"] - initial_energy) > 1e-6:
            print(f"*** WARNING: {label}'s initial_energy ({rows[0]['initial_energy']}) doesn't "
                  f"match LAMMPS_PATH's ({initial_energy}) -- are these really the same starting "
                  f"system? This comparison assumes they are.")
        selected = select_genuine(rows, has_converged, has_is_saddle, label, REQUIRE_GENUINE)
        if not selected:
            raise SystemExit(f"no {'genuine' if REQUIRE_GENUINE else 'successful'} rows in {path}")
        kept = load_saddle_positions(selected, data0, label)
        all_selected[label] = kept
        print(f"{label}: {len(kept)} saddle(s) with readable, matching dump files\n")

    rows_a = all_selected[label_a]
    rows_b = all_selected[label_b]

    combined_positions = [r["positions_saddle"] for r in rows_a] + [r["positions_saddle"] for r in rows_b]
    combined_source = [label_a] * len(rows_a) + [label_b] * len(rows_b)
    combined_rows = rows_a + rows_b

    labels, n_clusters = cluster_by_max_disp(combined_positions, box, MAX_ATOM_DISP_SAME_THRESHOLD)
    print(f"{n_clusters} distinct saddle point(s) total across both runs "
          f"({len(combined_positions)} genuine saddles before dedup, threshold="
          f"{MAX_ATOM_DISP_SAME_THRESHOLD})")

    from collections import defaultdict
    by_cluster = defaultdict(list)
    for idx, cluster_id in enumerate(labels):
        by_cluster[cluster_id].append(idx)

    unique_rows = {label_a: [], label_b: []}
    n_shared = 0
    unique_csv_rows = []
    for cluster_id, idxs in by_cluster.items():
        sources = {combined_source[i] for i in idxs}
        energies_here = [combined_rows[i]["final_energy"] for i in idxs]
        spread = max(energies_here) - min(energies_here)
        if spread > 1e-3:
            print(f"*** NOTE: cluster {cluster_id} ({len(idxs)} dump(s) from {sorted(sources)}, "
                  f"judged the SAME saddle point) has final_energy varying by {spread:.4g} -- "
                  f"larger than expected for 'the same configuration'; worth a look.")
        representative = combined_rows[idxs[0]]
        row_out = dict(seed=representative["seed"], dump_file=representative["dump_file"],
                        final_energy=representative["final_energy"],
                        barrier=representative["final_energy"] - initial_energy)
        if sources == {label_a}:
            unique_rows[label_a].append(representative)
            unique_csv_rows.append(dict(run=label_a, **row_out))
        elif sources == {label_b}:
            unique_rows[label_b].append(representative)
            unique_csv_rows.append(dict(run=label_b, **row_out))
        else:
            n_shared += 1

    n_distinct_a = len({labels[i] for i in range(len(rows_a))})
    n_distinct_b = len({labels[i] for i in range(len(rows_a), len(combined_positions))})

    print(f"\n{label_a}: {n_distinct_a} distinct saddle(s) found, {len(unique_rows[label_a])} "
          f"of them found ONLY by {label_a}")
    print(f"{label_b}: {n_distinct_b} distinct saddle(s) found, {len(unique_rows[label_b])} "
          f"of them found ONLY by {label_b}")
    print(f"shared (both runs found it): {n_shared}")

    if not unique_rows[label_a] and not unique_rows[label_b]:
        raise SystemExit("\nevery distinct saddle was found by both runs -- nothing unique to plot")

    with open(args.csv_out, "w", newline="") as f:
        fieldnames = ["run", "seed", "dump_file", "final_energy", "barrier"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_csv_rows)
    print(f"\nwrote {args.csv_out} ({len(unique_csv_rows)} unique saddle(s))")

    all_barriers = [r["barrier"] for r in unique_csv_rows]
    if len(set(all_barriers)) > 1:
        bin_edges = np.linspace(min(all_barriers), max(all_barriers), args.bins + 1)
    else:
        bin_edges = np.linspace(all_barriers[0] - 0.5, all_barriers[0] + 0.5, args.bins + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    print(f"\n{'label':<20} {'n unique':>10} {'mean':>10} {'median':>10} {'std':>10}")
    for label, color in zip((label_a, label_b), COLORS):
        barriers = np.array([r["final_energy"] - initial_energy for r in unique_rows[label]])
        if len(barriers) == 0:
            print(f"{label:<20} {0:>10}  (nothing unique to this run)")
            continue
        ax.hist(barriers, bins=bin_edges, color=color, edgecolor=COLOR_INK, linewidth=0.5,
                 alpha=0.6, zorder=3, label=f"unique to {label}  (n={len(barriers)})")
        ax.axvline(barriers.mean(), color=color, linewidth=1.8, linestyle="--", zorder=4)
        print(f"{label:<20} {len(barriers):>10} {barriers.mean():>10.4f} "
              f"{np.median(barriers):>10.4f} {barriers.std():>10.4f}")

    ax.set_title(f"Saddle energies unique to each run  ({n_shared} shared saddle(s) excluded)",
                 color=COLOR_INK, fontsize=12, pad=10)
    ax.set_xlabel("barrier = final_energy − initial_energy  (reduced units)", color=COLOR_INK, fontsize=10)
    ax.set_ylabel("count  (NOT density -- the raw count is the point here)", color=COLOR_INK, fontsize=10)
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

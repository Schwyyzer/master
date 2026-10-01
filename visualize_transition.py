# -*- coding: utf-8 -*-
"""
Render a 3-panel figure (initial minimum -> saddle -> new minimum) for one
ART transition, for inclusion in a write-up.

Picks the atom that moved most between the initial and new minimum, takes
its N nearest neighbours (in the initial configuration) to give it local
context, unwraps all three frames' coordinates relative to that atom's
fixed initial position (so periodic wrapping can't make a neighbour jump
across the box between panels), and projects onto the 2D plane of greatest
variance across all three frames (via PCA/SVD) -- so whatever the actual
3D hop direction is, it ends up as in-plane as possible instead of being
viewed edge-on. The SAME projection basis is reused for all three panels,
so a given atom's on-page position is directly comparable panel to panel.

Reuses validate_saddles.py's config (RC, EPSILON_TABLE, SIGMA_TABLE,
LAMMPS_PATH, RESULTS_CSV, PUSH_MAGNITUDES, MAX_ATOM_DISP_SAME_THRESHOLD)
and physics/parsing (parse_lammps_data, parse_dump_frame, find_decisive_push,
com_removed_rmsd, build_neighbor_pairs) unchanged -- edit validate_saddles.py's
top config block, not this file, to point at your run.

Usage:
    python3 visualize_transition.py SEED [--neighbors 12] [--out transition.png]

With no SEED given, picks the successful row with the largest barrier
(final_energy - initial_energy) as a representative example.
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import validate_saddles as vs

# =====================================================================
# Figure styling -- categorical colors + status red from the project's
# validated palette (dataviz skill): slot 1 (blue) / slot 2 (orange) for
# the two atom types (color AND marker shape both carry identity, so the
# figure still reads in grayscale), slot 8 (red) reserved for the single
# highlighted atom so it never gets confused with either type.
# =====================================================================
COLOR_TYPE1 = "#2a78d6"
COLOR_TYPE2 = "#eb6834"
COLOR_HIGHLIGHT = "#e34948"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"


def pick_transition_frames(seed, data0, positions0, box):
    """For the given seed's row in RESULTS_CSV, return
    (positions_saddle, new_min, old_min, barrier) where new_min is whichever
    of the push-relax endpoints is NOT close to the shared starting minimum
    (old_min is the other one, included for a sanity check/diagnostic, not
    for plotting -- the figure uses positions0 itself as "initial")."""
    rows = vs.parse_results_csv(vs.RESULTS_CSV)
    matches = [r for r in rows if r["success"] and r["seed"] == seed]
    if not matches:
        raise SystemExit(f"seed {seed} not found as a successful row in {vs.RESULTS_CSV}")
    row = matches[0]

    positions_saddle, move, ids, types = vs.parse_dump_frame(row["dump_file"])
    if not np.array_equal(ids, data0["id"]) or not np.array_equal(types, data0["cid"]):
        raise SystemExit(f"seed {seed}'s dump atom ids/types don't match LAMMPS_PATH -- wrong file?")

    max_atom_disp = np.linalg.norm(move, axis=1).max()
    if max_atom_disp < 1e-14:
        raise SystemExit(f"seed {seed}'s MOVE vector is ~zero, nothing to push along")

    result = vs.find_decisive_push(
        positions_saddle, move, max_atom_disp,
        vs.PUSH_MAGNITUDES, vs.MAX_ATOM_DISP_SAME_THRESHOLD, data0, box,
    )
    if result is None or not result["decisive"]:
        raise SystemExit(
            f"seed {seed}: push/relax wasn't decisive (see validate_saddles.py's own output "
            f"for this seed) -- pick a different seed, this one's endpoints aren't trustworthy"
        )

    rmsd_plus_start = vs.com_removed_rmsd(result["pos_plus"], positions0, box)
    rmsd_minus_start = vs.com_removed_rmsd(result["pos_minus"], positions0, box)
    if rmsd_plus_start < rmsd_minus_start:
        old_min, new_min = result["pos_plus"], result["pos_minus"]
    else:
        old_min, new_min = result["pos_minus"], result["pos_plus"]

    barrier = row["final_energy"] - row["initial_energy"]
    return positions_saddle, new_min, old_min, barrier, row


def pick_representative_seed(data0, positions0, box):
    rows = [r for r in vs.parse_results_csv(vs.RESULTS_CSV) if r["success"]]
    if not rows:
        raise SystemExit(f"no successful rows in {vs.RESULTS_CSV}")
    rows.sort(key=lambda r: r["final_energy"] - r["initial_energy"], reverse=True)
    return rows[0]["seed"]


def select_local_cluster(initial, new_min, box, n_neighbors):
    """Atom with the largest minimum-image displacement between `initial`
    and `new_min`, plus its `n_neighbors` nearest neighbours in `initial`
    (minimum-image distance). Returns (cluster_indices, center_atom), with
    center_atom first in cluster_indices."""
    dr = initial - new_min
    dr -= box * np.round(dr / box)
    center_atom = int(np.argmax(np.linalg.norm(dr, axis=1)))

    d = initial - initial[center_atom]
    d -= box * np.round(d / box)
    dist = np.linalg.norm(d, axis=1)
    dist[center_atom] = np.inf   # exclude itself from "nearest neighbours"
    nearest = np.argsort(dist)[:n_neighbors]

    cluster_indices = [center_atom] + list(nearest)
    return cluster_indices, center_atom


def unwrap_relative(frame_positions, cluster_indices, ref_point, box):
    """Position of each cluster atom in this frame, minimum-image-unwrapped
    relative to a SINGLE fixed reference point (not that atom's own
    per-frame position) -- so unwrapping is consistent across all three
    frames and a real hop isn't masked by periodic wrapping."""
    rel = frame_positions[cluster_indices] - ref_point
    rel -= box * np.round(rel / box)
    return rel


def compute_projection_plane(rel_coords_list):
    """PCA/SVD over all frames' relative coordinates combined: returns a
    (3,2) orthonormal basis spanning the 2 directions of greatest combined
    positional variance, and the combined mean (for centering before
    projecting). Using the SAME basis for every panel is what makes the
    three panels directly comparable."""
    stacked = np.vstack(rel_coords_list)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T   # (3,2)
    return basis, mean


def project(rel_coords, basis, mean):
    return (rel_coords - mean) @ basis


def render_figure(frames, types, center_atom_pos_in_cluster, barrier, out_path):
    """frames: list of (title, projected_coords (n,2)) for the 3 panels, in
    order initial -> saddle -> new minimum. types: per-cluster-atom LAMMPS
    type (1 or 2), same order/length as each frame's coords, atom 0 is the
    highlighted (most-displaced) atom in every frame."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=COLOR_SURFACE)

    all_xy = np.vstack([coords for _, coords in frames])
    pad = 0.15 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 1e-9)
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    prev_highlight_xy = None
    for ax, (title, coords) in zip(axes, frames):
        ax.set_facecolor(COLOR_SURFACE)

        for k, (x, y) in enumerate(coords):
            is_highlight = (k == center_atom_pos_in_cluster)
            if is_highlight:
                ax.scatter(x, y, s=220, facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_INK,
                           linewidth=1.2, marker="*", zorder=5)
            else:
                color = COLOR_TYPE1 if types[k] == 1 else COLOR_TYPE2
                marker = "o" if types[k] == 1 else "s"
                ax.scatter(x, y, s=130, facecolor=color, edgecolor=COLOR_INK,
                           linewidth=0.6, marker=marker, alpha=0.9, zorder=3)

        # faint ghost line from the highlighted atom's position in the
        # PREVIOUS panel to its position in this one, so the hop itself
        # (not just its two endpoints) is visible
        hx, hy = coords[center_atom_pos_in_cluster]
        if prev_highlight_xy is not None:
            ax.annotate(
                "", xy=(hx, hy), xytext=prev_highlight_xy,
                arrowprops=dict(arrowstyle="-|>", color=COLOR_MUTED, lw=1.3,
                                 linestyle=(0, (2, 2)), shrinkA=8, shrinkB=8),
                zorder=2,
            )
        prev_highlight_xy = (hx, hy)

        ax.set_title(title, color=COLOR_INK, fontsize=12, pad=8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_GRID)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TYPE1,
               markeredgecolor=COLOR_INK, markersize=10, label="type 1"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_TYPE2,
               markeredgecolor=COLOR_INK, markersize=10, label="type 2"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=COLOR_HIGHLIGHT,
               markeredgecolor=COLOR_INK, markersize=14, label="most-displaced atom"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, labelcolor=COLOR_INK)

    fig.suptitle(f"barrier = {barrier:.4f}  (reduced units; axes are an arbitrary 2D projection)",
                 color=COLOR_MUTED, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?", default=None,
                         help="seed of the successful attempt to visualize "
                              "(default: the one with the largest barrier)")
    parser.add_argument("--neighbors", type=int, default=12,
                         help="number of nearest neighbours of the most-displaced "
                              "atom to include for context (default 12)")
    parser.add_argument("--out", type=str, default=None,
                         help="output image path (default: transition_seed<N>.png)")
    args = parser.parse_args()

    if vs.LAMMPS_PATH == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of validate_saddles.py before running this")

    data0 = vs.parse_lammps_data(vs.LAMMPS_PATH)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)

    seed = args.seed if args.seed is not None else pick_representative_seed(data0, positions0, box)
    positions_saddle, new_min, old_min, barrier, row = pick_transition_frames(seed, data0, positions0, box)

    cluster_indices, center_atom = select_local_cluster(positions0, new_min, box, args.neighbors)
    ref_point = positions0[center_atom].copy()
    types = data0["cid"][cluster_indices]

    rel_initial = unwrap_relative(positions0, cluster_indices, ref_point, box)
    rel_saddle = unwrap_relative(positions_saddle, cluster_indices, ref_point, box)
    rel_new_min = unwrap_relative(new_min, cluster_indices, ref_point, box)

    basis, mean = compute_projection_plane([rel_initial, rel_saddle, rel_new_min])

    frames = [
        ("Initial minimum", project(rel_initial, basis, mean)),
        ("Saddle point", project(rel_saddle, basis, mean)),
        ("New minimum", project(rel_new_min, basis, mean)),
    ]

    out_path = args.out or f"transition_seed{seed}.png"
    render_figure(frames, types, 0, barrier, out_path)

    print(f"seed={seed}  center_atom={data0['id'][center_atom]} (0-indexed {center_atom})  "
          f"cluster_size={len(cluster_indices)}  barrier={barrier:.6f}")


if __name__ == "__main__":
    main()

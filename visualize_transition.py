# -*- coding: utf-8 -*-
"""
Render a 3-panel figure (initial minimum -> saddle -> new minimum) for one
ART transition, for inclusion in a write-up.

Rather than flattening a local neighbourhood onto its best-fit plane (which
can visually overlap atoms that are actually far apart along the view axis),
this takes a thin SLICE of the real 3D structure:

  1. Rank every atom by minimum-image displacement between the initial and
     new minimum; the top `--top-movers` are the "significant" atoms
     actually driving the transition.
  2. Fit the plane that best contains those significant atoms' initial
     positions (PCA/SVD: the 2 directions of greatest positional variance
     among them; the 3rd, least-variance direction is the plane's normal)
     -- this is exactly the orientation that packs as many of the big
     movers as possible into a thin slab, not an arbitrary choice.
  3. Take every atom (not just the significant ones, for spectator context)
     within `--half-thickness` of that plane AND within `--lateral-radius`
     of its centroid in-plane, measured on the initial configuration, and
     report how many of the significant movers actually landed inside the
     slab. Both bounds matter: on a dense system, half_thickness alone
     (with no in-plane extent limit) can sweep up a large fraction of the
     whole box.
  4. Unwrap all three frames' slab atoms relative to a single fixed
     reference point (the plane's centroid), so periodic wrapping can't
     make a neighbour appear to jump between panels, and project onto the
     plane's own 2 in-plane directions -- the SAME basis for all three
     panels, so on-page position is directly comparable panel to panel.

Marker SIZE (not color) encodes each shown atom's own initial->new-minimum
displacement magnitude, kept separate from marker color/shape (which carry
atom-type identity) per the project's one-channel-per-job convention; the
single largest mover is additionally starred.

Reuses validate_saddles.py's config (RC, EPSILON_TABLE, SIGMA_TABLE,
LAMMPS_PATH, RESULTS_CSV, PUSH_MAGNITUDES, MAX_ATOM_DISP_SAME_THRESHOLD)
and physics/parsing (parse_lammps_data, parse_dump_frame, find_decisive_push,
com_removed_rmsd, build_neighbor_pairs) unchanged -- edit validate_saddles.py's
top config block, not this file, to point at your run.

Usage:
    python3 visualize_transition.py SEED [--top-movers 15] [--half-thickness 0.6]
                                          [--lateral-radius 3.0] [--out transition.png]

With no SEED given, picks the successful row with the largest barrier
(final_energy - initial_energy) as a representative example.
"""
import argparse

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
# largest mover so it never gets confused with either type.
# =====================================================================
COLOR_TYPE1 = "#2a78d6"
COLOR_TYPE2 = "#eb6834"
COLOR_HIGHLIGHT = "#e34948"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"

MIN_MARKER_SIZE = 50
MAX_MARKER_SIZE = 260


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


def select_slice(initial, new_min, box, n_top_movers, half_thickness, lateral_radius):
    """Pick a thin slice of the structure oriented to contain as many of
    the biggest movers as geometrically possible. Returns a dict with:
      disp               -- per-atom initial->new_min displacement (natoms,)
      significant        -- indices of the top `n_top_movers` by displacement
      in_slab            -- indices of atoms within half_thickness of the
                             fitted plane AND within lateral_radius of its
                             centroid in-plane (significant movers +
                             spectators) -- the lateral bound matters a lot
                             on a dense system, where "within half_thickness"
                             alone can sweep up a large fraction of the
                             whole box since it has no in-plane extent limit
      n_significant_in_slab -- how many of `significant` landed in `in_slab`
      star_atom           -- the single largest mover's index
      basis (3,2)          -- the plane's in-plane directions
      centroid (3,)         -- reference point the plane passes through
    """
    dr = initial - new_min
    dr -= box * np.round(dr / box)
    disp = np.linalg.norm(dr, axis=1)

    significant = np.argsort(disp)[::-1][:n_top_movers]
    star_atom = int(np.argmax(disp))

    sig_pos = initial[significant]
    centroid = sig_pos.mean(axis=0)
    centered = sig_pos - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T       # in-plane directions, (3,2)
    normal = vt[2]         # out-of-plane normal, (3,)

    rel_all = initial - centroid
    rel_all -= box * np.round(rel_all / box)
    perp_dist = np.abs(rel_all @ normal)
    lateral_dist = np.linalg.norm(rel_all @ basis, axis=1)

    in_slab = np.where((perp_dist <= half_thickness) & (lateral_dist <= lateral_radius))[0]
    n_significant_in_slab = len(np.intersect1d(significant, in_slab))

    # The plane is a least-squares fit through ALL significant movers, so
    # the single largest one (the headline atom) can still end up outside
    # half_thickness if it's an outlier relative to the rest of the pack --
    # force it in regardless, since it's the one atom this figure must show.
    star_forced_in = star_atom not in in_slab
    if star_forced_in:
        in_slab = np.append(in_slab, star_atom)

    return dict(
        disp=disp, significant=significant, in_slab=in_slab,
        n_significant_in_slab=n_significant_in_slab, star_atom=star_atom,
        basis=basis, centroid=centroid, star_forced_in=star_forced_in,
        star_perp_dist=perp_dist[star_atom],
    )


def unwrap_relative(frame_positions, indices, ref_point, box):
    """Position of each selected atom in this frame, minimum-image-unwrapped
    relative to a SINGLE fixed reference point (not a per-frame point) --
    so unwrapping is consistent across all three frames."""
    rel = frame_positions[indices] - ref_point
    rel -= box * np.round(rel / box)
    return rel


def project(rel_coords, basis):
    return rel_coords @ basis


def render_figure(frames, types, disp, star_pos_in_slab, barrier, slice_info, out_path):
    """frames: list of (title, projected_coords (n,2)) for the 3 panels, in
    order initial -> saddle -> new minimum, same atom order throughout.
    types/disp: per-slab-atom LAMMPS type and initial->new_min displacement
    magnitude, same order/length as each frame's coords."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), facecolor=COLOR_SURFACE)

    all_xy = np.vstack([coords for _, coords in frames])
    pad = 0.15 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 1e-9)
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    disp_max = max(disp.max(), 1e-9)
    sizes = MIN_MARKER_SIZE + (MAX_MARKER_SIZE - MIN_MARKER_SIZE) * (disp / disp_max)

    prev_star_xy = None
    for ax, (title, coords) in zip(axes, frames):
        ax.set_facecolor(COLOR_SURFACE)

        for k, (x, y) in enumerate(coords):
            is_star = (k == star_pos_in_slab)
            if is_star:
                ax.scatter(x, y, s=sizes[k] + 120, facecolor=COLOR_HIGHLIGHT, edgecolor=COLOR_INK,
                           linewidth=1.2, marker="*", zorder=5)
            else:
                color = COLOR_TYPE1 if types[k] == 1 else COLOR_TYPE2
                marker = "o" if types[k] == 1 else "s"
                ax.scatter(x, y, s=sizes[k], facecolor=color, edgecolor=COLOR_INK,
                           linewidth=0.6, marker=marker, alpha=0.9, zorder=3)

        # faint ghost arrow from the star atom's position in the PREVIOUS
        # panel to its position in this one, so the hop itself is visible
        sx, sy = coords[star_pos_in_slab]
        if prev_star_xy is not None:
            ax.annotate(
                "", xy=(sx, sy), xytext=prev_star_xy,
                arrowprops=dict(arrowstyle="-|>", color=COLOR_MUTED, lw=1.3,
                                 linestyle=(0, (2, 2)), shrinkA=10, shrinkB=10),
                zorder=2,
            )
        prev_star_xy = (sx, sy)

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
               markeredgecolor=COLOR_INK, markersize=14, label="largest mover"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, labelcolor=COLOR_INK)

    n_sig = slice_info["n_significant_in_slab"]
    n_top = len(slice_info["significant"])
    fig.suptitle(
        f"barrier = {barrier:.4f}   |   {n_sig}/{n_top} top-displacement atoms captured in this slice   |   "
        f"marker size ~ displacement (reduced units; axes are the slice's own in-plane directions)",
        color=COLOR_MUTED, fontsize=9.5, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?", default=None,
                         help="seed of the successful attempt to visualize "
                              "(default: the one with the largest barrier)")
    parser.add_argument("--top-movers", type=int, default=15,
                         help="number of highest-displacement atoms used to fit the slice "
                              "plane's orientation (default 15)")
    parser.add_argument("--half-thickness", type=float, default=0.6,
                         help="slab half-thickness (perpendicular to the slice) in reduced "
                              "units (default 0.6, roughly half a particle diameter)")
    parser.add_argument("--lateral-radius", type=float, default=3.0,
                         help="max in-plane distance from the slice's centroid, in reduced "
                              "units (default 3.0) -- bounds the slab laterally, which matters "
                              "a lot on a dense system where the perpendicular bound alone can "
                              "sweep up a large fraction of the whole box")
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

    slice_info = select_slice(positions0, new_min, box, args.top_movers,
                               args.half_thickness, args.lateral_radius)
    in_slab = slice_info["in_slab"]
    basis, centroid = slice_info["basis"], slice_info["centroid"]

    star_pos_in_slab = int(np.where(in_slab == slice_info["star_atom"])[0][0])
    types = data0["cid"][in_slab]
    disp = slice_info["disp"][in_slab]

    frames = [
        ("Initial minimum", project(unwrap_relative(positions0, in_slab, centroid, box), basis)),
        ("Saddle point", project(unwrap_relative(positions_saddle, in_slab, centroid, box), basis)),
        ("New minimum", project(unwrap_relative(new_min, in_slab, centroid, box), basis)),
    ]

    out_path = args.out or f"transition_seed{seed}.png"
    render_figure(frames, types, disp, star_pos_in_slab, barrier, slice_info, out_path)

    print(f"seed={seed}  star_atom={data0['id'][slice_info['star_atom']]} "
          f"(0-indexed {slice_info['star_atom']})  slab_size={len(in_slab)}  "
          f"{slice_info['n_significant_in_slab']}/{len(slice_info['significant'])} top movers captured  "
          f"barrier={barrier:.6f}")
    if slice_info["star_forced_in"]:
        print(f"  NOTE: the star atom itself sat {slice_info['star_perp_dist']:.3f} out of the fitted "
              f"plane and/or beyond --lateral-radius={args.lateral_radius}, and was force-included "
              f"anyway -- it's an outlier relative to the other top movers. Consider a larger "
              f"--half-thickness/--lateral-radius, fewer --top-movers, or accepting it'll look "
              f"slightly offset from the rest of the slab.")
    if slice_info["n_significant_in_slab"] < len(slice_info["significant"]):
        missed = np.setdiff1d(slice_info["significant"], in_slab)
        print(f"  {len(missed)} top mover(s) fell outside the slab (ids: "
              f"{list(data0['id'][missed])}) -- try a larger --half-thickness/--lateral-radius or "
              f"fewer --top-movers if you want them included")


if __name__ == "__main__":
    main()

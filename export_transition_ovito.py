# -*- coding: utf-8 -*-
"""
Export one ART transition (start -> saddle -> relaxed new minimum) as a
single multi-frame LAMMPS dump file, directly readable by OVITO as a
3-frame trajectory you can step through or animate.

Reuses visualize_transition.py's pick_transition_frames (which in turn
reuses validate_saddles.py's config/physics/parsing) completely unchanged
-- the SAME push/relax/decisive-divergence logic already used to validate
a seed's saddle and pick its 3-panel slice figure, just written out here
as full 3D coordinates for every atom instead of a flattened slice.

LAMMPS_PATH/RC/EPSILON_TABLE/SIGMA_TABLE still come from validate_saddles.py's
own top config block (edit that, not this file, to point at your system) --
only the results CSV and seed are this script's own arguments, since
that's the part you're actually varying call to call.

Usage:
    python3 export_transition_ovito.py SEED [--csv results.csv] [--out transition_seed<N>.lammpstrj]

With no --csv given, uses validate_saddles.py's own RESULTS_CSV.
"""
import argparse

import numpy as np

import validate_saddles as vs
import visualize_transition as vt


def write_dump_frame(f, timestep, positions, ids, types, box):
    natoms = len(positions)
    f.write("ITEM: TIMESTEP\n")
    f.write(f"{timestep}\n")
    f.write("ITEM: NUMBER OF ATOMS\n")
    f.write(f"{natoms}\n")
    f.write("ITEM: BOX BOUNDS pp pp pp\n")
    f.write(f"0 {box[0]}\n0 {box[1]}\n0 {box[2]}\n")
    f.write("ITEM: ATOMS id type x y z\n")
    for i in range(natoms):
        f.write(f"{ids[i]} {types[i]} {positions[i, 0]:.8f} {positions[i, 1]:.8f} {positions[i, 2]:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", type=int, help="seed of the successful attempt to export")
    parser.add_argument("--csv", type=str, default=None,
                         help="results.csv to read (default: validate_saddles.py's own RESULTS_CSV)")
    parser.add_argument("--out", type=str, default=None,
                         help="output .lammpstrj path (default: transition_seed<N>.lammpstrj)")
    args = parser.parse_args()

    if vs.LAMMPS_PATH == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of validate_saddles.py before running this")

    data0 = vs.parse_lammps_data(vs.LAMMPS_PATH)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)

    positions_saddle, new_min, old_min, barrier, row = vt.pick_transition_frames(
        args.seed, data0, positions0, box, results_csv=args.csv)

    out_path = args.out or f"transition_seed{args.seed}.lammpstrj"
    with open(out_path, "w") as f:
        write_dump_frame(f, 0, positions0, data0["id"], data0["cid"], box)
        write_dump_frame(f, 1, positions_saddle, data0["id"], data0["cid"], box)
        write_dump_frame(f, 2, new_min, data0["id"], data0["cid"], box)

    print(f"wrote {out_path}: 3 frames for seed={args.seed}  barrier={barrier:.6f}")
    print("  frame 0 (timestep 0) = initial minimum (start)")
    print("  frame 1 (timestep 1) = saddle point")
    print("  frame 2 (timestep 2) = new minimum (relaxed)")
    print("Open directly in OVITO (File > Load File -- it auto-detects the multi-frame LAMMPS")
    print("dump) and step through frames 0/1/2 with the animation slider. If atoms appear to")
    print("jump across a box edge between frames, that's periodic wrapping, not a real")
    print("discontinuity -- OVITO's 'Unwrap trajectories' modifier fixes the display.")


if __name__ == "__main__":
    main()

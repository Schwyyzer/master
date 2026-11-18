# -*- coding: utf-8 -*-
"""
For every saddle in your repertoire that validate_saddles.py did NOT
confirm adjacent to the starting minimum (neither pos_plus nor pos_minus
matches positions0), take whichever of the two sides is CLOSER to the
start and relax it FURTHER than find_decisive_push's original push+relax
did, to tell apart two different explanations for "not adjacent":

  1. GENUINELY not adjacent: the saddle connects two other basins. More
     relaxation just confirms that -- it force-converges, somewhere that
     still isn't the starting minimum.
  2. SPURIOUSLY not adjacent: the original relax_unconstrained hit its
     max_iter (5000, in find_decisive_push) without actually force-
     converging, so pos_plus/pos_minus wasn't really sitting at its
     minimum yet. relax_unconstrained itself doesn't report which of
     these happened -- it returns positions either way -- which is
     exactly the ambiguity this script resolves: relax_further below is
     the same algorithm, except it tells you whether THIS run actually
     reached tol_force or just ran out of budget again.

Reads saddle_repertoire/ directly, not results.csv -- it's already the
deduplicated set of distinct saddles validate_saddles.py has found (see
that file's module docstring), so this naturally runs once per distinct
saddle rather than once per original CSV row that happened to match it.
Must be run from the same directory as validate_saddles.py, same reason
as check_saddle_order.py.

Run with: python3 recheck_nonadjacent.py
"""
import csv
import numpy as np

from validate_saddles import (
    parse_lammps_data, build_neighbor_pairs, compute_energy, force,
    max_atom_displacement, adjacency_verdict, classify_mechanism,
    load_repertoire, save_repertoire,
    RC, RELAXATION_STEP_MAGNITUDE, ADJACENCY_MAX_DISP_THRESHOLD,
)

# =====================================================================
# Settings -- edit these
# =====================================================================
LAMMPS_PATH = "FILL_ME_IN"   # same risk/reason as validate_saddles.py's own
REPERTOIRE_DIR = "saddle_repertoire"

# 10x find_decisive_push's original budget (max_iter=5000) -- the TARGET
# (FURTHER_TOL_FORCE) is deliberately unchanged from that original run's
# tol_force=1e-3: this is about giving the same convergence criterion
# enough iterations to actually be reached, not loosening what counts as
# converged. Raise further for individual stubborn cases the summary at
# the end flags as still not converged.
FURTHER_MAX_ITER = 50000
FURTHER_TOL_FORCE = 1e-3

OUTPUT_CSV = "nonadjacent_recheck.csv"

# If True, also overwrite the matching repertoire entry (pos_plus/
# pos_minus and everything derived from them: max_disp_plus_minus,
# decisive, mechanism) with the refined result for every saddle that
# turned out spuriously non-adjacent, so future validate_saddles.py runs
# benefit too. Off by default -- review the printed/CSV results first;
# this permanently changes a file other runs read from.
UPDATE_REPERTOIRE = False


def relax_further(positions, data, box, tol_force, max_iter, rebuild_every=50):
    """Same algorithm as validate_saddles.py's relax_unconstrained, plus
    reporting whether THIS call actually force-converged (max_force <=
    tol_force) rather than exhausting max_iter -- that function discards
    this, returning only positions either way. Returns (positions,
    converged, final_max_force, iterations_used)."""
    positions = positions.copy()
    Lambda = RELAXATION_STEP_MAGNITUDE
    Iter = 0
    neighbour_pairs = build_neighbor_pairs(positions, RC, box)
    max_force = np.inf
    while max_force > tol_force and Iter < max_iter:
        if Iter % rebuild_every == 0:
            neighbour_pairs = build_neighbor_pairs(positions, RC, box)
        Force_vector = force(positions, data, neighbour_pairs)
        max_force = np.max(np.linalg.norm(Force_vector, axis=1))
        pos_trial = positions + Lambda * Force_vector
        E_trial = compute_energy(pos_trial, data['cid'], box, neighbour_pairs)
        E_current = compute_energy(positions, data['cid'], box, neighbour_pairs)
        if E_trial < E_current:
            positions = pos_trial % box
            Lambda *= 1.05
        else:
            Lambda *= 0.5
            if Lambda < 1e-10:
                return positions, False, max_force, Iter
        Iter += 1
    return positions, bool(max_force <= tol_force), float(max_force), Iter


def main():
    if LAMMPS_PATH == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of this script before running")

    data0 = parse_lammps_data(LAMMPS_PATH)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)
    pairs0 = build_neighbor_pairs(positions0, RC, box)
    initial_energy = compute_energy(positions0, data0["cid"], box, pairs0)

    repertoire, usable = load_repertoire(REPERTOIRE_DIR, data0["natoms"], initial_energy)
    if not usable:
        raise SystemExit(f"{REPERTOIRE_DIR} is not usable for this system (see the warning above) -- "
                          f"fix LAMMPS_PATH or REPERTOIRE_DIR first.")
    if not repertoire:
        raise SystemExit(f"no entries in {REPERTOIRE_DIR} -- run validate_saddles.py "
                          f"(with USE_REPERTOIRE=True) first.")

    nonadjacent = []
    for e in repertoire:
        if e["failed"]:
            continue
        max_disp_plus_start = max_atom_displacement(e["pos_plus"], positions0, box)
        max_disp_minus_start = max_atom_displacement(e["pos_minus"], positions0, box)
        adjacent, _side, _note = adjacency_verdict(max_disp_plus_start, max_disp_minus_start,
                                                     ADJACENCY_MAX_DISP_THRESHOLD)
        if not adjacent:
            nonadjacent.append((e, max_disp_plus_start, max_disp_minus_start))

    print(f"{len(nonadjacent)} of {len(repertoire)} distinct saddle(s) in the repertoire are not "
          f"confirmed adjacent to the starting minimum.")
    if not nonadjacent:
        print("Nothing to recheck.")
        return
    print(f"Relaxing the closer side further on each (max {FURTHER_MAX_ITER} iterations, "
          f"tol_force={FURTHER_TOL_FORCE})...\n")

    rows = []
    n_now_adjacent = 0
    n_still_not_converged = 0

    for e, max_disp_plus_start, max_disp_minus_start in nonadjacent:
        side = "plus" if max_disp_plus_start < max_disp_minus_start else "minus"
        start_pos = e["pos_plus"] if side == "plus" else e["pos_minus"]
        old_dist = min(max_disp_plus_start, max_disp_minus_start)

        new_pos, converged, final_force, iters_used = relax_further(
            start_pos, data0, box, FURTHER_TOL_FORCE, FURTHER_MAX_ITER)
        new_dist = max_atom_displacement(new_pos, positions0, box)
        now_adjacent = new_dist < ADJACENCY_MAX_DISP_THRESHOLD

        n_now_adjacent += now_adjacent
        n_still_not_converged += not converged

        if now_adjacent:
            verdict = ("NOW ADJACENT -- further relaxation reached the starting minimum; the "
                       "original push+relax simply hadn't finished descending")
        elif converged:
            verdict = "STILL NOT ADJACENT, but force-converged -- confirmed a genuinely different minimum"
        else:
            verdict = (f"STILL NOT ADJACENT and STILL hasn't force-converged after {iters_used} more "
                       f"iterations (final max force={final_force:.4g}) -- inconclusive, try raising "
                       f"FURTHER_MAX_ITER for this one specifically")

        print(f"repertoire saddle #{e['id']} (first seed={e['first_seed']}, relaxing {side} side)")
        print(f"  before: max_atom_disp(start)={old_dist:.5f}")
        print(f"  after {iters_used} further iterations ({'converged' if converged else 'DID NOT converge'}, "
              f"final max force={final_force:.4g}): max_atom_disp(start)={new_dist:.5f}")
        print(f"  {verdict}\n")

        rows.append(dict(
            repertoire_id=e["id"], first_seed=e["first_seed"], side_relaxed=side,
            max_disp_start_before=old_dist, max_disp_start_after=new_dist,
            further_relax_converged=converged, further_relax_iterations=iters_used,
            further_relax_final_force=final_force, now_adjacent=now_adjacent,
        ))

        if UPDATE_REPERTOIRE and now_adjacent:
            if side == "plus":
                e["pos_plus"] = new_pos
            else:
                e["pos_minus"] = new_pos
            e["max_disp_plus_minus"] = float(max_atom_displacement(e["pos_plus"], e["pos_minus"], box))
            e["decisive"] = bool(e["max_disp_plus_minus"] > ADJACENCY_MAX_DISP_THRESHOLD)
            e["mechanism"] = classify_mechanism(e["pos_plus"], e["pos_minus"], box) if e["decisive"] else None

    if UPDATE_REPERTOIRE:
        save_repertoire(REPERTOIRE_DIR, repertoire, data0["natoms"], initial_energy)
        print(f"Updated {REPERTOIRE_DIR} with the refined result for each saddle that turned out "
              f"spuriously non-adjacent.\n")

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["repertoire_id", "first_seed", "side_relaxed", "max_disp_start_before",
                      "max_disp_start_after", "further_relax_converged", "further_relax_iterations",
                      "further_relax_final_force", "now_adjacent"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"=== Summary: {n_now_adjacent}/{len(nonadjacent)} were spuriously non-adjacent -- further "
          f"relaxation reached the starting minimum after all ===")
    remaining = len(nonadjacent) - n_now_adjacent
    if remaining > 0:
        print(f"{remaining} remain not adjacent even after further relaxation "
              f"({n_still_not_converged} of those still haven't force-converged -- those specifically "
              f"are the ones worth raising FURTHER_MAX_ITER on, not the whole batch).")
    print(f"Wrote per-saddle results to {OUTPUT_CSV}")
    if not UPDATE_REPERTOIRE and n_now_adjacent > 0:
        print(f"\nUPDATE_REPERTOIRE is False -- the repertoire still has the OLD, less-relaxed pos_plus/"
              f"pos_minus for the {n_now_adjacent} saddle(s) that turned out spuriously non-adjacent. "
              f"Set it True and re-run to persist the refined result.")


if __name__ == "__main__":
    main()

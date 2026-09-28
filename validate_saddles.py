# -*- coding: utf-8 -*-
"""
Validate saddle points found by run_parallel(_no_tracking).jl: for each
successful row in a results.csv, read its dump file (which already
contains both the final saddle POSITIONS and the final MOVE vector --
see write_lammps_frame in lammps_and_document.jl, the "ITEM: MOVE" block
right after "ITEM: ATOMS"), push the system along that unstable-mode
direction in BOTH signs, run a full UNCONSTRAINED relaxation from each,
and check the two results are genuinely different configurations.

This is the standard check for a genuine first-order saddle: it should
connect exactly two basins, one on each side of the unstable direction.

Physics (parse_lammps_data / force / compute_energy / relax) is copied
verbatim from art_driver.py, already cross-checked against the Julia
implementation earlier in this project. Config constants below (RC,
EPSILON_TABLE, SIGMA_TABLE) default to match this repo's CURRENT
config.jl -- NOT necessarily what actually produced your CSV, since
you've been tuning rc locally. This script checks that for you: it
recomputes the shared starting minimum's energy and compares it against
KNOWN_CSV_INITIAL_ENERGY (the constant value in your CSV's initial_energy
column) before doing anything else. A mismatch means fix RC/EPSILON_TABLE/
SIGMA_TABLE/LAMMPS_PATH first -- nothing below is trustworthy until that
check passes.

Push magnitude: rather than one guessed constant, PUSH_MAGNITUDES is a
list tried smallest-first per saddle, scaled so the SINGLE MOST-
DISPLACED atom moves by that amount (not the whole flattened 3N-vector's
norm -- those are very different once a mode is delocalized, and one
real example showed ~93% of a mode's weight on a single atom).

IMPORTANT #1: escalation stops on DECISIVE divergence, not merely on
"relax() didn't throw" -- a push too small to escape the saddle's own
neighborhood relaxes right back to (nearly) the same point on both sides
without ever raising an exception, so treating "no exception" as success
(an earlier version of this script did) silently settles for the
smallest, least conclusive magnitude every time and reports every saddle
as "suspicious" regardless of whether it's real.

IMPORTANT #2: "decisive" is judged by max_atom_displacement (the largest
single per-atom displacement between the two relaxed endpoints), not
whole-system RMSD (an even earlier version of this script used RMSD).
On a 1700+-atom system with a transition localized to ~1 atom, a real
hop dilutes in whole-system-RMSD terms by ~sqrt(natoms) -- easily
burying a completely genuine transition under a plausible-looking
threshold. See find_decisive_push's docstring for the numbers this was
calibrated against and the exact escalation logic.
"""
import csv
import numpy as np
from scipy.spatial import cKDTree

# =====================================================================
# Config -- EDIT THESE to match what actually produced your CSV
# =====================================================================
RESULTS_CSV = "results.csv"
LAMMPS_PATH = "FILL_ME_IN"          # the exact LAMMPS_PATH that run used

RC = 6.0                             # defaults mirror config.jl as it
EPSILON_TABLE = np.array([           # currently stands in this repo --
    [0.0, 0.0, 0.0],                 # NOT guaranteed to match your run;
    [0.0, 1.0, 1.0],                 # the initial_energy check below
    [0.0, 1.0, 1.0],                 # tells you if these are wrong
])
SIGMA_TABLE = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 11 / 12],
    [0.0, 11 / 12, 5 / 6],
])
RELAXATION_STEP_MAGNITUDE = 0.0005

KNOWN_CSV_INITIAL_ENERGY = -13010.936594987796   # from your CSV's initial_energy column

PUSH_MAGNITUDES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]   # tried smallest-first,
                                                    # per saddle, escalating on non-
                                                    # decisive (not merely non-crashing)
                                                    # results -- see module docstring

# Decisiveness is judged by the single LARGEST per-atom displacement between
# the two relaxed endpoints, not whole-system RMSD -- see
# find_decisive_push's docstring for why (RMSD dilutes a localized
# single-atom hop by sqrt(natoms), easily burying a completely real
# transition under a naive threshold on a 1700+-atom system). Back-
# calculating from an actual run: rows that "returned to start" showed
# whole-system RMSD ~0.0001, rows that (per the energies) genuinely
# landed somewhere else showed whole-system RMSD ~0.005 -- consistent
# with one atom hopping by roughly RMSD*sqrt(1728) =~ 0.2. This default
# sits well below that real-hop scale and well above the numerical-noise
# scale; MAX_DISP_PLUS_MINUS is printed for every saddle so you can
# sanity check/adjust it against your own data.
MAX_ATOM_DISP_SAME_THRESHOLD = 0.05


# =====================================================================
# Physics -- copied verbatim from art_driver.py (already validated
# against the Julia implementation), just reading RC/EPSILON_TABLE/
# SIGMA_TABLE/RELAXATION_STEP_MAGNITUDE from this module's globals above
# instead of art_driver.py's own hardcoded ones.
# =====================================================================
def parse_lammps_data(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]

    natoms = None
    for ln in lines:
        if ln.endswith("atoms"):
            natoms = int(ln.split()[0])
            break
    if natoms is None:
        raise RuntimeError("Could not find 'atoms' line")

    xlo = xhi = ylo = yhi = zlo = zhi = None
    for ln in lines:
        if ln.endswith("xlo xhi"):
            xlo, xhi = map(float, ln.split()[:2])
        elif ln.endswith("ylo yhi"):
            ylo, yhi = map(float, ln.split()[:2])
        elif ln.endswith("zlo zhi"):
            zlo, zhi = map(float, ln.split()[:2])
    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise RuntimeError("Box bounds not found")

    lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo

    atoms_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("Atoms"):
            atoms_idx = i
            break
    if atoms_idx is None:
        raise RuntimeError("Could not find 'Atoms' section")

    start = atoms_idx + 2
    atom_lines = lines[start:start + natoms]

    id_arr = np.zeros(natoms, dtype=int)
    cid_arr = np.zeros(natoms, dtype=int)
    x_arr = np.zeros(natoms)
    y_arr = np.zeros(natoms)
    z_arr = np.zeros(natoms)
    for i, ln in enumerate(atom_lines):
        parts = ln.split()
        if len(parts) < 5:
            raise RuntimeError(f"Atom line has too few columns: {ln}")
        id_arr[i] = int(parts[0])
        cid_arr[i] = int(parts[1])
        x_arr[i] = float(parts[2])
        y_arr[i] = float(parts[3])
        z_arr[i] = float(parts[4])

    return {
        "natoms": natoms, "id": id_arr, "cid": cid_arr,
        "x": x_arr, "y": y_arr, "z": z_arr,
        "lx": lx, "ly": ly, "lz": lz,
    }


def force(positions, data, neighbors):
    total_force = np.zeros((len(positions), 3))
    i = neighbors[:, 0]
    j = neighbors[:, 1]
    dr = positions[i] - positions[j]
    box = np.array([data['lx'], data['ly'], data['lz']])
    dr -= box * np.round(dr / box)
    r2 = np.sum(dr**2, axis=1)
    ti = data['cid'][i]
    tj = data['cid'][j]
    eps = EPSILON_TABLE[ti, tj]
    sig = SIGMA_TABLE[ti, tj]
    sig6 = sig**6
    sig12 = sig6**2
    pref = 24 * eps * (2 * sig12 / r2**7 - sig6 / r2**4)
    fij = pref[:, None] * dr
    np.add.at(total_force, i, fij)
    np.add.at(total_force, j, -fij)
    return total_force


def compute_energy(coords, types, box, pairs):
    i = pairs[:, 0]
    j = pairs[:, 1]
    dr = coords[i] - coords[j]
    dr -= box * np.round(dr / box)
    r2 = np.sum(dr**2, axis=1)
    ti = types[i]
    tj = types[j]
    eps = EPSILON_TABLE[ti, tj]
    sig = SIGMA_TABLE[ti, tj]
    inv_r6 = (sig**2 / r2)**3
    energy = 4 * eps * (inv_r6**2 - inv_r6)
    return np.sum(energy)


def build_neighbor_pairs(positions, rc, box):
    tree = cKDTree(positions % box, boxsize=box)
    pairs = np.array(list(tree.query_pairs(rc)), dtype=int)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=int)
    return pairs


def relax_unconstrained(positions, data, box, tol_force=1e-3, max_iter=5000, rebuild_every=50):
    """Full unconstrained relaxation -- same adaptive accept/reject step-size
    scheme as the Julia project's relax(..., nothing, ...) path."""
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
                raise ArithmeticError("Relaxation step magnitude too small.")
        Iter += 1
    return positions


# =====================================================================
# Dump / CSV parsing
# =====================================================================
def parse_dump_frame(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    n_idx = next(i for i, l in enumerate(lines) if l.strip() == "ITEM: NUMBER OF ATOMS")
    N = int(lines[n_idx + 1].strip())

    atoms_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("ITEM: ATOMS"))
    ids = np.zeros(N, dtype=int)
    types = np.zeros(N, dtype=int)
    positions = np.zeros((N, 3))
    for k in range(N):
        parts = lines[atoms_idx + 1 + k].split()
        ids[k] = int(parts[0])
        types[k] = int(parts[1])
        positions[k] = [float(parts[2]), float(parts[3]), float(parts[4])]

    move_idx = next(i for i, l in enumerate(lines) if l.strip() == "ITEM: MOVE")
    N_move = int(lines[move_idx + 1].strip())
    if N_move != N:
        raise ValueError(f"MOVE section atom count ({N_move}) != ATOMS section ({N}) in {path}")
    move = np.zeros((N, 3))
    for k in range(N):
        parts = lines[move_idx + 2 + k].split()
        move[k] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return positions, move, ids, types


def parse_results_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("seed"):
                continue
            rows.append({
                "seed": int(row["seed"]),
                "success": row["success"].strip().lower() == "true",
                "dump_file": row["dump_file"],
                "final_energy": float(row["final_energy"]),
            })
    return rows


def com_removed_rmsd(a, b, box):
    dr = a - b
    dr -= box * np.round(dr / box)
    dr -= dr.mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(dr**2, axis=1)))


def max_atom_displacement(a, b, box):
    """Largest single-atom displacement between two configurations
    (minimum-image). Unlike com_removed_rmsd, this does NOT get diluted
    by system size -- a real single-atom hop in a 1728-atom system moves
    ONE atom by a real amount and leaves the other 1727 essentially
    where they were; the whole-system RMSD divides by all 1728 atoms and
    can bury that signal under the detection threshold (a hop of ~0.2
    dilutes to ~0.2/sqrt(1728) =~ 0.005 in RMSD terms). This metric
    answers "did at least one atom end up somewhere meaningfully
    different" directly, which is the actually relevant question for a
    localized transition."""
    dr = a - b
    dr -= box * np.round(dr / box)
    return np.max(np.linalg.norm(dr, axis=1))


def find_decisive_push(positions_saddle, move, max_atom_disp, magnitudes, max_disp_threshold, data, box):
    """
    Push BOTH sides by the SAME magnitude each round, starting from the
    smallest in `magnitudes` and escalating.

    IMPORTANT #1: a magnitude only counts as "done" if the two relaxed
    endpoints are DECISIVELY different -- not merely because relax()
    didn't throw. A push too small to escape the saddle's own
    neighborhood relaxes right back to (nearly) the same point on both
    sides *without ever raising an exception*, so "no error" is not
    evidence of anything; stopping on that basis (an earlier version of
    this script did) silently settles for the smallest, least
    conclusive magnitude every time and reports "suspicious" universally
    regardless of whether the saddles are real.

    IMPORTANT #2: "decisively different" is judged by max_atom_displacement
    (the single largest per-atom displacement between the two relaxed
    endpoints), NOT the whole-system COM-removed RMSD. On a system with
    ~1700+ atoms and a transition localized to essentially one atom (per
    the earlier ~93%-weight-on-one-atom finding), a real hop of e.g. 0.2
    dilutes to only ~0.2/sqrt(1728) =~ 0.005 in whole-system RMSD terms
    -- comfortably under a naive threshold like 0.02 even though it's a
    completely real, decisive transition. max_atom_displacement answers
    "did at least one atom end up somewhere meaningfully different"
    directly, without that dilution. com_removed_rmsd is still computed
    and returned for reference/diagnostics.

    Escalates to the next larger magnitude whenever the current one
    isn't decisive, and stops escalating (reporting the last magnitude
    that relaxed on both sides, decisive or not) once a magnitude makes
    either side throw -- at that point the run has left the local basin
    structure the earlier decisive-or-not results were sampling, so a
    larger magnitude wouldn't add information.

    Returns a dict with mag, pos_plus, pos_minus, e_plus, e_minus,
    rmsd_plus_minus, max_disp_plus_minus, decisive (bool) -- or None if
    nothing relaxed cleanly even at the smallest magnitude.
    """
    last_result = None
    for mag in magnitudes:
        push = move * (mag / max_atom_disp)
        try:
            pos_plus = relax_unconstrained(positions_saddle + push, data, box)
            pos_minus = relax_unconstrained(positions_saddle - push, data, box)
        except ArithmeticError:
            break

        pairs_plus = build_neighbor_pairs(pos_plus, RC, box)
        pairs_minus = build_neighbor_pairs(pos_minus, RC, box)
        e_plus = compute_energy(pos_plus, data["cid"], box, pairs_plus)
        e_minus = compute_energy(pos_minus, data["cid"], box, pairs_minus)
        rmsd_plus_minus = com_removed_rmsd(pos_plus, pos_minus, box)
        max_disp_plus_minus = max_atom_displacement(pos_plus, pos_minus, box)

        last_result = {
            "mag": mag, "pos_plus": pos_plus, "pos_minus": pos_minus,
            "e_plus": e_plus, "e_minus": e_minus,
            "rmsd_plus_minus": rmsd_plus_minus,
            "max_disp_plus_minus": max_disp_plus_minus,
            "decisive": max_disp_plus_minus > max_disp_threshold,
        }
        if last_result["decisive"]:
            return last_result

    return last_result


# =====================================================================
# Main
# =====================================================================
def main():
    if LAMMPS_PATH == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of this script before running")

    data0 = parse_lammps_data(LAMMPS_PATH)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)
    pairs0 = build_neighbor_pairs(positions0, RC, box)
    initial_energy = compute_energy(positions0, data0["cid"], box, pairs0)

    if abs(initial_energy - KNOWN_CSV_INITIAL_ENERGY) > 1e-6:
        print(f"*** WARNING: computed initial_energy = {initial_energy}")
        print(f"*** does NOT match the CSV's recorded initial_energy = {KNOWN_CSV_INITIAL_ENERGY}")
        print(f"*** (difference = {initial_energy - KNOWN_CSV_INITIAL_ENERGY})")
        print("*** LAMMPS_PATH and/or RC/EPSILON_TABLE/SIGMA_TABLE do NOT match what actually")
        print("*** produced this run -- fix that before trusting anything below.\n")
    else:
        print("initial_energy matches the CSV's recorded value -- LAMMPS_PATH and config")
        print("are at least consistent with the starting point of this run.\n")

    rows = parse_results_csv(RESULTS_CSV)
    successes = [r for r in rows if r["success"]]
    print(f"{len(successes)} successful saddle(s) to validate out of {len(rows)} attempts in the CSV")
    print(f"push magnitudes tried (per side) = {PUSH_MAGNITUDES}, "
          f"same-configuration max-atom-displacement threshold = {MAX_ATOM_DISP_SAME_THRESHOLD}\n")

    n_pass = n_fail = n_error = 0

    for r in successes:
        line = f"seed={r['seed']}  dump={r['dump_file']}"
        try:
            positions_saddle, move, ids, types = parse_dump_frame(r["dump_file"])
        except (OSError, ValueError, StopIteration) as e:
            print(f"{line}  SKIPPED (couldn't read/parse dump file: {e})")
            n_error += 1
            continue

        if not np.array_equal(ids, data0["id"]) or not np.array_equal(types, data0["cid"]):
            print(f"{line}  SKIPPED (this dump's atom ids/types don't match LAMMPS_PATH's -- "
                  f"wrong LAMMPS_PATH for this run?)")
            n_error += 1
            continue

        per_atom_disp = np.linalg.norm(move, axis=1)
        max_atom_disp = per_atom_disp.max()
        if max_atom_disp < 1e-14:
            print(f"{line}  SKIPPED (MOVE vector in dump is ~zero, nothing to push along)")
            n_error += 1
            continue
        line += f"  (raw max per-atom |move| in dump = {max_atom_disp:.6g})"

        try:
            result = find_decisive_push(
                positions_saddle, move, max_atom_disp, PUSH_MAGNITUDES, MAX_ATOM_DISP_SAME_THRESHOLD, data0, box)

            if result is None:
                print(f"{line}\n  FAILED: never relaxed cleanly on both sides at any of {PUSH_MAGNITUDES}\n")
                n_error += 1
                continue

            mag = result["mag"]
            pos_plus, pos_minus = result["pos_plus"], result["pos_minus"]
            e_plus, e_minus = result["e_plus"], result["e_minus"]
            rmsd_plus_minus = result["rmsd_plus_minus"]
            max_disp_plus_minus = result["max_disp_plus_minus"]
            rmsd_plus_start = com_removed_rmsd(pos_plus, positions0, box)
            rmsd_minus_start = com_removed_rmsd(pos_minus, positions0, box)

            is_different = result["decisive"]
            exhausted = (not is_different) and (mag == PUSH_MAGNITUDES[-1])
            verdict = ("PASS -- two distinct minima" if is_different else
                       "SUSPICIOUS -- both sides relaxed to (nearly) the same configuration" +
                       (" (exhausted all push magnitudes)" if exhausted else
                        f" (stopped escalating: mag={mag} was the last to relax on both sides)"))
            n_pass += is_different
            n_fail += not is_different

            print(f"{line}\n"
                  f"  push_used=+-{mag:.3g}{'  (largest tried)' if mag == PUSH_MAGNITUDES[-1] else ''}  "
                  f"E_plus={e_plus:.6f}  E_minus={e_minus:.6f}  "
                  f"(saddle E={r['final_energy']:.6f}, barrier={r['final_energy'] - initial_energy:.6f})\n"
                  f"  max_atom_disp(plus,minus)={max_disp_plus_minus:.5f}  "
                  f"[whole-system RMSD(plus,minus)={rmsd_plus_minus:.5f}]\n"
                  f"  RMSD(plus,start)={rmsd_plus_start:.5f}  "
                  f"RMSD(minus,start)={rmsd_minus_start:.5f}\n"
                  f"  {verdict}\n")
        except Exception as e:
            print(f"{line}\n  ERROR: {e}\n")
            n_error += 1

    print(f"=== Summary: {n_pass} confirmed / {n_fail} suspicious / {n_error} skipped, "
          f"out of {len(successes)} successes ===")
    if n_fail > 0:
        print("Suspicious ones already escalated through all of PUSH_MAGNITUDES without ever")
        print("seeing decisive divergence (see 'exhausted all push magnitudes' vs. 'stopped")
        print("escalating' in each line above -- the latter means a larger push started")
        print("blowing up before ever deciding, worth extending PUSH_MAGNITUDES further with")
        print("smaller steps between values; the former genuinely tried the whole range).")
    if n_error > 0:
        print(f"{n_error} skipped/failed -- if these are all 'never relaxed cleanly at any of")
        print("PUSH_MAGNITUDES', re-check the initial_energy match printed above first: a")
        print("config/LAMMPS_PATH mismatch would produce exactly this kind of uniform failure.")


if __name__ == "__main__":
    main()

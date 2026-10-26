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
the initial_energy column read directly from your CSV's own first row
(every results.csv this repo's run_parallel*.jl scripts produce, old or
new, carries this column, so there's no constant to hand-update when you
point this script at a different run) before doing anything else. A
mismatch means fix RC/EPSILON_TABLE/SIGMA_TABLE/LAMMPS_PATH first --
nothing below is trustworthy until that check passes.

Works unchanged against results.csv from EITHER run_parallel_no_tracking.jl
(phase1/phase2) or run_parallel_fortran.jl / art_core_fortran.jl (the
unified Newton+FIRE algorithm matching art_woconewer.f90): both write dump
files in the same format (just a different vector in the "ITEM: MOVE"
section -- the final phase-2 move vs. the final tracked Lanczos
eigenvector -- which doesn't matter here since the push is always
rescaled to a target per-atom magnitude regardless of the vector's own
original scale), and both CSVs carry the seed/success/dump_file/
final_energy/initial_energy columns this script actually reads.

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
from scipy.optimize import linear_sum_assignment

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

# ---------------------------------------------------------------------
# Mechanism classification (free volume hop / pair swap / chain / ...)
# ---------------------------------------------------------------------
# Classifies each CONFIRMED saddle's transition mechanism from the same
# pos_plus/pos_minus pair find_decisive_push already computes above --
# no extra relaxations, just a shape analysis of the displacement field
# between the two basins a saddle actually connects (not the local
# unstable eigenvector/MOVE vector at the saddle itself, which only
# captures the onset direction and can miss atoms that only move during
# the subsequent relaxation into each basin).
MECHANISM_CSV = "mechanism_classification.csv"
ACTIVE_FRACTION = 0.2      # an atom counts as "active" if its pos_plus/pos_minus
                            # displacement exceeds this fraction of the single
                            # largest displacement in that saddle's field


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
    # Shift by V(rc) so energy vanishes continuously at the cutoff --
    # matches Version1.1.jl's compute_energy (and the validated
    # art_driver.py port). Without this, every absolute energy computed
    # here was systematically offset from results.csv's own values (by
    # an amount proportional to how many pairs are within cutoff), which
    # is exactly what made the "computed initial_energy doesn't match
    # the CSV" warning fire every time regardless of whether
    # RC/EPSILON_TABLE/SIGMA_TABLE/LAMMPS_PATH actually matched. PASS/FAIL
    # verdicts and barrier heights were unaffected (the same shift-less
    # formula was applied consistently everywhere in this file) -- only
    # that one absolute-value sanity check was ever wrong.
    inv_rc6 = (sig**2 / (RC * RC))**3
    shift = 4 * eps * (inv_rc6**2 - inv_rc6)
    # Mask to r<=RC directly rather than trusting `pairs` alone: it's
    # normally already built at exactly RC via build_neighbor_pairs, but
    # relax_unconstrained only rebuilds it every `rebuild_every` steps,
    # so it can go briefly stale as positions drift between rebuilds --
    # same defensive mask used throughout the rest of this project's
    # force()/compute_energy.
    mask = (r2 <= RC * RC) & (r2 > 0)
    energy = np.where(mask, energy - shift, 0.0)
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
                "initial_energy": float(row["initial_energy"]),
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


def cluster_by_max_disp(configs, box, threshold):
    """Greedy clustering of a list of configurations (each an Nx3 array,
    same atom count/order): a config joins the first existing cluster
    whose representative (its first member) is within `threshold`
    max_atom_displacement, else it starts a new cluster. Returns
    (labels, n_clusters) where labels[i] is the cluster index of
    configs[i]. Used to test whether the SADDLE POINTS THEMSELVES (not
    their push/relax endpoints) are actually distinct from each other --
    a uniform max_atom_disp and a small set of repeating energies across
    unrelated seeds (different kicked atoms) is the signature of an ART
    search rediscovering the same handful of transitions over and over,
    not 50 genuinely different ones."""
    reps = []
    labels = []
    for pos in configs:
        assigned = None
        for ci, rep in enumerate(reps):
            if max_atom_displacement(pos, rep, box) < threshold:
                assigned = ci
                break
        if assigned is None:
            reps.append(pos)
            assigned = len(reps) - 1
        labels.append(assigned)
    return labels, len(reps)


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


def classify_mechanism(pos_plus, pos_minus, box, active_fraction=ACTIVE_FRACTION):
    """
    Characterize the atomic rearrangement a saddle connects, from the two
    relaxed endpoints find_decisive_push already produced (pos_plus,
    pos_minus) -- the actual transition, not just the saddle's local
    unstable direction.

    Reduces the per-atom displacement field to atom-order/count-invariant
    numbers:
      - participation_ratio / n_eff: the standard inverse-participation-
        ratio localization measure (sum(d^2))^2 / (N*sum(d^4)), as an
        EFFECTIVE atom count with no threshold -- one atom moving gives
        ~1, two equally-displaced atoms give ~2, N equally-displaced
        atoms give ~N, continuously.
      - n_active: a thresholded version (any atom moving more than
        active_fraction of the single largest displacement), used to
        build a concrete atom set for the swap/chain detection below,
        since that needs an actual set, not a continuous number.
      - radius_of_gyration: how spatially compact the active atoms are
        (PBC-unwrapped relative to the most-displaced one) -- a hop or
        pair swap is compact; a propagating chain is elongated.
      - longest_cycle / n_cycles: solves the optimal assignment (Hungarian
        algorithm) between {active atoms' pos_plus} and {active atoms'
        pos_minus} -- i.e. "which active atom ends up where which other
        active atom started" -- and decomposes it into cycles. A fixed
        point (atom i maps to itself) means it moved without exchanging
        places with another active atom (a hop into free volume with no
        one taking its place); a length-2 cycle is a direct pair swap; a
        length>=3 cycle is a ring -- the signature of free volume
        propagating through a chain of atoms (each moving into the next
        one's vacated spot).

    Returns a dict with all of the above plus a best-guess `category`
    string from simple rules on top of them -- meant as a sorting aid to
    pick representatives for manual OVITO inspection, not a rigorous
    physical classification. Tune active_fraction/the rules below against
    your own data if a category looks wrong on inspection.
    """
    dr = pos_plus - pos_minus
    dr -= box * np.round(dr / box)
    disp = np.linalg.norm(dr, axis=1)
    max_disp = disp.max()

    s2 = np.sum(disp**2)
    s4 = np.sum(disp**4)
    participation_ratio = (s2 * s2) / (len(disp) * s4) if s4 > 0 else 0.0
    n_eff = participation_ratio * len(disp)

    active = np.where(disp > active_fraction * max_disp)[0]
    n_active = len(active)

    if n_active >= 2:
        anchor_pos = pos_plus[active[np.argmax(disp[active])]]
        unwrapped = []
        for idx in active:
            d = pos_plus[idx] - anchor_pos
            d -= box * np.round(d / box)
            unwrapped.append(anchor_pos + d)
        unwrapped = np.array(unwrapped)
        centroid = unwrapped.mean(axis=0)
        radius_of_gyration = float(np.sqrt(np.mean(np.sum((unwrapped - centroid) ** 2, axis=1))))

        D = np.zeros((n_active, n_active))
        for a, i in enumerate(active):
            for b, j in enumerate(active):
                d = pos_plus[i] - pos_minus[j]
                d -= box * np.round(d / box)
                D[a, b] = np.linalg.norm(d)
        row_ind, col_ind = linear_sum_assignment(D)
        sigma = {int(active[a]): int(active[col_ind[a]]) for a in row_ind}

        visited = set()
        cycles = []
        for i in active:
            i = int(i)
            if i in visited:
                continue
            cyc = [i]
            visited.add(i)
            j = sigma[i]
            while j != i:
                cyc.append(j)
                visited.add(j)
                j = sigma[j]
            cycles.append(cyc)
        nontrivial_cycles = [c for c in cycles if len(c) >= 2]
    else:
        radius_of_gyration = 0.0
        nontrivial_cycles = []

    longest_cycle = max((len(c) for c in nontrivial_cycles), default=0)
    n_cycles = len(nontrivial_cycles)

    if n_active <= 1:
        category = "single-atom hop"
    elif longest_cycle == 2 and n_cycles == 1 and n_active == 2:
        category = "pair swap"
    elif longest_cycle >= 3:
        category = "chain / propagating free volume"
    elif longest_cycle == 0 and n_active <= 3:
        category = "localized hop (few atoms, no exchange)"
    else:
        category = "collective / diffuse"

    return dict(n_active=n_active, n_eff=n_eff, participation_ratio=participation_ratio,
                max_disp=float(max_disp), radius_of_gyration=radius_of_gyration,
                longest_cycle=longest_cycle, n_cycles=n_cycles, category=category)


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

    rows = parse_results_csv(RESULTS_CSV)
    if not rows:
        raise SystemExit(f"no rows parsed from {RESULTS_CSV}")

    # Sanity check: every row in a given results.csv shares the same starting
    # minimum, so its initial_energy column is one constant value -- read it
    # straight from the CSV's own first row instead of a hand-pasted
    # constant that goes stale the moment this script points at a new run.
    csv_initial_energy = rows[0]["initial_energy"]
    if any(abs(r["initial_energy"] - csv_initial_energy) > 1e-6 for r in rows):
        print("*** WARNING: this CSV's own initial_energy column is not constant across rows --")
        print("*** it may be the concatenation of more than one run. Treating row 1's value")
        print(f"*** ({csv_initial_energy}) as the reference; the check below may be meaningless.\n")

    if abs(initial_energy - csv_initial_energy) > 1e-6:
        print(f"*** WARNING: computed initial_energy = {initial_energy}")
        print(f"*** does NOT match the CSV's recorded initial_energy = {csv_initial_energy}")
        print(f"*** (difference = {initial_energy - csv_initial_energy})")
        print("*** LAMMPS_PATH and/or RC/EPSILON_TABLE/SIGMA_TABLE do NOT match what actually")
        print("*** produced this run -- fix that before trusting anything below.\n")
    else:
        print("initial_energy matches the CSV's recorded value -- LAMMPS_PATH and config")
        print("are at least consistent with the starting point of this run.\n")

    successes = [r for r in rows if r["success"]]
    print(f"{len(successes)} successful saddle(s) to validate out of {len(rows)} attempts in the CSV")
    print(f"push magnitudes tried (per side) = {PUSH_MAGNITUDES}, "
          f"same-configuration max-atom-displacement threshold = {MAX_ATOM_DISP_SAME_THRESHOLD}\n")

    n_pass = n_fail = n_error = 0
    saddle_seeds = []
    saddle_positions = []
    mechanisms = []

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

        saddle_seeds.append(r["seed"])
        saddle_positions.append(positions_saddle)

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

            mechanism_line = ""
            if is_different:
                mechanism = classify_mechanism(pos_plus, pos_minus, box)
                mechanisms.append(dict(seed=r["seed"], dump_file=r["dump_file"],
                                        barrier=r["final_energy"] - initial_energy, **mechanism))
                mechanism_line = (f"  mechanism={mechanism['category']}  n_active={mechanism['n_active']}  "
                                   f"n_eff={mechanism['n_eff']:.2f}  Rg={mechanism['radius_of_gyration']:.3f}\n")

            print(f"{line}\n"
                  f"  push_used=+-{mag:.3g}{'  (largest tried)' if mag == PUSH_MAGNITUDES[-1] else ''}  "
                  f"E_plus={e_plus:.6f}  E_minus={e_minus:.6f}  "
                  f"(saddle E={r['final_energy']:.6f}, barrier={r['final_energy'] - initial_energy:.6f})\n"
                  f"  max_atom_disp(plus,minus)={max_disp_plus_minus:.5f}  "
                  f"[whole-system RMSD(plus,minus)={rmsd_plus_minus:.5f}]\n"
                  f"  RMSD(plus,start)={rmsd_plus_start:.5f}  "
                  f"RMSD(minus,start)={rmsd_minus_start:.5f}\n"
                  f"  {verdict}\n"
                  f"{mechanism_line}")
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

    # =================================================================
    # Are the SADDLE POINTS THEMSELVES distinct from each other, or is
    # this ART search mostly rediscovering the same handful of
    # transitions? Compares each dump's saddle positions directly --
    # no push/relax involved -- so this is independent of everything
    # above. A uniform max_atom_disp(plus,minus) and a small set of
    # repeating "far" energies across unrelated seeds (different kicked
    # atoms) in the per-saddle output above is exactly what this would
    # also show as low cluster diversity.
    # =================================================================
    if len(saddle_positions) >= 2:
        labels, n_clusters = cluster_by_max_disp(saddle_positions, box, MAX_ATOM_DISP_SAME_THRESHOLD)
        print(f"\n=== Saddle-point diversity: {n_clusters} distinct cluster(s) "
              f"among {len(saddle_positions)} saddles (threshold={MAX_ATOM_DISP_SAME_THRESHOLD}) ===")
        from collections import defaultdict
        by_cluster = defaultdict(list)
        for seed, label in zip(saddle_seeds, labels):
            by_cluster[label].append(seed)
        for ci in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
            seeds = by_cluster[ci]
            print(f"  cluster {ci}: {len(seeds)} saddle(s) -- seeds {seeds}")
        if n_clusters < len(saddle_positions) / 2:
            print("Fewer clusters than half the saddle count: this ART search is substantially")
            print("rediscovering the same handful of transitions rather than finding distinct")
            print("saddle points -- worth revisiting phase-1 direction selection / kick-direction")
            print("continuity (diagnosed earlier as the low-diversity issue, not yet addressed).")

    # =================================================================
    # Mechanism classification summary -- group CONFIRMED saddles by
    # transition type (see classify_mechanism's docstring) and suggest
    # one representative dump per category to actually open in OVITO,
    # instead of looking through all of them by hand.
    # =================================================================
    if mechanisms:
        with open(MECHANISM_CSV, "w", newline="") as f:
            fieldnames = ["seed", "dump_file", "category", "barrier", "n_active", "n_eff",
                          "participation_ratio", "max_disp", "radius_of_gyration",
                          "longest_cycle", "n_cycles"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in mechanisms:
                writer.writerow({k: m[k] for k in fieldnames})
        print(f"\nWrote per-saddle mechanism descriptors to {MECHANISM_CSV} "
              f"({len(mechanisms)} confirmed saddles)")

        from collections import defaultdict
        by_category = defaultdict(list)
        for m in mechanisms:
            by_category[m["category"]].append(m)

        print(f"\n=== Mechanism breakdown: {len(by_category)} categories among "
              f"{len(mechanisms)} confirmed saddles ===")
        for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
            group = sorted(by_category[cat], key=lambda m: m["n_active"])
            rep = group[len(group) // 2]   # median by n_active -- a "typical" example, not an outlier
            print(f"  {cat}: {len(group)} saddle(s) -- seeds "
                  f"{[m['seed'] for m in group]}")
            print(f"      representative: seed={rep['seed']}  dump={rep['dump_file']}  "
                  f"n_active={rep['n_active']}  n_eff={rep['n_eff']:.2f}  "
                  f"Rg={rep['radius_of_gyration']:.3f}  barrier={rep['barrier']:.5f}")
        print(f"\nOpen each category's representative dump_file in OVITO for one example of each")
        print(f"mechanism -- see {MECHANISM_CSV} for the full per-saddle numbers if you want to")
        print(f"pick a different representative (e.g. by barrier height instead of n_active), or")
        print(f"tune ACTIVE_FRACTION/the category rules in classify_mechanism if a label looks")
        print(f"wrong once you've looked at it.")


if __name__ == "__main__":
    main()

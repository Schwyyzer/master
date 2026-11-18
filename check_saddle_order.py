# -*- coding: utf-8 -*-
"""
Independently verify that saddles found by run_parallel_fortran.jl (or the
older phase1/phase2 drivers) are genuine FIRST-ORDER saddles: the Hessian
AT the saddle itself should have exactly one eigenvalue below zero (the
unstable/reaction-coordinate direction), plus a handful of exact zero
modes from translational symmetry (3 expected under periodic boundaries --
PBC breaks rotational invariance, so there's no corresponding rotational
zero mode here), with every other eigenvalue positive.

This is a DIFFERENT, complementary check from validate_saddles.py's
push+relax test, not a replacement for it: that one confirms pushing
along the ONE eigenvector ART tracked connects two distinct basins, which
is consistent with a first-order saddle but doesn't rule out e.g. a
second-order saddle where pushing along just one of its two unstable
directions still happens to separate into two different endpoints. This
script instead counts the Hessian's eigenvalues directly, at the saddle
itself -- no push or relaxation involved.

Must be run from the same directory as validate_saddles.py: physics
(parse_lammps_data/build_neighbor_pairs/compute_energy/RC/EPSILON_TABLE/
SIGMA_TABLE) and the max_atom_displacement metric are imported from it
directly rather than re-copied, so the two scripts can't drift apart --
which also means this script's correctness depends on validate_saddles.py's
OWN "Config -- EDIT THESE" block already matching what produced your CSV.

Full dense diagonalization (scipy.linalg.eigh) is the default -- O(natoms^3)
but UNAMBIGUOUSLY correct, no risk of missing an eigenvalue a partial
solve's search window didn't reach. For a large system this is slow
per-saddle (seconds to tens of seconds), so this is meant to run on
DISTINCT saddles, not blindly on every row of a large results.csv: it
deduplicates structurally-identical successes within its own run (same
max_atom_displacement metric/default threshold as validate_saddles.py's
saddle repertoire) before doing any eigenvalue work at all, same
reasoning as that feature -- same starting system, many seeds, ART
rediscovers the same handful of transitions. This dedup is self-contained
to a single run of THIS script (not persisted across runs the way
validate_saddles.py's saddle_repertoire/ is); point this script at the
representative dump files from that repertoire or from
mechanism_classification.csv instead of a raw results.csv if you want to
skip straight to only the already-known-distinct saddles.

PARTIAL_EIGENSOLVE below is an opt-in faster alternative for triaging a
large CSV before committing to the full check on survivors -- see its
comment for an important caveat on why it uses which="SA" and not the
more common shift-invert "closest to zero" recipe (e.g. this project's
own Julia lowest_modes).

Run with: python3 check_saddle_order.py
"""
import csv
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from validate_saddles import (
    parse_lammps_data, parse_dump_frame, build_neighbor_pairs,
    compute_energy, max_atom_displacement, RC, EPSILON_TABLE, SIGMA_TABLE,
)

# =====================================================================
# Settings -- edit these
# =====================================================================
RESULTS_CSV = "results.csv"
LAMMPS_PATH = "FILL_ME_IN"   # the EXACT LAMMPS_PATH that produced this CSV
                              # -- see validate_saddles.py's own note on
                              # this; same risk, same reason it's not
                              # defaulted to a guess.

# Same setting/reasoning as validate_saddles.py's REQUIRE_CONVERGED (see
# that file for the full explanation): success=True does not mean the
# search force-converged, only that it wasn't discarded by a guard. The
# Hessian's eigenvalues are well-defined at any configuration, converged
# or not -- but "how many are negative" only answers "what TYPE of
# critical point is this" when it IS one; at a position that timed out
# (gradient still well above force_tolerance) it isn't a critical point
# at all, so the count describes local curvature at an arbitrary location,
# not a saddle classification. True (default) excludes success=True-but-
# not-converged rows; has no effect if the CSV's "converged" column is
# absent, which is printed loudly rather than silently skipped.
REQUIRE_CONVERGED = True

# Mass used in the Hessian -- [dummy, type1, type2], matching
# EPSILON_TABLE/SIGMA_TABLE's 1-indexed convention (index 0 unused).
# [0.0, 1.0, 1.0] (unit mass) matches run_parallel_fortran.jl's active
# pipeline, which overrides config.jl's own mass=[2.0,1.0] to unit mass
# (see that file's header for why). Set to [0.0, 2.0, 1.0] instead if
# these saddles came from one of the legacy phase1/phase2 drivers, which
# do NOT perform that override and rely on config.jl's own mass-weighted
# Hessian directly.
MASS = np.array([0.0, 1.0, 1.0])

# Same default/meaning as art_core_fortran.jl's own zero_eigenvalue_threshold:
# |eigenvalue| below this counts as a (translational) zero mode rather
# than a genuine positive or negative direction. Exactly 3 such modes are
# EXPECTED under periodic boundaries (translations only -- PBC breaks
# rotational invariance, so there's no rotational zero mode to go with
# them here); a different count is itself worth noticing and is printed
# either way, not silently assumed.
ZERO_EIGENVALUE_THRESHOLD = 0.1

# Same metric and default value as validate_saddles.py's
# MAX_ATOM_DISP_SAME_THRESHOLD -- skip the expensive eigenvalue check
# entirely for a saddle that structurally matches one already checked
# earlier in THIS run (see the module docstring for why this is scoped
# to a single run rather than persisted like that script's repertoire).
DEDUP_MAX_DISP_THRESHOLD = 0.05

# Opt-in: scipy.sparse.linalg.eigsh(which="SA") instead of a full dense
# solve -- the ALGEBRAICALLY smallest N_EIGENVALUES eigenvalues, sorted
# ascending. Deliberately NOT the more common shift-invert "closest to
# zero by magnitude" recipe (sigma=0, which="LM" -- what this project's
# own Julia lowest_modes uses to track the softest mode live during a
# search): that recipe can silently MISS a large-magnitude negative
# eigenvalue if enough small-magnitude positive ones crowd it out of the
# window, since it ranks by |eigenvalue - 0| regardless of sign. "SA" has
# no such blind spot -- it always returns the true lowest N_EIGENVALUES
# values, sorted, with no bias from how spread out the positive spectrum
# is. The only way it can be wrong is if the window isn't wide enough to
# reach a positive eigenvalue at all; this script checks for exactly
# that (every computed value still negative) and automatically escalates
# THAT ONE saddle to the full dense solve instead of reporting a
# possibly-incomplete count.
PARTIAL_EIGENSOLVE = False
N_EIGENVALUES = 20

OUTPUT_CSV = "first_order_check.csv"


# =====================================================================
# Hessian -- mirrors Hessian.jl's build_hessian_fast (unit-mass case by
# default; see MASS above), vectorized over pairs the same way
# validate_saddles.py's own force()/compute_energy() already are.
# =====================================================================
def build_hessian(positions, cid, mass, box, pairs):
    natoms = positions.shape[0]
    rank = 3 * natoms
    i = pairs[:, 0]
    j = pairs[:, 1]

    dr = positions[i] - positions[j]
    dr -= box * np.round(dr / box)
    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)

    ti = cid[i]
    tj = cid[j]
    eps = EPSILON_TABLE[ti, tj]
    sig = SIGMA_TABLE[ti, tj]
    mi = mass[ti]
    mj = mass[tj]

    diag_i = 1.0 / mi
    diag_j = 1.0 / mj
    offdiag = -1.0 / np.sqrt(mi * mj)

    sr2 = (sig ** 2) / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2

    dV = -4.0 * eps * (12.0 * sr12 / r - 6.0 * sr6 / r)
    ddV = 4.0 * eps * (12.0 * 13.0 * sr12 / r2 - 6.0 * 7.0 * sr6 / r2)

    fn0 = dV / r
    fn1 = ddV - fn0
    invr2 = 1.0 / r2

    dx, dy, dz = dr[:, 0], dr[:, 1], dr[:, 2]
    xx = fn1 * dx * dx * invr2 + fn0
    yy = fn1 * dy * dy * invr2 + fn0
    zz = fn1 * dz * dz * invr2 + fn0
    xy = fn1 * dx * dy * invr2
    xz = fn1 * dx * dz * invr2
    yz = fn1 * dy * dz * invr2

    # (row-component, col-component, value), 0-indexed -- same 9 entries
    # per pair's 3x3 block as build_hessian_fast
    comps = [(0, 0, xx), (1, 1, yy), (2, 2, zz),
             (0, 1, xy), (1, 0, xy), (0, 2, xz), (2, 0, xz), (1, 2, yz), (2, 1, yz)]

    rows, cols, vals = [], [], []
    for a, b, c in comps:
        rows.append(3 * i + a); cols.append(3 * j + b); vals.append(offdiag * c)
        rows.append(3 * j + a); cols.append(3 * i + b); vals.append(offdiag * c)
        rows.append(3 * i + a); cols.append(3 * i + b); vals.append(diag_i * c)
        rows.append(3 * j + a); cols.append(3 * j + b); vals.append(diag_j * c)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(rank, rank)).tocsr()


def eigenvalues_for_saddle(H):
    rank = H.shape[0]
    if PARTIAL_EIGENSOLVE:
        k = min(N_EIGENVALUES, rank - 2)
        evals = np.sort(scipy.sparse.linalg.eigsh(H, k=k, which="SA", return_eigenvectors=False))
        if np.all(evals < 0):
            evals = np.sort(scipy.linalg.eigh(H.toarray(), eigvals_only=True))
            return evals, "full (partial window of %d never reached a positive eigenvalue)" % k
        return evals, f"partial (k={k}, SA)"
    evals = np.sort(scipy.linalg.eigh(H.toarray(), eigvals_only=True))
    return evals, "full"


def classify_eigenvalues(evals, threshold):
    n_negative = int(np.sum(evals <= -threshold))
    n_zero = int(np.sum(np.abs(evals) < threshold))
    n_positive = int(np.sum(evals >= threshold))
    if n_negative == 1:
        label = "FIRST-ORDER SADDLE (confirmed)"
    elif n_negative == 0:
        label = "NOT A SADDLE -- this is a local minimum (0 negative eigenvalues)"
    else:
        label = f"NOT FIRST-ORDER -- {n_negative} negative eigenvalues (higher-order saddle)"
    return n_negative, n_zero, n_positive, label


# =====================================================================
# results.csv parsing -- header-driven; the diagnostic columns
# (crit_eigenvalue/second_eigenvalue/is_saddle/converged) are read if
# present and simply ignored for an older CSV that lacks them.
# =====================================================================
def parse_results_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_converged = "converged" in (reader.fieldnames or [])
        for row in reader:
            if not row.get("seed"):
                continue

            def optfloat(name):
                v = row.get(name)
                return float(v) if v not in (None, "") else None

            def optbool(name):
                v = row.get(name)
                return v.strip().lower() == "true" if v not in (None, "") else None

            rows.append({
                "seed": int(row["seed"]),
                "success": row["success"].strip().lower() == "true",
                "dump_file": row["dump_file"],
                "initial_energy": float(row["initial_energy"]),
                "crit_eigenvalue": optfloat("crit_eigenvalue"),
                "second_eigenvalue": optfloat("second_eigenvalue"),
                "is_saddle": optbool("is_saddle"),
                "converged": optbool("converged"),
            })
    return rows, has_converged


def csv_agreement_note(is_saddle, n_negative):
    if is_saddle is None:
        return ""
    if is_saddle and n_negative == 1:
        return "  [AGREES -- CSV's tracked eigenvalue and the full count both say first-order]"
    if is_saddle and n_negative == 0:
        return ("  [CSV's tracked eigenvalue was negative, but the full spectrum here has NO negative "
                 "eigenvalues at all -- the search's single-eigenvalue estimate disagrees outright, not "
                 "just about the count]")
    if is_saddle and n_negative >= 2:
        return (f"  [CSV's tracked eigenvalue was negative (consistent with SOME instability), but the "
                 f"full spectrum has {n_negative} negative eigenvalues, not 1 -- a higher-order saddle, "
                 f"not first-order; the search's mode tracking only ever saw one of them]")
    if not is_saddle and n_negative >= 1:
        return (f"  [CSV did NOT call this a saddle, but {n_negative} negative eigenvalue(s) found here "
                f"-- worth a second look]")
    return "  [AGREES -- CSV and the full count both say this isn't a saddle]"


def main():
    if LAMMPS_PATH == "FILL_ME_IN":
        raise SystemExit("set LAMMPS_PATH at the top of this script before running")

    data0 = parse_lammps_data(LAMMPS_PATH)
    box = np.array([data0["lx"], data0["ly"], data0["lz"]])
    positions0 = np.stack([data0["x"], data0["y"], data0["z"]], axis=1)
    pairs0 = build_neighbor_pairs(positions0, RC, box)
    initial_energy = compute_energy(positions0, data0["cid"], box, pairs0)

    rows, has_converged_col = parse_results_csv(RESULTS_CSV)
    if not rows:
        raise SystemExit(f"no rows parsed from {RESULTS_CSV}")

    csv_initial_energy = rows[0]["initial_energy"]
    if abs(initial_energy - csv_initial_energy) > 1e-6:
        print(f"*** WARNING: computed initial_energy = {initial_energy}")
        print(f"*** does NOT match the CSV's recorded initial_energy = {csv_initial_energy}")
        print("*** LAMMPS_PATH and/or validate_saddles.py's RC/EPSILON_TABLE/SIGMA_TABLE do NOT")
        print("*** match what actually produced this run -- fix that before trusting anything below.\n")
    else:
        print("initial_energy matches the CSV's recorded value -- LAMMPS_PATH and physics")
        print("are at least consistent with the starting point of this run.\n")

    successes = [r for r in rows if r["success"]]
    if REQUIRE_CONVERGED:
        if not has_converged_col:
            print("*** WARNING: REQUIRE_CONVERGED=True but this CSV has no 'converged' column -- "
                  "cannot exclude timeouts, so nothing is being excluded. Set REQUIRE_CONVERGED=False")
            print("*** to silence this, or confirm this CSV's driver never writes timeouts as "
                  "success=True before trusting the counts below.\n")
        else:
            n_before = len(successes)
            successes = [r for r in successes if r["converged"]]
            n_excluded = n_before - len(successes)
            if n_excluded > 0:
                print(f"excluded {n_excluded} success=True row(s) that did NOT actually converge "
                      f"(timed out) -- REQUIRE_CONVERGED=True; a non-stationary point isn't a saddle "
                      f"of any order, see that setting's comment.\n")

    method_desc = (f"partial (k={N_EIGENVALUES}, SA, auto-escalates to full if inconclusive)"
                   if PARTIAL_EIGENSOLVE else "full dense")
    print(f"{len(successes)} successful saddle(s) to check out of {len(rows)} attempts in the CSV")
    print(f"eigenvalue method: {method_desc}")
    print(f"zero-mode threshold = {ZERO_EIGENVALUE_THRESHOLD}, "
          f"within-run dedup threshold = {DEDUP_MAX_DISP_THRESHOLD}\n")

    checked_positions = []
    checked_results = []   # parallel list: (n_negative, n_zero, n_positive, label)
    output_rows = []
    n_first_order = n_minimum = n_higher_order = n_error = n_dedup = 0

    for r in successes:
        line = f"seed={r['seed']}  dump={r['dump_file']}"
        try:
            positions, _move, ids, types = parse_dump_frame(r["dump_file"])
        except (OSError, ValueError, StopIteration) as e:
            print(f"{line}  SKIPPED (couldn't read/parse dump file: {e})")
            n_error += 1
            continue

        if not np.array_equal(ids, data0["id"]) or not np.array_equal(types, data0["cid"]):
            print(f"{line}  SKIPPED (atom ids/types don't match LAMMPS_PATH -- wrong LAMMPS_PATH?)")
            n_error += 1
            continue

        dup_idx = next((k for k, prev in enumerate(checked_positions)
                         if max_atom_displacement(positions, prev, box) < DEDUP_MAX_DISP_THRESHOLD), None)

        if dup_idx is not None:
            n_negative, n_zero, n_positive, label = checked_results[dup_idx]
            print(f"{line}\n  matches already-checked saddle #{dup_idx} -- reusing its result\n  {label}"
                  f"{csv_agreement_note(r['is_saddle'], n_negative)}\n")
            n_dedup += 1
        else:
            try:
                pairs = build_neighbor_pairs(positions, RC, box)
                H = build_hessian(positions, data0["cid"], MASS, box, pairs)
                evals, method = eigenvalues_for_saddle(H)
            except Exception as e:
                print(f"{line}\n  ERROR building/diagonalizing Hessian: {e}\n")
                n_error += 1
                continue

            n_negative, n_zero, n_positive, label = classify_eigenvalues(evals, ZERO_EIGENVALUE_THRESHOLD)
            lowest = evals[:min(8, len(evals))]

            print(f"{line}  [{method}]")
            print(f"  {n_negative} negative, {n_zero} ~zero (expect 3, translational), {n_positive} positive")
            print(f"  lowest {len(lowest)}: {np.array2string(lowest, precision=5)}")
            if r["crit_eigenvalue"] is not None:
                print(f"  CSV recorded: crit_eigenvalue={r['crit_eigenvalue']:.5f}  "
                      f"second_eigenvalue={r['second_eigenvalue']}  is_saddle={r['is_saddle']}  "
                      f"converged={r['converged']}")
            print(f"  {label}{csv_agreement_note(r['is_saddle'], n_negative)}\n")

            checked_positions.append(positions)
            checked_results.append((n_negative, n_zero, n_positive, label))

        if n_negative == 1:
            n_first_order += 1
        elif n_negative == 0:
            n_minimum += 1
        else:
            n_higher_order += 1

        output_rows.append(dict(
            seed=r["seed"], dump_file=r["dump_file"],
            n_negative=n_negative, n_zero=n_zero, n_positive=n_positive,
            classification=label, csv_crit_eigenvalue=r["crit_eigenvalue"],
            csv_is_saddle=r["is_saddle"], deduplicated=(dup_idx is not None),
        ))

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["seed", "dump_file", "n_negative", "n_zero", "n_positive",
                      "classification", "csv_crit_eigenvalue", "csv_is_saddle", "deduplicated"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"=== Summary: {n_first_order} first-order / {n_minimum} actually minima / "
          f"{n_higher_order} higher-order / {n_error} skipped, out of {len(successes)} successes ===")
    print(f"({len(checked_positions)} structurally distinct saddle(s) actually diagonalized, "
          f"{n_dedup} reused as within-run duplicates)")
    print(f"Wrote per-saddle results to {OUTPUT_CSV}")
    if n_minimum > 0 or n_higher_order > 0:
        print("\nSaddles that aren't confirmed first-order are worth a closer look -- cross-reference")
        print("validate_saddles.py's push+relax verdict for the same seed, and whether the CSV's own")
        print("crit_eigenvalue/second_eigenvalue already hinted at it (see the bracketed notes above).")


if __name__ == "__main__":
    main()

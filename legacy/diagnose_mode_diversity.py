# -*- coding: utf-8 -*-
"""
Diagnostic for the "only ever find ~2 distinct saddle points" question.

Hypothesis being tested: phase 1 picks its escape direction by fully
diagonalizing the Hessian of the *relaxed, post-kick* structure and taking
its lowest-nonzero eigenvector (see new_start_1_1_1.py, right after
`positions = relax(positions, data, neighbour_pairs, first_move)`). That
diagonalization has no explicit mechanism tying it to *which* atom you
kicked -- if the starting minimum has one or two persistently-soft global
modes, `lowest_nonzero_mode` will keep returning (a sign-flipped version
of) the same mode almost regardless of the kick, which would explain
finding only ~2 distinct escape pathways.

This script does NOT change anything about the simulation. It reuses the
exact physics functions from new_start_1_1_1.py (copied verbatim below,
unmodified) and just adds instrumentation:

  1. Diagonalizes the Hessian of the UNKICKED, already-relaxed structure
     and prints its lowest ~20 eigenvalues. If 1-2 of them sit far below
     the rest, that's the dominant global mode showing up directly.

  2. For N different random kicks (different atom, different direction),
     reproduces exactly the phase-1 direction-selection step and records,
     per trial:
       - which atom was kicked
       - the eigenvalue of the chosen mode
       - what fraction of the mode's norm sits on the kicked atom itself
         (low + similar across trials => the mode is a global, delocalized
         mode, not something localized around the kick)
       - the full mode vector, for a pairwise comparison afterwards

  3. Prints the pairwise |cosine similarity| matrix between all trials'
     chosen directions. Values clustered near 1.0 (up to sign, which is
     already accounted for by the same sign convention new_start_1_1_1.py
     uses) mean essentially every kick is finding the same mode.

Run this next to your real config_small_relaxed.lammps file (edit
`inpath` below if needed), with the same Python environment you use for
new_start_1_1_1.py (numpy, scipy).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, ArpackError

# =====================================================================
# Physical parameters -- copied verbatim from new_start_1_1_1.py
# =====================================================================
first_move_modifier = 0.1
relaxation_step_magnitude = 0.0005
max_relaxation_steps = 50

mass = {1: 2.0, 2: 1.0}
epsilon_table = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0]
])
sigma_table = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 11/12],
    [0.0, 11/12, 5/6]
])
rc = 3  # LJ cutoff -- keep this matched to whatever you're investigating


# =====================================================================
# Functions copied verbatim from new_start_1_1_1.py (unmodified)
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

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

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


def perpendicular_forces(forces, moves):
    f = forces.reshape(-1)
    m = moves.reshape(-1)
    denom = np.dot(m, m)
    if denom > 0:
        return (f - (np.dot(f, m) / denom) * m).reshape(forces.shape)
    return forces.copy()


def force(positions, data, neighbors, eps_matrix=epsilon_table, sigma_matrix=sigma_table):
    total_force = np.zeros((len(positions), 3))
    i = neighbors[:, 0]
    j = neighbors[:, 1]
    ri = positions[i]
    rj = positions[j]
    dr = ri - rj
    box = np.array([data['lx'], data['ly'], data['lz']])
    dr -= box * np.round(dr / box)
    r2 = np.sum(dr**2, axis=1)
    ti = data['cid'][i]
    tj = data['cid'][j]
    eps = eps_matrix[ti, tj]
    sig = sigma_matrix[ti, tj]
    sig6 = sig**6
    sig12 = sig6**2
    pref = 24 * eps * (2 * sig12 / r2**7 - sig6 / r2**4)
    fij = pref[:, None] * dr
    np.add.at(total_force, i, fij)
    np.add.at(total_force, j, -fij)
    return total_force


def compute_energy(coords, types, box, eps_matrix, sigma_matrix, pairs):
    i = pairs[:, 0]
    j = pairs[:, 1]
    ri = coords[i]
    rj = coords[j]
    dr = ri - rj
    dr -= box * np.round(dr / box)
    r2 = np.sum(dr**2, axis=1)
    ti = types[i]
    tj = types[j]
    eps = eps_matrix[ti, tj]
    sig = sigma_matrix[ti, tj]
    inv_r6 = (sig**2 / r2)**3
    energy = 4 * eps * (inv_r6**2 - inv_r6)
    return np.sum(energy)


def relax(positions, data, neighbour_pairs, moves, eps_matrix=epsilon_table,
          sigma_matrix=sigma_table, maxIter=max_relaxation_steps):
    tol = 1e-5
    Lambda = relaxation_step_magnitude
    Iter = 0
    maxIter = 50

    max_force = np.inf
    while max_force > tol and Iter < maxIter:
        if Iter % 1 == 0:
            box = np.array([data['lx'], data['ly'], data['lz']])
            tree = cKDTree(positions % box, boxsize=box)
            neighbour_pairs = np.array(list(tree.query_pairs(3 + rc)))

        box = np.array([data['lx'], data['ly'], data['lz']])
        Force_vector = force(positions, data, neighbour_pairs, eps_matrix, sigma_matrix)
        perp_forces = perpendicular_forces(Force_vector, moves)
        max_force = np.max(np.linalg.norm(perp_forces, axis=1))

        pos_trial = positions + Lambda * perp_forces
        E_trial = compute_energy(pos_trial, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs)
        E_current = compute_energy(positions, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs)

        if E_trial < E_current:
            positions = pos_trial % box
            Lambda *= 1.05
        else:
            Lambda *= 0.5
            if Lambda < 1e-10:
                break
        Iter += 1
    return positions


def lowest_nonzero_mode(arr):
    arr = np.asarray(arr)
    temp = arr.copy()
    if len(arr) < 4:
        raise ValueError(f"Need at least 4 eigenvalues, got {len(arr)}")
    remaining = sorted(arr, key=lambda x: abs(x))[3:]
    for i in range(len(temp)):
        if min(remaining) == temp[i]:
            return i


def build_hessian_fast(positions, data, neighbour_pairs):
    natoms = data['natoms']
    cid = data['cid']
    lx, ly, lz = data['lx'], data['ly'], data['lz']
    box = np.array([lx, ly, lz])
    rank = 3 * natoms

    i = neighbour_pairs[:, 0]
    j = neighbour_pairs[:, 1]
    npairs = len(i)

    dr = positions[i] - positions[j]
    dr -= box * np.round(dr / box)
    dx, dy, dz = dr[:, 0], dr[:, 1], dr[:, 2]
    r2 = dx*dx + dy*dy + dz*dz
    r = np.sqrt(r2)

    ti = cid[i]
    tj = cid[j]
    eps = epsilon_table[ti, tj]
    sig = sigma_table[ti, tj]
    mi = np.array([mass[int(x)] for x in ti])
    mj = np.array([mass[int(x)] for x in tj])
    diag_i = 1.0 / mi
    diag_j = 1.0 / mj
    offdiag = -1.0 / np.sqrt(mi * mj)

    sr2 = (sig * sig) / r2
    sr6 = sr2**3
    sr12 = sr6 * sr6

    dV = -4.0 * eps * (12.0 * sr12 / r - 6.0 * sr6 / r)
    ddV = 4.0 * eps * (12.0 * 13.0 * sr12 / r2 - 6.0 * 7.0 * sr6 / r2)
    fn0 = dV / r
    fn1 = ddV - fn0
    invr2 = 1.0 / r2

    xx = fn1 * dx*dx * invr2 + fn0
    yy = fn1 * dy*dy * invr2 + fn0
    zz = fn1 * dz*dz * invr2 + fn0
    xy = fn1 * dx*dy * invr2
    xz = fn1 * dx*dz * invr2
    yz = fn1 * dy*dz * invr2

    diag_blocks = np.zeros((natoms, 3, 3))
    np.add.at(diag_blocks[:, 0, 0], i, diag_i * xx)
    np.add.at(diag_blocks[:, 1, 1], i, diag_i * yy)
    np.add.at(diag_blocks[:, 2, 2], i, diag_i * zz)
    np.add.at(diag_blocks[:, 0, 1], i, diag_i * xy)
    np.add.at(diag_blocks[:, 1, 0], i, diag_i * xy)
    np.add.at(diag_blocks[:, 0, 2], i, diag_i * xz)
    np.add.at(diag_blocks[:, 2, 0], i, diag_i * xz)
    np.add.at(diag_blocks[:, 1, 2], i, diag_i * yz)
    np.add.at(diag_blocks[:, 2, 1], i, diag_i * yz)
    np.add.at(diag_blocks[:, 0, 0], j, diag_j * xx)
    np.add.at(diag_blocks[:, 1, 1], j, diag_j * yy)
    np.add.at(diag_blocks[:, 2, 2], j, diag_j * zz)
    np.add.at(diag_blocks[:, 0, 1], j, diag_j * xy)
    np.add.at(diag_blocks[:, 1, 0], j, diag_j * xy)
    np.add.at(diag_blocks[:, 0, 2], j, diag_j * xz)
    np.add.at(diag_blocks[:, 2, 0], j, diag_j * xz)
    np.add.at(diag_blocks[:, 1, 2], j, diag_j * yz)
    np.add.at(diag_blocks[:, 2, 1], j, diag_j * yz)

    n_off = 18 * npairs
    n_diag = 9 * natoms
    rows = np.empty(n_off + n_diag, dtype=np.int32)
    cols = np.empty(n_off + n_diag, dtype=np.int32)
    vals = np.empty(n_off + n_diag, dtype=np.float64)
    ptr = 0

    components = [
        (0, 0, xx), (1, 1, yy), (2, 2, zz),
        (0, 1, xy), (1, 0, xy), (0, 2, xz),
        (2, 0, xz), (1, 2, yz), (2, 1, yz),
    ]
    for a, b, comp in components:
        n = npairs
        rows[ptr:ptr+n] = 3*i + a
        cols[ptr:ptr+n] = 3*j + b
        vals[ptr:ptr+n] = offdiag * comp
        ptr += n
        rows[ptr:ptr+n] = 3*j + a
        cols[ptr:ptr+n] = 3*i + b
        vals[ptr:ptr+n] = offdiag * comp
        ptr += n

    atom_ids = np.arange(natoms)
    for a in range(3):
        for b in range(3):
            n = natoms
            rows[ptr:ptr+n] = 3*atom_ids + a
            cols[ptr:ptr+n] = 3*atom_ids + b
            vals[ptr:ptr+n] = diag_blocks[:, a, b]
            ptr += n

    H = coo_matrix((vals, (rows, cols)), shape=(rank, rank))
    return H


# =====================================================================
# Diagnostic driver (new code)
# =====================================================================
def cosine(u, v):
    u = u.reshape(-1)
    v = v.reshape(-1)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def run_diagnostic(inpath, n_trials=20, seed0=0):
    data = parse_lammps_data(inpath)
    N = data['natoms']
    box = np.array([data['lx'], data['ly'], data['lz']])
    positions0 = np.stack([data['x'], data['y'], data['z']], axis=1) % box

    tree = cKDTree(positions0, boxsize=box)
    pairs0 = np.array(list(tree.query_pairs(rc)))

    print(f"natoms = {N}, box = {box}, npairs(rc={rc}) = {len(pairs0)}\n")

    # --- 1) spectrum of the pristine, unkicked structure -----------------
    H0 = build_hessian_fast(positions0, data, pairs0)
    try:
        vals0 = eigsh(H0.tocsr(), k=min(20, 3*N-1), sigma=1e-6, which='LM',
                      return_eigenvectors=False)
    except ArpackError:
        vals0 = eigsh(H0.tocsr(), k=min(10, 3*N-1), sigma=1e-6, which='LM',
                      return_eigenvectors=False)
    vals0 = np.sort(vals0)
    print("=== Lowest eigenvalues of the UNKICKED relaxed structure ===")
    print(vals0)
    print("(look for 1-2 values sitting far below the rest -- that's a")
    print(" dominant global soft mode independent of any kick)\n")

    # --- 2) per-trial: kick, relax, pick phase-1 direction ----------------
    directions = []
    kicked_atoms = []
    print("=== Per-trial phase-1 direction selection ===")
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + t)
        idx = int(rng.integers(0, N))
        kick_dir = rng.uniform(-1, 1, size=3)
        kick_dir /= np.linalg.norm(kick_dir)
        first_move = np.zeros((N, 3))
        first_move[idx] = kick_dir * first_move_modifier

        positions = (positions0 + first_move) % box
        positions = relax(positions, data, pairs0, first_move)

        H = build_hessian_fast(positions, data, pairs0)
        try:
            vals, vecs = eigsh(H.tocsr(), k=10, sigma=1e-3, which='LM', return_eigenvectors=True)
        except ArpackError:
            vals, vecs = eigsh(H.tocsr(), k=10, sigma=1e-6, which='LM', return_eigenvectors=True)

        mode_idx = lowest_nonzero_mode(vals)
        mode = vecs[:, mode_idx].copy()

        # same sign convention as new_start_1_1_1.py
        if np.dot(mode, first_move.reshape(-1)) < 0:
            mode = -mode

        mode3 = mode.reshape(-1, 3)
        atom_norms = np.linalg.norm(mode3, axis=1)
        local_frac = atom_norms[idx] / np.linalg.norm(mode3)

        directions.append(mode)
        kicked_atoms.append(idx)
        print(f"trial {t:2d}: kicked atom {idx:4d}  mode_eigval={vals[mode_idx]:+.6f}  "
              f"local_fraction_on_kicked_atom={local_frac:.4f}")

    # --- 3) pairwise similarity between chosen directions -----------------
    directions = np.array(directions)
    n = len(directions)
    sim = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            sim[a, b] = abs(cosine(directions[a], directions[b]))

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("\n=== |cosine similarity| between different trials' phase-1 directions ===")
    print("(near 1.0 across the board => (almost) every kick finds the same mode)")
    print(sim)
    off_diag_mean = (sim.sum() - n) / (n * n - n)
    print(f"\nmean off-diagonal |cosine similarity| = {off_diag_mean:.4f}")


if __name__ == "__main__":
    inpath = r"/home/schwyyzer/Desktop/Master Thesis/config_small_relaxed.lammps"
    run_diagnostic(inpath, n_trials=20, seed0=0)

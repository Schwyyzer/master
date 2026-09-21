# -*- coding: utf-8 -*-
"""
Standalone ART driver for the real mdyn.lammps system, reusing the exact
physics functions from new_start_1_1_1.py (copied verbatim -- parse,
force, energy, relax, Hessian, calculate_moves, angle_between) since no
Julia runtime is available in this sandbox. Adds a clean per-attempt
wrapper + a small batch driver on top.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, ArpackError

# =====================================================================
# Config (defaults straight from new_start_1_1_1.py unless noted)
# =====================================================================
first_move_modifier = 0.1
eigenvalue_cutoff = -0.2
move_phase1_modifier = 0.1
move_phase2_modifier = 0.001
dot_product_saddle_cutoff = 0.00025
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
rc = 3.0

# =====================================================================
# Functions copied verbatim from new_start_1_1_1.py
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
          sigma_matrix=sigma_table, maxIter=max_relaxation_steps, skin=3.0):
    tol = 1e-5
    Lambda = relaxation_step_magnitude
    Iter = 0
    maxIter = 50
    box = np.array([data['lx'], data['ly'], data['lz']])

    if isinstance(moves, np.ndarray):
        max_force = np.inf
        while max_force > tol and Iter < maxIter:
            if Iter % 1 == 0:
                tree = cKDTree(positions % box, boxsize=box)
                neighbour_pairs = np.array(list(tree.query_pairs(skin + rc)))

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
    else:
        tol_force = 0.001
        Lambda = relaxation_step_magnitude
        max_force = np.inf
        while max_force > tol_force and Iter < 5000:
            if Iter % 50 == 0:
                tree = cKDTree(positions % box, boxsize=box)
                neighbour_pairs = np.array(list(tree.query_pairs(rc)))
            Force_vector = force(positions, data, neighbour_pairs, eps_matrix, sigma_matrix)
            max_force = np.max(np.linalg.norm(Force_vector, axis=1))
            pos_trial = positions + Lambda * Force_vector
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


def calculate_moves(positions, data, neighbour_pairs, move_phase2_modifier):
    k = 15
    H = build_hessian_fast(positions, data, neighbour_pairs)
    try:
        eigenvalues, eigenvectors = eigsh(H.tocsr(), k=k, sigma=1e-6, which='LM', return_eigenvectors=True)
        if lowest_nonzero_mode(eigenvalues) > 9:
            print('Warning, 10 eigenvalues not enough')
    except ArpackError:
        print('Arpack Error prevented, lowering amount of eigenvalues calculated')
        eigenvalues, eigenvectors = eigsh(H.tocsr(), k=10, sigma=1e-6, which='LM', return_eigenvectors=True)
    moves = move_phase2_modifier*eigenvectors[:, lowest_nonzero_mode(eigenvalues)]
    moves = moves.reshape(-1, 3)
    return moves, eigenvalues[lowest_nonzero_mode(eigenvalues)]


def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


# =====================================================================
# Per-attempt driver (new code, mirrors new_start_1_1_1.py's main loop body)
# =====================================================================
def run_attempt(data, positions0, box, seed, verbose=False, max_total_iter=2000):
    rng = np.random.default_rng(seed)
    N = data['natoms']

    tree = cKDTree(positions0, boxsize=box)
    neighbour_pairs = np.array(list(tree.query_pairs(rc)))

    random_particle_index = int(rng.integers(0, N))
    kick_dir = rng.uniform(-1, 1, size=3)
    kick_dir /= np.linalg.norm(kick_dir)
    first_move = np.zeros((N, 3))
    first_move[random_particle_index] = first_move_modifier * kick_dir

    positions = positions0 + first_move
    t0 = time.time()
    positions = relax(positions, data, neighbour_pairs, first_move)

    H = build_hessian_fast(positions, data, neighbour_pairs)
    eigenvalues, eigenvectors = eigsh(H.tocsr(), k=10, sigma=1e-3, which='LM', return_eigenvectors=True)

    moves = move_phase1_modifier * eigenvectors[:, lowest_nonzero_mode(eigenvalues)]
    moves = moves.reshape(-1, 3)
    if np.dot(moves.reshape(-1), first_move.reshape(-1)) < 0:
        moves *= -1

    iteration = 0
    while eigenvalues[lowest_nonzero_mode(eigenvalues)] > eigenvalue_cutoff and iteration < max_total_iter:
        positions += moves
        positions = relax(positions, data, neighbour_pairs, moves)
        H = build_hessian_fast(positions, data, neighbour_pairs)
        eigenvalues = eigsh(H.tocsr(), k=10, sigma=0, which='LM', return_eigenvectors=False)
        iteration += 1
        if verbose:
            print(f"  phase1 it={iteration} eig={eigenvalues[lowest_nonzero_mode(eigenvalues)]:.5f}")

    if iteration == 0 or eigenvalues[lowest_nonzero_mode(eigenvalues)] > eigenvalue_cutoff:
        return dict(success=False, reason="phase1_no_destabilize", seed=seed,
                     kicked_atom=random_particle_index, iteration=iteration,
                     steps_part1=iteration, elapsed=time.time()-t0)

    steps_part1 = iteration

    # eigenvectors from before the phase-1 loop are stale (the loop only kept
    # eigenvalues, via return_eigenvectors=False, matching new_start_1_1_1.py);
    # recompute once with vectors for the phase-1 -> phase-2 handoff direction.
    eigenvalues, eigenvectors = eigsh(H.tocsr(), k=10, sigma=1e-6, which='LM', return_eigenvectors=True)
    crit_eigenvalue = eigenvalues[lowest_nonzero_mode(eigenvalues)]
    moves = move_phase2_modifier * eigenvectors[:, lowest_nonzero_mode(eigenvalues)]
    moves = moves.reshape(-1, 3)

    Force_vector = force(positions, data, neighbour_pairs)
    angles = []

    while (crit_eigenvalue < 0 and iteration < max_total_iter and
           abs(np.dot(moves.reshape(-1), Force_vector.reshape(-1))) / move_phase2_modifier > dot_product_saddle_cutoff):
        old_move = moves
        moves, crit_eigenvalue = calculate_moves(positions, data, neighbour_pairs, move_phase2_modifier)
        if np.dot(moves.reshape(-1), Force_vector.reshape(-1)) > 0:
            moves *= -1
        positions += moves
        positions = relax(positions, data, neighbour_pairs, moves)
        Force_vector = force(positions, data, neighbour_pairs)
        angle = angle_between(old_move.reshape(-1), moves.reshape(-1)) / np.pi * 180
        angles.append(angle)
        iteration += 1
        if verbose:
            print(f"  phase2 it={iteration} eig={crit_eigenvalue:.5f} angle={angle:.1f}")

    final_energy = compute_energy(positions, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs)
    success = crit_eigenvalue < 0 and iteration < max_total_iter and (iteration - steps_part1) > 0

    return dict(
        success=success, reason=None if success else "phase2_did_not_converge",
        seed=seed, kicked_atom=random_particle_index,
        iteration=iteration, steps_part1=steps_part1,
        crit_eigenvalue=crit_eigenvalue, final_energy=final_energy,
        positions=positions.copy() if success else None,
        angles=angles, elapsed=time.time() - t0,
    )

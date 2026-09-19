# -*- coding: utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from scipy.spatial import cKDTree
import time
from scipy.sparse.linalg import eigsh,ArpackError, lobpcg
from datetime import datetime
import os
import plot_diagnostics

#%%
first_move_modifier=0.1
eigenvalue_cutoff = -0.2
move_phase1_modifier=0.1
move_phase2_modifier = 0.001
#force_cutoff_saddlepoint = 1.5
force_on_atom_cutoff_saddlepoint = 0.08
dot_product_saddle_cutoff = 0.00025
relaxation_step_magnitude=0.0005
max_relaxation_steps = 50



mass = {1: 2.0, 2: 1.0}  # mass(1)=2.0, mass(2)=1.0
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
rc = 3 # LJ cutoff
move_phase2_modifier_original = move_phase2_modifier
#_range = 4  #2.5 nearest neighbors cutoff hessian
#range2 = _range ** 2
#ZERO_TOL = 1e-15 # 0 in Hessian

def d_V(r, a, b):

    if r <= rc and r > 0.0:
        s = sigma_table[(a, b)]
        eps = epsilon_table[(a, b)]

        return -4.0 * eps * (12.0 * (s / r) ** 12 / r - 6.0 * (s / r) ** 6 / r)
    else:
        return 0.0

#@njit()
def dd_V(r, a, b):

    if r <= rc and r > 0.0:
        s = sigma_table[(a, b)]
        eps = epsilon_table[(a, b)]
        return 4.0 * eps * (12.0 * 13.0 * (s / r) ** 12 / r ** 2 - 6.0 * 7.0 * (s / r) ** 6 / r ** 2)
    else:
        return 0.0

def create_dump_file():
    os.makedirs("dump", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("dump", f"trajectory_{timestamp}.dump")
    filename2 = os.path.join("plots", f"trajectory_{timestamp}.txt")

    # create/clear the file
    open(filename, "w").close()

    return filename, filename2
def write_lammps_frame(filename, timestep, positions, ids, types, box):
    """
    Append a frame to a LAMMPS dump file.

    filename : output dump file
    timestep : current iteration
    positions: (N,3) numpy array
    ids      : array of atom ids
    types    : array of atom types
    box      : (lx, ly, lz)
    """

    lx, ly, lz = box
    N = len(ids)

    with open(filename, "a") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")

        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{N}\n")

        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"0 {lx}\n")
        f.write(f"0 {ly}\n")
        f.write(f"0 {lz}\n")

        f.write("ITEM: ATOMS id type x y z\n")

        for i in range(N):
            x, y, z = positions[i]
            f.write(f"{ids[i]} {types[i]} {x} {y} {z}\n")
def parse_lammps_data(path):


    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]

    # --- find number of atoms ---
    natoms = None
    for ln in lines:
        if ln.endswith("atoms"):
            natoms = int(ln.split()[0])
            break
    if natoms is None:
        raise RuntimeError("Could not find 'atoms' line")

    # --- find box bounds ---
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

    # --- find the "Atoms" section ---
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
        # format: id type x y z ix iy iz
        parts = ln.split()
        if len(parts) < 5:
            raise RuntimeError(f"Atom line has too few columns: {ln}")

        id_arr[i] = int(parts[0])
        cid_arr[i] = int(parts[1])
        x_arr[i] = float(parts[2])
        y_arr[i] = float(parts[3])
        z_arr[i] = float(parts[4])

    return {
        "natoms": natoms,
        "id": id_arr,
        "cid": cid_arr,
        "x": x_arr,
        "y": y_arr,
        "z": z_arr,
        "lx": lx,
        "ly": ly,
        "lz": lz,
    }
'''
def perpendicular_forces(forces, moves):
    denom = np.sum(moves * moves, axis=1, keepdims=True)

    return forces - moves * np.divide(
        np.sum(forces * moves, axis=1, keepdims=True),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0
    )
'''

def perpendicular_forces(forces, moves):
    f = forces.reshape(-1)
    m = moves.reshape(-1)
    denom = np.dot(m, m)
    if denom > 0:
        return (f - (np.dot(f, m) / denom) * m).reshape(forces.shape)
    return forces.copy()

def force(positions, data, neighbors,
          eps_matrix=epsilon_table,
          sigma_matrix=sigma_table):

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

    # scalar prefactor
    pref = 24 * eps * (
        2 * sig12 / r2**7
        - sig6 / r2**4
    )

    # vector force
    fij = pref[:, None] * dr

    # accumulate onto particle i

    np.add.at(total_force, i,  fij)
    np.add.at(total_force, j, -fij)


    return total_force
#%%
def check_new_minimum(positions_new,positions_initial):
    positions_new = relax(positions_new, data, neighbour_pairs, None)
    positions_initial = relax(positions_initial, data, neighbour_pairs, None)
    dr = positions_new - positions_initial
    dr -= box * np.round(dr / box)
    max_disp = np.max(np.linalg.norm(dr, axis=1))
    '''
    tree = cKDTree(positions_new, boxsize=box)
    neighbour_pairs = np.array(list(tree.query_pairs(1.1)))

    tree2 = cKDTree(positions_initial, boxsize=box)
    neighbour_pairs2 = np.array(list(tree2.query_pairs(1.1)))
    if neighbour_pairs != neighbour_pairs2:
        print('different neighbor lists')
       '''
    return max_disp
def relax(positions, data, neighbour_pairs, moves,eps_matrix = epsilon_table, sigma_matrix = sigma_table, maxIter=max_relaxation_steps ):
    '''relaxes the system perpendicular to moves. if moves is None, relaxes the system completely in all dimensions'''
    tol=1e-5
    delE=1
    Lambda=relaxation_step_magnitude #0.0005
    Iter=0

    maxIter=50


    if type(moves) == np.ndarray:
        '''
        energy1=compute_energy(positions, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs)
        while abs(delE)>tol and Iter<maxIter:

            Force_vector=force(positions, data, neighbour_pairs, eps_matrix, sigma_matrix)
            Forces_perpendicular = perpendicular_forces(Force_vector , moves)
            positions+= Lambda*Forces_perpendicular
            energy2=compute_energy(positions, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs)
            delE = energy1-energy2
            energy1=energy2
            Iter+=1
        return positions
        '''
        max_force = np.inf
        while max_force > tol and Iter < maxIter:

            # keep pair list current as atoms move significantly
            if Iter % 1 == 0:
                tree = cKDTree(positions % box, boxsize=box)
                neighbour_pairs = np.array(list(tree.query_pairs(3+rc)))

            Force_vector = force(positions, data, neighbour_pairs, eps_matrix, sigma_matrix)
            perp_forces = perpendicular_forces(Force_vector, moves)
            max_force    = np.max(np.linalg.norm(perp_forces, axis=1))

            pos_trial = positions + Lambda * perp_forces
            E_trial   = compute_energy(pos_trial, data['cid'], box,
                                       epsilon_table, sigma_table, neighbour_pairs)
            E_current = compute_energy(positions, data['cid'], box,
                                       epsilon_table, sigma_table, neighbour_pairs)
            #print(f'current:{E_current[0]}, trial:{E_trial[0]}')
            #print("needs fixing. Return statement incorrect, maxIter defined incorrectly")

            #return positions + Lambda*perp_forces,Lambda*perp_forces, Force_vector

            if E_trial < E_current:          # accepted: move and nudge Lambda up
                positions  = pos_trial % box
                Lambda    *= 1.05
            else:                            # rejected: undo, shrink Lambda
                Lambda    *= 0.5
                if Lambda < 1e-10:
                    print(f"  Lambda collapsed at iter {Iter}, max_F={max_force:.5f}")
                    break

            Iter += 1
        return positions
        return positions + Lambda*perp_forces,Lambda*perp_forces, Force_vector

    else:
        tol_force = 0.001          # switch to force-based convergence
        Lambda    = relaxation_step_magnitude
        max_force = np.inf

        while max_force > tol_force and Iter < 5000:

            # keep pair list current as atoms move significantly
            if Iter % 50 == 0:
                tree = cKDTree(positions % box, boxsize=box)
                neighbour_pairs = np.array(list(tree.query_pairs(rc)))

            Force_vector = force(positions, data, neighbour_pairs, eps_matrix, sigma_matrix)
            max_force    = np.max(np.linalg.norm(Force_vector, axis=1))

            pos_trial = positions + Lambda * Force_vector
            E_trial   = compute_energy(pos_trial, data['cid'], box,
                                       epsilon_table, sigma_table, neighbour_pairs)
            E_current = compute_energy(positions, data['cid'], box,
                                       epsilon_table, sigma_table, neighbour_pairs)

            if E_trial < E_current:          # accepted: move and nudge Lambda up
                positions  = pos_trial % box
                Lambda    *= 1.05
            else:                            # rejected: undo, shrink Lambda
                Lambda    *= 0.5
                if Lambda < 1e-10:
                    print(f"  Lambda collapsed at iter {Iter}, max_F={max_force:.5f}")
                    break

            Iter += 1

        return positions
#%%
def compute_energy(coords, types, box, eps_matrix, sigma_matrix, pairs):

    #start = time.time()
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
    #print(time.time()-start)
    return np.sum(energy)
    return np.sum(energy), energy

#%%
def lowest_nonzero_mode(arr):


    arr = np.asarray(arr)
    temp=arr.copy()

    if len(arr) < 4:
        raise ValueError(
            f"Need at least 4 eigenvalues, got {len(arr)}"
        )

    # remove the 3 eigenvalues closest to zero
    remaining = sorted(arr, key=lambda x: abs(x))[3:]

    for i in range(len(temp)):
        if min(remaining) ==temp[i]:
            return i






def build_hessian_fast(positions, data, neighbour_pairs):
    #start=time.time()
    natoms = data['natoms']
    cid    = data['cid']

    lx, ly, lz = data['lx'], data['ly'], data['lz']
    box = np.array([lx, ly, lz])

    rank = 3 * natoms

    # ============================================================
    # Pair indices
    # ============================================================
    i = neighbour_pairs[:, 0]
    j = neighbour_pairs[:, 1]

    npairs = len(i)

    # ============================================================
    # Geometry
    # ============================================================
    dr = positions[i] - positions[j]
    dr -= box * np.round(dr / box)

    dx = dr[:, 0]
    dy = dr[:, 1]
    dz = dr[:, 2]

    r2 = dx*dx + dy*dy + dz*dz
    r  = np.sqrt(r2)


    # ============================================================
    # Types and masses
    # ============================================================
    ti = cid[i]
    tj = cid[j]

    eps = epsilon_table[ti, tj]
    sig = sigma_table[ti, tj]

    mi = np.array([mass[int(x)] for x in ti])
    mj = np.array([mass[int(x)] for x in tj])

    diag_i  = 1.0 / mi
    diag_j  = 1.0 / mj
    offdiag = -1.0 / np.sqrt(mi * mj)

    # ============================================================
    # LJ derivatives
    # ============================================================
    sr2  = (sig * sig) / r2
    sr6  = sr2**3
    sr12 = sr6 * sr6

    dV = -4.0 * eps * (
        12.0 * sr12 / r
        - 6.0 * sr6 / r
    )

    ddV = 4.0 * eps * (
        12.0 * 13.0 * sr12 / r2
        - 6.0 * 7.0 * sr6 / r2
    )

    fn0 = dV / r
    fn1 = ddV - fn0

    invr2 = 1.0 / r2

    xx = fn1 * dx*dx * invr2 + fn0
    yy = fn1 * dy*dy * invr2 + fn0
    zz = fn1 * dz*dz * invr2 + fn0

    xy = fn1 * dx*dy * invr2
    xz = fn1 * dx*dz * invr2
    yz = fn1 * dy*dz * invr2

    # ============================================================
    # Dense diagonal accumulation
    # ============================================================
    diag_blocks = np.zeros((natoms, 3, 3))

    # Hii contributions
    np.add.at(diag_blocks[:, 0, 0], i, diag_i * xx)
    np.add.at(diag_blocks[:, 1, 1], i, diag_i * yy)
    np.add.at(diag_blocks[:, 2, 2], i, diag_i * zz)

    np.add.at(diag_blocks[:, 0, 1], i, diag_i * xy)
    np.add.at(diag_blocks[:, 1, 0], i, diag_i * xy)

    np.add.at(diag_blocks[:, 0, 2], i, diag_i * xz)
    np.add.at(diag_blocks[:, 2, 0], i, diag_i * xz)

    np.add.at(diag_blocks[:, 1, 2], i, diag_i * yz)
    np.add.at(diag_blocks[:, 2, 1], i, diag_i * yz)

    # Hjj contributions
    np.add.at(diag_blocks[:, 0, 0], j, diag_j * xx)
    np.add.at(diag_blocks[:, 1, 1], j, diag_j * yy)
    np.add.at(diag_blocks[:, 2, 2], j, diag_j * zz)

    np.add.at(diag_blocks[:, 0, 1], j, diag_j * xy)
    np.add.at(diag_blocks[:, 1, 0], j, diag_j * xy)

    np.add.at(diag_blocks[:, 0, 2], j, diag_j * xz)
    np.add.at(diag_blocks[:, 2, 0], j, diag_j * xz)

    np.add.at(diag_blocks[:, 1, 2], j, diag_j * yz)
    np.add.at(diag_blocks[:, 2, 1], j, diag_j * yz)

    # ============================================================
    # COO storage preallocation
    # ============================================================

    # 18 offdiag entries per pair
    # 9 diagonal entries per atom

    n_off = 18 * npairs
    n_diag = 9 * natoms

    rows = np.empty(n_off + n_diag, dtype=np.int32)
    cols = np.empty(n_off + n_diag, dtype=np.int32)
    vals = np.empty(n_off + n_diag, dtype=np.float64)

    ptr = 0

    # ============================================================
    # Off-diagonal blocks
    # ============================================================

    components = [
        (0,0,xx),
        (1,1,yy),
        (2,2,zz),
        (0,1,xy),
        (1,0,xy),
        (0,2,xz),
        (2,0,xz),
        (1,2,yz),
        (2,1,yz),
    ]

    for a, b, comp in components:

        # Hij
        n = npairs

        rows[ptr:ptr+n] = 3*i + a
        cols[ptr:ptr+n] = 3*j + b
        vals[ptr:ptr+n] = offdiag * comp

        ptr += n

        # Hji
        rows[ptr:ptr+n] = 3*j + a
        cols[ptr:ptr+n] = 3*i + b
        vals[ptr:ptr+n] = offdiag * comp

        ptr += n

    # ============================================================
    # Diagonal blocks
    # ============================================================

    atom_ids = np.arange(natoms)

    for a in range(3):
        for b in range(3):

            n = natoms

            rows[ptr:ptr+n] = 3*atom_ids + a
            cols[ptr:ptr+n] = 3*atom_ids + b
            vals[ptr:ptr+n] = diag_blocks[:, a, b]

            ptr += n

    # ============================================================
    # Build sparse matrix
    # ============================================================

    H = coo_matrix((vals, (rows, cols)), shape=(rank, rank))

    #H.sum_duplicates()
    #print(time.time()-start)
    diff = H - H.T
    if np.max(np.abs(diff)) > 1e-8:
        raise ValueError
    return H

def document(positions, data, iteration, energies, crit_eigenvalue, crit_eigenvalues):
    energies.append(compute_energy(positions, data['cid'], box, epsilon_table, sigma_table, neighbour_pairs))
    crit_eigenvalues.append(crit_eigenvalue)
    forces.append(np.linalg.norm(force(positions, data, neighbour_pairs)))
    iteration +=1
    if iteration%20 ==0:
        print('Iteration ',iteration,' completed.')
    if iteration%5==0:

        if dump_file:
            write_lammps_frame(
                dump_file,
                iteration,
                positions,
                data["id"],
                data["cid"],
                (data["lx"], data["ly"], data["lz"])
            )
    return energies,forces, iteration, crit_eigenvalues

def calculate_moves(positions, data, neighbour_pairs, move_phase2_modifier):
    k=15
    H = build_hessian_fast(positions, data, neighbour_pairs)
    try:
        eigenvalues, eigenvectors = eigsh(H.tocsr(), k=k, sigma=1e-6, which='LM', return_eigenvectors=True)
        if lowest_nonzero_mode(eigenvalues)>9:
            print('Warning, 10 eigenvalues not enough')
    except ArpackError:
        print('Arpack Error prevented, lowering amount of eigenvalues calculated')
        eigenvalues, eigenvectors = eigsh(H.tocsr(), k=10, sigma=1e-6, which='LM', return_eigenvectors=True)
    moves = move_phase2_modifier*eigenvectors[:,lowest_nonzero_mode(eigenvalues)]
    moves = moves.reshape(-1, 3)
    return moves, eigenvalues[lowest_nonzero_mode(eigenvalues)]

def angle_between(v1, v2):
    cos_theta = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)
#%%
condition = True
attempt = 0
pot_success=[]
curiosity = []
while attempt < 1000 and len(pot_success)<=10:
    total_start_time = time.time()
    move_phase2_modifier = move_phase2_modifier_original
    #condition = False
    attempt+=1
    dump_file, details_file=create_dump_file()
    inpath = r"/home/schwyyzer/Desktop/Master Thesis/config_small_relaxed.lammps"
    #inpath2 = r"C:\Users\Yanik\Desktop\Master Thesis\config_small_relaxed.lammps"
    inpath2 = r"C:\Users\Yanik\Desktop\Master Thesis\sample_qr_0.01.lammps\relax.0.lammps"
    print("Parsing LAMMPS file:", inpath)
    try:
        data = parse_lammps_data(inpath)
    except:
        data = parse_lammps_data(inpath2)
    N = len(data['x'])
    positions = np.empty((N, 3), dtype=np.float64)
    box=[data['lx'], data['ly'], data['lz']]
    for i in range(N):
        positions[i, 0] = data['x'][i]
        positions[i, 1] = data['y'][i]
        positions[i, 2] = data['z'][i]

    positions = positions%box
    initial_positions = positions.copy()

    # this creates an array of all pairs of particles with a distance less than a threshold
    tree = cKDTree(positions, boxsize=box)
    neighbour_pairs = np.array(list(tree.query_pairs(rc)))
    eigenvalues, eigenvectors= None, None
    energies = []
    crit_eigenvalues=[]
    forces = []
#    break
    forces.append(np.linalg.norm(force(positions, data, neighbour_pairs)))
    start=time.time()
    random_particle_index = int(np.floor(np.random.rand() * N))
    first_move_component = np.random.uniform(size=3)*2-1
    #first_move_component = np.array([-0.303493,-0.0457965,0.00647575])
    #random_particle_index=9
    '''
    first_move_component=np.array([-0.805597,0.371716,0.176326])
    random_particle_index=255
    '''
    first_move=np.zeros([data['natoms'],3])
    first_move[random_particle_index, 0:3]+=first_move_modifier * first_move_component/np.linalg.norm(first_move_component)

    '''
    ===============================================================================
    Part 1
    ===============================================================================
    '''


    positions+=first_move

    #add relax here, compare with identical first move
    positions = relax(positions, data, neighbour_pairs, first_move)
    forces.append(np.linalg.norm(force(positions, data, neighbour_pairs)))
    H = build_hessian_fast(positions, data, neighbour_pairs)
#    break
    eigenvalues, eigenvectors = eigsh(H.tocsr(), k=10, sigma=1e-3, which='LM', return_eigenvectors=True)


    moves = move_phase1_modifier*eigenvectors[:,lowest_nonzero_mode(eigenvalues)]
    moves = moves.reshape(-1, 3)
    moves_phase1 = moves.copy()

    if np.dot(moves.reshape(-1), first_move.reshape(-1)) < 0:
        moves *= -1
        #print('flip')

    iteration=0
    while eigenvalues[lowest_nonzero_mode(eigenvalues)] >eigenvalue_cutoff:
        positions+=moves
        positions = relax(positions, data, neighbour_pairs, moves)
        H = build_hessian_fast(positions, data, neighbour_pairs)
        eigenvalues = eigsh(H.tocsr(), k=10, sigma=0, which='LM', return_eigenvectors=False)


        energies,forces, iteration, crit_eigenvalues = document(positions, data, iteration, energies, eigenvalues[lowest_nonzero_mode(eigenvalues)], crit_eigenvalues)


    '''
    ===============================================================================
    Part 1 Complete!
    ===============================================================================
    '''
    #%%
    print('\n Part 1 Complete!',np.round((time.time() - start)/(iteration+1),3),'seconds per iteration \n')
    crit_eigenvalue=eigenvalues[lowest_nonzero_mode(eigenvalues)]
    #iteration =0
    steps_part1=iteration
    start= time.time()
    angles=[]
    Force_vector=force(positions, data, neighbour_pairs, epsilon_table , sigma_table)
    #np.linalg.norm(force(positions, data, neighbour_pairs))>force_cutoff_saddlepoint
    per_atom_max=1
    '''abs(np.dot(moves.reshape(-1), Force_vector.reshape(-1)))/move_phase2_modifier > dot_product_saddle_cutoff'''
    while crit_eigenvalue <0 and iteration<2000 and abs(np.dot(moves.reshape(-1), Force_vector.reshape(-1)))/move_phase2_modifier > dot_product_saddle_cutoff :
        old_move=moves
        moves, crit_eigenvalue = calculate_moves(positions, data,neighbour_pairs,move_phase2_modifier)
        #Force_vector=force(positions, data, neighbour_pairs, epsilon_table , sigma_table)


        if np.dot(moves.reshape(-1), Force_vector.reshape(-1)) > 0:
            moves *= -1
            #print('flip')
        positions+=moves
        #pre_relax=positions
        positions = relax(positions, data, neighbour_pairs, moves)


        Force_vector = force(positions, data, neighbour_pairs)  # shape (N, 3)
        #per_atom_max = np.max(np.linalg.norm(force_array, axis=1))
        energies,forces, iteration, crit_eigenvalues = document(positions, data, iteration, energies, crit_eigenvalue, crit_eigenvalues)
        angle = angle_between(old_move.reshape(-1), moves.reshape(-1))/np.pi * 180
        if angle > 170 and iteration - steps_part1 > 3:
            move_phase2_modifier*=0.5
            print('Large angle between moves, saddlepoint might be closeby.')
            curiosity.append(positions.copy())

        angles.append(angle)
    print('\n Part 2 Complete!',np.round((time.time() - start)/(iteration-steps_part1+1) ,3),'seconds per iteration \n')
    if crit_eigenvalue<0 and iteration <2000 and iteration-steps_part1>0:
        runtime = time.time() - total_start_time
        condition = False
        print('\n we might have a success \n')
        fully_relaxed_perpendicular = relax(positions.copy(), data, neighbour_pairs, moves, maxIter = 100)
        pot_success.append(dump_file)
        plot_diagnostics.save_overview(energies, forces,crit_eigenvalues, angles, dump_file, steps_part1)
        # displacement of every atom
        dr = positions - initial_positions

        box = np.array([data['lx'], data['ly'], data['lz']])

        # minimum image convention
        dr -= box * np.round(dr / box)

        displacements = np.linalg.norm(dr, axis=1)

        max_disp_atom_id = np.argmax(displacements)
        max_displacement = displacements[max_disp_atom_id]
        with open(details_file, "w") as f:

            f.write("Simulation Summary\n")
            f.write("==================\n\n")

            f.write(f"Runtime (s): {runtime:.2f}\n\n")

            f.write(f"Randomly selected atom ID: {data['id'][random_particle_index]}\n")
            f.write("Random atom initial coordinates:\n")
            f.write(
                f"({initial_positions[random_particle_index,0]:.6f}, "
                f"{initial_positions[random_particle_index,1]:.6f}, "
                f"{initial_positions[random_particle_index,2]:.6f})\n"
            )

            f.write("Random atom final coordinates:\n")
            f.write(
                f"({positions[random_particle_index,0]:.6f}, "
                f"{positions[random_particle_index,1]:.6f}, "
                f"{positions[random_particle_index,2]:.6f})\n\n"
            )

            f.write(f"Atom with largest displacement ID: {data['id'][max_disp_atom_id]}\n")

            f.write("Initial coordinates:\n")
            f.write(
                f"({initial_positions[max_disp_atom_id,0]:.6f}, "
                f"{initial_positions[max_disp_atom_id,1]:.6f}, "
                f"{initial_positions[max_disp_atom_id,2]:.6f})\n"
            )

            f.write("Final coordinates:\n")
            f.write(
                f"({positions[max_disp_atom_id,0]:.6f}, "
                f"{positions[max_disp_atom_id,1]:.6f}, "
                f"{positions[max_disp_atom_id,2]:.6f})\n"
            )

            f.write(f"\nTotal displacement: {max_displacement:.6f}\n\n")

            f.write("Simulation Parameters\n")
            f.write("---------------------\n")
            f.write(f"first_move_modifier              = {first_move_modifier}\n")
            f.write(f"eigenvalue_cutoff                = {eigenvalue_cutoff}\n")
            f.write(f"move_phase1_modifier             = {move_phase1_modifier}\n")
            f.write(f"move_phase2_modifier             = {move_phase2_modifier_original}\n")
            f.write(f'dot_product_saddle_cutoff        = {dot_product_saddle_cutoff}\n')
            f.write(f'relaxation_step_magnitude        = {relaxation_step_magnitude}\n')
            f.write(f'max_relaxation_steps             = {max_relaxation_steps}\n')
            #f.write(f"force_cutoff_saddlepoint         = {force_cutoff_saddlepoint}\n")
            #f.write(f"force_on_atom_cutoff_saddlepoint = {force_on_atom_cutoff_saddlepoint}\n")

    if iteration%5 !=0 :
        if dump_file:
            print('saving file')
            write_lammps_frame(
                dump_file,
                iteration,
                positions,
                data["id"],
                data["cid"],
                (data["lx"], data["ly"], data["lz"])
            )



#dot product eigenvector force vector is 0 to determine saddlepoint
#check dot prduct eigenvector with force vector negative to ensure correct direction




#%%

'''

start=time.time()
for i in range(100):
    eigenvalues, eigenvectors = eigsh(H.tocsr(), k=5, sigma=1e-6, which='LM', return_eigenvectors=True)
print(time.time()-start)

start=time.time()
for i in range(100):
    eigenvalues, eigenvectors = eigsh(H.tocsr(), k=5, sigma=1e-6, which='LM', return_eigenvectors=True)

print(time.time()-start)
'''
'''
#%%
all_moves=[]

for i in range(10):
    moves, crit_eigenvalue = calculate_moves(positions, data,neighbour_pairs)
    all_moves.append(moves)

all_ang=[]

for i in range(9):
    all_ang.append(angle_between(all_moves[i].reshape(-1), all_moves[i+1].reshape(-1)))


#%%


def compare_coo_matrices(A, B, tol=1e-10, verbose=True):
    """
    Compare two sparse matrices element-by-element.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Preferably COO matrices.
    tol : float
        Numerical tolerance.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    identical : bool
        True if matrices are equal within tolerance.
    """

    # convert both to COO
    A = A.tocoo()
    B = B.tocoo()

    # shape check
    if A.shape != B.shape:
        if verbose:
            print("Different shapes:")
            print("A:", A.shape)
            print("B:", B.shape)
        return False

    # combine duplicate entries
    A.sum_duplicates()
    B.sum_duplicates()

    # sort entries lexicographically
    A_order = np.lexsort((A.col, A.row))
    B_order = np.lexsort((B.col, B.row))

    A_rows = A.row[A_order]
    A_cols = A.col[A_order]
    A_vals = A.data[A_order]

    B_rows = B.row[B_order]
    B_cols = B.col[B_order]
    B_vals = B.data[B_order]

    # compare sparsity pattern
    same_pattern = (
        np.array_equal(A_rows, B_rows)
        and np.array_equal(A_cols, B_cols)
    )

    if not same_pattern:
        if verbose:
            print("Different sparsity patterns.")

            A_set = set(zip(A_rows, A_cols))
            B_set = set(zip(B_rows, B_cols))

            only_A = A_set - B_set
            only_B = B_set - A_set

            print(f"Entries only in A: {len(only_A)}")
            print(f"Entries only in B: {len(only_B)}")

            if len(only_A) > 0:
                print("First few only in A:", list(only_A)[:10])

            if len(only_B) > 0:
                print("First few only in B:", list(only_B)[:10])

        return False

    # compare values
    diff = np.abs(A_vals - B_vals)

    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    identical = np.all(diff < tol)

    if verbose:
        print("Same sparsity pattern:", same_pattern)
        print("Max abs difference:", max_diff)
        print("Mean abs difference:", mean_diff)

        if not identical:
            idx = np.argmax(diff)

            print("\nLargest discrepancy:")
            print(f"Position: ({A_rows[idx]}, {A_cols[idx]})")
            print(f"A = {A_vals[idx]}")
            print(f"B = {B_vals[idx]}")
            print(f"diff = {diff[idx]}")

    return identical


H_old=build_hessian_fast(positions, data, neighbour_pairs)
eigenvalues, v1 = eigsh(H_old.tocsr(), k=10, sigma=0, which='LM', return_eigenvectors=True)
H = build_hessian_fast(positions, data, neighbour_pairs)
a=0
for i in range(20):

    eigenvalues, v2 = eigsh(H.tocsr(), k=10, sigma=0, which='LM', return_eigenvectors=True)

    overlap = abs(np.dot(v1[:,3], v2[:,3])) / (
        np.linalg.norm(v1[:,3]) * np.linalg.norm(v2[:,3])
)
    print(overlap)
    print(abs(np.dot(v1[:,3], v2[:,3])))
    print(max(abs(v1[:,3] +v2[:,3])))

    v1 = v2




'''
#%%
from scipy.sparse.linalg import lobpcg, LinearOperator

def calculate_moves2(positions, data, neighbour_pairs, move_phase2_modifier,
                     prev_eigvecs=None):
    start=time.time()
    N   = data['natoms']
    k   = 5          # 3 zero modes + 4 non-zero with safety margin
    dim = 3 * N

    H = build_hessian_fast(positions, data, neighbour_pairs).tocsr()

    # ------------------------------------------------------------------
    # Diagonal (Jacobi) preconditioner
    # Protects zero diagonal entries that appear for the translational modes
    # ------------------------------------------------------------------
    diag          = H.diagonal().copy()
    diag[np.abs(diag) < 1e-10] = 1.0
    M_inv         = 1.0 / diag            # shape (dim,)

    def precond(x):
        # LOBPCG calls this with both 1-D (single vector) and 2-D
        # (block of vectors) inputs.  Must handle both to avoid the
        # (dim,) × (dim,1)  →  (dim, dim) broadcasting MemoryError.
        if x.ndim == 2:
            return M_inv[:, np.newaxis] * x   # (dim,1) × (dim,k) → (dim,k)
        return M_inv * x                       # (dim,)  × (dim,)  → (dim,)

    M = LinearOperator((dim, dim), matvec=precond, matmat=precond)

    # ------------------------------------------------------------------
    # Warm-start from previous iteration, or fresh orthonormal guess
    # ------------------------------------------------------------------
    if prev_eigvecs is not None and prev_eigvecs.shape == (dim, k):
        X = prev_eigvecs.copy()
    else:
        X = np.random.default_rng().standard_normal((dim, k))
        X, _ = np.linalg.qr(X)          # LOBPCG requires orthonormal columns

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    eigenvalues, eigenvectors = lobpcg(
        H, X,
        M       = M,
        largest = False,
        tol     = 0.05,     # 1e-6 is unachievable for higher modes; 1e-4 is enough for ART
        maxiter = 500,
    )

    # LOBPCG does not guarantee sorted output
    order        = np.argsort(eigenvalues)
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    mode_idx = lowest_nonzero_mode(eigenvalues)
    moves    = move_phase2_modifier * eigenvectors[:, mode_idx]
    print(time.time()-start)
    return moves.reshape(-1, 3), float(eigenvalues[mode_idx]), eigenvectors



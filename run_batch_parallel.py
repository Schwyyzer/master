import os, time, sys, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import art_driver as art

art.rc = 2.5  # user-specified floor
art.move_phase2_modifier = 0.005  # found to converge faster without changing
                                   # the eventual outcome vs the 0.001 default
# eigenvalue_cutoff left at its original -0.2: pushing it deeper made phase 1
# (fixed direction, never re-diagonalized) get stuck oscillating forever
# instead of ever reaching the deeper threshold -- confirmed empirically.

# Usage: python3 run_batch_parallel.py [n_attempts] [seed0] [n_workers]
#        LAMMPS_PATH=/path/to/your.lammps python3 run_batch_parallel.py ...
LAMMPS_PATH = os.environ.get("LAMMPS_PATH", "mdyn.lammps")
N_ATTEMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 300
N_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, os.cpu_count() - 1)


def _one_attempt(seed):
    # runs in a forked worker process; art.rc=2.5 was already set in the
    # parent before the pool was created, so the fork inherits it.
    import time as _t
    data = art.parse_lammps_data(LAMMPS_PATH)
    box = np.array([data['lx'], data['ly'], data['lz']])
    positions0 = np.stack([data['x'], data['y'], data['z']], axis=1) % box
    t0 = _t.time()
    try:
        res = art.run_attempt(data, positions0, box, seed=seed, verbose=False)
    except Exception as e:
        res = dict(success=False, reason=f"exception: {e}", seed=seed,
                    iteration=None, steps_part1=None, final_energy=None, crit_eigenvalue=None)
    res['elapsed'] = _t.time() - t0
    return res


if __name__ == "__main__":
    t_start = time.time()
    results = []
    seeds = [SEED0 + k for k in range(N_ATTEMPTS)]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_one_attempt, s): s for s in seeds}
        done = 0
        for fut in as_completed(futs):
            seed = futs[fut]
            res = fut.result()
            done += 1
            status = "SUCCESS" if res['success'] else f"fail({res.get('reason')})"
            print(f"[{done}/{N_ATTEMPTS}] seed={seed} {status}  it={res.get('iteration')} "
                  f"phase1={res.get('steps_part1')} E={res.get('final_energy')} "
                  f"eig={res.get('crit_eigenvalue')} time={res['elapsed']:.1f}s  "
                  f"total_elapsed={time.time()-t_start:.1f}s", flush=True)
            results.append(res)

            # save incrementally so a disrupted session doesn't lose finished work
            with open("batch_results.pkl", "wb") as f:
                pickle.dump(results, f)

    n_success = sum(1 for r in results if r['success'])
    print(f"\n=== batch done: {n_success}/{N_ATTEMPTS} successes, total wall time {time.time()-t_start:.1f}s ===")

    successes = [r for r in results if r['success']]
    if len(successes) >= 2:
        data0 = art.parse_lammps_data(LAMMPS_PATH)
        box = np.array([data0['lx'], data0['ly'], data0['lz']])
        positions0 = np.stack([data0['x'], data0['y'], data0['z']], axis=1) % box

        n = len(successes)
        disps = []
        for r in successes:
            dr = r['positions'] - positions0
            dr -= box * np.round(dr / box)
            dr -= dr.mean(axis=0, keepdims=True)  # remove rigid-translation zero mode
            disps.append(dr)

        rmsd = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                d = disps[a] - disps[b]
                rmsd[a, b] = np.sqrt(np.mean(np.sum(d**2, axis=1)))

        energies = np.array([r['final_energy'] for r in successes])
        n_energy_clusters = len(np.unique(np.round(energies, 4)))
        offdiag = rmsd[~np.eye(n, dtype=bool)]

        print(f"distinct final-energy clusters (rounded to 1e-4): {n_energy_clusters} out of {n} successes")
        print(f"pairwise structural RMSD (COM-removed, minimum-image): "
              f"min={offdiag.min():.4f} max={offdiag.max():.4f} mean={offdiag.mean():.4f}")
        np.savetxt("pairwise_rmsd.csv", rmsd, delimiter=",")
        print("full pairwise RMSD matrix: pairwise_rmsd.csv")
    elif len(successes) == 1:
        print("only 1 success -- not enough to compare diversity.")
    else:
        print("0 successes -- see README notes on the phase-2 mode-jump issue before re-running.")

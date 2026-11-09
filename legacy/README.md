# Legacy code

This folder holds superseded Julia drivers/cores and Python tooling that are
no longer part of the active ART pipeline, kept around for reference rather
than deleted outright.

The active pipeline is `run_parallel_fortran.jl` (+ `art_core_fortran.jl`),
matching the Fortran reference `art_woconewer.f90`. It includes
`Version1.1.jl`/`config.jl`/`neighbor_pairs.jl`/`Hessian.jl`/
`particle_movement.jl`/`lammps_and_document.jl` at the repo root, plus
`experiment_settings.jl` for its tunable constants.

Everything in this folder predates that design and uses an older
phase1/phase2 (destabilization-push / Newton-step) ART formulation with
mass-weighted Hessians (`config.jl`'s own `mass = [2.0, 1.0]`, not overridden
to unit mass the way the active driver does it):

- **`run_parallel.jl` -> `art_core.jl` -> `mode_tracking.jl`** -- the
  original phase1/phase2 driver with Lanczos mode tracking.
- **`run_parallel_no_tracking.jl` -> `art_core_no_tracking.jl`** -- same,
  without mode tracking (fresh lowest-mode solve every iteration).
- **`run_parallel_rqi.jl` -> `art_core_rqi.jl` -> `rqi_tracking.jl`** --
  same, with Rayleigh quotient iteration instead of Lanczos.
- **`eigenvalueDecomp.jl`** -- older, buggy `track_mode`/`eig_from_csv_lobpcg`,
  superseded by `mode_tracking.jl`/`rqi_tracking.jl`.
- **`test_mode_tracking.jl`, `test_mode_tracking_precond.jl`,
  `test_rqi_tracking.jl`** -- standalone test/diagnostic scripts for the
  above.
- **`art_driver.py`** -- the Python reference port used to validate the
  Julia implementation (`validate_saddles.py` still references it in
  comments for that history).
- **`diagnose_mode_diversity.py`, `new_start_1_1_1.py`,
  `run_batch_parallel.py`** -- older Python analysis/driver scripts for the
  phase1/phase2 pipeline. (`new_start_1_1_1.py` imports a `plot_diagnostics`
  module that doesn't exist anywhere in this repo -- it was already broken
  before this move, not something that broke in moving it.)
- **`mdyn.lammps`** -- an old LAMMPS input file associated with the above.

These files' `include()`/import paths were updated to account for the move
(mostly adding `../` for references to files that stayed at the repo root),
but their logic is untouched. If you need the old phase1/phase2 behavior for
comparison, it should still run as before from this folder.

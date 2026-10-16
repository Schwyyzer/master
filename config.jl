first_move_modifier=0.1
eigenvalue_cutoff = -0.2
move_phase1_modifier=0.1
move_phase2_modifier = 0.001
#force_cutoff_saddlepoint = 1.5
force_on_atom_cutoff_saddlepoint = 0.08
dot_product_saddle_cutoff = 0.00025

#=================================================================#
#Relaxation parameters

relaxation_step_magnitude=0.0005
max_relaxation_steps = 50
relaxation_step_magnitude_multiplier_success = 1.05
relaxation_step_magnitude_multiplier_failure = 0.5
#inpath2 = "C:\\Users\\Yanik\\Desktop\\Master Thesis\\config_intermediate.lammps"
inpath2 = "//home//schwyyzer//Desktop//Master Thesis//config_small_relaxed.lammps"
#csvfile = "C:\\Users\\Yanik\\Desktop\\Master Thesis\\results_eigenvectors(in).csv"
csvfile = nothing

# `mass` is intentionally NOT `const` -- run_parallel_fortran.jl overrides it
# after this file loads (see that file's header). A plain untyped global
# forces every function that reads it (force, build_hessian_fast, ...) into
# dynamically-typed, boxed lookups on every single access -- a real cost
# inside the hottest loops in the program. Pinning the TYPE (not the value)
# with this annotation keeps those reads fast while still allowing the
# later reassignment.
global mass::Vector{Float64} = [2.0, 1.0]

# These three are never reassigned anywhere else in the codebase, so they
# can be `const` outright -- the strongest, simplest fix for the same
# boxed-global-read problem described above.
const epsilon_table = [1.0 1.0; 1.0 1.0]
const sigma_table = [1.0 11/12; 11/12 5/6]
const rc = 6.0 # LJ cutoff

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
#
# Guarded (like SimData in lammps_and_document.jl): Julia only allows a
# global's TYPE to be declared once per process. config.jl gets included
# again on any worker that already ran it in a previous invocation of this
# script within the same Julia session (e.g. a prior run that was
# interrupted before reaching its own rmprocs(workers()) cleanup, leaving
# the worker processes -- and their already-typed `mass` -- alive for the
# next run to reuse). Re-running `global mass::Vector{Float64} = ...` on
# such a worker errors ("cannot set type for global Main.mass"); plain
# reassignment of an already-typed global is always fine, so that's the
# fallback.
if isdefined(@__MODULE__, :mass)
    global mass = Float64[2.0, 1.0]
else
    global mass::Vector{Float64} = [2.0, 1.0]
end

# These three are never reassigned anywhere else in the codebase, so they
# can be `const` outright -- the strongest, simplest fix for the same
# boxed-global-read problem described above. Same re-include guard as
# `mass` above -- redeclaring an existing `const` (even to an equal value)
# is unreliable across Julia versions, from a warning to a hard error.
if !isdefined(@__MODULE__, :epsilon_table)
    const epsilon_table = [1.0 1.0; 1.0 1.0]
end
if !isdefined(@__MODULE__, :sigma_table)
    const sigma_table = [1.0 11/12; 11/12 5/6]
end
if !isdefined(@__MODULE__, :rc)
    const rc = 6.0 # LJ cutoff
end

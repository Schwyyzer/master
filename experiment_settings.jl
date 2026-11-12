#=
Tunable settings for run_parallel_fortran.jl, split out into their own file
so they're easy to find and sweep without scrolling through the driver
logic. Edit the values below; everything else in run_parallel_fortran.jl
stays the same.

Defaults here reflect the most recent values this project was actually run
with (not necessarily the originally-committed ones) -- if you're mid-sweep
on something, these should already match what you had.
=#

LAMMPS_PATH = "//home//schwyyzer//Desktop//Master Thesis//config_small_relaxed.lammps"
TARGET_SUCCESSES = 1000
# Opt-in: set to an Int to run a FIXED BUDGET of attempts instead of running
# until TARGET_SUCCESSES successes are found. TARGET_SUCCESSES is ignored
# when this is set (not a race between the two -- this one wins outright).
# Useful for a clean, unbiased success-RATE measurement (e.g. comparing
# TRUST_REGION_EPS settings): targeting successes means a worse-performing
# setting runs more attempts to reach the same count, so wall-clock time and
# n_attempts both confound the comparison; targeting a fixed attempt count
# does not.
TARGET_ATTEMPTS = 1000
N_WORKERS = max(1, Sys.CPU_THREADS - 1)   # leave a core free for the OS/master
MAX_ITERATIONS = 500                      # matches art_woconewer.f90's max_number_of_art_iterations

# When true: no per-attempt CSV log (and no flush() after every attempt --
# a real cost over thousands of failed attempts), no per-success dump
# files, no per-attempt trajectory files, no progress printlns, no
# diversity/RMSD report. Instead, at the
# very end of the run, print ONLY the config variables that were actually
# used, the starting position, and the saddle point positions found --
# nothing else. Use this when you just want the result, not the
# documentation of how the run got there.
MINIMAL_OUTPUT = false

# run_art_attempt_fortran's own keyword defaults (art_core_fortran.jl),
# duplicated here as named settings -- rather than left implicit -- so
# that (a) MINIMAL_OUTPUT's end-of-run summary has something concrete to
# print as "the config variables that were used", and (b) they can be
# tuned from this one place without touching art_core_fortran.jl. Values
# unchanged from those defaults, so behavior is identical either way.
EIGENVALUE_THRESHOLD = 3.0
DELTA_X_MAX = 0.125
# 0.1, not the reference's original 0.025 -- validated in Python on an
# 80-seed, 500-atom sample (not yet at Julia scale, which is what this
# setting is here for you to check): raising the destabilization-phase
# push from 0.025 to 0.1 took the non-degenerate-abort success rate from
# 37/80 (46%) to 53/80 (66%). That's a net +16, but not a free lunch --
# 20 previously-stuck seeds started reaching the Newton-step branch, while
# 4 previously-fine seeds started degenerating instead (likely overshoot:
# a push this much bigger isn't uniformly an improvement, just a net one
# here). Most degenerate-mode failures never reach the Newton-step branch
# at all under the ORIGINAL 0.025 (81% of them, in that same sample) --
# that's the structural reason this lever matters much more than
# TRUST_REGION_EPS below, which only ever acts after that branch is
# reached. Set back to 0.025 for an A/B comparison against the original.
DELTA_LM = 0.1
FORCE_TOLERANCE = 1e-3
ZERO_EIGENVALUE_THRESHOLD = 0.1
MAX_ZERO_EIGENVALUE_THRESHOLD_NUMBER = 10
MAX_CONSECUTIVE_SECOND_EIG_NEGATIVE = 5
KICK_MAGNITUDE = 0.18
STARTING_FIRE_ITERATIONS = 5
LANCZOS_ITERATIONS = 10

# Trust-region-adaptive cap on the Newton-step branch's step size, in
# place of the flat DELTA_X_MAX clamp above (see art_core_fortran.jl's
# header for what this is). Set to `nothing` for now -- turned off so the
# DELTA_LM test above is an isolated, clean comparison against the 37/80
# vs 53/80 Python result, not conflated with a second lever at the same
# time. Once success meant "converged && is_saddle" this mattered (6/22
# -> 7/22, one rescued near-miss); now that success matches the
# reference's own looser criterion, that rescue effect is mostly moot --
# and it structurally can't touch most of what DELTA_LM above fixes,
# since 81% of degenerate failures never even reach the branch this
# clamp operates on. Re-enable (e.g. 0.1) if you want to layer it back on
# top once DELTA_LM is validated at scale.
TRUST_REGION_EPS = nothing

# :single (default) is the plain Fortran-reference kick -- one random atom,
# random direction, magnitude KICK_MAGNITUDE. :shear is a new, opt-in
# alternative (phase 1 of a planned generalization, see
# art_core_fortran.jl's run_art_attempt_fortran docstring): the atom
# nearest the kicked one also gets displaced by KICK_MAGNITUDE, in the
# OPPOSITE direction -- a local two-atom shear/dipole instead of a
# single-atom monopole kick. results.csv records which atom that was
# (shear_partner_atom column; blank under :single).
KICK_MODE = :single

# false (default, unchanged behavior): during the destabilization/"phase 1"
# branch (before the tracked eigenvalue first drops below
# -EIGENVALUE_THRESHOLD), both the FIRE relaxation and the DELTA_LM push
# use the FRESHLY re-tracked eigenvector every iteration -- only the push
# MAGNITUDE is fixed, not its direction. Set true to additionally freeze
# the DIRECTION to whatever it was on the first iteration (the kick
# direction, refined once) for as long as the search stays in that
# branch -- a straight-line push instead of a re-aimed one, testing
# whether re-tracking the direction during phase 1 is actually buying
# anything over fixing it too, now that the step size is already fixed.
# Guards and the branch decision itself are unaffected either way (see
# art_core_fortran.jl's run_art_attempt_fortran docstring).
FREEZE_PUSH_DIRECTION = false

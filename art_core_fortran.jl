#=
art_core_fortran.jl

ART implementation matching the working Fortran reference art_woconewer.f90
(kick/neighbor-list/FIRE/Newton-step machinery, unchanged), with two
validated additions from "Adaptative step magnitude for ART" (hugo, Sept
2024): a second-eigenvalue validity guard and an MEP-proximity diagnostic.
See run_art_attempt_fortran's docstring for what those are and how they
were validated; this header covers what's unchanged from the plain
Fortran port (see git history for the original, longer rationale against
art_core_no_tracking.jl's phase1/phase2 design):

  - NO MASS-WEIGHTING ANYWHERE. The driver sets the global `mass` to
    [1.0, 1.0] before calling build_hessian_fast (see run_parallel_fortran.jl)
    so that function reduces to a plain, non-mass-weighted Hessian.
  - NO initial relaxation before the first Hessian/eigenvector calc.
  - Kick: a single atom (uniform over 1:natoms), magnitude 0.15, direction
    = normalize(uniform cube sample in [-0.5,0.5]^3).
  - A STATIC neighbor list (skin = rc+1.0), built once from the starting
    minimum and reused for the entire run, relying on
    force/compute_energy/build_hessian_fast to filter every pair to r<=rc
    internally regardless of what candidate list they are handed.
  - Lowest-eigenvalue/eigenvector TRACKING via a fixed-length (10-step),
    warm-started Lanczos run every iteration, not a from-scratch
    k-eigenvalue reselection.
  - ONE unified loop, not phase1/phase2: FIRE-relax perpendicular to the
    tracked eigenvector, rebuild the Hessian, retrack the mode, then
    either a clamped Newton step (eigenvalue < -eigenvalue_threshold) or
    a small fixed push (otherwise).
  - FIRE relaxation (restricted perpendicular to the tracked eigenvector),
    not adaptive steepest descent -- see relax_positions_fire.
  - Convergence is FORCE-MAGNITUDE ONLY (max|F| < force_tolerance).
  - A degenerate-mode abort guard: if |eigenvalue| stays below
    zero_eigenvalue_threshold for more than max_zero_eigenvalue_threshold_number
    consecutive iterations (likely a translational mode), the attempt is
    discarded with no result.

NOT adopted from the trust-region paper: its g/alpha step (a single
combined relaxation+climbing direction with a trust-region-derived step
size), tried as a wholesale replacement for the FIRE+Newton-step loop
above. Empirically, on the dense 500-atom test system used to validate
this file, it failed in two ways depending on configuration: with no
perpendicular relaxation at all, a raw kick routinely left 2+ unstable
directions immediately (e.g. one seed: eigenvalue0=-622, eigenvalue1=-552
right after the kick, before any step), which the method has no mechanism
to resolve; adding FIRE settling back in fixed that, but then the paper's
own g-direction -- which scales its climbing component by <h,q0>, the
current force's alignment with the tracked mode -- was observed to
collapse toward zero as FIRE relaxation proceeded (the alignment faded
rather than concentrated), so the climbing drive self-extinguished and
searches fizzled back toward stability instead of crossing a saddle. The
old Fortran algorithm never had this failure mode because its
destabilization push (delta_lm * q0) is unconditional -- fixed magnitude,
independent of current alignment -- so it keeps making progress regardless.
See git history on this file for the full trust-region derivation and the
empirical dead ends, kept there rather than here since they didn't end up
shaping the algorithm actually used.

PARTIALLY adopted since then, as an opt-in: `trust_region_eps` (default
`nothing`, meaning "off, behave exactly as above") replaces the Newton-
step branch's flat `delta_x_max` clamp with a curvature-adaptive one,
`min(2*eps*|h|/|lowest_eval|, delta_x_max)` -- the paper's Eq 5,
specialized to a pure step along q0 (where g.Hg = q0.H.q0 = lowest_eval
exactly, no approximation needed). This is NOT the g/alpha scheme above:
the step DIRECTION is still exactly q0 as before, only the cap on its
MAGNITUDE becomes curvature-aware instead of a flat number. Validated in
Python (scratchpad's art_trust_step_only.py) on the same 22-seed, 500-atom
sample this file's docstrings reference: eps=0.1 found one additional
genuine saddle (seed 7, which previously ran to max_iterations without
converging) with zero regressions elsewhere (6/22 -> 7/22), while
eps=0.05/0.2/0.3 reproduced the flat-clamp baseline's outcome exactly
(same successes, though not always identical intermediate eigenvalues --
the cap only ever binds tighter than delta_x_max, never looser, so eps
values where it rarely binds are indistinguishable from `nothing`). Traced
to a single iteration where the eps=0.1 cap came out ~6% tighter than
0.125, which was enough to redirect that seed's chaotic trajectory onto a
convergent path -- a real, reproducible, mechanistically-understood
effect, but based on one seed changing out of 22, not yet validated at
scale. See run_parallel_fortran.jl's TRUST_REGION_EPS setting to use this.

This is meant to be included AFTER Version1.1.jl (force / compute_energy /
perpendicular_forces / config.jl / neighbor_pairs.jl / Hessian.jl /
particle_movement.jl), same as art_core_no_tracking.jl.
=#

using LinearAlgebra
using SparseArrays
using Random

# Guarded (see config.jl's `mass`/`rc`/etc. for the full explanation):
# redefining a struct type -- which this file's own include-chain does on
# any worker that already ran it in an earlier, still-alive invocation of
# this script within the same Julia session -- is a hard error in Julia,
# unlike redefining a function or plain variable.
if !isdefined(@__MODULE__, :FortranAttemptResult)
    Base.@kwdef struct FortranAttemptResult
        success::Bool
        seed::Int
        kicked_atom::Int
        kick_dir::Vector{Float64}
        initial_energy::Float64
        final_energy::Float64 = NaN
        crit_eigenvalue::Float64 = NaN
        second_eigenvalue::Float64 = NaN
        force_magnitude::Float64 = NaN
        mep_proximity::Float64 = NaN
        iteration::Int = 0
        converged::Bool = false
        is_saddle::Bool = false
        degenerate_mode_aborted::Bool = false
        second_eigenvalue_aborted::Bool = false
        positions::Union{Matrix{Float64},Nothing} = nothing
        dump_file::Union{String,Nothing} = nothing
        trajectory_file::Union{String,Nothing} = nothing
        elapsed_time::Float64 = NaN
        error::Union{String,Nothing} = nothing
    end
end

# Flat (3*natoms, atom-major x,y,z) <-> natoms x 3 matrix, matching the
# `reshape(v, 3, :)'` convention already used elsewhere in this codebase
# (particle_movement.jl's calculate_moves, art_core_no_tracking.jl).
flat_to_matrix(v::AbstractVector) = permutedims(reshape(v, 3, :))
matrix_to_flat(M::AbstractMatrix) = vec(permutedims(M))

"""
    write_trajectory_csv(path, step_norms)

One row per outer iteration: `step_norm` is the magnitude of that
iteration's push along the tracked eigenvector (`delta_x` in the Newton
branch, clamped either by `delta_x_max` or the trust-region cap; `delta_lm`
in the destabilization branch) -- i.e. the norm of the actual displacement
applied to the full 3N-dimensional configuration that iteration, not just
a per-atom quantity. This equals the push scalar directly (not a separate
norm() computation) because `evec` is always unit-norm (renormalized at
the end of `lanczos_lowest_eigenvalues`), so `delta_x .* evec_matrix` (or
`delta_lm .* evec_matrix`) as a flattened vector has Euclidean norm exactly
`abs(delta_x)` (or `delta_lm`).
"""
function write_trajectory_csv(path, step_norms)
    open(path, "w") do io
        println(io, "iteration,step_norm")
        for (i, s) in enumerate(step_norms)
            println(io, "$i,$s")
        end
    end
end

"""
    lanczos_lowest_eigenvalues(H, v0; n_iter=10)

Fixed-length Lanczos tridiagonalization of symmetric `H`, warm-started from
`v0` (need not be unit norm), matching art_woconewer.f90's calc_lowest_eval
when eval_tolerance<0 (always runs exactly `n_iter` steps -- no
early-convergence check; ql_start_size = ql_max_iteration = 10 in the
reference). Returns `(lowest_eval, second_eval, evec)`: the two most
negative Ritz values found among the `n_iter` Ritz values of the resulting
tridiagonal matrix, and the unit eigenvector corresponding to the lowest,
reconstructed from the Lanczos basis.

Neither is necessarily H's true global minimum/second eigenvalue, but
whichever the Krylov subspace built from `v0` captures -- that's the point
for the lowest one: started from a good previous estimate, this
reconverges to the SAME physical mode almost every iteration, giving
robust mode continuity that a from-scratch reselection lacks. For the
second value, this relies on the trust-region paper's own empirical claim
that a small, non-expanded Krylov run already approximates it well enough
to use as a validity check (see run_art_attempt_fortran) -- not
independently verified here beyond confirming it doesn't cause regressions
(see that function's docstring for how it was validated).
"""
function lanczos_lowest_eigenvalues(H, v0::Vector{Float64}; n_iter::Int=10)
    n = length(v0)
    Q = zeros(Float64, n, n_iter)
    alpha = zeros(Float64, n_iter)
    beta  = zeros(Float64, n_iter)

    beta_start = norm(v0)
    beta_start = beta_start > 0 ? beta_start : 1.0
    q_prev = zeros(Float64, n)
    r = copy(v0)

    for k in 1:n_iter
        qk = k == 1 ? r ./ beta_start : r ./ beta[k-1]
        Q[:, k] = qk

        u = H * qk

        new_r = k == 1 ? u .- beta_start .* q_prev : u .- beta[k-1] .* Q[:, k-1]
        alpha[k] = dot(qk, new_r)
        new_r = new_r .- alpha[k] .* qk
        bk = norm(new_r)
        # guard against exact Krylov-subspace breakdown (v0 already lies in
        # a low-dimensional invariant subspace of H); not present in the
        # reference but needed since Julia doesn't silently continue on a
        # division that would otherwise be by exactly zero
        beta[k] = bk > 1e-300 ? bk : 1e-300
        r = new_r
    end

    T = SymTridiagonal(alpha, beta[1:end-1])
    decomp = eigen(T)
    order = sortperm(decomp.values)
    lowest_eval = decomp.values[order[1]]
    second_eval = decomp.values[order[2]]
    z = decomp.vectors[:, order[1]]

    evec = Q * z
    evec ./= norm(evec)

    return lowest_eval, second_eval, evec
end

"""
    relax_positions_fire(positions, data, neighbors, restriction_vector, n_iterations, box)

Restricted FIRE relaxation, matching art_woconewer.f90's relax_positions
subroutine: velocity-Verlet position/velocity updates with a fixed
per-step "mass" constant (NOT atom-mass-dependent -- Fortran uses
delta_t^2/(2*1.0365e-28) uniformly), forces kept perpendicular to
`restriction_vector` (an natoms x 3 matrix, assumed unit norm as a flat
vector) throughout, and FIRE's adaptive-timestep/mixing schedule. All FIRE
state (velocity, timestep, alpha) is local to this call, matching
Fortran's locals in `relax_positions` -- nothing persists between calls.
Returns the relaxed `positions` after `n_iterations` steps.
"""
function relax_positions_fire(positions, data, neighbors, restriction_vector, n_iterations, box)
    natoms = size(positions, 1)

    delta_t = 0.01e-15
    n_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.01
    alpha = alpha_start
    f_alpha = 0.99
    delta_t_max = 10.0 * delta_t
    mass_const = 1.0365e-28
    nsteps_since_p_neg = 0

    # Preallocated once per call and reused for every one of the
    # n_iterations substeps below (via force!/perpendicular_forces!)
    # instead of each substep allocating its own fresh arrays -- this
    # loop is the hottest one in the whole program (n_iterations grows
    # over the course of an attempt, and this runs every outer ART
    # iteration), so a fresh zeros(natoms,3)/positions-copy on every
    # substep was a lot of avoidable GC pressure.
    velocities = zeros(natoms, 3)
    forces = zeros(natoms, 3)
    last_forces = zeros(natoms, 3)

    force!(forces, positions, data, neighbors)
    perpendicular_forces!(forces, restriction_vector)

    # `positions` is mutated in place from here on (see below) -- safe
    # because this function's one call site (run_art_attempt_fortran)
    # always passes a freshly-allocated array that nothing else holds a
    # reference to, and immediately rebinds its own `positions` to this
    # function's return value anyway.
    for _ in 1:n_iterations
        h_parameter = delta_t^2 / (2.0 * mass_const)

        @. positions += velocities * delta_t + forces * h_parameter
        last_forces .= forces
        force!(forces, positions, data, neighbors)
        perpendicular_forces!(forces, restriction_vector)

        @. velocities += (last_forces + forces) * (h_parameter / delta_t)

        p = dot(forces, velocities)
        v_norm = norm(velocities)
        @. velocities = (1.0 - alpha) * velocities + alpha * forces * v_norm

        if p > 0
            nsteps_since_p_neg += 1
            if nsteps_since_p_neg > n_min
                delta_t = min(delta_t * f_inc, delta_t_max)
                alpha *= f_alpha
            end
        else
            nsteps_since_p_neg = 0
            delta_t *= f_dec
            velocities .= 0.0
            alpha = alpha_start
        end
    end

    return positions
end

"""
    run_art_attempt_fortran(seed, positions0, data, pairs_static, box; kwargs...)

One full ART attempt matching art_woconewer.f90's unified FIRE+Newton-step
search loop exactly, plus two additions from "Adaptative step magnitude
for ART" (hugo, Sept 2024):

  - A second-eigenvalue validity guard (that paper's section 1.1): if the
    Lanczos run's SECOND-lowest Ritz value stays negative for more than
    `max_consecutive_second_eig_negative` CONSECUTIVE iterations, the
    attempt is discarded (2+ sustained unstable directions mean the
    eventual saddle may not be adjacent to the starting minimum). The
    paper's own description aborts on the FIRST negative reading; that was
    tried and tested against a known-good seed (one that reaches a
    genuine, adjacent saddle under the plain Fortran algorithm) and wrongly
    rejected it over a transient dip -- this FIRE-based search evidently
    passes through brief, recoverable multi-instability excursions that a
    literal instant-abort doesn't tolerate. Requiring several consecutive
    negative readings (mirroring how the pre-existing zero-eigenvalue guard
    already requires `max_zero_eigenvalue_threshold_number` consecutive
    iterations, not one) was validated to reproduce the plain algorithm's
    behavior exactly, seed for seed, iteration count for iteration count,
    final eigenvalue for final eigenvalue, across a 22-seed batch on the
    dense 500-atom test system (6/22 successes either way, identical
    per-seed outcomes) -- i.e. this guard no longer changes behavior at
    all on that sample; it's a dormant safety net for cases genuinely
    worth discarding, not validated to ever have actually fired correctly
    yet.
  - mep_proximity = <q0,h>/|h| is tracked (its running minimum returned,
    not enforced) -- the paper monitors this as a diagnostic, staying
    close to 1 if the search is following a minimum energy path, without
    prescribing a hard cutoff.

The paper's OWN step-size/direction scheme (a single trust-region-sized
step along g = h - (1+gamma)<h,q0>q0, replacing FIRE and the
destabilization/Newton split entirely) was tried and NOT adopted -- see
this file's header comment for why it failed on this system, and git
history for the fuller derivation.

`pairs_static` must be a STATIC neighbor list built once (skin = rc+1.0)
from the starting minimum and shared across every attempt (see
run_parallel_fortran.jl); it is never rebuilt here.

Keyword defaults: eigenvalue_threshold=3.0, delta_x_max=0.125,
delta_lm=0.025, force_tolerance=1e-3, zero_eigenvalue_threshold=0.1,
max_zero_eigenvalue_threshold_number=10, kick_magnitude=0.15,
starting_fire_iterations=5, lanczos_iterations=10, max_iterations=200 are
the reference's own constants verbatim, unchanged from the plain Fortran
port. max_consecutive_second_eig_negative=5 is new and unvalidated beyond
"doesn't regress the known-good sample" -- there's no reference value for
it since the paper's own description (abort on the first negative reading)
is what the validation showed to be too strict here.

`elapsed_time` (wall-clock seconds for this ONE attempt, from function
entry to return) and, if `trajectory_file` is given, a per-iteration
step-norm CSV (see `write_trajectory_csv`) are recorded on every non-crash
return (success or either guard abort) -- added for comparing the trust-
region-adaptive Newton-step clamp against the flat `delta_x_max` one: both
computational cost (elapsed_time) and the actual step-size behavior over
the course of a run (the trajectory) are otherwise invisible in the
single-row-per-attempt results.csv.

`success` matches art_woconewer.f90's own output-writing criterion
exactly: true whenever the attempt reaches the end of the loop (by force
convergence OR by hitting max_iterations) WITHOUT being discarded by the
degenerate-mode or second-eigenvalue guard. The reference writes every
non-degenerate attempt's configuration to its own numbered output file
regardless of whether force_magnitude actually dropped below
force_tolerance and regardless of whether the final point has negative
curvature -- it does not bake a stricter "is this actually a converged
saddle" check into the search itself, leaving that to separate validation
(this project's own validate_saddles.jl/.py). `converged` and `is_saddle`
are still computed and returned so a caller can filter on them, but
neither gates `success` or whether `positions`/`dump_file` get populated
-- every non-aborted attempt returns both now. (An earlier version of
this file required `converged && is_saddle` for `success`; that was a
stricter criterion we'd added ourselves, not something the reference
does -- see git history.)
"""
function run_art_attempt_fortran(
    seed::Int,
    positions0::Matrix{Float64},
    data,
    pairs_static::Matrix{Int},
    box;
    dump_file::Union{String,Nothing} = nothing,
    trajectory_file::Union{String,Nothing} = nothing,
    max_iterations::Int = 200,
    eigenvalue_threshold::Float64 = 3.0,
    delta_x_max::Float64 = 0.125,
    delta_lm::Float64 = 0.025,
    force_tolerance::Float64 = 1e-3,
    zero_eigenvalue_threshold::Float64 = 0.1,
    max_zero_eigenvalue_threshold_number::Int = 10,
    max_consecutive_second_eig_negative::Int = 5,
    kick_magnitude::Float64 = 0.15,
    starting_fire_iterations::Int = 5,
    lanczos_iterations::Int = 10,
    trust_region_eps::Union{Float64,Nothing} = nothing,
)
    t_start = time()   # whole-attempt wall-clock; read at every return below

    rng = MersenneTwister(seed)
    natoms = data["natoms"]

    initial_energy = compute_energy(positions0, data["cid"], box, epsilon_table, sigma_table, pairs_static)

    kicked_atom = rand(rng, 1:natoms)
    kick_dir = rand(rng, 3) .- 0.5
    kick_dir ./= norm(kick_dir)

    iteration = 0

    try
        evec = zeros(3 * natoms)
        evec[3*(kicked_atom-1)+1] = kick_dir[1]
        evec[3*(kicked_atom-1)+2] = kick_dir[2]
        evec[3*(kicked_atom-1)+3] = kick_dir[3]

        positions = positions0 .+ kick_magnitude .* flat_to_matrix(evec)

        H = build_hessian_fast(positions, data, pairs_static)
        lowest_eval, second_eval, evec = lanczos_lowest_eigenvalues(H, evec; n_iter=lanczos_iterations)

        n_fire = starting_fire_iterations
        zero_eig_count = 0
        second_eig_neg_count = 0
        converged = false
        force_mag = Inf   # must predate the loop: read in the final return after it ends
        mep_proximity_min = Inf
        current_forces = zeros(natoms, 3)   # reused every iteration via force!
        step_norms = Float64[]   # one entry per push actually applied: length(step_norms)
                                   # == iteration on the success return (that iteration's
                                   # push happens before the convergence check below), but
                                   # == iteration-1 on a guard-abort return (the aborting
                                   # iteration detects the guard BEFORE ever deciding on/
                                   # applying a push) -- written to trajectory_file at
                                   # whichever return below fires

        while !converged
            iteration += 1

            positions = relax_positions_fire(positions, data, pairs_static, flat_to_matrix(evec), n_fire, box)

            H = build_hessian_fast(positions, data, pairs_static)
            lowest_eval, second_eval, evec = lanczos_lowest_eigenvalues(H, evec; n_iter=lanczos_iterations)

            force!(current_forces, positions, data, pairs_static)
            h_flat = matrix_to_flat(current_forces)
            h_norm = norm(h_flat)

            h_dot_q0 = dot(h_flat, evec)
            mep_proximity = h_norm > 0 ? abs(h_dot_q0) / h_norm : 1.0
            mep_proximity_min = min(mep_proximity_min, mep_proximity)

            if h_dot_q0 > 0.0
                evec = -evec
            end

            force_mag = maximum(abs.(current_forces))

            if abs(lowest_eval) < zero_eigenvalue_threshold
                zero_eig_count += 1
                if zero_eig_count > max_zero_eigenvalue_threshold_number
                    energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, pairs_static)
                    if trajectory_file !== nothing
                        write_trajectory_csv(trajectory_file, step_norms)
                    end
                    return FortranAttemptResult(;
                        success=false, seed, kicked_atom, kick_dir, initial_energy,
                        final_energy=energy, crit_eigenvalue=lowest_eval, second_eigenvalue=second_eval,
                        force_magnitude=force_mag, mep_proximity=mep_proximity_min,
                        iteration, degenerate_mode_aborted=true,
                        trajectory_file, elapsed_time=time() - t_start,
                        error="possible translational mode, aborted (|eigenvalue|<$(zero_eigenvalue_threshold) " *
                              "for >$(max_zero_eigenvalue_threshold_number) consecutive iterations)",
                    )
                end
            else
                zero_eig_count = 0
            end

            if second_eval < 0
                second_eig_neg_count += 1
                if second_eig_neg_count > max_consecutive_second_eig_negative
                    energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, pairs_static)
                    if trajectory_file !== nothing
                        write_trajectory_csv(trajectory_file, step_norms)
                    end
                    return FortranAttemptResult(;
                        success=false, seed, kicked_atom, kick_dir, initial_energy,
                        final_energy=energy, crit_eigenvalue=lowest_eval, second_eigenvalue=second_eval,
                        force_magnitude=force_mag, mep_proximity=mep_proximity_min,
                        iteration, second_eigenvalue_aborted=true,
                        trajectory_file, elapsed_time=time() - t_start,
                        error="second-lowest eigenvalue stayed negative for >$(max_consecutive_second_eig_negative) " *
                              "consecutive iterations (currently $(second_eval)) -- sustained 2+ unstable " *
                              "directions, the eventual saddle may not be adjacent to the starting minimum",
                    )
                end
            else
                second_eig_neg_count = 0
            end

            evec_matrix = flat_to_matrix(evec)
            if lowest_eval < -eigenvalue_threshold
                delta_x = dot(h_flat, evec) / lowest_eval

                if trust_region_eps === nothing
                    # matches the reference literally: clamps the MAGNITUDE to
                    # delta_x_max without preserving delta_x's sign. Harmless in
                    # practice -- the sign-alignment above guarantees
                    # dot(F,evec)<=0 and lowest_eval<0 here, so delta_x is
                    # always >=0 already, and the clamp is a no-op on sign.
                    if abs(delta_x) > delta_x_max
                        delta_x = delta_x_max
                    else
                        n_fire += 1
                    end
                else
                    # Trust-region-adaptive cap (see this file's header for
                    # what this is and how it was validated). Only ever
                    # tighter than delta_x_max, never looser, so this is a
                    # pure restriction of the original behavior, not a
                    # wholesale change to it.
                    cap = min(2.0 * trust_region_eps * h_norm / abs(lowest_eval), delta_x_max)
                    if abs(delta_x) > cap
                        delta_x = delta_x > 0 ? cap : -cap
                    else
                        n_fire += 1
                    end
                end
                push!(step_norms, abs(delta_x))
                positions = positions .+ delta_x .* evec_matrix
            else
                n_fire = starting_fire_iterations
                push!(step_norms, delta_lm)
                positions = positions .+ delta_lm .* evec_matrix
            end

            if force_mag < force_tolerance
                converged = true
            end
            if iteration > max_iterations
                break
            end
        end

        energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, pairs_static)
        is_saddle = lowest_eval < 0
        # Reaching here means neither guard above fired -- matches the
        # reference's own "write output unless degenerate-aborted"
        # criterion exactly (see this function's docstring).
        success = true

        if dump_file !== nothing
            open(dump_file, "w") do io end
            write_lammps_frame(
                dump_file, iteration, positions, data["id"], data["cid"],
                (data["lx"], data["ly"], data["lz"]), flat_to_matrix(evec)
            )
        end
        if trajectory_file !== nothing
            write_trajectory_csv(trajectory_file, step_norms)
        end

        return FortranAttemptResult(;
            success, seed, kicked_atom, kick_dir, initial_energy,
            final_energy=energy, crit_eigenvalue=lowest_eval, second_eigenvalue=second_eval,
            force_magnitude=force_mag, mep_proximity=mep_proximity_min,
            iteration, converged, is_saddle,
            positions,
            dump_file,
            trajectory_file, elapsed_time=time() - t_start,
        )

    catch e
        return FortranAttemptResult(;
            success=false, seed, kicked_atom, kick_dir, initial_energy,
            iteration, elapsed_time=time() - t_start,
            error="[iteration=$(iteration)] " * sprint(showerror, e),
        )
    end
end

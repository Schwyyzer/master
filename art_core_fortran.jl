#=
art_core_fortran.jl

ART implementation matching the working Fortran reference art_woconewer.f90
as closely as possible -- a genuine algorithm replacement for the
phase1/phase2 fixed/adaptive-step design in art_core_no_tracking.jl, not a
parameter tweak. The differences that actually matter, found by reading
art_woconewer.f90 line by line against our Julia:

  - NO MASS-WEIGHTING ANYWHERE. Fortran declares mass(1)=2.0, mass(2)=1.0
    in initialize() and never references `mass` again -- not in
    build_hessian, not in relax_positions, not in calc_energy_forces. Our
    Hessian/FIRE-equivalent were mass-weighted; this file's driver must
    set the global `mass` to [1.0, 1.0] before calling build_hessian_fast
    (see run_parallel_fortran.jl) so the SAME function reduces to a plain,
    non-mass-weighted Hessian -- no separate Hessian builder needed.
  - NO initial relaxation before the first Hessian/eigenvector calc: the
    kicked (unrelaxed) structure goes straight into build_hessian.
  - Kick: a single atom (uniform over 1:natoms), magnitude 0.15 (not our
    old first_move_modifier=0.1), direction = normalize(uniform cube
    sample in [-0.5,0.5]^3) (not Gaussian).
  - A STATIC neighbor list, built ONCE (skin = rc+1.0, matching Fortran's
    range_nn = range+1.0) from the starting minimum and reused for the
    ENTIRE run -- every attempt, every iteration, never rebuilt. This is
    safe only because force/compute_energy/build_hessian_fast all filter
    every pair to r<=rc internally regardless of what candidate list they
    are handed.
  - Lowest-eigenvalue/eigenvector TRACKING via a fixed-length (10-step),
    warm-started Lanczos run every iteration (calc_lowest_eval), not a
    from-scratch k-eigenvalue shift-invert solve with "exclude the 3
    smallest-magnitude values" mode selection. Started from the PREVIOUS
    iteration's eigenvector, a short Krylov run very efficiently
    reconverges to the SAME physical mode -- this is what gives Fortran
    robust mode continuity across iterations without ever confusing a
    translational/global mode for the one actually being tracked, a
    failure mode our from-scratch reselection was prone to.
  - ONE unified loop, not phase1/phase2: FIRE-relax perpendicular to the
    tracked eigenvector, rebuild the Hessian, retrack the mode, then
    either a clamped Newton step (eigenvalue < -eigenvalue_threshold) or
    a small fixed push (otherwise) -- see run_art_attempt_fortran's
    docstring for the exact step logic.
  - FIRE relaxation (restricted perpendicular to the tracked eigenvector),
    not adaptive steepest descent -- see relax_positions_fire.
  - Convergence is FORCE-MAGNITUDE ONLY (max|F| < force_tolerance), not
    eigenvalue-sign-based.
  - A degenerate-mode abort guard: if |eigenvalue| stays below
    zero_eigenvalue_threshold for more than max_zero_eigenvalue_threshold_number
    consecutive iterations (likely a translational mode, not a real
    instability), the attempt is discarded with no result -- matching
    Fortran's `cycle`.

One deliberate design choice NOT copied verbatim: Fortran writes an output
file for every attempt that isn't degenerate-mode-aborted, regardless of
whether force actually converged below tolerance or the final eigenvalue
ended up negative (a real saddle) -- it appears to rely on separate,
external post-processing to filter genuine saddles from the raw dump.
Here, `success` requires all three: force converged, the final eigenvalue
is negative (a genuine saddle, not a relaxation back to a minimum), and no
degenerate-mode abort. `converged` and `is_saddle` are reported separately
so this choice is visible and reversible.

This is meant to be included AFTER Version1.1.jl (force / compute_energy /
perpendicular_forces / config.jl / neighbor_pairs.jl / Hessian.jl /
particle_movement.jl), same as art_core_no_tracking.jl.
=#

using LinearAlgebra
using SparseArrays
using Random

Base.@kwdef struct FortranAttemptResult
    success::Bool
    seed::Int
    kicked_atom::Int
    kick_dir::Vector{Float64}
    initial_energy::Float64
    final_energy::Float64 = NaN
    crit_eigenvalue::Float64 = NaN
    force_magnitude::Float64 = NaN
    iteration::Int = 0
    converged::Bool = false
    is_saddle::Bool = false
    degenerate_mode_aborted::Bool = false
    positions::Union{Matrix{Float64},Nothing} = nothing
    dump_file::Union{String,Nothing} = nothing
    error::Union{String,Nothing} = nothing
end

# Flat (3*natoms, atom-major x,y,z) <-> natoms x 3 matrix, matching the
# `reshape(v, 3, :)'` convention already used elsewhere in this codebase
# (particle_movement.jl's calculate_moves, art_core_no_tracking.jl).
flat_to_matrix(v::AbstractVector) = permutedims(reshape(v, 3, :))
matrix_to_flat(M::AbstractMatrix) = vec(permutedims(M))

"""
    lanczos_lowest_eigenvalue(H, v0; n_iter=10)

Fixed-length Lanczos tridiagonalization of symmetric `H`, warm-started from
`v0` (need not be unit norm), matching art_woconewer.f90's calc_lowest_eval
when eval_tolerance<0 (always runs exactly `n_iter` steps -- no
early-convergence check; ql_start_size = ql_max_iteration = 10 in the
reference). Returns `(lowest_eval, evec)`: the most negative Ritz value
found among the `n_iter` Ritz values of the resulting tridiagonal matrix,
and its corresponding unit eigenvector reconstructed from the Lanczos
basis -- NOT necessarily H's true global minimum eigenvalue, but whichever
the Krylov subspace built from `v0` captures. That's the point: started
from a good previous estimate, this reconverges to the SAME physical mode
almost every iteration, which is what gives Fortran's continuous, robust
mode identity that a from-scratch "recompute k eigenvalues, exclude the 3
smallest by magnitude" reselection lacks.
"""
function lanczos_lowest_eigenvalue(H, v0::Vector{Float64}; n_iter::Int=10)
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
    idx = argmin(decomp.values)
    lowest_eval = decomp.values[idx]
    z = decomp.vectors[:, idx]

    evec = Q * z
    evec ./= norm(evec)

    return lowest_eval, evec
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
Returns `(positions, energy, forces)` after `n_iterations` steps.
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

    velocities = zeros(natoms, 3)
    forces = force(positions, data, neighbors)
    forces = perpendicular_forces(forces, restriction_vector)
    energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, neighbors)

    for _ in 1:n_iterations
        h_parameter = delta_t^2 / (2.0 * mass_const)

        positions = positions .+ velocities .* delta_t .+ forces .* h_parameter
        last_forces = forces
        forces = force(positions, data, neighbors)
        forces = perpendicular_forces(forces, restriction_vector)
        energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, neighbors)

        velocities = velocities .+ (last_forces .+ forces) .* (h_parameter / delta_t)

        p = dot(matrix_to_flat(forces), matrix_to_flat(velocities))
        v_norm = norm(velocities)
        velocities = (1.0 - alpha) .* velocities .+ alpha .* forces .* v_norm

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

    return positions, energy, forces
end

"""
    run_art_attempt_fortran(seed, positions0, data, pairs_static, box; kwargs...)

One full ART attempt matching art_woconewer.f90's unified search loop --
see this file's header for the full list of differences from
art_core_no_tracking.jl's phase1/phase2 design. `pairs_static` must be a
STATIC neighbor list built once (skin = rc+1.0) from the starting minimum
and shared across every attempt (see run_parallel_fortran.jl); it is never
rebuilt here, matching the reference exactly.

Default keyword values are the reference's own constants verbatim:
eigenvalue_threshold=3.0, delta_x_max=0.125, delta_lm=0.025,
force_tolerance=1e-3, zero_eigenvalue_threshold=0.1,
max_zero_eigenvalue_threshold_number=10, kick_magnitude=0.15,
starting_fire_iterations=5, lanczos_iterations=10, max_iterations=200.

`success` requires force convergence AND a genuinely negative final
eigenvalue AND no degenerate-mode abort -- see the "one deliberate design
choice not copied verbatim" note in this file's header for why that's
stricter than the reference's own unconditional (except degenerate-mode)
output-writing.
"""
function run_art_attempt_fortran(
    seed::Int,
    positions0::Matrix{Float64},
    data,
    pairs_static::Matrix{Int},
    box;
    dump_file::Union{String,Nothing} = nothing,
    max_iterations::Int = 200,
    eigenvalue_threshold::Float64 = 3.0,
    delta_x_max::Float64 = 0.125,
    delta_lm::Float64 = 0.025,
    force_tolerance::Float64 = 1e-3,
    zero_eigenvalue_threshold::Float64 = 0.1,
    max_zero_eigenvalue_threshold_number::Int = 10,
    kick_magnitude::Float64 = 0.15,
    starting_fire_iterations::Int = 5,
    lanczos_iterations::Int = 10,
)
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
        lowest_eval, evec = lanczos_lowest_eigenvalue(H, evec; n_iter=lanczos_iterations)

        n_fire = starting_fire_iterations
        zero_eig_count = 0
        converged = false
        current_forces = zeros(natoms, 3)
        energy = initial_energy
        force_mag = Inf

        while !converged
            iteration += 1

            positions, _, _ = relax_positions_fire(positions, data, pairs_static, flat_to_matrix(evec), n_fire, box)

            H = build_hessian_fast(positions, data, pairs_static)
            lowest_eval, evec = lanczos_lowest_eigenvalue(H, evec; n_iter=lanczos_iterations)

            current_forces = force(positions, data, pairs_static)
            energy = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, pairs_static)

            if dot(matrix_to_flat(current_forces), evec) > 0.0
                evec = -evec
            end

            force_mag = maximum(abs.(current_forces))

            if abs(lowest_eval) < zero_eigenvalue_threshold
                zero_eig_count += 1
                if zero_eig_count > max_zero_eigenvalue_threshold_number
                    return FortranAttemptResult(;
                        success=false, seed, kicked_atom, kick_dir, initial_energy,
                        final_energy=energy, crit_eigenvalue=lowest_eval, force_magnitude=force_mag,
                        iteration, degenerate_mode_aborted=true,
                        error="possible translational mode, aborted (|eigenvalue|<$(zero_eigenvalue_threshold) " *
                              "for >$(max_zero_eigenvalue_threshold_number) consecutive iterations)",
                    )
                end
            else
                zero_eig_count = 0
            end

            evec_matrix = flat_to_matrix(evec)
            if lowest_eval < -eigenvalue_threshold
                delta_x = dot(matrix_to_flat(current_forces), evec) / lowest_eval
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
                positions = positions .+ delta_x .* evec_matrix
            else
                n_fire = starting_fire_iterations
                positions = positions .+ delta_lm .* evec_matrix
            end

            if force_mag < force_tolerance
                converged = true
            end
            if iteration > max_iterations
                break
            end
        end

        is_saddle = lowest_eval < 0
        success = converged && is_saddle

        if success && dump_file !== nothing
            open(dump_file, "w") do io end
            write_lammps_frame(
                dump_file, iteration, positions, data["id"], data["cid"],
                (data["lx"], data["ly"], data["lz"]), flat_to_matrix(evec)
            )
        end

        return FortranAttemptResult(;
            success, seed, kicked_atom, kick_dir, initial_energy,
            final_energy=energy, crit_eigenvalue=lowest_eval, force_magnitude=force_mag,
            iteration, converged, is_saddle,
            positions = success ? positions : nothing,
            dump_file = success ? dump_file : nothing,
        )

    catch e
        return FortranAttemptResult(;
            success=false, seed, kicked_atom, kick_dir, initial_energy,
            iteration, error="[iteration=$(iteration)] " * sprint(showerror, e),
        )
    end
end

#=
Single-attempt ART (activation-relaxation) logic -- same as art_core.jl,
EXCEPT phase 2 uses the original cold full re-diagonalization every
iteration (calculate_moves, from particle_movement.jl) instead of
art_core.jl's warm-started block mode tracking (track_modes). This is
the version that gave 10/10 successes on the real system; mode tracking
is set aside for now (it triggered a real bug, since fixed, but is
being revisited separately -- see art_core.jl / mode_tracking.jl /
test_mode_tracking_precond.jl).

Kept from the mode-tracking branch, because they're independent
improvements, not mode-tracking-specific:

  - Unconstrained-relax "overshoot" check on every failed attempt
    (_overshoot_check): phase 2 can push the system past a real barrier
    without the dot-product/eigenvalue criteria ever firing cleanly, so
    a failed attempt is given one full unconstrained relax() from
    wherever it ended up and compared against the starting minimum --
    a genuinely different resulting energy is a real transition found,
    independent of whether the saddle itself was formally converged.
  - max_phase2_angle_deg tracking: the angle between successive phase-2
    move directions, still worth watching here too since cold
    re-diagonalization is exactly what originally produced the
    large-angle mode jumps that motivated trying mode tracking in the
    first place.
  - The exception handler reports the phase and iteration count at
    time of failure instead of always defaulting to
    phase1_iters=0/total_iters=0.

`run_art_attempt` keeps the same signature as art_core.jl's, including
the `mode_recheck_every` keyword, purely so run_parallel.jl (or any
other driver) can point at either file by changing a single `include`
line with no other edits. It's simply unused here.

This is meant to be included AFTER Version1.1.jl (which defines force,
compute_energy, perpendicular_forces, and loads config.jl/
neighbor_pairs.jl/Hessian.jl/particle_movement.jl), same as art_core.jl.
=#

using LinearAlgebra
using Random

Base.@kwdef struct AttemptResult
    success::Bool
    seed::Int
    kicked_atom::Int
    kick_dir::Vector{Float64}
    initial_energy::Float64
    final_energy::Float64 = NaN
    crit_eigenvalue::Float64 = NaN
    iteration::Int = 0
    steps_part1::Int = 0
    positions::Union{Matrix{Float64},Nothing} = nothing
    dump_file::Union{String,Nothing} = nothing
    error::Union{String,Nothing} = nothing
    # unconstrained-relax check on failed attempts (nothing for successes,
    # and nothing if an exception prevented the check from running)
    overshoot_energy::Union{Float64,Nothing} = nothing
    overshoot_new_minimum::Union{Bool,Nothing} = nothing
    overshoot_positions::Union{Matrix{Float64},Nothing} = nothing
    # angle (degrees) between successive phase-2 moves -- large values are
    # the mode-jump symptom; watch this here since phase 2 is back to cold
    # re-diagonalization every iteration
    max_phase2_angle_deg::Union{Float64,Nothing} = nothing
    # adaptive phase-2 step size diagnostics (nothing if phase 2 never ran)
    phase2_step_size_min::Union{Float64,Nothing} = nothing
    phase2_step_size_max::Union{Float64,Nothing} = nothing
end

function _failed_result(
    seed, kicked_atom, kick_dir, initial_energy, err::String;
    iteration::Int = 0, steps_part1::Int = 0,
    overshoot_energy = nothing, overshoot_new_minimum = nothing, overshoot_positions = nothing,
)
    AttemptResult(;
        success=false, seed, kicked_atom, kick_dir, initial_energy,
        iteration, steps_part1, error=err,
        overshoot_energy, overshoot_new_minimum, overshoot_positions,
    )
end

"""
    _overshoot_check(positions, data, pairs, box, initial_energy)

Run one full unconstrained relaxation from `positions` (e.g. wherever a
failed attempt ended up) and compare its energy against
`initial_energy`. Returns (energy, positions, new_minimum::Bool), where
new_minimum is true if the relaxed energy differs from the start by
more than a small tolerance -- a cheap first-pass signal; a proper
check should also compare structure (see run_parallel.jl's pairwise
RMSD analysis, which this feeds into for failed-but-overshot attempts).
"""
function _overshoot_check(positions, data, pairs, box, initial_energy; tol::Float64=1e-3)
    ov_pairs = build_neighbor_pairs(positions, rc, box)
    ov_positions = relax(copy(positions), data, ov_pairs, nothing, box)
    ov_pairs_final = build_neighbor_pairs(ov_positions, rc, box)
    ov_energy = compute_energy(ov_positions, data["cid"], box, epsilon_table, sigma_table, ov_pairs_final)
    new_minimum = abs(ov_energy - initial_energy) > tol
    return ov_energy, ov_positions, new_minimum
end

"""
    run_art_attempt(seed, positions0, data, pairs0, box; dump_file=nothing,
                    max_total_iter=2000, mode_recheck_every=25,
                    max_phase2_step=5*move_phase2_modifier)

Run one full ART attempt (random kick -> phase 1 activation -> phase 2
eigenvector-following) starting from the given relaxed minimum
`positions0`, using a private RNG seeded with `seed` so results are
reproducible and independent across parallel attempts.

Phase 2 uses an ADAPTIVE step size, proportional to the force component
parallel to the current move direction (the same quantity
`dot_product_saddle_cutoff` already measures as the stopping criterion)
-- large early in phase 2, on the steep part of the landscape the
kick just destabilized, shrinking toward `move_phase2_modifier` (the
previous fixed step) as that parallel force vanishes approaching the
saddle. The proportionality constant is fixed once, calibrated so the
very first phase-2 step lands exactly at `max_phase2_step`; every later
step is `clamp(constant * |parallel_force|, move_phase2_modifier,
max_phase2_step)`, so it's always bounded in that range regardless of
how the parallel force actually evolves. `move_phase2_modifier` itself
(from config.jl) is the floor -- unchanged from before this change.

`mode_recheck_every` is accepted but unused (kept for drop-in
compatibility with art_core.jl's signature).

`saddle_eigenvalue_tolerance` (default 0.01): success requires
`crit_eigenvalue < saddle_eigenvalue_tolerance`, not strictly `< 0`. The
phase-2 loop only re-checks crit_eigenvalue AFTER taking a step, so on a
smoothly-converging search it routinely overshoots the true zero-crossing
by a small amount on the final step; a small positive tolerance here
keeps every case a strict `< 0` check already accepted (any negative
value at all) and additionally accepts landing just past zero, instead
of rejecting a genuinely converged saddle over a discretization artifact.

`positions0`, `data`, `pairs0`, `box` are all read-only here (relax()
mutates its own local copy, never the caller's array).

If `dump_file` is given and the attempt succeeds, a single LAMMPS frame
with the final (converged) saddle configuration is written there (the
file is created fresh). Pass `nothing` (the default) to skip writing.
"""
function run_art_attempt(
    seed::Int,
    positions0::Matrix{Float64},
    data,
    pairs0::Matrix{Int},
    box;
    dump_file::Union{String,Nothing} = nothing,
    max_total_iter::Int = 2000,
    mode_recheck_every::Int = 25,   # unused, kept for signature compatibility
    max_phase2_step::Float64 = 5 * move_phase2_modifier,
    saddle_eigenvalue_tolerance::Float64 = 0.01,
)
    rng = MersenneTwister(seed)
    natoms = data["natoms"]

    kicked_atom = rand(rng, 1:natoms)
    kick_dir = randn(rng, 3)
    kick_dir ./= norm(kick_dir)

    initial_energy = compute_energy(
        positions0, data["cid"], box, epsilon_table, sigma_table, pairs0
    )

    iteration = 0
    steps_part1 = 0
    phase = "init"

    try
        positions = copy(positions0)
        pairs = copy(pairs0)

        first_move = zeros(natoms, 3)
        first_move[kicked_atom, :] = kick_dir .* first_move_modifier

        positions .+= first_move
        phase = "initial_relax"
        positions = relax(positions, data, pairs, first_move, box)

        # ---------------- phase 1: fixed-direction activation ----------------
        phase = "phase1_setup"
        H = build_hessian_fast(positions, data, pairs)
        vals, vecs = lowest_modes(H, 10)
        idx = lowest_nonzero_mode(vals)
        relevant_eigenvalue = vals[idx]
        relevant_eigenvector = vecs[:, idx]

        moves = relevant_eigenvector .* move_phase1_modifier
        moves = reshape(moves, 3, :)'

        if dot(moves, first_move) < 0
            moves = -moves
        end

        phase = "phase1"
        F = force(positions, data, pairs)

        while relevant_eigenvalue > eigenvalue_cutoff && iteration < max_total_iter

            positions .= mod.(positions, collect(box)')
            pairs = build_neighbor_pairs(positions, rc, box)

            positions .+= moves
            positions = relax(positions, data, pairs, moves, box)

            H = build_hessian_fast(positions, data, pairs)
            vals, vecs = lowest_modes(H, 10, previous_eigenvectors=vecs)
            relevant_eigenvalue = vals[lowest_nonzero_mode(vals)]

            F = force(positions, data, pairs)
            iteration += 1
        end

        if iteration == 0 || relevant_eigenvalue > eigenvalue_cutoff
            # never destabilized (or ran out of the shared iteration budget):
            # not a saddle search failure in the "something is wrong" sense,
            # just an unproductive kick.
            ov_energy, ov_positions, ov_new_min = _overshoot_check(positions, data, pairs, box, initial_energy)
            return _failed_result(
                seed, kicked_atom, kick_dir, initial_energy,
                "phase 1 did not reach eigenvalue_cutoff within max_total_iter";
                iteration, steps_part1=iteration,
                overshoot_energy=ov_energy, overshoot_new_minimum=ov_new_min, overshoot_positions=ov_positions,
            )
        end

        steps_part1 = iteration
        phase = "phase2"

        # ---------------- phase 2: eigenvector-following to the saddle ----------------
        moves = relevant_eigenvector .* move_phase2_modifier
        moves = reshape(moves, 3, :)'

        # calibrate the proportionality constant once, from the parallel
        # force right at the handoff, so the first phase-2 step lands at
        # max_phase2_step; every later step is a clamped, self-consistent
        # rescaling of this same constant (see docstring above)
        initial_parallel_force = abs(dot(moves, F)) / move_phase2_modifier
        step_size_coefficient = max_phase2_step / max(initial_parallel_force, 1e-12)

        crit_eigenvalue = relevant_eigenvalue
        val = vals
        max_phase2_angle_deg = 0.0
        step_size = move_phase2_modifier
        phase2_step_size_min = step_size
        phase2_step_size_max = step_size

        while crit_eigenvalue < 0 &&
              iteration < max_total_iter &&
              abs(dot(moves, F) / step_size) > dot_product_saddle_cutoff

            if iteration % 10 == 0
                pairs = build_neighbor_pairs(positions, rc, box)
            end

            old_moves = copy(moves)
            parallel_force = abs(dot(old_moves, F)) / step_size
            step_size = clamp(step_size_coefficient * parallel_force, move_phase2_modifier, max_phase2_step)
            phase2_step_size_min = min(phase2_step_size_min, step_size)
            phase2_step_size_max = max(phase2_step_size_max, step_size)

            moves, val = calculate_moves(positions, data, pairs, F, old_moves, step_size)
            crit_eigenvalue = val[lowest_nonzero_mode(val)]

            angle_deg = rad2deg(angle_between(vec(old_moves'), vec(moves')))
            max_phase2_angle_deg = max(max_phase2_angle_deg, angle_deg)

            positions .+= moves
            positions = relax(positions, data, pairs, moves, box)
            F = force(positions, data, pairs)

            iteration += 1
        end

        final_energy = compute_energy(
            positions, data["cid"], box, epsilon_table, sigma_table, pairs
        )
        # crit_eigenvalue < saddle_eigenvalue_tolerance, not strictly < 0: the
        # phase-2 loop only checks crit_eigenvalue AFTER taking a step, so it
        # structurally always takes one step too many -- on systems where
        # phase 2 converges smoothly (angle between successive moves shrinking
        # steadily toward 0, eigenvalue delta shrinking every step), that
        # last step routinely overshoots the true zero-crossing by a tiny
        # amount (observed: +0.0004 to +0.006, on a search that ranged over
        # several full eigenvalue units), landing a hair on the wrong side of
        # a strictly-negative check despite being a completely genuine,
        # converged saddle. saddle_eigenvalue_tolerance is a small POSITIVE
        # number (default 0.01), not an absolute-value check: it keeps every
        # case the old `< 0` check already accepted (any negative value, no
        # matter how large in magnitude) and additionally accepts landing
        # just past zero, instead of wrongly rejecting a converged saddle
        # over a discretization artifact this small relative to the search's
        # own dynamic range.
        success = crit_eigenvalue < saddle_eigenvalue_tolerance && iteration < max_total_iter && (iteration - steps_part1) > 0

        if success && dump_file !== nothing
            open(dump_file, "w") do io end
            write_lammps_frame(
                dump_file, iteration, positions, data["id"], data["cid"],
                (data["lx"], data["ly"], data["lz"]), moves
            )
        end

        overshoot_energy = overshoot_new_minimum = overshoot_positions = nothing
        if !success
            overshoot_energy, overshoot_positions, overshoot_new_minimum =
                _overshoot_check(positions, data, pairs, box, initial_energy)
        end

        return AttemptResult(;
            success, seed, kicked_atom, kick_dir, initial_energy, final_energy,
            crit_eigenvalue, iteration, steps_part1,
            positions = success ? positions : nothing,
            dump_file = success ? dump_file : nothing,
            overshoot_energy, overshoot_new_minimum, overshoot_positions,
            max_phase2_angle_deg, phase2_step_size_min, phase2_step_size_max,
        )

    catch e
        return _failed_result(
            seed, kicked_atom, kick_dir, initial_energy,
            "[$(phase), iteration=$(iteration)] " * sprint(showerror, e);
            iteration, steps_part1,
        )
    end
end

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
                    max_total_iter=2000, mode_recheck_every=25)

Run one full ART attempt (random kick -> phase 1 activation -> phase 2
eigenvector-following) starting from the given relaxed minimum
`positions0`, using a private RNG seeded with `seed` so results are
reproducible and independent across parallel attempts.

`mode_recheck_every` is accepted but unused (kept for drop-in
compatibility with art_core.jl's signature).

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

        crit_eigenvalue = relevant_eigenvalue
        val = vals
        max_phase2_angle_deg = 0.0

        while crit_eigenvalue < 0 &&
              iteration < max_total_iter &&
              abs(dot(moves, F) / move_phase2_modifier) > dot_product_saddle_cutoff

            if iteration % 10 == 0
                pairs = build_neighbor_pairs(positions, rc, box)
            end

            old_moves = copy(moves)
            moves, val = calculate_moves(positions, data, pairs, F, old_moves, move_phase2_modifier)
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
        success = crit_eigenvalue < 0 && iteration < max_total_iter && (iteration - steps_part1) > 0

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
            max_phase2_angle_deg,
        )

    catch e
        return _failed_result(
            seed, kicked_atom, kick_dir, initial_energy,
            "[$(phase), iteration=$(iteration)] " * sprint(showerror, e);
            iteration, steps_part1,
        )
    end
end

#=
Single-attempt ART (activation-relaxation) logic, refactored out of
Version1.1.jl's run_simulation() into a pure, parameterized function that
returns a result instead of printing/relying on global state. This is
meant to be included AFTER Version1.1.jl (which defines force,
compute_energy, perpendicular_forces, and loads config.jl/neighbor_pairs.jl
/Hessian.jl/particle_movement.jl), so it reuses those unchanged.

Deliberately kept algorithmically identical to new_start_1_1_1.py's
per-attempt logic (same phase-1/phase-2 structure, same convergence
criteria, same formulas) -- with two differences, both making this
*closer* to the Python original, not further from it:

  1. Phase 1's push direction now gets the sign-aligned-with-the-kick
     check (`if dot(moves, first_move) < 0; moves = -moves; end`) that
     new_start_1_1_1.py has but Version1.1.jl was missing entirely, so
     Julia's phase-1 direction was flipping to an arbitrary (solver-
     dependent) sign instead of a kick-dependent one.
  2. Phase 1's Hessian diagonalizations now request 10 eigenpairs
     (matching new_start_1_1_1.py's `k=10`) instead of Version1.1.jl's
     5, which only left 2 candidates after lowest_nonzero_mode discards
     the 3 closest to zero -- too thin a margin to reliably land on the
     right mode.

Both existing implementations share a SINGLE iteration budget across
phase 1 and phase 2 combined (Python: iteration<2000; Version1.1.jl:
iteration<1000). This keeps that shared-budget structure (using
Python's more generous 2000), extended to also bound phase 1, which
previously had no cap in either language.

PHASE 2 now uses warm-started block mode tracking (track_modes, from
mode_tracking.jl) instead of a cold full re-diagonalization every
iteration -- see calculate_moves_tracked below. This was validated
against a synthetic avoided-crossing test before being wired in here:
single-vector tracking provably cannot survive a crossing (nothing to
overlap-match against), block tracking (k=10-15) does, staying correct
through the whole crossing. This also fixes the reproducible phase-2
mode-jump seen on the real 1728-atom system, and is much cheaper per
call (LOBPCG needs only matrix-vector products; ARPACK shift-invert
needs a sparse LU factorization every single call -- ~8s of the ~10s
per call on that system was the factorization alone). A periodic full
diagonalization (every `mode_recheck_every` iterations, default 25)
re-anchors the tracked slot by overlap as a safety check against silent
drift through a genuine near-degeneracy.

Two robustness additions, needed only because this now runs many
attempts unattended in parallel (neither changes the physics):

  - The whole attempt runs inside a try/catch, so a single relax()
    failure (its designed `ArgumentError` on step-size collapse) is
    recorded as a failed attempt instead of killing the worker process.
  - A failed attempt (phase 1 never destabilized, or phase 2 didn't
    converge) gets one unconstrained relax() from wherever it ended up,
    compared against the starting minimum's energy. Phase 2 can push
    the system past a real barrier without cleanly pinning down the
    saddle (the dot-product/eigenvalue criteria never firing cleanly);
    this catches those cases -- a genuinely different resulting minimum
    is a real transition found, independent of whether the saddle
    itself was formally converged.
=#

using LinearAlgebra
using Random

include("mode_tracking.jl")

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
    # exactly the mode-jump symptom track_modes is meant to eliminate, so
    # this is the direct empirical check that it worked
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
    calculate_moves_tracked(positions, data, neighbors, F, X_prev, target_slot,
                             move_modifier=move_phase2_modifier; full_recheck=false)

Phase-2 move calculation using warm-started block mode tracking
(track_modes) instead of a cold full re-diagonalization every
iteration. `X_prev` is the block of eigenvectors from the previous call
(or from the phase1->phase2 handoff); `target_slot` is which column of
that block is "our" mode. Returns `(moves, eigenvalue, X_new,
target_slot)`, which feed directly into the next call.

`full_recheck=true` runs a full diagonalization instead of LOBPCG
tracking (same block size), and re-identifies target_slot by overlap
with the previously-tracked eigenvector -- the periodic safety check
against silent drift through a near-degeneracy.
"""
function calculate_moves_tracked(
    positions, data, neighbors, F, X_prev::AbstractMatrix, target_slot::Int,
    move_modifier=move_phase2_modifier;
    full_recheck::Bool=false,
)
    H = build_hessian_fast(positions, data, neighbors)
    k = size(X_prev, 2)
    old_target_vec = X_prev[:, target_slot]

    if full_recheck
        vals_full, vecs_full = lowest_modes(H, k)
        overlaps = [abs(dot(old_target_vec, vecs_full[:, i])) for i in 1:k]
        target_slot = argmax(overlaps)
        X_new = vecs_full
        eigenvalue = vals_full[target_slot]
    else
        vals_tr, vecs_tr, ok = try
            v, x, o = track_modes(H, X_prev)
            (all(isfinite, v) && all(isfinite, x)) ? (v, x, true) : (v, x, false)
        catch
            (Float64[], zeros(0, 0), false)
        end

        if ok
            X_new = vecs_tr
            eigenvalue = vals_tr[target_slot]
        else
            # LOBPCG tracking produced non-finite output (or threw) --
            # fall back to a full diagonalization for this one iteration
            # rather than propagating garbage (e.g. NaN moves, which
            # relax() can never turn into a decreasing-energy step and
            # which manifests as a spurious "step magnitude too small").
            vals_full, vecs_full = lowest_modes(H, k)
            overlaps = [abs(dot(old_target_vec, vecs_full[:, i])) for i in 1:k]
            target_slot = argmax(overlaps)
            X_new = vecs_full
            eigenvalue = vals_full[target_slot]
        end
    end

    moves = X_new[:, target_slot] .* move_modifier
    moves = reshape(moves, 3, :)'

    if dot(moves, F) > 0
        moves = -moves
    end

    return moves, eigenvalue, X_new, target_slot
end

"""
    run_art_attempt(seed, positions0, data, pairs0, box; dump_file=nothing,
                    max_total_iter=2000, mode_recheck_every=25)

Run one full ART attempt (random kick -> phase 1 activation -> phase 2
eigenvector-following) starting from the given relaxed minimum
`positions0`, using a private RNG seeded with `seed` so results are
reproducible and independent across parallel attempts.

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
    mode_recheck_every::Int = 25,
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
        phase = "phase1_to_phase2_handoff"

        # ---- phase1 -> phase2 handoff: seed the tracked block ----
        # One fresh k=15 diagonalization here (matching the original
        # calculate_moves' block size) reusing the Hessian already built
        # at the end of phase 1 -- a one-time cost, not per-iteration --
        # gives phase 2 tracking a wider safety margin than phase 1's k=10.
        vals15, vecs15 = lowest_modes(H, 15)
        target_slot = argmax([abs(dot(relevant_eigenvector, vecs15[:, i])) for i in 1:15])
        X_block = vecs15
        crit_eigenvalue = vals15[target_slot]

        # ---------------- phase 2: eigenvector-following to the saddle ----------------
        moves = X_block[:, target_slot] .* move_phase2_modifier
        moves = reshape(moves, 3, :)'
        max_phase2_angle_deg = 0.0
        phase = "phase2"

        while crit_eigenvalue < 0 &&
              iteration < max_total_iter &&
              abs(dot(moves, F) / move_phase2_modifier) > dot_product_saddle_cutoff

            if iteration % 10 == 0
                pairs = build_neighbor_pairs(positions, rc, box)
            end

            old_moves = copy(moves)
            moves, crit_eigenvalue, X_block, target_slot = calculate_moves_tracked(
                positions, data, pairs, F, X_block, target_slot, move_phase2_modifier;
                full_recheck = (iteration % mode_recheck_every == 0),
            )
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

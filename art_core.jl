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

One more thing worth being explicit about: both existing implementations
share a SINGLE iteration budget across phase 1 and phase 2 combined
(Python: iteration<2000; Version1.1.jl: iteration<1000), not independent
per-phase budgets. This keeps that shared-budget structure (using
Python's more generous 2000), just extends it to also bound phase 1,
which previously had no cap in either language.

One robustness addition on top, needed only because this now runs many
attempts unattended in parallel (does not change the physics):

  - The whole attempt runs inside a try/catch, so a single relax()
    failure (its designed `ArgumentError` on step-size collapse) is
    recorded as a failed attempt instead of killing the worker process.
    Phase 1 previously had no iteration cap at all in either language --
    fine to babysit interactively, not fine unattended, since one kick
    that never destabilizes would strand a worker forever. It now
    shares the same max_total_iter budget as phase 2.
=#

using LinearAlgebra
using Random

struct AttemptResult
    success::Bool
    seed::Int
    kicked_atom::Int
    kick_dir::Vector{Float64}
    initial_energy::Float64
    final_energy::Float64
    crit_eigenvalue::Float64
    iteration::Int
    steps_part1::Int
    positions::Union{Matrix{Float64},Nothing}
    dump_file::Union{String,Nothing}
    error::Union{String,Nothing}
end

function _failed_result(seed, kicked_atom, kick_dir, initial_energy, err::String)
    AttemptResult(false, seed, kicked_atom, kick_dir, initial_energy, NaN, NaN, 0, 0, nothing, nothing, err)
end

"""
    run_art_attempt(seed, positions0, data, pairs0, box; dump_file=nothing,
                    max_total_iter=2000)

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
)
    rng = MersenneTwister(seed)
    natoms = data["natoms"]

    kicked_atom = rand(rng, 1:natoms)
    kick_dir = randn(rng, 3)
    kick_dir ./= norm(kick_dir)

    initial_energy = compute_energy(
        positions0, data["cid"], box, epsilon_table, sigma_table, pairs0
    )

    try
        positions = copy(positions0)
        pairs = copy(pairs0)

        first_move = zeros(natoms, 3)
        first_move[kicked_atom, :] = kick_dir .* first_move_modifier

        positions .+= first_move
        positions = relax(positions, data, pairs, first_move, box)

        # ---------------- phase 1: fixed-direction activation ----------------
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

        iteration = 0
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
            return _failed_result(seed, kicked_atom, kick_dir, initial_energy,
                                   "phase 1 did not reach eigenvalue_cutoff within max_total_iter")
        end

        steps_part1 = iteration

        # ---------------- phase 2: eigenvector-following to the saddle ----------------
        moves = relevant_eigenvector .* move_phase2_modifier
        moves = reshape(moves, 3, :)'

        crit_eigenvalue = relevant_eigenvalue
        val = vals

        while crit_eigenvalue < 0 &&
              iteration < max_total_iter &&
              abs(dot(moves, F) / move_phase2_modifier) > dot_product_saddle_cutoff

            if iteration % 10 == 0
                pairs = build_neighbor_pairs(positions, rc, box)
            end

            old_moves = copy(moves)
            moves, val = calculate_moves(positions, data, pairs, F, old_moves, move_phase2_modifier)
            crit_eigenvalue = val[lowest_nonzero_mode(val)]

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

        return AttemptResult(
            success, seed, kicked_atom, kick_dir, initial_energy, final_energy,
            crit_eigenvalue, iteration, steps_part1,
            success ? positions : nothing,
            success ? dump_file : nothing,
            nothing,
        )

    catch e
        return _failed_result(seed, kicked_atom, kick_dir, initial_energy, sprint(showerror, e))
    end
end

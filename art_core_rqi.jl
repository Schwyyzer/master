#=
Single-attempt ART (activation-relaxation) logic -- same as
art_core_no_tracking.jl (the validated 10/10, then 50/50 baseline),
EXCEPT phase 2 uses the cheap single-vector RQI tracker
(track_eigenvector_step, from rqi_tracking.jl) instead of a cold full
re-diagonalization every iteration.

Rationale: your own 50/50 run logged max_phase2_angle_deg every step
under the cold baseline -- check that column in its results.csv. If
it's consistently ~1-2 degrees (small enough that a single Newton/
Jacobi-Davidson-style correction step should land right on the new
mode), this should be a large speedup for the same result: no
diagonalization at all in the common case, just one Hessian-vector
product per phase-2 iteration.

This is a middle ground between the two things already tried:
  - art_core.jl's block mode tracking (track_modes/LOBPCG) is general
    enough to survive an eigenvalue crossing, but costs an LOBPCG solve
    (iterative, several matrix-vector products, a QR, an overlap-match)
    every call, and needed a real bug fix (non-PD preconditioner) to be
    trustworthy at all -- see mode_tracking.jl / test_mode_tracking_precond.jl.
  - art_core_no_tracking.jl's calculate_moves is a full ARPACK
    shift-invert diagonalization (sparse LU factorization) every call --
    correct and simple, but the most expensive option, and itself not
    immune to the same "large angle" symptom track_modes was meant to
    fix, since it just takes whatever is currently the lowest nonzero
    mode rather than tracking a specific one.
  - This file: ONE cheap correction step, explicitly capped at
    `max_angle_deg`, with automatic fallback to a full diagonalization
    (same k=15 block + overlap-match as art_core.jl's full_recheck path)
    whenever that step doesn't trust itself (hit the cap, or produced a
    non-finite result) -- not just periodically. Also still does the
    periodic full_recheck safety check (`mode_recheck_every`) as a
    second, independent line of defense against slow silent drift that
    never trips the per-step guardrail.

AttemptResult gains one field versus art_core_no_tracking.jl:
`n_full_rechecks`, counting how many phase-2 iterations fell back to a
full diagonalization (periodic + guardrail-triggered combined) out of
`iteration - steps_part1` total phase-2 iterations. That ratio is the
direct empirical answer to "does the 1-2 degree assumption actually
hold on the real system": near 0 (beyond the periodic ones) confirms it
and means this is running about as cheap as it can; a high count means
the guardrail is earning its keep and phase 2 isn't as smooth as hoped.

This is meant to be included AFTER Version1.1.jl (same as art_core.jl /
art_core_no_tracking.jl) and AFTER rqi_tracking.jl.
=#

using LinearAlgebra
using Random

include("rqi_tracking.jl")

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
    # angle (degrees) between successive phase-2 moves
    max_phase2_angle_deg::Union{Float64,Nothing} = nothing
    # how many phase-2 iterations fell back to a full diagonalization
    # (periodic mode_recheck_every ticks + guardrail trips combined)
    n_full_rechecks::Union{Int,Nothing} = nothing
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
`initial_energy`. Returns (energy, positions, new_minimum::Bool).
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
    calculate_moves_rqi(positions, data, neighbors, F, v_prev, move_modifier=move_phase2_modifier;
                         full_recheck=false, max_angle_deg=10.0, k_full=15)

Phase-2 move calculation using the cheap single-vector RQI tracker
instead of a cold full re-diagonalization. `v_prev` is the tracked
eigenvector from the previous call (or from the phase1->phase2
handoff). Returns `(moves, eigenvalue, v_new, did_full_recheck)`.

Falls back to one full diagonalization (k=15 block, re-identified by
overlap with `v_prev`, sign-aligned so it stays continuous) whenever
either `full_recheck=true` is passed (the periodic safety check), or
the RQI step itself reports it hit its angle guardrail / produced a
non-finite result (it doesn't trust its own correction that time).
"""
function calculate_moves_rqi(
    positions, data, neighbors, F, v_prev::AbstractVector,
    move_modifier=move_phase2_modifier;
    full_recheck::Bool=false, max_angle_deg::Real=10.0, k_full::Int=15,
)
    H = build_hessian_fast(positions, data, neighbors)

    did_full_recheck = full_recheck
    local v_new, eigenvalue

    if !full_recheck
        eigenvalue, v_new, angle_deg = track_eigenvector_step(H, v_prev; max_angle_deg=max_angle_deg)
        if angle_deg >= max_angle_deg - 1e-9 || !isfinite(eigenvalue) || !all(isfinite, v_new)
            did_full_recheck = true
        end
    end

    if did_full_recheck
        vals_full, vecs_full = lowest_modes(H, k_full)
        overlaps = [abs(dot(v_prev, vecs_full[:, i])) for i in 1:k_full]
        slot = argmax(overlaps)
        v_new = vecs_full[:, slot]
        if dot(v_new, v_prev) < 0
            v_new = -v_new   # keep sign continuous with v_prev
        end
        eigenvalue = vals_full[slot]
    end

    moves = v_new .* move_modifier
    moves = reshape(moves, 3, :)'

    if dot(moves, F) > 0
        moves = -moves
    end

    return moves, eigenvalue, v_new, did_full_recheck
end

"""
    run_art_attempt(seed, positions0, data, pairs0, box; dump_file=nothing,
                    max_total_iter=2000, mode_recheck_every=25, max_angle_deg=10.0)

Run one full ART attempt (random kick -> phase 1 activation -> phase 2
eigenvector-following) starting from the given relaxed minimum
`positions0`, using a private RNG seeded with `seed` so results are
reproducible and independent across parallel attempts.

`positions0`, `data`, `pairs0`, `box` are all read-only here (relax()
mutates its own local copy, never the caller's array).

If `dump_file` is given and the attempt succeeds, a single LAMMPS frame
with the final (converged) saddle configuration is written there.
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
    max_angle_deg::Real = 10.0,
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
        # Seed the tracker directly from phase 1's last eigenvector on its
        # last Hessian -- no extra diagonalization needed at the handoff
        # (matches art_core_no_tracking.jl's seeding, unlike art_core.jl's
        # block version which needed a k=15 diagonalization here just to
        # build X_prev's columns).
        v_track = relevant_eigenvector ./ norm(relevant_eigenvector)
        moves = v_track .* move_phase2_modifier
        moves = reshape(moves, 3, :)'

        crit_eigenvalue = relevant_eigenvalue
        max_phase2_angle_deg = 0.0
        n_full_rechecks = 0

        while crit_eigenvalue < 0 &&
              iteration < max_total_iter &&
              abs(dot(moves, F) / move_phase2_modifier) > dot_product_saddle_cutoff

            if iteration % 10 == 0
                pairs = build_neighbor_pairs(positions, rc, box)
            end

            old_moves = copy(moves)
            moves, crit_eigenvalue, v_track, did_full = calculate_moves_rqi(
                positions, data, pairs, F, v_track, move_phase2_modifier;
                full_recheck = (iteration % mode_recheck_every == 0),
                max_angle_deg = max_angle_deg,
            )
            n_full_rechecks += did_full

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
            max_phase2_angle_deg, n_full_rechecks,
        )

    catch e
        return _failed_result(
            seed, kicked_atom, kick_dir, initial_energy,
            "[$(phase), iteration=$(iteration)] " * sprint(showerror, e);
            iteration, steps_part1,
        )
    end
end

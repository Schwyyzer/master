
using DelimitedFiles
using Printf
using Dates

function relax(positions, data, neighbors, moves, box;
               step_magnitude=relaxation_step_magnitude,
               max_steps=max_relaxation_steps,
               verbose::Bool=false)
    #=
    Same numerics as the original relax(): perpendicular-to-`moves`
    steepest-descent-with-adaptive-step-size relaxation (or full,
    unconstrained relaxation when moves === nothing). The only change
    is that the original's per-iteration disk dumps (a whole
    `relaxation_data_<timestamp>/iteration_NNNNN/*.txt` tree, written
    on *every* gradient step of *every* relax() call) are now gated
    behind `verbose=true` instead of always running -- for a batch of
    many parallel ART attempts that I/O dominates the runtime and can
    also race across concurrent OS processes. Pass verbose=true to get
    the exact original debug output back for a single interactive run.
    =#

    Iter = 0
    tol = 1e-5
    positions .= mod.(positions, collect(box)')
    neighbors = build_neighbor_pairs(positions, rc, box)

    initialEnergy = compute_energy(
        positions, data["cid"], box,
        epsilon_table, sigma_table, neighbors
    )

    io = nothing
    output_dir = nothing

    if verbose
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS_ffffff")
        output_dir = "/home/schwyyzer/Desktop/Master Thesis/relaxation_data_$timestamp"
        mkpath(output_dir)

        if moves !== nothing
            writedlm(joinpath(output_dir, "moves.txt"), moves)
        end
        writedlm(joinpath(output_dir, "initial_positions.txt"), positions)
        writedlm(joinpath(output_dir, "neighbors_used.txt"), neighbors)

        io = open(joinpath(output_dir, "summary.txt"), "w")
        println(io, "# Iteration  E_current  E_trial  step_magnitude  max_force")
    end

    if moves !== nothing

        max_force = Inf

        while max_force > tol && Iter < max_steps

            Force_vector = force(positions, data, neighbors)
            Force_vector_perp = perpendicular_forces(Force_vector, moves)
            max_force = maximum(norm.(eachrow(Force_vector_perp)))

            trial_positions = positions + step_magnitude * Force_vector_perp

            E_current = compute_energy(
                positions, data["cid"], box, epsilon_table, sigma_table, neighbors
            )
            E_trial = compute_energy(
                trial_positions, data["cid"], box, epsilon_table, sigma_table, neighbors
            )

            iter_dir = nothing
            if verbose
                println("Iteration = ", Iter)
                println("E_current = ", E_current)
                println("E_trial   = ", E_trial)
                println("step      = ", step_magnitude)
                println("max force = ", max_force)

                iter_dir = joinpath(output_dir, @sprintf("iteration_%05d", Iter))
                mkpath(iter_dir)

                writedlm(joinpath(iter_dir, "positions.txt"), positions)
                writedlm(joinpath(iter_dir, "trial_positions.txt"), trial_positions)
                writedlm(joinpath(iter_dir, "force.txt"), Force_vector)
                writedlm(joinpath(iter_dir, "force_perp.txt"), Force_vector_perp)
                writedlm(joinpath(iter_dir, "trial_move.txt"), step_magnitude .* Force_vector_perp)
                writedlm(joinpath(iter_dir, "moves.txt"), moves)

                open(joinpath(iter_dir, "scalars.txt"), "w") do scalar_io
                    println(scalar_io, "iteration ", Iter)
                    println(scalar_io, "E_current ", E_current)
                    println(scalar_io, "E_trial ", E_trial)
                    println(scalar_io, "step_magnitude ", step_magnitude)
                    println(scalar_io, "max_force ", max_force)
                end

                println(
                    io,
                    Iter, " ",
                    @sprintf("%.16e", E_current), " ",
                    @sprintf("%.16e", E_trial), " ",
                    @sprintf("%.16e", step_magnitude), " ",
                    @sprintf("%.16e", max_force)
                )
            end

            # ------------------------------------------------
            # Accept/reject trial step
            # ------------------------------------------------
            if E_trial < E_current

                positions .= mod.(trial_positions, [box[1] box[2] box[3]])
                step_magnitude *= relaxation_step_magnitude_multiplier_success
                accepted = true

            else

                step_magnitude *= relaxation_step_magnitude_multiplier_failure
                accepted = false

                if step_magnitude < 1e-10

                    if verbose
                        F = force(positions, data, neighbors)
                        Fperp = perpendicular_forces(F, moves)
                        δ = 1e-6

                        E0 = compute_energy(positions, data["cid"], box, epsilon_table, sigma_table, neighbors)
                        Eplus = compute_energy(positions .+ δ .* Fperp, data["cid"], box, epsilon_table, sigma_table, neighbors)
                        Eminus = compute_energy(positions .- δ .* Fperp, data["cid"], box, epsilon_table, sigma_table, neighbors)

                        open(joinpath(iter_dir, "finite_difference.txt"), "w") do fd_io
                            println(fd_io, "E0 ", E0)
                            println(fd_io, "Eplus ", Eplus)
                            println(fd_io, "Eminus ", Eminus)
                            println(fd_io, "numerical_dE_dalpha ", (Eplus - Eminus) / (2δ))
                            println(fd_io, "minus_F_dot_Fperp ", -dot(F, Fperp))
                        end

                        writedlm(joinpath(iter_dir, "diagnostic_force.txt"), F)
                        writedlm(joinpath(iter_dir, "diagnostic_force_perp.txt"), Fperp)
                    end

                    if verbose
                        close(io)
                    end
                    throw(ArgumentError("Relaxation step magnitude too small."))
                end
            end

            if verbose
                open(joinpath(iter_dir, "accepted.txt"), "w") do accepted_io
                    println(accepted_io, accepted)
                end
            end

            Iter += 1
        end

    else

        # ====================================================
        # UNCONSTRAINED RELAXATION
        # ====================================================

        max_force = Inf

        while max_force > tol && Iter < 5000

            Force_vector = force(positions, data, neighbors)
            max_force = maximum(norm.(eachrow(Force_vector)))

            trial_positions = positions + step_magnitude * Force_vector

            E_current = compute_energy(
                positions, data["cid"], box, epsilon_table, sigma_table, neighbors
            )
            E_trial = compute_energy(
                trial_positions, data["cid"], box, epsilon_table, sigma_table, neighbors
            )

            iter_dir = nothing
            if verbose
                println("Iteration = ", Iter)
                println("E_current = ", E_current)
                println("E_trial   = ", E_trial)
                println("step      = ", step_magnitude)
                println("max force = ", max_force)

                iter_dir = joinpath(output_dir, @sprintf("iteration_%05d", Iter))
                mkpath(iter_dir)

                writedlm(joinpath(iter_dir, "positions.txt"), positions)
                writedlm(joinpath(iter_dir, "trial_positions.txt"), trial_positions)
                writedlm(joinpath(iter_dir, "force.txt"), Force_vector)
                writedlm(joinpath(iter_dir, "trial_move.txt"), step_magnitude .* Force_vector)

                open(joinpath(iter_dir, "scalars.txt"), "w") do scalar_io
                    println(scalar_io, "iteration ", Iter)
                    println(scalar_io, "E_current ", E_current)
                    println(scalar_io, "E_trial ", E_trial)
                    println(scalar_io, "step_magnitude ", step_magnitude)
                    println(scalar_io, "max_force ", max_force)
                end

                println(
                    io,
                    Iter, " ",
                    @sprintf("%.16e", E_current), " ",
                    @sprintf("%.16e", E_trial), " ",
                    @sprintf("%.16e", step_magnitude), " ",
                    @sprintf("%.16e", max_force)
                )
            end

            if E_trial < E_current

                positions .= mod.(trial_positions, [box[1] box[2] box[3]])
                step_magnitude *= relaxation_step_magnitude_multiplier_success
                accepted = true

            else

                step_magnitude *= relaxation_step_magnitude_multiplier_failure
                accepted = false

                if step_magnitude < 1e-10
                    if verbose
                        close(io)
                    end
                    throw(ArgumentError("Relaxation step magnitude too small."))
                end
            end

            if verbose
                open(joinpath(iter_dir, "accepted.txt"), "w") do accepted_io
                    println(accepted_io, accepted)
                end
            end

            Iter += 1
        end
    end

    if verbose
        close(io)
    end

    if verbose
        relaxedEnergy = compute_energy(
            positions, data["cid"], box, epsilon_table, sigma_table, neighbors
        )
        println("lowered energy by ", initialEnergy - relaxedEnergy)
    end

    return positions
end



function calculate_moves(positions, data, neighbors, F,old_move, move_modifier=move_phase2_modifier)
    H=build_hessian_fast(positions, data, neighbors)
    #vals, vecs = track_mode(H, old_move)
    vals, vecs = lowest_modes(H, 15)  # k=15 to match new_start_1_1_1.py's calculate_moves
    moves = vecs[:,lowest_nonzero_mode(vals)] .* move_modifier
    moves = reshape(moves, 3, :)'

    dot_product = dot(moves, F)
    if dot_product > 0
        moves = -moves
    end
    return moves, vals
end

using LinearAlgebra

function angle_between(v1, v2)
    @assert size(v1) == size(v2)

    c = dot(v1, v2) / (norm(v1) * norm(v2))

    # protect against roundoff errors
    c = clamp(c, -1.0, 1.0)

    return acos(c)  # radians
end

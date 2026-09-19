include("lammps_and_document.jl")
include("config.jl")
include("neighbor_pairs.jl")
include("Hessian.jl")
include("particle_movement.jl")
include("eigenvalueDecomp.jl")
using BenchmarkTools






move_phase2_modifier_original = move_phase2_modifier

function d_V(r, a, b)
    if r <= rc && r > 0
        s = sigma_table[a, b]
        eps = epsilon_table[a, b]

        return -4.0 * eps * (12.0 * (s / r) ^ 12 / r - 6.0 * (s / r) ^ 6 / r)
    else
        return 0.0
    end
end

function dd_V(r, a, b)
    if r <= rc && r > 0
        s = sigma_table[a, b]
        eps = epsilon_table[a, b]

        return 4.0 * eps * (12.0 * 13.0 * (s / r) ^ 12 / r ^ 2 - 6.0 * 7.0 * (s / r) ^ 6 / r ^ 2)
    else
        return 0.0
    end
end


function perpendicular_forces(forces, moves)
    force_vector = vec(forces')
    move_vector  = vec(moves')
    denominator  = dot(move_vector, move_vector)
    perp_vector  = force_vector .- (dot(force_vector, move_vector) / denominator) .* move_vector
    return reshape(perp_vector, 3, :)'
end



function force(positions, data, neighbors;
               eps_matrix=epsilon_table,
               sigma_matrix=sigma_table)

    N = size(positions,1)
    F = zeros(N,3)

    lx = data["lx"]
    ly = data["ly"]
    lz = data["lz"]

    cid = data["cid"]

    @inbounds for p in axes(neighbors,1)

        i = neighbors[p,1]
        j = neighbors[p,2]

        dx = positions[i,1] - positions[j,1]
        dy = positions[i,2] - positions[j,2]
        dz = positions[i,3] - positions[j,3]

        dx -= lx * round(dx/lx)
        dy -= ly * round(dy/ly)
        dz -= lz * round(dz/lz)

        r2 = dx*dx + dy*dy + dz*dz

        r2 = dx*dx + dy*dy + dz*dz
        r  = sqrt(r2)

        if r <= rc && r > 0
            ti = cid[i]; tj = cid[j]
            eps = eps_matrix[ti,tj]
            sig = sigma_matrix[ti,tj]
            sig6  = sig^6
            sig12 = sig6^2
            pref = 24 * eps * (2*sig12/r2^7 - sig6/r2^4)

            fx = pref * dx; fy = pref * dy; fz = pref * dz
            F[i,1] += fx; F[i,2] += fy; F[i,3] += fz
            F[j,1] -= fx; F[j,2] -= fy; F[j,3] -= fz
        end
    end
    #writedlm("//home//schwyyzer//Desktop//Master Thesis//neighbors1.txt",neighbors)
    #writedlm("//home//schwyyzer//Desktop//Master Thesis//positions1.txt",positions)
    #writedlm("//home//schwyyzer//Desktop//Master Thesis//Force1.txt",F)
    return F
end

function compute_energy(positions, cid, box, eps_matrix, sigma_matrix, neighbors)
    energy = 0.0
    counter=1
    completeEnergy=zeros(size(neighbors,1))

    @inbounds for p in axes(neighbors,1)

        i = neighbors[p,1]
        j = neighbors[p,2]

        dx = positions[i,1] - positions[j,1]
        dy = positions[i,2] - positions[j,2]
        dz = positions[i,3] - positions[j,3]

        dx -= box[1] * round(dx / box[1])
        dy -= box[2] * round(dy / box[2])
        dz -= box[3] * round(dz / box[3])

        r2 = dx*dx + dy*dy + dz*dz
        r = sqrt(r2)

        ti = cid[i]
        tj = cid[j]

        eps = eps_matrix[ti,tj]
        sig = sigma_matrix[ti,tj]

        if r <= rc && r > 0
            sig6  = sig^6
            sig12 = sig6^2

            energy += 4 * eps * (sig12/r^12 - sig6/r^6)
            completeEnergy[counter]=4 * eps * (sig12/r^12 - sig6/r^6)
        end
        counter+=1
    end

    #writedlm("//home//schwyyzer//Desktop//Master Thesis//neighbors.txt",neighbors)
    #writedlm("//home//schwyyzer//Desktop//Master Thesis//completeEnergy.txt",completeEnergy)
    return energy
end

#julia> writedlm("//home//schwyyzer//Desktop//Master Thesis//pos.txt",positions)

function same_neighbor_list(A, B)
    pairs_A = Set((min(row[1], row[2]), max(row[1], row[2])) for row in eachrow(A))
    pairs_B = Set((min(row[1], row[2]), max(row[1], row[2])) for row in eachrow(B))
    return pairs_A == pairs_B
end
function run_simulation()

    data = parse_lammps_data(inpath2)
    #println("Parsed LAMMPS data: ", data)
    positions = hcat(data["x"], data["y"], data["z"])
    box = (data["lx"], data["ly"], data["lz"])
    pairs = build_neighbor_pairs(positions, rc, box)
    #println("Number of neighbor pairs: ", size(pairs, 1))
    count = test(positions, rc, box)

    f=force(positions, data, pairs)
    a=size(unique(eachrow(pairs)) |> collect, 1)
    #println("Number of unique pairs: ", a)
    same = same_neighbor_list(pairs, count)
   # println("Neighbor lists identical: ", same)

    if any(pairs[:,1] .== pairs[:,2])
        throw(ArgumentError("Self-interaction detected in neighbor pairs."))
    end




    #===========================================================#

    dump_file, details_file = create_dump_file()
    energies = Float64[]
    forces = Float64[]
    crit_eigenvalues = Float64[]

    push!(forces, norm(f))

    random_particle_index = rand(1:size(positions,1))
    #random_particle_index = 10
    first_move_component = randn(3)
    #@show typeof(first_move_component)
    #first_move_component = [-0.303493,-0.0457965,0.00647575]
    #@show typeof(first_move_component)

    first_move = zeros(size(positions))
    first_move[random_particle_index, :] = first_move_component ./norm(first_move_component) .*first_move_modifier

    starting_positions = copy(positions)

    positions .+= first_move

    positions = relax(positions, data, pairs, first_move, box)
    push!(forces, norm(force(positions, data, pairs)))

    H = build_hessian_fast(positions, data, pairs)
#=
    @btime pairs = build_hessian_fast($positions, $data, $pairs)
    @btime p = build_hessian($positions, $data, $pairs)
    @btime a,b = lowest_modes(H,5)
    println("contains NaN = ",
        any(isnan, H.nzval))
    return
=#
    println(norm(H-H'))
    #vals, vecs = eig_from_csv_lobpcg(H, csvfile)
    vals, vecs = lowest_modes(H, 5)
    relevant_eigenvalue = vals[lowest_nonzero_mode(vals)]
    #@show typeof(relevant_eigenvalue)
    relevant_eigenvector = vecs[:, lowest_nonzero_mode(vals)]
    moves = relevant_eigenvector .* move_phase1_modifier
    moves = reshape(moves, 3, :)'
    iteration = 1
    F = nothing
    #println("Critical eigenvalue: ", relevant_eigenvalue)
    #@show moves
    relevant_eigenvalue = vals[lowest_nonzero_mode(vals)]

    #@show typeof(relevant_eigenvalue)
    #@show typeof(eigenvalue_cutoff)
    while relevant_eigenvalue > eigenvalue_cutoff
        if iteration%1==0

            positions .= mod.(positions, collect(box)')
            pairs = build_neighbor_pairs(positions, rc, box)
            #count = test(positions, rc, box)
            #same = same_neighbor_list(pairs, count)
            #println("Neighbor lists identical: ", same)
        end
        #@show(size(positions))
        #@show(size(moves))
        positions .+= moves
        positions = relax(positions, data, pairs, moves, box)



        H = build_hessian_fast(positions, data, pairs)
        #relevant_eigenvalue, relevant_eigenvector = lowest_modes(H, 1, previous_eigenvectors=reshape(relevant_eigenvector, :, 1))
        @assert maximum(abs.(H - H')) < 1e-8

        vals, vecs = lowest_modes(H, 5, previous_eigenvectors=vecs)
        #@show relevant_eigenvalue

        relevant_eigenvalue=vals[lowest_nonzero_mode(vals)]

        #@show typeof(forces)
        #@show typeof(iteration)
        #@show typeof(energies)
        #@show typeof(crit_eigenvalues)
        F = force(positions, data, pairs)
        energies, forces, iteration, crit_eigenvalues =
        document(
                positions,
                data,
                pairs,
                F,
                box,
                dump_file,
                forces,
                iteration,
                energies,
                relevant_eigenvalue,
                crit_eigenvalues,
                moves
            )
        #println("Critical eigenvalue: ", relevant_eigenvalue)
    end

    if iteration ==1
        return
    end



    moves = relevant_eigenvector .* move_phase2_modifier
    moves = reshape(moves, 3, :)'
    println("Part 1 completed. Starting Part 2.")
    while relevant_eigenvalue < 0 && iteration < 1000 && abs(dot(moves, F)/move_phase2_modifier) > dot_product_saddle_cutoff
        if iteration%10==0
            pairs = build_neighbor_pairs(positions, rc, box)
        end

        old_moves = copy(moves)
        moves, val= calculate_moves(positions, data, pairs,F, old_moves, move_phase2_modifier)
        temp = reshape(old_moves, :, 1)
        relevant_eigenvalue=val[lowest_nonzero_mode(val)]
        angle = rad2deg(angle_between(reshape(old_moves, :, 1), reshape(moves, :, 1)))
        if angle >10
            println("Warning: Angle between old and new moves is greater than 10 degrees. Angle: ", angle)
        end
        positions .+= moves
        positions = relax(positions, data, pairs, moves, box)
        F = force(positions, data, pairs)

        energies, forces, iteration, crit_eigenvalues =
        document(
                positions,
                data,
                pairs,
                F,
                box,
                dump_file,
                forces,
                iteration,
                energies,
                val,
                relevant_eigenvalue,
                moves
            )
    end
    println("Part 2 completed. Simulation finished. \n      Evaluating final configuration...")
    success=false
    if crit_eigenvalues[end] < 0 && iteration < 1000
        println("Final configuration might be a saddle point.")
        success=true
    else
        println("Final configuration is not a saddle point.")
    end

    return positions,success

end

using Dates
using LinearAlgebra
using Printf

function create_dump_file()
    mkpath("dump")   # create directory if it doesn't exist
    mkpath("plots")

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")

    working_dir = "//home//schwyyzer//Desktop//Master Thesis"

    filename = joinpath(working_dir,"dump", "trajectory_julia$(timestamp).dump")
    filename2 = joinpath(working_dir, "plots", "trajectory_julia$(timestamp).txt")

    # create/clear the file
    open(filename, "w") do io
    end

    return filename, filename2
end

function parse_lammps_data(path)

    lines = [strip(line) for line in readlines(path)]

    # --- find number of atoms ---
    natoms = nothing

    for line in lines
        if endswith(line, "atoms")
            natoms = parse(Int, split(line)[1])
            break
        end
    end

    if isnothing(natoms)
        error("Could not find 'atoms' line")
    end

    # --- find box bounds ---
    xlo = ylo = zlo = nothing
    xhi = yhi = zhi = nothing

    for line in lines
        if endswith(line, "xlo xhi")
            vals = split(line)
            xlo = parse(Float64, vals[1])
            xhi = parse(Float64, vals[2])

        elseif endswith(line, "ylo yhi")
            vals = split(line)
            ylo = parse(Float64, vals[1])
            yhi = parse(Float64, vals[2])

        elseif endswith(line, "zlo zhi")
            vals = split(line)
            zlo = parse(Float64, vals[1])
            zhi = parse(Float64, vals[2])
        end
    end

    if any(isnothing.((xlo, xhi, ylo, yhi, zlo, zhi)))
        error("Box bounds not found")
    end

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    # --- find the "Atoms" section ---
    atoms_idx = nothing

    for (i, line) in enumerate(lines)
        if startswith(line, "Atoms")
            atoms_idx = i
            break
        end
    end

    if isnothing(atoms_idx)
        error("Could not find 'Atoms' section")
    end

    start = atoms_idx + 2
    atom_lines = lines[start:start + natoms - 1]

    id_arr = zeros(Int, natoms)
    cid_arr = zeros(Int, natoms)

    x_arr = zeros(natoms)
    y_arr = zeros(natoms)
    z_arr = zeros(natoms)

    for (i, line) in enumerate(atom_lines)

        parts = split(line)

        if length(parts) < 5
            error("Atom line has too few columns: $line")
        end

        id_arr[i] = parse(Int, parts[1])
        cid_arr[i] = parse(Int, parts[2])

        x_arr[i] = parse(Float64, parts[3])
        y_arr[i] = parse(Float64, parts[4])
        z_arr[i] = parse(Float64, parts[5])
    end

    return Dict(
        "natoms" => natoms,
        "id" => id_arr,
        "cid" => cid_arr,
        "x" => x_arr,
        "y" => y_arr,
        "z" => z_arr,
        "lx" => lx,
        "ly" => ly,
        "lz" => lz,
    )
end


function document(
    positions,
    data,
    neighbor_pairs,
    F,
    box,
    dump_file,
    forces,
    iteration,
    energies,
    crit_eigenvalue,
    crit_eigenvalues,
    move
)

    push!(
        energies,
        compute_energy(
            positions,
            data["cid"],
            [data["lx"], data["ly"], data["lz"]],
            epsilon_table,
            sigma_table,
            neighbor_pairs
        )
    )

    #push!(crit_eigenvalues, crit_eigenvalue)

    push!(
        forces,
        norm(F)
    )

    iteration += 1

    if iteration % 20 == 0
        println("Iteration ", iteration, " completed.")
    end

    if !isnothing(dump_file)
        write_lammps_frame(
            dump_file,
            iteration,
            positions,
            data["id"],
            data["cid"],
            (data["lx"], data["ly"], data["lz"]),
            move
        )
    end

    return energies, forces, iteration, crit_eigenvalues

end


function write_lammps_frame(
    filename,
    timestep,
    positions,
    ids,
    types,
    box,
    move
)

    lx, ly, lz = box
    N = length(ids)

    open(filename, "a") do io

        # Position frame
        println(io, "ITEM: TIMESTEP")
        println(io, timestep)

        println(io, "ITEM: NUMBER OF ATOMS")
        println(io, N)

        println(io, "ITEM: BOX BOUNDS pp pp pp")
        println(io, "0 ", lx)
        println(io, "0 ", ly)
        println(io, "0 ", lz)

        println(io, "ITEM: ATOMS id type x y z")

        @inbounds for i in eachindex(ids)
            @printf(
                io,
                "%d %d %.12f %.12f %.12f\n",
                ids[i],
                types[i],
                positions[i,1],
                positions[i,2],
                positions[i,3]
            )
        end

        # Separate move matrix
        println(io, "ITEM: MOVE")
        println(io, N)

        @inbounds for i in 1:N
            @printf(
                io,
                "%.12f %.12f %.12f\n",
                move[i,1],
                move[i,2],
                move[i,3]
            )
        end
    end
end


using SparseArrays, DelimitedFiles

function export_sparse(A, path)
    n, m = size(A)
    I, J, V = findnz(A)          # 1-based row/col indices, as Julia stores them
    open(path, "w") do io
        println(io, n, " ", m, " ", length(V))
        writedlm(io, hcat(I, J, V), ' ')
    end
end

using DelimitedFiles
using SparseArrays

#daqta = readdlm("C:\\Users\\Yanik\\Desktop\\Master Thesis\\newmatrix.txt")

#rows = Int.(daqta[:, 1]) .+ 1  # Julia uses 1-based indexing
#cols = Int.(daqta[:, 2]) .+ 1
#values = Float64.(daqta[:, 3])

#matrix = sparse(rows, cols, values)

function compare_matrices(A, B)
    if size(A) != size(B)
        println("Matrices have different sizes: ", size(A), " vs ", size(B))
        return false
    end

    diff = A - B
    max_diff = maximum(abs.(diff))

    if max_diff < 1e-8
        println("Matrices are approximately equal.")
        return true
    else
        println("Matrices differ. Maximum absolute difference: ", max_diff)
        C=A-B
        for i in 1:size(C,1)
            for j in 1:size(C,2)
                if abs(C[i,j]) < 1e-8
                    C[i,j] = 0
                end
            end
        end
        return C
    end
end

function compare_hessians(H1, H2; tol=1e-6, nshow=10)
    A, B = sparse(H1), sparse(H2)
    D = A - B
    Di, Dj, Dv = findnz(D)
    keep = abs.(Dv) .> tol
    Di, Dj, Dv = Di[keep], Dj[keep], Dv[keep]
    if isempty(Dv)
        println("identical to within tol=", tol)
        return true
    end
    println("max |H1-H2| = ", maximum(abs.(Dv)), "   (", length(Dv), " entries exceed tol)")
    ord = sortperm(abs.(Dv), rev=true)
    for k in ord[1:min(nshow, length(ord))]
        r, c = Di[k], Dj[k]
        ar, ac = div(r-1,3)+1, div(c-1,3)+1
        println("  (", r, ",", c, ")  atoms (", ar, ",", ac, ")  H1=", A[r,c], "  H2=", B[r,c],
                ar==ac ? "  [same-atom]" : "  [cross-atom]")
    end
    return false
end
# H1 = build_hessian(positions, data, neighbour_pairs)
# H2 = build_hessian_fast(positions, data, neighbour_pairs)
# compare_hessians(H1, H2)

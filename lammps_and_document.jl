using Dates
using LinearAlgebra
using Printf

#=
SimData replaces the Dict{String,Any} this used to return. A Dict with
Any-typed values forces every function that reads from it (force,
compute_energy, build_hessian_fast -- the hottest loops in the whole
program) into dynamically-typed, boxed lookups: `data["lx"]` has no
compile-time type, so neither does anything computed from it, and Julia
can't specialize/inline that code at all. A concretely-typed struct fixes
this for any code that accesses fields with `.` -- which the hot functions
below have been switched to.

Every OTHER file in this codebase still says `data["lx"]`, `data["cid"]`,
or (in a couple of test scripts) `data["x"] = ...` -- the getindex/
setindex! methods below make that keep working unchanged, so none of those
call sites needed to change. They're just slower than `.lx` because the
field is looked up by a runtime string each call; fine for the places that
still do it, since none of them are in a per-pair or per-iteration loop.
=#
# Guarded so this file can be `include`d more than once in the same Julia
# session (e.g. re-running a test script interactively) without hitting
# "invalid redefinition of constant SimData" -- Julia disallows redefining
# a struct type in-place, unlike a plain variable or a method.
if !isdefined(@__MODULE__, :SimData)
    mutable struct SimData
        natoms::Int
        id::Vector{Int}
        cid::Vector{Int}
        x::Vector{Float64}
        y::Vector{Float64}
        z::Vector{Float64}
        lx::Float64
        ly::Float64
        lz::Float64
    end
end

Base.getindex(d::SimData, k::AbstractString) = getproperty(d, Symbol(k))
Base.setindex!(d::SimData, v, k::AbstractString) = setproperty!(d, Symbol(k), v)

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

    return SimData(natoms, id_arr, cid_arr, x_arr, y_arr, z_arr, lx, ly, lz)
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

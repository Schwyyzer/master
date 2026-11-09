include("lammps_and_document.jl")
include("config.jl")
include("neighbor_pairs.jl")
include("Hessian.jl")
include("particle_movement.jl")

function perpendicular_forces(forces, moves)
    force_vector = vec(forces')
    move_vector  = vec(moves')
    denominator  = dot(move_vector, move_vector)
    perp_vector  = force_vector .- (dot(force_vector, move_vector) / denominator) .* move_vector
    return reshape(perp_vector, 3, :)'
end

"""
    perpendicular_forces!(forces, moves)

In-place version of `perpendicular_forces`: projects `moves`'s component
out of `forces` by mutating `forces` directly, instead of allocating the
transposed/flattened/reshaped copies `perpendicular_forces` does. `dot` and
broadcasting on an AbstractArray already work elementwise regardless of
shape, so `dot(forces, moves)` on the natoms x 3 matrices gives exactly the
same scalar as `perpendicular_forces`'s flatten-first version -- no
transpose/vec/reshape needed. Matches `perpendicular_forces`'s own
(unconditional) division; `moves` is always a normalized Lanczos
eigenvector in this codebase's call sites, so it is never the zero vector.
"""
function perpendicular_forces!(forces, moves)
    denom = dot(moves, moves)
    coeff = dot(forces, moves) / denom
    @. forces -= coeff * moves
    return forces
end



function force(positions, data, neighbors;
               eps_matrix=epsilon_table,
               sigma_matrix=sigma_table)

    N = size(positions,1)
    F = zeros(N,3)
    force!(F, positions, data, neighbors; eps_matrix, sigma_matrix)
    return F
end

"""
    force!(F, positions, data, neighbors; eps_matrix=epsilon_table, sigma_matrix=sigma_table)

In-place version of `force`: accumulates into a caller-owned `F` (zeroed
first) instead of allocating a fresh array every call. Use this in any
loop that calls `force`/`force!` many times per attempt (FIRE relaxation,
the outer ART iteration) -- a fresh `zeros(N,3)` on every one of those
calls is a lot of avoidable GC pressure at the call counts this runs at.
"""
function force!(F, positions, data, neighbors;
                 eps_matrix=epsilon_table,
                 sigma_matrix=sigma_table)

    fill!(F, 0.0)

    # `data.lx`/`data.cid` (field access) rather than `data["lx"]` (string-
    # keyed lookup): now that `data` is a concretely-typed SimData struct,
    # this is what actually makes this loop type-stable -- see SimData's
    # docstring in lammps_and_document.jl.
    lx = data.lx
    ly = data.ly
    lz = data.lz

    cid = data.cid

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

            # Shift by V(rc) so the pair energy vanishes continuously at the
            # cutoff instead of jumping discontinuously. Without this, every
            # pair crossing r=rc during relaxation introduces a spurious energy
            # jump of magnitude 4*eps*((sig/rc)^12-(sig/rc)^6), which corrupts
            # relax()'s energy-based accept/reject comparisons. This shift is
            # a per-pair constant (depends only on rc and the pair's eps/sig),
            # so it does not affect forces or the Hessian.
            sigc6 = (sig/rc)^6
            sigc12 = sigc6^2
            shift = 4 * eps * (sigc12 - sigc6)

            energy += 4 * eps * (sig12/r^12 - sig6/r^6) - shift
        end
    end

    return energy
end

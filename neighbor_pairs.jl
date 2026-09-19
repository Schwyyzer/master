function build_neighbor_pairs(positions, cutoff, box)
    cutoff+=0.5
    cutoff2 = cutoff^2
    Np = size(positions, 1)

    # ---------------------------------------------------------
    # Number of cells in each direction
    # Cell size is >= cutoff
    # ---------------------------------------------------------
    nx = max(1, floor(Int, box[1] / cutoff))
    ny = max(1, floor(Int, box[2] / cutoff))
    nz = max(1, floor(Int, box[3] / cutoff))

    cell_dx = box[1] / nx
    cell_dy = box[2] / ny
    cell_dz = box[3] / nz

    ncells = nx * ny * nz

    cells = [Int[] for _ in 1:ncells]

    @inline function cellid(ix, iy, iz)
        return ix + (iy - 1) * nx + (iz - 1) * nx * ny
    end

    # ---------------------------------------------------------
    # Assign particles to cells
    # ---------------------------------------------------------
    @inbounds for p in 1:Np

        x = mod(positions[p, 1], box[1])
        y = mod(positions[p, 2], box[2])
        z = mod(positions[p, 3], box[3])

        ix = min(floor(Int, x / cell_dx) + 1, nx)
        iy = min(floor(Int, y / cell_dy) + 1, ny)
        iz = min(floor(Int, z / cell_dz) + 1, nz)

        c = cellid(ix, iy, iz)

        push!(cells[c], p)
    end

    # ---------------------------------------------------------
    # Pair buffers
    # ---------------------------------------------------------
    pairs_i = Int[]
    pairs_j = Int[]

    # Rough size hint. Adjust if your density is very different.
    sizehint!(pairs_i, 20 * Np)
    sizehint!(pairs_j, 20 * Np)

    # ---------------------------------------------------------
    # Loop over cells
    # ---------------------------------------------------------
    @inbounds for iz in 1:nz
        for iy in 1:ny
            for ix in 1:nx

                c1 = cellid(ix, iy, iz)
                particles1 = cells[c1]

                isempty(particles1) && continue

                # Unique neighboring cells for this source cell.
                # This matters when nx, ny, or nz are small and periodic wrapping
                # maps multiple offsets to the same cell.
                neighbor_cells = Int[]

                for dzc in -1:1
                    for dyc in -1:1
                        for dxc in -1:1

                            jx = mod1(ix + dxc, nx)
                            jy = mod1(iy + dyc, ny)
                            jz = mod1(iz + dzc, nz)

                            c2 = cellid(jx, jy, jz)

                            already_seen = false
                            for existing in neighbor_cells
                                if existing == c2
                                    already_seen = true
                                    break
                                end
                            end

                            if !already_seen
                                push!(neighbor_cells, c2)
                            end
                        end
                    end
                end

                # ---------------------------------------------------------
                # Compare particles with particles in neighboring cells
                # ---------------------------------------------------------
                for c2 in neighbor_cells

                    particles2 = cells[c2]

                    isempty(particles2) && continue

                    if c2 == c1

                        # Same cell: only check each pair once
                        n1 = length(particles1)

                        for a in 1:(n1 - 1)

                            p1 = particles1[a]

                            x1 = positions[p1, 1]
                            y1 = positions[p1, 2]
                            z1 = positions[p1, 3]

                            for b in (a + 1):n1

                                p2 = particles1[b]

                                dx = x1 - positions[p2, 1]
                                dy = y1 - positions[p2, 2]
                                dz = z1 - positions[p2, 3]

                                dx -= box[1] * round(dx / box[1])
                                dy -= box[2] * round(dy / box[2])
                                dz -= box[3] * round(dz / box[3])

                                r2 = dx*dx + dy*dy + dz*dz

                                if r2 < cutoff2
                                    push!(pairs_i, p1)
                                    push!(pairs_j, p2)
                                end
                            end
                        end

                    elseif c2 > c1

                            # Different cells: process each cell-cell pair only once
                        for p1 in particles1

                            x1 = positions[p1, 1]
                            y1 = positions[p1, 2]
                            z1 = positions[p1, 3]

                            for p2 in particles2

                                dx = x1 - positions[p2, 1]
                                dy = y1 - positions[p2, 2]
                                dz = z1 - positions[p2, 3]

                                dx -= box[1] * round(dx / box[1])
                                dy -= box[2] * round(dy / box[2])
                                dz -= box[3] * round(dz / box[3])

                                r2 = dx*dx + dy*dy + dz*dz

                                if r2 < cutoff2

                                    # Keep pair order consistent: smaller index first
                                    if p1 < p2
                                        push!(pairs_i, p1)
                                        push!(pairs_j, p2)
                                    else
                                        push!(pairs_i, p2)
                                        push!(pairs_j, p1)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # ---------------------------------------------------------
    # Convert to Matrix{Int}, shape npairs × 2
    # ---------------------------------------------------------
    npairs = length(pairs_i)

    pairs = Matrix{Int}(undef, npairs, 2)

    @inbounds for k in 1:npairs
        pairs[k, 1] = pairs_i[k]
        pairs[k, 2] = pairs_j[k]
    end

    return pairs

end

function test(positions, cutoff, box)

    count = 0
    cutoff2 = cutoff^2

    @inbounds for i in 1:size(positions,1)
        for j in i+1:size(positions,1)

            dx = positions[i,1] - positions[j,1]
            dy = positions[i,2] - positions[j,2]
            dz = positions[i,3] - positions[j,3]

            dx -= box[1] * round(dx / box[1])
            dy -= box[2] * round(dy / box[2])
            dz -= box[3] * round(dz / box[3])

            r2 = dx*dx + dy*dy + dz*dz

            if r2 < cutoff2
                count += 1
            end
        end
    end

    list=zeros(count, 2)
    pos=1


    @inbounds for i in 1:size(positions,1)
        for j in i+1:size(positions,1)

            dx = positions[i,1] - positions[j,1]
            dy = positions[i,2] - positions[j,2]
            dz = positions[i,3] - positions[j,3]

            dx -= box[1] * round(dx / box[1])
            dy -= box[2] * round(dy / box[2])
            dz -= box[3] * round(dz / box[3])

            r2 = dx*dx + dy*dy + dz*dz

            if r2 < cutoff2
                list[pos,:]=[i j]
                pos+=1
            end
        end
    end
    return list
end

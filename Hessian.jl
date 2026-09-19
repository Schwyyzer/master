using SparseArrays
using IterativeSolvers
using LinearAlgebra
using Arpack

function build_hessian(
    positions,
    data,
    neighbors
)

    natoms = data["natoms"]
    cid    = data["cid"]

    lx = data["lx"]
    ly = data["ly"]
    lz = data["lz"]

    rank = 3 * natoms

    npairs = size(neighbors,1)

    n_off  = 18 * npairs
    n_diag = 9  * natoms

    rows = Vector{Int}(undef, n_off + n_diag)
    cols = Vector{Int}(undef, n_off + n_diag)
    vals = Vector{Float64}(undef, n_off + n_diag)

    # ---------------------------------------------------
    # Dense diagonal blocks
    # ---------------------------------------------------

    diag_blocks = zeros(Float64, natoms, 3, 3)

    ptr = 1

    @inbounds for p in 1:npairs

        i = neighbors[p,1]
        j = neighbors[p,2]

        dx = positions[i,1] - positions[j,1]
        dy = positions[i,2] - positions[j,2]
        dz = positions[i,3] - positions[j,3]

        dx -= lx * round(dx/lx)
        dy -= ly * round(dy/ly)
        dz -= lz * round(dz/lz)

        r2 = dx*dx + dy*dy + dz*dz
        r  = sqrt(r2)
        if !(r <= rc && r > 0)
            continue
        end
        ti = cid[i]
        tj = cid[j]

        eps = epsilon_table[ti,tj]
        sig = sigma_table[ti,tj]

        mi = mass[ti]
        mj = mass[tj]

        diag_i  = 1.0/mi
        diag_j  = 1.0/mj
        offdiag = -1.0/sqrt(mi*mj)

        sr2  = (sig*sig)/r2
        sr6  = sr2^3
        sr12 = sr6^2

        dV =
            -4.0*eps*(
                12.0*sr12/r -
                6.0*sr6/r
            )

        ddV =
            4.0*eps*(
                12.0*13.0*sr12/r2 -
                6.0*7.0*sr6/r2
            )

        fn0 = dV/r
        fn1 = ddV - fn0

        invr2 = 1.0/r2

        xx = fn1*dx*dx*invr2 + fn0
        yy = fn1*dy*dy*invr2 + fn0
        zz = fn1*dz*dz*invr2 + fn0

        xy = fn1*dx*dy*invr2
        xz = fn1*dx*dz*invr2
        yz = fn1*dy*dz*invr2

        comps = (
            (1,1,xx),
            (2,2,yy),
            (3,3,zz),
            (1,2,xy),
            (2,1,xy),
            (1,3,xz),
            (3,1,xz),
            (2,3,yz),
            (3,2,yz)
        )

        # ---------------------------------------
        # diagonal accumulation
        # ---------------------------------------

        for (a,b,c) in comps

            diag_blocks[i,a,b] += diag_i*c
            diag_blocks[j,a,b] += diag_j*c

        end

        # ---------------------------------------
        # off-diagonal blocks
        # ---------------------------------------

        for (a,b,c) in comps

            rows[ptr] = 3*(i-1) + a
            cols[ptr] = 3*(j-1) + b
            vals[ptr] = offdiag * c
            ptr += 1

            rows[ptr] = 3*(j-1) + a
            cols[ptr] = 3*(i-1) + b
            vals[ptr] = offdiag * c
            ptr += 1

        end
    end

    # ---------------------------------------------------
    # diagonal blocks
    # ---------------------------------------------------

    for atom in 1:natoms
        for a in 1:3
            for b in 1:3

                rows[ptr] = 3*(atom-1) + a
                cols[ptr] = 3*(atom-1) + b
                vals[ptr] = diag_blocks[atom,a,b]

                ptr += 1
            end
        end
    end

    H = sparse(rows[1:ptr-1], cols[1:ptr-1], vals[1:ptr-1], rank, rank)

    return H
end




function lowest_modes(
    H,
    nev;
    previous_eigenvectors = nothing,
    tol = 1e-5,
    maxiter = 500
)
    BLAS.set_num_threads(16)
    n = size(H,1)

    # --------------------------------------------------
    # Initial guess
    # --------------------------------------------------

    if previous_eigenvectors === nothing

        X = randn(n, nev)

    else

        if size(previous_eigenvectors) != (n, nev)
            throw(ArgumentError(
                "previous_eigenvectors must have size ($n,$nev)"
            ))
        end

        X = copy(previous_eigenvectors)

        # small perturbation avoids stagnation
        X .+= 1e-6 .* randn(size(X))
    end

    # --------------------------------------------------
    # Orthonormalize initial guess
    # --------------------------------------------------

    X = Matrix(qr(X).Q[:,1:nev])

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    rank(X)
    minimum(svdvals(X))

#    result = lobpcg(
#        H,
#        false,               # smallest eigenvalues
#        X;
#        maxiter = maxiter,
#        tol = tol
#    )


    vals, vecs = eigs(
        H;
        nev=nev,
        sigma=0.0,
        tol = 1e-5,
        which=:LM
    )
    #vals = result.λ
    #vecs = result.X

    order = sortperm(vals)

    vals = vals[order]
    vecs = vecs[:,order]
    #println("Eigenvalues: ", vals)
    if nev ==1
        return vals[1], vecs[:,1]
    else
        return vals, vecs
    end

end


function lowest_nonzero_mode(arr)

    length(arr) >= 4 ||
        throw(ArgumentError(
            "Need at least 4 eigenvalues, got $(length(arr))"
        ))

    # remove the 3 eigenvalues closest to zero
    remaining = sort(arr; by=abs)[4:end]

    target = minimum(remaining)

    return findfirst(==(target), arr)
end



using SparseArrays
using Base.Threads

using SparseArrays
using Base.Threads

function build_hessian_fast(
    positions,
    data,
    neighbors
)

    natoms = data["natoms"]
    cid    = data["cid"]

    lx = data["lx"]
    ly = data["ly"]
    lz = data["lz"]

    rank = 3 * natoms

    Nthreads  = Threads.maxthreadid()     # computed ONCE, reused below and at the reshape
    diagonals = zeros(3*Nthreads, rank)   # sized to match Nthreads, not a hardcoded 32

    npairs = size(neighbors,1)

    # 18 off-diagonal entries per pair
    n_off = 18 * npairs

    rows = Vector{Int}(undef, n_off)
    cols = Vector{Int}(undef, n_off)
    vals = Vector{Float64}(undef, n_off)

    # --------------------------------------------------
    # OFF-DIAGONAL TERMS
    # --------------------------------------------------

    Threads.@threads for p in 1:npairs

        i = neighbors[p,1]
        j = neighbors[p,2]

        dx = positions[i,1] - positions[j,1]
        dy = positions[i,2] - positions[j,2]
        dz = positions[i,3] - positions[j,3]

        dx -= lx * round(dx/lx)
        dy -= ly * round(dy/ly)
        dz -= lz * round(dz/lz)

        r2 = dx*dx + dy*dy + dz*dz
        r  = sqrt(r2)

        ti = cid[i]
        tj = cid[j]

        eps = epsilon_table[ti,tj]
        sig = sigma_table[ti,tj]

        mi = mass[ti]
        mj = mass[tj]

        offdiag = -1.0/sqrt(mi*mj)

        # `neighbors` is a Verlet/skin list padded beyond rc (see
        # build_neighbor_pairs), so it also contains pairs with
        # rc < r <= rc+0.5 that must NOT contribute to the Hessian.
        # Zeroing their components (rather than `continue`) keeps every
        # pair's fixed 18-slot block in rows/cols/vals written, since
        # this loop runs multi-threaded with a per-pair pointer and a
        # skipped write would leave that block as uninitialized memory.
        if r <= rc && r > 0

            sr2  = (sig*sig)/r2
            sr6  = sr2^3
            sr12 = sr6^2

            dV =
                -4.0*eps*(
                    12.0*sr12/r -
                    6.0*sr6/r
                )

            ddV =
                4.0*eps*(
                    12.0*13.0*sr12/r2 -
                    6.0*7.0*sr6/r2
                )

            fn0 = dV/r
            fn1 = ddV - fn0

            invr2 = 1.0/r2

            xx = fn1*dx*dx*invr2 + fn0
            yy = fn1*dy*dy*invr2 + fn0
            zz = fn1*dz*dz*invr2 + fn0

            xy = fn1*dx*dy*invr2
            xz = fn1*dx*dz*invr2
            yz = fn1*dy*dz*invr2
        else
            xx = yy = zz = xy = xz = yz = 0.0
        end

        comps = (
            (1,1,xx),
            (2,2,yy),
            (3,3,zz),
            (1,2,xy),
            (2,1,xy),
            (1,3,xz),
            (3,1,xz),
            (2,3,yz),
            (3,2,yz)
        )

        ptr = 18*(p-1) + 1

        @inbounds for (a,b,c) in comps

            rows[ptr] = 3*(i-1) + a
            cols[ptr] = 3*(j-1) + b
            vals[ptr] = offdiag*c
            ptr += 1

            rows[ptr] = 3*(j-1) + a
            cols[ptr] = 3*(i-1) + b
            vals[ptr] = offdiag*c
            ptr += 1

            tid = threadid()

            # contribution to atom i
            diagonals[
                3*(tid-1)+a,
                3*(i-1)+b
            ] += c / mi

            # contribution to atom j
            diagonals[
                3*(tid-1)+a,
                3*(j-1)+b
            ] += c / mj

        end
    end

    # --------------------------------------------------
    # reduce per-thread diagonal accumulators
    # --------------------------------------------------

    diag_blocks =
        dropdims(
            sum(
                reshape(diagonals, 3, Nthreads, rank),   # same Nthreads as the allocation above
                dims = 2
            ),
            dims = 2
        )
    vals_diag = reshape(diag_blocks, 9*natoms)

    col = collect(1:3*natoms)
    rows_diag = [v for i in 1:3:3*natoms for _ in 1:3 for v in i:i+2]
    cols_diag = repeat(col, inner=3)

    rows_all = [rows; rows_diag]
    cols_all = [cols; cols_diag]
    vals_all = [vals; vals_diag]

    H = sparse(rows_all, cols_all, vals_all, rank, rank)

    return H
end


using SparseArrays
using Base.Threads

function build_hessian_claude(
    positions,
    data,
    neighbors
)

    natoms = data["natoms"]
    cid    = data["cid"]

    lx = data["lx"]
    ly = data["ly"]
    lz = data["lz"]

    rank = 3 * natoms

    npairs = size(neighbors,1)

    # per pair: 9 comps * (Hij + Hji + raw contribution to Hii + raw contribution to Hjj)
    n_entries = 36 * npairs

    rows = Vector{Int}(undef, n_entries)
    cols = Vector{Int}(undef, n_entries)
    vals = Vector{Float64}(undef, n_entries)

    Threads.@threads for p in 1:npairs

        i = neighbors[p,1]
        j = neighbors[p,2]

        dx = positions[i,1] - positions[j,1]
        dy = positions[i,2] - positions[j,2]
        dz = positions[i,3] - positions[j,3]

        dx -= lx * round(dx/lx)
        dy -= ly * round(dy/ly)
        dz -= lz * round(dz/lz)

        r2 = dx*dx + dy*dy + dz*dz
        r  = sqrt(r2)

        ti = cid[i]
        tj = cid[j]

        eps = epsilon_table[ti,tj]
        sig = sigma_table[ti,tj]

        mi = mass[ti]
        mj = mass[tj]

        diag_i  = 1.0/mi
        diag_j  = 1.0/mj
        offdiag = -1.0/sqrt(mi*mj)

        # see build_hessian_fast: `neighbors` is a padded skin list, so
        # zero out (rather than skip) any pair beyond the real cutoff to
        # avoid leaving this pair's fixed-position slots uninitialized.
        if r <= rc && r > 0

            sr2  = (sig*sig)/r2
            sr6  = sr2^3
            sr12 = sr6^2

            dV =
                -4.0*eps*(
                    12.0*sr12/r -
                    6.0*sr6/r
                )

            ddV =
                4.0*eps*(
                    12.0*13.0*sr12/r2 -
                    6.0*7.0*sr6/r2
                )

            fn0 = dV/r
            fn1 = ddV - fn0

            invr2 = 1.0/r2

            xx = fn1*dx*dx*invr2 + fn0
            yy = fn1*dy*dy*invr2 + fn0
            zz = fn1*dz*dz*invr2 + fn0

            xy = fn1*dx*dy*invr2
            xz = fn1*dx*dz*invr2
            yz = fn1*dy*dz*invr2
        else
            xx = yy = zz = xy = xz = yz = 0.0
        end

        comps = (
            (1,1,xx),
            (2,2,yy),
            (3,3,zz),
            (1,2,xy),
            (2,1,xy),
            (1,3,xz),
            (3,1,xz),
            (2,3,yz),
            (3,2,yz)
        )

        ptr = 36*(p-1) + 1

        @inbounds for (a,b,c) in comps

            # Hij
            rows[ptr] = 3*(i-1) + a
            cols[ptr] = 3*(j-1) + b
            vals[ptr] = offdiag*c
            ptr += 1

            # Hji
            rows[ptr] = 3*(j-1) + a
            cols[ptr] = 3*(i-1) + b
            vals[ptr] = offdiag*c
            ptr += 1

            # raw contribution to Hii -- sparse() sums these across all
            # pairs that touch atom i, no manual reduction needed
            rows[ptr] = 3*(i-1) + a
            cols[ptr] = 3*(i-1) + b
            vals[ptr] = diag_i*c
            ptr += 1

            # raw contribution to Hjj
            rows[ptr] = 3*(j-1) + a
            cols[ptr] = 3*(j-1) + b
            vals[ptr] = diag_j*c
            ptr += 1

        end
    end

    H = sparse(rows, cols, vals, rank, rank)

    return H
end

using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using IterativeSolvers

"""
    eig_from_csv_lobpcg(H, csvfile)

Compute eigenpairs of sparse symmetric matrix H using
eigenvectors stored in csvfile as the initial block.

Assumes each COLUMN of the CSV file is an eigenvector.
"""


function eig_from_csv_lobpcg(H, csvfile=nothing)

    nev = 5  # desired number of eigenpairs

    if csvfile === nothing
        @info "No starting vectors provided, using random initialization."
        X0 = randn(size(H,1), nev)
    else
        X0 = Matrix(
            CSV.read(
                csvfile,
                DataFrame;
                header=false
            )
        )

        @assert size(X0,1) == size(H,1)

        # infer nev from supplied vectors
        nev = size(X0,2)
    end

    # orthonormalize initial block
    X0 = Matrix(qr(X0).Q[:, 1:nev])

    results = lobpcg(
        Symmetric(H),   # recommended for Hessians
        false,
        X0;
        maxiter=500,
        tol=1e-3
    )

    λ = results.λ
    V = results.X

    return λ, V
end



using IterativeSolvers
using LinearAlgebra

function track_mode(H, v_old)

    v_old ./= norm(v_old)

    X0 = reshape(v_old, :, 1)

    result = lobpcg(
        H,
        false,
        X0;
        maxiter = 20,
        tol = 1e-8
    )

    λ = result.λ[1]
    v = result.X[:,1]

    return λ, v
end

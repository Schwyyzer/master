#=
Replacement for the mode-tracking functions in eigenvalueDecomp.jl
(track_mode / eig_from_csv_lobpcg), which produce unreliable results.

Two concrete bugs in the existing track_mode(H, v_old):

  1. It calls `lobpcg(H, false, X0; ...)` on the raw sparse matrix, not
     `Symmetric(H)`. eig_from_csv_lobpcg (the other LOBPCG function in
     the same file) *does* wrap it as Symmetric, with the comment
     "# recommended for Hessians" -- track_mode just doesn't. LOBPCG's
     correctness depends on the operator being (and being recognized
     as) symmetric.
  2. It asks LOBPCG for the algebraically SMALLEST eigenvalue of the
     WHOLE matrix ("largest=false"), with no preconditioner. That's a
     fundamentally different question from "track the eigenvalue
     nearest to whatever v_old represents" -- the Hessian's smallest
     eigenvalues are always the ~3 near-zero translational modes (or,
     near a saddle, whatever is currently most unstable), which need
     not be anywhere near v_old's eigenvalue. Nothing in the plain
     lobpcg(A, false, X0) call keeps it anchored to v_old; only a good
     starting guess and a small number of iterations happen to keep it
     close some of the time, which matches the "sometimes flat out
     wrong" symptom.

track_modes() below fixes both: it wraps Symmetric(H), adds a Jacobi
(diagonal) preconditioner (matrix-vector products only -- no sparse LU
factorization, so this is much cheaper per call than a fresh ARPACK
shift-invert diagonalization), and -- the part that actually matters
for avoiding the phase-2 mode-jump problem -- matches each OUTPUT
eigenvector back to the INPUT column it has the highest overlap with,
instead of trusting eigenvalue order. A plain "take whichever is now
lowest" selection is exactly what silently jumps to a different mode
whenever some other part of the system becomes momentarily softer.
See test_mode_tracking.jl for a synthetic demonstration of exactly
that failure and confirmation that the overlap-based matching avoids
it.
=#

using LinearAlgebra
using SparseArrays
using IterativeSolvers

"""
    track_modes(H, X_prev; maxiter=100, tol=1e-8)

Track a block of eigenmodes of the symmetric sparse matrix `H`, warm
started from `X_prev` (an n×k matrix; columns need not be orthonormal
-- they're re-orthonormalized internally). Returns `(vals, vecs, overlaps)`
where column `i` of `vecs` is the mode matched to column `i` of
`X_prev` (not necessarily sorted by eigenvalue), and `overlaps[i] =
|<vecs[:,i], X_prev[:,i]>|` after orthonormalization, so you can tell
how confidently each column was tracked (close to 1 = confident;
noticeably below 1 = passing through a near-degeneracy, worth a
sanity check against a full diagonalization).
"""
function track_modes(H, X_prev::AbstractMatrix; maxiter::Int=100, tol::Real=1e-8)

    n, k = size(X_prev)
    Hs = Symmetric(H)

    X0 = Matrix(qr(X_prev).Q[:, 1:k])

    d = diag(H)
    d[abs.(d) .< 1e-10] .= 1.0
    Pinv = Diagonal(1.0 ./ d)

    result = lobpcg(
        Hs,
        false,      # smallest eigenvalues
        X0;
        P = Pinv,
        maxiter = maxiter,
        tol = tol,
    )

    vals_new = result.λ
    vecs_new = result.X

    # match each new eigenvector back to the old column it overlaps most
    # with (greedy assignment -- fine for the small k this is meant for)
    overlap = abs.(X0' * vecs_new)   # k x k

    assignment = Vector{Int}(undef, k)
    used = falses(k)
    for i in 1:k
        row = copy(overlap[i, :])
        row[used] .= -Inf
        j = argmax(row)
        assignment[i] = j
        used[j] = true
    end

    vals = vals_new[assignment]
    vecs = vecs_new[:, assignment]
    overlaps = [overlap[i, assignment[i]] for i in 1:k]

    return vals, vecs, overlaps
end

"""
    track_mode(H, v_old)

Single-mode (k=1) convenience wrapper around track_modes.

IMPORTANT, found by testing: this does NOT reliably survive an
eigenvalue crossing/near-degeneracy, even with the Symmetric wrap and
preconditioner fixed. With only one output vector there is nothing to
overlap-match against -- LOBPCG's "smallest eigenvalue" objective will
converge to the true global minimum given enough iterations regardless
of the warm start, so once some other mode becomes more unstable than
the one you're tracking, this returns THAT mode instead, silently.
(Confirmed with a synthetic avoided-crossing test: k=1 tracking jumps
to the wrong branch right after the crossing, exactly like the
original bug; a block of k>=5-8, tracked via track_modes directly and
matched by overlap, stays correct through the entire crossing.)

Kept only for call sites that truly have just one vector to hand in.
For actual mode-following in the ART loop (phase 1/2, calculate_moves),
call track_modes with a block of several eigenvectors (a similar block
size to whatever calculate_moves currently requests via full
diagonalization) and track the specific column that corresponds to
your mode of interest -- do not rely on this single-vector form there.
"""
function track_mode(H, v_old)
    v_old = v_old ./ norm(v_old)
    vals, vecs, overlaps = track_modes(H, reshape(v_old, :, 1))
    return vals[1], vecs[:, 1]
end

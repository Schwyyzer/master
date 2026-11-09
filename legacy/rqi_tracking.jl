#=
Cheap single-vector eigenvector tracker for phase 2 of the ART loop,
purpose-built for the case where the tracked mode is known to rotate by
only a degree or two between successive calls (confirmed empirically:
50/50 saddle points found over ~12h using the cold full-re-diagonalization
baseline, with max_phase2_angle_deg logged every step -- check that
column in your results.csv to see the actual distribution before
picking max_angle_deg below).

Unlike mode_tracking.jl's track_modes (general-purpose block LOBPCG,
needed to survive an eigenvalue crossing -- see its own docstring and
test_mode_tracking.jl for why a single vector can't do that safely),
this deliberately gives up that generality for speed: ONE preconditioned
Newton/Jacobi-Davidson-style correction step per call, using the same
abs()-floored Jacobi preconditioner validated in mode_tracking.jl (see
test_mode_tracking_precond.jl for why the floor needs abs() -- phase 2
operates where H has a negative eigenvalue, so diag(H) is not reliably
positive, and an unguarded 1/(d-lambda) can divide by a near-zero or
wrong-signed number). No block, no QR, no iterative eigensolver library
call at all -- one Hessian-vector product, one elementwise division, a
couple of dot products.

Because it only trusts a small rotation, it also enforces one: the
correction step is capped at `max_angle_deg`. If the natural correction
would exceed that, the tracker still returns a result but flags it
(`angle_deg >= max_angle_deg`, i.e. "I hit the guardrail, don't trust
this") so the caller can fall back to a full diagonalization for that
one iteration instead of silently accepting a large, unverified jump --
exactly the failure mode that made single-vector LOBPCG tracking unsafe
in the first place.
=#

using LinearAlgebra
using SparseArrays

"""
    track_eigenvector_step(H, v_old; max_angle_deg=10.0)

One preconditioned correction step tracking the eigenvector nearest
`v_old` (assumed already ~unit norm) onto the (possibly slightly
different) matrix `H`. Returns `(eigenvalue, v_new, angle_deg)` where
`angle_deg` is the actual angle turned -- compare it against what you
expect (~1-2 degrees) and treat `angle_deg >= max_angle_deg` as "this
call doesn't trust its own correction, redo with a full
diagonalization."

Math: one step of preconditioned residual correction on the eigenvector
equation Hv = lambda*v, in the spirit of Jacobi-Davidson / preconditioned
inverse iteration, but skipping their subspace (Rayleigh-Ritz)
combination step in favor of directly applying the correction -- a
reasonable trade only because `v_old` is already assumed close to the
answer (a fresh subspace projection buys little when the starting guess
is this good, and costs an extra QR/small eigenproblem every call).

    lambda = v_old' H v_old                   (Rayleigh quotient)
    r      = H v_old - lambda v_old            (residual; -> 0 at a true
                                                 eigenvector)
    c      = r ./ max(|diag(H) - lambda|, floor)   (Jacobi-preconditioned
                                                      correction direction)
    c      = c - (v_old' c) v_old              (keep it perpendicular to
                                                 v_old, like a Newton step
                                                 on the unit sphere)
    [clip ||c|| so the resulting rotation angle <= max_angle_deg]
    v_new  = normalize(v_old - c)
"""
function track_eigenvector_step(H, v_old::AbstractVector; max_angle_deg::Real=10.0)
    v_old = v_old ./ norm(v_old)

    Hv = H * v_old
    λ = dot(v_old, Hv)
    r = Hv .- λ .* v_old

    d = diag(H)
    # Relative floor (scaled to |lambda|, not a fixed absolute constant)
    # so this doesn't under-floor on a large system where |lambda| itself
    # may be large, and abs() so a negative (d - lambda) -- expected in
    # phase 2's regime -- can't flip the correction's sign or blow it up.
    floor_val = max(1e-6 * max(1.0, abs(λ)), 1e-10)
    denom = max.(abs.(d .- λ), floor_val)
    c = r ./ denom
    c = c .- dot(v_old, c) .* v_old

    cn = norm(c)
    if cn > 0
        max_ratio = tan(deg2rad(max_angle_deg))
        if cn > max_ratio
            c = c .* (max_ratio / cn)
        end
    end

    v_new = v_old .- c
    v_new ./= norm(v_new)

    angle_deg = rad2deg(acos(clamp(dot(v_old, v_new), -1.0, 1.0)))

    Hv_new = H * v_new
    λ_new = dot(v_new, Hv_new)

    return λ_new, v_new, angle_deg
end

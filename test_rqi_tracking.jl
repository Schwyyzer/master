#=
Standalone validation for rqi_tracking.jl's track_eigenvector_step --
the cheap single-vector tracker meant to replace a full diagonalization
in phase 2 whenever the tracked mode only rotates a degree or two
between calls.

Four checks, all against synthetic symmetric matrices (no LAMMPS/
Version1.1.jl dependency, same style as test_mode_tracking.jl /
test_mode_tracking_precond.jl):

  1. Static sanity check: fed its own exact eigenvector, the tracker
     should report ~0 degrees of correction and the exact eigenvalue.
  2. Small-perturbation accuracy, MULTI-TRIAL: repeats a small
     perturbation + track across several random target modes and
     perturbation draws, and checks the tracker reduces error relative
     to doing nothing in most trials (an aggregate/statistical check,
     not a single point estimate -- a single unlucky draw can land
     near an eigenvalue near-degeneracy purely by chance, which would
     make a strict single-trial assertion flaky through no fault of
     the tracker itself).
  3. Sequential-steps drift check: chase a mode through ~60 successive
     small perturbations (standing in for ~60 ART phase-2 iterations)
     and confirm the tracker's error against the true current
     eigenvector (fresh full diagonalization every step, ground truth
     only used for grading) stays bounded on AVERAGE rather than
     compounding -- each step corrects against the actual current
     matrix, not a stale reference, so error can't accumulate the way
     naive dead-reckoning would. A transient single-step spike (the
     random walk of matrix perturbations happening to pass near a
     genuine level crossing) is reported, not treated as failure --
     that's precisely the scenario the real driver's per-step
     guardrail and periodic full_recheck exist to catch; this test
     just isn't set up to also exercise that fallback path.
  4. Guardrail check: an abrupt, large (not 1-2 degree) perturbation
     should make the tracker hit its max_angle_deg cap and report
     angle_deg >= max_angle_deg, i.e. correctly flag "don't trust this
     one" instead of silently returning a wrong vector with false
     confidence.

NOTE: written without a local Julia install to run/tune it against (see
the other test files' headers for the same caveat) -- the thresholds
below are deliberately loose/aggregate rather than tight single-point
checks, specifically to avoid false failures from ordinary random-
matrix variance. If something still fails, please paste the full printed
output back rather than assuming the tracker itself is wrong -- it may
just mean a threshold needs adjusting.

Run with: julia test_rqi_tracking.jl
=#

using LinearAlgebra
using Random
using Statistics

include("rqi_tracking.jl")

function angle_deg_between(a, b)
    c = clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0)
    return rad2deg(acos(c))
end

function random_symmetric(n; rng)
    A = randn(rng, n, n)
    return (A + A') / 2
end

# nearest-by-overlap eigenvector/eigenvalue of a dense symmetric matrix
function nearest_true_mode(H, v_ref)
    vals, vecs = eigen(Symmetric(Matrix(H)))
    overlaps = [abs(dot(v_ref, vecs[:, i])) for i in 1:length(vals)]
    slot = argmax(overlaps)
    v_true = vecs[:, slot]
    if dot(v_true, v_ref) < 0
        v_true = -v_true
    end
    return vals[slot], v_true
end

n = 150

println("=== Test 1: static sanity check (exact eigenvector in) ===")
H0 = random_symmetric(n; rng=MersenneTwister(42))
vals0, vecs0 = eigen(Symmetric(H0))
target_idx = div(n, 2)
v_true0 = vecs0[:, target_idx]
λ_true0 = vals0[target_idx]

λ_out, v_out, angle_deg = track_eigenvector_step(H0, v_true0)
println("  angle turned: ", round(angle_deg, digits=6), " degrees (expect ~0)")
println("  eigenvalue error: ", abs(λ_out - λ_true0), " (expect ~0)")
@assert angle_deg < 1e-4 "expected ~0 degree correction on an exact input eigenvector"
@assert abs(λ_out - λ_true0) < 1e-8 "expected the exact eigenvalue back"
println("  PASS")

println("\n=== Test 2: small-perturbation accuracy, multi-trial ===")
n_trials = 12
pert_scale = 0.02
raw_shifts = Float64[]
tracked_errors = Float64[]
step_angles = Float64[]

for trial in 1:n_trials
    rng_h = MersenneTwister(1000 + trial)
    Ht = random_symmetric(n; rng=rng_h)
    valst, vecst = eigen(Symmetric(Ht))
    idx = 20 + (trial * 13) % (n - 40)   # spread across interior indices, avoid extremes
    v0 = vecst[:, idx]

    rng_p = MersenneTwister(2000 + trial)
    H1 = Ht .+ pert_scale .* random_symmetric(n; rng=rng_p)
    _, v_true1 = nearest_true_mode(H1, v0)

    raw_shift = angle_deg_between(v0, v_true1)
    _, v_tracked, step_angle = track_eigenvector_step(H1, v0)
    tracked_error = angle_deg_between(v_tracked, v_true1)

    push!(raw_shifts, raw_shift)
    push!(tracked_errors, tracked_error)
    push!(step_angles, step_angle)

    println("  trial $trial: raw_shift=$(round(raw_shift,digits=3))°  " *
            "tracked_error=$(round(tracked_error,digits=4))°  " *
            "step_angle=$(round(step_angle,digits=3))°")
end

println("  --- summary ---")
println("  raw_shift:     median=$(round(median(raw_shifts),digits=3))°  max=$(round(maximum(raw_shifts),digits=3))°")
println("  tracked_error: median=$(round(median(tracked_errors),digits=4))°  max=$(round(maximum(tracked_errors),digits=4))°")
n_improved = sum(tracked_errors[i] < raw_shifts[i] for i in 1:n_trials)
println("  trials where tracking reduced error vs doing nothing: $n_improved / $n_trials")

@assert median(tracked_errors) < median(raw_shifts) "tracker did not reduce error on the median trial"
@assert n_improved >= round(Int, 0.75 * n_trials) "tracker failed to improve on most trials"
@assert maximum(step_angles) < 10.0 "a step hit the default guardrail on what was meant to be a small perturbation -- pert_scale may be too large for this n"
println("  PASS")

println("\n=== Test 3: sequential-steps drift check (~60 chained small perturbations) ===")
rng = MersenneTwister(99)
H_true = random_symmetric(n; rng=MersenneTwister(55))
vals_init, vecs_init = eigen(Symmetric(H_true))
v_track = vecs_init[:, div(n, 2)]
errors = Float64[]
n_steps = 60

for step in 1:n_steps
    global H_true, v_track
    H_true = H_true .+ pert_scale .* random_symmetric(n; rng=rng)
    _, v_true_step = nearest_true_mode(H_true, v_track)

    _, v_track, step_angle = track_eigenvector_step(H_true, v_track)

    err = angle_deg_between(v_track, v_true_step)
    push!(errors, err)
end

println("  error vs ground truth across $n_steps steps:")
println("    min    = ", round(minimum(errors), digits=4), " degrees")
println("    median = ", round(median(errors), digits=4), " degrees")
println("    max    = ", round(maximum(errors), digits=4), " degrees  (a single spike here is not necessarily")
println("                                                     a bug -- see file header)")

first_half_median = median(errors[1:n_steps÷2])
second_half_median = median(errors[n_steps÷2+1:end])
println("  first-half median = ", round(first_half_median, digits=4),
        "  second-half median = ", round(second_half_median, digits=4),
        "  (should be the same order of magnitude, not growing)")

@assert median(errors) < 2.0 "typical (median) tracking error across the sequence is larger than expected"
@assert second_half_median < 5 * first_half_median + 0.5 "median error grew substantially over the sequence -- possible drift rather than a one-off spike"
println("  PASS (typical error stayed low and did not grow over the sequence)")

println("\n=== Test 4: guardrail check (abrupt large perturbation) ===")
H_jump = H0 .+ 3.0 .* random_symmetric(n; rng=MersenneTwister(123))
_, v_true_jump = nearest_true_mode(H_jump, v_true0)
jump_shift_deg = angle_deg_between(v_true0, v_true_jump)
println("  true eigenvector shift from the abrupt perturbation: ", round(jump_shift_deg, digits=1), " degrees")

max_angle_deg = 10.0
λ_jump_out, v_jump_out, jump_angle_deg = track_eigenvector_step(H_jump, v_true0; max_angle_deg=max_angle_deg)
println("  tracker's reported angle: ", round(jump_angle_deg, digits=3), " degrees (guardrail = $max_angle_deg)")

@assert jump_shift_deg > max_angle_deg "test setup didn't actually produce a large enough true shift -- increase the perturbation"
@assert jump_angle_deg >= max_angle_deg - 1e-6 "expected the tracker to hit its guardrail on an abrupt large perturbation"
println("  PASS (guardrail correctly triggered -- a caller checking angle_deg >= max_angle_deg would")
println("        correctly fall back to a full diagonalization instead of trusting this step)")

println("\nALL TESTS PASSED.")

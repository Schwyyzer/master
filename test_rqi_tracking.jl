#=
Standalone validation for rqi_tracking.jl's track_eigenvector_step --
the cheap single-vector tracker meant to replace a full diagonalization
in phase 2 whenever the tracked mode only rotates a degree or two
between calls.

REVISION NOTE: the first version of this test used dense, unstructured
random (GOE-like) matrices, and Test 2 failed -- tracked_error came out
*larger* than doing nothing in 11/12 trials. That wasn't a bug in the
tracker: track_eigenvector_step uses a Jacobi (diagonal) preconditioner,
which only approximates H^-1 well when H is close to diagonally
dominant. A real molecular Hessian is (LJ interactions are short-ranged,
so each atom's own curvature dominates its row); a dense matrix with
iid N(0,1) entries is not (every off-diagonal entry is just as large as
the diagonal ones -- there's no structure for a diagonal preconditioner
to exploit). Testing a Jacobi-preconditioned method against a matrix
type it was never meant for was the wrong test. This version builds an
actual synthetic LJ particle system instead (same approach as
test_mode_tracking.jl: make_synthetic_system / mini_force / mini_energy
/ mini_relax!, real build_hessian_fast), so the Hessian has the
structure the preconditioner is actually designed around.

Four checks:

  1. Static sanity check: fed its own exact eigenvector, the tracker
     should report ~0 degrees of correction and the exact eigenvalue.
  2. Small-perturbation accuracy, multi-trial: several random position
     nudges of the size test_mode_tracking.jl used for "one ART-step-
     sized move" (delta ~ N(0, 0.01) per coordinate), tracking a few
     different interior modes, checking the tracker lands closer to
     the true new mode (fresh dense diagonalization, overlap-matched)
     than doing nothing.
  3. Sequential-steps drift check: chase a mode through ~40 successive
     small position nudges (a real phase-2-shaped trajectory) and
     confirm the tracker's error against the true current eigenvector
     stays bounded rather than compounding.
  4. Guardrail check: one abrupt, much larger position jump should
     produce a true shift beyond max_angle_deg and make the tracker
     report hitting its cap, rather than silently returning a
     confidently-wrong vector.

NOTE: written without a local Julia install to run/tune it against
(same caveat as the other test files here). If a threshold still
trips, please paste the full printed output back -- that's useful
diagnostic information either way, not necessarily a sign the tracker
itself is broken.

Run with: julia test_rqi_tracking.jl
=#

include("config.jl")
include("neighbor_pairs.jl")
include("Hessian.jl")
include("rqi_tracking.jl")

using LinearAlgebra
using SparseArrays
using Random
using Statistics

# ============================================================
# Synthetic LJ system (verbatim approach from test_mode_tracking.jl)
# ============================================================
function make_synthetic_system(N::Int, box::NTuple{3,Float64}; seed::Int=1, min_dist::Float64=1.0)
    rng = MersenneTwister(seed)
    positions = zeros(N, 3)
    boxv = [box[1], box[2], box[3]]
    placed = 0
    attempts = 0
    while placed < N
        attempts += 1
        attempts > 200_000 && error("could not place $N atoms with min_dist=$min_dist in this box")
        cand = rand(rng, 3) .* boxv
        ok = true
        for i in 1:placed
            d = positions[i, :] .- cand
            d .-= boxv .* round.(d ./ boxv)
            if norm(d) < min_dist
                ok = false
                break
            end
        end
        if ok
            placed += 1
            positions[placed, :] = cand
        end
    end
    cid = rand(rng, 1:2, N)
    data = Dict(
        "natoms" => N, "id" => collect(1:N), "cid" => cid,
        "x" => positions[:, 1], "y" => positions[:, 2], "z" => positions[:, 3],
        "lx" => box[1], "ly" => box[2], "lz" => box[3],
    )
    return positions, data
end

function mini_force(positions, cid, neighbors, box)
    N = size(positions, 1)
    F = zeros(N, 3)
    for p in 1:size(neighbors, 1)
        i, j = neighbors[p, 1], neighbors[p, 2]
        dx = positions[i, 1] - positions[j, 1]
        dy = positions[i, 2] - positions[j, 2]
        dz = positions[i, 3] - positions[j, 3]
        dx -= box[1] * round(dx / box[1])
        dy -= box[2] * round(dy / box[2])
        dz -= box[3] * round(dz / box[3])
        r2 = dx^2 + dy^2 + dz^2
        r = sqrt(r2)
        if r <= rc && r > 0
            ti, tj = cid[i], cid[j]
            eps = epsilon_table[ti, tj]
            sig = sigma_table[ti, tj]
            sig6 = sig^6
            sig12 = sig6^2
            pref = 24 * eps * (2 * sig12 / r2^7 - sig6 / r2^4)
            F[i, 1] += pref * dx; F[i, 2] += pref * dy; F[i, 3] += pref * dz
            F[j, 1] -= pref * dx; F[j, 2] -= pref * dy; F[j, 3] -= pref * dz
        end
    end
    return F
end

function mini_energy(positions, cid, neighbors, box)
    e = 0.0
    for p in 1:size(neighbors, 1)
        i, j = neighbors[p, 1], neighbors[p, 2]
        dx = positions[i, 1] - positions[j, 1]
        dy = positions[i, 2] - positions[j, 2]
        dz = positions[i, 3] - positions[j, 3]
        dx -= box[1] * round(dx / box[1])
        dy -= box[2] * round(dy / box[2])
        dz -= box[3] * round(dz / box[3])
        r2 = dx^2 + dy^2 + dz^2
        r = sqrt(r2)
        if r <= rc && r > 0
            ti, tj = cid[i], cid[j]
            eps = epsilon_table[ti, tj]
            sig = sigma_table[ti, tj]
            sig6 = sig^6
            sig12 = sig6^2
            e += 4 * eps * (sig12 / r^12 - sig6 / r^6)
        end
    end
    return e
end

function mini_relax!(positions, cid, box; steps::Int=300, step0::Float64=0.0005)
    step = step0
    for _ in 1:steps
        pairs = build_neighbor_pairs(positions, rc, box)
        F = mini_force(positions, cid, pairs, box)
        maximum(norm.(eachrow(F))) < 1e-3 && break
        trial = mod.(positions .+ step .* F, [box[1] box[2] box[3]])
        if mini_energy(trial, cid, pairs, box) < mini_energy(positions, cid, pairs, box)
            positions .= trial
            step *= 1.05
        else
            step *= 0.5
            step < 1e-10 && break
        end
    end
    return positions
end

function angle_deg_between(a, b)
    c = clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0)
    return rad2deg(acos(c))
end

# dense ground-truth eigenpair nearest v_ref by overlap
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

# build the Hessian at a given position perturbation `delta` away from
# the relaxed base positions
function hessian_at(positions_base, data_base, boxtuple, delta)
    positions = positions_base .+ delta
    data = deepcopy(data_base)
    data["x"] = positions[:, 1]; data["y"] = positions[:, 2]; data["z"] = positions[:, 3]
    pairs = build_neighbor_pairs(positions, rc, boxtuple)
    H = build_hessian_fast(positions, data, pairs)
    return H, positions
end

N = 60
box = (6.0, 6.0, 6.0)
positions0, data0 = make_synthetic_system(N, box; seed=1)
boxtuple = (data0["lx"], data0["ly"], data0["lz"])
mini_relax!(positions0, data0["cid"], boxtuple)
data0["x"] = positions0[:, 1]; data0["y"] = positions0[:, 2]; data0["z"] = positions0[:, 3]

pairs0 = build_neighbor_pairs(positions0, rc, boxtuple)
H0 = build_hessian_fast(positions0, data0, pairs0)
@assert maximum(abs.(H0 - H0')) < 1e-8 "sanity check failed: Hessian isn't symmetric"

dense0 = Matrix(H0)
vals0_all, vecs0_all = eigen(Symmetric(dense0))
order0 = sortperm(vals0_all)
vals0_all = vals0_all[order0]; vecs0_all = vecs0_all[:, order0]
println("synthetic system: N=$N, npairs=$(size(pairs0,1)), rank=$(size(H0,1))")
println("lowest 10 eigenvalues (dense ground truth): ", round.(vals0_all[1:10], digits=5))

first_nonzero_idx = findfirst(v -> abs(v) > 1e-6, vals0_all)
println("skipping $(first_nonzero_idx-1) near-zero mode(s); interior modes start at index $first_nonzero_idx\n")

println("=== Test 1: static sanity check (exact eigenvector in) ===")
target_idx = first_nonzero_idx
v_true0 = vecs0_all[:, target_idx]
λ_true0 = vals0_all[target_idx]

λ_out, v_out, angle_deg = track_eigenvector_step(H0, v_true0)
println("  angle turned: ", round(angle_deg, digits=6), " degrees (expect ~0)")
println("  eigenvalue error: ", abs(λ_out - λ_true0), " (expect ~0)")
@assert angle_deg < 1e-4 "expected ~0 degree correction on an exact input eigenvector"
@assert abs(λ_out - λ_true0) < 1e-8 "expected the exact eigenvalue back"
println("  PASS")

println("\n=== Test 2: small-perturbation accuracy, multi-trial ===")
println("(delta ~ N(0, 0.01) per coordinate -- same 'one ART-step-sized move' scale")
println(" test_mode_tracking.jl used)")
n_trials = 10
raw_shifts = Float64[]
tracked_errors = Float64[]
step_angles = Float64[]

for trial in 1:n_trials
    idx = first_nonzero_idx + (trial - 1) % 5   # cycle through a few interior modes
    v0 = vecs0_all[:, idx]

    rng_p = MersenneTwister(500 + trial)
    delta = randn(rng_p, N, 3) .* 0.01
    H1, _ = hessian_at(positions0, data0, boxtuple, delta)

    _, v_true1 = nearest_true_mode(H1, v0)
    raw_shift = angle_deg_between(v0, v_true1)

    _, v_tracked, step_angle = track_eigenvector_step(H1, v0)
    tracked_error = angle_deg_between(v_tracked, v_true1)

    push!(raw_shifts, raw_shift)
    push!(tracked_errors, tracked_error)
    push!(step_angles, step_angle)

    println("  trial $trial (mode idx=$idx): raw_shift=$(round(raw_shift,digits=3))°  " *
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
@assert maximum(step_angles) < 10.0 "a step hit the default guardrail on what was meant to be a small perturbation"
println("  PASS")

println("\n=== Test 3: sequential-steps drift check (~40 chained small perturbations) ===")
rng = MersenneTwister(99)
positions_track = copy(positions0)
v_track = copy(v_true0)
errors = Float64[]
n_steps = 40
step_delta_scale = 0.005   # smaller than Test 2's single-shot 0.01, so 40 of
                            # them chain up to a comparable total wander without
                            # any single step being a big jump

for step in 1:n_steps
    global positions_track, v_track
    delta = randn(rng, N, 3) .* step_delta_scale
    H_step, positions_track = hessian_at(positions_track, data0, boxtuple, delta)

    _, v_true_step = nearest_true_mode(H_step, v_track)
    _, v_track, step_angle = track_eigenvector_step(H_step, v_track)

    err = angle_deg_between(v_track, v_true_step)
    push!(errors, err)
end

println("  error vs ground truth across $n_steps steps:")
println("    min    = ", round(minimum(errors), digits=4), " degrees")
println("    median = ", round(median(errors), digits=4), " degrees")
println("    max    = ", round(maximum(errors), digits=4), " degrees")

first_half_median = median(errors[1:n_steps÷2])
second_half_median = median(errors[n_steps÷2+1:end])
println("  first-half median = ", round(first_half_median, digits=4),
        "  second-half median = ", round(second_half_median, digits=4),
        "  (should be the same order of magnitude, not growing)")

@assert median(errors) < 2.0 "typical (median) tracking error across the sequence is larger than expected"
@assert second_half_median < 5 * first_half_median + 0.5 "median error grew substantially over the sequence -- possible drift rather than a one-off spike"
println("  PASS (typical error stayed low and did not grow over the sequence)")

println("\n=== Test 4: guardrail check (abrupt large perturbation) ===")
rng_jump = MersenneTwister(123)
delta_jump = randn(rng_jump, N, 3) .* 0.5   # 50x Test 2's scale -- an abrupt, not
                                             # gradual, change
H_jump, _ = hessian_at(positions0, data0, boxtuple, delta_jump)
_, v_true_jump = nearest_true_mode(H_jump, v_true0)
jump_shift_deg = angle_deg_between(v_true0, v_true_jump)
println("  true eigenvector shift from the abrupt perturbation: ", round(jump_shift_deg, digits=1), " degrees")

max_angle_deg = 10.0
λ_jump_out, v_jump_out, jump_angle_deg = track_eigenvector_step(H_jump, v_true0; max_angle_deg=max_angle_deg)
println("  tracker's reported angle: ", round(jump_angle_deg, digits=3), " degrees (guardrail = $max_angle_deg)")

@assert jump_shift_deg > max_angle_deg "test setup didn't actually produce a large enough true shift -- increase delta_jump's scale"
@assert jump_angle_deg >= max_angle_deg - 1e-6 "expected the tracker to hit its guardrail on an abrupt large perturbation"
println("  PASS (guardrail correctly triggered -- a caller checking angle_deg >= max_angle_deg would")
println("        correctly fall back to a full diagonalization instead of trusting this step)")

println("\nALL TESTS PASSED.")

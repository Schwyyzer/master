#=
Self-contained test/validation script for mode_tracking.jl.

Run with:  julia test_mode_tracking.jl

Does NOT need your real LAMMPS file -- it builds a small synthetic LJ
configuration itself, and a separate abstract test matrix, so it runs
in seconds. Three tests:

  1. Sanity check: on a single static Hessian, do track_modes' results
     match a full dense eigendecomposition?
  2. Perturb-and-retrack: build H1, get its eigenmodes, nudge the
     positions slightly (like one ART step), build H2, and check
     whether OLD track_mode and NEW track_modes each correctly
     re-find the continuation of the same mode in H2 (ground truth
     from a fresh dense diagonalization of H2, matched by overlap).
  3. Synthetic avoided crossing: a hand-built matrix where two
     eigenvalues swap order as a parameter t sweeps 0->1. Compares
     "naive: always take the algebraically lowest eigenvalue" (what a
     cold full re-diagonalization + blind lowest-mode selection does)
     against track_modes' overlap-based tracking, through the
     crossing. This is the exact mechanism behind the phase-2
     mode-jump failures seen on the real system.

Each test prints PASS/FAIL plus the numbers behind the verdict.
=#

include("config.jl")
include("neighbor_pairs.jl")
include("Hessian.jl")
include("eigenvalueDecomp.jl")   # old, buggy track_mode -- for contrast
include("mode_tracking.jl")      # new track_modes / track_mode

using LinearAlgebra
using SparseArrays
using Random

const PASS = "PASS"
const FAIL = "FAIL"
verdict(ok) = ok ? PASS : FAIL

# ============================================================
# Build a small synthetic LJ configuration (no external file needed)
# ============================================================
function make_synthetic_system(N::Int, box::NTuple{3,Float64}; seed::Int=1, min_dist::Float64=0.85)
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

N = 60
box = (6.0, 6.0, 6.0)   # denser than the real system, so the test is well-connected
                        # (and meaningful) whatever `rc` you currently have set in config.jl
positions, data = make_synthetic_system(N, box; seed=1)
boxtuple = (data["lx"], data["ly"], data["lz"])
pairs = build_neighbor_pairs(positions, rc, boxtuple)
H1 = build_hessian_fast(positions, data, pairs)
@assert maximum(abs.(H1 - H1')) < 1e-8 "sanity check failed: Hessian isn't symmetric"
println("synthetic system: N=$N, npairs=$(size(pairs,1)), rank=$(size(H1,1))\n")

# dense ground truth for H1
dense1 = Matrix(H1)
vals1_all, vecs1_all = eigen(Symmetric(dense1))
order1 = sortperm(vals1_all)
vals1_all = vals1_all[order1]; vecs1_all = vecs1_all[:, order1]
println("H1 lowest 10 eigenvalues (dense ground truth): ", round.(vals1_all[1:10], digits=5))

# ============================================================
# TEST 1: static sanity check
# ============================================================
println("\n=== TEST 1: track_modes vs dense ground truth on a static Hessian ===")
# skip the (near-)zero translational modes, however many there are, rather
# than assuming exactly 3 -- pick the first eigenvalue clearly away from zero
target_idx = findfirst(v -> abs(v) > 1e-6, vals1_all)
println("skipping $(target_idx-1) near-zero mode(s); tracking index $target_idx")
v_target = vecs1_all[:, target_idx]
lam_target = vals1_all[target_idx]

vals_t, vecs_t, overlaps_t = track_modes(H1, reshape(v_target, :, 1); maxiter=200, tol=1e-10)
ok1 = isapprox(vals_t[1], lam_target; atol=1e-4) && overlaps_t[1] > 0.999
println("target eigenvalue=$(round(lam_target,digits=6))  track_modes result=$(round(vals_t[1],digits=6))  overlap=$(round(overlaps_t[1],digits=6))")
println(verdict(ok1))

# ============================================================
# TEST 2: perturb positions slightly, re-track vs dense ground truth,
#         old track_mode vs new track_modes
# ============================================================
println("\n=== TEST 2: re-tracking after a small perturbation (one ART-step-sized move) ===")
rng2 = MersenneTwister(2)
delta = randn(rng2, N, 3) .* 0.01
positions2 = positions .+ delta
data2 = deepcopy(data)
data2["x"] = positions2[:, 1]; data2["y"] = positions2[:, 2]; data2["z"] = positions2[:, 3]
pairs2 = build_neighbor_pairs(positions2, rc, boxtuple)
H2 = build_hessian_fast(positions2, data2, pairs2)
@assert maximum(abs.(H2 - H2')) < 1e-8

dense2 = Matrix(H2)
vals2_all, vecs2_all = eigen(Symmetric(dense2))
order2 = sortperm(vals2_all)
vals2_all = vals2_all[order2]; vecs2_all = vecs2_all[:, order2]

overlaps_true = [abs(dot(v_target, vecs2_all[:, i])) for i in 1:10]
best_true_idx = argmax(overlaps_true)
true_answer_vec = vecs2_all[:, best_true_idx]
true_answer_val = vals2_all[best_true_idx]
println("H2 lowest 10 eigenvalues (dense ground truth): ", round.(vals2_all[1:10], digits=5))
println("true continuation of the tracked mode: eigenvalue=$(round(true_answer_val,digits=5)) (index $best_true_idx among lowest 10)")

# old (buggy) track_mode
λ_old, v_old_result = track_mode(H2, copy(v_target))
overlap_old = abs(dot(v_old_result, true_answer_vec))
ok2_old = isapprox(λ_old, true_answer_val; atol=0.05) && overlap_old > 0.9
println("\nOLD track_mode (no Symmetric wrap, no preconditioner, largest=false on raw H):")
println("  result eigenvalue=$(round(λ_old,digits=5))  overlap_with_true_answer=$(round(overlap_old,digits=5))")
println("  ", verdict(ok2_old))

# new track_modes
vals_new, vecs_new, overlaps_new = track_modes(H2, reshape(v_target, :, 1); maxiter=200, tol=1e-10)
overlap_new = abs(dot(vecs_new[:, 1], true_answer_vec))
ok2_new = isapprox(vals_new[1], true_answer_val; atol=0.01) && overlap_new > 0.95
println("\nNEW track_modes (Symmetric + Jacobi preconditioner + overlap matching):")
println("  result eigenvalue=$(round(vals_new[1],digits=5))  overlap_with_true_answer=$(round(overlap_new,digits=5))")
println("  ", verdict(ok2_new))

# ============================================================
# TEST 3: synthetic avoided crossing -- the actual mode-jump mechanism
# ============================================================
println("\n=== TEST 3: avoided crossing (the mode-jump failure mode, isolated) ===")
println("Two eigenvalues cross as t: 0 -> 1 (index 5 goes -1 -> +1, index 6 goes +1 -> -1).")
println("A cold full-diagonalization + \"take the lowest\" selection is expected to jump")
println("branches right after the crossing; overlap-based track_modes should not.\n")

n_abs = 50
rng3 = MersenneTwister(3)
base_vals = sort(rand(rng3, n_abs) .* 9 .+ 1)
Qfull, _ = qr(randn(rng3, n_abs, n_abs))
Qfull = Matrix(Qfull)

function H_at(t::Float64)
    vals = copy(base_vals)
    vals[6] = -1.0 + 2.0 * t   # rises from -1 to +1
    vals[7] = 1.0 - 2.0 * t    # falls from +1 to -1 -- crosses vals[6] at t=0.5
    theta = 0.3 * sin(pi * t)
    c, s = cos(theta), sin(theta)
    Qt = copy(Qfull)
    v6, v7 = Qfull[:, 6], Qfull[:, 7]
    Qt[:, 6] = c .* v6 .- s .* v7
    Qt[:, 7] = s .* v6 .+ c .* v7
    return Symmetric(Qt * Diagonal(vals) * Qt'), vals, Qt
end

H0, _, Q0 = H_at(0.0)
v_prev = Q0[:, 6]

all_ok = true
for t in 0.0:0.1:1.0
    H, vals_true, Qt = H_at(t)
    true_target_vec = Qt[:, 6]   # by construction, the smooth continuation

    w, V = eigen(H)
    naive_idx = argmin(w)
    naive_vec = V[:, naive_idx]
    naive_matches_true = abs(dot(naive_vec, true_target_vec)) > 0.9

    vals_tr, vecs_tr, overlaps_tr = track_modes(sparse(Matrix(H)), reshape(v_prev, :, 1); maxiter=200, tol=1e-10)
    tracked_vec = vecs_tr[:, 1]
    tracked_overlap_true = abs(dot(tracked_vec, true_target_vec))

    println("t=$(round(t,digits=1))  true_eig=$(round(-1+2*t,digits=3))  " *
            "naive(global-min)_eig=$(round(w[naive_idx],digits=3)) matches_true=$naive_matches_true  " *
            "tracked_eig=$(round(vals_tr[1],digits=3)) overlap_with_true=$(round(tracked_overlap_true,digits=3))")

    global all_ok
    if tracked_overlap_true < 0.9
        all_ok = false
    end
    global v_prev = tracked_vec
end

println("\ntrack_modes stayed on the correct branch through the entire crossing: ", verdict(all_ok))

println("\n=== SUMMARY ===")
println("Test 1 (static sanity check):        ", verdict(ok1))
println("Test 2 (old track_mode after perturb): ", verdict(ok2_old), "   <- expect this to FAIL")
println("Test 2 (new track_modes after perturb): ", verdict(ok2_new))
println("Test 3 (avoided crossing, tracking):   ", verdict(all_ok))

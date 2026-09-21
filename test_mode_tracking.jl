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
     three things at each t: (a) naive full-diagonalization + "take
     the lowest" selection, (b) single-vector (k=1) track_modes, and
     (c) block (k=8) track_modes -- through the crossing.

     IMPORTANT, found in the first round of testing: (a) and (b) BOTH
     fail right after the crossing, and for the same underlying
     reason -- with only one output vector there's nothing to
     overlap-match against, so LOBPCG's "find the smallest
     eigenvalue" objective just converges to the new true global
     minimum once some other mode overtakes yours, regardless of the
     warm start. Only (c), a block with more than one vector, gives
     overlap-matching something to actually disambiguate. This is why
     track_mode (singular) now carries an explicit warning in
     mode_tracking.jl, and why any real use in the ART loop should go
     through track_modes with a block of several vectors, not the
     single-vector wrapper.

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

# Minimal standalone LJ force + steepest-descent relax, so this script
# doesn't need to include Version1.1.jl (and its BenchmarkTools
# dependency) just to knock down the worst overlaps from random
# placement. Doesn't need to be precise -- just enough that the test
# Hessian has genuine near-zero translational modes and modest-sized
# nonzero eigenvalues, instead of the huge, pathological curvatures a
# raw random packing produces.
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

# Same adaptive accept/reject step-size scheme as the project's own
# relax() (grow on success, shrink on failure) -- proven to converge
# well within a modest step budget on the real system.
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

N = 60
box = (6.0, 6.0, 6.0)   # denser than the real system, so the test is well-connected
                        # (and meaningful) whatever `rc` you currently have set in config.jl
positions, data = make_synthetic_system(N, box; seed=1)
boxtuple = (data["lx"], data["ly"], data["lz"])
mini_relax!(positions, data["cid"], boxtuple)
data["x"] = positions[:, 1]; data["y"] = positions[:, 2]; data["z"] = positions[:, 3]

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
println("Two eigenvalues cross as t: 0 -> 1 (index 6 goes -1 -> +1, index 7 goes +1 -> -1).")
println("Comparing naive full-diag+lowest, single-vector (k=1) tracking, and block")
println("(k=8) tracking through the crossing.\n")

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

H0, vals0, Q0 = H_at(0.0)

# --- single-vector (k=1) tracking, warm-started at the target ---
v_prev_single = Q0[:, 6]

# --- block (k=8) tracking, warm-started from the 8 lowest eigenpairs of H0,
#     with the tracked target identified by whichever slot best matches it
#     (mirrors how this would actually be seeded: one full diagonalization,
#     then track that whole block from then on) ---
kblock = 8
order0 = sortperm(vals0)
X_prev_block = Q0[:, order0[1:kblock]]
target_slot = argmax([abs(dot(X_prev_block[:, i], Q0[:, 6])) for i in 1:kblock])
println("block tracking: following slot $target_slot of $kblock\n")

single_ok = true
block_ok = true
for t in 0.0:0.1:1.0
    H, vals_true, Qt = H_at(t)
    true_target_vec = Qt[:, 6]   # by construction, the smooth continuation
    Hsparse = sparse(Matrix(H))

    w, V = eigen(H)
    naive_idx = argmin(w)
    naive_matches_true = abs(dot(V[:, naive_idx], true_target_vec)) > 0.9

    vals_s, vecs_s, _ = track_modes(Hsparse, reshape(v_prev_single, :, 1); maxiter=200, tol=1e-10)
    single_vec = vecs_s[:, 1]
    single_overlap = abs(dot(single_vec, true_target_vec))

    vals_b, vecs_b, _ = track_modes(Hsparse, X_prev_block; maxiter=200, tol=1e-10)
    block_vec = vecs_b[:, target_slot]
    block_overlap = abs(dot(block_vec, true_target_vec))

    println("t=$(round(t,digits=1))  true_eig=$(round(-1+2*t,digits=3))  " *
            "naive_matches_true=$naive_matches_true  " *
            "single(k=1)_overlap=$(round(single_overlap,digits=3))  " *
            "block(k=$kblock)_overlap=$(round(block_overlap,digits=3))")

    global single_ok, block_ok
    single_overlap < 0.9 && (single_ok = false)
    block_overlap < 0.9 && (block_ok = false)
    global v_prev_single = single_vec
    global X_prev_block = vecs_b
end

println("\nsingle-vector (k=1) tracking survived the whole crossing: ", verdict(single_ok),
        "  (expected to FAIL -- this is why track_mode carries a warning)")
println("block (k=$kblock) tracking survived the whole crossing:     ", verdict(block_ok))

println("\n=== SUMMARY ===")
println("Test 1 (static sanity check):            ", verdict(ok1))
println("Test 2 (old track_mode after perturb):    ", verdict(ok2_old))
println("Test 2 (new track_modes after perturb):   ", verdict(ok2_new))
println("Test 3 (single-vector through crossing):  ", verdict(single_ok), "   <- expect this to FAIL")
println("Test 3 (block-k=$kblock through crossing):    ", verdict(block_ok))

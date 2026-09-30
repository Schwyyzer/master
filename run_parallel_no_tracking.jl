#=
Run many independent ART attempts in parallel (as separate OS processes,
via Julia's Distributed stdlib) until a target number of saddle points
have been found, then report: a diversity summary, any failed attempts
that overshot into a different minimum on unconstrained relax, and the
barrier energy of every saddle point found.

This is the NO-MODE-TRACKING variant: phase 2 uses the original cold
full re-diagonalization every iteration (art_core_no_tracking.jl /
calculate_moves), i.e. the version that gave 10/10 then 50/50 successes
on the real system, plus the overshoot-relax check and barrier-energy
summary that were added alongside mode tracking but don't depend on it.
Mode tracking is set aside for now (see art_core.jl / mode_tracking.jl /
rqi_tracking.jl to revisit it later).

Phase 2's step size is now ADAPTIVE: proportional to the force parallel
to the move direction, large early in phase 2 (steep, just-destabilized
landscape) and shrinking toward the old fixed move_phase2_modifier
(now the floor) as that parallel force vanishes approaching the saddle
-- capped at MAX_PHASE2_STEP below. See art_core_no_tracking.jl's
run_art_attempt docstring for the exact calibration. First test: floor
unchanged, ceiling set to 5x it -- watch phase2_step_size_min/max in
the CSV to see the actual taper, and max_phase2_angle_deg in case
bigger early steps reintroduce mode-jump problems the fixed small step
didn't have.

WHY SEPARATE PROCESSES AND NOT Threads.@threads:
Phase 1/2 both diagonalize the Hessian through Arpack.jl, which wraps the
classic Fortran ARPACK library. That library keeps iteration state in
Fortran SAVE (i.e. static/global) variables across its reverse-
communication calls, and is not safe to call concurrently from multiple
threads inside one process -- doing so risks silent corruption, not just
a crash. Separate OS processes (Distributed) each get their own copy of
that static state, so this sidesteps the problem entirely rather than
working around it with locks (which would also serialize the single most
expensive step per iteration, eliminating most of the parallel speedup).

BEFORE WALKING AWAY FOR THE DAY: this was written without access to a
Julia install to compile/run it against your real data (sandboxed
environment, no network access to get one). Please run a quick smoke
test first -- set TARGET_SUCCESSES = 2 and N_WORKERS = 2 below, confirm
it runs to completion and results.csv looks sane, THEN scale up to 50 /
however many workers your machine has.

Usage:  julia run_parallel.jl
=#

using Distributed
using Dates
using Printf
using LinearAlgebra
using DelimitedFiles

# ============================================================
# Settings -- edit these
# ============================================================
LAMMPS_PATH = "//home//schwyyzer//Desktop//Master Thesis//config_small_relaxed.lammps"
TARGET_SUCCESSES = 50
N_WORKERS = max(1, Sys.CPU_THREADS - 1)   # leave a core free for the OS/master
MAX_TOTAL_ITER = 2000                     # shared phase1+phase2 budget, matches
                                           # new_start_1_1_1.py's `iteration<2000`
                                           # (Version1.1.jl used 1000; phase 1 had
                                           # no cap at all in either language)
MODE_RECHECK_EVERY = 25                   # unused by this variant, kept for
                                           # signature compatibility with art_core.jl
MAX_PHASE2_STEP_MULTIPLIER = 5.0          # ceiling on the adaptive phase-2 step size,
                                           # as a multiple of move_phase2_modifier (the
                                           # floor, unchanged from before this change).
                                           # First test per the user's request: 5x.
                                           # (Actual MAX_PHASE2_STEP is computed below,
                                           # once config.jl -- and so move_phase2_modifier
                                           # -- has actually been loaded.)
SADDLE_EIGENVALUE_TOLERANCE = 0.01        # success requires crit_eigenvalue < this, not
                                           # strictly < 0 -- on a smoothly-converging search
                                           # the discrete phase-2 loop routinely overshoots
                                           # the true zero-crossing by a small amount on its
                                           # last step (observed: +0.0004 to +0.006), which
                                           # a strict < 0 check wrongly rejects as failure
                                           # despite being a genuinely converged saddle. This
                                           # is a small POSITIVE tolerance, not a magnitude
                                           # check -- it still accepts every case the old
                                           # check did (any negative value, however large).

OUTPUT_DIR = "//home//schwyyzer//Desktop//Master Thesis//parallel_art_run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
DUMP_DIR = joinpath(OUTPUT_DIR, "dumps")
LOG_PATH = joinpath(OUTPUT_DIR, "results.csv")

mkpath(DUMP_DIR)

# ============================================================
# Worker setup
# ============================================================
if nprocs() == 1
    addprocs(N_WORKERS; exeflags=`--threads=1`)
end

@everywhere begin
    include("Version1.1.jl")   # defines force/compute_energy/perpendicular_forces
                                # and loads config.jl, neighbor_pairs.jl, Hessian.jl,
                                # particle_movement.jl, eigenvalueDecomp.jl
    include("art_core_no_tracking.jl")  # defines run_art_attempt / AttemptResult
                                # (cold full re-diagonalization phase 2,
                                # no mode tracking -- see file header)
    using LinearAlgebra
    BLAS.set_num_threads(1)    # one attempt per process at a time -- avoid
                                # every worker independently oversubscribing BLAS
end

MAX_PHASE2_STEP = MAX_PHASE2_STEP_MULTIPLIER * move_phase2_modifier
println("phase-2 adaptive step size: floor=$move_phase2_modifier  ceiling=$MAX_PHASE2_STEP " *
        "($(MAX_PHASE2_STEP_MULTIPLIER)x)")

@everywhere lammps_path = $LAMMPS_PATH
@everywhere max_total_iter = $MAX_TOTAL_ITER
@everywhere mode_recheck_every = $MODE_RECHECK_EVERY
@everywhere max_phase2_step = $MAX_PHASE2_STEP
@everywhere saddle_eigenvalue_tolerance = $SADDLE_EIGENVALUE_TOLERANCE
@everywhere begin
    data = parse_lammps_data(lammps_path)
    positions0 = hcat(data["x"], data["y"], data["z"])
    box = (data["lx"], data["ly"], data["lz"])
    pairs0 = build_neighbor_pairs(positions0, rc, box)

    # Thin wrapper closing over this worker's OWN copy of the shared,
    # read-only setup data, so a dispatch only has to ship the tiny
    # (seed, dump_path) pair over IPC instead of re-serializing the
    # position/pair arrays on every single attempt.
    function attempt_wrapper(seed::Int, dump_path::String)
        run_art_attempt(
            seed, positions0, data, pairs0, box;
            dump_file = dump_path,
            max_total_iter = max_total_iter,
            mode_recheck_every = mode_recheck_every,
            max_phase2_step = max_phase2_step,
            saddle_eigenvalue_tolerance = saddle_eigenvalue_tolerance,
        )
    end
end

println("natoms = $(data["natoms"]), box = $box, npairs(rc=$rc) = $(size(pairs0,1))")
println("workers = $(workers())  (each pinned to 1 thread; BLAS pinned to 1 thread)")

# ============================================================
# Dispatch loop: keep every worker busy until TARGET_SUCCESSES
#
# Everything from here on is wrapped in a function rather than left as
# top-level script code. This isn't just style: a `while` loop at
# top-level script scope has "soft scope" in Julia, so `n_attempts +=
# 1` inside it (when `n_attempts` was already defined outside the
# loop) is genuinely ambiguous and, when run non-interactively via
# `julia run_parallel.jl` (as opposed to the REPL), resolves to
# creating a brand new local -- which then fails with UndefVarError
# the moment it's read before being assigned. A function has ordinary,
# unambiguous lexical scoping throughout, which sidesteps this
# entirely (and lets Julia type-infer everything properly, which is
# also just faster than top-level global-variable code).
# ============================================================
function run_batch()
    seed_state = Ref(1)
    next_seed() = (s = seed_state[]; seed_state[] += 1; s)

    function dispatch!(w)
        s = next_seed()
        dump_path = joinpath(DUMP_DIR, "saddle_$(lpad(s, 6, '0')).dump")
        return remotecall(attempt_wrapper, w, s, dump_path)
    end

    pending = Dict{Int,Future}()   # worker_id => Future
    for w in workers()
        pending[w] = dispatch!(w)
    end

    successes = AttemptResult[]
    overshoot_finds = AttemptResult[]   # failed attempts whose unconstrained
                                         # relax landed in a genuinely different minimum
    n_attempts = 0
    start_time = time()

    log_io = open(LOG_PATH, "w")
    println(log_io,
        "seed,success,kicked_atom,phase1_iters,total_iters,crit_eigenvalue,initial_energy," *
        "final_energy,max_phase2_angle_deg,phase2_step_size_min,phase2_step_size_max," *
        "overshoot_energy,overshoot_new_minimum,dump_file,error"
    )
    flush(log_io)

    live_workers = Set(workers())

    while length(successes) < TARGET_SUCCESSES && !isempty(live_workers)
        for w in collect(keys(pending))
            fut = pending[w]
            if isready(fut)
                # run_art_attempt already catches every ordinary
                # algorithm-level failure and returns a failed
                # AttemptResult instead of throwing. This extra layer
                # only guards against the worker PROCESS itself dying
                # (e.g. OOM) -- rare, but this run is meant to survive
                # unattended, so one dead worker shouldn't take the
                # whole batch down with it.
                local res
                try
                    res = fetch(fut)
                catch e
                    println("worker $w appears to have died ($(sprint(showerror, e))); dropping it, continuing with the rest")
                    delete!(pending, w)
                    delete!(live_workers, w)
                    continue
                end
                n_attempts += 1

                err_field = res.error === nothing ? "" : replace(res.error, "\n" => " | ", "," => ";")
                dump_field = res.dump_file === nothing ? "" : res.dump_file
                angle_field = res.max_phase2_angle_deg === nothing ? "" : round(res.max_phase2_angle_deg, digits=2)
                step_min_field = res.phase2_step_size_min === nothing ? "" : res.phase2_step_size_min
                step_max_field = res.phase2_step_size_max === nothing ? "" : res.phase2_step_size_max
                ov_e_field = res.overshoot_energy === nothing ? "" : res.overshoot_energy
                ov_new_field = res.overshoot_new_minimum === nothing ? "" : res.overshoot_new_minimum
                println(log_io,
                    "$(res.seed),$(res.success),$(res.kicked_atom),$(res.steps_part1),$(res.iteration)," *
                    "$(res.crit_eigenvalue),$(res.initial_energy),$(res.final_energy),$angle_field," *
                    "$step_min_field,$step_max_field,$ov_e_field,$ov_new_field,$dump_field,$err_field"
                )
                flush(log_io)

                if res.success
                    push!(successes, res)
                    elapsed = round(time() - start_time, digits=1)
                    @printf(
                        "[%2d/%2d] SUCCESS  seed=%-6d atom=%-5d E=%.6f  iters=%-4d (phase1=%-4d)  max_angle=%.1f°  step=[%.5f,%.5f]  elapsed=%.1fs  attempts_so_far=%d\n",
                        length(successes), TARGET_SUCCESSES, res.seed, res.kicked_atom,
                        res.final_energy, res.iteration, res.steps_part1,
                        something(res.max_phase2_angle_deg, NaN),
                        something(res.phase2_step_size_min, NaN), something(res.phase2_step_size_max, NaN),
                        elapsed, n_attempts
                    )
                else
                    if res.overshoot_new_minimum == true
                        push!(overshoot_finds, res)
                        @printf(
                            "  [overshoot] seed=%-6d atom=%-5d landed in a different minimum on unconstrained relax (E=%.6f vs start %.6f)\n",
                            res.seed, res.kicked_atom, res.overshoot_energy, res.initial_energy
                        )
                    end
                    if n_attempts % 20 == 0
                        elapsed = round(time() - start_time, digits=1)
                        println("  ...$n_attempts attempts tried so far, $(length(successes)) successes, elapsed=$(elapsed)s")
                    end
                end

                if length(successes) >= TARGET_SUCCESSES
                    break
                end

                pending[w] = dispatch!(w)
            end
        end
        sleep(0.05)
    end

    close(log_io)
    rmprocs(workers())

    return successes, overshoot_finds, n_attempts
end

successes, overshoot_finds, n_attempts = run_batch()

# ============================================================
# Diversity summary
# ============================================================
function print_diversity_summary(successes, n_attempts)
    n = length(successes)
    println("\n=== Done: $n saddle points found after $n_attempts attempts ===")
    println("full per-attempt log: $LOG_PATH")

    if n < 2
        println("fewer than 2 successes -- not enough to compare diversity.")
        return
    end

    box_vec = [data["lx"] data["ly"] data["lz"]]

    # Displacement of each success's final structure relative to the shared
    # starting minimum, minimum-image wrapped, with the mean (rigid
    # translation) removed -- that's one of the periodic system's own zero
    # modes, not a real structural difference between saddle points.
    disps = Vector{Matrix{Float64}}(undef, n)
    for k in 1:n
        dr = successes[k].positions .- positions0
        dr .-= box_vec .* round.(dr ./ box_vec)
        dr .-= sum(dr, dims=1) ./ size(dr, 1)
        disps[k] = dr
    end

    rmsd = zeros(n, n)
    for a in 1:n, b in 1:n
        d = disps[a] .- disps[b]
        rmsd[a, b] = sqrt(sum(d .^ 2) / size(d, 1))
    end

    energies = [s.final_energy for s in successes]
    n_energy_clusters = length(unique(round.(energies, digits=4)))

    offdiag = [rmsd[a, b] for a in 1:n for b in 1:n if a != b]

    println("distinct final-energy clusters (rounded to 1e-4): $n_energy_clusters out of $n successes")
    println("pairwise structural RMSD (COM-removed, minimum-image):")
    println("  min  = ", minimum(offdiag))
    println("  max  = ", maximum(offdiag))
    println("  mean = ", sum(offdiag) / length(offdiag))
    println("(near-zero RMSD between most pairs => most attempts are landing on the same")
    println(" saddle point; a broad spread of nonzero values => genuinely different saddles)")

    rmsd_path = joinpath(OUTPUT_DIR, "pairwise_rmsd.csv")
    writedlm(rmsd_path, rmsd, ',')
    println("full pairwise RMSD matrix: $rmsd_path")
end

# ============================================================
# Overshoot summary: failed attempts whose unconstrained relax landed
# in a genuinely different minimum than the start (see art_core.jl's
# _overshoot_check) -- real transitions found even where phase 2 never
# cleanly converged on the saddle itself.
# ============================================================
function print_overshoot_summary(overshoot_finds)
    n = length(overshoot_finds)
    println("\n=== Failed attempts that overshot into a different minimum: $n ===")
    n == 0 && return
    for r in overshoot_finds
        @printf("  seed=%-6d atom=%-5d E_overshoot=%.6f  ΔE=%.6f\n",
                r.seed, r.kicked_atom, r.overshoot_energy, r.overshoot_energy - r.initial_energy)
    end
end

# ============================================================
# Barrier energies: the actual point of an ART search. For each
# converged saddle, barrier = saddle_energy - starting_minimum_energy.
# ============================================================
function print_barrier_summary(successes)
    n = length(successes)
    println("\n=== Barrier energies of all saddle points found ($n) ===")
    if n == 0
        println("no successes -- nothing to report.")
        return
    end

    barriers = [(s.seed, s.final_energy - s.initial_energy) for s in successes]
    sort!(barriers, by = x -> x[2])

    for (seed, barrier) in barriers
        @printf("  seed=%-6d barrier = %.6f\n", seed, barrier)
    end

    vals = [b for (_, b) in barriers]
    @printf("\nmin barrier  = %.6f\n", minimum(vals))
    @printf("max barrier  = %.6f\n", maximum(vals))
    @printf("mean barrier = %.6f\n", sum(vals) / n)
end

print_diversity_summary(successes, n_attempts)
print_overshoot_summary(overshoot_finds)
print_barrier_summary(successes)

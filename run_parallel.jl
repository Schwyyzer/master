#=
Run many independent ART attempts in parallel (as separate OS processes,
via Julia's Distributed stdlib) until a target number of saddle points
have been found, then report a first-pass diversity summary.

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
    include("art_core.jl")     # defines run_art_attempt / AttemptResult
    using LinearAlgebra
    BLAS.set_num_threads(1)    # one attempt per process at a time -- avoid
                                # every worker independently oversubscribing BLAS
end

@everywhere lammps_path = $LAMMPS_PATH
@everywhere max_total_iter = $MAX_TOTAL_ITER
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
        )
    end
end

println("natoms = $(data["natoms"]), box = $box, npairs(rc=$rc) = $(size(pairs0,1))")
println("workers = $(workers())  (each pinned to 1 thread; BLAS pinned to 1 thread)")

# ============================================================
# Dispatch loop: keep every worker busy until TARGET_SUCCESSES
# ============================================================
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
n_attempts = 0
start_time = time()

log_io = open(LOG_PATH, "w")
println(log_io, "seed,success,kicked_atom,phase1_iters,total_iters,crit_eigenvalue,initial_energy,final_energy,dump_file,error")
flush(log_io)

while length(successes) < TARGET_SUCCESSES
    for w in collect(keys(pending))
        fut = pending[w]
        if isready(fut)
            res = fetch(fut)
            n_attempts += 1

            err_field = res.error === nothing ? "" : replace(res.error, "\n" => " | ", "," => ";")
            dump_field = res.dump_file === nothing ? "" : res.dump_file
            println(log_io,
                "$(res.seed),$(res.success),$(res.kicked_atom),$(res.steps_part1),$(res.iteration)," *
                "$(res.crit_eigenvalue),$(res.initial_energy),$(res.final_energy),$dump_field,$err_field"
            )
            flush(log_io)

            if res.success
                push!(successes, res)
                elapsed = round(time() - start_time, digits=1)
                @printf(
                    "[%2d/%2d] SUCCESS  seed=%-6d atom=%-5d E=%.6f  iters=%-4d (phase1=%-4d)  elapsed=%.1fs  attempts_so_far=%d\n",
                    length(successes), TARGET_SUCCESSES, res.seed, res.kicked_atom,
                    res.final_energy, res.iteration, res.steps_part1, elapsed, n_attempts
                )
            elseif n_attempts % 20 == 0
                elapsed = round(time() - start_time, digits=1)
                println("  ...$n_attempts attempts tried so far, $(length(successes)) successes, elapsed=$(elapsed)s")
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

# ============================================================
# Diversity summary
# ============================================================
n = length(successes)
println("\n=== Done: $n saddle points found after $n_attempts attempts ===")
println("full per-attempt log: $LOG_PATH")

if n >= 2
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
else
    println("fewer than 2 successes -- not enough to compare diversity.")
end

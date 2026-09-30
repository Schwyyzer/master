#=
Run many independent ART attempts in parallel, using art_core_fortran.jl's
run_art_attempt_fortran -- the algorithm rewritten to match the working
Fortran reference art_woconewer.f90 (unified FIRE+Newton-step loop,
warm-started Lanczos mode tracking, no mass-weighting, static neighbor
list, force-magnitude convergence). See art_core_fortran.jl's header for
the full list of differences from run_parallel_no_tracking.jl's
phase1/phase2 design.

Modeled directly on run_parallel_no_tracking.jl; the differences are:
  - includes art_core_fortran.jl instead of art_core_no_tracking.jl
  - sets the global `mass` to [1.0, 1.0] AFTER config.jl loads (config.jl's
    own mass=[2,1] is loaded first by Version1.1.jl's include chain, then
    overridden here) -- build_hessian_fast is reused UNCHANGED; dividing
    every mass factor by 1.0 makes it exactly the plain, non-mass-weighted
    Hessian the reference actually uses
  - builds ONE static neighbor list (skin = rc+1.0, matching Fortran's
    range_nn = range+1.0) from the starting minimum, shared read-only by
    every attempt and never rebuilt -- not a fresh pairs0 handed to
    run_art_attempt like the old driver; this file's attempt_wrapper reuses
    the module-level `pairs_static` directly, and TARGET_SUCCESSES now
    counts attempts that reach a genuinely converged, genuinely negative
    final eigenvalue (see art_core_fortran.jl's `success` docstring note)

Usage:  julia run_parallel_fortran.jl
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
MAX_ITERATIONS = 200                      # matches art_woconewer.f90's max_number_of_art_iterations

OUTPUT_DIR = "//home//schwyyzer//Desktop//Master Thesis//parallel_art_fortran_run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
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
    include("art_core_fortran.jl")  # defines run_art_attempt_fortran / FortranAttemptResult
    using LinearAlgebra
    BLAS.set_num_threads(1)    # one attempt per process at a time -- avoid
                                # every worker independently oversubscribing BLAS

    # The reference declares mass(1)=2.0/mass(2)=1.0 but never actually
    # uses them anywhere in the algorithm -- overriding to unit mass here
    # makes build_hessian_fast (reused unchanged) reduce to exactly the
    # plain, non-mass-weighted Hessian art_woconewer.f90 builds.
    mass = [1.0, 1.0]
end

@everywhere lammps_path = $LAMMPS_PATH
@everywhere max_iterations = $MAX_ITERATIONS
@everywhere begin
    data = parse_lammps_data(lammps_path)
    positions0 = hcat(data["x"], data["y"], data["z"])
    box = (data["lx"], data["ly"], data["lz"])

    # Static neighbor list: built ONCE from the starting minimum with a
    # skin of rc+1.0 (build_neighbor_pairs already adds 0.5 internally, so
    # pass rc+0.5 to get the same rc+1.0 total padding as Fortran's
    # range_nn = range+1.0), and reused for every attempt and every
    # iteration -- never rebuilt, matching the reference's fixed nn_list.
    # Safe only because force/compute_energy/build_hessian_fast all filter
    # every pair to r<=rc internally regardless of the candidate list.
    pairs_static = build_neighbor_pairs(positions0, rc + 0.5, box)

    function attempt_wrapper(seed::Int, dump_path::String)
        run_art_attempt_fortran(
            seed, positions0, data, pairs_static, box;
            dump_file = dump_path,
            max_iterations = max_iterations,
        )
    end
end

println("natoms = $(data["natoms"]), box = $box, npairs(static skin list, rc=$rc) = $(size(pairs_static,1))")
println("mass (overridden for this driver) = $mass")
println("workers = $(workers())  (each pinned to 1 thread; BLAS pinned to 1 thread)")

# ============================================================
# Dispatch loop: keep every worker busy until TARGET_SUCCESSES
# (see run_parallel_no_tracking.jl for why this is wrapped in a function)
# ============================================================
function run_batch()
    seed_state = Ref(1)
    next_seed() = (s = seed_state[]; seed_state[] += 1; s)

    function dispatch!(w)
        s = next_seed()
        dump_path = joinpath(DUMP_DIR, "saddle_$(lpad(s, 6, '0')).dump")
        return remotecall(attempt_wrapper, w, s, dump_path)
    end

    pending = Dict{Int,Future}()
    for w in workers()
        pending[w] = dispatch!(w)
    end

    successes = FortranAttemptResult[]
    degenerate_aborts = 0
    n_attempts = 0
    start_time = time()

    log_io = open(LOG_PATH, "w")
    println(log_io,
        "seed,success,kicked_atom,iterations,crit_eigenvalue,converged,is_saddle," *
        "force_magnitude,initial_energy,final_energy,degenerate_mode_aborted,dump_file,error"
    )
    flush(log_io)

    live_workers = Set(workers())

    while length(successes) < TARGET_SUCCESSES && !isempty(live_workers)
        for w in collect(keys(pending))
            fut = pending[w]
            if isready(fut)
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
                println(log_io,
                    "$(res.seed),$(res.success),$(res.kicked_atom),$(res.iteration),$(res.crit_eigenvalue)," *
                    "$(res.converged),$(res.is_saddle),$(res.force_magnitude),$(res.initial_energy)," *
                    "$(res.final_energy),$(res.degenerate_mode_aborted),$dump_field,$err_field"
                )
                flush(log_io)

                if res.degenerate_mode_aborted
                    degenerate_aborts += 1
                end

                if res.success
                    push!(successes, res)
                    elapsed = round(time() - start_time, digits=1)
                    @printf(
                        "[%2d/%2d] SUCCESS  seed=%-6d atom=%-5d E=%.6f  iters=%-4d  eig=%.5f  elapsed=%.1fs  attempts_so_far=%d\n",
                        length(successes), TARGET_SUCCESSES, res.seed, res.kicked_atom,
                        res.final_energy, res.iteration, res.crit_eigenvalue,
                        elapsed, n_attempts
                    )
                else
                    if n_attempts % 20 == 0
                        elapsed = round(time() - start_time, digits=1)
                        println("  ...$n_attempts attempts tried so far, $(length(successes)) successes, " *
                                "$degenerate_aborts degenerate-mode aborts, elapsed=$(elapsed)s")
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

    return successes, degenerate_aborts, n_attempts
end

successes, degenerate_aborts, n_attempts = run_batch()

# ============================================================
# Diversity summary
# ============================================================
function print_diversity_summary(successes, n_attempts, degenerate_aborts)
    n = length(successes)
    println("\n=== Done: $n saddle points found after $n_attempts attempts ($degenerate_aborts degenerate-mode aborts) ===")
    println("full per-attempt log: $LOG_PATH")

    if n < 2
        println("fewer than 2 successes -- not enough to compare diversity.")
        return
    end

    box_vec = [data["lx"] data["ly"] data["lz"]]

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

    rmsd_path = joinpath(OUTPUT_DIR, "pairwise_rmsd.csv")
    writedlm(rmsd_path, rmsd, ',')
    println("full pairwise RMSD matrix: $rmsd_path")
end

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

print_diversity_summary(successes, n_attempts, degenerate_aborts)
print_barrier_summary(successes)

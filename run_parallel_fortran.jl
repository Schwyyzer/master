#=
Run many independent ART attempts in parallel, using art_core_fortran.jl's
run_art_attempt_fortran -- the algorithm matching the working Fortran
reference art_woconewer.f90 (unified FIRE+Newton-step loop, warm-started
Lanczos mode tracking, no mass-weighting, static neighbor list, force-
magnitude convergence), plus two validated additions from the trust-region
paper (a second-eigenvalue validity guard, an MEP-proximity diagnostic),
plus an opt-in trust-region-adaptive Newton-step clamp (TRUST_REGION_EPS
below; see art_core_fortran.jl's header for what this is, how it differs
from the paper's own g/alpha scheme, and how it was validated).
See art_core_fortran.jl's header for the full list of differences from
run_parallel_no_tracking.jl's phase1/phase2 design.

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
    the module-level `pairs_static` directly, and TARGET_SUCCESSES counts
    attempts that reach the end of the search loop WITHOUT being discarded
    by the degenerate-mode/second-eigenvalue guards -- matching
    art_woconewer.f90's own "write output unless degenerate" criterion
    exactly, not "strictly converged to a negative eigenvalue" (see
    art_core_fortran.jl's `success` docstring). That stricter check isn't
    gone -- `converged` and `is_saddle` are in every row of the CSV and in
    every FortranAttemptResult, so you can filter on them, or (recommended,
    and what the reference's own workflow implies) run validate_saddles.jl/
    .py over the dump files afterward for an independent saddle check.

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

# When true: no per-attempt CSV log (and no flush() after every attempt --
# a real cost over thousands of failed attempts), no per-success dump
# files, no progress printlns, no diversity/RMSD report. Instead, at the
# very end of the run, print ONLY the config variables that were actually
# used, the starting position, and the saddle point positions found --
# nothing else. Use this when you just want the result, not the
# documentation of how the run got there.
MINIMAL_OUTPUT = false

# run_art_attempt_fortran's own keyword defaults (art_core_fortran.jl),
# duplicated here as named settings -- rather than left implicit -- so
# that (a) MINIMAL_OUTPUT's end-of-run summary has something concrete to
# print as "the config variables that were used", and (b) they can be
# tuned from this one place without touching art_core_fortran.jl. Values
# unchanged from those defaults, so behavior is identical either way.
EIGENVALUE_THRESHOLD = 3.0
DELTA_X_MAX = 0.125
DELTA_LM = 0.025
FORCE_TOLERANCE = 1e-3
ZERO_EIGENVALUE_THRESHOLD = 0.1
MAX_ZERO_EIGENVALUE_THRESHOLD_NUMBER = 10
MAX_CONSECUTIVE_SECOND_EIG_NEGATIVE = 5
KICK_MAGNITUDE = 0.15
STARTING_FIRE_ITERATIONS = 5
LANCZOS_ITERATIONS = 10

# Trust-region-adaptive cap on the Newton-step branch's step size, in
# place of the flat DELTA_X_MAX clamp above (see art_core_fortran.jl's
# header for what this is and how it was validated in Python: on a
# 22-seed, 500-atom sample, eps=0.1 found one additional genuine saddle
# with zero regressions elsewhere, 6/22 -> 7/22). Set to `nothing` to
# fall back to the original flat DELTA_X_MAX clamp exactly (e.g. for an
# A/B comparison against this run).
TRUST_REGION_EPS = 0.1

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
@everywhere minimal_output = $MINIMAL_OUTPUT
@everywhere eigenvalue_threshold = $EIGENVALUE_THRESHOLD
@everywhere delta_x_max = $DELTA_X_MAX
@everywhere delta_lm = $DELTA_LM
@everywhere force_tolerance = $FORCE_TOLERANCE
@everywhere zero_eigenvalue_threshold = $ZERO_EIGENVALUE_THRESHOLD
@everywhere max_zero_eigenvalue_threshold_number = $MAX_ZERO_EIGENVALUE_THRESHOLD_NUMBER
@everywhere max_consecutive_second_eig_negative = $MAX_CONSECUTIVE_SECOND_EIG_NEGATIVE
@everywhere kick_magnitude = $KICK_MAGNITUDE
@everywhere starting_fire_iterations = $STARTING_FIRE_ITERATIONS
@everywhere lanczos_iterations = $LANCZOS_ITERATIONS
@everywhere trust_region_eps = $TRUST_REGION_EPS
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
            dump_file = minimal_output ? nothing : dump_path,
            max_iterations = max_iterations,
            eigenvalue_threshold = eigenvalue_threshold,
            delta_x_max = delta_x_max,
            delta_lm = delta_lm,
            force_tolerance = force_tolerance,
            zero_eigenvalue_threshold = zero_eigenvalue_threshold,
            max_zero_eigenvalue_threshold_number = max_zero_eigenvalue_threshold_number,
            max_consecutive_second_eig_negative = max_consecutive_second_eig_negative,
            kick_magnitude = kick_magnitude,
            starting_fire_iterations = starting_fire_iterations,
            lanczos_iterations = lanczos_iterations,
            trust_region_eps = trust_region_eps,
        )
    end
end

if !MINIMAL_OUTPUT
    println("natoms = $(data["natoms"]), box = $box, npairs(static skin list, rc=$rc) = $(size(pairs_static,1))")
    println("mass (overridden for this driver) = $mass")
    println("workers = $(workers())  (each pinned to 1 thread; BLAS pinned to 1 thread)")
end

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
    guard_aborts = 0
    n_attempts = 0
    start_time = time()

    log_io = MINIMAL_OUTPUT ? nothing : open(LOG_PATH, "w")
    if !MINIMAL_OUTPUT
        println(log_io,
            "seed,success,kicked_atom,iterations,crit_eigenvalue,second_eigenvalue,converged,is_saddle," *
            "force_magnitude,mep_proximity,initial_energy,final_energy,degenerate_mode_aborted," *
            "second_eigenvalue_aborted,dump_file,error"
        )
        flush(log_io)
    end

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

                if !MINIMAL_OUTPUT
                    err_field = res.error === nothing ? "" : replace(res.error, "\n" => " | ", "," => ";")
                    dump_field = res.dump_file === nothing ? "" : res.dump_file
                    println(log_io,
                        "$(res.seed),$(res.success),$(res.kicked_atom),$(res.iteration),$(res.crit_eigenvalue)," *
                        "$(res.second_eigenvalue),$(res.converged),$(res.is_saddle),$(res.force_magnitude)," *
                        "$(res.mep_proximity),$(res.initial_energy),$(res.final_energy)," *
                        "$(res.degenerate_mode_aborted),$(res.second_eigenvalue_aborted),$dump_field,$err_field"
                    )
                    flush(log_io)
                end

                if res.degenerate_mode_aborted || res.second_eigenvalue_aborted
                    guard_aborts += 1
                end

                if res.success
                    push!(successes, res)
                    if !MINIMAL_OUTPUT
                        elapsed = round(time() - start_time, digits=1)
                        @printf(
                            "[%2d/%2d] SUCCESS  seed=%-6d atom=%-5d E=%.6f  iters=%-4d  eig=%.5f  elapsed=%.1fs  attempts_so_far=%d\n",
                            length(successes), TARGET_SUCCESSES, res.seed, res.kicked_atom,
                            res.final_energy, res.iteration, res.crit_eigenvalue,
                            elapsed, n_attempts
                        )
                    end
                else
                    if !MINIMAL_OUTPUT && n_attempts % 20 == 0
                        elapsed = round(time() - start_time, digits=1)
                        println("  ...$n_attempts attempts tried so far, $(length(successes)) successes, " *
                                "$guard_aborts guard aborts (degenerate-mode or second-eigenvalue), elapsed=$(elapsed)s")
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

    if !MINIMAL_OUTPUT
        close(log_io)
    end
    rmprocs(workers())

    return successes, guard_aborts, n_attempts
end

successes, guard_aborts, n_attempts = run_batch()

# ============================================================
# Diversity summary
# ============================================================
function print_diversity_summary(successes, n_attempts, guard_aborts)
    n = length(successes)
    println("\n=== Done: $n saddle points found after $n_attempts attempts " *
            "($guard_aborts guard aborts: degenerate-mode or second-eigenvalue) ===")
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

# ============================================================
# Minimal summary (MINIMAL_OUTPUT=true): config + starting position +
# saddle positions, nothing else -- no CSV, no dump files, no RMSD report.
# ============================================================
function print_positions(positions)
    for i in axes(positions, 1)
        @printf("%6d  %.12f  %.12f  %.12f\n", i, positions[i,1], positions[i,2], positions[i,3])
    end
end

function print_minimal_summary(successes, positions0)
    println("=== config ===")
    println("rc = $rc")
    println("mass = $mass")
    println("eigenvalue_threshold = $EIGENVALUE_THRESHOLD")
    println("delta_x_max = $DELTA_X_MAX")
    println("trust_region_eps = $TRUST_REGION_EPS")
    println("delta_lm = $DELTA_LM")
    println("force_tolerance = $FORCE_TOLERANCE")
    println("zero_eigenvalue_threshold = $ZERO_EIGENVALUE_THRESHOLD")
    println("max_zero_eigenvalue_threshold_number = $MAX_ZERO_EIGENVALUE_THRESHOLD_NUMBER")
    println("max_consecutive_second_eig_negative = $MAX_CONSECUTIVE_SECOND_EIG_NEGATIVE")
    println("kick_magnitude = $KICK_MAGNITUDE")
    println("starting_fire_iterations = $STARTING_FIRE_ITERATIONS")
    println("lanczos_iterations = $LANCZOS_ITERATIONS")
    println("max_iterations = $MAX_ITERATIONS")

    println("\n=== starting position ($(size(positions0,1)) atoms) ===")
    print_positions(positions0)

    # "success" now means "not discarded by the degenerate-mode/second-
    # eigenvalue guard" (matching the Fortran reference's own output
    # criterion -- see art_core_fortran.jl's success docstring), not
    # "independently confirmed saddle". converged/is_saddle are shown per
    # entry so you can see which of these also meet the stricter check;
    # validate_saddles.jl/.py over these positions is the real filter.
    println("\n=== exported attempt positions ($(length(successes)) found) ===")
    for s in successes
        println("\n--- seed=$(s.seed)  final_energy=$(s.final_energy)  crit_eigenvalue=$(s.crit_eigenvalue) " *
                "converged=$(s.converged)  is_saddle=$(s.is_saddle) ---")
        print_positions(s.positions)
    end
end

if MINIMAL_OUTPUT
    print_minimal_summary(successes, positions0)
else
    print_diversity_summary(successes, n_attempts, guard_aborts)
    print_barrier_summary(successes)
end

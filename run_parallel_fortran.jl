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
  - each attempt also gets `elapsed_time` (whole-attempt wall-clock
    seconds, in the CSV) and, unless MINIMAL_OUTPUT, a per-iteration
    step-norm trajectory CSV under TRAJECTORY_DIR (path in the CSV's
    `trajectory_file` column) -- added for comparing the trust-region-
    adaptive Newton-step clamp (TRUST_REGION_EPS) against the flat
    DELTA_X_MAX one: both computational cost and the actual step-size
    behavior over a run are otherwise invisible in results.csv's one row
    per attempt. See art_core_fortran.jl's write_trajectory_csv docstring
    for the trajectory file's exact format.
  - TARGET_ATTEMPTS (opt-in, nothing by default) can replace TARGET_SUCCESSES
    as the stopping criterion: run a FIXED budget of attempts instead of
    running until a target success count is reached. Useful for comparing
    two settings' success RATE on equal footing -- targeting a success
    count means a worse-performing setting burns more attempts (and more
    wall-clock time) to reach the same count, confounding any such
    comparison; a fixed attempt budget doesn't.

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
# Opt-in: set to an Int to run a FIXED BUDGET of attempts instead of running
# until TARGET_SUCCESSES successes are found. TARGET_SUCCESSES is ignored
# when this is set (not a race between the two -- this one wins outright).
# Useful for a clean, unbiased success-RATE measurement (e.g. comparing
# TRUST_REGION_EPS settings): targeting successes means a worse-performing
# setting runs more attempts to reach the same count, so wall-clock time and
# n_attempts both confound the comparison; targeting a fixed attempt count
# does not.
TARGET_ATTEMPTS = nothing
N_WORKERS = max(1, Sys.CPU_THREADS - 1)   # leave a core free for the OS/master
MAX_ITERATIONS = 200                      # matches art_woconewer.f90's max_number_of_art_iterations

# When true: no per-attempt CSV log (and no flush() after every attempt --
# a real cost over thousands of failed attempts), no per-success dump
# files, no per-attempt trajectory files, no progress printlns, no
# diversity/RMSD report. Instead, at the
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
# 0.1, not the reference's original 0.025 -- validated in Python on an
# 80-seed, 500-atom sample (not yet at Julia scale, which is what this
# setting is here for you to check): raising the destabilization-phase
# push from 0.025 to 0.1 took the non-degenerate-abort success rate from
# 37/80 (46%) to 53/80 (66%). That's a net +16, but not a free lunch --
# 20 previously-stuck seeds started reaching the Newton-step branch, while
# 4 previously-fine seeds started degenerating instead (likely overshoot:
# a push this much bigger isn't uniformly an improvement, just a net one
# here). Most degenerate-mode failures never reach the Newton-step branch
# at all under the ORIGINAL 0.025 (81% of them, in that same sample) --
# that's the structural reason this lever matters much more than
# TRUST_REGION_EPS below, which only ever acts after that branch is
# reached. Set back to 0.025 for an A/B comparison against the original.
DELTA_LM = 0.1
FORCE_TOLERANCE = 1e-3
ZERO_EIGENVALUE_THRESHOLD = 0.1
MAX_ZERO_EIGENVALUE_THRESHOLD_NUMBER = 10
MAX_CONSECUTIVE_SECOND_EIG_NEGATIVE = 5
KICK_MAGNITUDE = 0.15
STARTING_FIRE_ITERATIONS = 5
LANCZOS_ITERATIONS = 10

# Trust-region-adaptive cap on the Newton-step branch's step size, in
# place of the flat DELTA_X_MAX clamp above (see art_core_fortran.jl's
# header for what this is). Set to `nothing` for now -- turned off so the
# DELTA_LM test above is an isolated, clean comparison against the 37/80
# vs 53/80 Python result, not conflated with a second lever at the same
# time. Once success meant "converged && is_saddle" this mattered (6/22
# -> 7/22, one rescued near-miss); now that success matches the
# reference's own looser criterion, that rescue effect is mostly moot --
# and it structurally can't touch most of what DELTA_LM above fixes,
# since 81% of degenerate failures never even reach the branch this
# clamp operates on. Re-enable (e.g. 0.1) if you want to layer it back on
# top once DELTA_LM is validated at scale.
TRUST_REGION_EPS = nothing

OUTPUT_DIR = "//home//schwyyzer//Desktop//Master Thesis//parallel_art_fortran_run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
DUMP_DIR = joinpath(OUTPUT_DIR, "dumps")
TRAJECTORY_DIR = joinpath(OUTPUT_DIR, "trajectories")
LOG_PATH = joinpath(OUTPUT_DIR, "results.csv")

mkpath(DUMP_DIR)
mkpath(TRAJECTORY_DIR)

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

    function attempt_wrapper(seed::Int, dump_path::String, trajectory_path::String)
        run_art_attempt_fortran(
            seed, positions0, data, pairs_static, box;
            dump_file = minimal_output ? nothing : dump_path,
            trajectory_file = minimal_output ? nothing : trajectory_path,
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
# Dispatch loop: keep every worker busy until TARGET_SUCCESSES successes
# (or, if TARGET_ATTEMPTS is set, until that many total attempts complete)
# (see run_parallel_no_tracking.jl for why this is wrapped in a function)
# ============================================================
function run_batch()
    seed_state = Ref(1)
    next_seed() = (s = seed_state[]; seed_state[] += 1; s)

    function dispatch!(w)
        s = next_seed()
        dump_path = joinpath(DUMP_DIR, "saddle_$(lpad(s, 6, '0')).dump")
        trajectory_path = joinpath(TRAJECTORY_DIR, "trajectory_$(lpad(s, 6, '0')).csv")
        return remotecall(attempt_wrapper, w, s, dump_path, trajectory_path)
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
            "second_eigenvalue_aborted,elapsed_time,dump_file,trajectory_file,error"
        )
        flush(log_io)
    end

    live_workers = Set(workers())

    # TARGET_ATTEMPTS, when set, overrides TARGET_SUCCESSES as the stopping
    # criterion entirely (see its definition above).
    done() = TARGET_ATTEMPTS === nothing ? length(successes) >= TARGET_SUCCESSES : n_attempts >= TARGET_ATTEMPTS

    while !done() && !isempty(live_workers)
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
                    trajectory_field = res.trajectory_file === nothing ? "" : res.trajectory_file
                    println(log_io,
                        "$(res.seed),$(res.success),$(res.kicked_atom),$(res.iteration),$(res.crit_eigenvalue)," *
                        "$(res.second_eigenvalue),$(res.converged),$(res.is_saddle),$(res.force_magnitude)," *
                        "$(res.mep_proximity),$(res.initial_energy),$(res.final_energy)," *
                        "$(res.degenerate_mode_aborted),$(res.second_eigenvalue_aborted),$(res.elapsed_time)," *
                        "$dump_field,$trajectory_field,$err_field"
                    )
                    flush(log_io)
                end

                if res.degenerate_mode_aborted || res.second_eigenvalue_aborted
                    guard_aborts += 1
                end

                progress() = TARGET_ATTEMPTS === nothing ?
                    "[$(length(successes))/$TARGET_SUCCESSES successes]" :
                    "[$n_attempts/$TARGET_ATTEMPTS attempts, $(length(successes)) successes]"

                if res.success
                    push!(successes, res)
                    if !MINIMAL_OUTPUT
                        elapsed = round(time() - start_time, digits=1)
                        @printf(
                            "%s SUCCESS  seed=%-6d atom=%-5d E=%.6f  iters=%-4d  eig=%.5f  elapsed=%.1fs  attempts_so_far=%d\n",
                            progress(), res.seed, res.kicked_atom,
                            res.final_energy, res.iteration, res.crit_eigenvalue,
                            elapsed, n_attempts
                        )
                    end
                else
                    if !MINIMAL_OUTPUT && n_attempts % 20 == 0
                        elapsed = round(time() - start_time, digits=1)
                        println("  ...$(progress()) so far, " *
                                "$guard_aborts guard aborts (degenerate-mode or second-eigenvalue), elapsed=$(elapsed)s")
                    end
                end

                if done()
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

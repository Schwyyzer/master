#=
Validate saddle points found by run_parallel(_no_tracking).jl: for each
successful row in a results.csv, read its dump file (which already
contains both the final saddle POSITIONS and the final MOVE vector --
see write_lammps_frame in lammps_and_document.jl, the "ITEM: MOVE"
block right after "ITEM: ATOMS"), push the system along that unstable-
mode direction in BOTH signs by a modest, fixed displacement, run a
full UNCONSTRAINED relaxation from each (the same relax(...,nothing,...)
path _overshoot_check already uses), and check the two results are
genuinely different configurations.

This is the standard check for a genuine first-order saddle: it should
connect exactly two basins, one on each side of the unstable direction.
If both sides relax back to the SAME configuration, the "saddle" isn't
doing what a real first-order saddle does, and is worth treating as
suspect (loose convergence criteria, or actually a higher-order/
non-saddle stationary point).

NOTE on push magnitude: this deliberately does NOT reuse the tiny MOVE
vector's own magnitude (~move_phase2_modifier, 0.001 by default) as the
push distance -- that's the size of the LAST phase-2 step, chosen to
converge onto the saddle precisely, not to escape it decisively. Right
at a saddle the net force is ~0 by construction, so a push that small
risks relaxing you right back to (numerically) the same point, or
taking a very long time to diverge either way, telling you nothing.

How large a push actually is needed varies a lot: some modes are
delocalized over many atoms, some (per an earlier real example) put
~93% of their weight on a single atom, and a fixed guess that works for
one can be a wild overshoot (straight into a neighbor) for the other.
So PUSH_MAGNITUDES below is a small LIST of candidate magnitudes,
scaled so the single MOST-displaced atom moves by that amount -- tried
smallest-first per side, per saddle, using the first one that relaxes
without throwing. Widen or adjust this list freely; the point isn't
this exact list, it's not committing to one guessed number.

Also worth knowing before reading the printed angles: your CSV's
max_phase2_angle_deg column sitting at ~179-180 degrees for essentially
every success is very likely a SIGN-CONVENTION artifact, not a real
direction reversal every single iteration -- calculate_moves flips the
sign of its returned eigenvector whenever dot(moves, F) > 0, and that
condition can flip from one iteration to the next as relax() drives F
back down; two directions ~180 degrees apart are the same physical axis,
just opposite bookkeeping sign. It doesn't affect this script (pushing
+move and -move covers both signs of the axis regardless of which one
happened to be reported last), but it means max_phase2_angle_deg isn't
by itself evidence of instability -- worth separately confirming with a
version of angle_between that takes the smaller of the two angles
(itself vs. its negation) if you want a cleaner diagnostic later.

NOTE: written without a local Julia install to run it against (same
caveat as this repo's other scripts) -- if something errors or a
number looks implausible, paste it back rather than assuming your data
is bad.

Run with: julia validate_saddles.jl
=#

include("Version1.1.jl")
using LinearAlgebra
using Printf

# ============================================================
# Settings -- edit these
# ============================================================
RESULTS_CSV = "results.csv"
LAMMPS_PATH = "FILL_ME_IN"   # the EXACT LAMMPS_PATH value the run_parallel*.jl
                              # settings block used to PRODUCE this specific CSV --
                              # not necessarily config.jl's inpath2 or this run_
                              # parallel*.jl's current default, if either changed
                              # since. Getting this wrong won't obviously error
                              # (same atom count/types is plausible even from the
                              # wrong file) -- it'll just silently validate against
                              # the wrong starting minimum, so this is deliberately
                              # not defaulted to a guess.
PUSH_MAGNITUDES = [0.01, 0.02, 0.05, 0.1, 0.2]   # tried in order per side, per
                        # saddle -- the smallest one that gets that side to
                        # relax without throwing is used, instead of betting
                        # everything on one guessed constant (your first two
                        # dumps showed ~93% of the mode's weight on a single
                        # atom, so the right push size may vary a lot saddle
                        # to saddle depending on how localized each mode is)
RMSD_SAME_THRESHOLD = 0.02   # below this, the two relaxed endpoints are called
                               # "the same configuration". Not a principled
                               # number -- the script always prints the actual
                               # RMSD too, so eyeball it and adjust freely

# ============================================================
# Dump file parsing (matches write_lammps_frame's exact format)
# ============================================================
function parse_dump_frame(path)
    lines = readlines(path)

    n_idx = findfirst(l -> strip(l) == "ITEM: NUMBER OF ATOMS", lines)
    n_idx === nothing && error("no 'ITEM: NUMBER OF ATOMS' found in $path")
    N = parse(Int, strip(lines[n_idx + 1]))

    atoms_idx = findfirst(l -> startswith(strip(l), "ITEM: ATOMS"), lines)
    atoms_idx === nothing && error("no 'ITEM: ATOMS' section found in $path")

    ids = zeros(Int, N)
    types = zeros(Int, N)
    positions = zeros(N, 3)
    for k in 1:N
        parts = split(strip(lines[atoms_idx + k]))
        ids[k] = parse(Int, parts[1])
        types[k] = parse(Int, parts[2])
        positions[k, 1] = parse(Float64, parts[3])
        positions[k, 2] = parse(Float64, parts[4])
        positions[k, 3] = parse(Float64, parts[5])
    end

    move_idx = findfirst(l -> strip(l) == "ITEM: MOVE", lines)
    move_idx === nothing && error(
        "no 'ITEM: MOVE' section in $path -- this dump predates the move " *
        "vector being saved, or was written by different code."
    )
    N_move = parse(Int, strip(lines[move_idx + 1]))
    N_move == N || error("MOVE section atom count ($N_move) != ATOMS section ($N) in $path")

    move = zeros(N, 3)
    for k in 1:N
        parts = split(strip(lines[move_idx + 1 + k]))
        move[k, 1] = parse(Float64, parts[1])
        move[k, 2] = parse(Float64, parts[2])
        move[k, 3] = parse(Float64, parts[3])
    end

    return positions, move, ids, types
end

# ============================================================
# results.csv parsing -- header-driven (column order doesn't matter,
# and it's fine if this CSV predates newer columns this repo has since
# added, e.g. phase2_step_size_min/max)
# ============================================================
function parse_results_csv(path)
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = split(lines[1], ",")
    col = Dict(String(name) => i for (i, name) in enumerate(header))
    for required in ("seed", "success", "dump_file", "final_energy")
        haskey(col, required) || error("CSV missing required column '$required'")
    end

    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) < length(header) && continue   # defensive: skip malformed rows
        field(name) = String(parts[col[name]])
        push!(rows, (
            seed = parse(Int, field("seed")),
            success = strip(field("success")) == "true",
            dump_file = field("dump_file"),
            final_energy = something(tryparse(Float64, field("final_energy")), NaN),
        ))
    end
    return rows
end

# COM-removed, minimum-image RMSD -- same definition run_parallel(_no_tracking).jl's
# print_diversity_summary already uses, so numbers here are directly comparable
function com_removed_rmsd(a, b, box_vec)
    dr = a .- b
    dr .-= box_vec .* round.(dr ./ box_vec)
    dr .-= sum(dr, dims=1) ./ size(dr, 1)
    return sqrt(sum(dr .^ 2) / size(dr, 1))
end

# ============================================================
# Main
# ============================================================
LAMMPS_PATH == "FILL_ME_IN" && error("set LAMMPS_PATH at the top of this script before running")

data0 = parse_lammps_data(LAMMPS_PATH)
box = (data0["lx"], data0["ly"], data0["lz"])
box_vec = [data0["lx"] data0["ly"] data0["lz"]]
positions0 = hcat(data0["x"], data0["y"], data0["z"])
pairs0 = build_neighbor_pairs(positions0, rc, box)
initial_energy = compute_energy(positions0, data0["cid"], box, epsilon_table, sigma_table, pairs0)

# Sanity check: every row in a given results.csv shares the same starting
# minimum, so its initial_energy column is one constant value -- if what
# THIS script just computed (from LAMMPS_PATH + the CURRENT config.jl's rc/
# epsilon_table/sigma_table) doesn't match that constant, this script is
# validating against a different potential than the one that actually
# produced the CSV, and nothing below is trustworthy until that's fixed.
# Paste the value from your CSV's initial_energy column here:
KNOWN_CSV_INITIAL_ENERGY = -13010.936594987796
if abs(initial_energy - KNOWN_CSV_INITIAL_ENERGY) > 1e-6
    println("*** WARNING: computed initial_energy = $initial_energy")
    println("*** does NOT match the CSV's recorded initial_energy = $KNOWN_CSV_INITIAL_ENERGY")
    println("*** (difference = $(initial_energy - KNOWN_CSV_INITIAL_ENERGY))")
    println("*** LAMMPS_PATH and/or config.jl (rc/epsilon_table/sigma_table) do NOT match")
    println("*** what actually produced this run -- fix that before trusting anything below.\n")
else
    println("initial_energy matches the CSV's recorded value -- LAMMPS_PATH and config.jl")
    println("are at least consistent with the starting point of this run.\n")
end

rows = parse_results_csv(RESULTS_CSV)
successes = filter(r -> r.success, rows)
println("$(length(successes)) successful saddle(s) to validate out of $(length(rows)) attempts in the CSV")
println("push magnitudes tried (per side) = $PUSH_MAGNITUDES, same-configuration RMSD threshold = $RMSD_SAME_THRESHOLD\n")

# Try pushing positions_saddle by `sign * move`, rescaled so the single
# most-displaced atom moves by each candidate magnitude in turn, until one
# relaxes without throwing. Returns (success, relaxed_positions, energy,
# magnitude_used).
function try_relax_push(positions_saddle, move, max_atom_disp, sign, magnitudes, data, box)
    for mag in magnitudes
        push_dir = sign .* move .* (mag / max_atom_disp)
        pos0 = positions_saddle .+ push_dir
        try
            pairs0 = build_neighbor_pairs(pos0, rc, box)
            pos = relax(pos0, data, pairs0, nothing, box)
            pairs_final = build_neighbor_pairs(pos, rc, box)
            e = compute_energy(pos, data["cid"], box, epsilon_table, sigma_table, pairs_final)
            return true, pos, e, mag
        catch
            continue
        end
    end
    return false, nothing, NaN, NaN
end

n_pass = 0
n_fail = 0
n_error = 0

for r in successes
    print("seed=$(r.seed)  dump=$(r.dump_file)")

    local positions_saddle, move, ids, types
    try
        positions_saddle, move, ids, types = parse_dump_frame(r.dump_file)
    catch e
        println("  SKIPPED (couldn't read/parse dump file: $(sprint(showerror, e)))")
        global n_error += 1
        continue
    end

    if ids != data0["id"] || types != data0["cid"]
        println("  SKIPPED (this dump's atom ids/types don't match LAMMPS_PATH's -- " *
                "wrong LAMMPS_PATH for this run?)")
        global n_error += 1
        continue
    end

    # Rescale by the MOST-displaced single atom, not the whole flattened 3N-
    # vector's norm -- see the header note on delocalization. Direction sign
    # is handled inside try_relax_push, not baked in here.
    per_atom_disp = sqrt.(sum(move .^ 2, dims=2))   # Nx1
    max_atom_disp = maximum(per_atom_disp)
    if max_atom_disp < 1e-14
        println("  SKIPPED (MOVE vector in dump is ~zero, nothing to push along)")
        global n_error += 1
        continue
    end
    print("  (raw max per-atom |move| in dump = $(round(max_atom_disp, digits=6)))")

    data = data0   # cid/id are identical (checked above); only positions differ per-call,
                    # and those are always passed explicitly, so this is safe to share

    try
        ok_plus, pos_plus, e_plus, mag_plus = try_relax_push(positions_saddle, move, max_atom_disp, 1.0, PUSH_MAGNITUDES, data, box)
        ok_minus, pos_minus, e_minus, mag_minus = try_relax_push(positions_saddle, move, max_atom_disp, -1.0, PUSH_MAGNITUDES, data, box)

        if !ok_plus || !ok_minus
            which = !ok_plus && !ok_minus ? "both sides" : (!ok_plus ? "the + side" : "the - side")
            println("\n  FAILED: $which never relaxed cleanly at any of $PUSH_MAGNITUDES\n")
            global n_error += 1
            continue
        end

        rmsd_plus_minus = com_removed_rmsd(pos_plus, pos_minus, box_vec)
        rmsd_plus_start = com_removed_rmsd(pos_plus, positions0, box_vec)
        rmsd_minus_start = com_removed_rmsd(pos_minus, positions0, box_vec)

        is_different = rmsd_plus_minus > RMSD_SAME_THRESHOLD
        verdict = is_different ? "PASS -- two distinct minima" :
                                  "SUSPICIOUS -- both sides relaxed to (nearly) the same configuration"
        global n_pass += is_different
        global n_fail += !is_different

        @printf(
            "\n  push_used=[+%.3g,-%.3g]  E_plus=%.6f  E_minus=%.6f  (saddle E=%.6f, barrier=%.6f)\n  RMSD(plus,minus)=%.5f  RMSD(plus,start)=%.5f  RMSD(minus,start)=%.5f\n  %s\n\n",
            mag_plus, mag_minus, e_plus, e_minus, r.final_energy, r.final_energy - initial_energy,
            rmsd_plus_minus, rmsd_plus_start, rmsd_minus_start,
            verdict
        )
    catch e
        println("\n  ERROR: $(sprint(showerror, e))\n")
        global n_error += 1
    end
end

println("=== Summary: $n_pass confirmed / $n_fail suspicious / $n_error skipped, out of $(length(successes)) successes ===")
if n_fail > 0
    println("Suspicious ones are worth a second look -- try widening PUSH_MAGNITUDES with some")
    println("larger values first (the range tried may simply not have escaped the saddle's")
    println("immediate neighborhood) before concluding the search criteria are too loose.")
end
if n_error > 0
    println("$n_error skipped/failed -- if these are all 'never relaxed cleanly at any of")
    println("PUSH_MAGNITUDES', re-check the initial_energy match printed above first: a")
    println("config/LAMMPS_PATH mismatch would produce exactly this kind of uniform failure.")
end

# traces/trace_tools.sh — sourced by the trace-set scripts (traces/{m2,gen,m3,vu,at,ctrl}/regen.sh):
# select and check the trace tools of a build of this tree (docs/design/06-software-
# verification.md §5.5-§5.6).
#   trace_tools BUILD_DIR   sets TT_GEN (tsr-trace-gen) and TT_REPLAY (tsr-trace-replay) and exits
#                           the script before anything is deleted if either is missing or lacks the
#                           --ops modes below (an older generator writes cmd.bin traces)
#   trace_replay SET...     replays every trace of the sets under the fixed and two random schedules
#                           (SW §4.4, §5.4) with the race detector fatal (TSR_SIM_RACE=off without one)
# The generator is deterministic: same build, same options -> byte-identical sets. TSR_SIM_SCHED and
# TSR_TRACE are cleared so that nothing in the environment changes a run.

trace_tools() {
    TT_BUILD=$(cd "$1" 2>/dev/null && pwd) || { echo "$0: no build directory $1" >&2; exit 2; }
    TT_GEN=$TT_BUILD/bin/tsr-trace-gen
    TT_REPLAY=$TT_BUILD/bin/tsr-trace-replay
    for f in "$TT_GEN" "$TT_REPLAY"; do
        [ -x "$f" ] || { echo "$0: $f not found: build the tsr-trace-gen and tsr-trace-replay targets" >&2; exit 2; }
    done
    if ! "$TT_GEN" --help 2>&1 | grep -q -- '--ops gen|m2|m3|vu|at|ctrl'; then
        echo "$0: $TT_GEN does not support --ops gen|m2|m3|vu|at|ctrl" >&2
        exit 2
    fi
    unset TSR_SIM_SCHED TSR_TRACE
}

# TSR_SIM_RACE=off replays without the race detector (a build without sw/sim/race.cmake).
trace_replay() {
    TSR_SIM_RACE=${TSR_SIM_RACE:-fatal} "$TT_REPLAY" -q --sched fixed --sched random:1 --sched random:2 "$@"
}

# --- trace_replay_kernel SET... ---
# The same three schedules through the `replay` kernel of the build's production image (BUILD_DIR/fw/
# tsr-fw.elf, SW §7.7), race detector fatal: there a trace is one launch, so an op that reads SPM
# nothing wrote in the trace is an SPM_STALE abort (SW §4.6, SW-4; the tile-only route of
# trace_replay has no launch window). Skipped with a note when the image is not built.
trace_replay_kernel() {
    local img=$TT_BUILD/fw/tsr-fw.elf
    if [ ! -f "$img" ]; then
        echo "$0: $img not built: the kernel-route replay (SPM_STALE check) is skipped" >&2
        return 0
    fi
    TSR_SIM_RACE=${TSR_SIM_RACE:-fatal} "$TT_REPLAY" -q --sched fixed --sched random:1 --sched random:2 \
        --replay-image "$img" "$@"
}
# --- end of trace_replay_kernel ---

# traces/v2_tools.sh — sourced by the v2 trace-set scripts (traces/{m2,gen,m3,vu,at,ctrl}/regen.sh):
# select and check the trace v2 tools of a build of this tree (P1.10, docs/v2/06-software-
# verification.md §5.5-§5.6).
#   v2_tools BUILD_DIR   sets V2_GEN (tsr-trace-gen v2) and V2_REPLAY (tsr-trace-replay v2) and exits
#                        the script before anything is deleted if either is missing or is a v1 tool
#                        (the v1 generator of tag v1-m4 has no --ops vu|at|ctrl and writes cmd.bin
#                        traces; build/v1 holds those, traces/v1_tools.sh selects them for traces/model)
#   v2_replay SET...     replays every trace of the sets under the fixed and two random schedules
#                        (SW §4.4, §5.4) with the race detector fatal (TSR_SIM_RACE=off without one)
# The generator is deterministic: same build, same options -> byte-identical sets. TSR_SIM_SCHED and
# TSR_TRACE are cleared so that nothing in the environment changes a run.

v2_tools() {
    V2_BUILD=$(cd "$1" 2>/dev/null && pwd) || { echo "$0: no build directory $1" >&2; exit 2; }
    V2_GEN=$V2_BUILD/bin/tsr-trace-gen
    V2_REPLAY=$V2_BUILD/bin/tsr-trace-replay
    for f in "$V2_GEN" "$V2_REPLAY"; do
        [ -x "$f" ] || { echo "$0: $f not found: build the tsr-trace-gen and tsr-trace-replay targets" >&2; exit 2; }
    done
    if ! "$V2_GEN" --help 2>&1 | grep -q -- '--ops gen|m2|m3|vu|at|ctrl'; then
        echo "$0: $V2_GEN is not tsr-trace-gen v2 (trace v2 ops traces)" >&2
        exit 2
    fi
    unset TSR_SIM_SCHED TSR_TRACE
}

# TSR_SIM_RACE=off replays without the race detector (a build without sw/sim/race.cmake).
v2_replay() {
    TSR_SIM_RACE=${TSR_SIM_RACE:-fatal} "$V2_REPLAY" -q --sched fixed --sched random:1 --sched random:2 "$@"
}

# --- added 2026-10-30 (tracegen, SPM_STALE): v2_replay_kernel SET... ---
# The same three schedules through the `replay` kernel of the build's production image (BUILD_DIR/fw/
# tsr-fw.elf, SW §7.7), race detector fatal: there a trace is one launch, so an op that reads SPM
# nothing wrote in the trace is an SPM_STALE abort (SW §4.6, V2-SW-4; the tile-only route of
# v2_replay has no launch window). Skipped with a note when the image is not built.
v2_replay_kernel() {
    local img=$V2_BUILD/fw/tsr-fw.elf
    if [ ! -f "$img" ]; then
        echo "$0: $img not built: the kernel-route replay (SPM_STALE check) is skipped" >&2
        return 0
    fi
    TSR_SIM_RACE=${TSR_SIM_RACE:-fatal} "$V2_REPLAY" -q --sched fixed --sched random:1 --sched random:2 \
        --replay-image "$img" "$@"
}
# --- end of the 2026-10-30 addition ---

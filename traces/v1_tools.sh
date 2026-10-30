# traces/v1_tools.sh — sourced by traces/model/regen.sh: select and check the older trace tools. (traces/m2,
# traces/gen and traces/m3 are made by tsr-trace-gen through traces/v2_tools.sh; the cmd.bin
# traces/model set stays as the input of tsr-trace-conv, SW §5.6.)
#
# The tree no longer has the generator, replay tool and per-exec TSR_TRACE dump that write cmd.bin
# traces, so that set can only be reproduced with a binary copy of the older tools: build/v1 (the
# default) or another build directory. `v1_tools BUILD_DIR [FILE...]`
# runs before a regen script deletes anything and exits the script if
#   - BUILD_DIR/bin lacks tsr-trace-gen, tsr-trace-replay or one of the extra FILEs named, or
#   - a one-trace probe of that tsr-trace-gen is not a cmd.bin trace (meta.json "version": 1 with the
#     run fields "cmd_addr_beat" and "args"): the current tsr-trace-gen writes `ops` traces, so a
#     current build is refused here instead of after the old set is gone.
# Then it points the dynamic loader at BUILD_DIR/bin and forces the simulated device:
#   - the build/v1 binaries carry the RUNPATH of the tree they were built in (build/xrt/bin of the
#     dev host, as an absolute path), so without LD_LIBRARY_PATH llama-completion would load
#     whatever libggml-tesseract that tree holds (current code once build/xrt is rebuilt from this tree);
#   - build/v1/env.sh selects the card (TSR_DEVICE=xrt, TSR_XCLBIN); it is never sourced here.
# Sets V1_BUILD (absolute), V1_GEN and V1_REPLAY.

v1_tools() {
    local b=$1 f probe
    shift
    V1_BUILD=$(cd "$b" 2>/dev/null && pwd) || { echo "$0: no build directory $b" >&2; exit 2; }
    V1_GEN=$V1_BUILD/bin/tsr-trace-gen
    V1_REPLAY=$V1_BUILD/bin/tsr-trace-replay
    for f in "$V1_GEN" "$V1_REPLAY" "$@"; do
        case $f in /*) ;; *) f=$V1_BUILD/bin/$f ;; esac
        [ -e "$f" ] || { echo "$0: $f not found: the older trace tools are in build/v1" >&2; exit 2; }
    done
    probe=$(mktemp -d "${TMPDIR:-/tmp}/tsr-v1probe.XXXXXX")
    if ! "$V1_GEN" -q --ops m2 -o "$probe/t" -n 1 -s 1 > /dev/null 2>&1 \
        || ! grep -q '"version": 1,' "$probe"/t/*/meta.json 2> /dev/null \
        || ! grep -q '"cmd_addr_beat"' "$probe"/t/*/meta.json 2> /dev/null \
        || ! grep -q '"args"' "$probe"/t/*/meta.json 2> /dev/null; then
        rm -rf "$probe"
        echo "$0: $V1_GEN does not write cmd.bin traces; these recipes need the older trace tools (build/v1)" >&2
        exit 2
    fi
    rm -rf "$probe"
    export LD_LIBRARY_PATH="$V1_BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TSR_DEVICE=sim
    unset TSR_XCLBIN
}

# traces/v1_tools.sh — sourced by traces/model/regen.sh: select and check the v1 tools. (traces/m2,
# traces/gen and traces/m3 are trace v2 sets from P1.10 on, made by tsr-trace-gen v2 through
# traces/v2_tools.sh; the v1 traces/model set stays as the input of tsr-trace-conv, SW §5.6.)
#
# X0 (docs/v2 EX §12) deleted tsr-trace-gen, tsr-trace-replay and the per-exec TSR_TRACE dump from
# the tree, so the v1 trace sets can only be reproduced with the v1 tools of tag v1-m4: the binary
# copy in build/v1 (the default) or a build of a v1-m4 checkout. `v1_tools BUILD_DIR [FILE...]`
# runs before a regen script deletes anything and exits the script if
#   - BUILD_DIR/bin lacks tsr-trace-gen, tsr-trace-replay or one of the extra FILEs named, or
#   - a one-trace probe of that tsr-trace-gen is not a v1 trace (meta.json "version": 1 with the v1
#     run fields "cmd_addr_beat" and "args"): a v2 tsr-trace-gen (P1.10) writes `ops` traces, so a
#     v2 build is refused here instead of after the old set is gone.
# Then it points the dynamic loader at BUILD_DIR/bin and forces the simulated device:
#   - the build/v1 binaries carry the RUNPATH of the tree they were built in (build/xrt/bin of the
#     dev host, as an absolute path), so without LD_LIBRARY_PATH llama-completion would load
#     whatever libggml-tesseract that tree holds (v2 code once build/xrt is rebuilt from this tree);
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
        [ -e "$f" ] || { echo "$0: $f not found: the v1 tools are in build/v1 (or a build of tag v1-m4)" >&2; exit 2; }
    done
    probe=$(mktemp -d "${TMPDIR:-/tmp}/tsr-v1probe.XXXXXX")
    if ! "$V1_GEN" -q --ops m2 -o "$probe/t" -n 1 -s 1 > /dev/null 2>&1 \
        || ! grep -q '"version": 1,' "$probe"/t/*/meta.json 2> /dev/null \
        || ! grep -q '"cmd_addr_beat"' "$probe"/t/*/meta.json 2> /dev/null \
        || ! grep -q '"args"' "$probe"/t/*/meta.json 2> /dev/null; then
        rm -rf "$probe"
        echo "$0: $V1_GEN does not write v1 (cmd.bin) traces; these recipes need the v1 tools (build/v1, tag v1-m4)" >&2
        exit 2
    fi
    rm -rf "$probe"
    export LD_LIBRARY_PATH="$V1_BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TSR_DEVICE=sim
    unset TSR_XCLBIN
}

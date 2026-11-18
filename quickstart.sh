#!/usr/bin/env bash
# quickstart.sh: try Tesseract without a card. Builds the stack, boots the firmware on the simulator, downloads a small
# model, runs it on the simulated NPU and writes a profile report of one decode token. Every step is idempotent: a
# second run goes straight to the model run.
#
#   ./quickstart.sh          build, check, download the model, run it on the simulator and profile one decode token
#   ./quickstart.sh chat     the same setup, then a chat with the model on the simulator
#   ./quickstart.sh -h       this text
#
#   TSR_JOBS        build jobs (default: the number of CPUs, at most 8)
#   TSR_BUILD       build tree (default: <repo>/build/dev)
#   TSR_QS_MODEL    a GGUF file to use instead of the download (no checksum)
#   TSR_QS_TOKENS   tokens of the profiled run (default 33)
#
# The model is Qwen2.5 0.5B Instruct Q8_0 (about 0.7 GB) from the Hugging Face repository Qwen/Qwen2.5-0.5B-Instruct-GGUF.
# It lands in <repo>/models/, and its size and sha256 are checked. The report is <build>/quickstart/report.html. The
# run uses the simulator only (TSR_DEVICE=sim); it never opens a card.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
N=7

# The model's pins. TSR_QS_URL, TSR_QS_SIZE, TSR_QS_SHA256 and TSR_QS_MODELS exist for the tests of this script.
MODEL_NAME=qwen2.5-0.5b-instruct-q8_0.gguf
MODEL_URL=${TSR_QS_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/$MODEL_NAME}
MODEL_SIZE=${TSR_QS_SIZE:-675710816}
MODEL_SHA256=${TSR_QS_SHA256:-ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e}
MODEL_DIR=${TSR_QS_MODELS:-$here/models}
PROMPT_TEXT="The history of the printing press begins"

say() { echo "[quickstart] $*"; }
# A path as shown to the visitor: relative to the checkout when the script runs from its top directory.
show() { if [ "$PWD" = "$here" ]; then echo "${1#"$here"/}"; else echo "$1"; fi; }
die() { echo "[quickstart] error: $*" >&2; exit 1; }
step() { say "$1/$N $2 ..."; }

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# A command whose output goes to a log file; on failure the end of the log is printed and the script stops.
logged() {                      # logged LOG DESCRIPTION COMMAND...
    local log=$1 what=$2
    shift 2
    if ! "$@" > "$log" 2>&1; then
        tail -n 40 "$log" >&2
        die "$what failed; the full output is in $log"
    fi
}

set_build() {
    build=${TSR_BUILD:-$here/build/dev}
    case "$build" in /*) ;; *) build=$PWD/$build ;; esac
    qs="$build/quickstart"
    jobs=${TSR_JOBS:-$(nproc 2>/dev/null || echo 4)}
    if [ "$jobs" -gt 8 ] 2>/dev/null && [ -z "${TSR_JOBS:-}" ]; then jobs=8; fi
    case "$jobs" in ''|*[!0-9]*) die "TSR_JOBS must be a number, not '$jobs'" ;; esac
    # How the visitor reruns this script: from where they are, with the settings that differ from the defaults.
    self=./quickstart.sh
    [ "$PWD" = "$here" ] || self=$here/quickstart.sh
    case "$self" in *" "*) self=\"$self\" ;; esac
    if [ -n "${TSR_BUILD:-}" ]; then self="TSR_BUILD=$(printf '%q' "$TSR_BUILD") $self"; fi
    if [ -n "${TSR_QS_MODEL:-}" ]; then self="TSR_QS_MODEL=$(printf '%q' "$TSR_QS_MODEL") $self"; fi
}

need() { command -v "$1" > /dev/null 2>&1 || die "$1 not found: $2"; }

file_size() { stat -c %s "$1"; }
file_sha() { sha256sum "$1" | cut -d' ' -f1; }

step_prerequisites() {
    step 1 "prerequisites"
    [ "$(id -u)" != 0 ] || die "do not run as root: everything installs in user space"
    [ "$(uname -s)" = Linux ] || die "Linux is required (this is $(uname -s))"
    [ "$(uname -m)" = x86_64 ] || die "x86-64 is required (the pinned RISC-V toolchain is linux-x64; this is $(uname -m))"
    need git "install git"
    need cmake "install CMake 3.20 or newer"
    need make "install make"
    need cc "install a C compiler"
    need c++ "install a C++ compiler"
    need python3 "install Python 3"
    need curl "install curl"
    need sha256sum "install coreutils"
    local v
    v=$(cmake --version | sed -n '1s/[^0-9]*\([0-9][0-9.]*\).*/\1/p')
    [ -n "$v" ] || die "cannot read the CMake version"
    [ "$(printf '%s\n3.20\n' "$v" | sort -V | head -n 1)" = 3.20 ] || die "CMake $v is too old: 3.20 or newer is required"
}

step_sources() {
    step 2 "submodules and patches"
    if [ ! -e "$here/third_party/llama.cpp/.git" ]; then
        [ -e "$here/.git" ] || die "$here is not a git checkout, so the submodules cannot be fetched (clone with: git clone --recurse-submodules)"
        say "  fetching the submodules (git submodule update --init)"
        git -C "$here" submodule update --init
    fi
    "$here/third_party/apply-patches.sh" | sed 's/^/[quickstart]   /'
}

step_toolchain() {
    step 3 "RISC-V toolchain"
    mkdir -p "$qs"
    logged "$qs/toolchain.log" "the RISC-V toolchain install" "$here/sw/ctrl/tools/install-toolchain.sh" --steps xpack,check
    say "  $(grep -a 'check: ok' "$qs/toolchain.log" | sed 's/.*check: ok (\([^;]*\);.*/\1/' | head -n 1)"
}

step_build() {
    if [ -f "$build/CMakeCache.txt" ]; then step 4 "build ($jobs jobs)"; else step 4 "build ($jobs jobs; the first build takes a while)"; fi
    mkdir -p "$qs"
    local t0=$SECONDS
    if [ ! -f "$build/CMakeCache.txt" ]; then
        say "  configure, log: $(show "$qs")/configure.log"
        logged "$qs/configure.log" "the CMake configure" cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release
    fi
    say "  build, log: $(show "$qs")/build.log"
    logged "$qs/build.log" "the build" cmake --build "$build" -j"$jobs" \
        --target tsr_fw llama-cli llama-completion tsr-selftest tsr-mon-probe
    say "  built in $((SECONDS - t0)) s"
}

step_selftest() {
    step 5 "selftest on the simulator"
    mkdir -p "$qs"
    TSR_IMAGE=$build/fw/tsr-fw.elf logged "$qs/selftest.log" "tsr-selftest" "$build/bin/tsr-selftest" --device sim
    say "  $(tail -n 1 "$qs/selftest.log")"
}

# Sets model to the GGUF file to run. Prints one line.
step_model() {
    step 6 "model"
    if [ -n "${TSR_QS_MODEL:-}" ]; then
        [ -f "$TSR_QS_MODEL" ] || die "TSR_QS_MODEL is not a file: $TSR_QS_MODEL"
        model=$TSR_QS_MODEL
        say "  using $(show "$model") (TSR_QS_MODEL, not checked)"
        return 0
    fi
    model=$MODEL_DIR/$MODEL_NAME
    mkdir -p "$MODEL_DIR"
    if [ -f "$model" ] && [ "$(file_size "$model")" = "$MODEL_SIZE" ] && [ "$(file_sha "$model")" = "$MODEL_SHA256" ]; then
        say "  $(show "$model") is present, size and sha256 match"
        return 0
    fi
    if [ -e "$model" ]; then
        say "  $model does not match the pinned size and sha256, downloading it again"
        rm -f "$model"
    fi
    local part=$model.part have=0
    [ ! -f "$part" ] || have=$(file_size "$part")
    if [ "$have" -gt "$MODEL_SIZE" ]; then rm -f "$part"; have=0; fi
    if [ "$have" -lt "$MODEL_SIZE" ]; then
        say "  downloading $MODEL_URL"
        if [ "$have" -gt 0 ]; then
            say "  to $model ($((MODEL_SIZE / 1000000)) MB, resuming at $((have / 1000000)) MB)"
        else
            say "  to $model ($((MODEL_SIZE / 1000000)) MB)"
        fi
        curl -L --fail --retry 3 -C - -o "$part" "$MODEL_URL" || die "the download failed (run $self again to resume it)"
    fi
    [ "$(file_size "$part")" = "$MODEL_SIZE" ] || die "the downloaded file has $(file_size "$part") bytes, not $MODEL_SIZE; $part is kept for a resume"
    say "  checking sha256"
    local sha
    sha=$(file_sha "$part")
    if [ "$sha" != "$MODEL_SHA256" ]; then
        rm -f "$part"
        die "sha256 of the download is $sha, not $MODEL_SHA256; the file is removed"
    fi
    mv "$part" "$model"
    say "  downloaded and checked: $(show "$model")"
}

sim_env() {
    export TSR_DEVICE=sim TSR_SIM_CARD=k4qpipe TSR_SIM_GDN=1 LLAMA_ATTN_ROT_DISABLE=1 TSR_IMAGE=$build/fw/tsr-fw.elf
}

step_run() {
    step 7 "run on the simulator and profile one decode token"
    local ntok=${TSR_QS_TOKENS:-33} dump="$qs/dump" t0=$SECONDS
    case "$ntok" in ''|*[!0-9]*|0) die "TSR_QS_TOKENS must be a positive number, not '$ntok'" ;; esac
    rm -rf "$dump"
    say "  decoding $ntok tokens with every layer on the simulated NPU, log: $(show "$dump")/sched_and_run.log"
    if ! TSR_BUILD=$build PROMPT=$PROMPT_TEXT "$here/sw/tools/prof/capture_sim.sh" "$model" "$dump" "$ntok" > "$qs/capture.log" 2>&1; then
        tail -n 30 "$dump/sched_and_run.log" >&2 2> /dev/null || tail -n 30 "$qs/capture.log" >&2 || true
        die "the simulator run failed; see $qs/capture.log"
    fi
    local text
    text=$(cat "$dump/run.out")
    say "  prompt: $PROMPT_TEXT"
    say "  generated text:"
    printf '%s\n' "${text#"$PROMPT_TEXT"}" | sed 's/^/[quickstart]    /'
    say "  run time:  $((SECONDS - t0)) s on this host"
    logged "$qs/report.log" "the profile report" \
        python3 "$here/sw/tools/prof/prof_report.py" sched "$dump/sched_and_run.log" "$dump/sched.json"
    logged "$qs/report.log" "the profile report" \
        python3 "$here/sw/tools/prof/prof_report.py" report --sim "$dump" --out "$qs" --model-name "Qwen2.5 0.5B Instruct Q8_0" \
            --commands "# commands (./quickstart.sh)
sw/tools/prof/capture_sim.sh MODEL DUMP $ntok
sw/tools/prof/prof_report.py report --sim DUMP --out OUT"
    python3 -I - "$qs/summary.json" <<'PYEOF'
import json
import sys

d = json.load(open(sys.argv[1]))
s = d["summary"]
p = s["perf"]
cyc = p["CYCLES"]


def line(label, value):
    print("[quickstart]   %-34s %s" % (label, value))


print("[quickstart] one decode token (launch %d), in the simulator's cycle model:" % d["launch"])
line("device cycles", "{:,}".format(int(cyc)))
line("device time at %.0f MHz" % (s["core_hz"] / 1e6), "%.2f ms" % s["t_dev_ms"])
for name, keys in (("MVM busy", ("MVM_CYCLES",)), ("  of it attention (ATTQK, ATTPV)", ("ATT_QK_CYCLES", "ATT_PV_CYCLES")),
                   ("VU busy", ("VU_CYCLES",)), ("DMA busy", ("DMA_CYCLES",)), ("QACT busy", ("QACT_CYCLES",))):
    line(name, "%5.1f %% of the launch's cycles" % (100.0 * sum(p.get(k, 0) for k in keys) / cyc))
PYEOF
    local rq
    rq=$(show "$qs")
    say "  report: $rq/report.html"
    say "  Perfetto trace: $rq/trace.perfetto.json, numbers: $rq/summary.json"
    say "next:"
    say "  open $rq/report.html in a browser"
    say "  $self chat      chat with the model on the simulated NPU"
    say "  $(show "$build")/bin/tsr-top      in a second terminal while the chat runs"
}

step_chat() {
    step 7 "chat with the model on the simulated NPU (type /exit to leave)"
    sim_env
    exec "$build/bin/llama-cli" -m "$model" -ngl 99 -fa on -ctk q8_0 -ctv q8_0
}

main() {
    local mode=run
    case "${1:-}" in
        "")          ;;
        chat)        mode=chat ;;
        -h|--help)   usage; exit 0 ;;
        *)           usage >&2; die "unknown argument '$1'" ;;
    esac
    [ $# -le 1 ] || die "too many arguments"
    set_build
    step_prerequisites
    step_sources
    step_toolchain
    step_build
    step_selftest
    step_model
    if [ "$mode" = chat ]; then step_chat; else step_run; fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi

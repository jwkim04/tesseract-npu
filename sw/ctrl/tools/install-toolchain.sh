#!/usr/bin/env bash
# sw/ctrl/tools/install-toolchain.sh - install the pinned RISC-V tools in user space, without sudo
# (docs/v2/06-software-verification.md §9.6, V2-SW-7; README risk 14). The pins are data in
# docs/v2/toolchain.lock; this script installs exactly those, verifies them and records what it
# found in <prefix>/installed.lock.
#
# Usage: sw/ctrl/tools/install-toolchain.sh [options]
#   --prefix DIR   install root (default: $TSR_TOOLS_PREFIX, else <repo>/build/tools/riscv, which
#                  .gitignore covers; SW §9.6's ~/.local/opt layout: --prefix ~/.local/opt/tsr-riscv)
#   --lock FILE    pin file (default: <repo>/docs/v2/toolchain.lock)
#   --steps LIST   comma-separated subset of xpack,dtc,spike,sources,check,sanity (default: all,
#                  always run in that order)
#   --jobs N       parallel make jobs (default: $TSR_JOBS, else 4)
#   --force        redo the selected steps even when their stamps say they are done
#   --offline      never download: use what is already under the prefix (still verified)
#   --python PY    Python >= 3.8 (stdlib only) for the sanity step (default: $TSR_PYTHON, else python3)
#   -h, --help
#
# Steps and results (all under the prefix):
#   xpack    dl/xpack-riscv-none-elf-gcc-<ver>-linux-x64.tar.gz (+ .sha) and
#            xpack-riscv-none-elf-gcc-<ver>/ (gcc, g++, binutils, newlib, riscv-none-elf-gdb).
#            The archive is untrusted data: its size and sha256 must equal the lock and the
#            release's published .sha file; every member must lie inside its single top directory
#            (no absolute path, no '..', no symlink leaving it, no device or hard-link member); it
#            is unpacked without owners into a fresh staging directory, then moved into place.
#            Nothing from the archive is executed during installation.
#   dtc      src/dtc at the pinned commit, built with make (-j N) into dtc/bin (Spike's build
#            needs dtc; at run time Spike needs it only without --disable-dtb).
#   spike    src/riscv-isa-sim at the pinned commit, configured without Boost, built out of tree
#            (src/riscv-isa-sim/build, -j N), installed into spike/.
#   sources  src/riscv-tests (+ its env submodule) and src/riscv-arch-test (sparse, blobless) at
#            the pinned commits; built and run by the rvtests harness, not here.
#   check    compiler version, multilib rv32imc/ilp32 (or the rv32im/ilp32 fallback of SW §6.1),
#            libgcc path, GDB runs, a C++20 freestanding probe with the firmware flags of the lock
#            compiles and links against libgcc; writes installed.lock and env.sh.
#   sanity   sw/sim/tests/rvtests on Spike: riscv-tests as published (profile "upstream"), the
#            riscv-tests in the test-profile env + riscv-arch-test (profile "test"), 20 random
#            programs. Outputs in sanity/.
# Every step is idempotent: a finished step writes .stamps/<step> holding the pins it used; a
# re-run with unchanged pins and intact outputs skips it. Git sources are checked against commit
# and tree ids from the lock. The script refuses to run as root and never calls sudo.
set -euo pipefail
umask 022

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../.." && pwd)
PREFIX=${TSR_TOOLS_PREFIX:-$REPO/build/tools/riscv}
LOCK=$REPO/docs/v2/toolchain.lock
JOBS=${TSR_JOBS:-4}
STEPS_ALL="xpack dtc spike sources check sanity"
STEPS=$STEPS_ALL
FORCE=0
OFFLINE=0
PY=${TSR_PYTHON:-python3}
CUR_STEP=setup

log() { printf '[install-toolchain] %s\n' "$*" >&2; }
die() { printf '[install-toolchain] error (%s): %s\n' "$CUR_STEP" "$*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX=$2; shift 2 ;;
        --prefix=*) PREFIX=${1#*=}; shift ;;
        --lock) LOCK=$2; shift 2 ;;
        --lock=*) LOCK=${1#*=}; shift ;;
        --steps) STEPS=${2//,/ }; shift 2 ;;
        --steps=*) STEPS=${1#*=}; STEPS=${STEPS//,/ }; shift ;;
        -j|--jobs) JOBS=$2; shift 2 ;;
        --jobs=*) JOBS=${1#*=}; shift ;;
        --force) FORCE=1; shift ;;
        --offline) OFFLINE=1; shift ;;
        --python) PY=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done
for s in $STEPS; do
    case " $STEPS_ALL " in *" $s "*) ;; *) die "unknown step '$s' (steps: $STEPS_ALL)" ;; esac
done
[ "$(id -u)" != 0 ] || die "do not run as root: everything installs in user space"
[ -f "$LOCK" ] || die "lock file $LOCK not found"
case "$JOBS" in ''|*[!0-9]*) die "--jobs needs a number" ;; esac
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)

need() {
    local t
    for t in "$@"; do
        command -v "$t" >/dev/null 2>&1 || die "host tool '$t' not found"
    done
}

# lock_get KEY: the value of "KEY = value" in the lock (the file is parsed, never sourced).
lock_get() {
    local v
    v=$(awk -v k="$1" '
        index($0, "#") == 1 { next }
        { i = index($0, " = "); if (i > 0 && substr($0, 1, i - 1) == k) { print substr($0, i + 3); f = 1; exit } }
        END { if (!f) exit 1 }' "$LOCK") || die "$LOCK has no key '$1'"
    printf '%s' "$v"
}

stamp_ok() { [ "$FORCE" = 0 ] && [ -f "$PREFIX/.stamps/$1" ] && [ "$(cat "$PREFIX/.stamps/$1")" = "$2" ]; }
stamp_set() { mkdir -p "$PREFIX/.stamps"; printf '%s' "$2" > "$PREFIX/.stamps/$1"; }
sha256_of() { sha256sum "$1" | awk '{ print $1 }'; }

download() {    # URL OUT
    [ "$OFFLINE" = 0 ] || die "--offline: $2 is missing or fails its check and cannot be downloaded"
    log "downloading $1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --connect-timeout 30 -o "$2.part" "$1" || die "download failed: $1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2.part" "$1" || die "download failed: $1"
    else
        die "need curl or wget"
    fi
    mv "$2.part" "$2"
}

# check_archive ARCHIVE TOPDIR: refuse members outside TOPDIR/, '..' components, whitespace in
# names, members other than files / directories / symlinks, and symlinks resolving outside TOPDIR.
check_archive() {
    local arc=$1 top=$2 bad name target r
    bad=$(tar -tzf "$arc" | awk -v t="$top" '
        { if (!(index($0, t "/") == 1 || $0 == t) || $0 ~ /(^|\/)\.\.(\/|$)/ || $0 ~ /[[:space:]]/) { print; exit } }')
    [ -z "$bad" ] || die "archive member '$bad' is not a plain path inside $top/"
    bad=$(tar -tvzf "$arc" | awk '{ c = substr($1, 1, 1); if (c != "-" && c != "d" && c != "l") { print; exit } }')
    [ -z "$bad" ] || die "archive has a member that is not a file, directory or symlink: $bad"
    while IFS=$'\t' read -r name target; do
        [ -n "$name" ] || continue
        case "$target" in /*) die "archive symlink $name -> $target is absolute" ;; esac
        r=$(realpath -m "/__tsr_archive_root__/$(dirname "$name")/$target")
        case "$r" in
            "/__tsr_archive_root__/$top"/*) ;;
            *) die "archive symlink $name -> $target leaves $top/" ;;
        esac
    done < <(tar -tvzf "$arc" | awk 'substr($1, 1, 1) == "l" {
                 i = index($0, " -> "); n = split(substr($0, 1, i - 1), f, " ");
                 print f[n] "\t" substr($0, i + 4) }')
}

# git_pinned DIR REPO COMMIT TREE [SPARSE_PATH...]: DIR checked out at COMMIT (fetched by id,
# depth 1; blobless sparse checkout of the given paths if any), verified against COMMIT and TREE.
git_pinned() {
    local dir=$1 repo=$2 commit=$3 tree=$4 p
    shift 4
    if [ ! -d "$dir/.git" ] || [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$commit" ]; then
        [ "$OFFLINE" = 0 ] || die "--offline: $dir is not at $commit"
        log "fetching $repo at $commit"
        rm -rf "$dir"
        mkdir -p "$dir"
        git -C "$dir" init -q
        git -C "$dir" remote add origin "$repo"
        if [ $# -gt 0 ]; then
            local pats=()
            for p in "$@"; do pats+=("/$p"); done
            git -C "$dir" sparse-checkout set --no-cone "${pats[@]}"
            git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit" || die "git fetch $repo $commit failed"
        else
            git -C "$dir" fetch -q --depth 1 origin "$commit" || die "git fetch $repo $commit failed"
        fi
        git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD || die "checkout of $commit in $dir failed"
    fi
    [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || die "$dir: HEAD is not $commit"
    [ "$(git -C "$dir" rev-parse 'HEAD^{tree}')" = "$tree" ] || die "$dir: tree of $commit is not $tree"
}

# --------------------------------------------------------------------------------------------
step_xpack() {
    local url sha_url sha size dir pfx arc id pub got stage
    url=$(lock_get xpack_gcc.url); sha_url=$(lock_get xpack_gcc.sha_url)
    sha=$(lock_get xpack_gcc.sha256); size=$(lock_get xpack_gcc.size)
    dir=$(lock_get xpack_gcc.dir); pfx=$(lock_get xpack_gcc.prefix)
    id="$url $sha"
    if stamp_ok xpack "$id" && [ -x "$PREFIX/$dir/bin/${pfx}gcc" ] && [ -x "$PREFIX/$dir/bin/${pfx}gdb" ]; then
        log "xpack: $dir is installed (stamp matches)"
        return 0
    fi
    need tar gzip sha256sum realpath awk stat
    mkdir -p "$PREFIX/dl"
    arc="$PREFIX/dl/${url##*/}"
    if [ -f "$arc" ] && [ "$(sha256_of "$arc")" != "$sha" ]; then
        log "xpack: $arc fails its sha256, downloading it again"
        rm -f "$arc"
    fi
    [ -f "$arc" ] || download "$url" "$arc"
    if [ "$OFFLINE" = 0 ]; then
        download "$sha_url" "$arc.sha"
        pub=$(awk '{ print $1; exit }' "$arc.sha")
        [ "$pub" = "$sha" ] || die "the published sha256 ($pub, $sha_url) differs from the lock ($sha)"
    fi
    got=$(sha256_of "$arc")
    [ "$got" = "$sha" ] || die "sha256 of $arc is $got, the lock says $sha"
    [ "$(stat -c %s "$arc")" = "$size" ] || die "size of $arc is not $size"
    check_archive "$arc" "$dir"
    stage=$(mktemp -d "$PREFIX/.stage-xpack.XXXXXX")
    tar -xzf "$arc" -C "$stage" --no-same-owner --no-same-permissions || die "unpacking $arc failed"
    [ "$(ls -A "$stage")" = "$dir" ] || die "$arc does not unpack to exactly $dir/"
    rm -rf "${PREFIX:?}/$dir"
    mv "$stage/$dir" "$PREFIX/$dir"
    rmdir "$stage"
    stamp_set xpack "$id"
    log "xpack: installed $PREFIX/$dir"
}

step_dtc() {
    local repo commit tree flags target id src
    repo=$(lock_get dtc.repo); commit=$(lock_get dtc.commit); tree=$(lock_get dtc.tree)
    flags=$(lock_get dtc.make_flags); target=$(lock_get dtc.make_target)
    id="$repo $commit $flags $target"
    if stamp_ok dtc "$id" && [ -x "$PREFIX/dtc/bin/dtc" ]; then
        log "dtc: installed (stamp matches)"
        return 0
    fi
    need git make cc bison flex
    src="$PREFIX/src/dtc"
    git_pinned "$src" "$repo" "$commit" "$tree"
    git -C "$src" clean -fdxq
    rm -rf "$PREFIX/dtc"
    log "dtc: building $(lock_get dtc.tag) with -j$JOBS (log: $PREFIX/src/dtc.build.log)"
    # shellcheck disable=SC2086  # flags is a list of make variables
    make -C "$src" -j"$JOBS" $flags PREFIX="$PREFIX/dtc" "$target" > "$PREFIX/src/dtc.build.log" 2>&1 \
        || die "dtc build failed; see $PREFIX/src/dtc.build.log"
    "$PREFIX/dtc/bin/dtc" --version >&2 || die "the built dtc does not run"
    stamp_set dtc "$id"
}

step_spike() {
    local repo commit tree flags id src b
    repo=$(lock_get spike.repo); commit=$(lock_get spike.commit); tree=$(lock_get spike.tree)
    flags=$(lock_get spike.configure_flags)
    id="$repo $commit $flags"
    if stamp_ok spike "$id" && [ -x "$PREFIX/spike/bin/spike" ]; then
        log "spike: installed (stamp matches)"
        return 0
    fi
    [ -x "$PREFIX/dtc/bin/dtc" ] || die "Spike's configure needs dtc: run the dtc step first"
    need git make gcc g++
    src="$PREFIX/src/riscv-isa-sim"
    git_pinned "$src" "$repo" "$commit" "$tree"
    b="$src/build"
    rm -rf "$b" "$PREFIX/spike"
    mkdir -p "$b"
    log "spike: configuring and building $commit with -j$JOBS (logs: $PREFIX/src/spike.*.log)"
    # shellcheck disable=SC2086  # flags is a list of configure options
    (cd "$b" && PATH="$PREFIX/dtc/bin:$PATH" ../configure --prefix="$PREFIX/spike" $flags) \
        > "$PREFIX/src/spike.configure.log" 2>&1 || die "configure failed; see $PREFIX/src/spike.configure.log"
    PATH="$PREFIX/dtc/bin:$PATH" make -C "$b" -j"$JOBS" > "$PREFIX/src/spike.build.log" 2>&1 \
        || die "build failed; see $PREFIX/src/spike.build.log"
    make -C "$b" install > "$PREFIX/src/spike.install.log" 2>&1 \
        || die "install failed; see $PREFIX/src/spike.install.log"
    "$PREFIX/spike/bin/spike" --help > /dev/null 2>&1 || die "the installed spike does not run"
    stamp_set spike "$id"
}

step_sources() {
    local rt env_c sub at_paths id
    rt="$PREFIX/src/riscv-tests"
    env_c=$(lock_get riscv_tests.env_commit)
    id="$(lock_get riscv_tests.commit) $env_c $(lock_get arch_test.commit) $(lock_get arch_test.paths)"
    if stamp_ok sources "$id" && [ -f "$rt/env/p/riscv_test.h" ] \
        && [ -f "$PREFIX/src/riscv-arch-test/riscv-test-suite/env/arch_test.h" ]; then
        log "sources: present (stamp matches)"
        return 0
    fi
    need git awk
    git_pinned "$rt" "$(lock_get riscv_tests.repo)" "$(lock_get riscv_tests.commit)" "$(lock_get riscv_tests.tree)"
    sub=$(git -C "$rt" ls-tree HEAD env | awk '{ print $3 }')
    [ "$sub" = "$env_c" ] || die "riscv-tests records env at $sub, the lock says $env_c"
    git_pinned "$rt/env" "$(lock_get riscv_tests.env_repo)" "$env_c" "$(lock_get riscv_tests.env_tree)"
    read -r -a at_paths <<< "$(lock_get arch_test.paths)"
    git_pinned "$PREFIX/src/riscv-arch-test" "$(lock_get arch_test.repo)" "$(lock_get arch_test.commit)" \
        "$(lock_get arch_test.tree)" "${at_paths[@]}"
    stamp_set sources "$id"
}

step_check() {
    local dir pfx bin gcc gdb ver mlib mdir libgcc probe flags dtcv spikep line
    dir=$(lock_get xpack_gcc.dir); pfx=$(lock_get xpack_gcc.prefix)
    bin="$PREFIX/$dir/bin"; gcc="$bin/${pfx}gcc"; gdb="$bin/${pfx}gdb"
    [ -x "$gcc" ] || die "$gcc not found: run the xpack step"
    ver=$("$gcc" -dumpversion)
    [ "$ver" = "$(lock_get xpack_gcc.gcc_version)" ] || die "$gcc reports version $ver"
    "$gcc" --version | head -n 1 | grep -q xPack || die "$gcc is not the xPack build"
    mlib=$(lock_get xpack_gcc.multilib)
    mdir=$("$gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -print-multi-directory)
    if [ "$mdir" != "$mlib" ]; then
        log "check: -march=rv32imc_zicsr_zifencei selects multilib '$mdir', not $mlib"
        [ "$mdir" = "$(lock_get xpack_gcc.multilib_fallback)" ] || die "neither $mlib nor the fallback multilib is selected"
    fi
    libgcc=$("$gcc" -march=rv32imc_zicsr_zifencei -mabi=ilp32 -print-libgcc-file-name)
    [ -f "$libgcc" ] || die "libgcc $libgcc missing"
    "$gdb" --version | head -n 1 | grep -q "$(lock_get xpack_gcc.gdb_version)" || die "$gdb is not GDB $(lock_get xpack_gcc.gdb_version)"
    "$gdb" -nx -batch -ex 'set architecture riscv:rv32' > /dev/null || die "$gdb does not run"
    # The firmware flags of SW §6.1 (lock key firmware.cxxflags): a freestanding C++20 unit with
    # 64-bit division (libgcc) and CSR / fence.i, linked at 0 against libgcc only.
    probe="$PREFIX/.probe"
    rm -rf "$probe"; mkdir -p "$probe"
    cat > "$probe/probe.cpp" <<'EOF'
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <type_traits>
template <typename T> struct box { T v; constexpr T get() const { return v; } };
static_assert(std::is_unsigned_v<uint32_t>, "freestanding <type_traits>");
static volatile uint32_t sink;
extern "C" uint64_t probe_div(uint64_t a, uint64_t b) { return a / b + a % b; }
extern "C" [[noreturn]] void _start() {
    constexpr box<uint32_t> b{7};
    for (uint32_t i = 0; i < b.get(); ++i) sink = sink + i;
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    asm volatile("fence.i");
    sink = (uint32_t)probe_div(sink + c, 3);
    for (;;) {}
}
EOF
    flags=$(lock_get firmware.cxxflags)
    # shellcheck disable=SC2086  # flags is a list of compiler options
    "$bin/${pfx}g++" $flags -Wall -Wextra -Werror -c "$probe/probe.cpp" -o "$probe/probe.o" \
        || die "the firmware flags do not compile a freestanding C++20 unit"
    # shellcheck disable=SC2086
    "$bin/${pfx}g++" $flags -Wl,-Ttext=0 "$probe/probe.o" "$libgcc" -o "$probe/probe.elf" \
        || die "the firmware probe does not link against $libgcc"
    "$bin/${pfx}readelf" -A "$probe/probe.elf" | grep -q 'c2p0_zicsr2p0_zifencei2p0' \
        || die "the probe is not rv32imc_zicsr_zifencei"
    dtcv=absent; [ -x "$PREFIX/dtc/bin/dtc" ] && dtcv=$("$PREFIX/dtc/bin/dtc" --version 2>&1 | head -n 1)
    spikep=absent; [ -x "$PREFIX/spike/bin/spike" ] && spikep="$PREFIX/spike/bin/spike"
    {
        echo "# $PREFIX/installed.lock - written by sw/ctrl/tools/install-toolchain.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# pins: $LOCK"
        echo "prefix = $PREFIX"
        echo "xpack_gcc.gcc = $gcc"
        echo "xpack_gcc.reported = $("$gcc" --version | head -n 1)"
        echo "xpack_gcc.multilib_selected = $mdir"
        echo "xpack_gcc.libgcc = $libgcc"
        echo "xpack_gcc.gdb = $gdb"
        echo "xpack_gcc.gdb_reported = $("$gdb" --version | head -n 1)"
        echo "dtc.reported = $dtcv"
        echo "spike.path = $spikep"
        for line in xpack_gcc.sha256 dtc.commit spike.commit riscv_tests.commit riscv_tests.env_commit arch_test.commit; do
            echo "$line = $(lock_get "$line")"
        done
        echo "firmware.cxxflags = $flags"
        echo "firmware.probe = ok ($probe/probe.elf)"
    } > "$PREFIX/installed.lock"
    {
        echo "# source me: PATH and CMake hints for the Tesseract RISC-V tools (install-toolchain.sh)"
        echo "export PATH=\"$bin:$PREFIX/spike/bin:$PREFIX/dtc/bin:\$PATH\""
        echo "export TSR_RISCV_PREFIX=\"$bin/$pfx\""
        echo "export TSR_SPIKE=\"$PREFIX/spike/bin/spike\""
        echo "export TSR_TOOLS_PREFIX=\"$PREFIX\""
    } > "$PREFIX/env.sh"
    log "check: ok ($("$gcc" --version | head -n 1); multilib $mdir; $("$gdb" --version | head -n 1))"
    log "check: wrote $PREFIX/installed.lock and $PREFIX/env.sh"
}

step_sanity() {
    local h out gp
    [ -x "$PREFIX/spike/bin/spike" ] || die "Spike is not installed (steps dtc, spike)"
    "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null || die "$PY is not Python >= 3.8 (--python)"
    h="$REPO/sw/sim/tests/rvtests"
    out="$PREFIX/sanity"
    gp="$PREFIX/$(lock_get xpack_gcc.dir)/bin/$(lock_get xpack_gcc.prefix)"
    log "sanity: riscv-tests as published on Spike (profile upstream)"
    "$PY" "$h/rvtests.py" riscv-tests --env upstream --tools-prefix "$PREFIX" --gcc-prefix "$gp" \
        --out "$out" -j "$JOBS" || die "riscv-tests (upstream env) failed on Spike"
    log "sanity: riscv-tests (profile env) and riscv-arch-test on Spike (profile test)"
    "$PY" "$h/rvtests.py" all --env tsr --tools-prefix "$PREFIX" --gcc-prefix "$gp" \
        --out "$out" -j "$JOBS" || die "the test-profile suites failed on Spike"
    log "sanity: 20 random programs on Spike (profile test)"
    "$PY" "$h/rv_random.py" run --tools-prefix "$PREFIX" --gcc-prefix "$gp" --out "$out/random" \
        --count 20 -j "$JOBS" || die "random programs failed on Spike"
}

# --------------------------------------------------------------------------------------------
if command -v flock >/dev/null 2>&1; then
    exec 9> "$PREFIX/.install.lock"
    flock -n 9 || die "another install-toolchain.sh is running on $PREFIX"
fi
trap 'rm -rf "$PREFIX"/.stage-xpack.*' EXIT
log "prefix $PREFIX, lock $LOCK, steps: $STEPS, -j$JOBS"
for s in $STEPS_ALL; do
    case " $STEPS " in
        *" $s "*) CUR_STEP=$s; "step_$s" ;;
    esac
done
CUR_STEP=done
log "done; for a shell: . $PREFIX/env.sh"

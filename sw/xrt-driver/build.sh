#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# build.sh - build the XRT 2024.2 (2.18.179) PCIe drivers xocl.ko and
# xclmgmt.ko with the kernel-compatibility patches in ./patches, the same
# way DKMS builds them from /usr/src/xrt-2.18.179.
#
# Source of the 2.18.179 DKMS tree (pick one, default --fetch):
#   --fetch            shallow-clone Xilinx/XRT tag 202420.2.18.179 (+ the
#                      libqdma submodule) and assemble the DKMS tree exactly
#                      as src/CMake/dkms.cmake installs it
#   --deb FILE         extract it from xrt_202420.2.18.179_24.04-amd64-xrt.deb
#   --src DIR          use an existing DKMS tree, e.g. /usr/src/xrt-2.18.179
#                      (it is copied, never modified)
#
# Other options:
#   -k, --kernel K     kernel to build for: a release (7.0.0-31-generic, uses
#                      /lib/modules/K/build) or a kernel build directory.
#                      Repeat for several kernels. Default: $(uname -r)
#   -w, --work DIR     work directory; only the src/, build/, out/, logs/,
#                      fetch/, deb/ entries inside it are (re)created.
#                      Default: <repo>/build/xrt-driver/work inside the
#                      Tesseract checkout (git-ignored), else ./xrt-driver-work
#   -j, --jobs N       parallel jobs per module build (default 4)
#   --no-patches       build the unpatched tree (reproduces the failure)
#   --strict           also fail on warnings outside the known, justified set
#   -h, --help
#
# Exit status is non-zero if any build fails, if a module is missing, if the
# log has a compiler error, or if modpost reports anything (undefined
# symbols, section mismatches).
#
# Nothing here needs root and nothing outside the work directory is written.

set -euo pipefail
export LC_ALL=C

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_DIR=$HERE/patches
MANIFEST_PRISTINE=$HERE/manifest/xrt-2.18.179-dkms.sha256
MANIFEST_PATCHED=$HERE/manifest/xrt-2.18.179-dkms-patched.sha256

XRT_REPO=https://github.com/Xilinx/XRT.git
XRT_TAG=202420.2.18.179
XRT_VER=2.18.179
LIBQDMA=src/runtime_src/core/pcie/driver/linux/xocl/lib/libqdma

# Values baked into driver/include/version.h of the official deb
# (CMake/version.cmake output for the release build of this tag).
VH_BRANCH=2024.2
VH_HASH=3ade2e671e5ab463400813fc2846c57edf82bb10
VH_HASH_DATE="Tue, 5 Nov 2024 13:38:03 -0800"
VH_DATE_RFC="Tue, 05 Nov 2024 13:50:38 -0800"
VH_DATE="2024-11-05 13:50:38"
VH_HEAD_COMMITS=7940
VH_BRANCH_COMMITS=4

die()  { echo "build.sh: error: $*" >&2; exit 1; }
info() { echo "== $*"; }

usage() { sed -n '3,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

mode=fetch; mode_arg=""; jobs=4; patches=1; strict=0
if [ -d "$HERE/../../build" ] && [ -f "$HERE/../../.gitignore" ]; then
	work=$(cd "$HERE/../.." && pwd)/build/xrt-driver/work
else
	work=$PWD/xrt-driver-work
fi
kernels=()
while [ $# -gt 0 ]; do
	case $1 in
	--fetch) mode=fetch ;;
	--deb) mode=deb; mode_arg=${2:?--deb needs a file}; shift ;;
	--src) mode=src; mode_arg=${2:?--src needs a directory}; shift ;;
	-k|--kernel) kernels+=("${2:?--kernel needs an argument}"); shift ;;
	-w|--work) work=${2:?--work needs a directory}; shift ;;
	-j|--jobs) jobs=${2:?--jobs needs a number}; shift ;;
	--no-patches) patches=0 ;;
	--strict) strict=1 ;;
	-h|--help) usage; exit 0 ;;
	*) usage >&2; die "unknown argument: $1" ;;
	esac
	shift
done
[ ${#kernels[@]} -gt 0 ] || kernels=("$(uname -r)")
for t in make gcc patch sha256sum; do command -v $t >/dev/null || die "$t not found"; done

mkdir -p "$work"
work=$(cd "$work" && pwd)
src=$work/src
rm -rf "$src" "$work/build" "$work/out" "$work/logs"
mkdir -p "$work/logs" "$work/out"

# ---------------------------------------------------------------------------
# 1. obtain the pristine DKMS tree in $src
# ---------------------------------------------------------------------------

# Reproduce src/CMake/dkms.cmake: install the listed files into the DKMS layout.
assemble_from_git() {
	local git=$1 out=$2
	local cm=$git/src/CMake/dkms.cmake
	local core=$git/src/runtime_src/core
	local drv=$core/pcie/driver/linux
	list() {
		awk -v v="$1" '$0 ~ "^SET \\("v"$" {f=1; next}
			f && /^ *\)/ {f=0} f {gsub(/^ +| +$/, ""); if ($0 != "") print}' "$cm"
	}
	local f n=0
	mkdir -p "$out/driver"
	for f in $(list XRT_DKMS_DRIVER_SRCS) $(list XRT_DKMS_DRIVER_INCLUDES); do
		install -D -m 644 "$drv/$f" "$out/driver/$f"; n=$((n+1))
	done
	for f in $(list XRT_DKMS_CORE_INCLUDES); do
		install -D -m 644 "$core/$f" "$out/driver/$f"; n=$((n+1))
	done
	for f in $(list XRT_DKMS_COMMON_XRT_DRV); do
		install -D -m 644 "$core/$f" "$out/driver/common/$(basename "$f")"; n=$((n+1))
	done
	for f in $(list XRT_DKMS_COMMON_XRT_DRV_INCLUDES); do
		install -D -m 644 "$core/$f" "$out/driver/include/$(basename "$f")"; n=$((n+1))
	done
	[ $n -gt 200 ] || die "dkms.cmake parsing found only $n files"
	# generated files: dkms.conf and driver/include/version.h
	sed "s/@XRT_VERSION_STRING@/$XRT_VER/" \
		"$git/src/CMake/config/dkms-xocl/dkms.conf.in" > "$out/dkms.conf"
	sed -e "s/@XRT_VERSION_STRING@/$XRT_VER/g" \
	    -e "s/@XRT_BRANCH@/$VH_BRANCH/g" \
	    -e "s/@XRT_HASH@/$VH_HASH/g" \
	    -e "s/@XRT_HASH_DATE@/$VH_HASH_DATE/g" \
	    -e "s/@XRT_DATE_RFC@/$VH_DATE_RFC/g" \
	    -e "s/@XRT_DATE@/$VH_DATE/g" \
	    -e "s/@XRT_MODIFIED_FILES@//g" \
	    -e "s/@XRT_VERSION_MAJOR@/${XRT_VER%%.*}/g" \
	    -e "s/@XRT_VERSION_MINOR@/$(echo $XRT_VER | cut -d. -f2)/g" \
	    -e "s/@XRT_VERSION_PATCH@/${XRT_VER##*.}/g" \
	    -e "s/@XRT_HEAD_COMMITS@/$VH_HEAD_COMMITS/g" \
	    -e "s/@XRT_BRANCH_COMMITS@/$VH_BRANCH_COMMITS/g" \
		"$git/src/CMake/config/version.h.in" > "$out/driver/include/version.h"
}

case $mode in
fetch)
	command -v git >/dev/null || die "git not found"
	g=$work/fetch/XRT
	if [ -d "$g/.git" ] && [ "$(git -C "$g" rev-parse HEAD)" = "$VH_HASH" ]; then
		info "reusing clone $g ($XRT_TAG)"
	else
		rm -rf "$work/fetch"; mkdir -p "$work/fetch"
		info "cloning $XRT_REPO tag $XRT_TAG"
		git -c advice.detachedHead=false clone -q --depth 1 --branch "$XRT_TAG" "$XRT_REPO" "$g"
	fi
	[ "$(git -C "$g" rev-parse HEAD)" = "$VH_HASH" ] || die "tag $XRT_TAG is not $VH_HASH"
	git -C "$g" submodule update -q --init --depth 1 "$LIBQDMA"
	info "assembling DKMS tree from src/CMake/dkms.cmake"
	assemble_from_git "$g" "$src"
	;;
deb)
	command -v dpkg-deb >/dev/null || die "dpkg-deb not found"
	[ -f "$mode_arg" ] || die "no such file: $mode_arg"
	rm -rf "$work/deb"; mkdir -p "$work/deb"
	info "extracting $(basename "$mode_arg")"
	dpkg-deb -x "$mode_arg" "$work/deb"
	[ -d "$work/deb/usr/src/xrt-$XRT_VER" ] || die "deb has no usr/src/xrt-$XRT_VER"
	cp -a "$work/deb/usr/src/xrt-$XRT_VER" "$src"
	;;
src)
	[ -f "$mode_arg/dkms.conf" ] && [ -d "$mode_arg/driver/xocl" ] ||
		die "$mode_arg does not look like an XRT DKMS tree (dkms.conf, driver/xocl)"
	info "copying $mode_arg"
	mkdir -p "$src"
	cp -a "$mode_arg/." "$src/"
	;;
esac

grep -q "^PACKAGE_VERSION=\"$XRT_VER\"" "$src/dkms.conf" ||
	die "source tree is not XRT $XRT_VER (dkms.conf: $(grep PACKAGE_VERSION "$src/dkms.conf"))"

# Identify the tree against the manifests of the official deb's DKMS tree.
tree_state() {
	local m
	for m in "$MANIFEST_PRISTINE" "$MANIFEST_PATCHED"; do
		[ -f "$m" ] || continue
		if (cd "$src" && sha256sum --quiet --strict -c "$m" >/dev/null 2>&1) &&
		   [ "$(cd "$src" && find . -type f | wc -l)" -eq "$(wc -l < "$m")" ]; then
			[ "$m" = "$MANIFEST_PRISTINE" ] && echo pristine || echo patched
			return
		fi
	done
	echo unknown
}
state=$(tree_state)
info "source tree: $state"

# ---------------------------------------------------------------------------
# 2. patches
# ---------------------------------------------------------------------------
if [ $patches = 1 ]; then
	if [ "$state" = patched ]; then
		info "patch series already applied in the source tree, skipping"
	else
		[ "$state" = pristine ] ||
			echo "build.sh: warning: source differs from the official 2.18.179 tree; trying the patches anyway" >&2
		for p in "$PATCH_DIR"/[0-9][0-9][0-9][0-9]-*.patch; do
			if ! patch -d "$src" -p1 --forward --batch --no-backup-if-mismatch -s < "$p" \
				> "$work/logs/patch.log" 2>&1; then
				cat "$work/logs/patch.log" >&2
				die "patch $(basename "$p") does not apply"
			fi
			echo "   applied $(basename "$p")"
		done
		state=$(tree_state)
		[ "$state" = patched ] || [ ! -f "$MANIFEST_PATCHED" ] ||
			echo "build.sh: warning: patched tree does not match $MANIFEST_PATCHED" >&2
	fi
fi

# ---------------------------------------------------------------------------
# 3. build for every kernel, exactly like DKMS' MAKE[0]:
#    cd driver/xocl; make KERNEL_SRC=<kdir>   (= userpf "make all", then mgmtpf)
#    The two PFs are built one after the other because they compile the
#    shared ../*.c into the same object files with different -DPF flags.
# ---------------------------------------------------------------------------

# Warnings that are known and justified (see README.md, "Warnings").
KNOWN_NOPROTO="SYSMON_TO_MILLDEGREE accel_deadlock_detector_ioctl aim_ioctl am_ioctl
arch_msi_check_device asm_ioctl cu_isr ert_user_thread find_user_node
get_opcode_name intc_isr kind_to_string lapc_ioctl mailbox_isr mailbox_listen
mailbox_post_response mailbox_set mailbox_versal_isr read_completion
sglist_index spc_ioctl store_pcie_link_info timeout_msg trace_fifo_full_ioctl
trace_fifo_lite_ioctl trace_funnel_ioctl trace_s2mm_ioctl ucu_isr
vmr_xgq_basic_op xocl_check_slot_topology xocl_describe xocl_fini_nifd
xocl_init_nifd xocl_program_shell xocl_set_cma_bank xocl_userpf_probe
xocl_userpf_remove xrt_cu_intr_thread"
KNOWN_NOPROTO=$(echo $KNOWN_NOPROTO)
known_warning() {
	# gcc quotes identifiers with ' or with U+2018/U+2019 depending on the
	# locale (kbuild unexports LC_ALL), so the patterns avoid the quotes.
	case $1 in
	*"no previous prototype for "*"[-Wmissing-prototypes]"*)
		# the 38 functions that already warn on the 6.8 GA kernel
		local f=${1#*no previous prototype for }
		f=${f#"${f%%[A-Za-z0-9_]*}"}; f=${f%%[!A-Za-z0-9_]*}
		case " $KNOWN_NOPROTO " in *" $f "*) return 0 ;; esac
		return 1 ;;
	*"strrchr"*"reading 1 or more bytes from a region of size 0 [-Wstringop-overread]"*) return 0 ;;
	*"__builtin_strlen"*"reading 1 or more bytes from a region of size 0 [-Wstringop-overread]"*) return 0 ;;
	*'"FIELD_GET" redefined'*) return 0 ;;
	*"the compiler differs from the one used to build the kernel"*) return 0 ;;
	*"pahole version differs from the one used to build the kernel"*) return 0 ;;
	esac
	return 1
}

fail=0
summary=()
for k in "${kernels[@]}"; do
	if [ -d "$k" ]; then kdir=$(cd "$k" && pwd)
	else kdir=/lib/modules/$k/build; fi
	[ -f "$kdir/Makefile" ] && [ -f "$kdir/include/config/kernel.release" ] ||
		die "no kernel build directory for '$k' ($kdir)"
	krel=$(cat "$kdir/include/config/kernel.release")
	b=$work/build/$krel; log=$work/logs/$krel.log
	rm -rf "$b"; mkdir -p "$b"
	cp -a "$src/." "$b/"
	info "building for $krel ($kdir), log: $log"
	rc=0
	for pf in userpf mgmtpf; do
		(cd "$b/driver/xocl/$pf" && make -j"$jobs" KERNEL_SRC="$kdir" all) >> "$log" 2>&1 || rc=1
	done
	errs=$(grep -c -E '(error|Error):' "$log" || true)
	modpost=$(grep -E '^(ERROR|WARNING): modpost:|undefined!' "$log" || true)
	ok=1
	[ $rc = 0 ] || ok=0
	[ "$errs" = 0 ] || ok=0
	[ -z "$modpost" ] || { ok=0; echo "$modpost" >&2; }
	mkdir -p "$work/out/$krel"
	for m in userpf/xocl.ko mgmtpf/xclmgmt.ko; do
		if [ -f "$b/driver/xocl/$m" ]; then cp "$b/driver/xocl/$m" "$work/out/$krel/"
		else ok=0; echo "build.sh: $krel: $(basename $m) was not built" >&2; fi
	done
	nwarn=$(grep -c 'warning:' "$log" || true)
	unknown=$(grep 'warning:' "$log" | while IFS= read -r l; do known_warning "$l" || echo "$l"; done)
	nunk=0; [ -z "$unknown" ] || nunk=$(echo "$unknown" | wc -l)
	if [ -n "$unknown" ]; then
		echo "build.sh: $krel: warnings outside the known set:" >&2
		echo "$unknown" | sed 's/^/   /' >&2
		[ $strict = 0 ] || ok=0
	fi
	if [ $ok = 1 ]; then
		vm=$(modinfo -F vermagic "$work/out/$krel/xocl.ko" 2>/dev/null | awk '{print $1}')
		summary+=("$(printf '%-22s OK    errors=0 modpost=clean warnings=%s (new: %s) vermagic=%s' "$krel" "$nwarn" "$nunk" "$vm")")
	else
		fail=1
		summary+=("$(printf '%-22s FAIL  errors=%s rc=%s warnings=%s (new: %s)' "$krel" "$errs" "$rc" "$nwarn" "$nunk")")
	fi
done

echo
info "summary (patches: $([ $patches = 1 ] && echo applied || echo none), source: $state)"
printf '   %s\n' "${summary[@]}"
[ $fail = 0 ] && echo "   modules: $work/out/<kernel>/{xocl,xclmgmt}.ko"
exit $fail

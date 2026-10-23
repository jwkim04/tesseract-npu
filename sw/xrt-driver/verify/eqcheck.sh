#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# eqcheck.sh -k KDIR -a TREE_A -b TREE_B [-w WORK] [-j N]
#
# Build two XRT DKMS trees (e.g. the pristine 2.18.179 tree and the patched
# one) against the same kernel and compare every object file, to show that
# the patch series does not change the generated code on that kernel.
#
# Both trees are built in the same directory (WORK/tree) so __FILE__ strings
# match. Source line numbers are the only other input that legitimately
# differs, so they are neutralized:
#   - __LINE__ is mapped to __COUNTER__ (the XRT headers build unique enum
#     names from __LINE__, so it cannot be a constant), and
#   - UBSAN is turned off (UBSAN_SANITIZE=n), because its check records
#     carry the real source line and column.
# Debug info, .comment and the build-id note are stripped before comparing.
#
# Example (Ubuntu 24.04 GA kernel headers, pristine vs patched):
#   ./build.sh --fetch --no-patches -w /tmp/xrt-a -k <kdir>
#   ./build.sh --fetch              -w /tmp/xrt-b -k <kdir>
#   verify/eqcheck.sh -k <kdir> -a /tmp/xrt-a/src -b /tmp/xrt-b/src
set -euo pipefail

kdir=""; a=""; b=""; work=$PWD/eqcheck-work; jobs=4
while [ $# -gt 0 ]; do
	case $1 in
	-k) kdir=$2; shift ;;
	-a) a=$2; shift ;;
	-b) b=$2; shift ;;
	-w) work=$2; shift ;;
	-j) jobs=$2; shift ;;
	*) sed -n '4,22p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
	esac
	shift
done
[ -f "$kdir/Makefile" ] && [ -d "$a/driver" ] && [ -d "$b/driver" ] ||
	{ echo "usage: $0 -k KDIR -a TREE_A -b TREE_B [-w WORK]" >&2; exit 2; }

mkdir -p "$work"; work=$(cd "$work" && pwd)
rm -rf "$work/tree" "$work/obj-a" "$work/obj-b"
KC="-U__LINE__ -D__LINE__=__COUNTER__ -Wno-builtin-macro-redefined"

for side in a b; do
	t=$a; [ $side = b ] && t=$b
	rm -rf "$work/tree"; mkdir -p "$work/tree"
	cp -a "$t/." "$work/tree/"
	find "$work/tree" \( -name '*.o' -o -name '*.ko' -o -name '.*.cmd' \) -delete
	for pf in userpf mgmtpf; do
		(cd "$work/tree/driver/xocl/$pf" &&
		 make -j"$jobs" KERNEL_SRC="$kdir" KCFLAGS="$KC" UBSAN_SANITIZE=n all) \
			> "$work/build-$side-$pf.log" 2>&1 ||
			{ echo "build of tree $side ($pf) failed, see $work/build-$side-$pf.log" >&2; exit 1; }
		# the PFs rebuild the shared ../*.c with different flags: snapshot per PF
		(cd "$work/tree/driver" && find . -name '*.o' ! -name '*.mod.o' | while read -r o; do
			mkdir -p "$work/obj-$side/$pf/$(dirname "$o")"
			objcopy --strip-debug -R .comment -R .note.gnu.build-id "$o" "$work/obj-$side/$pf/$o"
		done)
	done
done

n=0; d=0
while read -r f; do
	n=$((n+1))
	if ! cmp -s "$work/obj-a/$f" "$work/obj-b/$f"; then d=$((d+1)); echo "DIFF $f"; fi
done < <(cd "$work/obj-a" && find . -name '*.o' | sort)
[ "$(cd "$work/obj-b" && find . -name '*.o' | wc -l)" = "$n" ] || { echo "object sets differ"; d=$((d+1)); }
echo "compared $n objects against $(cat "$kdir/include/config/kernel.release"): $d differ"
[ $d = 0 ]

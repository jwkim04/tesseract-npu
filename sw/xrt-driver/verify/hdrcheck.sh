#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# hdrcheck.sh KDIR...
#
# For each kernel build directory, read the kernel's own headers and print
# on which side of each patch's API change the kernel is (1 = new API: the
# old interface is gone or the new one is present), next to the guard the
# patch uses. A row where a LINUX_VERSION_CODE guard and the headers
# disagree means a distribution backport and needs a look; feature-test
# guards ("ifdef") follow the headers by construction.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: $0 KDIR..." >&2; exit 2; }

has() { grep -qE -- "$1" "$2" 2>/dev/null && echo 1 || echo 0; }
not() { [ "$1" = 1 ] && echo 0 || echo 1; }

#      patch  guard                    probe
checks=(
 "0001|include (no guard)|vmalloc_dropped"
 "0002|ifdef I2C_CLASS_SPD|spd_gone"
 "0003|>= 6.10|kfifo"
 "0004|>= 6.11|remove_void"
 "0005|>= 6.12|no_llseek_gone"
 "0006|ifdef FOP_UNSIGNED_OFFSET|fop"
 "0007|>= 6.13|iommu_present_gone"
 "0008|>= 6.13|ns_string"
 "0009|>= 6.14|drm_date_gone"
 "0010|ifndef crc32c_le|crc32c_le_gone"
 "0011|>= 6.15|del_timer_gone"
 "0012|>= 6.16|timer_container_of"
 "0013|>= 6.16|bin_attr_const"
 "0014|>= 6.17|pfn_t_gone"
 "0015|ifndef ida_simple_get|ida_simple_gone"
 "0016|include (no guard)|drm_print_not_via_drm_mm"
 "0019|>= 6.15|p2pdma_zero_refcount"
)

probe() {
	local i=$1/include
	case $2 in
	vmalloc_dropped) not "$(has 'linux/vmalloc.h' "$i/asm-generic/io.h")" ;;
	spd_gone) not "$(has 'define I2C_CLASS_SPD' "$i/linux/i2c.h")" ;;
	kfifo) not "$(has 'struct circ_buf[[:space:]]+xmit;' "$i/linux/serial_core.h")" ;;
	remove_void) has 'void \(\*remove\)\(struct platform_device \*\)' "$i/linux/platform_device.h" ;;
	no_llseek_gone) not "$(has 'define no_llseek' "$i/linux/fs.h")" ;;
	fop) has 'define FOP_UNSIGNED_OFFSET' "$i/linux/fs.h" ;;
	iommu_present_gone) not "$(has 'iommu_present' "$i/linux/iommu.h")" ;;
	ns_string) has 'MODULE_INFO\(import_ns, ns\)' "$i/linux/module.h" ;;
	drm_date_gone) not "$(has 'char \*date;' "$i/drm/drm_drv.h")" ;;
	crc32c_le_gone) not "$(has 'define crc32c_le' "$i/linux/crc32c.h")" ;;
	del_timer_gone) not "$(has 'int del_timer_sync' "$i/linux/timer.h")" ;;
	timer_container_of) has 'define timer_container_of' "$i/linux/timer.h" ;;
	bin_attr_const) has 'ssize_t \(\*read\)\(struct file \*, struct kobject \*, const struct bin_attribute' "$i/linux/sysfs.h" ;;
	pfn_t_gone) [ -e "$i/linux/pfn_t.h" ] && echo 0 || echo 1 ;;
	ida_simple_gone) not "$(has 'define ida_simple_get' "$i/linux/idr.h")" ;;
	drm_print_not_via_drm_mm) not "$(has 'drm/drm_print.h' "$i/drm/drm_mm.h")" ;;
	# The 0019 change itself is in mm/mm_init.c, not in the headers; this
	# probes vmf_insert_page_mkwrite() (ec2e0cc67f9c), from the same v6.15
	# patch series as b7e282378773.
	p2pdma_zero_refcount) has 'vmf_insert_page_mkwrite' "$i/linux/mm.h" ;;
	esac
}

rc=0
for k in "$@"; do
	v=$(sed -n 's/#define LINUX_VERSION_CODE //p' "$k/include/generated/uapi/linux/version.h")
	echo "== $(cat "$k/include/config/kernel.release")  LINUX_VERSION_CODE $((v>>16)).$(((v>>8)&255)).$((v&255))"
	for c in "${checks[@]}"; do
		IFS='|' read -r p g pr <<<"$c"
		side=$(probe "$k" "$pr")
		note=""
		case $g in
		">= "*)
			want=${g#>= }; maj=${want%.*}; min=${want#*.}
			gv=$([ "$v" -ge $(( (maj<<16) + (min<<8) )) ] && echo 1 || echo 0)
			[ "$gv" = "$side" ] || { note="  <-- guard selects $gv, headers say $side"; rc=1; } ;;
		esac
		printf '   %s  %-28s headers=%s%s\n' "$p" "$g" "$side" "$note"
	done
done
exit $rc

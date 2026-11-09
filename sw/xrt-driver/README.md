# XRT 2024.2 PCIe drivers on Ubuntu 24.04 HWE kernels (6.17, 7.0)

XRT 2024.2 (`xrt_202420.2.18.179_24.04-amd64-xrt.deb`, version 2.18.179) builds its Alveo PCIe drivers with DKMS:
`xocl.ko` (user PF) and `xclmgmt.ko` (management PF), from `/usr/src/xrt-2.18.179`. That source targets the Ubuntu
24.04 GA kernel (6.8) and does not build on the HWE kernels 6.17 and 7.0. This directory turns it into a tree that
builds cleanly and behaves the same on 6.8, 6.17 and 7.0. It belongs to the platform, not to the Tesseract core.

- `patches/0001..0019-*.patch`: a `git format-patch` series against the DKMS tree, one API change per patch. Patch
  paths are relative to the DKMS tree root, so they apply with `patch -p1` inside `/usr/src/xrt-2.18.179`.
- `build.sh`: gets the DKMS tree (`--fetch` from the git tag `202420.2.18.179`, `--deb`, or `--src` an existing tree),
  applies the series, builds both modules per kernel the way DKMS does, and fails on errors and modpost findings
  (`--strict`: also on any warning outside the known set). `--no-patches` builds the unpatched tree.
- `verify/eqcheck.sh` builds two trees against one kernel and compares every object file (on 6.8.0-147 the patched
  and the pristine tree are identical), and `verify/hdrcheck.sh` shows, per kernel header tree, which side of every
  patched API the headers are on. `manifest/*.sha256` has the sha256 of all 223 files of the official tree.

The deb downloads without a login from `https://download.amd.com/opendownload/xrt/xrt_202420.2.18.179_24.04-amd64-xrt.deb`
(sha256 `5e76278a105bb994280f954d4d48e910b945b662ee762acba7a5ac7a784fa688`). The patched tree builds with 0 errors and
clean modpost on 7.0.0-28, -31 and -38, 6.17.0-35, 6.11.0-29 and 6.14.0-37. The remaining warnings are
`-Wmissing-prototypes` (identical on the unpatched 6.8 build), a gcc 13 `-Wstringop-overread` false positive in libfdt,
`FIELD_GET` redefined by libqdma (the same result as the kernel's), and the compiler and `pahole` version notes.

## The patches

Every guard matches the mainline headers on both sides of the boundary and the Ubuntu 6.8, 6.11, 6.14, 6.17 and 7.0
header trees (`verify/hdrcheck.sh`). Each patch's commit message has the full reasoning.

| # | Change | Guard |
| --- | --- | --- |
| 0001 | `#include <linux/vmalloc.h>` in xocl_drv.h | none |
| 0002 | xiic adapter class without `I2C_CLASS_SPD` | `#ifdef I2C_CLASS_SPD` |
| 0003 | ulite TX from `tty_port::xmit_fifo` | kernel < 6.10 keeps the old code |
| 0004 | void `platform_driver::remove` for 57 subdevs, via `XOCL_PLATFORM_REMOVE()` | kernel >= 6.11 |
| 0005 | omit `.llseek = no_llseek` | kernel < 6.12 keeps it |
| 0006 | `.fop_flags = FOP_UNSIGNED_OFFSET` on the DRM fops (without it every `open()` of the render node returns EINVAL) | `#ifdef FOP_UNSIGNED_OFFSET` |
| 0007 | p2p: `device_iommu_mapped()` instead of `iommu_present()` | kernel < 6.13 keeps the old call |
| 0008 | `MODULE_IMPORT_NS("DMA_BUF")` in xocl_drv.c and libxdma.c | kernel >= 6.13 |
| 0009 | omit `drm_driver::date` | kernel < 6.14 keeps it |
| 0010 | restore the `crc32c_le` alias of `crc32c()` | `#ifndef crc32c_le` |
| 0011 | `timer_delete[_sync]()` (xrt_cu, mailbox, ert_user, libqdma) | kernel < 6.15 keeps the old calls |
| 0012 | `timer_container_of()` (same files) | kernel >= 6.16 |
| 0013 | `BIN_ATTRIBUTE_CONST` on 42 bin_attribute callbacks and 15 `*_bin_attrs[]` | kernel >= 6.16 |
| 0014 | CMA and userptr fault: `vmf_insert_mixed(..., page_to_pfn())` | kernel < 6.17 keeps the pfn_t path |
| 0015 | provide `ida_simple_get/remove` as the kernel defined them | `#ifndef ida_simple_get` |
| 0016 | `#include <drm/drm_print.h>` for the DRM log macros | none |
| 0017 | `static inline T` and `static const T` order | none |
| 0018 | no empty if and else bodies (`dbg_init()` as `do { } while (0)`) | none |
| 0019 | p2p: set every page of a remapped P2P BAR chunk to refcount 1 after `devm_memremap_pages()` (without it a CPU access to a mapped P2P BO is a SIGBUS) | kernel >= 6.15 |

Except for 0006 and 0019, which would still compile without the patch but fail at run time, the patches leave the
behaviour of 6.8 unchanged. 0019 has no header macro to guard on, because the kernel change is inside `mm/mm_init.c`,
so it uses a version guard checked against the mainline sources (v6.14 against v6.15). Upstream XRT's own changes that
alter behaviour (the CRC, ida ranges, ulite NUL bytes, the remove warning, the DRM log macros) are not used as they
are.

## Applying it (needs sudo)

1. `sudo apt install ./xrt_202420.2.18.179_24.04-amd64-xrt.deb`. Its DKMS step is expected to fail on 6.17 and 7.0
   kernels, and the user-space part installs anyway. `./build.sh --src /usr/src/xrt-2.18.179 -w /tmp/xrt-check -k
   "$(uname -r)"` prints `source tree: pristine`.
2. Patch the DKMS source in place, a dry run first (the `--src` check should then print `source tree: patched`):

   ```
   cd /usr/src/xrt-2.18.179
   for p in <repository>/sw/xrt-driver/patches/*.patch; do
       patch -p1 --dry-run --forward --batch < "$p" > /dev/null || { echo "FAILS: $p"; break; }
       sudo patch -p1 --forward --batch --no-backup-if-mismatch < "$p" || break
   done
   ```

3. Rebuild and install for every kernel that has headers (future kernel updates rebuild automatically, because
   `AUTOINSTALL="yes"` and `/usr/src` holds the patched tree):

   ```
   sudo dkms remove -m xrt -v 2.18.179 --all; sudo dkms add -m xrt -v 2.18.179
   for k in $(ls /lib/modules); do [ -e /lib/modules/$k/build/Makefile ] || continue
       sudo dkms build -m xrt -v 2.18.179 -k "$k" && sudo dkms install -m xrt -v 2.18.179 -k "$k"
   done
   ```

4. The deb's postinst installs the udev rules and loads the modules only when its own DKMS step succeeded, so copy
   `driver/xocl/userpf/99-xocl.rules` and `driver/xocl/mgmtpf/99-xclmgmt.rules` from `/usr/src/xrt-2.18.179` to
   `/etc/udev/rules.d/`, run `sudo udevadm control --reload-rules`, `sudo modprobe xclmgmt && sudo modprobe xocl` and
   `sudo udevadm trigger`.

Do not `apt install --reinstall` the deb afterwards, because it restores the unpatched `/usr/src` tree. After a
reinstall or upgrade of the same version, repeat steps 2 to 4.

## Checks on the card

The U200 needs its deployment platform packages (gen3x16_xdma_2 base, plus validate, CMC and SC firmware) from the AMD
U200 download page. `lspci -d 10ee: -nn` shows two functions (mgmt PF and user PF), `modinfo -F vermagic xocl` equals
`uname -r`, `sudo dmesg | grep -E 'drm_open_helper|remove callback returned|failed to get the same CRC'` shows nothing,
`sudo xbmgmt examine` shows the card, shell and SC and CMC versions (`sudo xbmgmt program --base --device <mgmt BDF>`
only if it reports a mismatched or golden shell, then a full power cycle), `xbutil examine -d <user BDF> -r all` lists
the user PF as ready (this exercises 0006 and 0013), and `xbutil validate -d <user BDF>` passes. 0003 and 0014 run only
with a UART-lite subdevice and with CMA or userptr BOs, so a clean validate does not cover them. P2P is off after every
cold boot, so the plain `validate` skips the p2p test, and P2P is the only path that exercises 0019 and 0007. It needs
"Above 4G decoding" in the BIOS and a 64 GB I/O window (XRT's `p2p.rst`): `sudo xbutil configure --device <user BDF>
--p2p enable` (it may ask for a warm reboot), then `xbutil validate -d <user BDF> -r p2p`. Tesseract uses the card
without P2P, so 0019 is covered by build, header and object-level evidence only.

Limits. On a machine where an IOMMU is registered but does not translate the Alveo card, 0007 hands the P2P BAR to the
management PF on 6.13 and later, which is the per-device answer the check is meant to give. On 6.13 and later
`DRM_IOCTL_VERSION` reports the date as "0" (0009), which the kernel does itself. RHEL-only code paths are left
untouched. A kernel built with gcc 14 or newer also needs upstream XRT's `stat_read()` `typeof` change, and a future
Ubuntu kernel may need more patches: `./build.sh -k <new kernel>` shows it, and `verify/hdrcheck.sh <kdir>` flags version
guards that its headers contradict (0019 cannot be checked that way: check that the kernel's
`__init_zone_device_page()` still zeroes `MEMORY_DEVICE_PCI_P2PDMA` pages, or run the P2P checks).

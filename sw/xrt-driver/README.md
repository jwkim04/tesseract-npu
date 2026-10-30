# XRT 2024.2 PCIe drivers on Ubuntu 24.04 HWE kernels (6.17, 7.0)

XRT 2024.2 (`xrt_202420.2.18.179_24.04-amd64-xrt.deb`, version 2.18.179) builds its Alveo
PCIe drivers with DKMS: `xocl.ko` (user PF) and `xclmgmt.ko` (management PF), from
`/usr/src/xrt-2.18.179`. That source targets the Ubuntu 24.04 GA kernel (6.8). It does not
build on the HWE kernels of the development host (7.0.0-38, which the card runs on, 7.0.0-31 and
6.17.0-35 as fallbacks): 413 compiler errors on 7.0.

This directory turns that tree into one that builds cleanly and behaves the same on 6.8,
6.17 and 7.0:

| Path | What |
| --- | --- |
| `patches/0001..0019-*.patch` | `git format-patch` series against the DKMS tree, one API change per patch |
| `build.sh` | gets the 2.18.179 DKMS tree (git tag, the deb, or an existing tree), applies the series, builds both modules per kernel the way DKMS does, fails on errors and modpost findings |
| `verify/eqcheck.sh` | builds two trees against one kernel and compares every object file (used for the 6.8 no-change proof and to check the review fixes) |
| `verify/hdrcheck.sh` | shows, per kernel header tree, which side of every patched API the headers are on |
| `manifest/*.sha256` | sha256 of all 223 files of the official DKMS tree, pristine and patched |

> **Scope.** The driver port that the U200 card runs on: the 19-patch series, how it was verified at
> build, header and object level, and what the card has exercised (section "On the card"). It belongs
> to the platform, not to the Tesseract core: the v1 and v2 cards use it alike.

Status: all patched builds pass (0 errors, modpost clean, no warnings outside the known set
below). The series is loaded on the development host and the U200 card runs on it (kernel
7.0.0-38, see "On the card").

| Kernel headers | Result | Warnings (all known, see "Warnings") |
| --- | --- | --- |
| 7.0.0-31-generic (7.0.14) | OK | 51 |
| 7.0.0-38-generic (7.0.14), the kernel of the card runs | OK | 51 |
| 6.17.0-35-generic (6.17.13) | OK | 63 |
| 7.0.0-28-generic (7.0.12), also installed | OK | 51 |
| 6.8.0-147-generic (6.8.12), GA kernel, extracted deb | OK, object code identical to the unpatched tree | 49 (same as unpatched) |
| 6.11.0-29-generic, 6.14.0-37-generic, extracted debs | OK | 63 |

The 6.8, 6.11 and 6.14 header packages were fetched with `apt-get download` and unpacked
into `build/xrt-driver/kheaders/` (nothing installed). They give real Ubuntu builds on
both sides of every guard.

## Source tree

- The deb downloads without a login from
  `https://download.amd.com/opendownload/xrt/xrt_202420.2.18.179_24.04-amd64-xrt.deb`
  (18,770,694 bytes, sha256 `5e76278a105bb994280f954d4d48e910b945b662ee762acba7a5ac7a784fa688`).
  The `xilinx.com/bin/public/openDownload?filename=...` link redirects to a 404.
- `usr/src/xrt-2.18.179` in the deb has 223 files: `dkms.conf`, `driver/xocl/` (userpf,
  mgmtpf, subdev, lib incl. libqdma and libfdt), `driver/common/` and `driver/include/`.
  The tree is byte-identical to tag `202420.2.18.179` (commit `3ade2e67`, libqdma
  submodule `d66f224`) assembled by the install rules in `src/CMake/dkms.cmake`. Only the two
  generated files, `dkms.conf` and `driver/include/version.h`, are not in git. `build.sh
  --fetch` regenerates both from their `.in` templates, using the values recorded in the
  deb. The result matches the pristine manifest exactly.
- `dkms.conf`: `MAKE[0]="cd driver/xocl; make KERNEL_SRC=${kernel_source_dir}; cd ../.."`,
  modules `xocl` (`driver/xocl/userpf`) and `xclmgmt` (`driver/xocl/mgmtpf`). The DKMS
  module name is `xrt`: the deb's postinst runs `common.postinst xrt 2.18.179`, although
  `PACKAGE_NAME` says `xrt-xocl`. `driver/xocl/Makefile` runs `make all` in userpf and
  then in mgmtpf. The two must run one after the other, because both compile the shared
  `../*.c` into the same object files with different `-DPF=` flags.
- Patch paths are relative to the DKMS tree root (`a/driver/xocl/...`), so they apply
  with `patch -p1` inside `/usr/src/xrt-2.18.179` and to the tree `build.sh` assembles.

## The patches

Every guard was checked against mainline headers on both sides of the boundary and against
the Ubuntu 6.8.0-147, 6.11.0-29, 6.14.0-37, 6.17.0-35 and 7.0.0-28/31/38 header trees
(`verify/hdrcheck.sh`): every version guard matches the headers of every tree, so these
kernels carry no conflicting backports. Each patch's commit message has the full
reasoning.

| # | Change | Mainline commit (first release) | Guard | Behaviour on new kernels | Upstream XRT |
| --- | --- | --- | --- | --- | --- |
| 0001 | `#include <linux/vmalloc.h>` in xocl_drv.h | 690da22dbfa8 asm-generic/io.h stops including it (6.10) | none (see below) | header only | 04350f685, same |
| 0002 | xiic adapter class without `I2C_CLASS_SPD` | e61bcf42d290 (6.10) | `#ifdef I2C_CLASS_SPD` | no driver can match class bit 7 any more; same probing | 04350f685 switches at 6.8, dropping the bit on 6.8/6.9 where it exists |
| 0003 | ulite TX from `tty_port::xmit_fifo` | 1788cf6a91d9 (6.10) | `< 6.10` keeps old code | same bytes, same register writes: kbuild compiles with `-funsigned-char` (since 6.2), so the old `char` was zero-extended exactly like the `u8` now; same `icount.tx` and wake-up threshold | based on 04350f685, which drops NUL bytes (`if (ch <= 0)`) |
| 0004 | void `platform_driver::remove` for 57 subdevs, via `XOCL_PLATFORM_REMOVE()` | 0edb555a65d1 (6.11) | `>= 6.11` | wrapper prints the core's old "remove callback returned a non-zero value" warning; probe error paths still call the int function | 04350f685 renames to `__fn` and drops the warning |
| 0005 | omit `.llseek = no_llseek` | cb787f4ac0c2 (6.12) | `< 6.12` keeps it | exact: `no_llseek` is `NULL` since 868941b14441 (6.0) | 04350f685, same |
| 0006 | `.fop_flags = FOP_UNSIGNED_OFFSET` on the DRM fops (run-time fix) | 641bb4394f40 (6.12) | `#ifdef FOP_UNSIGNED_OFFSET` | without it every `open()` of the render node returns EINVAL plus a WARN; with it, same as the 6.8 DRM core setting `FMODE_UNSIGNED_OFFSET` | 6517a2a87 uses a 6.12 version check |
| 0007 | p2p: `device_iommu_mapped()` instead of `iommu_present()` | 6632863226d8 (6.13) | `< 6.13` keeps old call | same answer whenever the IOMMU covers the card (always, with VT-d/AMD-Vi on, including `iommu=pt`) | f41db917c, same |
| 0008 | `MODULE_IMPORT_NS("DMA_BUF")` in xocl_drv.c and libxdma.c | cdd30ebb1b9f (6.13) | `>= 6.13` | exact: same `import_ns=DMA_BUF` modinfo record | f41db917c + 30c9f58ca |
| 0009 | omit `drm_driver::date` | cb2e1c2136f7 (6.14) | `< 6.14` keeps it | exact: since 6.13 `DRM_IOCTL_VERSION` returns "0" and ignores the field | f41db917c, same |
| 0010 | restore the `crc32c_le` alias of `crc32c()` | 8df36829045a (6.15) | `#ifndef crc32c_le` | exact: same CRC-32C (Castagnoli, no inversion) as 6.8's libcrc32c | f41db917c switches to `crc32_le()`, which is IEEE CRC-32: a different checksum |
| 0011 | `timer_delete[_sync]()` (xrt_cu, mailbox, ert_user, libqdma) | 8fa7292fee5c (6.15) | `< 6.15` keeps old calls | exact: the old names were inline wrappers | f41db917c, libqdma 0a578f1 |
| 0012 | `timer_container_of()` (same files) | 41cb08555c41 (6.16) | `>= 6.16` | exact: same `container_of()` | f41db917c; libqdma 0a578f1 tests `> 6.16.0`, which breaks 6.16.0 (fixed here) |
| 0013 | `BIN_ATTRIBUTE_CONST` on 42 bin_attribute callbacks and 15 `*_bin_attrs[]` | 97d06802d10a, 9bec944506fa (6.16) | `>= 6.16` | qualifier only | f41db917c (also consts objects; not needed) |
| 0014 | CMA/userptr fault: `vmf_insert_mixed(..., page_to_pfn())` | 984921edea68, 21aa65bf82a7 (6.17) | `< 6.17` keeps pfn_t path | same pfn, same special PTE: PFN_MAP without PFN_DEV was never devmap. The P2P/import branch of the same fault handler is not touched here; see 0019 | f41db917c, same shape |
| 0015 | provide `ida_simple_get/remove` as the kernel defined them | c8a09fc9664f (6.18) | `#ifndef ida_simple_get` | exact: `ida_alloc_range(ida, start, end - 1)`, end 0 = no limit | a2657e374 passes `end` as an inclusive max: minors limited to 0, so a second card fails to probe |
| 0016 | `#include <drm/drm_print.h>` for DRM_ERROR/WARN/INFO/DEBUG | ea722522d505 drm_mm.h stops including it (6.19) | none (see below) | exact: the macro definitions in 6.8 and 7.0 are byte-identical | dd3d69add redefines them as `pr_*()`, changing log prefixes and how DRM_DEBUG is switched on |
| 0017 | `static inline T` / `static const T` order | f5982cceb347 turns on `-Wextra` (6.10) | none | specifier order only (339 warnings per build) | d1fbc5cea fixed 2 of the 6 files |
| 0018 | no empty if/else bodies (`dbg_init()` as `do { } while (0)`) | f5982cceb347 (6.10) | none | empty statement either way | not addressed upstream |
| 0019 | p2p: set every page of a remapped P2P BAR chunk to refcount 1 after `devm_memremap_pages()` (run-time fix) | b7e282378773 (6.15): `MEMORY_DEVICE_PCI_P2PDMA` pages now start at refcount 0 and the driver must set it | `>= 6.15` | without it `vm_insert_page()` fails with EINVAL and every CPU access to a mapped P2P BO is a SIGBUS. With it, the counts are as on 6.8: idle 1, +1 per mapping, never 0. The release-time busy test (`> 1`) keeps its meaning | not addressed upstream (master p2p.c unchanged) |

0019 comes last only because it was added after review. It does not depend on 0001-0018
and applies on its own as well.

Unguarded changes: 0001 and 0016 only add an `#include` of a header that exists on every
supported kernel and was already included indirectly where it still is. 0017 and 0018 are
plain C that every compiler accepts. A version guard on these would only add a way to break
on a distribution kernel that backports the include cleanup.

0006 and 0019 are the two changes whose omission would still compile but fail at run time.
0006 is guarded by the macro that comes with the requirement, not by a version number.
0019 has no such macro, because the kernel change is inside `mm/mm_init.c` and not in any
header. So it uses a version guard, checked against the mainline sources (v6.14 vs
v6.15) and against the Ubuntu kernel source diffs from the archive. `linux-hwe-6.14` 6.14.0-37 does
not touch `mm/mm_init.c`. `linux-hwe-7.0` 7.0.0-38 and `linux-hwe-6.17` 6.17.0-42 touch
it, but not the refcount switch. As a header proxy, `vmf_insert_page_mkwrite()` (from the
same v6.15 patch series) is absent from 6.8/6.11/6.14 and present in 6.17/7.0
(`verify/hdrcheck.sh`).

### Upstream XRT survey

`Xilinx/XRT` master (2026-10-21) has these driver-tree commits after the 2024.2 branch
point (`35058d94`). Kernel-compat ones that were used or adapted: 04350f685 (6.12 build: vmalloc.h, no_llseek, void
remove, I2C_CLASS_SPD, ulite kfifo), 6517a2a87 (FOP_UNSIGNED_OFFSET), f41db917c (6.16+:
bin_attribute const, timers, MODULE_IMPORT_NS, pfn_t, drm date, iommu_present, crc32c),
30c9f58ca (libxdma import_ns), a2657e374 (ida), dd3d69add (DRM_* macros), d1fbc5cea
(inline order) and libqdma commit 0a578f1 (timers, via the libqdma submodule).

Upstream changes that alter behaviour were not taken as-is. These are the CRC
(0010), ida ranges (0015), ulite NUL bytes (0003), the remove warning (0004), the 6.8
I2C class (0002), DRM log macros (0016) and the 6.16.0 off-by-one (0012).

Upstream has no fix at all for 0019 (P2P page refcounts, 6.15). Master's `p2p.c` was last
changed by 30c9f58ca (2026-01-30) and still maps P2PDMA pages that start at refcount 0.

Not needed for these Ubuntu kernels: the RHEL-only guards (e9fe46bc1, 30c9f58ca, 58074959d,
316d7d420), the Debian 6.7.12 eventfd boundary (daf5297a6), and the whole libqdma submodule
bumps (6517a2a87, e0ae8baf7, which also pull in a QDMA 2024.1 driver update). Also not
needed: the `stat_read()` `typeof` change in f41db917c. It only matters when the kernel is
built with gcc 14 or newer (`__seg_gs` per-CPU pointers). The 6.17 and 7.0 HWE kernels are
built with gcc 13.3 (`CONFIG_GCC_VERSION=130300`), and DKMS uses the same compiler.
Feature work and security fixes on master were left out on purpose.

### The U50 write-up as a checklist

https://jsupermaster.github.io/posts/xrt-alveo-u50-kernel6.17.html lists nine changes for
6.17. Each was re-checked against real headers:

| Write-up item | Verified boundary | Here | Difference |
| --- | --- | --- | --- |
| iommu_present removed "in 6.8" | 6.13 (6.8 still has it) | 0007 | guard at 6.13 |
| I2C_CLASS_SPD (6.10) | 6.10 | 0002 | feature test |
| UART circ_buf to kfifo (6.10) | 6.10 | 0003 | same |
| void remove (6.11), solved with `-Wno-error=incompatible-pointer-types` | 6.11 | 0004 | real fix; calling an int function through a void pointer is not acceptable |
| no_llseek, aliased to `noop_llseek` | 6.12 | 0005 | `noop_llseek` returns 0 where `no_llseek` returned -ESPIPE; the field is left NULL instead |
| crc32c_le "6.13", MODULE_IMPORT_NS (6.13) | 6.15 / 6.13 | 0010 / 0008 | feature test for the alias |
| timers "6.15" | del_timer* 6.15, from_timer 6.16 | 0011 / 0012 | separate boundaries |
| pfn_t "6.16", drm date "6.16" | pfn_t 6.17, date 6.14 | 0014 / 0009 | a 6.16 date guard breaks 6.14/6.15 |
| vmalloc.h "6.17" | 6.10 | 0001 | |
| FOP_UNSIGNED_OFFSET "6.17" | 6.12 | 0006 | feature test |

Not in the write-up: 0013 (6.16; its `-Wno-error` flag hides those errors too), 0015
(6.18), 0016 (6.19), and 0019 (6.15). 0019 is a run-time failure: a build with only the
write-up's nine changes compiles, but mapped P2P buffers would get SIGBUS on 6.15+.

## Verification done here

1. `build.sh --fetch --strict` from an empty directory, with the 19-patch series, for
   7.0.0-31, 7.0.0-38 and 6.17.0-35, and in a second empty directory for 7.0.0-28,
   6.8.0-147, 6.11.0-29 and 6.14.0-37: all OK, exit 0, with the warning counts in the
   status table. `--deb` and `--src` modes give the same result on 7.0.0-31.
   `--no-patches` fails on 7.0 and builds on 6.8, as expected.
2. Real DKMS, without root. The deb's DKMS tree was patched with the README's step-3 loop,
   which matches the patched manifest. Then `dkms add` + `dkms build` ran with
   `--dkmstree/--sourcetree` in the work directory, for the three kernels, using the deb's
   own `dkms.conf`. All `built`, with `make.log` free of errors and modpost findings. As
   non-root, kmodsign could not read the MOK key; as root DKMS signs as usual.
3. No change on 6.8: `verify/eqcheck.sh` builds the pristine and the patched tree against
   6.8.0-147 and compares all 228 objects (per-source objects of both PFs and the linked
   `xocl.o`/`xclmgmt.o`, debug info stripped, `__LINE__` and UBSAN line records
   neutralized): 0 differ. Negative control: dropping `I2C_CLASS_SPD` from xiic.c on 6.8
   makes `xiic.o` and `xclmgmt.o` differ.
4. Preprocessor check of the run-time items: `.fop_flags` is present on 6.14/7.0 and
   absent on 6.11; `.date` is present on 6.11 and absent on 6.14/7.0.
5. `modinfo` on 7.0: `vermagic 7.0.0-31-generic SMP preempt mod_unload modversions`,
   `import_ns: DMA_BUF`, `depends:` empty. (On 6.8 it is `libcrc32c`; since 6.15 crc32c
   is in the built-in CRC library.)
6. Review fixes (0003 comment/cast, new 0019), object level, with `verify/eqcheck.sh`:
   - 6.8.0-147, pristine vs. the 19-patch series: 228 objects, 0 differ.
   - 6.14.0-37, the 18-patch series vs. the 19-patch series: 231 objects, 0 differ.
     0019 is compiled out below 6.15, and dropping the `(char)` cast in 0003 changes no code.
   - 7.0.0-31 and 6.17.0-35, 18 vs. 19 patches: only `subdev/p2p.o` and the linked
     `xocl.o` differ. `ulite.o` is byte-identical, which confirms that the cast removal
     changes no code. In the 7.0 `p2p.o`, after the `devm_memremap_pages()` call, a loop
     stores 1 to offset 0x34 of each `struct page` (`_refcount`). That is the field
     `p2p_mem_chunk_release()`'s busy test reads.
7. 0019 boundary, from sources (not in the headers): mainline `mm/mm_init.c` v6.8/v6.14
   vs v6.15/v6.17/v7.0, `mm/memory.c` (`vm_insert_page()` refuses refcount 0 in all of
   them), `mm/memremap.c` (no ops needed for P2PDMA, `free_zone_device_folio()` only on
   0), and the Ubuntu source diffs `linux-hwe-6.14_6.14.0-37.37~24.04.1`,
   `linux-hwe-7.0_7.0.0-38.38~24.04.4` and `linux-hwe-6.17_6.17.0-42.42` from
   archive.ubuntu.com (none changes the P2PDMA refcount rule).

To repeat 3 for any kernel:

```
./build.sh --fetch --no-patches -w /tmp/xrt-a -k <kdir>
./build.sh --fetch              -w /tmp/xrt-b -k <kdir>
verify/eqcheck.sh -k <kdir> -a /tmp/xrt-a/src -b /tmp/xrt-b/src
verify/hdrcheck.sh /lib/modules/*/build
```

## On the card

The series was installed with DKMS and loaded on the development host (Ubuntu 24.04, kernel
7.0.0-38-generic). The U200 (`xilinx_u200_gen3x16_xdma_base_2`) came up on it, and every card run of
the project used these drivers (measured on the card, v1; logs in `build/card/`, not in git;
[`docs/platform-u200.md`](../../docs/platform-u200.md) §14):

| Check | Result |
| --- | --- |
| `xrt-smi examine` (`xbutil examine`) | XRT 2.18.179 with `xocl` and `xclmgmt` 2.18.179 (hash `3ade2e67`), the user PF listed, `Device Ready: Yes`, device status HEALTHY |
| `xrt-smi validate` | 8 tests passed: aux-connection, pcie-link, sc-version (passed with a warning: SC firmware 4.2 was running, 4.6.21 expected; `xbmgmt program --base` updated it, `docs/platform-u200.md` §14.1), verify, dma (16 MB buffer on bank0), iops (440,542 IOPS), mem-bw (67.2 GB/s over the four DDR4 banks), m2m (bank0 to bank1 11.8 GB/s) |
| xclbin load and BO paths | every M0, M3 and M4 run loaded an xclbin and moved data through device-only BOs of the four DDR4 banks; the host-to-card and card-to-host rates are in `docs/platform-u200.md` §14 |
| P2P | disabled (`P2P Status: disabled`); patch 0019 was therefore **not** exercised |

Not covered by what ran: patch 0019 (P2P), and no record shows whether the UART-lite (0003) and
CMA/userptr (0014) paths were reached; both need hardware this card does not use. The list under
"What each check covers" below says which patches the checks that did run depend on.

## Warnings

What remains after the series:

| Warning | Count | Kernels | Why it is left |
| --- | --- | --- | --- |
| `no previous prototype for ...` `[-Wmissing-prototypes]` | 43 (38 functions) | all, incl. 6.8 | identical set on the unpatched 6.8 build. Non-static functions such as `mailbox_isr` or `xocl_userpf_probe` without a header prototype; the link and modpost are clean. Adding prototypes or `static` is not a kernel-API change. |
| `-Wstringop-overread` in libfdt `fdt_ro.c` (`strrchr`) and `fortify-string.h` (`strlen`) | 4 | all, incl. 6.8 | identical on 6.8; a gcc 13 false positive on a zero-length array in libfdt |
| `"FIELD_GET" redefined` | 14 | 6.11, 6.14, 6.17 only | libqdma defines its own `FIELD_GET`. On these kernels `<linux/bitfield.h>` gets pulled in too, so in 13 files libqdma's definition wins, as it does on 6.8. In `qdma_platform.c` the kernel's wins, but that file does not use the macro. Both compute `(reg & mask) >> ctz(mask)`. 7.0 does not include bitfield.h there, so no warning. Removing libqdma's macro (as the 2025.2 QDMA release did) would change code on 6.8 and 7.0. |
| `the compiler differs from the one used to build the kernel` | 2 | all | same gcc 13.3.0-6ubuntu2~24.04.1; the kernel recorded it as `x86_64-linux-gnu-gcc-13` |
| `pahole version differs from the one used to build the kernel` | 2 | 7.0 | pahole is not installed. Module BTF is skipped anyway ("unavailability of vmlinux"), exactly as in a stock Ubuntu DKMS build |

There are no objtool warnings, no modpost warnings or undefined symbols, and no section
mismatches. `build.sh --strict` fails on any warning not in this table.

## Applying it (needs sudo)

1. Install XRT. Its DKMS step is expected to fail on 6.17/7.0 kernels; the user-space
   part installs anyway.

   ```
   sudo apt install ./xrt_202420.2.18.179_24.04-amd64-xrt.deb
   ```

2. Check the installed source is the expected pristine tree (prints `source tree: pristine`
   and builds for the running kernel as a normal user):

   ```
   cd sw/xrt-driver
   ./build.sh --src /usr/src/xrt-2.18.179 -w /tmp/xrt-check -k "$(uname -r)"
   ```

3. Patch the DKMS source in place (a dry run first, then for real):

   ```
   cd /usr/src/xrt-2.18.179
   for p in <repository>/sw/xrt-driver/patches/*.patch; do
       patch -p1 --dry-run --forward --batch < "$p" > /dev/null || { echo "FAILS: $p"; break; }
       sudo patch -p1 --forward --batch --no-backup-if-mismatch < "$p" || break
   done
   ```

   The loop dry-runs each patch on the tree as left by the earlier ones. To see the
   result: `./build.sh --src /usr/src/xrt-2.18.179 -w /tmp/xrt-check2 -k "$(uname -r)"`
   should print `source tree: patched`.

4. Rebuild and install with DKMS for every kernel that has headers. `dkms remove --all`
   clears the failed build state and the registration; it never deletes /usr/src. If it
   reports that the module is not added, nothing was registered, which is also fine.

   ```
   sudo dkms remove -m xrt -v 2.18.179 --all
   sudo dkms add -m xrt -v 2.18.179
   for k in 7.0.0-31-generic 7.0.0-38-generic 6.17.0-35-generic 7.0.0-28-generic; do
       sudo dkms build   -m xrt -v 2.18.179 -k "$k" &&
       sudo dkms install -m xrt -v 2.18.179 -k "$k"
   done
   dkms status -m xrt
   ```

   Expected: `xrt/2.18.179, <kernel>, x86_64: installed` for each kernel. A generic form of
   the list is `for k in $(ls /lib/modules); do [ -e /lib/modules/$k/build/Makefile ] || continue; ...`.
   Future kernel updates rebuild automatically, because `AUTOINSTALL="yes"` and /usr/src
   now holds the patched tree.

5. The deb's postinst installs the udev rules and loads the modules only when its own DKMS
   step succeeded, so do that part by hand:

   ```
   sudo install -m 644 /usr/src/xrt-2.18.179/driver/xocl/userpf/99-xocl.rules /etc/udev/rules.d/
   sudo install -m 644 /usr/src/xrt-2.18.179/driver/xocl/mgmtpf/99-xclmgmt.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo modprobe xclmgmt && sudo modprobe xocl
   sudo udevadm trigger
   ```

   Without a card the modules load and sit idle.

Notes:

- Do not `apt install --reinstall` the deb afterwards: it restores the unpatched /usr/src
  tree. After a reinstall or upgrade of the same version, repeat steps 3 to 5.
- To undo: `sudo dkms remove -m xrt -v 2.18.179 --all`, then reverse the patches
  (`for p in $(ls -r .../patches/*.patch); do sudo patch -p1 -R --batch < "$p"; done` in
  /usr/src/xrt-2.18.179), or reinstall the deb.
- Secure Boot is off. DKMS on Ubuntu still signs the modules with the MOK key if one
  exists, which does no harm.

## Checks on the card

The U200 needs its deployment platform packages (gen3x16_xdma_2 base, plus validate, CMC
and SC firmware) from the AMD U200 download page for `xbmgmt program` and `xbutil validate`.

```
lspci -d 10ee: -nn                                  # two functions: mgmt PF (.0) and user PF (.1)
lsmod | grep -E '^(xocl|xclmgmt) '
modinfo -F vermagic xocl ; modinfo -n xocl          # vermagic == $(uname -r), path under updates/dkms
sudo dmesg | grep -iE 'xocl|xclmgmt' | tail -50     # probe messages
sudo dmesg | grep -E 'drm_open_helper|remove callback returned|failed to get the same CRC'   # expect nothing
ls -l /dev/dri/renderD* /dev/xclmgmt* /dev/xfpga/   # 0666 per the udev rules
source /opt/xilinx/xrt/setup.sh
sudo xbmgmt examine                                 # card, shell, SC/CMC versions, mgmt BDF
sudo xbmgmt program --base --device <mgmt BDF>      # only if examine reports a mismatched or golden shell; then a full power cycle
xbutil examine                                      # user PF listed and ready (needs a working open() of the render node: patch 0006)
xbutil examine -d <user BDF> -r all
xbutil validate -d <user BDF>                       # aux-connection, pcie-link, sc-version, verify, dma, ... all PASSED
```

P2P is off after every cold boot, so the plain `validate` above skips the p2p test. P2P is
the only path that exercises 0019. P2P needs "Above 4G decoding" in the BIOS and a 64 GB
I/O window; see XRT's `p2p.rst`.

```
sudo xbutil configure --device <user BDF> --p2p enable   # may ask for a WARM reboot (BAR resize); reboot, do not power-cycle
xbutil examine -d <user BDF>                             # P2P status: enabled (not "no iomem")
xbutil validate -d <user BDF> -r p2p                     # maps an XCL_BO_FLAGS_P2P buffer and reads/writes it from the CPU: PASSED
sudo dmesg | grep -iE 'p2p|SIGBUS|refcount|bad page' | tail -30   # "reserved P2P mem chunk ... ret: 0", no page/refcount warnings
sudo xbutil configure --device <user BDF> --p2p disable  # optional afterwards; works without a reboot
```

Without 0019, the p2p test on 6.15+ (6.17.0-35, 7.0.0-x) dies with SIGBUS on the first
access to the mapped P2P buffer.

What each check covers:

- `xbutil examine` reporting the device as ready exercises 0006 (render node `open()`) and
  0013 (the sysfs bin files: `ip_layout`, `mem_topology`, ...).
- `xbmgmt examine` exercises xclmgmt's sysfs and, through the mailbox, 0010.
- The dma and verify tests of `xbutil validate` exercise the BO and xclbin paths.
- `xbutil validate -r p2p` with P2P enabled exercises 0019 (`vm_insert_page()` of P2P
  pages in the fault handler) and 0007 (the IOMMU test that decides whether the P2P BAR
  address is reported to the management PF).
- 0003 and 0014 only run with a UART-lite subdevice and with CMA/userptr BOs respectively,
  so a clean validate does not cover them.

## Limitations

- The card exercised part of the series (section "On the card"); the rest is build-, header- and
  object-level evidence.
- 0007 can differ from 6.8 only on a machine where an IOMMU is registered but does not
  translate the Alveo card. Then 6.13+ hands the P2P BAR to the management PF; that is the
  per-device answer the check is meant to give. No 6.13+ API gives the old bus-wide answer.
- 0009: on 6.13+ kernels `DRM_IOCTL_VERSION` reports the date as "0" instead of
  20241105. The kernel did that itself; it is not a choice of this port.
- RHEL-only code paths in the tree were left untouched.
- A kernel built with gcc 14 or newer would also need upstream f41db917c's `stat_read()`
  change (see the survey). A future Ubuntu kernel may need more patches;
  `./build.sh -k <new kernel>` shows it, and `verify/hdrcheck.sh <kdir>` flags version
  guards that its headers contradict.
- 0019 cannot be checked by `hdrcheck.sh` directly, because the kernel change is in
  `mm/mm_init.c`. For a new kernel, check that its `__init_zone_device_page()` still
  zeroes `MEMORY_DEVICE_PCI_P2PDMA` pages (6.15+), or run the P2P checks above.

## Review log

Findings from the independent review, and what was done:

| Finding | Verdict | Action |
| --- | --- | --- |
| P2P BOs: on 6.15+ `vm_insert_page()` fails on the refcount-0 P2PDMA pages, so mapped P2P buffers get SIGBUS (reported twice) | confirmed in v6.14/v6.15/v7.0 `mm/mm_init.c`, `mm/memory.c` and `drivers/pci/p2pdma.c`, plus the Ubuntu source diffs | new patch 0019, P2P items added to the hardware checklist, 0014's commit message clarified |
| Optional part of that finding: add `dev_pagemap_ops` with a free callback (`page_free` up to 6.17, `folio_free` in 7.0) | not taken | With 0019 the count never reaches 0, so the callback would never run. On 6.8 xocl has no ops either, and a stray drop to 0 hits the same `WARN_ON_ONCE` there. Adding ops would add code and behaviour that 6.8 does not have. |
| 0003: the old `char` was not sign-extended, because kbuild uses `-funsigned-char` | confirmed (6.8.0-147, 6.17.0-35 and 7.0.0-31 Makefiles; mainline since v6.2) | cast dropped, comment, commit message and table row corrected. `ulite.o` is byte-identical before and after on 6.17 and 7.0. |

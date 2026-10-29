# XRT host transport and card bring-up (M3 part C)

Status: 2026-10-24. The "xrt" transport, the device-access layer under it, the "mock" adapter
model, `tsr_xrt_selftest` and their tests are implemented and pass without hardware, and the
unchanged host binaries ran over `TSR_DEVICE=xrt` against hardware emulation of the k1 kernel
(§9). Nothing has run on a card: the U200 and XRT are not installed, the U200 development platform
is only extracted on the dev server (§8 lists exactly what is untested). The hardware xclbin
`build/u200/k1/tsr_u200_k1.xclbin` is built (`docs/platform-u200.md` §8.2). Source-of-truth order: `docs/spec.md` > the kernel contract below (fixed by the lead;
the hardware side is `rtl/platform/alveo_xrt/` and its document `docs/platform-u200.md`, both
by the adapter agent) > this document.

**v2 (2026-10-30).** This document describes the v1 transport (tag `v1-m4`). In v2 the transport
and the adapter change (`docs/v2/01-execution-model.md` §9–§10):
- The kernel is `ap_ctrl_none`; the xrt kdev opens it through `xrt::ip` (no `xrt::kernel`, no
  `xrt::run`), polls, and binds the channels by writing `ch<c>_base` and then
  `ADAPTER_CTRL.MEM_EN` (0x0E0). `cmd_addr`, `cmd_count`, `arg0`–`7` and the `ap_ctrl_hs` control
  registers are gone.
- The `ap_ctrl_hs` run path is removed in X0, the first M6 commit (EX §12): `tsr_transport_exec`,
  `tsr_exec_result`, the kdev `start` / `wait` / `settle` / `kargs`, `regs_start` / `regs_wait`,
  `kds_start` / `kds_wait`, `TSR_XRT_CTRL`, and `tsr_xrt_selftest`'s program checks. The run
  sequencer, queued-start behaviour and interrupt output of `tsr_xrt_ctrl` go with it.
- The v2 API (EX §10.1) loads and boots a firmware image and then launches kernels through the
  launch queue and completion ring (EX §4), with a debug-aware wait and DB's recovery ladder; one
  protocol layer serves sim, mock, vkernel and xrt (EX §10.2, SW §9.1). `tsr-selftest` launches
  the `selftest` kernel (SW §7.6). `TSR_XRT_TIMEOUT_MS` / `TSR_XRT_ABORT_MS` become
  `TSR_LAUNCH_TIMEOUT_MS` / `TSR_ABORT_MS`; `TSR_IMAGE`, `TSR_BOOT_MS`, `TSR_RESET_MS` and
  `TSR_ATTACH` are new.
- The device memory (§4.2: one device-only BO per channel, `TSR_XRT_DMA*`, `TSR_XRT_CH_GIB`) is
  unchanged, except that the BO's memory bank comes from `xrt::xclbin` instead of an
  `xrt::kernel` (*confirm* in P3.1).

The card results in `docs/platform-u200.md` §14 (M0, M3 and M4) were measured with this v1
transport. A working binary copy of the v1 tools, xclbins and an `env.sh` is kept in `build/v1`
(not in git) for card runs until v2 runs on the card.

## 1. Layers

```
ggml-tesseract backend, tools          tsr_transport.h (C API: open, config, write, read, exec, csr_read)
  └─ tsr_transport.c                   dispatcher: TSR_DEVICE = sim | xrt | mock | vkernel | <registered kind>
       ├─ "sim"   tsr_transport_sim.c  ISA simulator in-process (unchanged)
       └─ protocol layer  tsr_kdev.cpp  kernel contract on top of any kdev (§3)
            ├─ "xrt"   tsr_transport_xrt.cpp   XRT 2.18 native API (built with -DTSR_WITH_XRT=ON)
            ├─ "mock"  tsr_kdev_mock.cpp       adapter register model around the ISA simulator
            ├─ "vkernel" tsr_kdev_vkernel.cpp  the Verilated kernel in a server process (docs/platform-u200.md §13)
            └─ others  tsr::register_kdev()
```

`tsr_kdev.h` defines the device-access layer: a `tsr::kdev` gives 32-bit `s_axi_control`
register access, per-channel memory access at any byte offset / size, `start(args)` /
`wait(timeout)` and `settle()`. The protocol layer never touches `AP_CTRL`, `GIE` / `IER` / `ISR`
or the argument registers itself: under XRT the scheduler owns them (and `ap_done` is
clear-on-read), so they belong to `start` / `wait`, and `settle` (default: one `AP_CTRL` read) is
the only `AP_CTRL` access the protocol layer makes, at open and after every completed `wait`,
when no run is outstanding (§3). Kdevs that own `AP_CTRL` (mock, vkernel, xrt with
`TSR_XRT_CTRL=regs`) use the shared helpers `tsr::regs_start` / `tsr::regs_wait`, so the
register sequence the mock is tested with is the one the card sees in register mode;
`tsr::kds_start` / `tsr::kds_wait` replay what the XRT scheduler does for `xrt::run` (write the
argument map and `AP_CTRL = 1` without reading `AP_CTRL` first, complete on the first read with
`ap_done`; xocl `cu_hls.c`, ERT `scheduler.cpp`), and the mock runs either (`TSR_MOCK_CTRL`).

A second kdev implementation plugs in either directly (`tsr::transport_open_kdev(std::unique_ptr<kdev>)`
returns an ordinary `tsr_transport *`) or by name (`tsr::register_kdev("<kind>", factory)`
before `tsr_transport_open`, which makes `TSR_DEVICE=<kind>` work for the backend and every tool;
"vkernel" is built in, docs/platform-u200.md §13).
`tsr::transport_kdev(t)` returns the kdev behind a transport (tools use it for `ADAPTER_*`);
`tsr::transport_adapter_log(t)` returns what the protocol layer saw in ADAPTER_STATUS (§3).

## 2. Kernel contract (software view)

**v1 (2026-10-30, W0 of `docs/v2/README.md`).** The v2 contract is EX §9.2: `ap_ctrl_none`, only
`ch<c>_base` args, `ADAPTER_CTRL.MEM_EN` at 0x0E0, `ADAPTER_VERSION` 2.

Vitis RTL kernels `tsr_u200_k1` (NUM_CH 1, T_MAX 1) and `tsr_u200_k4` (NUM_CH 4, T_MAX 8),
`ap_ctrl_hs`, one clock `ap_clk`, `ap_rst_n`, `interrupt`; `s_axi_control` AXI4-Lite with a
13-bit address space, `m_axi_ch<c>` AXI4 512-bit for c < NUM_CH. `s_axi_control` (byte offsets):

| Offset | Register | Software use |
| --- | --- | --- |
| 0x000 | AP_CTRL: bit0 ap_start, bit1 ap_done (COR), bit2 ap_idle, bit3 ap_ready (COR), bit7 auto_restart (unsupported) | `start` / `wait` only |
| 0x004 / 0x008 / 0x00C | GIE / IER bit0 / ISR bit0 (toggle on write) | XRT scheduler (interrupt mode) |
| 0x010 / 0x014 | cmd_addr (u64, ch0 beat address) | kernel arg, offset 0x10, 8 bytes |
| 0x018 | cmd_count (u32); 0x01C reserved | kernel arg |
| 0x020 + 4i | arg0..arg7 (u32) | kernel args |
| 0x040 + 8c | ch<c>_base (u64, 4 KiB aligned) | kernel buffer arg bound to `m_axi_ch<c>` |
| 0x0F0 | ADAPTER_STATUS, R / W1C sticky: bit0 RRESP, bit1 BRESP, bit2 response without request, bit3 base misaligned; [31:16] saturating error count | read after every run, cleared with 0xFFFFFFFF |
| 0x0F4 | ADAPTER_VERSION (tsr_xrt_ctrl v1: {16'h0001, 8'h00, NUM_CH}) | printed, NUM_CH cross-checked by the selftest |
| 0x1000 + off | the core CSR `off` (spec CSR map), any time, also while running | config, STATUS, ERR_*, PC, PERF, CTRL.ABORT |

The xrt kdev finds the arguments by register offset, not by name (`rtl/platform/alveo_xrt/kernel_k*.xml`
uses `cmd_addr`, `cmd_count`, `arg0..7`, `ch0..3`, but any names work). It refuses an xclbin whose
kernel has more than one compute unit, is not `ap_ctrl_hs`, declares an `s_axi_control` range
below 0x2000 (XRT's register access checks `offset + 4 <= range`, `xrt_kernel.cpp`
`get_cuidx_or_error`; the range comes from the kernel XML port, `xclbin_parser.cpp`
`get_address_range`), or has an argument outside the map above.

## 3. Protocol layer (tsr_kdev.cpp)

**v1 (2026-10-30, W0 of `docs/v2/README.md`).** The exec path below is deleted in X0. The v2
protocol layer (open with `SOFT_RESET` and bind, load and VERIFY, boot, launch, wait, cancel, stop,
soft reset, log) is EX §10.2 and SW §9.1.

**Open.** Reads ADAPTER_VERSION, then ID / VERSION / CFG0..2 / CAPS through the window. Fails
with a clear message if ID is not 0x54535231 ("TSR1"), the ISA major differs, CFG0.NUM_CH differs
from the number of channel buffers, or a buffer is not 4 KiB aligned. The configuration comes
from the CSRs; BEAT_BITS, NIREG, LOOP_DEPTH, RD_OUTSTANDING and ADDR_BEAT_W from the U200 v1
values; `ch_bytes` is the smallest bound buffer (rounded down to a beat, at most 16 GiB). A core
left BUSY by an earlier process is stopped with CTRL.ABORT (fail if it does not stop within
`TSR_XRT_ABORT_MS`). Then `settle` reads `AP_CTRL` until the kernel is idle (fail if it is not
within `TSR_XRT_ABORT_MS`): that consumes a completion nobody consumed — the run the ABORT just
stopped, or a run an earlier process started and did not wait for (a `TSR_XRT_CTRL=regs` process
that died; reported as "consumed a completion left by an earlier run"). This is required because
accepting a start does not clear `ap_done` in `tsr_xrt_ctrl` (only a read does) and the XRT
scheduler never reads `AP_CTRL` before a start: a leftover `ap_done` would complete the next
`xrt::run` at once, with the program still running, and every later run would then complete on
the previous one's `ap_done` (review finding, reproduced on the RTL and on the RTL-faithful mock;
fixed in the host, because clearing `ap_done` on accept would break the adapter's queued-start
behaviour). A stale ADAPTER_STATUS is reported, kept in the adapter log (`at_open`) and cleared.

**Exec.** `start({cmd_addr, cmd_count, arg0..7})` (every ch<c>_base is the buffer address), then
`wait(TSR_XRT_TIMEOUT_MS)`. On timeout: PC and STATUS are read and reported, CTRL.ABORT is written
through the window, and the transport waits `TSR_XRT_ABORT_MS` for the core to stop (spec: ABORT
is honoured between instructions, ERR_CODE = ABORTED, ERR_PC = first unexecuted instruction);
the exec returns an error either way. If the core does not stop (a stuck AXI transaction), the
transport refuses every further exec ("reset the card, e.g. xbutil reset"). After the
completion, `settle` reads `AP_CTRL` once: the kernel must be idle. If it is not, the completion
was a stale `ap_done` that appeared after the open (not expected while this process holds the
compute unit exclusively: a safety net, and the path the mock tests exercise): the transport says so, waits for this run to end within the rest of `TSR_XRT_TIMEOUT_MS` (reading
`AP_CTRL`, which also consumes this run's `ap_done`, so the next run cannot complete on it), and
otherwise stops it like a timeout. Then STATUS / ERR_CODE / ERR_PC / ERR_INFO are read through
the window and ADAPTER_STATUS is read; non-zero → the decoded bits are printed, recorded in the
adapter log (`last_run`, `sticky` bits, `runs`, `count`), the register is cleared and the exec
fails even if the core reported DONE without ERR. `exec` returns 0 only for DONE without ERR, no
timeout and a clean adapter. Tools that must tell an adapter error from a core error (the
selftest) read `tsr::transport_adapter_log`: the register itself is clean after every exec.

**Memory.** Range-checked against `ch_bytes`, then `kdev::mem_write` / `mem_read`.
**CSR reads** (`tsr_transport_csr_read`) go to `0x1000 + off`; `tsr::transport_perf` reads a
64-bit PERF counter LO first (the core snapshots on LO, rtl-m2 P5).

## 4. The xrt kdev and device memory

### 4.1 Control

**v1 (2026-10-30, W0 of `docs/v2/README.md`).** Both `TSR_XRT_CTRL` modes are deleted in X0. v2 uses
`xrt::ip` register access with polling and never starts an `xrt::run` (EX §9.1).

`TSR_XRT_CTRL=run` (default): `xrt::kernel(device, uuid, name, cu_access_mode::exclusive)` and
one `xrt::run` with the channel buffers bound once; `start` sets cmd_addr (u64), cmd_count and
arg0..7 (u32) and calls `run.start()`, `wait` is `run.wait(timeout)`. The XRT scheduler writes
the argument registers, sets ap_start and catches ap_done (polling or interrupt). Exclusive
access is what XRT 2.18 requires for `read_register` / `write_register` (deprecated in favour of
`xrt::ip`, which cannot share a compute unit with `xrt::run`; the warning is silenced locally).
`TSR_XRT_CTRL=regs`: the transport writes the argument registers and AP_CTRL itself and polls
ap_done (`regs_start` / `regs_wait`), bypassing the scheduler: the same sequence as the mock.
It is the fallback if the scheduler path misbehaves on the card.

The xclbin is loaded with `device.load_xclbin` (the Alveo shim of 2.18 runs the legacy flow:
`register_xclbin` is a no-op there, `shim.cpp` `register_xclbin`, and `create_hw_context` returns
slot 0). Only under XRT's no-op shim (the ctest, §6) it uses `register_xclbin`, the only flow that
shim implements.

### 4.2 Buffers: one device-only BO per channel

Each channel gets one `xrt::bo(device, size, xrt::bo::flags::device_only, kernel.group_id(ch arg))`.
Size: `TSR_XRT_CH_GIB` (1..16), default the bank size from the xclbin memory topology (16 GiB on
the U200 DDR4 banks), shrunk in 256 MiB steps if the allocation fails (down to 1 GiB). Evidence
from the XRT 2.18.179 sources (`build/xrt-driver/xrt-202420`, driver `.../core/pcie/driver/linux/xocl`):

- A device-only BO has no host pages: `xocl_create_bo` (`userpf/xocl_bo.c`) reserves a
  `drm_mm` node in the bank (`xocl_drm.c`, PAGE_SIZE alignment, so bases are 4 KiB aligned) and
  allocates no pages for `XOCL_BO_DEV_ONLY`; the 4 GiB limit in that file applies only to
  host-memory BOs on kernels < 4.13. Host RAM therefore does not grow with the channel size.
- `xrt::bo::write` / `read` on such a BO (`common/api/xrt_bo.cpp`, class `buffer_dbuf`) call
  `unmgd_pwrite` / `unmgd_pread` at `bo.address() + offset` = `xclUnmgdPwrite` / `Pread`
  (`pcie/linux/shim.cpp`) = ioctls `DRM_IOCTL_XOCL_PWRITE_UNMGD` / `PREAD_UNMGD`. The driver
  (`xocl_migrate_unmgd`, `xocl_init_unmgd`) pins the caller's pages for the duration of the
  call (`get_user_pages_fast`), builds a one-time scatter list starting at the caller's byte
  offset and runs the XDMA engine on it: one transfer, no bounce copy, any offset (also beyond
  4 GiB: 64-bit `paddr` and `size`). The ioctls carry the same flags as `CREATE_BO`
  (`DRM_AUTH | DRM_UNLOCKED | DRM_RENDER_ALLOW`), so no root is needed.
- Limits: the XDMA fast path waits at most 10 s per descriptor batch (`libxdma.c`
  `xdma_xfer_fastpath`, `subdev/xdma.c` `xdma_migrate_bo`), a descriptor covers < 256 MiB.
  Transfers are therefore issued in chunks of `TSR_XRT_DMA_CHUNK_MIB` (default 64 MiB), which
  also bounds the pinned page list per call. The engine's alignment register is read by
  `libxdma.c` (`engine_alignments`) but never enforced in software: whether the shell's XDMA
  accepts unaligned host / card addresses is a property of the IP configuration, checked on the
  card by the selftest's unaligned ranges (§5).
- `xrt::bo::copy` (M2M / KDMA) would need a host-backed staging BO and a second device-side
  copy: it first queries the platform's M2M, then KDMA (`xrt.ini`), and otherwise copies through
  host buffers, which a device-only destination does not have (it throws). Whether the U200
  shell has M2M is a platform property (`xbutil examine -r platform`).

| `TSR_XRT_DMA` | Path | Host RAM | Status |
| --- | --- | --- | --- |
| `direct` (default) | device-only BO, `bo.write` / `bo.read` = unmanaged XDMA from the caller's memory, in chunks | the caller's buffer only | best: one PCIe transfer, no copy |
| `staged` | the same ioctls through an aligned bounce buffer (`TSR_XRT_DMA_CHUNK_MIB`), edges read-modify-written, `TSR_XRT_DMA_ALIGN` default 64 | one chunk | fallback for an XDMA that needs aligned addresses, or memory that cannot be pinned |
| `m2m` | host-backed staging BO per channel + `sync` + `bo.copy` into the device-only BO | one chunk per channel (host and device) | only for shells without unmanaged DMA; needs M2M or KDMA. The staging BO (device memory in the channel's bank) is allocated at open, before the channel BO, so that the default full-bank channel BO shrinks around it (allocated lazily after it, it found the bank full and every transfer failed: review finding) |

A failed `direct` transfer is retried once through `staged`; if that works the transport stays
on `staged` (one warning). The chunk / alignment / read-modify-write logic is
`tsr::dma_engine` (shared with the mock and tested there, §6).

### 4.3 Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `TSR_DEVICE` | `sim` | `xrt` for the card, `mock` for the adapter model, `vkernel` for the Verilated kernel (`TSR_VKERNEL_*`, docs/platform-u200.md §13) |
| `TSR_XCLBIN` | (required) | xclbin to load; loading the one already on the card is cheap |
| `TSR_XRT_DEVICE` | `0` | device index or PCIe BDF |
| `TSR_XRT_KERNEL` | `tsr_u200_k4`, else `tsr_u200_k1` | kernel name |
| `TSR_XRT_CH_GIB` | bank size (16); 3 under `XCL_EMULATION_MODE=hw_emu` (§9) | channel buffer size in GiB |
| `TSR_XRT_CTRL` | `run` | `run` (XRT scheduler) or `regs` (§4.1) |
| `TSR_XRT_DMA`, `TSR_XRT_DMA_CHUNK_MIB`, `TSR_XRT_DMA_ALIGN` | `direct`, 64, 1; `staged`, 64, 4096 under `XCL_EMULATION_MODE=hw_emu` (§9) | §4.2 |
| `TSR_XRT_TIMEOUT_MS` | 60000 (0 = none) | exec timeout, every kdev kind |
| `TSR_XRT_ABORT_MS` | 10000 | wait for the core after CTRL.ABORT |
| `TSR_MOCK_NUM_CH`, `TSR_MOCK_TMAX`, `TSR_MOCK_CH_GIB` | 4, 1, 16 | mock core configuration |
| `TSR_MOCK_DMA_ALIGN`, `TSR_MOCK_DMA_CHUNK_KIB`, `TSR_MOCK_DMA_BOUNCE` | 1, 65536, 0 | DMA rules the mock enforces |
| `TSR_MOCK_CTRL` | `regs` | `regs` (`regs_start` / `regs_wait`) or `kds` (the XRT scheduler's sequence, the `xrt::run` path) |
| `TSR_MOCK_INJECT` | — | tool tests: `N:V[,N:V...]` ORs V into ADAPTER_STATUS when the N-th run ends; N = 0: before the open |

With `TSR_XRT_CH_GIB` below 16 the core still range-checks against its CH_BYTES = 16 GiB: a
faulty program could address memory past the buffer (other BOs, or a DECERR that shows up in
ADAPTER_STATUS). The backend never does (its heap ends at `ch_bytes`).

## 5. tsr_xrt_selftest

**v1 (2026-10-30, W0 of `docs/v2/README.md`).** Replaced in v2 by `tsr-selftest`, whose device
programs are the `selftest` kernel, compared with the same image on the in-process simulator (SW
§7.6, V2-SW-28); the trace replays go through the `replay` kernel. A v1 binary is in `build/v1/bin`.

`bin/tsr_xrt_selftest [--device KIND] [--mem-mib N] [--no-mem] [--no-programs] [--traces DIR]... [--max-traces N] [--seed N] [-v]`,
default device `$TSR_DEVICE` or `xrt`. Exit code 0 = every check passed, 1 = a check failed,
2 = the transport did not open.

1. Identity: kdev description (device name / BDF, xclbin UUID, kernel, control and DMA mode,
   per-channel memory group, size, device address), ID, VERSION, CFG0..2, CAPS decoded,
   ADAPTER_VERSION (NUM_CH field checked), ADAPTER_STATUS clean at open (the value the protocol
   layer found and cleared: a leftover of an earlier process fails the check) and after it, bases
   4 KiB aligned.
2. Memory per channel: pattern write / read-back at [0, 64), [1, 2), [63, 65), a 4 KiB-crossing
   range, a 1 MiB + 13 range at an odd offset, a range across 4 GiB, one at 12 GiB + 4095 and the
   last 777 bytes of the channel, each with 64 guard bytes on both sides that must stay intact
   and an exact read into an odd host address; channel aliasing (same offset, different data,
   at 1 MiB and 4 GiB + 1 MiB); host <-> card bandwidth of an `--mem-mib` transfer per channel
   (wall clock).
3. Programs, each compared with the ISA simulator given the same program and memory (STATUS,
   ERR_CODE / PC / INFO and every stored byte bit-exact; PERF_RETIRED and PERF_CH_RD_BEATS[c >= 1]
   equal, as in the RTL testbenches; cycles, MVM cycles and channel-0 beats printed): END; LD +
   ST through every ARG register as an address + LI / ADDI (catches argument packing); LD -> ST
   on every channel (200-line bursts split at 4 KiB, an aligned 4 KiB burst, one line,
   cross-channel); QACT A32 + A256 (K = 1024, T = min(2, T_MAX), zero block and subnormal
   inputs); MVM Q8_0 (N = 4C + 3) and Q4_K (N = 2C + 1), K = 512, T = 1 and T = T_MAX with ACC;
   a DRAM_RANGE fault.
4. `--traces DIR`: every trace directory under DIR that the device can replay exactly (same
   NUM_CH / LANES / K_MAX / SPM / VL / NSREG / CAPS; trace T_MAX ≤ device T_MAX unless the trace
   ends in T_RANGE; same DRAM limit, or no DRAM_RANGE ending and every region inside the buffer)
   is loaded (cmd.bin clipped to the channel like `tsr_trace_replay`, then the init regions),
   run, and compared: final STATUS / ERR_*, the exec return code (0 exactly for DONE without
   ERR), every expect region, PERF_RETIRED and PERF_CH_RD_BEATS[c >= 1]. Writes outside the expect
   regions are not detected (the card's memory is not zeroed or scanned).
5. Every program and trace run also fails when the adapter flagged an error during it
   (`transport_adapter_log(t).last_run`), even with matching results or an expected error ending
   (before, a trace run with a BRESP error counted as passed and the tool exited 0: review
   finding); the last check requires no adapter error in any run and a clean register.

## 6. Tests (no hardware) and measured results

Build: `cmake -S . -B <build> -DTSR_WITH_XRT=ON [-DTSR_XRT_ROOT=...]` (§7). All numbers below are
from the scratch build of 2026-10-23 with `TSR_WITH_XRT=ON` against the unpacked 2.18.179 deb.

| ctest | What | Result |
| --- | --- | --- |
| `transport_kdev` | `dma_engine` against a strict fake DMA (rules asserted on every raw call, shadow memory, guard bytes): 9 rule sets (align 1 / 64 / 4096, chunks 7 B .. 64 KiB, forced bounce) × 256 transfers at offsets 0 .. 16 GiB incl. across 4 GiB, sizes 0 .. 3 chunks + 7, misaligned host pointers; the mock's registers against `tsr_xrt_ctrl` (ap_done / ap_ready kept over an accept, 1-bit IER / ISR, ADAPTER_STATUS W1C of the count); protocol on the mock, once with `regs_start` / `regs_wait` and once with the scheduler's sequence (`kds_*`): config decode, arg packing (ARG0..7 as store addresses past 4 GiB, LI / ADDI, CMD_ADDR lo / hi split via a fetch fault at beat 2^32 + 5), window offsets, PERF via window, delayed ap_done (300 polls), odd-offset transfers across 4 GiB, range refusals, timeout → ABORT → ABORTED (ERR_PC 0, 40 ms timeout), ADAPTER_STATUS after a run (RRESP + orphan + count 3) and its W1C, misaligned base (core not started), stuck core → transport refuses runs, open with a BUSY core / stale status / wrong ID / stuck core, stale completions (an earlier process's run still BUSY at open or ended with its `ap_done` left; a stale `ap_done` after the open before a run that ends by itself and before one that must be aborted: results and the following runs exact, nothing left in AP_CTRL), the adapter log, `TSR_MOCK_CTRL` / `TSR_MOCK_INJECT`, registry, no AP_CTRL / IER / ISR / argument access outside start / wait / settle | 9613 checks, 0 failed (the protocol layer without `settle`: 10 failed, all in the scheduler-sequence stale-completion cases) |
| `transport_xrt_noop` (XRT builds) | the xrt kdev on XRT's no-op shim with generated metadata-only xclbins: xclbin load, kernel pick (k4, else k1), args by offset, memory groups (found the slot bits in `group_id`, fixed), bank-size default, device-only BOs, 3 × `xrt::run` start / wait with u64 / u32 args; refusals: range 0x1000, a foreign argument, no Tesseract kernel, `TSR_XRT_CH_GIB` > bank, bad `TSR_XRT_CTRL`; register access and unmanaged DMA fail cleanly (the shim implements neither); `TSR_XRT_DMA=m2m` allocates the staging buffers at open (the shim does not enforce bank capacity, so the full-bank case itself is not reproducible here) | 28 checks, 0 failed |
| `transport_ggml_e2e_mock`, `..._tmax8` | `test-ggml-tesseract` (bit-exact backend checks) on `TSR_DEVICE=mock`, DMA align 64 / chunk 64 KiB; T_MAX 8, 3 channels, align 4096 / chunk 12 KiB | pass |
| `tools_xrt_selftest_mock` | selftest, 4 ch T_MAX 1, DMA align 4096 / chunk 1 MiB, + `traces/m2`, `traces/gen` | pass: 124 traces replayed, 80 skipped (other configs) |
| `tools_xrt_selftest_mock_tmax8` | the same at T_MAX 8 (the k4 configuration), align 64, `TSR_MOCK_CTRL=kds` | pass: 172 replayed, 32 skipped |
| `tools_xrt_selftest_mock_1ch` | 1 channel (the k1 configuration), programs only | pass |
| `tools_xrt_selftest_mock_faults` | `test_xrt_selftest_faults.sh`: with `regs` and `kds`, a BRESP error in a DONE trace, an RRESP error in a DRAM_RANGE-ending trace, ADAPTER_STATUS left by an earlier process, a BRESP error in the END program → exit 1 with the failure named; the same runs without injection → exit 0 | pass (11 cases; the selftest before the fix exits 0 on the first injected case) |
| `tools_xrt_selftest_mock_m3`, `..._m3_tmax8` | `traces/m3/{tmax1,errors}`, `traces/m3/{tmax8,errors8}` (if present; T_MAX 8 with `kds`) | 132 / 132 and 100 / 100 |

Also run by hand: all 290 `traces/model` traces (Qwen2.5-0.5B Q8_0 and Q4_K_M requant) through
the mock with 4 KiB DMA alignment: 290 passed (0.98 s); two deliberately corrupted traces (one
expect byte, one ERR_PC) are reported as failures with exit code 1; `test-backend-ops test -o
MUL_MAT -b Tesseract0` (the `ggml_tesseract_backend_ops` case list) on `TSR_DEVICE=mock` with DMA
align 64 / chunk 256 KiB: 1402 / 1402 passed (59.8 s); `test-tsr-kdev`, `test-tsr-xrt-noop` and
the selftest (traces m2, gen, m3/tmax1 at T_MAX 1; m3/tmax8, m3/errors8 at T_MAX 8) under
AddressSanitizer + UBSan with leak detection: clean (the sanitizer run found one buffer overflow
in the test itself, fixed).

After the review fixes (2026-10-23, scratch builds with `TSR_WITH_XRT=ON`, with and without llama):
`transport_kdev`, `transport_xrt_noop`, `transport_ggml_e2e_mock`, `..._tmax8` and the six
`tools_xrt_selftest_mock*` ctests pass; `test-ggml-tesseract` on `TSR_DEVICE=mock` with
`TSR_MOCK_CTRL=kds`, T_MAX 8, align 64 / chunk 64 KiB: 28 / 28 cases OK. Not re-run: the whole-tree
ctest, the hand runs and the sanitizer runs below.

The whole tree with `TSR_WITH_XRT=ON`: every non-model ctest passes except `sim_trace`,
`sim_trace_ubsan` and `test-tokenizers-ggml-vocabs`, which fail identically on the tree before
this work (stale-region-file check in `sw/sim`, and a vocab download).

## 7. Building without an installed XRT

The 2.18.179 deb (sha256 `5e76278a…a688`, `sw/xrt-driver/README.md`) unpacks without
installing:

```
flock <scratchpad>/xrtlocal.lock sh -c 'test -d build/xrt-local/opt/xilinx/xrt || dpkg-deb -x build/xrt-driver/deb-dl/xrt_202420.2.18.179_24.04-amd64-xrt.deb build/xrt-local'
cmake -S . -B <build> -DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=build/xrt-local/opt/xilinx/xrt
```

The XRT headers include `<uuid/uuid.h>` (package `uuid-dev`, an XRT dependency, not installed on
this host). When CMake does not find it, `sw/transport/xrt-compat/uuid/uuid.h` declares the six
libuuid functions the headers call, and the system `libuuid.so.1` is linked by soname. With XRT
installed at `/opt/xilinx/xrt` (which pulls `uuid-dev`) the stand-in is not used. Binaries find
`libxrt_coreutil` through the build RPATH; XRT itself finds its driver shim through `XILINX_XRT`
or `/opt/xilinx/xrt`, so with only the unpacked deb `tsr_xrt_selftest` stops at "No such library
'/opt/xilinx/xrt/lib/libxrt_core.so.2'" (exit 2), and without `TSR_XCLBIN` at "TSR_XCLBIN is
not set". With `XILINX_XRT` pointing at the unpacked tree and `XCL_EMULATION_MODE=hw_emu`, XRT
loads its hw_emu shim from it instead (§9).

## 8. Card bring-up procedure, and what is untested

**v1 (2026-10-30, W0 of `docs/v2/README.md`).** This procedure is the v1 one, and the card has since
run it (M3 and M4, `docs/platform-u200.md` §14). The v2 bring-up is W6 of `docs/v2/README.md` §10
(identity, bind, load and VERIFY, HELLO, PING, launch round trip, `tsr-selftest`, GDB smoke).

1. Install XRT 2.18.179 and the patched DKMS driver (`sw/xrt-driver/README.md`), the U200
   platform, flash the shell (`xbmgmt`), `xbutil validate`. Build with `-DTSR_WITH_XRT=ON`.
2. M0 bandwidth (`tsr_bw`, separate work).
3. `TSR_XCLBIN=<k1.xclbin> bin/tsr_xrt_selftest`: identity, memory incl. unaligned and 4 GiB
   ranges, every program on one channel. No checked-in trace set fits NUM_CH 1 with a 4 MiB SPM.
4. `TSR_XCLBIN=<k4.xclbin> bin/tsr_xrt_selftest --traces traces/m3/tmax8 --traces traces/m3/errors8 --traces traces/m2 --traces traces/gen --traces traces/model`
   (T_MAX 1 traces replay on the T_MAX 8 core except T_RANGE endings).
5. `TSR_DEVICE=xrt bin/test-ggml-tesseract`, `test-backend-ops -b Tesseract0 -o MUL_MAT`, then the
   M3 gate: Qwen2.5-0.5B Q8_0 with `-ngl 99 -nkvo`.
6. If something fails: `TSR_XRT_CTRL=regs` (scheduler path), `TSR_XRT_DMA=staged` /
   `TSR_XRT_DMA_ALIGN=64` (DMA alignment), `TSR_XRT_TIMEOUT_MS` (long programs).

Untested (no card, no driver, no installed platform; hardware emulation of k1 ran, §9, and covers
part of the list below in simulation only): every real register access and DMA; the
XDMA alignment the shell accepts; DMA bandwidth; `xrt::run` with the scheduler / ERT on this
kernel, and window accesses while it runs; `TSR_XRT_CTRL=regs` on XRT; the `staged` fallback
and `m2m` on XRT; `load_xclbin` on a card (the noop test uses `register_xclbin`); the default
16 GiB device-only BOs (the noop test used 64 MiB banks); timeout / ABORT against real hardware;
interrupt mode; behaviour after `xbutil reset`; `settle` (the `AP_CTRL` read at open and after every
completion) next to the real scheduler / ERT — modelled on the mock with the scheduler's sequence
(`kds_*`, from the xocl / ERT sources), not run against them; the `m2m` staging allocation with a
full bank (the noop shim does not enforce bank capacity).

## 9. Hardware emulation (hw_emu) of `tsr_u200_k1` (2026-10-24)

The host binaries, unchanged and built with `-DTSR_WITH_XRT=ON` against the unpacked XRT (§7),
ran over `TSR_DEVICE=xrt` against the k1 kernel in Vitis hardware emulation: XRT's hw_emu shim
(`libxrt_hwemu.so`, in the unpacked deb), the extracted development platform, and xsim from Vivado
2024.2. This needed no root, no driver and no install. Everything below is simulation. Nothing ran on a card.

**xclbin.** `TSR_VIVADO_LOCK=<lock> PLATFORM_REPO_PATHS=$PWD/build/platforms/root/opt/xilinx/platforms
hw/u200/build_xclbin.sh k1 hw_emu` (the `.xo` manifest matched, so no repackaging): `v++` rc 0 after
2 min 17 s wall (project, block design, output products, then `config_hw_emu` compile and
elaborate; nothing is synthesized for hw_emu). The largest process reached 2.98 GB RSS and the
whole `v++` process tree at most 4.8 GB (5 s samples). Output: `tsr_u200_k1.xclbin`, 23.6 MB,
"HW Emulation Binary". In its memory topology bank1 (DDR[1], 0x800000000, 16 GiB) is the only bank
used. Warnings: `[v++ 60-889]` (`kernel_frequency=300` equals the default), 84 × `IP-LOCK-01` (the
platform's 2021.1 IP in the 2024.2 catalogue) and `[v++ 60-1455]` (debuggable symbols).

**Runtime environment** (environment variables only; nothing under `/opt/xilinx` is used):

```
set +u; source /tool/Xilinx/Vitis/2024.2/settings64.sh          # xsim, emconfigutil
export XILINX_XRT=$PWD/build/xrt-local/opt/xilinx/xrt LD_LIBRARY_PATH=$XILINX_XRT/lib
PLATFORM_REPO_PATHS=$PWD/build/platforms/root/opt/xilinx/platforms \
  emconfigutil --platform xilinx_u200_gen3x16_xdma_2_202110_1 --od <dir>
export EMCONFIG_PATH=<dir> XCL_EMULATION_MODE=hw_emu XRT_INI_PATH=<dir>/xrt.ini  # [Emulation] debug_mode=batch
export TSR_DEVICE=xrt TSR_XCLBIN=<k1 hw_emu xclbin> TSR_XRT_TIMEOUT_MS=0
```

`TSR_XRT_CH_GIB=3 TSR_XRT_DMA=staged TSR_XRT_DMA_ALIGN=4096`, the workarounds for the two simulator
defects below, are the xrt kdev's defaults under `XCL_EMULATION_MODE=hw_emu` (`xrt_env_defaults`,
tsr_kdev.cpp; ctest `transport_kdev`). Each one applies only when that variable is unset or empty,
so a value set in the environment wins. One line on stderr names the defaults that were applied:
`tsr_transport(xrt): XCL_EMULATION_MODE=hw_emu, defaults applied: TSR_XRT_CH_GIB=3 TSR_XRT_DMA=staged TSR_XRT_DMA_ALIGN=4096 (...)`.

The shim starts xsim in `.run/<pid>/` next to the executable. Run one emulation at a time under
`TSR_VIVADO_LOCK`.

**Two defects in the 2024.2 simulator models.** Both are in Xilinx code, both have an
environment-variable workaround (now the hw_emu defaults above), and neither is on the card's
path (XDMA, DDR4). The card, and every other emulation mode, keeps the defaults (`direct`, 16 GiB).
1. *A channel buffer of 4 GiB or more crashes the simulator.* The simulator backs a device-only
   BO with a file-mapped region (`.run/<pid>/hw_emu/device0/<n>/region_0x800000000_0x8bfffffff.mmap.dat`,
   created for its "ALLOC_BO p2p" call). The constructor of that storage,
   `xsc::mmapStorageEngine::mmapStorageEngine(unsigned int, std::string)` in
   `Vivado/2024.2/data/xsim/ip/xscl/libxscl.so`, takes the size as 32 bits: its disassembly does
   `mov %esi,%ebp`, then `mmap(NULL, size, ...)`. A 16 GiB BO therefore maps 0 bytes. The first
   DDR write then segfaults in `mmapStorageEngine::write`: `simulate.log` shows `FATAL_ERROR ...
   sim_qdma.cpp, line 106`, and `xsimcrash.log` has the stack. The host then spins forever in the
   shim's non-blocking socket read: the selftest hung in its first transfer (128 bytes at
   0x800000000) for 11 min until it was stopped, because no exec timeout applies to a transfer.
   3 GiB channels (and 1 GiB, in a probe) work. 4 GiB truncates to 0 bytes like 16 GiB.
2. *Host DMA through the simulator loses the tail of a transfer that starts off a 4 KiB boundary.*
   For [off, off + n), only ceil(n / 4096) pages counted from the first partial page arrive; the
   bytes in the last page are lost, for writes and reads alike. A probe with single-page transfers
   around it measured: 0xfbd + 0x1086 loses [0x2000, 0x2043), 0x800 + 0x1000 loses
   [0x1000, 0x1800); 0x10 + 0x2ff0, 0 + 0x2010 and 0x1000 + 0x1001 are intact. The host shim
   issues one call per range (HAL log), so the loss is in the simulator's host-DMA path (sim_qdma,
   then SmartConnect `transport_dbg`, then sim_ddr). With the default `direct` DMA the selftest
   fails exactly there: [0xffd, +4099) and [0x10011, +1048589) fail with "first difference at"
   0x2000 / 0x110000, and LD->ST with device 0x00 vs simulator 0xf1 at 0x4805000; the other 22
   checks pass. `TSR_XRT_DMA=staged TSR_XRT_DMA_ALIGN=4096` (§4.2) starts every raw transfer on a
   4 KiB boundary, and with it every range passes.

**Results** (`TSR_XRT_CH_GIB=3`, `staged` / 4096 unless noted):

| Run | Result |
| --- | --- |
| `tsr_xrt_selftest --no-mem --no-programs` (16 GiB, `direct`) | PASS, 8 checks: `load_xclbin`, kernel `tsr_u200_k1` (exclusive), 16 GiB device-only BO at 0x800000000 (memory group 1), ID "TSR1", ISA 1.0, NUM_CH 1, T_MAX 1, SPM 4 MiB, ADAPTER_VERSION 0x00010001, ADAPTER_STATUS clean; 30.5 s |
| `tsr_xrt_selftest --mem-mib 1 --traces <26 one-channel traces> --traces traces/m3/tmax1 --traces traces/gen/tmax1 --traces traces/m3/errors`, `TSR_XRT_CTRL=run` (`xrt::run` through XRT's hw_emu scheduler) | PASS, 44 checks: the memory ranges of a 3 GiB channel (no 4 GiB crossing); 9 programs bit-exact against the ISA simulator (END, ARG packing, LD->ST, QACT A32 + A256, MVM Q8_0 / Q4_K with and without ACC, DRAM_RANGE fault); 19 traces replayed (8 m3, 4 m2 and 7 error endings: ILLEGAL_OP × 3, BAD_FMT, T_RANGE × 2, SPM_RANGE); 163 skipped (156 NUM_CH 4, 6 with regions beyond 3 GiB, 1 DRAM_RANGE at 16 GiB); ADAPTER_STATUS clean throughout; 115.5 s |
| the same with `TSR_XRT_CTRL=regs` | PASS, 44 checks, the same 19 traces; PERF cycles of the 9 programs equal to the `run` mode; 105.5 s |
| `tsr_xrt_selftest --mem-mib 1` + 7 new one-channel traces, with none of `TSR_XRT_CH_GIB` / `TSR_XRT_DMA` / `TSR_XRT_DMA_ALIGN` set (the hw_emu defaults; same xclbin) | the defaults line on stderr; 3.00 GiB channel, `staged`, align 4096; PASS, 31 checks: 6 memory ranges, the 9 programs bit-exact with the PERF cycles of the rows above, 6 traces replayed, 1 skipped; 75.6 s. With `TSR_XRT_CH_GIB=1` set (`--no-mem --no-programs`), the 1 GiB value is used, only the two DMA defaults apply, PASS, 8 checks |
| `test-backend-ops test -o MUL_MAT -b Tesseract0`, 3 cases: q8_0 m 16 n 3 / 9 k 256, q4_K m 16 n 3 k 256 | 3 / 3 OK; 30.5 s; host process plus xsim at most 0.8 GB RSS |

The one-channel traces come from `tsr-trace-gen -q --num-ch 1` with these option sets:
`--ops m3 -n 6 -s 61 --max-wbeats 2048`; `--ops m3 -n 6 -s 65 --max-wbeats 2048`;
`--ops m2 -n 4 -s 62`; `--ops m3 -n 4 -s 63 --max-wbeats 1024 --errors 100`;
`--ops m2 -n 6 -s 64 --errors 100`. No checked-in set fits NUM_CH 1 (§8).

Speed: each exec of a short program (27–1,017 cycles) took about 1.3 s of wall time with `run`
and 0.7 s with `regs`. One 1-channel MVM trace of 136,818 cycles (ISA cycle model) added 45 s,
i.e. about 3 K cycles/s with the upload. That is roughly 100× slower than the Verilated kernel
(docs/platform-u200.md §13.1, 355 K cycles/s on k1). The host-to-card figure the selftest prints
(0.05 GB/s) is emulation time and says nothing about the card.

Not run in hw_emu: the k4 kernel; 16 GiB channels and the ranges across 4 GiB and at 12 GiB
(defect 1); `direct` DMA beyond the failing ranges (defect 2); `m2m`; DRAM_RANGE-ending traces
(they need the 16 GiB channel); timeout / ABORT; `test-ggml-tesseract` (546.7 M cycles on k1:
about two days at this speed) and model runs.

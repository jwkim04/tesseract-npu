# Host transport and the card (XRT)

This document describes the host side of the device interface: the transport `tsr_transport` (`sw/transport`),
the protocol layer that every kind shares, the `xrt` kdev that drives an Alveo card through XRT 2.18, the card
selftest, the tests that need no hardware and the card bring-up. The kernel on the other side is described in
[`platform-u200.md`](platform-u200.md). The launch protocol is [EX](design/01-execution-model.md) §2 to §4.
Normative: EX §9 and §10, [SW](design/06-software-verification.md) §9.1 and §9.2,
[DB](design/05-debug.md) §7 and §9. `sw/transport/README.md` is the developer summary of the same code.

## 1. Layers

<p align="center"><img src="fig/launch.svg" alt="Swimlane of a launch: llama.cpp, ggml-tesseract, tsr_transport, card DRAM and the control core. At model load the weights are written to DRAM and the firmware image is loaded and booted; each launch writes the inputs and the op list, pushes a descriptor into the launch queue, the control core runs the kernel and issues tile ops to the units, and the host polls the completion ring and reads the results back. A note names the execution modes: full mode runs every supported kind on the device with one launch per token, matmul mode runs MUL_MAT only." width="100%"></p>

```
ggml-tesseract, tsr_rt, tools         tsr_transport.h: open, close, config, write, read, csr_read / csr_write,
                                      image open, boot, launch, wait, call, cancel, stop, soft reset, log
  └─ tsr_transport.c                  dispatcher: TSR_DEVICE = sim | mock | vkernel | xrt | <registered kind>
       └─ protocol layer  tsr_kdev.cpp        one implementation for every kind (§3)
            │   uses tsr_image.cpp (firmware images) and tsr_logdec.cpp (log and TRAP decoder)
            └─ kdev  (tsr_kdev.h)             raw register and memory access to one kernel
                 ├─ "sim"      tsr_kdev_sim.cpp      the simulator in-process behind the core CSR port, no adapter
                 ├─ "mock"     tsr_kdev_mock.cpp     the sim kdev inside the ap_ctrl_none adapter register model
                 ├─ "vkernel"  tsr_kdev_vkernel.cpp  client of a Verilated kernel server
                 ├─ "xrt"      tsr_transport_xrt.cpp xrt::ip + one device-only BO per channel (-DTSR_WITH_XRT=ON)
                 └─ others     tsr::register_kdev()
```

The host does not run programs. It binds the channel memory, loads a firmware image into the control core's TCMs, boots
it and launches kernels (EX §2.1). A kdev offers `reg_read` and `reg_write` on `s_axi_control` (32-bit, offsets below
0x2000), `mem_write` and `mem_read` on a channel buffer, and a time base (`now_ns()` and `idle()`: it sleeps on `xrt`
and advances the model on the simulated kinds). `tsr::dma_engine` turns byte-granular requests into transfers that
obey a DMA path's rules (offset, size and host address multiples of `align`, at most `max_xfer` bytes each), with
read-modify-write of unaligned edges through a bounce buffer. `tsr::register_kdev("<kind>", factory)` before
`tsr_transport_open` makes `TSR_DEVICE=<kind>` work for the backend and every tool. The image loader parses the
firmware ELF with the `tsr-elf` library, checks the segments, the kernel table and `.note.tsr` (EX §3.1) and that the
stored image id equals the computed one (EX §3.2), and maps `mepc` to function and line (DB §6.3). The log decoder
decodes the device log words (DB §6.1) and TRAP records for `TSR_LOG` and `tsr-log`.

## 2. Kernel contract (software view)

The kernels `tsr_u200_k1` (NUM_CH 1, T_MAX 1) and `tsr_u200_k4` (NUM_CH 4, T_MAX 4) are `ap_ctrl_none` RTL
kernels with `ap_clk` and `ap_rst_n`, the control-core pair `ap_clk_2` and `ap_rst_n_2`, `s_axi_control`
(AXI4-Lite, 13-bit address space) and `m_axi_ch<c>` (AXI4, 512-bit) for c < NUM_CH. The full map is
[`platform-u200.md`](platform-u200.md) §2 and EX §9.2. What the software uses (byte offsets):

| Offset | Register | Software use |
| --- | --- | --- |
| 0x000-0x03C | reserved, read 0 | never accessed |
| 0x040 + 8c | `ch<c>_base` lo / hi | written at bind with the device address of the buffer of channel c (4 KiB aligned) while `MEM_EN` = 0 |
| 0x0E0 | `ADAPTER_CTRL` | bit0 `MEM_EN`, set after the bases. It must read back 1. |
| 0x0F0 | `ADAPTER_STATUS` | read and cleared (W1C) at open, read again on `PLAT_ERR`. Sticky bits 0-5 and a saturating count in [31:16]. |
| 0x0F4 | `ADAPTER_VERSION` | `{16'h0001, 8'h00, NUM_CH}`: major 1 and NUM_CH are checked at open |
| 0x1000 + off | core CSR window | every access of the launch protocol (EX §5), at any time |

The `mock` kdev models this map, including `MEM_EN`, `BASE_LOCKED`, `REQ_HELD` and `plat_err_i`. The `sim` kdev
has no adapter (`has_adapter()` is false), so the protocol skips `ADAPTER_STATUS` and the bind.

The `xrt` kdev finds the arguments by register offset, not by name, and refuses an xclbin whose kernel is
missing (it takes `TSR_XRT_KERNEL`, else `tsr_u200_k4`, else `tsr_u200_k1`), has more than one compute unit or an
`s_axi_control` range below 0x2000 (XRT's register access checks `offset + 4 <= range`), is neither
`ap_ctrl_none` nor, as EX §9.1's fallback, `ap_ctrl_hs` (never started), or has an argument outside `ch<c>` at
0x40 + 8c (8 bytes, with a memory connection, contiguous from c = 0, at most `TSR_NUM_CH_MAX` = 16).

## 3. Protocol layer (tsr_kdev.cpp)

One implementation serves every kind (SW-18). It follows EX §2.1, §3, §4.8-§4.9, §8.4-§8.6 and §10, as refined
by SW §9.1 and DB §4.1, §6.1, §6.3, §7.3-§7.4, §9.2.

- Open. Reads `ADAPTER_VERSION` (major 1, NUM_CH equal to the number of channel buffers), then `ID`
  (0x54535231, "TSR1"), `VERSION` (capability major 1), `CFG0`-`CFG3`, `CAPS`, `CTL_ID`, `RV_CFG`, `Q_DEPTH`,
  `RV_FREQ` (0 means "no `clk_rv`"), `LQ_CFG`, `PERF_INFO` and `DBG_ID`, and clears `ADAPTER_STATUS`. The host
  counters (`LQ_SEQ`, `LCQ_RD`, `LOG_RD`) and the `LQ_W` shadow are read from the device, because the rings keep
  counting across processes and only a card reset zeroes them. Unless `TSR_ATTACH=1` it then writes
  `SOFT_RESET`, waits for `SR_BUSY` = 0 (`TSR_RESET_MS`), which stops a control core left running by an earlier
  process and discards the records it left unread, and binds the channels. A debug configuration left armed
  (DB keeps it across `SOFT_RESET`) is reported, and every wait then reads `DBG_STATUS` at its backoff polls.
- Bind. `MEM_EN` = 0, the `ch<c>_base` registers, `MEM_EN` = 1, checked. The open fails with "bind failed" if
  `MEM_EN` does not read back 1, `ADAPTER_STATUS` shows a misaligned or locked base, or a base reads back
  different from the buffer's address. `ch_bytes` is the smallest bound buffer.
- Boot (`tsr_transport_boot`). Checks the image's `.note.tsr` requirements against the device, then VERIFY of
  every ITCM segment (CRC read back by the device), a cold load (`LOAD_ADDR`, `LOAD_DATA` words, receive CRC,
  VERIFY CRC) of each segment that does not match and of every DTCM segment, `RV_BOOT` = 0, `RV_RUN` = 1, and
  waits for HELLO (`TSR_BOOT_MS`) with the image id, the SDK ABI and the kernel count checked. A boot with no
  ITCM segment written is a warm restart (EX §3.5). A booted device is soft-reset first.
- Launch. Host-side credit: at most `LQ_DEPTH` (8) launches are outstanding, with no device read. The `LQ_W`
  words that differ from the shadow copy are written, then `LQ_PUSH` (posted writes, EX §4.8).
  `tsr_transport_call` is launch plus wait.
- Wait. Consumes the completion ring (an OK DONE or PONG of the oldest launch: record word r0, plus r2 unless
  the kernel is `graph` or `replay`; every word otherwise), delivers events to the callback and writes `LCQ_RD`.
  Poll policy: busy-poll while the elapsed time is below the last service time of the same kernel, then every
  max(10 us, 1 % of the longer of that time and the time waited so far) with `idle()` and the transport mutex
  released. The log is drained every poll with `TSR_LOG`, else every 10 ms.
- Timeout and recovery. `TSR_LAUNCH_TIMEOUT_MS` (default 60,000; 0 = none) counts only while `STATUS.RV_HALTED`
  = 0 and `DBG_STATUS.STOP_ACTIVE` = 0, and is off with `DBG_CFG.SERIAL` = 1. On a timeout the transport follows
  DB's recovery ladder (DB §7.3): a hang dump (`TSR_DUMP_DIR`), `UNIT_HUNG` ends in "card reset needed" (the
  device is FAILED), `LQ_CANCEL` covering every pushed launch with `TSR_ABORT_MS` for the answers, then
  `SOFT_RESET` and a warm restart. The wait returns `TSR_TE_TIMEOUT`.
- TRAP. The log is drained and decoded, the trap is reported (cause, `mepc` to function and line), and the
  device is soft-reset and warm-restarted, unless `TSR_GDB_WAIT` leaves it trapped for the debugger. The wait
  returns 0 with the TRAP record.
- PLAT_ERR. One report per episode, covering the launches pushed before it was seen, which answer `PLAT_ERR`.
  `ADAPTER_STATUS` is cleared once all of them have answered, and later launches answer normally.
- Cancel, stop, soft reset, log. `tsr_transport_cancel` and `_stop` write the `LQ_CANCEL` and `LQ_STOP` watermarks
  (EX §4.9), moved up behind the answered launches every 2^30 launches so that they never wrap onto new ones.
  `tsr_transport_soft_reset` is `CTRL.SOFT_RESET` with the wait (outstanding launches return `TSR_TE_LOST`).
- Memory and CSR. Memory is range-checked against `ch_bytes`. `tsr_transport_csr_read` goes to `0x1000 + off`, and a
  failed read returns 0xFFFFFFFF. A 64-bit PERF counter is read low half first (the core snapshots the high half).
- Time base and threads. On `sim`, `mock` and `vkernel` every timeout and the poll policy count device time (core
  cycles at a nominal 276.3 MHz), so results and recoveries do not depend on host load. On `xrt` they count host
  time. One mutex per transport serializes every call, and `tsr_transport_wait` releases it between polls.
- Errors. Calls return 0 or a negative `TSR_TE_*` code (`tsr_transport_strerror`): `INVAL` (-1), `IO` (-2),
  `TIMEOUT` (-3), `LOST` (-4), `DEVICE` (-5), `IMAGE` (-6), `STATE` (-7). A message goes to stderr, and a malformed
  environment value makes the open fail.

## 4. The xrt kdev and device memory

`sw/transport/tsr_transport_xrt.cpp`, built only with `-DTSR_WITH_XRT=ON` (§7). The protocol on top of it is §3.

### 4.1 Control

The kernel is `ap_ctrl_none`, so the kdev opens its compute unit with `xrt::ip`, which XRT opens with exclusive
access (EX §2.4, §9.1). Register access is `xrt::ip::read_register` and `write_register`, a load or store to an
mmap'ed BAR. There is no `xrt::kernel` and no `xrt::run`: XRT 2.18 refuses an `ap_ctrl_none` compute unit there
("AP_CTRL_NONE is only supported by XRT native API xrt::ip"), and the driver touches no control or interrupt
registers of such a unit, so a halted control core cannot confuse XRT's scheduler. EX §9.1's fallback,
`ap_ctrl_hs` with `interrupt="false"` that is never started, opens the same way. A second process cannot open
the compute unit while the first holds it.

The xclbin is loaded with `device.load_xclbin` (the Alveo shim of 2.18 runs the legacy flow). Only under XRT's
no-op shim (`XCL_EMULATION_MODE=noop`, §6) the kdev uses `register_xclbin`, the only flow that shim implements.
Kernel, compute unit and arguments are discovered from the xclbin (§2).

### 4.2 Buffers: one device-only BO per channel

Each channel gets one `xrt::bo(device, size, xrt::bo::flags::device_only, group)`, where `group` is the memory
index of the bank that argument `ch<c>` is connected to (`xrt::xclbin::arg::get_mems()`, EX §9.1). Size:
`TSR_XRT_CH_GIB` (1 to 16), default the bank size from the xclbin memory topology (16 GiB on the U200 DDR4
banks), shrunk in 256 MiB steps if the allocation fails (down to 1 GiB). From the XRT 2.18.179 sources
(`core/pcie/driver/linux/xocl`):

- A device-only BO has no host pages. `xocl_create_bo` reserves a `drm_mm` node in the bank (PAGE_SIZE
  alignment, so bases are 4 KiB aligned), so host RAM does not grow with the channel size.
- `xrt::bo::write` and `read` on such a BO are the ioctls `DRM_IOCTL_XOCL_PWRITE_UNMGD` and `PREAD_UNMGD`. The
  driver pins the caller's pages for the duration of the call, builds a one-time scatter list starting at the
  caller's byte offset and runs the XDMA engine on it: one transfer, no bounce copy, any offset (also beyond
  4 GiB), no root needed.
- The XDMA fast path waits at most 10 s per descriptor batch and a descriptor covers less than 256 MiB, so
  transfers are issued in chunks of `TSR_XRT_DMA_CHUNK_MIB` (default 64 MiB), which also bounds the pinned page
  list per call. Whether the shell's XDMA accepts unaligned host and card addresses is a property of the IP
  configuration, checked by the selftest's unaligned ranges (§5).
- `xrt::bo::copy` (M2M or KDMA) needs a host-backed staging BO and a second device-side copy. Whether the U200
  shell has M2M is a platform property (`xbutil examine -r platform`).

| `TSR_XRT_DMA` | Path | Host RAM | Use |
| --- | --- | --- | --- |
| `direct` (default) | device-only BO, `bo.write` and `bo.read` = unmanaged XDMA from the caller's memory, in chunks | the caller's buffer only | one PCIe transfer, no copy |
| `staged` | the same ioctls through an aligned bounce buffer (`TSR_XRT_DMA_CHUNK_MIB`), edges read-modify-written, `TSR_XRT_DMA_ALIGN` default 64 | one chunk | an XDMA that needs aligned addresses, or memory that cannot be pinned |
| `m2m` | host-backed staging BO per channel, `sync`, then `bo.copy` into the device-only BO | one chunk per channel | shells without unmanaged DMA; needs M2M or KDMA. The staging BO is allocated at open, before the channel BO. |

A failed `direct` transfer is retried once through `staged`. If that works the transport stays on `staged`
(one warning). The chunk, alignment and read-modify-write logic is `tsr::dma_engine`, tested on the mock (§6).

### 4.3 Environment

Read at open. A malformed value makes the open fail. These are the variables of the `xrt` kind. The other
transport variables are in [`sw/transport/README.md`](../sw/transport/README.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `TSR_XCLBIN` | required | xclbin to load. Loading the one already on the card is cheap. |
| `TSR_XRT_DEVICE` | `0` | device index or PCIe BDF |
| `TSR_XRT_KERNEL` | `tsr_u200_k4`, else `tsr_u200_k1` | kernel name |
| `TSR_XRT_CH_GIB` | bank size (16); 3 under `XCL_EMULATION_MODE=hw_emu` (§9) | channel buffer size in GiB |
| `TSR_XRT_DMA`, `TSR_XRT_DMA_CHUNK_MIB`, `TSR_XRT_DMA_ALIGN` | `direct`, 64, 1; `staged`, 64, 4096 under hw_emu (§9) | §4.2 |

With `TSR_XRT_CH_GIB` below 16 the core still range-checks against its `CH_BYTES` of 16 GiB, so a faulty launch
could address memory past the buffer. The backend never does: its heap ends at `ch_bytes`.

## 5. tsr-selftest

`tsr-selftest [--device KIND] [--image ELF] [--mem-mib N] [--no-mem] [--no-kernel] [--ids LIST] [--no-shadow]
[--rate-n N] [--traces DIR]... [--traces-serial DIR]... [--max-traces N] [--trw MIB] [--debug] [--log-out FILE]
[--seed N] [-v]` (`sw/tools/tsr_selftest.cpp`; SW §9.5, §7.6, SW-28). The default device is `$TSR_DEVICE`, else
`xrt`. `mock` and `sim` check the tool and the protocol path. Exit code 0 means every check passed, 1 a check failed,
2 the transport did not open or bad usage. Sections:

1. Identity: the kdev description and the identity, configuration and status registers, and the adapter (version,
   `ADAPTER_STATUS`, `MEM_EN`, bases equal to the buffers' device addresses) or "no adapter".
2. Memory per channel: pattern write and read-back at aligned, unaligned, 4 KiB- and 4 GiB-crossing and
   end-of-channel ranges with guard bytes, channel aliasing, and the bandwidth of an `--mem-mib` transfer.
3. Boot: the image (id, base kernels `graph`, `selftest` and `replay` at 0, 1 and 2), a boot with the receive CRC,
   VERIFY and HELLO, and a second boot that must be a warm restart.
4. PING: one round trip with the `ARG0` echo, then `--rate-n` sequential round trips and `--rate-n` pipelined
   launches. These are reported, not gated.
5. The `selftest` kernel (SW §7.6): its ids and five misuses, each run on the device and on a shadow simulator with
   the same image, inputs and arguments. The answers, every output byte, the per-unit completion counts, the kernel
   events and the log records must be equal.
6. `--traces DIR`: every trace below DIR replayed through the device (kind `ops` through the `replay` kernel, kind
   `launch` as full launches of their own image). `--traces-serial DIR` uses the `replay` kernel's `SERIAL` flag.
   Traces made for another configuration are skipped.
7. `--debug`: the DMI smoke (halt, read `dpc` and `dcsr`, single step, resume, PING after).
8. Log and status: every drained word decodes, no record is dropped, no sticky error in `STATUS`, the recovery
   counters are 0, `ADAPTER_STATUS` is clean at the end and no `PLAT_ERR` was seen.

`--log-out FILE` keeps every drained log word, raw, for `tsr-log`. `--trw MIB` runs the kernel tests and the trace
replays with the trace writer on (a ring of MIB MiB at the top of the channel 0 IO region) and runs the DB §8.7
self-checks against the PERF deltas of the whole run. A failing `selftest` kernel case lists every differing line
(`TSR_SELFTEST_DUMP=<dir>` keeps both images), and a failing trace replay says how many beats differ
(`TSR_DEVRUN_DUMP=<dir>` keeps the device's and the expected bytes).

## 6. Tests (no hardware)

`ctest --test-dir build/dev -N` lists them.

- `transport_kdev` (`test-tsr-kdev`): the protocol layer on the `mock` kdev. Identity checks and adapter faults at
  open, bind (`MEM_EN`, misaligned base, `BASE_LOCKED`, held requests), timeouts, a stale core at open (with and
  without `TSR_ATTACH`), `PLAT_ERR` during a launch, a hung unit, the registry, `TSR_MOCK_*` parsing,
  `xrt_env_defaults`, and the chunked and aligned transfer engine under DMA rules from align 1 to 4096.
- `transport_protocol` (`test-tsr-protocol`): one binary for every kind it can run (`sim`, `mock`, `vkernel` when a
  server is named). Cold and warm boot, image-id mismatch, `.note.tsr` refusal, VERIFY with an injected TCM fault,
  PING, 0-7 arguments, `BAD_DESC`, the credit rule, forced `LQ_OVERFLOW` and `RING_RD_BAD`, events, the log, cancel
  and stop, `SOFT_RESET` mid-launch, the debug-aware timeout, the recovery ladder and TRAP handling.
- `transport_xrt_noop` (XRT builds only): the `xrt` kdev on XRT's userspace with its no-op shim
  (`XCL_EMULATION_MODE=noop`, `libxrt_noop`; no card, no driver) and generated metadata-only xclbins of
  `tsr_u200_k4`. Xclbin load, kernel pick, arguments by offset, the memory bank of each channel argument,
  device-only buffers, the compute unit opened through `xrt::ip` for `ap_ctrl_none` and for the `ap_ctrl_hs` fallback,
  refusals with a reason, and register access and unmanaged DMA failing cleanly (the shim implements neither). It
  builds with `-DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=build/xrt-local/opt/xilinx/xrt` and `--target test-tsr-xrt-noop`.
- `transport_ggml_e2e_mock*`, `tools_selftest_mock*`, `tools_log_selftest`, `tools_trace_replay_device`: the ggml
  backend and `tsr-selftest` on the `mock` kdev (DMA rules that force the bounce path, `T_MAX` 8, one channel, the
  fault cases that must fail with the reason named, trace replays), `tsr-log` end to end, and `tsr-trace-replay
  --device`.

## 7. Building without an installed XRT

The 2.18.179 deb (`sw/xrt-driver/README.md` has its sha256) unpacks without installing:

```
test -d build/xrt-local/opt/xilinx/xrt || dpkg-deb -x build/xrt-driver/deb-dl/xrt_202420.2.18.179_24.04-amd64-xrt.deb build/xrt-local
cmake -S . -B <build> -DTSR_WITH_XRT=ON -DTSR_XRT_ROOT=build/xrt-local/opt/xilinx/xrt
```

The XRT headers include `<uuid/uuid.h>` (package `uuid-dev`). When CMake does not find it,
`sw/transport/xrt-compat/uuid/uuid.h` declares the six libuuid functions the headers call, and the system
`libuuid.so.1` is linked by soname. Binaries find `libxrt_coreutil` through the build RPATH. XRT itself finds its
driver shim through `XILINX_XRT` or `/opt/xilinx/xrt`, so with only the unpacked deb a program that opens a device
stops at "No such library '/opt/xilinx/xrt/lib/libxrt_core.so.2'" (exit 2). With `XILINX_XRT` pointing at the
unpacked tree and `XCL_EMULATION_MODE=hw_emu`, XRT loads its hw_emu shim from it (§9).

## 8. Card bring-up procedure

Prerequisites: XRT 2.18.179 with the patched driver ([`sw/xrt-driver`](../sw/xrt-driver/README.md)), the U200
development platform and the shell flashed with `xbmgmt`, and `xbutil validate` passing
([`platform-u200.md`](platform-u200.md)). The card is used through `xrt::ip` as an ordinary user. In order:

1. Load the release xclbin and run the card selftest: `TSR_XCLBIN=<tsr_u200_k4.xclbin> tsr-selftest --device xrt`.
   It checks identity, bind, cold load with VERIFY, HELLO, the PING round trip, the `selftest` kernel against
   the shadow, trace replays and the DMI smoke (§5).
2. Run the backend tests (`test-ggml-tesseract`, `test-backend-ops -o MUL_MAT -b Tesseract0`) with
   `TSR_DEVICE=xrt`. Then a model-level check: generate on the card with `klcheck run --device xrt` and replay
   the tokens on the simulator with `TSR_SIM_CARD=k4full TSR_SIM_GDN=1 TSR_SIM_TMAX=4`
   ([`klcheck.md`](klcheck.md)). Every row must be bit-identical.
3. A GDB smoke: halt, register read, breakpoint in a kernel, step, resume, the launch then completing
   (`tsr-dbg gdb --stdio --device xrt --elf build/xrt/fw/tsr-fw.elf --halt-on-boot`, [`prof-dbg.md`](prof-dbg.md) §1).

If something fails: `TSR_LOG=stderr` and `tsr-log` for the device log, `TSR_DUMP_DIR` for the hang dump of the
recovery ladder, `TSR_ATTACH=1` to open a device without resetting it, `TSR_XRT_DMA=staged` and
`TSR_XRT_DMA_ALIGN=64` for DMA alignment, `TSR_LAUNCH_TIMEOUT_MS` for long launches and `TSR_GDB_WAIT=launch` to
keep a trapped device. A card that XRT reports as stuck needs `xbutil reset`.

## 9. Hardware emulation (hw_emu)

The `xrt` kdev also runs against a Vitis hardware-emulation build, without root, a driver or a card, with XRT's hw_emu
shim (`libxrt_hwemu.so`, in the unpacked deb), the extracted development platform and xsim from Vivado 2024.2. The
host binaries are built as in §7. The xclbin comes from `TSR_VIVADO_LOCK=<lock>
PLATFORM_REPO_PATHS=$PWD/build/platforms/root/opt/xilinx/platforms hw/u200/build_xclbin.sh k1 hw_emu`. The runtime
environment is `source /tool/Xilinx/Vitis/2024.2/settings64.sh`, `XILINX_XRT=$PWD/build/xrt-local/opt/xilinx/xrt`
with `$XILINX_XRT/lib` on `LD_LIBRARY_PATH`, an `emconfigutil --platform xilinx_u200_gen3x16_xdma_2_202110_1 --od
<dir>` configuration (`EMCONFIG_PATH=<dir>`, `XRT_INI_PATH=<dir>/xrt.ini` with `[Emulation] debug_mode=batch`),
`XCL_EMULATION_MODE=hw_emu`, `TSR_DEVICE=xrt`, `TSR_XCLBIN=<k1 hw_emu xclbin>` and `TSR_LAUNCH_TIMEOUT_MS=0`. The
shim starts xsim in `.run/<pid>/` next to the executable, so run one emulation at a time under `TSR_VIVADO_LOCK`.
Hardware emulation is simulation, so a transfer rate printed under it says nothing about the card.

Under `XCL_EMULATION_MODE=hw_emu` the kdev defaults `TSR_XRT_CH_GIB=3`, `TSR_XRT_DMA=staged` and
`TSR_XRT_DMA_ALIGN=4096` (`xrt_env_defaults` in `tsr_kdev.cpp`, ctest `transport_kdev`). Each default applies only
when its variable is unset or empty, so a value set in the environment wins, and one line on stderr names the
defaults that were applied. They work around two defects in the 2024.2 simulator models. Both are in Xilinx code, and
neither is on the card's path (XDMA, DDR4).

1. A channel buffer of 4 GiB or more crashes the simulator. The simulator backs a device-only BO with a file-mapped
   region, and the constructor of that storage (`xsc::mmapStorageEngine::mmapStorageEngine(unsigned int,
   std::string)` in `libxscl.so`) takes the size as 32 bits. A 16 GiB BO therefore maps 0 bytes, the first DDR write
   segfaults (`simulate.log` shows `FATAL_ERROR ... sim_qdma.cpp`), and the host then spins forever in the shim's
   socket read. 3 GiB channels (and 1 GiB) work.
2. Host DMA through the simulator loses the tail of a transfer that starts off a 4 KiB boundary: for [off, off + n)
   only ceil(n / 4096) pages counted from the first partial page arrive, for writes and reads alike. With `direct`
   DMA the selftest's unaligned ranges fail there. `staged` with a 4096 alignment starts every raw transfer on a 4
   KiB boundary, and with it every range passes.

Not covered in hw_emu: the k4 kernel, 16 GiB channels and the ranges across 4 GiB and at 12 GiB, `direct` DMA
beyond the failing ranges, `m2m`, timeout and cancel.

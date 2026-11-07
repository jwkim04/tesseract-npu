# Profiling and debugging

This document says how Tesseract is debugged and profiled: what exists in the tree today, how the debug
design works, which measurements the profiler reports and which are measured on the card, how the U200 can
be reached for debugging. It is the working summary; the normative design is [DB](design/05-debug.md).

> **Scope.** Implementation status and design summary, not the contract; the contract is
> [DB](design/05-debug.md) with [EX](design/01-execution-model.md) §5-§6 (CSR map, PERF) and
> [CC](design/04-control-core.md) §3 and §9. Status words: *on the simulator* (implemented here and tested),
> *designed* (specified in `docs/design`, no implementation), *planned* (work not yet completed).
> The debug tools and W4 RTL exist; the integrated card gate is W6. Measured numbers carry their
> provenance; *estimate* means not measured. Card measurements were taken before the control core was
> added, with the host issuing the ops.

---

## 1. What exists today

**Implemented and tested on the simulator** (`TSR_DEVICE=sim` or `mock`; status checked 2026-11-02).
The control-core and debug RTL also exist, with module-level tests. The reduced image `k4m200`
(no vector unit, no attention, no dispatch history) ran on the card on 2026-11-06 and a real GDB session
worked on it (below). The full image has not been on the card.

| Capability | Where | Tests (`ctest -R`) |
|---|---|---|
| Debug-module model: the DMI profile of DB §9.1, `dmcontrol`, abstract register commands, the program buffer, system bus access with CC's refusals (control-window writes, TCM writes while the hart runs, the DM range), the hart's debug mode (halt request, `ebreakm`, step, two execute triggers) | `sw/sim/tsr_dm.c`, `sw/sim/include/tsr_dm.h` | `sim_dm`, `sim_dm_fw`, `sim_dm_ubsan` |
| DBG block model: the register block at core CSR 0x800-0xAFC, the stop logic (stop at a push index, `QSTEP`, release), serial mode, tile breakpoints on the dispatcher's enqueue tap, a 512-entry dispatch history with the pushing `pc`, unit probes (model values), the RISC-V snapshot, RV watchpoints, halt-on-boot, the DMI window gate (`DBG_CFG.DMI_EN`), DB's control-window registers, completion records (REC), the trace writer into a DRAM ring, markers, PERF slots 61-63 | `sw/sim/tsr_dbg.c`, `sw/sim/include/tsr_dbg.h` | `sim_dbg`, `sim_dbg_ubsan`, `sim_w1b_e2e` (a GDB-style DMI session on a firmware run leaves every observable unchanged) |
| Race detector (vector clocks over the units, the RISC-V and the host; `SPM_STALE`), schedules, determinism | `sw/sim/tsr_race.c`, `tsr_sched.c` | `sim_race`, `sim_race_rv`, `sim_sched_determinism` |
| Firmware trap, panic and log contract: `__tsr_crash` and `__tsr_dbg` records, the trap handler (TRAP log record and TRAP answer record, `RT_STATE` = TRAPPED), binary log records through the log ring | `sw/ctrl/rt/trap.cpp`, `rt/log.cpp`, `include/tsr/log.hpp` (`TSR_LOG`, `tsr::panic`) | `fw_runtime` |
| Transport: log drain and decoding with symbols and DWARF lines (`function+0x1c (file.cpp:123)`), the TRAP report, hang dumps, the recovery ladder (dump, `LQ_CANCEL`, `SOFT_RESET` and warm restart), debug-aware waits that do not count time while the hart is halted or a tile stop or serial mode is active | `sw/transport/tsr_kdev.cpp`, `tsr_logdec.cpp`, `tsr_image.cpp` | `transport_kdev`, `transport_protocol` |
| `tsr-log`: decodes saved log words with the firmware ELF; `tsr-selftest --debug`: a DMI smoke (halt, read `dpc` / `dcsr`, single step, resume, then a PING) | `sw/tools/tsr_log.cpp`, `tsr_selftest.cpp` | `tools_log_selftest`, `tools_selftest_mock` (the mock run uses `--debug`) |
| `libtsr_dbg`, standalone `tsr-dbg gdb`, and the embedded GDB service: registers, memory, software/hardware breakpoints, watchpoints, step, launch stops, tile commands, history, dumps and recovery | `sw/tools/dbg/`, `sw/tools/tsr_dbg.cpp` | `tools_dbg_lib`, `tools_dbg_rsp`, `tools_dbg_embed`, `tools_dbg_rsp_drop`, `tools_dbg_embed_drop`, `rt_sim_gdb_embed` |
| Real RISC-V GDB through `tsr-dbg gdb --stdio --device sim`: boot stop, symbol breakpoints, `stepi`, memory/CSR reads, kernel launch, backtrace, `finish`, history and detach | `sw/tools/dbg/tests/gdb_smoke.sh` | `tools_dbg_gdb`; passed with the pinned `riscv-none-elf-gdb`, not skipped |
| `tsr-prof trw` timeline export and checks, `tsr-prof stats`, extended `TSR_STATS` counters | `sw/tools/tsr_prof.cpp`, `sw/tools/dbg/trw.cpp`, `sw/transport/tsr_kdev.cpp` | `tools_prof_trw`, `tools_dbg_lib` |
| Trace capture: one trace directory per launch (`TSR_TRACE`), replay and conversion tools | `sw/sim/tsr_trace.c`, `sw/tools` | `sim_trace`, `trace_v2_roundtrip`, `tools_trace_gen_replay` |

Environment variables, as implemented (`grep` of `"TSR_..."` in `sw/transport` and `sw/sim`):

| Variable | Effect |
|---|---|
| `TSR_LOG=stderr\|<file>`, `TSR_LOG_LEVEL=0..5` | decode the device log there, drained at every poll (otherwise every 10 ms); the level written to `LOG_CFG.LEVEL` (0 FATAL ... 5 TRACE, default 3) |
| `TSR_STATS=1` | transport statistics at close (launches, launch wall time, CSR reads / writes, DRAM bytes, log words and drops, boot, timeouts, traps, restarts) and, in the ggml backend, the backend report at free: graph_compute calls, launches, op records per launch, OPL cache hits and misses, wall time in graph_compute and in launches, and the device's PERF counters over the same span |
| `TSR_DUMP_DIR=<dir>` | hang and post-mortem dumps (recovery ladder level 0) |
| `TSR_GDB=tcp:<port>\|unix:<path>` | start the embedded RSP service when the owning executable links `tsr_dbg_embed` |
| `TSR_DEBUGGER=sock:<path>` | start the embedded monitor-command socket |
| `TSR_GDB_WAIT=boot\|launch` | with the embedded service, halt at boot or kernel entry for GDB; also retain a trapped device instead of restarting it |
| `TSR_TILE_SERIAL=0\|1`, `TSR_TILE_HIST=0\|1` | write `DBG_CFG.SERIAL` / `HIST_EN` at open (needs the DBG block, present on the simulator) |
| `TSR_ATTACH=1` | open without `SOFT_RESET` or bind, for diagnosis |
| `TSR_SIM_LOG=stderr\|<file>` (optional prefix `rv,`) | simulator event log: pushes, starts, completions and RISC-V retire lines |
| `TSR_TRACE=<dir>` | one trace per launch |
| `TSR_SIM_RACE=off\|report\|fatal`, `TSR_SIM_SCHED=fixed\|random:<seed>[,min,max]` | race detector, schedules |
| `TSR_DEBUG=1\|2` | the ggml backend's lowering and placement log (not the device) |

An executable without the embedded-service registration reports that `TSR_GDB` / `TSR_DEBUGGER`
are unavailable. Automatic transport profiling via `TSR_PROF`, `TSR_PROF_TRW_MIB`, `TSR_PROF_EVERY`
and `TSR_PROF_RV_SAMPLE_US` remains inactive; use `tsr-prof trw` / `stats` explicitly. The `trw`
command itself reads `TSR_PROF_TRW_MIB` as its default ring size.

Model limits of the simulator's DBG block (`tsr_dbg.h`): the unit probes and the RISC-V snapshot are model
values; `DMI_RESP_ERR` and `RV_DFAULT` are never set; the apply of an RV watchpoint completes at once; and
the core has no progress watchdog, so `STATUS.UNIT_HUNG` never rises and `HUNG_STOP` has no trigger.

**Remaining verification and implementation.**

| Item | Work package | Notes |
|---|---|---|
| Two-clock vkernel and full-system debug/replay integration | W5 | Module-level `tb/dbg` and `tb/ctrl` tests exist; integrated RTL GDB remains unverified |
| Hang triage on the card, card measurements of the DMI cost | W6 | M6 gate (`docs/design/README.md` §2). The GDB smoke is done on `k4m200` (below) |
| `tsr-dbg diverge`, `tsr-prof report` and periodic firmware PC sampling | deferred | The CLI reports `diverge` as unimplemented; trace export and stats are available |

RTL limitations remain in the current interface: `RV_SAMPLE` / `RVWP_APPLY` writes acknowledge
before their CDC work completes, so callers can see a stale sample or unfinished apply. The
simulator completes these operations synchronously and does not establish RTL completion safety.
`DBG_STATUS.SBA_REFUSED` has no RTL setter, although bus refusals and DM `sberror` work.
On the card (`k4m200`, 2026-11-06): `tsr-selftest --debug` and a `riscv-none-elf-gdb` session through
`tsr-dbg gdb --stdio --device xrt --elf build/xrt/fw/tsr-fw.elf --halt-on-boot` worked: halt at boot, software and hardware
breakpoints (one in the `selftest` kernel, with a backtrace and `finish`), a hardware watchpoint, `stepi`, register and CSR
reads, memory write and read back, `monitor` commands, detach, and the launch answered. A stale `RV_SAMPLE` was not seen,
but while the hart is halted the sample is the debug ROM's park loop, so this does not prove the handshake; `SBA_REFUSED`
was not exercised. `monitor hist` is empty in this image (`HIST_DEPTH` 0).
See [handoff.md](handoff.md) for the active link and the remaining card gates.

---

## 2. The debugging design

The design in one page; DB has the registers, the state machines and the decisions (DB-1 to DB-23).

<p align="center"><img src="fig/debug.svg" alt="On the host, GDB talks the remote serial protocol to a debug service in the process that owns the device, or to the standalone tsr-dbg; the service and the profiling and log tools use tsr_transport, which reaches the core CSR window over PCIe. In the window: the DMI at 0xC00 leads to the RISC-V debug module, the Ibex control core (halt, step, breakpoints) and the system bus access to the TCMs and the SPM; the DBG block at 0x800 to 0xAFC holds the stop logic for the queue heads, the 512-entry dispatch history and the unit probes; the log ring, the performance counters and the trace writer to a DRAM ring serve observation. Simulator tools and module RTL exist; integrated card verification remains." width="100%"></p>

The figure is the same design as the diagram below, drawn as the host and the device see it;
§1 separates the implemented tools and RTL from the remaining integration gates.

```
 GDB (RSP over TCP / Unix socket / stdio)
   |
 debug service in the process that owns the device (libtsr_dbg), or tsr-dbg
   |        |                        |
   | DMI ops| DBG / STATUS / PERF    | XRT buffer reads
   v        v                        v
 core CSR window:  0xC00-0xDFC DMI  ->  riscv-dbg debug module  ->  Ibex (halt, step, triggers)
                                              +-- system bus access -> TCMs, SPM, windows
                   0x800-0xAFC DBG block: stop logic -> queue-head holds, history ring, unit probes,
                                          trace writer -> DRAM ring
                   0x400-0x7FC log ring, PERF (EX §6) incl. DB slots 61-63
```

**RISC-V debug (DB §3, §4).** The debug module is riscv-dbg (Debug Spec 0.13.2) in the `clk_rv` domain; the
host reaches its registers through the DMI window of the core CSR port (CSR 0xC00 + 4r; U200
`s_axi_control` 0x1C00 + 4r), enabled only by `DBG_CFG.DMI_EN` so that generic CSR dumps stay side-effect
free. A GDB RSP server runs inside the process that owns the device (`xrt::ip` opens the compute unit
exclusively) and drives DMI directly: no OpenOCD and no software JTAG TAP. Memory access goes through the
connected system bus master (SBA), which can read memory without halting the hart; SBA writes to the
control window, to the core window and to the TCMs while the hart runs are refused (D15). The debugger
offers two hardware breakpoints (Ibex execute triggers), unlimited software breakpoints (ITCM is
writable; `fence.i` follows), two data watchpoints, and halt at boot or at kernel entry
(`TSR_GDB_WAIT=boot|launch`). `RV_HALTED` is derived from the instruction-fetch address lying in the debug
module's memory range. `HOLD_ON_HALT` holds the queue heads while the hart is halted, so that a halted
hart and a quiescent datapath are the same observable state.

**Tile-level debug (DB §5).** A tile op has no instruction boundary: up to five units run concurrently
(DMA, QACT, MVM, VU, SYNC). The comparable stop point is the **push-order prefix**: every entry with
push index below T has completed and no entry at or above T has started. The DBG block reaches it by
holding every queue head whose push index is at least T, so a stop costs no dispatcher gate. On top of it:
`STOP_AT` a push index, `QSTEP` op by op, **serial mode** (one op in flight at a time, in push order;
the launch timeout is disabled), tile breakpoints on opcodes, push indices, tickets and SPM or DRAM
ranges (`BEFORE` / `AFTER`, optionally halting the RISC-V), a 512-entry dispatch history that records the
pushing C++ line (`file:line` through the ELF's DWARF), unit probes (head state, age, outstanding reads
per channel) and a RISC-V snapshot from Ibex's crash-dump port. Tile debug changes nothing in the program:
its state is evidence and survives `QABORT`, `QCLEAR`, the RISC-V reset and `SOFT_RESET`. The invariant that
ties the two layers together: **a halted core with `HOLD_ON_HALT` and a drained datapath equals the
simulator after the same number of retired tile ops** (and, for the RISC-V, the same retired
instructions given the co-simulation log; [`ctrl-riscv.md`](ctrl-riscv.md) §4.6). `tsr-dbg diverge`
uses it to find the first wrong op of a launch by comparing history entries, written SPM lines and DRAM
regions with the simulator at checkpoints in serial mode (DB §9.4).

**Log, crash, trap (DB §6; on the simulator).** Firmware text is binary: a 32-bit header (`N`, `LEVEL`,
`SOURCE`, `TYPE`, timestamp flag, launch sequence, sync nibble 0xA) and, for FMT records, the address of a
format string in the non-loaded section `.tsr_logfmt` plus 32-bit arguments, so no string occupies a TCM.
Records go through the 256-word core-domain log ring (readable even when the queues are wedged), are never
torn, and are dropped whole with a counter when the ring is full (an EVENT "dropped n" follows). A trap
fills `__tsr_crash` (44 words: `mcause`, `mepc`, `mtval`, `mstatus`, MTIME, launch, kernel, `RT_STATE`,
`RT_INFO`, `x1`-`x31`), pushes a FATAL TRAP record, posts a TRAP answer to the host and parks in a WFI loop;
the transport prints one report line (`TRAP: <cause> (mcause n) at function+0x1c (file.cpp:123), mtval ...;
RT_STATE <phase> kernel <index> (<name>), RT_INFO ..., MTIME ...`, with the log tail before the trap) and
restarts the device unless `TSR_GDB_WAIT` is set.

**Hang triage and recovery (DB §7).** Every DBG register is a core-domain register mux, so a read never
waits on a unit, on `clk_rv` or on memory (the XRT shell's AXI firewall cuts a control slave that does not
answer, and only a card reset recovers it). `tsr-dbg status` would apply the triage table of DB §7.2
(spinning on a join, on credits, a push without credit, WFI, a stalled channel read, a hung unit, a held
head, an absent `clk_rv`, a tripped firewall). The recovery ladder, which the transport's timeout path
already follows: level 0 hang dump (`TSR_DUMP_DIR`), level 1 debug release (`RELEASE`, serial off, holds
off, hart resumed; P3.7), level 2 `LQ_CANCEL` covering the outstanding launches, level 3 `SOFT_RESET` and a
warm restart (the TCM image stays and is verified), level 4 a card reset (`xrt-smi reset`). A hung unit
cannot finish `SOFT_RESET`; there is no unit reset (CC-22, EX §8.6).

---

## 3. Profiling

**Software first.** The PERF space (EX §6, normative) is gated by `ACTIVE` (a launch is in progress) and
`QHALT` (the hart is halted and the datapath quiescent), so PERF does not move between launches and the
difference of two reads is one launch. DB adds three slots at the top of the range: 61 `TRW_RECORDS`,
62 `TRW_DROPS`, 63 `DBG_STOP_CYCLES` (not gated by `QHALT`; a measurement is valid only if it is 0).
Derived quantities need no new counters: DDR idle of channel c is `CYCLES - CH_RD_BEATS[c] - CH_WAIT[c]`;
issue-bound MVM time is `MVM_EMPTY`, dependency-bound `MVM_WAIT`.

**The utilisation split** (`docs/spec.md` §14): the DDR utilisation `U` is the bytes read per token times
tokens per second over the M0 bandwidth, and `U` is about `(1 - H)(1 - G)(1 - F) E`.

| Term | Meaning | Measured from |
|---|---|---|
| H | share of wall-clock time in which no launch is active (host CPU work, PCIe round trips, synchronisation) | host time stamps and `PERF_CYCLES` |
| G | share of device time outside MVMs (LD, QACT, VU, ST, issue gaps) | `1 - PERF_MVM_CYCLES / PERF_CYCLES` |
| F | MVM fill, broadcast and tail inside `PERF_MVM_CYCLES` | max-channel weight beats against `PERF_MVM_CYCLES` |
| E | effective read bandwidth while the MVM reads weights, over the M0 bandwidth | `PERF_CH_RD_BEATS`, `PERF_CH_WAIT` |

Further metrics: the effective bandwidth of channel c is 64 B x `CH_RD_BEATS[c]` x f_kernel / `CYCLES`;
the wait ratio `CH_WAIT[c] / CYCLES`; the channel imbalance is the maximum over the mean of the beats;
MVM cycles per beat is `MVM_CYCLES` over the maximum over channels of rows x `ROW_BEATS` (1.00 for large MVMs
on ideal memory, [`rtl-m3.md`](rtl-m3.md) §13).

**Measured on the card (k4, 276.3 MHz; decode, 65-token minus 1-token run, `TSR_STATS`;
[`platform-u200.md`](platform-u200.md) §14.2):**

| Per decode token | Gemma 4 12B Q8_0 | Qwen2.5-32B Q4_K_M |
|---|---|---|
| wall time | 221.9 ms | 340.6 ms |
| device busy (launch in progress) | 184.5 ms (83.2 %) | 298.8 ms (87.7 %) |
| exec start and wait (36 µs and 35 µs each) | 10.1 ms (4.6 %) | 13.6 ms (4.0 %) |
| backend host work | 6.3 ms (2.8 %) | 8.5 ms (2.5 %) |
| outside the backend (CPU ops, scheduling, sampling) | 20.9 ms (9.4 %) | 19.7 ms (5.8 %) |
| host total (H) | 37.3 ms (16.8 %) | 41.8 ms (12.3 %) |
| `PERF_CYCLES` per token | 50.98 M | 82.56 M |
| `PERF_MVM_CYCLES` | 49.64 M (97.4 %), so G is 2.6 % | 81.11 M (98.2 %), so G is 1.8 % |
| read beats per channel | 49.55 M (97.2 % of cycles) | 80.97 M (98.1 %) |
| bytes read; effective read rate | 12.69 GB; 68.8 GB/s (17.2 GB/s per channel) | 20.74 GB; 69.4 GB/s (17.4 GB/s per channel) |
| `PERF_CH_WAIT` per channel | 37 K cycles (0.07 %) | 57 K cycles (0.07 %) |

G here is derived from the two counters. An exec was one `MUL_MAT`, 281 and 385 device round trips per
token. At 276 MHz the weight stream's limit is 64 B per cycle (17.7 GB/s per
channel), slightly below the 18.0 GB/s that the DRAM gives at 300 MHz (the M0 result, [`m0-bw.md`](m0-bw.md)
§7.1), so decode is bound by the weight stream; closing 300 MHz would add about 4-5 % (*estimate*,
[`platform-u200.md`](platform-u200.md) §14.2). The first optimisation priority of the spec is to keep the
DDR busy: remove host round trips (H) first, then overlap non-MVM ops with MVMs (G), then fill and tail (F),
then DDR efficiency (E). The target is `U` of at least 85 % for 32B Q4_K_M decode at M9 (target, not
measured).

**Timelines (designed; P3.8).** Completion records to the RISC-V (REC, 64-entry FIFO) serve firmware-level
profiling. The host timeline comes from the **hardware trace writer** (TRW, DB §8.3): 32-byte records
(per tile op the push index, ticket, end time and the times `RUN`, `QWAIT` and `HWAIT`; plus `MARK`, `STOP`,
`RVHALT` and `DROP` records) written by a second write source of channel 0's `tsr_mem_arb` into a DRAM
ring (default 64 MiB), with markers around every kernel launch (`LAUNCH_BEGIN` / `LAUNCH_END`) and every
op-list node (`NODE_BEGIN` / `NODE_END`).
`tsr-prof` with source `trw` produces Perfetto JSON: process `Tesseract0`, one thread per unit, a RISC-V
thread (launches, nodes, PC samples), counter tracks per channel; reports per node, layer and unit and the
line "DDR idle while active". The self-checks of DB §8.7 (retired ops, per-unit busy sums, per-channel
beats) run on every profiled launch. The model of the trace writer exists on the simulator
(`sw/sim/tsr_dbg.c`); the tool and the RTL do not. Milestone tok/s is always measured with the profiler off.

**`TSR_STATS`** is partly there: the transport and the backend print the quantities listed in §1; the
per-channel read rates, idle shares, `MVM_WAIT` / `MVM_EMPTY` shares and `RV_WFI` percentage arrive with
P3.8. The cost of reading about 16 counters per launch is 16-32 µs (*estimate*, DB §8.6).

---

## 4. Access paths and JTAG

The register meaning is the same on every target; debug access always enters through the core's CSR port.
The core has no JTAG port, and vendor primitives stay in `rtl/tech/`.

| Target | DBG, DMI and PERF access | JTAG | Signal-level debug |
|---|---|---|---|
| `sim`, `mock` | function calls (the simulator is the reference model) | none | none |
| `vkernel` (Verilator, P5.3) | the adapter window | the DTM test (planned) | waveforms |
| U200 (XRT) | PCIe CSR window, `s_axi_control` 0x1000 + offset | none | ILA in debug builds, XVC over PCIe |
| Zynq / Kria (idea) | PS AXI-Lite (UIO) | optional BSCANE2 to CSR bridge | ILA over XVC or the board's USB JTAG |
| ASIC, boards without a host bus (idea) | JTAG TAP + DTM to the CSR port | main path | none |

### 4.1 U200

**No JTAG to user logic.**

- The XRT shell is a DFX design: the user kernel lives in the reconfigurable region, and configuration
  primitives such as BSCANE2 can be placed only in the static shell (UG909's DFX rules; to be confirmed on
  this shell).
- The shell's **user** debug network (an AXI-to-BSCAN debug bridge at PF1 BAR0 0x1C00000) is reached only
  by XVC over PCIe (xocl `xvc_pub`) and has no USB-JTAG fallback (`supports_jtag_fallback = false` in the
  platform's `ext_metadata.json`). The **management** debug network does have a fallback, but it does not
  reach the user kernel.
- So JTAG would ride the same PCIe link, slower than the CSR window, and show nothing new. No BSCANE2, no
  JTAG-to-CSR bridge and no JTAG-to-AXI master are built for the U200; architectural debugging uses the
  debug module and the DBG block through the CSR window.

**ILA, debug builds only.**

- For what the debug block cannot see: AXI deadlocks, firewall trips, CDC, cycles inside an op. The
  trigger is the internal net `dbg_stop_evt` of the DBG block (DB §5.2).
- Built from a separate link configuration (`hw/u200/k4_dbg.cfg`, to be written; the regular `k1.cfg` /
  `k4.cfg` builds carry no ILA and have no `[debug]` or `[profile]` sections).
- Interfaces: `v++` `[debug]` (`chipscope=tsr_k4:m_axi_ch0` ... `m_axi_ch3`, `chipscope=tsr_k4:s_axi_control`,
  `protocol=all` for System ILA and the protocol checker). Core-internal signals: Vivado Tcl hooks in the
  debug configuration's `[vivado]` section mark the chosen nets `MARK_DEBUG` and create and connect the ILA
  on the synthesized netlist (`create_debug_core`, `connect_debug_port`). No ILA instance and no debug
  attribute goes into `rtl/` outside `rtl/tech/` (spec rule 6). Which nets survive synthesis and the exact
  hook step are settled with the first debug build.
- Connect: `debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u<dev>.<idx> --hw_server` (xocl names its nodes
  `xfpga/<subdev>.u<device id>.<index>`), then `debug_hw --vivado --ltx_file <file>.ltx` (Vitis 2024.2).
- Division of labour: the debug module and the DBG block for wrong values, which op, where a hang sits;
  the ILA for bus-level and sub-op problems.

### 4.2 Zynq / Kria (idea)

Main path: PS AXI-Lite (UIO mmap). On a flat design, `rtl/tech/xilinx/tsr_jtag_bscan.sv` (BSCANE2 on USER3 /
USER4, clear of the debug hub's default USER1) can bridge JTAG to the CSR port: a backup when Linux is dead,
and a check of the ASIC DTM on an FPGA before silicon. Whether the Kria shell (dfx-mgr) is flat or DFX is
checked when the port starts. ILA over XVC on Ethernet or the carrier's USB JTAG.

### 4.3 ASIC and boards without a host bus (idea)

- JTAG TAP + DTM is the main path, in the RISC-V debug specification's DTM format: IR 5 bits, IDCODE 0x01,
  `dtmcs` 0x10, `dmi` 0x11, BYPASS 0x1F. A real DTM's `dmi` address is the CSR offset divided by 4, so the
  debug module registers sit at DMI address 0x300 (0xC00 / 4); a standard debugger such as OpenOCD would
  take `-dbgbase 0x300` (*confirm*). The same register meaning holds everywhere.
- A platform-layer arbiter (`tsr_csr_arb`) arbitrates the host bus (if any) and JTAG onto the one core CSR
  port; the CDC is a handshake, with TCK well below F_CORE. pulp riscv-dbg's `dmi_jtag_tap` and `dmi_jtag`
  (Apache-2.0) are reuse candidates.
- Without a host bus, a boot ROM is needed (the TCMs are undefined after reset; [`ctrl-riscv.md`](ctrl-riscv.md)
  §3.4) and a debug memory port is an open item.

### 4.4 Several CSR masters

Only the master that owns execution writes `CTRL`, the launch queue and the loader and reads the PERF
counters: on the U200 and Zynq the host bus; on a board without a host bus the JTAG DTM. Any other master
reads only the registers that have no side effect. Reason: a PERF low-half read moves the single snapshot
slot, so the owner's next high-half read would return the live value and a 64-bit value could tear.
Inside the host, one thread under the transport's mutex reads PERF. Per-master snapshot slots are an open
item for targets where JTAG and a host bus are used at once.

---

## 5. Design reference

The design is specified in DB ([`docs/design/05-debug.md`](design/05-debug.md)). §1 lists what exists and what is
planned, and §2 summarises the design.

---

## 6. Deliberately not done, and open items

**Deliberately not done**

- **A debug SPM window and a halt-on-error path**: the debugger reaches the SPM through SBA,
  and error evidence is the unit error records, the dispatch history and the trace writer.
- **Instruction patching for breakpoints in the tile stream**: tile breakpoints are matchers on the
  enqueue tap, not instruction rewrites.
- **Halting inside a tile op** (mid-MVM): not an observation point of the push-order semantics; inside an
  op, use a repro trace with waveforms, or the ILA.
- **On the U200**: an own BSCANE2 or JTAG bridge, and an ILA in the regular build.
- **llama.cpp**: no patch to the pinned commit for profiling; no `cb_eval` node stops or timing, because
  they cut splits and change what is measured (numcheck keeps its own `cb_eval` use for numerics).
- **Existing definitions**: PERF indices of EX §6, `CAPS` bits and error codes stay as defined; no
  always-on RTL instrumentation beyond DB's configurable blocks (`HAS_TRW`, `HIST_DEPTH` are the knobs that
  trade area; DB §11).

**Open items**

- PCIe MMIO latency per CSR read and write is not measured. It sets the cost of the DMI path (DB §3.10,
  *estimate*), of `RV_SAMPLE` and of PERF reads per launch; the card measurement is W6.
- Whether a new process can open a compute unit left by a dead one without disturbing it (hang dumps of a
  hung process; `TSR_ATTACH`): to be checked on the card.
- The U200 debug build (`k4_dbg.cfg`): which nets survive synthesis and the exact hook step.
- Profiling on the card: the full image (`k4full`) has the trace writer and the history. `tsr-selftest --trw` ran the writer through the
  kernel tests and 647 trace replays (16,247 records, none dropped, no hang on channel 0), and the GDB smoke (halt, breakpoints, a hardware
  breakpoint in a kernel, `monitor hist`, `monitor trw`) passed on `k4full`. Open: the DB §8.7 self-check of records against PERF differs by
  0.2 to 5 % on the card (`platform-u200.md` §8.8); a timeline of a model run (`tsr-prof trw` of a `graph` launch) was not made.
- Complete the RTL sample/apply handshake and the `SBA_REFUSED` diagnostic described in §1.
- `tsr-dbg diverge`, `tsr-prof report` and periodic firmware PC sampling.
- Confirmations of third-party facts: riscv-dbg and Ibex behaviours marked *confirm* in CC §3 and DB §3.
- Per-master PERF snapshot slots for targets with two CSR masters (§4.4).

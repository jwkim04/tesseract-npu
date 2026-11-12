# Profiling and debugging

This document says how Tesseract is debugged and profiled: which tools exist, how the debug design works, what the
profiler reports, how the U200 can be reached for debugging, and how to watch a running card with `tsr-top`. The
normative design is [DB](design/05-debug.md) with [EX](design/01-execution-model.md) §5 and §6 (CSR map, PERF) and
[CC](design/04-control-core.md) §3 and §9.

## 1. What exists

The tools run on the simulator (`TSR_DEVICE=sim` or `mock`) and on the card through the `xrt` transport (`--device
xrt`).

| Capability | Where | Tests (`ctest -R`) |
|---|---|---|
| Debug-module model: the DMI profile of DB §9.1, `dmcontrol`, abstract register commands, the program buffer, system bus access with CC's refusals, the hart's debug mode (halt request, `ebreakm`, step, two execute triggers) | `sw/sim/tsr_dm.c` | `sim_dm`, `sim_dm_fw`, `sim_dm_ubsan` |
| DBG block model: the register block at core CSR 0x800-0xAFC, the stop logic (stop at a push index, `QSTEP`, release), serial mode, tile breakpoints, a 512-entry dispatch history with the pushing `pc`, unit probes, the RISC-V snapshot, RV watchpoints, halt-on-boot, the DMI window gate (`DBG_CFG.DMI_EN`), completion records (REC), the trace writer into a DRAM ring, markers, PERF slots 61-63 | `sw/sim/tsr_dbg.c` | `sim_dbg`, `sim_dbg_ubsan`, `sim_w1b_e2e` |
| Race detector (vector clocks over the units, the RISC-V and the host; `SPM_STALE`), schedules, determinism | `sw/sim/tsr_race.c`, `tsr_sched.c` | `sim_race`, `sim_race_rv`, `sim_sched_determinism` |
| Firmware trap, panic and log contract: `__tsr_crash` and `__tsr_dbg` records, the trap handler (TRAP log record and answer, `RT_STATE` = TRAPPED), binary log records through the log ring | `sw/ctrl/rt/trap.cpp`, `rt/log.cpp` | `fw_runtime` |
| Transport: log drain and decoding with symbols and DWARF lines, the TRAP report, hang dumps, the recovery ladder, debug-aware waits that do not count time while the hart is halted or a tile stop or serial mode is active | `sw/transport/` | `transport_kdev`, `transport_protocol` |
| `tsr-log` decodes saved log words with the firmware ELF. `tsr-selftest --debug` runs a DMI smoke (halt, read `dpc` and `dcsr`, single step, resume, PING). | `sw/tools/tsr_log.cpp`, `tsr_selftest.cpp` | `tools_log_selftest`, `tools_selftest_mock` |
| `libtsr_dbg`, the standalone `tsr-dbg gdb` and the embedded GDB service: registers, memory, software and hardware breakpoints, watchpoints, step, launch stops, tile commands, history, dumps and recovery. Real RISC-V GDB works through `tsr-dbg gdb --stdio --device sim`. | `sw/tools/dbg/`, `sw/tools/tsr_dbg.cpp` | `tools_dbg_*`, `rt_sim_gdb_embed` (`tools_dbg_gdb` needs the pinned `riscv-none-elf-gdb`) |
| `tsr-prof trw` (timeline export and checks), `tsr-prof stats`, the `TSR_STATS` counters | `sw/tools/tsr_prof.cpp`, `sw/tools/dbg/trw.cpp` | `tools_prof_trw`, `tools_dbg_lib` |
| The profile report of one decode token (§3.1) | `sw/ggml-tesseract/tsr-prof.cpp`, `sw/tools/prof/` | `tools_prof_report_*` |
| Trace capture (one trace directory per launch, `TSR_TRACE`), replay and conversion tools | `sw/sim/tsr_trace.c`, `sw/tools` | `sim_trace`, `trace_v2_roundtrip`, `tools_trace_gen_replay` |
| `tsr-top`, the live monitor of the card (§7) | `sw/tools/top/`, `sw/transport/tsr_mon.*` | `tools_top_*`, `transport_mon` |

The transport's debug variables (`TSR_LOG`, `TSR_LOG_LEVEL`, `TSR_STATS`, `TSR_DUMP_DIR`, `TSR_GDB_WAIT`,
`TSR_TILE_SERIAL`, `TSR_TILE_HIST`, `TSR_ATTACH`, `TSR_MON*`) are in [`sw/transport/README.md`](../sw/transport/README.md).
The others:

| Variable | Effect |
|---|---|
| `TSR_GDB=tcp:<port>\|unix:<path>` | start the embedded RSP service when the owning executable links `tsr_dbg_embed` (otherwise the transport reports that it is unavailable) |
| `TSR_DEBUGGER=sock:<path>` | start the embedded monitor-command socket |
| `TSR_DUMP_PLAN=<dir>`, `TSR_PROF_TRW_MIB=<n>` | the ggml backend writes its plans, launches, refusals and the trace writer's records of the whole run into `<dir>` (§3.1), starting the ring itself (`n` MiB, default 32). No record and no output bit changes. |
| `TSR_OPL_PROF=1`, `TSR_OPL_KINDS=<file>` | the backend sets the launch descriptor's PROF bit (a marker before every record) and appends the kind of every record of every launch to the file |
| `TSR_TRACE`, `TSR_SIM_LOG`, `TSR_SIM_RACE`, `TSR_SIM_SCHED`, `TSR_SIM_VU_TIMING`, `TSR_SIM_CARD` | trace capture, the simulator's event log, race detector, schedules and cycle model ([`sw/sim/README.md`](../sw/sim/README.md)) |
| `TSR_DEBUG=1\|2` | the ggml backend's lowering and placement log (not the device) |

Automatic transport profiling through `TSR_PROF`, `TSR_PROF_EVERY` and `TSR_PROF_RV_SAMPLE_US` is not wired in. Use
`tsr-prof trw` and `tsr-prof stats` explicitly (`trw` reads `TSR_PROF_TRW_MIB` as its default ring size).

Limits. The simulator's DBG block has model values for the unit probes and the RISC-V snapshot, never sets
`DMI_RESP_ERR` and `RV_DFAULT`, and has no progress watchdog, so `STATUS.UNIT_HUNG` never rises. In the RTL,
`RV_SAMPLE` and `RVWP_APPLY` writes acknowledge before their CDC work completes, so a caller can see a stale sample,
and `DBG_STATUS.SBA_REFUSED` has no setter (bus refusals and the DM `sberror` work). A configuration with
`HIST_DEPTH` 0 has no dispatch history. `tsr-dbg diverge` and periodic firmware PC sampling are not implemented.

## 2. The debugging design

DB has the registers, the state machines and the decisions (DB-1 to DB-23). In short:

<p align="center"><img src="fig/debug.svg" alt="On the host, GDB talks the remote serial protocol to a debug service in the process that owns the device, or to the standalone tsr-dbg; the service and the profiling and log tools use tsr_transport, which reaches the core CSR window over PCIe. In the window: the DMI at 0xC00 leads to the RISC-V debug module, the Ibex control core (halt, step, breakpoints) and the system bus access to the TCMs and the SPM; the DBG block at 0x800 to 0xAFC holds the stop logic for the queue heads, the 512-entry dispatch history and the unit probes; the log ring, the performance counters and the trace writer to a DRAM ring serve observation." width="100%"></p>

RISC-V debug (DB §3, §4). The debug module is riscv-dbg (Debug Spec 0.13.2) in the `clk_rv` domain. The host reaches
its registers through the DMI window of the core CSR port (CSR 0xC00 + 4r; U200 `s_axi_control` 0x1C00 + 4r),
enabled only by `DBG_CFG.DMI_EN` so that generic CSR dumps stay side-effect free. A GDB RSP server runs inside the
process that owns the device (`xrt::ip` opens the compute unit exclusively) and drives DMI directly, with no OpenOCD
and no software JTAG TAP. Memory access goes through the system bus master (SBA), which can read memory without
halting the hart. SBA writes to the control window, the core window and the TCMs while the hart runs are refused
(D15). The debugger offers two hardware breakpoints (Ibex execute triggers), unlimited software breakpoints (ITCM
is writable; `fence.i` follows), two data watchpoints, and halt at boot or at kernel entry
(`TSR_GDB_WAIT=boot|launch`). `HOLD_ON_HALT` holds the queue heads while the hart is halted, so that a halted hart
and a quiescent datapath are the same observable state.

Tile-level debug (DB §5). A tile op has no instruction boundary: up to five units run concurrently. The comparable
stop point is the push-order prefix: every entry with push index below T has completed and no entry at or above T
has started. The DBG block reaches it by holding every queue head whose push index is at least T. On top of it:
`STOP_AT` a push index, `QSTEP` op by op, serial mode (one op in flight at a time, in push order; the launch
timeout is disabled), tile breakpoints on opcodes, push indices, tickets and SPM or DRAM ranges, the 512-entry
dispatch history that records the pushing C++ line, unit probes and a RISC-V snapshot from Ibex's crash-dump port.
Tile debug changes nothing in the program, and its state survives `QABORT`, `QCLEAR`, the RISC-V reset and
`SOFT_RESET`. A halted core with `HOLD_ON_HALT` and a drained datapath equals the simulator after the same number of
retired tile ops (DB §9.4).

Log, crash, trap (DB §6). Firmware text is binary: a 32-bit header (`N`, `LEVEL`, `SOURCE`, `TYPE`, timestamp flag,
launch sequence, sync nibble 0xA) and, for FMT records, the address of a format string in the non-loaded section
`.tsr_logfmt` plus 32-bit arguments. Records go through the 256-word core-domain log ring (readable even when the
queues are wedged), are never torn, and are dropped whole with a counter when the ring is full. A trap fills
`__tsr_crash` (44 words: `mcause`, `mepc`, `mtval`, `mstatus`, MTIME, launch, kernel, `RT_STATE`, `RT_INFO`,
`x1`-`x31`), pushes a FATAL TRAP record, posts a TRAP answer to the host and parks in a WFI loop. The transport
prints one report line (`TRAP: <cause> (mcause n) at function+0x1c (file.cpp:123), ...`, with the log tail) and
restarts the device unless `TSR_GDB_WAIT` is set.

Hang triage and recovery (DB §7). Every DBG register is a core-domain register mux, so a read never waits on a unit,
on `clk_rv` or on memory. (The XRT shell's AXI firewall cuts a control slave that does not answer, and only a card
reset recovers it.) The transport's timeout path follows the recovery ladder: level 0 hang dump (`TSR_DUMP_DIR`),
level 1 debug release (serial off, holds off, hart resumed), level 2 `LQ_CANCEL` covering the outstanding launches,
level 3 `SOFT_RESET` and a warm restart, level 4 a card reset (`xrt-smi reset`). A hung unit cannot finish
`SOFT_RESET`, and there is no unit reset (CC-22, EX §8.6).

## 3. Profiling

How to measure and what to report with a result: [`performance.md`](performance.md). The PERF space (EX §6) is gated by `ACTIVE` (a launch is in progress) and `QHALT` (the hart is halted and the
datapath quiescent), so PERF does not move between launches and the difference of two reads is one launch. DB adds
three slots at the top of the range: 61 `TRW_RECORDS`, 62 `TRW_DROPS`, 63 `DBG_STOP_CYCLES` (a measurement is valid
only if it is 0). DDR idle of channel c is `CYCLES - CH_RD_BEATS[c] - CH_WAIT[c]` (with `CAPS` bit 16 an MVM can deliver two
beats in one cycle, one on each read port of the channel, so the difference can be too small by the cycles with two beats: use
`TSR_MVM_DUAL=0` for this measurement), issue-bound MVM time is
`MVM_EMPTY` and dependency-bound `MVM_WAIT`.

The utilization split (`docs/spec.md` §14). The DDR utilization `U` is the bytes read per token times tokens per
second over the DDR read bandwidth measured with `tsr_bw` ([`bandwidth-kernel.md`](bandwidth-kernel.md)), and `U` is about `(1 - H)(1 - G)(1 - F) E`:

| Term | Meaning | Measured from |
|---|---|---|
| H | share of wall-clock time in which no launch is active (host CPU work, PCIe round trips, synchronization) | host time stamps and `PERF_CYCLES` |
| G | share of device time outside MVMs (LD, QACT, VU, ST, issue gaps) | `1 - PERF_MVM_CYCLES / PERF_CYCLES` |
| F | MVM time above the ideal inside `PERF_MVM_CYCLES`: fill, broadcast and tail, and on two weight streams the time the DDR cannot follow (§3.1) | the ideal MVM cycles (§3.1: the weight beats of the slowest channel over its streams) against `PERF_MVM_CYCLES` |
| E | effective read bandwidth while the MVM reads weights, over the `tsr_bw` bandwidth; on two weight streams it can be above 100 % (§3.1) | `PERF_CH_RD_BEATS`, `PERF_CH_WAIT` |

The effective bandwidth of channel c is 64 B x `CH_RD_BEATS[c]` x f_kernel / `CYCLES`, and the channel imbalance is
the maximum over the mean of the beats. The limit of a weight stream is 64 B per core cycle, and a cluster has one
stream, or two on a core with `CAPS` bit 16 ([rtl-mvm.md](rtl-mvm.md) §15). The order of optimization follows the
terms: host round trips (H), overlap of non-MVM ops with MVMs (G), fill and tail (F), DDR efficiency (E).

`TSR_STATS=1` prints, at close, the transport statistics (launches, launch wall time, CSR reads and writes, DRAM bytes,
log words and drops, boot, timeouts, traps, restarts) and, from the ggml backend, graph_compute calls, launches, op
records per launch, OPL cache hits and misses, wall time in graph_compute and in launches, and the device's PERF
counters over the same span. Timelines come from the hardware trace writer (TRW, DB §8.3): 32-byte records (per tile
op the push index, ticket, end time and the times `RUN`, `QWAIT` and `HWAIT`, plus `MARK`, `STOP`, `RVHALT` and
`DROP` records) written by a second write source of channel 0's `tsr_mem_arb` into a DRAM ring, with markers around
every kernel launch and every op-list node. `tsr-prof trw` produces Perfetto JSON (process `Tesseract0`, one thread
per unit, a RISC-V thread, counter tracks per channel) and runs the self-checks of DB §8.7 (retired ops, per-unit
busy sums, per-channel beats) on every profiled launch. A trace writer's configuration survives `SOFT_RESET` and the
end of the process that set it, so a ring started with `tsr-dbg` records the launches of a later run with no change
to that run. A speed measurement is always taken with the profiler off.

### 3.1 The profile report

`sw/tools/prof/prof_report.py` turns a run of the ggml backend into one page and one Perfetto file. With
`TSR_DUMP_PLAN=<dir>` the backend writes, next to a normal run and without changing a record or an output bit, one
`plan_<id>.json` for each graph it builds (the ggml nodes with their placement, the fusion groups, the op-list
records with their argument words, dependencies, DRAM ranges, weight bytes, ideal MVM cycles and the firmware's SPM
tile plan, and why each list ended), `launches.jsonl` (one line per launch with the PERF deltas),
`cpu_fallback.json` (the nodes `supports_op` refused, with the reason), `trw.raw` (the trace writer's records) and
`meta.json`. It starts the trace writer itself when the device opens and stops it when the backend is freed, so the
card needs no `tsr-dbg` step. It also sets the PROF bit: the `NODE_BEGIN` marker before every record, with its push
index, joins a trace op to its record. A MUL_MAT that the builder lowers as MUL_MAT_BEGIN and MUL_MAT_END (SW §9.3)
is two records with the node's tag. The BEGIN has the reads, the weight bytes and the ideal MVM cycles of the node, the
END the write bytes, and the ST is pushed by the END, so it joins to the END (the ST of a BEGIN that the executor closes early joins to the record that closed it, SW §7.3). The tables of the profile report list the
two records, and the cycles, bytes and ideal of a node are the sum of its records.

```sh
sw/tools/prof/capture_card.sh MODEL build/prof-report/card 33        # card: waits for an idle card, one run, trace writer off afterward (WARM=1 replays the first-ATTPV trace first)
TSR_SIM_DRAM_DIR=build/simdram sw/tools/prof/capture_sim.sh MODEL build/prof-report/sim 33   # simulator, same model, prompt and tokens
NO_DUMP=1 sw/tools/prof/capture_card.sh MODEL build/prof-report/card_noprof 33               # optional: the same run without the profiler
EXPECT_CYCLES=<cycles of the unprofiled run> REF_RUN=build/prof-report/card_noprof sw/tools/prof/make_report.sh \
    build/prof-report/card build/prof-report/sim build/prof-report "Model name"
```

`report.html` is one self-contained file that opens from `file://` and makes no request. It is a profiler window, laid
out like Nsight Systems and Nsight Compute: a toolbar (model, image, commit, bitstream, clock, launch selector, theme
toggle), a tree on the left (launches, then layers, then records), a timeline, and a bottom pane with four tabs. The
timeline has a time ruler in cycles and in microseconds, a group for the card and one for the simulator, and one row
per unit: the RISC-V issue band, MVM, VU, DMA, QACT, ATT. A unit whose ops overlap in time gets further rows
("overlap 1"). Slices are colored by the kind group of their record. The wheel zooms, a drag pans, a click selects,
hovering shows the op, and Selection zooms to the selected record. The tabs are Summary (speed, device cycles, the
utilization split H, G, F, E and U, cycles by group, the top records, card against simulator, the checks and the
cycles of every decode launch), Records (by kind and by record, sortable and filterable: count, cycles, share, ideal,
efficiency, GB/s, simulator cycles), Graph (the scheduler splits, the lists and why each ended, every ggml node with
its placement, the CPU fallbacks) and Details (the selected record: cycles on the card and in the simulator, traffic,
SPM tiles, argument words, DRAM ranges, its graph node and its ops). One record is selected at a time, and the tree,
the timeline, the tables and Details follow it. A launch other than the reported one is listed in the tree and the
selector but has no records; run the report again with `--launch N`. `trace.perfetto.json` has the same token for
Perfetto or chrome://tracing, and `summary.json` has the numbers. The Perfetto file adds flow arrows from a record to
its ops, counter tracks, and on every slice the node name, kind, layer, shape, bytes, measured and ideal cycles. Slice
ids are `card:r<record>.<UNIT><n>` and `sim:...` (`args.id`). The trace has no cluster field, so MVM is one row with
overlap rows where ops overlap.

The design of the page is fixed by a short spec at the top of `sw/tools/prof/report.css`. The tokens are the single
source for both themes.

| Part | Rule |
|---|---|
| Regions | Toolbar 32 px, tree 264 px, timeline over a draggable splitter over the bottom pane, status bar 20 px. Under 900 px the page is one column, the tree is a drawer, and the timeline scrolls sideways. |
| Type | 12 px system sans for labels, 12 px monospace with tabular figures for every number and id, 11 px for headers, nothing above 13 px. |
| Spacing | 2, 4, 8, 12 px. Table and timeline rows are 20 px. |
| Rules | 1 px lines. No shadows, gradients, pills, emoji or hero headings. Corner radius at most 2 px. |
| Color | Gray chrome, one accent for selection and focus, green and red only for pass and fail text. Light by default, dark under `prefers-color-scheme`, and a toggle that overrides it. |
| Kind groups | One fixed palette in both themes: attention, router, experts, dense, residual, head, in the order of `args.group` in the Perfetto file. Ops with no record are gray. |
| Numbers | Right-aligned, comma-grouped, with the unit in the header or in its own column. Labels are short nouns: Cycles, Duration, Efficiency. |
| Output | The page depends only on the dump, so equal inputs give the same bytes. |

A record's cycles are the increase of the launch's completion frontier (the latest end of any op so far, from
LAUNCH_BEGIN) over its ops, so they add up to the launch with the setup and the tail.

The ideal cycles of an MVM record are the weight phase of its MVMs at the stream limit of the core. An MVM streams the
weights once. A weight stream takes one beat per cycle, and the slowest channel holds `rows = ceil(N / NUM_CH)` of the N
rows of the matrix, each of ROW_BEATS beats. On one stream the MVM takes `rows x ROW_BEATS` cycles. An MVM of T ≤ T_MAX / 2
tokens runs on two streams when the core has the dual weight stream (`CAPS` bit 16, [rtl-mvm.md](rtl-mvm.md) §15) and
`TSR_MVM_DUAL` is not 0. It then takes `ceil(rows / 2) x ROW_BEATS` cycles. A MUL_MAT record has one MVM per token tile of
the firmware's plan, the last tile with the remaining tokens. A MUL_MAT_ID record of a decode token has one MVM of one
token per used expert, and so does a batch record that the planner does not group. A MUL_MAT_ID record of a batch (`T > 1`)
that the planner groups runs the pairs of one expert as the rows of one tile, and the routing decides the tiles. Its ideal
and weight bytes are the lower bound: the `T x N_USED` pairs in full tiles of the plan's `tile_rows` rows and one tile with
the rest. The real run reads the weights of more tiles when the pairs spread over many experts, so its MVM time is further
above the ideal than a dense record's, and the record's tile count in the report is the planned least. The trace's MVMs of
the record are the tiles of the routing, so no MVM op is matched to a planned pass: the plan file has `grouped` and no
`mvm_passes` for the record, and an MVM op of it has no ideal of its own. Any other record can mix MVMs on one and on two
streams, and its ideal is the sum. `meta.json` has `mvm_dual`, 1 when MVMs of T ≤ T_MAX / 2 ran on two streams, and the
Summary tab shows it (a dump without the field is read as one stream). The bytes of an MVM op are the weight bytes of its record over its MVMs. The stream rate ceiling of the Summary
tab (E at clock) is the rate the streams ask for over the `tsr_bw` bandwidth: the weight bytes per ideal cycle, `NUM_CH` x
64 B on one stream and up to twice that on two, at the core clock. The ideal is the limit of the streams, not of the
memory. On the card two streams ask for more than the memory gives ([performance.md](performance.md)), so E at clock is
above 100 % and the DDR, not the core, sets the time of such an MVM. The efficiency of such an MVM is then about
1 / (E at clock) at best. F holds the time the DDR cannot follow, about 1 - 1 / (E at clock) of the MVM time, and above
that the fill, broadcast and tail. E, the DDR read rate over the ideal MVM time, is above 100 %.

The report checks the record cycles against the launch span and `PERF_CYCLES`, the sums of RUN against the unit counters
(DB §8.7), `RETIRED` against the ops, the markers against the records, the trace writer's drops and the simulator against
the card. The dump and the publisher of `tsr-top` (§7) run in the same process and only read, so neither moves a counter,
a record or an output bit.

### 3.2 Comparing a card run with the simulator

1. Card: `tsr-dbg --device xrt --attach -ex "trw start 16"` (16 MiB holds about 500,000 records), then the run with
   `TSR_OPL_PROF=1 TSR_OPL_KINDS=run.kinds` (the graph kernel pushes a `NODE_BEGIN` marker before every record, and
   the backend writes the kind of every record of every launch to `run.kinds`), then `tsr-dbg --device xrt --attach
   -ex "trw dump run.json run.raw" -ex "trw stop"`.
2. Simulator: the same command line with `TSR_DEVICE=sim TSR_SIM_CARD=k4dual TSR_SIM_GDN=1 TSR_SIM_LOG=w,sim.log`
   (the release image's configuration; the prefix `w,` also writes the eight words of every push).
3. `sw/tools/trw_vs_sim.py kinds --trw run.raw --sim sim.log --kinds run.kinds --launch 2` gives, per record kind, the
   cycles the record adds to the launch on both sides. `units`, `gaps` and `record` give the same by op class and for
   one record. Launch 0 is llama.cpp's warm-up pass, 1 the prompt, the rest are decode tokens. The two op streams
   must be the same (the tool refuses a unit whose op counts differ). The card's run times of VU ops include their
   wait for the op before them, so the VU is compared by the gaps between completions.

## 4. Access paths and JTAG

The register meaning is the same on every target, and debug access always enters through the core's CSR port. The
core has no JTAG port, and vendor primitives stay in `rtl/tech/`. `sim` and `mock` use function calls. `vkernel`
uses the adapter window. The U200 uses the PCIe CSR window (`s_axi_control` 0x1000 + offset), with an ILA in debug
builds and XVC over PCIe for signal-level debug.

### 4.1 U200

There is no JTAG to user logic. The XRT shell is a DFX design: the user kernel lives in the reconfigurable region,
and configuration primitives such as BSCANE2 can be placed only in the static shell. The shell's user debug network
is reached only by XVC over PCIe and has no USB-JTAG fallback (`supports_jtag_fallback = false` in the platform's
`ext_metadata.json`). JTAG would ride the same PCIe link, slower than the CSR window, so no JTAG bridge is built.
Architectural debugging uses the debug module and the DBG block through the CSR window.

An ILA is for what the debug block cannot see (AXI deadlocks, firewall trips, CDC, cycles inside an op). It is built
only from a separate debug link configuration, never in the regular `k1.cfg` and `k4.cfg` builds, with the trigger
on the internal net `dbg_stop_evt` of the DBG block (DB §5.2). `v++` `[debug]` options (`chipscope=tsr_k4:m_axi_ch0`
..., `chipscope=tsr_k4:s_axi_control`, `protocol=all`) cover the interfaces, and Vivado Tcl hooks in the
configuration's `[vivado]` section mark core-internal nets `MARK_DEBUG`. No ILA instance and no debug attribute goes
into `rtl/` outside `rtl/tech/`. Connect with `debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u<dev>.<idx> --hw_server`,
then `debug_hw --vivado --ltx_file <file>.ltx`.

### 4.2 Zynq / Kria

The access path is PS AXI-Lite (UIO mmap). On a flat design a BSCANE2 bridge on USER3 or USER4 (clear of the debug hub's
default USER1) can carry JTAG to the CSR port as a backup when Linux is dead. This is a design note, not a build.

### 4.3 ASIC and boards without a host bus

JTAG TAP plus DTM is the main path, in the RISC-V debug specification's DTM format (IR 5 bits, IDCODE 0x01, `dtmcs`
0x10, `dmi` 0x11, BYPASS 0x1F). The `dmi` address is the CSR offset divided by 4, so the debug module registers sit at DMI
address 0x300. A platform-layer arbiter (`tsr_csr_arb`) arbitrates the host bus (if any) and JTAG onto the one core CSR
port. Without a host bus a boot ROM is needed, since the TCMs are undefined after reset.

### 4.4 Several CSR masters

Only the master that owns execution writes `CTRL`, the launch queue and the loader and reads the PERF counters: on the
U200 and Zynq the host bus, on a board without a host bus the JTAG DTM. Any other master reads only registers that
have no side effect, because a PERF low-half read moves the single snapshot slot and a 64-bit value could tear.
Inside the host, one thread under the transport's mutex reads PERF.

## 5. Design reference

The design is specified in DB ([`docs/design/05-debug.md`](design/05-debug.md)). §1 lists the tools and §2
summarizes the design.

## 6. Deliberately not done

- A debug SPM window and a halt-on-error path: the debugger reaches the SPM through SBA, and error evidence is the
  unit error records, the dispatch history and the trace writer.
- Instruction patching for breakpoints in the tile stream: tile breakpoints are matchers on the enqueue tap.
- Halting inside a tile op (mid-MVM): it is not an observation point of the push-order semantics. Inside an op, use
  a repro trace with waveforms, or the ILA.
- On the U200: an own BSCANE2 or JTAG bridge, and an ILA in the regular build.
- llama.cpp: no patch to the pinned commit for profiling, and no `cb_eval` node stops or timing, because they cut
  splits and change what is measured (numcheck keeps its own `cb_eval` use for numerics).
- Always-on RTL instrumentation beyond DB's configurable blocks (`HAS_TRW` and `HIST_DEPTH` trade area; DB §11), and
  any change to the PERF indices of EX §6, the `CAPS` bits and the error codes.

## 7. Live monitor: `tsr-top`

`tsr-top` shows the card the way `nvtop` shows a GPU: a header per device, meters, a time series and a process
table. It is a Python 3 `curses` program (standard library only) that reads the records published by the process
that owns the card (SW-87). Build target: `bin/tsr-top`, a launcher for `sw/tools/top/tsr_top.py`. The XRT transport
opens the compute unit with exclusive access (EX §2.4), so while llama.cpp or any other tool runs on the card no
second process can read the CSRs or the PERF counters. The owner therefore writes what it can already read into
shared memory, and the monitor never touches the compute unit while an owner is alive.

### 7.1 Running it

```sh
build/dev/bin/tsr-top                      # the live screen
build/dev/bin/tsr-top --once --size 100x36 # one frame as plain text, no curses
build/dev/bin/tsr-top --json               # one snapshot as JSON (devices, processes, rates, the raw records)
```

Keys: `q` or `F10` quit, `F2` (or `s`) setup (switches panels on and off), `F6` (or `o`) sorts the process table by a
column (the same column again reverses it), `+` and `-` set the refresh interval (0.1 s to 10 s, default 1 s, `-d S`
at start), `Tab` selects the next device (the card, and one simulated device per process that publishes from a
simulator), `Esc` closes a menu. `--no-color` gives a monochrome terminal and `--ascii` replaces the box drawing
characters. Other options: `--dir` (record directory), `--window S` (the span of `--once` and `--json`, default 1
s), `--no-sensors`, `--include-dead` (also show the record that `TSR_MON_KEEP` left behind), `--token-window S` (the
span of the tokens/s average, default 3 s), `--sensor-interval S` (how often `xbutil` runs, default 10 s) and
`--xbutil PATH`.

### 7.2 The record and the publisher

- The file is `/dev/shm/tsr-mon-<pid>` (`TSR_MON_DIR` moves it): one 4 KiB page of plain little-endian fields,
  described by `tsr_mon_rec` in `sw/transport/tsr_mon.h`. It is created whole (a temporary name, then a rename) and
  removed when the transport closes or the process exits cleanly. A record whose pid is dead is ignored by `tsr-top`
  and removed by it. The page is behind a seqlock: the writer makes the counter odd, stores the page, makes it even,
  and the reader keeps a copy only if the counter was even and unchanged, so the writer never waits for a reader. The
  layout has a number (1). `tsr_top.py` holds the same field table, and the ctest `tools_top_layout` compares it with
  the table that the C side prints. A reader skips a layout number it does not know.
- `tsr_opl_dev` (`sw/ggml-tesseract/tsr-opl-dev.cpp`) attaches the publisher when it opens the device and brackets
  every `graph` launch, so the ggml backend and `tsr_rt` both publish. The ggml backend adds the mode (`M6` when the
  device takes the matrix products only, else `M7`), the kinds, the dominant weight type, the device memory in use
  and the tokens.
- After a launch the writer updates the counters (no device access). At most every `TSR_MON_PERF_MS` (100 ms) it
  also reads the PERF counters through `tsr_transport_perf_summary` (about 80 CSR reads) and `RV_FREQ`. PERF does
  not move between launches (EX §6), no record, descriptor or tensor changes, and the output bits are the same with
  the publisher on and off (ctest `tools_top_identity`).
- The record holds the process (pid, uid, command line, times), launches, host wall time in launches, op records, the
  launch in flight, all 64 PERF totals and the change between the last two reads, tokens, device memory in use, the
  clocks, the configuration (channels, `T_MAX`, lanes, SPM, `K_MAX`), `CAPS`, the capability, the Basis version, the
  image id and path, the kind table, the device line (name, BDF, kernel), `$TSR_XCLBIN`, and the model name and
  quantization. `TSR_MON=0` turns the publisher off. `TSR_MON_KEEP=1` leaves the file at exit (flag `EXITING`).
- The capacity of the card is a separate small file, `tsr-cap-<bdf>` (the PCIe address of the card) in the same
  directory: four text lines, `tsr-cap 1`, `bdf <bdf>`, `num_ch <n>` and `ch_bytes <bytes>`. An owner open of the xrt
  transport writes it after it measured the channel capacity ([xrt-host.md](xrt-host.md) §4.4), whatever `TSR_MON` says.
  It stays after the process exits, since the capacity belongs to the card. `tsr-top` reads it to show an idle card
  against its real capacity (32 GiB on this U200), not the 64 GiB that the xclbin memory topology implies. Without it
  `tsr-top` uses the size that an idle probe (`--probe`) read, which is the topology, and else shows `n/a`.

### 7.3 What each part of the screen shows

- Device line: board and shell, `PCIe GEN g@wx`, state (`IDLE`, `HELD`, `OWNED`, `BUSY`, `LAUNCH t` for a launch in
  flight longer than 1 s). Clock line: `CORE 200MHz  RV 100MHz`, from sysfs `icap.u.*/clock_freqs` of the card's user
  function (clock ID 0 is the core, 1 the RISC-V control core), else from the clock the owner fitted from
  `DBG_MTIME`. Sensor line: FPGA temperature, fan, power and limit, from the sysfs hwmon of the card's user function
  and from `xbutil examine -r thermal electrical mechanical -f JSON` every 10 s. None of these touches the compute
  unit.
- `COMPUTE`: the busiest compute unit as a share of wall time, with its name (`MVM`, `VU`, `QACT`, `ATT`, `ATT_QK`,
  `ATT_PV`; the DMA moves data, it is not compute), from the change of PERF over the window. `MEMORY`: the device
  memory in use against the memory of the card, for example `12.0Gi/32.0Gi` on this U200 (four channels of 8 GiB). The
  total is the `mem_total` of the owner's record, which its transport measured at open. For an idle card it is the
  capacity recorded for the card (§7.2), which gives `0.0Mi/32.0Gi`, and `n/a` when none is known (a process that
  holds the card without publishing leaves the use unknown, so the meter shows `n/a`). `MEM BW`: the read rate of all
  DDR channels together, for example `43% 30.7GB/s`. The percentage is the share of what the DRAM gives:
  `NUM_CH` x 18 GB/s on this card, as `tsr_bw` measures it ([bandwidth-kernel.md](bandwidth-kernel.md) §7). `COMPUTE`
  and `MEM BW` come from the change of PERF. They show `n/a` while there are no rates: before the first PERF read, and
  when the rates are idle (see below). From 96 columns the three meters share one row. On a narrower terminal
  `COMPUTE` and `MEMORY` share the first row and `MEM BW` has a second row, as wide as the two together. The time
  series box draws two lines over the last two minutes (less when the box is narrow), both from 0 to 100 %: the
  `COMPUTE` share, and the device memory in use as a share of the memory of the card, which is the `MEMORY` meter over
  time. The memory line rises while a model loads, when no launch runs and the compute line stays at 0. The legend
  gives the current value of each line, or `n/a` where it is not known. Without colors (`--once`, `--no-color`) the
  memory line is drawn with heavy glyphs and its legend ends with `(heavy)`.
- Process table: `PID USER MODEL MODE TOK/S LAUNCH/s CPU HOST MEM COMMAND`. `MODE` is `M6` or `M7`, and `TOK/S` is
  `pp n` while a prefill runs. A process that has the card open but does not publish is listed dimmed with `(no
  tsr-mon)`. User, CPU % and RSS come from `/proc/<pid>`, and the other users of the card from the `/proc/*/fd`
  links to its render node (own processes only).
- Setup panels (off by default): the image line (kernel name, id, `CAPS`, kind count and table signature, mode),
  unit meters `ACT` (active share of wall time, 1-H), `MVM VU DMA QACT ATT` (busy share of the active cycles),
  `EMPTY` (MVM idle, queue empty: issue-bound) and `WAIT` (MVM idle, queue entry cannot start: dependency-bound),
  `CH0..CH3` bars with the read rate against the stream limit (64 B per core cycle for each weight stream, two on a
  core with `CAPS` bit 16), and the utilization split of §3.

Rates cover the newest PERF read and the newest one at least one refresh interval older. When no new PERF read came
for more than 2.5 s and no launch is in flight, the rates are shown as idle.

### 7.4 The card when nobody owns it

`tsr-top` does not open the card by default, because a run that starts while another process holds the compute unit
exclusively would fail to open it. `--probe` runs `tsr-mon-probe`, which opens the card in attach mode (no reset, no
bind, as `tsr-dbg --attach`), reads the identity, the fitted core clock and the lifetime PERF counters, and closes
it. It runs only if no tsr-mon record has a live pid, no process has the card's render node open, and the xclbin that
`TSR_XCLBIN` names is the one the card has (its UUID, so that the open cannot load another image), then at most
every 15 s. A process of another user that starts while the probe holds the card still fails to open it, which is
why the probe is off by default.

### 7.5 Limits

- The counters are read after a launch, not during it. A long prefill ubatch is one launch of seconds, and the
  screen shows the previous values and `LAUNCH t` until it ends.
- Tokens are counted from the graph. `tok_decode` (the `TOK/S` column) counts the tokens the main model generates, and
  `tok_prefill` (`pp n`) its prompt tokens. A pass of one token is one decoded token, and a pass of more tokens whose
  rows are not all outputs is prefill. With speculative decoding (SW §7.11) a verification batch (at most 16 tokens, every
  row an output, the positions one run) is one decoded token, and the drafts the previous batch accepted are added when
  the next batch of the same context shows where it starts: the accepted drafts of the last batch of a generation are
  never counted. A batch of at most 16 tokens with every row an output whose positions are not one run (several
  sequences, one row each) counts one decoded token per row. Any other batch of at most 16 tokens with every row an
  output, such as a perplexity ubatch of 16 or fewer, counts as decoded tokens too. The position of a pass comes from
  the `inp_pos` input and the output rows from `out_ids`. A pass of the draft model's context, recognised by
  streaming less than 0.9 of the weight bytes of the largest pass, counts only in `passes`. When `inp_pos` stays on
  the CPU (matmul mode, or ROPE on the CPU) the positions are unknown: a speculative run then shows no decode rate
  and its verification batches may count as prefill. The first pass after the scheduler re-splits a graph can be
  missed in `passes`. The tokens of a pass are counted when its position is known, so a pass longer than the 3 s
  window shows as a spike.
- Before the first launch the record has the process, the memory in use and the configuration but no PERF read, so
  the meters that come from PERF show `n/a`. On a simulated transport the clock is simulated, so `ACT` and the GB/s
  over wall time are not hardware figures. An owner started with `TSR_MON=0` is visible only as a process that holds
  the card.
- The model name comes from `$TSR_MON_MODEL`, `-m`, `--model`, `-hf` or the first mapped `.gguf`, and the
  quantization is the dominant weight type (the file name's `Q4_K_M` is preferred when it has one).

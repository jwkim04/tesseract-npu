# Tesseract profiler and debugger — design plan

Status: **plan, not implemented** (2026-10-23). Nothing described here exists in the code yet,
except the facts in §1, which were checked against the code on that date (file and line given).
The contract is the spec section "프로파일러와 디버거" in `docs/spec.md`, plus the two
구현 명확화 items it relies on ("실행 사이 PERF 불변", "START와 SPM") and the M9 `WAITM`
amendment. This document holds what the spec leaves out: tools, file formats, metrics,
environment variables, the register-by-register map, area estimates and the verification plan.
Source-of-truth order: `docs/spec.md` > ISA simulator (`sw/sim`) > this document. Names of new
functions, files and options below are proposals; PD1 and PD2 fix them.
(2026-10-24) The spec's M8 plan revision drops LOOP, JUMP and WAITM (and with them the M9 `WAITM`
amendment) in favour of a RISC-V control core: see the spec section "제어 코어", the paragraph
"제어 코어 디버그" in the section "프로파일러와 디버거", and `docs/ctrl-riscv.md` §10. §2 (PD4),
§3.2, §3.3, §3.5 and §9 follow that revision; the remaining "without LOOP / JUMP" conditions
(§5.7, §6.2) now always hold and are tidied at PD4.
(2026-10-30) **v2: the command-path debugger and profiler of this plan are superseded by DB**
(`docs/v2/05-debug.md`, D14–D18). v2 removes the host command-buffer path, and with it every part
of this plan that stops, steps, splits or records host command-buffer runs: the PD2 debug block
(§3, deleted with `tsr_cmdproc`; its CSR range is reused by DB's DBG block at 0x800–0xAFC), the
debug SPM window (§3.5), prefix runs and halt emulation (§4.6), the per-exec shadow transport
(§4.5), the instruction-index divergence finder (§4.7), per-exec records and `split` attribution
(§5.1, §5.4), the out-of-order decision (§5.11), the trace ring (§6.3), and GDB through OpenOCD
(PD4). DB replaces them with RISC-V debug over DMI and an RSP server, tile-level stop points,
serial mode, tile breakpoints, a dispatch history, unit probes, a hardware trace writer and
TSR_STATS v2; the PERF space and its ACTIVE / QHALT gating are EX §6. Still valid: the ILA
debug-build recipe and no JTAG on the U200 (§7.2), dumps under `TSR_DUMP_DIR` (§4.4), Perfetto
output and `tsr-prof` (§5.8), and the rule that every CSR access completes in bounded time.
Source-of-truth order: `docs/spec.md` > `docs/v2/*` > `docs/ctrl-riscv.md` > this document.

The plan answers a user request with three parts: a **profiler**; **debugging** (dump the chip
state, set breakpoints, halt and step); and whether the FPGA's **JTAG** can reach user logic.

| Question | Answer |
|---|---|
| Dump the chip state | PD1, no RTL: CSRs, PERF, PC, ERR_*, ADAPTER_STATUS, the program and its map to ggml nodes, DRAM, and SPM through an ST dump program. PD2 adds R0–R15, the last instruction, resolved addresses, unit phase, outstanding memory requests and an SPM window (§3, §4.4). |
| Breakpoints, halt, step | PD2 debug block: HALT, RESUME, STEP, 4 instruction-address breakpoints and a retire-count breakpoint, only at instruction boundaries (§3). PD1 emulates "stop before instruction k" with prefix runs (§4.6). |
| JTAG | U200: no; the CSR window over PCIe is the path, and ILAs (debug builds only) are reached by XVC over PCIe. Zynq/Kria: optional BSCANE2 bridge. ASIC or a board without a host bus: JTAG TAP + DTM is the main path (§7). |
| Profiler | Software first: a record per device run with PERF deltas and host times, mapped to ggml nodes and layers; Perfetto timelines and reports. RTL counters and a trace ring only when a measurement says so (§5, §6). |

---

## 1. What exists today

Checked 2026-10-23.

- **CSR space**: 12-bit core CSR address (`tsr_pkg.sv:92`). Undefined offsets read 0 and ignore
  writes (spec 구현 명확화). Free: 0x080–0x0FC (ARG decode covers only 0x060–0x07F,
  `tsr_csr.sv:148`), and everything above the PERF counters up to 0xFFC. PERF counter i sits at 0x100 + 8i (LO) / +4 (HI), with NPERF = 5 + 2 × NUM_CH counters
  (`tsr_pkg.sv:125-135`, `tsr_csr.sv:150-152`): 13 on k4, 7 on k1, at most 37 (16 channels,
  ending at 0x227). So 0x800–0x9FC and 0xA00–0xA3C are free.
- **PERF**: CYCLES (BUSY cycles), RETIRED (END counts, a faulting instruction does not),
  MVM_CYCLES, VU_CYCLES and WAITM_CYCLES (hardwired 0), CH_RD_BEATS[c] (channel 0 includes
  instruction fetch), CH_WAIT[c]. Cleared only by SOFT_RESET and hardware reset, never by START
  (`tsr_csr.sv:34-39`, `tsr_sim.c:33`). One LO-read snapshot slot: reading counter i's LO
  latches its HI for the next HI read of i; any other read of HI returns the live value
  (`tsr_csr.sv:236-243`). Counters increment only on BUSY, retire, MVM hold, read beats and
  outstanding reads (`tsr_csr.sv:311-320`), so they do not move between runs (now spec
  구현 명확화 "실행 사이 PERF 불변").
- **Command processor** (`tsr_cmdproc.sv`): one instruction in flight; the boundary is `S_NEXT`,
  which checks ABORT first, then end of buffer, then the fetch-range flag, then pops
  (`:376-402`). `pop = rdy && insn_valid && !run_abort` has no range-error term, so the
  range-error entry (instruction word 0) is loaded into `ir_q` (`:324-325`, `:485-487`).
  `ir_q` keeps w0–w6 only; w7 survives only as a nonzero flag (`:149`). R1–R15 (`rf_q`, `:146`)
  survive END, errors and ABORT until the next START (R1–R7 := 0, R8–R15 := ARG) or SOFT_RESET.
  Cost per instruction: NOP/LI 4 cycles, LD/ST/QACT 10 + unit, MVM checks about 46 cycles.
- **SPM**: three fixed-priority clients DMA, QACT, MVM (`tsr_spm.sv:6`, `TSR_SPM_NCL = 3`,
  `tsr_pkg.sv:83`); only units dispatched in a run write it. START does not touch it in the
  RTL or in the simulator (`tsr_sim.c:576-586`); `tb/core` `spm_fill` / `spm_compare` already
  compare SPM across runs (now spec 구현 명확화 "START와 SPM").
- **Memory sources**: `TSR_NSRC = 3` (fetch, DMA, weight stream), so source id 3 is free
  (`tsr_pkg.sv:73-78`).
- **Simulator**: `tsr_sim_run` checks ABORT at the loop top (`tsr_sim.c:883`); the fetch-beat
  cache is a local variable; each instruction's model cycles are computed (`:913-922`) but not
  exported. Only CTRL.ABORT may be written from another thread during `tsr_sim_run`
  (`tsr_sim.h:57-59`).
- **Transport** (`sw/transport/tsr_transport.h`): open, write, read, a blocking exec that returns
  {status, err_code, err_pc, err_info}, and `tsr_transport_csr_read`. No CSR write, no timing, no
  PERF in the result. The kdev `wait(timeout)` contract leaves the run outstanding after a
  timeout (`tsr_kdev.h:103-107`), so a short-wait loop is allowed. The kdev open writes ABORT to
  a core left BUSY by an earlier process, reads `AP_CTRL` (clear on read) and clears a stale
  ADAPTER_STATUS (`docs/xrt-host.md` §3). `TSR_TRACE` (sim only) picks the trace directory
  itself: `<dir>/<seq 6 digits>/`, first free seq (`tsr_transport.h:36-37`); the writer's
  stale-file cleanup removes only `meta.json`, `cmd.bin`, `init/` and `expect/` files
  (`tsr_trace.c:133-159`).
- **U200 window**: core CSR `off` < 0x1000 is reachable at `s_axi_control` 0x1000 + off at any
  time, also mid-run; a window read takes 10 kernel cycles (`docs/platform-u200.md` §2). The
  adapter's round-robin arbiter already shares the core CSR port between the host window and the
  run sequencer (`tsr_xrt_ctrl.sv:253`). PCIe MMIO latency per read is **not measured**.
- **Backend** (`sw/ggml-tesseract/ggml-tesseract.cpp`): `TSR_DEBUG` is the log level and
  `TSR_DBG` its logging macro (`:78-91`). `tsr_exec` counts execs and aborts with ERR_PC and the
  failing instruction (`:1060-1075`). `tsr_run_range` copies `[first, first+n)` plus END into a
  CMD half and runs it (`:1077-1093`); programs larger than a half run in pieces at tile cuts
  (`:1095-1126`); every program it runs ends with END at pc = CMD_COUNT − 1. Node names
  (`seg.names`) are cleared at flush (`:1156`); `tsr_prog` keeps no names.
- **Selftest**: `tsr_xrt_selftest` decides whether a trace fits the device by comparing the
  configuration, including `CAPS`, exactly (`sw/tools/tsr_xrt_selftest.cpp:545`). This is why
  the debug block is discovered through `DBG_ID`, not a `CAPS` bit.
- **llama.cpp (pinned)**: no per-split or per-op timer; `cb_eval` cuts splits at observed nodes
  and synchronizes after each, which changes Tesseract batching. The CLI, completion and server
  tools install their own SIGINT handlers (`tools/cli/cli.cpp:52`,
  `tools/completion/completion.cpp:403`, `tools/server/server.cpp:537`).
- **XRT**: when a process exits with an `xrt::run` outstanding, KDS aborts the command; if the
  compute unit does not finish within the abort time it sets `bad_state` ("CU hangs, please reset
  device") and the card needs `xbutil reset` (xocl `kds_core.c`, e.g.
  `build/xrt-driver/final-r1-deb/src/driver/common/kds_core.c:792-794`).
- **Platform** (`xilinx_u200_gen3x16_xdma_2_202110_1`, `hw.xsa` `ext_metadata.json`): the user
  debug network is PF1 BAR0 0x1C00000 (`ep_debug_bscan_user_00`, an AXI-to-BSCAN debug bridge),
  `supports_jtag_fallback = false`; the management network has `supports_jtag_fallback = true`
  but is not user-visible. xocl exposes the user network as `xvc_pub` (XVC over PCIe).
  `hw/u200/k1.cfg` and `k4.cfg` have no `[debug]` or `[profile]` sections.

---

## 2. Phases

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** PD1–PD5 below plan for the command-buffer
path. v2 debug and profiling are DB (`docs/v2/05-debug.md`), delivered with M6 (D18); the work
packages are P1.12, P1.13, P3.7 and P3.8 of `docs/v2/06-software-verification.md` §11.

The spec's phase table is the contract; this is the detail. Milestone pass criteria do not
change, and no milestone depends on a PD phase.

| Phase | When | Scope | RTL and cost | Questions it answers |
|---|---|---|---|---|
| **PD1** software debugger and run records | After M3 part C merges. Until then no part C file is changed (`sw/transport`, `sw/tools`, …). The M3 gate does not depend on PD1. | Program map and `nodes.json` (§4.3); post-mortem and hang dumps (§4.4); shadow transport, first (§4.5); prefix-run search and halt emulation (§4.6); divergence finder and repro (§4.7, complete on sim); exec record with host times and LO-tracked PERF deltas (§5.1); `tsr-prof report / perfetto / check`; simulator debug model (resumable `tsr_sim_run`, atomic HALT, instruction hook, `TSR_SIM_FAULT`); transport additions (§4.1); `tsr-dbg` v0 (sim: everything; hardware: dump, prefix runs, diverge, repro); `tsr_xrt_selftest` measures CSR read latency and the overhead of an END-only exec; the U200 ILA debug-build recipe (§7.2), written down, built only when needed. | None. Software estimate: debugger ≈ 3 K lines (shadow ≈ 400, simulator model and hooks ≈ 500, backend map and dumps ≈ 400, `tsr-dbg` ≈ 2 K), profiler ≈ 1.8 K lines. | If the M3 hardware result is wrong, from which exec, instruction and node? If it hangs, where? What does one exec cost on the host and in XRT, and what does a CSR read cost? |
| **PD2** RTL debug block and exact attribution | M4, with the U200 technology mapping and timing work for the k4 bitstream | `rtl/core/tsr_dbg.sv` and hooks (§3); STATUS bit4; time counters stopped while halted; SPM client 3; `dbg_trig`. Transport halt handler, timeout rule, SIGINT, exit ABORT (§4.2). `TSR_DEBUGGER`, `TSR_DEBUGGER_BREAK`. tb-only retire probe, `split` attribution, a `ddr4` memory regime, `tsr-prof calibrate / predict / regress`, `perf/` baselines, the M4 out-of-order report (§5.11). `tsr_xrt_selftest` debug section. | FF ≈ 1.0 K, LUT ≈ 1.6 K on the U200 (§3.7). | First wrong instruction, element and channel of a 32B mismatch on the card? Which unit and which channel's outstanding request does a hang sit on? Expected 32B tok/s before M4 runs? |
| **PD3** vector unit | M6–M7 (and at M5: the drain-rule test, §3.8) | `DBG_S0..15`, VU and scalar codes in `DBG_UNIT`, two write watches, divergence checkpoints at layer boundaries. Profiler: VU/SOP spans, host round trips per layer, prefill efficiency per T. | +0.15 K FF, +0.4 K LUT | Once in-layer host round trips are gone (M6), which op goes wrong first? Who overwrote this SPM line? |
| **PD4** control core | M8–M9 | Control-core debug (spec "제어 코어 디버그", `docs/ctrl-riscv.md` §10): debug module, `tsr-dbg rv-bridge`, GDB through OpenOCD; Perfetto tracks from the completion records; `DBG_HIST*`; `tsr-dbg attach` for `sw/runner`; decision on extension counters and the trace ring for the command-buffer path (§6). | `DBG_HIST*` at most +0.3 K FF, +0.2 K LUT; the debug module and the completion records are part of the control-core RTL (`docs/ctrl-riscv.md` §12.1); trace ring 4 BRAM36 if built | In token n of a C++ kernel or the resident loop, which ticket, op or RISC-V instruction went wrong first? |
| **PD5** JTAG port | When a port starts (after M9) | `tsr_csr_arb`, `tsr_jtag_dtm`, BSCANE2 pre-check on Kria, Verilator + OpenOCD CI, `openocd` transport in `tsr-dbg`. Not on the U200. | DTM + CDC ≈ 0.25 K FF / 0.2 K LUT, arbiter ≈ 0.1 K LUT | Can the same debug map be used when the host bus is missing or dead? |

---

## 3. Debug block (PD2)

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** The whole PD2 debug block
(instruction-boundary halt, STEP, IA and retire breakpoints, `DBG_IR*`, `DBG_OPA*`, `DBG_R*`,
`DBG_HIST*`, `STATUS.HALTED`, the debug SPM window as SPM client 3) is deleted with `tsr_cmdproc`
(D14). DB §3–§5 and §10 define the v2 RISC-V debug, tile-level debug and the DBG block at
0x800–0xAFC; the debugger reaches SPM through the debug module's SBA (V2-CC-16). This section is
kept as a record.

### 3.1 Boundary, check order and drain

The core halts only at an instruction boundary, the place where ABORT is checked today
(`tsr_cmdproc` `S_NEXT`, `tsr_sim_run` loop top). The order there becomes:

1. ABORT → ends the run `ABORTED` (existing).
2. Halt conditions. If any is true the core halts, and every true condition is recorded in
   `DBG_STATUS.CAUSE`:
   - HALT: a HALT request is pending.
   - STEP: one instruction retired since a STEP.
   - BP: `DBG_CFG.BP_EN[i]` and `DBG_IA` = `DBG_BP<i>`.
   - RBP: `DBG_CFG.RBP_EN` and `DBG_RETIRED` = `DBG_RBP` (equality mod 2³²; `DBG_RBP` = 0 halts
     before the first instruction).
   - WATCH (PD3): the instruction that just retired wrote a range overlapping a watch range (the
     range is the one cmdproc already computes for its range checks).

   Skip rule: when the core leaves a halt by RESUME or STEP, the boundary it halted at is
   re-checked without BP and RBP (HALT still counts), so it does not halt at the same place
   again. With `DBG_CFG.TRIG_ONLY`, a matching BP or RBP does not halt; it pulses `dbg_trig`
   for one cycle (§3.6).
3. End of buffer → `ILLEGAL_OP` (existing).
4. Fetch range → `DRAM_RANGE` (existing).
5. Decode, checks, execute (existing).

Because 2 comes before 3 and 4, the core can halt before position `CMD_COUNT` or before an
out-of-range fetch; resuming then ends with the original error. The halt decision is computed
into a register in the cycle before the boundary; the target is zero extra cycles per
instruction when debugging is off.

**Invariant.** At a halt, the architectural state (R0–R15, s0–s15, the SPM lines written by this
run, DRAM) equals the ISA simulator after the same number of retired instructions. The debug
block's verification (§3.8) and the divergence finder (§4.7) rest on it.

**Drain.** In an implementation that overlaps instructions (M5 MVM-to-MVM overlap; out-of-order
execution, if it is ever adopted), a true halt condition stops issue, and the core halts only
after every issued instruction has retired, like `FENCE`. `DBG_RETIRED` and the state are taken
at that point, so the invariant still holds. STEP is the same: one instruction is issued and
drained. `TRIG_ONLY` does not wait (it does not halt).

### 3.2 Register map

Core CSR offsets (U200: `s_axi_control` 0x1000 + offset, i.e. 0x1800–0x19FC, through the
existing window). Offsets not listed are reserved and read 0. Every read is side-effect free and
answers even when a unit is stuck (it is a register mux): the XRT shell's AXI firewall cuts a
control slave that does not answer, and recovering needs a card reset.

| Offset | Name | Access | Content | Phase |
|---|---|---|---|---|
| 0x800 | `DBG_ID` | R | [7:0] block version (1), [11:8] NBP, bit12 SPM window, bit13 WATCH, bit14 s registers, bit15 IA history. 0 when the block is absent | PD2 |
| 0x804 | `DBG_CTRL` | W | bit0 HALT, bit1 RESUME, bit2 STEP; reads 0. Several bits in one write: only one is taken, STEP > RESUME > HALT | PD2 |
| 0x808 | `DBG_CFG` | RW | [7:0] BP_EN, bit8 RBP_EN, bit9 TRIG_ONLY, [11:10] WATCH_EN (PD3) | PD2 |
| 0x80C | `DBG_STATUS` | R/W1C | bit0 HALTED, bit1 HALT_PEND, bit2 STEP_PEND, bit3 SPM_ERR (W1C), [12:8] CAUSE (bit8 HALT, bit9 STEP, bit10 BP, bit11 RBP, bit12 WATCH), [23:16] BP_HIT | PD2 |
| 0x810 | `DBG_NEXT_PC` | R | Index of the next instruction; `ERR_PC` if the run is aborted here | PD2 |
| 0x814 | `DBG_IA` | R | Instruction address of `DBG_NEXT_PC`: absolute ch0 position in 32 B units, low 32 bits, = 2 × base + `DBG_NEXT_PC`, base = the run's start beat latched at its start: `CMD_ADDR` for a host START, the CALL's w1 for a CALL (M8) | PD2 |
| 0x818 | `DBG_RETIRED` | R | Instructions retired in this run; START clears it; wraps at 2³² | PD2 |
| 0x81C | `DBG_RBP` | RW | Retire-count breakpoint | PD2 |
| 0x820–0x83C | `DBG_BP0`–`DBG_BP7` | RW | IA breakpoints; NBP of them exist (U200: 4), the rest read 0 and ignore writes | PD2 |
| 0x840–0x85C | `DBG_IR0`–`DBG_IR7` | R | Words w0–w7 of the last popped instruction (below) | PD2 |
| 0x860–0x868 | `DBG_OPA1`–`DBG_OPA3` | R | Its resolved addresses A1–A3 (`R[x] + imm`, 32-bit wrap) | PD2 |
| 0x86C | `DBG_UNIT` | R | [2:0] phase (0 idle, 1 boundary, 2 checks, 3 issue, 4 unit running, 5 drain, 6 halted), [6:4] unit holding the instruction (0 none, 1 DMA, 2 QACT, 3 MVM, 4 VU, 5 scalar; 6 reserved, was WAITM), bit8 next instruction fetched, bit9 fetch idle, [15:12] unit idle (DMA, QACT, MVM, VU), [23:16] beats in the fetch buffer | PD2 |
| 0x870–0x87C | `DBG_RDOUT0`–`DBG_RDOUT3` | R | Outstanding reads per channel, 8 bits each: channel c is byte c mod 4 of `DBG_RDOUT[c/4]` | PD2 |
| 0x880 | `DBG_WROUT` | R | [7:0] DMA write bursts waiting for their B response | PD2 |
| 0x884–0x89C | `DBG_WATCH*` | RW | Two watch ranges: 0x884 kinds (SPM, or DRAM channel c), 0x890/0x894 range 0 start/end, 0x898/0x89C range 1 | PD3 |
| 0x8C0–0x8FC | `DBG_R0`–`DBG_R15` | RW | Integer registers. Writes take effect only while halted, R1–R15 only | PD2 |
| 0x900–0x93C | `DBG_S0`–`DBG_S15` | RW | f32 scalar registers (bit patterns); writes only while halted | PD3 |
| 0x940–0x95C | `DBG_HIST0`–`DBG_HIST7` | R | IAs of the last 8 retired instructions (0 = most recent) | PD4 |
| 0x980 | `DBG_SPM_LINE` | RW | SPM window line address | PD2 |
| 0x984 | `DBG_SPM_CMD` | W | bit0 READ (line → `DBG_SPM_DATA*`), bit1 WRITE (`DBG_SPM_DATA*` → line). The CSR write completes after the SPM operation | PD2 |
| 0x988, 0x98C | `DBG_SPM_BE_LO`, `_HI` | RW | Byte mask of WRITE (reset value all ones) | PD2 |
| 0x9C0–0x9FC | `DBG_SPM_DATA0`–`15` | RW | 64 B line buffer; `DBG_SPM_DATA<i>` = bytes 4i … 4i+3 | PD2 |

**`DBG_IR*` and `DBG_OPA*` after a run ends.** They belong to the last instruction popped into
the check pipeline. w0–w6 come from cmdproc's `ir_q`; w7 is captured by the debug block (+32
FF), since `ir_q` keeps only a nonzero flag for it. After ABORT or end of buffer nothing was
popped, so they are the previous instruction's. After a fetch-range error the pop loads the
range-error entry, so `DBG_IR*` read 0 (the empty instruction the fetch unit delivers with the
error) while `DBG_OPA*` still hold the previous instruction's addresses. Before the first pop
after START both are undefined. (Alternative, rejected for cost: a separate IR copy in the debug
block loaded only without a range error, 256 FF.)

### 3.3 Halted-state rules

| Item | Rule |
|---|---|
| STATUS | BUSY \| HALTED (bit4, mirrors `DBG_STATUS.HALTED` so the host's wait loop needs one read). DONE and `done` (U200 `ap_done`) stay off: the run has not ended. The STATUS W1C ignores HALTED, as it ignores BUSY and the reserved bit3. (M8) A halted CALL leaves STATUS unchanged: HALTED mirrors host runs only, and the CALL's halt shows in `DBG_STATUS.HALTED` and `CTL_STATUS.CALL_BUSY` (spec "명령 버퍼 경로와의 공존"). |
| Fetch | The fetch unit may keep prefetching ch0 until its buffer (16 beats) is full. |
| DBG_CTRL | One bit per write is taken: STEP > RESUME > HALT. RESUME releases the halt. STEP runs one instruction and halts at the next boundary; if that instruction is END or faults, the run ends normally. RESUME and STEP while not halted are ignored; HALT while not BUSY or already halted is ignored. When a run ends (END, error, ABORT), pending HALT and STEP requests are cleared, so `HALT_PEND` never outlives its run. |
| ABORT | Ends the run at that boundary: `ERR_CODE` = `ABORTED`, `ERR_PC` = `DBG_NEXT_PC`, `ERR_INFO` = 0, `PC` unchanged. |
| SOFT_RESET | As SOFT_RESET while BUSY: the run ends at that boundary like an ABORT, then the reset applies and STATUS becomes 0. |
| START | Ignored (BUSY). (M8) During a halted CALL a host START is not ignored: it waits (`CTL_STATUS.START_WAIT`) and starts after the CALL ends. |
| Registers, SPM | Writes through `DBG_R*` and `DBG_S*` take effect only while halted; SPM window writes while halted or not BUSY (§3.5). The next instruction sees them. `DBG_R0` writes are ignored. |
| DRAM | The host may read and write DRAM while halted; the next instruction sees the writes as if made at the boundary. The running command buffer is the exception: it may already be prefetched. |
| PERF | Time counters (CYCLES, MVM_CYCLES, VU_CYCLES, CH_WAIT[c]) do not count while halted, so `PERF_CYCLES` becomes "BUSY and not HALTED". Event counters (RETIRED, CH_RD_BEATS[c]) count real events; the only event possible while halted is ch0 prefetch. A run with halts therefore ends with the same results, `PERF_RETIRED` and `PERF_CH_RD_BEATS[c ≥ 1]` as without. |
| Kept across runs | START clears HALT/STEP requests, CAUSE, BP_HIT, `DBG_RETIRED` and the IA history. `DBG_CFG`, `DBG_BP*`, `DBG_RBP` and the watch ranges survive START and SOFT_RESET (the debugger's setup); only a hardware reset clears them. |
| Latency | A HALT takes effect when the instruction in flight ends: up to the longest MVM, e.g. the output head, 569,845 cycles (1.90 ms) on Qwen2.5-0.5B and about 3.23 M beats per channel (≈ 10.8 ms at 300 MHz) on Qwen2.5-32B. No upper bound on the halted time. A unit that is stuck never reaches the boundary, so the core does not halt; §3.4 and §4.4 diagnose that case. |

### 3.4 Always-readable state and post-mortem

- Every `DBG_*` read is valid mid-run and reflects that cycle.
- A hang is diagnosed from `PC`, `DBG_UNIT`, `DBG_RDOUT*`, `DBG_WROUT`, `DBG_STATUS.HALT_PEND`,
  and two reads of `PERF_*` some time apart (§4.4).
- After a run ends (END, error, ABORT), R1–R15, `DBG_IR*`, `DBG_OPA*` and the SPM stay as they
  were until the next START (SOFT_RESET clears R). A faulting instruction changes no state, so
  this is the state just before the error; that is why there is no halt-on-error.

### 3.5 SPM window

- Read: write `DBG_SPM_LINE`, write `DBG_SPM_CMD` = READ, read `DBG_SPM_DATA0..15`. Write:
  write `DBG_SPM_LINE`, `DBG_SPM_DATA0..15` (and the byte mask if not all ones), then
  `DBG_SPM_CMD` = WRITE; only bytes with their `DBG_SPM_BE` bit set change.
- It works only when not BUSY or while halted. In any other state, for a line outside the SPM,
  or when the port is not granted within 1024 cycles, it does nothing and sets `SPM_ERR`; a CSR
  access therefore always completes.
- (M8) With the control core the condition is: (no command-buffer run, host or CALL, is executing,
  or that run is halted) and `CTL_STATUS.QQUIESCENT` and (RV_RUN = 0, or the RISC-V is halted and
  no abstract command is executing). With RV_RUN = 0 and nothing queued this is the rule above.
- RTL: a fourth `tsr_spm` client (`TSR_SPM_NCL` 3 → 4), the lowest-priority one with
  `HAS_CTRL = 0`. With `HAS_CTRL = 1` client 3 is shared 2:1 with the RISC-V SPM bridge, the debug
  window first, and `tsr_spm` arbitrates round robin (`docs/ctrl-riscv.md` §5.3). The read data bus
  is shared and qualified per client, so the response lands in the 512-bit line buffer.
- Normal paths (backend, traces) never use it.
- Tools compare only lines this run wrote: the simulator starts SPM at 0xA5, the hardware's
  initial SPM is undefined.
- Speed: 18 CSR accesses per line (LINE, CMD, 16 DATA). At an assumed, unmeasured 1 µs per PCIe
  access, the whole 4 MiB SPM takes about 1.2 s; dump only what is needed.

### 3.6 RTL hooks and parameters

- `tsr_csr`: route 0x800–0x9FF to `tsr_dbg` (one more read pipeline stage is fine, the CSR port
  is latency-insensitive); STATUS bit4; gate the time-counter increments with `halted`.
- `tsr_cmdproc`: after the ABORT check in `S_NEXT`, `dbg_stop` → new state `S_HALT`; a resume
  pulse returns to `S_NEXT` with the skip flag; `run_abort` in `S_HALT` takes the existing ABORT
  path. Adds: the IA (computed from `CMD_ADDR` and the next pc), the per-run retire counter, a
  read port on `rf_rd` and a write port used only while halted, the w7 capture, and observation
  wires for phase, unit and A1–A3.
- `tsr_spm`: client 3. `tsr_mem_arb`: outstanding-read counts. `tsr_dma`: B responses
  outstanding. `tsr_fetch`: occupancy, idle, valid.
- `dbg_trig`: a registered one-cycle pulse inside `tsr_dbg` (halt, or a TRIG_ONLY match). It is
  not a core boundary port: the U200 debug build probes it as an internal net (§7.2).
- Parameters: `HAS_DBG` (U200 1), `HAS_DBG_SPM` (1), `DBG_NBP` (4), `HAS_TRC` (0; §6.3).
  `HAS_DBG` = 0 builds nothing and every `DBG_*` offset reads 0. The spec's 구성 파라미터 table
  gets these rows when PD2 lands.
- Keep `tsr_dbg` a separate module with small hooks: PD2 lands with the M4 timing work in the
  same files (cmdproc, csr, spm).

### 3.7 Cost (estimates, before synthesis)

- FF ≈ 1.0 K: BP 128, IA / RETIRED / RBP 96, SPM line buffer 512 + byte mask 64, w7 32,
  control ≈ 150.
- LUT ≈ 1.6 K: read mux (≈ 48 words × 32 bits) ≈ 0.7 K, SPM buffer load ≈ 0.3 K, register
  read/write ≈ 0.2 K, comparators and counters ≈ 0.2 K, control ≈ 0.2 K. No BRAM, URAM or DSP;
  the SPM 3:1 → 4:1 mux fits the existing LUT6s.

| Device | LUT | FF |
|---|---|---|
| U200 VU9P (1,182 K LUT, 2,364 K FF) | 0.14 % | 0.04 % |
| ZU3EG | 2.3 % | 0.7 % |
| K26 (ZU5EV) | 1.4 % | 0.4 % |
| Z-7020 | 3.0 % | 0.9 % |

`HAS_DBG_SPM` = 0: FF ≈ 0.4 K, LUT ≈ 1.1 K. Timing: the read mux is absorbed in the CSR read
pipeline and the halt decision is a register output, so the cmdproc paths do not get deeper.

### 3.8 Verification of the debug block

- **Simulator model** (PD1; the simulator is the reference for the block's semantics):
  - Every regression trace (`traces/m2`, `gen`, `m3`, `model`) replayed with a STEP at every
    instruction ends exactly like the replay without halts.
  - The state after k STEPs (SPM, DRAM) equals the state after a prefix run with
    `CMD_COUNT` = k: two independent mechanisms cross-check each other.
  - On `shadow:mock` with `TSR_SIM_FAULT`, the divergence finder names the injected instruction
    and line exactly.
  - Directed unit tests of the halted-state rules.
- **RTL directed** (`tb/cmdproc`, `tb/core`): HALT during an MVM (the MVM completes first);
  HALT and ABORT in the same cycle; ABORT, SOFT_RESET and START while halted; BP at pc 0, at
  `CMD_COUNT`, at END and before an out-of-range fetch; STEP that ends in an error or END; RBP = 0;
  the skip rule; TRIG_ONLY; several `DBG_CTRL` bits in one write; HALT racing the run's own end
  (`HALT_PEND` cleared); writes ignored while not halted; SPM window refusal and timeout; time
  counters stopped while halted; STATUS mirror and W1C; `HAS_DBG` = 0 reads 0.
  **At M5** (MVM overlap): HALT and STEP between two overlapped MVMs (drain rule, §3.1).
- **RTL random**: the program generator times a random halt schedule (cycle HALTs, BPs, RBPs,
  STEP counts), on the `m3_abort` infrastructure. At every halt, compare with the simulator
  model advanced to the same `DBG_RETIRED`: R0–R15, `DBG_NEXT_PC`, `DBG_IA`, `PC`, CAUSE,
  BP_HIT, written SPM lines. Final results, `PERF_RETIRED` and `PERF_CH_RD_BEATS[c ≥ 1]` must
  equal the run without halts. `HAS_DBG` 1 and 0 builds give the same trace results and the same
  cycles per instruction.
- **Mutation**: drop the skip rule, swap the check order, drop the PERF gating; the tests must
  catch each.
- **fastreplay** `--dbg-step` over every trace set; **vkernel** `tsr-dbg` batch scripts in
  ctest, with a test-build-only RTL fault-injection define that flips one MVM output bit, which
  the divergence finder must name.
- **Hardware** (M4): a debug section in `tsr_xrt_selftest`: step its own programs instruction by
  instruction against the simulator; SPM window round trip; DRAM read/write while halted; a
  timeout while halted must not write ABORT.

---

## 4. Software debugger

### 4.1 Transport additions (PD1)

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** The v2 transport API is EX §10;
`tsr_transport_csr_write` is part of it. The exec-based additions below (halt handler, prefix runs,
exec records) have no v2 counterpart; DB §9.2 lists what DB needs from the transport.

Proposed C API on top of `tsr_transport.h` (part C's file; changed only after part C merges):

```c
int  tsr_transport_csr_write(tsr_transport * t, uint32_t off, uint32_t val);

typedef struct tsr_exec_times { uint64_t t_call_ns, t_started_ns, t_done_ns, t_ret_ns; } tsr_exec_times;
typedef struct tsr_exec_hooks {
    uint32_t poll_ms;                                          // 0: plain blocking wait
    void *   ud;
    void   (*sample)(void * ud, tsr_transport * t, uint64_t t_ns);   // PC samples (§5.3)
} tsr_exec_hooks;
int  tsr_transport_exec_timed(tsr_transport * t, uint64_t cmd_addr_beat, uint32_t cmd_count,
                              const uint32_t args[8], tsr_exec_result * res,
                              tsr_exec_times * tm, const tsr_exec_hooks * hk);
int  tsr_transport_perf_read(tsr_transport * t, uint32_t first, uint32_t n, uint64_t * out, int lo_only);

// Called in the exec thread when STATUS.HALTED is seen; writes RESUME, STEP or ABORT and returns.
typedef void (*tsr_halt_fn)(void * ud, tsr_transport * t);
void tsr_transport_set_halt_handler(tsr_transport * t, tsr_halt_fn fn, void * ud);

// Program map of the next exec (JSON); the sim transport writes it as nodes.json into that
// exec's TSR_TRACE directory, shadow and dumps keep it too.
void tsr_transport_set_annotation(tsr_transport * t, const char * json);

// Observe-only open (§4.4): never writes CTRL, AP_CTRL or ADAPTER_STATUS, never reads AP_CTRL.
tsr_transport * tsr_transport_open_ex(const char * kind, unsigned flags);   // TSR_OPEN_OBSERVE
```

The existing `tsr_transport_exec` becomes `exec_timed(…, NULL, NULL)`. All PERF reads of one
process happen in one thread under the backend's device mutex (one LO snapshot slot).

### 4.2 Waiting, halts, timeouts, signals, exit

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** v2 waits are EX §4.8 and SW §9.1: the
timeout counts only while the RISC-V is not halted and no tile stop is active, the transport mutex
is released between polls, and a timeout follows DB's recovery ladder (DB §7.3).

- **Wait loop.** Only when the debugger or PC sampling is on, the kdev wait becomes a loop of
  short waits (≤ 50 ms) that reads STATUS (one read sees HALTED). Otherwise the wait is
  unchanged and costs nothing.
- **Halt handler.** When the loop sees HALTED it calls the handler (the debugger REPL) in the
  exec thread. The handler writes RESUME, STEP or ABORT and returns; the exec keeps waiting.
  Running the REPL in the exec thread keeps the kdev rule "one caller at a time" and the single
  PERF snapshot slot intact.
- **XRT mode.** With `TSR_DEBUGGER` set, the xrt transport uses `TSR_XRT_CTRL=regs` (the
  transport writes the arguments and AP_CTRL and polls `ap_done`, no scheduler command). Reason:
  under the scheduler, a process that dies while its run is outstanding makes KDS abort the
  command, and a halted compute unit never finishes, so the card goes to bad state and needs
  `xbutil reset` (§1); every bit of debug state is lost. In register mode nothing is outstanding
  at the driver, so a killed process leaves the halted core and its state for a dump (§4.4).
  Whether register access works during an outstanding `xrt::run` at all is a spec open item;
  today's run-mode timeout path (CTRL.ABORT through the window, `tsr_kdev.h:104-105`) and PC
  sampling in run mode depend on it too.
- **Timeouts.** `TSR_XRT_TIMEOUT_MS` counts non-halted time only; a halted core never gets a
  timeout ABORT (today's kdev timeout path writes ABORT regardless, so this changes in PD2 before
  the RTL block exists on a card). Before any timeout ABORT, the hang dump is written (§4.4).
- **SIGINT.** The handler is installed only while an exec wait runs with `TSR_DEBUGGER` set, and
  the previous handler is restored when the wait ends (`sigaction`, saving the old action). The
  first SIGINT while not halted only sets a flag; the wait loop then writes `DBG_CTRL.HALT`. A
  SIGINT while halted, or a second one, is passed to the previous handler, so llama.cpp's own
  interrupt and shutdown handling keeps working.
- **Exit.** On a normal exit while halted, the transport writes ABORT so the next process finds
  an idle core. SIGKILL cannot be handled; in register mode the core stays halted, and the next
  normal open aborts it, so a dump must come first.

### 4.3 Program map and `nodes.json`

- The backend records, per `MUL_MAT`, its first instruction index (`insns.size()` before
  `tsr_mm_emit`) in `tsr_prog.mm_first`, so cached programs share it. At exec time it combines
  these ranges with the current segment's node and weight names, taken before `tsr_flush`
  clears `seg.names`.
- Entry fields: pc range, node name, weight tensor name, layer, role, op, fmt, N, K, T, tile
  index, expected beats per channel (`rows_c × ROW_BEATS` for MVM, `n_lines` for LD/ST),
  model cycles, token / split / exec numbers.
- Layer and role come from the llama.cpp node name `<role>-<layer>` (e.g. `ffn_gate-12`); the
  output head is `result_output`; if the name does not parse, the weight name
  `blk.<layer>.<role>.weight` is used.
- Piece runs (a program larger than a CMD half, the `split` mode, halt emulation): device pc =
  program pc − piece start + prologue length; the map of that exec stores the offset, so a
  device pc maps back to the program pc.
- Traces: the annotation call (§4.1) makes the sim transport write `nodes.json` into the trace
  directory it chose for that exec. `remove_stale` in `tsr_trace.c` is extended to remove
  `nodes.json` and `actual/`. Replayers ignore both, so existing traces and replayers are
  unchanged.
- Users: failure messages (ERR_PC → node and layer), `where`, breakpoints by node, dumps, and
  every profiler report.

### 4.4 Post-mortem and hang dumps

`TSR_DUMP_DIR=<dir>`: on an exec error, a timeout, an adapter error or a shadow mismatch, the
transport writes `<dir>/<exec seq>/` before the next START:

- `state.json`: every core CSR (raw and decoded), `DBG_*` if present, `PERF_*`, adapter registers
  (U200 ADAPTER_STATUS), the reason, device and build identity.
- `cmd.bin`, `nodes.json`; with the profiler on, the last exec records (a software flight
  recorder).
- SPM (`TSR_DUMP_SPM=used|all|none`, default `used` = operand lines of the instructions this
  run executed): through the SPM window, or without the debug block by an ST dump program (§4.6).
- DRAM regions the program reads and writes. v1 programs address DRAM from R0 only, so a
  simulator dry run of the same program gives the regions; from M8 (RLD) addresses depend on
  data and the real run's registers are needed.

**Timeouts.** Before the ABORT: PC, STATUS and PERF are read twice, 100 ms apart, to record
whether the run was progressing. After the ABORT the SPM is read only if the core actually
stopped. If it did not (a stuck AXI transaction), the transport already refuses every further
exec (`docs/xrt-host.md` §3); the dump then holds CSRs only (PC, PERF, ADAPTER_STATUS, `DBG_*`)
and says that the SPM was not read.

**Hang triage without the debug block** (LO deltas of two reads; `f` = core clock):

| STATUS | PC, RETIRED | ΔMVM_CYCLES | ΔCH_RD_BEATS[c] | ΔCH_WAIT[c] | Reading |
|---|---|---|---|---|---|
| BUSY | advancing | – | – | – | Progressing, maybe slow: compare with the cycle model |
| BUSY | fixed | ≈ Δt · f | 0 | > 0 on c | The MVM waits for read data on channel c (AXI read stalled) |
| BUSY | fixed | ≈ Δt · f | 0 | 0 | MVM internal stall, no read outstanding |
| BUSY | fixed | 0 | 0 | > 0 on c | LD or fetch read stalled (opcode at PC; on ch0 maybe fetch) |
| BUSY | fixed | 0 | 0 | 0 | ST write response, QACT or the command processor; `DBG_UNIT` and `DBG_WROUT` (PD2) decide |
| not BUSY, no `ap_done` | – | – | – | – | The core finished but the completion was lost: adapter, XRT or interrupt path |

**Observe-only dump.** `tsr-dbg dump --device <kind>` opens the transport observe-only: it never
writes CTRL, AP_CTRL or ADAPTER_STATUS and never reads AP_CTRL (clear on read); it reads the core
CSR window and ADAPTER_STATUS. The normal open would destroy the evidence (it ABORTs a BUSY core,
consumes `ap_done`, clears ADAPTER_STATUS). This gives a post-mortem of a process that died with
the core halted or hung, provided that process used register mode (§4.2). To confirm on the
card: that opening the compute unit in a new process does not disturb it.

### 4.5 Shadow transport

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** The per-exec shadow is deleted. v2 compares
whole launches with the simulator (`tsr-selftest` with its in-process simulator shadow, SW §7.6) and
replays traces through the `replay` kernel.

`TSR_DEVICE=shadow:<kind>` (kind = `xrt`, `mock`, `vkernel` or another registered kind; the full
`TSR_DEVICE` set is `sim | xrt | mock | <registered kind> | shadow:<kind>`):

- Every write and exec goes to the primary device and to an ISA simulator.
- Per exec, compared bit for bit: STATUS and ERR_*; the regions the simulator wrote with ST
  (read back from the primary); ΔPERF_RETIRED; ΔPERF_CH_RD_BEATS[c ≥ 1]; channel 0 by the ch0
  equation of §5.7 when it applies. With NUM_CH = 1 (k1, the M3 gate bitstream) channel 0 is the
  only beat check.
- On a mismatch the exec is written as a trace to `TSR_SHADOW_DIR`: init, expect (simulator),
  `actual/ch<c>_<offset>.bin` (what the primary wrote), `nodes.json`. `TSR_SHADOW_STOP=1` stops
  the process at the first mismatch.
- It finds the first wrong exec of a real llama.cpp run, so it is the first M3 bring-up tool.
- Cost: the simulator's DRAM (32B needs file-backed DRAM, about 20 GB of weights) and speed; on
  32B a few tokens are the practical limit.

### 4.6 Devices without the debug block

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** Prefix runs and halt emulation need command
buffers and are deleted. v2 bisects with tile stop points at push indices (DB §5).

`DBG_ID` = 0: bitstreams before PD2, or `HAS_DBG` = 0. Existing CSRs and runs answer most of the
same questions; the tools always say that they run in this mode.

- **Stop before instruction k**: a prefix run with `CMD_COUNT` = k from the program's own start
  (pc 0 starts at a beat boundary, so no copy is needed). It ends with `ILLEGAL_OP`,
  `ERR_PC` = k; SPM and DRAM keep the state after k instructions.
- **Read SPM**: an ST-only program in the free CMD half copies the wanted lines to a scratch DRAM
  area (IO staging) outside the trace's regions; the host reads it. Works because START keeps
  SPM (spec 구현 명확화). Needs a core that stopped.
- **Resume / run a piece**: instructions k … end cannot start in place: instruction pc sits at
  beat `CMD_ADDR + ⌊pc/2⌋`, so a run cannot start at an odd pc. The tool copies the piece to the
  other CMD half (the backend's `tsr_run_range` path) or a scratch CMD area, beat-aligned, with
  an `LI` prologue for registers and END. It never patches the running command buffer or a
  cached program. R1–R15 are not readable without the block, so the prologue loads the values
  the simulator reports; v1 lowering uses only R0, so its prologue is empty.
- **Limits**: registers are the simulator's, not the hardware's; extra run boundaries change the
  timing (prefetch, DRAM page state).

### 4.7 Divergence finder (`tsr-dbg diverge <trace>`)

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** Replaced by DB §9.4 (`tsr-dbg diverge`),
which bisects by push index with tile stop points and compares against the simulator.

1. Run the simulator to the end: instruction count n and the write set W_i of every retired
   instruction (from the instruction hook).
2. **With the debug block**: load the trace on the device and halt at RBP checkpoints
   k₁ < k₂ < … (default: node range ends from `nodes.json`; otherwise doubling steps). At each
   halt advance the simulator to the same retire count and compare R1–R15, `DBG_NEXT_PC`,
   `DBG_IA`, `PC`, "no error", the SPM lines (window) and DRAM beats written since the last
   checkpoint. For the first bad interval (k_{j−1}, k_j], reload, halt at k_{j−1} and STEP,
   comparing write sets, to the first wrong instruction i. An error or hang on the device bounds
   the search.
3. **Without the debug block**: bisection with prefix runs. **Before every probe the tool
   reloads `cmd.bin` and the init regions**: within a split, ggml-alloc reuses IO memory, so a
   longer run may have overwritten the inputs of earlier instructions, and probing on top of it
   reports false divergences. Probes read SPM with the ST dump program. For a live llama.cpp run,
   the shadow transport first captures the failing exec as a trace; the search runs on that
   trace.
4. **Report**: i, PC, IA, disassembly, resolved addresses, node and layer; the first wrong
   element (line, f32 index, expected and actual, ulp); for an MVM, element → output row →
   channel c = ⌊row / R⌋ → cluster c.
5. **`repro/`**: a one-instruction trace in the existing format: a prologue that restores the
   source SPM lines with LD, `LI` for the registers it reads, the instruction, END. It replays
   unchanged in fastreplay (`--waves`), cocotb and `tsr_xrt_selftest --traces`.
6. A mismatch that disappears when halts are inserted is reported as such; the next steps are
   prefix runs without halts, or TRIG_ONLY + ILA.

For scale: one 0.5B token is 145 execs and 821 instructions, a few instructions per exec, so
stepping is cheap; from M7 (one split per token, thousands of instructions) the checkpoints
matter.

### 4.8 `tsr-dbg`

- Forms: `tsr-dbg trace <dir> --device <kind>` (owns the device); `tsr-dbg attach <socket>`
  (joins the debugger inside a backend process); `dump`, `diverge`, `repro`.
- Execution: `run`, `c`, `si [n]`, `halt` (Ctrl-C), `abort`, `reset`, `q`.
- Breakpoints: `b <pc | *ia | node <regex> | ret <n>>`, `tb` (trigger only), `info b`, `d`.
- State: `info reg | unit | perf` (perf since run start).
- Memory: `x/<n><f> spm:<line> | ch<c>:<beat>`; formats hex, f32, f16, i8, QA block (QS, D, BS
  decoded), weight block.
- Program: `disas` (`tsr_insn_str` moved into a shared library), `where` (node, layer, split,
  exec, token).
- Writes and checks: `set R<i>=<v>`, `set spm:<line> …`, `compare` (current state against the
  simulator at the same retire count), `dump <dir>`.
- Batch mode for CI and agents: `-ex`, `-x <script>`, `--json` (one JSON line per command).
- In the backend's REPL, `p <node>` prints that node's dst or src1 from the IO window as f32.

### 4.9 Debugger environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `TSR_DEBUGGER=1` | backend | Open a REPL on `/dev/tty` at a halt or an exec error (post-mortem), in the exec thread |
| `TSR_DEBUGGER=sock:<path>` | backend | The same over a Unix socket for `tsr-dbg attach` (llama-server, no terminal) |
| `TSR_DEBUGGER_BREAK=node:<re>;layer:<n>;exec:<n>;tok:<n>;pc:<n>;ret:<n>` | backend | Conditions turned into IA breakpoints per exec (IA = 2 × CMD start beat + the node's first pc) or an entry RBP (`exec`, `tok`: RBP = 0). Without the block: prefix runs |
| `TSR_DUMP_DIR=<dir>`, `TSR_DUMP_SPM=used\|all\|none` | transport | Dumps (§4.4) |
| `TSR_DEVICE=shadow:<kind>`, `TSR_SHADOW_DIR`, `TSR_SHADOW_STOP=1` | transport | Shadow transport (§4.5) |
| `TSR_SIM_FAULT=<retired>:<spm line \| ch:beat>:<bit>` | sim | Test only: flip one bit after that retire count |

The names avoid `TSR_DBG`, which is the backend's logging macro next to `TSR_DEBUG` (log level).
With all of them unset the backend touches no debug CSR.

---

## 5. Profiler

### 5.1 Exec record and PERF deltas

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** v2 records launches, not execs: one answer
record per launch with its DURATION (EX §4.4), PERF gated by ACTIVE so that the difference of two
reads is one launch (EX §6), and TSR_STATS v2 (DB §8.6).

Code: the recorder library `sw/prof/` (`libtsr_prof`, C++ without dependencies: exec-record
ring, host events, streaming JSON and Chrome-trace writers), linked by the transport and the
backend. The program map is built in `sw/ggml-tesseract` (§4.3) and written by the recorder.

One record per `tsr_transport_exec`:

- Host times (ns, `CLOCK_MONOTONIC`): `t_call`, `t_started` (start returned), `t_done` (wait
  returned), `t_ret`.
- CMD beat, instruction count, final STATUS / ERR_*, exec seq, token and split numbers.
- PERF deltas. Counters do not move between runs (spec 구현 명확화), so the value after exec n
  is the value before exec n + 1: one read set per exec.
- **LO tracking.** At open the transport reads every counter as 64 bits (LO, then HI). After each
  exec it reads only the LO halves and adds `(LO_now − LO_prev) mod 2³²` to its 64-bit copy. A
  counter grows by at most 1 per cycle and a run's BUSY time is shorter than its host time, so
  this is exact while the exec's host time is below 2³² / f_core. With margin: an exec whose
  host time exceeds 0.5 × 2³² / f_core (7.2 s at 300 MHz, 4.3 s at 500 MHz) is followed by a full
  64-bit re-read. After any SOFT_RESET the transport issues (e.g. on a timeout) it re-reads too.
- Spans: the program-map entries of this exec (§4.3), each with expected beats, model cycles and,
  when measured, device cycles and the attribution mode.
- Optional PC samples (§5.3).
- Records go to a ring and stream to disk; `TSR_PROF_EVERY=n` keeps full detail for every n-th
  token and totals for the rest.

**Scope and cost.** Records and the self-check (§5.7) exist only when `TSR_PROF` or a shadow
transport is on. On the card each read is a PCIe MMIO read, unmeasured; at an assumed 1 µs and
the minimum set (§5.10: 7 counters on k4), 0.5B costs 145 × 7 µs ≈ 1.0 ms per token against
7.22 ms of modelled device time (≈ 14 %), while 32B (≈ 385 Tesseract splits per token,
`docs/llama-pin.md`) costs ≈ 2.7 ms per token (< 1 % at ≥ 1 s per token). Milestone tok/s is
measured with the profiler off; breakdowns come from a separate run.

### 5.2 Host events and token boundary

- Events with time and bytes: `graph_compute` entry and exit, `set_tensor` / `get_tensor` /
  memset, staging writes and reads, lazy weight conversion, program-cache hit and miss.
- Gaps between consecutive Tesseract events are shown as "other backends" (CPU). llama.cpp has
  no split or op timer, and `cb_eval` is not used (§8).
- Token boundary: by default the end of the split that contains `result_output`. Tools can mark
  it explicitly through the proc address `ggml_backend_tesseract_prof_mark(const char * name)`
  (`ggml_backend_reg_get_proc_address`), e.g. numcheck and a llama-bench wrapper.

### 5.3 PC samples

- kdev transports, `TSR_PROF_SAMPLE_MS` ≥ 1: in each turn of the short-wait loop (§4.2) the
  transport reads `PC` and the LO halves of `PERF_MVM_CYCLES` and
  `PERF_CH_RD_BEATS[NUM_CH − 1]`. With NUM_CH = 1 (k1) that is ch0, fetch beats included.
  This shows the progress of long MVMs (the 32B output head runs ≈ 10.8 ms) and where a hang
  sits.
- The simulator cannot be sampled (no CSR access may race `tsr_sim_run` except ABORT); it uses
  the instruction hook instead.

### 5.4 Per-instruction attribution

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** `split` attribution and instruction-index
modes are deleted. v2 attributes time to ops through the trace writer's records and NODE markers (DB
§8.3–§8.4).

| Mode | Devices | Value | Perturbation |
|---|---|---|---|
| `model` (default) | all | Simulator: the cycle model, exact for the model. Measured devices: an estimate (`est`): the exec's ΔMVM_CYCLES is split over its MVM spans by max-channel weight beats, ΔCYCLES − ΔMVM_CYCLES over the other spans by model ratio | none |
| `split` | all | The exec runs as pieces (tile cuts or single instructions); each piece is copied to the other CMD half with END (§4.6) and gets its own PERF delta. Device cycles are measured. Single-instruction pieces subtract the fixed cost of an END-only run, measured on the device | more execs; never used for wall-clock metrics |
| `retire` | vkernel, fastreplay | A tb-only probe `tb/common/hdl/tsr_prof_probe.sv` bound to `tsr_core` (not synthesized, no `rtl/` change), passing per instruction the retire cycle and pc; DMA / QACT / MVM occupancy (handshake to done); fetch wait (`insn_ready && !insn_valid`); issue gap (BUSY with no unit holding an instruction); per-1024-cycle beats and waits; via DPI-C | none (RTL simulation) |

`split` relies on SPM surviving START, so it is used only for programs that do not read R1–R15
or that get an `LI` prologue (v1 lowering uses only R0). The expected values of a piece count
its prologue and END: ΔRETIRED = prologue + n + 1, ch0 fetch beats ⌈(prologue + n + 1) / 2⌉
(the piece starts at a beat boundary). Fixed per-instruction costs are deterministic in the RTL
(NOP/LI 4, LD/ST 10 + unit, MVM checks ≈ 46), so vkernel values transfer to the card; memory
dependent DMA time is measured with `split`.

### 5.5 Device sources

| Device | Device time | Per instruction | Notes |
|---|---|---|---|
| sim | cycle model | instruction hook | Host times measure the simulator; card estimates come from `tsr-prof predict` |
| mock | cycle model + adapter model | instruction hook | Exercises the kdev path (samples, LO reads, times) |
| vkernel | RTL cycles | `retire` | Memory regimes rand / fast / slow, plus a `ddr4` regime fitted to M0 (first-beat latency distribution and channel bandwidth). About 160 K cycles/s: a 0.5B token ≈ 15 s |
| fastreplay | RTL cycles (core only) | `retire` | `--prof out.json` over a trace set |
| xrt (U200) | measured PERF through the CSR window | `model`, `split`, PC samples | f_core is the xclbin's kernel clock; a long exec whose ΔCYCLES / (t_done − t_started) differs by more than 2 % raises a warning |
| zynq / Kria | measured PERF (PS AXI-Lite) | as xrt | Window reads are cheap: read every counter |
| ASIC | measured PERF over JTAG or SPI | `model` | Slow access: the minimum set and `TSR_PROF_EVERY` |

### 5.6 Metrics and reference values

- Device share **D** = ΣΔPERF_CYCLES / f_core / wall time.
- Exec overhead = (t_ret − t_call) − ΔPERF_CYCLES / f_core.
- **G** = 1 − ΣΔPERF_MVM_CYCLES / ΣΔPERF_CYCLES: the share of device time spent outside MVMs,
  i.e. the upper bound of what overlapping non-MVM instructions with MVMs could hide. Its
  wall-clock share is G × D.
- **F** = 1 − Σ max_c(rows_c × ROW_BEATS) / ΣΔPERF_MVM_CYCLES: MVM fill, broadcast and tail
  inside `PERF_MVM_CYCLES`. This is what M5's MVM-to-MVM overlap hides, and G does not include
  it. Small for dense decode; large for short MoE expert matrices (spec: 1–2 µs of fill per MVM).
- Effective bandwidth of channel c: BW_c = 64 B × ΔCH_RD_BEATS[c] × f_core / ΔCYCLES; channel 0
  minus the ⌈CMD_COUNT / 2⌉ fetch beats when the ch0 equation (§5.7) applies.
- **SOL%** = BW_c / M0 measured channel read bandwidth (`docs/m0-bw.md`; `perf/u200-m0.json`).
- Wait ratio W_c = ΔCH_WAIT[c] / ΔCYCLES; channel imbalance = max_c / mean_c beats.
- MVM cycles per beat = ΔMVM_CYCLES / max_c(rows_c × ROW_BEATS) (1.00 for large MVMs on ideal
  memory, `docs/rtl-m3.md` §13).

Reference values (2026-10-23; Qwen2.5-0.5B Q8_0, prompt "Hi", one token,
`traces/model/qwen05b-q8_0`, NUM_CH 4, simulator cycle model): 145 execs, 821 instructions,
2,166,429 cycles (7.22 ms at 300 MHz), 2,080,256 MVM cycles → **G = 3.98 %**; max-channel weight
beats 2,065,056 → **F = 0.73 %**. RTL, fast memory regime: G ≈ 8.4–8.6 % (rtl-m3 §13 gives
2.08 M MVM of 2.27 M total cycles; a replay on the current core build during the planning review
gave 8.56 %).

| Shape (per token) | Execs | Share of device time | MVM share inside |
|---|---|---|---|
| gate + up (K 896, N 4864, two MVMs) | 24 | 41.8 % | 97.0 % |
| output head (K 896, N 151936) | 1 | 26.8 % | 98.3 % |
| down (K 4864, N 896) | 24 | 21.2 % | 95.4 % |
| q / o (K 896, N 896) | 48 | 8.3 % | 91.6 % |
| k / v (K 896, N 128) | 48 | 1.8 % | 67.6 % |

### 5.7 Self-check

Done on every exec while the profiler or a shadow transport is on; a violation is marked in the
record and fails `tsr-prof check`. On the card it also catches kernel misbehaviour.

- ΔPERF_RETIRED = ERR_PC for a run that ended with an error (the instructions before the
  faulting one; the faulting one does not count), and END's pc + 1 for a DONE run (END counts);
  programs without LOOP / JUMP.
- c ≥ 1: ΔPERF_CH_RD_BEATS[c] = Σ expected read beats of the executed instructions on channel c
  (MVM: rows_c × ROW_BEATS; LD: `n_lines`; ST reads no DRAM).
- **ch0**: ΔPERF_CH_RD_BEATS[0] = Σ expected beats + ⌈CMD_COUNT / 2⌉, only for runs **without
  LOOP / JUMP that ended with an END that is the last instruction (pc = CMD_COUNT − 1)**. If END
  is not last, the prefetcher may already have read up to 16 beats past it (within
  `CMD_COUNT`), depending on timing. Lowering always puts END last; `tb/core` `core_perf` asserts the same equation, and it
  holds for all 145 runs of `traces/model/qwen05b-q8_0`. With NUM_CH = 1 this is the only beat
  check.
- Inequalities: ΔCH_WAIT[c] ≤ ΔCYCLES; ΔMVM_CYCLES ≥ Σ max-channel weight beats.
- sim: Σ hook cycles = ΔCYCLES, Σ hook MVM cycles = ΔMVM_CYCLES.
- vkernel / fastreplay (`retire`): (first retire − BUSY rise) + Σ intervals between retires +
  (BUSY fall − last retire) = ΔPERF_CYCLES (BUSY also covers the start before the first retire
  and the drain after END); Σ MVM occupancy = ΔPERF_MVM_CYCLES.
- Pieces: the prologue and END are counted as in §5.4.

### 5.8 Outputs and `tsr-prof`

- Record JSON (`tesseract-prof` v1): a `meta` block with the measurement conditions (spec rule 7:
  model and file hash, prompt length, generated tokens, build git hash; plus device, xclbin UUID
  and kernel, clock, T_MAX, attribution mode, profiler overhead), the exec records, host events.
- Chrome trace JSON for Perfetto (ui.perfetto.dev, chrome://tracing):
  - process "host": thread `graph_compute` (split ranges, node names), thread `transport`
    (write, read, stage, exec, with bytes), thread "other backends" (gaps);
  - process `Tesseract0`: thread `exec` (device ranges, length ΔCYCLES / f_core aligned to end at
    `t_done`, clipped at `t_call`; the gap between completion and detection stays as exec
    overhead), thread `insn` (spans or instructions with node, layer, fmt, N, K, T, beats per
    channel, model and measured cycles, attribution), and with `retire` also DMA, QACT, MVM and
    fetch-wait tracks;
  - counter tracks: GB/s per channel, CH_WAIT ratio, PC samples; flows from split to exec.
- `tsr-prof` (`sw/tools/tsr_prof.py`, standard library only, run with `python3 -I`):

| Command | Does |
|---|---|
| `report [--by role\|layer\|node\|shape] [--csv]` | Tables per role, layer, node, shape; per-token breakdown (device, exec overhead × count, other backends, copies); the out-of-order line (G, G × D, F) |
| `perfetto` | Regenerate the Chrome trace |
| `roofline -o x.svg` | x = 2·N·K·T / weight bytes, y = FLOP/s; roofs: M0 measured bandwidth and the datapath (one weight beat per channel per cycle × T); SOL% per point |
| `diff A B [--key node\|shape]` | Δms, ΔGB/s, Δbeats; deterministic differences first. Sim against card gives the model error per instruction kind |
| `check` | The self-check; exit status 1 on a violation |
| `regress --baseline perf/…` | Compare with a baseline (§5.9) |
| `calibrate hw.prof.json [sim.prof.json] -o perf/model-u200.json` | Fit fixed costs per instruction kind, cycles per beat per channel, DMA latency (two-point) |
| `predict sim32b.prof.json --calib perf/model-u200.json` | Predict 32B tok/s on the card before M4 |

`tsr-trace-replay --prof` and `tsr_xrt_selftest --traces … --prof` profile trace sets per exec
without llama.cpp.

### 5.9 Regression baselines and measurement conditions

- `perf/<device>/<model>-<config>.prof.json`.
- Deterministic metrics must match exactly in a sim ctest: execs per token, instructions, beats
  per channel, model cycles, totals per role. The 0.5B summary of `traces/model/qwen05b-q8_0` is
  the first golden.
- Hardware metrics must stay within max(3 %, 3σ of five baseline runs) under the same
  conditions: tok/s, device ms per token, exec overhead.
- Spec rule 7 is unchanged. The record's `meta` block carries the rule-7 conditions; as a
  practice (not a rule), performance numbers written from PD1 on keep their record under
  `perf/`. Existing numbers (`docs/m0-bw.md`, `docs/rtl-m3.md`) are not affected.

### 5.10 Profiler environment variables

| Variable | Values | Meaning |
|---|---|---|
| `TSR_PROF` | `<prefix>` | On; writes `<prefix>.prof.json` and `<prefix>.trace.json` |
| `TSR_PROF_LEVEL` | `exec`, `span` (default), `insn` | Record granularity |
| `TSR_PROF_ATTR` | `model` (default), `split` | §5.4 (`retire` is selected by the tb tools' `--prof`) |
| `TSR_PROF_PERF` | `all`, `min`, `off` | Counters read per exec: `min` = CYCLES, RETIRED, MVM_CYCLES, CH_RD_BEATS[*] (7 on k4, 4 on k1); `all` = every counter (13 on k4, 7 on k1). Default `min` on xrt, `all` elsewhere; revisited after the M3 latency measurement |
| `TSR_PROF_SAMPLE_MS` | n | PC sample interval (§5.3) |
| `TSR_PROF_EVERY` | n | Full detail every n-th token |
| `TSR_PROF_M0` | `<file>` | Bandwidth reference, default `perf/u200-m0.json` |

With `TSR_PROF` unset every hook is a NULL check.

### 5.11 The out-of-order decision (spec 미결정 사항)

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** Moot in v2: there is no command-buffer path.
The RISC-V issues asynchronously to per-unit queues (D3), and `MVM_WAIT` / `MVM_EMPTY` (EX §6) split
the MVM idle time into dependency-bound and issue-bound shares.

Measured on M4 32B Q4_K_M decode `tsr-prof` records, with the conditions stated:

| G × D | Decision |
|---|---|
| < 3 % | Keep sequential execution |
| 3–10 % | Measure again after M6 (fewer host round trips raise D) |
| > 10 % | Design it together with M5's MVM-to-MVM overlap |

The non-MVM time is broken down into LD, QACT, ST and issue gaps with the vkernel `retire` probe
and card `split` runs. F is reported next to it: it is M5's target, not the out-of-order one.
Expected for 32B by the cycle model (planning estimate): about 1–2 % (MVMs are larger, ≈ 1.1 M
weight beats per channel per layer against ≈ 14 K non-MVM cycles per layer).

---

## 6. Conditional RTL instrumentation

**Superseded (2026-10-30, W0 of `docs/v2/README.md`).** The extension counters are now EX §6's PERF
slot table (`PERF_INFO` at 0x080, slots X + 0 onwards, DB's slots 61–63). The trace ring `TRC_*`
(§6.3) is deleted; DB's hardware trace writer (DB §8.3) is the timeline source.

Added only when its criterion is met by a measurement; the RTL contract document of that
milestone records the measurement. Everything here only observes, outside the bit-exact
datapath.

### 6.1 Extension counters and `PERF_INFO`

- Indices are fixed: 5 + 2C = `PERF_FETCH_BEATS`; from 6 + 2C the unit-occupancy and issue-gap
  counters (DMA busy, QACT busy, issue gap, fetch stall; exact order fixed when they are
  introduced). A later group implies the earlier one, so an index never moves.
- `PERF_INFO` (0x080) [7:0] = number of implemented counters; 0 (register absent) means
  5 + 2 × NUM_CH. Indices 5 + 2 × NUM_CH … 63 (offsets up to 0x2FC) are reserved for this
  (spec CSR map).
- `PERF_CTRL` (0x084, proposed, not yet in the spec map) bit0 SNAP_ALL: sample every counter at
  once for a consistent multi-counter read.

### 6.2 Go criteria

| Item | Added when | Where |
|---|---|---|
| `PERF_FETCH_BEATS` | The ch0 equation fails on the card, or from M8 (LOOP) / M9 (JUMP) the fetch count no longer follows from the program | index 5 + 2C |
| Unit occupancy and issue-gap counters | Card `split` non-MVM time and the vkernel `retire` estimate differ by more than 2 % of ΔCYCLES while G × D is in 3–10 % | from index 6 + 2C |
| All-counter snapshot | Reading PERF per exec costs more than 2 % of device time per token (M3 latency measurement) | `PERF_CTRL` 0x084 |
| Trace ring | §6.3 | `TRC_*` 0xA00–0xA3C |

Costs: one 64-bit counter ≈ 70 LUT + 64 FF (U200 < 0.01 %, ZU3EG ≈ 0.1 %); the snapshot shadow
NPERF × 64 FF (U200 k4: 832 FF).

### 6.3 Trace ring (optional, `HAS_TRC`)

- One 128-bit record per retired instruction into an on-chip ring: 1,024 records on the U200 (4
  BRAM36). Record: `PERF_CYCLES` low 48 bits at retire, IA 32 bits, w0[15:0], run number low 16,
  16 reserved.
- Registers: `TRC_CFG` 0xA00 (ring depth; 0 when absent), `TRC_CTRL` 0xA04 (enable, stop when
  full, start/stop on a BP, clear), `TRC_HEAD` 0xA08 (next write position, bit31 wrapped),
  `TRC_RD_IDX` 0xA0C, `TRC_DATA0..3` 0xA10–0xA1C.
- The host reads it while halted or after the run. Start and stop conditions reuse the debug
  block's BP comparators.
- Streaming to DRAM (memory source id 3, free today) only if a measurement shows the ring is
  not enough.
- Added if either holds: on resident programs (M8–M9, where the host no longer sees exec
  boundaries) `model` attribution is off by more than 5 % against vkernel `retire`; or a
  hardware-only defect that vkernel does not reproduce needs instruction history. First check
  whether `DBG_HIST` (the last 8 IAs) is enough.

---

## 7. Access paths and JTAG

### 7.1 Per target

| Target | `DBG_*` and PERF access | JTAG | Signal-level debug |
|---|---|---|---|
| sim, mock | function calls (the simulator is the reference model) | — | — |
| vkernel | the adapter window (Verilator) | (PD5) remote_bitbang + OpenOCD to test the DTM | waveforms |
| fastreplay | the core CSR port (core only, Verilator) | — | waveforms |
| U200 (XRT) | PCIe CSR window, `s_axi_control` 0x1000 + offset | none | ILA in debug builds, XVC over PCIe |
| Zynq / Kria | PS AXI-Lite (UIO) | optional BSCANE2 → CSR bridge | ILA over XVC (Ethernet) or the board's USB JTAG |
| ASIC, boards without a host bus | JTAG TAP + DTM → CSR arbiter | main path | — |

The debug block's register meaning is the same everywhere; debug access always enters through
the core's CSR port. The core has no JTAG port, and vendor primitives stay in `rtl/tech/`.

### 7.2 U200

**No JTAG to user logic.**

- The XRT shell is a DFX design: the user kernel lives in the reconfigurable region, and
  configuration primitives such as BSCANE2 can be placed only in the static shell (UG909's DFX
  rules; to be confirmed on this shell, spec 미결정 사항).
- The shell's **user** debug network (an AXI-to-BSCAN debug bridge at PF1 BAR0 0x1C00000) is
  reached only by XVC over PCIe (xocl `xvc_pub`) and has no USB-JTAG fallback
  (`supports_jtag_fallback = false`). The **management** debug network does have a fallback, but
  it does not reach the user kernel.
- So JTAG would ride the same PCIe link, slower than the CSR window, and show nothing new. No
  BSCANE2, no JTAG-to-CSR bridge and no JTAG-to-AXI master are built for the U200; architectural
  debugging uses the debug block through the CSR window.

**ILA, debug builds only.**

- For what the debug block cannot see: AXI deadlocks, firewall trips, CDC, cycles inside an
  instruction.
- Built from a separate link configuration `hw/u200/k4_dbg.cfg` (underscore, like `k4.cfg` and
  `k4_floorplan.tcl`); the regular `k1.cfg` / `k4.cfg` builds carry no ILA.
- Interfaces: v++ `[debug]` (`chipscope=tsr_k4:m_axi_ch0` … `m_axi_ch3`,
  `chipscope=tsr_k4:s_axi_control`, `protocol=all` for System ILA and the protocol checker).
- Core-internal signals: Vivado Tcl hooks in the debug configuration's `[vivado]` section mark
  the chosen nets `MARK_DEBUG` and create and connect the ILA on the synthesized netlist
  (`create_debug_core`, `connect_debug_port`). No ILA instance and no debug attribute goes into
  `rtl/` outside `rtl/tech/` (spec rule 6); if an RTL instance were ever needed, it would be a probe wrapper in
  `rtl/tech/xilinx` with an empty generic implementation. The trigger is `dbg_trig` (a TRIG_ONLY
  breakpoint pins a given instruction). Which nets survive synthesis and the exact hook step are
  settled with the first debug build (spec 미결정 사항, together with 300 MHz and SLR timing).
- Connect: `debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u<dev>.<idx> --hw_server` (xocl names its
  nodes `xfpga/<subdev>.u<device id>.<index>`,
  `build/xrt-driver/final-r1-deb/src/driver/xocl/xocl_subdev.c:339-342`), then
  `debug_hw --vivado --ltx_file <file>.ltx` (Vitis 2024.2).
- Division of labour: debug block for wrong values, which instruction, where a hang sits; ILA for
  bus-level and sub-instruction problems.

### 7.3 Zynq / Kria

- Main path: PS AXI-Lite (UIO mmap).
- On a flat design, `rtl/tech/xilinx/tsr_jtag_bscan.sv` (BSCANE2 on USER3 / USER4, clear of the
  debug hub's default USER1) can bridge JTAG to the CSR arbiter. Uses: a backup when Linux is
  dead, and checking the ASIC DTM on an FPGA before silicon. Whether the Kria shell (dfx-mgr) is
  flat or DFX is checked when the port starts.
- ILA over XVC on Ethernet (PS AXI debug bridge) or the carrier's USB JTAG.

### 7.4 ASIC and boards without a host bus

- JTAG TAP + DTM is the main path, in the RISC-V debug specification's DTM format: IR 5 bits,
  IDCODE 0x01, `dtmcs` 0x10, `dmi` 0x11, BYPASS 0x1F; `dmi` = {address 10 bits = CSR offset / 4,
  data 32, op 2}. `dmi` reaches the Tesseract CSR space, not a RISC-V debug module.
- `tsr_csr_arb` (platform layer) arbitrates the host bus (if any) and JTAG onto the one core CSR
  port; the CDC is a handshake, with TCK well below F_CORE (e.g. ≤ F_CORE / 8). pulp `riscv-dbg`
  (`dmi_jtag_tap`, `dmi_jtag`, Apache-2.0) is a reuse candidate.
- Host side: OpenOCD (`irscan` / `drscan`), and an `openocd` transport in `tsr-dbg`. CI:
  Verilator remote_bitbang + OpenOCD; the same scripts on Kria through BSCANE2.
- Without a host bus, a debug memory port (system bus access through memory source id 3) is an
  open item.

### 7.5 Several CSR masters

- Only the master that owns execution writes CTRL, CMD_*, ARG* and STATUS and reads PERF_*:
  on the U200 and Zynq the host bus together with its adapter's start sequencer (already a
  second requester on the core CSR port, `tsr_xrt_ctrl.sv`), on a board without a host bus the
  JTAG DTM.
- Any other master accesses `DBG_*` and only reads the remaining registers; it does not read
  PERF_* at all. Reason: a LO read moves the single snapshot slot, so the owner's next HI read
  would return the live HI and its 64-bit value could tear.
- Inside the host, one thread under the backend's device mutex reads PERF.
- Per-master snapshot slots are an open item for targets where JTAG and a host bus are used at
  once.

---

## 8. Deliberately not done

- **Halt on error**: a faulting instruction changes no state, and R1–R15, `DBG_IR*`, `DBG_OPA*`
  and the SPM stay until the next START, so a post-mortem read gives the same information without
  another RTL path.
- **Instruction injection (program buffer)**: one more cmdproc error path; revisit only if the
  SPM window is too slow.
- **Patching breakpoints** (replacing an instruction): breaks the no-self-modification rule and
  the sharing of cached programs. IA comparison is enough.
- **Halting inside an instruction** (mid-MVM): not an observation point of sequential semantics;
  inside an instruction, use the repro trace with waveforms, or the ILA.
- **A GDB stub**: the source level is a ggml graph; the program map and a dedicated CLI fit
  better.
- **On the U200**: an own BSCANE2 or JTAG bridge, and an ILA in the regular build.
- **llama.cpp**: no patch to the pinned commit for profiling; no `cb_eval` node stops or timing,
  because they cut splits and change what is measured. numcheck keeps its own `cb_eval` use for
  numerics.
- **Existing definitions**: PERF indices, `CAPS` bits and error codes stay as they are; no
  always-on RTL instrumentation in the default build.

---

## 9. Open items and follow-ups

The spec's 미결정 사항 holds the decisions (XRT register access during a run, PCIe CSR read
latency, the U200 debug build, forced release of a stuck unit, extension counters and the trace
ring, the porting targets; `JUMP`-chain addressing was dropped with JUMP). Follow-ups for the lead:

- After M3 part C merges: the spec's XRT 어댑터 paragraph should give the window base
  (`s_axi_control` 0x1000 + offset, `docs/platform-u200.md`; it says "인자 뒤 오프셋" today, and
  the spec's access table refers to that paragraph instead of repeating the number) and may say
  that the debug block appears at 0x1800–0x19FC; the spec tree's `build/u200/` line should read `hw/u200/`
  (where `k1.cfg`, `k4.cfg`, `k4_floorplan.tcl` and later `k4_dbg.cfg` live).
- When PD2 lands: the spec's 구성 파라미터 table gets `HAS_DBG`, `HAS_DBG_SPM`, `DBG_NBP`,
  `HAS_TRC`; `DBG_ID` bit assignments are frozen; `TSR_DEVICE` documentation lists `shadow:`.
- At M3 on the card: measure the PCIe read latency and the END-only exec overhead
  (`tsr_xrt_selftest`), and whether a new process can open the compute unit without disturbing a
  halted core (§4.4).

# Control core (RISC-V): design record

This document records the control core of Tesseract: what was decided, why, what it costs and what was
rejected. The control core is an embedded lowRISC Ibex (RV32IMC) in its own clock domain. It runs the
firmware kernels that the host launches and issues every tile op asynchronously to per-unit queues.
Section numbers are cited by code and other documents and stay stable.

> **Scope.** Design record and rationale, not the contract: the normative design is
> [CC](design/04-control-core.md), [EX](design/01-execution-model.md) and [DB](design/05-debug.md), and `docs/design` wins
> where they differ. Status words: *on the simulator* (firmware, interpreter, queue model and debug models
> in `sw/ctrl` and `sw/sim`), *designed* (specified in `docs/design`, no implementation), *planned* (the
> control-core RTL is W4). *estimate* means not measured; *confirm* marks a recalled third-party fact that
> is checked when the dependency is vendored. Card and Vivado measurements were taken before the control
> core was added, with the host issuing the ops; they are the only hardware numbers.

**User decisions (2026-10-24, D1-D13)**

| # | Decision | Status |
|---|---|---|
| D1 | Embed a RISC-V control processor in the Tesseract core. Simple: in-order, no branch prediction, no superscalar, no out-of-order, no cache. | designed (CC §3); the interpreter `tsr_rv` runs the firmware on the simulator |
| D2 | Code, data and stack live in local SRAM (TCMs): instruction fetch never uses DRAM bandwidth; code regions are small. | designed (CC §4.2, 128 KiB + 64 KiB); the firmware links into them (`sw/ctrl/rt/link.ld`) |
| D3 | The RISC-V issues the existing 256-bit tile instructions **asynchronously**: issue and continue (fork); several tile ops in flight on different units; software joins later. | on the simulator (queue model, firmware issue path) |
| D4 | The SPM is mapped into the RISC-V address space (same memory space, a bus connection). | on the simulator (`tsr_bus.c`; window and line-buffered alias) |
| D5 | The RISC-V has its own clock domain; its clock depends on the core implementation. | designed (CC §11); the simulator models the clock ratio (`TSR_SIM_RV_RATIO`) |
| D6 | Kernels in C++ (stock RISC-V GCC / Clang, no custom compiler); tile and SIMD operations as a C++ header library; a JIT (torch.compile, tinygrad, later JAX PJRT) calls precompiled kernels. | on the simulator (`sw/ctrl`; xPack GCC 14.2.0-3); `tsr_rt` and the ggml backend launch precompiled kernels |
| D8 | Scalar registers: one bank s0-s15 (about 512 FF) inside the VU. | designed (CC §1) |
| D10 | M8 / M9 decode steps are C++ kernels that issue tile ops directly; position-dependent addresses are computed in C++. | the `graph` kernel issues tile ops directly |
| D11 | One checker, `tsr_chk`, in front of the dispatcher. | designed (CC-9) |
| D12 | U200 RISC-V clock: planning value 166.7 MHz; 250 MHz only if the Ibex out-of-context run closes there; 125 MHz fallback. | designed (CC §3.3, §11.1); the T1 gate is planned |
| D13 | Timing gate: freeze the control-core RTL only after the k1, k4t1 and k4 out-of-context runs meet timing and the k4 link kernel clock stays at or above 276.3 MHz with `HAS_CTRL = 1`. | CC-23; gates T1-T4 (CC §14.3); planned |

**User decisions (2026-10-30, D14-D18)**

| # | Decision | Status |
|---|---|---|
| D14 | The host does not execute tile ops. It loads the firmware image and LAUNCHes kernels; every tile op is issued by the RISC-V (EX §2-§4). | implemented (`452e27b`) |
| D15 | No host debug access to the issue port. Bring-up, the card selftest and trace replay run as RISC-V kernels. | on the simulator: no host path to the issue port; `tsr_sim_push` is a simulator test hook, never reachable through `tsr_transport` (SW-19) |
| D16 | One execution path in the tree: the software path runs end to end on the simulator before the RTL and the card. | on the simulator (W3) |
| D17 | Device KV cache in q8_0; attention arrives as llama.cpp `FLASH_ATTN_EXT` (a quantized V cache requires Flash Attention). The klcheck / numcheck baselines move to FA on. | on the simulator: AT references and units (`tsr_ref_at.c`, `tsr_at.c`); the device-KV backend is W3b |
| D18 | The design has full debug: RISC-V Debug Spec module, halt / resume / step, breakpoints and GDB source-level debugging over PCIe (a GDB RSP server on the DMI window), plus tile-level debug. | on the simulator: debug module and DBG block models (`tsr_dm.c`, `tsr_dbg.c`); the GDB server is P3.7, the RTL W4 (see [`prof-dbg.md`](prof-dbg.md)) |

The numbers D7 and D9 are not used.

Milestones: M6 is the control-core hardware (control core, issue queues, vector unit, attention and
KV ops, second SPM read path, debug), M7 the RISC-V kernel library and graph execution (one launch per
token), M8 device MoE and PyTorch on the kernels, M9 the resident decode loop (optional). The
technical defaults chosen with D14-D18 are decisions of `docs/design` ([README](design/README.md) §2): even /
odd SPM banks for the second read path (CC-12), `ap_ctrl_none` with polling (EX-6), a
host-readable log ring in the core domain (EX-24), and a k4 link that stays at or above 276.3 MHz
with the control core (CC-23). The M8 and M9 pass criteria are that the host writes only 2
arguments per token and that the host only polls a ring buffer.

| Question | Answer |
|---|---|
| Which core? | lowRISC **Ibex**, pinned and unmodified: RV32IMC + Zicsr + Zifencei, no F, with the "maxperf" pipeline (writeback stage, branch target ALU, single-cycle multiplier; CC-34) (§2, CC §3.2). |
| How does it issue tile ops? | A memory-mapped issue port (8 staging words + push), per-unit in-order queues, 32-bit tickets, up to 2 hardware dependencies per op, SYNC join points and FENCE barriers. Software checks the credit before a push (§4, CC §7). |
| How does it see the SPM? | A 4 MiB window at 0x4000_0000 and a line-buffered alias at 0x5000_0000, through a bridge that is one client of the 2R2W even / odd banked `tsr_spm` (§3.5, CC §10). |
| DRAM? | No direct window. Data moves by tile DMA (§3.6). |
| What does the control path add to the datapath? | A checker (`tsr_chk`), a dispatcher with unit queues, SPM banking and round-robin arbitration, DMA burst reservation, a CSR block, the CDC library (§5, CC §8-§10). |
| Clocks on the U200? | A second kernel clock `ap_clk_2`; planning value 166.7 MHz (500 / 3), SLR1 (§7, CC §11). |
| Software? | `sw/ctrl`: freestanding C++20 headers, crt0, runtime, linker script, ELF image, kernels (§8; [`sw/ctrl/README.md`](../sw/ctrl/README.md)). |
| Verification? | RV32 interpreter against Spike, queue model with schedules, race detector, firmware host build against the RISC-V build (on the simulator); RTL co-simulation in lockstep is W5 (§9). |
| Debug? | RISC-V Debug Spec 0.13.2 module, DMI through the core CSR window, an RSP server in the device-owning process, tile-level debug (§10, DB). |
| Cost? | About 12.5-23 K LUT net, 49 RAMB36 on the U200 (*estimate*, CC §14.5; §12). |

---

## 1. Datapath facts

The facts below describe the datapath that the control core is designed around.

- **Dispatch protocol.** A unit takes one command at a time through
  `cmd_valid / cmd_ready / cmd / done`; units report no errors (every architectural check is made
  before dispatch); there is no flush or cancel ([`rtl-m2.md`](rtl-m2.md) §3.0). The checks live in
  `tsr_chk` and the dispatch handshake is at the queue heads (CC §8-§9).
- **The datapath is concurrent in structure.** Every unit has its own SPM client port
  (0 DMA, 1 QACT, 2 MVM) and its own memory source per channel (`tsr_pkg.sv`). The tree keeps three
  SPM clients and three memory sources per channel; source 0 is tied off as AUX (CC-26).
- **SPM**: one read and one write port, fixed priority lowest client index first, registered grant
  (`tsr_spm.sv`). A higher client that streams continuously starves a lower one without bound, and
  the DMA's LD flow control relies on being client 0. The design uses two even / odd banks with
  per-bank round robin and a DMA burst reservation instead (CC §10).
- **Memory arbiter**: round robin per burst among the sources with one outstanding budget per
  channel, response routing head-of-line; sources may drop their response ready "only for reasons
  local to that source" (rule R4), which is the no-deadlock argument (`tsr_mem_arb.sv`).
- **PERF**: the time counters are gated by ACTIVE and QHALT (EX §6).
- **Clock and reset**: one clock `clk`, synchronous `rst_n` ([`rtl-m2.md`](rtl-m2.md) §1). The
  technology-wrapper layer already listed CDC synchronisers; none existed.
- **U200**: the kernel tops had one clock `ap_clk`; the platform offers DATA 300 MHz (scalable) and
  KERNEL 500 MHz clocks ([`platform-u200.md`](platform-u200.md) §8.2). The XRT shell's AXI firewall
  cuts a control slave that does not answer, and recovery needs a card reset; so no CSR access may
  wait on an unbounded event (CC §6.1 rule).
- **Timing margins and the k4 link**. Out-of-context WNS after the margin changes
  ([`rtl-m3.md`](rtl-m3.md) §14.3, measured in Vivado 2024.2, out of context): k1 +0.100 ns,
  k4t1 +0.109 ns, k4 +0.022 ns. The k1 link closed at 300 MHz (WNS +0.017 ns; kernel clock
  +0.060 ns). The k4 link missed 300 MHz by 0.285 ns, and `v++` scaled the kernel clock to
  **276.3 MHz** (measured in Vivado 2024.2, linked; [`platform-u200.md`](platform-u200.md) §8.2
  and §14.1); the plan had projected -0.1 to -0.2 ns (*estimate*).
- **SLR1 budget** ([`rtl-m3.md`](rtl-m3.md) §14.5): floorplan C' puts 25,068 LUT, 8 RAMB36, 128 of
  160 URAM and 32 DSP of the core into SLR1's dynamic region (205,440 LUT / 420 RAMB36 / 160 URAM /
  1,320 DSP); DDR[2]'s controller adds about 30.5 K LUT and 61 RAMB36 there (measured in the M0
  link). The control core's additions are budgeted in CC §14.1 (§12.1 below).
- **Host control cost**: with the host issuing the ops, 21 register writes and 6 reads per exec, 145
  execs per Qwen2.5-0.5B decode token (about 3,000 posted writes and 870 reads per token). A measured
  exec cost on the card is 35-36 µs (measured on the card, k4, 276.3 MHz;
  [`platform-u200.md`](platform-u200.md) §14.2). A launch costs one descriptor push into the launch
  queue (EX §4).
- **Reference fp helpers**: `tsr_fadd` and the other helpers perform one native IEEE binary32
  operation with explicit DAZ / FTZ / NaN bit tests and never touch the FP environment
  (`sw/sim/tsr_ref.c`). They compile unchanged for a soft-float RV32 target; the firmware needs
  none of them (SW-8: no f32 math in firmware).

---

## 2. Core choice and ISA

### 2.1 What the workload needs (estimates)

- **Issue rate** (*estimate*, plan). A device-resident decode token is about 1 K tile ops on
  Qwen2.5-0.5B and 3-5 K on 32B. At 25-40 RISC-V cycles per op (§4.8) that is about 0.04 M cycles per
  0.5B token and 0.2 M per 32B token. CC §3.3 restates this at 1.5 K and 4 K tile ops
  (*estimate*, CC §3.3) and adds the measured executor cost: see §12.4.
- **Decision latency.** The latency-critical points are device decisions: TOPK result, expert ids,
  expert MVMs (MoE, M8) and sampling. With 1-2 µs per decision and 30 MoE layers that is under 0.1 ms
  per token against about 43 ms of weight streaming for Gemma 4 26B-A4B (about 3.3 GB per token at
  76.8 GB/s; *estimate*).
- So 50 MIPS sustained is plenty (100 MHz at CPI 1.2-1.6 leaves margin). Fmax and IPC do not decide
  the choice. What does: verification maturity (the core becomes part of every device-resident
  result), standard debug (C++ kernels need GDB), vendor independence and an ASIC path, Verilator,
  and an integration that leaves the core unmodified.

### 2.2 Candidates

| | Ibex | PicoRV32 | VexRiscv | SERV | In-house RV32IM |
|---|---|---|---|---|---|
| Source | SystemVerilog (lowRISC) | Verilog-2005, one file | SpinalHDL (Scala) generator to Verilog | Verilog | SystemVerilog (ours) |
| License | Apache-2.0 | ISC | MIT | ISC | project |
| Pipeline | 2-stage in-order (IF, ID/EX); optional WB stage, branch predictor and ICache, all off by default | multi-cycle, not pipelined; CPI about 4 (README) | 2-5 stages, plugins (prediction, caches optional) | bit-serial, 32-64 cycles / instruction | 3-5 stages, to design |
| ISA options | RV32I/E, M (slow / fast / single-cycle), C, B, Zicsr, Zifencei, M + U modes, PMP | RV32I/E/M/C; own IRQ scheme, no standard trap CSRs | RV32IM(A)(F)(C), M/S/U modes | RV32I (+M), Zicsr | as built |
| Verification | UVM + riscv-dv random programs with Spike co-simulation, formal work, OpenTitan sign-off; README grades configurations ("small" fully verified, *confirm*) | riscv-formal | regressions with riscv-tests / arch tests, partial riscv-formal; depends on the plugin set | riscv-arch-test (riscof) | none yet: months of work (*estimate*) |
| Debug | RISC-V Debug Spec 0.13.2 with an external debug module (pulp riscv-dbg, as in OpenTitan) | none | own DebugPlugin (needs a SpinalHDL OpenOCD fork); recent versions add a debug-spec module | none standard | must implement debug mode (dcsr, dpc, dret) and still reuse a DM |
| ASIC history | OpenTitan Earl Grey silicon and others | several shuttles | several SoCs and shuttles | several shuttles | none |
| Verilator | yes (Ibex simple system) | yes | yes (generated) | yes | yes |
| FPGA area (*estimate*) | 2.5-4 K LUT with the FPGA register file (no published figure for this configuration) | 0.75-2 K LUT (7-series README) | 1-2 K LUT (README configurations) | 0.2-0.4 K LUT | 1.5-3 K LUT |
| Fmax UltraScale+ -2 (*estimate*) | 125-250 MHz (no published UltraScale+ figure; register file to ALU adder to data address to TCM / slave decode limits it; planning value 166.7 MHz) | 250-450 MHz on 7-series (README) | 200-350 MHz on Artix-7 (README), more on UltraScale+ | high | 300 MHz possible |
| Throughput (*estimate*) | CPI 1.2-1.6 | Fmax / 4 | CPI 1.2-1.5 | under 10 MIPS | CPI 1.2-1.5 |
| Concern | lower Fmax (irrelevant here, own clock); third-party style (async reset, `prim_*`) | no standard traps or debug | the source of truth is Scala; regeneration needs SpinalHDL; generated Verilog is not reviewable source | too slow for decision latency | the verification and debug work is the cost, not the pipeline |

### 2.3 Decision: Ibex

- **Configuration** (an Ibex top parameter set; the table is CC §3.2): RV32E 0, RV32B none,
  BranchPredictor 0, ICache 0, PMPEnable 0, SecureIbex 0, MHPMCounterNum 0, DbgTriggerEn 1 with 2
  hardware breakpoints, register file in LUTRAM on Xilinx (flops or latches on an ASIC, chosen by
  the technology layer). The pipeline is Ibex's "maxperf" one (CC-34), not the "small" one:
  WritebackStage 1, BranchTargetALU 1, RV32M single-cycle. It stays in order, with no prediction and
  no cache (D1). Measured on the simulator's
  cycle tables (the W2a firmware of `92f0b17` under both tables, `fw_m6_perf`): 1,977 RISC-V cycles per op
  record on small and 1,547 on maxperf (-21.8 %); the writeback stage accounts for about 300 cycles, the branch
  target ALU about 113 and the multiplier about 23 (CC §3.2). Maxperf has +22 % gates in Ibex's own
  table (32.48 against 26.60 kGE) and adds two candidate clk_rv limiters; the fallback order, decided
  in advance, is in CC §3.2 and the T1 gate (CC §14.3). The cycle costs per instruction class that
  the simulator models are the table of CC §3.2; the maxperf load / store timing is derived from
  Ibex's RTL and stays *confirm* until the RVFI bench (P5.2).
- **Why Ibex**: SystemVerilog with generic `prim_*` implementations (vendor independence; the only
  FPGA-specific piece is the clock gate, mapped through `rtl/tech`); the most mature open
  verification and silicon history; the standard debug spec (GDB works as it is); Verilator support;
  in order with no predictor and no cache by default, so D1 needs no modification; simple
  `req / gnt / rvalid` instruction and data ports, so TCMs and the control window attach without a bus
  fabric; the issue port is MMIO, so the core stays unmodified.
- **Integration rules**: Ibex and riscv-dbg are vendored at pinned commits in `third_party/ibex` and
  `third_party/riscv-dbg` (with the few pulp `common_cells` files the DM needs) and never edited
  (CC §3.1). They are not in the tree yet (W4). They keep their own style (asynchronous active-low
  resets, `prim_*` modules) and their own lint waivers, one waiver file per third-party tree;
  `prim_clock_gating` binds to `rtl/tech/<target>/tsr_clk_gate.sv`, a pass-through on the FPGA. The
  tested Verilator version is 5.046; the vendoring step checks that the pinned Ibex builds with it.
- **Bus rule** (every data-port and instruction-port slave): `gnt` comes from a register
  (`gnt = ~outstanding_q`), never from the address decode. Ibex's `data_addr_o` comes combinationally
  from the ALU adder and `data_gnt_i` feeds its stall logic in the same cycle (*confirm* on the pinned
  commit), so an address-dependent grant would put register file, adder, decode, grant and PC logic on
  one path, the likely Fmax limiter. Every wait (credit, SPM bridge, core window, debug-module
  arbitration) is a late `rvalid` instead. With the writeback stage the next request can come in the
  cycle the previous response returns, so a TCM access (fixed next-cycle response) does not set
  `outstanding_q` and the TCMs take one access per cycle (CC §4.3, *confirm*).
- **Measure first**: the out-of-context synthesis of `tsr_ctrl_rv` with the TCMs on `xcu200-fsgd2104-2-e`
  (gate T1, planned) replaces the area and Fmax estimates before the control-core RTL is frozen.
- **Runner-up**: VexRiscv (smaller, faster) if Ibex's integration or F_RV turns out to be a problem.
  Not PicoRV32 (no standard traps or debug), not SERV (decision latency), not in-house (verification
  cost with no gain for this workload).

### 2.4 ISA subset and the F question

**RV32IMC + Zicsr + Zifencei**, machine mode only (Ibex has U mode; the runtime never uses it), no
PMP, no A, no F / D, no B.

- **M**: address arithmetic (expert x `EXP_STRIDE`, row splits, KV offsets).
- **C**: about 25-30 % smaller code (typical for RV32, *estimate*), so a smaller ITCM; Ibex handles it
  at full speed.
- **Zicsr**: traps, interrupts, `mcycle` / `minstret`, WFI, debug CSRs. **Zifencei**: `fence.i` after
  code is written into ITCM.
- **No A**: one hart; host handshakes use sequence numbers in the launch queue and completion ring
  (EX §4).
- **B** (Zba / Zbb / Zbs, and Zcb / Zcmp): options of Ibex, decided by SDK code size and cycles; not
  taken (CC §3.2).
- **No F**, although f32 scalars would be convenient:
  1. *Availability.* Ibex has no F. F-capable verified cores are larger (an FPU is 5-10 K LUT,
     *estimate*) and bring another verification story.
  2. *Bit-exactness does not need it.* The spec's fp rules are implemented by `tsr_ref` as one IEEE
     operation plus explicit bit tests. IEEE basic operations are deterministic, so the same C
     compiled for RV32 with soft-float gives the same bits as on the host. The firmware needs even
     less: it has no f32 math at all (SW-8). Every f32 constant (eps, 1/n, scales, RoPE
     constants) is computed by the host and passed in records or arguments, and per-element math is the
     vector unit's.
  3. *Volume.* The per-head bookkeeping of an online softmax is vectorised across heads in the VU
     (ASMAX, AMERGE, ANORM; AT §6.6), not scalar code on the RISC-V.
  4. *Fallback* if a profile ever shows scalar fp above 1 % of token time: move that math to VU ops, or
     choose a core with Zfinx / F (open item, §15).

---

## 3. Address map

### 3.1 RISC-V view

32-bit byte addresses, no MMU, PMP off. Every unmapped access is an access fault (instruction, load
or store); every region's size beyond its parameter is unmapped. The map is CC §4.1; the constants
are generated from one table source (`sw/tables/tsr_tables.py`, EX-30) into `tsr/mmio.hpp`, the
SystemVerilog package and the simulator.

| Range | Region | Fetch | Load / store | Notes |
|---|---|---|---|---|
| 0x0000_0000-0x0001_FFFF | ITCM, `ITCM_BYTES` 128 KiB (range reserved to 0x000F_FFFF) | yes | byte / half / word | data reads steal the fetch port for one cycle; stores then `fence.i`; vector table at `RV_BOOT`, first fetch at `RV_BOOT` + 0x80 |
| 0x0010_0000-0x0010_FFFF | DTCM, `DTCM_BYTES` 64 KiB (reserved to 0x001F_FFFF) | no | byte / half / word | .rodata, .data, .bss, stack |
| 0x1000_0000-0x1000_0FFF | debug module memory | yes | word | riscv-dbg layout; halt address +0x800, exception address +0x808 (*confirm*) |
| 0x2000_0000-0x2000_0FFF | control window (§3.2) | no | word only | clk_rv registers |
| 0x2000_1000-0x2000_1FFF | core window | no | word loads only | read-only view of the host CSR space at the same offsets (CC §5.3) |
| 0x4000_0000-0x403F_FFFF | SPM window, exact (`SPM_BYTES` 4 MiB) | no | byte / half / word | 0x4000_0000 + 64 x line + offset, little-endian |
| 0x5000_0000-0x503F_FFFF | SPM window, line-buffered alias (CC §10.5) | no | byte / half / word | loads may hit one 512-bit line buffer; stores write through and invalidate it |

- The instruction port decodes ITCM and the debug module; the data port decodes all regions. The
  debug module has one slave port, which both Ibex ports can hit in the same cycle: a 2:1 arbiter,
  data first, stalls the fetch through `instr_gnt`.
- **TCMs** are `tsr_sram_1r1w` (no true dual-port type in `rtl/tech`, no 2RW macro on an ASIC): the
  read port serves fetch, data reads and, while RV_RUN = 0, the host's loader and VERIFY reader; the
  write port takes data stores while RV_RUN = 1 and the host loader while RV_RUN = 0, never both.
  `.rodata` is linked into DTCM, so normal code never reads ITCM as data.
- The TCMs are 128 + 64 KiB (CC-3). The debug module's system bus master is connected as a
  second data-bus master with the refusals of CC §4.3.

### 3.2 Control window (RISC-V side, offsets from 0x2000_0000)

The table is CC §5.1-§5.2 (normative); this is the map of groups.

| Offset | Registers | Content |
|---|---|---|
| 0x000-0x01C | `ISSUE_W0`-`W7` | staged instruction words; kept after a push |
| 0x020-0x03C | `ISSUE_DEP0`, `DEP1`, `PUSH`, `TICKET`, `CREDIT`, `UNITS`, `STATUS`, `PIDX` | dependencies, the push (WMASK, REC), the ticket of the last push, credit per unit, built units, status, push index |
| 0x040 + 4u, 0x080 + 4u | `DONE_SEQ[u]`, `ISSUED_SEQ[u]` | completions per unit (a synchronised copy that may lag, never leads), next sequence per unit |
| 0x0C0 + 16u, 0x180 + 4u | `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0`, `ERR_PIDX` | first error of unit u |
| 0x140-0x14C | `QCTRL`, `QSTATUS`, `WAIT_TICKET`, `WAIT_CREDIT` | queue control through the issue FIFO (abort, hold set / clear, clear), status, interrupt waits |
| 0x160-0x17C | `REC_*`, `TRACE_MARK`, `REC_CFG` | DB: completion records and trace marks |
| 0x1C4 | `SPM_LB_INV` | invalidate the SPM line buffer |
| 0x200-0x294 | `LQ_*`, `LCQ_*`, `LOG_*`, `RT_*` | launch queue, completion ring, log ring, runtime state (EX §4.5) |
| 0x300-0x30C | `MTIME`, `MTIMECMP` | core time, timer compare |
| 0x400-0x7FC | `DBG_FLAGS` and reserved | DB |

- Reads in the control window have no side effects (`REC_POP` is a separate write), so a debugger's
  memory view cannot pop or push anything. Control-window writes through the debug module's SBA are
  refused (D15, CC §4.3).
- **Not reentrant.** There is one staging set. An interrupt handler, or a debugger running code through
  the program buffer, that issues between the interrupted code's stores corrupts that push and its
  ticket. Rule: handlers never issue; GDB inferior calls that issue are unsupported. A second staging
  bank is not provided (no use case worth its cost).
- **Hold sources.** The queue heads are held while any of three sources is set: the RISC-V's
  (`QCTRL`, applied in push order), the host's (`CTRL.HOST_HOLD`) and the debug halt's
  (`HOLD_ON_HALT` while halted); DB adds a per-unit hold. Each agent changes only its own source, so
  neither can drop the other's hold (CC §9.6).

### 3.3 Host view: core CSRs

EX §5 is the only host CSR map; CC §6 maps the control core's hardware onto its offsets: `CTL_ID`
0x01C, `RV_BOOT` 0x028, `RV_FREQ` 0x02C, `RV_CFG` 0x030, `CTRL` / `STATUS` 0x020 / 0x024, the loader at
0x0C0-0x0D4 and the DMI window at 0xC00-0xDFC. The PERF space is gated by ACTIVE (EX §6). The host and
the kernels exchange messages through the launch queue and completion ring (EX §4). Every host access
completes without `clk_rv` (core-domain copies; an access that must reach `clk_rv` ends after 4,096
core cycles with a sticky bit), so a stopped or relocking `ap_clk_2` never hangs the CSR port, and
`RV_CLK_OK` / `RV_FREQ` tell the host.

### 3.4 Loading the image

EX §3 defines the loader: a CSR loader (`LOAD_ADDR`, `LOAD_DATA`, `LOAD_CRC`) with a receive CRC plus
an on-device VERIFY read-back (`LOAD_COUNT`, `LOAD_CTRL`), a 64-bit image id in `.tsr_id` echoed in
HELLO, boot at `RV_BOOT` + 0x80, and a warm restart (`SOFT_RESET`, reload DTCM, VERIFY ITCM). The host
sequence is CC §6.4 and is implemented in the transport
([`sw/transport/README.md`](../sw/transport/README.md)). A cold load writes about 192 KiB (49 K posted
CSR writes) once per process; EX-8 estimates at most 10 ms per cold load (*estimate*, EX §3.3). A
DMA path into the TCMs (not needed at these sizes) and code overlays through the SPM are rejected.
Host-less targets (no host bus) would need a boot ROM (`HAS_BOOTROM`, an open item, §15).

### 3.5 SPM window

- Byte address 0x4000_0000 + 64 x line + offset; little-endian, as the SPM layout (f32 element i of a
  line is bytes 4i ... 4i+3), so `volatile float *` views of SPM vectors work directly.
- Uncached and non-speculative (Ibex is in order and does not speculate loads). Loads: byte, half,
  word. Stores: byte, half, word through `tsr_spm`'s per-byte enables (no read-modify-write).
  Misaligned accesses are split by Ibex into two aligned ones and are not atomic. Accesses at or
  beyond `SPM_BYTES` fault.
- One access at a time; each completes before the next. The bridge answers a store only after `tsr_spm`
  accepted the write, and the issue logic forwards a push only while no bridge store is outstanding,
  so an op pushed after a store sees it (§5.5) whatever Ibex's LSU does internally.
- **Bound.** The bridge waits at most 4,096 clk_rv cycles for the core side; then the access ends with
  a bus error (load or store access fault) and `BRIDGE_TIMEOUT`, so the hart stays reachable by the
  debugger. A late answer is discarded by its request tag.
- **Line-buffered alias.** The alias at 0x5000_0000 serves loads from a one-line buffer (invalidated by any
  store to either alias, any control-window or core-window access, any rising interrupt line and
  `SPM_LB_INV`, so rule 2(c) of §5.5 holds for the alias too). The executor reads op-list records in
  place through it (§8.1).
- Latency (*estimate*, 166.7 / 276 MHz): exact store 40-60 ns, exact load 55-80 ns, a 64-byte line
  through the alias about 160 ns instead of about 1.1 µs.
- Intended use: small control data (op-list records, indices, flags, argument blocks). Bulk copies are
  slow by design.

### 3.6 No DRAM window

The RISC-V gets no direct DRAM access:

- DRAM bandwidth is the product: weight streams use every channel at line rate. A 4-byte RISC-V access
  would cost a full 64-byte beat and 0.3-0.5 µs of DRAM latency (*estimate*) and would share
  `tsr_mem_arb` with the weight streams.
- It would be a second ordering path into DRAM beside DMA, with its own rules against LD / ST.
- The DRAM address space is per channel (`(ch, beat)`), not one flat physical space; other targets
  (HBM pseudo-channels, Zynq, ASIC) differ again.
- Data moves only by tile DMA. DRAM structures shared with the host (op lists, argument blocks, KV
  planes) are reached by LD / ST through the SPM.

### 3.7 Interrupts and timer

Ibex's interrupt inputs are level-sensitive; every one is computed or held in clk_rv from synchronised
values (CC §12.3):

- `irq_fast[0]`: the `WAIT_TICKET` ticket is done. `irq_fast[1]`: a unit entered error state
  (`ISSUE_STATUS.ERR_IRQ`, sticky, W1C). `irq_fast[2]`: a launch descriptor is waiting
  (`LQ_AVAIL` not zero and not ACTIVE). `irq_fast[3]`: the completion ring has room. `irq_fast[4]`: DB
  completion records at least half full. `irq_fast[5]`: the `WAIT_CREDIT` unit has a credit.
  `irq_timer`: MTIME at or above MTIMECMP. The others are tied to 0.
- Ibex's `mtvec` is vectored only (*confirm*); the runtime installs a vector table at `RV_BOOT`. Joins
  need no handler: WFI wakes on an interrupt pending and enabled in `mie` even with `mstatus.MIE` = 0
  (privileged spec; *confirm* for Ibex). The SDK never sets `mstatus.MIE`.
- MTIME counts core clock cycles, so RISC-V timestamps share the time base of the completion records
  and of the PERF counters. Ibex has no `time` CSR; the runtime reads MTIME.
- The host has no interrupt from the device (`ap_ctrl_none`, polling; EX-6);
  `host_irq_o` is a core boundary output for later targets (EX §9.3).

---

## 4. Asynchronous tile issue

The normative text is CC §7-§9 and EX §7-§8. This section records the decisions.

<p align="center"><img src="fig/issue.svg" alt="Upper panel: a kernel on the control core calls issue(op, after(a, b)); the push checks the credit and assigns a ticket, writes the op with up to two dependencies into the 32-entry issue FIFO that crosses from the clk_rv domain to the core clock domain; the checker tsr_chk checks it; the dispatcher appends it to the in-order queue of its unit. Lower panel: the queue contents of one decode layer of flash attention with three KV chunks on the DMA, QACT, MVM and VU units: ATTQK and ATTPV on the MVM, ASMAX and AMERGE on the VU, with ATTPV and AMERGE one chunk behind." width="100%"></p>

The figure shows one tile op from the kernel's `issue()` to its unit queue (above) and the queue contents
of a flash-attention decode layer (below); §4.1 to §4.6 record the decisions behind it.

### 4.1 Issue port

- Memory-mapped (§3.2), not a custom instruction: a custom instruction needs a coprocessor interface
  that Ibex does not have, i.e. a modified core.
- Push: store the words the op uses into `ISSUE_W0`-`W7`, optionally `ISSUE_DEP0` / `DEP1`, then store
  `ISSUE_PUSH` with the word mask (WMASK; words with a clear bit are sent as 0) and the REC flag. At the
  push the issue logic, in one step, takes the words, deps and flags; picks the unit from w0's opcode
  (a static opcode-to-unit table, `ISSUE_UNITS` marks the built units; CC-7); assigns the ticket
  {unit, `ISSUED_SEQ[unit]`++}; writes the entry into the 32-entry issue FIFO; resets the deps to none;
  sets `ISSUE_TICKET`.
- **Credits.** A push needs a free slot of its unit's queue (counted from the push until the unit
  accepts the op) and a free issue-FIFO entry; `ISSUE_CREDIT` shows the minimum (at most 15).
  **Software checks the credit before it pushes**: the library's `issue()` spins on `ISSUE_CREDIT` (a
  local read, interrupts stay serviceable). The hardware wait is only a safety net: a push without a
  credit completes late when a credit returns, for at most `ISSUE_WAIT_MAX` = 65,536 clk_rv cycles,
  then it ends with a store access fault and PUSH_TIMEOUT. The count does not advance while a debug stop
  is active (DB §3.9). While the store waits, Ibex can take neither an interrupt nor a debug halt, and
  the wait can be long in normal operation: with an MVM queue of 4, the 6th outstanding MVM waits for a
  whole MVM, 311,040 core cycles (about 1.04 ms) for a 32B ffn_up / down, more than 65,536 cycles at
  any F_RV considered (393 µs at 166.7 MHz).
- An opcode the issue path does not accept (or whose unit is absent) still gets a ticket, on the SYNC
  unit, and completes with `ILLEGAL_OP`: every push has a ticket and a completion.

### 4.2 Units and queues

The unit table and the depths are CC §7.4. In M6 there are five units with queues: 0 DMA (LD, ST,
KVST; depth 8), 1 QACT (4), 2 MVM (MVM, ATTQK, ATTPV; 4), 3 VU (VOP, RED, ROPE, SOP and AT's vector
ops; 16), 5 SYNC (NOP, FENCE, QCLEAR; 4). Unit 4 is reserved for a future attention engine and is
absent in M6 (`ISSUE_UNITS[4]` = 0, depth 0). Each queue is in order: an op starts after the previous
op of its unit is done, through the unit's dispatch port (`cmd_valid / cmd_ready / done`). Entries
hold the checked unit command, the deps, the flags and the push index. Depths are parameters
(`Q_DEPTH`, reported in EX §5.1); the issue-FIFO depth is independent of them because the credit
includes the free FIFO entries, so a credit-checked push never waits for FIFO space. The queue depths
are 8 / 4 / 4 / 16 / 0 / 4; measured kernels may still move them (§12.4: a 31-deep DMA queue removes
3.2 points of issue loss in the M7 layer test).

### 4.3 Tickets and joins

- Ticket = {unit[31:28], seq[27:0]}. The sequence counts from 0 per unit after a hardware reset and is
  never reset otherwise (RISC-V reset, `SOFT_RESET`, QABORT and QCLEAR keep counting), so tickets stay
  unique across RISC-V restarts.
- Ticket (u, s) is done iff (`DONE_SEQ[u]` - s) mod 2^28 is in [1, 2^27). Completions are in order per
  unit, so one counter per unit is the whole completion state: no per-ticket storage. A ticket held for
  more than 2^27 later ops of its unit becomes ambiguous; the library never holds tickets that long.
- **Stale tickets.** A ticket pushed before the last QCLEAR is stale: as a dependency it is `BAD_DEP`
  (the dispatcher keeps a clear point per unit), and `error(t)` on it is undefined.
- **Push index.** The dispatcher numbers ticketed entries in push order (32 bits) and keeps the number
  with each queue entry; error records keep it (`ERR_PIDX`). It orders errors across units, which
  per-unit tickets cannot. The RISC-V reads the next one as `ISSUE_PIDX` (CC-32), which is how the
  executor maps a hardware error back to an op-list record.
- Joins: `done(t)` (one local load); `join(t)` (spin); `join_irq(t)` (`WAIT_TICKET` = t, then WFI until
  done); `group` joins the latest ticket per unit; `join_all()` pushes a FENCE and joins it.

### 4.4 Dependencies, SYNC and FENCE

- Up to **2 dependencies per op** in hardware: the op does not start before both tickets are done. The
  check is at the head of its queue against the core-domain completion counters (no CDC on this path);
  a waiting head blocks only its own queue.
- Deps on the op's own unit are implied by queue order (the library drops them); deps on one unit keep
  only the latest (in-order units); more than two are joined by a SYNC NOP (its ticket is one dep) or by
  a software join.
- `BAD_DEP` (detected by the checker in push order): a dep on a ticket not pushed yet, on a reserved
  unit, or below its unit's clear point.
- **SYNC unit**: `NOP` completes when its deps are done (a join point with a ticket). `FENCE` completes
  when every op pushed before it, on every unit, is done. `QCLEAR` is a FENCE that then clears the error
  states (§4.5). The dispatcher takes the snapshot a FENCE waits for when it pops the FIFO in push order.
- **Completion rule (invariant I1).** Every entry (one that runs, a poisoned one, an `ABORTED` one, a
  `DEP_FAILED` one, a NOP, a FENCE, a QCLEAR) completes only at the head of its queue and only after both
  its deps are done (a `BAD_DEP` dep is ignored: it may never complete); a FENCE or QCLEAR also waits
  for its snapshot. Hence **done(t) implies that every op t depends on, transitively (its deps, the
  earlier entries of its unit, the entries a FENCE covers, and theirs), is done.** `join_all()` after an
  error therefore still waits for every op pushed before it, and the race detector's clock joins along
  dependency edges (§9.2) are sound.
- Why in hardware: without dependencies the RISC-V must join before every dependent push, so it runs at
  most one stage ahead; with them it pushes a whole pipelined loop and the queues hold the schedule
  (§8.7). Cost: two 28-bit modular compares per queue head (*estimate*: 150-250 LUT per head, six more
  compares at the SYNC head).

### 4.5 Errors

- **Checks at dispatch**, in push order, by `tsr_chk` in the spec's order with the issue-path rules
  first: `ILLEGAL_OP` -> `BAD_DEP` -> `BAD_FMT` -> `K_RANGE` -> `T_RANGE` -> `OPERAND` -> `SPM_RANGE` ->
  `DRAM_RANGE` (EX §8.1). A failing op enters its queue as a poisoned entry and changes no state.
- **At the head** (after the deps), in this order: a poisoned entry completes with its code; an entry of
  a unit in error state completes `ABORTED`; an entry with a dep that completed with an error completes
  `DEP_FAILED`; otherwise it runs. None of these runs the unit.
- **Unit error state.** A unit enters it when one of its entries completes with a nonzero code (at the
  head, in queue order, never at dispatch, so which earlier entries of that unit ran does not depend on
  timing). The first such entry sets `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0` and `ERR_PIDX`. While
  the state holds, every later entry of the unit completes `ABORTED`, and dependents elsewhere fail in
  turn, so an error propagates along queue order and dependency chains.
- **Errors are visible before completion (invariant I2).** `D[u]` does not advance past `ERR_SEQ` until
  the RISC-V side has acknowledged unit u's record (a handshake of about 5-8 core cycles, *estimate*,
  paid once per error state). So `DONE_SEQ` covering a ticket implies that its record is visible:
  `error(t)` is decidable at any clock ratio, and a failed op can never read as a success.
- **First error.** `first_error()` is the record with the smallest `ERR_PIDX`. For a valid program the
  set of failing entries does not depend on the schedule, so neither does the result.
- **QABORT** (RISC-V `QCTRL`, when the dispatcher pops it; host `CTRL.QABORT`, `LQ_CANCEL`, RV_RUN 1 to 0
  and `SOFT_RESET`, at once): every entry not started completes `ABORTED`, in order, each after its deps;
  a running op completes normally (units have no cancel); all six unit slots enter error state until a
  QCLEAR, so every entry pushed after the abort also completes `ABORTED`, whatever the unit and its
  timing.
- **QCLEAR.** RISC-V (`QCTRL`): a SYNC entry with a ticket, in push order, never aborted. It completes
  when every entry pushed before it has completed, then clears all error states and records and sets each
  unit's clear point to its current sequence. Host (`CTRL.QCLEAR`): accepted only when nothing is queued
  or running, the issue FIFO is empty and the RISC-V is quiet; otherwise QCLEAR_REFUSED.
- Why stop on error: a queue holds a pipeline; running past a failed LD would compute on stale SPM data
  silently.
- **Codes**: 1-8 as in the spec; **11 `DEP_FAILED`**, **12 `BAD_DEP`**, **13 `OPERAND`**; 9, 10, 14,
  15 reserved (EX §8.1).

### 4.6 ABORT, SOFT_RESET, resets and halts

The event table is CC §12.1. In summary: QABORT completes queued entries `ABORTED` and lets running ops
finish; QCLEAR applies after every earlier entry; a hold stops heads from starting but non-running
completions drain; RV_RUN 1 to 0 and the debug module's `ndmreset` run the RISC-V reset sequence (§7.3)
with a QABORT; **`SOFT_RESET`** (host `CTRL` bit 10) is EX's device-level reset, run by a core-domain
sequencer that needs no `clk_rv` (CC §12.2, CC-31); a debug halt continues the queues unless
`HOLD_ON_HALT`. There is no partial datapath reset and no cancel of a running op (CC-22); a
wedged unit is recovered by a card reset (EX §8.6).

**Quiescent state.** `QQUIESCENT` = no op held by a datapath unit, the dispatcher and the issue FIFO
idle, every non-empty queue held, and no non-running entry left to drain. Held queued ops are allowed:
they have had no effect. With the RISC-V also halted, the architectural state (RISC-V registers and CSRs,
TCMs, SPM lines written, DRAM) equals the simulator after the same number of retired RISC-V instructions
and the same completion vector `DONE_SEQ[0..5]`, given that run's co-simulation log: the values of
timing-dependent MMIO reads and the trap and interrupt entries with their retire positions (§9.4). With
`HOLD_ON_HALT` it is reached at any breakpoint once the running ops finish, however far ahead the
RISC-V had pushed. This is the control-core analogue of the debug invariant of DB §5.8 and is checked by
random-halt tests (§9.3).

### 4.7 Address fields and the scalar bank

Every address field of an op is an immediate: the RISC-V computes absolute addresses and writes them
(D14). There is one s0-s15 bank inside the VU, read by the host at `SREG` 0x9C0 and by the RISC-V
through the core window (CC §1, EX §5.1). The s bank is written only by VU queue ops in VU queue
order, so it follows rules 2 and 4 of §5.5 like SPM bytes.

### 4.8 Issue cost (estimates and measurements)

- RISC-V side per op (*estimate*, plan): the credit read, 3-7 word stores (only the words the op uses),
  0-2 dep stores, the push and a ticket load, about 6-12 local MMIO accesses and 10-25 instructions to
  compute the words: **25-40 RISC-V cycles** (150-240 ns at 166.7 MHz). CC §7.6 restates it as 15-40
  clk_rv cycles per op.
- Core side (*estimate*, CC §7.6): push to start of an idle unit with done deps about 16-18 core cycles
  (FIFO sync 2-3, pop 1, checker 8, enqueue 1, head load 1, evaluation 1, cmd_valid 1). Checker
  throughput about 34 M ops/s at 276 MHz (8 core cycles per op).
- *Measured on the simulator* (Ibex maxperf table, 166.7 MHz; [CC](design/04-control-core.md) §3.2,
  [SW](design/06-software-verification.md) §7.8): a NOP push costs 36 RISC-V cycles, an 8-word VOP push about
  100 (`kernel.hpp` header comment); on MUL_MAT's decode fast path the issue-path window accesses take
  about 25 cycles per pushed op and the whole per-op cost including the executor is about 100 cycles. The
  executor's cost per op record is in §12.4.

---

## 5. Concurrency in the core

### 5.1 Block diagram

The block diagram is CC §2.1 (two clock domains: `clk_rv` with Ibex, the TCMs, the debug module and
the control window; the core clock with the issue FIFO crossing, `tsr_chk`, `tsr_issue` with the unit
queues, `tsr_csr`, `tsr_spm`, the units) and [`rtl-m2.md`](rtl-m2.md) §2. The queue heads own the
units' command registers.

### 5.2 Checker and dispatcher

- **Checker** (`tsr_chk`): the decode and check pipeline, its own module. One instance serves the
  dispatcher only. Eight core cycles per op (the MVM weight extent too, for power-of-two `NUM_CH`;
  about 46 otherwise); up to four tagged operand extents per op (CC §8.2). The checks are the table of
  CC §8.1.
- **Dispatcher** (`tsr_issue`): pops the issue FIFO in push order. Control entries act at the pop
  (QABORT, QHOLD set / clear, TRACE_MARK); QCLEAR and op entries get the push index and, for FENCE and
  QCLEAR, the snapshot; op entries run the checker; the result (checked command or poisoned entry) goes
  into its unit queue. Room is guaranteed: credits count an op from its push until its unit accepts it, so
  the dispatcher never blocks on a full queue. State machine: CC §8.3.
- **Queue head**: an entry that runs starts when its deps are done, no hold applies and the unit may
  accept it; it then loads the unit's command register. S[u] counts at the accept (credit return), D[u] at
  `done` or at the completion of an entry that does not run, at most one per cycle per unit (CC §9).
- Not in the tree: `tsr_chk`, `tsr_issue` and this design's `tsr_csr` are W4 (`tsr_core.sv` documents
  the X0 state: the units' command ports are idle).

### 5.3 SPM arbitration

- `tsr_spm` uses **round robin** instead of fixed priority, separately for the read and the write
  port, keeping the registered grant: the last granted client has the lowest priority next. A waiting
  client is granted after at most `NCL` - 1 other grants. It is built per bank on a 2R2W even / odd
  organisation (CC §10.1-§10.2, CC-12). The tree still has the three-client, fixed priority
  `tsr_spm` (W4).
- **No-deadlock argument under round robin** (to be written into the `tsr_mem_arb`, `tsr_dma` and
  `tsr_spm` contracts; CC §10.3): (i) every SPM client raises a request only with its payload ready (rule
  R2) and never waits on memory while it holds a request (the DMA's LD writes beats already in its queue;
  QACT, the MVM gather and the VU write results they hold; the bridge holds its data; ST and WS reserve
  buffer space before they request memory); (ii) a granted request transfers in that cycle; (iii) round
  robin grants a waiting client after at most `NCL` - 1 other grants. So every SPM request completes
  within a bounded number of cycles that does not depend on memory progress. That replaces R4's "local
  reasons only" by "bounded, memory-independent reasons", which is what head-of-line routing needs.
- **Head-of-line cost.** Under round robin the DMA's beat queue could fill while other clients write; it
  would then drop `rd_rsp_ready`, and the channel's response path is head-of-line, so weight beats behind
  a DMA beat would wait. The design removes the problem at its source: **DMA LD burst reservation**
  (CC-13). An LD issues a read burst only when its beat queue has room for the whole burst (queue 512 beats, 8
  RAMB36), and `rd_rsp_ready` is constant 1 (CC §10.4).
- The VU needs 2 reads + 1 write per cycle; the even / odd banks give it two read ports (lines L and L+1
  are always in different banks).

### 5.4 DRAM channels

`tsr_mem_arb` round-robins bursts between its sources with one outstanding budget. An LD or ST that runs beside an MVM takes its beats out of that channel's weight stream, and the slowest
channel ends the MVM, so the MVM ends later by at least the LD's beat count. Overlap therefore hides an
LD's latency and fixed costs, not its transfer time. On top of the beats come DDR4 effects: a ST between
weight bursts on one channel adds read / write bus turnarounds, and an LD in another row adds row
conflicts. M0 measured read and write streams separately and together on the card (read 18.01 GB/s
per channel, 72.05 GB/s on four; write 15.92 GB/s per channel, 63.68 on four; read + write 35.6 + 35.6 =
71.2 GB/s on four; measured on the card, M0;
[`m0-bw.md`](m0-bw.md) §7.1), but not the penalty of short writes interleaved with weight bursts: that is
still open and is a W6 measurement (§15). If the penalty exceeds the overlap gain, the SDK can make ST
depend on the same-channel MVM (no hardware change). The DRAM map keeps the op-list, IO and AUX
regions in ch0's low 512 MiB and adds a per-channel compute heap between the weights and the KV
planes (EX §11, SW-26).

### 5.5 Memory model (adopted in the spec)

This section is cited by the race detector (`sw/sim/tsr_race.c`, rules 2 and 4). EX §2.3 (H1-H5) states
the host-side rules; the rules below keep the numbering that the race detector cites.

1. **Per-op results.** Each op's result is defined by the spec's semantics of that instruction and is
   bit-exact whatever runs beside it, if the program is race-free.
2. **Order** comes only from:
   - (a) *queue order*: ops on one unit execute and complete in push order; an op sees every effect of
     the earlier ops of its unit;
   - (b) *dependencies*: an op with a dep on ticket t starts after t is done and sees its effects;
   - (c) *the RISC-V*: after it observed t done (a `DONE_SEQ` read, the WAIT interrupt) its later SPM
     loads see t's effects; an op pushed after a RISC-V store completed sees the store;
   - (d) *FENCE and QCLEAR*: such a ticket is done only when every op pushed before it is done;
   - (e) *completion*: every entry, also one that completes without running, completes only after its
     deps (invariant I1, §4.4), so done(t) implies everything t depends on, transitively, is done.

   Two ops on different units with no dependency chain may run in any order or overlap.
3. **Visible** means the unit's `done`: every SPM write accepted by `tsr_spm`, every DRAM write response
   received ([`rtl-m2.md`](rtl-m2.md) §3.0).
4. **Race**: two accesses to the same SPM byte, DRAM beat or s register, at least one a write, by two
   agents (queue ops, the RISC-V, the host) not ordered by rules 2 and 5. A racy program is a software
   bug and its results are undefined; the hardware does not detect races; the simulator's race detector
   does (§9.2).
5. **Host**: DRAM the host wrote before posting a launch descriptor (the posted writes complete before
   `LQ_PUSH`) is seen by ops pushed after the RISC-V popped that descriptor. DRAM written by ops is seen by
   the host after it reads the answer record the runtime posted after joining those ops (H1 and H2 of
   EX §2.3).
6. Not used.
7. As-if optimisations inside a unit stay allowed (MVM-to-MVM overlap, VU accept-ahead; invariant I3 of
   CC §9.4).
8. **Scalar registers**: the s bank's accesses follow rules 2 and 4 like SPM bytes.

### 5.6 SPM port contention (estimates from the spec's formulas)

Qwen2.5-32B decode shapes, T = 1, Q4_K weights (attn_v and the output head converted to Q8_0), 300 MHz,
every channel at 1 beat per cycle (*estimate*, plan). Duration is the weight beats per channel for an MVM;
the duty is the share of that duration the op needs on an SPM port.

| Op | Duration alone (cycles) | Read-port duty | Write-port duty |
|---|---|---|---|
| MVM ffn_up / gate, N 27648, K 5120 | 311,040 | 0.03 % | 0.56 % |
| MVM ffn_down, N 5120, K 27648 | 311,040 | 0.16 % | 0.10 % |
| MVM attn_k, N 1024, K 5120 | 11,520 | 0.80 % | 0.56 % |
| QACT A256, K 5120 | about 430 | about 75-90 % | about 20-25 % |
| LD of x, 5120 f32 | at least 320 + DRAM latency | 0 | up to 100 % |
| ST of y, 5120 f32 | at least 320 + DRAM latency | up to 100 % | 0 |

- A long MVM leaves more than 99 % of both SPM ports to everything else; at T = 8 its traffic grows 8x
  and stays under 5 %.
- Short ops (QACT, LD, ST, VOP) share the ports among themselves roughly as if serialised.
- The cost an MVM pays for concurrency is the channel beats of concurrent LD / ST and their DDR effects
  (§5.4), not SPM ports.

### 5.7 What concurrency buys (estimates)

- **Dense decode, large model** (32B): non-MVM time is about 1-2 % (planning estimate), so overlap gains
  at most about 1-2 %.
- **Dense decode, small model** (0.5B): G = 3.98 % in the cycle model, plus the host control cost the
  control core removes (§1). With the host issuing the ops, about 10 ms per token were spent outside
  the device on the 0.5B model on the card (measured on the card, k4: 18.29 ms per token against 8.25
  ms of device cycles; [`platform-u200.md`](platform-u200.md) §14.1).
- **Latency-bound phases**: the matvec batch of §8.7 with 0.5B shapes about 15-20 %; attention over KV
  tiles: LD of tile i+1 beside VU work on tile i; MoE (M8): expert ids read from SPM and expert MVMs
  pushed back to back, with no host round trip per layer.
- These gains assume no DDR penalty beyond the stolen beats (§5.4). The main return is host independence
  and latency hiding in short-op phases, not dense-MVM throughput.

---

## 6. Host and control core

The host does not execute tile ops. It launches kernels and the RISC-V issues every tile op, so
nothing else shares the unit queues (EX §2-§4, CC §1).

---

## 7. Clock domains and CDC

### 7.1 Crossings

The crossing table is CC §11.2. The structures: the issue FIFO (RV to core, async FIFO with gray
pointers, 32 entries of 387 bits, LUTRAM); the counters `ISSUED_SEQ[u]`, `S[u]`, `D[u]`, MTIME and the
clk_rv cycle count (gray-coded, 2-flop synchronisers); the error records, the SPM bridge, the core
window, the LQ / LCQ registers, the loader and DMI (tagged handshakes); the log words (a 16 x 32-bit
CDC FIFO); levels and sticky events (synchronisers).

Rules:

- **Gray counters step by at most 1 per source cycle.** Queue drains (aborted and poisoned entries,
  QABORT) complete at most one entry per cycle per unit; an assertion on every gray counter checks it. A
  counter that can move by more than one step at a time (the host-written `LCQ_RD` and `LOG_RD`, and
  everything `SOFT_RESET` flushes) never crosses as a gray code: the RISC-V sees it only through a
  handshake.
- **Nothing crosses into a domain in reset.** Every core to clk_rv request path waits for the
  synchronised "clk_rv domain out of reset" level and, until then or while clk_rv is absent, answers
  locally within the CSR timeout (read 0, write dropped, sticky bit).
- **No CSR access waits on clk_rv without a timeout** (§3.3).

### 7.2 Structures and constraints

- Generic, vendor-independent modules in `rtl/core/cdc/` (planned, W4): `tsr_cdc_fifo` (gray pointers;
  LUTRAM written in the source domain, read into a register in the destination domain), `tsr_cdc_gray`,
  `tsr_cdc_hs` (handshake; one variant with a 1-bit request tag, used by every request path that can time
  out). They use `tsr_sync` and `tsr_rst_sync` from `rtl/tech` (Xilinx: `ASYNC_REG`).
- Constraints (U200: a kernel XDC packaged in the `.xo`): `set_max_delay -datapath_only` of the smaller
  period on gray buses and handshake data, `set_bus_skew` on gray buses, `ASYNC_REG` on synchroniser
  flops. Do not use `set_clock_groups` (it would override the max delays), and do not rely on platform
  clock groups: the platform's clocks come from one clocking block, Vivado may time them as related, and
  300 MHz against 500 or 166.7 MHz has a worst-case edge separation of about 0.67 ns, so one inter-clock
  path missing from the exceptions fails in the link (the out-of-context run cannot see the platform
  clocks), and `v++` auto-scaling may then lower `ap_clk`, the core clock. Gates: `report_cdc` on the
  out-of-context run lists only the intended crossings, all safe; on the linked design, zero unconstrained
  `ap_clk` / `ap_clk_2` paths (CC §11.3).
- `CTRL_SYNC` = 1 (targets where clk_rv = clk): the CDC structures become plain register stages.

### 7.3 Reset sequencing

CC §12.2 is normative. The decisions recorded here:

- **Hardware reset**: `ap_rst_n` resets the core domain; the clk_rv domain's reset is `ap_rst_n` through
  `tsr_rst_sync` (asynchronous assert, synchronous deassert after 2-3 clk_rv cycles). `ap_rst_n_2` is not
  used for CDC endpoints (an async FIFO or counter whose two sides can be reset independently loses
  pointer agreement); at most it is AND-ed into the Ibex / DM reset as `rst_rv_n`. Minimum `ap_rst_n`
  low: max(`RST_PIPE` + 1 core cycles, 3 clk_rv cycles).
- Ibex and riscv-dbg use asynchronous active-low resets: they get theirs from `tsr_rst_sync`, a documented
  exception to [`rtl-m2.md`](rtl-m2.md) §1 for third-party blocks.
- **RISC-V reset** (RV_RUN 1 to 0, ndmreset) is a sequence, not a pulse: the core applies QABORT at once;
  the clk_rv sequencer stops granting new accesses on both Ibex ports and waits for the outstanding one
  (the bridge bound at most), dropping a push that still waits for a credit; it asserts the reset of Ibex,
  the staging registers, deps, `WAIT_*`, the RISC-V hold source and the line buffer for at least 3 clk_rv
  cycles; it raises RV_RESET_DONE. Not reset by it: the CDC endpoints, the core-domain rings, MTIME, the
  TCMs and the debug module's DMI side. So no pointer resynchronisation is ever needed and a debugger
  stays attached across RISC-V resets.
- **Boot order**: core out of reset, the host loads the image, RV_RUN = 1, the RISC-V reset deasserts 2-3
  clk_rv cycles after the synchronised RV_RUN, fetch at `RV_BOOT` + 0x80.
- **`SOFT_RESET`** is not a RISC-V reset: it is the device-level reset of EX §8.5, run by the SR sequencer
  in the core domain (drop RV_RUN, wait for QIDLE, flush the rings and PERF, clear error states).

### 7.4 U200 clock, placement and timing

- **Ports**: `ap_clk_2` / `ap_rst_n_2` on the kernel tops (designed, EX §9, CC §13.3; the tree's adapter
  is the X0 register-only adapter without them). The platform's two kernel clocks (DATA 300 MHz, KERNEL
  500 MHz) are clock-wizard outputs that XRT programs from the xclbin's clock topology (*confirm*), so a
  direct frequency request for `ap_clk_2` is probably supported (`--kernel_frequency "0:300|1:166.7"` or
  `--clock.freqHz`, *confirm* the option form). Only if that fails, divide 500 MHz inside the kernel with
  BUFGCE_DIV in `rtl/tech/xilinx/tsr_clk_div.sv` (CC §11.1).
- **F_RV** (D12): the requirement is at least 100 MHz (*estimate*, §2.1). Planning value 166.7 MHz
  (500 / 3); 250 MHz (500 / 2) only if the out-of-context run of `tsr_ctrl_rv` closes there with at
  least 0.3 ns margin; 125 MHz (500 / 4) as the fallback. Measured on the simulator (`41eb0da`, Ibex
  maxperf): clk_rv at 250 MHz instead of 166.7 MHz scales the RISC-V's time by about 0.67 but gains only
  1.6 % of a 0.5B decode layer's launch time, because the launch is MVM-bound; 250 MHz is headroom for M7
  and launch latency, not a requirement (CC §3.3).
- **Scaling**: if `v++` lowers `ap_clk_2` to close timing, nothing breaks (the CDC is asynchronous);
  software reads `RV_FREQ` and never assumes F_RV.
- **Placement**: SLR1, beside the CSR block, issue path, SPM and DMA (all of the RISC-V's partners are
  there in C'); no new SLR crossing. SLR1 has about 140 K LUT and 350 RAMB36 left after C' and DDR[2]'s
  controller (*estimate*); the control core needs about 14.5-25 K LUT and 49 RAMB36 (CC §14.5). The LUT
  count is not the constraint; local congestion and logic depth next to `tsr_spm` are. Only the bridge's
  request / response registers sit next to `tsr_spm`; the rest (Ibex, TCMs, CDC) goes in a `ctrl_rv`
  sub-pblock away from the URAM columns (CC §14.2, floorplan C'').
- **Dispatch timing**. k4t1's worst out-of-context path is the unit command-register clock enable (0
  levels, +0.109 ns after the margin changes; [`rtl-m3.md`](rtl-m3.md) §14.9). A second command source in
  front of the units would add a LUT level exactly there; the design instead makes the queue head's
  command register the *only* source (CC §9.1: `cmd_q` / `cmd_v_q`), so the unit-side clock enables
  stay one flop away.
- **Timing gate** (D13, CC-23): T1 `tsr_ctrl_rv` out of context at 6.0 ns with at least 0.3 ns
  margin; T2 the k4 core out of context with floorplan C''; T3 the k4 link at or above **276.3 MHz**
  (the measured result without the control core), target 300 MHz after the gather job-pipe fix
  (CC §14.4); T4 the k1 link at 300 MHz. All four are planned (W5). A link needs a host with 14 GB RAM
  and swap: the k4 link took 3 h 51 min (measured).

---

## 8. Software

### 8.1 Device SDK

The SDK is `sw/ctrl` ([`sw/ctrl/README.md`](../sw/ctrl/README.md); normative: SW §6-§7):

```
sw/ctrl/
├── include/tsr/        kernel.hpp (op builders, tickets, issue, joins, group, spm_ring, SPM views),
│                       mmio.hpp / vu.hpp / at.hpp (generated from the table source),
│                       rt.hpp (kernel ABI, TSR_KERNEL), log.hpp, opl.hpp (op-list format),
│                       exec.hpp / plan*.hpp (executor interface and tile planners),
│                       backend_dev.hpp / backend_host.hpp (MMIO backend, simulator backend, §8.5)
├── rt/                 crt0.S, link.ld, runtime.cpp (launch dispatcher), trap.cpp, log.cpp, libc.cpp
├── kernels/            graph.cpp (op-list executor), k_mulmat.cpp and the M7 kind files, selftest.cpp, replay.cpp
├── ref/                C++ references of the kernels (host side)
├── host/               CMake of the host build of the kernels (shim, simulator backend) and of the tsr-elf target
├── tools/              tsr-elf sources (image tool), install-toolchain.sh
├── tests/              the fw_* tests (ctest labels fw and rv)
└── cmake/              riscv32.cmake, tsr_note.S
```

- Compiler: xPack `riscv-none-elf-gcc` 14.2.0-3 (multilib rv32imc / ilp32), pinned in
  `docs/design/toolchain.lock`; `-march=rv32imc_zicsr_zifencei -mabi=ilp32`.
- Freestanding C++20: `-ffreestanding -fno-exceptions -fno-rtti -fno-threadsafe-statics
  -fno-use-cxa-atexit -nostartfiles -ffp-contract=off`, `-Os` for the image and `-O2` for
  `kernels/graph.cpp` and `kernels/k_*.cpp` (SW-34).
- No heap, no libstdc++ headers, libgcc only; no f32 math in firmware (SW-8); inline assembly only for
  CSRs, WFI and `fence.i`.
- The `graph` executor reads op-list records in place through the line-buffered SPM alias, keeps its ticket and
  push-index tables (32 KiB) in DTCM, and executes at most 4,096 records per launch (SW-29, SW-35).
- The image (production `tsr-fw.elf`): ITCM 71,588 B of 131,072 B (budget 81,920 B) and DTCM 49,448 B of
  65,536 B (budget 57,344 B), with the kernels `graph`, `selftest` and `replay` (`tsr-elf size`, tree of
  `a775222`, with the M7 kinds, which are on the simulator and not yet emitted by the backend). The W2a image
  was ITCM 25.8 KiB and DTCM 46.3 KiB (measured, `92f0b17`).

### 8.2 Header library (API sketch)

The listing is a design sketch, updated to the as-built names. The as-built library is
`sw/ctrl/include/tsr/kernel.hpp` (with the generated `mmio.hpp`, `vu.hpp`, `at.hpp`); it differs from the
sketch in these points: the unit of an op comes from a static opcode table and `ISSUE_UNITS`; `deps`
has a constructor and `after()` folds same-unit pairs; the VU and AT builders are generated; `group`,
`spm_ring` (a scratch ring with release tickets), `join2`, `err_units()`, `pidx_now()`,
`stop_requested()` and the `spm_lb_ptr` alias view are additions; and `dram` handles carry the channel
in bits [31:28] and the beat in bits [27:0].

```cpp
namespace tsr {
struct spm  { uint32_t line; };                    // SPM line address
struct dram { uint8_t ch; uint32_t beat; };        // channel + beat
constexpr uint32_t handle(dram d) { return (uint32_t(d.ch) << 28) | (d.beat & 0x0FFF'FFFFu); }  // ABI form
constexpr dram from_handle(uint32_t h) { return {uint8_t(h >> 28), h & 0x0FFF'FFFFu}; }
struct wref { uint32_t w_off; uint8_t fmt; };      // symmetric-allocated weight tensor (FMT_*)
enum class qa : uint8_t { a32 = 0, a256 = 1 };
constexpr qa qa_for(wref w) { return w.fmt == 0 ? qa::a32 : qa::a256; }   // Q8_0 -> A32, Q4_K -> A256
constexpr uint32_t f32_lines(uint32_t n) { return (4 * n + 63) / 64; }

struct op { uint32_t w[8]; uint8_t used; };        // one 256-bit tile instruction, used-word mask
constexpr op LD  (spm dst, dram src, uint32_t lines);
constexpr op ST  (spm src, dram dst, uint32_t lines);
constexpr op QACT(qa kind, spm dst, spm src, uint32_t k, uint32_t t, uint32_t src_stride = 0);
constexpr op MVM (spm dst, spm x, wref w, uint32_t n, uint32_t k, uint32_t t,
                  uint32_t dst_stride = 0, bool acc = false);
// VOP, RED, ROPE, SOP, ASMAX, AMASK, AMERGE, ANORM, ATTQK, ATTPV, KVST (generated), NOP, FENCE likewise

struct ticket { uint32_t v = 0xF000'0000u;         // default: none (always done)
                bool none() const { return (v >> 28) == 15; }
                uint32_t unit() const { return v >> 28; } uint32_t seq() const { return v & 0x0FFF'FFFF; } };
struct deps { ticket a, b; };
inline deps after(ticket a = {}, ticket b = {}) { return {a, b}; }   // library folds same-unit deps

ticket   issue(const op&, deps = {}, uint32_t flags = 0);   // async: waits only for a credit; not reentrant
bool     done(ticket);
void     join(ticket);                 // spin; join_irq(ticket) uses WAIT_TICKET + WFI
void     join_all();                   // FENCE + join
uint32_t error(ticket);                // needs done; 0 or its code; undefined for tickets before the last qclear()
uint32_t first_error();                // the error with the smallest push index over all units, 0 if none
ticket   qclear();                     // QCTRL.QCLEAR: a SYNC ticket, done = every earlier entry done, states cleared
template <class T> volatile T* spm_ptr(spm l, uint32_t byte_off = 0);   // 0x4000_0000 + 64 l + off
}
```

The device backend of `issue`, as built (`kernel.hpp`; `hw::push` stages the words of `op::used`, the deps
that are not none, writes `ISSUE_PUSH` and returns `ISSUE_TICKET`):

```cpp
TSR_HOT ticket issue(const op & o, deps d = {}, uint32_t flags = 0) {
    const uint32_t u = detail::push_unit(o.w[0]);          // static opcode -> unit table (ISSUE_UNITS, CAPS)
    while (detail::credit_of(hw::cw_read(CW_ISSUE_CREDIT), u) == 0u) hw::pause();   // local spin
    const uint32_t push = (uint32_t(o.used) << ISSUE_PUSH_WMASK_LSB) | (flags & ~ISSUE_PUSH_WMASK);
    return detail::pushed(hw::push(o, d.a.v, d.b.v, push));
}
TSR_HOT bool done(ticket t) {                              // 1 <= (DONE_SEQ - seq) mod 2^28 < 2^27
    if (t.none()) return true;
    return seq_done(hw::cw_read(CW_DONE_SEQ(t.unit())), t.seq());
}
inline uint32_t error(ticket t) {                          // valid once done(t): the record arrived first (I2)
    if (t.none()) return 0u;
    const uint32_t code = hw::cw_read(CW_ERR_CODE(t.unit())) & 0xFFu;
    if (code == 0u) return 0u;                             // unit not in error state
    const uint32_t x = (t.seq() - hw::cw_read(CW_ERR_SEQ(t.unit()))) & TICKET_SEQ;
    if (x >= 0x0800'0000u) return 0u;                      // t completed before the first error
    return x == 0u ? code : ERR_ABORTED;                   // the failing entry, or ABORTED after it
}
```

`issue()` relies on the single hart and on not being reentered (§3.2): nobody can take the credit between
the check and the push.

### 8.3 Runtime, image format, launch protocol, kernel ABI

The as-built design (SW-10):

- **Image and kernel table**: EX §3 and SW §6.2. One static ELF, no relocations; `.vectors` at `RV_BOOT`
  with `_start` at +0x80; `.text` in ITCM; `.rodata`, `.data`, `.bss` and the stack (8 KiB, top of
  DTCM) in DTCM; a kernel table `.tsr_kernels` of 36-byte entries (name hash, entry, nargs, flags,
  argument-block lines, name) sorted by rank and name (`graph`, `selftest`, `replay` are kernels 0, 1, 2);
  `.note.tsr` records the Basis ABI, the Tesseract Core capability and TCM sizes it needs, required CAPS bits and its Basis version;
  `.tsr_id` holds the 64-bit image id (a SHA-256 prefix of the loaded segments) that `tsr-elf stamp`
  writes.
- **Launch protocol**: EX §4. The host writes an 8-word descriptor into the launch queue LQ (depth 8) by
  posted CSR writes; the runtime pops it, validates it, calls the kernel and posts exactly one 8-word
  record per descriptor in order in the completion ring LCQ (depth 8); the host polls. A hard cancel and
  a soft stop work through watermarks on launch numbers; kernel events (records 0xC0-0xFF) and a
  256-word log ring share the path.
- **Runtime R1-R10**: EX §4.7 (normative), implemented in `sw/ctrl/rt/runtime.cpp`: boot and HELLO; take
  (WFI on `irq_fast[2]`); validate (`BAD_DESC` precedes `ABORTED`); run (argument block loaded into the
  top 16 SPM lines when the kernel's flag says so, TRACE_MARK around the kernel); `join_all`; first error
  by smallest `ERR_PIDX`; post (WFI on `irq_fast[3]` while the ring is full) with DURATION in core cycles;
  `qclear` after an error; back to take. Traps and panics: `rt/trap.cpp` and DB §6.3.
- **Kernel ABI** (SW §6.5): up to 7 arguments of 32 bits (integers, SPM line numbers, or DRAM handles
  with the channel in bits [31:28]); a kernel that needs more takes an argument block of up to 1 KiB that
  the runtime loads into the SPM argument range (the top 16 lines).
- **Launch statelessness** (SW-4): a launch's results depend only on the image, the descriptor
  arguments and DRAM contents; the simulator reports `SPM_STALE` for reads of SPM lines that nothing wrote
  since the pop.

### 8.4 Host API (`tsr_transport`)

The host API is `sw/transport/tsr_transport.h` (EX §10.1 verbatim plus SW §9.1 conveniences):
`tsr_transport_open`, `tsr_image_open`, `tsr_transport_boot`,
`tsr_transport_launch`, `tsr_transport_wait`, `tsr_transport_call`, `tsr_transport_cancel`,
`tsr_transport_stop`, `tsr_transport_soft_reset`, log drain and the raw CSR accessors, over one protocol
layer for the `sim`, `mock`, `vkernel` and `xrt` kinds. See [`sw/transport/README.md`](../sw/transport/README.md)
and [`xrt-host.md`](xrt-host.md). Launches are posted writes, up to `LQ_DEPTH` outstanding, and the
estimate for a launch round trip is 4-8 µs (*estimate*, EX §4.11).

### 8.5 Host build of the same headers

The same kernel sources compile for the host with `backend_host.hpp` (`TSR_FW_HOST`): `issue` calls the
simulator's queue model through its test hooks, without the RV interpreter. Fast unit tests, host GDB,
ASan / UBSan on kernel logic. It is a convenience, not an oracle: the reference is the RV binary on
`tsr_rv` (exact control flow, exact MMIO timing model). The headers use fixed-width types for device
addresses and never hand host pointers to the device. The test `fw_kernels_host` and its `rv` twins check
that the host build pushes the same op stream (words, deps, flags) as the RISC-V build (SW §6.8).

### 8.6 JIT integration (sketch)

**Idea (not scheduled).** A registry of precompiled kernels, launch by name, and a tinygrad or
torch.compile backend that maps scheduled kernels onto it; no code generation, weight placement reused
from the backend. The first such client exists in a simpler form: `tsr_rt` (the PyTorch path) emits one-record op lists and launches the `graph` kernel
([`sw/rt/README.md`](../sw/rt/README.md), SW §9.4), and PyTorch on the full kernel library is M8. The
caveat stands: matching generic framework kernels onto a fixed kernel set covers the LLM hot path
(matmul, norms, attention, sampling), not arbitrary graphs.

### 8.7 Example: matvec batch with prefetch

**Design sketch; the as-built library is `kernel.hpp`.** The as-built form of a kernel is
`extern "C" int32_t name(const tsr_args * args)` with the declaration `TSR_KERNEL(name, nargs, flags,
blk_lines)` (`rt.hpp`); `args->blk` points to the argument block in the SPM window. The pattern below is
exercised on the simulator by the `selftest` kernel (case 6, `CONC`) and is the push order of the MUL_MAT
kind (`sw/ctrl/kernels/k_mulmat.cpp`). The listing below is a sketch that compiles against `kernel.hpp` in the
host build (`-DTSR_FW_HOST`); it is not part of the image.

```cpp
#include <tsr/kernel.hpp>
using namespace tsr;

struct mv_job { uint32_t x, y, w_off, fmt, n, k; };   // y = W x; x, y: DRAM handles of f32 vectors; 24 B

// Double-buffered batch: the LD of x[i+1] and the ST of y[i-1] run on the DMA unit while MVM i
// streams its weights. The RISC-V never waits inside the loop except for a free queue slot, and
// joins once at the end. a->blk points into the SPM argument range (the runtime LDed the block and
// joined before the call; 1 KiB holds 42 jobs). Buffers: X 2048 lines (K <= 32768 f32),
// Q >= 584 lines (QA_LINES(A256, 32768)), Y 2048 lines (N <= 32768), all far below the argument range.
TSR_KERNEL(matvec_batch, 2, TSR_KFLAG_ARGBLK, 16);
extern "C" int32_t matvec_batch(const tsr_args * a) {
  const volatile mv_job * job = reinterpret_cast<const volatile mv_job *>(a->blk);
  const uint32_t n = a->a[1];
  static constexpr spm X[2] = {{0}, {2048}}, Q[2] = {{4096}, {5120}}, Y[2] = {{6144}, {8192}};
  ticket ld[2], qa_[2], mv[2], st[2];                                 // "none": already done
  ld[0] = issue(LD(X[0], from_handle(job[0].x), f32_lines(job[0].k)));
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t b = i & 1;
    const wref w{job[i].w_off, uint8_t(job[i].fmt)};
    const uint32_t jn = job[i].n, jk = job[i].k, jy = job[i].y;
    // QACT i reads X[b] (LD i) and overwrites Q[b], which MVM i-2 read
    qa_[b] = issue(QACT(qa_for(w), Q[b], X[b], jk, 1), after(ld[b], mv[b]));
    // MVM i reads Q[b] and overwrites Y[b], which ST i-2 read
    mv[b] = issue(MVM(Y[b], Q[b], w, jn, jk, 1), after(qa_[b], st[b]));
    // prefetch x[i+1] into X[b^1] (QACT i-1 read it); pushed before ST i, so it does not queue behind MVM i
    if (i + 1 < n)
      ld[b ^ 1] = issue(LD(X[b ^ 1], from_handle(job[i + 1].x), f32_lines(job[i + 1].k)), after(qa_[b ^ 1]));
    st[b] = issue(ST(Y[b], from_handle(jy), f32_lines(jn)), after(mv[b]));
  }
  join_all();
  return int32_t(first_error());                                      // 0, or the first failing code
}
```

- **Schedule.** DMA queue order: LD0, LD1, ST0, LD2, ST1, LD3, ... LD_{i+1} waits for QACT_{i-1}; ST_i
  waits for MVM_i. In steady state MVM_i runs while the DMA does ST_{i-1} and LD_{i+1}, and QACT_{i+1} runs
  as soon as LD_{i+1} and MVM_{i-1} are done, i.e. during MVM_i, so the MVM unit runs back to back. Every
  op has at most two dependencies, so no SYNC NOP is needed.
- **Without hardware dependencies** the loop would `join(qa_[b])` before each MVM and `join(mv[b])` before
  each ST: the RISC-V would wait every iteration and the DMA could not run ahead.
- **Gain** (*estimate*, plan, 300 MHz, MVM fill about 300-600 cycles, DRAM latency about 150 cycles, no DDR
  penalty beyond the stolen beats, §5.4): 0.5B shape (N = K = 896, Q8_0, 3,360 weight beats per channel):
  sequential about 4,650-4,950 cycles per job, pipelined about 3,770-4,070 (MVM + fill + 112 ch0 beats
  stolen by LD / ST), about 15-20 %. 32B shape (N = K = 5120, Q4_K, 57,600 beats): about 59,500 against
  about 58,700, about 1.4 %. The pattern pays where ops are short; on long bandwidth-bound MVMs it saves
  only the fixed costs (§5.4).

---

## 9. Verification

CC §15 and SW §10 are normative. What exists is on the simulator; the RTL half is W4 / W5.

### 9.1 Simulator

- **`sw/sim/tsr_rv.c`**: an RV32IMC + Zicsr interpreter that models the chosen Ibex configuration's CSRs
  and WARL behaviour (vectored `mtvec`, the implemented counters, misaligned splitting, access faults,
  WFI, the debug-mode subset the debug module needs), with two cycle tables (Ibex maxperf and small;
  `TSR_SIM_RV_CORE`, default maxperf). In-house rather than linking Spike: deterministic co-scheduling
  with the tile model, MMIO and trace hooks, speed, and the simulator stays one C library linked into the
  llama.cpp backend. Spike is the interpreter's **oracle**: riscv-tests and riscv-arch-test signatures,
  random programs compared instruction by instruction, and `test_ref_fp.c` built for RV32. With the pinned
  toolchain installed the tests `rv_isa`, `rv_arch`, `rv_lockstep` and `rv_random` run; without it the
  Spike-based tests skip (SW §10.2; `ctest -L rv`).
- **Queue model** (`tsr_q.c`) and the **scheduler** (`tsr_sched.c`): the control window, queues and units as
  an event model on the core time base with every rule of §4-§5 (completion rule, error state at the head,
  I2 ordering, hold sources, QCLEAR in push order). Op durations come from the cycle model or are
  randomised (`TSR_SIM_SCHED=fixed|random:<seed>[,<min>,<max>]`); RISC-V time from a CPI table scaled by
  F_RV / F_CORE (`TSR_SIM_RV_RATIO`). An op's effects are applied atomically at its modelled completion;
  for race-free programs the choice is invisible.
- `TSR_SIM_LOG` gives RISC-V retire lines (prefix `rv,`) and push / start / done events.
- The simulator's device cycles are an informational cycle model; hardware counters are compared against
  it, not bit-matched.

### 9.2 Race detector

- **Happens-before** by vector clocks over the agents {the units, the RISC-V, the host}, with the edges of
  §5.5 (SW §4.6): queue order, dependencies, RISC-V push, RISC-V observation of completion (a `DONE_SEQ`
  read returning v joins the clocks of every completion of that unit below v; the WAIT interrupt likewise),
  RISC-V program order, FENCE and QCLEAR, and the launch protocol (EX §2.3: a host DRAM write completed
  before `LQ_PUSH` of launch s happens before every op of s; every op of s happens before the host's read of
  s's answer record). The joins along dependency edges are sound because of invariant I1.
- **Access sets per op** from its operands: LD, ST; QACT with T and stride; MVM's x, dst (ACC also reads
  dst) and weights per channel; VU and AT ops from their documents; RISC-V SPM accesses per byte; the s
  bank as 16 locations.
- **Shadow state**: per SPM line the last writer and the latest read per agent; per DRAM channel an
  interval map. A report names both tickets (or the RISC-V pc), the ranges and the ops.
- Also: reads of never-written SPM (`SPM_STALE`, the launch-stateless check), dependencies on stale tickets,
  a host QCLEAR that would be refused.
- `TSR_SIM_RACE=off|report|fatal` (ctest uses `fatal`; the tests `sim_race`, `sim_race_rv` and the `_race`
  variants). The overhead measured at W1b is 1.0-1.3x on the test programs (measured on the simulator,
  `1ed7460`); the plan estimated 2-3x.
- Detection does not depend on the schedule for valid programs (§9.5), so one run finds every race.

### 9.3 RTL

**Planned (W4 / W5).** The plan, kept as the test list for the RTL; the normative version is CC §15:

- **Ibex itself**: a pinned, upstream-verified commit; its own DV is not rerun. Our integration (TCMs,
  buses, control window) runs riscv-tests and riscv-arch-test from ITCM under Verilator; signatures
  against Spike.
- **Issue path and queues** (`tb/ctrl`, cocotb + Verilator): credits, the safety-net stall and its
  timeout; ticket numbering and wrap; deps including `BAD_DEP` and `DEP_FAILED`; SYNC NOP, FENCE and
  QCLEAR; invariants I1 and I2 as assertions on every test; an `ILLEGAL_OP` push then `join_all()` with
  ops in flight (must wait); error state at the head and `first_error()` with errors on two units; QABORT
  then pushes (all `ABORTED` until QCLEAR); QCLEAR right after a push; hold sources; RISC-V reset during
  a bridge access and during a push; SPM bridge widths, alignment, faults, ordering against pushes, bound;
  every CSR access with clk_rv stopped and with the clk_rv domain held in reset (must complete,
  RV_CLK_OK = 0); DMI timeout followed by the late answer (discarded).
- **Two clocks**: a Verilator C++ bench with independent `clk` and `clk_rv`, ratios of F_RV / F_CORE from
  0.31 to 1.67 and random phase; every test at every ratio, including a directed error test per ratio; the
  `CTRL_SYNC` = 1 build as well.
- **SPM and memory**: `tb/dma` and `tb/spm` with the banked SPM, and the deadlock case of §5.3.
- **Concurrency stress**: random race-free programs (checked by the race detector) run on RTL and
  simulator under the random-latency memory model of `tb/common` and random SPM contention: final state
  bit-exact.
- **Random halts**: HOLD_ON_HALT halts at random points of stress programs; at each quiescent point
  compare with the simulator advanced to the same retired count and completion vector (§4.6).
- **Mutation campaign** (like [`rtl-m3.md`](rtl-m3.md) §13.3): drop a dependency compare, complete an
  aborted or poisoned entry before its deps, reorder a queue, skip a poison, advance D[u] before the
  record acknowledge, step a gray counter by 2, let one hold source's write clear another, a CDC FIFO
  full flag one cycle late, a credit returned at `done` instead of accept. The tests must catch each.
- **Vivado**: gates T1-T4 (§7.4); `report_cdc` clean; zero unconstrained `ap_clk` / `ap_clk_2` paths.

### 9.4 Co-simulation and traces

- **Lockstep** (planned, W5): Ibex's RVFI outputs in simulation builds give per retired instruction {pc,
  insn, rd, value, memory address, data} and trap / interrupt entries (*confirm* the fields of the pinned
  commit); `tsr_rv` steps in lockstep. Loads of timing-dependent registers (`DONE_SEQ`, `ISSUE_CREDIT`,
  MTIME, `QSTATUS`, `LQ_AVAIL`, the error records) take the RTL value, and trap and interrupt entries are
  taken at the RTL's retire position; every other value must match. That log is what the quiescent-state
  invariant (§4.6) is conditioned on.
- **Trace format** (on the simulator): a trace is a directory per launch with `meta.json`, the firmware image
  by id, `ops/push.bin` (the push-ordered tile-op stream with relative tickets), the initial DRAM regions
  and the expected regions and tickets (SW §5; [`traces/README.md`](../traces/README.md)). `TSR_TRACE`
  dumps one trace per launch.
- **Two replays**: *full* (firmware on the simulator, later RISC-V + core RTL) and *tile-only* (a bench
  pushes `push.bin` through the issue path with its dependencies, no RISC-V), which separates core
  concurrency bugs from RISC-V integration bugs. Tile-only replay into the RTL is planned (W5).

### 9.5 Bit-exactness for concurrent programs

A program (RV image + inputs + host message script) is **valid** if (1) it is race-free (§9.2) and (2) its
observable behaviour does not depend on the schedule: the op stream per unit, the answer-record stream, the
RISC-V's SPM store stream (address, data, order) and the final SPM and DRAM are the same under every
schedule. That is the definition; the check is the determinism test: at least 8 random schedules in the
simulator, comparing all four (`sim_sched_determinism`, `fw_kernels_rv` and the other `_rv` tests run 8
schedules with the race detector fatal). The simulator's schedule space (a CPI table, random op latencies)
need not cover every RTL timing, so lockstep co-simulation at every clock ratio (§9.4) covers the rest: a
program that passes the 8 schedules but diverges on the RTL shows up as a timing-dependent MMIO value
followed by a different store or push, which is reported against the program, not the RTL.

For a valid program the final state is unique, and the RTL and the hardware must reproduce the simulator's
exactly: the DRAM regions written, the SPM lines written (where compared), the answer records and the
ticket error records (including error paths: §4.5 makes them deterministic). Not compared: RISC-V poll
iteration counts, MTIME, `PERF_*`, completion records. Kernel rule (SW-30): timing-dependent values
(MTIME, credits, poll counts) must not influence data, stores or pushed ops, except in logging that the
comparison excludes.

---

## 10. Debug and profiling

The normative text is DB ([`docs/design/05-debug.md`](design/05-debug.md)); the working summary and the implementation
status are in [`prof-dbg.md`](prof-dbg.md). In summary: the debugger talks the GDB remote serial protocol to
a server built into the device-owning process (or `tsr-dbg`) that drives the debug module's DMI directly
through the CSR window (no OpenOCD, no software JTAG TAP); memory is reached through the connected system
bus access (SBA); tile-level debug adds stop points, serial mode, tile breakpoints, a 512-entry dispatch
history and unit probes; the profiler adds a hardware trace writer to a DRAM ring and per-channel rates
to the `TSR_STATS` report. The DBG register block is at core CSR 0x800-0xAFC (DB §10.1).

### 10.1 RISC-V debug

The debug module is pulp riscv-dbg (Debug Spec 0.13.2: halt, resume, step, abstract register access,
an 8-word program buffer, ndmreset) in the clk_rv domain with `DmBaseAddress` 0x1000_0000 and its
two-scratch debug ROM (*confirm*); Ibex's two execute triggers are the hardware breakpoints; software
breakpoints go in ITCM (writable); the host path is the CSR window `DMI` (0xC00) through a tagged
handshake. SBA is connected and carries all debugger memory access (CC-16, DB-3). ASIC and JTAG
targets would reach the DMI through the real DTM at DMI word address 0x300
(0xC00 / 4) (see [`prof-dbg.md`](prof-dbg.md)).

### 10.2 Profiling

Completion records (128 bits per entry, a 64-entry FIFO behind per-unit holding registers, dropped and
counted when full) serve firmware (DB §8.2); the host timeline comes from a hardware trace writer to a
DRAM ring on channel 0 (DB §8.3, DB-16). The PERF counters are EX's PERF space (EX §6); Ibex
`mcycle` / `minstret` give the RISC-V's own cost.

---

## 11. Roadmap

### 11.1 Phases

The plan is the work packages of [`docs/design/README.md`](design/README.md) §10: W0 (contract and X0), W1
(simulator and references), W2 (firmware SDK and kernels), W3 (host, backend, tools), W4 (RTL), W5
(Verilator and Vivado integration), W6 (card).

### 11.2 Milestones

M6 is the control core and the M6 backend (MUL_MAT as op lists through the `graph` kernel); M7 is one
launch per decode token with attention on the device; M8 is device MoE; M9 is the resident loop. The
status per milestone is in `docs/spec.md` §19.

### 11.3 Scalar control

The tile ISA has no scalar-control ops: the RISC-V does the arithmetic, reads the SPM directly, loops
in C++ and polls. The issue path refuses opcodes it does not accept with `ILLEGAL_OP` (EX §7.2). `TOPK`
and `DQROW` stay tile ops (M8), `VMASK` arrives with M9. On a target with its own CPU (Zynq's PS) host
independence would come from that CPU driving the same MMIO issue port, with the same tickets and
queues, so no second mechanism is needed.

---

## 12. Cost

### 12.1 U200 (estimates)

**Estimates** (CC §14.5; *estimate*, not synthesised; net of the datapath logic that the issue path makes unnecessary):

| Block | Domain | LUT | FF | RAMB36 | DSP |
|---|---|---|---|---|---|
| Ibex (maxperf pipeline) | rv | 3-5 K | 1.7-2.8 K | 0 | 3-4 |
| riscv-dbg `dm_top` + SBA arbiter | rv | 1-2 K | 1-1.5 K | 0 | 0 |
| ITCM 128 KiB + DTCM 64 KiB | rv | 0.1-0.3 K | - | 48 | 0 |
| bus, control window, LQ / LCQ / log RV side, timer, line buffer | rv | 2-3.5 K | 3-4 K | 0 | 0 |
| CDC (FIFOs, gray counters, handshakes) | both | 0.6-1.2 K | 1.5-2.5 K | 0 | 0 |
| `tsr_issue` + `tsr_chk` (queues, 12 dep compares, SYNC snapshot) | core | 4-7 K | 3-5 K | 0 | 1-2 |
| `tsr_csr` (EX map, DMI, PERF, rings, SR sequencer, loader) | core | 4-6 K | 3.5-4.5 K | 1 | 0 |
| datapath sequencing and register file not needed with the issue path | core | -2 to -3 K | -2 to -3 K | 0 | 0 |
| **Total, net (without SPM / DMA changes)** | | **about 12.5-23 K** | **about 10-18.5 K** | **49** | **4-6** |

Against the SLR1 budget (CC §14.1): control core + issue path + CSR block 14.5-25 K LUT before the credit row;
the VU about 48 K LUT, 15 RAMB36, 366 DSP (+-30 %); DB 4.2-6.3 K LUT; SLR1 total about 125-143 K LUT
(62-72 %) against the 65 % gate (CC-24; the estimate straddles the gate, so the out-of-context VU run decides
whether the shrink knobs of CC §14.1 are needed). Attention cluster modes add about 19-27 K LUT per cluster in SLR0 / SLR2.

The 64 PERF slots are in the core domain (EX §6) and are budgeted in `tsr_csr`; 64 x 64-bit
incrementers alone are about 4 K LUT.

### 12.2 ASIC (estimates)

These are the only ASIC numbers (*estimate*; not redone for the maxperf pipeline or the current
storage sizes):

- Ibex small about 24-27 kGE (Ibex README configuration table, *confirm*; maxperf is 32.48 kGE in the same
  table); DM about 15-30 kGE; control window, CDC, dispatcher and CSRs about 30-50 kGE excluding storage.
- **Storage** is about 25 Kb in this estimate (issue FIFO 32 x about 300 b = 9.6 Kb, record FIFO 64 x 128 b = 8 Kb,
  queues about 7.5 Kb). As flops with read muxes that is about 125-175 kGE (5-7 GE per bit), 3-5x Ibex; the
  FPGA's cheap LUTRAM hides it. With 2-port register-file macros, or with an 8-entry issue FIFO (about
  45-65 kGE) it is smaller. The issue-FIFO entry is 387 bits (CC §7.2) and the queues are about 12 Kb of
  LUTRAM (CC §14.5), so the storage share is larger than this estimate.
- PERF counters as flops (29 x 64 b) are about 20 kGE; EX §6 has 64 PERF slots in the core domain.
- TCMs: 128 KiB of 1RW / 1R1W SRAM macros are about 0.2-0.4 mm^2 in 28 nm, through `tsr_sram_*` wrappers (no
  2RW macro, which would be 1.5-2x the area of a 1R1W one); the TCMs total 192 KiB.
- The total is about 0.4-0.7 mm^2 in 28 nm next to the 4 MiB SPM (about 8-15 mm^2): a few percent. Clock: an own PLL output, or the core clock with `CTRL_SYNC` = 1.

### 12.3 Effort (estimates)

The software path was estimated at 5-7 K lines (interpreter 2 K, queue model 1 K, race detector 0.8 K, SDK
1.5-2 K, transport 0.6 K) and the RTL at 3.5-4.5 K lines of SystemVerilog plus about 3.5 K lines of benches
(about 20 K lines vendored). The implemented software path (`sw/sim`, `sw/ctrl`, `sw/transport`) covers
W1-W3a and is larger than that estimate.

### 12.4 Measured RISC-V costs on the simulator

All values are device-time measurements of the **simulator's** cycle model with the Ibex maxperf table
(`TSR_SIM_RV_CORE=maxperf`, default since `41eb0da`), a fixed schedule, `clk_rv` 166.7 MHz against the 276.3 MHz
core clock, unless noted; they are not card measurements. "Work" means the RISC-V's busy cycles up to the
launch's last push, without the iterations of wait loops after their first test (`sw/ctrl/tests/fw_prof.h`).
Sources: commit messages of `92f0b17` and `41eb0da`; [SW](design/06-software-verification.md) §7.8 for the
rest.

| Quantity | Value | Provenance |
|---|---|---|
| Launch: descriptor push to kernel entry | 2.19 µs (W2a, small table); 1.68 µs (maxperf) | measured on the simulator (`92f0b17`; maxperf: after `41eb0da`) |
| Launch: empty launch, push to DONE in the LCQ | 5.57 µs (W2a, small table); 4.23 µs (maxperf) | measured on the simulator (`92f0b17`; maxperf: after `41eb0da`) |
| Executor, work per op record (average over the 0.5B decode layer's 7 MUL_MAT + 2 JOIN records), deep queues | 1,518 (W2a firmware) to 387 cycles (maxperf); small table 1,966 to 485; per MUL_MAT record 498 (small 624) | measured on the simulator (`41eb0da`; per-record split: SW §7.8) |
| Same by W2a's method (every busy cycle, U200 queue depths) | 1,547 to 437 cycles (maxperf). The commit headline "1,976 to 387" sets W2a's measurement (small table, every busy cycle) against the new maxperf deep-queue work; like for like it is 1,547 to 437 or 1,518 to 387 | measured on the simulator (`41eb0da`) |
| One Qwen2.5-0.5B decode layer (7 MUL_MAT + 2 JOIN records) | RISC-V 110.4 to 39.4 µs against 197 µs of MVM work; MVM issue-idle 8.1 % to 0.0 % | measured on the simulator (`41eb0da`) |
| JOIN record / NOP record | 140 / 85 cycles (W2a: 185 / 156) | measured on the simulator (after `41eb0da`) |
| Image | ITCM 32,808 B, DTCM 47,968 B (executor optimisation) | measured (`tsr-elf size`, after `41eb0da`) |

W2b (the M7 kinds: RMS_NORM, ROPE, element-wise ops, GLU, GET_ROWS, SET_ROWS_KV, FLASH_ATTN, CPY, UNARY) is committed
(`a775222`) and verified at layer level on the simulator; the backend does not emit these kinds yet (W3b). Its measured
costs (SW §7.8, simulator cycle model, re-run on the tree of `a775222`): every short kind meets the 350-cycle
requirement at the Qwen2.5-0.5B decode shapes (223 to 336 RISC-V cycles of work per record; RMS_NORM 324, ROPE 325),
while Gemma 4's per-head norms (405 to 455), ROPE with a frequency-factor table or `freq_scale` other than 1 (439 to
475) and records whose scalar-register values change still exceed it (open, SW §7.8). A real Qwen2.5-0.5B M7 decode
layer (15 records) loses 5.1 % of its launch time to issue with the U200 queue depths (2.1 % with deep queues, 2.1 %
with only the DMA queue 31 deep, 2.5 % at 250 MHz), because the 8-deep DMA queue fills with LD / ST ops that wait
behind an ST and because FLASH_ATTN's planning (2,679 cycles per record at n_kv 256) sits on the critical path
between V and the output projection. A deeper DMA queue in the RTL and a leaner FLASH_ATTN planner are the two open
remedies.

---

## 13. Deliberately not done

- No cache, branch prediction, superscalar or out-of-order execution (D1).
- No custom RISC-V instructions: the core stays unmodified; issue is MMIO.
- No direct DRAM access from the RISC-V (§3.6); no instruction fetch from SPM or DRAM.
- No F extension (§2.4); no atomics, no second hart, no operating system.
- No hardware race detection (the simulator does it).
- No cancellation of a running tile op: QABORT drops entries that have not started.
- No second issue staging bank (`issue()` is not reentrant, §3.2).
- No epoch field on issue-FIFO entries: after an abort every unit stays in error state until a QCLEAR that is
  ordered with the pushes, which gives the same guarantee (§4.5).
- No line buffer on the exact SPM window; the line buffer is on a separate alias address (§3.5) so that the
  exact window keeps its simple ordering.
- No unit reset: a wedged unit needs a card reset (CC-22, EX §8.6).

---

## 14. Review revisions

The design was reviewed against the sources (spec, rtl-m2, rtl-m3 §14, prof-dbg, platform-u200,
xrt-host, `tsr_spm.sv`, `tsr_dma.sv`, `tsr_mem_arb.sv`, `tsr_csr.sv`). Every finding was
checked; none reopened a user decision. The `docs/design` design set was then reviewed twice (two reviewers, 42
findings in total, [`docs/design/README.md`](design/README.md) §5 and §7). Verdicts of the first review:

| Finding | Verdict | Applied in |
|---|---|---|
| Aborted / poisoned entries and FENCE completed without waiting for deps; `join_all()` returned early after any `ILLEGAL_OP` (blocker) | confirmed, fixed: completion rule I1 | §4.4, §4.5, §5.5 2(e) |
| Error record could arrive after `DONE_SEQ` (reported twice) | confirmed, fixed: I2 | §4.5, §7.1 |
| Queue control out of band; QIDLE racy; abort scope undefined | confirmed, fixed: RISC-V QCTRL through the FIFO, all units in error state until QCLEAR, ticketed QCLEAR. Rejected part: an epoch counter | §3.2, §4.5, §4.6, §13 |
| Credit wait longer than `ISSUE_WAIT_MAX` in normal operation | confirmed (1.04 ms against 393 µs), fixed: software credit check, hardware wait as safety net | §4.1, §8.2 |
| Halted-and-drained state unreachable, not verifiable | confirmed, fixed: QQUIESCENT | §4.6, §9.4 |
| CSR accesses that wait on clk_rv; stale DMI answers; bridge unbounded | confirmed, fixed: core-domain copies, tagged handshakes, timeouts, bridge bound, RV_CLK_OK | §3.3, §3.5, §7.1 |
| Reset sequencing | confirmed, fixed | §7.3 |
| s0-s15 outside the race definition | confirmed, fixed: one bank (D8), covered by the rules of §5.5 | §4.7, §5.5 |
| FENCE snapshot in reserved words | confirmed, fixed: the dispatcher takes the snapshot | §4.4 |
| Error state at dispatch; first_error order | confirmed, fixed: push index | §4.3, §4.5 |
| Stale tickets after QCLEAR; crt0 | confirmed, fixed | §4.3, §8.3 |
| Staging not reentrant | confirmed, documented; a second staging bank rejected | §3.2 |
| Launch protocol deadlock; launch rate | confirmed, fixed: the LQ / LCQ protocol | §8.3, §8.4 |
| Validity defined by a test | confirmed, fixed | §9.5 |
| Round-robin deadlock argument; DMA head-of-line; DDR penalties | confirmed, argument written; the DMA burst reservation closes the head-of-line case | §5.3, §5.4 |
| Cost table optimistic | confirmed, fixed | §12 |
| QHOLD level against pulses | confirmed, fixed: three hold sources, each with one owner | §3.2 |
| 300 MHz margin | confirmed, fixed: timing gate, dispatch register, placement | §7.4 |
| CDC crossing list incomplete; interrupt clear semantics | confirmed, fixed | §7.1, §3.7 |
| Ibex grant path; Fmax optimistic | plausible (*confirm*), fixed: registered grants, planning F_RV 166.7 MHz | §2.3, §7.4 |
| True dual-port ITCM | confirmed, fixed | §3.1 |
| ASIC storage; FIFO depth; host-less boot | confirmed, fixed | §3.4, §4.2, §12.2 |
| Clock and reset on the U200 | confirmed (*confirm* the `v++` options), fixed | §7.2-§7.4 |

---

## 15. Open items and answered questions

| Question | Answer |
|---|---|
| s-register bank (D8) | one bank in the VU (CC §1, EX §5.1) |
| Execution of layer programs (D10) | the `graph` kernel executes op lists and issues tile ops directly (SW §7) |
| RISC-V clock on the U200 (D12) | planning value 166.7 MHz; the T1 gate decides 250 / 166.7 / 125 MHz (CC §3.3, §14.3) |
| Timing gate (D13) | CC-23; gates T1-T4 (CC §14.3) |
| `ap_clk_2` generation | CC §11.1: a `v++` frequency request first, `tsr_clk_div` in `rtl/tech/xilinx` as the fallback; confirmed at the first smoke link (planned) |
| Stuck units | no unit reset; hang triage and the recovery ladder end in a card reset (EX §8.6, DB §7) |
| Launch path: a DRAM ring with a doorbell; an interrupt to the host; `ap_ctrl_hs` | the LQ / LCQ rings with posted writes (EX §4); `ap_ctrl_none` with polling (EX-6); `host_irq_o` is a boundary output for later targets (EX §9.3) |
| XRT window during `xrt::run` | `ap_ctrl_none`, `xrt::ip` register access only (EX §9) |
| DDR penalty of interleaved LD / ST | open: M0 measured read, write and read + write streams ([`m0-bw.md`](m0-bw.md)), not short writes between weight bursts; a W6 measurement |
| F (soft-float) | none in firmware (SW-8) |
| Zba / Zbb / Zbs | not taken; an SDK-measured option (CC §3.2) |
| TCM sizes | 128 KiB ITCM + 64 KiB DTCM (CC-3); the production image uses ITCM 53 %, DTCM 76 % (§8.1) |
| Queue depths and `ISSUE_FIFO_DEPTH` | 8 / 4 / 4 / 16 / 0 / 4 and 32 (CC §7.4); the DMA depth may grow (§12.4) |
| SPM arbitration, DMA head-of-line loss | even / odd banks, round robin per bank, DMA burst reservation (CC §10, CC-12, CC-13) |
| Debug transport speed (`remote_bitbang`); SBA as a second master | RSP server driving DMI directly, SBA connected (DB §3, §4; DB-1, CC-16); DMI cost over PCIe is an *estimate* (DB §3.10) until the card |
| Boot ROM for host-less targets | open (`HAS_BOOTROM`) |
| SPM bridge line buffer | answered: the line-buffered alias (CC §10.5) |
| Licenses (Ibex Apache-2.0, riscv-dbg SHL-0.51 with the planned RTL license) | open; `docs/spec.md` §20 |
| M7 attention as a C++ kernel | answered: FLASH_ATTN is a graph-executor kind on the simulator (AT §6.6, SW §7.5) |
| Measurements on the card: PCIe posted-write and read latency | open; they set the image-load time and the launch cost; W6 (EX §4.11) |

Remaining open items of the control core, all tied to W4-W6: the T1-T4 timing gates and the Ibex RVFI bench that
confirms the maxperf load / store and CSR-flush timing (P5.2); the vendoring of Ibex and riscv-dbg at pinned
commits and the *confirm* items of CC §3 and §19; the RMS_NORM and ROPE records of Gemma 4 shapes
against the 350-cycle requirement and the M7 issue loss of §12.4; the PCIe latencies on the card.

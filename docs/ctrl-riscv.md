# Tesseract control core (RISC-V) — design plan

Status: **plan, not implemented** (2026-10-24, revision 3: the user decided the §15 questions,
D8–D13 below; revision 2 applied the review findings, §14). Nothing described here exists in the
code yet, except the facts in §1, which were checked against the code on that date (file and line
given). The contract is the spec section "제어 코어" in `docs/spec.md` (applied 2026-10-24 with the
edits of the other spec sections it needs); this document holds the rationale, the register
detail, the cost and the verification plan. Source-of-truth order: `docs/spec.md` > ISA simulator
(`sw/sim`) > this document. Names of new modules, registers, files and options are proposals;
phases CR1 and CR2 fix them. Every number marked *estimate* is not measured; everything about
Ibex, riscv-dbg, OpenOCD and Vitis that is marked *confirm* is recalled from their documentation
and must be checked when the dependency is pinned. The decided questions and the remaining open
items are in §15.

**User decisions (2026-10-24, fixed; this plan designs around them)**

| # | Decision |
|---|---|
| D1 | Embed a RISC-V control processor in the Tesseract core. Simple: in-order, no branch prediction, no superscalar, no out-of-order, no cache. |
| D2 | Code, data and stack live in local SRAM (TCMs): instruction fetch never uses DRAM bandwidth; code regions are small. |
| D3 | The RISC-V issues the existing 256-bit tile instructions **asynchronously**: issue and continue (fork); several tile ops in flight on different units; software joins later. |
| D4 | The SPM is mapped into the RISC-V address space (same memory space, a bus connection). |
| D5 | The RISC-V has its own clock domain; its clock depends on the core implementation. |
| D6 | Kernels in C++ (stock RISC-V GCC / Clang, no custom compiler); tile and SIMD operations as a C++ header library; a JIT (torch.compile, tinygrad, later JAX PJRT) calls precompiled kernels. |
| D7 | The llama.cpp path (host command buffers in DDR fetched by `tsr_cmdproc`, sequential semantics) stays. |

**User decisions on the §15 questions (2026-10-24, fixed)**

| # | Decision |
|---|---|
| D8 | Scalar registers: the issue path has its own s0–s15 bank (≈ 512 FF); command-buffer runs (host START, CALL) keep theirs. No race on s registers between the two paths can occur (§4.7). |
| D9 | Host ownership: the transport alternates llama.cpp execs and RISC-V launches — an exec starts only when no launch is outstanding, a launch only when BUSY = 0. No SPM split, no hardware change (§6, §8.4). |
| D10 | M8 / M9 decode steps are C++ kernels that issue tile ops directly; position-dependent addresses are computed in C++. CALL is used only for position-independent command-buffer programs. No register-relative lowering mode in the backend (§6, §11.1). |
| D11 | One shared checker: ops pushed during a command-buffer run wait for it (acceptable given D9) (§5.2). |
| D12 | U200 RISC-V clock: planning value 166.7 MHz; 250 MHz only if the Ibex out-of-context run closes there; 125 MHz fallback (§7.4). |
| D13 | CR2 timing gate: the control-core RTL is frozen only after the k1, k4t1 and k4 out-of-context runs and the k4 v++ link meet 300 MHz with `HAS_CTRL = 1`. The k4 link waits for the host RAM upgrade the user will do when the card goes in (§7.4). |

**User decisions (2026-10-30, the v2 restructure; they supersede D7–D11 and D13)**

| # | Decision |
|---|---|
| D14 | The host command-buffer execution path is removed: no host START, no CMD_* / ARG*, no `tsr_cmdproc` sequencer, no `tsr_fetch`, no R0–R15 / LI / ADDI / END. The host loads the firmware image and LAUNCHes kernels; every tile op is issued by the RISC-V. This reverses D7 and makes D9 (alternation), D10 (CALL is dropped) and D11 (the checker serves only the dispatcher) obsolete; D8 shrinks to one s0–s15 bank. |
| D15 | No host debug access to the issue port. Bring-up, the card selftest and trace replay run as RISC-V kernels. |
| D16 | v1 is deleted now, not after v2 passes on the card. Tag `v1-m4` (M4 passed) keeps it; a copy of the v1 binaries (`build/v1`, not in git) runs models on the card meanwhile. |
| D17 | Device KV cache in q8_0; attention arrives as llama.cpp `FLASH_ATTN_EXT` (a quantized V cache requires Flash Attention). The klcheck / numcheck baselines move to FA on. |
| D18 | The first v2 has full debug: RISC-V Debug Spec module, halt / resume / step, breakpoints and GDB source-level debugging over PCIe (a GDB RSP server on the DMI window), plus tile-level debug. |

Milestones after the restructure: M6 = core v2 hardware (control core, issue queues, vector unit,
attention and KV, second SPM read path, debug), M7 = RISC-V kernel library and graph execution
(one launch per token), M8 = device MoE and PyTorch on the kernels (M5 merged here), M9 = resident
decode loop (optional). Technical defaults chosen with D14–D18 (proposals until the M6 design
fixes them): even/odd SPM banks for the second read path; `ap_ctrl_none` with polling instead of
`ap_ctrl_hs`; a host-readable log FIFO in the core domain; D13 restated as "with the control core
the k4 link stays at or above the current 276 MHz"; first models Qwen2.5 and Gemma 4 dense (GELU
in the VU). The sections of this document that describe the command-buffer path are rewritten in
the M6 design.

This replaces the M8/M9 scalar ISA growth (MULI, RLD, LOOP, WAITM, JUMP); the M8 and M9 pass
criteria (M8: the host writes only 2 arguments per token; M9: the host only polls a ring buffer)
stay. Everything below that is not D1–D13 was a proposal; what the spec section "제어 코어"
adopted is now the contract, and the names of new modules, files and options stay proposals until
CR1 / CR2.

| Question | Answer (proposal unless marked D*n*) |
|---|---|
| Which core? | lowRISC **Ibex**, pinned and unmodified, a configuration close to its "small" one: RV32IMC + Zicsr + Zifencei, no F (§2). |
| How does it issue tile ops? | A memory-mapped issue port (8 staging words + push), per-unit in-order queues, 32-bit tickets, up to 2 hardware dependencies per op, SYNC join points and FENCE barriers. Software checks the credit before a push; every entry, run or not, completes only after its dependencies (§4). |
| How does it see the SPM? | A 16 MiB window at 0x4000_0000 (4 MiB mapped), uncached, byte/half/word, through SPM client 3 shared with the debug window; with `HAS_CTRL = 1` `tsr_spm` becomes round robin (§3.5, §5.3). |
| DRAM? | No direct window. Data moves by tile DMA (§3.6). |
| What changes in the core? | A checker split out of `tsr_cmdproc`, a dispatcher with unit queues, SPM round robin with a 4th client, PERF gating by "the host run is executing", the CTL CSR block, the CDC library. With `HAS_CTRL = 0` nothing changes (§5). |
| And the command-buffer path? | Unchanged. The RISC-V can also run a position-independent command buffer as one queued op (`CALL`, unit CMD; D10); command-buffer runs are exclusive with datapath queue ops, with a drain rule (a host START before a CALL), so they behave exactly as today; the transport alternates host execs and launches (D9) (§6). |
| Clocks on the U200? | A second kernel clock `ap_clk_2`; planning value 166.7 MHz (500 / 3), 250 MHz only if Ibex closes there, 125 MHz fallback (D12); SLR1. Keeping the core's 300 MHz is the CR2 gate (D13, §7). |
| Software? | `sw/ctrl` SDK (freestanding C++20 headers, crt0, linker script, ELF image), kernel ABI, host API in `tsr_transport`, mailbox launch protocol, JIT sketch, a pipelined matvec example (§8). |
| Verification? | RV32 interpreter in the simulator (Spike as its oracle), queue model with random schedules, a vector-clock race detector, RTL co-simulation in lockstep, a trace extension, and a definition of bit-exactness for concurrent programs (§9). |
| Debug? | RISC-V Debug Spec 0.13.2 (riscv-dbg debug module), DMI through the core CSR window, GDB via OpenOCD; per-ticket completion records for the profiler (§10). |
| Cost? | ≈ 12–20 K LUT, ≈ 32–34 RAMB36 on the U200 (estimate, §12). |

---

## 1. What exists today

Checked 2026-10-24.

- **One instruction in flight.** `tsr_cmdproc` executes one instruction at a time and offers the
  next command only after the previous unit's `done` (`tsr_cmdproc.sv:6-7`, states `S_ISSUE` /
  `S_WAIT` at `:451-460`; `docs/rtl-m2.md` §3.0). The dispatch protocol is
  `cmd_valid / cmd_ready / cmd / done`; units report no errors (every architectural check is in
  cmdproc before dispatch); there is no flush or cancel. Check cost per instruction: NOP / LI 4
  cycles, LD / ST / QACT 10 + unit (`tsr_cmdproc.sv:45-48`), MVM checks ≈ 46 (serial divider,
  `:49-53`). The dispatch registers are `dma_cmd_q`, `qact_cmd_q`, `mvm_cmd_q` with valid flops
  (`:176-178`, `:675`).
- **The datapath is already concurrent in structure.** Every unit has its own SPM client port
  (0 DMA, 1 QACT, 2 MVM: `tsr_pkg.sv:83-86`) and its own memory source per channel (0 FETCH,
  1 DMA, 2 WS; source id 3 free: `tsr_pkg.sv:73-78`). Only cmdproc serialises them.
- **SPM**: one read and one write port, fixed priority lowest client index first, registered
  grant (`tsr_spm.sv:20-30`, `:123-131`). A higher client that streams continuously starves a
  lower one without bound. The DMA's LD flow control relies on being client 0: "a DMA SPM write
  stalls at most one cycle … rd_rsp_ready stays 1 for the whole LD" (`tsr_dma.sv:35-45`).
  `docs/prof-dbg.md` §3.5 already plans a fourth, lowest-priority client for the debug SPM
  window (`TSR_SPM_NCL` 3 → 4).
- **Memory arbiter**: round robin per burst among FETCH / DMA / WS with one outstanding budget
  per channel, response routing head-of-line; sources may drop their response ready "only for
  reasons local to that source" (rule R4), which is the no-deadlock argument
  (`tsr_mem_arb.sv:9-23`, `:38-46`).
- **PERF**: only `PERF_CYCLES` is gated by BUSY; `MVM_CYCLES`, `CH_RD_BEATS[c]` and `CH_WAIT[c]`
  count their events whenever they happen (`tsr_csr.sv:311-320`). Today nothing happens
  outside a run, which is why the spec's "실행 사이 PERF 불변" holds.
- **CSR space**: 12-bit offsets (`tsr_pkg.sv:92`). Used or planned: 0x000–0x07C, 0x080
  (`PERF_INFO`), 0x100–0x2FC (PERF and its reserved extension indices), 0x800–0x9FC (`DBG_*`),
  0xA00–0xA3C (`TRC_*`). Free: **0x300–0x7FC and 0xA40–0xFFC**.
- **Clock and reset**: one clock `clk`, synchronous `rst_n` (`docs/rtl-m2.md` §1). The spec's
  technology-wrapper layer already lists "CDC 동기화기"; none exists yet.
- **U200**: the kernel tops have one clock `ap_clk` at 300 MHz (`docs/platform-u200.md` §1); the
  platform offers DATA 300 MHz (scalable) and KERNEL 500 MHz (§8.2 there; `docs/rtl-m3.md`
  §14.5). Adapter offset 0x01C is reserved (`docs/platform-u200.md` §2). The core CSR window is
  `s_axi_control` 0x1000 + offset, 10 kernel cycles per window access; PCIe latency is not
  measured. The XRT shell's AXI firewall cuts a control slave that does not answer, and recovery
  needs a card reset (`docs/prof-dbg.md` §3.2).
- **Timing margins** (`docs/rtl-m3.md` §14, `docs/platform-u200.md` §8.2): out of context k4
  +0.015 ns (worst: SPM write-port bank select → URAM cascade enable), k4t1 +0.108 ns (worst:
  cmdproc state → DMA / MVM command register clock enables, 0 levels; then SPM write grant → SPM
  input register); the k1 link closed at +0.011 ns; the k4 link has not run and is projected at
  −0.1 to −0.2 ns (*estimate* there).
- **SLR1 budget** (`docs/rtl-m3.md` §14.5): floorplan C' puts 25,068 LUT, 8 RAMB36, 128 of 160
  URAM and 32 DSP of SLR1's dynamic region (205,440 LUT / 420 RAMB36 / 160 URAM / 1,320 DSP)
  into the core; DDR[2]'s controller adds about 30.5 K LUT and 61 RAMB36 there (estimate from
  the M0 link, `docs/platform-u200.md` §8.2). k1 is entirely in SLR1: 33,276 LUT, 40 RAMB36, 128
  URAM, 182 DSP (routed).
- **Host control cost** (spec 미결정 사항, vkernel count 2026-10-24): 21 register writes and 6
  reads per exec, 145 execs per Qwen2.5-0.5B decode token: about 3,000 posted writes and 870
  reads per token. Their time on the card is not measured.
- **Reference fp helpers**: `tsr_fadd` etc. perform one native IEEE binary32 operation with
  explicit DAZ / FTZ / NaN bit tests and never touch the FP environment (`tsr_ref.c:9-16`,
  `:99-104`). They compile unchanged for a soft-float RV32 target.
- **Software**: `tsr_transport.h` has open / write / read / exec / `csr_read` only (no CSR
  write; PD1 proposes it); `sw/runner/` is empty; the simulator is strictly sequential with no
  CAPS feature (`tsr_sim.h:1-29`). Today's lowering emits R0-relative (absolute) addresses.

---

## 2. Core choice and ISA

### 2.1 What the workload needs (estimates)

- **Issue rate.** A device-resident decode token is about 1 K tile ops on Qwen2.5-0.5B (24
  layers × ≈ 40) and 3–5 K on 32B (64 layers × 40–80 with attention tiles). At 25–40 RISC-V
  cycles per op (§4.8) that is ≈ 0.04 M cycles per 0.5B token and ≈ 0.2 M per 32B token. At
  0.5B's 7.6 ms per token this is ≈ 5 M cycles/s; 32B needs far less.
- **Decision latency.** The latency-critical points are device decisions: TOPK result → expert
  ids → expert MVMs (MoE), sampling → next token. With ≈ 1–2 µs per decision and 48 MoE layers,
  under 0.1 ms per token against ≈ 26 ms of weight streaming for Qwen3-30B-A3B (≈ 2 GB per token
  at 76.8 GB/s).
- So ≥ 50 MIPS sustained is plenty (≥ 100 MHz at CPI 1.2–1.6 leaves margin). Fmax and IPC do not
  decide the choice. What does: verification maturity (the core becomes part of every
  device-resident result), standard debug (C++ kernels need GDB), vendor independence and an ASIC
  path (spec 시스템 구조), Verilator (every RTL test runs there), and an integration that leaves
  the core unmodified.

### 2.2 Candidates

| | Ibex | PicoRV32 | VexRiscv | SERV | In-house RV32IM |
|---|---|---|---|---|---|
| Source | SystemVerilog (lowRISC) | Verilog-2005, one file | SpinalHDL (Scala) generator → Verilog | Verilog | SystemVerilog (ours) |
| License | Apache-2.0 | ISC | MIT | ISC | project |
| Pipeline | 2-stage in-order (IF, ID/EX); optional WB stage, branch predictor and ICache, all off by default | multi-cycle, not pipelined; CPI ≈ 4 (README) | 2–5 stages, plugins (prediction, caches optional) | bit-serial, ≈ 32–64 cycles / instruction | 3–5 stages, to design |
| ISA options | RV32I/E, M (slow / fast / single-cycle), C, B, Zicsr, Zifencei, M + U modes, PMP | RV32I/E/M/C; own IRQ scheme (custom instructions), no standard trap CSRs | RV32IM(A)(F)(C), M/S/U modes | RV32I (+M), Zicsr | as built |
| Verification | UVM + riscv-dv random programs with Spike co-simulation, formal work, OpenTitan sign-off; README grades configurations ("small" fully verified, *confirm*) | riscv-formal | regressions with riscv-tests / arch tests, partial riscv-formal; depends on the plugin set | riscv-arch-test (riscof) | none yet: riscv-formal, arch tests, riscv-dv and co-sim to build (months, *estimate*) |
| Debug | RISC-V Debug Spec 0.13.2 with an external debug module (pulp riscv-dbg, as in OpenTitan) | none | own DebugPlugin (needs a SpinalHDL OpenOCD fork); recent versions add a debug-spec module | none standard | must implement debug mode (dcsr, dpc, dret) and still reuse a DM |
| ASIC history | OpenTitan Earl Grey silicon and others | several shuttles | several SoCs and shuttles | several shuttles | none |
| Verilator | yes (Ibex simple system) | yes | yes (generated) | yes | yes |
| FPGA area (*estimate*) | 2.5–4 K LUT with the FPGA register file (no published figure for this configuration; CR1 measures) | 0.75–2 K LUT (7-series README) | 1–2 K LUT (README configurations) | 0.2–0.4 K LUT | 1.5–3 K LUT |
| Fmax UltraScale+ -2 (*estimate*) | 125–250 MHz (no published UltraScale+ figure; the path register file → ALU adder → data address → TCM / slave decode limits it; planning value 166.7 MHz; CR1 measures) | 250–450 MHz on 7-series (README) | 200–350 MHz on Artix-7 (README), more on UltraScale+ | high | 300 MHz possible |
| Throughput (*estimate*) | CPI ≈ 1.2–1.6 | Fmax / 4 | CPI ≈ 1.2–1.5 | < 10 MIPS | CPI ≈ 1.2–1.5 |
| Concern | lower Fmax (irrelevant here, own clock); third-party style (async reset, `prim_*`) | no standard traps or debug | the source of truth is Scala; regeneration needs SpinalHDL; generated Verilog is not reviewable source | too slow for decision latency | the verification and debug work is the cost, not the pipeline |

### 2.3 Decision (proposal): Ibex

- **Configuration** (an Ibex top parameter set, close to "small"): RV32E 0, RV32M = fast
  (multi-cycle multiplier, MUL ≈ 3–4 cycles, DIV ≈ 37 cycles, *confirm*), RV32B none,
  BranchTargetALU 0, WritebackStage 0, ICache 0, BranchPredictor 0, PMPEnable 0, SecureIbex 0,
  MHPMCounterNum 0 (raise only if profiling needs it), DbgTriggerEn 1 with 2 hardware
  breakpoints. Register file: the FPGA (LUTRAM) variant on Xilinx, flops or latches on an ASIC,
  chosen by the technology layer. Deviations from the verified configuration are exactly the
  register-file variant and the triggers.
- **Why**: SystemVerilog with generic `prim_*` implementations (vendor independence; the only
  FPGA-specific piece is the clock gate, mapped through `rtl/tech`); the most mature open
  verification and silicon history; the standard debug spec (GDB / OpenOCD work as they are);
  Verilator support; in-order with no predictor and no cache by default, so D1 needs no
  modification; simple `req / gnt / rvalid` instruction and data ports, so TCMs and the control
  window attach without a bus fabric; the issue port is MMIO, so the core stays unmodified.
- **Integration rules**: Ibex and riscv-dbg are vendored at pinned commits in
  `third_party/ibex` and `third_party/riscv-dbg` (with the few pulp `common_cells` files the DM
  needs) and never edited (a patch series if ever required, like `sw/xrt-driver`). They keep
  their own style — asynchronous active-low resets, `prim_*` modules — and their own lint
  waivers (Ibex ships `lint/verilator_waiver.vlt`); spec rule 6 (Verilator lint clean, no vendor
  primitives outside `rtl/tech`) applies to Tesseract code, with one waiver file per third-party
  tree. `prim_clock_gating` binds to `rtl/tech/<target>/tsr_clk_gate.sv`: a pass-through on the
  FPGA (WFI then only stops fetch, functionally the same), an ICG cell on an ASIC. The tested
  Verilator version is 5.046; Ibex's own CI uses an older one, so the vendoring step checks that
  the pinned Ibex builds with it.
- **Bus rule** (every data-port and instruction-port slave): `gnt` comes from a register
  (`gnt = ~outstanding_q`), never from the address decode. Ibex's `data_addr_o` comes
  combinationally from the ALU adder and `data_gnt_i` feeds its stall logic in the same cycle
  (*confirm* on the pinned commit), so an address-dependent grant would put register file → adder →
  decode → grant → PC logic on one path, the likely Fmax limiter. Every wait (credit, SPM bridge,
  core window, debug-module arbitration) is a late `rvalid` instead. With WritebackStage 0 Ibex
  waits for `rvalid` anyway, so normal accesses lose no cycle; the second half of a misaligned
  access (Ibex issues it while the first is outstanding) waits one cycle more, because each slave
  holds one access at a time.
- **Measure first** (CR1): a ten-minute out-of-context synthesis of exactly this configuration
  with the TCMs on `xcu200-fsgd2104-2-e` replaces the area and Fmax estimates above before CR2
  starts.
- **Runner-up**: VexRiscv (smaller, faster) if Ibex's integration or F_RV turns out to be a
  problem. Not PicoRV32 (no standard traps or debug), not SERV (decision latency), not in-house
  (verification cost with no gain for this workload).

### 2.4 ISA subset and the F question

**RV32IMC + Zicsr + Zifencei**, machine mode only (Ibex has U mode; the runtime never uses it),
no PMP, no A, no F / D, no B in v1.

- **M**: address arithmetic (expert × `EXP_STRIDE`, row splits ⌈N/C⌉, KV offsets).
- **C**: about 25–30 % smaller code (typical for RV32, *estimate*), so a smaller ITCM; Ibex
  handles it at full speed.
- **Zicsr**: traps, interrupts, `mcycle` / `minstret`, WFI, debug CSRs. **Zifencei**: `fence.i`
  after code is written into ITCM.
- **No A**: one hart; host handshakes use sequence / acknowledge registers (§3.3).
- **B** (Zba / Zbb / Zbs): Ibex has options; an open item decided by SDK code size and cycles.
- **No F in v1**, although f32 scalars are needed for scales and softmax bookkeeping:
  1. *Availability.* Ibex has no F. F-capable verified cores (for example OpenHW's CV32E40P with
     its FPU) are larger (an FPU is ≈ 5–10 K LUT, *estimate*) and bring another verification
     story.
  2. *Bit-exactness does not need it.* The spec's fp rules are implemented by `tsr_ref` as one
     IEEE operation plus explicit bit tests (§1). IEEE basic operations are deterministic, so the
     same C compiled for RV32 with soft-float (libgcc) gives the same bits as on the host, and the
     simulator executes the same RV binary anyway. Rule: device-side scalar math that must follow
     the spec's fp rules calls `tsr_fadd` / `tsr_fmul` / `tsr_fdiv` / `tsr_fsqrt` / `tsr_frecip`,
     never bare `float` operators or libm (libm's `expf` is not the VU's EXP approximation).
     Checked by building `sw/sim/tests/test_ref_fp.c` for RV32 and running it on the interpreter
     and on Spike (CR1).
  3. *Volume.* Per-element math stays on the vector unit. The per-head bookkeeping of an online
     softmax (running max m, sum l, rescale exp(m_old − m_new)) is vectorised across heads: m and
     l are SPM vectors of `n_head` elements updated by a few short VU ops per KV tile (VOP.MAX,
     SUB, EXP, MUL, ADD), not scalar code on the RISC-V. What remains scalar is rare: sampling
     arithmetic over TOPK candidates after the VU has done their softmax (≈ 40–100 operations per
     token), and constants at load time. At ≈ 50–150 cycles per soft-float operation (*estimate*)
     that is ≤ 15 K cycles ≈ 90 µs per token at 166.7 MHz.
  4. *Fallback* if a profile shows soft-float above 1 % of token time: move that math to SOP / VU
     tile ops, or choose a core with Zfinx / F (open item).

---

## 3. Address map

### 3.1 RISC-V view

32-bit byte addresses, no MMU, PMP off. Every unmapped access is an access fault (instruction,
load or store); every region's size beyond its parameter is unmapped.

| Range | Region | Fetch | Load / store | Notes |
|---|---|---|---|---|
| 0x0000_0000–0x000F_FFFF | ITCM, `ITCM_BYTES` (U200 64 KiB) | yes | yes, byte / half / word (overlays, software breakpoints; then `fence.i`); a data-port read takes the fetch port for one cycle | vector table at `CTL_BOOT` (256-byte aligned, default 0); first fetch at `CTL_BOOT` + 0x80 (Ibex: `mtvec` from `boot_addr_i[31:8]`, *confirm*) |
| 0x0010_0000–0x001F_FFFF | DTCM, `DTCM_BYTES` (U200 64 KiB) | no | yes | .rodata, .data, .bss, stack |
| 0x1000_0000–0x1000_0FFF | debug module memory | yes | yes, 32-bit | halt address +0x800, exception address +0x808 (riscv-dbg layout, *confirm*); not restricted to debug mode (§10.1) |
| 0x2000_0000–0x2000_0FFF | control window (§3.2), clk_rv registers | no | 32-bit only | issue port, tickets, queue state, mailbox, timer |
| 0x2000_1000–0x2000_1FFF | core window | no | 32-bit reads only (a store is an access fault) | ID, VERSION, CFG0–2, CAPS, CTL_ID, CTL_CFG (constants, held in clk_rv); s0–s15 of the issue-path bank (M6, §4.7) through a handshake with the SPM bridge's bound |
| 0x4000_0000–0x40FF_FFFF | SPM window (§3.5), `SPM_BYTES` mapped (U200 4 MiB) | no | byte / half / word | byte address = 0x4000_0000 + 64 × line + offset |

- The instruction port decodes ITCM and the debug module; the data port decodes all regions.
- **ITCM** is built from `tsr_sram_1r1w`, like the DTCM (no true dual-port memory type in
  `rtl/tech`, and no 2RW macro on an ASIC): the read port serves instruction fetch, the write port
  takes the data port's stores and the host loader (never both at once: the loader works only
  while RV_RUN = 0). The rare data-port reads of ITCM (debugger memory reads, overlay
  verification) register the request and take the read port for one cycle with `instr_gnt` = 0.
  `.rodata` is linked into DTCM, so normal code never reads ITCM as data.
- **Debug module**: it has one slave port, which Ibex's instruction port (debug ROM, program
  buffer) and data port (DM flags) can hit in the same cycle; a 2:1 arbiter, data first, stalls
  the fetch through `instr_gnt`.

### 3.2 Control window (RISC-V side, offsets from 0x2000_0000)

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x000–0x01C | `ISSUE_W0`–`W7` | RW | staged instruction words; kept after a push |
| 0x020 | `ISSUE_DEP0` | RW | dependency ticket 0; unit field 15 = none. Reset to none by every push |
| 0x024 | `ISSUE_DEP1` | RW | dependency ticket 1, same rules |
| 0x028 | `ISSUE_PUSH` | W | push the staged op. wdata [15:8] WMASK: words w_i with bit i clear are sent as 0 (w0 always sent), so an op needs to write only the words it uses; bit0 REC: write a completion record. Without a credit the store completes late (§4.1) |
| 0x02C | `ISSUE_TICKET` | R | ticket of the last push or QCLEAR |
| 0x030 | `ISSUE_CREDIT` | R | 4 bits per unit (unit u at [4u+3:4u]): min(free slots of its queue, free issue-FIFO entries, 15) |
| 0x034 | `ISSUE_QMAP` | R | unit of each opcode class, 4 bits each: [3:0] LD/ST, [7:4] QACT, [11:8] MVM, [15:12] DQROW, [19:16] VOP/RED/ROPE/TOPK/SOP, [23:20] CALL, [27:24] NOP/FENCE; 15 = class absent |
| 0x038 | `ISSUE_STATUS` | R/W1C | bit0 PUSH_TIMEOUT (a push waited `ISSUE_WAIT_MAX` and failed); bit1 ERR_IRQ (sticky: set when a unit enters error state; drives `irq_fast[1]`, §3.7) |
| 0x040 + 4u | `DONE_SEQ[u]` | R | ops completed on unit u, 28 bits; a synchronised copy that may lag by a few cycles, never leads |
| 0x080 + 4u | `ISSUED_SEQ[u]` | R | sequence the next push to unit u gets |
| 0x0C0 + 16u | `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0` of unit u | R | first error since the last clear (§4.5); `ERR_CODE` 0 = none. Arrive with the error state, before `DONE_SEQ[u]` passes `ERR_SEQ` |
| 0x140 | `QCTRL` | W | control entries, sent through the issue FIFO in push order: bit0 QABORT, bit1 QHOLD set, bit2 QHOLD clear (the RISC-V's hold source), bit3 QCLEAR (a SYNC entry with a ticket, read from `ISSUE_TICKET`; §4.5). One bit per write |
| 0x144 | `QSTATUS` | R | bit0 QIDLE (`ISSUED_SEQ[u]` = `DONE_SEQ[u]` for every u and the issue FIFO empty: lags safely), bit1 hold in effect, [4:2] hold sources (RISC-V, host, debug halt), [13:8] error state per unit (`ERR_CODE` ≠ 0, same crossing as the records), bit16 a command-buffer run holds the queues |
| 0x148 | `WAIT_TICKET` | RW | `irq_fast[0]` is pending while this ticket is done |
| 0x14C | `WAIT_CREDIT` | RW | (optional) unit u: `irq_fast[5]` is pending while `ISSUE_CREDIT` of u ≠ 0 |
| 0x160 | `REC_COUNT` | R | completion records waiting |
| 0x164–0x170 | `REC_DATA0`–`3` | R | head record, 128 bits (§10.2) |
| 0x174 | `REC_POP` | W | drop the head record |
| 0x178 | `TRACE_MARK` | W | a marker entry through the issue FIFO; its record (unit 14) is stamped with the core time at dispatch |
| 0x180 + 4u | `ERR_PIDX[u]` | R | global push index of unit u's first error (§4.3, §4.5) |
| 0x200–0x21C | `H2R_MSG0`–`7` | R | host → RISC-V message (clk_rv copy, delivered by the doorbell transfer) |
| 0x220 | `H2R_SEQ` | R | sequence the host last posted (updated after the words); `irq_fast[2]` pending while `H2R_SEQ` ≠ `H2R_ACK` |
| 0x224 | `H2R_ACK` | RW | the RISC-V writes the sequence it consumed |
| 0x240–0x25C | `R2H_MSG0`–`7` | RW | RISC-V → host message |
| 0x260 | `R2H_SEQ` | RW | written after the message words: copies them into the transfer register and posts the message; the store completes once copied |
| 0x264 | `R2H_ACK` | R | sequence the host last consumed; `irq_fast[3]` pending while `R2H_ACK` = `R2H_SEQ` (slot free) |
| 0x300 / 0x304 | `MTIME_LO` / `HI` | R | core time: core clock cycles since reset (gray-synchronised) |
| 0x308 / 0x30C | `MTIMECMP_LO` / `HI` | RW | `irq_timer` while MTIME ≥ MTIMECMP |

- Reads in the control window have no side effects (`REC_POP` is a separate write), so a
  debugger's memory view cannot pop or push anything. The offsets are generated from one table for
  the RTL, the simulator and the SDK (like `tsr_pkg.sv` ↔ `tsr_isa.h`).
- **Not reentrant.** There is one staging set (`ISSUE_W*`, `ISSUE_DEP*`, `ISSUE_TICKET`). An
  interrupt handler, or a debugger running code through the program buffer, that issues between
  the interrupted code's stores corrupts that push and its ticket. Rule: handlers never issue; GDB
  inferior calls that issue are unsupported. A second staging bank for handlers is not proposed (no
  use case worth its cost).
- **Hold sources.** The queue heads are held while any of three sources is set: the RISC-V's
  (`QCTRL` bits 1–2, applied in push order), the host's (`CTL_CTRL.HOST_HOLD`) and the debug halt's
  (`HOLD_ON_HALT` while halted). Each agent changes only its own source, so neither can drop the
  other's hold.

### 3.3 Host view: core CSRs

All through the existing CSR port (U200: `s_axi_control` 0x1000 + offset). With `HAS_CTRL = 0`
every offset below reads 0 and ignores writes.

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x300–0x4FC | `CTL_PERF_*` | R | counter slots 0–63, 64 bits each (LO / HI as `PERF_*`, own snapshot slot). Built only where defined (21 + 2 × `NUM_CH`, 29 on k4); the other slots read 0. Index 0 counts cycles while RV_RUN = 1; the others count events outside an executing host run (a START waiting for the drain is outside, §6): 1–6 unit busy cycles (DMA, QACT, MVM, VU: the unit holds an op; CMD: a CALL runs; SYNC: an entry waits at the head); 7–12 head-of-queue cycles waiting on dependencies; 13 cycles queue heads are held by a command-buffer run or its drain; 14 SPM bridge accesses; 15 SPM bridge wait cycles; 16 completion records dropped; 17 push stall cycles; 18 CALL runs; 19 CALL instructions retired; 20 cycles a host START waited for the drain; 32 + c read beats of channel c; 48 + c wait cycles of channel c. 14, 15 and 17 count clk_rv cycles or events (§7.1) |
| 0xB00 | `CTL_ID` | R | 0: no control core. [7:0] version (1), [11:8] unit ids in use (6), bit12 debug module, bit13 completion records, bit14 SPM bridge |
| 0xB04 | `CTL_CFG` | R | [15:0] ITCM KiB, [31:16] DTCM KiB |
| 0xB08 | `CTL_CTRL` | RW | bit0 RV_RUN (1: release the RISC-V from reset), bit1 HOST_HOLD (the host's hold source), bit2 HOLD_ON_HALT (a debug halt sets the halt hold source). Write-1 pulses, read 0: bit8 QABORT, bit9 QCLEAR |
| 0xB0C | `CTL_STATUS` | R / W1C | levels: bit0 RV_RUNNING, bit1 RV_HALTED (debug), bit2 RV_SLEEP (WFI), bit3 QIDLE (no entry queued or running, issue FIFO empty; exact while the RISC-V is not pushing), bit4 QERR (a unit in error state), bit5 CALL_BUSY, bit6 QQUIESCENT (§4.6), bit7 RV_CLK_OK (clk_rv runs and its domain is out of reset); bit16 LAST_RUN_CALL (R1–R15, `PC` and the `DBG_*` run state belong to a CALL), bit17 START_WAIT (a host START waits for the drain), [20:18] hold sources (host, RISC-V, debug halt), bit21 RV_RESET_DONE (the RISC-V reset sequence has completed, §7.3). Sticky, W1C: bit8 LOAD_REFUSED, bit9 DMI_TIMEOUT, bit10 QCLEAR_REFUSED, bit11 PUSH_TIMEOUT, bit12 LOAD_TIMEOUT, bit13 BRIDGE_TIMEOUT, bit14 MBOX_TIMEOUT |
| 0xB10 | `CTL_BOOT` | RW | boot address (Ibex `boot_addr_i`), 256-byte aligned: [7:0] read 0, writes ignored. Taken when RV_RUN rises |
| 0xB14 | `CTL_LOAD_ADDR` | RW | TCM byte address of the load port (word aligned); a write clears `CTL_LOAD_CRC` |
| 0xB18 | `CTL_LOAD_DATA` | RW | write: the word goes into a core-side holding register and from there to `CTL_LOAD_ADDR`, then the address adds 4; read: the word there (no increment). Only while RV_RUN = 0; otherwise ignored and LOAD_REFUSED. An access that cannot complete within 4,096 core cycles (clk_rv absent or in reset) reads 0 or is dropped, and sets LOAD_TIMEOUT |
| 0xB1C | `CTL_LOAD_CRC` | R | CRC-32 of the words accepted since the last `CTL_LOAD_ADDR` write (computed in the core domain) |
| 0xB20 | `CTL_RV_FREQ` | R | clk_rv cycles in the last 2^16 core cycles: F_RV = value × F_CORE / 65,536; 0 = clk_rv absent |
| 0xB40–0xB5C | `CTL_H2R_MSG0`–`7` | RW | host → RISC-V message words (core-domain registers) |
| 0xB60 | `CTL_H2R_SEQ` | RW | written after the words: the doorbell. Copies the words into the transfer register at once; the transfer to clk_rv follows |
| 0xB64 | `CTL_H2R_ACK` | R | last sequence the RISC-V consumed (core-domain copy) |
| 0xB80–0xB9C | `CTL_R2H_MSG0`–`7` | R | RISC-V → host message words (core-domain copies, updated before `CTL_R2H_SEQ`) |
| 0xBA0 | `CTL_R2H_SEQ` | R | last sequence the RISC-V posted (core-domain copy) |
| 0xBA4 | `CTL_R2H_ACK` | RW | the host writes the sequence it consumed |
| 0xBC0–0xBD4 | `CTL_DONE_SEQ0`–`5` | R | completions per unit (core-domain counters) |
| 0xBE0–0xBF4 | `CTL_ISSUED_SEQ0`–`5` | R | issued per unit (synchronised copies) |
| 0xC00–0xDFC | `DMI` | R / W | debug module register r (DMI address, 7 bits) at 0xC00 + 4r; a read is a DMI read, a write a DMI write. The CSR transfer ends when the DM answers or after 4,096 core cycles; then a read returns 0, a write is dropped and DMI_TIMEOUT is set; a late answer is recognised by its request tag and discarded (§7.1) |

- The spec rule "모든 CSR 읽기는 부수 효과가 없다" holds: the mailbox is a single message slot
  per direction with sequence / acknowledge registers, not a FIFO the host would pop by reading.
- **Every host access completes without clk_rv.** Every host-readable `CTL_*` register is a
  core-domain register: the mailbox words, sequences and acknowledges the RISC-V writes reach
  core-side copies through handshakes (§7.1), so a read is a register mux (the `docs/prof-dbg.md`
  §3.2 rule). Host writes land in core-side registers and complete at once. The accesses that must
  wait for clk_rv — a `DMI` access, a `CTL_LOAD_DATA` read, a `CTL_LOAD_DATA` write or a doorbell
  while the previous transfer is still in flight — end after at most 4,096 core cycles with a
  sticky bit. A stopped or relocking `ap_clk_2`, a wrong divider setting or a clk_rv domain in reset
  (the bring-up cases) therefore never hangs the CSR port; `CTL_STATUS.RV_CLK_OK` tells the host.
- `CAPS` gets no bit: CAPS describes the ISA a command buffer can use, and the control core
  changes nothing a command buffer sees (`CALL` is issue-only). Discovery is `CTL_ID`, as the
  debug block uses `DBG_ID`; trace tools that compare CAPS exactly (`tsr_xrt_selftest`) keep
  working.

### 3.4 Loading the image

1. After a hardware reset RV_RUN = 0 (the RISC-V is held in reset; TCM contents are undefined).
2. For each ELF `PT_LOAD` segment: write `CTL_LOAD_ADDR` = its physical address, then its words to
   `CTL_LOAD_DATA`; compare `CTL_LOAD_CRC` with the host's CRC. `.bss` is not loaded (crt0 zeroes
   it).
3. Write `CTL_BOOT`, set RV_RUN; wait for the runtime's hello message (R2H) carrying the image id.

Time: 128 KiB is 32 K posted writes; at an assumed 0.2–1 µs each (unmeasured) 7–33 ms, once per
model load. Later code overlays need no hardware: the host puts the code in DRAM, the resident
runtime LDs it into SPM and copies it into ITCM through the SPM window (16 K word reads at
≈ 55–80 ns ≈ 1 ms per 64 KiB, *estimate*), then `fence.i`. A DMA path into the TCMs (a 512-bit
path into the RISC-V domain and a new DMA mode) is rejected for v1: not needed at these sizes.

**Host-less targets** (`docs/prof-dbg.md` §7.4): the TCMs are undefined after reset and CSR writes
are the only load path, so a board without a host bus needs either a small boot ROM in the RV
domain (it LDs the image from DRAM or flash into SPM and copies it through the SPM window, the
overlay path) or the slow JTAG → `CTL_LOAD_*` route. Proposal: the boot ROM as a parameter
(`HAS_BOOTROM`), off on the U200.

### 3.5 SPM window

- Byte address 0x4000_0000 + 64 × line + offset; little-endian, as the SPM layout (f32 element i
  of a line is bytes 4i … 4i+3), so `volatile float *` views of SPM vectors work directly.
- Uncached and non-speculative (Ibex is in-order and does not speculate loads).
- Loads: byte, half, word. Stores: byte, half, word through `tsr_spm`'s per-byte enables (no
  read-modify-write). Misaligned accesses are split by Ibex into two aligned ones and are not
  atomic; only aligned word accesses are single-copy atomic. Accesses at or beyond `SPM_BYTES`
  fault.
- One access at a time; each completes before the next (Ibex waits for `rvalid`). The bridge
  answers a store only after `tsr_spm` accepted the write, and the issue logic forwards a push only
  while no bridge store is outstanding, so an op pushed after a store sees it (§5.5) whatever
  Ibex's LSU does internally.
- **Bound.** The bridge waits at most 4,096 clk_rv cycles for the core side (a stopped core clock,
  a core in reset); then the access ends with a bus error (`data_err`: load or store access
  fault) and `CTL_STATUS.BRIDGE_TIMEOUT`, so the hart stays reachable by the debugger. A late answer
  is discarded (request tag, §7.1).
- Path: RISC-V request register → handshake CDC → SPM client 3. Client 3 is shared with the debug
  SPM window (`docs/prof-dbg.md` §3.5) by a 2:1 mux, debug first: the debug window only acts while
  the system is quiescent or halted (§10.1), so it cannot starve the RISC-V in normal operation.
  The read response is the 512-bit shared bus, the bridge selects the word.
- Arbitration (with `HAS_CTRL = 1`): `tsr_spm` becomes round robin over its four clients (§5.3). A
  waiting client gets the port after at most three other grants, i.e. within ≈ 4 core cycles.
  With the fixed priority that prof-dbg planned, the RISC-V would be bounded only by the longest
  stream of a higher client, e.g. an LD of 65,536 lines (218 µs).
- Latency (*estimate*, 166.7 MHz RISC-V, 300 MHz core): store ≈ 4–6 RISC-V + 4–7 core cycles
  ≈ 40–60 ns; load ≈ 55–80 ns (plus `SPM_RD_LAT` 4).
- Accesses during a command-buffer run are allowed (§6).
- Intended use: small control data — TOPK results, expert ids and weights, flags, ring indices,
  argument blocks. Bulk copies are slow by design.

### 3.6 No DRAM window

The RISC-V gets no direct DRAM access:

- DRAM bandwidth is the product: weight streams use every channel at line rate. A 4-byte RISC-V
  access would cost a full 64-B beat and ≈ 0.3–0.5 µs of DRAM latency (*estimate*) and would share
  `tsr_mem_arb` with the weight streams.
- It would be a second ordering path into DRAM beside DMA, with its own rules against LD / ST.
- The DRAM address space is per channel (`(ch, beat)`), not one flat physical space; other targets
  (HBM pseudo-channels, Zynq, ASIC) differ again.
- Data moves only by tile DMA. DRAM structures shared with the host (the M9 token ring, argument
  blocks, model tables) are reached by LD / ST through the SPM. Memory source id 3 stays free
  (prof-dbg reserves it for trace streaming).

### 3.7 Interrupts and timer

Ibex's interrupt inputs are level-sensitive; every one below is computed or held in clk_rv from
synchronised values, and the RISC-V clears each cause at a clk_rv register:

- `irq_fast[0]`: the `WAIT_TICKET` ticket is done (from the synchronised `DONE_SEQ`).
- `irq_fast[1]`: `ISSUE_STATUS.ERR_IRQ`, a clk_rv sticky bit set on the rising edge of the
  synchronised "some unit in error state" level, cleared by W1C and independent of QCLEAR. The
  handler clears it and returns; recovery (join, QCLEAR) runs in thread context, since a QCLEAR
  completes only after every earlier entry (§4.5).
- `irq_fast[2]`: H2R message pending (`H2R_SEQ` ≠ `H2R_ACK`, both clk_rv registers).
- `irq_fast[3]`: R2H slot free (`R2H_ACK` = `R2H_SEQ`, the acknowledge synchronised). True most of
  the time, so software enables it in `mie` only while it waits to post.
- `irq_fast[4]`: completion records at least half full.
- `irq_fast[5]` (optional): the `WAIT_CREDIT` unit has a credit.
- `irq_timer`: MTIME ≥ MTIMECMP. `irq_software`, `irq_external` and `irq_nm` are tied to 0.

Ibex's `mtvec` is vectored only (*confirm*); the runtime installs a vector table at `CTL_BOOT`.
Joins need no handler: WFI wakes on an interrupt pending and enabled in `mie` even with
`mstatus.MIE` = 0 (privileged spec; *confirm* for Ibex). MTIME is the core's clock (300 MHz on the
U200), so RISC-V timestamps share the time base of the completion records and of `PERF_CYCLES`.
Ibex has no `time` CSR; the runtime reads MTIME. Host side: v1 has no device → host interrupt for
R2H; the host polls the CSR window at an unmeasured PCIe read cost (open item, §15).

---

## 4. Asynchronous tile issue

### 4.1 Issue port

- Memory-mapped (§3.2), not a custom instruction: a custom instruction needs a coprocessor
  interface that Ibex does not have, i.e. a modified core. (Stock compilers would emit one with
  `.insn`; the core is the blocker.)
- Push: store the words the op uses into `ISSUE_W0`–`W7`, optionally `ISSUE_DEP0` / `ISSUE_DEP1`,
  then store `ISSUE_PUSH` with the word mask. At the push the issue logic, in one step: takes
  w0–w7 (masked), the deps and the flags; picks the unit from w0's opcode (`ISSUE_QMAP`); assigns
  the ticket {unit, `ISSUED_SEQ[unit]`++}; writes the entry into the issue FIFO; resets the deps
  to none; sets `ISSUE_TICKET`.
- **Credits.** A push needs a free slot of its unit's queue (counted from the push until the unit
  accepts the op) and a free issue-FIFO entry; `ISSUE_CREDIT` shows the minimum. **Software checks
  the credit before it pushes**: the library's `issue()` spins on `ISSUE_CREDIT` (a local read,
  interrupts stay serviceable; optionally WFI on `irq_fast[5]`). The hardware wait is only a safety
  net: a push without a credit completes late (`rvalid`) when a credit returns, for at most
  `ISSUE_WAIT_MAX` = 65,536 RISC-V cycles, then it ends with a bus error (store access fault) and
  PUSH_TIMEOUT, so a stuck queue never wedges the hart. While the store waits Ibex can take neither
  an interrupt nor a debug halt, and the wait can be long in normal operation: with an MVM queue of
  4, the 6th outstanding MVM waits for a whole MVM, 311,040 core cycles (≈ 1.04 ms) for a 32B
  ffn_up / down, more than 65,536 cycles at any F_RV considered (393 µs at 166.7 MHz).
- An opcode the issue path does not accept (or whose unit is absent, QMAP 15) still gets a ticket,
  on the SYNC unit, and completes with `ILLEGAL_OP`: every push has a ticket and a completion.

### 4.2 Units and queues

| Id | Unit | Opcodes | Depth (U200 proposal) | Why |
|---|---|---|---|---|
| 0 | DMA | LD, ST (DQROW if the DMA implements it, M7) | 8 | double-buffered loops keep 2–3 LD / ST ahead; KV tiles more |
| 1 | QACT | QACT | 4 | one or two ahead of each MVM |
| 2 | MVM | MVM | 4 | back-to-back MVMs (3 per MoE expert) |
| 3 | VU | VOP, RED, ROPE, TOPK, SOP (M6 on) | 8 | many short ops |
| 4 | CMD | CALL (§6) | 2 | |
| 5 | SYNC | NOP (join point), FENCE (barrier), QCLEAR (§4.5) | 4 | no datapath |
| 6–13 | reserved | a second DMA engine, a DQROW unit | | |
| 14 | — | software marker records | | |
| 15 | none | "no dependency" | | |

- Each queue is in order: an op starts after the previous op of its unit is done, through the
  unit's existing dispatch port (`cmd_valid / cmd_ready / done`, unchanged). Entries hold the
  *checked* unit command (`tsr_dma_cmd_t` 101 b, `tsr_qact_cmd_t` 133 b, `tsr_mvm_cmd_t` 294 b,
  §5.2), the deps, the flags and the push index; SYNC entries also hold a FENCE / QCLEAR snapshot.
- Depths are parameters (`Q_DEPTH_*`); measured kernels fix them (open item). The issue-FIFO depth
  `ISSUE_FIFO_DEPTH` is independent of them, because `ISSUE_CREDIT` includes the free FIFO entries:
  a credit-checked push never waits for FIFO space. U200: 32 (LUTRAM; pushes made while the
  dispatcher waits during a command-buffer run, §5.2, collect there); an ASIC: 8 (§12.2).

### 4.3 Tickets and joins

- Ticket = {unit[31:28], seq[27:0]}. The sequence counts from 0 per unit after a hardware reset
  and is never reset otherwise (RISC-V reset, SOFT_RESET, QABORT and QCLEAR keep counting), so
  tickets stay unique across RISC-V restarts.
- Ticket (u, s) is done iff (`DONE_SEQ[u]` − s) mod 2^28 ∈ [1, 2^27). Completions are in order
  per unit, so one counter per unit is the whole completion state: no per-ticket storage. A
  ticket held for more than 2^27 later ops of its unit becomes ambiguous (≈ 3 × 10^5 32B tokens
  of MVMs); the library never holds tickets that long.
- **Stale tickets.** A ticket pushed before the last QCLEAR is stale: as a dependency it is
  `BAD_DEP` (the core keeps a clear point per unit), and `error(t)` on it is undefined (§4.5).
- **Push index.** The dispatcher numbers ticketed entries in push order (32 bits, mod 2^32) and
  keeps the number with each queue entry; error records keep it (`ERR_PIDX`). It orders errors
  across units, which per-unit tickets cannot (§4.5).
- Joins: `done(t)` (one local load); `join(t)` (spin); `join_irq(t)` (`WAIT_TICKET` = t, then WFI
  until done); a `group` joins the latest ticket per unit (at most 6 compares); `join_all()` pushes
  a FENCE and joins it.

### 4.4 Dependencies, SYNC and FENCE

- Up to **2 dependencies per op** in hardware: the op does not start before both tickets are
  done. The check is at the head of its queue against the core-domain completion counters (no CDC
  on this path); a waiting head blocks only its own queue.
- Deps on the op's own unit are implied by queue order (the library drops them); deps on one unit
  keep only the latest (in-order units); more than two are joined by a SYNC NOP (its ticket is one
  dep) or by a software join.
- `BAD_DEP` (detected by the checker in push order: the dispatcher knows how many ops of each unit
  it has seen): a dep on a ticket not pushed yet, on a reserved unit (6–14), or below its unit's
  clear point (§4.3).
- **SYNC unit**: `NOP` completes when its deps are done (a join point with a ticket). `FENCE`
  completes when every op pushed before it, on every unit, is done. `QCLEAR` is a FENCE that then
  clears the error states (§4.5). The snapshot they wait for is taken by the dispatcher: it pops the
  FIFO in push order, so its per-unit counts at the entry's position equal `ISSUED_SEQ[0..5]` at the
  push. The snapshot (6 × 28 bits) is a side field of the SYNC queue entry; the instruction words
  stay as software wrote them, and the checker applies the spec's rule unchanged (NOP and FENCE use
  no words: w1–w7 must be 0, else `ILLEGAL_OP`). `after(fence_ticket)` orders an op after
  everything pushed before the fence.
- **Completion rule (invariant I1).** Every entry — one that runs, a poisoned one, an `ABORTED`
  one, a `DEP_FAILED` one, a NOP, a FENCE, a QCLEAR — completes only at the head of its queue and
  only after both its deps are done (a `BAD_DEP` dep is ignored: it may never complete); a FENCE or
  QCLEAR also waits for its snapshot. The same comparators serve all cases. Hence **done(t) implies
  that every op t depends on, transitively (its deps, the earlier entries of its unit, the entries a
  FENCE covers, and theirs), is done.** `join_all()` after an error therefore still waits for every
  op pushed before it, and the race detector's clock joins along dependency edges (§9.2) are sound.
- Why in hardware: without dependencies the RISC-V must join before every dependent push, so it
  runs at most one stage ahead; with them it pushes a whole pipelined loop and the queues hold the
  schedule (§8.7). Cost: two 28-bit modular compares per queue head, each against a 6:1 mux of the
  counters, ≈ 150–250 LUT per head; the SYNC head adds six compares (*estimate*).

### 4.5 Errors

- **Checks at dispatch**, in push order, by the same checker as cmdproc (§5.2) and in the spec's
  order, with the issue-path rules first: `ILLEGAL_OP` (register fields ra, rb, rc, rd not 0; an
  opcode not accepted on the issue path; the spec's existing rules) → `BAD_DEP` → `BAD_FMT` →
  `K_RANGE` → `T_RANGE` → `SPM_RANGE` → `DRAM_RANGE`. A failing op enters its queue as a poisoned
  entry; it changes no state, as in a command buffer.
- **At the head** (after the deps, §4.4), in this order: a poisoned entry completes with its code;
  an entry of a unit in error state completes `ABORTED`; an entry with a dep that completed with an
  error completes `DEP_FAILED`; otherwise it runs. None of these runs the unit.
- **Unit error state.** A unit enters it when one of its entries completes with a nonzero code —
  at the head, in queue order, never at dispatch, so which earlier entries of that unit ran does not
  depend on timing. The first such entry sets `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0` and
  `ERR_PIDX`. While the state holds, every later entry of the unit completes `ABORTED`, and
  dependents elsewhere fail in turn, so an error propagates along queue order and dependency chains.
  `ISSUE_STATUS.ERR_IRQ` is set.
- **Errors are visible before completion (invariant I2).** `D[u]` does not advance past `ERR_SEQ`
  until the RV side has acknowledged unit u's record (a handshake of ≈ 5–8 core cycles, *estimate*,
  paid once per error state; later entries of the unit simply wait). So `DONE_SEQ` covering a ticket
  implies that its record, and the error-state bit, are already visible: `error(t)` is decidable
  from the per-unit records at any clock ratio, and a failed op can never read as a success.
- **First error.** `first_error()` is the record with the smallest `ERR_PIDX`. For a valid program
  the set of failing entries does not depend on the schedule, so neither does the result (and a
  `DEP_FAILED` record always has a larger push index than its cause).
- **QABORT** (host `CTL_CTRL`, at once; RISC-V `QCTRL`, when the dispatcher pops it): every entry
  not started completes `ABORTED`, in order, each after its deps; a running op completes normally
  (units have no cancel); a running CALL ends at its next instruction boundary. All six units enter
  error state (`ERR_CODE` ABORTED, `ERR_SEQ` = the first entry of the unit not started) until a
  QCLEAR, so every entry pushed after the abort, and every entry still in the FIFO, also completes
  `ABORTED`, whatever the unit and its timing. Which entries had started when the abort landed is
  timing-dependent, as for a host ABORT of a command buffer; the error codes report it.
- **QCLEAR.**
  - RISC-V (`QCTRL` bit 3): a SYNC entry with a ticket, in push order, never aborted. It completes
    when every entry pushed before it has completed, then clears all error states and records and
    sets each unit's clear point to its current sequence; its own `D` increment waits until the RV
    side shows the cleared records (as I2). Entries pushed after it run normally. Software joins its
    ticket (the library's `qclear()`).
  - Host (`CTL_CTRL` bit 9): accepted only when no entry is queued or running, the issue FIFO is
    empty and the RISC-V is quiet (`RV_RESET_DONE`, or halted with no push in flight); otherwise
    QCLEAR_REFUSED. The host cannot order a clear against a running RISC-V's pushes, so it must stop
    or halt the RISC-V first.
- Why stop on error: a queue holds a pipeline; running past a failed LD would compute on stale
  SPM data silently. The command-buffer path stops at the first error for the same reason.
- **Codes**: 1–8 keep their meaning; new **11 `DEP_FAILED`**, **12 `BAD_DEP`** (issue path only);
  9 `LOOP_NEST` and 10 `WAIT_TIMEOUT` become reserved (§11.3). `ERR_INFO`: as the spec for range
  errors (the overflowing address), else 0; for CALL the program's `ERR_PC`. `ERR_W0`: the failing
  op's w0; for CALL the program's `ERR_INFO`.

### 4.6 ABORT, SOFT_RESET, resets and halts

| Event | Queued entries | Op running on a unit | CALL program running | RISC-V | Notes |
|---|---|---|---|---|---|
| QABORT: RISC-V (`QCTRL`, in push order) or host (`CTL_CTRL`, at once) | complete `ABORTED`, in order, after their deps; all six units in error state until QCLEAR | runs to completion (no cancel, `docs/rtl-m2.md` §3.0) | ends at its next instruction boundary; ticket `ABORTED`, `ERR_INFO` = `ERR_PC` | continues | sequences keep counting |
| QCLEAR: RISC-V (ticketed SYNC entry) | applies after every earlier entry completed | — | — | continues | clear point = current sequence |
| QCLEAR: host | refused (QCLEAR_REFUSED) unless idle and the RISC-V quiet (§4.5) | | | | |
| Hold (any source) | stay queued; heads do not start (entries already `ABORTED` still drain) | completes | continues | continues | |
| `CTRL.ABORT` (host) | unchanged | | — | | host runs only, as today; a START still waiting for the drain ends at once (`ABORTED`, `ERR_PC` 0, DONE \| ERR) |
| `CTRL.SOFT_RESET` (host) | unchanged | | ends at its next boundary (ticket `ABORTED`), then the reset applies | not reset | as today: STATUS, `ERR_*`, PC, R1–R15, s0–s15 of bank 0 (§4.7), `PERF_*`; immediate when no run is active; a START waiting for the drain ends without starting. Queues, TCMs, mailbox, sequences, bank 1 (the issue-path s bank) and `CTL_PERF` are kept |
| RV_RUN 1 → 0, or the DM's ndmreset | QABORT (at once) | completes | aborted | reset sequence (§7.3): outstanding accesses finish, then Ibex, staging, deps, `WAIT_*` and the RISC-V hold source are reset; TCMs and mailbox kept | sequences kept |
| Debug halt (DM) | continue, unless HOLD_ON_HALT (then the halt hold source while halted) | | | halted | |
| Hardware reset | all state reset; TCM and SPM contents undefined; the host reloads the image | | | | sequences 0 |

SOFT_RESET keeps today's meaning and acts on the command-buffer path only (D7); the queues and the
RISC-V are stopped with `CTL_CTRL.QABORT` and RV_RUN, which the host API does (`tsr_ctrl_reset`,
§8.4). The SOFT_RESET CSR write itself completes at once, as today.

**Quiescent state.** `CTL_STATUS.QQUIESCENT` = no datapath op running (§6), the dispatcher and the
issue FIFO idle, every non-empty queue held, and no `ABORTED` entry left to drain. Held queued ops
are allowed: they have had no effect. With the RISC-V also halted, the architectural state (RISC-V
registers and CSRs, TCMs, SPM lines written, DRAM) equals the simulator after the same number of
retired RISC-V instructions and the same completion vector `DONE_SEQ[0..5]`, given that run's
co-simulation log: the values of timing-dependent MMIO reads and the trap and interrupt entries
with their retire positions (§9.4). With HOLD_ON_HALT it is reached at any breakpoint once the
running ops finish, however far ahead the RISC-V had pushed. This is the control-core analogue of
the debug block's invariant (`docs/prof-dbg.md` §3.1) and is checked by random-halt tests (§9.3).

### 4.7 Register fields and R0–R15

- Issue-path ops must have **ra = rb = rc = rd = 0** (else `ILLEGAL_OP`): the RISC-V computes
  absolute addresses and writes them as the immediates, so every address field is
  A = R0 + imm = imm.
- **R0–R15 belong to the command-buffer path.** A queued op never reads or writes them, so queued
  ops share no integer-register state. `LI`, `ADDI` and `END` are not accepted on the issue path.
- `CALL` is the bridge: its program starts with R1–R7 = 0, R8–R12 = w3–w7 (ARG0–ARG4),
  R13–R15 = 0 (as START does with ARG0–ARG7); results come back through SPM or DRAM.
- **s0–s15 (M6). Decided (D8): a separate bank for the issue path.** Issue-path ops that read or
  write scalar registers (RED, SOP, VOP with a scalar operand, and MVM if it takes one) use bank 1;
  command-buffer runs (host START, CALL) use bank 0, which every run starts at +0.0 (the
  simulator's START rule today; the initial value is fixed at M6). Bank 1 is written only by VU
  queue ops, in VU queue order, and read by the RISC-V through the core window after a join;
  scalars pass between the paths through SPM words (SOP.LDS / STS). Bank 1 is +0.0 after a
  hardware reset (proposal, fixed at M6) and keeps its value across SOFT_RESET, the RISC-V reset
  and QCLEAR. Cost ≈ 512 FF and a 2:1 read mux (*estimate*). So a CALL that lands between SOP.LDS
  and VOP.MULS on the VU queue cannot overwrite the scalar (the MoE example of the spec relies on
  s1), and no race on s registers between the two paths can occur: bank 0 belongs to one
  command-buffer run at a time, and only bank 1 is in the race definition, between VU queue ops and
  the RISC-V's core-window reads (§5.5 rule 4). The rejected alternative was one shared bank with
  s0–s15 in the race definition and CALL / START as writers.

### 4.8 Issue cost (estimates)

- RISC-V side per op: the credit read, 3–7 word stores (only the words the op uses), 0–2 dep
  stores, the push and a ticket load ≈ 6–12 local MMIO accesses ≈ 8–16 cycles, plus 10–25
  instructions to compute the words: **25–40 RISC-V cycles** (150–240 ns at 166.7 MHz).
- Core side: issue FIFO crossing 2–3 core cycles; checker 8–10 cycles (≈ 46 for MVM); queue entry
  1 cycle. Checker throughput ≈ 30 M ops/s (≈ 6.5 M MVMs/s) at 300 MHz.
- Per token: 0.5B ≈ 1 K ops → ≈ 0.04 M RISC-V cycles (0.24 ms at 166.7 MHz) and ≈ 15 K core
  cycles of checks, both overlapped with execution as long as the RISC-V keeps the queues fed.

---

## 5. Concurrency in the core

### 5.1 Block diagram

```
          clk_rv domain                         ¦                 clk (core) domain
                                                ¦
   ┌─────────┐ instr ┌──────┐ wr port ◄─────────¦◄── CTL_LOAD_* holding reg (tsr_ctl_csr, host CSR port)
   │  Ibex   │──────►│ ITCM │                   ¦
   │ RV32IMC │       └──────┘                   ¦
   │         │ data ─┬──► DTCM                  ¦      ┌─────────┐   ┌─────────────────────────┐
   └────▲────┘       ├──► control window ─push──¦─FIFO►│ checker │──►│ unit queues + dep check │─► cmdproc dispatch regs ─► DMA, QACT, MVM, VU
  debug │            │      │  ctrl entries ────¦─(same│ tsr_chk │   │ SYNC; CMD ─► tsr_cmdproc│   (existing dispatch)
   ┌────┴─┐          │      ▲  ▲  ▲      FIFO)  ¦      └────▲────┘   └────┬────────────────────┘
   │  DM  │◄─ DMI ───┼──────┼──┼──┼──hs, tagged─¦◄── DMI (CSR port, 4,096-cycle timeout)
   └──────┘          │      │  │  └── gray ─────¦◄── D[u], S[u] ◄─────────┘ (done / accept counts, ≤ 1 per cycle)
                     │      │  └───── gray ─────¦◄── core cycle counter (MTIME)
                     │      └──────── hs ───────¦◄─► mailbox: transfer registers ↔ core-side copies (tsr_ctl_csr)
                     └──► SPM bridge ── hs ─────¦──► SPM client 3 (2:1 with the debug window) ─► tsr_spm (RR, 4 clients)
```

### 5.2 Checker and dispatcher

- **Checker** (`tsr_chk`): the decode and check pipeline of `tsr_cmdproc` (S_D0–S_D5 and the error
  priority of S_EXEC) becomes its own module with an input select: cmdproc's instruction register
  or the dispatcher's FIFO head. For issue-path ops the register reads are forced to 0 and the
  issue-path rules (§4.5) are added. One instance (decided, D11): during a command-buffer run
  (host START or CALL) it serves cmdproc and the dispatcher waits — the queue heads are held during
  such runs anyway (§6) — otherwise it serves the dispatcher. For cmdproc this is a refactor with
  the same pipeline and cycle counts, verified by the existing replays with `HAS_CTRL = 0` and 1.
- **Dispatcher**: pops the issue FIFO in push order. Control entries act at the pop (QABORT, QHOLD
  set / clear, TRACE_MARK); QCLEAR and op entries get the push index and, for FENCE and QCLEAR, the
  snapshot (§4.4); op entries run the checker; the result (checked command or poisoned entry) goes
  into its unit queue. Room is guaranteed: credits count an op from its push until its unit accepts
  it, so the dispatcher never blocks on a full queue. Consequence of the single checker: an entry
  pushed during a command-buffer run waits at the dispatcher until the run ends, and so does every
  entry behind it, including a RISC-V QABORT (the host's QABORT acts at once). Accepted (D11):
  host execs and launches alternate (D9), so the runs that delay pushes are CALLs the kernel
  itself issued; a second checker instance (≈ 1–2 K LUT, *estimate*) is not built.
- **Queue head**: an entry that runs starts when its deps are done, no hold applies, the
  exclusivity and drain rules allow it (§6) and the unit is free; it then loads cmdproc's existing
  dispatch register of that unit (§7.4). S[u] counts at the accept (credit return), D[u] at `done`
  or at the completion of an entry that does not run (§4.4), at most one per cycle per unit, and
  never past `ERR_SEQ` before the record acknowledge (§4.5). Queues live in LUTRAM (≈ 7.5 Kb in
  total).
- `HAS_CTRL = 0` builds none of it, including the SPM round robin (§5.3): today's core, bit for bit
  and cycle for cycle.

### 5.3 SPM arbitration

- With `HAS_CTRL = 1`, `tsr_spm` changes from fixed priority to **round robin**, separately for the
  read and the write port, keeping the registered grant: the last granted client has the lowest
  priority next. A waiting client is granted after at most `NCL` − 1 = 3 other grants. With
  `HAS_CTRL = 0` it keeps fixed priority, with PD2's debug window as the lowest client (a
  parameter, `SPM_ARB`).
- With one active client (every command-buffer run, as long as the RISC-V does not touch the SPM
  during it, §6) the grant sequence is today's, so the existing replays and their cycle counts are
  unchanged.
- Timing: the request mux becomes 4-way. With a binary-encoded registered select and a separate
  one-hot ready flop, one output bit is one LUT6 (2 select + 4 data inputs), the same depth as
  today's 3-way AND-OR. k4's worst path (SPM write-port bank select → URAM cascade enable,
  +0.015 ns) is after the input register; k4t1 already lists "SPM write grant → SPM input
  register" among its worst paths, and that one does see the change. Gate and mitigations: §7.4.
- **No-deadlock argument under round robin** (to write into the `tsr_mem_arb`, `tsr_dma` and
  `tsr_spm` contracts at CR2; today's R4 and `tsr_dma.sv:35-45` assume the DMA is the
  highest-priority SPM writer): (i) every SPM client raises a request only with its payload ready
  (rule R2) and never waits on memory while it holds a request — the DMA's LD writes beats already
  in its queue; QACT, the MVM gather and the VU write results they hold; the bridge holds its data;
  ST and WS reserve buffer space before they request memory; (ii) a granted request transfers in
  that cycle; (iii) round robin grants a waiting client after at most `NCL` − 1 other grants. So
  every SPM request completes within a bounded number of cycles that does not depend on memory
  progress, and the DMA's `rd_rsp_ready` drops only for such a bounded time. That replaces R4's
  "local reasons only" by "bounded, memory-independent reasons", which is what head-of-line
  routing needs. Simulation assertions: a raised SPM request is never withdrawn and does not wait
  on a memory handshake; a granted request always transfers.
- **Head-of-line cost.** Under round robin the DMA's beat queue can fill while other clients write
  (QACT ≈ 20–25 % write duty for a few hundred cycles, the bridge, MVM gather < 1 %, §5.6); it then
  drops `rd_rsp_ready`, and the channel's response path is head-of-line, so weight beats behind a
  DMA beat wait. Measured in CR2 with `tb/dma` `sys_rtl_spm_contend` extended to four clients, plus
  a deadlock case: LD and MVM on one channel while the gather, QACT and the bridge saturate the write
  port. If it matters: the letter of `docs/rtl-m2.md` D2 (reserve queue space per burst: queue 8 →
  128–256 beats, ≈ 1–2 K LUT of LUTRAM or 8–16 RAMB36, *estimate*), a per-source response skid in
  `tsr_mem_arb`, or DMA-first write priority with a bounded run (other clients granted at least
  every N cycles).
- M6's vector unit needs 2 reads + 1 write per cycle and therefore the planned second read port or
  banking anyway (`docs/rtl-m2.md` §5); round robin extends to it.

### 5.4 DRAM channels

`tsr_mem_arb` already round-robins bursts between FETCH, DMA and WS with one outstanding budget;
nothing changes there. An LD or ST that runs beside an MVM takes its beats out of that channel's
weight stream, and the slowest channel ends the MVM, so the MVM ends later by at least the LD's
beat count. Overlap therefore hides an LD's latency and fixed costs, not its transfer time. On top
of the beats come DDR4 effects that are not measured (M0 has not run on the card): a ST between
weight bursts on one channel adds read / write bus turnarounds, and an LD in another row adds row
conflicts; both can cost more than the beat count and could cancel the gains of §5.7 and §8.7.
Card runbook item (proposal): `tsr_bw` with a read stream at 64-beat bursts and concurrent writes
at 1–10 % duty on the same channel, and two interleaved read streams. If the measured penalty is
larger than the overlap gain, the SDK makes ST depend on the same-channel MVM (no hardware
change). The v1 DRAM map puts IO on ch0 only, which makes cluster 0 the straggler; the C++ runtime
should spread activations, KV blocks and rings over the channels (the low regions of ch1–ch3 are
free in the spec's map). That is a software choice; the hardware needs nothing.

### 5.5 Memory model (adopted in the spec, 2026-10-24)

1. **Per-op results.** Each op's result is defined by the spec's semantics of that instruction and
   is bit-exact whatever runs beside it, if the program is race-free.
2. **Order** comes only from:
   - (a) *queue order*: ops on one unit execute and complete in push order; an op sees every effect
     of the earlier ops of its unit;
   - (b) *dependencies*: an op with a dep on ticket t starts after t is done and sees its effects;
   - (c) *the RISC-V*: after it observed t done (a `DONE_SEQ` read, the WAIT interrupt) its later
     SPM loads see t's effects; an op pushed after a RISC-V store completed sees the store;
   - (d) *FENCE and QCLEAR*: such a ticket is done only when every op pushed before it is done;
   - (e) *completion*: every entry, also one that completes without running, completes only after
     its deps (invariant I1, §4.4), so done(t) implies everything t depends on, transitively, is
     done;
   - (f) *command-buffer runs*: a run (host START or CALL) starts only after every datapath op that
     started before it has completed, and no queue op starts until the run ends (§6).
   Two ops on different units with no dependency chain may run in any order or overlap.
3. **Visible** means the unit's `done`: every SPM write accepted by `tsr_spm`, every DRAM write
   response received (`docs/rtl-m2.md` §3.0).
4. **Race**: two accesses to the same SPM byte, DRAM beat or issue-bank s register, at least one a
   write, by two agents — queue ops, CALL runs, host command-buffer runs, the RISC-V, the host — not
   ordered by rules 2 and 5. A racy program is a software bug and its results are undefined; the
   hardware does not detect races; the simulator's race detector does (§9.2).
5. **Host**: DRAM the host wrote before posting an H2R message (with the transfer complete) is seen
   by ops pushed after the RISC-V read that message. DRAM written by ops is seen by the host after
   it reads an R2H message the RISC-V posted after joining those ops. Host command-buffer runs and
   control-core launches are ordered by the transport's ownership rule (D9, §6, §8.4): an exec
   starts only when no launch is outstanding, a launch only when no host run is active.
6. **Command-buffer runs** are exclusive with datapath queue ops (§6) and internally sequential as
   the spec defines.
7. As-if optimisations inside a unit stay allowed (M5 MVM-to-MVM overlap).
8. **Scalar registers**: the issue path uses its own s-register bank and command-buffer runs keep
   theirs (decided, D8, §4.7), so no s-register race between the two paths exists; the issue
   bank's accesses follow rules 2 and 4 like SPM bytes.

### 5.6 SPM port contention (estimates from the spec's formulas)

Qwen2.5-32B decode shapes, T = 1, Q4_K weights (attn_v and the output head converted to Q8_0),
300 MHz, every channel at 1 beat per cycle. Duration = weight beats per channel for an MVM; the
duty is the share of that duration the op needs on an SPM port.

| Op | Duration alone (cycles) | SPM reads (lines) | SPM writes (lines) | Read-port duty | Write-port duty |
|---|---|---|---|---|---|
| MVM ffn_up / gate, N 27648, K 5120 | 311,040 | 92 (QA broadcast) | 1,728 (y) | 0.03 % | 0.56 % |
| MVM ffn_down, N 5120, K 27648 | 311,040 | 493 | 320 | 0.16 % | 0.10 % |
| MVM attn_q, N 5120, K 5120 | 57,600 | 92 | 320 | 0.16 % | 0.56 % |
| MVM attn_k, N 1024, K 5120 | 11,520 | 92 | 64 | 0.80 % | 0.56 % |
| MVM output head, N 152064, K 5120, Q8_0 | 3,231,360 | 90 | 9,504 | < 0.01 % | 0.29 % |
| QACT A256, K 5120 | ≈ 430 (1.1 cycles per source line + 83) | 320 | 92 | ≈ 75–90 % | ≈ 20–25 % |
| QACT A32, K 5120 | ≈ 2,600 (16 cycles per 32-block) | 320 | 90 | ≈ 12 % | ≈ 3.5 % |
| LD of x, 5120 f32 | ≥ 320 + DRAM latency | 0 | 320 | 0 | up to 100 % |
| ST of y, 5120 f32 | ≥ 320 + DRAM latency | 320 | 0 | up to 100 % | 0 |
| VOP, n 5120, two sources (M6) | ≥ 640 with one read port | 640 | 320 | 100 % | 50 % |

- A long MVM leaves more than 99 % of both SPM ports to everything else; at T = 8 its traffic grows
  8× and stays under 5 %.
- Short ops (QACT, LD, ST, VOP) share the ports among themselves roughly as if serialised: no gain
  and no loss against sequential execution.
- The cost an MVM pays for concurrency is the channel beats of concurrent LD / ST and their DDR
  effects (§5.4), not SPM ports.

### 5.7 What concurrency buys (estimates)

- **Dense decode, large model** (32B): non-MVM time ≈ 1–2 % (prof-dbg §5.11 planning estimate), so
  overlap gains at most ≈ 1–2 %.
- **Dense decode, small model** (0.5B): G = 3.98 % in the cycle model, plus the host control cost
  the control core removes: ≈ 3,000 posted writes and 870 reads per token (§1). At an assumed
  0.2 µs per write and 1 µs per read (unmeasured) that is ≈ 1.5 ms per token beside 7.6 ms of
  device time (≈ 20 %).
- **Latency-bound phases**: the matvec batch of §8.7 with 0.5B shapes ≈ 15–20 %; attention over KV
  tiles (M7): LD of tile i+1 beside VU work on tile i, up to ≈ 2× for that phase when the two are
  comparable; MoE (M8): expert ids read from SPM and expert MVMs pushed back to back, with no host
  round trip per layer.
- These overlap gains assume no DDR penalty beyond the stolen beats (§5.4); they are small enough
  that turnarounds and row conflicts could cancel them.
- So the main return is host independence and latency hiding in short-op phases, not dense-MVM
  throughput.

---

## 6. Coexistence with the command-buffer path

| | A: mode bit (host chooses the issuer) | B: `CALL` op, cmdproc as a queued unit (**proposed**) |
|---|---|---|
| Hardware | a mux at each unit's dispatch port: cmdproc or the queues own the units; `CTL_CTRL.MODE` | cmdproc becomes unit 4 (CMD); an exclusivity interlock with the queue heads |
| llama.cpp path | unchanged in mode 0 | unchanged: host START as today |
| RISC-V reuse of lowered programs | no (the host switches modes; the RISC-V cannot run a command buffer) | yes, for position-independent programs, the only ones CALL is used for (D10, below) |
| Ordering between the two | host-managed mode switches (drain, switch, resume) | tickets and dependencies: a CALL is an op like any other |
| Risk | mode-switch races, two separate worlds | START interplay with queued CALLs (rules below) |

**Rules of B.**

- `CALL` (opcode 0x09, issue path only, `ILLEGAL_OP` inside a command buffer — no nesting): w1
  command buffer start (ch0 beat), w2 instruction count, w3–w7 → R8–R12. The run uses the existing
  fetch, check and execute path and the debug block. It does not change `STATUS`, `ERR_*`, `done`,
  `PERF_*`, `CMD_*` or `ARG*` (host-run state); it reports through its ticket (END: success; an
  error: §4.5), through `CTL_PERF_*` and through `CTL_STATUS.CALL_BUSY`.
- **Host-visible state a CALL changes**: R1–R15 (`DBG_R*`), s0–s15 of bank 0, `PC`, `DBG_IR*`,
  `DBG_OPA*`, `DBG_RETIRED`, `DBG_NEXT_PC`, `DBG_IA`, `DBG_UNIT`, `DBG_HIST*`. So a CALL after a
  failed host run destroys that run's post-mortem state (`docs/prof-dbg.md` §3.4): tools read it
  before the RISC-V may CALL again (the transport's ownership rule below gives them that window),
  and `CTL_STATUS.LAST_RUN_CALL` says whose state it is. `DBG_IA` is computed from the base latched
  at the run's start (`CMD_ADDR` for START, w1 for CALL), not from the `CMD_ADDR` CSR. A halted CALL
  shows `DBG_STATUS.HALTED` and `CALL_BUSY`; `STATUS.HALTED` mirrors host runs only, so the
  prof-dbg rule "STATUS = BUSY | HALTED" holds for host runs and STATUS is untouched by CALLs.
- **Running** means: a datapath unit (DMA, QACT, MVM, VU) holds an accepted op. SYNC entries never
  count, and neither does an entry that completes without running.
- **Exclusive, with a drain rule**: a command-buffer run (host START or CALL) starts when no
  datapath op is running. While a host START is pending, or a CALL at the CMD head has its deps
  done, no other queue head starts a new datapath op (priority: host START, then CALL); entries
  that complete without running still complete. While the run executes, no queue head starts a
  datapath op (pushes still enter the queues). So a run waits at most for the longest running op
  (≈ 10.8 ms for the 32B output head), a continuously fed pipeline cannot starve it, and a FENCE
  that depends on a queued CALL cannot deadlock against it (`call; join_all()` is the normal case).
  Hence command-buffer runs behave exactly as today: sequential semantics, DONE ⇒ memory
  quiescent, the debug block's invariant, `PERF_*`; and a `HAS_CTRL = 1` core replays every
  existing trace identically (with RV_RUN = 0, see the next rule).
- **RISC-V SPM accesses during a run** are not blocked (holding the bridge for a whole run would
  trip the bridge bound of §3.5). Results are unaffected for race-free programs; the run's cycle
  counts may differ, because the bridge then competes for the SPM ports. The replays with
  identical cycle counts run with RV_RUN = 0.
- **Host START while datapath queue ops run or CALLs wait**: accepted as today (BUSY at once,
  `CTL_STATUS.START_WAIT`); it starts when the running datapath ops end, ahead of a ready CALL. A
  host START while a CALL runs or is halted is accepted the same way and starts after that CALL
  ends (a transport that keeps the ownership rule below never does this). A `CTRL.ABORT` while it
  waits ends the run at once (`ABORTED`, `ERR_PC` = 0, DONE | ERR, so the adapter's `ap_done`
  follows); a SOFT_RESET ends it without starting.
- **PERF gating**: every `PERF_*` counter counts only while the host run executes, from its actual
  start (cmdproc leaves the START wait) to its end, with prof-dbg's halted-time rule unchanged.
  `PERF_CYCLES` therefore equals today's BUSY count whenever there is no wait, which is always the
  case with `HAS_CTRL = 0`; the wait itself goes to `CTL_PERF` 20, and the queue ops' unit and
  channel events in that window go to `CTL_PERF_*`. During the executing run only its own units are
  active (exclusivity; a datapath op that started before has completed, including its DRAM write
  responses), so the values equal a `HAS_CTRL = 0` core's and "실행 사이 PERF 불변" and the prof-dbg
  PERF deltas stay exact. Cost ≈ 10 LUT.
- **Ownership (transport, decided, D9)**: host lowering uses the SPM freely from line 0, and RISC-V
  kernels keep live SPM buffers across ops, so a host run that lands between two queue ops would
  overwrite them, and a mutex inside one host process cannot order autonomous device activity.
  Rule: one process owns the compute unit (XRT exclusive mode, `docs/xrt-host.md` §4.1), and the
  transport alternates the two paths: `tsr_transport_exec` starts only when no launch is
  outstanding (every launch's R2H DONE consumed and QIDLE); `tsr_ctrl_launch` starts only when
  BUSY = 0; a resident loop (M9) counts as an outstanding launch, so execs are refused while it
  runs. This is software only: the SPM is not split between host lowering and RISC-V kernels, and
  the hardware does not change. (The rejected alternative partitioned the SPM so that both paths
  could run at once.) Command-buffer runs remain agents in the race definition (§5.5 rule 4).
- What exclusivity costs: no overlap between a CALLed program and datapath queue ops; a CALLed
  program is sequential inside anyway. Relaxing it is an open item.
- **Arguments and lowering (decided, D10)**: the M8 and M9 decode steps are C++ kernels that issue
  their tile ops directly (§11.1); the position-dependent operands (KV write address, embedding
  row, per-block attention counts, loop offsets) are computed in C++ and written as immediates.
  CALL is used only for position-independent command-buffer programs, whose operands do not
  change with the position; its 5 argument registers (R8–R12) are for such programs' own needs.
  The backend gets no register-relative lowering mode: lowering keeps emitting R0-relative
  (absolute) addresses. (The rejected alternative would have lowered stage-B layer programs with
  their position-dependent operands relative to argument registers, `R[x] + imm`, so that the
  RISC-V could CALL them per token.)

---

## 7. Clock domains and CDC

### 7.1 Crossings

| Crossing | Direction | Structure | Width | Latency (*estimate*) |
|---|---|---|---|---|
| Tile issue and control entries (QABORT, QHOLD set / clear, QCLEAR, TRACE_MARK) | RV → core | async FIFO, gray pointers, `ISSUE_FIFO_DEPTH` entries (U200 32, LUTRAM) | ≈ 300 b (256 instruction + 2 × 32 deps + unit + flags + kind) | 2–3 core cycles |
| Issued counters `ISSUED_SEQ[u]` | RV → core | gray-coded 28-bit counters (6), 2-flop synchronisers | 168 b | 2–3 core cycles |
| Credits S[u], completions D[u] | core → RV | gray-coded 28-bit counters (6 each), 2-flop synchronisers; each steps by at most 1 per core cycle | 336 b | 2–3 RV cycles |
| Error records (and their clear) | core → RV | tagged handshake per unit; D[u] does not pass `ERR_SEQ` before the acknowledge (§4.5) | ≈ 6 × 128 b ({seq, code, info, w0, push index}) | ≈ 5–8 core cycles |
| Completion records | core → RV | async FIFO, 64 × 128 b, fed by a 1-entry holding register per unit and a round-robin arbiter | 128 b | 2–3 RV cycles |
| SPM bridge | RV ↔ core | 4-phase request / acknowledge with a request tag, data held stable, one access in flight; RV-side bound 4,096 clk_rv cycles | 70 b request / 33 b response | 40–80 ns round trip |
| Core window (s0–s15 of the issue bank) | RV ↔ core | as the SPM bridge, same bound | 32 b | ≈ the bridge |
| Mailbox | core ↔ RV | the doorbell (`CTL_H2R_SEQ`, `R2H_SEQ`) copies the 8 words into a transfer register in the writer's domain; a tagged handshake carries words + sequence into the reader's copy, sequence last; acknowledges cross the same way. Host reads only core-domain registers | 8 × 32 + 32 b | ≈ 10–20 core cycles |
| `CTL_LOAD_DATA` | core → RV | write: core-side holding register + handshake into the TCM write port; read: tagged request / response; CSR timeout 4,096 core cycles | 32 b + address | |
| DMI | core ↔ RV | tagged handshake (2-phase toggle with a 1-bit request tag); CSR timeout 4,096 core cycles; after a timeout the path answers locally (read 0, write dropped, DMI_TIMEOUT) until the late answer with the old tag has returned and been discarded | 41 b / 34 b | |
| MTIME | core → RV | 64-bit gray counter | 64 b | |
| clk_rv cycle count (`CTL_RV_FREQ`, RV_CLK_OK) | RV → core | 16-bit gray counter, sampled every 2^16 core cycles | 16 b | |
| RV-domain `CTL_PERF` (14 bridge accesses, 15 bridge wait cycles, 17 push stalls) | RV → core | 32-bit gray counters in clk_rv, extended to 64 bits in the core domain (the core samples every cycle, so it sees every wrap) | 3 × 32 b | |
| Sticky events (PUSH_TIMEOUT, BRIDGE_TIMEOUT) | RV → core | toggle synchroniser into the `CTL_STATUS` sticky bits | | |
| Status levels (RV_RUNNING, RV_HALTED, RV_SLEEP, RV_RESET_DONE, "clk_rv domain out of reset") and interrupt sources | both | 2-flop synchronisers; RV_RESET_DONE takes two stages more than the FIFO write pointer, so the core never sees it before the last pointer update | | |
| RV_RUN, hold sources | core → RV / RV → core | levels, 2-flop synchronisers | | |
| Resets | | reset synchronisers (§7.3) | | |

Rules:

- **Gray counters step by at most 1 per source cycle.** Queue drains (aborted and poisoned entries,
  QABORT) complete at most one entry per cycle per unit; an assertion on every gray counter checks
  it.
- **No dropped records below a full FIFO.** Up to six units can complete in one cycle; each unit
  has a 1-entry record holding register, a round-robin arbiter moves one per cycle into the record
  FIFO, and a unit's completion waits for its holding register only when REC is set (records never
  stall execution: a record is dropped and counted in `CTL_PERF` 16 only when the FIFO is full).
- **Nothing crosses into a domain in reset.** Every core → clk_rv request path waits for the
  synchronised "clk_rv domain out of reset" level and, until then or while clk_rv is absent,
  answers locally within the CSR timeout (read 0, write dropped, sticky bit).
- **No CSR access waits on clk_rv without a timeout** (§3.3).

### 7.2 Structures and constraints

- Generic, vendor-independent modules in `rtl/core`: `tsr_cdc_fifo` (gray pointers; LUTRAM written in
  the source domain, read into a register in the destination domain), `tsr_cdc_gray` (counter),
  `tsr_cdc_hs` (handshake; one variant with a 1-bit request tag, used by every request path that
  can time out). They use `tsr_sync` (2–3 flops) and `tsr_rst_sync` from `rtl/tech` (Xilinx:
  `ASYNC_REG`; generic: plain flops), as the spec's layer table already foresees.
- Constraints (U200: a kernel XDC packaged in the `.xo`): `set_max_delay -datapath_only` of the
  smaller period on gray buses and handshake data, `set_bus_skew` on gray buses, `ASYNC_REG` on
  synchroniser flops. Do not use `set_clock_groups` (it would override the max delays), and do not
  rely on platform clock groups: the platform's clocks come from one clocking block, Vivado may
  time them as related, and 300 MHz against 500, 250 or 166.7 MHz has a worst-case edge separation
  of ≈ 0.67 ns, so one inter-clock path missing from the exceptions fails in the link (the
  out-of-context run cannot see the platform clocks), and v++ auto-scaling may then lower `ap_clk`,
  the core clock. Gates for CR2: `report_cdc` on the out-of-context run lists only the intended
  crossings, all safe; on the linked design, `report_cdc` and `report_timing` between `ap_clk` and
  `ap_clk_2` show zero unconstrained inter-clock paths.
- `CTRL_SYNC` = 1 (targets where clk_rv = clk): the CDC structures become plain register stages.

### 7.3 Reset sequencing

- **Hardware reset**: `ap_rst_n` resets the core domain as today. The RISC-V domain's hardware reset
  is `ap_rst_n` through `tsr_rst_sync` in clk_rv (asynchronous assert, synchronous deassert after
  2–3 clk_rv cycles). `ap_rst_n_2` is not used for CDC endpoints: an async FIFO or counter whose two
  sides can be reset independently loses pointer agreement; at most it is AND-ed into the Ibex / DM
  reset, entering the core as `rst_rv_n` (spec 코어 경계 인터페이스). Minimum `ap_rst_n` low:
  max(`RST_PIPE` + 1 core cycles, 3 clk_rv cycles) (U200: 4 core cycles or 18 ns at 166.7 MHz;
  Vitis holds it far longer, not measured). Nothing crosses before both domains are out of reset
  (§7.1 rules): the RISC-V stays held (RV_RUN = 0) until the host sets RV_RUN, and the counters are
  0 on both sides after reset.
- Ibex and riscv-dbg use asynchronous active-low resets: they get theirs from `tsr_rst_sync`
  (asynchronous assert, synchronous deassert in clk_rv) — a documented exception to
  `docs/rtl-m2.md` §1 for third-party blocks.
- **RISC-V reset** (RV_RUN 1 → 0, ndmreset) is a sequence, not a pulse:
  1. the core applies QABORT at once (§4.6);
  2. the clk_rv sequencer stops granting new accesses on both Ibex ports (`gnt` = 0) and waits until
     the outstanding one — SPM bridge, push, core window, DM — has completed, or until the bridge
     bound expires; a push still waiting for a credit is dropped without waiting (it never enters
     the FIFO and gets no ticket);
  3. it asserts the reset of Ibex, the control window's staging registers, deps, `WAIT_*` and the
     RISC-V hold source (minimum width 3 clk_rv cycles);
  4. it raises RV_RESET_DONE.
  Not reset by it: the issue-FIFO and counter endpoints, the SPM bridge and DMI endpoints, the
  error-record and completion-record endpoints, the mailbox registers and their copies, MTIME, the
  TCMs. So no pointer resynchronisation is ever needed and no half-written FIFO entry or stale
  acknowledge can reach the restarted hart; a late bridge answer for a request whose master went
  into reset is completed and discarded by its tag.
- **Boot order**: core out of reset → host loads the image → RV_RUN = 1 → the RISC-V reset deasserts
  2–3 clk_rv cycles after the synchronised RV_RUN → fetch at `CTL_BOOT` + 0x80.
- The debug module's DMI side follows the hardware reset only, so a debugger stays attached across
  RISC-V resets (the debug spec's ndmreset behaviour).

### 7.4 U200 clock, placement and timing

- **Ports**: `ap_clk_2` / `ap_rst_n_2` on `tsr_u200_k1` / `k4` (kernel.xml, `gen_tops.py`; those
  files belong to `docs/platform-u200.md`'s owner). The platform's two kernel clocks (DATA 300 MHz,
  KERNEL 500 MHz) are clock-wizard outputs that XRT programs from the xclbin's clock topology
  (*confirm*), so a direct frequency request for clock 1 is probably supported: in the CR2 smoke
  link try `--kernel_frequency "0:300|1:166.7"` (legacy form) or `--clock.freqHz` / `--clock.id` on
  `ap_clk_2` (*confirm* the option form). Only if that fails, divide 500 MHz inside the kernel with
  BUFGCE_DIV in `rtl/tech/xilinx/tsr_clk_div.sv` (the primitive stays in `rtl/tech`; clock
  modifying blocks in a DFX reconfigurable region on UltraScale+, *confirm* in UG909).
- **F_RV** (decided, D12): the requirement is ≥ 100 MHz (*estimate*, §2.1). Planning value
  166.7 MHz (500 / 3); 250 MHz (500 / 2) only if the CR1 out-of-context run of Ibex with its TCMs
  closes there with margin; 125 MHz (500 / 4) as the fallback (≈ 83 MIPS at CPI 1.5, still
  ≥ 50 MIPS). How the clock is made (a v++ frequency request or BUFGCE_DIV, above) is checked in
  the CR2 smoke link.
- **Scaling**: if v++ lowers `ap_clk_2` to close timing, nothing breaks (the CDC is asynchronous);
  software reads `CTL_RV_FREQ` and never assumes F_RV.
- **Placement**: SLR1, beside the CSR block, cmdproc, SPM and DMA (all of the RISC-V's partners are
  there in C'); no new SLR crossing. SLR1 has ≈ 140 K LUT and ≈ 350 RAMB36 left after C' and
  DDR[2]'s controller (*estimate* from §1); the control core needs ≈ 12–20 K LUT and ≈ 32–34 RAMB36
  (§12.1). The LUT count is not the constraint; local congestion and logic depth next to `tsr_spm`
  are. Only client 3's request / response registers sit next to `tsr_spm`; the rest (Ibex, TCMs,
  CDC, CTL_PERF) goes in a pblock in SLR1 away from the URAM columns. k1 lives entirely in SLR1 and
  keeps fitting.
- **Dispatch timing.** k4t1's worst path is cmdproc state → DMA / MVM command register clock
  enables (0 levels, +0.108 ns). A second command source in front of the units would add a LUT
  level exactly there. The queue heads therefore load cmdproc's existing dispatch registers
  (`dma_cmd_q` / `qact_cmd_q` / `mvm_cmd_q` and their valid flops, §1) through a 2:1 select on their
  D inputs, selected by a quasi-static "run executing" flop; the unit's command-register enable
  stays one flop with 0 levels, and cmdproc's dispatch timing and cycle counts are unchanged
  (*confirm* when the RTL is written; if a separate stage were needed instead, `HAS_CTRL = 1`
  command-buffer runs would take one more cycle per dispatched instruction and the "identical
  cycle counts" claim would have to say so).
- **CTL_PERF read mux**: its own CSR pipeline stage (the CSR port is latency-insensitive), and only
  the defined counters are built (§3.3).
- **CR2 timing gate** (decided, D13): the control-core RTL is frozen only after the out-of-context
  runs of k1, k4t1 and k4 (C') with `HAS_CTRL = 1`, and the k4 v++ link, meet 300 MHz, with the
  same SPM-margin proposals of `docs/platform-u200.md` §8.2 applied to both builds (with and
  without the control core). Margins today: k4 +0.015 ns out of context, the k4 link projected at
  −0.1 to −0.2 ns, k1's link +0.011 ns (§1). A failed link would make v++ lower the scalable core
  clock (to about 283–291 MHz, *estimate* there): `HAS_CTRL = 1` must not cost the core's 300 MHz.
  Bring `HAS_CTRL = 1` up on k1 first (one SLR), knowing that its margin is small too. The k4 link
  needs a host with ≥ 32 GB (`docs/platform-u200.md` §8.2); it runs after the host RAM upgrade the
  user will do when the card goes in, so the freeze waits for it.

---

## 8. Software

### 8.1 Device SDK

```
sw/ctrl/
├── include/tsr/kernel.hpp    # op builders, ticket, issue, after, join, group, SPM views
├── include/tsr/mmio.hpp      # control window offsets (generated from the shared register table)
├── include/tsr/backend_dev.hpp, backend_host.hpp   # MMIO backend / simulator backend (§8.5)
├── rt/crt0.S, rt/link.ld, rt/runtime.cpp   # vectors, traps, mailbox loop, kernel table
├── kernels/                  # matvec_batch.cpp (§8.7), moe_layer.cpp, decode_step.cpp (M8), decode_loop.cpp (M9)
├── tools/tsr-elf             # image check, kernel table dump, image id
└── cmake/riscv32.cmake       # toolchain file
```

- Compilers: `riscv64-unknown-elf-gcc` ≥ 12 (multilib rv32imc / ilp32) or Clang ≥ 16
  (`--target=riscv32-unknown-elf`); `-march=rv32imc_zicsr_zifencei -mabi=ilp32` (GCC ≥ 12 needs
  `_zicsr_zifencei` spelled out).
- Freestanding C++20 (C++17 also works): `-ffreestanding -fno-exceptions -fno-rtti
  -fno-threadsafe-statics -fno-use-cxa-atexit -nostartfiles -ffp-contract=off`, `-Os` by default.
- No heap by default (no `operator new`; a static arena if a kernel needs one); libgcc for
  soft-float and integer helpers; no newlib in the default image (a small formatter writes debug
  text to a DRAM log ring in debug builds). Inline assembly only for CSRs, WFI and `fence.i`.

### 8.2 Header library (API sketch)

```cpp
namespace tsr {
struct spm  { uint32_t line; };                    // SPM line address
struct dram { uint8_t ch; uint32_t beat; };        // channel + beat
constexpr uint32_t handle(dram d) { return (uint32_t(d.ch) << 28) | (d.beat & 0x0FFF'FFFFu); }  // ABI form, §8.3
constexpr dram from_handle(uint32_t h) { return {uint8_t(h >> 28), h & 0x0FFF'FFFFu}; }
struct wref { uint32_t w_off; uint8_t fmt; };      // symmetric-allocated weight tensor (TSR_FMT_*)
enum class qa : uint8_t { a32 = 0, a256 = 1 };
constexpr qa qa_for(wref w) { return w.fmt == 0 ? qa::a32 : qa::a256; }   // Q8_0 -> A32, Q4_K -> A256
constexpr uint32_t f32_lines(uint32_t n) { return (4 * n + 63) / 64; }

struct op { uint32_t w[8]; uint8_t used; };        // one 256-bit tile instruction, used-word mask
constexpr op LD  (spm dst, dram src, uint32_t lines);
constexpr op ST  (spm src, dram dst, uint32_t lines);
constexpr op QACT(qa kind, spm dst, spm src, uint32_t k, uint32_t t, uint32_t src_stride = 0);
constexpr op MVM (spm dst, spm x, wref w, uint32_t n, uint32_t k, uint32_t t,
                  uint32_t dst_stride = 0, bool acc = false);
// VOP, RED, ROPE, TOPK, SOP (M6/M8), DQROW (M7), CALL, NOP, FENCE likewise

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

The device backend of `issue`, `done` and `error`:

```cpp
inline ticket issue(const op& o, deps d, uint32_t flags) {
  volatile uint32_t* w = mmio::ctrl;                     // 0x2000_0000
  const unsigned u = unit_of(o);                         // opcode class -> unit (ISSUE_QMAP, read at boot)
  while (((w[0x030 / 4] >> (4 * u)) & 0xFu) == 0) {}     // ISSUE_CREDIT: local spin, interrupts serviceable
  for (unsigned i = 0; i < 8; ++i)
    if (o.used & (1u << i)) w[i] = o.w[i];               // ISSUE_W0..W7: only the words this op uses
  if (!d.a.none()) w[0x020 / 4] = d.a.v;                 // ISSUE_DEP0
  if (!d.b.none()) w[0x024 / 4] = d.b.v;                 // ISSUE_DEP1
  w[0x028 / 4] = (uint32_t(o.used) << 8) | flags;        // ISSUE_PUSH: does not wait after the credit check
  return ticket{w[0x02C / 4]};                           // ISSUE_TICKET
}
inline bool done(ticket t) {
  if (t.none()) return true;
  const uint32_t x = (mmio::ctrl[0x040 / 4 + t.unit()] - t.seq()) & 0x0FFF'FFFFu;
  return x - 1u < 0x07FF'FFFFu;                          // 1 <= (DONE_SEQ - seq) mod 2^28 < 2^27
}
inline uint32_t error(ticket t) {                        // valid once done(t): the record arrived first (I2)
  if (t.none()) return 0;
  const volatile uint32_t* r = mmio::ctrl + (0x0C0 + 16 * t.unit()) / 4;   // ERR_SEQ, ERR_CODE
  if (r[1] == 0) return 0;                               // unit not in error state
  const uint32_t x = (t.seq() - r[0]) & 0x0FFF'FFFFu;
  if (x >= 0x0800'0000u) return 0;                       // t completed before the first error
  return x == 0 ? r[1] : TSR_ERR_ABORTED;                // the failing entry, or ABORTED after it
}
```

`issue()` relies on the single hart and on not being reentered (§3.2): nobody can take the credit
between the check and the push.

### 8.3 Runtime, image format, launch protocol, kernel ABI

- **Linker script**: `.vectors` at ITCM 0 (`CTL_BOOT` = 0); `.text` in ITCM; `.rodata`, `.data`,
  `.bss` and the stack (8 KiB default, top of DTCM) in DTCM. `.rodata` and `.data` are loaded by the
  host directly into DTCM, so crt0 copies nothing.
- **Kernel table** `.tsr_kernels` (emitted by `TSR_KERNEL(name, …)`): per kernel {name hash 32,
  entry 32, nargs 8, flags 8 (bit0: argument block), name[22]}. **Metadata** `.note.tsr`: SDK ABI
  version, required control-core version, minimum ITCM / DTCM sizes, required CAPS bits.
- **Image** = one static ELF (no relocations at load). The host loads `PT_LOAD` segments by
  physical address; the image id is the SHA-256 of the loaded bytes, echoed by the runtime's hello
  message.
- **Kernel ABI** (proposal): up to 5 arguments of 32 bits: integers, SPM line numbers, or DRAM
  handles (ch[31:28], beat[27:0]: up to 16 GiB per channel, `handle()` in §8.2). A kernel that
  needs more takes an argument block: the host writes it into the DRAM IO region and passes its
  handle as argument 0; the runtime LDs it into the **SPM argument range** (proposal: the top 16
  lines of the SPM, 1 KiB), joins, and passes the kernel a pointer into the SPM window. Kernels never
  put buffers into that range (buffers from line 0 upwards cannot collide with it).
- **crt0**: sp, gp, zero `.bss`, `init_array`; set `H2R_ACK` = `H2R_SEQ` (drop any message posted
  before this start, so the interrupt is not pending on stale data); read `R2H_SEQ` and continue
  numbering from it; push `qclear()` and join it (clears what a QABORT or a dead program left;
  that program's entries still in the FIFO drain `ABORTED` before it, §4.5); enable the interrupt
  sources in `mie`; enter the runtime loop.
- **Runtime loop**: wait for an H2R message (poll, or WFI on `irq_fast[2]`); copy it into DTCM and
  acknowledge it at once (`H2R_ACK`). `LAUNCH` {kernel id, launch id, args[5]} → stage the argument
  block if the kernel's flag says so → call the kernel → `join_all` (its FENCE completes only after
  every entry pushed before it, aborted ones included, §4.4) → post R2H `DONE` {launch id, return
  value, first failing ticket, its code}; after a failure, `qclear()` and join it, so the next launch
  starts without error state (debug builds also assert `QSTATUS.QIDLE` after the final join).
  `PING`; `STOP`. Posting waits for the R2H slot (`R2H_ACK` = `R2H_SEQ`, WFI on `irq_fast[3]`);
  since the runtime acknowledges H2R on receipt and every host wait loop drains R2H (§8.4), the two
  single slots cannot deadlock. Debug text goes to a DRAM log ring, never to R2H. The trap handler
  posts R2H `TRAP` {mcause, mepc, mtval} and parks in a WFI loop, where the host can attach the
  debugger.
- **M8 per token**: H2R `LAUNCH` {decode_step, token id, position}: the host writes 2 argument words
  plus the kernel id and the sequence number, which is what the M8 criterion ("인자 2개만 쓰기")
  counts.

### 8.4 Host API (`tsr_transport`)

Proposed C additions (to `tsr_transport.h` after its owner agrees; they use PD1's
`tsr_transport_csr_write`):

```c
typedef struct tsr_ctrl_image tsr_ctrl_image;                  // parsed ELF + kernel table
typedef struct tsr_ctrl_result { uint32_t status, ret, err_ticket, err_code; } tsr_ctrl_result;

int  tsr_ctrl_image_open (const char * elf_path, tsr_ctrl_image ** out);
void tsr_ctrl_image_close(tsr_ctrl_image * img);
int  tsr_ctrl_kernel_id  (const tsr_ctrl_image * img, const char * name);     // -1 if absent
int  tsr_ctrl_load (tsr_transport * t, const tsr_ctrl_image * img);           // tsr_ctrl_reset, TCM writes, CRC check
int  tsr_ctrl_boot (tsr_transport * t, const tsr_ctrl_image * img, uint64_t timeout_ns);  // CTL_BOOT, RV_RUN = 1, hello
int  tsr_ctrl_launch(tsr_transport * t, int kernel_id, const uint32_t * args, int nargs, uint32_t * launch_id);
int  tsr_ctrl_wait (tsr_transport * t, uint32_t launch_id, uint64_t timeout_ns, tsr_ctrl_result * res);
int  tsr_ctrl_mbox_send(tsr_transport * t, const uint32_t msg[8]);
int  tsr_ctrl_mbox_recv(tsr_transport * t, uint32_t msg[8], uint64_t timeout_ns);
int  tsr_ctrl_reset(tsr_transport * t);    // RV_RUN = 0, QABORT, wait RV_RESET_DONE and QIDLE, QCLEAR
```

- **Open** (proposal): unless the caller asks to attach (an observe / attach flag, like PD1's
  `TSR_OPEN_OBSERVE`), the transport stops a control core an earlier process left running:
  `tsr_ctrl_reset` after today's ABORT of a BUSY core. One process owns the compute unit.
- **Ownership** (D9, §6): `tsr_transport_exec` waits under the device mutex until no launch is
  outstanding (every launch's R2H DONE consumed and `CTL_STATUS.QIDLE`); `tsr_ctrl_launch` waits for
  BUSY = 0; while a resident loop runs, execs fail with a clear message.
- **Wait loops**: every host wait (`mbox_send` waiting for `H2R_ACK`, `wait`, `mbox_recv`) also
  drains R2H into a host-side queue, so a runtime blocked on posting never blocks the host.
- Implementations: `sim` (in process: the RV interpreter and the queue model; `tsr_ctrl_wait` steps
  the simulator), `mock` (adapter model + sim), `vkernel` (the Verilated kernel with a second
  clock), `xrt` (CSR window accesses as in `TSR_XRT_CTRL=regs`; a control-core launch is not an
  `ap_ctrl_hs` run; wrapping it in one through the reserved adapter offset 0x01C is an open item;
  under D9 no control-core access overlaps an `xrt::run`, so only an `ap_ctrl_hs`-wrapped launch
  would depend on the spec open item about window access during a run).
- **Launch cost**: each launch on the single-slot mailbox needs at least one PCIe read
  (≈ 1–2 µs, *estimate*, unmeasured). That caps fine-grained launch rates (CR5); a DRAM launch ring
  with a doorbell is an open item, to be decided by the per-launch cost measured on the card.

### 8.5 Host build of the same headers

The same kernel sources compile for the host with `backend_host.hpp`: `issue` calls the simulator's
queue model directly, without the RV interpreter. Fast unit tests, host GDB, ASan / UBSan on kernel
logic. It is a convenience, not an oracle: the reference is the RV binary on `tsr_rv` (exact control
flow, exact soft-float, exact MMIO timing model). The headers use fixed-width types for device
addresses and never hand host pointers to the device; spec-fp math goes through `tsr_ref` on both
sides, so the two builds compute the same bits.

### 8.6 JIT integration (sketch)

- **Registry**: the ELF's kernel table becomes a host-side registry (C++, and a Python binding)
  with per-kernel constraints (formats, K multiples, maximum sizes).
- **Launch by name**: `tsr_ctrl_kernel_id(img, "rmsnorm_f32")`, then `tsr_ctrl_launch` with device
  addresses (handles) and sizes, or an argument block (§8.3).
- **No code generation**: the JIT selects and parameterises precompiled kernels; whatever it cannot
  map runs on the host.
- **First target**: whichever has the cheaper extension point when CR5 starts — tinygrad (a device
  backend: an allocator over `tsr_transport` regions and a "compiler" that maps a scheduled kernel to
  a registry entry by pattern) or torch.compile (a backend that partitions the FX graph with a
  pattern table — linear with GGUF-quantized weights through a tensor subclass, RMSNorm, SiLU × mul,
  SDPA — and leaves the rest eager on the CPU). JAX later through a PJRT plugin with custom calls.
- Weight placement stays the backend's (symmetric allocation, repacking with the `sw/sim` reference
  code); the JIT layer reuses it.
- Caveats: matching generic framework kernels onto a fixed kernel set covers the LLM hot path
  (matmul, norms, attention, sampling), not arbitrary graphs; the launch rate of the mailbox
  (§8.4) limits how fine-grained the JIT's kernels can be.

### 8.7 Example: matvec batch with prefetch

```cpp
#include <tsr/kernel.hpp>
using namespace tsr;

struct mv_job { uint32_t x, y; wref w; uint32_t n, k; };   // y = W x; x, y: DRAM handles of f32 vectors; 24 B

// Double-buffered batch: the LD of x[i+1] and the ST of y[i-1] run on the DMA unit while MVM i
// streams its weights. The RISC-V never waits inside the loop except for a free queue slot, and
// joins once at the end. job points into the SPM argument range (the runtime LDed the block and
// joined before the call; 1 KiB holds 42 jobs). Buffers: X 2048 lines (K <= 32768 f32),
// Q >= 584 lines (QA_LINES(A256, 32768)), Y 2048 lines (N <= 32768), all far below the argument range.
TSR_KERNEL(matvec_batch, const mv_job* job, uint32_t n) {
  static constexpr spm X[2] = {{0}, {2048}}, Q[2] = {{4096}, {5120}}, Y[2] = {{6144}, {8192}};
  ticket ld[2], qa_[2], mv[2], st[2];                                 // "none": already done
  ld[0] = issue(LD(X[0], from_handle(job[0].x), f32_lines(job[0].k)));
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t b = i & 1;
    const mv_job j = job[i];
    // QACT i reads X[b] (LD i) and overwrites Q[b], which MVM i-2 read
    qa_[b] = issue(QACT(qa_for(j.w), Q[b], X[b], j.k, 1), after(ld[b], mv[b]));
    // MVM i reads Q[b] and overwrites Y[b], which ST i-2 read
    mv[b] = issue(MVM(Y[b], Q[b], j.w, j.n, j.k, 1), after(qa_[b], st[b]));
    // prefetch x[i+1] into X[b^1] (QACT i-1 read it); pushed before ST i, so it does not queue behind MVM i
    if (i + 1 < n)
      ld[b ^ 1] = issue(LD(X[b ^ 1], from_handle(job[i + 1].x), f32_lines(job[i + 1].k)), after(qa_[b ^ 1]));
    st[b] = issue(ST(Y[b], from_handle(j.y), f32_lines(j.n)), after(mv[b]));
  }
  join_all();
  return first_error();                                               // 0, or the first failing code
}
```

- **Schedule.** DMA queue order: LD0, LD1, ST0, LD2, ST1, LD3, … LD_{i+1} waits for QACT_{i-1};
  ST_i waits for MVM_i. In steady state MVM_i runs while the DMA does ST_{i-1} and LD_{i+1}, and
  QACT_{i+1} runs as soon as LD_{i+1} and MVM_{i-1} are done, i.e. during MVM_i, so the MVM unit
  runs back to back. Every op has at most two dependencies, so no SYNC NOP is needed.
- **Without hardware dependencies** the loop would `join(qa_[b])` before each MVM and
  `join(mv[b])` before each ST: the RISC-V would wait every iteration and the DMA could not run
  ahead.
- **Gain** (*estimate*, 300 MHz, MVM fill ≈ 300–600 cycles, DRAM latency ≈ 150 cycles, no DDR
  penalty beyond the stolen beats, §5.4): 0.5B shape (N = K = 896, Q8_0, 3,360 weight beats per
  channel): sequential ≈ 4,650–4,950 cycles per job, pipelined ≈ 3,770–4,070 (MVM + fill + 112 ch0
  beats stolen by LD / ST) → ≈ 15–20 %. 32B shape (N = K = 5120, Q4_K, 57,600 beats): ≈ 59,500 vs
  ≈ 58,700 → ≈ 1.4 %. The pattern pays where ops are short; on long bandwidth-bound MVMs it saves
  only the fixed costs (§5.4).

---

## 9. Verification

### 9.1 Simulator (CR1)

- **`sw/sim/tsr_rv.c`**: an RV32IMC + Zicsr interpreter that models the chosen Ibex configuration's
  CSRs and WARL behaviour (vectored `mtvec`, the implemented counters, misaligned splitting, access
  faults, WFI, the debug-mode subset `tsr-dbg` needs); ≈ 1.5–2.5 K lines (*estimate*). In-house
  rather than linking Spike: deterministic co-scheduling with the tile model, MMIO and trace hooks,
  speed, and the simulator stays one C library linked into the llama.cpp backend. Spike is the
  interpreter's **oracle** (the spec's oracle chain: Spike → `tsr_rv` → Ibex RTL): riscv-tests and
  riscv-arch-test signatures, plus random programs (riscv-dv's Python generator or a small in-house
  one) compared instruction by instruction; and `test_ref_fp.c` built for RV32 (§2.4).
- **Queue model**: the control window, queues and units as an event model on the core time base,
  with every rule of §4–§6 (completion rule, error state at the head, I2 ordering, drain rule,
  hold sources, QCLEAR in push order). Op durations come from the existing cycle model
  (`tsr_sim.c`) or are randomised (`TSR_SIM_SCHED=fixed|random:<seed>[,<min>,<max>]`); RISC-V time
  from a CPI table scaled by F_CORE / F_RV (`TSR_SIM_RV_RATIO`). An op's effects are applied
  atomically at its modelled completion; for race-free programs the choice is invisible. The
  device backend's credit waits are exercised with 32B shapes (a decode step pushing a layer
  ahead).
- `TSR_SIM_LOG` gains RISC-V retire lines and push / start / done events.

### 9.2 Race detector

- **Happens-before** by vector clocks over the agents {6 units, CALL runs, host command-buffer
  runs, the RISC-V, the host}, with the edges of §5.5: queue order, dependencies, RISC-V push,
  RISC-V observation of completion (a `DONE_SEQ` read returning v joins the clocks of every
  completion of that unit below v; the WAIT interrupt likewise), RISC-V program order, FENCE and
  QCLEAR, the run exclusivity edges, the transport's exec / launch ordering, mailbox messages. The
  joins along dependency edges are sound because of invariant I1.
- **Access sets per op** from its operands: LD, ST; QACT with T and stride; MVM's x, dst (ACC also
  reads dst) and weights per channel [W_OFF, W_OFF + rows_c × ROW_BEATS); VU ops; a CALL or host
  run is the union of its program's accesses; RISC-V SPM accesses per byte; issue-bank s registers
  as 16 locations (bank 0 of the command-buffer runs is not tracked, D8).
- **Shadow state**: per SPM line the last writer and the latest read per agent; per DRAM channel an
  interval map. A report names both tickets (or the RISC-V pc), the ranges and the ops.
- Also: reads of never-written SPM (existing rule), dependencies on one's own unit (lint),
  dependencies on stale tickets, a host QCLEAR that would be refused.
- `TSR_SIM_RACE=off|report|fatal` (tests use `fatal`). ≈ 0.8 K lines, ≈ 2–3× slower when on
  (*estimate*).
- Detection does not depend on the schedule for valid programs (§9.5), so one run finds every race.

### 9.3 RTL

- **Ibex itself**: a pinned, upstream-verified commit; its own DV is not rerun. Our integration
  (TCMs, buses, control window) runs riscv-tests (rv32ui / um / uc / mi -p) and riscv-arch-test
  (RV32IMC, Zicsr, privilege subset) from ITCM under Verilator; signatures against Spike.
- **Issue path and queues** (`tb/ctrl`, cocotb + Verilator): credits, the safety-net stall and its
  timeout; ticket numbering and wrap (counters preset near 2^28); deps including `BAD_DEP` (unpushed,
  reserved unit, stale) and `DEP_FAILED`; SYNC NOP, FENCE and QCLEAR; invariants I1 and I2 as
  assertions on every test; an `ILLEGAL_OP` push then `join_all()` with ops in flight (must wait);
  error state at the head and `first_error()` with errors on two units; QABORT then pushes (all
  `ABORTED` until QCLEAR); QCLEAR right after a push (ordered); hold sources (host read-modify-write
  of `CTL_CTRL` does not drop the RISC-V hold); SOFT_RESET during a CALL and during a START wait;
  ABORT during a START wait; RISC-V reset during a bridge access and during a push; CALL
  exclusivity: `call; join_all()`, and a host START against a continuously fed pipeline (bounded
  wait); mailbox sequence / acknowledge with both sides busy; SPM bridge widths, alignment, faults,
  ordering against pushes, bound; every CSR access with clk_rv stopped and with the clk_rv domain
  held in reset (must complete, RV_CLK_OK = 0); DMI timeout followed by the late answer (discarded);
  the SPM-window and quiescence rules (§10.1).
- **Two clocks**: a Verilator C++ bench with independent `clk` and `clk_rv`, ratios
  F_RV / F_CORE ∈ {0.31, 0.42, 0.55, 0.83, 1.0, 1.67} and random phase; every test at every ratio,
  including a directed error test per ratio (`error(t)` consistent with `done(t)`); the
  `CTRL_SYNC` = 1 build as well.
- **Core regression**: `HAS_CTRL` = 0 and 1 replay every existing trace set bit-exactly, with
  identical cycle counts for command-buffer runs (exclusivity; round robin with one client; the
  shared dispatch register, §7.4; RV_RUN = 0).
- **SPM and memory**: `tb/dma` `sys_rtl_spm_contend` with four clients, and the deadlock case of
  §5.3 (LD and MVM on one channel while the gather, QACT and the bridge saturate the write port).
- **Concurrency stress**: a generator of random race-free programs (random units, sizes and
  dependencies, checked by the race detector) run on RTL and simulator under the random-latency
  memory model of `tb/common` and random SPM contention: final state bit-exact.
- **Random halts**: HOLD_ON_HALT halts at random points of stress programs; at each quiescent point
  compare with the simulator advanced to the same retired count and completion vector, given the
  co-simulation log (§4.6).
- **Mutation campaign** (like `docs/rtl-m3.md` §13.3): drop a dependency compare, complete an
  aborted or poisoned entry before its deps, reorder a queue, skip a poison, advance D[u] before the
  record acknowledge, apply a RISC-V QCLEAR out of band, step a gray counter by 2, let one hold
  source's write clear another, count SYNC as running in the exclusivity check, round robin back to
  fixed priority, a CDC FIFO full flag one cycle late, a credit returned at `done` instead of
  accept. The tests must catch each.
- **Vivado**: out-of-context runs of the RISC-V domain (its Fmax, CR1) and of the whole core with
  `HAS_CTRL = 1` (k1, k4t1, k4 with floorplan C'), the k4 link (the CR2 timing gate, §7.4);
  `report_cdc` clean; zero unconstrained `ap_clk` ↔ `ap_clk_2` paths on the linked design.

### 9.4 Co-simulation and traces

- **Lockstep**: Ibex's RVFI outputs (simulation builds) give per retired instruction {pc, insn, rd,
  value, memory address, data} and trap / interrupt entries (*confirm* the RVFI fields of the pinned
  commit); `tsr_rv` steps in lockstep. Loads of timing-dependent registers (`DONE_SEQ`,
  `ISSUE_CREDIT`, MTIME, `REC_*`, `QSTATUS`, `H2R_SEQ`, `R2H_ACK`, the error records) take the RTL
  value, and trap and interrupt entries are taken at the RTL's retire position; every other value
  must match. That log is what the quiescent-state invariant (§4.6) is conditioned on. On the tile
  side the simulator applies each op's effects when the RTL reports its completion, so SPM and DRAM
  can be compared at any point; in this "follow the RTL schedule" mode even racy programs compare,
  which helps debugging.
- **Trace extension** (a trace stays a directory): `meta.json` gains `"ctrl"` {image file, boot,
  F_RV ratio, schedule}; `rv/image.elf`; `mbox_in.bin` (host messages, each with the R2H sequence
  after which it is posted); `ops/u<k>.bin` (the op stream per unit as pushed: words, deps, flags);
  `expect/r2h.bin` (every R2H message) and `expect/tickets.bin` (per unit: count and error record),
  besides the existing `init/` and `expect/` DRAM regions. `TSR_TRACE` dumps one trace per launch.
- **Two replays**: *full* (RISC-V + core RTL) and *tile-only* (a bench pushes `ops/u*.bin` through
  the issue path with their dependencies, no RISC-V), which separates core concurrency bugs from
  RISC-V integration bugs.

### 9.5 Bit-exactness for concurrent programs

A program (RV image + inputs + host message script) is **valid** if (1) it is race-free (§9.2) and
(2) its observable behaviour does not depend on the schedule: the op stream per unit, the R2H
stream, the RISC-V's SPM store stream (address, data, order) and the final SPM and DRAM are the
same under every schedule. That is the definition; the check is the determinism test: at least 8
random schedules in the simulator, comparing all four. The simulator's schedule space (a CPI table,
random op latencies) need not cover every RTL timing, so lockstep co-simulation at every clock
ratio (§9.4) covers the rest: a program that passes the 8 schedules but diverges on the RTL shows
up as a timing-dependent MMIO value followed by a different store or push, which is reported
against the program, not the RTL.

For a valid program the final state is unique, and the RTL and the hardware must reproduce the
simulator's exactly: the DRAM regions written (`expect/`), the SPM lines written (where compared,
as `tb/core` does today), the R2H messages and the ticket error records (including error paths:
§4.5 makes them deterministic). Not compared: RISC-V poll iteration counts, MTIME, `PERF_*`,
`CTL_PERF_*`, completion records (as `perf` is reference-only today). Kernel rule: timing-dependent
values (MTIME, credits, poll counts) must not influence data, stores or pushed ops, except in
logging that the comparison excludes.

---

## 10. Debug and profiling

### 10.1 RISC-V debug

- **Debug module**: pulp riscv-dbg (Debug Spec 0.13.2: halt, resume, step, abstract register
  access, an 8-word program buffer, ndmreset) in the clk_rv domain; Ibex execute triggers (2) as
  hardware breakpoints; software breakpoints in ITCM (writable).
- **Configuration** (*confirm* each when the commits are pinned): with the DM at 0x1000_0000
  (not 0) riscv-dbg needs its two-scratch debug ROM (`HasSndScratch` = 1, `DmBaseAddress` =
  0x1000_0000); Ibex implements `dscratch1`. The DM memory is accessible in and out of debug mode:
  as far as known, `ibex_top` has no debug-mode output to restrict it with, and OpenTitan does not
  restrict it either. The DM slave arbiter is §3.1's. riscv-dbg advertises system bus access in
  `sbcs` whatever is connected; with its master port tied idle an OpenOCD SBA access would leave
  `sbbusy` stuck, so the port answers every request with `gnt` = 1 and an error response, `sberror`
  is set and OpenOCD falls back to the program buffer. (Connecting SBA as a second master on the
  clk_rv bus would give non-halting memory reads: open item.)
- **Path**: host → CSR window `DMI` (0xC00, §3.3) → tagged handshake → DM.
- **GDB → OpenOCD** (riscv target):
  - U200, sim, vkernel: a `tsr-dbg rv-bridge` process that speaks OpenOCD's `remote_bitbang`
    protocol and emulates a JTAG TAP + DTM (dtmcs, dmi) in software, as Spike's remote bitbang does;
    its dmi accesses become CSR window accesses. Slow (one message per JTAG bit) — measured in CR4; a
    dedicated faster transport only if needed.
  - ASIC and JTAG targets (`docs/prof-dbg.md` §7.4): the real DTM's `dmi` reaches the CSR space at
    word address = offset / 4, so the DM sits at DMI address 0x300 (0xC00 / 4); OpenOCD's riscv
    target takes `-dbgbase 0x300` (*confirm* in the OpenOCD version used). Same register meaning
    everywhere.
- prof-dbg §8's "no GDB stub" stays for the tile core, whose source level is a ggml graph; the
  RISC-V has standard debug because its source level is C++.
- **Halt everything**: DM halt stops the RISC-V only; with HOLD_ON_HALT the queue heads are held too
  (the halt hold source). RISC-V halted + `CTL_STATUS.QQUIESCENT` is the quiescent state of §4.6;
  held queued ops are allowed.
- **Debug SPM window** (PD2): prof-dbg's rule "not BUSY or halted" becomes: allowed iff (no
  command-buffer run is executing, or that run is halted) and QQUIESCENT and (RV_RUN = 0, or the
  RISC-V is halted and no abstract command is executing). With RV_RUN = 0 and nothing queued this
  is exactly today's rule, so the window works at a tile-core breakpoint, its main use; held queued
  ops do not refuse it. In the other cases it refuses as today (SPM_ERR).
- prof-dbg rules that say BUSY mean "a command-buffer run is active" (host or CALL), except the
  STATUS mirror and "START is ignored", which follow host runs only: a host START during a CALL
  waits for it (§6). For a CALL, `DBG_IA` uses the latched run base (§6) and
  `CTL_STATUS.LAST_RUN_CALL` tells post-mortem tools whose state they read.
- The tile-core debug block (PD2) is otherwise unchanged for command-buffer runs, CALLed ones
  included.

### 10.2 Profiling

- **Completion records** (REC flag per push, or a global enable): 128 bits = t_end [47:0] (core
  cycles), duration [71:48] (start → done, saturating 24 bits = 56 ms at 300 MHz; the 32B output
  head's 3.23 M cycles fit), unit [75:72], seq [103:76], error code [107:104], reserved [127:108];
  `TRACE_MARK` records carry a 32-bit marker and are stamped at dispatch, in push order. 64-entry
  FIFO behind per-unit holding registers (§7.1); never stalls execution: when the FIFO is full,
  records are dropped and counted (`CTL_PERF` 16).
- The runtime drains records into a DRAM log (ST) or forwards them over R2H. `tsr-prof` gains a
  `ctrl` source: a Perfetto track per unit plus a RISC-V track (launches, markers); unit utilisation,
  dependency wait, "MVM idle while ops are queued" (the G metric of prof-dbg §5.11 per kernel).
- `CTL_PERF_*` for totals; Ibex `mcycle` / `minstret` (and mhpmcounters if enabled) for the RISC-V's
  own cost.
- prof-dbg phases: **PD4** (resident programs) becomes the RISC-V debug and profiling phase (CR4);
  its LOOP / JUMP / WAITM items drop. The trace ring (prof-dbg §6.3) is superseded by completion
  records on the queue path; for command buffers its criteria stay.

---

## 11. Roadmap

### 11.1 Phases

| Phase | When | Content | RTL |
|---|---|---|---|
| **CR1** software | can start after M4, beside M5–M7 | `tsr_rv` + Spike checks, queue model (all rules of §4–§6), race detector, random schedules, SDK skeleton (headers, crt0, linker script, toolchain file), ELF loader and `tsr_ctrl_*` on `sim`, `matvec_batch` on sim; the Ibex out-of-context synthesis on the U200 part (area, Fmax, F_RV choice) | none (one OOC run) |
| **CR2** RTL | M8 | vendored Ibex / riscv-dbg, `rtl/core/ctrl/`, `tsr_chk` split, `tsr_issue`, SPM round robin + client 3, `tsr_ctl_csr` (CTL, DMI, CTL_PERF), CDC library, `tb/ctrl`, two-clock bench, co-simulation, Vivado OOC + `report_cdc`, U200 kernels with `ap_clk_2` (direct frequency first), vkernel with two clocks; the RR no-deadlock argument written into the rtl-m2 / rtl-m3 contracts; gate (D13): 300 MHz on the k1, k4t1 and k4 out-of-context runs and the k4 link with `HAS_CTRL = 1`, the link after the host RAM upgrade (§7.4) | yes |
| **CR3** M8 gate | M8 | device MoE and the decode step in C++, issuing tile ops directly, position-dependent addresses computed in C++ (router MVM → TOPK → expert ids from SPM → expert MVMs); the host writes 2 arguments per token; vkernel, then the card. CALL only for position-independent programs; no register-relative lowering mode (D10, §6) | — |
| **CR4** resident loop | M9 | decode loop on the RISC-V, sampling (TOPK + VU softmax + a software PRNG), token ring in DRAM, `sw/runner` as the host side, GDB through `rv-bridge`, records in `tsr-prof` | — |
| **CR5** JIT | after M9 | registry, a tinygrad or torch.compile backend, then PJRT; launch ring if the measured per-launch cost requires it | — |

### 11.2 Milestones

- **M8** becomes "제어 코어 + C++ SDK + 디바이스 MoE" instead of "레지스터 확장"; pass criterion
  unchanged (the host writes only 2 arguments per token).
- **M9**: the decode loop in C++ on the RISC-V, its decode steps issuing tile ops directly as in M8
  (D10); pass criterion unchanged (the host only polls a ring buffer).
- M5–M7 are unchanged (command-buffer path). Option, not a change: if M7's attention needs loops (n_kv
  changes every token), write it as a C++ kernel and pull CR2 forward; decide at the end of M6.

### 11.3 The scalar ISA growth

| Item | Planned | Now | Why |
|---|---|---|---|
| `MULI` 0x04 | M8 | dropped; opcode reserved (`ILLEGAL_OP`) | the RISC-V does the arithmetic |
| `RLD` 0x05 | M8 | dropped | the RISC-V reads the SPM directly |
| `LOOP` 0x06 | M8 | dropped; `LOOP_DEPTH` and error 9 `LOOP_NEST` reserved | loops in C++ |
| `WAITM` 0x07 | M9 | dropped; error 10 `WAIT_TIMEOUT`, STATUS bit3 WAITING and `PERF_WAITM_CYCLES` (index 4 stays, reads 0) reserved | mailbox + RISC-V polling |
| `JUMP` 0x08 | M9 | dropped | RISC-V control flow + CALL |
| CAPS bit2, bit4 | — | reserved, never set, never reused | old tools and traces |
| `TOPK`, `DQROW` | M8, M7 | stay tile ops | data-parallel work belongs to the units |
| `LI`, `ADDI`, R0–R15 | M2 | stay | command buffers, CALL arguments |
| `CALL` 0x09 | — | new, issue path only | §6 |
| `NOP`, `FENCE` on the issue path | — | SYNC unit semantics (§4.4) | join points, barriers |
| errors 11, 12 | — | new, issue path only | §4.5 |

Keep them for command-buffer-only targets? No. A target without the control core runs host-driven
command buffers (M1–M7 style). On an SoC with its own CPU (Zynq's PS) host independence comes from
that CPU driving the same MMIO issue port over AXI-Lite, with the same tickets and queues, so no
second mechanism is needed. The encodings stay reserved, so the decision can be reversed without
conflicts.

---

## 12. Cost

### 12.1 U200 (estimates, not synthesised)

| Block | Domain | LUT | FF | RAMB36 | DSP |
|---|---|---|---|---|---|
| Ibex (small-derived, FPGA register file, 2 triggers) | clk_rv | 2.5–4 K | 1.5–2.5 K | 0 | 0–4 |
| riscv-dbg DM (1 hart, program buffer 8, SBA answered with an error) + DM slave arbiter | clk_rv | 1–2 K | 1–1.5 K | 0 | 0 |
| ITCM 64 KiB + DTCM 64 KiB (both `tsr_sram_1r1w`) | clk_rv | ≈ 0.1 K | — | 32 | 0 |
| Bus decode, control window, mailbox transfer registers, timer | clk_rv | 1.5–2.5 K | 2–3 K | 0 | 0 |
| CDC (issue FIFO, counters, tagged handshakes, record FIFO and holding registers) | both | 0.5–1 K | 1–1.5 K | 0–2 | 0 |
| Dispatcher, 6 queues (≈ 7.5 Kb LUTRAM), dep compares (≈ 150–250 LUT per head, §4.4), SYNC snapshot, CMD | clk | 2–4 K | 1.5–2.5 K | 0 | 0 |
| SPM client 3, round robin, bridge | clk | 0.6–1 K | 0.2–0.4 K | 0 | 0 |
| CTL CSRs, mailbox copies, DMI bridge (1–1.5 K LUT) + `CTL_PERF`: 29 defined counters × 64 bits (≈ 1.9 K LUT of incrementers, ≈ 0.5–0.8 K LUT read mux) | clk | 3.5–5 K | 3–3.5 K | 0 | 0 |
| **Total** | | **≈ 12–20 K** | **≈ 11–17 K** | **32–34** | **0–4** |

- The first revision gave 10–17 K LUT; it had counted 64 `CTL_PERF` counters in 2–3.5 K LUT
  (64 × 64-bit incrementers alone are ≈ 4 K LUT, plus a 1–1.5 K LUT read mux) and ≈ 60 LUT per queue
  head. These figures stay estimates until the CR1 / CR2 synthesis.
- Against k4's core (270,910 LUT) + 4–7 %; against k1's (30,890) + 40–65 %; ≈ 1.2–2 % of the U200's
  dynamic region (979 K LUT); for scale, prof-dbg's debug block is ≈ 1.6 K LUT.
- The checker split adds no logic (a refactor); PERF gating ≈ 10 LUT; the shared dispatch register
  adds a 2:1 select per command bit (≈ 0.3–0.5 K LUT for 528 bits, *estimate*, counted in the
  dispatcher row), not a register.
- Not in the table: the issue-path s bank (D8, §4.7), ≈ 512 FF and a 2:1 read mux (*estimate*),
  built with the vector unit at M6.

### 12.2 ASIC (estimates)

- Ibex small ≈ 24–27 kGE (Ibex README configuration table, *confirm*); DM ≈ 15–30 kGE; control
  window, CDC, dispatcher and CSRs ≈ 30–50 kGE excluding storage.
- **Storage** ≈ 25 Kb: issue FIFO 32 × ≈ 300 b = 9.6 Kb, record FIFO 64 × 128 b = 8 Kb, queues
  ≈ 7.5 Kb. As flops with read muxes that is ≈ 125–175 kGE (5–7 GE per bit), 3–5× Ibex; the FPGA's
  cheap LUTRAM hides it. With 2-port register-file macros, or with an 8-entry issue FIFO (§4.2) and
  a 16-entry record FIFO (≈ 9 Kb), ≈ 45–65 kGE.
- `CTL_PERF` (29 × 64 b) as flops ≈ 20 kGE.
- TCMs: 128 KiB of 1RW / 1R1W SRAM macros ≈ 0.2–0.4 mm² in 28 nm, through `tsr_sram_*` wrappers (no
  2RW macro, which would be ≈ 1.5–2× the area of a 1R1W one).
- Total ≈ 0.4–0.7 mm² in 28 nm next to the 4 MiB SPM (≈ 8–15 mm²): a few percent.
- Optional boot ROM for host-less targets (§3.4). Clock: an own PLL output, or the core clock with
  `CTRL_SYNC` = 1.

### 12.3 Effort (estimates)

- CR1 ≈ 5–7 K lines: interpreter 2 K, queue model 1 K, race detector 0.8 K, SDK 1.5–2 K, transport
  0.6 K.
- CR2 ≈ 3.5–4.5 K lines of SystemVerilog (control subsystem 2.5–3 K, core changes 1–1.5 K) and
  ≈ 3.5 K lines of benches; ≈ 20 K lines vendored (not ours).

---

## 13. Deliberately not done

- No cache, branch prediction, superscalar or out-of-order execution (D1).
- No custom RISC-V instructions: the core stays unmodified; issue is MMIO.
- No direct DRAM access from the RISC-V (§3.6); no instruction fetch from SPM or DRAM.
- No F extension in v1 (§2.4); no atomics, no second hart, no operating system.
- No hardware race detection (the simulator does it).
- No cancellation of a running tile op: QABORT drops entries that have not started.
- No use of R0–R15 by queued ops.
- No overlap between command-buffer runs and datapath queue ops in v1.
- No SPM partition between host lowering and RISC-V kernels: the transport alternates execs and
  launches (D9).
- No register-relative lowering mode in the backend; CALL only for position-independent programs
  (D10).
- No second checker instance for the issue path (D11).
- No line buffer (cache) in the SPM bridge.
- No second issue staging bank (`issue()` is not reentrant, §3.2).
- No epoch field on issue-FIFO entries: after an abort every unit stays in error state until a
  QCLEAR that is ordered with the pushes, which gives the same guarantee (§4.5).
- No change to the llama.cpp path, the encodings of existing instructions, the `PERF_*` indices,
  SOFT_RESET's meaning or error codes 1–8; no `CAPS` bit for the control core.

---

## 14. Review revisions (2026-10-24)

The first revision was reviewed against the sources (spec, rtl-m2, rtl-m3 §14, prof-dbg,
platform-u200, xrt-host, `tsr_spm.sv`, `tsr_cmdproc.sv`, `tsr_dma.sv`, `tsr_mem_arb.sv`,
`tsr_csr.sv`). Every finding was checked; none reopened a user decision. Verdicts:

| Finding | Verdict | Applied in |
|---|---|---|
| Aborted / poisoned entries and FENCE completed without waiting for deps; `join_all()` returned early after any `ILLEGAL_OP` (blocker) | confirmed, fixed: completion rule I1 | §4.4, §4.5, §5.5 2(e), §8.3, §9.3 |
| Error record could arrive after `DONE_SEQ` (reported twice) | confirmed, fixed: I2 (D[u] waits for the record acknowledge) | §4.5, §7.1, §9.3 |
| Queue control out of band; QIDLE racy; abort scope undefined (reported twice) | confirmed, fixed: RISC-V QCTRL through the FIFO, all units in error state until QCLEAR, ticketed QCLEAR, RV-side QIDLE. Rejected part: an epoch counter (the error state until an ordered QCLEAR already covers stale FIFO entries); "every op pushed before the abort completes ABORTED" cannot hold for running ops (no cancel) | §3.2, §4.5, §4.6, §13 |
| Credit wait longer than `ISSUE_WAIT_MAX` in normal operation | confirmed (1.04 ms vs 393 µs), fixed: software credit check, hardware wait as safety net | §4.1, §8.2 |
| START / CALL starvation; CALL + `join_all()` deadlock | confirmed, fixed: drain rule, "running" defined | §6, §5.2, §9.3 |
| PERF gating during the START wait | confirmed, fixed: PERF counts from the actual start | §6, §3.3 |
| SPM window condition dropped "or halted" | confirmed, fixed: QQUIESCENT | §10.1, §4.6 |
| Halted-and-drained state unreachable, not verifiable | confirmed, fixed | §4.6, §9.4 |
| CSR accesses that wait on clk_rv; stale DMI answers; bridge unbounded (reported twice) | confirmed, fixed: core-domain copies, tagged handshakes, timeouts, bridge bound, RV_CLK_OK | §3.3, §3.5, §7.1 |
| Reset sequencing | confirmed, fixed | §7.3 |
| SOFT_RESET semantics changed | confirmed, fixed: today's SOFT_RESET, command-buffer path only | §4.6 |
| s0–s15 outside the race definition | confirmed; proposal: separate issue-path bank (revision 3: decided, D8) | §4.7, §5.5, §15 |
| Host runs vs RISC-V kernels on the SPM | confirmed; proposal (a) serialise exec and launch, (b) partition (revision 3: (a) decided, D9) | §5.5, §6, §8.4 |
| FENCE snapshot in reserved words | confirmed, fixed with a cheaper variant: the dispatcher takes the snapshot | §4.4 |
| Error state at dispatch; first_error order | confirmed, fixed: push index | §4.3, §4.5 |
| Stale tickets after QCLEAR; crt0 | confirmed, fixed | §4.3, §8.3 |
| Staging not reentrant | confirmed, documented; a second staging bank rejected (cost, no use case) | §3.2 |
| Mailbox deadlock; launch rate | confirmed, fixed (ack on receipt, hosts drain R2H); launch ring open | §8.3, §8.4, §15 |
| CALL changes host-visible state | confirmed, listed; IA base, LAST_RUN_CALL, ABORT on a pending START | §6, §10.1 |
| Validity defined by a test | confirmed, fixed | §9.5 |
| Round-robin deadlock argument; DMA head-of-line; DDR penalties (reported twice) | confirmed, argument written, tests and card measurement added; RR built only with `HAS_CTRL = 1` | §5.3, §5.4, §9.3 |
| Cost table optimistic | confirmed, fixed | §12 |
| Kernel ABI; CALL needs register-relative lowering | confirmed, fixed: ABI, SPM argument range, CR3 issues directly (revision 3: no register-relative mode, D10) | §6, §8.3, §8.7, §11 |
| QHOLD level vs pulses | confirmed, fixed: three hold sources. W1S / W1C host bits not added: each `CTL_CTRL` bit has one owner, so a host read-modify-write cannot drop the RISC-V's hold | §3.2, §3.3 |
| Follow-ups missing from the spec text | confirmed, added | spec text, 참고 |
| 300 MHz margin | confirmed, fixed: CR2 gate, shared dispatch register, placement (revision 3: gate decided with k1 added, D13) | §7.4, §11 |
| CDC crossing list incomplete | confirmed, fixed | §7.1, §10.2 |
| Interrupt clear semantics | confirmed, fixed | §3.7 |
| Ibex grant path; Fmax optimistic | plausible (*confirm*), fixed: registered grants, planning F_RV 166.7 MHz | §2.2, §2.3, §7.4 |
| Ibex / riscv-dbg integration details | plausible (*confirm*), fixed | §3.1, §10.1 |
| True dual-port ITCM | confirmed, fixed | §3.1, §8.3, §12 |
| ASIC storage; FIFO depth; host-less boot | confirmed, fixed | §3.4, §4.2, §12.2 |
| Clock and reset on the U200 | confirmed (*confirm* the v++ options), fixed | §7.2–§7.4 |

---

## 15. Open items

The spec's 미결정 사항 records the decisions; the follow-ups are here.

**Decided by the user (2026-10-24)**; the body text was changed to match (revision 3):

- [x] **s-register bank** (was: decide before adopting the memory model) — D8: a separate
  issue-path bank (≈ 512 FF); command-buffer runs keep theirs, so no s-register race between the
  two paths can occur (§4.7, §5.5, §9.2). Rejected: one shared bank in the race definition.
- [x] **Host ownership** — D9: the transport alternates llama.cpp execs and RISC-V launches (an
  exec only when no launch is outstanding, a launch only when BUSY = 0). No SPM split, no hardware
  change (§6, §8.4).
- [x] **CALL of lowered layer programs** — D10: M8 / M9 decode steps are C++ kernels that issue
  tile ops directly and compute position-dependent addresses in C++; CALL only for
  position-independent command-buffer programs; no register-relative lowering mode in the backend
  (§6, §11).
- [x] **Single checker** (was a follow-up) — D11: one checker shared with cmdproc; ops pushed
  during a command-buffer run, and a RISC-V QABORT behind them, wait for the run's end. Accepted
  given D9 (§5.2).
- [x] **RISC-V clock on the U200** — D12: planning value 166.7 MHz; 250 MHz only if the CR1 Ibex
  out-of-context run closes there; 125 MHz fallback (§7.4). How the clock is made is a follow-up
  (below).
- [x] **CR2 timing gate** — D13: freeze the control-core RTL only after the k1, k4t1 and k4
  out-of-context runs and the k4 v++ link meet 300 MHz with `HAS_CTRL = 1`; the k4 link runs after
  the host RAM upgrade the user will do when the card goes in (§7.4, §11.1).

Follow-ups:

- **`ap_clk_2` generation** (CR2 smoke link): whether v++ sets the frequency of `ap_clk_2`
  directly (`--kernel_frequency`, `--clock.*`, *confirm*); BUFGCE_DIV inside the kernel only if it
  does not (§7.4).
- **Exclusivity**: whether command-buffer runs may later overlap queue ops (needs per-run PERF
  attribution and a weaker debug invariant).
- **Stuck units** (spec 미결정 "멈춘 유닛의 강제 해제"): it extends to the queues — with a stuck
  op QIDLE never comes, the RISC-V's QCLEAR never completes and the host's is refused.
- **Launch path**: a DRAM launch ring with a doorbell for CR5, by the measured per-launch cost;
  an R2H-pending interrupt to the host through the adapter (a new IER / ISR bit; adapter offset
  0x01C is reserved); whether a control-core launch should be an `ap_ctrl_hs` run.
- **XRT window during `xrt::run`** (spec 미결정): the control core's load, mailbox and DMI all use
  the window, so run mode depends on that test; otherwise the control core uses `TSR_XRT_CTRL=regs`.
- **DDR penalty of interleaved LD / ST** (§5.4): measure on the card before trusting §5.7 / §8.7.
- **F**: revisit if a profile shows soft-float above 1 % of token time (SOP / VU first, then a
  Zfinx / F core).
- **Zba / Zbb / Zbs**: by SDK code size and cycles (CR1 measurements).
- **TCM sizes** (64 + 64 KiB proposed): fixed by the first SDK build's code and data sizes plus the
  M9 loop's tables.
- **Queue depths** (8 / 4 / 4 / 8 / 2 / 4) and `ISSUE_FIFO_DEPTH` (U200 32, ASIC 8): fixed by
  measured kernels (CR3).
- **SPM arbitration**: round robin as proposed; the DMA head-of-line loss measured on the contention
  bench decides whether the DMA needs burst reservation, `tsr_mem_arb` a response skid, or the DMA
  a bounded write priority.
- **Debug**: transport speed (`remote_bitbang` with a software DTM; a faster DMI transport only if
  measured too slow, CR4); SBA as a second master for non-halting memory reads.
- **Boot ROM** for host-less targets (`HAS_BOOTROM`).
- **SPM bridge line buffer** invalidated at joins: only if SPM-window bulk reads become a bottleneck.
- **Licenses**: Ibex (Apache-2.0) and riscv-dbg (SHL-0.51) with the project's planned RTL license
  (Solderpad or CERN-OHL-S).
- **M7 attention** as a C++ kernel (pulls CR2 forward): decide at the end of M6.
- **Measurements on the card**: PCIe posted-write and read latency (they set the image-load time,
  the per-launch cost and the host-control cost of §5.7).

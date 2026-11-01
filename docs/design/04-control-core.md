# Tesseract, area CC — RISC-V control core, issue path, dispatcher, CDC, floorplan

Status: **design contract for M6, not implemented** (2026-10-30). Area CC of the design set
(`docs/design/`): EX execution model, VU vector unit, AT attention / KV, CC control core (this file),
DB debug, SW software and verification. It is the contract for the RTL parts of `docs/ctrl-riscv.md`
(§2–§7, §9.3, §11–§12), following the user decisions of 2026-10-30 (`docs/ctrl-riscv.md`, D14–D18): the
host does not issue tile ops (D14), the host has no debug access to the issue port (D15), the
design has full debug (D18). It is also designed around D1–D6 and D12 (`docs/ctrl-riscv.md`, user decisions of 2026-10-24).
Source-of-truth order: `docs/spec.md` > `docs/design/*` > `docs/ctrl-riscv.md`. Where this file and
`docs/ctrl-riscv.md` disagree, this file wins for M6. Every number marked *estimate* is not
measured; every Ibex, riscv-dbg or Vitis fact marked *confirm* is recalled from their documentation
and is checked when the dependency is vendored or at the first smoke link.

---

## 1. Scope

Area CC owns, in RTL: the RISC-V control subsystem (Ibex, debug module instance, TCMs, buses,
control window), the issue path (issue FIFO, dispatcher, checker, per-unit queues, queue heads,
SYNC unit, tickets, error states), the SPM arbitration and banking, every clock-domain crossing,
resets, interrupts and timer, the host CSR block (`tsr_csr`), the platform adapter's control and
clock changes, the U200 floorplan and timing plan, and the RTL verification of all of it.

| Item | Design |
|---|---|
| Who issues tile ops | the RISC-V only, through the issue port (D14) |
| Tile-op checks | `tsr_chk` in front of the dispatcher (§8) |
| s registers | one bank s0–s15, inside the VU, read by the host at `SREG` 0x9C0 and by the RISC-V through the core window (§5.3, EX §5.1) |
| Host ↔ RISC-V messages | **launch queue LQ and completion ring LCQ** (EX §4) |
| Host CSR map | **EX §5 is the only map**; CC's registers sit inside it (§6) |
| `SOFT_RESET` | **device-level reset sequence** (EX §8.5), run by CC's core-domain SR sequencer without `clk_rv` (§12.2) |
| Unit ids | 0 DMA, 1 QACT, 2 MVM, 3 VU, **4 AT**, 5 SYNC (CC-6) |
| Opcode → unit | static opcode → unit table, `ISSUE_UNITS` mask (CC-7) |
| SPM | **2R2W even / odd line banks**, per-bank client ports, round robin (§10, CC-12) |
| DMA LD flow control | **burst reservation**, `rd_rsp_ready` constant 1 (CC-13) |
| Host debug SPM window | **none**; the debugger reaches the SPM through the DM's system bus (CC-16) |
| Platform control | **`ap_ctrl_none`** (no KDS start / done), `xrt::ip` and polling (EX §9, CC-19) |
| TCMs | **128 KiB ITCM + 64 KiB DTCM** (CC-3) |
| Timing gate (D13) | k4 link kernel clock **≥ 250 MHz** with all M6 hardware (user decision, 2026-11-02); current request 300 MHz (§14.3) |
| Core boundary | `host_irq_o` and `plat_err_i` (EX §9.3; §13.1 here) |

Not in this area: the meaning of each tile op (EX, VU, AT), the memory model as an architectural
rule (EX; §9 implements it), the host CSR map, the launch protocol (LQ / LCQ) and `SOFT_RESET`
semantics (EX), the kernel ABI (SW), GDB, completion-record formats and tile-level debug semantics
(DB), the ISA simulator's RV32 model, queue model, race detector and trace converter (SW). CC
implements in RTL what EX and DB specify for the core.

---

## 2. Structure

### 2.1 Block diagram

```
 clk_rv domain (SLR1 pblock "ctrl_rv")              ¦  core clock domain (ap_clk)
                                                    ¦
 ┌────────┐ instr ┌────────────┐                    ¦
 │  Ibex  │──────►│ ITCM 128K  │◄─► load port ◄─────¦── LOAD_* holding reg + VERIFY reader (tsr_csr)
 │RV32IMC │       └────────────┘                    ¦
 │        │ data ┬─► DTCM 64K  ◄─► load port        ¦      ┌────────────────── tsr_issue ─────────────────┐
 └──▲─────┘      ├─► control window ──issue FIFO────¦─────►│ dispatcher ─► tsr_chk ─► 6 queues ─► heads ──┼─► DMA, QACT, MVM, VU (AT ops
  dbg│  ┌────┐   │     (tsr_ctl_win)  ◄─ gray D/S ──¦◄─────│ D[u], S[u] counters, error states, records   │   on units 0/2/3; unit 4 absent)
     └──│ DM │◄──┼─── DMI hs (tagged) ◄─────────────¦◄──── tsr_csr DMI window (0xC00) ◄── host CSR port  (cmd_valid/ready/done)
        │    │SBA│                                  ¦      └──────────────────────────────────────────────┘
        └────┘─┬─┤─► SPM window / LB alias ── hs ───¦─────► tsr_spm_bridge ─► tsr_spm client 3 (2R2W even/odd)
               │ ├─► core window (read-only) ─ hs ──¦─────► tsr_csr read port 2 (ID, CFG, SREG 0x9C0, PERF, Q_*, ...)
               │ ├─► LQ / LCQ regs ─── tagged hs ───¦─────► LQ ring, LCQ ring (core domain, EX §4)
               │ └─► LOG_PUSH ── log CDC FIFO ──────¦─────► log ring (256 words) ─► host 0x400–0x7FC
               └ MTIME, irq levels ◄── gray / sync ─¦◄──── core cycle counter, LQ_AVAIL / LCQ_FREE levels, SR sequencer
```

### 2.2 Hierarchy, domains, placement

| Module (new = N, changed = C) | File | Domain | SLR (U200 k4) |
|---|---|---|---|
| `tsr_core` (C: new boundary, §13.1) | `rtl/core/tsr_core.sv` | both | — |
| `tsr_ctrl_rv` (N): Ibex, `dm_top`, TCMs, `tsr_rv_bus`, `tsr_ctl_win` (with the line buffer) | `rtl/ctrl/` (W4: not `rtl/core/ctrl/`) | clk_rv | SLR1, pblock `ctrl_rv` |
| `tsr_ctrl_cdc` (N): every crossing of §11; `tsr_ctrl` = `tsr_ctrl_rv` + `tsr_ctrl_cdc` + reset synchronisers, the instance `tsr_core` takes | `rtl/ctrl/tsr_ctrl_cdc.sv`, `rtl/ctrl/tsr_ctrl.sv` | both | SLR1 |
| `tsr_core_dp` (N): the whole core-clock datapath; the op-list replay bench instantiates it alone (§15.3) | `rtl/core/tsr_core_dp.sv` | core | — |
| `tsr_issue` (N): dispatcher, queues, heads, SYNC, counters, error states | `rtl/core/tsr_issue.sv` | core | SLR1 |
| `tsr_chk` (N) | `rtl/core/tsr_chk.sv` | core | SLR1 |
| `tsr_spm_bridge` (N): RISC-V SPM client | `rtl/core/tsr_spm_bridge.sv` | core | SLR1, next to `tsr_spm` |
| `tsr_csr` (C, rewritten) | `rtl/core/tsr_csr.sv` | core | SLR1 |
| `tsr_spm` (C: 2R2W), `tsr_dma` (C), `tsr_qact` (C: port adapter), `tsr_mvm` (C: port adapter, gather slices) | `rtl/core/` | core | SLR1 (clusters SLR0 / SLR2) |
| `tsr_cdc_fifo`, `tsr_cdc_gray`, `tsr_cdc_hs` (N); `tsr_sync`, `tsr_rst_sync` until `rtl/tech` takes them | `rtl/ctrl/cdc/` (W4: not `rtl/core/cdc/`) | both | — |
| `tsr_sync`, `tsr_rst_sync`, `tsr_clk_div` (N) | `rtl/tech/<target>/` | — | — |
| VU (`tsr_vu`, including AT's vector ops 0x34–0x37); AT's cluster modes (ATTQK / ATTPV) and KVST in the DMA | areas VU / AT | core | VU and KVST in SLR1; cluster modes in SLR0 / SLR2 (§14) |

---

## 3. Control core: Ibex and the debug module

### 3.1 Pinning

- **Ibex**: `lowRISC/ibex`, vendored unmodified into `third_party/ibex` at the commit that
  OpenTitan's `hw/vendor/lowrisc_ibex.lock.hjson` records on the vendoring day (a commit that
  OpenTitan's sign-off flow has run), together with the `vendor/lowrisc_ip` `prim_*` files that
  Ibex itself vendors. If that commit is older than 6 months, the newest `master` commit with green
  Ibex CI instead. Hash, date, the Verilator 5.046 build check and the configuration's verification
  grade (Ibex README table) go to `rtl/ctrl/VENDOR.md` (W4: a file in the submodule would modify
  it; pinned: Ibex `7a4697f5`, 2026-10-16).
- **Debug module**: `pulp-platform/riscv-dbg` (Debug Spec 0.13.2), latest release tag on the
  vendoring day (v0.10.1, `deed17e1`, on 2026-10-31), unmodified, in `third_party/riscv-dbg`,
  with only the `common_cells` file `dm_top` needs (`fifo_v3`, from the `third_party/common_cells`
  submodule at v1.40.0). The JTAG DTM (`dmi_jtag`) is not used: our DMI
  arrives from the CSR window (§11).
- Both keep their own style (asynchronous active-low resets, `prim_*`), with one Verilator waiver
  file per tree; `prim_clock_gating` binds to `rtl/tech/<target>/tsr_clk_gate.sv` (pass-through on
  the FPGA). Licences (Apache-2.0, SHL-0.51) are recorded beside them.

### 3.2 Configuration (`ibex_top` parameters, names *confirm* on the pinned commit)

| Parameter | Value | Why |
|---|---|---|
| `RV32E` | 0 | 16 registers would cost code size |
| `RV32M` | `RV32MFast` (T1, 2026-10-31, §14.3: maxperf with `RV32MSingleCycle` had +0.20 ns at 6.0 ns, under the 0.3 ns rule) | MUL 3 cycles, MULH* 4, DIV / REM 38 (2 by zero) (CC-34). `tsr_rv`'s cycle table still models the single-cycle multiplier (MUL 1, MULH* 2): ≤ +1.3 % RISC-V cycles on the W2a firmware, 0 on the optimised executor (below), so the simulator's SW §7.8 projections stay within that bound |
| `RV32B` | `RV32BNone` | code-size gain unmeasured; Zba/Zbb stays an SDK-measured option |
| `RegFile` | `RegFileFPGA` (LUTRAM); flops on an ASIC (tech layer) | |
| `WritebackStage`, `BranchTargetALU` | 1 / 1 | Ibex's "maxperf" pipeline (CC-34): loads and stores leave ID/EX after the request, taken branches 2 cycles instead of 3; still in order, no prediction, no cache (D1) |
| `ICache`, `ICacheECC`, `BranchPredictor` | 0 | D1 |
| `PMPEnable` | 0 | machine mode only |
| `MHPMCounterNum` | 0 | `mcycle` / `minstret` suffice; MTIME for timestamps |
| `DbgTriggerEn` / `DbgHwBreakNum` | 1 / 2 | two hardware breakpoints (D18) |
| `SecureIbex`, `ICacheScramble` | 0 | |
| `DmHaltAddr` / `DmExceptionAddr` | 0x1000_0800 / 0x1000_0808 | riscv-dbg layout at base 0x1000_0000 (*confirm*) |
| `hart_id_i` | 0 | one hart |
| `boot_addr_i` | `RV_BOOT` (EX §5.1, 256-byte aligned) | first fetch at `RV_BOOT` + 0x80, vectored `mtvec` there (*confirm*) |
| `fetch_enable_i` | constant on | the run / stop control is the reset (§12) |

ISA: **RV32IMC + Zicsr + Zifencei**, machine mode only, no A, no F/D (`docs/ctrl-riscv.md` §2.4
keeps its argument: spec-fp scalars go through the `tsr_ref` helpers in soft-float).

**Pipeline (CC-34).** The pipeline is Ibex's "maxperf" one (Ibex `README.md` configuration
table: RV32IMC with a 1-cycle multiplier, branch target ALU and writeback stage, 3.13 CoreMark/MHz
and 32.48 kGE against "small" at 2.47 and 26.60 kGE, both verification status green;
`ibex_configs.yaml`), with the other parameters of this table. "maxperf" itself has
`DbgTriggerEn` 0 and `RegFileFF`, and in current Ibex `RV32ZC` = Zca + Zcb + Zcmp; we keep two
triggers, LUTRAM and, if the pinned commit has `RV32ZC`, Zca only (the ISA stays RV32IMC; Zcb /
Zcmp are an SDK-measured option like Zba / Zbb). Cycle costs (Ibex
`doc/03_reference/pipeline_details.rst` "Multi- and Single-Cycle Instructions" and
`rtl/ibex_id_stage.sv`; `tsr_rv` models both configurations, SW §4.2):

| Class | maxperf | small | Source |
|---|---|---|---|
| ALU, LUI / AUIPC, fence, not-taken branch, CSR read | 1 | 1 | 0 stall cycles |
| CSR write other than `mscratch` / `mepc` | 2 | 2 | pipeline flush (*confirm* 1 cycle) |
| taken branch | 2 | 3 | "2 – N; 1 – N with BranchTargetALU" |
| JAL, JALR, `fence.i` | 2 | 2 | "Jump 1 – N"; `fence.i` is a jump |
| load / store, TCM | 1 | 2 | WB stage: ID/EX does not wait for the response (`outstanding_memory_access`, `rtl/ibex_id_stage.sv` `stall_mem`; derived from the RTL, not stated in Ibex's documentation, which says loads and stores stall at least one cycle and leaves the writeback stage's effect "not yet documented": *confirm* with the RVFI bench, P5.2, and T1) |
| load whose rd the next instruction reads | + 1 | — | `stall_ld_hz`: load data is not forwarded from WB |
| load / store, window (§5, §10.5) | latency | 1 + latency | the next instruction waits for the response |
| MUL / MULH* | 1 / 2 | 3 / 4 | single-cycle 0 / 1, fast 2 / 3 stall cycles |
| DIV*, REM* (by zero) | 38 (2) | 38 (2) | "1 or 37" stall cycles |
| jump / taken branch to a 32-bit instruction at pc[1] = 1 | + 1 | + 1 | second word fetch (*confirm*) |

With the writeback stage Ibex issues the next data request in the cycle the previous response
returns, so the data bus (§4.3) must take a TCM access every cycle: `outstanding_q` holds `data_gnt`
low only behind an access outside the TCMs (TCM responses always come in the next cycle). This row
and §4.3's one-TCM-access-per-cycle rule carry most of maxperf's measured gain (the writeback stage
≈ 300 of ≈ 430 cycles per W2a record, below); both rest on reading `ibex_id_stage.sv`, so they stay
*confirm* (RVFI bench P5.2, T1), like the CSR flush, 32-bit target fetch and load-use items.

*Measured* (simulator timing model, fixed schedule, `fw_m6_perf` on 0.5B decode MUL_MAT op lists,
`-Os`): with the W2a firmware (92f0b17) small costs 1,977 RISC-V cycles per op record and maxperf
1,547 (−21.8 %): the writeback stage saves ≈ 300 cycles (TCM loads and stores, net of 51
load-use stalls), the branch target ALU ≈ 113 (taken branches), the single-cycle multiplier ≈ 23
(maxperf with `RV32MFast`: 1,567, +1.3 %; on the optimised executor below it makes no difference,
which has no multiplications left on its per-op path). Descriptor push → kernel entry 2.19 → 1.70 µs, empty
launch 5.50 → 4.26 µs. With the optimised executor of the executor-optimisation workflow (final
tree, hot path `-O2`, SW §6.1) small costs 520 and maxperf 437 cycles per MUL_MAT op record
(−16 %), a JOIN record 163 → 140, a NOP record 99 → 85; one 0.5B decode layer costs the RISC-V
48.3 → 39.4 µs; push → kernel entry 2.19 → 1.68 µs, empty launch 5.50 → 4.23 µs.

**Fallback (decided now, applied by T1, §14.3).** T1 runs `tsr_ctrl_rv` out of context in both
configurations. If maxperf makes a lower clock class (250 / 166.7 / 125 MHz with ≥ 0.3 ns margin)
than small and the failing paths are the multiplier's, `RV32MFast` with the writeback stage and the
branch target ALU is taken (+1.3 % cycles on the W2a firmware, none on the optimised executor). If it still lands a class lower, the configuration
with the lower time per op record (cycles / F_RV) is taken: with the W2a firmware small at 250 MHz
(7.9 µs) beats maxperf at 166.7 MHz (9.3 µs), and maxperf at 166.7 MHz beats small at 125 MHz
(15.8 µs).

`dm_top` configuration: `NrHarts` 1, `BusWidth` 32, `DmBaseAddress` 0x1000_0000 (which needs its
two-scratch debug ROM, `HasSndScratch`, *confirm*), `SelectableHarts` 1; its slave port on both
Ibex ports (2:1, data first, the fetch stalls through `instr_gnt`); its **system-bus master (SBA)
connected** to the RISC-V data bus (§4.3, CC-16); `ndmreset_o` starts the RISC-V reset
sequence (§12.2).

**Halted status.** `dm_top` does not export a per-hart halted level (*confirm*). `RV_HALTED` is
derived in hardware: the address of the last granted instruction fetch lies in the DM range
0x1000_0000–0x1000_0FFF (debug ROM park loop, program buffer and abstract-command buffer all
execute there, *confirm*). It feeds `STATUS.RV_HALTED` (EX §5.3), the debug-halt hold source
(§9.6) and `QQUIESCENT`. `tb/ctrl` cross-checks it against `dmstatus.allhalted`; if the vendored
riscv-dbg runs anything outside that range, the fallback reads `dmstatus` over the DMI.

### 3.3 Clock and performance

- `clk_rv` planning value **166.7 MHz** (D12; 250 MHz only if the out-of-context run of
  `tsr_ctrl_rv` closes there with ≥ 0.3 ns margin; 125 MHz fallback).
- Effect of 250 MHz (*measured*, simulator at `TSR_SIM_RV_RATIO` = 250 / 276.3, where the core
  window and the SPM bridge keep their core-cycle part, SW §4.2): the cycles per op record barely
  move (the per-op path has few crossing accesses), so the RISC-V time scales by ≈ 0.67. W2a
  firmware, maxperf, `-Os`: one 0.5B decode layer costs the RISC-V 110.4 µs at 166.7 MHz and
  72.3 µs at 250 MHz, the MVM's issue-bound gaps fall from 8.1 % to 2.5 % of its span, the empty
  launch from 4.26 to 2.84 µs. With the optimised executor (final tree of the
  executor-optimisation workflow, maxperf, hot path `-O2`, SW §6.1) the MVM no longer idles on
  issue at any of the three clocks (issue-bound gaps 0.0 %): the layer's RISC-V time is 50.6 /
  39.4 / 26.3 µs at 125 / 166.7 / 250 MHz against 197 µs of MVM work, and launch push → DONE is
  224.0 / 220.2 / 216.6 µs (MVM-bound; 250 MHz gains 1.6 %); the empty launch takes 5.64 / 4.23 /
  2.82 µs. M7 projection (SW §7.8: the layer plus 25 short VU records through a generic test
  kind, U200 queue depths): the RISC-V's speed costs 12.7 / 6.2 / 1.6 % of the layer's time at
  125 / 166.7 / 250 MHz against a near-ideal issuer (small: 19.8 / 11.2 / 4.1 %), because a
  dependent chain of short records outruns an issuer that can bank only ≈ 4 of them in the DMA
  queue (depth 8, §7.4); W2b's VU kinds take the MUL_MAT fast path's form to close that. 250 MHz is
  headroom for M7 and for launch latency, not a requirement: the planning value stays 166.7 MHz.
- Throughput need (*estimate*): a Qwen2.5-0.5B decode token in one launch is ≈ 1.5 K tile ops
  (24 layers × ≈ 60), 32B ≈ 4 K; issue cost per op 15–40 RISC-V cycles (§7.6) → 0.5B ≈
  0.15–0.36 ms of RISC-V time per token against ≈ 8.25 ms of device time at 276 MHz
  (`docs/platform-u200.md` §14.1). *Measured* (SW §7.8, maxperf, 166.7 MHz): M6's 0.5B decode layer
  is 9 op records and ≈ 35 tile ops; the executor works 3,636 RISC-V cycles = 21.8 µs per layer
  (≈ 100 cycles per pushed op including the executor and MUL_MAT's planning) against 239.6 µs of
  device time, ≈ 0.52 ms per 24-layer token, all of it overlapped (the MVM never idles on issue).
  The survey's ≈ 18 K per-head attention ops per 32B token do not occur: VU's row × stride operand
  model (VU-1) and AT's group-range ops (AT §9.3: ≈ 0.8 K attention ops per 32B token at 4K
  context) cover all heads per op.

---

## 4. RISC-V address map, TCMs and buses

### 4.1 Address map (32-bit byte addresses; everything not listed is an access fault)

| Range | Region | Fetch | Load / store | Notes |
|---|---|---|---|---|
| 0x0000_0000–0x0001_FFFF | ITCM, `ITCM_BYTES` = 128 KiB | yes | byte / half / word | data reads steal the fetch port for one cycle (`instr_gnt` = 0); stores then `fence.i` |
| 0x0010_0000–0x0010_FFFF | DTCM, `DTCM_BYTES` = 64 KiB | no | byte / half / word | .rodata, .data, .bss, stack |
| 0x1000_0000–0x1000_0FFF | debug module memory | yes | word (byte / half too in debug mode) | riscv-dbg layout; its debug ROM reads its flags with `lbu` (W4 correction) |
| 0x2000_0000–0x2000_0FFF | control window (§5) | no | word only | clk_rv registers |
| 0x2000_1000–0x2000_1FFF | core window (§5.3) | no | word loads only (store = fault) | read-only view of the core CSR space |
| 0x4000_0000–0x403F_FFFF | SPM window, exact (`SPM_BYTES`) | no | byte / half / word | 0x4000_0000 + 64 × line + offset, little-endian |
| 0x5000_0000–0x503F_FFFF | SPM window, **line-buffered alias** (§10.5) | no | byte / half / word | loads may hit the line buffer; stores write through and invalidate it |

Region sizes are parameters; every address in a region beyond its size faults. The ITCM range is
reserved up to 0x000F_FFFF and the DTCM up to 0x001F_FFFF, so larger TCMs keep the map.

### 4.2 TCMs

- **Size**: ITCM 128 KiB, DTCM 64 KiB (CC-3). `RV_CFG` (EX §5.1) reports them.
- **Implementation**: `tsr_sram_1r1w` with byte enables. Read port = fetch, data-port read, or the
  load port (EX's `LOAD_DATA` read and the VERIFY reader) while `RV_RUN` = 0; write port =
  data-port stores (Ibex or SBA) while `RV_RUN` = 1, or the host load port while `RV_RUN` = 0, never
  both. Read latency 1, no output register (Ibex wants the word the cycle after `gnt`). The
  Xilinx mapping target is RAMB36 in 32K × 1 columns (no output mux): ITCM 32 RAMB36, DTCM 16
  RAMB36 (*confirm* the inference in the first out-of-context run; the wrapper may instantiate
  `RAMB36E2` only inside `rtl/tech/xilinx`). W4 T1 runs: Vivado's default inference builds 8-high /
  4-high cascades (fetch address on the BRAM enables, cascade mux) and `rtl/tech/xilinx`'s
  wrapper maps both TCMs to URAM; `RAM_STYLE BLOCK` + `CASCADE_HEIGHT 1`
  (`rtl/ctrl/viv/tsr_ctrl_tcm.xdc`) gives the 32K × 1 columns, 48 RAMB36.
- Contents are undefined after a hardware reset; the host loads and verifies the image (EX §3.3,
  §6.4 here).

### 4.3 Bus rules

- Instruction port: decode ITCM / DM; `instr_gnt` from a register (`docs/ctrl-riscv.md` §2.3
  rule: never from the address decode).
- Data port: one access outstanding; `data_gnt` = ¬`outstanding_q`; every wait (credit, bridge,
  core window, DM arbitration) is a late `rvalid`. Errors return `data_err` (load / store access
  fault). With Ibex's writeback stage (§3.2, CC-34) the next request can come in the cycle the
  previous response returns: a TCM access (fixed next-cycle response) does not set `outstanding_q`,
  so the TCMs take one access per cycle; only accesses outside the TCMs hold `data_gnt` low until
  their `rvalid` (*confirm*: derived from Ibex's RTL, checked by the RVFI bench P5.2 and T1; the
  `tsr_rv` maxperf table assumes it, §3.2).
- **Two masters** on the data side: Ibex's data port and the DM's SBA master, round robin per
  access. SBA restrictions (D15, CC-33): **every SBA write to the control window
  0x2000_0000–0x2000_0FFF** (issue staging and push, `QCTRL`, `TRACE_MARK`, `LQ_POP`, `LCQ_POST`,
  `LOG_PUSH` and every other register there) and to the core window answers with an error and does
  nothing; SBA writes to ITCM / DTCM while `RV_RUN` = 0 answer with an error (the load port owns the
  write port then; the debugger uses EX's `LOAD_*`); SBA accesses to the DM range answer with an
  error. Everything else (TCM and SPM reads and writes, control-window and core-window reads)
  works for SBA, so the debugger can read memory without halting the hart.

---

## 5. Control window (RISC-V side, offsets from 0x2000_0000)

Word accesses only (a byte / half access faults). Reads have no side effects. The table is
generated from the single table source (EX-30, §13.4) for the RTL, the simulator and the SDK.
Owners: CC (§5.1, §5.2 timer), EX (0x200–0x2FC, EX §4.5), DB (0x160–0x17C and 0x400–0x7FC,
DB §10.2).

### 5.1 Issue port, tickets, queue control

| Offset | Name | Access | Content (reset value) |
|---|---|---|---|
| 0x000–0x01C | `ISSUE_W0`–`W7` | RW | staged instruction words; kept after a push (0) |
| 0x020 | `ISSUE_DEP0` | RW | dependency ticket; unit field 15 = none. Reset to 0xF000_0000 by every push |
| 0x024 | `ISSUE_DEP1` | RW | same |
| 0x028 | `ISSUE_PUSH` | W | push. wdata [15:8] WMASK (word i with bit i clear is sent as 0; w0 is always sent), bit0 REC (completion record wanted, DB §8.2), other bits ignored. A push without a credit completes late (safety net, §7.3) |
| 0x02C | `ISSUE_TICKET` | R | ticket of the last push or `QCTRL.QCLEAR` |
| 0x030 | `ISSUE_CREDIT` | R | unit u at [4u+3:4u], u = 0..5: min(free slots of queue u, free issue-FIFO entries, 15); [31:24] 0 |
| 0x034 | `ISSUE_UNITS` | R | bit u = unit u is built (constant) |
| 0x038 | `ISSUE_STATUS` | R/W1C | bit0 PUSH_TIMEOUT, bit1 ERR_IRQ (sticky, set on the rising edge of "some unit in error state") |
| 0x03C | `ISSUE_PIDX` | R | 32-bit push index that the next ticketed entry will receive. Counted in `clk_rv` when an entry is written into the issue FIFO: +1 per `ISSUE_PUSH` and per `QCTRL.QCLEAR`; QABORT, QHOLD and `TRACE_MARK` entries and a push dropped by a RISC-V reset do not advance it. It equals the dispatcher's pidx for that entry (§8.3, asserted). Reset only by a hardware reset. Not timing-dependent: only the RISC-V's own pushes change it (CC-32) |
| 0x040 + 4u | `DONE_SEQ[u]` | R | [27:0] completions of unit u (synchronised copy; may lag, never leads); [31:28] 0 |
| 0x080 + 4u | `ISSUED_SEQ[u]` | R | [27:0] sequence the next push to unit u gets |
| 0x0C0 + 16u | `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0` of unit u | R | first error since the last clear; `ERR_CODE` 0 = none; visible before `DONE_SEQ[u]` passes `ERR_SEQ` (I2, §9.4) |
| 0x140 | `QCTRL` | W | one bit per write, in push order through the issue FIFO: bit0 QABORT, bit1 QHOLD set, bit2 QHOLD clear, bit3 QCLEAR (gets a SYNC ticket) |
| 0x144 | `QSTATUS` | R | bit0 QIDLE (`ISSUED_SEQ` = `DONE_SEQ` for every u and the issue FIFO empty), bit1 hold in effect, bit2 host hold, bit3 RISC-V hold, bit4 debug-halt hold, [13:8] error state of unit u at bit 8+u, [21:16] queue u non-empty |
| 0x148 | `WAIT_TICKET` | RW | `irq_fast[0]` pending while this ticket is done; unit field 15 disarms (reset 0xF000_0000) |
| 0x14C | `WAIT_CREDIT` | RW | [2:0] unit, bit31 enable: `irq_fast[5]` pending while that unit's credit ≠ 0 (reset 0) |
| 0x160–0x17C | `REC_COUNT`, `REC_DATA0`–`3`, `REC_POP`, `TRACE_MARK`, `REC_CFG` | | DB's completion records and trace marks (DB §8.2, §8.4, §10.2) |
| 0x180 + 4u | `ERR_PIDX[u]` | R | push index of unit u's first error |
| 0x1C4 | `SPM_LB_INV` | W | invalidate the SPM line buffer (§10.5) |

`LOG_PUSH` is EX's, at 0x280 (EX §4.5); 0x1C0 is reserved.

### 5.2 Launch queue, completion ring, log, timer

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x200–0x294 | `LQ_HEAD_W0`–`W7`, `LQ_AVAIL`, `LQ_HEAD_SEQ`, `LQ_POP`, `LQ_CANCEL`, `LQ_STOP`, `LQ_FLAGS`, `LCQ_W0`–`W7`, `LCQ_POST`, `LCQ_FREE`, `LOG_PUSH`, `LOG_FREE`, `LOG_CFG`, `RT_STATE`, `RT_INFO` | | EX §4.5 (normative). CC implements them as EX §4.1 requires (§11.2, CC-30): reads of the LQ / LCQ registers are tagged handshakes into the core domain; `LQ_POP` and `LCQ_POST` commit in the core domain and fault there; `LCQ_W0`–`W7` and `RT_*` are `clk_rv` registers (`RT_*` mirrored to the host by a handshake on write); `LOG_PUSH` goes through the log CDC FIFO; `LOG_FREE` = `LOG_DEPTH` − (LOG_WR_rv − LOG_RD_snap) |
| 0x300 / 0x304 | `MTIME_LO` / `HI` | R | core clock cycles since reset (gray-synchronised) |
| 0x308 / 0x30C | `MTIMECMP_LO` / `HI` | RW | `irq_timer` while MTIME ≥ MTIMECMP (reset all ones) |
| 0x400–0x7FC | `DBG_FLAGS` and reserved | | DB (DB §10.2) |

### 5.3 Core window (0x2000_1000 + CSR offset, word loads)

The core window is a read-only view of the whole host CSR space of EX §5 (same offsets): `ID`,
`CFG*`, `CAPS`, `STATUS`, `SREG0`–`15` (0x9C0), PERF, `Q_*`, the LCQ and log rings. A load is a
tagged handshake into `tsr_csr`'s second read requester (§6.6), bounded like the bridge (4,096
clk_rv cycles, then a load fault and `BRIDGE_TIMEOUT`). RISC-V reads never change the host's PERF
LO/HI snapshot. Core-window loads of `LOAD_DATA` (0x0C4) and of the DMI window (0xC00–0xDFC) return
0 without accessing the loader or the debug module, so the RISC-V can neither disturb a load nor
race the host's debugger. Latency ≈ 50–70 ns (*estimate*). Typical use: s registers after a RED /
SOP join, PERF for in-kernel profiling, the debug-halt state.

---

## 6. Host CSR map (core CSR port; U200 `s_axi_control` 0x1000 + offset)

### 6.1 Map

EX §5 is the only host CSR map (EX-11). CC implements the following parts of it in `tsr_csr` and the
control subsystem:

| EX offset | Registers | CC implementation |
|---|---|---|
| 0x000–0x01C | `ID` (0x5453_5231 "TSR1"), `VERSION` 0x0000_0100 (Tesseract Core capability 1.0), `CFG*`, `CAPS`, `CFG3`, `CTL_ID` | constants from the table source |
| 0x020 / 0x024 | `CTRL`, `STATUS` | level and pulse bits as EX §5.2–§5.3; `RV_RUN` drives the RISC-V reset sequence (§12.2); `SOFT_RESET` starts the SR sequencer (§12.2) |
| 0x028–0x03C | `RV_BOOT`, `RV_FREQ`, `RV_CFG`, `WDOG_CFG`, `RT_STATE`, `RT_INFO` | `RV_FREQ` = clk_rv cycles in the last 2^16 core cycles (16-bit gray counter); `RT_*` are core-domain mirrors of the RISC-V's registers |
| 0x040–0x09C | LQ, LCQ, log ring control, `PERF_INFO`, `Q_DEPTH` (0x088) | core-domain rings and counters (EX §4.1–§4.2) |
| 0x0C0–0x0D4 | `LOAD_*` | holding register + tagged handshake into the TCM write port; VERIFY reader on the TCM read port (§4.2, §6.4) |
| 0x100–0x2FC | `PERF_0`–`63` | EX §6 (§6.3 here) |
| 0x300–0x7FC | LCQ ring, log ring | 8 × 8 words of LUTRAM; 256 × 32 b ring (LUTRAM or one RAMB18) |
| 0x800–0xAFC | `DBG_*` (DB), `SREG0`–`15` at 0x9C0 | DB's block; `SREG` wired from the VU's s bank, present whenever the VU is built |
| 0xC00–0xDFC | DMI | tagged 2-phase handshake to the DM, gated by `DBG_CFG.DMI_EN` (DB §3.2) |
| 0xE00–0xF1C | `Q_*` | core-domain originals of `DONE_SEQ`, `ISSUED_SEQ` (synchronised copy), head state, error records and push indices |

Rules of `docs/ctrl-riscv.md` §3.3: no read has a side effect (except the PERF
LO snapshot, and EX's `LOAD_DATA` and DMI rules); every host access completes without
clk_rv (core-domain copies; accesses that must reach clk_rv end after 4,096 core cycles with a
sticky bit); unmapped offsets read 0.

### 6.2 What the host can and cannot do

The host loads and verifies the image, starts and stops the RISC-V, launches kernels through the
LQ, holds the queue heads, aborts and clears the queues, soft-resets the device, reads every
counter, record and the log, and drives the DM through DMI. It **cannot push a tile op** (D15): no
CSR reaches the issue staging registers, and SBA writes to the whole control window are refused
(§4.3).

### 6.3 PERF counters

EX §6 is normative: one counter space, time counters gated by ACTIVE ∧ ¬QHALT, event counters by
ACTIVE, zeroed by `SOFT_RESET`, per-channel slots at 5 + c and 5 + C + c, extension base X = 5 +
2C. CC implements every counter and adds two extension slots: X + 20 `SPM_RD_WAIT` and X + 21
`SPM_WR_WAIT` (client-cycles a raised request waited for its bank port, summed over clients).
Counters kept in clk_rv (`PUSH_STALL`, `BRIDGE_WAIT`) are 32-bit gray counters gated by a
synchronised ACTIVE and extended to 64 bits in the core domain.

### 6.4 Image load and boot (host sequence)

EX §3.3–§3.5 are normative (cold load with receive CRC, VERIFY read-back CRC, warm restart, boot).
CC's hardware: each `LOAD_DATA` write goes into a core-side holding register and through a tagged
handshake into the TCM write port (4,096-cycle bound, `LOAD_TIMEOUT`); `LOAD_DATA` reads and the
VERIFY reader use the TCM read port through the same handshake, one word per request, while
`RV_RUN` = 0. Then `RV_BOOT` ← vector-table address and `CTRL.RV_RUN` ← 1; the runtime posts
HELLO (EX §4.7 R1). 192 KiB ≈ 49 K posted writes, once per process.

### 6.5 Log ring

EX §4.10 is normative: a 256-word core-domain ring, drop-new when full (words counted in
`LOG_DROPPED`), host reads at 0x400 + 4 × (k mod 256) and writes `LOG_RD`; records use DB's header
(DB §6.1) and are never torn because the SDK checks `LOG_FREE`. CC's hardware: `LOG_PUSH` (control
window 0x280) → 16 × 32-bit CDC FIFO → the ring; the RISC-V-side `LOG_FREE` of EX §4.1. The ring
works when the queues wedge.

### 6.6 `tsr_csr` read requesters

Two requesters share one access pipeline (`rtl/core/tsr_csr.sv:9-14`): the host CSR port
and the core-window handshake, round robin per access. Only host LO reads touch the snapshot. The
LQ / LCQ handshakes of §5.2 use a third, dedicated port into the ring logic, so a core-window
read never delays the runtime's pop or post.

---

## 7. Issue path

### 7.1 Push (clk_rv)

At the `ISSUE_PUSH` store, in one clk_rv step, if a credit exists for unit u = `UNIT_OF(w0[7:0])`
(static table, §7.4) and a FIFO entry is free: write the issue-FIFO entry {OP, w0–w7 masked,
DEP0, DEP1, REC, ticket, push PC}; ticket = {u, `ISSUED_SEQ[u]`}; `ISSUED_SEQ[u]`++;
`ISSUE_PIDX`++; `ISSUE_TICKET` ← ticket; DEP0/DEP1 ← none; `rvalid`. The push PC is Ibex's
`crash_dump_o.current_pc` at the store (DB §5.6, *confirm* that it is the store's PC). No
SPM-window store can be outstanding at a push: the data port holds one access at a time and a
bridge store completes only after `tsr_spm` accepted it, so an op pushed after a completed
SPM-window store sees it.

### 7.2 Issue-FIFO entry (387 bits, `tsr_cdc_fifo`, 32 entries LUTRAM)

| Bits | Field |
|---|---|
| [1:0] | kind: 0 OP, 1 CTRL, 2–3 reserved |
| [257:2] | w0–w7 (OP); CTRL: w0[3:0] = 1 QABORT, 2 QHOLD set, 3 QHOLD clear, 4 QCLEAR, 5 TRACE_MARK (w1 = marker value, DB §8.4) |
| [289:258] / [321:290] | DEP0 / DEP1 (OP) |
| [322] | REC |
| [354:323] | ticket {unit[31:28], seq[27:0]} (OP and QCLEAR; the dispatcher asserts it equals its own count) |
| [386:355] | push PC (OP and TRACE_MARK; DB's history ring and trace records) |

### 7.3 Credits

- A queue slot belongs to an entry from its push until its unit accepts it (or it completes
  without running). Credit of u = min(`Q_DEPTH[u]` − (`ISSUED_SEQ[u]` − S_sync[u]), FIFO free
  entries, 15), where S[u] counts accepts and non-running completions in the core domain.
- **Software checks the credit before pushing** (`docs/ctrl-riscv.md` §4.1); the hardware wait is
  a safety net: at most `ISSUE_WAIT_MAX` = 65,536 clk_rv cycles, then a store access fault and
  PUSH_TIMEOUT. The count does not advance while DB's `dbg_stop` level is 1 (DB §3.9), so a debug
  stop never turns a waiting push into a fault. While a push waits, Ibex takes neither interrupts
  nor a debug halt.
- Because credits include FIFO space, the dispatcher never blocks on a full queue.

### 7.4 Units, queues, depths

| Id | Unit | Opcodes (static `UNIT_OF`, generated from EX §7.2) | `Q_DEPTH` (U200 M6) |
|---|---|---|---|
| 0 | DMA | LD, ST, KVST (AT) | 8 |
| 1 | QACT | QACT | 4 |
| 2 | MVM | MVM, ATTQK, ATTPV (AT) | 4 |
| 3 | VU | VOP, RED, ROPE, SOP, ASMAX, AMASK, AMERGE, ANORM (AT); TOPK and DQROW from M8, VMASK from M9 | 16 |
| 4 | AT engine | reserved; **absent in M6** (`ISSUE_UNITS[4]` = 0, `Q_DEPTH[4]` = 0, SPM client 5 tied off). The parameter depth 8 applies if a later bitstream builds it | 0 |
| 5 | SYNC | NOP (join point), FENCE (barrier), QCLEAR (via `QCTRL`); every opcode not accepted by the issue path or whose unit is not built | 4 |
| 6–13 | reserved | — | — |
| 14 | marker records (DB); never a valid dependency | — | — |
| 15 | none ("no dependency") | — | — |

Opcodes 0x01–0x09, and every opcode an M6
bitstream does not implement (TOPK, DQROW, VMASK), are not accepted: SYNC ticket, `ILLEGAL_OP`.
Queues are LUTRAM, written by the dispatcher, read into a head register (§9.1). Depths are
parameters; `Q_DEPTH` (EX §5.1, 0x088) reports them.

### 7.5 Tickets

Ticket = {unit[31:28], seq[27:0]}, seq from 0 per unit after a hardware reset, never reset
otherwise. Ticket (u, s) is done iff (D[u] − s) mod 2^28 ∈ [1, 2^27). Push index: 32-bit count of
ticketed entries (ops and QCLEAR) in push order (error ordering, `ERR_PIDX`); the RISC-V reads the
next one as `ISSUE_PIDX`, as in `docs/ctrl-riscv.md` §4.3.

### 7.6 Issue cost (*estimate*)

RISC-V side: credit read 2 cycles, 1–7 word stores (staging words are kept after a push, so a loop
rewrites only changed words), 0–2 dep stores, push, optional ticket read: **15–40 clk_rv cycles**
per op including address arithmetic (90–240 ns at 166.7 MHz). *Measured* on the simulator (SW §4.2
window latencies: a control-window access costs 3 clk_rv cycles on Ibex maxperf, 4 on small,
against the 2 assumed here for the credit read; *confirm* with the bus RTL): on MUL_MAT's decode
fast path the issue-path window accesses take ≈ 25 cycles per pushed op (per op ≈ 4.8 staging-word
stores, 0.85 dep stores, the push, the ticket read and ≈ 0.5–1 credit read, SW §7.8 profile), and
the whole per-op cost including the executor's loop and the kind handler is ≈ 100 cycles. Core
side: push → start of an idle unit with done deps ≈ 16–18 core cycles (FIFO sync 2–3, pop 1,
checker 8, enqueue 1, head load 1, evaluation 1, cmd_valid 1).

---

## 8. Dispatcher and checker

### 8.1 Operands and the MVM weight extent

`tsr_chk` takes its operands from w1..w3 directly (addresses are immediates), and `w0[31:16] ≠ 0` on
an op without an x16 field gives `ILLEGAL_OP` in C1. The MVM weight extent uses a serial divider and
shift-add for odd `NUM_CH`; for power-of-two `NUM_CH` it is R = (N + C − 1) >> log2 C and
R × ROW_BEATS in one DSP (2 cycles). The dispatch handshake is in the queue heads (§9).

### 8.2 `tsr_chk` pipeline (one op at a time)

C0 capture entry (+ per-word nonzero flags) → C1 decode: opcode class, reserved fields, sub codes,
unit built → ILLEGAL_OP → C2 K / T / fmt, QA terms, strides, MVM R → C3 QA_LINES, (T−1)·stride,
MVM DSP stage 1 → C4 extents (48 bit), MVM DSP stage 2 → C5 end addresses → C6 limit compares →
C7 priority, command build, queue write. **8 core cycles per op** (MVM too for power-of-two
`NUM_CH`; ≈ 46 otherwise): ≈ 34 M ops/s at 276 MHz, ≥ 3× the RISC-V's best issue rate.

Error priority (EX §8.1, spec order `docs/spec.md` §21 (Error check order) with the issue-path rules):
`ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` → `T_RANGE` → `OPERAND` (13, `ERR_INFO` = the
offending value) → `SPM_RANGE` → `DRAM_RANGE`, range operands in field order, `ERR_INFO` = the
overflowing start address (spec range-check rule, `docs/spec.md` §21). Opcodes from VU / AT plug
in as classes: each supplies its reserved-field masks, used words, sub legality, `OPERAND`
conditions in their order, **up to four operand extents, each tagged SPM or DRAM and read or
write** (at most one DRAM extent, checked on every channel as for MVM weights), as formulas computable in C2–C4, and its
command struct (§16). Four SPM extents are needed by VOP and ROPE (VU §5.9) and by AMASK and
AMERGE (AT §5.8). The same four tagged extents feed DB's tile-breakpoint matcher (DB §5.3). A
merging AMERGE lists six (Orun and st each as a read and a write entry): the checker class keeps the
first four, and the simulator's matcher takes the same four, so a watch on `st` never hits.

### 8.3 Dispatcher state machine (`tsr_issue`)

```
D_IDLE  : FIFO non-empty → D_POP
D_POP   : register the entry.
          CTRL QABORT   → abort_all (§12.1); → D_IDLE
          CTRL QHOLD    → set / clear the RISC-V hold source; → D_IDLE
          CTRL TRACE_MARK → stamp MTIME, hand {value, pidx, push PC} to DB's REC / TRW (DB §8.4); no ticket, no pidx; → D_IDLE
          CTRL QCLEAR   → SYNC entry {QCLEAR, pidx, snapshot = cnt[0..5]}, cnt[5]++; → D_ENQ
          OP            → u = UNIT_OF(w0[7:0]); start tsr_chk; BAD_DEP check of DEP0/DEP1; → D_CHK
D_CHK   : wait for tsr_chk (C7) → D_ENQ
D_ENQ   : write queue u: run entry or poisoned entry {code, info, w0}; cnt[u]++, pidx++; → D_IDLE
```

`cnt[u]` counts pushes per unit in push order (equals `ISSUED_SEQ[u]` at that position). **BAD_DEP**
for a dep (v, s), v ≠ 15: v ∈ 6..14 or unit v not built; or (s − cnt[v]) mod 2^28 < 2^27 (not
pushed yet; includes the op's own ticket); or (clear_pt[v] − s) mod 2^28 ∈ [1, 2^27) (pushed before
the last QCLEAR). A BAD_DEP dep is replaced by none and the entry is poisoned with BAD_DEP. Deps on
the op's own unit are legal (implied by queue order). FENCE / NOP with w1–w7 ≠ 0 → ILLEGAL_OP.

---

## 9. Queue heads, dependencies, SYNC, errors

### 9.1 Head structure (per unit u)

LUTRAM queue → **head register** (prefetched as soon as the previous head leaves) → evaluation
flops (`dep_ok_q`, `snap_ok_q`, `failed_q`, `hold_q`) → **command register** `cmd_q` / `cmd_v_q`
driving the unit's `cmd_valid / cmd` (the unit-side clock enables are one flop away, 0 LUT levels,
the k4t1 worst-path structure of `docs/rtl-m3.md` §14.3).

### 9.2 Evaluation (registered, every cycle)

- `dep_ok` = done(DEP0) ∧ done(DEP1) against the core-domain D[] (6:1 mux + 28-bit modular
  compare, registered). SYNC FENCE / QCLEAR also need `snap_ok` = ∀v: D[v] has reached snap[v].
- `failed(t)` for dep t = (v, s): unit v in error state ∧ (s − ERR_SEQ[v]) mod 2^28 < 2^27.
- `hold` = host hold ∨ RISC-V hold ∨ (debug halt ∧ HOLD_ON_HALT) ∨ `dbg_hold[u]` (DB, §16).
- `hard_hold` = `dbg_hold[u]` ∧ ¬`abort_active` ∧ ¬`SR_BUSY`, where `abort_active` is set by any
  QABORT (host, RISC-V, `LQ_CANCEL`, RISC-V reset, `SOFT_RESET`) and cleared by the next QCLEAR.
  `SR_BUSY` enters this term delayed by the latency of a QABORT (two cycles: the dispatcher
  registers the pulse and applies it a cycle later), so a held NOP / FENCE / error-coded head is
  never decided with its own code between `SOFT_RESET` and the abort: it completes ABORTED, as
  in the simulator, whose step 1 aborts first.
  DB's hold blocks non-running completions too (DB-9), except after a QABORT: aborted entries
  change nothing but counters, so they drain, and `SOFT_RESET` and the RISC-V reset always reach
  QIDLE.

### 9.3 Head state machine

```
H_EMPTY : queue non-empty → H_LOAD
H_LOAD  : head ← queue[rd]; → H_EVAL                (an entry is never started in the cycle it
                                                      becomes head: DB's hold has one cycle to act)
H_EVAL  : if !dep_ok (or SYNC FENCE/QCLEAR !snap_ok): stay
          elif hard_hold (and no dbg_step[u]) : stay      (DB's hard hold, §9.2)
          elif poisoned            → COMPLETE(code)
          elif unit u in error     → COMPLETE(ABORTED)
          elif a dep failed        → COMPLETE(DEP_FAILED)
          elif SYNC NOP / FENCE    → COMPLETE(0)
          elif SYNC QCLEAR         → clear (§9.5); COMPLETE(0) after the RV side shows it
          elif hold (and no dbg_step[u]) : stay
          elif unit may accept     → cmd_v_q ← 1; → H_ISSUE
H_ISSUE : on cmd_valid ∧ cmd_ready: S[u]++, cmd_v_q ← 0; head leaves (next head prefetched); → H_EMPTY / H_LOAD
COMPLETE(code): if code ≠ 0 and unit u not in error state: enter error state, record handshake (I2);
          S[u]++, D[u]++ (after the record acknowledge if it set the error state); → H_EMPTY / H_LOAD
Unit done: D[u]++
```

"Unit may accept": the unit's `cmd_ready`. Units have no error output: no op may fail after it has
started (EX §7.2), so every error is a dispatch-time poison. The MVM and the VU may accept the next
op before their done (the M5 as-if overlap; the VU's accept-ahead, VU §7.8), provided their done
pulses stay in accept order (I3). At most one D[u] increment per cycle per unit (gray-counter
rule).

Latency targets: done of t → start of an idle dependent on another unit ≤ 3 core cycles; done →
start of the next entry of the same unit ≤ 2 cycles.

### 9.4 Invariants (asserted in every simulation)

- **I1 completion rule** (`docs/ctrl-riscv.md` §4.4): every entry completes only at its head and
  only after both deps are done (a BAD_DEP dep ignored); FENCE / QCLEAR also after their snapshot.
  Hence done(t) implies every op t depends on, transitively, is done.
- **I2 errors visible before completion**: D[u] does not pass ERR_SEQ[u] until the clk_rv side
  acknowledged unit u's record (≈ 5–8 core + 3–4 clk_rv cycles, *estimate*, once per error state).
- **I3**: a unit holds at most the ops it accepted, completes them in accept order; S[u] − D[u] ≤ 2
  for MVM (overlap) and VU (accept-ahead, VU §7.8; tables `ACCEPT_MAX`), ≤ 1 for every other unit.
  (Corrected 2026-10-31, W4-A: the VU's limit is 2 as in §15.2, §16 and the table source, not 3.)
- **I4**: no SPM or memory traffic from a unit that holds no op (the integration check U2 of
  `rtl/core/tsr_core_dp.sv`, per unit; W4-A: moved there from `tsr_core.sv`).

### 9.5 Errors, QABORT, QCLEAR

- Error state, ERR_* capture, propagation, first error by smallest push index: as
  `docs/ctrl-riscv.md` §4.5. Codes (EX §8.1): 1–8 as spec; **11 DEP_FAILED**, **12 BAD_DEP**,
  **13 OPERAND**; 9, 10, 14, 15 reserved. ERR_INFO: range errors the overflowing address, OPERAND
  the offending value, else 0. ERR_W0: the entry's w0.
- **QABORT** (RISC-V `QCTRL` when the dispatcher pops it; host `CTRL.QABORT`, `LQ_CANCEL` covering
  the running launch, RV_RUN 1 → 0 and `SOFT_RESET` at once): all six units enter error state with
  ERR_CODE ABORTED and ERR_SEQ[u] = S[u] (the first
  entry not started); running ops finish (no cancel); every later entry completes ABORTED, in
  order, after its deps.
- **QCLEAR (RISC-V)**: SYNC entry; when every earlier entry is done, clears all error states and
  records, sets clear_pt[u] = snap[u]; its own D increment waits until the clk_rv side shows the
  cleared records. **QCLEAR (host)**: accepted only when QIDLE, the issue FIFO empty and the
  RISC-V quiet (RV_RESET_DONE, or halted with no access outstanding); else QCLEAR_REFUSED.

### 9.6 Holds and quiescence

Hold sources: host (`CTRL.HOST_HOLD`), RISC-V (`QCTRL`, in push order), debug halt
(`RV_HALTED` ∧ `HOLD_ON_HALT`), and DB's per-unit `dbg_hold[u]`. Each has one owner. A hold stops
heads from starting; non-running completions (poisoned, ABORTED, DEP_FAILED, SYNC) still drain,
except under DB's hard hold (§9.2). `QQUIESCENT` = no op held by a datapath unit, dispatcher and
FIFO idle, every non-empty queue held, no non-running entry left to drain; a head held by
`hard_hold` counts as held whether or not it would run. With the RISC-V halted it is the quiescent
state of `docs/ctrl-riscv.md` §4.6.

---

## 10. SPM: even / odd banks, arbitration, RISC-V bridge

### 10.1 Organisation

The 4 MiB SPM is two **line-interleaved banks**: bank E holds even lines, bank O odd lines,
each 1R1W, so the SPM has **2 read and 2 write ports** (CC-12; the technical default of D14–D18,
`docs/design/README.md` §2). Each bank is 32 K lines = 2 depth banks × 8 column slices of
16 K × 64 (4 cascaded URAM288): **128 URAM**. The r3 per-slice kept input registers
(`docs/rtl-m3.md` §14.9) are kept per (parity, depth bank, slice).

### 10.2 Client port contract

Per client c and bank b ∈ {E, O}:
- read request `rd_req_valid[c][b]` / `rd_req_ready[c][b]` / `rd_req_line[c][b]` (line[0] = b,
  asserted); response `rd_rsp_valid[c][b]` exactly `SPM_RD_LAT` cycles after the accepting edge,
  data on the shared bus `rd_rsp_data[b]`; no backpressure; in request order per (c, b).
- write `wr_valid[c][b]` / `wr_ready[c][b]` / `wr_line` / `wr_data` / `wr_be[63:0]`.
- A client may raise requests on both banks in the same cycle (two lines per cycle). Rule R2 per
  (c, b): a raised request stays with unchanged payload until accepted.
- Ordering: a read accepted after a write to the same line was accepted returns the new data (one
  bank owns the line, its input stage is shared by reads and writes).
- **Arbitration per bank port**: registered one-hot grant, round robin (the client granted last
  has the lowest priority), the owner keeps the grant while the port idles. `ready` = grant flop:
  no combinational path from any input to any output (R5 kept). A waiting client is granted after
  at most NCL − 1 other grants.
- Clients: 0 DMA, 1 QACT, 2 MVM, 3 RISC-V bridge, 4 VU, 5 AT (absent → tied off). `NCL` = 6.

**Single-port adapter** `tsr_spm_cport` (client side, used by DMA, QACT, MVM, bridge): the client
keeps its single-port interface (one read and one write request per cycle); the adapter steers by line[0],
`ready` = the steered bank's grant, response valid = OR of the two, data = 2:1 select by the bank
that answered. A streaming client alternates banks and keeps both grants when alone: one line per
cycle. The adapter's `ready` depends on the client's own request line (inside the
client, one LUT): accepted because R5 is about the SPM module boundary.

**VU**: one client using both bank ports; it reads operand pairs (lines L, L+1 are always in
different banks) and can therefore stream two source operands at one line each per cycle whatever
their alignment (alternate cycles: A-pair, B-pair), and write one line per cycle (VU area designs
the reorder buffers).

### 10.3 No-deadlock argument (to go into the rtl-m2 / rtl-m3 contracts)

(i) every client raises an SPM request only with its payload ready and never waits on memory while
it holds a request (the DMA's LD writes beats already in its queue, ST reserves queue space before
reading; QACT, gather, VU, AT write results they hold; the bridge holds its data); (ii) a granted
request transfers in that cycle; (iii) round robin grants a waiting client within NCL − 1 grants.
So every SPM request completes in bounded time independent of memory progress. With the DMA LD
reservation (§10.4) no memory-response path ever waits on the SPM.

### 10.4 DMA LD burst reservation (CC-13)

`tsr_dma` LD issues a read burst only when its beat queue has room for the whole burst; the queue
grows to 512 beats (8 RAMB36 in 512 × 72 columns); `rd_rsp_ready` becomes **constant 1**
(asserted). Effect: SPM contention (VU streaming writes, bridge, QACT) can no longer stall a
channel's head-of-line response path, so MVM weight beats never wait behind a DMA beat
(`docs/ctrl-riscv.md` §5.3). 512 beats in flight cover 1.7–1.9 µs of DRAM latency at one beat
per cycle (276–300 MHz; *estimate*: enough for the measured 17.2–17.4 GB/s per channel).

### 10.5 RISC-V SPM bridge and line buffer

- Core side `tsr_spm_bridge` (client 3): read = full line, returns the word (or the whole line for
  a line-buffer fill); write = the word placed at its offset with `wr_be` = byte enables shifted by
  4 × word. Answers a store only after `tsr_spm` accepted it.
- RISC-V side bound 4,096 clk_rv cycles → access fault + `BRIDGE_TIMEOUT`; late answers discarded
  by tag (`docs/ctrl-riscv.md` §3.5).
- **Line buffer** (`tsr_spm_lb`, one 512-bit line + tag in clk_rv): loads from the 0x5000_0000
  alias hit it when the tag matches (next-cycle `rvalid`), else fill it with one bridge read.
  Invalidated by: any store to either SPM alias, any control-window or core-window access, any
  rising edge of an interrupt source line, `SPM_LB_INV`. Every observation of a completion
  (`DONE_SEQ` / `QSTATUS` / record reads, the WAIT interrupt) therefore invalidates it, so memory
  model rule 2(c) holds for the alias too. Speed (*estimate*): 64 B in ≈ 160 ns instead of
  ≈ 1.1 µs (16 exact-window loads), for op lists and argument blocks (M7).
- Latency (*estimate*, 166.7 / 276 MHz): exact store ≈ 40–60 ns, exact load ≈ 55–80 ns.

---

## 11. Clock domains and CDC

<p align="center"><img src="../fig/clocks.svg" alt="Two tinted clock-domain regions side by side. The XRT shell, a grey box at the top, supplies the kernel clocks ap_clk and ap_clk_2 and their resets as lines in each domain's colour, and carries the CSR port and the memory ports into the core domain. The core clock domain (blue, ap_clk, 276.3 MHz in the k4 build) holds the platform adapter, tsr_csr, the issue path with tsr_chk and the unit queues, tsr_spm_bridge, the DMA, QACT, the MVM clusters, tsr_spm and one memory arbiter per channel. The control core domain (orange, clk_rv from ap_clk_2, 166.7 MHz planning value) holds Ibex, the debug module, ITCM, DTCM, tsr_rv_bus and the control window. On the boundary tsr_ctrl_cdc has one synchroniser cell per bundle of crossings, each an arrow from the source block to the destination block with its width. From clk_rv to the core clock: the issue FIFO (gray pointers, 32 x 387 b), the log FIFO (16 x 32 b), the gray counters ISSUED_SEQ, clk_rv cycles and PERF, the RT_STATE and RT_INFO handshake, and the RV levels and sticky events through two-flop synchronisers. From the core clock to clk_rv: the S, D and MTIME gray counters, the error records (6 x 128 b) and the levels. Both ways: the SPM bridge and core window handshake, the LQ and LCQ register handshake, the LOAD handshake into the TCM load port and the DMI handshake. The core reset reaches clk_rv through a reset synchroniser. A legend shows the cell symbols and the constraints: ASYNC_REG, a maximum delay equal to the smaller clock period, and a bus skew limit on the gray buses." width="100%"></p>

The figure shows the two clock domains, the crossings between them and the constraints that §11.1-§11.3 describe.

### 11.1 Clocks

- Core clock = `ap_clk`, requested at 300 MHz. The earlier k4 image without the control core
  ran at 276.3 MHz after v++ scaling (`docs/platform-u200.md` §14.1).
- `clk_rv` comes from `ap_clk_2`, using U200 platform clock ID 1. `ap_clk` uses ID 0.
  The link configs use `kernel_frequency=0:300|1:166.666667` for these two scalable clocks.
  Vitis 2024.2 truncates the RV request to 166 MHz in its metadata and timing constraints.
  `[clock] id` selects fixed reference clocks and rejects scalable IDs 0/1 on this platform.
  The pinned U200 flow also has an `INIT_DESIGN.PRE` compatibility hook for the installed
  Vitis/Vivado clock-calculator API mismatch (`docs/platform-u200.md` §8). It validates the
  native calculation before the implementation clock override is applied.
  The BD hook checks the actual clock and reset nets;
  the implementation hook checks the synthesized clocks, RV FFs and reset sources. The final
  pre-bitstream hook checks the scaled frequencies in `_new_clk_freq` (the clock `PERIOD`
  properties do not reflect that scaling). Software reads `RV_FREQ` rather than assuming F_RV.
- `CTRL_SYNC` = 1 (targets with clk_rv = clk): every CDC structure becomes plain register stages.

### 11.2 Crossings

| Crossing | Dir | Structure | Width |
|---|---|---|---|
| Issue FIFO (ops, QCTRL and TRACE_MARK entries) | RV → core | `tsr_cdc_fifo`, gray pointers, 32 entries LUTRAM | 387 b |
| `ISSUED_SEQ[u]` | RV → core | `tsr_cdc_gray` 28-bit × 6 | 168 b |
| S[u], D[u] | core → RV | `tsr_cdc_gray` 28-bit × 12, ≤ 1 step per core cycle (asserted) | 336 b |
| Error records + clear | core → RV | tagged `tsr_cdc_hs` per unit; I2 acknowledge. While clk_rv is not OK a handshake is answered locally, so when it is OK again `tsr_ctrl_cdc` pulses `rec_resync_o` and `tsr_issue` sends all six records once more | 6 × 128 b |
| SPM bridge | RV ↔ core | tagged 2-phase `tsr_cdc_hs`, one access, data held stable | 64 b req / 513 b rsp |
| Core window | RV ↔ core | same structure, shared with the bridge's handshake slot; the read data is `line[31:0]` | 64 b req / 513 b rsp (32 b of data) |
| LQ / LCQ registers (EX §4.5) | RV ↔ core | tagged 2-phase `tsr_cdc_hs` (own slot): reads return the core-domain value; `LQ_POP` and `LCQ_POST` (with the 8 staged words) commit and fault in the core domain (CC-30) | 301 b req / 33 b rsp |
| `LOG_RD` snapshot for `LOG_FREE` | core → RV | part of the LQ / LCQ handshake (the `rdata` of its response) | 32 b |
| Log words | RV → core | `tsr_cdc_fifo` 16 × 32 b | 32 b |
| `RT_STATE`, `RT_INFO` mirrors | RV → core | tagged handshake on each write | 64 b |
| `LOAD_*` | core → RV | holding register + tagged handshake into the TCM write port; VERIFY / `LOAD_DATA` reads through the read port | 65 b req / 33 b rsp |
| DMI | core ↔ RV | tagged 2-phase handshake; 4,096-core-cycle CSR bound; late answers discarded | 41 b / 34 b |
| MTIME | core → RV | 64-bit gray counter | 64 b |
| clk_rv cycle count, clk_rv PERF counters | RV → core | 16-bit / 32-bit gray counters | 112 b |
| Sticky events (PUSH_TIMEOUT, BRIDGE_TIMEOUT) | RV → core | toggle synchronisers | |
| Levels (RV_RUN, holds, RV_HALTED, RV_SLEEP, RV_RESET_DONE, "clk_rv out of reset", ACTIVE, `irq_fast[2]` = LQ_AVAIL ≠ 0 ∧ ¬ACTIVE, `irq_fast[3]` = LCQ_FREE ≠ 0, `dbg_stop`, `dmcontrol.dmactive`) | both | 2-flop `tsr_sync`; RV_RESET_DONE two stages behind the FIFO write pointer; the core → RV bundle is registered once in the core domain before the synchroniser (its bits are combinational functions of several registers: `report_cdc` CDC-10) | |
| `RV_BOOT` | core → RV | 2-flop `tsr_sync` on the 32-bit bus; quasi-static: `tsr_csr` ignores `RV_BOOT` writes while `CTRL.RV_RUN` = 1, so the bus only changes while the hart is held, and the sequencer samples it at the reset release (also after a debug ndmreset) two cycles behind the synchronised RV_RUN; `report_cdc` waiver in `tsr_ctrl_cdc.xdc` | 32 b |

The widths are the exact ones of `rtl/ctrl/tsr_ctrl_pkg.sv` and the instances in `tsr_ctrl_cdc.sv`. The LQ / LCQ request is {`lcq_w[7:0]` 8 × 32 b, `wdata` 32 b, `we` 1 b, `off` 12 b} = 301 b and its response {`fault`, `rdata`} = 33 b; the LOAD request is {`we`, `addr` 32 b, `wdata` 32 b} = 65 b and its response {`err`, `rdata`} = 33 b; the bridge / core-window response is {`err`, `line[511:0]`} = 513 b; a request of the bridge and the core window is {`kind`, `be`, `addr`, `wdata`} = 64 b.

Rules (`docs/ctrl-riscv.md` §7.1): gray counters step ≤ 1 per source cycle; a counter that
can move by more than one step at a time (the host-written `LCQ_RD` and `LOG_RD`, and everything
`SOFT_RESET` flushes) never crosses as a gray code: the RISC-V sees it only through a handshake
(EX §4.1); nothing crosses into a domain in reset (requests answered locally within the bound);
no CSR access waits on clk_rv without a bound.

### 11.3 Constraints (kernel XDC packaged in the `.xo`)

`set_max_delay -datapath_only` = min(period) on every gray bus and handshake data path,
`set_bus_skew` on gray buses, `ASYNC_REG` on synchroniser flops; no `set_clock_groups` (it would
override the max delays; the platform clocks come from one clocking block and may be timed as
related). Gates: `report_cdc` on the out-of-context run lists only the intended crossings, all
safe; zero unconstrained `ap_clk` ↔ `ap_clk_2` paths in the linked design.

The constraints cover every crossing of the design, `tsr_dbg_rv`'s included, not only `tsr_ctrl`'s:
`rtl/ctrl/viv/tsr_ctrl_cdc.xdc` selects the cells by the instance names `u_sync*` (`tsr_sync`), `u_rst*`
(`tsr_rst_sync`), `u_hs*` (`tsr_cdc_hs`) and by the register names of the gray pointers, a convention
`rtl/ctrl/viv/check_cdc_names.py` checks in the lint. It is plain XDC, because Vivado's constraint
reader accepts no `if`, `foreach`, `proc` or `dict` (a first version with them was refused when the IP was
read: "Command 'if' is not supported in the xdc constraint file"). `package_xo.tcl` copies it into the
kernel IP with `ap_clk` / `ap_clk_2` as the two clock ports, in the IP's synthesis file group (used in
synthesis, implementation and out of context, processed late). The `report_cdc` waivers are not
constraints (`rtl/ctrl/viv/tsr_cdc_waivers.tcl`, Tcl): the out-of-context scripts source them, and so does
the post-route hook `hw/u200/cdc_check.tcl` of the k1 / k4 links, which then runs the gate
(`rtl/ctrl/viv/cdc_gate.tcl`: no Critical or Warning `report_cdc` row between the two clocks, and
`report_clock_interaction` showing "Max Delay Datapath Only" in both directions) and fails the
implementation run when it does not hold. `report_cdc` on the whole core (`tb/core/viv/core_ooc.sh`) and on
the packaged IP (`tb/platform/viv/xo_cdc.tcl`) are part of T2 / T3. Every signal enters a synchroniser
straight from a flop that drives nothing else (CDC-10: no combinational logic before it; CDC-11: no other
fan-out): the level bundle of `tsr_ctrl_cdc`, `tsr_dbg_rv`'s levels and the reset sources of the clk_rv
synchronisers have a register of their own (`dont_touch`, so that equal registers of two blocks are not
merged into one with two synchronisers behind it).

---

## 12. Resets, aborts, interrupts, timer

### 12.1 Events

| Event | Queued entries | Op running on a unit | RISC-V | Kept |
|---|---|---|---|---|
| QABORT (RISC-V `QCTRL`, in push order; host `CTRL.QABORT` or `LQ_CANCEL` covering the running launch, at once) | complete ABORTED in order after their deps; all six units in error state until QCLEAR | runs to completion | continues | sequences, records |
| QCLEAR (RISC-V ticketed; host only when idle) | applies after every earlier entry | — | continues | clear point = sequence |
| Hold (any source) | stay; heads do not start | completes | continues | |
| RV_RUN 1 → 0, or DM `ndmreset` | QABORT at once | completes | reset sequence (§12.2) | TCMs, LQ / LCQ / LOG rings, sequences, error states |
| **`SOFT_RESET`** (host `CTRL` bit 10, EX §8.5) | QABORT at once, then drained | completes | reset and left in reset | DRAM, SPM, TCMs, s bank, sequences, the items EX §8.5 lists |
| Debug halt | continue (held with HOLD_ON_HALT) | completes | halted | |
| Hardware reset (`ap_rst_n`) | everything reset; TCM / SPM undefined | | held (RV_RUN = 0) | — |

`SOFT_RESET` is EX's device-level reset (EX §8.5).
There is no partial datapath reset and no cancel of a running op (CC-22); a wedged unit is
recovered by a card reset (EX §8.6).

### 12.2 Reset sequencing (from `docs/ctrl-riscv.md` §7.3)

- Hardware reset: `ap_rst_n` resets the core domain (with `RST_PIPE`); the clk_rv domain's reset is
  `ap_rst_n` through `tsr_rst_sync` (asynchronous assert, synchronous deassert). `ap_rst_n_2`
  (`rst_rv_n`) is AND-ed only into the Ibex / DM reset, never into CDC endpoints. Minimum
  `ap_rst_n` low: max(`RST_PIPE` + 1 core cycles, 3 clk_rv cycles).
- RISC-V reset (RV_RUN 1 → 0 or `ndmreset`): (1) the core applies QABORT and sets ANSWERED :=
  TAKEN (EX §8.5); (2) the clk_rv sequencer stops granting on both Ibex ports and waits for the
  outstanding access (bridge bound at most); a push waiting for a credit is dropped (no FIFO entry,
  no ticket, no `ISSUE_PIDX` step); (3) resets Ibex, staging, deps, `WAIT_*`, the RISC-V hold
  source, `LCQ_W*`, `RT_*` and the line buffer for ≥ 3 clk_rv cycles; (4) raises RV_RESET_DONE.
  CDC endpoints, the core-domain rings, MTIME, TCMs and the DM's DMI side are not reset by it.
- **SR sequencer** (core domain, `tsr_csr`; implements EX §8.5 and needs no `clk_rv`, CC-31).
  A `CTRL.SOFT_RESET` write sets `SR_BUSY` and completes at once. Then:
  1. drop `RV_RUN` (starting the RISC-V reset sequence above, with its QABORT); clear DB's stop
     state and force `dbg_hold` = 0 (DB §5.2); wait at most 2^16 core cycles for `RV_RESET_DONE`,
     else set `SR_RV_TIMEOUT` and go on (no `clk_rv`);
  2. wait until QIDLE holds with no datapath unit holding an op (the issue FIFO, every queue and
     every running op drained); a hung unit never finishes this step (EX §8.6);
  3. in the core domain: TAKEN := PUSHED, ANSWERED := TAKEN, LCQ_RD := LCQ_WR, LOG_RD := LOG_WR,
     both watermark VALID bits := 0, the host QCLEAR effects (error states and records cleared,
     clear points set), PERF zeroed and the snapshot dropped, the sticky `STATUS` bits and
     `LOG_DROPPED` cleared;
  4. `SR_BUSY` := 0.
  `RV_RUN` writes are refused (`RV_RUN_REFUSED`) while `SR_BUSY`. The clk_rv-side state (LOG_WR_rv,
  the interrupt levels) reloads from the core domain when the RISC-V leaves reset.
- Boot: core out of reset → image load → RV_RUN = 1 → Ibex reset deasserts 2–3 clk_rv cycles after
  the synchronised RV_RUN (and `ndmreset` low) → fetch at `RV_BOOT` + 0x80.

### 12.3 Interrupts (Ibex, level-sensitive, computed in clk_rv)

| Input | Source | Cleared by |
|---|---|---|
| `irq_fast[0]` | `WAIT_TICKET` done | rewriting / disarming `WAIT_TICKET` |
| `irq_fast[1]` | `ISSUE_STATUS.ERR_IRQ` | W1C |
| `irq_fast[2]` | `LQ_AVAIL` ≠ 0 ∧ ¬ACTIVE (EX §4.5; synchronised level) | `LQ_POP` |
| `irq_fast[3]` | `LCQ_FREE` ≠ 0 (synchronised level; the runtime enables it only while it waits to post) | posting / disabling |
| `irq_fast[4]` | DB: `REC_COUNT` ≥ 32 with `REC_CFG` bit1 (DB §8.2) | `REC_POP` |
| `irq_fast[5]` | `WAIT_CREDIT` | disarm |
| `irq_fast[6]` | reserved for DB | |
| `irq_timer` | MTIME ≥ MTIMECMP | MTIMECMP write |
| others | tied 0 | |

MTIME counts core clock cycles (the time base of PERF and of DB's records). WFI wakes on a
pending enabled interrupt with `mstatus.MIE` = 0 (*confirm* for Ibex).

---

## 13. Changes per RTL module, platform

### 13.1 Core boundary (`tsr_core`)

Inputs `clk`, `rst_n`, `clk_rv`, `rst_rv_n`, `plat_err_i` (EX §9.3); the CSR port; one memory
port per channel; output `host_irq_o` = `CTRL.IRQ_EN` ∧ (LCQ_WR ≠ LCQ_RD) (EX §5.2; not connected
on the U200). The control core is mandatory (D14), so there is no `HAS_CTRL` parameter. Further
parameters: `ITCM_BYTES`, `DTCM_BYTES`, `Q_DEPTH_*`, `ISSUE_FIFO_DEPTH`, `HAS_AT` (0 in M6),
`HAS_SBA`, `CTRL_SYNC`, `LQ_DEPTH`, `LCQ_DEPTH`, `LOG_DEPTH`.

### 13.2 Module table

| Module | Fate | Change |
|---|---|---|
| `tsr_core.sv` | rewritten | new boundary; instantiates `tsr_ctrl_rv`, `tsr_ctrl_cdc`, `tsr_core_dp`; integration checks I1–I4 of §9.4 |
| `tsr_core_dp.sv` | new | core-domain top: `tsr_issue`, `tsr_csr`, units, `tsr_spm`, arbiters; issue-entry stream input |
| `tsr_csr.sv` | rewritten | EX §5 map; no run control; two read requesters plus the LQ / LCQ port; LQ / LCQ / log rings; SR sequencer; DMI and load holding registers with the VERIFY reader |
| `tsr_pkg.sv` | changed | unit ids; codes 11 / 12 / 13; generated `tsr_tables_pkg` (EX-30) for offsets, opcodes and units |
| `tsr_spm.sv` | changed | 2R2W even / odd, per-bank client ports, round robin (§10) |
| `tsr_dma.sv` | changed | LD burst reservation, queue 512 beats, `rd_rsp_ready` = 1; `tsr_spm_cport`; KVST with meta-beat read-modify-write (AT §8.3) |
| `tsr_qact.sv` | changed | `tsr_spm_cport` only; logic unchanged |
| `tsr_mvm.sv`, `tsr_mvm_gather.sv` | changed | `tsr_spm_cport` on client 2; dispatch from the MVM queue head; ATTQK / ATTPV commands; gather full-line ATT path (AT §8.1); gather slice FORM 1 (§14.4) |
| `tsr_mem_arb.sv` | changed | source 0 = AUX; `tsr_rslice` gets parameter `FORM` (§14.4); a second write source on channel `TRW_CH` = 0 for DB's trace writer (DB §8.3), burst-arbitrated with the DMA |
| `tsr_mvm_cluster`, `_front`, `_back`, `_actbuf`, `tsr_mac_lane` | changed (area AT) | ATTQK and ATTPV modes (AT §8) |
| `_ws`, `tsr_fp_*`, `tsr_vpipe`, `tsr_rpipe` | unchanged | |

### 13.3 Platform adapter (`rtl/platform/alveo_xrt`)

EX §9.1–§9.3 are normative (CC-19):

- Control protocol **`ap_ctrl_none`** (kernel.xml `hwControlProtocol="ap_ctrl_none"`,
  `interrupt="false"`): no `ap_start` / `ap_done`, no start sequencer; the host uses `xrt::ip`
  register access and polls. The pinned XRT 2.18 knows only `AP_CTRL_HS`, `CHAIN`, `NONE`, `ME`
  and the adapter types (`core/include/xclbin.h:335-338`), and routes `AP_CTRL_NONE` to `xrt::ip`
  (`xrt_kernel.cpp:1560-1561`). Fallback if v++ 2024.2 refuses it (checked in the first smoke
  link): declare `ap_ctrl_hs` with `interrupt="false"`, make 0x000 read a constant ap_idle = 1, and
  never call `xrt::run`. `user_managed` is only a later *confirm* alternative for a newer XRT.
- `s_axi_control` (EX §9.2): 0x000–0x03C reserved, read 0 (no GIE / IER / ISR); 0x040 + 8c
  `ch<c>_base` lo / hi, written while `MEM_EN` = 0; 0x0E0 `ADAPTER_CTRL.MEM_EN` latches and
  alignment-checks all bases on its rising edge, locks them while set, and holds core memory
  requests while 0; 0x0F0 `ADAPTER_STATUS` (bits 0–2 response errors, bit3 base misaligned at
  `MEM_EN`, bit4 `REQ_HELD`, bit5 `BASE_LOCKED`, [31:16] saturating error count); 0x0F4
  `ADAPTER_VERSION` = {16'h0001, 8'h00, NUM_CH}; 0x1000 + off the core CSR window.
- `plat_err_i` = `ADAPTER_STATUS` bits 0–2 or error count ≠ 0; `host_irq_o` is unconnected.
- Kernel arguments: only `ch<c>` (0x040 + 8c, `void*` on `m_axi_ch<c>`) for connectivity.
- Second clock `ap_clk_2` / `ap_rst_n_2` on `tsr_u200_k1` / `k4` (`gen_tops.py`, `kernel_k*.xml`).
- `tsr_xrt_reset` unchanged for `ap_clk`; `ap_rst_n_2` passes to the core as `rst_rv_n`.

### 13.4 One register table

The single table source is `sw/tables/tsr_tables.py` with SW's generator `gen_tables.py`
(EX-30). CC owns its rows: the control window (§5), the CC parts of the host map (§6.1), unit
ids, queue depths, the issue-FIFO entry layout and the `UNIT_OF` map. The generator emits
`rtl/core/tsr_tables_pkg.sv`, `sw/sim/include/tsr_tables.h`, `sw/ctrl/include/tsr/mmio.hpp` and
`tb/common/tables.py`, with `--check` in lint.

---

## 14. Floorplan, timing and cost on the U200

### 14.1 SLR1 budget (pblock `pblock_dynamic_SLR1`: 199,680 LUT sites, 420 RAMB36, 160 URAM, 1,320 DSP; `docs/platform-u200.md` §8.1)

| Content | LUT | RAMB36 | URAM | DSP | Source |
|---|---|---|---|---|---|
| Core part in SLR1 before the control core (C') | 25.5 K | 8 | 128 | 32 | measured, `docs/platform-u200.md` §8.1 |
| DDR[2] controller + memory subsystem | 30.5 K | 61 | 0 | 3 | measured (M0 link), `docs/rtl-m3.md` §14.5 |
| less: logic of the measured core part that the control core replaces | −2 to −3 K | 0 | 0 | 0 | *estimate* |
| control core + issue path + CSR block (§14.5, before the row above; Ibex maxperf, CC-34) | 14.5–25 K | 49 | 0 | 4–6 | *estimate* |
| SPM 2R2W muxes, `tsr_spm_cport`s, DMA reservation | 3–5 K | 8 | 0 | 0 | *estimate* |
| VU incl. AT's vector ops, M6 configuration (VU §7.10; ± 30 %) | 48 K | 15 | 0 | 366 | *estimate* |
| DB (DB §11) | 4.2–6.3 K | 7 | 0 | 0 | *estimate* |
| AT in SLR1: KVST in the DMA, central-gather ATT path (AT §8.4) | 2–5 K | 0 | 0 | 0 | *estimate* |
| **Total** | **≈ 125–143 K (62–72 %; ≈ 158 K = 79 % with VU + 30 %)** | **148 (35 %)** | **128** | **405–407 (31 %)** | |

**Measured in W4-B (2026-11-01; out of context, xcu200-2, 3.333 ns; `tb/vu/viv/vu_ooc.sh`, `tb/core/viv/core_ooc.sh`)**:
the estimates above for the VU, DB and AT are low. VU, M6 configuration (16 EXP / 8 back lanes), placed and routed: 84.0 K LUT,
101.7 K FF, 433 DSP, 16 RAMB36 + 25 RAMB18 (WNS +0.003 ns); with the shrink knobs (8 / 4): 75.3 K LUT, 297 DSP
(the 16 lanes alone are 49.6 K / 39.5 K LUT). DB: 11.3 K LUT, 7.1 K FF, 7 RAMB36 + 2 RAMB18 (WNS +0.28 ns at 276.3 MHz).
AT cluster modes (`HAS_ATT`, T_MAX 8, synthesis): +35 K LUT, +27 K FF, +39 DSP, +20 RAMB36 per cluster; KVST +3.9 K LUT in
the DMA, the gather ATT path +4.5 K LUT. With the whole VU in SLR1 the total is 152–179 K LUT (76–90 %), above the gate
for either knob setting (the knobs save 8.7 K LUT against an excess of 31–50 K); with the VU lanes in SLR0 (remedy 3: implemented
2026-11-01 as the parameter `VU_LANE_PIPE`, `rtl/core/tsr_vu_lanes.sv`: that many register stages on each side of the
lane array, so every lane exit and every partial drain read takes 2 × `VU_LANE_PIPE` cycles longer — +4 at the 2 stages
of k4 — and the results are bit-exact; the lanes in SLR0 by `hw/u200/k4_floorplan.tcl`) 111–130 K
LUT (56–65 %). The integration builds k4 with 16 / 8 lanes and `VU_LANE_PIPE` 2 and k1 with 8 / 4 and no pipe
(`rtl/platform/alveo_xrt/gen_tops.py`); the k4 setting is final only if SLR0 takes the lanes' 49.6 K LUT and 368 DSP
(fall back to 8 / 4 if not).
Whole-core synthesis of k1 (`core_ooc.sh k1 ... VU_EXP_LANES=8 VU_BACK_LANES=4`, W4-B): 167 K LUT, 190.6 K FF, 535 DSP,
80 RAMB36 + 33 RAMB18 + 140 URAM; post-synthesis WNS -0.73 ns at 3.333 ns (Ibex instruction path to the ITCM URAM
cascade; the DB stop logic and trace writer paths are within -0.46 ns).

AT's cluster modes (ATTQK / ATTPV, ≈ 19–27 K LUT per cluster *estimated*) are in SLR0 / SLR2 with the clusters
and do not count here. **Budget gate (CC-24)**: SLR1 ≤ 65 % of the pblock's LUT sites
(≈ 130 K) after synthesis. The estimate straddles it, so the out-of-context VU run is a
precondition of T2 (§14.3), and above the gate the remedies apply in this order, none of which
changes a result:

1. VU shrink knobs `VU_EXP_LANES` 8 and `VU_BACK_LANES` 4 (VU §7.10): −8.7 K LUT, −124 DSP; EXP
   and the half-rate functions run slower (the cycle model reads `CFG3`);
2. DB `HAS_TRW` = 0 and `HIST_DEPTH` = 0 (DB §11): ≈ −1.5–2.2 K LUT, −7 RAMB36; `DBG_ID` reports it;
3. the VU lanes move to SLR0 behind registered pipes (VU §7.10): ≈ 1.6 K SLLs, +8–12 cycles of VU
   latency.

There is no AT unit to move out: AT's ops already run on the clusters (SLR0 / SLR2), the DMA and
the VU.

### 14.2 Placement

- `ctrl_rv` sub-pblock of `pblock_dynamic_SLR1` holding ≥ 64 RAMB36 sites (TCMs: 48) and not
  overlapping the URAM columns of the SPM: Ibex, DM, TCMs, `tsr_ctl_win`, the clk_rv side of the
  CDC. Its paths do not interact with the core clock except through the CDC.
- `tsr_issue`, `tsr_chk`, `tsr_csr`: SLR1 between the CSR bridge and the units. Only
  `tsr_spm_bridge`'s request / response registers and the VU's SPM-facing stage sit beside
  `tsr_spm`. Clusters (with AT's modes) stay in SLR0 / SLR2 (C' unchanged). New floorplan name:
  **C''** (`hw/u200/k4_floorplan.tcl` extended; `tb/core/viv/core_slr.tcl` likewise).
- k1 (NUM_CH 1, T_MAX 1) stays entirely in SLR1 with the control core, the VU at its minimum
  configuration and the cluster's AT modes (*estimate*: fits with the VU shrink knobs) and remains
  the fast bring-up link (≈ 1 h). *Against the measured whole-core synthesis of k1 (167 K LUT, §14.1), with the
  DDR[1] subsystem (≈ 30 K LUT, M0 link) and the adapter (≈ 4 K) that is ≈ 201 K LUT in `pblock_dynamic_SLR1`'s
  199,680 sites: the T4 link is where this is decided; the remedies are the lanes (or the cluster) in SLR0 behind
  `VU_LANE_PIPE` as in k4.*
- C'' is `hw/u200/k4_floorplan.tcl` (the v++ link) and, through `tb/core/viv/core_slr.tcl`, the out-of-context runs.
  The control core's sub-pblock `pblock_tsr_ctrl_rv` is `CLOCKREGION_X0Y6:CLOCKREGION_X0Y8` (108 RAMB36, no URAM:
  SLR1's URAM columns are X1 and X2). The TCMs are RAMB36 in every flow: `rtl/ctrl/viv/tsr_ctrl_tcm.xdc` (§4.2) is read
  by `core_ooc.tcl`, `kernel_ooc.tcl` and the kernel IP (`package_xo.tcl`).
- Keep the entire per-cluster attention job pipe `g_ajp.u_ajp` and the ATT/MVM selector beside
  its cluster. The selected job crosses through `g_jp` once. Reparented broadcast decode LUTs
  and carry cells belong beside the destination cluster; the lane wrapper's writer/reduction
  helpers belong in SLR1. Keep the stages on either end of the crossing pipes separate.

### 14.3 Timing gates (D13)

| Gate | Run | Criterion |
|---|---|---|
| T1 | out-of-context `tsr_ctrl_rv` on xcu200-2 | 6.0 ns met with ≥ 0.3 ns margin; Fmax measured (250 / 166.7 / 125 MHz choice, D12) |
| T2 | out-of-context k4 core with control core, VU, AT modes, DB, floorplan C''; preceded by the out-of-context `tsr_vu` run whose utilisation fixes the VU configuration against the §14.1 gate | WNS ≥ 0 at 3.333 ns (the core without the control core: +0.022 ns, `docs/rtl-m3.md` §14.3); SLR1 ≤ 65 % LUT |
| T3 | k4 v++ link (all M6 hardware) | **kernel clock ≥ 250 MHz** (user decision, 2026-11-02); setup and hold met at the accepted clocks; zero unconstrained `ap_clk` ↔ `ap_clk_2` paths; `report_cdc` clean |
| T4 | k1 v++ link | 300 MHz (bring-up) |

The user lowered the K4 acceptance floor from 276.3 MHz to 250 MHz on 2026-11-02. The running
attempt still requests 300 MHz and RV 166.7 MHz (Vitis records 166 MHz). Do not repeat an
implementation that meets the 250 MHz floor just to reach 300 MHz. The early clock/reset
gate verifies the requested clock constraints; the final gate checks the accepted scaled
frequencies. A route-only stop needs routed timing at an accepted core frequency, with RV,
hold and CDC checks, before timing success is claimed. This decision changes neither the
RV clock choice nor the T1, T2 and T4 criteria. The older 276.3 MHz card results remain
historical measurements.

**T1 measured** (W4, 2026-10-31; `rtl/ctrl/viv/ctrl_ooc.sh`, `tsr_ctrl` = `tsr_ctrl_rv` + CDC out of
context, xcu200-2, TCMs in 32K × 1 RAMB36 columns, default directives): clk_rv WNS at 6.0 ns: maxperf
+0.20 ns (an earlier run of the same flow +0.30 ns), limited by the single-cycle multiplier
(Ibex ID → `mult1_res` DSPs → WB register); maxperf with `RV32MFast` +0.31 ns (limited by Ibex's
taken-branch fetch address into the ITCM BRAM address); small +0.39 ns. At 4.0 ns maxperf misses by
1.74 ns, so 250 MHz is out of reach and the planning value 166.7 MHz stays. By §3.2's fallback
rule the RTL default is maxperf with `RV32MFast` (`tsr_ctrl.RV32M`). Core clock side of the CDC at
3.62 ns: +0.09 to +0.26 ns, the worst path from the dispatcher's `if_ready_i` (1.6 ns input budget)
into the issue FIFO's LUTRAM read. Cost: 8.5–8.7 K LUT, 13.5–13.6 K FF, 284 LUTRAM, 48 RAMB36,
4 DSP (1 with `RV32MFast`). `report_cdc`: 22 + 64 endpoints, all safe once the waivers of
`tsr_ctrl_cdc.xdc` (handshake data, FIFO LUTRAM, gray and level buses, the quasi-static `RV_BOOT` bus) are applied.

**CDC gate measured** (W4-B review, 2026-11-01; `tb/core/viv/core_ooc.sh k1 .. synth_only`, whole `tsr_core`
with the VU, the attention modes and the DB block, `clk` 3.333 ns and `clk_rv` 6.0 ns): on the integrated tree
`report_cdc` was not clean — 16 × CDC-10 (combinational logic before a synchroniser: the level bundle of
`tsr_ctrl_cdc` and `tsr_dbg_rv`'s `{rv_run, flags}`), 2 × CDC-11 (the core reset copy that feeds both core logic
and the reset synchronisers of the clk_rv domain), 1 × CDC-6 (the event toggles) and 161 × CDC-26 (the LUTRAM of
`tsr_dbg_rv`'s two FIFOs, outside the waiver by instance name) — because the first constraint file was scoped
to `tsr_ctrl`, named instances of `tsr_ctrl` only, and had never run on a whole core (the block-level runs see
the crossings' sources as ports). After the changes of §11.3: 45 Info endpoints (CDC-3 42, CDC-9 3), waivers
CDC-6 687, CDC-15 2,045 and CDC-26 548 endpoints, 1,464 `ASYNC_REG` cells, `set_bus_skew` on 630 endpoints
(slack +3.19 ns), both directions "Max Delay Datapath Only": gate PASS. The packaged kernel IP gives the same
result on its own clocks (`docs/platform-u200.md` §7).

The control-core RTL is frozen after T1–T3. Links run on the build host (14 GB + swap; the k4
link took 3 h 51 min, `docs/platform-u200.md` §8.2); T2 before every T3.

### 14.4 The gather job-pipe paths (cheap fix, part of T3)

The k4 link missed 300 MHz by 0.285 ns in the job pipes' central slice
`u_gather/g_asm[c].u_jp/g_slices.g_s[1]`: the central take (`sel_q` → `jv[sel_q]` → `take` →
`jr[c]`, `rtl/core/tsr_mvm_gather.sv:396-407`) is that slice's `m_ready`, and `tsr_rslice` turns
`m_ready` into the clock enable of its ≈ 577-bit `main_q` (`m_free`,
`rtl/core/tsr_mem_arb.sv:89-108`); the previous slice's `main_v_q` crosses the SLR into its
`s_fire`. Fix, in this order:

1. **`tsr_rslice` FORM 1** (two slots s0 / s1, write / read pointers, occupancy flops): slot i
   loads `s_data` whenever it is free (enable = its own occupancy flop), `m_data` = slot[rp];
   `m_ready` and `s_valid` drive only ≈ 5 control flops. It is **cycle-identical** at the ports
   (both are 2-entry FIFOs with registered `s_ready` = count < 2 and `m_valid` = count > 0), so every
   replay stays bit- and cycle-exact. Use it for the last stage of every job pipe and of every
   pipe whose `m_ready` is a wide take (result pipes, arbiter response slices where measured
   worst). Cost: one 2:1 mux level on `m_data`, no flops added.
2. Pin the job-pipe stage-0 registers to their assembler's pblock (1,288 drifted into SLR1 in the
   r3 k4 run, `docs/rtl-m3.md` §14.8).
3. Only if still short: `JOB_PIPE` 3 (one more cycle per MVM; results identical).

Expected (*estimate*): the −0.285 ns group disappears; the next limiters (arbiter / result slices
at +0.06 ns out of context) bring the link to 290–300 MHz, worth up to +4–5 % decode
(`docs/platform-u200.md` §14.2).

### 14.5 Cost of this area (U200, *estimate*)

| Block | Domain | LUT | FF | RAMB36 | DSP |
|---|---|---|---|---|---|
| Ibex (maxperf pipeline of §3.2) | rv | 3–5 K | 1.7–2.8 K | 0 | 3–4 |
| riscv-dbg `dm_top` + SBA arbiter | rv | 1–2 K | 1–1.5 K | 0 | 0 |
| ITCM 128 KiB + DTCM 64 KiB | rv | 0.1–0.3 K | — | 48 | 0 |
| bus, control window, LQ / LCQ / log RV side, timer, line buffer | rv | 2–3.5 K | 3–4 K | 0 | 0 |
| CDC (FIFOs, gray counters, handshakes) | both | 0.6–1.2 K | 1.5–2.5 K | 0 | 0 |
| `tsr_issue` + `tsr_chk` (6 queues ≈ 12 Kb LUTRAM, 12 dep compares, SYNC snapshot, 4-extent checker class) | core | 4–7 K | 3–5 K | 0 | 1–2 |
| `tsr_csr` (EX map, DMI, PERF counters, `Q_*` copies, LQ / LCQ / log rings, SR sequencer, loader with VERIFY) | core | 4–6 K | 3.5–4.5 K | 1 | 0 |
| less: logic of the measured core part that the control core replaces | core | −2 to −3 K | −2 to −3 K | 0 | 0 |
| **Total, net (without SPM / DMA changes)** | | **≈ 12.5–23 K** | **≈ 10–18.5 K** | **49** | **4–6** |

Ibex maxperf against small (CC-34): +22 % gates in Ibex's own table (32.48 against 26.60 kGE):
the writeback-stage registers and forwarding muxes, the branch target adder and the single-cycle
multiplier (three 17 × 17 products, i.e. three DSP48E2, instead of one multi-cycle MAC). On the
U200 that is ≈ +0.5–1 K LUT, +0.2–0.3 K FF and +2 DSP, +0.3–0.5 points of the SLR1 pblock (§14.1),
small against the gate's remedies (−8.7 K LUT for the VU knobs alone).

Fmax (*estimate*): clk_rv domain 150–220 MHz (planning 166.7 MHz); every core-clock addition is
built flop → ≤ 3 LUT levels → flop with registered boundaries, so it is not expected to set the
core clock (verified by T2). Maxperf adds two candidate clk_rv limiters, the writeback-stage
forwarding mux in front of the ALU operands and the single-cycle multiplier (unregistered DSPs and
a 34-bit add in one cycle); T1 measures both configurations and §3.2's fallback applies.

---

## 15. Verification

The oracle chain stays: Spike → the ISA simulator's RV32 model (`tsr_rv`, SW) and queue model → RTL
→ card. Bit-exactness of concurrent programs is defined as in `docs/ctrl-riscv.md` §9.5
(valid = race-free and schedule-independent; final DRAM / SPM / LCQ record stream / ticket error
records must equal the simulator's, SW §4.7).

### 15.1 Existing unit benches

`tb/dma` (extended: LD reservation, `rd_rsp_ready` = 1 asserted, KVST with read-modify-write
interleaved with LD / ST of the same channel, `sys_rtl_spm_contend` with six clients and the
deadlock case "LD and MVM on one channel while gather, QACT, VU and bridge saturate both write
ports"), `tb/fp`, `tb/mvm`, `tb/mvmback`, `tb/mvmfront`, `tb/qact` (through `tsr_spm_cport`),
`tb/spm` (rewritten for 2R2W: per-bank ordering, R2, round-robin bound NCL − 1, response latency,
write→read order, random multi-client traffic against a model), `tb/tech` (+ `tsr_sync` /
`tsr_rst_sync` / `tsr_clk_div` models).

### 15.2 New block benches

- `tb/chk`: `tsr_chk` against the simulator's check function (random and directed ops of every
  class incl. VU / AT ones, every error priority pair including `OPERAND`, the four tagged
  extents, boundary extents): code, info and command fields bit-exact.
- `tb/issue`: `tsr_issue` with unit models of random latency (units have no error output):
  credits; ticket wrap (counters preset near 2^28); `ISSUE_PIDX` = dispatcher pidx for every
  entry, with QABORT / QHOLD / TRACE_MARK interleaved; BAD_DEP (unpushed, own ticket, reserved
  unit, stale); DEP_FAILED; NOP / FENCE / QCLEAR; QABORT then pushes; host QCLEAR refused /
  accepted; holds (each source cannot clear another); DB `dbg_hold` / `dbg_step`, the hard hold and
  its release by QABORT and `SOFT_RESET`; the SR sequencer (steps 1–4, with clk_rv stopped,
  `SR_RV_TIMEOUT`); I1–I4 as assertions in every test (I3 ≤ 2 for MVM and VU); done → dependent
  start ≤ 3 cycles.
- `tb/ctrl` (cocotb + Verilator, `tsr_ctrl_rv` + `tsr_ctrl_cdc` + a core-domain stub): riscv-tests
  (rv32ui / um / uc / mi -p) and riscv-arch-test (RV32IMC, Zicsr, privilege subset) from ITCM,
  signatures against Spike; every control-window register; push safety net and timeout (and its
  suspension under `dbg_stop`); bridge widths, alignment, faults, bound; line-buffer invalidation on
  each listed event; SBA refusals over the whole control window and for TCM writes while `RV_RUN` =
  0; LQ / LCQ / LOG handshakes with both sides busy, `LQ_POP` / `LCQ_POST` faults, and the host
  writing `LCQ_RD` by +8 and `LOG_RD` by +256 while the runtime polls `LCQ_FREE` / `LOG_FREE`
  (never over-reported); load, `LOAD_DATA` reads and VERIFY through the TCM ports; DMI timeout then
  a late answer; core-window loads of 0x0C4 and 0xC00–0xDFC return 0; every CSR access with clk_rv
  stopped or held in reset (must complete, RV_CLK_OK = 0); RISC-V reset during a bridge access and
  during a waiting push; `RV_HALTED` from the fetch range, cross-checked against
  `dmstatus.allhalted`.

### 15.3 Op-list replay into the issue port (Verilator, `tb/opreplay`)

A C++ bench instantiates `tsr_core_dp` alone and drives its issue-entry stream (D15 forbids a host
path; this is a simulation port only, never in a bitstream). Inputs: SW's `ops` traces
(`push.bin` records {w0–w7, deps, push value, kind} + `init/` + `expect/` DRAM regions + expected
per-unit completion counts and error records, SW §5). Memory regimes of
`tb/fastreplay/memmodel.h` (rand / fast / slow / ordered). Schedule perturbation: random push gaps,
random host-hold pulses, random SPM contention from a phantom read client, random clock-enable gaps
per unit (simulation only). Sources (SW §5.6):
- `traces/model-v1conv`: the 290 model traces converted by `tsr-trace-conv` into serial chains
  (each op depends on the previous one), so the units' results must equal the recorded expectations;
- `traces/m2`, `traces/gen`, `traces/m3`, `traces/ctrl`, `traces/vu` and
  `traces/at` (free mode: ≤ 2 deps per op from the generator, SYNC NOP joins, error injection);
- M6 op-list traces captured from the simulator (full decode tokens of Qwen2.5-0.5B, FA tiles).
Pass: every run bit-exact (DRAM expect regions, SPM lines written, counts, records), in every
memory regime, at `RST_PIPE` 0 and 3, with the generic and the Xilinx filelists.

### 15.4 Full system with the RISC-V

- **Lockstep co-simulation**: Ibex RVFI (simulation build, *confirm* field names) against `tsr_rv`
  per retired instruction; loads of timing-dependent registers (`DONE_SEQ`, `ISSUE_CREDIT`, MTIME,
  `QSTATUS`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`, error records, core-window PERF) and trap /
  interrupt entries take the RTL value and position; everything else (including `ISSUE_PIDX` and
  `ISSUED_SEQ`) must match. The simulator applies tile-op effects at the RTL's completion times, so
  SPM / DRAM compare at any point.
- **Two clocks**: independent `clk` / `clk_rv`, ratios F_RV / F_CORE ∈ {0.31, 0.42, 0.55, 0.6,
  0.83, 1.0, 1.67} with random phase, plus the `CTRL_SYNC` = 1 build; every test at every ratio,
  including a directed error test (`error(t)` consistent with `done(t)`).
- **CDC jitter mode** (CC-27): in simulation `tsr_sync` randomly takes 2 or 3 stages per
  transition and per bit, which emulates bus skew; gray-code and handshake correctness must hold.
- **Random halts**: DB's halt + HOLD_ON_HALT at random points of stress programs; at each
  quiescent point compare with the simulator advanced to the same retired count and completion
  vector, given the co-simulation log.
- **Concurrency stress**: random race-free programs from SW's generator (checked by the race
  detector), run on RTL and simulator under random memory latency; final state bit-exact; at least
  8 random simulator schedules per program for the determinism test.
- **vkernel** with two clocks: llama.cpp end to end (Qwen2.5-0.5B Q8_0) bit-identical against
  `TSR_DEVICE=sim`.

### 15.5 Mutation campaign

Drop a dep compare; complete a non-running entry before its deps; reorder a queue; skip a poison;
advance D[u] before the record acknowledge; apply a RISC-V QCLEAR out of band; step a gray counter
by 2; one hold source clearing another; credit returned at done instead of accept; round robin back
to fixed priority on one bank port; a CDC FIFO full flag one cycle late; the line buffer not
invalidated on one event; a unit started in the cycle it becomes head; `tsr_rslice` FORM 1 data
enable including `m_ready`; `ISSUE_PIDX` advanced by a QHOLD; `LCQ_FREE` computed from a
gray-coded `LCQ_RD`; the hard hold kept after a QABORT; the SR sequencer skipping step 2. Each must
be killed by the benches above.

### 15.6 Vivado

T1–T4 of §14.3; `report_cdc` on every out-of-context run; `report_timing` between `ap_clk` and
`ap_clk_2` on the linked design; utilisation per SLR against §14.1.

---

## 16. Interfaces with other areas

| Area | CC provides | CC assumes from the area |
|---|---|---|
| **EX** (execution model) | issue port, credits, tickets, `ISSUE_PIDX`, ≤ 2 hardware deps, SYNC NOP / FENCE / QCLEAR, I1 / I2, error states and records, holds, QIDLE / QQUIESCENT; unit ids 0–5, queue depths (`Q_DEPTH`), latencies of §7.6 / §9.3; the RTL of EX's LQ / LCQ / log rings with handshake CDC, the SR sequencer, the loader with VERIFY; MTIME; line-buffered SPM alias | the opcode table and the static `UNIT_OF` mapping (EX §7.2–§7.3); the host CSR map (EX §5); the memory model as an architectural rule (this file implements rules 2(a)–(e), 3, 4 of `docs/ctrl-riscv.md` §5.5); error-code list (1–8, 11–13) and check order; CAPS bits; the launch protocol (EX §4) and `SOFT_RESET` semantics (EX §8.5) |
| **VU** | queue 3 (depth 16), dispatch `cmd_valid / cmd_ready / cmd / done` with accept-ahead (I3 ≤ 2), SPM client 4 with both bank ports (read pairs, one write per cycle), checker class slot with four tagged extents, placement next to `tsr_spm` in SLR1, `SREG` read path (0x9C0) into `tsr_csr` | per opcode (including AT's 0x34–0x37): reserved masks, used words, sub legality, `OPERAND` conditions, ≤ 4 extent formulas, the command struct; the s-register bank (16 × 32 b flops) exported; no run-time errors; resource estimate for §14.1 (VU §7.10) |
| **AT** | KVST on the DMA queue, ATTQK / ATTPV on the MVM queue (depth 4), their checker classes; SPM client 5 and unit 4 tied off in M6 (`HAS_AT` = 0); memory source 0 (`TSR_SRC_AUX`) tied off; PERF X + 16 … X + 19 wiring | its ops' check rules and command structs (AT §5.8), its cluster-mode RTL (AT §8) and SLR placement (SLR0 / SLR2 with the clusters) |
| **DB** | DM instance (Debug Spec 0.13.2) with DMI at CSR 0xC00–0xDFC gated by `DMI_EN`, SBA master on the RISC-V bus (all control-window writes refused), `RV_HALTED`, `ndmreset` → reset sequence, HOLD_ON_HALT, per-unit `dbg_hold[u]` / `dbg_step[u]` inputs with the hard-hold rule of §9.2, head info (`head_valid`, `head_pidx`, `head_w0`, head state, hold sources) one cycle before a possible start, per-unit start / completion strobes {seq, pidx, code, ran} with MTIME (non-running completions included), the enqueue tap {pidx, ticket, unit, code, w0–w7, deps, REC, push PC, four tagged extents}, the push PC in the issue FIFO, TRACE_MARK entries, the REC push flag, `ISSUE_WAIT_MAX` suspension under `dbg_stop`, the second write source on `TRW_CH` = 0, per-channel outstanding counts, the `SREG` wiring; reserved windows (control window 0x160–0x17C and 0x400–0x7FC; CSR 0x800–0xAFC; `irq_fast[4]`, `[6]`; PERF 61–63) | the GDB RSP server on DMI (and that it never writes the issue staging or `QCTRL` through the program buffer: handlers and debugger code do not issue), record / trace formats, tile-level debug semantics, the log record format |
| **SW** | the table rows of §13.4, boot / load sequence, timeouts, `xrt::ip` control model (`ap_ctrl_none`), `tsr_core_dp` replay bench, two-clock vkernel | `tsr_rv` (RV32IMC + Zicsr, Ibex CSR behaviour), the queue model with random schedules and the race detector, the trace converter and the trace format, the table generator, the SDK (crt0, linker script for §4.1, `issue()` checking credits, the runtime R1–R10), the transport (`xrt::ip`, polling `LCQ_WR`) |

---

## 17. Decisions

| Id | Topic | Choice | Alternatives | Rationale |
|---|---|---|---|---|
| CC-1 | Ibex pin | OpenTitan's locked Ibex commit at vendoring, unmodified; RV32IMC, `RV32MFast`, FPGA register file, no PMP / cache / predictor / WB stage, 2 triggers (pipeline amended by CC-34: WB stage, branch target ALU; the multiplier stays `RV32MFast` after T1) | Ibex master tip; VexRiscv; CV32E40P | The OpenTitan lock is a state that a silicon sign-off flow ran on, which matters more than features; the configuration is the verified "small" one plus two documented deviations (register file variant, triggers). VexRiscv's Scala source and CV32E40P's FPU are not needed (D1, no F). |
| CC-2 | Debug module | upstream riscv-dbg release, `dm_top` without its JTAG DTM, DMI from the CSR window, SBA connected | OpenTitan's patched `rv_dm`; own DM | D18 needs a standard DM; GDB-over-PCIe (DB) speaks DMI directly, so the JTAG DTM is dead weight; upstream avoids OpenTitan's TL-UL wrapper. |
| CC-3 | TCMs | ITCM 128 KiB, DTCM 64 KiB, RAMB36 32K × 1 columns, read latency 1 | 64 + 64 KiB; URAM | M7's graph executor, two model families, M8 MoE, sampling and debug formatting must fit without overlays; 16 more RAMB36 of ≈ 351 free is cheap. URAM is kept for the SPM (32 free). Columns avoid an output mux at read latency 1. |
| CC-4 | Address map | ctrl-riscv map + 0x5000_0000 line-buffered SPM alias | only the exact window | Op lists and argument blocks are read word by word; the alias makes that ≈ 7× faster with an invalidation rule that keeps the memory model. |
| CC-5 | Issue port | memory-mapped staging (8 words, 2 deps, push with word mask), credits checked by software, hardware wait as a 65,536-cycle safety net (suspended under `dbg_stop`) | custom instruction; blocking push | Ibex stays unmodified; a blocking push would block interrupts and debug halts for up to a 32B FFN MVM (≈ 1 ms). |
| CC-6 | Unit ids | 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT engine (absent in M6), 5 SYNC | id 4 unused, AT at 6 | Using id 4 keeps six units, six counters and the CDC widths of the plan. |
| CC-7 | Opcode → unit | static table generated from EX §7.2 (EX-30), `ISSUE_UNITS` mask and `Q_DEPTH` | QMAP register | Software has the table at compile time; one mapping fewer to keep consistent. EX §7.3 adopts it. |
| CC-8 | Queue depths | DMA 8, QACT 4, MVM 4, VU 16, SYNC 4 (unit 4 absent in M6; parameter 8); FIFO 32 | 8 / 4 / 4 / 8 | Many short VU ops per layer (norms, softmax steps, RoPE) want a deeper VU queue; LUTRAM cost is ≈ 100 LUT. Parameters, re-tuned from M7 profiles. |
| CC-9 | Checker | one `tsr_chk`, 8 cycles / op, check order with `OPERAND`, four tagged extents per class, MVM extent by shift + DSP for power-of-two `NUM_CH` | pipelined checker; keep the 46-cycle serial MVM path; 3 SPM + 1 DRAM extents | 34 M ops/s is ≥ 3× the RISC-V's issue rate; the DSP path removes the 46-cycle MVM checks that would gate short MoE expert MVMs (M8); VOP, ROPE, AMASK and AMERGE have four SPM operands. |
| CC-10 | Dispatch | queue-head command registers drive the units directly; an entry never starts in the cycle it becomes head | 0-cycle head bypass | Keeps unit clock enables one flop away (k4t1 worst-path structure) and gives DB one cycle to hold a breakpointed entry; the cycle is hidden while the unit is busy. |
| CC-11 | Runtime unit errors | units have no error output; every error is found at dispatch (EX §7.2) | optional `err` code at done (the draft's choice) | EX's rule "no op may fail after it has started" keeps the first error schedule-independent and the head logic simple; VU and AT define data-dependent cases as values. |
| CC-12 | SPM banking | even / odd line interleave, 2R2W, per-bank client ports, registered round robin, 128 URAM | duplicate the array (no URAM room); depth banking; combinational per-line ready | Interleave gives two reads and two writes per cycle at no URAM cost; line pairs always hit both banks, so the VU streams two operands at full rate; per-bank ports keep R5 (ready from flops). |
| CC-13 | DMA LD | burst reservation, 512-beat queue (8 RAMB36), `rd_rsp_ready` = 1 | rely on fixed SPM priority; response skid in `tsr_mem_arb` | Priority is keeping DDR busy: with round robin and a VU writing, a DMA queue could fill and stall the head-of-line response path, starving MVM weight beats; reservation removes the coupling for 8 RAMB36. |
| CC-14 | s registers | one bank in the VU, read-only; host and RISC-V read it at `SREG0`–`15`, core CSR 0x9C0–0x9FC (RISC-V via the core window 0x2000_1000 + 0x9C0); writes only through ops | RISC-V-writable s registers; `SREG` at 0x0C0 (the draft; collides with EX's `LOAD_*`) | D14 leaves one bank; writes stay ops (SOP.MOVI) so they are ordered by queue order and deps; one location for every reader. |
| CC-15 | Core window | read-only view of the whole CSR space, no snapshot side effects; loads of `LOAD_DATA` and of the DMI window return 0 without access | a few copied registers | One mechanism gives the RISC-V ID / CFG, s registers, PERF and records; no new register copies; the RISC-V cannot disturb a load or race the host debugger. |
| CC-16 | Debugger memory path | DM SBA on the RISC-V bus; no host SPM window client | prof-dbg's client-3 debug window | The SBA path reaches TCMs, SPM and the read-only registers without halting; the SPM keeps six clients instead of seven; control-window writes are refused to SBA (D15, CC-33). |
| CC-17 | Log | EX's 256-word core-domain ring (drop-new, never torn, DB's header); CC's part: `LOG_PUSH` → 16-entry CDC FIFO, and `LOG_FREE` from a `clk_rv` push count and a handshaked `LOG_RD` | the draft's 1,024-word overwrite-oldest ring with windowed reads; DRAM log ring | Works when queues wedge; never stalls the RISC-V by default; does not consume issue credits; whole records need no resynchronisation (DB-12). |
| CC-18 | Clocks | `ap_clk_2` 166.7 MHz via v++ clock request, BUFGCE_DIV ÷ 3 of 500 MHz as fallback; asynchronous CDC | `ap_clk` / 2 synchronous | D5 / D12; independent of core-clock scaling; the CDC design of ctrl-riscv is kept. |
| CC-19 | Kernel control | `ap_ctrl_none` per EX §9 (fallback: declare `ap_ctrl_hs`, never run); `xrt::ip` and polling; no interrupt on the U200; `MEM_EN` latch | `user_managed` (the draft; not in XRT 2.18's control types); `ap_ctrl_hs` with runs | A halted RISC-V under an outstanding `xrt::run` breaks KDS; the pinned XRT routes `AP_CTRL_NONE` to `xrt::ip`, which is the register-level path the host needs. |
| CC-20 | ID / VERSION | `ID` is 0x5453_5231 "TSR1", `VERSION` 0x0000_0100: Tesseract Core capability 1.0 (EX-12, SW §6.9) | "TSR2" (the draft) | Host open code refuses a different capability major, so a tool built for another major fails cleanly without a new magic. |
| CC-21 | PERF | EX §6 (ACTIVE / QHALT gating, zeroed by `SOFT_RESET`, X = 5 + 2C); CC adds X + 20 / X + 21 SPM wait counters | 64 free-running fixed slots with `PERF_CLEAR` (the draft) | ACTIVE gives the property "PERF does not change between executions" per launch with no software bit; one counter space for the whole device. |
| CC-22 | Recovery | no partial datapath reset and no op cancel; `SOFT_RESET` is EX's device-level sequence (SR sequencer, CC-31); a wedged unit needs a card reset | a datapath soft reset; per-unit kill | Resetting units with outstanding memory transactions breaks the "no pre-reset response" contract of the arbiters; the log ring, `Q_*` and DB's evidence give the post-mortem first. |
| CC-23 | Timing gate | k4 link ≥ 250 MHz with all M6 hardware | Requiring 276.3 or 300 MHz for K4 acceptance | D13 amended by the user's 2026-11-02 decision. Keep the current 300 MHz request; do not repeat a qualifying implementation only to reach it. |
| CC-24 | Floorplan | C'': control core, issue path, VU and DB next to the SPM in SLR1; SLR1 ≤ 65 % LUT after synthesis, else in order: VU shrink knobs, DB `HAS_TRW` / `HIST_DEPTH` off, VU lanes to SLR0 behind pipes | VU in SLR0 / SLR2 from the start; "AT moves out" (the draft; AT has no SLR1 unit) | The VU needs 1.5 Kb per cycle to and from the SPM; the recomputed budget (§14.1, ≈ 62–71 %) straddles the gate, and every remedy keeps results unchanged. |
| CC-25 | Gather fix | `tsr_rslice` FORM 1 (cycle-identical) + pinning stage 0 | `JOB_PIPE` 3 first; registered take | Removes the measured worst paths without changing any cycle, so all replays stay exact; deeper pipes only if needed. |
| CC-26 | Memory source 0 | `TSR_SRC_AUX`, tied off in M6 | delete the source | Keeps the arbiter shape; AT streams KV through the clusters' weight path, so nothing claims it yet. |
| CC-27 | CDC verification | Verilator two-clock benches with ratios and a jitter mode in `tsr_sync`, plus `report_cdc` | formal CDC tool | Verilator cannot model metastability; randomised synchroniser latency per bit exercises skew; Vivado's structural check covers the rest. |
| CC-28 | Register table | CC's rows in the single table source `sw/tables/tsr_tables.py` (EX-30) | `rtl/core/ctrl/tsr_ctl_map.py` (the draft) | One source for every area; the RTL, simulator, SDK and benches must agree bit for bit; `--check` in lint catches drift. |
| CC-29 | `RV_HALTED` | derived from the fetch address being in the DM range; cross-checked against `dmstatus` in `tb/ctrl` | modify riscv-dbg | Keeps the DM unmodified; the debug ROM, program buffer and abstract commands all run there; the DMI fallback exists if the vendored DM differs. |
| CC-30 | LQ / LCQ / log CDC | the RISC-V reaches EX's rings only by tagged handshakes (reads return the core-domain value; pop and post commit in the core domain); log words by a CDC FIFO with `LOG_FREE` from a `clk_rv` push count and a handshaked `LOG_RD` | gray-coded counter copies; periodic snapshots | The host-written `LCQ_RD` / `LOG_RD` and the `SOFT_RESET` flushes jump by more than one step, which a gray crossing cannot carry; a handshake per read is exact and costs ≈ 50–70 ns, which the runtime pays a few times per launch. |
| CC-31 | SR sequencer | a core-domain sequencer for EX §8.5 (RISC-V reset with a 2^16-cycle bound, drain, clear list, `SR_BUSY`), independent of `clk_rv` | a host-scripted sequence; reset of the clk_rv domain only | The host needs one atomic "known state" operation that cannot be raced by the RISC-V and completes when `clk_rv` is absent. |
| CC-32 | `ISSUE_PIDX` | a `clk_rv` counter of ticketed pushes at control-window 0x03C, equal to the dispatcher's pidx | a software push counter initialised from `Q_PIDX`; reading `Q_PIDX` | Exact and timing-independent like `ISSUED_SEQ` (only the RISC-V's own pushes move it), costs ≈ 40 FF, and survives firmware that pushes outside the SDK's counter; `Q_PIDX` lags by the issue FIFO. |
| CC-33 | SBA write refusal | every SBA write to the control and core windows is refused in hardware, and SBA TCM writes while `RV_RUN` = 0 | refuse only issue staging, push and `QCTRL` | `LQ_POP`, `LCQ_POST`, `LOG_PUSH` and `TRACE_MARK` have protocol effects; a whole-window rule is simpler to verify than a list and matches the debug server's policy. |
| CC-34 | Ibex pipeline (2026-10-30, executor optimisation; amends CC-1) | Ibex "maxperf" pipeline: `WritebackStage` 1, `BranchTargetALU` 1, `RV32MSingleCycle` (T1 took the fallback `RV32MFast`, 2026-10-31, §14.3); the TCMs take one data access per cycle (§4.3); `tsr_rv` models the pipeline with the single-cycle multiplier and is the simulator default. Fallback by T1: `RV32MFast`, then the configuration with the lower time per op record (§3.2) | "small" (CC-1); maxperf with `RV32MFast`; a Rocket core (rejected by the user, 2026-10-30) | Measured on the simulator (§3.2): −21.8 % RISC-V cycles per op record with the W2a firmware (1,977 → 1,547) and −16 % with the optimised executor (520 → 437), launch entry 2.19 → 1.68 µs; ≈ +0.5–1 K LUT, +2 DSP (+0.3–0.5 points of SLR1, §14.5). The single-cycle multiplier is worth ≤ 1.3 % (0 on the optimised executor) but is the verified maxperf / OpenTitan form; it goes first if T1 shows it on the critical path. |

---

## 18. Risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| R1 | Ibex + TCMs do not close 166.7 MHz on -2 (maxperf: forwarding mux and single-cycle multiplier, CC-34) | lower issue rate | §3.2's fallback (`RV32MFast`, then the configuration with the lower time per op record); 125 MHz fallback still ≈ 3–8 M ops/s; T1 decides early |
| R2 | SLR1 congestion (control core + VU + DB + SPM muxes: ≈ 62–72 % LUT in §14.1, up to 79 % with the VU at + 30 %) | missed T2 / T3 | budget gate CC-24 and its ordered remedies; out-of-context VU run before T2 |
| R3 | SPM 2R2W muxes (6 clients) land on the SPM's historical worst paths | core clock below 276 MHz | per-slice kept registers kept; client request registers next to the SPM; T2 measures; fallback: `SPM_RD_LAT` 5 (latency only) |
| R4 | `ap_clk_2` frequency request or `ap_ctrl_none` with `m_axi` args not accepted by v++ 2024.2 | build / host rework | BUFGCE_DIV fallback; EX §9.1 fallback (`ap_ctrl_hs`, never run); checked in the first smoke link |
| R5 | 4-hour k4 links on a 14 GB host | slow iteration | OOC T2 first, k1 links for the control core, at most 2–3 k4 links in M6 |
| R6 | CDC bug not seen in Verilator | rare hangs / corruption on the card | jitter mode, ratio sweep, `report_cdc`, assertions on gray steps and handshakes; selftest at several F_RV |
| R7 | Issue-bound phases (short VU ops in small models, the `graph` executor's per-op cost) | RISC-V becomes the bottleneck, DDR idles | staging words kept across pushes (15-cycle loop issue); VU / AT ops cover all heads per op; measured with MTIME, `PUSH_STALL` and `MVM_EMPTY` |
| R8 | `RV_HALTED` heuristic wrong for a DM corner (e.g. abstract command outside the range) | wrong hold / quiescence | DB verifies against riscv-dbg's layout; `tb/ctrl` cross-check; fallback: read `dmstatus` via DMI |
| R9 | A wedged unit (no cancel) | only a card reset recovers | progress watchdog, log ring, `Q_*` and DB evidence first; decide a forced release with DB if the card shows it |
| R10 | DDR penalty of LD / ST interleaved with weight streams | overlap gains cancelled | measure with `tsr_bw` mixed traffic; SDK orders ST after same-channel MVMs |

---

## 19. Open items (confirmations, not decisions)

- Pinned Ibex and riscv-dbg commits; their exact parameter / port names (§3, *confirm*), and
  whether the program buffer and abstract commands always execute in the DM range (the
  `RV_HALTED` heuristic; fallback `dmstatus` over DMI).
- v++ option form for the `ap_clk_2` frequency; whether Vitis 2024.2 packages an RTL kernel with
  `m_axi` arguments as `ap_ctrl_none` and honours its `sp=` connectivity (first smoke link; EX §9.1
  fallback otherwise).
- The out-of-context VU utilisation, which fixes the VU configuration against the 65 % gate
  (§14.1); the measured AT cluster-mode cost in SLR0 / SLR2.
- Measured `tsr_ctrl_rv` Fmax (T1) fixes F_RV per D12, for both Ibex configurations (§3.2 fallback,
  CC-34); the `tsr_rv` cycle tables' *confirm* items (CSR flush, 32-bit target fetch, trap
  entry) against the RVFI bench (SW P5.2).

# CC: RISC-V control core, issue path, dispatcher, CDC, floorplan

Area CC specifies the RTL of the RISC-V control subsystem, the issue path, the SPM arbitration, the clock
domain crossings, resets, the host CSR block, the platform adapter's control and clock interface, and the U200
floorplan and timing plan. It implements D14, D15 and D18 and is built around D1–D6 and D12
([spec](../spec.md) §16): the host does not issue tile ops, the host has no debug access to the issue port, and
the design has full debug. The other areas are EX (execution model), VU, AT, DB (debug) and SW (software and
verification). Numbers marked estimate are design figures.

---

## 1. Scope

Area CC owns, in RTL: the RISC-V control subsystem (Ibex, debug module instance, TCMs, buses, control window),
the issue path (issue FIFO, dispatcher, checker, per-unit queues, queue heads, SYNC unit, tickets, error
states), the SPM arbitration and banking, every clock-domain crossing, resets, interrupts and timer, the host
CSR block (`tsr_csr`), the platform adapter's control and clock changes, the U200 floorplan and timing plan,
and the RTL verification of all of it.

The design choices of this area are CC-1 to CC-34 (§17): the RISC-V is the only issuer through the issue port
(D14); `tsr_chk` checks every op in front of the dispatcher (§8); there is one s register bank s0–s15 inside the
VU (§5.3); the host CSR map is EX §5 only, and CC's registers sit inside it (§6); `SOFT_RESET` is run by CC's
core-domain SR sequencer without `clk_rv` (§12.2); the SPM has 2R2W even and odd line banks with round-robin
client ports (§10); the core clock and `clk_rv` are separate and asynchronous, and the release image runs 200
MHz and 100 MHz (§11.1).

Not in this area: the meaning of each tile op (EX, VU, AT), the memory model as an architectural rule (EX; §9
implements it), the host CSR map, the launch protocol (LQ and LCQ) and `SOFT_RESET` semantics (EX), the kernel
ABI (SW), GDB, completion-record formats and tile-level debug semantics (DB), the ISA simulator's RV32 model,
queue model, race detector and trace converter (SW). CC implements in RTL what EX and DB specify for the core.

---

## 2. Structure

### 2.1 Block diagram

```
 clk_rv domain (control Pblock, SLR0)               ¦  core clock domain (ap_clk)
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

| Module | File | Domain | Placement (release image) |
|---|---|---|---|
| `tsr_core` | `rtl/core/tsr_core.sv` | both | |
| `tsr_ctrl_rv`: Ibex, `dm_top`, TCMs, `tsr_rv_bus`, `tsr_ctl_win` (with the line buffer) | `rtl/ctrl/` | clk_rv | control Pblock, SLR0 |
| `tsr_ctrl_cdc`: every crossing of §11; `tsr_ctrl` = `tsr_ctrl_rv` + `tsr_ctrl_cdc` + reset synchronizers, the instance `tsr_core` takes | `rtl/ctrl/tsr_ctrl_cdc.sv`, `rtl/ctrl/tsr_ctrl.sv` | both | control Pblock, SLR0 |
| `tsr_core_dp`: the whole core-clock datapath; the op-list replay bench instantiates it alone (§15.3) | `rtl/core/tsr_core_dp.sv` | core | |
| `tsr_issue`: dispatcher, queues, heads, SYNC, counters, error states | `rtl/core/tsr_issue.sv` | core | control Pblock, SLR0 |
| `tsr_chk` | `rtl/core/tsr_chk.sv` | core | control Pblock, SLR0 |
| `tsr_spm_bridge`: RISC-V SPM client | `rtl/core/tsr_spm_bridge.sv` | core | SLR1, next to `tsr_spm` |
| `tsr_csr` | `rtl/core/tsr_csr.sv` | core | control Pblock, SLR0 |
| `tsr_spm` (2R2W), `tsr_dma`, `tsr_qact`, `tsr_mvm` | `rtl/core/` | core | SPM, DMA and QACT in SLR1; clusters in SLR0 and SLR2 |
| `tsr_cdc_fifo`, `tsr_cdc_gray`, `tsr_cdc_hs` | `rtl/ctrl/cdc/` | both | |
| `tsr_sync`, `tsr_rst_sync`, `tsr_clk_div` | `rtl/tech/<target>/` | | |
| VU (`tsr_vu`, including AT's vector ops 0x34–0x37); AT's cluster modes (ATTQK, ATTPV) and KVST in the DMA | areas VU and AT | core | VU in SLR2; cluster modes with the clusters in SLR0 and SLR2; KVST in SLR1 (§14) |

---

## 3. Control core: Ibex and the debug module

### 3.1 Pinning

- Ibex: `lowRISC/ibex`, vendored unmodified into `third_party/ibex` at a commit that OpenTitan's sign-off flow
  has run, together with the `vendor/lowrisc_ip` `prim_*` files that Ibex itself vendors. The commit, the
  Verilator build check and the configuration's verification grade are recorded in `rtl/ctrl/VENDOR.md`.
- Debug module: `pulp-platform/riscv-dbg` (Debug Spec 0.13.2), a release tag, unmodified, in
  `third_party/riscv-dbg`, with only the `common_cells` file `dm_top` needs (`fifo_v3`, from
  `third_party/common_cells`). The JTAG DTM (`dmi_jtag`) is not used: the DMI arrives from the CSR window
  (§11).
- Both keep their own style (asynchronous active-low resets, `prim_*`), with one Verilator waiver file per
  tree; `prim_clock_gating` binds to `rtl/tech/<target>/tsr_clk_gate.sv` (pass-through on the FPGA). Licenses
  (Apache-2.0, SHL-0.51) are recorded beside them.

### 3.2 Configuration (`ibex_top` parameters)

| Parameter | Value | Why |
|---|---|---|
| `RV32E` | 0 | 16 registers would cost code size |
| `RV32M` | `RV32MFast` | MUL 3 cycles, MULH* 4, DIV and REM 38 (2 by zero) (CC-34). With `RV32MSingleCycle` the multiplier path limits `clk_rv`; the simulator's `tsr_rv` cycle table models the single-cycle multiplier (MUL 1, MULH* 2), a difference of at most 1.3 % RISC-V cycles on the executor |
| `RV32B` | `RV32BNone` | Zba and Zbb stay an SDK-measured option |
| `RegFile` | `RegFileFPGA` (LUTRAM); flops on an ASIC (tech layer) | |
| `WritebackStage`, `BranchTargetALU` | 1 / 1 | Ibex's "maxperf" pipeline (CC-34): loads and stores leave ID/EX after the request, taken branches 2 cycles instead of 3; still in order, no prediction, no cache (D1) |
| `ICache`, `ICacheECC`, `BranchPredictor` | 0 | D1 |
| `PMPEnable` | 0 | machine mode only |
| `MHPMCounterNum` | 0 | `mcycle` and `minstret` suffice; MTIME for timestamps |
| `DbgTriggerEn`, `DbgHwBreakNum` | 1, 2 | two hardware breakpoints (D18) |
| `SecureIbex`, `ICacheScramble` | 0 | |
| `DmHaltAddr`, `DmExceptionAddr` | 0x1000_0800, 0x1000_0808 | riscv-dbg layout at base 0x1000_0000 |
| `hart_id_i` | 0 | one hart |
| `boot_addr_i` | `RV_BOOT` (EX §5.1, 256-byte aligned) | first fetch at `RV_BOOT` + 0x80, vectored `mtvec` there |
| `fetch_enable_i` | constant on | the run and stop control is the reset (§12) |

ISA: RV32IMC + Zicsr + Zifencei, machine mode only, no A, no F or D. Spec-fp scalars go through the `tsr_ref`
helpers in soft-float, and firmware does no f32 arithmetic (SW-8).

Pipeline (CC-34). The pipeline is Ibex's "maxperf" one (RV32IMC with a branch target ALU and writeback stage),
with the other parameters of this table. "maxperf" itself has `DbgTriggerEn` 0 and `RegFileFF`; this design
keeps two triggers, LUTRAM and, if the pinned commit has `RV32ZC`, Zca only (the ISA stays RV32IMC). Cycle
costs (`tsr_rv` models both configurations, SW §4.2):

| Class | maxperf | small |
|---|---|---|
| ALU, LUI and AUIPC, fence, not-taken branch, CSR read | 1 | 1 |
| CSR write other than `mscratch` and `mepc` | 2 | 2 |
| taken branch | 2 | 3 |
| JAL, JALR, `fence.i` | 2 | 2 |
| load or store, TCM | 1 | 2 |
| load whose rd the next instruction reads | + 1 | |
| load or store, window (§5, §10.5) | latency | 1 + latency |
| MUL, MULH* (single-cycle multiplier) | 1 / 2 | 3 / 4 |
| DIV*, REM* (by zero) | 38 (2) | 38 (2) |
| jump or taken branch to a 32-bit instruction at pc[1] = 1 | + 1 | + 1 |

With the writeback stage Ibex issues the next data request in the cycle the previous response returns, so the
data bus (§4.3) takes a TCM access every cycle: `outstanding_q` holds `data_gnt` low only behind an access
outside the TCMs (TCM responses always come in the next cycle). This and §4.3's one-TCM-access-per-cycle rule
carry most of maxperf's gain over "small".

`dm_top` configuration: `NrHarts` 1, `BusWidth` 32, `DmBaseAddress` 0x1000_0000 (which needs its two-scratch
debug ROM, `HasSndScratch`), `SelectableHarts` 1; its slave port on both Ibex ports (2:1, data first, the fetch
stalls through `instr_gnt`); its system-bus master (SBA) connected to the RISC-V data bus (§4.3, CC-16);
`ndmreset_o` starts the RISC-V reset sequence (§12.2).

Halted status. `dm_top` does not export a per-hart halted level. `RV_HALTED` is derived in hardware: the
address of the last granted instruction fetch lies in the DM range 0x1000_0000–0x1000_0FFF (debug ROM park
loop, program buffer and abstract-command buffer all execute there). It feeds `STATUS.RV_HALTED` (EX §5.3),
the debug-halt hold source (§9.6) and `QQUIESCENT`. `tb/ctrl` cross-checks it against `dmstatus.allhalted`; if
the vendored riscv-dbg runs anything outside that range, the fallback reads `dmstatus` over the DMI.

### 3.3 Clock and issue throughput

- `clk_rv` is chosen by the implementation and reported by `RV_FREQ` (D12). The release image runs it at 100
  MHz (CC-23). The cycles per op record of the RISC-V barely depend on the clock ratio (the per-op path has few
  crossing accesses), so the RISC-V time per op scales with the period.
- Throughput need (estimate): a decode token of a 24-layer model is about 1.5 K tile ops (24 layers × about
  60), a 32B token about 4 K; issue cost per op is 15–40 RISC-V cycles (§7.6). VU's row × stride operand model
  (VU-1) and AT's group-range ops (AT §9.3) cover all heads per op, which keeps the op count at tens per layer.

---

## 4. RISC-V address map, TCMs and buses

### 4.1 Address map (32-bit byte addresses; everything not listed is an access fault)

| Range | Region | Fetch | Load / store | Notes |
|---|---|---|---|---|
| 0x0000_0000–0x0001_FFFF | ITCM, `ITCM_BYTES` = 128 KiB | yes | byte / half / word | data reads steal the fetch port for one cycle (`instr_gnt` = 0); stores then `fence.i` |
| 0x0010_0000–0x0010_FFFF | DTCM, `DTCM_BYTES` = 64 KiB | no | byte / half / word | .rodata, .data, .bss, stack |
| 0x1000_0000–0x1000_0FFF | debug module memory | yes | word (byte / half too in debug mode) | riscv-dbg layout; its debug ROM reads its flags with `lbu` |
| 0x2000_0000–0x2000_0FFF | control window (§5) | no | word only | clk_rv registers |
| 0x2000_1000–0x2000_1FFF | core window (§5.3) | no | word loads only (store = fault) | read-only view of the core CSR space |
| 0x4000_0000–0x403F_FFFF | SPM window, exact (`SPM_BYTES`) | no | byte / half / word | 0x4000_0000 + 64 × line + offset, little-endian |
| 0x5000_0000–0x503F_FFFF | SPM window, line-buffered alias (§10.5) | no | byte / half / word | loads may hit the line buffer; stores write through and invalidate it |

Region sizes are parameters; every address in a region beyond its size faults. The ITCM range is reserved up to
0x000F_FFFF and the DTCM up to 0x001F_FFFF, so larger TCMs keep the map.

### 4.2 TCMs

- Size: ITCM 128 KiB, DTCM 64 KiB (CC-3). `RV_CFG` (EX §5.1) reports them.
- Implementation: `tsr_sram_1r1w` with byte enables. Read port = fetch, data-port read, or the load port (EX's
  `LOAD_DATA` read and the VERIFY reader) while `RV_RUN` = 0; write port = data-port stores (Ibex or SBA) while
  `RV_RUN` = 1, or the host load port while `RV_RUN` = 0, never both. Read latency 1, no output register (Ibex
  wants the word the cycle after `gnt`). The Xilinx mapping is RAMB36 in 32K × 1 columns (no output mux): ITCM
  32 RAMB36, DTCM 16 RAMB36, 48 in all. Vivado's default inference would build 8-high and 4-high cascades with
  the fetch address on the BRAM enables, and `rtl/tech/xilinx`'s wrapper would map both TCMs to URAM, so
  `RAM_STYLE BLOCK` and `CASCADE_HEIGHT 1` (`rtl/ctrl/viv/tsr_ctrl_tcm.xdc`) select the columns. The wrapper
  may instantiate `RAMB36E2` only inside `rtl/tech/xilinx`.
- Contents are undefined after a hardware reset; the host loads and verifies the image (EX §3.3; the host sequence is CC §6.4).

### 4.3 Bus rules

- Instruction port: decode ITCM and DM; `instr_gnt` from a register, never from the address decode.
- Data port: one access outstanding; `data_gnt` = ¬`outstanding_q`; every wait (credit, bridge, core window, DM
  arbitration) is a late `rvalid`. Errors return `data_err` (load or store access fault). With Ibex's writeback
  stage (§3.2, CC-34) the next request can come in the cycle the previous response returns: a TCM access (fixed
  next-cycle response) does not set `outstanding_q`, so the TCMs take one access per cycle; only accesses
  outside the TCMs hold `data_gnt` low until their `rvalid`. The RVFI bench (§15.4) checks this against the
  `tsr_rv` maxperf table.
- Two masters on the data side: Ibex's data port and the DM's SBA master, round robin per access. SBA
  restrictions (D15, CC-33): every SBA write to the control window 0x2000_0000–0x2000_0FFF (issue staging and
  push, `QCTRL`, `TRACE_MARK`, `LQ_POP`, `LCQ_POST`, `LOG_PUSH` and every other register there) and to the core
  window answers with an error and does nothing; SBA writes to ITCM and DTCM while `RV_RUN` = 0 answer with an
  error (the load port owns the write port then; the debugger uses EX's `LOAD_*`); SBA accesses to the DM range
  answer with an error. Everything else (TCM and SPM reads and writes, control-window and core-window reads)
  works for SBA, so the debugger can read memory without halting the hart.

---

## 5. Control window (RISC-V side, offsets from 0x2000_0000)

Word accesses only (a byte or half access faults). Reads have no side effects. The table is generated from the
single table source (EX-30, §13.4) for the RTL, the simulator and the SDK. Owners: CC (§5.1, §5.2 timer), EX
(0x200–0x2FC, EX §4.5), DB (0x160–0x17C and 0x400–0x7FC, DB §10.2).

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
| 0x040 + 4u | `DONE_SEQ[u]` | R | [27:0] completions of unit u (synchronized copy; may lag, never leads); [31:28] 0 |
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
| 0x200–0x294 | `LQ_HEAD_W0`–`W7`, `LQ_AVAIL`, `LQ_HEAD_SEQ`, `LQ_POP`, `LQ_CANCEL`, `LQ_STOP`, `LQ_FLAGS`, `LCQ_W0`–`W7`, `LCQ_POST`, `LCQ_FREE`, `LOG_PUSH`, `LOG_FREE`, `LOG_CFG`, `RT_STATE`, `RT_INFO` | | EX §4.5 (normative). CC implements them as EX §4.1 requires (§11.2, CC-30): reads of the LQ and LCQ registers are tagged handshakes into the core domain; `LQ_POP` and `LCQ_POST` commit in the core domain and fault there; `LCQ_W0`–`W7` and `RT_*` are `clk_rv` registers (`RT_*` mirrored to the host by a handshake on write); `LOG_PUSH` goes through the log CDC FIFO; `LOG_FREE` = `LOG_DEPTH` − (LOG_WR_rv − LOG_RD_snap) |
| 0x300 / 0x304 | `MTIME_LO` / `HI` | R | core clock cycles since reset (gray-synchronized) |
| 0x308 / 0x30C | `MTIMECMP_LO` / `HI` | RW | `irq_timer` while MTIME ≥ MTIMECMP (reset all ones) |
| 0x400–0x7FC | `DBG_FLAGS` and reserved | | DB (DB §10.2) |

### 5.3 Core window (0x2000_1000 + CSR offset, word loads)

The core window is a read-only view of the whole host CSR space of EX §5 (same offsets): `ID`, `CFG*`, `CAPS`,
`STATUS`, `SREG0`–`15` (0x9C0), PERF, `Q_*`, the LCQ and log rings. A load is a tagged handshake into
`tsr_csr`'s second read requester (§6.6), bounded like the bridge (4,096 clk_rv cycles, then a load fault and
`BRIDGE_TIMEOUT`). RISC-V reads never change the host's PERF LO and HI snapshot. Core-window loads of
`LOAD_DATA` (0x0C4) and of the DMI window (0xC00–0xDFC) return 0 without accessing the loader or the debug
module, so the RISC-V can neither disturb a load nor race the host's debugger. Latency is about 50–70 ns
(estimate). Typical use: s registers after a RED or SOP join, PERF for in-kernel profiling, the debug-halt
state.

---

## 6. Host CSR map (core CSR port; U200 `s_axi_control` 0x1000 + offset)

### 6.1 Map

EX §5 is the only host CSR map (EX-11). CC implements the following parts of it in `tsr_csr` and the control
subsystem:

| EX offset | Registers | CC implementation |
|---|---|---|
| 0x000–0x01C | `ID` (0x5453_5231 "TSR1"), `VERSION` 0x0000_0100 (Tesseract Core capability 1.0), `CFG*`, `CAPS`, `CFG3`, `CTL_ID` | constants from the table source |
| 0x020 / 0x024 | `CTRL`, `STATUS` | level and pulse bits as EX §5.2–§5.3; `RV_RUN` drives the RISC-V reset sequence (§12.2); `SOFT_RESET` starts the SR sequencer (§12.2) |
| 0x028–0x03C | `RV_BOOT`, `RV_FREQ`, `RV_CFG`, `WDOG_CFG`, `RT_STATE`, `RT_INFO` | `RV_FREQ` = clk_rv cycles in the last 2^16 core cycles (16-bit gray counter); `RT_*` are core-domain mirrors of the RISC-V's registers |
| 0x040–0x09C | LQ, LCQ, log ring control, `PERF_INFO`, `Q_DEPTH` (0x088) | core-domain rings and counters (EX §4.1–§4.2) |
| 0x0C0–0x0D4 | `LOAD_*` | holding register + tagged handshake into the TCM write port; VERIFY reader on the TCM read port (§4.2, §6.4) |
| 0x100–0x2FC | `PERF_0`–`63` | EX §6, CC §6.3 |
| 0x300–0x7FC | LCQ ring, log ring | 8 × 8 words of LUTRAM; 256 × 32 b ring (LUTRAM or one RAMB18) |
| 0x800–0xAFC | `DBG_*` (DB), `SREG0`–`15` at 0x9C0 | DB's block; `SREG` wired from the VU's s bank, present whenever the VU is built |
| 0xC00–0xDFC | DMI | tagged 2-phase handshake to the DM, gated by `DBG_CFG.DMI_EN` (DB §3.2) |
| 0xE00–0xF1C | `Q_*` | core-domain originals of `DONE_SEQ`, `ISSUED_SEQ` (synchronized copy), head state, error records and push indices |

Rules: no read has a side effect (except the PERF LO snapshot, and EX's `LOAD_DATA` and DMI rules); every host
access completes without clk_rv (core-domain copies; accesses that must reach clk_rv end after 4,096 core cycles
with a sticky bit); unmapped offsets read 0.

### 6.2 What the host can and cannot do

The host loads and verifies the image, starts and stops the RISC-V, launches kernels through the LQ, holds the
queue heads, aborts and clears the queues, soft-resets the device, reads every counter, record and the log, and
drives the DM through DMI. It cannot push a tile op (D15): no CSR reaches the issue staging registers, and SBA
writes to the whole control window are refused (§4.3).

### 6.3 PERF counters

EX §6 is normative: one counter space, time counters gated by ACTIVE ∧ ¬QHALT, event counters by ACTIVE, zeroed
by `SOFT_RESET`, per-channel slots at 5 + c and 5 + C + c, extension base X = 5 + 2C. CC implements every
counter and adds two extension slots: X + 20 `SPM_RD_WAIT` and X + 21 `SPM_WR_WAIT` (client-cycles a raised
request waited for its bank port, summed over clients). Counters kept in clk_rv (`PUSH_STALL`, `BRIDGE_WAIT`)
are 32-bit gray counters gated by a synchronized ACTIVE and extended to 64 bits in the core domain.

### 6.4 Image load and boot (host sequence)

EX §3.3–§3.5 are normative (cold load with receive CRC, VERIFY read-back CRC, warm restart, boot). CC's
hardware: each `LOAD_DATA` write goes into a core-side holding register and through a tagged handshake into the
TCM write port (4,096-cycle bound, `LOAD_TIMEOUT`); `LOAD_DATA` reads and the VERIFY reader use the TCM read
port through the same handshake, one word per request, while `RV_RUN` = 0. Then `RV_BOOT` ← vector-table
address and `CTRL.RV_RUN` ← 1; the runtime posts HELLO (EX §4.7 R1). 192 KiB is about 49 K posted writes, once
per process.

### 6.5 Log ring

EX §4.10 is normative: a 256-word core-domain ring, drop-new when full (words counted in `LOG_DROPPED`), host
reads at 0x400 + 4 × (k mod 256) and writes `LOG_RD`; records use DB's header (DB §6.1) and are never torn
because the SDK checks `LOG_FREE`. CC's hardware: `LOG_PUSH` (control window 0x280) → 16 × 32-bit CDC FIFO →
the ring; the RISC-V-side `LOG_FREE` of EX §4.1. The ring works when the queues wedge.

### 6.6 `tsr_csr` read requesters

Two requesters share one access pipeline: the host CSR port and the core-window handshake, round robin per
access. Only host LO reads touch the snapshot. The LQ and LCQ handshakes of §5.2 use a third, dedicated port
into the ring logic, so a core-window read never delays the runtime's pop or post.

---

## 7. Issue path

### 7.1 Push (clk_rv)

At the `ISSUE_PUSH` store, in one clk_rv step, if a credit exists for unit u = `UNIT_OF(w0[7:0])` (static
table, §7.4) and a FIFO entry is free: write the issue-FIFO entry {OP, w0–w7 masked, DEP0, DEP1, REC, ticket,
push PC}; ticket = {u, `ISSUED_SEQ[u]`}; `ISSUED_SEQ[u]`++; `ISSUE_PIDX`++; `ISSUE_TICKET` ← ticket;
DEP0 and DEP1 ← none; `rvalid`. The push PC is Ibex's `crash_dump_o.current_pc` at the store (DB §5.6). No
SPM-window store can be outstanding at a push: the data port holds one access at a time and a bridge store
completes only after `tsr_spm` accepted it, so an op pushed after a completed SPM-window store sees it.

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

- A queue slot belongs to an entry from its push until its unit accepts it (or it completes without running).
  Credit of u = min(`Q_DEPTH[u]` − (`ISSUED_SEQ[u]` − S_sync[u]), FIFO free entries, 15), where S[u] counts
  accepts and non-running completions in the core domain.
- Software checks the credit before pushing; the hardware wait is a safety net: at most `ISSUE_WAIT_MAX` =
  65,536 clk_rv cycles, then a store access fault and PUSH_TIMEOUT. The count does not advance while DB's
  `dbg_stop` level is 1 (DB §3.9), so a debug stop never turns a waiting push into a fault. While a push waits,
  Ibex takes neither interrupts nor a debug halt.
- Because credits include FIFO space, the dispatcher never blocks on a full queue.

### 7.4 Units, queues, depths

| Id | Unit | Opcodes (static `UNIT_OF`, generated from EX §7.2) | `Q_DEPTH` (U200) |
|---|---|---|---|
| 0 | DMA | LD, ST, KVST (AT) | 8 |
| 1 | QACT | QACT | 4 |
| 2 | MVM | MVM, ATTQK, ATTPV (AT) | 4 |
| 3 | VU | VOP, RED, ROPE, SOP, GDN, ASMAX, AMASK, AMERGE, ANORM (AT); TOPK, DQROW and VMASK where built | 16 |
| 4 | AT engine | reserved; absent (`ISSUE_UNITS[4]` = 0, `Q_DEPTH[4]` = 0, SPM client 5 tied off). The parameter depth 8 applies if a bitstream builds it | 0 |
| 5 | SYNC | NOP (join point), FENCE (barrier), QCLEAR (via `QCTRL`); every opcode not accepted by the issue path or whose unit is not built | 4 |
| 6–13 | reserved | | |
| 14 | marker records (DB); never a valid dependency | | |
| 15 | none ("no dependency") | | |

Opcodes 0x01–0x09, and every opcode the bitstream does not implement (TOPK, DQROW, VMASK in the release
image), are not accepted: SYNC ticket, `ILLEGAL_OP`. Queues are LUTRAM, written by the dispatcher, read into a
head register (§9.1). Depths are parameters; `Q_DEPTH` (EX §5.1, 0x088) reports them.

### 7.5 Tickets

Ticket = {unit[31:28], seq[27:0]}, seq from 0 per unit after a hardware reset, never reset otherwise. Ticket
(u, s) is done iff (D[u] − s) mod 2^28 ∈ [1, 2^27). Push index: 32-bit count of ticketed entries (ops and
QCLEAR) in push order (error ordering, `ERR_PIDX`); the RISC-V reads the next one as `ISSUE_PIDX`.

### 7.6 Issue cost (estimate)

RISC-V side: credit read 2 cycles, 1–7 word stores (staging words are kept after a push, so a loop rewrites
only changed words), 0–2 dep stores, push, optional ticket read: 15–40 clk_rv cycles per op including address
arithmetic. A control-window access costs 3 clk_rv cycles on Ibex maxperf and 4 on small in the simulator's
model (SW §4.2). Core side: push → start of an idle unit with done deps is about 16–18 core cycles (FIFO sync
2–3, pop 1, checker 8, enqueue 1, head load 1, evaluation 1, cmd_valid 1).

---

## 8. Dispatcher and checker

### 8.1 Operands and the MVM weight extent

`tsr_chk` takes its operands from w1..w3 directly (addresses are immediates), and `w0[31:16] ≠ 0` on an op
without an x16 field gives `ILLEGAL_OP` in C1. The MVM weight extent uses a serial divider and shift-add for
odd `NUM_CH`; for power-of-two `NUM_CH` it is R = (N + C − 1) >> log2 C and R × ROW_BEATS in one DSP (2
cycles). The dispatch handshake is in the queue heads (§9).

### 8.2 `tsr_chk` pipeline (one op at a time)

C0 capture entry (+ per-word nonzero flags) → C1 decode: opcode class, reserved fields, sub codes, unit built →
ILLEGAL_OP → C2 K / T / fmt, QA terms, strides, MVM R → C3 QA_LINES, (T−1)·stride, MVM DSP stage 1 → C4 extents
(48 bit), MVM DSP stage 2 → C5 end addresses → C6 limit compares → C7 priority, command build, queue write. 8
core cycles per op (MVM too for power-of-two `NUM_CH`; about 46 otherwise): at least 3 times the RISC-V's best
issue rate.

Error priority (EX §8.1, spec order [spec](../spec.md) §21 (Error check order) with the issue-path rules):
`ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` → `T_RANGE` → `OPERAND` (13, `ERR_INFO` = the offending
value) → `SPM_RANGE` → `DRAM_RANGE`, range operands in field order, `ERR_INFO` = the overflowing start address
(spec range-check rule, spec §21). Opcodes from VU and AT plug in as classes: each supplies its reserved-field
masks, used words, sub legality, `OPERAND` conditions in their order, up to four operand extents, each tagged
SPM or DRAM and read or write (at most one DRAM extent, checked on every channel as for MVM weights), as
formulas computable in C2–C4, and its command struct (§16). Four SPM extents are needed by VOP and ROPE (VU
§5.9) and by AMASK and AMERGE (AT §5.8). The same four tagged extents feed DB's tile-breakpoint matcher (DB
§5.3). A merging AMERGE lists six (Orun and st each as a read and a write entry): the checker class keeps the
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

`cnt[u]` counts pushes per unit in push order (equals `ISSUED_SEQ[u]` at that position). BAD_DEP for a dep
(v, s), v ≠ 15: v ∈ 6..14 or unit v not built; or (s − cnt[v]) mod 2^28 < 2^27 (not pushed yet; includes the
op's own ticket); or (clear_pt[v] − s) mod 2^28 ∈ [1, 2^27) (pushed before the last QCLEAR). A BAD_DEP dep is
replaced by none and the entry is poisoned with BAD_DEP. Deps on the op's own unit are legal (implied by queue
order). FENCE and NOP with w1–w7 ≠ 0 → ILLEGAL_OP.

---

## 9. Queue heads, dependencies, SYNC, errors

### 9.1 Head structure (per unit u)

LUTRAM queue → head register (prefetched as soon as the previous head leaves) → evaluation flops (`dep_ok_q`,
`snap_ok_q`, `failed_q`, `hold_q`) → command register `cmd_q` and `cmd_v_q` driving the unit's `cmd_valid` and
`cmd` (the unit-side clock enables are one flop away, 0 LUT levels).

### 9.2 Evaluation (registered, every cycle)

- `dep_ok` = done(DEP0) ∧ done(DEP1) against the core-domain D[] (6:1 mux + 28-bit modular compare,
  registered). SYNC FENCE and QCLEAR also need `snap_ok` = ∀v: D[v] has reached snap[v].
- `failed(t)` for dep t = (v, s): unit v in error state ∧ (s − ERR_SEQ[v]) mod 2^28 < 2^27.
- `hold` = host hold ∨ RISC-V hold ∨ (debug halt ∧ HOLD_ON_HALT) ∨ `dbg_hold[u]` (DB §13).
- `hard_hold` = `dbg_hold[u]` ∧ ¬`abort_active` ∧ ¬`SR_BUSY`, where `abort_active` is set by any QABORT (host,
  RISC-V, `LQ_CANCEL`, RISC-V reset, `SOFT_RESET`) and cleared by the next QCLEAR. `SR_BUSY` enters this term
  delayed by the latency of a QABORT (two cycles: the dispatcher registers the pulse and applies it a cycle
  later), so a held NOP, FENCE or error-coded head is never decided with its own code between `SOFT_RESET` and
  the abort: it completes ABORTED, as in the simulator, whose step 1 aborts first. DB's hold blocks non-running
  completions too (DB-9), except after a QABORT: aborted entries change nothing but counters, so they drain,
  and `SOFT_RESET` and the RISC-V reset always reach QIDLE.

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

"Unit may accept": the unit's `cmd_ready`. Units have no error output: no op may fail after it has started (EX
§7.2), so every error is a dispatch-time poison. The MVM and the VU may accept the next op before their done
(the as-if overlap; the VU's accept-ahead, VU §7.8), provided their done pulses stay in accept order (I3). At
most one D[u] increment per cycle per unit (gray-counter rule).

Latency targets: done of t → start of an idle dependent on another unit at most 3 core cycles; done → start of
the next entry of the same unit at most 2 cycles.

### 9.4 Invariants (asserted in every simulation)

- I1 completion rule: every entry completes only at its head and only after both deps are done (a BAD_DEP dep
  ignored); FENCE and QCLEAR also after their snapshot. Hence done(t) implies every op t depends on,
  transitively, is done.
- I2 errors visible before completion: D[u] does not pass ERR_SEQ[u] until the clk_rv side acknowledged unit
  u's record (about 5–8 core + 3–4 clk_rv cycles, once per error state).
- I3: a unit holds at most the ops it accepted, completes them in accept order; S[u] − D[u] ≤ 2 for MVM
  (overlap) and VU (accept-ahead, VU §7.8; tables `ACCEPT_MAX`), ≤ 1 for every other unit.
- I4: no SPM or memory traffic from a unit that holds no op (the integration check U2 of
  `rtl/core/tsr_core_dp.sv`, per unit).

### 9.5 Errors, QABORT, QCLEAR

- Error state, ERR_* capture, propagation, first error by smallest push index: as EX §8.2. Codes (EX §8.1): 1–8
  as spec; 11 DEP_FAILED, 12 BAD_DEP, 13 OPERAND; 9, 10, 14, 15 reserved. ERR_INFO: range errors the
  overflowing address, OPERAND the offending value, else 0. ERR_W0: the entry's w0.
- QABORT (RISC-V `QCTRL` when the dispatcher pops it; host `CTRL.QABORT`, `LQ_CANCEL` covering the running
  launch, RV_RUN 1 → 0 and `SOFT_RESET` at once): all six units enter error state with ERR_CODE ABORTED and
  ERR_SEQ[u] = S[u] (the first entry not started); running ops finish (no cancel); every later entry completes
  ABORTED, in order, after its deps.
- QCLEAR (RISC-V): SYNC entry; when every earlier entry is done, clears all error states and records, sets
  clear_pt[u] = snap[u]; its own D increment waits until the clk_rv side shows the cleared records. QCLEAR
  (host): accepted only when QIDLE, the issue FIFO empty and the RISC-V quiet (RV_RESET_DONE, or halted with no
  access outstanding); else QCLEAR_REFUSED.

### 9.6 Holds and quiescence

Hold sources: host (`CTRL.HOST_HOLD`), RISC-V (`QCTRL`, in push order), debug halt (`RV_HALTED` ∧
`HOLD_ON_HALT`), and DB's per-unit `dbg_hold[u]`. Each has one owner. A hold stops heads from starting;
non-running completions (poisoned, ABORTED, DEP_FAILED, SYNC) still drain, except under DB's hard hold (§9.2).
`QQUIESCENT` = no op held by a datapath unit, dispatcher and FIFO idle, every non-empty queue held, no
non-running entry left to drain; a head held by `hard_hold` counts as held whether or not it would run. With
the RISC-V halted it is the quiescent state.

---

## 10. SPM: even / odd banks, arbitration, RISC-V bridge

### 10.1 Organization

The 4 MiB SPM is two line-interleaved banks: bank E holds even lines, bank O odd lines, each 1R1W, so the SPM
has 2 read and 2 write ports (CC-12). Each bank is 32 K lines = 2 depth banks × 8 column slices of 16 K × 64
(4 cascaded URAM288): 128 URAM. The per-slice kept input registers are kept per (parity, depth bank, slice).

### 10.2 Client port contract

Per client c and bank b ∈ {E, O}:

- read request `rd_req_valid[c][b]`, `rd_req_ready[c][b]`, `rd_req_line[c][b]` (line[0] = b, asserted);
  response `rd_rsp_valid[c][b]` exactly `SPM_RD_LAT` cycles after the accepting edge, data on the shared bus
  `rd_rsp_data[b]`; no backpressure; in request order per (c, b).
- write `wr_valid[c][b]`, `wr_ready[c][b]`, `wr_line`, `wr_data`, `wr_be[63:0]`.
- A client may raise requests on both banks in the same cycle (two lines per cycle). Rule R2 per (c, b): a
  raised request stays with unchanged payload until accepted.
- Ordering: a read accepted after a write to the same line was accepted returns the new data (one bank owns
  the line, its input stage is shared by reads and writes).
- Arbitration per bank port: registered one-hot grant, round robin (the client granted last has the lowest
  priority), the owner keeps the grant while the port idles. `ready` = grant flop: no combinational path from
  any input to any output (R5). A waiting client is granted after at most NCL − 1 other grants.
- Clients: 0 DMA, 1 QACT, 2 MVM, 3 RISC-V bridge, 4 VU, 5 AT (absent → tied off). `NCL` = 6.

Single-port adapter `tsr_spm_cport` (client side, used by DMA, QACT, MVM, bridge): the client keeps its
single-port interface (one read and one write request per cycle); the adapter steers by line[0], `ready` = the
steered bank's grant, response valid = OR of the two, data = 2:1 select by the bank that answered. A streaming
client alternates banks and keeps both grants when alone: one line per cycle. The adapter's `ready` depends on
the client's own request line (inside the client, one LUT): accepted because R5 is about the SPM module
boundary.

VU: one client using both bank ports; it reads operand pairs (lines L, L+1 are always in different banks) and
can therefore stream two source operands at one line each per cycle whatever their alignment (alternate cycles:
A-pair, B-pair), and write one line per cycle (VU designs the reorder buffers).

### 10.3 No-deadlock argument

(i) every client raises an SPM request only with its payload ready and never waits on memory while it holds a
request (the DMA's LD writes beats already in its queue, ST reserves queue space before reading; QACT, gather,
VU, AT write results they hold; the bridge holds its data); (ii) a granted request transfers in that cycle;
(iii) round robin grants a waiting client within NCL − 1 grants. So every SPM request completes in bounded time
independent of memory progress. With the DMA LD reservation (§10.4) no memory-response path ever waits on the
SPM.

### 10.4 DMA LD burst reservation (CC-13)

`tsr_dma` LD issues a read burst only when its beat queue has room for the whole burst; the queue is 512 beats
(8 RAMB36 in 512 × 72 columns); `rd_rsp_ready` is constant 1 (asserted). SPM contention (VU streaming writes,
bridge, QACT) therefore cannot stall a channel's head-of-line response path, so MVM weight beats never wait
behind a DMA beat. 512 beats in flight cover the DRAM latency at one beat per cycle.

### 10.5 RISC-V SPM bridge and line buffer

- Core side `tsr_spm_bridge` (client 3): read = full line, returns the word (or the whole line for a
  line-buffer fill); write = the word placed at its offset with `wr_be` = byte enables shifted by 4 × word.
  Answers a store only after `tsr_spm` accepted it.
- RISC-V side bound 4,096 clk_rv cycles → access fault + `BRIDGE_TIMEOUT`; late answers discarded by tag.
- Line buffer (`tsr_spm_lb`, one 512-bit line + tag in clk_rv): loads from the 0x5000_0000 alias hit it when
  the tag matches (next-cycle `rvalid`), else fill it with one bridge read. Invalidated by: any store to either
  SPM alias, any control-window or core-window access, any rising edge of an interrupt source line,
  `SPM_LB_INV`. Every observation of a completion (`DONE_SEQ`, `QSTATUS` and record reads, the WAIT interrupt)
  therefore invalidates it, so memory model rule 2(c) holds for the alias too. It serves op lists and argument
  blocks: one bridge read fills a 64 B line that later reads hit, instead of 16 exact-window loads.
- Latency (estimate): an exact store or load crosses the bridge into the core domain and takes tens of core
  cycles; the cycle tables of SW §4.2 model it.

---

## 11. Clock domains and CDC

<p align="center"><img src="../fig/clocks.svg" alt="Two tinted clock-domain regions side by side. The XRT shell, a gray box at the top, supplies the kernel clocks ap_clk and ap_clk_2 and their resets as lines in each domain's color, and carries the CSR port and the memory ports into the core domain. The core clock domain (blue, ap_clk, 200 MHz) holds the platform adapter, tsr_csr, the issue path with tsr_chk and the unit queues, tsr_spm_bridge, the DMA, QACT, the MVM clusters, tsr_spm and one memory arbiter per channel. The control core domain (orange, clk_rv from ap_clk_2, 100 MHz) holds Ibex, the debug module, ITCM, DTCM, tsr_rv_bus and the control window. On the boundary tsr_ctrl_cdc has one synchronizer cell per bundle of crossings, each an arrow from the source block to the destination block with its width. From clk_rv to the core clock: the issue FIFO (gray pointers, 32 x 387 b), the log FIFO (16 x 32 b), the gray counters ISSUED_SEQ, clk_rv cycles and PERF, the RT_STATE and RT_INFO handshake, and the RV levels and sticky events through two-flop synchronizers. From the core clock to clk_rv: the S, D and MTIME gray counters, the error records (6 x 128 b) and the levels. Both ways: the SPM bridge and core window handshake, the LQ and LCQ register handshake, the LOAD handshake into the TCM load port and the DMI handshake. The core reset reaches clk_rv through a reset synchronizer. A legend shows the cell symbols and the constraints: ASYNC_REG, a maximum delay equal to the smaller clock period, and a bus skew limit on the gray buses." width="100%"></p>

The figure shows the two clock domains, the crossings between them and the constraints that §11.1–§11.3 describe.

### 11.1 Clocks

- Core clock = `ap_clk`; `clk_rv` comes from `ap_clk_2`, using U200 platform clock ID 1 (`ap_clk` uses ID 0).
  The link configs request both scalable clocks with `kernel_frequency=0:<core MHz>|1:<rv MHz>`; the release
  image requests `0:200|1:100`. `[clock] id` selects fixed reference clocks and rejects scalable IDs 0 and 1 on
  this platform.
- The flow has four clock checks. An `INIT_DESIGN.PRE` compatibility hook validates the native clock
  calculation of the installed Vitis and Vivado versions before the implementation clock override is applied.
  The block-design hook checks the actual clock and reset nets. The implementation hook checks the synthesized
  clocks, RV FFs and reset sources. The final pre-bitstream hook checks the scaled frequencies in
  `_new_clk_freq` (the clock `PERIOD` properties do not reflect that scaling) and accepts exactly the
  requested clocks. Software reads `RV_FREQ` rather than assuming F_RV.
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
| Sticky events (PUSH_TIMEOUT, BRIDGE_TIMEOUT) | RV → core | toggle synchronizers | |
| Levels (RV_RUN, holds, RV_HALTED, RV_SLEEP, RV_RESET_DONE, "clk_rv out of reset", ACTIVE, `irq_fast[2]` = LQ_AVAIL ≠ 0 ∧ ¬ACTIVE, `irq_fast[3]` = LCQ_FREE ≠ 0, `dbg_stop`, `dmcontrol.dmactive`) | both | 2-flop `tsr_sync`; RV_RESET_DONE two stages behind the FIFO write pointer; the core → RV bundle is registered once in the core domain before the synchronizer (its bits are combinational functions of several registers: `report_cdc` CDC-10) | |
| `RV_BOOT` | core → RV | 2-flop `tsr_sync` on the 32-bit bus; quasi-static: `tsr_csr` ignores `RV_BOOT` writes while `CTRL.RV_RUN` = 1, so the bus only changes while the hart is held, and the sequencer samples it at the reset release (also after a debug ndmreset) two cycles behind the synchronized RV_RUN; `report_cdc` waiver in `tsr_ctrl_cdc.xdc` | 32 b |

The widths are the exact ones of `rtl/ctrl/tsr_ctrl_pkg.sv` and the instances in `tsr_ctrl_cdc.sv`. The LQ and
LCQ request is {`lcq_w[7:0]` 8 × 32 b, `wdata` 32 b, `we` 1 b, `off` 12 b} = 301 b and its response {`fault`,
`rdata`} = 33 b; the LOAD request is {`we`, `addr` 32 b, `wdata` 32 b} = 65 b and its response {`err`, `rdata`}
= 33 b; the bridge and core-window response is {`err`, `line[511:0]`} = 513 b; a request of the bridge and the
core window is {`kind`, `be`, `addr`, `wdata`} = 64 b.

Rules: gray counters step at most 1 per source cycle; a counter that can move by more than one step at a time
(the host-written `LCQ_RD` and `LOG_RD`, and everything `SOFT_RESET` flushes) never crosses as a gray code: the
RISC-V sees it only through a handshake (EX §4.1); nothing crosses into a domain in reset (requests answered
locally within the bound); no CSR access waits on clk_rv without a bound.

### 11.3 Constraints (kernel XDC packaged in the `.xo`)

`set_max_delay -datapath_only` at most min(period) on every gray bus and handshake data path, `set_bus_skew` on
gray buses, `ASYNC_REG` on synchronizer flops; no `set_clock_groups` (it would override the max delays; the
platform clocks come from one clocking block and may be timed as related). Gates: `report_cdc` on the
out-of-context run lists only the intended crossings, all safe; zero unconstrained `ap_clk` ↔ `ap_clk_2` paths in
the linked design. The XDC caps both the datapath and gray-bus skew bounds at 3.333 ns; a clock with a period
above that keeps the original crossing budget, and a faster clock still requires its smaller period.

The constraints cover every crossing of the design, `tsr_dbg_rv`'s included, not only `tsr_ctrl`'s:
`rtl/ctrl/viv/tsr_ctrl_cdc.xdc` selects the cells by the instance names `u_sync*` (`tsr_sync`), `u_rst*`
(`tsr_rst_sync`), `u_hs*` (`tsr_cdc_hs`) and by the register names of the gray pointers, a convention
`rtl/ctrl/viv/check_cdc_names.py` checks in the lint. It is plain XDC, because Vivado's constraint reader
accepts no `if`, `foreach`, `proc` or `dict`. `package_xo.tcl` copies it into the kernel IP with `ap_clk` and
`ap_clk_2` as the two clock ports, in the IP's synthesis file group (used in synthesis, implementation and out
of context, processed late). The `report_cdc` waivers are not constraints (`rtl/ctrl/viv/tsr_cdc_waivers.tcl`,
Tcl): the out-of-context scripts source them, and so does the post-route hook `hw/u200/cdc_check.tcl` of the
k1 and k4 links, which then runs the gate (`rtl/ctrl/viv/cdc_gate.tcl`: no Critical or Warning `report_cdc` row
between the two clocks, and `report_clock_interaction` showing "Max Delay Datapath Only" in both directions)
and fails the implementation run when it does not hold. `report_cdc` on the whole core
(`tb/core/viv/core_ooc.sh`) and on the packaged IP (`tb/platform/viv/xo_cdc.tcl`) are part of T2 and T3. Every
signal enters a synchronizer straight from a flop that drives nothing else (CDC-10: no combinational logic
before it; CDC-11: no other fan-out): the level bundle of `tsr_ctrl_cdc`, `tsr_dbg_rv`'s levels and the reset
sources of the clk_rv synchronizers have a register of their own (`dont_touch`, so that equal registers of two
blocks are not merged into one with two synchronizers behind it).

### 11.4 Narrow physical transport to the RV region

`RV_LOCAL_TRANSPORT` is an implementation option on `tsr_core` and `tsr_xrt_kernel`, enabled in the release
image; K1 and generic instances default to the direct connections. The complete control subsystem, its TCMs and
the RV debug block can then share an outer SLR. The CDC structures and widths in §11.2 remain unchanged inside
that region. The links use the core clock and the existing hardware reset at both ends and introduce no clock
domain.

| Physical link | Transfer | Completion |
|---|---|---|
| Central SPM bridge to RV region | Four 128-bit segments, valid and error | Assemble all 64 bytes before answering the existing CDC handshake |
| RV region to central issue dispatcher | Three 129-bit segments, valid and accepted return | Retain the source FIFO head until the dispatcher accepts the complete 387-bit entry |
| RV region to central LQ/LCQ port | 128-bit data, 12-bit offset, write, END and valid | Ordinary access uses one segment; LCQ_POST uses two, then commits once |

SPM uses the existing bridge's held response register, which the bridge releases only after the fourth segment
is sent. The transport forwards each held CDC request once and rearms after the request drops. RV timeouts
still drain late responses through the tagged handshake. The RV's 64-byte line buffer, its invalidation rules
and next-cycle hits are unchanged (§10.5).

The issue link adds no queue credit. A partially transferred entry remains in the source FIFO, and the dispatcher
sees valid only after full assembly. `Q_FIFO` includes that reserved head until the acceptance acknowledgment
returns, one core cycle after dispatch, and idle and quiescent status also require this count to be zero, so
QCLEAR and reset draining cannot pass an entry while it is in transit. Soft RV reset does not reset either
transport endpoint. An LCQ_POST exposes neither half of its record to the CSR port: the assembled request
commits through the existing CSR logic and returns its existing fault and data response.

Register stages. With the option set, `tsr_core_dp` also registers the connections that run from the units and
the memory ports into the control plane. `tsr_issue` takes each unit's done strobe through one register
(`DONE_REG`), so the dispatcher, the queue states, the completion taps of the DB block and `perf_retired` see a
completion one cycle later (`perf_busy` keeps the unit's own timing); the DB block's memory taps go through two
register stages (`MT_PIPE`), so `DBG_RDOUT` and `DBG_WROUT` lag the memory ports by two cycles and the probes are
not compared with the model (DB §9.1). `XSLR_REG` (set with `RV_LOCAL_TRANSPORT` in the release image) does the
same for the blocks that sit two SLRs from the control plane, the whole vector unit in SLR2 and the trace
writer's write mux: the VU's links pass three register stages on the command and its ready, the done strobe and
the 16 s registers and two on the three PERF strobes and the progress strobe; the command goes through a valid
and ready pipe (`tsr_rpipe`), so the VU starts a command three cycles later and its done reaches `tsr_issue`
three cycles later (`DONE_REG` adds the fourth). The issue path's limit of two ops in flight (I3) counts the
commands in the pipe, and the unit's own checks (`tsr_core_dp` U4) are made on the unit's side of the stages.
The trace writer's request and data streams pass a `tsr_rslice` each and its burst response two flops; each
MVM cluster's attention-stall strobe passes two registers in `tsr_mvm`; the RISC-V's SPM request passes one
register stage in the SPM link end; each pair link of the MVM gather passes one `tsr_rpipe` stage; in
`tsr_xrt_kernel` the held event of each channel passes two register stages before the OR that sets `REQ_HELD`.
The elaboration without the option has none of these stages.

The stages shift the diagnostics, not the results: `perf_busy` of the VU includes 6 more cycles per isolated VU
op, the progress strobes reach the watchdog 2 cycles late (5 for the VU), and `ATT_PATH_STALL` counts 4 cycles
later; HUNG does not change at the default `WDOG_CFG.L` of 24. `FAST_CLK` (set in the release image, for a 4.0 ns
core clock) is the same kind of change: the PERF counters of `tsr_csr` take their clear, enable and increment
from registers, so every count is the single-stage count shifted by one cycle (two for the DB block's
increments) and a read in the cycle after an event misses it; everything the host reads after an op's done is
equal. The units' own stages (attention emitters, PV convert pipeline, KVST placer) lengthen those ops by a few
cycles and change no result. Sequence counters, error records and other control and debug connections remain
direct.

---

## 12. Resets, aborts, interrupts, timer

### 12.1 Events

| Event | Queued entries | Op running on a unit | RISC-V | Kept |
|---|---|---|---|---|
| QABORT (RISC-V `QCTRL`, in push order; host `CTRL.QABORT` or `LQ_CANCEL` covering the running launch, at once) | complete ABORTED in order after their deps; all six units in error state until QCLEAR | runs to completion | continues | sequences, records |
| QCLEAR (RISC-V ticketed; host only when idle) | applies after every earlier entry | | continues | clear point = sequence |
| Hold (any source) | stay; heads do not start | completes | continues | |
| RV_RUN 1 → 0, or DM `ndmreset` | QABORT at once | completes | reset sequence (§12.2) | TCMs, LQ / LCQ / LOG rings, sequences, error states |
| `SOFT_RESET` (host `CTRL` bit 10, EX §8.5) | QABORT at once, then drained | completes | reset and left in reset | DRAM, SPM, TCMs, s bank, sequences, the items EX §8.5 lists |
| Debug halt | continue (held with HOLD_ON_HALT) | completes | halted | |
| Hardware reset (`ap_rst_n`) | everything reset; TCM and SPM undefined | | held (RV_RUN = 0) | |

`SOFT_RESET` is EX's device-level reset (EX §8.5). There is no partial datapath reset and no cancel of a
running op (CC-22); a wedged unit is recovered by a card reset (EX §8.6).

### 12.2 Reset sequencing

- Hardware reset: `ap_rst_n` resets the core domain (with `RST_PIPE`); the clk_rv domain's reset is `ap_rst_n`
  through `tsr_rst_sync` (asynchronous assert, synchronous deassert). `ap_rst_n_2` (`rst_rv_n`) is AND-ed only
  into the Ibex and DM reset, never into CDC endpoints. Minimum `ap_rst_n` low: max(`RST_PIPE` + 1 core
  cycles, 3 clk_rv cycles).
- RISC-V reset (RV_RUN 1 → 0 or `ndmreset`): (1) the core applies QABORT and sets ANSWERED := TAKEN (EX §8.5);
  (2) the clk_rv sequencer stops granting on both Ibex ports and waits for the outstanding access (bridge bound
  at most); a push waiting for a credit is dropped (no FIFO entry, no ticket, no `ISSUE_PIDX` step); (3) resets
  Ibex, staging, deps, `WAIT_*`, the RISC-V hold source, `LCQ_W*`, `RT_*` and the line buffer for at least 3
  clk_rv cycles; (4) raises RV_RESET_DONE. CDC endpoints, the core-domain rings, MTIME, TCMs and the DM's DMI
  side are not reset by it.
- SR sequencer (core domain, `tsr_csr`; implements EX §8.5 and needs no `clk_rv`, CC-31). A `CTRL.SOFT_RESET`
  write sets `SR_BUSY` and completes at once. Then:
  1. drop `RV_RUN` (starting the RISC-V reset sequence above, with its QABORT); clear DB's stop state and force
     `dbg_hold` = 0 (DB §5.2); wait at most 2^16 core cycles for `RV_RESET_DONE`, else set `SR_RV_TIMEOUT` and
     go on (no `clk_rv`);
  2. wait until QIDLE holds with no datapath unit holding an op (the issue FIFO, every queue and every running
     op drained); a hung unit never finishes this step (EX §8.6);
  3. in the core domain: TAKEN := PUSHED, ANSWERED := TAKEN, LCQ_RD := LCQ_WR, LOG_RD := LOG_WR, both watermark
     VALID bits := 0, the host QCLEAR effects (error states and records cleared, clear points set), PERF
     zeroed and the snapshot dropped, the sticky `STATUS` bits and `LOG_DROPPED` cleared;
  4. `SR_BUSY` := 0.

  `RV_RUN` writes are refused (`RV_RUN_REFUSED`) while `SR_BUSY`. The clk_rv-side state (LOG_WR_rv, the
  interrupt levels) reloads from the core domain when the RISC-V leaves reset.
- Boot: core out of reset → image load → RV_RUN = 1 → Ibex reset deasserts 2–3 clk_rv cycles after the
  synchronized RV_RUN (and `ndmreset` low) → fetch at `RV_BOOT` + 0x80.

### 12.3 Interrupts (Ibex, level-sensitive, computed in clk_rv)

| Input | Source | Cleared by |
|---|---|---|
| `irq_fast[0]` | `WAIT_TICKET` done | rewriting or disarming `WAIT_TICKET` |
| `irq_fast[1]` | `ISSUE_STATUS.ERR_IRQ` | W1C |
| `irq_fast[2]` | `LQ_AVAIL` ≠ 0 ∧ ¬ACTIVE (EX §4.5; synchronized level) | `LQ_POP` |
| `irq_fast[3]` | `LCQ_FREE` ≠ 0 (synchronized level; the runtime enables it only while it waits to post) | posting or disabling |
| `irq_fast[4]` | DB: `REC_COUNT` ≥ 32 with `REC_CFG` bit1 (DB §8.2) | `REC_POP` |
| `irq_fast[5]` | `WAIT_CREDIT` | disarm |
| `irq_fast[6]` | reserved for DB | |
| `irq_timer` | MTIME ≥ MTIMECMP | MTIMECMP write |
| others | tied 0 | |

MTIME counts core clock cycles (the time base of PERF and of DB's records). WFI wakes on a pending enabled
interrupt with `mstatus.MIE` = 0.

---

## 13. Changes per RTL module, platform

### 13.1 Core boundary (`tsr_core`)

Inputs `clk`, `rst_n`, `clk_rv`, `rst_rv_n`, `plat_err_i` (EX §9.3); the CSR port; one memory port per channel
and, in a core with the dual weight stream (`CAPS` bit 16, EX §7.5), a read-only second one;
output `host_irq_o` = `CTRL.IRQ_EN` ∧ (LCQ_WR ≠ LCQ_RD) (EX §5.2; not connected on the U200). The control core is
mandatory (D14), so there is no `HAS_CTRL` parameter. Further parameters: `ITCM_BYTES`, `DTCM_BYTES`,
`Q_DEPTH_*`, `ISSUE_FIFO_DEPTH`, `HAS_AT` (0: no unit 4), `HAS_SBA`, `CTRL_SYNC`, `LQ_DEPTH`, `LCQ_DEPTH`,
`LOG_DEPTH`.

### 13.2 Module table

| Module | Change |
|---|---|
| `tsr_core.sv` | the boundary above; instantiates `tsr_ctrl_rv`, `tsr_ctrl_cdc`, `tsr_core_dp`; integration checks I1–I4 of §9.4 |
| `tsr_core_dp.sv` | core-domain top: `tsr_issue`, `tsr_csr`, units, `tsr_spm`, arbiters; issue-entry stream input |
| `tsr_csr.sv` | EX §5 map; no run control; two read requesters plus the LQ / LCQ port; LQ / LCQ / log rings; SR sequencer; DMI and load holding registers with the VERIFY reader |
| `tsr_pkg.sv` | unit ids; codes 11, 12 and 13; generated `tsr_tables_pkg` (EX-30) for offsets, opcodes and units |
| `tsr_spm.sv` | 2R2W even / odd, per-bank client ports, round robin (§10) |
| `tsr_dma.sv` | LD burst reservation, queue 512 beats, `rd_rsp_ready` = 1; `tsr_spm_cport`; KVST with meta-beat read-modify-write (AT §8.3) |
| `tsr_qact.sv` | `tsr_spm_cport` only; logic unchanged |
| `tsr_mvm.sv`, `tsr_mvm_gather.sv` | `tsr_spm_cport` on client 2; dispatch from the MVM queue head; ATTQK and ATTPV commands; gather full-line ATT path (AT §8.1); gather slice FORM 1 (§14.4) |
| `tsr_mem_arb.sv` | source 0 = AUX; `tsr_rslice` has the parameter `FORM` (§14.4); a second write source on channel `TRW_CH` = 0 for DB's trace writer (DB §8.3), burst-arbitrated with the DMA |
| `tsr_mvm_cluster`, `_front`, `_back`, `_actbuf`, `tsr_mac_lane` | ATTQK and ATTPV modes (area AT, AT §8) |
| `_ws`, `tsr_fp_*`, `tsr_vpipe`, `tsr_rpipe` | unchanged |

### 13.3 Platform adapter (`rtl/platform/alveo_xrt`)

EX §9.1–§9.3 are normative (CC-19): the control protocol is `ap_ctrl_none`; `s_axi_control` (EX §9.2) has 0x000–
0x03C reserved and reading 0, `ch<c>_base` at 0x040 + 8c, `ADAPTER_CTRL.MEM_EN` at 0x0E0 (latches and
alignment-checks all bases on its rising edge, locks them while set, holds core memory requests while 0),
`ADAPTER_STATUS` at 0x0F0, `ADAPTER_VERSION` at 0x0F4 and the core CSR window at 0x1000 + off. `plat_err_i` =
`ADAPTER_STATUS` bits 0–2 or error count ≠ 0; `host_irq_o` is unconnected. The kernel arguments are only
`ch<c>`. `tsr_u200_k1` and `tsr_u200_k4` take a second clock `ap_clk_2` and `ap_rst_n_2` (`gen_tops.py`,
`kernel_k*.xml`); `tsr_xrt_reset` serves `ap_clk` and `ap_rst_n_2` passes to the core as `rst_rv_n`.

### 13.4 One register table

The single table source is `sw/tables/tsr_tables.py` with SW's generator `gen_tables.py` (EX-30). CC owns its
rows: the control window (§5), the CC parts of the host map (§6.1), unit ids, queue depths, the issue-FIFO entry
layout and the `UNIT_OF` map. The generator emits `rtl/core/tsr_tables_pkg.sv`, `sw/sim/include/tsr_tables.h`,
`sw/ctrl/include/tsr/mmio.hpp` and `tb/common/tables.py`, with `--check` in lint.

---

## 14. Floorplan, timing and cost on the U200

The floorplan of `tsr_u200_k4` is a set of Pblocks for major functional blocks. Each block goes whole into one
Pblock: the platform's Pblock of an SLR, or one of the two Pblocks of its own below. No constraint names a LUT,
a register, a pipeline stage, a site or a BEL; Vivado places and routes inside the regions. `hw/u200/
k4_major_blocks.tcl` is the map (applied as the `place_design` pre-hook and, for the out-of-context core, by
`tb/core/viv/core_slr.tcl`), and `hw/viz/k4_pblock_preview.py` draws it. A Pblock does not exclude other logic:
clusters and DSP and BRAM columns inside it stay usable.

### 14.1 Resource budget

Estimated cost of the blocks of this area (U200):

| Block | Domain | LUT | FF | RAMB36 | DSP |
|---|---|---|---|---|---|
| Ibex (maxperf pipeline of §3.2) | rv | 3–5 K | 1.7–2.8 K | 0 | 3–4 |
| riscv-dbg `dm_top` + SBA arbiter | rv | 1–2 K | 1–1.5 K | 0 | 0 |
| ITCM 128 KiB + DTCM 64 KiB | rv | 0.1–0.3 K | | 48 | 0 |
| bus, control window, LQ / LCQ / log RV side, timer, line buffer | rv | 2–3.5 K | 3–4 K | 0 | 0 |
| CDC (FIFOs, gray counters, handshakes) | both | 0.6–1.2 K | 1.5–2.5 K | 0 | 0 |
| `tsr_issue` + `tsr_chk` (6 queues about 12 Kb LUTRAM, 12 dep compares, SYNC snapshot, 4-extent checker class) | core | 4–7 K | 3–5 K | 0 | 1–2 |
| `tsr_csr` (EX map, DMI, PERF counters, `Q_*` copies, rings, SR sequencer, loader with VERIFY) | core | 4–6 K | 3.5–4.5 K | 1 | 0 |

Ibex maxperf against small (CC-34) is 22 % more gates in Ibex's own table: the writeback-stage registers and
forwarding muxes, the branch target adder and the single-cycle multiplier option. On the U200 that is about
0.5–1 K LUT and 2 DSP more. The `clk_rv` domain is expected to reach 150–220 MHz; every core-clock addition is
built flop → at most 3 LUT levels → flop with registered boundaries.

The cost of the other blocks is in their areas: the VU in VU §7.10 and §7.12.9, DB in DB §11, AT's cluster modes
in AT §8.4. In the release image the control plane (control core, issue path with its two checker classes, CSR
block, DB core side) takes about 6–8 K CLBs of the control Pblock and the vector unit about 11–14 K CLBs of the
VU Pblock. The SLR1 strip also holds the shell's own logic, the SPM, DMA, QACT and the DDR1 and DDR2 subsystems,
so the control plane and the VU do not fit there beside them: that is why both have Pblocks in SLR0 and SLR2
(CC-24).

### 14.2 Placement

The map of `tsr_u200_k4`:

- SLR0: clusters 0 and 2 with their result assemblers, the pair source of clusters 0 and 2, the DDR0 bridges and
  the DDR0 arbiter; and the control Pblock (`pblock_tsr_ctrl_plane`): the control core (Ibex, debug module,
  TCMs, bus, control window, every clock-domain crossing, the RV side of the DB block), the issue path with its
  two checker classes, the CSR block and the DB block's core side, with the issue and LCQ ends of the control
  links. The control Pblock is the two clock-region rows at the SLR1 boundary (Y3 and Y4) of all six columns,
  without X2Y3. It leaves out the DDR0 controller's clock regions (X2Y1 to X2Y3) and the rows below, where
  cluster 0 has its home, and holds all of row Y4 at X0 to X2, under SLR1's dynamic Pblock where SPM, DMA and
  QACT are.
- SLR2: clusters 1 and 3 with their result assemblers, the pair source of clusters 1 and 3, the DDR3 bridges and
  arbiter; and the VU Pblock (`pblock_tsr_vu`): the whole vector unit, front end and both lane banks, in clock
  regions X0Y10 to X2Y14, the three left columns of all five rows, next to the SPM across the SLR1 boundary and
  off the DDR3 controller.
- SLR1, the data hub: the SPM, the RISC-V SPM bridge (and the SPM end of its link), DMA with KVST, QACT, the
  DDR1 and DDR2 arbiters and bridges and the central gather receiver. The stages around the receiver, the
  DMA-side boundary slices of channels 0 and 3 and the adapter's control logic are left to Vivado.

The bridges of a channel are the AXI4 bridge of its memory port and the read-only bridge of its second weight
stream.

The reasons. The control plane in SLR0: the SLR1 strip is nearly full with the shell, SPM, DMA and QACT, and the
control plane is the difference between a placement that routes and one that does not. The vector unit in
SLR2: with its front end in SLR1 it would take a large share of the SLR1 strip, and splitting it across SLRs
makes its packet cross SLR1 twice; one SLR leaves only its SPM ports (two reads, one write, 512 bits each) and
its command, done and register links to cross one boundary toward SLR1 and two toward the control plane. Two
Pblocks of their own keep large blocks off the DDR controllers' clock regions, which fill up and limit the
controller clocks.

Region contents (placed or out of context): the control Pblock has 16,300 CLBs, 252 RAMB36 and 840 DSP; the
control plane takes 7,600 CLBs, 48 RAMB36 and 8 DSP of it. The VU Pblock has 25,320 CLBs and 420 RAMB36; the
vector unit takes 10.6 to 13.8 K CLBs (42 to 55 %) and 16 RAMB36.

Placement rules for the blocks themselves:

- Keep the entire per-cluster attention job pipe `g_ajp.u_ajp` and the ATT/MVM selector beside its cluster. In
  K4, `MVM_GATHER_PAIR=1` joins selected jobs locally for pairs {0,2} and {1,3}. Each cluster has two source
  slots, followed by two local and two central shared slots per pair: eight slots per pair, one tagged job per
  cycle per link. The central take owns completion. Startup adds one cycle; numerical results are unchanged.
  Broadcast decode LUTs and carry cells belong beside the destination cluster; the lane wrapper's writer and
  reduction helpers belong in SLR1.
- With 16 physical VU lanes, bank 0 contains lanes 0–3 and 8–11 in SLR0 and bank 1 contains 4–7 and 12–15 in
  SLR2. The release image uses `VU_FRONT_LANES=8`, EXP 8 and BACK 4 (VU §7.10): bank 0 contains physical lanes
  {0,1,4,5}, bank 1 {2,3,6,7}, each with two back-equipped and two front-only lanes. Physical lane l retains
  separate logical contexts l and l+8. Keep local input and output pipe ends and front and back selection, with
  distinct bank control, reset and selection-token registers. Logical lane order and arithmetic are unchanged.
- DDR0 and DDR3 bridges and arbiters stay in SLR0 and SLR2 beside their controllers. Their weight-stream
  endpoints follow their cluster. `DDR_LOCAL_PIPE=1` adds five central DMA boundary slices per outer channel:
  read request and response, write request, data and response. The write slices follow the central trace mux;
  only DMA responses return to SLR1. DDR1 and DDR2 remain central. All streams retain their width and full-rate
  handshakes.

The k1 kernel (`NUM_CH` 1, `T_MAX` 1) has no SLR split; it is the bring-up link.

### 14.3 Timing gates

| Gate | Run | Criterion |
|---|---|---|
| T1 | out-of-context `tsr_ctrl_rv` on xcu200-2 | meets the `clk_rv` period of the release image with at least 0.3 ns margin |
| T2 | out-of-context k4 core with control core, VU, AT modes, DB and the floorplan of §14.2 | WNS ≥ 0 at the core clock period |
| T3 | k4 v++ link (all hardware of the release image) | the requested clocks (core 200 MHz, `clk_rv` 100 MHz): legal routing, setup and hold met at the accepted clocks, the RISC-V timing met, zero unconstrained `ap_clk` ↔ `ap_clk_2` paths, `report_cdc` clean |
| T4 | k1 v++ link | meets its requested clocks |

The early clock and reset gate verifies the requested clock constraints; the final gate checks the accepted
scaled frequencies (`hw/u200/clock_final_check.tcl` accepts exactly the clocks of the request for `tsr_u200_k4`;
a scaled-down clock fails the link). A route-only stop needs routed timing at an accepted core frequency, with
RV, hold and CDC checks, before timing success is claimed. A link that meets the requested clocks is not
repeated to reach a faster clock.

The `report_cdc` gate on the whole core lists only the intended crossings: Info endpoints, waivers for
handshake data, FIFO LUTRAM, gray and level buses and the quasi-static `RV_BOOT` bus, `ASYNC_REG` on the
synchronizer cells, `set_bus_skew` on the gray endpoints, and both directions "Max Delay Datapath Only" (§11.3).
The control-core RTL is frozen only after T1–T3; T2 precedes every T3.

### 14.4 The gather job-pipe paths

The central take of the gather job pipes (`sel_q` → `jv[sel_q]` → `take` → `jr[c]`,
`rtl/core/tsr_mvm_gather.sv`) is the `m_ready` of the central slice `u_gather/g_asm[c].u_jp/g_slices.g_s[1]`, and
`tsr_rslice` turns `m_ready` into the clock enable of its wide `main_q`. The structure that removes the path:

1. `tsr_rslice` FORM 1 (two slots s0 and s1, write and read pointers, occupancy flops): slot i loads `s_data`
   whenever it is free (enable = its own occupancy flop), `m_data` = slot[rp]; `m_ready` and `s_valid` drive only
   about 5 control flops. It is cycle-identical at the ports (both forms are 2-entry FIFOs with registered
   `s_ready` = count < 2 and `m_valid` = count > 0), so every replay stays bit- and cycle-exact. It is used for
   the last stage of every job pipe and of every pipe whose `m_ready` is a wide take (result pipes, arbiter
   response slices). Cost: one 2:1 mux level on `m_data`, no flops added.
2. The job-pipe stage-0 registers belong to their assembler's Pblock.
3. If a link is still short, `JOB_PIPE` 3 adds one cycle per MVM; results are identical.

### 14.5 Cost of this area (estimate)

The net cost of the control subsystem, issue path and CSR block (§14.1) is about 12.5–23 K LUT, 10–18.5 K FF,
49 RAMB36 and 4–6 DSP. The SPM 2R2W muxes, the `tsr_spm_cport`s and the DMA reservation add 3–5 K LUT and
8 RAMB36.

---

## 15. Verification

The oracle chain: Spike → the ISA simulator's RV32 model (`tsr_rv`, SW) and queue model → RTL → card.
Bit-exactness of concurrent programs is as in [spec](../spec.md) §10 (valid = race-free and
schedule-independent; the final DRAM, SPM, LCQ record stream and ticket error records must equal the
simulator's, SW §4.7).

### 15.1 Unit benches

`tb/dma` (LD reservation, `rd_rsp_ready` = 1 asserted, KVST with read-modify-write interleaved with LD and ST
of the same channel, `sys_rtl_spm_contend` with six clients and the deadlock case "LD and MVM on one channel
while gather, QACT, VU and bridge saturate both write ports"), `tb/fp`, `tb/mvm`, `tb/mvmback`, `tb/mvmfront`,
`tb/qact` (through `tsr_spm_cport`), `tb/spm` (2R2W: per-bank ordering, R2, round-robin bound NCL - 1, response
latency, write to read order, random multi-client traffic against a model) and `tb/tech` (with the `tsr_sync`,
`tsr_rst_sync` and `tsr_clk_div` models).

### 15.2 Block benches

- `tb/chk`: `tsr_chk` against the simulator's check function (random and directed ops of every class including
  VU and AT ones, every error priority pair including `OPERAND`, the four tagged extents, boundary extents):
  code, info and command fields bit-exact.
- `tb/issue`: `tsr_issue` with unit models of random latency: credits; ticket wrap (counters preset near 2^28);
  `ISSUE_PIDX` = dispatcher pidx for every entry, with QABORT, QHOLD and TRACE_MARK interleaved; BAD_DEP
  (unpushed, own ticket, reserved unit, stale); DEP_FAILED; NOP, FENCE and QCLEAR; QABORT then pushes; host
  QCLEAR refused and accepted; holds (each source cannot clear another); DB `dbg_hold` and `dbg_step`, the hard
  hold and its release by QABORT and `SOFT_RESET`; the SR sequencer (steps 1–4, with clk_rv stopped,
  `SR_RV_TIMEOUT`); I1–I4 as assertions in every test; done to dependent start at most 3 cycles.
- `tb/ctrl` (cocotb and Verilator, `tsr_ctrl_rv` + `tsr_ctrl_cdc` + a core-domain stub): riscv-tests (rv32ui,
  um, uc, mi -p) and riscv-arch-test (RV32IMC, Zicsr, privilege subset) from ITCM with signatures against Spike;
  every control-window register; push safety net and timeout (and its suspension under `dbg_stop`); bridge
  widths, alignment, faults, bound; line-buffer invalidation on each listed event; SBA refusals; LQ, LCQ and
  LOG handshakes with both sides busy, `LQ_POP` and `LCQ_POST` faults, and the host writing `LCQ_RD` and
  `LOG_RD` while the runtime polls `LCQ_FREE` and `LOG_FREE` (never over-reported); load, `LOAD_DATA` reads and
  VERIFY; DMI timeout then a late answer; every CSR access with clk_rv stopped or held in reset (must complete,
  RV_CLK_OK = 0); RISC-V reset during a bridge access and during a waiting push; `RV_HALTED` against
  `dmstatus.allhalted`.

### 15.3 Op-list replay into the issue port (Verilator, `tb/opreplay`)

A C++ bench instantiates `tsr_core_dp` alone and drives its issue-entry stream (D15 forbids a host path; this is
a simulation port only, never in a bitstream) with SW's `ops` traces (SW §5, §5.6: `traces/m2`, `gen`, `m3`,
`ctrl`, `vu`, `at` and op-list traces captured from the simulator) in the memory regimes of
`tb/fastreplay/memmodel.h` (rand, fast, slow, ordered), with random push gaps, host-hold pulses, SPM contention
from a phantom read client and clock-enable gaps per unit (simulation only). Pass: every run bit-exact (DRAM
expect regions, SPM lines written, counts, records) in every memory regime, at `RST_PIPE` 0 and 3, with the
generic and the Xilinx filelists.

### 15.4 Full system with the RISC-V

- Lockstep co-simulation: Ibex RVFI (simulation build) against `tsr_rv` per retired instruction; loads of
  timing-dependent registers (`DONE_SEQ`, `ISSUE_CREDIT`, MTIME, `QSTATUS`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`,
  error records, core-window PERF) and trap and interrupt entries take the RTL value and position; everything
  else (including `ISSUE_PIDX` and `ISSUED_SEQ`) must match.
- Two clocks: independent `clk` and `clk_rv`, ratios F_RV / F_CORE ∈ {0.31, 0.42, 0.5, 0.55, 0.6, 0.83, 1.0,
  1.67} with random phase, plus the `CTRL_SYNC` = 1 build; every test at every ratio. CDC jitter mode (CC-27):
  in simulation `tsr_sync` randomly takes 2 or 3 stages per transition and per bit.
- Random halts (DB's halt and HOLD_ON_HALT at random points of stress programs, compared at each quiescent
  point with the simulator advanced to the same retired count), concurrency stress (random race-free programs
  on RTL and simulator under random memory latency, bit-exact final state, at least 8 random simulator
  schedules per program) and vkernel with two clocks (llama.cpp end to end bit-identical against
  `TSR_DEVICE=sim`).

### 15.5 Mutation campaign

Drop a dep compare; complete a non-running entry before its deps; reorder a queue; skip a poison; advance D[u]
before the record acknowledge; apply a RISC-V QCLEAR out of band; step a gray counter by 2; one hold source
clearing another; credit returned at done instead of accept; round robin back to fixed priority on one bank
port; a CDC FIFO full flag one cycle late; the line buffer not invalidated on one event; a unit started in the
cycle it becomes head; `tsr_rslice` FORM 1 data enable including `m_ready`; `ISSUE_PIDX` advanced by a QHOLD;
`LCQ_FREE` computed from a gray-coded `LCQ_RD`; the hard hold kept after a QABORT; the SR sequencer skipping
step 2. Each must be killed by the benches above.

### 15.6 Vivado

T1–T4 of §14.3; `report_cdc` on every out-of-context run; `report_timing` between `ap_clk` and `ap_clk_2` on the
linked design; utilization per SLR.

---

## 16. Interfaces with other areas

| Area | CC provides | CC assumes from the area |
|---|---|---|
| EX | issue port, credits, tickets, `ISSUE_PIDX`, at most 2 hardware deps, SYNC NOP, FENCE and QCLEAR, I1 and I2, error states and records, holds, QIDLE and QQUIESCENT; unit ids 0–5, queue depths (`Q_DEPTH`); the RTL of EX's LQ, LCQ and log rings with handshake CDC, the SR sequencer, the loader with VERIFY; MTIME; the line-buffered SPM alias | the opcode table and the static `UNIT_OF` mapping (EX §7.2–§7.3); the host CSR map (EX §5); the memory model as an architectural rule ([spec](../spec.md) §16); the error-code list and check order; CAPS bits; the launch protocol (EX §4) and `SOFT_RESET` semantics (EX §8.5) |
| VU | queue 3 (depth 16), dispatch with accept-ahead (I3 ≤ 2), SPM client 4 with both bank ports, a checker class slot with four tagged extents, the `SREG` read path (0x9C0) into `tsr_csr` | per opcode (including AT's 0x34–0x37): reserved masks, used words, sub legality, `OPERAND` conditions, up to 4 extent formulas, the command struct; the s-register bank exported; no run-time errors |
| AT | KVST on the DMA queue, ATTQK and ATTPV on the MVM queue (depth 4), their checker classes; SPM client 5 and unit 4 tied off (`HAS_AT` = 0); memory source 0 tied off; PERF X + 16 … X + 19 wiring | its ops' check rules and command structs (AT §5.8), its cluster-mode RTL (AT §8) |
| DB | DM instance (Debug Spec 0.13.2) with DMI at CSR 0xC00–0xDFC gated by `DMI_EN`, SBA master on the RISC-V bus (all control-window writes refused), `RV_HALTED`, `ndmreset`, HOLD_ON_HALT, per-unit `dbg_hold[u]` and `dbg_step[u]` with the hard-hold rule of §9.2, head info, per-unit start and completion strobes, the enqueue tap, the push PC in the issue FIFO, TRACE_MARK entries, the REC push flag, `ISSUE_WAIT_MAX` suspension under `dbg_stop`, the second write source on `TRW_CH` = 0, per-channel outstanding counts, the `SREG` wiring; reserved windows (control window 0x160–0x17C and 0x400–0x7FC; CSR 0x800–0xAFC; `irq_fast[4]` and `[6]`; PERF 61–63) | the GDB RSP server on DMI (it never writes the issue staging or `QCTRL` through the program buffer), record and trace formats, tile-level debug semantics, the log record format |
| SW | the table rows of §13.4, the boot and load sequence, timeouts, the `xrt::ip` control model, the `tsr_core_dp` replay bench, the two-clock vkernel | `tsr_rv`, the queue model with random schedules and the race detector, the trace format, the table generator, the SDK (crt0, linker script for §4.1, `issue()` checking credits, the runtime R1–R10), the transport |

---

## 17. Decisions

| Id | Decision | Reason |
|---|---|---|
| CC-1 | Ibex: OpenTitan's locked commit at vendoring, unmodified; RV32IMC, `RV32MFast`, FPGA register file, no PMP, cache or predictor, 2 triggers (the pipeline is amended by CC-34) | The OpenTitan lock is a state that a silicon sign-off flow ran on, which matters more than features; the configuration is the verified "small" one plus documented deviations |
| CC-2 | Debug module: upstream riscv-dbg `dm_top` without its JTAG DTM, DMI from the CSR window, SBA connected | D18 needs a standard DM; GDB over PCIe (DB) speaks DMI directly, so the JTAG DTM is dead weight |
| CC-3 | TCMs: ITCM 128 KiB, DTCM 64 KiB, RAMB36 32K × 1 columns, read latency 1 (48 RAMB36) | The graph executor, the model kernels, debug formatting and the stack fit without overlays; columns avoid an output mux at read latency 1; URAM stays for the SPM |
| CC-4 | A 0x5000_0000 alias of the SPM whose loads hit a one-line buffer, invalidated by SPM stores, window accesses, interrupt edges and `SPM_LB_INV` | Op lists and argument blocks are read word by word; the alias makes that about 7 times faster while keeping the memory model |
| CC-5 | Issue port: memory-mapped staging (8 words, 2 deps, push with word mask), credits checked by software, a 65,536-cycle hardware safety net suspended under `dbg_stop` | Ibex stays unmodified; a blocking push would block interrupts and debug halts for up to a long FFN MVM |
| CC-6 | Unit ids: 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT engine (absent), 5 SYNC; 6–13 reserved, 14 DB marker, 15 none | Keeps six units, six counters and the CDC widths |
| CC-7 | Opcode to unit: a static `UNIT_OF` generated from EX §7.2 (EX-30); `ISSUE_UNITS` and `Q_DEPTH` report built units; no `ISSUE_QMAP` | Software has the table at compile time; one mapping fewer to keep consistent |
| CC-8 | Queue depths: DMA 8, QACT 4, MVM 4, VU 16, SYNC 4 (unit 4 absent; parameter 8); issue FIFO 32 | Many short VU ops per layer want a deeper VU queue for about 100 LUT |
| CC-9 | Checker: one `tsr_chk`, 8 cycles per op, EX's check order with `OPERAND`, four tagged extents per class, MVM extent by shift and DSP for power-of-two `NUM_CH` | It is at least 3 times the issue rate; the DSP path removes the 46-cycle MVM check; VOP, ROPE, AMASK and AMERGE have four SPM operands |
| CC-10 | Dispatch: queue-head command registers drive the units; an entry never starts in the cycle it becomes head | Clock enables stay one flop away; DB gets one cycle to hold a breakpointed entry; the cycle is hidden while the unit is busy |
| CC-11 | Withdrawn: runtime unit errors. Units have no error output; every error is found at dispatch (EX §7.2) | EX's rule that no op fails after it has started keeps the first error schedule-independent |
| CC-12 | SPM banking: even and odd line interleave, 2R2W, per-bank client ports, registered round robin, 6 clients, 128 URAM, a single-port adapter for single-port clients | A second read path without extra URAM; line pairs always hit both banks, so the VU streams two operands at full rate; per-bank ports keep R5 (ready from flops) |
| CC-13 | DMA LD: burst reservation with a 512-beat queue (8 RAMB36); `rd_rsp_ready` constant 1 | SPM contention can never stall a channel's response path, which would starve MVM weight beats |
| CC-14 | s registers: one bank in the VU, read-only at SREG0-15 (core CSR 0x9C0) for the host and the RISC-V; writes only through ops | D14 leaves one bank; writes stay ordered by queues and deps; one location for every reader |
| CC-15 | Core window: a read-only view of the whole CSR space at 0x2000_1000 with no snapshot side effects; loads of `LOAD_DATA` and the DMI window return 0 | One mechanism gives the RISC-V ID, CFG, s registers, PERF and records; it cannot disturb a load or the debugger |
| CC-16 | Debugger memory path: DM SBA as a second master on the RISC-V data bus; no host debug SPM client | Reaches TCM, SPM and the read-only registers without halting; the SPM keeps six clients |
| CC-17 | Log: EX's 256-word ring; CC's part: `LOG_PUSH` into a 16-entry CDC FIFO, `LOG_FREE` from a `clk_rv` push count and a handshaked `LOG_RD` | Never stalls, never torn, works with wedged queues |
| CC-18 | RISC-V clock: `ap_clk_2` through a v++ clock request (the release image requests 100 MHz); fully asynchronous CDC | D5 and D12; independent of core-clock scaling |
| CC-19 | Kernel control: `ap_ctrl_none` per EX §9; `xrt::ip` and polling; `MEM_EN` latch; no interrupt on the U200 | XRT 2.18 has no `user_managed` type and routes AP_CTRL_NONE to `xrt::ip`, the register-level path the host needs |
| CC-20 | Withdrawn: a second ID and VERSION. `ID` 'TSR1' and `VERSION` 0x0000_0100 (Tesseract Core capability 1.0) per EX-12 | The host open code already refuses a different capability major |
| CC-21 | Withdrawn: a separate PERF scheme. EX §6's counter space; CC adds X + 20 SPM_RD_WAIT and X + 21 SPM_WR_WAIT | One counter space with protocol-defined gating |
| CC-22 | Recovery: no partial datapath reset and no op cancel; `SOFT_RESET` per EX through the SR sequencer; a wedged unit needs a card reset | Unit resets with outstanding AXI bursts break the arbiter contracts |
| CC-23 | Timing gate (D13): the k4 link meets the requested clocks (core 200 MHz, `clk_rv` 100 MHz in the release image) with setup, hold, RISC-V timing and CDC checks passing; T1–T4 in §14.3 | The control core and the vector unit must not lower the core clock; a link that meets the request is not repeated to go faster |
| CC-24 | Floorplan: Pblocks for major blocks only: the control plane in a Pblock in SLR0 at the SLR1 boundary, the VU in a Pblock in SLR2, clusters 0 and 2 with the DDR0 path in SLR0, clusters 1 and 3 with the DDR3 path in SLR2, the data hub in SLR1 | The SLR1 strip is nearly full without the control plane and the VU; the Pblocks keep large blocks off the DDR controllers' clock regions; every constraint leaves leaf placement to Vivado |
| CC-25 | Gather fix: `tsr_rslice` FORM 1 (cycle-identical) for the central job-pipe stages, stage 0 in its assembler's Pblock; `JOB_PIPE` 3 only if a link is still short | Removes the worst paths without changing any cycle |
| CC-26 | Memory source 0 is `TSR_SRC_AUX`, tied off | Keeps the arbiter shape; nothing claims it |
| CC-27 | CDC verification: Verilator two-clock benches over a ratio sweep, a jitter mode in `tsr_sync`, `report_cdc`, zero unconstrained `ap_clk` / `ap_clk_2` paths | Verilator cannot model metastability; jitter exercises skew |
| CC-28 | Register table: CC's rows in `sw/tables/tsr_tables.py` (EX-30) | One source for every area |
| CC-29 | `RV_HALTED` is derived from the last instruction-fetch address in the DM range, cross-checked against `dmstatus`; DMI fallback | Keeps riscv-dbg unmodified |
| CC-30 | LQ, LCQ and log CDC: the RISC-V reaches EX's rings only by tagged handshakes (reads return core-domain values; pop and post commit in the core domain); no gray-coded copies of multi-step counters | The host-written LCQ_RD and LOG_RD and the SOFT_RESET flushes jump by more than one step |
| CC-31 | SR sequencer: a core-domain sequencer for EX's SOFT_RESET (RISC-V reset within 2^16 cycles, drain, clear list, SR_BUSY), independent of `clk_rv` | The host needs an atomic known-state operation even without `clk_rv` |
| CC-32 | `ISSUE_PIDX`: a `clk_rv` counter of ticketed pushes at control window 0x03C, equal to the dispatcher's pidx; +1 per ISSUE_PUSH and per QCLEAR only | Exact and timing-independent like ISSUED_SEQ, for about 40 FF |
| CC-33 | SBA write refusal: every SBA write to the control and core windows is refused in hardware, and SBA TCM writes while RV_RUN = 0 | LQ_POP, LCQ_POST, LOG_PUSH and TRACE_MARK have protocol effects; a whole-window rule is simple to verify |
| CC-34 | Ibex pipeline (amends CC-1): "maxperf": WritebackStage 1, BranchTargetALU 1, with `RV32MFast`; the TCMs take one data access per cycle; the simulator's default timing model | Fewer RISC-V cycles per op record than "small" for about 0.5–1 K LUT and 2 DSP; `RV32MFast` keeps the `clk_rv` paths short |

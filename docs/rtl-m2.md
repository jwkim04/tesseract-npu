# Tesseract M2 RTL — architecture and interface contract

Status: M2 architecture v1 (2026-10-22). Contract for the M2 module agents.
Source-of-truth order: `docs/spec.md` > ISA simulator behaviour (`sw/sim/tsr_sim.c`; numerics
`sw/sim/tsr_ref.c`) > this document > `rtl/core/tsr_pkg.sv`. When this document needs a change,
the lead edits it; spec changes go through §11 (proposals).

**M2 scope** (spec ISA table, 단계 = M2): an RTL core that executes `NOP END LI ADDI LD ST QACT
FENCE` bit-exactly like the simulator, including every error those instructions can raise, the
CSRs, command fetch and the perf counters (present, not bit-compared). `MVM` (M3) is absent:
`tsr_core #(.HAS_MVM(0))` decodes opcode 0x20 as `ILLEGAL_OP`; M2 traces contain no MVM. Every
structure that M3 needs (weight-stream source per channel, MVM SPM client, MVM dispatch struct,
MVM perf counter) exists as a tied-off port.

---

## 1. Conventions (all blocks)

- **Language/lint**: SystemVerilog-2017, `rtl/lint.sh` (Verilator 5 `--lint-only -Wall`, every
  module as its own top plus reduced/edge `tsr_core` configs) must stay clean. No blanket waivers.
  The only waiver in `rtl/core` is `UNUSEDPARAM` around the constant catalogues of the two
  packages. Unused inputs: `wire unused_ok = &{1'b0, a, b};` (Verilator's default `*unused*`
  regexp). Vendor primitives/attributes only in `rtl/tech/<target>/`.
- **Clock/reset**: one clock `clk`. `rst_n` is synchronous, active low, already synchronised by
  the platform layer. Reset every control flop (valid bits, FSM state, counters, pointers); wide
  datapath registers need no reset. Verilator builds randomise initial state (`--x-initial
  unique`, `+verilator+rand+reset+2`), so missing resets show up in simulation.
- **valid/ready** (all streams, internal and external):
  - R1 transfer happens at a rising edge where `valid && ready`;
  - R2 `valid` must not depend combinationally on `ready`; once raised, `valid` and the payload
    stay unchanged until the transfer (the memory model asserts this on the core boundary);
  - R3 `ready` may depend on `valid`, but prefer registered `ready` across module boundaries;
  - R4 a requester of a stream **without backpressure** (SPM read data, fixed-latency fp
    pipelines) or of a stream it must never stall (memory read data, see §4 M6) reserves space for
    the response before it issues the request;
  - R5 no combinational path from a module's inputs to its outputs across the boundaries listed in
    §3–§5 unless stated (exception: tied-off M3 ports);
  - R6 1-cycle "pulse" signals (`done`, `run_start`, …) are registered.
- **Timing (300 MHz, U200)**: every `m_*` core-boundary output comes from a flop (tsr_mem_arb);
  memory ports cross SLRs, so arb and DMA may add pipeline stages freely (parameterise depth).
  Keep logic depth at ~6 LUT levels; wide compares/adds (range checks, 64-bit fetch limit) are
  pipelined over cycles in cmdproc (instruction rate is not critical).
- **Data byte order**: byte *j* of a 64-byte beat/line is bits `[8j+7:8j]` (little-endian);
  f32 element *i* of a line is bits `[32i+31:32i]`; instruction word *wi* is `insn[i]`
  (`tsr_insn_t = logic [7:0][31:0]`); instruction *pc* is bits `[256*(pc%2) +: 256]` of fetch
  beat `CMD_ADDR + pc/2`.
- **Widths**: ISA-level quantities (addresses `R[x]+imm`, lengths, strides, PC) are 32-bit fields
  in the dispatch structs regardless of parameters; cmdproc's range checks guarantee the bits a
  unit does not use are zero (units slice what they need, e.g. `cmd.spm_line[SPM_AW-1:0]`).
- **Parameters**: `tsr_core` takes the spec table (`NUM_CH BEAT_BITS LANES T_MAX K_MAX SPM_BYTES
  VL NIREG NSREG LOOP_DEPTH RD_OUTSTANDING CH_BYTES ADDR_BEAT_W`, U200 v1 defaults in
  `tsr_pkg::TSR_DEF_*`) plus implementation parameters `HAS_MVM` (1 since M3, the U200 v1
  bitstream; M2 test builds pin 0), `SPM_RD_LAT` (2),
  `FETCH_BEATS` (16). `CH_BYTES` is `longint unsigned` (pass `-GCH_BYTES="64'd1073741824"`).
  Elaboration checks (`$fatal`) mirror `tsr_sim_create`: `BEAT_BITS == 512`, `NUM_CH` 1..16,
  `T_MAX` 1..8, `K_MAX` multiple of 256 in [256, 65280], `SPM_BYTES` positive multiple of 64,
  `NIREG == 16`, `ADDR_BEAT_W` 1..40; with `HAS_MVM = 1` also `LANES == 128`, `MVM_WS_FIFO` a
  power of two ≥ 64 and the MVM latencies in range (rtl-m3 §13.1; M2 required `HAS_MVM == 0`).
- **Files**: `rtl/filelist.f` (generic target, used by lint and the cocotb runner; packages
  first). New modules go into it (ask the lead; it is a shared file). Target filelists
  (`rtl/filelist_u200.f`, M4) swap `tech/generic` for `tech/xilinx`.

---

## 2. Block diagram and ownership

```
                      CSR port (csr_valid/ready/we/addr/wdata/rdata)          done
                                   │                                           ▲
                         ┌─────────▼─────────┐  run_start/args/abort/srst      │
                         │      tsr_csr      │──────────────────────┐          │
                         │ regs STATUS ERR_* │◄─ run_finish/err/pc ─┐│          │
                         │ PERF_* (64-bit)   │◄─ perf events ─────┐ ││          │
                         └───────────────────┘                    │ ││          │
   ┌─────────────┐ insn stream (tsr_fetch_insn_t)  ┌──────────────┴─▼▼───────┐ │
   │  tsr_fetch  │────────────────────────────────►│      tsr_cmdproc        │─┘
   │ prefetch ch0│◄── start/stop/cmd_addr/count ───│ checks · R0..R15 · LI/  │
   └──────┬──────┘                                 │ ADDI · dispatch · errors│
          │ src FETCH                              └──┬─────────┬────────┬───┘
          │                        tsr_dma_cmd_t      │         │tsr_qact│tsr_mvm_cmd_t (M3,
          │                     ┌─────────────────────▼──┐ ┌────▼──────┐ │ tied off)
          │                     │        tsr_dma         │ │ tsr_qact  │ ▼
          │                     │ LD/ST, burst split,    │ │ A32/A256  │ [MVM unit, M3]
          │                     │ write-resp accounting  │ │ (tsr_fp_*)│
          │                     └──┬───────────┬─────────┘ └────┬──────┘
          │      src DMA (per ch)  │           │ SPM client 0   │ SPM client 1   SPM client 2 (M3)
          │                        │           └───────┐  ┌─────┘
          │                        │                ┌──▼──▼───────────────┐
          │                        │                │      tsr_spm        │ 1R1W, line-addressed,
          │                        │                │ tsr_sram_1r1w banks │ SPM_BYTES/64 x 512 b
          │                        │                └─────────────────────┘
   ┌──────▼────────────────────────▼──┐   ┌───────────────────┐       ┌───────────────────┐
   │ tsr_mem_arb ch0: FETCH, DMA, WS* │   │ tsr_mem_arb ch1   │  ...  │ tsr_mem_arb chC-1 │
   └──────────────┬───────────────────┘   └─────────┬─────────┘       └─────────┬─────────┘
            m_* [0]  (memory port)             m_* [1]                     m_* [C-1]
   * WS = M3 weight stream source, tied off in M2.
```

| Block | Agent | Files (owner edits only these) | Contract |
|---|---|---|---|
| (a) control | cmdproc agent | `rtl/core/tsr_csr.sv`, `tsr_cmdproc.sv`, `tsr_fetch.sv`, `tb/cmdproc/` | §3.1, §3.5, §6, §7 |
| (b) memory | DMA agent | `rtl/core/tsr_dma.sv`, `tsr_mem_arb.sv`, `tb/dma/` | §3.2, §4 |
| (c) fp library | fp agent | `rtl/core/tsr_fp_pkg.sv`, `tsr_fp_mul.sv`, `tsr_fp_div.sv`, `tb/fp/` | §8.2 |
| (d) SPM | SPM agent | `rtl/core/tsr_spm.sv`, `rtl/tech/generic/tsr_sram_1r1w.sv`, `tb/spm/` | §5 |
| (e) QACT | QACT agent | `rtl/core/tsr_qact.sv`, `tb/qact/` | §3.3, §8.1 |
| (f) top | core agent | `rtl/core/tsr_core.sv`, `tb/core/` (trace replay) | §1, §2, §9 |
| infra | architecture | `rtl/core/tsr_pkg.sv`, `rtl/filelist.f`, `rtl/lint.sh`, `tb/common/`, `tb/conftest.py`, `tb/pytest.ini`, `tb/run_all.sh`, `traces/m2/`, this file | all |

The skeletons in `rtl/core` carry the final port lists and a behavioural contract comment; their
bodies are tied off and lint clean. `tsr_core.sv` already contains the complete M2 netlist (all
instances and wiring); the core agent keeps the wiring and adds integration checks. Port-list
changes are interface changes: agree them with the lead (update this file, `tsr_pkg.sv`,
`tsr_core.sv` and `tb/common/isa.py` together).

---

## 3. Internal interfaces

### 3.0 Dispatch protocol (cmdproc → unit), common to DMA, QACT, (M3) MVM

| Signal | Dir (unit) | Meaning |
|---|---|---|
| `cmd_valid` | in | cmdproc offers one checked instruction (R2: held until `cmd_ready`) |
| `cmd_ready` | out | unit idle and able to start (registered) |
| `cmd` | in | `tsr_<unit>_cmd_t`, valid with `cmd_valid`; the unit latches it at the transfer |
| `done` | out | 1-cycle pulse: **all architectural effects complete and visible** |

- Exactly one instruction is in flight in the whole core (sequential semantics, spec 실행 모델).
  cmdproc never offers the next command before the previous unit's `done`.
- `done` is never in the cycle of the accepting edge (≥ 1 cycle later). "Visible" means: every SPM
  line write of the instruction has been accepted by `tsr_spm` (§5 ordering guarantees later
  reads see it) and, for ST, every DRAM write response has been received.
- Units never report errors: every architectural check is done in cmdproc before dispatch (§7);
  operand values a unit receives are always legal. Units may `assert` legality in simulation.
- No flush/cancel: ABORT and SOFT_RESET act only between instructions (§6), so a dispatched
  instruction always runs to completion.

### 3.1 tsr_fetch ↔ tsr_cmdproc

`tsr_fetch_insn_t` = `{range_err[1], pc[32], insn[256]}` (289 bits).

| Signal | From | Meaning |
|---|---|---|
| `start` | cmdproc | pulse; fetch latches `cmd_addr[63:0]` (beat), `cmd_count[31:0]` (cmdproc holds them stable for the run) |
| `insn_valid/insn_ready/insn` | fetch→cmdproc | instructions pc = 0, 1, … in order; cmdproc pops one per instruction |
| `stop` | cmdproc | pulse at the end of a run (END, error, abort): issue nothing new, drain and discard outstanding responses |
| `idle` | fetch | no request outstanding and no run active |

Rules: F1 fetch reads only beats `CMD_ADDR + pc/2` for `pc < CMD_COUNT` and only beats below the
DRAM limit `min(CH_BYTES/64, 2^ADDR_BEAT_W)`, compared in 64 bits (`CMD_ADDR_HI != 0` is out of
range). F2 the first instruction in a beat at/after the limit is delivered with `range_err = 1`
(insn 0); cmdproc then reports `DRAM_RANGE` with `ERR_INFO = low32(CMD_ADDR + pc/2)` **only when
that pc is executed** (spec: 범위 오류는 그 명령어를 실행할 차례에만 보고). F3 prefetch depth
`BUF_BEATS` (default 16 beats = 32 instructions); a burst (≤ 64 beats, inside one 64-beat block,
below the limit) is requested only when its buffer space is reserved (R4), so `rd_rsp_ready` = 1.
F4 cmdproc waits for `idle` after `stop` before `run_finish` (DONE ⇒ memory port quiescent).
Fetch beats are counted in `PERF_CH_RD_BEATS[0]` (allowed to differ from the simulator).

### 3.2 tsr_dma_cmd_t (cmdproc → tsr_dma), 101 bits

`{is_st[1], ch[4], spm_line[32], dram_beat[32], n_lines[32]}`: LD (`is_st=0`) copies DRAM beats
`dram_beat .. +n_lines-1` of channel `ch` to SPM lines `spm_line .. +n_lines-1`; ST the reverse.
Guaranteed: `ch < NUM_CH`, `n_lines ≥ 1` (cmdproc retires `n_lines == 0` itself, no traffic),
`spm_line + n_lines ≤ SPM lines`, `dram_beat + n_lines ≤ DRAM limit`.
DMA rules: D1 bursts `len = min(left, 64 - addr % 64)` (never > 64, never across a 4 KiB
boundary); D2 one read id (local 0) per LD, data arrive in order; reserve SPM-write/FIFO room
(R4) so `rd_rsp_ready` only depends on local state; D3 ST: SPM reads (no backpressure, R4) into a
write-data FIFO; `wr_req` and `wr_dat` offered independently (§4 M5); `wr_dat_last` on each
burst's last beat; D4 `done` after the last SPM write accepted (LD) / the last write response
(ST). The DMA uses only the port of channel `ch`.

### 3.3 tsr_qact_cmd_t (cmdproc → tsr_qact), 133 bits

`{kind[1], dst_line[32], src_line[32], k[16], t[4], src_stride[32], qa_lines[16]}`.
Guaranteed: `kind` 0 = A32, 1 = A256; `0 < k ≤ K_MAX`, `k % (kind ? 256 : 32) == 0`;
`1 ≤ t ≤ T_MAX`; `src_stride` already resolved (`w5 == 0` → `k/16`); `qa_lines =
QA_LINES(kind, k)` (= destination stride); `dst_line + t*qa_lines ≤ SPM lines`;
`src_line + (t-1)*src_stride + k/16 ≤ SPM lines`. Numerics and layout: §8.1.

### 3.4 tsr_mvm_cmd_t (cmdproc → MVM unit, M3), 294 bits

`{fmt, acc, dst_line, x_line, w_off, n, k[16], t[4], dst_stride, y_lines, rows_per_ch,
row_beats, qa_lines[16]}` — all checks of §7 (MVM row) done, `dst_stride` resolved, `R =
ceil(N/NUM_CH)`, `ROW_BEATS(fmt, K)` and `QA_LINES` precomputed by cmdproc (it needs them for the
range checks). Ports exist on cmdproc (`mvm_cmd_*`, `mvm_done`) and are tied off in `tsr_core`.

### 3.5 tsr_csr ↔ tsr_cmdproc (run control)

| Signal | Dir | Meaning |
|---|---|---|
| `run_start` | csr→cmd | pulse when START is accepted; `run_cmd_addr[63:0]`, `run_cmd_count`, `run_args[8]` valid in that cycle only (cmdproc latches) |
| `run_abort` | csr→cmd | level; sampled by cmdproc before each instruction |
| `run_soft_reset` | csr→cmd | pulse (only while not BUSY): R1..R15 := 0, cur_pc := 0 |
| `run_finish` | cmd→csr | pulse; run ended **and** the core is quiescent (fetch idle, units idle) |
| `run_err_code[3:0]`, `run_err_pc`, `run_err_info` | cmd→csr | with `run_finish`; code 0 = END |
| `cur_pc[31:0]` | cmd→csr | PC CSR |
| `perf_retire` | cmd→csr | pulse per retired instruction (END included, faulting instruction not) |

### 3.6 Perf events (→ tsr_csr)

`perf_retire` (cmdproc), `perf_mvm_busy` (MVM unit, M3; 0), `perf_ch_rd_beat[c]` (arb c: memory
read beat transferred), `perf_ch_wait[c]` (arb c: ≥ 1 read outstanding and no read beat this
cycle). `PERF_CYCLES` counts cycles with STATUS.BUSY. VU/WAITM counters read 0.

---

## 4. Memory ports

Core boundary, one port per channel, packed `[NUM_CH]` vectors (channel *c* = index *c*):

| Channel | Signals (prefix `m_`) | Dir |
|---|---|---|
| read request | `rd_req_valid`, `rd_req_ready`, `rd_req_addr[ADDR_BEAT_W]`, `rd_req_len[7]`, `rd_req_id[4]` | core→mem |
| read response | `rd_rsp_valid`, `rd_rsp_ready`, `rd_rsp_data[512]`, `rd_rsp_id[4]`, `rd_rsp_last` | mem→core |
| write request | `wr_req_valid`, `wr_req_ready`, `wr_req_addr`, `wr_req_len[7]`, `wr_req_id[4]` | core→mem |
| write data | `wr_dat_valid`, `wr_dat_ready`, `wr_dat_data[512]`, `wr_dat_last` | core→mem |
| write response | `wr_rsp_valid`, `wr_rsp_ready`, `wr_rsp_id[4]` | mem→core |

Rules (asserted by `tb/common/mem_model.py`):
- M1 addresses in beats; `len` is the beat count 1..64 (7 bits, 0 never issued); a burst stays
  inside one 64-beat (4 KiB) block: `addr % 64 + len ≤ 64`; `addr + len ≤ DRAM limit`.
- M2 at most `RD_OUTSTANDING` read requests outstanding per channel (from the request transfer to
  the transfer of its `last` beat), summed over all sources — enforced by `tsr_mem_arb`.
- M3 ids: `id = {source[1:0], local[1:0]}`, source `0 FETCH`, `1 DMA`, `2 WS` (M3), `3` reserved.
  The memory returns read data in order **within one id**; beats of different ids may interleave
  (at beat granularity). Write responses: one per write request, in order within one id.
- M4 full-beat writes only (no byte strobes). Write data beats follow write-request order, no
  interleaving between bursts; `wr_dat_last` exactly on each burst's last beat.
- M5 the core never waits for `wr_req_ready` before offering `wr_dat_valid`, nor for
  `wr_dat_ready` before offering `wr_req_valid` (AXI-style independence; the model runs modes
  where the memory accepts data only after the request, or the request only after data).
- M6 sources reserve space before requesting (R4): a source's `rd_rsp_ready` may drop only for
  reasons local to that source, so the shared response channel never deadlocks across sources.
- M7 `STATUS.DONE` implies no transaction in flight (all reads returned, all write responses in).
- M8 every `m_*` output is driven by a flop.

`tsr_mem_arb` (one per channel) source side: read sources `s_rd_req_*[TSR_NSRC]`, per-source
`s_rd_rsp_valid`/`s_rd_rsp_ready`, shared `s_rd_rsp_data/last/id` (local id); write source = DMA
(`s_wr_*`). Arbitration round-robin per burst; channel 0 has FETCH + DMA (+ WS in M3), other
channels DMA (+ WS); unused sources are tied off in `tsr_core`.

---

## 5. SPM

- Geometry: `SPM_BYTES/64` lines × 512 bits, line-addressed, `SPM_AW = clog2(lines)`. Contents
  undefined at start (programs never read SPM they did not write); no reset of the array.
- Built from `tsr_sram_1r1w` (rtl/tech): `DEPTH, WIDTH, RD_LAT ≥ 1`, byte enables, read-first on
  same-address collision. U200 (M4): URAM288 cascades (512 b = 8 × 72-bit columns; 64 K lines =
  16 × 4 K deep); generic: behavioural array. Bank/cascade structure is the SPM agent's choice.
- Client ports (`TSR_SPM_CL_*`): 0 DMA, 1 QACT, 2 MVM (M3, tied off). Per client:
  `rd_req_valid/ready/line` → `rd_rsp_valid[c]` exactly `RD_LAT` (= `SPM_RD_LAT`, default 2)
  cycles after the accepting edge with the line on the shared `rd_rsp_data`, **no backpressure**
  (R4); `wr_valid/ready/line/data/be[64]` (M2 clients write full lines, `be` all ones).
- Single read and single write port in M2; fixed priority by client index. Only the dispatched
  unit drives requests in M2, so there is no contention, but `ready` may be low and must be
  honoured. Ordering: a read accepted after a write was accepted returns the new data (even
  across internal pipelining); same-cycle same-line read returns old data.
- Port plan: **M2** clients 0/1 time-share the 1R1W port (sequential). **M3** MVM uses client 2:
  activation broadcast (read QA lines once per MVM) and result gather (write y lines; ACC reads
  old y lines first) — still one active unit, same 1R1W port. **M6** VU needs 2 reads + 1 write
  per cycle: split the SPM into 2 line-interleaved banks (even/odd) or add a second read port;
  the client interface stays the same.

---

## 6. CSR block

CSR port protocol: master holds `csr_valid/addr/we/wdata` until `csr_ready`; the core performs
the access one or more cycles after it first sees `csr_valid` (tsr_csr: capture, S1 write side
effects / read select, S2 read mux — 4 cycles per transfer including the handshake edge) and
raises `csr_ready` (registered) for one cycle with `csr_rdata` (reads; 0 for writes); the
transfer ends at the edge with `csr_valid && csr_ready`. A write's effect is visible to every
later access and to the core when the transfer ends. Unmapped or unaligned offsets read 0, writes ignored. Map, CTRL/STATUS
bits, CFG fields: `tsr_pkg` (= spec CSR map; reference `tsr_sim_csr_read/_write`).
- CFG0 = `NUM_CH[7:0] | LANES[19:8] | T_MAX[23:20]`, CFG1 = `SPM_BYTES/64`, CFG2 = `VL[7:0] |
  NSREG[15:8] | (K_MAX/256)[31:16]`, CAPS = 0 (M2), ID = `0x54535231`, VERSION = `0x0100`.
- CTRL write handles SOFT_RESET, then START, then ABORT (simulator order):
  - SOFT_RESET when not BUSY: STATUS, ERR_*, PERF_* := 0, `run_soft_reset` (R1..R15, PC := 0);
    CMD_ADDR/CMD_COUNT/ARG* kept. **When BUSY** (not modelled by the simulator): treated as ABORT;
    the reset is applied when the run ends (instead of recording DONE/ERR). START in the same
    write is ignored. See proposal P3.
  - START when not BUSY: STATUS := BUSY, ERR_* := 0, abort flag := 0, `run_start`. START while
    BUSY is ignored (abort flag kept).
  - ABORT: abort flag := 1 (cleared by an accepted START and by SOFT_RESET). cmdproc checks it
    before each instruction: `ABORTED`, `ERR_PC` = first instruction not executed, `ERR_INFO` 0.
- `run_finish`: STATUS := DONE | (code ? ERR : 0), BUSY := 0, ERR_CODE/PC/INFO latched.
  `done` output = STATUS.DONE. STATUS W1C clears DONE/ERR only.
- PC: cmdproc sets `cur_pc := pc` after the ABORT check of instruction pc (so it reads
  CMD_COUNT after a past-the-end error; unchanged when aborted) — same as `sim->pc`.
- PERF: 64-bit counters at `0x100 + 8i` (LO) / `+4` (HI), `i < 5 + 2*NUM_CH`; cleared by reset and
  SOFT_RESET only. A LO read snapshots HI; the next HI read of that counter returns the snapshot.

---

## 7. Architectural checks (all in tsr_cmdproc, before any state change)

Order inside one instruction: `ILLEGAL_OP → BAD_FMT → K_RANGE → T_RANGE → SPM_RANGE →
DRAM_RANGE`; inside a range class, operands in field order A1, A2, A3. Range test for start
address `a` (32-bit `R[x] + imm`, wrapped) and extent `e` (computed in ≥ 36 bits, no wrap):
`a + e ≤ limit`; `ERR_INFO = a` of the failing operand, else 0. SPM limit = `SPM_BYTES/64`
lines, DRAM limit = `min(CH_BYTES/64, 2^ADDR_BEAT_W)` beats. Before decode (per pc, §3.1):
ABORT → past CMD_COUNT (`ILLEGAL_OP`, `ERR_PC = CMD_COUNT`) → fetch range (`DRAM_RANGE`).

| Instruction | ILLEGAL_OP if | then |
|---|---|---|
| `NOP` `END` `FENCE` | `w0[31:8] ≠ 0` or any of `w1..w7 ≠ 0` | END: finish OK |
| `LI` | `w0[27:8] ≠ 0` or `w2..w7 ≠ 0` | `R[rd] := w1` (rd = 0 discarded) |
| `ADDI` | `w0[27:20] ≠ 0` or `w0[15:8] ≠ 0` or `w2..w7 ≠ 0` | `R[rd] := R[ra] + w1` |
| `LD` `ST` | `w0[31:24] ≠ 0` (rc, rd) or `w0[15:12] ≠ 0` (flags) or `w4..w7 ≠ 0` or `sub ≥ NUM_CH` | SPM: `A1 = R[ra]+w1`, `e = w3`; DRAM: `A2 = R[rb]+w2`, `e = w3`; `w3 == 0` retires without traffic |
| `QACT` | `w0[31:24] ≠ 0` or `w0[15:12] ≠ 0` or `w6, w7 ≠ 0` or `sub ≥ 2` | `K = w3`: K_RANGE if `K == 0 ∨ K % bs ≠ 0 ∨ K > K_MAX` (bs 32/256); `T = w4`: T_RANGE if `T == 0 ∨ T > T_MAX`; SPM A1 `dst`, `e = T·QA_LINES`; SPM A2 `src`, `e = (T−1)·stride + K/16` with `stride = w5 ? w5 : K/16` |
| `MVM` (HAS_MVM=1, M3) | `w0[31:28] ≠ 0` or `w0[15:13] ≠ 0` or `w5[31:20] ≠ 0` or `w7 ≠ 0` | BAD_FMT `sub ≥ 2`; K_RANGE on `w5[15:0]`; T_RANGE on `w5[19:16]`; SPM A1 `dst`, `e = (T−1)·dstride + ceil(4N/64)` (`dstride = w6 ? w6 : ceil(4N/64)`); SPM A2 `x`, `e = T·QA_LINES`; DRAM A3 `w_off`, `e = ceil(N/NUM_CH)·ROW_BEATS` |
| `MVM` with HAS_MVM=0, `MULI RLD LOOP WAITM JUMP DQROW VOP RED ROPE TOPK SOP`, undefined | always | — |

Notes: all 32-bit fields (K, T, stride, n_lines) are checked in full width (e.g. `T = 0x80000001`
is T_RANGE; `(T−1)·stride` with `stride ≥ 2^31` overflows 32 bits and must still fail). QACT's
`T > T_MAX` check comes before any multiply, so `(T−1)·stride` needs at most 3×32 bits.
`QA_LINES`, `ROW_BEATS` and limits: `tsr_pkg` functions. `QA_MISMATCH` is never raised.

---

## 8. Numerics

### 8.1 QACT (tsr_qact, bit-exact `tsr_qact()` in sw/sim/tsr_ref.c)

Per token t: source `x[0..K-1]` = f32 at SPM `src_line + t·src_stride` (16 per line);
destination image at `dst_line + t·qa_lines`: QS (`ceil(K/64)` lines of int8 q), D
(`ceil(4·nblk/64)` lines, f32 per block), BS (A256 only, `ceil(2·K/16/64)` lines, int16 per 16 q).
Every line of the image is written whole; bytes after the last valid element of each region are 0.
- **A32** (block 32): `amax` = largest `|DAZ(x)|` over non-NaN x (as unsigned magnitude bits;
  0 if none); `d = fdiv(amax, 127.0)`; `id = (d == ±0) ? +0 : fdiv(1.0, d)`;
  `q = q8_round_away(fmul(x, id))`; `D = fp16_to_f32(f32_to_fp16(d))`.
- **A256** (block 256): `max` = `DAZ(x)` of the **first** non-NaN element with the strictly
  largest `|DAZ(x)|`; if `amax == 0`: `d = +0`, all q = 0, BS = 0; else `iscale = fdiv(−127.0,
  max)`, `q = q8_round_even(fmul(iscale, x))`, `d = fdiv(1.0, iscale)`; `BS[j] = Σ q[16j..16j+15]`.
- fp ops are the spec rules (DAZ inputs, RNE, FTZ of subnormal *rounded* results, canonical NaN
  `0x7FC00000`); `q8_round_*`: NaN → 0, round, saturate to [−128, 127] (A32 half away from zero,
  A256 half to even). Special-value consequences (A32 `amax = Inf` → `D = Inf`, q = 0; A256
  `max = ±Inf` → `d = ∓Inf`, q = 0; tiny `max` → `iscale = ∓Inf`, saturated q, x = 0 → NaN → 0)
  follow from these rules — no special cases in hardware.
- Throughput target M2: ≥ 1 source line per ~2 cycles outside the per-block divides (an
  iterative divider per block is acceptable in M2; overlap blocks if cheap).

### 8.2 fp library (tsr_fp_*)

- `tsr_fp_pkg` (functions, combinational, one cycle each): `tsr_f32_daz`, `tsr_f32_is_nan`,
  `tsr_f32_amag`, `tsr_fp16_to_f32`, `tsr_f32_to_fp16`, `tsr_f32_to_q8(x, half_even)`,
  `tsr_i32_to_f32` (M3).
- `tsr_fp_mul #(LAT)`: fixed latency, `ce` stall, `in_valid → out_valid`.
- `tsr_fp_div`: iterative, `in_valid/in_ready`, `out_valid/out_ready`, correctly rounded; recip
  = `div(1.0, x)` (bit-identical to `tsr_frecip`).
- M3 adds `tsr_fp_add` (term accumulation, ACC) with the same rules.
- Mandatory vectors (spec 구현 명확화): `0x3F7FFFFF × 0x00800000 = 0x00800000`,
  `0x3F1FA0BA × 0x00CD4708 = 0x00800000`, `0x00FFFFFF / 2 = 0x00800000`,
  `0x3F7FFFFE × 0x00800000 = 0`; plus random and exhaustive (fp16, q8 over all exponents) sweeps
  against `common.tsr_ref`.

---

## 9. Verification infrastructure

- `rtl/lint.sh` — lint gate (§1). `tb/run_all.sh [--no-lint] [pytest args]` — lint + all tests.
- `python3 -m pytest tb [--tsr-seed N] [--waves] [--tsr-traces DIR] [-k pattern]`.
  Build directories: `$TSR_SIM_BUILD_ROOT` (default `build/cocotb`; agents set it to their
  scratch), `make -j$TSR_BUILD_JOBS` (default 4). One simulation build at a time per agent.
- Per block: `tb/<block>/test_<block>.py` (unique basename, no `__init__.py`) holding the cocotb
  tests and a pytest function `run("<toplevel>", __name__, parameters=..., extra_sources=[...])`.
  Test-only wrappers go in `tb/<block>/hdl/`. Run reduced configs where useful (small SPM, 1–3
  channels, T_MAX 8).
- `tb/common`:
  - `isa.py` — constants/encoders/struct layouts mirroring `tsr_pkg.sv` (`DMA_CMD_T.pack(...)`
    etc. to drive struct ports; widths/field order are checked against the RTL by the self-test).
  - `tsr_ref.py` — `ref()` → ctypes `libtsr_ref.so` (`build/dev/sw/sim/`): `fadd/fsub/fmul/fdiv/
    frecip/fsqrt/daz/i32_to_f32`, `fp16_to_f32`, `f32_to_fp16`, `q8_round_away/even`,
    `qact(kind, x_bits, K) → QA image`, `qa_split`, `qa_*lines`, `row_beats`, `dot_row`,
    `tree16`. All f32 values are 32-bit patterns (NaN payloads preserved).
  - `trace.py` — trace directories (tsr_trace.h): `load`, `find`, `load_all`, `Trace.core_parameters()`,
    `initial_dram()`, `allowed_read_beats()`, `expect_beats()`, `disasm()`.
  - `mem_model.py` — `MemModel(dut, num_ch=, addr_w=, rd_outstanding=, limit_beats=, seed=, ...)`
    or `MemModel.for_trace(dut, tr)`; random latency/backpressure/stalls, per-id ordering,
    cross-id interleave, write ordering modes, protocol assertions (§4), `written/read/
    read_unwritten` beat sets, `check(idle=True)`.
  - `csr_driver.py` — `CsrDriver(dut)`: `read`, `write`, `read64`, `start_run`, `final_state`, `perf`.
  - `replay.py` — `await run_trace(dut, tr)`: full trace replay with every comparison of the
    tsr_trace.h rules (final CSRs, expect data, written set == expect set, reads ⊆ init ∪ cmd).
  - `pycore.py` — `PyCore`: Python statement of §3.1/§3.2/§7 (fetch, burst split, every check in
    order, LI/ADDI, LD/ST, QACT via tsr_ref) running over a real memory port (`SeqMaster`). The
    self-test replays `traces/m2` with it through `MemModel` + `replay.compare` (172/172 checked),
    so the harness is proven before the RTL exists; cmdproc/DMA agents can diff against it.
  - `runner.py` — `run(toplevel, module, ...)`; `tbutil.py` — clock/reset/seed.
  - `test_common.py` — the infrastructure self-test (`pytest tb/common`).
  - Timing discipline for TB components: drive right after a rising edge, sample at the falling
    edge. (With Verilator, values read inside a RisingEdge callback can already include that
    edge's register updates; falling-edge sampling avoids the ambiguity.)
- `traces/m2/` (regenerate: `traces/m2/regen.sh`; generator `tsr-trace-gen --ops m2`), 172 traces,
  all replayed by `tsr-trace-replay`:

  | Set | Config | Traces | Notes |
  |---|---|---|---|
  | `tmax1` | U200 v1 | 60 | ~15 % error endings, K edges (32/256/K_MAX), copies |
  | `tmax8` | U200 v1, T_MAX 8 | 16 | multi-token, src_stride gaps / 0-encoding |
  | `small1ch` | 1 ch, 64 KiB SPM, 1 GiB ch, ADDR_BEAT_W 22, K_MAX 4096, T_MAX 2 | 16 | DRAM limit < channel size |
  | `ch3` | 3 ch, 256 KiB SPM, T_MAX 3 | 8 | non-power-of-2 NUM_CH |
  | `errors` | U200 v1 | 40 | every trace ends in an error (all M2-reachable kinds) |
  | `errors8` | U200 v1, T_MAX 8 | 32 | all errors, multi-token range checks incl. `(T−1)·stride ≥ 2^32` |

  Programs cover register-relative addressing on every address field (incl. ARG registers and
  32-bit wrap), LI/ADDI on R0..R15 (R0 writes discarded), NOP/FENCE, both QA kinds (pairs on one
  source), T up to T_MAX, `src_stride` 0/explicit with gaps, special activation values, LD/ST up
  to 320 lines across 4 KiB boundaries, operands ending exactly at the SPM end and the DRAM limit,
  zero-length LD/ST at the limits; error endings: reserved fields of every M2 instruction, non-M2
  opcodes, channel ≥ NUM_CH, QACT kind ≥ 2, K/T/SPM/DRAM range (with lower-priority violations in
  the same instruction), fetch beyond the limit mid-program and at CMD_ADDR (incl.
  `CMD_ADDR_HI ≠ 0`), missing END.
- `tb/core/test_core.py` (core agent) — the core-level tests of §12: trace replay of every
  `traces/m2` configuration (one build per configuration, traces back to back without reset,
  several memory regimes), implementation-parameter variants, directed tests and random
  programs against the ISA simulator in lockstep. It reuses `tb/common/simlib.py` (simulator
  through ctypes) and `ProgGen` from `tb/common/proggen.py` (moved from tb/cmdproc in M3; tb/cmdproc
  keeps re-exports).

---

## 10. Room for M3 (MVM clusters)

- `tsr_mem_arb` source `TSR_SRC_WS` per channel (one cluster weight streamer per channel, id
  `{2, local}`), sharing `RD_OUTSTANDING` with DMA/fetch; streamers reserve buffer space (R4).
- SPM client `TSR_SPM_CL_MVM`: QA broadcast reads and y gather writes (+ ACC reads).
- `tsr_mvm_cmd_t` and `mvm_cmd_*`/`mvm_done` on cmdproc; checks of §7 MVM row (HAS_MVM = 1).
- `perf_mvm_busy` input of tsr_csr. `tsr_fp_add`, `tsr_i32_to_f32` in the fp library.
- Clusters live per channel (SLR placement) between the arbiter's WS port and a broadcast/gather
  bus to the SPM client; pipeline depth across SLRs is a parameter (spec 배치).

---

## 11. Spec proposals (for the lead)

- P1 **CSR port handshake** (spec lists only the signals): "the master holds valid/addr/we/wdata
  until ready; ready is a one-cycle completion strobe and rdata is valid with it; unaligned or
  unmapped offsets read 0 and ignore writes."
- P2 **Memory port details**: `len_beats` is a count 1..64; id = 4 bits `{source, local}`;
  responses in order per id, read beats may interleave across ids; one write response per write
  request; full-beat writes (no strobes); request and data channels independent (core never waits
  for one handshake before offering the other).
- P3 **SOFT_RESET while BUSY**: acts as ABORT at the next instruction boundary, then the soft
  reset is applied when the run ends (STATUS goes from BUSY to 0, no DONE); START in the same
  write is ignored. The simulator cannot observe this case (CSR access may not race its run).
- P4 **DONE ⇒ quiescent**: STATUS.DONE (and `done`) rise only after all memory transactions of
  the run completed, including discarded instruction prefetches.
- P5 **PERF semantics in hardware**: `PERF_CYCLES` = cycles with BUSY; `PERF_CH_WAIT[c]` = cycles
  with ≥ 1 outstanding read on channel c and no read beat; a LO read snapshots HI.
- P6 **CAPS and MVM-less bitstreams**: M2 test bitstreams lack MVM but CAPS cannot say so (CAPS
  lists optional M6+ features). Either keep it (M2 bitstreams are never used with llama.cpp) or
  assign the reserved CAPS bit1 later. No change proposed for v1.
- P7 **Reset**: state that `rst_n`/`ap_rst_n` is synchronous active-low and that SPM contents
  survive SOFT_RESET but are undefined after a hardware reset (already implied by "SPM 초기값").
- P8 **PERF fetch beats** (core agent): `PERF_CH_RD_BEATS[0]` mixes LD beats with instruction
  fetch beats, so the spec's rule "LD (and MVM weight) beats per channel equal the hardware"
  cannot be checked on channel 0 (fetch beats may differ). Either exclude fetch beats from
  `PERF_CH_RD_BEATS` or add a `PERF_FETCH_BEATS` counter after the per-channel ones (index
  `5 + 2*NUM_CH`, so existing indices stay). Not needed for M2 correctness.
- P9 **CSR reads while BUSY**: state that every CSR may be read at any time without side effects
  (except the PERF LO/HI snapshot of P5) — the XRT host polls STATUS and PERF during a run;
  tb/core core_random reads random CSRs during half of its runs.
- P10 **Hardware reset**: in addition to P7, state that after `ap_rst_n` every CSR reads as after
  power-up (CMD_ADDR/CMD_COUNT/ARG* = 0, unlike SOFT_RESET) and that the memory side must be reset
  together with the core (outstanding transactions are dropped, not completed).

---

## 12. Core integration (M2 status, core agent)

`tsr_core.sv` keeps the §2 netlist unchanged; it adds simulation-only integration checks
(`ifndef SYNTHESIS`, armed by the first reset edge like the block checks):
- I1 one instruction in flight: at most one unit is offered a command, never while a unit holds
  one; `done` only from the unit that holds the instruction; no MVM command with `HAS_MVM = 0`.
- I2 a unit drives SPM requests only while it holds an instruction (its `done` really means its
  SPM effects are complete); DMA and QACT never use the SPM in the same cycle.
- I3 memory requests / write data only during a run; in the first cycle of `done` no request or
  write data is offered and no read is outstanding on any channel (P4 / M7). Write responses are
  checked by the memory model (`MemModel.check(idle=True)` at DONE).

Integration changes outside the netlist (no functional RTL bug was found at core level):
`tsr_fp_div.sv`'s simulation assertion is now armed by a reset edge like the other blocks' (it
could fire at 0 ns on Verilator's first evaluation for some seeds, failing any simulation that
contains tsr_qact; tb/qact's xfail mark for it is removed); `rtl/lint.sh` also lints every
`tb/*/hdl` toplevel and tsr_core with edge implementation parameters.

**Core tests** (`tb/core/test_core.py`; `python3 -m pytest tb/core`, 2-3.5 min with warm builds;
`tb/run_all.sh` = lint + every block + core: 84 passed, 4 opt-in skipped, 532 s wall with every
Verilator build cold, one simulation at a time):

| pytest item | Build | Content |
|---|---|---|
| `test_core_traces[errors+tmax1]` | U200 v1 | 100 traces x 3 memory regimes, 100 random programs |
| `test_core_traces[errors8+tmax8]` | U200 v1, T_MAX 8 | 48 traces x 3 regimes, 100 random programs |
| `test_core_traces[small1ch]` | 1 ch, 64 KiB SPM, ADDR_BEAT_W 22 | 16 traces x 3 regimes, 150 random |
| `test_core_traces[ch3]` | 3 ch, 256 KiB SPM, T_MAX 3 | 8 traces x 3 regimes, 150 random |
| `test_core_impl[lat4_u200]` | U200 v1, SPM_RD_LAT 4 | 100 traces, 60 random |
| `test_core_impl[lat3_t8]` | T_MAX 8, SPM_RD_LAT 3, FETCH_BEATS 64, RD_OUTSTANDING 4 | 48 traces, 60 random |
| `test_core_impl[lat1_small]` | small1ch, SPM_RD_LAT 1, FETCH_BEATS 2, RD_OUTSTANDING 1 | 16 traces x 3, 150 random |
| `test_core_impl[lat4_ch3]` | ch3, SPM_RD_LAT 4, FETCH_BEATS 4, RD_OUTSTANDING 2 | 8 traces x 3, 150 random |
| `test_core_directed[c2]` | 2 ch, T_MAX 4, 64 KiB SPM, RD_OUTSTANDING 4 | every directed test, 300 random |
| `test_core_directed[tiny]` | 2 ch of 32 beats, 64-line SPM, K_MAX 256 | CSR map, 500 random (limits everywhere) |
| `test_core_directed[u200]` | U200 v1 | throughput, STATUS W1C, MVM illegal, PERF |

- Trace replay = `common.replay.run_trace` (STATUS, ERR_CODE, ERR_PC, ERR_INFO, expect regions
  bit-exact, written-beat set == expect beats, reads inside init / command beats, quiescent port at
  DONE) plus PERF deltas: retired == the trace's, `PERF_CH_RD_BEATS[c >= 1]` == the trace's (LD
  beats), channel-0 counter == beats seen at the port, MVM / VU / WAITM 0. Memory regimes: `rand`
  (MemModel defaults), `fast` (always ready, latency 0..2), `slow` (read latency 20..160, write
  latency 10..120, heavy backpressure, stalls), `ordered` (no cross-id interleave, write data only after its request).
- Random programs: `ProgGen` (tb/common) on the RTL and the simulator in lockstep (the RTL SPM is
  first filled with the simulator's 0xA5 pattern through an LD); final CSRs, PC, R1..R15, retired,
  LD beats, DRAM and the whole SPM (ST dumps) compared. Programs whose executed QACT overlaps its
  destination and source are skipped: undefined (spec 겹치는 피연산자) and really different here
  (tsr_qact reads ahead of its writes; the simulator works token by token) — the tiny
  configuration produced such a mismatch before the filter.
- Directed: CSR constants / full CSR map vs the simulator, STATUS W1C and START while BUSY,
  SOFT_RESET idle / with START / while BUSY at random points (P3), hardware reset at random points
  of a run (P10), ABORT at random cycle offsets (effects == simulator run of the executed prefix,
  ERR_PC, PC, retired), back-to-back programs, MVM at M2 = ILLEGAL_OP with prefix effects, PERF.
- Mutation check of the core tests (scratch copies of the RTL): wiring mutants of tsr_core (SPM
  response swap, perf wiring, abort tie-off, fetch id, fetch channel, QACT write enables, QACT SPM
  latency, MVM perf input) and unit mutants seen through the core (arbiter outstanding limit + 1,
  ST done before its write responses, QACT BS / D tail zeroing, CFG0, STATUS W1C, SOFT_RESET while
  BUSY) are all killed. The two survivors are equivalent: the DMA uses SPM_RD_LAT only to size its
  queue (8 entries for 2 and for 3), and tsr_fetch ignores `rd_rsp_last`.

**Throughput** (U200 v1, ideal memory: always ready, latency 0..1; `core_throughput`):
LD / ST of 4096 lines 1.01 cycles per beat (one channel at a time, ~38 cycles of overhead per
instruction); QACT A256 K 32768 1.11 cycles per source line, A32 8.03 (divider bound);
NOP / LI / ADDI 4.25 cycles each; a 1-line LD 19 cycles, a 1-line ST 21 cycles.

**Timing at 300 MHz (U200)** — whole-core out-of-context place and route with Vivado 2024.2
(`tb/core/viv/core_ooc.tcl`, xcu200-fsgd2104-2-e, 3.333 ns, I/O budget 1.6 ns, no pblocks, so the
SLR crossings of the real floorplan are not modelled; post-route estimates only):

| Build | SRAM mapping | SPM_RD_LAT | WNS / WHS (ns) | Worst paths |
|---|---|---|---|---|
| U200, SPM 256 KiB | generic -> 8 URAM | 2 | +0.121 / +0.015 | QACT multiplier operands; SPM URAM -> DMA queue +0.45 |
| U200 v1 (4 MiB) | generic -> 1032 RAMB36 | 2 | -2.150 / +0.010 | SPM BRAM read -> DMA queue / QACT; SPM write enables |
| U200 v1 (4 MiB) | generic -> 1032 RAMB36 | 4 | -2.189 / +0.028 | SPM write address / enable fan-out into 1032 BRAMs |
| U200 v1 (4 MiB) | URAM (`ram_style ultra`, scratch) | 2 | -0.209 / +0.038 | URAM cascade read -> DMA queue, QACT response reg |
| U200 v1 (4 MiB) | URAM (`ram_style ultra`, scratch) | 4 | **+0.120 / +0.010** | QACT multiplier operands; SPM write address -> URAM cascade +0.134 |

Utilisation (U200 v1, URAM, SPM_RD_LAT 4): 16.4 K LUT, 22.7 K FF, 32 DSP, 136 URAM (128 SPM + 8
QACT line buffer), 1525 LUTRAM, no latches, no critical warnings.

Consequences for M4 (structures expected to be critical, in order):
1. SPM: the generic behavioural `tsr_sram_1r1w` is mapped to block RAM by Vivado (1032 RAMB36 for
   4 MiB) and fails by > 2 ns. `rtl/tech/xilinx/tsr_sram_1r1w.sv` must force URAM (attribute or XPM)
   and the U200 build must use `SPM_RD_LAT 4`. Both are drop-in: every M2 trace and random test
   passes with SPM_RD_LAT 1, 2, 3 and 4 (`test_core_impl`); with URAM and 4 the core meets 300 MHz
   out of context. The QACT line buffer (64 lines) should stay in BRAM (the attribute here also
   moved it to URAM).
2. DMA <-> channel arbiters across SLRs: one DMA engine serves all channels while each
   `tsr_mem_arb` sits next to its DDR4 controller (three SLRs on U200). LD data goes arbiter
   response flop -> SLL crossing -> 4 x 512-bit channel select -> DMA queue write in one cycle; ST
   data is one shared 512-bit payload flop fanned out to every arbiter. Not covered by the
   out-of-context runs. Every DMA <-> arbiter stream is valid/ready and latency-insensitive (the
   DMA tolerates any read latency), so M4 can add per-channel register slices (Laguna) without
   logic changes; give that path a depth parameter.
3. SPM write side: 512-bit data, 64 byte enables and the address from the SPM input register to
   32 URAMs per bank (+0.134 ns at 4 MiB): keep the replicated input registers.
4. QACT: the merged multiplier scale register drives all 32 DSPs (+0.12 ns, as in the QACT
   agent's run); `-keep_equivalent_registers` or a per-lane copy helps.
5. Synchronous reset fan-out into three SLRs: the platform layer should deliver a pipelined,
   per-SLR replicated `rst_n`.
6. cmdproc / fetch / CSR / PERF counters: comfortable (>= +0.5 ns in the passing whole-core runs).

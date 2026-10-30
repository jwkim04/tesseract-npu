# Tesseract core RTL: M2 infrastructure, interfaces and contract

This document is the RTL contract of the core infrastructure: the conventions every block follows, the
internal interfaces between the dispatch side and the units, the memory ports, the SPM, the CSR block, the
numerics units (QACT and the fp library), and the verification infrastructure. The MVM unit has its own
document, [`rtl-m3.md`](rtl-m3.md).

> **Scope.** Record of the RTL of tag `v1-m4` (M2 core infrastructure) and of what the tree keeps after X0
> (`452e27b`): the datapath units, a CSR skeleton and the platform adapter. `tsr_fetch` and `tsr_cmdproc` are
> deleted; the v2 issue path, control core, vector unit, attention modes, banked SPM and debug block are
> *designed* ([CC](v2/04-control-core.md) §13) and *planned* for RTL (W4). Authority: `docs/spec.md` >
> simulator (`sw/sim`) > `docs/v2` > this file > `rtl/core/tsr_pkg.sv`. Section numbers are cited by RTL and
> testbench comments; v1-only sections are stubs (`git show v1-m4:docs/rtl-m2.md`).

Status words: *kept* (in the tree, unchanged), *deleted* (X0), *designed* (in `docs/v2`, no RTL), *planned*
(W4). The tree holds `tsr_dma`, `tsr_qact`, `tsr_mvm*`, `tsr_spm`, `tsr_mem_arb`, the fp units, `tsr_csr` as a
skeleton (`ID`, `VERSION`, `CFG*`, `CAPS`, `PERF_*`) and the platform adapter; the units' command ports are
idle until the v2 dispatcher exists.

**M2 scope (v1).** An RTL core that executes `NOP END LI ADDI LD ST QACT FENCE` bit-exactly like the
simulator, including every error those instructions can raise, the CSRs, command fetch and the perf
counters (present, not bit-compared). `MVM` (M3) was absent from M2: `tsr_core #(.HAS_MVM(0))` decoded
opcode 0x20 as `ILLEGAL_OP`. Every structure that M3 needs (weight-stream source per channel, MVM SPM
client, MVM dispatch struct, MVM perf counter) existed as a tied-off port. In the tree the MVM unit is
built by default (`HAS_MVM` = 1) and no unit receives a command until the v2 dispatcher exists.

---

## 1. Conventions (all blocks)

- **Language/lint**: SystemVerilog-2017, `rtl/lint.sh` (Verilator 5 `--lint-only -Wall`, every module as
  its own top plus reduced and edge `tsr_core` configurations, the Xilinx technology wrappers, the XRT
  platform and the M0 bandwidth kernel) must stay clean. No blanket waivers. The waivers in `rtl/core` are
  `UNUSEDPARAM` around the constant catalogues of the packages (`tsr_pkg`, `tsr_tables_pkg`). Unused
  inputs: `wire unused_ok = &{1'b0, a, b};` (Verilator's default `*unused*` regexp). Vendor
  primitives/attributes only in `rtl/tech/<target>/`.
- **Clock/reset**: one clock `clk`. `rst_n` is synchronous, active low, already synchronised by the
  platform layer. Reset every control flop (valid bits, FSM state, counters, pointers); wide datapath
  registers need no reset. Verilator builds randomise initial state (`--x-initial unique`,
  `+verilator+rand+reset+2`), so missing resets show up in simulation. v2 adds a second clock `clk_rv`
  for the control core ([`ctrl-riscv.md`](ctrl-riscv.md) §7; designed, W4); every crossing is a CDC
  structure of `rtl/core/cdc/` (planned), and Ibex and riscv-dbg keep their own reset style (CC §3.1).
- **valid/ready** (all streams, internal and external):
  - R1 transfer happens at a rising edge where `valid && ready`;
  - R2 `valid` must not depend combinationally on `ready`; once raised, `valid` and the payload stay
    unchanged until the transfer (the memory model asserts this on the core boundary);
  - R3 `ready` may depend on `valid`, but prefer registered `ready` across module boundaries;
  - R4 a requester of a stream **without backpressure** (SPM read data, fixed-latency fp pipelines) or of
    a stream it must never stall (memory read data, see §4 M6) reserves space for the response before it
    issues the request;
  - R5 no combinational path from a module's inputs to its outputs across the boundaries listed in §3-§5
    unless stated;
  - R6 1-cycle "pulse" signals (`done`, ...) are registered.
- **Timing (300 MHz, U200)**: every `m_*` core-boundary output comes from a flop (tsr_mem_arb); memory
  ports cross SLRs, so arb and DMA may add pipeline stages freely (parameterise depth). Keep logic depth at
  about 6 LUT levels; wide compares/adds (range checks) are pipelined over cycles in the checker (`tsr_chk`
  in v2, `tsr_cmdproc` in v1; instruction rate is not critical).
- **Data byte order**: byte *j* of a 64-byte beat/line is bits `[8j+7:8j]` (little-endian); f32 element *i*
  of a line is bits `[32i+31:32i]`; instruction word *wi* is `insn[i]` (`tsr_insn_t = logic [7:0][31:0]`).
- **Widths**: ISA-level quantities (addresses, lengths, strides) are 32-bit fields in the dispatch structs
  regardless of parameters; the checker's range checks guarantee the bits a unit does not use are zero
  (units slice what they need, e.g. `cmd.spm_line[SPM_AW-1:0]`).
- **Parameters**: `tsr_core` takes the spec table (`NUM_CH BEAT_BITS LANES T_MAX K_MAX SPM_BYTES VL NSREG
  RD_OUTSTANDING CH_BYTES ADDR_BEAT_W`, defaults in `tsr_pkg::TSR_DEF_*`: 4 channels of 16 GiB, a 4 MiB
  SPM, `K_MAX` 32768, `VL` 16, `NSREG` 16 and `T_MAX` 1; the k4 bitstream sets `T_MAX` 8) plus implementation
  parameters `HAS_MVM` (default 1), `SPM_RD_LAT` (default 2; the U200 bitstreams use 4, §12), the `MVM_*`
  latencies and pipe depths ([`rtl-m3.md`](rtl-m3.md) §1) and `RST_PIPE` (§11 P10). v1's `NIREG`,
  `LOOP_DEPTH` and `FETCH_BEATS` are gone. `CH_BYTES` is `longint unsigned` (pass
  `-GCH_BYTES="64'd1073741824"`). Elaboration checks (`$fatal`) mirror `tsr_sim_create`: `BEAT_BITS ==
  512`, `NUM_CH` 1..16, `T_MAX` 1..8, `K_MAX` a multiple of 256 in [256, 65280], `SPM_BYTES` a positive
  multiple of 64, `NSREG` 1..16, `VL` 1..255, `LANES` 1..4095, `CH_BYTES` a positive multiple of 64,
  `ADDR_BEAT_W` 1..40, `RD_OUTSTANDING` at least 1; with `HAS_MVM = 1` also `LANES == 128`, `MVM_WS_FIFO` a
  power of two of at least 64 and the MVM latencies in range ([`rtl-m3.md`](rtl-m3.md) §1).
- **Files**: `rtl/filelist.f` (generic target, used by lint and the cocotb runner; packages first) and
  `rtl/filelist_xilinx.f` (swaps `tech/generic` for `tech/xilinx`). New modules go into them; both are shared
  files. `rtl/core/tsr_tables_pkg.sv` and the SDK/simulator tables are generated from one source
  (`sw/tables/tsr_tables.py`, `gen_tables.py --check` runs in lint).

---

## 2. Block diagram and ownership

<p align="center"><img src="fig/core.svg" alt="Block diagram of tsr_core in two clock domains. The clk_rv domain holds the control core (Ibex) with its instruction and data memories, the debug module and the control window. The core clock domain holds the CSR block (host port, launch queue, completion ring, log ring, PERF), the issue FIFO crossing, the checker tsr_chk, the dispatcher with its unit queues feeding the DMA, QACT, MVM clusters and vector unit, the scratchpad tsr_spm with the RISC-V bridge, and one memory arbiter per DDR4 channel. Blocks of the v1 core that stay are drawn solid; v2 additions are drawn as planned." width="100%"></p>

The figure shows the v2 core. Solid blocks exist in the tree (the units, the SPM, the arbiters, the CSR
skeleton); the control core, the issue path, the vector unit, the debug block and the CDC structures are
planned (W4). The v1 core had, instead of the control core and the issue path, an instruction-fetch unit
(`tsr_fetch`, prefetch over channel 0) feeding a command processor (`tsr_cmdproc`: checks, the R0-R15
register file, `LI`/`ADDI`, dispatch, errors) that offered one checked command at a time to the DMA, QACT
and MVM units; both modules are deleted. The X0 netlist (`tsr_core.sv`) has the CSR port, the DMA, QACT and
MVM units, the SPM, and one `tsr_mem_arb` per channel (read sources AUX tied off, DMA, weight stream; write
source DMA); the units' command ports are idle until the dispatcher arrives.

| Block | Files | State in the tree | Contract |
|---|---|---|---|
| (a) control | `rtl/core/tsr_csr.sv` | skeleton: `ID`, `VERSION`, `CFG*`, `CAPS`, `PERF_*`; rewritten in W4 | §3.5, §6 |
| (a) v1 control | `tsr_cmdproc.sv`, `tsr_fetch.sv`, `tb/cmdproc/` | deleted (X0); checks continue in `tsr_chk` (planned) | §3.1, §3.5, §7 |
| (b) memory | `tsr_dma.sv`, `tsr_mem_arb.sv` (with `tsr_rslice`), `tsr_rpipe.sv`, `tsr_vpipe.sv`, `tb/dma/` | kept; KVST and the LD burst reservation are designed | §3.2, §4 |
| (c) fp library | `tsr_fp_pkg.sv`, `tsr_fp_mul.sv`, `tsr_fp_div.sv`, `tsr_fp_add.sv`, `tb/fp/` | kept | §8.2 |
| (d) SPM | `tsr_spm.sv`, `rtl/tech/{generic,xilinx}/tsr_sram_1r1w.sv`, `tb/spm/` | kept (three clients, 1R1W, fixed priority); the 2R2W banked SPM is designed | §5 |
| (e) QACT | `tsr_qact.sv`, `tb/qact/` | kept | §3.3, §8.1 |
| MVM unit | `tsr_mvm*.sv`, `rtl/tech/*/tsr_mac_lane.sv`, `tb/mvm*/` | kept; attention modes designed | [`rtl-m3.md`](rtl-m3.md) |
| (f) top | `tsr_core.sv` | the X0 netlist | §1, §2, §9 |
| infra | `tsr_pkg.sv`, `tsr_tables_pkg.sv`, filelists, `rtl/lint.sh`, `tb/common/`, `tb/conftest.py`, `tb/pytest.ini`, `tb/run_all.sh`, `traces/m2/`, this file | kept | all |

The ownership column of the first revision named the module agents; the contract comments in the sources
carry the final port lists. Port-list changes are interface changes: update this file, `tsr_pkg.sv`,
`tsr_core.sv` and `tb/common/isa.py` together.

---

## 3. Internal interfaces

### 3.0 Dispatch protocol (dispatcher → unit), common to DMA, QACT and MVM

In v1 the sender was `tsr_cmdproc`; in v2 it is the queue head of each unit (CC §9.1: `cmd_q` / `cmd_v_q`
drive the unit's `cmd_valid` / `cmd`). The protocol is unchanged.

| Signal | Dir (unit) | Meaning |
|---|---|---|
| `cmd_valid` | in | the dispatch side offers one checked command (R2: held until `cmd_ready`) |
| `cmd_ready` | out | unit idle and able to start (registered) |
| `cmd` | in | `tsr_<unit>_cmd_t`, valid with `cmd_valid`; the unit latches it at the transfer |
| `done` | out | 1-cycle pulse: **all architectural effects complete and visible** |

- A unit holds one command at a time. v1 offered no command before the previous unit's `done` (exactly one
  instruction in flight in the whole core). v2 runs the units concurrently under in-order queues: a unit
  holds at most the ops it accepted and completes them in accept order (CC §9.4 invariant I3: at most 2
  outstanding for the MVM, 3 for the VU, 1 for every other unit).
- `done` is never in the cycle of the accepting edge (at least 1 cycle later). "Visible" means: every SPM
  line write of the instruction has been accepted by `tsr_spm` (§5 ordering guarantees later reads see it)
  and, for ST, every DRAM write response has been received.
- Units never report errors: every architectural check is done before dispatch (§7); operand values a
  unit receives are always legal. Units may `assert` legality in simulation.
- No flush/cancel: a dispatched command always runs to completion. v2's QABORT drops queue entries that
  have not started (EX §8, CC §9.5); it never cancels a running op.

### 3.1 tsr_fetch ↔ tsr_cmdproc

**History: v1 (deleted in X0).** `tsr_fetch` prefetched the command buffer over channel 0 (default 16 beats,
32 instructions) and delivered instructions `pc = 0, 1, ...` to `tsr_cmdproc` as `tsr_fetch_insn_t =
{range_err, pc, insn}`, with a `start` / `stop` / `idle` handshake, a fetch-range rule (a beat at or above
the DRAM limit is delivered with `range_err`, reported only when that `pc` is executed) and the guarantee
that a run ends with the memory port quiescent. v2 has no instruction fetch from the host: the control core
fetches from its ITCM (CC §4). Fetch beats are gone from `PERF_CH_RD_BEATS[0]`. See
`git show v1-m4:docs/rtl-m2.md` §3.1.

### 3.2 tsr_dma_cmd_t (dispatcher → tsr_dma), 101 bits

`{is_st[1], ch[4], spm_line[32], dram_beat[32], n_lines[32]}`: LD (`is_st=0`) copies DRAM beats
`dram_beat .. +n_lines-1` of channel `ch` to SPM lines `spm_line .. +n_lines-1`; ST the reverse.
Guaranteed: `ch < NUM_CH`, `n_lines ≥ 1` (the checker retires `n_lines == 0` itself, no traffic),
`spm_line + n_lines ≤ SPM lines`, `dram_beat + n_lines ≤ DRAM limit`.
DMA rules: D1 bursts `len = min(left, 64 - addr % 64)` (never > 64, never across a 4 KiB boundary); D2
one read id (local 0) per LD, data arrive in order; reserve SPM-write/FIFO room (R4) so `rd_rsp_ready`
only depends on local state; D3 ST: SPM reads (no backpressure, R4) into a write-data FIFO; `wr_req` and
`wr_dat` offered independently (§4 M5); `wr_dat_last` on each burst's last beat; D4 `done` after the last
SPM write accepted (LD) / the last write response (ST). The DMA uses only the port of channel `ch`.
v2 changes (designed): the LD reserves queue space for a whole burst before it requests (queue 512 beats,
`rd_rsp_ready` constant 1; CC §10.4, V2-CC-13), and KVST (a store of quantized KV rows, AT) joins LD and ST.

### 3.3 tsr_qact_cmd_t (dispatcher → tsr_qact), 133 bits

`{kind[1], dst_line[32], src_line[32], k[16], t[4], src_stride[32], qa_lines[16]}`.
Guaranteed: `kind` 0 = A32, 1 = A256; `0 < k ≤ K_MAX`, `k % (kind ? 256 : 32) == 0`; `1 ≤ t ≤ T_MAX`;
`src_stride` already resolved (`w5 == 0` → `k/16`); `qa_lines = QA_LINES(kind, k)` (= destination
stride); `dst_line + t*qa_lines ≤ SPM lines`; `src_line + (t-1)*src_stride + k/16 ≤ SPM lines`. Numerics
and layout: §8.1.

### 3.4 tsr_mvm_cmd_t (dispatcher → MVM unit, M3), 294 bits

`{fmt, acc, dst_line, x_line, w_off, n, k[16], t[4], dst_stride, y_lines, rows_per_ch, row_beats,
qa_lines[16]}`: all checks of EX §8.1 (the MVM row) done, `dst_stride` resolved, `R = ceil(N/NUM_CH)`,
`ROW_BEATS(fmt, K)` and `QA_LINES` precomputed by the checker (it needs them for the range checks;
power-of-two `NUM_CH` computes `R` and `R × ROW_BEATS` with shifts and one DSP, CC §8.1). The unit's
ports (`mvm_cmd_*`, `mvm_done`) exist on `tsr_core` and are idle in the X0 netlist.

### 3.5 tsr_csr ↔ tsr_cmdproc (run control)

**History: v1 (deleted in X0).** `run_start` / `run_abort` / `run_soft_reset` from the CSR block and
`run_finish` / `run_err_*` / `cur_pc` / `perf_retire` back to it were the run control between `tsr_csr` and
`tsr_cmdproc`. In v2 the CSR block hosts the launch queue and completion ring, the loader with VERIFY, the
SR sequencer that implements `SOFT_RESET`, and the `Q_*` queue state; its interface to the issue path is
the credit, the queue state and the completion counters (CC §6, §12.2; EX §5). `PERF_RETIRED` becomes the
sum of the `DONE_SEQ` advances (EX §6).

### 3.6 Perf events (→ tsr_csr)

Kept in the tree: `perf_mvm_busy` (MVM unit), `perf_ch_rd_beat[c]` (arbiter c: memory read beat
transferred) and `perf_ch_wait[c]` (arbiter c: at least 1 read outstanding and no read beat this cycle).
v1's `perf_retire` (cmdproc) and `PERF_CYCLES` counting cycles with `STATUS.BUSY` are gone; the tree's
`PERF_CYCLES` and `PERF_RETIRED` read 0 until the v2 sources exist (the ACTIVE gate and the queue
completions, EX §6). The VU counters read 0 (no VU in RTL).

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
- M1 addresses in beats; `len` is the beat count 1..64 (7 bits, 0 never issued); a burst stays inside one
  64-beat (4 KiB) block: `addr % 64 + len ≤ 64`; `addr + len ≤ DRAM limit`.
- M2 at most `RD_OUTSTANDING` read requests outstanding per channel (from the request transfer to the
  transfer of its `last` beat), summed over all sources, enforced by `tsr_mem_arb`.
- M3 ids: `id = {source[1:0], local[1:0]}`, source `0 AUX` (v1: FETCH; tied off since X0, V2-CC-26), `1 DMA`,
  `2 WS` (the weight stream), `3` reserved. The memory returns read data in order **within one id**; beats
  of different ids may interleave (at beat granularity). Write responses: one per write request, in order
  within one id.
- M4 full-beat writes only (no byte strobes). Write data beats follow write-request order, no interleaving
  between bursts; `wr_dat_last` exactly on each burst's last beat.
- M5 the core never waits for `wr_req_ready` before offering `wr_dat_valid`, nor for `wr_dat_ready` before
  offering `wr_req_valid` (AXI-style independence; the model runs modes where the memory accepts data only
  after the request, or the request only after data).
- M6 sources reserve space before requesting (R4): a source's `rd_rsp_ready` may drop only for reasons
  local to that source, so the shared response channel never deadlocks across sources. (v2 strengthens the
  SPM side of this argument: "bounded, memory-independent reasons", CC §10.3.)
- M7 v1: `STATUS.DONE` implied no transaction in flight (all reads returned, all write responses in). In v2
  the same follows from §3.0: when every issued entry is done (`QIDLE`) no transaction is in flight, and the
  runtime joins everything before it posts an answer (EX §4.7 R6).
- M8 every `m_*` output is driven by a flop.

`tsr_mem_arb` (one per channel) source side: read sources `s_rd_req_*[TSR_NSRC]`, per-source
`s_rd_rsp_valid`/`s_rd_rsp_ready`, shared `s_rd_rsp_data/last/id` (local id); write source = DMA
(`s_wr_*`). Arbitration round-robin per burst; every channel has DMA and WS, AUX is tied off in `tsr_core`
(v1: channel 0 also had FETCH). The hardware trace writer of DB (a second write source on channel 0) is
designed (CC §13.2, DB §8.3).

---

## 5. SPM

- Geometry: `SPM_BYTES/64` lines × 512 bits, line-addressed, `SPM_AW = clog2(lines)`. Contents undefined at
  start (programs never read SPM they did not write); no reset of the array.
- Built from `tsr_sram_1r1w` (rtl/tech): `DEPTH, WIDTH, RD_LAT ≥ 1`, byte enables, read-first on
  same-address collision. U200: URAM288 cascades (512 b = 8 × 72-bit columns; the SPM banks are `ram_style
  ultra`, 128 URAM at 4 MiB, §12 item 1); generic: behavioural array.
- Client ports (`TSR_SPM_CL_*`): 0 DMA, 1 QACT, 2 MVM. Per client: `rd_req_valid/ready/line` →
  `rd_rsp_valid[c]` exactly `RD_LAT` (= `SPM_RD_LAT`) cycles after the accepting edge with the line on the
  shared `rd_rsp_data`, **no backpressure** (R4); `wr_valid/ready/line/data/be[64]`.
- Single read and single write port; fixed priority by client index, with a registered grant. In v1 only
  the dispatched unit drove requests, so there was no contention; `ready` may be low and must be honoured.
  Ordering: a read accepted after a write was accepted returns the new data (even across internal
  pipelining); same-cycle same-line read returns old data.
- Port plan: **M2** clients 0/1 time-share the 1R1W port. **M3** the MVM uses client 2: activation
  broadcast (read QA lines once per MVM) and result gather (write y lines; ACC reads old y lines first).
  **v2 (designed, CC §10)**: two line-interleaved banks (even / odd lines, each 1R1W: 2 read and 2 write
  ports, 128 URAM unchanged), per-bank round-robin arbitration with a registered one-hot grant, six clients
  (0 DMA, 1 QACT, 2 MVM, 3 RISC-V bridge, 4 VU, 5 AT, absent in M6 and tied off), a single-port adapter
  `tsr_spm_cport` for the clients that keep the one-request-per-cycle interface, and the line-buffered
  RISC-V alias. The client interface stays as above.

---

## 6. CSR block

CSR port protocol: master holds `csr_valid/addr/we/wdata` until `csr_ready`; the core performs the access
one or more cycles after it first sees `csr_valid` (tsr_csr: capture, S1 write side effects / read select,
S2 read mux: 4 cycles per transfer including the handshake edge) and raises `csr_ready` (registered) for
one cycle with `csr_rdata` (reads; 0 for writes); the transfer ends at the edge with `csr_valid &&
csr_ready`. A write's effect is visible to every later access and to the core when the transfer ends.
Unmapped or unaligned offsets read 0, writes ignored. The protocol is unchanged in v2.

**What the tree keeps (X0).** `ID` = `0x54535231`, `VERSION` = `0x0200` (ISA 2.0), `CFG0` = `NUM_CH[7:0] |
LANES[19:8] | T_MAX[23:20]`, `CFG1` = `SPM_BYTES/64`, `CFG2` = `VL[7:0] | NSREG[15:8] | (K_MAX/256)[31:16]`,
`CAPS` from a parameter (0 by default), and the 64-bit `PERF_*` counters at `0x100 + 8i` (LO) / `+4` (HI),
`i < 5 + 2*NUM_CH`, with the tear-free read rule (a LO read snapshots HI; the next HI read of that counter
returns the snapshot). No register is writable; `CTRL` and `STATUS` read 0. `MVM_CYCLES`, `CH_RD_BEATS[c]`
and `CH_WAIT[c]` count the events of §3.6; `CYCLES`, `RETIRED` and the VU slots read 0.

**v2 (designed).** The host CSR map is EX §5, the only one: the loader (`LOAD_*` with receive CRC and
VERIFY), the launch queue and completion ring, the log ring, `CTRL`/`STATUS` (`RV_RUN`, hold, abort and
clear pulses, `SOFT_RESET`), `Q_*`, `PERF` v2 (EX §6), the DBG block (DB §10.1) and the DMI window; CC §6
maps the hardware onto it, CC §6.6 gives the read requesters, and `tsr_csr` is rewritten in W4.

**History: v1 (deleted in X0).** The v1 `CTRL` write handled SOFT_RESET, then START, then ABORT; START
latched `CMD_ADDR`, `CMD_COUNT` and `ARG0..7` into the run; `STATUS` was `BUSY | DONE | ERR` with W1C for
`DONE`/`ERR`; `ERR_CODE`, `ERR_PC`, `ERR_INFO` and `PC` described the run; SOFT_RESET while BUSY acted as
ABORT and applied when the run ended (proposal P3 of §11). The v1 CFG fields and PERF counter layout are
the ones above. See `git show v1-m4:docs/rtl-m2.md` §6.

---

## 7. Architectural checks (all in tsr_cmdproc, before any state change)

**History: v1 (deleted in X0); the checks continue in `tsr_chk` (planned).** `tsr_cmdproc` checked every
instruction before dispatch, in the order `ILLEGAL_OP → BAD_FMT → K_RANGE → T_RANGE → SPM_RANGE →
DRAM_RANGE` (operands in field order A1, A2, A3), after the fetch-side checks (ABORT, past `CMD_COUNT`,
fetch range). v2's checker takes the same pipeline from the issue FIFO head and adds `BAD_DEP` after
`ILLEGAL_OP` and `OPERAND` after `T_RANGE`, with up to four tagged operand extents per op (EX §8.1, CC §8.2);
A1..A3 are the immediates `w1..w3`, since there is no register file. The per-instruction legality table
(reserved fields, sub codes, operand extents for NOP/FENCE, LD/ST, QACT and MVM) is EX §7-§8, and the v1
table is `git show v1-m4:docs/rtl-m2.md` §7. Implementation notes that carry over: all 32-bit fields (K, T,
stride, n_lines) are checked in full width (`T = 0x80000001` is `T_RANGE`; `(T-1)·stride` with `stride ≥
2^31` overflows 32 bits and must still fail); QACT's `T > T_MAX` check comes before any multiply, so
`(T-1)·stride` needs at most 3 × 32 bits; `QA_LINES`, `ROW_BEATS` and the limits are `tsr_pkg` functions;
`QA_MISMATCH` is never raised.

---

## 8. Numerics

### 8.1 QACT (tsr_qact, bit-exact `tsr_qact()` in sw/sim/tsr_ref.c)

Per token t: source `x[0..K-1]` = f32 at SPM `src_line + t·src_stride` (16 per line); destination image at
`dst_line + t·qa_lines`: QS (`ceil(K/64)` lines of int8 q), D (`ceil(4·nblk/64)` lines, f32 per block), BS
(A256 only, `ceil(2·K/16/64)` lines, int16 per 16 q). Every line of the image is written whole; bytes after
the last valid element of each region are 0.
- **A32** (block 32): `amax` = largest `|DAZ(x)|` over non-NaN x (as unsigned magnitude bits; 0 if none);
  `d = fdiv(amax, 127.0)`; `id = (d == ±0) ? +0 : fdiv(1.0, d)`; `q = q8_round_away(fmul(x, id))`; `D =
  fp16_to_f32(f32_to_fp16(d))`.
- **A256** (block 256): `max` = `DAZ(x)` of the **first** non-NaN element with the strictly largest
  `|DAZ(x)|`; if `amax == 0`: `d = +0`, all q = 0, BS = 0; else `iscale = fdiv(−127.0, max)`, `q =
  q8_round_even(fmul(iscale, x))`, `d = fdiv(1.0, iscale)`; `BS[j] = Σ q[16j..16j+15]`.
- fp ops are the spec rules (DAZ inputs, RNE, FTZ of subnormal *rounded* results, canonical NaN
  `0x7FC00000`); `q8_round_*`: NaN → 0, round, saturate to [−128, 127] (A32 half away from zero, A256 half
  to even). Special-value consequences (A32 `amax = Inf` → `D = Inf`, q = 0; A256 `max = ±Inf` → `d = ∓Inf`,
  q = 0; tiny `max` → `iscale = ∓Inf`, saturated q, x = 0 → NaN → 0) follow from these rules: no special
  cases in hardware.
- Throughput target M2: at least 1 source line per about 2 cycles outside the per-block divides (an iterative
  divider per block is acceptable in M2; overlap blocks if cheap).

### 8.2 fp library (tsr_fp_*)

- `tsr_fp_pkg` (functions, combinational, one cycle each): `tsr_f32_daz`, `tsr_f32_is_nan`,
  `tsr_f32_amag`, `tsr_fp16_to_f32`, `tsr_f32_to_fp16`, `tsr_f32_to_q8(x, half_even)`, `tsr_i32_to_f32`.
- `tsr_fp_mul #(LAT)`: fixed latency, `ce` stall, `in_valid → out_valid`.
- `tsr_fp_div`: iterative, `in_valid/in_ready`, `out_valid/out_ready`, correctly rounded; recip =
  `div(1.0, x)` (bit-identical to `tsr_frecip`).
- `tsr_fp_add` (term accumulation, ACC; M3) with the same rules.
- Mandatory vectors (`docs/spec.md` §21, "FTZ decision point"): `0x3F7FFFFF × 0x00800000 = 0x00800000`,
  `0x3F1FA0BA × 0x00CD4708 = 0x00800000`, `0x00FFFFFF / 2 = 0x00800000`, `0x3F7FFFFE × 0x00800000 = 0`;
  plus random and exhaustive (fp16, q8 over all exponents) sweeps against `common.tsr_ref`.

---

## 9. Verification infrastructure

- `rtl/lint.sh` is the lint gate (§1). `tb/run_all.sh [--no-lint] [pytest args]` runs lint plus every test
  under `tb/`; `tb/run_all.sh -k dma` runs the tests matching a name and `tb/run_all.sh mvm --tsr-seed 7`
  one directory with another seed (options: `--tsr-seed N`, `--waves`, `-k pattern`). Build directories:
  `$TSR_SIM_BUILD_ROOT` (default `build/cocotb`), `make -j$TSR_BUILD_JOBS` (default 4).
- **What survives X0** (header of `tb/run_all.sh`): the block tests `tb/common` (the infrastructure
  self-test), `tb/dma` (DMA and arbiter), `tb/fp`, `tb/qact`, `tb/spm`, `tb/mvmfront`, `tb/mvmback`,
  `tb/mvm`; `tb/tech` (the Xilinx technology wrappers against the generic ones); the C++ memory model
  self-test of `tb/fastreplay`; and `tb/platform` (adapter lint, the `hw/u200/build_xclbin.sh` platform
  checks) with `tb/platform/bw` (the M0 bandwidth kernel and `tsr_bw` on the mock / XRT no-op shim). Slow
  tests are opt-in (`TSR_FP_SLOW=1`, `TSR_PLATFORM_SLOW=1`, or `-m slow`).
- **What X0 deleted**: the benches of the v1 command path (`tb/cmdproc`; `tb/core` `test_core.py`,
  `test_core_m3.py` and `mutate_core.py`; the `cmd.bin` replay of `tb/fastreplay`; the v1 parts of
  `tb/common`: `csr_driver`, `proggen`, `pycore`, `replay`, `simlib`, `trace`; the kernel bench of
  `tb/platform` and `tb/platform/vkernel`). Their v2 replacements are planned: `tb/core` v2, `tb/fastreplay`
  v2 (tile-only replay of `push.bin` traces into the issue FIFO), `tb/platform` v2, `tb/platform/vkernel`
  with two clocks, and the new `tb/ctrl`, `tb/vu`, `tb/at`, `tb/rv`, `tb/dbg` (W4/W5; CC §15, SW §10.3).
  Until then tag `v1-m4` and `build/v1` are the comparison path.
- Per block: `tb/<block>/test_<block>.py` (unique basename, no `__init__.py`) holding the cocotb tests and
  a pytest function `run("<toplevel>", __name__, parameters=..., extra_sources=[...])`. Test-only wrappers
  go in `tb/<block>/hdl/`. Run reduced configs where useful (small SPM, 1-3 channels, T_MAX 8).
- `tb/common`:
  - `isa.py` and `tables.py`: constants/encoders/struct layouts mirroring `tsr_pkg.sv` and the generated
    tables (`DMA_CMD_T.pack(...)` etc. to drive struct ports; widths/field order are checked against the
    RTL by the self-test).
  - `tsr_ref.py`: `ref()` → ctypes `libtsr_ref.so` (`build/dev/sw/sim/`): `fadd/fsub/fmul/fdiv/frecip/
    fsqrt/daz/i32_to_f32`, `fp16_to_f32`, `f32_to_fp16`, `q8_round_away/even`, `qact(kind, x_bits, K) → QA
    image`, `qa_split`, `qa_*lines`, `row_beats`, `dot_row`, `tree16`. All f32 values are 32-bit patterns
    (NaN payloads preserved).
  - `mem_model.py`: `MemModel(dut, num_ch=, addr_w=, rd_outstanding=, limit_beats=, seed=, ...)`; random
    latency/backpressure/stalls, per-id ordering, cross-id interleave, write ordering modes, protocol
    assertions (§4), `written/read/read_unwritten` beat sets, `check(idle=True)`.
  - `mvm_model.py`: `mvm_reference`, the MVM reference used by `tb/mvm` (tied to the simulator's op-level
    API by `test_mvm_model_vs_simulator`).
  - `runner.py`: `run(toplevel, module, ...)`; `tbutil.py`: clock/reset/seed. `test_common.py`: the
    infrastructure self-test.
  - Timing discipline for TB components: drive right after a rising edge, sample at the falling edge. (With
    Verilator, values read inside a RisingEdge callback can already include that edge's register updates;
    falling-edge sampling avoids the ambiguity.)
- `traces/m2/` (regenerate: `traces/m2/regen.sh build/dev`; generator `tsr-trace-gen --ops m2`): the M2
  regression sets as trace v2 `ops` traces (SW §5.6; `traces/README.md`), with v1's seeds and configurations
  (`tmax1` 60, `tmax8` 16, `small1ch` 16, `ch3` 8, `errors` 40, `errors8` 32 programs; the v1 data was 172
  `cmd.bin` traces, `git show v1-m4:traces/m2/regen.sh`). They replay today on the simulator (tile-only
  and through the `replay` kernel, race detector fatal). RTL replay returns with the tile-only replay bench
  (W5). The trace data is not in git; the recipes and `traces/MANIFEST.sha256` are.

---

## 10. Room for M3 (MVM clusters)

**History: v1.** M2 reserved, as tied-off ports, the pieces the MVM unit needed: the `TSR_SRC_WS` memory
source per channel (one cluster weight streamer per channel, id `{2, local}`, sharing `RD_OUTSTANDING`),
the SPM client `TSR_SPM_CL_MVM` (QA broadcast reads, y gather writes, ACC reads), `tsr_mvm_cmd_t` and the
`mvm_cmd_*` / `mvm_done` ports, the `perf_mvm_busy` input of `tsr_csr`, and `tsr_fp_add` / `tsr_i32_to_f32`
in the fp library; clusters live per channel (SLR placement) between the arbiter's WS port and a
broadcast/gather bus to the SPM client, with pipeline depth across SLRs as a parameter. All of it was filled
in M3 and is described in [`rtl-m3.md`](rtl-m3.md).

---

## 11. Spec proposals

**History: v1.** The M2 agents' proposals for the spec, all adopted into `docs/spec.md` (§10, §21) or
superseded by EX: P1 the CSR port handshake (§6), P2 memory port details (§4), P3 SOFT_RESET while BUSY as
ABORT (v1 only), P4 DONE implies a quiescent memory port (§4 M7), P5 PERF semantics in hardware, P6 CAPS for
MVM-less bitstreams, P7 reset (`rst_n` synchronous, SPM undefined after a hardware reset), P8 `PERF_FETCH_BEATS`
(moot: fetch is deleted), P9 CSR reads at any time without side effects (a PERF LO read moves the snapshot),
P10 hardware reset. One proposal still describes the tree and is kept:

- P10 **Reset timing (`RST_PIPE`, M3 part C).** `tsr_core` parameter `RST_PIPE` = P (0..8; an
  implementation parameter, [`rtl-m3.md`](rtl-m3.md) §14.4 item 3) puts P register stages on the reset:
  every flop of the core sees `rst_n` exactly P cycles late, at assertion and at release. At the core
  boundary:
  - the outputs (memory requests and write data, `csr_ready`) go quiet P cycles after `rst_n` falls: from
    the (P + 1)-th rising edge with `rst_n` low on. In the first P cycles of a reset the core may still offer
    requests and write data; the memory side, in reset with the core, must drop them;
  - `rst_n` must stay low for **at least P + 1 cycles** (rising edges). Any pulse reaches every flop (P
    cycles late), but the memory side is reset by the same `rst_n` without the delay: with a shorter pulse it
    leaves reset while the core still offers pre-reset requests, accepts them, and their responses arrive
    after the core's reset;
  - the core leaves reset P cycles after `rst_n` rises (the CSR port's handshake absorbs the delay: the
    master holds `csr_valid` until `csr_ready`);
  - the registers of the reset pipe start in reset (configuration INIT / simulation initial 0), so after
    configuration the core is held in reset until `rst_n` has been high for P cycles.
  P = 0 (the default, used by the generic test builds) keeps the outputs quiet from the first rising edge
  with `rst_n` low; one cycle of reset suffices. U200 bitstreams use P = 3
  ([`rtl-m3.md`](rtl-m3.md) §14.2); the kernel-level rule for `ap_rst_n` (the adapter's reset chain in front
  of the core resets the AXI bridges with the core, `tsr_xrt_reset.sv`) is in
  [`platform-u200.md`](platform-u200.md) §5, and XRT holds `ap_rst_n` far longer. The v1 tests (`core_hw_reset`,
  `m3_hw_reset`) are deleted with the core benches; the contract stands for the v2 benches.

---

## 12. Core integration (M2 status)

**History: v1.** `tsr_core.sv` kept the §2 netlist and added simulation-only integration checks
(`ifndef SYNTHESIS`, armed by the first reset edge): I1 one instruction in flight (at most one unit offered a
command, `done` only from the unit that holds it), I2 a unit drives SPM requests only while it holds an
instruction, I3 memory requests only during a run and a quiescent port at `done`. The tree keeps the checks
per unit (X0 dropped I3 and the one-in-flight rule): a unit's `done` only with an op in flight, SPM and
weight-stream traffic only while the unit holds an op, the MVM's SPM read accounting and, per channel, its
weight-stream accounting (requests, bursts and beats against rows × `ROW_BEATS`). The v1 core tests (`tb/core/test_core.py`: trace replay of every `traces/m2` configuration under four memory
regimes, implementation-parameter variants, directed tests and random programs against the ISA simulator in
lockstep, a mutation check of the core wiring; 84 passed in the full run of `tb/run_all.sh` at the time) were
deleted by X0 and are reproducible at tag `v1-m4`.

**Throughput** (measured in RTL simulation, v1 core, U200 parameters, ideal memory: always ready, latency
0..1): LD / ST of 4096 lines 1.01 cycles per beat (one channel at a time, about 38 cycles of overhead per
instruction); QACT A256 K 32768 1.11 cycles per source line, A32 8.03 (divider bound); in the v1 core NOP /
LI / ADDI cost 4.25 cycles each and a 1-line LD 19 cycles, a 1-line ST 21 cycles.

**Timing at 300 MHz (U200)** (measured in Vivado 2024.2, out of context, v1; whole-core out-of-context place
and route with `tb/core/viv/core_ooc.tcl`, xcu200-fsgd2104-2-e, 3.333 ns, I/O budget 1.6 ns, no pblocks, so
the SLR crossings of the real floorplan are not modelled; post-route estimates only). The M2 core:

| Build | SRAM mapping | SPM_RD_LAT | WNS / WHS (ns) | Worst paths |
|---|---|---|---|---|
| U200, SPM 256 KiB | generic -> 8 URAM | 2 | +0.121 / +0.015 | QACT multiplier operands; SPM URAM -> DMA queue +0.45 |
| U200 v1 (4 MiB) | generic -> 1032 RAMB36 | 2 | -2.150 / +0.010 | SPM BRAM read -> DMA queue / QACT; SPM write enables |
| U200 v1 (4 MiB) | generic -> 1032 RAMB36 | 4 | -2.189 / +0.028 | SPM write address / enable fan-out into 1032 BRAMs |
| U200 v1 (4 MiB) | URAM (`ram_style ultra`, scratch) | 2 | -0.209 / +0.038 | URAM cascade read -> DMA queue, QACT response reg |
| U200 v1 (4 MiB) | URAM (`ram_style ultra`, scratch) | 4 | **+0.120 / +0.010** | QACT multiplier operands; SPM write address -> URAM cascade +0.134 |

Utilisation (U200 v1, URAM, SPM_RD_LAT 4): 16.4 K LUT, 22.7 K FF, 32 DSP, 136 URAM (128 SPM + 8 QACT line
buffer), 1525 LUTRAM, no latches, no critical warnings. The k4 and k1 results with the MVM unit and the
floorplan are in [`rtl-m3.md`](rtl-m3.md) §14.

Consequences for M4 (structures expected to be critical, in order; all still describe retained blocks):
1. SPM: the generic behavioural `tsr_sram_1r1w` is mapped to block RAM by Vivado (1032 RAMB36 for 4 MiB) and
   fails by more than 2 ns. `rtl/tech/xilinx/tsr_sram_1r1w.sv` must force URAM and the U200 build must use
   `SPM_RD_LAT 4`. Both are drop-in: every M2 trace and random test passed with SPM_RD_LAT 1, 2, 3 and 4;
   with URAM and 4 the core meets 300 MHz out of context. **Done (M3 part C):** the Xilinx technology wrapper
   (selected by `rtl/filelist_xilinx.f`; `tb/tech` checks it cycle by cycle against the generic one) picks the
   RAM style by size: the SPM banks are URAM (`ram_style ultra`, 4 banks × 32 = 128 URAM at 4 MiB, the read
   stages absorbed into the URAM output / cascade registers), the QACT line buffer, activation buffers and
   weight FIFOs block RAM, tiny test sizes LUTRAM. The U200 bitstream uses `SPM_RD_LAT 4`
   ([`rtl-m3.md`](rtl-m3.md) §14.2); whole-core Vivado runs with it: [`rtl-m3.md`](rtl-m3.md) §14.3 (128 URAM,
   QACT 8 RAMB36; k1 / k4 T_MAX 1 / k4 T_MAX 8 meet 300 MHz out of context).
2. DMA <-> channel arbiters across SLRs: one DMA engine serves all channels while each `tsr_mem_arb` sits next
   to its DDR4 controller (three SLRs on U200). LD data goes arbiter response flop -> SLL crossing -> 4 × 512-bit
   channel select -> DMA queue write in one cycle; ST data is one shared 512-bit payload flop fanned out to
   every arbiter. Every DMA <-> arbiter stream is valid/ready and latency-insensitive (the DMA tolerates any
   read latency), so per-channel register slices can be added without logic changes. This was done in the
   linked k4 design (`AXI_SLICES`, [`platform-u200.md`](platform-u200.md) §8.1).
3. SPM write side: 512-bit data, 64 byte enables and the address from the SPM input register to 32 URAMs per
   bank (+0.134 ns at 4 MiB): keep the replicated input registers.
4. QACT: the merged multiplier scale register drives all 32 DSPs (+0.12 ns): `-keep_equivalent_registers` or a
   per-lane copy helps.
5. Synchronous reset fan-out into three SLRs: the platform layer delivers a pipelined, per-SLR replicated
   `rst_n`. **Done (M3 part C):** the adapter's `tsr_xrt_reset` (one copy per bridge, a chain for the core) and
   the core's `RST_PIPE` stages with per-block kept copies ([`rtl-m3.md`](rtl-m3.md) §14.4 item 3; boundary
   contract in §11 P10).
6. cmdproc / fetch / CSR / PERF counters: comfortable (at least +0.5 ns in the passing whole-core runs); v1
   modules, deleted. The v2 additions to the core clock domain (CC §14.5) are built flop, at most 3 LUT levels,
   flop, with registered boundaries.

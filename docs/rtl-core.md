# Tesseract core RTL: conventions, interfaces and contract

This document is the RTL contract of the core infrastructure: the conventions every block follows, the internal interfaces between
the dispatch side and the units, the memory ports, the SPM, the CSR block, the numerics units (QACT and the fp library) and the
verification infrastructure. The MVM unit has its own document, [rtl-mvm.md](rtl-mvm.md). Authority: `docs/spec.md` > simulator
(`sw/sim`) > `docs/design` > this file > `rtl/core/tsr_pkg.sv`. Section numbers are cited by RTL and testbench comments.

## 1. Conventions (all blocks)

- Language and lint. The RTL is SystemVerilog-2017. `rtl/lint.sh` (Verilator 5 `--lint-only -Wall`, every module as its own top
  plus reduced and edge `tsr_core` configurations, the Xilinx technology wrappers, the XRT platform and the bandwidth kernel)
  stays clean. There are no blanket waivers. The waivers in `rtl/core` are `UNUSEDPARAM` around the constant catalogues of the
  packages (`tsr_pkg`, `tsr_tables_pkg`). Unused inputs are marked with `wire unused_ok = &{1'b0, a, b};`. Vendor primitives and
  attributes appear only in `rtl/tech/<target>/`.
- Clock and reset. The core has one clock `clk`. `rst_n` is synchronous, active low and already synchronised by the platform
  layer. Every control flop (valid bits, FSM state, counters, pointers) is reset, and wide datapath registers need no reset.
  Verilator builds randomise initial state (`--x-initial unique`, `+verilator+rand+reset+2`), so missing resets show up in
  simulation. A reset value must not be loaded from a flop that has no reset: that flop is 0 after configuration on the FPGA and
  random in simulation, so the bug shows only on the card ([platform-u200.md](platform-u200.md) §8.8). `run(rand_reset=0)` or
  `TSR_RAND_RESET=0` starts everything at zero as the FPGA does. The control core runs on a second clock `clk_rv`
  ([ctrl-riscv.md](ctrl-riscv.md) §7). Every crossing is a CDC structure of `rtl/ctrl/cdc/`, and Ibex and riscv-dbg keep their own
  reset style (CC §3.1).
- valid and ready (all streams, internal and external):
  - R1 A transfer happens at a rising edge where `valid && ready`.
  - R2 `valid` must not depend combinationally on `ready`. Once raised, `valid` and the payload stay unchanged until the transfer.
    The memory model asserts this on the core boundary.
  - R3 `ready` may depend on `valid`, but registered `ready` is preferred across module boundaries.
  - R4 A requester of a stream without backpressure (SPM read data, fixed-latency fp pipelines) or of a stream it must never stall
    (memory read data, see §4 M6) reserves space for the response before it issues the request.
  - R5 There is no combinational path from a module's inputs to its outputs across the boundaries listed in §3 to §5 unless
    stated.
  - R6 One-cycle pulse signals (`done`, ...) are registered.
- Timing. Every `m_*` core-boundary output comes from a flop (`tsr_mem_arb`). Memory ports cross SLRs, so the arbiter and DMA may
  add pipeline stages freely, with the depth as a parameter. Logic depth stays at about 6 LUT levels per stage. Wide compares and
  adds (range checks) are pipelined over cycles in the checker (`tsr_chk`), where the instruction rate is not critical.
- Data byte order. Byte j of a 64-byte beat or line is bits `[8j+7:8j]` (little-endian). F32 element i of a line is bits
  `[32i+31:32i]`. Instruction word wi is `insn[i]` (`tsr_insn_t = logic [7:0][31:0]`).
- Widths. ISA-level quantities (addresses, lengths, strides) are 32-bit fields in the dispatch structs regardless of parameters.
  The checker's range checks guarantee that the bits a unit does not use are zero, and units slice what they need (for example
  `cmd.spm_line[SPM_AW-1:0]`).
- Parameters. `tsr_core` takes the spec table (`NUM_CH BEAT_BITS LANES T_MAX K_MAX SPM_BYTES VL NSREG RD_OUTSTANDING CH_BYTES
  ADDR_BEAT_W`, defaults in `tsr_pkg::TSR_DEF_*`: 4 channels of 16 GiB, a 4 MiB SPM, `K_MAX` 32768, `VL` 16, `NSREG` 16 and
  `T_MAX` 1) and implementation parameters. These are `HAS_MVM` (default 1), `SPM_RD_LAT` (default 2, the U200 images use 4, §12),
  the `MVM_*` latencies and pipe depths ([rtl-mvm.md](rtl-mvm.md) §1) and `RST_PIPE` (§11). `CH_BYTES` is `longint unsigned`.
  Elaboration checks (`$fatal`) mirror `tsr_sim_create`: `BEAT_BITS == 512`, `NUM_CH` 1 to 16, `T_MAX` 1 to 8, `K_MAX` a multiple
  of 256 in [256, 65280], `SPM_BYTES` a positive multiple of 64, `NSREG` 1 to 16, `VL` 1 to 255, `LANES` 1 to 4095, `CH_BYTES` a
  positive multiple of 64, `ADDR_BEAT_W` 1 to 40 and `RD_OUTSTANDING` at least 1. With `HAS_MVM` 1, `LANES` must be 128,
  `MVM_WS_FIFO` a power of two of at least 64, and the MVM latencies in range.
- Files. `rtl/filelist.f` (the generic target, used by lint and the cocotb runner, packages first) and `rtl/filelist_xilinx.f`
  (which swaps `tech/generic` for `tech/xilinx`) are shared files, and a new module is added to both. `rtl/core/tsr_tables_pkg.sv`
  and the SDK and simulator tables are generated from one source (`sw/tables/tsr_tables.py`), and `gen_tables.py --check` runs in
  lint.

## 2. Block diagram and ownership

<p align="center"><img src="fig/core.svg" alt="Block diagram of tsr_core in two clock domains. The clk_rv domain (100 MHz) holds the control core (Ibex) with its instruction and data memories, the debug module and the control window. The core clock domain (200 MHz) holds the CSR block (host port, launch queue, completion ring, log ring, PERF), the issue FIFO crossing, the checker tsr_chk, the dispatcher with its unit queues feeding the DMA (LD, ST, KVST), QACT, MVM clusters (with ATTQK and ATTPV) and vector unit, the scratchpad tsr_spm with the RISC-V bridge, and one memory arbiter per DDR4 channel." width="100%"></p>

The figure shows `tsr_core`. The netlist is `tsr_core.sv` with the data path in `tsr_core_dp.sv`.

| Block | Files | Contract |
|---|---|---|
| (a) control | `tsr_csr.sv`, `tsr_chk.sv`, `tsr_issue.sv` | §3.5, §6, §7, CC §6 to §9 |
| (b) memory | `tsr_dma.sv`, `tsr_mem_arb.sv` (with `tsr_rslice`), `tsr_rpipe.sv`, `tsr_vpipe.sv`, `tb/dma/` | §3.2, §4 |
| (c) fp library | `tsr_fp_pkg.sv`, `tsr_fp_mul.sv`, `tsr_fp_div.sv`, `tsr_fp_add.sv`, `tb/fp/` | §8.2 |
| (d) SPM | `tsr_spm.sv`, `tsr_spm_cport.sv`, `tsr_spm_bridge.sv`, `rtl/tech/{generic,xilinx}/tsr_sram_1r1w.sv`, `tb/spm/` | §5 |
| (e) QACT | `tsr_qact.sv`, `tb/qact/` | §3.3, §8.1 |
| MVM unit | `tsr_mvm*.sv`, `rtl/tech/*/tsr_mac_lane.sv`, `tb/mvm*/` | [rtl-mvm.md](rtl-mvm.md) |
| (f) top | `tsr_core.sv`, `tsr_core_dp.sv` | §1, §2, §9, §12 |
| infra | `tsr_pkg.sv`, `tsr_tables_pkg.sv`, filelists, `rtl/lint.sh`, `tb/common/`, `tb/conftest.py`, `tb/pytest.ini`, `tb/run_all.sh` | all |

The contract comments in the sources carry the final port lists. A port-list change is an interface change: update this file,
`tsr_pkg.sv`, `tsr_core.sv` and `tb/common/isa.py` together.

## 3. Internal interfaces

### 3.0 Dispatch protocol (dispatcher to unit), common to DMA, QACT and MVM

The sender is the queue head of each unit in the issue path (CC §9.1: `cmd_q` and `cmd_v_q` drive the unit's `cmd_valid` and
`cmd`).

| Signal | Dir (unit) | Meaning |
|---|---|---|
| `cmd_valid` | in | the dispatch side offers one checked command (R2: held until `cmd_ready`) |
| `cmd_ready` | out | the unit is idle and able to start (registered) |
| `cmd` | in | `tsr_<unit>_cmd_t`, valid with `cmd_valid`. The unit latches it at the transfer |
| `done` | out | one-cycle pulse: all architectural effects are complete and visible |

- The units run concurrently under in-order queues. A unit holds at most the ops it accepted and completes them in accept order
  (CC §9.4 invariant I3: at most 2 outstanding for the MVM, 3 for the VU, 1 for every other unit).
- `done` is never in the cycle of the accepting edge. "Visible" means that every SPM line write of the instruction has been
  accepted by `tsr_spm` (§5 ordering guarantees later reads see it) and, for ST, every DRAM write response has been received.
- Units never report errors. Every architectural check is done before dispatch (§7), so operand values a unit receives are always
  legal. Units may `assert` legality in simulation.
- There is no flush or cancel. A dispatched command always runs to completion. QABORT drops queue entries that have not started
  (EX §8, CC §9.5), and never cancels a running op.

### 3.1 Instruction fetch

The core does not fetch instructions over a memory port. The control core fetches from its ITCM (CC §4).

### 3.2 `tsr_dma_cmd_t` (dispatcher to `tsr_dma`), 101 bits

`{is_st[1], ch[4], spm_line[32], dram_beat[32], n_lines[32]}`. LD (`is_st` 0) copies DRAM beats `dram_beat .. +n_lines-1` of
channel `ch` to SPM lines `spm_line .. +n_lines-1`. ST does the reverse. The checker guarantees `ch < NUM_CH`, `n_lines >= 1` (it
retires `n_lines == 0` itself, with no traffic), `spm_line + n_lines <= SPM lines` and `dram_beat + n_lines <= DRAM limit`. The
DMA rules are:

- D1 Bursts have `len = min(left, 64 - addr % 64)`: never more than 64 beats, never across a 4 KiB boundary.
- D2 One read id (local 0) per LD, and data arrive in order. The DMA reserves SPM-write and FIFO room (R4), so `rd_rsp_ready`
  depends only on local state. The LD reserves queue space for a whole burst before it requests (queue of 512 beats,
  `rd_rsp_ready` constant 1, CC §10.4, CC-13).
- D3 ST reads the SPM (no backpressure, R4) into a write-data FIFO. `wr_req` and `wr_dat` are offered independently (§4 M5), and
  `wr_dat_last` marks each burst's last beat.
- D4 `done` comes after the last SPM write is accepted (LD) or the last write response (ST).

The DMA uses only the port of channel `ch`. KVST, a store of quantised KV rows (AT), joins LD and ST in the same engine.

### 3.3 `tsr_qact_cmd_t` (dispatcher to `tsr_qact`), 133 bits

`{kind[1], dst_line[32], src_line[32], k[16], t[4], src_stride[32], qa_lines[16]}`. The checker guarantees `kind` 0 = A32 and 1 =
A256, `0 < k <= K_MAX`, `k % (kind ? 256 : 32) == 0`, `1 <= t <= T_MAX`, `src_stride` already resolved (`w5 == 0` gives `k/16`),
`qa_lines = QA_LINES(kind, k)` (the destination stride), `dst_line + t*qa_lines <= SPM lines` and `src_line + (t-1)*src_stride +
k/16 <= SPM lines`. Numerics and layout are in §8.1.

### 3.4 `tsr_mvm_cmd_t` (dispatcher to the MVM unit), 360 bits

`{local_scale, local_softcap, scale[32], post_scale[32], fmt, acc, dst_line, x_line, w_off, n, k[16], t[4], dst_stride, y_lines,
rows_per_ch, row_beats, qa_lines[16]}`. The first four fields carry the SCALE and softcap epilogue and are used only when
`LOCAL_SCALE` or `LOCAL_SOFTCAP` is built. All checks of EX
§8.1 (the MVM row) are done, `dst_stride` is resolved, and `R = ceil(N/NUM_CH)`, `ROW_BEATS(fmt, K)` and `QA_LINES` are
precomputed by the checker, which needs them for the range checks. Power-of-two `NUM_CH` computes `R` and `R * ROW_BEATS` with
shifts and one DSP (CC §8.1).

### 3.5 `tsr_csr` and the issue path (run control)

The CSR block hosts the launch queue and completion ring, the loader with VERIFY, the SR sequencer that implements `SOFT_RESET`,
and the `Q_*` queue state. Its interface to the issue path is the credit, the queue state and the completion counters (CC §6,
§12.2, EX §5). `PERF_RETIRED` is the sum of the `DONE_SEQ` advances (EX §6).

### 3.6 Perf events (to `tsr_csr`)

`perf_mvm_busy` comes from the MVM unit, `perf_ch_rd_beat[c]` from arbiter c (a memory read beat transferred) and
`perf_ch_wait[c]` from arbiter c (at least one read outstanding and no read beat this cycle). The other counters are EX §6 and the
unit blocks that count them.

## 4. Memory ports

The core boundary has one port per channel, as packed `[NUM_CH]` vectors (channel c is index c).

| Channel | Signals (prefix `m_`) | Dir |
|---|---|---|
| read request | `rd_req_valid`, `rd_req_ready`, `rd_req_addr[ADDR_BEAT_W]`, `rd_req_len[7]`, `rd_req_id[4]` | core to mem |
| read response | `rd_rsp_valid`, `rd_rsp_ready`, `rd_rsp_data[512]`, `rd_rsp_id[4]`, `rd_rsp_last` | mem to core |
| write request | `wr_req_valid`, `wr_req_ready`, `wr_req_addr`, `wr_req_len[7]`, `wr_req_id[4]` | core to mem |
| write data | `wr_dat_valid`, `wr_dat_ready`, `wr_dat_data[512]`, `wr_dat_mask[16]`, `wr_dat_last` | core to mem |
| write response | `wr_rsp_valid`, `wr_rsp_ready`, `wr_rsp_id[4]` | mem to core |

The rules are asserted by `tb/common/mem_model.py`:

- M1 Addresses are in beats. `len` is the beat count 1 to 64 (7 bits, 0 is never issued). A burst stays inside one 64-beat (4 KiB)
  block: `addr % 64 + len <= 64` and `addr + len <= DRAM limit`.
- M2 At most `RD_OUTSTANDING` read requests are outstanding per channel, from the request transfer to the transfer of its `last`
  beat, summed over all sources. `tsr_mem_arb` enforces it.
- M3 Ids are `{source[1:0], local[1:0]}` with source 0 AUX (tied off, CC-26), 1 DMA, 2 WS (the weight stream) and 3 reserved. The
  memory returns read data in order within one id. Beats of different ids may interleave at beat granularity. There is one write
  response per write request, in order within one id.
- M4 A write beat carries a 16-bit word mask, one bit per 32-bit word. A full beat has all bits set. Write data beats follow
  write-request order with no interleaving between bursts, and `wr_dat_last` marks exactly each burst's last beat.
- M5 The core never waits for `wr_req_ready` before offering `wr_dat_valid`, nor for `wr_dat_ready` before offering `wr_req_valid`
  (AXI-style independence). The model runs modes where the memory accepts data only after the request, or the request only after
  data.
- M6 Sources reserve space before requesting (R4). A source's `rd_rsp_ready` may drop only for reasons local to that source, so
  the shared response channel never deadlocks across sources. On the SPM side the reasons are bounded and independent of memory
  (CC §10.3).
- M7 When every issued entry is done (`QIDLE`), no transaction is in flight: all reads have returned and all write responses are
  in (§3.0). The runtime joins everything before it posts an answer (EX §4.7 R6).
- M8 Every `m_*` output is driven by a flop.

`tsr_mem_arb`, one per channel, has read sources `s_rd_req_*[TSR_NSRC]`, per-source `s_rd_rsp_valid` and `s_rd_rsp_ready`, and
shared `s_rd_rsp_data`, `last` and `id` (local id). The write sources are the DMA and channel 0's trace writer (CC §13.2, DB
§8.3). Arbitration is round robin per burst. Every channel has DMA and WS, and AUX is tied off in `tsr_core`.

## 5. SPM

- Geometry. `SPM_BYTES/64` lines of 512 bits, line-addressed, `SPM_AW = clog2(lines)`. Contents are undefined at start, programs
  never read SPM they did not write, and the array has no reset.
- Storage is `tsr_sram_1r1w` (`rtl/tech`) with `DEPTH`, `WIDTH`, `RD_LAT >= 1`, byte enables and read-first on same-address
  collision. On the U200 it is URAM288 cascades (512 bits as 8 columns of 72 bits, `ram_style ultra`, 128 URAM at 4 MiB, §12). The
  generic target is a behavioural array.
- `tsr_spm` has two line-interleaved banks, even and odd lines, each 1R1W, so 2 read and 2 write ports with 128 URAM unchanged. It
  has per-bank round-robin arbitration with a registered one-hot grant (CC §10). The client ports are `TSR_SPM_CL_*`: 0 DMA, 1
  QACT, 2 MVM, 3 RISC-V bridge, 4 VU (both bank ports) and 5 AT (reserved, tied off). Clients 0 to 3 keep a single
  request per cycle through the single-port adapter `tsr_spm_cport`.
- Per client: `rd_req_valid/ready/line`, with `rd_rsp_valid[c]` exactly `RD_LAT` (= `SPM_RD_LAT`) cycles after the accepting edge
  and the line on the shared `rd_rsp_data`, with no backpressure (R4), and `wr_valid/ready/line/data/be[64]`. `ready` may be low
  and must be honoured.
- Ordering. A read accepted after a write was accepted returns the new data, also across internal pipelining. A same-cycle
  same-line read returns old data.

## 6. CSR block

The CSR port protocol: the master holds `csr_valid`, `addr`, `we` and `wdata` until `csr_ready`. The core performs the access one
or more cycles after it first sees `csr_valid` and raises `csr_ready` (registered) for one cycle with `csr_rdata` (reads, 0 for
writes). In `tsr_csr` a transfer takes 4 cycles including the handshake edge: capture, decode with write side effects and read
select, respond. LOAD_DATA, DMI and the DBG block take extra bounded states. A write's effect is visible to every later access and
to the core when the transfer ends. Unmapped or unaligned offsets read 0 and ignore writes.

The host CSR map is EX §5, the only one: `ID` (0x54535231) at 0x000, `VERSION` (major in bits [15:8], minor in [7:0]), `CFG0`
(`NUM_CH[7:0] | LANES[19:8] | T_MAX[23:20]`), `CFG1` (SPM lines), `CFG2` (`VL[7:0] | NSREG[15:8] | (K_MAX/256)[31:16]`), `CAPS`,
`CTRL` and `STATUS`, the loader (`LOAD_*` with receive CRC and VERIFY), the launch queue and completion ring, the log ring, `Q_*`,
the 64-bit `PERF_*` counters at 0x100 + 8i (LO) and +4 (HI) with the tear-free read rule (a LO read snapshots HI, and the next HI
read of that counter returns the snapshot), the DBG block (DB §10.1) and the DMI window. CC §6 maps the hardware onto it, and CC
§6.6 gives the read requesters.

## 7. Architectural checks (before any state change)

The checker `tsr_chk` takes ops from the head of the issue FIFO and checks every op before dispatch, with up to four tagged
operand extents per op (EX §8.1, CC §8.2). The check order and the per-op legality table (reserved fields, sub codes, operand
extents for NOP and FENCE, LD and ST, QACT and MVM) are EX §7 and §8, and the operands are the immediates `w1` to `w3`. All 32-bit
fields (K, T, stride, n_lines) are checked in full width: `T = 0x80000001` is `T_RANGE`, and `(T-1)*stride` with `stride >= 2^31`
overflows 32 bits and must still fail. QACT's `T > T_MAX` check comes before any multiply, so `(T-1)*stride` needs at most 3 x 32
bits. `QA_LINES`, `ROW_BEATS` and the limits are `tsr_pkg` functions. `QA_MISMATCH` is never raised.

## 8. Numerics

### 8.1 QACT (`tsr_qact`, bit-exact `tsr_qact()` in `sw/sim/tsr_ref.c`)

Per token t, the source `x[0..K-1]` is f32 at SPM `src_line + t*src_stride` (16 per line). The destination image at `dst_line +
t*qa_lines` has QS (`ceil(K/64)` lines of int8 q), D (`ceil(4*nblk/64)` lines, f32 per block) and, for A256 only, BS
(`ceil(2*K/16/64)` lines, int16 per 16 q). Every line of the image is written whole, and bytes after the last valid element of
each region are 0.

- A32 (block 32). `amax` is the largest `|DAZ(x)|` over non-NaN x, as unsigned magnitude bits (0 if none). `d = fdiv(amax,
  127.0)`, `id = (d == +-0) ? +0 : fdiv(1.0, d)`, `q = q8_round_away(fmul(x, id))` and `D = fp16_to_f32(f32_to_fp16(d))`.
- A256 (block 256). `max` is `DAZ(x)` of the first non-NaN element with the strictly largest `|DAZ(x)|`. If `amax == 0`, `d = +0`,
  all q are 0 and BS is 0. Otherwise `iscale = fdiv(-127.0, max)`, `q = q8_round_even(fmul(iscale, x))`, `d = fdiv(1.0, iscale)`
  and `BS[j] = sum q[16j..16j+15]`.
- The fp ops follow the spec rules: DAZ inputs, RNE, FTZ of subnormal rounded results and canonical NaN `0x7FC00000`. `q8_round_*`
  maps NaN to 0, rounds and saturates to [-128, 127] (A32 half away from zero, A256 half to even). The special-value consequences
  (for example A32 `amax = Inf` gives `D = Inf` and q = 0) follow from these rules, with no special cases in hardware.
- The throughput target is at least one source line per about 2 cycles outside the per-block divides.

### 8.2 fp library (`tsr_fp_*`)

`tsr_fp_pkg` has combinational functions of one cycle each: `tsr_f32_daz`, `tsr_f32_is_nan`, `tsr_f32_amag`, `tsr_fp16_to_f32`,
`tsr_f32_to_fp16`, `tsr_f32_to_q8(x, half_even)` and `tsr_i32_to_f32`. `tsr_fp_mul #(LAT)` has fixed latency, a `ce` stall and
`in_valid` to `out_valid`. `tsr_fp_div` is iterative with `in_valid/in_ready` and `out_valid/out_ready`, and correctly rounded
(`recip` is `div(1.0, x)`, bit-identical to `tsr_frecip`). `tsr_fp_add` ([rtl-mvm.md](rtl-mvm.md) §8.1) follows the same rules.
The mandatory vectors (`docs/spec.md` §21, "FTZ decision point") are `0x3F7FFFFF * 0x00800000 = 0x00800000`, `0x3F1FA0BA *
0x00CD4708 = 0x00800000`, `0x00FFFFFF / 2 = 0x00800000` and `0x3F7FFFFE * 0x00800000 = 0`, plus random and exhaustive (fp16, q8
over all exponents) sweeps against `common.tsr_ref`.

## 9. Verification infrastructure

- `rtl/lint.sh` is the lint gate (§1). `tb/run_all.sh [--no-lint] [pytest args]` runs lint plus every test under `tb/`.
  `tb/run_all.sh -k dma` runs the tests matching a name, and `tb/run_all.sh mvm --tsr-seed 7` runs one directory with another seed
  (options `--tsr-seed N`, `--waves`, `-k pattern`). Build directories are `$TSR_SIM_BUILD_ROOT` (default `build/cocotb`), built
  with `make -j$TSR_BUILD_JOBS`. Slow tests are opt-in (`TSR_FP_SLOW=1`, `TSR_PLATFORM_SLOW=1`, or `-m slow`).
- Per block, `tb/<block>/test_<block>.py` (unique basename, no `__init__.py`) holds the cocotb tests and a pytest function
  `run("<toplevel>", __name__, parameters=..., extra_sources=[...])`. Test-only wrappers go in `tb/<block>/hdl/`.
- `tb/common` holds `isa.py` and `tables.py` (constants, encoders and struct layouts that mirror `tsr_pkg.sv` and the generated
  tables, checked against the RTL by the self-test), `tsr_ref.py` (the `libtsr_ref.so` wrapper from `build/dev/sw/sim/`: the fp
  operations, the conversions, `qact`, `row_beats`, `dot_row`, with f32 values as bit patterns so NaN payloads are preserved),
  `mem_model.py` (`MemModel`: random latency, backpressure and stalls, per-id ordering, cross-id interleave, write ordering modes,
  the protocol assertions of §4 and `check(idle=True)`), `mvm_model.py`, `runner.py`, `tbutil.py` and `test_common.py`.
- Testbench components drive right after a rising edge and sample at the falling edge. With Verilator, values read inside a
  RisingEdge callback can already include that edge's register updates, and falling-edge sampling avoids the ambiguity.
- The trace sets under `traces/` are regenerated by their `regen.sh` scripts (`tsr-trace-gen --ops <set>`,
  [traces/README.md](../traces/README.md)). They replay on the simulator and on the core. The trace data is not in git, but the
  recipes and `traces/MANIFEST.sha256` are.

## 10. MVM attachment points

`tsr_core` provides these for the MVM unit: the `TSR_SRC_WS` memory source per channel (one cluster weight streamer per channel,
id `{2, local}`, sharing `RD_OUTSTANDING`), the SPM client `TSR_SPM_CL_MVM`, `tsr_mvm_cmd_t` with the `mvm_cmd_*` and `mvm_done`
ports, the `perf_mvm_busy` input of `tsr_csr`, and `tsr_fp_add` and `tsr_i32_to_f32` in the fp library. Clusters live per channel
(SLR placement) between the arbiter's WS port and a broadcast and gather bus to the SPM client, with the pipeline depth across
SLRs as a parameter ([rtl-mvm.md](rtl-mvm.md)).

## 11. Reset timing (`RST_PIPE`)

The `tsr_core` parameter `RST_PIPE` = P (0 to 8, an implementation parameter, [rtl-mvm.md](rtl-mvm.md) §14.4) puts P register
stages on the reset. Every flop of the core sees `rst_n` exactly P cycles late, at assertion and at release. At the core boundary:

- The outputs (memory requests and write data, `csr_ready`) go quiet P cycles after `rst_n` falls, from the (P + 1)-th rising edge
  with `rst_n` low on. In the first P cycles of a reset the core may still offer requests and write data, and the memory side, in
  reset with the core, must drop them.
- `rst_n` must stay low for at least P + 1 cycles (rising edges). Any pulse reaches every flop, P cycles late, but the memory side
  is reset by the same `rst_n` without the delay. With a shorter pulse it leaves reset while the core still offers pre-reset
  requests, accepts them, and their responses arrive after the core's reset.
- The core leaves reset P cycles after `rst_n` rises. The CSR port's handshake absorbs the delay, because the master holds
  `csr_valid` until `csr_ready`.
- The registers of the reset pipe start in reset (configuration INIT, initial 0 in simulation), so after configuration the core is
  held in reset until `rst_n` has been high for P cycles.

P = 0, the default and the value of the generic test builds, keeps the outputs quiet from the first rising edge with `rst_n` low,
and one cycle of reset suffices. The U200 images use P = 3 ([rtl-mvm.md](rtl-mvm.md) §14.2). The kernel-level rule for `ap_rst_n`,
where the adapter's reset chain in front of the core resets the AXI bridges with the core (`tsr_xrt_reset.sv`), is
[platform-u200.md](platform-u200.md) §5.

## 12. Core integration

`tsr_core.sv` has simulation-only integration checks (`ifndef SYNTHESIS`, armed by the first reset edge): a unit's `done` only
with an op in flight, SPM and weight-stream traffic only while the unit holds an op, the MVM's SPM read accounting and, per
channel, its weight-stream accounting ([rtl-mvm.md](rtl-mvm.md) §13.1). LD and ST move one beat per cycle, one channel at a time.
QACT A256 takes about one cycle per source line, and A32 is bound by the per-block divide.

Critical structures at the U200 core clock, in order:

1. SPM. The generic behavioural `tsr_sram_1r1w` is mapped to block RAM by Vivado, which is far too slow for 4 MiB. The Xilinx
   technology wrapper (selected by `rtl/filelist_xilinx.f`, checked cycle by cycle against the generic one by `tb/tech`) picks the
   RAM style by size. The SPM banks are URAM (`ram_style ultra`, 4 banks of 32 URAM, the read stages absorbed into the URAM output
   and cascade registers), the QACT line buffer, activation buffers and weight FIFOs are block RAM, and tiny test sizes are
   LUTRAM. The U200 images use `SPM_RD_LAT` 4 ([rtl-mvm.md](rtl-mvm.md) §14.2), and every trace and random test passes with
   `SPM_RD_LAT` 1 to 4.
2. DMA and the channel arbiters across SLRs. One DMA engine serves all channels while each `tsr_mem_arb` sits next to its DDR4
   controller. LD data goes from the arbiter response flop over the SLL crossing and a 4 x 512-bit channel select into the DMA
   queue write. ST data is one shared 512-bit payload flop fanned out to every arbiter. Every DMA and arbiter stream is valid and
   ready and latency-insensitive, so per-channel register slices (`AXI_SLICES`, [platform-u200.md](platform-u200.md) §4) are added
   without logic changes.
3. SPM write side. The 512-bit data, 64 byte enables and the address from the SPM input register go to 32 URAMs per bank. The
   replicated input registers stay ([rtl-mvm.md](rtl-mvm.md) §14.9).
4. QACT. The merged multiplier scale register drives all 32 DSPs, and `-keep_equivalent_registers` or a per-lane copy helps.
5. Synchronous reset fan-out into three SLRs. The adapter's `tsr_xrt_reset` (one copy per bridge, a chain for the core) and the
   core's `RST_PIPE` stages with per-block kept copies ([rtl-mvm.md](rtl-mvm.md) §14.4 item 3, boundary contract in §11)
   distribute it.
6. CSR and PERF counters have comfortable timing. The control-core logic in the core clock domain (CC §14.5) is built as flop, at
   most 3 LUT levels, flop, with registered boundaries.


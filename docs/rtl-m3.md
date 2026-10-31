# Tesseract MVM unit: RTL architecture, numerics and results (M3)

This document is the RTL record of the MVM unit and of the implementation work that took it to the U200: the
architecture of one cluster per DDR4 channel, its interfaces, the bit-exact numerics, throughput, the
verification of the unit, and the Vivado results (§14). The control core does not change the behaviour of
the weight MVM.

> **Scope.** Record of the RTL of the MVM unit (M3 MVM unit, integration, Vivado results of part C). The tree
> holds the MVM unit and its block benches; the dispatcher and the core-level benches are planned (W4, W5).
> §13 are results of RTL simulation and §14 are Vivado 2024.2 measurements. The card measurements and the
> results of §13 and §14 were taken before the control core was added, with the host issuing the ops, and stay
> true for the blocks in the tree. Normative for the issue path and the attention modes:
> [AT](design/03-attention-kv.md) §5-§6, §8 and [CC](design/04-control-core.md) §8-§10, §14. Authority:
> `docs/spec.md` > simulator (`sw/sim`) > `docs/design` > this file > [`rtl-m2.md`](rtl-m2.md) > `tsr_pkg.sv`.

Status words: *in the tree*, *designed*, *planned*. In the tree: `rtl/core/tsr_mvm*.sv`,
`tsr_fp_add.sv`, `tsr_vpipe.sv`, `tsr_rpipe.sv`, the technology wrappers in `rtl/tech/{generic,xilinx}` and the
block benches `tb/mvmfront`, `tb/mvmback`, `tb/mvm`, `tb/fp`, `tb/tech`. The dispatcher is the issue path
(CC §8-§9, planned). The simulator's `ex_mvm` (`tsr_sim.c`) and `tsr_ref.c` are the numeric oracle.

**Results at a glance.** The MVM unit runs on the card: the k1 xclbin (1 channel, T_MAX 1, 300 MHz)
passed the M3 gate on the Alveo U200, and the k4 xclbin (4 channels, T_MAX 8, floorplan C', kernel clock
276.3 MHz after `v++` scaling) ran Qwen2.5-0.5B Q8_0 at 54.7 tok/s and Qwen2.5-32B Q4_K_M at 2.93 tok/s
(measured on the card, k4; [`platform-u200.md`](platform-u200.md) §14, [`performance.md`](performance.md)).
The weight stream is busy 97-98 % of the cycles in decode. The k4 link ran (§14.8).

**Control core and attention modes (designed, W4).** The MVM unit is dispatched from CC's queue heads
([CC](design/04-control-core.md) §8-§9; the checks are in `tsr_chk`), reads the SPM through the
even / odd banks (CC §10), and is checked by tile-only replays of `push.bin` traces (§13). The weight MVM itself
(§4-§7, the numerics and `RB_MIN`) does not change. The cluster gains two attention modes: `ATTQK` (0x21: scores and
row maxima over KV groups, the Q8_0 path with segment counters, per-slot activation read addresses and a gather
line buffer per pass row) and `ATTPV` (0x22: P·V with block-floating-point weights accumulated in the MAC DSPs,
a PV mode of `tsr_mac_lane`, a 64-beat delay FIFO, a capture bank and convert / add pipes). Their contract is AT
§5.2-§5.3 and §6; their structure and cost are AT §8.1, §8.2 and §8.4 (about 19-27 K LUT per cluster,
*estimate*); §6.5 below points to them.

**M3 scope.** The MVM unit (spec §8, "MVM cluster microarchitecture", stages 1-10: one cluster per
channel, the activation broadcast, the result gather with ACC) integrated into `tsr_core` with `HAS_MVM = 1`,
executing every legal `MVM` bit-exactly like the simulator (Q8_0 with A32 QA, Q4_K with A256 QA, any legal K / N
/ T, ACC, `dst_stride`), reading exactly `rows_c × ROW_BEATS` weight beats per
channel, and sustaining one weight beat per cycle per cluster at T = 1 when the memory delivers (rows of at
least `RB_MIN` beats, §7.5). DSP packing and the URAM / BRAM mapping (`rtl/tech/xilinx`) were done in M3 part C
(§7.2, §9.3, §14). Not in M3: overlapping consecutive MVMs (spec §8, "Overlapping consecutive MVMs", M5; the MVM
queue head may accept the next MVM before the previous one is done, CC §9.3 and invariant I3). The U200 platform adapter, kernel packaging and
the XRT transport were also M3 (the M3 gate was "Qwen2.5-0.5B Q8_0 runs on the card", spec §19 milestone table)
and have their own documents.

**M2 is unchanged.** Every rule of [`rtl-m2.md`](rtl-m2.md) applies as written: conventions R1-R6 and the timing
budget (§1), the dispatch protocol (§3.0), `tsr_mvm_cmd_t` (§3.4), the memory-port rules M1-M8 and `tsr_mem_arb`
(§4), the SPM client contract (§5). `HAS_MVM = 0` builds stay bit-identical to M2. M3 filled the tied-off M3 ports
of `tsr_core` (rtl-m2 §10).

---

## 1. Conventions

- Everything in `docs/rtl-m2.md` §1 (SystemVerilog-2017, `rtl/lint.sh` clean with targeted
  waivers only, synchronous active-low `rst_n`, reset every control flop, valid/ready rules
  R1–R6, ~6 LUT levels per 300 MHz stage, little-endian byte order, 32-bit ISA fields).
- fp arithmetic only through `tsr_fp_mul` (`MUL_LAT`), `tsr_fp_add` (`ADD_LAT`, §8) and the
  `tsr_fp_pkg` conversions `tsr_fp16_to_f32`, `tsr_i32_to_f32`. No other fp operation exists in
  the MVM unit, and every one of them is a separately rounded spec operation (no FMA).
- Integer sums are exact (two's complement, widths of §7.6), so their order is free.
- **Parameters.** From `tsr_core`: `NUM_CH T_MAX K_MAX SPM_BYTES SPM_RD_LAT ADDR_BEAT_W BEAT_BITS`.
  `LANES` must be 128 (= `BEAT_BITS / 4`, one Q4_K beat; `tsr_core` checks it when
  `HAS_MVM = 1`, CFG0 still reports it). Implementation parameters (results are identical for
  every legal value; tests vary them; defaults in `tsr_pkg`):

  | Parameter | Default | Range | Meaning |
  |---|---|---|---|
  | `WS_FIFO_BEATS` | 512 | power of 2, ≥ 64 | weight FIFO per cluster (§4) |
  | `BC_PIPE` | 1 | ≥ 0 | broadcast pipe stages tsr_mvm → cluster (`tsr_vpipe`), §3.1; since part C also the weight-stream response pipe arbiter → cluster (§14.4) |
  | `RES_PIPE` | 1 | ≥ 0 | result pipe stages cluster → gather (`tsr_rpipe`), §3.5; since part C also the weight-stream request pipe and the gather's job / start pipes (`JOB_PIPE`, §14.4) |
  | `ACT_RD_LAT` | 2 | ≥ 1 | activation buffer read latency (§5) |
  | `MAC_LAT` | 3 | ≥ 1 | `tsr_mac_lane` latency (§7.2) |
  | `MUL_LAT` | 4 | ≥ 1 | `tsr_fp_mul` latency (its 300 MHz default) |
  | `ADD_LAT` | 5 | 1..8 | `tsr_fp_add` latency; `TSR_DEF_FADD_LAT` = its measured 300 MHz default (§8.1) |
  | `WS_RD_LAT` | 2 | 1..4 | weight FIFO RAM read latency (`tsr_mvm_ws.RD_LAT`, `tsr_mvm_front.WS_RD_LAT`, `TSR_DEF_WS_RD_LAT`) |

  `tsr_core` exposes `MVM_WS_FIFO`, `MVM_BC_PIPE`, `MVM_RES_PIPE` and the latencies
  `MVM_ACT_RD_LAT`, `MVM_MAC_LAT`, `MVM_MUL_LAT`, `MVM_ADD_LAT`, `MVM_WS_RD_LAT` (additive,
  defaults `TSR_DEF_*`; the tests vary them, §13.1).
- **Files** (`rtl/filelist.f` order: packages, tech, fp, SPM, arbiter, pipes, DMA, QACT, MVM
  bottom-up, control, core): `tech/generic/tsr_mac_lane.sv`, `core/tsr_fp_add.sv`,
  `core/tsr_vpipe.sv`, `core/tsr_rpipe.sv`, `core/tsr_mvm_ws.sv`, `core/tsr_mvm_front.sv`,
  `core/tsr_mvm_actbuf.sv`, `core/tsr_mvm_back.sv`, `core/tsr_mvm_cluster.sv`,
  `core/tsr_mvm_gather.sv`, `core/tsr_mvm.sv`. A new file is a filelist change (ask the lead).
- **Lint**: `rtl/lint.sh` lints every new module as its own top with edge parameters (NUM_CH 1 / 3
  / 16, T_MAX 1 / 8, K_MAX 256 / 65280, latencies 1 / 8, pipes 0 / 3, FIFO 64 / 1024). The
  integration step adds `tsr_core -GHAS_MVM=1` configurations (U200, 1 / 3 / 16 channels, T_MAX
  8, small K_MAX, pipe and FIFO extremes).
- The skeletons in `rtl/core` (and `tsr_mac_lane`) carry the final port lists and a behavioural
  contract comment; bodies are tied off and lint clean. `tsr_vpipe` / `tsr_rpipe` are complete.
  Port-list changes are interface changes: agree them with the lead (update this file,
  `tsr_pkg.sv`, the skeletons and `tb/common/isa.py` together).

---

## 2. Block diagram and ownership

<p align="center"><img src="fig/cluster.svg" alt="tsr_mvm, shared by all clusters, reads the QA lines from the SPM read port and broadcasts the setup of the MVM tile op, which starts the cluster's weight requests, then the lines through each cluster's own tsr_vpipe into the cluster's activation buffer. DDR4 channel c streams weight beats into the front end (stream requester with a 512-beat FIFO, chunk parser, meta + unpack); tagged beats enter the back end once the activation buffer is complete: 4 MAC lanes, integer scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials per row, tree16. Row results return through the cluster's tsr_rpipe to the shared gather, which writes 16 rows per SPM line, or with ACC reads the old line and adds first. Planned, drawn dashed: the attention modes ATTQK and ATTPV, with the p broadcast and gather ATT path in tsr_mvm and the w-unit, capture bank and convert pipes in the cluster." width="100%"></p>

The figure draws the datapath of one cluster as in the tree (solid) and the planned attention modes
(dashed, §6.5); the text diagram below is the same structure in port detail.

```
   dispatcher* ──tsr_mvm_cmd_t──►┌──────────────────────────── tsr_mvm ─────────────────────────────┐
     (§3.0)    ◄──── done ───────┤ accept · N=0 · ccmd · broadcast reader          tsr_mvm_gather   │
                                 │      │ SPM rd (client 2)                 assemblers · ACC RMW    │
   tsr_spm client 2 ◄────────────┤──────┘ (broadcast first, then ACC)      · tails · SPM wr ───────┼──► tsr_spm
                                 │  bc: {ccmd | bcline}                         ▲ res: {row,last,y}  │
                                 └───┬──────────────┬───────────────────────────┼──────────────┬───┘
                        tsr_vpipe    │BC_PIPE       │  ...            tsr_rpipe │RES_PIPE      │
                        (per cluster)▼              ▼                           │              │
             ┌──────────────── tsr_mvm_cluster #(CH = c) ───────────────────────┴──┐           │
             │ setup: first = c·R, rows_c, beats_c ──fcmd──┐                        │   (c = 1..C-1:
             │ bcline ──► tsr_mvm_actbuf (QS 2×T·512, DBS T·512) ◄─ qs/dbs reads ─┐ │    same)
             │                                             ▼                      │ │
             │   tsr_mvm_front: tsr_mvm_ws ─FIFO─► parser · meta · unpack ─beat─► tsr_mvm_back
             │                  (stage 1)          (stages 2-4)   tsr_mvm_beat_t  (stages 5-9:
             │                     │  ▲                                           MAC tsr_mac_lane,
             └─────────────────────┼──┼───────────────────────────────────────── int, fp term,
                                   ▼  │ WS source (id {2,0})                     acc, tree) ──► res
                         tsr_mem_arb channel c ──► m_* port c (DDR4 channel c, same SLR)
```

| Block | Agent | Files (owner edits only these) | Contract | Tests |
|---|---|---|---|---|
| (fp) | fp agent | `rtl/core/tsr_fp_add.sv`, `tb/fp/` (extend) | §8 | `tb/fp` |
| (front) | front agent | `rtl/core/tsr_mvm_ws.sv`, `tsr_mvm_front.sv`, `tb/mvmfront/` | §3.2, §3.3, §4 | `tb/mvmfront/test_mvm_front.py` |
| (back) | back agent | `rtl/core/tsr_mvm_back.sv`, `tsr_mvm_actbuf.sv`, `rtl/tech/generic/tsr_mac_lane.sv`, `tb/mvmback/` | §3.3–§3.5, §5, §7 | `tb/mvmback/test_mvm_back.py` |
| (top) | top agent | `rtl/core/tsr_mvm.sv`, `tsr_mvm_cluster.sv`, `tsr_mvm_gather.sv`, `tb/mvm/` | §3.0, §3.1, §3.2, §3.5, §6 | `tb/mvm/test_mvm.py` |
| (int) | integration | `rtl/core/tsr_core.sv` (HAS_MVM = 1 wiring, checks) | §10.4 | core-level replays (planned, W5) |
| infra | architecture | `tsr_pkg.sv`, `tsr_vpipe.sv`, `tsr_rpipe.sv`, `rtl/filelist.f`, `rtl/lint.sh`, `tb/common/`, `tb/conftest.py`, `tb/pytest.ini`, `tb/run_all.sh`, this file | all | `tb/common/test_common.py` |

\* The dispatcher is the queue head of the MVM unit (`tsr_issue`, planned, CC §9.1). In the tree the
unit's command port (`mvm_cmd_*`, `mvm_done`) exists and is idle.

Block tests that need several RTL blocks (e.g. the back end with the real activation buffer, or a
cluster with a stub back end) put wrappers in `tb/<block>/hdl/` (linted by `rtl/lint.sh`).

---

## 3. Internal interfaces

### 3.0 Dispatch (dispatcher → tsr_mvm)

The `tsr_mvm_cmd_t` fields come from `tsr_chk` through CC's queue head, and the guarantees below are
`tsr_chk`'s (CC §8.1). They state what the MVM unit may assume.

`tsr_mvm_cmd_t` (294 bits) and the dispatch protocol are those of `docs/rtl-m2.md` §3.0/§3.4.
`tsr_chk` guarantees:
`fmt` ∈ {0, 1}; `0 < k ≤ K_MAX`, `k % bs == 0` (bs 32 / 256); `1 ≤ t ≤ T_MAX`;
`y_lines = ceil(n/16)`; `dst_stride` resolved (`w6 = 0` → `y_lines`; 0 when n = 0);
`rows_per_ch = ceil(n/NUM_CH)`; `row_beats = ROW_BEATS(fmt, k)`; `qa_lines = QA_LINES(fmt, k)`;
`dst_line + (t−1)·dst_stride + y_lines ≤ SPM lines`; `x_line + t·qa_lines ≤ SPM lines`;
`w_off + rows_per_ch·row_beats ≤ DRAM limit` (so every weight beat of every channel is in range and
`ADDR_BEAT_W` bits hold every address). **Not** guaranteed: absence of overlaps (dst vs x, token
dst ranges with `dst_stride < y_lines` and t > 1) — undefined results (spec §21, "Overlapping operands"),
but the unit must still finish normally (no hang, DONE ⇒ quiescent).

- `cmd_ready`: register, 1 only while the unit is idle. `done`: registered 1-cycle pulse, never in
  the accepting cycle, when every y line write of the instruction has been accepted by `tsr_spm`
  **and** every weight beat requested on every channel has been received (no WS read outstanding,
  no WS request pending): rtl-m2 §3.0 "visible" plus §4 M7 (DONE ⇒ quiescent).
- **N = 0**: no SPM access, no memory traffic; `done` ≥ 1 cycle after the accept (tsr_sim.c
  `ex_mvm` reads/writes nothing: `rows = 0` for every channel, `y_lines = 0`).
- **N ≥ 1** (§6.1): one `tsr_mvm_ccmd_t` to every cluster, the broadcast of the T QA images,
  per-cluster weight streams of exactly `rows_c × ROW_BEATS` beats, row results gathered into
  the SPM. Clusters with `rows_c = 0` (c·R ≥ N: e.g. any N < NUM_CH, or N = 5 on 4 channels:
  R = 2, channel 3 empty) do nothing.
- `perf_mvm_busy` (→ `tsr_csr`, `PERF_MVM_CYCLES`): registered, 1 in every cycle from the cycle
  after the accepting edge up to and including the cycle of `done` = cycles the unit holds an
  instruction. Informational (not bit-compared, spec §21, "Performance counters").

### 3.1 Broadcast pipe (tsr_mvm → every cluster)

Per cluster c one `tsr_vpipe #(.VW(2), .DEPTH(BC_PIPE))` carrying
`{bc_cmd_valid, bc_line_valid}` and `{bc_cmd, bc_line}`; no backpressure (the clusters accept
everything, R4 on the reader side). All clusters see the same items, each delayed by its pipe.

Per MVM with N ≥ 1, in this order: one `bc_cmd` (`tsr_mvm_ccmd_t`, 137 bits), then
`T × QA_LINES` `bc_line` items (`tsr_mvm_bcline_t`, 527 bits) with gaps allowed, the final one
with `last = 1`; `bc_cmd_valid` and `bc_line_valid` are never 1 in the same cycle. The next MVM's
`bc_cmd` is sent only after `done` of the previous MVM (so after every cluster's last result).

| `tsr_mvm_ccmd_t` field | Value |
|---|---|
| `fmt`, `k`, `n`, `rows_per_ch` (R), `w_off` | from `tsr_mvm_cmd_t` |
| `row_beats[15:0]` | `cmd.row_beats` (≤ 1084) |
| `d_lines[7:0]` | QA D-region lines `tsr_qa_d_lines(fmt, k)` = `ceil(4·(k/bs)/64)` (≤ 128) |

SPM line `x_line + t·qa_lines + l` (t < T, l < qa_lines; one read per cycle, in this order) becomes
`bc_line = {last: t = T−1 ∧ l = qa_lines−1, tok: t, dbs: l ≥ qs_lines, addr: dbs ? l − qs_lines :
l, data: the line}` with `qs_lines = ceil(k/64)`. QA layout: spec §6, "Quantized activation (QA) layout",
`tsr_ref.c` 418–447 (QS, then D, then BS for A256, each line-aligned).

### 3.2 Cluster setup and front command

On `bc_cmd` the cluster wrapper computes (multi-cycle arithmetic is fine; it overlaps the broadcast):
`first = CH × R` (64-bit), `rows_c = tsr_mvm_rows_c(n, R, first)` = `max(0, min(R, N − first))`
(spec §8, stage 1, the stream requester; `tsr_ref.c` 335–352), `beats_c = rows_c × row_beats` (≤ 2^40 − 1: the dst SPM range check, which precedes DRAM_RANGE, bounds
rows_c ≤ N ≤ 16 × (2^26 − 1) because SPM_BYTES < 2^32, and exactly 2^40 would need a power-of-two
ROW_BEATS ≤ 1024 with rows_c ≥ 2^30; the maximum 2^40 − 1 is Q8_0 K 61664/61696, ROW_BEATS 1025.
`TSR_MVM_BEATS_W` = 40 has no spare bit: widening SPM_BYTES beyond 32 bits needs 41). If
`rows_c ≥ 1` it sends one `tsr_mvm_fcmd_t {fmt, k, row_beats, w_off, rows: rows_c, beats:
beats_c}` (137 bits, valid/ready) to the front end and registers `cfg_fmt`, `cfg_d_lines` for the
back end; with `rows_c = 0` nothing (the cluster only absorbs the broadcast). Row r of the channel
(global row `first + r`) starts at beat `w_off + r × row_beats` (spec §4, "Symmetric allocation"), so the channel's
rows are one contiguous range of `beats_c` beats from `w_off`.

Activation gating: `act_ready` := 0 at `bc_cmd`, := 1 at the edge that writes the `bc_line` with
`last = 1` into the activation buffer. Front → back beats pass only while `act_ready = 1`
(`beat_valid_back = beat_valid & act_ready`, `beat_ready_front = beat_ready & act_ready`). The
front end starts requesting weights at once, so the DRAM latency hides behind the broadcast.

### 3.3 Tagged beat stream (front → back): `tsr_mvm_beat_t`, 1175 bits

One item per **QUANT** beat, in stream order (META beats stay in the front end's meta register;
spec stages 2–4). Stream order is the row walk of `tsr_dot_row` (`tsr_ref.c` 600–635): rows
r = 0..rows_c−1; per row chunks c = 0.. (Q8_0: 32 blocks = 1 meta + 16 quant beats; Q4_K: 4
superblocks = 1 + 8 beats; the row's last chunk may be partial: Q8_0 `rem` blocks → 1 +
ceil(rem/2) beats, Q4_K `rem` superblocks → 1 + 2·rem beats; spec §5, "Row size", `tsr_ref.c` 256–294).
Term index i inside the row = block b (Q8_0) or superblock s (Q4_K).

| Field | Q8_0, quant beat j of chunk c | Q4_K, quant beat j of chunk c |
|---|---|---|
| `blk[10:0]` | b = 32c + 2j (even) | s = 4c + j/2 |
| `two` | block b+1 present: 2j+1 < blocks in chunk | 0 |
| `half` | 0 | j % 2 (1: second beat of s; term of s completes) |
| `d_a` (f32) | `fp16_to_f32(meta[4j +: 2])` = d of b | `fp16_to_f32(meta[16(j/2) +: 2])` = d |
| `d_b` (f32) | `fp16_to_f32(meta[4j+2 +: 2])` = d of b+1; +0 if !two | `fp16_to_f32(meta[16(j/2)+2 +: 2])` = dmin |
| `sc[g]` (6 b) | 0 | `sc_(4·half+g)`: `get_scale_min_k4(4·half+g, meta[16(j/2)+4 +: 12])` |
| `m[k]` (6 b) | 0 | `m_k`, k = 0..7, same scales (both halves carry all 8) |
| `w[L]` (s8) | L < 64: beat byte L (L ≥ 32: 0 if !two); L ≥ 64: 0 | lane 32g + l: beat byte 32·(g/2) + l, low nibble for even g, high nibble for odd g (0..15) |
| `row_last` | last quant beat of the row | same |
| `last` | `row_last` of row rows_c−1 | same |

Consequence used by the back end: lane L multiplies activation K position `32b + L` (Q8_0) or
`256s + 128·half + L` (Q4_K), and lane group g (lanes 32g..32g+31) is block b+g (Q8_0, g < 2)
or sub-block j = 4·half + g (Q4_K). Derivation: Q8_0 block b of a chunk at quant beat b/2, bytes
32·(b%2) (`block_offsets`, 283–294); Q4_K `qs[32g' + l]` low nibble = sub-block 2g' at x
64g'+l, high nibble = sub-block 2g'+1 at x 64g'+32+l (`term_q4_K_ptr`, 548–556), beat 2s+h holds
qs bytes [64h, 64h+64). The 6-bit unpack is ggml `get_scale_min_k4` (`tsr_ref.c` 359–367):
`j < 4: sc = q[j] & 63, m = q[j+4] & 63; j ≥ 4: sc = (q[j+4] & 0xF) | ((q[j−4] >> 6) << 4),
m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)` on the 12 scale bytes. Meta entries / quant bytes of
absent blocks (partial chunks) are never used, whatever the DRAM holds there (the repacker writes
zeros, but random DRAM content is legal). Golden model: `tb/common/mvm_model.py` `front_beats()`.

### 3.4 Activation buffer ports (tsr_mvm_actbuf, §5)

Write: `wr_valid`, `wr_line` (`tsr_mvm_bcline_t`; `last` ignored), one line per cycle, no
backpressure. Read: `qs_re/qs_addr` → `qs_data[T_MAX][1023:0]` and `dbs_re/dbs_addr` →
`dbs_data[T_MAX][511:0]`, both exactly `ACT_RD_LAT` cycles later (fully pipelined). The back end
addresses (per beat): QS word `b/4` (Q8_0; lanes 0..63 = half `(b/2) % 2` of the word) or
`2s + half` (Q4_K, all 1024 bits); D line `(b or s) >> 4`, element `(b or s) & 15` (Q8_0 block
b+1 is element +1 of the same line, b even); BS line `cfg_d_lines + s/2`, int16 elements
`16·(s%2) .. +15`.

### 3.5 Row-result streams

- back → wrapper: `res_valid/res_ready`, `res_y[T_MAX][31:0]`: one item per row, in row order,
  `res_y[t]` = `tsr_dot_row(fmt, row, K, QA_t)` for every t < T_MAX (tokens ≥ T are don't-care).
- wrapper → gather (through `tsr_rpipe #(.W(33 + 32·T_MAX), .DEPTH(RES_PIPE))` per cluster):
  `{res_row[31:0] = first + r, res_last = (r == rows_c−1), res_y}`. R1/R2 on every hop.

### 3.6 SPM client 2 (TSR_SPM_CL_MVM)

Unchanged `tsr_spm` contract (rtl-m2 §5): reads have fixed latency `SPM_RD_LAT` and no
backpressure; writes with 64 byte enables. The MVM unit is the only active SPM client while it
holds an instruction. Read port users: the broadcast reader (first) and the gather's ACC reads
(only after the broadcast completed — a cluster cannot produce a result before its `act_ready`,
which follows the last broadcast line, whose read has returned). Write port: the gather only.

---

## 4. Weight-stream memory protocol (tsr_mvm_ws, spec stage 1)

- Source `TSR_SRC_WS` (2) of `tsr_mem_arb` channel c; the arbiter, its rslices, round robin and
  the shared `RD_OUTSTANDING` budget are unchanged (rtl-m2 §4, §10). Local id 0 for every burst
  (memory-side id `{2, 0}` = 8): one id, so the memory returns the data in request order (M3).
- Requests in address order from `w_off`: `len = min(64 − addr % 64, beats_left)` — ≤ 64 beats,
  never across a 64-beat (4 KiB) block (M1). The lens sum to exactly `beats_c = rows_c ×
  ROW_BEATS`: no over-fetch, no prefetch beyond the channel's rows (spec §21, "Performance counters":
  `PERF_CH_RD_BEATS[c]` MVM beats = `rows_c × ROW_BEATS`; spec §8, stage 1: "R × ROW_BEATS is an upper
  bound and an allocation size, not the amount read"). `rows_c = 0` → no request at all. Reference: `tsr_sim.c`
  `ex_mvm` (`cap_read(sim, c, woff, rows_c × row_beats)`).
- **Reserve before request** (R4 / M6): the weight FIFO has `WS_FIFO_BEATS` entries; a burst is
  loaded into the request register only when `WS_FIFO_BEATS − (beats buffered + beats requested
  and not received) ≥ len`, and that space is taken at the load. `rd_rsp_ready` is therefore
  constant 1 (the shared response channel never waits for the WS source).
- **FIFO sizing**: at one beat per cycle, Little's law needs `round trip + 64 ≤ WS_FIFO_BEATS`
  (round trip = request handshake at the WS port → first data beat at the WS port, including the
  arbiter's 3 cycles, the SLR crossing and the DDR4 controller; the + 64 is the burst granularity
  of the reservation). Default 512 sustains 1 beat/cycle up to a round trip of ~440 cycles
  (1.47 µs at 300 MHz) — about 4× the expected DDR4-2400 closed-page latency incl. interconnect
  (~100–350 ns). In-flight bursts ≤ 512/64 + 1 < `RD_OUTSTANDING` (16), so the outstanding limit
  is not the bottleneck. Cost: one `tsr_sram_1r1w` 512 × 512 bits (8 BRAM36) per cluster.
  Measured (tb/mvmfront, tsr_mem_arb + MemModel, defaults): fixed memory latency 400 → 1.000
  cycles/beat, 520 → 1.14; first tagged beat 16 cycles after the command with a 1-cycle memory.
  `tsr_mvm_ws` loads at most one burst every 2 cycles; its `idle` is the `cmd_ready` register.
  The cluster drives the front end's `beat_ready` from flops (back-end ready AND `act_ready`):
  inside the front end it fans out through one LUT to ~2.5 K clock enables.
- The front end consumes one FIFO beat per cycle (META beats included: they occupy a cycle of the
  stream and produce no tagged beat).
- At the last beat consumed the WS source has nothing outstanding; `idle` = no command, nothing
  outstanding, FIFO empty.

---

## 5. Activation buffer organisation (tsr_mvm_actbuf)

Spec §8, "Resource estimate": "an activation buffer of 1024 bits × 292 per token per cluster" = one full QA image per token
(QS + D + BS) at K_MAX 32768. Organisation (width scales with T_MAX because the T tokens share
every weight beat and are read together):

| Region | Per token | Geometry (K_MAX 32768) | Addressed by |
|---|---|---|---|
| QS | `QS_WORDS = ceil(K_MAX/128)` words × 1024 bits = 2 banks (even / odd QS line) × 512 bits | 256 words | QS word w = lines {2w+1, 2w} |
| DBS | `DBS_LINES = tsr_actbuf_dbs_lines(K_MAX)` × 512 bits: D lines at 0..d_lines−1, then BS lines | 72 lines (A256 8 + 64; A32 64) | line − ceil(K/64) |

Total per token at K_MAX 32768: 256 × 1024 + 72 × 512 bits = 32,768 B + 4,608 B = 37,376 B =
`QA_LINES(A256, K_MAX) × 64`, exactly the spec's figure (QS 256 words + D/BS 36 words of 1024 bits;
A32 needs QS 256 + D 32 words; the spec allows the separate Dx/BS memory). Writes: one 512-bit line per cycle into bank
(tok, addr[0]) word addr >> 1 (QS) or DBS line addr of token tok. Reads: one QS word and one DBS
line per cycle for all tokens. Generic: `tsr_sram_1r1w` per (token, bank) and per token for DBS,
`RD_LAT = ACT_RD_LAT`; U200 T_MAX 1: 2 × 8 + 8 = 24 BRAM36 per cluster (`rtl/tech/xilinx`
block RAM mapping; measured 24 RAMB36 per token per cluster, §9.3). Reads and
writes of one MVM never overlap in time; `act_ready` orders them (§3.2).

---

## 6. tsr_mvm: command flow, broadcast, gather

### 6.1 Flow (top agent)

1. Accept `cmd` (idle). N = 0: `done` (§3.0), nothing else.
2. Send `bc_cmd` (§3.1) to every cluster; start the gather with `{acc, t, n, dst_line,
   dst_stride, expect_ch}` where `expect_ch[c] = (c × R < N)` (channels with rows; ≥ 1 bit set).
3. Broadcast: SPM reads `x_line .. x_line + t·qa_lines − 1`, one per cycle when the client port
   allows; each response → one `bc_line` (§3.1). Reads have no backpressure; the broadcast pipes
   take one item per cycle unconditionally.
4. `done` when the broadcast is complete and the gather reports done (§6.3).

### 6.2 Cluster (top agent)

`tsr_mvm_cluster #(CH)`: §3.1 input, §3.2 setup / gating, `tsr_mvm_front` + `tsr_mvm_actbuf` +
`tsr_mvm_back`, §3.5 result tagging (row counter), WS port pass-through. A cluster processes one
MVM at a time; bc items of the next MVM arrive only after its last result was taken. Clusters with
`rows_c = 0` stay silent. Placement: in the SLR of channel c's DDR controller (spec §8, placement); on the
U200 at T_MAX 8 two clusters do not fit SLR1's dynamic region, so clusters 1 and 2 move out of
SLR1 and get their weights through the weight-stream pipes (§14.5).

### 6.3 Gather (tsr_mvm_gather): partial lines, byte enables, ACC, tails

Semantics (tsr_sim.c `ex_mvm`, the write loop): token tk < T writes f32 element n_row of its y vector at
SPM line `dst + tk·stride + n_row/16`, lane `n_row % 16`, value `y` or (ACC) `fadd(y_old, y)`
(spec §21, "ACC addition": one f32 addition with the fp rules, old value first — addition is
commutative bit-exactly), and the bytes after element N−1 in the token's last line are 0 (also
with ACC; spec §21, "Tail bytes").

Partial-line scheme (each cluster c owns rows [cR, cR + rows_c), consecutive; a 16-element line
may hold rows of several channels when R % 16 ≠ 0):
- One **assembler per cluster** collects the cluster's consecutive rows that fall into the same
  SPM line (all T tokens at once — every result item carries all tokens). The group is complete at
  a row with `n_row % 16 == 15` or `res_last`. It then emits one **job per token** tk < T:
  `{line = base_tk + n_row/16, lanes [lo, hi] of this cluster, tail = the group holds row N−1}`
  with `base_tk = dst + tk·stride` (computed once per MVM, T ≤ 8 adds).
- A job writes its line with `spm_wr_be` = 4 bytes per lane in [lo, hi], plus every lane > hi
  when `tail` (data 0). Hence each y byte (incl. the tail) is written by exactly one job of
  exactly one cluster per token, and nothing outside the y lines is written. Only the owner of
  row N−1 writes the tail.
- **ACC read-modify-write** (hazard-free by construction): a job first reads its SPM line (the
  read is issued only when space for the response is reserved, R4), adds `fadd(old_lane, y_lane)`
  for its lanes [lo, hi] (16 `tsr_fp_add` lanes, one line per cycle), and writes back with the
  byte enables above. Two jobs that share an SPM line belong to different clusters and have
  disjoint lane sets, so whatever order their reads and writes take, each lane's old value is
  read before the only write of that lane, and no job overwrites another job's lanes (a read
  may see the other job's new lanes; they are not written back). Different tokens never share a
  line (`stride ≥ y_lines` whenever t > 1; overlapping token ranges are undefined). The old values
  are those before the MVM: nothing else writes y during the MVM. Checked in tb/common
  (`test_mvm_model_gather_jobs`): `mvm_model.gather_jobs` applied in random job orders with
  random ACC read/write interleavings equals the ex_mvm write loop and writes every y byte once.
- Throughput: the job queue drains at one SPM write per cycle (ACC: one read + one write per
  cycle; `tsr_spm` has separate read and write ports). Jobs per token ≤ `y_lines + NUM_CH − 1`.
  `res_ready` falls only when an assembler's job queue is full (local reason).
- `done` (registered pulse): every cluster with `expect_ch[c]` delivered `res_last`, every
  assembler and queue is empty, and the last SPM write was accepted.

### 6.4 done and quiescence

`tsr_mvm.done` = broadcast complete ∧ gather done. Quiescence follows: a cluster's last result
needs its last weight beat, which is the last of the exactly `beats_c` requested beats, so no WS
read is outstanding; no SPM read is outstanding (the gather's done waits for its last write,
which follows its last read). The integration step asserts it in simulation (§10.4).

### 6.5 Attention modes in the cluster (designed)

The cluster gains `ATTQK` and `ATTPV` for attention over the q8_0 KV cache; the weight MVM of this section
is unchanged. The contract is [AT](design/03-attention-kv.md) §5.2-§5.3 (operands) and §6 (numerics and the
normative flash-attention sequence); the hardware structure, throughput and cost are AT §8 and §9:
`ATTQK` uses the front end's Q8_0 path with row and segment counters, per-slot activation read addresses and a
gather line buffer per pass row (at most 128 rows, 2 RAMB36) with a row-maximum tracker; `ATTPV` adds a
64-beat delay FIFO, a weight unit (`w = p·d`, 16 `fmul` per cycle, shared exponent alignment), a PV mode of
`tsr_mac_lane` that accumulates block-floating-point products exactly in the DSPs, a capture bank and convert /
add pipes. `RB_MIN` and the throughput statements of §7.5 and §9 apply to weight MVMs only. Both modes are on the
simulator (`sw/sim/tsr_at.c`, `tsr_ref_at.c`); no attention RTL exists (W4).

---

## 7. Back end: numerics and microarchitecture (tsr_mvm_back)

### 7.1 Per-beat arithmetic (bit-exact; per token t, every t < T_MAX)

Q8_0 (`term_q8_0_ptr`, `tsr_ref.c` 534–538; `dot32_i8` 525–531):
```
isum_g  = Σ_{L<32} w[32g+L] · qs_t[32(b+g) + L]                       g = 0, and g = 1 if two
term    = fmul( fmul(d_g, Dx_t[b+g]), i32_to_f32(isum_g) )            d_0 = d_a, d_1 = d_b
```
Q4_K (`term_q4_K_ptr`, 543–567), sub-block j = 4·half + g:
```
S_j     = Σ_{L<32} w[32g+L] · qs_t[256s + 128·half + 32g + L]           (q_w ∈ [0, 15])
isum    = Σ_{j<8} sc_j · S_j                       (half 0 contributes j 0..3, half 1 j 4..7)
msum    = Σ_{j<8} m_j · (BS_t[16s + 2j] + BS_t[16s + 2j + 1])
term    = fsub( fmul(fmul(d_a, Dx_t[s]), i32_to_f32(isum)),
                fmul(fmul(d_b, Dx_t[s]), i32_to_f32(msum)) )             at half = 1
```
Row accumulation (`tsr_dot_row` 600–635, spec §7, "Row accumulation"): 16 partials start at +0.0 per row; term
i → `p[i % 16] = fadd(p[i % 16], term)` in term order; y = `tsr_tree16(p)` (592–598):
`l1[k] = fadd(p[2k], p[2k+1])`, `l2[k] = fadd(l1[2k], l1[2k+1])`, `l3[k] = fadd(l2[2k],
l2[2k+1])`, `y = fadd(l3[0], l3[1])`. A partial without terms stays +0.0; the first add of a
partial is a real `fadd(+0.0, t)` (−0.0 becomes +0.0, NaN canonical). `fmul`/`fsub`/`fadd` =
`tsr_fp_mul` / `tsr_fp_add` (`tsr_ref.c` 99–101, rules 54–71); `fp16_to_f32` exact (136–158);
`i32_to_f32` RNE (107–132; isum / msum may exceed 24 bits).

### 7.2 MAC lanes

`TSR_MVM_GROUPS = 4` instances of `tsr_mac_lane #(.N(32), .T(T_MAX), .LAT(MAC_LAT))` per
cluster (128 lanes; Q8_0 uses groups 0–1). Activations: Q4_K the QS word's 1024 bits per token
(lane L = byte L); Q8_0 the selected 512-bit half to lanes 0..63 (lanes 64..127 weights are 0).
`tsr_mac_lane` is the tech wrapper of spec §8, "DSP packing"; its sums are exact `SUM_W = 21`-bit two's
complement in every implementation. `rtl/tech/generic/tsr_mac_lane.sv` is behavioural; the Xilinx
variant `rtl/tech/xilinx/tsr_mac_lane.sv` (selected by `rtl/filelist_xilinx.f`; done in M3 part C
instead of M4; tb/tech compares it with the generic lane cycle by cycle, and its synthesised
netlist in post-synthesis simulation) puts every product on a DSP48E2 and packs two tokens per DSP
(WP486 INT8 packing, the weight as the shared B operand): tokens 2s and 2s + 1 of lane i share
slot s with `AD = x[2s]·2^18 + x[2s+1]` in the pre-adder, `M = AD · w`, and a constant 2^17 on the
post-adder's C port, which folds WP486's fabric sign correction into the DSP, so `P[33:18] =
x[2s]·w` and `P[15:0] = x[2s+1]·w` exactly; an odd T leaves its last token alone in a slot. DSPs per
instance: N × ⌈T / 2⌉, per cluster (4 instances, N = 32) 128 × ⌈T_MAX / 2⌉: packing needs a second
token, so T_MAX 1 and 2 both take 128. The adder trees stay in fabric; the MAC_LAT − 1 internal
registers are allocated M, P, a first adder-tree cut, AD, the input (A / D / B) registers, then
further cuts (wrapper header; U200: MAC_LAT 9, a cut after every level, §14.4 item 4). Measured
(Vivado 2024.2, routed final runs of §14.3): **128 MAC DSPs per cluster at T_MAX 1 (one token per
DSP), 512 packed at T_MAX 8, 2,048 for the four clusters of k4** (75 % of its 2,728 DSPs); the
lanes' fabric per cluster is 2,130 LUT / 2,212 FF at T_MAX 1 and 17,040 LUT / 17,696 FF at T_MAX 8
(§9.3).

### 7.3 Accumulators: two lanes, 8 partials each

- Two accumulate lanes per token, each one `tsr_fp_add` (`ADD_LAT`) with its 8 partials: lane 0
  holds p0, p2, …, p14, lane 1 p1, p3, …, p15 (`p[k]` in lane k % 2, slot k / 2).
- Q8_0: the two terms of a beat (b even → lane 0, b+1 → lane 1) are accumulated in the same cycle.
  Within a lane a slot is hit once per 8 quant beats, so two terms of one partial are ≥ 8 cycles
  apart (spec §7: "a pipelined fp adder of latency at most 8 cycles, with II = 1"). Requirement: from the
  issue of `fadd(p, term)` to the cycle in which the updated p can be the operand of the next
  issue is ≤ 8 cycles (partials in registers read in the issue cycle and written from `out_valid`
  make this `ADD_LAT ≤ 8`; any extra read stage must be bypassed).
- Q4_K: one term per superblock (≥ 2 quant beats apart), into lane s % 2.
- Rows: two partial sets per lane; row r+1 accumulates into the other set while row r's last adds
  drain; when row r's last add has completed its set is copied into the tree's input registers
  (snapshot) and reset to +0.0 (or marked "fresh": the next first add then uses +0.0).

### 7.4 Tree (background, streaming) — revised by the lead after part A

The tree reduces each snapshot in the fixed order of §7.1 (`tsr_tree16`: 8 + 4 + 2 + 1 adds).
Throughput, not latency, must bound it: rows in the M3 gate model are short (Qwen2.5-0.5B K = 896
Q8_0: 15 beats incl. META; its 152 K-row output head dominates the token) and M8 MoE expert rows
are shorter still (Gemma 4 26B-A4B `ffn_down_exps` K = 704: Q8_0 12 beats; Q4_K does not apply, as
704 is not a multiple of 256). Suggested structure: one pipelined `tsr_fp_add` per tree level and
token (level 1 issues the 8 pairs of a snapshot on 8 consecutive cycles, level 2 its 4 pairs as
level-1 results arrive, …), with a ring of snapshot / intermediate registers deep enough for every
row in flight (latency ≈ 4·ADD_LAT + 15), so a new snapshot can enter level 1 every 8 cycles. Other
structures are fine if they meet §7.5 and keep the exact pair order. Each result goes to a result FIFO whose space was reserved before
the row's first beat was accepted.

### 7.5 Throughput, RB_MIN, stalls

- Q8_0: 32 terms per 17-beat chunk (≤ 2 terms per cycle) — the two accumulate lanes take them at
  full rate. Q4_K: ≤ 1 term per 2 cycles (4 per 9-beat chunk).
- **RB_MIN** (requirement, revised): with `res_ready = 1` and beats offered every cycle, the back
  end never drops `beat_ready` when every row has `ROW_BEATS ≥ RB_MIN`, with **RB_MIN ≤ 9**
  (target 8 with the streaming tree of §7.4; the part-A plan of `4·ADD_LAT + 12` = 32 at ADD_LAT 5
  made the M3 gate model's K = 896 rows tree-bound and is withdrawn). Also the accumulator side:
  a row's last adds draining while the next row (≥ RB_MIN beats) starts must not stall. Shorter
  rows (e.g. Q8_0 K = 32: 2 beats, Q4_K K = 256: 3 beats) may stall at row starts (correctness
  unaffected). The back agent documents the achieved bound in its header and tests it at
  ROW_BEATS = RB_MIN for both formats, T = 1 and T_MAX.
- Pipeline (suggested): input reg → act read (ACT_RD_LAT) → MAC (MAC_LAT) → int scale (2) →
  i32→f32 (2) ∥ d·Dx (MUL_LAT, starts at the act read) → ·f32(sum) (MUL_LAT) → Q4_K fsub
  (ADD_LAT) → accumulate (ADD_LAT). All stages ce = 1: `beat_ready` falls only to reserve result
  space and partial-set / tree capacity, never to hold a beat already inside.

### 7.6 Integer widths (any SPM content is a legal QA image)

The ISA does not restrict MVM inputs to QACT output: QS may hold −128, D any f32 (NaN, Inf,
subnormal → DAZ in fmul), BS any int16 (LD can load arbitrary bytes). Exact widths:

| Value | Bound | Width (signed) |
|---|---|---|
| group sum (Q8_0 isum, Q4_K S_j) | 32 · 128 · 128 = 2^19 | 21 (`TSR_MVM_SUM_W`) |
| Q4_K isum | 8 · 63 · 61440 < 2^25 | 26 |
| Q4_K msum | 8 · 63 · 65536 < 2^25 (QACT BS: < 2^21) | 26 (`TSR_MVM_ISUM_W`) |

The spec table's "msum 23 bits" holds for QACT-produced BS only (proposal SP1).

---

## 8. fp library additions (fp agent)

### 8.1 tsr_fp_add

`tsr_fp_add #(LAT)`: `clk rst_n ce in_valid a b sub → out_valid y`, bit-exact `tsr_fadd`
(`sub = 0`) / `tsr_fsub` (`sub = 1`) for every input (`tsr_ref.c` 99–100: DAZ inputs, RNE in the
binary32 range, FTZ of a subnormal result with its sign, canonical NaN, IEEE signed zeros), fixed
latency `1 ≤ LAT ≤ 8` with the `ce` semantics of `tsr_fp_mul`, fully pipelined. Default LAT = the
smallest that meets 300 MHz within ~6 LUT levels per stage = **5** (Vivado 2024.2 out of context,
xcu200 at 3.333 ns: WNS +1.10 ns, ≤ 5 LUT levels per segment; LAT 3 and 4 close timing but need 8
levels, outside the rtl-m2 §1 budget). `tb/fp` test_fp_default_lats checks that
`tsr_pkg::TSR_DEF_FADD_LAT` / `TSR_DEF_FMUL_LAT` equal the module defaults;
users pass `LAT` explicitly and work for any 1..8. Mandatory vectors: cancellation to exact zero
(+0 for x − x, −0 + −0 = −0), exact subnormal results flushed with sign (0x00C00000 − 0x00800000
→ +0, its negation → −0), DAZ of subnormal operands, Inf − Inf, NaN payloads → 0x7FC00000,
overflow to Inf, RNE ties at every alignment distance, operands 2^−126 apart in exponent, plus
random and the `fp_sweep.cpp` harness (add / sub classes) against `common.tsr_ref`.

### 8.2 tsr_i32_to_f32

The `tsr_fp_pkg` function exists and is verified (tb/fp: directed, every exponent with random
mantissas, opt-in exhaustive 2^32 sweep). It is ~7–9 LUT levels as one function: the back end may
use it as one stage between registers only if timing allows, else a two-stage split (normalise |
round) checked bit-for-bit against the function on the same vectors.

---

## 9. Throughput, timing and cost

### 9.1 Throughput (T = 1, memory delivering 1 beat/cycle/channel)

Weight-bound: an MVM takes `max_c rows_c × ROW_BEATS` cycles plus overhead (dispatch ~10, setup
~20, broadcast `T × QA_LINES`, back-end pipeline ~30 + tree ≤ 4·ADD_LAT + 8, gather ~10–20), as
the simulator's cycle model (`tsr_sim.c` header: 64 + T·QA_LINES + max beats). Measured at
default parameters with an ideal memory (§13.2): 69 + T·QA_LINES (Q8_0) / 72 + T·QA_LINES (Q4_K)
cycles of overhead per MVM at T = 1 (within 5 cycles of the model), 95–100 + T·QA_LINES at T = 8,
steady state 1.0000 cycles per weight beat (part C pipes: 1–3.5 cycles more per MVM, §14.7; at
the U200 bitstream parameters of §14.2, 11–12 cycles more than at the defaults, steady state still
1.0000: §13.2 `u200_rst3`).
Example: Qwen2.5-32B `ffn_down` (Q4_K, N 5120, K 27648, 4 channels): 1280 × 243 = 311,040
cycles = 1.04 ms at 300 MHz; broadcast 493 lines (0.16 %). Rows shorter than RB_MIN (≤ 9, §7.5)
are tree-bound; tiny K with many channels and tokens may be gather-bound (≤ 1 SPM line write per
cycle).

### 9.2 300 MHz: critical structures and register slices

1. **Broadcast bus** (137 + 527 bits, one source flop → NUM_CH clusters in three SLRs): one
   `tsr_vpipe` per cluster; U200: `BC_PIPE ≥ 2` for clusters in another SLR (a flop on each side
   of the SLL crossing; Laguna registers in M4), inside the cluster one more register in front of
   the activation-buffer RAMs (2·T_MAX + T_MAX RAMs share the write data).
2. **Result bus** (33 + 32·T_MAX bits per cluster): `tsr_rpipe` (registered valid and ready),
   `RES_PIPE ≥ 2` across SLRs.
3. **Activation buffer read** (T_MAX × 1024 + 512 bits): RAM output register (`ACT_RD_LAT ≥ 2`),
   then the Q8_0 half select (2:1) and the Q4_K / Q8_0 lane select into the MAC input register.
4. **MAC lanes**: products + 32→1 adder trees; generic `MAC_LAT ≥ 3` (product register, about two
   adder levels per stage); DSP cascades in M4. U200 with the DSP-packed lane: `MAC_LAT 9`
   (§14.2, §14.4 item 4).
5. **Weight unpack**: nibble / byte select per lane (1–2 LUT levels) into a registered
   `tsr_mvm_beat_t`; the parser's counters and the meta register select (32:1 × 16 bits for
   Q8_0 d, 4:1 × 128 bits for Q4_K) are registered before use.
6. **Integer scale**: 4 (6 × 17) and 8 (6 × 18) products + 4- and 8-input sums, two stages.
7. **i32 → f32** (§8.2) and the fp pipelines (`MUL_LAT 4`, `ADD_LAT` §8.1).
8. **Accumulator loop**: 8:1 partial read mux (32 bits) into the adder input register; adder output
   to the slot registers (1 LUT); the ≤ 8-cycle recurrence of §7.3.
9. **Gather**: 16 `tsr_fp_add` lanes for ACC; SPM write data / 64 byte enables fan out to every
   SPM RAM (rtl-m2 §12: keep `SPM_RD_LAT 4` and the replicated SPM input registers on U200).
10. **Not critical** (multi-cycle by construction): `CH × R`, `rows_c × row_beats`, `dst +
    tk·stride`, `c × R < N` — computed once per MVM over several cycles; the WS address / beat
    counters (40 bits, 5 CARRY8) and the burst length (`beats_left ≥ 64` = OR of the upper bits).

The memory side is unchanged (every `m_*` from a flop in `tsr_mem_arb`).

### 9.3 Cost of T_MAX (synthesis T_MAX = 1 and 8; tests up to 8)

Replicated per token in every cluster: 128 MAC products (U200: DSP48E2, two tokens per DSP from
T = 2, §7.2), 4 adder trees (fabric), activation buffer (QS 2 banks + DBS: 24 BRAM36 at K_MAX
32768), integer scale (12 small multipliers), 2 `i32_to_f32`, 4 `tsr_fp_mul` (8 DSP), 4
`tsr_fp_add` (2 accumulate, 1 Q4_K subtract, 1 tree), 3 × 16 × 32 partial / snapshot flops, +32
result-bus wires, +512 bits per gather assembler. Shared per cluster: front end, weight FIFO (8
BRAM36). The pre-synthesis estimate here was ~6–8 K LUT, 136 DSP and 24 BRAM36 per token per
cluster, i.e. 200–250 K LUT, 2.2 K DSP (packing: ~1.1 K) and 768 BRAM36 for T_MAX 8 × 4 clusters;
the measurement below replaces it. Its DSP figures were 2× low: 128 products × 8 tokens × 4
clusters = 4,096, which packing halves once, to the 2,048 measured.

Measured (§14: Vivado 2024.2, `rtl/filelist_xilinx.f`, the U200 parameters of §14.2, hierarchical
utilisation of the routed final k1 / k4 T_MAX 1 / k4 T_MAX 8 runs; per cluster):

| Per cluster | T_MAX 1 | T_MAX 8 | Growth |
|---|---|---|---|
| MAC DSPs (4 × `tsr_mac_lane`; packed from T_MAX 2) | 128 | 512 | 128 per pair of tokens (128 × ⌈T_MAX / 2⌉) |
| non-MAC DSPs: integer scale (back end) | 12 | 96 | 12 per token |
| non-MAC DSPs: 4 `tsr_fp_mul` (2 DSP each) | 8 | 64 | 8 per token |
| non-MAC DSPs: cluster level, outside the back end (per-MVM setup arithmetic) | 2 | 2 | – |
| **DSP per cluster** | **150** | **674** | |
| MAC lanes, fabric (adder trees, MAC_LAT 9 cut registers) | 2,130 LUT / 2,212 FF | 17,040 LUT / 17,696 FF | ~2.1 K LUT / 2.2 K FF per token |
| whole cluster | 9,283 LUT / 14,331 FF / 32 RAMB36 | 57,586 LUT / 64,575 FF / 200 RAMB36 | ~6.9 K LUT, 7.2 K FF, 24 RAMB36 per token |

Outside the clusters: QACT 32 DSP (its 16 `tsr_fp_mul`) and 8 RAMB36, the SPM 128 URAM; nothing
else uses DSPs, RAMB36 or URAM. Whole core: k1 182 DSP (128 MAC + 22 + 32), k4 T_MAX 1 632
(512 MAC + 88 + 32), **k4 T_MAX 8 2,728 (2,048 packed MAC DSPs + 648 non-MAC in the clusters + 32
QACT)**, 808 RAMB36, 128 URAM, 270.9 K LUT (the four clusters 230.3 K, their MAC lanes 68.2 K). The
VU9P has 6,840 DSP / 2,160 RAMB36 / 960 URAM; the U200 platform's dynamic region 5,880 / 1,860 /
800 (`platforminfo`, §14.5): k4 uses 46 % of its DSPs, 43 % of its RAMB36 and 16 % of its URAM.
The T_MAX choice for the bitstream: §14.6 (T_MAX 8 recommended).

---

## 10. Verification

Oracle chain (spec §17, "Verification strategy"): `libtsr_ref` (bit-exact numerics) for the blocks, the ISA simulator
for the core. Bit-exact first; every test must be shown to fail on at least one deliberate
mutation (scratch copy) and the report says which.

### 10.1 Infrastructure (tb/common, architecture)

(In the tree: `tsr_ref.py`, `isa.py`, `mvm_model.py`, `mem_model.py`.)

- `tsr_ref.py` (`ref()`): every reference function the M3 tests need — `fadd fsub fmul daz
  i32_to_f32 fp16_to_f32`, `tree16`, `dot_row`, `term_q8_0 / term_q4_K / term(fmt, ...)`,
  `repack_row / unrepack_row`, `row_beats`, `rows_per_ch`, `row_location`, `tensor_ch_beats`,
  `dequant_row`, `get_scale_min_k4`, `fmt_*` (block size, chunk blocks, quant bytes, ggml block
  bytes, `fmt_valid`, `fmt_k_valid`), `qa_kind_for_fmt`, QA helpers (`qa_*lines`, `qa_split`,
  `qact`). f32 values cross as bit patterns.
- `isa.py`: `MVM_CCMD_T` (137), `MVM_BCLINE_T` (527), `MVM_FCMD_T` (137), `MVM_BEAT_T` (1175)
  with `pack_array / unpack_array` for the packed-array fields, `enc_mvm(...)`, `M3_OPS`;
  `test_common.py` `loop_structs` checks widths, field positions and `tsr_actbuf_*` /
  `tsr_mvm_rows_c` against `tsr_pkg.sv` through `tb/common/hdl/tsr_tb_loop.sv`.
- `mvm_model.py` — golden model of the MVM blocks on libtsr_ref (itself verified: every term is
  recomputed from the tagged beat stream and must equal `tsr_term_*`, every row `tsr_dot_row`):
  - front: `front_beats(fmt, K, rows)` → the exact §3.3 stream (dicts; extra keys row / chunk /
    beat), `pack_beat / unpack_beat / beat_fields`;
  - back: `QaView.of(kind, K, img)`, `beat_math(fmt, beat, view)` (group sums S, Dx, Q4_K part /
    msum per token), `term_q8_0_values / term_q4_K_values`, `terms_from_beats`, `accumulate`,
    `tree16`; `Cluster(fmt, K, rows, qas)` → `.beats`, `.terms[r][t]` (`Term`: i, partial, lane,
    bits, isum, msum), `.partials[r][t]`, `.y[r][t]`, `.actbuf(t_max)`, `.results()`;
  - activation buffer / broadcast: `actbuf_image(kind, K, qa)`, `bc_lines(kind, K, qas)`,
    `ccmd(...)`, `fcmd(...)`;
  - gather / whole MVM: `gather_jobs(N, num_ch, T, dst, stride)` (`Job` with `be()`),
    `apply_mvm(spm, ...)` (the ex_mvm write loop), `mvm_rows`, `mvm_reference(spm, read_beats,
    ...)`, `layout_channels(fmt, K, rows, num_ch)` (per-channel DRAM images at W_OFF);
  - data: `rand_f16 / rand_f32 / rand_blocks / rand_row(raw=) / rand_qa(mode='qact'|'raw')` with
    special-value rates, `ftz_window_dx(d)` (a Dx whose product with d lies in the FTZ boundary
    window [2^−126 − 2^−150, 2^−126)).
- `mem_model.MemModel.src_beats[c][s]` counts read beats per memory-side source (id[3:2]). It is the
  port-level form of the spec's "MVM weight beats per channel = rows_c × ROW_BEATS" rule (source WS).
- `tsr_vpipe` / `tsr_rpipe` are tested by `test_pipes_sim` (`tb/common/hdl/tsr_tb_pipes.sv`, DEPTH 0 / 1 / 3).

### 10.2 Block tests (spec §17, "Block-level test coverage")

- **fp** (`tb/fp`): §8.1 vectors incl. the FTZ boundary classes, signed zeros, Inf, NaN; random;
  ce stalls; LAT 1..8; sweep harness.
- **front** (`tb/mvmfront`): K random over block multiples incl. partial chunks (Q8_0 K % 1024 ≠
  0, Q4_K K % 1024 ≠ 0), K = block size, K_MAX; rows 1..many; channels with N < C and odd N
  (through `rows`); random DRAM bytes (special fp16 d, −128 weights); `MemModel` with random
  latency / backpressure / stalls and the outstanding limit (a one-channel DUT wrapper in
  `tb/mvmfront/hdl` exposes the WS port with the `m_` naming and id `{2, local}`); checks: tag stream ==
  `front_beats`, burst rules, exactly `beats` requested (`MemModel.src_beats`), id 8,
  `rd_rsp_ready` = 1, 1 beat/cycle with an ideal memory, FIFO sizing (no stall with latency ≤
  WS_FIFO_BEATS − 64 − ε).
- **back** (`tb/mvmback`, with the real activation buffer): rows / QA from `mvm_model` (realistic
  and raw incl. subnormal / Inf / NaN / signed zero d, Dx and products in the FTZ boundary range
  [2^−126 − 2^−150, 2^−126)), Q8_0 odd block counts, Q4_K partial chunks, BS extremes (msum
  26 bits), T_MAX 1 and 8, latency parameters at their extremes; row results bit-exact; term /
  partial probes where useful; throughput: no `beat_ready` drop at ROW_BEATS = RB_MIN.
- **top** (`tb/mvm`): `tsr_mvm` with real clusters (or the gather / cluster alone with stubs):
  broadcast order and mapping, N = 0, rows_c = 0 clusters, R % 16 ≠ 0 (shared lines), N % 16 ≠ 0
  (tails), ACC, strides with gaps, T = T_MAX, results identical for `BC_PIPE` / `RES_PIPE` 0, 1,
  3 (spec §8: results do not change with the SLR pipeline depth parameters), SPM written bytes == y lines
  exactly, done timing and quiescence.

### 10.3 Trace sets

`traces/gen` (M1 generator, 24 + 8 traces with MVM, U200 / T_MAX 8) and `traces/m3`
(`traces/m3/regen.sh`, `tsr-trace-gen --ops m3`): `tmax1` 64 (U200, three long weight
streams), `tmax8` 32, `small1ch` 24 (1 ch, ADDR_BEAT_W 22, K_MAX 4096, T_MAX 2), `ch3` 16, `ch16`
12 (16 channels, K_MAX 8192, T_MAX 2), `errors` / `errors8` 68 each (every MVM fault and every M2
error ending). Replayed bit-exactly by `tsr-trace-replay` and by the RTL core (§10.4, planned).

### 10.4 Integration (HAS_MVM = 1)

The core-level bench is the tile-only replay of `traces/m3` and `traces/gen` into the issue FIFO under the
memory regimes of rtl-m2 §12 (rand / fast / slow / ordered), planned (W5), with PERF checks
(`PERF_CH_RD_BEATS[c]` == the trace's for c ≥ 1, i.e. LD + `rows_c × ROW_BEATS`; MVM cycles > 0),
implementation-parameter variants (WS FIFO 64, pipes 0 / 3, SPM_RD_LAT 1 / 4, RD_OUTSTANDING 1) and
throughput (1 weight beat/cycle/cluster at T = 1 with an ideal memory for ROW_BEATS ≥ RB_MIN).
Simulation-only checks in `tsr_core` (I1, I2, I4, §13.1): MVM SPM / WS traffic only while the MVM unit holds
an op; at `done` no WS request pending and no WS read outstanding on any channel; per MVM and channel exactly
`rows_c × ROW_BEATS` weight beats requested, in bursts that follow the §4 rule (I4).

---

## 11. Spec proposals (for the lead)

- SP1 **Integer widths for arbitrary QA content**: "The MVM accepts any byte of the SPM as QA (also legal
  when it is not QACT output). If BS is an arbitrary int16, the Q4_K msum needs a signed 26 bits (23 bits for a
  BS made by QACT)." (Spec §7, "Integer widths" table note.)
- SP2 **LANES**: "LANES = BEAT_BITS / 4 (the number of weights in one Q4_K beat) and is not an independent
  parameter" —
  the cluster structure (stage 4/6) fixes 128 lanes for 512-bit beats; CFG0 keeps reporting it.
- SP3 **PERF_MVM_CYCLES in hardware**: cycles in which the MVM unit holds an instruction (accept →
  done), like rtl-m2 P5 for the other counters.
- SP4 **N = 0 MVM**: reads nothing from the SPM either (no observable difference; states that the
  x operand is not accessed when there are no rows).
- SP5 (superseded: §7.4 / §7.5 require RB_MIN ≤ 9) **Row-boundary throughput**: spec stage 9 says the previous row's tree finishes in the background; add
  that this holds for rows of at least RB_MIN = 4·ADD_LAT + 12 beats (shorter rows, e.g.
  K = 896 Q8_0, stall at row starts; performance only).
- SP6 **Stage 2 tags**: the front end absorbs META beats (they carry no work for the back end);
  the tag set of stage 2 is then {QUANT data, block/superblock, half, row end, MVM end}.

---

## 12. Reference map

(Line numbers of `tsr_ref.c` match the tree at the time of writing. Use the symbol names when they drift.)

| Behaviour | Reference |
|---|---|
| fp rules (DAZ, RNE, FTZ, canonical NaN) | `tsr_ref.c` 54–71 (`daz_bits`, `fin`), 99–101 (`tsr_fadd/fsub/fmul`) |
| int → f32, fp16 → f32 | 107–132, 136–158 |
| ROW_BEATS, chunk geometry, byte placement | 256–269 (`chunk_beats`, `tsr_row_beats`), 283–294 (`block_offsets`) |
| rows per channel, row location | 335–352 |
| Q4_K 6-bit scales | 359–367 (`tsr_get_scale_min_k4`) |
| QA layout | 418–447 |
| block terms | 525–538 (Q8_0), 543–567 (Q4_K) |
| 16 partials, tree, row walk | 592–598 (`tsr_tree16`), 600–635 (`tsr_dot_row`) |
| MVM checks / weight beats / compute / write | `tsr_sim.c` `chk_mvm` and `ex_mvm`; the checks of the issue path are EX §8.1 |

---

## 13. Integration and results (M3 part B, integration step)

Results of RTL simulation of `tsr_core` with the MVM unit. The core-level benches that produced them are
planned again as tile-only replays of `push.bin` traces into the issue FIFO (W5). The results stay valid for
the blocks in the tree: the MVM unit's cycle counts, the bit-exactness of its numerics and its quiescence
rules.

### 13.1 tsr_core with HAS_MVM = 1

- **Default `HAS_MVM = 1`** (`tsr_core` parameter): the U200 bitstream executes llama.cpp graphs,
  which need `MVM`; the default parameters are the bitstream's. `rtl/lint.sh` lints both values.
  `HAS_MVM = 0` keeps the M2 netlist (tie-offs in `g_nomvm`).
- **Wiring** (§2, rtl-m2 §10): `tsr_mvm` on the dispatch port (`mvm_cmd_*`, `mvm_done`), SPM
  client `TSR_SPM_CL_MVM` (read and write port), weight-stream source `TSR_SRC_WS` of arbiter c ↔
  cluster c (request, local id, `rd_rsp_ready`; the shared response data / last / id of channel c),
  `perf_mvm_busy` → `tsr_csr`. No port, `tsr_pkg` or filelist change.
- **New parameters (additive)**: `MVM_WS_FIFO` (512), `MVM_BC_PIPE` (1), `MVM_RES_PIPE` (1) and the
  latencies `MVM_ACT_RD_LAT` (2), `MVM_MAC_LAT` (3), `MVM_MUL_LAT` (4), `MVM_ADD_LAT` (5),
  `MVM_WS_RD_LAT` (2) (defaults = `tsr_pkg::TSR_DEF_*`); since part C also `RST_PIPE` (0, reset
  register stages, §14.4 item 3). Elaboration checks with `HAS_MVM = 1`:
  `LANES == 128`, `MVM_WS_FIFO` a power of two ≥ 64, `MVM_ADD_LAT` 1..8, `MVM_WS_RD_LAT` 1..4, the
  other latencies ≥ 1. The `$fatal` on `HAS_MVM = 1` is gone.
- **Simulation checks** (`ifndef SYNTHESIS`, armed by the first reset edge), extending rtl-m2 §12:
  - I1 a unit pulses `done` only while it holds an op (never in its accepting cycle); MVM `cmd_ready` low
    while it holds the op; `perf_mvm_busy` equals "holds an op" in every cycle (the PERF_MVM_CYCLES meaning).
  - I2 a unit drives SPM requests only while it holds an op (DMA / QACT: a request in the accepting cycle is
    allowed; the MVM unit strictly after its accept); MVM SPM read responses only while it holds the op;
    weight-stream requests / responses only while the MVM unit holds an op; WS `rd_rsp_ready` constant 1.
  - I4 at `mvm_done`: no WS request offered, WS beats requested == received on every channel
    (counted at the arbiter's source port), no MVM SPM request / write offered, no MVM SPM read
    outstanding; per MVM and channel exactly `rows_c × ROW_BEATS` beats requested (never more at any time),
    every request at the next address with `len = min(64 − addr % 64, beats left)` (§4).
- **Lint**: `rtl/lint.sh` lints the default core (U200 with MVM), 6 `HAS_MVM = 0` configurations
  and 10 `HAS_MVM = 1` configurations (T_MAX 8; 1 / 2 / 3 / 16 channels; K_MAX 256 with a 64-line
  SPM and 32-beat channels; pipes 0 and 3 with FIFO 64 / 1024, SPM_RD_LAT 1 / 4, RD_OUTSTANDING
  1; all MVM latencies 1; ACT 4 / MAC 8 / MUL 8 / ADD 8 / WS_RD 4): 79 runs, all clean.

### 13.2 Core-level results

All runs were bit-exact against the simulator: DRAM, the whole SPM, the PERF counters, the per-source read
beats at the port and quiescence at `done`. The core with the MVM unit replayed every trace of `traces/m2`,
`traces/gen`, `traces/m3` and `traces/model` (Qwen2.5-0.5B Q8_0 and Q4_K_M-requant, 1 token), 1,402 further
traces at T_MAX 1 and at T_MAX 8, and random programs, under four memory regimes (rand, fast, slow, ordered),
at the default and at the U200 parameters, on 1, 2, 3, 4 and 16 channels, and with the implementation
parameters at their extremes (WS FIFO 64 / 128 / 1024, pipes 0 / 3, SPM_RD_LAT 1 / 3 / 4, RD_OUTSTANDING 1 /
2, all MVM latencies 1, ACT_RD 4 / MAC 8 / MUL 8 / ADD 8 / WS_RD 4). 0 runs failed and nothing was skipped.
Simulation speed (one process): U200 T_MAX 1 ~160 K cycles/s, T_MAX 8 ~21 K, 16 channels ~17 K, 1 channel
~300 K.

**Throughput** (ideal memory = always ready, latency 0..1, on every channel at
once; two-point method with R and 3R rows per channel; U200):

| Shape | ROW_BEATS | Steady state | Fixed overhead per MVM |
|---|---|---|---|
| Q8_0 K 896 / 4096 / 4864, T 1 | 15 / 68 / 81 | 1.0000 cycles/beat | 85 / 139 / 153 (= 69 + T·QA_LINES) |
| Q4_K K 1024 / 2304 / 4096, T 1 | 9 / 21 / 36 | 1.0000 | 91 / 114 / 145 (= 72 + T·QA_LINES) |
| the same six shapes at T 8 (U200 T_MAX 8 build) | 15 … 81 / 9 … 36 | 1.0000 (Q4_K K 1024: 0.9994) | Q8_0 223 / 671 / 783, Q4_K 252.5 / 436 / 684 (≈ 95–100 + T·QA_LINES) |

M3 gate shapes (Qwen2.5-0.5B Q8_0, one token; cycles accept → done = PERF_MVM_CYCLES; the
simulator's cycle model is 64 + T·QA_LINES + max beats):

| MVM | Weight beats / channel | Cycles | Cycles / beat | Simulator model | at 300 MHz |
|---|---|---|---|---|---|
| K 896 × N 896 (q, o) | 3,360 | 3,445 | 1.025 | 3,440 | 11.5 µs |
| K 896 × N 4864 (gate, up) | 18,240 | 18,325 | 1.005 | 18,320 | 61.1 µs |
| K 4864 × N 896 (down) | 18,144 | 18,297 | 1.008 | 18,294 | 61.0 µs |
| K 896 × N 151936 (output head) | 569,760 | 569,845 | 1.0001 | 569,840 | 1.90 ms |

At the U200 bitstream parameters (§14.2: SPM_RD_LAT 4, 2-stage broadcast / result / job / weight
pipes, MAC_LAT 9, RST_PIPE 3; Xilinx filelist, 2026-10-24) the
steady state stays 1.0000 cycles per beat for all six shapes; the fixed overhead per MVM is Q8_0
K 896 / 4096 / 4864 99 / 151 / 165 cycles and Q4_K K 1024 / 2304 / 4096 104 / 126 / 157 (the
default-parameter build of the same run: 88 / 140 / 154 and 94.5 / 115 / 146), the gate shapes
3,459 / 18,339 / 18,309 cycles and the output head 569,860 (default build: 3,448 / 18,328 /
18,298 / 569,848): 11–12 cycles more per MVM, 0.3 % on the K 896 × N 896 MVM.

The whole 1-token Qwen2.5-0.5B Q8_0 trace set (145 traces) spends 2.08 M MVM cycles (2.27 M
cycles in total incl. LD / QACT; 7.6 ms at 300 MHz) under the fast regime (latency 0..2), 3.03 M
MVM cycles under rand and 7.73 M under slow; the output head alone 569,845 / 822,464 / 2,103,586.

**PERF**: `PERF_MVM_CYCLES` equals, for every MVM, the cycles from the one after the
dispatch handshake through `mvm_done` counted at the port by the testbench; N = 0 MVMs (also with
ACC) are held 1 cycle with no SPM or weight traffic; a faulting MVM counts 0; per MVM the weight
beats per channel are `rows_c × ROW_BEATS`, the SPM reads `T × QA_LINES` (+ one per gather job
with ACC) and the SPM writes exactly one per gather job (`mvm_model.gather_jobs`); no MVM SPM /
weight traffic and no `mvm_done` outside a held MVM; PERF_CH_RD_BEATS[c] == port beats (LD + WS).

**Chains**: MVM → ST of y, MVM → MVM ACC, x = the previous MVM's y as raw QA, LD of raw bytes → MVM, MVM →
QACT of y, rotating regimes, whole SPM compared.

**Reset**: at the U200 bitstream parameters with RST_PIPE 3, hardware reset with the RST_PIPE contract of
rtl-m2 P10 (12 resets, 11 in a run, 4 with the minimum 4-cycle pulse) and inside MVM programs with weight
reads in flight (24 resets, 21 while busy, 15 inside an MVM): the CSRs, PERF and the port are in their reset
state and the next program is exact (24 / 24); 10 / 10 core-level tests passed, 184 s (2026-10-24).

**Integration findings**: no functional RTL bug was found at core level, so no MVM block file was
changed by the integration step.

### 13.3 Mutation campaign

31 mutants of the MVM unit and its wiring in `tsr_core` (scratch copies of `rtl/`) were run against the
core-level replays and the directed tests. **28 were killed, 3 survived; the 3 survivors are the three
mutants marked equivalent in advance**: `cmd_ready` masked in the done cycle (it is already 0 there, I1), WS
`rd_rsp_ready` tied 1 (the streamer's ready is constant 1, §4, I2 asserts it) and a broadcast pipe one stage
deeper than `MVM_BC_PIPE` (equivalent by contract, since implementation parameters change timing only; the
parameter values are exercised end to end by the variants, but no test checks that the requested depth
arrives).

The mutants covered the weight-stream request channel, length and over-fetch, the weight response data, the
PERF input, the SPM client of the MVM and its read latency, the broadcast stride and `last`, the gather's
expected clusters, `perf_mvm_busy` and `done` timing, `rows_c` off by one, the activation gate, the Q8_0 and
Q4_K term order and partial-lane assignment, the `msum` width, the QS half select, the tail bytes and byte
enables, the ACC read-modify-write and the dispatched rows per channel. One mutant (Q8_0 absent block b+1
used) is killed only by tests with raw weight windows, because the repacked rows of the traces hold zeros
there, so the term is +0. Every mutant whose first failure involved an RTL assertion (`perf_mvm_busy` from
the accept cycle, `done` before the last SPM write, the SPM response client, the SPM read latency, the ACC
old-line pairing) was rerun with every simulation check removed (`define SYNTHESIS` in the scratch copy): all
5 are still killed by the testbenches.

### 13.4 Part C: the part B open items, the end-to-end check, what remains

Where part C closed the open items of part B (they were: whole-core Vivado timing with four
clusters, resources and the URAM / BRAM mapping, the U200 platform adapter and packaging, and the
cost of simulating the large configurations):

| Part B item | Part C result |
|---|---|
| Vivado timing of the whole core | §14: 300 MHz out of context for k1 (WNS +0.100 ns), k4 T_MAX 1 (+0.109) and k4 T_MAX 8 (+0.022, floorplan C') with the final RTL (r3, §14.9; r2: +0.121 / +0.108 / +0.015), with the U200 parameters of §14.2 (SPM_RD_LAT 4, MVM_BC_PIPE 2, MVM_RES_PIPE 2, MVM_MAC_LAT 9, RST_PIPE 3); the RTL changes are bit-exact (§14.7, §14.9) |
| Resources, URAM / BRAM mapping | measured (§14.3, §14.6); `rtl/tech/xilinx` (`rtl/filelist_xilinx.f`): URAM SPM banks, BRAM activation buffers / weight FIFOs / line buffers, LUTRAM for tiny sizes, DSP-packed MAC lanes (two tokens per DSP48E2), checked against the generic RTL by `tb/tech` |
| U200 platform adapter, kernel packaging | `rtl/platform/alveo_xrt`, `docs/platform-u200.md`: ap_ctrl_hs control bridge with the core CSR window and ADAPTER_STATUS, AXI4 bridges, reset distribution, kernel tops `tsr_u200_k1` / `tsr_u200_k4`, `.xo` packaging (integrity checks passed), `v++` configurations (k1 linked at 300 MHz; the k4 link ran later and closed at 276.3 MHz: §14.8) |
| M0 bandwidth kernel | `tsr_bw` (`docs/m0-bw.md`) |
| M3 gate (Qwen2.5-0.5B Q8_0 on the card) | passed later on the card: k1 (1 channel, 300 MHz), then k4 (4 channels, 276.3 MHz, 54.7 tok/s); [`platform-u200.md`](platform-u200.md) §14. At the time of part C: not run, with the pre-hardware check below |
| Simulation cost | unchanged per configuration; the tile-only replays (planned, W5) carry the large runs |

**End to end before the card** (`docs/platform-u200.md` §13): the host software ran against the Verilated
kernel, the adapter RTL around the core at the U200 bitstream parameters of §14.2 on
`rtl/filelist_xilinx.f`, with the checks of the kernel bench (AXI4 / AXI4-Lite, memory rules, RTL
assertions, quiescence) applied to every cycle. Results (2026-10-24; every session 0 findings):
- Qwen2.5-0.5B Q8_0, "The history of the printing press" (6 tokens) + 3 greedy tokens, logits at
  8 positions: the 4-channel / T_MAX 1 kernel and the k1 configuration (1 channel) produce logits and
  tokens **bit-identical** to `TSR_DEVICE=sim` (`klcheck compare --expect-identical` 8 / 8 rows; tokens +
  logits sha256 equal), and so does the T_MAX 8 configuration (k4), whose prefill ran as T = 6 MVMs:
  3,141,698 cycles for the 6 prompt tokens against 13,669,379 at T_MAX 1 (MVM cycles 2,108,260 against
  12,504,944).
- Cycles from `PERF_*` per decoded token (169 MVMs): 2,281,377 at 4 channels / T_MAX 1
  (PERF_MVM_CYCLES 2,084,158; 7.60 ms at 300 MHz; part B measured 2.08 M MVM cycles for the same
  token on the bare core at the default parameters) and 8,476,150 for k1 (28.25 ms). Simulation wall time
  20.2 s (4 channels) and 28.3 s (k1) per decoded token.

**Regression wiring**: `rtl/lint.sh` also runs `rtl/tech/xilinx/lint.sh`,
`rtl/platform/alveo_xrt/lint.sh` and `rtl/platform/alveo_xrt/bw/lint.sh` (205 lint runs and the two
generated-file checks, all clean, 3 min). `tb/run_all.sh` collects `tb/tech`, `tb/platform` and
`tb/platform/bw`. After the margin changes (§14.9; 2026-10-24, every build cold) `rtl/lint.sh` had 213 lint
runs and 4 XSim elaborations, all clean (239 s).

**What remained after part C** (hardware; all of it was done afterwards, see [`platform-u200.md`](platform-u200.md)
§14 for the card results):
- The kernel tops carried §14.2's `MVM_MAC_LAT` 9 and `RST_PIPE` 3 (`gen_tops.py` `COMMON`; `.xo`
  repackaged 2026-10-24 07:15, k1's again at 21:35 for its relink; k4's predates §14.9 and
  `build_xclbin.sh` repackages it); the packaged k1 top replays the 0.5B model bit-identically
  (`docs/platform-u200.md` §13.1, and again on the r3 RTL, §14.9). Floorplan C' is implemented for
  the kernel in `hw/u200/k4_floorplan.tcl` (§14.8).
- The k1 hardware link met 300 MHz: first WNS +0.011 ns, then, relinked on the r3 RTL (§14.9),
  kernel clock WNS +0.060 ns with +0.075 ns on the worst kernel path (§14.8). The k4 link was not yet
  run: its memory was estimated at 25–27 GB against the dev server's 14 GB; its out-of-context margin is
  +0.022 ns (r3). It ran later (3 h 51 min with swap): routed WNS -0.285 ns at 300 MHz, 809 failing endpoints,
  and `v++` scaled the kernel clock to **276.3 MHz** (measured in Vivado 2024.2, linked; §14.8).
- On the card: DMA, interrupts, the XRT scheduler path, the per-exec host cost (21 register writes
  and 6 reads per exec in the scheduler sequence, 145 execs per decoded token of the 0.5B model) and
  the channel capacity of the 64 GB card (`docs/spec.md` §20, Open items). Measured later on the card:
  16 GiB per channel without aliasing ([`m0-bw.md`](m0-bw.md) §7.1), 35-36 µs per exec
  ([`platform-u200.md`](platform-u200.md) §14.2).

---

## 14. Vivado results (part C)

Whole-core out-of-context (OOC) runs of `tsr_core` on the U200 part with the Xilinx target
(`rtl/filelist_xilinx.f`: URAM / BRAM / DSP-packed tech wrappers). Every number below is a
Vivado 2024.2 measurement of the core's RTL unless it is marked as an estimate (provenance: measured in
Vivado 2024.2, out of context; linked results are labelled linked). The out-of-context runs do not include the
shell, the AXI bridges, the DDR4 controllers and the memory subsystem; the linked designs have them (§14.8).
The bitstreams built from this RTL ran on the card (k1 and k4; [`platform-u200.md`](platform-u200.md) §14). The
U200 development platform is extracted (not installed); its `platforminfo` gives the DDR bank SLRs and the
per-SLR dynamic region used in §14.5. The control core, issue path, vector unit and attention modes are
not in these numbers: their estimates are CC §14.1 and §14.5, and their floorplan is called C'' (CC §14.2).

### 14.1 Flow

- Scripts (tb/core/viv): `core_ooc.tcl` (flow: synthesis, or `from=<post_synth.dcp>`, then
  opt / place / phys_opt / route / optional post-route phys_opt; `dcp`, `floorplan=`, `place=` /
  `phys=` / `route=` directives, `prphys`, `unc=`, `filelist=` (default the Xilinx filelist),
  `NAME=VALUE` generics with the last one winning), `core_rpt.tcl` (reports, also standalone on a
  checkpoint), `core_slr.tcl` (SLR floorplan, below), `core_ooc.sh <k1|k4t1|k4> <outdir> [args]`
  (the three U200 configurations with the bitstream parameters, `TSR_VIVADO_LOCK` for a shared
  host). Summary lines: `grep COOC <outdir>/vivado.log`.
- Part xcu200-fsgd2104-2-e, `general.maxThreads 4`, default synthesis (flatten rebuilt), default
  opt / phys_opt / route directives; the floorplanned 4-channel runs place with
  `AltSpreadLogic_high` and run post-route `phys_opt_design`. Clock 3.333 ns (300 MHz), every
  I/O budgeted 1.6 ns outside (the platform side is registered), no clock uncertainty beyond
  Vivado's default (the platform MMCM's jitter is not modelled), ports unplaced.
- Dev server (14 GB RAM, swap): 4-channel T_MAX 8 synthesis 8 min / 9.6 GB; implementation
  40 min wall, Vivado peak 13.5 GB (with congestion, the stopped run below: > 3.5 h of routing);
  the r3 run (§14.9): synthesis 7 min 12 s / 10.4 GB, implementation 35.5 min / 14.2 GB.

### 14.2 U200 bitstream parameters (result of this section)

| tsr_core parameter | U200 | Generic default | Why |
|---|---|---|---|
| `SPM_RD_LAT` | 4 | 2 | URAM output registers (rtl-m2 §12, xtech) |
| `MVM_BC_PIPE` | 2 | 1 | broadcast and weight-response pipes: one register on each side of an SLR crossing |
| `MVM_RES_PIPE` | 2 | 1 | result, weight-request and gather job / start pipes: the same |
| `MVM_MAC_LAT` | 9 | 3 | a register after every adder level of the DSP-packed MAC lane (5 and 7 missed, §14.4) |
| `RST_PIPE` | 3 | 0 | reset register stages with per-block copies (new, §14.4) |

All other parameters keep their `tsr_pkg` defaults (CH_BYTES 16 GiB, WS FIFO 512, ACT_RD_LAT 2,
MUL_LAT 4, ADD_LAT 5, WS_RD_LAT 2). Every value is an implementation parameter: results are
identical (bit-exact replays below), only latencies change.

### 14.3 Runs

`k1` = NUM_CH 1, T_MAX 1 (bring-up kernel), `k4t1` = NUM_CH 4, T_MAX 1, `k4` = NUM_CH 4,
T_MAX 8 (target kernel). RTL revisions: r0 = M3 part B + rtl/tech/xilinx; r1 = first fix set; r2 = the
fixes of §14.4 (the RTL of the first k1 / M0 links); r3 = final, r2 + the margin changes of §14.9.
Floorplans (§14.5): none, B (clusters 0 / 2 in SLR0, 1 in SLR1, 3 in SLR2, arbiter c
next to DDR[c]), C (clusters 0 / 2 in SLR0, 1 / 3 in SLR2), C' (C with all arbiters and the rest of
the core in SLR1 and the end stages of every crossing pipe pinned); in B, C and C' the SLR1 pblock
is the dynamic-region stand-in CLOCKREGION_X0Y5:X2Y9.

| Run | RTL, MAC_LAT / RST_PIPE | Floorplan | WNS / WHS (ns) | TNS (ns) / failing endpoints | CLB LUT | FF | DSP | RAMB36 | URAM |
|---|---|---|---|---|---|---|---|---|---|
| k1 | r0, 5 / – | none | +0.051 / +0.030 | 0 / 0 | 30,287 | 38,032 | 182 | 40 | 128 |
| k4t1 | r0, 5 / – | synthesis only | (est. +0.734) | – | 62,812 | 85,010 | 632 | 136 | 128 |
| k4 | r0, 5 / – | none | **−0.877** / +0.012 | −10,273 / 46,606 | 269,314 | 277,040 | 2,728 | 808 | 128 |
| k4 | r1, 7 / 2 | B | post-place +0.114; router −0.787 after its first iteration, routing congestion level 5–6 in the gather; stopped after 3.5 h of routing | – | – | – | – | – | – |
| k4 | r1, 7 / 2 | C | **−0.159** / +0.021 | −27.0 / 727 | 266,467 | 309,875 | 2,728 | 808 | 128 |
| k4 | r2, 9 / 3 | C' | +0.015 / +0.020 | 0 / 0 | 270,910 | 345,578 | 2,728 | 808 | 128 |
| k1 | r2, 9 / 3 | none | +0.121 / +0.024 | 0 / 0 | 30,890 | 43,154 | 182 | 40 | 128 |
| k4t1 | r2, 9 / 3 | C' | +0.108 / +0.005 | 0 / 0 | 67,071 | 112,192 | 632 | 136 | 128 |
| **k4** | **r3, 9 / 3** | **C'** | **+0.022 / +0.019** | **0 / 0** | 272,025 | 349,199 | 2,728 | 808 | 128 |
| **k1** | **r3, 9 / 3** | none | **+0.100 / +0.009** | **0 / 0** | 30,902 | 44,516 | 182 | 40 | 128 |
| **k4t1** | **r3, 9 / 3** | C' | **+0.109 / +0.012** | **0 / 0** | 67,345 | 113,748 | 632 | 136 | 128 |

So 300 MHz closes out of context for all three configurations with the r2 and the final (r3) RTL
and parameters; the measured Fmax of the earlier k4 states was 237 MHz (r0, no floorplan) and
286 MHz (r1, C). Worst paths of the r2 runs (r3: §14.9):
- k4 (+0.015): SPM write-port bank select -> URAM cascade enable (4 levels, tsr_spm, as the xtech
  OOC run of tsr_spm found); then broadcast pipe last stage -> cluster command register (0
  levels, SLR crossing landing, +0.027); activation write data -> BRAM (+0.030); activation
  read address -> BRAM (+0.033); QACT multiplier operand (+0.035); result pipe -> assembler buffer
  (+0.036); back-end control fan-out (tree level-1 operand select, accumulate clock enables,
  +0.05..0.10); assembler -> job pipe slice (+0.089). No MAC-tree, gather-select or reset path
  is below +0.08 ns; the only SLR-crossing path below +0.1 ns is the broadcast landing above.
- k1 (+0.121): SPM write / read address -> URAM cascade (3 levels); activation RAM enables.
- k4t1 (+0.108): dispatcher state (the queue head's command register, CC §9.1) -> DMA / MVM command register clock enables (0 levels); SPM
  write grant -> SPM input register; SPM read address -> URAM cascade; per-part reset copies
  (+0.19..0.30; in this run Vivado drove cluster 3's resets from a replica of the gather's copy,
  i.e. two `keep` copies ended up merged).

### 14.4 What failed and the fixes (all bit-exact; only latencies change)

The first k4 run (r0, no floorplan) failed in 46,606 endpoints. Grouped by structure: result
pipe -> gather buffers (13.8 K endpoints, −0.76 ns, D and CE), rst_n -> back-end resets (5.6 K,
−0.58), activation write data -> BRAMs (9.0 K, −0.74), weight register -> MAC DSP B ports (2.2 K,
−0.73), MAC tree -> integer-scale DSP A ports (2.0 K, −0.81), gather output register clock enable
(1.9 K, −0.68), gather job select -> write queue (−0.88). The unconstrained placer put clusters
0, 1, 2, the gather and the rest of the core into SLR1 (and SLR1's DSP columns overflowed into
SLR0), so congestion inflated every route. Causes and fixes:

1. **Broadcast pipes merged by synthesis.** The four per-cluster broadcast `tsr_vpipe`s have one
   source; Vivado merged them into one pipe (and with it every cluster's `aw_q` / `fmt_q` /
   activation write register): clusters 1–3 had no copy of their own and cluster 0's registers
   drove all four clusters across three SLRs (1,560 SLL nets from one cluster's `wd_q`). Fix:
   `tsr_vpipe` parameter `KEEP_REGS` (named `KEEP` in these runs, §14.8; Xilinx `keep` +
   `shreg_extract = "no"` on the stage registers), used for the broadcast and weight-response
   pipes (`tsr_mvm`) and the gather's start pipes;
   measured: 4 separate broadcast pipes after the fix (pipes 8.9 K FF instead of 5.7 K).
2. **Gather job select.** The job data was one NUM_CH × 2 × T_MAX : 1 (64:1 at k4) select of the
   assembly buffers whose select lines fanned out to every buffer flop: −0.88 ns, and in
   floorplan B the router's congested nets were exactly these (`buf_q`, the select). First fix
   (r1): registered selects (`req_q`, `jb_q`, `jy_q` per cluster, cycle-identical) — still
   −0.16 in C. Final fix (r2): `tsr_mvm_gather` split into one assembler per cluster
   (`tsr_mvm_gasm`, the buffers and everything that grows with T_MAX; a job is a 2 × T_MAX : 1
   select of flops with registered selects) and a central part (round robin over NUM_CH job-pipe
   heads, credits, ACC, write queue), with `JOB_PIPE` register slices in between (`tsr_mvm`
   passes `RES_PIPE`) and a kept start pipe per assembler. The floorplan puts each assembler next
   to its cluster, so only assembled jobs cross to the centre. Port list unchanged; with
   `JOB_PIPE = 0` the cycle behaviour at the ports is the old one (tb/mvm `gather_perf` latencies
   unchanged).
3. **Reset.** One `rst_n` net to every reset flop of a core spread over three SLRs (−0.58 at the
   port with its 1.6 ns budget; the platform's reset copy is also a single register). Fix: new
   `tsr_core` parameter `RST_PIPE` (0..8, default 0 = no extra flops): with P ≥ 1 every flop sees
   `rst_n` exactly P cycles late, through P − min(P, 2) shared stages and min(P, 2) kept stages
   per block (CSR, DMA, QACT, SPM, each arbiter; inside `tsr_mvm`, new parameter
   `RST_LOCAL`: top, gather and every cluster with its pipes). The first per-block copy can sit
   at the SLR boundary, the second (replicated by phys_opt) at the loads. The core is a P-cycle
   delayed copy of the `RST_PIPE = 0` core; the registers start in reset (INIT 0). Boundary
   consequence (§14.7, §14.8): the outputs go quiet P cycles after `rst_n` falls, and `rst_n`
   must stay low for at least P + 1 cycles.
4. **MAC tree -> integer-scale DSP.** Vivado absorbs the MAC lane's output register into the
   integer-scale multiplier's AREG, so the last tree segment ends at a DSP A port: MAC_LAT 5
   (two adder levels + three levels) −0.81 (r0), MAC_LAT 7 (IN AD M P | 1 | 2 | 2) −0.07 in C;
   MAC_LAT 9 (a register after every adder level) has no MAC path among the worst 30 of the final
   k4 run. Cost: +5.9 K FF per T_MAX 8 cluster over MAC_LAT 7 (MAC lanes 17.7 K FF per cluster).
5. **Write registers with large clock enables.** The cluster's activation write register `aw_q`
   (527 flops) and the actbuf's address / data registers loaded only with a valid line; the
   gather's output register (592 flops) loaded on a queue-pointer compare. After the fix they load every
   cycle (every use is qualified by the registered valid / write enables; `out_q` holds the
   same entry whenever it is valid).
6. **SLR crossings with logic on both sides.** In C, the DMA's handshake with the arbiters of
   channels 0 / 3 (in SLR0 / SLR2) crossed with 6–8 levels on the DMA side, and the cluster's
   result FIFO read crossed into the first result-pipe slice. Fix: weight-stream pipes in
   `tsr_mvm` per cluster (requests: `tsr_rpipe` of `RES_PIPE` slices; responses: kept `tsr_vpipe`
   of `BC_PIPE` stages, legal because the streamer's `rd_rsp_ready` is constant 1, asserted since),
   so all four arbiters stay with the DMA in SLR1 and only pipes cross; the result pipe runs
   cluster -> assembler inside one SLR. The floorplan pins the first and last stage of every
   crossing pipe to the source / destination SLR (48 stages at k4), so each crossing is one
   register-to-register hop.

Results of the fixes, measured (k4, C'): 7,010 SLLs used (3,481 SLR0–SLR1, 3,529 SLR1–SLR2, of
17,280 per boundary), WNS +0.015 ns. The r1 fixes alone (1, 2 first step, 3 with RST_PIPE 2, 4
with MAC_LAT 7, 5) gave +0.114 post-place in B but did not route (congestion), and −0.159 routed
in C.

### 14.5 U200 budget and floorplan

Device (measured from the part database): xcu200 = VU9P, three SLRs of 394,080 CLB LUTs,
788,160 FFs, 720 RAMB36, 320 URAM, 2,280 DSP each; 17,280 SLLs per SLR boundary. Platform
`xilinx_u200_gen3x16_xdma_2_202110_1`, from `platforminfo` of the extracted development platform
(2026-10-23, `build/platforms/vpp-smoke/platforminfo.txt`; they match the AMD UG1120 figures
recorded at the platform research):
- DDR4 banks: bank0 (DDR[0]) SLR0, bank1 (DDR[1], the default bank) SLR1, bank2 (DDR[2]) SLR1,
  bank3 (DDR[3]) SLR2; PLRAM in every SLR; the host-memory bank in SLR2.
- Dynamic region per SLR (LUT / FF / RAMB36 / URAM / DSP): SLR0 388,160 / 776,320 / 720 / 320 /
  2,280, SLR1 205,440 / 410,880 / 420 / 160 / 1,320, SLR2 385,440 / 770,880 / 720 / 320 / 2,280;
  in total 979,040 LUT, 1,860 RAMB36, 800 URAM, 5,880 DSP; kernel clocks 300 MHz (default) and
  500 MHz.

Clock-region columns X0–X2 of SLR1 hold exactly SLR1's dynamic-region counts (205,440 LUT / 1,320
DSP / 160 URAM / 420 RAMB36), so these runs model the shell (static region) as the rest of SLR1
(pblock CLOCKREGION_X0Y5:X2Y9 for everything placed in SLR1). The platform's own
`pblock_dynamic_SLR1` (`hw.xsa` `impl.xdc`) is exactly these clock regions; with its constraints
(`SNAPPING_MODE ON`) it counts 199,680 LUT sites instead of 205,440 (cause not investigated; C' uses
25.5 K LUT there). The DDR4 controllers / memory interconnect that `v++` adds in the dynamic region
for DDR[0], DDR[2], DDR[3] are not modelled; the M0 link measures them at about 30.5 K LUT, 42.5 K
FF, 61 RAMB36 and 3 DSP per bank in its SLR (`docs/platform-u200.md` §8.2).

**Clusters in their DDR SLRs (DDR0 SLR0, DDR1 / DDR2 SLR1, DDR3 SLR2, as `platforminfo` records
them) does not fit**: two T_MAX 8
clusters in SLR1 need 2 × 674 + 32 (QACT) = 1,380 DSP of 1,320 and 2 × 200 + 8 = 408 of 420
RAMB36 (measured cluster sizes). Floorplan C' (measured, routed, 300 MHz met; r2 run, r3 in
§14.9):

| SLR | Content | CLB LUT | FF | RAMB36 | URAM | DSP |
|---|---|---|---|---|---|---|
| SLR0 | clusters 0, 2 + their assemblers | 122,929 (31 %) | 152,226 | 400 (56 %) | 0 | 1,348 (59 %) |
| SLR1 (X0–X2 only) | SPM, central gather, DMA, QACT, CSR, 4 arbiters, dispatch | 25,068 (12 % of the region) | 41,719 | 8 | 128 (80 % of the region) | 32 |
| SLR2 | clusters 1, 3 + their assemblers | 122,913 (31 %) | 151,633 | 400 (56 %) | 0 | 1,348 (59 %) |

Signals crossing an SLR boundary (C', measured: 7,006 nets crossing, 7,010 SLLs; r3: 8,240,
§14.9): per cluster
the broadcast pipe (529 bits) and the weight-stream response pipe (516 bits), 3,578 nets for the
four clusters; the gather job pipe (561 bits, 2,165 nets) and start pipe (68 bits, 272 nets); the
weight-stream request pipe (37 bits, 140 nets); the reset copies (one per part); and a few
hundred landing / handshake nets (e.g. 574 nets of two clusters' activation write register,
which the implementation moved to the SLR1 side of the crossing). The CSR port and the memory
ports (arbiters) are in SLR1; every cluster gets its weights from its arbiter there. The AXI side
of each channel (core memory port c -> adapter bridge -> DDR[c]) is outside the core: channels 0
and 3 cross between SLR1 and their DDR SLR in the adapter's AXI slices (`AXI_SLICES 2`),
channels 1 and 2 stay in SLR1. `hw/u200/k4_floorplan.tcl` implements C' for the kernel hierarchy on
the platform's own pblocks, with bridges 0 / 3 crossing in their AXI slices (`docs/platform-u200.md`
§8.1).

### 14.6 T_MAX 1 vs T_MAX 8

| (synthesis, final RTL r3) | k4t1 | k4 | Ratio |
|---|---|---|---|
| CLB LUT | 68,334 | 275,657 | 4.0 |
| FF | 113,671 | 348,978 | 3.1 |
| DSP | 632 | 2,728 | 4.3 |
| RAMB36 | 136 | 808 | 5.9 |
| URAM | 128 | 128 | 1 |

(r2: 68,082 / 274,248 CLB LUT and 112,185 / 345,379 FF, the same ratios.) Per cluster, T_MAX 1 -> 8
adds 7 × the per-token back end (5.3 K LUT, 4.7 K FF, 20 DSP: 12 integer-scale products, 8 for 4
`tsr_fp_mul`), the packed MAC lanes (128 -> 512 DSP: two tokens per DSP; 2.3 K -> 18.0 K LUT of
adder trees), 7 × 24 RAMB36 of activation buffer and the larger assembler (2 × 8 × 512 buffer
flops). Timing: both close at 300 MHz with the same parameters and floorplan (r3: k4t1 +0.109 ns, k4
+0.022 ns; r2 +0.108 / +0.015); T_MAX 8 has the smaller margin. Share of the dynamic region
(`platforminfo` figures of §14.5) used by k4: 28 % LUT, 46 % DSP, 16 % URAM, 43 % RAMB36 (808 of
1,860).

Recommendation: build the k4 bitstream with T_MAX 8: the simulator's cycle model gives 5.9×
(Qwen2.5-0.5B) / 7.2× (32B) faster prefill (T_MAX 8 software check), for 4.0× LUT, 4.3× DSP and
5.9× RAMB36, and it closes 300 MHz here with the parameters of §14.2 and floorplan C' (clusters 0
/ 2 in SLR0, 1 / 3 in SLR2, the rest and the arbiters in SLR1). T_MAX 1 uses the same RTL and
parameters; k1 stays a single-SLR design (SLR1, DDR[1]).

### 14.7 Verification of the RTL changes

The replays named here are RTL simulation; the block benches (`tb/mvm`, `tb/spm`, `tb/tech`, `tb/mvmback`) are in
the tree.

- Lint: `rtl/lint.sh` 82 runs clean (the new `tsr_mvm_gasm` is linted as its own top),
  `rtl/tech/xilinx/lint.sh` clean; extra Verilator lint of `tsr_core` (both filelists) with
  RST_PIPE 0 / 1 / 2 / 3 / 5 and HAS_MVM 0 / 1, of `tsr_mvm` with RST_LOCAL 1 / 2 and pipes 0 / 3,
  and of the gather / assembler with JOB_PIPE 0..3, NUM_CH 1 / 3 / 4 / 16, T_MAX 1 / 3 / 8.
- Block tests: `python3 -m pytest tb/mvm` 14 passed (gather in 4 configurations incl. the perf
  latencies and the no-combinational-path test; tsr_mvm with real clusters).
- Replays (every trace of traces/m2, gen, m3, model in all four memory regimes):
  default parameters (generic filelist) 778 traces / 3,112 runs, 0 failed; U200 parameters on
  `rtl/filelist_xilinx.f` (SPM_RD_LAT 4, BC_PIPE 2, RES_PIPE 2, MAC_LAT 9, RST_PIPE 3) 3,112 runs,
  0 failed; the same with `--reset-each` (hardware reset before every trace) on traces/m3 + gen
  1,264 runs, 0 failed. The r1 state was replayed the same way (MAC_LAT 7, RST_PIPE 2: 3,112 +
  1,264 runs, 0 failed).
- Core-level directed tests (RST_PIPE 0): the four configurations (2 channels T_MAX 4, 3 channels T_MAX 3,
  U200 T_MAX 1 and 8) pass (24 tests: CSR map / consts, hardware and soft reset, throughput including the M3
  gate shapes and the output head, PERF, MVM chains, random programs). With RST_PIPE 3 (scratch copy with the
  default changed) the 2-channel set fails only the "outputs quiet from the first reset cycle" check of the
  hardware-reset tests (expected, §14.4 item 3); with that check starting at reset cycle P and reset pulses of
  ≥ P + 1 cycles (scratch test copy) the whole set passes. The reset tests followed this rule afterwards
  (quiet check from reset cycle RST_PIPE, pulses of at least RST_PIPE + 1 cycles) and ran the bitstream
  parameters with RST_PIPE 3 (10 / 10 passed, §13.2).
- Throughput (U200 build at the default implementation parameters,
  ideal memory): steady state unchanged (1.0000 cycles per weight beat; Q4_K K 1024 0.9994); the
  fixed overhead per MVM grew by 1–3.5 cycles (job pipe and weight-stream pipes of one stage
  each): Q8_0 K 896 / 4096 / 4864 88 / 140 / 154 cycles (part B: 85 / 139 / 153), Q4_K K 1024 /
  2304 / 4096 94.5 / 115 / 146 (91 / 114 / 145); M3 gate shapes 3,448 / 18,328 / 18,298 cycles
  and the output head 569,848 (part B 3,445 / 18,325 / 18,297 / 569,845). At the U200 parameters
  (Xilinx filelist, 2026-10-24): steady state 1.0000 cycles
  per beat, 11–12 fixed cycles per MVM more than the default build (Q8_0 K 896 / 4096 / 4864:
  99 / 151 / 165; gate shapes 3,459 / 18,339 / 18,309, output head 569,860; §13.2).
- XSim (Vivado 2024.2 `xvlog` / `xelab`, the hw_emu simulator), 2026-10-24: the reset-stage
  registers of RST_PIPE / RST_LOCAL (`tsr_core` `g_sh.sh_q` / `g_rst.rb_q`, `tsr_mvm` `g_rst.rt_q` /
  `rg_q` / `rc_q`) load in a plain `always @(posedge clk)` next to their `initial ... = '0`
  (INIT 0). With `always_ff`, the initial block is a second writer, which IEEE 1800 9.2.2.4
  forbids, and `xelab` failed (VRFC 10-3818) for every RST_PIPE ≥ 1 / RST_LOCAL ≥ 1 configuration;
  Verilator and Vivado synthesis accept it. After the change `tsr_core` elaborates in XSim at RST_PIPE 0 / 1 / 2
  / 3 / 8 (Xilinx filelist, U200 parameters, k1 and k4 configurations), `tsr_mvm` at RST_LOCAL 1 /
  2, and the packaged `tsr_u200_k1` / `tsr_u200_k4` tops elaborate. A 2-channel RST_PIPE 3 core
  simulated in XSim asserts and releases every reset copy exactly 3 cycles after `rst_n` and
  answers CSR ID / CFG0. Replays after the change: default parameters 3,112 runs, U200 parameters
  on `rtl/filelist_xilinx.f` 3,112 runs, `--reset-each` 1,264 runs, 0 failed.
- Mutants (scratch copies, simulation checks off): the r1 gather without the `jy_q` forwarding
  and the r2 assembler with `last` on every group's final job, without the lane mask, or active
  without rows are killed by tb/mvm `gather_random` / done / `res_ready while idle` checks (the
  last survives only the 1-channel configuration, where every MVM has rows).

### 14.8 Links, not run, open

- **The card.** The k1 and k4 xclbins built from this RTL ran on the Alveo U200 ([`platform-u200.md`](platform-u200.md)
  §14, [`m0-bw.md`](m0-bw.md)). **The k4 link ran** (floorplan C', 3 h 51 min on the 14 GB dev host with up to 28 GB
  of swap): routed WNS **-0.285 ns** at 300 MHz, 809 failing endpoints, TNS -56.5 ns, so `v++` scaled the kernel
  clock to **276.3 MHz** (`AUTO-FREQ-SCALING-04`); the worst paths are in the gather's job pipes
  (`u_gather/g_asm[c].u_jp/g_slices.g_s[1]`). The projection below (-0.1 to -0.2 ns) was optimistic. On the card k4 at
  276.3 MHz ran Qwen2.5-0.5B Q8_0 at 54.7 tok/s (measured on the card, k4, 276.3 MHz). The fix proposal for
  these paths is CC §14.4. The hardware `v++` link of k1 ran on 2026-10-24 (the development platform, extracted
  not installed; `docs/platform-u200.md` §8.2): 57 min, routed WNS **+0.011 ns** (TNS 0, WHS
  +0.009 ns), the kernel clock kept at 300 MHz (auto-scaling computed 301.0 MHz); 33,276 LUT,
  49,929 FF, 40 RAMB36, 128 URAM, 182 DSP, all in SLR1. Its worst path was the one of the
  out-of-context k1 run (§14.3), the SPM read offset `g_in_reg.rd_off_s` → URAM cascade address
  `CAS_IN_ADDR_A` (3 URAM levels), with 0.110 ns less margin; next the bridge reset copy
  (+0.037 ns). **Relinked on the r3 RTL** (§14.9; 57 min 18 s with the repackaging): routed WNS
  +0.017 ns (the shell's XDMA clock), TNS 0, WHS +0.009 ns; kernel clock WNS **+0.060 ns** (shell
  logic on that clock), auto-scaling computed 305.5 MHz, 300 MHz kept. The worst kernel path is
  +0.075 ns, a gather job-pipe slice's `rdy_q` → `skid_q` clock enable (0 levels, fan-out 154),
  against +0.100 ns for the r3 k1 core out of context; the first link's paths: SPM cascade address
  +0.237 ns, reset copy → bridge counters +1.381 ns. 33,502 LUT, 51,212 FF, 40 RAMB36, 128 URAM,
  182 DSP, all in SLR1. hw_emu of k1 ran the host software (`docs/xrt-host.md` §9) on the r2 RTL;
  its xclbin was not rebuilt.
- Before the k4 link ran, its memory was estimated at 25–27 GB against the dev server's 14 GB (it needed swap, see
  above). From the r2 margin (+0.015 ns) and the 0.11 / 0.22 ns the first k1 and M0 links lost against
  their out-of-context runs it was projected at about −0.1 to −0.2 ns; the proposals of
  [`platform-u200.md`](platform-u200.md) §8.2 (SPM cascade address, `tsr_rslice` clock enable, bridge reset
  fan-out) and the back-end control copies were applied after the first links (§14.9: r3, commit `fec5dcf`). With them k4 has +0.022 ns out of context. The card runs and the k4
  link above came after that commit and measured -0.285 ns linked at 300 MHz.
- Platform facts are recorded: the DDR bank SLRs (bank0 SLR0, bank1 / bank2 SLR1, bank3 SLR2) and
  the per-SLR dynamic-region resources (`platforminfo`, §14.5), and the platform's pblocks
  (`pblock_dynamic_SLR0/1/2`; SLR1 = CLOCKREGION_X0Y5:X2Y9, the stand-in of these runs, counting
  199,680 LUT sites with the platform's constraints, §14.5).
- The kernel tops (`gen_tops.py` `COMMON`) carry §14.2's parameters (`.xo` repackaged
  2026-10-24 07:15, k1's again for its relink; k4's predates §14.9, `build_xclbin.sh` repackages
  it), and `hw/u200/k4_floorplan.tcl` implements floorplan C' on the platform's
  pblocks, applied by `hw/u200/k4.cfg`; it is validated on netlists (the k4 out-of-context
  checkpoint and a synthesized 4-channel kernel: every group found, 48 pipe stages pinned;
  `docs/platform-u200.md` §8.1), not yet placed with the shell.
- Reset copies: two of the `keep` copies ended up merged in the r2 k4t1 run (cluster 3 reset from
  a replica of the gather's copy); timing was met, but if the real floorplan needs them apart they
  want `DONT_TOUCH`. The r3 k4 run's worst path starts at a phys_opt replica of a cluster reset
  copy (`g_rst.rc_q[1][0]_replica` → cluster 0 `row_q` R, +0.022 ns, 3.112 ns of routing); the
  core's reset copies are not pinned by `core_slr.tcl` / `k4_floorplan.tcl`.
- In the v++ flow Vivado read the `tsr_vpipe` parameter named `KEEP` as the instance's KEEP
  property: 5 `[Netlist 29-72]` critical warnings in the first k1 link, harmless (every kept stage
  register keeps KEEP = yes and none is packed into an SRL; `docs/platform-u200.md` §8.2). The
  parameter is `KEEP_REGS` (same function) after the rename: the k1 relink's kernel synthesis has 0 critical
  warnings, and a `tsr_core` k1 synthesis with the clock XDC read before `synth_design` (the v++
  condition) gives exactly the 5 warnings on the r2 RTL and none on r3, with the kept registers
  still KEEP = yes (`u_bcp` 1,058, `u_wrp` 1,032, `u_sp` 134) and 0 SRL.
- Weak spots of r3 (§14.9): `tsr_rslice`'s `rdy_q` drives, in r3, every `skid_q` clock enable and
  the `main_q` select directly (the worst kernel path of both relinks, k1 +0.075 ns and M0
  +0.110 ns; out of context in k4 the arbiter response and result-pipe slices, +0.064 / +0.066 ns);
  in k4, 1,288 job-pipe stage-0 registers of assemblers 0–2 were placed in SLR1 outside their
  pblock (not LOC-fixed; r2: 75; cause not investigated); each SPM address bit drives, in r3, 32
  per-slice copies (the worst path of the `tsr_spm` run in the SLR1 pblock, +0.051 ns, inside the
  1.6 ns input budget; one copy per two slices is the untried middle option); the cluster's
  format register `fmt_q` (about 3,100 loads) stays shared, since cycle-identical per-token copies
  would have to come from `tsr_mvm_cluster`.
- `RST_PIPE ≥ 1` changes the core's reset boundary behaviour (outputs quiet P cycles after
  `rst_n` falls; `rst_n` low ≥ P + 1 cycles): in the kernel `ap_rst_n` must stay low for
  ≥ RST_PIPE + 1 = 4 cycles (`docs/platform-u200.md` §5; XRT holds it far longer, not measured on
  a card). [`rtl-m2.md`](rtl-m2.md) §11 (P10) states the contract; the core-level
  hardware-reset tests followed it (quiet check from reset cycle P, pulses of at least P + 1 cycles, a
  third of them exactly P + 1) at RST_PIPE 3 (§13.2).
- Not measured: the routed k4 at MAC_LAT 5 / 7 with the final RTL (only r0 / r1 data above);
  placement variance (one run per configuration; in the k1 runs of the SPM change, paths outside
  the SPM moved by 0.1–0.2 ns between runs); the linked k4; power.

### 14.9 Margin changes after the first links (r3, 2026-10-24)

The structures that limited the first k1 / M0 links and the r2 k4 run (§14.3, §14.8) were changed.
All changes are structural: no parameter and no latency changed, results and cycle counts are those
of r2 (the SPM's RAM inputs are the same in every cycle; a register slice's output data differs
only while it is not valid).
1. **SPM column slices** (`rtl/core/tsr_spm.sv`): each depth bank is 8 column slices of 64 bits
   (at 4 MiB 8 × `tsr_sram_1r1w` 16 K × 64, each one 4-deep URAM cascade: still 128 URAM). Every
   (bank, slice) RAM has its own `keep` copy of the input registers (read / write enable, read /
   write offset, byte enables), with the bank enables decoded before the register, so each
   cascade head is driven by a flop of its own (fan-out 1 instead of 32) with no LUT in between;
   only the write-data register stays shared per bank, and the last bank-select stage has one
   copy per slice (fan-out 64 instead of 512). The Xilinx wrapper's URAM rule
   (`rtl/tech/xilinx/tsr_sram_1r1w.sv`) is DEPTH ≥ 4096 and DEPTH × WIDTH ≥ 4096 × 64 in r3 (was
   × 72), so the 256 KiB SPM's 4 K × 64 slices stay in URAM (8 URAM, 0 RAMB36); no other RAM of
   the core has DEPTH ≥ 4096.
2. **`tsr_rslice` payload enables** and **reset copies per bridge part** (`docs/platform-u200.md`
   §8.3).
3. **Back-end control copies** (`tsr_mvm_back`): every token has its own registered copies of the
   shared accumulate / tree control (slot write enables and clear decodes, one-hot; per-lane
   bypass flags for Q4_K and Q8_0; issue tags at stages A4 / A8; level-1 selects; hold enables):
   83 `keep` flops per token. Their next values are computed once, so each copy equals the signal
   it replaces in every cycle, also in reset (a simulation assertion checks every copy).
4. **`tsr_vpipe` `KEEP` → `KEEP_REGS`** (§14.8).

Block-level runs (Vivado 2024.2, xcu200-2, 3.333 ns, I/O budget 1.6 ns, routed; one run each,
before = the r2 RTL):

| | r2 | r3 |
|---|---|---|
| `tsr_spm` 4 MiB, RD_LAT 4 (`tb/tech/viv/tech_ooc.tcl`): WNS / WHS | +0.072 / +0.059 (bank select → read data) | +0.147 / +0.045 (input port → a slice's address copy, 1 level, inside the input budget) |
| `tsr_spm`: register-to-register WNS; URAM cascade address / enable | +0.072; +0.081 (`rd_off_s` → `CAS_IN_ADDR_A`, fan-out 32, 2.16 ns of routing) | +0.442; +0.467 / +0.455 |
| `tsr_spm` placed in the SLR1 stand-in pblock: WNS | +0.076 (the cascade address, the first k1 link's path) | +0.051 (input port → address copy; register-to-register +0.288) |
| `tsr_spm`: LUT / FF / URAM | 1,160 / 1,156 / 128 | 1,152 / 2,290 / 128 |
| `tsr_mvm_cluster` T_MAX 8, MAC_LAT 9: WNS / WHS | +0.121 / +0.030 (`fmt_q` replica → accumulate operand) | +0.276 / +0.033 (weight-placement select) |
| `tsr_mvm_cluster`: largest level-1 control net (synthesis); back-end endpoints < 0.2 ns | 2,069 loads; 4 | 24 (664 copy flops, none merged, largest net 128); 0 |
| `tsr_mvm_cluster`: LUT / FF | 67,481 / 64,550 | 67,733 / 65,189 |

The SPM change alone in the k1 core run: WNS +0.121 → +0.193 ns (worst path then outside the
SPM), SPM cascade address / enable +0.565. Rejected SPM variants, measured: one address copy per
bank (`tsr_spm` +0.158, the cascade still its worst path; k1 core +0.064), `cascade_height = 2`
(+0.101, 3,264 LUT), 8 K-line banks (+0.098), `max_fanout` on the cascade heads
(register-to-register +0.284, but the generic filelist's synthesis grew to 15,183 FF).

**Whole core, r3** (the flow, parameters and floorplans of §14.1–§14.3; rows in §14.3): k1
+0.100 / +0.009 ns (134,283 endpoints), k4t1 +0.109 / +0.012 (340,288), k4 +0.022 / +0.019
(1,154,989; post-place +0.501, r2 +0.170), 0 failing endpoints in each. The structures of the
margin proposals, r2 → r3 (the same queries on the r2 and r3 routed checkpoints):
- SPM URAM inputs, worst: k1 +0.121 → +0.276 (in r3 the shared write data → `CAS_IN_DIN_B`, 3
  levels), k4t1 +0.198 → +0.364, k4 +0.015 → +0.265.
- k4 back end: clock enables +0.051 → +0.093, resets +0.103 → +0.180, data +0.050 → +0.093 (in r3 a
  5-level multiplier path, not control); largest back-end net 1,386 → 517 pins. The broadcast
  landing (r2's second, +0.027) is not among the 30 worst groups.
- New: the slices' `rdy_q` drives the payload enables directly; k4 arbiter response slice `rdy_q`
  → `skid_q` CE +0.064 (r2's worst arbiter-slice CE path +0.673), result-pipe slice +0.066.

Worst paths of the r3 runs:
- k4 (+0.022): a phys_opt replica of the cluster reset copy (`g_mvm.u_mvm/g_rst.rc_q[1][0]_replica`
  → cluster 0 `row_q` R, 0 levels, 3.112 ns of routing); activation write data → BRAM (+0.028);
  result-pipe slice `rdy_q` → `main_q` (+0.036, 1 level); result pipe → assembler buffer
  (+0.037); assembler `esel_q` → job-pipe stage 0 across an SLR (+0.041, 3 levels).
- k4t1 (+0.109): back-end QS address → activation BRAM (0 levels); job-pipe slice `rdy_q` →
  `skid_q` CE (+0.139); front-end beat register (+0.182); the worst SLR crossing, assembler →
  job pipe (+0.187).
- k1 (+0.100): QACT multiplier operand → DSP (5 levels); back-end / actbuf QS BRAM address and
  enables (+0.164 .. +0.197); SPM write grant → the per-slice byte-enable copies (+0.223).

SLR crossings (k4): 8,240 SLLs (SLR0–SLR1 4,269, SLR1–SLR2 3,971; r2 7,010) and 8,232 crossing
nets (r2 7,006). 1,288 stage-0 `main_q` registers of the job pipes of assemblers 0–2 were placed in
SLR1 although their pblock is SLR0 / SLR2 (not LOC-fixed; r2: 75), so for those bits the crossing
lies before the register (the 3-level path above); k4t1 6,940 SLLs (r2 7,005). Resources (routed,
CLB LUT / FF): k1 +12 / +1,362 (SPM FF 1,165 → 2,275), k4t1 +274 / +1,556, k4 +1,115 / +3,621;
DSP, RAMB36 and URAM unchanged. Run cost: k4 synthesis 7 min 12 s (10.4 GB), implementation
35.5 min (Vivado-reported peak 14.2 GB, 11.9 GB resident); k1 102 s + 259 s, k4t1 158 s + 609 s.

**Links on r3** (`docs/platform-u200.md` §8.2, `docs/m0-bw.md` §4.1): k1 and M0 relinked, both
300 MHz with timing met; kernel clock WNS +0.060 ns (k1) and +0.055 ns (M0), both on shell logic;
the worst kernel paths +0.075 ns (k1) and +0.110 ns (M0), both a slice's `rdy_q` fan-out.

**Verification** (RTL simulation; r3 against the r2 RTL that was HEAD before the change, replayed with the same
arguments, every result field of every run compared: cycles, PERF_*, MVM cycles, port beats, kernel fields;
0 runs differ, 0 failed):
- Replays of traces/m2 + gen + m3 + model, all regimes: default parameters 3,112 runs
  (45,331,704 simulated cycles), U200 parameters on `rtl/filelist_xilinx.f` 3,112 (45,389,109,
  equal to the r2 run in total and per group), `--reset-each` on gen + m3 1,264 (10,361,292);
  `tsr_u200_k4` on m3/tmax8 + m3/errors8 400 runs and on traces/tmp/tbo_tmax8 (300 traces, rand /
  fast) 600; kernel body 1,952, `AXI_SLICES` 1 / 3 496 each, U200 parameters `--reset-each` 688,
  `tsr_u200_k1` 448 + 224 `--reset-each`, `tsr_u200_k4` `--reset-each` 200.
- vkernel, Qwen2.5-0.5B Q8_0 (one prompt, 2 tokens), `klcheck compare --expect-identical` against
  `TSR_DEVICE=sim`: packaged k1 and k4 (T = 6 MVMs) 7 / 7 rows bit-identical, 290 runs each with 0
  findings; per-run PERF identical to HEAD (k1 PERF_CYCLES 59,314,184; k4 5,423,076).
- Blocks: tb/spm 18 passed; tb/tech 16 (SRAM equivalence in 21 configurations incl. 16 K × 64 and
  4 K × 64, 0 mismatches; tb/spm's tests on the Xilinx wrapper in 4 configurations; netlist
  simulations of 16 K × 64 and 4 K × 64, 0 mismatches); tb/mvm 14; tb/mvmback 18; the r3 back end
  in lockstep with the r2 one (all outputs and the per-token accumulators, tree and hold registers
  every cycle; random inputs incl. format flips, random tags and resets; 18 runs × 300 K cycles):
  0 mismatches. Mutants: 8 / 8 SPM mutants killed by tb/spm; of the 11 tb/mvmback mutants whose
  patterns changed, 10 killed and the known equivalent survives; the adapter campaign
  `docs/platform-u200.md` §10.3.
- `rtl/lint.sh` 213 lint runs and 4 XSim elaborations clean; `tb/run_all.sh` §13.4.

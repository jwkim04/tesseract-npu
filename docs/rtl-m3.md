# Tesseract M3 RTL — MVM unit architecture and interface contract

Status: M3 architecture v1 (2026-10-23). Contract for the M3 block agents (fp, front, back, top)
and the integration step. Integration (part B) done: `tsr_core` with `HAS_MVM = 1` (default since
part B) passes every trace set bit-exactly, see §13; Vivado timing and the U200 platform are part C. Source-of-truth order: `docs/spec.md` > ISA simulator behaviour
(`sw/sim/tsr_sim.c` `ex_mvm`; numerics `sw/sim/tsr_ref.c`) > this document > `docs/rtl-m2.md` >
`rtl/core/tsr_pkg.sv`. When this document needs a change, the lead edits it; spec changes go
through §11 (proposals).

**M3 scope** (spec ISA table: `MVM` 단계 = M3; the milestone figure of the spec export has no
text, so the scope is inferred from the spec): the MVM unit — one cluster per channel (spec "MVM
클러스터 마이크로아키텍처", stages 1–10), the activation broadcast, the result gather with ACC —
integrated into `tsr_core` with `HAS_MVM = 1`, executing every legal `MVM` bit-exactly like the
simulator (Q8_0 with A32 QA, Q4_K with A256 QA, any legal K / N / T, ACC, `dst_stride`,
register-relative operands), reading exactly `rows_c × ROW_BEATS` weight beats per channel, and
sustaining one weight beat per cycle per cluster at T = 1 when the memory delivers (rows of at
least `RB_MIN` beats, §7.5). Not in M3: DSP packing and URAM/BRAM mapping (`rtl/tech/xilinx`, M4),
overlapping consecutive MVMs (spec "연속 MVM 겹치기", M5). The U200 platform adapter, kernel packaging and the XRT
transport are also M3 (the M3 gate is "Qwen2.5-0.5B Q8_0 실기 동작", spec 마일스톤 table) but have their own
contract (M3 part C).

**M2 is unchanged.** Every rule of `docs/rtl-m2.md` applies as written: conventions R1–R6 and the
timing budget (§1), the dispatch protocol (§3.0), `tsr_mvm_cmd_t` (§3.4), the memory-port rules
M1–M8 and `tsr_mem_arb` (§4), the SPM client contract (§5). `HAS_MVM = 0` builds stay
bit-identical to M2. M3 only fills the tied-off M3 ports of `tsr_core` (rtl-m2 §10).

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
  | `BC_PIPE` | 1 | ≥ 0 | broadcast pipe stages tsr_mvm → cluster (`tsr_vpipe`), §3.1 |
  | `RES_PIPE` | 1 | ≥ 0 | result pipe stages cluster → gather (`tsr_rpipe`), §3.5 |
  | `ACT_RD_LAT` | 2 | ≥ 1 | activation buffer read latency (§5) |
  | `MAC_LAT` | 3 | ≥ 1 | `tsr_mac_lane` latency (§7.2) |
  | `MUL_LAT` | 4 | ≥ 1 | `tsr_fp_mul` latency (its 300 MHz default) |
  | `ADD_LAT` | 5 | 1..8 | `tsr_fp_add` latency; `TSR_DEF_FADD_LAT` = its measured 300 MHz default (§8.1) |
  | `WS_RD_LAT` | 2 | 1..4 | weight FIFO RAM read latency (`tsr_mvm_ws.RD_LAT`, `tsr_mvm_front.WS_RD_LAT`, `TSR_DEF_WS_RD_LAT`) |

  `tsr_core` exposes `MVM_WS_FIFO`, `MVM_BC_PIPE`, `MVM_RES_PIPE` and the latencies
  `MVM_ACT_RD_LAT`, `MVM_MAC_LAT`, `MVM_MUL_LAT`, `MVM_ADD_LAT`, `MVM_WS_RD_LAT` (additive,
  defaults `TSR_DEF_*`; the tests and tb/fastreplay vary them, §13.1).
- **Files** (`rtl/filelist.f` order: packages, tech, fp, SPM, arbiter, pipes, DMA, QACT, MVM
  bottom-up, control, core): `tech/generic/tsr_mac_lane.sv`, `core/tsr_fp_add.sv`,
  `core/tsr_vpipe.sv`, `core/tsr_rpipe.sv`, `core/tsr_mvm_ws.sv`, `core/tsr_mvm_front.sv`,
  `core/tsr_mvm_actbuf.sv`, `core/tsr_mvm_back.sv`, `core/tsr_mvm_cluster.sv`,
  `core/tsr_mvm_gather.sv`, `core/tsr_mvm.sv`. A new file is a filelist change (ask the lead).
- **Lint**: `rtl/lint.sh` lints every new module as its own top with edge parameters (NUM_CH 1 / 3
  / 16, T_MAX 1 / 8, K_MAX 256 / 65280, latencies 1 / 8, pipes 0 / 3, FIFO 64 / 1024). The
  integration step adds `tsr_core -GHAS_MVM=1` configurations (U200 v1, 1 / 3 / 16 channels, T_MAX
  8, small K_MAX, pipe and FIFO extremes).
- The skeletons in `rtl/core` (and `tsr_mac_lane`) carry the final port lists and a behavioural
  contract comment; bodies are tied off and lint clean. `tsr_vpipe` / `tsr_rpipe` are complete.
  Port-list changes are interface changes: agree them with the lead (update this file,
  `tsr_pkg.sv`, the skeletons and `tb/common/isa.py` together).

---

## 2. Block diagram and ownership

```
   tsr_cmdproc ──tsr_mvm_cmd_t──►┌──────────────────────────── tsr_mvm ─────────────────────────────┐
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
| (int) | integration | `rtl/core/tsr_core.sv` (HAS_MVM = 1 wiring, checks), `tb/core/` | §10.4 | `tb/core/test_core.py` |
| infra | architecture | `tsr_pkg.sv`, `tsr_vpipe.sv`, `tsr_rpipe.sv`, `rtl/filelist.f`, `rtl/lint.sh`, `tb/common/`, `tb/conftest.py`, `tb/pytest.ini`, `tb/run_all.sh`, this file | all | `tb/common/test_common.py` |

Block tests that need several RTL blocks (e.g. the back end with the real activation buffer, or a
cluster with a stub back end) put wrappers in `tb/<block>/hdl/` (linted by `rtl/lint.sh`).

---

## 3. Internal interfaces

### 3.0 Dispatch (tsr_cmdproc → tsr_mvm)

`tsr_mvm_cmd_t` (294 bits) and the dispatch protocol are those of `docs/rtl-m2.md` §3.0/§3.4.
`tsr_cmdproc` (HAS_MVM = 1, already verified against the simulator in `tb/cmdproc`) guarantees:
`fmt` ∈ {0, 1}; `0 < k ≤ K_MAX`, `k % bs == 0` (bs 32 / 256); `1 ≤ t ≤ T_MAX`;
`y_lines = ceil(n/16)`; `dst_stride` resolved (`w6 = 0` → `y_lines`; 0 when n = 0);
`rows_per_ch = ceil(n/NUM_CH)`; `row_beats = ROW_BEATS(fmt, k)`; `qa_lines = QA_LINES(fmt, k)`;
`dst_line + (t−1)·dst_stride + y_lines ≤ SPM lines`; `x_line + t·qa_lines ≤ SPM lines`;
`w_off + rows_per_ch·row_beats ≤ DRAM limit` (so every weight beat of every channel is in range and
`ADDR_BEAT_W` bits hold every address). **Not** guaranteed: absence of overlaps (dst vs x, token
dst ranges with `dst_stride < y_lines` and t > 1) — undefined results (spec 구현 명확화 "겹치는
피연산자"), but the unit must still finish normally (no hang, DONE ⇒ quiescent).

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
  instruction. Informational (not bit-compared, spec 구현 명확화 "성능 카운터").

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
l, data: the line}` with `qs_lines = ceil(k/64)`. QA layout: spec "양자화 활성값(QA) 레이아웃",
`tsr_ref.c` 418–447 (QS, then D, then BS for A256, each line-aligned).

### 3.2 Cluster setup and front command

On `bc_cmd` the cluster wrapper computes (multi-cycle arithmetic is fine; it overlaps the broadcast):
`first = CH × R` (64-bit), `rows_c = tsr_mvm_rows_c(n, R, first)` = `max(0, min(R, N − first))`
(spec 스트림 요청기; `tsr_ref.c` 335–352), `beats_c = rows_c × row_beats` (≤ 2^40 − 1: the dst SPM range check, which precedes DRAM_RANGE, bounds
rows_c ≤ N ≤ 16 × (2^26 − 1) because SPM_BYTES < 2^32, and exactly 2^40 would need a power-of-two
ROW_BEATS ≤ 1024 with rows_c ≥ 2^30; the maximum 2^40 − 1 is Q8_0 K 61664/61696, ROW_BEATS 1025.
`TSR_MVM_BEATS_W` = 40 has no spare bit: widening SPM_BYTES beyond 32 bits needs 41). If
`rows_c ≥ 1` it sends one `tsr_mvm_fcmd_t {fmt, k, row_beats, w_off, rows: rows_c, beats:
beats_c}` (137 bits, valid/ready) to the front end and registers `cfg_fmt`, `cfg_d_lines` for the
back end; with `rows_c = 0` nothing (the cluster only absorbs the broadcast). Row r of the channel
(global row `first + r`) starts at beat `w_off + r × row_beats` (spec 대칭 할당), so the channel's
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
ceil(rem/2) beats, Q4_K `rem` superblocks → 1 + 2·rem beats; spec 행 크기, `tsr_ref.c` 256–294).
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
  ROW_BEATS`: no over-fetch, no prefetch beyond the channel's rows (spec 구현 명확화 "성능 카운터":
  `PERF_CH_RD_BEATS[c]` MVM beats = `rows_c × ROW_BEATS`; spec 스트림 요청기 "R × ROW_BEATS는
  상한·할당 크기일 뿐 읽는 양이 아니다"). `rows_c = 0` → no request at all. Reference: `tsr_sim.c`
  759–769 (`cap_read(sim, c, woff, rows_c × row_beats)`).
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

Spec 자원 추정: "클러스터당 활성값 버퍼 1024비트 × 292 per token" = one full QA image per token
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
`RD_LAT = ACT_RD_LAT`; U200 T_MAX 1: 2 × 8 + 8 = 24 BRAM36 per cluster (M4 mapping). Reads and
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
`rows_c = 0` stay silent. Placement: in the SLR of channel c's DDR controller (spec 배치).

### 6.3 Gather (tsr_mvm_gather): partial lines, byte enables, ACC, tails

Semantics (tsr_sim.c `ex_mvm` 788–803): token tk < T writes f32 element n_row of its y vector at
SPM line `dst + tk·stride + n_row/16`, lane `n_row % 16`, value `y` or (ACC) `fadd(y_old, y)`
(spec 구현 명확화 "ACC 덧셈": one f32 addition with the fp rules, old value first — addition is
commutative bit-exactly), and the bytes after element N−1 in the token's last line are 0 (also
with ACC; spec 구현 명확화 "꼬리 바이트").

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
Row accumulation (`tsr_dot_row` 600–635, spec 행 누적): 16 partials start at +0.0 per row; term
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
`tsr_mac_lane` is the tech wrapper of spec "DSP 패킹" (two tokens per DSP for T ≥ 2 in
`rtl/tech/xilinx`, M4); its sums are exact `SUM_W = 21`-bit two's complement.

### 7.3 Accumulators: two lanes, 8 partials each

- Two accumulate lanes per token, each one `tsr_fp_add` (`ADD_LAT`) with its 8 partials: lane 0
  holds p0, p2, …, p14, lane 1 p1, p3, …, p15 (`p[k]` in lane k % 2, slot k / 2).
- Q8_0: the two terms of a beat (b even → lane 0, b+1 → lane 1) are accumulated in the same cycle.
  Within a lane a slot is hit once per 8 quant beats, so two terms of one partial are ≥ 8 cycles
  apart (spec 수치 규칙: "지연 8사이클 이하의 파이프라인 fp 덧셈기로 II=1"). Requirement: from the
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
Q8_0: 15 beats incl. META; its 152 K-row output head dominates the token) and M5 MoE expert rows
are shorter still (Qwen3-30B-A3B `ffn_down` K = 768: Q8_0 13 beats, Q4_K 7 beats). Suggested
structure: one pipelined `tsr_fp_add` per tree level and token (level 1 issues the 8 pairs of a
snapshot on 8 consecutive cycles, level 2 its 4 pairs as level-1 results arrive, …), with a ring of
snapshot / intermediate registers deep enough for every row in flight (latency ≈ 4·ADD_LAT + 15),
so a new snapshot can enter level 1 every 8 cycles. Other structures are fine if they meet §7.5
and keep the exact pair order. Each result goes to a result FIFO whose space was reserved before
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

The spec table's "msum 23비트" holds for QACT-produced BS only (proposal SP1).

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
steady state 1.0000 cycles per weight beat. Example: Qwen2.5-32B
`ffn_down` (Q4_K, N 5120, K 27648, 4 channels): 1280 × 243 = 311,040 cycles = 1.04 ms at 300 MHz;
broadcast 493 lines (0.16 %). Rows shorter than RB_MIN (≤ 9, §7.5) are tree-bound; tiny K with many
channels and tokens may be gather-bound (≤ 1 SPM line write per cycle).

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
   adder levels per stage); DSP cascades in M4.
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

### 9.3 Cost of T_MAX (synthesis T_MAX = 1; tests up to 8)

Replicated per token in every cluster: 128 MACs (U200 T = 1: 128 DSP48E2; T ≥ 2 packs two tokens
per DSP → 64 per token), 4 adder trees (if not in DSP cascades), activation buffer (QS 2 banks +
DBS: 24 BRAM36 at K_MAX 32768), integer scale (12 small multipliers), 2 `i32_to_f32`, 4
`tsr_fp_mul` (8 DSP), 4 `tsr_fp_add` (2 accumulate, 1 Q4_K subtract, 1 tree), 3 × 16 × 32
partial / snapshot flops, +32 result-bus wires, +512 bits per gather assembler. Rough generic cost
per token per cluster: ~6–8 K LUT, 136 DSP, 24 BRAM36 → T_MAX 8 × 4 clusters ≈ 200–250 K LUT,
2.2 K DSP (packing: ~1.1 K), 768 BRAM36 — fits a VU9P (1.18 M LUT, 6840 DSP, 2160 BRAM36) but the
T_MAX choice is a spec open item (M7). Shared per cluster: front end, weight FIFO (8 BRAM36).

---

## 10. Verification

Oracle chain (spec 검증 전략): `libtsr_ref` (bit-exact numerics) for the blocks, the ISA simulator
for the core. Bit-exact first; every test must be shown to fail on at least one deliberate
mutation (scratch copy) and the report says which.

### 10.1 Infrastructure (tb/common, architecture)

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
- `pycore.py`: `PyCore(cfg, master, has_mvm=True)` executes `MVM` exactly like `ex_mvm` over the
  memory port (WS id `{2, 0}` = 8, §4 bursts, `rows_c × ROW_BEATS` beats per channel counted in
  `mvm_beats`, all y first, then per token with ACC and zero tails). The self-test replays
  `traces/m2`, the MVM traces of `traces/gen` and every `traces/m3` set (long sets capped)
  through `MemModel` + `replay.compare`.
- `simlib.py` (moved from tb/cmdproc; `tb/cmdproc/simlib.py` re-exports it): `Sim(cfg).run(...)`
  → `SimResult` (final CSRs, captured trace, registers, SPM, retired, `perf` deltas). The shared
  library is built into `<build root>/simlib/<sha256 of both archives>/` from private copies of
  the archives, symbol-checked and atomically renamed: concurrent builders never race and a
  rebuild of build/dev cannot change a loaded library.
- `proggen.py` (moved from tb/cmdproc; `ProgGen(m3=False)` is stream-identical to M2, pinned by a
  digest test): `ProgGen(rnd, cfg, has_mvm, m3=True)` — `init_data()` (weight window of sane fp16
  halfwords, f32 activation window, raw window), `program()` with execution-oriented MVMs (both
  fmts, ACC, strides, T, N edge cases, K over partial chunks, x from an earlier QACT, register-
  relative A1..A3) and QACTs without dst/src overlap, `mvm_program()` (LD → QACT → MVM (→ ACC MVM)
  → ST chains); `executed_count(prog, final, count)` and `executed_overlaps(prog, n, args, cfg)`
  (undefined overlaps of an executed prefix: MVM dst vs x, token dst ranges, QACT dst vs src).
- `mem_model.MemModel.src_beats[c][s]` counts read beats per memory-side source (id[3:2]);
  `trace.Trace.expected_read_beats()` derives, from the program alone, the LD beats (source DMA)
  and the MVM weight beats (`rows_c × ROW_BEATS` per executed MVM, source WS) per channel; it equals
  the simulator's PERF read beats on channels ≥ 1 for all 488 traces of traces/m2, gen and m3.
  `replay.compare(..., check_sources=True)` checks them at the port; `replay.run_trace` does so by
  default (the M2 core passes: tb/core unchanged and green). This is the port-level form of the
  spec's "MVM weight beats per channel = rows_c × ROW_BEATS" rule (fetch beats are not checked).
- `replay.estimate_cycles` budgets MVMs (executed instructions only); `tsr_vpipe` / `tsr_rpipe`
  are tested by `test_pipes_sim` (`tb/common/hdl/tsr_tb_pipes.sv`, DEPTH 0 / 1 / 3).

### 10.2 Block tests (spec 블록 단위 테스트 범위)

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
  3 (spec "SLR 파이프라인 깊이 파라미터를 바꿔도 결과가 같음"), SPM written bytes == y lines
  exactly, done timing and quiescence.

### 10.3 Trace sets

`traces/gen` (M1 generator, 24 + 8 traces with MVM, U200 v1 / T_MAX 8) and `traces/m3`
(`traces/m3/regen.sh`, `tsr-trace-gen --ops m3`): `tmax1` 64 (U200 v1, three long weight
streams), `tmax8` 32, `small1ch` 24 (1 ch, ADDR_BEAT_W 22, K_MAX 4096, T_MAX 2), `ch3` 16, `ch16`
12 (16 channels, K_MAX 8192, T_MAX 2), `errors` / `errors8` 68 each (every MVM fault and every M2
error ending). Replayed bit-exactly by `tsr-trace-replay`, by `PyCore` in the tb/common self-test,
and by the RTL core (§10.4).

### 10.4 Integration (tb/core, HAS_MVM = 1)

Trace replay of `traces/m3` and `traces/gen` under the memory regimes of rtl-m2 §12 (rand / fast
/ slow / ordered) with `replay.run_trace` (incl. the per-source beat check of §10.1), PERF checks
(`PERF_CH_RD_BEATS[c]` == the trace's for c ≥ 1, i.e. LD + `rows_c × ROW_BEATS`; MVM cycles > 0),
lockstep random programs with `ProgGen(m3=True)` (`program()` and `mvm_program()`) vs the
simulator (whole SPM compared; `executed_overlaps` must stay empty), implementation-parameter variants (WS FIFO 64, pipes 0 / 3,
SPM_RD_LAT 1 / 4, RD_OUTSTANDING 1), throughput (1 weight beat/cycle/cluster at T = 1 with an
ideal memory for ROW_BEATS ≥ RB_MIN). Simulation-only checks in `tsr_core` (extend I1–I3): MVM
SPM / WS traffic only while the MVM unit holds an instruction; at `done` no WS request pending and
no WS read outstanding on any channel; per MVM and channel exactly `rows_c × ROW_BEATS` weight
beats requested, in bursts that follow the §4 rule (I4, §13.1).

---

## 11. Spec proposals (for the lead)

- SP1 **Integer widths for arbitrary QA content**: "MVM은 SPM의 어떤 바이트도 QA로 받는다 (QACT
  출력이 아니어도 합법). BS가 임의의 int16이면 Q4_K msum은 부호 26비트가 필요하다 (QACT가 만든 BS에서는
  23비트)." (spec 정수 폭 table note.)
- SP2 **LANES**: "LANES = BEAT_BITS / 4 (Q4_K beat 하나의 가중치 수)이며 독립 파라미터가 아니다" —
  the cluster structure (stage 4/6) fixes 128 lanes for 512-bit beats; CFG0 keeps reporting it.
- SP3 **PERF_MVM_CYCLES in hardware**: cycles in which the MVM unit holds an instruction (accept →
  done), like rtl-m2 P5 for the other counters.
- SP4 **N = 0 MVM**: reads nothing from the SPM either (no observable difference; states that the
  x operand is not accessed when there are no rows).
- SP5 (superseded: §7.4 / §7.5 now require RB_MIN ≤ 9) **Row-boundary throughput**: spec stage 9 says the previous row's tree finishes "뒤에서"; add
  that this holds for rows of at least RB_MIN = 4·ADD_LAT + 12 beats in v1 (shorter rows, e.g.
  K = 896 Q8_0, stall at row starts; performance only).
- SP6 **Stage 2 tags**: the front end absorbs META beats (they carry no work for the back end);
  the tag set of stage 2 is then {QUANT data, block/superblock, half, row end, MVM end}.

---

## 12. Reference map

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
| MVM checks / weight beats / compute / write | `tsr_sim.c` 736–757 / 759–769 / 771–786 / 788–803 |

---

## 13. Integration and results (M3 part B, integration step)

### 13.1 tsr_core with HAS_MVM = 1

- **Default `HAS_MVM = 1`** (`tsr_core` parameter): the U200 v1 bitstream executes llama.cpp graphs,
  which need `MVM`; the default parameters are the bitstream's. M2 test builds pin `HAS_MVM = 0`
  explicitly (tb/core/test_core.py `core_params(cfg, has_mvm=0)`; tb/fastreplay passes `HAS_MVM`
  always; rtl/lint.sh lints both). `HAS_MVM = 0` keeps the M2 netlist (tie-offs in `g_nomvm`).
- **Wiring** (§2, rtl-m2 §10): `tsr_mvm` on the cmdproc dispatch port (`mvm_cmd_*`, `mvm_done`), SPM
  client `TSR_SPM_CL_MVM` (read and write port), weight-stream source `TSR_SRC_WS` of arbiter c ↔
  cluster c (request, local id, `rd_rsp_ready`; the shared response data / last / id of channel c),
  `perf_mvm_busy` → `tsr_csr`. No port, `tsr_pkg` or filelist change.
- **New parameters (additive)**: `MVM_WS_FIFO` (512), `MVM_BC_PIPE` (1), `MVM_RES_PIPE` (1) and the
  latencies `MVM_ACT_RD_LAT` (2), `MVM_MAC_LAT` (3), `MVM_MUL_LAT` (4), `MVM_ADD_LAT` (5),
  `MVM_WS_RD_LAT` (2) (defaults = `tsr_pkg::TSR_DEF_*`). Elaboration checks with `HAS_MVM = 1`:
  `LANES == 128`, `MVM_WS_FIFO` a power of two ≥ 64, `MVM_ADD_LAT` 1..8, `MVM_WS_RD_LAT` 1..4, the
  other latencies ≥ 1. The `$fatal` on `HAS_MVM = 1` is gone.
- **Simulation checks** (`ifndef SYNTHESIS`, armed by the first reset edge), extending rtl-m2 §12:
  - I1 at most one unit offered a command, never while one holds an instruction; `done` only from
    the holder (never in its accepting cycle); MVM `cmd_ready` low while it holds the instruction;
    `perf_mvm_busy` equals "holds an instruction" in every cycle (the PERF_MVM_CYCLES meaning).
  - I2 SPM requests by DMA / QACT only while holding (or accepting) their instruction, by the MVM
    unit strictly after its accept; MVM SPM read responses only while it holds the instruction; no
    two units on the SPM in one cycle; weight-stream requests / responses only while the MVM unit
    holds an instruction; WS `rd_rsp_ready` constant 1.
  - I3 (M2) memory traffic only during a run; at DONE nothing offered, no read outstanding.
  - I4 at `mvm_done`: no WS request offered, WS beats requested == received on every channel
    (counted at the arbiter's source port, so channel-0 fetch reads cannot mask it), no MVM SPM
    request / write offered, no MVM SPM read outstanding; per MVM and channel exactly
    `rows_c × ROW_BEATS` beats requested (never more at any time), every request at the next
    address with `len = min(64 − addr % 64, beats left)` (§4).
- **Lint**: `rtl/lint.sh` lints the default core (U200 v1 with MVM), 6 `HAS_MVM = 0` configurations
  and 10 `HAS_MVM = 1` configurations (T_MAX 8; 1 / 2 / 3 / 16 channels; K_MAX 256 with a 64-line
  SPM and 32-beat channels; pipes 0 and 3 with FIFO 64 / 1024, SPM_RD_LAT 1 / 4, RD_OUTSTANDING
  1; all MVM latencies 1; ACT 4 / MAC 8 / MUL 8 / ADD 8 / WS_RD 4): 79 runs, all clean.

### 13.2 Core-level results (all bit-exact against the ISA simulator)

tb/core/test_core_m3.py (cocotb, `CoreH` harness of test_core.py, MemModel at the boundary):

| pytest item | Build | Result |
|---|---|---|
| `test_core_m3_traces[gen.tmax1+m2.errors+m2.tmax1+m3.errors+m3.tmax1]` | U200 v1 | 256 traces × 4 regimes (every regime on every trace; 1708 MVMs), 150 random programs (259 MVMs), whole SPM |
| `…[gen.tmax8+m2.errors8+m2.tmax8+m3.errors8+m3.tmax8]` | U200, T_MAX 8 | 156 traces × 4 regimes (1084 MVMs; 2 rotating passes by default since), 150 random (243 MVMs) |
| `…[m2.small1ch+m3.small1ch]` | 1 ch, 64 KiB, T_MAX 2 | 40 traces × 4 regimes (316 MVMs), 300 random (451 MVMs) |
| `…[m2.ch3+m3.ch3]` | 3 ch, T_MAX 3 | 24 traces × 4 regimes (300 MVMs), 300 random (427 MVMs) |
| `…[m3.ch16]` | 16 ch, K_MAX 8192, T_MAX 2 | 12 traces × 4 regimes (120 MVMs), 300 random (418 MVMs, 1.15 M weight beats; 60 by default since) |
| `test_core_m3_impl[u200_bitstream]` | U200, SPM_RD_LAT 4, BC_PIPE 2, RES_PIPE 2 | m3/tmax1 + gen/tmax1 (88 traces, 285 MVMs, rotating regimes), 16 random |
| `test_core_m3_impl[pipe0_fifo64_lat1]` | small1ch, pipes 0, WS FIFO 64, SPM_RD_LAT 1, RD_OUTSTANDING 1, WS_RD_LAT 1, FETCH_BEATS 2 | 40 traces × 2 (158 MVMs), 40 random |
| `test_core_m3_impl[pipe3_fifo1024_lat4]` | ch3, pipes 3, WS FIFO 1024, SPM_RD_LAT 4, RD_OUTSTANDING 2, WS_RD_LAT 4, FETCH_BEATS 64 | 24 traces × 2 (150 MVMs), 40 random |
| `test_core_m3_impl[fast_latencies]` | ch3, ACT_RD / MAC / MUL / ADD / WS_RD 1 | 24 traces × 2, 40 random |
| `test_core_m3_impl[slow_latencies]` | ch16, ACT_RD 4, MAC 8, MUL 8, ADD 8, WS_RD 4, WS FIFO 128, SPM_RD_LAT 3 | 12 traces × 2, 30 random (180 K weight beats) |
| `test_core_m3_directed[c2]` | 2 ch, T_MAX 4, 64 KiB SPM | M2 directed tests on the MVM core (CSR map, W1C, SOFT_RESET 21 / 24 while BUSY, hardware reset 12 / 12 in a run, ABORT 46 mid-program, back-to-back), m3_perf, m3_mvm_st (18 programs), m3_abort (24 aborted, 15 inside an MVM), m3_soft_reset (23 while BUSY, 9 inside an MVM), m3_hw_reset (22 / 12), 100 random (139 MVMs) |
| `test_core_m3_directed[c3t3]` | 3 ch, T_MAX 3, 256 KiB | m3_perf, m3_throughput (T 1 and 3), m3_mvm_st, m3_abort (23 / 12), m3_hw_reset (21 / 16) |
| `test_core_m3_directed[u200]` | U200 v1 | m3_throughput incl. the M3 gate shapes and the output head, m3_perf, m3_mvm_st (16 programs) |
| `test_core_m3_directed[u200t8]` | U200, T_MAX 8 | m3_throughput (T 1 and 8), m3_perf |
| `test_core_m3_fastreplay` | 5 configs | every traces/m3 + traces/gen trace × 4 regimes (C++ replay) |

Traces replayed per run: STATUS / ERR_* / expect data / written set / read set, per-source read
beats at the port (DMA == LD, WS == `rows_c × ROW_BEATS` of every executed MVM), PERF deltas
(`PERF_MVM_CYCLES > 0` iff an MVM executed and ≤ PERF_CYCLES, CH_RD_BEATS == port and == trace for
c ≥ 1), quiescence at DONE, and tsr_core I1..I4. Random programs (`ProgGen(m3=True)`, half
`program()`, half `mvm_program()`, CSR polling during a quarter of them): registers, PC, retired,
DRAM, per-source beats, whole SPM (ST dumps); `executed_overlaps()` is asserted empty.

tb/fastreplay (C++ replay, `run.py --has-mvm`), every check of fastreplay.cpp incl. RTL assertions:

| Trace set | Regimes | Runs | Simulated cycles | Wall (6 processes) |
|---|---|---|---|---|
| traces/m3 + gen + m2 (488) | all 4 | 1952, 0 failed | 12.7 M | 159 s incl. 2 builds |
| traces/model (290: Qwen2.5-0.5B Q8_0 + Q4_K_M-requant, 1 token) | all 4 | 1160, 0 failed | 32.6 M | 36 s |
| traces/tmp/tbo (1402, T_MAX 1) | all 4 | 5608, 0 failed | 552.6 M | 655 s |
| traces/tmp/tbo_tmax8 (1402, T_MAX 8) | all 4 | 5608, 0 failed | 216.6 M | 2481 s |
| U200 bitstream params (SPM_RD_LAT 4, BC_PIPE 2, RES_PIPE 2): model + m3/tmax1 + m3/errors + gen/tmax1 (446) | all 4 | 1784, 0 failed | 36.9 M | 59 s |
| same params: traces/tmp/tbo (1402) | rand | 1402, 0 failed | 114.1 M | 151 s |
| pipes 0, WS FIFO 64, SPM_RD_LAT 1, RD_OUTSTANDING 1, WS_RD_LAT 1, FETCH_BEATS 2: m3 + gen (316) | all 4 | 1264, 0 failed | 12.6 M | 230 s |
| pipes 3, WS FIFO 1024, SPM_RD_LAT 4, RD_OUTSTANDING 2, WS_RD_LAT 4, FETCH_BEATS 64: m3 + gen | all 4 | 1264, 0 failed | 10.3 M | 217 s |
| all MVM latencies 1 (ACT, MAC, MUL, ADD, WS_RD): m3 + gen | all 4 | 1264, 0 failed | 10.2 M | 216 s |
| ACT_RD 4, MAC 8, MUL 8, ADD 8, WS_RD 4, WS FIFO 128: m3 + gen | all 4 | 1264, 0 failed | 10.4 M | 350 s |

Nothing was skipped: every trace of traces/m3, gen, m2, model, tmp/tbo and tmp/tbo_tmax8 ran in
every regime. Simulation speed (one process): U200 T_MAX 1 ~160 K cycles/s, T_MAX 8 ~21 K,
16 channels ~17 K, 1 channel ~300 K; cocotb U200 ~11 K cycles/s, T_MAX 8 ~6 K, 16 channels ~1.3 K.

**Throughput** (`m3_throughput`, ideal memory = always ready, latency 0..1, on every channel at
once; two-point method with R and 3R rows per channel; U200 v1):

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

The whole 1-token Qwen2.5-0.5B Q8_0 trace set (145 traces) spends 2.08 M MVM cycles (2.27 M
cycles in total incl. LD / QACT; 7.6 ms at 300 MHz) under the fast regime (latency 0..2), 3.03 M
MVM cycles under rand and 7.73 M under slow; the output head alone 569,845 / 822,464 / 2,103,586.

**PERF** (`m3_perf`): `PERF_MVM_CYCLES` equals, for every MVM, the cycles from the one after the
dispatch handshake through `mvm_done` counted at the port by the testbench; N = 0 MVMs (also with
ACC) are held 1 cycle with no SPM or weight traffic; a faulting MVM counts 0; per MVM the weight
beats per channel are `rows_c × ROW_BEATS`, the SPM reads `T × QA_LINES` (+ one per gather job
with ACC) and the SPM writes exactly one per gather job (`mvm_model.gather_jobs`); no MVM SPM /
weight traffic and no `mvm_done` outside a held MVM; PERF_CH_RD_BEATS[c] == port beats (LD + WS,
plus exactly ceil(CMD_COUNT/2) fetch beats on channel 0).

**Directed**: `m3_mvm_st` (MVM → ST of y, MVM → MVM ACC, x = the previous MVM's y as raw QA, LD of
raw bytes → MVM, MVM → QACT of y, rotating regimes, whole SPM compared), `m3_abort` (ABORT at
random cycles, half timed inside a dispatched MVM: the MVM completes, ERR_PC / PC / retired, effects
and per-source read beats == the executed prefix, whole SPM afterwards), `m3_soft_reset` and
`m3_hw_reset` (inside MVM programs with weight reads in flight; CSRs, PERF, registers, quiet port,
then a fresh MVM program == a fresh simulator), plus the M2 directed tests (CSR map, W1C, SOFT_RESET,
hardware reset, ABORT, back-to-back) re-run on the HAS_MVM = 1 core.

**Integration findings**: no functional RTL bug was found at core level, so no MVM block file was
changed by the integration step. Testbench issues fixed on the way: tb/core
`reset_core()` now drives the memory-side inputs and the CSR port to 0 before reset. Before that,
a cocotb test that started a fresh simulation with `core_traces` saw Verilator's random input
values until the first MemModel drove them. A random `wr_rsp_valid` tripped tsr_dma's assertion
(the M2 suite never hit this because `core_csr_consts` ran first). Smaller fixes: the chain STs
skip the stride gaps an MVM never writes, the executed-MVM count is taken before the abort path
swaps the captured program, and the c3t3 throughput test now checks the real y extent (it had
silently skipped every shape).

**tb/run_all.sh** (lint + every block suite + the M2 and M3 core suites + the fastreplay smoke):
rtl/lint.sh all clean (79 runs), then 170 passed, 5 opt-in skipped (fp slow sweeps, QACT Vivado),
0 failed; 5206 s wall with every Verilator build cold (`TSR_BUILD_JOBS=4`, one simulation at a
time). The M3 core suite alone with the larger counts of the table above took 101 min.

### 13.3 Mutation campaign (tb/core/mutate_core.py, scratch copies of rtl/)

31 mutants (`python3 tb/core/mutate_core.py WORK [--only ...] [--noassert]`): stage 1 =
tb/fastreplay of traces/m3/ch3 + m3/tmax1 + gen/tmax1 under rand / fast / slow (two builds, stop
at the first failure), stage 2 (only if stage 1 passes) = cocotb `test_core_m3_directed[c2]`
(m3_perf, m3_mvm_st, m3_random, m3_abort). **28 killed, 3 survived; the 3 survivors are the three
mutants marked equivalent in advance.** Results: scratch `m3b/int/mut/results.json`.

| Mutant | What it breaks | Killed by |
|---|---|---|
| W_ws_req_ch_rot | cluster c's weight requests go to arbiter c+1 | stage 1: hang + reads outside init |
| W_ws_rsp_data_rev | weight response data of the channels reversed | stage 1: y mismatch |
| W_ws_len_msb | WS burst length 64 → 0 | stage 1: memory protocol (len 0) |
| W_perf_mvm_tied0 | PERF_MVM_CYCLES input unconnected | stage 1: PERF check |
| W_spm_rsp_client | MVM sees the QACT client's SPM read responses | stage 1: hang |
| W_mvm_spm_rd_lat | tsr_mvm told SPM_RD_LAT + 1 | stage 1: y mismatch (ACC) |
| M_bc_stride | broadcast skips a line at each token boundary | stage 1: y mismatch |
| M_bc_last_tok0 | `last` after token 0 (activations released early) | stage 1: y mismatch |
| M_expect_ch_le | gather expects a cluster with c·R == N | stage 1: hang (an MVM of traces/m3/ch3 with c·R == N) |
| M_busy_from_accept | perf_mvm_busy also in the accepting cycle | stage 1: I1 assertion + PERF; without assertions stage 2 (m3_perf exact cycles) |
| M_rows_c_plus1 / M_rows_c_minus1 | rows_c off by one (full / partial channel) | stage 1: per-source beats, PERF_CH_RD_BEATS |
| M_act_gate_off | weight beats pass before the activations are buffered | stage 1: y mismatch |
| M_ws_overfetch | one weight beat more than rows_c × ROW_BEATS | stage 1: hang + read outside init |
| M_q8_two_always | Q8_0 absent block b+1 used | **stage 2 only** (random programs: raw weight windows; the traces' repacked rows hold zeros there, so the term is +0) |
| M_tree_l1_order | tree level-1 pair order | stage 1: y mismatch |
| M_slot_perm_q8 | Q8_0 terms into permuted partials | stage 1: y mismatch |
| M_lane_swap_q8 | Q8_0 even / odd terms swap partial lanes | stage 1 (predicted equivalent, but lane 1 issues only with `two`: odd last chunks drop a real term) |
| M_msum_23b | Q4_K msum wraps at 23 bits | stage 1: y mismatch (raw BS, traces/m3/ch3) |
| M_q8_half_sel | Q8_0 QS half select on the wrong bit | stage 1: y mismatch |
| M_acc_tail_sum / M_acc_tail_kept / M_be_no_tail | tail lanes not zero (ACC sum, ACC kept, plain) | stage 1: y tail bytes |
| M_be_extra_lane | byte enables cover the lane below a job | stage 1: y mismatch (shared lines) |
| M_done_early | gather done before the last SPM write | stage 1: I4 assertion; without assertions stage 2 (m3_perf: write after done) |
| M_acc_old_lane | ACC adds the neighbour lane's old value | stage 1: y mismatch |
| M_acc_rd_align | ACC job paired with the old line one cycle late | stage 1: y mismatch |
| M_cmdproc_R_plus1 | dispatched rows_per_ch = ceil(N/C) + 1 | stage 1: per-source beats |
| W_mvm_done_late_ready | cmd_ready masked in the done cycle | survived — equivalent: cmd_ready is already 0 there (I1) |
| W_ws_rsp_ready_1 | WS rd_rsp_ready tied 1 | survived — equivalent: the streamer's ready is constant 1 (§4, I2 asserts it) |
| W_bc_pipe_param | broadcast pipe one stage deeper than MVM_BC_PIPE | survived — equivalent by contract (implementation parameters change timing only); the parameter values are exercised end to end by the variants, but no test checks that the requested depth arrives |

Every mutant whose first failure involved an RTL assertion (M_busy_from_accept, M_done_early,
W_spm_rsp_client, W_mvm_spm_rd_lat, M_acc_rd_align) was rerun with every simulation check
removed (`define SYNTHESIS` in the scratch copy): all 5 are still killed by the testbenches.

### 13.4 Open issues for part C

- **Vivado timing of the whole core with 4 clusters** (not run here): the per-block estimates of
  §9.2 and the back / top reports (i2f normalise stage, 6u × 17s multipliers, generic MAC lanes,
  16:1 accumulate operand mux, gather take path, cluster 32 × 16 product) need an out-of-context
  run of `tsr_core` (tb/core/viv/core_ooc.tcl with `HAS_MVM=1`), then a floorplanned one: clusters
  in the SLR of their DDR4 controller, `MVM_BC_PIPE` / `MVM_RES_PIPE` ≥ 2 across SLRs (verified
  bit-exact above), SPM in URAM with `SPM_RD_LAT 4` (verified), replicated per-SLR reset.
- **Resources**: not synthesised; per-token / per-cluster estimates in the block headers (back:
  ~8.5 K FF per token, 7 `tsr_fp_add`; gather ~9.9 K FF + 16 `tsr_fp_add`). URAM / BRAM mapping of
  the activation buffers and weight FIFOs (`rtl/tech/xilinx`, M4).
- **U200 platform adapter** (part C): AXI4 ↔ the core's memory ports (one per DDR4 channel, read
  id `{2, 0}` for weights, in order per id), AXI4-Lite ↔ the CSR port, `ap_rst_n` synchronisation
  and the XRT kernel packaging; the M3 gate is the Qwen2.5-0.5B Q8_0 run on hardware.
- **Simulation cost**: the T_MAX 8 and 16-channel builds simulate at 6 K / 1.3 K cycles/s in cocotb;
  their default trace passes and random-program counts are reduced (every regime still covered),
  and tb/fastreplay carries the full campaigns.

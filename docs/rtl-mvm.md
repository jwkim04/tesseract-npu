# Tesseract MVM unit: RTL architecture and numerics

This document is the RTL contract of the MVM unit: one cluster per DDR4 channel, its interfaces, the bit-exact numerics,
throughput, verification, and the structures that make it meet timing on the U200 (§14). The unit executes `MVM` (spec §8, "MVM
cluster microarchitecture") for every legal K, N and T bit-exactly like the simulator: Q8_0 with A32 QA, Q4_K with A256 QA, ACC
and `dst_stride`. It reads exactly `rows_c * ROW_BEATS` weight beats per channel and sustains one weight beat per cycle per
cluster at T = 1 when the memory delivers it. The queue head of the issue path dispatches it ([CC](design/04-control-core.md) §8,
§9) and it reads the SPM through the even and odd banks (CC §10). The oracle is `ex_mvm` in `tsr_sim.c` and `tsr_ref.c`.

The cluster also has the attention modes `ATTQK` (0x21) and `ATTPV` (0x22), whose contract is [AT](design/03-attention-kv.md)
§5.2, §5.3 and §6 and whose structure is AT §8 (§6.5). Authority: `docs/spec.md` > simulator > `docs/design` > this file >
[rtl-core.md](rtl-core.md) > `tsr_pkg.sv`. The rules of [rtl-core.md](rtl-core.md) apply as written: R1 to R6 and the timing
budget (§1), the dispatch protocol (§3.0), `tsr_mvm_cmd_t` (§3.4), the memory-port rules M1 to M8 (§4) and the SPM client contract
(§5).

## 1. Conventions

- [rtl-core.md](rtl-core.md) §1 applies. fp arithmetic goes only through `tsr_fp_mul` (`MUL_LAT`), `tsr_fp_add` (`ADD_LAT`, §8)
  and the `tsr_fp_pkg` conversions `tsr_fp16_to_f32` and `tsr_i32_to_f32`. Each is a separately rounded spec operation with no
  FMA. Integer sums are exact (widths of §7.6), so their order is free.
- `tsr_core` supplies `NUM_CH T_MAX K_MAX SPM_BYTES SPM_RD_LAT ADDR_BEAT_W BEAT_BITS`. `LANES` must be 128 (`BEAT_BITS / 4`, one
  Q4_K beat), which `tsr_core` checks when `HAS_MVM` is 1. The implementation parameters give identical results for every legal
  value, and the tests vary them (defaults in `tsr_pkg`):

  | Parameter | Default | Range | Meaning |
  |---|---|---|---|
  | `WS_FIFO_BEATS` | 512 | power of 2, at least 64 | weight FIFO per cluster (§4) |
  | `BC_PIPE` | 1 | 0 or more | broadcast pipe stages (`tsr_vpipe`, §3.1), also the weight-stream response pipe (§14.4) |
  | `RES_PIPE` | 1 | 0 or more | result pipe stages (`tsr_rpipe`, §3.5), also the weight-stream request pipe and the gather's job and start pipes (`JOB_PIPE`, §14.4) |
  | `ACT_RD_LAT` | 2 | 1 or more | activation buffer read latency (§5) |
  | `MAC_LAT` | 3 | 1 or more | `tsr_mac_lane` latency (§7.2) |
  | `MUL_LAT` | 4 | 1 or more | `tsr_fp_mul` latency |
  | `ADD_LAT` | 5 | 1 to 8 | `tsr_fp_add` latency (`TSR_DEF_FADD_LAT`, §8.1) |
  | `WS_RD_LAT` | 2 | 1 to 4 | weight FIFO RAM read latency (`TSR_DEF_WS_RD_LAT`) |

  `tsr_core` exposes them as `MVM_WS_FIFO`, `MVM_BC_PIPE`, `MVM_RES_PIPE`, `MVM_ACT_RD_LAT`, `MVM_MAC_LAT`, `MVM_MUL_LAT`,
  `MVM_ADD_LAT` and `MVM_WS_RD_LAT`, with defaults `TSR_DEF_*`.
- The files, in `rtl/filelist.f` order, are `tech/generic/tsr_mac_lane.sv`, `core/tsr_fp_add.sv`, `tsr_vpipe.sv`, `tsr_rpipe.sv`,
  `tsr_mvm_ws.sv`, `tsr_mvm_front.sv`, `tsr_mvm_actbuf.sv`, `tsr_mvm_back.sv`, `tsr_mvm_cluster.sv`, `tsr_mvm_gather.sv` and
  `tsr_mvm.sv`. `rtl/lint.sh` lints each as its own top with edge parameters and lints `tsr_core` with `HAS_MVM` 1 at several
  channel counts and parameter extremes.

## 2. Block diagram and ownership

<p align="center"><img src="fig/cluster.svg" alt="tsr_mvm, shared by all clusters, reads the QA lines from the SPM read port and broadcasts the setup of the MVM tile op, which starts the cluster's weight requests, then the lines through each cluster's own tsr_vpipe into the cluster's activation buffer. DDR4 channel c streams weight beats into the front end (stream requester with a 512-beat FIFO, chunk parser, meta + unpack); tagged beats enter the back end once the activation buffer is complete: 4 MAC lanes, integer scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials per row, tree16. Row results return through the cluster's tsr_rpipe to the shared gather, which writes 16 rows per SPM line, or with ACC reads the old line and adds first. The attention modes ATTQK and ATTPV add the p broadcast and gather ATT path in tsr_mvm and the w-unit, capture bank and convert pipes in the cluster." width="100%"></p>

| Block | Files | Contract | Tests |
|---|---|---|---|
| (fp) | `tsr_fp_add.sv` | §8 | `tb/fp` |
| (front) | `tsr_mvm_ws.sv`, `tsr_mvm_front.sv` | §3.2, §3.3, §4 | `tb/mvmfront` |
| (back) | `tsr_mvm_back.sv`, `tsr_mvm_actbuf.sv`, `rtl/tech/*/tsr_mac_lane.sv` | §3.3 to §3.5, §5, §7 | `tb/mvmback` |
| (top) | `tsr_mvm.sv`, `tsr_mvm_cluster.sv`, `tsr_mvm_gather.sv` | §3.0 to §3.2, §3.5, §6 | `tb/mvm` |
| (int) | `tsr_core.sv`, `tsr_core_dp.sv` (`HAS_MVM` wiring, checks) | §13.1 | `tb/core` |

A port-list change is an interface change: update this file, `tsr_pkg.sv` and `tb/common/isa.py` together. Block tests that need
several blocks put wrappers in `tb/<block>/hdl/`.

## 3. Internal interfaces

### 3.0 Dispatch (dispatcher to `tsr_mvm`)

`tsr_mvm_cmd_t` (360 bits) and the dispatch protocol are [rtl-core.md](rtl-core.md) §3.0 and §3.4. `tsr_chk` guarantees (CC §8.1)
`fmt` in {0, 1}, `0 < k <= K_MAX`, `k % bs == 0` (bs 32 or 256), `1 <= t <= T_MAX`, `y_lines = ceil(n/16)`, `dst_stride` resolved
(`w6 = 0` gives `y_lines`, and 0 when n = 0), `rows_per_ch = ceil(n/NUM_CH)`, `row_beats = ROW_BEATS(fmt, k)`, `qa_lines =
QA_LINES(fmt, k)`, `dst_line + (t-1)*dst_stride + y_lines <= SPM lines`, `x_line + t*qa_lines
<= SPM lines` and `w_off + rows_per_ch*row_beats <= DRAM limit`. It does not guarantee the absence of overlaps between dst
and x or between token dst ranges with `dst_stride < y_lines`. Results are then undefined (spec §21, "Overlapping operands"), but
the unit must still finish normally, with no hang and DONE implying quiescence.

- `cmd_ready` is a register, 1 only while the unit is idle. `done` is a registered one-cycle pulse, never in the accepting cycle,
  when every y line write has been accepted by `tsr_spm` and every requested weight beat has been received, with no WS read
  outstanding and none pending ([rtl-core.md](rtl-core.md) §3.0 "visible" and §4 M7).
- N = 0 does no SPM access and no memory traffic. `done` comes at least 1 cycle after the accept.
- N >= 1 (§6.1) sends one `tsr_mvm_ccmd_t` to every cluster and broadcasts the T QA images. Each cluster streams exactly `rows_c *
  ROW_BEATS` beats and the row results are gathered into the SPM. Clusters with `rows_c = 0` (`c*R >= N`, for example N = 5 on 4
  channels with R = 2 and channel 3 empty) do nothing.
- `perf_mvm_busy` (to `tsr_csr`, `PERF_MVM_CYCLES`) is registered and 1 from the cycle after the accepting edge up to and
  including the cycle of `done`. It is informational (spec §21, "Performance counters").

### 3.1 Broadcast pipe (`tsr_mvm` to every cluster)

Per cluster c, one `tsr_vpipe #(.VW(2), .DEPTH(BC_PIPE))` carries `{bc_cmd_valid, bc_line_valid}` and `{bc_cmd, bc_line}` with no
backpressure (the clusters accept everything, R4 on the reader side). All clusters see the same items, each delayed by its pipe.
Per MVM with N >= 1 come one `bc_cmd` (`tsr_mvm_ccmd_t`, 137 bits), then `T * QA_LINES` `bc_line` items (`tsr_mvm_bcline_t`, 527
bits) with gaps allowed, the last with `last` = 1. The two valids are never 1 in the same cycle. The next MVM's `bc_cmd` is sent
only after `done` of the previous MVM.

`tsr_mvm_ccmd_t` carries `fmt`, `k`, `n`, `rows_per_ch` (R) and `w_off` from the command, `row_beats[15:0]` (at most 1084), and
`d_lines[7:0]`, the QA D-region lines `tsr_qa_d_lines(fmt, k) = ceil(4*(k/bs)/64)` (at most 128). SPM line `x_line + t*qa_lines +
l` (one read per cycle, in this order) becomes `bc_line = {last: t = T-1 and l = qa_lines-1, tok: t, dbs: l >= qs_lines, addr: dbs
? l - qs_lines : l, data}` with `qs_lines = ceil(k/64)`. The QA layout is spec §6: QS, then D, then BS for A256, each
line-aligned.

### 3.2 Cluster setup and front command

On `bc_cmd` the cluster computes, over several cycles that overlap the broadcast, `first = CH * R` (64-bit), `rows_c =
tsr_mvm_rows_c(n, R, first) = max(0, min(R, N - first))` (spec §8, stage 1) and `beats_c = rows_c * row_beats`. `beats_c` is at
most 2^40 - 1: the dst SPM range check bounds `rows_c <= N <= 16 * (2^26 - 1)`, and the maximum 2^40 - 1 is Q8_0 K 61664 or 61696
with `ROW_BEATS` 1025. `TSR_MVM_BEATS_W` = 40 has no spare bit, and widening `SPM_BYTES` beyond 32 bits needs 41. If `rows_c >=
1`, the cluster sends one `tsr_mvm_fcmd_t {fmt, k, row_beats, w_off, rows, beats}` (137 bits) to the front end and registers
`cfg_fmt` and `cfg_d_lines` for the back end. With `rows_c = 0` it only absorbs the broadcast. Row r of the channel starts at beat
`w_off + r*row_beats` (spec §4, "Symmetric allocation"), so the channel's rows are one contiguous range of `beats_c` beats.

Activation gating. `act_ready` is 0 at `bc_cmd` and 1 at the edge that writes the `bc_line` with `last` = 1 into the activation
buffer. Front-to-back beats pass only while `act_ready` is 1. The front end starts requesting weights at once, so the DRAM latency
hides behind the broadcast.

### 3.3 Tagged beat stream (front to back): `tsr_mvm_beat_t`, 1175 bits

One item per QUANT beat, in stream order. META beats stay in the front end's meta register (spec stages 2 to 4). Stream order is
the row walk of `tsr_dot_row`: rows r = 0 to `rows_c` - 1, and per row chunks c = 0, 1, ... (Q8_0: 32 blocks = 1 meta and 16 quant
beats, Q4_K: 4 superblocks = 1 and 8 beats). The row's last chunk may be partial: Q8_0 `rem` blocks give 1 + ceil(rem/2) beats,
and Q4_K `rem` superblocks give 1 + 2*rem beats (spec §5, "Row size"). The term index i is the block b (Q8_0) or superblock s
(Q4_K).

| Field | Q8_0, quant beat j of chunk c | Q4_K, quant beat j of chunk c |
|---|---|---|
| `blk[10:0]` | b = 32c + 2j (even) | s = 4c + j/2 |
| `two` | block b+1 present: 2j+1 < blocks in chunk | 0 |
| `half` | 0 | j % 2 (1: second beat of s, the term of s completes) |
| `d_a` (f32) | `fp16_to_f32(meta[4j +: 2])` = d of b | `fp16_to_f32(meta[16(j/2) +: 2])` = d |
| `d_b` (f32) | `fp16_to_f32(meta[4j+2 +: 2])` = d of b+1, +0 if !two | `fp16_to_f32(meta[16(j/2)+2 +: 2])` = dmin |
| `sc[g]` (6 b) | 0 | `get_scale_min_k4(4*half+g, meta[16(j/2)+4 +: 12])` scale |
| `m[k]` (6 b) | 0 | the same call's min, k = 0 to 7 (both halves carry all 8) |
| `w[L]` (s8) | L < 64: beat byte L (L >= 32: 0 if !two). L >= 64: 0 | lane 32g + l: beat byte 32*(g/2) + l, low nibble for even g, high nibble for odd g (0 to 15) |
| `row_last`, `last` | last quant beat of the row, and of row `rows_c` - 1 | same |

Lane L multiplies activation K position 32b + L (Q8_0) or 256s + 128*half + L (Q4_K). Lane group g (lanes 32g to 32g + 31) is
block b + g (Q8_0, g < 2) or sub-block j = 4*half + g (Q4_K). Q8_0 block b of a chunk is at quant beat b/2, bytes 32*(b%2)
(`block_offsets`). For Q4_K, `qs[32g' + l]` low nibble is sub-block 2g' at x 64g' + l and the high nibble is sub-block 2g' + 1 at
x 64g' + 32 + l (`term_q4_K_ptr`), and beat 2s + h holds qs bytes [64h, 64h + 64). The 6-bit unpack is ggml `get_scale_min_k4`:
for j < 4, `sc = q[j] & 63` and `m = q[j+4] & 63`, and for j >= 4, `sc = (q[j+4] & 0xF) | ((q[j-4] >> 6)
<< 4)` and `m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)`, on the 12 scale bytes. Meta entries and quant bytes of absent blocks
(partial chunks) are never used, whatever the DRAM holds there. The golden model is `tb/common/mvm_model.py` `front_beats()`.

### 3.4 Activation buffer ports (`tsr_mvm_actbuf`, §5)

Write: `wr_valid` and `wr_line`, one line per cycle, no backpressure. Read: `qs_re/qs_addr` gives `qs_data[T_MAX][1023:0]` and
`dbs_re/dbs_addr` gives `dbs_data[T_MAX][511:0]`, both exactly `ACT_RD_LAT` cycles later and fully pipelined. The back end
addresses, per beat, QS word `b/4` (Q8_0, lanes 0 to 63 are half `(b/2) % 2` of the word) or `2s + half` (Q4_K, all 1024 bits), D
line `(b or s) >> 4` with element `(b or s) & 15`, and BS line `cfg_d_lines + s/2` with int16 elements `16*(s%2)` to +15.

### 3.5 Row-result streams

Back end to wrapper: `res_valid/res_ready` and `res_y[T_MAX][31:0]`, one item per row in row order, with `res_y[t] =
tsr_dot_row(fmt, row, K, QA_t)` for every t < `T_MAX`. Tokens at T or above are don't-care. Wrapper to gather, through `tsr_rpipe
#(.W(33 + 32*T_MAX), .DEPTH(RES_PIPE))` per cluster: `{res_row[31:0] = first + r, res_last = (r == rows_c-1), res_y}`.

### 3.6 SPM client 2 (`TSR_SPM_CL_MVM`)

The `tsr_spm` contract ([rtl-core.md](rtl-core.md) §5) is unchanged: fixed read latency `SPM_RD_LAT`, no backpressure, writes with
64 byte enables. The read port users are the broadcast reader and, only after the broadcast completed, the gather's ACC reads. A
cluster cannot produce a result before its `act_ready`, which follows the last broadcast line, whose read has returned. The gather
is the only write port user.

## 4. Weight-stream memory protocol (`tsr_mvm_ws`, spec stage 1)

- The source is `TSR_SRC_WS` (2) of `tsr_mem_arb` channel c, and the arbiter, its rslices and the shared `RD_OUTSTANDING` budget
  are unchanged ([rtl-core.md](rtl-core.md) §4, §10). Local id 0 for every burst (memory-side id `{2, 0}` = 8) means the memory
  returns the data in request order (M3).
- Requests go out in address order from `w_off` with `len = min(64 - addr % 64, beats_left)`, so at most 64 beats and never across
  a 4 KiB block (M1). The lens sum to exactly `beats_c = rows_c * ROW_BEATS`, with no over-fetch and no prefetch beyond the
  channel's rows (spec §21, "Performance counters", and spec §8, stage 1: "R x ROW_BEATS is an upper bound and an allocation size,
  not the amount read"). `rows_c = 0` means no request.
- Reserve before request (R4, M6). The weight FIFO has `WS_FIFO_BEATS` entries. A burst is loaded into the request register only
  when `WS_FIFO_BEATS - (beats buffered + beats requested and not received) >= len`, and that space is taken at the load.
  `rd_rsp_ready` is therefore constant 1.
- FIFO sizing. At one beat per cycle, Little's law needs `round trip + 64 <= WS_FIFO_BEATS`. The round trip is the request
  handshake at the WS port to the first data beat at the WS port, including the arbiter's 3 cycles, the SLR crossing and the DDR4
  controller, and the 64 is the burst granularity of the reservation. The default 512 sustains one beat per cycle up to a round
  trip of about 440 cycles. In-flight bursts are at most 512/64 + 1, below `RD_OUTSTANDING` (16). The cost is one `tsr_sram_1r1w`
  of 512 x 512 bits (8 BRAM36) per cluster. `tsr_mvm_ws` loads at most one burst every 2 cycles. The cluster drives the front
  end's `beat_ready` from flops (back-end ready AND `act_ready`), because inside the front end it fans out to about 2.5 K clock
  enables.
- The front end consumes one FIFO beat per cycle, META beats included. At the last beat consumed the WS source has nothing
  outstanding, and `idle` means no command, nothing outstanding and an empty FIFO.

## 5. Activation buffer organisation (`tsr_mvm_actbuf`)

Spec §8, "Resource estimate", sizes one full QA image per token (QS, D and BS) at `K_MAX` 32768. The width scales with `T_MAX`,
because the T tokens share every weight beat and are read together.

| Region | Per token | Geometry (`K_MAX` 32768) | Addressed by |
|---|---|---|---|
| QS | `QS_WORDS = ceil(K_MAX/128)` words of 1024 bits = 2 banks (even and odd QS line) of 512 bits | 256 words | QS word w = lines {2w+1, 2w} |
| DBS | `DBS_LINES = tsr_actbuf_dbs_lines(K_MAX)` x 512 bits: D lines at 0 to d_lines - 1, then BS lines | 72 lines (A256 8 + 64, A32 64) | line - ceil(K/64) |

The total per token is 256 * 1024 + 72 * 512 bits = 37,376 B = `QA_LINES(A256, K_MAX) * 64`, the spec's figure. A write is one
512-bit line per cycle into bank (tok, `addr[0]`) word `addr >> 1` (QS) or DBS line `addr` of token tok. A read is one QS word and
one DBS line per cycle for all tokens. The generic build uses `tsr_sram_1r1w` per (token, bank) and per token for DBS, with
`RD_LAT = ACT_RD_LAT`. At `T_MAX` 1 on the U200 this is 24 RAMB36 per token per cluster (§9.3). Reads and writes of one MVM never
overlap in time, and `act_ready` orders them.

## 6. `tsr_mvm`: command flow, broadcast, gather

### 6.1 Flow

1. Accept `cmd` when idle. For N = 0, `done` follows (§3.0) with nothing else.
2. Send `bc_cmd` to every cluster and start the gather with `{acc, t, n, dst_line, dst_stride, expect_ch}`, where `expect_ch[c] =
   (c*R < N)`.
3. Broadcast: SPM reads `x_line` to `x_line + t*qa_lines - 1`, one per cycle when the client port allows, and each response
   becomes one `bc_line`. Reads have no backpressure, and the broadcast pipes take one item per cycle unconditionally.
4. `done` comes when the broadcast is complete and the gather reports done (§6.3).

### 6.2 Cluster

`tsr_mvm_cluster #(CH)` has the §3.1 input, §3.2 setup and gating, `tsr_mvm_front`, `tsr_mvm_actbuf`, `tsr_mvm_back`, §3.5 result
tagging (a row counter) and the WS port pass-through. A cluster processes one MVM at a time, and the bc items of the next MVM
arrive only after its last result was taken. Clusters with `rows_c = 0` stay silent. A cluster sits in an SLR with room, and its
weights reach it through the weight-stream pipes (§14.5).

### 6.3 Gather (`tsr_mvm_gather`): partial lines, byte enables, ACC, tails

Semantics (the `ex_mvm` write loop). Token tk < T writes f32 element `n_row` of its y vector at SPM line `dst + tk*stride +
n_row/16`, lane `n_row % 16`, with value `y` or, for ACC, `fadd(y_old, y)` (spec §21, "ACC addition": old value first, which is
commutative bit-exactly). The bytes after element N - 1 in the token's last line are 0, also with ACC (spec §21, "Tail bytes").

- Each cluster c owns rows [cR, cR + rows_c), and a 16-element line may hold rows of several channels when `R % 16 != 0`. One
  assembler per cluster collects the cluster's consecutive rows that fall into the same SPM line, for all T tokens at once. The
  group is complete at a row with `n_row % 16 == 15` or `res_last`. It then emits one job per token: `{line = base_tk + n_row/16,
  lanes [lo, hi] of this cluster, tail = the group holds row N-1}` with `base_tk = dst + tk*stride`.
- A job writes its line with `spm_wr_be` = 4 bytes per lane in [lo, hi], plus every lane above hi when `tail` (data 0). Each y
  byte, tail included, is written by exactly one job of exactly one cluster per token, and nothing outside the y lines is written.
  Only the owner of row N - 1 writes the tail.
- ACC read-modify-write is hazard-free by construction. A job first reads its SPM line, with the read issued only when space for
  the response is reserved (R4), adds `fadd(old_lane, y_lane)` for its lanes (16 `tsr_fp_add` lanes, one line per cycle) and
  writes back with the byte enables above. Two jobs that share an SPM line belong to different clusters and have disjoint lane
  sets, so each lane's old value is read before the only write of that lane in any order, and no job overwrites another's lanes.
  Different tokens never share a line (`stride >= y_lines` whenever t > 1). The old values are those before the MVM, because
  nothing else writes y during the MVM. `test_mvm_model_gather_jobs` in `tb/common` checks `mvm_model.gather_jobs` in random job
  orders with random ACC read and write interleavings against the `ex_mvm` write loop.
- The job queue drains at one SPM write per cycle (ACC: one read and one write, on the separate ports). Jobs per token are at most
  `y_lines + NUM_CH - 1`. `res_ready` falls only when an assembler's job queue is full.
- `done` (a registered pulse) comes when every cluster with `expect_ch[c]` delivered `res_last`, every assembler and queue is
  empty, and the last SPM write was accepted.

### 6.4 `done` and quiescence

`tsr_mvm.done` is the broadcast complete AND the gather done. A cluster's last result needs its last weight beat, which is the
last of exactly `beats_c` requested beats, so no WS read is outstanding. No SPM read is outstanding either, since the gather's
done waits for its last write, which follows its last read. `tsr_core` asserts it in simulation (§13.1).

### 6.5 Attention modes in the cluster

The attention logic is in `tsr_mvm_att.sv` and `tsr_mvm_pv.sv`. `ATTQK` uses the front end's Q8_0 path with row and segment counters, per-slot activation read addresses and a gather line buffer
per pass row (at most 128 rows, 2 RAMB36) with a row-maximum tracker. `ATTPV` adds a 64-beat delay FIFO, a weight unit (`w = p*d`,
16 `fmul` per cycle, shared exponent alignment), a PV mode of `tsr_mac_lane` that accumulates block-floating-point products
exactly in the DSPs, a capture bank and convert and add pipes. The weight MVM is unchanged. `RB_MIN` and the throughput statements
of §7.5 and §9 apply to weight MVMs only. Both modes are on the simulator (`sw/sim/tsr_at.c`, `tsr_ref_at.c`).

## 7. Back end: numerics and microarchitecture (`tsr_mvm_back`)

### 7.1 Per-beat arithmetic (bit-exact, per token t, every t < `T_MAX`)

Q8_0 (`term_q8_0_ptr`, `dot32_i8` in `tsr_ref.c`):

```
isum_g  = sum_{L<32} w[32g+L] * qs_t[32(b+g) + L]                       g = 0, and g = 1 if two
term    = fmul( fmul(d_g, Dx_t[b+g]), i32_to_f32(isum_g) )              d_0 = d_a, d_1 = d_b
```

Q4_K (`term_q4_K_ptr`), sub-block j = 4*half + g:

```
S_j     = sum_{L<32} w[32g+L] * qs_t[256s + 128*half + 32g + L]         (q_w in [0, 15])
isum    = sum_{j<8} sc_j * S_j                     (half 0 contributes j 0 to 3, half 1 j 4 to 7)
msum    = sum_{j<8} m_j * (BS_t[16s + 2j] + BS_t[16s + 2j + 1])
term    = fsub( fmul(fmul(d_a, Dx_t[s]), i32_to_f32(isum)),
                fmul(fmul(d_b, Dx_t[s]), i32_to_f32(msum)) )             at half = 1
```

Row accumulation (`tsr_dot_row`, spec §7): 16 partials start at +0.0 per row. Term i goes to `p[i % 16] = fadd(p[i % 16], term)`
in term order, and `y = tsr_tree16(p)` with `l1[k] = fadd(p[2k], p[2k+1])`, `l2[k] = fadd(l1[2k], l1[2k+1])`, `l3[k] =
fadd(l2[2k], l2[2k+1])` and `y = fadd(l3[0], l3[1])`. A partial without terms stays +0.0. The first add of a partial is a real
`fadd(+0.0, t)`, so -0.0 becomes +0.0 and NaN is canonical. `fp16_to_f32` is exact and `i32_to_f32` is RNE (isum and msum may
exceed 24 bits).

### 7.2 MAC lanes

`TSR_MVM_GROUPS = 4` instances of `tsr_mac_lane #(.N(32), .T(T_MAX), .LAT(MAC_LAT))` per cluster give 128 lanes, and Q8_0 uses
groups 0 and 1. The activations are the QS word's 1024 bits per token for Q4_K (lane L = byte L) and the selected 512-bit half on
lanes 0 to 63 for Q8_0 (the weights of lanes 64 to 127 are 0). Sums are exact `SUM_W = 21`-bit two's complement in every
implementation. `rtl/tech/generic/tsr_mac_lane.sv` is behavioural. The Xilinx variant `rtl/tech/xilinx/tsr_mac_lane.sv` puts every
product on a DSP48E2 and packs two tokens per DSP (WP486 INT8 packing, the weight as the shared B operand): tokens 2s and 2s + 1
of lane i share slot s, with `AD = x[2s]*2^18 + x[2s+1]` in the pre-adder, `M = AD * w` and a constant 2^17 on the post-adder's C
port, which folds WP486's fabric sign correction into the DSP, so `P[33:18] = x[2s]*w` and `P[15:0] = x[2s+1]*w` exactly. An odd T
leaves its last token alone in a slot. `tb/tech` compares it with the generic lane cycle by cycle and in post-synthesis
simulation. A cluster has 128 * ceil(`T_MAX`/2) MAC DSPs, so `T_MAX` 1 and 2 both take 128. The adder trees stay in fabric. The
`MAC_LAT` - 1 internal registers are allocated in the order M, P, a first adder-tree cut, AD, the input (A, D, B) registers, then
further cuts. The U200 images use `MAC_LAT` 9, a cut after every level (§14.4 item 4).

### 7.3 Accumulators: two lanes, 8 partials each

- Two accumulate lanes per token, each one `tsr_fp_add` with its 8 partials: lane 0 holds p0, p2, ..., p14 and lane 1 holds p1,
  p3, ..., p15 (`p[k]` in lane k % 2, slot k / 2).
- Q8_0 accumulates the two terms of a beat (b even to lane 0, b + 1 to lane 1) in the same cycle. A slot is hit once per 8 quant
  beats, so two terms of one partial are at least 8 cycles apart (spec §7: "a pipelined fp adder of latency at most 8 cycles, with
  II = 1"). The time from the issue of `fadd(p, term)` to the cycle in which the updated p can be the operand of the next issue is
  at most 8 cycles, which makes `ADD_LAT <= 8` with partials in registers read in the issue cycle and written from `out_valid`.
  Q4_K accumulates one term per superblock, at least 2 quant beats apart, into lane s % 2.
- Each lane has two partial sets. Row r + 1 accumulates into the other set while row r's last adds drain. When row r's last add
  has completed, its set is copied into the tree's input registers (a snapshot) and reset to +0.0 (or marked fresh, so that the
  next first add uses +0.0).

### 7.4 Tree (background, streaming)

The tree reduces each snapshot in the fixed order of §7.1 (8 + 4 + 2 + 1 adds). Throughput, not latency, bounds it, because rows
are short for the models in use (K = 896 Q8_0 is 15 beats including META, and MoE expert rows are shorter). The structure is one
pipelined `tsr_fp_add` per tree level and token. Level 1 issues the 8 pairs of a snapshot on 8 consecutive cycles, level 2 its 4
pairs as level-1 results arrive, and so on, with a ring of snapshot and intermediate registers deep enough for every row in flight
(latency about 4*`ADD_LAT` + 15), so a new snapshot can enter level 1 every 8 cycles. Another structure is fine if it meets §7.5
and keeps the exact pair order. Each result goes to a result FIFO whose space was reserved before the row's first beat was
accepted.

### 7.5 Throughput, `RB_MIN`, stalls

- Q8_0 has 32 terms per 17-beat chunk, at most 2 terms per cycle, which the two accumulate lanes take at full rate. Q4_K has at
  most 1 term per 2 cycles.
- With `res_ready` = 1 and beats offered every cycle, the back end never drops `beat_ready` when every row has `ROW_BEATS >=
  RB_MIN`, with `RB_MIN` at most 9 (the target is 8 with the streaming tree). A row's last adds draining while the next row starts
  must not stall either. Shorter rows (Q8_0 K = 32 has 2 beats, Q4_K K = 256 has 3) may stall at row starts, with correctness
  unaffected. The tests check `ROW_BEATS = RB_MIN` for both formats, T = 1 and `T_MAX`.
- The pipeline is input register, activation read (`ACT_RD_LAT`), MAC (`MAC_LAT`), integer scale (2), i32 to f32 (2) in parallel
  with d*Dx (`MUL_LAT`, starting at the activation read), the product with f32(sum) (`MUL_LAT`), the Q4_K fsub (`ADD_LAT`) and
  accumulate (`ADD_LAT`). All stages have ce = 1. `beat_ready` falls only to reserve result space and partial-set and tree
  capacity, never to hold a beat already inside.

### 7.6 Integer widths (any SPM content is a legal QA image)

The ISA does not restrict MVM inputs to QACT output. QS may hold -128, D any f32 (NaN, Inf, subnormal, which is DAZ in fmul) and
BS any int16. The exact widths are:

| Value | Bound | Width (signed) |
|---|---|---|
| group sum (Q8_0 isum, Q4_K S_j) | 32 * 128 * 128 = 2^19 | 21 (`TSR_MVM_SUM_W`) |
| Q4_K isum | 8 * 63 * 61440 < 2^25 | 26 |
| Q4_K msum | 8 * 63 * 65536 < 2^25 (QACT BS: < 2^21) | 26 (`TSR_MVM_ISUM_W`) |

The spec table's "msum 23 bits" holds for QACT-produced BS only (§11).

## 8. fp library additions

### 8.1 `tsr_fp_add`

`tsr_fp_add #(LAT)` has the ports `clk rst_n ce in_valid a b sub` and `out_valid y`. It is bit-exact `tsr_fadd` (`sub` 0) and
`tsr_fsub` (`sub` 1) for every input (DAZ inputs, RNE in the binary32 range, FTZ of a subnormal result with its sign, canonical
NaN, IEEE signed zeros), with fixed latency `1 <= LAT <= 8`, the `ce` semantics of `tsr_fp_mul` and full pipelining. The default
LAT is the smallest that meets 300 MHz within about 6 LUT levels per stage, which is 5 (LAT 3 and 4 close timing but need 8
levels, outside the [rtl-core.md](rtl-core.md) §1 budget). `tb/fp` `test_fp_default_lats` checks that `TSR_DEF_FADD_LAT` and
`TSR_DEF_FMUL_LAT` equal the module defaults. The mandatory vectors are cancellation to exact zero (+0 for x - x, -0 + -0 = -0),
exact subnormal results flushed with sign, DAZ of subnormal operands, Inf - Inf, NaN payloads to 0x7FC00000, overflow to Inf, RNE
ties at every alignment distance and operands 2^-126 apart in exponent, plus random inputs and the `fp_sweep.cpp` harness against
`common.tsr_ref`.

### 8.2 `tsr_i32_to_f32`

The `tsr_fp_pkg` function is verified in `tb/fp` (directed, every exponent with random mantissas, and an opt-in exhaustive 2^32
sweep). It takes about 7 to 9 LUT levels as one function. The back end uses it as one stage only if timing allows, and otherwise
as a two-stage split (normalise, then round) checked bit for bit against the function.

## 9. Throughput, timing and cost

### 9.1 Throughput (T = 1, memory delivering 1 beat per cycle per channel)

The unit is weight-bound. An MVM takes `max_c rows_c * ROW_BEATS` cycles plus overhead: dispatch about 10, setup about 20, the
broadcast `T * QA_LINES`, the back-end pipeline about 30 plus the tree at most 4*`ADD_LAT` + 8, and the gather about 10 to 20. The
simulator's cycle model is `64 + T*QA_LINES + max beats`. The steady state is one cycle per weight beat. The pipes added for the
U200 (§14) add a few cycles of fixed overhead per MVM and nothing to the steady state. Rows shorter than `RB_MIN` are tree-bound,
and tiny K with many channels and tokens may be gather-bound, at one SPM line write per cycle.

### 9.2 Critical structures at the U200 core clock

1. Broadcast bus (137 + 527 bits, one source flop to `NUM_CH` clusters in three SLRs): one `tsr_vpipe` per cluster, `BC_PIPE >= 2`
   for clusters in another SLR (a flop on each side of the SLL crossing), and inside the cluster one more register in front of the
   activation-buffer RAMs, which share the write data.
2. Result bus (33 + 32*`T_MAX` bits per cluster): `tsr_rpipe` with registered valid and ready, `RES_PIPE >= 2` across SLRs.
3. Activation buffer read: a RAM output register (`ACT_RD_LAT >= 2`), then the Q8_0 half select and the lane select into the MAC
   input register.
4. MAC lanes: products and 32-to-1 adder trees. The generic lane has `MAC_LAT >= 3` (a product register, about two adder levels
   per stage), and the U200 DSP-packed lane has `MAC_LAT` 9.
5. Weight unpack: a nibble or byte select per lane into a registered `tsr_mvm_beat_t`. The parser's counters and the meta register
   select are registered before use.
6. Integer scale, i32 to f32 (§8.2) and the fp pipelines, each in registered stages.
7. Accumulator loop: an 8:1 partial read mux into the adder input register, and the adder output to the slot registers, under the
   recurrence of at most 8 cycles of §7.3.
8. Gather: 16 `tsr_fp_add` lanes for ACC. The SPM write data and 64 byte enables fan out to every SPM RAM
   ([rtl-core.md](rtl-core.md) §12).
9. Multi-cycle by construction, so not critical: `CH * R`, `rows_c * row_beats`, `dst + tk*stride`, `c*R < N`, the WS address and
   beat counters and the burst length.

### 9.3 Cost of `T_MAX`

Replicated per token in every cluster: 128 MAC products (DSP48E2, two tokens per DSP from T = 2), 4 adder trees (fabric), the
activation buffer (24 BRAM36 at `K_MAX` 32768), the integer scale (12 small multipliers), 2 `i32_to_f32`, 4 `tsr_fp_mul` (8 DSP),
4 `tsr_fp_add`, 3 x 16 x 32 partial and snapshot flops, 32 more result-bus wires and 512 more bits per gather assembler. Shared
per cluster: the front end and the weight FIFO (8 BRAM36). Per cluster, in the Xilinx build with the parameters of §14.2:

| Per cluster | `T_MAX` 1 | `T_MAX` 8 | Growth |
|---|---|---|---|
| MAC DSPs (packed from `T_MAX` 2) | 128 | 512 | 128 per pair of tokens |
| other DSPs (integer scale 12, `tsr_fp_mul` 8 per token, setup 2) | 22 | 162 | 20 per token |
| MAC lanes, fabric | 2,130 LUT / 2,212 FF | 17,040 LUT / 17,696 FF | about 2.1 K LUT, 2.2 K FF per token |
| whole cluster | 9,283 LUT / 14,331 FF / 32 RAMB36 | 57,586 LUT / 64,575 FF / 200 RAMB36 | about 6.9 K LUT, 7.2 K FF, 24 RAMB36 per token |

QACT uses 32 DSP (its 16 `tsr_fp_mul`) and 8 RAMB36, and the SPM uses 128 URAM. The attention modes add the per-cluster logic of
AT §8.

## 10. Verification

The oracle chain (spec §17) is `libtsr_ref` for the blocks and the ISA simulator for the core. Bit-exact comes first, and every
test is shown to fail on at least one deliberate mutation made in a scratch copy.

### 10.1 Infrastructure (`tb/common`)

`tb/common` holds `tsr_ref.py` (the `libtsr_ref` functions, with f32 values as bit patterns), `isa.py` (the packed layouts
`MVM_CCMD_T`, `MVM_BCLINE_T`, `MVM_FCMD_T` and `MVM_BEAT_T`, checked against `tsr_pkg.sv`), `mvm_model.py` (the golden model of
the blocks, itself verified by recomputing every term from the tagged beat stream: `front_beats`, `Cluster`, `gather_jobs`,
`apply_mvm`, `mvm_reference`, random data generators) and `mem_model.py`, whose `MemModel.src_beats[c][s]` counts read beats per
memory-side source.

### 10.2 Block tests (spec §17, "Block-level test coverage")

Each block has its own bench: `tb/fp` (the §8.1 vectors), `tb/mvmfront` (random K, partial chunks, odd N, random latency and
backpressure, the tag stream against `front_beats`, the burst rules, FIFO sizing), `tb/mvmback` (realistic and raw QA with
subnormal, Inf, NaN and signed-zero d, `T_MAX` 1 and 8, bit-exact rows, no `beat_ready` drop at `RB_MIN`) and `tb/mvm` (`tsr_mvm`
with real clusters or stubs: broadcast order, N = 0, empty clusters, shared lines, tails, ACC, strides with gaps, identical
results for `BC_PIPE` and `RES_PIPE` 0, 1 and 3, and `done` timing).

### 10.3 Trace sets

`traces/gen` and `traces/m3` (`traces/m3/regen.sh`) cover the MVM at `T_MAX` 1 and 8 on 1, 3, 4 and 16 channels, with `errors`
sets that end in every MVM fault. The simulator and the RTL core replay them bit-exactly.

### 10.4 Integration (`HAS_MVM` = 1)

The core-level bench replays `traces/m3` and `traces/gen` into the issue FIFO under the memory regimes of
[rtl-core.md](rtl-core.md) §9 with PERF checks, parameter variants and throughput checks. The checks of §13.1 run in every test.

## 11. Contract notes

- The MVM accepts any byte of the SPM as QA. If BS is an arbitrary int16, the Q4_K msum needs a signed 26 bits, and 23 bits for a
  BS made by QACT (spec §7, "Integer widths").
- `LANES = BEAT_BITS / 4`, the number of weights in one Q4_K beat, is not an independent parameter. `CFG0` reports it.
- `PERF_MVM_CYCLES` counts the cycles in which the MVM unit holds an instruction, accept to `done`.
- An N = 0 MVM does not access x. Rows shorter than `RB_MIN` stall at row starts, which affects performance only.
- The front end absorbs META beats. The tag set of stage 2 is {QUANT data, block or superblock, half, row end, MVM end}.

## 12. Reference map

The reference functions are in `sw/sim/tsr_ref.c` and `tsr_sim.c`: the fp rules (`daz_bits`, `fin`, `tsr_fadd`, `tsr_fsub`,
`tsr_fmul`), geometry (`chunk_beats`, `tsr_row_beats`, `block_offsets`, `tsr_rows_per_ch`, `tsr_row_location`),
`tsr_get_scale_min_k4`, the QA layout (`tsr_qa_*lines`, `tsr_qa_split`), the terms (`term_q8_0_ptr`, `dot32_i8`, `term_q4_K_ptr`),
`tsr_tree16`, `tsr_dot_row`, and the MVM checks and execution (`chk_mvm`, `ex_mvm`). The issue-path checks are EX §8.1.

## 13. Integration and core-level verification

### 13.1 `tsr_core` with `HAS_MVM` = 1

- `HAS_MVM` defaults to 1, because the U200 images execute llama.cpp graphs, which need `MVM`. With `HAS_MVM` 0 the unit is tied
  off (`g_nomvm`), and `rtl/lint.sh` lints both values.
- Wiring ([rtl-core.md](rtl-core.md) §10): `tsr_mvm` on the dispatch port, SPM client `TSR_SPM_CL_MVM`, the weight-stream source
  `TSR_SRC_WS` of arbiter c to cluster c, and `perf_mvm_busy` to `tsr_csr`. With `HAS_MVM` 1, elaboration requires `LANES == 128`,
  `MVM_WS_FIFO` a power of two of at least 64, `MVM_ADD_LAT` 1 to 8, `MVM_WS_RD_LAT` 1 to 4 and the other latencies at least 1.
- Simulation checks (`ifndef SYNTHESIS`, armed by the first reset edge) extend [rtl-core.md](rtl-core.md) §12:
  - I1 A unit pulses `done` only while it holds an op, never in its accepting cycle. The MVM `cmd_ready` is low while it holds the
    op, and `perf_mvm_busy` equals "holds an op" in every cycle.
  - I2 A unit drives SPM requests only while it holds an op (DMA and QACT may request in the accepting cycle, the MVM strictly
    after it). MVM SPM read responses and weight-stream traffic occur only while the MVM holds the op, and WS `rd_rsp_ready` is
    constant 1.
  - I4 At `mvm_done` no WS request is offered, WS beats requested equal received on every channel, and no MVM SPM request, write
    or read is outstanding. Per MVM and channel exactly `rows_c * ROW_BEATS` beats are requested, never more at any time, at the
    next address with `len = min(64 - addr % 64, beats left)` (§4).

### 13.2 Core-level verification

The core with the MVM unit replays every trace set of `traces/` that contains an MVM and random programs, under four memory
regimes, at the default and at the U200 parameters, on 1, 2, 3, 4 and 16 channels and with the implementation parameters at their
extremes. Every run is bit-exact against the simulator: DRAM, the whole SPM, the PERF counters, the per-source read beats at the
port and quiescence at `done`. The throughput tests use an ideal memory and a two-point method, and check a steady state of 1.0000
cycles per beat and the fixed overhead per MVM against the cycle model. The tests also cover chains (MVM to ST, MVM to MVM with
ACC, MVM to QACT, LD to MVM) and hardware resets under [rtl-core.md](rtl-core.md) §11, also inside MVM programs with weight reads
in flight.

### 13.3 Mutation campaign

Mutants of the MVM unit and its wiring in `tsr_core` are applied to scratch copies of `rtl/` and run against the core-level
replays and the directed tests. Every mutant is killed except three that are equivalent: `cmd_ready` masked in the done cycle
(already 0 there, I1), WS `rd_rsp_ready` tied to 1 (constant 1, §4, and I2 asserts it) and a broadcast pipe one stage deeper than
`MVM_BC_PIPE` (parameters change timing only). The mutants cover the weight-stream request length and over-fetch, the PERF input,
the SPM client, the broadcast stride and `last`, the gather's expected clusters, `done` timing, `rows_c` off by one, the
activation gate, the term order and partial-lane assignment, the `msum` width, the QS half select, the tail bytes and byte
enables, and the ACC read-modify-write. The mutant that uses the absent Q8_0 block b + 1 is killed only by tests with raw weight
windows, because repacked rows hold zeros there.

## 14. Implementation for the U200

This section records the structures that make the core meet timing on the U200. Every change here is bit-exact and only latencies
change. They were found in out-of-context runs of `tsr_core` on the U200 part with the Xilinx target (`rtl/filelist_xilinx.f`:
URAM, BRAM and DSP-packed technology wrappers), which do not include the shell, the AXI bridges, the DDR4 controllers and the
memory subsystem. The linked design adds them ([platform-u200.md](platform-u200.md) §8, §9).

### 14.1 Flow

The scripts are in `tb/core/viv`. `core_ooc.tcl` runs synthesis, `opt_design`, `place_design`, `phys_opt_design`, `route_design`
and an optional post-route `phys_opt_design`, `core_slr.tcl` applies a Pblock plan ([platform-u200.md](platform-u200.md) §8.1),
and `core_ooc.sh <k1|k4t1|k4|k4min> <outdir> [args]` runs the configurations with the image parameters (`TSR_VIVADO_LOCK`
serialises runs). The part is `xcu200-fsgd2104-2-e`, the clock 3.333 ns, and every I/O is budgeted 1.6 ns outside.

### 14.2 U200 bitstream parameters

| `tsr_core` parameter | U200 | Generic default | Why |
|---|---|---|---|
| `SPM_RD_LAT` | 4 | 2 | URAM output registers ([rtl-core.md](rtl-core.md) §12) |
| `MVM_BC_PIPE` | 2 | 1 | broadcast and weight-response pipes: one register on each side of an SLR crossing |
| `MVM_RES_PIPE` | 2 | 1 | result, weight-request and gather job and start pipes: the same |
| `MVM_MAC_LAT` | 9 | 3 | a register after every adder level of the DSP-packed MAC lane (§14.4 item 4) |
| `RST_PIPE` | 3 | 0 | reset register stages with per-block copies (§14.4 item 3) |

All other parameters keep their `tsr_pkg` defaults (`CH_BYTES` 16 GiB, WS FIFO 512, `ACT_RD_LAT` 2, `MUL_LAT` 4, `ADD_LAT` 5,
`WS_RD_LAT` 2). Every value is an implementation parameter, so results are identical and only latencies change.

### 14.3 Critical structures

The configurations are `k1` (`NUM_CH` 1, `T_MAX` 1), `k4t1` (`NUM_CH` 4, `T_MAX` 1) and `k4` (`NUM_CH` 4, the image's `T_MAX`).
With the structures of §14.4 and §14.9 in place, these bound timing:

- the SPM URAM inputs, the read and write offset registers into the cascade address and enable and the shared write data into the
  cascade data input (§14.9 item 1);
- activation write data to the BRAMs, and the back-end QS address to the activation BRAMs;
- the result-pipe slices' `rdy_q` into the payload registers' clock enables, and the result pipe into the assembler buffer;
- SLR crossing landings: the broadcast pipe's last stage into the cluster command register, and the assembler's select into the
  job-pipe stage 0;
- the back-end control fan-out (tree level-1 operand select, accumulate clock enables, `fmt_q`, §14.9 item 3) and the QACT
  multiplier operands;
- the unit command-register clock enables, driven by the dispatcher's queue head (CC §9.1), the SPM write grant into the per-slice
  byte-enable copies, and the per-part reset copies, which phys_opt may replicate or rewire between equal copies.

### 14.4 Structural decisions

1. Broadcast pipes are kept per cluster. The four per-cluster broadcast `tsr_vpipe`s have one source, and synthesis would merge
   them into one pipe, so that one cluster's registers would drive all four clusters across three SLRs. `tsr_vpipe` has the
   parameter `KEEP_REGS` (Xilinx `keep` and `shreg_extract = "no"` on the stage registers), used for the broadcast and
   weight-response pipes and the gather's start pipes. A destination cluster's registers never drive another SLR's cluster. The
   first stage is shared when `BC_PIPE` is at least 2 (`u_bc_head`), and each cluster keeps a tail of `BC_PIPE` - 1 stages beside
   its own decode. All reset copies have identical INIT and shift recurrence, so total latency and reset flush match independent
   pipes. `BC_PIPE` 0 and 1 keep the independent path. The parameter is not named `KEEP`, because Vivado reads a parameter of that
   name as the instance's KEEP property.
2. The gather job select is split. A single `NUM_CH * 2 * T_MAX : 1` select of the assembly buffers with select lines that fan out
   to every buffer flop is too slow and congests the router. `tsr_mvm_gather` is one assembler per cluster (`tsr_mvm_gasm`: the
   buffers and everything that grows with `T_MAX`, where a job is a `2 * T_MAX : 1` select of flops with registered selects) and a
   central part (round robin over the `NUM_CH` job-pipe heads, credits, ACC, write queue), with `JOB_PIPE` register slices in
   between (`tsr_mvm` passes `RES_PIPE`) and a kept start pipe per assembler. The Pblock plan puts each assembler next to its
   cluster, so only assembled jobs cross to the centre. The port list is unchanged, and with `JOB_PIPE` 0 the cycle behaviour at
   the ports is the unsplit one.
3. Reset is distributed (`RST_PIPE`). A single `rst_n` net to every reset flop of a core spread over three SLRs cannot meet
   timing. With `RST_PIPE` P of at least 1, every flop sees `rst_n` exactly P cycles late, through P - min(P, 2) shared stages and
   min(P, 2) kept stages per block (CSR, DMA, QACT, SPM and each arbiter, and inside `tsr_mvm` the parameter `RST_LOCAL` for the
   top, the gather and every cluster with its pipes). The first per-block copy can sit at the SLR boundary, and the second,
   replicated by phys_opt, at the loads. Inside a cluster the last stage is made of copies, one per token of the back end, one for
   the front end and one for the rest (`tsr_mvm_cluster` `RST_COPY`, set by `tsr_mvm` when `RST_LOCAL` > 0), because one register
   per cluster would drive about 700 pins over a whole SLR. The reset is cycle-identical to the single stage. The core is a
   P-cycle delayed copy of the `RST_PIPE` 0 core, the registers start in reset (INIT 0), the outputs go quiet P cycles after
   `rst_n` falls, and `rst_n` must stay low for at least P + 1 cycles ([rtl-core.md](rtl-core.md) §11). The reset-stage registers
   load in a plain `always @(posedge clk)` next to their `initial ... = '0`, because `always_ff` with an initial value is illegal
   by IEEE 1800 9.2.2.4 and rejected by XSim.
4. MAC latency. Vivado absorbs the MAC lane's output register into the integer-scale multiplier's AREG, so the last tree segment
   would end at a DSP A port. `MAC_LAT` 9, a register after every adder level, leaves no MAC path among the critical ones. The
   cost is about 5.9 K FF per cluster at `T_MAX` 8 over `MAC_LAT` 7.
5. Write registers load every cycle. The cluster's activation write register `aw_q` (527 flops), the actbuf's address and data
   registers and the gather's output register `out_q` (592 flops) load every cycle instead of only with a valid line, so they have
   no large clock enable. Every use is qualified by the registered valid and write enables, and `out_q` holds the same entry
   whenever it is valid.
6. SLR crossings carry no logic on either side. The weight-stream pipes in `tsr_mvm` per cluster (requests through `tsr_rpipe` of
   `RES_PIPE` slices, responses through kept `tsr_vpipe` of `BC_PIPE` stages, legal because the streamer's `rd_rsp_ready` is
   constant 1) let the arbiters sit where the Pblock plan puts them and cross only pipes. The result pipe runs cluster to
   assembler inside one SLR.

### 14.5 U200 budget and cluster placement

The xcu200 is a VU9P of three SLRs, with 17,280 SLLs per SLR boundary. In the platform DDR[0] is in SLR0, DDR[1] and DDR[2] in
SLR1 and DDR[3] in SLR2, and SLR1's dynamic region is the clock-region columns X0 to X2, the rest being the static shell
([platform-u200.md](platform-u200.md) §8). The DDR4 controllers and memory interconnect that `v++` adds cost about 30.5 K LUT,
42.5 K FF, 61 RAMB36 and 3 DSP per bank in its SLR. Clusters are too large to sit beside their DDR controllers in SLR1 (two
`T_MAX` 8 clusters need 2 * 674 + 32 (QACT) = 1,380 DSP of 1,320), so the clusters of channels 0 and 2 sit in SLR0 and those of
channels 1 and 3 in SLR2, with the SPM, central gather, DMA, QACT and CSR in SLR1. Per cluster the crossing signals are the
broadcast pipe (529 bits), the weight-stream response pipe (516 bits), the gather job pipe (561 bits), the start pipe (68 bits),
the weight-stream request pipe (37 bits) and the reset copies. The AXI side of channels 0 and 3 crosses between SLR1 and their DDR
SLR in the adapter's AXI slices, and channels 1 and 2 stay in SLR1. The Pblock plan of the release image is
[platform-u200.md](platform-u200.md) §8.1.

### 14.6 Choosing `T_MAX`

`T_MAX` 1 to 8 uses the same RTL and parameters. A larger `T_MAX` speeds up prefill, because the T tokens share one weight pass,
at a cost in LUT, DSP and RAMB36 that grows almost linearly (§9.3). URAM does not grow. The release image uses `T_MAX` 4, and `k1`
stays a single-SLR design (SLR1, DDR[1]).

### 14.7 Verification of the structures

The block benches (`tb/mvm`, `tb/spm`, `tb/tech`, `tb/mvmback`) and the core-level replays of §13 run with the U200 parameters on
`rtl/filelist_xilinx.f` and with `--reset-each`. `rtl/lint.sh` lints `tsr_core` with `RST_PIPE` 0, 1, 2, 3 and 5, `tsr_mvm` with
`RST_LOCAL` 1 and 2, and the gather with `JOB_PIPE` 0 to 3, and XSim elaborates them. With `RST_PIPE` 3 the core-level reset tests
check "outputs quiet" from reset cycle `RST_PIPE` and use pulses of at least `RST_PIPE` + 1 cycles.

### 14.8 Known weak spots

`tsr_rslice`'s `rdy_q` drives every `skid_q` clock enable and the `main_q` select directly, which makes it the worst structure
inside the bridges and the register slices ([platform-u200.md](platform-u200.md) §8.2). The central consumer of the gather's job
pipes takes a wide `m_ready` (`rtl/core/tsr_mem_arb.sv`), and FORM 1 slices with pointer copies are used on the last stage of
every job pipe ([platform-u200.md](platform-u200.md) §8.10). Job-pipe stage-0 registers of assemblers may be placed in SLR1
outside their Pblock when no leaf cell is fixed. Each SPM address bit drives 32 per-slice copies (§14.9 item 1), and one copy per
two slices is the untried middle option. The cluster's `fmt_q` (about 3,100 loads) stays shared. Two `keep` copies of a reset may
merge, and phys_opt replicates a cluster reset copy at its loads. The core's reset copies are in no Pblock, and copies that must
stay apart want `DONT_TOUCH`. `RST_PIPE` of 1 or more changes the core's reset boundary behaviour, so in the kernel `ap_rst_n`
must stay low for at least `RST_PIPE` + 1 = 4 cycles ([platform-u200.md](platform-u200.md) §5).

### 14.9 Margin structures

All of these are structural. No parameter and no latency changes, and results and cycle counts equal those of the structure
without them.

1. SPM column slices (`rtl/core/tsr_spm.sv`). Each depth bank is 8 column slices of 64 bits (at 4 MiB, 8 `tsr_sram_1r1w` of 16 K x
   64, each one a 4-deep URAM cascade, still 128 URAM). Every (bank, slice) RAM has its own `keep` copy of the input registers
   (read and write enable and offset, byte enables), with the bank enables decoded before the register, so each cascade head is
   driven by a flop of its own (fan-out 1 instead of 32). Only the write-data register stays shared per bank, and the last
   bank-select stage has one copy per slice. The Xilinx wrapper's URAM rule is `DEPTH >= 4096` and `DEPTH * WIDTH >= 4096 * 64`,
   so the 256 KiB SPM's 4 K x 64 slices stay in URAM. One address copy per bank, `cascade_height = 2`, 8 K-line banks and
   `max_fanout` on the cascade heads were measured and rejected.
2. `tsr_rslice` payload enables and reset copies per bridge part ([platform-u200.md](platform-u200.md) §8.2).
3. Back-end control copies (`tsr_mvm_back`). Every token has its own registered copies of the shared accumulate and tree control
   (slot write enables and clear decodes, per-lane bypass flags, issue tags, level-1 selects, hold enables), 83 `keep` flops per
   token. Their next values are computed once, so each copy equals the signal it replaces in every cycle, also in reset, and a
   simulation assertion checks every copy.
4. `tsr_vpipe` names its keep parameter `KEEP_REGS` (§14.4 item 1).


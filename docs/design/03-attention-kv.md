# AT: attention and the KV cache on the device

Area AT specifies how attention runs on the device: the KV cache layout, the seven attention ops, their
bit-exact numerics, the flash-attention kernel sequence and the mask summary. It implements D17: the KV
cache lives on the device in q8_0, and attention arrives as llama.cpp `FLASH_ATTN_EXT`. The other areas
are EX (execution model), VU (vector unit), CC (control core), DB (debug) and SW (software and
verification). llama.cpp references are relative to `third_party/llama.cpp`. Numbers marked estimate are
design figures.

---

## 1. Summary

- Attention runs in the MVM clusters, not in the VU: one SPM port moves 64 B per cycle while the four DDR
  channels deliver 256 B per cycle, so only the clusters, each streaming its own channel, can consume the KV
  cache at the DDR rate (§3).
- The KV cache holds exactly llama.cpp's q8_0 values (the same 32-element blocks, fp16 scales and
  `quantize_row_q8_0_ref` rounding); only the byte placement changes, so `get_tensor` and `set_tensor` convert
  losslessly (§4.6). Cells form 16-cell groups, group u living in channel u mod C. K planes are token-major, V
  planes slice-major (64-value slices of all cells are contiguous); both use the Q8_0 chunk format (§4).
- Seven ops in the opcode blocks EX reserves for AT: `KVST` 0x14 (DMA unit), `ATTQK` 0x21 and `ATTPV` 0x22 (MVM
  unit, scores and P·V over a group range), and `ASMAX` 0x34, `AMASK` 0x35, `AMERGE` 0x36, `ANORM` 0x37 (VU
  unit, executed by the VU's lanes, VU §5.11). Unit 4 is absent (§5).
- Numerics. Scores use the spec's Q8_0 row rule on each head segment. P·V uses an exact integer accumulation
  with a 24-bit block-floating-point weight per 64-cell segment and one f32 addition per segment. The softmax
  is two-pass within a chunk and online across chunks (§6).
- GQA. Query heads of one KV head (and up to `T_MAX` query tokens in prefill) are the MVM's token slots. Head
  subsets cover G > `T_MAX` (§5.2): a global layer with G = 16 takes 16 / `T_MAX` passes per decode token (four
  in the release image).
- Masks. The host backend reduces llama.cpp's f16 KQ mask to a mask summary of cell runs (§7.3); the device
  never reads f16. Causal and sliding-window prefill tiles use per-row bounds inside `ATTQK`; anything else
  takes a general f32 path.
- Not supported: attention logit softcap, ALiBi, sinks, sparse `n_kv_max`, DK ≠ DV (MLA), head dimensions other
  than 64–512, and K-shift. The KV Hadamard rotation is off (§2.5). Attention never uses DQROW (§5.9).

---

## 2. What llama.cpp hands the device (pinned commit)

### 2.1 Graph shape

Qwen2.5 (`src/models/qwen2.cpp`): Q, K and V projections with bias, RoPE NEOX on Q and K, then `build_attn`
with kq_scale = 1/sqrtf(n_embd_head).

Gemma 4 (`src/models/gemma4.cpp`): embedding × √n_embd; Q RMSNorm per head with weight; K RMSNorm per head
with weight and V `ggml_rms_norm` without weight; V = K's projection when `wv` is absent; RoPE with
`freq_factors` on full-attention layers; attention scale 1.0; an iSWA cache of two caches, the SWA one sized
`pad(min(size, n_swa·n_seq + n_ubatch), 256)`; different head dimensions on SWA and full layers. There is
no attention softcap: `attn_soft_cap` is never set for Gemma 4.

`build_attn` with a KV cache (`src/llama-graph.cpp`): optional Hadamard rotations of Q, K and V; `cpy_k`
and `cpy_v`, which are `SET_ROWS` into the caches; the `get_k` and `get_v` views; `build_attn_mha`; an
optional output un-rotation; `wo`. The FA branch of `build_attn_mha` permutes q, k and v, calls
`ggml_flash_attn_ext(q, k, v, kq_mask, kq_scale, f_max_alibi_bias, softcap)`, applies sinks, `n_kv_max` and
F32 precision, and reshapes the result to `[n_head·D, n_tokens]`.

### 2.2 KV cache tensors

- `k` and `v` are 3D `[E, kv_size, n_stream]` with E = n_head_kv·D, type q8_0, named `cache_*k_l<il>` and
  `cache_*v_l<il>` (`src/llama-kv-cache.cpp`). The buffer is cleared to zero after allocation.
- With FA, V is not transposed, so both caches have one q8_0 row per cell. The FA views `get_k` and `get_v`
  are `[D, n_head_kv, n_kv, n_stream]` with row stride `row_size(E)`. `n_kv` is padded to max(n_pad, 256).
- The pinned commit refuses a quantized V cache without FA (`src/llama-context.cpp`).

### 2.3 `FLASH_ATTN_EXT` semantics (CPU reference)

API (`ggml/include/ggml.h`): q `[D, n_batch, n_head, ne3]`, k and v `[D, n_kv, n_head_kv, ne3]`, mask
`[n_kv, n_batch, ne32, ne33]` in F16, result `[D, n_head, n_batch, ne3]` permuted so each token's heads are
contiguous. GQA: kv head = q head / (n_head / n_head_kv), so consecutive query heads share a KV head.

For q8_0 K and V the CPU takes the per-row `one_chunk` path (`ggml/src/ggml-cpu/ops.cpp`). Per query row:

1. Q is quantized with the K type's `vec_dot_type` (q8_0, x86 `quantize_row_q8_0`).
2. For each cell: if the mask is −inf, the cell is skipped. Otherwise `s = vec_dot_q8_0_q8_0(k, q)·scale`,
   with a softcap if one is set, and the mask value is added.
3. Online softmax with `expf`.
4. V is dequantized to f32 and accumulated as `VKQ += v·expf(s − M)` in f32.
5. Finally `S_inv = S == 0 ? 0 : 1/S`.

### 2.4 `SET_ROWS` and the mask

- `SET_ROWS` quantizes each f32 source row with the destination type's `from_float` (`quantize_row_q8_0`)
  into row `idx[i]` (I64).
- The mask is F16 `[n_kv, n_tokens/n_stream, 1, n_stream]`. Its values are 0 or −inf. Future positions
  (p0 > p1), other sequences and empty cells are masked. SWA standard masks p1 − p0 ≥ n_swa
  (`src/llama-hparams.h`).

### 2.5 Hadamard KV rotation

With a quantized cache, uniform head dimensions in the cache and head_dim % 64 = 0, llama.cpp rotates Q and
K by an orthonormal Walsh-Hadamard matrix of size n_rot_k, and V by one of size n_rot_v = 64
(`src/llama-kv-cache.cpp`), and un-rotates the attention output. The rotation is a `MUL_MAT` with an F32
graph-input src0 and the hint `GGML_HINT_SRC0_IS_HADAMARD`, which the CPU runs as a fast Walsh-Hadamard
transform. Every device run and every baseline uses `LLAMA_ATTN_ROT_DISABLE=1` (SW-16, SW §8 R4; AT-17).
The device attention is agnostic to rotation: it computes on whatever q, k and v the graph supplies.
Appendix A records the bit-exact FWHT definition, for a later VU function.

### 2.6 What the device accepts (backend `supports_op` contract)

| Op | Device accepts when | Otherwise |
|---|---|---|
| `FLASH_ATTN_EXT` | q F32 contiguous per token (`[D, n_head, T]` before the permute); k and v q8_0 views of KV-cache tensors in a Tesseract KV buffer (§4.6); D = DK = DV ∈ {64, 128, 256, 512}; n_head % n_head_kv = 0; mask F16 or none; `max_bias` = 0, `logit_softcap` = 0, no sinks, `n_kv_max` = 0; precision default or F32; ne3 = n_stream (1 in M7 mode). k and v start at the plane's cell 0 with cells `row_size(E)` and heads `row_size(D)` apart (llama.cpp's `get_k` and `get_v`); T ≤ 680, so that the MSUM and the FA_NEW indices fit one list's 256 preload lines (SW-43; a longer ubatch's FA runs on the CPU) | false. With a device KV buffer the scheduler would then copy the cache per op, so the backend reports the unsupported feature at graph build (SW R10) |
| `SET_ROWS` | dst is a KV-cache tensor (q8_0, Tesseract KV buffer) viewed from cell 0 with rows of E values (`cpy_k` and `cpy_v`, one stream); src F32 with rows of E; idx I64 < 2^31 | false; the scheduler then stops on the pre-allocated cache tensor, after the backend names the cause (SW R10) |
| Hadamard `MUL_MAT` | never (rotation is off, AT-17) | |
| Other attention ops (`SOFT_MAX`, attention `MUL_MAT` on the cache) | never (D17: FA only) | |

The KV type must be q8_0 for both K and V when the cache is offloaded. Other KV types are unsupported (run
with `-nkvo`).

---

## 3. Why the clusters, not the VU

The core's SPM port moves 64 B per cycle while the four channels deliver 256 B per cycle at 97–98 % beat
efficiency, so any scheme that loads K and V tiles into SPM reaches at most 25 % of the DDR rate. The VU
processes 16 f32 per cycle against about 60 values in a q8_0 beat. Each cluster consumes its own channel at one
beat per cycle, and its MAC DSPs are idle while the decode token is in its attention phase: attention needs
about G multiply-adds per KV byte for Q·Kᵀ and G for P·V (G is the GQA ratio, at most 8 for the target models),
and 4 channels × 64 values × 8 = 2,048 per cycle is what the clusters' MAC lanes already provide. So Q·Kᵀ is a
segmented form of the existing Q8_0 MVM (K rows stream as weights, query heads are token slots), P·V is a new
accumulate mode of the same lanes (§8), and the VU only does the O(rows × cells) softmax pass and the
O(rows × D) merges.

---

## 4. KV cache layout in device DRAM

### 4.1 Terms

| Term | Meaning |
|---|---|
| cell | a KV slot (llama.cpp cell index j, 0 ≤ j < kv_size), one token |
| E | n_embd_gqa = n_head_kv · D: values per cell row. E % 64 = 0, 64 ≤ E ≤ K_MAX |
| group | 16 consecutive cells: group u = cells [16u, 16u + 16) |
| plane | the device image of one KV tensor (one layer, one stream, K or V) |
| C | NUM_CH (4 on the U200) |
| LG | local group capacity per channel: LG ≥ ⌈kv_size / (16·C)⌉ |
| GB | beats per group per plane = 17·E/64 |
| slice | 64 consecutive values of a cell row (2 q8_0 blocks): slice s = elements [64s, 64s+64), 0 ≤ s < E/64 |

### 4.2 Channel interleave

Group u lives in channel c = u mod C at local index ℓ = ⌊u / C⌋. Each plane occupies the same beat range
[B, B + LG·GB) in every channel (symmetric allocation, [spec](../spec.md) §4). A run of n groups is therefore
split over the channels to within one group. A 16-cell group gives exactly one 64-byte line of f32 scores
per head row (§5.2), so every score line comes from one cluster.

### 4.3 K plane (token-major)

Within a group, the 16 cell rows are concatenated in cell order as a sequence of 16·E/32 q8_0 blocks. That
sequence is cut into Q8_0 chunks of 32 blocks (spec weight layout, spec §5: one meta beat with 32 fp16
scales, then 16 quant beats). For cell r = j mod 16 and block b of its row (0 ≤ b < E/32):

```
β = r·(E/32) + b;   κ = β >> 5;   o = β & 31
meta : beat B + ℓ·GB + κ·17,               bytes [2o, 2o+2)   = fp16 scale d
quant: beat B + ℓ·GB + κ·17 + 1 + (o>>1),   bytes [32(o&1), +32) = q[32]
```

For E = 1024 (Qwen2.5-14B and 32B) a cell row is exactly one chunk (17 beats). For E < 1024, 1024/E rows
share a chunk; for E > 1024, a row spans E/1024 chunks.

### 4.4 V plane (slice-major)

For slice s of a cell in local group ℓ, the chunk at beat B + (s·LG + ℓ)·17 holds the 16 cells' slice s:

```
meta : bytes [4r, 4r+4)  = fp16 d of blocks 2s and 2s+1 of cell r
quant: beat (chunk + 1 + r), all 64 bytes = q[64s .. 64s+63] of cell r
```

So a slice-s region of a channel, [B + s·LG·17, B + (s+1)·LG·17), is contiguous over all its groups. That is
what `ATTPV` streams (§8.2). The plane size per channel is (E/64)·LG·17 = LG·GB beats, the same as K.

### 4.5 Region, alignment, capacity

- Region. KV planes come from the heap above 512 MiB that EX defines (EX §11). They are allocated
  symmetrically, top-down from the end of each channel. Weights grow bottom-up from 512 MiB. The allocator
  places one plane per (layer, stream, K or V).
- Alignment. B is beat-aligned. 64-beat (4 KiB) alignment is recommended, not required; the requester splits
  bursts at 4 KiB anyway ([spec](../spec.md) §2, Core boundary interface).
- Capacity per cell: the sum over layers of 2·E·34/32 bytes. Examples: Qwen2.5-0.5B (D 64, n_head_kv 2, 24
  layers) 6,528 B per cell; Qwen2.5-32B (D 128, n_head_kv 8, 64 layers) 139,264 B per cell, 4.56 GB at 32K
  cells (1.14 GB per channel); Gemma 4 12B 182,784 B per cell over all 48 layers.

Gemma 4 12B has a 1,536-cell SWA cache (pad(min(size, n_swa·n_seq + n_ubatch), 256) with n_swa 1024 and
n_ubatch 512). Its global layers have E = n_head_kv·D = 512 < 1024, so two cells share a K chunk and every K
meta beat that KVST touches is read-modify-written (§5.1). They have no `attn_v`: V is the K projection,
normalized without weight, and is stored in its own V plane like any other V.

### 4.6 Correspondence with ggml, identification, clearing

- Identification. The backend treats a tensor as a KV-cache tensor when it is q8_0, 3D
  `[E, kv_size, n_stream]` in a Tesseract buffer, named `cache_*k_l<n>` or `cache_*v_l<n>`. Its layout is
  fixed by E and kv_size, both known at `init_tensor`, and does not depend on D. The tensor is recognized
  from the `init_tensor` records, and the planes are placed at the buffer's first data access (llama.cpp's
  clear), because ggml sets a buffer's usage after `init_tensor`; a buffer counts only if all its tensors
  qualify (SW-47).
- Clear. The plane is zeroed on clear: zero scales and zero quants, which is also the ggml image.
- Lossless conversion. `set_tensor` and `get_tensor` on row ranges (state save and load) convert between
  ggml rows (`block_q8_0 {fp16 d, int8 qs[32]}`, 34 B) and the plane by the address functions of §4.3 and
  §4.4. This is a pure byte permutation, so state files are bit-identical to the CPU's.
- Shared references. `tsr_kv_addr_k`, `tsr_kv_addr_v` and `tsr_kv_{pack,unpack}_row` live in
  `sw/sim/tsr_ref_at.c` (`include/tsr_ref_at.h`, part of `libtsr_ref`) and are shared by the simulator, the
  backend and the RTL package.

---

## 5. Operations

All seven ops follow the common encoding: w0 = {x16 = w0[31:16], flags[15:12], sub[11:8], opcode[7:0]} (EX
§7.1). Every reserved field or bit must be 0, otherwise `ILLEGAL_OP`. Every error is found at dispatch;
nothing fails after it has started. Data-dependent conditions (NaN, empty rows, zero weights) have defined
results.

Shape fields shared by `ATTQK` and `ATTPV`, and the derived values used throughout:

| Symbol | Encoding | Meaning |
|---|---|---|
| E | w4[8:0] = E/64 (1–511: the 9-bit field; E ≤ 32,704) | row values per cell |
| D | w4[10:9] = dcode, D = 64·2^dcode | head dimension (64, 128, 256, 512) |
| T_q | w4[13:11] + 1 | query tokens in the pass |
| GS | w4[16:14] + 1 | heads per KV head in the pass |
| G | w0[21:16] = x16[5:0] = G − 1 | heads per KV head (GQA ratio, 1–64) |
| hg0 | w0[27:22] = x16[11:6] | first head (within each KV-head group) of the pass |
| n_head | G·E/D | |
| R | T_q·n_head | rows of the score and output matrices |
| row r(q, g, i) | q·n_head + g·G + hg0 + i | for q < T_q, g < E/D, i < GS |
| slot t(q, i) | q·GS + i | MVM token slot, < T_MAX |

The rows touched by one op are its pass rows. Rows of other passes (other hg0) are left unchanged.

### 5.1 `KVST`: store K or V rows into a plane (0x14, unit 0 DMA)

| Field | Bits | Meaning |
|---|---|---|
| sub | w0[11:8] | 0 K plane, 1 V plane; 2–15 `ILLEGAL_OP` |
| x16 | w0[21:16] = x16[5:0] | n_rows − 1 (n_rows 1–64); x16[15:6] = w0[31:22] = 0 |
| w1 | | src: first QA(A32, E) vector (SPM line) |
| w2 | | plane base B (beat, every channel) |
| w3 | [8:0] E/64, [15:9] 0, [31:16] LG | |
| w4 | | c0: first cell |
| w5 | | src stride in lines (0 → QA_LINES(A32, E)) |
| w6, w7 | | 0 |

Semantics: for i < n_rows, cell c0 + i receives row i. The source is QA vector i at src + i·stride. Its q
bytes are QS[0..E) and its scales are fp16(D[b]) with `tsr_f32_to_fp16`; this conversion is exact for QACT
output. Bytes are placed per §4.3 or §4.4.

- The software sequence for `SET_ROWS` is `QACT A32 (K = E, T rows)` followed by `KVST`. QACT A32 is
  `quantize_row_q8_0_ref` ([spec](../spec.md) §6), so the stored values equal llama.cpp's except where x86
  `quantize_row_q8_0` rounds a tie differently (about 1 value in 3–4·10^5).
- Partial meta beats (K with E % 1024 ≠ 0; every V cell) are written by read-modify-write inside the DMA
  unit: read the beat, merge the bytes, write it. This is safe because DMA ops run one at a time in queue
  order, and readers in other units are ordered by dependencies (the memory model, spec §16). A meta beat
  that the op fully covers is written without a read. No core-boundary change (write strobes) is needed.
  Byte enables would give an identical architectural result.
- Cells c0..c0+n_rows−1 must be consecutive. The kernel splits non-consecutive `SET_ROWS` indices into
  runs.

### 5.2 `ATTQK`: scores and row maxima (0x21, unit 2 MVM)

| Field | Bits | Meaning |
|---|---|---|
| sub | w0[11:8] | KV format: 0 q8_0; else `BAD_FMT` |
| flags | w0[12] CAUSAL, [13] SLIDE, [14] MAXOUT, [15] 0 | |
| x16 | x16[5:0] = w0[21:16] = G − 1; x16[11:6] = w0[27:22] = hg0; x16[15:12] = w0[31:28] = 0 | |
| w1 | | dst: score rows, row r at dst + r·rstride (lines) |
| w2 | | x: T_q QA(A32, n_head·D) vectors at stride QA_LINES(A32, n_head·D) |
| w3 | | K plane base B |
| w4 | [16:0] shape (above), [31:17] rstride (0 → n_groups) | |
| w5 | [15:0] u0, [31:16] n_groups (1–4095) | groups [u0, u0 + n_groups) |
| w6 | [15:0] c_lo, [31:16] c_hi | cell bounds relative to 16·u0, c_lo ≤ c_hi ≤ 16·n_groups |
| w7 | | mdst: row-maximum vector (SPM line), 0 unless MAXOUT |

Semantics: for every pass row r(q, g, i) and every cell j, with j_rel = j − 16·u0 ∈ [0, 16·n_groups):

```
lo_q = max(0, c_lo − SLIDE·(T_q − 1 − q));   hi_q = c_hi − CAUSAL·(T_q − 1 − q)
s[r][j_rel] = (lo_q ≤ j_rel < hi_q) ? SEG(K row of cell j, segment g ; Q vector q, head h = g·G + hg0 + i)
                                    : −inf
SPM f32 at line dst + r·rstride + ⌊j_rel/16⌋, word j_rel mod 16
MAXOUT: f32 word r of the vector at mdst = max_tot(M_FLOOR, s[r][0 .. 16·n_groups))
```

- SEG is the spec's Q8_0 row dot product ([spec](../spec.md) §7, `tsr_dot_row`) applied to the D-element
  segment [g·D, (g+1)·D) of the cell's K row, with the QA blocks of head h of query vector q (§6.2).
- max_tot uses the spec's total order (−inf < … < +inf < NaN, spec §9).
- M_FLOOR = −2^100 (0xF1800000, §6.3).
- CAUSAL and SLIDE describe a prefill tile whose T_q query tokens occupy consecutive cells: row q sees
  [lo_q, hi_q). With c_hi = (cell of the last query) + 1, CAUSAL gives each query its own cell as the last
  visible one. SLIDE moves the window start with the query (SWA).

### 5.3 `ATTPV`: P·V partial outputs and row sums (0x22, unit 2 MVM)

| Field | Bits | Meaning |
|---|---|---|
| sub | | KV format: 0 q8_0; else `BAD_FMT` |
| flags | w0[12] LSUM; [15:13] 0 | |
| x16 | as `ATTQK` | |
| w1 | | dst: channel partials |
| w2 | | p: weight rows, row r at p + r·pstride, cell j_rel at word j_rel |
| w3 | | V plane base B |
| w4 | [16:0] shape, [31:17] pstride (0 → n_groups) | |
| w5 | [15:0] u0, [31:16] n_groups | |
| w6 | [15:0] LG, [31:16] 0 | |
| w7 | | cstride (lines; 0 → R·D/16) |

Output, for every channel c < C and pass row r: O_c[r][d] at line dst + c·cstride + r·D/16 + ⌊d/16⌋, word
d mod 16; with LSUM, l_c[r] at line dst + C·cstride + c·⌈R/16⌉ + ⌊r/16⌋, word r mod 16.

The numeric definition is §6.4–§6.5. Channel c accumulates only the cells of its own groups, so the op
produces C partials. AMERGE adds them in a fixed tree.

### 5.4 `ASMAX`: softmax numerator (0x34, unit 3 VU)

w1 dst, w2 src (may equal dst), w3 mvec (row maxima, word r), w4 [15:0] n_lines (1–4095), [31:16] stride
(0 → n_lines), w5 R (1–1024), w6 scale (f32 bits), w7 0.

```
for r < R, k < 16·n_lines:   dst[r][k] = EXP( fmul( fsub(src[r][k], mvec[r]), scale ) )
```

Rows are at src + r·stride and dst + r·stride. `scale` must be finite, normal and > 0, otherwise `OPERAND`.
The positive scale is what makes max(s) the right offset. EXP is VU's `tsr_m_exp` (§11).

### 5.5 `AMASK`: general mask path (0x35, unit 3 VU)

w1 dst, w2 src (rows at stride), w3 mask (f32 rows), w4 [15:0] n_lines, [31:16] stride, w5 [15:0] R,
[31:16] rpq (rows per query, = n_head), w6 mask stride (lines between query rows), w7 mdst (row maxima out).

```
q = ⌊r / rpq⌋;  m = mask[q][k]
dst[r][k] = (bits(m) == 0xFF800000) ? −inf : fadd(src[r][k], m)
mdst[r]   = max_tot(M_FLOOR, max_k dst[r][k])
```

The −inf test reproduces the CPU's skip of masked cells (§2.3). A finite mask value is added to the raw
score, and ASMAX and AMERGE scale the sum (AT-9): the device computes EXP(scale·(s + m)), where the CPU
computes exp(scale·s + m). The two agree when m = ±0 or scale = 1. So the backend's f32 copy holds
fdiv(m, scale) (§7.3; `tsr_msum_mask_f32`), which gives the CPU's form up to f32 rounding for every finite m
(NMSE about 10^-13 against ggml-cpu in `ref_at_ggml`).

### 5.5a Optional local FAST attention

`CAPS.ATT_LOCAL` (bit 12) permits three local score and probability slots per owner on a four-channel core. The
feature is optional, is not in the release image, and does not change the FLASH_ATTN record. Slot 0 keeps the
existing SPM operation; ATTQK and ATTPV select slots 1–3 in `w0[29:28]`, and ASMAX uses `w7[1:0]` for the slot
and `w7[17:2]` for the chunk's global first group `u0`. The firmware selects local sequences only for FAST
masking (raw local ATTQK requires MAXOUT, raw local ASMAX requires `w1 == w2`), for at most 224 full query rows,
128 groups per chunk and 4,480 lines per owner per slot (`rows * round_up(groups, 4) / 4` must fit the owner
depth, `sw/sim/include/tsr_local_tile.h`); the simulator and the raw instruction checker reject unsupported
geometry before execution. Ownership remains global group `u mod 4`. ATTQK commits score lines locally and sends
compact per-line maxima to the existing global maximum path, ASMAX receives that maximum and uses the VU EXP lanes
of the same outer group, and ATTPV reads P locally; scores and P are not written to their legacy SPM
destinations, which still satisfy the range and extent checks. PV partials, LSUM, AMERGE and ANORM keep their
shared path and the `((c0+c1)+(c2+c3))` order. Software owns the reuse of the three physically separate 1R1W
banks per owner (QK completion precedes ASMAX, ASMAX completion precedes PV, PV's consumption precedes the next
QK on that slot); there is no hardware tag or rollback for malformed raw schedules. The firmware selects local
or legacy for the whole record and uses the legacy path for GENERAL, mixed or oversized records; list cuts, CPU
readers and final outputs still use DDR.

### 5.6 `AMERGE`: channel tree and online merge (0x36, unit 3 VU)

flags w0[12] INIT. w1 Orun (R rows of D/16 lines), w2 parts (an `ATTPV` dst written with LSUM), w3 mk (chunk
row maxima), w4 st (m_run vector at line st, l_run vector at line st + ⌈R/16⌉), w5 [15:0] R, [17:16] dcode,
[31:18] 0, w6 scale (f32, as ASMAX), w7 cstride (as the producing ATTPV).

```
Ok[r][d] = CTREE_c( parts O_c[r][d] );   lk[r] = CTREE_c( parts l_c[r] )
INIT:  Orun = Ok;  l_run = lk;  m_run = mk
else:  m' = max_tot(m_run[r], mk[r])
       a  = EXP(fmul(fsub(m_run[r], m'), scale));  b = EXP(fmul(fsub(mk[r], m'), scale))
       Orun[r][d] = fadd(fmul(Orun[r][d], a), fmul(Ok[r][d], b))
       l_run[r]   = fadd(fmul(l_run[r], a), fmul(lk[r], b));   m_run[r] = m'
```

CTREE over c = 0..C−1 is pairwise by levels, left to right. At each level, element 2k + 1 is added to
element 2k, and an unpaired last element passes unchanged. For C = 4 that is ((x0 + x1) + (x2 + x3)); for
C = 3, ((x0 + x1) + x2).

### 5.7 `ANORM`: final normalization (0x37, unit 3 VU)

w1 out (row r at out + r·D/16: exactly the FA output layout `[token][head][D]`), w2 Orun, w3 l_run, w4
[15:0] R, [17:16] dcode, w5–w7 0.

```
out[r][d] = fmul( Orun[r][d], RECIP( max_tot(l_run[r], 2^-126) ) )      RECIP = tsr_frecip (IEEE 1/x)
```

The 2^-126 floor makes an empty row (l = 0, O = 0) produce 0, as the CPU's `S == 0 ? 0 : 1/S` does. Every
nonzero l is at least 2^-126 because of FTZ, so the floor changes nothing else.

### 5.8 Checks and error codes

The order is EX's (EX §8.1): `ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` → `T_RANGE` → `OPERAND` →
`SPM_RANGE` → `DRAM_RANGE`. No new code is needed.

| Op | `K_RANGE` | `T_RANGE` | `OPERAND` (`ERR_INFO` = the value) | `SPM_RANGE` / `DRAM_RANGE` extents |
|---|---|---|---|---|
| KVST | E/64 = 0 | | LG = 0 (0); c0 + n_rows > 16·C·LG (c0) | src: (n_rows−1)·stride + QA_LINES(A32, E); plane: B + LG·GB |
| ATTQK | E/64 = 0; G·E > K_MAX | T_q·GS > T_MAX | E % D ≠ 0 (E); hg0 + GS > G (hg0 + GS); n_groups ∉ [1, 4095] (n_groups); c_lo > c_hi or c_hi > 16·n_groups (c_hi); pass rows (E/D)·GS·T_q > 128 (rows); 0 < rstride < n_groups (rstride) | dst: (max pass row)·rstride + n_groups; x: T_q·QA_LINES(A32, G·E); mdst: ⌈(max pass row + 1)/16⌉; plane: B + (ℓ_max+1)·GB, ℓ_max = ⌊(u0 + n_groups − 1)/C⌋ |
| ATTPV | as ATTQK | as ATTQK | as ATTQK except c_lo and c_hi; ℓ_max ≥ LG (u0 + n_groups − 1); (E/D)·16·⌈n_groups/C⌉ > PCAP (n_groups); 0 < pstride < n_groups | p: (max pass row)·pstride + n_groups; dst: (C−1)·cstride + R·D/16, with LSUM + C·⌈R/16⌉ after dst + C·cstride; plane: B + ((E/64 − 1)·LG + ℓ_max + 1)·17 |
| ASMAX | | | n_lines ∉ [1, 4095]; R ∉ [1, 1024]; scale not finite, not normal or ≤ 0 (scale bits) | src, dst: (R−1)·stride + n_lines; mvec ⌈R/16⌉ |
| AMASK | | | as ASMAX for n_lines and R; rpq = 0 | src, dst, mask ((⌈R/rpq⌉−1)·mstride + n_lines), mdst |
| AMERGE, ANORM | | | R ∉ [1, 1024]; AMERGE scale as ASMAX | every operand by its extent |

The `ILLEGAL_OP` conditions in addition to reserved fields: `KVST` sub ≥ 2; `ATTQK` w7 ≠ 0 without MAXOUT;
`ATTQK` and `ATTPV` x16[15:12] (w0[31:28]) ≠ 0.

PCAP is the p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX)·16 = 9,344 on the U200 (the
activation buffer of one token slot). It is advertised in CFG3 (§5.10).

### 5.9 DQROW (0x12)

DQROW is VU's op (EX §7.2: VU owns the encoding): SPM → f32 dequantization on the VU after an LD of the
repacked row (VU §5.6). It is simulator only.

- Attention does not need it. K and V are never dequantized outside `ATTPV`. llama.cpp keeps the input layer,
  and so the `tok_embd` `GET_ROWS`, on the CPU (`src/llama-model.cpp`; SW R2); the f32 row arrives as an input
  of the device split.
- A device embedding lookup would be LD (ROW_BEATS beats of the row) + DQROW, bit-identical to
  `dequantize_row_*` for Q8_0 and Q4_K tensors ([spec](../spec.md) §21, Agreement with ggml).
- When the GGUF `tok_embd` type is neither Q8_0 nor Q4_K, a DQROW of the load-time Q8_0 conversion differs
  from the CPU's `GET_ROWS` in every element. The exact alternative is an f32 copy of `tok_embd` and an `LD`
  of the row.

### 5.10 CAPS, CFG3, units

- `CAPS` bit5 ATT (EX): all seven AT ops are present. Bits 12–15 are reserved for AT (bit 12 is
  `ATT_LOCAL`, §5.5a).
- `CFG3[31:16]` = PCAP / 16 (584 on the U200).
- Units. KVST is in the DMA class (unit 0). ATTQK and ATTPV are in the MVM class (unit 2). ASMAX, AMASK,
  AMERGE and ANORM are in the VU class (unit 3); the VU executes them (VU §5.11, VU-27). Unit 4 is absent
  (`ISSUE_UNITS[4]` = 0, `Q_DEPTH[4]` = 0, CC's SPM client 5 tied off) and stays reserved for a dedicated
  attention engine. EX's PERF index X+2 `ATT_CYCLES` therefore reads 0. AT's counters are X + 16 … X + 19
  (§10, EX §6).
- Opcodes. The AT quantize-class block 0x19–0x1B and 0x15–0x17, 0x23 stay reserved for AT.

---

## 6. Numerics (bit-exact definitions)

The fp rules are the spec's: DAZ and FTZ, RNE, canonical NaN ([spec](../spec.md) §7 and §21, FTZ decision
point). fadd, fsub, fmul and frecip are `tsr_fadd` and so on (`sw/sim/tsr_ref.c`).

### 6.1 KV quantization

As §5.1: QACT A32 values. The stored scale is fp16 of the QA D value, which is exact.

### 6.2 Scores (`ATTQK`)

Segment g of cell j's K row is blocks b = g·D/32 + k, k < D/32. The query QA vector q holds head h in its
blocks h·D/32 + k: QA blocks are 32-aligned within heads, so one QACT over the whole Q row (K = n_head·D)
equals per-head QACT.

```
t_k = fmul( fmul( fp16_to_f32(dK_b), Dx_q[h·D/32 + k] ), i32_to_f32( Σ_{i<32} qK_b[i]·qx[...] ) )
SEG = tsr_tree16(p) with p_k = fadd(+0.0, t_k) for k < D/32, p_k = +0.0 otherwise
```

This is exactly `tsr_dot_row(Q8_0, segment-as-row, K = D, head-h QA)`, the spec row rule with 16 partials
([spec](../spec.md) §7). For D/32 = 2^m ≤ 16 it equals z(pairwise tree of t_0..t_(D/32−1)), where z maps −0
to +0; the hardware may use that short tree. The reference test proves the equality (§13.2).

### 6.3 Row maximum and M_FLOOR

m_r = max_tot(M_FLOOR, scores), with M_FLOOR = −2^100.

- A row with no visible cell in a chunk gets m = M_FLOOR. Its p then equals EXP((−inf − M_FLOOR)·scale) =
  EXP(−inf) = +0.
- Merges stay finite: a = EXP((M_FLOOR − m')·scale) = 0 against a real maximum, and EXP(0) = 1 between two
  floors.
- A real score below −2^100 would underflow its p to 0. That needs |q|·|k| > 10^30, which is not a practical
  case.

### 6.4 P·V (`ATTPV`): block floating point per segment

Per channel c, slice s (head group g = ⌊s/(D/64)⌋), slot t = t(q, i) and row r = r(q, g, i):

1. Segments. U_c = the op's groups u ≡ c (mod C), ascending. Segments are consecutive runs of 4 elements of
   U_c (64 cells; the last segment may be shorter). The order is fixed by u0 and C, so the result depends on
   NUM_CH (AT-11).
2. Weights. For each segment S, each cell j ∈ S with p_r[j] not ±0, and e ∈ {0, 1}:
   w_{j,e} = fmul(p_r[j], fp16_to_f32(dV_{j, 2s+e})). Cells with p = ±0 are skipped (they contribute nothing,
   like the CPU's skip of −inf cells).
3. Non-finite rule. If any w_{j,e} is NaN or ±Inf, then c_L = canonical NaN for every lane L of block e
   (L ∈ [32e, 32e+32)).
4. Exponent. E_S = max of ε(w) = ⌊log2|w|⌋ over the finite nonzero w_{j,e} of both blocks. If there are
   none, c_L = +0 for all L.
5. Integer weights. W_{j,e} = RNE_int(w_{j,e} · 2^(23 − E_S)), an exact power-of-two scaling followed by
   round-half-even to an integer. |W| < 2^24.
6. Exact sum. A_L = Σ_{j∈S} W_{j,⌊L/32⌋} · qV_j[64s + L], an exact integer with |A| < 2^37.
7. Segment value. c_L = F(A_L, E_S − 23), where F(A, k) = RNE24(A)·2^k: A rounded to 24 significant bits
   with RNE, then scaled exactly. A result below 2^-126 becomes ±0 (sign of A); overflow gives ±Inf.
8. Accumulation. O[L] starts at +0.0, and O[L] = fadd(O[L], c_L) per segment in segment order. After the
   last segment, O_c[r][(s − g·D/64)·64 + L] = O[L].

Each product keeps at least 24 significant bits relative to the segment maximum. One f32 addition is made per
64 cells, against the CPU's one fused multiply-add per cell. The device is no less precise than the CPU.

### 6.5 Row sums (`ATTPV` LSUM)

The same segments of U_c, once per row (not per slice):

- E_p = max ε(p) over the finite nonzero p_r[j] of the segment;
- P_j = RNE_int(p_r[j]·2^(23 − E_p));
- cl = F(Σ P_j, E_p − 23), or canonical NaN if any p in the segment is non-finite, or +0 if no p is finite
  and nonzero;
- l_c[r] = fadd(l_c[r], cl) per segment, from +0.0.

### 6.6 The composite FA (normative kernel algorithm)

The tile-op sequence of a composite kernel is part of the numerical contract (SW §7.5, which points here).
ATTQK and ATTPV run on the MVM queue, ASMAX, AMASK, AMERGE and ANORM on the VU queue (VU §5.11). The FA
kernel (`k_fattn.cpp`) and its reference `tsr_ref_fa` (§13.1) are defined as:

1. Query tiles and head subsets. The pass rule (AT-15) gives GS = min(G, T_MAX) and T_q = ⌊T_MAX/GS⌋. The
   kernel iterates query tiles [q0, q0 + T_q′) of T_q tokens (the last one may be shorter); a tile has
   R = T_q′·n_head rows. The head subsets hg0 = 0, GS, … (< G; the last one has GS′ = G − hg0 heads) run
   inside each chunk of the tile (step 4), not as outer passes: AMERGE and ANORM have no row selection, so
   for G > T_MAX (Gemma 4 global layers, G = 16) a pass-outermost order cannot be expressed; every op is
   row-independent, so each row gets the per-pass result (AT-22).
2. Q. QACT A32 over each of the tile's tokens' Q rows (K = n_head·D), one QA vector per token. The scale is
   not applied to Q (ASMAX and AMERGE apply it).
3. Chunks. The tile's chunk list comes from the mask summary (§7.3) by the chunk planner
   `sw/sim/include/tsr_fa_plan.h`, which the reference and the firmware compile (one source, so one chunk
   list; its header states the rules, which are normative):
   1. a GENERAL MSUM: one GENERAL sub-range over all n_kv cells;
   2. T_q′ = 1: each of the row's 0–2 runs is a FAST sub-range over the groups it intersects, with c_lo and
      c_hi from the run and no flags (a wrapped SWA ring: two sub-ranges; they may share a group, streamed
      twice with disjoint bounds);
   3. T_q′ > 1: one FAST sub-range when every row has one run and the runs are
      [max(0, lo − SLIDE·d_q), hi − CAUSAL·d_q) with d_q = T_q′ − 1 − q for some lo, hi, CAUSAL, SLIDE (each
      flag only when the bounds differ between rows); otherwise one GENERAL sub-range over the hull of the
      runs (it needs the f32 mask copy, §7.3, even when the MSUM itself is not GENERAL);
   4. NCG = min(128, C·⌊PCAP/(16·E/D)⌋, max(4C, roundup_4C(⌈n_total/2⌉))) groups, n_total = the sub-ranges'
      groups; a sub-range [U_lo, U_hi) is cut at U_lo + k·NCG (at least two chunks when there are ≥ 8C
      groups, §7.4);
   5. FAST with T_q′ > 1: a cut strictly inside the spread of the rows' upper (CAUSAL) or lower (SLIDE)
      bounds is replaced by the group boundaries just below and above it (at most 8 rounds); a tile that
      still cannot be encoded is refused (the record is BAD_RECORD; data-dependent, so `supports_op` cannot
      see it: tiny windows next to a causal edge);
   6. per chunk: c_lo = 0 and no SLIDE when the lower bound is at or below the chunk's first cell, else the
      bound and SLIDE; c_hi = 16·n_groups and no CAUSAL when every row's upper bound is at or past its end,
      else the bound and CAUSAL; verified against every row's run;
   7. order: by (u0, c_lo); then every chunk holding a cell written in this launch moves to the end, keeping
      its order ("newest last"); the record's FA_NEW flags word (§12) names those cells (the layer's
      SET_ROWS_KV indices);
   8. a tile with no visible cell: one chunk, group 0, c_lo = c_hi = 0 (output 0, as the CPU).
4. Per chunk k (every op covers the tile's R rows):
   - for each head subset: FAST `ATTQK`(MAXOUT, c_lo, c_hi, CAUSAL/SLIDE) → S, MK; GENERAL
     `ATTQK`(c_lo = 0, c_hi = 16·n_groups, no MAXOUT) → S;
   - GENERAL: `AMASK`(f32 mask copy) → S, MK;
   - `ASMAX`(in place, MK, scale) → p;
   - for each head subset: `ATTPV`(LSUM) → parts;
   - `AMERGE`(INIT if k = 0).
5. End. `ANORM` writes the tile's output rows; then ST.

`scale` is ggml's op_params[0] f32, unchanged. This sequence, the planner's rules and the pass rule are the
numerical contract; changing any of them changes logits and requires a kind numerics revision (SW §6.9). The
issue order across chunks is not part of it: the firmware runs ATTPV and AMERGE one chunk behind ATTQK and
ASMAX (§7.1), which changes no op's operands.

### 6.7 Agreement with the CPU (requirement: KL level)

Sources of difference, each at or below the level the existing MUL_MAT already has:

| # | Source | Size |
|---|---|---|
| 1 | Q and the KV cache are quantized by the ref rounding, the CPU by x86 AVX | rare ties (about 1 per 3–4·10^5) |
| 2 | Score accumulation order (tree vs AVX) | f32 rounding |
| 3 | The max is taken over raw scores per chunk, then scaled inside EXP; the CPU scales first and keeps a running max | f32 rounding; normalization cancels the offset |
| 4 | `tsr_m_exp` vs libm `expf` | VU's accuracy bound (§11) |
| 5 | P·V block-float with 24-bit weights and f32 per 64 cells, vs sequential f32 FMA | at most f32 level |
| 6 | Row sums block-float vs sequential f32 | at most f32 level |
| 7 | Normalization: exact reciprocal then multiply | identical form to the CPU's |

No 8-bit quantization of P and no re-quantization of V occur. Those were rejected because they add errors of
order 10^-3 to 10^-2 relative (AT-6).

---

## 7. Data flow (FA kernel on the RISC-V)

### 7.1 Decode, one layer

Sketch in the SDK style:

```cpp
// SPM: QA (Q row as QA A32), kq/vq (QA A32 of the new token's K and V rows), S[3] score/p buffers,
// P[2] ATTPV outputs, MK[4] row maxima, Orun + ST (running state), OUT; a GENERAL chunk also LDs its
// f32 mask rows (§7.3) into M[k%2], after the AMASK of chunk k-2 (k_fattn.cpp). Planes kp/vp from the
// SET_ROWS/FLASH_ATTN records. One head subset (G <= T_MAX); with more, each ATTQK / ATTPV is one per
// head subset (§6.6 step 4).
ticket ks = issue(KVST(KPLANE, kq, kp.base, E, 1, cell, LG));             // DMA
ticket vs = issue(KVST(VPLANE, vq, vp.base, E, 1, cell, LG));             // DMA
ticket qa = issue(QACT(qa::a32, QA, q, n_head * D, 1));                    // QACT
ticket qk[NC], sm[NC], pv[NC], mg = {};
for (k = 0; k < NC; ++k) {                                                 // newest cells in chunk NC-1
  qk[k] = issue(ATTQK(S[k%3], QA, kp.base, shp, ch[k], MAXOUT, MK[k%4]), after(qa, k == NC-1 ? ks : none));
  sm[k] = issue(ASMAX(S[k%3], S[k%3], MK[k%4], ch[k].lines, R, scale), after(qk[k]));          // VU
  if (k >= 1) {                                                            // ATTPV one chunk behind
    pv[k-1] = issue(ATTPV(P[(k-1)%2], S[(k-1)%3], vp.base, shp, ch[k-1], LSUM), after(sm[k-1], k-1 == NC-1 ? vs : none));
    mg = issue(AMERGE(Orun, P[(k-1)%2], MK[(k-1)%4], ST, R, D, scale, k-1 == 0), after(pv[k-1]));
  }
}
pv[NC-1] = issue(ATTPV(P[(NC-1)%2], S[(NC-1)%3], vp.base, shp, ch[NC-1], LSUM), after(sm[NC-1], vs));
mg = issue(AMERGE(Orun, P[(NC-1)%2], MK[(NC-1)%4], ST, R, D, scale, NC == 1), after(pv[NC-1]));
ticket no = issue(ANORM(OUT, Orun, ST_l, R, D), after(mg));               // → QACT for wo
```

Queue orders on each unit:

- MVM: ATTQK0, ATTQK1, ATTPV0, ATTQK2, ATTPV1, …, ATTPV(NC−1).
- VU: ASMAX0, ASMAX1, AMERGE0, ASMAX2, AMERGE1, …
- DMA: KVST(K), KVST(V).

Buffer reuse is ordered by queue order: ATTQK k writes S[k mod 3], which ATTPV k−3 read earlier in the MVM
queue; P[2] likewise. MK needs four buffers: with three, ATTQK k (MVM, MAXOUT) writes the buffer AMERGE k−3
(VU) reads, and nothing orders that MVM op after that VU op. With four, ATTQK k writes MK[k mod 4], last read
by AMERGE k−4; ATTQK k follows ATTPV k−2 on the MVM queue, ATTPV k−2 depends on ASMAX k−2, and ASMAX k−2
follows AMERGE k−4 on the VU queue (AMERGE runs one chunk behind), so the reuse is ordered. Every op has at
most two dependencies.

### 7.2 Prefill

- Query tiles of T_q = ⌊T_MAX/GS⌋ tokens, with GS = min(G, T_MAX). Gemma 4 SWA layers (G = 2) take
  T_q = T_MAX/2; a model with G ≥ T_MAX takes T_q = 1.
- One QACT per token builds the tile's QA vectors.
- KVST with n_rows = the ubatch run of consecutive cells, after QACT with T ≤ T_MAX per QACT.
- For causal single-sequence tiles: FAST with CAUSAL. For non-wrapped SWA: CAUSAL + SLIDE.
- K and V are re-streamed once per (tile, head subset). Prefill attention bytes are therefore
  (ubatch/T_q)·(cells) per layer.

### 7.3 Masks: the mask summary (MSUM)

Producer. The backend derives an MSUM from each F16 KQ mask whenever the scheduler writes the mask
(`set_tensor`), and stores it next to the mask in device memory. Then the cached op list stays valid while
the mask changes per token. MSUM layout (little-endian words):

```
hdr : 'MSUM', n_q, n_kv, flags (bit0 GENERAL), f32_mask handle (the reserved copy; valid iff GENERAL), 3 × 0
row q: lo0, hi0, lo1, hi1     (at most two runs of cells with mask == +0/−0; every other cell is −inf)
```

GENERAL is set when some row has a value other than ±0 / −inf (finite nonzero, +inf or NaN; the reference
`tsr_msum_build_f16` treats all three alike) or more than two runs, and by the producer when the planner
would refuse one of a reader's tiles (§6.6 rules 5 and 6, data-dependent: the promotion keeps the rows' runs
and makes every tile plan under rule 1; SW-44). The backend then also writes an f32 copy of the mask, with
row stride ⌈n_kv/16⌉ lines. It writes the copy too when a multi-row query tile of a FLASH_ATTN that reads the
mask takes the GENERAL path although the MSUM is not GENERAL (§6.6 rule 3: rows with two runs, holes, a
wrapped SWA ring in prefill); without it that record is BAD_RECORD (`tsr_ref_fa`: TSR_FA_E_MASK). The
producer can tell by running the planner (`tsr_fa_plan`) on the MSUM's tiles. AMASK reads that copy. Its
values are fdiv(m, scale) with the reading FA op's scale (§5.5), so a mask read by FA ops with different
scales needs one copy per scale. Every FLASH_ATTN registers as a reader of its mask, the copy of each
distinct scale has a fixed handle, and FLASH_ATTN word 13 names it whether or not it was written, so a cached
op list stays valid while the mask changes (SW-48).

Kernel use, per pass:

- every row has one run, and the runs fit lo_q, hi_q of §5.2 for some c_lo, c_hi, CAUSAL, SLIDE → FAST over
  the groups that intersect the runs;
- decode with two runs (wrapped SWA ring) → two FAST sub-ranges, each its own chunk;
- otherwise → GENERAL over the union of the runs (or all n_kv cells if GENERAL is set).

Groups with no visible cell for any pass row are never streamed. The 256-cell padding of n_kv and empty tail
cells cost no bandwidth.

A row with no visible cell at all gives O = 0, l = 0 and output 0, like the CPU. llama.cpp never builds such
a row for the supported models.

### 7.4 Overlap (D3) and SPM budget

Attention of layer l cannot overlap the next layer's weight MVMs: wo needs its output. Inside the layer:

- the K and V projections and the K and V quantization and KVST overlap ATTQK of all but the last chunk;
- ASMAX of chunk k overlaps ATTPV k−1 and ATTQK k+1;
- AMERGE k−1 overlaps ATTQK k+1 and ATTPV k.

The exposed parts are the first ATTQK's fill, ASMAX of the last chunk while ATTPV of the previous chunk
still runs, and the tail: the last AMERGE, ANORM, then QACT for wo.

In the OPL executor, FLASH_ATTN loads Q from DRAM (LD Q, then QACT), and that LD follows the layer's
SET_ROWS_KV KVSTs on the DMA queue, so no chunk overlaps the KVSTs (about 100 core cycles per decode KVST).
The newest-last order stays, because it is part of the numerics (§6.6 rule 7).

SPM traffic per chunk of n cells and R rows (estimate):

| Op | Read lines | Write lines |
|---|---|---|
| ATTQK | QA (T_q·QA_LINES) | R·n/16 |
| ASMAX | R·n/16 | R·n/16 |
| ATTPV | R·n/16 (p to the clusters) | C·R·D/16 (+ l) |
| AMERGE | (C+1)·R·D/16 + small | R·D/16 + small |

For 32B (R = 40, 2,048 cells per chunk) this averages about 0.6 lines per cycle on each SPM port over the
chunk's MVM cycles. The ports carry it with the second read path and round robin (CC §10). Scratch per layer
for 32B: 3 × 5,120 (S) + 2 × 1,292 (P) + 330 (state) + 90 (QA) ≈ 18.4 K lines (1.15 MiB) of the scratch ring
(SW §7.3).

### 7.5 Gemma 4 specifics

- QK-norm. Q and K RMSNorm per head with weight, V RMSNorm without weight. These are VU work before QACT and
  KVST (SW §7.5 RMS_NORM, per-head form; VU §8.1).
- Scale. 1.0: ASMAX scale = 1.0.
- SWA layers. 12B: D = 256, n_head_kv = 8, G = 2. A separate cache of 1,536 cells (n_swa 1024): planes
  LG = 1,536/(16·C) = 24 on the U200. The MSUM runs carry the window; after a wrap a decode row has two runs
  (two FAST sub-ranges).
- Full layers. 12B: D = 512, n_head_kv = 1, G = 16 > T_MAX: GS = T_MAX, so 16 / T_MAX passes per decode token
  (hg0 = 0, GS, …) and the full-layer K and V planes are streamed that many times per token. E = 512, so
  KVST read-modify-writes every K meta beat (§4.5).
- K = V source on full layers without `wv`: two SET_ROWS from different tensors, nothing special.
- The embedding scale (a MUL by `inp_scale_rows`) and the final-logit softcap are not AT.

---

## 8. Hardware structure (guidance; the contract is §5–§6)

### 8.1 `ATTQK` in the cluster

The stream requester streams the channel's groups (contiguous beats) and the chunk parser counts blocks with a
row boundary every E/32 blocks and a segment boundary every D/32 blocks; it is the Q8_0 path unchanged except
for the counters. The broadcast copies the T_q QA vectors, and query q's vector is written into the GS slot
columns of that query; slot t reads head h(t, g) of its vector with the per-slot read address
base(t, g) = (g·G + hg0 + i)·D, advanced by G·D at each segment boundary. MAC lanes and terms are unchanged
(Q8_0: 64 lanes, 2 terms per beat per slot, two slots per DSP as in the MVM). D/32 ∈ {2, 4, 8, 16} terms per
(segment, slot) go through the two accumulate adders per slot (the first tree level) and the tree adders, then
z(); at most 2 additions per cycle per slot. The gather has a line buffer per pass row (at most 128 rows × 64 B,
2 RAMB36) written once a row's 16 cells are complete (one full-line SPM write, no byte enables, no ACC); cells
outside [lo_q, hi_q) are replaced by -inf, a max tracker per row reads, compares and writes per line, and
MAXOUT writes ⌈rows/16⌉ lines at the end. The op runs at one beat per cycle per channel while the gather keeps
up; the gather needs 64·C·GS·T_q/(17·D) lines per cycle, so above 1 (only D = 64) the op is gather-bound (§9.1).

### 8.2 `ATTPV` in the cluster

V beats go through a 64-beat delay FIFO to the MAC (DSP accumulate); the w-unit (16 fmul per cycle) computes
w = p·d from the meta beat and the probabilities, takes the segment's exponent maximum E_S and aligns (RNE) the
weights W(slot, e); at segment end 512 accumulators are captured and 8 conversion-and-add pipes (integer to f32,
×2^k, fadd) drain them in at most 64 cycles under the next segment.

- p broadcast: line (row r, group u) goes only to cluster u mod C, into slot column t(r) at position (g(r),
  local cell); the SPM read rate equals the gather rate formula above.
- DSP use: 64 lanes × 8 slots = 512 DSPs, unpacked (A = W, 25 bits; B = the V byte), accumulating in P (48 bits)
  over a segment. Slots 4–7 use lanes 64–127, which receive copies of bytes 0–63. `tsr_mac_lane` has a PV mode
  (Xilinx: dynamic OPMODE and an A/D input mux; generic: behavioral; `tb/tech` checks the two variants against
  each other).
- The exponent E_S needs all w of a segment before its first product, so quant beats wait in the delay FIFO
  while the w-unit computes the segment's 128·GS·T_q weights. At the end of a slice, per slot, 4 lines go to SPM
  through the gather.

### 8.3 `KVST` in the DMA

The DMA reads QS and D lines from SPM, converts 32 scales per meta beat, writes full quant beats, and
read-modify-writes partial meta beats (§5.1). One token's K or V goes to one channel.

### 8.4 Cost (estimate, per cluster unless noted)

The attention modes add about 19–27 K LUT, 29 K FF, 16–32 DSP and 4–6 RAMB36 per cluster: the ATTQK
counters, per-slot addresses, segment-tree control, gather line buffers and max trackers (3–4 K LUT); the
ATTPV DSP input muxes and mode control (8–12 K LUT); the ATTPV w-unit with 16 fmul, the exponent maximum and
16 aligners (3–4 K LUT, 16–32 DSP); and the ATTPV delay FIFO, capture bank, 8 convert-and-add pipes and O/l
storage (5–7 K LUT, 22 K FF). KVST in the DMA (SLR1) adds 1–2 K LUT. The AT vector ops in the VU reuse the
EXP lanes and add about 2 K LUT (VU §7.10). DSP and RAM budgets are not the limit; LUT and timing are (CC §14).

---

## 9. Throughput (estimates)

### 9.1 Op rates

| Model | D | G (decode GS) | ATTQK and ATTPV efficiency in decode (gather or p rate) | Prefill (GS·T_q = 8) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 7 | 61 % (1.65 lines/cycle needed) | 53 % |
| Qwen2.5-1.5B, 3B, 7B | 128 | 6, 8, 7 | 100 % | 100 % |
| Qwen2.5-14B, 32B | 128 | 5 | 100 % | 100 % |
| Gemma 4 12B SWA / full | 256 / 512 | 2 / 16 | 100 % | 100 % |

Fill per ATT op is about 300–400 cycles: dispatch, broadcast, pipeline, the 64-beat segment delay, and the
gather drain. The simulator's cycle model charges 350 (§13.4).

### 9.2 One decode layer (cycle composition)

A decode layer of a 32B model at 4K context costs about 39,900 cycles, about 89 % of which keep the DDR busy: the
K and V streams (ATTQK and ATTPV, two chunks each) take 17,854 cycles each at 0.975 beats per cycle per channel,
the ATT op fills about 1,400, the exposed head (QACT of Q to the first ATTQK) about 300 and the tail (the last
AMERGE, ANORM and the QACT for wo) about 2,500.

### 9.3 Issue cost

The RISC-V issues 4 + 4·NC attention ops per layer: 2 KVST, 1 QACT and 1 ANORM, plus ATTQK, ASMAX, ATTPV and
AMERGE per chunk (NC chunks). For 32B at 4K that is 12 per layer, replacing about 18 K per-head ops per
token. The issue overlaps the unit work (§7.4).

---

## 10. PERF counters (slots X + 16 … X + 19, EX §6)

| Slot | Name | Kind | Meaning |
|---|---|---|---|
| X + 16 | `ATT_QK_CYCLES` | time | the MVM unit holds an ATTQK (a subset of `MVM_CYCLES`) |
| X + 17 | `ATT_PV_CYCLES` | time | the MVM unit holds an ATTPV |
| X + 18 | `ATT_PATH_STALL` | time | some cluster in an ATT op holds its stream because the gather (ATTQK) or the p broadcast (ATTPV) is behind: the gather- or broadcast-bound share |
| X + 19 | `KVST_RMW` | event | meta beats read-modify-written by KVST |

VU's slots X + 13 … X + 15 come first; at `NUM_CH` = 16 (X = 37) AT's end at 56, below DB's 61–63.

K and V read beats are already counted in `CH_RD_BEATS[c]`. For the spec's F metric, useful beats include the
planes' beats (the simulator's cycle model reports both).

---

## 11. Decisions

| Id | Decision | Reason |
|---|---|---|
| AT-1 | Where attention computes: QKᵀ and PV in the MVM clusters (one channel each); the VU only does softmax passes and merges | Only the clusters consume the KV cache at the 256 B/cycle DDR rate; one SPM port is 64 B/cycle |
| AT-2 | KV values are exactly llama.cpp's q8_0 (blocks along each head row, fp16 scale, `quantize_row_q8_0_ref` rounding); only byte placement changes | Device KV equals the CPU's; state save and load stay lossless |
| AT-3 | Channel distribution: 16-cell groups round-robin over channels (group u in channel u mod C) | Balances channels at any context; one score line per group and row |
| AT-4 | Plane layouts: K token-major in Q8_0 chunks; V slice-major (each 64-value slice of all cells contiguous) | Whole scores per pass for K; one DSP accumulator per slice for V |
| AT-5 | KV store op: KVST (DMA unit) stores QACT A32 output into consecutive cells; partial meta beats by read-modify-write inside the DMA | QACT is already bit-exact `quantize_row_q8_0_ref`; no write strobes or boundary change |
| AT-6 | PV numerics: block floating point per 64-cell channel segment: 24-bit integer weights, exact integer sums in the DSP P register, one f32 add per segment | Reuses the MAC DSPs and is at least as precise as the CPU's f32 path |
| AT-7 | PV segment length: 4 local groups = 64 cells per channel | 7.5 conversions per cycle per cluster instead of 30 for 16-cell segments |
| AT-8 | Softmax structure: two-pass within a chunk (max from ATTQK, EXP by ASMAX); online merge across chunks (AMERGE) | Per-cell rescaling is impossible in DSP accumulation; chunks fit the slot buffers and pipeline |
| AT-9 | Scale placement: inside ASMAX and AMERGE as EXP((s − m)·scale); ATTQK returns raw scores | One rounding as on the CPU, free in an op that reads every score |
| AT-10 | Row sums: ATTPV computes l with the same block-float rule (LSUM flag) | Saves a read pass over the scores and keeps l consistent with O |
| AT-11 | Channel partials: ATTPV writes C per-channel partials; AMERGE adds them in a fixed pairwise tree; results depend on NUM_CH | Avoids serializing the clusters; bit-exactness is per configuration |
| AT-12 | Row max floor: M_FLOOR = −2^100 (0xF1800000) in ATTQK MAXOUT and AMASK | Keeps merges finite for rows with no visible cell |
| AT-13 | Empty rows: ANORM multiplies by RECIP(max(l, 2^-126)) | Reproduces the CPU's `S == 0 ? 0 : 1/S` at no cost |
| AT-14 | Mask handling: a host-built mask summary (at most 2 cell runs per row) with CAUSAL and SLIDE bounds in ATTQK; an f32 copy and AMASK only for GENERAL masks | The VU has no f16; the cached op list stays valid; masked bandwidth is skipped |
| AT-15 | GQA and prefill mapping: MVM slots = (query token, head in the KV-head group); head subsets (hg0, GS) when G > T_MAX; T_q = floor(T_MAX/GS) | K streams once per pass and every G works (Gemma 4 12B global layers: 16 / T_MAX passes) |
| AT-16 | Units and opcodes: KVST 0x14 (DMA), ATTQK 0x21 and ATTPV 0x22 (MVM), ASMAX, AMASK, AMERGE, ANORM 0x34–0x37 (VU); unit 4 absent; no new error codes | The ops use existing datapaths; a separate queue would only add arbitration |
| AT-17 | Hadamard KV rotation is off (`LLAMA_ATTN_ROT_DISABLE=1` everywhere, SW-16); the bit-exact FWHT is recorded in Appendix A | Baselines are fixed with rotation off; q8_0 gains little |
| AT-18 | Unsupported FA features: no attention softcap, ALiBi, sinks, sparse n_kv_max, DK ≠ DV, D outside 64–512, K-shift | No target model uses them; each can be added later without a layout change |
| AT-19 | Chunk policy: NCG = min(128 groups, PCAP limit, at least 2 chunks for ≥ 8C groups), segment-aligned; the chunk with this launch's new cells last | Overlaps VU and MVM work; deterministic, so part of the numeric contract |
| AT-20 | DQROW per VU §5.6 (SPM to f32 on the VU, simulator only); a device embedding lookup would be LD + DQROW, with an f32 `tok_embd` copy for other GGUF types; never used by attention | VU owns the encoding; the lookup stays on the CPU |
| AT-21 | KV region: symmetric planes allocated top-down from each channel's end; one plane per (layer, stream, K or V); LG from kv_size | Equal channel shares; KV growth stays away from the weights |
| AT-22 | FA head subsets and chunk planner: head subsets inside each chunk; one chunk planner (`tsr_fa_plan.h`) for `tsr_ref_fa` and the firmware, its rules normative (§6.6); FA_NEW flags word; MK[4] | G > T_MAX cannot run as outer passes; one planner gives bit-identical outputs by construction; MK[3] raced |

---

## 12. Interfaces with other areas

| Area | AT provides | AT assumes |
|---|---|---|
| EX | Encodings and checks of 0x14, 0x21, 0x22, 0x34–0x37 within EX's blocks; unit mapping (DMA, MVM, VU; unit 4 absent); `CAPS` bit5 and `CFG3[31:16]` = PCAP/16; PERF X + 16 … X + 19; KV placement in the heap (§4.5) | EX's w0 format with x16 usable by AT; check order and `OPERAND` (13); no run-time errors; X+2 `ATT_CYCLES` reads 0 while unit 4 is absent; 0x15–0x17, 0x19–0x1B, 0x23 stay reserved for AT |
| VU | The semantics of ASMAX, AMASK, AMERGE and ANORM, which execute on the VU (unit 3; VU §5.11, §7.4); their use profile (§7, §9); Appendix A (FWHT) for a later function | EXP = EXPK(·, P1) with EXP(−inf) = +0, EXP(±0) = 1 exactly, +0 below its underflow threshold, canonical NaN for NaN, at most 0.53 ulp (model); RECIP = `tsr_frecip`; the multi-pass AMERGE at about 4–5 cycles per output line; per-head RMSNorm for Gemma QK-norm (VU §8.1); VOP for the K and V bias and ROPE before QACT |
| CC | Ops per layer (4 + 4·NC) and queue use: MVM at most 3 AT entries in flight, VU at most 4, DMA 2 per layer; checker classes with up to four tagged extents (AMASK and AMERGE use four SPM extents) | Queue depths MVM 4, VU 16, DMA 8 (CC §7.4); two dependencies per op; SPM round robin with the MVM client carrying ATT broadcast and gather traffic; client 5 (AT) tied off; the four-extent checker slot (CC §8.2) |
| DB | ATT ops are ordinary queue entries with no state across ops (every intermediate is in SPM: scores, p, partials, running state), so tile stops and QSTEP need nothing AT-specific; PERF counters (§10) | Completion records carry w0, so profiles separate ATTQK and ATTPV; the watchdog covers the MVM unit (the longest ATT op is about 0.1 ms) |
| SW | Plane layout and address functions (§4); KV tensor identification and lossless `get_tensor` and `set_tensor` (§4.6); the `supports_op` table (§2.6); MSUM format (§7.3); the normative FA algorithm (§6.6); C references (§13.1) | SW §7.2, §7.5 and §9.3: the backend allocates planes from kv_size and E at `init_tensor`, zeroes them on clear, and builds MSUM (and the f32 mask copy when GENERAL) in `set_tensor` of tensors named `kq_mask`; the MSUM is a preload entry. OPL SET_ROWS_KV args 8–11 = {plane base, E/64 \| LG<<16, plane kind, stream}. OPL FLASH_ATTN args 12–13 = {MSUM (preload byte offset), f32-mask handle} and args 16–21 = {K base, V base, E/64 \| LG<<16, D, G, flags}; softcap must be 0. The flags word: bit0 FA_NEW, bits [7:1] 0, bits [15:8] the preload line of the I64 new-cell indices (byte offset / 64), bits [31:16] their count (0 without FA_NEW); the backend points it at the layer's SET_ROWS_KV index entry, whose cells §6.6 rule 7 moves last (the AMERGE order is part of the numerics: without the word the chunk order, hence the output bits, differ from the reference run with those new cells). The `k_fattn` and `k_kv` kernels implement §6.6 and §7. klcheck and numcheck run with `-fa on -ctk q8_0 -ctv q8_0` and `LLAMA_ATTN_ROT_DISABLE=1` on both CPU and device |

---

## 13. Verification

### 13.1 References (`sw/sim/tsr_ref_at.c`, `sw/sim/include/tsr_ref_at.h`, in `libtsr_ref`)

| Function | What it defines |
|---|---|
| `tsr_kv_addr_k`, `tsr_kv_addr_v`, `tsr_kv_pack_row`, `tsr_kv_unpack_row` | §4.3–§4.4 |
| `tsr_ref_kvst` | KVST |
| `tsr_ref_attqk` | per-segment SEG via `tsr_dot_row`, bounds, M_FLOOR max |
| `tsr_ref_attpv` | block float with `__int128` sums, F(), NaN rule, LSUM |
| `tsr_ref_asmax`, `tsr_ref_amask`, `tsr_ref_amerge`, `tsr_ref_anorm` | the vector ops |
| `tsr_ref_fa` | the composite of §6.6, taking MSUM and the chunk policy |

The simulator's units call these functions, and the host C++ reference in `sw/ctrl/ref/` composes them in
kernel order (SW §7.5).

### 13.2 Reference self-tests (`sw/sim/tests/test_ref_at.c`: ctests `ref_at`, `ref_at_ggml`)

- Layout: ggml rows to planes and back for random E (64–4096, multiples of 64), kv_size and C ∈ {1, 2, 3, 4,
  16}, every byte placed once; KVST equals `quantize_row_q8_0_ref` + layout, byte for byte.
- ATTQK SEG equals `tsr_dot_row` on the segment repacked as a standalone Q8_0 row and the short tree
  z(pairwise) (§6.2), over random and directed terms (-0, NaN, Inf, FTZ boundary values of
  [spec](../spec.md) §21).
- ATTPV: A_L against an independent 128-bit model, c_L against an exact rational model rounded once (F), the
  cases p = 0 skip, all-zero segments, non-finite w per block, E_S extremes, q = -128, partial last segments,
  n_groups not a multiple of C, and the double-precision bound |O - exact| ≤ (segments + 1)·2^-24·Σ|p·v|.
- Composite `tsr_ref_fa` against ggml's CPU FA on the same q8_0 cache (NMSE ≤ 5·10^-4, the `test-backend-ops`
  tolerance) over D ∈ {64, 128, 256, 512}, G ∈ {1, 2, 5, 7, 8, 16}, n_kv ∈ {256, 1,024, 4,096, 16,384}, T_q 1–8
  and masks (causal, SWA, wrapped SWA, holes, GENERAL with finite values), and against a double-precision
  softmax.

### 13.3 Independent spec model

`sw/tools/numcheck.py` holds a numpy model of the seven ops and the composite. It is written from this
document only, does not share code with `tsr_ref`, and must agree bit for bit (spec §17, numcheck B1b).

### 13.4 Simulator and traces

- `tsr_sim` executes the ops, with a cycle model: ATTQK and ATTPV: fill + max(max_c beats_c, pass rows ×
  n_groups), plus 68 for ATTPV (the 64-beat segment delay and drain); KVST: 40 + beats + RMW reads; VU ops:
  lines / VU rate.
- The trace generator `--ops at` (`traces/at`) covers every check row of §5.8 and the directed numeric cases
  above, with C ∈ {1, 3, 4, 16} and T_MAX ∈ {1, 8}.

### 13.5 RTL

cocotb block tests cover the cluster ATTQK mode (segment trees at D = 64–512, per-slot addressing, gather line
assembly, -inf bounds with CAUSAL and SLIDE, max trackers), the cluster ATTPV mode (w-unit, delay FIFO, DSP
accumulate and restart at segment boundaries under random stalls, capture and drain, NaN blocks), `tb/tech`
(Xilinx against generic `tsr_mac_lane` in PV mode, cycle by cycle) and `tb/dma` (KVST with RMW interleaved with
LD and ST of the same channel). At core level, `traces/at` and FA op-list traces from real models replay
bit-exact against the simulator under random schedules. The back ends and gather sustain one beat per cycle at
D = 128, GS·T_q = 8, and the predicted gather-bound rate at D = 64.

### 13.6 Backend and model level

- `test-backend-ops`: its K, V and dst tensors are never KV-cache tensors in a KV buffer, so `supports_op`
  refuses its `FLASH_ATTN_EXT` and `SET_ROWS` cases by design (SW-52). `ggml_tesseract_attn` (SET_ROWS +
  FLASH_ATTN_EXT in llama.cpp's graph shape through KV buffers: bit-exact against `tsr_ref_fa`, NMSE vs CPU)
  and `ggml_tesseract_kvstate` (state save and load) carry these checks, on sim, vkernel and the card.
- M6 mode runs with `offload_kqv` = false, so FA runs on the CPU, and the AT ops are verified by the
  `selftest` kernel's test 5, `traces/at` and `ggml_tesseract_attn` (SW §10.4).
- M7 mode (D17 baselines: `-fa on -ctk q8_0 -ctv q8_0`, rotation off, KV on the device): Qwen2.5-0.5B Q8_0
  `--expect-identical` device vs sim; Qwen2.5-32B Q4_K_M mean KL ≤ 3 × floor and top-1 ≥ 80 %; Gemma 4 12B
  the KL criterion; long prompts (2K and 16K tokens) to exercise multiple chunks, prefill CAUSAL tiles and,
  for Gemma, SLIDE, the wrapped SWA ring and the multi-pass full layers.
- numcheck dumps per-layer FA outputs on device and simulator: bit-exact against the numpy model; fp error
  bound against fp64.

---

## Appendix A. FWHT (recorded for a later VU function)

`tsr_ref_fwht(x[n])`, n ∈ {64, 128, 256, 512}, bit-identical to the CPU (`ggml/src/ggml-cpu/ops.cpp`) for
normal inputs:

```
sc = tsr_frecip(tsr_fsqrt((float)n));            // = 1.0f/sqrtf(n)
for i < n: y[i] = tsr_fmul(x[i], sc)
for (len = 1; len < n; len <<= 1)
  for (i = 0; i < n; i += 2·len) for (j = 0; j < len; ++j)
    { u = y[i+j]; v = y[i+len+j]; y[i+j] = tsr_fadd(u, v); y[i+len+j] = tsr_fsub(u, v); }
```

The CPU's SIMD stages use FMA(u, v, −1), which is exactly u − v. Stage order and pairing are the same, so the
result matches the CPU except for subnormals (FTZ).

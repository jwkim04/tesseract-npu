# Tesseract — attention and the KV cache on the device (area AT)

Status: **design contract for M6, proposal** (2026-10-30). One of the six area documents in
`docs/design/` (EX execution model, VU vector unit, AT attention and KV (this file), CC control core,
DB debug, SW software and verification). It implements user decision D17 (`docs/ctrl-riscv.md`, user decisions of 2026-10-30):
the KV cache lives on the device in q8_0, and attention arrives as llama.cpp `FLASH_ATTN_EXT`.
Source-of-truth order is the project's: `docs/spec.md` > ISA simulator (`sw/sim`) > this document.
The spec changes this document needs are listed in §14; they are applied in W0
(`docs/design/README.md`). Every number marked *estimate* is not measured. Gemma 4 12B shapes are read
from its GGUF (`models/gemma4/gemma-4-12b-it-Q8_0.gguf`, 2026-10-30); 31B shapes marked *confirm*
are recalled from the HF config and must be read from a GGUF before they are relied on.

llama.cpp citations are relative to `third_party/llama.cpp` (pinned at e3797b184).

---

## 1. Summary

- **Attention runs in the MVM clusters**, not in the VU. One SPM port moves 64 B per cycle; the
  four DDR channels deliver 256 B per cycle. Only the clusters, each streaming its own channel,
  can consume the KV cache at the DDR rate (§3).
- **The KV cache holds exactly llama.cpp's q8_0 values.** Same 32-element blocks, same fp16
  scales, same `quantize_row_q8_0_ref` rounding. Only the byte placement changes, so `get_tensor`
  and `set_tensor` convert losslessly (§4.6).
- **Layout.** Cells form 16-cell *groups*; group u lives in channel u mod C. K planes are
  token-major (a cell's whole row is contiguous); V planes are *slice-major* (64-value slices of
  all cells are contiguous). Both use the existing Q8_0 chunk format (§4).
- **Seven new ops** in the opcode blocks EX reserved for AT (`docs/design/01-execution-model.md` §7.2):
  `KVST` 0x14 (DMA unit, store new K/V rows), `ATTQK` 0x21 and `ATTPV` 0x22 (MVM unit, scores and
  P·V over a group range), and `ASMAX` 0x34, `AMASK` 0x35, `AMERGE` 0x36, `ANORM` 0x37 (VU unit,
  softmax steps, executed by the VU's lanes, VU §5.11). Unit 4 stays absent in M6 (§5).
- **Numerics.** Scores use the spec's Q8_0 row rule on each head segment. P·V uses an exact
  integer accumulation with a 24-bit block-floating-point weight per 64-cell segment and one f32
  addition per segment. The softmax is two-pass within a chunk and online across chunks. No
  precision is dropped below what the CPU keeps (§6).
- **GQA.** Query heads of one KV head (and up to 8 query tokens in prefill) are the MVM's T
  "token slots". Head subsets cover G > 8 (§5.2), e.g. Gemma 4 12B's global layers (G = 16: two
  passes per decode token).
- **Throughput** (*estimate*). Qwen2.5-32B decode at 4K context: about 40 K core cycles per layer
  against 35.7 K of pure KV streaming (89 %). Attention is about 3 % of the token time at 4K and
  about 19 % at 32K (§9).
- **Masks.** The host backend reduces llama.cpp's f16 KQ mask to a *mask summary* of cell runs
  (§7.3). The device never reads f16. Causal and sliding-window prefill tiles use per-row bounds
  inside `ATTQK`; anything else takes a general f32 path.
- **Kept out:** attention logit softcap, ALiBi, sinks, sparse `n_kv_max`, DK ≠ DV (MLA), head
  dimensions other than 64–512, and K-shift. The KV Hadamard rotation is off, following SW's
  SW-16 (§2.5).
- **DQROW** is VU's (VU §5.6, SPM → f32 on the VU, RTL in M8). Attention never uses it, and
  `tok_embd` stays on the CPU input layer through M7 (§5.9).
- **Gates.** M6 runs llama.cpp with attention on the CPU (`offload_kqv` = false); the AT ops are
  verified on their own in M6, and device attention with the KV cache on the device is gated in M7
  (§13.6).

---

## 2. What llama.cpp hands the device (pinned commit)

### 2.1 Graph shape

**Qwen2.5** (`src/models/qwen2.cpp:83-104`): Q, K and V projections with bias, RoPE NEOX on Q and
K, then `build_attn(..., kq_scale = 1/sqrtf(n_embd_head))`.

**Gemma 4** (`src/models/gemma4.cpp`):
- embedding × √n_embd (`:171`);
- Q RMSNorm per head with weight (`:232`);
- K RMSNorm per head with weight, and V `ggml_rms_norm` without weight (`:263-264`);
- V = K's projection when `wv` is absent (`:255`);
- RoPE with `freq_factors` on full-attention layers;
- attention scale 1.0 (`:12`);
- an iSWA cache: two caches, the SWA one sized `pad(min(size, n_swa·n_seq + n_ubatch), 256)`
  (`src/llama-kv-cache-iswa.cpp:73`);
- different head dimensions on SWA and full layers (`conversion/gemma.py:696-701`).

There is no attention softcap: `attn_soft_cap` is never set for Gemma 4. Its final-logit softcap
is not attention.

**`build_attn` with a KV cache** (`src/llama-graph.cpp:3216-3290`):
1. optional Hadamard rotations of Q, K and V (`:3227-3234`);
2. `cpy_k` and `cpy_v`, which are `SET_ROWS` into the caches (`:3254-3255`);
3. `get_k` and `get_v` views;
4. `build_attn_mha`;
5. optional output un-rotation (`:3272-3274`);
6. `wo`, `wo_b`.

**`build_attn_mha`, FA branch** (`:2975-3018`):
- q, k and v are permuted;
- `ggml_flash_attn_ext(q, k, v, kq_mask, kq_scale, f_max_alibi_bias, softcap)` (`:2992`);
- sinks (`:2996`), `n_kv_max` (`:2998`) and F32 precision (`:2999`) are applied;
- the result is reshaped to `[n_head·D, n_tokens]`.

### 2.2 KV cache tensors

- `k` and `v` are 3D `[E, kv_size, n_stream]` with E = n_head_kv·D, type q8_0
  (`src/llama-kv-cache.cpp:233-234`), named `cache_*k_l<il>` / `cache_*v_l<il>`.
- The buffer is cleared to zero after allocation (`:293`).
- With FA, V is not transposed (`:1404-1418`), so both caches have one q8_0 row per cell.
- The FA views `get_k` and `get_v` (`:1301-1351`) are `[D, n_head_kv, n_kv, n_stream]` with row
  stride `row_size(E)`.
- `n_kv` is padded to max(n_pad, 256) (`:1285-1298`).
- The pinned commit refuses a quantized V cache without FA (`src/llama-context.cpp:566-569`,
  `:4079-4088`).

### 2.3 `FLASH_ATTN_EXT` semantics (CPU reference)

API (`ggml/include/ggml.h:2482-2519`):
- q `[D, n_batch, n_head, ne3]`, k and v `[D, n_kv, n_head_kv, ne3]`;
- mask `[n_kv, n_batch, ne32, ne33]`, F16;
- result `[D, n_head, n_batch, ne3]`, permuted, so each token's heads are contiguous;
- GQA: kv head = q head / (n_head / n_head_kv), so consecutive query heads share a KV head.

For q8_0 K and V the CPU takes the per-row `one_chunk` path (`ggml/src/ggml-cpu/ops.cpp:8933-9169`).
Tiling and split-KV apply only to F16/F32 K (`:9580`, `:9637`). Per query row:
1. Q is quantized with the K type's `vec_dot_type` (q8_0, x86 `quantize_row_q8_0`) (`:9051`).
2. For each cell: if the mask is −inf, the cell is **skipped** (`:9059`). Otherwise
   `s = vec_dot_q8_0_q8_0(k, q)·scale`, with a softcap if one is set (`:8998`, `:9071`), and the
   mask value is added.
3. Online softmax with `expf`.
4. V is dequantized to f32 (`:9113`) and accumulated as `VKQ += v·expf(s − M)` in f32.
5. Finally `S_inv = S == 0 ? 0 : 1/S` (`:9157`).

### 2.4 `SET_ROWS` and the mask

- `SET_ROWS` (`ggml/include/ggml.h:1743-1758`; CPU `ops.cpp:5530-5620`) quantizes each f32 source
  row with the destination type's `from_float` (`quantize_row_q8_0`, `ggml-cpu.c:272-275`) into row
  `idx[i]` (I64).
- The mask is F16 `[n_kv, n_tokens/n_stream, 1, n_stream]` (`src/llama-graph.cpp:1051`). Its
  values are 0 or −inf (`src/llama-kv-cache.cpp:1589-1590`, `:1714`, `:1719`).
- Masking: future positions (p0 > p1), other sequences and empty cells are masked. SWA standard
  masks p1 − p0 ≥ n_swa (`src/llama-hparams.h:506-540`).

### 2.5 Hadamard KV rotation

With a quantized cache, uniform head dimensions in the cache and head_dim % 64 = 0, llama.cpp
rotates Q and K by an orthonormal Walsh-Hadamard matrix of size n_rot_k, and V by one of size
n_rot_v = 64 (`src/llama-kv-cache.cpp:316-365`). It un-rotates the attention output. The rotation
is a `MUL_MAT` with an F32 graph-input src0 and the hint `GGML_HINT_SRC0_IS_HADAMARD`
(`src/llama-impl.h:57-77`), which the CPU runs as a fast Walsh-Hadamard transform
(`ggml-cpu.c:1262-1266`, `ops.cpp:12432-12505`).

SW decided to run every device run and every baseline with `LLAMA_ATTN_ROT_DISABLE=1` (SW-16,
`docs/design/06-software-verification.md` §8 R4). AT adopts that (AT-17). The device attention is
agnostic to rotation: it computes on whatever q, k and v the graph supplies. Appendix A records the
bit-exact FWHT definition, in case a later milestone offloads the rotation as a VU function.

### 2.6 What the device accepts (backend `supports_op` contract, M6/M7)

| Op | Device accepts when | Otherwise |
|---|---|---|
| `FLASH_ATTN_EXT` | q F32 contiguous per token (`[D, n_head, T]` before the permute); k and v q8_0 views of KV-cache tensors in a Tesseract KV buffer (§4.6); D = DK = DV ∈ {64, 128, 256, 512}; n_head % n_head_kv = 0; mask F16 or none; `max_bias` = 0, `logit_softcap` = 0, no sinks, `n_kv_max` = 0; precision default or F32; ne3 = n_stream (M7: 1). As built (W3b): k / v start at the plane's cell 0 with cells `row_size(E)` and heads `row_size(D)` apart (llama.cpp's `get_k` / `get_v`); T ≤ 680, so that the MSUM and the FA_NEW indices fit one list's 256 preload lines (SW SW-43; a longer ubatch's FA runs on the CPU) | false. With a device KV buffer the scheduler would then copy the cache per op, so the backend reports the unsupported feature at graph build (SW R10) |
| `SET_ROWS` | dst is a KV-cache tensor (q8_0, Tesseract KV buffer) viewed from cell 0 with rows of E values (`cpy_k` / `cpy_v`, one stream); src F32 with rows of E; idx I64 < 2^31 | false; the scheduler then stops on the pre-allocated cache tensor, after the backend names the cause (SW R10) |
| Hadamard `MUL_MAT` | never (rotation is off, AT-17) | — |
| Other attention ops (`SOFT_MAX`, attention `MUL_MAT` on the cache) | never (D17: FA only) | — |

The KV type must be q8_0 for both K and V when the cache is offloaded. Other KV types are
documented as unsupported (run with `-nkvo`).

---

## 3. Why the clusters, not the VU

- **SPM bandwidth.** The core's SPM port moves 64 B per cycle; the four channels together deliver
  256 B per cycle at 97–98 % beat efficiency (`docs/platform-u200.md` §14.1). Any scheme that loads
  K/V tiles into SPM and computes from there reaches at most 25 % of the DDR rate.
- **VU arithmetic.** The VU processes 16 f32 per cycle. A q8_0 beat holds about 60 values.
- **Clusters.** Each cluster consumes its own channel at one beat per cycle and has 512 DSPs
  (T_MAX 8) that sit idle while the decode token is in its attention phase.
- **Work per KV byte.** Attention needs about G multiply-adds per KV byte for Q·Kᵀ and G for P·V.
  G is the GQA ratio, ≤ 8 for the target models. 4 channels × 64 values × 8 = 2,048 per cycle is
  what the clusters' MAC lanes already provide.

So Q·Kᵀ is a segmented form of the existing Q8_0 MVM (K rows stream as weights, query heads are
token slots). P·V is a new accumulate mode of the same lanes (§8). The VU only does the
O(rows × cells) softmax pass and the O(rows × D) merges.

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

Group u lives in **channel c = u mod C** at local index **ℓ = ⌊u / C⌋**. Each plane occupies the
same beat range [B, B + LG·GB) in every channel (symmetric allocation, `docs/spec.md` §4).
A run of n groups is therefore split over the channels to within one group. A 16-cell group gives
exactly one 64-byte line of f32 scores per head row (§5.2), so every score line comes from one
cluster.

### 4.3 K plane (token-major)

Within a group, the 16 cell rows are concatenated in cell order as a sequence of 16·E/32 q8_0
blocks. That sequence is cut into Q8_0 chunks of 32 blocks (spec weight layout,
`docs/spec.md` §5: one meta beat with 32 fp16 scales, then 16 quant beats). For cell
r = j mod 16 and block b of its row (0 ≤ b < E/32):

```
β = r·(E/32) + b;   κ = β >> 5;   o = β & 31
meta : beat B + ℓ·GB + κ·17,               bytes [2o, 2o+2)   = fp16 scale d
quant: beat B + ℓ·GB + κ·17 + 1 + (o>>1),   bytes [32(o&1), +32) = q[32]
```

For E = 1024 (Qwen2.5-14B/32B) a cell row is exactly one chunk (17 beats). For E < 1024, 1024/E
rows share a chunk; for E > 1024, a row spans E/1024 chunks.

### 4.4 V plane (slice-major)

For slice s of a cell in local group ℓ, the chunk at beat **B + (s·LG + ℓ)·17** holds the 16
cells' slice s:

```
meta : bytes [4r, 4r+4)  = fp16 d of blocks 2s and 2s+1 of cell r
quant: beat (chunk + 1 + r), all 64 bytes = q[64s .. 64s+63] of cell r
```

So a slice-s region of a channel, [B + s·LG·17, B + (s+1)·LG·17), is contiguous over all its
groups. That is what `ATTPV` streams (§8.2). The plane size per channel is (E/64)·LG·17 = LG·GB
beats, the same as K.

### 4.5 Region, alignment, capacity

- **Region.** KV planes come from the heap above 512 MiB that EX defines (`01-execution-model.md`
  §11). They are allocated symmetrically, top-down from the end of each channel. Weights grow
  bottom-up from 512 MiB. The allocator places one plane per (layer, stream, K/V).
- **Alignment.** B is beat-aligned. 64-beat (4 KiB) alignment is recommended, not required; the
  requester splits bursts at 4 KiB anyway (`docs/spec.md` §2, Core boundary interface).
- **Capacity:** per cell, Σ over layers of 2·E·34/32 bytes.

| Model (D, n_head_kv, layers) | Bytes per cell | 4K cells | 32K cells | Per channel at 32K |
|---|---|---|---|---|
| Qwen2.5-0.5B (64, 2, 24) | 6,528 | 26.7 MB | 214 MB | 53 MB |
| Qwen2.5-7B (128, 4, 28) | 30,464 | 125 MB | 998 MB | 250 MB |
| Qwen2.5-32B (128, 8, 64) | 139,264 | 570 MB | 4.56 GB | 1.14 GB |
| Gemma 4 12B, 8 full layers (512, 1) | 8,704 | 35.7 MB | 285 MB | 71 MB |
| Gemma 4 12B, 40 SWA layers (256, 8), 1,536-cell SWA cache | 174,080 | 267 MB (fixed) | 267 MB | 67 MB |
| Gemma 4 31B, 10 full layers (512, 4) *confirm* | 43,520 | 178 MB | 1.43 GB | 357 MB |
| Gemma 4 31B, 50 SWA layers (256, 16), 1,536-cell SWA cache *confirm* | 435,200 | 668 MB (fixed) | 668 MB | 167 MB |

Gemma 4 12B (GGUF): 182,784 B per cell over all 48 layers. Its SWA cache holds
pad(min(size, n_swa·n_seq + n_ubatch), 256) = 1,536 cells (n_swa 1024, n_ubatch 512). Its global
layers have E = n_head_kv·D = 512 < 1024, so two cells share a K chunk and every K meta beat that
KVST touches is read-modify-written (§5.1). They have no `attn_v`: V is the K projection,
normalized without weight, and is stored in its own V plane like any other V.

32B Q4_K_M weights take about 5.2 GB per channel (`docs/spec.md` §4). Weights plus a 32K cache
use about 40 % of a 16 GiB channel; 128K contexts also fit (*estimate*).

### 4.6 Correspondence with ggml, identification, clearing

- **Identification.** The backend treats a tensor as a KV-cache tensor when it is q8_0, 3D
  `[E, kv_size, n_stream]` in a Tesseract buffer, named `cache_*k_l<n>` or `cache_*v_l<n>`
  (`src/llama-kv-cache.cpp:236-237`). Its layout is fixed by E and kv_size, both known at
  `init_tensor`, and does not depend on D. As built (W3b, SW SW-47): the tensor is recognised from
  the `init_tensor` records, and the planes are placed at the buffer's first data access (llama.cpp's
  clear), because ggml sets a buffer's usage after `init_tensor`; a buffer counts only if all its
  tensors qualify.
- **Clear.** The plane is zeroed on clear (`:293`): zero scales and zero quants, which is also the
  ggml image.
- **Lossless conversion.** `set_tensor` and `get_tensor` on row ranges (state save and load)
  convert between ggml rows (`block_q8_0 {fp16 d, int8 qs[32]}`, 34 B) and the plane by the address
  functions of §4.3 and §4.4. This is a pure byte permutation, so state files are bit-identical to
  the CPU's.
- **Shared references.** `tsr_kv_addr_k`, `tsr_kv_addr_v` and `tsr_kv_{pack,unpack}_row` live in
  `sw/sim/tsr_ref_at.c` (`include/tsr_ref_at.h`, part of `libtsr_ref`) and are shared by the
  simulator, the backend and the RTL package.

---

## 5. Operations

All seven ops follow the common encoding: w0 = {x16 = w0[31:16], flags[15:12], sub[11:8], opcode[7:0]}
(`01-execution-model.md` §7.1). Every reserved field or bit must be 0, otherwise `ILLEGAL_OP`.
Every error is found at dispatch; nothing fails after it has started. Data-dependent conditions
(NaN, empty rows, zero weights) have defined results.

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
| R | T_q·n_head | rows of the score / output matrices |
| row r(q, g, i) | q·n_head + g·G + hg0 + i | for q < T_q, g < E/D, i < GS |
| slot t(q, i) | q·GS + i | MVM token slot, < T_MAX |

The rows touched by one op are its *pass rows*. Rows of other passes (other hg0) are left
unchanged.

### 5.1 `KVST` — store K or V rows into a plane (0x14, unit 0 DMA)

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

Semantics: for i < n_rows, cell c0 + i receives row i. The source is QA vector i at
src + i·stride. Its q bytes are QS[0..E) and its scales are fp16(D[b]) with `tsr_f32_to_fp16`;
this conversion is exact for QACT output. Bytes are placed per §4.3 or §4.4.

- The software sequence for `SET_ROWS` is `QACT A32 (K = E, T rows)` followed by `KVST`. QACT A32
  is `quantize_row_q8_0_ref` (`docs/spec.md` §6), so the stored values equal llama.cpp's except
  where x86 `quantize_row_q8_0` rounds a tie differently (about 1 value in 3–4·10^5,
  `docs/spec.md` §6).
- **Partial meta beats** (K with E % 1024 ≠ 0; every V cell) are written by read-modify-write
  inside the DMA unit: read the beat, merge the bytes, write it. This is safe because DMA ops run
  one at a time in queue order, and readers in other units are ordered by dependencies (memory
  model, `docs/ctrl-riscv.md` §5.5). A meta beat that the op fully covers is written without a
  read. No core-boundary change (write strobes) is needed. A later revision may switch to byte
  enables with an identical architectural result.
- Cells c0..c0+n_rows−1 must be consecutive. The kernel splits non-consecutive `SET_ROWS` indices
  into runs.

### 5.2 `ATTQK` — scores and row maxima (0x21, unit 2 MVM)

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

- **SEG** is the spec's Q8_0 row dot product (`docs/spec.md` §7, `tsr_dot_row`) applied to
  the D-element segment [g·D, (g+1)·D) of the cell's K row, with the QA blocks of head h of query
  vector q (§6.2).
- **max_tot** uses the spec's total order (−inf < … < +inf < NaN, `docs/spec.md` §9).
- **M_FLOOR** = −2^100 (0xF1800000, §6.3).
- **CAUSAL and SLIDE** describe a prefill tile whose T_q query tokens occupy consecutive cells:
  row q sees [lo_q, hi_q). With c_hi = (cell of the last query) + 1, CAUSAL gives each query its
  own cell as the last visible one. SLIDE moves the window start with the query (SWA).

### 5.3 `ATTPV` — P·V partial outputs and row sums (0x22, unit 2 MVM)

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

Output, for every channel c < C and pass row r:
- O_c[r][d] at line dst + c·cstride + r·D/16 + ⌊d/16⌋, word d mod 16;
- with LSUM, l_c[r] at line dst + C·cstride + c·⌈R/16⌉ + ⌊r/16⌋, word r mod 16.

The numeric definition is §6.4–6.5. Channel c accumulates only the cells of its own groups, so the
op produces C partials. AMERGE adds them in a fixed tree.

### 5.4 `ASMAX` — softmax numerator (0x34, unit 3 VU)

w1 dst, w2 src (may equal dst), w3 mvec (row maxima, word r), w4 [15:0] n_lines (1–4095),
[31:16] stride (0 → n_lines), w5 R (1–1024), w6 scale (f32 bits), w7 0.

```
for r < R, k < 16·n_lines:   dst[r][k] = EXP( fmul( fsub(src[r][k], mvec[r]), scale ) )
```

Rows are at src + r·stride and dst + r·stride. `scale` must be finite, normal and > 0, otherwise
`OPERAND`. The positive scale is what makes max(s) the right offset. EXP is VU's `tsr_m_exp`
(§11).

### 5.5 `AMASK` — general mask path (0x35, unit 3 VU)

w1 dst, w2 src (rows at stride), w3 mask (f32 rows), w4 [15:0] n_lines, [31:16] stride,
w5 [15:0] R, [31:16] rpq (rows per query, = n_head), w6 mask stride (lines between query rows),
w7 mdst (row maxima out).

```
q = ⌊r / rpq⌋;  m = mask[q][k]
dst[r][k] = (bits(m) == 0xFF800000) ? −inf : fadd(src[r][k], m)
mdst[r]   = max_tot(M_FLOOR, max_k dst[r][k])
```

The −inf test reproduces the CPU's skip of masked cells (§2.3). A finite mask value is added to
the raw score, and ASMAX / AMERGE scale the sum (AT-9): the device computes EXP(scale·(s + m)),
where the CPU computes exp(scale·s + m). The two agree when m = ±0 or scale = 1. So the backend's
f32 copy holds fdiv(m, scale) (§7.3; `tsr_msum_mask_f32`), which gives the CPU's form up to f32
rounding for every finite m (W1a decision; NMSE ≈ 10^-13 against ggml-cpu in `ref_at_ggml`).

### 5.5a Optional local FAST attention

`CAPS.ATT_LOCAL` (bit 12) permits three local score/probability slots on a
four-channel core. Slot 0 keeps the existing SPM operation. ATTQK and ATTPV
select slots 1–3 in `w0[29:28]`; ASMAX uses `w7[1:0]` for the slot and
`w7[17:2]` for the chunk's global first group `u0`. Reserved bits remain zero.
The feature is optional and does not change the FLASH_ATTN record.

The firmware selects local sequences only for FAST masking. Raw local ATTQK
requires MAXOUT; raw local ASMAX requires `w1 == w2`. There is no raw
record-level FAST tag. Local geometry permits at most 224 full query rows,
at most 128 groups per chunk, and at most 4,480 lines per owner per slot.
Full rows include every GQA subset. Define `stride = round_up(groups, 4)`;
`rows * stride / 4` must fit the owner depth. The shared helper is
`sw/sim/include/tsr_local_tile.h`. The simulator and raw instruction checker
reject unsupported geometry before execution; absence of the capability does
not reinterpret a nonzero slot as SPM.

Ownership remains global group `u mod 4`, including nonzero `u0`. The local
address derives from the full row/relative-group position, divided by four.
ATTQK commits score lines locally and sends compact per-line maxima to the
existing global maximum path. ASMAX receives that common maximum and uses
existing VU EXP lanes in the same outer group. ATTPV reads P locally into the
existing activation buffer. The operation does not write scores/P to their
legacy SPM destinations; those fields still satisfy SPM range/extent checks
and retain geometry and dependency meaning.
PV partials, LSUM, AMERGE and ANORM retain their existing shared path and
`((c0+c1)+(c2+c3))` arithmetic order.

There are three physically separate 1R1W banks per owner. One may produce QK,
one may process ASMAX, and another may supply P to PV. A slot is not a global
SPM alias or a DDR handle. Software owns its reuse: QK completion precedes
ASMAX, ASMAX completion precedes PV, and PV's consumption precedes the next
QK that reuses that slot. The supplied firmware relies on the ordered MVM
queue and explicit VU dependencies. There is no independently tagged hardware
generation or automatic rollback for malformed raw schedules. Completion
follows the last required write; reset/abort follows EX's existing drain
contract, including accepted DDR responses.

The firmware selects local or legacy for the whole record before issuing
writes. It preflights all selected query tiles/chunks and uses the legacy path
for GENERAL, mixed or oversized records. Multi-tile records whose first tile
has at most 16 groups also retain legacy execution to avoid planning overhead.
This coverage rule does not reject previously accepted FLASH_ATTN records.
It can leave later large chunks of a causal prefill on the legacy path.
Legacy and local commands coexist on the same image. Local slots are private
to the record; list cuts, CPU readers and final outputs still use DDR.

### 5.6 `AMERGE` — channel tree and online merge (0x36, unit 3 VU)

flags w0[12] INIT. w1 Orun (R rows of D/16 lines), w2 parts (an `ATTPV` dst written with LSUM),
w3 mk (chunk row maxima), w4 st (m_run vector at line st, l_run vector at line st + ⌈R/16⌉),
w5 [15:0] R, [17:16] dcode, [31:18] 0, w6 scale (f32, as ASMAX), w7 cstride (as the producing
ATTPV).

```
Ok[r][d] = CTREE_c( parts O_c[r][d] );   lk[r] = CTREE_c( parts l_c[r] )
INIT:  Orun = Ok;  l_run = lk;  m_run = mk
else:  m' = max_tot(m_run[r], mk[r])
       a  = EXP(fmul(fsub(m_run[r], m'), scale));  b = EXP(fmul(fsub(mk[r], m'), scale))
       Orun[r][d] = fadd(fmul(Orun[r][d], a), fmul(Ok[r][d], b))
       l_run[r]   = fadd(fmul(l_run[r], a), fmul(lk[r], b));   m_run[r] = m'
```

**CTREE** over c = 0..C−1 is pairwise by levels, left to right. At each level, element 2k + 1 is
added to element 2k, and an unpaired last element passes unchanged. For C = 4 that is
((x0 + x1) + (x2 + x3)); for C = 3, ((x0 + x1) + x2).

### 5.7 `ANORM` — final normalization (0x37, unit 3 VU)

w1 out (row r at out + r·D/16: exactly the FA output layout `[token][head][D]`), w2 Orun, w3 l_run,
w4 [15:0] R, [17:16] dcode, w5–w7 0.

```
out[r][d] = fmul( Orun[r][d], RECIP( max_tot(l_run[r], 2^-126) ) )      RECIP = tsr_frecip (IEEE 1/x)
```

The 2^-126 floor makes an empty row (l = 0, O = 0) produce 0, as the CPU's `S == 0 ? 0 : 1/S`
does. Every nonzero l is at least 2^-126 because of FTZ, so the floor changes nothing else.

### 5.8 Checks and error codes

The order is EX's (`01-execution-model.md` §8.1): `ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` →
`K_RANGE` → `T_RANGE` → `OPERAND` → `SPM_RANGE` → `DRAM_RANGE`. No new code is needed.

| Op | `K_RANGE` | `T_RANGE` | `OPERAND` (`ERR_INFO` = the value) | `SPM_RANGE` / `DRAM_RANGE` extents |
|---|---|---|---|---|
| KVST | E/64 = 0 | — | LG = 0 (0); c0 + n_rows > 16·C·LG (c0) | src: (n_rows−1)·stride + QA_LINES(A32, E); plane: B + LG·GB |
| ATTQK | E/64 = 0; G·E > K_MAX | T_q·GS > T_MAX | E % D ≠ 0 (E); hg0 + GS > G (hg0 + GS); n_groups ∉ [1, 4095] (n_groups); c_lo > c_hi or c_hi > 16·n_groups (c_hi); pass rows (E/D)·GS·T_q > 128 (rows); 0 < rstride < n_groups (rstride) | dst: (max pass row)·rstride + n_groups; x: T_q·QA_LINES(A32, G·E); mdst: ⌈(max pass row + 1)/16⌉; plane: B + (ℓ_max+1)·GB, ℓ_max = ⌊(u0 + n_groups − 1)/C⌋ |
| ATTPV | as ATTQK | as ATTQK | as ATTQK except c_lo/c_hi; ℓ_max ≥ LG (u0 + n_groups − 1); (E/D)·16·⌈n_groups/C⌉ > PCAP (n_groups); 0 < pstride < n_groups | p: (max pass row)·pstride + n_groups; dst: (C−1)·cstride + R·D/16, with LSUM + C·⌈R/16⌉ after dst + C·cstride; plane: B + ((E/64 − 1)·LG + ℓ_max + 1)·17 |
| ASMAX | — | — | n_lines ∉ [1, 4095]; R ∉ [1, 1024]; scale not finite, not normal or ≤ 0 (scale bits) | src, dst: (R−1)·stride + n_lines; mvec ⌈R/16⌉ |
| AMASK | — | — | as ASMAX for n_lines and R; rpq = 0 | src, dst, mask ((⌈R/rpq⌉−1)·mstride + n_lines), mdst |
| AMERGE, ANORM | — | — | R ∉ [1, 1024]; AMERGE scale as ASMAX | every operand by its extent |

The `ILLEGAL_OP` conditions in addition to reserved fields: `KVST` sub ≥ 2; `ATTQK` w7 ≠ 0 without
MAXOUT; `ATTQK` and `ATTPV` x16[15:12] (w0[31:28]) ≠ 0.

PCAP is the p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX)·16 = 9,344 on the
U200 (the activation buffer of one token slot, `docs/rtl-m3.md` §5). It is advertised in CFG3
(§5.10).

### 5.9 DQROW (0x12)

DQROW is VU's op (EX §7.2: VU owns the encoding): SPM → f32 dequantisation on the VU after an LD
of the repacked row (VU §5.6), simulator from M6, RTL in M8.

- **Not needed by attention, nor in M6/M7.** K and V are never dequantized outside `ATTPV`.
  llama.cpp keeps the input layer, and so the `tok_embd` `GET_ROWS`, on the CPU
  (`src/llama-model.cpp:1688-1689`; SW R2); the f32 row arrives as an input of the device split.
- **M8** (device decode step, spec stage B): the embedding lookup is LD (ROW_BEATS beats of the
  row) + DQROW, bit-identical to `dequantize_row_*` for Q8_0 and Q4_K tensors
  (`docs/spec.md` §21, Agreement with ggml).
- **The f32 alternative.** When the GGUF `tok_embd` type is neither Q8_0 nor Q4_K, a DQROW of the
  load-time Q8_0 conversion differs from the CPU's `GET_ROWS` in every element. The exact
  alternative is an f32 copy of `tok_embd` (3.1 GB for Qwen2.5-32B, *estimate*) and an `LD` of the
  row. That choice belongs to M8.

### 5.10 CAPS, CFG3, units

- `CAPS` bit5 ATT (EX): all seven AT ops are present. Bits 12–15 are reserved for AT and 0 at Tesseract Core capability 1.0.
- `CFG3[31:16]` = PCAP / 16 (U200 T_MAX 8: 584).
- **Units.** KVST is in the DMA class (unit 0). ATTQK and ATTPV are in the MVM class (unit 2).
  ASMAX, AMASK, AMERGE and ANORM are in the VU class (unit 3); the VU executes them (VU §5.11,
  VU-27). **Unit 4 is absent in M6** (`ISSUE_UNITS[4]` = 0, `Q_DEPTH[4]` = 0, CC's SPM client 5
  tied off) and stays reserved for a future dedicated attention engine. EX's PERF index X+2
  `ATT_CYCLES` therefore reads 0. AT's counters are X + 16 … X + 19 (§10, EX §6).
- **Opcodes.** The AT quantize-class block 0x19–0x1B and 0x15–0x17, 0x23 stay reserved for AT.

---

## 6. Numerics (bit-exact definitions)

The fp rules are the spec's: DAZ and FTZ, RNE, canonical NaN (`docs/spec.md` §7 and §21, FTZ decision point).
fadd, fsub, fmul and frecip are `tsr_fadd` etc. (`sw/sim/tsr_ref.c:99-104`).

### 6.1 KV quantization

As §5.1: QACT A32 values. The stored scale is fp16 of the QA D value, which is exact.

### 6.2 Scores (`ATTQK`)

Segment g of cell j's K row is blocks b = g·D/32 + k, k < D/32. The query QA vector q holds head h
in its blocks h·D/32 + k: QA blocks are 32-aligned within heads, so one QACT over the whole Q row
(K = n_head·D) equals per-head QACT.

```
t_k = fmul( fmul( fp16_to_f32(dK_b), Dx_q[h·D/32 + k] ), i32_to_f32( Σ_{i<32} qK_b[i]·qx[...] ) )
SEG = tsr_tree16(p) with p_k = fadd(+0.0, t_k) for k < D/32, p_k = +0.0 otherwise
```

This is exactly `tsr_dot_row(Q8_0, segment-as-row, K = D, head-h QA)`, the spec row rule with
16 partials (`docs/spec.md` §7). For D/32 = 2^m ≤ 16 it equals z(pairwise tree of t_0..t_(D/32−1)),
where z maps −0 to +0; the hardware may use that short tree. The reference test proves the
equality (§12).

### 6.3 Row maximum and M_FLOOR

m_r = max_tot(M_FLOOR, scores), with M_FLOOR = −2^100.

- A row with no visible cell in a chunk gets m = M_FLOOR. Its p then equals EXP((−inf − M_FLOOR)·scale)
  = EXP(−inf) = +0.
- Merges stay finite: a = EXP((M_FLOOR − m')·scale) = 0 against a real maximum, and EXP(0) = 1
  between two floors.
- A real score below −2^100 would underflow its p to 0. That needs |q|·|k| > 10^30, which is not a
  practical case.

### 6.4 P·V (`ATTPV`): block floating point per segment

Per channel c, slice s (head group g = ⌊s/(D/64)⌋), slot t = t(q, i) and row r = r(q, g, i):

1. **Segments.** U_c = the op's groups u ≡ c (mod C), ascending. Segments are consecutive runs of
   4 elements of U_c (64 cells; the last segment may be shorter). The order is fixed by u0 and C,
   so the result depends on NUM_CH (AT-11).
2. **Weights.** For each segment S, each cell j ∈ S with p_r[j] not ±0, and e ∈ {0, 1}:
   w_{j,e} = fmul(p_r[j], fp16_to_f32(dV_{j, 2s+e})). Cells with p = ±0 are skipped (they
   contribute nothing, like the CPU's skip of −inf cells).
3. **Non-finite rule.** If any w_{j,e} is NaN or ±Inf, then c_L = canonical NaN for every lane L
   of block e (L ∈ [32e, 32e+32)).
4. **Exponent.** E_S = max of ε(w) = ⌊log2|w|⌋ over the finite nonzero w_{j,e} of both blocks. If
   there are none, c_L = +0 for all L.
5. **Integer weights.** W_{j,e} = RNE_int(w_{j,e} · 2^(23 − E_S)), an exact power-of-two scaling
   followed by round-half-even to an integer. |W| < 2^24.
6. **Exact sum.** A_L = Σ_{j∈S} W_{j,⌊L/32⌋} · qV_j[64s + L], an exact integer with |A| < 2^37.
7. **Segment value.** c_L = F(A_L, E_S − 23), where F(A, k) = RNE24(A)·2^k: A rounded to 24
   significant bits with RNE, then scaled exactly. A result below 2^-126 becomes ±0 (sign of A);
   overflow gives ±Inf.
8. **Accumulation.** O[L] starts at +0.0, and O[L] = fadd(O[L], c_L) per segment in segment order.
   After the last segment, O_c[r][(s − g·D/64)·64 + L] = O[L].

Each product keeps at least 24 significant bits relative to the segment maximum. One f32 addition
is made per 64 cells, against the CPU's one fused multiply-add per cell. The device is no less
precise than the CPU.

### 6.5 Row sums (`ATTPV` LSUM)

The same segments of U_c, once per row (not per slice):

- E_p = max ε(p) over the finite nonzero p_r[j] of the segment;
- P_j = RNE_int(p_r[j]·2^(23 − E_p));
- cl = F(Σ P_j, E_p − 23), or canonical NaN if any p in the segment is non-finite, or +0 if no p
  is finite and nonzero;
- l_c[r] = fadd(l_c[r], cl) per segment, from +0.0.

### 6.6 The composite FA (normative kernel algorithm)

Per SW's rule, the tile-op sequence of a composite kernel is part of the numerical contract
(`docs/design/06-software-verification.md` §7.5, which points here). ATTQK and ATTPV run on the MVM
queue, ASMAX, AMASK, AMERGE and ANORM on the VU queue (VU §5.11). The FA kernel (`k_fattn.cpp`) and
its reference `tsr_ref_fa` (§13.1) are defined as:

1. **Query tiles and head subsets.** The pass rule (AT-15) gives GS = min(G, T_MAX) and T_q =
   ⌊T_MAX/GS⌋. The kernel iterates query tiles [q0, q0 + T_q′) of T_q tokens (the last one may be
   shorter); a tile has R = T_q′·n_head rows. The head subsets hg0 = 0, GS, … (< G; the last one
   has GS′ = G − hg0 heads) run **inside each chunk** of the tile (step 4), not as outer passes:
   AMERGE and ANORM have no row selection, so for G > T_MAX (Gemma 4 global layers, G = 16) a
   pass-outermost order cannot be expressed; every op is row-independent, so each row gets the
   per-pass result (W1a decision, reconciled here at the W2b integration).
2. **Q.** QACT A32 over each of the tile's tokens' Q rows (K = n_head·D), one QA vector per token.
   The scale is not applied to Q (ASMAX and AMERGE apply it).
3. **Chunks.** The tile's chunk list comes from the mask summary (§7.3) by the chunk planner
   `sw/sim/include/tsr_fa_plan.h`, which the reference and the firmware compile (one source, so one
   chunk list; its header states the rules, which are normative):
   1. a GENERAL MSUM: one GENERAL sub-range over all n_kv cells;
   2. T_q′ = 1: each of the row's 0–2 runs is a FAST sub-range over the groups it intersects, with
      c_lo / c_hi from the run and no flags (a wrapped SWA ring: two sub-ranges; they may share a
      group, streamed twice with disjoint bounds);
   3. T_q′ > 1: one FAST sub-range when every row has one run and the runs are [max(0, lo −
      SLIDE·d_q), hi − CAUSAL·d_q) with d_q = T_q′ − 1 − q for some lo, hi, CAUSAL, SLIDE (each flag
      only when the bounds differ between rows); otherwise one GENERAL sub-range over the hull of
      the runs (it needs the f32 mask copy, §7.3, even when the MSUM itself is not GENERAL);
   4. NCG = min(128, C·⌊PCAP/(16·E/D)⌋, max(4C, roundup_4C(⌈n_total/2⌉))) groups, n_total = the
      sub-ranges' groups; a sub-range [U_lo, U_hi) is cut at U_lo + k·NCG (at least two chunks when
      there are ≥ 8C groups, §7.4);
   5. FAST with T_q′ > 1: a cut strictly inside the spread of the rows' upper (CAUSAL) or lower
      (SLIDE) bounds is replaced by the group boundaries just below and above it (at most 8
      rounds); a tile that still cannot be encoded is refused (the record is BAD_RECORD; data-
      dependent, so `supports_op` cannot see it: tiny windows next to a causal edge);
   6. per chunk: c_lo = 0 and no SLIDE when the lower bound is at or below the chunk's first cell,
      else the bound and SLIDE; c_hi = 16·n_groups and no CAUSAL when every row's upper bound is at
      or past its end, else the bound and CAUSAL; verified against every row's run;
   7. order: by (u0, c_lo); then every chunk holding a cell written in this launch moves to the
      end, keeping its order ("newest last"); the record's FA_NEW flags word (§12) names those
      cells (the layer's SET_ROWS_KV indices);
   8. a tile with no visible cell: one chunk, group 0, c_lo = c_hi = 0 (output 0, as the CPU).
4. **Per chunk k** (every op covers the tile's R rows):
   - for each head subset: FAST `ATTQK`(MAXOUT, c_lo, c_hi, CAUSAL/SLIDE) → S, MK; GENERAL
     `ATTQK`(c_lo = 0, c_hi = 16·n_groups, no MAXOUT) → S;
   - GENERAL: `AMASK`(f32 mask copy) → S, MK;
   - `ASMAX`(in place, MK, scale) → p;
   - for each head subset: `ATTPV`(LSUM) → parts;
   - `AMERGE`(INIT if k = 0).
5. **End.** `ANORM` writes the tile's output rows; then ST.

`scale` is ggml's op_params[0] f32, unchanged. This sequence, the planner's rules and the pass rule
are the numerical contract; changing any of them changes logits and requires a firmware ABI bump and
new klcheck baselines (SW §7.5). The issue order across chunks is not part of it: the firmware runs
ATTPV and AMERGE one chunk behind ATTQK / ASMAX (§7.1), which changes no op's operands.

### 6.7 Agreement with the CPU (requirement: KL level)

Sources of difference, each at or below the level the existing MUL_MAT already has:

| # | Source | Size |
|---|---|---|
| 1 | Q and the KV cache are quantized by the ref rounding, the CPU by x86 AVX | rare ties (≈ 1 per 3–4·10^5) |
| 2 | Score accumulation order (tree vs AVX) | f32 rounding |
| 3 | The max is taken over raw scores per chunk, then scaled inside EXP; the CPU scales first and keeps a running max | f32 rounding; normalization cancels the offset |
| 4 | `tsr_m_exp` vs libm `expf` | VU's accuracy target (§11) |
| 5 | P·V block-float with 24-bit weights and f32 per 64 cells, vs sequential f32 FMA | ≤ f32 level |
| 6 | Row sums block-float vs sequential f32 | ≤ f32 level |
| 7 | Normalization: exact reciprocal then multiply | identical form to the CPU's |

No 8-bit quantization of P and no re-quantization of V occur. Those were rejected because they add
errors of order 10^-3 to 10^-2 relative (AT-6).

---

## 7. Data flow (FA kernel on the RISC-V)

### 7.1 Decode, one layer

Sketch in the SDK style of `docs/ctrl-riscv.md` §8:

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
- **MVM:** ATTQK0, ATTQK1, ATTPV0, ATTQK2, ATTPV1, …, ATTPV(NC−1).
- **VU:** ASMAX0, ASMAX1, AMERGE0, ASMAX2, AMERGE1, …
- **DMA:** KVST(K), KVST(V).

Buffer reuse is ordered by queue order: ATTQK k writes S[k mod 3], which ATTPV k−3 read earlier
in the MVM queue; P[2] likewise. MK needs four buffers (W2b kvfa; the earlier sketch had MK[3]):
with three, ATTQK k (MVM, MAXOUT) writes the buffer AMERGE k−3 (VU) reads, and nothing orders that
MVM op after that VU op. With four, ATTQK k writes MK[k mod 4], last read by AMERGE k−4; ATTQK k
follows ATTPV k−2 on the MVM queue, ATTPV k−2 depends on ASMAX k−2, and ASMAX k−2 follows AMERGE k−4
on the VU queue (AMERGE runs one chunk behind), so the reuse is ordered. Every op has at most two
dependencies.

### 7.2 Prefill

- Query tiles of T_q = ⌊T_MAX/GS⌋ tokens: Qwen2.5-32B T_q = 1, Qwen2.5-3B T_q = 1, Gemma 4 SWA
  (G = 2) T_q = 4.
- One QACT per token builds the tile's QA vectors.
- KVST with n_rows = the ubatch run of consecutive cells, after QACT with T ≤ T_MAX per QACT.
- For causal single-sequence tiles: FAST with CAUSAL. For non-wrapped SWA: CAUSAL + SLIDE.
- K and V are re-streamed once per (tile, head subset). Prefill attention bytes are therefore
  (ubatch/T_q)·(cells) per layer: about 0.13 s per 512-token ubatch for 32B at n_past = 0, against
  about 18 s of weight streaming for the same ubatch at T_MAX 8 (*estimate*).

### 7.3 Masks: the mask summary (MSUM)

**Producer.** The backend derives an MSUM from each F16 KQ mask whenever the scheduler writes the
mask (`set_tensor`), and stores it next to the mask in device memory. Then the cached op list
stays valid while the mask changes per token. MSUM layout (little-endian words):

```
hdr : 'MSUM', n_q, n_kv, flags (bit0 GENERAL), f32_mask handle (the reserved copy; valid iff GENERAL), 3 × 0
row q: lo0, hi0, lo1, hi1     — at most two runs of cells with mask == +0/−0; every other cell is −inf
```

**GENERAL** is set when some row has a value other than ±0 / −inf (finite nonzero, +inf or NaN; the
reference `tsr_msum_build_f16` treats all three alike) or more than two runs, and by the producer
when the planner would refuse one of a reader's tiles (§6.6 rules 5 / 6, data-dependent: the
promotion keeps the rows' runs and makes every tile plan under rule 1; W3b, SW SW-44). The backend then
also writes an f32 copy of the mask, with row stride ⌈n_kv/16⌉ lines. It writes the copy too when a
multi-row query tile of a FLASH_ATTN that reads the mask takes the GENERAL path although the MSUM is
not GENERAL (§6.6 rule 3: rows with two runs, holes, a wrapped SWA ring in prefill; W2b kvfa); without
it that record is BAD_RECORD (`tsr_ref_fa`: TSR_FA_E_MASK). The producer can tell by running the
planner (`tsr_fa_plan`) on the MSUM's tiles. AMASK reads that copy. Its
values are fdiv(m, scale) with the reading FA op's scale (§5.5), so a mask read by FA ops with
different scales needs one copy per scale. As built (W3b, SW SW-48): every FLASH_ATTN registers
as a reader of its mask, the copy of each distinct scale has a fixed handle, and FLASH_ATTN word 13
names it whether or not it was written, so a cached op list stays valid while the mask changes.

**Kernel use, per pass:**
- every row has one run, and the runs fit lo_q, hi_q of §5.2 for some c_lo, c_hi, CAUSAL, SLIDE
  → FAST over the groups that intersect the runs;
- decode with two runs (wrapped SWA ring) → two FAST sub-ranges, each its own chunk;
- otherwise → GENERAL over the union of the runs (or all n_kv cells if GENERAL is set).

Groups with no visible cell for any pass row are never streamed. The 256-cell padding of n_kv and
empty tail cells cost no bandwidth.

**M9** (resident decode) computes the runs from positions: [0, pos] for full attention, and the
window for SWA.

A row with no visible cell at all gives O = 0, l = 0 and output 0, like the CPU. llama.cpp never
builds such a row for the supported models.

### 7.4 Overlap (D3) and SPM budget

Attention of layer l cannot overlap the next layer's weight MVMs: wo needs its output. Inside the
layer:
- the K/V projections and the K/V quantization and KVST overlap ATTQK of all but the last chunk;
- ASMAX of chunk k overlaps ATTPV k−1 and ATTQK k+1;
- AMERGE k−1 overlaps ATTQK k+1 and ATTPV k.

The exposed parts are the first ATTQK's fill, ASMAX of the last chunk while ATTPV of the previous
chunk still runs, and the tail: the last AMERGE, ANORM, then QACT for wo.

As built (W2b, OPL executor): FLASH_ATTN loads Q from DRAM (LD Q, then QACT), and that LD follows the
layer's SET_ROWS_KV KVSTs on the DMA queue, so no chunk overlaps the KVSTs (about 100 core cycles per
decode KVST). The newest-last order stays, because it is part of the numerics (§6.6 rule 7). Measured
(SW §7.8, `fw_kvfa_perf`): the RISC-V's work for a Qwen2.5-32B decode FLASH_ATTN at 4,096 cells is
20.3 µs against 140.0 µs on the units, the MVM never waits for a push, and the span equals a near-ideal
issuer's; inside a whole 0.5B decode layer at n_kv 256 the record's 16 µs of RISC-V work does reach the
critical path (between V and wo) once the 8-entry DMA queue has held the RISC-V back (SW §7.8).

SPM traffic per chunk of n cells and R rows (*estimate*):

| Op | Read lines | Write lines |
|---|---|---|
| ATTQK | QA (T_q·QA_LINES) | R·n/16 |
| ASMAX | R·n/16 | R·n/16 |
| ATTPV | R·n/16 (p to the clusters) | C·R·D/16 (+ l) |
| AMERGE | (C+1)·R·D/16 + small | R·D/16 + small |

For 32B (R = 40, 2,048 cells per chunk) this averages about 0.6 lines per cycle on each SPM port
over the chunk's 17.8 K MVM cycles. The ports carry it with the second read path and round robin
(`docs/design/04-control-core.md` §10).

Scratch per layer for 32B: 3 × 5,120 (S) + 2 × 1,292 (P) + 330 (state) + 90 (QA) ≈ 18.4 K lines
(1.15 MiB) of the ≈ 3.7 MiB scratch ring (SW §7.3).

### 7.5 Gemma 4 specifics

Values for the 12B are read from its GGUF (§4.5); the 31B values are *confirm*.

- **QK-norm.** Q and K RMSNorm per head with weight, V RMSNorm without weight. These are VU work
  before QACT/KVST (SW §7.5 RMS_NORM, per-head form; VU §8.1).
- **Scale.** 1.0: ASMAX scale = 1.0.
- **SWA layers.** 12B: D = 256, n_head_kv = 8, G = 2 (31B: G = 2, *confirm*). A separate cache of
  1,536 cells (n_swa 1024): planes LG = 1,536/(16·C) = 24 on the U200. The MSUM runs carry the
  window; after a wrap a decode row has two runs (two FAST sub-ranges).
- **Full layers.** 12B: D = 512, n_head_kv = 1, G = 16 > T_MAX: GS = 8, **two passes per decode
  token** (hg0 = 0, 8), so the full-layer K and V planes are streamed twice per token. 31B: D =
  512, n_head_kv = 4, G = 8, one pass (*confirm*). E = 512 on the 12B, so KVST read-modify-writes
  every K meta beat (§4.5).
- **K = V source** on full layers without `wv`: two SET_ROWS from different tensors, nothing special.
- **Embedding scale** (a MUL by `inp_scale_rows`) and the final-logit softcap are not AT.

---

## 8. Hardware structure (guidance; the contract is §5–§6)

### 8.1 `ATTQK` in the cluster

- **Front end.** The stream requester streams the channel's groups, which are contiguous beats.
  The chunk parser counts blocks with a row boundary every E/32 blocks and a segment boundary every
  D/32 blocks. It is the Q8_0 path unchanged except for the counters.
- **Activations.** The broadcast copies the T_q QA vectors. Query q's vector is written into the
  GS slot columns of that query (write-enable replication, no extra cycles). Slot t reads head
  h(t, g) of its vector: the per-token activation SRAMs (`docs/rtl-m3.md` §5) get per-slot read
  addresses, base(t, g) = (g·G + hg0 + i)·D, advanced by G·D at each segment boundary.
- **MAC lanes and terms.** Unchanged (Q8_0: 64 lanes, 2 terms per beat per slot). DSP packing of
  two slots per DSP works as in the MVM, because slots share the K value.
- **Segment trees.** D/32 ∈ {2, 4, 8, 16} terms per (segment, slot). The two accumulate adders per
  slot form the first tree level, and the tree adders finish the rest, then z(). Required rate:
  at most 2 additions per cycle per slot.
- **Gather.** Per cluster, a line buffer per pass row (at most 128 rows × 64 B = 8 KiB; 2 RAMB36).
  A row's line is written once its 16 cells are complete: one full-line SPM write, no byte enables,
  no ACC. Cells outside [lo_q, hi_q) are replaced by −inf at the gather. A max tracker per row
  (LUTRAM) reads, compares and writes per line. MAXOUT writes ⌈rows/16⌉ lines at the end.
- **Throughput.** One beat per cycle per channel while the gather keeps up. The gather needs
  64·C·GS·T_q/(17·D) lines per cycle (15.06·GS·T_q/D at C = 4). Above 1 the op is gather-bound;
  only D = 64 is affected (§9.1).

### 8.2 `ATTPV` in the cluster

```
V beats ──► delay FIFO (64 beats) ─────────────────────────────► MAC (DSP accumulate) ─► capture ─► convert ─► O
   │ meta (32 × fp16)                                               ▲   W(slot, e)      (512×38b)  (8 pipes:   (f32,
   └──► w-unit: 16 fmul/cycle  w = p·d  ─► E_S max ─► align (RNE) ──┘                            i→f32, ×2^k,  LUTRAM)
              p from slot columns (p broadcast, scatter per channel)                               fadd)
```

- **p broadcast.** Line (row r, group u) goes only to cluster u mod C, into slot column t(r) at
  position (g(r), local cell). The SPM read rate equals the gather rate formula above.
- **DSP use.** 64 lanes × 8 slots = 512 DSPs, unpacked: A = W (25 bits), B = the V byte.
  OPMODE accumulates in P (48 bits) over a segment and restarts at the next. Slots 4–7 use lanes
  64–127, which receive copies of bytes 0–63. tsr_mac_lane gains a PV mode (Xilinx: dynamic OPMODE
  and an A/D input mux; generic: behavioural). tb/tech checks the two variants against each other.
- **Segment pipeline.** The exponent E_S needs all w of a segment before its first product, so
  quant beats wait in a 64-beat FIFO while the w-unit computes the segment's 128·GS·T_q weights
  (16 per cycle). At segment end the 512 accumulators are captured, and 8 conversion-and-add pipes
  drain them in at most 64 cycles, under the next segment.
- **Output.** At the end of a slice, per slot, 4 lines go to SPM through the gather.

### 8.3 `KVST` in the DMA

The DMA reads QS and D lines from SPM, converts 32 scales per meta beat, writes full quant beats,
and read-modify-writes partial meta beats (§5.1). One token's K or V goes to one channel.

### 8.4 Cost (*estimate*, U200, per cluster unless noted)

| Item | LUT | FF | DSP | RAMB36 |
|---|---|---|---|---|
| ATTQK: counters, per-slot addresses, segment-tree control, gather line buffers, max trackers | 3–4 K | 3 K | 0 | 2 |
| ATTPV: DSP input muxes and mode control | 8–12 K | 1 K | 0 | 0 |
| ATTPV: w-unit (16 fmul), exponent max, 16 aligners | 3–4 K | 3 K | 16–32 | 0 |
| ATTPV: delay FIFO, capture bank, 8 convert/add pipes, O/l storage | 5–7 K | 22 K | 0 | 2–4 |
| **Per cluster** | **19–27 K** | **29 K** | **16–32** | **4–6** |
| **Four clusters** (SLR0/SLR2: about +40–55 K LUT each, from ≈ 40 % with the DDR controller, `docs/platform-u200.md` §8.2, to ≈ 50–54 %) | 76–108 K | 116 K | 64–128 | 16–24 |
| KVST in the DMA (SLR1) | 1–2 K | 1 K | 0 | 0 |
| AT vector ops in the VU (EXP reuse; AMERGE / ANORM sequencing, row-scalar buffers; VU §7.10) | ≈ 2 K | 1.5 K | 0 | 2 |
| SLR1 share (KVST + central-gather ATT path; CC §14.1) | 2–5 K | | 0 | 0 |

SLR0 and SLR2 use 1,348 of 2,280 DSPs each (59 %, `docs/rtl-m3.md` §14.5), so about 930
are free in each. The DSP and RAM budgets are not the issue; LUT and timing are (§15).

---

## 9. Throughput (*estimates*)

### 9.1 Op rates

| Model | D | G (decode GS) | ATTQK / ATTPV efficiency in decode (gather / p rate) | Prefill (GS·T_q = 8) |
|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 7 | 61 % (1.65 lines/cycle needed) | 53 % |
| Qwen2.5-1.5B / 3B / 7B | 128 | 6 / 8 / 7 | 100 % | 100 % |
| Qwen2.5-14B / 32B | 128 | 5 | 100 % | 100 % |
| Gemma 4 12B SWA / full | 256 / 512 | 2 / 16 (decode GS 2 / 8; full layers 2 passes) | 100 % | 100 % |
| Gemma 4 31B SWA / full *confirm* | 256 / 512 | 2 / 8 | 100 % | 100 % |

Fill per ATT op is about 300–400 cycles: dispatch, broadcast, pipeline, the 64-beat segment
delay, and the gather drain.

### 9.2 Qwen2.5-32B, 4K context, one decode layer (276.3 MHz, 0.975 beats/cycle/channel)

| Part | Cycles |
|---|---|
| K stream (ATTQK, 2 chunks): 4,096 cells / 4 channels × 17 beats | 17,854 |
| V stream (ATTPV, 2 chunks) | 17,854 |
| ATT op fills (4 × ≈ 350) | ≈ 1,400 |
| exposed head: QACT of Q → first ATTQK (K/V prep and KVST hidden) | ≈ 300 |
| tail: last AMERGE (≈ 1,600: 5 lines read per output line, 320 lines), ANORM (320), QACT for wo | ≈ 2,500 |
| **Total** | **≈ 39,900 (144 µs); DDR busy ≈ 89 % of it** |

**Per decode token:** 64 × 144 µs ≈ 9.2 ms, against 8.27 ms of pure KV streaming (570 MB at
69 GB/s).

### 9.3 Share of the decode token time (device time; weights at the measured 69 GB/s)

| Model | Weights per token | Attention at 4K | Share | Attention at 32K | Share |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | ≈ 8.25 ms (`docs/platform-u200.md` §14.1) | ≈ 0.8 ms | ≈ 9 % | ≈ 6.0 ms | ≈ 42 % |
| Qwen2.5-32B Q4_K_M | ≈ 300 ms | ≈ 9.2 ms | ≈ 3.0 % | ≈ 70 ms | ≈ 19 % |
| Gemma 4 12B Q4_K_M (7.1 GB file) | ≈ 100 ms | ≈ 3.6 ms (SWA ≤ 1,024 visible cells; full layers streamed twice) | ≈ 3.5 % | ≈ 11 ms | ≈ 10 % |
| Gemma 4 31B Q4_K_M *confirm* | ≈ 275 ms | ≈ 9.7 ms | ≈ 3.4 % | ≈ 28 ms | ≈ 9 % |

The RISC-V issues 4 + 4·NC attention ops per layer: 2 KVST, 1 QACT and 1 ANORM, plus ATTQK,
ASMAX, ATTPV and AMERGE per chunk. For 32B at 4K that is 12 per layer and ≈ 0.8 K per token:
≈ 0.2 ms of RISC-V time at 166.7 MHz, overlapped (`docs/ctrl-riscv.md` §4.8). This replaces the
survey's ≈ 18 K per-head ops.

---

## 10. PERF counters (slots X + 16 … X + 19, `01-execution-model.md` §6)

| Slot | Name | Kind | Meaning |
|---|---|---|---|
| X + 16 | `ATT_QK_CYCLES` | time | the MVM unit holds an ATTQK (a subset of `MVM_CYCLES`) |
| X + 17 | `ATT_PV_CYCLES` | time | the MVM unit holds an ATTPV |
| X + 18 | `ATT_PATH_STALL` | time | some cluster in an ATT op holds its stream because the gather (ATTQK) or the p broadcast (ATTPV) is behind: the gather- or broadcast-bound share |
| X + 19 | `KVST_RMW` | event | meta beats read-modify-written by KVST |

VU's slots X + 13 … X + 15 come first; at `NUM_CH` = 16 (X = 37) AT's end at 56, below DB's 61–63.

K and V read beats are already counted in `CH_RD_BEATS[c]`. For the spec's F metric, useful beats
include the planes' beats (the simulator's cycle model reports both).

---

## 11. Decisions

| Id | Topic | Choice | Alternatives | Rationale |
|---|---|---|---|---|
| AT-1 | Where attention computes | Q·Kᵀ and P·V in the MVM clusters; the VU only does softmax passes and merges | VU dot products with K/V tiles LDed to SPM; a separate attention engine with its own DSPs | One SPM port is 64 B/cycle against 256 B/cycle of DDR, so a VU or SPM-tile design is capped at 25 % of the rate the priority demands. A separate engine needs up to 1,024 more DSPs per SLR (two channels at G = 8) against ≈ 930 free, and it duplicates the stream path. The clusters are idle during attention and already stream one beat per cycle per channel. |
| AT-2 | KV values | Exactly llama.cpp's q8_0 (blocks along each head row, fp16 scale, ref rounding); only bytes move | V transposed with blocks along tokens; int8 P with V^T in a standard MVM | Any other quantization axis makes the device KV differ from the CPU's in every element. The KL criterion is relative to a 1-ulp floor, so that error would likely fail it. Keeping the bytes also makes state save/load lossless. |
| AT-3 | Channel distribution | 16-cell groups round-robin over channels (u mod C) | contiguous cell ranges per channel; per-cell interleave | Contiguous ranges leave channels idle at short contexts. Per-cell interleave splits score lines across clusters, needing partial-line gather writes. A 16-cell group is one score line per row, so every gather write is a full line. |
| AT-4 | Plane layouts | K token-major in Q8_0 chunks; V slice-major (all cells' 64-value slice contiguous) | both token-major; both slice-major | Token-major K gives whole scores per row in one pass at ≤ 1 gather line per cycle for D ≥ 128. Slice-major K would need D/64 passes and double the gather rate. Slice-major V lets each DSP keep one accumulator for a whole slice. Token-major V would need D/64 accumulators per lane in fabric (≈ 26 K LUT per cluster more). Both layouts depend only on E and kv_size, so they are fixed at allocation. |
| AT-5 | KV store | KVST (DMA) from QACT A32 output, consecutive cells; partial meta beats by read-modify-write in the DMA | a fused quantize-and-store op; write strobes on the memory port | QACT already implements `quantize_row_q8_0_ref` bit-exactly. RMW keeps the core boundary and the platform unchanged; EX added no write strobes. Its cost is one extra beat read per partial meta beat (≈ 1–3 µs per layer in decode, off the critical path except for the last chunk). |
| AT-6 | P·V numerics | Block floating point per 64-cell channel segment: 24-bit integer weights, exact integer sums, one f32 add per segment; P stays f32-precise | int8 P (A32 quantization); 16-bit P; f32 FMAs at full rate | Int8 P costs about 1 % relative error per output (block max/254 over 32 cells). 16-bit P costs about 10^-5. Both are far above the CPU's f32 path and risk the KL criterion. f32 FMAs at 2,048 per cycle cost about 500 K LUT. The block-float form reuses the 512 MAC DSPs and is at least as precise as the CPU. |
| AT-7 | Segment length | 4 local groups = 64 cells per channel | 16 cells; whole op | 16-cell segments need 30 f32 conversions and additions per cycle per cluster; 64-cell segments need 7.5. A whole-op exponent would need a pre-pass over the V scales or lose precision on outlier tokens. |
| AT-8 | Softmax structure | Two-pass within a chunk (max from ATTQK, EXP by ASMAX), online merge across chunks (AMERGE) | fully online per cell like the CPU; one global two-pass over all cells | Per-cell online rescaling of the accumulators cannot be done in the DSP accumulation. One global pass needs all scores in SPM (32K × 40 rows = 5 MiB for 32B) and prevents overlap. Chunks of ≤ 128 groups fit the slot buffers and pipeline across units. |
| AT-9 | Where the scale goes | Inside ASMAX and AMERGE: EXP((s − m)·scale); ATTQK returns raw scores | scale Q before QACT; scale the QA block scales | Scaling Q before QACT changes the fp16 rounding of every Q block scale (≈ 2^-11 relative in each term). Scaling the QA D area costs one VU op per token per layer and an fp rounding per term. Inside EXP it is one rounding, as on the CPU, and it is free in an op that already reads every score. |
| AT-10 | Row sums | Computed by ATTPV (LSUM) with the same block-float rule | a VU RED.SUM pass; fused into ASMAX | LSUM saves a full read pass over R·n scores per chunk, which would add 0.29 lines/cycle on the SPM read port. It keeps l consistent with O (same p values and segments). The cost is one accumulator per slot. |
| AT-11 | Channel partials and NUM_CH dependence | ATTPV outputs C partials; AMERGE adds them in a fixed tree; results depend on NUM_CH | ordered ACC in the gather; making results independent of C | The gather's ACC in channel order would serialize or buffer the clusters. Independence from C would need segments spanning channels, which the clusters cannot see. Bit-exactness is defined per configuration (the simulator reads CFG), as with every other op. |
| AT-12 | Row max floor | M_FLOOR = −2^100 in ATTQK and AMASK | −inf; special-case logic in merges | With −inf, rows with no visible cell in a chunk produce NaN (EXP(−inf − (−inf))). A finite floor keeps every merge finite with no extra op, and changes nothing for real scores. |
| AT-13 | Empty rows | ANORM uses RECIP(max(l, 2^-126)) | require non-empty rows; special case in the kernel | It reproduces the CPU's `S == 0 ? 0 : 1/S` exactly at no cost. |
| AT-14 | Masks | Host-built mask summary (≤ 2 runs per row) with CAUSAL/SLIDE bounds in ATTQK; f32 copy and AMASK only for GENERAL | the device reads f16 masks (VU has no f16); per-token op-list patching | The VU has no f16 (survey). The summary keeps SW's cached op list valid per token. Bounds in ATTQK skip all masked bandwidth for the common cases and make the max exact. |
| AT-15 | GQA and prefill mapping | Slots = (query token, head-in-group); head subsets (hg0, GS) for G > T_MAX; T_q = ⌊T_MAX/GS⌋ | one KV head per op with heads as T; K re-streamed per head | All KV heads in one op stream K once per pass. Head subsets make every G work, including MQA and T_MAX 1 bitstreams (k1 runs attention as G passes for bring-up). |
| AT-16 | Units and opcodes | KVST 0x14 (DMA), ATTQK 0x21, ATTPV 0x22 (MVM), ASMAX 0x34, AMASK 0x35, AMERGE 0x36, ANORM 0x37 (VU); unit 4 absent; no new error codes (OPERAND covers shapes) | a separate ATT unit 4 queue sharing the clusters; quantize-class KV ops | The ops use existing datapaths, so a separate queue would only add arbitration for one resource. The numbers fall inside EX's AT blocks. |
| AT-17 | Hadamard KV rotation | Off (adopt SW-16); FWHT semantics recorded (Appendix A) | offload the rotation as a VU FWHT, bit-identical to the CPU | SW already fixed the baselines with rotation off. Rotation matters most for ≤ 5-bit caches. Keeping it on needs a new VU function in M6. It can return in M8 if KL or quality data asks for it. |
| AT-18 | Unsupported features | No softcap, ALiBi, sinks, sparse n_kv_max, MLA (DK ≠ DV), D outside 64–512, K-shift | implement softcap via VU TANH | No target model uses them (Qwen2.5, Gemma 4 dense). Softcap needs max handling in the capped domain. Each can be added later without changing the layout. |
| AT-19 | Chunk policy | NCG = min(128, PCAP limit, ≥ 2 chunks for ≥ 8C groups), segment-aligned; newest cells last | fixed chunk size; per-model tuning | It overlaps VU and MVM work for every context of 8C groups or more. The newest-last order lets KVST overlap earlier chunks. It is deterministic, which makes it part of the numeric contract. |
| AT-20 | DQROW | DQROW per VU §5.6 (SPM → f32 on the VU, RTL in M8); the M8 embedding lookup is LD (ROW_BEATS) + DQROW; an f32 `tok_embd` copy remains the exact alternative for other GGUF types; attention never uses it | the spec's DRAM-row → f32 on the DMA unit with a dequant stage (the draft); drop | VU owns the encoding (EX §7.2) and already has the fp datapath; the embedding lookup is on the CPU until M8. The f32 table is decided in M8 when `tok_embd` is not Q8_0 or Q4_K. |
| AT-21 | KV region | Symmetric planes, top-down from each channel's end; one plane per (layer, stream, K/V); LG from kv_size | interleave KV in the weight heap; ch0-only | Symmetric planes keep every channel's share equal. Top-down keeps the KV growth away from weight uploads. |
| AT-22 | FA head subsets and chunk planner (W1a / W2b; reconciled at the W2b integration) | The head subsets of a query tile run inside each chunk (all ATTQK of the chunk per subset, AMASK, one ASMAX, all ATTPV per subset, one AMERGE over all R rows); the chunk planner is one source, `sw/sim/include/tsr_fa_plan.h`, compiled by `tsr_ref_fa` and the firmware; its rules 1–8 (§6.6) are normative; FA_NEW in the flags word names the newest cells; MK[4] | passes outermost (the original §6.6); separate planners in the reference and the firmware; MK[3] | AMERGE / ANORM have no row selection, so G > T_MAX (Gemma 4 global, G = 16) cannot run as outer passes; one planner gives one chunk list, hence bit-identical outputs, by construction (`fw_kvfa_plan` checks it against W1a's code); MK[3] races ATTQK k against AMERGE k−3 |

---

## 12. Interfaces with other areas

| Area | AT provides | AT assumes |
|---|---|---|
| **EX** | Encodings and checks of 0x14, 0x21, 0x22, 0x34–0x37 within EX's blocks; unit mapping (DMA, MVM, VU; unit 4 absent); `CAPS` bit5 and `CFG3[31:16]` = PCAP/16; PERF X + 16 … X + 19; KV placement in the heap (§4.5) | EX's w0 format with x16 usable by AT; check order and `OPERAND` (13); no run-time errors (all AT checks are at dispatch); X+2 `ATT_CYCLES` reads 0 while unit 4 is absent; 0x15–0x17, 0x19–0x1B, 0x23 stay reserved for AT |
| **VU** | The semantics of ASMAX, AMASK, AMERGE and ANORM, which execute on the VU (unit 3; VU §5.11, §7.4); their use profile (§7, §9); Appendix A (FWHT) for a later function | Confirmed by VU: EXP = EXPK(·, P1) with EXP(−inf) = +0, EXP(±0) = 1 exactly, +0 below its underflow threshold, canonical NaN for NaN, ≤ 0.53 ulp (*model*, exhaustive measurement pending); RECIP = `tsr_frecip`; the multi-pass AMERGE at ≈ 4–5 cycles per output line; per-head RMSNorm for Gemma QK-norm (VU §8.1); VOP for the K/V bias and ROPE before QACT |
| **CC** | Ops per layer (4 + 4·NC) and queue use: MVM ≤ 3 AT entries in flight, VU ≤ 4, DMA 2 per layer; checker classes with up to four tagged extents (AMASK and AMERGE use four SPM extents) | Queue depths MVM 4, VU 16, DMA 8 (`04-control-core.md` §7.4); two dependencies per op; SPM round robin with the MVM client carrying ATT broadcast and gather traffic; client 5 (AT) tied off; the four-extent checker slot (CC §8.2) |
| **DB** | ATT ops are ordinary queue entries with no state across ops (every intermediate is in SPM: scores, p, partials, running state), so tile stops and QSTEP need nothing AT-specific; PERF counters (§10) | Completion records carry w0, so profiles separate ATTQK and ATTPV; the watchdog covers the MVM unit (the longest ATT op is about 0.1 ms) |
| **SW** | Plane layout and address functions (§4); KV tensor identification and lossless `get/set_tensor` (§4.6); the `supports_op` table (§2.6); MSUM format (§7.3); the normative FA algorithm (§6.6); C references (§13.1); AT trace generator requirements | Adopted in SW §7.2, §7.5, §9.3: the backend allocates planes from kv_size and E at `init_tensor`, zeroes them on clear, and builds MSUM (and the f32 mask copy when GENERAL) in `set_tensor` of tensors named `kq_mask`; the MSUM is a preload entry. OPL **SET_ROWS_KV** args 8–11 = {plane base, E/64 \| LG<<16, plane kind, stream}. OPL **FLASH_ATTN** args 12–13 = {MSUM (preload byte offset), f32-mask handle} and args 16–21 = {K base, V base, E/64 \| LG<<16, D, G, flags}; softcap must be 0. The flags word (W2b): bit0 FA_NEW, bits [7:1] 0, bits [15:8] the preload line of the I64 new-cell indices (byte offset / 64), bits [31:16] their count (0 without FA_NEW); the backend points it at the layer's SET_ROWS_KV index entry, whose cells §6.6 rule 7 moves last (AMERGE order is part of the numerics: without the word the chunk order, hence the output bits, differ from the reference run with those new cells). The `k_fattn` and `k_kv` kernels implement §6.6 and §7. klcheck and numcheck run with `-fa on -ctk q8_0 -ctv q8_0` and `LLAMA_ATTN_ROT_DISABLE=1` on both CPU and device |

---

## 13. Verification

### 13.1 References (`sw/sim/tsr_ref_at.c`, `sw/sim/include/tsr_ref_at.h`, in `libtsr_ref`)

| Function | What it defines |
|---|---|
| `tsr_kv_addr_k`, `tsr_kv_addr_v`, `tsr_kv_pack_row`, `tsr_kv_unpack_row` | §4.3–4.4 |
| `tsr_ref_kvst` | KVST |
| `tsr_ref_attqk` | per-segment SEG via `tsr_dot_row`, bounds, M_FLOOR max |
| `tsr_ref_attpv` | block float with `__int128` sums, F(), NaN rule, LSUM |
| `tsr_ref_asmax`, `tsr_ref_amask`, `tsr_ref_amerge`, `tsr_ref_anorm` | the vector ops |
| `tsr_ref_fa` | the composite of §6.6, taking MSUM and the chunk policy |

The simulator's units call these functions, and the host C++ reference in `sw/ctrl/ref/` composes
them in kernel order (SW §7.5).

### 13.2 Reference self-tests (`sw/sim/tests/test_ref_at.c`: ctests `ref_at`, `ref_at_ggml`)

- **Layout.** ggml rows → planes → rows round trip for random E (64–4096, multiples of 64),
  kv_size and C ∈ {1, 2, 3, 4, 16}. Every byte is placed once.
- **KVST** equals `quantize_row_q8_0_ref` + layout, byte for byte.
- **ATTQK SEG** equals `tsr_dot_row` on the segment repacked as a standalone Q8_0 row. It also
  equals the short tree z(pairwise) (§6.2), over random and directed terms (−0, NaN, Inf, FTZ
  boundary values of `docs/spec.md` §21, FTZ decision point).
- **ATTPV:**
  - A_L is checked against an independent 128-bit model;
  - c_L against an exact rational model rounded once (F);
  - the cases p = 0 skip, all-zero segments, non-finite w per block, E_S extremes (w near 2^-126
    and 2^15), q = −128, partial last segments, n_groups not a multiple of C;
  - a double-precision bound: |O − exact| ≤ (segments + 1)·2^-24·Σ|p·v|.
- **Composite** `tsr_ref_fa` against ggml's CPU FA on the same q8_0 cache: NMSE ≤ 5·10^-4, the
  `test-backend-ops` tolerance, over shapes D ∈ {64, 128, 256, 512}, G ∈ {1, 2, 5, 7, 8, 16},
  n_kv ∈ {256, 1,024, 4,096, 16,384}, T_q 1–8, masks (causal, SWA, wrapped SWA, holes, GENERAL
  with finite values); and against a double-precision softmax.

### 13.3 Independent spec model

`sw/tools/numcheck.py` gains a numpy model of the seven ops and the composite. It is written from
this document only, does not share code with `tsr_ref`, and must agree bit for bit
(`docs/spec.md` Verification strategy, numcheck B1b).

### 13.4 Simulator and traces

- `tsr_sim` executes the ops, with a cycle model:
  - ATTQK and ATTPV: fill + max(max_c beats_c, pass rows × n_groups), plus 68 for ATTPV;
  - KVST: 40 + beats + RMW reads;
  - VU ops: lines / VU rate.
- The trace generator `--ops at` (SW `traces/at`) covers every check row of §5.8 and the
  directed numeric cases above, with C ∈ {1, 3, 4, 16} and T_MAX ∈ {1, 8}.

### 13.5 RTL

- **cocotb block tests:**
  - cluster ATTQK mode: segment trees at D = 64–512, per-slot addressing with hg0/GS/T_q, gather
    line assembly, −inf bounds with CAUSAL/SLIDE, max trackers;
  - cluster ATTPV mode: w-unit, delay FIFO, DSP accumulate and restart at segment boundaries
    under random stalls, capture and drain, NaN blocks;
  - tb/tech: Xilinx vs generic `tsr_mac_lane` in PV mode, cycle by cycle;
  - tb/dma: KVST with RMW interleaved with LD/ST of the same channel.
- **Core level.** Replay of `traces/at` and of FA op-list traces from real models, bit-exact
  against the simulator under random schedules (SW's race detector, CC's lockstep co-simulation).
- **Rate.** Back ends and gather sustain one beat per cycle at D = 128, GS·T_q = 8, and the
  predicted gather-bound rate at D = 64.

### 13.6 Backend and model level

- **`test-backend-ops`** on sim, vkernel and xrt: `FLASH_ATTN_EXT` with q8_0 K/V (the shapes
  above, n_stream 1), `SET_ROWS` into q8_0 KV tensors, and state save/load round trips. NMSE vs
  CPU, and bit-identical across sim, vkernel and xrt. *As built (W3b, SW SW-52):* test-backend-ops'
  K / V / dst tensors are never KV-cache tensors in a KV buffer, so supports_op refuses its
  `FLASH_ATTN_EXT` and `SET_ROWS` cases by design; `ggml_tesseract_attn` (SET_ROWS + FLASH_ATTN_EXT in
  llama.cpp's graph shape through KV buffers: bit-exact against `tsr_ref_fa`, NMSE vs CPU) and
  `ggml_tesseract_kvstate` (state save / load) carry these checks instead.
- **M6 gate** (SW §10.4): `offload_kqv` = false, so FA runs on the CPU. In M6 the AT ops are
  verified by the `selftest` kernel's test 5, `traces/at` and `ggml_tesseract_attn` on sim, vkernel
  and the card (not `test-backend-ops`, see above).
- **M7 gate** (D17 baselines: `-fa on -ctk q8_0 -ctv q8_0`, rotation off, KV on the device):
  - Qwen2.5-0.5B Q8_0: `--expect-identical` device vs sim;
  - Qwen2.5-32B Q4_K_M: mean KL ≤ 3 × floor and top-1 ≥ 80 %;
  - Gemma 4 12B (and 31B when a GGUF is available): the KL criterion;
  - long prompts (2K and 16K tokens) to exercise multiple chunks, prefill CAUSAL tiles and, for
    Gemma, SLIDE, the wrapped SWA ring and the two-pass full layers.
- **numcheck** dumps per-layer FA outputs on device and simulator: bit-exact against the numpy
  model; fp error bound against fp64.

### 13.7 Performance

- `ATT_*` and `CH_RD_BEATS` against the cycle model.
- On the card: decode tokens per second at 4K and 16K contexts, against §9.3.
- DDR utilization during attention: target ≥ 85 % for D ≥ 128 (*target*).

---

## 14. Changes to existing documents (applied in W0)

The integration of `docs/design` edits no file outside `docs/design`; these edits are work package W0 of
`docs/design/README.md`.

| Document | Change |
|---|---|
| `docs/spec.md` §4 | DRAM map KV row: "symmetric planes, top-down; layout `docs/design/03-attention-kv.md` §4" |
| `docs/spec.md` §20 | Open item "KV cache layout" closed by §4 and D17 |
| `docs/spec.md` §11 | ISA table: KVST 0x14, ATTQK 0x21, ATTPV 0x22, ASMAX–ANORM 0x34–0x37 (M6) |
| `docs/spec.md` §12 | Attention no longer stays on the CPU; `-nkvo` is no longer required; supports_op per §2.6 |
| `docs/spec.md` §15 | Stage-A ops: `FLASH_ATTN_EXT` only (no `SOFT_MAX`, no attention `MUL_MAT`) |
| `docs/klcheck.md` "Settings" table | FA on, KV q8_0, rotation off; `offload_kqv` = false in M6 and true (KV on the device) in M7 |
| `docs/rtl-m3.md` | MVM cluster gains ATTQK and ATTPV modes (§8); RB_MIN unchanged for weight MVMs |

---

## 15. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Timing: the ATTPV mode adds a mux before every MAC DSP and a capture bank; ATTQK changes the gather, whose job pipes are the k4 link's worst paths (WNS −0.285 ns, 276.3 MHz) | the k4 link drops below 276 MHz (D13) | DSP input registers absorb the mux; the gather ATT path writes full lines only and bypasses the partial-line assembler; OOC runs of the cluster per mode before the link; fallback: an ATTPV with half the slots active per cycle (bit-exact, half rate for G > 4) |
| LUT growth of 76–108 K over four clusters (*estimate*; SLR0/SLR2 to ≈ 50–54 %) | SLR0/SLR2 congestion | per-mode floorplan check in the first OOC; trim the capture bank to 4 drain pipes and 128-cell segments if needed (numeric contract change before freeze only) |
| KL with FA on differs from the FA-off settings baselines | a gate fails for reasons unrelated to AT | compare the CPU FA-on vs FA-off floor first; device differences are at f32 level (§6.7) |
| VU's EXP accuracy or AT vector ops not built in time | softmax wrong or missing | the requirement list in §12; until VU lands, the simulator's ops let the firmware and backend proceed; AMERGE/ANORM can be emulated by standard VOPs (slower, numerics defined by the same formulas) |
| SPM port contention during attention (≈ 0.6 lines/cycle per port) | ATT ops stall, DDR idles | three score buffers give two MVM ops of slack; `ATT_PATH_STALL` and CC's bank-conflict counter measure it; fuse EXP into the ATTPV broadcast later if needed |
| D = 64 models are gather- and broadcast-bound (61 %) | 0.5B decode at long context slower than DDR allows | accepted (test model); a wider gather (2 lines per cycle) would fix it |
| Gemma 4 31B dimensions (D 256/512, n_head_kv 16/4, n_swa 1024, 50/10 layers) are unconfirmed (the 12B is read from its GGUF) | 31B capacity and rate rows off | read a 31B GGUF when available; the ops cover D ≤ 512 and any G |
| Gemma 4 12B global layers need two head passes (G = 16 > T_MAX) and RMW every K meta beat (E = 512) | full-layer K/V streamed twice per decode token; extra KVST reads | 8 of 48 layers, ≈ 71 MB per token at 4K (§9.3); `KVST_RMW` and `ATT_*` measure it |
| llama.cpp KV operations outside FA (K-shift, defrag, n_stream > 1) | scheduler copies or aborts | SW R10: report at graph build; layout already supports streams |
| RMW meta writes race with an unordered reader | stale scale read | memory model requires the dependency (SW race detector flags it); KVST meta writes are atomic per beat |
| Prefill with G > 4 runs one query per pass | prefill attention bytes ×8 vs ideal | negligible beside weight streaming at T_MAX 8 (§7.2) |

---

## 16. Open items

- Gemma 4 31B shapes (*confirm* from a GGUF when one is available; the 12B is confirmed).
- VU's exhaustive EXP measurement (the *model* figure is ≤ 0.53 ulp, VU §13).
- Timing and area of the ATTPV mode (DSP input mux, capture bank) and of the ATTQK gather path at
  the k4 link (≥ 276 MHz); fallback: a half-rate ATTPV, bit-exact (§15).
- Whether M8 wants the Hadamard rotation back (Appendix A) and the f32 embedding table (§5.9).
- Card measurement of the DDR cost of KVST's scattered single-beat writes and RMW reads during
  ATTQK / ATTPV streams (`docs/ctrl-riscv.md` §5.4).

---

## Appendix A. FWHT (recorded for a later VU function; not in M6/M7)

`tsr_ref_fwht(x[n])`, n ∈ {64, 128, 256, 512}, bit-identical to the CPU
(`ggml/src/ggml-cpu/ops.cpp:12452-12505`) for normal inputs:

```
sc = tsr_frecip(tsr_fsqrt((float)n));            // = 1.0f/sqrtf(n)
for i < n: y[i] = tsr_fmul(x[i], sc)
for (len = 1; len < n; len <<= 1)
  for (i = 0; i < n; i += 2·len) for (j = 0; j < len; ++j)
    { u = y[i+j]; v = y[i+len+j]; y[i+j] = tsr_fadd(u, v); y[i+len+j] = tsr_fsub(u, v); }
```

The CPU's SIMD stages use FMA(u, v, −1), which is exactly u − v. Stage order and pairing are the
same, so the result matches the CPU except for subnormals (FTZ).

# VU: vector unit and approximate math

Area VU specifies the vector unit: its ISA, its bit-exact numerics and its microarchitecture. It
applies the global tables of EX ([01-execution-model.md](01-execution-model.md)), runs on the SPM of CC
([04-control-core.md](04-control-core.md)) and executes the softmax ops of AT
([03-attention-kv.md](03-attention-kv.md)). Numbers marked estimate are design figures. Numbers marked
model come from a numpy and mpmath model of the algorithms in §6; the exhaustive tests of §10 check
the reference implementation against them.

---

## 1. Summary

- One operand model for every VU op: an op works on `rows` rows of `m` 32-bit elements, each row starts on an
  SPM line, each operand has its own row stride in lines (stride 0 repeats one vector over all rows), and a
  per-row scalar operand `c` comes from an s register or from an SPM vector. One op covers all heads, all
  tokens, or both.
- Ops (unit 3): `VOP` 0x30 (25 element functions, including the fp16 forms GELU16 and GEGLU16 of the ggml CPU),
  `RED` 0x31 (segmented SUM, SUMSQ and MAX with the fused row post-ops RCP and RMS), `ROPE` 0x32 (angles by
  ggml-cpu's own recurrence), `SOP` 0x40 (15 scalar functions), `GDN` 0x39 (`HAS_GDN`, `CAPS` bit 8, §7.12: the
  fused Gated DeltaNet state update for one head) and AT's vector ops `ASMAX`, `AMASK`, `AMERGE`, `ANORM`
  (0x34–0x37, semantics by AT, executed by the VU's lanes, §5.11). `TOPK` 0x33 and `DQROW` 0x12 are simulator
  only, and `VMASK` 0x38 is specified but not implemented; the release image has none of the three.
- No op fails at run time. Every check runs at dispatch; shape limits use EX's code 13 `OPERAND`.
- Numerics are bit-exact: EXP is an integer algorithm with f32 input and output (at most 0.53 ulp, model); RECIP
  is correctly rounded everywhere; SIN and COS use an exact 64-bit 2/π reduction and f32 polynomials (absolute
  error at most 2^-23.6, model); SILU, SIGMOID, GELU-tanh and TANH are built from these; GELU16 rounds its input
  and output to binary16 and clamps at ±10, as the ggml CPU does; reductions keep the spec's 128 partial sums.
- Microarchitecture: with `VU_FRONT_LANES` = 16 the front lanes stream one line per cycle, each with 3 fp
  multipliers, 1 fp adder and an EXP core; `VU_FRONT_LANES` = 8 processes a line in two parts while retaining all
  16 logical lane histories. The release image uses 8 front, 8 EXP and 4 back lanes. The back lanes (1 ± e unit,
  an exact reciprocal, 2 multipliers) run SILU, GELU, SIGMOID, TANH and RECIP at half rate. A 9-adder reduction
  unit, an angle unit for RoPE, pair reads on CC's even and odd SPM and credit-based flow with no global clock
  enable complete the unit.

---

## 2. Inputs

- Spec: [spec](../spec.md) §9 (structure, scalar unit, approximate functions, RoPE, RMSNorm example), §11 (the ISA
  rows, VOP and SOP codes, ROPE and DQROW semantics), §7 (fp rules) and §21 (FTZ decision point, Tail bytes, Error
  check order, Range check, Zero length).
- RTL primitives: `tsr_fp_add` (latency 5, at most 8 allowed), `tsr_fp_mul` (latency 4, 2 DSP48E2) and
  `tsr_fp_div` (iterative, one op per 15 cycles). `PERF_VU_CYCLES` is index 3.
- EX: opcodes 0x12, 0x13, 0x30–0x33, 0x38–0x3F, 0x40, 0x41–0x47; 0x34–0x37 are AT's ops, which run on unit 3 (EX
  §7.2). w0[31:16] = `x16`, op-specific. Code 13 `OPERAND` and the check order (EX §8.1). `CAPS` bits 0, 3, 6, 7,
  8 and `CFG3[15:0]`. PERF slots X + 13 … X + 15 (EX §6). The s bank is +0.0 after a hardware reset and unchanged
  by `SOFT_RESET`. No op may fail after it starts.
- CC: the SPM has 2 read and 2 write ports in even and odd banks; the VU is client 4 with both bank ports and
  pair reads (CC §10.1–§10.2). `SREG0`–`15` at core CSR 0x9C0–0x9FC are read-only (EX §5.1); the RISC-V reads
  them through the core window (CC §5.3). VU queue depth 16 (CC §7.4). The checker class has four operand
  extents, each tagged SPM or DRAM (CC §8.2). I3 allows S − D ≤ 2 for the VU, as for the MVM (CC §9.4); units have
  no error output.
- Pinned llama.cpp: CPU GELU and GEGLU use an fp16 table (`GGML_GELU_FP16`, `ggml/src/ggml-cpu/vec.h`): they
  clamp at ±10, round x to fp16 and read `ggml_table_gelu_f16`, built with libm `tanhf`; CPU SiLU is
  `x/(1+expf(-x))`. RoPE angles follow the recurrence of `ggml_rope_cache_init` (`ggml/src/ggml-cpu/ops.cpp`),
  with `rope_yarn` for YaRN. FA is required for a quantized V cache; the SWA mask rule is in
  `src/llama-hparams.h`.

---

## 3. What the target models need

Gemma 4 applies an embedding scale sqrt(n_embd) as a MUL by the F32 graph input `inp_scale_rows` [1, T], freq
factors on full-attention layers (proportional RoPE uses 1e30 freq factors), per-head Q and K norms with weight
and a V norm without weight, GEGLU, a layer output scale and a final softcap as scale, tanh, scale; it has no
attention-logit softcap.

| ggml node | Qwen2.5 | Gemma 4 dense | VU ops (§8) |
|---|---|---|---|
| RMS_NORM (+ MUL weight) | attn, ffn, output norm | + post norms, per-head Q and K norm with weight, per-head V norm without weight | `RED.SUMSQ` (post RMS) + `VOP.MULCB` or `MULC` |
| ADD | QKV bias, residuals | residuals | `VOP.ADD` (bias: b stride 0), or MVM ACC |
| SCALE, MUL by 1-element or per-row tensor | | embedding × `inp_scale_rows` (per-row c vector), softcap 1/cap and cap, `layer_output_scale` | `VOP.MULC` (`cmode` 1 or 2), `ADDMULC` |
| GLU | SWIGLU | GEGLU (the CPU's fp16 form) | `VOP.SWIGLU`, `GEGLU16` |
| ROPE | NEOX 64 / 128 | NEOX 256 / 512 | `ROPE` |
| TANH | | final logit softcap | `VOP.TANH` |
| GET_ROWS (token_embd) | CPU | CPU | `LD` + `DQROW` where the image has it |
| CPY, CONT, f32 GET_ROWS | | | `VOP.COPY` |
| FLASH_ATTN_EXT | softmax pieces | idem + SWA mask | AT's ops (AT §6.6): ATTQK bounds from the mask summary, ASMAX, AMASK, AMERGE, ANORM on the VU; `VMASK` is not used for llama.cpp KV caches |

---

## 4. Programming model

### 4.1 Data and fp rules

- Elements: VU elements are 32-bit words.
- fp rules: every function marked "fp" follows the spec's rules: IEEE binary32, round-to-nearest-even,
  DAZ + FTZ as fixed in [spec](../spec.md) §21 (FTZ decision point), and canonical NaN 0x7FC00000
  (spec §7). It is built only from the primitives of §6.1 and integer operations.
- Raw functions: functions marked "raw" copy bits (COPY, SETC, SOP.MOVI, LDS and STS, the unmasked
  elements of VMASK, ROPE's unrotated tail). They also move integer data such as positions or indices.
- No subnormal output: no VU fp result is ever subnormal, and no fp NaN result is anything but
  0x7FC00000.

### 4.2 The 2-D operand model

- Rows and extents. An op processes `rows` rows of `m` elements. Define L(m) = ⌈m/16⌉. Row r of an
  operand with line address X and stride s_X occupies lines [X + r·s_X, X + r·s_X + L(m)). Element j of
  row r is the word at byte 64·(X + r·s_X) + 4j. Element j sits in lane j mod 16 of line ⌊j/16⌋.
  Results never depend on the number of hardware lanes.
- Stride 0 is legal for sources: every row then reads the same vector, which is how a norm weight or a
  bias is broadcast over heads or tokens.
- Destination stride: with rows > 1 the destination stride must be ≥ L(m), else `OPERAND`. Destination
  rows never overlap.
- Single row: with rows ≤ 1, strides are not used and may hold any value.
- Empty ops (n = 0 rule): m = 0 or rows = 0: the op reads and writes nothing in the SPM. Exception: a
  RED with rows = 1 and an s destination writes the empty-row result (§6.2). Range checks still apply,
  with extent 0 ([spec](../spec.md) §21, Range check and Zero length).
- Tail rule per row: when m mod 16 ≠ 0, words [m mod 16, 16) of each destination row's last line are
  written 0. This is spec §21 (Tail bytes) applied row by row.

### 4.3 The per-row scalar operand c

`cmode` 0 means no c. With `cmode` 1, c = s[si] for every element. With `cmode` 2, c_r is the 32-bit word
r of the vector starting at SPM line C, for every element of row r (extent ⌈rows/16⌉ lines). RED writes
such vectors (§5.3), so reductions feed the next op without a RISC-V round trip.

### 4.4 Scalar registers s0–s15

- One bank of 16 × 32 bits, inside the VU.
- Writers: RED (TO_S) and SOP (MOVI, LDS, arithmetic), only.
- Readers: VOP (`cmode` 1), the RED post-op (sk, se), ROPE (sm) and SOP. A VU op reads its s operands
  when it starts, after every earlier VU op's s write.
- Reset: +0.0 after a hardware reset. `SOFT_RESET`, RISC-V reset, QABORT and QCLEAR do not change the
  bank (EX §7.4). Kernels write every s register they read; the SDK kernel prologue does so with
  SOP.MOVI.
- RISC-V: read-only, through the core window at 0x2000_1000 + 0x9C0 + 4i (CC §5.3). The read returns the
  value written by VU op t once the RISC-V has observed done(t). There is no RISC-V write path; values
  go in with `SOP.MOVI`, which queue order and dependencies order.
- Host: `SREG0`–`15` at core CSR 0x9C0 + 4i, read-only (EX §5.1, DB §10.1), present whenever the VU is
  built. This is the one read location for every reader. There is no debug write path.
- Race definition: the 16 registers are 16 locations (SW §4.6). A core-window read of s_i must be
  ordered after the last VU op that writes s_i and before the next one.

### 4.5 Ordering, completion, aliasing

- Order: VU ops run in VU queue order. Each sees every SPM and s effect of earlier VU ops. As-if overlap
  inside the VU is allowed (§7.8).
- Done means: every SPM write accepted by `tsr_spm`, and the s write performed.
- No run-time errors: NaN, Inf and arbitrary bits are values, not errors (EX §7.2).
- Aliasing: a source may coincide with the destination only exactly (same start line and same stride;
  for ROPE, the same start and the same token stride). The op then works in place, element by element.
  Any other overlap between a source extent (including c, pos and table) and the destination extent is
  undefined; the simulator's checker flags it (SW). Sources may overlap each other freely.

---

## 5. Encodings

### 5.1 Common rules

- w0 = {x16 = w0[31:16], flags[15:12], sub[11:8], opcode[7:0]} (EX §7.1). A field that an op or function
  does not use must be 0, else `ILLEGAL_OP`.
- Field widths: SPM line addresses w[23:0] with w[31:24] = 0; element counts m 24 bits; rows, strides,
  head counts and T 16 bits. SPMs larger than 2^24 lines are outside these encodings.
- Unit: every op here runs on unit 3 (VU), queue depth 16 (CC §7.4).
- Range checks: [spec](../spec.md) §21 (Range check), with the extents of §5.9.

### 5.2 VOP (0x30)

| Field | Bits | Meaning |
|---|---|---|
| opcode | w0[7:0] | 0x30 |
| cmode | w0[11:8] | 0 none, 1 s register, 2 SPM row vector; 3–15 `ILLEGAL_OP` |
| flags | w0[15:12] | 0 |
| func | x16[7:0] | function, table below; undefined codes `ILLEGAL_OP` |
| si | x16[11:8] | s register for `cmode` 1, else 0 |
| | x16[15:12] | 0 |
| d | w1[23:0] | destination line |
| a | w2[23:0] | source a line (0 if the function reads no a) |
| b | w3[23:0] | source b line (0 if unused) |
| c | w4[23:0] | row-vector line for `cmode` 2, else 0 |
| m | w5[23:0] | elements per row |
| rows | w6[15:0] | rows |
| d_stride | w6[31:16] | lines |
| a_stride | w7[15:0] | lines (0 if a is unused) |
| b_stride | w7[31:16] | lines (0 if b is unused) |

Functions take inputs a, b, c. Formulas use §6 names. F = one line per cycle, H = half rate (§7.4). A
function that uses c requires `cmode` ≠ 0; one that does not requires `cmode` = 0.

| func | Name | Reads | Result y (per element) | Rate |
|---|---|---|---|---|
| 0x00 | ADD | a, b | FA(a, b) | F |
| 0x01 | SUB | a, b | FS(a, b) | F |
| 0x02 | MUL | a, b | FM(a, b) | F |
| 0x03 | MAX | a, b | MAXT(a, b) | F |
| 0x04 | MULC | a, c | FM(a, c) | F |
| 0x05 | ADDC | a, c | FA(a, c) | F |
| 0x06 | SILU | a | SILU(a) | H |
| 0x07 | SWIGLU | a, b | FM(SILU(a), b) | H |
| 0x08 | EXP | a | EXPK(a, P1) | F |
| 0x09 | COPY | a | a (raw) | F |
| 0x0A | MIN | a, b | MINT(a, b) | F |
| 0x0B | SUBC | a, c | FS(a, c) | F |
| 0x0C | MULCB | a, b, c | FM(FM(a, c), b) | F |
| 0x0D | MULCADD | a, b, c | FA(FM(a, c), b) | F |
| 0x0E | ADDMULC | a, b, c | FM(FA(a, b), c) | F |
| 0x0F | EXPSUB | a, b | EXPSUB(a, b) | F |
| 0x10 | EXPSUBC | a, c | EXPSUB(a, c) | F |
| 0x11 | SIGMOID | a | SIG(a) | H |
| 0x12 | GELU | a | GELU(a) | H |
| 0x13 | GEGLU | a, b | FM(GELU(a), b) | H |
| 0x14 | TANH | a | TANH(a) | H |
| 0x15 | RECIP | a | RC(a) (correctly rounded) | H |
| 0x16 | SETC | c | c (raw) | F |
| 0x17 | GELU16 | a | GELU16(a): the ggml-CPU fp16 form (§6.5) | H |
| 0x18 | GEGLU16 | a, b | GEGLU16(a, b) (§6.5) | H |
| 0x19–0xFF | | | `ILLEGAL_OP` | |

llama.cpp graphs use GELU16 and GEGLU16 (SW-31); GELU and GEGLU (exact tanh form) serve PyTorch.

### 5.3 RED (0x31)

| Field | Bits | Meaning |
|---|---|---|
| kind | w0[11:8] | 0 SUM, 1 SUMSQ, 2 MAX; 3–15 `ILLEGAL_OP` |
| TO_S | w0[12] | 1: result to s[sd] (rows must be 1); 0: per-row vector at d |
| | w0[15:13] | 0 |
| sd | x16[3:0] | destination s register if TO_S, else 0 |
| post | x16[5:4] | 0 NONE, 1 RCP, 2 RMS, 3 `ILLEGAL_OP`; MAX requires 0 |
| | x16[7:6] | 0 |
| sk | x16[11:8] | RMS scale register (typically 1/n), else 0 |
| se | x16[15:12] | RMS epsilon register, else 0 |
| d | w1[23:0] | result vector line (rows words); 0 if TO_S |
| a | w2[23:0] | source line |
| m | w5[23:0] | elements per row |
| rows | w6[15:0] | rows (w6[31:16] = 0) |
| a_stride | w7[15:0] | lines (w7[31:16] = 0) |

w3 and w4 are 0.

### 5.4 ROPE (0x32)

| Field | Bits | Meaning |
|---|---|---|
| mode | w0[11:8] | 0 NORM (adjacent pairs), 1 NEOX (half-split pairs); 2–15 `ILLEGAL_OP` |
| FF | w0[12] | recurrence form: the table holds freq factors ff_i (φ_i = FD(θ_i, ff_i)) |
| FS | w0[13] | recurrence form: multiply by freq_scale in s[sfs] |
| ATAB | w0[14] | A-table form (YaRN): θ_i = FM(Pf, A_i) from the table; FF, FS and sts must be 0, else `ILLEGAL_OP` |
| | w0[15] | 0 |
| sm | x16[3:0] | s register holding M (attn_factor, or the folded YaRN mscale with ATAB) |
| sts | x16[7:4] | s register holding TS = `powf(freq_base, −2/n_dims)` (recurrence form; 0 with ATAB) |
| sfs | x16[11:8] | s register holding freq_scale when FS, else 0 |
| | x16[15:12] | 0 |
| d | w1[23:0] | destination line of token 0 |
| src | w2[23:0] | source line of token 0 |
| pos | w3[23:0] | line of the I32 position vector (word t = position of token t) |
| table | w4[23:0] | line of the f32 table of n_dims/2 entries: freq factors (FF) or A_i (ATAB); 0 when neither |
| head_dim | w5[15:0] | elements per head |
| n_dims | w5[31:16] | rotated elements per head |
| n_heads | w6[15:0] | heads per token |
| T | w6[31:16] | tokens |
| src_stride | w7[15:0] | lines between tokens; 0 = H |
| d_stride | w7[31:16] | lines between tokens; 0 = H |

H = n_heads × head_dim / 16 lines (one token's heads, contiguous).

Legal shapes: head_dim is a multiple of 16 in [16, 1024]. n_dims is in [1, head_dim]: a multiple of 32
for NEOX, a multiple of 16 for NORM. Otherwise `OPERAND` (§5.9). With these rules every pair half
starts on a line boundary: line j pairs with line j + n_dims/32 in NEOX. They cover Qwen2.5 (64/64,
128/128) and Gemma 4 (256/256, 512/512).

### 5.5 TOPK (0x33; simulator only)

| Field | Bits | Meaning |
|---|---|---|
| k | x16[4:0] | 1 … 16 (x16[15:5] = 0) |
| d | w1[23:0] | output line of row 0: k pairs (word 2i index u32, word 2i+1 value) |
| a | w2[23:0] | source line |
| m | w5[23:0] | elements per row |
| rows | w6[15:0] | rows |
| d_stride | w6[31:16] | lines; 0 = ⌈k/8⌉ |
| a_stride | w7[15:0] | lines (w7[31:16] = 0) |

w3 and w4 are 0. The release image reports `CAPS` bit3 = 0 and refuses the opcode with `ILLEGAL_OP`.

### 5.6 DQROW (0x12)

| Field | Bits | Meaning |
|---|---|---|
| fmt | w0[11:8] | 0 Q8_0, 1 Q4_K; 2–15 `BAD_FMT` |
| d | w1[23:0] | f32 output line of row 0 |
| src | w2[23:0] | SPM line of row 0's repacked bytes (as stored in DRAM, [spec](../spec.md) §5; loaded by LD) |
| K | w5 | elements per row; `K_RANGE` if K = 0, K > K_MAX, or K not a multiple of the block size |
| rows | w6[15:0] | rows (w6[31:16] = 0) |
| src_stride | w7[15:0] | lines; 0 = ROW_BEATS(fmt, K) |
| d_stride | w7[31:16] | lines; 0 = L(K) |

x16, w3 and w4 are 0. The RISC-V computes the row's channel and offset (`tsr_row_location`) and pushes
`LD` (ROW_BEATS beats), then `DQROW` with a dependency on the LD. DQROW is in the simulator only; the
release image reports `CAPS` bit6 = 0 and refuses the opcode with `ILLEGAL_OP`.

### 5.7 VMASK (0x38; not implemented)

VMASK masks by κ_j = k0 + j, so it is valid only for caches where cell j holds position k0 + j, such as
the private cache of a resident decode loop. llama.cpp's KV caches are rings that are reused after a wrap
(Gemma 4's 1,536-cell SWA cache) and can be reordered by `seq_rm` or a state load; the FA kernel
therefore never uses VMASK and takes its bounds from AT's mask summaries (AT §7.3). The encoding and
semantics below are fixed; neither the simulator nor the RTL implements VMASK (`ILLEGAL_OP`, `CAPS`
bit7 = 0).

| Field | Bits | Meaning |
|---|---|---|
| sub | w0[11:8] | 0 (position window); others `ILLEGAL_OP` |
| x | w1[23:0] | operand line (in place) |
| W | w2 | window size, u32; 0 = no window |
| k0 | w3 | i32 position of column 0 |
| q0 | w4 | i32 query position of row group 0 |
| m | w5[23:0] | columns |
| rows | w6[15:0] | rows |
| G | w6[31:16] | rows per query position (≥ 1) |
| x_stride | w7[15:0] | lines |
| P | w7[31:16] | row period; 0 = none |

### 5.8 SOP (0x40)

Word layout: w1[3:0] sd, [7:4] sa, [11:8] sb (w1[31:12] = 0); w2 imm (f32 bits); w3[27:0] SPM word
address for LDS and STS (w3[31:28] = 0); w4–w7 = 0 and x16 = 0. Fields that a function does not use are
0.

| sub | Name | Effect |
|---|---|---|
| 0 | MOVI | sd ← imm (raw) |
| 1 | ADD | sd ← FA(sa, sb) |
| 2 | MUL | sd ← FM(sa, sb) |
| 3 | ADDI | sd ← FA(sa, imm) |
| 4 | MULI | sd ← FM(sa, imm) |
| 5 | SQRT | sd ← SQ(sa) |
| 6 | RECIP | sd ← RC(sa) |
| 7 | LDS | sd ← SPM word (raw) |
| 8 | STS | SPM word ← sa (raw; writes exactly 4 bytes, no tail rule) |
| 9 | SUB | sd ← FS(sa, sb) |
| 10 | DIV | sd ← FD(sa, sb) |
| 11 | RSQRT | sd ← RC(SQ(sa)) (two roundings, as SQRT then RECIP) |
| 12 | MAX | sd ← MAXT(sa, sb) |
| 13 | MIN | sd ← MINT(sa, sb) |
| 14 | EXP | sd ← EXPK(sa, P1) |
| 15 | | `ILLEGAL_OP` |

### 5.9 Dispatch checks

`OPERAND` (13). Conditions are tested in this order; `ERR_INFO` is the value shown.

| Op | Condition | `ERR_INFO` |
|---|---|---|
| VOP | rows > 1 and d_stride < L(m) | d_stride |
| RED | TO_S and rows ≠ 1 | rows |
| ROPE | ATAB with FF, FS or sts ≠ 0 → `ILLEGAL_OP` (before `OPERAND`) | 0 |
| ROPE | head_dim = 0, head_dim mod 16 ≠ 0, or head_dim > 1024 | head_dim |
| ROPE | n_dims = 0, n_dims > head_dim, NEOX with n_dims mod 32 ≠ 0, or NORM with n_dims mod 16 ≠ 0 | n_dims |
| ROPE | T > 1, n_heads > 0 and d_stride′ < H (d_stride′ = d_stride, or H if 0) | d_stride |
| TOPK | k = 0 or k > 16 | k |
| TOPK | rows > 1 and d_stride′ < ⌈k/8⌉ | d_stride |
| DQROW | rows > 1 and d_stride′ < L(K) | d_stride |
| VMASK | G = 0 | 0 |
| VMASK | rows > 1 and x_stride < L(m) | x_stride |

SPM extents in lines. E(n, s, len) = 0 if n = 0 or len = 0, else (n − 1)·s + len. Operands are checked
in field order (w1, w2, w3, w4). `ERR_INFO` of `SPM_RANGE` is the start of the first failing operand.

| Op | Extents |
|---|---|
| VOP | d E(rows, d_stride, L(m)); a, b likewise when read; c ⌈rows/16⌉ (rows ≥ 1, `cmode` 2) |
| RED | d ⌈rows/16⌉ (not TO_S); a E(rows, a_stride, L(m)) |
| ROPE | with T′ = T if n_heads > 0 else 0: d E(T′, d_stride′, H); src E(T′, src_stride′, H); pos ⌈T′/16⌉; table (FF or ATAB) T′ > 0 ? ⌈n_dims/32⌉ : 0 |
| TOPK | d E(rows, d_stride′, ⌈k/8⌉); a E(rows, a_stride, L(m)) |
| DQROW | d E(rows, d_stride′, L(K)); src E(rows, src_stride′, ROW_BEATS(fmt, K)) |
| VMASK | x E(rows, x_stride, L(m)) |
| SOP LDS / STS | word address + 1 ≤ `SPM_BYTES`/4; `ERR_INFO` = the word address |

VOP and ROPE need four SPM extents, as do AT's AMASK and AMERGE. CC's checker class slot has four operand
extents, each tagged SPM or DRAM (CC §8.2); no VU op has a DRAM operand. The checks of AT's vector ops
are AT §5.8's rows.

### 5.10 CAPS, CFG3, PERF, debug state

- `CAPS`: bit0: VOP, RED, ROPE and SOP as defined here. bit3: TOPK. bit6: DQROW in the SPM form. bit7:
  VMASK. bit8: GDN (the fused state update, §7.12; the RTL builds with `HAS_GDN`). Bits 9–11: reserved
  for VU extensions, 0. The simulator reports the bits of the configuration it implements.
- `CFG2` is unchanged: VL = 16, NSREG = 16.
- `CFG3[15:0]`: [4:0] front lanes with an EXP core, [9:5] back lanes, [15:10] 0. This is information for
  cycle models only; results never depend on it.
- PERF. Index 3 `VU_CYCLES` counts cycles in which the VU holds at least one accepted op (from accept to
  its done), gated as EX §6 defines. Extension slots, in this order:

  | Slot | Name | Kind | Counts |
  |---|---|---|---|
  | X + 13 | `VU_SPM_WAIT` | time | cycles with a raised, ungranted VU SPM request |
  | X + 14 | `VU_HAZARD` | time | cycles in which an accepted op waits for an earlier VU op (§7.8) |
  | X + 15 | `VU_LINES` | event | destination lines written by VU ops |

- Debug state exported for DB to place in `DBG_*`: w0 of the oldest held op; its current row and line;
  the number of ops held; flags (reduction draining, waiting for SPM, waiting on a hazard). DB exposes
  the per-unit state through `DBG_U3_*` (DB §5.5); the VU-specific fields are readable through DB's probe
  words when DB places them, and no contract depends on them.

### 5.11 AT's vector ops on the VU (0x34–0x37)

AT defines the semantics (AT §5.4–§5.7, normative) and the checks (AT §5.8); the VU executes the ops on
unit 3 with the datapath of §7. This table restates the encodings for the VU decoder; both come from the
single table source (EX-30). For every op x16 = 0 and unused words are 0, else `ILLEGAL_OP`.

| Op | w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7 |
|---|---|---|---|---|---|---|---|---|
| `ASMAX` 0x34 | sub 0, flags 0 | dst | src (may equal dst) | mvec (row maxima) | [15:0] n_lines (1–4095), [31:16] stride (0 → n_lines) | R (1–1024) | scale (f32, finite, normal, > 0) | 0 |
| `AMASK` 0x35 | sub 0, flags 0 | dst | src | mask (f32 rows) | [15:0] n_lines, [31:16] stride | [15:0] R, [31:16] rpq | mask stride (lines) | mdst (row maxima out) |
| `AMERGE` 0x36 | flags [12] INIT | Orun | parts (an ATTPV dst with LSUM) | mk (chunk row maxima) | st (m_run at st, l_run at st + ⌈R/16⌉) | [15:0] R, [17:16] dcode | scale | cstride (as the producing ATTPV) |
| `ANORM` 0x37 | sub 0, flags 0 | out | Orun | l_run | [15:0] R, [17:16] dcode | 0 | 0 | 0 |

Semantics in VU notation (AT §5.4–§5.7; EXP = EXPK(·, P1), max_tot = MAXT, RECIP = RC; M_FLOOR =
0xF1800000):

- ASMAX: dst[r][k] = EXPK(FM(FS(src[r][k], mvec[r]), scale), P1).
- AMASK: dst[r][k] = (bits(m) = 0xFF800000) ? −∞ : FA(src[r][k], m) with m = mask[⌊r/rpq⌋][k];
  mdst[r] = MAXT(M_FLOOR, MAXT over the row).
- AMERGE: Ok = CTREE over the C channel partials (pairwise by levels, left to right), lk likewise; INIT
  copies; otherwise m′ = MAXT(m_run, mk), a = EXPK(FM(FS(m_run, m′), scale), P1),
  b = EXPK(FM(FS(mk, m′), scale), P1), Orun ← FA(FM(Orun, a), FM(Ok, b)), l_run ← FA(FM(l_run, a),
  FM(lk, b)), m_run ← m′.
- ANORM: out[r][d] = FM(Orun[r][d], RC(MAXT(l_run[r], 2⁻¹²⁶))).

Datapath mapping and rates: §7.4. These ops never fail at run time; their operand limits are `OPERAND`
checks at dispatch (AT §5.8).

---

## 6. Numerics (bit-exact)

### 6.1 Notation and primitives

- fp primitives: FA = `tsr_fadd`, FS = `tsr_fsub`, FM = `tsr_fmul`, FD = `tsr_fdiv`, SQ = `tsr_fsqrt`,
  RC = `tsr_frecip` (`sw/sim/tsr_ref.c`). They implement DAZ, RNE, FTZ and canonical NaN, and are
  correctly rounded.
- Other notation: DAZ(x) maps exponent-0 encodings to signed zero. Constants are bit patterns
  (Appendix A). "|x|" clears the sign bit; "−x" flips it.
- Integer operations are unsigned unless marked signed. `>>` is a floor shift. A shift by 64 or more of a
  64-bit value gives 0. Every intermediate fits in 64 bits, except the two SIN and COS products of §6.6,
  which are exact 128-bit products.
- The C reference is `sw/sim/tsr_ref_math.c`, under SW §4.5's rules. Its API: `tsr_m_expk(float, unsigned
  k)`, `tsr_m_exp`, `tsr_m_sigmoid`, `tsr_m_silu`, `tsr_m_gelu_tanh`, `tsr_m_gelu16`, `tsr_m_tanh`,
  `tsr_m_sincos`, `tsr_m_sin`, `tsr_m_cos`; the binary16 helpers `tsr_f32_to_f16_rne` (gradual underflow)
  and `tsr_f16_to_f32` used by GELU16. RC is `tsr_frecip`; there is no `tsr_m_recip_approx`. VOP,
  reductions, ROPE, TOPK, SOP, DQROW and VMASK are in `tsr_ref_vu.c` (`include/tsr_ref_vu.h`); the AT
  vector ops are in `tsr_ref_at.c` (AT §13.1). Both are part of `libtsr_ref`.

### 6.2 Reductions (RED)

Each row is reduced on its own.

- SUM: set p_0 … p_127 = +0.0. For i = 0 … m − 1 in order, p_(i mod 128) ← FA(p_(i mod 128), x_i).
- SUMSQ: the same with FM(x_i, x_i) in place of x_i (the square is rounded first).
- Tree: the result is the complete binary tree over p_0 … p_127. Level 1 is FA(p_{2k}, p_{2k+1}). Each
  level pairs adjacent results left to right; there are 7 levels. This is the spec's rule
  ([spec](../spec.md) §9).
- MAX: the maximum of the DAZ'd elements under the total order of §6.8. Any NaN gives 0x7FC00000.
- Empty rows (m = 0): SUM and SUMSQ give +0.0; MAX gives −∞ (0xFF800000).
- Post-op, applied to each row result y: NONE: y. RCP: RC(y). RMS: RC(SQ(FA(FM(y, s[sk]), s[se]))). RMS
  is bit-identical to the spec recipe SOP.MULI → ADDI → SQRT → RECIP (spec §9, RMSNorm example) when
  s[sk] equals the MULI immediate.

Implementation freedom (exact). A partial is initialized as FA(+0, x) (the first update of a +0
partial; never a raw load of x). A computed tree value q can be −0: the spec's FTZ keeps the sign, so
for example FA(0x80C00000, 0x00800000) = 0x80000000, while FA(−0, +0) = +0. Tree values are never
subnormal, so FA(q, +0) = z(q), where z maps −0 to +0 and leaves every other value unchanged. When the
hardware skips a subtree whose partials were all unwritten (+0), it therefore passes z(q) upward in
place of the computed subtree value q. With this rule, a short row (m < 128) uses only the partials
p_0 … p_{m−1} and needs ⌈m/8⌉ tree cycles (§7.5), and the result is bit-identical to the complete
128-partial tree. Directed tests cover subtrees that sum to −0 at m = 8k + r (§10).

### 6.3 EXPK(v, κ): exponential with a folded constant

EXPK(v, κ) computes exp(K·v), with K and its fixed-point constant LK ≈ |K|·log₂e·2³⁸ set by κ:

| κ | K | Use | sK | LK |
|---|---|---|---|---|
| P1 | +1 | EXP, EXPSUB | 0 | 0x5C551D94AE (396,564,993,198) |
| N1 | −1 | SIGMOID, SILU | 1 | 0x5C551D94AE |
| N2 | −2 | TANH | 1 | 0xB8AA3B295C (= 2 × LK(P1)) |
| NG | −C2G | GELU, C2G = 2·√(2/π)·0.044715 | 1 | 0x6969F3DD6 (28,296,822,230) |

LK = round(|K|·log₂(e)·2³⁸). LK(N2) is exactly twice LK(P1), so EXPK(v, N2) = EXPK(2v, N1) whenever 2v
is exact.

Algorithm (u = bits of v):

1. v ← DAZ(v). If v is NaN, return 0x7FC00000.
2. sgn = sign(v) XOR sK.
3. If v = ±∞: return sgn ? +0 : +∞. If v = ±0: return 0x3F800000.
4. Ev = biased exponent (1 … 254); Mv = 2²³ | fraction.
5. If Ev > 156: return sgn ? +0 : +∞ (saturate).
6. P = Mv × LK (< 2⁶⁴). T = P >> (156 − Ev). This is |K·v·log₂e| in units of 2⁻³². If T ≥ 2⁴⁰, saturate
   as in step 5.
7. t = sgn ? −T : T (signed). n = t >> 32 (arithmetic, floor). F = t mod 2³² ∈ [0, 2³²).
8. j = F >> 26 (6 bits). G = F mod 2²⁶.
9. Polynomial for 2^(G·2⁻³²) − 1, Taylor to the cubic term, in units of 2⁻³²:
   - X3 = (G × P3F) >> 32
   - S2 = P2F + X3
   - X2 = (G × S2) >> 32
   - S1 = P1F + X2
   - PF = (G × S1) >> 32
   - with P3F = 0x0E35846C, P2F = 0x3D7F7BFF, P1F = 0xB17217F8, which are round(ln³2/6·2³²),
     round(ln²2/2·2³²) and round(ln2·2³²).
10. M = T64[j] + ((T64[j] × PF) >> 32), where T64[j] = round(2^(j/64)·2³⁰) (Appendix A). M is in
    [2³⁰, 2³¹] and represents 2^frac in Q1.30.
11. Normalize: if M ≥ 2³¹, then drop = 8 and n ← n + 1; else drop = 7.
12. Round to 24 bits: sig = M >> drop; rem = M mod 2^drop; half = 2^(drop−1). If rem > half, or rem = half
    and sig is odd: sig ← sig + 1. If sig = 2²⁴: sig ← 2²³ and n ← n + 1.
13. be = n + 127. If be ≥ 255 return +∞; if be ≤ 0 return +0; else return (be << 23) | (sig − 2²³).

EXPK never returns −0 or a subnormal.

Accuracy (model): at most 0.53 ulp against exact exp(K·v) for every κ (0.5261 ulp for P1, N1 and N2,
0.5316 ulp for NG over all 2³² inputs, §6.9). EXPK is monotonic. In hardware it takes 11 DSP per core
(estimate).

### 6.4 Correctly rounded reciprocal, square root, division

RC, SQ and FD are the spec's correctly rounded operations. VOP.RECIP, the RED post-ops and the
reciprocal inside SIGMOID, SILU, GELU and TANH all use RC. The implementation is free (§7.3). The
contract is bit equality with `tsr_frecip`, `tsr_fsqrt` and `tsr_fdiv`.

### 6.5 Composite functions

- SIG(x) = RC(FA(ONE, EXPK(x, N1))).
- SILU(x) = FM(x, SIG(x)). This keeps the spec's form x·R(1 + e^−x), with an exact R.
- GELU(x), the tanh form, using 0.5·(1 + tanh u) = σ(2u) and 2u = C2G·x·(x² + KG): w = FM(x, x);
  w = FA(w, KG); v = FM(w, x); y = FM(x, RC(FA(ONE, EXPK(v, NG)))). KG = f32(1/0.044715) = 0x41B2E92F;
  C2G sits inside LK(NG).
- TANH(x): if x is NaN, return 0x7FC00000. Let ax = |DAZ(x)|. If ax < 2⁻⁶ (bits < 0x3C800000), the
  small path: q = FM(ax, ax); w = FS(THREE, q); t = FM(FM(w, ax), THIRD), where THREE = 0x40400000 and
  THIRD = 0x3EAAAAAB. Else: e = EXPK(ax, N2); t = FM(FS(ONE, e), RC(FA(ONE, e))). y = sign(x) ? −t : t.
- GELU16(x), the ggml CPU's form (`ggml_vec_gelu_f32` with `GGML_GELU_FP16`, `ggml/src/ggml-cpu/vec.h`).
  Let H(v) = F32(F16(v)), where F16 rounds DAZ(v) to binary16 with round-to-nearest-even and gradual
  underflow (fp16 subnormals are kept, as F16C `_cvtss_sh` in rounding mode 0 does; overflow gives ±∞)
  and F32 widens exactly. Then, on x′ = DAZ(x): NaN → 0x7FC00000; x′ ≤ −10 → +0; x′ ≥ 10 → x′;
  otherwise H(GELU(H(x′))), with GELU of this section evaluated in f32 on the fp16-exact input. The CPU
  reads `ggml_table_gelu_f16`, whose entries are F16 of `ggml_gelu_f32`: 0.5·x·(1 + `tanhf`(u)) in f32
  with libm `tanhf` (and, in the pinned `-march=native` build, the FMA contraction of 1 + a·x·x). SW's
  `ref_math_gelu16` enumerates all 65,536 fp16 inputs and compares with the table. GELU16 differs from
  the table in 272 entries (0.415 %): 265 by one fp16 ulp and 7 by two. In all 272 GELU16 is the
  correctly rounded fp16 value of the exact GELU and the table entry is not: the residual is the table's
  own f32 error (`tanhf` and the f32 steps; for x ≪ 0, 1 + `tanhf`(u) cancels, which gives the 2-ulp
  entries). A bound of one fp16 ulp for every entry is therefore not reachable without reproducing the
  CPU's cancellation. The gate is: fewer than 1 % of the entries differ, no sign-of-zero mismatch, and a
  difference above one ulp only where GELU16 is the correctly rounded value.
- GEGLU16(a, b) = (DAZ(a) ≤ −10) ? +0 : FM(GELU16(a), b), as `ggml_vec_geglu_f32`.
- EXPSUB(a, x) = (DAZ(a) = −∞) ? +0 : EXPK(FS(a, x), P1). Masked scores (−∞) become 0 even when the row
  maximum is −∞. That makes an online softmax that starts on a fully masked block well defined instead
  of NaN.
- Specials:

  | Function | at −∞ | at +∞ |
  |---|---|---|
  | EXP | +0 | +∞ |
  | SIG | +0 | 1 |
  | SILU | NaN (as the CPU's `x/(1+expf(-x))`) | +∞ |
  | GELU | NaN | +∞ |
  | GELU16 | +0 | +∞ |
  | TANH | −1 | +1 |

  Finite inputs never give NaN in SIG, SILU, GELU or TANH. GELU of a large negative finite x is −0,
  because v overflows to −∞ and the reciprocal goes to 0.

### 6.6 SIN and COS

Input and specials. θ ← DAZ(θ). NaN or ±∞ give SIN = COS = 0x7FC00000. Let s be the sign, E the biased
exponent and M = 2²³ | fraction.

1. E = 0: SIN = θ (a signed zero), COS = ONE.
2. E < 115 (|θ| < 2⁻¹²): SIN = θ, COS = ONE. The error is below 0.1 ulp; cos rounds to 1.

Quadrant reduction (exact integers; C64 = round(2/π·2⁶⁴) = 0xA2F9836E4E44152A):

3. P = M × C64 (88-bit exact). sh = 214 − E. θ·2/π ≈ P·2^−sh.
   - sh ≥ 62: k = P >> sh; F = (P >> (sh − 62)) mod 2⁶².
   - 0 ≤ sh < 62: k = P >> sh; F = (P << (62 − sh)) mod 2⁶².
   - sh < 0: k = P << −sh (only k mod 4 is used); F = 0.
4. If F ≥ 2⁶¹: k ← k + 1 and F ← F − 2⁶² (signed, in [−2⁶¹, 2⁶¹)). q = k mod 4.
5. R = F × PIO2, with PIO2 = round(π/2·2⁶²) = 0x6487ED5110B4611A. This is a signed exact product of at
   most 125 bits. r = the f32 nearest-even to R·2⁻¹²⁴ (+0 if R = 0). |r| is 0 or ≥ 2⁻⁶¹, so r is normal.

Polynomials (Cephes `sinf` and `cosf` coefficients on [−π/4, π/4]):

6. z = FM(r, r).
   - sn: t = FM(SN3, z); t = FA(t, SN2); t = FM(t, z); t = FA(t, SN1); t = FM(t, z); t = FM(t, r);
     sn = FA(t, r).
   - cs: t = FM(CS3, z); t = FA(t, CS2); t = FM(t, z); t = FA(t, CS1); t = FM(t, z); t = FM(t, z);
     h = FM(HALF, z); u = FS(t, h); cs = FA(u, ONE).
   - Coefficients: SN1 0xBE2AAAA3, SN2 0x3C08839E, SN3 0xB94CA1F9; CS1 0x3D2AAAA5, CS2 0xBAB6061A,
     CS3 0x37CCF5CE; HALF 0x3F000000.

Quadrant and sign:

7. (S, C) = q = 0: (sn, cs); 1: (cs, −sn); 2: (−sn, −cs); 3: (−cs, sn).
8. SIN = s ? −S : S. COS = C.

Accuracy (model): absolute error at most 7.5·10⁻⁸ = 2⁻²³·⁶⁶, and at most 1.47 ulp where |y| > 10⁻³. For
|θ| < 2²⁴ the reduction error is at most 2⁻⁴¹ of a quadrant. Larger θ stay deterministic but are
meaningless for RoPE: ulp(θ) ≥ 2 there.

### 6.7 ROPE

For each token t < T:

- p = word t of `pos` (int32). Pf = `tsr_i32_to_f32`(p).
- Angles, recurrence form (ATAB = 0; the CPU's `ggml_rope_cache_init` for ext_factor = 0,
  `ggml/src/ggml-cpu/ops.cpp`): θ_0 = Pf and θ_{i+1} = FM(θ_i, s[sts]) for i < n_dims/2 − 1, where s[sts]
  holds the bits of the host's `powf(freq_base, -2.0f/n_dims)` (the same libm as ggml-cpu). φ_i =
  FF ? FD(θ_i, ff_i) : θ_i, with ff_i word i of the table; then φ_i ← FS ? FM(s[sfs], φ_i) : φ_i.
- Angles, A-table form (ATAB = 1, for YaRN with ext_factor ≠ 0): φ_i = FM(Pf, A_i), where the host folds
  freq_base, freq_scale, the freq factors and the YaRN ramp mix into A_i in double and rounds once.
- c_i = FM(M, COS(φ_i)), s_i = FM(M, SIN(φ_i)), with M = s[sm] (the CPU's `cosf(θ)·mscale`).

For each head h < n_heads of token t:

- The pair is (j0, j1) = (i, i + n_dims/2) in NEOX, or (2i, 2i + 1) in NORM. Indices are relative to the
  head's first element. The head is at word h·head_dim of the token's H-line span.
- x0 = src[j0], x1 = src[j1].
- y[j0] = FS(FM(x0, c_i), FM(x1, s_i)).
- y[j1] = FA(FM(x0, s_i), FM(x1, c_i)).
- Elements [n_dims, head_dim) are copied raw.

The rotation is ggml's `rotate_pairs` without FMA contraction.

Software side. Software computes the constants once per model and layer type:

- recurrence form: TS = `powf(freq_base, -2.0f/n_dims)` in f32 with the host libm, M = attn_factor,
  freq_scale (FS only if ≠ 1), and the freq-factor table as given by the GGUF (Gemma 4 global layers:
  `rope_freqs` [256] with 1e30 entries);
- A-table form (YaRN only): A_i and M in double with one rounding (`rope_yarn` in ggml-cpu;
  [spec](../spec.md) §9).
- On Gemma 4 global layers, ff_i = 10³⁰ gives φ_i ≈ 10⁻³⁰·θ_i (a normal f32). φ_i < 2⁻¹² then takes the
  identity path of §6.6, and those pairs change only by the x0 − x1·φ rounding, as on the CPU.
- With the recurrence the VU's angles φ_i are bit-identical to the CPU's for ext_factor = 0. The
  remaining differences are libm `cosf` and `sinf` against §6.6 and the compiler's possible FMA
  contraction of the CPU rotation (§6.9). The serial FM chain costs n_dims/2 × 4 cycles per token (at
  most 1 K cycles for n_dims 512) and is interleaved across tokens in the angle unit (§7.6).

### 6.8 Order, raw ops, VMASK, TOPK, DQROW

- Total order ([spec](../spec.md) §9): non-NaN values are compared after DAZ by key = sign ? ~bits :
  bits | 2³¹, so −∞ < … < −0 < +0 < … < +∞. NaN is above everything. MAXT(a, b): if either is NaN,
  0x7FC00000; else the larger DAZ'd value. MINT(a, b): if both are NaN, 0x7FC00000; if one is NaN, the
  other (DAZ'd); else the smaller. By the order, NaN loses a MIN.
- VMASK (64-bit signed integer arithmetic): q_r = q0 + ⌊(P ≠ 0 ? r mod P : r) / G⌋ and κ_j = k0 + j.
  Element (r, j) is masked iff κ_j > q_r, or W ≠ 0 and q_r − κ_j ≥ W. That is `is_masked_swa` with
  standard SWA (`llama-hparams.h`) plus the causal rule. Masked elements become 0xFF800000. Others keep
  their bits. Tails follow §4.2. Rows ordered (token, head) use G = n_head and P = 0. Rows ordered
  (head, token) use G = 1 and P = T.
- TOPK, per row: the candidates are (j, x̂_j), where x̂ = DAZ(x) and NaN becomes 0x7FC00000. Sort by x̂
  descending under the total order; on a tie the smaller j comes first. Pair i < k goes to words 2i (j as
  u32) and 2i + 1 (x̂). If m < k, the missing pairs are (0xFFFFFFFF, 0xFF800000). Tails follow §4.2.
- DQROW: row r = `tsr_dequant_row`(fmt, the bytes at line src + r·src_stride′, K) (`sw/sim/tsr_ref.c`;
  ggml-exact for finite scales, spec §11 and §21, Agreement with ggml). Tails follow §4.2.

### 6.9 Accuracy and differences from the ggml CPU backend

The model figures are error bounds of the algorithms. The exhaustive test over all 2³² inputs (`ref_math_acc
--full`, against double libm; §10 item 2) holds every function to 2 times its figure, with ulp taken at
max(⌊log₂|r|⌋, −126) and results below 2⁻¹²⁶ (the FTZ region) not measured.

| Function | Model figure |
|---|---|
| EXPK | at most 0.53 ulp |
| RC, SQ, FD, RECIP, RSQRT steps | at most 0.5 ulp each (correctly rounded) |
| SIG | at most 2.0 ulp |
| SILU | at most 2.3 ulp where σ(x) ≥ 2⁻¹²⁶; below that SIG flushes to 0 and SILU returns −0 |
| GELU | at most 11 ulp for x ≥ −3 and 29 ulp for x ≥ −5; absolute at most 5.2·10⁻⁷ on [−8, 8] |
| TANH | at most 9.8 ulp and absolute at most 1.2·10⁻⁷ (the n = 2 − d variant would give 26 ulp, hence 1 − e); the worst case is just above the 2⁻⁶ switch, where 1 − e cancels |
| SIN, COS | absolute at most 2⁻²³·⁶⁶ and at most 1.5 ulp where \|y\| > 10⁻³ |

GELU's relative error grows in the far negative tail by about 1.5·|2u| ulp. There the result is below
10⁻¹⁰ in magnitude.

No VU op is bit-exact with the CPU in general; the KL criterion covers the differences, and the VU
follows the CPU's form wherever the CPU's form is a defined integer or f32 sequence. Known sources:

- GELU: llama.cpp graphs use GELU16 and GEGLU16, which reproduce the CPU's fp16 rounding and clamps; the
  residual is the CPU table's libm `tanhf`. The exact tanh GELU would differ from the CPU by up to
  3.9·10⁻³ relative per element (model), on every GEGLU element of every Gemma 4 layer, which is why it
  is not used for llama.cpp.
- RMSNorm: the CPU sums in double and divides by n; the VU uses f32 with 128 partial sums and 1/n.
- RoPE: the angles are bit-identical (recurrence form); `cosf` and `sinf` differ from §6.6, and the CPU
  compiler may contract x0·c − x1·s into an FMA.
- SiLU: the CPU's SIMD exp approximations.
- FA softmax: the CPU's FA kernel uses its own expf and f16 paths (AT §6.7).

---

## 7. Microarchitecture

### 7.1 Block diagram

```
 VU queue head (checked command) ──► sequencer: address generators d,a,b,c,pos,table; row/line counters;
        │                             hazard check against held ops (§7.8); credits = free output slots
        │                                   │ pair reads (2 lines of one operand per cycle)
        │          tsr_spm client 4  ◄──────┘ bank E + bank O read ports;  write: 1 line/cycle
        ▼                                   │
  s0–s15 (16×32 FF) ──► SREG (CC), SOP      ▼ input reorder (2 lines per source) ─► a, b, c (+ angles)
                     ┌──────────────── front lane × 16 (one line per cycle) ─────────────────┐
                     │ FM_A ─┐                                                               │
   a,b,c,const ──────┤       ├─► FA ─► FM_C ─► EXPK ─► cmp/select ─► full-rate result ───────┼─┐
                     │ FM_B ─┘   ▲ partial RF (2×8 phases)                                   │ │
                     └───────────┼───────────────────────────────────────────────────────────┘ │
                                 │                       back lane × 8 (half rate):            │
                                 │   e ─► ONE (1+e, 1−e) ─► RC (exact) ─► FM_R ─► FM_S ────────┤
                                 ▼                                                             ▼
             reduction: 8-input subtree (7 FA) ─► top (2 FA, scheduled) ─► post (FM, FA, SQ, RC) ─► s / row vector
             angle unit (ROPE): θ = FM(Pf, A_i) ─► int reduction ─► sin/cos polys (FM, FA) ─► ×M ─► angle buffer
                                                                                               ▼
                                              output FIFO (≥ pipeline depth, 64 lines) ─► SPM write
```

### 7.2 Front lane (× 16)

- Units: FM_A and FM_B take operands, constants or lane registers in parallel. FA takes FM_A, FM_B,
  operands, constants or the lane's partial register file (RED accumulation, 8 phases,
  double-buffered). FM_C follows FA. EXPK takes the raw operand, the FA output (EXPSUB) or the FM_C
  output (GELU). Compare and select handle MAX, MIN, VMASK, the EXPSUB −∞ rule, raw ops and tail zeroing.
- Blocks: FM and FA are the existing `tsr_fp_mul` (latency 4) and `tsr_fp_add` (latency 5), used with
  `ce` tied high.
- EXPK core (`tsr_vu_exp`): Mv × LK (24 × 40 bits, 2–3 DSP); barrel shift and saturation; three Horner
  products (26 × 28, 26 × 30 and 26 × 32 bits, 2 DSP each); the T64 ROM (64 × 31 bits, about 31 LUT6);
  T64 × PF (31 × 26 bits, 2 DSP); rounding. Latency about 18 cycles (estimate), at most 6 LUT levels
  per stage.
- RED accumulation: each lane's FA updates phase (line index mod 8) of its partials. A partial is updated
  every 8 cycles, so latency 5 suffices ([spec](../spec.md) §9).

### 7.3 Back lane (× 8)

- ONE computes FA(ONE, e) and FS(ONE, e): exact-equivalent specialized adders for e ≥ 0 or NaN, which is
  all EXPK produces.
- RC (`tsr_vu_rcp`, pipelined, correctly rounded): a 256-segment quadratic approximation with
  coefficients in one RAMB18 gives a quotient candidate within one unit of the 24-bit floor. One exact
  24 × 25-bit correction product chooses the floor and the RNE rounding. There are no ties for 1/m.
  Latency about 12 (estimate). The design is free; equality with `tsr_frecip` is the contract, verified
  exhaustively over the 2²³ mantissas and the exponent edges.
- FM_R and FM_S.
- fp16 rounding stages for GELU16 and GEGLU16: one at the lane input (H(x) and the ±10 clamp decision, in
  front lanes 0–7) and one after FM_R (H of the GELU result, before FM_S for GEGLU16); integer logic only.
- SRL delay lines carry a, b, the clamp decision and the TANH small-path value alongside.

Half-rate functions present each line as two 8-element halves on consecutive cycles. Front lanes 0–7 feed
back lanes 0–7, and front lanes 8–15 idle.

### 7.4 Function mapping and latency

| Function | Units | Rate |
|---|---|---|
| ADD, SUB, ADDC, SUBC | FA | F |
| MUL, MULC | FM_A | F |
| MULCB | FM_A → FM_C | F |
| MULCADD | FM_A → FA | F |
| ADDMULC | FA → FM_C | F |
| MAX, MIN, COPY, SETC (VMASK) | cmp/select | F |
| EXP; EXPSUB(C) | EXPK; FA → EXPK | F |
| SIGMOID, SILU, SWIGLU | EXPK → ONE → RC → FM_R (→ FM_S) | H |
| GELU, GEGLU | FM_A → FA → FM_C → EXPK → ONE → RC → FM_R (→ FM_S) | H |
| GELU16, GEGLU16 | H → FM_A → FA → FM_C → EXPK → ONE → RC → FM_R → H, clamp select (→ FM_S) | H |
| TANH | EXPK → ONE → RC → FM_R; small path FM_A → FA → FM_C → FM_S | H |
| RECIP | RC | H |
| ROPE rotation | FM_A ∥ FM_B → FA | F (one output line per cycle) |
| RED SUM / SUMSQ / MAX | (FM_A →) FA or cmp into partials | F (§7.5) |
| ASMAX | FA (as FS, m_r from a row vector) → FM_C (× scale) → EXPK | F |
| AMASK | FA (src + mask, with the −∞ select) → cmp; row maxima through the reduction unit's MAX path with M_FLOOR | F (two sources, pair reads) |
| AMERGE | prepass over the row vectors (at most 64 lines): MAXT, FS, FM, EXPK give a_r, b_r (row buffer) and the l_run update; then per output line: CTREE of the C partial lines on FA (C − 1 passes), FM_A(Orun, a_r) ∥ FM_B(Ok, b_r) → FA | about 4–5 cycles per output line at C = 4 (5 line reads, 4 FA uses); INIT about 3 |
| ANORM | prepass: RC(MAXT(l_r, 2⁻¹²⁶)) on the back lanes into the row buffer; then FM_A(Orun, rinv_r) | F (after the prepass) |

There are two latency classes, short (FM_A … FM_C, cmp, EXPK) and long (the half-rate functions). An op uses
one class for all its elements. An op that waits for the one before it takes 59 cycles (60 when it reads both
a and b) from accept to done, plus its stream time, in the short class and 95 (96) in the long class, and
every lane-pipe stage (`VU_LANE_PIPE`) adds 2 cycles.

### 7.5 Reduction unit and row post

- Draining a row: at a row's end its 128 partials (double-buffered) drain through an 8-input subtree adder
  (7 FA, 3 levels). It consumes 8 partials per cycle and yields 16 subtree sums per row, or ⌈m/8⌉ when
  m < 128 (§6.2). A scheduled 2-FA top combiner finishes the last 4 levels.
- Partial ring: a row of L lines holds min(L, 8) of the 16 ring entries from its first line until its drain
  ends, and the drain starts 17 cycles after the row's last line. A row with L ≥ 8 therefore leaves room
  for one more row (two rows in flight) and a row with L = 4 for three more.
- Throughput: in a stream of rows a row takes max(s, G, (s + 17 + G) / R) cycles, where s = L(m) · 16 / F is
  its stream time, G = min(⌈m/8⌉, 16) its drain and R = ⌊16 / min(L, 8)⌋ the rows in flight. Rows of at
  least 256 elements at 16 lanes and rows of at least 512 at 8 lanes are stream-bound; rows of 128
  elements wait for the ring: 21 cycles per row at 16 lanes and 25 at 8.
- Post unit: FM, FA, a pipelined exact SQ and an exact RC, at one row per 4 cycles or better. Results are
  packed 16 per line into the row vector, or written to s[sd].
- Result latency: 108 cycles after the last drain (106 to an s register), plus 2 per lane-pipe stage, the
  same for every kind and post-op: the post pipeline (FM, FA, SQ, RC) is fixed and its stages are
  bypassed.

### 7.6 Angle unit and ROPE engine

- Per angle: φ_i from the recurrence θ_{i+1} = FM(θ_i, TS) (one FM chain per token; chains of up to 4
  tokens interleave to hide the FM latency), the optional FD by ff_i (two iterative `tsr_fp_div`
  instances, at most 8 cycles per angle) and FM by freq_scale; or FM(Pf, A_i) in the A-table form. Then
  the integer reduction: a 24 × 64 product (4 DSP) and a 63 × 63 product (multi-cycle, 3–4 DSP). Then the
  two polynomials on a time-multiplexed FM/FA pair, and c, s = FM(M, ·).
- Target: at most 4 cycles per angle (8 with freq factors, for example Gemma 4 global layers: about 2 K
  cycles per token and layer, estimate).
- Angle buffer: 512 entries × 64 bits (1 RAMB36), double-buffered per token.
- Rotation: in the front lanes. NEOX reads line pairs at j and j + n_dims/32 and writes one output line per
  cycle.
- Optional reuse: a ROPE whose (p, table contents, n_dims, M, mode) match the previous token's may reuse
  the angles, for example K after Q. The VU re-reads pos and table and compares the values before reusing.

### 7.7 SPM access

- Client: client 4 of CC's 2R2W even/odd SPM (CC §10.1–§10.2).
- Pair reads: in each read cycle the VU fetches two consecutive lines of one operand (lines L and L + 1
  always fall in different banks). With two streaming sources it alternates an a-pair cycle and a b-pair
  cycle, so each source gets one line per cycle whatever the parity of its start. There is no same-parity
  stall and no placement rule.
- Costs: a row with an odd line count ends with a single-line read (at most 1 slot per row). The c row
  vector takes one read slot per 16 rows. Single-source ops use half the read slots, which leaves the
  rest to other clients.
- Writes: one line per cycle to the bank of the line.
- Flow control: reads are issued only against free output-FIFO credits. A stalled write therefore never
  needs a clock enable in the lanes; the pipeline simply stops issuing reads.
- Buffers: the output FIFO holds 64 lines (8 RAMB36, or LUTRAM); the input reorder buffers hold 2 lines
  per operand.
- Contention: competing clients (DMA, QACT, gather, bridge, AT) get round-robin slots, which slows the VU
  (`VU_SPM_WAIT`).

### 7.8 Sequencer, accept-ahead and hazards

- Accept-ahead. The VU raises `cmd_ready` for the next op while one earlier op is still in its pipeline
  (S − D ≤ 2). Its done pulses stay in accept order, and it never reports a run-time error. This needs
  CC's I3 relaxed for unit 3, as for the MVM (§9).
- Hazards. Before the next op issues its first read, the VU compares its source extents (a, b, c, pos,
  table) and its destination extent against the destination extents of held ops (RAW, WAW), and its s
  reads against held s writes. Any overlap makes it wait until those ops are done. A wait is counted in
  `VU_HAZARD`.
- Gain. Independent ops overlap their fill and drain. Dependent chains pay the latency.

### 7.9 Throughput

Cycles of an op that waits for the op before it, from the done of that op to its own. F, E and B are the
front, EXP and back lanes, P the lane-pipe stages and X = 2 P (a 16 / 16 / 8 lane build with X = 0; the
release image 8 / 8 / 4 with P = 2 and X = 4).

| Op | Cost |
|---|---|
| VOP, front class | rows × L(m) × 16/F + ⌈rows/16⌉ (c vector) + 59 + X (+1 with a and b) |
| VOP, EXP, EXPSUB, EXPSUBC | rows × L(m) × 16/E + ⌈rows/16⌉ + 59 + X (+1 with a and b) |
| VOP, half rate (SILU, SIGMOID, GELU, TANH, RECIP, SWIGLU, GEGLU) | rows × L(m) × 16/B + ⌈rows/16⌉ + 95 + X (+1 with a and b) |
| RED | the partial ring of §7.5: the last drain ends after the rows' recurrence, then + 108 + X (106 + X to an s register) |
| ROPE | (T − 1) × max(A, R) + A + R + D: A = angles per token (5 per angle of n_dims/2; 121/16 with freq factors; 17/16 with the A table), R = n_heads × head_dim/16 × 16/F, D = 127 + X (136 + X with freq factors, 134 + X with the table) |
| ASMAX | R × n_lines × 16/F + 60 + X |
| AMERGE (INIT; a merging op + 73) | R × D/16 × (244 × 16/F + 121 + 4 P) / 64 + 161 + 5 × ⌈R/16⌉ + 11 × (16/F − 1) + 6 P, C = 4 |
| ANORM | R × D/16 × 16/F + 149 + 3 × ⌈R/16⌉ + 2 × (16/F − 1) + 4 P |
| SOP | 53 + 5X/4 (LDS 58 + 5X/4, SQRT and RSQRT 68, DIV 25) |
| AMASK, DQROW, TOPK | estimates: AMASK as ASMAX, DQROW rows × L(K) × 16/F + 30, TOPK rows × (L(m) × 16/F + k) + 20 |

Independent ops do better than these figures: the unit overlaps their fill and drain (§7.8). An independent
op costs its stream time plus about 3 cycles when that is longer than half its accept-to-done latency (about
50), else about half that latency. The simulator's cycle model charges the dependent figure for every op
(SW-80); `TSR_SIM_VU_TIMING` names the lane configuration (`u200`, `k4` or `F,E,B,P`). The figures hold to
within 2 % for the op shapes that firmware kinds issue; a ROPE whose rotation is nearly as long as its
angles takes up to 7 % more per extra token in the RTL than in the model, and decode (T = 1) is exact.

### 7.10 Area, placement and timing

Resource use of `tsr_vu` in the release configuration (8 front, 8 EXP and 4 back lanes, `VU_LANE_PIPE` 2,
`HAS_GDN` 1; out of context, place and route at 4.0 ns): about 63.7 K LUT, 73.6 K FF, 289 DSP, 34 URAM, 16
RAMB36 and 13 RAMB18. Without `HAS_GDN` it is about 58.7 K LUT, 68.8 K FF and 249 DSP. A 16 / 16 / 8 lane
build is larger by the lane arithmetic. The resource use depends on the lane knobs and the VU's share of the
placement spread; CC §14.1 carries the budget.

Shrink knobs, as elaboration parameters (results unchanged):

- `VU_FRONT_LANES` 8: halves the physical front arithmetic and processes each architectural line in two
  parts. Each physical lane l retains separate partial histories for logical lanes l and l + 8, including
  their line-index-mod-8 phases; the partial address carries the context through every pipeline stage. Two
  drain read ports reconstruct the original 16 logical words and leave the reduction tree and order
  unchanged. NEOX uses four subslots with independent output-line and half-line indices. Architectural
  lines remain 512 bits; the ISA and the numerics are unchanged.
- `VU_EXP_LANES` 8: EXP and EXPSUB run at half rate, saving about 4.4 K LUT and 88 DSP.
- `VU_BACK_LANES` 4: SILU, GELU and the others run at quarter rate, saving about 4.3 K LUT and 36 DSP.

Placement. The SPM is in SLR1. In the release image the VU has its own Pblock in SLR2, and its physical lane
banks sit behind registered pipes (CC §14.2). With 8 front lanes each bank holds eight logical partial
histories and computes two pairs of the first reduction level locally, so its output pipe carries 128 bits of
ordinary results and 64 bits of reduction results per cycle; the 16-front-lane configuration returns its raw
partials. Both add `2 × VU_LANE_PIPE` cycles of transport per round trip. The injector orders front and back
exits so that only one is consumed per cycle, and `tsr_vu_lane_share` selects that exit beside the lanes. With
8 front lanes the input pipe carries selected A and B words and scalar C1 and C2 values (a 366-bit packet);
each bank expands elementwise C values and RoPE angles beside its lanes and stores two contexts of four C1 and
four C2 words, and angle-half release is delayed by `VU_LANE_PIPE`, so reuse cannot overwrite a final bank
read.

Timing rules: at most 6 LUT levels per stage; DSPs fully registered; no global clock enable (credits
replace back-pressure); 512-bit buses registered once at the bank boundary and once per lane group.

Optional DSP use in the fp units: `tsr_fp_add_dsp` (`rtl/core/tsr_fp_add_dsp.sv`) keeps the algorithm of
`tsr_fp_add` and does the far-path right shift of the small significand as `m_small * 2^k` in one DSP48E2
(`NORM_DSP` does the near-path left shift the same way). It has the same ports and results as `tsr_fp_add` and is
bit-exact against the reference. The define `TSR_FP_ADD_DSP` makes every `tsr_fp_add` an instance of it; no
filelist sets it.

### 7.11 Local attention and MVM epilogue consumers

The optional local-intermediate configuration (AT §5.5a; not in the release image) reuses the eight front/EXP
and four back lanes as two banks of four front/EXP and two back lanes; a bank serves the two owners in its SLR
and adds no EXP or TANH datapath. Normal VU work drains before local ownership changes, and local work reserves
return capacity before injection. A local ASMAX sequencer reads and writes its owner's score and P bank and
receives the global row maximum; QK sends scalar maxima through a balanced registered tree with bounded
credits, and the local scalar reduction preserves the MAXT special-value and tie behavior. A completed dense
MVM job can enter a 16-slot ordered local epilogue queue per pair before its final gather transfer: SCALE is an
FP32 multiply, SOFTCAP an FP32 multiply, the existing TANH and an FP32 multiply with the two original graph
coefficients, keeping all three rounding boundaries. Raw MVM `w0[13]` selects local SCALE (`w6` destination
stride, `w7` FP32 scale) and `w0[14]` selects SOFTCAP (`w6` first scale, `w7` final scale, implicit dense SPM
destination stride); the bits are mutually exclusive, incompatible with ACC, and require capability bits 13
and 14. Tail masks stay attached to the job and inactive output words stay zero.

### 7.12 Fused GDN state update (GDN, opcode 0x39; `HAS_GDN`)

The op is integrated in `tsr_vu` behind `HAS_GDN` (`CAPS` bit 8). With `HAS_GDN` 0 there is no decode, no
engine and no lane mux, and every other op runs cycle for cycle as without the option. The C reference
`sw/sim/tsr_ref_gdn.c`, the simulator's VU unit (`sw/sim/tsr_vu.c`), the engine `rtl/core/tsr_vu_gdn*.sv`
and the VU agree bit for bit (§7.12.10). The release image has `HAS_GDN` 1. The card test is the `selftest`
kernel's id 11 (`tsr-selftest --ids 11`). Decisions VU-28 to VU-37 are in §11.

The element update runs on the VU's front lanes, which are the structure it needs: FM_A and FM_B in
parallel into FA (VU-30, VU-34). The engine has no update array of its own. What it holds is what the lanes
do not have: the DSP dot array, the converter, the setup engine, the scalar chain and the row and table
buffers. The engine is inside `tsr_vu`, so it lands in the VU's Pblock beside the lanes. Nothing goes to
SLR1 (VU-31).

#### 7.12.1 The op

Gated DeltaNet layers keep a recurrent state S per head, d_v × d_k f32. For one token and one head, with k
and q (d_k, L2-normalized by earlier ops), v (d_v), and the scalars α = exp(g) and β:

```
for each row i (rows are independent):
    u_i  = S_i · k
    e_i  = β (v_i − α u_i)
    S'_i = α S_i + e_i k            (written back)
    o_i  = S'_i · q
```

A head with d_k = d_v = 128 is 64 KiB. Composed from VOP and RED, the recurrence needs four passes over S
(decay, S·k, update, S'·q). The fused op reads and writes S once. The pinned llama.cpp computes the same
recurrence (`build_delta_net_autoregressive` in `src/models/delta-net-base.cpp`, and
`ggml_compute_forward_gated_delta_net` in `ggml-cpu/ops.cpp`: decay S, delta = β (v − S·k), S += k δ,
o = S·q, a row of S being S_v contiguous words dotted with k and q). The CPU kernel scales o by 1/√S_v; the
graph version scales q. This op takes q as given.

Throughput: 8 state elements per cycle (one front-lane beat) at 8 front lanes, 16 with 16 front lanes. The
cost per head is (rows + 1) × m / 8 cycles when heads follow each other (§7.12.8).

#### 7.12.2 Operands and where they live

| Operand | Form | Where |
|---|---|---|
| S | d_v rows of m = d_k f32, dense (row stride m / 16 lines), updated in place or to another tile | SPM tile, loaded by `LD` from the state tensor in DRAM, stored back by `ST` |
| k, q | m f32 each | SPM vectors (earlier VU ops L2-normalize them) |
| v | d_v f32 | SPM vector |
| o | d_v f32 | SPM vector, tail rule of §4.2 |
| α, β | f32 | s registers (`SOP.EXP` gives α = exp(g)) |

The state tile comes through the SPM and the existing DMA, not through a direct stream (VU-31).

- Traffic per element at 8 elements per cycle: LD writes 4 B, the op reads 4 B and writes 4 B, ST reads 4 B.
  The SPM carries 64 B per cycle each way, half of the 128 B per cycle each way of the 2R + 2W SPM (CC §10).
  The GDN phase of a layer has no competing MVM work (the next layer waits for its output).
- The op reads the state tile of a head once, one SPM line per cycle, and writes S' one line per two cycles
  (the lanes' rate). Its reads go through the VU's own reader (`tsr_vu_rd`), its writes through the VU's
  writer (`tsr_vu_wr`).
- A head tile is 64 KiB. Three tiles (LD of the next head, the op, ST of the previous one) are 192 KiB of
  the 4 MiB SPM.
- A direct DDR to unit to DDR stream would need a new DRAM path, an arbiter against the weight streams and
  an SLR crossing, and it would not be a tile op that firmware can order with the checker, queues and
  hazards.

#### 7.12.3 Numerics

All f32 operations are the VU primitives (§6.1): DAZ, RNE, FTZ, canonical NaN. The dot products are not f32
sums. They are exact sums of integer products in a block frame, converted to f32 once.

With `ab = FM(α, β)`, `kp_j = FM(ab, k_j)` and `bv_i = FM(β, v_i)`:

```
TK = table(kp)         TQ = table(q)            head tables, 24-bit entries
κ  = rowdot(k, TQ)     u'_i = rowdot(S_i, TK)   w_i = rowdot(S_i, TQ)
e_i   = FS(bv_i, u'_i)
S'_ij = FA(FM(α, S_ij), FM(e_i, k_j))
o_i   = FA(FM(α, w_i), FM(e_i, κ))
```

u' = αβ (S_i · k) and w_i = S_i · q, so e_i = β v_i − αβ u_i and o_i = α (S_i · q) + e_i (k · q) = S'_i · q.
The output needs no second pass over S'. It uses the pseudo row below to get κ = k · q.

- table(x) (m ≤ 256 elements). DAZ. E_max is the largest biased exponent among the nonzero elements (0 if
  none). Entry j is sign × ⌊(M_j + 2^(sh−1)) / 2^sh⌋ with M_j the 24-bit significand, sh = E_max − E_j
  (sh = 0 keeps M_j; entries are 0 for zero elements and for sh ≥ 25). Entries are below 2^24 in magnitude;
  entry × 2^(E_max − 150) is the element. The table is the same for every row of the head.
- rowdot(row, T). E_s is the largest biased exponent of the row's nonzero elements. Element j becomes
  A_j = sign × ((M_j << 2) >> (E_s − E_j)), a 26-bit window below the row's frame, truncated (0 for a shift
  of 26 or more). I = Σ_j A_j T_j is computed exactly (|I| < 2^58). The result is the f32 nearest to
  I × 2^(E_s + E_max,T − 302): RNE, overflow to ±Inf, a result below 2^-126 − 2^-150 flushed to a signed
  zero (the rule of FM), I = 0 gives +0.
- Order does not matter. The sum is an integer sum, so lane count, beat order and accumulation order change
  nothing. There is one rounding per dot product.
- Inf and NaN. Any Inf or NaN among a row's elements, or among a table's elements, makes the rowdot result
  the canonical NaN. A bad state row poisons that row's e, S' and o. A bad k or q poisons the head.
- κ from a pseudo row. The op takes κ = rowdot(k, TQ), the row dot product of the vector k itself with TQ.
  The hardware streams k through the dot array as an extra first row; its u' is dropped.
- The update on the lanes is bit-exact with the reference. The lane micro-op U_ROT computes
  `FA(FM(a, c1), FM(b, c2))`. With a = S_ij, c1 = α, b = k_j, c2 = e_i it is `FA(FM(S_ij, α), FM(k_j, e_i))`,
  and `FM` is commutative, so it equals `FA(FM(α, S_ij), FM(e_i, k_j))` of the reference for all inputs, NaN
  and Inf included.

The C reference is `tsr_ref_gdn` (`sw/sim/tsr_ref_gdn.c`, header `tsr_ref_gdn.h`) with the pieces
`tsr_ref_gdn_table`, `tsr_ref_gdn_rowdot`, `tsr_ref_gdn_fx2f`, `tsr_ref_gdn_e`, `tsr_ref_gdn_o` and
`tsr_ref_gdn_upd_elem`. Row bound: with a 26-bit window and 24-bit entries, 256 terms stay below 2^58, so
d_k ≤ 256 (the 48-bit DSP accumulators are exact for the two 13-bit pieces of an entry, §7.12.5).

Differences from the CPU kernel: it decays S in place and then dots (this design folds α β into k), it sums
in whatever order `ggml_vec_dot_f32` uses, and its `ggml_vec_mad_f32` may contract to an FMA. All of these
are rounding differences of a few ulp against float64 (§7.12.6).

#### 7.12.4 Encoding

`GDN` = 0x39 (the first of the opcodes reserved for VU extensions, EX §7.2), unit 3, `CAPS` bit 8
(`HAS_GDN`; an op on a device without the bit completes ILLEGAL_OP on SYNC like every opcode whose CAPS bit
is 0).

| Field | Bits | Meaning |
|---|---|---|
| opcode | w0[7:0] | 0x39 |
| | w0[15:8] | 0 |
| sa | x16[3:0] | s register holding α |
| sb | x16[7:4] | s register holding β |
| | x16[15:8] | 0 |
| d | w1[23:0] | S' tile, line of row 0 (may equal a: in place) |
| a | w2[23:0] | S tile, line of row 0 |
| k | w3[23:0] | k vector line (m words) |
| q | w4[23:0] | q vector line (m words) |
| v | w5[23:0] | v vector line (rows words) |
| o | w6[23:0] | o vector line (rows words, tail rule of §4.2) |
| m | w7[15:0] | d_k: a multiple of 16, 64 ≤ m ≤ 256 |
| rows | w7[31:16] | d_v, 1 ≤ rows ≤ 1,024 |

Rows of the tile are dense, stride m / 16 lines, for a and d. One op is one head. Checks, in this order (the
simulator's `chk_gdn` and the RTL class in `tsr_vu_chk` give the same code, ERR_INFO and extents):

- `ILLEGAL_OP`: any reserved bit (`TSR_GDN_RSV_W0` … `W7` of the generated table).
- `OPERAND` (ERR_INFO = the raw field): m not a multiple of 16 or outside [64, 256] (m); rows 0 or above
  1,024 (rows); d ≠ a with overlapping extents of d and a, E(rows, m / 16, m / 16) lines each (d); o
  overlapping the extents of d or a (o).
- `SPM_RANGE` (ERR_INFO = the start line), the six range operands in field order d, a, k, q, v, o: d and a
  E(rows, m / 16, m / 16) lines, k and q m / 16 lines, v and o ⌈rows / 16⌉ lines.

The lower limit of m is 64 for every build (4 beats of the 16-lane build). Vectors and the tile may overlap
d or o: the op reads k, q and v completely before it writes a word. The extent list of a passing op has four
SPM entries: d W, a R, k R, o W (CC §8.2: four slots; every written range is listed, the race detector reads
them), and the s registers sa and sb.

#### 7.12.5 Microarchitecture

The engine `tsr_vu_gdn` sits behind the injector: the VU's generator makes slots for the op (a CFG slot, the lines
of k, q and v, the m / 16 lines of k again as the pseudo row, and one slot per line of S), their SPM reads go
through `tsr_vu_rd` like every operand, and the injector hands each slot to the engine when it can accept it
(`cfg_ok`, `vec_ok`, `row_ok`). The slots count the op's writes (one per S' line, one more per o line), so the
op's done waits for the lanes' results.

- Setup engine (`tsr_vu_gdn_coef`): one word per cycle in the order k, q, v; one shared FM computes ab, kp_j and
  bv_i, and the key and the two tables are aligned to their frames (round half up) and written to RAM; about
  4 m + rows + 45 cycles per head. Two banks hold the coefficients of two heads, so the setup of head h + 1 runs
  while the rows of head h flow.
- Ingest: a row line (16 words) per cycle goes to both row buffer copies, and a pipelined maximum tree over the
  exponents gives the row's E_s (E_s = 255 is the row's bad flag). The ingest runs at twice the rate of the dot
  array and the update, so the rows of the next head wait in their slots while its setup runs (VU-36). 64 row
  slots (`GDN_NSLOT`) hold rows from the first ingest line until the update is launched; the first line of a
  row takes a credit, which returns when the row's update is launched, and `row_ok` asks for two free slots.
- Dot array (`tsr_vu_gdn_dot`): lane l holds elements j ≡ l (mod 8) of every row (RAM B1) and of both tables
  (RAM T). Per beat a lane aligns its word (a 26-bit right shift), applies the sign with the DSP pre-adder and
  multiplies by the four table pieces (the high 12-bit signed and low 13-bit pieces of the K' entry and of the Q
  entry). The four products go into four DSP48E2 cascades (`tsr_vu_gdn_mac`); because the lanes start one
  after another, the cascade adds the 8 lanes' products of one beat with no fabric adder and no skew
  registers, and one accumulator slice per chain adds the beats of the row. Everything is exact (|P| < 2^47 for
  256 elements). The converter (`tsr_vu_gdn_fx2f`, 8 stages) converts the K sum and, one cycle later, the Q
  sum.
- Scalar chain: one shared FM and one shared FA give e = FS(bv_i, u'), FM(α, w), FM(e, κ) and o. The pseudo row's
  w becomes κ of the bank. The o words of a head are collected in lines of 16 (words past `rows` are 0) and
  offered to the writer's result port (the one TOPK uses; an op GDN is alone, so TOPK is idle).
- Update on the lanes (`tsr_vu_gdn_upd` and the mux in `tsr_vu`): a row whose e is known and for which the writer
  has room is launched. The launch takes its ⌈m / 16⌉ lines from the output FIFO of the writer and reads the
  row again from RAM B2 together with the key from RAM KF, a beat per cycle, with `uop = U_ROT`, a = S, b = k,
  c1 = α, c2 = e, `neg` 0. The select is a registered level of the engine, replicated per lane; no other lane
  input has two sources in any cycle, because an op GDN is held back from every other op (VU-35). The lanes
  have no back pressure and none is needed: the credits were taken at the launch. A launch waits for 16 free
  lines in the 64-line output FIFO (`oc_ok`).
- RAM styles. Every buffer is a RAM with `ram_style = "ultra"` (`tsr_vu_gdn_ram`, parameter `GDN_STYLE`; "block"
  maps the same RAMs to RAMB36, about 43 of them): per lane the row buffer B1 (dot pass), its copy B2 (update
  pass), the table RAM T and the key RAM KF, plus the scratch RAM and the bv RAM. Two copies of the row are
  needed because three accesses per cycle (ingest write, dot read, update read) exceed the two ports of a URAM.
- Parameters: `HAS_GDN`, `GDN_NSLOT` (64; a power of two ≥ 4), `GDN_STYLE`; `LANES` of the engine is
  `FRONT_LANES` (8 or 16); latencies `RD_LAT` 3, `FM_LAT` 4 and `FA_LAT` 5 follow the lanes' primitives. The dot
  array needs no DSP48E2 primitive: Vivado maps the written registers, pre-adder, multiplier and cascade adder
  into the slices with `PCIN` cascades, provided that all lanes of a chain are in one module.

#### 7.12.6 The dot products (VU-29)

Each row needs two d_k-long dot products (S_i · k and S_i · q), and the head needs a third (k · q). Scheme (A)
sums in f32 adder trees (per dot product 8 multipliers and 9 adders; two engines need 16 multipliers and 18
adders). Scheme (B) is the exact integer sum in DSP48E2 accumulators: align each row element to the row's
largest exponent (a 26-bit window; the table is aligned once per head), multiply by the table entry, add in the
DSP cascade, convert once. With a 26-bit row window and 24-bit table entries (B) matches the f32 tree and
numpy's matmul against a float64 reference and is slightly better on wide exponent spreads (one rounding
instead of seven); narrower windows lose accuracy quickly (a 24-bit window costs a factor 1.3 to 4, 20-bit
entries 7 to 12, 16-bit entries more than 100), which fixes the widths 26 / 24, the DSP use (two 13-bit pieces
per entry, 4 DSP per lane for the two tables) and the accumulator bound of §7.12.3. Over 32,768 gated
recurrence steps of a 128 × 128 head, (A), (B) and a float32 textbook recurrence agree to a few percent, so the
dot summation is not what limits accuracy and (B) is chosen on area (about 17 % of the CLB and 14 % of the LUT
of two (A) engines; `tb/gdn/gdn_accuracy.py`). Decision VU-29: (B).

#### 7.12.7 The element update runs on the front lanes (VU-30, VU-34)

The element update is `FA(FM(α, S), FM(e, k))` on `tsr_fp_mul` and `tsr_fp_add`. A front lane has FM_A and FM_B
in parallel into FA for ROPE and AMERGE (U_ROT, §7.2), and the GDN update is U_ROT with the operands of
§7.12.5, so the engine has no update array: the front lanes take one beat of 8 elements per cycle, the rate of
the dot array. A block-frame update in the DSP is 1.3 to 30 times less accurate than the f32 update and needs a
second set of row shifters and a normalizer per lane, so the element update remains f32 on the VU's primitives
(`TSR_FP_ADD_DSP` of §7.10 applies to it unchanged).

#### 7.12.8 Throughput and latency

- Rate: one beat of 8 words per cycle in the dot array and the update, whatever the shape (m = 64, 128, 256).
  Heads of one shape follow each other with no gap in the launches: 128 × 128 heads take
  (rows + 1) × m / 8 = 2,064 cycles each (the pseudo row is the +1); with 16 front lanes half.
- A head that depends on the previous one (the same tile updated in place) at the release lanes (8 front lanes
  behind two lane-pipe stages) takes T = (rows + 1) · m / 8 + ⌈19 rows / 16⌉ + 9 m / 2 + 122 cycles, exact to 2
  cycles: 2,914 for 128 × 128. At 16 front lanes the same form with ⌈21 rows / 16⌉ and (rows + 1) · m / 16 is
  within 1.5 %. The simulator's `cyc_gdn` is this formula (SW-78).
- Setup of a head: about 4 m + rows + 45 cycles, overlapped with the rows of the previous head (up to 63 rows
  wait in the slots, more than the setup, VU-36). A head after an idle engine pays the setup once, then 66
  cycles from the tables to the first launch.
- Latency from the launch of a row to its S' line in the SPM: RD_LAT + 1 for the operands, then the lane latency
  (`TSR_VU_LF` = 42 plus the lane-pipe stages) and the writer. A launch takes m / 16 of the 64 output-FIFO
  lines, and the lanes' results drain at one line per two cycles, so at most about 27 lines are held at once.

#### 7.12.9 Area, placement and timing

`tsr_vu` in the release configuration (`FRONT_LANES` 8, `EXP_LANES` 8, `BACK_LANES` 4, `LANE_PIPE` 2,
`SPM_RD_LAT` 4, `NUM_CH` 4; out of context, `tb/vu/viv/vu_ooc.sh`):

| `tsr_vu` | LUT | FF | DSP | URAM | RAMB36 / 18 |
|---|---|---|---|---|---|
| `HAS_GDN` 0 | 58.7 K | 68.8 K | 249 | 0 | 16 / 13 |
| `HAS_GDN` 1 | 63.7 K | 73.6 K | 289 | 34 | 16 / 13 |
| difference | +5.0 K | +4.8 K | +40 | +34 | 0 |

The engine `tsr_vu_gdn` is 5.4 K LUT, 4.7 K FF, 40 DSP and 34 URAM (the dot array with its 4 chains and the B1
and T RAMs 1.6 K LUT, 36 DSP, 16 URAM; the operand reader with RAM B2 and KF 0.7 K LUT, 16 URAM; the converter
0.6 K LUT; the setup engine 0.6 K LUT, 2 DSP, 1 URAM; the scalar chain 0.8 K LUT, 2 DSP, 1 URAM; ingest,
exponent tree, slots, launch control and o-line assembler 1.1 K LUT). With `HAS_GDN` 0 every other op runs cycle
for cycle as without the option. The engine's paths meet 4.0 ns out of context with at least 0.46 ns of slack.
The VU Pblock (`CLOCKREGION_X0Y10:X2Y14` in SLR2, `hw/u200/k4_major_blocks.tcl`) holds URAM (160), RAMB36 (420)
and DSP (1,320), so the engine lands inside it with no extra constraint; URAM is chosen over `GDN_STYLE` "block"
(VU-33) because it leaves the RAMB36 for the VU's own buffers.

#### 7.12.10 Verification

- Reference (`sw/sim/tsr_ref_gdn.c`, ctests `ref_gdn`, `ref_gdn_np`): the conversion against exact long-double
  arithmetic rounded by the hardware conversion (every length and exponent, the RNE ties, the carry, the overflow
  edge, the 2^-126 - 2^-150 edge), exact integer dot products, the window edge, the tables, the head against the
  float64 recurrence, α = 0, β = 0, zero state, in place, tails, poison by Inf and NaN. `tests/ref_gdn_np.py` is an
  independent numpy model written from §7.12.3 and matches the C bit for bit; mutants of the C (truncation, the
  window width and edge, the tie rule, the FTZ edge, operand order) are killed.
- Simulator (`sw/sim/tests/test_sim_vu.c`: test_gdn, test_gdn_checks, test_gdn_misc): the op against
  `tsr_ref_gdn` on SPM images, every check code with ERR_INFO and the check order, the extents, CAPS gating, the
  cycle model.
- Engine (`tb/gdn/test_gdn.py`, Verilator and cocotb, bit-exact against `libtsr_ref.so`): `gdn_fx2f`, `gdn_dot`
  (including the saturating case that fills the 48-bit accumulators at m = 256), `gdn_upd`, `gdn_coef` and
  `gdn_unit` with 8 and 16 lanes, other latencies, 8 slots and block RAM; mutants of the RTL
  (`test_gdn_mutants.py`, opt-in) are killed.
- VU (`tb/vu/test_vu_unit.py`: `vu_gdn`, `vu_gdn_mixed` with RAW against VOP and RED on S' and o, the same tile in
  two consecutive heads, the s registers of the heads; `tb/vu/test_vu.py` `vu_chk_gdn`), the core
  (`tb/core/test_core.py` `test_core_k4gdn`: two GDN heads from the firmware through the issue path, with
  DRAM, `CAPS` 0x121 and the final CSR state equal to the simulator's) and the card (`tsr-selftest --ids 11`:
  four heads of 20 × 128 checked against `tsr_ref_gdn` and the shadow simulator; it runs only with CAPS bit 8).

Design limits. `d_k` is limited to 256 (the accumulator bound) and `d_v` to the engine's `R_MAX` (1,024); the
pseudo row means that k is read twice. GDN holds the VU: while it runs no other VU op overlaps it (VU-35), so
the firmware orders a layer's work so that the heads form one run. The URAM read-first rule holds: the ingest
never writes line g of a freed slot before one cycle after the update read of that line.

### 7.13 The simulator's cycle model

The simulator (`sw/sim/tsr_vu.c`, `tsr_at.c`, SW-80) charges every VU op the dependent-op figure of §7.9,
and `TSR_SIM_VU_TIMING` names the lane configuration. The RTL unit's own timing benches
(`tb/vu/test_vu_timing.py`: `vu_sweep`, `vu_sweep_words`, `vu_stream`) time every op shape of §7.9 repeated
four times, dependent and independent, at four lane configurations (16 / 16 / 8 and 8 / 8 / 4, each with
and without two lane-pipe stages), and the model reproduces the dependent figure of every shape. The
independent SOP.LDS and SOP.MOVI ops overlap in the RTL, so the model stays slightly above it on a Gated
DeltaNet decode token.

Compared with a 16-lane VU, the release VU differs in three ways: an 8-lane front takes 2 cycles per line and
the half-rate functions 4; every dependent op pays a latency of 60 to 100 cycles; a reduction of rows of 128
elements is bound by the partial ring (25 cycles per row, not 16).

---

## 8. Usage (informative; SW builds the kernels)

### 8.1 Norms

- RMSNorm of one vector: `RED.SUMSQ TO_S sd=s1 post=RMS sk=s2 se=s3` (s2 = 1/n and s3 = eps, set once by
  `SOP.MOVI`), then `VOP.MULCB d = (x × s1) × w` (cmode 1). That is 2 ops, against 7 in the spec recipe,
  with identical bits.
- Gemma 4 per-head Q, K and V norms: one `RED.SUMSQ` over all q, k and v head rows (12B SWA layers:
  rows = 16 + 8 + 8, m = 256; 12B global layers: 16 + 1 + 1, m = 512; post RMS) gives a row vector. Then
  `VOP.MULCB` on the q rows (c = row vector, b = q_norm weight with stride 0), `MULCB` on the k rows, and
  `MULC` on the v rows: 4 ops.
- Prefill: rows = T tokens; c is the per-token vector.

### 8.2 FFN and residuals

- Activation: `VOP.SWIGLU` (Qwen) or `GEGLU16` (Gemma): one op.
- Residuals: Qwen an `MVM` with ACC; Gemma a post-norm `MULCB`, then `ADD`, with `out_scale` fused as `ADDMULC
  (cur + attn_out) × s`. A QKV bias is one `VOP.ADD` over the adjacent q, k and v outputs (stride 0 for T > 1).

### 8.3 RoPE

One `ROPE` over q and k when they are adjacent, with T tokens.

### 8.4 Attention softmax

The FA kernel's sequence is AT §6.6 (normative): per chunk ATTQK (or ATTQK + AMASK), ASMAX, ATTPV and AMERGE,
and one ANORM per pass. The generic VOP / RED composition of an online softmax (RED.MAX, VOP.MAX, EXPSUB,
EXPSUBC, RED.SUM, MUL, ADD) stays available to other kernels; EXPSUB's -inf rule keeps a fully masked first
block finite (§6.5).

### 8.5 Output head

The Gemma softcap is `VOP.MULC` (1/cap), `VOP.TANH`, `VOP.MULC` (cap) over the vocabulary; greedy sampling is
`TOPK` k = 1 where the image has TOPK, otherwise the control core selects the maximum.

---

## 9. Interfaces with other areas

| Area | VU provides | VU assumes |
|---|---|---|
| EX | Encodings and checks for 0x12, 0x30–0x33, 0x38, 0x39, 0x40 (0x13, 0x3A–0x3F, 0x41–0x47 stay reserved); the `x16` usage of §5; `OPERAND` conditions and `ERR_INFO` values (§5.9); `CAPS` bits 0, 3, 6, 7, 8; `CFG3[15:0]`; s bank reset +0.0, unchanged by `SOFT_RESET`; no run-time errors; execution of AT's 0x34–0x37 | Opcode blocks, `x16`, code 13 and the check order of EX §7–§8; PERF slots X + 13 … X + 15; ACTIVE and QHALT gating of `VU_CYCLES`; `SREG` at 0x9C0 |
| AT | Execution of ASMAX, AMASK, AMERGE and ANORM (§7.4, §7.9); EXPK (EXP(-inf) = +0, EXP(±0) = 1, no -0 or subnormal results); RC = `tsr_frecip`; per-head RMSNorm for Gemma's QK norm (§8.1) | AT owns scores, P·V, the KV layout and quantization, GQA mapping, the kq scale, the semantics and checks of its vector ops (AT §5.4–§5.8) and the FA sequence (AT §6.6) |
| CC | Unit 3 dispatch with `cmd_ready` (accept-ahead) and `done` (no error output); SPM client 4; the s bank exported to `SREG` and the core window; the checker class (reserved masks, used words, conditions, four tagged extents) | Four extents per checker class (CC §8.2); I3 S - D ≤ 2 for unit 3 (CC §9.4); queue depth 16; the VU Pblock in SLR2 (CC §14) |
| DB | Debug state of §5.10; `VU_CYCLES` and three extension counters; ops complete in bounded time; hold points are op boundaries only | `DBG_U3_*` probes, `SREG` exposure, halting at the queue head (no mid-op halt, no s-register debug write) |
| SW | §6 as the text for `tsr_ref_math.c`, the numpy model and the exhaustive tests; reference names (§6.1); usage recipes (§8); b row periods greater than 1 are lowered as one op per period block | The simulator semantics, `tsr_ref_math.c` (SW §4.5 rules), `vu.hpp` builders from `sw/tables/tsr_tables.py`, traces `traces/vu`, and test-backend-ops, numcheck and klcheck. SW computes TS, M, freq_scale and (YaRN only) A_i on the host, and maps llama.cpp's GELU and GEGLU to GELU16 and GEGLU16 |

---

## 10. Verification

The simulator is the oracle; every VU op has a C reference, and the RTL must match it bit for bit.

1. References: `tsr_ref_math.c` (EXPK, the composites including GELU16, SIN and COS), `tsr_ref_vu.c` (VOP,
   reductions, ROPE in both angle forms, TOPK, SOP, DQROW, VMASK) and `tsr_ref_at.c` (AT's vector ops). An
   independent numpy model written from this text alone (SW-23) is compared bit for bit over all 2^32 inputs for
   each unary function (EXPK × 4 κ, SIG, SILU, GELU, TANH, SIN, COS; opt-in `slow`, default 2^24 strided points
   plus specials). Properties: EXPK monotonic in v for K > 0, EXPK(v, N2) = EXPK(2v, N1), EXPK never -0 or
   subnormal, SIN and COS within [-1, 1]. GELU16: all 65,536 fp16 inputs against the pinned
   `ggml_table_gelu_f16` (`ref_math_gelu16`, SW §10.1), the clamps at ±10 and the fp16 subnormal inputs. ROPE
   recurrence: angles bit-identical to a C transcription of `ggml_rope_cache_init` (ext_factor = 0) for the
   target models' bases and n_dims at positions 0 … 2^17.
2. Accuracy against exact math: exhaustive double and mpfr comparisons check the table of §6.9. Gate: each
   maximum is at most 2 times its model figure, otherwise the coefficients are reviewed.
3. Simulator (`test_sim_vu.c`, `ex_vop`, `ex_red`, `ex_rope`, `ex_dqrow`, `ex_sop`, `ex_topk` and AT's `ex_asmax`,
   `ex_amask`, `ex_amerge`, `ex_anorm`): every function and kind; every `ILLEGAL_OP`, `OPERAND` and
   `SPM_RANGE` case with `ERR_INFO`; m = 0 and rows = 0; in-place aliasing; tails; NaN, ±inf, ±0, subnormals and
   the FTZ boundary vectors of [spec](../spec.md) §21; random op streams against a second, straightforward
   Python implementation of §4–§6. The race detector gets the access sets of §5.9 and the aliasing rule of §4.5.
4. RTL unit benches (`tb/vu`, cocotb and Verilator, `common.tsr_ref` via ctypes): `tsr_vu_exp` and `tsr_vu_rcp`
   exhaustive on a Verilator C++ bench (opt-in; RC over all 2^23 mantissas times exponent edges in the default
   run); the angle unit (10^6 random inputs plus specials); the reduction unit (random rows, m in 0 … 2048,
   skip-rule cases including subtrees that sum to -0, for example 0x80C00000 + 0x00800000 at m = 8k + r, §6.2);
   the VU top with an SPM model (random streams of dependent and independent ops compared with the simulator
   on SPM bytes, s registers, completion order and error codes; random `ready` on both bank ports; hazard cases
   across accept-ahead; cycle assertions against §7.9); out-of-context timing for `tsr_vu`.
5. Integration: kernel op-list replays (`traces/vu`) in the Verilator fast replay with the VU; the card selftest
   kernel runs a fixed VU vector set against simulator goldens.
6. Against llama.cpp: `test-backend-ops` for every node the backend sends to the VU (RMS_NORM, MUL, ADD, SCALE,
   GLU, ROPE, TANH, GET_ROWS, CPY, FLASH_ATTN_EXT with AT) with NMSE tolerances; numcheck B1 and B1b on VU nodes
   (bit-exact against `libtsr_ref` and the numpy model) and B3 (error against exact math compared with the
   CPU's) per node; klcheck with FA on (D17).

---

## 11. Decisions

| Id | Decision | Reason |
|---|---|---|
| VU-1 | Operand model: rows × m with line-aligned rows, a line stride per operand (0 repeats), a per-row scalar c from an s register or an SPM row vector, for every VU op | Every per-head and per-token pattern becomes one op; line alignment keeps the SPM side one line per operand per cycle |
| VU-2 | Reduction order: the spec's 128 partials and complete binary tree per row; skipping unwritten subtrees passes z(q) (−0 → +0), which is exact | Keeps the spec contract and cheap short rows; z() fixes the −0 case that FTZ creates |
| VU-3 | Segmented RED: a per-row SPM vector (or an s register for one row) with fused post-ops RCP and RMS (exact SQRT and RECIP) | Per-head norms and softmax sums in one op; RMSNorm in 2 ops instead of 7, bit-identical |
| VU-4 | EXP algorithm: fixed-point integer EXPK with f32 in and out, 2³⁸-scaled log2e constants, a 64-entry 2^(j/64) table, a cubic Taylor term | At most 0.53 ulp (model), about 600 LUT per lane, exact portable definition |
| VU-5 | Reciprocal: all reciprocals correctly rounded (= `tsr_frecip`); no RECIP_APPROX | Exact rounding is cheap with one correction product and needs no table specification |
| VU-6 | SIN and COS: exact 64-bit 2/π integer reduction, f32 Cephes polynomials, identity path below 2^-12 | Angles reach 2^18 where f32 Cody-Waite loses precision; the reduction runs once per angle |
| VU-7 | Function set: SILU, SWIGLU, SIGMOID, GELU and GEGLU (tanh as x·σ(2u)), GELU16 and GEGLU16, TANH, EXP, EXPSUB(C), RECIP, MIN, MULCB, MULCADD, ADDMULC, SETC | Covers Qwen2.5 and Gemma 4 dense with one shared EXP and RC datapath |
| VU-8 | Lane split: 16 front lanes (including EXP) at full rate, 8 back lanes for SILU, GELU, SIGMOID, TANH and RECIP at half rate | Half-rate activations cost under 0.15 % of token time; EXP stays full rate for softmax |
| VU-9 | Flow control: credit-based reads against output-FIFO space, no clock enable in the lanes, fixed latency classes | Avoids a clock enable fanning out to about 45 K FFs |
| VU-10 | SPM access: CC's pair reads (alternating a-pair and b-pair cycles), no operand parity rule, no operand caches | Two sources stream at full rate whatever their alignment |
| VU-11 | s-register access: one bank, written only by VU ops (SOP.MOVI for constants); read-only for the RISC-V and the host at SREG0-15, core CSR 0x9C0 | Writes stay ordered by queues and deps; one read location for every reader |
| VU-12 | s-register reset: +0.0 at hardware reset; kept across `SOFT_RESET`, RISC-V reset, QABORT and QCLEAR | EX's requirement; kernels initialize what they read |
| VU-13 | Attention mask: VMASK 0x38 is specified and not implemented (`ILLEGAL_OP`); llama.cpp masks use AT's mask summaries | VMASK assumes cell j holds position k0 + j, which llama.cpp's ring-shaped, reorderable caches break |
| VU-14 | TOPK: fully specified (rows, padding, total order); simulator only; an RTL option uses a running-threshold filter | The control core selects experts and samples with integer compares; keeps 4 K LUT out of the image |
| VU-15 | DQROW: opcode 0x12 as SPM to f32 on the VU after an LD of the repacked row; simulator only (`CAPS` bit6 = 0) | The VU has the fp datapath; the embedding lookup stays on the CPU |
| VU-16 | Error reporting: OPERAND (13) with the offending value for shape limits; enumeration mismatches are ILLEGAL_OP | Follows EX; the values make failures self-explaining |
| VU-17 | Field widths: 24-bit line addresses and m; 16-bit rows, strides, head counts and T; function codes and s indices in x16 | Every op fits one 256-bit instruction with headroom |
| VU-18 | Softmax ops: no generic fused online-softmax VOP; AT's vector ops (ASMAX, AMASK, AMERGE, ANORM) on the VU, two VU ops per chunk | AT fixed the dataflow; its ops fuse what the 7-op composition needed |
| VU-19 | ROPE shape and angles: T tokens with a position vector; angles by ggml-cpu's recurrence θ_{i+1} = FM(θ_i, TS), FD by freq factor, FM by freq_scale; the A-table form only for YaRN; head_dim a multiple of 16 (≤ 1024), n_dims a multiple of 32 (NEOX) or 16 (NORM) | Angles bit-identical to the CPU for ext_factor = 0 (the A-table form differs by up to 2.9·10⁻³ rad at 32 K) |
| VU-20 | Accept-ahead: up to 2 VU ops held; RAW, WAW and s-register hazards checked by extent intervals (CC's I3 ≤ 2 for unit 3) | Short independent ops overlap their fill and drain |
| VU-21 | EXPSUB(−∞, x) = +0, even when x is −∞ or NaN | Fully masked blocks stay finite without an ordering rule |
| VU-22 | MAX and MIN: the spec total order; MAX propagates NaN, MIN returns the non-NaN operand | One comparator serves MAX, MIN and TOPK, consistent with the spec |
| VU-23 | Raw operations: COPY, SETC, MOVI, LDS, STS, VMASK pass-through and the ROPE tail copy bits without fp canonicalization | COPY serves any 32-bit data; positions and indices pass unchanged |
| VU-24 | Reduction unit size: an 8-input subtree (7 FA) plus 2 scheduled FA; rows shorter than 256 elements reduce at half rate | Saves about 5.5 K LUT; the half-rate cases are short rows |
| VU-25 | Accuracy contract: functions are defined by their algorithms (bit-exact); error bounds are informative | Bit-exactness between simulator, RTL and card is the verification basis |
| VU-26 | CPU-compatible GELU: GELU16 (0x17) and GEGLU16 (0x18) clamp at ±10 and round input and result to binary16; llama.cpp GELU and GEGLU map to them; the exact GELU serves PyTorch | The pinned CPU rounds every GEGLU element to fp16 (up to 3.9·10⁻³ relative); matching its form removes that gap for about 0.6 K LUT |
| VU-27 | AT vector ops on the VU: ASMAX, AMASK, AMERGE and ANORM run on unit 3 with the front lanes, the MAX path and a multi-pass AMERGE (about 4–5 cycles per output line at C = 4) | Reuses the EXPK lanes and pair reads; matches AT's merge tail for about 2 K LUT |
| VU-28 | Fused GDN op: opcode 0x39, one head per op: state tile in the SPM, k, q and v vectors, α and β in s registers; u, e, S' and o in one pass; d_k a multiple of 16 in [64, 256], d_v in [1, 1024]; `CAPS` bit 8, `HAS_GDN` | S is read and written once; the option is built into the release image |
| VU-29 | GDN dot products: exact integer sums of aligned mantissa products in DSP48E2 cascades (26-bit row window, 24-bit table entries), one RNE conversion per dot | The accuracy of f32 trees, no f32 adder, order independent; 36 DSP, no CLB-heavy adder |
| VU-30 | GDN element update: S' = FA(FM(α, S), FM(e, k)) is the front lanes' micro-op U_ROT (FM_A and FM_B into FA); no update array of its own | The lanes already are that structure and give the bit-exact result; a block-frame update in the DSP is 1.3 to 30 times less accurate |
| VU-31 | GDN state tile and placement: through the SPM with `LD` and `ST`; the GDN-specific part (dot array, converter, setup engine, scalar chain, buffers) is inside `tsr_vu`, so in the VU's Pblock in SLR2 | Half of the SPM bandwidth at 8 elements per cycle; no new DRAM path or SLR crossing; the VU Pblock has the DSP and URAM |
| VU-32 | GDN scalar algebra: a table of αβ k, bv = β v, o = α (S · q) + e κ with κ = k · q from a pseudo row | Both dots on the old S in one pass; no second pass over S' |
| VU-33 | GDN buffers: per-lane URAM for the row (two copies), tables and key, 64 row slots; lanes started one cycle apart; DSP chain in one module | Cascade without skew registers; the VU Pblock has 160 URAM and no other block there uses any |
| VU-34 | GDN lane operands: the engine's operand beats take the lanes' inputs and the writer's descriptors through a registered select after the injector | No lane or writer change; the mux does not exist with `HAS_GDN` 0 |
| VU-35 | GDN hazards and extents: a GDN op excludes every other VU op; extents d W, a R, k R, o W | One source for the lane inputs; four slots of CC §8.2 |
| VU-36 | GDN line-wide ingest: row buffers take a 16-word line per cycle, 64 row slots | Back-to-back heads at (rows + 1) × m / 8 cycles; the next head's setup is hidden |
| VU-37 | GDN credits: the writer's output credits for a row are taken at its launch; o lines use the TOPK result port | The lanes have no back pressure; one port serves both |

---

## Appendix A. Constants

| Name | Value | Definition |
|---|---|---|
| ONE, HALF, THREE | 0x3F800000, 0x3F000000, 0x40400000 | 1, 0.5, 3 |
| THIRD | 0x3EAAAAAB | f32(1/3) |
| KG | 0x41B2E92F | f32(1/0.044715) |
| LK(P1), LK(N1) | 0x5C551D94AE | round(log₂e·2³⁸) |
| LK(N2) | 0xB8AA3B295C | 2·LK(P1) |
| LK(NG) | 0x6969F3DD6 | round(2·√(2/π)·0.044715·log₂e·2³⁸) |
| P1F, P2F, P3F | 0xB17217F8, 0x3D7F7BFF, 0x0E35846C | round(ln2·2³²), round(ln²2/2·2³²), round(ln³2/6·2³²) |
| C64 | 0xA2F9836E4E44152A | round(2/π·2⁶⁴) |
| PIO2 | 0x6487ED5110B4611A | round(π/2·2⁶²) |
| SN1, SN2, SN3 | 0xBE2AAAA3, 0x3C08839E, 0xB94CA1F9 | Cephes sinf |
| CS1, CS2, CS3 | 0x3D2AAAA5, 0xBAB6061A, 0x37CCF5CE | Cephes cosf |

T64[j] = round(2^(j/64)·2³⁰). 2^(j/64) is irrational for j ≠ 0, so no ties occur. Row i lists j = 8i … 8i + 7:

```
0x40000000 0x40B268FA 0x4166C34C 0x421D1462 0x42D561B4 0x438FB0CB 0x444C0740 0x450A6ABB
0x45CAE0F2 0x468D6FAE 0x47521CC6 0x4818EE22 0x48E1E9BA 0x49AD1598 0x4A7A77D4 0x4B4A169C
0x4C1BF829 0x4CF022CA 0x4DC69CDD 0x4E9F6CD4 0x4F7A9930 0x50582888 0x51382182 0x521A8AD7
0x52FF6B55 0x53E6C9DA 0x54D0AD5A 0x55BD1CDB 0x56AC1F75 0x579DBC57 0x5891FAC1 0x5988E209
0x5A82799A 0x5B7EC8F2 0x5C7DD7A4 0x5D7FAD59 0x5E8451D0 0x5F8BCCDB 0x60962665 0x61A3666D
0x62B39509 0x63C6BA64 0x64DCDEC3 0x65F60A7F 0x6712460B 0x683199ED 0x69540EC9 0x6A79AD56
0x6BA27E65 0x6CCE8AE1 0x6DFDDBCC 0x6F307A41 0x70666F76 0x719FC4B9 0x72DC8374 0x741CB528
0x75606374 0x76A7980F 0x77F25CCE 0x7940BB9E 0x7A92BE8B 0x7BE86FBA 0x7D41D96E 0x7E9F0606
```

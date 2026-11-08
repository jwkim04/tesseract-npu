# Tesseract — vector unit and approximate math (area VU)

Status: **design contract for M6, not implemented** (2026-10-30). Area VU of the design set in
`docs/design/`: EX execution model (`01-execution-model.md`), VU (this file), AT attention and KV, CC
control core (`04-control-core.md`), DB debug, SW software and verification
(`06-software-verification.md`). Order of authority, as in the other design documents: `docs/spec.md`
(to be updated from these documents) > ISA simulator > this document. Every number marked
*estimate* is unmeasured. Numbers marked *model* come from a numpy/mpmath model of the algorithms
in §6 run on 2026-10-30 (scratch scripts, not the C reference). The exhaustive tests of §10 replace
them.

---

## 1. Summary

- **One operand model for every VU op.** An op works on `rows` rows of `m` 32-bit elements. Each
  row starts on an SPM line, and each operand has its own row stride in lines, where stride 0
  repeats one vector over all rows. A per-row scalar operand `c` comes from an s register or from
  an SPM vector (`c[r]`). One op therefore covers all heads, all tokens, or both. This closes the
  survey's gap of about 18 K VU ops per 32B decode token.
- **Ops** (unit 3, VU):
  - `VOP` 0x30: 25 element functions, including GELU16 / GEGLU16, the ggml-CPU-compatible fp16
    GELU forms that llama.cpp graphs use.
  - `RED` 0x31: segmented SUM, SUMSQ and MAX. Results go to a per-row SPM vector or to an s
    register, with fused row post-ops RCP and RMS.
  - `ROPE` 0x32: T tokens with a position vector; angles by ggml-cpu's own recurrence, so they
    are bit-identical to the CPU's for ext_factor = 0.
  - `TOPK` 0x33: segmented, k ≤ 16. The simulator has it from M6; the RTL comes in M8.
  - `DQROW` 0x12: re-specified as SPM → f32 dequantisation inside the VU. Simulator from M6, RTL in
    M8 (`CAPS` bit6 = 0 in M6 bitstreams).
  - `VMASK` 0x38 (new): position-window mask for caches where cell j holds position k0 + j. M9;
    neither the simulator nor the RTL implements it before (llama.cpp KV caches use AT's mask
    summaries).
  - `SOP` 0x40: 15 scalar functions.
  - AT's vector ops `ASMAX`, `AMASK`, `AMERGE`, `ANORM` (0x34–0x37): semantics by AT, executed by
    the VU's lanes (§5.11).
  - `GDN` 0x39 (`HAS_GDN`, `CAPS` bit 8; §7.12): the fused Gated DeltaNet state update for one
    head, S read and written once. Dot products as exact integer sums in DSP48E2 accumulators, URAM row
    buffers, the element update on the front lanes, 8 state elements per cycle at 8 front lanes.
- **No op fails at run time.** Every check runs at dispatch. Shape limits use EX's new code 13
  `OPERAND`.
- **Numerics, all bit-exact:**
  - EXP is an integer algorithm with f32 input and output: ≤ 0.53 ulp (*model*), every
    intermediate fits in 64 bits.
  - RECIP is correctly rounded everywhere; RECIP_APPROX is removed.
  - SIN and COS use an exact 64-bit 2/π reduction and f32 polynomials: absolute error ≤ 2^−23.6
    (*model*).
  - SILU, SIGMOID, GELU-tanh (computed as x·σ(2u)) and TANH are built from these; GELU16 rounds
    its input and output to binary16 and clamps at ±10, as the ggml CPU does.
  - Reductions keep the spec's 128 partial sums.
- **Microarchitecture:**
  - The default 16 front lanes stream one line per cycle. Each has 3 fp multipliers, 1 fp
    adder and an EXP core. `VU_FRONT_LANES=8` processes a line in two parts while retaining
    all 16 logical lane histories; the K4 routing fallback uses 8 front / 8 EXP / 4 back lanes.
  - 8 back lanes have 1 ± e units, an exact reciprocal and 2 multipliers. SILU, GELU, SIGMOID,
    TANH and RECIP run at half rate through them.
  - A 9-adder reduction unit and an angle unit for RoPE.
  - Pair reads on CC's even/odd SPM, credit-based flow with no global clock enable.
  - Size: ≈ 48 K LUT, ≈ 366 DSP, ≈ 15 RAMB36 (*estimate*), in SLR1 next to the SPM.

---

## 2. Inputs and what changes

### 2.1 Facts relied on

- **Spec VU:** `docs/spec.md` §9 (structure, scalar unit, approximate functions, RoPE,
  RMSNorm example) and §11 (the ISA rows, VOP and SOP codes, ROPE and DQROW semantics). fp rules:
  §7. FTZ, tail bytes, check order, range check and length 0: §21 (FTZ decision point, Tail bytes,
  Error check order, Range check, Zero length).
- **Simulator:** VOP/RED/ROPE/TOPK/SOP/DQROW return `ILLEGAL_OP` (`sw/sim/tsr_sim.c:839-842`). The
  fp primitives are `tsr_fadd`…`tsr_frecip` (`sw/sim/tsr_ref.c:99-104`), and `tsr_dequant_row` is
  at `sw/sim/tsr_ref.c:369`. `sw/sim/tsr_ref_math.c` does not exist.
- **RTL:**
  - `tsr_fp_add`: LAT 5, 421 LUT / 350 FF, 300 MHz out of context, LAT ≤ 8 allowed
    (`rtl/core/tsr_fp_add.sv:15-25`, `:68-73`).
  - `tsr_fp_mul`: LAT 4, 2 DSP48E2 (`rtl/core/tsr_fp_mul.sv:11-42`).
  - `tsr_fp_div`: iterative, one op per 15 cycles (`rtl/core/tsr_fp_div.sv:12-18`).
  - `PERF_VU_CYCLES` is index 3 (`rtl/core/tsr_pkg.sv:129`).
- **EX:**
  - VU opcodes 0x12, 0x13, 0x30–0x33, 0x38–0x3F, 0x40, 0x41–0x47. 0x34–0x37 are AT's ops, which
    run on unit 3 (EX §7.2).
  - w0[31:16] = `x16`, op-specific.
  - Code 13 `OPERAND`, with check order `ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` →
    `T_RANGE` → `OPERAND` → `SPM_RANGE` → `DRAM_RANGE`.
  - `CAPS` bits 0, 3, 6, 7, 8 and `CFG3[15:0]`.
  - PERF slots X + 13 … X + 15 (EX §6).
  - s bank: +0.0 after a hardware reset, unchanged by `SOFT_RESET`.
  - No op may fail after it starts.
  - Sources: `01-execution-model.md` §6, §7.1–§7.5, §8.1, §14.
- **CC:**
  - SPM: 2 read + 2 write ports, even/odd banks. The VU is client 4 with both bank ports and pair
    reads (`04-control-core.md` §10.1–§10.2).
  - Registers: `SREG0`–`15` at core CSR 0x9C0–0x9FC, read-only (EX §5.1, CC §6.1). The RISC-V
    reads them through the core window (CC §5.3).
  - VU queue depth 16 (CC §7.4).
  - Checker class slot: four operand extents, each tagged SPM or DRAM (CC §8.2).
  - I3 allows S − D ≤ 2 for the VU, as for the MVM (CC §9.4); units have no error output.
  - SLR1 budget and the 65 % gate (CC §14.1).
- **Pinned llama.cpp:**
  - Gemma 4 (`src/models/gemma4.cpp`):
    - attention scale 1.0 (`:12`); embedding scale √n_embd (`:171`), applied as a MUL by the F32
      graph input `inp_scale_rows` [1, T] (`src/llama-graph.cpp:2776-2783`);
    - freq factors on full-attention layers (`:212`);
    - per-head Q/K norms with weight (`:232`) and V norm without weight (`:264`);
    - GEGLU (`:312`, `:364`), `out_scale` (`:400`), final softcap scale-tanh-scale (`:434-437`).
  - Proportional RoPE through 1e30 freq factors: `conversion/gemma.py:747-762`.
  - CPU GELU and GEGLU use an fp16 table: `GGML_GELU_FP16` is defined unconditionally
    (`ggml/src/ggml-cpu/vec.h:46`); `ggml_vec_gelu_f32` / `ggml_vec_geglu_f32` (`vec.h:999-1013`,
    `:1529-1543`) clamp at ±10, round x to fp16 and read `ggml_table_gelu_f16`, built with libm
    `tanhf` (`ggml-cpu.c:3894`). CPU SiLU is `x/(1+expf(-x))` (`vec.h:1060-1062`).
  - SWA mask rule: `src/llama-hparams.h:506-517`. RoPE: angles by the recurrence of
    `ggml_rope_cache_init` (`ggml/src/ggml-cpu/ops.cpp:6287-6301`, `theta_scale` at `:6455`),
    `rope_yarn` (`:6270-6285`), rotation `:6375-6391`.
  - FA is required for a quantized V cache: `src/llama-context.cpp:566-569`.
  - Gemma 4 sizes with a type label in the pinned commit: 31B dense (60 layers), 26B-A4B (MoE),
    E2B, E4B (`gemma4.cpp:28-34`). The 12B dense model on the build host loads as
    `LLM_TYPE_UNKNOWN`, which changes nothing in the graph; its dimensions are read from its GGUF
    (§3, SW §8).

### 2.2 Changes against `docs/spec.md`

| Item | spec revision 0.2 | This document |
|---|---|---|
| VOP | 1-D length n; 10 functions; scalar only from s | rows × m with per-operand strides; c from s or an SPM row vector; 25 functions (incl. GELU16 / GEGLU16) |
| RED | 1-D, result to an s register | per-row results to an SPM vector or (rows = 1) an s register; post-ops RCP, RMS |
| ROPE | one token; θ from a host table | T tokens, position vector, source and destination strides; legal head_dim / n_dims fixed; angles by ggml-cpu's recurrence (freq factors, freq_scale), a host A table only for YaRN |
| TOPK | k pairs, M8 | rows, padding rule, throughput target; RTL M8, simulator M6 |
| SOP | 9 functions | 15 (adds SUB, DIV, RSQRT, MAX, MIN, EXP) |
| DQROW | DRAM row → f32, unit unspecified | SPM copy of the repacked row (loaded by LD) → f32, VU unit, rows; RTL in M8 |
| VMASK | — | new, 0x38, M9 |
| AT vector ops | — | ASMAX, AMASK, AMERGE, ANORM (0x34–0x37, AT's semantics) executed by the VU (§5.11) |
| RECIP_APPROX | approximate | **removed**; every reciprocal is correctly rounded (`tsr_frecip`) |
| EXP, SIN, COS | "defined by `tsr_ref_math.c`" | defined here bit for bit (§6) |
| s registers | two banks | one bank, written only by VU ops, read-only for the RISC-V and the host |
| Register fields ra–rd | must be 0 | `x16` carries VU fields |

---

## 3. What the target models need

| Model (decode, T = 1) | n_embd | layers | heads / KV heads | head_dim | n_ff | vocab | RoPE |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 896 | 24 | 14 / 2 | 64 | 4864 | 151,936 | NEOX, n_dims = head_dim |
| Qwen2.5-32B | 5120 | 64 | 40 / 8 | 128 | 27,648 | 152,064 | NEOX |
| Gemma 4 12B (read from `models/gemma4/gemma-4-12b-it-Q8_0.gguf`) | 3840 | 48 (40 SWA + 8 global, pattern 5 : 1) | 16 / 8 (SWA), 16 / 1 (global) | 256 (SWA), 512 (global) | 15,360 | 262,144 | NEOX; SWA: base 1e4, n_dims 256; global: base 1e6, n_dims 512 with `rope_freqs` [256] (1e30 factors); final softcap 30; window 1024 |
| Gemma 4 31B (*confirm from the GGUF*; HF config as recalled) | 5376 | 60 | 32 / 16 (SWA), 32 / 4 (global) | 256 (SWA), 512 (global) | 21,504 | 262,144 | NEOX; global layers: n_dims 512 with 1e30 freq factors on ¾ of the pairs |

| ggml node | Qwen2.5 | Gemma 4 dense | VU ops (§8) |
|---|---|---|---|
| RMS_NORM (+ MUL weight) | attn, ffn, output norm | + post norms, per-head Q/K norm with weight, per-head V norm without weight | `RED.SUMSQ` (post RMS) + `VOP.MULCB` / `MULC` |
| ADD | QKV bias, residuals | residuals | `VOP.ADD` (bias: b stride 0), or MVM ACC |
| SCALE, MUL by 1-element or per-row tensor | — | embedding × `inp_scale_rows` (per-row c vector), softcap 1/cap and cap, `layer_output_scale` | `VOP.MULC` (`cmode` 1 or 2), `ADDMULC` |
| GLU | SWIGLU | GEGLU (the CPU's fp16 form) | `VOP.SWIGLU`, `GEGLU16` |
| ROPE | NEOX 64 / 128 | NEOX 256 / 512 | `ROPE` |
| TANH | — | final logit softcap | `VOP.TANH` |
| GET_ROWS (token_embd) | CPU through M7 (SW R2) | CPU through M7 | M8: LD + `DQROW` |
| CPY / CONT / f32 GET_ROWS | — | — | `VOP.COPY` |
| FLASH_ATTN_EXT | softmax pieces | idem + SWA mask | AT's ops (AT §6.6): ATTQK bounds from the mask summary, ASMAX, AMASK, AMERGE, ANORM on the VU; `VMASK` is not used for llama.cpp KV caches |

Gemma 4 has no attention-logit softcap. `attn_soft_cap` is not set for gemma4, and
`llama-graph.cpp:2993` applies a softcap only if it is.

---

## 4. Programming model

### 4.1 Data and fp rules

- **Elements:** VU elements are 32-bit words.
- **fp rules:** every function marked "fp" follows the spec's rules: IEEE binary32,
  round-to-nearest-even, DAZ + FTZ as fixed in `docs/spec.md` §21 (FTZ decision point), and canonical NaN 0x7FC00000
  (`docs/spec.md` §7). It is built only from the primitives of §6.1 and integer operations.
- **Raw functions:** functions marked "raw" copy bits (COPY, SETC, SOP.MOVI / LDS / STS, the
  unmasked elements of VMASK, ROPE's unrotated tail). They also move integer data such as positions
  or indices.
- **No subnormal output:** no VU fp result is ever subnormal, and no fp NaN result is anything but
  0x7FC00000.

### 4.2 The 2-D operand model

- **Rows and extents.** An op processes `rows` rows of `m` elements. Define L(m) = ⌈m/16⌉. Row r
  of an operand with line address X and stride s_X occupies lines [X + r·s_X, X + r·s_X + L(m)).
  Element j of row r is the word at byte 64·(X + r·s_X) + 4j. Element j sits in lane j mod 16 of
  line ⌊j/16⌋. Results never depend on the number of hardware lanes.
- **Stride 0 is legal for sources:** every row then reads the same vector, which is how a norm
  weight or a bias is broadcast over heads or tokens.
- **Destination stride:** with rows > 1 the destination stride must be ≥ L(m), else `OPERAND`.
  Destination rows never overlap.
- **Single row:** with rows ≤ 1, strides are not used and may hold any value.
- **Empty ops (n = 0 rule):**
  - m = 0 or rows = 0: the op reads and writes nothing in the SPM.
  - Exception: a RED with rows = 1 and an s destination writes the empty-row result (§6.2).
  - Range checks still apply, with extent 0 (`docs/spec.md` §21, Range check and Zero length).
- **Tail rule per row:** when m mod 16 ≠ 0, words [m mod 16, 16) of each destination row's last
  line are written 0. This is `docs/spec.md` §21 (Tail bytes) applied row by row.

### 4.3 The per-row scalar operand c

`cmode` 0 means no c. With `cmode` 1, c = s[si] for every element. With `cmode` 2, c_r is the
32-bit word r of the vector starting at SPM line C, for every element of row r (extent ⌈rows/16⌉
lines). RED writes such vectors (§5.3), so reductions feed the next op without a RISC-V round trip.

### 4.4 Scalar registers s0–s15

- One bank of 16 × 32 bits, inside the VU.
- **Writers:** RED (TO_S) and SOP (MOVI, LDS, arithmetic), only.
- **Readers:** VOP (`cmode` 1), the RED post-op (sk, se), ROPE (sm) and SOP. A VU op reads its s
  operands when it starts, after every earlier VU op's s write.
- **Reset:** +0.0 after a hardware reset. `SOFT_RESET`, RISC-V reset, QABORT and QCLEAR do not
  change the bank (EX §7.4). Kernels must write every s register they read; the SDK kernel
  prologue does so with SOP.MOVI.
- **RISC-V:** read-only, through the core window at 0x2000_1000 + 0x9C0 + 4i (CC §5.3). The read
  returns the value written by VU op t once the RISC-V has observed done(t). There is no RISC-V
  write path; values go in with `SOP.MOVI`, which queue order and dependencies order.
- **Host:** `SREG0`–`15` at core CSR 0x9C0 + 4i, read-only (EX §5.1, DB §10.1), present whenever
  the VU is built. This is the one read location for every reader. There is no debug write path
  in M6.
- **Race definition:** the 16 registers are 16 locations (SW §4.6). A core-window read of s_i must
  be ordered after the last VU op that writes s_i and before the next one.

### 4.5 Ordering, completion, aliasing

- **Order:** VU ops run in VU queue order. Each sees every SPM and s effect of earlier VU ops.
  As-if overlap inside the VU is allowed (§7.8).
- **Done** means: every SPM write accepted by `tsr_spm`, and the s write performed.
- **No run-time errors:** NaN, Inf and arbitrary bits are values, not errors (EX §7.2).
- **Aliasing:**
  - A source may coincide with the destination only exactly (same start line and same stride;
    for ROPE, the same start and the same token stride). The op then works in place, element by
    element.
  - Any other overlap between a source extent (including c, pos and table) and the destination
    extent is undefined; the simulator's checker flags it (SW).
  - Sources may overlap each other freely.

---

## 5. Encodings

### 5.1 Common rules

- **w0** = {x16 = w0[31:16], flags[15:12], sub[11:8], opcode[7:0]} (EX §7.1). A field that an op or
  function does not use must be 0, else `ILLEGAL_OP`.
- **Field widths:**
  - SPM line addresses: w[23:0], with w[31:24] = 0.
  - Element counts m: 24 bits.
  - rows, strides, head counts and T: 16 bits.
  - Bigger SPMs (> 2^24 lines) are outside these encodings.
- **Unit:** every op here runs on unit 3 (VU), queue depth 16 (CC §7.4).
- **Range checks:** `docs/spec.md` §21 (Range check), with the extents of §5.9.

### 5.2 VOP (0x30)

| Field | Bits | Meaning |
|---|---|---|
| opcode | w0[7:0] | 0x30 |
| cmode | w0[11:8] | 0 none, 1 s register, 2 SPM row vector; 3–15 `ILLEGAL_OP` |
| flags | w0[15:12] | 0 |
| func | x16[7:0] | function, table below; undefined codes `ILLEGAL_OP` |
| si | x16[11:8] | s register for `cmode` 1, else 0 |
| — | x16[15:12] | 0 |
| d | w1[23:0] | destination line |
| a | w2[23:0] | source a line (0 if the function reads no a) |
| b | w3[23:0] | source b line (0 if unused) |
| c | w4[23:0] | row-vector line for `cmode` 2, else 0 |
| m | w5[23:0] | elements per row |
| rows | w6[15:0] | rows |
| d_stride | w6[31:16] | lines |
| a_stride | w7[15:0] | lines (0 if a is unused) |
| b_stride | w7[31:16] | lines (0 if b is unused) |

Functions take inputs a, b, c. Formulas use §6 names. F = one line per cycle, H = half rate (§7.4).
A function that uses c requires `cmode` ≠ 0; one that does not requires `cmode` = 0.

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
| 0x19–0xFF | — | | `ILLEGAL_OP` | |

Codes 0–9 keep the spec's numbers. MULS and ADDS become MULC and ADDC with `cmode` 1. llama.cpp
graphs use GELU16 / GEGLU16 (SW-31); GELU / GEGLU (exact tanh form) serve M8 and PyTorch.

### 5.3 RED (0x31)

| Field | Bits | Meaning |
|---|---|---|
| kind | w0[11:8] | 0 SUM, 1 SUMSQ, 2 MAX; 3–15 `ILLEGAL_OP` |
| TO_S | w0[12] | 1: result to s[sd] (rows must be 1); 0: per-row vector at d |
| — | w0[15:13] | 0 |
| sd | x16[3:0] | destination s register if TO_S, else 0 |
| post | x16[5:4] | 0 NONE, 1 RCP, 2 RMS, 3 `ILLEGAL_OP`; MAX requires 0 |
| — | x16[7:6] | 0 |
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
| — | w0[15] | 0 |
| sm | x16[3:0] | s register holding M (attn_factor, or the folded YaRN mscale with ATAB) |
| sts | x16[7:4] | s register holding TS = `powf(freq_base, −2/n_dims)` (recurrence form; 0 with ATAB) |
| sfs | x16[11:8] | s register holding freq_scale when FS, else 0 |
| — | x16[15:12] | 0 |
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

**Legal shapes:**
- head_dim is a multiple of 16 in [16, 1024].
- n_dims is in [1, head_dim]: a multiple of 32 for NEOX, a multiple of 16 for NORM.
- Otherwise `OPERAND` (§5.9).

With these rules every pair half starts on a line boundary: line j pairs with line j + n_dims/32
in NEOX. They cover Qwen2.5 (64/64, 128/128) and Gemma 4 (256/256, 512/512).

### 5.5 TOPK (0x33; simulator M6, RTL M8)

| Field | Bits | Meaning |
|---|---|---|
| k | x16[4:0] | 1 … 16 (x16[15:5] = 0) |
| d | w1[23:0] | output line of row 0: k pairs (word 2i index u32, word 2i+1 value) |
| a | w2[23:0] | source line |
| m | w5[23:0] | elements per row |
| rows | w6[15:0] | rows |
| d_stride | w6[31:16] | lines; 0 = ⌈k/8⌉ |
| a_stride | w7[15:0] | lines (w7[31:16] = 0) |

w3 and w4 are 0.

### 5.6 DQROW (0x12, re-specified)

| Field | Bits | Meaning |
|---|---|---|
| fmt | w0[11:8] | 0 Q8_0, 1 Q4_K; 2–15 `BAD_FMT` |
| d | w1[23:0] | f32 output line of row 0 |
| src | w2[23:0] | SPM line of row 0's repacked bytes (as stored in DRAM, `docs/spec.md` §5; loaded by LD) |
| K | w5 | elements per row; `K_RANGE` if K = 0, K > K_MAX, or K not a multiple of the block size |
| rows | w6[15:0] | rows (w6[31:16] = 0) |
| src_stride | w7[15:0] | lines; 0 = ROW_BEATS(fmt, K) |
| d_stride | w7[31:16] | lines; 0 = L(K) |

x16, w3 and w4 are 0. The RISC-V computes the row's channel and offset (`tsr_row_location`) and
pushes `LD` (ROW_BEATS beats), then `DQROW` with a dependency on the LD. DQROW is in the simulator
from M6; its RTL comes with M8 (the device embedding lookup), so an M6 bitstream reports `CAPS`
bit6 = 0 and refuses the opcode with `ILLEGAL_OP`.

### 5.7 VMASK (0x38, new; M9)

VMASK masks by κ_j = k0 + j, so it is valid only for caches where cell j holds position k0 + j,
such as the M9 resident decode loop's own cache. llama.cpp's KV caches are rings that are reused
after a wrap (Gemma 4's 1,536-cell SWA cache) and can be reordered by `seq_rm` or a state load;
the FA kernel therefore never uses VMASK and takes its bounds from AT's mask summaries (AT §7.3).
The encoding and semantics below are fixed now; neither the simulator nor the RTL implements
VMASK before M9 (`ILLEGAL_OP` until then, `CAPS` bit7).

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

Word layout: w1[3:0] sd, [7:4] sa, [11:8] sb (w1[31:12] = 0); w2 imm (f32 bits);
w3[27:0] SPM word address for LDS/STS (w3[31:28] = 0); w4–w7 = 0 and x16 = 0. Fields that a
function does not use are 0.

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
| 15 | — | `ILLEGAL_OP` |

### 5.9 Dispatch checks

**`OPERAND` (13).** Conditions are tested in this order; `ERR_INFO` is the value shown.

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
| VMASK (M9) | G = 0 | 0 |
| VMASK (M9) | rows > 1 and x_stride < L(m) | x_stride |

**SPM extents** in lines. E(n, s, len) = 0 if n = 0 or len = 0, else (n − 1)·s + len. Operands
are checked in field order (w1, w2, w3, w4). `ERR_INFO` of `SPM_RANGE` is the start of the first
failing operand.

| Op | Extents |
|---|---|
| VOP | d E(rows, d_stride, L(m)); a, b likewise when read; c ⌈rows/16⌉ (rows ≥ 1, `cmode` 2) |
| RED | d ⌈rows/16⌉ (not TO_S); a E(rows, a_stride, L(m)) |
| ROPE | with T′ = T if n_heads > 0 else 0: d E(T′, d_stride′, H); src E(T′, src_stride′, H); pos ⌈T′/16⌉; table (FF or ATAB) T′ > 0 ? ⌈n_dims/32⌉ : 0 |
| TOPK | d E(rows, d_stride′, ⌈k/8⌉); a E(rows, a_stride, L(m)) |
| DQROW | d E(rows, d_stride′, L(K)); src E(rows, src_stride′, ROW_BEATS(fmt, K)) |
| VMASK (M9) | x E(rows, x_stride, L(m)) |
| SOP LDS / STS | word address + 1 ≤ `SPM_BYTES`/4; `ERR_INFO` = the word address |

VOP and ROPE need four SPM extents, as do AT's AMASK and AMERGE. CC's checker class slot has four
operand extents, each tagged SPM or DRAM (CC §8.2); no VU op has a DRAM operand. The checks of
AT's vector ops are AT §5.8's rows.

### 5.10 CAPS, CFG3, PERF, debug state

- **`CAPS`:**
  - bit0: VOP, RED, ROPE and SOP as defined here (M6).
  - bit3: TOPK (RTL in M8; the simulator implements it from M6 and reports it when configured).
  - bit6: DQROW in the SPM form (RTL in M8; the simulator from M6).
  - bit7: VMASK (M9).
  - bit8: GDN (the fused state update, §7.12; the simulator implements it, RTL builds with `HAS_GDN`).
  - bits 9–11: reserved for VU extensions, 0.
- **`CFG2`** is unchanged: VL = 16, NSREG = 16.
- **`CFG3[15:0]`:** [4:0] front lanes with an EXP core (16), [9:5] back lanes (8), [15:10] 0.
  This is information for cycle models only; results never depend on it.
- **PERF.** Index 3 `VU_CYCLES` counts cycles in which the VU holds at least one accepted op (from
  accept to its done), gated as EX §6 defines. Requested extension slots, in this order:

  | Slot | Name | Kind | Counts |
  |---|---|---|---|
  | X + 13 | `VU_SPM_WAIT` | time | cycles with a raised, ungranted VU SPM request |
  | X + 14 | `VU_HAZARD` | time | cycles in which an accepted op waits for an earlier VU op (§7.8) |
  | X + 15 | `VU_LINES` | event | destination lines written by VU ops |

- **Debug state** exported for DB to place in `DBG_*`:
  - w0 of the oldest held op;
  - its current row and line;
  - the number of ops held;
  - flags: reduction draining, waiting for SPM, waiting on a hazard.

  DB exposes the per-unit state through `DBG_U3_*` (DB §5.5); the VU-specific fields above are
  readable through DB's probe words when DB places them, and are not needed for any contract.

### 5.11 AT's vector ops on the VU (0x34–0x37)

AT defines the semantics (AT §5.4–§5.7, normative) and the checks (AT §5.8); the VU executes the
ops on unit 3 with the datapath of §7. This table restates the encodings for the VU decoder; both
come from the single table source (EX-30). For every op x16 = 0 and unused words are 0, else
`ILLEGAL_OP`.

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
- AMERGE: Ok = CTREE over the C channel partials (pairwise by levels, left to right), lk likewise;
  INIT copies; otherwise m′ = MAXT(m_run, mk), a = EXPK(FM(FS(m_run, m′), scale), P1),
  b = EXPK(FM(FS(mk, m′), scale), P1), Orun ← FA(FM(Orun, a), FM(Ok, b)), l_run ← FA(FM(l_run, a),
  FM(lk, b)), m_run ← m′.
- ANORM: out[r][d] = FM(Orun[r][d], RC(MAXT(l_run[r], 2⁻¹²⁶))).

Datapath mapping and rates: §7.4. These ops never fail at run time; their operand limits are
`OPERAND` checks at dispatch (AT §5.8).

---

## 6. Numerics (bit-exact)

### 6.1 Notation and primitives

- **fp primitives:** FA = `tsr_fadd`, FS = `tsr_fsub`, FM = `tsr_fmul`, FD = `tsr_fdiv`, SQ =
  `tsr_fsqrt`, RC = `tsr_frecip` (`sw/sim/tsr_ref.c:99-104`). They implement DAZ, RNE, FTZ and
  canonical NaN, and are correctly rounded.
- **Other notation:** DAZ(x) maps exponent-0 encodings to signed zero. Constants are bit patterns
  (Appendix A). "|x|" clears the sign bit; "−x" flips it.
- **Integer operations:**
  - Unsigned unless marked signed.
  - `>>` is a floor shift. A shift by ≥ 64 of a 64-bit value gives 0.
  - Every intermediate fits in 64 bits, except the two SIN/COS products of §6.6, which are exact
    128-bit products.
- **The C reference:**
  - Location: `sw/sim/tsr_ref_math.c`, under SW §4.5's rules.
  - API: `tsr_m_expk(float, unsigned k)`, `tsr_m_exp`, `tsr_m_sigmoid`, `tsr_m_silu`,
    `tsr_m_gelu_tanh`, `tsr_m_gelu16`, `tsr_m_tanh`, `tsr_m_sincos`, `tsr_m_sin`, `tsr_m_cos`; the
    binary16 helpers `tsr_f32_to_f16_rne` (gradual underflow) and `tsr_f16_to_f32` used by GELU16.
  - `tsr_m_recip_approx` is dropped; RC is `tsr_frecip`.
  - VOP, reductions, ROPE, TOPK, SOP, DQROW and, from M9, VMASK go into `tsr_ref_vu.c`
    (`include/tsr_ref_vu.h`); the AT vector ops into `tsr_ref_at.c` (AT §13.1). Both are part of
    `libtsr_ref`.

### 6.2 Reductions (RED)

Each row is reduced on its own.

- **SUM:** set p_0 … p_127 = +0.0. For i = 0 … m − 1 in order, p_(i mod 128) ← FA(p_(i mod 128),
  x_i).
- **SUMSQ:** the same with FM(x_i, x_i) in place of x_i (the square is rounded first).
- **Tree:** the result is the complete binary tree over p_0 … p_127. Level 1 is FA(p_{2k},
  p_{2k+1}). Each level pairs adjacent results left to right; there are 7 levels. This is the
  spec's rule, `docs/spec.md` §9.
- **MAX:** the maximum of the DAZ'd elements under the total order of §6.8. Any NaN gives
  0x7FC00000.
- **Empty rows** (m = 0): SUM and SUMSQ give +0.0; MAX gives −∞ (0xFF800000).
- **Post-op**, applied to each row result y:
  - NONE: y.
  - RCP: RC(y).
  - RMS: RC(SQ(FA(FM(y, s[sk]), s[se]))).
  - RMS is bit-identical to the spec recipe SOP.MULI → ADDI → SQRT → RECIP (`docs/spec.md` §9, RMSNorm example)
    when s[sk] equals the MULI immediate.

**Implementation freedom (exact).** A partial is initialised as FA(+0, x) (the first update of a
+0 partial; never a raw load of x). A computed tree value q can be −0: the spec's FTZ keeps the
sign, so for example FA(0x80C00000, 0x00800000) = 0x80000000, while FA(−0, +0) = +0. Tree values
are never subnormal, so FA(q, +0) = z(q), where z maps −0 to +0 and leaves every other value
unchanged. When the hardware skips a subtree whose partials were all unwritten (+0), it therefore
passes z(q) upward in place of the computed subtree value q. With this rule, a short row (m < 128)
uses only the partials p_0 … p_{m−1} and needs ⌈m/8⌉ tree cycles (§7.5), and the result is
bit-identical to the complete 128-partial tree. Directed tests cover subtrees that sum to −0 at
m = 8k + r (§10).

### 6.3 EXPK(v, κ) — exponential with a folded constant

EXPK(v, κ) computes exp(K·v), with K and its fixed-point constant LK ≈ |K|·log₂e·2³⁸ set by κ:

| κ | K | Use | sK | LK |
|---|---|---|---|---|
| P1 | +1 | EXP, EXPSUB | 0 | 0x5C551D94AE (396,564,993,198) |
| N1 | −1 | SIGMOID, SILU | 1 | 0x5C551D94AE |
| N2 | −2 | TANH | 1 | 0xB8AA3B295C (= 2 × LK(P1)) |
| NG | −C2G | GELU, C2G = 2·√(2/π)·0.044715 | 1 | 0x6969F3DD6 (28,296,822,230) |

LK = round(|K|·log₂(e)·2³⁸). LK(N2) is exactly twice LK(P1), so EXPK(v, N2) = EXPK(2v, N1)
whenever 2v is exact.

**Algorithm** (u = bits of v):
1. v ← DAZ(v). If v is NaN, return 0x7FC00000.
2. sgn = sign(v) XOR sK.
3. If v = ±∞: return sgn ? +0 : +∞. If v = ±0: return 0x3F800000.
4. Ev = biased exponent (1 … 254); Mv = 2²³ | fraction.
5. If Ev > 156: return sgn ? +0 : +∞ (saturate).
6. P = Mv × LK (< 2⁶⁴). T = P >> (156 − Ev). This is |K·v·log₂e| in units of 2⁻³². If T ≥ 2⁴⁰,
   saturate as in step 5.
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
11. Normalise: if M ≥ 2³¹, then drop = 8 and n ← n + 1; else drop = 7.
12. Round to 24 bits:
    - sig = M >> drop; rem = M mod 2^drop; half = 2^(drop−1).
    - If rem > half, or rem = half and sig is odd: sig ← sig + 1.
    - If sig = 2²⁴: sig ← 2²³ and n ← n + 1.
13. be = n + 127. If be ≥ 255 return +∞; if be ≤ 0 return +0; else return
    (be << 23) | (sig − 2²³).

EXPK never returns −0 or a subnormal.

**Accuracy (*model*, 259 K samples):**
- ≤ 0.526 ulp against exact exp(K·v) for every κ (P1 on [−88, 88] and on tiny |v|, N2 on
  [−44, 44], NG on [−1240, 1240]). Exhaustive (all 2³² inputs, §6.9): 0.5261 ulp for P1, N1
  and N2, 0.5316 ulp for NG.
- Monotonic on a 285 K-step sweep.
- In hardware: 11 DSP per core (*estimate*).

### 6.4 Correctly rounded reciprocal, square root, division

RC, SQ and FD are the spec's correctly rounded operations. VOP.RECIP, the RED post-ops and the
reciprocal inside SIGMOID, SILU, GELU and TANH all use RC. The implementation is free (§7.3). The
contract is bit equality with `tsr_frecip` / `tsr_fsqrt` / `tsr_fdiv`.

### 6.5 Composite functions

- **SIG(x)** = RC(FA(ONE, EXPK(x, N1))).
- **SILU(x)** = FM(x, SIG(x)). This keeps the spec's form x·R(1 + e^−x), with an exact R.
- **GELU(x)**, the tanh form, using 0.5·(1 + tanh u) = σ(2u) and 2u = C2G·x·(x² + KG):
  - w = FM(x, x); w = FA(w, KG); v = FM(w, x)
  - y = FM(x, RC(FA(ONE, EXPK(v, NG))))
  - KG = f32(1/0.044715) = 0x41B2E92F; C2G sits inside LK(NG).
- **TANH(x):**
  - If x is NaN, return 0x7FC00000. Let ax = |DAZ(x)|.
  - If ax < 2⁻⁶ (bits < 0x3C800000), the small path: q = FM(ax, ax); w = FS(THREE, q);
    t = FM(FM(w, ax), THIRD), where THREE = 0x40400000 and THIRD = 0x3EAAAAAB.
  - Else: e = EXPK(ax, N2); t = FM(FS(ONE, e), RC(FA(ONE, e))).
  - y = sign(x) ? −t : t.
- **GELU16(x)**, the ggml CPU's form (`vec.h:999-1013` with `GGML_GELU_FP16`, `vec.h:46`). Let
  H(v) = F32(F16(v)), where F16 rounds DAZ(v) to binary16 with round-to-nearest-even and gradual
  underflow (fp16 subnormals are kept, as F16C `_cvtss_sh` in rounding mode 0 does; overflow gives
  ±∞) and F32 widens exactly. Then, on x′ = DAZ(x):
  - NaN → 0x7FC00000;
  - x′ ≤ −10 → +0;
  - x′ ≥ 10 → x′;
  - otherwise H(GELU(H(x′))), with GELU of §6.5 above evaluated in f32 on the fp16-exact input.
  The CPU reads `ggml_table_gelu_f16`, whose entries are F16 of `ggml_gelu_f32`:
  0.5·x·(1 + `tanhf`(u)) in f32 with libm `tanhf` (and, in the pinned `-march=native` build, the
  FMA contraction of 1 + a·x·x). SW's `ref_math_gelu16` enumerates all 65,536 fp16 inputs and
  compares with the table. **Target:** mismatches < 1 % of the entries, no sign-of-zero mismatch,
  and a mismatch above 1 fp16 ulp only where GELU16 is the correctly rounded fp16 value of the
  exact GELU and the table entry is not. **Measured** (pinned table on the dev host, 2026-10-30):
  272 of 65,536 entries differ (0.415 %; 0.728 % of the 37,376 inputs in (−10, 10)), 265 by
  1 fp16 ulp and 7 by 2 ulp (x ≈ −4.73 and −5.137 … −5.156, |y| < 2·10⁻⁶). In all 272 GELU16 is
  the correctly rounded value and the table entry is not: the residual is the table's own f32
  error (`tanhf` and the f32 steps; for x ≪ 0, 1 + `tanhf`(u) cancels, which gives the 2-ulp
  entries), not a rounding-boundary tie between two correct f32 values. A target of "each ≤ 1
  fp16 ulp" is therefore not reachable without reproducing the CPU's cancellation.
- **GEGLU16(a, b)** = (DAZ(a) ≤ −10) ? +0 : FM(GELU16(a), b), as `ggml_vec_geglu_f32` (`vec.h:1529-1543`).
- **EXPSUB(a, x)** = (DAZ(a) = −∞) ? +0 : EXPK(FS(a, x), P1). Masked scores (−∞) become 0 even
  when the row maximum is −∞. That makes an online softmax that starts on a fully masked block
  well defined instead of NaN.
- **Specials:**

  | Function | at −∞ | at +∞ |
  |---|---|---|
  | EXP | +0 | +∞ |
  | SIG | +0 | 1 |
  | SILU | NaN (as the CPU's `x/(1+expf(-x))`) | +∞ |
  | GELU | NaN | +∞ |
  | GELU16 | +0 | +∞ |
  | TANH | −1 | +1 |

  Finite inputs never give NaN in SIG, SILU, GELU or TANH. GELU of a large negative finite x is
  −0, because v overflows to −∞ and the reciprocal goes to 0.

### 6.6 SIN and COS

**Input and specials.** θ ← DAZ(θ). NaN or ±∞ give SIN = COS = 0x7FC00000. Let s be the sign, E
the biased exponent and M = 2²³ | fraction.
1. E = 0: SIN = θ (a signed zero), COS = ONE.
2. E < 115 (|θ| < 2⁻¹²): SIN = θ, COS = ONE. The error is below 0.1 ulp; cos rounds to 1.

**Quadrant reduction** (exact integers; C64 = round(2/π·2⁶⁴) = 0xA2F9836E4E44152A):

3. P = M × C64 (88-bit exact). sh = 214 − E. θ·2/π ≈ P·2^−sh.
   - sh ≥ 62: k = P >> sh; F = (P >> (sh − 62)) mod 2⁶².
   - 0 ≤ sh < 62: k = P >> sh; F = (P << (62 − sh)) mod 2⁶².
   - sh < 0: k = P << −sh (only k mod 4 is used); F = 0.
4. If F ≥ 2⁶¹: k ← k + 1 and F ← F − 2⁶² (signed, in [−2⁶¹, 2⁶¹)). q = k mod 4.
5. R = F × PIO2, with PIO2 = round(π/2·2⁶²) = 0x6487ED5110B4611A. This is a signed exact ≤ 125-bit
   product. r = the f32 nearest-even to R·2⁻¹²⁴ (+0 if R = 0). |r| is 0 or ≥ 2⁻⁶¹, so r is
   normal.

**Polynomials** (Cephes `sinf` / `cosf` coefficients on [−π/4, π/4]):

6. z = FM(r, r).
   - sn: t = FM(SN3, z); t = FA(t, SN2); t = FM(t, z); t = FA(t, SN1); t = FM(t, z); t = FM(t, r);
     sn = FA(t, r).
   - cs: t = FM(CS3, z); t = FA(t, CS2); t = FM(t, z); t = FA(t, CS1); t = FM(t, z); t = FM(t, z);
     h = FM(HALF, z); u = FS(t, h); cs = FA(u, ONE).
   - Coefficients: SN1 0xBE2AAAA3, SN2 0x3C08839E, SN3 0xB94CA1F9; CS1 0x3D2AAAA5,
     CS2 0xBAB6061A, CS3 0x37CCF5CE; HALF 0x3F000000.

**Quadrant and sign:**

7. (S, C) = q = 0: (sn, cs); 1: (cs, −sn); 2: (−sn, −cs); 3: (−cs, sn).
8. SIN = s ? −S : S. COS = C.

**Accuracy (*model*, 200 K samples on [−4, 4] and [0, 2¹⁸]):**
- Absolute error ≤ 7.5·10⁻⁸ = 2⁻²³·⁶⁶.
- ≤ 1.47 ulp where |y| > 10⁻³.
- Exhaustive on |θ| ≤ 2¹⁸ (§6.9): absolute 7.82·10⁻⁸ = 2⁻²³·⁶¹, 1.562 ulp where |y| > 10⁻³.
- For |θ| < 2²⁴ the reduction error is ≤ 2⁻⁴¹ of a quadrant. Larger θ stay deterministic but are
  meaningless for RoPE: ulp(θ) ≥ 2 there.

### 6.7 ROPE

For each token t < T:
- p = word t of `pos` (int32). Pf = `tsr_i32_to_f32`(p).
- **Angles, recurrence form** (ATAB = 0; the CPU's `ggml_rope_cache_init` for ext_factor = 0,
  `ops.cpp:6287-6301`): θ_0 = Pf and θ_{i+1} = FM(θ_i, s[sts]) for i < n_dims/2 − 1, where s[sts]
  holds the bits of the host's `powf(freq_base, -2.0f/n_dims)` (the same libm as ggml-cpu,
  `ops.cpp:6455`). φ_i = FF ? FD(θ_i, ff_i) : θ_i, with ff_i word i of the table; then φ_i ←
  FS ? FM(s[sfs], φ_i) : φ_i.
- **Angles, A-table form** (ATAB = 1, for YaRN with ext_factor ≠ 0): φ_i = FM(Pf, A_i), where the
  host folds freq_base, freq_scale, the freq factors and the YaRN ramp mix into A_i in double and
  rounds once.
- c_i = FM(M, COS(φ_i)), s_i = FM(M, SIN(φ_i)), with M = s[sm] (the CPU's `cosf(θ)·mscale`).

For each head h < n_heads of token t:
- The pair is (j0, j1) = (i, i + n_dims/2) in NEOX, or (2i, 2i + 1) in NORM. Indices are relative
  to the head's first element. The head is at word h·head_dim of the token's H-line span.
- x0 = src[j0], x1 = src[j1].
- y[j0] = FS(FM(x0, c_i), FM(x1, s_i)).
- y[j1] = FA(FM(x0, s_i), FM(x1, c_i)).
- Elements [n_dims, head_dim) are copied raw.

The rotation is ggml's `rotate_pairs` (`ops.cpp:6375-6391`) without FMA contraction.

**Software side.** Software computes the constants once per model and layer type:
- recurrence form: TS = `powf(freq_base, -2.0f/n_dims)` in f32 with the host libm, M =
  attn_factor, freq_scale (FS only if ≠ 1), and the freq-factor table as given by the GGUF (Gemma 4
  global layers: `rope_freqs` [256] with 1e30 entries);
- A-table form (YaRN only): A_i and M in double with one rounding (`ops.cpp:6270-6285`;
  `docs/spec.md` §9).
- On Gemma 4 global layers, ff_i = 10³⁰ gives φ_i ≈ 10⁻³⁰·θ_i (a normal f32). φ_i < 2⁻¹² then
  takes the identity path of §6.6, and those pairs change only by the x0 − x1·φ rounding, as on
  the CPU.
- With the recurrence the VU's angles φ_i are bit-identical to the CPU's for ext_factor = 0. The
  remaining differences are libm `cosf` / `sinf` against §6.6 and the compiler's possible FMA
  contraction of the CPU rotation (§6.9). The serial FM chain costs n_dims/2 × 4 cycles per token
  (≤ 1 K cycles for n_dims 512) and is interleaved across tokens in the angle unit (§7.6).

### 6.8 Order, raw ops, VMASK, TOPK, DQROW

- **Total order** (`docs/spec.md` §9):
  - Non-NaN values are compared after DAZ by key = sign ? ~bits : bits | 2³¹, so −∞ < … < −0 < +0
    < … < +∞. NaN is above everything.
  - MAXT(a, b): if either is NaN, 0x7FC00000; else the larger DAZ'd value.
  - MINT(a, b): if both are NaN, 0x7FC00000; if one is NaN, the other (DAZ'd); else the smaller.
    By the order, NaN loses a MIN.
- **VMASK** (M9; 64-bit signed integer arithmetic):
  - q_r = q0 + ⌊(P ≠ 0 ? r mod P : r) / G⌋ and κ_j = k0 + j.
  - Element (r, j) is masked iff κ_j > q_r, or W ≠ 0 and q_r − κ_j ≥ W. That is
    `is_masked_swa` with standard SWA (`llama-hparams.h:513-517`) plus the causal rule.
  - Masked elements become 0xFF800000. Others keep their bits. Tails follow §4.2.
  - Rows ordered (token, head) use G = n_head and P = 0. Rows ordered (head, token) use G = 1 and
    P = T.
- **TOPK**, per row:
  - The candidates are (j, x̂_j), where x̂ = DAZ(x) and NaN becomes 0x7FC00000.
  - Sort by x̂ descending under the total order; on a tie the smaller j comes first.
  - Pair i < k goes to words 2i (j as u32) and 2i + 1 (x̂). If m < k, the missing pairs are
    (0xFFFFFFFF, 0xFF800000). Tails follow §4.2.
- **DQROW:** row r = `tsr_dequant_row`(fmt, the bytes at line src + r·src_stride′, K)
  (`sw/sim/tsr_ref.c:369`; ggml-exact for finite scales, `docs/spec.md` §11 and §21, Agreement with ggml). Tails
  follow §4.2.

### 6.9 Accuracy and differences from the ggml CPU backend

The *model* figures came from samples; the measured column is the exhaustive maximum over all 2³²
inputs (`ref_math_acc --full`, against double libm, 2026-10-30; §10 item 2). ulp(r) is taken at
max(⌊log₂|r|⌋, −126), and results whose exact value is below 2⁻¹²⁶ (the FTZ region) are not
measured. Every measured maximum is within the 2× gate of §10 item 2, so the coefficients stand.

| Function | *Model* figure (the §10 gate is 2×) | Measured, all 2³² inputs |
|---|---|---|
| EXPK P1, N1, N2 | ≤ 0.53 ulp | 0.5261 ulp |
| EXPK NG | ≤ 0.53 ulp | 0.5316 ulp (worst at v = 1010.27, a result near 5·10⁻³²) |
| RC, SQ, FD, RECIP, RSQRT steps | ≤ 0.5 ulp each (correctly rounded) | — (exact by definition, §6.4) |
| SIG | ≤ 2.0 ulp | 2.463 ulp (x ≈ −16.64) |
| SILU | ≤ 2.3 ulp | 2.868 ulp where σ(x) ≥ 2⁻¹²⁶ (x ≈ −16.68); below that SIG flushes to 0 and SILU returns −0 |
| GELU | ≤ 11 ulp for x ≥ −3; ≤ 29 ulp for x ≥ −5; absolute ≤ 5.2·10⁻⁷ on [−8, 8] | 13.12 ulp for x ≥ −3; 31.33 ulp for x ≥ −5; absolute 5.16·10⁻⁷ on [−8, 8] |
| TANH | ≤ 9.8 ulp; absolute ≤ 1.2·10⁻⁷ (the n = 2 − d variant would give 26 ulp, hence 1 − e) | 16.77 ulp (x = 0x3C80003F, just above the 2⁻⁶ switch to 1 − e, where 1 − e cancels); absolute 1.191·10⁻⁷ |
| SIN, COS | absolute ≤ 2⁻²³·⁶⁶; ≤ 1.5 ulp where \|y\| > 10⁻³ | on \|θ\| ≤ 2¹⁸: absolute 7.80·10⁻⁸ (SIN) and 7.82·10⁻⁸ (COS) = 2⁻²³·⁶¹; 1.562 ulp where \|y\| > 10⁻³ |

GELU's relative error grows in the far negative tail by about 1.5·|2u| ulp. There the result is
below 10⁻¹⁰ in magnitude.

No VU op is bit-exact with the CPU in general; the KL criterion covers the differences, and the
VU follows the CPU's form wherever the CPU's form is a defined integer or f32 sequence. Known
sources:
- **GELU:** llama.cpp graphs use GELU16 / GEGLU16, which reproduce the CPU's fp16 rounding and
  clamps; the residual is the CPU table's libm `tanhf` (measured by `ref_math_gelu16`). The exact
  tanh GELU (≤ 10⁻⁶ relative) would differ from the CPU by up to 3.9·10⁻³ relative per element
  (*model*), on every GEGLU element of every Gemma 4 layer, which is why it is not used for
  llama.cpp.
- **RMSNorm:** the CPU sums in double and divides by n; the VU uses f32 with 128 partial sums and
  1/n.
- **RoPE:** the angles are bit-identical (recurrence form); `cosf` / `sinf` against §6.6, and the
  CPU compiler may contract x0·c − x1·s into an FMA.
- **SiLU:** the CPU's SIMD exp approximations.
- **FA softmax:** the CPU's FA kernel uses its own expf and f16 paths (AT §6.7).

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

- **Units:**
  - FM_A and FM_B take operands, constants or lane registers in parallel.
  - FA takes FM_A, FM_B, operands, constants or the lane's partial register file (RED
    accumulation, 8 phases, double-buffered).
  - FM_C follows FA.
  - EXPK takes the raw operand, the FA output (EXPSUB) or the FM_C output (GELU).
  - Compare/select handles MAX, MIN, VMASK, the EXPSUB −∞ rule, raw ops and tail zeroing.
- **Blocks:** FM and FA are the existing `tsr_fp_mul` (LAT 4) and `tsr_fp_add` (LAT 5), used with
  `ce` tied high.
- **EXPK core** (new, `tsr_vu_exp`):
  - Mv × LK: 24 × 40 bits, 2–3 DSP.
  - Barrel shift and saturation.
  - Three Horner products: 26 × 28, 26 × 30 and 26 × 32 bits, 2 DSP each.
  - T64 ROM: 64 × 31 bits, about 31 LUT6.
  - T64 × PF: 31 × 26 bits, 2 DSP.
  - Rounding.
  - Latency ≈ 18 cycles (*estimate*), ≤ 6 LUT levels per stage.
- **RED accumulation:** each lane's FA updates phase (line index mod 8) of its partials. A partial
  is updated every 8 cycles, so LAT 5 suffices (`docs/spec.md` §9).

### 7.3 Back lane (× 8)

- **ONE:** computes FA(ONE, e) and FS(ONE, e). These are exact-equivalent specialised adders for
  e ≥ 0 or NaN, which is all EXPK produces.
- **RC** (new `tsr_vu_rcp`, pipelined, correctly rounded):
  - Recommended: a 256-segment quadratic approximation, with coefficients in one RAMB18.
  - It gives a quotient candidate within one unit of the 24-bit floor.
  - One exact 24 × 25-bit correction product chooses the floor and the RNE rounding. There are no
    ties for 1/m.
  - Latency ≈ 12 (*estimate*).
  - The design is free; equality with `tsr_frecip` is the contract. Verification is exhaustive
    over the 2²³ mantissas and the exponent edges.
- **FM_R and FM_S.**
- **fp16 rounding stages** for GELU16 / GEGLU16: one at the lane input (H(x) and the ±10 clamp
  decision, in front lanes 0–7) and one after FM_R (H of the GELU result, before FM_S for
  GEGLU16); integer logic only.
- **SRL delay lines** carry a, b, the clamp decision and the TANH small-path value alongside.

Half-rate functions present each line as two 8-element halves on consecutive cycles. Front lanes 0–7
feed back lanes 0–7, and front lanes 8–15 idle.

### 7.4 Function mapping and latency

| Function | Units | Rate |
|---|---|---|
| ADD, SUB, ADDC, SUBC | FA | F |
| MUL, MULC | FM_A | F |
| MULCB | FM_A → FM_C | F |
| MULCADD | FM_A → FA | F |
| ADDMULC | FA → FM_C | F |
| MAX, MIN, COPY, SETC (VMASK from M9) | cmp/select | F |
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
| AMERGE | prepass over the row vectors (≤ 64 lines): MAXT, FS, FM, EXPK give a_r, b_r (row buffer) and the l_run update; then per output line: CTREE of the C partial lines on FA (C − 1 passes), FM_A(Orun, a_r) ∥ FM_B(Ok, b_r) → FA | ≈ 4–5 cycles per output line at C = 4 (5 line reads, 4 FA uses); INIT ≈ 3 |
| ANORM | prepass: RC(MAXT(l_r, 2⁻¹²⁶)) on the back lanes into the row buffer; then FM_A(Orun, rinv_r) | F (after the prepass) |

There are two fixed latency classes: short (FM_A … FM_C, cmp: about 20 cycles including SPM read
and write) and long (the full chain: about 64 cycles) (*estimates*). An op uses one class for all
its elements.

### 7.5 Reduction unit and row post

- **Draining a row:** at a row's end its 128 partials (double-buffered) drain through an 8-input
  subtree adder (7 FA, 3 levels). It consumes 8 partials per cycle and yields 16 subtree sums per
  row, or ⌈m/8⌉ when m < 128 (§6.2). A scheduled 2-FA top combiner finishes the last 4 levels.
- **Throughput:** a row costs max(L(m), min(⌈m/8⌉, 16)) cycles. Rows of ≥ 256 elements run at
  full rate (Gemma head_dim 256 / 512, prefill norms, attention blocks of ≥ 256 keys); rows of 64
  or 128 elements run at half rate.
- **Post unit:** FM, FA, a pipelined exact SQ and an exact RC, at ≥ 1 row per 4 cycles. Results
  are packed 16 per line into the row vector, or written to s[sd].
- **Result latency:** about 45 cycles after the last line, plus about 30 for RMS (*estimate*).

### 7.6 Angle unit and ROPE engine

- **Per angle:** φ_i from the recurrence θ_{i+1} = FM(θ_i, TS) (one FM chain per token; chains of
  up to 4 tokens interleave to hide the FM latency), the optional FD by ff_i (two iterative
  `tsr_fp_div` instances, ≤ 8 cycles per angle) and FM by freq_scale; or FM(Pf, A_i) in the
  A-table form. Then the integer reduction: a 24 × 64 product (4 DSP) and a 63 × 63 product
  (multi-cycle, 3–4 DSP). Then the two polynomials on a time-multiplexed FM/FA pair, and c, s =
  FM(M, ·).
- **Target:** ≤ 4 cycles per angle (≤ 8 with freq factors, e.g. Gemma 4 global layers: ≈ 2 K
  cycles per token and layer, *estimate*).
- **Angle buffer:** 512 entries × 64 bits (1 RAMB36), double-buffered per token.
- **Rotation:** in the front lanes. NEOX reads line pairs at j and j + n_dims/32 and writes one
  output line per cycle.
- **Optional, invisible reuse:** a ROPE whose (p, table contents, n_dims, M, mode) match the
  previous token's may reuse the angles, for example K after Q. The VU re-reads pos and table and
  compares the values before reusing.

### 7.7 SPM access

- **Client:** client 4 of CC's 2R2W even/odd SPM (CC §10.1–§10.2).
- **Pair reads:** in each read cycle the VU fetches two consecutive lines of one operand (lines L
  and L + 1 always fall in different banks). With two streaming sources it alternates an a-pair
  cycle and a b-pair cycle, so each source gets one line per cycle **whatever the parity of its
  start**. There is no same-parity stall and no placement rule.
- **Costs:**
  - A row with an odd line count ends with a single-line read (≤ 1 slot per row).
  - The c row vector takes one read slot per 16 rows.
  - Single-source ops use half the read slots, which leaves the rest to other clients.
- **Writes:** one line per cycle to the bank of the line.
- **Flow control:** reads are issued only against free output-FIFO credits (rule R4,
  `docs/rtl-m2.md` §1). A stalled write therefore never needs a clock enable in the lanes; the
  pipeline simply stops issuing reads.
- **Buffers:** the output FIFO holds 64 lines (8 RAMB36, or LUTRAM); the input reorder buffers
  hold 2 lines per operand.
- **Contention:** competing clients (DMA, QACT, gather, bridge, AT) get round-robin slots, which
  slows the VU (`VU_SPM_WAIT`).

### 7.8 Sequencer, accept-ahead and hazards

- **Accept-ahead.** The VU raises `cmd_ready` for the next op while one earlier op is still in its
  pipeline (S − D ≤ 2). Its done pulses stay in accept order, and it never reports a run-time
  error. This needs CC's I3 relaxed for unit 3, as for the MVM (§9).
- **Hazards.** Before the next op issues its first read, the VU compares:
  - its source extents (a, b, c, pos, table) and its destination extent against the destination
    extents of held ops (RAW, WAW);
  - its s reads against held s writes.

  Any overlap makes it wait until those ops are done. A wait is counted in `VU_HAZARD`.
- **Gain.** Independent ops then overlap their fill and drain (about 20–64 cycles each). Dependent
  chains pay the latency.

### 7.9 Throughput targets (*estimates*, core clock 276.3 MHz)

| Op | Cost |
|---|---|
| VOP, rate F | rows × L(m) cycles + ≤ 1 per row (odd line count) + ⌈rows/16⌉ (c vector) + latency (20 / 64) |
| VOP, rate H | 2 × rows × L(m) + same overheads |
| RED | rows × max(L(m), min(⌈m/8⌉, 16)) + about 45 (+30 RMS) |
| ROPE | T × (4 × n_dims/2 [unless reused] + n_heads × head_dim/16) + about 40 |
| COPY, SETC (VMASK from M9) | rows × L(m) + about 20 |
| ASMAX | R × n_lines + about 64 |
| AMASK | R × n_lines (+ ≤ 1 per row) + about 45 (row maxima) |
| AMERGE | prepass ≈ 6 × ⌈R/16⌉ + 64; then R × D/16 × (4–5) (INIT: × 3); 32B, R = 40, D = 128: ≈ 1,600 (AT §9.2) |
| ANORM | prepass ≈ 4 × ⌈R/16⌉ + 64; then R × D/16 |
| DQROW (M8) | 16 outputs per cycle: rows × L(K) + about 30 |
| TOPK (M8) | m/16 per row typical (running-threshold filter, one insertion per cycle); m per row worst case (ascending input) |
| SOP | ≤ 30 cycles (DIV ≤ 40) |

For TOPK with m = 262,144: about 59 µs typical, 0.95 ms worst case.

### 7.10 Area, placement and timing

| Part | LUT | FF | DSP | RAMB36 |
|---|---|---|---|---|
| 16 front lanes (3 FM, FA, EXPK, cmp, partials) | 24.7 K | 26 K | 272 | 0 |
| 8 back lanes (ONE, RC, 2 FM, fp16 rounding stages, delays) | 9.2 K | 8.3 K | 72 | 4 |
| Reduction (9 FA) + post (FM, FA, SQ, RC) | 5.5 K | 5 K | 12 | 0 |
| Angle unit (recurrence, 2 dividers) + buffer | 2.6 K | 2.5 K | 10 | 1 |
| Sequencer, address generators, hazards, reorder, s bank, SOP | 3.6 K | 3 K | 0 | 0 |
| AT vector ops: multi-pass sequencing, row-scalar buffers (a, b, 1/l) | 2.0 K | 1.5 K | 0 | 2 |
| Output FIFO | 0.3 K | 0.2 K | 0 | 8 |
| **Total M6** | **≈ 48 K** | **≈ 46.5 K** | **≈ 366** | **≈ 15** |
| Deferred: TOPK (M8) / DQROW front end (M8) / VMASK (M9) | +4 K / +1.3 K / +1 K | +3 K / +1 K / +0.5 K | 0 | 0 |

All figures are *estimates* (± 30 %; FM ≈ 140 LUT assumed). CC's SLR1 budget carries this total
(CC §14.1). DSP is not scarce in SLR1: about 1,285 are free.

**Measured (W4-B, 2026-11-01; `tb/vu/viv/vu_ooc.sh`, out of context after routing, xcu200-2, 3.333 ns; the estimates
above are about 40 % low).** M6 configuration, 16 EXP / 8 back lanes: 84.0 K LUT (72.6 K logic, 4.8 K LUTRAM, 6.5 K SRL),
101.7 K FF, 433 DSP, 16 RAMB36 + 25 RAMB18 (the RAMB18s are the lanes' EXPK and RC ROMs), WNS +0.003 ns / WHS +0.019 ns.
Per block (LUT): 16 lanes 49.6 K (368 DSP), angle unit 8.2 K (48 DSP), reduction and post 9.4 K, injector 6.1 K, generator 3.6 K,
writer 3.2 K, reader 2.1 K, top 1.8 K. Knobs 8 / 4: 75.3 K LUT, 89.0 K FF, 297 DSP (lanes 39.5 K / 232 DSP; the non-lane part
is 34.4 K at 16 / 8 and 35.8 K at 8 / 4). M8 options (`HAS_TOPK`, `HAS_DQROW`), synthesis: +6.3 K LUT (TOPK 5.6 K, DQROW
table 1.2 K), WNS +0.25 ns. Fixes the runs forced: the reciprocal seed is a ROM (a run-time divider: 63 logic levels), the
angle buffers are LUTRAM (73 K FF), delay lines up to 6 stages in flip-flops (−5.8 K LUT), the credit counters, the partial
write, the hazard comparison and the slot-FIFO head registered. The timing margin is thin (the worst path moves between runs;
hot spots: the lanes' micro-op pipeline merged by synthesis and fanned out 16 ways, the f32-to-fp16 rounding at lane stage S1,
the AMERGE batch control).

**Shrink knobs**, as elaboration parameters (results unchanged):
- `VU_FRONT_LANES` 8: halve physical front arithmetic and process each architectural line in
  two parts. Use `VU_EXP_LANES=8` and `VU_BACK_LANES=4` for the K4 fallback. Each physical
  lane l retains separate partial histories for logical lanes l and l+8, including their
  original line-index-mod-8 phases. The partial address carries the context through every
  pipeline stage. Two drain read ports reconstruct the original 16 logical words and leave
  the reduction tree/order unchanged. NEOX uses four subslots with independent output-line
  and half-line indices. Architectural lines remain 512 bits; ISA and numerical revisions
  are unchanged. The resource saving needs synthesis measurement.
- `VU_EXP_LANES` 8: −4.4 K LUT, −88 DSP; EXP and EXPSUB go to half rate.
- `VU_BACK_LANES` 4: −4.3 K LUT, −36 DSP; SILU, GELU and the others go to quarter rate.
- The earlier estimate for 16 front / 8 EXP / 4 back lanes was about 38 K LUT; measured
  full-width synthesis above exceeded those early estimates. Do not use it as the area
  prediction for the eight-front-lane fallback.

**Placement:** the SPM is in SLR1. Vivado places the SPM-facing pair-read registers and
output FIFO. K4 distributes physical lane banks across SLR0/2 behind registered pipes
(CC §14.2). The half-width banks each have two back-equipped and two front-only lanes.
The original separate-return design used about 3.4 K boundary
connections. The injector orders front and back exits so that only one is consumed per cycle.
`tsr_vu_lane_share` selects that exit beside the lanes, before the output pipe. With 16 front lanes, one 512-bit
result replaces separate 512-bit front and 256-bit live back results at BACK_LANES=8. The
512-bit reduction drain remains independent. A back-valid token travels with the inputs and
is delayed by S0 plus LB cycles, reset with the lanes. The change removes 256 data wires and
adds at most one selection wire. With eight front lanes each bank holds eight logical
partial histories and computes two pairs of the first reduction level. Its output pipe
carries 128 bits of ordinary results and 64 bits of reduction results per cycle.
The former raw drain needed 256 reduction bits per bank. The local pair selection,
staging register and FP operation use the same SUM/MAX order and pipeline depth as the
previous central first level. The central unit consumes these results on the same cycle.
The 16-front-lane configuration still returns its raw partials.
Both configurations add `2 × VU_LANE_PIPE` cycles of transport per round trip.
The K4 value is 2. The reducer and the ordinary exit path remain independent.

With eight front lanes, the input pipe carries selected A/B words and scalar
C1/C2 values. Each bank expands elementwise C values and RoPE angles beside its
lanes. The packet is 366 bits: 256 operand bits, 64 scalar coefficient bits,
32 existing control bits and 14 coefficient metadata bits. A separate 74-bit
channel writes bank-local angles. Including reset, the declared input is 441 bits
per bank, down from 545. These widths are not measured SLL counts.

Each bank stores two contexts of four C1 and four C2 words, 512 bits in total.
Existing CL1/CL2 load slots preload them over the operand payload. Arbitrary-word
scalar selection stays at the injector; elementwise expansion moves to the bank.
An explicit upper-quad bit preserves back-four subslot addressing. The single
angle generator writes each sine/cosine pair to its consuming bank. Across both
banks, angle storage remains 65,536 bits. NORM and NEOX use different ownership
and local address mappings. The full sixteen-front-lane configuration retains
its central coefficient path.

Angle writes currently go directly from the registered generator output to local
LUTRAM; they do not pass through the operand pipe. This path needs native timing
signoff. Angle-half release is delayed by `VU_LANE_PIPE` so that reuse cannot
overwrite a final bank read. No arithmetic or operand-pipe stages were added.
Directed baseline comparisons at pipe depths 0/2/3 matched output bits, order and
cycles. This is evidence for those workloads, not a guarantee that delayed angle
reuse can never stall another workload. See `docs/platform-u200.md` §8.6.12.

**Timing:**
- ≤ 6 LUT levels per stage (`docs/rtl-m2.md` §1).
- DSPs fully registered.
- No global clock enable: credits replace back-pressure.
- 512-bit buses registered once at the bank boundary and once per lane group.
- Historical reference: the earlier four-cluster link achieved 276.3 MHz. The current full
  K4 acceptance floor is 250 MHz, with legal routing and passing setup, hold and CDC checks.

**DSP use in the fp units (area measurement, 2026-11-08).** Question: does an fp unit that spends more
DSP48E2 use fewer CLB tiles under the same conditions? The placed K4 image is short of CLB tiles (SLR0 / 1 / 2
at 79.6 / 87.7 / 84.5 % with 47–58 % LUT) and has DSP to spare (33 / 2 / 43 %).

**Method.** Vivado 2024.2, out of context, `xcu200-fsgd2104-2-e`, clock 5.0 ns (the core clock), input and
output delay 2.4 ns, default `synth_design`, `opt_design`, `place_design`, `phys_opt_design`, `route_design`
for every row. The units have registered inputs and outputs. A primitive row is an array of 32 independent
instances with their own ports (`tb/fp/viv/fp_area.sh`, `tsr_fp_arr.sv`); the table shows per-instance values
after route. CLB and LUT do not change between place and route in any run. CLB is the tile count of
`report_utilization`. It also carries placement spread: unconstrained, the placer used 1.7 to 2 times LUT / 8,
and the same logic reruns within 1.3 % (16 against 32 instances: 84.8 against 84.4 CLB for the adder, 75.6
against 74.6 for `tsr_fp_add_dsp`; the define path against the module: 74.9 against 74.6). Read CLB together
with LUT. The IP rows add 98 flip-flops per instance (the wrapper registers the operands and the result) and
two cycles of latency. The IP is an area reference only and is not instantiated anywhere.

| fp add, LAT 5 | DSP | CLB | LUT | FF | CARRY8 | WNS ns | CLB saved per DSP (tiles / LUT÷8) |
|---|---|---|---|---|---|---|---|
| `tsr_fp_add` (default RTL) | 0 | 84.4 | 403 | 347 | 16 | +2.14 | |
| same RTL, `use_dsp = "yes"` on the module | 9 | 67.8 | 330 | 252 | 1 | +1.22 | 1.8 / 1.0 |
| `tsr_fp_add_dsp`: alignment shift as a one-hot product | 1 | 74.6 | 349 | 319 | 16 | +1.95 | 9.8 / 6.8 |
| `tsr_fp_add_dsp`, `NORM_DSP`: also the near-path shift | 2 | 69.0 | 326 | 303 | 16 | +1.52 | 7.7 / 4.8 (second DSP alone: 5.6 / 2.9) |
| Xilinx IP, `No_Usage`, latency 5 | 0 | 66.4 | 321 | 336 | 17 | +2.39 | |
| Xilinx IP, `Full_Usage`, latency 5 | 2 | 43.6 | 167 | 322 | 10 | +2.02 | 11.4 / 9.6 (against the IP row above) |

| fp mul, LAT 4 | DSP | CLB | LUT | FF | CARRY8 | WNS ns | CLB saved per DSP (tiles / LUT÷8) |
|---|---|---|---|---|---|---|---|
| `tsr_fp_mul` (default RTL, the `*` is inferred) | 2 | 26.2 | 104 | 163 | 8 | +2.32 | |
| same RTL, `use_dsp = "no"` | 0 | 120.6 | 679 | 242 | 53 | +0.89 | 47 / 36 (the default against this row) |
| same RTL, `use_dsp = "yes"` | 6 | 21.7 | 86 | 132 | 0 | +0.81 | 1.1 / 0.6 (for the 4 extra) |
| Xilinx IP, `No_Usage`, latency 4 | 0 | 109.3 | 537 | 536 | 78 | +2.31 | |
| Xilinx IP, `Full_Usage`, latency 4 | 2 | 27.5 | 104 | 202 | 9 | +2.46 | 41 / 27 |
| Xilinx IP, `Max_Usage`, latency 4 | 3 | 24.0 | 82 | 170 | 4 | +2.35 | 28 / 19 (third DSP alone: 3.5 / 2.8) |

At the IP's own default latency (11 for `Full_Usage`, 12 for `No_Usage` adders) the IP rows give the same picture
(47.1 against 84.1 CLB). The IP adder without DSP is already 21 % smaller than `tsr_fp_add` in CLB (66.4
against 84.4); some of the gap in the first rows is design, not DSP.

**`tsr_fp_add_dsp`** (`rtl/core/tsr_fp_add_dsp.sv`, nowhere instantiated by default) keeps the algorithm of
`tsr_fp_add` and does the far-path right shift of the small significand as `m_small * 2^k` in one DSP48E2
(24 × 17 unsigned; shifts of 17 to 25 take a second shift by 17 in fabric, shifts of 26 and more give the
sticky bit alone). The sticky bit is the OR of the shifted-out product bits, which removes the comparator chain.
`NORM_DSP` does the near-path left shift by the leading-zero count the same way (second DSP, shifts above 16 in
fabric). Same ports, same results, LAT 5 to 8, DSP input registers are [C1] and the P register is [C2], so
the latency and the cut registers are those of `tsr_fp_add`. It is bit-exact: `tb/fp/test_fp.py`
`test_fp_add_dsp` (the four `tsr_fp_add` cocotb tests at LAT 5 to 8 and `NORM_DSP`, random `ce` and gaps) and
`test_fp_sweep_add_dsp` (9.8 M sums against `libtsr_ref` through the `TSR_FP_ADD_DSP` define). Five mutants
of the window select, the sticky range and the shift split were run: four fail the test, one is equivalent (a
contribution below a quarter ulp never changes the rounded result).

**Whole VU, K4 configuration** (`tb/vu/viv/vu_ooc.sh`: `FRONT_LANES=8 EXP_LANES=8 BACK_LANES=4 LANE_PIPE=2
SPM_RD_LAT=4 NUM_CH=4`, 5.0 ns, place only, no route). The define `TSR_FP_ADD_DSP=1` (`=2`: with `NORM_DSP`)
makes every `tsr_fp_add` an instance of `tsr_fp_add_dsp`; no filelist sets it. The K4 VU has 35 fp adders
(8 front-lane FA, 8 back-lane ONE, 4 bank-reduction, 8 reduction and post, 7 in the angle unit) and 47
multipliers. Two clock periods (5.00 and 5.02 ns) show the rerun spread.

| tsr_vu | DSP | CLB | LUT | FF | CARRY8 | WNS ns | CLB saved per DSP (tiles / LUT÷8) |
|---|---|---|---|---|---|---|---|
| default RTL | 249 | 12,645 (5.02 ns: 12,611) | 58,273 | 68,841 | 1,825 | +0.76 (+0.70) | |
| `TSR_FP_ADD_DSP=1` | 284 | 12,282 (12,388) | 56,900 | 67,962 | 1,825 | +0.52 (+0.77) | 10.4 (6.4) / 4.9 |
| `TSR_FP_ADD_DSP=2` | 315 | 12,564 | 56,624 | 67,466 | 1,825 | +0.71 | 1.2 / 3.1 |
| `use_dsp = "yes"` on `tsr_fp_add` | 543 | 12,517 | 56,850 | 65,968 | 1,328 | +0.63 | 0.4 / 0.6 |

The default RTL reruns within 0.3 % in CLB; a changed netlist moves CLB by up to 2 % (the `=1` and `=2`
rows differ by 2.3 % in CLB but only 0.5 % in LUT). Use the LUT column for the trend and the CLB column as
the tile count that the placer produced.

**Result.**
- Yes: every DSP variant uses fewer CLB tiles than the default at the same function and the same latency,
  and meets 5.0 ns (primitives routed, WNS +0.8 to +2.5 ns; VU placed, +0.5 to +0.8 ns). The saving per DSP
  depends on what the DSP replaces.
- A 24 × 24 mantissa product: 36 to 47 CLB per DSP. Keep every multiplier in DSP.
- A shifter done as a one-hot product, the first DSP of an fp adder: 7 to 10 CLB per DSP (4.9 to 10.4 in the
  VU). The second DSP of the adder (the near-path shift) saves 3 to 6.
- Carry-chain adders moved by `use_dsp = "yes"`: 1 to 2 CLB per DSP and 0.9 ns (adder) to 1.5 ns (multiplier)
  less slack. Not worth the DSP.
- The Xilinx adder IP with two DSP reaches 10 to 11 CLB per DSP. An fp adder is mostly shifters, exponent
  and rounding logic; a DSP takes only the shifts.
- Whole VU: 35 DSP (alignment shift) save 2.3 % of the VU's CLB tiles (about 220 to 360 tiles, 0.2 % of the
  device) and 2.4 % of its LUT. This does not relieve the CLB pressure by itself.
- `NORM_DSP` and `use_dsp = "yes"` in the VU cost 31 and 259 more DSP than the alignment shift alone and
  save only 0.3 K and 0.05 K LUT more than it.

Reproduce (all Vivado runs through the slot wrapper):
`hw/u200/vivado_slot.sh tb/fp/viv/fp_area.sh build/fp_area/add_dsp kind=add_dsp` (kinds `add`, `mul`, `add_dsp`,
`add_dsp2`, `ip_add`, `ip_mul`; keys `usedsp=yes|no`, `ipdsp=`, `iplat=`, `n=`, `define=`),
`tb/fp/viv/fp_area_table.py build/fp_area` for the table, and
`hw/u200/vivado_slot.sh tb/vu/viv/vu_ooc.sh build/vu_k4 place_only clk=5.0 iodly=2.4 FRONT_LANES=8 EXP_LANES=8
BACK_LANES=4 LANE_PIPE=2 SPM_RD_LAT=4 NUM_CH=4 [define=TSR_FP_ADD_DSP=1] [usedsp=tsr_fp_add]`.
The VU unit tests pass with the define (`tb/vu/test_vu_unit.py`: half_width, lane_pipe, u200 and knobs with
`=1`, half_width and lane_pipe with `=2`; for the run, add `tsr_fp_add_dsp.sv` to `VU_SOURCES` and
`+define+TSR_FP_ADD_DSP=<n>` to `rtl/filelist.f`, and revert both after).

---

### 7.11 Local attention and MVM epilogue consumers

The optional local-intermediate configuration reuses eight front/EXP and four
back lanes as two banks of four front/EXP and two back lanes. A bank serves
the two owners in its SLR (0/2 or 1/3). It does not add an EXP or TANH datapath.
Normal VU work drains before local ownership changes. Local work reserves
return capacity before injection; ASMAX and MVM epilogues arbitrate for the
same lanes and may not both consume a return.

The local ASMAX sequencer reads/writes its owner's score/P bank and receives
the global row maximum (AT §5.5a). Its compact command is 54 bits per bank.
QK sends scalar maxima through a balanced registered tree with bounded
credits. Local scalar reduction preserves the existing MAXT special-value
and tie behavior. Generic VU input/reduction paths remain for old operations.

A completed dense MVM job can instead enter a 16-slot ordered local epilogue
queue per pair before its final gather transfer. SCALE uses the existing FP32 multiply
and its specified rounding. SOFTCAP uses FP32 multiply, the existing TANH,
then FP32 multiply with the two original graph coefficients. It preserves
all three rounding boundaries; it is not a fused multiply/tanh approximation.
Four front lanes take four issue cycles per 16-value line for each scale;
two back lanes take eight for TANH. These are issue-rate bounds, not measured
end-to-end latency. Tail masks remain attached to the job and inactive output
words stay zero.

Raw MVM `w0[13]` selects local SCALE, with normal `w6` destination stride and
`w7` FP32 scale. `w0[14]` selects SOFTCAP, with `w6` first scale, `w7` final
scale and implicit dense SPM destination stride. The bits are mutually
exclusive and incompatible with ACC; they require capability bits 13 and 14
respectively. Legacy MVM encodings retain their meaning. Firmware still
honors the public output tensor stride when storing the final result to DDR.
No whole-N local allocation is required: queue backpressure bounds resident
completed output, while the ordinary MVM command and final completion remain.

K4 transports each paired gather job as two 281-bit flits instead of one
562-bit packet. The queue, pair arbitration and epilogue endpoint form a whole
functional source block next to its consumers. A central receiver reassembles
complete jobs and exposes none after only the first flit. Source acceptance
reserves full-job storage until the second flit transfers. Both pairs together
still deliver at most one result line per cycle; one pair alone delivers at
most one line per two cycles. Reset discards partial packets. This physically
narrows the shared result interface, including legacy clients. It is not a
claim that routed SLL counts halve. QA broadcast and generic VU interfaces
remain because their dense and fallback clients still need them.

The candidate Pblocks group whole collectors, local attention engines and VU
banks with clusters 0/2 in SLR0 and 1/3 in SLR2. The receiver and remaining
shared logic remain free to place. This is major-block floorplanning only;
ordinary synthesis/implementation must establish memory mapping, timing and
routing. No cell/stage placement or optimizer ownership constraint is used.

---

### 7.12 Fused GDN state update (GDN, opcode 0x39; `HAS_GDN`)

Status (2026-11-08): the op is integrated in `tsr_vu` behind `HAS_GDN` (default 0: then no decode, no engine, no lane mux,
and every other op runs cycle for cycle as before). `CAPS` bit 8. The C reference `sw/sim/tsr_ref_gdn.c`, the simulator's VU
unit (`sw/sim/tsr_vu.c`), the engine `rtl/core/tsr_vu_gdn*.sv` and the VU agree bit for bit (§7.12.10). The `tsr_u200_k4` top
has `HAS_GDN` 1 on branch `gdn-vu`; no image with it has run on the card. The card test is the `selftest` kernel's id 11
(`tsr-selftest --ids 11`). Decisions VU-28 … VU-37 in §11.

The element update runs on the VU's front lanes, which are exactly the structure it needs: FM_A and FM_B in parallel into FA
(VU-30, VU-34). The engine has no update array of its own. What stays in it is what the lanes do not have: the DSP dot array, the
converter, the setup engine, the scalar chain and the row and table buffers. The engine is inside `tsr_vu`, so it lands in
`pblock_tsr_vu` (SLR2) beside the lanes. Nothing goes to SLR1 (VU-31).

#### 7.12.1 The op

Gated DeltaNet layers keep a recurrent state S per head, d_v × d_k f32. For one token and one head, with
k and q (d_k, L2-normalised by earlier ops), v (d_v), and the scalars α = exp(g) and β:

```
for each row i (rows are independent):
    u_i  = S_i · k
    e_i  = β (v_i − α u_i)
    S'_i = α S_i + e_i k            (written back)
    o_i  = S'_i · q
```

The planned model has 48 layers, 48 heads and d_k = d_v = 128. One head is 64 KiB, a layer 3 MiB, a
sequence 144 MiB. A decode token reads and writes all of it: 37,748,736 state elements, 151 MB each way.
Composed from VOP and RED, the recurrence needs four passes over S (decay, S·k, update, S'·q). The
fused op reads and writes S once. The pinned llama.cpp computes the same recurrence (`build_delta_net_autoregressive`
in `src/models/delta-net-base.cpp`, and `ggml_compute_forward_gated_delta_net` in `ggml-cpu/ops.cpp`: decay S, delta = β
(v − S·k), S += k δ, o = S·q, a row of S being S_v contiguous words dotted with k and q). The CPU kernel
scales o by 1/√S_v; the graph version scales q. This op takes q as given.

Throughput target: 8 state elements per cycle (one front-lane beat). At 250 MHz that is 19.0 ms per token for the planned
state; at 200 MHz 23.8 ms. The measured cost per head is (rows + 1) × m / 8 cycles when heads follow each other (§7.12.8).
With 16 front lanes the same op runs at 16 elements per cycle.

#### 7.12.2 Operands and where they live

| Operand | Form | Where |
|---|---|---|
| S | d_v rows of m = d_k f32, dense (row stride m / 16 lines), updated in place or to another tile | SPM tile, loaded by `LD` from the state tensor in DRAM, stored back by `ST` |
| k, q | m f32 each | SPM vectors (earlier VU ops L2-normalise them) |
| v | d_v f32 | SPM vector |
| o | d_v f32 | SPM vector, tail rule of §4.2 |
| α, β | f32 | s registers (`SOP.EXP` gives α = exp(g)) |

The state tile comes through the SPM and the existing DMA, not through a direct stream (VU-31).
- Traffic per element at 8 elements per cycle: LD writes 4 B, the op reads 4 B and writes 4 B, ST reads 4 B. The SPM
  carries 64 B per cycle each way, half of the 128 B per cycle each way of the 2R + 2W SPM (CC §10). The GDN phase of a layer has
  no competing MVM work (the next layer waits for its output). DRAM traffic is 302 MB per token, 12.6 GB/s at 24 ms.
- The op reads the state tile of a head once, one SPM line per cycle, and writes S' one line per two cycles (the lanes' rate).
  The op's reads go through the VU's own reader (`tsr_vu_rd`), its writes through the VU's writer (`tsr_vu_wr`).
- A head tile is 64 KiB. Three tiles (LD of the next head, the op, ST of the previous one) are 192 KiB of the 4 MiB SPM.
- A direct DDR to unit to DDR stream would need a new DRAM path, an arbiter against the weight streams and an SLR crossing. It
  saves traffic that the SPM carries without trouble, and it would not be a tile op that firmware can order with the checker,
  queues and hazards.

#### 7.12.3 Numerics

All f32 operations are the VU primitives (§6.1): DAZ, RNE, FTZ, canonical NaN. The dot products are not
f32 sums. They are exact sums of integer products in a block frame, converted to f32 once.

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

- **table(x)** (m ≤ 256 elements). DAZ. E_max is the largest biased exponent among the nonzero elements
  (0 if none). Entry j is sign × ⌊(M_j + 2^(sh−1)) / 2^sh⌋ with M_j the 24-bit significand, sh = E_max − E_j
  (sh = 0 keeps M_j; entries are 0 for zero elements and for sh ≥ 25). Entries are below 2^24 in
  magnitude; entry × 2^(E_max − 150) is the element. The table is the same for every row of the head.
- **rowdot(row, T)**. E_s is the largest biased exponent of the row's nonzero elements. Element j becomes
  A_j = sign × ((M_j << 2) >> (E_s − E_j)), a 26-bit window below the row's frame, truncated (0 for a shift
  of 26 or more). I = Σ_j A_j T_j is computed exactly (|I| < 2^58). The result is the f32 nearest to
  I × 2^(E_s + E_max,T − 302): RNE, overflow to ±Inf, a result below 2^-126 − 2^-150 flushed to a signed zero
  (the rule of FM), I = 0 gives +0.
- **Order does not matter.** The sum is an integer sum, so lane count, beat order and accumulation order
  change nothing. There is one rounding per dot product.
- **Inf and NaN.** Any Inf or NaN among a row's elements, or among a table's elements, makes the rowdot
  result the canonical NaN. A bad state row poisons that row's e, S' and o. A bad k or q poisons the head.
- **κ from a pseudo row.** The op takes κ = rowdot(k, TQ), the row dot product of the vector k itself
  with TQ. The hardware streams k through the dot array as an extra first row; its u' is dropped.
- **The update on the lanes is bit-exact with the reference.** The lane micro-op U_ROT computes `FA(FM(a, c1), FM(b, c2))`. With
  a = S_ij, c1 = α, b = k_j, c2 = e_i it is `FA(FM(S_ij, α), FM(k_j, e_i))`, and `FM` is commutative, so it equals
  `FA(FM(α, S_ij), FM(e_i, k_j))` of the reference for all inputs, NaN and Inf included (`tsr_fmul` is symmetric in its operands).

The C reference is `tsr_ref_gdn` (`sw/sim/tsr_ref_gdn.c`, header `tsr_ref_gdn.h`) with the pieces
`tsr_ref_gdn_table`, `tsr_ref_gdn_rowdot`, `tsr_ref_gdn_fx2f`, `tsr_ref_gdn_e`, `tsr_ref_gdn_o` and
`tsr_ref_gdn_upd_elem`. Row bound: with a 26-bit window and 24-bit entries, 256 terms stay below 2^58, so
d_k ≤ 256 (the 48-bit DSP accumulators are exact for the two 13-bit pieces of an entry, §7.12.5).

Differences from the CPU kernel: it decays S in place and then dots (ours folds α β into k), it sums in
whatever order `ggml_vec_dot_f32` uses, and its `ggml_vec_mad_f32` may contract to an FMA. All of these
are rounding differences of a few ulp; §7.12.6 measures them against float64.

#### 7.12.4 Encoding

`GDN` = 0x39 (the first of the opcodes reserved for VU extensions, EX §7.2), unit 3, `CAPS` bit 8 (`HAS_GDN`; an op on a device
without the bit completes ILLEGAL_OP on SYNC like every opcode whose CAPS bit is 0).

| Field | Bits | Meaning |
|---|---|---|
| opcode | w0[7:0] | 0x39 |
| — | w0[15:8] | 0 |
| sa | x16[3:0] | s register holding α |
| sb | x16[7:4] | s register holding β |
| — | x16[15:8] | 0 |
| d | w1[23:0] | S' tile, line of row 0 (may equal a: in place) |
| a | w2[23:0] | S tile, line of row 0 |
| k | w3[23:0] | k vector line (m words) |
| q | w4[23:0] | q vector line (m words) |
| v | w5[23:0] | v vector line (rows words) |
| o | w6[23:0] | o vector line (rows words, tail rule of §4.2) |
| m | w7[15:0] | d_k: a multiple of 16, 64 ≤ m ≤ 256 |
| rows | w7[31:16] | d_v, 1 ≤ rows ≤ 1,024 |

Rows of the tile are dense, stride m / 16 lines, for a and d. One op is one head; firmware issues 48 per layer and overlaps
`LD` and `ST` with it. Checks, in this order (the simulator's `chk_gdn` and the RTL class in `tsr_vu_chk` give the same code,
ERR_INFO and extents, tested bit for bit):
- `ILLEGAL_OP`: any reserved bit (`TSR_GDN_RSV_W0` … `W7` of the generated table).
- `OPERAND` (ERR_INFO = the raw field): m not a multiple of 16 or outside [64, 256] (m); rows 0 or above 1,024 (rows); d ≠ a with
  overlapping extents of d and a, E(rows, m / 16, m / 16) lines each (d); o overlapping the extents of d or a (o).
- `SPM_RANGE` (ERR_INFO = the start line), the six range operands in field order d, a, k, q, v, o: d and a E(rows, m / 16,
  m / 16) lines, k and q m / 16 lines, v and o ⌈rows / 16⌉ lines.

The lower limit of m is 64 for every build (4 beats of the 16-lane build). Vectors and the tile may overlap d or o: the op reads
k, q and v completely before it writes a word. The extent list of a passing op has four SPM entries: d W, a R, k R, o W (CC §8.2:
four slots; every written range is listed, the race detector reads them), and the s registers sa and sb.

#### 7.12.5 Microarchitecture

```
 SPM ──rd──► inj ──A line──► gen slots: CFG, K, Q, V lines, pseudo row (k), rows of S
                │
                │ gdn_v + slot + line (once per slot, registered at the injector output)
                v
        ┌── GDN engine (tsr_vu_gdn) ───────────────────────────────────────────────────────────┐
        │ vector line buffer ─► setup engine (1 FM): KF RAM (k), T RAM (TK, TQ), bv RAM, scratch │
        │ ingest (16 words per cycle): B1 RAM ─► dot array: 8 lanes, align, 4 DSP48E2 cascades   │
        │                       B2 RAM (copy)     │ accumulators (36 DSP48E2), converter         │
        │                          │              v                                              │
        │                          │        scalar chain (1 FM, 1 FA): e = FS(bv, u'), o         │
        │                          v              │ e                      o ─► o line buffer ──┼─► writer, result port
        │         update reader: S beat (B2) + k beat (KF) + α + e + destination line            │
        └──────────────┬────────────────────────────────────────────────────────────────────────┘
                       │  u_v, u_s, u_k, u_alpha, u_e, u_line, u_part, u_tag
                       v
        lane input mux (tsr_vu): U_ROT, a = S, b = k, c1 = α, c2 = e, descriptor ──► front lanes (FM_A ∥ FM_B → FA)
                                                                                  ──► tsr_vu_wr: assembler, FIFO, SPM
```

**The slots (`tsr_vu_gen`, kind `SK_GDN`).** An op GDN makes, in this order: one CFG slot (m, rows, the o line; α and β come from
the op constants), the lines of k, q and v (`GS_VK`, `GS_VQ`, `GS_VV`, the last line of v with a tail of rows mod 16 words), the m / 16
lines of k again (the pseudo row, `GS_ROW`), and rows × m / 16 lines of S (`GS_ROW`, one slot per line). Their SPM reads go
through `tsr_vu_rd` like every operand. The injector (`tsr_vu_inj`) takes a slot when the engine says it can accept it:
`cfg_ok` for the command, `vec_ok` for a vector line, `row_ok` for the first line of a row. A command and a vector line keep the
injector four cycles (the engine's line buffer is free again before the next vector slot), a row line one. The engine sees
`gdn_v` once per slot with the registered slot (`topk_slot`) and line (`pa_data`). The slots count the op's writes (one per S' line,
one more per ⌈rows / 16⌉ o line) so that the op's done waits for the lanes' results.

**Setup engine** (`tsr_vu_gdn_coef`). One word per cycle in, in the order k, q, v, from a 16-word line buffer. One shared FM
computes ab, kp_j and bv_i. It writes the raw key to the KF RAM, keeps kp and q in a scratch RAM with their largest exponents and
Inf/NaN flags, then aligns both to their frames (round half up) and writes the table words. About 4 m + rows + 45 cycles per
head (735 for 128 × 128). Two banks hold the coefficients of two heads: key, tables, α, κ, bv. The setup of head h + 1 runs
while the rows of head h flow.

**Ingest.** A row line (16 words) per cycle goes to both row buffer copies, two beats per RAM word at 8 lanes (the low half
is beat 2i, the high half beat 2i + 1, one address per line). The exponents of each line go through a pipelined maximum tree
(four levels for 16 words) into the row's E_s. E_s = 255 is the row's bad flag. The row is complete a few cycles after its last
line. The ingest runs at twice the rate of the dot array and the update (two beats per cycle against one), so the rows of the
next head wait in their slots while its setup runs (VU-36).

**Slots and credits.** 64 row slots (`GDN_NSLOT`) hold rows from the first ingest line until the update is launched. A slot
is a region of NB_MAX beats (32 at 8 lanes: 16 RAM words) in each RAM copy. The first line of a row takes a credit; the credit returns when the row's update is
launched. `row_ok` asks for two free slots, because the injector may have fired a first line (at most every four cycles) that
the engine has not counted yet.

**Dot array** (`tsr_vu_gdn_dot`). Lane l holds elements j ≡ l (mod 8) of every row (RAM B1) and of both tables (RAM T). A launch (slot,
bank, E_s, beats) starts lane 0 one cycle later and lane l, l cycles after that: the descriptor travels down a register
chain. Per beat a lane reads its word, aligns it (a 26-bit right shift in two LUT stages), applies the sign as `(a ^ s) + s`
(the pre-adder of the DSP) and multiplies by the four table pieces: high piece (12 bits signed) and low piece (13 bits) of the K'
entry and of the Q entry. The four products go into four DSP48E2 cascades (`tsr_vu_gdn_mac`): P_l(t+1) = P_(l−1)(t) +
product_l(t). Because the lanes start one after another, the cascade adds the 8 lanes' products of one beat with no fabric adder
and no skew registers. One accumulator slice per chain adds the beats of the row. After the last beat the four 48-bit sums
(I = hi · 2^13 + lo) are valid for one cycle. Everything is exact: |P| < 2^47 for 256 elements.

**Converter** (`tsr_vu_gdn_fx2f`). One pipeline of 8 stages (add the pieces, sign and magnitude, leading zeros, two shift stages,
round, pack) converts the K sum and, one cycle later, the Q sum.

**Scalar chain.** One shared FM and one shared FA, FIFOs between them. u' and a read of bv_i give e = FS(bv_i, u') (priority on the
FA); w gives FM(α, w); e gives FM(e, κ); the two products give o. The pseudo row's w becomes κ of the bank. The o words of a head
are collected in lines of 16 (the words past `rows` are 0) and offered to the writer's result port (the one TOPK uses; an op GDN
is alone, so TOPK is idle). Each o line is one write of the op.

**The update on the lanes** (`tsr_vu_gdn_upd` and the mux in `tsr_vu`). A row whose e is known and for which the writer has
room is launched. The launch takes its ⌈m / 16⌉ lines from the output FIFO of the writer (`oc_take`, up to 16 lines at once)
and reads the row again from RAM B2 together with the key from RAM KF, a beat per cycle. The beat reaches the lane inputs
with `uop = U_ROT`, a = S, b = k, c1 = α, c2 = e, `neg` 0 (an FA), in the lane cycle where the injector's own operands would be, and
its write descriptor (tag, SPM destination, line, half-line, 16 words) takes the injector's place on the descriptor path to the
writer. The select is a registered level of the engine, replicated per lane; no other lane input has two sources in any cycle, because an op GDN is held back
from every other op (VU-35). The writer assembles two beats into a line, as for any front op, and the lane exit order holds because only the
engine feeds the lanes. The lanes have no back pressure and none is needed: the credits were taken at the launch.

**Back pressure.** There is none on the lanes. The injector keeps to `cfg_ok`, `vec_ok` and `row_ok`; the launch waits for output
credits (`oc_ok`: 16 free lines, registered); a slow SPM write slows the launches and then the ingest.

**RAM styles and the buffer choice.** Every buffer is a RAM with `ram_style = "ultra"` (`tsr_vu_gdn_ram`, parameter `GDN_STYLE`; "block" maps the same
RAMs to RAMB36): per lane the row buffer B1 (dot pass), its copy B2 (update pass), the table RAM T and the key RAM KF, plus
the scratch RAM and the bv RAM. Two copies of the row are needed because three accesses per cycle (ingest write, dot read,
update read) exceed the two ports of a URAM. A lane's row RAM word is 64 bits wide (two beats), 1,024 words deep for 64 slots of 16 words: a quarter of a URAM
(4,096 × 72). The URAM count is therefore set by the number of RAMs (4 per lane, the scratch RAM and the bv RAM: 34) and not by their size;
the slots could grow to 256 at no cost in URAM. The VU's Pblock holds URAM (§7.12.9), so URAM is the choice; the block RAM form
needs about 43 RAMB36 (counted from the RAM sizes) and no more fabric.

**Parameters.** `HAS_GDN`, `GDN_NSLOT` (64; a power of two ≥ 4), `GDN_STYLE`. `FRONT_LANES` is the engine's `LANES` (8 or 16);
`NB_MAX` = 256 / `LANES`. Latencies `RD_LAT` 3, `FM_LAT` 4 and `FA_LAT` 5 follow the lanes' primitives.

**DSP48E2 by inference.** The dot array needs no DSP48E2 primitive. Vivado 2024.2 maps the written
registers and the pre-adder, multiplier and cascade adder into the slices with `PCIN` cascades and
accumulators (`(PCIN+((D'+A2)*B'')')'`, no fabric adder, `CARRY8` count 0 in `tsr_vu_gdn_dot`), provided
that all lanes of a chain are in one module: when the P registers cross module boundaries Vivado uses
the C port and keeps the accumulators in fabric adders (found in the first build, 32 DSP and 24 CARRY8).
The unisim `DSP48E2.v` model does not compile in Verilator 5.046 (`Unsupported: tristate in top-level IO`,
and it needs `glbl`); it elaborates and runs in XSim (checked with `P = A × B` and a 100 ns GSR wait). The
repo's netlist simulations use it that way (`tb/tech/viv/net_sim.sh`). Since inference works, the RTL
is plain SystemVerilog that Verilator simulates directly.

#### 7.12.6 The dot products: f32 adder trees (A) or wide fixed point in the DSP (B)

Each row needs two d_k-long dot products (S_i · k and S_i · q), and the head needs a third (k · q).
At 8 elements per cycle that is 16 products and 16 additions per cycle, in a design where the f32 adders
are the CLB-heavy part (§7.10: 84 CLB for `tsr_fp_add`, 75 with the DSP shifter, 66 for the Xilinx IP
without DSP; a DSP replacing a shifter saves 10 CLB, a DSP replacing a 24 × 24 mantissa product saves 47).

- **(A) f32 adders in a fixed tree.** 8 multipliers, one accumulator adder per lane (lane l sums elements
  l, l + 8, … in order), then ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)) on one shared adder. Per dot product:
  8 `tsr_fp_mul` and 9 `tsr_fp_add`; the loop of an accumulator needs 5 interleaved rows (adder latency 5).
  Two engines (u', w): 16 multipliers and 18 adders.
- **(B) Exact integer sums in DSP48E2 accumulators.** Align each row element to the row's largest exponent
  (a 26-bit window; the table is aligned once per head), multiply by the table entry (a 26 × 13-bit product
  per piece, two pieces per entry), add in the DSP cascade, convert once. No f32 adder at all. The cost is
  the aligner (a 26-bit shifter per lane), the pre-adder sign, the converter (once for all lanes) and two
  RAM reads per beat.

**Accuracy.** Script `tb/gdn/gdn_accuracy.py` (PyTorch float64 reference; run with `build/tools/torchvenv`).
`dot`: 200 trials of a 128 × 128 state against a unit k, error relative to |S_row| |k| (one f32 ulp of a
result is 6.0·10⁻⁸ of its own magnitude); max over rows and trials, (median of the per-trial maxima).

| data | f32 matmul | A: f32 tree | **B: 26 / 24** | B: 24 / 24 | B: 26 / 20 | B: 26 / 18 | B: 26 / 16 |
|---|---|---|---|---|---|---|---|
| gauss | 4.2·10⁻⁸ (2.4·10⁻⁸) | 4.0·10⁻⁸ (2.5·10⁻⁸) | 4.2·10⁻⁸ (1.9·10⁻⁸) | 5.6·10⁻⁸ (3.2·10⁻⁸) | 4.9·10⁻⁷ (2.3·10⁻⁷) | 2.2·10⁻⁶ (8.7·10⁻⁷) | 9.1·10⁻⁶ (3.5·10⁻⁶) |
| heavy | 6.1·10⁻⁸ (2.8·10⁻⁸) | 6.1·10⁻⁸ (2.9·10⁻⁸) | 6.3·10⁻⁸ (3.6·10⁻⁸) | 2.4·10⁻⁷ (1.3·10⁻⁷) | 4.4·10⁻⁷ (2.0·10⁻⁷) | 1.7·10⁻⁶ (8.0·10⁻⁷) | 6.8·10⁻⁶ (3.4·10⁻⁶) |
| log-uniform | 6.2·10⁻⁸ (2.5·10⁻⁸) | 5.5·10⁻⁸ (2.5·10⁻⁸) | 4.5·10⁻⁸ (2.6·10⁻⁸) | 1.4·10⁻⁷ (7.7·10⁻⁸) | 4.8·10⁻⁷ (2.1·10⁻⁷) | 2.0·10⁻⁶ (8.3·10⁻⁷) | 7.8·10⁻⁶ (3.2·10⁻⁶) |
| orthogonal | 3.5·10⁻⁸ (2.0·10⁻⁸) | 3.9·10⁻⁸ (2.2·10⁻⁸) | 3.5·10⁻⁸ (1.4·10⁻⁸) | 5.5·10⁻⁸ (3.0·10⁻⁸) | 5.1·10⁻⁷ (2.3·10⁻⁷) | 2.0·10⁻⁶ (9.5·10⁻⁷) | 7.7·10⁻⁶ (3.5·10⁻⁶) |

Columns: the f32 matmul of numpy, scheme (A), and scheme (B) with the row window / table entry widths in bits.

(B) with a 26-bit row window and 24-bit table entries is as accurate as the f32 tree and as numpy's matmul
and slightly better on wide exponent spreads and near-orthogonal rows (one rounding instead of seven).
A 24-bit window costs a factor 1.3 to 4 on the harder data, 20-bit entries a factor 7 to 12, 18-bit entries 30 to 50,
16-bit entries more than 100. That settles the widths of VU-29: 26 / 24 bits. They also fix the DSP use (a 24-bit
entry is two 13-bit pieces, so 4 DSP per lane for the two tables) and the accumulator bound (§7.12.3).

`system`: a head of 128 × 128 over 32,768 steps with gating, six input scenarios (α ≈ 0.9, 0.99 and
0.9999 per token with log-normal variation; correlated keys; two key channels 30 × larger and
heavy-tailed v; v of order 10³). Same inputs for all runs; the reference is the plain recurrence in float64
from the same f32 inputs. "lit32" is the textbook recurrence in PyTorch float32 (`torch.mv` for both
dots). Relative error ‖o − o_ref‖ / ‖o_ref‖ per step and ‖S − S_ref‖_F / ‖S_ref‖_F after 32,768 steps, in
units of 10⁻⁷ (one f32 ulp is 0.6 · 10⁻⁷):

| scenario | o, median over steps (A / B / lit32) | o, worst step (A / B / lit32) | S after 32,768 steps (A / B / lit32) |
|---|---|---|---|
| fast | 1.36 / 1.25 / 1.54 | 5.10 / 4.73 / 4.92 | 0.90 / 0.89 / 0.92 |
| typical | 2.43 / 2.37 / 2.53 | 4.30 / 4.11 / 4.33 | 2.25 / 2.26 / 2.26 |
| slow | 4.88 / 4.84 / 4.95 | 7.70 / 7.22 / 7.22 | 4.78 / 4.79 / 4.76 |
| correlated | 2.49 / 2.43 / 2.58 | 4.30 / 4.14 / 4.20 | 2.33 / 2.33 / 2.33 |
| outliers | 2.50 / 2.43 / 2.61 | 8.02 / 8.91 / 8.37 | 2.19 / 2.14 / 2.24 |
| large | 2.43 / 2.37 / 2.53 | 3.91 / 3.97 / 4.37 | 2.25 / 2.25 / 2.26 |

All three agree to a few percent. The error floor of 2 to 5 · 10⁻⁷ (4 to 8 ulp) is the f32 rounding of the
state itself, which every scheme has, and it does not grow over the run (the error of S after 8,192, 16,384,
24,576 and 32,768 steps is flat). (B) is marginally the best (1 to 3 % below (A) in the median). The
way the dots are summed is therefore not what limits the accuracy, so the cheaper scheme can be chosen on area.

**Area of the dot array**, measured out of context with the method of §7.10 (xcu200-fsgd2104-2-e, 5.0 ns, place and route; CLB
tiles carry the placer's spread, read them with the LUT count). One engine of (A) (`tb/gdn/viv/tsr_gdn_dota.sv`, an area
reference, 8 `tsr_fp_mul`, 8 lane accumulators and 1 shared tree adder `tsr_fp_add`): 1,041 CLB, 4,568 LUT, 5,203 FF,
208 CARRY8, 16 DSP. Two engines (u' and w): 2,082 CLB, 9,136 LUT, 32 DSP. Scheme (B), the same job for both dots
(`tsr_vu_gdn_dot`, 8 lanes with their row buffers and tables, plus `tsr_vu_gdn_fx2f`):

| | CLB | LUT | FF | CARRY8 | DSP | URAM | WNS ns |
|---|---|---|---|---|---|---|---|
| (A) one engine, 8 `tsr_fp_mul` + 9 `tsr_fp_add` | 1,041 | 4,568 | 5,203 | 208 | 16 | 0 | +2.32 |
| (A) two engines (u', w), twice the line above | 2,082 | 9,136 | 10,406 | 416 | 32 | 0 | |
| (B) dot array, 8 lanes (36 DSP48E2 chains and accumulators, B1 and T RAMs) | 258 | 877 | 989 | 0 | 36 | 16 | +1.84 (placed) |
| (B) converter `tsr_vu_gdn_fx2f` | 94 | 415 | 536 | 22 | 0 | 0 | +2.46 |
| (B) total | 352 | 1,292 | 1,525 | 22 | 36 | 16 | |

The dot products of (B) take 17 % of the CLB of (A) and 14 % of the LUT, with 4 more DSP and the 16 URAM that hold the row
and the tables (the row buffer is needed by (A) as well, in LUTRAM or URAM).

**Decision VU-29:** (B). The same accuracy as (A), no f32 adder in the dot products: 352 CLB and 36 DSP instead of 2,082 CLB
and 32 DSP.

#### 7.12.7 The element update runs on the front lanes (VU-30, VU-34)

The element update is `FA(FM(α, S), FM(e, k))` on `tsr_fp_mul` and `tsr_fp_add`. A front lane has FM_A and FM_B in parallel into
FA for ROPE and AMERGE (U_ROT, §7.2). The GDN update is U_ROT with the operands of §7.12.5. There is no update array in the
engine: the first design's array was 61 % of the engine's LUT (4,786 of 7,784, 32 of 72 DSP) and is now the lanes the VU
already has. The front lanes (8 at K4) take one beat of 8 elements per cycle, the rate of the dot array.

A block-frame update in the DSP was measured against the f32 update (`gdn_accuracy.py update`, ‖S − S_ref‖_F / ‖S_ref‖_F after
8,192 steps):

| α per token | f32 update | block frame, 26-bit window, rounded | truncated | 24-bit window, truncated |
|---|---|---|---|---|
| ≈ 0.9 | 0.9 · 10⁻⁷ | 2.2 · 10⁻⁷ | 2.3 · 10⁻⁷ | 4.2 · 10⁻⁷ |
| ≈ 0.999 | 4.1 · 10⁻⁷ | 5.2 · 10⁻⁷ | 8.7 · 10⁻⁷ | 1.0 · 10⁻⁵ |
| ≈ 0.99999 | 4.8 · 10⁻⁷ | 7.5 · 10⁻⁷ | 1.4 · 10⁻⁶ | 1.5 · 10⁻⁵ |

The best variant is 1.3 to 2.5 times worse, the cheaper ones up to 30 times, and it needs a second set of row shifters and a
normaliser per lane. Not chosen. The element update remains f32 and uses the VU's primitives, so `TSR_FP_ADD_DSP` of §7.10 applies to it
unchanged.

#### 7.12.8 Throughput and latency

Measured on the RTL (`tb/gdn/test_gdn.py` gdn_unit_rate, the engine with a lane model; `tb/vu` vu_gdn on the VU), cycles are exact:

- **Rate.** One beat of 8 words per cycle in the dot array and the update, whatever the shape (m = 64, 128, 256). Heads of one
  shape follow each other with no gap in the launches: 128 × 128 heads take (rows + 1) × m / 8 = 129 × 16 = 2,064 cycles each (the
  pseudo row is the +1, 0.8 %). The planned model: 48 × 48 heads × 2,064 = 4.76 M cycles = **19.0 ms per token** at 250 MHz
  (23.8 ms at 200 MHz). With 16 front lanes: half. Mixed shapes: eight heads (4 × 128 × 128, 2 × 64 × 256, 2 × 128 × 64) take 15,555 cycles
  for 14,480 beats, the S' lines leave over 14,737 cycles (98 %); with 16 lanes 11,259 cycles, S' over 10,449.
- **Setup** of a head: about 4 m + rows + 45 cycles (128 × 128: 735 including the vector delivery), overlapped with the rows of the
  previous head. The ingest takes two beats per cycle, so up to 63 rows (1,008 beats for 128-wide rows) wait in the slots while the
  next head sets up; that is more than the setup (VU-36). A head after an idle engine pays the setup once: about 4 m + rows + 45
  cycles, then 66 cycles from the tables to the first launch.
- **Latency.** From the launch of a row to its S' line in the SPM: RD_LAT + 1 for the operands, then the lane latency (`TSR_VU_LF` = 42 plus the lane-pipe stages)
  and the writer.
- **Rows in flight and slots:** (3 NB + 50) / NB rows are between the first ingested line and the launch: 6.1 for m = 128. 64 slots hold
  them with a wide margin.
- **Credits.** A launch needs 16 free lines in the 64-line output FIFO and takes m / 16 of them: the lanes' results drain at one
  line per two cycles, so at most about 27 lines are held at once.

#### 7.12.9 Area, placement and timing

Out of context, `xcu200-fsgd2104-2-e`, `tsr_vu` in the K4 configuration (`FRONT_LANES` 8, `EXP_LANES` 8, `BACK_LANES` 4,
`LANE_PIPE` 2, `SPM_RD_LAT` 4, `NUM_CH` 4), input and output delay 1.9 ns at 4.0 ns and 2.4 ns at 5.0 ns, default
`synth_design`, `opt_design`, `place_design`, `phys_opt_design`, `route_design` (`tb/vu/viv/vu_ooc.sh`, through
`hw/u200/vivado_slot.sh`, 2026-11-08). CLB is the tile count of `report_utilization`: it carries the placer's spread in an empty
region, and in the image the placer packs tighter (the VU is 10.6 to 13.8 K tiles there, §8.8 of `docs/platform-u200.md`).

| `tsr_vu`, K4 | clock | CLB | LUT | FF | DSP | URAM | RAMB36 / 18 | WNS ns | WHS ns |
|---|---|---|---|---|---|---|---|---|---|
| `HAS_GDN` 0 | 4.0 ns | 13,082 | 58,696 | 68,842 | 249 | 0 | 16 / 13 | +0.297 | +0.013 |
| `HAS_GDN` 1 | 4.0 ns | 13,647 | 63,707 | 73,604 | 289 | 34 | 16 / 13 | +0.262 | +0.010 |
| `HAS_GDN` 0 | 5.0 ns | 12,947 | 58,698 | 68,843 | 249 | 0 | 16 / 13 | +0.515 | +0.008 |
| `HAS_GDN` 1 | 5.0 ns | 13,618 | 63,672 | 73,598 | 289 | 34 | 16 / 13 | +0.788 | +0.011 |
| **`HAS_GDN` 1 minus 0** | 4.0 ns | **+565** | **+5,011** | **+4,762** | **+40** | **+34** | 0 | | |

The op costs 8.5 % of the unit's LUT, 40 DSP and 34 URAM, and no RAMB. With `HAS_GDN` 0 the behaviour of every op is the one before
the integration (cycle for cycle, tests of §7.12.10). The synthesized unit is not quite the same size: 60,177 LUT and 68,837 FF
against 59,756 and 68,829 for the tree before the integration (+0.7 %, synthesis only; the injector +278, the sequencer +87, the top
+59 LUT; the cause is not isolated). 0 failing endpoints in all four runs. The worst setup path of
every run is a path of the unchanged unit (4.0 ns: lane-to-lane delay line, `HAS_GDN` 0: the angle buffer's write address), and
the WNS differs by up to 0.27 ns between the runs at one clock (the worst path moves, §7.10). The paths that start or end in the
engine, at 4.0 ns (`HAS_GDN` 1, `VOOC gdn_*` lines):

| Paths | worst slack ns | Path |
|---|---|---|
| into the engine | +0.460 | `u_coef` bank bit to the address of a key RAM (URAM), no logic, fanout over 8 lanes |
| out of the engine | +0.460 | the same net |
| engine to the lanes' input registers | +1.247 | the key RAM's output to the lane input, the mux after the injector |
| engine to the writer | +1.223 | the o-line valid to the writer's result FIFO |
| dot array (`u_dot`) | +0.925 | row RAM to the zero flag of the aligner, 7 levels |
| setup engine (`u_coef`) | +0.760 | the vector shift register to the exponent maximum, 3 levels |
| converter (`u_cvt`) | +1.269 | |

Every path of the engine has at least +0.46 ns at 4.0 ns, above the +0.3 ns the 250 MHz link needs, and no stage had to be added. At 5.0 ns
the worst engine path has +0.80 ns. Not covered: the effect of an image's occupancy (the out-of-context region is empty).

Where the resources go (HAS_GDN 1, 4.0 ns, Vivado hierarchical report after route):

| Block | LUT | FF | DSP | URAM |
|---|---|---|---|---|
| dot array: 8 lanes, 4 chains, accumulators, RAM B1, T | 1,552 | 1,017 | 36 | 16 |
| operand reader: RAM B2, KF | 741 | 300 | 0 | 16 |
| converter | 648 | 520 | 0 | 0 |
| setup engine (one `tsr_fp_mul`, scratch RAM) | 580 | 441 | 2 | 1 |
| scalar chain (one `tsr_fp_mul`, one `tsr_fp_add`) and the bv RAM | 820 | 509 | 2 | 1 |
| ingest, exponent tree, slots, launch control, o-line assembler | 1,085 | 1,867 | 0 | 0 |
| **`tsr_vu_gdn`** | **5,426** | **4,654** | **40** | **34** |

The rest of the increment (the engine is 5,426 LUT, the unit grows by 5,011) is the lane mux, the generator and injector slots, the
checker class and the credit logic, offset by logic that Vivado shares. An earlier form of the unit had its own update array (4,786 LUT, 32 DSP; 7,784 LUT, 72 DSP, 34 URAM in all, beside the SPM in SLR1,
which has no CLB left at 86 %). The engine is 2,358 LUT and 32 DSP smaller than that, although its ingest takes a whole line per
cycle and it has 64 row slots (VU-36).

**The VU Pblock** (`pblock_tsr_vu`, `CLOCKREGION_X0Y10:X2Y14` in SLR2, `hw/u200/k4_major_blocks.tcl`; raw range of the Pblock from
the device: 25,680 SLICE (the fitted Pblock 25,320 CLB), 420 RAMB36, 840 RAMB18, 160 URAM288, 1,320 DSP48E2). The unit with the op
uses 13.6 K CLB tiles (54 %, an upper bound as above), 289 DSP (22 %), 34 URAM (21 %), 16 RAMB36 (4 %). No other K4 block uses URAM in
SLR2 (the local intermediate banks are off). **The Pblock is not changed** and no constraint is added: Vivado places the engine
inside the VU's existing Pblock.

**Block RAM instead of URAM:** `GDN_STYLE` "block" needs about 43 RAMB36 (counted from the RAM sizes, not synthesized: per lane two for
each of B1 and B2, half each for T and KF, then the scratch and bv RAMs) and no fabric logic; the Pblock has 404 free. URAM is the choice
(VU-33): it leaves the RAMB36 for the VU's own buffers and the SPM-side logic that may grow.

**Not measured:** the full K4 image with `HAS_GDN` 1 (no link was run; the 250 MHz link of branch `k4f250` has to take it), the
16-front-lane build with the engine (tested functionally, no OOC run), the DSP-aligned adder variant (§7.10) with the shared lanes.

Reproduce: `hw/u200/vivado_slot.sh tb/vu/viv/vu_ooc.sh build/gdn_vu_ooc/g1_4.0 FRONT_LANES=8 EXP_LANES=8 BACK_LANES=4 LANE_PIPE=2
SPM_RD_LAT=4 NUM_CH=4 HAS_GDN=1 clk=4.0 iodly=1.9` (`HAS_GDN=0` for the unit without it; `synth_only` for the synthesis estimate;
summary: `grep ^VOOC <out>/vivado.log`).

#### 7.12.10 Verification

- **Reference** (`sw/sim/tsr_ref_gdn.c`, ctest `ref_gdn`, `ref_gdn_np`). `tests/test_ref_gdn.c`: the conversion
  against exact long-double arithmetic rounded by the hardware conversion (4 M random cases of every length and
  exponent, the RNE ties, the carry, the overflow edge, the 2⁻¹²⁶ − 2⁻¹⁵⁰ edge), integer dot products (exact),
  the window edge (elements 24, 25 and 26 binades below the row maximum), the tables, the head against the float64 recurrence
  (S' within 1.6 · 10⁻⁷ of the row maximum, o within 1.3 · 10⁻⁷ of Σ|S'q|), α = 0, β = 0, zero state, in place, tails, poison by Inf
  and NaN. `tests/ref_gdn_np.py` is an independent numpy model written from §7.12.3 (the conversion is derived on the
  binary32 grid, where the C works from the bit length); it matches the C bit for bit on random and edge heads
  (Inf, NaN, subnormals, zero rows, exponent spreads; 600 in ctest, 2,000 in a longer run) and on 20,000 conversions. Nine mutants of the C (truncation instead of
  round half up, the window width and its edge, the tie rule, the FTZ edge, the operand order of e, κ) are killed by one
  or both.
- **Simulator** (`sw/sim/tests/test_sim_vu.c`: test_gdn, test_gdn_checks, test_gdn_misc): the op against `tsr_ref_gdn` on SPM
  images (m 64 / 128 / 256, rows 1 … 100, out of place and in place, model-like and arbitrary data, d on k and o on v), every
  check code with ERR_INFO and the check order, the extents, CAPS gating, the cycle model.
- **Engine** (`tb/gdn/test_gdn.py`, Verilator and cocotb, bit-exact against `libtsr_ref.so`): `gdn_fx2f` (6,000 conversions, one per
  cycle, the tag, the latency), `gdn_dot` (576 rows, the exact sums, including the saturating case that fills the 48-bit
  accumulators at m = 256), `gdn_upd` (the operand reader: S and k words, α, e, tag, destination line and half-line, last, latency),
  `gdn_coef` (40 heads: key, tables, bv, exponents, Inf/NaN flags) and `gdn_unit` on `tsr_gdn_tb` (the engine with a model of the
  lanes: 32 heads of nine data classes driven as the injector drives it, with random pace, random writer credit and `o_line_ready`;
  every S' line and o line and the credit taken per row). `gdn_unit_latency`, `gdn_unit_rate`; the same with LANES = 16; other
  latencies, 8 slots, block RAM. Mutants of the RTL (`test_gdn_mutants.py`, opt-in): alignment width, sign, restart, the
  converter's tie / FTZ / overflow, table rounding and sign, operand timing, destination line, tag, the o tail, the credit per row,
  the row-credit gate, the vector tail, the rows' destination lines, operand order of FS, κ bank, the pseudo row's release, the
  row exponent are killed.
- **VU** (`tb/vu/test_vu_unit.py`, the unit on a real SPM against the simulator): `vu_gdn` (every data class and shape, one head at a
  time and several heads back to back), `vu_gdn_mixed` (RAW with VOP and RED on S' and o, the same tile in two consecutive heads,
  a VOP over a vector that GDN read, the s registers of the heads); with 8 and with 16 front lanes; and every other test of the unit
  unchanged with `HAS_GDN` 1. `tb/vu/test_vu.py` `vu_chk_gdn`: the checker class against the simulator on directed and random words
  (code, ERR_INFO, extents, payload).
- **Core** (`tb/core/test_core.py` `test_core_k4gdn`): the k4 core with `HAS_GDN` 1 runs two GDN heads (LD, SOP MOVI, GDN, ST) from the
  firmware on the RTL Ibex through the issue path and the checker; DRAM, the identity registers (`CAPS` 0x121) and the final CSR
  state equal the simulator's.
- **Card** (`tsr-selftest --ids 11`, `selftest` kernel id 11): four heads of 20 × 128 (two out of place, two in place), checked
  against `tsr_ref_gdn` on the device's DRAM and against the shadow simulator. It runs only with CAPS bit 8; without it the kernel
  returns RET CAPS.
- **Lint.** `rtl/lint.sh` (Verilator -Wall with the lane, shape, RAM-style and slot variants) and the Xilinx filelist.

#### 7.12.11 Integration state, next steps, risks

Done: the VU integration with `HAS_GDN`, the checker class, `CAPS` bit 8, the generated tables (opcode, CAPS, encoding, limits), the
simulator's VU unit, the card test (selftest id 11).

Next, in order:
1. **Link.** A k4 image with `HAS_GDN` 1 (this branch's `tsr_u200_k4`), merged into the 250 MHz link branch. The engine is in the
   VU's Pblock; the map is unchanged (`hw/u200/k4_major_blocks.tcl`). Show the Pblocks before the build.
2. **Firmware kind. Done on the simulator** (branch `gdn-fw`, SW §7.10, SW-78): the existing GATED_DELTA_NET kind takes the op
   when `CAPS` bit 8 is set and S_v ≥ 64 (no new kind: `LD` of the tile, the op, `ST`, double-buffered over the heads, the
   next head's state loading during the current one) and keeps its composition otherwise; numerics revision 2 with a row of
   `fw_kind_revs` for each path. The output is scaled after the op (`VOP.MULC`), not q before it. Heads of a launch use
   distinct tiles and token slots, so their ops are independent. On the simulator (model with 16 front lanes, which charges the op
   1.8 K cycles; 2.8 K with this section's 8 front lanes) the fused step costs 3.9 K core cycles per head and token at the default
   RISC-V clock ratio: the firmware's eleven pushes per step (1,827 RISC-V instructions) limit it, not the VU (SW §7.10).
3. **Lowering. Done** (the composed kind's lowering serves both paths, SW §7.10; the scale is applied to the output, as the CPU
   kernel does). The note that follows is the original plan: q scaled by 1/√S_v
   before the op; the state tensor layout (rows of S_v contiguous words, [S_v, S_v, H, n_seqs]) is the one assumed here;
   KDA gates (a per-row α) and the K-snapshot output of the CPU kernel (speculative decoding) are not covered. Check long-sequence
   state and logits against llama.cpp.
4. **The surrounding ops.** Convolution (SSM conv1d), L2 norm of q and k, the gates g and β (SOFTPLUS, SIGMOID, EXP: SOFTPLUS
   needs a stable algorithm and a cost, `docs/design/README.md` §12.6), the output gate and norm.
5. **Area.** The DSP-ALU adder (the Xilinx IP reaches 44 CLB with two DSP, §7.10) for the lanes' FA would help every op.

Risks (also in §12):
- The VU Pblock is 42 to 55 % used before this op (§7.10). The op adds the engine's logic; the OOC figures are in §7.12.9.
- Vivado's DSP cascade inference is version-dependent. The structure is checked on 2024.2. If a later version changes it,
  `tsr_vu_gdn_mac` takes DSP48E2 primitives behind an `ifdef`, with this module as the simulation model (the unisim model runs in
  XSim, not in Verilator).
- The URAM read-first rule for a read and a write of one address in one cycle. The ingest never writes line g of a freed slot
  before one cycle after the update read of that line.
- `d_k` is limited to 256 (the accumulator bound) and `d_v` to the engine's `R_MAX` (1,024); the pseudo row means that k is read twice.
- GDN holds the VU: while it runs no other VU op overlaps it (VU-35). A layer's other VU work waits for its heads; the firmware
  orders the layer so that the heads form one run.
- End-to-end quality against llama.cpp is not measured, only the state and the output against float64.

## 8. Usage (informative; SW builds the kernels)

### 8.1 Norms

- **RMSNorm of one vector:** `RED.SUMSQ TO_S sd=s1 post=RMS sk=s2 se=s3` (s2 = 1/n and s3 = eps,
  set once by `SOP.MOVI`), then `VOP.MULCB d = (x × s1) × w` (cmode 1). That is 2 ops, against 7
  in the spec recipe, with identical bits.
- **Gemma 4 per-head Q/K/V norms:** one `RED.SUMSQ` over all q, k and v head rows (12B SWA
  layers: rows = 16 + 8 + 8, m = 256; 12B global layers: 16 + 1 + 1, m = 512; 31B SWA: 32 + 16 +
  16, *confirm*; post RMS) gives a row vector. Then `VOP.MULCB` on the q rows (c = row vector,
  b = q_norm weight with stride 0), `MULCB` on the k rows, and `MULC` on the v rows: 4 ops.
- **Prefill:** rows = T tokens; c is the per-token vector.

### 8.2 FFN and residuals

- **Activation:** `VOP.SWIGLU` (Qwen) or `GEGLU` (Gemma): one op.
- **Residuals:**
  - Qwen: an `MVM` with ACC.
  - Gemma: post-norm `MULCB`, then `ADD`. `out_scale` is fused as
    `ADDMULC (cur + attn_out) × s`.
- **Qwen QKV bias:** one `VOP.ADD` over the adjacent q, k and v outputs (b = the adjacent biases;
  stride 0 for T > 1).

### 8.3 RoPE

One `ROPE` over q and k when they are adjacent (n_heads = 40 + 8 for Qwen2.5-32B), with T tokens.

### 8.4 Attention softmax

The FA kernel's sequence is AT §6.6 (normative): per chunk, ATTQK (scores and row maxima on the
MVM) or ATTQK + AMASK, then ASMAX on the VU, ATTPV (P·V and row sums on the MVM), and AMERGE on
the VU; ANORM at the end. That is two VU ops per chunk for all heads and tokens of a pass, plus
one ANORM per pass. The generic VOP / RED composition of an online softmax (RED.MAX, VOP.MAX,
EXPSUB, EXPSUBC, RED.SUM, MUL, ADD) stays available to other kernels; EXPSUB's −∞ rule keeps a
fully masked first block finite (§6.5).

### 8.5 Output head

- **Gemma softcap:** `VOP.MULC` (1/cap), `VOP.TANH`, `VOP.MULC` (cap) over the vocabulary. That
  is about 64 K cycles (0.23 ms) per token for 262,144 logits.
- **Greedy sampling** (M9): `TOPK` k = 1.

### 8.6 VU ops and time per decode token (attention excluded; *estimates*, latencies included)

| Model | VU ops per layer | VU cycles per layer | Per token | Share of token time |
|---|---|---|---|---|
| Qwen2.5-0.5B | 7 | ≈ 1.5 K (latency-dominated) | ≈ 0.13 ms | ≈ 0.7 % of 18.3 ms |
| Qwen2.5-32B | 7 | ≈ 6.3 K | ≈ 1.45 ms | ≈ 0.4 % of 341 ms |
| Gemma 4 31B | ≈ 17 | ≈ 10.5 K | ≈ 2.3 ms | ≈ 0.8 % of about 275 ms (Q4_K_M, bandwidth-bound) |

These times sit on the decode critical path. MVMs wait for them.

---

## 9. Interfaces with other areas

| Area | VU provides | VU assumes |
|---|---|---|
| **EX** | Encodings and checks for 0x12, 0x30–0x33, 0x38, 0x39, 0x40 (0x13, 0x3A–0x3F, 0x41–0x47 stay reserved; 0x39 is GDN, §7.12.4); the `x16` usage of §5; `OPERAND` conditions and `ERR_INFO` values (§5.9); `CAPS` bits 0, 3, 6, 7, 8; `CFG3[15:0]`; s bank reset +0.0 and unchanged by `SOFT_RESET`; no run-time errors; DQROW's unit = 3 (VU); execution of AT's 0x34–0x37 | Opcode blocks, `x16`, code 13 and the check order of EX §7–§8; PERF slots X + 13 … X + 15 (EX §6); ACTIVE / QHALT gating of `VU_CYCLES`; `SREG` at 0x9C0 |
| **AT** | Execution of ASMAX, AMASK, AMERGE and ANORM with the mapping of §7.4 and the rates of §7.9; EXPK (EXP(−∞) = +0, EXP(±0) = 1, no −0 or subnormal results, ≤ 0.53 ulp *model*); RC = `tsr_frecip`; per-head RMSNorm for Gemma's QK norm (§8.1); VOP for the K/V bias and RoPE before QACT; numerics in §6 | AT owns scores (QKᵀ), P·V, the KV layout and its quantization (no VU op writes Q8_0 data), GQA mapping, the kq scale (Gemma 4: 1.0), the semantics and checks of its vector ops (AT §5.4–§5.8) and the FA sequence (AT §6.6) |
| **CC** | Unit 3 dispatch with `cmd_ready` (accept-ahead) / `done` (no error output); SPM client 4 (pair reads, one write per cycle, R2 / R4 kept); the s bank (16 × 32 FF) exported to `SREG` (0x9C0) and the core window; the checker class (reserved masks, used words, conditions, four tagged extents, command struct ≈ 260 bits); resource figures §7.10 | Taken in CC: four extents per checker class (CC §8.2); I3 S − D ≤ 2 for unit 3 (CC §9.4); queue depth 16; SLR1 placement beside `tsr_spm` and the budget gate of CC §14.1 with the knobs of §7.10 as its first remedy |
| **DB** | Debug state of §5.10; `VU_CYCLES` and three extension counters; ops always complete in bounded time (≤ ≈ 1 M cycles: COPY of the whole SPM, TOPK worst case); hold points are op boundaries only | DB places the per-unit probes in `DBG_U3_*`, exposes `SREG` (0x9C0), and halts at the queue head (no mid-op halt); there is no s-register debug write in M6 |
| **SW** | §6 as the text for `tsr_ref_math.c`, the numpy model and the exhaustive tests (including `ref_math_gelu16`); reference names (§6.1); usage recipes (§8); accuracy figures to measure (§6.9); the rule that b row periods > 1 (SW's op-list ADD/MUL "b row period") are lowered as one op per period block, because the VU has strides, not periods | SW implements the simulator semantics, `tsr_ref_math.c` (rules of SW §4.5: `tsr_f*` calls plus integers, no libm, 128-bit products via a portable two-word helper, RV32-clean), `vu.hpp` builders from `sw/tables/tsr_tables.py` (rows filled from §5), traces `traces/vu`, and test-backend-ops / numcheck / klcheck (FA on). SW computes TS, M, freq_scale and (YaRN only) A_i on the host, and maps llama.cpp's GELU / GEGLU to GELU16 / GEGLU16 |

---

## 10. Verification

The simulator is the oracle; every VU op has a C reference, and the RTL must match it bit for bit.

1. **References.**
   - `tsr_ref_math.c`: EXPK, the composites (incl. GELU16), SIN/COS. `tsr_ref_vu.c`: VOP,
     reductions, ROPE (both angle forms), TOPK, SOP; DQROW reuses `tsr_dequant_row`; VMASK from
     M9. `tsr_ref_at.c`: AT's vector ops (AT §13.1).
   - An independent numpy model is written from this text alone (SW SW-23). It is compared bit
     for bit over all 2³² inputs for each unary function: EXPK × 4 κ, SIG, SILU, GELU, TANH, SIN,
     COS. Opt-in `slow`; the default runs 2²⁴ strided points plus specials.
   - Properties: EXPK monotonic in v for K > 0; EXPK(v, N2) = EXPK(2v, N1); EXPK never −0 or
     subnormal; SIN/COS never outside [−1, 1].
   - GELU16: all 65,536 fp16 inputs against the pinned `ggml_table_gelu_f16` (`ref_math_gelu16`,
     SW §10.1); the clamps at ±10 and the fp16 subnormal inputs as directed cases.
   - ROPE recurrence form: angles bit-identical to a C transcription of `ggml_rope_cache_init`
     (ext_factor = 0) for Qwen2.5 (base 1e6, n_dims 64 / 128) and Gemma 4 (base 1e4 / 1e6, n_dims
     256 / 512 with `rope_freqs`) at positions 0 … 2¹⁷.
2. **Accuracy against exact math.**
   - Exhaustive double / mpfr comparisons fill the table of §6.9 with measured maxima.
   - Gate: each measured maximum is ≤ 2 × its *model* figure. Otherwise the coefficients are
     reviewed before M6 RTL freeze.
3. **Simulator.**
   - `ex_vop`, `ex_red`, `ex_rope`, `ex_dqrow`, `ex_sop`, `ex_topk` and AT's `ex_asmax`,
     `ex_amask`, `ex_amerge`, `ex_anorm` with every check of §5.9 and AT §5.8.
   - `test_sim_vu.c`:
     - every function and kind;
     - every `ILLEGAL_OP` / `OPERAND` / `SPM_RANGE` case, with `ERR_INFO`;
     - m = 0 and rows = 0;
     - in-place aliasing; tails;
     - NaN, ±∞, ±0, subnormals and the FTZ boundary vectors of `docs/spec.md` §21 (FTZ decision point).
   - Random op streams (shapes, strides including 0, cmodes, functions) against a second,
     straightforward Python implementation of §4–§6.
   - The race detector gets the access sets of §5.9 and the aliasing rule of §4.5.
4. **RTL unit benches** (`tb/vu`, cocotb + Verilator, `common.tsr_ref` via ctypes as `tb/fp`
   does):
   - `tsr_vu_exp` and `tsr_vu_rcp`: exhaustive on a Verilator C++ bench (opt-in; RC over all 2²³
     mantissas × exponent edges in the default run).
   - Angle unit: 10⁶ random inputs + specials.
   - Reduction unit: random rows, m in 0 … 2048, skip-rule cases, including subtrees that sum to
     −0 (e.g. 0x80C00000 + 0x00800000) at m = 8k + r (§6.2); the same cases in `test_sim_vu`.
   - VU top with an SPM model:
     - random streams of dependent and independent ops, compared with the simulator on SPM bytes,
       s registers, completion order and error codes;
     - random `ready` on both bank ports;
     - hazard cases (RAW, WAW and s RAW across accept-ahead);
     - cycle assertions against §7.9.
   - Out-of-context timing at 3.333 ns for `tsr_vu`.
5. **Integration.**
   - Kernel op-list replays (`traces/vu`, M6) in the Verilator fastreplay with the VU.
   - The card selftest kernel runs a fixed VU vector set and compares against simulator goldens.
6. **Against llama.cpp.**
   - `test-backend-ops` for every node the backend sends to the VU (RMS_NORM, MUL, ADD, SCALE, GLU
     SWIGLU / GEGLU, ROPE, TANH, GET_ROWS, CPY, and FLASH_ATTN_EXT with AT), with NMSE tolerances.
   - numcheck B1 / B1b on VU nodes (bit-exact against `libtsr_ref` and the numpy model), and B3
     (error against exact math compared with the CPU's) per node.
   - klcheck with FA on (D17) for Qwen2.5-0.5B Q8_0, Qwen2.5-32B Q4_K_M, Gemma 4 12B (and 31B when
     a GGUF is available).
   - M6 pass criterion: 0.5B logits bit-identical to `TSR_DEVICE=sim`, and the 32B KL criterion,
     both with `offload_kqv` = false (attention on the CPU, SW §10.4); in M6 the VU's attention
     ops are verified by the selftest, `traces/at` and `ggml_tesseract_attn`. Device attention
     with the KV cache on the device is part of the M7 gate.

---

## 11. Decisions

- **VU-1 Operand model.**
  - *Choice:* rows × m with line-aligned rows, a line stride per operand (0 = repeat) and a
    per-row scalar c (s or SPM vector), for every VU op.
  - *Alternatives:* element-granular strides or gathers; separate per-head opcodes; 3-D shapes.
  - *Rationale:* every per-head and per-token pattern in Qwen2.5 and Gemma 4 becomes one op. That
    cuts the survey's ≈ 18 K ops per token to tens per layer. Line alignment keeps the SPM side
    one line per operand per cycle, and the address generators are counters. Element-granular
    strides would need a gather network across lanes.
- **VU-2 Reduction order.**
  - *Choice:* keep the spec's 128 partials (i mod 128) and the complete tree, per row. Allow the
    skip of unwritten subtrees, passing z(q) (−0 → +0) for a skipped pairing, which is exact.
  - *Alternatives:* a line tree first, then 8 phases; 16 partials with interleaved rows.
  - *Rationale:* it is already the contract (`docs/spec.md` §9), it is independent of lane counts,
    and the skip makes short rows cheap without changing bits. The draft's claim that a tree value
    is never −0 was wrong (FTZ keeps the sign); z() restores exactness (§6.2).
- **VU-3 Segmented RED output and fused post-ops.**
  - *Choice:* per-row vector (or an s register for one row), with post RCP and RMS (exact SQ, RC).
  - *Alternatives:* vector SQRT / RECIP passes; a scalar loop on the RISC-V.
  - *Rationale:* per-head norms and softmax sums become one op with the spec's rounding sequence,
    so the 1-row RMSNorm needs 2 ops instead of 7, bit-identical.
- **VU-4 EXP algorithm.**
  - *Choice:* a fixed-point integer algorithm (§6.3), f32 in and out, with constants as hex
    integers.
  - *Alternatives:* Cody-Waite reduction plus an f32 polynomial (about 7 FM + 5 FA per lane);
    table + f32 polynomial.
  - *Rationale:* the brief asked for hex f32 coefficients with f32 rounding at each step. This
    choice departs from that deliberately.
    - ≤ 0.53 ulp (*model*), against about 1–2 ulp for f32 chains.
    - About 600 LUT per lane instead of about 2.6 K (FA cost 421 LUT each). That matters because
      SLR1 is the constrained resource.
    - Integers make the definition exact and portable (all in uint64).
    - The constant K folds sign, the ×2 of TANH and GELU's C2G into the reduction for free.
- **VU-5 Reciprocal.**
  - *Choice:* every reciprocal is correctly rounded (= `tsr_frecip`); RECIP_APPROX is removed.
  - *Alternatives:* a bit-defined Newton-Raphson approximation.
  - *Rationale:* a correction product makes exact rounding cheap (about 5 DSP + 1 RAMB18 per
    lane). The contract is then a library function, with exhaustive RTL proof over 2²³ mantissas
    and no table to specify.
- **VU-6 SIN/COS.**
  - *Choice:* exact 64-bit 2/π integer reduction, f32 Cephes polynomials, a small-angle identity
    below 2⁻¹².
  - *Alternatives:* 3-piece f32 Cody-Waite π/2; CORDIC; tables.
  - *Rationale:* θ reaches 2¹⁸ (positions × A_0), where f32 Cody-Waite loses precision. The
    integer reduction is exact for |θ| < 2²⁴ and runs once per angle, not per element, so its
    multipliers are cheap.
- **VU-7 Function set.**
  - *Choice:* SILU, SWIGLU, SIGMOID, GELU / GEGLU in tanh form computed as x·σ(2u), GELU16 /
    GEGLU16 (VU-26), TANH with a small-argument path, EXP, EXPSUB(C), RECIP, MIN, the fused
    MULCB / MULCADD / ADDMULC, SETC.
  - *Alternatives:* erf GELU; separate tanh-based GELU; no fused forms.
  - *Rationale:* these cover Qwen2.5 and Gemma 4 dense end to end with one shared EXP + RC
    datapath. σ(2u) reuses SiLU's datapath exactly. erf GELU waits for a model that needs it.
- **VU-8 Lane split.**
  - *Choice:* 16 front lanes (incl. EXPK) at full rate, 8 back lanes for the half-rate functions.
  - *Alternatives:* 16 back lanes (+9 K LUT); EXP at half rate.
  - *Rationale:* decode costs for SwiGLU / GEGLU at half rate are < 0.15 % of token time. EXP stays
    at full rate because softmax over long contexts of small models (0.5B, 4 K) would otherwise
    exceed the KV bandwidth time.
- **VU-9 Flow control.**
  - *Choice:* credit-based (reads issued against output space), no clock enable in the lanes,
    fixed latency classes.
  - *Alternatives:* global `ce` stall.
  - *Rationale:* a `ce` fanning out to about 45 K FFs at ≥ 276 MHz is a timing risk; credits obey
    R4 and cost one FIFO.
- **VU-10 SPM access.**
  - *Choice:* CC's pair reads (alternate a-pairs and b-pairs); no operand placement rule; no
    operand caches.
  - *Alternatives:* skew buffers with a same-parity penalty; caching repeated b rows.
  - *Rationale:* pairs are parity-proof by construction. Two sources stream at full rate whatever
    their alignment, so lowering has no parity constraint.
- **VU-11 s-register access.**
  - *Choice:* one bank; writes only by VU ops (SOP.MOVI for constants); read-only for the RISC-V
    (core window) and the host (`SREG`).
  - *Alternatives:* RISC-V MMIO writes; SPM words only.
  - *Rationale:* writes stay in queue order with deps, so no new race class exists. The read
    location is `SREG0`–`15` at core CSR 0x9C0 for the host and, through the core window, for the
    RISC-V (CC CC-14, DB DB-21).
- **VU-12 s reset.**
  - *Choice:* +0.0 at hardware reset; kept across `SOFT_RESET`, RISC-V reset, QABORT, QCLEAR.
  - *Alternatives:* clear on launch.
  - *Rationale:* EX's requirement. Kernels initialise what they read, and the SDK prologue does
    it.
- **VU-13 Mask.**
  - *Choice:* `VMASK` (0x38), an in-place position-window mask (causal + SWA; row-to-position by G
    and P), deferred to M9 (neither simulator nor RTL before; `CAPS` bit7). llama.cpp attention
    masks are AT's (mask summaries, ATTQK bounds, AMASK).
  - *Alternatives:* VMASK in M6 for the FA kernel (the draft); drop the opcode.
  - *Rationale:* VMASK assumes cell j holds position k0 + j, which llama.cpp's ring-shaped and
    reorderable KV caches break (Gemma 4's 1,536-cell SWA ring wraps); AT's design handles them.
    The M9 resident loop owns a linear cache where VMASK is exact; deferring it saves ≈ 1 K LUT in
    SLR1 now.
- **VU-14 TOPK.**
  - *Choice:* fully specified now (rows, padding, total order); simulator in M6, RTL in M8 with a
    running-threshold filter.
  - *Alternatives:* RTL in M6; omit.
  - *Rationale:* M6 / M7 sample on the host; M8 (MoE) and M9 need it. Keeping 4 K LUT out of the
    first k4 link helps SLR1.
- **VU-15 DQROW.**
  - *Choice:* keep 0x12, re-specified as SPM → f32 on the VU (LD brings the repacked row);
    simulator from M6, RTL in M8 (`CAPS` bit6 = 0 in M6 bitstreams). The M8 embedding lookup is
    LD (ROW_BEATS) + DQROW; an f32 `tok_embd` copy remains the exact alternative for other GGUF
    types (AT §5.9).
  - *Alternatives:* DRAM → SPM dequantisation inside the DMA unit (AT's draft); RTL in M6.
  - *Rationale:* the VU already has the fp datapath. The DMA stays a pure mover, and the RISC-V
    computes the row location anyway. Token embedding stays on the CPU through M7 (SW R2), so the
    RTL waits for M8 and saves ≈ 1.3 K LUT in SLR1 now.
- **VU-16 Errors.**
  - *Choice:* EX's `OPERAND` (13) with the offending value; enumeration mismatches are
    `ILLEGAL_OP`.
  - *Alternatives:* a VU-specific code.
  - *Rationale:* follows EX EX-17; values make failures self-explaining.
- **VU-17 Field widths.**
  - *Choice:* 24-bit line addresses and m; 16-bit rows, strides, heads, T; functions and s indices
    in `x16`.
  - *Alternatives:* 32-bit words per field (does not fit 8 words); 20-bit addresses.
  - *Rationale:* everything fits in one 256-bit op with headroom to 1 GiB of SPM. Larger shapes
    split into several ops.
- **VU-18 Softmax ops.**
  - *Choice:* no generic fused online-softmax VOP; the attention softmax uses AT's vector ops
    (ASMAX, AMASK, AMERGE, ANORM) executed by the VU (VU-27), two VU ops per chunk.
  - *Alternatives:* compose 7 generic VU ops per KV block (the draft); a VSOFTMAX block op.
  - *Rationale:* AT fixed the dataflow (scores and P·V on the clusters, AT §6.6); its ops fuse the
    softmax steps the generic composition needed. 0x39–0x3F stay free.
- **VU-19 ROPE shape and angles.**
  - *Choice:* T tokens with a position vector; angles by ggml-cpu's recurrence θ_0 = f32(p),
    θ_{i+1} = FM(θ_i, TS) with TS from an s register, then FD by the freq factor and FM by
    freq_scale when given; the A-table form θ = FM(f32(p), A_i) only for YaRN; c and s scaled by M
    via FM; head_dim multiple of 16 (≤ 1024), n_dims multiple of 32 (NEOX) or 16 (NORM).
  - *Alternatives:* the A-table form for every model (the draft); element-granular pairs;
    per-token ops.
  - *Rationale:* the recurrence makes the angles bit-identical to the CPU's for ext_factor = 0;
    the A-table form differed by up to 3.7·10⁻⁴ rad at position 4,096 and 2.9·10⁻³ rad at 32 K
    (checked numerically), the dominant RoPE difference in long-context KL runs. Line-aligned pair
    halves keep the rotation at one line per cycle, and one op per layer covers q and k for all
    tokens.
- **VU-20 Accept-ahead.**
  - *Choice:* up to 2 VU ops held; RAW, WAW and s hazards checked by extent intervals.
  - *Alternatives:* strictly one op (drain about 20–64 cycles per op).
  - *Rationale:* many short independent ops (norm rows, per-head groups, AT's per-chunk ops) would
    otherwise pay full latency each. CC's I3 allows S − D ≤ 2 for unit 3 (CC §9.4).
- **VU-21 EXPSUB −∞ rule.**
  - *Choice:* EXPSUB(−∞, x) = +0, also for x = −∞ or NaN.
  - *Alternatives:* IEEE (NaN for −∞ − (−∞)).
  - *Rationale:* makes fully masked blocks harmless in online softmax without a software ordering
    rule.
- **VU-22 MAX / MIN.**
  - *Choice:* the spec's total order; MAX propagates NaN, MIN returns the non-NaN operand.
  - *Alternatives:* IEEE-2019 minimum / maximum.
  - *Rationale:* one comparator implements both and TOPK, consistent with `docs/spec.md` §9.
- **VU-23 Raw ops.**
  - *Choice:* COPY, SETC, MOVI / LDS / STS, VMASK pass-through and the ROPE tail copy bits.
  - *Alternatives:* canonicalise as fp.
  - *Rationale:* COPY then serves CPY / CONT / GET_ROWS of any 32-bit data, and positions and
    indices pass unchanged.
- **VU-24 Reduction unit size.**
  - *Choice:* an 8-input subtree (7 FA) + 2 scheduled FA, rows < 256 elements at half rate.
  - *Alternatives:* a 16-input tree + 7-FA top (22 FA, full rate for every row).
  - *Rationale:* saves about 5.5 K LUT. The half-rate cases are short rows (head_dim 64 / 128 or
    small blocks), which no target model reduces per head.
- **VU-25 Accuracy contract.**
  - *Choice:* functions are defined by their algorithms (bit-exact); error bounds are informative
    and measured.
  - *Alternatives:* define functions by error bounds.
  - *Rationale:* bit-exactness between the simulator, the RTL and the card is the project's
    verification basis. Bounds without an algorithm cannot be replayed.
- **VU-26 CPU-compatible GELU.**
  - *Choice:* VOP 0x17 GELU16 and 0x18 GEGLU16 reproduce the ggml CPU's `GGML_GELU_FP16` form
    (clamps at ±10, binary16 rounding of the input and of the result, §6.5); llama.cpp's GELU and
    GEGLU map to them; the exact tanh GELU stays for M8 / PyTorch.
  - *Alternatives:* the exact tanh GELU for llama.cpp (the draft); a 65,536-entry fp16 table ROM
    per lane (≈ 17 RAMB36 per lane for the ±10 range, and its content would depend on the build
    host's libm `tanhf`).
  - *Rationale:* the pinned CPU rounds every GEGLU element of every Gemma 4 layer to fp16 (relative
    error up to 3.9·10⁻³); an exact device GELU would differ from it at fp16 level everywhere,
    against a KL floor of a 1-ulp perturbation. The computed form matches the table except at fp16
    rounding boundaries, for ≈ 0.6 K LUT; the mismatch rate is measured exhaustively.
- **VU-27 AT's vector ops on the VU.**
  - *Choice:* ASMAX, AMASK, AMERGE and ANORM (AT's semantics) run on unit 3 with the front lanes,
    the reduction unit's MAX path and a multi-pass AMERGE (prepass for a_r, b_r; then ≈ 4–5 cycles
    per output line at C = 4); ANORM uses the back lanes' RC in a prepass.
  - *Alternatives:* a separate AT vector unit; generic VOP / RED sequences (§8.4's composition).
  - *Rationale:* the VU already streams two SPM operands per cycle and has EXPK in every front
    lane; the multi-pass mapping matches AT's 1,600-cycle merge tail for 32B and costs ≈ 2 K LUT
    and 2 RAMB36.

- **VU-28 Fused GDN op.**
  - *Choice:* one op, opcode 0x39, one head per op: state tile in the SPM (rows × m, dense), vectors k, q, v, scalars
    α, β from s registers; it computes u, e, S' and o in one pass over S. m a multiple of 16 with 64 ≤ m ≤ 256,
    1 ≤ rows ≤ 1,024. `CAPS` bit 8; `tsr_vu` parameter `HAS_GDN` (default 0, then nothing of the op exists and every
    other op runs cycle for cycle as before).
  - *Alternatives:* VOP / RED composition (four passes over S), an op for all heads of a layer, an op per row.
  - *Rationale:* S (151 MB per token) is read and written once; every head has its own k, q, v, α, β; 48 ops per
    layer cost nothing next to 2,064 cycles each, and the firmware overlaps `LD` and `ST` with them. The lower limit of m
    is four beats of the widest build, so that one limit serves 8 and 16 front lanes.
- **VU-29 Dot products in block fixed point.**
  - *Choice:* exact integer sums of aligned mantissa products in DSP48E2 cascades: a 26-bit row window (truncated),
    24-bit table entries in two 13-bit pieces, one RNE conversion per dot product, Inf and NaN poison the row.
  - *Alternatives:* f32 adder trees with a fixed order (A); 20-, 18- and 16-bit table entries.
  - *Rationale:* the same accuracy as (A) (dot error 4 · 10⁻⁸ of |S||k|; 32 K-step error of o and S equal to a
    few percent); no f32 adder (16 + 18 for two engines of (A)); order independent, so lane count and beat order are
    free; narrower entries lose a factor 7 to 50 (§7.12.6).
- **VU-30 The element update runs on the front lanes.**
  - *Choice:* S'_ij = FA(FM(α, S_ij), FM(e_i, k_j)) is the lane micro-op U_ROT (FM_A and FM_B in parallel into FA, the
    same `tsr_fp_mul` and `tsr_fp_add`) with a = S, b = k, c1 = α, c2 = e. The unit has no update array of its own.
  - *Alternatives:* an update array in the unit (8 × FM, FM, FA: 4,786 LUT and 32 DSP, 61 % of the unit); S' as integer
    products in a block frame with a per-element conversion in the DSP lanes.
  - *Rationale:* the lanes already are that structure; U_ROT equals the reference bit for bit because `FM` is
    commutative; the block-frame form is 1.3 to 30 times less accurate in S and needs a second row aligner and a
    normaliser per lane; the f32 form is bit-exact with the CPU kernel's decay and multiply-add up to FMA contraction.
- **VU-31 State tile through the SPM; the unit inside the VU.**
  - *Choice:* `LD` and `ST` move the tile (existing DMA), the op reads and writes the SPM line by line through the VU's
    reader and writer. The GDN-specific part (dot array, converter, setup engine, scalar chain, row and table buffers)
    is a module of `tsr_vu`, so it lands in the VU's Pblock beside the lanes. Nothing of it goes to SLR1.
  - *Alternatives:* a direct DDR stream into and out of the unit; the unit beside the SPM in SLR1 with its own update
    array and its own SPM client (the first design).
  - *Rationale:* 16 B of SPM traffic per element is half of the SPM bandwidth at 8 elements per cycle; a direct
    stream needs a new DRAM path, arbitration and an SLR crossing. The VU Pblock has the DSP and URAM the unit needs
    (§7.12.9) and the lanes it shares; SLR1, which has little CLB left, is not touched.
- **VU-32 Algebra of the scalars.**
  - *Choice:* the head table holds αβ k, the setup engine computes β v, e = β v − αβ (S · k); o = α (S · q) +
    e κ with κ = k · q from a pseudo row that streams k through the dot array.
  - *Alternatives:* o = S' · q as a second dot product over the new row (a second pass and a second frame per row);
    κ from two VU ops (MUL, RED) or from a separate MAC.
  - *Rationale:* both dots of a row run on the old S in one pass; no second pass over S'; the pseudo row costs one row
    time per head (0.8 %) and no hardware.
- **VU-33 URAM row buffers, lane cascade.**
  - *Choice:* per lane four URAM (row buffer for the dot pass, its copy for the update pass, the table RAM, the key
    RAM), 64 row slots, a 64-bit RAM word per lane at 8 lanes (two beats), the lanes started one cycle apart so that the
    DSP cascade adds a beat without skew registers; all lanes of a chain in one module so that Vivado infers the cascade.
  - *Alternatives:* wide RAMs with skew registers (about 45 CLB of flip-flops); one buffer with two read passes
    (three accesses per cycle on two URAM ports); LUTRAM or FF buffers; block RAM (`GDN_STYLE`, 48 RAMB36 more).
  - *Rationale:* the VU Pblock has URAM and no other block uses it there; the pairs of RAM copies and the per-lane skew
    cost no CLB.
- **VU-34 GDN lane operands by a mux after the injector.**
  - *Choice:* the engine's operand reader delivers registered beats (S word, k word, α, e, destination line and half-line,
    tag). `tsr_vu` selects them over the injector's lane inputs and write descriptor with one registered level of the
    engine, replicated per lane: U_ROT, coefficient 0, no back path, 16 words per beat.
  - *Alternatives:* a new lane micro-op; the engine feeding the writer directly; its own writer.
  - *Rationale:* no lane or writer changes; the select is a registered signal; the lane exit order holds because only
    the engine feeds the lanes while a GDN op runs; with `HAS_GDN` 0 the mux is not built.
- **VU-35 GDN hazards and extents.**
  - *Choice:* a GDN op is held back from every other VU op in both directions (`a.gdn != b.gdn`); two GDN ops overlap
    under the usual extent rules. The checker lists four SPM extents: d W, a R, k R, o W (CC §8.2: four slots; every
    written range is listed for the race detector); q and v are not listed and the s registers sa and sb are.
  - *Alternatives:* per-resource tracking of the lane mux; listing q and v and dropping the o write.
  - *Rationale:* the lane inputs have one source at a time; the four-slot limit keeps the issue path unchanged.
- **VU-36 Line-wide ingest and 64 row slots.**
  - *Choice:* the row buffers take a whole 16-word line per cycle (two beats per RAM word at 8 lanes) and the
    exponent tree takes 16 exponents. The ingest therefore runs at twice the rate of the dot array and the update, and the
    slots hold the rows that wait while the next head is set up.
  - *Alternatives:* one beat per cycle ingest (a bubble of 22 % per head at 128 × 128); 16 slots.
  - *Rationale:* same-shape heads run back to back at (rows + 1) × m / 8 cycles with no gap in the launches; the setup
    of head h + 1 (4 m + rows + 45 cycles) is hidden behind 63 buffered rows.
- **VU-37 Credits at the row launch; o lines through the result port.**
  - *Choice:* the unit takes the writer's output-FIFO credits for the S' lines of a row when it launches the row (up
    to 16 lines, from a registered "16 free" level); the o lines (16 words, zero tail) go to the writer's result port
    that TOPK uses, which is idle while a GDN op runs.
  - *Alternatives:* credits per beat; a second writer port.
  - *Rationale:* the lanes have no back pressure, so the credit must be there before the beat is issued; one port
    serves both; the write counts of the op (one per S' line, one per o line) make its done wait for the lanes' results.

---

## 12. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| SLR1 area: VU ≈ 48 K LUT (*estimate*, ± 30 %); SLR1 straddles the 65 % gate (CC §14.1) | k4 link below 276 MHz, longer links | out-of-context VU run before T2; the gate's remedies in order: knobs `VU_EXP_LANES`, `VU_BACK_LANES` (≈ −9 K), DB's optional blocks, lanes to SLR0 behind pipes |
| New arithmetic RTL (EXPK, RC, reduction, angle recurrence, AT's multi-pass ops) | silent mismatches | exhaustive C++ benches against the C reference; numpy model independence; RED −0 skip cases |
| GELU16 differs from the CPU's libm-based table at fp16 rounding boundaries | residual Gemma 4 KL gap | `ref_math_gelu16` mismatch rate (target < 1 %, ≤ 1 fp16 ulp each); Gemma 4 12B KL with and without `GGML_GELU_FP16` before the RTL freeze (SW §10.4) |
| Attention becomes VU-bound for small models at long context (AMERGE ≈ 4–5 cycles per output line, ASMAX one pass per chunk) | DDR idles during attention for 0.5B | AT's chunk policy overlaps ASMAX / AMERGE with the next ATTQK / ATTPV; `VU_HAZARD` / `VU_SPM_WAIT` and `ATT_PATH_STALL` profiles; a fused op later in 0x39–0x3F |
| SPM contention (QACT, DMA, gather, bridge) | VU slower than §7.9 | `VU_SPM_WAIT`, `SPM_RD_WAIT`; pair reads use half the slots for one-source ops |
| Gemma 4 31B dimensions unverified (no 31B GGUF on the build host; 12B is read and confirmed) | 31B head_dim / layer assumptions wrong | limits cover head_dim ≤ 1024 and any vocab; confirm with a 31B GGUF before relying on it |
| KL on Gemma 4 with FA on has no baseline yet | M7 gate unclear for Gemma | SW sets the FA-on baselines (D17); GELU16 and the RoPE recurrence follow the CPU's forms |
| TOPK worst case (sorted input) is 16× slower | M9 sampling latency spikes | deterministic bound n cycles (0.95 ms for 262 K); acceptable for greedy decode |
| The GDN engine in the VU Pblock (§7.12.9: +5.0 K LUT, +40 DSP, +34 URAM; the unit is 54 % of the Pblock's CLB tiles in the out-of-context placement) in an image whose SLR2 is at 85 % CLB | the router has less room around the VU; a link below the clock | the Pblock holds 160 URAM and 1,320 DSP, of which the unit uses 34 and 289; all engine paths have ≥ +0.46 ns at 4.0 ns out of context; `HAS_GDN` 0 removes the engine; the link of the 250 MHz branch is the test |
| GDN DSP cascade inference differs in another Vivado version | the dot array loses its cascades and the accumulators fall back to fabric adders | the structure is checked on 2024.2; `tsr_vu_gdn_mac` is the one module to change to DSP48E2 primitives (the unisim model runs in XSim, not in Verilator) |
| GDN quality against llama.cpp is measured on float64 only | a model-level difference not seen at the state level | long-sequence logits against the pinned llama.cpp before the firmware kind is accepted (§7.12.11, item 4) |

## 13. Open items (measurements and confirmations only)

1. Exhaustive measurements replace the *model* accuracy figures of §6.9 (EXP ≤ 0.53 ulp, SIGMOID
   2.0, SILU 2.3, GELU 11 ulp for x ≥ −3, TANH 9.8 ulp, SIN / COS ≤ 2⁻²³·⁶⁶ absolute) before the M6
   RTL freeze; gate: measured ≤ 2 × model. *Done in W0 (2026-10-30):* every figure measured over
   all 2³² inputs and within the gate (§6.9).
2. The `ref_math_gelu16` mismatch rate (*measured in W0:* 0.415 %, GELU16 correctly rounded in
   every mismatch, §6.5) and the Gemma 4 12B KL gap (VU-26, open).
3. Gemma 4 31B dimensions from a GGUF, when one is available (12B is confirmed, §3).
4. The out-of-context `tsr_vu` utilisation, which fixes the knob setting against CC's gate.

Resolved by the integration: CC took the four tagged extents and I3 ≤ 2 for unit 3 and carries
§7.10 in its budget; EX fixed the PERF slots X + 13 … X + 15; AT's softmax runs on the VU as its
vector ops (§5.11).

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

T64[j] = round(2^(j/64)·2³⁰). 2^(j/64) is irrational for j ≠ 0, so no ties occur. Row i lists j =
8i … 8i + 7:

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

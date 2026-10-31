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
  - 16 front lanes stream one line per cycle. Each has 3 fp multipliers, 1 fp adder and an EXP
    core.
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
  - `CAPS` bits 0, 3, 6, 7, 8–11 and `CFG3[15:0]`.
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
  - bits 8–11: reserved for VU extensions, 0.
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

**Shrink knobs**, as elaboration parameters (results unchanged):
- `VU_EXP_LANES` 8: −4.4 K LUT, −88 DSP; EXP and EXPSUB go to half rate.
- `VU_BACK_LANES` 4: −4.3 K LUT, −36 DSP; SILU, GELU and the others go to quarter rate.
- Minimum configuration: about 38 K LUT.

**Placement:** SLR1 next to `tsr_spm`. The SPM-facing stage (pair-read registers, output FIFO)
sits beside the URAM columns, with the lanes in adjacent clock regions. Fallback, the third remedy
of CC's 65 % gate (CC-24, after the shrink knobs and DB's optional blocks): the lanes move to
SLR0 behind registered pipes. That takes
about 1.6 K SLLs (2 × 512 read + 512 write + control) and adds 8–12 cycles of latency; results
are unchanged.

**Timing:**
- ≤ 6 LUT levels per stage (`docs/rtl-m2.md` §1).
- DSPs fully registered.
- No global clock enable: credits replace back-pressure.
- 512-bit buses registered once at the bank boundary and once per lane group.
- Gate: CC's T2 (out-of-context core at 3.333 ns), and the k4 link ≥ 276.3 MHz.

---

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
| **EX** | Encodings and checks for 0x12, 0x30–0x33, 0x38, 0x40 (0x13, 0x39–0x3F, 0x41–0x47 stay reserved); the `x16` usage of §5; `OPERAND` conditions and `ERR_INFO` values (§5.9); `CAPS` bits 0, 3, 6, 7; `CFG3[15:0]`; s bank reset +0.0 and unchanged by `SOFT_RESET`; no run-time errors; DQROW's unit = 3 (VU); execution of AT's 0x34–0x37 | Opcode blocks, `x16`, code 13 and the check order of EX §7–§8; PERF slots X + 13 … X + 15 (EX §6); ACTIVE / QHALT gating of `VU_CYCLES`; `SREG` at 0x9C0 |
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

# numcheck — one-prompt numeric verification

## Purpose

`klcheck` (`docs/klcheck.md`) judges the device at the model level: KL of the logits against the
CPU and against the CPU's own 1-ulp sensitivity, over several prompts and generated tokens.
numcheck asks a narrower question with as few model runs as possible: **is every number the device
produces the number the spec says it must produce, and how far is it from exact math?**

- One short prompt (a fixed English sentence, T tokens), one prefill batch, logits for every position.
- Three runs per model, each its own process: `cpu`, `floor` (cpu with a 1-ulp nudged input) and
  `dev` (Tesseract). Every run records every graph node.
- The comparison is offline: (A) drift of every node and of the logits, (B) every weight
  MUL_MAT checked element by element: B1 bit-exact against the reference model (`libtsr_ref`), B1b
  bit-exact against an independent numpy model of the spec text, B2 inside a proven fp32 error
  bound, B3 the error against exact math compared with the CPU's.

Tools: `tsr-numcheck` (`sw/tools/numcheck.cpp`, the dumper, links llama.cpp) and
`sw/tools/numcheck.py` (the comparer; native helpers `sw/tools/numcheck_helper.c` and
`libtsr_ref`). Results for three models are at the end of this document.

## Method

### Runs

| Run | Backend | Input | Purpose |
| --- | --- | --- | --- |
| `cpu` | ggml CPU only (`TSR_DEVICE=none`, `devices = {}`, `n_gpu_layers = 0`) | `embd`: `token_embd.weight` rows, ggml `to_float` (bit-exact with the token path) | reference |
| `floor` | same as `cpu` | `embd+1ulp`: every element `nextafterf` ±1 ulp (klcheck's `embd_nudge`, sequence 0, position = token index, `--seed`, default 1) | the CPU's own sensitivity |
| `dev` | Tesseract (`TSR_DEVICE=sim`, `devices = {Tesseract0}`, `n_gpu_layers = 999`: every layer and the output head) | `embd` | device under test |
| `cpu-tok` (check) | CPU, no observation | token ids | embd path == token path, observing does not change CPU numerics |
| `dev-noobs` (check) | Tesseract, no observation | `embd` | observing does not change device numerics |

All three main runs use the embd input path, so their graphs are identical node for node (with
token input llama.cpp computes a different branch of its input selection). Model and context
settings are klcheck's (`docs/klcheck.md`, "Settings fixed in every phase"): flash attention off,
KV cache f16 on the host (`offload_kqv = false`, required by the backend), CPU weight repack off
(`use_extra_bufts = false`), threads `-t` (default online CPUs / 2), `n_ctx` = T rounded up to 256,
`n_batch = n_ubatch = 256` (T must fit one ubatch), every position an output. The prompt is
truncated to T tokens; T defaults to 16 for models above 8 B parameters and 32 otherwise.
`TSR_SIM_THREADS` defaults to 4 when unset.

Observation uses `llama_context_params.cb_eval`: the callback asks for every computed non-view
node, so the scheduler computes the graph one node at a time, synchronizes, and the node's output
(and, for a weight MUL_MAT, its src1) is read with `ggml_backend_tensor_get`, which also reads
device buffers. On `dev` the src1 read is the scheduler's input copy in the Tesseract buffer, i.e.
exactly the activation the device quantized.

### (A) Drift

For every node (matched by index `i`; the node lists of the three runs are verified identical):
NMSE(dev vs cpu), NMSE(floor vs cpu), their ratio, max |Δ|, cosine, and whether dev and cpu are
bit-identical. Summaries: per layer at `l_out-N`, `result_norm`, `result_output`; the worst nodes by
NMSE and by ratio; the first node where dev diverges from cpu (it must be the first device node).
Logits: KL(cpu‖dev) and KL(cpu‖floor) per position in float64, top-1 agreement, and the positions
where the floor did not move at all (see "The floor partly vanishes" below).

### (B) Every weight MUL_MAT of the dev run

A weight MUL_MAT is one whose src0 is a GGUF tensor. The weight rows are read from the GGUF (all
splits) with `pread` in row chunks and converted to the device format exactly as
`sw/ggml-tesseract/tsr-weights.cpp` does (native Q8_0 / Q4_K bytes unchanged; F32 straight to
`quantize_row_q8_0_ref`; every other type ggml `to_float` then `quantize_row_q8_0_ref`), using ggml's
own functions through `libnumcheck_helper.so`.

- **B1, bit-exact.** `y_dev == nc_model_mm(device blocks, x_dev)` bit for bit for every element
  (row × token). `nc_model_mm` runs `tsr_qact` once per token and `tsr_repack_row` + `tsr_dot_row`
  once per row from `libtsr_ref` (the oracle). This covers the backend's conversion, repacking, row
  allocation over channels, T tiling, staging and QACT placement on real data. Any mismatch is a
  defect; the report lists row, token, both values and the ulp distance.
- **B1b, bit-exact against the spec text.** `y_dev == spec_model_y(...)` bit for bit for every
  element, where `spec_model_y` (numcheck.py) evaluates the spec's 수치 규칙 in numpy without any
  `libtsr_ref` code: activation ints and scales from the numpy QACT model `np_qact`, weight ints and
  scales parsed in numpy, block terms `(d·Dx)·f32(isum)` (Q4_K: `− (dmin·Dx)·f32(msum)`) with every
  multiply and subtract rounded separately in binary32 with DAZ / FTZ, 16 partials `p_(i mod 16)`
  from +0.0, then the left-paired 4-level tree. Unlike B2, B1b sees the accumulation order: a
  sequential sum differs from the device in 31.5 % of the boundtest rows and in 67–87 % of the
  elements of real tensors. B1 and B1b together make the result independent of either model alone:
  the simulator's MVM (`sw/sim/tsr_sim.c`) and the B1 model both call `tsr_dot_row`, so B1 alone
  would not notice a defect shared by `libtsr_ref` and the simulator.
- **B2, proven bound.** `|y_dev − y_q64| ≤ bound` for every element. `y_q64` is the float64 value of
  the same quantized operands (device-format weight ints and scales × QA ints and scales, spec
  formulas, no rounding to f32), with the activation from `np_qact`, which is itself compared byte
  for byte with `tsr_qact` on every real token vector. The bound is derived below. B2 is a worst-case
  rounding envelope: it catches gross errors (a wrong scale, operand or dropped term) without any
  reference model, but **not the summation order**. A sequential sum stays inside it (32B
  `blk.35.ffn_down`: max |err|/bound 0.306 for the sequential mutant vs 0.110 for the device;
  selftest: the mutant fails B1 and B1b and passes B2). The order is checked by B1 and B1b.
- **x without a src1 dump** (`--src1-from-producer`). A weight MUL_MAT outside the dumper's
  `--mm-layers` has no src1 dump, but its src1 is the output of a node that is always dumped (llama.cpp
  names: `attn_norm-N` for attn_q / k / v, `kqv_out-N` for attn_output, `ffn_norm-N` for ffn_gate / up,
  `ffn_swiglu-N` for ffn_down, `result_norm` for the output head). The device reads a byte copy of
  that output. With the option, the comparer first verifies the mapping bit for bit on every src1 dump
  of the dev and cpu runs, and only then uses the producer output as x for the other weight MUL_MATs
  (otherwise WARN and those stay unsampled). A wrong x could not hide a defect: B1 would fail. This
  lets a sampled dump (`--mm-layers`, small src1 files) still be checked on every weight MUL_MAT.
- **B3, exact-math error.** `rel_err = ‖y − W x‖ / ‖W x‖`, where W is ggml `to_float` of the
  **original** GGUF weight in float64 and x the run's own src1. Computed for dev (x_dev, y_dev) and
  cpu (x_cpu, y_cpu), per token first. **Dominant tokens:** a token that holds more than half of a
  tensor's `‖Wx‖²` (in the dev or the cpu run) is listed and left out of the tensor's ratio, because
  pooled over all tokens it alone would decide the value. On 32B, token 0 carries a massive
  activation and holds > 99 % of `‖Wx‖²` in several `ffn_down`; pooled over all tokens,
  `blk.4.ffn_down` then measures one token (ratio 1.345, while tokens 1–15 give 0.40–0.92). The
  report keeps the all-token values in separate columns, plus the median of per-token ratios.
  Per weight type: pooled `sqrt(Σ‖y−Wx‖² / Σ‖Wx‖²)` over the type's tensors (dominant tokens left out)
  and the median / range of per-tensor ratios. For a native type the ratio rel_err_dev / rel_err_cpu
  should be about 1; for a converted type it shows the requantization cost.
- **Verdict** (exit code 0 / 1 / 3; 2 = usage or input error, or a crash, with no report written).
  - **PASS** iff B1 and B1b have 0 mismatches, B2 has 0 violations, the numpy QACT model equals
    `tsr_qact` on the real activations, the model-free boundtest passed (≥ 10⁶ dot products in
    domain, B2 bound and numpy spec model), `dev-noobs` (if given) is bit-identical, and every
    selected weight MUL_MAT was checked on the device.
  - **FAIL** if any of the above fails.
  - **INCOMPLETE** if nothing failed but a selected weight MUL_MAT could not be checked: it ran off
    the device, its weight is missing from the GGUF, its shapes do not match, or its op is
    MUL_MAT_ID (MoE, not modelled).
  - WARN: native tensor with B3 ratio > 1.5 (dominant tokens left out); weight MUL_MATs without a
    src1 dump (not sampled; explicit via the dumper's `--mm-layers`); producer mapping not
    verified; B2 out-of-domain elements; dev diverging before the first device node; run
    directories of the wrong kind (run / input / device); no `--dev-noobs` (observation invariance
    not checked).
- **Input checks** (exit 2 before any work). cpu, floor, dev (and `--dev-noobs`) must name the
  same model file (`realpath`), with the same `info.model_desc` / `n_params` / `n_layer` /
  `n_embd`, tokens and `n_vocab`. `-m` must be the dev run's model (if that file no longer exists:
  the same file name). Every weight MUL_MAT
  node's `weight_type` and shapes must match the GGUF tensor. `--dev-noobs` must be a dev run with
  input `embd` made with `--no-observe`, not the observed dev directory. `--mm-layers` is
  validated first. Two quantizations of one model have identical tokens, so the old token-only
  check let mixed run sets through.
- **Layer drift where the floor moved.** At positions where the floor logits are bit-identical to cpu
  (the 1-ulp nudge vanished, see the results), the floor contributes nothing to NMSE(floor). The
  layer table therefore also gives the NMSE ratio over the other positions ("ratio, floor moved").

### B2 bound: derivation

Spec references: 수치 규칙 (block terms, row accumulation, fp 공통 규칙) and 구현 명확화 (v0.2.1)
(FTZ 판정 시점, QACT 입력).

Notation: `u = 2⁻²⁴`, `λ = 2⁻¹²⁶`, `γₙ = n·u / (1 − n·u)`. `fl(·)` is one f32 operation under the
spec rules: DAZ on inputs, IEEE RNE in the binary32 exponent range, then FTZ (a subnormal result
becomes ±0).

**One operation.** Let x be the exact result (finite, inputs 0 or normal after DAZ).
- If `|x| ≥ 2⁻¹²⁶`: `fl(x) = x(1+δ)`, `|δ| ≤ u`.
- If `|x| < 2⁻¹²⁶`: the result is ±0, or ±2⁻¹²⁶ when `|x| ≥ 2⁻¹²⁶ − 2⁻¹⁵⁰`. So `fl(x) = x + η`,
  `|η| < λ`. An addition whose exact result is subnormal is exact before the flush, the same case.
- int → f32 is `i(1+δ)`, exact for `|i| ≤ 2²⁴`.
- Only one of δ, η is nonzero for a given operation.

**Operands.**
- `d`, `dmin` (fp16 → f32) are exact and never subnormal in f32. `Dx` is the f32 QA block scale
  (DAZ applied, as the device's multiply does).
- Q8_0 `|isum| ≤ 32·128·128 = 2¹⁹` and Q4_K `|msum| ≤ 8·63·4096 < 2²⁴` convert exactly; Q4_K
  `|isum| ≤ 8·63·61440 ≈ 2^24.9` can round (one more δ).
- A product `fl(â·f)` of an f32 result â (0 or normal) and an integer-valued f (0 or `|f| ≥ 1`) is 0
  or at least 2⁻¹²⁶ in magnitude, so **it never flushes**. Inside a term only `d·Dx`, `dmin·Dx` and
  the Q4_K subtraction can flush.

**Q8_0 term** `t̂ = fl(fl(d·Dx)·isum)`, exact value `T = d·Dx·isum`:
`t̂ − T = T[(1+δ₁)(1+δ₂) − 1] + η₁·isum·(1+δ₂)`, so

  `|t̂ − T| ≤ γ₂|T| + c₁·λ(1+u)|isum|`, with `c₁ = [|d·Dx| < 2⁻¹²⁵]` (2⁻¹²⁶ suffices; 2⁻¹²⁵ is a safe over-approximation).

**Q4_K term** `t₁ = fl(fl(d·Dx)·fl(isum))`, `t₂ = fl(fl(dmin·Dx)·msum)`, `t̂ = fl(t₁ − t₂)`; exact
`T₁ = d·Dx·isum`, `T₂ = dmin·Dx·msum`, `T = T₁ − T₂`, `M = |T₁| + |T₂|`:
- `|t₁ − T₁| ≤ γ₃|T₁| + c₁λ(1+u)²|isum|`, `|t₂ − T₂| ≤ γ₂|T₂| + c₁ₘλ(1+u)|msum|`.
- `t̂ − T = (t₁ − T₁)(1+δ₅) − (t₂ − T₂)(1+δ₅) + T·δ₅ + η₅`, and `γₖ(1+u) + u ≤ γₖ₊₁`, so

  `|t̂ − T| ≤ γ₄|T₁| + γ₃|T₂| + λ[c₁(1+u)³|isum| + c₁ₘ(1+u)²|msum| + c₃] ≤ γ₄·M + ε`

  where `c₁ₘ = [|dmin·Dx| < 2⁻¹²⁵]` and `c₃` marks a subtraction that can flush: it needs
  `|t₁ − t₂| < 2⁻¹²⁶`, and `|t₁ − t₂ − T| ≤ γ₄M + 2λ(|isum| + |msum|)`, so `c₃ = [M > 0 and
  |T| < 2⁻¹²⁵ + γ₄M + 2λ(|isum| + |msum|)]` (M = 0 means both products are exactly 0).

**Both formats.** With `γc = γ₂` (Q8_0) or `γ₄` (Q4_K), `Mᵢ = |Tᵢ|` (Q8_0) or `M` (Q4_K) and `εᵢ`
the λ terms: `|t̂ᵢ − Tᵢ| ≤ γc·Mᵢ + εᵢ` and `|t̂ᵢ| ≤ (1+γc)Mᵢ + εᵢ`.

**Row accumulation** over n terms (Q8_0 blocks or Q4_K super-blocks; spec 행 누적):
- Partial `p_(i mod 16)` starts at +0, so its first addition is exact; the fixed tree adds 4 levels.
  Every term passes through at most `h = ⌈n/16⌉ + 3` rounded additions.
- There are at most `n + 15` additions; each can flush once (`|η| < λ`), and that error is later
  scaled by at most `(1+u)^h ≤ 1 + γ_h`.
- Standard summation analysis (Higham, *Accuracy and Stability of Numerical Algorithms*, Lemma 3.1,
  §4.2): `|ŷ − Σt̂ᵢ| ≤ γ_h·Σ|t̂ᵢ| + (n+15)·λ·(1+γ_h)`.

With `S₀ = ΣMᵢ`, `E = Σεᵢ` and `Y = ΣTᵢ` (exact):

  **`|ŷ − Y| ≤ γc·S₀ + E + γ_h·((1+γc)·S₀ + E) + (n+15)·λ·(1+γ_h)`**

**Reference and domain.**
- `y_q64` evaluates the same integers and scales in float64. `d·Dx` is exact in binary64
  (11 × 24 bits); each term takes 1 (Q8_0) or 3 (Q4_K) roundings, the sum at most n − 1. The bound
  adds `γ64_(n+4)·S₀·1.01` (u64 = 2⁻⁵³) for that, is computed in float64 and multiplied by
  `1 + 2⁻⁴⁰`. An element violates B2 iff `|y_dev − y_q64| > bound`.
- Domain: all values finite, `S₀ < 2¹²⁶` and `|d·Dx|, |dmin·Dx| < 2¹²⁶` (nothing overflows).
  Elements outside are counted, not judged; there were none on model data.
- The ACC flag (`y_out = y_old + y`) is not covered; the backend does not use it for weight
  MUL_MATs (K ≤ K_MAX for every model here), and B1 would show it.

**Independence from `tsr_ref`** (B1b, B2). `np_qact` uses numpy's native binary32 `*` and `/` with
explicit DAZ, FTZ and canonical NaN; rounding through exact float64 (`rint` half-even,
`floor(|v|+0.5)` half-away), saturation to [−128, 127], NaN → 0; numpy's float16 cast; A256 max =
first largest |x| (`argmax`, NaN excluded). Weight scales and nibbles are parsed in numpy (own
`get_scale_min_k4`). The integer dots use float32 matmul, exact because every partial sum is an
integer below 2²⁴; integer zeros are normalized to +0 (int → f32 of 0). `spec_model_y` uses the
same binary32 primitives (`fmul`, `fsub`, `fadd`: numpy RNE, DAZ on inputs, FTZ and canonical NaN on
results); Q4_K `f32(isum)` is the RNE cast of the exact integer. Partials are padded to a multiple
of 16 terms with −0.0, the exact identity of RNE addition (`p + (−0) = p`, also for `p = ±0`), so
the 16 lanes are added as whole vectors. The only shared code with B1 is ggml's weight conversion
(`quantize_row_q8_0_ref` for converted types), which defines the device format.

### Validation of the tools

- **Bound and spec-model self-test (`numcheck.py boundtest`, seed 1, 2.8 s).** 1,024,152 dot
  products (row × token) from `libtsr_ref` (`tsr_qact` + `tsr_dot_row`, or a QA image built
  directly), random K (Q8_0 32–32768, Q4_K 256–27648), 1,023,361 in domain (791 `q8_raw` rows have
  amax > 127·65504, so A32's fp16 D is Inf). **B2: 0 violations.** |err|/bound median 0.042, p99
  0.29, max 0.995; on the 936,564 rows where no term can flush: median 0.043, max 0.634. The bound is
  nearly attained where a flush of d·Dx dominates (error |isum|·|d·Dx|, slack λ|isum|) and within a
  factor 1.6 in the normal range; halving it would produce violations (the selftest asserts this
  sensitivity). **B1b model: 0 mismatches on all 1,024,152 rows**, the out-of-domain (Inf / NaN)
  and FTZ rows included. Negative control: the sequential-sum mutant differs from `libtsr_ref` in
  282,850 of the 897,176 rows with more than one term (31.5 %; 4.8 % in `q4_directqa` up to 64.8 %
  in `q8_cancel`); the boundtest fails if the mutant is caught in fewer than 5 % of those rows.

  | case | operands | in domain | viol. | FTZ-slack rows | median | max | max without FTZ slack |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `q8_quant` | `quantize_row_q8_0_ref` of Gaussian / t₂ weights, x scales 10⁻³–10² | 201,880 | 0 | 0 | 0.065 | 0.610 | 0.610 |
  | `q8_raw` | raw q incl. −128, fp16 d incl. subnormal and 65504, x up to 10^5.5 | 103,569 | 0 | 0 | 0.067 | 0.634 | 0.634 |
  | `q8_extreme` | q = ±127/−128, x all at ±amax (max \|isum\|) | 40,232 | 0 | 0 | 0.060 | 0.547 | 0.547 |
  | `q8_cancel` | blocks b+h = −(block b), same QA: exact sum 0 (+ one tiny block) | 92,296 | 0 | 0 | 2e-11 | 0.283 | 0.283 |
  | `q8_directqa` | any q, any f32 Dx 2⁻¹⁴⁰…2⁶⁰ ± (DAZ / flush of d·Dx) | 82,176 | 0 | 21,093 | 0.071 | 0.985 | 0.619 |
  | `q4_random` | random scales / nibbles, d, dmin incl. subnormal fp16 | 160,752 | 0 | 0 | 0.059 | 0.497 | 0.497 |
  | `q4_cancel` | x second half = −first half, same weights: exact 0 | 72,928 | 0 | 0 | 0 | 0.195 | 0.195 |
  | `q4_termcancel` | sc = m, q = 1–2, dmin ≈ d: T₁ ≈ T₂ inside every term | 70,496 | 0 | 0 | 0.024 | 0.315 | 0.315 |
  | `q4_bigisum` | sc = 63, q = 15, x ≈ amax: \|isum\| > 2²⁴ (rounded int → f32) | 43,208 | 0 | 0 | 0.065 | 0.422 | 0.422 |
  | `q4_tiny` | amax 10⁻⁴¹–10⁻³¹: Dx near 2⁻¹²⁶, d·Dx flushes, iscale = Inf | 70,728 | 0 | 55,028 | 0.008 | 0.995 | 0.289 |
  | `q4_huge` | amax 10¹⁸–10²⁴, large d | 31,096 | 0 | 0 | 0.057 | 0.477 | 0.477 |
  | `q4_directqa` | any q, consistent BS, any f32 Dx 2⁻¹⁴⁰…2⁴⁰ ± | 54,000 | 0 | 10,676 | 0.062 | 0.932 | 0.414 |

- **QACT model and fp32 primitives.** `np_qact` vs `tsr_qact` on 47,942 vectors (A32 and A256,
  full QA image incl. D, BS and zero tails): spec example `{127, 2.5, −0.5}`, every half-integer
  boundary, ties of the largest |x| with opposite signs, zero / subnormal / NaN / ±Inf blocks, amax
  10⁻⁴⁵…10³⁸, random bit patterns, 40k random vectors: **0 mismatches**. numpy fmul / fdiv / fadd /
  frecip / f32→fp16 vs `tsr_*` on 44,048 operations including the spec's FTZ boundary vectors:
  0 mismatches.
- **Comparer selftest (`numcheck.py selftest`, ctest `tools_numcheck_selftest`, ~4 s, 18 checks).**
  Boundtest, then a synthetic GGUF (q8_0, q4_K, q6_K, f16, f32 weights) with fake run dirs: clean →
  PASS with 6 weight MUL_MATs; a 1-ulp change of one device output → FAIL with exactly that
  row/token, 1 ulp; a 1 % change → caught by B1 and B2; the **sequential-sum mutant** as the q8_0
  tensor's device output (295 of 576 outputs differ) → FAIL by B1 and B1b with exactly those
  elements, 0 B2 violations; src1 dumps of 3 weights dropped → 3 not sampled, and with
  `--src1-from-producer` the mapping is verified on the other 3 (× 2 runs) and the 3 are checked with
  B1 / B3 results identical to the clean run; a producer differing from its src1 dump by 1 ulp → the
  fallback is refused (WARN, 3 not sampled); B3 with a token holding ~all of ‖Wx‖² (ratio 2 there,
  0.5 elsewhere) → listed as dominant, tensor ratio 0.5; 2 weight MUL_MATs off the device →
  INCOMPLETE (exit 3); a cpu run from another model, or `-m` that is not the dev run's model → exit
  2; `--dev-noobs` = the dev dir → exit 2, a proper `--no-observe` run → PASS with bit-identical
  logits; `--mm-layers out` → output head only; `--mm-layers bogus` and a crash (missing dump file) →
  exit 2 with no report left in `-o` (an older one is removed first).
- **Producer mapping on real graphs.** 338 / 338 src1 dumps (dev + cpu) bit-identical to the named
  producer's output on each 0.5B model, 72 / 72 on 32B.
- **Dumper selftest (`tsr-numcheck selftest`, ctest `tools_numcheck_dump_selftest`).** Read-back of
  permuted, transposed and strided/offset views and of F16 / BF16 / Q8_0 / I32 tensors against
  ggml's own CPU `CONT`; name and option parsing. A mutant with two strides swapped fails 8 checks.
- **Run consistency (`tsr-numcheck selfcheck ROOT`, run by `all`).** Run identities; identical node
  lists across cpu / floor / dev; every dump file present with the right size; cpu / floor entirely
  on CPU, every dev weight MUL_MAT on Tesseract; dev == dev-noobs logits and cpu == cpu-tok logits
  bit for bit (when present); the floor moved. Logits files that do not hold exactly T × V floats
  fail the header check and are not compared (no read past the buffer). `all` removes ROOT/cpu-tok
  and ROOT/dev-noobs left by an earlier `all` when the current `--checks` level does not redo them,
  so selfcheck never compares new runs with stale ones.
- **Further checks on 0.5B Q8_0** (dumper development): a second `all --checks full` reproduced all
  775 files of cpu, floor, dev bit for bit (also after the dumper was rebuilt: the runs used for the
  results below were re-made with the final binary and are byte-identical to the first ones); every
  src1 dump is byte-identical to an earlier node's output (its producer); `TSR_SIM_TMAX=8`
  reproduced all 775 dev files; a `--nodes` / `--mm-layers` filtered run matched the unfiltered one.
- **Independent spec model on real tensors before it became B1b.** A one-off numpy script during
  integration (own Q5_0 → Q8_0 requantization as well) reproduced 7 tensors bit for bit. That check
  is now B1b and runs on every weight MUL_MAT of every compare (results below). Sequential-sum
  mutant on real tensors: differs in 67 % (0.5B `blk.3.ffn_down`, q4_K) to 87 % (`blk.5.ffn_down`,
  q6_K → q8_0) of the elements; 32B `blk.35.ffn_down` 84.9 %.

## Usage

Build (only in `build/numcheck`; Release, the options of `build/dev`; at most -j4):

```
cmake -S /home/limagin-dev/tesseract -B /home/limagin-dev/tesseract/build/numcheck -DCMAKE_BUILD_TYPE=Release
cmake --build build/numcheck -j4 --target tsr-numcheck numcheck_helper tsr_ref_shared   # + llama-quantize if needed
```

Dumps (one process per run; `all` runs them as children, records exit code, wall time and peak RSS
of each in ROOT/`runs.json`, then runs `selfcheck`):

```
tsr-numcheck all -m MODEL [-o ROOT] [-T N] [--seed 1] [--nodes REGEX] [--mm-layers LIST] [--prompt TEXT]
                 [-t N] [--device sim] [--checks none|cheap|full]
tsr-numcheck run -m MODEL -o RUN_DIR --run cpu|floor|dev [--device sim] [-T N] [--seed 1] [--nodes REGEX]
                 [--mm-layers LIST] [--no-observe] [--input tokens|embd] [--prompt TEXT] [-t N] [-v]
tsr-numcheck selfcheck ROOT
tsr-numcheck selftest
```

- `--checks`: `none` = cpu, floor, dev; `cheap` (default) adds `cpu-tok`; `full` also `dev-noobs`.
  Runs of an earlier `all` in the same ROOT that the current level does not redo are removed first.
  Without `-o`, a private temporary directory is used and removed (ctest).
- `--nodes REGEX` (ECMAScript, searched in the node name) dumps a subset; indices stay those of the
  unfiltered graph.
- `--mm-layers LIST` restricts src1 dumps (and with them the B checks) of weight MUL_MATs: layer
  numbers and ranges `A-B`, `output` for weights outside `blk.N` (output head), `all` (default) or
  `none`. A weight MUL_MAT outside the list keeps `weight` / `weight_type` with `src1_file = null`.
- `--no-observe`: no callback, logits only (`nodes = []`).

Comparison (`--build DIR` goes before the subcommand; exit code 0 = PASS, 1 = FAIL, 3 = INCOMPLETE,
2 = usage / input error or crash, in which case `-o` holds no report):

```
python3 sw/tools/numcheck.py compare --cpu R/cpu --floor R/floor --dev R/dev [--dev-noobs R/dev-noobs] \
        -o R/report [-m MODEL] [--mm-layers 0,1,31,output] [--src1-from-producer] [--threads 4] \
        [--boundtest-report FILE]
python3 sw/tools/numcheck.py boundtest [--rows 1000000] [-o FILE]   # model-free B2 / QACT validation
python3 sw/tools/numcheck.py selftest -o DIR                        # boundtest + synthetic runs + injected faults
```

`-m` defaults to the model in the dev run's `meta.json` (for split files the first shard; the
others are found by name); if given, it must be that model. `--mm-layers` takes the dumper's
syntax (`N`, `A-B`, `output` / `out` / `head`, comma-separated, or `all`). Without
`--boundtest-report` compare runs the boundtest inline (~3 s); with it, a passed boundtest JSON of
the current version (3: B2 bound + numpy spec model) is reused, older ones are refused (rerun
`boundtest`). Weight rows are read in chunks of about `--chunk-elems / (K/32 · T)` rows, so the
comparer stays below 0.6 GB RSS on 0.5B and about 1 GB on 32B.

The three models of the results section:

```
# 1. Qwen2.5-0.5B Q8_0
tsr-numcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o build/numcheck-runs/q8_0-0.5b -T 32 --checks full
# 2. Qwen2.5-0.5B Q4_K_M, requantized locally from the Q8_0 file
build/numcheck/bin/llama-quantize --allow-requantize models/qwen2.5-0.5b-instruct-q8_0.gguf \
    models/qwen2.5-0.5b-instruct-q4_k_m-requant.gguf Q4_K_M
tsr-numcheck all -m models/qwen2.5-0.5b-instruct-q4_k_m-requant.gguf -o build/numcheck-runs/q4_k_m-0.5b -T 32 --checks full
# 3. Qwen2.5-32B Q4_K_M (file-backed simulator DRAM, ~20 GiB free needed)
mkdir -p build/simdram
TSR_SIM_THREADS=4 TSR_SIM_DRAM_DIR=/home/limagin-dev/tesseract/build/simdram \
  tsr-numcheck all -m models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf -o build/numcheck-runs/q4_k_m-32b \
    -T 16 --mm-layers 0,1,31,62,63,output --checks none
# each:
python3 sw/tools/numcheck.py boundtest -o build/numcheck-runs/boundtest.json
python3 sw/tools/numcheck.py compare --cpu R/cpu --floor R/floor --dev R/dev [--dev-noobs R/dev-noobs] \
    -o R/report --boundtest-report build/numcheck-runs/boundtest.json --src1-from-producer
```

ctest: `tools_numcheck_selftest` (comparer, no model), `tools_numcheck_dump_selftest` (dumper, no
model), `tools_numcheck_dump_qwen05b` (label `model`: `all --checks full` on 0.5B Q8_0, ~9 s).

## File format

### Run directory (version 1)

```
<run_dir>/meta.json           written last (atomic rename): the run is complete iff it exists
<run_dir>/logits.f32          f32 [T][n_vocab]
<run_dir>/t/NNNNNN.f32        node i's output
<run_dir>/t/NNNNNN.src1.f32   src1 of weight MUL_MAT i (only within --mm-layers)
```

All `.f32` files are little-endian float32 in ggml element order (ne0 fastest). Non-contiguous
tensors are gathered element by element; F16 / BF16 / I32 outputs are converted to f32 and quantized
outputs dequantized with ggml's `to_float` (none occur with an f16 KV cache).

`meta.json`: `format` `"tsr-numcheck"`, `version` 1, `model` (absolute path of the GGUF given, the
first split), `run`, `device` (`none` | `sim` | `xrt`), `input` (`embd` | `embd+1ulp`; `tokens` only
for the cpu-tok check run), `seed`, `n_tokens`, `tokens`, `n_vocab`, `logits_file`, `info`
(settings, prompt, timings, counts, `TSR_*` environment) and `nodes`. Per node:

| Field | Meaning |
| --- | --- |
| `i` | index among the **computed non-view** nodes, in graph order. Skipped: `NONE`, `VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, and nodes without `GGML_TENSOR_FLAG_COMPUTE` (llama.cpp builds all input-embedding branches with `ggml_build_forward_select` and computes only the selected one; the others hold stale memory; `info.n_not_computed`, 4 for Qwen2) |
| `name`, `op`, `ne` | `ggml_tensor.name`, `ggml_op_name`, `[ne0, ne1, ne2, ne3]`. **Names are not unique** (`Qcur-0` is the MUL_MAT, the bias ADD and the ROPE): match nodes by `i` |
| `file` | `t/<i, 6 digits>.f32` |
| `backend` | `ggml_backend_dev_name` of the device owning the node's output buffer (`CPU` or `Tesseract0`) = the backend that computed it |
| `weight`, `weight_type` | for `MUL_MAT` / `MUL_MAT_ID` whose src0 is a leaf named like a GGUF tensor (a scheduler copy `<backend>#<name>#<n>` resolves to `<name>`): that name and `ggml_type_name(src0->type)` (the GGUF type, e.g. `q6_K` also when the device converted it); else `null` |
| `src1_file`, `src1_ne` | weight MUL_MATs within `--mm-layers`: the src1 the computing backend read (on dev the Tesseract input copy); else `null` |

Qwen2 graphs have 25 dumped nodes per layer (RMS_NORM, MUL, Q / K / V MUL_MATs with bias ADDs and
ROPEs, two SET_ROWS, `kq` MUL_MAT, SOFT_MAX, `kqv` MUL_MAT, CONT, `attn_output` MUL_MAT, residual
ADD, RMS_NORM, MUL, `ffn_gate` / `ffn_up` MUL_MATs, GLU, `ffn_down` MUL_MAT, residual ADD) plus the
last layer's two `GET_ROWS` (output ids), `norm`, `result_norm`, `result_output`: 605 nodes for 24
layers, 1605 for 64. Weight MUL_MATs: 7 per layer + the output head. On dev all of them run on
Tesseract0; the attention MUL_MATs (`kq`, `kqv`; KV cache on the host) and every other node run on
the CPU.

### Report directory

`report.md` (human-readable), `report.json` (version 2; everything except per-node rows:
`selfcheck`, `logits`, `drift`, `mm` with per-tensor entries (`b1`, `b1b`, `b2`, `b3` with
`dominant_tokens`, the all-token values and `per_token` errors) and per-type entries, `verdict`,
`fails`, `coverage_gaps`, `warnings`), `drift_nodes.json` (one row per node). All three are removed
when compare starts and written at the end (temporary name + rename). ROOT/`runs.json` (from
`all`): exit code, wall time and peak RSS (`wait4`) per run.

## Runtime

Host: 16 cores, 14 GB RAM; `TSR_SIM_THREADS=4`, T_MAX = 1, `-t 8`. Peak RSS of the 32B runs is
mostly mmap'd GGUF / simulator-DRAM file pages (reclaimable page cache), not anonymous memory.

| Model | T | cpu | floor | dev | checks | dumps per run | compare |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 0.5B Q8_0 | 32 | 0.24 s, 776 MiB | 0.24 s, 776 MiB | 3.8 s, 1215 MiB | cpu-tok 0.17 s, dev-noobs 3.9 s | 155 MiB (605 nodes + 169 src1) | 31 s, 549 MiB |
| 0.5B Q4_K_M | 32 | 0.21 s, 602 MiB | 0.21 s, 602 MiB | 4.7 s, 1014 MiB | cpu-tok 0.15 s, dev-noobs 4.6 s | 155 MiB | 33 s, 550 MiB |
| 32B Q4_K_M | 16 | 15.9 s, 12.4 GiB | 15.9 s, 12.6 GiB | 107 s (load + conversion 47 s, decode 59 s), 11.8 GiB | dev-noobs 103 s (made separately with `run --no-observe`) | 841 MiB (1605 nodes + 36 src1) | 388 s, 1.0 GiB (all 449 weight MUL_MATs, B1b and `--dev-noobs` included; 275–315 s before B1b) |

- 32B `tsr-numcheck all` (cpu, floor, dev, selfcheck): **2 min 21 s** wall. The dev decode (59 s
  for 16 tokens) is far below the 16 × ~15 s per-token estimate of `docs/klcheck.md`. A likely
  reason (not measured): a prefill MUL_MAT makes its 16 passes (T_MAX = 1) over one weight matrix
  back to back, so only the first pass reads the simulator DRAM files from disk and the others hit
  the page cache.
- The boundtest alone takes 2.8 s (3.7 s wall, 73 MiB) with the spec-model check; the comparer
  selftest 4 s. B1b doubles the 0.5B compare time (15 s → 31 s, mostly the 152k-row output head).
- Local Q4_K_M requantization of the 0.5B model: 0.9 s. Disk: 559 MiB per 0.5B root, 2.5 GiB for 32B.

## Results

All three models: **PASS**. One prompt per model (the built-in sentence, 32 or 16 tokens), one
prefill batch; reports in `build/numcheck-runs/<tag>/report/`, the shared boundtest in
`build/numcheck-runs/boundtest.json` (version 3). No sim, backend or spec defect was found. The
numbers below come from the final comparer on the existing dumps (the dumper's changes since then
touch only `selfcheck` and `all`, not what a run writes).

| | 0.5B Q8_0 (T = 32) | 0.5B Q4_K_M requant (T = 32) | 32B Q4_K_M (T = 16) |
| --- | --- | --- | --- |
| weight MUL_MATs (all on Tesseract0) | 169: q8_0 169 | 169: q5_0 132 → q8_0, q8_0 13, q4_K 12, q6_K 12 → q8_0 | 449: q4_K 384, q6_K 65 → q8_0 |
| **B1** (libtsr_ref) mismatches / elements | **0 / 14,594,048** | **0 / 14,594,048** | **0 / 76,881,920** (36 with src1 dumps, 413 with x from the producer node) |
| **B1b** (numpy spec model) mismatches | **0** | **0** | **0** |
| **B2** violations / elements | **0 / 14,594,048** | **0 / 14,594,048** | **0 / 76,881,920** |
| B2 \|err\|/bound median / p99 / max | 0.025 / 0.123 / 0.509 | 0.025 / 0.123 / 0.505 | 0.010 / 0.070 / 0.393 |
| B2 out of domain / FTZ slack active | 0 / 0 | 0 / 0 | 0 / 0 |
| numpy QACT vs `tsr_qact` on real x | identical (169 × 32 vectors) | identical | identical (449 × 16) |
| **B3** dev/cpu, native types: median (range); pooled | q8_0 1.001 (0.965–1.044); 0.971 | q8_0 1.000 (0.971–1.016); 1.002 · q4_K 1.000 (0.991–1.018); 1.000 | q4_K 0.999 (0.919–1.026); 0.997 |
| B3 dev/cpu, converted types: median (range); pooled | – | q5_0 1.251 (1.058–1.759); 1.240 · q6_K 0.520 (0.471–0.619); 0.510 | q6_K 0.615 (0.190–0.838); 0.423 |
| B3 tensors with a dominant token (token 0, > 50 % of ‖Wx‖²) | 4 | 4 | 10 (7 of them q6_K `ffn_down`) |
| KL(cpu‖dev) mean / KL(cpu‖floor) mean / **ratio** | 1.477e-3 / 1.068e-3 / **1.38** | 2.610e-3 / 1.788e-3 / **1.46** | 3.894e-3 / 1.835e-3 / **2.12** |
| ratio over positions where the floor moved | 1.13 (pos. 7–31) | 1.29 (pos. 4–31) | 2.06 (pos. 1–15) |
| top-1 dev / floor vs cpu | 32/32 / 32/32 | 30/32 / 30/32 | 16/16 / 16/16 |
| dev == dev-noobs logits; cpu (embd) == cpu-tok | bit-identical; bit-identical | bit-identical; bit-identical | bit-identical; not run |
| first dev/cpu divergence | node 2 `Qcur-0` = first device node | same | same |
| layer-output NMSE ratio dev/floor (floor moved) | 1.41 (layer 0) → 1.12 (layer 23) | 2705 (layer 0) → 1.16 (layer 23) | 49 (layer 0), 258 (layer 2) → 1.85 (31) → 1.10 (63) |
| `result_output` NMSE dev / floor | 3.99e-4 / 2.90e-4 | 8.70e-4 / 6.67e-4 | 2.04e-3 / 1.26e-3 |

**What this shows.**
- **The device computes exactly what the spec says, on every weight MUL_MAT of all three models**
  (B1 and B1b, 106 M elements each). The spec text is reproduced
  twice: by `libtsr_ref` (B1) and by the independent numpy model (B1b), which shares no code with
  the simulator. This covers weight conversion (q5_0 / q6_K → q8_0 by `to_float` +
  `quantize_row_q8_0_ref`), repacking into chunks, row allocation over channels, T tiling, staging,
  QACT placement and the 16-partial / fixed-tree accumulation order, on real data, including the
  152k-row output heads.
- **Observed vs normal execution.** With `cb_eval` asking for every node, the scheduler hands the
  device one node per graph compute; normally several MUL_MATs share one program (e.g. `ffn_gate`
  and `ffn_up`). The two are tied by `dev-noobs`: dev logits without observation are bit-identical
  to the observed ones on all three models (32B: 16/16 rows, from a separate
  `tsr-numcheck run --run dev --no-observe -T 16` of 103 s, passed to compare as `--dev-noobs`).
  Observation therefore does not change device numerics on any of them.
- **The fixed fp32 order uses at most ~0.5 of the proven bound** (B2) on real data, and typically
  1–2.5 % of it. B2 bounds the rounding error of any evaluation order of the spec's operations; it
  does not check the order itself (B1 / B1b do).
- **Against exact math, the device is as accurate as the CPU on native types** (B3 ≈ 1.00 for q8_0
  and q4_K, every tensor within 0.92–1.04), as expected: same weights, same 8-bit activation rule,
  only the summation order differs.

**Converted types (B3).**
- **q5_0 → q8_0 costs ~25 %** more error than the CPU's direct q5_0 path (0.5B requant: K = 896 is
  not a multiple of 256, so llama-quantize fell back to q5_0 for 132 tensors). Q5_0 grid points are
  not on the Q8_0 grid (amax 16·d → d₈ = 16·d/127), so requantization adds up to half a Q8_0 step per
  weight. Within the spec's design (8-bit storage for every non-native type); a deliberate trade-off,
  not a defect.
- **q6_K → q8_0 is *more* accurate than the CPU on every q6_K tensor** (0.5B: median 0.52, max
  0.62; 32B: median 0.62, range 0.19–0.84, pooled 0.42). The CPU computes q6_K against Q8_K
  activations (one scale per 256 values); the converted device path uses A32 (one scale per 32).
  With outlier activations the 256-wide block wastes resolution: 32B `blk.1.ffn_down` (K = 27648)
  has CPU rel_err 15.0 % vs device 2.8 % over tokens 1–15 (9.2 % vs 1.7 % over all tokens); at token
  0, row 2159 the CPU gives 7.43 for an exact 8.74 (one x of 17.1 in a block with rms 0.127), the
  device 8.50. The output head: CPU 4.1 % vs device 1.7 %.
- **Earlier version of this document:** it reported a q6_K range up to 1.345 and called
  `blk.4.ffn_down` "the one exception, where the requantization cost shows". That was an artifact of
  pooling over tokens: token 0 holds 99.99 % of that tensor's ‖Wx‖², so the pooled value measured
  one token (1.57 there; tokens 1–15: 0.40–0.92, together 0.615). B3 now leaves such tokens out of
  the ratio and lists them (all-token values stay in the report).

**Drift and the floor.**
- **The floor partly vanishes.** Floor logits are bit-identical to cpu at positions 0–6 (0.5B
  Q8_0), 0–3 (0.5B Q4_K_M) and 0 (32B), so KL(cpu‖floor) is exactly 0 there. The 1-ulp nudge
  itself survives: at those positions the residual stream (every `RMS_NORM`, norm `MUL` and residual
  `ADD`) differs from cpu in every layer. What absorbs it is **each** int8 activation quantization
  of the CPU: all 169 weight MUL_MAT outputs, the output head included, are bit-identical at these
  positions (0.5B, both models). A 1-ulp change (≤ 2⁻²³ relative) moves `x/d` by at most
  127·2⁻²³ ≈ 1.5·10⁻⁵, so an int flips only for an element that close to a rounding boundary; a
  position stays unchanged until some quantization at that position, or at an earlier one through
  attention, happens to flip. This
  affects klcheck's floor as well. The reports also give ratios over the positions where the floor
  moved.
- **32B early layers.** Layer-output ratios of 50–300 in layers 0–3 (positions where the floor
  moved), and up to 66,404 over all positions from layer 4 on, are not device error.
  - From layer 4, token 0 carries a massive activation (`ffn_out-4`, channel 4675: cpu 7885.5,
    dev 8079.8) that dominates NMSE. There the floor is identical to cpu, so its NMSE is ~1e-8.
  - The 2.5 % difference comes from upstream: each run's own `ffn_down` is accurate for its input
    (cpu 7885.50 vs exact 7885.70; dev 8079.79 vs exact 8083.04). Their inputs `ffn_swiglu-4` already
    differ by 3.8 %.
  - The upstream difference starts in layers 0–3. There the CPU reference itself is the less
    accurate one: B3 on the q6_K `ffn_down` of layers 0–3 gives CPU 1.4 %, 15.0 %, 1.3 %, 2.7 % vs
    device 0.59 %, 2.8 %, 0.54 %, 0.84 % (layer 1 over tokens 1–15; no dominant token in the
    others).
  - Excluding token 0, the ratio falls to 1.85 by layer 31 and 1.10 at layer 63; `result_norm` 1.04.
- **Logit KL ratio.** 1.38 / 1.46 / 2.12, all below the spec's limit of 3. The 32B value from one
  16-token prompt is a coarse estimate: per-position KL ratios over positions 1–15 range from 0.2 to
  4.8. klcheck's 4 × 64-token run gave 1.36 on the same model. Part of the measured dev–cpu distance
  is the CPU's own q6_K error described above. KL against the CPU cannot separate that from device
  error; B1–B3 do.
- **0.5B Q4_K_M layer 0** (ratio 2705) is the q5_0 requantization: the device multiplies by different
  (requantized) weights from the first MUL_MAT on, while the floor hardly moves in layer 0.

**Not covered here.** MUL_MAT_ID (MoE) weights are tagged by the dumper but not modelled by B
(a model with them gets INCOMPLETE). The ACC accumulate path is not used by these graphs. The
attention MUL_MATs run on the CPU (KV cache on the host, v1). Observation invariance on 32B (see
above).

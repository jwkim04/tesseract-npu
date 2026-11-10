# numcheck: one-prompt numeric verification

numcheck checks, with one short prompt per model, that the device computes exactly what the specification
says on every weight matmul, and how far its results are from exact math. It is bit-exact where
[klcheck](klcheck.md) is statistical. Normative: `docs/spec.md` §7 (Numeric rules) and §17 (Verification
strategy, *Model-level numeric check (numcheck)*), [SW](design/06-software-verification.md) §10.1 and §10.4,
SW-22 and SW-27.

- One short prompt (a fixed English sentence, T tokens), one prefill batch, logits for every position.
- Three runs per model, each its own process: `cpu`, `floor` (cpu with a 1-ulp nudged input) and `dev`
  (Tesseract, on any transport kind). Every run records every graph node.
- The comparison is offline. (A) The drift of every node and of the logits. (B) Every weight MUL_MAT checked
  element by element: B1 bit-exact against the reference model (`libtsr_ref`), B1b bit-exact against an
  independent numpy model of the spec text, B2 inside a proven fp32 error bound, B3 the error against exact math
  compared with the CPU's.

Tools: `tsr-numcheck` (`sw/tools/numcheck.cpp`, the dumper, links llama.cpp) and `sw/tools/numcheck.py` (the
comparer; native helpers `sw/tools/numcheck_helper.c` and `libtsr_ref`).

## Method

### Runs

| Run | Backend | Input | Purpose |
| --- | --- | --- | --- |
| `cpu` | ggml CPU only (`TSR_DEVICE=none`, `n_gpu_layers = 0`) | `embd`: `token_embd.weight` rows, ggml `to_float` (bit-exact with the token path) | reference |
| `floor` | same as `cpu` | `embd+1ulp`: every element `nextafterf` +-1 ulp (klcheck's `embd_nudge`, sequence 0, `--seed`, default 1) | the CPU's own sensitivity |
| `dev` | Tesseract (`TSR_DEVICE` from `--device`; `n_gpu_layers = 999`: every layer and the output head) | `embd` | device under test |
| `cpu-tok` (check) | CPU, no observation | token ids | embd path equals token path, observing does not change CPU numerics |
| `dev-noobs` (check) | Tesseract, no observation | `embd` | observing does not change device numerics |

All three main runs use the embd input path, so their graphs are identical node for node. Model and context
settings are klcheck's ([klcheck.md](klcheck.md), "Settings fixed in every phase"): flash attention on, a q8_0 K
and V cache, `LLAMA_ATTN_ROT_DISABLE=1` (set by the tool), `offload_kqv` false with `--phase m6` (the default)
and true with `--phase m7`. Each run records them in `meta.json` `info`, and `selfcheck` and `numcheck.py
compare` refuse a set whose runs differ in them. Further: CPU weight repack off, `-t` threads, `n_ctx` = T
rounded up to 256, `n_batch = n_ubatch = 256` (T must fit one ubatch), every position an output. T defaults to 16
for models above 8 B parameters and 32 otherwise. `TSR_SIM_THREADS` defaults to 4 when unset. The device kinds
follow the phase as in klcheck: `--phase m6` sets `TSR_KINDS=m6`, and `--phase m7` keeps the caller's
`TSR_KINDS` (recorded in `meta.json` `info.TSR_KINDS`).

Observation uses `llama_context_params.cb_eval`. The callback asks for every computed non-view node, so the
scheduler computes the graph one node at a time (one device launch per observed node: numcheck needs no device
mode of its own, SW-27), and the node's output (and, for a weight MUL_MAT, its src1) is read with
`ggml_backend_tensor_get`. On `dev` the src1 read is the scheduler's input copy in the Tesseract buffer, that is,
exactly the activation the device quantized.

### (A) Drift

For every node (matched by index `i`; the node lists of the three runs must be identical): NMSE(dev vs cpu),
NMSE(floor vs cpu), their ratio, max |delta|, cosine, and whether dev and cpu are bit-identical. Summaries are
given per layer at `l_out-N`, `result_norm` and `result_output`, for the worst nodes, and for the first node
where dev diverges from cpu (it must be the first device node). Logits: KL(cpu||dev) and KL(cpu||floor) per
position in float64, top-1 agreement, and the positions where the floor did not move at all. The layer table
also gives the NMSE ratio over the positions where the floor moved ("ratio, floor moved").

### (B) Every weight MUL_MAT of the dev run

A weight MUL_MAT is one whose src0 is a GGUF tensor. The weight rows are read from the GGUF (all splits) in row
chunks and converted to the device format as `sw/ggml-tesseract/tsr-weights.cpp` does (native Q8_0 and Q4_K bytes
unchanged, every other type ggml `to_float` then `quantize_row_q8_0_ref`) with ggml's own functions through
`libnumcheck_helper.so`.

- B1, bit-exact. `y_dev == nc_model_mm(device blocks, x_dev)` bit for bit for every element. `nc_model_mm` runs
  `tsr_qact` once per token and `tsr_repack_row` plus `tsr_dot_row` once per row from `libtsr_ref` (the oracle).
  This covers the backend's conversion, repacking, row allocation over channels, T tiling, staging and QACT
  placement on real data. Any mismatch is a defect, and the report lists row, token, both values and the ulp
  distance.
- B1b, bit-exact against the spec text. `y_dev == spec_model_y(...)` bit for bit, where `spec_model_y`
  (`numcheck.py`) evaluates the numeric rules of `docs/spec.md` §7 in numpy without any `libtsr_ref` code:
  activation ints and scales from the numpy QACT model `np_qact`, weight ints and scales parsed in numpy, block
  terms `(d*Dx)*f32(isum)` (Q4_K: `- (dmin*Dx)*f32(msum)`) with every multiply and subtract rounded separately in
  binary32 with DAZ and FTZ, 16 partials `p_(i mod 16)` from +0.0, then the left-paired 4-level tree. B1 and B1b
  together make the result independent of either model alone: the simulator's MVM and the B1 model both call
  `tsr_dot_row`, so B1 alone would not notice a defect shared by `libtsr_ref` and the simulator.
- B2, proven bound. `|y_dev - y_q64| <= bound` for every element, where `y_q64` is the float64 value of the same
  quantized operands (spec formulas, no rounding to f32). B2 is a worst-case rounding envelope. It catches gross
  errors (a wrong scale, operand or dropped term) without any reference model, but not the summation order: a
  sequential sum stays inside it. The order is checked by B1 and B1b.
- B3, exact-math error. `rel_err = ||y - W x|| / ||W x||`, where W is ggml `to_float` of the original GGUF weight
  in float64 and x the run's own src1, for dev and cpu, per token first. A token that holds more than half of a
  tensor's `||Wx||^2` is dominant. It is listed and left out of the tensor's ratio, because pooled over all
  tokens it alone would decide the value (a massive activation at token 0 does this in some `ffn_down` tensors).
  The report keeps the all-token values in separate columns, and gives per weight type the pooled ratio and the
  median and range of per-tensor ratios.
- `--src1-from-producer`. A weight MUL_MAT outside the dumper's `--mm-layers` has no src1 dump, but its src1 is
  the output of a node that is always dumped (`attn_norm-N` for attn_q, k and v, `ffn_norm-N` for ffn_gate and
  up, `ffn_swiglu-N` for ffn_down, `result_norm` for the head, the layer's FLASH_ATTN_EXT node for attn_output).
  The comparer first verifies the mapping bit for bit on every src1 dump of the dev and cpu runs, and only then
  uses the producer output as x for the other weight MUL_MATs (otherwise WARN and those stay unsampled). A wrong
  x cannot hide a defect, since B1 would fail.

Verdict (exit code 0 PASS, 1 FAIL, 3 INCOMPLETE, 2 usage or input error or a crash, with no report written):

- PASS iff B1 and B1b have 0 mismatches, B2 has 0 violations, the numpy QACT model equals `tsr_qact` on the real
  activations, the model-free boundtest passed (at least 10^6 dot products in domain), `dev-noobs` (if given) is
  bit-identical, and every selected weight MUL_MAT was checked on the device.
- FAIL if any of the above fails. INCOMPLETE if nothing failed but a selected weight MUL_MAT could not be
  checked: it ran off the device, its weight is missing from the GGUF, its shapes do not match, or its op is
  MUL_MAT_ID (MoE, not modeled).
- WARN: a native tensor with B3 ratio above 1.5, weight MUL_MATs without a src1 dump, an unverified producer
  mapping, B2 out-of-domain elements, dev diverging before the first device node, run directories of the wrong
  kind, no `--dev-noobs`.
- Input checks (exit 2 before any work). cpu, floor, dev (and `--dev-noobs`) must name the same model file with
  the same `info.model_desc`, `n_params`, `n_layer`, `n_embd`, tokens and `n_vocab`. `-m` must be the dev run's
  model. Every weight MUL_MAT node's `weight_type` and shapes must match the GGUF tensor. `--dev-noobs` must be a
  dev run with input `embd` made with `--no-observe`.

### Reading the report

- A native type (Q8_0, Q4_K) has a B3 ratio dev / cpu of about 1: same weights, same 8-bit activation rule, only
  the summation order differs. A converted type shows the requantization cost. Q5_0 to Q8_0 costs accuracy
  against the CPU's direct Q5_0 path, since Q5_0 grid points are not on the Q8_0 grid. Q6_K to Q8_0 is more
  accurate than the CPU, because the CPU computes Q6_K against Q8_K activations (one scale per 256 values) while
  the converted device path uses A32 (one scale per 32). KL against the CPU cannot separate the CPU's own error
  from the device's. B1 to B3 do.
- The floor can vanish. At early positions the floor logits can be bit-identical to cpu, so KL(cpu||floor) is
  exactly 0 there. The nudge survives in the residual stream, but each int8 activation quantization of the CPU
  absorbs it: a 1-ulp change moves `x/d` by at most 127 * 2^-23, so an int flips only for an element that close
  to a rounding boundary. This affects klcheck's floor as well.
- Large layer-output ratios in early layers of a big model, caused by one massive activation at token 0, are not
  device error. The CPU reference can be the less accurate side there (check B3 of those `ffn_down` tensors).
- MUL_MAT_ID weights are tagged by the dumper but not modeled by B (INCOMPLETE). The ACC accumulate path is not
  used by weight MUL_MATs here and is not covered by B2.

### B2 bound: derivation

Notation: `u = 2^-24`, `lambda = 2^-126`, `gamma_n = n*u / (1 - n*u)`. `fl(.)` is one f32 operation under the spec
rules (`docs/spec.md` §7 *Common fp rules*, §21 *FTZ decision point*): DAZ on inputs, IEEE RNE, then FTZ.

- One operation: if `|x| >= 2^-126` then `fl(x) = x(1+delta)` with `|delta| <= u`, otherwise `fl(x) = x + eta` with
  `|eta| < lambda`. Only one of delta and eta is nonzero for a given operation. `d` and `dmin` (fp16 to f32) are
  exact, and the integer sums of a block convert exactly except for the Q4_K `isum`. A product of an f32 result and
  an integer-valued factor never flushes, so inside a term only `d*Dx`, `dmin*Dx` and the Q4_K subtraction can flush.
- A Q8_0 term `t = fl(fl(d*Dx)*isum)` with exact value `T` obeys `|t - T| <= gamma_2 |T| + c1 lambda (1+u) |isum|`
  with `c1 = [|d*Dx| < 2^-125]`. A Q4_K term (`d*Dx*isum` minus `dmin*Dx*msum`, with `M = |T1| + |T2|`) obeys `|t - T|
  <= gamma_4 M + eps`, where eps collects the lambda terms of the flushes.
- Row accumulation over n terms (spec §7 *Row accumulation*): partial `p_(i mod 16)` starts at +0 and the fixed tree
  adds 4 levels, so every term passes through at most `h = ceil(n/16) + 3` rounded additions, and at most `n + 15`
  additions can each flush once. Standard summation analysis (Higham, *Accuracy and Stability of Numerical
  Algorithms*, Lemma 3.1) gives, with `S0 = sum M_i`, `E = sum eps_i`, `Y = sum T_i` and `gc = gamma_2` (Q8_0) or
  `gamma_4` (Q4_K):

  `|y - Y| <= gc*S0 + E + gamma_h*((1+gc)*S0 + E) + (n+15)*lambda*(1+gamma_h)`

- `y_q64` evaluates the same integers and scales in float64. The bound adds `gamma64_(n+4) * S0 * 1.01` (u64 =
  2^-53) for the float64 evaluation and is multiplied by `1 + 2^-40`. An element violates B2 iff `|y_dev - y_q64| >
  bound`. The domain is all values finite, `S0 < 2^126` and `|d*Dx|, |dmin*Dx| < 2^126`. Elements outside it are
  counted, not judged.
- Independence from `tsr_ref` (B1b, B2): `np_qact` uses numpy's native binary32 operations with explicit DAZ, FTZ and
  canonical NaN, weight scales and nibbles are parsed in numpy, the integer dots use float32 matmul (exact because
  every partial sum is an integer below 2^24), and partials are padded to a multiple of 16 terms with -0.0, the
  exact identity of RNE addition. The only code shared with B1 is ggml's weight conversion, which defines the device
  format.

### Validation of the tools

- `numcheck.py boundtest` draws random dot products from `libtsr_ref` over many operand families (quantized Gaussian
  and heavy-tailed weights, raw q including -128, extreme `isum`, exact cancellation, any f32 scale including flushes
  of `d*Dx`, Q4_K with random scales and nibbles, `|isum| > 2^24`, tiny and huge amax). B2 must have 0 violations and
  the B1b model 0 mismatches, Inf, NaN and FTZ rows included. It asserts the bound's sensitivity (halving it must
  produce violations) and a negative control (the sequential-sum mutant must differ from `libtsr_ref` on a sizeable
  share of the rows). `np_qact` is compared with `tsr_qact` on A32 and A256 vectors (the spec example `{127, 2.5,
  -0.5}`, every half-integer boundary, ties of the largest |x|, zero, subnormal, NaN and Inf blocks, random bit
  patterns), and numpy's `fmul`, `fdiv`, `fadd`, `frecip` and f32-to-fp16 with `tsr_*` on the spec's FTZ boundary
  vectors.
- `numcheck.py selftest` (ctest `tools_numcheck_selftest`) runs the boundtest, then a synthetic GGUF with fake run
  directories. A clean set passes. A 1-ulp change of one device output fails with exactly that row and token, a 1 %
  change is caught by B1 and B2, and the sequential-sum mutant fails by B1 and B1b with 0 B2 violations. Dropped
  src1 dumps are checked with `--src1-from-producer` with identical results. A cpu run from another model, a wrong
  `-m` or `--dev-noobs` equal to the dev directory exit 2.
- `tsr-numcheck selftest` (ctest `tools_numcheck_dump_selftest`) reads back permuted, transposed and strided views and
  F16, BF16, Q8_0 and I32 tensors against ggml's own CPU `CONT`, and tests name and option parsing.
  `tsr-numcheck selfcheck ROOT` (run by `all`) checks run identities, identical node lists, every dump file present
  with the right size, cpu and floor entirely on the CPU, every dev weight MUL_MAT on Tesseract, dev equal to
  dev-noobs logits and cpu equal to cpu-tok logits bit for bit, and that the floor moved.

## Usage

```
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j8 --target tsr-numcheck numcheck_helper tsr_ref_shared

tsr-numcheck all -m MODEL [-o ROOT] [-T N] [--seed 1] [--nodes REGEX] [--mm-layers LIST] [--prompt TEXT]
                 [-t N] [--device sim|mock|vkernel|xrt] [--phase m6|m7] [--checks none|cheap|full]
tsr-numcheck run -m MODEL -o RUN_DIR --run cpu|floor|dev [--device sim] [--phase m6|m7] [-T N] [--seed 1]
                 [--nodes REGEX] [--mm-layers LIST] [--no-observe] [--input tokens|embd] [--prompt TEXT] [-t N] [-v]
tsr-numcheck selfcheck ROOT
tsr-numcheck selftest

python3 sw/tools/numcheck.py --build build/dev compare --cpu R/cpu --floor R/floor --dev R/dev \
        [--dev-noobs R/dev-noobs] -o R/report [-m MODEL] [--mm-layers 0,1,31,output] \
        [--src1-from-producer] [--threads 4] [--boundtest-report FILE]
python3 sw/tools/numcheck.py --build build/dev boundtest [--rows 1000000] [-o FILE]   # model-free B2 and QACT validation
python3 sw/tools/numcheck.py --build build/dev selftest -o DIR                        # boundtest, synthetic runs, injected faults
```

`all` runs the dumps as children, records exit code, wall time and peak RSS of each in `ROOT/runs.json`, then runs
`selfcheck`. `numcheck.py` needs Python 3 with numpy. It reads GGUF metadata with the `gguf-py` of the pinned
llama.cpp and finds `libtsr_ref.so` and `libnumcheck_helper.so` in the directory given by `--build` (before the
subcommand; default `build/numcheck`).

- `--checks`: `none` is cpu, floor, dev. `cheap` (default) adds `cpu-tok`. `full` also adds `dev-noobs`. Runs of an
  earlier `all` in the same ROOT that the level does not redo are removed first. Without `-o` a private temporary
  directory is used and removed.
- `--nodes REGEX` (ECMAScript, searched in the node name) dumps a subset. Indices stay those of the unfiltered
  graph. `--no-observe` records logits only.
- `--mm-layers LIST` restricts src1 dumps (and with them the B checks) of weight MUL_MATs: layer numbers and
  ranges `A-B`, `output` (also `out`, `head`) for weights outside `blk.N`, `all` (default) or `none`.
- `compare` takes `-m` from the dev run's `meta.json` by default. Without `--boundtest-report` it runs the
  boundtest inline. With it, a passed boundtest JSON of the current version (3) is reused. Weight rows are read in
  chunks, so the comparer's memory does not depend on the model size.

A model that does not fit in RAM runs with file-backed simulator DRAM (`TSR_SIM_DRAM_DIR`) and a sampled dump
(`--mm-layers 0,1,31,output --checks none`, then `compare --src1-from-producer`). ctest: `tools_numcheck_selftest`,
`tools_numcheck_dump_selftest`, `tools_numcheck_dump_qwen05b` and `tools_numcheck_kinds_qwen05b` (label `model`:
the dump on Qwen2.5-0.5B Q8_0, and `info.TSR_KINDS` with the backend's kinds for both phases).

## File format

Run directory (format version 2):

```
<run_dir>/meta.json           written last (atomic rename): the run is complete iff it exists
<run_dir>/logits.f32          f32 [T][n_vocab]
<run_dir>/t/NNNNNN.f32        node i's output
<run_dir>/t/NNNNNN.src1.f32   src1 of weight MUL_MAT i (only within --mm-layers)
```

All `.f32` files are little-endian float32 in ggml element order (ne0 fastest). Non-contiguous tensors are
gathered element by element. F16, BF16 and I32 outputs are converted to f32 and quantized outputs dequantized
with ggml's `to_float`. `meta.json` has `format` `"tsr-numcheck"`, `version` 2 (version 1 dumps, with flash
attention off and an f16 KV cache, are still read), `model`, `run`, `device`, `input`, `seed`, `n_tokens`,
`tokens`, `n_vocab`, `logits_file`, `versions` (capability, Basis version, the kind table with every kind's
revision), `info` (settings as `phase`, `flash_attn`, `type_k`, `type_v`, `offload_kqv` and `attn_rot_disable`,
the prompt, timings, `TSR_*` environment) and `nodes`. Per node:

| Field | Meaning |
| --- | --- |
| `i` | index among the computed non-view nodes, in graph order. Skipped: `NONE`, `VIEW`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, and nodes without `GGML_TENSOR_FLAG_COMPUTE` (llama.cpp builds all input-embedding branches and computes only the selected one). |
| `name`, `op`, `ne` | `ggml_tensor.name`, `ggml_op_name`, `[ne0, ne1, ne2, ne3]`. Names are not unique (`Qcur-0` is the MUL_MAT, the bias ADD and the ROPE): match nodes by `i`. |
| `file` | `t/<i, 6 digits>.f32` |
| `backend` | the device owning the node's output buffer (`CPU` or `Tesseract0`), that is, the backend that computed it |
| `weight`, `weight_type` | for `MUL_MAT` and `MUL_MAT_ID` whose src0 is a leaf named like a GGUF tensor: that name and the GGUF type (`q6_K` also when the device converted it). Otherwise `null`. |
| `src1_file`, `src1_ne` | weight MUL_MATs within `--mm-layers`: the src1 the computing backend read. Otherwise `null`. |

Report directory: `report.md` (human-readable), `report.json` (format version 2: `selfcheck`, `logits`,
`drift`, `mm` with per-tensor and per-type entries (`b1`, `b1b`, `b2`, `b3` with `dominant_tokens`), `verdict`,
`fails`, `coverage_gaps`, `warnings`, `versions`) and `drift_nodes.json` (one row per node). All three are removed
when compare starts and written at the end (temporary name and rename).

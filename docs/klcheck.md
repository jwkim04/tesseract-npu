# klcheck: model-level KL check

`sw/tools/klcheck.cpp` implements the whole-model row of the oracle chain in `docs/spec.md` §17
(Verification strategy) and its *Model-level KL criterion*: the Tesseract device is compared with
the ggml CPU backend on whole-model logits, against a floor given by the CPU compared with itself
under a 1-ulp input perturbation. Pass (the specification's values): **mean KL(CPU ‖ device) ≤ 3 ×
floor and top-1 agreement ≥ 80 %**, where the floor is the mean over three floor runs with different
seeds (`--floor-seeds`, below). The check is part of the M1 and M4 gates (done) and of the M6 and M7
gates (measured on the simulator; M6 also on the card with the reduced `k4m200` image and the full `k4full` image, rows below; M7 also on the card with the full image, with attention and the KV cache on the device since the ATTPV fix, rows below).

> **Scope.** User documentation and measurement record of `klcheck`. Status words: *on the
> simulator* means a run through the software stack with `TSR_DEVICE=sim` or `mock`; *on the card*
> means a run on the Alveo U200. Card measurements, and the rows with the FA-off settings (below),
> were taken before the control core was added, with the host issuing the ops, except the `k4m200` rows. Every result names its
> settings and provenance, and results with different settings are not comparable. Normative:
> `docs/spec.md` §17, [SW](design/06-software-verification.md) §10.1 and §10.4 (gates), SW-22 (the
> settings) and [AT](design/03-attention-kv.md) §13.

## Method

This follows openTPU `tools/validate.py` (https://github.com/FeSens/openTPU; its README section on
validation and the script itself were read for this):

1. **Device run.** Greedy-generate G tokens per prompt with all layers and the output head on the
   Tesseract device (`-ngl 99`, `TSR_DEVICE=sim`). The KV cache and attention are on the host in M6
   (`offload_kqv = false`, i.e. `-nkvo`) and on the device in M7 (`--phase m7`: the whole decode
   token is one launch; see the settings below). The prompt is decoded in one batch (prefill), then one
   token per decode. The full logits are recorded at the last `--prompt-tail` prompt positions and
   at every generated position except the last. Row r is the output at token index `row0 + r` and
   predicts token `row0 + r + 1`. With the default `--prompt-tail 1` that gives G rows per prompt,
   one for each generated token, which is exactly openTPU's set (`golden.logits(ids + got[:-1])[len(ids)-1:]`).
2. **CPU run (reference).** Feed the same sequence (prompt + device tokens) to the CPU backend alone
   and record the logits at the same positions, so every position is compared with the same
   context. The default `--replay batch` decodes each sequence in one `llama_decode`, as openTPU
   runs its golden model over the whole sequence. `--replay step` mirrors the device's call pattern
   (prefill + one token per decode).
3. **Floor.** Repeat the CPU run with its input perturbed by one f32 ulp (`--perturb-ulp SEED`).
   `klcheck all` makes N such runs (`--floor-seeds N`, default 3) with the seeds S, S+1, ...,
   S+N−1, where S is `--seed` (default 1).
4. **Compare.** For each position, compute in double precision `KL(P_cpu ‖ P_x) = Σ p_cpu (log p_cpu − log p_x)`
   (nats) for x = device and x = floor. Report mean, median and max KL, top-1 agreement
   (`argmax x == argmax cpu`), the largest |Δlogit|, the smallest cosine similarity and the number
   of bit-identical rows. Each statistic is reported for all positions, for the prompt-tail rows and
   for the generated rows.

**Pass:** `mean KL(cpu ‖ dev) ≤ max(--min-kl, --kl-ratio × F)` and `top-1 ≥ --min-top1`, where F is
the floor: the mean over the floor runs of each run's `mean KL(cpu ‖ floor)`. With one floor run F is
that run's mean KL. The defaults (`3`, `0.8`, `--min-kl 0`) are the spec values. openTPU also
lets any mean KL ≤ 1e-3 nats pass (`MAX_KL`); `--min-kl 1e-3` reproduces that. The check also
fails if a file holds non-finite logits or if the files do not contain the same sequences.

The text output has one line per floor run (mean, median, max, top-1 and the device ratio against
that run alone), and the criterion line gives the ratio against the mean floor and against each
seed.

### Floor seeds

One seed's floor is a random draw: the direction of every nudge comes from the seed, and the mean
KL can be set by a few ill-conditioned positions. On Gemma 4 12B Q4_K_M (M7, below) the same device
run gives a ratio of 2.19 against seed 1 and 1.39, 1.69 and 1.51 against seeds 2, 3 and 4. The
criterion therefore uses the mean of N floor runs, three by default (three CPU replays cost about
0.5 s on 0.5B). `--floor-seeds 1` gives the single floor of the earlier runs, with identical
numbers. Floors use the seeds S to S+N−1, so `--seed 5` alone now makes floors 5, 6 and 7. The
results below use the single seed 1 unless a row says otherwise.

### The 1-ulp perturbation (floor)

- **Where:** the input embeddings. The floor run gives llama.cpp `batch.embd` instead of token ids.
  The rows come straight from the GGUF's `token_embd.weight` (found in any split) and are
  dequantized with `ggml_get_type_traits(type)->to_float`, the same function the CPU's `GET_ROWS`
  uses (F32 rows are copied). For Qwen2 the embedding is not scaled, so without the nudge this path
  reproduces the token path **bit for bit**. `klcheck all` checks this on every run (an extra CPU
  replay with `--input embd`, compared with `--expect-identical`). That check guarantees the floor
  measures the nudge and nothing else.
- **How:** element j of the embedding of token index `pos` in sequence `seq` becomes
  `nextafterf(x, ±inf)`. The sign is the top bit of `splitmix64(key + j)`, where
  `key = splitmix64(splitmix64(splitmix64(seed) ^ seq) ^ (pos << 1))`. Every element moves by
  exactly one ulp (0 becomes ± the smallest subnormal). The pattern does not depend on batching,
  threads or `--resume`. `klcheck embd MODEL TOKEN --perturb-ulp S --seq Q --pos P` prints
  perturbed rows.
- **How this differs from openTPU:** openTPU multiplies the embedding output by
  `1 + 2^-23 · N(0,1)` (seed 1). In f32 that factor is exactly 1 for about 30 % of the elements and
  moves the rest by a Gaussian-distributed one to a few ulp. Here every element moves by exactly
  ±1 ulp. The two are of the same magnitude.

openTPU's golden is fp32 transformers with the device's quantization emulated. Here the oracle is
the ggml CPU backend (the oracle chain of `docs/spec.md` §17). It quantizes activations to Q8_0 / Q8_K like the device's
QACT A32 / A256, so the device and the CPU differ mainly in accumulation order. (Q8_K activations
are bit-identical; for Q8_0 the x86 CPU backend uses its AVX `quantize_row_q8_0`, which differs
from QACT A32 = `quantize_row_q8_0_ref` in about 1 of 3-4e5 values: `id = 127/amax` and
half-even rounding, see `docs/llama-pin.md`.) An int8 value that rounds differently then spreads
through later layers, which is why the floor is not tiny.

### Settings fixed in every phase

From M6 on, every device run and every CPU reference and floor run uses flash
attention on, a q8_0 K and V cache and llama.cpp's Hadamard KV rotation off (D17, SW-22,
[SW](design/06-software-verification.md) §10.4, [AT](design/03-attention-kv.md) §14). The KV cache and
attention stay on the host in M6 (`offload_kqv` = false) and move to the device in M7
(`offload_kqv` = true). The tool has had these settings since P3.3: they are the defaults of every
phase (`--phase m6`; `--phase m7` turns `offload_kqv` on), `klcheck all` passes them to every child,
and each file records them (header version 2, below); `compare` warns when two files differ in them.
`--fa`, `-ctk` / `-ctv`, `--offload-kqv` and `--attn-rot on` exist for experiments only. Some
results below ("FA-off settings") were measured with flash attention off and an f16 KV cache on the
host, and are not comparable with the M6 ones; the table in "Results" names the settings of every row.

| Setting | Value | Why |
| --- | --- | --- |
| Flash attention | **on** (the default; `--fa off` only for experiments), recorded in the file | the pinned llama.cpp refuses a quantized V cache without FA (`src/llama-context.cpp:566-569`), and the device's attention is `FLASH_ATTN_EXT` only (D17); the device, CPU and floor runs use the same FA setting |
| KV cache | **q8_0 for K and V** (`-ctk q8_0 -ctv q8_0`) in every run; on the host in M6 (`offload_kqv=false`), on the device in M7 (`offload_kqv=true`) | D17; fixing the KV type already in M6 keeps the baselines constant from M6 to M7, so the M7 delta measures the device attention only |
| Hadamard KV rotation | **off** (`LLAMA_ATTN_ROT_DISABLE=1`, set by the tool for every phase and recorded in the file; `--attn-rot on` unsets it) | With a quantized cache llama.cpp rotates Q, K and V by an F32 graph-input matrix that the MVM cannot use, which would split attention into CPU islands (SW-16, AT-17); the CPU reference uses the same setting, so the comparison stays fair |
| CPU weight repack (`use_extra_bufts`) | off (`--repack` turns it on) | repacking copies the weights into anonymous memory; 32B Q4_K_M (19 GB) does not fit in 14 GB RAM, while mmap'd weights can be evicted |
| Device selection | `--device sim\|mock\|vkernel\|xrt`: `TSR_DEVICE` = that value, `devices = {Tesseract0}`, `n_gpu_layers = 999`. `--device cpu`: `TSR_DEVICE=none`, `devices = {}` | the device phase checks that a Tesseract device is registered; a CPU phase never opens a transport. `sim` and `mock` run on the simulator; `vkernel` is not available until the two-clock server exists (P5.3); `xrt` needs a build with `-DTSR_WITH_XRT=ON` and an xclbin (the card path is work package W6, [`docs/xrt-host.md`](xrt-host.md)) |
| Device kinds | `--phase m6` sets `TSR_KINDS=m6` (the device takes MUL_MAT only). `--phase m7` passes the caller's `TSR_KINDS` on to the backend: unset means every kind of the booted image, and a list such as `all,-RMS_NORM` is read by the backend ([`sw/ggml-tesseract/README.md`](../sw/ggml-tesseract/README.md)). `tsr-numcheck run` does the same | an M7 run can move single kinds back to the CPU to find the kind behind a difference. The value is not in the `.klc` header (it has no free bytes): `klcheck run` prints a `TSR_KINDS` line on device runs, `report.json` has `tsr_kinds` (below) and `tsr-numcheck` writes it to `meta.json` (`info.TSR_KINDS`) |
| Threads | `-t` (default: online CPUs / 2) | the simulator uses `TSR_SIM_THREADS` (default min(CPUs, 8)) |
| Context | `n_ctx` = longest sequence rounded up to 256, `n_outputs_max` = rows actually needed | keeps the KV cache, compute buffers and the device IO region small |

## Usage

```
klcheck all -m MODEL [-o DIR] [--device sim] [--seed 1] [--floor-seeds 3] [-n 64] [--n-prompts N] [--prompts FILE]
            [--prompt-tail 1] [--chat] [--replay batch|step] [-t N] [--phase m6|m7] [--resume]
            [--kl-ratio 3] [--min-top1 0.8] [--min-kl 0] [--no-embd-check]
            [--fa on|off] [-ctk TYPE] [-ctv TYPE] [--offload-kqv on|off] [--attn-rot on|off]   # experiments
klcheck run -m MODEL --device sim|mock|vkernel|xrt|cpu --out FILE [--tokens FILE] [--perturb-ulp SEED] [...]
klcheck compare --ref CPU.klc --dev DEV.klc [--floor FLOOR.klc]... [--json FILE] [--tsr-kinds LIST] [--expect-identical]
klcheck info FILE [-m MODEL]          # header, timings, per-sequence text
klcheck embd MODEL TOKEN... [--perturb-ulp S --seq Q --pos P]
klcheck selftest [DIR]
```

`klcheck all` runs every phase as its own child process, one after another, so two models are
never loaded at once. Each phase prints its exact command line:

```
klcheck run --device sim --out DIR/dev.klc -m MODEL [-n G ...]                      # generate
klcheck run --device cpu --tokens DIR/dev.klc --out DIR/cpu.klc -m MODEL            # reference
klcheck run --device cpu --tokens DIR/dev.klc --perturb-ulp 1 --out DIR/floor-seed1.klc -m MODEL   # also seeds 2, 3
klcheck run --device cpu --tokens DIR/dev.klc --input embd --out DIR/cpu-embd.klc -m MODEL
klcheck compare --ref DIR/cpu.klc --dev DIR/cpu-embd.klc --expect-identical -q  # embd identity
klcheck compare --ref DIR/cpu.klc --dev DIR/dev.klc --floor DIR/floor-seed1.klc --floor DIR/floor-seed2.klc \
        --floor DIR/floor-seed3.klc --tsr-kinds m6 --json DIR/report.json
```

The floor files are `floor-seed<S>.klc` for N > 1 and `floor.klc` for `--floor-seeds 1`.
`compare --floor` can be given once per floor run; two floor files with the same seed are an
error (exit 2), since one floor would count twice. `--tsr-kinds` only records the device run's
`TSR_KINDS` in the report; `klcheck all` passes it.

The exit code is 0 for PASS, 1 for FAIL and 2 for an error. Without `-o`, the files go to a
private temporary directory that is removed at the end. `report.json` has the summary plus one
entry per position (KL device, KL floor, the three argmaxes, |Δlogit|, cosine). The floor fields:

- `floors`: one entry per run (`file`, `seed`, `n`, `kl_mean`, `kl_median`, `kl_max`, `top1`,
  `bit_identical`, `ratio` = the device's mean KL over that run's); `floor` is still the first file.
- `n_floor`, `floor_kl` (the floor F of the criterion) and `tsr_kinds` (`"m6"`, the caller's list,
  `"unset"`, or `null` when there is no device run to record: `compare` run by hand, `--device cpu`).
- Per position, `kl_floor` is the mean over the runs and `kl_floors` lists each run's value (only
  with more than one run). `summary.floor*` pool the rows of every run, so its `n` is N times the
  number of positions.
- `ratio` is `null` when the floor KL is 0 and the device KL is not (it was written as `inf`, which
  is not valid JSON). With both at 0 it is 0.
- `argmax_floor` of a position is the first run's.

The fields are additions: the `.klc` format and the earlier `report.json` fields are unchanged.

- **Prompts:** 8 built-in prompts (English prose, Python, a question, Korean, arithmetic Q&A, a list,
  a story, JSON; 7–25 tokens with the Qwen2.5 tokenizer), or `--prompts FILE` with one prompt per
  line (`\n`, `\t`, `\\` escapes; blank lines and `#` lines are skipped). `--chat` wraps each prompt in
  the model's chat template. Generation does not stop at end-of-generation tokens, and the count of
  such tokens is recorded.
- **`--replay`:** a CPU replay defaults to `batch` (openTPU's golden run); a device replay
  (`--device sim|mock|vkernel|xrt --tokens FILE`) defaults to `step`, which repeats the generating
  run's prefill and single-token graphs, so that `compare --expect-identical` can require every row
  bit-identical across devices (the M6 gate's item 2, SW §10.4: generate on `xrt`, replay the tokens
  on `sim`).
- **`--resume`:** a phase keeps the complete records of an existing output file and continues after
  them. A finished phase is skipped before its model is loaded. `klcheck all --resume` therefore
  restarts a long 32B run where it stopped. A killed and resumed device run was bit-identical to an
  uninterrupted one (256/256 rows). A directory made with `--floor-seeds 1` has `floor.klc`; with the
  default of three seeds `--resume` runs the seed-1 floor again as `floor-seed1.klc`.

### Logits file (`.klc`)

The file is little-endian. A 512-byte `klc_header` (magic `TSRKLC01`, version 2) is followed by
`n_seq` records. Each record is a 48-byte `klc_seq` (`KSEQ`, index, n_prompt, n_gen, row0,
n_rows, timings, EOG count), then `int32 tokens[n_prompt + n_gen]`, then
`f32 logits[n_rows][n_vocab]`, with `row0 + n_rows = n_prompt + n_gen − 1`. The header records the
model (description, parameter count, size, file), the device, the mode (generate / replay
batch|step), the input path (tokens / embd / embd+1ulp and seed), flash attention, threads, repack,
chat and timings, and since version 2 (P3.3) the settings in five words of the former reserved
area: `type_k`, `type_v` (ggml types), `offload_kqv`, `attn_rot_disable` (LLAMA_ATTN_ROT_DISABLE was 1
for the run) and `phase` (6 / 7). Version 1 files are still read, as f16 / f16 on the host with
rotation not disabled, and `--resume` refuses to continue one with the M6 options. `klcheck info` prints the settings line. `n_seq` counts complete records and is rewritten after each one. A phase streams
one sequence at a time (G = 64: 39 MB per sequence for a 152k vocabulary), and `compare` streams
rows, so neither depends on the model size.

## Results

Every row names its device, its settings and its provenance. "FA-off settings" are flash attention off
and an f16 KV cache on the host; "M6 settings" are flash attention on, a q8_0 K and V cache, rotation
off and `offload_kqv` false (`--phase m6`); "M7 settings" are the same with `offload_kqv` true, so the
KV cache and attention are on the device and a decode token is one launch (`--phase m7`). The limits
are ratio ≤ 3 and top-1 ≥ 80 %; every row passes. Every ratio is against the single floor seed 1; the
runs against several seeds are described in the text.

| Model | Device | Settings | Positions | Mean KL device / floor | Ratio | Top-1 | Provenance |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen2.5-0.5B Q8_0 | sim | FA off | 512 (8 × 64) | 1.095e-3 / 1.136e-3 | 0.964 | 98.05 % | measured on the simulator |
| Qwen2.5-0.5B Q8_0 | xrt, k1 | FA off | 512 | 1.0950e-3 / 1.1357e-3 | 0.964 | 98.05 % | measured on the card (k1, 300 MHz) |
| Qwen2.5-0.5B Q8_0 | xrt, k4 | FA off | 512 | 1.0950e-3 / 1.1357e-3 | 0.964 | 98.05 % | measured on the card (k4, 276.3 MHz) |
| Qwen2.5-0.5B Q8_0 | sim | M6 | 512 (8 × 64) | 2.595e-3 / 2.530e-3 | 1.026 | 97.07 % | measured on the simulator (`83ac601`) |
| Qwen2.5-0.5B Q8_0 | sim | M6 | 128 (4 × 32; ctest `tools_klcheck_qwen05b`) | 2.089e-3 / 2.912e-3 | 0.717 | 96.88 % | measured on the simulator (`83ac601`) |
| Qwen2.5-0.5B Q8_0 | xrt, k4m200 | M6 | 512 (8 × 64) | 2.5947e-3 / 2.5301e-3 | 1.026 | 97.07 % | measured on the card (k4m200, 200 / 100 MHz); 512 of 512 rows bit-identical to `sim` |
| Qwen2.5-0.5B Q8_0 | xrt, k4full | M6 | 512 (8 × 64) | 2.5947e-3 / 2.4101e-3 (mean of 3 floors) | 1.077 (1.026 against seed 1) | 97.07 % | measured on the card (k4full, second and fourth link, 200 / 100 MHz; the same figures to every digit on both); 512 of 512 rows bit-identical to `sim` at `T_MAX` 4 |
| Qwen2.5-32B Q4_K_M | sim | FA off | 256 (4 × 64) | 2.550e-3 / 1.876e-3 | 1.360 | 98.83 % | measured on the simulator (M1) |
| Qwen2.5-32B Q4_K_M | xrt, k4 | FA off | 512 (8 × 64) | 2.4113e-3 / 1.8487e-3 | 1.304 | 99.02 % | measured on the card (M4) |
| Qwen2.5-32B Q4_K_M | sim | M6 | 512 (8 × 64) | 2.886e-3 / 2.246e-3 | 1.285 | 98.44 % | measured on the simulator (`83ac601`) |
| Qwen2.5-32B Q4_K_M | xrt, k4m200 | M6 | 32 (2 × 16) | 1.3362e-3 / 1.7817e-3 (mean of 3 floors) | 0.750 | 100 % | measured on the card (k4m200, 200 / 100 MHz); 32 of 32 rows bit-identical to `sim` |
| Qwen2.5-32B Q4_K_M | xrt, k4full | M6 | 32 (2 × 16) | 1.3362e-3 / 1.7817e-3 (mean of 3 floors) | 0.750 | 100 % | measured on the card (k4full, second and fourth link, 200 / 100 MHz; the same figures to every digit on both); 32 of 32 rows bit-identical to `sim` |
| Qwen2.5-0.5B Q8_0 | sim | M7 | 512 (8 × 64) | 2.22e-3 / 2.43e-3 | 0.914 | 98.05 % | measured on the simulator (M7) |
| Qwen2.5-0.5B Q8_0 | sim | M7 | 34 (2 × 17) | not recorded | 0.765 | not recorded | measured on the simulator (M7) |
| Qwen2.5-32B Q4_K_M | sim | M7 | 512 (8 × 64) | 3.113e-3 / 2.549e-3 | 1.221 | 98.63 % | measured on the simulator (M7) |
| Qwen2.5-0.5B Q8_0 | xrt, k4full | M7, vector-unit kinds on the device, attention and KV on the host (`--offload-kqv off`) | 512 (8 × 64) | 2.6613e-3 / 2.3609e-3 (mean of 3 floors) | 1.127 | 98.63 % | measured on the card (k4full, second and fourth link, 200 / 100 MHz; the same figures to every digit on both); 512 of 512 rows bit-identical to `sim` |
| Qwen2.5-32B Q4_K_M | xrt, k4full | M7, vector-unit kinds on the device, attention and KV on the host | 32 (2 × 16) | 1.3848e-3 / 1.7817e-3 (mean of 3 floors) | 0.777 | 100 % | measured on the card (k4full, second and fourth link, 200 / 100 MHz; the same figures to every digit on both); 32 of 32 rows bit-identical to `sim` |
| Qwen2.5-0.5B Q8_0 | xrt, k4full | M7 (attention and KV on the device) | 512 (8 × 64) | 2.2216e-3 / 2.3307e-3 (mean of 3 floors) | 0.953 (0.914 against seed 1) | 98.05 % | measured on the card (k4full, fourth link, 200 / 100 MHz, one launch per token); 512 of 512 rows bit-identical to `sim` at `T_MAX` 4. The second link's image, with the ATTPV fault of `platform-u200.md` §8.8, gave 2.4596 / 4.686e-3, ratio 525, top-1 34.18 %, 0 of 512 rows equal |
| Qwen2.5-32B Q4_K_M | xrt, k4full | M7 (attention and KV on the device) | 32 (2 × 16) | 1.3104e-3 / 1.7817e-3 (mean of 3 floors) | 0.735 | 100 % | measured on the card (k4full, fourth link, 200 / 100 MHz); 32 of 32 rows bit-identical to `sim`. The second link's image gave 4.609 / 9.996e-3, ratio 461, top-1 21.88 %, 0 of 32 rows equal |
| Gemma 4 12B Q4_K_M | sim | M6 | 512 (8 × 64) | 9.504e-2 / 7.023e-2 | 1.353 | 96.48 % | measured on the simulator |
| Gemma 4 12B Q4_K_M | sim | M7 | 512 (8 × 64) | 1.281e-1 / 5.849e-2 | 2.190 | 95.12 % | measured on the simulator (M7) |

On the card the 0.5B run gives the simulator's mean KL to seven digits (1.095007e-3), and the small
identity check (`klcheck compare --expect-identical`, sim against xrt, 8 of 8 rows bit-identical on k1
and on k4, [`docs/platform-u200.md`](platform-u200.md) §14) passed. The three 32B ratios use different settings (and
the M1 run had 4 prompts) and are not directly comparable. The 32B ratio sits above the 0.5B one in
every setting; `docs/numcheck.md` shows that part of the device-to-CPU distance on 32B is the CPU's
own error on the Q6_K tensors that the device converts to Q8_0, which a KL against the CPU cannot
separate from device error.

### Qwen2.5-0.5B-Instruct Q8_0 with the FA-off settings

Measurement conditions: `models/qwen2.5-0.5b-instruct-q8_0.gguf`, build `build/dev` (Release,
GGML_NATIVE with AVX512, llama.cpp pin `e3797b18` + patch 0001), `TSR_DEVICE=sim` with default
T_MAX = 1, 8 simulator threads, `-t 8`, FA off, f16 KV on the host, CPU without repack. Command:
`klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o DIR -n 64`: 8 built-in prompts (7, 13,
11, 12, 25, 12, 7, 11 tokens), G = 64, prompt tail 1, giving **512 positions**.

| | mean KL (nats) | median | max | top-1 vs CPU | bit-identical rows |
| --- | --- | --- | --- | --- | --- |
| device (sim) vs CPU | **1.095e-3** | 6.17e-4 | 1.27e-2 | **98.05 %** (502/512) | 0/512 |
| floor (CPU + 1 ulp, seed 1) vs CPU | **1.136e-3** | 5.06e-4 | 5.58e-2 | 98.63 % | 7/512 |

These are the numbers with the output head on the device (re-measured after the supports_op fix, see
[`docs/llama-pin.md`](llama-pin.md) "How llama.cpp places weights"). The first run of this table had,
unnoticed, the head on the CPU: Tesseract's supports_op rejected llama.cpp's 512-token placement
probe for `output.weight`. Moving the 136 M-MAC head (28 % of the per-token MACs) to the device
changed the logits only at the accumulation-order level, so the table above and the paragraph below
agree to the digits shown (mean KL 1.0950071e-3 before, 1.0950067e-3 after). The supporting
measurements and timings marked "(head on CPU)" are from that first run.

**Ratio 0.964 (limit 3), top-1 98.05 % (limit 80 %): PASS.** The largest |Δlogit| is 1.24 for the
device and 1.53 for the floor. The smallest cosine similarity of a logits row is 0.99846 for the
device and 0.99742 for the floor. Per prompt, the device/floor ratio ranges from 0.49 to 2.02 and the
device's top-1 agreement from 95.3 % to 100 %.

Supporting measurements on the same 512 positions (head on CPU):

| Check | Result |
| --- | --- |
| Floor seeds 1, 2, 3, 4 | 1.136e-3, 0.936e-3, 1.248e-3, 1.066e-3 (top-1 98.6, 99.4, 97.7, 98.4 %), so the device ratio is 0.96 / 1.17 / 0.88 / 1.03 |
| CPU `--replay step` vs CPU `--replay batch` (the CPU's own accumulation-order change) | KL 9.21e-4, top-1 98.44 %, i.e. 0.81 × floor |
| Device vs CPU `--replay step` | KL 1.144e-3, top-1 98.44 % |
| `--prompt-tail 8` (566 positions) | all 1.114e-3 vs floor 1.086e-3 (ratio 1.03, top-1 98.06 %): PASS. The 54 prompt rows alone: device 1.30e-3 vs floor 6.1e-4 (ratio 2.1, top-1 98.1 %). The 1-ulp floor is smaller early in a sequence (26/54 of those rows are bit-identical), while the device's rounding differences are present from the first token |
| `TSR_SIM_TMAX=8` device run vs T_MAX = 1 | bit-identical, 566/566 rows (prefill MVMs with T > 1 change nothing) |
| Device run repeated; CPU replay repeated; `--input embd` vs tokens | all bit-identical |
| ctest `tools_klcheck_qwen05b` (4 prompts × 32 tokens, FA-off settings at the time) | 1.168e-3 vs floor 9.94e-4, ratio 1.18, top-1 99.22 %: PASS (re-run with the head on the device: 1.1683e-3, ratio 1.176, top-1 99.22 %) |

In short, the device sits at the floor (ratio ≈ 1). Its distance from the CPU is the same size as
the CPU's distance from itself under a 1-ulp input change, or under its own batch-vs-step
accumulation-order change.

Timings (0.5B, simulator; other jobs shared the machine during part of the runs):

| Phase | Time |
| --- | --- |
| Device decode | 84 ms/token with the head on the device (68–73 ms/token with the head on CPU; up to 170 ms/token under load) |
| Device prefill (T_MAX = 1) | ~53–60 ms per prompt token |
| Model load | 0.17 s |
| Device phase, 8 × 64 | 48.4 s (head on CPU: 41.0 s uncontended, 61.7 s under load) |
| CPU replay | ~0.12 s per sequence (batch, ~75 tokens); 14–15 ms per single-token decode |
| Whole `klcheck all` (device, CPU, floor, embd check, compare) | 53 s, peak RSS 1.24 GB (head on CPU: 46 s, 1.1 GB) |

Each logits file is 311 MB (8 × 64 rows × 151936 × 4 B).

On the card the same command with `--device xrt` took 20.6 s of device decoding on k1 (40 ms per
token) and 9.5 s on k4 (18 ms per token), `build/card/klc_all` and `build/card/k4/klc_all`.

### Qwen2.5-0.5B-Instruct Q8_0 with the M6 settings (on the simulator)

Measured on the simulator (`83ac601`): the backend's MUL_MATs as op lists
of the `graph` kernel on the simulator kind (`TSR_DEVICE=sim`: the RISC-V interpreter and the queue
model behind the core CSR port; T_MAX 1, 4 simulator threads), attention and KV on the host, the M6
settings. Command: the ctest `tools_klcheck_qwen05b`, `klcheck all -m
models/qwen2.5-0.5b-instruct-q8_0.gguf --device sim --phase m6 --n-prompts 4 -n 32` (128 generated
positions):

| | mean KL (nats) | median | max | top-1 vs CPU | bit-identical rows |
| --- | --- | --- | --- | --- | --- |
| device (sim) vs CPU | **2.089e-3** | 1.19e-3 | 2.82e-2 | **96.88 %** | 0/128 |
| floor (CPU + 1 ulp, seed 1) vs CPU | **2.912e-3** | 1.26e-3 | 1.08e-1 | 98.44 % | 7/128 |

**Ratio 0.717 (limit 3), top-1 96.88 % (limit 80 %): PASS.** This is against floor seed 1. The ctest
now makes three floors: against their mean the ratio is 0.899 (0.717 / 1.055 / 1.005 per seed).
Max |Δlogit| 0.84 (floor 1.36), min
cosine 0.99741. Device decode 140–147 ms/token on the simulator; the whole `klcheck all` 25 s wall
(peak RSS 1.26 GB). The device run's tokens replayed with `klcheck run --device sim --tokens` and
`--device mock --tokens` (step replay) gave 128/128 rows bit-identical to the generating run with
`compare --expect-identical`: the M6 gate's identity check (item 2) works end to end on the simulator
kinds, and `sim` and `mock` (the adapter model) agree bit for bit (the `tools_klcheck_identity_qwen05b`
ctest repeats it on 2 prompts × 8 tokens).

**Full prompt set** (`build/dev`, `TSR_DEVICE=sim`, T_MAX 1, 8 simulator threads, fixed schedule, the
M6 settings, all 8 built-in prompts × 64 tokens = 512 positions), measured on the simulator
(`83ac601`):

| Check | Result |
| --- | --- |
| generate on sim, replay the tokens on sim (`--tokens`, step), `compare --expect-identical` | **512/512 rows bit-identical** |
| KL vs CPU (CPU, floor and embd runs on the device tokens) | mean KL 2.595e-3 vs floor 2.530e-3, **ratio 1.026**, top-1 97.07 %: PASS; embd path bit-identical |

Decode 90–95 ms/token on the simulator (gen 47.1 s for 512 tokens, prefill 6.0 s); 145.00 launches per token (one per split), 1.17 MUL_MAT records
per launch, 0 staged chunks (`llama-completion` with its default host output buffer: 1 per token);
OPL cache 73,385 hits / 855 misses (98.9 %; every decode pass after the first hits).

### Qwen2.5-32B-Instruct Q4_K_M

Command for the M6 settings (the first split is enough: llama.cpp loads the others, and klcheck finds
`token_embd.weight` in whichever split holds it):

```sh
mkdir -p build/simdram
TSR_SIM_DRAM_DIR=build/simdram build/dev/bin/klcheck all \
    -m models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf \
    -o build/klcheck-32b-sim-m6 --device sim --phase m6 -n 64 --resume
```

The M1 and M4 rows were made with flash attention off and an f16 KV cache (4 prompts for
M1); `klcheck` sets those with `--fa off -ctk f16 -ctv f16` (experiments only), and
`--n-prompts N` limits the prompt set. Rerun the same command to resume after an interruption.
`TSR_SIM_DRAM_DIR` matters only to the device phase (the CPU phases open no transport). It must
exist and have ~20 GiB free: the weights take 19.3 GiB in simulator DRAM (13.9 GiB Q4_K and
5.4 GiB Q8_0 converted from Q6_K), which does not fit in 14 GB of RAM, so the DRAM files are sparse,
unlinked and gone at exit, and every token re-reads them through the page cache. The logits files
take 39 MB per prompt each (311 MB for 8 prompts).

| Run | Positions | Device phase | Whole `klcheck all` | Provenance |
| --- | ---: | --- | --- | --- |
| M1, sim, FA-off settings, 4 prompts | 256 | 2,535 s wall; decode ~9.0 s/token (9,044–10,074 ms) | 44 min 23 s | measured on the simulator (M1) |
| M6, sim, M6 settings, 8 prompts | 512 | 4,434 s wall: prefill 232.0 s (98 prompt tokens), decode 4,170.0 s for 512 tokens = 8.1 s/token (7.9–9.0 s per sequence) | 1 h 17 min 06 s, peak RSS 13.0 GiB (mostly file pages) | measured on the simulator (`83ac601`) |
| M4, card, FA-off settings, 8 prompts | 512 | 196.9 s wall: model load 17.6 s, decode 342 ms/token | about 6 min | measured on the card (M4) |

The M6 run (report `build/klcheck-32b-sim-m6/report.json`, not in git):

| | mean KL (nats) | median | max | top-1 vs CPU | bit-identical rows |
| --- | --- | --- | --- | --- | --- |
| device (sim) vs CPU | **2.886e-3** | 1.79e-4 | 2.15e-1 | **98.44 %** | 0/512 |
| floor (CPU + 1 ulp, seed 1) vs CPU | **2.246e-3** | 1.93e-4 | 1.23e-1 | 99.22 % | 0/512 |

**Ratio 1.285 (limit 3), top-1 98.44 % (limit 80 %): PASS.** Per sequence the device/floor ratio runs
from 0.68 to 1.96; the largest KL is at sequence 0, row 44, for the device and the floor alike. Max
|Δlogit| 3.15 (floor 5.40), min cosine 0.9654. The embd input path equals the token path bit for bit
on this model too. The M1 run (4 prompts, FA-off settings) gave mean KL 2.5504e-3 against a floor of
1.8758e-3 (ratio 1.360, top-1 98.83 %), and the M4 run on the card (8 prompts, FA-off settings) 2.4113e-3
against 1.8487e-3 (ratio 1.304, top-1 99.02 %).

`TSR_SIM_TMAX=8` cuts the prefill passes of the simulator (from 43 to 7 for 4 prompts, about 12 % of
a run) with bit-identical results (566/566 rows on 0.5B). The default `T_MAX` of the simulated core
is 1 (the k1 value; the k4 bitstream has 8), and the gate runs keep it.

### M7 settings (on the simulator)

The M7 backend runs the whole decode token as one launch with the KV cache on the device
(`TSR_DEVICE=sim`, T_MAX 1, `TSR_KINDS` unset: the device takes every kind of the booted image).
Command, as used for the 8 built-in prompts × 64 tokens of the rows above:

```sh
build/dev/bin/klcheck all -m MODEL --device sim --phase m7 -n 64 -o DIR
```

The 32B run takes the first split and `TSR_SIM_DRAM_DIR` as in the M6 command above. Gemma 4 uses the
sliding-window ring cache of llama-cli (`--swa-full off`, the default). The rows of the table were
made with the single floor seed 1, before the default became three seeds.

- Qwen2.5: 32B ratio 1.221, top-1 98.63 % (M6: 1.285 and 98.44 %). 0.5B: 0.914 on 8 × 64 and
  0.765 on 2 × 17. A run with three floors (0.5B, 2 prompts × 16 tokens, seeds 1 to 3) gave mean KL
  4.13e-3 against a mean floor of 4.03e-3: ratio 1.026 against the mean, 0.813 / 1.321 / 1.067 per
  seed, top-1 100 %, embd path bit-identical.
- Gemma 4 12B Q4_K_M: M7 ratio 2.190 against seed 1, 1.39 / 1.69 / 1.51 against seeds 2 / 3 / 4,
  1.65 against the mean of seeds 1 to 4, top-1 95.12 %. M6: 1.353 against seed 1, 1.06 against the
  mean, top-1 96.48 %. All pass (limit 3), but the ratio is higher than for Qwen2.5 and depends on the
  seed. The excess sits in one prompt. In the M7 run sequence 7 has mean KL 0.72 against a seed-1
  floor of 0.133, and over seeds 1 to 4 that floor ranges from 0.133 to 0.423: these are
  ill-conditioned positions where the CPU floor is itself unstable. The device's RMS_NORM is
  bit-exact to its rule and differs from ggml-cpu's double sum at rounding level
  ([VU](design/02-vector-unit.md) §6.9).

## Tests

- `tools_klcheck_selftest` (no model): the 1-ulp nudge (exactly one ulp, deterministic, balanced
  directions); writing, reading and resuming files, including a torn record; KL against an
  independent long-double evaluation (1e-9 relative); the verdict as the ratio limit crosses the
  measured value; incomplete files fail.
- `tools_klcheck_selftest` also checks the version 2 header: the settings survive a write / read, a
  version 1 file reads as f16 / f16 on the host, version 3 is refused.
- `tools_klcheck_selftest` also covers several floor runs: three floors with different noise, each
  run's numbers equal a `compare` against that run alone, the floor is the mean, the verdict follows
  the mean, `--floor` works more than once on the command line, the `report.json` fields for one and
  three floors, the floor file names, and a floor of KL 0 (the ratio is written as `null`).
- `tools_klcheck_qwen05b` (registered only if `models/qwen2.5-0.5b-instruct-q8_0.gguf` exists,
  label `model`, ~25 s): `klcheck all --device sim --phase m6 --n-prompts 4 -n 32` with the default
  criteria (three floor runs; a repeated floor seed would exit 2) and the embd identity check.
- `tools_klcheck_kinds_qwen05b` (same condition, label `model`, ~3 s): `test_tsr_kinds.sh` runs
  `klcheck run` with `TSR_DEBUG=1`: `--phase m7` with `TSR_KINDS=all,-RMS_NORM` must show that list in the
  backend and in klcheck's `TSR_KINDS` line, `--phase m6` with `TSR_KINDS=all` must show `m6`, and
  `--phase m7` without the variable must show `unset`.
- `tools_klcheck_identity_qwen05b` (same condition, label `model`, ~10 s): generate 2 prompts × 8
  tokens on `sim`, replay the tokens on `mock` (step), `compare --expect-identical`.

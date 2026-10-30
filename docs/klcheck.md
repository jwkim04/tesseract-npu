# klcheck: model-level KL check

`sw/tools/klcheck.cpp` implements the whole-model row of the oracle chain in `docs/spec.md` §17
(Verification strategy) and its *Model-level KL criterion*: the Tesseract device is compared with
the ggml CPU backend on whole-model logits, against a floor given by the CPU compared with itself
under a 1-ulp input perturbation. Pass (the specification's values): **mean KL(CPU ‖ device) ≤ 3 ×
floor and top-1 agreement ≥ 80 %**. The check is part of the M1 and M4 gates (done) and of the M6 gate
(32B on the simulator done; on the card it follows the v2 bitstream).

> **Scope.** User documentation and measurement record of `klcheck`. Status words: *on the
> simulator* means a run through the v2 software stack with `TSR_DEVICE=sim` or `mock`; *on the card
> (v1)* means a run on the Alveo U200 with the v1 core (tag `v1-m4`). Every result names its
> settings and provenance, and results with different settings are not comparable. Normative:
> `docs/spec.md` §17, [SW](v2/06-software-verification.md) §10.1 and §10.4 (gates), V2-SW-22 (the
> settings) and [AT](v2/03-attention-kv.md) §13.

## Method

This follows openTPU `tools/validate.py` (https://github.com/FeSens/openTPU; its README section on
validation and the script itself were read for this):

1. **Device run.** Greedy-generate G tokens per prompt with all layers and the output head on the
   Tesseract device (`-ngl 99`, `TSR_DEVICE=sim`), KV cache and attention on the host
   (`offload_kqv = false`, i.e. `-nkvo`; v1 and M6. From M7 the KV cache and attention are on the
   device, see the settings below). The prompt is decoded in one batch (prefill), then one
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
4. **Compare.** For each position, compute in double precision `KL(P_cpu ‖ P_x) = Σ p_cpu (log p_cpu − log p_x)`
   (nats) for x = device and x = floor. Report mean, median and max KL, top-1 agreement
   (`argmax x == argmax cpu`), the largest |Δlogit|, the smallest cosine similarity and the number
   of bit-identical rows. Each statistic is reported for all positions, for the prompt-tail rows and
   for the generated rows.

**Pass:** `mean KL(cpu ‖ dev) ≤ max(--min-kl, --kl-ratio × mean KL(cpu ‖ floor))` and
`top-1 ≥ --min-top1`. The defaults (`3`, `0.8`, `--min-kl 0`) are the spec values. openTPU also
lets any mean KL ≤ 1e-3 nats pass (`MAX_KL`); `--min-kl 1e-3` reproduces that. The check also
fails if a file holds non-finite logits or if the files do not contain the same sequences.

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

**v2 settings.** From M6 on, every device run and every CPU reference and floor run uses flash
attention on, a q8_0 K and V cache and llama.cpp's Hadamard KV rotation off (D17, V2-SW-22,
[SW](v2/06-software-verification.md) §10.4, [AT](v2/03-attention-kv.md) §14). The KV cache and
attention stay on the host in M6 (`offload_kqv` = false) and move to the device in M7
(`offload_kqv` = true). The tool has had these settings since P3.3: they are the defaults of every
phase (`--phase m6`; `--phase m7` turns `offload_kqv` on), `klcheck all` passes them to every child,
and each file records them (header version 2, below); `compare` warns when two files differ in them.
`--fa`, `-ctk` / `-ctv`, `--offload-kqv` and `--attn-rot on` exist for experiments only. The v1
results below ("v1 settings") were measured with flash attention off and an f16 KV cache on the host,
and are not comparable with the M6 ones; the table in "Results" names the settings of every row.

| Setting | Value | Why |
| --- | --- | --- |
| Flash attention | **v2: on** (the default; `--fa off` only for experiments), recorded in the file. v1: off (`--fa off`) | v2: the pinned llama.cpp refuses a quantized V cache without FA (`src/llama-context.cpp:566-569`), and the device's attention is `FLASH_ATTN_EXT` only (D17); the device, CPU and floor runs use the same FA setting. v1: `auto` resolves to *off* on the device run (layer on Tesseract0, FA on CPU) and *on* on a CPU-only run, which would compare two attention kernels |
| KV cache | **v2: q8_0 for K and V** (`-ctk q8_0 -ctv q8_0`) in every run; on the host in M6 (`offload_kqv=false`), on the device in M7 (`offload_kqv=true`). v1: f16, on the host (`offload_kqv=false`) | v2: D17; fixing the KV type already in M6 keeps the baselines constant from M6 to M7, so the M7 delta measures the device attention only. v1: v1 constraint (`docs/llama-pin.md`) |
| Hadamard KV rotation | **v2: off** (`LLAMA_ATTN_ROT_DISABLE=1`, set by the tool for every phase and recorded in the file; `--attn-rot on` unsets it). v1: not applicable (f16 cache) | With a quantized cache llama.cpp rotates Q, K and V by an F32 graph-input matrix that the MVM cannot use, which would split attention into CPU islands (V2-SW-16, V2-AT-17); the CPU reference uses the same setting, so the comparison stays fair |
| CPU weight repack (`use_extra_bufts`) | off (`--repack` turns it on) | repacking copies the weights into anonymous memory; 32B Q4_K_M (19 GB) does not fit in 14 GB RAM, while mmap'd weights can be evicted |
| Device selection | `--device sim\|mock\|vkernel\|xrt`: `TSR_DEVICE` = that value, `devices = {Tesseract0}`, `n_gpu_layers = 999`. `--device cpu`: `TSR_DEVICE=none`, `devices = {}` | the device phase checks that a Tesseract device is registered; a CPU phase never opens a transport. `sim` and `mock` run the v2 stack on the simulator; `vkernel` is not available until the v2 server exists (P5.3); `xrt` needs a build with `-DTSR_WITH_XRT=ON` and an xclbin (the v1 `xrt` transport ran this check on the card; the v2 card path is work package W6, [`docs/xrt-host.md`](xrt-host.md)) |
| Threads | `-t` (default: online CPUs / 2) | the simulator uses `TSR_SIM_THREADS` (default min(CPUs, 8)) |
| Context | `n_ctx` = longest sequence rounded up to 256, `n_outputs_max` = rows actually needed | keeps the KV cache, compute buffers and the device IO region small |

## Usage

```
klcheck all -m MODEL [-o DIR] [--device sim] [--seed 1] [-n 64] [--n-prompts N] [--prompts FILE]
            [--prompt-tail 1] [--chat] [--replay batch|step] [-t N] [--phase m6|m7] [--resume]
            [--kl-ratio 3] [--min-top1 0.8] [--min-kl 0] [--no-embd-check]
            [--fa on|off] [-ctk TYPE] [-ctv TYPE] [--offload-kqv on|off] [--attn-rot on|off]   # experiments
klcheck run -m MODEL --device sim|mock|vkernel|xrt|cpu --out FILE [--tokens FILE] [--perturb-ulp SEED] [...]
klcheck compare --ref CPU.klc --dev DEV.klc [--floor FLOOR.klc] [--json FILE] [--expect-identical]
klcheck info FILE [-m MODEL]          # header, timings, per-sequence text
klcheck embd MODEL TOKEN... [--perturb-ulp S --seq Q --pos P]
klcheck selftest [DIR]
```

`klcheck all` runs every phase as its own child process, one after another, so two models are
never loaded at once. Each phase prints its exact command line:

```
klcheck run --device sim --out DIR/dev.klc -m MODEL [-n G ...]                      # generate
klcheck run --device cpu --tokens DIR/dev.klc --out DIR/cpu.klc -m MODEL            # reference
klcheck run --device cpu --tokens DIR/dev.klc --perturb-ulp 1 --out DIR/floor.klc -m MODEL
klcheck run --device cpu --tokens DIR/dev.klc --input embd --out DIR/cpu-embd.klc -m MODEL
klcheck compare --ref DIR/cpu.klc --dev DIR/cpu-embd.klc --expect-identical -q  # embd identity
klcheck compare --ref DIR/cpu.klc --dev DIR/dev.klc --floor DIR/floor.klc --json DIR/report.json
```

The exit code is 0 for PASS, 1 for FAIL and 2 for an error. Without `-o`, the files go to a
private temporary directory that is removed at the end. `report.json` has the summary plus one
entry per position (KL device, KL floor, the three argmaxes, |Δlogit|, cosine).

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
  uninterrupted one (256/256 rows).

### Logits file (`.klc`)

The file is little-endian. A 512-byte `klc_header` (magic `TSRKLC01`, version 2) is followed by
`n_seq` records. Each record is a 48-byte `klc_seq` (`KSEQ`, index, n_prompt, n_gen, row0,
n_rows, timings, EOG count), then `int32 tokens[n_prompt + n_gen]`, then
`f32 logits[n_rows][n_vocab]`, with `row0 + n_rows = n_prompt + n_gen − 1`. The header records the
model (description, parameter count, size, file), the device, the mode (generate / replay
batch|step), the input path (tokens / embd / embd+1ulp and seed), flash attention, threads, repack,
chat and timings, and since version 2 (P3.3) the v2 settings in five words of the former reserved
area: `type_k`, `type_v` (ggml types), `offload_kqv`, `attn_rot_disable` (LLAMA_ATTN_ROT_DISABLE was 1
for the run) and `phase` (6 / 7). Version 1 files are still read; they show as the v1 settings
(phase "v1", f16 / f16 on the host, rotation not disabled), and `--resume` refuses to continue one with
v2 options. `klcheck info` prints the settings line. `n_seq` counts complete records and is rewritten after each one. A phase streams
one sequence at a time (G = 64: 39 MB per sequence for a 152k vocabulary), and `compare` streams
rows, so neither depends on the model size.

## Results

Every row names its device, its settings and its provenance. "v1 settings" are flash attention off
and an f16 KV cache on the host; "M6 settings" are flash attention on, a q8_0 K and V cache, rotation
off and `offload_kqv` false (`--phase m6`). The limits are ratio ≤ 3 and top-1 ≥ 80 %; every row
passes.

| Model | Device | Settings | Positions | Mean KL device / floor | Ratio | Top-1 | Provenance |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen2.5-0.5B Q8_0 | sim | v1 | 512 (8 × 64) | 1.095e-3 / 1.136e-3 | 0.964 | 98.05 % | measured on the simulator (v1) |
| Qwen2.5-0.5B Q8_0 | xrt, k1 | v1 | 512 | 1.0950e-3 / 1.1357e-3 | 0.964 | 98.05 % | measured on the card (v1, k1, 300 MHz) |
| Qwen2.5-0.5B Q8_0 | xrt, k4 | v1 | 512 | 1.0950e-3 / 1.1357e-3 | 0.964 | 98.05 % | measured on the card (v1, k4, 276.3 MHz) |
| Qwen2.5-0.5B Q8_0 | sim | M6 | 512 (8 × 64) | 2.595e-3 / 2.530e-3 | 1.026 | 97.07 % | measured on the simulator (v2, `83ac601`) |
| Qwen2.5-0.5B Q8_0 | sim | M6 | 128 (4 × 32; ctest `tools_klcheck_qwen05b`) | 2.089e-3 / 2.912e-3 | 0.717 | 96.88 % | measured on the simulator (v2, `83ac601`) |
| Qwen2.5-32B Q4_K_M | sim | v1 | 256 (4 × 64) | 2.550e-3 / 1.876e-3 | 1.360 | 98.83 % | measured on the simulator (v1, M1) |
| Qwen2.5-32B Q4_K_M | xrt, k4 | v1 | 512 (8 × 64) | 2.4113e-3 / 1.8487e-3 | 1.304 | 99.02 % | measured on the card (v1, M4) |
| Qwen2.5-32B Q4_K_M | sim | M6 | 512 (8 × 64) | 2.886e-3 / 2.246e-3 | 1.285 | 98.44 % | measured on the simulator (v2, `83ac601`) |

On the card the 0.5B run gives the simulator's mean KL to seven digits (1.095007e-3), and the small
identity check (`klcheck compare --expect-identical`, sim against xrt, 8 of 8 rows bit-identical on k1
and on k4, [`docs/platform-u200.md`](platform-u200.md) §14) passed. The three 32B ratios use different settings (and
the M1 run had 4 prompts) and are not directly comparable. The 32B ratio sits above the 0.5B one in
every setting; `docs/numcheck.md` shows that part of the device-to-CPU distance on 32B is the CPU's
own error on the Q6_K tensors that the device converts to Q8_0, which a KL against the CPU cannot
separate from device error.

### Qwen2.5-0.5B-Instruct Q8_0 with the v1 settings

Measurement conditions: `models/qwen2.5-0.5b-instruct-q8_0.gguf`, build `build/dev` (Release,
GGML_NATIVE with AVX512, llama.cpp pin `e3797b18` + patch 0001), `TSR_DEVICE=sim` with default
T_MAX = 1, 8 simulator threads, `-t 8`, FA off, f16 KV on the host, CPU without repack. Command:
`klcheck all -m models/qwen2.5-0.5b-instruct-q8_0.gguf -o DIR -n 64`: 8 built-in prompts (7, 13,
11, 12, 25, 12, 7, 11 tokens), G = 64, prompt tail 1, giving **512 positions**. The run was made with the v1 stack
(before tag `v1-m4`).

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
| ctest `tools_klcheck_qwen05b` (4 prompts × 32 tokens, v1 settings at the time) | 1.168e-3 vs floor 9.94e-4, ratio 1.18, top-1 99.22 %: PASS (re-run with the head on the device: 1.1683e-3, ratio 1.176, top-1 99.22 %) |

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

On the card (v1) the same command with `--device xrt` took 20.6 s of device decoding on k1 (40 ms per
token) and 9.5 s on k4 (18 ms per token), `build/card/klc_all` and `build/card/k4/klc_all`.

### Qwen2.5-0.5B-Instruct Q8_0 with the M6 settings (v2 on the simulator)

Measured on the simulator (v2, `83ac601`) on the v2 software stack: the backend's MUL_MATs as op lists
of the `graph` kernel on the simulator kind (`TSR_DEVICE=sim`: the RISC-V interpreter and the queue
model behind the core CSR port; T_MAX 1, 4 simulator threads), attention and KV on the host, the M6
settings. Command: the ctest `tools_klcheck_qwen05b`, `klcheck all -m
models/qwen2.5-0.5b-instruct-q8_0.gguf --device sim --phase m6 --n-prompts 4 -n 32` (128 generated
positions):

| | mean KL (nats) | median | max | top-1 vs CPU | bit-identical rows |
| --- | --- | --- | --- | --- | --- |
| device (sim) vs CPU | **2.089e-3** | 1.19e-3 | 2.82e-2 | **96.88 %** | 0/128 |
| floor (CPU + 1 ulp, seed 1) vs CPU | **2.912e-3** | 1.26e-3 | 1.08e-1 | 98.44 % | 7/128 |

**Ratio 0.717 (limit 3), top-1 96.88 % (limit 80 %): PASS.** Max |Δlogit| 0.84 (floor 1.36), min
cosine 0.99741. Device decode 140–147 ms/token on the simulator; the whole `klcheck all` 25 s wall
(peak RSS 1.26 GB). The device run's tokens replayed with `klcheck run --device sim --tokens` and
`--device mock --tokens` (step replay) gave 128/128 rows bit-identical to the generating run with
`compare --expect-identical`: the M6 gate's identity check (item 2) works end to end on the simulator
kinds, and `sim` and `mock` (the adapter model) agree bit for bit (the `tools_klcheck_identity_qwen05b`
ctest repeats it on 2 prompts × 8 tokens).

**Full prompt set** (`build/dev`, `TSR_DEVICE=sim`, T_MAX 1, 8 simulator threads, fixed schedule, the
M6 settings, all 8 built-in prompts × 64 tokens = 512 positions), measured on the simulator (v2,
`83ac601`):

| Check | Result |
| --- | --- |
| generate on sim, replay the tokens on sim (`--tokens`, step), `compare --expect-identical` | **512/512 rows bit-identical** |
| v1 vs v2, same M6 settings: this tree's `klcheck` binary on `build/v1/bin`'s libraries (v1 backend, transport and simulator; `LD_LIBRARY_PATH=build/v1/bin`) against the v2 run | **512/512 bit-identical**, same generated tokens |
| v1's own `build/v1/bin/klcheck` (`--fa on`; it has no `-ctk` / `-ctv`, so its KV cache is f16) against v2 with `--fa on -ctk f16 -ctv f16` | **512/512 bit-identical**, same generated tokens |
| KL vs CPU (CPU, floor and embd runs on the v2 device tokens) | mean KL 2.595e-3 vs floor 2.530e-3, **ratio 1.026**, top-1 97.07 %: PASS; embd path bit-identical |

Decode 90–95 ms/token on the simulator (gen 47.1 s for 512 tokens, prefill 6.0 s; v1 on the same
host: 86–88 ms/token); 145.00 launches per token (one per split, v1's 145 execs), 1.17 MUL_MAT records
per launch, 0 staged chunks (`llama-completion` with its default host output buffer: 1 per token);
OPL cache 73,385 hits / 855 misses (98.9 %; every decode pass after the first hits). The v2 stack
uses 17 % more device cycles than v1 in M6 (a launch per split; the MVM cycles are identical).

### Qwen2.5-32B-Instruct Q4_K_M

Command for the M6 settings (the first split is enough: llama.cpp loads the others, and klcheck finds
`token_embd.weight` in whichever split holds it):

```sh
mkdir -p build/simdram
TSR_SIM_DRAM_DIR=build/simdram build/dev/bin/klcheck all \
    -m models/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf \
    -o build/klcheck-32b-sim-m6 --device sim --phase m6 -n 64 --resume
```

The M1 and M4 rows were made with the v1 tools (flash attention off, f16 KV cache, 4 prompts for
M1); this tree's `klcheck` can set those with `--fa off -ctk f16 -ctv f16` (experiments only), and
`--n-prompts N` limits the prompt set. Rerun the same command to resume after an interruption.
`TSR_SIM_DRAM_DIR` matters only to the device phase (the CPU phases open no transport). It must
exist and have ~20 GiB free: the weights take 19.3 GiB in simulator DRAM (13.9 GiB Q4_K and
5.4 GiB Q8_0 converted from Q6_K), which does not fit in 14 GB of RAM, so the DRAM files are sparse,
unlinked and gone at exit, and every token re-reads them through the page cache. The logits files
take 39 MB per prompt each (311 MB for 8 prompts).

| Run | Positions | Device phase | Whole `klcheck all` | Provenance |
| --- | ---: | --- | --- | --- |
| M1, sim, v1 settings, 4 prompts | 256 | 2,535 s wall; decode ~9.0 s/token (9,044–10,074 ms) | 44 min 23 s | measured on the simulator (v1, M1) |
| M6, sim, M6 settings, 8 prompts | 512 | 4,434 s wall: prefill 232.0 s (98 prompt tokens), decode 4,170.0 s for 512 tokens = 8.1 s/token (7.9–9.0 s per sequence) | 1 h 17 min 06 s, peak RSS 13.0 GiB (mostly file pages) | measured on the simulator (v2, `83ac601`) |
| M4, card, v1 settings, 8 prompts | 512 | 196.9 s wall: model load 17.6 s, decode 342 ms/token | about 6 min | measured on the card (v1, M4) |

The M6 run (report `build/klcheck-32b-sim-m6/report.json`, not in git):

| | mean KL (nats) | median | max | top-1 vs CPU | bit-identical rows |
| --- | --- | --- | --- | --- | --- |
| device (sim) vs CPU | **2.886e-3** | 1.79e-4 | 2.15e-1 | **98.44 %** | 0/512 |
| floor (CPU + 1 ulp, seed 1) vs CPU | **2.246e-3** | 1.93e-4 | 1.23e-1 | 99.22 % | 0/512 |

**Ratio 1.285 (limit 3), top-1 98.44 % (limit 80 %): PASS.** Per sequence the device/floor ratio runs
from 0.68 to 1.96; the largest KL is at sequence 0, row 44, for the device and the floor alike. Max
|Δlogit| 3.15 (floor 5.40), min cosine 0.9654. The embd input path equals the token path bit for bit
on this model too. The M1 run (4 prompts, v1 settings) gave mean KL 2.5504e-3 against a floor of
1.8758e-3 (ratio 1.360, top-1 98.83 %), and the M4 run on the card (8 prompts, v1 settings) 2.4113e-3
against 1.8487e-3 (ratio 1.304, top-1 99.02 %).

`TSR_SIM_TMAX=8` cuts the prefill passes of the simulator (from 43 to 7 for 4 prompts, about 12 % of
a run) with bit-identical results (566/566 rows on 0.5B). The default `T_MAX` of the simulated core
is 1 (the k1 value; the k4 bitstream has 8), and the gate runs keep it.

## Tests

- `tools_klcheck_selftest` (no model): the 1-ulp nudge (exactly one ulp, deterministic, balanced
  directions); writing, reading and resuming files, including a torn record; KL against an
  independent long-double evaluation (1e-9 relative); the verdict as the ratio limit crosses the
  measured value; incomplete files fail.
- `tools_klcheck_selftest` also checks the version 2 header: the settings survive a write / read, a
  version 1 file reads with the v1 settings, version 3 is refused.
- `tools_klcheck_qwen05b` (registered only if `models/qwen2.5-0.5b-instruct-q8_0.gguf` exists,
  label `model`, ~25 s): `klcheck all --device sim --phase m6 --n-prompts 4 -n 32` with the default
  criteria and the embd identity check.
- `tools_klcheck_identity_qwen05b` (same condition, label `model`, ~10 s): generate 2 prompts × 8
  tokens on `sim`, replay the tokens on `mock` (step), `compare --expect-identical`.

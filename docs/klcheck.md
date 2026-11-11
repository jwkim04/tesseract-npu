# klcheck: model-level KL check

`klcheck` (`sw/tools/klcheck.cpp`) compares the Tesseract device with the ggml CPU backend on
whole-model logits. The distance between the two is judged against a floor: the distance between the
CPU and the CPU run with its input perturbed by one f32 ulp. This is the model-level row of the
oracle chain in `docs/spec.md` §17 (Verification strategy, *Model-level KL criterion*). The related
settings are SW-22 and [SW](design/06-software-verification.md) §10.1 and §10.4.

A device passes when its mean KL against the CPU is at most 3 times the floor and its top-1
agreement with the CPU is at least 80 %.

## Method

1. Device run. Greedy-generate G tokens per prompt with all layers and the output head on the device (`-ngl 99`,
   `TSR_DEVICE` from `--device`). The prompt is decoded in one batch, then one token per decode. The full logits are
   recorded at the last `--prompt-tail` prompt positions (default 1) and at every generated position except the
   last. Row r is the output at token index `row0 + r` and predicts token `row0 + r + 1`.
2. CPU run (reference). Feed the same sequence (prompt plus device tokens) to the CPU backend alone and record the
   logits at the same positions. The default `--replay batch` decodes each sequence in one `llama_decode`, and
   `--replay step` repeats the device's call pattern (prefill, then one token per decode).
3. Floor. Repeat the CPU run with its input perturbed by one f32 ulp (`--perturb-ulp SEED`). `klcheck all` makes N
   such runs (`--floor-seeds N`, default 3) with the seeds S, S+1, ..., S+N-1, where S is `--seed` (default 1).
4. Compare. For each position compute in double precision `KL(P_cpu || P_x) = sum p_cpu (log p_cpu - log p_x)` in
   nats for x = device and x = floor. Report mean, median and max KL, top-1 agreement (`argmax x == argmax cpu`),
   the largest |dlogit|, the smallest cosine similarity and the number of bit-identical rows, for all positions, the
   prompt-tail rows and the generated rows.

### Pass

`mean KL(cpu || dev) <= max(--min-kl, --kl-ratio x F)` and `top-1 >= --min-top1`, where F is the floor: the mean
over the floor runs of each run's `mean KL(cpu || floor)`. The defaults (`--kl-ratio 3`, `--min-top1 0.8`,
`--min-kl 0`) are the values of the specification. `--min-kl 1e-3` also lets any mean KL up to 1e-3 nats pass, the
absolute form of openTPU's `tools/validate.py`, which the method follows. The check also fails if a file holds
non-finite logits or if the files do not contain the same sequences. The text output has one line per floor run
(mean, median, max, top-1 and the device ratio against that run alone), and the criterion line gives the ratio
against the mean floor and against each seed.

The direction of every nudge comes from the seed, so one seed's floor is a random draw whose mean KL can be set by a
few ill-conditioned positions. The criterion therefore uses the mean of N floor runs. A model with experts or a large
floor needs more rows than a dense one: use at least 8 prompts of 64 tokens.

### The 1-ulp perturbation

The nudge is applied to the input embeddings. The floor run gives llama.cpp `batch.embd` instead of token ids. The
rows come from the GGUF's `token_embd.weight` (found in any split) and are dequantized with
`ggml_get_type_traits(type)->to_float`, the function the CPU's `GET_ROWS` uses. Without the nudge this path
reproduces the token path bit for bit, and `klcheck all` checks this on every run (an extra CPU replay with `--input
embd`, compared with `--expect-identical`), so the floor measures the nudge and nothing else. Models with M-RoPE or
IMROPE (`LLAMA_ROPE_TYPE_MROPE`, `_IMROPE`) need four positions per token in llama.cpp's section-major layout
(`pos[j * n_tokens + i]`), and klcheck writes `[p, p, p, 0]` for every token in that layout. Element j of the
embedding of token index `pos` in sequence `seq` becomes `nextafterf(x, +-inf)`, with the sign taken from the top bit
of `splitmix64(key + j)` and `key = splitmix64(splitmix64(splitmix64(seed) ^ seq) ^ (pos << 1))`. Every element
moves by exactly one ulp (0 becomes plus or minus the smallest subnormal), independent of batching, threads and
`--resume`. `klcheck embd MODEL TOKEN --perturb-ulp S --seq Q --pos P` prints perturbed rows.

The CPU backend quantizes activations to Q8_0 or Q8_K like the device's QACT A32 or A256, so the device and the CPU
differ mainly in accumulation order. Q8_K activations are bit-identical. For Q8_0 the x86 CPU backend uses an AVX
quantizer that differs from QACT A32 (`quantize_row_q8_0_ref`) in about 1 of 3e5 values
([llama-pin.md](llama-pin.md)). An int8 value that rounds differently spreads through later layers, which is why the
floor is not tiny.

### Settings fixed in every phase

Every device, CPU and floor run of a check uses flash attention, a q8_0 K and V cache and llama.cpp's Hadamard KV
rotation off (SW-22, [SW](design/06-software-verification.md) §10.4, [AT](design/03-attention-kv.md) §2.5, AT-17).
`--phase m6` keeps the KV cache and attention on the host (`offload_kqv` false) and sets `TSR_KINDS=m6,+MUL_MAT_ID`
(a dense model has no MUL_MAT_ID node, so this is `m6` for it). `--phase m7` moves the KV cache and attention to the
device, so a decode token is one launch, and passes the caller's `TSR_KINDS` on: unset means every kind of the
booted image, and a list such as `all,-RMS_NORM` is read by the backend
([`sw/ggml-tesseract/README.md`](../sw/ggml-tesseract/README.md)), which moves single kinds back to the CPU to find
the kind behind a difference. `klcheck run` prints a `TSR_KINDS` line on device runs and `report.json` has
`tsr_kinds`. `klcheck all` passes the settings to every child, each file records them (header version 2), and
`compare` warns when two files differ in them. `--fa`, `-ctk` / `-ctv`, `--offload-kqv` and `--attn-rot on` exist for
experiments only.

- Flash attention is on because the pinned llama.cpp refuses a quantized V cache without it and the device's
  attention is `FLASH_ATTN_EXT` only. The KV type is q8_0 in both phases, so they differ only in where attention
  runs.
- The Hadamard rotation (`LLAMA_ATTN_ROT_DISABLE=1`, set by the tool) is off because with a quantized cache llama.cpp
  rotates Q, K and V by an F32 graph-input matrix that the MVM cannot use, which would split attention into CPU
  islands (SW-16, AT-17). The CPU reference uses the same setting.
- CPU weight repack (`use_extra_bufts`) is off (`--repack` turns it on), because repacking copies the weights into
  anonymous memory, which a large model does not fit. Mapped weights can be evicted.
- `--device sim|mock|vkernel|xrt` sets `TSR_DEVICE`, `devices = {Tesseract0}` and `n_gpu_layers = 999`. `--device
  cpu` sets `TSR_DEVICE=none` and `devices = {}`, so a CPU phase never opens a transport. `xrt` needs a build with
  `-DTSR_WITH_XRT=ON` and an xclbin ([xrt-host.md](xrt-host.md)).
- Threads are `-t` (default online CPUs / 2), and the simulator uses `TSR_SIM_THREADS`. `n_ctx` is the longest
  sequence rounded up to 256 and `n_outputs_max` the rows actually needed, which keeps the KV cache, the compute
  buffers and the device IO region small.

## Usage

```
klcheck all -m MODEL [-o DIR] [--device sim] [--seed 1] [--floor-seeds 3] [-n 64] [--n-prompts N] [--prompts FILE]
            [--prompt-tail 1] [--chat] [--replay batch|step] [-t N] [-ub 512] [--phase m6|m7] [--resume]
            [--kl-ratio 3] [--min-top1 0.8] [--min-kl 0] [--no-embd-check] [--verify-k K [--verify-identical]]
klcheck run -m MODEL --device sim|mock|vkernel|xrt|cpu --out FILE [--tokens FILE] [--perturb-ulp SEED] [--verify-k K [--verify-reject on|off]] [...]
klcheck compare --ref CPU.klc --dev DEV.klc [--floor FLOOR.klc]... [--json FILE] [--tsr-kinds LIST] [--expect-identical]
klcheck info FILE [-m MODEL]          # header, timings, per-sequence text
klcheck embd MODEL TOKEN... [--perturb-ulp S --seq Q --pos P]
klcheck selftest [DIR]
```

`klcheck run -h` lists the options of `run`. `klcheck all` runs every phase as its own child process, one after
another, so two models are never loaded at once, and prints each phase's exact command line: a `run` that
generates on the device (`DIR/dev.klc`), a CPU `run` with `--tokens DIR/dev.klc` (the reference), one CPU `run` per
floor seed with `--perturb-ulp` (`floor-seed<S>.klc`, or `floor.klc` for `--floor-seeds 1`), a CPU `run` with
`--input embd` compared with `--expect-identical` (the embd identity), and a final `compare` with one `--floor` per
run and `--json DIR/report.json`. Two floor files with the same seed are an error (exit 2), since one floor would
count twice. The exit code is 0 for PASS, 1 for FAIL and 2 for an error. Without `-o` the files go to a private
temporary directory that is removed at the end.

`report.json` has the summary plus one entry per position (KL device, KL floor, the three argmaxes, |dlogit|,
cosine). `floors` has one entry per floor run (`file`, `seed`, `n`, `kl_mean`, `kl_median`, `kl_max`, `top1`,
`bit_identical`, `ratio`, the device's mean KL over that run's), `floor_kl` is the floor F of the criterion,
`tsr_kinds` is `"m6"`, the caller's list, `"unset"`, or `null` when there is no device run to record, and `versions`
has the capability, the Basis version and the kind table signature of the device run. Per position, `kl_floor` is
the mean over the runs and `kl_floors` lists each run's value. `ratio` is `null` when the floor KL is 0 and the
device KL is not.

Reading the output. The verdict line is PASS or FAIL with the ratio and the top-1 value. A ratio near 1 means the
device sits at the floor: its distance from the CPU is the size of the CPU's own sensitivity to a 1-ulp change of
the input. `bit-identical rows` against the CPU is normally 0, since the accumulation order differs. A model with a
large floor, such as a MoE model whose router can pick another expert, has a large mean KL for both runs, set by a
few positions. Compare the medians and the ratio against each seed there.

- Prompts. 8 built-in prompts (English prose, Python, a question, Korean, arithmetic Q&A, a list, a story, JSON), or
  `--prompts FILE` with one prompt per line (`\n`, `\t`, `\\` escapes; blank lines and `#` lines are skipped). `--chat`
  wraps each prompt in the model's chat template. Generation does not stop at end-of-generation tokens.
- `--replay`. A CPU replay defaults to `batch`. A device replay (`--device sim|mock|vkernel|xrt --tokens FILE`)
  defaults to `step`, which repeats the generating run's prefill and single-token graphs, so that `compare
  --expect-identical` can require every row bit-identical across devices. To compare a card with the simulator,
  generate on `xrt` and replay the tokens on `sim` (with `TSR_SIM_CARD=k4dual TSR_SIM_GDN=1 TSR_SIM_TMAX=4`).
- `--resume`. A phase keeps the complete records of an existing output file and continues after them, and a finished
  phase is skipped before its model is loaded, so `klcheck all --resume` restarts a long run where it stopped. A
  resumed device run is bit-identical to an uninterrupted one.
- Large models need file-backed simulator DRAM (`TSR_SIM_DRAM_DIR=<dir>`).

### Logits file (`.klc`)

The file is little-endian. A 512-byte `klc_header` (magic `TSRKLC01`, version 2) is followed by `n_seq` records. Each
record is a 48-byte `klc_seq` (`KSEQ`, index, n_prompt, n_gen, row0, n_rows, timings, EOG count), then `int32
tokens[n_prompt + n_gen]`, then `f32 logits[n_rows][n_vocab]`, with `row0 + n_rows = n_prompt + n_gen - 1`. The
header records the model, the device, the mode, the input path (tokens, embd or embd+1ulp, and the seed), threads,
repack, chat, timings, the capability, the Basis version and the kind table signature of the device run, and the
settings `type_k`, `type_v`, `offload_kqv`, `attn_rot_disable` and `phase` (6 or 7). Version 1 files (flash
attention off, f16 cache on the host) are still read, and `--resume` refuses to continue one with the settings
above. `klcheck info` prints the settings line. `n_seq` counts complete records and is rewritten after each one. A
phase streams one sequence at a time and `compare` streams rows, so neither depends on the model size. A file takes
`n_rows x n_vocab x 4` bytes per sequence.

## Verification windows

`klcheck run --device D --tokens FILE --verify-k K` replays the tokens of a file the way multi-token prediction
decodes them. The context keeps `K` recurrent snapshots (`n_rs_seq` = K). After the prompt the recorded tokens are
decoded in windows `[i, i + nd]` of up to `K + 1` tokens, every row an output, as a verification batch is. With
`--verify-reject on` (the default) the accepted length `p` of the windows cycles through `0 ... nd`: the tokens after
the first `p + 1` are replaced by wrong ones, only the rows of the first `p + 1` positions are kept, and the window
is rolled back with `llama_memory_seq_rm` to position `i + p`, so the next window reads the snapshot of that
position. `--verify-reject off` accepts every draft. The file is a replay file (`klcheck info`: `verify K 3,
rollbacks`). `klcheck all ... --verify-k K [--verify-identical]` adds a device phase that compares the windows with
the CPU against the same floor (the criterion of "Pass") and with the plain device run's rows. `--verify-identical`
makes the second comparison an `--expect-identical`: the device's numerics do not depend on `T`, so every row must
be bit-identical, rollbacks included. A window of `T` tokens is one launch
([SW](design/06-software-verification.md) §7.11).

## Tests

- `tools_klcheck_selftest` (no model): the 1-ulp nudge (exactly one ulp, deterministic, balanced directions), writing,
  reading and resuming files including a torn record, KL against an independent long-double evaluation (1e-9
  relative), the verdict as the ratio limit crosses the measured value, incomplete files, the version 2 header
  (settings survive a write and read, a version 1 file reads as f16 / f16 on the host, version 3 is refused) and
  several floor runs (each run's numbers equal a `compare` against that run alone, the floor is the mean, the
  `report.json` fields and floor file names, a floor of KL 0 gives ratio `null`).
- `tools_klcheck_qwen05b`, `tools_klcheck_kinds_qwen05b` and `tools_klcheck_identity_qwen05b` (label `model`,
  registered when `models/qwen2.5-0.5b-instruct-q8_0.gguf` exists): `klcheck all --device sim --phase m6 --n-prompts 4
  -n 32` with the default criteria and the embd identity check, the `TSR_KINDS` handling of both phases
  (`test_tsr_kinds.sh`: `--phase m7` with `TSR_KINDS=all,-RMS_NORM` shows that list in the backend and in klcheck's
  `TSR_KINDS` line, `--phase m6` with `TSR_KINDS=all` shows `m6`, `--phase m7` without the variable shows
  `unset`), and generating 2 prompts of 8 tokens on `sim` and replaying them on `mock` with `--expect-identical`.
- `ggml_tesseract_graph_*` (`sw/ggml-tesseract/tests/m7_tests.py`) run klcheck on tiny generated models in full mode
  with the criterion above.

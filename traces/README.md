# Trace sets (`traces/`)

A trace records one launch of the device (or one tile-only push stream) with the DRAM it reads, the
DRAM it writes and the answer. Replaying it on any implementation of the core must reproduce the
recorded results bit for bit. The sets here are the regression suite of the simulator and the firmware
today and the stimulus of the RTL later.

> **Scope.** User and developer documentation of the trace sets and their tools. Normative:
> [SW](../docs/design/06-software-verification.md) §5 (trace format) and §10.2 (ctest plan). The trace data
> is **not in git**: git holds only the `regen.sh` scripts, the two tool wrappers
> (`v2_tools.sh`, `v1_tools.sh`), `m3/trace_stats.py` and `MANIFEST.sha256`. Every set except `model`
> regenerates byte-identically from fixed seeds.

## The format in one page

A trace is a directory ([SW](../docs/design/06-software-verification.md) §5.1 to §5.5):

```text
<trace>/meta.json              configuration, kind, image or push stream, the expected answer, schedule, perf
<trace>/ops/push.bin           the pushed entries in push order, 48-byte records (kind ops)
<trace>/init/ch<c>_<off>.bin   DRAM before the launch (the ranges the launch reads before it writes)
<trace>/expect/ch<c>_<off>.bin DRAM written by the launch, at the answer
<trace>/expect/tickets.bin     per-unit completion counts and error records
<set>/images/<id>.elf          firmware images, shared by the traces of a set (kind launch)
```

`meta.json` (format version 2) has a `kind`: `ops` is a tile-only push stream (the sets below), `launch`
is an image plus a launch descriptor (what `TSR_TRACE` captures). A push record holds the eight
instruction words, two dependencies as tickets relative to the unit's base sequence, the push
value and the entry kind (op, queue control, trace mark), so a trace does not depend on absolute
sequence numbers. DRAM outside the init regions is zero.

Replay compares ([SW](../docs/design/06-software-verification.md) §5.4), everywhere: the answer
record, every expect region bit for bit, the set of written DRAM beats against the expect regions,
the set of read DRAM against init and earlier writes, the per-unit completion counts and the error
records; and, where the push stream is observable (the simulator), the per-unit projections of
`push.bin`. Every generated trace is also a legal single launch under the launch-stateless rule: each
SPM line an op reads is written earlier in the same trace.

## The sets

| Set | Traces | Content | Generator |
| --- | ---: | --- | --- |
| `m2` | 172 | QACT groups and LD to ST copies, no MVM: `tmax1` 60, `tmax8` 16, `small1ch` 16 (1 channel, 64 KiB SPM), `ch3` 8, `errors` 40 (every M2 fault kind and the endings `BAD_DEP`, `DEP_FAILED` chains, `ABORTED`, `QABORT` recipe, `QCLEAR` recovery), `errors8` 32 | `tsr-trace-gen --ops m2` |
| `gen` | 32 | LD, QACT, MVM (with ACC), ST programs with interleaved groups: `tmax1` 24, `tmax8` 8 | `--ops gen` |
| `m3` | 284 | MVM-centred groups mixed with the M2 groups: `tmax1` 64, `tmax8` 32, `small1ch` 24, `ch3` 16, `ch16` 12, `errors` 68 and `errors8` 68 (the 23 MVM fault kinds and the 12 M2 kinds and endings) | `--ops m3` |
| `vu` | 240 | every vector-unit op with legal and illegal encodings: `m8` 96, `m6` 32 (capabilities of the M6 bitstream: `TOPK` and `DQROW` refused), `errors` 88 (40 fault kinds), `small1ch` 24 | `--ops vu` |
| `at` | 160 | the seven attention ops and the flash-attention chain across the DMA, QACT, MVM and VU queues: `tmax8` 64, `tmax1` 24, `errors` 56, `ch3` 16 | `--ops at` |
| `ctrl` | 148 | the issue path: credits and back-pressure, data-flow graphs over every unit, hold windows, `FENCE`, `QABORT` and `QCLEAR`, `BAD_DEP`, `DEP_FAILED`: `tmax8` 64, `errors` 36, `deep` 16 (queue depth 31), `shallow` 16 (depth 1), `nocaps` 16 (no VU or AT) | `--ops ctrl` |
| `model-v1conv` | 290 | regression traces converted from the earlier op format: every split of a one-token Qwen2.5-0.5B Q8_0 run and of a Q4_K_M requantization (145 each), each instruction a push that depends on the previous one (a serial chain keeps the sequential semantics of the source) | `tsr-trace-conv` |
| `model` | 290 | the input of the conversion (`cmd.bin` traces of the earlier op format); the current tools cannot regenerate it | none |

On-disk sizes: `m3` 166 MB, `model` 998 MB, `model-v1conv` 1001 MB, `at` 32 MB, `ctrl` 26 MB, `m2`
12 MB, `vu` 12 MB, `gen` 5.3 MB. The counts are the number of directories with a `meta.json`
(`find traces/<set> -name meta.json | wc -l`).

`model-v1conv` is the check that the results of the tile units match the results recorded when the
traces were made: the replay must match the recorded results bit for bit.

## Regenerate a set

```sh
traces/m3/regen.sh build/dev        # any of m2 gen m3 vu at ctrl; the argument is the build directory (default build/dev)
traces/model-v1conv/regen.sh build/dev
```

Each `regen.sh` needs `bin/tsr-trace-gen` and `bin/tsr-trace-replay` of the build directory (targets
`tsr-trace-gen`, `tsr-trace-replay`, `tsr-trace-conv`; `traces/v2_tools.sh` checks the tools before anything is deleted). It
deletes the set, runs the generator with the seeds and options written in its header, then replays
every trace under the fixed schedule and two random ones (`random:1`, `random:2`) with the race
detector fatal, tile-only and through the `replay` kernel of the build's production image
(`build/dev/fw/tsr-fw.elf`; skipped with a note when the image is not built). A failing replay fails
the script.

`traces/model-v1conv/regen.sh` converts `traces/model` (it needs that set on disk) with
`tsr-trace-conv`, replays the result, and replaces the old set only after every replay passed.

## Verify a set against the manifest

`MANIFEST.sha256` holds, per set, the number of files and the digest of the files in byte-order
path order (the `regen.sh` of the set is excluded):

```sh
cd traces && find <set> -type f ! -name regen.sh -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

| Set | Files |
| --- | ---: |
| `m2` | 1,211 |
| `gen` | 557 |
| `vu` | 1,757 |
| `at` | 4,066 |
| `ctrl` | 5,811 |
| `m3` | 5,105 |
| `model-v1conv` | 3,140 |

The digests are those of the dev host (the sets on disk match them). The generator draws some random
inputs through libm (normal deviates, the ROPE constants), so another libm may produce different
bytes: the replay still has to pass, but the digest can differ. A frozen reference set goes into git
once the ISA is frozen for the RTL, after the M6 RTL.

## Replay

```sh
tsr-trace-replay -q --sched fixed --sched random:1 traces/m3            # fresh simulators, tile-only
tsr-trace-replay -q --replay-image build/dev/fw/tsr-fw.elf traces/m3    # through the replay kernel
tsr-trace-replay -q --serial --replay-image build/dev/fw/tsr-fw.elf traces/model-v1conv
tsr-trace-replay -q --device mock --shadow traces/ctrl                  # through a transport
```

`PATH` is a trace or a set (searched recursively for `meta.json`). Exit status 0 means all passed, 1 a
mismatch, 2 a trace that could not be read or set up. Routes:

| Route | Meaning |
| --- | --- |
| default | a fresh simulator per trace and schedule (`--sched S`, repeatable; `--rv-ratio`, `--threads`, `--tmax`); kind `ops` goes in through the issue port, kind `launch` boots the trace's image and pushes its descriptor |
| `--replay-image ELF` | kind `ops` goes through the `replay` kernel of that image, as a real launch (so `SPM_STALE` is checked); `--serial` makes every record depend on the previous one, which the `model-v1conv` traces require |
| `--device KIND` | replays through the transport (`sim`, `mock`; `vkernel` and `xrt` later): the device boots `--replay-image` (default `$TSR_IMAGE`, else the production image), kind `ops` goes through its `replay` kernel; the answer and every expect region are compared. A trace made for another device configuration is skipped (counted, not failed). `--shadow` also runs every launch on an in-process simulator and compares |

`TSR_SIM_THREADS` sets the MVM threads of each simulator (default 1 for the tool); results do not
depend on it. The ctests `tools_trace_gen_replay`, `tools_trace_replay_device` and the firmware test
`fw_replay_sim` replay samples of the sets; `fw_replay_sim_all` (label `slow`) replays the whole sets
through the replay kernel, and `sim_trace` and `trace_v2_roundtrip` test the format and the library.

## Generate and convert

```sh
tsr-trace-gen -o DIR [-n COUNT] [-s SEED] --ops gen|m2|m3|vu|at|ctrl [--t-max T] [--num-ch C] [--spm-kib S] \
              [--caps none|m6|m8|all|N] [--q-depth D0,D1,D2,D3,D4,D5] [--errors PCT] [...]
tsr-trace-conv [-q] [--replay] [--sched S]... [--limit N] [--pick P1,P2,...] -o OUT IN...
```

`tsr-trace-gen` writes `ops` traces directly: random tile programs with dependencies, concurrency and
error injection (a faulting op, `BAD_DEP`, a `DEP_FAILED` chain, the `QABORT` or `QCLEAR` recipe), and
runs each on the simulator to fill the answer, the expect regions and the tickets. `tsr-trace-gen
--help` lists every option; `--mutant-stale` produces a trace that must be refused (a test of the
launch-stateless check). `tsr-trace-conv` reads traces of the earlier op format (`cmd.bin`) and writes `ops` traces.

## Capture a trace

```sh
TSR_TRACE=build/trace-run TSR_DEVICE=sim build/dev/bin/llama-completion -m MODEL -ngl 99 -nkvo ...
```

`TSR_TRACE=<dir>` makes the simulator dump one `launch` trace per launch under `<dir>/<seq, 6 digits>/`
and the firmware images under `<dir>/images/`. A process that creates several simulators writes the second
and later ones to `sim<N>/<seq>/`. Replay the result with `tsr-trace-replay <dir>`. A launch captured at sequence s
replays on a freshly booted simulator, where it runs as sequence 0.

## What replays where

| Target | Status |
| --- | --- |
| Simulator, tile-only | on the simulator: all sets, fixed and random schedules, race detector fatal |
| Simulator, through the `replay` kernel | on the simulator: all sets; `model-v1conv` with `--serial` |
| Transport kinds `sim`, `mock` | on the simulator: `tsr-trace-replay --device`, `tsr-selftest --traces` |
| RTL, tile-only replay into the issue FIFO | planned (W4, W5): a Verilator bench pushes `push.bin` straight into the issue FIFO with no control core |
| RTL with the control core (Ibex lockstep against `tsr_rv`) | planned (W5) |
| The card | planned (W6); `--device xrt` after the bitstream with the control core |

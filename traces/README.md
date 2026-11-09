# Trace sets (`traces/`)

A trace records one launch of the device (or one tile-only push stream) with the DRAM it reads, the DRAM it writes
and the answer. Replaying it on any implementation of the core must reproduce the recorded results bit for bit. The
sets here are the regression suite of the simulator and the firmware and the stimulus of the RTL benches.
Normative: [SW](../docs/design/06-software-verification.md) §5 (trace format) and §10.2 (ctest plan). The trace data
is not in git. Git holds only the `regen.sh` scripts, the tool wrapper `trace_tools.sh` and `MANIFEST.sha256`. Every
set except `model` regenerates byte-identically from fixed seeds.

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

`meta.json` (format version 2) has a `kind`: `ops` is a tile-only push stream (the sets below), `launch` is an image
plus a launch descriptor (what `TSR_TRACE` captures). A push record holds the eight instruction words, two
dependencies as tickets relative to the unit's base sequence, the push value and the entry kind, so a trace does not
depend on absolute sequence numbers. DRAM outside the init regions is zero. Replay compares (SW §5.4) the answer
record, every expect region bit for bit, the set of written DRAM beats against the expect regions, the set of read
DRAM against init and earlier writes, the per-unit completion counts and the error records, and, where the push
stream is observable (the simulator), the per-unit projections of `push.bin`. Every generated trace is also a legal
single launch under the launch-stateless rule.

## The sets

| Set | Content | Generator |
| --- | --- | --- |
| `m2` | QACT groups and LD to ST copies, no MVM: `tmax1`, `tmax8`, `small1ch` (1 channel, 64 KiB SPM), `ch3`, `errors` and `errors8` (every fault kind of these units and the endings `BAD_DEP`, `DEP_FAILED` chains, `ABORTED`, `QABORT` recipe, `QCLEAR` recovery) | `tsr-trace-gen --ops m2` |
| `gen` | LD, QACT, MVM (with ACC), ST programs with interleaved groups: `tmax1`, `tmax8` | `--ops gen` |
| `m3` | MVM-centred groups mixed with the `m2` groups: `tmax1`, `tmax8`, `small1ch`, `ch3`, `ch16`, `errors` and `errors8` | `--ops m3` |
| `vu` | every vector-unit op with legal and illegal encodings: `m8`, `m6` (a configuration without `TOPK` and `DQROW`), `errors`, `small1ch` | `--ops vu` |
| `at` | the seven attention ops and the flash-attention chain across the DMA, QACT, MVM and VU queues: `tmax8`, `tmax1`, `errors`, `ch3` | `--ops at` |
| `ctrl` | the issue path: credits and back-pressure, data-flow graphs over every unit, hold windows, `FENCE`, `QABORT` and `QCLEAR`, `BAD_DEP`, `DEP_FAILED`: `tmax8`, `errors`, `deep` (queue depth 31), `shallow` (depth 1), `nocaps` (no VU or AT) | `--ops ctrl` |
| `model-v1conv` | regression traces converted from the legacy `cmd.bin` op format: every split of a one-token Qwen2.5-0.5B Q8_0 run and of a Q4_K_M requantization, each instruction a push that depends on the previous one | `tsr-trace-conv` |
| `model` | the input of the conversion (`cmd.bin` traces). The current tools cannot regenerate it. | none |

`traces/<set>/regen.sh [build-dir]` (default `build/dev`; `m2`, `gen`, `m3`, `vu`, `at`, `ctrl`, `model-v1conv`)
needs `bin/tsr-trace-gen` and `bin/tsr-trace-replay` of the build directory (targets `tsr-trace-gen`,
`tsr-trace-replay`, `tsr-trace-conv`; `trace_tools.sh` checks the tools before anything is deleted). It deletes the
set, runs the generator with the seeds and options written in its header, then replays every trace under the fixed
schedule and two random ones (`random:1`, `random:2`) with the race detector fatal, tile-only and through the
`replay` kernel of the build's production image. A failing replay fails the script. `model-v1conv/regen.sh` converts
`traces/model` (it needs that set on disk) and replaces the old set only after every replay passed.

`MANIFEST.sha256` holds, per set, the number of files and the digest of the files in byte-order path order (the
`regen.sh` of the set is excluded):

```sh
cd traces && find <set> -type f ! -name regen.sh -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
```

The generator draws some random inputs through libm (normal deviates, the ROPE constants), so another libm may produce
different bytes. The replay still has to pass, but the digest can differ.

## Replay

```sh
tsr-trace-replay -q --sched fixed --sched random:1 traces/m3            # fresh simulators, tile-only
tsr-trace-replay -q --replay-image build/dev/fw/tsr-fw.elf traces/m3    # through the replay kernel
tsr-trace-replay -q --serial --replay-image build/dev/fw/tsr-fw.elf traces/model-v1conv
tsr-trace-replay -q --device mock --shadow traces/ctrl                  # through a transport
```

`PATH` is a trace or a set (searched recursively for `meta.json`). Exit status 0 means all passed, 1 a mismatch, 2
a trace that could not be read or set up. By default each trace and schedule gets a fresh simulator (`--sched S`,
repeatable; `--rv-ratio`, `--threads`, `--tmax`): kind `ops` goes in through the issue port, kind `launch` boots the
trace's image and pushes its descriptor. With `--replay-image ELF` kind `ops` goes through the `replay` kernel of
that image, as a real launch (so `SPM_STALE` is checked), and `--serial` makes every record depend on the previous
one, which `model-v1conv` requires. With `--device KIND` (`sim`, `mock`, `vkernel`, `xrt`) the replay goes through the
transport: the device boots `--replay-image` (default `$TSR_IMAGE`, else the production image), and the answer and
every expect region are compared. A trace made for another device configuration is skipped (counted, not failed).
`--shadow` also runs every launch on an in-process simulator and compares. `tsr-selftest --traces DIR` does the same
inside the card selftest ([`docs/xrt-host.md`](../docs/xrt-host.md) §5). `TSR_SIM_THREADS` sets the MVM threads of
each simulator (default 1 for the tool). The ctests `tools_trace_gen_replay`, `tools_trace_replay_device` and
`fw_replay_sim` replay samples of the sets, `fw_replay_sim_all` (label `slow`) replays them all through the replay
kernel, and `sim_trace` and `trace_v2_roundtrip` test the format and the library.

## Generate, convert and capture

```sh
tsr-trace-gen -o DIR [-n COUNT] [-s SEED] --ops gen|m2|m3|vu|at|ctrl [--t-max T] [--num-ch C] [--spm-kib S] \
              [--caps none|m6|m8|all|N] [--q-depth D0,D1,D2,D3,D4,D5] [--errors PCT] [...]
tsr-trace-conv [-q] [--replay] [--sched S]... [--limit N] [--pick P1,P2,...] -o OUT IN...
TSR_TRACE=build/trace-run TSR_DEVICE=sim build/dev/bin/llama-completion -m MODEL -ngl 99 ...
```

`tsr-trace-gen` writes `ops` traces directly: random tile programs with dependencies, concurrency and error
injection, run on the simulator to fill the answer, the expect regions and the tickets. `--help` lists every option,
and `--mutant-stale` produces a trace that must be refused (a test of the launch-stateless check). `tsr-trace-conv`
reads traces of the legacy `cmd.bin` format and writes `ops` traces. `TSR_TRACE=<dir>` makes the simulator dump
one `launch` trace per launch under `<dir>/<seq, 6 digits>/` and the firmware images under `<dir>/images/`. A process
that creates several simulators writes the second and later ones to `sim<N>/<seq>/`. A launch captured at sequence
s replays on a freshly booted simulator, where it runs as sequence 0.

The RTL replays the sets too: a Verilator bench pushes `push.bin` straight into the issue FIFO with no control core,
and the cocotb core benches (`tb/core`) replay them with the control core.

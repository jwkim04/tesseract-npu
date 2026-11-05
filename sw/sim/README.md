# Simulator and references (`sw/sim`)

`sw/sim` is the oracle of the project. It holds two C11 libraries: `tsr_ref`, the numerics and
layouts of the device bit for bit, and `tsr_sim`, a model of the NPU core and of the control core
that runs real firmware images. The ggml backend, the transport, the tools, the RTL testbenches
(through `ctypes`) and the firmware tests all call this code; the numerics exist nowhere else.

> **Scope.** Developer documentation of the simulator as built in this tree. Normative:
> [SW](../../docs/design/06-software-verification.md) §4 (simulator), §5 (trace format) and §10
> (verification), [EX](../../docs/design/01-execution-model.md) §5 to §8 (CSRs, PERF, ISA, errors),
> [VU](../../docs/design/02-vector-unit.md) §4 to §6 and [AT](../../docs/design/03-attention-kv.md) §4 to §6
> (vector and attention numerics). Where this directory and `docs/spec.md` disagree, the
> specification wins and the simulator is fixed. The simulator has no timing of its own that the
> card has to match: device cycles come from an informational cycle model, and nothing in the
> bit-exact contract depends on them.

## Two libraries

| Library | Files | Content |
| --- | --- | --- |
| `tsr_ref` (static) and `libtsr_ref.so` (`tsr_ref_shared`) | `tsr_ref.c`, `tsr_ref_math.c`, `tsr_ref_vu.c`, `tsr_ref_at.c` | The reference numerics. `tsr_ref.c`: every f32 operation with DAZ, FTZ and canonical NaN, device weight layout and repack, QA quantization (`tsr_qact`), the dot products and the fixed accumulation tree (`tsr_dot_row`). `tsr_ref_math.c`: the vector unit's math (`EXPK`, reciprocal, `SIN`, `COS`, `GELU16`, composites), exhaustive over all 2^32 inputs where the input space allows. `tsr_ref_vu.c`: `VOP`, `RED`, `ROPE`, `TOPK`, `SOP`, `DQROW`. `tsr_ref_at.c`: the KV plane layout, `KVST`, `ATTQK`, `ATTPV`, `ASMAX`, `AMASK`, `AMERGE`, `ANORM`, the composite flash attention `tsr_ref_fa` (its chunk planner is `include/tsr_fa_plan.h`, shared with the firmware) |
| `tsr_sim` (static) and `libtsr_sim.so` (`tsr_sim_shared`, `libtsr_ref` linked in) | `tsr_sim.c`, `tsr_q.c`, `tsr_sched.c`, `tsr_vu.c`, `tsr_at.c`, `tsr_rv.c`, `tsr_bus.c`, `tsr_race.c`, `tsr_trace.c`, `tsr_dm.c`, `tsr_dbg.c` | The device model, below |

Plain C11 compiled with `-ffp-contract=off -fno-fast-math`: every f32 operation is a separately
rounded IEEE binary32 operation, and the reference never reads or writes MXCSR (the same process
also runs ggml's CPU code). Nothing in `tsr_ref` depends on ggml; ggml-layout blocks appear as plain
structs.

## What `tsr_sim` models

| Part | File | Models |
| --- | --- | --- |
| Core | `tsr_sim.c` | SPM, per-channel DRAM (lazily mapped; `TSR_SIM_DRAM_DIR` file-backed), the host CSR port (identity, `CTRL`, `STATUS`, firmware loader with receive CRC and `VERIFY`, launch queue LQ, completion ring LCQ, log ring, watermarks, `PERF`, `Q_*`, `SREG`, `SOFT_RESET`), the units `NOP`, `LD`, `ST`, `QACT`, `MVM`, `FENCE` and the PERF counters |
| Queue model | `tsr_q.c` | the control window and issue port, the issue FIFO, the dispatcher and the checker (error order of EX §8.1), one in-order queue per unit with tickets, dependencies, `SYNC`, `QCLEAR`, `QHOLD`, `QABORT`, error capture and poisoning, invariants I1 to I3 |
| Schedule | `tsr_sched.c` | the time base's policies (below) |
| Vector and attention units | `tsr_vu.c`, `tsr_at.c` | decode, dispatch checks, operand extents and a cycle model of the VU ops and of the attention ops (`KVST`, `ATTQK`, `ATTPV`, `ASMAX`, `AMASK`, `AMERGE`, `ANORM`); every numeric result is a call into `tsr_ref_vu.c` or `tsr_ref_at.c` |
| Control core | `tsr_rv.c`, `tsr_bus.c` | `tsr_rv`: an RV32IMC + Zicsr + Zifencei interpreter of the control core's Ibex (machine mode, interrupts, debug mode, two cycle tables `small` and `maxperf`); `tsr_bus.c`: the firmware bus profile (ITCM, DTCM, debug module memory, control window, core window, SPM window and its line-buffered alias), a test profile (RAM with HTIF, for riscv-tests), the system-bus master for the debug module, the ELF loader, and the agent that runs the hart on the core's time base |
| Race detector | `tsr_race.c` | vector clocks over the units, the RISC-V and the host; reports races on SPM bytes, DRAM beats and the `s` registers, and `SPM_STALE` (a launch that reads SPM it did not write) |
| Trace | `tsr_trace.c` | reading, writing and strict validation of trace directories, the image id, capture of one trace per launch, replay with the comparisons of SW §5.4 |
| Debug | `tsr_dm.c`, `tsr_dbg.c` | a behavioural riscv-dbg debug module on the built-in hart (DMI profile of DB §9.1) and the tile-level debug models of DB (stop points, step, serial mode, tile breakpoints, dispatch history, unit probes, the trace writer) |

The optional models (units, RISC-V, race detector, trace, debug) are separate CMake snippets
(`units.cmake`, `rv.cmake`, `race.cmake`, `trace.cmake`, `dm.cmake`, `dbg.cmake`; reference parts
in `ref_math.cmake`, `refvu.cmake`, `refat.cmake`). `-DTSR_SIM_SKIP_SNIPPETS="race;dbg"` leaves some
out, and every subset builds. A unit queue exists for the DMA, QACT, MVM, VU and SYNC units; the
unit-4 slot of the issue path is reserved and absent in the M6 configuration (queue depth 0).

Headers in `include/`: `tsr_sim.h` (the C API: create with `tsr_sim_create_opts`, host CSR
port `tsr_sim_csr_read` / `_write`, time base `tsr_sim_advance` / `_now` / `_run_until_idle`, DRAM
access, access capture, test hooks that push tile ops as the control core would), `tsr_ref.h`,
`tsr_ref_math.h`, `tsr_ref_vu.h`, `tsr_ref_at.h`, `tsr_fa_plan.h`, `tsr_isa.h` and the generated
`tsr_tables.h` (opcodes, units, errors, the CSR map, PERF slots, encodings; generated from
`sw/tables/tsr_tables.py` by `sw/tables/gen_tables.py`, checked by the ctest `tables_check`),
`tsr_rv.h`, `tsr_bus.h`, `tsr_race.h`, `tsr_trace.h`, `tsr_dm.h`, `tsr_dbg.h`. The internal
interface between the files is `tsr_sim_int.h` (not installed).

## Time, schedules and determinism

Time is one clock in core cycles (`MTIME`). Nothing moves unless the caller advances time:
`tsr_sim_advance(sim, max_cycles)` runs the model until a host-visible event (a completion record,
log words, the RISC-V halted or trapped, a reset finished, the queues idle) or the budget ends. Ties
are broken by (time, event kind, unit, sequence), so a run is reproducible. A wait loop of the
firmware (the same timing register read four times in a row with no other access) is
fast-forwarded to the next event, and `WFI` jumps to the next interrupt; poll counts are therefore
not architectural.

`TSR_SIM_SCHED` picks the schedule:

- `fixed` (default): op durations from the cycle model, a 2-cycle issue crossing, the RISC-V-side
  copies of `DONE_SEQ`, credits, `QSTATUS` and error records without lag.
- `random:<seed>[,<min>,<max>]`: every op duration is multiplied by a factor drawn from `[min, max]`
  (default 0.25 to 4), crossings take 2 to 8 cycles, the RISC-V cost of each instruction varies by
  one cycle, and the RISC-V-side copies lag the core. The generator is xoshiro128**, one stream per
  purpose, so adding a draw in one place never shifts another.

A program is *valid* when it is race-free and, under every schedule, the per-unit op streams (words,
flags, dependencies as relative tickets), the completion records r0 to r6, the RISC-V's SPM stores
and the final written DRAM and SPM lines are the same ([SW](../../docs/design/06-software-verification.md)
§4.7). The tests run the fixed schedule and seven random ones with the race
detector fatal. Never compared: poll counts, `MTIME`, `PERF`, record word r7 (duration), completion
records (REC) and the log. RTL and the card have to reproduce the expect DRAM regions, the answer
record, the per-unit completion counts and the error records.

`TSR_SIM_RACE=off|report|fatal` controls the race detector. `off` is the default for model runs
(the detector costs a factor of 1.0 to 1.3 on firmware and random tile programs, measured on the
simulator at `1ed7460`; the design estimate was 2 to 3); the ctests that run firmware or queue programs set `fatal`. Besides races, the
detector checks the launch-stateless rule: `SPM_STALE` is reported when an op or the RISC-V reads an
SPM line that nothing wrote since the runtime popped the launch descriptor.

The RISC-V time base: `TSR_SIM_RV_CORE=maxperf|small` selects the Ibex cycle table (default
`maxperf`, CC-34; timing only, every architectural result is the same) and `TSR_SIM_RV_RATIO`
the ratio of the control-core clock to the core clock (0.05 to 4, default 166.7 / 276.3 = 0.603).
Both are informational estimates until the RVFI lockstep against the RTL exists (work package W5).

## Environment

| Variable | Values | Read by |
| --- | --- | --- |
| `TSR_SIM_SCHED` | `fixed`, `random:<seed>[,<min>,<max>]` | `tsr_sim_create_opts` |
| `TSR_SIM_RACE` | `off`, `report`, `fatal` | `tsr_sim_create_opts` (ignored with a note when the library was built without `race.cmake`) |
| `TSR_SIM_RV_CORE` | `maxperf`, `small` | `tsr_sim_create_opts` |
| `TSR_SIM_RV_RATIO` | number in `[0.05, 4]` | `tsr_sim_create_opts` |
| `TSR_SIM_LOG` | `stderr` or a file, optionally prefixed `rv,` | `tsr_sim_create_opts`: queue events (push, enqueue, start, done) and one line per `tsr_sim_op`; `rv,` adds one line per retired RISC-V instruction |
| `TSR_TRACE` | a directory | `tsr_sim_create_opts`: one trace per launch, in `<dir>/<seq, 6 digits>/`, firmware images in `<dir>/images/` |
| `TSR_TRACE_KR_DUMP` | any value | `tsr_trace.c`: with verbose replay through the replay kernel, prints the captured launch (debugging aid) |
| `TSR_SIM_THREADS` | 1 to 64 | MVM worker threads of the simulator behind the `sim` and `mock` transport kinds (default `min(online CPUs, 8)`) and of the tools that create simulators themselves (`tsr-trace-replay`, `tsr-trace-conv`, the `tsr-selftest` shadow: default 1; `tsr-numcheck`: 4 when unset). The library default is 1. Results do not depend on it |
| `TSR_SIM_TMAX`, `TSR_SIM_CH_GIB`, `TSR_SIM_DRAM_DIR` | 1 to 8 (default 1), 1 to 16 (default 16), a directory | the `sim` transport kind (configuration of the simulated core: `T_MAX`, channel size, file-backed DRAM for models that do not fit in RAM) |
| `TSR_SIM_ATT_LOCAL`, `TSR_SIM_MVM_LOCAL_SCALE`, `TSR_SIM_MVM_LOCAL_SOFTCAP` | 0 or 1 (default 1) | `sim` transport feature controls for matched legacy comparisons. The four-owner capabilities are reported only on four-channel configurations. Local execution uses the existing functional cycle model. |
| `TSR_REF_MATH_FULL`, `TSR_REF_MATH_THREADS` | `1`, a number | the opt-in exhaustive `ref_math_*_full` tests |

A value that does not parse makes `tsr_sim_create_opts` fail with a message. The options exist as
fields of `tsr_sim_opts` too; a field that is set wins over the variable.

## Use

**By the transport.** `TSR_DEVICE=sim` (the default) opens the simulator in-process behind the core
CSR port, with no adapter; `TSR_DEVICE=mock` puts the same simulator inside a model of the
`ap_ctrl_none` platform adapter with fault injection. The host stack, the backend and the tools see
the same register protocol as on a card ([`sw/transport/README.md`](../transport/README.md)). The
simulator never reaches the card path: test hooks such as `tsr_sim_push` are not reachable through
the transport.

**By tests.** The unit tests link `tsr_sim` and drive the C API directly; the firmware tests
(`sw/ctrl/tests`) boot a firmware image on it. `libtsr_ref.so` and `libtsr_sim.so` serve Python
through `ctypes` (the numpy models of `ref_math_np` and `ref_vu_np`, `sw/tools/numcheck.py`).

**From the command line.** `tsr-trace-replay` replays trace directories on fresh simulators;
`tsr-trace-gen` and `tsr-trace-conv` write and convert them; `tsr-selftest --device sim` runs the
`selftest` kernel with a simulator shadow ([`traces/README.md`](../../traces/README.md),
[`sw/ctrl/README.md`](../ctrl/README.md)). `TSR_SIM_LOG=rv,stderr` is the quickest way to see what
the firmware does.

## Speed

The simulator is single-process, with `TSR_SIM_THREADS` worker threads for large MVMs. Measured on
the simulator (`83ac601`, dev host with 16 cores): Qwen2.5-0.5B decodes at 90 to 95 ms per token
with 145 launches per token; Qwen2.5-32B at about 8 s per token with file-backed DRAM
(`TSR_SIM_DRAM_DIR`, about 20 GiB free) and 385 launches per token. A 32B klcheck run (8 prompts of
64 tokens) takes about 77 minutes. The RISC-V interpreter itself is fast; the cost is the MVM
arithmetic.

## Tests

`ctest --test-dir build/dev -N` lists them. The names by area:

| Area | Tests | What they check |
| --- | --- | --- |
| Tables | `tables_check`, `tables_selftest` | the generated headers equal the table source |
| Reference math | `ref_math_specials`, `ref_math_acc_*`, `ref_math_np_*`, `ref_math_gelu16`, `ref_math_dump_smoke` | the VU's math against double libm and against an independent numpy model, bit for bit; `*_full` (label `slow`) over all 2^32 inputs with `TSR_REF_MATH_FULL=1` |
| References | `ref_layout`, `ref_fp`, `ref_ggml_*`, `ref_dot`, `ref_ctypes`, `ref_vu`, `ref_vu_np`, `ref_at`, `ref_at_ggml` | layouts, f32 primitives, QA quantization and dequantization against ggml, dot products, the VU and AT references against numpy models and (flash attention) against ggml-cpu's `FLASH_ATTN_EXT` |
| Core and queues | `sim_basic`, `sim_mvm`, `sim_capture`, `sim_csr`, `sim_queue`, `sim_transport`, `sim_vu`, `sim_at` and `*_ubsan` | CSR port, loader, rings, checker order, dependencies, errors, every VU and AT op, with UBSan variants |
| Schedules and races | `sim_sched_unit`, `sim_sched_determinism`, `sim_race*`, `sim_*_race`, `sim_rv_issue*`, `sim_w1b_e2e` | the random schedule's streams and determinism, the race detector on directed programs and on firmware, the W1 end-to-end program |
| RISC-V | `rv_unit`, `rv_unit_enc`, `rv_agent`, `rv_sim`, `rv_timing`, `rv_isa`, `rv_arch`, `rv_lockstep`, `rv_random`, `rv_random_long`, `rv_spike_*`, `rv_harness_selftest` | the interpreter against hand-encoded sequences, the GNU assembler, riscv-tests, riscv-arch-test and random programs, in lockstep against Spike; label `rv` |
| Debug | `sim_dm*`, `sim_dbg*` | the debug-module model (halt, resume, abstract commands, system bus, breakpoints, a GDB-style session) and the tile-level debug models |
| Trace | `sim_trace*`, `trace_v2_roundtrip` | trace read, write, validation, capture and replay under three schedules |

The Spike-based tests (`rv_spike_*`, `rv_arch`, `rv_lockstep`, `rv_random`) need the pinned tools of
`sw/ctrl/tools/install-toolchain.sh` and report "Skipped" without them; `rv_random_long` runs only
with `TSR_RV_RANDOM_LONG=1`. The conformance harness is `sw/sim/tests/rvtests/` (standard-library
Python); `-DTSR_SPIKE=<path>` and `-DTSR_RISCV_TOOLS=<prefix>` point it at another install.

## Limits

- The cycle model is informational: unit busy times follow simple formulas in `tsr_sim.c`, `tsr_vu.c`
  and `tsr_at.c`, and the RISC-V cost follows a fixed table. Hardware counters are compared against
  it, not bit-matched.
- Effects of an op are applied atomically at its modelled completion time. For race-free programs
  this is invisible; the race detector rejects the others.
- The debug-module model does not model the debug ROM (abstract commands act on the interpreter
  directly, DB-18).
- `VMASK` (M9) has no handler and completes `ILLEGAL_OP`; `TOPK` and `DQROW` exist in the simulator
  (RTL in M8).
- Ibex facts that were recalled from its documentation and not yet checked against the vendored
  RTL are marked *confirm* in `tsr_rv.c`; the RVFI lockstep bench of W5 closes them.

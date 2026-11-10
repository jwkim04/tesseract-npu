# Simulator and references (`sw/sim`)

`sw/sim` is the oracle of the project. It holds two C11 libraries: `tsr_ref`, the numerics and layouts of the device
bit for bit, and `tsr_sim`, a model of the NPU core and of the control core that runs real firmware images. The ggml
backend, the transport, the tools, the RTL testbenches (through `ctypes`) and the firmware tests all call this code.
The numerics exist nowhere else. Where this directory and `docs/spec.md` disagree, the specification wins and the
simulator is fixed. Device cycles come from an informational cycle model, and nothing in the bit-exact contract depends
on them. Normative: [SW](../../docs/design/06-software-verification.md) §4 (simulator), §5 (trace format) and §10
(verification), [EX](../../docs/design/01-execution-model.md) §5 to §8, [VU](../../docs/design/02-vector-unit.md) §4 to
§6 and [AT](../../docs/design/03-attention-kv.md) §4 to §6.

`tsr_ref` (static, and `libtsr_ref.so` as `tsr_ref_shared`) is plain C11 compiled with `-ffp-contract=off
-fno-fast-math`: every f32 operation is a separately rounded IEEE binary32 operation, and it never touches MXCSR. It
does not depend on ggml. `tsr_ref.c` has the f32 operations with DAZ, FTZ and canonical NaN, the device weight layout
and repack, QA quantization (`tsr_qact`), the dot products and the fixed accumulation tree (`tsr_dot_row`).
`tsr_ref_math.c` has the vector unit's math (`EXPK`, reciprocal, `SIN`, `COS`, `GELU16`), exhaustive over all 2^32
inputs where the input space allows. `tsr_ref_vu.c` has `VOP`, `RED`, `ROPE`, `TOPK`, `SOP` and `DQROW`,
`tsr_ref_at.c` the KV plane layout, `KVST`, `ATTQK`, `ATTPV`, `ASMAX`, `AMASK`, `AMERGE`, `ANORM` and the composite
flash attention `tsr_ref_fa`, and `tsr_ref_gdn.c` the fused Gated DeltaNet op.

`tsr_sim` (static, and `libtsr_sim.so` as `tsr_sim_shared`) models the device: the core (`tsr_sim.c`: SPM, per-channel
DRAM, lazily mapped or file-backed with `TSR_SIM_DRAM_DIR`, the host CSR port with the firmware loader, launch queue,
completion ring and log ring, the units `LD`, `ST`, `QACT`, `MVM`, `FENCE` and the PERF counters), the queue model
(`tsr_q.c`: the issue port and FIFO, the dispatcher and checker with the error order of EX §8.1, one in-order queue
per unit with tickets and dependencies), the vector and attention units (`tsr_vu.c`, `tsr_at.c`: decode, dispatch
checks and a cycle model, with every numeric result a call into the reference files), the control core (`tsr_rv.c`
and `tsr_bus.c`: an RV32IMC + Zicsr + Zifencei interpreter of Ibex with the firmware bus profile, a test profile
for riscv-tests, and the ELF loader), the race detector (`tsr_race.c`: vector clocks over the units, the RISC-V and
the host, reporting races on SPM bytes, DRAM beats and `s` registers and `SPM_STALE`, a launch that reads SPM it did
not write), traces (`tsr_trace.c`: reading, writing, validation, capture and replay with the comparisons of SW §5.4),
and the debug models (`tsr_dm.c`, `tsr_dbg.c`: a behavioral riscv-dbg module and the tile-level debug blocks of DB).
The optional parts are CMake snippets (`units`, `rv`, `race`, `trace`, `dm`, `dbg`, and `ref_math`, `refvu`, `refat`,
`refgdn`), `-DTSR_SIM_SKIP_SNIPPETS="race;dbg"` leaves some out, and every subset builds. `include/tsr_sim.h` is the C
API (`tsr_sim_create_opts`, `tsr_sim_csr_read` and `_write`, `tsr_sim_advance`, `_now`, `_run_until_idle`, DRAM access
and test hooks that push tile ops as the control core would), and `tsr_tables.h` is generated from
`sw/tables/tsr_tables.py` (checked by the ctest `tables_check`).

## Time, schedules and determinism

Time is one clock in core cycles (`MTIME`). Nothing moves unless the caller advances time: `tsr_sim_advance(sim,
max_cycles)` runs the model until a host-visible event (a completion record, log words, the RISC-V halted or
trapped, a reset finished, the queues idle) or the budget ends. Ties are broken by (time, event kind, unit,
sequence), so a run is reproducible. A wait loop of the firmware is fast-forwarded to the next event and `WFI` jumps to
the next interrupt, so poll counts are not architectural. `TSR_SIM_SCHED` picks the schedule: `fixed` (default; op
durations from the cycle model, a 2-cycle issue crossing, no lag on the RISC-V-side copies of `DONE_SEQ`, credits,
`QSTATUS` and error records) or `random:<seed>[,<min>,<max>]` (every op duration is multiplied by a factor drawn from
`[min, max]`, default 0.25 to 4, crossings take 2 to 8 cycles, the RISC-V cost of each instruction varies by one cycle
and its copies lag the core).

A program is valid when it is race-free and, under every schedule, the per-unit op streams (words, flags, dependencies
as relative tickets), the completion records r0 to r6, the RISC-V's SPM stores and the final written DRAM and SPM
lines are the same (SW §4.7). The tests run the fixed schedule and seven random ones with the race detector fatal.
Poll counts, `MTIME`, `PERF`, record word r7 (duration), completion records (REC) and the log are never compared.

## Environment

`TSR_SIM_SCHED` as above. `TSR_SIM_RACE=off|report|fatal` is the race detector (`off` by default for model runs, and
the ctests that run firmware or queue programs set `fatal`). `TSR_SIM_RV_CORE=maxperf|small` is the Ibex cycle table
(timing only, CC-34) and `TSR_SIM_RV_RATIO` the ratio of the control-core clock to the core clock, 0.05 to 4 (default
166.7 / 276.3). `TSR_SIM_LOG=stderr|<file>` logs queue events and one line per `tsr_sim_op`, and with the prefix `rv,`
one line per retired RISC-V instruction and with `w,` the eight words of every push (`TSR_SIM_LOG=rv,stderr` is the
quickest way to see what the firmware does). `TSR_SIM_VU_TIMING=u200|k4|F,E,B,P` sets the VU lanes of the cycle model
(16 / 16 / 8 and no lane pipe by default, `k4` is the release image's 8 / 8 / 4 behind two lane-pipe stages; timing
only, SW-80). `TSR_TRACE=<dir>` dumps one trace per launch. `TSR_SIM_THREADS` (1 to 64) sets the MVM worker threads:
the library default is 1, the `sim` and `mock` transport kinds default to `min(online CPUs, 8)`, `tsr-numcheck` to 4,
and results do not depend on it. `TSR_REF_MATH_FULL=1` enables the exhaustive `ref_math_*_full` tests. The variables
that configure the simulated core behind the `sim` transport (`TSR_SIM_TMAX`, `_CH_GIB`, `_DRAM_DIR`, `_VU`, `_GDN`,
`_CARD`, `_ATT_LOCAL`, `_MVM_LOCAL_*`) are in [`sw/transport/README.md`](../transport/README.md). A value that does not
parse makes `tsr_sim_create_opts` fail with a message, and a field of `tsr_sim_opts` that is set wins over the
variable.

## Use and test

`TSR_DEVICE=sim` (the default) opens the simulator in-process behind the core CSR port, and `TSR_DEVICE=mock` puts it
inside a model of the `ap_ctrl_none` platform adapter with fault injection, so the host stack, the backend and the
tools see the same register protocol as on a card. `libtsr_ref.so` and `libtsr_sim.so` serve Python through `ctypes`
(`ref_math_np`, `ref_vu_np`, `sw/tools/numcheck.py`). `tsr-trace-replay`, `tsr-trace-gen` and `tsr-trace-conv` replay,
write and convert trace directories ([`traces/README.md`](../../traces/README.md)).

`ctest --test-dir build/dev -N` lists the tests. By prefix: `tables_*` (generated headers equal their source),
`ref_math_*` (the VU's math against double libm and an independent numpy model, bit for bit; `*_full`, label `slow`,
covers all 2^32 inputs), `ref_*` (layouts, f32 primitives, QA quantization against ggml, dot products, the VU, AT, GDN
and MoE-router references against numpy models and ggml-cpu), `sim_*` (CSR port, loader, rings, checker order,
dependencies, errors, every VU and AT op, schedules, the race detector, the debug models, trace capture and replay;
several with UBSan variants), `trace_v2_roundtrip`, and `rv_*` (label `rv`: the interpreter against hand-encoded
sequences, the GNU assembler, riscv-tests, riscv-arch-test and random programs, in lockstep against Spike). The
Spike-based tests need the pinned tools of `sw/ctrl/tools/install-toolchain.sh` and report "Skipped" without them,
`-DTSR_SPIKE=<path>` and `-DTSR_RISCV_TOOLS=<prefix>` point the harness (`sw/sim/tests/rvtests/`) at another
install, and `rv_random_long` needs `TSR_RV_RANDOM_LONG=1`.

Limits. Unit busy times follow simple formulas in `tsr_sim.c`, `tsr_vu.c` and `tsr_at.c`, and the RISC-V cost follows
a fixed table. Effects of an op are applied atomically at its modeled completion time, which is invisible for
race-free programs, and the race detector rejects the others. The debug-module model does not model the debug ROM
(DB-18). `VMASK` has no handler and completes `ILLEGAL_OP`. Ibex facts that are not checked against the vendored RTL
are marked *confirm* in `tsr_rv.c`.

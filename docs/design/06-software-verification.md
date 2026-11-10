# SW: software stack, firmware SDK, simulator and verification

Area SW owns everything that is not RTL: the ISA simulator (the bit-exact oracle), the C references, the
firmware SDK and runtime that run on the RISC-V control core, the firmware kernels and the graph executor, the
host stack (transport, ggml backend, `tsr_rt`, PyTorch, tools), the trace format, the version scheme and the
verification method. The decisions D1–D6, D12 and D14–D18 ([spec](../spec.md) §16) are fixed. Decisions SW-1 to
SW-88 are in §13. Numbers marked estimate are design figures.

---

## 1. Scope and summary

- Simulator. `sw/sim` holds an in-house RV32IMC + Zicsr interpreter (`tsr_rv`, with Spike as its oracle in
  tests only), an event-driven model of the issue queues on the core-cycle time base (fixed or random
  schedules), a vector-clock race detector, the VU and AT op semantics, and `tsr_ref_math.c`. The host starts
  work only through the launch queue and completion ring (D14).
- Trace format. One directory per launch: the firmware image (by id), the launch descriptor, DRAM init and
  expect regions, the DONE record, per-unit ticket records, and the pushed op stream in push order
  (`ops/push.bin`) with relative tickets. Tile-only traces replay through the standard `replay` kernel, so
  they also run on the card without host access to the issue port (D15).
- Firmware. `sw/ctrl` is a freestanding C++20 SDK: crt0, linker script, a runtime that implements EX's launch
  contract R1–R10 (EX §4), traps and panic per DB §6.3, binary logging per DB §6.1, a kernel table and the
  tile and VU header library. The base kernels are `graph` (the op-list executor), `selftest` and `replay`.
- Graph execution. The ggml backend turns each split into a pre-resolved op list in DRAM (§7.2) and launches
  `graph` once; what the list holds depends on the execution mode ([design/README.md](README.md) §2), and in full mode
  mode one decode token is one launch (§8). The host computes the dependencies between ops; the firmware maps them to tickets and
  issues ahead, so units overlap (D3).
- Host. EX's transport API (EX §10) with one protocol layer over every kdev: `sim` (in process), `mock`,
  `vkernel` and `xrt` (`ap_ctrl_none` with polling).
- Versions. Four numbers (capability, Basis version, Basis ABI, kind numerics revision) identify what runs
  (§6.9).
- Toolchain. The RISC-V compiler is the xPack `riscv-none-elf-gcc`; Spike and dtc are built from source; all in
  user space (§9.6).
- Pass criteria. The criteria of each execution mode ([design/README.md](README.md) §2) are in §10.4.

---

## 2. Host interface (D14, D16)

The host starts work only through EX's launch queue and completion ring (§6.4); the transport API is load,
launch and wait (§9.1). Tile ops are issued by firmware: their address fields are immediates and ra, rb, rc and
rd must be 0 ([spec](../spec.md) §11), tiling is in `plan.hpp` (shared by host and firmware), and op lists live
in the OPL region (§7.2).

---

## 3. Repository layout (software)

```
sw/
├── tables/                  # tsr_tables.py: the single table source (EX-30); gen_tables.py: the generator
├── sim/                     # libtsr_sim, libtsr_ref (C, host only)
│   ├── include/             # tsr_isa.h (+ generated tsr_tables.h), tsr_ref*.h, tsr_sim.h, tsr_rv.h, tsr_bus.h,
│   │                        # tsr_dm.h, tsr_dbg.h, tsr_race.h, tsr_trace.h
│   ├── tsr_ref.c            # references: fp helpers, layouts, QACT, dot products, dequant
│   ├── tsr_ref_vu.c, tsr_ref_at.c, tsr_ref_math.c, tsr_ref_gdn.c   # VU, AT, approximate-math, GDN references
│   ├── tsr_sim.c, tsr_q.c   # core state and host CSR port, loader, rings, tile units; issue port, dispatcher,
│   │                        # checker, queues, tickets, SYNC, errors, time base
│   ├── tsr_vu.c, tsr_at.c   # VU and AT unit semantics: checks, extents, execution, cycle models
│   ├── tsr_rv.c, tsr_bus.c  # RV32IMC+Zicsr interpreter; RISC-V address map and MMIO
│   ├── tsr_dm.c, tsr_dbg.c  # debug module model (DB §9.1's DMI profile); DB simulator models
│   ├── tsr_sched.c, tsr_race.c, tsr_trace.c   # schedules; race detector and SPM_STALE; trace read, write, replay
│   └── tests/
├── ctrl/                    # firmware SDK (RISC-V), separate CMake project (ExternalProject)
│   ├── cmake/, include/tsr/ # toolchain file, .note.tsr; kernel, mmio, vu, at, plan, opl, log, rt, exec headers
│   ├── rt/                  # crt0.S, link.ld, runtime.cpp, trap.cpp, log.cpp, libc.cpp
│   ├── kernels/             # graph.cpp (executor), k_*.cpp (kinds), selftest.cpp, replay.cpp
│   ├── host/, ref/, tests/  # tsr-elf and the host build of the kernels (§6.8); C++ references; fw_* ctests
│   └── tools/               # tsr-elf/, install-toolchain.sh (§9.6)
├── transport/               # tsr_transport (C API), protocol layer, kdevs: sim, mock, vkernel, xrt
├── ggml-tesseract/          # backend: weights, tsr-opl.* (op-list builder), tsr-build.* (lowering), tsr-kvbuf.*, tsr-msum.*
├── rt/, torch/              # tsr_rt on op lists; PyTorch above tsr_rt
└── tools/                   # tsr-selftest, tsr-trace-*, tsr-log, klcheck, numcheck, tsr_bw, tsr-dbg, tsr-prof, tsr-top, tsr-mtp
```

`third_party/` holds no RISC-V tool sources. The toolchains live outside the repository (§9.6), and their
versions are recorded in `docs/design/toolchain.lock`.

---

## 4. Simulator

### 4.1 Model overview

The simulator is one C library, linked into the backend, the tools and the tests. It models the core (core
clock domain: SPM, per-channel DRAM, the host CSR port of EX §5 with the LQ, LCQ and log rings, `Q_*`, PERF and
DMI, the dispatcher, the checker, the unit queues, and the units DMA, QACT, MVM and VU with AT's ops on them,
plus SYNC) and the control core (RISC-V clock domain: `tsr_rv`, ITCM and DTCM, the control window, the core
window, the SPM window and the debug module).

Functional results come only from the C references (`tsr_ref*.c`; [spec](../spec.md) §22, rule 3). An op's
effects (SPM and DRAM writes, scalar-register writes) are applied atomically at its modeled completion time.
For race-free programs this choice is invisible.

### 4.2 RISC-V model `tsr_rv` (SW-1)

- ISA: RV32IMC + Zicsr + Zifencei, M-mode only (CC's configuration). An unimplemented encoding or CSR raises an
  illegal-instruction trap (`mcause` 2, `mtval` = the instruction bits). `fence.i` flushes nothing (there is no
  cache); it costs what a jump costs. A misaligned load or store is split into two aligned accesses in address
  order, like Ibex, and is not atomic; if the second half faults, the trap reports the original address.
- CSRs: exactly CC's list for the pinned Ibex configuration: `mstatus`, `misa` (read only), `mie`, `mip`,
  `mtvec` (vectored only: MODE reads 1, base 256-byte aligned), `mscratch`, `mepc`, `mcause`, `mtval`,
  `mcycle[h]`, `minstret[h]`, `mcountinhibit`, `mhpmcounter3-31[h]` and `mhpmevent3-31` (read 0), `mvendorid`,
  `marchid`, `mimpid`, `mhartid`, `tselect`, `tdata1-3` (2 execute triggers), `dcsr`, `dpc`, `dscratch0-1`
  (debug mode only).
- Interrupts: `irq_fast[0..5]` and `irq_timer`, vectored at `mtvec` + 4 × cause. WFI wakes on
  (mip & mie) not 0 whatever `mstatus.MIE` is.
- Bus: the CC address map (CC §4.1). An unmapped access raises an access fault (instruction 1, load 5, store
  7). The register semantics are CC's and EX's (issue port, credits, `ISSUE_PIDX`, `DONE_SEQ`, error records,
  `QCTRL`, `QSTATUS`, `WAIT_*`, the LQ, LCQ and log registers of EX §4.5, `MTIME`), including the core-window
  rules (CC §5.3) and the SBA refusals (CC §4.3).
- Test bus profile (used only by `rv_isa`, `rv_arch`, `rv_random` and riscv-arch-test, never by firmware): RAM
  at 0x8000_0000 (size from the ELF), HTIF `tohost` and `fromhost`, signature dump. riscv-tests `-p` and
  riscv-arch-test link at 0x8000_0000 and end through HTIF, and Spike puts its debug module at 0x0 and its boot
  ROM at 0x1000, which overlaps CC's ITCM; the firmware profile is CC §4.1. Random lockstep programs are placed
  at 0x8000_0000 in both Spike and `tsr_rv`.
- Timing (schedule only, never architectural): two cycle tables, one per Ibex configuration of CC §3.2
  (`tsr_rv.h`: `TSR_RV_CPI_MAXPERF`, `TSR_RV_CPI_SMALL`), selected by `TSR_SIM_RV_CORE` = `maxperf` (default,
  CC-34) or `small`. Bus latencies are in `tsr_bus.h`: the control window 3 RV cycles; the core window and the
  SPM bridge cross into the core domain; a line-buffer hit 1. `tsr_sched.c` varies the CPI and the handshake
  crossings under random schedules. The RV-to-core clock ratio comes from `TSR_SIM_RV_RATIO`.
  `tsr_rv_get_stats` counts instructions and cycles per class for the perf harnesses.
- Debug hooks: halt, resume, step, abstract register access, program buffer and triggers come from `tsr_dm.c`,
  which implements area DB's module contract, so GDB over the DMI window works against `TSR_DEVICE=sim`.

### 4.3 Core model

`tsr_q.c` implements the issue rules of EX §7–§8 and CC §7–§9 exactly: credits, tickets {unit, seq 28 b},
`ISSUE_PIDX`, up to 2 deps, SYNC NOP, FENCE and QCLEAR, QABORT, hold sources (with DB's hard hold, CC §9.2),
checks at dispatch in EX's order (ILLEGAL_OP → BAD_DEP → BAD_FMT → K_RANGE → T_RANGE → OPERAND → SPM_RANGE →
DRAM_RANGE), poisoned entries, error state at the head, invariants I1–I3, and per-unit error records with push
indices. Queue depths, unit ids and the opcode to unit map come from the generated table and the config.

The unit code paths in `tsr_sim.c` (LD, ST, QACT, MVM and their checks) start when the queue head starts entry
e. VU and AT ops are in §4.5. The host CSR port serves EX's map (EX §5): `CTRL` and `STATUS`, the loader with
receive CRC and VERIFY, `RV_BOOT`, the LQ, LCQ and log rings and watermarks, `SOFT_RESET` (EX §8.5, CC §12.2),
PERF, `Q_*`, `SREG`, the DBG block (`tsr_dbg.c`) and DMI (`tsr_dm.c`). Reads have no side effects except EX's
PERF LO snapshot ([spec](../spec.md) §21, CSR reads while running).

### 4.4 Scheduler and time base (SW-2)

- Time is in core cycles (u64). The event kinds are unit-op completion, dispatcher crossing, RISC-V progress
  (RV cycles × ratio), the LQ, LCQ and log crossings, and the `MTIMECMP` timer. Ties are broken
  deterministically by (time, event-kind priority, unit id, sequence).
- `TSR_SIM_SCHED`: `fixed` (default): op durations come from the cycle model, the issue-FIFO crossing takes 2
  cycles, and the RISC-V-side synchronized copies (`DONE_SEQ`, credits, `QSTATUS`, error records, interrupt
  levels) show the core state without lag. `random:<seed>[,<min>,<max>]`: every op duration is multiplied by a
  factor drawn uniformly from [min, max] (default 0.25–4); crossings take 2–8 cycles; the RV CPI of each
  instruction varies by ±1 cycle. The generator is `xoshiro128**`, seeded from `<seed>`.
- Fast-forward. WFI jumps to the next event that sets a pending, enabled interrupt. A spin also jumps: the same
  timing-dependent register (`DONE_SEQ[u]`, `ISSUE_CREDIT`, `QSTATUS`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) read
  4 times in a row with an unchanged value and no store, push or other MMIO access in between. The hart then
  jumps to the next event that changes any of these registers. Poll counts are therefore not architectural,
  which the determinism rules already require (§4.7).
- Host interaction: `tsr_sim_advance()` runs until a host-visible event (an `LCQ_WR` change, log words, RV
  halted or trapped, `RV_RESET_DONE`, `SR_BUSY` falling, a `QIDLE` change) or until a cycle budget runs out.

### 4.5 Device op semantics and references

| Ops | Semantics source | Reference code | Simulator |
|---|---|---|---|
| LD, ST, QACT, MVM, NOP, FENCE | [spec](../spec.md) §11 and §21 | `tsr_ref.c` | register fields must be 0; END, LI, ADDI, CALL and opcodes 0x04–0x09 are ILLEGAL_OP |
| VOP (including GELU-tanh, GELU16, GEGLU16, TANH), RED (segmented, post RCP and RMS), ROPE (angle recurrence), TOPK, SOP, DQROW, GDN | VU §5–§6 and §7.12 (base: spec §9 and §11) | `tsr_ref_vu.c` (ops), `tsr_ref_math.c` (approximations), `tsr_ref_gdn.c` | `tsr_vu.c`. TOPK and DQROW exist in the simulator only; VMASK is not implemented |
| KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM | AT §5–§6 (vector ops executed by the VU, VU §5.11) | `tsr_ref_at.c` (AT §13.1) | `tsr_at.c` |
| QCLEAR, QABORT, hold control entries | area EX | `tsr_q.c` | `tsr_q.c` |

Cycle models (SW-80). The durations of the ops are estimates, not part of the bit-exact contract, but the RTL
benches time them and the card's trace writer reports them (DB §8.3), so the models follow what they show.
`tsr_vu.c` and `tsr_at.c` take the VU's lanes from `tsr_sim_opts.vu_timing` (`TSR_SIM_VU_TIMING`: `u200`, the
default, 16 / 16 / 8 lanes and no lane pipe; `k4`, the release image's 8 / 8 / 4 lanes behind two lane-pipe
stages; or `F,E,B,P`) and charge the cycles of an op that waits for the one before it as the RTL unit takes
them (VU §7.9, §7.13). QACT takes 16 cycles per 32-block of an A32 vector and 17 per 256-block of an A256 one,
plus 44 and 72. A store takes its lines plus 110 cycles (54 up to 8 lines) and an MVM starts 144 cycles before
its first weight beat counts. `TSR_SIM_CARD=k4full` sets the card configuration: `T_MAX` 4, no local-intermediate
and no GDN `CAPS` bits (`TSR_SIM_GDN=1` adds the GDN op of the release image), VU `k4`, `clk_rv` at half the core
clock. The unit tests
`sim_vu` and `sim_at` hold numbers the RTL unit produces at both lane configurations; `tb/vu/test_vu_timing.py`
times op shapes on the RTL unit and compares them with the model (opt-in), and `sw/tools/trw_vs_sim.py`
compares a card run with a simulator run op by op ([prof-dbg.md](../prof-dbg.md)). The model charges every op
as if it waited for the one before; the RTL overlaps independent ops (VU §7.8), so a stream of small
independent ops is overcharged.

`tsr_ref_math.c`. Area VU fixes each algorithm: range reduction, polynomial or table, the coefficients as f32
bit patterns, and the evaluation order. SW implements the file under these rules:

1. Every f32 operation is a call to `tsr_fadd`, `tsr_fsub`, `tsr_fmul` or `tsr_fdiv` (DAZ, FTZ, canonical NaN,
   [spec](../spec.md) §21, FTZ decision point). Everything else is integer bit manipulation. No libm, no
   `float` operators, built with `-ffp-contract=off`.
2. The API takes and returns `float` and is pure:

   ```c
   float tsr_m_expk(float v, unsigned kappa);   // EXPK(v, κ), VU §6.3 (κ = P1, N1, N2, NG)
   float tsr_m_exp(float x);                    // EXPK(x, P1)
   float tsr_m_sigmoid(float x);                // RC(FA(1, EXPK(x, N1)))
   float tsr_m_silu(float x);                   // FM(x, RC(FA(1, EXPK(x, N1)))), VU §6.5
   float tsr_m_gelu_tanh(float x);              // VU §6.5
   float tsr_m_gelu16(float x);                 // ggml-CPU-compatible fp16 form, VU §6.5
   float tsr_m_tanh(float x);
   void  tsr_m_sincos(float t, float *s, float *c);  float tsr_m_sin(float t);  float tsr_m_cos(float t);
   ```

   The names are VU's (VU §6.1). There is no RECIP_APPROX: every reciprocal is `tsr_frecip`.
3. The same file compiles for RV32 (soft-float libgcc is never needed, because the `tsr_f*` helpers are what
   get called) and for Python ctypes (cocotb).
4. Each function's verification is in §10.1 (SW-23).

### 4.6 Race detector and the launch-stateless rule (SW-3, SW-4)

- Happens-before: vector clocks over the agents {each unit, the RISC-V, the host}. The edges are queue order,
  dependencies, RISC-V program order, a RISC-V push after a completed store, the RISC-V observing a completion
  (a `DONE_SEQ` read, the WAIT interrupt), FENCE and QCLEAR, and the launch protocol (EX §2.3): a host DRAM
  write completed before `LQ_PUSH` of launch s happens before every op of s (H1); every op of s happens before
  the host's read of s's answer record and everything the host does after it (H2).
- Access sets per op come from its operands (VU and AT ops from their documents). RISC-V SPM accesses are
  tracked per byte. Issue-bank scalar registers are tracked as 16 locations.
- Launch-stateless rule (SW-4): a launch's observable results depend only on (image, descriptor args, DRAM
  contents). The detector reports `SPM_STALE` when an op or the RISC-V reads an SPM line that nothing wrote
  since the runtime popped the descriptor. Dependence on DTCM state left by an earlier launch has no mechanical
  check; it is caught by the trace round trip (§10.1): every captured launch trace is replayed on a freshly
  booted simulator and must match.
- `TSR_SIM_RACE=off|report|fatal`. ctest uses `fatal`. Model runs default to `off` (the detector costs a
  factor of 2–3).

### 4.7 Determinism (what is bit-exact)

A program (image + descriptor args + DRAM inputs) is valid if it is race-free and the following are the same
under every schedule: the per-unit op streams (words, push flags, deps as relative tickets); the LCQ records
(r0–r6; r7 DURATION is timing); the RISC-V's SPM store stream; the final written DRAM and written SPM lines.
The check runs at least 8 schedules (`fixed` + 7 random seeds) and compares all four. RTL and the card
reproduce the expect DRAM regions, the DONE record (r0–r6), the per-unit completion counts and the per-unit
error records. Never compared: poll counts, `MTIME`, `PERF_*`, record word r7, completion records (REC) and
log content.

Firmware rule (SW-30): no value read from a timing-dependent register (`DONE_SEQ`, credits, `QSTATUS`,
`MTIME`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) may influence pushed words, deps, SPM stores or records. Such
values may only end a wait loop. `ISSUE_PIDX` and `ISSUED_SEQ` are not timing-dependent (only the RISC-V's own
pushes move them). The error records (`ERR_CODE`, `ERR_SEQ`, `ERR_INFO`, `ERR_W0`, `ERR_PIDX`) are exact for
every entry covered by a done ticket (CC I2), so after a join they may decide RET and records
(`tsr::first_error`, `error`, `err_units`, the runtime's R7); `QSTATUS`'s error bits are a separately
synchronized level that I2 does not order with `DONE_SEQ`, so they may only end waits or make the
STOP_ON_ERROR decision (§7.3). One credit value is exact: under QHOLD, after `Q_DEPTH[u]` pushes to a unit
whose queue was empty when the hold took effect, none of which can complete without running, credit(u) is 0
(CC §7.3); `selftest` id 10 relies on it. This holds by construction for every kernel in §7.

At the llama.cpp level, the same image and the same inputs give bit-identical logits on `sim`, `mock`,
`vkernel` and `xrt`.

### 4.8 C API and the register table

The register and encoding table `sw/tables/tsr_tables.py` (plain Python data, no dependencies) is the single
source (EX-30). One generator (`sw/tables/gen_tables.py`) produces from it the RTL package
(`rtl/core/tsr_tables_pkg.sv`), the C header (`sw/sim/include/tsr_tables.h`, included by `tsr_isa.h`), the
firmware headers (`sw/ctrl/include/tsr/{mmio,vu,at}.hpp`) and the Python module for cocotb
(`tb/common/tables.py`), with `--check` in lint. EX, CC, DB, VU and AT own the table rows; SW owns the
generator.

The simulator API is `sw/sim/include/tsr_sim.h`: `tsr_sim_create(cfg, opts)` with `tsr_sim_opts` (`dram_dir`,
`threads`, `sched`, `rv_ratio`, `race`, `log`), `tsr_sim_csr_read` and `_write` (the EX §5 map: CSRs, rings,
`Q_*`, DBG, DMI), `tsr_sim_advance(max_cycles)` returning the host-visible event bits (`TSR_SIM_EV_LCQ`, `_LOG`,
`_HALT`, `_RESET`, `_QIDLE`, `_BUDGET`), `tsr_sim_dram`, `tsr_sim_spm`, `tsr_sim_now`, launch capture
(`tsr_sim_capture_launch`, `_take`), race reports, and the RISC-V debug view. The test hooks `tsr_sim_push`
(an `ISSUE_PUSH` returning the ticket), `tsr_sim_qctrl` and `tsr_sim_run_until_idle` are never reachable through
`tsr_transport` (D15, SW-19). `tsr_config` includes `itcm_bytes`, `dtcm_bytes`, `ctl_id` and the queue depths.

### 4.9 Environment (simulator and `sim` transport)

| Variable | Values | Default |
|---|---|---|
| `TSR_SIM_SCHED` | `fixed`, `random:<seed>[,<min>,<max>]` | `fixed` |
| `TSR_SIM_RV_RATIO` | F_RV / F_CORE, 0.05–4 | 0.5 |
| `TSR_SIM_RV_CORE` | `maxperf`, `small`: the Ibex cycle table (§4.2, CC §3.2; timing only) | `maxperf` |
| `TSR_SIM_VU_TIMING` | the VU's lanes in the cycle model: `u200` (16 / 16 / 8, no lane pipe), `k4` (8 / 8 / 4, two lane-pipe stages) or `F,E,B,P` (§4.5, SW-80; timing only; CFG3 reports the EXP and back lanes) | `u200` |
| `TSR_SIM_CARD` | `k4full`: the card configuration in one variable: `T_MAX` 4, none of the local-intermediate and GDN `CAPS` bits, `TSR_SIM_VU_TIMING=k4`, `TSR_SIM_RV_RATIO` 0.5; the single variables still override (`sim` transport) | off |
| `TSR_SIM_GDN` | 1: the image has the fused GDN op (`CAPS` bit 8, the release image); 0: the composed GATED_DELTA_NET path | the simulator's `CAPS` |
| `TSR_SIM_RACE` | `off`, `report`, `fatal` | `off` (ctest: `fatal`) |
| `TSR_SIM_LOG` | `stderr` or a file, optional prefixes `rv,` (RV retire lines) and `w,` (the eight words of every push): push, start and done events | off |
| `TSR_SIM_THREADS`, `TSR_SIM_DRAM_DIR`, `TSR_SIM_TMAX`, `TSR_SIM_CH_GIB` | MVM worker threads, file-backed DRAM, `T_MAX`, channel size | |
| `TSR_TRACE` | directory: one trace per launch (`<dir>/<seq 6 digits>/`) | off |

---

## 5. Trace format (SW-5)

### 5.1 Directory

```
<trace>/meta.json
<trace>/ops/push.bin                 # pushed entries in push order (5.2)
<trace>/init/ch<c>_<off>.bin         # DRAM before the launch
<trace>/expect/ch<c>_<off>.bin       # DRAM written by the launch
<trace>/expect/tickets.bin           # per-unit counts and error records (5.3)
<set>/images/<image id hex>.elf      # firmware images, shared by every trace of a set
```

`meta.json` (format version 2):

```json
{ "format": "tesseract-trace", "version": 2,
  "kind": "launch",                         // "launch": image + descriptor; "ops": tile-only push stream
  "config": { "num_ch":4, "beat_bits":512, "lanes":128, "t_max":8, "k_max":32768,
              "spm_bytes":4194304, "vl":16, "nsreg":16, "rd_outstanding":16,
              "ch_bytes":17179869184, "addr_beat_w":28, "caps":N, "ctl_id":N, "cfg3":N,
              "itcm_bytes":131072, "dtcm_bytes":65536,
              "q_depth":[8,4,4,16,0,4] },                              // units 0-5; unit 4 absent
  "image":  { "id":"<16 hex>", "file":"../images/<id>.elf" },          // kind launch; 64-bit id, EX §3.2
  "launch": { "kernel":"graph", "index":0, "prof":0,
              "args":[a0,a1,a2,a3,a4,a5,a6] },                         // kind launch; EX §4.3
  "ops":    { "file":"ops/push.bin", "count":N, "load":{"ch":0,"offset":B} },  // "load": where a
            // replayer places push.bin for the replay kernel (kind ops); chosen outside every region
  "done":   { "status":0, "ret":0, "err_code":0, "err_ticket_rel":4026531840,
              "err_pidx_rel":0, "err_info":0, "err_w0":0 },          // DONE record r0-r6, EX §4.4
  "init":   [ {"ch":0,"offset":0,"size":160,"file":"init/ch0_0000000000.bin"} ],
  "expect": [ ... ],
  "sched":  "fixed", "rv_ratio":0.603,
  "perf":   { "core_cycles":N, "rv_instret":N, "unit_busy":[..], "ch_rd_beats":[..] } }
```

`perf` is informational ([spec](../spec.md) §21, Performance counters).

### 5.2 `ops/push.bin`

The file is an array of 48-byte little-endian records, one per entry the RISC-V sent through the issue FIFO, in
push order:

```c
typedef struct tsr_trace_push {   // 48 bytes
    uint32_t w[8];     // kind 0: instruction words as pushed (masked words are 0)
                       // kind 1: w[0] = QCTRL value; kind 2: w[0] = TRACE_MARK value
    uint32_t dep[2];   // relative tickets {unit[31:28], rel_seq[27:0]}; unit 15 = none
    uint32_t push;     // the ISSUE_PUSH value (word mask [15:8], REC bit0); 0 for kinds 1, 2
    uint32_t kind;     // 0 op, 1 queue control (QCTRL), 2 trace mark
} tsr_trace_push;
```

`rel_seq` = (seq − base_u) mod 2^28. `base_u` is `ISSUED_SEQ[u]` at kernel entry (EX §4.7 R5; kind `launch`) or
when the replay kernel started (kind `ops`). The per-unit op stream compared by the determinism check (§4.7)
is the projection of `push.bin` onto one unit. A QCLEAR's ticket counts on the SYNC unit.

### 5.3 `expect/tickets.bin`

```c
struct { uint32_t n_units, reserved[3]; }                    // header, 16 bytes
struct tsr_trace_unit {                                      // n_units x 32 bytes
    uint32_t count;          // entries of this unit completed during the launch
    uint32_t err_code;       // 0 = no error record
    uint32_t err_rel_seq;    // ERR_SEQ - base_u
    uint32_t err_info, err_w0;
    uint32_t err_rel_pidx;   // ERR_PIDX - ISSUE_PIDX at kernel entry (EX §4.7 R5)
    uint32_t reserved[2];
};
```

### 5.4 Replay rules

- DRAM outside the init regions is zero ([spec](../spec.md) §21, Trace replay rules). The replayer loads the init
  regions.
- Kind `launch`: boot the image (HELLO), push the recorded descriptor, wait for its DONE record.
- Kind `ops`: write `push.bin` at `ops.load` and launch the standard `replay` kernel (§7.7) with (handle, count,
  flags 0; flags SERIAL for the converted model traces), plus BUF with a chunk buffer outside every SPM operand
  of the trace's ops. On the simulator this is `tsr_trace2_replay_opts.replay_image`;
  `tsr_trace2_replay_sim`'s default is the tile-only route through the issue port.
- Compared, everywhere: the DONE fields; every expect region, bit-exactly; the set of written DRAM beats, which
  must equal the union of the expect regions; the set of read DRAM, which must lie inside the init regions or
  have been written earlier in the launch; the per-unit completion counts (host `Q_DONE[u]` deltas, EX §5.4);
  the error records (on the card, the DONE record plus `Q_ERR_*`).
- Also compared where the push stream is observable (sim always; Verilator through an issue-FIFO monitor): the
  per-unit projections of `push.bin`.
- RTL tile-only replay: a Verilator bench pushes `push.bin` straight into the issue FIFO with no RISC-V (CC
  §15.3). It separates core concurrency bugs from RISC-V integration bugs.

### 5.5 Capture

`TSR_TRACE=<dir>` dumps one `launch` trace per launch (descriptor to answer record). The capture follows these
rules:

- init = DRAM read before it was written, copied at first read;
- expect = DRAM written, taken at the answer record;
- the image is written once per id into `<dir>/images/`;
- the launch's own LAUNCH_BEGIN and LAUNCH_END markers (DB §8.4: [27:16] its kernel index, [15:0] its SEQ) are
  stored with the SEQ field 0, launch-relative like the tickets. A launch captured at SEQ s therefore replays
  on a freshly booted simulator, where it runs as SEQ 0 (§10.1); every other marker value is stored as pushed.

`tsr-trace-gen` writes `ops` traces directly (random tile programs with dependencies, concurrency and error
injection). It runs each on the simulator to fill `done`, `expect` and `tickets`.

### 5.6 Trace sets (SW-6)

| Set | Content |
|---|---|
| `traces/m2` | `ops` traces from `tsr-trace-gen` (`regen.sh`, fixed seeds and configurations) with absolute immediate addresses; endings include ILLEGAL_OP, BAD_DEP, DEP_FAILED chains, ABORTED after an error state and QABORT and QCLEAR |
| `traces/gen` | the generator's LD, QACT, MVM, ST programs |
| `traces/m3` | MVM boundary cases and the MVM error kinds |
| `traces/vu`, `traces/at` | generator `--ops vu` and `--ops at` |
| `traces/ctrl` | concurrency, credits, errors, holds |
| `traces/model-v1conv` | recorded model traces (`traces/model`, on disk, not in git) converted with `tsr-trace-conv`. Each instruction becomes a push that depends on the previous push (a serial chain, replayed with the `replay` flag SERIAL), and init and expect are kept. This checks the units' results against the recorded ones |
| `traces/model-launch` | model launches captured with `TSR_TRACE`: one launch per split in matmul mode, one per token in full mode |

`tsr-trace-conv` converts only traces without register-relative operands; the m2, gen and m3 sets are
generated, not converted.

The generated sets are not in git: each `regen.sh` reproduces its set byte-identically from fixed seeds, and
`traces/MANIFEST.sha256` holds a digest per set.

Every generated trace obeys the launch-stateless rule (§4.6, SW-4) as one launch: each SPM line in the read
extent of an op that runs (its checker extents, so a strided operand includes its gap lines) is written by an
earlier op of the same trace. `tsr-trace-gen` checks this in its verification. Each `regen.sh` (its whole set)
and `tools_trace_gen_replay` (its small per-mode sets, plus a `--mutant-stale` trace that both checks must
flag) also replay through the `replay` kernel (§7.7) under the fixed and two random schedules with
`TSR_SIM_RACE=fatal`, where any other read is an `SPM_STALE` abort.

---

## 6. Firmware SDK (`sw/ctrl`)

### 6.1 Toolchain and build

- Compiler: the supported compiler is the pinned xPack `riscv-none-elf-gcc` (§9.6), chosen by
  `TSR_RISCV_PREFIX`. Other bare-metal GCCs work only because the SDK includes nothing from libstdc++: only
  `<stdint.h>`, `<stddef.h>`, `<stdarg.h>` and the SDK's own `tsr/traits.hpp`, with `libgcc` from the rv32im /
  ilp32 multilib. Clang 16 or newer (`--target=riscv32-unknown-elf`) is an optional second build.
- Flags: `-march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os -ffreestanding -fno-exceptions -fno-rtti
  -fno-threadsafe-statics -fno-use-cxa-atexit -ffp-contract=off -nostdlib -nostartfiles -g -std=c++20`, then
  `libgcc.a` only (integer helpers).
- Hot path (SW-34): `kernels/graph.cpp` and every `kernels/k_*.cpp` (the executor and the kind
  implementations, which run once per op record) are compiled with `-O2` after those flags
  (`TSR_FW_HOT_FLAGS` and `TSR_FW_HOT_GLOBS` in `sw/ctrl/CMakeLists.txt`, recorded as `firmware.cxxflags_hot` in
  `docs/design/toolchain.lock`); the runtime, `selftest`, `replay` and libc stay `-Os`. At `-Os` GCC divides by
  a constant with `divu` (38 cycles) where `-O2` multiplies.
- libgcc comes from `-print-libgcc-file-name` for the same `-march`, falling back to the `rv32im_zicsr/ilp32`
  multilib; code without C instructions runs on an RV32IMC core.
- Firmware float: none (SW-8). Every f32 constant (eps, 1/n, scales, the RoPE `theta_scale` and freq-factor
  tables) is computed by the host and passed as bits. If firmware ever needs spec fp math, it calls the
  `tsr_ref` helpers.
- CMake: the firmware is its own CMake project (`ExternalProject_Add(tsr_fw …)`), because one CMake project
  cannot have two C++ compilers. It produces `tsr-fw.elf` (with DWARF) and `tsr-fw.map`. Its path is compiled
  into the host libraries as the default image (`TSR_IMAGE` overrides, EX §10.2). The RISC-V toolchain is a
  required dependency of the build; configure fails with the install hint of §9.6.

### 6.2 Image format

EX §3.1–§3.2 are normative. Linker script memory map (CC §4.1): ITCM at 0x0000_0000, DTCM at 0x0010_0000,
sizes `ITCM_BYTES` and `DTCM_BYTES` from `RV_CFG` (U200: 128 KiB and 64 KiB).

| Section | Region | Loaded | Content |
|---|---|---|---|
| `.vectors` | ITCM 0x0 (`RV_BOOT` = 0) | yes | 32 vector slots × 4 B (`j` to handlers), reset entry at +0x80 |
| `.text` | ITCM | yes | code |
| `.tsr_kernels` | DTCM | yes | kernel table (below), sorted by index |
| `.tsr_kinds` | DTCM | yes | kind table (below, §6.9): the op list kinds the image provides with their numerics revisions |
| `.rodata`, `.data` | DTCM | yes | loaded directly by the host; crt0 copies nothing |
| `.tsr_id` | DTCM | yes | 8-byte image id (EX §3.2) |
| `.tsr_dbg` | DTCM | yes | DB's debugger flags (DB §6.3) |
| `.tsr_crash` | DTCM | no (zeroed by crt0) | DB's crash record (DB §6.3) |
| `.bss`, `.stack` | DTCM | no | zeroed by crt0; stack 8 KiB at the top of DTCM; 512 B trap stack |
| `.tsr_logfmt` | VMA 0xF000_0000 | no (non-allocated `INFO` section with contents, not in a `PT_LOAD`; GNU ld's `NOLOAD` would drop the strings) | log format strings (§6.6) |
| `.note.tsr` | | no | ELF note "TSR", type 1 (below) |

Kernel table entry (36 bytes; `static_assert`ed in `tsr/rt.hpp`):

```c
struct tsr_kernel_desc {
    uint32_t name_hash;    // FNV-1a 32 of name (tools only)
    uint32_t entry;        // int32_t (*)(const tsr_args *)
    uint8_t  nargs;        // 0..7 (descriptor ARG0..ARG6)
    uint8_t  flags;        // bit0 ARGBLK: ARG0 is a DRAM handle of an argument block (≤ 1 KiB)
    uint16_t argblk_lines; // lines loaded into the SPM argument range when ARGBLK (1..16)
    char     name[24];     // NUL-padded
};
```

`.note.tsr` descriptor (32 bytes, eight words; §6.9):

| Word | Field | Value |
|---|---|---|
| 0 | `sdk_abi` | the Basis ABI, `TSR_SDK_ABI` of `tsr/version.h` |
| 1 | `cap_min` | the Tesseract Core capability the image needs, `VERSION[15:0]` encoding (0x0100 = 1.0) |
| 2, 3 | `itcm_min`, `dtcm_min` | bytes (`link.ld` fills them at link time; the host compares them with `RV_CFG` KiB × 1024) |
| 4 | `caps_req` | `CAPS` bits the image needs |
| 5 | `n_kernels` | entries of `.tsr_kernels` |
| 6 | `basis_version` | the Tesseract Basis version, major << 16 \| minor << 8 \| patch (0x0001_0000 = 1.0.0); 0 = built before versioning (refused) |
| 7 | reserved | 0 |

Words 0, 1 and 6 come from the link's `--defsym` (`sw/ctrl/cmake/tsr_version.cmake`: `tsr/version.h` and
`VERSION_VALUE`). Kind table entry (4 bytes; `tsr/rt.hpp`, `TSR_KIND(NAME)` beside each kind's handler):
`{u8 kind, u8 rev = opl::kind_rev(kind), u16 0}`, unique kinds in any order; a kind declared twice in one image
is a link error.

Kernel order: the table is sorted by (rank, name), rank 0 `graph`, 1 `selftest`, 2 `replay` (EX §4.6's base
kernels at indices 0–2), 5 for every other kernel; `TSR_KERNEL` places each entry in section
`.tsr_kernels.<rank>.<name>` and `link.ld` sorts by name. Tools look kernels up by name.

Image id (SW-9, EX §3.2). `tsr-elf` computes it after linking: SHA-256 over every `PT_LOAD` segment in
ascending physical address, each as (u32 paddr LE, u32 padded filesz LE, padded bytes), with the 8 bytes of
`.tsr_id` taken as zero; the id is the first 8 digest bytes read as a little-endian `uint64`. `tsr-elf` writes
it into `.tsr_id`. The runtime echoes it in HELLO (r2 = bits [31:0], r3 = bits [63:32]). The host recomputes it
from the ELF and compares, so a HELLO with the right id proves that this exact image was loaded and booted; the
load CRC and VERIFY (EX §3.3) prove the transfer.

### 6.3 crt0 and the runtime (EX §4.7, normative)

crt0, at `RV_BOOT` + 0x80:

1. Set `sp`, `gp`, `mscratch` (trap stack) and `mtvec` (vectored, base 0), zero `.bss`, run `init_array`.
2. Initialize `__tsr_dbg` and `__tsr_crash` (DB §6.3), `RT_STATE` := BOOT.

The runtime then follows EX §4.7, with R4's validation before R3's coverage check: a descriptor that is both
invalid and covered by a watermark answers BAD_DESC, by EX §4.4's precedence (BAD_DESC above ABORTED):

- R1 Boot. Push `qclear()` and join it; `RT_STATE` := IDLE; post HELLO as an event (r2 and r3 = image id, r4 =
  [15:0] Basis ABI | [31:16] kind table signature, r5 = kernel count, r6 = Tesseract Basis version, r7 = boot
  core cycles; §6.9). Descriptors pushed before HELLO stay queued.
- R2 Take. WFI on `irq_fast[2]` (`LQ_AVAIL` ≠ 0 ∧ ¬ACTIVE); copy `LQ_HEAD_W0`–`7` and `LQ_HEAD_SEQ` into DTCM;
  write `LQ_POP`.
- R3 Canceled or stopped before start. Covered by `LQ_CANCEL` or `LQ_STOP` (EX §4.9): answer DONE / ABORTED
  without running.
- R4 Validate. Unknown TYPE, reserved bits set, KERNEL beyond the table, or a bad argument-block handle →
  DONE / BAD_DESC (args beyond the kernel's `nargs` are ignored). PING → PONG.
- R5 Run. `RT_STATE` := KERNEL | index; PROF → `REC_CFG.ALL` and NODE markers; stage the argument block (LD
  into the top 16 SPM lines, one join); `launch_pidx` = `ISSUE_PIDX`, `base_u` = `ISSUED_SEQ[u]`; DB's launch
  hook `tsr_rt_launch_hook(kernel, seq, args)` (DB §3.8; a0 = kernel index, a1 = SEQ, a2 = the `tsr_args`
  pointer at `__tsr_launch_halt`); TRACE_MARK LAUNCH_BEGIN; call the kernel; TRACE_MARK LAUNCH_END (DB §8.4).
- R6 Join. `join_all()`.
- R7 Result. First error = smallest `ERR_PIDX` over the units in error state; PLAT_ERR from `LQ_FLAGS`; status by
  EX §4.4's precedence. RET is the kernel's int32 return value.
- R8 Answer. WFI on `irq_fast[3]` while `LCQ_FREE` = 0; post DONE with ANSWER = 1.
- R9 Clear. If any unit is in error state, push `qclear()` and join it.
- R10. `RT_STATE` := IDLE; continue at R2. Debug builds assert `QSTATUS.QIDLE` here.

There is no mailbox, no STOP or BYE and no LOGCTL message: a soft stop is EX's `LQ_STOP` watermark (kernels
poll `tsr::stop_requested()`), and the log level is `LOG_CFG.LEVEL`. A trapped runtime parks (§6.6); the host
recovers with `tsr_transport_soft_reset` and a warm restart (EX §3.5).

### 6.4 Launch descriptors and answer records (EX §4.3–§4.4)

EX §4.3 (8-word descriptor: TYPE, PROF, KERNEL, ARG0–ARG6) and EX §4.4 (8-word records: DONE 0x01, PONG 0x02,
HELLO 0x80, TRAP 0x81, kernel events 0xC0–0xFF; launch STATUS 0–7 with precedence) are the only formats. SW
defines only the meaning of RET per kernel. For the base kernels:

| Kernel result | DONE STATUS (EX §4.4) | RET (r2) |
|---|---|---|
| success | 0 OK | 0 |
| a tile op failed | 1 TILE_ERR (r3–r6: ticket, `ERR_INFO`, `ERR_W0`, `ERR_PIDX` of the first error) | `graph`: index of the failing op record + 1 (1 … 4096); `replay`: relative push index of the first error + 1 |
| the kernel rejected its input (bad op list, scratch overflow, CRC) | 3 KERNEL_ERR | negative: `graph` returns 0x8000_0001 … 0x8000_0007 (§7.3) |
| canceled or stopped | 5 ABORTED, 7 STOPPED | kernel-defined |

`tsr::panic` and traps produce a TRAP record (type 0x81, STATUS 4), never a DONE (§6.6).

### 6.5 Kernel ABI

```cpp
struct tsr_args {
    uint32_t a[7];                 // descriptor ARG0..ARG6 (EX §4.3)
    const volatile uint8_t * blk;  // SPM-window pointer to the argument block, or nullptr
    uint32_t seq;                  // launch sequence number (LQ_HEAD_SEQ)
    uint32_t flags;                // bit0 PROF
};
#define TSR_KERNEL(name, nargs, flags, blk_lines) \
    extern "C" int32_t name(const tsr_args * args); /* + table entry in .tsr_kernels */
```

- Arguments are 32-bit integers, SPM line numbers or DRAM handles ({ch[31:28], beat[27:0]}).
- A kernel returns RET ≥ 0 for success (0 normally) or a negative code (KERNEL_ERR), and it must leave every op
  it pushed joinable (the runtime joins all, R6).
- Kernels never pop, never post answers, may post events (types 0xC0–0xFF), and never issue from interrupt
  handlers (`issue()` is not reentrant).
- Long kernels poll `tsr::stop_requested()` at safe points (EX §4.9).
- Kernels obey the launch-stateless rule (§4.6) and the timing rule (§4.7).

### 6.6 Traps, panic and logging (SW-11)

- Traps and panic: DB §6.3 is normative. The handler switches to the trap stack; with a debugger attached
  (`__tsr_dbg.attached`) it restores the registers and stops at `__tsr_trap_halt`; otherwise it fills
  `__tsr_crash`, logs a TRAP record (LEVEL FATAL, SOURCE 2), posts EX's TRAP record (type 0x81, ANSWER = 1 if
  ACTIVE: `mcause`, `mepc`, `mtval`, `RT_STATE`, `RT_INFO`, MTIME) and parks in a WFI loop with `RT_STATE` =
  TRAPPED. `ebreak` outside debug mode is a trap like any other.
- `tsr::panic(code, a0, a1, a2)`: enters the trap handler with `mcause` = 24 (custom) and `mtval` = code, after
  logging its arguments at level FATAL. It is used for violated internal invariants (for example a
  ring-allocator corruption), never for data errors, which return codes. GDB shows it as SIGABRT (DB §4.7).
- Logging: binary records through EX's log ring (EX §4.10), with DB's header (DB §6.1, the only definition). A
  FMT record carries the address of its format string in `.tsr_logfmt`, then the arguments.
  - `TSR_LOG(level, "fmt", args...)` stores the literal in `.tsr_logfmt` (not loaded, so it costs no TCM and
    does not enter the image id) and emits the record.
  - Supported conversions: `%d %u %x %08x %f` (argument = f32 bits), `%t` (ticket), 64-bit integers as two
    words, and `%s` for strings in loaded sections (the host reads them from the ELF).
  - Before writing, the runtime reads `LOG_FREE`. If the whole record does not fit, it drops the record and
    counts it in DTCM; the next record that fits is preceded by an EVENT record "dropped n". With
    `LOG_CFG.BLOCK` it waits instead. The ring therefore never holds partial records.
  - Interrupt handlers do not log. Compile-time level filter: `TSR_LOG_MAX_LEVEL` (default 3 INFO in release,
    4 DEBUG in debug builds). Runtime filter: `LOG_CFG.LEVEL` (the host sets it from `TSR_LOG_LEVEL`, 0–5).
  - The host decodes with `tsr-log` or the transport (`TSR_LOG=stderr|<file>`).

### 6.7 Tile and VU header library (`tsr/kernel.hpp` and friends)

`tsr/kernel.hpp` (namespace `tsr`) provides the types `spm`, `dram`, `handle`, `wref`, `qa`, `op`, `ticket` and
`deps` with `after()`; `issue(op, deps, flags)` (credit spin, then push; single hart, not reentrant); `done`,
`join`, `join_irq`, `join2` (a SYNC NOP over two tickets) and `join_all` (FENCE + join); `error(ticket)`,
`first_error()` and `qclear()`; `pidx_now()` (CC's `ISSUE_PIDX`: the next push index, exact);
`stop_requested()` (EX §4.9); `spm_ptr<T>(spm, byte_off)`; one constexpr builder per opcode and sub-function
(LD, ST, QACT, MVM, NOP, FENCE; the VU and AT builders are generated from `sw/tables/tsr_tables.py` into
`vu.hpp` and `at.hpp`, so their encodings track VU and AT); `group` (the latest ticket per unit, folded to at
most two deps or a `join2` chain); `spm_ring` (`init(first_line, lines)`; `alloc(lines, parity, &must_wait)`
returns a contiguous block that never wraps plus the deps that must precede any write to it, namely the
releases of older overlapping blocks folded to at most two, with older ones joined in software,
deterministically by count and never by `done()`; `release(block, after)`); and `TSR_LOG(level, fmt, ...)` and
`[[noreturn]] panic(code, a0, a1, a2)` (a trap with `mcause` 24; DB §6.1, §6.3).

`plan.hpp` is freestanding C++ (no STL, no heap). The host backend compiles it too (SW-20). It holds the
MUL_MAT tiling (`tsr_mm_plan_tiling` and `tsr_mm_lines_per_token`), the T-tile choice for an SPM budget,
`QA_LINES`, `ROW_BEATS`, and the per-kind scratch requirements of §7.5. The host therefore rejects in
`supports_op` exactly what the firmware cannot plan.

### 6.8 Host build of the kernels (SW-25)

The kernel sources also compile for the host against `backend_host.hpp`: `issue()` calls `tsr_sim_push`,
`done()` and `join()` advance the simulator, and `spm_ptr` maps to `tsr_sim_spm`. The host build runs the same
kernels in milliseconds under ASan and UBSan. It is a convenience. The reference is the RV binary on `tsr_rv`,
and ctest runs both builds (§10.2).

### 6.9 Versions (SW-54, normative)

Four public numbers: the Tesseract Core capability (hardware contract), the Tesseract Basis version (the
firmware SDK and the host side), the Basis ABI and each kind's numerics revision. [versioning.md](../versioning.md)
describes them for users.

| Number | Value | Source | Carried in |
|---|---|---|---|
| Tesseract Core capability MAJOR.MINOR | 1.0 | `sw/tables/tsr_tables.py` `ISA_MAJOR` / `ISA_MINOR` | `VERSION` (EX §7.5), `.note.tsr` `cap_min` (the image's need) |
| Tesseract Basis MAJOR.MINOR.PATCH (SemVer) | 1.0.0 | `sw/ctrl/include/tsr/version.h` | `.note.tsr` word 6 and HELLO r6, both from the link's `__tsr_basis_version` (major << 16 \| minor << 8 \| patch); `tsr_fw_image(... BASIS_VERSION <code>)` links a test image as another version |
| Basis ABI | 1 | `version.h` `TSR_SDK_ABI` (the only source) | `.note.tsr` word 0, HELLO r4[15:0] |
| Kind revision (u8, 0 = absent) | 1 for every kind of §7.2 except GATED_DELTA_NET, which is at 2 (§7.10, SW-78) | `tsr/opl.hpp` `kind_rev` | `.tsr_kinds`; HELLO r4[31:16] = the kind table signature |

- Capability. Bumped by the hardware owner (EX §7.5): major for an incompatible hardware-contract change (any
  tile op's result bits included), minor for an addition present in every configuration of that minor; an
  optional feature is a `CAPS` bit.
- Basis version. MAJOR.MINOR.PATCH, bumped at release with `CHANGELOG.md` and an annotated tag
  `basis-vX.Y.Z`. Major: a large change, an ABI break or a break of a host interface. Minor: a small addition
  or change of functions: new kinds, kernels, kind variants, ops, flags, host API or tools, a planner accepting
  more shapes, any kind revision bump. Patch: a small fix that leaves the inputs and outputs of every function
  unchanged: every output bit of every kind, every interface and every planner's acceptance.
- Basis ABI. Covers `tsr_args`, `TSR_KERNEL` and the kernel table entry, the `.note.tsr` layout and the
  `.tsr_kinds` entry, the launch protocol as the runtime implements it (descriptor, record and HELLO layouts,
  R1–R10), the op list format (header, record layout, every kind's argument words and their meaning, removing a
  kind) and a planner accepting less (SW-20). Bumped by the breaking change itself.
- Kind revision. Bumped in the change that can alter any output bit of the kind for the same record words,
  input bytes and configuration (`CFG0`–`CFG3`, `CAPS`) on the same capability; not for push order, scheduling
  or scratch use, nor for accepting new inputs (a Basis minor). A new kind starts at 1; revisions never
  decrease and are never reused. Procedure: change `opl::kind_rev`, run
  `test_fw_kind_revs tsr-fw.elf sw/ctrl/tests/kind_revs.txt --update`, add a line to the CHANGELOG. The
  `fw_kind_revs` test compares each kind's outputs on a fixed corpus with the recorded digest at that
  revision, so (capability, configuration, kind, revision) identifies a result. The corpus takes the decode
  fast entries that production decode runs (a retire hook counts the entries into the image's general-path
  and fast-path functions, and each launch states which it must take), so a numerics change confined to a fast
  entry changes a digest too. `--update` applies the table's rules first: it refuses a digest that moved at an
  unchanged revision, a bump without a change and a lowered revision, so it records only what a revision bump
  announced.
- Kind table. `TSR_KIND(NAME)` beside each handler (NOP and JOIN in `graph.cpp`) emits `{kind, rev, 0}` into
  `.tsr_kinds`; the signature is FNV-1a 32 over the 64 revision bytes (0 for an absent kind), folded to 16 bits
  (`opl::kinds_sig`). The runtime computes it at R1; the host computes it from the ELF.

Who refuses what:

| Check | Where | Rule |
|---|---|---|
| device identity | `tsr_transport_open`, reads only, in this order | `ADAPTER_VERSION` major 1 and `NUM_CH` (kinds with an adapter); `ID` = "TSR1"; `VERSION` major = 1 (any minor); `CTL_ID` ≠ 0 right after it |
| image | `tsr_image_open` | `.note.tsr` present; `sdk_abi` = `TSR_SDK_ABI`; `basis_version` ≠ 0; `.tsr_kinds` well formed |
| device vs image | `tsr_transport_boot`, before any load | same capability major, device minor ≥ `cap_min`'s; `CAPS` ⊇ `caps_req`; TCM sizes; the ABI |
| HELLO | every cold and warm boot | image id; r4[15:0] = the note's ABI; r4[31:16] = the signature of `.tsr_kinds`; r5 kernel count; r6 = the note's Basis version; no other record first |
| backend vs image | ggml-tesseract and `sw/rt` at open | the image's Basis (major, minor) ≥ the backend's (patch ignored; a newer image is fine). Not a transport rule, so captured traces of older images stay replayable |
| launch | supports_op; the builder | only kinds the booted image declares; the executor still answers BAD_RECORD for a kind it lacks and BAD_VERSION for an op list format it does not know |

Internal identifiers, never shown as product versions: `CTL_ID[7:0]` = 1 (control-core revision; 0 = no control
core), `ADAPTER_VERSION` major 1, `DBG_ID[7:0]` = 1 (DB §10.1), the kind table signature, the image id.
File-format versions: OPL header 1, trace meta 2, `.klc` 2, numcheck meta 2, DBG_INFO 1, CRASH 1, MSUM 1.
Display (`tsr_versions_str`): "Tesseract Core 1.0 (VU ATT GDN), Tesseract Basis 1.0.0 (ABI 1, N kinds)".

---

## 7. Kernels and the graph executor

### 7.1 Kernel set

| Kernel | Args | Purpose |
|---|---|---|
| `graph` | a0 OPL handle, a1 OPL bytes, a2 flags (bit0 VERIFY_CRC) | the op-list executor (§7.3) |
| `selftest` | a0 test id, a1 data handle, a2 data area lines, a3 seed, a4 flags (parameters in line 0 of the data area; layouts in `sw/ctrl/ref/selftest_abi.hpp`) | card and sim self-test programs (§7.6) |
| `replay` | a0 `push.bin` handle, a1 count, a2 flags (bit0 SERIAL: every op record depends on the previous op record instead of its recorded deps; bit1 BUF: a3 and a4 valid), a3 SPM line of the chunk buffers, a4 records per chunk (a multiple of 4, at most 1,364) | tile-only trace replay, SPM dumps (§7.7) |

These are EX's base kernels (EX §4.6). PING is part of the runtime (descriptor TYPE 0x02), not a kernel. MOE_ROUTE
and MUL_MAT_ID are op list kinds of `graph` (§7.9), not kernels.

### 7.2 Op list format (OPL, SW-12)

An op list is one 64-byte-aligned little-endian blob in device DRAM, normally in the OPL region (ch0, 0–64 MiB,
§9.3).

Header (64 bytes):

| Off | Field | Meaning |
|---|---|---|
| 0 | `magic` u32 | 0x4C50_4F54 ("TOPL") |
| 4 | `version` u16, `hdr_bytes` u16 | 1, 64 |
| 8 | `n_ops` u32 | number of op records, 1–4096 (SW-29) |
| 12 | `n_slots` u32 | 64-byte slots in the op area |
| 16 | `ops_off` u32 | byte offset of the op area (64) |
| 20 | `flags` u32 | bit0 REC (completion records on every push), bit1 MARK (TRACE_MARK with the op index before each op; pushed before the executor reads the op's record, so a record it then rejects has its marker too, §7.3), bit2 SERIAL (join after every op: debugging), bit3 STOP_ON_ERROR (stop pushing once an error is seen; the backend always sets it) |
| 24 | `n_pre` u32 | entries in the preload table |
| 28 | `pre_off` u32 | byte offset of the preload table |
| 32 | `pre_lines` u32 | SPM lines of the preload area |
| 36 | `graph_id` u32 | host hash, logged by the executor |
| 40 | `total_bytes` u32 | blob size |
| 44 | `crc32` u32 | CRC-32/ISO-HDLC of bytes [64, total_bytes), checked when the launch flag VERIFY_CRC is set |
| 48–63 | reserved | 0 |

Preload entry (16 bytes; SW-15): `src` (DRAM handle), `bytes`, `pre_byte` (byte offset inside the preload area,
64-aligned, assigned by the host), reserved (0). Preload entries carry the small graph inputs that firmware reads
with the RISC-V (KV write indices, GET_ROWS ids, each KQ mask's mask summary MSUM, AT §7.3) and the positions that
ROPE reads. They are loaded once at launch start with a single join, so no kernel joins on an input load inside the
op stream.

Op record: one or more 64-byte slots.

```
word0: [7:0] kind, [11:8] n_slots - 1, [15:12] 0, [23:16] kind flags, [31:24] n_args
word1: [15:0] dep0, [31:16] dep1      (op indices < own index; 0xFFFF = none)
word2..: args[n_args]                  (n_args <= 16 * n_slots - 2)
```

- An op with more than two predecessors is preceded by `JOIN` records, which the builder emits.
- All DRAM addresses are handles of 64-byte-aligned beats. Row strides are in lines.
- All f32 parameters are bit patterns computed by the host (§6.1).

Kinds and argument words. Matmul mode uses 0x01 and 0x10 (and 0x1F with MUL_MAT_ID); full mode uses every kind. The
Gated DeltaNet kinds (0x2D to 0x2F) are numbered from the top of the free range downward. Rev is the kind's
numerics revision at Tesseract Basis 1.0.0 (`opl::kind_rev`, §6.9). Shapes are flattened to rows: `rows` =
ne1·ne2·ne3, and every row of a tensor starts on a line (ensured by `supports_op`).

| Kind | Name | Rev | Args (word index: meaning) | ggml source |
|---|---|---|---|---|
| 0x00 | NOP | 1 | none (no tile ops; its ticket is "none") | placeholder |
| 0x01 | JOIN | 1 | none (SYNC NOP over dep0, dep1) | builder |
| 0x10 | MUL_MAT | 1 | 0 x, 1 x stride, 2 y, 3 y stride, 4 W_OFF (beats), 5 fmt (0 Q8_0, 1 Q4_K), 6 N, 7 K, 8 T, 9 addend handle (0xFFFF_FFFF none), 10 addend stride (0 = row 0 broadcast to every token), 11 flags (bit0 ACC_DST: y += W·x) | MUL_MAT; fused MUL_MAT+ADD(bias or residual) |
| 0x11 | ADD | 1 | 0 dst, 1 dst stride, 2 a, 3 a stride, 4 b, 5 b stride, 6 n (ne0), 7 rows, 8 b row period (b row = r mod p, p ≥ 1), 9 flags (bit0 b is one f32 scalar) | ADD |
| 0x12 | MUL | 1 | as ADD | MUL (norm weights, Gemma `out_scale`) |
| 0x13 | SCALE | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 s (f32), 7 b (f32) | SCALE (`x·s`, then `+b` if b ≠ 0) |
| 0x14 | RMS_NORM | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 eps (f32), 7 inv_n = `tsr_fdiv(1, n)` (f32), 8 weight handle (0xFFFF_FFFF none), 9 weight row period | RMS_NORM, fused RMS_NORM+MUL(weight) |
| 0x15 | GLU | 1 | 0 dst, 1 dst stride, 2 gate, 3 gate stride, 4 up, 5 up stride, 6 n, 7 rows, 8 variant (0 SWIGLU, 1 GEGLU16: ggml-CPU-compatible fp16 GELU, VU §6.5; 2 GEGLU: exact tanh form, for PyTorch) | GLU split form |
| 0x16 | ROPE | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 pos preload byte offset (I32 per token), 5 table handle (AUX f32 × n_dims/2: the freq-factor table, or the A table with mode bit1; 0xFFFF_FFFF none), 6 TS = `powf(freq_base, -2.0f/n_dims)` bits (host libm, as ggml-cpu), 7 M = attn_factor (f32), 8 freq_scale (f32; 1.0 = none), 9 head_dim, 10 n_dims, 11 n_heads, 12 n_tokens, 13 mode (bit0 NEOX; bit1 ATAB: YaRN form with a host-folded A table, VU §6.7) | ROPE: the angle recurrence of ggml-cpu (VU §6.7), bit-identical angles for ext_factor = 0 |
| 0x17 | GET_ROWS | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 ids preload byte offset (I32), 5 n_ids, 6 row lines | GET_ROWS on f32 activations (`inp_out_ids`) |
| 0x18 | SET_ROWS_KV | 1 | 0 cache base handle, 1 cache row stride (AT units), 2 src, 3 src stride, 4 idx preload byte offset (I64; the host checks idx < 2^31), 5 n_rows, 6 row elements, 7 which (0 K, 1 V), 8 plane base, 9 E/64 \| LG << 16, 10 plane kind, 11 stream (AT §12) | SET_ROWS into a KV buffer: QACT A32 + KVST per run of consecutive cells (AT §5.1) |
| 0x19 | FLASH_ATTN | 1 | 0 dst, 1 dst token stride, 2 q, 3 q head stride, 4 q token stride, 5 K cache base, 6 V cache base, 7 n_kv, 8 head_dim, 9 n_head, 10 n_head_kv, 11 n_tokens, 12 MSUM preload byte offset, 13 f32-mask handle (with a mask: always the copy for this record's scale, written only when needed, because the cached list outlives per-token mask changes and a rule-3 GENERAL tile reads it also when the MSUM is not GENERAL; 0xFFFF_FFFF only without a mask; SW-48), 14 scale (f32), 15 softcap (must be 0), 16 K plane base, 17 V plane base, 18 E/64 \| LG << 16, 19 D, 20 G, 21 flags (AT §12; two slots) | FLASH_ATTN_EXT with K/V in KV buffers; the device never reads the F16 mask |
| 0x1A | CPY | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 row lines, 5 rows | CPY/CONT f32 → f32 with contiguous rows |
| 0x1B | UNARY | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 func (0 TANH, 1 GELU16 (llama.cpp GELU), 2 SILU, 3 EXP, 4 GELU exact tanh form, 5 SIGMOID) | UNARY ops the VU has |
| 0x1C | MUL_MAT_SCALE | 1 | MUL_MAT words 0–11, requiring no addend, addend stride 0 and flags 0; word 12 FP32 scale | single-consumer MUL_MAT → SCALE |
| 0x1D | MUL_MAT_SOFTCAP | 1 | same restricted MUL_MAT words 0–11; words 12/13 exact FP32 pre/post scales | MUL_MAT → SCALE → TANH → SCALE |
| 0x1E | MOE_ROUTE | 1 | 0 logits, 1 logits stride (lines), 2 ids (first line of each row of the ARGSORT tensor), 3 ids stride (lines), 4 weights (packed f32, k per token), 5 n_expert (≤ 512), 6 n_used k (≤ 16), 7 T, 8 flags (0) | SOFT_MAX, ARGSORT + VIEW, GET_ROWS, SUM_ROWS, CLAMP, DIV of `build_moe_ffn` (§7.9, SW-61) |
| 0x1F | MUL_MAT_ID | 1 | 0 dst, 1 dst stride, 2 x, 3 x slot stride (0: one x row per token), 4 x token stride, 5 ids, 6 ids stride (bytes), 7 W_OFF of expert 0 (beats), 8 EXP_STRIDE (beats per channel), 9 fmt, 10 N, 11 K, 12 T, 13 n_used, 14 n_expert, 15 scale table handle (0xFFFF_FFFF none), 16 flags (0) | MUL_MAT_ID; with the per-expert scale chain REPEAT + GET_ROWS + MUL (§7.9, SW-62) |
| 0x20–0x2C | free (MOE_ROUTE and MUL_MAT_ID take 0x1E and 0x1F, the lowest free numbers; the Gated DeltaNet kinds take 0x2D–0x2F, numbered downward from the top of the free range) | none | | |
| 0x2D | SOFTPLUS | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows | UNARY SOFTPLUS (§7.10, SW-69) |
| 0x2E | SSM_CONV | 1 | 0 dst, 1 dst token stride, 2 x (token rows), 3 x token stride, 4 state in, 5 state out, 6 taps (d_conv rows of C), 7 C, 8 d_conv, 9 T, 10 n_seqs, 11 flags (bit0 SILU) | SSM_CONV with its CONCAT window, state CPY and SILU (§7.10, SW-68, SW-71) |
| 0x2F | GATED_DELTA_NET | 2 | 0 dst, 1 state out, 2 state in, 3 q, 4 q head stride, 5 q token stride, 6 k, 7 k head stride, 8 k token stride, 9 v, 10 v head stride, 11 v token stride, 12 g, 13 g token stride, 14 beta, 15 beta token stride, 16 S_v, 17 H_k, 18 H_v, 19 T, 20 n_seqs, 21 scale (f32) | GATED_DELTA_NET, scalar gate; K > 1 state snapshots are a chain of these records (§7.10, §7.11, SW-67, SW-83) |
| 0x30–0x3F | test kinds | none | | |

MUL_MAT_SCALE and MUL_MAT_SOFTCAP need the booted image's kind entry and `CAPS.MVM_LOCAL_SCALE` (bit 13) or
`CAPS.MVM_LOCAL_SOFTCAP` (bit 14). Firmware refuses an unsupported explicit record before it issues tiles, and
does not roll back earlier records. They keep MUL_MAT's DDR handles, full-N weight layout, strides and
dependencies, and publish no intermediate tensor. Without the capability the builder emits the separate sequence
before launch. The ggml builder fuses only addressable FP32 sole-consumer chains whose intermediates are not
observed, views or graph outputs, and whose memory accesses do not conflict. SCALE's additive constant must be
positive-zero bits, and SOFTCAP keeps both exact graph coefficients (the host never recomputes a reciprocal). The
release image has neither capability bit, so these kinds do not run on it.

Argument rules (`tsr/opl.hpp`). The words of each kind are a struct in word order (`opl::ew_args` for ADD and
MUL, `scale_args`, `rms_norm_args`, `glu_args`, `rope_args`, `get_rows_args`, `set_rows_kv_args`,
`flash_attn_args`, `cpy_args`, `unary_args`; indices in the namespaces `opl::ew`, `scl`, `rms`, `glu`, `rope`,
`getrows`, `setrows`, `fa`, `cpy`, `unary`), written by `writer::add_<kind>` and read by `record::args_as`. These
structural rules hold on every device and are checked by `opl::check_args` (the host reader) and by the kinds
(BAD_RECORD):

- ADD and MUL: flags bit0 (b is one f32 scalar) only; the b row period is at least 1 in every record (1 with a
  scalar b). RMS_NORM: a weight's row period is at least 1; without a weight the word is ignored.
- GLU variant at most 2, UNARY func at most 5, ROPE mode bits 0–1 only, SET_ROWS_KV which at most 1;
  FLASH_ATTN's softcap word is 0 (the bits: -0.0 is refused too).
- Every preload reference lies inside the list's preload area (`pre_lines` lines): ROPE 4·n_tokens bytes at a
  line-aligned offset (the VU reads the positions as an SPM operand), GET_ROWS 4·n_ids at a multiple of 4,
  SET_ROWS_KV 8·n_rows at a multiple of 8, FLASH_ATTN the MSUM image of n_tokens rows (32 + 16·n_tokens bytes) at
  a line-aligned offset. The writer's entries (`pre_positions`, `pre_ids`, `pre_kv_idx`, `pre_msum`) are
  64-aligned, so every alignment holds for a reference to an entry's start.

Word semantics the kernels fix (the backend's contract). Where the table leaves a choice:

- Every full-mode kind has kind flags (word0 [23:16]) 0. Empty ops (n, rows, n_ids or n_tokens 0) are BAD_RECORD, except
  SET_ROWS_KV with n_rows 0, which pushes nothing (ticket none).
- ADD and MUL with FLAG_B_SCALAR: b is an f32 vector of per-row scalars and row r uses word r mod p (p = b row
  period): p = 1 one scalar for every row, p at least rows one scalar per row (Gemma's `inp_scale_rows` [1, T]:
  the c-vector form of VU §5.2), 1 < p < rows period blocks. The b stride word is unused with the flag. Without
  the flag a b stride of 0 broadcasts b row 0.
- SCALE: y = FA(FM(x, s), b): VOP.MULC by s12 = s, then VOP.ADDC by s13 = b only when b's bits are not 0 (-0.0
  adds), the bits VOP.MULCADD would give with a b row of copies of b.
- RMS_NORM: the weight rows are dense, L(n) = ceil(n/16) lines apart (no stride word).
- ROPE: dst and src strides are lines between tokens; a token's heads are contiguous (H = n_heads · head_dim/16
  lines; ggml's [head_dim, n_head, T] views, also a fused-qkv view whose token stride is the qkv row); n_dims is at
  most head_dim (the unrotated tail is copied); mode bit1 (ATAB) needs a table (else BAD_RECORD) and makes the TS
  and freq_scale words unused; a table without ATAB is the freq-factor table (ROPE FF); freq_scale 1.0 sets no FS
  (FM(1.0, phi) = phi for every VU angle).
- GET_ROWS: an id that is negative or puts the row beyond 2^32 beats gets the LD beat 0xFFFF_FFFF, so the op fails
  with DRAM_RANGE (TILE_ERR on the op, never a panic); an id beyond its channel fails the same way at the DMA
  unit. The ids must lie inside the list's preload area (else BAD_RECORD).
- SET_ROWS_KV: word 10 (plane kind) is the KV format and must be 0 (q8_0); words 0, 1 and 11 (cache handle, row
  stride, stream) are informational and the plane base (word 8) is authoritative; row elements E are at most
  K_MAX; every index must have hi word 0 and a cell below 16·C·LG, else BAD_RECORD before any push. A cell
  written twice keeps the later row (DMA queue order).
- FLASH_ATTN: q head stride = D/16 (heads contiguous per token); dst token stride at least n_head·D/16 when T > 1;
  n_kv at most 2^20; E/64 in 1–511 (the 9-bit field); head_dim, n_head and n_head_kv agree with D, G and E; a
  GENERAL tile needs word 13 (the f32 mask copy) also when the MSUM is not GENERAL (AT §7.3); the flags word (word
  21): bit0 FA_NEW, bits [7:1] 0, bits [15:8] the preload line of the I64 new-cell indices, bits [31:16] their
  count (0 without FA_NEW). The backend points it at the layer's SET_ROWS_KV index entry (AT §6.6 rule 7, §12).

Shapes and device limits are the kinds' planners' (`tsr/plan.hpp`, section `M7 planning`: `tsr::plan_<kind>`, shared by
the firmware and the backend's supports_op, SW-20). `plan::dev_cfg` carries C (CFG0 NUM_CH), PCAP (CFG3), VL and
CAPS besides the MUL_MAT fields. A value of 0 means unknown, and a kind that needs it plans nothing (BAD_RECORD),
so a host that fills only the MUL_MAT fields keeps rejecting the other kinds.

### 7.3 Executor (`graph` kernel)

SPM layout (U200 defaults; L = `SPM_BYTES`/64 = 65,536 lines):

| Lines | Use |
|---|---|
| [L-16, L) | runtime argument range |
| [L-16-4096, L-16) | OPL area (256 KiB); a larger op list streams in two 128 KiB halves |
| below it, `pre_lines` (header, at most 256) | preload area |
| [0, rest) | scratch ring (`spm_ring`), about 3.7 MiB |

DTCM tables: `ticket[4096]` (the op's done ticket) and `pidx[4096]` (the op's first push index), 32 KiB in
total. The ticket table has one sentinel word in front (`TICKET_NONE`: a dep is looked up as entry dep + 1
with 0xFFFF wrapping to 0). The scratch ring's record table holds 128 blocks (2 KiB).

Algorithm:

1. LD the header and the op list (or the first half) into the OPL area and join it; LD the preload entries
   and join once more, so a list with preload entries costs two joins at launch start and none inside the op
   stream. Validate magic, version, `n_ops` at most 4096 and (if VERIFY_CRC) the CRC. A failure returns
   0x8000_000x (below).
2. For op i = 0 … n_ops-1:
   1. The op's control-window accesses first, because each one invalidates the SPM line buffer (CC §10.5):
      with MARK, `TRACE_MARK` NODE_BEGIN | i; `pidx[i]` = `pidx_now()` (`ISSUE_PIDX`: exact and
      timing-independent); `RT_INFO` = i. Then read the record through the line-buffered SPM alias, in place
      (records are not copied, except one whose lines wrap the streamed OPL area). Check it (BAD_RECORD,
      BAD_DEP). A streamed list pushes the next chunk's LD when the next record starts in a new chunk (before
      this record's pushes) and reads `pidx[i]` again.
   2. `d` = after(ticket[dep0], ticket[dep1]).
   3. Call the kind handler with (args, d, ring). `args` points at the record's argument words in place, and a
      handler reads the words it needs before its first push (a push invalidates the line buffer). It
      allocates scratch (its writes also wait for the ring's release deps), pushes its tile ops, and returns
      one ticket T_i whose completion implies every tile op of op i is done (I1, CC §7). A MUL_MAT record
      first goes to the decode fast path, a short full-mode record to its kind's fast entry (both below).
   4. `ticket[i]` = T_i. The handler releases its scratch blocks with `after` = T_i.
   5. Every 16 ops, and before every software join, read `QSTATUS` error bits. If an error is seen and
      STOP_ON_ERROR is set, stop pushing. This decision is timing-dependent but changes only which later ops
      are skipped; the result is an error either way.
3. Run `join_all()`.
4. Read `first_error()`. If it is nonzero, find the op i whose `pidx[i]` <= `ERR_PIDX` < `pidx[i+1]` and return
   i + 1. Otherwise return 0.

Executor return codes:

| Code | Meaning |
|---|---|
| 0 | success (DONE STATUS OK) |
| 1 … 4096 | the failing op is index ret-1 (DONE STATUS TILE_ERR, set by the runtime's R7 from the first error) |
| 0x8000_0001 | BAD_MAGIC |
| 0x8000_0002 | BAD_VERSION |
| 0x8000_0003 | TOO_MANY_OPS |
| 0x8000_0004 | BAD_RECORD (unknown kind, wrong n_args, slot overflow) |
| 0x8000_0005 | BAD_DEP (dep at or beyond its own index) |
| 0x8000_0006 | SCRATCH (the op cannot be planned in the ring) |
| 0x8000_0007 | CRC |

Codes from 0x8000_0000 up are negative as int32, so the runtime answers DONE STATUS KERNEL_ERR with them in RET
(§6.4).

Overlap (D3). The executor runs ahead of the units, limited only by credits and the ring. Independent ops
overlap on different units: Q, K and V projections, gate and up, and a ST of one op beside the MVM of the next.

Hot path (SW-35). The per-record RISC-V cost sets how far the executor runs ahead (§7.8). The rules:

- Records are read in place. The kind table in .rodata holds per kind the handler, word0's expected n_args and
  reserved bits and the minimum slot count, so the record checks are two compares. Variants of the loop
  (`op_loop`) exist for in-place and streamed lists, and without MARK and SERIAL. With MARK (a PROF launch)
  the record's NODE_BEGIN marker comes before the record is read, so a malformed record has its marker too.
- MUL_MAT's decode fast path (`k_mulmat.hpp`, `mulmat_t1_fast`): for T = 1 a leaf function reads its argument
  words in place, plans with `plan.hpp`'s `mm_plan` (shifts only, no divide), allocates with
  `spm_ring::alloc_dma_first` (the ring's release tickets of the blocks it reuses are earlier ops' STs, which
  the DMA queue order implies for the first LD: no deps and no software join), takes one `ISSUE_CREDIT` read
  for the op's 4–5 pushes when the per-unit fields and the issue-FIFO bound cover them, and pushes exactly the
  ops, words and deps of `kind_mulmat`. Every other case (T not 1, a record `kind_mulmat` refuses, a ring
  wrap, reused blocks released by another unit) returns before changing anything, and `kind_mulmat` runs it.
- `spm_ring` (§6.7) is O(1) amortized (a power-of-two table of 4-word records, 128 blocks): the executor can
  run further ahead before a full table makes `alloc` join its oldest block.
- `exec_state` (`opl.hpp`, symbol `__tsr_exec_state`) counts the MUL_MAT records that left the fast path and
  the ring's software joins; a launch rejected before step 2 reports 0 (nothing carries over, SW-4).

A fast path never changes what is pushed: the tile-op order, operand words, deps and scratch placement of
every record are fixed by the kind. The tests pin it: `fw_m6_e2e` and `fw_exec_rv` compare digests of their
captured launch traces (push records, DONE, expect regions; `tests/fw_trace_digest.h`) with recorded ones;
`fw_kernels_host` runs T = 1 records through `mulmat_t1_fast` and through the general path on twin simulators
and compares every push; `fw_hdr_host` compares the ring's blocks, deps and joined tickets with the reference
algorithm; `fw_exec_rv`, `fw_m6_e2e` and `fw_m6_perf` assert PERF PUSH_STALL = 0 (CC §7.3). A deliberate
change of these pushes is a decision: record it here and update the digests. Planning stays in the firmware
(SW-20).

Full-mode kinds in the executor (SW-36, SW-37, SW-39):

- Kind table. Kinds 0x11–0x1B have handlers (`exec.hpp`), argument counts and, for the ten short kinds (all but
  FLASH_ATTN), a fast entry `kind_<name>_fast`; `zz_kind_stubs.cpp` defines each weakly (a handler as BAD_RECORD
  "not in this image", a fast entry as "not taken") and a kernel source's strong definitions replace them. A
  wrong n_args is the record check's BAD_RECORD, before any handler.
- Dispatch (SW-37). A full-mode record goes to its fast entry first, with the record's argument words in place, its
  kind flags and d from the loop's registers: a leaf in its common path, which takes the record and returns
  its ticket, or returns KIND_NOT_FAST having pushed and changed nothing. The executor then calls the kind's
  handler out of line. The handler begins with the same fast entry, so both push the same ops. FLASH_ATTN and
  the test kinds keep the direct handler call.
- The one-tile block (SW-36). Every single-tile record of a DMA-bracketed kind uses one fixed SPM block below
  the preload area (`plan.hpp` `one_block_lines`: 1,024 lines on the U200) instead of a ring block: ADD, MUL,
  SCALE, RMS_NORM, GLU, ROPE and UNARY (their only tile), GET_ROWS and CPY (every copy tile) and SET_ROWS_KV
  (every run). A tile is DMA-bracketed when its first write to the block is an LD, its last read a DMA op (ST,
  KVST) and every other op on it lies between them by deps. The DMA unit runs its queue one op after another
  (CC §8.3), so the next such tile's first LD starts after this tile's last DMA op: the lines need no dep, no
  ring record and no release, and no concurrency is lost. MUL_MAT and FLASH_ATTN keep the ring, whose planning
  budget `ring_budget` is 60,144 lines on the U200. The block size is a trade: the budget sets MUL_MAT's
  prefill T-tile, and 1,024 lines keep the T-tile of the common output heads. A GLU row that does not fit the
  block takes a ring block. `plan.hpp` pins the T-tiles of the test shapes by `static_assert`.
- s-register shadow. `kind_ctx::sreg_valid` and `sreg_val` record, per launch, the s registers whose last
  writing push was an `SOP.MOVI` and its value; `sreg_movi` pushes a MOVI only when the shadow does not
  already hold the value (RMS_NORM's 1/n and eps, ROPE's TS and M, SCALE's s and b repeat across records). A
  push that writes s[i] otherwise (RED with TO_S, SOP other than STS) clears bit i (`sreg_clobber`) in the
  same record. The shadow starts empty at every launch (SW-4) and depends only on the pushes (§4.7).
  Registers: RMS_NORM s1 (RED TO_S result), s2 = 1/n, s3 = eps; ROPE s4 = M, s5 = TS, s6 = freq_scale; SCALE
  and scalar-b ADD and MUL s12, s13.
- Staged-word reuse (SW-39). A fast path's push skips the `ISSUE_W` stores of words the previous push of the
  same record staged with the same value (CC §5.1 keeps staged words; WMASK is unchanged, so the pushed words
  are exactly the op's). The SKIP claims are static per call site, and host builds store every word and check
  each claim.

DTCM state is rebuilt at every launch (SW-4).

### 7.4 MUL_MAT kernel

The kernel follows the lowering of [spec](../spec.md) §12, so its numerics match the recorded model traces bit for
bit (`traces/model-v1conv` checks this). Every trace of that set has T_MAX 1, so the push-stream comparison there
covers single-tile ops only; for T > T_MAX the kernel pushes the LD of tile t+1 before the ST of tile t (the double
buffering below). Per T tile (`plan.hpp`, T_MAX and the ring budget):

1. LD x (tile).
2. If an addend is present, LD the addend rows into the y block (the bias row repeated when its stride is 0).
3. QACT (A32 for Q8_0, A256 for Q4_K).
4. MVM with ACC when there is an addend or ACC_DST.
5. ST y.

Fusing a bias or residual through ACC gives y = addend + W·x with one f32 addition. f32 addition is commutative, so
this is bit-identical to ggml's separate ADD on the device. Tiles are double-buffered in the ring: the LD of tile
t+1 waits only for the QACT that read its buffer.

### 7.5 Full-mode kernels

Kernel sources: `sw/ctrl/kernels/k_elem.cpp` (ADD, MUL, SCALE, UNARY and the element-wise tile engine),
`k_glu.cpp`, `k_rows.cpp` (GET_ROWS, CPY), `k_norm.cpp` (RMS_NORM), `k_rope.cpp`, `k_kv.cpp` (SET_ROWS_KV),
`k_fattn.cpp` (FLASH_ATTN); planners `tsr/plan_elem.hpp`, `plan_norm.hpp`, `plan_attn.hpp`; references
`sw/ctrl/ref/elem_ref.cpp`, `rmsnorm_ref.cpp`, `rope_ref.cpp`, `kv_ref.cpp`, `fattn_ref.cpp`. Each kernel's header
comment states its tile-op sequence; the table summarizes them.

| Kind | Tile composition (primitives from VU/AT) | Scratch (lines, per tile) | Numerics reference |
|---|---|---|---|
| ADD/MUL/SCALE | per row tile (one at decode shapes; else MUL_MAT's double-buffered tiles, §7.4): LD the a rows, LD the b part (b rows, b row 0 for a broadcast, the p rows of a period, or the scalars' line(s)); VOP ADD / MUL in place (b stride L, or 0 for a broadcast; one VOP per period block, VU §9); with FLAG_B_SCALAR VOP ADDC / MULC with `cmode` 2 (the scalars as a c vector: one row, per-row scalars such as Gemma's `inp_scale_rows`) or `cmode` 1 (one scalar for several rows: SOP.LDS s12 from the loaded line); SCALE: SOP.MOVI s12 = s (s13 = b when b ≠ 0) through the s-register shadow, VOP.MULC c = s12, then VOP.ADDC c = s13 (the bits of MULCADD with a b row of copies of b); ST | rows·L + the b part (the VOPs are in place); a single tile in the one-tile block (§7.3) | `elem_ref.cpp` (`tsr_ref_vop` per row) |
| RMS_NORM (+weight) | per row tile, block [x rows \| w \| per-row vector]: LD x, LD w (one row for period 1, the tile's rows for p > 1; tiles never cross a period block), `SOP.MOVI` s2 = 1/n and s3 = eps through the shadow, `RED.SUMSQ` (post RMS, sk = s2, se = s3; after the tile's last LD) → s1 (one row) or the per-row vector, `VOP.MULCB` in place (`cmode` 1 / 2, b = w with stride 0 for period 1), or `VOP.MULC` without weight; ST. Per-head norms (Gemma q / k / v norms) are the same ops with rows = n_head·T and n = head_dim (VU §8.1; one record per tensor: VU §8.1's fused q / k / v form has one weight and no OPL record) | rows·L + w + ⌈rows/16⌉ | `rmsnorm_ref.cpp` (same order); bit-identical to the spec's 7-op lowering `docs/spec.md` §9 (VU §6.2) |
| GLU | LD gate rows, LD up rows, VOP SWIGLU / GEGLU16 / GEGLU in place on the gate rows, ST | 2·rows·L | `elem_ref.cpp` (`tsr_m_silu`, `tsr_m_gelu16`, `tsr_m_gelu_tanh`) |
| ROPE | per token tile (one at decode; else multiples of 16 tokens so the positions start on a preload line): LD x (H = n_heads·head_dim/16 lines per token), LD the freq-factor (or A) table (per tile), `SOP.MOVI` TS, M and freq_scale through the shadow (freq_scale only when ≠ 1.0), ROPE over the tile's tokens in place (positions: the I32 words in the preload area, read by the VU as an SPM operand), ST | tokens·H + table | `rope_ref.cpp` (`tsr_ref_vu` ROPE, VU §6.7) |
| GET_ROWS | per tile: one LD per run of consecutive ids when the source rows are dense, else per id (the RISC-V reads the ids from the preload area), one ST for dense dst rows; a row beyond half the budget moves in chunks | tile rows·row lines; the one-tile block when it fits | copy (`elem_ref.cpp`) |
| SET_ROWS_KV | per run of consecutive cells (index order, at most 64 rows): LD the source rows, QACT A32 (K = E, at most T_MAX rows per QACT), KVST of the run (AT §5.1) | run·(E/16 + QA_LINES(A32, E)); the one-tile block when it fits | `kv_ref.cpp` (`tsr_qact` + `tsr_ref_kvst` per row) |
| FLASH_ATTN | AT §6.6 (normative; reconciled with `tsr_ref_fa`: query tiles, the head subsets inside each chunk, the shared chunk planner `sw/sim/include/tsr_fa_plan.h`): per query tile LD Q, QACT per token; per chunk ATTQK per head subset (FAST) or ATTQK + AMASK (GENERAL), ASMAX, ATTPV per head subset (LSUM), AMERGE, with ATTPV / AMERGE issued one chunk behind ATTQK / ASMAX; ANORM; ST. Buffers S[3], P[2], MK[4] and, for GENERAL chunks, the mask rows M[2] (AT §7.1; the mask LD into M[k mod 2] waits for the AMASK of chunk k − 2), every op ≤ 2 deps | one ring block per op (`plan_attn.hpp` `fa_plan`: 18,948 lines for Qwen2.5-32B decode) | `fattn_ref.cpp` (`tsr_ref_fa`, AT §13.1) |
| CPY | LD rows, ST rows (one run when dense on both sides); rows beyond half the budget in chunks | block; the one-tile block when it fits | copy (`elem_ref.cpp`) |
| UNARY | LD, VOP func in place (TANH, GELU16, SILU, EXP, GELU, SIGMOID), ST | rows·L | `elem_ref.cpp` (`tsr_ref_math`) |

The numerics of a composite kernel are defined by its tile-op sequence. A host C++ reference in `sw/ctrl/ref/`
composes the same `tsr_ref` calls in the same order, and the kernel tests compare the firmware against it
bit-exactly. Changing a sequence changes logits, so the change is a kind revision (§6.9) with new klcheck
baselines.

### 7.6 `selftest` kernel (SW-28)

| Id | Test |
|---|---|
| 1 | LD/ST round trips on every channel: 1, 63, 64, 65 and 4,096 lines, cross-channel copies |
| 2 | QACT A32/A256 on host-written data |
| 3 | MVM Q8_0 and Q4_K: T = 1, T = T_MAX, ACC |
| 4 | each VU function and reduction |
| 5 | each AT op |
| 6 | concurrency: a pipeline of independent matrix-vector ops on all units |
| 7 | errors: DRAM_RANGE, ILLEGAL_OP, BAD_DEP, DEP_FAILED chain, QABORT then QCLEAR recovery |
| 8 | SPM window byte/half/word, ordering of a store before a push |
| 9 | log records and PING |
| 10 | credits: fill every queue under QHOLD; each credit must read 0 (exact under the hold, §4.7) |
| 11 | GDN: four heads of 20 × 128 (two out of place, two in place) with LD, `SOP MOVI` of alpha and beta, the op and ST; runs only when `CAPS` bit 8 is set, otherwise the kernel returns RET CAPS (VU §7.12) |

`tsr-selftest` runs each test on the device and, in process, on the simulator with the same image, inputs and
args. It compares DONE, every output byte and the per-unit completion counts.

### 7.7 `replay` kernel

1. LD chunk 0 (and chunk 1) of `push.bin` into the buffers, then read `base_u` = `ISSUED_SEQ[u]` for every unit
   and `pidx0` = `ISSUE_PIDX`, so that no push of the kernel lies between the replayed records of a stream of up
   to two chunks.
2. Stream `push.bin` through SPM in double-buffered chunks (at most 1,364 records = 1,023 lines; without BUF the
   two buffers sit just below the runtime argument range). The kernel's own later chunk LDs are added back when
   DMA deps are mapped.
3. For each record:
   - kind 0: issue the words with the push value and deps mapped back as (u, base_u + rel_seq); with flag SERIAL,
     the deps are replaced by the ticket of the previous kind-0 record (a dependency chain: with I1 this gives
     strictly sequential execution);
   - kind 1: write QCTRL;
   - kind 2: write TRACE_MARK.
4. Run `join_all()`. RET = 0 if no unit is in error state, else 1 + (`ERR_PIDX` of the first error - `pidx0` - the
   kernel's own chunk LDs pushed before it), i.e. relative to the first replayed record; the runtime reports the
   error record itself in DONE r3–r6 (EX §4.6). Negative RET (KERNEL_ERR): 0x8000_0101 BAD_ARGS, 0x8000_0102
   TOO_LONG (more than 130 chunks), 0x8000_0103 STREAM (a chunk LD failed).

### 7.8 Performance targets (estimates)

These are design targets. The test harnesses `fw_exec_perf`, `fw_m6_perf`, `fw_exec_perf_m7`, `fw_kernels_*_perf`,
`fw_kvfa_perf` and `fw_m7_layer_perf` measure them on the simulator with `TSR_SIM_SCHED=fixed` and report "work",
the RISC-V's busy cycles up to the launch's last op push without the iterations of wait loops after their first
test (`fw_prof.h`).

- Executor: at most 400 RISC-V cycles of work per graph op on average, on the CC §3.2 maxperf configuration with
  deep queues. For a small decode layer (about 25 ops per layer against about 230 µs of MVM time) the RISC-V then
  stays ahead of the MVM, which never idles on issue. The design requirement is the issue rate, not the cycle
  count.
- Short full-mode kinds: at most 350 RISC-V cycles of work per decode record for ADD, MUL, SCALE (also with a bias),
  RMS_NORM, GLU, ROPE, UNARY, GET_ROWS, CPY and SET_ROWS_KV at the decode shapes, checked by `fw_exec_perf_m7`
  (`--strict` asserts it for every kind). Records outside those shapes may exceed it: per-head norms with more
  than one row, ROPE with a freq-factor table or freq_scale other than 1, and records whose s-register values
  change (each MOVI is one push more).
- Issue loss: against a near-ideal issuer (F_RV = 4 · F_CORE), `fw_m7_layer_perf` reports the cost of the
  RISC-V's speed for one whole decoder layer. It comes from the DMA queue depth (8 entries fill with the decode
  chain's LD and ST ops waiting behind an ST that waits for its MVM) and from FLASH_ATTN's own planning time on
  the critical path between V and the output projection.
- Launch: EX §4.11 (empty-launch round trip at most 10 µs; descriptor push to kernel entry at most 5 µs; answer
  record to host at most one poll interval).
- Image size: ITCM at most 96 KiB of 128 KiB (SW-75); DTCM at most 56 KiB of 64 KiB including the 8 KiB stack,
  the 512 B trap stack and the 32 KiB executor tables. The budget applies to the production image
  (`tsr-fw.elf`; `fw_elf_image`, `fw_elf_test_image`); test images (production plus test kernels) are stamped
  against the whole ITCM. `tsr-elf size` reports the sections.

### 7.9 MoE kinds (SW-60 to SW-66, SW-76, SW-77)

Sources: `sw/ctrl/kernels/k_moe.cpp`, planners `tsr/plan_moe.hpp`, references `sw/ctrl/ref/moe_ref.cpp` and
`tsr_ref_moe_route` (`sw/sim/tsr_ref_vu.c`), backend `tsr-build.{h,cpp}` and `ggml-tesseract.cpp`. The spec
side is [spec](../spec.md) §4 (expert tensors) and §13.

The graph. llama.cpp's `build_moe_ffn` emits per layer: the router MUL_MAT (`ffn_gate_inp`, F32 in the GGUF),
SOFT_MAX, ARGSORT (descending) with a VIEW of the first k, GET_ROWS of the probabilities by those numbers,
SUM_ROWS, CLAMP, DIV, the expert input norm, MUL_MAT_ID for gate and up (one tensor `ffn_gate_up_exps` or two),
the GLU, MUL_MAT_ID for down, optionally RESHAPE + REPEAT + GET_ROWS of `ffn_down_exps.scale` by the same
numbers and a MUL, the MUL by the normalized weights, and the VIEWs and ADDs that sum the experts. Models
differ in the router input (some put RMS_NORM, SCALE and MUL by `ffn_gate_inp.scale` in front), in the expert
tensors (one gate-up tensor with a GEGLU of two views, or two tensors with a SWIGLU) and in the per-expert
output scale (without it the down MUL_MAT_ID is lowered alone). `hparams.expert_weights_scale` is 0 for the
Qwen3 MoE family, so no SCALE node follows the DIV. Nothing is patched in llama.cpp (SW-77).

SW-60 Expert tensors (EXP_STRIDE). A 3D tensor `[K, N, n_expert]` whose name contains `_exps` is placed as a
weight with `n_expert` matrices of N × K: expert e at `W_OFF + e × EXP_STRIDE` on every channel, with
`EXP_STRIDE` = `R × ROW_BEATS` beats ([spec](../spec.md) §4); one expert MVM uses all four channels, as a
dense MUL_MAT does. `tsr_store_expert_rows` and `tsr_load_expert_rows` scatter and gather per expert. The
conversion at load is the dense one (Q5_0 and Q6_K to Q8_0), a split upload stages the tensor and stores it
when complete, and an ANY-buffer source converts lazily with the expert count in the cache key. Other 3D
tensors stay raw. The router is an ordinary 2D weight converted to Q8_0 ([spec](../spec.md) §5).

SW-61 MOE_ROUTE (0x1E, 9 words `LOGITS, LOGITS_STRIDE, IDS, IDS_STRIDE, W, N_EXPERT, N_USED, T, FLAGS`;
`opl::moer`). Per tile of 64 tokens (one ring block, `plan_moe.hpp`):

1. LD the logits rows; the softmax on the VU (RED.MAX to a vector, VOP.EXPSUBC with that vector as c, RED.SUM
   with post-op RCP, VOP.MULC); join.
2. The RISC-V selects per token the k largest probabilities by integer compares on the words and writes the
   selected values to one line and the numbers to another. RED.SUM with post-op RCP over the k values and
   VOP.MULC give the weights; join.
3. For T > 1 the RISC-V packs the weights (k words per token). The STs write the numbers into the first line
   of each row of the IDS tensor (rows `IDS_STRIDE` lines apart, words k to 15 of the line are 0) and the
   weights, packed, into W (the DIV result `[k, T]`).

The arithmetic is `tsr_ref_moe_route` (VU §6.2 trees, EXPK, RC): `m` = max, `e` = EXPSUB(l, m), `r` = RC(tree
sum of e), `p` = FM(e, r); `(id, q)` = the top k of `p`; `r2` = RC(tree sum of the k values q in selection
order), `w` = FM(q, r2). ggml's graph does the same with `expf`, a double sum and a true division, so the
device differs at rounding level and its weights sum to 1 within 1e-6. The CLAMP is not computed: the k largest
of n probabilities sum to at least k/n, so the clamp is the identity for every row without a NaN (the builder
requires `clamp_min × n <= k`), and a NaN passes it.

SW-63 Tie rule and NaN. The selection is the order of the VU TOPK (VU §6.8): the DAZ'd value under the total
order (-inf < … < +inf < NaN), larger first, equal values by the smaller index; every NaN is the canonical NaN,
ranks first, and ties by the smaller index. The firmware insertion loop (`select_topk`) is checked against
`tsr_ref_topk` (`fw_kernels_moe_host`). ggml's CPU sorts the indices with `std::sort`, so equal values come out
in an unspecified order and NaN breaks the strict weak ordering; the device's order is the defined one.

SW-62 MUL_MAT_ID (0x1F, 17 words `DST, DST_STRIDE, X, X_SLOT_STRIDE, X_TOK_STRIDE, IDS, IDS_STRIDE (bytes), W_OFF,
EXP_STRIDE, FMT, N, K, T, N_USED, N_EXPERT, SCALE, FLAGS`; `opl::mmid`). One MUL_MAT of T = 1 per (token, slot)
against `W_OFF + id × EXP_STRIDE`. The x rows are shared by the slots (`X_SLOT_STRIDE` 0: gate-up) or one per
slot (down); y row (token, slot) is row `token × N_USED + slot` of the dense dst. With a `SCALE` table (F32
`[n_expert]`) the y block is multiplied by `table[id]` (VOP.MULC with an s register) before its ST. The handler
LDs a chunk of the ids rows (up to 128 lines) and the scale table into the one-tile block, joins, and reads the
numbers with the RISC-V. Per token it takes one ring block `[x | QA | y_0 … y_(k-1)]` and pushes LD, QACT and
MVM of slot j + 1 before the ST of slot j (MUL_MAT's double buffering). An id outside [0, N_EXPERT) gets W_OFF
0xFFFF_FFFF: the MVM fails with DRAM_RANGE and the launch ends with TILE_ERR on this op. The returned ticket
is the last ST. The numerics are the dense MUL_MAT's (`mulmat_ref_token`), then `tsr_ref_vop_elem(MULC)` with
the scale.

Joins. Both kinds wait inside the op stream, which the executor contract forbids to other kinds
(`tsr/exec.hpp`): MOE_ROUTE twice per tile of 64 tokens, MUL_MAT_ID once per chunk of ids. The wait is on the
handler's own ops; records already pushed keep running and only the issue of later records stops.

SW-64 Backend: fusion of the router and the scale chain. supports_op takes MUL_MAT_ID and, with the kinds in
the image, the nodes of the two chains by their local shape. `tsr_build_fusion` finds the groups node for node
with the use counts of llama.cpp's graph. The SOFT_MAX heads six nodes that become one MOE_ROUTE record (ids
into the ARGSORT tensor, weights into the DIV tensor). The MUL_MAT_ID heads REPEAT, GET_ROWS and MUL: its
record carries the scale table, writes its own tensor, and a CPY record moves the result into the MUL's tensor
(no record when the allocator put both at the same address). A group member that did not fuse aborts at
graph_compute naming the node: a graph other than llama.cpp's `build_moe_ffn` is not supported. `TSR_KINDS`
without MOE_ROUTE leaves the router chain on the CPU; without MUL_MAT_ID the experts stay on the CPU.

SW-65 Matmul and full modes. Matmul mode (`TSR_KINDS=m6,+MUL_MAT_ID`; klcheck and numcheck `--phase m6` set it): the device
runs the matrix products (the router MUL_MAT, both MUL_MAT_IDs per layer); llama.cpp's own CPU ops do the
softmax, the top-k, the renormalization, the GLU, the scale, the weights and the sums, so the CPU's selection
is the reference. Full mode: the whole layer is part of the one launch per token. The selection is software on the
RISC-V in both modes: nothing needs the TOPK RTL (`CAPS` bit 3), which the release image leaves off.

SW-66 Prefill. The kinds run an MVM of T = 1 per (token, slot), so a prefill of T tokens reads T × k experts
of every layer once each, with no reuse. A grouping by expert (a counting sort of the (token, slot) pairs per
ids chunk, one MVM of up to T_MAX gathered rows per expert) pays when T × k / n_expert is at least T_MAX and
does not change the numerics; it is not built.

SW-76 Outputs of a route with several tiles. A route of more than 64 tokens does not write the ids and the
weights where the graph has them: the MOE_ROUTE record writes them to a scratch window of the device (`T`
lines of ids, then `ceil(T k / 16)` lines of packed weights), and two CPY records after it move them into the
ARGSORT and DIV tensors. The record reads the logits tile by tile, and ggml-alloc may put the DIV or ARGSORT
tensor over logits that a later tile still has to read; the CPY records run after the record is done, so no
placement of the two tensors can hurt. The scratch window is taken at the first such route from the top of IO,
grown when needed and shared by all layers (the OPL builder orders the next route's record after the CPYs that
read it). A route of up to 64 tokens is one tile with its outputs in place. The record's words and numerics
do not change, so there is no kind revision. Without CPY (`TSR_KINDS`) the router of a graph of more than 64
tokens stays on the CPU. Test: `ggml_tesseract_moe` (`run_route_layouts` in `test-tsr-moe.cpp`).

Limits. n_expert at most 512 for MOE_ROUTE and a multiple of 16 (the ids rows are whole lines); k at most 16;
at most 4,096 experts per tensor; N a multiple of 16; K a multiple of the format's block; softmax gating with
a normalized top-k only (no bias, group selection, sigmoid or softmax-after-top-k: those graphs run the
router on the CPU).

### 7.10 Gated DeltaNet and the Qwen3.5 / 3.8 layer graph (SW-67 to SW-74, SW-78, SW-79, SW-82)

The three kinds SOFTPLUS, SSM_CONV and GATED_DELTA_NET, their planners, references, the backend lowering and
the tests cover a whole decode graph of the family in one launch per token. The kinds compose existing VU
operations, so they run on any image that has the VU (`CAPS` bit 0). GATED_DELTA_NET has a second, fused path
for an image whose VU has the GDN op (`CAPS` bit 8, VU §7.12): numerics revision 2, chosen at run time
(SW-78). The release image has the op. No llama.cpp patch is needed (SW-73).

The models. llama.cpp calls the family `qwen35`. It is hybrid: layer `il` is full attention when
`(il + 1) % full_attention_interval == 0` (interval 4), otherwise Gated DeltaNet (GDN). The sizes that matter
are E (hidden), S_v (value head size, 128), H_k and H_v (key and value heads), `d_conv` (4) and the conv
channels `C = 2 · H_k · S_v + H_v · S_v`. The attention layers use D 256, a joint Q and gate projection,
per-head q and k norms, `ROPE` in IMROPE mode, `FLASH_ATTN_EXT` and a SIGMOID gate on the attention output.
IMROPE carries four position streams; for text all three section streams equal the token position (the
fourth is 0), which makes IMROPE the NEOX rotation over `n_dims` (SW-72). `graph_compute` checks the streams
of the position input and stops with a message if they differ (vision positions).

The GDN layer graph (one token; nodes in graph order) and its device records:

| Step | ggml nodes | Device |
|---|---|---|
| norm, projections | `RMS_NORM` + `MUL`; `MUL_MAT` qkv (C), z, alpha, beta | RMS_NORM with weight (SW-42), MUL_MAT |
| conv state | `GET_ROWS` of `cache_r_l`, `CONCAT` with the transposed qkv, `SSM_CONV`, `SILU`, a `CPY` of the last `d_conv - 1` columns | CPY ×2 at the CONCAT (the window), one SSM_CONV (SILU and the state CPY folded in), SW-71 |
| q, k | views of the conv output, `RMS_NORM` and `SCALE` by 1 / sqrt(S_v) | RMS_NORM, SCALE |
| gates | `ADD` dt bias, `SOFTPLUS`, `MUL` ssm_a to g; `SIGMOID` of beta | ADD, SOFTPLUS, MUL, UNARY SIGMOID |
| state | `GET_ROWS` of `cache_s_l`, `SCALE` by 0 in place for a new sequence | GET_ROWS, SCALE |
| recurrence | `GATED_DELTA_NET` (scalar gate, K = 1), a `CPY` of the new state into `cache_s_l` | GATED_DELTA_NET, CPY (rows regrouped) |
| output | `RMS_NORM` + `MUL`, `SILU` of z, `MUL`, `MUL_MAT` ssm_out, residual `ADD` | RMS_NORM, UNARY SILU, MUL, MUL_MAT |

State layouts. `cache_s_l` is `H_v` states per cell, state `h` being `S_v` rows of `S_v` f32, row `j` holding
`S[.][j]` (the ggml-cpu layout; one row is `S_v / 16` lines). The recurrence of `ggml_gated_delta_net`, per
token and head, is `S *= exp(g); delta_j = (v_j - S_j · k) β; S_j += k delta_j; o_j = (S_j · q) / sqrt(S_v)` (q
and k head `h mod H_k`). `cache_r_l` holds `d_conv - 1` rows of `C` f32 per cell (time-major rows of channel
vectors), not ggml's channel-major order. The caches are in a Tesseract buffer (a device window, SW-72) and only
the device reads them, so the layout is the device's; llama.cpp copies and clears whole cells, which are byte
copies. The conv taps (`ssm_conv1d.weight`, `[d_conv, C]` tap-fastest in the GGUF) are re-laid out as `d_conv`
rows of `C` f32 when the weight is uploaded.

Kinds (record words in `tsr/opl.hpp`, planners in `tsr/plan_gdn.hpp`, handlers in `kernels/gdn_kinds.cpp`,
references in `ref/gdn_ref.cpp`). SOFTPLUS and SSM_CONV are at numerics revision 1, GATED_DELTA_NET at revision 2
(`kind_revs.txt`, SW-78).

| Kind | Id | Words | Tile composition | Numerics |
|---|---|---|---|---|
| SOFTPLUS | 0x2D | dst, dst stride, src, src stride, n, rows | row tile of 5 buffers; 20 VOPs (SW-69) | `softplus_ref` |
| SSM_CONV | 0x2E | dst, dst token stride, x, x token stride, state in, state out, taps, C, d_conv, T, n_seqs, flags (bit 0 SILU) | per sequence and channel tile: LD taps, state rows and token rows; `d_conv` VOP.MUL, `d_conv - 1` VOP.ADD in tap order; optional VOP.SILU; ST the T output rows and the new state rows (SW-68) | `sconv_ref` |
| GATED_DELTA_NET | 0x2F | dst, state out, state in, q / k / v with head and token strides, g and beta with token strides, S_v, H_k, H_v, T, n_seqs, scale | per (sequence, head): LD the state tile, then per token 13 VU ops (below) or, on the fused path, 5 (SW-78); ST the state tile (SW-67, SW-70) | `gdn_ref` (`gdn_ref_step` composed, `gdn_ref_step_fused`) |

GATED_DELTA_NET per head and token on the composed path: `SOP.EXP` alpha = exp(g); `VOP.MULC` S = S alpha;
`VOP.MUL` P = S · k (broadcast of k over the rows) and `RED.SUM` u (one value per row; the fixed 128-partial
tree of VU §6.2); `VOP.SUB` and `VOP.MULC` delta = (v - u) β; `VOP.MULCADD` S = k δ + S (δ_j a per-row scalar);
`VOP.MUL` P = S · q and `RED.SUM` w; `VOP.MULC` o = w · scale. Heads run in order with the next head's state
loading while the VU works on the current one; the next token's vectors load two tokens ahead (SW-70). One
ring block holds two state tiles, the product tile and two token slots (3 · S_v · L + 2 (5 L + 2) lines,
L = S_v / 16; 3,156 lines for S_v = 128; S_v up to 256 fits). The result tensor is
`[S_v H_v, T n_seqs + S_v n_seqs]` rows: the attention rows, then the new states.

The fused path (SW-78, numerics revision 2). When the device has the GDN op (`CAPS` bit 8 with the VU) and S_v
is 64 or more (the op's range of `m` is 64 to 256, VU §7.12.4), the handler runs one `GDN` op per head and
token in place of the 13 VU ops. The record, the planner and the ring block are the same, so the host plans
one way for every device. The choice is `plan::gdn::use_fused(caps, S_v)` in `tsr/plan_gdn.hpp`, read from the
device's `CAPS` at each launch; the firmware, the references and the tests use that function and nothing else.
Per head and token the fused step pushes `SOP.LDS` g, `SOP.EXP` alpha = exp(g), `SOP.LDS` β, `GDN` (d = a = the
state tile, in place; k, q, v, o in the token slot; m = rows = S_v), `VOP.MULC` o = o' · scale. S_v of 16 and 32
take the composed path on every device.

What revision 2 changes, for a record with S_v at least 64 on a device with `CAPS` bit 8 (everything else
produces revision 1's bits):

- The two dot products per row, S_j · k and S_j · q, are exact integer sums in a block-floating-point frame,
  rounded once to f32 (VU §7.12.3, VU-29), not `RED.SUM` trees of products rounded to f32: the vector (k scaled
  by αβ, or q) is a table of 24-bit entries relative to its largest exponent (round half up), and the row's
  elements are 26-bit windows below the row's largest exponent (truncation, shifts of 26 or more give 0).
- α is not applied to the state before the first dot product. The op uses u' = αβ (S_j · k) (k scaled by
  FM(α, β) first), e_j = FS(FM(β, v_j), u'_j), S'_ji = FA(FM(α, S_ji), FM(e_j, k_i)) and
  o_j = FA(FM(α, w_j), FM(e_j, κ)) with w_j = S_j · q on the old state and κ = k · q. It is the same recurrence
  mathematically; the rounding points differ.
- A row with Inf or NaN in its elements, or a k or q with one, gives canonical NaN for that row's e, S' and o
  (the head's, for k or q).
- The scale is applied to the output vector after the op, as on the composed path (`o = FM(o', scale)`).

The oracle `gdn_ref` takes the `CAPS` of the device it models and calls `tsr_ref_gdn` (VU §7.12,
`sw/sim/tsr_ref_gdn.c`) for the fused path. `fw_gdn_ggml` checks both paths against float64 and ggml-cpu up to
512 tokens: the error of either path equals the CPU's own.

SW-82 The state on its cache cell. The pinned graph passes the recurrent state through DRAM three times per
layer and token: `build_rs` gathers the layer's state rows by the ids of `rs_s_copy` (`GET_ROWS`), the
recurrence reads the copy and writes the new state into its own result, and a `CPY` moves it into the
sequence's cache cell. While a sequence keeps its cell (`ids[j] = kv_head + j`: every decode token, every
prefill ubatch after the first, and the first one when the zero-state cell `rs_z` is the sequence's own cell)
the gather and the copy change nothing the graph can see, and the record reads the cell and writes the new
state back into it (`STATE_IN` = `STATE_OUT` = the cell; in-place state is a case of `fw_gdn_rv`, for both
paths). `tsr_build_fusion` (`tsr_gdn_state_group`) recognizes the group at the `GATED_DELTA_NET` node: the
state operand is `RESHAPE` / `VIEW` over `GET_ROWS(RESHAPE(cache), ids)` with one id per sequence, the new
state's only reader is one `CPY` into rows of the same cache, nothing else reads the gathered copy, and no node
between the gather and the `CPY` touches the cache rows the `CPY` writes. The backend reads the ids of the
launch and absorbs the gather and the `CPY` only if they are the identity; the plan cache keeps the ids'
answer and the backend reads them again at every reuse (`tsr_plan_state_holds`), so another answer rebuilds
the plan with the three passes. `TSR_GDN_CELL=0` turns the group off; the results are bit-identical with and
without it. `TSR_STATS` prints how many records ran on their cell.

SW-69 Softplus. The ggml node is `softplus(x) = log(1 + exp(x))`; the CPU computes `x > 20 ? x : logf(1 + expf(x))`,
which loses the relative accuracy of small results. `SOFTPLUS` computes the stable form
`max(x, 0) + log1p(exp(-|x|))`. The VU has no log: `log1p(h)` for `h = exp(-|x|)` in (0, 1] is `2 atanh(s)` with
`s = h / (2 + h)` in (0, 1/3], by the odd series to `z^5` (`z = s²`, coefficients `2 / (2k + 1)`), 20 VOPs for
any row length. Against float64 the error is below 5 ulp for results above 1e-3 and below 4.2e-7 relative for
every result above 1e-30.

Not supported: more than one sequence in a ubatch end to end (the kinds and the lowering take `n_seqs` > 1 and
the kind tests cover it; no llama.cpp run does), vision positions and a KDA gate. Snapshots and rollback are
§7.11.

### 7.11 Multi-token prediction: the verification batch on the device (SW-83, SW-85, SW-86 and SW-88)

The verification batch, the draft head and the rollback run on the device with no RTL, firmware, record or ABI
change and no llama.cpp patch: everything is host lowering of the graphs the pinned llama.cpp builds.

What the pinned llama.cpp does. `llama_context_params.n_rs_seq = K` keeps `1 + K` groups of rows in every
recurrent cache (`mem_size · (1 + K)` rows, group `g` at rows `g · mem_size …`). A ubatch writes the state after
each of its last `min(T, K + 1)` tokens into the groups, most recent first (group `s` is the state `s` tokens
back): `ggml_gated_delta_net(…, K + 1)` returns the attention rows followed by `K + 1` state snapshots and one
`CPY` of the `min(T, K + 1)` snapshots goes into a 3D view of the cache; in `build_conv_state` one `CPY` per
slot copies the window of the `CONCAT` that starts `s` tokens earlier into the group `s`.
`llama_memory_seq_rm(seq, p0, -1)` sets `rs_idx = pos - (p0 - 1)` and moves the position back; the next
graph's `rs_s_copy` then names the snapshot cell, so the gather reads the snapshot. The MTP drafter
(`--spec-type draft-mtp`) is three decodes per step: the draft context runs K single-token steps (an `eh_proj`
of the concatenated normed embedding and hidden state, one full-attention layer with its FFN, the output head),
the target decodes `[id_last, d_0 … d_{K-1}]` with every row an output, and the draft context then catches up
on the same `K + 1` tokens. The MTP block is in the target GGUF or in a second model (`-md`).

SW-83 Snapshots by chains of K = 1 records. A record has one `STATE_OUT`, so a node with `nw = min(T, K + 1)`
slots becomes `nw` records on its one sequence: the first runs tokens `0 … T - nw` and stores slot `nw - 1`,
and record `j >= 1` runs token `T - nw + j` from the state record `j - 1` wrote and stores slot `nw - 1 - j`
(`tsr_lower_gdn_slots`). Each is the K = 1 record of its tokens, so a snapshot is bit for bit what a kernel that
stored its tile after each token would write, and the attention rows are those of one record of `T` tokens. The
`SSM_CONV` record writes slot 0 itself; slots `1 … nw - 1` are `CPY` records from the window (rows
`T - s … T - s + d_conv - 2`) into the group (`tsr_lower_conv_group`). A decode token with `n_rs_seq > 0`
therefore runs exactly the records of a context without snapshots. The state group of SW-82 takes the slots'
`CPY` (`absorb_cpy`): the records write the cache's groups directly, reading the cell when the gather ids are
the identity and the gather's copy otherwise (after a rejection). More than one sequence with more than one slot
per ubatch is refused with a message. Without the group (`TSR_GDN_CELL=0`) the chain writes the slots into the
result tensor's state rows and the generic `CPY` records move them. The cost is one extra read and write of
the state per token and layer.

With `GGML_SCHED_DEBUG=2` the verification graph and the draft graph are two scheduler splits (the CPU's
token-embedding lookup and one Tesseract split); `TSR_STATS` shows one launch per verification, per draft step
and for the catch-up decode.

SW-85 CONCAT of two rows. The MTP head's `concat(e_norm, h_norm)` (dim 0, F32, rows of whole lines) is two
`CPY` records into the result's rows (`tsr_lower_concat`); nothing changes numerically, and the draft graph has
no CPU island.

SW-88 The output head in row blocks. `mm_plan` tiles `T` by `t_tile = (budget / 2) / (x + QA + y lines per token)`
and every tile streams all the weight rows again; for a large vocabulary matrix (N = 248,320, y = 15,520 lines
per token, budget 60,144 lines) that is one token per tile. A matrix for which T_MAX tokens do not fit one tile
is therefore stored at upload (`tsr_place`) as `P` in {2, 4, 8} row blocks of `N / P` rows (N a multiple of
`16 P`; the smallest `P` whose block plans `t_tile = T_MAX`), each symmetric-allocated over the channels like an
expert (`n_exp = P`, EXP_STRIDE). A `MUL_MAT` is one record per block, block `p` writing its `N / P` values of
every token's row at line `p · N / (16 P)` of the row (`Y_STRIDE` stays the whole row's). Every weight row is
streamed once for all `T <= T_MAX` tokens; the bits are unchanged (an output row does not depend on `N`).
`T = 1` also takes `P` records. Epilogues are not fused into a split head. `TSR_HEAD_SPLIT=0` keeps matrices
whole.

SW-86 Tools and tests. `tsr-mtp` (`sw/tools/tsr_mtp.cpp`) is llama.cpp's `speculative-simple` loop with the
device counters (hook `ggml_backend_tesseract_counters`) read around the draft, verify, process and `seq_rm`
phases, a plain greedy mode on a second target context (the token-equality baseline), `--mtp-k-list`,
`--mtp-rounds`, `--mtp-force-reject`, the generated ids per prompt and a JSON summary. `klcheck run --verify-k K`
replays a token file in windows of up to `K + 1` tokens with every row an output, the accepted length cycling
through `0 … K` (the tokens behind it replaced by wrong ones, then `seq_rm`); `klcheck all --verify-k K
--verify-identical` adds it as a device phase compared with the CPU against the floor and with the plain
device run. `test-tsr-mtp` (ctest `ggml_tesseract_mtp_*`, §10.2) checks every accepted length 0 … K and every
rollback: all rows of the verification batches are bit-identical to plain decoding on the same device, within
`cos >= 0.9995` of the CPU, and every verification is one launch.

Not supported: more than one sequence with snapshots in one ubatch; the draft head's own `K` chain heads
(`step35`: one head per draft step) and DFlash; probabilistic (sampled) drafting.

---

## 8. One launch per decode token: what llama.cpp's graph needs

The full-mode target is that every decode graph produces exactly one Tesseract split. This needs:

1. R1, every node after the input layer is supported (§9.3 supports_op table), including the nodes of the reserve
   graph (prefill T up to n_ubatch). A node whose pre-allocated tensor sits in a Tesseract buffer must be
   supported, or the scheduler aborts ([llama-pin.md](../llama-pin.md)); that applies to SET_ROWS into the KV
   cache. `GGML_OP_DUP` and a SET_ROWS whose dst is not a KV tensor stay unsupported: the unselected embedding
   branches of `ggml_build_forward_select` appear in every decode graph, and a supported DUP there would create a
   second Tesseract split. supports_op does not test `GGML_TENSOR_FLAG_COMPUTE` (test-backend-ops asks before
   the flag is set); `graph_compute` skips flagless nodes and makes no launch for a split that has none (SW-40).
2. R2, token embedding stays on the CPU. llama.cpp keeps the input layer on the CPU. Its GET_ROWS forms a CPU split
   before the device split and costs no launch. The scheduler copies the n_embd f32 result in with `set_tensor`.
3. R3, KV cache on the device (full mode): run without `-nkvo` (`offload_kqv` = true), with `-ctk q8_0 -ctv q8_0` and
   `-fa on`. The pinned llama.cpp refuses a quantized V cache without FA. Matmul mode runs with `offload_kqv` = false
   (§10.4).
4. R4, Hadamard KV rotation off (SW-16). With a quantized KV cache and head_dim % 64 = 0, llama.cpp rotates Q, K
   and V and un-rotates the attention output by `MUL_MAT(rot, x)`, where `rot` is an F32 graph input. That src0
   is not a weight and cannot become an MVM weight, so the rotations would split attention into CPU islands.
   Device runs and every baseline therefore set `LLAMA_ATTN_ROT_DISABLE=1`. The backend warns once if it sees
   `attn_inp_k_rot` or `attn_inp_v_rot`.
5. R5, graph inputs (`inp_pos`, `kq_mask`, `k_idxs`, `v_idxs`, `inp_out_ids`) are written by the scheduler through
   `set_tensor` before the launch. They are copies, not launches. The builder lists the index inputs in the
   preload table.
6. R6, compute buffers fit the device (SW-26, SW-46). A buffer gets one window: ch0 IO, else the low 512 MiB of
   ch1–3, else symmetric WEIGHT-heap reservations of at least 64 MiB between the weights and the KV planes;
   `get_max_size` is 512 MiB (at least the largest single tensor). Chunks are sized from the reserve graph, so
   the reserved logits chunk (vocabulary × n_ubatch × 4 bytes) stays on the device. A chunk that still does not
   fit is staged and runs in its own launch (one extra launch per token, reported by TSR_STATS as "staged
   chunks"); only MUL_MAT is staged, because the full-mode kinds address their operands in place (SW-53). supports_op
   refuses a full-mode op with an operand in host memory or in a buffer that will be (an ANY buffer of leaf data
   only: control vectors and LoRA adapters; every buffer under `TSR_FORCE_HOST`), which then runs on the CPU. A
   compute buffer that gets no window is placed at its first data access, after the scheduler split the graph:
   graph_compute then aborts naming the buffer and the remedies (a smaller `-ub` or `-c`, or `TSR_KINDS=m6`).
   With the KV cache on the device, `TSR_FORCE_HOST` needs `-nkvo`.
7. R7, graph reuse. llama.cpp reuses the graph while the KV bucket (256 cells) is unchanged
   ([llama-pin.md](../llama-pin.md), KV padding and graph reuse). The backend caches the op list by (graph
   identity, node addresses). Per token the host writes the inputs, launches once and reads the logits. A new
   bucket builds one new op list.
8. R8, trailing CPU ops are allowed. A final logit softcap (SCALE, TANH, SCALE) runs on the device: the VU has TANH
   (VU §6.5).
9. R9, no backend sampling (`-bs`) in full mode. Sampling stays on the host.
10. R10, not supported in full mode: context shift (K-shift RoPE on a q8_0 cache), KV defrag through ggml ops, and
    multi-sequence streams above 1. supports_op logs once, before the scheduler stops on the pre-allocated cache
    tensor, an op other than SET_ROWS writing into the device cache and a refused SET_ROWS into it, naming the
    cause: several sequence streams (`-np N` without `--kv-unified`: use `--kv-unified` or `-nkvo`), more than
    2,048 cells per ubatch (the I64 indices are one preload entry of at most 256 lines: use `-ub` of at most
    2048) or another reason (for example `TSR_FORCE_HOST`).

Ops in the two model families (pinned llama.cpp):

| Model | Ops on the device in full mode |
|---|---|
| Qwen2.5 (`src/models/qwen2.cpp`) | RMS_NORM, MUL, MUL_MAT (+ bias ADD, fused), ROPE NEOX, SET_ROWS (K, V), FLASH_ATTN_EXT (GQA), ADD (residual, fused), GLU SWIGLU, GET_ROWS (`inp_out_ids`), MUL_MAT (head) |
| Gemma 4 dense (`src/models/gemma4.cpp`) | MUL by `inp_scale_rows` (the sqrt(n_embd) embedding scale as an F32 graph input [1, T]: VOP.MULC with a per-row c vector; in the leading CPU split or on the device, one launch either way), RMS_NORM (+MUL; per-head q/k norm; V without weight), MUL_MAT, ROPE NEOX (freq_factors on full-attention layers; a different freq_base on SWA layers), SET_ROWS into two caches (iSWA), FLASH_ATTN_EXT (scale 1.0, SWA mask), ADD, GLU GEGLU (GEGLU16), MUL (`layer_output_scale`), GET_ROWS, MUL_MAT (head, tied to `token_embd`), [SCALE, TANH, SCALE] |

The Gemma 4 12B shape used by the tests: 48 layers (pattern 5 SWA : 1 full), n_embd 3840, n_head 16, n_head_kv 8
(SWA) / 1 (full), head_dim 256 / 512, n_ff 15,360, vocabulary 262,144, n_swa 1024, rope base 1e6 / 1e4 (SWA), rope
dims 512 / 256 with `rope_freqs` [256] on full layers, final logit softcap 30, no `attn_v` on full layers (V = K
projection), no `output.weight` (tied), `layer_output_scale` [1]. The pinned llama.cpp loads it as
`LLM_TYPE_UNKNOWN` (no size label), which changes nothing in the graph.

---

## 9. Host software

### 9.1 Transport API (`sw/transport/tsr_transport.h`)

EX §10.1 is the API, verbatim: open / close / name / config / write / read / csr_read / csr_write,
`tsr_image_open` / `close` / `kernel` / `id` (64-bit), `tsr_transport_boot` (warm if possible, else cold; HELLO
checked), `tsr_transport_launch(t, const tsr_launch *, uint32_t *seq)` with `tsr_launch {type, flags, kernel,
arg[7]}` (blocks only at `LQ_DEPTH` outstanding), `tsr_transport_wait(t, seq, timeout_ns, tsr_record *)`,
`tsr_transport_call`, `_cancel`, `_stop`, `_set_event_cb`, `_soft_reset` (`SOFT_RESET`, wait `SR_BUSY` = 0) and
`_log_drain`. SW adds only conveniences on top of it, not a second protocol: `tsr_image_open_default` (`$TSR_IMAGE`
or the build's `tsr-fw.elf`), `tsr_image_path`, `tsr_transport_image` (the booted image or NULL), the version
queries `tsr_image_versions`, `tsr_transport_versions`, `tsr_basis_version` and `tsr_versions_str` (§6.9),
`tsr_record_decode` (the fields of a record: type, status, error code, SEQ, RET, the first error's ticket, info,
w0 and push index, DURATION in core cycles, 0 for an OK DONE or PONG because EX §4.8's minimal reads skip r7),
`tsr_status_name` and `tsr_transport_get_stats` (launches, wall time, CSR and byte counts, log words and drops,
staged chunks).

Protocol layer (`tsr_kdev.cpp`, SW-18). One implementation serves every kind and follows EX §2.1, §3, §4.8 and
§10.2:

- Open:
  1. Read and check `ID` (0x5453_5231), the `VERSION` major (1), `CFG*`, `CAPS`, `CFG3`, `CTL_ID` not 0, `RV_CFG`,
     `Q_DEPTH` and `RV_FREQ` not 0 (else "no `clk_rv`").
  2. Clear `ADAPTER_STATUS`.
  3. Read the host counters (`LQ_SEQ`, `LCQ_RD`, `LOG_RD`) and the `LQ_W` shadow from the device: the rings count
     on across processes, only a card reset zeroes them.
  4. Unless `TSR_ATTACH=1`: `SOFT_RESET` and wait `SR_BUSY` = 0 (`TSR_RESET_MS`), which stops a control core left
     running by an earlier process and discards the records and log words it left unread (counted in one
     message, not decoded); then bind (channel bases, `MEM_EN`, EX §9.2).
  5. If DB's configuration is armed (`SERIAL`, `ERR_STOP`, `HUNG_STOP`, `HALT_ON_BOOT`, an enabled TBP, a stop:
     DB keeps them across `SOFT_RESET`, DB §7.3), report it; every wait then reads `DBG_STATUS` and `DBG_CFG` at
     its backoff polls, as with `TSR_TILE_SERIAL=1` or a debugger's CSR write.
- Boot: per EX §3.3–§3.5: warm restart when VERIFY of every ITCM segment matches, else a cold load (`LOAD_ADDR`,
  `LOAD_DATA` words, receive CRC, VERIFY CRC); `RV_BOOT` = 0; `RV_RUN` = 1; wait for HELLO (`TSR_BOOT_MS`) and
  check the image id, the Basis ABI, the kind table signature, the kernel count and the Tesseract Basis version
  (§6.9).
- Launch: EX §4.8: at most `LQ_DEPTH` launches outstanding (host-side credit, no read); write the `LQ_W` words
  that differ from the shadow copy; write `LQ_PUSH`.
- Wait:
  - Consume the LCQ (EX §4.8): read `LCQ_WR`, read r0 of each new record (r2–r6 only when needed), deliver events
    to the callback, write `LCQ_RD`.
  - Poll policy: busy-poll while the elapsed time is below the last service time of the same kernel (measured by
    the host from the push or the previous answer, since the minimal reads skip r7 DURATION), then poll every
    max(10 µs, 1 % of that time) with sleeps; between polls call `kdev::idle()` and release the transport mutex
    (DB's in-process debug service needs it). The simulated kinds (`sim`, `mock`, `vkernel`) count timeouts and
    the poll policy in device time (MTIME at the nominal core clock), `xrt` in host time.
  - Drain the log every poll if `TSR_LOG` is set, else every 10 ms.
  - Timeout (`TSR_LAUNCH_TIMEOUT_MS`, default 60,000; 0 = none): time counts only while `STATUS.RV_HALTED` = 0
    and `DBG_STATUS.STOP_ACTIVE` = 0 (read once per backoff poll); with `DBG_CFG.SERIAL` = 1 the timeout is off
    (DB §4.1). On a timeout the transport follows DB's recovery ladder (DB §7.3): level 0 hang dump
    (`TSR_DUMP_DIR`), level 2 `LQ_CANCEL` covering the outstanding launches and a wait of `TSR_ABORT_MS` for
    DONE / ABORTED, then level 3 `SOFT_RESET` + warm restart (EX §3.5) if the RISC-V is stuck. A hung unit
    (`UNIT_HUNG`) ends with "card reset needed" (EX §8.6).
- TRAP record: drain the log, report the decoded trap (cause, `mepc` to function and line); attach GDB if
  `TSR_GDB_WAIT` asks for it, else `SOFT_RESET` + warm restart.
- PLAT_ERR (EX §8.6): one report per episode (`STATUS.PLAT_ERR` at a backoff poll or a DONE PLAT_ERR answer). The
  episode covers the launches pushed before it was seen; `ADAPTER_STATUS` (W1C, which drives `plat_err_i`) is
  cleared once all of them have answered, so they answer PLAT_ERR and later launches answer normally even when
  launches are pipelined. An adapter error raised during the episode extends it to the launches pushed so far.
- Watermarks (EX §4.9): a watermark written since the last `SOFT_RESET` is rewritten just below the oldest
  unanswered launch 2^30 launches after its write, so it never wraps onto new launches (it would after 2^31).

kdev (EX §10.2). Methods: `reg_read`, `reg_write`, `mem_read`, `mem_write`, `num_ch`, `ch_bytes`, `buf_bytes`, `ch_addr`,
`detect_capacity`, `describe`; `KREG_ADAPTER_CTRL` and `virtual void idle(uint32_t poll_iter, uint64_t interval_ns)` (EX's `advance`
hint), with `now_ns()` (the time base), `simulated()`, `has_adapter()` and `sim()` (`sw/transport/tsr_kdev.h`).

| kdev | `idle()` and specifics |
|---|---|
| `sim` | in-process simulator behind the core CSR port; `idle()` = `tsr_sim_advance(budget)`; a host read of `LCQ_WR`, `STATUS` or `LOG_WR` first runs the model to the next host-visible event (EX §10.2) |
| `mock` | the `sim` kdev plus the `ap_ctrl_none` adapter register model (channel bases, `MEM_EN`, `ADAPTER_STATUS`, error injection) |
| `vkernel` | the Verilated kernel server with two clocks; `idle()` runs the server's clocks |
| `xrt` | `ap_ctrl_none` compute unit opened through `xrt::ip` with exclusive access; registers through `read_register` and `write_register`; one device-only BO per channel (memory topology from `xrt::xclbin`, EX §9.1) whose usable size is the capacity measured at open; `idle()` sleeps (backoff) |

The `xrt` kdev reads the `TSR_XRT_DMA*` and `TSR_XRT_CH_GIB` settings.

### 9.2 Environment (host)

| Variable | Meaning | Default |
|---|---|---|
| `TSR_DEVICE` | `sim`, `mock`, `vkernel`, `xrt`, `none` | `sim` |
| `TSR_IMAGE` | firmware ELF (EX §10.2) | build tree `tsr-fw.elf` |
| `TSR_LAUNCH_TIMEOUT_MS`, `TSR_ABORT_MS`, `TSR_BOOT_MS`, `TSR_RESET_MS` | wait timeout per launch (debug-aware, §9.1), cancel wait, HELLO wait, `SOFT_RESET` wait (EX §10.2) | 60000, 10000, 2000, 15000 |
| `TSR_LOG`, `TSR_LOG_LEVEL` | decode the log to `stderr` or a file; `LOG_CFG.LEVEL` 0 FATAL … 5 TRACE (DB §6.1) | off, 3 |
| `TSR_ATTACH` | open without `SOFT_RESET` or load (debugger, post-mortem) | 0 |
| `TSR_TRACE` | per-launch trace dumps | off |
| `TSR_STATS` | launches per token, ops per launch, fused ops, staged chunks (DB §8.6) | off |
| `TSR_MON`, `TSR_MON_DIR`, `TSR_MON_PERF_MS`, `TSR_MON_KEEP`, `TSR_MON_MODEL` | the live monitor record of the owner (SW-87, [prof-dbg.md](../prof-dbg.md) §7): `0` turns it off; directory; least ms between PERF reads; keep the file at exit; model name | on, `/dev/shm`, 100 |
| `TSR_GDB`, `TSR_GDB_WAIT`, `TSR_DEBUGGER`, `TSR_DUMP_DIR`, `TSR_TILE_SERIAL`, `TSR_PROF*` | DB §9.5 | off |
| `LLAMA_ATTN_ROT_DISABLE` | must be 1 for device runs and baselines (R4) | set by klcheck, numcheck and the tests |

### 9.3 `sw/ggml-tesseract`

Matmul mode: attention stays on the CPU (`offload_kqv` = false, §10.4). `graph_compute(split)` builds an OPL with
one MUL_MAT record per MUL_MAT node (and one MUL_MAT_ID record per MUL_MAT_ID node), plus JOINs (two MUL_MATs
are dependent only if one reads the other's output; host-side hazard analysis over DRAM byte ranges covers WAR
and WAW), stages host-memory chunks (`tsr_lower_staged`, `tsr_act_loc`), writes the OPL into the OPL cache (an
LRU in the 64 MiB OPL region, keyed by content hash), launches `graph` and waits. On DONE STATUS TILE_ERR it
maps RET-1 to the ggml node and aborts with the node name, the error code and `ERR_INFO`. `supports_op`
(MUL_MAT rules) gives one launch per split. `tsr-opl.{h,cpp}` holds the builder, hazards, encoding and cache;
tiling is in `plan.hpp`. MUL_MAT src0 is resolved by `tsr_resolve_weight`; a raw-buffer src0
(`test-backend-ops` puts src0 in non-WEIGHTS buffers) is converted lazily (`tsr_lazy_*`).

Full mode. Builder passes:

1. Drop no-op nodes (NONE, VIEW, RESHAPE, PERMUTE, TRANSPOSE).
2. Resolve each tensor to a handle and line stride.
3. Fuse (SW-42): MUL_MAT + ADD whose other operand is a bias or the residual (via the MUL_MAT addend); RMS_NORM +
   MUL(weight).
4. Compute dependencies (RAW, WAR, WAW over the DRAM byte ranges of ggml-alloc addresses, `kind_ranges`),
   transitively reduce them, and emit JOINs above two.
5. Build the preload table (index inputs, positions, MSUMs; SW-43).
6. Encode.

The result is cached per graph identity (R7, SW-45).

supports_op (full mode). Every condition below also requires every operand's base offset and row stride to be
multiples of 64 bytes.

| Op | Supported when |
|---|---|
| NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE | always |
| MUL_MAT | the matmul-mode rules (`docs/spec.md` §12) |
| ADD, MUL | F32; src1 broadcast by whole rows (a row period) or a single scalar |
| SCALE | F32 |
| RMS_NORM | F32 rows |
| GLU | SWIGLU, or GEGLU lowered to GEGLU16 (the CPU's fp16-table form, VU §6.5); split form or non-swapped single form |
| ROPE | NORM/NEOX, F32, I32 positions that are a graph input, freq_factors F32 or none, head_dim a multiple of 16 (≤ 1024) and n_dims a multiple of 32 (NEOX) or 16 (NORM) (VU §5.4); IMROPE as NEOX when the positions are llama.cpp's `inp_pos` input and its streams are equal (text, SW-72); no other mrope / vision modes |
| GET_ROWS | F32 src0 in a compute buffer, I32 ids that are a graph input |
| SET_ROWS | dst in a KV buffer, F32 src, I64 ids that are a graph input (any other SET_ROWS: false, R1) |
| FLASH_ATTN_EXT | AT §2.6: Q F32; K/V q8_0 views in KV buffers; D = DK = DV ∈ {64, 128, 256, 512}; mask F16 or none; `max_bias` = 0; `logit_softcap` = 0; no sinks; `n_kv_max` = 0 |
| CPY, CONT | F32 → F32 with contiguous rows; a copy whose two sides have rows of different length but the same values, into one dense run, regroups into the source's rows (the new GDN state into its cache cell) |
| UNARY | TANH; GELU lowered to GELU16; SILU; SIGMOID; SOFTPLUS (kind SOFTPLUS, SW-69). A tensor whose rows break the 64-byte rule is one flat row when it is contiguous (the `[1, H_v]` gate tensors) |
| GATED_DELTA_NET | F32; scalar gate (`g->ne[0] = 1`), K state snapshots (K > 1 with one sequence only: a chain of records, §7.11), S_v a multiple of 16 up to 256, H_v a multiple of H_k, g / beta contiguous with H_v a multiple of 16 when there is more than one token, state contiguous; q / k / v rows with heads and tokens 64-byte spaced (§7.10) |
| CONCAT, SSM_CONV, CPY of a conv state | the group of §7.10 (SW-71): CONCAT(state, transposed tokens) along time, SSM_CONV with F32 taps, d_conv 2 … 8, C a multiple of 16, and the state CPY (one per rollback slot, §7.11), with the state cache in a Tesseract buffer; any other CONCAT: F32 along dim 0 with rows of whole lines (two CPY records, SW-85) |
| GET_ROWS, SCALE, CPY on a recurrent-state cache | when the cache is in a Tesseract buffer; a cache in another backend's memory keeps the op on the CPU (SW-72) |
| DUP, nodes without `GGML_TENSOR_FLAG_COMPUTE` | false (R1) |
| MUL_MAT_ID | `MUL_MAT_ID` kind in the image; weights a loadable 3D expert tensor (WEIGHTS buffer, or raw bytes converted at graph_compute, or llama.cpp's unallocated probe leaf), `N` a multiple of 16, x F32 rows of whole lines, ids I32 rows, result of its own (§7.9, SW-62) |
| SOFT_MAX, ARGSORT, GET_ROWS, SUM_ROWS, CLAMP, DIV, REPEAT of the MoE chains | `MOE_ROUTE` (the router chain) or `MUL_MAT_ID` + `CPY` + `MUL` (the scale chain) in the image, and the node's own sources match `build_moe_ffn` (§7.9, SW-64); the fusion pass takes the group |
| everything else | false |

One lowering for supports_op and the builder (SW-20, SW-41). `tsr_lower_m7` turns a node into its record (§7.2
words); supports_op runs it in probe mode (handles 0, the KV shape from `tsr_kv_identify`, no device state) and
asks the kind's planner (`plan_kind` on the device's CFG0–3 and CAPS values) and `opl::check_args`, so the host
refuses exactly what the firmware cannot plan or the builder cannot encode (`ggml_tesseract_units` sweeps
shapes of every kind against independently built words). The 64-byte rule: every row operand's base and row
stride are multiples of 64 bytes, and a result that is a view must have rows of whole lines (n·4 % 64 = 0),
because the kinds store whole lines and a row ending inside a line would overwrite the viewed tensor's bytes
after it; a result of its own may be one row ending inside a line (ggml-alloc pads it). KV views must be the
ones llama.cpp builds with one stream: starting at the plane's cell 0, cells `row_size(E)` and heads
`row_size(D)` apart. Operands in host memory are refused (R6, §8).

Preload table (pass 5, SW-43). One entry per (source handle, size): positions, K indices, V indices, the MSUM
and the last layer's `inp_out_ids` (and per cache for iSWA models); index inputs must be graph leaves. FA_NEW
(word 21) names the indices of the SET_ROWS_KV that wrote the record's K plane earlier in the same
graph_compute, as its own entry when that record went into an earlier list, so the chunk order (hence the
output) does not depend on where the builder cut its lists. A list is cut early at 4096 records or when an
entry does not fit the 256-line preload area; a FLASH_ATTN's own entries, the MSUM (ceil((32 + 16 T) / 64)
lines) and FA_NEW's K indices (ceil(8 T / 64)), must fit one fresh list together: T at most 680. supports_op
refuses a FLASH_ATTN of more tokens, which then runs on the CPU. FA_NEW across a cut reads the K indices from
DRAM at the start of the FLASH_ATTN's list. The FLASH_ATTN is not a ggml consumer of the indices, so ggml-alloc may
reuse their bytes after the last SET_ROWS_KV that reads them. In a layer with its own KV only the V SET_ROWS_KV lies
between the two (llama.cpp's `build_attn` expands q, k and v before the cache writes). A layer that shares KV (Gemma 3n
and 4, E2B and E4B) attends to a plane that an earlier layer wrote, with the nodes of the layers in between, and their
results lie on the indices. When a node between a K SET_ROWS_KV and the last FLASH_ATTN that reads its plane overwrites
the indices (the byte ranges of the node results), graph_compute puts a CPY record right after the SET_ROWS_KV that
copies the indices to a device area of its own (`tsr_lower_kidx_snapshot`; 8 slots of 8 KiB, one per index tensor,
taken at the first use like the MOE_ROUTE scratch). A FLASH_ATTN in a later list than the CPY preloads FA_NEW from
the copy; one in the list of the CPY reads the original, which is still intact at the start of that list. The words
and the chunk order are the same, so the output bits do not depend on the copy. When the copy cannot be taken (no room
for the area, an image without CPY), graph_compute checks that no node between the two writes the indices whenever a
cut separates them, and aborts naming the node otherwise.

Plan cache (R7, SW-45). A graph's lists are kept under `tsr_graph_key` (node and source addresses, ops, flags,
shapes, strides, op parameters; not values) salted with a placement epoch; a hit re-registers the FLASH_ATTN
mask readers, checks their words and runs the lists. Builds that stage a MUL_MAT or convert a weight lazily are
not kept.

Compute heap (R6, SW-46). `get_alloc_size` pads to 64 bytes only where ggml allows it (`[TAG_ALLOC_SIZE_EXPAND]`:
leaves, views, quantized types, MUL_MAT, FLASH_ATTN_EXT); ggml-alloc pads every block to the 64-byte alignment,
which holds a single row's whole-line store.

Knobs. `TSR_KINDS=m6` restricts the device to MUL_MAT (the matmul mode; the matmul-mode e2e ctests run with it and
`TSR_COMPUTE_WINDOWS=io`); `TSR_KINDS=m6,+MUL_MAT_ID` adds MUL_MAT_ID (§7.9). `klcheck` and `tsr-numcheck`
`--phase m6` set the matmul-mode restriction (SW-49).

Versions (§6.9, SW-54). At open the backend refuses an image whose Tesseract Basis (major, minor) is older than
its own and builds a kind mask from the booted image's `.tsr_kinds` (`TSR_KINDS=m6` intersects it with NOP,
JOIN and MUL_MAT); supports_op, the MUL_MAT path included, refuses a kind outside the mask ("kind ADD is not in
the booted image (Tesseract Basis X.Y.Z, image <id>)", at `TSR_DEBUG=2`), so a subset image leaves those ops on
the CPU; the builder asserts the mask before each launch. The description reads "Tesseract Core 1.0 (VU ATT
GDN; sim, C=4, …), Tesseract Basis 1.0.0 (ABI 1, N kinds, image <id>)";
`get_proc_address("ggml_backend_tesseract_versions")` returns the `tsr_versions` of the opened device. `sw/rt`
applies the Basis rule and requires MUL_MAT.

Weight placement. llama.cpp places a weight by a probe op (`src/llama-model-loader.cpp`); its ROPE probe has
every parameter 0, which the planner would refuse, so probe mode plans a ROPE with `n_dims` 0 and a freq-factor
tensor as a full rotation: a model's `rope_freqs` lives on the device (else every full-attention layer's ROPE
would read it from the CPU and cut the split).

KV buffers (SW-17, SW-47). `init_tensor` records each tensor; a buffer of usage ANY whose tensors are all
`cache_<tag>k_l<n>` or `cache_<tag>v_l<n>`, q8_0, 3D contiguous `[E, kv_size, n_stream]` with E % 64 = 0,
64 <= E <= min(K_MAX, 511 · 64), LG < 2^16 and a plane < 2^28 beats is a KV candidate (AT §4.6; a buffer with
any other tensor keeps the ordinary buffer rules). Its planes are placed at the first data access (llama.cpp's
clear): one plane per (tensor, stream), top-down in the symmetric WEIGHT heap (`alloc_top`, the same offset on
every channel, 4 KiB aligned, the first tensor highest); when they do not fit, the lazily converted weights are
dropped and the allocation retried, else the backend aborts suggesting a smaller context or `-nkvo`.
`set_tensor`, `get_tensor`, `memset_tensor` and `clear` take any byte range and convert in windows of whole
local groups (8 MiB of plane bytes) with AT's `tsr_kv_set_bytes` and `tsr_kv_get_bytes`; a window is read from
the device only when the access does not cover all its rows. llama.cpp's state save and load is lossless
(`ggml_tesseract_kvstate`). llama.cpp's size-0 dummy buffers of the memory-fit pass have no context; supports_op
treats their cache tensors as KV tensors (the builder's `kv_probe`).

Mask summaries (SW-32, SW-44, SW-48). A tensor is a mask when its name contains `kq_mask`, F16,
`[n_kv, n_q, 1, 1]`, in device memory; supports_op also requires n_kv = K's cells and n_q = Q's tokens.
`get_alloc_size` reserves, behind the mask, its MSUM (32 + 16 · n_q bytes) and one f32 copy (n_q · 64 ·
ceil(n_kv/16) bytes). Each FLASH_ATTN is registered as a reader (scale, E, D, G) on every build and every
cached reuse; the reader set only grows until the mask's shape changes or its buffer is freed or reset. On each
`set_tensor` or `memset_tensor` of a mask the producer writes the MSUM (`tsr_msum_build_f16`), plans every
query tile of every reader with the shared planner (`tsr_fa_plan`) and writes the f32 copy of a scale
(`tsr_msum_mask_f32`, fdiv(m, scale)) only when the MSUM is GENERAL or a tile takes the GENERAL path. When the
planner would refuse a tile (AT §6.6 rule 5 and 6), the producer promotes the MSUM to GENERAL (flags bit0,
header word 4 = the reserved copy's handle) and writes the copy, so the executor never refuses for mask data
and there is no CPU fallback. One copy per distinct scale: the first in the reservation, further ones as side
allocations of the WEIGHT heap (never evicting weights); an all-±0 / -inf mask gives one image for every scale.
FLASH_ATTN word 13 always names the copy of its scale (§7.2), so the cached list stays valid while the mask
changes. An FA without a mask gets an all-visible MSUM (`nomask`). Summaries are forgotten on buffer free,
reset and LINEAR clear; after a partial `set_tensor` or a memset the mask is read back from the device.
`TSR_STATS` adds the mask summary counts. Test hooks: `ggml_backend_tesseract_kv_plane`, `_mask_summary`,
`_att_cfg`.

### 9.4 `sw/rt` and `sw/torch`

`tsr_rt::linear` emits one `graph` launch per IO tile, with one MUL_MAT record per N part × K part (K parts after
the first with ACC_DST). The OPL cache holds the op lists. The public API and the numerics are defined in the
header comment of `sw/rt/tsr_rt.h`. The runtime boots the image like the ggml backend (§6.9): its Basis major and
minor must be at least the host's, and it must provide MUL_MAT and JOIN.

- Token tiling. A record carries the call's T (one IO tile). The firmware cuts it into tiles of min(T_MAX, its
  scratch ring) tokens (SW-20), so the runtime does not tile. On the release image T_MAX is 4: a 32-token call is
  eight tiles inside one launch, and the weight stream is read once per tile, not once per token. The tests run
  `tsr_rt` and the PyTorch suite at T_MAX 1, 4 and 8 (`rt_sim`, `rt_sim_tmax4`, `rt_sim_tmax8`, `torch_pytest`,
  `torch_pytest_tmax4`, `torch_pytest_tmax8`); the bits do not depend on T_MAX.
- Scaled result. `try_linear_scale` runs MUL_MAT_SCALE (0x1C) when the image has the kind and
  `CAPS.MVM_LOCAL_SCALE` and the weight has one K part. It returns false, with nothing transferred, otherwise; the
  caller then runs `linear` and multiplies on the CPU. The release image has no such capability.
- PyTorch. `sw/torch` partitions the aten inference graph of a `torch.compile`d model and sends each weight matmul
  (`mm`, `addmm`, `linear`, `matmul`, `mv` with one weight operand) to `tsr_rt` as one launch. Bias, scale factors
  that do not qualify for MUL_MAT_SCALE, activations, norms, attention and embeddings run on the CPU. A bounded FX
  path takes a bias-free FP32 right-weight `mm`, `matmul` or `linear` followed by a sole positive power-of-two
  scalar multiply in [1, 2^127] as MUL_MAT_SCALE when the image allows; FP16 and BF16 casts stay on the ordinary
  path. Weights that arrive as ggml blocks (Q8_0, Q4_K) go through `QuantLinear` (SW-81). Lowering whole graphs
  onto the full-mode kinds is not built; [torch-backend.md](../torch-backend.md) §8 says what the backend would need.
- Checks. `rt_*` (device against an independent model, bit for bit), `torch_pytest*` (the extension against the
  numpy spec model; compiled models against eager with `reference_linears()`), and
  `sw/torch/tools/hf_check.py` (logits of the device against the reference model, the llama.cpp path's oracle).

### 9.5 Tools

| Tool | Function |
|---|---|
| `tsr-selftest` | identity (Tesseract Core capability, CSRs, `CTL_ID`, `RV_FREQ`, `Q_DEPTH`, adapter); the image's and the host's versions and kinds (§6.9; `--version`); host memory tests; load + VERIFY + HELLO + image id + PING round-trip time and pipelined launch rate (EX §4.11); the `selftest` kernel tests with in-process sim shadow; `--traces DIR` replays through `replay` or full launches; `--debug` DMI halt/resume/step smoke (DB's checks); log drain |
| `tsr-trace-gen` | `ops` traces: `--ops m2|m3|vu|at|ctrl`, errors, concurrency, fixed seeds |
| `tsr-trace-replay` | replay on the simulator (full or tile-only), `--schedules N` determinism mode; `--device KIND` replays through a transport |
| `tsr-trace-conv` | converts `traces/model` into `traces/model-v1conv` (§5.6) |
| `tsr-elf` | image id, kernel table, kind table (`kinds`: one line per kind and revision, the signature), `.note.tsr` (`note`: `cap_min` and `basis_version` as versions), sizes, `.tsr_logfmt` dump, `--version` |
| `tsr-log` | decode raw log words (DB §6.1 header) with an ELF |
| `klcheck`, `numcheck` | settings: FA on (SW-22, §10.4; `--fa` records `on`); KV type q8_0/q8_0; `offload_kqv` = false in matmul mode and true in full mode; `LLAMA_ATTN_ROT_DISABLE=1` in every phase (recorded in the `.klc` header and the numcheck run metadata). numcheck's per-node observation (`cb_eval`) makes the scheduler compute one node per `graph_compute`, which is one launch per node, so numcheck needs no device mode (SW-27). klcheck runs the SWA layers' KV cache as the ring llama-cli and llama-server use (`--swa-full off`, the default; recorded in the `.klc` header, 0 = full-size), so long Gemma 4 prompts wrap it. The `embd` and floor inputs of klcheck and numcheck carry the token-row scale the graph gives token rows only (sqrt(n_embd) for Gemma 3, 3n and 4, `inp_scale_rows`). `klcheck -v` marks each `llama_decode` (the graph tests' launch count per decode). A device run records the Tesseract Core capability, the Tesseract Basis version and the kind table signature (`.klc` header, 0 = not recorded; `report.json` `versions`); numcheck's device run records them with every kind's revision (`meta.json` and `report.json` `versions`) |
| `tsr_bw` | separate DRAM bandwidth kernel |
| `tsr-mtp`, `tsr-dbg`, `tsr-prof`, `tsr-top` | multi-token prediction loop (§7.11); debug client (DB §9); trace writer and PERF summary (DB §8; [prof-dbg.md](../prof-dbg.md)); live monitor (SW-87) |

### 9.6 Toolchain installation without sudo (SW-7)

The build host needs no root. The tools live in user space, with versions pinned in `docs/design/toolchain.lock`
and installed by `sw/ctrl/tools/install-toolchain.sh`, which verifies every pin and writes the observed facts to
`installed.lock`. CMake finds the tools through `TSR_RISCV_PREFIX`, `TSR_SPIKE` and `PATH`.

| Tool | Plan |
|---|---|
| RISC-V GCC + GDB | xPack GNU RISC-V Embedded GCC `riscv-none-elf-gcc`, the pinned release tarball checked against its sha256 and unpacked under the install prefix. It includes `riscv-none-elf-gdb`, which serves for GDB over DMI (D18). The multilib rv32imc/ilp32 matches `-march=rv32imc_zicsr_zifencei -mabi=ilp32` |
| Clang (optional) | conda-forge `clang` and `lld` in a separate conda environment; uses the xPack `libgcc.a` |
| dtc | built from source (`make NO_PYTHON=1 NO_YAML=1`); Spike calls it at run time |
| Spike | `riscv-isa-sim` at a pinned commit; the build needs no Boost for the features used |
| riscv-tests, riscv-arch-test | pinned clones built by the harness (`sw/sim/rvtests.cmake`) with XLEN 32 |

---

## 10. Verification

The C simulator in `sw/sim` is the bit-exact oracle. RTL blocks, firmware kinds and the host stack are checked
against it. Every output bit that a change moves needs a reason, and a firmware change that moves a kind's output
bits is a kind numerics revision (§6.9, `fw_kind_revs`). Checks run in this order: unit semantics against the C
references, firmware against the host references, the whole stack against ggml on the CPU (NMSE and KL), then the
same image on every transport.

### 10.1 Oracle chain for this area

| Object | Oracle | Check |
|---|---|---|
| `tsr_ref_math.c` | (a) an independent numpy bit model in `sw/sim/tests/ref_math_np.py`, written from the VU document; (b) double-precision libm for accuracy | (a) bit-exact over all 2^32 inputs per unary function (opt-in `slow`; default: 2^24 strided + specials); (b) max ulp error at most VU's bound, reported; (c) RTL: VU blocks against the C function, exhaustively on a Verilator C++ bench (opt-in), 10^6 random + specials by default |
| `tsr_m_gelu16` | the pinned ggml CPU table `ggml_table_gelu_f16` (built by `ggml_gelu_f32` with libm `tanhf`) | all 65,536 fp16 inputs enumerated; mismatches reported; target (VU §6.5): below 1 % of the entries, no sign-of-zero mismatch, more than 1 fp16 ulp only where GELU16 is correctly rounded and the table entry is not (the residual is the table's own f32 error, which the VU does not reproduce) |
| VU and AT references in `tsr_ref_vu.c` and `tsr_ref_at.c` | numpy spec models (numcheck B1b, [spec](../spec.md) §17) | bit-exact on generated vectors; directed RED cases where a subtree sums to -0 (for example 0x80C00000 + 0x00800000) at m = 8k + r (VU §6.2) |
| `tsr_rv` | Spike | on the test bus profile (§4.2: RAM at 0x8000_0000, HTIF): riscv-tests rv32ui/um/uc/mi `-p` and riscv-arch-test RV32IMC/Zicsr signatures; random programs (in-house Python generator, placed at 0x8000_0000 on both) compared retire by retire with `spike --log-commits`; `test_ref_fp.c` built for RV32 and run on both |
| Queue model | EX §7–§8 and CC §7–§9 | directed tests through `tsr_sim_push`; invariants I1–I3 asserted inside the model on every run |
| Race detector | none | positive and negative cases; mutated executors (a dropped WAR dep, a dropped ring release) must be reported |
| Firmware kernels | `sw/ctrl/ref` C++ references; the host build | RV build on sim bit-exact with the reference and with the host build; 8 schedules; race `fatal` |
| Backend and executor | ggml CPU | `test-backend-ops` NMSE per supported op (sim); klcheck KL (sim, then card) |
| Cross-transport | sim | identical logits and DRAM outputs on `mock`, `vkernel`, `xrt` for the same image and inputs |
| Launch traces | fresh-boot replay | every `TSR_TRACE` capture in ctest replays bit-exact on a freshly booted simulator (catches DTCM state leaks, SW-4) |

### 10.2 ctest plan

`ctest -N` lists the tests; the groups below say what each one pins. Every RISC-V test runs the fixed and 7
random schedules with the race detector fatal (§4.7) and compares outputs bit for bit with the reference of
the kind.

| Group | Tests and what they check |
|---|---|
| References | `ref_*` (layouts, fp helpers, ggml fp16, QACT and dequant against ggml, dot products, ctypes); `ref_math_<fn>` including `ref_math_gelu16` (the 65,536 fp16 inputs against the pinned table); `ref_vu`, `ref_at` |
| RISC-V model | `rv_isa`, `rv_arch`, `rv_lockstep`, `rv_random` (riscv-tests and arch-test signatures on `tsr_rv` and, when Spike is installed, on Spike; commit logs compared retire by retire) |
| Simulator models | `sim_basic` (EX §15 item 2 on the simulator), `sim_mvm`, `sim_queue`, `sim_race`, `sim_sched_determinism`, `sim_vu`, `sim_at`, `sim_dm`, `sim_dbg`, `sim_trace`, `sim_transport`, `sim_w1b_e2e` (all models together on the test firmware: 8 schedules, launch capture with fresh-boot replays, a GDB-style DMI session that leaves the results unchanged) |
| Runtime | `fw_runtime` (R1–R10: boot, HELLO id, PING, BAD_DESC, argument block, 8 outstanding launches, back-pressure with a full LCQ, cancel and stop watermarks, TRAP record, panic as a TRAP with `mcause` 24, log drop and BLOCK, QCLEAR after an RV reset), `fw_runtime_stale` (a mutant image gives exactly one `SPM_STALE`) |
| Image and versions | `fw_elf*` (`tsr-elf`, `link.ld`, image id, test images, the note's versions, the kind table and its mutations, HELLO r4 and r6), `fw_kind_revs` (§6.9: every kind's output digest on a fixed corpus against `sw/ctrl/tests/kind_revs.txt` at its revision; the image declares exactly the kinds with a revision; each launch takes its fast or general path), `ggml_tesseract_versions_*`, `rt_sim_basis_old`, `rt_sim_no_mulmat` |
| Executor and headers | `fw_hdr_*` (builders, tickets, `group`, `spm_ring` against the reference algorithm), `fw_exec_*` (the op list format and the executor; streamed lists' trace digests pinned; PERF PUSH_STALL = 0), `fw_m6_e2e` (the production image through the loader with VERIFY, HELLO, PING, a two-layer MUL_MAT list, `selftest` ids 1–10, `replay` of trace samples; every captured launch replayed on fresh simulators), `fw_m6_perf`, `fw_selftest_sim`, `fw_replay_sim` |
| Kernels | `fw_kernels_host` (ASan/UBSan; fast against general path push for push on twin simulators), `fw_kernels_rv`, `_rv_v1conv`, `_rv_shapes` (`slow`), mutant images that must be caught (`_mutant_race`, `_mutant_v1`); the full-mode kinds: `fw_kernels_elem_*`, `fw_kernels_normrope_*` (planner invariants, handler lists against `elem_ref`, refusals, `_perf`, `_ggml` in LLAMA builds), `fw_kvfa_*` (the shared chunk planner, GENERAL tiles, the MSUM n_kv and preload-extent refusals, NMSE at most 1e-9 and row-softmax KL at most 1e-12 against ggml-cpu), `fw_opl_m7`, `fw_plan_cxx17`, `fw_exec_perf_m7` (the 350-cycle gate, §7.8), MoE `fw_kernels_moe_host`, Gated DeltaNet `fw_gdn_rv`, `_errors`, `_composed`, `fw_gdn_ggml` (§7.10) |
| Whole layers | `fw_m7_layer_qwen05b`, `fw_m7_layer_gemma12b` (one decoder layer per OPL list: every record's output and both KV planes bit-exact against the per-kind references composed in op order, no write outside them), `fw_m7_layer_ggml` (against ggml-cpu), `fw_m7_layer_perf` (§7.8) |
| Backend and runtime on the transports | `ggml_tesseract_e2e*`, `ggml_tesseract_scalar_nmse`, `ggml_tesseract_backend_ops` (`_bigt` is `slow`) and `ggml_tesseract_backend_ops_<op>` (test-backend-ops on `sim` per full-mode op filter: every supported case passes, a minimum number is supported, the unsupported cases are counted), `rt_sim*`, `rt_mock*`, `rt_vkernel_k4t1` (opt-in), `torch_pytest*` (`TSR_WITH_TORCH`), `transport_*`, `ggml_tesseract_units` (OPL builder, fusion, supports_op against independently built words), `ggml_tesseract_opl_launch*` |
| KV and attention | `ggml_tesseract_attn` (SET_ROWS + FLASH_ATTN with a KV buffer in llama.cpp's `build_attn` graph shape: D 64 to 512, GQA, decode and prefill, causal, SWA, wrapped ring, holes; per run the device cache equals the history plus `quantize_row_q8_0_ref` of the new rows, the output equals `tsr_ref_fa` on the device's own cache, NMSE against the CPU at most 5e-4 (1e-6 for ±0 and -inf masks); a second FLASH_ATTN on the same plane with the K indices overwritten, as in layers that share KV, equal to `tsr_ref_fa` bit for bit, SW-89), `_tmax8`, `_split1` (every record its own list); `ggml_tesseract_kvbuf` (identification, placement, conversion against independent `tsr_kv_pack_row` images, MSUMs and copies, rule-5 and rule-6 promotion at a small PCAP); `ggml_tesseract_kvstate` (llama.cpp's state API: device to device logits bit-identical, the device's state file read by a CPU context byte for byte) |
| Whole graphs | `ggml_tesseract_graph_qwen_tiny*`, `_gemma_tiny`, `_gemma_tiny_shared*` (the last two layers share the KV of the first two, as in Gemma 4 E4B; one run cuts the op lists in prefill), `_qwen05b` (label `model`), `_qwen35_tiny*`, `_qwen3moe_tiny*`, `ggml_tesseract_mtp_*`, `ggml_tesseract_verify_*`, `ggml_tesseract_gdn_logits_*`, `ggml_tesseract_head_blocks`: exactly one Tesseract split per decode graph (`GGML_SCHED_DEBUG=2`), one launch per decode token and per prefill ubatch, the embd identity, and klcheck's criterion against the CPU on synthetic tiny models (SW-21; `klcheck run --phase m7 -v` on `sim`, `tiny.cmake`, `m7_tests.py`); the MTP tests check every accepted length and rollback bit-identical to plain decoding (§7.11) |
| Tools and traces | `tools_klcheck_*`, `tools_numcheck_*` (FA on, q8_0 KV, rotation off), `tools_selftest_mock*`, `tools_trace_gen_replay`, `tools_trace_replay_device`, `tools_log_selftest`, `tools_klcheck_identity_qwen05b` (label `model`: generate on `sim`, replay the tokens on `sim` and `mock`, `compare --expect-identical`), `trace_v2_roundtrip`, `trace_conv_model` |
| Transport | `transport_protocol` (EX §15 item 2, one binary for every kind: cold and warm boot, image-id mismatch, VERIFY with an injected TCM fault on vkernel, 0–7 args, credit rule, forced `LQ_OVERFLOW` and `RING_RD_BAD`, events while waiting, the debug-aware timeout, TRAP decoding against `addr2line`, the log paths; it runs on `sim` and `mock`, `vkernel` only with `TSR_VKERNEL_SERVER` or `TSR_VKERNEL_SOCKET` set), `transport_kdev` (protocol on mock: timeouts, stale core at open, adapter faults), `transport_xrt_noop` (`TSR_WITH_XRT`) |

### 10.3 tb (pytest) plan

The RTL areas own the RTL benches. The layout of the benches:

| Group | Benches |
|---|---|
| Core and platform | `tb/core` (issue-port tile-only replays of `push.bin`, MVM concurrency under queues); `tb/fastreplay` (C++ tile-only replay, then the full-system bench with Ibex and RVFI lockstep); `tb/platform` (`ap_ctrl_none` adapter; regmap, errors, misaligned access and reset directed tests); `tb/platform/vkernel` (two clocks); `tb/run_all.sh` |
| Units | `tb/dma`, `tb/fp`, `tb/mvm`, `tb/mvmback`, `tb/mvmfront`, `tb/qact`, `tb/spm`, `tb/tech` (unit ports; `tb/spm` and `tb/dma` extended by EX and AT for round robin and banking) |
| Control core, VU, AT, debug | `tb/ctrl` (issue path, two clocks), `tb/vu`, `tb/at`, `tb/rv` (riscv-tests from ITCM, signatures against Spike), `tb/dbg` |

### 10.4 Pass criteria

Common baseline settings (SW-22): `-fa on`, `-ctk q8_0 -ctv q8_0`, `LLAMA_ATTN_ROT_DISABLE=1`, klcheck's other fixed
settings ([klcheck.md](../klcheck.md), Settings fixed in every phase). `offload_kqv` is false in matmul mode
(attention on the CPU) and true in full mode. The CPU reference and floor runs use the same KV type, FA and
rotation settings.

Matmul-mode criteria (the spec criterion; MVMs are issued by RISC-V kernels; `offload_kqv` = false, so attention runs on the
CPU and the device AT ops are verified by the selftest, `traces/at` and `ggml_tesseract_attn`, AT §13.6):

1. `tsr-selftest --device xrt` passes every test, including VU, AT and errors, with sim shadow, plus replays of
   `traces/m3`, `traces/vu`, `traces/at`, `traces/ctrl` and `traces/model-v1conv`.
2. 0.5B Q8_0: `klcheck run --device xrt` generates; `klcheck run --device sim --tokens` replays the same tokens;
   `klcheck compare --expect-identical` must report every row bit-identical (8 prompts × 64 tokens).
3. 32B Q4_K_M: `klcheck all --device xrt` meets the KL criterion (mean KL at most 3 × floor, top-1 at least 80 %,
   [spec](../spec.md) §17).
4. GDB smoke on the card (DB): halt, register read, breakpoint in a kernel, step, resume, with the launch then
   completing correctly.
5. Recorded only: tokens per second, launches per token, launch wall time.
6. Before the card: the same 0.5B identity on `vkernel` (2 tokens) and items 2 and 3 on `sim`.

Full-mode criteria:

1. Qwen2.5-0.5B Q8_0, Qwen2.5-32B Q4_K_M and Gemma 4 12B (Q8_0 and Q4_K_M) decode with exactly one launch per
   decode token, KV cache on the device (`offload_kqv` = true). Checked with `TSR_STATS` over at least 64
   generated tokens with `GGML_SCHED_DEBUG=1` showing one Tesseract split per decode graph, and "staged chunks =
   0" for Gemma 4 12B at `-ub 512` (R6). Prefill is one launch per ubatch.
2. Logits on the card bit-identical to sim for Qwen 0.5B (8 × 64) and Gemma 4 12B (2 prompts × 16 tokens).
3. KL criterion on the card for Qwen 32B and the Gemma 4 models of item 1, including the 2K and 16K long-prompt
   runs of AT §13.6 (multiple chunks, prefill CAUSAL tiles, SLIDE, the wrapped SWA ring).
4. `test-backend-ops` on sim passes every supported op, and `ggml_tesseract_attn` passes.
5. Recorded only: DDR utilization U ([spec](../spec.md) §14) with the H/G/F/E split; the Gemma 4 12B KL with
   the CPU reference built with and without `GGML_GELU_FP16` (the residual of GELU16, VU §6.9).

Hybrid models (Gated DeltaNet, §7.10) and MoE models (§7.9) use the same full-mode criteria: one launch per decode token,
logits bit-identical between sim and card, and the KL criterion. Multi-token prediction adds `klcheck all
--verify-k K --verify-identical` (§7.11). A prefill ubatch above about 450 tokens (Qwen2.5) or 240 (Gemma 4) is
cut into several launches by the 256-line preload area, and a FLASH_ATTN of more than 680 tokens runs on the CPU
(SW-43).

---

## 11. Release configuration

| Item | Value |
|---|---|
| Tesseract Core capability | 1.0 (`VERSION` 0x0000_0100; EX §7.5) |
| Tesseract Basis | 1.0.0, Basis ABI 1 (§6.9) |
| Image | `tsr_u200_k4`: four channels, `T_MAX` 4, VU with 8 front, 8 EXP and 4 back lanes behind two lane-pipe stages, `CAPS` 0x121 (VU, ATT, GDN), core clock 200 MHz, `clk_rv` 100 MHz (CC §14) |
| Firmware kinds | the 21 kinds of §7.2 (MUL_MAT_SCALE and MUL_MAT_SOFTCAP are declared but need `CAPS` bits 13 and 14, which the image does not have; TOPK, DQROW and VMASK are not in the image) |
| Firmware size | ITCM at most 96 KiB of 128 KiB, DTCM at most 56 KiB of 64 KiB (§7.8) |
| Simulator | `TSR_SIM_CARD=k4full` with `TSR_SIM_GDN=1` models the image (§4.5, §4.9) |
| Execution modes | Matmul mode and full mode ([design/README.md](README.md) §2) |

---

## 12. Interfaces with other areas

| Area | SW assumes from it | SW provides to it |
|---|---|---|
| EX (execution model) | the host CSR map (EX §5); the launch protocol (LQ / LCQ, descriptor and record formats, STATUS precedence, runtime contract R1–R10, cancel and stop watermarks, EX §4); image format, 64-bit image id, loader with VERIFY and warm restart (EX §3); `SOFT_RESET` (EX §8.5); unit ids and the opcode table (EX §7); error codes 1–8, 11–13 and the check order (EX §8.1); PERF (EX §6); the `ap_ctrl_none` adapter (EX §9); the transport API (EX §10); the DRAM map (EX §11) | the simulator models of every EX register and ring (deterministic advance); the SDK runtime; the base kernels `graph`, `replay`, `selftest`; the trace format and the trace sets; the table generator |
| VU | encodings, checks and numerics of VOP, RED, ROPE, TOPK, SOP, DQROW and GDN (VU §5–§6, §7.12), including GELU16 and GEGLU16 and the RoPE angle recurrence; AT's vector ops on the VU (VU §5.11); the `tsr_ref_math` algorithms with coefficients as bit patterns; per-op cycle model | `tsr_ref_math.c` and its exhaustive tests; VU semantics in sim; `traces/vu`; kernels and the generated `vu.hpp` builders; host-computed constants (TS, M, 1/n, eps) |
| AT | the KV plane layout and its C conversion (AT §4); KVST, ATTQK, ATTPV and the vector ops (AT §5–§6); the normative FA sequence (AT §6.6); the MSUM format (AT §7.3); the `supports_op` contract (AT §2.6); the SET_ROWS_KV and FLASH_ATTN argument layout (AT §12) | SET_ROWS_KV and FLASH_ATTN kernels; KV buffers in the backend; the MSUM producer and f32 mask copies; AT semantics in sim; `traces/at`; `ggml_tesseract_attn` |
| CC | Ibex configuration and CSR list; the RISC-V address map and control window (CC §4–§5) including `ISSUE_PIDX`; TCM sizes; vector table and boot at `RV_BOOT` + 0x80; reset sequence and SR sequencer; interrupt map; `MTIME`; `RV_FREQ`; RVFI in simulation builds; F_RV | `link.ld` and crt0 matching the map; `tsr_rv` modeling the configuration (and its test bus profile); Spike harness; riscv-tests on RTL; lockstep co-simulation bench |
| DB | debug module behind DMI at CSR 0xC00–0xDFC; halt, resume, step, triggers, `HOLD_ON_HALT`; the GDB RSP server design; tile-level debug semantics; the log record header (DB §6.1); the trap and panic contract (DB §6.3); the debug-aware wait and recovery ladder (DB §4.1, §7.3) | `tsr_transport_csr_read` and `_write`; `tsr_dm.c` and the DB simulator models; `libtsr_dbg` and `tsr-dbg`; `tsr-prof` `trw`; firmware with DWARF, `__tsr_dbg`, `__tsr_crash` and the launch hook; `riscv-none-elf-gdb` in the toolchain; the log decoder |

---

## 13. Decisions

| ID | Decision | Reason |
|---|---|---|
| SW-1 | RISC-V model: the in-house `tsr_rv` interpreter inside `sw/sim`; Spike only as test oracle, on a separate test bus profile at 0x8000_0000 | Keeps the simulator one deterministic C library with exact co-scheduling against the tile model and MMIO hooks, usable inside the llama.cpp backend. |
| SW-2 | Time model: event-driven on core cycles, `fixed` by default, `random:<seed>` for determinism tests, spin and WFI fast-forward | Untimed hides every ordering bug and cycle-by-cycle is slow on large shapes. Randomized durations expose schedule dependence cheaply, and fast-forward bounds the cost of polling joins. |
| SW-3 | Race detector policy: vector clocks; `fatal` in ctest, `off` in model runs, `report` for nightly runs | The 2–3 times cost (estimate) matters only on long model runs. Every kernel and executor path is already covered by ctest with `fatal`. |
| SW-4 | Launches are stateless: results depend only on image, args and DRAM; `SPM_STALE` check; fresh-boot trace replay | Makes every per-launch trace self-contained and replayable on any target from a fresh boot. The op list and tables are small enough to reload per launch. |
| SW-5 | Trace layout: one directory per launch; a single push-ordered `ops/push.bin` with relative tickets; images shared by id | Replay needs the global push order (a dep must not point to an unpushed ticket), and the per-unit view is a filter of it. Relative tickets make traces independent of history, and shared images avoid copies. |
| SW-6 | Trace sets: generate m2, m3 and gen with `tsr-trace-gen` and fixed seeds; convert the recorded model traces as a serial chain (`replay` SERIAL) | Generators and seeds exist, and the error expectations come from the simulator. Model traces are large and error-free and convert exactly, which gives a strong unit-equivalence check against the recorded results. |
| SW-7 | Toolchain: xPack `riscv-none-elf-gcc` (with GDB) in user space, the only supported compiler; Spike and dtc from source; Clang optional | No sudo is needed, versions are pinned and the result is reproducible. The xPack bundle includes GDB for D18, and Clang alone lacks an rv32 compiler-rt. |
| SW-8 | Firmware language rules: freestanding C++20, no heap, exceptions or RTTI, `-Os` (hot path `-O2`, SW-34), libgcc only, no libstdc++ headers; no f32 math in firmware (the host precomputes constants) | Avoids soft-float cost and any firmware-side fp rounding path (the spec fp rules stay in host-computed constants and device ops). C++ templates keep the op builders zero-cost (D6). |
| SW-9 | Image identity (EX §3.2): build-time SHA-256 over the `PT_LOAD` segments, first 8 bytes as a little-endian `uint64`, stored in an 8-byte `.tsr_id`, echoed in HELLO r2 and r3 | Hashing on Ibex would take tens of milliseconds. 64 bits tell builds apart, and the load CRC and VERIFY prove the transfer. Excluding the not-loaded log strings keeps the id stable across log-text edits. |
| SW-10 | Host to RISC-V protocol: EX's LQ / LCQ protocol (EX-1 to EX-5) with the runtime contract R1–R10; up to `LQ_DEPTH` launches outstanding | EX owns the launch protocol. The queue lets the host pipeline launches with posted writes only, and one channel carries launches, answers, HELLO, TRAP and events. |
| SW-11 | Logging: binary records (format-string address plus args) with DB's header (DB §6.1) through EX's log ring; whole-record drop with a counter; runtime filter `LOG_CFG.LEVEL` | No TCM is spent on strings, and the ring is readable even when queues are wedged, unlike a DRAM ring that needs a working DMA. Whole-record drop keeps the stream parseable. |
| SW-12 | Graph execution model: the host builds a pre-resolved ggml-level op list (OPL), computes the dependencies and caches it per graph; firmware kernels per kind | Static tile programs cannot express per-token KV write rows without register-relative addressing (not in the ISA, D14). Parsing ggml views and hazards in firmware would cost TCM and RISC-V cycles. |
| SW-13 | Data placement in full mode: tensors stay in DRAM between ops; per-op SPM scratch from a ring with release tickets; fusion of bias and residual through MVM ACC and of RMS_NORM × weight | DRAM round trips cost about 2 % of a large-model token (estimate) and are simple to make race-free. Fusion through ACC is bit-identical (one commutative f32 add) and removes the most common round trips. |
| SW-14 | Matmul mode: the same `graph` executor with MUL_MAT-only op lists, one launch per split (adopted by EX as EX-29) | One mechanism for both modes, so full mode only adds kinds and supports_op entries. The splits depend only on the MUL_MAT rules of `supports_op`. |
| SW-15 | Index inputs: preload all RISC-V-read graph inputs at launch start with one join: KV write indices, row ids, positions and each KQ mask's MSUM | A join per SET_ROWS or FLASH_ATTN would stall run-ahead behind every queued DMA op, two or three times per layer per token. One join at the start costs one DMA latency. |
| SW-16 | KV rotation: `LLAMA_ATTN_ROT_DISABLE=1` in device runs and in every baseline | The rotation matrix is an F32 graph input that MVM cannot use, and leaving it on the CPU breaks one launch per token. |
| SW-17 | KV buffers (details: SW-47): recognize llama.cpp KV buffers by tensor names at first access; AT layout in a KV heap; converting get and set accessors; context shift unsupported in full mode | llama.cpp allocates the KV cache through the device buffer type without a usage flag ([llama-pin.md](../llama-pin.md), KV cache placement), so names are the stable signal. Converting accessors keep state save and load working whatever layout AT picks. |
| SW-18 | Host protocol: one protocol layer over all kdevs implementing EX §10; `sim` is a kdev with `idle()` | The exact host code path that runs on the card (bind, load, VERIFY, launch, wait, recovery, log) is then tested on every sim and mock run. |
| SW-19 | Test access to queues: `tsr_sim_push` and `tsr_sim_qctrl` test hooks in the simulator library, never in `tsr_transport` | Lets directed queue tests and tile-only sim replays run without firmware, while D15 still holds for every real device path. |
| SW-20 | Planning code: `plan.hpp` is freestanding and compiled for host and firmware | `supports_op` must reject exactly what firmware cannot plan; one source makes that true by construction. |
| SW-21 | Test models: synthetic tiny Gemma 4 (also with shared KV), Qwen2, Qwen3.5-style and Qwen3 MoE GGUFs generated by a gguf-py script at build time; real models for the pass criteria | The smallest real dense Gemma 4 is too slow for ctest on sim. Generated models cover SWA and global layers, per-head norms and GEGLU in seconds without binaries in git. |
| SW-22 | Baselines: FA on, q8_0 / q8_0 KV, rotation off for device and CPU runs in both modes; KV on host (`offload_kqv` = false) in matmul mode, on device in full mode | D17 requires FA on. Fixing the KV type and rotation in both modes keeps baselines constant, so the full-mode delta measures the device attention only. |
| SW-23 | Approximate-math verification: exhaustive 2^32 bit-exact check against an independent numpy model, ulp bounds against double, RTL exhaustive opt-in; GELU16 enumerated over all 65,536 fp16 inputs against the pinned ggml table | Unary f32 functions are fully enumerable, which leaves no gap for coding errors. The numpy model is written from the VU text and shares no code. Double precision is enough for f32 ulp classification. |
| SW-24 | Launch watchdog: 60 s default, counted only while the RISC-V is not halted and no tile stop is active, off in SERIAL mode; the mutex is released between polls; on timeout DB's ladder: dump, `LQ_CANCEL`, then `SOFT_RESET` + warm restart | A wedged queue or a parked hart must not hang llama.cpp, but a GDB user stopped at a breakpoint must not have the device reset under them (D18). The dump preserves the post-mortem before any destructive step. |
| SW-25 | Kernel test builds: a host build (ASan, UBSan) plus an RV build on the simulator, the RV build being the reference | The host build finds logic bugs in milliseconds with sanitizers; only the RV build proves control flow, MMIO timing behavior and ABI. |
| SW-26 | Compute buffers (amended by SW-46): chunks are placed per channel between the weights and the KV planes, sized from the reserve graph; `get_max_size` is at least the largest tensor | A large logits chunk fills a fixed window. Sizing from the reserve graph keeps every chunk on the device (one launch per token) and spreads activation DMA over the channels. |
| SW-27 | Numcheck launches: rely on per-node observation, which gives one launch per node | llama.cpp's `cb_eval` already splits the graph per observed node, so no backend mode means no second code path to keep correct. |
| SW-28 | Card selftest: the `selftest` kernel with an in-process sim shadow comparison | D15 requires kernels. Running the same image on the simulator gives exact expectations for any test parameters without stored golden files. |
| SW-29 | Op-list size: at most 4096 ops per launch (32 KiB of DTCM tables); larger graphs are split by the builder | Decode graphs need on the order of 1,300 to 2,000 ops (estimate). A fixed cap keeps DTCM use static and the executor simple. |
| SW-30 | Timing independence: firmware never lets a timing-dependent read influence pushes, deps, stores or records (only waits); the ring allocator decides deps by count; `ISSUE_PIDX` and `ISSUED_SEQ` are exact | This is what makes programs valid in the sense of §4.7, so card logits can be bit-identical to the simulator. |
| SW-31 | GELU in llama.cpp graphs: GGML GELU and GEGLU map to the VU's GELU16 and GEGLU16 (the CPU's fp16-table form); the exact tanh GELU stays for PyTorch | The pinned ggml CPU defines `GGML_GELU_FP16` and rounds x and the result to fp16 (relative error up to 3.9e-3). Matching that form removes an fp16-level difference on every GEGLU element. |
| SW-32 | Mask summaries (amended by SW-48): the backend writes the MSUM (and, for GENERAL masks, an f32 copy) on `set_tensor` of tensors named `kq_mask`, reserving the bytes through `get_alloc_size`; the FA record refers to it through the preload table | The VU has no f16. The summary keeps the cached op list valid per token and lets ATTQK skip masked bandwidth (AT-14). |
| SW-33 | Unselected graph branches (flag rule amended by SW-40): `GGML_OP_DUP` and non-KV SET_ROWS are unsupported; `graph_compute` skips flagless nodes and makes no launch for an empty split | `ggml_build_forward_select` leaves unselected embedding branches in every decode graph, and the scheduler ignores the flag when it assigns backends; a supported DUP would add a second split and launch per token. |
| SW-34 | Firmware optimization flags: `-Os` for the image; the executor and kind sources (`kernels/graph.cpp`, `kernels/k_*.cpp`) add `-O2` (§6.1) | The cheap records bound the issue rate, and at `-Os` GCC divides by a constant with `divu`. The whole image at `-O2` adds code size for nothing per record. |
| SW-35 | Executor hot path: records read in place through the line-buffered alias after the op's control-window accesses; a streamed list's chunk LD after the record's check and before its handler | Per-record RISC-V cost sets how far the executor runs ahead of the units (§7.3, §7.8). |
| SW-36 | The one-tile block: one fixed SPM block below the preload area (1,024 lines on the U200; `one_block_lines`) for every single-tile record of a DMA-bracketed kind (ADD, MUL, SCALE, RMS_NORM, GLU, ROPE, UNARY; every tile of GET_ROWS and CPY; every SET_ROWS_KV run), in the fast and the general path alike | The DMA queue runs one op after another, so DMA-bracketed tiles are serialized whatever their SPM lines: the fixed block needs no dep, no record and no release and costs no concurrency. 1,024 lines keep MUL_MAT's tiling of the common model sizes, at the cost of a ring block for GLU rows that do not fit. |
| SW-37 | Full-mode dispatch: the kind table holds a fast entry per short kind (`kind_<name>_fast`, a leaf in its common path, KIND_NOT_FAST when it does not take the record, weak "not taken" stubs); the executor calls it with (args, kflags, d) and falls back to the handler out of line, re-reading word0 and word1 | The handler's prologue saves registers and the refusal tests were laid out as fall-through; the leaf form saves cycles on every short record. |
| SW-38 | ITCM budget for the production image: 80 KiB with the full-mode kinds, amended by SW-75; test images are stamped against the whole ITCM | The per-kind leaf fast paths are what meets the issue-rate requirement; CC-3 sized the ITCM for the graph executor, the model families, MoE, sampling and debug formatting. |
| SW-39 | Staged-word reuse and hot entries: `kf::push_s<SKIP, TICKET>`: a fast path's push without the `ISSUE_W` stores of words the previous push of its record staged with the same value (CC §5.1 keeps staged words; WMASK unchanged, so the pushed words are the op's) and without unused `ISSUE_TICKET` reads; host builds check every SKIP claim | Meets the 350-cycle short-record requirement (§7.8) without an OPL change. The SKIP claims are static per call site and checked on the host, and the RISC-V tests check the results bit for bit. |
| SW-40 | Flagless nodes: supports_op does not test `GGML_TENSOR_FLAG_COMPUTE`; DUP, a SET_ROWS whose destination is not a KV-cache tensor and the quantized `tok_embd` GET_ROWS are refused by their own rules; `graph_compute` skips flagless nodes and makes no launch for a split without computed nodes | test-backend-ops asks supports_op before `ggml_build_forward_expand` sets the flag, so the rule would refuse every op there and the test-backend-ops gate item would test nothing. |
| SW-41 | supports_op of the full-mode kinds: supports_op and the builder share one lowering (`tsr_lower_m7`): probe mode (handles 0, the KV shape from `tsr_kv_identify`, no device state), then `opl::check_args` and the kind's planner (`plan_kind` on the device's CFG0–3 and CAPS). The 64-byte rule: every row operand's base and row stride are multiples of 64 bytes, and a result that is a view has rows of whole lines. KV views as llama.cpp builds them with one stream; operands in host memory refused (SW-53). A ROPE weight-placement probe with a freq-factor tensor is planned as a full rotation | The host refuses exactly what the firmware cannot plan or the builder cannot encode, by construction. Refusing the probe would leave a model's `rope_freqs` on the CPU and cut each token into extra launches. |
| SW-42 | Fusion conditions (§9.3 pass 3, SW-13): MUL_MAT + ADD (a bias row: addend stride 0; the residual: its row stride) and RMS_NORM + MUL (a weight of row period 1 or dense per-row weights) when the first node's ggml use count over the whole graph is 1, it is not a view or an output, every operand is in device memory, and no node in between writes the other operand or reads or writes the result (byte ranges); the fused record takes the first node's place | Bit-identical to the unfused device records (`ggml_tesseract_opl_launch`); whole-graph use counts keep consumers in other splits and observed nodes safe. |
| SW-43 | List cuts and FA_NEW: a split is one list unless it reaches 4,096 records or a preload entry does not fit the 256-line area; FA_NEW names the indices of the SET_ROWS_KV that wrote the FA's K plane earlier in the same `graph_compute`, added as a preload entry of the FA's own list when the lists were cut between them | One launch per decode token; the chunk order, hence the output, does not depend on the cuts. |
| SW-44 | Data-dependent FA refusals: the mask producer plans every query tile of every registered reader at `set_tensor` (and on a new reader); a tile the planner would refuse promotes the MSUM to GENERAL with the f32 copy, so the executor never refuses for mask data | Keeps one launch per token. The GENERAL path is implemented by the firmware and `tsr_ref_fa`, and the simulator reads the same MSUM, so it stays the bit-exact oracle. |
| SW-45 | Plan cache: a graph's launched lists are kept under `tsr_graph_key` (node and source addresses, ops, flags, types, shapes, strides, op parameters; not values) salted with a placement epoch (bumped on a buffer free, a weight release, a lazy eviction) | A decode token reruns its list and its inputs change only through the preload table, so almost every token is a cache hit. |
| SW-46 | Compute windows: a LINEAR buffer gets one window in one channel: ch0 IO, else the low 512 MiB of ch1–3 (the channel with the most free bytes), else symmetric WEIGHT-heap reservations of at least 64 MiB between the weights and the KV planes, shared out per channel; `get_max_size` is 512 MiB | Reservations from the WEIGHT heap need no reserve-graph pass, and the logits chunks at `-ub 512` stay on the device (no staged chunks). |
| SW-47 | KV buffers (SW-17 in detail): recognized from the `init_tensor` records (every tensor `cache_<tag>k_l<n>` or `cache_<tag>v_l<n>`, q8_0, 3D contiguous, E % 64 = 0, 64 <= E <= min(K_MAX, 511 · 64), LG < 2^16, plane < 2^28 beats); planes placed at the first data access | ggml decides a buffer's usage after `init_tensor`. Top-down planes stay clear of the bottom-up weights and the compute-heap reservations, and state files are bit-identical to the CPU's (`ggml_tesseract_kvstate`). |
| SW-48 | Mask readers and f32 copies: masks are tensors named `*kq_mask*`, F16 `[n_kv, n_q, 1, 1]`; every FLASH_ATTN registers as a reader (scale, E, D, G) on every build and cached reuse; the reader set only grows until the shape changes or the buffer is freed or reset | The scales are known only from the graph. A growing reader set keeps copy handles stable, so cached lists stay valid while the mask changes per token; rule-3 GENERAL tiles (prefill rows with two runs, holes, a wrapped SWA ring) need the copy although the MSUM is not GENERAL. |
| SW-49 | Device kinds per mode: `TSR_KINDS=m6` restricts the device to MUL_MAT; `klcheck run` and `tsr-numcheck run` set it to m6 for `--phase m6` and keep the caller's value for `--phase m7` (unset: every kind; recorded in report.json `tsr_kinds` and meta.json `info.TSR_KINDS`) | SW-14: matmul mode uses the MUL_MAT-only splits; with the full-mode kinds and `offload_kqv` false the matmul mode phase would be a hybrid of neither mode. |
| SW-50 | Full-mode graph tests: tiny Qwen2 (Q8_0 and pure Q4_K, no Q6_K) and Gemma 4 (Q8_0) from `gen_tiny_models.py`; `m7_tests.py graph` counts launches per `llama_decode` from `klcheck -v` markers and TSR_DEBUG launch lines, splits from `GGML_SCHED_DEBUG=2` | The Q4_K_M mix converts Q6_K to Q8_0 on the device, so a tiny model's floor cannot absorb it. A 4-layer random model's 1-ulp floor is degenerate, so the Gemma threshold is set from injected-error runs (sliding window, RoPE base, eps). |
| SW-51 | klcheck settings for SWA models: `--swa-full off` (default): the SWA ring llama-cli and llama-server use, recorded in a spare header word (0 = older file, read as full-size); the `embd` and floor rows carry llama.cpp's token-row scale (sqrt(n_embd) for Gemma 3, 3n and 4), in klcheck and numcheck | Long Gemma 4 prompts must wrap the ring (full mode gate item 3). Without the scale the Gemma 4 floor is meaningless; with it the embd path equals the token path bit for bit. |
| SW-52 | test-backend-ops coverage of the full-mode kinds: one ctest per op filter (`ggml_tesseract_backend_ops_<op>`) with a minimum count of supported cases (0 for SCALE and SOFTCAP, whose cases have 10- or 100-float rows, and for SET_ROWS, ROPE_SET_ROWS and FLASH_ATTN_EXT, whose tensors are not KV-cache tensors) | Per-op reports; the minimum stops a supports_op regression from passing as "everything unsupported". test-backend-ops cannot place its K, V and dst in a KV buffer (amends AT §13.6). |
| SW-53 | Host memory and the full-mode kinds: only MUL_MAT is staged; supports_op refuses a full-mode op with an operand in a HOSTED buffer or one that will be (ANY with leaf data only: control vectors, LoRA adapters; every buffer under `TSR_FORCE_HOST`), so it runs on the CPU | Correct runs with a control vector and `TSR_FORCE_HOST -nkvo` instead of aborts. Staging all kinds is a lowering of its own for a case that does not occur at `-ub 512 -c 4096`. |
| SW-54 | Versions (§6.9, EX §7.5, EX-12): four numbers: the Tesseract Core capability in `VERSION`; the Tesseract Basis SemVer in `tsr/version.h` (`.note.tsr` word 6, HELLO r6); the Basis ABI; a numerics revision per kind (`opl::kind_rev`, `.tsr_kinds`, HELLO r4[31:16] = the kind table signature), guarded by `fw_kind_revs`. Images name the capability they need (`cap_min`); a core without the control core is refused by `CTL_ID` = 0; the backends refuse an image of an older Basis (major, minor) and route only the booted image's kinds | The capability is the hardware contract and the Basis version the software's, so a software release never regenerates RTL packages. The bit-exact tests compare kernels with references that move with them, so only a digest guard makes (kind, revision) a reproducibility key; recording revisions in reports is enough for reproducibility, and enforcing them would break subset images. |
| SW-60 | Expert tensors: a 3D tensor whose name contains `_exps` is a DEVWEIGHT of n_expert matrices at `W_OFF + e × EXP_STRIDE` per channel (`EXP_STRIDE` = `R × ROW_BEATS`), stored, read back and converted lazily per expert; other 3D tensors stay raw | Every expert MVM keeps the four-channel stream of a dense MUL_MAT and the address is one multiply; per-expert allocations would give no stride. The name is the only information at upload time, and a wrong guess only costs a lazy conversion. |
| SW-61 | MOE_ROUTE: one kind: softmax on the VU, top-k by integer compares on the RISC-V, weights renormalized on the VU, two joins per tile of 64 tokens, ids into the argsort tensor's first line per row, weights packed | No RTL change and no float in the firmware. The VU already holds the exact softmax pieces (RED, EXPSUB, RC), and the unit tests compare the firmware with `tsr_ref_moe_route`, built from the same VU references. |
| SW-62 | MUL_MAT_ID: one MUL_MAT of T = 1 per (token, slot) at `W_OFF + id × EXP_STRIDE`, ids read by the RISC-V from DRAM (a chunk in the one-tile block, one join), optional scale `table[id]` applied by VOP.MULC after the dot product, an out-of-range id gives DRAM_RANGE on the MVM | The op list stays static and cacheable (the ids change per token, the list does not). ggml's order (scale, weight after the down MVM, slot-order adds) keeps the CPU a fair reference. |
| SW-63 | Selection order: the order of the VU TOPK: DAZ'd value under the total order, larger first, NaN first, equal values by the smaller index | A rule that does not depend on a library's sort; the firmware loop and `tsr_ref_topk` agree on ties, NaN, ±0 and subnormals. |
| SW-64 | Graph fusion of the MoE chains: `tsr_build_fusion` takes the router chain (SOFT_MAX to DIV, six nodes) into one MOE_ROUTE record and the scale chain into the MUL_MAT_ID record plus a CPY into the MUL's tensor; supports_op accepts the chain nodes by their own sources; a member that did not fuse aborts at graph_compute naming it | The router is one data-dependent step. The allocator may put the MUL's result over the MUL_MAT_ID's inputs (no in-place write is safe), so the record writes its own tensor and a CPY moves it. supports_op cannot see consumers, so an unusual graph fails loudly instead of reading unwritten tensors. |
| SW-65 | MoE paths in matmul mode and full mode: `--phase m6` (klcheck, numcheck) sets `TSR_KINDS=m6,+MUL_MAT_ID`: the device runs the matrix products and the CPU selects; full mode runs everything in one launch; no path needs the TOPK RTL | The matmul-mode baseline for a MoE model is llama.cpp's own selection, so the full-mode delta measures the firmware selection. Matmul mode needs the firmware image with the MUL_MAT_ID kind but no bitstream change. |
| SW-66 | Prefill and token grouping: per-token expert MVMs (T = 1) in prefill; grouping by expert designed (counting sort of the (token, slot) pairs per ids chunk, MVMs of up to T_MAX rows), not built | Grouping pays when T × k / n_expert is at least T_MAX; it needs a pair list in SPM and a second push order; numerics are unchanged either way. |
| SW-67 | Gated DeltaNet recurrence: `GATED_DELTA_NET` (0x2F) composes the delta rule from existing VU operations (`SOP.EXP`, `VOP.MULC`, `VOP.MUL` + `RED.SUM`, `VOP.SUB`, `VOP.MULCADD`) per head and token, FP32 state in DRAM, head tiles in the SPM; order and rounding are `gdn_ref_step` | No RTL, no new opcode, the numerics are the oracle's, and it runs on any image with the VU. It is the baseline against which the fused op (SW-78) is measured. |
| SW-68 | Convolution kind and conv state: `SSM_CONV` (0x2E) works on channel vectors: the window of `d_conv - 1 + T` rows of C channels, taps as `d_conv` rows of C, summed in tap order, SILU optional; the state cache holds `d_conv - 1` rows of C per cell and belongs to the device; the taps are re-laid out at weight upload | Channel vectors are what the VU lanes take, and the state needs no transpose. llama.cpp only copies, clears and saves whole cells, which are byte copies. |
| SW-69 | Softplus: `SOFTPLUS` (0x2D) is the stable composition `max(x, 0) + log1p(exp(-|x|))` with `log1p` from the odd atanh series: 20 VOPs; UNARY gains SIGMOID (func 5) | The VU has no log. The composition is within 5 ulp of float64 (above 1e-3) and costs one launch's worth of cycles per layer; a new VU function would be RTL for H_v values per layer. |
| SW-70 | Tiling of the recurrence: one ring block per op: two state tiles, the product tile and two token slots; heads in order, the next head's state loads during the current head; tokens in order; S_v a multiple of 16 up to 256; the planner (`plan_gdn.hpp`) is shared by firmware and host | The recurrence is serial in tokens and independent in heads. Prefetching the next head's state hides its DMA, and a block per op gives the planner one decision. |
| SW-71 | The SSM_CONV group and ggml-alloc lifetimes: the pinned graph is `CONCAT`, `SSM_CONV`, `SILU` with a state `CPY`. At the CONCAT two CPY records fill the CONCAT's result buffer with the window (state rows, then token rows); at the SSM_CONV one record reads the window, folds in the SILU and writes the new state rows into the cache | ggml-alloc frees the CONCAT's inputs after the CONCAT and may give their memory to nodes before the SSM_CONV, so the record must not read them there. |
| SW-72 | Recurrent state caches and IMROPE: buffers holding `cache_r_l*` and `cache_s_l*` are device windows (not leaf data); `GET_ROWS`, `SCALE` (zeroing), `CPY` and the group records address them in place; a cache in another backend's memory (`-nkvo`) keeps every op that touches it on the CPU. Text positions are `[p, p, p, 0]` | The graph reads and writes the state in place; one copy per token would be the cost of the state. |
| SW-73 | llama.cpp is unchanged for GDN: no `third_party/patches/0003-*` for it: the backend takes the fused `GATED_DELTA_NET` node, the conv group and the state nodes as the pinned graph emits them | `resolve_fused_ops` keeps the fused node when the backend supports it. The llama.cpp-side facts needed (recurrent memory names, the build_rs form, the mixed input select) are in the pinned source. |
| SW-74 | GDN test method: tiny Qwen3.5-style models (S_v 32 and 128, 3 GDN + 1 attention layer) with klcheck graph tests; `test-tsr-gdn-logits` compares the logits of every step of 1, 2, 5 and 17 token prompts with the CPU's (cosine at least 0.9995, largest difference at most 3 % of the logit range) | klcheck's KL missed a convolution that returned zeros on the tiny model: a random tiny model hides a wrong hidden state. The embedding input path must give an M-RoPE model four section-major positions per token (`[p, p, p, 0]`), as llama.cpp reads them, so that the CPU's embd path equals its token path. |
| SW-75 | ITCM budget of the production image: 96 KiB (98,304 B), with 32 KiB of the 128 KiB ITCM left for sampling and debug formatting (amends SW-38); `tsr-elf` takes 96 KiB as its default and `sw/ctrl/CMakeLists.txt` passes it to the production image as `--budget-itcm`; test images stay stamped against the whole ITCM | The MoE and Gated DeltaNet kinds together exceed 80 KiB. The general paths are already compiled for size, and each fast path is a leaf because a shared body costs 22–29 cycles per record. No feature is removed and no kind loses its fast path. |
| SW-76 | MOE_ROUTE outputs with several tiles: a route of more than 64 tokens writes its ids and weights to a device scratch window (taken from the top of IO, grown when needed, shared by all layers) and two CPY records move them into the ARGSORT and DIV tensors after the record. One tile (decode, up to 64 tokens): unchanged. No firmware change, no kind revision: same words, same output bits | The allocator's layout is not a fact the record can rely on: a record that reads for a long time cannot be told apart from its outputs by address. |
| SW-77 | MoE graph variants: llama.cpp's `qwen3moe` graph (the router group of SW-64, separate gate and up experts with SWIGLU, no per-expert scale, Q and K norms) runs on the existing kinds and lowering | `build_moe_ffn` is the same function for every MoE model, so the fusion pass already finds the group by its nodes; the missing scale group only means the down MUL_MAT_ID is lowered alone. |
| SW-78 | Fused GATED_DELTA_NET path and numerics revision 2: `GATED_DELTA_NET` runs the VU's GDN op (opcode 0x39) per head and token when `CAPS` bit 8 and the VU are present and S_v is at least 64, else the composition of revision 1. The choice is made per launch by `plan::gdn::use_fused(CAPS, S_v)`, which the firmware, `gdn_ref` and the tests share. The record, the planner and the ring block do not change. The kind is at revision 2: its outputs change on a device with the bit (block fixed-point dot products, α folded into k, output from the old state and κ = k · q, NaN rows), and are revision 1's on a device without it and for S_v below 64 | The kind stays one kind with one record, so the host lowering, the planner and the plan cache are untouched and an image with or without the op runs the same op lists. The revision key of [versioning.md](../versioning.md) includes `CAPS`, and the bits on a device with the bit change, so the revision goes up. The composed path is the revision-1 oracle and its digest in `kind_revs.txt` does not move. |
| SW-79 | Test method of the two paths: `fw_kind_revs` has two rows for GATED_DELTA_NET: the run on the simulator's `CAPS` (fused for S_v at least 64; the corpus adds S_v 128 and 256) and a second run on a simulator without bit 8 (the composed path, the corpus of revision 1, so its digest is revision 1's). `fw_gdn_rv` runs with the simulator's `CAPS`, `fw_gdn_rv_composed` without bit 8, and the reference takes the `CAPS`. The `sim` transport has `TSR_SIM_GDN=0` to clear `CAPS` bit 8, for a matched replay of an image without the op | Without the second row the composed path could change unseen, because the simulator reports bit 8 and every default run takes the fused path. |
| SW-80 | The VU, QACT, store and MVM cycle models follow the RTL and the card (§4.5; VU §7.9, §7.13): the VU's cycles come from its lanes (`tsr_sim_opts.vu_timing`, `TSR_SIM_VU_TIMING`; the default is the U200 image's 16 / 16 / 8 lanes, the release image is `k4`) and from latencies measured on the RTL unit (dependent ops, the reduction's partial ring, the two-stage ROPE, the AT vector ops). QACT A32 16 cycles per 32-block + 44, A256 17 per 256-block + 72. Store lines + 110, MVM fill 144. `TSR_SIM_CARD=k4full` is the card configuration | A simulator that undercharges the VU and QACT cannot explain card timing. The unit overlaps independent ops (VU §7.8), so the model slightly overcharges a stream of small independent ops. The default stays 16 lanes because the lane count is an elaboration parameter and the unit tests and trace replays are not the card. |
| SW-81 | Weights as ggml blocks in the PyTorch backend: `tesseract_torch.QuantLinear` holds a Q8_0 or Q4_K weight as a uint8 buffer of ggml blocks `[N, row_bytes]` and calls the custom op `tesseract::linear_blocks`, which uploads the blocks unchanged (`Device.upload_blocks`, cached by storage, format and K like a float weight) and runs `linear` (QACT A32 or A256, f32 result) | The blocks stay byte-identical (the test reads them back) and y equals the numpy spec model of the format; a Q4_K weight is read at 0.5625 bytes per element, a Q8_0 weight at 1.0625. |
| SW-82 | The GDN state on its cache cell: when the ids of the layer's state gather say that every sequence keeps its cell (`rs_s_copy[j]` is the cell the state `CPY` writes, `kv_head + j`), the `GATED_DELTA_NET` record reads the cell and writes the new state back into it (`STATE_IN` = `STATE_OUT` = the cell) | The record already supports `STATE_IN` = `STATE_OUT`; the graph was the obstacle. ggml's result is a new tensor and the cell indices are runtime data, so true in-place is only possible for the identity, which is every decode token, every later prefill ubatch and the first ubatch of the first sequence of a cache. |
| SW-83 | Rollback snapshots as chains of K = 1 records: with `n_rs_seq` = K the `GATED_DELTA_NET` node (K + 1 slots) is a chain of `min(T, K + 1)` records on its one sequence: the first runs the tokens up to `T - nw` and stores slot `nw - 1`, each later record one token from the state before it and stores the next slot | No firmware, record, ABI or numerics change: each record is the K = 1 record of its tokens, so every snapshot and every attention row is bit-identical to a record that stores its tile after each token. |
| SW-84 | The profile dump of the ggml backend ([prof-dbg.md](../prof-dbg.md) §3.1): `TSR_DUMP_PLAN=<dir>` makes the backend write the plan of every graph it builds (nodes with placement, fusion groups, records with arguments, DRAM ranges, weight bytes, ideal MVM cycles and SPM tile plan, the reason each list ended), one line per launch with its PERF deltas, the nodes `supports_op` refused with the reason, and the trace writer's records; it starts and stops the trace writer in the process (a ring window taken from the IO allocator) and turns the PROF bit on. It reads what the builder decided and changes no record, no schedule and no output bit | `cb_eval` cuts splits and changes what is measured (prof-dbg.md §6). A llama.cpp patch is not needed: the splits come from `GGML_SCHED_DEBUG=2` and the refusals from `supports_op`. A tool outside the backend cannot know which node a record came from or what the builder fused. |
| SW-85 | `CONCAT` of two rows as two `CPY` records: `concat(e_norm, h_norm)` along dim 0 (F32, rows of whole lines) is two `CPY` records into the result's rows | Without it the draft graph has two CPU islands and two device launches per step; with it one split and one launch. The concatenated row is what the CPU quantizes block by block, so nothing changes numerically. |
| SW-86 | Multi-token prediction runs on llama.cpp unchanged: the pinned drafter (`--spec-type draft-mtp`) drives a target context with `n_rs_seq` = K and a draft context built from the MTP block on the same device: per step K draft launches, one verification launch with every row an output, one catch-up launch | Everything the graphs need is in the pinned source (SW-73's argument again). |
| SW-87 | The live monitor ([prof-dbg.md](../prof-dbg.md) §7): the process that owns the card publishes a one-page record in shared memory (`/dev/shm/tsr-mon-<pid>`, `sw/transport/tsr_mon.h`: a versioned layout behind a seqlock) from `tsr_opl_dev`, so the ggml backend and `tsr_rt` both publish; it is on by default (`TSR_MON=0` turns it off), updates its counters after every launch and reads the PERF counters at most every `TSR_MON_PERF_MS` (100 ms). `tsr-top` (Python `curses`, `sw/tools/top`) reads the records, the board sensors from `xbutil` and the PCIe link from sysfs, and never opens the compute unit while an owner is alive; `tsr-mon-probe` (attach mode, `tsr-top --probe`) is opt-in for an idle card | A monitor that opens the card itself would fail while a run is on the card (the XRT open is exclusive) and could make a starting run fail. Nothing the owner does changes: PERF does not move between launches, the record is a copy, and klcheck logits are bit-identical with the publisher on and off. |
| SW-88 | The output head in row blocks: a matrix for which T_MAX tokens do not fit one tile of `mm_plan` is stored at upload as `P` in {2, 4, 8} row blocks of `N / P` rows (N a multiple of `16 P`; the smallest `P` that plans `t_tile` = T_MAX), each symmetric over the channels like an expert | A very large vocabulary matrix gets `t_tile` 1, so a verification batch would read the matrix once per token and the MVM would not be weight-bound at T above 1. |
| SW-89 | FA_NEW across a cut with a copy of the K indices: when ggml-alloc puts a node's result on the K indices of a SET_ROWS_KV before a later FLASH_ATTN on the same plane (layers that share KV), a CPY right after the SET_ROWS_KV copies them into a backend area, and a FLASH_ATTN in a later list preloads FA_NEW from the copy (SW-43, §9.3) | Without it the FLASH_ATTN of a sharing layer read overwritten bytes whenever a cut separated it from the SET_ROWS_KV (a ubatch of a few hundred tokens), so Gemma 4 E4B aborted in prefill at `-ub 512`. Re-emitting or reordering would change the list structure of every graph; the copy costs one CPY per index tensor and only in graphs that need it. The result bits do not change. |

---

## 14. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| `tsr_rv` diverges from Ibex in a corner (CSR WARL, misaligned split, interrupt timing) | sim and card differ in firmware behavior | Spike signatures and random lockstep; RVFI lockstep in the full-system bench; the timing rule makes timing differences harmless |
| Host hazard analysis misses a dependency | racy op lists; card results differ from sim | race detector `fatal` on every ctest graph; 8 schedules; executor mutation tests; the `SERIAL` flag and DB's serial mode for bisecting on the card |
| The RISC-V cannot stay ahead on small models (SPM-window reads of op records) | the MVM waits for issue | compact pre-resolved records (at most 1 slot for most kinds); the line-buffered SPM alias; `TRACE_MARK` records and `MVM_EMPTY` to see it |
| llama.cpp graph features outside supports_op (rotation, context shift, new ops after a pin bump) | extra splits or scheduler aborts | R1–R10 checked in the graph ctests with `GGML_SCHED_DEBUG`; rotation off by default; clear errors for unsupported KV operations |
| GELU16 cannot reproduce the CPU's libm `tanhf` table entries exactly | a residual fp16-ulp difference on some GEGLU elements | `ref_math_gelu16` (target of VU §6.5) and the Gemma 4 KL with and without `GGML_GELU_FP16` |
| Toolchain availability (xPack multilib, Spike build dependencies) | blocks the firmware build | the libgcc fallback multilib; Clang + conda as a second route; Spike optional for the build, needed only for `rv_*` tests |
| Simulator speed for large models with RV interpretation | slow tests on sim | fast-forward; race detector off by default; `TSR_SIM_THREADS`; short prompts for the sim identity runs |
| Log ring full during long kernels | lost diagnostics | whole-record drop with counters; host drains during waits; levels; BLOCK mode |
| Composite-kernel sequence changes after baselines | KL baselines silently stale | sequence documented per kernel; a kind revision (§6.9) forces a new baseline |

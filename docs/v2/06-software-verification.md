# Tesseract v2 — area SW: software stack, firmware SDK, simulator, verification and work plan

Status: **design, not implemented** (2026-10-30). One of the six v2 area documents in `docs/v2/`
(EX execution model, VU vector unit, AT attention and KV, CC control core, DB debug, SW this
document). Source-of-truth order: `docs/spec.md` > ISA simulator (`sw/sim`) > the v2 area
documents > `docs/ctrl-riscv.md`. The user decisions D1–D6, D12 and D14–D18
(`docs/ctrl-riscv.md:15-46`) are fixed. Every choice marked V2-SW-n (§13) was made under the user's
delegation. Numbers marked *estimate* are not measured. *confirm* marks a fact about a third-party
tool that must be checked when the tool is pinned.

---

## 1. Scope and summary

Area SW owns everything that is not RTL: the ISA simulator (the bit-exact oracle), the C
references, the firmware SDK and runtime that run on the RISC-V control core, the firmware kernels
and the graph executor, the host stack (transport, ggml backend, `tsr_rt`, PyTorch, tools), the
trace format, and the verification and work plan.

Summary of the design:

- **Simulator.** `sw/sim` gets an in-house RV32IMC + Zicsr interpreter (`tsr_rv`), with Spike as
  its oracle in tests only. It also gets an event-driven model of the issue queues on the
  core-cycle time base (fixed or random schedules), a vector-clock race detector, the VU and AT op
  semantics, and `tsr_ref_math.c`. The host command path (START, CMD_*, ARG*, R0–R15, LI, ADDI,
  END, fetch, CALL) is deleted (D14).
- **Trace v2.** One directory per launch: the firmware image (by id), the launch descriptor, DRAM
  init and expect regions, the DONE record, per-unit ticket records, and the pushed op stream in
  push order (`ops/push.bin`) with relative tickets. Tile-only traces replay through a standard
  `replay` kernel, so they also run on the card without host access to the issue port (D15).
- **Firmware.** `sw/ctrl` is a freestanding C++20 SDK: crt0, linker script, a runtime that
  implements EX's launch contract R1–R10 over the launch queue and completion ring (EX §4), traps
  and panic per DB §6.3, binary logging with DB's record header (DB §6.1) through the log ring, a
  kernel table and the tile/VU header library. Three base kernels exist from M6: `graph` (the
  op-list executor), `selftest` and `replay` (EX §4.6).
- **Graph execution.** The ggml backend turns each split into a pre-resolved op list in DRAM
  (format OPL v1, §7.2) and launches `graph` once. In M6 the list holds only MUL_MAT records; M7
  adds the op kinds that make one decode token one launch (§8). The host computes the dependencies
  between ops. The firmware maps them to tickets and issues ahead, so units overlap (D3).
- **Host.** EX's transport API (EX §10) with one protocol layer (bind, load and VERIFY, boot,
  launch, wait, cancel, stop, soft reset, log) over every kdev: `sim` (in process), `mock`,
  `vkernel` and `xrt` (`ap_ctrl_none` with polling). Weight upload is unchanged.
- **Toolchain.** No RISC-V compiler, Spike or RISC-V GDB is installed on the build host (checked
  2026-10-30). The plan installs the xPack `riscv-none-elf-gcc` (with its GDB) and builds Spike
  and dtc from source, all in user space, no sudo (§9.6).
- **Gates.** M6: on the card, Qwen2.5-0.5B Q8_0 logits bit-identical to `TSR_DEVICE=sim`, and the
  32B Q4_K_M KL criterion met with FA on and attention on the CPU (`offload_kqv` = false, §10.4). M7: Qwen2.5 and Gemma 4 dense run with exactly
  one launch per decode token, with logits bit-identical to sim, the KL criterion met, and the DDR
  utilization measured.
- **Work plan.** P1 references and simulator → P2 SDK and kernels on sim → P3 host and llama.cpp
  end to end on sim → P4 RTL → P5 Verilator full system → P6 card (§11). The software path works
  end to end on the simulator before any v2 RTL exists.

---

## 2. What v2 removes from the software (D14, D16)

| Item | Where today | v2 |
|---|---|---|
| Host START / `CTRL` / `STATUS.BUSY|DONE|ERR` run protocol, `CMD_ADDR_*`, `CMD_COUNT`, `ARG0-7`, `ERR_PC`, `PC` | `tsr_isa.h`, `tsr_sim.c`, transports | deleted in X0 (EX §12); launches go through EX's launch queue and completion ring (§6.4) |
| R0–R15, `LI`, `ADDI`, `END`, register-relative address fields | `tsr_sim.c`, `tsr_trace_gen.c` | deleted; address fields are immediates and ra/rb/rc/rd must be 0 (`docs/spec.md:416`) |
| Command fetch, CMD region use for programs, `CALL` | `tsr_sim.c`, `tsr-lower.cpp`, `tsr_rt.cpp` | deleted; the CMD region becomes the OPL region (§7.2) |
| `tsr_transport_exec`, `tsr_exec_result`, `kargs`, `kdev::start/wait/settle`, `regs_*`/`kds_*`, `TSR_XRT_CTRL=run|regs` | `sw/transport` | replaced by load / launch / wait (§9.1) |
| MUL_MAT instruction emission and the program cache (`tsr_mm_emit`, `tsr_prog_cache`) | `tsr-lower.cpp` | tiling moves to `plan.hpp`, shared by host and firmware; emission moves to the firmware MUL_MAT kernel; the op-list cache replaces the program cache |
| Trace v1 (`cmd.bin`) writing and replay | `tsr_trace.c`, tools, tb | trace v2 (§5); a v1 reader survives only inside the converter |
| `tsr_xrt_selftest` (host-driven programs) | `sw/tools` | `tsr-selftest`, whose device programs are the `selftest` kernel (D15) |
| `tsr_cmdproc`/`tsr_fetch` benches, the M2 core suite, `cmd.bin` fastreplay | `tb/` | deleted or rewritten (§10.3) |

`TSR_DEVICE=none` and the weight path (`tsr-weights`, `tsr-wlayout`, `tsr-alloc`, symmetric
allocation, load-time Q8_0 conversion) are unchanged. Every deletion of this table happens at once
in X0, the first M6 commit (EX §12, D16); tests that depend on a deleted part return with their
v2 replacement (§10.2).

---

## 3. Repository layout (software, v2)

```
sw/
├── tables/                  # tsr_tables.py: the single table source (EX/CC/DB/VU/AT own rows, V2-EX-30)
│                            # gen_tables.py: generator (SV package, C header, mmio/vu/at.hpp, tb Python)
├── sim/                     # libtsr_sim, libtsr_ref (C, host only)
│   ├── include/             # tsr_isa.h (+ generated tsr_tables.h), tsr_ref.h, tsr_ref_math.h, tsr_sim.h, tsr_trace.h
│   ├── tsr_ref.c            # existing references (fp helpers, layouts, QACT, dot products, dequant)
│   ├── tsr_ref_vu.c         # VU op references: VOP, RED, ROPE, TOPK, SOP, DQROW (VU §5-§6)
│   ├── tsr_ref_at.c         # AT references: KV planes, KVST, ATTQK, ATTPV, vector ops, FA (AT §13.1)
│   ├── tsr_ref_math.c       # EXPK, SIGMOID, SILU, GELU (tanh, fp16 form), TANH, SIN, COS (algorithms: VU §6)
│   ├── tsr_sim.c            # core state, host CSR port (EX §5), loader, LQ/LCQ/log rings, SOFT_RESET, PERF,
│   │                        # op handler table, tile units (LD, ST, QACT, MVM)
│   ├── tsr_vu.c, tsr_at.c   # VU and AT unit semantics: checks, extents, execution, cycle models
│   ├── tsr_q.c              # issue port, dispatcher, checker order, unit queues, tickets, deps, SYNC, errors, time base
│   ├── tsr_sim_int.h        # internal interface between the sim's files (handler table, accessors, hooks)
│   ├── tsr_rv.c, tsr_bus.c  # RV32IMC+Zicsr interpreter; RISC-V address map and MMIO
│   ├── tsr_dm.c             # debug module model (area DB's contract)
│   ├── tsr_sched.c          # random schedules and fast-forward (P1.8; the fixed time base is in tsr_q.c)
│   ├── tsr_race.c           # race detector
│   ├── tsr_trace.c          # trace v2 read / write / capture / replay
│   └── tests/
├── ctrl/                    # firmware SDK (RISC-V), separate CMake project (ExternalProject)
│   ├── cmake/riscv32.cmake  # toolchain file
│   ├── include/tsr/         # kernel.hpp, mmio.hpp (generated), vu.hpp, at.hpp, plan.hpp, opl.hpp, log.hpp,
│   │                        # backend_dev.hpp, backend_host.hpp
│   ├── rt/                  # crt0.S, link.ld, runtime.cpp, trap.cpp, log.cpp
│   ├── kernels/             # graph.cpp (executor), k_mulmat.cpp, k_norm.cpp, k_elem.cpp, k_rope.cpp,
│   │                        # k_glu.cpp, k_rows.cpp, k_kv.cpp, k_fattn.cpp, selftest.cpp, replay.cpp
│   ├── ref/                 # host C++ references of composite kernels (tests)
│   └── tools/tsr-elf/       # image id, kernel table, log-format table, size report
├── transport/               # tsr_transport (C API), protocol layer, kdevs: sim, mock, vkernel, xrt
├── ggml-tesseract/          # backend: weights (unchanged), tsr-opl.* (op-list builder), tsr-kvbuf.* (M7)
├── rt/, torch/              # tsr_rt on op lists; PyTorch unchanged above tsr_rt
└── tools/                   # tsr-selftest, tsr-trace-gen/-replay/-conv, tsr-log, klcheck, numcheck, tsr_bw
```

`third_party/` gains no RISC-V tool sources. The toolchains live outside the repository (§9.6),
and their versions are recorded in `docs/v2/toolchain.lock` (created in P1.11).

---

## 4. Simulator

### 4.1 Model overview

The simulator stays one C library, linked into the backend, the tools and the tests. It models:

- the **core** (core clock domain): SPM, per-channel DRAM, the host CSR port (EX §5: CSRs, the LQ /
  LCQ / log rings, `Q_*`, PERF, DMI), the dispatcher, the checker, the unit queues, and the units
  DMA, QACT, MVM and VU with AT's ops on them (EX §7.3), plus SYNC;
- the **control core** (RISC-V clock domain): `tsr_rv`, ITCM and DTCM, the control window, the
  core window, the SPM window and the debug module.

Functional results come only from the C references (`tsr_ref*.c`), as today
(`docs/spec.md:1086` rule 3). An op's effects (SPM and DRAM writes, scalar-register writes) are
applied atomically at its modelled completion time. For race-free programs this choice is
invisible (`docs/ctrl-riscv.md:1326-1333`).

### 4.2 RISC-V model `tsr_rv` (V2-SW-1)

- **ISA:** RV32IMC + Zicsr + Zifencei, M-mode only (CC's configuration). An unimplemented encoding
  or CSR raises an illegal-instruction trap (`mcause` 2, `mtval` = the instruction bits).
  `fence.i` flushes nothing (there is no cache) and retires in 1 cycle.
- **Misaligned load/store:** split into two aligned accesses in address order, like Ibex
  (`docs/ctrl-riscv.md:393-395`), and not atomic. If the second half faults, the trap reports the
  original address.
- **CSRs:** exactly CC's list for the pinned Ibex configuration. Expected: `mstatus`, `misa` (read
  only), `mie`, `mip`, `mtvec` (vectored only: MODE reads 1, base 256-byte aligned), `mscratch`,
  `mepc`, `mcause`, `mtval`, `mcycle[h]`, `minstret[h]`, `mcountinhibit`, `mhpmcounter3-31[h]`
  and `mhpmevent3-31` (read 0), `mvendorid`/`marchid`/`mimpid`/`mhartid`, `tselect`,
  `tdata1-3` (2 execute triggers), `dcsr`, `dpc`, `dscratch0-1` (debug mode only). CC's document
  is authoritative.
- **Interrupts:** `irq_fast[0..5]` and `irq_timer`, as in `docs/ctrl-riscv.md:433-455` and as
  restated by EX/CC. Entry is vectored at `mtvec` + 4 × cause. WFI wakes on (mip & mie) ≠ 0
  whatever `mstatus.MIE` is.
- **Bus:** the CC address map (CC §4.1). An unmapped access raises an access fault (instruction
  1, load 5, store 7). The control-window and core-window offsets come from the generated table
  (§4.8). The register semantics are CC's and EX's (issue port, credits, `ISSUE_PIDX`,
  `DONE_SEQ`, error records, `QCTRL`, `QSTATUS`, `WAIT_*`, the LQ / LCQ / log registers of EX
  §4.5, `MTIME`), including the core-window rules (CC §5.3) and the SBA refusals (CC §4.3).
- **Test bus profile** (used only by `rv_isa`, `rv_arch`, `rv_random` and riscv-arch-test, never
  by firmware): RAM at 0x8000_0000 (size from the ELF), HTIF `tohost` / `fromhost`, signature dump.
  riscv-tests `-p` and riscv-arch-test link at 0x8000_0000 and end through HTIF, and Spike puts its
  debug module at 0x0 and its boot ROM at 0x1000, which overlaps CC's ITCM; the firmware profile
  is CC §4.1. Random lockstep programs are placed at 0x8000_0000 in both Spike and `tsr_rv`.
- **Timing** (schedule only, never architectural; *estimate*: `TSR_RV_CPI_DEFAULT` in `tsr_rv.h`
  and the window latencies in `tsr_bus.h`; P1.8's `tsr_sched.c` varies them): ALU and
  CSR 1 cycle; load/store to ITCM/DTCM 2; taken branch or jump 3; `mul` 3; `div`/`rem` 37;
  control-window access 4 RV cycles plus the modelled crossing (2–3 core cycles); SPM-window
  access 10–14 RV cycles. The RV-to-core clock ratio comes from `TSR_SIM_RV_RATIO`, default
  166.7 / 276.3 = 0.603 (the k4 core clock).
- **Debug hooks:** halt/resume/step, abstract register access, program buffer and triggers come
  from `tsr_dm.c`, which implements area DB's module contract. GDB over the DMI window therefore
  works against `TSR_DEVICE=sim`.
- **Size:** ≈ 2 K lines (*estimate*).

### 4.3 Core model

`tsr_q.c` implements the issue rules of EX §7–§8 and CC §7–§9 exactly: credits, tickets {unit,
seq 28 b}, `ISSUE_PIDX`, up to 2 deps, SYNC NOP / FENCE / QCLEAR, QABORT, hold sources (with DB's
hard hold, CC §9.2), checks at dispatch in EX's order (ILLEGAL_OP → BAD_DEP → BAD_FMT → K_RANGE →
T_RANGE → OPERAND → SPM_RANGE → DRAM_RANGE), poisoned entries, error state at the head, invariants
I1–I3, and per-unit error records with push indices. Queue depths, unit ids and the opcode → unit
map come from the generated table and the config, so they change without code changes.

The existing unit code paths in `tsr_sim.c` (LD, ST, QACT, MVM and their checks) are kept; only
their entry changes, from "cmdproc executes instruction i" to "the queue head starts entry e". VU
and AT ops are new (§4.5). The host CSR port serves EX's map (EX §5): `CTRL` / `STATUS`, the
loader with receive CRC and VERIFY, `RV_BOOT`, the LQ / LCQ / log rings and watermarks,
`SOFT_RESET` (EX §8.5, CC §12.2), PERF v2, `Q_*`, `SREG`, the DBG block and DMI (`tsr_dm.c`).
Reads have no side effects except EX's PERF LO snapshot (`docs/spec.md:693`).

### 4.4 Scheduler and time base (V2-SW-2)

- **Time** is in core cycles (u64). The event kinds are unit-op completion, dispatcher crossing,
  RISC-V progress (RV cycles × ratio), the LQ / LCQ / log crossings, and the `MTIMECMP` timer.
- Ties are broken deterministically by (time, event-kind priority, unit id, sequence).
- **`TSR_SIM_SCHED`:**
  - `fixed` (default): op durations come from the existing cycle model and crossings take
    2 cycles.
  - `random:<seed>[,<min>,<max>]`: every op duration is multiplied by a factor drawn uniformly
    from [min, max] (default 0.25–4); crossings take 2–8 cycles; the RV CPI of each instruction
    varies by ±1 cycle. The generator is xoshiro128**, seeded from `<seed>`.
- **Fast-forward.** WFI jumps to the next event that sets a pending, enabled interrupt. A spin
  also jumps. A spin is the same timing-dependent register (`DONE_SEQ[u]`, `ISSUE_CREDIT`,
  `QSTATUS`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) read 4 times in a row with an unchanged value
  and no store, push or other MMIO access in between. The hart then jumps to the next event that
  changes any of these registers. This bounds simulation time when a 32B MVM (311 K core cycles,
  `docs/ctrl-riscv.md:813`) is joined by polling. Poll counts are therefore not architectural, which
  the determinism rules already require (§4.7).
- **Host interaction:** `tsr_sim_advance()` runs until a host-visible event (an `LCQ_WR` change,
  log words, RV halted or trapped, `RV_RESET_DONE`, `SR_BUSY` falling, a `QIDLE` change) or until a
  cycle budget runs out (§4.8).

### 4.5 Device op semantics and references

| Ops | Semantics source | Reference code | Status |
|---|---|---|---|
| LD, ST, QACT, MVM, NOP, FENCE | `docs/spec.md:394-473`, `:1021-1076` | `tsr_ref.c` (unchanged) | kept; register fields must be 0; END/LI/ADDI/CALL and opcodes 0x04–0x09 are ILLEGAL_OP |
| VOP (incl. GELU-tanh, GELU16, GEGLU16, TANH), RED (segmented, post RCP / RMS), ROPE (angle recurrence), TOPK, SOP, DQROW | VU §5–§6 (base: `docs/spec.md:288-321`, `:437-472`) | `tsr_ref_vu.c` (ops), `tsr_ref_math.c` (approximations) | in the simulator: `tsr_vu.c` (P1.4; TOPK and DQROW from M6 in the simulator, their RTL is M8). VMASK is M9 |
| KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM | AT §5–§6 (vector ops executed by the VU, VU §5.11) | `tsr_ref_at.c` (AT §13.1) | in the simulator: `tsr_at.c` (P1.5) |
| QCLEAR, QABORT, hold control entries | area EX | `tsr_q.c` | new |

**`tsr_ref_math.c`.** Area VU fixes each algorithm: range reduction, polynomial or table, the
coefficients as f32 bit patterns, and the evaluation order. SW implements the file under these
rules:

1. Every f32 operation is a call to `tsr_fadd`/`tsr_fsub`/`tsr_fmul`/`tsr_fdiv` (DAZ, FTZ,
   canonical NaN, `docs/spec.md:1025`). Everything else is integer bit manipulation. No libm, no
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

   The names are VU's (VU §6.1). RECIP_APPROX does not exist: every reciprocal is `tsr_frecip`.
3. The same file compiles for RV32 (soft-float libgcc is never needed, because the `tsr_f*`
   helpers are what get called) and for Python ctypes (cocotb).
4. Each function's verification is in §10.1 (V2-SW-23).

### 4.6 Race detector and the launch-stateless rule (V2-SW-3, V2-SW-4)

- **Happens-before:** vector clocks over the agents {each unit, the RISC-V, the host}. The edges
  are those of `docs/ctrl-riscv.md:1336-1354` without CALL and host runs:
  - queue order;
  - dependencies;
  - RISC-V program order;
  - a RISC-V push after a completed store;
  - the RISC-V observing a completion (a `DONE_SEQ` read, the WAIT interrupt);
  - FENCE and QCLEAR;
  - the launch protocol (EX §2.3): a host DRAM write completed before `LQ_PUSH` of launch s
    happens before every op of s (H1); every op of s happens before the host's read of s's
    answer record and everything the host does after it (H2, the runtime joins before it posts).
- **Access sets per op** come from its operands (VU and AT ops from their documents). RISC-V SPM
  accesses are tracked per byte. Issue-bank scalar registers are tracked as 16 locations.
- **Launch-stateless rule (V2-SW-4):** a launch's observable results depend only on (image,
  descriptor args, DRAM contents). The detector reports `SPM_STALE` when an op or the RISC-V reads
  an SPM line that nothing wrote since the runtime popped the descriptor. Dependence on DTCM state left by an
  earlier launch has no mechanical check. It is caught instead by the trace round trip (§10.1):
  every captured launch trace is replayed on a freshly booted simulator and must match.
- `TSR_SIM_RACE=off|report|fatal`. ctest uses `fatal`. Model runs default to `off` (the 2–3×
  slowdown is an *estimate*); the nightly model run uses `report`.

### 4.7 Determinism (what is bit-exact)

The definition follows `docs/ctrl-riscv.md:1419-1437` and `docs/spec.md:821`, restated for
v2. A program (image + descriptor args + DRAM inputs) is **valid** if it is race-free and the
following are the same under every schedule:

- the per-unit op streams (words, push flags, deps as relative tickets);
- the LCQ records (r0–r6; r7 DURATION is timing);
- the RISC-V's SPM store stream;
- the final written DRAM and written SPM lines.

The check runs ≥ 8 schedules (`fixed` + 7 random seeds) and compares all four. For a valid program
the final state is unique. RTL and the card must reproduce:

- the expect DRAM regions;
- the DONE record (r0–r6);
- the per-unit completion counts;
- the per-unit error records.

Never compared: poll counts, `MTIME`, `PERF_*`, record word r7, completion records (REC) and log
content.

Firmware rule (V2-SW-30): no value read from a timing-dependent register (`DONE_SEQ`, credits,
`QSTATUS`, `MTIME`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) may influence pushed words, deps, SPM
stores or records. Such values may only end a wait loop. `ISSUE_PIDX` and `ISSUED_SEQ` are not
timing-dependent (only the RISC-V's own pushes move them). This holds by construction for every
kernel in §7.

At the llama.cpp level, the same image and the same inputs give bit-identical logits on `sim`,
`mock`, `vkernel` and `xrt`. That is the M6 and M7 end-to-end check.

### 4.8 C API (v2) and the register table

The register and encoding table `sw/tables/tsr_tables.py` (plain Python data, no dependencies) is
the single source (V2-EX-30). One generator (`sw/tables/gen_tables.py`, P1.1) produces from it the
RTL package (`rtl/core/tsr_tables_pkg.sv`), the C header (`sw/sim/include/tsr_tables.h`, included
by `tsr_isa.h`), the firmware headers (`sw/ctrl/include/tsr/{mmio,vu,at}.hpp`) and the Python
module for cocotb (`tb/common/tables.py`), with `--check` in lint. EX, CC, DB, VU and AT own the
table rows; SW owns the generator.

```c
typedef struct tsr_sim_opts {
    const char * dram_dir;   // file-backed DRAM (as today)
    uint32_t     threads;    // MVM worker threads (results independent of it)
    const char * sched;      // "fixed" | "random:<seed>[,<min>,<max>]"
    double       rv_ratio;   // F_RV / F_CORE, default 166.7/276.3
    uint32_t     race;       // 0 off, 1 report, 2 fatal
    void *       log;        // FILE* for TSR_SIM_LOG, or NULL
} tsr_sim_opts;

tsr_sim *  tsr_sim_create(const tsr_config * cfg, const tsr_sim_opts * opts);
void       tsr_sim_destroy(tsr_sim * sim);
const tsr_config * tsr_sim_config(const tsr_sim * sim);
uint8_t *  tsr_sim_dram(tsr_sim * sim, uint32_t ch);
uint8_t *  tsr_sim_spm(tsr_sim * sim);                       // debug view
uint64_t   tsr_sim_dram_limit_beats(const tsr_sim * sim);

uint32_t   tsr_sim_csr_read (tsr_sim * sim, uint32_t off);   // EX §5 map: CSRs, rings, Q_*, DBG, DMI
void       tsr_sim_csr_write(tsr_sim * sim, uint32_t off, uint32_t val);

#define TSR_SIM_EV_LCQ     1u   // LCQ_WR changed (a record was posted)
#define TSR_SIM_EV_LOG     2u   // log words available
#define TSR_SIM_EV_HALT    4u   // RV halted (debug) or parked after a TRAP record (incl. panic)
#define TSR_SIM_EV_RESET   8u   // RV_RESET_DONE rose or SR_BUSY fell
#define TSR_SIM_EV_QIDLE  16u
#define TSR_SIM_EV_BUDGET 32u   // max_cycles elapsed
uint32_t   tsr_sim_advance(tsr_sim * sim, uint64_t max_cycles);
uint64_t   tsr_sim_now(const tsr_sim * sim);                 // core cycles

// Test hooks: never reachable through tsr_transport (D15).
uint32_t   tsr_sim_push(tsr_sim * sim, const uint32_t w[8], uint32_t dep0, uint32_t dep1,
                        uint32_t push_val);                  // as an ISSUE_PUSH; returns the ticket
uint32_t   tsr_sim_qctrl(tsr_sim * sim, uint32_t bits);      // QCTRL write; QCLEAR returns its ticket
int        tsr_sim_run_until_idle(tsr_sim * sim, uint64_t max_cycles);

// Launch capture (trace v2) and race reports.
void       tsr_sim_capture_launch(tsr_sim * sim, int on);    // capture each descriptor .. answer record
int        tsr_sim_capture_take(tsr_sim * sim, tsr_trace2 * out);   // 0 = one launch taken
size_t     tsr_sim_race_reports(const tsr_sim * sim, const tsr_race_report ** out);

// RISC-V debug view
uint32_t   tsr_sim_rv_reg(const tsr_sim * sim, unsigned i);  // x0..x31
uint32_t   tsr_sim_rv_pc(const tsr_sim * sim);
uint64_t   tsr_sim_rv_instret(const tsr_sim * sim);
```

`tsr_config` loses `nireg` and `loop_depth` and gains `itcm_bytes`, `dtcm_bytes`, `ctl_id` and the
queue depths.

### 4.9 Environment (simulator and `sim` transport)

| Variable | Values | Default |
|---|---|---|
| `TSR_SIM_SCHED` | `fixed`, `random:<seed>[,<min>,<max>]` | `fixed` |
| `TSR_SIM_RV_RATIO` | F_RV / F_CORE, 0.05–4 | 0.603 (166.7 / 276.3) |
| `TSR_SIM_RACE` | `off`, `report`, `fatal` | `off` (ctest: `fatal`) |
| `TSR_SIM_LOG` | `stderr` or a file: RV retire lines (optional), push / start / done events | off |
| `TSR_SIM_THREADS`, `TSR_SIM_DRAM_DIR`, `TSR_SIM_TMAX`, `TSR_SIM_CH_GIB` | as today | as today |
| `TSR_TRACE` | directory: one trace v2 per launch (`<dir>/<seq 6 digits>/`) | off |

---

## 5. Trace format v2 (V2-SW-5)

### 5.1 Directory

```
<trace>/meta.json
<trace>/ops/push.bin                 # pushed entries in push order (5.2)
<trace>/init/ch<c>_<off>.bin         # DRAM before the launch (as v1)
<trace>/expect/ch<c>_<off>.bin       # DRAM written by the launch (as v1)
<trace>/expect/tickets.bin           # per-unit counts and error records (5.3)
<set>/images/<image id hex>.elf      # firmware images, shared by every trace of a set
```

`meta.json` (version 2):

```json
{ "format": "tesseract-trace", "version": 2,
  "kind": "launch",                         // "launch": image + descriptor; "ops": tile-only push stream
  "config": { "num_ch":4, "beat_bits":512, "lanes":128, "t_max":8, "k_max":32768,
              "spm_bytes":4194304, "vl":16, "nsreg":16, "rd_outstanding":16,
              "ch_bytes":17179869184, "addr_beat_w":28, "caps":N, "ctl_id":N, "cfg3":N,
              "itcm_bytes":131072, "dtcm_bytes":65536,
              "q_depth":[8,4,4,16,0,4] },                              // units 0-5; unit 4 absent in M6
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

`perf` is informational, as in v1 (`docs/spec.md:1061`).

### 5.2 `ops/push.bin`

The file is an array of 48-byte little-endian records, one per entry the RISC-V sent through the
issue FIFO, in push order:

```c
typedef struct tsr_trace_push {   // 48 bytes
    uint32_t w[8];     // kind 0: instruction words as pushed (masked words are 0)
                       // kind 1: w[0] = QCTRL value; kind 2: w[0] = TRACE_MARK value
    uint32_t dep[2];   // relative tickets {unit[31:28], rel_seq[27:0]}; unit 15 = none
    uint32_t push;     // the ISSUE_PUSH value (word mask [15:8], REC bit0); 0 for kinds 1, 2
    uint32_t kind;     // 0 op, 1 queue control (QCTRL), 2 trace mark
} tsr_trace_push;
```

`rel_seq` = (seq − base_u) mod 2^28. `base_u` is `ISSUED_SEQ[u]` at kernel entry (EX §4.7 R5;
kind `launch`) or when the replay kernel started (kind `ops`). The per-unit op stream
compared by the determinism check (§4.7) is the projection of `push.bin` onto one unit. A
QCLEAR's ticket counts on the SYNC unit.

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

- DRAM outside the init regions is zero, as in v1 (`docs/spec.md:1062`). The replayer loads the
  init regions.
- **Kind `launch`:** boot the image (HELLO), push the recorded descriptor, wait for its DONE
  record.
- **Kind `ops`:** write `push.bin` at `ops.load` and launch the standard `replay` kernel (§7.7)
  with (handle, count, flags 0; flags SERIAL for `traces/model-v1conv`).
- **Compared, everywhere:**
  - the DONE fields;
  - every expect region, bit-exactly;
  - the set of written DRAM beats, which must equal the union of the expect regions;
  - the set of read DRAM, which must lie inside the init regions or have been written earlier in
    the launch;
  - the per-unit completion counts (host `Q_DONE[u]` deltas, EX §5.4);
  - the error records (on the card, the DONE record plus `Q_ERR_*`).
- **Also compared where the push stream is observable** (sim always; Verilator through an issue-FIFO
  monitor): the per-unit projections of `push.bin`.
- **RTL tile-only replay:** a Verilator bench pushes `push.bin` straight into the issue FIFO with no
  RISC-V (`docs/ctrl-riscv.md:1415-1417`). It separates core concurrency bugs from RISC-V
  integration bugs.

### 5.5 Capture

`TSR_TRACE=<dir>` dumps one `launch` trace per launch (descriptor to answer record). The capture
follows v1's rules:

- init = DRAM read before it was written, copied at first read;
- expect = DRAM written, taken at the answer record;
- the image is written once per id into `<dir>/images/`.

`tsr-trace-gen` writes `ops` traces directly (random tile programs with dependencies, concurrency
and error injection). It runs each on the simulator to fill `done`, `expect` and `tickets`.

### 5.6 Migration of the 778 v1 traces (V2-SW-6)

| Set | Count | Facts (counted 2026-10-30) | v2 plan |
|---|---|---|---|
| `traces/m2` (git) | 172 | 164 use LI/ADDI, 163 register-relative operands, 93 end in an error (44 ILLEGAL_OP) | **regenerate** with `tsr-trace-gen` v2 (`regen.sh`, same seeds and configurations) as `ops` traces. Random register bases become random absolute immediates. v1-only endings (fetch beyond the DRAM limit, missing END, nonzero `CMD_ADDR_HI`) are dropped. v2 endings are added: BAD_DEP, DEP_FAILED chains, ABORTED after an error state, QABORT/QCLEAR |
| `traces/gen` (git) | 32 | all register-relative, 5 errors | regenerate (as m2) |
| `traces/m3` (disk, regenerable) | 284 | 280 register-relative, 156 errors | regenerate (MVM boundary cases and the 23 MVM error kinds keep their generator code paths) |
| `traces/model` (disk, regenerable) | 290 | R0 only, no LI/ADDI, no errors | **convert** with `tsr-trace-conv` into `traces/model-v1conv`. Each v1 instruction becomes a push that depends on the previous push (a serial chain, which keeps v1's sequential semantics; replayed with the `replay` flag SERIAL), END is dropped, and init/expect are kept. This is the regression check that the units' results did not change from v1 to v2 |
| `traces/tmp` (tbo dumps, no script, not in git) | 2,804 | — | dropped; `TSR_TRACE` dumps of `test-backend-ops` on v2 replace them when needed |
| new | — | — | `traces/vu`, `traces/at` (generator `--ops vu|at`), `traces/ctrl` (concurrency, credits, errors, holds), `traces/model-v2` (0.5B Q8_0 and Q4_K_M "Hi", one launch per split in M6, then one launch per token in M7) |

Mechanical conversion of the register-relative sets would also be possible (LI and ADDI depend only
on immediates and ARG). It is rejected because 93 + 5 + 156 error expectations would have to be
re-derived anyway, and the generators exist with fixed seeds.

X0 deleted the whole v1 generator and trace library (EX §12 "X0 as done"): `tsr-trace-gen` v2
starts from `v1-m4:sw/tools/tsr_trace_gen.c` (random programs, MVM boundary and error generators,
seeds), and `tsr-trace-conv` takes its v1 reader from `v1-m4:sw/sim/tsr_trace.c`. Until then the
v1 sets are reproduced only by the v1 tools of `build/v1` (`traces/*/regen.sh`).

---

## 6. Firmware SDK (`sw/ctrl`)

### 6.1 Toolchain and build

- **Compiler:** the supported compiler is the pinned xPack `riscv-none-elf-gcc` (§9.6), chosen by
  `TSR_RISCV_PREFIX`. Other bare-metal GCCs work only because the SDK includes nothing from
  libstdc++: only `<stdint.h>`, `<stddef.h>`, `<stdarg.h>` and the SDK's own `tsr/traits.hpp`,
  with `libgcc` from the rv32im / ilp32 multilib. (Ubuntu noble's `gcc-riscv64-unknown-elf` 13.2
  ships no libstdc++ headers and no rv32imc multilib, and needs sudo.) Clang ≥ 16
  (`--target=riscv32-unknown-elf`) is an optional second build.
- **Flags:** `-march=rv32imc_zicsr_zifencei -mabi=ilp32 -Os -ffreestanding -fno-exceptions
  -fno-rtti -fno-threadsafe-statics -fno-use-cxa-atexit -ffp-contract=off -nostdlib
  -nostartfiles -g -std=c++20`, then `libgcc.a` only (integer helpers).
- **libgcc:** taken from `-print-libgcc-file-name` for the same `-march`. If that multilib is
  absent, the `rv32im_zicsr/ilp32` one is used (*confirm* the xPack multilib set). Code without
  C instructions runs on an RV32IMC core.
- **Firmware float:** none in M6/M7 (V2-SW-8). Every f32 constant (eps, 1/n, scales, the RoPE
  `theta_scale` and freq-factor tables) is computed by the host and passed as bits. If firmware
  ever needs spec fp math, it calls the `tsr_ref` helpers (`docs/ctrl-riscv.md:229-240`).
- **CMake:** the firmware is its own CMake project (`ExternalProject_Add(tsr_fw …)`), because one
  CMake project cannot have two C++ compilers. It produces `tsr-fw.elf` (with DWARF) and
  `tsr-fw.map`. Its path is compiled into the host libraries as the default image (`TSR_IMAGE`
  overrides, EX §10.2). The RISC-V toolchain is a **required** dependency of a v2 build; configure
  fails with the install hint of §9.6.

### 6.2 Image format

EX §3.1–§3.2 are normative. Linker script memory map (CC §4.1): ITCM at 0x0000_0000, DTCM at
0x0010_0000, sizes `ITCM_BYTES` / `DTCM_BYTES` from `RV_CFG` (U200: 128 KiB / 64 KiB).

| Section | Region | Loaded | Content |
|---|---|---|---|
| `.vectors` | ITCM 0x0 (`RV_BOOT` = 0) | yes | 32 vector slots × 4 B (`j` to handlers), reset entry at +0x80 |
| `.text` | ITCM | yes | code |
| `.tsr_kernels` | DTCM | yes | kernel table (below), sorted by index |
| `.rodata`, `.data` | DTCM | yes | loaded directly by the host; crt0 copies nothing |
| `.tsr_id` | DTCM | yes | 8-byte image id (EX §3.2) |
| `.tsr_dbg` | DTCM | yes | DB's debugger flags (DB §6.3) |
| `.tsr_crash` | DTCM | no (zeroed by crt0) | DB's crash record (DB §6.3) |
| `.bss`, `.stack` | DTCM | no | zeroed by crt0; stack 8 KiB at the top of DTCM; 512 B trap stack |
| `.tsr_logfmt` | VMA 0xF000_0000 | **no** (`NOLOAD`, not in a `PT_LOAD`) | log format strings (§6.6) |
| `.note.tsr` | — | no | ELF note "TSR", type 1 (below) |

Kernel table entry (32 bytes):

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

`.note.tsr` descriptor (32 bytes): `sdk_abi` (1), `ctl_id_min`, `itcm_min`, `dtcm_min`,
`caps_req`, `n_kernels`, two reserved words.

**Image id (V2-SW-9, EX §3.2).** `tsr-elf` computes it after linking: SHA-256 over every `PT_LOAD`
segment in ascending physical address, each as (u32 paddr LE, u32 padded filesz LE, padded bytes),
with the 8 bytes of `.tsr_id` taken as zero; the id is the first 8 digest bytes read as a
little-endian `uint64`. `tsr-elf` writes it into `.tsr_id`. The runtime echoes it in HELLO (r2 =
bits [31:0], r3 = bits [63:32]). The host recomputes it from the ELF and compares, so a HELLO with
the right id proves that this exact image was loaded and booted; the load CRC and VERIFY (EX §3.3)
prove the transfer.

### 6.3 crt0 and the runtime (EX §4.7, normative)

crt0, at `RV_BOOT` + 0x80:

1. Set `sp`, `gp`, `mscratch` (trap stack) and `mtvec` (vectored, base 0), zero `.bss`, run
   `init_array`.
2. Initialise `__tsr_dbg` and `__tsr_crash` (DB §6.3), `RT_STATE` := BOOT.

The runtime then follows EX §4.7 exactly:

- **R1 Boot.** Push `qclear()` and join it; `RT_STATE` := IDLE; post HELLO as an event (r2 / r3 =
  image id, r4 = [15:0] SDK ABI | [31:16] runtime version, r5 = kernel count, r7 = boot core
  cycles). Descriptors pushed before HELLO stay queued.
- **R2 Take.** WFI on `irq_fast[2]` (`LQ_AVAIL` ≠ 0 ∧ ¬ACTIVE); copy `LQ_HEAD_W0`–`7` and
  `LQ_HEAD_SEQ` into DTCM; write `LQ_POP`.
- **R3 Cancelled or stopped before start.** Covered by `LQ_CANCEL` or `LQ_STOP` (EX §4.9): answer
  DONE / ABORTED without running.
- **R4 Validate.** Unknown TYPE, reserved bits set, KERNEL beyond the table, or a bad
  argument-block handle → DONE / BAD_DESC (args beyond the kernel's `nargs` are ignored). PING →
  PONG.
- **R5 Run.** `RT_STATE` := KERNEL | index; PROF → `REC_CFG.ALL` and NODE markers; stage the
  argument block (LD into the top 16 SPM lines, one join); `launch_pidx` = `ISSUE_PIDX`, `base_u` =
  `ISSUED_SEQ[u]`; DB's launch hook `tsr_rt_launch_hook(kernel, seq)` (DB §3.8); call the kernel.
- **R6 Join.** `join_all()`.
- **R7 Result.** First error = smallest `ERR_PIDX` over the units in error state; PLAT_ERR from
  `LQ_FLAGS`; status by EX §4.4's precedence. RET is the kernel's int32 return value.
- **R8 Answer.** WFI on `irq_fast[3]` while `LCQ_FREE` = 0; post DONE with ANSWER = 1.
- **R9 Clear.** If any unit is in error state, push `qclear()` and join it.
- **R10.** `RT_STATE` := IDLE; continue at R2. Debug builds assert `QSTATUS.QIDLE` here.

There is no mailbox, no STOP / BYE and no LOGCTL message: a soft stop is EX's `LQ_STOP` watermark
(kernels poll `tsr::stop_requested()`), and the log level is `LOG_CFG.LEVEL`. A trapped runtime
parks (§6.6); the host recovers with `tsr_transport_soft_reset` and a warm restart (EX §3.5).

### 6.4 Launch descriptors and answer records (EX §4.3–§4.4)

EX §4.3 (8-word descriptor: TYPE, PROF, KERNEL, ARG0–ARG6) and EX §4.4 (8-word records: DONE 0x01,
PONG 0x02, HELLO 0x80, TRAP 0x81, kernel events 0xC0–0xFF; launch STATUS 0–7 with precedence) are
the only formats. SW defines only the meaning of RET per kernel. For the base kernels:

| Kernel result | DONE STATUS (EX §4.4) | RET (r2) |
|---|---|---|
| success | 0 OK | 0 |
| a tile op failed | 1 TILE_ERR (r3–r6: ticket, `ERR_INFO`, `ERR_W0`, `ERR_PIDX` of the first error) | `graph`: index of the failing op record + 1 (1 … 4096); `replay`: relative push index of the first error + 1 |
| the kernel rejected its input (bad op list, scratch overflow, CRC) | 3 KERNEL_ERR | negative: `graph` returns 0x8000_0001 … 0x8000_0007 (§7.3) |
| cancelled or stopped | 5 ABORTED / 7 STOPPED | kernel-defined |

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

- Arguments are 32-bit integers, SPM line numbers or DRAM handles ({ch[31:28], beat[27:0]},
  `docs/ctrl-riscv.md:1094-1097`).
- A kernel returns RET ≥ 0 for success (0 normally) or a negative code (KERNEL_ERR), and it must
  leave every op it pushed joinable (the runtime joins all, R6).
- Kernels never pop, never post answers, may post events (types 0xC0–0xFF), and never issue from
  interrupt handlers (`issue()` is not reentrant, `docs/ctrl-riscv.md:316-320`).
- Long kernels poll `tsr::stop_requested()` at safe points (EX §4.9).
- Kernels obey the launch-stateless rule (§4.6) and the timing rule (§4.7).

### 6.6 Traps, panic and logging (V2-SW-11)

- **Traps and panic:** DB §6.3 is normative. The handler switches to the trap stack; with a
  debugger attached (`__tsr_dbg.attached`) it restores the registers and stops at
  `__tsr_trap_halt`; otherwise it fills `__tsr_crash`, logs a TRAP record (LEVEL FATAL, SOURCE 2),
  posts EX's TRAP record (type 0x81, ANSWER = 1 if ACTIVE: `mcause`, `mepc`, `mtval`, `RT_STATE`,
  `RT_INFO`, MTIME) and parks in a WFI loop with `RT_STATE` = TRAPPED. `ebreak` outside debug mode
  is a trap like any other.
- **`tsr::panic(code, a0, a1, a2)`:** enters the trap handler with `mcause` = 24 (custom) and
  `mtval` = code, after logging its arguments at level FATAL. It is used for violated internal
  invariants (for example a ring-allocator corruption), never for data errors, which return
  codes. GDB shows it as SIGABRT (DB §4.7).
- **Logging:** binary records through EX's log ring (EX §4.10), with DB's header (DB §6.1, the only
  definition): [7:0] N payload words, [11:8] LEVEL (0 FATAL, 1 ERROR, 2 WARN, 3 INFO, 4 DEBUG,
  5 TRACE), [15:12] SOURCE, [18:16] TYPE (0 FMT, 1 TEXT, 2 TRAP, 3 EVENT, 4 BLOB), [19] TS (the
  first payload word is MTIME[31:0]), [27:20] LSEQ, [31:28] SYNC 0xA. A FMT record carries the
  address of its format string in `.tsr_logfmt`, then the arguments.
  - `TSR_LOG(level, "fmt", args...)` stores the literal in `.tsr_logfmt` (not loaded, so it costs
    no TCM and does not enter the image id) and emits the record.
  - Supported conversions: `%d %u %x %08x %f` (argument = f32 bits), `%t` (ticket), 64-bit
    integers as two words, and `%s` for strings in loaded sections (the host reads them from the
    ELF).
  - Before writing, the runtime reads `LOG_FREE`. If the whole record does not fit, it drops the
    record and counts it in DTCM; the next record that fits is preceded by an EVENT record
    "dropped n". With `LOG_CFG.BLOCK` it waits instead. The ring therefore never holds partial
    records.
  - Interrupt handlers do not log. Compile-time level filter: `TSR_LOG_MAX_LEVEL` (default 3 INFO
    in release, 4 DEBUG in debug builds). Runtime filter: `LOG_CFG.LEVEL` (the host sets it from
    `TSR_LOG_LEVEL`, 0–5).
  - The host decodes with `tsr-log` or the transport (`TSR_LOG=stderr|<file>`).

### 6.7 Tile/VU header library (`tsr/kernel.hpp` and friends)

The base is `docs/ctrl-riscv.md:1090-1155`, with `CALL` removed and these additions:

```cpp
namespace tsr {
// --- types (unchanged): spm, dram, handle(), from_handle(), wref, qa, op, ticket, deps, after()
ticket   issue(const op &, deps = {}, uint32_t flags = 0);  // credit spin, then push (single hart, not reentrant)
bool     done(ticket);  void join(ticket);  void join_irq(ticket);
ticket   join2(ticket a, ticket b);           // SYNC NOP over a and b (one ticket)
void     join_all();                          // FENCE + join
uint32_t error(ticket);  uint32_t first_error();  ticket qclear();
uint32_t pidx_now();                          // CC's ISSUE_PIDX (0x03C): next push index, exact
bool     stop_requested();                    // EX §4.9: STOP_VALID and this launch covered
template <class T> volatile T * spm_ptr(spm l, uint32_t byte_off = 0);

// --- op builders: one constexpr builder per opcode and sub-function; VU and AT builders are
//     generated from sw/tables/tsr_tables.py (vu.hpp, at.hpp), so their encodings track VU / AT.
constexpr op LD(spm, dram, uint32_t lines);  constexpr op ST(spm, dram, uint32_t lines);
constexpr op QACT(qa, spm dst, spm src, uint32_t k, uint32_t t, uint32_t src_stride = 0);
constexpr op MVM(spm dst, spm x, wref w, uint32_t n, uint32_t k, uint32_t t,
                 uint32_t dst_stride = 0, bool acc = false);
constexpr op NOP();  constexpr op FENCE();

// --- unit/ticket groups: latest ticket per unit; folded to <= 2 deps or a join2 chain
struct group { ticket t[16]; void add(ticket); deps as_deps(); };

// --- SPM scratch ring with release tickets (used by the executor, available to kernels)
class spm_ring {
public:
    void   init(uint32_t first_line, uint32_t lines);
    // Allocates a contiguous block (never wrapping) and returns the deps that must precede any
    // write to it (releases of older overlapping blocks, folded to <= 2; older ones are joined in
    // software, deterministically by count, never by done()).
    spm    alloc(uint32_t lines, uint32_t parity /* 0 even, 1 odd, 2 any */, deps * must_wait);
    void   release(spm block, ticket after);   // block becomes reusable once `after` is done
};

// --- logging and panic (DB §6.1, §6.3)
#define TSR_LOG(level, fmt, ...)
[[noreturn]] void panic(uint32_t code, uint32_t a0 = 0, uint32_t a1 = 0, uint32_t a2 = 0);  // trap, mcause 24
}
```

`plan.hpp` is freestanding C++ (no STL, no heap). The host backend compiles it too (V2-SW-20). It
holds the MUL_MAT tiling (today's `tsr_mm_plan_tiling` and `tsr_mm_lines_per_token`), the T-tile
choice for an SPM budget, `QA_LINES`, `ROW_BEATS`, and the per-kind scratch requirements of §7.5.
The host therefore rejects in `supports_op` exactly what the firmware cannot plan.

### 6.8 Host build of the kernels (V2-SW-25)

The kernel sources also compile for the host against `backend_host.hpp`:

- `issue()` calls `tsr_sim_push`;
- `done()` and `join()` advance the simulator;
- `spm_ptr` maps to `tsr_sim_spm`.

The host build runs the same kernels in milliseconds under ASan and UBSan. It is a convenience. The
reference is the RV binary on `tsr_rv`, and ctest runs both builds (§10.2).

---

## 7. Kernels and the graph executor

### 7.1 Kernel set

| Kernel | Milestone | Args | Purpose |
|---|---|---|---|
| `graph` | M6 (MUL_MAT records only), M7 (all kinds) | a0 OPL handle, a1 OPL bytes, a2 flags (bit0 VERIFY_CRC) | the op-list executor (§7.3) |
| `selftest` | M6 | a0 test id, a1 data handle, a2 size/count, a3 seed, a4 flags | card and sim self-test programs (§7.6) |
| `replay` | M6 | a0 `push.bin` handle, a1 count, a2 flags (bit0 SERIAL: every op record depends on the previous op record instead of its recorded deps) | tile-only trace replay, SPM dumps (§7.7) |

These are EX's base kernels (EX §4.6). PING is part of the runtime (descriptor TYPE 0x02), not a
kernel. MoE kernels, decode steps, the resident loop and JIT
entry points come with M8 and M9.

### 7.2 Op list format OPL v1 (V2-SW-12)

An op list is one 64-byte-aligned blob in device DRAM, normally in the OPL region (today's CMD
region, ch0 0–64 MiB, §9.3). It is little-endian.

**Header (64 bytes):**

| Off | Field | Meaning |
|---|---|---|
| 0 | `magic` u32 | 0x4C50_4F54 ("TOPL") |
| 4 | `version` u16, `hdr_bytes` u16 | 1, 64 |
| 8 | `n_ops` u32 | number of op records, 1–4096 (V2-SW-29) |
| 12 | `n_slots` u32 | 64-byte slots in the op area |
| 16 | `ops_off` u32 | byte offset of the op area (64) |
| 20 | `flags` u32 | bit0 REC (completion records on every push), bit1 MARK (TRACE_MARK with the op index before each op), bit2 SERIAL (join after every op: debugging), bit3 STOP_ON_ERROR (stop pushing once an error is seen; the backend always sets it) |
| 24 | `n_pre` u32 | entries in the preload table |
| 28 | `pre_off` u32 | byte offset of the preload table |
| 32 | `pre_lines` u32 | SPM lines of the preload area |
| 36 | `graph_id` u32 | host hash, logged by the executor |
| 40 | `total_bytes` u32 | blob size |
| 44 | `crc32` u32 | CRC-32/ISO-HDLC of bytes [64, total_bytes), checked when the launch flag VERIFY_CRC is set |
| 48–63 | reserved | 0 |

**Preload entry (16 bytes; V2-SW-15):**

| Field | Meaning |
|---|---|
| `src` | DRAM handle |
| `bytes` | size in bytes |
| `pre_byte` | byte offset inside the preload area, 64-aligned, assigned by the host |
| reserved | 0 |

Preload entries carry the small graph inputs that firmware reads with the RISC-V (KV write indices,
GET_ROWS ids, each KQ mask's mask summary MSUM, AT §7.3) and the positions that ROPE reads. They
are loaded once at launch start with a single join, so no kernel joins on an input load inside the
op stream.

**Op record:** one or more 64-byte slots.

```
word0: [7:0] kind, [11:8] n_slots - 1, [15:12] 0, [23:16] kind flags, [31:24] n_args
word1: [15:0] dep0, [31:16] dep1      (op indices < own index; 0xFFFF = none)
word2..: args[n_args]                  (n_args <= 16 * n_slots - 2)
```

- An op with more than two predecessors is preceded by `JOIN` records, which the builder emits.
- All DRAM addresses are handles of 64-byte-aligned beats.
- Row strides are in lines.
- All f32 parameters are bit patterns computed by the host (§6.1).

**Kinds and argument words.** M6 implements 0x01 and 0x10; M7 the rest. Shapes are flattened to
rows: `rows` = ne1·ne2·ne3, and every row of a tensor starts on a line (ensured by `supports_op`).

| Kind | Name | Args (word index: meaning) | ggml source |
|---|---|---|---|
| 0x00 | NOP | — (no tile ops; its ticket is "none") | placeholder |
| 0x01 | JOIN | — (SYNC NOP over dep0, dep1) | builder |
| 0x10 | MUL_MAT | 0 x, 1 x stride, 2 y, 3 y stride, 4 W_OFF (beats), 5 fmt (0 Q8_0, 1 Q4_K), 6 N, 7 K, 8 T, 9 addend handle (0xFFFF_FFFF none), 10 addend stride (0 = row 0 broadcast to every token), 11 flags (bit0 ACC_DST: y += W·x) | MUL_MAT; fused MUL_MAT+ADD(bias or residual) |
| 0x11 | ADD | 0 dst, 1 dst stride, 2 a, 3 a stride, 4 b, 5 b stride, 6 n (ne0), 7 rows, 8 b row period (b row = r mod p, p ≥ 1), 9 flags (bit0 b is one f32 scalar) | ADD |
| 0x12 | MUL | as ADD | MUL (norm weights, Gemma `out_scale`) |
| 0x13 | SCALE | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 s (f32), 7 b (f32) | SCALE (`x·s`, then `+b` if b ≠ 0) |
| 0x14 | RMS_NORM | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 eps (f32), 7 inv_n = `tsr_fdiv(1, n)` (f32), 8 weight handle (0xFFFF_FFFF none), 9 weight row period | RMS_NORM, fused RMS_NORM+MUL(weight) |
| 0x15 | GLU | 0 dst, 1 dst stride, 2 gate, 3 gate stride, 4 up, 5 up stride, 6 n, 7 rows, 8 variant (0 SWIGLU, 1 GEGLU16: ggml-CPU-compatible fp16 GELU, VU §6.5; 2 GEGLU: exact tanh form, M8 / PyTorch) | GLU split form |
| 0x16 | ROPE | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 pos preload byte offset (I32 per token), 5 table handle (AUX f32 × n_dims/2: the freq-factor table, or the A table with mode bit1; 0xFFFF_FFFF none), 6 TS = `powf(freq_base, -2.0f/n_dims)` bits (host libm, as ggml-cpu), 7 M = attn_factor (f32), 8 freq_scale (f32; 1.0 = none), 9 head_dim, 10 n_dims, 11 n_heads, 12 n_tokens, 13 mode (bit0 NEOX; bit1 ATAB: YaRN form with a host-folded A table, VU §6.7) | ROPE: the angle recurrence of ggml-cpu (VU §6.7), bit-identical angles for ext_factor = 0 |
| 0x17 | GET_ROWS | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 ids preload byte offset (I32), 5 n_ids, 6 row lines | GET_ROWS on f32 activations (`inp_out_ids`) |
| 0x18 | SET_ROWS_KV | 0 cache base handle, 1 cache row stride (AT units), 2 src, 3 src stride, 4 idx preload byte offset (I64; the host checks idx < 2^31), 5 n_rows, 6 row elements, 7 which (0 K, 1 V), 8 plane base, 9 E/64 \| LG << 16, 10 plane kind, 11 stream (AT §12) | SET_ROWS into a KV buffer: QACT A32 + KVST per run of consecutive cells (AT §5.1) |
| 0x19 | FLASH_ATTN | 0 dst, 1 dst token stride, 2 q, 3 q head stride, 4 q token stride, 5 K cache base, 6 V cache base, 7 n_kv, 8 head_dim, 9 n_head, 10 n_head_kv, 11 n_tokens, 12 MSUM preload byte offset, 13 f32-mask handle (0xFFFF_FFFF unless the MSUM is GENERAL), 14 scale (f32), 15 softcap (must be 0), 16 K plane base, 17 V plane base, 18 E/64 \| LG << 16, 19 D, 20 G, 21 flags (AT §12; two slots) | FLASH_ATTN_EXT with K/V in KV buffers; the device never reads the F16 mask |
| 0x1A | CPY | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 row lines, 5 rows | CPY/CONT f32 → f32 with contiguous rows |
| 0x1B | UNARY | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 func (0 TANH, 1 GELU16 (llama.cpp GELU), 2 SILU, 3 EXP, 4 GELU exact tanh form) | UNARY ops the VU has |
| 0x20–0x3F | reserved for M8 | MUL_MAT_ID, TOPK/ARGMAX, SOFT_MAX | |

### 7.3 Executor (`graph` kernel)

**SPM layout** (U200 defaults; `L` = `SPM_BYTES`/64 = 65,536 lines):

| Lines | Use |
|---|---|
| [L−16, L) | runtime argument range |
| [L−16−4096, L−16) | OPL area (256 KiB); a larger op list streams in two 128 KiB halves |
| [.., ..) of `pre_lines` | preload area (header `pre_lines`, ≤ 256 lines) |
| [0, rest) | scratch ring (`spm_ring`), ≈ 3.7 MiB |

**DTCM tables:** `ticket[4096]` (the op's done ticket) and `pidx[4096]` (the op's first push
index), 32 KiB in total.

**Algorithm:**

1. LD the header and the op list (or the first half) into the OPL area, LD the preload entries, and
   join once. Validate magic, version, `n_ops` ≤ 4096 and (if VERIFY_CRC) the CRC. A failure
   returns 0x8000_000x (below).
2. For op i = 0 … n_ops−1:
   1. Read the record through the SPM window.
   2. `d` = after(ticket[dep0], ticket[dep1]).
   3. `pidx[i]` = `pidx_now()` (`ISSUE_PIDX`: exact and timing-independent).
   4. Call the kind handler with (args, d, ring). It allocates scratch (its writes also wait for
      the ring's release deps), pushes its tile ops, and returns one ticket T_i whose completion
      implies every tile op of op i is done. The last tile op depends on the others, or a `join2`
      provides that (I1, `docs/ctrl-riscv.md:544-550`).
   5. `ticket[i]` = T_i. The handler releases its scratch blocks with `after` = T_i.
   6. Every 16 ops, and before every software join, read `QSTATUS` error bits. If an error is
      seen and STOP_ON_ERROR is set, stop pushing. This decision is timing-dependent but changes
      only which later ops are skipped; the result is an error either way, and error traces are
      generated with STOP_ON_ERROR = 0.
3. Run `join_all()`.
4. Read `first_error()`. If it is nonzero, find the op i whose `pidx[i]` ≤ `ERR_PIDX` < `pidx[i+1]`
   and return i + 1. Otherwise return 0.

**Executor return codes:**

| Code | Meaning |
|---|---|
| 0 | success (DONE STATUS OK) |
| 1 … 4096 | the failing op is index ret−1 (DONE STATUS TILE_ERR, set by the runtime's R7 from the first error) |
| 0x8000_0001 | BAD_MAGIC |
| 0x8000_0002 | BAD_VERSION |
| 0x8000_0003 | TOO_MANY_OPS |
| 0x8000_0004 | BAD_RECORD (unknown kind, wrong n_args, slot overflow) |
| 0x8000_0005 | BAD_DEP (dep ≥ own index) |
| 0x8000_0006 | SCRATCH (the op cannot be planned in the ring) |
| 0x8000_0007 | CRC |

Codes ≥ 0x8000_0000 are negative as int32, so the runtime answers DONE STATUS KERNEL_ERR with
them in RET (§6.4).

**Overlap (D3).** The executor runs ahead of the units, limited only by credits and the ring.
Independent ops overlap on different units: Q/K/V projections, gate and up, and a ST of one op
beside the MVM of the next. Every unit therefore starts the moment its inputs are done, with no
RISC-V round trip in between.

**DTCM state** is rebuilt at every launch (V2-SW-4).

### 7.4 MUL_MAT kernel

The kernel follows v1's lowering (`docs/spec.md:502-512`), so its numerics match v1 bit for bit
(`traces/model-v1conv` checks this). Per T tile (`plan.hpp`, T_MAX and the ring budget):

1. LD x (tile).
2. If an addend is present, LD the addend rows into the y block (the bias row repeated when its
   stride is 0).
3. QACT (A32 for Q8_0, A256 for Q4_K).
4. MVM with ACC when there is an addend or ACC_DST.
5. ST y.

Fusing a bias or residual through ACC gives y = addend + W·x with one f32 addition. f32 addition is
commutative, so this is bit-identical to ggml's separate ADD on the device. Tiles are
double-buffered in the ring: the LD of tile t+1 waits only for the QACT that read its buffer.

### 7.5 M7 kernels

| Kind | Tile composition (primitives from VU/AT) | Scratch (lines, per tile) | Numerics reference |
|---|---|---|---|
| ADD/MUL/SCALE | LD a (+ LD b), VOP ADD / MUL (b stride 0 for a broadcast row; one op per period block for row periods > 1, VU §9), MULC / ADDC with `cmode` 1 (scalar) or 2 (per-row c vector, e.g. Gemma's `inp_scale_rows`), MULCADD for SCALE with a bias; ST; rows batched to the VL stream | 2–3 × row lines | `tsr_ref` per element |
| RMS_NORM (+weight) | LD x, `RED.SUMSQ` (post RMS, sk = 1/n, se = eps, set by `SOP.MOVI`) → per-row vector (or s register for one row); `VOP.MULCB` (`cmode` 2, b = weight with stride 0), or `VOP.MULC` without weight; ST. Per-head norms (Gemma q / k / v norms) are the same two ops with rows = n_head·T and m = head_dim (VU §8.1) | 2 × row lines | `sw/ctrl/ref/rmsnorm_ref.cpp` (same order); bit-identical to the spec's 7-op lowering `docs/spec.md:311-321` (VU §6.2) |
| GLU | LD gate, LD up, VOP SWIGLU or GEGLU16, ST | 2 × row lines | `tsr_m_silu` / `tsr_m_gelu16` |
| ROPE | LD x, LD the freq-factor (or A) table once per op, `SOP.MOVI` of TS, M and freq_scale, ROPE over all tokens (positions from the preload area), ST | 2 × head_dim × n_heads / 16 | VU §6.7 (`tsr_ref_vu.c` ROPE) |
| GET_ROWS | per id (RISC-V reads it from the preload area): LD row → ST row | 2 × row lines | copy |
| SET_ROWS_KV | per run of consecutive cells (indices from the preload area): LD src rows, QACT A32 (K = E), KVST (AT §5.1) | per AT | `tsr_ref_kvst` |
| FLASH_ATTN | the sequence of AT §6.6 (normative): QACT of Q; per pass and per chunk (chunk list from the preloaded MSUM) ATTQK (FAST) or ATTQK + AMASK (GENERAL), ASMAX, ATTPV (LSUM), AMERGE; ANORM; ST | per AT (§7.4 there) | `tsr_ref_fa` (AT §13.1) |
| CPY | LD rows, ST rows | 2 × block | copy |
| UNARY | LD, VOP func (TANH, GELU16, SILU, EXP, GELU), ST | 2 × row lines | `tsr_ref_math` |

The numerics of a composite kernel are defined by its tile-op sequence. Each kernel's header
comment states the sequence. A host C++ reference in `sw/ctrl/ref/` composes the same `tsr_ref`
calls in the same order, and the kernel tests compare the firmware against it bit-exactly.
Changing a sequence changes logits, so the change goes with a firmware ABI bump and new klcheck
baselines.

### 7.6 `selftest` kernel (V2-SW-28)

| Id | Test |
|---|---|
| 1 | LD/ST round trips on every channel: 1, 63, 64, 65 and 4,096 lines, cross-channel copies |
| 2 | QACT A32/A256 on host-written data |
| 3 | MVM Q8_0 and Q4_K: T = 1, T = T_MAX, ACC |
| 4 | each VU function and reduction |
| 5 | each AT op |
| 6 | concurrency: a `matvec_batch`-style pipeline (`docs/ctrl-riscv.md:1264-1301`) on all units |
| 7 | errors: DRAM_RANGE, ILLEGAL_OP, BAD_DEP, DEP_FAILED chain, QABORT then QCLEAR recovery |
| 8 | SPM window byte/half/word, ordering of a store before a push |
| 9 | log records and PING |
| 10 | credits: fill every queue |

`tsr-selftest` runs each test on the device and, in process, on the simulator with the same image,
inputs and args. It compares DONE, every output byte and the per-unit completion counts.

### 7.7 `replay` kernel

1. Read `base_u` = `ISSUED_SEQ[u]` for every unit and `pidx0` = `ISSUE_PIDX`.
2. Stream `push.bin` through SPM in double-buffered 64 KiB chunks.
3. For each record:
   - kind 0: issue the words with the push value and deps mapped back as (u, base_u + rel_seq);
     with flag SERIAL, the deps are replaced by the ticket of the previous kind-0 record (a
     dependency chain: with I1 this is v1's sequential semantics);
   - kind 1: write QCTRL;
   - kind 2: write TRACE_MARK.
4. Run `join_all()`. RET = 0 if no unit is in error state, else 1 + (`ERR_PIDX` of the first error
   − `pidx0`); the runtime reports the error record itself in DONE r3–r6 (EX §4.6).

### 7.8 Performance targets (*estimates*, checked in P3/P6)

- **Executor:** ≤ 400 RISC-V cycles per graph op on average (2.4 µs at 166.7 MHz). For Qwen2.5-0.5B
  (≈ 25 ops per layer, ≈ 230 µs of MVM time per layer) the RISC-V needs ≈ 60 µs per layer and
  stays ahead.
- **Launch:** EX §4.11 (empty-launch round trip ≤ 10 µs target; descriptor push → kernel entry
  ≤ 5 µs; answer record → host ≤ one poll interval). The v1 exec cost was 35 µs per exec
  (`docs/platform-u200.md:1049`).
- **Image size:** ITCM ≤ 48 KiB of 128 KiB; DTCM ≤ 56 KiB of 64 KiB including the 8 KiB stack,
  the 512 B trap stack and the 32 KiB executor tables.

---

## 8. One launch per decode token: what llama.cpp's graph needs

The M7 target is that every decode graph produces exactly one Tesseract split. This needs:

1. **R1 — every node after the input layer is supported** (§9.3 supports_op table), including the
   nodes of the reserve graph (prefill T ≤ n_ubatch). A node whose pre-allocated tensor sits in a
   Tesseract buffer must be supported, or the scheduler aborts (`docs/llama-pin.md:174-175`).
   That applies to SET_ROWS into the KV cache. Conversely, `GGML_OP_DUP` and SET_ROWS whose dst is
   not a KV tensor stay unsupported, and `supports_op` returns false for nodes without
   `GGML_TENSOR_FLAG_COMPUTE`: every pinned decode graph holds unselected embedding branches from
   `ggml_build_forward_select` (DUP(`inp_mixed_embd`), GET_ROWS, SET_ROWS;
   `src/llama-graph.cpp:2745-2753`), the scheduler's assignment passes ignore the flag
   (`ggml-backend.cpp:1298-1317`), and a supported DUP there would create a second Tesseract split.
   `graph_compute` skips nodes without the flag and makes no launch for a split that has none.
2. **R2 — token embedding stays on the CPU.** llama.cpp keeps the input layer on the CPU
   (`src/llama-model.cpp:1688-1689`). Its GET_ROWS forms a CPU split before the device split and
   costs no launch. The scheduler copies the n_embd f32 result in with `set_tensor`.
3. **R3 — KV cache on the device (M7):** run without `-nkvo` (`offload_kqv` = true), with
   `-ctk q8_0 -ctv q8_0` and `-fa on`. The pinned llama.cpp refuses a quantized V cache without FA
   (`src/llama-context.cpp:566-569`). M6 runs with `offload_kqv` = false (§10.4).
4. **R4 — Hadamard KV rotation off (V2-SW-16).** With a quantized KV cache and head_dim % 64 = 0,
   llama.cpp rotates Q, K and V and un-rotates the attention output by `MUL_MAT(rot, x)`, where
   `rot` is an F32 graph input (`src/llama-kv-cache.cpp:316-356`, `src/llama-graph.cpp:3228`,
   `:3273`). That src0 is not a weight and cannot become an MVM weight, so the rotations would
   split attention into CPU islands. Device runs and every baseline therefore set
   `LLAMA_ATTN_ROT_DISABLE=1`. The backend warns once if it sees `attn_inp_k_rot`/`attn_inp_v_rot`.
5. **R5 — graph inputs** (`inp_pos`, `kq_mask`, `k_idxs`, `v_idxs`, `inp_out_ids`) are written by
   the scheduler through `set_tensor` before the launch. They are copies, not launches. The builder
   lists the index inputs in the preload table.
6. **R6 — compute buffers fit the device (V2-SW-26).** Compute-buffer chunks are placed in a
   per-channel compute heap between the top of the weights and the bottom of the KV planes (EX
   §11), sized from the reserve graph, plus ch0 [64, 320) MiB and ch1–ch3 [0, 512) MiB; there are
   no fixed 512 MiB windows, and the buffer type's `get_max_size` is at least the largest single
   tensor. The reserved logits chunk then stays on the device: Qwen 152,064 × 512 × 4 B = 297 MiB
   at `-ub 512` (`docs/llama-pin.md:320-333`); Gemma 4 262,144 × 512 × 4 B = 512 MiB (`n_outputs`
   = min(n_ubatch, n_outputs_max), `src/llama-context.cpp:346`), with the softcap's SCALE / TANH /
   SCALE beside it. A chunk that still does not fit is staged and runs in its own launch: correct,
   but it costs one extra launch per token, which TSR_STATS reports as "staged chunks".
7. **R7 — graph reuse.** llama.cpp reuses the graph while the KV bucket (256 cells) is unchanged
   (`docs/llama-pin.md:290-300`). The backend caches the op list by (graph identity, node
   addresses). Per token the host writes the inputs, launches once and reads the logits. A new
   bucket builds one new op list.
8. **R8 — trailing CPU ops are allowed.** Gemma's final logit softcap (SCALE, TANH, SCALE,
   `src/models/gemma4.cpp:434-437`, cap 30) runs on the device: the VU has TANH (VU §6.5).
9. **R9 — no backend sampling** (`-bs`) in M7. Sampling stays on the host until M8
   (`docs/llama-pin.md:302-311`).
10. **R10 — not supported in M7:** context shift (K-shift RoPE on a q8_0 cache), KV defrag through
    ggml ops, and multi-sequence streams > 1. The backend reports these clearly instead of
    aborting inside the scheduler where it can (an op-list build error naming the feature).

Ops in the two model families (pinned llama.cpp):

| Model | Ops on the device in M7 |
|---|---|
| Qwen2.5 (`src/models/qwen2.cpp`) | RMS_NORM, MUL, MUL_MAT (+ bias ADD, fused), ROPE NEOX, SET_ROWS (K, V), FLASH_ATTN_EXT (GQA), ADD (residual, fused), GLU SWIGLU, GET_ROWS (`inp_out_ids`), MUL_MAT (head) |
| Gemma 4 dense (`src/models/gemma4.cpp`) | MUL by `inp_scale_rows` (the √n_embd embedding scale as an F32 graph input [1, T], `src/llama-graph.cpp:2776-2783`: VOP.MULC with a per-row c vector; in the leading CPU split or on the device, one launch either way), RMS_NORM (+MUL; per-head q/k norm; V without weight), MUL_MAT, ROPE NEOX (freq_factors on full-attention layers; freq_base 1e4 on SWA layers), SET_ROWS into two caches (iSWA), FLASH_ATTN_EXT (scale 1.0, SWA mask), ADD, GLU GEGLU (GEGLU16, :312), MUL (`layer_output_scale`), GET_ROWS, MUL_MAT (head, tied to `token_embd`), [SCALE, TANH, SCALE] |

Gemma 4 12B, read from `models/gemma4/gemma-4-12b-it-Q8_0.gguf` (gguf-py, 2026-10-30): 48 layers
(pattern 5 SWA : 1 full → 40 + 8), n_embd 3840, n_head 16, n_head_kv 8 (SWA) / 1 (full), head_dim
256 / 512, n_ff 15,360, vocab 262,144, n_swa 1024, rope base 1e6 / 1e4 (SWA), rope dims 512 / 256
with `rope_freqs` [256] on full layers, final logit softcap 30, no `attn_v` on full layers (V = K
projection), no `output.weight` (tied), `layer_output_scale` [1]. The pinned llama.cpp loads it as
`LLM_TYPE_UNKNOWN` (no size label at `gemma4.cpp:28-34`), which changes nothing in the graph. The
31B values in VU §3 and AT §4.5 stay *confirm* until a 31B GGUF is read.

---

## 9. Host software

### 9.1 Transport v2 API (`sw/transport/tsr_transport.h`)

EX §10.1 is the API, verbatim: open / close / name / config / write / read / csr_read /
csr_write, `tsr_image_open` / `close` / `kernel` / `id` (64-bit), `tsr_transport_boot` (warm if
possible, else cold; HELLO checked), `tsr_transport_launch(t, const tsr_launch *, uint32_t *seq)`
with `tsr_launch {type, flags, kernel, arg[7]}` (blocks only at `LQ_DEPTH` outstanding),
`tsr_transport_wait(t, seq, timeout_ns, tsr_record *)`, `tsr_transport_call`, `_cancel`, `_stop`,
`_set_event_cb`, `_soft_reset` (`SOFT_RESET`, wait `SR_BUSY` = 0) and `_log_drain`. SW adds only
conveniences on top of it, not a second protocol:

```c
int             tsr_image_open_default(tsr_image ** out);   // $TSR_IMAGE or the build's tsr-fw.elf
const char *    tsr_image_path(const tsr_image * img);
const tsr_image * tsr_transport_image(const tsr_transport * t);   // booted image or NULL

// Decoded view of a record (accessors over tsr_record.w[], EX §4.4)
typedef struct tsr_result {
    uint8_t  type, status, err_code;       // r0
    uint32_t seq;                          // r1
    int32_t  ret;                          // r2 (DONE)
    uint32_t err_ticket, err_info, err_w0, err_pidx;   // r3-r6 (DONE, first error)
    uint32_t duration;                     // r7, core cycles
} tsr_result;
void tsr_record_decode(const tsr_record * rec, tsr_result * out);
const char * tsr_status_name(uint8_t status);   // "OK", "TILE_ERR", … (EX §4.4)

typedef struct tsr_transport_stats {
    uint64_t launches, launch_wall_ns, csr_reads, csr_writes, bytes_written, bytes_read,
             log_words, log_records_dropped, staged_chunks;
} tsr_transport_stats;
void tsr_transport_get_stats(const tsr_transport * t, tsr_transport_stats * s);
```

**Protocol layer (`tsr_kdev.cpp`, V2-SW-18).** One implementation serves every kind and follows
EX §2.1, §3, §4.8 and §10.2:

- **Open:**
  1. Read and check `ID` (0x5453_5231), the `VERSION` major (2), `CFG*`, `CAPS`, `CFG3`,
     `CTL_ID` ≠ 0, `RV_CFG`, `Q_DEPTH` and `RV_FREQ` ≠ 0 (else "no `clk_rv`").
  2. Clear `ADAPTER_STATUS`.
  3. Unless `TSR_ATTACH=1`: `SOFT_RESET` and wait `SR_BUSY` = 0 (`TSR_RESET_MS`), which stops a
     control core left running by an earlier process; then bind (channel bases, `MEM_EN`, EX §9.2).
- **Boot:** per EX §3.3–§3.5: warm restart when VERIFY of every ITCM segment matches, else a cold
  load (`LOAD_ADDR`, `LOAD_DATA` words, receive CRC, VERIFY CRC); `RV_BOOT` = 0; `RV_RUN` = 1; wait
  for HELLO (`TSR_BOOT_MS`) and check the image id, SDK ABI and kernel count.
- **Launch:** EX §4.8: at most `LQ_DEPTH` launches outstanding (host-side credit, no read); write
  the `LQ_W` words that differ from the shadow copy; write `LQ_PUSH`.
- **Wait:**
  - Consume the LCQ (EX §4.8): read `LCQ_WR`, read r0 of each new record (r2–r6 only when needed),
    deliver events to the callback, write `LCQ_RD`.
  - Poll policy: busy-poll while the elapsed time is below the last DURATION of the same kernel,
    then poll every max(10 µs, 1 % of that duration) with sleeps; between polls call
    `kdev::idle()` and **release the transport mutex** (DB's in-process debug service needs it).
  - Drain the log every poll if `TSR_LOG` is set, else every 10 ms.
  - **Timeout** (`TSR_LAUNCH_TIMEOUT_MS`, default 60,000; 0 = none): time counts only while
    `STATUS.RV_HALTED` = 0 and `DBG_STATUS.STOP_ACTIVE` = 0 (read once per backoff poll); with
    `DBG_CFG.SERIAL` = 1 the timeout is off (DB §4.1). On a timeout the transport follows DB's
    recovery ladder (DB §7.3): level 0 hang dump (`TSR_DUMP_DIR`), level 2 `LQ_CANCEL` covering the
    outstanding launches and a wait of `TSR_ABORT_MS` for DONE / ABORTED, then level 3
    `SOFT_RESET` + warm restart (EX §3.5) if the RISC-V is stuck. A hung unit (`UNIT_HUNG`) ends
    with "card reset needed" (EX §8.6).
- **TRAP record:** drain the log, report the decoded trap (cause, `mepc` → function and line);
  attach GDB if `TSR_GDB_WAIT` asks for it, else `SOFT_RESET` + warm restart.

**kdev v2** (EX §10.2).

- Keeps: `reg_read`, `reg_write`, `mem_read`, `mem_write`, `num_ch`, `ch_bytes`, `ch_addr`,
  `describe`.
- Drops: `start`, `wait`, `settle`, `kargs`, `regs_*`, `kds_*`.
- Adds `KREG_ADAPTER_CTRL` and `virtual void idle(uint32_t poll_iter)` (EX's `advance` hint):

| kdev | `idle()` and specifics |
|---|---|
| `sim` | in-process simulator behind the core CSR port; `idle()` = `tsr_sim_advance(budget)`; a host read of `LCQ_WR`, `STATUS` or `LOG_WR` first runs the model to the next host-visible event (EX §10.2). Replaces `tsr_transport_sim.c` |
| `mock` | the `sim` kdev plus the `ap_ctrl_none` adapter register model (channel bases, `MEM_EN`, `ADAPTER_STATUS`, error injection) |
| `vkernel` | the Verilated kernel server with two clocks (P5); `idle()` runs the server's clocks |
| `xrt` | `ap_ctrl_none` compute unit opened through `xrt::ip` with exclusive access; registers through `read_register`/`write_register`; one device-only BO per channel as today; `idle()` sleeps (backoff). *confirm* the BO bank lookup without an `xrt::kernel` (memory topology from `xrt::xclbin`, EX §9.1) |

`TSR_XRT_CTRL` disappears. The `TSR_XRT_DMA*` and `TSR_XRT_CH_GIB` settings stay.

### 9.2 Environment (host)

| Variable | Meaning | Default |
|---|---|---|
| `TSR_DEVICE` | `sim`, `mock`, `vkernel`, `xrt`, `none` | `sim` |
| `TSR_IMAGE` | firmware ELF (EX §10.2) | build tree `tsr-fw.elf` |
| `TSR_LAUNCH_TIMEOUT_MS`, `TSR_ABORT_MS`, `TSR_BOOT_MS`, `TSR_RESET_MS` | wait timeout per launch (debug-aware, §9.1), cancel wait, HELLO wait, `SOFT_RESET` wait (EX §10.2) | 60000, 10000, 2000, 15000 |
| `TSR_LOG`, `TSR_LOG_LEVEL` | decode the log to `stderr` or a file; `LOG_CFG.LEVEL` 0 FATAL … 5 TRACE (DB §6.1) | off, 3 |
| `TSR_ATTACH` | open without `SOFT_RESET` or load (debugger, post-mortem) | 0 |
| `TSR_TRACE` | per-launch trace v2 dumps | off |
| `TSR_STATS` | as today, plus launches per token, ops per launch, fused ops, staged chunks (DB §8.6) | off |
| `TSR_GDB`, `TSR_GDB_WAIT`, `TSR_DEBUGGER`, `TSR_DUMP_DIR`, `TSR_TILE_SERIAL`, `TSR_PROF*` | DB §9.5 | off |
| `LLAMA_ATTN_ROT_DISABLE` | must be 1 for device runs and baselines (R4) | set by klcheck/numcheck/tests |

### 9.3 `sw/ggml-tesseract`

**M6 (P3.2):** attention stays on the CPU (`offload_kqv` = false, §10.4).

- `graph_compute(split)`:
  1. Build an OPL with one MUL_MAT record per MUL_MAT node, plus JOINs. Two MUL_MATs are
     dependent only if one reads the other's output; host-side hazard analysis over DRAM byte
     ranges covers WAR and WAW.
  2. Stage host-memory chunks as v1 does (X0 deleted that code with the emission; P3.2 restores
     `tsr_lower_staged` and `tsr_act_loc` from tag `v1-m4`, EX §12 "X0 as done").
  3. Write the OPL into the OPL cache (an LRU in the 64 MiB OPL region, keyed by content hash).
  4. Launch `graph` and wait.
  5. On DONE STATUS TILE_ERR, map RET−1 to the ggml node and abort with the node name, the error
     code and `ERR_INFO` (as v1 does on `STATUS.ERR`).
- `supports_op` (MUL_MAT rules) is unchanged, so splits are v1's: one launch per split, 145 (0.5B)
  and 385 (32B) per token (`docs/platform-u200.md:1053`).
- New: `tsr-opl.{h,cpp}` (builder, hazards, encoding, cache).
- Deleted in X0 (EX §12): the emission and program-cache parts of `tsr-lower.*`; tiling moves to
  `plan.hpp`.
- Weights, buffers and the load-time conversion are unchanged. The lazy conversion of raw-buffer
  src0 (`tsr_lazy_*`; `test-backend-ops` puts src0 in non-WEIGHTS buffers) and
  `tsr_resolve_weight` were deleted in X0 and come back from tag `v1-m4`.

**M7 (P3.5):**

- **Builder passes:**
  1. Drop no-op nodes (NONE, VIEW, RESHAPE, PERMUTE, TRANSPOSE).
  2. Resolve each tensor to a handle and line stride.
  3. Fuse: MUL_MAT + ADD whose other operand is a bias or the residual (via the MUL_MAT addend);
     RMS_NORM + MUL(weight).
  4. Compute dependencies (RAW, WAR, WAW over the DRAM byte ranges of ggml-alloc addresses),
     transitively reduce them, and emit JOINs above two.
  5. Build the preload table (index inputs, positions, MSUMs).
  6. Encode.
  The result is cached per graph identity (R7).
- **Mask summaries (AT §7.3).** On `set_tensor` of a tensor whose name contains `kq_mask` (the
  scheduler's input copy of each KQ mask), the backend writes its MSUM next to it and, if the mask
  is GENERAL, an f32 copy of the mask; the buffer type's `get_alloc_size` reserves the bytes for
  both on those tensors. The FLASH_ATTN record refers to the MSUM through the preload table, so the
  cached op list stays valid while the mask changes per token.
- **KV buffers (V2-SW-17):**
  - At the first data access (clear), a buffer whose tensors are all named `cache_k_l*`/`cache_v_l*`
    is placed in a KV heap in the layout AT defines.
  - Its `set_tensor`/`get_tensor`/`memset_tensor`/`clear` convert between ggml `block_q8_0` and
    the device layout with AT's C reference, so llama.cpp's state save/load keeps working.
  - FA and SET_ROWS are supported only when K/V/dst are views of such tensors.
- **supports_op (M7).** Every condition below also requires every operand's base offset and row
  stride to be multiples of 64 bytes.

| Op | Supported when |
|---|---|
| NONE, RESHAPE, VIEW, PERMUTE, TRANSPOSE | always |
| MUL_MAT | v1 rules (`docs/spec.md:491-498`) |
| ADD, MUL | F32; src1 broadcast by whole rows (a row period) or a single scalar |
| SCALE | F32 |
| RMS_NORM | F32 rows |
| GLU | SWIGLU, or GEGLU lowered to GEGLU16 (the CPU's fp16-table form, VU §6.5); split form or non-swapped single form |
| ROPE | NORM/NEOX, F32, I32 positions that are a graph input, freq_factors F32 or none, head_dim a multiple of 16 (≤ 1024) and n_dims a multiple of 32 (NEOX) or 16 (NORM) (VU §5.4), no mrope/vision modes |
| GET_ROWS | F32 src0 in a compute buffer, I32 ids that are a graph input |
| SET_ROWS | dst in a KV buffer, F32 src, I64 ids that are a graph input (any other SET_ROWS: false, R1) |
| FLASH_ATTN_EXT | AT §2.6: Q F32; K/V q8_0 views in KV buffers; D = DK = DV ∈ {64, 128, 256, 512}; mask F16 or none; `max_bias` = 0; `logit_softcap` = 0; no sinks; `n_kv_max` = 0 |
| CPY, CONT | F32 → F32 with contiguous rows |
| UNARY TANH, GELU | TANH; GELU lowered to GELU16 |
| DUP, nodes without `GGML_TENSOR_FLAG_COMPUTE` | false (R1) |
| everything else | false |

### 9.4 `sw/rt` and `sw/torch`

`tsr_rt::linear` emits a one-op OPL (MUL_MAT; K parts as further records with ACC_DST) instead of
a cached program. The OPL cache replaces the CMD-slot cache. The public API and the numerics
(`sw/rt/tsr_rt.h:1-49`) are unchanged, so `sw/torch` needs only a rebuild. The PyTorch move onto
the M7 kernels is M8.

### 9.5 Tools

| Tool | v2 |
|---|---|
| `tsr-selftest` (was `tsr_xrt_selftest`) | identity (CSRs, `CTL_ID`, `RV_FREQ`, `Q_DEPTH`, adapter); host memory tests (unchanged); load + VERIFY + HELLO + image id + PING round-trip time and pipelined launch rate (EX §4.11); the `selftest` kernel tests with in-process sim shadow; `--traces DIR` v2 replays through `replay` or full launches; `--debug` DMI halt/resume/step smoke (DB's checks); log drain |
| `tsr-trace-gen` | v2 `ops` traces: `--ops m2|m3|vu|at|ctrl`, errors, concurrency, seeds kept |
| `tsr-trace-replay` | v2 on the simulator (full or tile-only), `--schedules N` determinism mode; `--device KIND` replays through a transport |
| `tsr-trace-conv` | v1 → v2 for `traces/model` (§5.6) |
| `tsr-elf` | image id, kernel table, sizes, `.tsr_logfmt` dump |
| `tsr-log` | decode raw log words (DB §6.1 header) with an ELF |
| `klcheck`, `numcheck` | settings move to FA on (V2-SW-22, §10.4); `--fa` records `on`; KV type q8_0/q8_0; `offload_kqv` = false in M6 and true in M7; `LLAMA_ATTN_ROT_DISABLE=1` in every phase (recorded in the `.klc` header and the numcheck run metadata). numcheck's per-node observation (`cb_eval`) makes the scheduler compute one node per `graph_compute`, which on v2 is one launch per node, so numcheck needs no device mode (V2-SW-27) |
| `tsr_bw` | unchanged (separate M0 bandwidth kernel) |

### 9.6 Toolchain installation without sudo (V2-SW-7)

Checked on the build host (Ubuntu 24.04, 2026-10-30): no `riscv*-elf-gcc`, no `clang` binary
(only `libclang` 18), no `ld.lld`, no Spike, no OpenOCD, and a host-only `gdb`. Verilator 5.046 and
GCC 13.3 are present.

| Tool | Plan (user space, under `~/.local/opt`, versions in `docs/v2/toolchain.lock`) |
|---|---|
| RISC-V GCC + GDB | xPack GNU RISC-V Embedded GCC `riscv-none-elf-gcc`, pinned release 14.2.0-3 linux-x64 tarball from `github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack` (*confirm* that the release exists and its multilibs), unpacked to `~/.local/opt/xpack-riscv-none-elf-gcc-14.2.0-3`. It includes `riscv-none-elf-gdb`, which serves for GDB over DMI (D18) |
| Clang (optional) | conda-forge `clang`/`lld` in a separate conda env; uses the xPack `libgcc.a` |
| dtc | build `dtc` from source (`make NO_PYTHON=1 NO_YAML=1 PREFIX=~/.local/opt/dtc install`); Spike calls it at run time (*confirm*) |
| Spike | `riscv-isa-sim` at a pinned commit, `configure --prefix=~/.local/opt/spike`, `make -j4` (*confirm* that the build does not need Boost for the features used) |
| riscv-tests, riscv-arch-test | pinned clones under `build/rvtests`, built with `RISCV_PREFIX=riscv-none-elf-` and XLEN 32 (*confirm* the make variables) |

`sw/ctrl/tools/install-toolchain.sh` (P1.11) does these steps and writes the lock file. CMake finds
the tools through `TSR_RISCV_PREFIX`, `TSR_SPIKE` and `PATH`.

---

## 10. Verification

### 10.1 Oracle chain for this area

| Object | Oracle | Check |
|---|---|---|
| `tsr_ref_math.c` | (a) an independent numpy bit model in `sw/sim/tests/ref_math_np.py`, written from the VU document; (b) double-precision libm for accuracy | (a) bit-exact over all 2^32 inputs per unary function (opt-in `slow`; default: 2^24 strided + specials); (b) max ulp error ≤ VU's bound, reported; (c) RTL: VU blocks against the C function, exhaustively on a Verilator C++ bench (opt-in), 10^6 random + specials by default |
| `tsr_m_gelu16` | the pinned ggml CPU table `ggml_table_gelu_f16` (built by `ggml_gelu_f32` with libm `tanhf`) | all 65,536 fp16 inputs enumerated; mismatches reported; target (VU §6.5): < 1 % of the entries, no sign-of-zero mismatch, > 1 fp16 ulp only where GELU16 is correctly rounded and the table entry is not. Measured 0.415 % (265 at 1 ulp, 7 at 2 ulp; GELU16 correctly rounded in all) — the residual is the table's own f32 error (`tanhf`, cancellation for x ≪ 0), which the VU does not reproduce |
| VU/AT references in `tsr_ref_vu.c` / `tsr_ref_at.c` | numpy spec models (as numcheck B1b, `docs/spec.md:813`) | bit-exact on generated vectors; directed RED cases where a subtree sums to −0 (e.g. 0x80C00000 + 0x00800000) at m = 8k + r (VU §6.2) |
| `tsr_rv` | Spike | on the test bus profile (§4.2: RAM at 0x8000_0000, HTIF): riscv-tests rv32ui/um/uc/mi `-p` and riscv-arch-test RV32IMC/Zicsr signatures; random programs (in-house Python generator, placed at 0x8000_0000 on both) compared retire by retire with `spike --log-commits`; `test_ref_fp.c` built for RV32 and run on both (`docs/ctrl-riscv.md:1318-1325`) |
| Queue model | EX §7–§8 and CC §7–§9 | directed tests through `tsr_sim_push`; invariants I1–I3 asserted inside the model on every run |
| Race detector | — | positive and negative cases; mutated executors (a dropped WAR dep, a dropped ring release) must be reported |
| Firmware kernels | `sw/ctrl/ref` C++ references; the host build | RV build on sim bit-exact with the reference and with the host build; 8 schedules; race `fatal` |
| Backend and executor | ggml CPU | `test-backend-ops` NMSE per supported op (sim); klcheck KL (sim, then card) |
| Cross-transport | sim | identical logits and DRAM outputs on `mock`, `vkernel`, `xrt` for the same image and inputs |
| Launch traces | fresh-boot replay | every `TSR_TRACE` capture in ctest replays bit-exact on a freshly booted simulator (catches DTCM state leaks, V2-SW-4) |

### 10.2 ctest plan

The 42 existing project ctests:

| Fate | Tests |
|---|---|
| **Kept unchanged** (7) | `ref_layout`, `ref_fp`, `ref_ggml_fp16`, `ref_ggml_qact`, `ref_ggml_dequant`, `ref_dot`, `ref_ctypes` |
| **Kept, run on v2** (same assertions, new transport underneath) (15) | `ggml_tesseract_e2e`, `_tmax8`, `_hosted`, `_hosted_tmax8`, `ggml_tesseract_scalar_nmse`, `ggml_tesseract_backend_ops`, `rt_sim`, `rt_sim_tmax8`, `rt_mock`, `rt_mock_tmax8`, `rt_vkernel_k4t1`, `torch_pytest`, `torch_pytest_tmax8`, `transport_ggml_e2e_mock`, `_tmax8` |
| **Rewritten** for v2 semantics (13) | `sim_basic` (EX §15 item 2 on the simulator: CSRs, load with VERIFY, LQ / LCQ launch flow, HELLO, PING, BAD_DESC, cancel and stop, `SOFT_RESET`, errors, holds), `sim_mvm` (MVM through the queue), `sim_trace`, `sim_trace_ubsan` (trace v2), `sim_transport` (boot / launch / wait / soft reset), `transport_kdev` (protocol on mock: timeouts, stale core at open, adapter faults), `transport_xrt_noop` (xrt kdev against the no-op shim), `ggml_tesseract_units` (OPL builder: hazards, fusion, encoding), `tools_trace_gen_replay` (v2), `tools_xrt_selftest_mock`, `_tmax8`, `_1ch`, `_faults` → `tools_selftest_mock*` |
| **Rewritten settings** (FA on, q8_0 KV, rotation off) (5) | `tools_klcheck_selftest`, `tools_klcheck_qwen05b`, `tools_numcheck_selftest`, `tools_numcheck_dump_selftest`, `tools_numcheck_dump_qwen05b` |
| **Deleted** (2) | `tools_xrt_selftest_mock_m3`, `_m3_tmax8` (v1 m3 replays over the adapter), replaced by `tools_selftest_mock_traces` over the regenerated v2 m3 set |

New ctests:

- `ref_math_<fn>`, including `ref_math_gelu16` (the 65,536 fp16 inputs against the pinned
  `ggml_table_gelu_f16`);
- `ref_vu`, `ref_at`;
- `rv_isa` (the self-checking riscv-tests on `tsr_rv`, and on Spike when it is installed; without
  Spike the Spike runs are reported as not run), `rv_arch` (riscv-arch-test signatures of
  `tsr_rv` equal to Spike's), `rv_lockstep` (the riscv-tests on both, commit logs compared retire
  by retire) and `rv_random` (the last three skipped with a message if Spike is absent);
- `sim_queue`, `sim_race`, `sim_sched_determinism`, `sim_vu`, `sim_at`, `sim_dm`;
- `fw_runtime` (R1–R10: boot, HELLO id, PING, BAD_DESC, argument block, 8 outstanding launches,
  cancel and stop watermarks, TRAP record, panic as a TRAP with `mcause` 24, log drop and BLOCK);
- `fw_kernels_host` (ASan/UBSan) and `fw_kernels_rv` (per kind, 8 schedules, race `fatal`);
- `fw_selftest_sim`, `fw_replay_sim`;
- `trace_v2_roundtrip`, `trace_conv_model` (sample of 10 converted model traces);
- `transport_protocol` (EX §15 item 2, one binary for every kind: cold and warm boot, image-id
  mismatch, VERIFY with an injected TCM fault on vkernel, 0–7 args, credit rule, forced
  `LQ_OVERFLOW` and `RING_RD_BAD`, events while waiting, the debug-aware timeout);
- `ggml_tesseract_attn` (SET_ROWS + FLASH_ATTN with a KV buffer vs CPU, NMSE);
- `ggml_tesseract_graph_qwen_tiny` and `ggml_tesseract_graph_gemma_tiny`: exactly one Tesseract
  split (asserted with `GGML_SCHED_DEBUG=2`) and one launch per decode token, KL vs CPU on
  synthetic tiny models (V2-SW-21: a gguf-py script
  generates 2-layer Qwen2 and 4-layer Gemma 4 GGUFs with random weights, SWA + global layers and
  a 64-row vocabulary at build time; no binary is checked in).

### 10.3 tb (pytest) plan

The RTL areas own the RTL benches. The plan here fixes their fate:

| Fate | Benches |
|---|---|
| **Deleted** | `tb/cmdproc`; the M2 command-path suite `tb/core/test_core.py`; the v1 parts of `tb/common` (`csr_driver`, `proggen`, `pycore` command path, `replay`, `trace` v1) |
| **Rewritten** | `tb/core/test_core_m3.py` → `tb/core` v2 (issue-port tile-only replays of `push.bin`, MVM concurrency under queues); `tb/fastreplay` (C++ tile-only replay, then the full-system bench with Ibex and RVFI lockstep); `tb/platform` (`ap_ctrl_none` adapter; carries over the regmap / errors / misaligned / reset directed tests of `v1-m4:tb/platform/kernel_bench.cpp`, deleted whole in X0); `tb/platform/vkernel` (two clocks); `tb/run_all.sh` |
| **Kept** | `tb/dma`, `tb/fp`, `tb/mvm`, `tb/mvmback`, `tb/mvmfront`, `tb/qact`, `tb/spm`, `tb/tech` (unit ports; `tb/spm` and `tb/dma` extended by EX/AT for round robin and banking) |
| **New** | `tb/ctrl` (issue path, two clocks), `tb/vu`, `tb/at`, `tb/rv` (riscv-tests from ITCM, signatures against Spike), `tb/dbg` |

### 10.4 Gates

Common baseline settings (V2-SW-22): `-fa on`, `-ctk q8_0 -ctv q8_0`, `LLAMA_ATTN_ROT_DISABLE=1`,
klcheck's other fixed settings (`docs/klcheck.md:66-75`). `offload_kqv` is false in M6 (attention
on the CPU) and true in M7. The CPU reference and floor runs use the same KV type, FA and rotation
settings.

**M6 gate** (the spec criterion; MVMs are issued by RISC-V kernels because no other path exists;
`offload_kqv` = false, so attention runs on the CPU and the device AT ops are verified by the
selftest, `traces/at` and `ggml_tesseract_attn`, AT §13.6):

1. `tsr-selftest --device xrt` passes every test, including VU, AT and errors, with sim shadow,
   plus replays of `traces/m3`-v2, `traces/vu`, `traces/at`, `traces/ctrl` and
   `traces/model-v1conv`.
2. 0.5B Q8_0: `klcheck run --device xrt` generates; `klcheck run --device sim --tokens` replays the
   same tokens; `klcheck compare --expect-identical` must report every row bit-identical
   (8 prompts × 64 tokens).
3. 32B Q4_K_M: `klcheck all --device xrt` meets the KL criterion (mean KL ≤ 3 × floor, top-1
   ≥ 80 %, `docs/spec.md:801`).
4. GDB smoke on the card (DB): halt, register read, breakpoint in a kernel, step, resume, with the
   launch then completing correctly.
5. Recorded, not gated: tok/s for 0.5B and 32B (M4: 54.7 and 2.93 tok/s), launches per token,
   launch wall time.
6. Before the card: the same 0.5B identity on `vkernel` (2 tokens) and items 2–3 on `sim`.

**M7 gate:**

1. Qwen2.5-0.5B Q8_0, Qwen2.5-32B Q4_K_M, Gemma 4 12B (Q8_0 and Q4_K_M, on disk) and Gemma 4 31B
   (when a GGUF is on the build host; until then 12B alone gates Gemma) decode with **exactly one
   launch per decode token**, KV cache on the device (`offload_kqv` = true). Measured by
   `TSR_STATS` over ≥ 64 generated tokens with `GGML_SCHED_DEBUG=1` showing one Tesseract split per
   decode graph, and "staged chunks = 0" for Gemma 4 12B at `-ub 512` (R6). Prefill is one launch
   per ubatch.
2. Logits on the card bit-identical to sim for Qwen 0.5B (8 × 64) and Gemma 4 12B (2 prompts × 16
   tokens).
3. KL criterion on the card for Qwen 32B and the Gemma 4 models of item 1, including the 2K and
   16K long-prompt runs of AT §13.6 (multiple chunks, prefill CAUSAL tiles, SLIDE, the wrapped SWA
   ring).
4. `test-backend-ops` on sim passes every supported op, and `ggml_tesseract_attn` passes.
5. Recorded, not gated: DDR utilization U (`docs/spec.md:565-587`) with the H/G/F/E split for 32B
   and the largest Gemma 4; the Gemma 4 12B KL with the CPU reference built with and without
   `GGML_GELU_FP16` (the residual of GELU16, VU §6.9).

---

## 11. Work breakdown (P1–P6)

Sizes are *estimates*: lines of new or changed code, and ideal days for one implementer.

| Pkg | Content | Depends on | KLOC | Days |
|---|---|---|---|---|
| **P0** | X0 (EX §12): delete every v1-only part of every layer in one commit; remove the dependent ctests and benches | — | −(large) | 1 |
| **P1.1** | table source `sw/tables/tsr_tables.py` (EX / CC / DB / VU / AT rows from `docs/v2`) + `gen_tables.py` (SV, C, C++, Python, `--check`) | P0 | 0.6 | 2 |
| P1.2 | v2 host CSR port in `sw/sim` (EX §5: `CTRL` / `STATUS`, loader with VERIFY, LQ / LCQ / log rings and watermarks, `SOFT_RESET`, `Q_*`, PERF v2, `MTIME`) | P1.1 | 1.2 | 4 |
| P1.3 | `tsr_ref_math.c` (incl. GELU16), VU/AT references, numpy models, exhaustive tests, `ref_math_gelu16` | VU, AT docs | 1.6 | 6 |
| P1.4 | VU unit semantics in sim + cycle model | P1.3 | 1.2 | 4 |
| P1.5 | AT unit semantics in sim + cycle model | P1.3 | 1.0 | 4 |
| P1.6 | queue model `tsr_q.c` (dispatcher, checker with `OPERAND` and four tagged extents, queues, `ISSUE_PIDX`, errors, I1–I3) | P1.2 | 1.5 | 5 |
| P1.7 | `tsr_rv` + `tsr_bus` | P1.1 | 2.0 | 6 |
| P1.8 | scheduler, random schedules, fast-forward | P1.6, P1.7 | 0.5 | 2 |
| P1.9 | race detector + `SPM_STALE` | P1.6 | 1.0 | 4 |
| P1.10 | trace v2 library, capture, sim replay, `tsr-trace-conv` (v1 reader from `v1-m4:sw/sim/tsr_trace.c`), `tsr-trace-gen` v2 (from `v1-m4:sw/tools/tsr_trace_gen.c`), set regeneration | P1.6 | 2.0 | 6 |
| P1.11 | toolchain install script, Spike/riscv-tests/arch-test harness, random-program lockstep | P1.7 | 0.6 | 3 |
| P1.12 | debug-module model `tsr_dm.c` (DB's DMI profile, DB §9.1) | P1.7, DB doc | 0.8 | 3 |
| P1.13 | DB simulator models: stop logic, QSTEP, serial mode, the hard hold, history ring, REC FIFO, TRW records, log records (DB §5, §6.1, §8) | P1.6, P1.12 | 1.2 | 5 |
| **P2.1** | `sw/ctrl` build (ExternalProject, toolchain file), `tsr-elf` | P1.11 | 0.6 | 2 |
| P2.2 | crt0, `link.ld`, runtime R1–R10 (EX §4.7), traps and panic (DB §6.3), log records (DB §6.1) | P2.1, P1.2 | 0.9 | 4 |
| P2.3 | `kernel.hpp`, generated `vu.hpp`/`at.hpp`/`mmio.hpp`, `spm_ring`, host backend | P2.1, P1.6 | 1.2 | 4 |
| P2.4 | `plan.hpp`, OPL v1, executor core | P2.3 | 1.2 | 5 |
| P2.5 | M6 kernels: MUL_MAT kind, `selftest`, `replay` | P2.4 (VU/AT selftest parts after P1.4/P1.5) | 1.2 | 4 |
| P2.6 | M7 kinds: ADD/MUL/SCALE/RMS_NORM/GLU/ROPE/GET_ROWS/CPY/UNARY + references | P2.4, P1.4 | 1.5 | 5 |
| P2.7 | M7 SET_ROWS_KV and FLASH_ATTN + references | P2.4, P1.5 | 1.2 | 6 |
| **P3.1** | transport v2 (EX §10): protocol layer, debug-aware wait and recovery ladder, `sim`/`mock` kdevs, `xrt` kdev (`ap_ctrl_none`, compiles; card in P6), `vkernel` kdev port | P2.2, P1.2 | 1.6 | 6 |
| P3.2 | backend M6 (OPL builder MUL_MAT, cache, deletions; lazy src0 conversion, staging and `tsr_resolve_weight` restored from `v1-m4`, §9.3), `tsr_rt`/torch port | P2.5, P3.1 | 1.0 | 4 |
| P3.3 | tools: `tsr-selftest`, trace tools, `tsr-log`, klcheck/numcheck settings | P3.1, P2.5 | 1.5 | 5 |
| P3.4 | **M6 on sim**: ctest green, 0.5B and 32B klcheck on sim (FA on), `traces/model-v2` | P3.2, P3.3 | — | 2 (+ ≈ 1 day of 32B sim time) |
| P3.5 | backend M7: supports_op set (incl. the DUP / no-COMPUTE rule), KV buffers, MSUM producer and f32 mask copies (AT §7.3), builder (fusion, hazards, preload), compute heap (R6) | P2.6, P2.7, P3.2 | 2.3 | 9 |
| P3.6 | tiny-model generator; **M7 on sim**: one launch per token, `test-backend-ops`, klcheck Qwen and Gemma | P3.5 | 0.3 | 3 (+ sim time) |
| P3.7 | `libtsr_dbg` + `tsr-dbg`: DMI driver (DB §3.5–§3.6), RSP server (§4), monitor and tile commands, dumps (§7.4), the embedded `TSR_GDB` service; riscv-tests `debug/gdbserver.py` subset on sim | P1.12, P2.2, P3.1 | 3.0 | 10 |
| P3.8 | `tsr-prof` `trw` source and TSR_STATS v2 (DB §8.6) | P1.13 | 0.8 | 3 |
| **P4** | RTL (owners EX, VU, AT, CC, DB; sizes in their documents): command-path deletion, issue path and queues, VU, AT and SPM banks, Ibex + TCM + CDC, debug module + DMI, `ap_ctrl_none` adapter, unit benches | area docs; references from P1.3 | ≈ 8–12 SV + benches | per area |
| **P5.1** | tile-only fastreplay v2 (`push.bin` into the issue FIFO) | P4 issue path, P1.10 | 0.8 | 3 |
| P5.2 | full-system Verilator bench: Ibex + RVFI lockstep with `tsr_rv`, two clocks, random ratios | P4, P1.7, P2.2 | 1.5 | 6 |
| P5.3 | `vkernel` v2 (two clocks, `ap_ctrl_none`) | P4, P3.1 | 0.6 | 3 |
| P5.4 | riscv-tests / arch-test on RTL | P5.2, P1.11 | 0.3 | 2 |
| P5.5 | 0.5B on `vkernel` bit-identical to sim (2 tokens); v2 trace sets replayed | P5.3, P3.4 | — | 2 (+ run time) |
| P5.6 | Vivado OOC and the k4 link with the control core (≥ 276 MHz, D13 as restated) | P4 | — | 3 (links ≈ 4 h each with swap, `docs/platform-u200.md`) |
| **P6.1** | card bring-up: load, HELLO, PING, `tsr-selftest` | P5.6 | — | 2 |
| P6.2 | v2 trace replays on the card | P6.1 | — | 1 |
| P6.3 | GDB smoke (DB) | P6.1, P3.7 | — | 1 |
| P6.4 | **M6 gate** (§10.4) | P6.1–P6.3, P3.4 | — | 2 |
| P6.5 | **M7 gate** | P6.4, P3.6 | — | 3 |
| P6.6 | performance: U, H/G/F/E, launch costs | P6.4 | — | 2 |

- **Order.** P0 (the X0 deletion) first, then P1.1. P1.2–P1.13 in parallel where their
  dependencies allow. P2 on the simulator. P3.4 (M6 on sim) before any v2 RTL is needed. P4 runs
  in parallel from the area documents onward.
- **Critical path to M6:** P0 → P1.1 → P1.6/P1.7 → P2.2–P2.5 → P3.1–P3.2 → P3.4, then (with P4)
  P5.2 → P5.5 → P5.6 → P6.4; P3.7 (debug tools) must be done before P6.3, which the M6 gate needs.
- **Software totals:** ≈ 35 KLOC in P1–P3, ≈ 135 ideal days (*estimate*).

---

## 12. Interfaces with other areas

| Area | SW assumes from it | SW provides to it |
|---|---|---|
| **EX** (execution model) | the host CSR map (EX §5); the launch protocol (LQ / LCQ, descriptor and record formats, STATUS precedence, runtime contract R1–R10, cancel and stop watermarks, EX §4); image format, 64-bit image id, loader with VERIFY and warm restart (EX §3); `SOFT_RESET` (EX §8.5); unit ids and the opcode table (EX §7); error codes 1–8, 11–13 and the check order (EX §8.1); PERF v2 (EX §6); the `ap_ctrl_none` adapter (EX §9); the transport API (EX §10); the DRAM map (EX §11); the D16 deletion order (EX §12) | the simulator models of every EX register and ring (deterministic advance); the SDK runtime; the base kernels `graph`, `replay`, `selftest`; trace v2 and the trace migration; the table generator |
| **VU** | encodings, checks and numerics of VOP / RED / ROPE / TOPK / SOP / DQROW (VU §5–§6), including GELU16 / GEGLU16 and the RoPE angle recurrence; AT's vector ops on the VU (VU §5.11); the `tsr_ref_math` algorithms with coefficients as bit patterns; accuracy figures to measure; per-op cycle model | `tsr_ref_math.c` and its exhaustive tests; VU semantics in sim; `traces/vu`; kernels and the generated `vu.hpp` builders; host-computed constants (TS, M, 1/n, eps) |
| **AT** | the KV plane layout and its C conversion (AT §4); KVST, ATTQK, ATTPV and the vector ops (AT §5–§6); the normative FA sequence (AT §6.6); the MSUM format (AT §7.3); the `supports_op` contract (AT §2.6); the SET_ROWS_KV / FLASH_ATTN argument layout (AT §12) | SET_ROWS_KV and FLASH_ATTN kernels; KV buffers in the backend; the MSUM producer and f32 mask copies; AT semantics in sim; `traces/at`; `ggml_tesseract_attn` |
| **CC** | Ibex configuration and CSR list; the RISC-V address map and control window (CC §4–§5) including `ISSUE_PIDX`; TCM sizes; vector table and boot at `RV_BOOT` + 0x80; reset sequence and SR sequencer; interrupt map; `MTIME`; `RV_FREQ`; RVFI in simulation builds; F_RV | `link.ld`/crt0 matching the map; `tsr_rv` modelling the configuration (and its test bus profile); Spike harness; riscv-tests on RTL; lockstep co-simulation bench |
| **DB** | debug module behind DMI at CSR 0xC00–0xDFC; halt / resume / step, triggers, `HOLD_ON_HALT`; the GDB RSP server design; tile-level debug semantics; the log record header (DB §6.1); the trap and panic contract (DB §6.3); the debug-aware wait and recovery ladder (DB §4.1, §7.3) | `tsr_transport_csr_read`/`write`; `tsr_dm.c` and the DB simulator models (P1.12, P1.13); `libtsr_dbg` and `tsr-dbg` (P3.7); `tsr-prof` `trw` (P3.8); firmware with DWARF, `__tsr_dbg`, `__tsr_crash` and the launch hook; `riscv-none-elf-gdb` in the toolchain; the log decoder |

---

## 13. Decisions

| Id | Topic | Choice | Alternatives | Rationale |
|---|---|---|---|---|
| V2-SW-1 | RISC-V model | in-house `tsr_rv` interpreter inside `sw/sim`; Spike only as test oracle (on a separate test bus profile at 0x8000_0000) | link Spike as a library; Ibex RTL model as reference; QEMU | Keeps the simulator one deterministic C library with exact co-scheduling against the tile model and MMIO hooks. It also stays usable inside the llama.cpp backend. Spike still anchors ISA correctness through signatures and lockstep random programs, and RVFI lockstep anchors Ibex-specific behaviour (`docs/ctrl-riscv.md:1318-1325`). |
| V2-SW-2 | Time model | event-driven on core cycles, `fixed` by default, `random:<seed>` for determinism tests, spin/WFI fast-forward | cycle-by-cycle lockstep; untimed (ops complete at push) | Untimed hides every ordering bug. Cycle-by-cycle is slow on 32B shapes. Events with randomized durations expose schedule dependence cheaply, and fast-forward bounds the cost of polling joins. |
| V2-SW-3 | Race detector policy | vector clocks; `fatal` in ctest, `off` in model runs, `report` nightly | always on; hardware detection | The 2–3× cost (*estimate*) matters only on long model runs. Every kernel and executor path is already covered by ctest with `fatal`. |
| V2-SW-4 | Launch state | launches are stateless: results depend only on image, args and DRAM; `SPM_STALE` check; fresh-boot trace replay | traces with SPM and DTCM snapshots; resident state across launches | Makes every per-launch trace self-contained and replayable on any target from a fresh boot. The op list and tables are small enough to reload per launch (≈ 150 KB for 32B, *estimate*). Resident state can come later (M9) with its own trace rules. |
| V2-SW-5 | Trace layout | one directory per launch; single push-ordered `ops/push.bin` with relative tickets; images shared by id | per-unit `ops/u<k>.bin` (spec text); one ELF copy per trace | Replay needs the global push order (a dep must not point to an unpushed ticket); per-unit files lose it, and the per-unit view is a filter. Relative tickets make traces independent of history. Sharing images avoids 290 copies. The spec's trace paragraph (`docs/spec.md:823-830`) needs the matching edit. |
| V2-SW-6 | v1 trace migration | regenerate m2/m3/gen with the v2 generator; convert the 290 model traces as a serial chain (`replay` SERIAL); drop `traces/tmp` | convert everything (EX's and CC's drafts) | Generators and seeds exist, and 254 error expectations must be re-derived for v2 anyway. Model traces are large and error-free, and they convert exactly, which gives a strong v1 → v2 unit-equivalence check. Adopted by EX §12 and CC §15.3. |
| V2-SW-7 | Toolchain | xPack `riscv-none-elf-gcc` 14.2.0-3 (with GDB) in user space, the only supported compiler; Spike and dtc from source; Clang optional | distro packages (need sudo; no rv32imc multilib, no libstdc++ headers); building GCC from source; Clang only | No sudo is needed, versions are pinned, and the result is reproducible. The xPack bundle includes GDB for D18. Clang alone lacks an rv32 compiler-rt here. |
| V2-SW-8 | Firmware language rules | freestanding C++20, no heap/exceptions/RTTI, `-Os`, libgcc only, no libstdc++ headers; no f32 math in firmware (the host precomputes constants) | allow soft-float in kernels; C | Avoids soft-float cost and any risk of a firmware-side fp rounding path (the spec fp rules stay in host-computed constants and in device ops). C++ templates keep the op builders zero-cost (D6). |
| V2-SW-9 | Image identity | EX §3.2: build-time SHA-256 over the `PT_LOAD` segments, first 8 bytes as a little-endian `uint64`, stored in an 8-byte `.tsr_id`, echoed in HELLO r2 / r3 | a 16-byte id (the draft); runtime hashing of TCMs; CRC only | Hashing 128 KiB on Ibex would take ≈ 80 ms (*estimate*). 64 bits tell builds apart; the load CRC and VERIFY prove the transfer. Excluding the not-loaded log strings keeps the id stable across log-text edits. |
| V2-SW-10 | Host ↔ RISC-V protocol | **withdrawn**: EX's LQ / LCQ protocol (V2-EX-1 … V2-EX-5) with the runtime contract R1–R10; up to `LQ_DEPTH` launches outstanding | the single-slot H2R / R2H mailbox with one outstanding launch (the draft) | EX owns the launch protocol; the queue lets the host pipeline launches with posted writes only, and one channel carries launches, answers, HELLO, TRAP and events. |
| V2-SW-11 | Logging | binary records (format-string address + args) with DB's header (DB §6.1) through EX's log ring; whole-record drop with a counter; runtime filter `LOG_CFG.LEVEL` | text over the LCQ; DRAM log ring; SW's own 2-bit-level header and a LOGCTL message (the draft) | No TCM is spent on strings, and the ring is readable even when queues are wedged, unlike a DRAM ring that needs a working DMA (`docs/ctrl-riscv.md:1190`). Whole-record drop keeps the stream parseable. One header for the encoder and the decoders. |
| V2-SW-12 | Graph execution model | host builds a pre-resolved ggml-level op list (OPL), computes the dependencies, and caches it per graph; firmware kernels per kind | host emits tile-op programs for a replay kernel; firmware parses ggml tensors and computes hazards | Static tile programs cannot express per-token KV write rows without register-relative addressing (removed, D14). Parsing ggml views and hazards in firmware would cost TCM and RISC-V cycles. The host does the ggml work once per graph, and the firmware reads fixed-size args. |
| V2-SW-13 | Data placement in M7 | tensors stay in DRAM between ops; per-op SPM scratch from a ring with release tickets; fusion of bias/residual through MVM ACC and of RMS_NORM × weight | keep intermediates resident in SPM; no fusion | DRAM round trips cost ≈ 2 % of a 32B token (*estimate*) and are simple to make race-free. Fusion through ACC is bit-identical (one commutative f32 add) and removes the most common round trips. SPM residency is an optimisation for after the gate. |
| V2-SW-14 | M6 llama.cpp path | the same `graph` executor with MUL_MAT-only op lists, one launch per v1 split (adopted by EX as V2-EX-29) | a `mul_mat` kernel per MUL_MAT; keeping tile-program lowering with `tsr.oplist_seq` (EX's draft) | One mechanism from M6 to M7, so M7 only adds kinds and supports_op entries. Splits stay v1's, so M6's KL and identity gates compare like with like. |
| V2-SW-15 | Index inputs | preload all RISC-V-read graph inputs at launch start with one join: KV write indices, row ids, positions and each KQ mask's MSUM | LD and join per op | A join per SET_ROWS or FLASH_ATTN would stall run-ahead behind every queued DMA op (2–3 × layers per token). One join at the start costs one DMA latency. |
| V2-SW-16 | KV rotation | `LLAMA_ATTN_ROT_DISABLE=1` in device runs and in every baseline; device rotation is a follow-up | device f32 MUL_MAT for graph-input matrices; a Walsh-Hadamard kernel; leave the rotations on the CPU | The rotation matrix is an F32 graph input that MVM cannot use, and leaving it on the CPU breaks one launch per token. Comparing against a CPU with the same setting keeps the KL criterion fair. q8_0 KV gains little from rotation, compared with 4-bit KV. |
| V2-SW-17 | KV buffers | recognise llama.cpp KV buffers by tensor names at first access; AT layout in a KV heap; converting get/set; context shift unsupported in M7 | a ggml-identical byte layout on the device; a custom buffer type | llama.cpp allocates the KV cache through the device buffer type without a usage flag (`docs/llama-pin.md:193-200`). Names are the stable signal. Converting accessors keep state save/load working whatever layout AT picks. |
| V2-SW-18 | Host protocol | one protocol layer over all kdevs implementing EX §10; `sim` becomes a kdev with `idle()` | separate sim transport (today) | The exact host code path that runs on the card (bind, load, VERIFY, launch, wait, recovery, log) is then tested on every sim and mock run. |
| V2-SW-19 | Test access to queues | `tsr_sim_push`/`tsr_sim_qctrl` test hooks in the simulator library, never in `tsr_transport` | no hooks (every test needs firmware); expose to the host | Lets directed queue tests and tile-only sim replays run without firmware, while D15 still holds for every real device path. |
| V2-SW-20 | Planning code | `plan.hpp` freestanding, compiled for host and firmware | separate host and firmware implementations | `supports_op` must reject exactly what firmware cannot plan; one source makes that true by construction. |
| V2-SW-21 | Gemma test models | synthetic tiny Gemma 4 and Qwen2 GGUFs generated by a gguf-py script at build time; real 12B (on disk) and 31B (when available) for gates | only real models; checked-in binaries | The smallest real dense Gemma 4 is 12B, too slow for ctest on sim. Generated models cover SWA/global layers, per-head norms and GEGLU in seconds without binaries in git. |
| V2-SW-22 | Baselines | FA on, q8_0/q8_0 KV, rotation off for device and CPU runs from M6 on; KV on host (`offload_kqv` = false) in M6, on device in M7 | keep FA off until M7 | D17 requires FA on. Fixing the KV type and rotation already in M6 keeps baselines constant from M6 to M7, so the M7 delta measures the device attention only. |
| V2-SW-23 | Approximate-math verification | exhaustive 2^32 bit-exact check against an independent numpy model, ulp bounds against double, RTL exhaustive opt-in; GELU16 enumerated over all 65,536 fp16 inputs against the pinned ggml table | random sampling only; MPFR | Unary f32 functions are fully enumerable, which leaves no gap for coding errors. The numpy model is written from the VU text and does not share code. Double-precision references are enough for f32 ulp classification. |
| V2-SW-24 | Launch watchdog | 60 s default, counted only while the RISC-V is not halted and no tile stop is active, off in SERIAL mode; the mutex is released between polls; on timeout DB's ladder: dump, `LQ_CANCEL`, then `SOFT_RESET` + warm restart | an unconditional timeout with dump and reset (the draft); no timeout | A wedged queue or a parked hart must not hang llama.cpp, but a GDB user stopped at a breakpoint inside llama.cpp must not have the device reset under them (D18). The dump preserves the post-mortem before any destructive step. |
| V2-SW-25 | Kernel test builds | host build (ASan/UBSan) plus RV build on the simulator, the RV build being the reference | RV only; host only | The host build finds logic bugs in milliseconds with sanitizers; only the RV build proves control flow, MMIO timing behaviour and ABI. |
| V2-SW-26 | Compute buffers | a per-channel compute heap between the weights and the KV planes, sized from the reserve graph, plus ch0 IO and ch1–ch3 [0, 512) MiB; `get_max_size` ≥ the largest tensor | fixed IO windows (ch0 IO + ch1–ch3 low 512 MiB, the draft); ch0 IO only (v1) | Gemma 4's 512 MiB logits chunk at `-ub 512` fills a whole fixed window; a heap sized from the reserve graph keeps every chunk on the device (one launch per token) and spreads activation DMA over the channels. |
| V2-SW-27 | numcheck on v2 | rely on per-node observation, which gives one launch per node | a special per-op launch mode in the backend | llama.cpp's `cb_eval` already splits the graph per observed node; no backend mode means no second code path to keep correct. |
| V2-SW-28 | Card selftest | `selftest` kernel with in-process sim shadow comparison | host-written tile programs (forbidden by D15); fixed expected files | D15 requires kernels. Running the same image on the simulator gives exact expectations for any test parameters without stored golden files. |
| V2-SW-29 | Op-list size | ≤ 4096 ops per launch (32 KiB of DTCM tables); larger graphs are split by the builder | unbounded with streamed tables | Decode graphs need ≈ 1,300–2,000 ops (*estimate*). A fixed cap keeps DTCM use static and the executor simple. |
| V2-SW-30 | Timing independence | firmware never lets a timing-dependent read influence pushes, deps, stores or records (only waits); the ring allocator decides deps by count; `ISSUE_PIDX` and `ISSUED_SEQ` are exact | allow opportunistic decisions | This is what makes programs valid in the sense of §4.7, so card logits can be bit-identical to the simulator. |
| V2-SW-31 | GELU in llama.cpp graphs | GGML GELU and GEGLU map to the VU's GELU16 / GEGLU16 (the CPU's fp16-table form); the exact tanh GELU stays for M8 / PyTorch | exact tanh GELU everywhere (the draft's default) | The pinned ggml CPU defines `GGML_GELU_FP16` (`vec.h:46`) and rounds x and the result to fp16 (relative error up to 3.9·10⁻³); matching that form removes an fp16-level difference on every GEGLU element of every Gemma 4 layer. |
| V2-SW-32 | Mask summaries | the backend writes the MSUM (and, for GENERAL masks, an f32 copy) on `set_tensor` of tensors named `kq_mask`, reserving the bytes through `get_alloc_size`; the FA record refers to it through the preload table | the device reads F16 masks; per-token op-list patching | The VU has no f16; the summary keeps the cached op list valid per token and lets ATTQK skip masked bandwidth (AT V2-AT-14). |
| V2-SW-33 | Unselected graph branches | `GGML_OP_DUP`, non-KV SET_ROWS and nodes without `GGML_TENSOR_FLAG_COMPUTE` are unsupported; `graph_compute` skips flagless nodes and makes no launch for an empty split | rely on "everything else false" | `ggml_build_forward_select` leaves unselected embedding branches in every decode graph, and the scheduler ignores the flag when it assigns backends; a supported DUP would add a second split and launch per token. |

---

## 14. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| `tsr_rv` diverges from Ibex in a corner (CSR WARL, misaligned split, interrupt timing) | sim/card mismatch in firmware behaviour | Spike signatures and random lockstep; RVFI lockstep in P5.2; the timing rule makes timing differences harmless |
| Host hazard analysis misses a dependency | racy op lists; card results differ from sim | race detector `fatal` on every ctest graph; 8 schedules; executor mutation tests; `SERIAL` flag and DB's serial mode for bisecting on the card |
| The RISC-V cannot stay ahead on small models (SPM-window reads of op records) | G grows for 0.5B | compact pre-resolved records (≤ 1 slot for most kinds); the line-buffered SPM alias; measured in P3/P6 with `TRACE_MARK` records and `MVM_EMPTY`; fall back to LD-ing hot records into DTCM if needed |
| llama.cpp graph features outside supports_op (rotation, context shift, new ops after a pin bump) | extra splits or scheduler aborts | R1–R10 checked in the graph ctests with `GGML_SCHED_DEBUG`; rotation off by default; clear errors for unsupported KV operations |
| GELU16 cannot reproduce the CPU's libm `tanhf` table entries exactly | a residual fp16-ulp difference on some GEGLU elements | measured by `ref_math_gelu16` (target of VU §6.5; W0: 0.415 % of the entries, 7 at 2 ulp, all where the table entry is the inexact one) and by the Gemma 4 12B KL with and without `GGML_GELU_FP16` before the VU RTL freeze |
| Toolchain availability (xPack multilib, Spike build deps) | blocks P2 | the libgcc fallback multilib; Clang + conda as a second route; Spike optional for the build, needed only for `rv_*` tests |
| Simulator speed for 32B/Gemma runs with RV interpretation | slow gates on sim | fast-forward; race detector off by default; `TSR_SIM_THREADS`; short prompts for the sim identity runs |
| `xrt::ip` with `ap_ctrl_none` and BO bank discovery | host cannot open the CU | *confirm* in P3.1 against the no-op shim and in hw_emu if needed; EX §9.1 fallback; the adapter keeps the channel-base registers |
| v1 deleted now (D16) while v2 is not on the card | no in-tree path to run models | `build/v1` binaries and tag `v1-m4` for comparisons; the sim path is end to end before RTL (P3.4) |
| k4 link time and 14 GB host RAM | slow P5.6/P6 iterations (≈ 4 h per link with swap) | k1 and OOC first; link only frozen RTL; card runs batched |
| Log ring full during long kernels | lost diagnostics | whole-record drop with counters; host drains during waits; levels; BLOCK mode |
| Composite-kernel sequence changes after baselines | KL baselines silently stale | sequence documented per kernel; firmware ABI bump forces rebaseline |
| No Gemma 4 31B GGUF on the build host | the M7 gate covers Gemma with 12B only | 12B is on disk and confirmed; 31B rows stay *confirm* and are added when a GGUF is available |

---

## 15. Open items (blocked on measurements or third-party checks)

- The xPack release and multilib set, the Spike build dependencies (dtc, Boost) and the
  riscv-tests make variables (*confirm*, §9.6).
- `xrt::ip` register access and BO memory-bank lookup for an `ap_ctrl_none` compute unit without an
  `xrt::kernel` (*confirm*, §9.1, P3.1).
- Card measurements: launch round trip and pipelined rate, PCIe CSR read and posted-write latency
  (still unmeasured, `docs/spec.md:994`), and the RISC-V run-ahead margin of the executor on 0.5B.
- The Gemma 4 KL gap of GELU16 (§10.4). The mismatch rate against the pinned table is measured
  (W0: 0.415 %, GELU16 correctly rounded in every mismatch, VU §6.5).
- Device support for llama.cpp's Hadamard KV rotation (V2-SW-16 follow-up), M8 or later.
- Spec edits that follow this document (at the spec v2 rewrite, README): the trace paragraph
  (`docs/spec.md:823-830`), the CMD region → OPL region in the DRAM map (`docs/spec.md:134`), and
  the transport row of the backend table (`docs/spec.md:489`).
- Unrelated observation: `git status` shows the `third_party/llama.cpp` submodule with modified
  content; the v2 design work made no changes there.

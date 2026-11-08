# Tesseract — area SW: software stack, firmware SDK, simulator, verification and work plan

Status: **design, not implemented** (2026-10-30). One of the six area documents in `docs/design/`
(EX execution model, VU vector unit, AT attention and KV, CC control core, DB debug, SW this
document). Source-of-truth order: `docs/spec.md` > ISA simulator (`sw/sim`) > the area
documents > `docs/ctrl-riscv.md`. The user decisions D1–D6, D12 and D14–D18
(`docs/ctrl-riscv.md`, user decisions) are fixed. Every choice marked SW-n (§13) was made under the user's
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
  semantics, and `tsr_ref_math.c`. The host starts work only through the launch queue and
  completion ring (D14).
- **Trace format.** One directory per launch: the firmware image (by id), the launch descriptor, DRAM
  init and expect regions, the DONE record, per-unit ticket records, and the pushed op stream in
  push order (`ops/push.bin`) with relative tickets. Tile-only traces replay through a standard
  `replay` kernel, so they also run on the card without host access to the issue port (D15).
- **Firmware.** `sw/ctrl` is a freestanding C++20 SDK: crt0, linker script, a runtime that
  implements EX's launch contract R1–R10 over the launch queue and completion ring (EX §4), traps
  and panic per DB §6.3, binary logging with DB's record header (DB §6.1) through the log ring, a
  kernel table and the tile/VU header library. Three base kernels exist from M6: `graph` (the
  op-list executor), `selftest` and `replay` (EX §4.6).
- **Graph execution.** The ggml backend turns each split into a pre-resolved op list in DRAM
  (the op list format, §7.2) and launches `graph` once. In M6 the list holds only MUL_MAT records; M7
  adds the op kinds that make one decode token one launch (§8). The host computes the dependencies
  between ops. The firmware maps them to tickets and issues ahead, so units overlap (D3).
- **Host.** EX's transport API (EX §10) with one protocol layer (bind, load and VERIFY, boot,
  launch, wait, cancel, stop, soft reset, log) over every kdev: `sim` (in process), `mock`,
  `vkernel` and `xrt` (`ap_ctrl_none` with polling).
- **Toolchain.** No RISC-V compiler, Spike or RISC-V GDB is installed on the build host (checked
  2026-10-30). The plan installs the xPack `riscv-none-elf-gcc` (with its GDB) and builds Spike
  and dtc from source, all in user space, no sudo (§9.6).
- **Gates.** M6: on the card, Qwen2.5-0.5B Q8_0 logits bit-identical to `TSR_DEVICE=sim`, and the
  32B Q4_K_M KL criterion met with FA on and attention on the CPU (`offload_kqv` = false, §10.4). M7: Qwen2.5 and Gemma 4 dense run with exactly
  one launch per decode token, with logits bit-identical to sim, the KL criterion met, and the DDR
  utilization measured.
- **Work plan.** P1 references and simulator → P2 SDK and kernels on sim → P3 host and llama.cpp
  end to end on sim → P4 RTL → P5 Verilator full system → P6 card (§11). The software path works
  end to end on the simulator before the RTL for the control core exists.

---

## 2. Host interface (D14, D16)

The host starts work only through EX's launch queue and completion ring (§6.4); the transport API is
load, launch and wait (§9.1). Tile ops are issued by firmware: their address fields are immediates
and ra/rb/rc/rd must be 0 (`docs/spec.md` §11), tiling is in `plan.hpp` (shared by host and
firmware), and op lists live in the OPL region (§7.2).

---

## 3. Repository layout (software)

```
sw/
├── tables/                  # tsr_tables.py: the single table source (EX/CC/DB/VU/AT own rows, EX-30)
│                            # gen_tables.py: generator (SV package, C header, mmio/vu/at.hpp, tb Python)
├── sim/                     # libtsr_sim, libtsr_ref (C, host only)
│   ├── include/             # tsr_isa.h (+ generated tsr_tables.h), tsr_ref.h, tsr_ref_math.h, tsr_ref_vu.h,
│   │                        # tsr_ref_at.h, tsr_sim.h, tsr_rv.h, tsr_bus.h, tsr_dm.h, tsr_dbg.h, tsr_race.h, tsr_trace.h
│   ├── tsr_ref.c            # references (fp helpers, layouts, QACT, dot products, dequant)
│   ├── tsr_ref_vu.c         # VU op references: VOP, RED, ROPE, TOPK, SOP, DQROW (VU §5-§6)
│   ├── tsr_ref_at.c         # AT references: KV planes, KVST, ATTQK, ATTPV, vector ops, FA (AT §13.1)
│   ├── tsr_ref_math.c       # EXPK, SIGMOID, SILU, GELU (tanh, fp16 form), TANH, SIN, COS (algorithms: VU §6)
│   ├── tsr_sim.c            # core state, host CSR port (EX §5), loader, LQ/LCQ/log rings, SOFT_RESET, PERF,
│   │                        # op handler table, tile units (LD, ST, QACT, MVM)
│   ├── tsr_vu.c, tsr_at.c   # VU and AT unit semantics: checks, extents, execution, cycle models
│   ├── tsr_q.c              # issue port, dispatcher, checker order, unit queues, tickets, deps, SYNC, errors, time base
│   ├── tsr_sim_int.h        # internal interface between the sim's files (handler table, accessors, hooks)
│   ├── tsr_rv.c, tsr_bus.c  # RV32IMC+Zicsr interpreter; RISC-V address map and MMIO
│   ├── tsr_dm.c             # debug module model (area DB's contract, DB §9.1's DMI profile)
│   ├── tsr_dbg.c            # DB simulator models: stop logic, QSTEP, serial mode, TBPs, history, probes, REC, TRW,
│   │                        # the DBG registers, the DMI gate, DB's halt requests, RV watchpoints (P1.13)
│   ├── tsr_sched.c          # TSR_SIM_SCHED, the random schedule's draws and RISC-V-side view (P1.8; the fixed
│   │                        # time base is in tsr_q.c, spin fast-forward in tsr_bus.c's RISC-V agent)
│   ├── tsr_race.c           # race detector and SPM_STALE
│   ├── tsr_trace.c          # trace read / write / capture / replay
│   └── tests/
├── ctrl/                    # firmware SDK (RISC-V), separate CMake project (ExternalProject)
│   ├── CMakeLists.txt       # the firmware project: tsr-fw.elf (+ .map, size report) and test images
│   ├── cmake/               # riscv32.cmake (toolchain file), tsr_note.S (.note.tsr)
│   ├── include/tsr/         # kernel.hpp, mmio.hpp (generated), vu.hpp, at.hpp, plan.hpp, opl.hpp, log.hpp,
│   │                        # rt.hpp (tsr_args, TSR_KERNEL), exec.hpp (kind handlers), traits.hpp,
│   │                        # backend_dev.hpp, backend_host.hpp
│   ├── rt/                  # crt0.S, link.ld, runtime.cpp, trap.cpp, log.cpp, libc.cpp (mem*), rt_int.hpp
│   ├── kernels/             # graph.cpp (executor), k_mulmat.cpp, k_norm.cpp, k_elem.cpp, k_rope.cpp,
│   │                        # k_glu.cpp, k_rows.cpp, k_kv.cpp, k_fattn.cpp, selftest.cpp, replay.cpp
│   ├── host/                # main-project side: tsr-elf, the tsr_fw ExternalProject, tsr_fw_paths.h,
│   │                        # the host build of the kernels (backend_host.cpp, shim_host.cpp, §6.8)
│   ├── ref/                 # host C++ references of composite kernels (tests): mulmat_ref.cpp,
│   │                        # selftest_abi.hpp, fw_sim_host.hpp
│   ├── tests/               # fw_* ctests (build, hdr, runtime, exec, kernels, e2e .cmake snippets)
│   └── tools/               # tsr-elf/ (image id, kernel table, log-format table, size report),
│                            # install-toolchain.sh (§9.6)
├── transport/               # tsr_transport (C API), protocol layer, kdevs: sim, mock, vkernel, xrt
├── ggml-tesseract/          # backend: weights, tsr-opl.* (op-list builder), tsr-kvbuf.* (M7)
├── rt/, torch/              # tsr_rt on op lists; PyTorch above tsr_rt
└── tools/                   # tsr-selftest, tsr-trace-gen/-replay/-conv, tsr-log, klcheck, numcheck, tsr_bw
```

`third_party/` holds no RISC-V tool sources. The toolchains live outside the repository (§9.6),
and their versions are recorded in `docs/design/toolchain.lock` (created in P1.11).

---

## 4. Simulator

### 4.1 Model overview

The simulator is one C library, linked into the backend, the tools and the tests. It models:

- the **core** (core clock domain): SPM, per-channel DRAM, the host CSR port (EX §5: CSRs, the LQ /
  LCQ / log rings, `Q_*`, PERF, DMI), the dispatcher, the checker, the unit queues, and the units
  DMA, QACT, MVM and VU with AT's ops on them (EX §7.3), plus SYNC;
- the **control core** (RISC-V clock domain): `tsr_rv`, ITCM and DTCM, the control window, the
  core window, the SPM window and the debug module.

Functional results come only from the C references (`tsr_ref*.c`)
(`docs/spec.md` §22, rule 3). An op's effects (SPM and DRAM writes, scalar-register writes) are
applied atomically at its modelled completion time. For race-free programs this choice is
invisible (`docs/ctrl-riscv.md` §9.1).

### 4.2 RISC-V model `tsr_rv` (SW-1)

- **ISA:** RV32IMC + Zicsr + Zifencei, M-mode only (CC's configuration). An unimplemented encoding
  or CSR raises an illegal-instruction trap (`mcause` 2, `mtval` = the instruction bits).
  `fence.i` flushes nothing (there is no cache); it costs what a jump costs (Ibex implements it
  as a jump to the next instruction).
- **Misaligned load/store:** split into two aligned accesses in address order, like Ibex
  (`docs/ctrl-riscv.md` §3.5), and not atomic. If the second half faults, the trap reports the
  original address.
- **CSRs:** exactly CC's list for the pinned Ibex configuration. Expected: `mstatus`, `misa` (read
  only), `mie`, `mip`, `mtvec` (vectored only: MODE reads 1, base 256-byte aligned), `mscratch`,
  `mepc`, `mcause`, `mtval`, `mcycle[h]`, `minstret[h]`, `mcountinhibit`, `mhpmcounter3-31[h]`
  and `mhpmevent3-31` (read 0), `mvendorid`/`marchid`/`mimpid`/`mhartid`, `tselect`,
  `tdata1-3` (2 execute triggers), `dcsr`, `dpc`, `dscratch0-1` (debug mode only). CC's document
  is authoritative.
- **Interrupts:** `irq_fast[0..5]` and `irq_timer`, as in `docs/ctrl-riscv.md` §3.7 and as
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
- **Timing** (schedule only, never architectural): two cycle tables, one per Ibex configuration
  of CC §3.2, from Ibex's documentation (`doc/03_reference/pipeline_details.rst` stall cycles,
  `rtl/ibex_id_stage.sv` for the writeback stage; *confirm* items checked by the RVFI bench P5.2),
  in `tsr_rv.h` (`TSR_RV_CPI_MAXPERF`, `TSR_RV_CPI_SMALL`), selected by `TSR_SIM_RV_CORE` =
  `maxperf` (default: CC's decided configuration, CC-34) | `small` (`tsr_sim_opts.rv_core`;
  `tsr_rv_set_core`). Per class, maxperf / small: ALU, LUI / AUIPC, fence, not-taken branch and
  CSR read 1 / 1; a CSR write other than `mscratch` / `mepc` 2 / 2 (pipeline flush); taken branch
  2 / 3; JAL, JALR, `fence.i` 2 / 2; load / store base 0 / 1 plus the bus latency (TCM 1), and on
  maxperf + 1 when the next instruction reads the loaded register; MUL 1 / 3, MULH* 2 / 4;
  DIV / REM 38 / 38 (2 by zero); + 1 for a jump or taken branch to a 32-bit instruction at
  pc[1] = 1 (second word fetch); traps and `mret` 3. Bus latencies (`tsr_bus.h`): control window
  3 RV cycles (`clk_rv` registers, so 3 / 4 RV cycles per access in total, maxperf / small); the
  maxperf TCM cost (base 0, one TCM access per cycle) is derived from Ibex's RTL and stays *confirm*
  (CC §3.2, §4.3; RVFI bench P5.2); the core window and the SPM
  bridge cross into the core domain, 2 RV cycles plus 12 / 15 core cycles converted at the run's
  ratio (9 / 11 RV cycles at 166.7 MHz, 13 / 16 at 250 MHz); line-buffer hit 1. P1.8's
  `tsr_sched.c` varies the CPI and the handshake crossings under random schedules. The RV-to-core
  clock ratio comes from `TSR_SIM_RV_RATIO`, default 166.7 / 276.3 = 0.603 (the k4 core clock).
  `tsr_rv_get_stats` counts instructions and cycles per class for the perf harnesses.
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

The unit code paths in `tsr_sim.c` (LD, ST, QACT, MVM and their checks) start when the queue head
starts entry e. VU and AT ops are in §4.5. The host CSR port serves EX's map (EX §5): `CTRL` / `STATUS`, the
loader with receive CRC and VERIFY, `RV_BOOT`, the LQ / LCQ / log rings and watermarks,
`SOFT_RESET` (EX §8.5, CC §12.2), PERF, `Q_*`, `SREG`, the DBG block (`tsr_dbg.c`) and DMI (`tsr_dm.c`).
Reads have no side effects except EX's PERF LO snapshot (`docs/spec.md` §21, CSR reads while running).

### 4.4 Scheduler and time base (SW-2)

- **Time** is in core cycles (u64). The event kinds are unit-op completion, dispatcher crossing,
  RISC-V progress (RV cycles × ratio), the LQ / LCQ / log crossings, and the `MTIMECMP` timer.
- Ties are broken deterministically by (time, event-kind priority, unit id, sequence).
- **`TSR_SIM_SCHED`:**
  - `fixed` (default): op durations come from the cycle model, the issue-FIFO crossing
    takes 2 cycles, and the RISC-V-side synchronised copies (`DONE_SEQ`, credits, `QSTATUS`, error
    records, interrupt levels) show the core state without lag (the W1a time base, bit-identical).
  - `random:<seed>[,<min>,<max>]`: every op duration is multiplied by a factor drawn uniformly
    from [min, max] (default 0.25–4); crossings take 2–8 cycles; the RV CPI of each instruction
    varies by ±1 cycle. The generator is xoshiro128**, seeded from `<seed>`.
- **Fast-forward.** WFI jumps to the next event that sets a pending, enabled interrupt. A spin
  also jumps. A spin is the same timing-dependent register (`DONE_SEQ[u]`, `ISSUE_CREDIT`,
  `QSTATUS`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) read 4 times in a row with an unchanged value
  and no store, push or other MMIO access in between. The hart then jumps to the next event that
  changes any of these registers. This bounds simulation time when a 32B MVM (311 K core cycles,
  `docs/ctrl-riscv.md` §5.6) is joined by polling. Poll counts are therefore not architectural, which
  the determinism rules already require (§4.7).
- **Host interaction:** `tsr_sim_advance()` runs until a host-visible event (an `LCQ_WR` change,
  log words, RV halted or trapped, `RV_RESET_DONE`, `SR_BUSY` falling, a `QIDLE` change) or until a
  cycle budget runs out (§4.8).

### 4.5 Device op semantics and references

| Ops | Semantics source | Reference code | Status |
|---|---|---|---|
| LD, ST, QACT, MVM, NOP, FENCE | `docs/spec.md` §11 and §21 | `tsr_ref.c` | register fields must be 0; END/LI/ADDI/CALL and opcodes 0x04–0x09 are ILLEGAL_OP |
| VOP (incl. GELU-tanh, GELU16, GEGLU16, TANH), RED (segmented, post RCP / RMS), ROPE (angle recurrence), TOPK, SOP, DQROW | VU §5–§6 (base: `docs/spec.md` §9 and §11) | `tsr_ref_vu.c` (ops), `tsr_ref_math.c` (approximations) | in the simulator: `tsr_vu.c` (P1.4; TOPK and DQROW from M6 in the simulator, their RTL is M8). VMASK is M9 |
| KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM | AT §5–§6 (vector ops executed by the VU, VU §5.11) | `tsr_ref_at.c` (AT §13.1) | in the simulator: `tsr_at.c` (P1.5) |
| QCLEAR, QABORT, hold control entries | area EX | `tsr_q.c` | in the simulator |

**Cycle models (SW-80).** The durations of the ops are estimates, not part of the bit-exact contract, but the card measures
them (the trace writer's RUN time of every op, DB §8.3) and the RTL benches time them, so the models follow what they show.
`tsr_vu.c` and `tsr_at.c` take the VU's lanes from `tsr_sim_opts.vu_timing` (`TSR_SIM_VU_TIMING`: `u200`, the default, 16 / 16 / 8 lanes
and no lane pipe; `k4`, the card image's 8 / 8 / 4 lanes behind two lane-pipe stages; or `F,E,B,P`) and charge the cycles of an op
that waits for the one before it as the RTL unit takes them (VU §7.9, §7.13: latencies of 59 to 95 cycles, the reduction's partial ring,
the two-stage ROPE, the AT vector ops). QACT takes 16 cycles per 32-block of an A32 vector and 17 per 256-block of an A256 one, plus 44
and 72 (`tsr_qact.sv`: the A32 quantiser is divider bound; `tb/qact` `qact_timing`). A store takes its lines plus 110 cycles
(54 up to 8 lines) and an MVM starts 144 cycles before its first weight beat counts, the DDR latencies of the card (the RTL benches
run on ideal memory). `TSR_SIM_CARD=k4full` sets the whole card image: `T_MAX` 4, `CAPS` 0x21, VU `k4`, `clk_rv` at half the core clock.
The unit tests `sim_vu` and `sim_at` hold numbers the RTL unit produces at both lane configurations; `tb/vu/test_vu_timing.py` times
op shapes and whole op streams on the RTL unit and compares them with the model (opt-in), and `sw/tools/trw_vs_sim.py` compares a card
run with a simulator run op by op ([prof-dbg.md](../prof-dbg.md) §3). The model charges every op as if it waited for the one before; the RTL
overlaps independent ops (VU §7.8), so a stream of small independent ops is overcharged.

**`tsr_ref_math.c`.** Area VU fixes each algorithm: range reduction, polynomial or table, the
coefficients as f32 bit patterns, and the evaluation order. SW implements the file under these
rules:

1. Every f32 operation is a call to `tsr_fadd`/`tsr_fsub`/`tsr_fmul`/`tsr_fdiv` (DAZ, FTZ,
   canonical NaN, `docs/spec.md` §21, FTZ decision point). Everything else is integer bit manipulation. No libm, no
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
4. Each function's verification is in §10.1 (SW-23).

### 4.6 Race detector and the launch-stateless rule (SW-3, SW-4)

- **Happens-before:** vector clocks over the agents {each unit, the RISC-V, the host}. The edges
  are those of `docs/ctrl-riscv.md` §9.2:
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
- **Launch-stateless rule (SW-4):** a launch's observable results depend only on (image,
  descriptor args, DRAM contents). The detector reports `SPM_STALE` when an op or the RISC-V reads
  an SPM line that nothing wrote since the runtime popped the descriptor. Dependence on DTCM state left by an
  earlier launch has no mechanical check. It is caught instead by the trace round trip (§10.1):
  every captured launch trace is replayed on a freshly booted simulator and must match.
- `TSR_SIM_RACE=off|report|fatal`. ctest uses `fatal`. Model runs default to `off` (the 2–3×
  slowdown is an *estimate*); the nightly model run uses `report`.

### 4.7 Determinism (what is bit-exact)

The definition follows `docs/ctrl-riscv.md` §9.5 and `docs/spec.md` §17. A program (image + descriptor args + DRAM inputs) is **valid** if it is race-free and the
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

Firmware rule (SW-30): no value read from a timing-dependent register (`DONE_SEQ`, credits,
`QSTATUS`, `MTIME`, `LQ_AVAIL`, `LCQ_FREE`, `LOG_FREE`) may influence pushed words, deps, SPM
stores or records. Such values may only end a wait loop. `ISSUE_PIDX` and `ISSUED_SEQ` are not
timing-dependent (only the RISC-V's own pushes move them). The error records (`ERR_CODE`,
`ERR_SEQ`, `ERR_INFO`, `ERR_W0`, `ERR_PIDX`) are exact for every entry covered by a done ticket
(CC I2), so after a join they may decide RET and records (`tsr::first_error`, `error`,
`err_units`, the runtime's R7); `QSTATUS`'s error bits are a separately synchronised level that
I2 does not order with `DONE_SEQ`, so they may only end waits or make the STOP_ON_ERROR decision
(§7.3). One credit value is exact: under QHOLD, after `Q_DEPTH[u]` pushes to a unit whose queue was
empty when the hold took effect, none of which can complete without running (SYNC: NOPs waiting
for a held op), credit(u) is 0 (CC §7.3: `ISSUED_SEQ` moves only with the RISC-V's pushes, and
S[u] cannot move while every head is held); `selftest` id 10 relies on it.
This holds by construction for every kernel in §7.

At the llama.cpp level, the same image and the same inputs give bit-identical logits on `sim`,
`mock`, `vkernel` and `xrt`. That is the M6 and M7 end-to-end check.

### 4.8 C API and the register table

The register and encoding table `sw/tables/tsr_tables.py` (plain Python data, no dependencies) is
the single source (EX-30). One generator (`sw/tables/gen_tables.py`, P1.1) produces from it the
RTL package (`rtl/core/tsr_tables_pkg.sv`), the C header (`sw/sim/include/tsr_tables.h`, included
by `tsr_isa.h`), the firmware headers (`sw/ctrl/include/tsr/{mmio,vu,at}.hpp`) and the Python
module for cocotb (`tb/common/tables.py`), with `--check` in lint. EX, CC, DB, VU and AT own the
table rows; SW owns the generator.

```c
typedef struct tsr_sim_opts {
    const char * dram_dir;   // file-backed DRAM
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

// Launch capture and race reports.
void       tsr_sim_capture_launch(tsr_sim * sim, int on);    // capture each descriptor .. answer record
int        tsr_sim_capture_take(tsr_sim * sim, tsr_trace2 * out);   // 0 = one launch taken
size_t     tsr_sim_race_reports(const tsr_sim * sim, const tsr_race_report ** out);

// RISC-V debug view
uint32_t   tsr_sim_rv_reg(const tsr_sim * sim, unsigned i);  // x0..x31
uint32_t   tsr_sim_rv_pc(const tsr_sim * sim);
uint64_t   tsr_sim_rv_instret(const tsr_sim * sim);
```

`tsr_config` includes `itcm_bytes`, `dtcm_bytes`, `ctl_id` and the queue depths.

### 4.9 Environment (simulator and `sim` transport)

| Variable | Values | Default |
|---|---|---|
| `TSR_SIM_SCHED` | `fixed`, `random:<seed>[,<min>,<max>]` | `fixed` |
| `TSR_SIM_RV_RATIO` | F_RV / F_CORE, 0.05–4 | 0.603 (166.7 / 276.3) |
| `TSR_SIM_RV_CORE` | `maxperf`, `small`: the Ibex cycle table (§4.2, CC §3.2; timing only) | CC's configuration (`maxperf`) |
| `TSR_SIM_VU_TIMING` | the VU's lanes in the cycle model: `u200` (16 / 16 / 8, no lane pipe), `k4` (8 / 8 / 4, two lane-pipe stages) or `F,E,B,P` (§4.5, SW-80; timing only; CFG3 reports the EXP and back lanes) | `u200` |
| `TSR_SIM_CARD` | `k4full`: the card image of the fourth link in one variable: `T_MAX` 4, `CAPS` 0x21 (no local intermediates), `TSR_SIM_VU_TIMING=k4`, `TSR_SIM_RV_RATIO` 0.5; the single variables still override (`sim` transport) | off |
| `TSR_SIM_RACE` | `off`, `report`, `fatal` | `off` (ctest: `fatal`) |
| `TSR_SIM_LOG` | `stderr` or a file, optional prefixes `rv,` (RV retire lines) and `w,` (the eight words of every push): push / start / done events | off |
| `TSR_SIM_THREADS`, `TSR_SIM_DRAM_DIR`, `TSR_SIM_TMAX`, `TSR_SIM_CH_GIB` | as today | as today |
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

`perf` is informational (`docs/spec.md` §21, Performance counters).

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

- DRAM outside the init regions is zero (`docs/spec.md` §21, Trace replay rules). The replayer loads the
  init regions.
- **Kind `launch`:** boot the image (HELLO), push the recorded descriptor, wait for its DONE
  record.
- **Kind `ops`:** write `push.bin` at `ops.load` and launch the standard `replay` kernel (§7.7)
  with (handle, count, flags 0; flags SERIAL for `traces/model-v1conv`), plus BUF with a chunk
  buffer outside every SPM operand of the trace's ops. On the simulator this is
  `tsr_trace2_replay_opts.replay_image`; `tsr_trace2_replay_sim`'s default is the tile-only route
  through the issue port.
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
  RISC-V (`docs/ctrl-riscv.md` §9.4). It separates core concurrency bugs from RISC-V
  integration bugs.

### 5.5 Capture

`TSR_TRACE=<dir>` dumps one `launch` trace per launch (descriptor to answer record). The capture
follows these rules:

- init = DRAM read before it was written, copied at first read;
- expect = DRAM written, taken at the answer record;
- the image is written once per id into `<dir>/images/`;
- the launch's own LAUNCH_BEGIN / LAUNCH_END markers (DB §8.4: [27:16] its kernel index, [15:0]
  its SEQ) are stored with the SEQ field 0, launch-relative like the tickets (W2a). A launch
  captured at SEQ s therefore replays on a freshly booted simulator, where it runs as SEQ 0
  (§10.1); every other marker value is stored as pushed.

`tsr-trace-gen` writes `ops` traces directly (random tile programs with dependencies, concurrency
and error injection). It runs each on the simulator to fill `done`, `expect` and `tickets`.

### 5.6 Trace sets (SW-6)

| Set | Count | Content |
|---|---|---|
| `traces/m2` | 172 | `ops` traces from `tsr-trace-gen` (`regen.sh`, fixed seeds and configurations) with absolute immediate addresses; endings include ILLEGAL_OP, BAD_DEP, DEP_FAILED chains, ABORTED after an error state and QABORT / QCLEAR |
| `traces/gen` | 32 | as m2 |
| `traces/m3` | 284 | MVM boundary cases and the 23 MVM error kinds |
| `traces/model-v1conv` | 290 | the recorded model traces (`traces/model`, on disk, not in git), converted with `tsr-trace-conv`. Each instruction becomes a push that depends on the previous push (a serial chain, replayed with the `replay` flag SERIAL), and init and expect are kept. This checks the units' results against the recorded ones |
| `traces/vu`, `traces/at` | 240, 160 | generator `--ops vu|at` |
| `traces/ctrl` | 148 | concurrency, credits, errors, holds |
| `traces/model-launch` | — | 0.5B Q8_0 and Q4_K_M "Hi", one launch per split in M6, then one launch per token in M7 |

`tsr-trace-conv` converts only traces without register-relative operands; the m2, gen and m3 sets
are generated, not converted.

The generated sets are not in git (W1b): each `regen.sh` reproduces its set byte-identically
from fixed seeds, and `traces/MANIFEST.sha256` holds a digest per set. A frozen reference set goes
into git once the ISA is frozen for the RTL.

Every generated trace obeys the launch-stateless rule (§4.6, SW-4) as one launch: each SPM line in
the read extent of an op that runs (its checker extents, so a strided operand includes its gap lines)
is written by an earlier op of the same trace. `tsr-trace-gen` checks this in its verification. Each
`regen.sh` (its whole set) and `tools_trace_gen_replay` (its small per-mode sets, plus a
`--mutant-stale` trace that both checks must flag) also replay through the `replay` kernel (§7.7)
under the fixed and two random schedules with `TSR_SIM_RACE=fatal`, where any other read is an
`SPM_STALE` abort. Until 2026-10-30, 41 of the 488 traces in m2, gen and m3 (m2 13, gen 5, m3 23; all
T_MAX > 1) had a strided QACT source loaded per token whose stride gaps nothing wrote. The generator
now also loads the gaps, with seeds and configurations unchanged. Those 41 traces changed and every
other trace stayed byte-identical; the set sizes (172, 32, 284; vu 240, at 160, ctrl 148) did not
change. traces/model-v1conv is converted, not generated, and is not affected.

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
- **Hot path** (SW-34): `kernels/graph.cpp` and every `kernels/k_*.cpp` (the executor and the
  kind implementations, which run once per op record) are compiled with `-O2` after those flags
  (`TSR_FW_HOT_FLAGS` / `TSR_FW_HOT_GLOBS` in `sw/ctrl/CMakeLists.txt`, recorded as
  `firmware.cxxflags_hot` in `docs/design/toolchain.lock`); the runtime, `selftest`, `replay` and libc
  stay `-Os`. *Measured* (simulator, Ibex maxperf, 166.7 MHz, fixed schedule, `fw_m6_perf` /
  `fw_exec_perf` on the final tree of the executor-optimisation workflow): per 0.5B MUL_MAT op
  record 442 → 437 RISC-V cycles, JOIN record 147 → 140, NOP record 92 → 85, one 0.5B decode
  layer 51.4 → 39.4 µs of RISC-V time (the launch's fixed part shrinks by ≈ 2,000 cycles), for
  +2.2 KiB of ITCM (30.6 → 32.8 KiB of the 48 KiB budget, §7.8). The whole image at `-O2` gains
  nothing more per record (437) for +5.8 KiB (only the launch entry, 1.68 → 1.54 µs); `-O3` or
  `-O2 -fpeel-loops` on the hot files give 440–442 for 1.5–2 KiB more (on the W2a executor
  `-fpeel-loops` was worth −12 % and `-O2` −9 %, gains the optimised executor's hand-written code
  took over); `-fno-tree-loop-distribute-patterns` changes nothing (no loop becomes a `memset` /
  `memcpy` call), `-falign-jumps=4 -falign-loops=4` ≤ 1 %, `-mtune=sifive-3-series` nothing. At
  `-Os` GCC divides by a constant with `divu` (38 cycles) where `-O2` multiplies: the W2a executor
  did 4.7 divisions per record at `-Os` and 2.3 at `-O2`.
- **libgcc:** taken from `-print-libgcc-file-name` for the same `-march`. If that multilib is
  absent, the `rv32im_zicsr/ilp32` one is used (*confirm* the xPack multilib set). Code without
  C instructions runs on an RV32IMC core.
- **Firmware float:** none in M6/M7 (SW-8). Every f32 constant (eps, 1/n, scales, the RoPE
  `theta_scale` and freq-factor tables) is computed by the host and passed as bits. If firmware
  ever needs spec fp math, it calls the `tsr_ref` helpers (`docs/ctrl-riscv.md` §2.4).
- **CMake:** the firmware is its own CMake project (`ExternalProject_Add(tsr_fw …)`), because one
  CMake project cannot have two C++ compilers. It produces `tsr-fw.elf` (with DWARF) and
  `tsr-fw.map`. Its path is compiled into the host libraries as the default image (`TSR_IMAGE`
  overrides, EX §10.2). The RISC-V toolchain is a **required** dependency of the build; configure
  fails with the install hint of §9.6.

### 6.2 Image format

EX §3.1–§3.2 are normative. Linker script memory map (CC §4.1): ITCM at 0x0000_0000, DTCM at
0x0010_0000, sizes `ITCM_BYTES` / `DTCM_BYTES` from `RV_CFG` (U200: 128 KiB / 64 KiB).

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
| `.tsr_logfmt` | VMA 0xF000_0000 | **no** (non-allocated `INFO` section with contents, not in a `PT_LOAD`; GNU ld's `NOLOAD` would drop the strings) | log format strings (§6.6) |
| `.note.tsr` | — | no | ELF note "TSR", type 1 (below) |

Kernel table entry (36 bytes; W2a: the fields below, `static_assert`ed in `tsr/rt.hpp`):

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
| 0 | `sdk_abi` | the Basis ABI (1), `TSR_SDK_ABI` of `tsr/version.h` |
| 1 | `cap_min` | the Tesseract Core capability the image needs, `VERSION[15:0]` encoding (0x0100 = 1.0) |
| 2, 3 | `itcm_min`, `dtcm_min` | bytes (W2a: `link.ld` fills them at link time; the host compares them with `RV_CFG` KiB × 1024) |
| 4 | `caps_req` | `CAPS` bits the image needs |
| 5 | `n_kernels` | entries of `.tsr_kernels` |
| 6 | `basis_version` | the Tesseract Basis version, major << 16 \| minor << 8 \| patch (0x0000_0100 = 0.1.0); 0 = built before versioning (refused) |
| 7 | reserved | 0 |

Words 0, 1 and 6 come from the link's `--defsym` (`sw/ctrl/cmake/tsr_version.cmake`: `tsr/version.h`
and `VERSION_VALUE`). Kind table entry (4 bytes; `tsr/rt.hpp`, `TSR_KIND(NAME)` beside each kind's
handler): `{u8 kind, u8 rev = opl::kind_rev(kind), u16 0}`, unique kinds in any order; a kind declared
twice in one image is a link error.

Kernel order (W2a): the table is sorted by (rank, name), rank 0 `graph`, 1 `selftest`, 2 `replay`
(EX §4.6's base kernels at indices 0–2), 5 for every other kernel; `TSR_KERNEL` places each entry in
section `.tsr_kernels.<rank>.<name>` and `link.ld` sorts by name. Tools look kernels up by name.

**Image id (SW-9, EX §3.2).** `tsr-elf` computes it after linking: SHA-256 over every `PT_LOAD`
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

The runtime then follows EX §4.7, with R4's validation before R3's coverage check: a descriptor
that is both invalid and covered by a watermark answers BAD_DESC, by EX §4.4's precedence
(BAD_DESC above ABORTED):

- **R1 Boot.** Push `qclear()` and join it; `RT_STATE` := IDLE; post HELLO as an event (r2 / r3 =
  image id, r4 = [15:0] Basis ABI | [31:16] kind table signature, r5 = kernel count, r6 = Tesseract
  Basis version, r7 = boot core cycles; §6.9). Descriptors pushed before HELLO stay queued.
- **R2 Take.** WFI on `irq_fast[2]` (`LQ_AVAIL` ≠ 0 ∧ ¬ACTIVE); copy `LQ_HEAD_W0`–`7` and
  `LQ_HEAD_SEQ` into DTCM; write `LQ_POP`.
- **R3 Cancelled or stopped before start.** Covered by `LQ_CANCEL` or `LQ_STOP` (EX §4.9): answer
  DONE / ABORTED without running.
- **R4 Validate.** Unknown TYPE, reserved bits set, KERNEL beyond the table, or a bad
  argument-block handle → DONE / BAD_DESC (args beyond the kernel's `nargs` are ignored). PING →
  PONG.
- **R5 Run.** `RT_STATE` := KERNEL | index; PROF → `REC_CFG.ALL` and NODE markers; stage the
  argument block (LD into the top 16 SPM lines, one join); `launch_pidx` = `ISSUE_PIDX`, `base_u` =
  `ISSUED_SEQ[u]`; DB's launch hook `tsr_rt_launch_hook(kernel, seq, args)` (DB §3.8; a0 = kernel
  index, a1 = SEQ, a2 = the `tsr_args` pointer at `__tsr_launch_halt`); TRACE_MARK LAUNCH_BEGIN; call
  the kernel; TRACE_MARK LAUNCH_END (DB §8.4).
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
  `docs/ctrl-riscv.md` §8.2).
- A kernel returns RET ≥ 0 for success (0 normally) or a negative code (KERNEL_ERR), and it must
  leave every op it pushed joinable (the runtime joins all, R6).
- Kernels never pop, never post answers, may post events (types 0xC0–0xFF), and never issue from
  interrupt handlers (`issue()` is not reentrant, `docs/ctrl-riscv.md` §3.2).
- Long kernels poll `tsr::stop_requested()` at safe points (EX §4.9).
- Kernels obey the launch-stateless rule (§4.6) and the timing rule (§4.7).

### 6.6 Traps, panic and logging (SW-11)

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

The base is `docs/ctrl-riscv.md` §8.2, with these additions:

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

`plan.hpp` is freestanding C++ (no STL, no heap). The host backend compiles it too (SW-20). It
holds the MUL_MAT tiling (`tsr_mm_plan_tiling` and `tsr_mm_lines_per_token`), the T-tile
choice for an SPM budget, `QA_LINES`, `ROW_BEATS`, and the per-kind scratch requirements of §7.5.
The host therefore rejects in `supports_op` exactly what the firmware cannot plan.

### 6.8 Host build of the kernels (SW-25)

The kernel sources also compile for the host against `backend_host.hpp`:

- `issue()` calls `tsr_sim_push`;
- `done()` and `join()` advance the simulator;
- `spm_ptr` maps to `tsr_sim_spm`.

The host build runs the same kernels in milliseconds under ASan and UBSan. It is a convenience. The
reference is the RV binary on `tsr_rv`, and ctest runs both builds (§10.2).

### 6.9 Versions (SW-54, normative)

Four public numbers, in CUDA's terms: the **Tesseract Core capability** (compute capability), the
**Tesseract Basis version** (toolkit version), the **Basis ABI** (driver ABI) and each kind's
**numerics revision**.

| Number | Value | Source | Carried in |
|---|---|---|---|
| Tesseract Core capability MAJOR.MINOR | 1.0 | `sw/tables/tsr_tables.py` `ISA_MAJOR` / `ISA_MINOR` | `VERSION` (EX §7.5), `.note.tsr` `cap_min` (the image's need) |
| Tesseract Basis MAJOR.MINOR.PATCH (SemVer) | 0.1.0 | `sw/ctrl/include/tsr/version.h` | `.note.tsr` word 6 and HELLO r6, both from the link's `__tsr_basis_version` (major << 16 \| minor << 8 \| patch); `tsr_fw_image(... BASIS_VERSION <code>)` links a test image as another version |
| Basis ABI | 1 | `version.h` `TSR_SDK_ABI` (the only source) | `.note.tsr` word 0, HELLO r4[15:0] |
| Kind revision (u8, 0 = absent) | 1 for the kinds present in the image (§7.2) | `tsr/opl.hpp` `kind_rev` | `.tsr_kinds`; HELLO r4[31:16] = the kind table signature |

- **Capability.** Bumped by the hardware owner (EX §7.5): major for an incompatible hardware-contract
  change (any tile op's result bits included), minor for an addition present in every configuration
  of that minor; an optional feature is a `CAPS` bit.
- **Basis version.** Bumped at release by the coordinator, with `CHANGELOG.md` (the section
  heading's "(unreleased)" becomes the date) and an annotated tag `basis-vX.Y.Z`. Major (from 1.0.0): an ABI break. Minor:
  new kinds, kernels, kind variants or flags, a planner accepting more shapes, any revision bump, new
  host API or tools. Patch: fixes and performance work that leave every output bit of every kind,
  every interface and every planner's acceptance unchanged. 0.x until the M7 card gate, then 1.0.0;
  in 0.x an incompatible change bumps the minor and the ABI.
- **Basis ABI.** Covers `tsr_args`, `TSR_KERNEL` and the kernel table entry, the `.note.tsr` layout
  and the `.tsr_kinds` entry, the launch protocol as the runtime implements it (descriptor, record
  and HELLO layouts, R1–R10), the op list format (header, record layout, every kind's argument words
  and their meaning, removing a kind) and a planner accepting less (SW-20). Bumped by the breaking
  change itself. ABI 1 is defined as of Basis 0.1.0.
- **Kind revision.** Bumped in the change that can alter any output bit of the kind for the same
  record words, input bytes and configuration (`CFG0`–`CFG3`, `CAPS`) on the same capability; not
  for push order, scheduling or scratch use, nor for accepting new inputs (a Basis minor). A new kind
  starts at 1; revisions never decrease and are never reused. Procedure: change `opl::kind_rev`, run
  `test_fw_kind_revs tsr-fw.elf sw/ctrl/tests/kind_revs.txt --update`, add a line to the
  CHANGELOG's unreleased section. `fw_kind_revs` compares each kind's outputs on a fixed corpus with the recorded
  digest at that revision, so (capability, configuration, kind, revision) identifies a result. The
  corpus takes the decode fast entries production decode runs (a retire hook counts the entries into
  the image's general-path and fast-path functions, and each launch states which it must take), so a
  numerics change confined to a fast entry changes a digest too. `--update` applies the table's rules
  first: it refuses a digest that moved at an unchanged revision, a bump without a change and a lowered
  revision, so it records only what a revision bump announced.
- **Kind table.** `TSR_KIND(NAME)` beside each handler (NOP and JOIN in `graph.cpp`) emits
  `{kind, rev, 0}` into `.tsr_kinds`; the signature is FNV-1a 32 over the 64 revision bytes (0 for an
  absent kind), folded to 16 bits (`opl::kinds_sig`: no kinds 0xB51B, NOP + JOIN + MUL_MAT 0x512E,
  the 14 kinds 0xFA6F, the 18 kinds with the MoE kinds 0xC341). The runtime computes it at R1; the host computes it from the ELF.

Who refuses what:

| Check | Where | Rule |
|---|---|---|
| device identity | `tsr_transport_open`, reads only, in this order | `ADAPTER_VERSION` major 1 and `NUM_CH` (kinds with an adapter); `ID` = "TSR1"; `VERSION` major = 1 (any minor); `CTL_ID` ≠ 0 right after it |
| image | `tsr_image_open` | `.note.tsr` present; `sdk_abi` = `TSR_SDK_ABI`; `basis_version` ≠ 0; `.tsr_kinds` well formed |
| device vs image | `tsr_transport_boot`, before any load | same capability major, device minor ≥ `cap_min`'s; `CAPS` ⊇ `caps_req`; TCM sizes; the ABI |
| HELLO | every cold and warm boot | image id; r4[15:0] = the note's ABI; r4[31:16] = the signature of `.tsr_kinds`; r5 kernel count; r6 = the note's Basis version; no other record first |
| backend vs image | ggml-tesseract and `sw/rt` at open | the image's Basis (major, minor) ≥ the backend's (patch ignored; a newer image is fine). Not a transport rule, so captured traces of older images stay replayable |
| launch | supports_op; the builder | only kinds the booted image declares; the executor still answers BAD_RECORD for a kind it lacks and BAD_VERSION for an op list format it does not know |

Internal identifiers, never shown as product versions: `CTL_ID[7:0]` = 1 (control-core revision;
0 = no control core), `ADAPTER_VERSION` major 1, `DBG_ID[7:0]` = 1 (DB §10.1), the kind table
signature, the image id. File-format versions are unchanged (OPL header 1, trace meta 2, `.klc` 2,
numcheck meta 2, DBG_INFO 1, CRASH 1, MSUM 1). Display (`tsr_versions_str`): "Tesseract Core 1.0 (VU
ATT), Tesseract Basis 0.1.0 (ABI 1, 14 kinds)".

---

## 7. Kernels and the graph executor

### 7.1 Kernel set

| Kernel | Milestone | Args | Purpose |
|---|---|---|---|
| `graph` | M6 (MUL_MAT records only), M7 (all kinds) | a0 OPL handle, a1 OPL bytes, a2 flags (bit0 VERIFY_CRC) | the op-list executor (§7.3) |
| `selftest` | M6 | a0 test id, a1 data handle, a2 data area lines, a3 seed, a4 flags (parameters in line 0 of the data area; layouts in `sw/ctrl/ref/selftest_abi.hpp`) | card and sim self-test programs (§7.6) |
| `replay` | M6 | a0 `push.bin` handle, a1 count, a2 flags (bit0 SERIAL: every op record depends on the previous op record instead of its recorded deps; bit1 BUF: a3 / a4 valid), a3 SPM line of the chunk buffers, a4 records per chunk (a multiple of 4, ≤ 1,364) | tile-only trace replay, SPM dumps (§7.7) |

These are EX's base kernels (EX §4.6). PING is part of the runtime (descriptor TYPE 0x02), not a
kernel. The MoE kinds MOE_ROUTE and MUL_MAT_ID are op list kinds of `graph` (§7.9), not kernels. Decode steps, the
resident loop and JIT entry points come with M8 and M9.

### 7.2 Op list format (OPL, SW-12)

An op list is one 64-byte-aligned blob in device DRAM, normally in the OPL region (ch0 0–64 MiB,
§9.3). It is little-endian.

**Header (64 bytes):**

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

**Preload entry (16 bytes; SW-15):**

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

**Kinds and argument words.** M6 implements 0x01 and 0x10; M7 the rest. The Gated DeltaNet kinds (0x2D to 0x2F) are numbered from the top of the free range downwards. Rev is the kind's numerics
revision at Tesseract Basis 0.1.0 (`opl::kind_rev`, §6.9). Shapes are flattened to
rows: `rows` = ne1·ne2·ne3, and every row of a tensor starts on a line (ensured by `supports_op`).

| Kind | Name | Rev | Args (word index: meaning) | ggml source |
|---|---|---|---|---|
| 0x00 | NOP | 1 | — (no tile ops; its ticket is "none") | placeholder |
| 0x01 | JOIN | 1 | — (SYNC NOP over dep0, dep1) | builder |
| 0x10 | MUL_MAT | 1 | 0 x, 1 x stride, 2 y, 3 y stride, 4 W_OFF (beats), 5 fmt (0 Q8_0, 1 Q4_K), 6 N, 7 K, 8 T, 9 addend handle (0xFFFF_FFFF none), 10 addend stride (0 = row 0 broadcast to every token), 11 flags (bit0 ACC_DST: y += W·x) | MUL_MAT; fused MUL_MAT+ADD(bias or residual) |
| 0x11 | ADD | 1 | 0 dst, 1 dst stride, 2 a, 3 a stride, 4 b, 5 b stride, 6 n (ne0), 7 rows, 8 b row period (b row = r mod p, p ≥ 1), 9 flags (bit0 b is one f32 scalar) | ADD |
| 0x12 | MUL | 1 | as ADD | MUL (norm weights, Gemma `out_scale`) |
| 0x13 | SCALE | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 s (f32), 7 b (f32) | SCALE (`x·s`, then `+b` if b ≠ 0) |
| 0x14 | RMS_NORM | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 eps (f32), 7 inv_n = `tsr_fdiv(1, n)` (f32), 8 weight handle (0xFFFF_FFFF none), 9 weight row period | RMS_NORM, fused RMS_NORM+MUL(weight) |
| 0x15 | GLU | 1 | 0 dst, 1 dst stride, 2 gate, 3 gate stride, 4 up, 5 up stride, 6 n, 7 rows, 8 variant (0 SWIGLU, 1 GEGLU16: ggml-CPU-compatible fp16 GELU, VU §6.5; 2 GEGLU: exact tanh form, M8 / PyTorch) | GLU split form |
| 0x16 | ROPE | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 pos preload byte offset (I32 per token), 5 table handle (AUX f32 × n_dims/2: the freq-factor table, or the A table with mode bit1; 0xFFFF_FFFF none), 6 TS = `powf(freq_base, -2.0f/n_dims)` bits (host libm, as ggml-cpu), 7 M = attn_factor (f32), 8 freq_scale (f32; 1.0 = none), 9 head_dim, 10 n_dims, 11 n_heads, 12 n_tokens, 13 mode (bit0 NEOX; bit1 ATAB: YaRN form with a host-folded A table, VU §6.7) | ROPE: the angle recurrence of ggml-cpu (VU §6.7), bit-identical angles for ext_factor = 0 |
| 0x17 | GET_ROWS | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 ids preload byte offset (I32), 5 n_ids, 6 row lines | GET_ROWS on f32 activations (`inp_out_ids`) |
| 0x18 | SET_ROWS_KV | 1 | 0 cache base handle, 1 cache row stride (AT units), 2 src, 3 src stride, 4 idx preload byte offset (I64; the host checks idx < 2^31), 5 n_rows, 6 row elements, 7 which (0 K, 1 V), 8 plane base, 9 E/64 \| LG << 16, 10 plane kind, 11 stream (AT §12) | SET_ROWS into a KV buffer: QACT A32 + KVST per run of consecutive cells (AT §5.1) |
| 0x19 | FLASH_ATTN | 1 | 0 dst, 1 dst token stride, 2 q, 3 q head stride, 4 q token stride, 5 K cache base, 6 V cache base, 7 n_kv, 8 head_dim, 9 n_head, 10 n_head_kv, 11 n_tokens, 12 MSUM preload byte offset, 13 f32-mask handle (with a mask: always the copy for this record's scale, written only when needed, because the cached list outlives per-token mask changes and a rule-3 GENERAL tile reads it also when the MSUM is not GENERAL; 0xFFFF_FFFF only without a mask; W3b, SW-48), 14 scale (f32), 15 softcap (must be 0), 16 K plane base, 17 V plane base, 18 E/64 \| LG << 16, 19 D, 20 G, 21 flags (AT §12; two slots) | FLASH_ATTN_EXT with K/V in KV buffers; the device never reads the F16 mask |
| 0x1A | CPY | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 row lines, 5 rows | CPY/CONT f32 → f32 with contiguous rows |
| 0x1B | UNARY | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows, 6 func (0 TANH, 1 GELU16 (llama.cpp GELU), 2 SILU, 3 EXP, 4 GELU exact tanh form, 5 SIGMOID) | UNARY ops the VU has |
| 0x1C | MUL_MAT_SCALE | 1 | MUL_MAT words 0–11, requiring no addend, addend stride 0 and flags 0; word 12 FP32 scale | single-consumer MUL_MAT → SCALE |
| 0x1D | MUL_MAT_SOFTCAP | 1 | same restricted MUL_MAT words 0–11; words 12/13 exact FP32 pre/post scales | MUL_MAT → SCALE → TANH → SCALE |
| 0x1E | MOE_ROUTE | 1 | 0 logits, 1 logits stride (lines), 2 ids (first line of each row of the ARGSORT tensor), 3 ids stride (lines), 4 weights (packed f32, k per token), 5 n_expert (≤ 512), 6 n_used k (≤ 16), 7 T, 8 flags (0) | SOFT_MAX, ARGSORT + VIEW, GET_ROWS, SUM_ROWS, CLAMP, DIV of `build_moe_ffn` (§7.9, SW-61) |
| 0x1F | MUL_MAT_ID | 1 | 0 dst, 1 dst stride, 2 x, 3 x slot stride (0: one x row per token), 4 x token stride, 5 ids, 6 ids stride (bytes), 7 W_OFF of expert 0 (beats), 8 EXP_STRIDE (beats per channel), 9 fmt, 10 N, 11 K, 12 T, 13 n_used, 14 n_expert, 15 scale table handle (0xFFFF_FFFF none), 16 flags (0) | MUL_MAT_ID; with the per-expert scale chain REPEAT + GET_ROWS + MUL (§7.9, SW-62) |
| 0x20–0x2C | free (the MoE kinds took 0x1E and 0x1F, the lowest free numbers; the Gated DeltaNet kinds took 0x2D–0x2F, numbered downward from the top of the free range) | — | | |
| 0x2D | SOFTPLUS | 1 | 0 dst, 1 dst stride, 2 src, 3 src stride, 4 n, 5 rows | UNARY SOFTPLUS (§7.10, SW-69) |
| 0x2E | SSM_CONV | 1 | 0 dst, 1 dst token stride, 2 x (token rows), 3 x token stride, 4 state in, 5 state out, 6 taps (d_conv rows of C), 7 C, 8 d_conv, 9 T, 10 n_seqs, 11 flags (bit0 SILU) | SSM_CONV with its CONCAT window, state CPY and SILU (§7.10, SW-68, SW-71) |
| 0x2F | GATED_DELTA_NET | 1 | 0 dst, 1 state out, 2 state in, 3 q, 4 q head stride, 5 q token stride, 6 k, 7 k head stride, 8 k token stride, 9 v, 10 v head stride, 11 v token stride, 12 g, 13 g token stride, 14 beta, 15 beta token stride, 16 S_v, 17 H_k, 18 H_v, 19 T, 20 n_seqs, 21 scale (f32) | GATED_DELTA_NET, scalar gate, K = 1 (§7.10, SW-67) |
| 0x30–0x3F | test kinds | — | | |

**Optional local epilogues (November 2026).** The two new kinds above require
both the booted image's kind entry and `CAPS.MVM_LOCAL_SCALE` (bit 13) or
`CAPS.MVM_LOCAL_SOFTCAP` (bit 14). Firmware refuses an unsupported explicit record before that record issues
tiles; it does not roll back earlier records. Their
DDRx handles, full-N weight layout, input/output strides and graph dependencies
retain MUL_MAT semantics. No intermediate DDR tensor is published. An older
image or core uses the original materialized sequence, selected before launch.
Existing kind revisions, Basis ABI 1 and Core capability 1.0 remain unchanged;
the new optional kinds start at revision 1.

The ggml builder fuses only addressable FP32, sole-consumer chains whose
intermediates are not observed, views or separate graph outputs, and whose
memory accesses do not conflict. Safe reuse of absorbed intermediate storage
is allowed. It
retains hoisting/dependency checks and refuses overlapping input/final output.
SCALE's additive constant must be positive-zero bits. SOFTCAP keeps both exact
graph coefficients; the host never recomputes a reciprocal. Its initial host
coverage requires `row_beats >= 32*Ttile`. That conservative service-rate guard
is not a latency proof. Nonqualifying output heads retain the existing sequence, possibly with an
independently qualified local SCALE prefix.

PyTorch has a separate bounded FX path: bias-free, right-weight FP32
mm/matmul/linear followed by a sole positive power-of-two scalar multiply in
[1, 2^127]. It preserves FP16/BF16 casts by leaving those cases on the existing
path. Missing kind/capability, verification mode or multiple K partitions keep
ordinary linear followed by the CPU multiply. No CPU TANH is silently replaced
by the device approximation. This change does not add PyTorch device attention.
The real `torch.compile` path, escaping intermediates and old firmware images
are regression cases, rather than only tests of the runtime API.

MoE is built on the simulator (§7.9; the next paragraph is the locality plan's note that preceded it). The shared dense-window metadata preserves
original global N, per-channel row ownership, expert base/stride and input/output
token origins. Current dense opcodes use expert zero and identity token order;
the helper is not an indexed-expert opcode. MoE still needs expert-bank packing,
MUL_MAT_ID/selection/combine lowering, explicit token-ID gather/scatter and
matching ggml/PyTorch regions. Final DDR materialization provides a compatible
boundary until those consumers are qualified. TOPK hardware remains a separate
optional configuration; locality does not enable it implicitly.

**M7 argument rules (W2b infra, 2026-10-30; `tsr/opl.hpp`).** The words of each kind are a struct in
word order (`opl::ew_args` for ADD / MUL, `scale_args`, `rms_norm_args`, `glu_args`, `rope_args`,
`get_rows_args`, `set_rows_kv_args`, `flash_attn_args`, `cpy_args`, `unary_args`; indices in the
namespaces `opl::ew`, `scl`, `rms`, `glu`, `rope`, `getrows`, `setrows`, `fa`, `cpy`, `unary`), written by
`writer::add_<kind>` and read by `record::args_as`. Structural rules that hold on every device, checked
by `opl::check_args` (the host reader) and by the kinds (BAD_RECORD):

- ADD / MUL: flags bit0 (b is one f32 scalar) only; the b row period is ≥ 1 in every record (1 with a
  scalar b). RMS_NORM: a weight's row period is ≥ 1; without a weight the word is ignored.
- GLU variant ≤ 2, UNARY func ≤ 4, ROPE mode bits 0–1 only, SET_ROWS_KV which ≤ 1; FLASH_ATTN's softcap
  word is 0 (the bits: −0.0 is refused too).
- Every preload reference lies inside the list's preload area (`pre_lines` lines): ROPE 4·n_tokens
  bytes at a line-aligned offset (the VU reads the positions as an SPM operand), GET_ROWS 4·n_ids at a
  multiple of 4, SET_ROWS_KV 8·n_rows at a multiple of 8, FLASH_ATTN the MSUM image of n_tokens rows
  (32 + 16·n_tokens bytes) at a line-aligned offset. The writer's entries (`pre_positions`, `pre_ids`,
  `pre_kv_idx`, `pre_msum`) are 64-aligned, so every alignment holds for a reference to an entry's start.

**M7 word semantics the kernels fixed (W2b, 2026-10-30; the backend's contract in W3b).** Where the
table above leaves a choice, the kernels, their planners and the references implement:

- Every M7 kind: kind flags (word0 [23:16]) 0 (none is defined). Empty ops (n, rows, n_ids or
  n_tokens 0) are BAD_RECORD, except SET_ROWS_KV with n_rows 0, which pushes nothing (ticket none).
- ADD / MUL with FLAG_B_SCALAR: b is an f32 vector of per-row scalars and row r uses word r mod p
  (p = b row period): p = 1 one scalar for every row, p ≥ rows one scalar per row (Gemma's
  `inp_scale_rows` [1, T]: the c-vector form of VU §5.2), 1 < p < rows period blocks; the b stride
  word is unused with the flag. Without the flag a b stride of 0 broadcasts b row 0.
- SCALE: y = FA(FM(x, s), b): VOP.MULC by s12 = s, then VOP.ADDC by s13 = b only when b's bits are not
  0 (−0.0 adds), the bits VOP.MULCADD would give with a b row of copies of b (SW §7.5).
- RMS_NORM: the weight rows are dense, L(n) = ⌈n/16⌉ lines apart (no stride word); the period word
  is ignored without a weight.
- ROPE: dst / src strides are lines between tokens; a token's heads are contiguous (H = n_heads ·
  head_dim / 16 lines; ggml's [head_dim, n_head, T] views, also a fused-qkv view whose token stride is
  the qkv row); n_dims ≤ head_dim (the unrotated tail is copied); mode bit1 (ATAB) needs a table (else
  BAD_RECORD) and makes the TS and freq_scale words unused; a table without ATAB is the freq-factor
  table (ROPE FF); freq_scale 1.0 sets no FS (FM(1.0, φ) = φ for every VU angle).
- GET_ROWS: an id that is negative or puts the row beyond 2^32 beats gets the LD beat 0xFFFF_FFFF, so
  the op fails with DRAM_RANGE (TILE_ERR on the op, never a panic); an id beyond its channel fails the
  same way at the DMA unit. The ids must lie inside the list's preload area (else BAD_RECORD).
- SET_ROWS_KV: word 10 (plane kind) is the KV format and must be 0 (q8_0); words 0, 1 and 11 (cache
  handle, row stride, stream) are informational, the plane base (word 8) is authoritative; row elements
  = E ≤ K_MAX; every index must have hi word 0 and a cell < 16·C·LG, else BAD_RECORD before any push.
  A cell written twice keeps the later row (DMA queue order).
- FLASH_ATTN: q head stride = D/16 (heads contiguous per token); dst token stride ≥ n_head·D/16 when T
  > 1; n_kv ≤ 2^20; E/64 in 1–511 (the 9-bit field); head_dim, n_head and n_head_kv agree with D, G and
  E; a GENERAL tile needs word 13 (the f32 mask copy) also when the MSUM is not GENERAL (AT §7.3); the
  flags word (word 21): bit0 FA_NEW, bits [7:1] 0, bits [15:8] the preload line of the I64 new-cell
  indices, bits [31:16] their count (0 without FA_NEW): the backend points it at the layer's
  SET_ROWS_KV index entry (AT §6.6 rule 7, §12).

Shapes and device limits are the kinds' planners' (`tsr/plan.hpp` "M7 planning": `tsr::plan_<kind>`,
shared by the firmware and the backend's supports_op, SW-20). `plan::dev_cfg` carries C (CFG0
NUM_CH), PCAP (CFG3), VL and CAPS besides the M6 fields; 0 means unknown, and a kind that needs the
value plans nothing (BAD_RECORD), so a host that fills only the M6 fields keeps rejecting M7 kinds.

### 7.3 Executor (`graph` kernel)

**SPM layout** (U200 defaults; `L` = `SPM_BYTES`/64 = 65,536 lines):

| Lines | Use |
|---|---|
| [L−16, L) | runtime argument range |
| [L−16−4096, L−16) | OPL area (256 KiB); a larger op list streams in two 128 KiB halves |
| [.., ..) of `pre_lines` | preload area (header `pre_lines`, ≤ 256 lines) |
| [0, rest) | scratch ring (`spm_ring`), ≈ 3.7 MiB |

**DTCM tables:** `ticket[4096]` (the op's done ticket) and `pidx[4096]` (the op's first push
index), 32 KiB in total. The ticket table has one sentinel word in front (`TICKET_NONE`, the
"no dep" entry: a dep is looked up as entry dep + 1 with 0xFFFF wrapping to 0); the symbol
`__tsr_exec_ticket` stays the 4,096-entry table. The scratch ring's record table holds 128 blocks
(2 KiB).

**Algorithm:**

1. LD the header and the op list (or the first half) into the OPL area and join it; LD the preload
   entries (their table is inside the list) and join once more, so a list with preload entries
   costs two joins at launch start and none inside the op stream. Validate magic, version,
   `n_ops` ≤ 4096 and (if VERIFY_CRC) the CRC. A failure returns 0x8000_000x (below).
2. For op i = 0 … n_ops−1:
   1. The op's control-window accesses first, because each one invalidates the SPM line buffer
      (CC §10.5): with MARK, `TRACE_MARK` NODE_BEGIN | i; `pidx[i]` = `pidx_now()`
      (`ISSUE_PIDX`: exact and timing-independent); `RT_INFO` = i. Then read the record through
      the line-buffered SPM alias (one line fill per 64-byte slot), in place: records are not
      copied, except one whose lines wrap the streamed OPL area (copied to DTCM). Check it
      (BAD_RECORD, BAD_DEP). A streamed list then pushes the next chunk's LD when the next record
      starts in a new chunk (before this record's pushes, as at W2a; the record is copied to DTCM
      first when it lies in the half that LD overwrites) and reads `pidx[i]` again.
   2. `d` = after(ticket[dep0], ticket[dep1]).
   3. `pidx[i]` (read in step 1, or after the chunk LD; before any push of op i) is the push index
      of the op's first push.
   4. Call the kind handler with (args, d, ring); args point at the record's argument words in
      place, and a handler reads the words it needs before its first push (a push invalidates the
      line buffer). It allocates scratch (its writes also wait for the ring's release deps), pushes
      its tile ops, and returns one ticket T_i whose completion implies every tile op of op i is
      done. The last tile op depends on the others, or a `join2` provides that (I1,
      `docs/ctrl-riscv.md` §4.4). A MUL_MAT record first goes to the decode fast path, a short M7
      record to its kind's fast entry (both below).
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

**Hot path (SW-35, executor optimisation, 2026-10-30).** The per-record cost of §7.8 comes from:

- *The loop* (`graph.cpp` `op_loop`, instantiated for in-place and streamed lists): the kind table
  in .rodata holds per kind the handler and word0's expected n_args and reserved bits plus the
  minimum slot count, so the record checks are two compares; the test hook (exec.hpp) is consulted
  only for kinds without a built-in handler. A streamed list pushes its next chunk's LD where W2a
  did: after the current record is checked, before its handler (the record is copied to DTCM first
  when it starts in the half that LD overwrites, at most once per 2,048 lines; `pidx[i]` is read
  again after the LD). Only the lines that hold the record's words are made readable (`ensure`),
  not its padding slots, as at W2a. (fix after the review: the first version pushed the chunk LD
  after the handler, one record later than W2a, which moved later DMA deps by one relative ticket
  and let one more record complete before a failing chunk load stopped the launch.) With MARK
  (a PROF launch) the record's NODE_BEGIN marker comes before the record is read, so it precedes
  that chunk LD (W2a: after it), and a malformed record has its marker (W2a: none): of the
  executor's own pushes, only these markers differ from W2a, in PROF launches only (a kind that
  keeps DMA tickets of the ring's deps may see the 128-block table's deps, `spm_ring` below).
- *MUL_MAT's decode fast path* (`sw/ctrl/kernels/k_mulmat.hpp`, `mulmat_t1_fast`): for T = 1 a leaf
  function reads its argument words in place, plans with `plan.hpp`'s `mm_plan` (on `mm_limits`
  computed once per launch; shifts only, no divide), allocates with `spm_ring::alloc_dma_first`
  (the ring's release tickets of the blocks it reuses are earlier ops' STs, which the DMA queue
  order implies for the first LD: no deps and no software join), takes one `ISSUE_CREDIT` read for
  the op's 4–5 pushes when the per-unit fields and the issue-FIFO bound cover them (kernel.hpp
  `credits_cover`; else `issue()`'s read per push), and pushes exactly the ops, words and deps of
  `kind_mulmat`. Every other case (T ≠ 1, a record `kind_mulmat` refuses, a ring wrap, reused
  blocks released by another unit) returns before changing anything, and `kind_mulmat` runs it.
  The general path (T > 1) is unchanged in what it pushes and is compiled for size.
- *`spm_ring`* (§6.7): O(1) amortized (a power-of-two table of 4-word records, the per-unit fold
  on a unit bit mask, the common case inline); the record table grew from 64 to 128 blocks, so the
  executor can run further ahead before a full table makes `alloc` join its oldest block. The
  deps rule is unchanged (per unit the later ticket, at most two units, else the unit whose last
  overlapped block is oldest is joined). With the larger table a reused block's release ticket is
  folded into the new block's deps where the 64-entry table had already joined it in software; a
  MUL_MAT drops these deps (they are STs, DMA), so its pushes are unchanged.
- *Diagnostics:* `exec_state` (opl.hpp, by symbol `__tsr_exec_state`) counts the MUL_MAT records
  that left the fast path and the ring's software joins and full-algorithm allocations, off the
  hot path; a launch rejected before step 2 (header, CRC, preload, layout) reports 0 for the ring's
  counters, not the previous launch's (nothing carries over, SW-4).

MUL_MAT's tile-op order, operand words, deps and scratch placement are W2a's: all 26 launch
traces of `fw_m6_e2e` (two T = 3 lists, a T = 1 decode list, selftest, replay) are byte-identical
(`ops/push.bin`, init and expect regions) between the W2a image (92f0b17) and this one, and so are
all 563 launch traces of `fw_exec_rv --full` (8 schedules: streamed lists, the failing chunk load
and the 4,096-op list included). Tests pin it: `fw_m6_e2e` (the graph launches) and `fw_exec_rv`
(the streamed lists) compare digests of their captured launch traces (push records, DONE, expect
regions; `tests/fw_trace_digest.h`) with the W2a image's, `fw_kernels_host` runs T = 1 records
with deps on every unit through `mulmat_t1_fast` and through the general path on twin simulators
and compares every push, `fw_hdr_host` compares the ring's blocks, deps and joined tickets
(`spm_ring::join_hook`, host builds only) with W2a's algorithm at 128 blocks, and `fw_exec_rv`,
`fw_m6_e2e` and `fw_m6_perf` assert PERF PUSH_STALL = 0 (no push waited for a credit, CC §7.3).
A deliberate change of these pushes is a decision: record it here and update the digests. Planning
stays in the firmware (SW-20 unchanged, the op list format unchanged): the T = 1 plan costs ≈ 35
instructions of a ≈ 280-instruction record, and plan fields carried in the record would still
have to be checked against QA_LINES and ⌈N/16⌉ for the block's safety, which is the same
arithmetic.

**M7 kinds in the executor (W2b infra, integration and fix, 2026-10-30 / 31; SW-36, SW-37,
SW-39).**

- *Kind table.* Kinds 0x11–0x1B have their handlers (`tsr::kind_add` … `kind_unary`, exec.hpp), their
  argument counts and, for the ten short kinds (all but FLASH_ATTN), a fast entry `kind_<name>_fast`
  in the table (16 bytes per kind). `sw/ctrl/kernels/zz_kind_stubs.cpp` defines each handler weakly
  as BAD_RECORD ("not in this image") and each fast entry weakly as "not taken"; a kernel source's
  strong definitions replace them (RISC-V images link every kernel object; the host archive
  `tsr_fw_host` extracts the first member that defines the symbol, and the stub file sorts after every
  kernel source). A wrong n_args is the record check's BAD_RECORD, before any handler (`fw_exec` sc_m7).
- *Dispatch (SW-37, integration).* An M7 record goes to its fast entry first, with the record's
  argument words in place, its kind flags and d = after(ticket[dep0], ticket[dep1]) from the loop's
  registers: a leaf in its common path (no call: nothing of the executor's state is saved around it),
  which takes the record and returns its ticket, or returns KIND_NOT_FAST having pushed and changed
  nothing; the executor then calls the kind's handler out of line (`call_m7_ool`, which re-reads word0
  / word1 so that nothing else stays live across the fast call). The handler begins with the same fast
  entry, so both push the same ops (host tests call the handlers). FLASH_ATTN and the test kinds keep
  the direct handler call. The fast paths' refusal tests carry `[[unlikely]]` hints (GCC otherwise laid
  them out as the fall-through: up to 30 RISC-V cycles per record of taken branches). MUL_MAT keeps
  `mulmat_t1_fast` before the table (unchanged instruction for instruction). `op_loop` has a variant
  for lists without MARK and SERIAL read in place (every production list unless profiled; W2b fix):
  the two per-record flag tests fold away (5–10 RISC-V cycles per record). `fw_m6_perf` gives 389 /
  403 / 377 cycles of work per MUL_MAT-list record (deep queues / U200 depths / no MARK; W2a-opt 387
  / 402 / 383), a JOIN record 135 (140).
- *The one-tile block (SW-36, integration; size W2b fix).* Every single-tile record of a
  DMA-bracketed kind uses one fixed SPM block below the preload area (`plan.hpp` `one_block_lines`:
  1,024 lines on the U200,
  `kind_ctx::one` / `one_lines`) instead of a ring block: ADD, MUL, SCALE, RMS_NORM, GLU, ROPE and UNARY
  (their only tile), GET_ROWS / CPY (every copy tile) and SET_ROWS_KV (every run). A tile is
  DMA-bracketed when its first write to the block is an LD, its last read a DMA op (ST, KVST) and
  every other op on it lies between them by deps; the DMA unit runs its queue one op after another
  (a dep on the op's own unit is implied by queue order, CC §8.3), so the next such tile's first LD starts after this
  tile's last DMA op, which followed every op of this tile: the lines need no dep, no ring record and
  no release, and no concurrency is lost (consecutive DMA-bracketed tiles are serialised by the DMA
  queue whatever their lines). Both the fast and the general path take it for the same records (so
  they still push the same ops, which `fw_kernels_elem_host` / `fw_kernels_normrope_host` compare).
  MUL_MAT (whose blocks the M6 trace digests pin) and FLASH_ATTN keep the ring, which shrinks by the
  block: the planning budget `ring_budget` is 60,144 lines on the U200 (61,168 before W2b). The M6
  digests of `fw_m6_e2e` and `fw_exec_rv` are unchanged (no M6 list wraps the ring). It saved 60–80
  RISC-V cycles per short record (the ring's record table was full of short blocks in steady state,
  so every allocation also joined and dropped its oldest record). The size is a trade (W2b fix): the
  budget sets MUL_MAT's prefill T-tile (two tiles in the budget, §7.4), and the integration's 4,096
  lines made the T-tile of every Qwen2.5 output head 2 instead of 3 (an all-logits prefill, as
  perplexity / KL runs do, streamed the largest weight matrix 256 times per 512 tokens instead of
  171: +50 % of its traffic). 1,024 lines keep 3 for Qwen2.5 0.5B–32B (Qwen2.5-72B's head, lines per
  token 10,160, needs a block ≤ 208 lines: 2) and hold every decode tile of Qwen2.5 0.5B / 32B and
  Gemma 4 12B but GLU's rows from Qwen2.5 1.5B up (1,120–3,456 lines) and Gemma 4 12B's GEGLU16
  (1,920): GLU's fast path takes a ring block for them (`spm_ring::alloc_dma_first`, as
  `mulmat_t1_fast`; 461 RISC-V cycles of work at 12B's n = 15,360, unit-bound, against 1,589 through
  the general path). `plan.hpp` pins the T-tiles of the 0.5B / 32B / 12B shapes by static_assert.
- *kind_ctx* adds `pl` (`plan::plan_limits_of(cfg)`, once per launch), the preload helpers `pre_ok`
  (the same rule as `opl::check_args`), `pre_line` (a line-aligned reference as an SPM operand) and
  `pre_lb` (RISC-V reads through the line-buffered alias; the preload loads were joined before op 0),
  `fail(code)`, the s-register shadow and `one` / `one_lines` (all after the M6 fields, which keep their
  offsets).
- *s-register shadow.* `kind_ctx::sreg_valid` / `sreg_val` record, per launch, the s registers whose
  last writing push was an `SOP.MOVI` and its value; `k_fast.hpp` `sreg_movi` pushes a MOVI only when
  the shadow does not already hold the value (RMS_NORM's 1/n and eps, ROPE's TS / M, SCALE's s / b
  repeat across records). VU ops run in VU queue order, so every later VU op reads that value. Rule: a
  push that writes s[i] otherwise (RED with TO_S, SOP other than STS) clears bit i (`sreg_clobber`) in
  the same record. The shadow starts empty at every launch (SW-4) and depends only on the pushes
  (§4.7). Registers: RMS_NORM s1 (RED TO_S result), s2 = 1/n, s3 = eps; ROPE s4 = M, s5 = TS, s6 =
  freq_scale; SCALE / scalar-b ADD and MUL s12, s13.
- *Shared planning rules* (`plan.hpp`): `plan_rows` (one tile when all rows fit the budget, no divide;
  else tiles of ⌊budget/2/lines-per-row⌋ rows, at least 1, at most 65,535) and `for_period_segments`
  (a row period p > 1 smaller than the op's rows splits the rows where r mod p wraps; p ≤ 1 is b stride 0).
- *Fast-path toolkit* (`sw/ctrl/kernels/k_fast.hpp`): `alloc_fast` (the one-tile block), `first_deps`,
  `push_batch` (one ISSUE_CREDIT read for the record's pushes), `push_s` (below), `vop_fast` /
  `red_fast` / `sop_movi` (VU words without the generated builders' run-time range checks, for planned
  fields; equal words by static_assert), `load_rows` / `store_rows`, and the general paths' out-of-line
  `push_g` / `alloc_g` / `fold_deps` (one copy per image). `k_mulmat.hpp` takes `drop_dma` and
  `NOT_FAST` from it.
- *Staged-word reuse and hot entries (SW-39, W2b fix).* `kf::push_s<SKIP, TICKET>` is the
  no-credit-check push without the `ISSUE_W` stores of the words in SKIP and without the
  `ISSUE_TICKET` read unless TICKET: CC §5.1 keeps the staged words after a push and sends word i of
  WMASK from `ISSUE_Wi`, so a word the previous push of the same record staged with the same value
  need not be stored again (3 RISC-V cycles each). SKIP is the call site's static claim from the
  sequence (a VOP's a line, m and rows word after a RED on the same rows; an ST's SPM line after the
  VOP whose d it is; SCALE's ADDC after its MULC differs only in w0); WMASK stays the op's, so the
  pushed words are exactly the op's, and host builds store every word and check each claim against
  the words staged since the sequence's `stage_reset` (panic otherwise), so the host twin tests run
  every claim. RMS_NORM and ROPE have a straight-line hot entry for their common decode record (one
  row with a weight; one token without a table or freq_scale) while the s-register shadow holds their
  parameters, which tail-calls the general fast form otherwise (no call in either path); a single
  entry for every shape measured about 60 RISC-V cycles more (register pressure). ADD, MUL and GLU
  wait until one ISSUE_CREDIT read covers their four pushes (as RMS_NORM), then push with `push_s`.
- *Measured* (`fw_exec_perf_m7`: decode records at n = 896 in chains, spin-free work, maxperf, deep
  queues; §7.8 has the other harnesses and the layer; W2b fix, 2026-10-31): ADD 335, MUL 336, SCALE
  305, SCALE with a bias 336, RMS_NORM 324, GLU 330, ROPE 325, UNARY 308, GET_ROWS 254, CPY 223
  RISC-V cycles of work per record: ≤ 350 met by every kind (`--strict` asserts 350 for each, no
  exemption); at the integration 341 / 342 / 322 / 380 / 395 / 341 / 420 / 315 / 264 / 230, before it
  428 / 429 / 406 / – / 474 / 423 / 466 / 406 / 349 / 317. Decode records outside these shapes that
  still miss 350 (§7.8): Gemma 4's per-head norms (rows > 1, 405–455), ROPE with a freq-factor table
  or freq_scale ≠ 1 (439–475), and records whose s-register values change (each MOVI is one push more).

**DTCM state** is rebuilt at every launch (SW-4).

### 7.4 MUL_MAT kernel

The kernel follows the lowering of `docs/spec.md` §12, so its numerics match the recorded model
traces bit for bit (`traces/model-v1conv` checks this). Every trace of that set has T_MAX 1, so the
push-stream comparison there covers single-tile ops only; for T > T_MAX the kernel pushes the LD of
tile t+1 before the ST of tile t (the double buffering below).
Per T tile (`plan.hpp`, T_MAX and the ring budget):

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

Kernel sources: `sw/ctrl/kernels/k_elem.cpp` (ADD, MUL, SCALE, UNARY and the element-wise tile engine),
`k_glu.cpp`, `k_rows.cpp` (GET_ROWS, CPY), `k_norm.cpp` (RMS_NORM), `k_rope.cpp`, `k_kv.cpp`
(SET_ROWS_KV), `k_fattn.cpp` (FLASH_ATTN); planners `tsr/plan_elem.hpp`, `plan_norm.hpp`,
`plan_attn.hpp`; references `sw/ctrl/ref/elem_ref.cpp`, `rmsnorm_ref.cpp`, `rope_ref.cpp`,
`kv_ref.cpp`, `fattn_ref.cpp`. Each kernel's header comment states its tile-op sequence; the table
summarises them (W2b, as built).

| Kind | Tile composition (primitives from VU/AT) | Scratch (lines, per tile) | Numerics reference |
|---|---|---|---|
| ADD/MUL/SCALE | per row tile (one at decode shapes; else MUL_MAT's double-buffered tiles, §7.4): LD the a rows, LD the b part (b rows, b row 0 for a broadcast, the p rows of a period, or the scalars' line(s)); VOP ADD / MUL in place (b stride L, or 0 for a broadcast; one VOP per period block, VU §9); with FLAG_B_SCALAR VOP ADDC / MULC with `cmode` 2 (the scalars as a c vector: one row, per-row scalars such as Gemma's `inp_scale_rows`) or `cmode` 1 (one scalar for several rows: SOP.LDS s12 from the loaded line); SCALE: SOP.MOVI s12 = s (s13 = b when b ≠ 0) through the s-register shadow, VOP.MULC c = s12, then VOP.ADDC c = s13 (the bits of MULCADD with a b row of copies of b); ST | rows·L + the b part (the VOPs are in place); a single tile in the one-tile block (§7.3) | `elem_ref.cpp` (`tsr_ref_vop` per row) |
| RMS_NORM (+weight) | per row tile, block [x rows \| w \| per-row vector]: LD x, LD w (one row for period 1, the tile's rows for p > 1; tiles never cross a period block), `SOP.MOVI` s2 = 1/n and s3 = eps through the shadow, `RED.SUMSQ` (post RMS, sk = s2, se = s3; after the tile's last LD) → s1 (one row) or the per-row vector, `VOP.MULCB` in place (`cmode` 1 / 2, b = w with stride 0 for period 1), or `VOP.MULC` without weight; ST. Per-head norms (Gemma q / k / v norms) are the same ops with rows = n_head·T and n = head_dim (VU §8.1; one record per tensor: VU §8.1's fused q / k / v form has one weight and no OPL record) | rows·L + w + ⌈rows/16⌉ | `rmsnorm_ref.cpp` (same order); bit-identical to the spec's 7-op lowering `docs/spec.md` §9 (VU §6.2) |
| GLU | LD gate rows, LD up rows, VOP SWIGLU / GEGLU16 / GEGLU in place on the gate rows, ST | 2·rows·L | `elem_ref.cpp` (`tsr_m_silu`, `tsr_m_gelu16`, `tsr_m_gelu_tanh`) |
| ROPE | per token tile (one at decode; else multiples of 16 tokens so the positions start on a preload line): LD x (H = n_heads·head_dim/16 lines per token), LD the freq-factor (or A) table (per tile), `SOP.MOVI` TS, M and freq_scale through the shadow (freq_scale only when ≠ 1.0), ROPE over the tile's tokens in place (positions: the I32 words in the preload area, read by the VU as an SPM operand), ST | tokens·H + table | `rope_ref.cpp` (`tsr_ref_vu` ROPE, VU §6.7) |
| GET_ROWS | per tile: one LD per run of consecutive ids when the source rows are dense, else per id (the RISC-V reads the ids from the preload area), one ST for dense dst rows; a row beyond half the budget moves in chunks | tile rows·row lines; the one-tile block when it fits | copy (`elem_ref.cpp`) |
| SET_ROWS_KV | per run of consecutive cells (index order, at most 64 rows): LD the source rows, QACT A32 (K = E, at most T_MAX rows per QACT), KVST of the run (AT §5.1) | run·(E/16 + QA_LINES(A32, E)); the one-tile block when it fits | `kv_ref.cpp` (`tsr_qact` + `tsr_ref_kvst` per row) |
| FLASH_ATTN | AT §6.6 (normative; at the W2b integration reconciled with `tsr_ref_fa`: query tiles, the head subsets inside each chunk, the shared chunk planner `sw/sim/include/tsr_fa_plan.h`): per query tile LD Q, QACT per token; per chunk ATTQK per head subset (FAST) or ATTQK + AMASK (GENERAL), ASMAX, ATTPV per head subset (LSUM), AMERGE, with ATTPV / AMERGE issued one chunk behind ATTQK / ASMAX; ANORM; ST. Buffers S[3], P[2], MK[4] and, for GENERAL chunks, the mask rows M[2] (AT §7.1; the mask LD into M[k mod 2] waits for the AMASK of chunk k − 2), every op ≤ 2 deps | one ring block per op (`plan_attn.hpp` `fa_plan`: 18,948 lines for Qwen2.5-32B decode) | `fattn_ref.cpp` (`tsr_ref_fa`, AT §13.1) |
| CPY | LD rows, ST rows (one run when dense on both sides); rows beyond half the budget in chunks | block; the one-tile block when it fits | copy (`elem_ref.cpp`) |
| UNARY | LD, VOP func in place (TANH, GELU16, SILU, EXP, GELU, SIGMOID), ST | rows·L | `elem_ref.cpp` (`tsr_ref_math`) |

The numerics of a composite kernel are defined by its tile-op sequence. Each kernel's header
comment states the sequence. A host C++ reference in `sw/ctrl/ref/` composes the same `tsr_ref`
calls in the same order, and the kernel tests compare the firmware against it bit-exactly.
Changing a sequence changes logits, so the change goes with a firmware ABI bump and new klcheck
baselines.

### 7.6 `selftest` kernel (SW-28)

| Id | Test |
|---|---|
| 1 | LD/ST round trips on every channel: 1, 63, 64, 65 and 4,096 lines, cross-channel copies |
| 2 | QACT A32/A256 on host-written data |
| 3 | MVM Q8_0 and Q4_K: T = 1, T = T_MAX, ACC |
| 4 | each VU function and reduction |
| 5 | each AT op |
| 6 | concurrency: a `matvec_batch`-style pipeline (`docs/ctrl-riscv.md` §8.7) on all units |
| 7 | errors: DRAM_RANGE, ILLEGAL_OP, BAD_DEP, DEP_FAILED chain, QABORT then QCLEAR recovery |
| 8 | SPM window byte/half/word, ordering of a store before a push |
| 9 | log records and PING |
| 10 | credits: fill every queue under QHOLD; each credit must read 0 (exact under the hold, §4.7) |

`tsr-selftest` runs each test on the device and, in process, on the simulator with the same image,
inputs and args. It compares DONE, every output byte and the per-unit completion counts.

### 7.7 `replay` kernel

1. LD chunk 0 (and chunk 1) of `push.bin` into the buffers, then read `base_u` = `ISSUED_SEQ[u]`
   for every unit and `pidx0` = `ISSUE_PIDX`, so that no push of the kernel lies between the
   replayed records of a stream of up to two chunks.
2. Stream `push.bin` through SPM in double-buffered chunks (at most 1,364 records = 1,023 lines;
   without BUF the two buffers sit just below the runtime argument range). The kernel's own later
   chunk LDs are added back when DMA deps are mapped.
3. For each record:
   - kind 0: issue the words with the push value and deps mapped back as (u, base_u + rel_seq);
     with flag SERIAL, the deps are replaced by the ticket of the previous kind-0 record (a
     dependency chain: with I1 this gives strictly sequential execution);
   - kind 1: write QCTRL;
   - kind 2: write TRACE_MARK.
4. Run `join_all()`. RET = 0 if no unit is in error state, else 1 + (`ERR_PIDX` of the first error
   − `pidx0` − the kernel's own chunk LDs pushed before it), i.e. relative to the first replayed
   record; the runtime reports the error record itself in DONE r3–r6 (EX §4.6). Negative RET
   (KERNEL_ERR): 0x8000_0101 BAD_ARGS, 0x8000_0102 TOO_LONG (more than 130 chunks), 0x8000_0103
   STREAM (a chunk LD failed).

### 7.8 Performance targets (*estimates*, checked in P3/P6)

- **Executor:** ≤ 400 RISC-V cycles per graph op on average (2.4 µs at 166.7 MHz). For Qwen2.5-0.5B
  (≈ 25 ops per layer, ≈ 230 µs of MVM time per layer) the RISC-V needs ≈ 60 µs per layer and
  stays ahead. *Measured at W2a* (simulator CPI model, `fw_m6_perf`, 0.5B decode MUL_MAT lists):
  1,986 RISC-V cycles per op record (1,047 instructions; 2,553 per MUL_MAT with its JOINs), the
  executor's own share ≈ 200 cycles per record (`fw_exec_perf`); one decode layer (7 MUL_MAT +
  2 JOIN) costs the RISC-V 107 µs against 197 µs of MVM time, and the MVM unit still idles on issue
  for ≈ 14 % of its span (the small k / v projections). *Measured after the executor optimisation*
  (§7.3 "Hot path", SW-35; simulator, fixed schedule, 166.7 / 276.3 MHz, `fw_m6_perf` on the same
  lists, Ibex maxperf (CC §3.2, the default) / small; "work" = the RISC-V's busy cycles up to the
  launch's last op push, without the iterations of wait loops after their first test, `fw_prof.h`):
  387 / 485 RISC-V cycles of work per op record with deep queues (278 instructions; 498 / 624 per
  MUL_MAT; W2a 1,518 / 1,966), 401 / 497 with the U200 queue depths (the difference: re-reads of
  `ISSUE_CREDIT` while the MVM queue is full, i.e. while the RISC-V is ahead anyway), 384 / 479 with
  the backend's flags (no MARK); by the W2a method (U200 depths, every busy cycle) 437 / 520
  (W2a 1,547 / 1,977). A JOIN record (one push) costs 140 / 163 (W2a 185 / 220), a NOP record 85 /
  99 (156 / 191), `fw_exec_perf`'s five-push test kind 476 / 595 (1,027 / 1,326). One 0.5B decode
  layer costs the RISC-V 39.4 / 48.3 µs against 197 µs of MVM time (W2a 110.4 / 141.0 µs), the MVM
  never idles on issue (0.0 %, W2a 8.1 / 13.5 %), and the layer's launch takes 220.2 / 224.1 µs
  (W2a 256.2 / 276.8 µs). The ≤ 400 target is met on the CC §3.2 configuration (maxperf) by one
  metric: the spin-free executor work with deep queues (387). With the U200 queue depths the same
  metric gives 401 and W2a's method (every busy cycle, U200 depths) 437, both above 400. The M6
  result is the issue rate, not the cycle count: the MVM never idles on issue (0.0 % of its span on
  both Ibex tables, at 125, 166.7 and 250 MHz). On small the executor costs 485 / 497 / 520 by the
  three metrics and still stays ahead of the MVM. The same lists with the
  whole image at `-Os` (no hot-path `-O2`, §6.1): 391 / 485 cycles of work per record (deep queues),
  442 / 521 by the W2a method, a layer 51.4 / 62.9 µs of RISC-V time and 232.1 / 238.9 µs push →
  DONE. At clk_rv 250 MHz (`TSR_SIM_RV_RATIO` 0.905, CC §3.3) the cycles per record barely move
  (396 / 492 work, deep queues) and the layer costs the RISC-V 26.3 / 31.6 µs (push → DONE 216.6 /
  219.1 µs); at 125 MHz 50.6 / 61.6 µs (224.0 / 229.4 µs). Per layer of a longer list (3 vs 1
  layers) the RISC-V works 3,636 / 4,498 cycles = 21.8 / 27.0 µs against 239.6 µs of device time.
  *Final tree* (after the review fix of the streamed chunk order, §7.3; `op_loop` for in-place lists
  is unchanged instruction for instruction, the code after it moved): every figure above within
  ±1 RV cycle and ±0.1 µs (maxperf / small: 387 / 484 deep-queue work, 402 / 497 with U200 depths,
  438 / 520 by W2a's method, 383 / 478 without MARK, a layer 39.5 / 48.3 µs of RISC-V time and
  220.3 / 224.2 µs push → DONE; 250 MHz 395 / 491, 125 MHz 384 / 482; JOIN 140 / 163; M7
  projection 656 / 831 cycles per VU record, 6.2 % / 11.2 % of the layer), ITCM 32,896 B.
- **M7 projection** (*measured*, executor-optimisation integration, 2026-10-30): `fw_m6_perf` with
  the `fw_exec` test image runs the decode layer with 25 short VU records per layer (test kind 0x38
  of `tests/fw/exec_kinds.cpp`: LD, 2 or 3 VOPs, ST, through the generic handler call as W2b's kinds
  will go; in chains at the attention norm, RoPE / KV / attention, FFN norm, GLU and residual
  points, split 3 : 12 : 4 : 3 : 3) and compares the launch with a reference run whose RISC-V runs
  at 4 × F_CORE (a near-ideal issuer). Such a record costs the RISC-V 656 / 832 cycles of work
  (maxperf / small; 3.9 / 5.0 µs at 166.7 MHz) and adds ≈ 2.0 µs to the layer's device time (with
  the near-ideal issuer the layer takes 290.7 µs against 239.6 µs without the records). Per layer
  the RISC-V works 116 / 147 µs against ≈ 291 µs of device time, so it is ahead on average; but each
  record queues an LD and an ST on the DMA unit (U200 depth 8, CC §7.4), so the in-order issuer
  banks at most ≈ 4 records ahead of a dependent chain, and the layer takes 310.0 / 327.5 µs instead
  of 290.7 µs: the RISC-V's speed costs 6.2 % / 11.2 % of the layer at 166.7 MHz, 1.6 % / 4.1 % at
  250 MHz and 12.7 % / 19.8 % at 125 MHz (the W2a firmware: 31 % / 39 % at 166.7 MHz). With deep
  queues (31) the cost is 0 after a launch's first layer (at 166.7 and 250 MHz; small at 125 MHz
  3.8 %). W2b's frequent VU kinds therefore take the MUL_MAT fast path's form (§7.3 "Hot path":
  leaf handler, arguments read in place, `alloc_dma_first`, one credit read per record; *estimate*
  ≈ 140 + 25–30 cycles per push, ≈ 260 per such record = 1.6 µs, below the ≈ 2.0 µs it adds on the
  device); a deeper DMA queue is the hardware alternative.
- **M7 kinds** (*measured*, W2b integration and fix, 2026-10-31; simulator, fixed schedule, Ibex maxperf,
  F_RV / F_CORE = 166.7 / 276.3 MHz unless noted; "work" = fw_prof.h's spin-free RISC-V cycles; the
  backend's list flags, no MARK). Short records at the 0.5B decode shapes (`fw_exec_perf_m7`, n = 896,
  deep / U200 queues): ADD 335, MUL 336, SCALE 305, SCALE with a bias 336, RMS_NORM 324 / 326, GLU 330,
  ROPE 325, UNARY 308, GET_ROWS 254, CPY 223: **≤ 350 met by every kind** (W2b fix: staged-word reuse,
  hot entries and the `op_loop` variant, SW-39; at the integration RMS_NORM 395, ROPE 420 and SCALE
  with a bias 380 missed it). The packages' harnesses at the larger decode shapes (deep / U200 queues;
  the units, not the RISC-V, set the pace there, so the figures include first tests of credit waits):
  `fw_kernels_elem_perf` ADD 32B residual 351 / 356, MUL 12B norm weight 350 / 357, per-head weight
  348 / 354, GLU 0.5B 351 / 355, GLU 12B (n 15,360, a ring block, SW-36) 461 / 470;
  `fw_kernels_normrope_perf` RMS_NORM 32B 347, ROPE 32B 325 / 331. **Not met** at decode shapes:
  Gemma 4's per-head RMS_NORMs (q_norm 16 × 256 with weight 443 / 455, v_norm 8 × 256 without 405 /
  408: the general fast form, about 120 cycles above the hot entry), ROPE with Gemma 4's rope_freqs
  table 465 / 475 and with freq_scale ≠ 1 439 / 440 (the general fast form), and every record whose
  s-register values change (one SOP.MOVI push more each; e.g. Gemma 4's ROPEs alternate TS between
  SWA and global layers). Those records are a few per layer of models whose layers take milliseconds
  on the device, so they do not move the issue loss below. SET_ROWS_KV decode record 345 (351 at the
  integration, 383 before); a Qwen2.5-32B decode FLASH_ATTN at 4,096 cells 3,378 cycles = 20.3 µs of
  RISC-V work against 140.0 µs on the units (the MVM never waits for a push; `fw_kvfa_perf`, AT §7.4).
  On Ibex small every figure is ≈ 1.25×.
  *One Qwen2.5-0.5B decode layer* (`fw_m7_layer_perf`: layer 0 with its GGUF weights; RMS_NORM, Q / K /
  V with bias addends, ROPE ×2, SET_ROWS_KV ×2, FLASH_ATTN at n_kv 256, wo + residual, RMS_NORM, gate,
  up, GLU, down + residual: 15 records + 1 JOIN; 3 vs 1 layers; W2b fix figures, the integration's in
  brackets): the RISC-V works 50.7 µs per layer (53.1) against 270.6 µs of device time; one layer's
  launch takes 286.5 µs push → DONE (287.1); per record (executor included, attributed between the
  executor's RT_INFO writes, averaged over the list's records of the kind, so the first of a list,
  which pushes its MOVIs, is in it) MUL_MAT 451 (455), RMS_NORM 350 (409), GLU 360 (402), ROPE 358
  (426), SET_ROWS_KV 362 (357), FLASH_ATTN 2,679 (2,686), JOIN 132 (138), with the U200 depths. Against
  a near-ideal issuer (F_RV = 4·F_CORE, 787.4 µs for 3 layers) the RISC-V's speed costs **5.1 %** (5.4)
  of the launch with the U200 queue depths, **2.1 %** (2.2) with deep queues and **2.1 %** with only the
  DMA queue 31 deep, **2.5 %** at F_RV 250 MHz with the U200 depths; on Ibex small 8.2 / 3.8 / 4.0 %
  (8.6 / 3.9 / 4.2). The executor-optimisation projection (25 generic short records, 656 cycles each)
  was 6.2 % (`fw_m6_perf`'s projection with its test kind still gives 6.1 %). The loss is not the short
  records': before the integration's per-record savings (≈ 60–90 cycles per short record) it was 5.5 /
  2.2 / 2.5 %, and the fix's further 40–90 cycles on RMS_NORM / ROPE took 0.3 points. It is
  (a) the DMA queue depth: 8 entries fill with the decode chain's LD / ST ops waiting behind an ST that
  waits for its MVM, so the RISC-V blocks on credits and reaches FLASH_ATTN late (a 31-deep DMA queue
  alone removes 3.0 points), and (b) FLASH_ATTN's own RISC-V time (2,679 cycles = 16 µs per record at
  n_kv 256, mostly its planning before the first ATTQK), on the critical path between V and wo: the
  MVM's issue-idle time (5.6 % of its span; near-ideal 0.9 %) is 42.1 of 45.6 µs FLASH_ATTN's over 3
  layers with the U200 depths (9.1 of 10.3 µs with deep queues). Both are open items of W2b (a deeper
  DMA queue in the CC RTL; planning the next
  FLASH_ATTN during credit waits, or pushing its first ops before the rest is planned).
- **Launch:** EX §4.11 (empty-launch round trip ≤ 10 µs target; descriptor push → kernel entry
  ≤ 5 µs; answer record → host ≤ one poll interval). An exec start and completion wait on the
  card cost 35 µs, measured before the control core was added (`docs/platform-u200.md`). *Measured at W2a* (`tsr-fw.elf`, fixed schedule, F_RV / F_CORE =
  166.7 / 276.3 MHz): descriptor push → kernel entry 605 core cycles = 2.19 µs; empty launch push →
  DONE in the LCQ 1,538 core cycles = 5.57 µs (device part); 8 outstanding launches 1,579 core
  cycles each (175 K launches/s). *Measured after the executor optimisation* (the runtime is
  unchanged; Ibex maxperf, the default / small): descriptor push → kernel entry 1.68 / 2.19 µs,
  empty launch 4.23 / 5.50 µs, 8 outstanding launches 4.34 / 5.64 µs each (230 K / 177 K
  launches/s); at 250 MHz 1.12 / 1.46 µs and 2.82 / 3.67 µs.
- **Image size:** ITCM ≤ 96 KiB of 128 KiB (SW-75, which amends SW-38: 80 KiB with the M7 kinds, 48 KiB before them);
  DTCM ≤ 56 KiB of 64 KiB including the 8 KiB stack,
  the 512 B trap stack and the 32 KiB executor tables. *Measured at W2a* (`tsr-elf size`): ITCM
  25,720 B, DTCM 46,312 B. After the executor optimisation (hot path `-O2`, §6.1): ITCM 32,808 B
  (32,896 B after the review fix's streamed-order code),
  DTCM 47,968 B (the ring's table +1 KiB, the ticket table's sentinel); the whole image at `-Os`
  would take ITCM 30,628 B. *With the M7 kinds* (W2b integration): ITCM 69,732 B (the packages landed
  at 75,196 B; the one-tile block and the shared out-of-line helpers took 5.5 KB off), DTCM 49,448 B;
  after the W2b fix (hot entries, GLU's ring block, the `op_loop` variant) ITCM 71,588 B, DTCM 49,448 B.
  Of the ITCM, the M7 kinds take ≈ 36 KB (FLASH_ATTN ≈ 12 KB with its planner, the element-wise and row
  kinds ≈ 14 KB, RMS_NORM / ROPE ≈ 9 KB, SET_ROWS_KV ≈ 3 KB): each short kind keeps its fast path as its
  own leaf (a shared body measured +22–29 RISC-V cycles per record) and the general paths are cold
  (compiled for size). *With the MoE kinds* (SW-61, SW-62): ITCM 78,520 B (was 71,712 B with the 14 kinds of the base: the two kinds take 6.8 KB, the general paths of both are
  cold), DTCM 49,680 B (+176 B). *With the Gated DeltaNet kinds alone* (SW-67 to SW-69, §7.10): ITCM 78,576 B, DTCM
  49,912 B. *With the MoE and the Gated DeltaNet kinds together* (21 kinds, SW-75): ITCM 84,056 B (64.1 % of
  128 KiB, 85.5 % of the budget), DTCM 50,088 B. Each set fits the old 80 KiB budget on its own; both together
  do not, and SW-75 raises the budget to 96 KiB (the part of the upper 48 KiB that SW-38 reserved for M8 MoE).
  The budget applies to the production image (`tsr-fw.elf`; `fw_elf_image`,
  `fw_elf_test_image`); test images (production plus test kernels) are stamped against the whole ITCM.

### 7.9 MoE kinds (SW-60 to SW-66, SW-76, SW-77)

Status: built and checked on the simulator (November 2026). On the card since 2026-11-09 with the K4 full-feature image ([klcheck.md](../klcheck.md), [performance.md](../performance.md)): the KL checks pass and the logits are bit-identical to the simulator. Source: `sw/ctrl/kernels/k_moe.cpp`,
planners `tsr/plan_moe.hpp`, references `sw/ctrl/ref/moe_ref.cpp` and `tsr_ref_moe_route` (`sw/sim/tsr_ref_vu.c`),
backend `tsr-build.{h,cpp}` and `ggml-tesseract.cpp`. Spec side: `docs/spec.md` §4 (expert tensors), §13.

**The graph.** For Gemma 4 26B-A4B (30 layers, 128 experts, 8 used, hidden 2816, expert width 704; a dense GELU
branch beside the experts) llama.cpp's `build_moe_ffn` emits, per layer and in this node order: the router input
(RMS_NORM, SCALE by 1/√2816, MUL by `ffn_gate_inp.scale`), the router MUL_MAT (`ffn_gate_inp`, F32 [2816, 128] in the
GGUF), SOFT_MAX, ARGSORT (descending) with a VIEW of the first k, GET_ROWS of the probabilities by those numbers,
SUM_ROWS, CLAMP (6.1035e-5, inf), DIV, the expert input norm, MUL_MAT_ID (gate-up: one tensor `ffn_gate_up_exps`, Q4_K
[2816, 1408, 128], src1 one row per token), two VIEWs and a GLU (GEGLU), MUL_MAT_ID (down: Q5_0 or Q8_0 [704, 2816,
128], src1 one row per slot), RESHAPE + REPEAT + GET_ROWS of `ffn_down_exps.scale` by the same numbers and a MUL,
the MUL by the normalised weights, eight VIEWs and seven ADDs; the dense branch, the norms and the residual follow.

**SW-60 Expert tensors (EXP_STRIDE).** A 3D tensor `[K, N, n_expert]` whose name contains `_exps` is placed as a
weight with `n_expert` matrices of N × K: expert e at `W_OFF + e × EXP_STRIDE` on every channel, with `EXP_STRIDE` =
`R × ROW_BEATS` beats (spec §4); one expert MVM uses all four channels, as a dense MUL_MAT does. Gemma 4: gate-up
8,800 beats (25 per row, 352 rows per channel), down 8,448 beats (Q8_0 K = 704: 12 beats, 704 rows per channel), 68.75
and 66.00 MiB per channel and layer. `tsr_store_expert_rows` / `tsr_load_expert_rows` scatter and gather per expert (the
row sink of the dense path per expert), the conversion at load is the dense one (Q5_0 to Q8_0, Q6_K to Q8_0), a split
upload stages the tensor and stores it when complete, an ANY-buffer source (test-backend-ops) converts lazily with
the expert count in the cache key. Other 3D tensors stay raw. The router is an ordinary 2D weight: the F32 matrix is
converted to Q8_0 (spec §5).

**SW-61 MOE_ROUTE (0x1E, rev 1, 9 words).** Words `LOGITS, LOGITS_STRIDE, IDS, IDS_STRIDE, W, N_EXPERT, N_USED, T,
FLAGS` (`opl::moer`). Per tile of 64 tokens (one ring block, `plan_moe.hpp`): LD the logits rows; the softmax on the
VU (RED.MAX → vector, VOP.EXPSUBC with that vector as c, RED.SUM with post-op RCP, VOP.MULC); **join**; the RISC-V
selects per token the k largest probabilities by integer compares on the words and writes the selected values to a
line and the numbers to another; RED.SUM with post-op RCP over the k values and VOP.MULC give the weights; **join**;
for T > 1 the RISC-V packs the weights (k words per token), and the STs write the numbers into the first line of
each row of the IDS tensor (the ARGSORT result: rows `IDS_STRIDE` lines apart, words k .. 15 of the line are 0) and
the weights, packed, into W (the DIV result `[k, T]`). The arithmetic is `tsr_ref_moe_route` in `sw/sim/tsr_ref_vu.c`
(VU §6.2 trees, EXPK, RC): `m` = max, `e` = EXPSUB(l, m), `r` = RC(tree sum of e), `p` = FM(e, r); `(id, q)` = the
top k of `p`; `r2` = RC(tree sum of the k values q in selection order), `w` = FM(q, r2). ggml's graph does the same with
`expf`, a double sum and a true division: the device differs at rounding level (probabilities by up to ~1e-6
relative, weights by up to 5 ulp on 512 random rows) and its weights sum to 1 within 1e-6.
The CLAMP is not computed: the k largest of n probabilities sum to at least k/n, so the clamp is the identity for
every row without a NaN (the builder requires `clamp_min × n ≤ k`), and a NaN passes it.

**SW-63 Tie rule and NaN.** The selection is the order of the VU TOPK (VU §6.8): the DAZ'd value under the total
order (−inf < … < +inf < NaN), larger first, equal values by the smaller index; every NaN is the canonical NaN, ranks
first, and ties by the smaller index. The firmware insertion loop (`select_topk`) is checked against `tsr_ref_topk` on
every row length 1 to 70 with every k, three value classes, and 4,000 rows of 128 and 512 words (`fw_kernels_moe_host`: 7,000 rows). ggml's
CPU sorts the indices with `std::sort` and `a > b`, so equal values come out in an unspecified order and NaN breaks
the strict weak ordering. Measured on the pinned ggml-cpu (`ref_moe`): for one pair of equal logits at places 8 and 9
of 128, ggml takes the smaller index in 241 of 512 rows and this rule in 512 of 512; with many equal values at the
boundary the selected sets differ in 511 of 512 rows; with a NaN logit every weight is NaN in both and the sets
differ (ggml: an arbitrary permutation, here 0 .. k−1). Ties in the f32 probabilities of a trained router are
rare (two logits within an ulp of each other), and the continuous case agrees (512 of 512 rows identical, except
where the device's exponential and Q8_0 router move the 8th and 9th probabilities across each other).

**SW-62 MUL_MAT_ID (0x1F, rev 1, 17 words).** Words `DST, DST_STRIDE, X, X_SLOT_STRIDE, X_TOK_STRIDE, IDS,
IDS_STRIDE (bytes), W_OFF, EXP_STRIDE, FMT, N, K, T, N_USED, N_EXPERT, SCALE, FLAGS` (`opl::mmid`). One MUL_MAT of T
= 1 per (token, slot) against `W_OFF + id × EXP_STRIDE`; x rows shared by the slots (`X_SLOT_STRIDE` 0: gate-up) or
one per slot (down); y row (token, slot) is row `token × N_USED + slot` of the dense dst; with a `SCALE` table
(F32 [n_expert]) the y block is multiplied by `table[id]` (VOP.MULC with an s register) before its ST. The handler
LDs a chunk of the ids rows (up to 128 lines) and the scale table into the executor's one-tile block, **joins**, and
reads the numbers with the RISC-V (the exact SPM window); per token it takes one ring block `[x | QA | y_0 … y_(k−1)]`
and pushes LD / QACT / MVM of slot j + 1 before the ST of slot j (MUL_MAT's double buffering), the next token's
block while this one's STs are pending. An id outside [0, N_EXPERT) gets W_OFF 0xFFFF_FFFF: the MVM fails with
DRAM_RANGE and the launch ends with TILE_ERR on this op. The returned ticket is the last ST. Numerics: the dense
MUL_MAT's (`mulmat_ref_token`), then `tsr_ref_vop_elem(MULC)` with the scale.

**Joins.** Both kinds wait inside the op stream, which the executor contract forbids to other kinds (`tsr/exec.hpp`):
MOE_ROUTE twice per tile of 64 tokens, MUL_MAT_ID once per chunk of ids (once for a token count below ~16 with
argsort-shaped ids rows, 16 tokens per chunk at 512-byte rows). The wait is on the handler's own ops; records already
pushed keep running and only the issue of later records stops.

**SW-64 Backend: fusion of the router and the scale chain.** supports_op takes MUL_MAT_ID (probe-mode lowering and
`plan_mul_mat_id`) and, with the kinds in the image, the nodes of the two chains by their local shape (SOFT_MAX
without mask or bias, ARGSORT descending of such a SOFT_MAX, GET_ROWS of its reshaped output by the sorted view, SUM_ROWS,
CLAMP, DIV, REPEAT of a 1D F32 leaf, GET_ROWS of a REPEAT). `tsr_build_fusion` finds the groups node for node with
the use counts of llama.cpp's graph: the SOFT_MAX heads six nodes that become one MOE_ROUTE record (ids into the
ARGSORT tensor, weights into the DIV tensor; the record sits at the SOFT_MAX); the MUL_MAT_ID heads REPEAT, GET_ROWS and
MUL: its record carries the scale table (the RESHAPE of the weight leaf, or the scheduler's copy of the reshaped view when the table stays in host memory, as in Gemma 4: no kind takes a MUL_MAT_ID weight that is not an expert tensor), writes its own tensor, and a CPY record moves the result into the MUL's
tensor (no record when the allocator put both at the same address, as it did in every graph looked at). A
group member that did not fuse aborts at graph_compute naming the node: a graph other than llama.cpp's
`build_moe_ffn` is not supported. A route with more than 64 tokens stores its outputs in a scratch area and copies them
(SW-76). The OPL builder knows both kinds' DRAM ranges (`kind_ranges`).
`TSR_KINDS` without MOE_ROUTE leaves the router chain on the CPU (the ids are copied to the device for the MUL_MAT_ID);
without MUL_MAT_ID the experts stay on the CPU (the loader's weight probe then places them in host memory).

**SW-65 M6 and M7 paths.** M6 (`TSR_KINDS=m6,+MUL_MAT_ID`; klcheck and numcheck `--phase m6` set it): the host issues the
matrix products (the router MUL_MAT, both MUL_MAT_IDs per layer); llama.cpp's own CPU ops do the softmax, the top-k, the
renormalisation, the GLU, the scale, the weights and the sums, because the CPU's `std::sort` and double sums are then
the reference of the selection and the ids reach the device unchanged. This is the card's M6 step too: MUL_MAT_ID is a
firmware kind, so it needs the new image's `tsr-fw.elf` (a firmware change, no bitstream change). M7 (all kinds): the
whole layer, router and experts, is part of the one launch per token. The selection is software on the RISC-V in both
senses: nothing needs the TOPK RTL (`CAPS` bit 3), which the K4 wrapper leaves off.

**SW-66 Prefill and token grouping.** The kinds run an MVM of T = 1 per (token, slot), so a prefill of T tokens
reads T × k experts of every layer once each, with no reuse (25 tokens read 200 experts, about 100 distinct ones if the
choices were uniform; 512 tokens read 4,096 for 128 experts), where a dense layer reads its weights once per T_MAX tokens. Grouping is
designed, not built: per ids chunk the RISC-V counting-sorts the (token, slot) pairs by expert, and per expert pushes
LDs of up to T_MAX x rows (gathered), one QACT and one MVM of T' tokens, one VOP.MULC (the scale is the same for the
group) and T' STs. Numerics do not change (an MVM row is independent of T), so the references stay valid. It pays when
tokens per expert reach T_MAX: T × k / n_expert ≥ T_MAX, i.e. T ≥ 64 (T_MAX 4) on Gemma 4. It is not built because
the pairs need a scratch list in SPM (the stack of the firmware is 8.7 KiB) and a second push order to verify, for a gain
that only long prompts see.

**SW-76 Outputs of a route with several tiles.** A route of more than 64 tokens (two or more tiles) does not write the
ids and the weights where the graph has them. The MOE_ROUTE record writes them to a scratch area of the device, `T` lines
of ids (one per token) and then the packed weights (`ceil(T k / 16)` lines), and two CPY records after it move them: the ids
into the first line of each of the `T` rows of the ARGSORT tensor, the weights into the DIV tensor (one row of
`ceil(T k / 16)` lines). The record reads the logits tile by tile and stores the outputs of a tile before it loads the next
one. ggml-alloc plans the graph node by node, with the group's nodes one after the other, so it may put the DIV tensor or the
ARGSORT tensor over logits that the record still has to read: the weights of Gemma 4's layer 28 lie at the start of the
logits at 256 tokens, and the layer test of Qwen3-30B-A3B (SW-77) met a layout of this kind in layer 0 at 70 tokens. The
first build refused such a layout and aborted at `-ub 256` and above for Gemma 4 (the default is `-ub 512`) and at `-ub 128` for Qwen3. The CPY records run after the record is done, so after every load, and no placement of the two tensors can hurt a
later tile. The scratch is one window in device DRAM, taken at the first such route from the top of IO (1 MiB, or more when the
route needs more; it is replaced by a larger window when a later route needs more: 192 KiB at 2,048 tokens and k = 8). All
layers share it: the OPL builder orders the next route's record after the CPYs that read it (their DRAM ranges overlap). A route
of up to 64 tokens (decode, a short prompt) is one tile and unchanged: one record, outputs in place. The record's words,
the firmware and the numerics do not change, so there is no kind revision and no new image: the CPY records copy the
words of the record unchanged. The kinds needed are MOE_ROUTE and CPY; without CPY (`TSR_KINDS`) the router of a graph of more
than 64 tokens stays on the CPU. Cost: two CPY records of
`T` and `ceil(T k / 16)` lines per layer and ubatch, a few microseconds of DMA against the experts' MVMs. Tests:
`ggml_tesseract_moe` (`run_route_layouts` in `test-tsr-moe.cpp`: 65 to 11,000 tokens with the weights at the start, in the
middle and at the end of the logits, the ids over the logits row for row and a tile in).

**Limits.** n_expert ≤ 512 for MOE_ROUTE and a multiple of 16 (the ids rows are whole lines); k ≤ 16; ≤ 4,096 experts
per tensor; N a multiple of 16; K a multiple of the format's block; softmax gating with a normalised top-k only (no
bias, group selection, sigmoid or softmax-after-top-k: those graphs run the router on the CPU).

**SW-77 The Qwen3-30B-A3B graph (llama.cpp `qwen3moe`).** `src/models/qwen3moe.cpp` calls the same `build_moe_ffn` with
softmax gating and `norm_w`, so its router group is the one of SW-64 and the kinds, planners and fusion pass did not change.
The model: 48 layers, hidden size 2,048, 128 experts of width 768 with 8 used, no shared expert, no dense branch. Where
the graph differs from Gemma 4's, in node order:

- Attention: a Q of 4,096 elements for a hidden size of 2,048 (32 heads of 128, 4 KV heads, G = 8), RMS_NORM with a
  `[128]` weight on every head of Q and K before the NEOX RoPE (as Gemma 4 and Qwen3.8 have), full attention in every
  layer, no V norm, no post norms, no layer scale, no soft cap. These are existing records.
- Router: the MUL_MAT of `ffn_gate_inp` (F32 `[2048, 128]`, converted to Q8_0 at load) reads the normed FFN input that the
  experts read. Gemma 4 puts RMS_NORM, SCALE and MUL by `ffn_gate_inp.scale` in front of it.
- Experts: gate and up are two tensors (`ffn_gate_exps`, `ffn_up_exps`, Q4_K `[2048, 768, 128]` in every layer), read by
  two MUL_MAT_IDs with the same ids and the same input (up first), and the activation is `swiglu_split` of their two
  results (GLU SWIGLU of two contiguous `[768, 8, T]` tensors, the split form the dense models use). Gemma 4 has one
  `ffn_gate_up_exps` and a GEGLU of two views of its result. The down tensor is Q4_K in 24 layers and Q6_K in the other
  24 (`[768, 2048, 128]`; Q6_K is converted to Q8_0 at load, SW-60); the device holds 19.8 GB of weights.
- No per-expert output scale: there is no REPEAT / GET_ROWS / MUL chain, so the scale group of SW-64 does not form. The
  down MUL_MAT_ID is lowered alone, with no table and no CPY, and the MUL by the router weights (`[1, 8, T]` against
  `[2048, 8, T]`) is an ordinary MUL record.

`hparams.expert_weights_scale` is 0 for the family, so `build_moe_ffn` adds no SCALE node after the DIV. Nothing was
patched in llama.cpp. A decode token is one launch of 1,397 records (the same count for a prefill ubatch): per layer 29
records, namely 5 MUL_MAT (q, k, v, o, router; the residual ADD after the output projection is fused into it, SW-42),
4 RMS_NORM (attention norm, Q norm, K norm, FFN norm), 2 ROPE, 2 SET_ROWS_KV, 1 FLASH_ATTN, 1 MOE_ROUTE, 3 MUL_MAT_ID, 1 GLU,
1 MUL (the router weights), 8 ADD (seven for the sum of the experts, one residual) and 1 JOIN; the end of the graph adds 5 (output norm,
the output MUL_MAT, two GET_ROWS of the output rows, one ADD). Tests (no model file needed): `ggml_tesseract_moe*` build the Qwen3 layer next to
the Gemma 4 one (`test-tsr-moe`: T = 1 to 256 against the CPU in the four kind configurations; 65, 70, 130 and 256 tokens are several
tiles, whose router outputs go through the scratch of SW-76), and `ggml_tesseract_graph_qwen3moe_tiny` / `_q4k` run a 2-layer model of the
same graph (`gen_tiny_models.py`: 64 experts, 8 used, head size 128 with G = 8, Q 4 times the hidden size) through klcheck
with the M7 settings (one launch per decode, one per prefill ubatch, KL against the CPU, `tiny.cmake`). Prefill of the real
model runs at the default `-ub 512` since SW-76: on the card a 512-token prompt gives 26.72 tok/s best and 26.60 median, and
513 of 513 rows are bit-identical to the simulator ([klcheck.md](../klcheck.md)).

### 7.10 Gated DeltaNet and the Qwen3.5 / 3.8 layer graph (SW-67 to SW-74)

Status (2026-11-09): the three kinds, their planners, references, the backend lowering and the tests are in. The
Qwen3.8 2B model runs its whole decode graph on the simulator in one launch per token and agrees with the CPU at
the usual klcheck floor (below). No RTL changed. The model ran on the card on 2026-11-09 (the K4 full-feature image; [klcheck.md](../klcheck.md), [performance.md](../performance.md)). The kinds compose existing VU
operations, so they run on any image that has the VU (`CAPS` bit 0, the K4 full-feature image).

**The models.** llama.cpp calls the family `qwen35`. It is hybrid: layer `il` is full attention when
`(il + 1) % full_attention_interval == 0` (interval 4), otherwise Gated DeltaNet ("GDN", the `recurrent_layers`
array of the GGUF). The sizes that matter:

| Model | Layers | GDN / attention | E | S_v | K heads / V heads | Conv channels C | State per layer | Attention |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8 2B (empero-ai distill, Q4_K_M) | 24 (+1 unused MTP block) | 18 / 6 | 2048 | 128 | 16 / 16 | 6144 | 1 MiB | 8 heads, 2 KV heads, D 256 |
| Qwen3.8 27B (ggml-org, Q4_K_M) | 64 | 48 / 16 | 5120 | 128 | 16 / 48 | 10240 | 3 MiB | 24 heads, 4 KV heads, D 256 |

(`d_conv` is 4 in both. C is `2 · H_k · S_v + H_v · S_v`. One sequence of the 27B holds 144 MiB of GDN state and
5.6 MiB of conv history in its 48 GDN layers.) The pinned llama.cpp supports the family (`qwen35`; the larger
MoE family `qwen4exp` is a separate arch). No llama.cpp patch is needed: everything below lowers from the graph
the pinned llama.cpp emits (SW-71, SW-72).

**The GDN layer graph** (one token; names are those of `src/models/qwen35.cpp`, nodes in graph order):

| Step | ggml nodes | Device |
|---|---|---|
| norm | `RMS_NORM` + `MUL` (`attn_norm`) | RMS_NORM with weight (fused, SW-42) |
| projections | `MUL_MAT` qkv (C), z (`attn_gate`, H_v S_v), alpha, beta (H_v each) | MUL_MAT |
| conv state | `GET_ROWS` of `cache_r_l` by `rs_s_copy`, `CONCAT` with the transposed qkv, `SSM_CONV`, `SILU`, a `CPY` of the last `d_conv - 1` columns into `cache_r_l` | CPY ×2 at the CONCAT (the window), one SSM_CONV (SILU and the state CPY folded in), SW-71 |
| q, k | views of the conv output, `RMS_NORM` (eps / n) and `SCALE` by 1 / √S_v (the L2 norm) | RMS_NORM, SCALE |
| gates | `ADD` dt bias, `SOFTPLUS`, `MUL` ssm_a → g; `SIGMOID` of beta | ADD, SOFTPLUS, MUL, UNARY SIGMOID |
| state | `GET_ROWS` of `cache_s_l`, `SCALE` by 0 in place for a new sequence | GET_ROWS, SCALE |
| recurrence | `GATED_DELTA_NET` (scalar gate, K = 1), a `CPY` of the new state into `cache_s_l` | GATED_DELTA_NET, CPY (rows regrouped) |
| output | `RMS_NORM` + `MUL` (ssm_norm), `SILU` of z, `MUL`, `MUL_MAT` ssm_out, residual `ADD` | RMS_NORM, UNARY SILU, MUL, MUL_MAT |
| FFN | norm, gate / up `MUL_MAT`, SWIGLU, down `MUL_MAT`, residual | as for every model |

The attention layers of the family: a joint Q and gate projection (`MUL_MAT` of 2 · H · D outputs, split by
views), per-head `RMS_NORM` + `MUL` of q and k, `ROPE` in IMROPE mode, `FLASH_ATTN_EXT` (D = 256), a SIGMOID
gate on the attention output (`MUL`), the output projection. IMROPE carries four position streams. For text all
three section streams equal the token position (the fourth is 0), which makes IMROPE the NEOX rotation over
`n_dims` (SW-72). `graph_compute` checks the streams of the position input and stops with a message if they
differ (vision positions).

**State layouts.** `cache_s_l` is `H_v` states per cell, state `h` being `S_v` rows of `S_v` f32, row `j` holding
`S[.][j]` (the ggml-cpu layout; one row is `S_v / 16` lines). The recurrence of `ggml_gated_delta_net`, per token
and head, is  `S *= exp(g); delta_j = (v_j - S_j · k) β; S_j += k delta_j; o_j = (S_j · q) / √S_v`  (q and k head
`h mod H_k`). `cache_r_l` holds `d_conv - 1` rows of `C` f32 per cell (time-major rows of channel vectors). This is
not ggml's channel-major order. The caches are in a Tesseract buffer (a device window, SW-72) and only the device
reads them, so the layout is the device's. llama.cpp copies and clears whole cells (sequence copy, clear, state
save and load), which keeps working because those are byte copies of a cell. The conv taps (`ssm_conv1d.weight`,
`[d_conv, C]` tap-fastest in the GGUF) are re-laid out as `d_conv` rows of `C` f32 when the weight is uploaded.

**Kinds** (record words in `tsr/opl.hpp`, planners in `tsr/plan_gdn.hpp`, handlers in `kernels/gdn_kinds.cpp`,
references in `ref/gdn_ref.cpp`). Kind ids are numbered from the top of the free range downwards (0x30 to 0x3F
stay test kinds). All are at numerics revision 1 (`kind_revs.txt`).

| Kind | Id | Words | Tile composition | Numerics |
|---|---|---|---|---|
| SOFTPLUS | 0x2D | dst, dst stride, src, src stride, n, rows | row tile of 5 buffers; 20 VOPs (SW-69) | `softplus_ref` |
| SSM_CONV | 0x2E | dst, dst token stride, x, x token stride, state in, state out, taps, C, d_conv, T, n_seqs, flags (bit 0 SILU) | per sequence and channel tile: LD taps, state rows and token rows; `d_conv` VOP.MUL, `d_conv - 1` VOP.ADD in tap order; optional VOP.SILU; ST the T output rows and the new state rows (SW-68) | `sconv_ref` |
| GATED_DELTA_NET | 0x2F | dst, state out, state in, q / k / v with head and token strides, g and beta with token strides, S_v, H_k, H_v, T, n_seqs, scale | per (sequence, head): LD the state tile, then per token 13 VU ops (below); ST the state tile (SW-67, SW-70) | `gdn_ref` |

GATED_DELTA_NET per head and token: `SOP.EXP` α = exp(g); `VOP.MULC` S = S α; `VOP.MUL` P = S · k (broadcast of k
over the rows) and `RED.SUM` u (one value per row; the fixed 128-partial tree of VU §6.2); `VOP.SUB` and
`VOP.MULC` delta = (v − u) β; `VOP.MULCADD` S = k δ + S (δ_j a per-row scalar); `VOP.MUL` P = S · q and `RED.SUM` w;
`VOP.MULC` o = w · scale. Heads run in order with the next head's state loading while the VU works on the
current one; the next token's vectors load two tokens ahead. One ring block holds two state tiles (the head in flight and the next one), the
product tile and two token slots (3 · S_v · L + 2 (5 L + 2) lines, L = S_v / 16; 3,156 lines for S_v = 128, so
the block is far below half of the planning budget; S_v up to 256 fits). The result tensor is `[S_v H_v, T n_seqs + S_v n_seqs]` rows: the attention
rows, then the new states. DRAM traffic per token and layer is the state read and written once.

**Softplus (SW-69).** The ggml node is `softplus(x) = log(1 + exp(x))`; the CPU computes `x > 20 ? x : logf(1 + expf(x))`,
which loses the relative accuracy of small results (`1 + exp(x)` rounds). `SOFTPLUS` computes the stable form
`max(x, 0) + log1p(exp(-|x|))`. The VU has EXP, RECIP and the multiply-add functions but no log. `log1p(h)` for
`h = exp(-|x|)` in (0, 1] is `2 atanh(s)` with `s = h / (2 + h)` in (0, 1/3], by the odd series to `z^5` (`z = s²`,
coefficients `2 / (2k + 1)`). That is 20 VOPs for any row length. Against float64 the error is below 5 ulp for results
above 1e-3 and below 4.2e-7 relative for every result above 1e-30; against ggml-cpu the absolute difference is below
1.2e-6 (test `fw_gdn_ggml`). A VU `LOG1P` function was not added: the gate vector is H_v values per layer, and one
SOFTPLUS launch is dominated by its launch cost (about 18.5 K core cycles).

**Cost.** *Measured before SW-80, which changed the model* (§4.5): at the default 16 lanes the isolated records below take 178.2 K and 504.9 K cycles
(2B, 27B) and 10.2 K per head and token; at the card image's 8 / 8 / 4 lanes (`TSR_SIM_VU_TIMING=k4`) 261.3 K, 754.2 K and 15.4 K, and the card
takes 15.3 K per head step ([performance.md](../performance.md) "Card against simulator"). The 27B figures further down scale the same way: a decode token
spends 23.5 M cycles in the recurrence at 16 lanes and 35.5 M at 8 lanes, against the 19.7 M of the old model. Isolated records, simulator cycle model, fixed schedule, 4 MiB SPM (`test_fw_gdn_rv <image> bench`; each launch
includes about 18 K cycles of launch overhead, shown by `softplus`): GATED_DELTA_NET 151.6 K cycles for the 2B
(16 heads) and 425.0 K for the 27B (48 heads) at one token; 1.655 M for four tokens and 6.576 M for sixteen, so
8.6 K cycles per head and token in steady state (13 dependent VU ops; each of the two `RED.SUM` takes 16 cycles per row,
2,048 cycles for the 128 rows). SSM_CONV 14.6 K (2B) and 19.0 K (27B), SOFTPLUS 18.5 K.

Whole decode tokens (`TSR_STATS=1`: PERF_CYCLES and the busy cycles of the units, which now include the VU and DMA).
The 27B was not run whole: the free disk (the file-backed simulator DRAM needs about 22 GB) was below the 60 GB
rule. Its row extrapolates the first 4 and first 8 layers of the real GGUF (truncated copies made with `sw/tools/gguf_truncate.py`,
N = 4 and N = 8, the 27B's shapes): one group of 3 Gated DeltaNet and 1 attention layer costs the
difference of the two, and the head (embedding, output projection) the rest of the 4-layer run; 16 groups and
the head make the 64 layers.

| Per decode token (core cycles) | 2B (24 layers, measured) | 27B (64 layers, from 4 and 8 layers) |
|---|---|---|
| Total | 10.51 M | 108.4 M |
| MVM busy | 5.70 M | 73.1 M |
| VU busy | 2.73 M | 22.0 M |
| DMA busy | 2.45 M | 18.4 M |
| Recurrence (isolated record cost × heads × layers) | 2.47 M (24 % of the token) | 19.7 M (18 %) |
| At 200 MHz (the card image's core clock) | 52.6 ms, 19 tokens/s | 542 ms, 1.85 tokens/s |
| At 276.3 MHz (the simulator's core clock) | 38.0 ms | 392 ms |

The busy cycles add up to about the total (2B: 10.9 M against 10.5 M; one group of the 27B: 6.76 M against 6.45 M):
a decode token is a chain of dependent ops, so the VU's time is not hidden behind the MVM's. In a group of four
layers the VU is busy 1.375 M cycles, of which the three recurrences take 1.23 M. At the 27B's 48 GDN layers the
recurrence takes 19.7 M cycles per decode token: 98 ms at 200 MHz, 71 ms at 276.3 MHz. The fused GDN op of the VU design
(8 elements per cycle at 200 MHz) is estimated at 23.8 ms per token, so the composed recurrence is about 4.1 times
slower than the fused one at the same clock, and a 27B token would take about 468 ms with the fused op instead of
542 ms (14 % less). Next to that, the composed kinds need no new hardware and their numerics are the oracle's;
they are the baseline against which a fused op or local state buffers are measured. What would shrink the
baseline in software: the state makes three passes through DRAM per layer (the `GET_ROWS` copy of the cell, the
record's read and write, the `CPY` back; each pass reads and writes it), which a record working on the cache cell in
place would cut to one (the DMA is 17 to 23 % of the token), and a reduction that does not serialise on the
16-cycle `RED.SUM`.

**Results** (simulator; klcheck at the usual floor: mean of the seeds 1 to 3, M7 settings, one launch per decode token;
[klcheck.md](../klcheck.md)):

- Qwen3.8 2B Q4_K_M, 8 prompts × 64 tokens (512 positions): mean KL 2.377e-3 against a floor of 2.240e-3, ratio 1.061,
  top-1 98.05 %. 744 records per launch, 18 Gated DeltaNet and 6 attention layers on the device.
- Long sequence: a 4,127-token prompt prefilled in ubatches of 512 (GATED_DELTA_NET records of 512 tokens) and 16 generated
  tokens, 79 positions (the last 63 prompt positions and the 16 generated): KL 4.250e-3 against 4.298e-3, ratio 0.989, top-1
  94.94 %, lowest cosine 0.9945. The state has been through 4,127 tokens in each of 18 layers. The simulator took 2,752 s
  for the prefill and 1.6 s per decode token at that context.
- The 27B's shapes, first 4 layers: ratio 1.062 (128 positions, top-1 100 %), logits cosine 0.99999 or better.
- `fw_gdn_ggml`: the recurrence against ggml-cpu and float64 up to 512 tokens, the error equal to the CPU's own.

**Not covered.** The whole 27B, a card run, more than one sequence in a ubatch end to end (the kinds and the
lowering take `n_seqs` > 1 and the kind tests cover it; no llama.cpp run does), snapshots and rollback (`K` > 1
state snapshots for multi-token prediction: such a node stays on the CPU), vision positions, and a KDA gate.

---

## 8. One launch per decode token: what llama.cpp's graph needs

The M7 target is that every decode graph produces exactly one Tesseract split. This needs:

1. **R1 — every node after the input layer is supported** (§9.3 supports_op table), including the
   nodes of the reserve graph (prefill T ≤ n_ubatch). A node whose pre-allocated tensor sits in a
   Tesseract buffer must be supported, or the scheduler aborts (`docs/llama-pin.md`, Scheduler: splits, split inputs, debugging).
   That applies to SET_ROWS into the KV cache. Conversely, `GGML_OP_DUP` and SET_ROWS whose dst is
   not a KV tensor stay unsupported, and `supports_op` returns false for nodes without
   `GGML_TENSOR_FLAG_COMPUTE`: every pinned decode graph holds unselected embedding branches from
   `ggml_build_forward_select` (DUP(`inp_mixed_embd`), GET_ROWS, SET_ROWS;
   `src/llama-graph.cpp:2745-2753`), the scheduler's assignment passes ignore the flag
   (`ggml-backend.cpp:1298-1317`), and a supported DUP there would create a second Tesseract split.
   `graph_compute` skips nodes without the flag and makes no launch for a split that has none.
   *As built (W3b, SW-40):* supports_op does not test the flag (test-backend-ops asks before the
   flag is set); the unselected branches are refused by their own rules (DUP, a non-KV SET_ROWS, the
   quantized `tok_embd` GET_ROWS), and `graph_compute` skips flagless nodes.
2. **R2 — token embedding stays on the CPU.** llama.cpp keeps the input layer on the CPU
   (`src/llama-model.cpp:1688-1689`). Its GET_ROWS forms a CPU split before the device split and
   costs no launch. The scheduler copies the n_embd f32 result in with `set_tensor`.
3. **R3 — KV cache on the device (M7):** run without `-nkvo` (`offload_kqv` = true), with
   `-ctk q8_0 -ctv q8_0` and `-fa on`. The pinned llama.cpp refuses a quantized V cache without FA
   (`src/llama-context.cpp:566-569`). M6 runs with `offload_kqv` = false (§10.4).
4. **R4 — Hadamard KV rotation off (SW-16).** With a quantized KV cache and head_dim % 64 = 0,
   llama.cpp rotates Q, K and V and un-rotates the attention output by `MUL_MAT(rot, x)`, where
   `rot` is an F32 graph input (`src/llama-kv-cache.cpp:316-356`, `src/llama-graph.cpp:3228`,
   `:3273`). That src0 is not a weight and cannot become an MVM weight, so the rotations would
   split attention into CPU islands. Device runs and every baseline therefore set
   `LLAMA_ATTN_ROT_DISABLE=1`. The backend warns once if it sees `attn_inp_k_rot`/`attn_inp_v_rot`.
5. **R5 — graph inputs** (`inp_pos`, `kq_mask`, `k_idxs`, `v_idxs`, `inp_out_ids`) are written by
   the scheduler through `set_tensor` before the launch. They are copies, not launches. The builder
   lists the index inputs in the preload table.
6. **R6 — compute buffers fit the device (SW-26).** Compute-buffer chunks are placed in a
   per-channel compute heap between the top of the weights and the bottom of the KV planes (EX
   §11), sized from the reserve graph, plus ch0 [64, 320) MiB and ch1–ch3 [0, 512) MiB; there are
   no fixed 512 MiB windows, and the buffer type's `get_max_size` is at least the largest single
   tensor. The reserved logits chunk then stays on the device: Qwen 152,064 × 512 × 4 B = 297 MiB
   at `-ub 512` (`docs/llama-pin.md`, Other notes for running llama.cpp on Tesseract); Gemma 4 262,144 × 512 × 4 B = 512 MiB (`n_outputs`
   = min(n_ubatch, n_outputs_max), `src/llama-context.cpp:346`), with the softcap's SCALE / TANH /
   SCALE beside it. A chunk that still does not fit is staged and runs in its own launch: correct,
   but it costs one extra launch per token, which TSR_STATS reports as "staged chunks".
   *As built (W3b, SW-46):* a buffer gets one window: ch0 IO, else the low 512 MiB of ch1–3, else
   symmetric WEIGHT-heap reservations of ≥ 64 MiB between the weights and the KV planes;
   `get_max_size` = 512 MiB. Gemma 4 12B at `-ub 512`: 0 staged chunks. Only MUL_MAT is staged: the
   M7 kinds address their operands in place, so they need device-resident buffers (SW-53).
   supports_op refuses an M7 op with an operand in host memory or in a buffer that will be (an ANY
   buffer of leaf data only: llama.cpp's control vectors and LoRA adapters; every buffer under
   `TSR_FORCE_HOST`), which then runs on the CPU (a control vector costs one CPU island per steered
   layer). A compute buffer that gets no window is placed at its first data access, after the
   scheduler split the graph: graph_compute then aborts naming the buffer and the remedies (a smaller
   `-ub` / `-c`, or `TSR_KINDS=m6`). With the KV cache on the device, `TSR_FORCE_HOST` needs `-nkvo`
   (a SET_ROWS into the device cache cannot run; the backend says so before the scheduler stops).
7. **R7 — graph reuse.** llama.cpp reuses the graph while the KV bucket (256 cells) is unchanged
   (`docs/llama-pin.md`, For M7: KV padding and graph reuse). The backend caches the op list by (graph identity, node
   addresses). Per token the host writes the inputs, launches once and reads the logits. A new
   bucket builds one new op list.
8. **R8 — trailing CPU ops are allowed.** Gemma's final logit softcap (SCALE, TANH, SCALE,
   `src/models/gemma4.cpp:434-437`, cap 30) runs on the device: the VU has TANH (VU §6.5).
9. **R9 — no backend sampling** (`-bs`) in M7. Sampling stays on the host until M8
   (`docs/llama-pin.md`, For M8: sampling).
10. **R10 — not supported in M7:** context shift (K-shift RoPE on a q8_0 cache), KV defrag through
    ggml ops, and multi-sequence streams > 1. The backend reports these clearly instead of
    aborting inside the scheduler where it can (an op-list build error naming the feature).
    *As built (W3b):* supports_op logs once, before the scheduler stops on the pre-allocated cache
    tensor, an op other than SET_ROWS writing into the device cache (context shift, defragmentation)
    and a refused SET_ROWS into it, naming the cause: several sequence streams (llama.cpp `-np N`
    without `--kv-unified`: use `--kv-unified` or `-nkvo`), more than 2,048 cells per ubatch (the
    I64 indices are one preload entry of at most 256 lines, so `-ub` > 2048 stops at context creation:
    use `-ub ≤ 2048`; open item: split the record) or another reason (e.g. `TSR_FORCE_HOST`).

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

### 9.1 Transport API (`sw/transport/tsr_transport.h`)

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
// §6.9: the device's capability and CAPS, the image's Basis version, ABI, kinds and revisions
int      tsr_image_versions(const tsr_image * img, tsr_versions * v);
int      tsr_transport_versions(const tsr_transport * t, tsr_versions * v);
uint32_t tsr_basis_version(void);                                  // this host library's Basis version
int      tsr_versions_str(const tsr_versions * v, int verbose, char * buf, size_t n);

// Decoded view of a record (accessors over tsr_record.w[], EX §4.4)
typedef struct tsr_result {
    uint8_t  type, status, err_code;       // r0
    uint32_t seq;                          // r1
    int32_t  ret;                          // r2 (DONE)
    uint32_t err_ticket, err_info, err_w0, err_pidx;   // r3-r6 (DONE, first error)
    uint32_t duration;                     // r7, core cycles (0 for an OK DONE / PONG: EX §4.8's
                                           // minimal reads skip r7, W3a)
} tsr_result;
void tsr_record_decode(const tsr_record * rec, tsr_result * out);
const char * tsr_status_name(uint8_t status);   // "OK", "TILE_ERR", … (EX §4.4)

typedef struct tsr_transport_stats {
    uint64_t launches, launch_wall_ns, csr_reads, csr_writes, bytes_written, bytes_read,
             log_words, log_records_dropped, staged_chunks;
} tsr_transport_stats;
void tsr_transport_get_stats(const tsr_transport * t, tsr_transport_stats * s);
```

**Protocol layer (`tsr_kdev.cpp`, SW-18).** One implementation serves every kind and follows
EX §2.1, §3, §4.8 and §10.2:

- **Open:**
  1. Read and check `ID` (0x5453_5231), the `VERSION` major (2), `CFG*`, `CAPS`, `CFG3`,
     `CTL_ID` ≠ 0, `RV_CFG`, `Q_DEPTH` and `RV_FREQ` ≠ 0 (else "no `clk_rv`").
  2. Clear `ADAPTER_STATUS`.
  3. Read the host counters (`LQ_SEQ`, `LCQ_RD`, `LOG_RD`) and the `LQ_W` shadow from the device:
     the rings count on across processes, only a card reset zeroes them.
  4. Unless `TSR_ATTACH=1`: `SOFT_RESET` and wait `SR_BUSY` = 0 (`TSR_RESET_MS`), which stops a
     control core left running by an earlier process and discards the records and log words it
     left unread (counted in one message, not decoded); then bind (channel bases, `MEM_EN`, EX §9.2).
  5. If DB's configuration is armed (`SERIAL`, `ERR_STOP`, `HUNG_STOP`, `HALT_ON_BOOT`, an enabled
     TBP, a stop: DB keeps them across `SOFT_RESET`, DB §7.3), report it; every wait then reads
     `DBG_STATUS` / `DBG_CFG` at its backoff polls, as with `TSR_TILE_SERIAL=1` or a debugger's
     CSR write.
- **Boot:** per EX §3.3–§3.5: warm restart when VERIFY of every ITCM segment matches, else a cold
  load (`LOAD_ADDR`, `LOAD_DATA` words, receive CRC, VERIFY CRC); `RV_BOOT` = 0; `RV_RUN` = 1; wait
  for HELLO (`TSR_BOOT_MS`) and check the image id, the Basis ABI, the kind table signature, the
  kernel count and the Tesseract Basis version (§6.9).
- **Launch:** EX §4.8: at most `LQ_DEPTH` launches outstanding (host-side credit, no read); write
  the `LQ_W` words that differ from the shadow copy; write `LQ_PUSH`.
- **Wait:**
  - Consume the LCQ (EX §4.8): read `LCQ_WR`, read r0 of each new record (r2–r6 only when needed),
    deliver events to the callback, write `LCQ_RD`.
  - Poll policy: busy-poll while the elapsed time is below the last service time of the same
    kernel (W3a: measured by the host from the push or the previous answer, since the minimal reads
    skip r7 DURATION), then poll every max(10 µs, 1 % of that time) with sleeps; between polls call
    `kdev::idle()` and **release the transport mutex** (DB's in-process debug service needs it).
    The simulated kinds (`sim`, `mock`, `vkernel`) count timeouts and the poll policy in device
    time (MTIME at the nominal 276.3 MHz), `xrt` in host time.
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
- **PLAT_ERR** (EX §8.6): one report per episode (`STATUS.PLAT_ERR` at a backoff poll or a DONE
  PLAT_ERR answer). The episode covers the launches pushed before it was seen; `ADAPTER_STATUS`
  (W1C, which drives `plat_err_i`) is cleared once all of them have answered, so they answer
  PLAT_ERR and later launches answer normally even when launches are pipelined. An adapter error
  raised during the episode extends it to the launches pushed so far.
- **Watermarks** (EX §4.9): a watermark written since the last `SOFT_RESET` is rewritten just
  below the oldest unanswered launch 2^30 launches after its write, so it never wraps onto new
  launches (it would after 2^31).

**kdev** (EX §10.2).

- Methods: `reg_read`, `reg_write`, `mem_read`, `mem_write`, `num_ch`, `ch_bytes`, `ch_addr`,
  `describe`.
- `KREG_ADAPTER_CTRL` and `virtual void idle(uint32_t poll_iter, uint64_t interval_ns)` (EX's
  `advance` hint), with `now_ns()` (the time base), `simulated()`, `has_adapter()` and `sim()`
  (as implemented in W3a, `sw/transport/tsr_kdev.h`):

| kdev | `idle()` and specifics |
|---|---|
| `sim` | in-process simulator behind the core CSR port; `idle()` = `tsr_sim_advance(budget)`; a host read of `LCQ_WR`, `STATUS` or `LOG_WR` first runs the model to the next host-visible event (EX §10.2) |
| `mock` | the `sim` kdev plus the `ap_ctrl_none` adapter register model (channel bases, `MEM_EN`, `ADAPTER_STATUS`, error injection) |
| `vkernel` | the Verilated kernel server with two clocks (P5); `idle()` runs the server's clocks |
| `xrt` | `ap_ctrl_none` compute unit opened through `xrt::ip` with exclusive access; registers through `read_register`/`write_register`; one device-only BO per channel as today; `idle()` sleeps (backoff). *confirm* the BO bank lookup without an `xrt::kernel` (memory topology from `xrt::xclbin`, EX §9.1) |

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
| `TSR_STATS` | as today, plus launches per token, ops per launch, fused ops, staged chunks (DB §8.6) | off |
| `TSR_GDB`, `TSR_GDB_WAIT`, `TSR_DEBUGGER`, `TSR_DUMP_DIR`, `TSR_TILE_SERIAL`, `TSR_PROF*` | DB §9.5 | off |
| `LLAMA_ATTN_ROT_DISABLE` | must be 1 for device runs and baselines (R4) | set by klcheck/numcheck/tests |

### 9.3 `sw/ggml-tesseract`

**M6 (P3.2):** attention stays on the CPU (`offload_kqv` = false, §10.4).

- `graph_compute(split)`:
  1. Build an OPL with one MUL_MAT record per MUL_MAT node, plus JOINs. Two MUL_MATs are
     dependent only if one reads the other's output; host-side hazard analysis over DRAM byte
     ranges covers WAR and WAW.
  2. Stage host-memory chunks (`tsr_lower_staged`, `tsr_act_loc`).
  3. Write the OPL into the OPL cache (an LRU in the 64 MiB OPL region, keyed by content hash).
  4. Launch `graph` and wait.
  5. On DONE STATUS TILE_ERR, map RET−1 to the ggml node and abort with the node name, the error
     code and `ERR_INFO`.
- `supports_op` (MUL_MAT rules) gives one launch per split: 145 (0.5B) and 385 (32B) per token
  (`docs/platform-u200.md` §14.2).
- `tsr-opl.{h,cpp}` holds the builder, hazards, encoding and cache. Tiling is in `plan.hpp`.
- MUL_MAT src0 is resolved by `tsr_resolve_weight`; raw-buffer src0 (`test-backend-ops` puts src0
  in non-WEIGHTS buffers) is converted lazily (`tsr_lazy_*`).

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
  is GENERAL (or a reader's tile takes the GENERAL path, AT §7.3), an f32 copy of the mask per FA
  scale; the buffer type's `get_alloc_size` reserves the bytes for both on those tensors (as built:
  below, SW-48). The FLASH_ATTN record refers to the MSUM through the preload table, so the
  cached op list stays valid while the mask changes per token.
- **KV buffers (SW-17):**
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
| MUL_MAT | the M6 rules (`docs/spec.md` §12) |
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
| GATED_DELTA_NET | F32; scalar gate (`g->ne[0] = 1`), K = 1, S_v a multiple of 16 up to 256, H_v a multiple of H_k, g / beta contiguous with H_v a multiple of 16 when there is more than one token, state contiguous; q / k / v rows with heads and tokens 64-byte spaced (§7.10) |
| CONCAT, SSM_CONV, CPY of a conv state | only as the group of §7.10 (SW-71): CONCAT(state, transposed tokens) along time, SSM_CONV with F32 taps, d_conv 2 … 8, C a multiple of 16, and the state CPY, with the state cache in a Tesseract buffer |
| GET_ROWS, SCALE, CPY on a recurrent-state cache | when the cache is in a Tesseract buffer; a cache in another backend's memory keeps the op on the CPU (SW-72) |
| DUP, nodes without `GGML_TENSOR_FLAG_COMPUTE` | false (R1) |
| MUL_MAT_ID | `MUL_MAT_ID` kind in the image; weights a loadable 3D expert tensor (WEIGHTS buffer, or raw bytes converted at graph_compute, or llama.cpp's unallocated probe leaf), `N` a multiple of 16, x F32 rows of whole lines, ids I32 rows, result of its own (§7.9, SW-62) |
| SOFT_MAX, ARGSORT, GET_ROWS, SUM_ROWS, CLAMP, DIV, REPEAT of the MoE chains | `MOE_ROUTE` (the router chain) or `MUL_MAT_ID` + `CPY` + `MUL` (the scale chain) in the image, and the node's own sources match `build_moe_ffn` (§7.9, SW-64); the fusion pass takes the group |
| everything else | false |

**M7 builder as built (W3b builder, 2026-10-31; `sw/ggml-tesseract/tsr-build.{h,cpp}`, the glue in
`ggml-tesseract.cpp`).**

- *One lowering for supports_op and the builder (SW-20).* `tsr_lower_m7` turns a node into its
  record (§7.2 words); supports_op runs it in probe mode (handles 0, the KV shape from
  `tsr_kv_identify`, no device state) and asks the kind's planner (`plan_kind` on the device's CFG0-3 /
  CAPS values) and `opl::check_args`, so the host refuses exactly what the firmware cannot plan or the
  builder cannot encode. `ggml_tesseract_units` sweeps 2,850 shapes of every kind against independently
  built words (1,234 supported). The 64-byte rule: every row operand's base and row stride are
  multiples of 64 bytes, and a result that is a view must have rows of whole lines (n·4 % 64 = 0):
  the kinds store whole lines, so a row ending inside a line would overwrite the viewed tensor's
  bytes after it (W3b fix: before it, only single-row views were refused). A result of its own may
  be one row ending inside a line (ggml-alloc pads it). KV views must be the ones llama.cpp builds
  with one stream: starting at the plane's cell 0, cells `row_size(E)` and heads `row_size(D)` apart
  (W3b fix: probe mode now refuses what the build mode cannot encode). Operands in host memory are
  refused (below, R6).
- *Flagless nodes (amends R1 / SW-33).* supports_op does not test `GGML_TENSOR_FLAG_COMPUTE`:
  test-backend-ops asks before `ggml_build_forward_expand` sets the flag
  (`tests/test-backend-ops.cpp:1415-1421`), so the rule would refuse every op there. The unselected
  embedding branches are refused anyway (DUP, a non-KV SET_ROWS, the quantized `tok_embd` GET_ROWS),
  and `graph_compute` skips flagless nodes; the tiny and 0.5B decode graphs are one CPU split
  (embedding) and one Tesseract split (`GGML_SCHED_DEBUG=2`).
- *Fusion (pass 3).* MUL_MAT + ADD of a bias row or the residual (the MUL_MAT's addend, stride 0 or
  the residual's row stride) and RMS_NORM + MUL by a weight (row period 1, or dense per-row weights),
  when the first node's only user (ggml's use count over the whole graph, not a view, not an output:
  `ggml_node_has_n_uses`) is the second, every operand is in device memory, and no node in between
  writes the other operand or reads / writes the second node's result (byte ranges of the tensors'
  addresses); the fused record takes the first node's place. Bit-identical to the unfused device
  records (`ggml_tesseract_opl_launch` F: bias, residual, a decode row, 3 rows, per-head rows, per-row
  weights). Qwen2.5-0.5B decode: 168 fused pairs per token (Q / K / V biases, both residuals, both
  norms per layer, the output norm); the last layer's attention residual follows a GET_ROWS and stays
  an ADD record.
- *Dependencies (pass 4)* use `tsr_opl_builder::kind_ranges`: whole lines of every row operand, row
  periods, per-row scalars, RMS_NORM's weight rows, ROPE's table, SET_ROWS_KV writing and FLASH_ATTN
  reading their KV planes on every channel, FLASH_ATTN's f32 mask copy. Preload sources are not
  ranges (loaded before op 0).
- *Preload table (pass 5).* One entry per (source handle, size): a decode list of Qwen2.5 has five
  (positions, K indices, V indices, the MSUM, the last layer's `inp_out_ids`), Gemma 4's iSWA eight
  (the K / V indices and the MSUM per cache); index inputs must be graph leaves.
  FA_NEW (word 21) names the indices of the SET_ROWS_KV that wrote the record's K plane earlier in the
  same graph_compute, as its own entry when that record went into an earlier list, so the chunk order
  (hence the output) does not depend on where the builder cut its lists. A list is cut early at 4096
  records or when an entry does not fit the 256-line preload area; a prefill ubatch of T tokens needs
  about (36 T + 32) / 64 lines for Qwen2.5 (positions, K and V indices, MSUM): T ≤ ~450 in one list,
  at 512 a few lists per layer (open item: a larger preload area). A FLASH_ATTN's own entries, the MSUM
  (⌈(32 + 16 T) / 64⌉ lines) and FA_NEW's K indices (⌈8 T / 64⌉), must fit one fresh list together:
  T ≤ 680. supports_op refuses a FLASH_ATTN of more tokens, which then runs on the CPU (extra splits
  in that prefill, correct: `ggml_tesseract_graph_qwen_tiny_ub1024`, a 692-token ubatch); before the
  W3b fix such a ubatch (T 681–1022) aborted in graph_compute. FA_NEW across a cut reads the K
  indices from DRAM again; the FLASH_ATTN is not their ggml consumer, so ggml-alloc may reuse their
  bytes after the SET_ROWS_KV: graph_compute checks that no node between the two writes them whenever
  a cut separates them (only the V SET_ROWS_KV does in llama.cpp's graphs) and aborts naming it
  otherwise.
- *FA rule-5 refusals (data-dependent).* The mask producer (W3b kv, `tsr-msum.h`) plans every query
  tile of every registered reader and promotes the MSUM to GENERAL (with the f32 copy) when a tile would
  be refused; the builder registers each FLASH_ATTN as a reader on every build and every cached reuse,
  so the executor never refuses a tile for mask data.
- *Plan cache (R7).* A graph's lists are kept under `tsr_graph_key` (node and source addresses, ops,
  flags, shapes, strides, op parameters; not values) salted with a placement epoch (bumped when a
  buffer is freed or a weight moves); a hit re-registers the FLASH_ATTN mask readers, checks their
  words and runs the lists. Builds that stage a MUL_MAT or convert a weight lazily are not kept.
  Qwen2.5-0.5B, 48 decode tokens on the simulator: 49 graph_computes, 49 launches, 46 plan hits.
- *Compute heap (R6).* A LINEAR buffer's window is ch0 IO, else the low 512 MiB of ch1-3 (the channel
  with the most free bytes), else the compute heap (symmetric WEIGHT-heap reservations of ≥ 64 MiB
  between the weights and the KV planes, shared out per channel); `get_max_size` is 512 MiB.
  `get_alloc_size` pads to 64 bytes only where ggml allows it (`[TAG_ALLOC_SIZE_EXPAND]`: leaves,
  views, quantized types, MUL_MAT, FLASH_ATTN_EXT); ggml-alloc pads every block to the 64-byte
  alignment, which holds a single row's whole-line store.
- *Knobs.* `TSR_KINDS=m6` restricts the device to MUL_MAT (M6 runs use the MUL_MAT-only splits; the M6 e2e
  ctests run with it and `TSR_COMPUTE_WINDOWS=io`, the IO-only windows).
- *Versions (§6.9, SW-54).* At open the backend refuses an image whose Tesseract Basis (major, minor) is
  older than its own and builds a kind mask from the booted image's `.tsr_kinds` (`TSR_KINDS=m6`
  intersects it with NOP, JOIN and MUL_MAT); supports_op, the MUL_MAT path included, refuses a kind
  outside the mask ("kind ADD is not in the booted image (Tesseract Basis 0.1.0, image <id>)", at
  `TSR_DEBUG=2`), so a subset image leaves those ops on the CPU; the builder asserts the mask before
  each launch. The description reads "Tesseract Core 1.0 (VU ATT; sim, C=4, …), Tesseract Basis
  0.1.0 (ABI 1, 14 kinds, image <id>)"; `get_proc_address("ggml_backend_tesseract_versions")` returns
  the `tsr_versions` of the opened device. `sw/rt` applies the Basis rule and requires MUL_MAT.
- *Weight placement.* llama.cpp places a weight by a probe op (`src/llama-model-loader.cpp:928`); its
  ROPE probe has every parameter 0, which the planner would refuse, so probe mode plans a ROPE with
  `n_dims` 0 and a freq-factor tensor as a full rotation: Gemma 4's `rope_freqs` lives on the device
  (else every full-attention layer's ROPE would read it from the CPU and cut the split).

**KV buffers and mask summaries as built (W3b kv, 2026-10-31; `sw/ggml-tesseract/tsr-kvbuf.{h,cpp}`,
`tsr-msum.{h,cpp}`, the "W3b kv" hunks of `ggml-tesseract.cpp`; SW-47, SW-48).**

- *KV buffers.* `init_tensor` records each tensor; a buffer of usage ANY whose tensors are **all**
  `cache_<tag>k_l<n>` / `cache_<tag>v_l<n>`, q8_0, 3D contiguous `[E, kv_size, n_stream]` with
  E % 64 = 0, 64 ≤ E ≤ min(K_MAX, 511 · 64), LG < 2^16 and a plane < 2^28 beats is a KV candidate
  (AT §4.6's identification; a buffer with any other tensor keeps the ordinary buffer rules). Its planes are placed at
  the first data access (llama.cpp's clear), not at `init_tensor` (AT §4.6 as amended): one plane per
  (tensor, stream), top-down in the symmetric WEIGHT heap (`alloc_top`, the same offset on every
  channel, 4 KiB aligned, the first tensor highest); when they do not fit, the lazily converted weights
  are dropped and the allocation retried, else the backend aborts suggesting a smaller context or
  `-nkvo`. `set_tensor` / `get_tensor` / `memset_tensor` / `clear` take any byte range and convert in
  windows of whole local groups (8 MiB of plane bytes) with AT's `tsr_kv_set_bytes` /
  `tsr_kv_get_bytes`; a window is read from the device only when the access does not cover all its
  rows. llama.cpp's state save / load is lossless (`ggml_tesseract_kvstate`: device → device logits
  bit-identical; a device state file loaded by a CPU context byte for byte). llama.cpp's size-0 dummy
  buffers of the memory-fit pass have no context; supports_op treats their cache tensors as KV
  tensors (the builder's `kv_probe`).
- *Mask summaries.* A tensor is a mask when its name contains `kq_mask` (llama.cpp's
  `attn_inp_kq_mask` and the scheduler's `<backend>#attn_inp_kq_mask#n` copies), F16,
  `[n_kv, n_q, 1, 1]`, in device memory; supports_op also requires n_kv = K's cells and n_q = Q's
  tokens. `get_alloc_size` reserves, behind the mask, its MSUM (32 + 16 · n_q bytes) and one f32 copy
  (n_q · 64 · ⌈n_kv/16⌉ bytes, twice the F16 mask; 64 MiB for a 512 × 32,768 mask). Each FLASH_ATTN is
  registered as a reader (scale, E, D, G) on every build and every cached reuse; the reader set only
  grows until the mask's shape changes or its buffer is freed / reset, and a new reader makes the
  producer redo its work from its host copy of the mask (the scheduler writes masks before
  `graph_compute`). On each `set_tensor` / `memset_tensor` of a mask the producer writes the MSUM
  (`tsr_msum_build_f16`), plans every query tile of every reader with the shared planner
  (`tsr_fa_plan`) and writes the f32 copy of a scale (`tsr_msum_mask_f32`, fdiv(m, scale)) only when
  the MSUM is GENERAL or a tile takes the GENERAL path. One copy per distinct scale: the first in the
  reservation, further ones as side allocations of the WEIGHT heap (never evicting weights); an
  all-±0 / −inf mask gives one image for every scale. FLASH_ATTN word 13 always names the copy of its
  scale (§7.2), so the cached list stays valid while the mask changes. An FA without a mask gets an
  all-visible MSUM (`nomask`, once per (n_q, n_kv)). Summaries are forgotten on buffer free, reset
  and LINEAR clear; after a partial `set_tensor` or a memset the mask is read back from the device.
- *Data-dependent FA refusals (AT §6.6 rule 5 / 6; SW-44).* When the planner would refuse a tile
  (−3), the producer promotes the MSUM to GENERAL (flags bit0, header word 4 = the reserved copy's
  handle; the rows keep their runs) and writes the copy, so every tile then plans under rule 1 and
  the executor never refuses for mask data; no CPU fallback. Never triggered at U200 parameters (0 of
  257,084 random tiny-SWA prefill tiles at PCAP 9,344; refusals appear only at PCAP ≤ 48 · E/D, which
  `ggml_tesseract_kvbuf` exercises).
- *TSR_STATS* adds "mask summaries N (GENERAL, promoted), f32 mask copies, re-produced for new
  readers". Test hooks: `ggml_backend_tesseract_kv_plane`, `_mask_summary`, `_att_cfg`.

**M7 on the simulator (W3b integration, 2026-10-31).** One launch per decode token and one Tesseract
split per decode graph for Qwen2.5-0.5B Q8_0, Gemma 4 12B Q4_K_M / Q8_0 and Qwen2.5-32B Q4_K_M; KL
within klcheck's criterion; the results and what is still open are in §10.4 (M7 gate, status).
`klcheck` / `tsr-numcheck` `--phase m6` now restrict the device to MUL_MAT (SW-49).

### 9.4 `sw/rt` and `sw/torch`

`tsr_rt::linear` emits one `graph` launch per IO tile, with one MUL_MAT record per N part × K part
(K parts after the first with ACC_DST). The OPL cache holds the op lists. The public API and the
numerics are defined in the header comment of `sw/rt/tsr_rt.h`. The runtime boots the image like the ggml backend
(§6.9): its Basis major and minor must be at least the host's, and it must provide MUL_MAT and JOIN.

- *Token tiling.* A record carries the call's T (one IO tile). The firmware cuts it into tiles of
  min(T_MAX, its scratch ring) tokens (SW-20), so the runtime does not tile. On the K4 image T_MAX is 4: a
  32-token call is eight tiles inside one launch, and the weight stream is read once per tile, not once per
  token. The tests run `tsr_rt` and the PyTorch suite at T_MAX 1, 4 and 8 (`rt_sim`, `rt_sim_tmax4`, `rt_sim_tmax8`,
  `torch_pytest`, `torch_pytest_tmax4`, `torch_pytest_tmax8`); the bits do not depend on T_MAX.
- *Scaled result.* `try_linear_scale` runs MUL_MAT_SCALE (0x1C) when the image has the kind and `CAPS.MVM_LOCAL_SCALE`
  and the weight has one K part. It returns false, with nothing transferred, otherwise; the caller then runs `linear`
  and multiplies on the CPU (the card image `tsr_u200_k4` has no such capability).
- *PyTorch.* `sw/torch` partitions the aten inference graph of a `torch.compile`d model and sends each weight matmul
  (`mm`, `addmm`, `linear`, `matmul`, `mv` with one weight operand) to `tsr_rt` as one launch. Bias, scale factors that do not
  qualify for MUL_MAT_SCALE, activations, norms, attention and embeddings run on the CPU. Weights that arrive as ggml blocks
  (Q8_0, Q4_K) go through `QuantLinear` (SW-81). Lowering whole graphs onto the M7 kinds is not built; §7 describes the kinds and
  `docs/torch-backend.md` §8 what the backend would need.
- *Checks.* `rt_*` (device against an independent model, bit for bit), `torch_pytest*` (the extension against the numpy spec
  model; compiled models against eager with `reference_linears()`), and `sw/torch/tools/hf_check.py` (Qwen2.5-0.5B-Instruct:
  logits of the device against the reference model, the llama.cpp path's oracle and the same run on the card). Results:
  `docs/torch-backend.md` §6.

### 9.5 Tools

| Tool | Function |
|---|---|
| `tsr-selftest` | identity (Tesseract Core capability, CSRs, `CTL_ID`, `RV_FREQ`, `Q_DEPTH`, adapter); the image's and the host's versions and kinds (§6.9; `--version`); host memory tests; load + VERIFY + HELLO + image id + PING round-trip time and pipelined launch rate (EX §4.11); the `selftest` kernel tests with in-process sim shadow; `--traces DIR` replays through `replay` or full launches; `--debug` DMI halt/resume/step smoke (DB's checks); log drain |
| `tsr-trace-gen` | `ops` traces: `--ops m2|m3|vu|at|ctrl`, errors, concurrency, fixed seeds |
| `tsr-trace-replay` | replay on the simulator (full or tile-only), `--schedules N` determinism mode; `--device KIND` replays through a transport |
| `tsr-trace-conv` | converts `traces/model` into `traces/model-v1conv` (§5.6) |
| `tsr-elf` | image id, kernel table, kind table (`kinds`: one line per kind and revision, the signature), `.note.tsr` (`note`: `cap_min` and `basis_version` as versions), sizes, `.tsr_logfmt` dump, `--version` |
| `tsr-log` | decode raw log words (DB §6.1 header) with an ELF |
| `klcheck`, `numcheck` | settings: FA on (SW-22, §10.4); `--fa` records `on`; KV type q8_0/q8_0; `offload_kqv` = false in M6 and true in M7; `LLAMA_ATTN_ROT_DISABLE=1` in every phase (recorded in the `.klc` header and the numcheck run metadata). numcheck's per-node observation (`cb_eval`) makes the scheduler compute one node per `graph_compute`, which is one launch per node, so numcheck needs no device mode (SW-27). W3b (M7): klcheck runs the SWA layers' KV cache as the ring llama-cli and llama-server use (`--swa-full off`, the default; recorded in the `.klc` header, 0 = an older file, full-size), so the long Gemma 4 prompts of the M7 gate wrap it; the `embd` and floor inputs of klcheck and numcheck carry the token-row scale the graph gives token rows only (√n_embd for Gemma 3 / 3n / 4, `inp_scale_rows`), without which the Gemma 4 floor was 8 nats; `klcheck -v` marks each `llama_decode` (the graph tests' launch count per decode). §6.9: a device run records the Tesseract Core capability, the Tesseract Basis version and the kind table signature (`.klc` header, 0 = not recorded; `report.json` `versions`); numcheck's dev run records them with every kind's revision (`meta.json` / `report.json` `versions`) |
| `tsr_bw` | separate M0 bandwidth kernel |

### 9.6 Toolchain installation without sudo (SW-7)

Checked on the build host (Ubuntu 24.04, 2026-10-30): no `riscv*-elf-gcc`, no `clang` binary
(only `libclang` 18), no `ld.lld`, no Spike, no OpenOCD, and a host-only `gdb`. Verilator 5.046 and
GCC 13.3 are present.

| Tool | Plan (user space, under `~/.local/opt`, versions in `docs/design/toolchain.lock`) |
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
| VU/AT references in `tsr_ref_vu.c` / `tsr_ref_at.c` | numpy spec models (as numcheck B1b, `docs/spec.md` §17) | bit-exact on generated vectors; directed RED cases where a subtree sums to −0 (e.g. 0x80C00000 + 0x00800000) at m = 8k + r (VU §6.2) |
| `tsr_rv` | Spike | on the test bus profile (§4.2: RAM at 0x8000_0000, HTIF): riscv-tests rv32ui/um/uc/mi `-p` and riscv-arch-test RV32IMC/Zicsr signatures; random programs (in-house Python generator, placed at 0x8000_0000 on both) compared retire by retire with `spike --log-commits`; `test_ref_fp.c` built for RV32 and run on both (`docs/ctrl-riscv.md` §9.1) |
| Queue model | EX §7–§8 and CC §7–§9 | directed tests through `tsr_sim_push`; invariants I1–I3 asserted inside the model on every run |
| Race detector | — | positive and negative cases; mutated executors (a dropped WAR dep, a dropped ring release) must be reported |
| Firmware kernels | `sw/ctrl/ref` C++ references; the host build | RV build on sim bit-exact with the reference and with the host build; 8 schedules; race `fatal` |
| Backend and executor | ggml CPU | `test-backend-ops` NMSE per supported op (sim); klcheck KL (sim, then card) |
| Cross-transport | sim | identical logits and DRAM outputs on `mock`, `vkernel`, `xrt` for the same image and inputs |
| Launch traces | fresh-boot replay | every `TSR_TRACE` capture in ctest replays bit-exact on a freshly booted simulator (catches DTCM state leaks, SW-4) |

### 10.2 ctest plan

Host-side, simulator and tool ctests:

| Group | Tests |
|---|---|
| **Reference** | `ref_layout`, `ref_fp`, `ref_ggml_fp16`, `ref_ggml_qact`, `ref_ggml_dequant`, `ref_dot`, `ref_ctypes` |
| **Backend, runtime and PyTorch on the transports** | `ggml_tesseract_e2e`, `_tmax8`, `_hosted`, `_hosted_tmax8`, `ggml_tesseract_scalar_nmse`, `ggml_tesseract_backend_ops` (W3a: without the one 2.1 M-token case, which runs as `ggml_tesseract_backend_ops_bigt`, `slow`), `rt_sim`, `rt_sim_tmax8`, `rt_mock`, `rt_mock_tmax8`, `rt_vkernel_k4t1` (opt-in, skipped until P5.3), `torch_pytest`, `torch_pytest_tmax8` (`TSR_WITH_TORCH`), `transport_ggml_e2e_mock`, `_tmax8` (W3a: defined in `sw/ggml-tesseract/CMakeLists.txt`) |
| **Simulator, transport and tools** | `sim_basic` (EX §15 item 2 on the simulator: CSRs, load with VERIFY, LQ / LCQ launch flow, HELLO, PING, BAD_DESC, cancel and stop, `SOFT_RESET`, errors, holds), `sim_mvm` (MVM through the queue), `sim_trace`, `sim_trace_ubsan` (the trace format), `sim_transport` (boot / launch / wait / soft reset), `transport_kdev` (protocol on mock: timeouts, stale core at open, adapter faults), `transport_xrt_noop` (xrt kdev against the no-op shim; `TSR_WITH_XRT`), `ggml_tesseract_units` (OPL builder: hazards, JOINs, encoding, OPL cache, staging plan; fusion is M7), `tools_trace_gen_replay`, `tools_selftest_mock`, `_tmax8`, `_1ch`, `_faults`, `tools_selftest_mock_traces` (trace-set replays through `tsr-selftest`) |
| **Model checks** (FA on, q8_0 KV, rotation off) | `tools_klcheck_selftest`, `tools_klcheck_qwen05b`, `tools_klcheck_kinds_qwen05b`, `tools_numcheck_selftest`, `tools_numcheck_dump_selftest`, `tools_numcheck_dump_qwen05b`, `tools_numcheck_kinds_qwen05b` |

Other ctests:

- `ref_math_<fn>`, including `ref_math_gelu16` (the 65,536 fp16 inputs against the pinned
  `ggml_table_gelu_f16`);
- `ref_vu`, `ref_at`;
- `rv_isa` (the self-checking riscv-tests on `tsr_rv`, and on Spike when it is installed; without
  Spike the Spike runs are reported as not run), `rv_arch` (riscv-arch-test signatures of
  `tsr_rv` equal to Spike's), `rv_lockstep` (the riscv-tests on both, commit logs compared retire
  by retire) and `rv_random` (the last three skipped with a message if Spike is absent);
- `sim_queue`, `sim_race`, `sim_sched_determinism`, `sim_vu`, `sim_at`, `sim_dm`, `sim_dbg`, and
  `sim_w1b_e2e` (the W1b models together on the W1a test firmware: 8 schedules with race `fatal`,
  launch capture with fresh-boot replays, a GDB-style DMI session that leaves the results unchanged);
- `fw_runtime` (R1–R10: boot, HELLO id, PING, BAD_DESC, argument block, 8 outstanding launches,
  R8 back-pressure with a full LCQ, cancel and stop watermarks (an invalid covered descriptor:
  BAD_DESC; a cancel landing in R8: cleared by R9), TRAP record, panic as a TRAP with `mcause` 24,
  log drop and BLOCK with a full ring, R1's QCLEAR after an RV reset)
  and `fw_runtime_stale` (a mutant image must give exactly one `SPM_STALE`);
- `fw_kernels_host` (ASan/UBSan; also MUL_MAT's decode fast path against the general path on twin
  simulators, push for push, §7.3) and `fw_kernels_rv` (per kind, 8 schedules, race `fatal`), with
  `fw_kernels_rv_v1conv` (a sample of `traces/model-v1conv` regrouped into MUL_MAT ops: the recorded outputs
  and push streams), `fw_kernels_rv_mutant_race` / `_mutant_v1` (mutant images must be caught) and
  the `slow` `fw_kernels_rv_v1conv_all` and `fw_kernels_rv_shapes` (0.5B / 32B / Gemma 4 12B shapes);
- `fw_selftest_sim`, `fw_replay_sim` (incl. SERIAL on a stream whose recorded deps are not the
  chain; `slow`: `fw_replay_sim_all`);
- W2a's package tests: `fw_elf`, `fw_elf_asan`, `fw_elf_link`, `fw_elf_image`, `fw_elf_hello`,
  `fw_elf_test_image`, `_nok`, `_hello` (P2.1: `tsr-elf`, `link.ld`, image id, test images; §6.9: the
  note's versions, the kind table and its mutations, HELLO r4 / r6), `fw_elf_kinds_exec_ref` (a subset
  image's kind table); `fw_kind_revs` (§6.9: every kind's output digest on a fixed corpus against
  `sw/ctrl/tests/kind_revs.txt` at its revision; the image declares exactly the kinds with a
  revision; each launch takes its fast or general path); `ggml_tesseract_versions_full`, `_subset`, `_m6`,
  `_old` (§6.9, §9.3: description, versions hook, supports_op by the booted image's kinds, an ADD of two
  MUL_MAT results routed by them, an image of an older Basis refused); `rt_sim_basis_old`,
  `rt_sim_no_mulmat` (§6.9: `tsr_rt` refuses an image of an older Basis and one without MUL_MAT);
  `fw_hdr_host`, `fw_hdr_rv`, `fw_hdr_rv_mutant`, `fw_hdr_constexpr` (P2.3: builders, tickets,
  `group`, `spm_ring` incl. its blocks, deps and joined tickets against W2a's algorithm, host
  backend); `fw_exec_rv`, `fw_exec_host`, `fw_exec_perf` (P2.4: the op list format and the executor; the
  streamed lists' launch-trace digests pinned to W2a's, PERF PUSH_STALL = 0);
- `fw_m6_e2e` (W2a integration: the production image through the loader with VERIFY, HELLO = the
  `tsr-elf` id, PING, a two-layer 0.5B-shaped MUL_MAT list through `graph`, `selftest` ids 1–10,
  `replay` of trace samples, `graph` again after the other kernels; 8 schedules, race `fatal`,
  `TSR_TRACE` capture, identical observables, every captured launch replayed on fresh simulators,
  the graph launches' trace digests pinned to W2a's, PERF PUSH_STALL = 0) and `fw_m6_perf` (§7.8 /
  EX §4.11 numbers; with the `fw_exec` image also §7.8's M7 projection; PUSH_STALL = 0);
- W2b's package tests (P2.6–P2.7; the M7 kinds): `fw_kernels_elem_host` (ASan/UBSan: planner
  invariants, fast vs general path push for push on twin simulators, handler lists against
  `elem_ref`), `fw_kernels_elem_rv` / `_rv_errors` / `_perf` and the `slow` `_rv_shapes`,
  `fw_kernels_elem_ggml` (LLAMA builds); `fw_kernels_normrope_host` / `_rv` / `_rv_prefill` /
  `_rv_refuse` / `_rv_mutant_race` / `_perf` (and `_ggml` in LLAMA builds); `fw_kvfa_plan` (the shared
  chunk planner against W1a's: with `fw_kvfa_kl` the only checks of the planner, which the bit-exact
  references share), `fw_kvfa_rv_<group>` (`slow`: 16K cells; `general` includes GENERAL tiles of
  four chunks, the mask-buffer rotation), `fw_kvfa_host` (incl. the MSUM n_kv and preload-extent
  refusals), `fw_kvfa_kl` (LLAMA builds; NMSE ≤ 10⁻⁹ and row-softmax KL ≤ 10⁻¹² against ggml-cpu, W2b
  fix: near the measured floor so that a one-cell planner error fails), `fw_kvfa_perf`; `fw_opl_m7`,
  `fw_plan_cxx17`, `fw_exec_perf_m7` (the ≤ 350 gate for every short kind and SCALE with a bias,
  §7.8), `fw_exec_perf_ref`. The elem host and RV decode lists include the s-register shadow's clobber
  rule (a SCALE after an SOP.LDS of s12 pushes its MOVI again). Every RV test runs the fixed and 7
  random schedules with the race detector fatal and compares the outputs bit for bit with the kind's
  reference;
- `fw_m7_layer_qwen05b`, `fw_m7_layer_gemma12b` (W2b integration: one whole decoder layer per OPL list —
  Qwen2.5-0.5B layer 0 with its GGUF weights, decode and an 8-token prefill tile; a Gemma 4 12B SWA
  layer shape with random weights — 8 schedules, race `fatal`, every record's output and both KV planes
  bit-exact against the per-kind references composed in op order, no write outside them, identical
  observables), `fw_m7_layer_ggml` (LLAMA builds: the same layers on ggml-cpu, NMSE and row-softmax KL,
  with the CPU's own new K / V cells and with the device's cache; gates from the measurements, W2b fix)
  and `fw_m7_layer_perf` (§7.8: per-kind RISC-V cycles inside the layer, issue loss against a
  near-ideal issuer, U200 / deep / deep-DMA queues, 166.7 / 250 MHz);
- `trace_v2_roundtrip`, `trace_conv_model` (sample of 10 converted model traces);
- `transport_protocol` (EX §15 item 2, one binary for every kind: cold and warm boot, image-id
  mismatch, VERIFY with an injected TCM fault on vkernel, 0–7 args, credit rule, forced
  `LQ_OVERFLOW` and `RING_RD_BAD`, events while waiting, the debug-aware timeout; W3a runs it on
  `sim` and `mock`, `vkernel` only with `TSR_VKERNEL_SERVER` / `TSR_VKERNEL_SOCKET` set, and adds
  the EX §4.11 numbers, TRAP decoding against `addr2line` and the log paths);
- W3a's further tests: `ggml_tesseract_opl_launch`, `_mock` (OPL builder lists with JOIN and
  ACC_DST records launched through the transport), `ggml_tesseract_backend_ops_bigt` (`slow`),
  `tools_selftest_mock_traces_tmax8`, `tools_log_selftest` (`tsr-log` on saved words),
  `tools_trace_replay_device` (`tsr-trace-replay --device` and replays of `TSR_TRACE` captures),
  `tools_klcheck_identity_qwen05b` (label `model`: generate on `sim`, replay the tokens on `sim`
  and `mock`, `compare --expect-identical`);
- `ggml_tesseract_attn` (SET_ROWS + FLASH_ATTN with a KV buffer vs CPU, NMSE). As built (W3b kv,
  `tests/test-tsr-attn.cpp`; also `_tmax8` at `TSR_SIM_TMAX=8` and `_split1` at `TSR_OPL_SPLIT=1`,
  every record in its own list): KV buffers through the ggml API (300
  random set / memset / get operations, state save / load, view resolution, device memory returned at
  free); 36 device MSUMs and their f32 copies against the references; 14 SET_ROWS + FLASH_ATTN_EXT
  cases in llama.cpp's `build_attn` graph shape (D 64 / 128 / 256 / 512, GQA, decode and prefill,
  causal, SWA, a wrapped SWA ring, holes, finite mask values, up to 2,048 cells; W3b fix: a wrapped
  ring of 2,048 cells whose new cell lies in the low run, three chunks that AT §6.6 rule 7 reorders,
  so the bits depend on FA_NEW), 29 runs with later
  steps through the cached op list while the mask changes; each run: the device cache equals the
  history plus `quantize_row_q8_0_ref` of the new rows, the output equals `tsr_ref_fa` on the device's
  own cache, MSUM and copy bit for bit, NMSE vs CPU ≤ 5·10⁻⁴ (≤ 10⁻⁶ for ±0 / −inf masks; measured
  ≤ 1.10·10⁻¹²); supports_op's placement rules on the device (W3b fix: a CPY into a view whose rows
  end inside a line is refused, one with whole-line rows leaves the viewed tensor's other bytes
  alone; an ADD reading a leaf-only buffer, i.e. host memory, is refused). With it `ggml_tesseract_kvbuf` (`tests/test-tsr-kvbuf.cpp`, no device: names and
  identification, placement, conversion against independent `tsr_kv_pack_row` images after random
  partial accesses over 6 shapes, state save / load, 300 random masks' MSUMs and copies, when a copy is
  written, readers, two scales, nomask, rule-5 / 6 promotion at a small PCAP) and
  `ggml_tesseract_kvstate` (`tests/test-tsr-kvstate.cpp`: llama.cpp's state API on the tiny Qwen2 with
  the cache on the device: device → device logits bit-identical, the device's state file read by a CPU
  context byte for byte, sequence state into a cleared context);
- `ggml_tesseract_graph_qwen_tiny` and `ggml_tesseract_graph_gemma_tiny`: exactly one Tesseract
  split (asserted with `GGML_SCHED_DEBUG=2`) and one launch per decode token, KL vs CPU on
  synthetic tiny models (SW-21: a gguf-py script
  generates 2-layer Qwen2 and 4-layer Gemma 4 GGUFs with random weights, SWA + global layers and
  a 64-row vocabulary at build time; no binary is checked in).
  As built (W3b, `sw/ggml-tesseract/tests/tiny.cmake`, `gen_tiny_models.py`, `m7_tests.py`): also
  `ggml_tesseract_graph_qwen_tiny_q4k` (the Qwen2 weights all Q4_K) and, label `model`,
  `ggml_tesseract_graph_qwen05b` (Qwen2.5-0.5B Q8_0, 4 × 33 tokens). Each runs `klcheck run --phase
  m7 -v` on `sim` (4 prompts, two of 251–343 tokens that cross the 256-cell KV bucket and wrap Gemma
  4's 256-cell SWA ring; `-ub 16`, 16 decode steps each) and asserts from the log: one Tesseract split
  per single-token decode graph, one launch per single-token decode (64; 0.5B: 128), one launch per
  prefill ubatch, none outside a decode, TSR_STATS' count; then the embd identity and klcheck's
  criterion against the CPU (Qwen2 as is; Gemma 4 with `--min-kl 3e-3`, because a 4-layer random
  model's 1-ulp floor is degenerate: ratio 4.2 already in M6, where only the bit-exact MUL_MATs run on
  the device; the bounds and injected-error measurements are in `tiny.cmake`). W3b: Qwen2 KL ratio
  0.47 (Q8_0) / 0.99 (Q4_K), Gemma 4 6.9·10⁻⁴ (ratio 4.8), 0.5B 0.89 (top-1 97 %); 1.3–5.5 s per tiny
  test, 23 s for 0.5B. `ggml_tesseract_graph_qwen_tiny_ub1024` (W3b fix): one prompt of 692 tokens
  (four prompts joined) at `-ub 1024`, one prefill ubatch whose FLASH_ATTN (T > 680, SW-43) runs on
  the CPU; prefill launches reported only, 16 decodes one launch each, KL ratio 0.66;
- Gated DeltaNet (§7.10, SW-74): `fw_gdn_rv` / `fw_gdn_rv_errors` (the three kinds on `tsr_rv`, 8 schedules, race
  `fatal`, bit-exact against `gdn_ref.cpp`, written DRAM equal to the output regions; refused records), `fw_gdn_ggml`
  (LLAMA builds: SOFTPLUS against float64 and ggml-cpu, SSM_CONV and GATED_DELTA_NET against ggml-cpu and float64
  up to 512 tokens), `fw_kind_revs` (the three kinds' corpus digests), `ggml_tesseract_units` (the lowering of the
  gate tensors, GATED_DELTA_NET at the 2B and 27B shapes, the conv group and its window records, the regrouped
  state CPY), `ggml_tesseract_graph_qwen35_tiny` and `_s128` (tiny Qwen3.5-style models, `gen_tiny_models.py`: 3 GDN +
  1 attention layer, S_v 32 and 128; one launch per decode token, the embd identity and klcheck's criterion) and `ggml_tesseract_gdn_logits_<model>` (the logits of every step of prompts of 1, 2, 5 and 17 tokens
  against the CPU). `test_fw_gdn_rv <image> bench` prints the cycles of one record per launch (§7.10).
- `ggml_tesseract_backend_ops_<op>` (W3b; SW §10.4 M7 gate item 4 with `ggml_tesseract_backend_ops`):
  test-backend-ops on `sim` per M7 op filter (ADD, MUL, RMS_NORM, SWIGLU, GEGLU, ROPE, GET_ROWS, CPY,
  CONT, TANH, GELU, SCALE, SOFTCAP, RMS_NORM_MUL_ADD, ADD_RMS_NORM, ADD_ADD, MUL_MAT_ADD,
  RMS_NORM_MUL_ROPE, SET_ROWS, ROPE_SET_ROWS, FLASH_ATTN_EXT with q8_0 K / V): every supported case
  passes, a minimum number is supported (0 for SCALE / SOFTCAP, whose cases have 10-float rows, and
  for the KV-cache ops, which test-backend-ops cannot place in a KV buffer), the unsupported cases are
  counted and listed.

### 10.3 tb (pytest) plan

The RTL areas own the RTL benches. The layout of the benches:

| Group | Benches |
|---|---|
| **Core and platform** | `tb/core` (issue-port tile-only replays of `push.bin`, MVM concurrency under queues); `tb/fastreplay` (C++ tile-only replay, then the full-system bench with Ibex and RVFI lockstep); `tb/platform` (`ap_ctrl_none` adapter; regmap / errors / misaligned / reset directed tests); `tb/platform/vkernel` (two clocks); `tb/run_all.sh` |
| **Units** | `tb/dma`, `tb/fp`, `tb/mvm`, `tb/mvmback`, `tb/mvmfront`, `tb/qact`, `tb/spm`, `tb/tech` (unit ports; `tb/spm` and `tb/dma` extended by EX/AT for round robin and banking) |
| **Control core, VU, AT, debug** | `tb/ctrl` (issue path, two clocks), `tb/vu`, `tb/at`, `tb/rv` (riscv-tests from ITCM, signatures against Spike), `tb/dbg` |

### 10.4 Gates

Common baseline settings (SW-22): `-fa on`, `-ctk q8_0 -ctv q8_0`, `LLAMA_ATTN_ROT_DISABLE=1`,
klcheck's other fixed settings (`docs/klcheck.md`, Settings fixed in every phase). `offload_kqv` is false in M6 (attention
on the CPU) and true in M7. The CPU reference and floor runs use the same KV type, FA and rotation
settings.

**M6 gate** (the spec criterion; MVMs are issued by RISC-V kernels;
`offload_kqv` = false, so attention runs on the CPU and the device AT ops are verified by the
selftest, `traces/at` and `ggml_tesseract_attn`, AT §13.6):

1. `tsr-selftest --device xrt` passes every test, including VU, AT and errors, with sim shadow,
   plus replays of `traces/m3`, `traces/vu`, `traces/at`, `traces/ctrl` and
   `traces/model-v1conv`.
2. 0.5B Q8_0: `klcheck run --device xrt` generates; `klcheck run --device sim --tokens` replays the
   same tokens; `klcheck compare --expect-identical` must report every row bit-identical
   (8 prompts × 64 tokens).
3. 32B Q4_K_M: `klcheck all --device xrt` meets the KL criterion (mean KL ≤ 3 × floor, top-1
   ≥ 80 %, `docs/spec.md` §17).
4. GDB smoke on the card (DB): halt, register read, breakpoint in a kernel, step, resume, with the
   launch then completing correctly.
5. Recorded, not gated: tok/s for 0.5B and 32B (M4, before the control core was added: 54.7 and 2.93 tok/s), launches per token,
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
5. Recorded, not gated: DDR utilization U (`docs/spec.md` §14) with the H/G/F/E split for 32B
   and the largest Gemma 4; the Gemma 4 12B KL with the CPU reference built with and without
   `GGML_GELU_FP16` (the residual of GELU16, VU §6.9).

*Status on the simulator (W3b integration, 2026-10-31; the card items wait for the bitstream with the control core).*
Settings `--phase m7` (FA on, q8_0 / q8_0 KV on the device, `LLAMA_ATTN_ROT_DISABLE=1`), `sim` with 4
channels and T_MAX 1; no llama.cpp patch beyond 0001.

| Item | Qwen2.5-0.5B Q8_0 | Gemma 4 12B Q4_K_M | Gemma 4 12B Q8_0 | Qwen2.5-32B Q4_K_M |
|---|---|---|---|---|
| 1: launches per decode token | 1 (504 / 504 decodes; 8 × 64 tokens) | 1 (64 / 64; 2 × 33) | 1 (64 / 64; 2 × 33) | 1 (smoke: 1 prompt × 2 tokens, 1 decode) |
| 1: splits per decode graph (`GGML_SCHED_DEBUG`) | CPU + 1 Tesseract (every listing) | CPU + 1 Tesseract | CPU + 1 Tesseract | CPU + 1 Tesseract (all 8 listings, incl. the reserve graphs) |
| 1: staged chunks (llama-completion `-ub 512 -c 4096`, 64 tokens) | 0 (298.5 MiB compute buffer on the device) | 0 (649.5 MiB) | 0 (649.5 MiB) | 0 in the smoke (klcheck context) |
| 1: prefill | 1 launch per ubatch (prompts of 7–25 tokens) | 1 per ubatch | 1 per ubatch | 1 per ubatch |
| 2 (sim side): a second sim run | bit-identical, 512 / 512 rows; also with `TSR_SIM_SCHED=random:7 TSR_SIM_RACE=fatal` | bit-identical, 66 / 66 rows | — | — |
| 3 (sim, short): KL vs CPU, klcheck's criterion | mean 2.22·10⁻³, floor 2.43·10⁻³, ratio 0.914, top-1 98.05 % (8 × 64) | 3.07·10⁻², floor 4.51·10⁻², ratio 0.680, top-1 96.97 % (2 × 33) | 2.23·10⁻², floor 2.65·10⁻², ratio 0.843, top-1 96.97 % (2 × 33) | (2 rows) mean 4.63·10⁻³, floor 4.80·10⁻³, ratio 0.964, top-1 2 / 2 |
| sim wall time per decode token | 89 ms | 0.82–0.86 s | 1.50–1.75 s | 7.44 s (prefill of 7 tokens 20.4 s, load 28 s) |
| device cycles per decode token (`PERF_CYCLES`, 276.3 MHz) | 2.48 M = 9.0 ms (M6: 2.63 M in 145 launches, attention on the CPU) | ≈ 35 M ≈ 0.13 s (mean over all launches) | ≈ 57 M ≈ 0.21 s (mean over all launches) | 83.5 M = 0.30 s (MVM 81.5 M; one decode by difference of 2- and 3-token runs; 341.7 ms per token on the card, measured before the control core was added) |

Item 4: the `ggml_tesseract_backend_ops*` ctests, `ggml_tesseract_attn` / `_tmax8`, `_kvbuf` and
`_kvstate` pass (`ctest -LE slow`: 278 / 278 after the W3b review fixes, which added `_attn_split1` and `_graph_qwen_tiny_ub1024`; `ggml_tesseract_backend_ops_bigt` passes). Not yet run:
the long-prompt runs of item 3 (2K and 16K tokens; a prefill ubatch of more than ≈ 450 tokens (Qwen2.5)
or ≈ 240 (Gemma 4) is cut into several launches by the 256-line preload area, and a FLASH_ATTN of more
than 680 tokens runs on the CPU, SW-43), the full 32B
klcheck (8 × 64; projected ≈ 72 min on this host: 512 tokens × 7.44 s + ≈ 230 s of prefill + 28 s load, then ≈ 3 min of CPU runs; M6 took 1 h 17 min), Gemma 4 31B (no GGUF), and every card item.

---

## 11. Work breakdown (P1–P6)

Sizes are *estimates*: lines of new or changed code, and ideal days for one implementer.

| Pkg | Content | Depends on | KLOC | Days |
|---|---|---|---|---|
| **P0** | one issue path (EX §12): tile ops are pushed only by firmware, in every layer | — | — | 1 |
| **P1.1** | table source `sw/tables/tsr_tables.py` (EX / CC / DB / VU / AT rows from `docs/design`) + `gen_tables.py` (SV, C, C++, Python, `--check`) | P0 | 0.6 | 2 |
| P1.2 | host CSR port in `sw/sim` (EX §5: `CTRL` / `STATUS`, loader with VERIFY, LQ / LCQ / log rings and watermarks, `SOFT_RESET`, `Q_*`, PERF, `MTIME`) | P1.1 | 1.2 | 4 |
| P1.3 | `tsr_ref_math.c` (incl. GELU16), VU/AT references, numpy models, exhaustive tests, `ref_math_gelu16` | VU, AT docs | 1.6 | 6 |
| P1.4 | VU unit semantics in sim + cycle model | P1.3 | 1.2 | 4 |
| P1.5 | AT unit semantics in sim + cycle model | P1.3 | 1.0 | 4 |
| P1.6 | queue model `tsr_q.c` (dispatcher, checker with `OPERAND` and four tagged extents, queues, `ISSUE_PIDX`, errors, I1–I3) | P1.2 | 1.5 | 5 |
| P1.7 | `tsr_rv` + `tsr_bus` | P1.1 | 2.0 | 6 |
| P1.8 | scheduler, random schedules, fast-forward | P1.6, P1.7 | 0.5 | 2 |
| P1.9 | race detector + `SPM_STALE` | P1.6 | 1.0 | 4 |
| P1.10 | trace library, capture, sim replay, `tsr-trace-conv`, `tsr-trace-gen`, set regeneration | P1.6 | 2.0 | 6 |
| P1.11 | toolchain install script, Spike/riscv-tests/arch-test harness, random-program lockstep | P1.7 | 0.6 | 3 |
| P1.12 | debug-module model `tsr_dm.c` (DB's DMI profile, DB §9.1) | P1.7, DB doc | 0.8 | 3 |
| P1.13 | DB simulator models: stop logic, QSTEP, serial mode, the hard hold, history ring, REC FIFO, TRW records, log records (DB §5, §6.1, §8) | P1.6, P1.12 | 1.2 | 5 |
| **P2.1** | `sw/ctrl` build (ExternalProject, toolchain file), `tsr-elf` | P1.11 | 0.6 | 2 |
| P2.2 | crt0, `link.ld`, runtime R1–R10 (EX §4.7), traps and panic (DB §6.3), log records (DB §6.1) | P2.1, P1.2 | 0.9 | 4 |
| P2.3 | `kernel.hpp`, generated `vu.hpp`/`at.hpp`/`mmio.hpp`, `spm_ring`, host backend | P2.1, P1.6 | 1.2 | 4 |
| P2.4 | `plan.hpp`, the op list format, executor core | P2.3 | 1.2 | 5 |
| P2.5 | M6 kernels: MUL_MAT kind, `selftest`, `replay` | P2.4 (VU/AT selftest parts after P1.4/P1.5) | 1.2 | 4 |
| P2.6 | M7 kinds: ADD/MUL/SCALE/RMS_NORM/GLU/ROPE/GET_ROWS/CPY/UNARY + references | P2.4, P1.4 | 1.5 | 5 |
| P2.7 | M7 SET_ROWS_KV and FLASH_ATTN + references | P2.4, P1.5 | 1.2 | 6 |
| **P3.1** | transport (EX §10): protocol layer, debug-aware wait and recovery ladder, `sim`/`mock` kdevs, `xrt` kdev (`ap_ctrl_none`, compiles; card in P6), `vkernel` kdev port | P2.2, P1.2 | 1.6 | 6 |
| P3.2 | backend M6 (OPL builder MUL_MAT, cache; lazy src0 conversion, staging and `tsr_resolve_weight`, §9.3), `tsr_rt`/torch port | P2.5, P3.1 | 1.0 | 4 |
| P3.3 | tools: `tsr-selftest`, trace tools, `tsr-log`, klcheck/numcheck settings | P3.1, P2.5 | 1.5 | 5 |
| P3.4 | **M6 on sim**: ctest green, 0.5B and 32B klcheck on sim (FA on), `traces/model-launch` | P3.2, P3.3 | — | 2 (+ ≈ 1 day of 32B sim time) |
| P3.5 | backend M7: supports_op set (incl. the DUP / no-COMPUTE rule), KV buffers, MSUM producer and f32 mask copies (AT §7.3), builder (fusion, hazards, preload), compute heap (R6) | P2.6, P2.7, P3.2 | 2.3 | 9 |
| P3.6 | tiny-model generator; **M7 on sim**: one launch per token, `test-backend-ops`, klcheck Qwen and Gemma | P3.5 | 0.3 | 3 (+ sim time) |
| P3.7 | `libtsr_dbg` + `tsr-dbg`: DMI driver (DB §3.5–§3.6), RSP server (§4), monitor and tile commands, dumps (§7.4), the embedded `TSR_GDB` service; riscv-tests `debug/gdbserver.py` subset on sim | P1.12, P2.2, P3.1 | 3.0 | 10 |
| P3.8 | `tsr-prof` `trw` source and the TSR_STATS counters (DB §8.6) | P1.13 | 0.8 | 3 |
| **P4** | RTL (owners EX, VU, AT, CC, DB; sizes in their documents): issue path and queues, VU, AT and SPM banks, Ibex + TCM + CDC, debug module + DMI, `ap_ctrl_none` adapter, unit benches | area docs; references from P1.3 | ≈ 8–12 SV + benches | per area |
| **P5.1** | tile-only fastreplay (`push.bin` into the issue FIFO) | P4 issue path, P1.10 | 0.8 | 3 |
| P5.2 | full-system Verilator bench: Ibex + RVFI lockstep with `tsr_rv`, two clocks, random ratios | P4, P1.7, P2.2 | 1.5 | 6 |
| P5.3 | `vkernel` (two clocks, `ap_ctrl_none`) | P4, P3.1 | 0.6 | 3 |
| P5.4 | riscv-tests / arch-test on RTL | P5.2, P1.11 | 0.3 | 2 |
| P5.5 | 0.5B on `vkernel` bit-identical to sim (2 tokens); trace sets replayed | P5.3, P3.4 | — | 2 (+ run time) |
| P5.6 | Vivado OOC and the k4 link with the control core (≥ 276 MHz, D13 as restated) | P4 | — | 3 (links ≈ 4 h each with swap, `docs/platform-u200.md`) |
| **P6.1** | card bring-up: load, HELLO, PING, `tsr-selftest` | P5.6 | — | 2 |
| P6.2 | trace replays on the card | P6.1 | — | 1 |
| P6.3 | GDB smoke (DB) | P6.1, P3.7 | — | 1 |
| P6.4 | **M6 gate** (§10.4) | P6.1–P6.3, P3.4 | — | 2 |
| P6.5 | **M7 gate** | P6.4, P3.6 | — | 3 |
| P6.6 | performance: U, H/G/F/E, launch costs | P6.4 | — | 2 |

- **Order.** P0 first, then P1.1. P1.2–P1.13 in parallel where their
  dependencies allow. P2 on the simulator. P3.4 (M6 on sim) before the RTL is needed. P4 runs
  in parallel from the area documents onward.
- **Critical path to M6:** P0 → P1.1 → P1.6/P1.7 → P2.2–P2.5 → P3.1–P3.2 → P3.4, then (with P4)
  P5.2 → P5.5 → P5.6 → P6.4; P3.7 (debug tools) must be done before P6.3, which the M6 gate needs.
- **Software totals:** ≈ 35 KLOC in P1–P3, ≈ 135 ideal days (*estimate*).

---

## 12. Interfaces with other areas

| Area | SW assumes from it | SW provides to it |
|---|---|---|
| **EX** (execution model) | the host CSR map (EX §5); the launch protocol (LQ / LCQ, descriptor and record formats, STATUS precedence, runtime contract R1–R10, cancel and stop watermarks, EX §4); image format, 64-bit image id, loader with VERIFY and warm restart (EX §3); `SOFT_RESET` (EX §8.5); unit ids and the opcode table (EX §7); error codes 1–8, 11–13 and the check order (EX §8.1); PERF (EX §6); the `ap_ctrl_none` adapter (EX §9); the transport API (EX §10); the DRAM map (EX §11) | the simulator models of every EX register and ring (deterministic advance); the SDK runtime; the base kernels `graph`, `replay`, `selftest`; the trace format and the trace sets; the table generator |
| **VU** | encodings, checks and numerics of VOP / RED / ROPE / TOPK / SOP / DQROW (VU §5–§6), including GELU16 / GEGLU16 and the RoPE angle recurrence; AT's vector ops on the VU (VU §5.11); the `tsr_ref_math` algorithms with coefficients as bit patterns; accuracy figures to measure; per-op cycle model | `tsr_ref_math.c` and its exhaustive tests; VU semantics in sim; `traces/vu`; kernels and the generated `vu.hpp` builders; host-computed constants (TS, M, 1/n, eps) |
| **AT** | the KV plane layout and its C conversion (AT §4); KVST, ATTQK, ATTPV and the vector ops (AT §5–§6); the normative FA sequence (AT §6.6); the MSUM format (AT §7.3); the `supports_op` contract (AT §2.6); the SET_ROWS_KV / FLASH_ATTN argument layout (AT §12) | SET_ROWS_KV and FLASH_ATTN kernels; KV buffers in the backend; the MSUM producer and f32 mask copies; AT semantics in sim; `traces/at`; `ggml_tesseract_attn` |
| **CC** | Ibex configuration and CSR list; the RISC-V address map and control window (CC §4–§5) including `ISSUE_PIDX`; TCM sizes; vector table and boot at `RV_BOOT` + 0x80; reset sequence and SR sequencer; interrupt map; `MTIME`; `RV_FREQ`; RVFI in simulation builds; F_RV | `link.ld`/crt0 matching the map; `tsr_rv` modelling the configuration (and its test bus profile); Spike harness; riscv-tests on RTL; lockstep co-simulation bench |
| **DB** | debug module behind DMI at CSR 0xC00–0xDFC; halt / resume / step, triggers, `HOLD_ON_HALT`; the GDB RSP server design; tile-level debug semantics; the log record header (DB §6.1); the trap and panic contract (DB §6.3); the debug-aware wait and recovery ladder (DB §4.1, §7.3) | `tsr_transport_csr_read`/`write`; `tsr_dm.c` and the DB simulator models (P1.12, P1.13); `libtsr_dbg` and `tsr-dbg` (P3.7); `tsr-prof` `trw` (P3.8); firmware with DWARF, `__tsr_dbg`, `__tsr_crash` and the launch hook; `riscv-none-elf-gdb` in the toolchain; the log decoder |

---

## 13. Decisions

| Id | Topic | Choice | Alternatives | Rationale |
|---|---|---|---|---|
| SW-1 | RISC-V model | in-house `tsr_rv` interpreter inside `sw/sim`; Spike only as test oracle (on a separate test bus profile at 0x8000_0000) | link Spike as a library; Ibex RTL model as reference; QEMU | Keeps the simulator one deterministic C library with exact co-scheduling against the tile model and MMIO hooks. It also stays usable inside the llama.cpp backend. Spike still anchors ISA correctness through signatures and lockstep random programs, and RVFI lockstep anchors Ibex-specific behaviour (`docs/ctrl-riscv.md` §9.1). |
| SW-2 | Time model | event-driven on core cycles, `fixed` by default, `random:<seed>` for determinism tests, spin/WFI fast-forward | cycle-by-cycle lockstep; untimed (ops complete at push) | Untimed hides every ordering bug. Cycle-by-cycle is slow on 32B shapes. Events with randomized durations expose schedule dependence cheaply, and fast-forward bounds the cost of polling joins. |
| SW-3 | Race detector policy | vector clocks; `fatal` in ctest, `off` in model runs, `report` nightly | always on; hardware detection | The 2–3× cost (*estimate*) matters only on long model runs. Every kernel and executor path is already covered by ctest with `fatal`. |
| SW-4 | Launch state | launches are stateless: results depend only on image, args and DRAM; `SPM_STALE` check; fresh-boot trace replay | traces with SPM and DTCM snapshots; resident state across launches | Makes every per-launch trace self-contained and replayable on any target from a fresh boot. The op list and tables are small enough to reload per launch (≈ 150 KB for 32B, *estimate*). Resident state can come later (M9) with its own trace rules. |
| SW-5 | Trace layout | one directory per launch; single push-ordered `ops/push.bin` with relative tickets; images shared by id | per-unit `ops/u<k>.bin` (spec text); one ELF copy per trace | Replay needs the global push order (a dep must not point to an unpushed ticket); per-unit files lose it, and the per-unit view is a filter. Relative tickets make traces independent of history. Sharing images avoids 290 copies. The spec's trace paragraph (`docs/spec.md` §17) needs the matching edit. |
| SW-6 | Trace sets | generate m2/m3/gen with `tsr-trace-gen` and fixed seeds; convert the 290 model traces as a serial chain (`replay` SERIAL) | convert everything (EX's and CC's drafts) | Generators and seeds exist, and the error expectations come from the simulator. Model traces are large and error-free, and they convert exactly, which gives a strong unit-equivalence check against the recorded results. Adopted by EX §12 and CC §15.3. |
| SW-7 | Toolchain | xPack `riscv-none-elf-gcc` 14.2.0-3 (with GDB) in user space, the only supported compiler; Spike and dtc from source; Clang optional | distro packages (need sudo; no rv32imc multilib, no libstdc++ headers); building GCC from source; Clang only | No sudo is needed, versions are pinned, and the result is reproducible. The xPack bundle includes GDB for D18. Clang alone lacks an rv32 compiler-rt here. |
| SW-8 | Firmware language rules | freestanding C++20, no heap/exceptions/RTTI, `-Os` (hot path `-O2`, SW-34), libgcc only, no libstdc++ headers; no f32 math in firmware (the host precomputes constants) | allow soft-float in kernels; C | Avoids soft-float cost and any risk of a firmware-side fp rounding path (the spec fp rules stay in host-computed constants and in device ops). C++ templates keep the op builders zero-cost (D6). |
| SW-9 | Image identity | EX §3.2: build-time SHA-256 over the `PT_LOAD` segments, first 8 bytes as a little-endian `uint64`, stored in an 8-byte `.tsr_id`, echoed in HELLO r2 / r3 | a 16-byte id (the draft); runtime hashing of TCMs; CRC only | Hashing 128 KiB on Ibex would take ≈ 80 ms (*estimate*). 64 bits tell builds apart; the load CRC and VERIFY prove the transfer. Excluding the not-loaded log strings keeps the id stable across log-text edits. |
| SW-10 | Host ↔ RISC-V protocol | EX's LQ / LCQ protocol (EX-1 … EX-5) with the runtime contract R1–R10; up to `LQ_DEPTH` launches outstanding | the single-slot H2R / R2H mailbox with one outstanding launch (the draft) | EX owns the launch protocol; the queue lets the host pipeline launches with posted writes only, and one channel carries launches, answers, HELLO, TRAP and events. |
| SW-11 | Logging | binary records (format-string address + args) with DB's header (DB §6.1) through EX's log ring; whole-record drop with a counter; runtime filter `LOG_CFG.LEVEL` | text over the LCQ; DRAM log ring; SW's own 2-bit-level header and a LOGCTL message (the draft) | No TCM is spent on strings, and the ring is readable even when queues are wedged, unlike a DRAM ring that needs a working DMA. Whole-record drop keeps the stream parseable. One header for the encoder and the decoders. |
| SW-12 | Graph execution model | host builds a pre-resolved ggml-level op list (OPL), computes the dependencies, and caches it per graph; firmware kernels per kind | host emits tile-op programs for a replay kernel; firmware parses ggml tensors and computes hazards | Static tile programs cannot express per-token KV write rows without register-relative addressing (not in the ISA, D14). Parsing ggml views and hazards in firmware would cost TCM and RISC-V cycles. The host does the ggml work once per graph, and the firmware reads fixed-size args. |
| SW-13 | Data placement in M7 | tensors stay in DRAM between ops; per-op SPM scratch from a ring with release tickets; fusion of bias/residual through MVM ACC and of RMS_NORM × weight | keep intermediates resident in SPM; no fusion | DRAM round trips cost ≈ 2 % of a 32B token (*estimate*) and are simple to make race-free. Fusion through ACC is bit-identical (one commutative f32 add) and removes the most common round trips. SPM residency is an optimisation for after the gate. |
| SW-14 | M6 llama.cpp path | the same `graph` executor with MUL_MAT-only op lists, one launch per split (adopted by EX as EX-29) | a `mul_mat` kernel per MUL_MAT; keeping tile-program lowering with `tsr.oplist_seq` (EX's draft) | One mechanism from M6 to M7, so M7 only adds kinds and supports_op entries. The splits depend only on the MUL_MAT rules of `supports_op`. |
| SW-15 | Index inputs | preload all RISC-V-read graph inputs at launch start with one join: KV write indices, row ids, positions and each KQ mask's MSUM | LD and join per op | A join per SET_ROWS or FLASH_ATTN would stall run-ahead behind every queued DMA op (2–3 × layers per token). One join at the start costs one DMA latency. |
| SW-16 | KV rotation | `LLAMA_ATTN_ROT_DISABLE=1` in device runs and in every baseline; device rotation is a follow-up | device f32 MUL_MAT for graph-input matrices; a Walsh-Hadamard kernel; leave the rotations on the CPU | The rotation matrix is an F32 graph input that MVM cannot use, and leaving it on the CPU breaks one launch per token. Comparing against a CPU with the same setting keeps the KL criterion fair. q8_0 KV gains little from rotation, compared with 4-bit KV. |
| SW-17 | KV buffers (as built: SW-47) | recognise llama.cpp KV buffers by tensor names at first access; AT layout in a KV heap; converting get/set; context shift unsupported in M7 | a ggml-identical byte layout on the device; a custom buffer type | llama.cpp allocates the KV cache through the device buffer type without a usage flag (`docs/llama-pin.md`, KV cache placement). Names are the stable signal. Converting accessors keep state save/load working whatever layout AT picks. |
| SW-18 | Host protocol | one protocol layer over all kdevs implementing EX §10; `sim` is a kdev with `idle()` | a separate sim transport | The exact host code path that runs on the card (bind, load, VERIFY, launch, wait, recovery, log) is then tested on every sim and mock run. |
| SW-19 | Test access to queues | `tsr_sim_push`/`tsr_sim_qctrl` test hooks in the simulator library, never in `tsr_transport` | no hooks (every test needs firmware); expose to the host | Lets directed queue tests and tile-only sim replays run without firmware, while D15 still holds for every real device path. |
| SW-20 | Planning code | `plan.hpp` freestanding, compiled for host and firmware | separate host and firmware implementations | `supports_op` must reject exactly what firmware cannot plan; one source makes that true by construction. |
| SW-21 | Gemma test models | synthetic tiny Gemma 4 and Qwen2 GGUFs generated by a gguf-py script at build time; real 12B (on disk) and 31B (when available) for gates | only real models; checked-in binaries | The smallest real dense Gemma 4 is 12B, too slow for ctest on sim. Generated models cover SWA/global layers, per-head norms and GEGLU in seconds without binaries in git. |
| SW-22 | Baselines | FA on, q8_0/q8_0 KV, rotation off for device and CPU runs from M6 on; KV on host (`offload_kqv` = false) in M6, on device in M7 | keep FA off until M7 | D17 requires FA on. Fixing the KV type and rotation already in M6 keeps baselines constant from M6 to M7, so the M7 delta measures the device attention only. |
| SW-23 | Approximate-math verification | exhaustive 2^32 bit-exact check against an independent numpy model, ulp bounds against double, RTL exhaustive opt-in; GELU16 enumerated over all 65,536 fp16 inputs against the pinned ggml table | random sampling only; MPFR | Unary f32 functions are fully enumerable, which leaves no gap for coding errors. The numpy model is written from the VU text and does not share code. Double-precision references are enough for f32 ulp classification. |
| SW-24 | Launch watchdog | 60 s default, counted only while the RISC-V is not halted and no tile stop is active, off in SERIAL mode; the mutex is released between polls; on timeout DB's ladder: dump, `LQ_CANCEL`, then `SOFT_RESET` + warm restart | an unconditional timeout with dump and reset (the draft); no timeout | A wedged queue or a parked hart must not hang llama.cpp, but a GDB user stopped at a breakpoint inside llama.cpp must not have the device reset under them (D18). The dump preserves the post-mortem before any destructive step. |
| SW-25 | Kernel test builds | host build (ASan/UBSan) plus RV build on the simulator, the RV build being the reference | RV only; host only | The host build finds logic bugs in milliseconds with sanitizers; only the RV build proves control flow, MMIO timing behaviour and ABI. |
| SW-26 | Compute buffers (amended by SW-46) | a per-channel compute heap between the weights and the KV planes, sized from the reserve graph, plus ch0 IO and ch1–ch3 [0, 512) MiB; `get_max_size` ≥ the largest tensor | fixed IO windows (ch0 IO + ch1–ch3 low 512 MiB, the draft); ch0 IO only | Gemma 4's 512 MiB logits chunk at `-ub 512` fills a whole fixed window; a heap sized from the reserve graph keeps every chunk on the device (one launch per token) and spreads activation DMA over the channels. |
| SW-27 | numcheck launches | rely on per-node observation, which gives one launch per node | a special per-op launch mode in the backend | llama.cpp's `cb_eval` already splits the graph per observed node; no backend mode means no second code path to keep correct. |
| SW-28 | Card selftest | `selftest` kernel with in-process sim shadow comparison | host-written tile programs (forbidden by D15); fixed expected files | D15 requires kernels. Running the same image on the simulator gives exact expectations for any test parameters without stored golden files. |
| SW-29 | Op-list size | ≤ 4096 ops per launch (32 KiB of DTCM tables); larger graphs are split by the builder | unbounded with streamed tables | Decode graphs need ≈ 1,300–2,000 ops (*estimate*). A fixed cap keeps DTCM use static and the executor simple. |
| SW-30 | Timing independence | firmware never lets a timing-dependent read influence pushes, deps, stores or records (only waits); the ring allocator decides deps by count; `ISSUE_PIDX` and `ISSUED_SEQ` are exact | allow opportunistic decisions | This is what makes programs valid in the sense of §4.7, so card logits can be bit-identical to the simulator. |
| SW-31 | GELU in llama.cpp graphs | GGML GELU and GEGLU map to the VU's GELU16 / GEGLU16 (the CPU's fp16-table form); the exact tanh GELU stays for M8 / PyTorch | exact tanh GELU everywhere (the draft's default) | The pinned ggml CPU defines `GGML_GELU_FP16` (`vec.h:46`) and rounds x and the result to fp16 (relative error up to 3.9·10⁻³); matching that form removes an fp16-level difference on every GEGLU element of every Gemma 4 layer. |
| SW-32 | Mask summaries (amended by SW-48) | the backend writes the MSUM (and, for GENERAL masks, an f32 copy) on `set_tensor` of tensors named `kq_mask`, reserving the bytes through `get_alloc_size`; the FA record refers to it through the preload table | the device reads F16 masks; per-token op-list patching | The VU has no f16; the summary keeps the cached op list valid per token and lets ATTQK skip masked bandwidth (AT AT-14). |
| SW-33 | Unselected graph branches (flag rule amended by SW-40) | `GGML_OP_DUP`, non-KV SET_ROWS and nodes without `GGML_TENSOR_FLAG_COMPUTE` are unsupported; `graph_compute` skips flagless nodes and makes no launch for an empty split | rely on "everything else false" | `ggml_build_forward_select` leaves unselected embedding branches in every decode graph, and the scheduler ignores the flag when it assigns backends; a supported DUP would add a second split and launch per token. |
| SW-34 | Firmware optimisation flags (2026-10-30, executor optimisation; amends SW-8) | `-Os` for the image; the executor and kind sources (`kernels/graph.cpp`, `kernels/k_*.cpp`) add `-O2` (§6.1) | `-Os` everywhere; `-O2` everywhere; `-O3` / `-O2 -fpeel-loops` on the hot files | Measured (§6.1): −1 % per MUL_MAT record, −5 to −7 % per JOIN / NOP record (the cheap records that bound M7) and −23 % RISC-V time per 0.5B layer (the launch's fixed part) for +2.2 KiB of ITCM; the whole image at `-O2` adds 5.8 KiB for nothing per record, `-O3` / peeling cost more ITCM for no gain on the optimised executor. Keeps the ITCM headroom for M7's kinds, which join the hot set by file name. |
| SW-35 | Executor hot path (2026-10-30, executor optimisation; §7.3 "Hot path", §7.8) | records read in place through the line-buffered alias after the op's control-window accesses (MARK before the record is read: in PROF launches a malformed record has its marker and a streamed list's chunk LD follows the marker of the record before which it is pushed; otherwise the executor's own pushes are W2a's, and every launch trace of `fw_m6_e2e`, `fw_exec_rv` and `fw_kernels_rv` is byte-identical to W2a's); a streamed list's chunk LD where W2a pushed it (after the record's check, before its handler); a decode fast path for T = 1 MUL_MAT records (leaf function, one credit read per op when it covers the pushes, `spm_ring::alloc_dma_first`); planning stays in the firmware (`plan.hpp`, shifts, limits per launch); `spm_ring` O(1) with a 128-block table; the MUL_MAT push stream unchanged, pinned by trace digests (§7.3) | move MUL_MAT planning into the OPL record (host-planned words); hand-written assembly; a per-push credit cache; inlining the fast path into the loop; the chunk LD after the handler (the first version: one record later than W2a, which changed later deps and a failing chunk load's outcome) | Measured (§7.8), as spin-free executor work with deep queues: 1,518 → 387 RISC-V cycles per 0.5B decode record on Ibex maxperf (1,966 → 485 on small); with U200 queue depths 401, by W2a's method 437 (both above 400). The M6 outcome: the MVM no longer idles on issue. Host planning would save ≈ 35 of ≈ 280 instructions and the firmware would still check the block geometry; the credit cache and inlining measured slower (more live registers); assembly is not maintainable. |
| SW-36 | The one-tile block (2026-10-31, W2b integration; size W2b fix; §7.3, `plan.hpp`) | one fixed SPM block below the preload area (1,024 lines on the U200; `one_block_lines`) for every single-tile record of a DMA-bracketed kind (ADD, MUL, SCALE, RMS_NORM, GLU, ROPE, UNARY; every tile of GET_ROWS / CPY; every SET_ROWS_KV run), in the fast and the general path alike; MUL_MAT and FLASH_ATTN keep the ring, whose budget shrinks by the block (60,144 lines); GLU's decode rows larger than the block take a ring block in the fast path (`alloc_dma_first`); `plan.hpp` pins MUL_MAT's prefill T-tiles of the model shapes by static_assert | `spm_ring::alloc_dma_first` per record (the packages' form); merging consecutive DMA-first blocks into one ring record; the block at the ring's bottom; 4,096 lines (the integration's: every decode tile of Qwen2.5 0.5B–72B and Gemma 4, but MUL_MAT's prefill T-tile of every Qwen2.5 output head drops from 3 to 2, +50 % head weight traffic in an all-logits prefill); 1,536 lines (also Qwen2.5 1.5B / 3B GLU, a 68-line margin on 32B's head) | The DMA queue runs one op after another, so DMA-bracketed tiles are serialised whatever their SPM lines: the fixed block needs no dep, no record and no release and costs no concurrency; it saved 60–80 RISC-V cycles per short record (the ring's table was full of short blocks, so every allocation joined and dropped its oldest) and the ring code left every fast path; at the top the M6 lists' ring blocks keep their lines (the M6 trace digests are unchanged). 1,024 lines keep MUL_MAT's tiling of Qwen2.5 0.5B–32B as before W2b (72B's head: 2) at the cost of GLU's ring block on larger models (about 130 RISC-V cycles once per layer whose device time is milliseconds; the general path would cost 1,250) |
| SW-37 | M7 dispatch (2026-10-31, W2b integration; §7.3) | the kind table holds a fast entry per short kind (`kind_<name>_fast`, a leaf in its common path, KIND_NOT_FAST when it does not take the record, weak "not taken" stubs); the executor calls it with (args, kflags, d) and falls back to the handler out of line, re-reading word0 / word1; `[[unlikely]]` on the fast paths' refusal tests | the handler with the fast path inlined (the packages' form); inlining the fast paths into `op_loop` | −10 to −25 RISC-V cycles per short record (the handler's prologue saved up to 12 registers, the refusal tests were laid out as fall-through); `op_loop` keeps nothing live across the call, so MUL_MAT's path is unchanged within ±2 cycles; inlining would grow `op_loop`'s register pressure for every kind |
| SW-38 | ITCM budget (2026-10-31, W2b integration; amends §7.8's 48 KiB; the figure is amended again by SW-75) | 80 KiB for the production image (69,732 B with the M7 kinds; M8 / M9 keep 48 KiB of the 128 KiB ITCM, CC-3); test images are stamped against the whole ITCM | keep 48 KiB (would need ≈ 22 KB out of the M7 kinds: shared fast-path bodies cost 22–29 RISC-V cycles per record, `-Os` fast paths more); 96 KiB | The per-kind leaf fast paths are what meets the issue-rate requirement; CC-3 sized the ITCM for "M7's graph executor, two model families, M8 MoE, sampling and debug formatting" |
| SW-39 | Staged-word reuse and hot entries (2026-10-31, W2b fix; §7.3, `k_fast.hpp`) | `kf::push_s<SKIP, TICKET>`: a fast path's push without the `ISSUE_W` stores of words the previous push of its record staged with the same value (CC §5.1 keeps staged words; WMASK unchanged, so the pushed words are the op's) and without unused `ISSUE_TICKET` reads; host builds check every SKIP claim; RMS_NORM / ROPE straight-line hot entries for their common decode record that tail-call the general fast form; ADD / MUL / GLU wait for one credit read covering their pushes; `op_loop` without MARK / SERIAL tests for plain in-place lists; the `fw_exec_perf_m7 --strict` gate holds every short kind (and SCALE with a bias) to 350 | regression ceilings for RMS_NORM / ROPE (the integration's I6); precomputed ROPE words in the record (an OPL change); a software shadow of the staged words across records; one fast entry for every RMS_NORM shape (+60 cycles: register pressure) | W2b's issue-rate requirement (≤ 350 RISC-V cycles of work per short decode record) without an OPL change: RMS_NORM 395 → 324, ROPE 420 → 325, SCALE with a bias 380 → 336; the SKIP claims are static per call site and checked on the host, the RISC-V tests check the results bit for bit |
| SW-40 | Flagless nodes (2026-10-31, W3b; amends §8 R1 and SW-33) | supports_op does not test `GGML_TENSOR_FLAG_COMPUTE`; DUP, a SET_ROWS whose destination is not a KV-cache tensor and the quantized `tok_embd` GET_ROWS are refused by their own rules; `graph_compute` skips flagless nodes and makes no launch for a split without computed nodes | the flag rule of R1 / SW-33 | test-backend-ops asks supports_op before `ggml_build_forward_expand` sets the flag (`tests/test-backend-ops.cpp:1415-1421`), so the rule would refuse every op there and the M7 gate's test-backend-ops item would test nothing. The unselected branches are refused anyway: the Qwen2.5-0.5B and Gemma 4 12B decode graphs are one CPU split (the embedding) and one Tesseract split (`GGML_SCHED_DEBUG`) |
| SW-41 | supports_op of the M7 kinds (2026-10-31, W3b; SW-20 as built) | supports_op and the builder share one lowering (`tsr_lower_m7`): probe mode (handles 0, the KV shape from `tsr_kv_identify`, no device state), then `opl::check_args` and the kind's planner (`plan_kind` on the device's CFG0-3 and CAPS). The 64-byte rule: every row operand's base and row stride are multiples of 64 bytes, and a result that is a view has rows of whole lines (the kinds store whole lines; a result of its own may be one row ending inside a line: ggml-alloc pads blocks to 64 bytes; W3b fix: multi-row views were accepted and their neighbours' bytes overwritten). KV views as llama.cpp builds them with one stream (cell 0; cells `row_size(E)`, heads `row_size(D)` apart) in probe and build mode alike; operands in host memory refused (SW-53). llama.cpp's weight-placement probe of a ROPE (every parameter 0, `src/llama-model-loader.cpp:998-1008`) with a freq-factor tensor is planned as a full rotation | a separate supports_op table; refusing the probe | The host refuses exactly what the firmware cannot plan or the builder cannot encode, by construction (2,850 shapes in `ggml_tesseract_units`, 1,234 supported, 0 disagreements). Refusing the probe left Gemma 4's `rope_freqs` on the CPU and cut each token into 3 launches |
| SW-42 | Fusion conditions (2026-10-31, W3b; §9.3 pass 3, SW-13) | MUL_MAT + ADD (a bias row: addend stride 0; the residual: its row stride) and RMS_NORM + MUL (a weight of row period 1 or dense per-row weights) when the first node's ggml use count over the whole graph is 1, it is not a view or an output, every operand is in device memory, and no node in between writes the other operand or reads / writes the result (byte ranges); the fused record takes the first node's place | fusing by adjacency only; use counts within the split | Bit-identical to the unfused device records (`ggml_tesseract_opl_launch` F); whole-graph use counts keep consumers in other splits and observed nodes safe. Qwen2.5-0.5B: 168 fused pairs per decode token; Gemma 4 12B: 289 per token |
| SW-43 | List cuts and FA_NEW (2026-10-31, W3b; §7.2 word 21, AT §12) | A split is one list unless it reaches 4,096 records or a preload entry does not fit the 256-line area; FA_NEW names the indices of the SET_ROWS_KV that wrote the FA's K plane earlier in the same `graph_compute`, added as a preload entry of the FA's own list when the lists were cut between them | cut per layer; FA_NEW only within one list | One launch per decode token; the chunk order, hence the output, does not depend on the cuts. Evidence (W3b fix): `ggml_tesseract_attn`'s wrapped ring of 2,048 cells with the new cell in the low run and three chunks (AT §6.6 rule 7 reorders them) is bit-exact to `tsr_ref_fa`, also as `ggml_tesseract_attn_split1` (`TSR_OPL_SPLIT=1`: FA_NEW in the FA's own list); a wrong FA_NEW count or FA_NEW only within one list changes 387–557 of 1,024 outputs. (The earlier evidence, Qwen2.5-0.5B with `TSR_OPL_SPLIT=7` bit-identical to one launch, cannot show it: in causal decode the new cell is in the last chunk already.) Prefill ubatches above ≈ 450 tokens (Qwen2.5) or ≈ 240 (Gemma 4 iSWA) need more than 256 preload lines and are cut (open item: a larger preload area); a FLASH_ATTN of T > 680 tokens (MSUM and FA_NEW indices together over 256 lines) is refused by supports_op and runs on the CPU (W3b fix; it aborted in graph_compute for T 681–1022) |
| SW-44 | Data-dependent FA refusals (2026-10-31, W3b; AT §6.6 rules 5 / 6) | The mask producer plans every query tile of every registered reader at `set_tensor` (and on a new reader); a tile the planner would refuse promotes the MSUM to GENERAL with the f32 copy, so the executor never refuses for mask data | a CPU fallback per launch; pre-planning FA in the builder; refusing in supports_op (it cannot see mask data) | Keeps one launch per token; the GENERAL path is implemented by the firmware and `tsr_ref_fa`, and the simulator reads the same MSUM, so it stays the bit-exact oracle. Never triggered at U200 parameters (0 of 257,084 random tiny-SWA prefill tiles at PCAP 9,344); exercised at a small PCAP in `ggml_tesseract_kvbuf` |
| SW-45 | Plan cache (2026-10-31, W3b; §8 R7) | A graph's launched lists are kept under `tsr_graph_key` (node and source addresses, ops, flags, types, shapes, strides, op parameters; not values) salted with a placement epoch (bumped on a buffer free, a weight release, a lazy eviction); a hit re-registers the FLASH_ATTN mask readers and is rebuilt if their words moved; builds that stage a MUL_MAT or convert a weight lazily are not kept; LRU of 16 graphs | key by the ggml_cgraph pointer; no cache | A decode token reruns its list and its inputs change only through the preload table: Qwen2.5-0.5B 8 × 64 tokens: 512 `graph_compute` calls, 6 builds, 506 hits |
| SW-46 | Compute windows (2026-10-31, W3b; amends SW-26 and §8 R6) | A LINEAR buffer gets one window in one channel: ch0 IO, else the low 512 MiB of ch1-3 (the channel with the most free bytes), else symmetric WEIGHT-heap reservations of ≥ 64 MiB between the weights and the KV planes, shared out per channel; `get_max_size` = 512 MiB; `get_alloc_size` pads to 64 bytes only where ggml allows it (`[TAG_ALLOC_SIZE_EXPAND]`: leaves, views, quantized types, MUL_MAT, FLASH_ATTN_EXT) | a heap sized from the reserve graph (SW-26's text); padding every tensor | Reservations from the WEIGHT heap need no reserve-graph pass; the logits chunks (Qwen2.5 297 MiB, Gemma 4 512 MiB at `-ub 512`) stay on the device: staged chunks 0 for Qwen2.5-0.5B and Gemma 4 12B at `-ub 512 -c 4096`. Padding every tensor trips ggml's `get_alloc_size` assertion (CONT in test-backend-ops) |
| SW-47 | KV buffers as built (2026-10-31, W3b; SW-17, AT §4.6) | Recognised from the `init_tensor` records (every tensor `cache_<tag>k_l<n>` / `cache_<tag>v_l<n>`, q8_0, 3D contiguous, E % 64 = 0, 64 ≤ E ≤ min(K_MAX, 511 · 64), LG < 2^16, plane < 2^28 beats); planes placed at the first data access (the clear), top-down in the symmetric WEIGHT heap, one per (tensor, stream), 4 KiB aligned; if they do not fit, lazy weights are dropped and the allocation retried, else abort; get / set / memset / clear convert any byte range in windows of whole local groups (8 MiB) with AT's reference, reading a window only when the access does not cover its rows | layout fixed at `init_tensor` (AT §4.6's text); whole-tensor conversion | ggml decides a buffer's usage after `init_tensor`, as for every other storage mode; top-down planes stay clear of the bottom-up weights and the compute-heap reservations; state files are bit-identical to the CPU's (`ggml_tesseract_kvstate`) |
| SW-48 | Mask readers and f32 copies (2026-10-31, W3b; amends SW-32, §7.2 word 13, AT §7.3) | Masks are tensors named `*kq_mask*`, F16 `[n_kv, n_q, 1, 1]`; every FLASH_ATTN registers as a reader (scale, E, D, G) on every build and cached reuse; the reader set only grows until the shape changes or the buffer is freed / reset; one f32 copy per distinct scale (the first in the `get_alloc_size` reservation, further ones as side allocations of the WEIGHT heap), written only when the MSUM is GENERAL or the planner puts a reader's tile on the GENERAL path; word 13 always names the copy of the record's scale (HANDLE_NONE only without a mask, whose FA gets an all-visible `nomask` MSUM); summaries are forgotten on free, reset and LINEAR clear | copies only for GENERAL masks (the draft); per-token op-list patching | The scales are known only from the graph; a growing reader set keeps copy handles stable, so cached lists stay valid while the mask changes per token; rule-3 GENERAL tiles (prefill rows with two runs, holes, a wrapped SWA ring) need the copy although the MSUM is not GENERAL. Every llama.cpp mask is ±0 / −inf, so decode writes no copy (0 copies in the 0.5B and 12B runs) |
| SW-49 | Device kinds per phase (2026-10-31, W3b) | `TSR_KINDS=m6` restricts the device to MUL_MAT; `klcheck run` / `tsr-numcheck run` set it to m6 for `--phase m6` and keep the caller's value for `--phase m7` (unset: every kind; recorded in report.json `tsr_kinds` and meta.json `info.TSR_KINDS`); the M6 e2e ctests run with `TSR_KINDS=m6 TSR_COMPUTE_WINDOWS=io` (the IO-only windows); `TSR_OPL_SPLIT=n` cuts lists for tests | M7 kinds in M6 runs | SW-14: M6 uses the MUL_MAT-only splits (145 launches per Qwen2.5-0.5B token); with the M7 kinds and `offload_kqv` false the M6 phase would be a hybrid of neither milestone |
| SW-50 | M7 graph tests (2026-10-31, W3b; §10.2, SW-21) | Tiny Qwen2 (Q8_0 and pure Q4_K, no Q6_K) and Gemma 4 (Q8_0) from `gen_tiny_models.py`; `m7_tests.py graph` counts launches per `llama_decode` from `klcheck -v` markers and TSR_DEBUG launch lines, splits from `GGML_SCHED_DEBUG=2`; one Tesseract split per single-token decode graph, one launch per single-token decode, one per prefill ubatch; KL with klcheck's criterion against the mean of the floor seeds 1 to 3 (tiny Gemma 4 with `--min-kl 3e-3`), top-1 ≥ 0.95 for the tiny models; the device run drops the shell's `TSR_KINDS` | a backend hook for launch counts; a Q4_K_M-like mix; the spec's top-1 0.8 for tiny models | The Q4_K_M mix converts Q6_K to Q8_0 on the device, so a tiny model's floor cannot absorb it (ratio 87 in M6); a 4-layer random model's 1-ulp floor is degenerate (ratio 4.2 already in M6); 3e-3 lies below the injected sliding-window, RoPE-base and eps errors, and 0.95 catches the sliding-window error that 0.8 would pass |
| SW-51 | klcheck settings for SWA models (2026-10-31, W3b; §9.5) | `--swa-full off` (default): the SWA ring llama-cli and llama-server use, recorded in a spare header word (0 = older file, read as full-size); the `embd` and floor rows carry llama.cpp's token-row scale (√n_embd for Gemma 3 / 3n / 4), in klcheck and numcheck | `llama_context_default_params`' full-size SWA cache; unscaled embd rows | Long Gemma 4 prompts must wrap the ring (M7 gate item 3); without the scale the Gemma 4 floor was 8.2 nats with top-1 0 %, with it the embd path equals the token path bit for bit |
| SW-52 | test-backend-ops coverage of the M7 kinds (2026-10-31, W3b; §10.4 item 4) | One ctest per op filter (`ggml_tesseract_backend_ops_<op>`, 21) with a minimum count of supported cases (0 for SCALE / SOFTCAP, whose cases have 10- or 100-float rows, and for SET_ROWS / ROPE_SET_ROWS / FLASH_ATTN_EXT, whose tensors are not KV-cache tensors); FA and SET_ROWS on the device are covered by `ggml_tesseract_attn` | one ctest for every op; exact supported counts | Per-op reports; the minimum stops a supports_op regression from passing as "everything unsupported"; test-backend-ops cannot place its K / V / dst in a KV buffer (amends AT §13.6) |
| SW-53 | Host memory and the M7 kinds (2026-10-31, W3b fix; amends §8 R6, SW-26 / 46) | Only MUL_MAT is staged; supports_op refuses an M7 op with an operand in a HOSTED buffer or one that will be (ANY with leaf data only: control vectors, LoRA adapters; every buffer under `TSR_FORCE_HOST`), so it runs on the CPU; a compute buffer without a window is found only at graph_compute, which aborts naming it and the remedies; a refused SET_ROWS into the device KV cache is reported before the scheduler stops (R10: streams, `TSR_FORCE_HOST` needs `-nkvo`) | staging every M7 operand through IO; a host fallback per record | Correct runs with a control vector (tiny Qwen2: 2 launches per token, text identical to the CPU) and `TSR_FORCE_HOST -nkvo` instead of aborts; staging all kinds is a lowering of its own for a case that does not occur at `-ub 512 -c 4096` (0 staged chunks for every model run) |
| SW-54 | Versions (2026-10-31; §6.9, EX §7.5, EX-12) | Four numbers: the Tesseract Core capability in `VERSION` (1.0); the Tesseract Basis SemVer in `tsr/version.h` (0.1.0; `.note.tsr` word 6, HELLO r6); the Basis ABI (1); a numerics revision per kind (`opl::kind_rev`, `.tsr_kinds`, HELLO r4[31:16] = the kind table signature), guarded by `fw_kind_revs`. Images name the capability they need (`cap_min`); a core without the control core is refused by `CTL_ID` = 0; the backends refuse an image of an older Basis (major, minor) and route only the booted image's kinds; `CTL_ID[7:0]`, `ADAPTER_VERSION` major and `DBG_ID[7:0]` are 1 | one product version for everything; the Basis version in `tsr_tables.py`; a runtime version in HELLO; a control-core version images compare with; pinning revisions in the host | The capability is the hardware contract and the Basis version the software's, so a software release never regenerates RTL packages; a capability per image works as a cubin's `sm_XY`; the existing bit-exact tests compare kernels with references that move with them, so only a digest guard makes (kind, revision) a reproducibility key; recording revisions in reports (klcheck, numcheck) is enough for reproducibility, enforcing them would break subset images |
| SW-60 | Expert tensors (2026-11-08, MoE; §7.9; spec §4) | A 3D tensor whose name contains `_exps` is a DEVWEIGHT of n_expert matrices at `W_OFF + e × EXP_STRIDE` per channel (`EXP_STRIDE` = `R × ROW_BEATS`), stored, read back and converted lazily per expert; other 3D tensors stay raw | Expert-major packing over channels (one expert per channel group); a separate allocation per expert; any 3D tensor | Every expert MVM keeps the four-channel stream of a dense MUL_MAT and the address is one multiply; per-expert allocations would be 3,840 records for 30 layers and give no stride; the name is the only information at upload time, a wrong guess only costs a lazy conversion |
| SW-61 | MOE_ROUTE (2026-11-08) | One kind: softmax on the VU, top-k by integer compares on the RISC-V, weights renormalised on the VU, two joins per tile of 64 tokens, ids into the argsort tensor's first line per row, weights packed | The VU TOPK op (RTL off in K4); firmware soft-float for the sum and a true division; separate SOFT_MAX / ARGSORT / GET_ROWS / DIV kinds | No RTL change and no float in the firmware; the VU already holds the exact softmax pieces (RED, EXPSUB, RC); the unit tests compare the firmware with `tsr_ref_moe_route`, built from the same VU references; the weights differ from ggml's true division by up to 5 ulp |
| SW-62 | MUL_MAT_ID (2026-11-08) | One MUL_MAT of T = 1 per (token, slot) at `W_OFF + id × EXP_STRIDE`, ids read by the RISC-V from DRAM (a chunk in the one-tile block, one join), optional scale `table[id]` applied by VOP.MULC after the dot product, out-of-range id → DRAM_RANGE on the MVM | The host reads the ids and issues W_OFF per op; ACC across experts with the weight multiplied into h (spec §13's sketch); one record per expert | The op list stays static and cacheable (the ids change per token, the list does not); ggml's order (scale, weight after the down MVM, slot-order adds) keeps the CPU a fair reference; the sketch quantises `w × h`, a different A32 input |
| SW-63 | Selection order (2026-11-08) | The order of the VU TOPK: DAZ'd value under the total order, larger first, NaN first, equal values by the smaller index | ggml's `std::sort` result (unspecified for ties); lowest index last | A rule that does not depend on a library's sort; the firmware loop and `tsr_ref_topk` agree on 7,000 rows (ties, NaN, ±0, subnormals); against ggml: 512 of 512 identical rows on continuous data, 241 of 512 for one tied pair at the boundary (§7.9) |
| SW-64 | Graph fusion of the MoE chains (2026-11-08) | `tsr_build_fusion` takes the router chain (SOFT_MAX … DIV, six nodes) into one MOE_ROUTE record and the scale chain into the MUL_MAT_ID record plus a CPY into the MUL's tensor; supports_op accepts the chain nodes by their own sources; a member that did not fuse aborts at graph_compute naming it | Lowering each node (seven new kinds); refusing the chains and running the router on the CPU; writing the MUL's tensor directly | The router is one data-dependent step; the allocator may put the MUL's result over the MUL_MAT_ID's inputs (no in-place write is safe), so the record writes its own tensor and a CPY moves it (no record when the addresses are equal, as seen); supports_op cannot see consumers, so an unusual graph fails loudly instead of silently reading unwritten tensors |
| SW-65 | M6 and M7 MoE paths (2026-11-08) | `--phase m6` (klcheck, numcheck) sets `TSR_KINDS=m6,+MUL_MAT_ID`: the device runs the matrix products and the CPU selects; M7 runs everything in one launch; no path needs the TOPK RTL | Redefining `m6` to include MUL_MAT_ID (the unit test pins it to NOP, JOIN, MUL_MAT); CPU selection only | The M6 baseline for a MoE model is llama.cpp's own selection, so the M7 delta measures the firmware selection; a card at the M6 step needs the new `tsr-fw.elf` but no bitstream |
| SW-66 | Prefill and token grouping (2026-11-08) | Per-token expert MVMs (T = 1) in prefill; grouping by expert designed (counting sort of the (token, slot) pairs per ids chunk, MVMs of up to T_MAX rows), not built | Grouping now | Grouping pays when T × k / n_expert ≥ T_MAX (T ≥ 64 at T_MAX 4 on Gemma 4); it needs a pair list in SPM and a second push order; numerics are unchanged either way |
| SW-67 | Gated DeltaNet recurrence (2026-11-09; §7.10) | `GATED_DELTA_NET` (0x2F) composes the delta rule from existing VU operations: `SOP.EXP`, `VOP.MULC`, `VOP.MUL` + `RED.SUM`, `VOP.SUB`, `VOP.MULCADD`, per head and token, FP32 state in DRAM, head tiles in the SPM; order and rounding are `gdn_ref_step` | a fused state-update RTL op (VU §7.12, 23.8 ms per token estimated); CPU fallback | No RTL, no new opcode, the numerics are the oracle's, and it runs on any image with the VU. It is 4.3 times slower than the fused op at the same clock (§7.10), which is the baseline a fused op is measured against. |
| SW-68 | Convolution kind and conv state (2026-11-09; §7.10) | `SSM_CONV` (0x2E) works on channel vectors: the window of `d_conv - 1 + T` rows of C channels, taps as `d_conv` rows of C, summed in tap order, SILU optional; the state cache holds `d_conv - 1` rows of C per cell and belongs to the device; the taps are re-laid out at weight upload | ggml's channel-major order with a transpose kernel; a CPU-side conv | Channel vectors are what the VU lanes take; the state needs no transpose. llama.cpp only copies, clears and saves whole cells, which are byte copies. |
| SW-69 | Softplus (2026-11-09; §7.10) | `SOFTPLUS` (0x2D) is the stable composition `max(x, 0) + log1p(exp(-|x|))` with `log1p` from the odd atanh series: 20 VOPs; UNARY gains SIGMOID (func 5) | a VU `LOG1P` function; the CPU's `logf(1 + expf(x))` form | The VU has no log. The composition is within 5 ulp of float64 (above 1e-3) and costs one launch's worth of cycles per layer; a new VU function would be RTL for H_v values per layer. |
| SW-70 | Tiling of the recurrence (2026-11-09; §7.10) | One ring block per op: two state tiles, the product tile and two token slots; heads in order, the next head's state loads during the current head; tokens in order; S_v a multiple of 16 up to 256; the planner (`plan_gdn.hpp`) is shared by firmware and host | a block per head; tokens in parallel across heads | The recurrence is serial in tokens and independent in heads. Prefetching the next head's state hides its DMA. A block per op gives the planner one decision (3,156 lines at S_v 128). |
| SW-71 | The SSM_CONV group and ggml-alloc lifetimes (2026-11-09; §7.10) | The pinned graph is `CONCAT → SSM_CONV → SILU` with a state `CPY`. At the CONCAT two CPY records fill the CONCAT's result buffer with the window (state rows, then token rows); at the SSM_CONV one record reads the window, folds in the SILU and writes the new state rows into the cache; the state CPY is absorbed | one SSM_CONV record at the SSM_CONV reading the CONCAT's inputs; materialising the CONCAT in ggml's order | ggml-alloc frees the CONCAT's inputs after the CONCAT and may give their memory to nodes before the SSM_CONV. The first design read them there and got zeros on the 2B model while the tiny model passed. The window lives exactly as long as ggml-alloc keeps the CONCAT result, and costs two small CPY records. |
| SW-72 | Recurrent state caches and IMROPE (2026-11-09; §7.10) | Buffers holding `cache_r_l*` / `cache_s_l*` are device windows (not leaf data); `GET_ROWS`, `SCALE` (zeroing), `CPY` and the group records address them in place; a cache in another backend's memory (`-nkvo`) keeps every op that touches it on the CPU; IMROPE with equal section positions is the NEOX rotation, and `graph_compute` checks the position streams | copy the state to the host each token; mrope on the device | The graph reads and writes the state in place; one copy per token would be the cost of the state. Text positions are `[p, p, p, 0]`. |
| SW-73 | llama.cpp is unchanged for GDN (2026-11-09; §7.10) | No `third_party/patches/0003-*` for GDN: the backend takes the fused `GATED_DELTA_NET` node, the conv group and the state nodes as the pinned graph emits them | a patch that builds a device-friendly graph | `resolve_fused_ops` keeps the fused node when the backend supports it. The only llama.cpp-side facts needed (recurrent memory names, the build_rs form, the mixed input select) are in the pinned source. |
| SW-74 | GDN test method (2026-11-09; §10.2) | Tiny Qwen3.5-style models (S_v 32 and 128, 3 GDN + 1 attention layer) with klcheck graph tests; `test-tsr-gdn-logits` compares the logits of every step of 1, 2, 5 and 17 token prompts with the CPU's (cosine ≥ 0.9995, largest difference ≤ 3 % of the logit range); klcheck's embd path writes the four M-RoPE position streams | klcheck's KL alone | klcheck's KL missed a convolution that returned zeros on the tiny model (ratio 1.2): a random tiny model hides a wrong hidden state. klcheck's embedding input path gave an M-RoPE model one position per token, where llama.cpp reads four, section-major; the CPU's embd path then differed from its token path (KL 6.5e-3 on the 2B, equal to the "floor"). klcheck now writes `[p, p, p, 0]` in that layout, and the embd identity is bit-exact (512 of 512 rows). |
| SW-75 | ITCM budget of the merged firmware (2026-11-09; §7.8; amends SW-38) | 96 KiB (98,304 B) for the production image, 32 KiB of the 128 KiB ITCM left for M9, sampling and debug formatting. The measured image with the MoE kinds (SW-61, SW-62) and the Gated DeltaNet kinds (SW-67 to SW-69) together is 84,056 B (DTCM 50,088 B, budget unchanged at 56 KiB). `tsr-elf` takes 96 KiB as its default and `sw/ctrl/CMakeLists.txt` passes it to the production image as `--budget-itcm`; test images stay stamped against the whole ITCM. The BIG_TEXT test image of `fw_elf` grows to 98 KiB of code so that it stays over the budget | keep 80 KiB and remove code (the general paths are already compiled for size; each fast path is a leaf because a shared body cost 22–29 cycles per record, SW-38); 112 KiB or the whole 128 KiB | SW-38 sized the budget for the M7 kinds and set aside the upper 48 KiB of the ITCM for "M8 MoE / M9" (CC-3). The two feature branches each measured below 80 KiB (78,520 B and 78,576 B) and the merge is 2.1 KB over it: this is that reserved use, taken as the amount the image needs plus 14 KB of headroom. No feature is removed and no kind loses its fast path |
| SW-76 | MOE_ROUTE outputs with several tiles (2026-11-09; §7.9; amends SW-64) | A route of more than 64 tokens writes its ids and weights to a device scratch window (taken from the top of IO, grown when needed, shared by all layers) and two CPY records move them into the ARGSORT and DIV tensors after the record. One tile (decode, up to 64 tokens): unchanged. No firmware change, no kind revision: same words, same output bits | the refusal of SW-64 for a layout that overlaps a later tile (it aborted Gemma 4 prefill at `-ub 256` and above, the default `-ub 512` included); an exact per-tile overlap check (accepts the layouts seen, still refuses others); firmware that holds the outputs of all tiles in the SPM until the last load (the ring cannot hold a block across its wrap, and T has no bound); reversing the tile order or one record per tile (a layout can still need a cycle); outputs in the unused tails of the ARGSORT rows (the weights are packed, a pack pass would be new firmware) | The allocator's layout is not a fact the record can rely on: a record that reads for a long time cannot be told apart from its outputs by address. The scratch makes the outputs independent of every placement and costs two small CPY records per layer in prefill. It needs no ITCM (the image stays 84,056 B) and leaves decode and its op lists as they were. The ctest cases place the outputs by hand over the logits of later tiles, which the old build refused and a direct store corrupts |
| SW-77 | Qwen3 MoE graph (2026-11-09; §7.9) | llama.cpp's `qwen3moe` graph (the router group of SW-64, separate gate and up experts with SWIGLU, no per-expert scale, Q and K norms) runs on the existing kinds and lowering; the work is tests (`test-tsr-moe` Qwen3 layer, two tiny `qwen3moe` models with klcheck graph tests at `--min-kl` 1e-3) and records; prefill at `-ub 64` until SW-76 handled the several-tile route, then at the default `-ub 512` | a Qwen3-specific fusion pass or kind; a llama.cpp patch | `build_moe_ffn` is the same function for both models, so the fusion pass already finds the group by its nodes; the missing scale group only means the down MUL_MAT_ID is lowered alone. Card: klcheck M7 8 x 64 ratio 0.813, top-1 96.68 %, 512 of 512 rows bit-identical to the simulator; decode 15.8 tok/s, 11.34 M device cycles per token; with SW-76, a 512-token prefill at `-ub 512`: 26.72 tok/s best, 26.60 median, 513 of 513 rows bit-identical to the simulator |
| SW-80 | The VU, QACT, store and MVM cycle models follow the RTL and the card (2026-11-09; §4.5; VU §7.9, §7.13) | The VU's cycles come from its lanes (`tsr_sim_opts.vu_timing`, `TSR_SIM_VU_TIMING`; the default stays the U200 image's 16 / 16 / 8 lanes, the card image is `k4`) and from latencies measured on the RTL unit: 59 to 95 cycles per dependent op, the reduction's partial ring, the two-stage ROPE, the AT vector ops. QACT A32 16 cycles per 32-block + 44, A256 17 per 256-block + 72. Store lines + 110, MVM fill 144. `TSR_SIM_CARD=k4full` is the whole card image. | Make `k4` the default; scale the old figures by the lane ratio; put SPM arbitration and accept-ahead into the model | On Qwen3.8 2B decode the card took 12.63 M cycles and the old model 10.83 M (card configuration): 1.61 M of the 1.80 M in GATED_DELTA_NET, whose VU ops take on the card exactly what the RTL unit alone takes (4.737 M cycles against `VU_CYCLES` 4.758 M per token), so the unit has no stall that the model lacks; the model had 16 lanes, latencies of 20 and 64 (60 and 95 measured, also at 16 lanes) and 16 cycles per row of a 128-row reduction (25). On Gemma 4 26B-A4B (16.31 M against 15.35 M) the QACT A32 (1,452 cycles for K = 2,816 against 226) and the VU are the larger parts. The default stays 16 lanes because the lane count is an elaboration parameter and the unit tests and trace replays are not the card; scaling by the lane ratio would fix neither the latencies nor the ring. The unit overlaps independent ops (VU §7.8); the model is 0.9 % above the RTL on a decode token for lack of it, and the card's `VU_SPM_WAIT` is inside the figures. |
| SW-81 | Weights as ggml blocks in the PyTorch backend (2026-11-09; §9.4) | `tesseract_torch.QuantLinear` holds a Q8_0 or Q4_K weight as a uint8 buffer of ggml blocks [N, row_bytes] and calls the custom op `tesseract::linear_blocks`, which uploads the blocks unchanged (`Device.upload_blocks`, cached by storage, format and K like a float weight) and runs `linear` (QACT A32 or A256, f32 result); the module adds the bias in float32 and casts once. Under `torch.compile` the op is one node that the partitioner does not touch. A weight that cannot be placed raises `TesseractError`. `QuantLinear.from_gguf` takes a `gguf.ReaderTensor`. T tiling stays in the firmware: `tsr_rt` sends the whole T of an IO tile in one record (SW-20) | a tensor subclass that the partitioner recognises (needs `__torch_dispatch__` through AOTAutograd, and the weight cache and the aten patterns would depend on it); dequantizing to float and uploading as Q8_0 (the Q4_K blocks and their bandwidth are lost, and Q4_K → Q8_0 changes the numerics); a loader that maps a whole GGUF onto a model (the layer tensors of the Qwen2.5 GGUF in use are rescaled versions of the checkpoint's, so a partial load is wrong, and the norms and RoPE parameters come from the same file) | The blocks stay byte-identical (the test reads them back) and y equals the numpy spec model of the format; a Q4_K weight is read at 0.5625 bytes per element, a Q8_0 weight at 1.0625. A custom op needs no change to the partitioner or to `tsr_rt`. The loader is a separate piece of work (`docs/torch-backend.md` §8) |

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
| The control core is not on the card yet | models run only on the simulator until P6 | the sim path is end to end before RTL (P3.4) |
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
  (still unmeasured, `docs/spec.md` §20), and the RISC-V run-ahead margin of the executor on 0.5B.
- The Gemma 4 KL gap of GELU16 (§10.4). The mismatch rate against the pinned table is measured
  (W0: 0.415 %, GELU16 correctly rounded in every mismatch, VU §6.5).
- Device support for llama.cpp's Hadamard KV rotation (SW-16 follow-up), M8 or later.
- Unrelated observation: `git status` shows the `third_party/llama.cpp` submodule with modified
  content; the design work made no changes there.

# Tesseract v2 design set (M6–M9)

Status: **integrated design, not implemented** (2026-10-30). Six area documents, reviewed by two
reviewers and integrated into one consistent set. Every number marked *estimate* is unmeasured;
every third-party fact marked *confirm* is checked when the dependency is pinned or at the first
smoke link. Source-of-truth order: `docs/spec.md` (after its v2 edits, §11) > ISA simulator
(`sw/sim`) > `docs/v2/*` > `docs/ctrl-riscv.md` > `docs/prof-dbg.md`.

---

## 1. Purpose and how to read

v2 turns Tesseract from a host-programmed tile engine into an NPU whose every tile op is issued by
an embedded RISC-V control core. The host loads a firmware image and launches kernels; it never
runs instruction programs and never touches the issue port (D14, D15). This directory is the
design contract for that change: M6 hardware and software, and the M7 graph execution that makes
one llama.cpp decode token one launch.

Read in this order: EX (the host-visible model and every global table), then CC (the RTL that
implements EX's model), VU and AT (the new compute), DB (debug and profiling), SW (simulator,
firmware, host stack, verification, work plan). Section 4 lists where each global table lives;
every other mention of a table is a pointer to that place.

| File | Area | Owns |
|---|---|---|
| `01-execution-model.md` | EX | launch protocol, host CSR map, opcode / unit / error-code / PERF tables, XRT adapter, transport API, DRAM map, v1 deletion order |
| `02-vector-unit.md` | VU | VU ISA and bit-exact numerics (EXP, RECIP, SIN/COS, GELU, RoPE, reductions), VU microarchitecture, execution of AT's vector ops |
| `03-attention-kv.md` | AT | KV cache layout and values, attention ops and numerics, the normative flash-attention sequence, masks |
| `04-control-core.md` | CC | Ibex and debug module, RISC-V map and control window, issue path, SPM banks, CDC, resets, floorplan and timing |
| `05-debug.md` | DB | GDB over PCIe, tile-level debug, logs and crash records, recovery, profiler, DBG registers |
| `06-software-verification.md` | SW | simulator and references, trace v2, firmware SDK and kernels, op-list format, llama.cpp backend, verification and work plan |

---

## 2. User decisions and milestones

**Decisions of 2026-10-30** (`docs/ctrl-riscv.md` D14–D18; they supersede D7–D11 and D13):

- **D14** — the host command-buffer execution path is removed entirely: no host START, no
  CMD_* / ARG*, no `tsr_cmdproc` sequencer, no `tsr_fetch`, no R0–R15 / LI / ADDI / END, no
  CALL. The host loads the firmware image and launches kernels that run on the RISC-V; every
  tile op is issued by the RISC-V. One s0–s15 scalar bank.
- **D15** — no host debug access to the issue port. Bring-up, the card selftest and trace replay
  run as RISC-V kernels.
- **D16** — v1 is deleted now (tag `v1-m4` keeps it; `build/v1` holds a working binary copy).
- **D17** — the device KV cache is q8_0; attention arrives as llama.cpp `FLASH_ATTN_EXT` (FA on;
  the pinned llama.cpp requires FA for a quantized V cache). klcheck / numcheck baselines move to
  FA on.
- **D18** — the first v2 has full debug: a RISC-V Debug Spec module, halt / resume / step,
  breakpoints and GDB source-level debugging over PCIe (a GDB RSP server on the DMI window, no
  OpenOCD), plus tile-level debug.

**Still valid:** D1 simple in-order RISC-V without caches or speculation; D2 code, data and
stack in TCMs; D3 asynchronous issue of 256-bit tile instructions (issue and continue, join
later); D4 SPM mapped into the RISC-V address space; D5 a separate RISC-V clock domain; D6
kernels in C++ with stock GCC/Clang and a header library; D12 RISC-V clock planning value
166.7 MHz (250 if Ibex closes there, 125 fallback). Proposal kept: lowRISC Ibex RV32IMC + Zicsr +
Zifencei.

**Technical defaults chosen under the user's delegation** (kept by this design): even / odd
line-interleaved SPM banks for the second read path (V2-CC-12); XRT `ap_ctrl_none` with polling
(V2-EX-6); a host-readable log ring in the core domain (V2-EX-24); D13 restated as "with the
control core the k4 link stays at or above 276.3 MHz" (V2-CC-23); GELU (tanh form) in the VU for
Gemma, plus the CPU-compatible GELU16 for llama.cpp graphs (V2-VU-26).

**Milestones:**

| Milestone | Content | Pass criterion |
|---|---|---|
| **M6** | core v2 hardware: control core, issue queues, vector unit, attention / KV ops, second SPM read path, debug | on v2, Qwen2.5-0.5B Q8_0 logits bit-identical to the simulator and the Qwen2.5-32B Q4_K_M KL criterion met, MVMs issued by RISC-V kernels (attention on the CPU, `offload_kqv` = false, SW §10.4); GDB smoke on the card |
| **M7** | RISC-V kernel library and graph execution: the llama.cpp backend writes each graph split as an op list into DRAM and launches it once | one decode token = one launch for Qwen2.5 and Gemma 4 dense (12B on disk; 31B when available), KV cache and attention on the device, logits bit-identical to sim and the KL criterion (SW §10.4) |
| **M8** | device MoE, PyTorch on the kernels, device embedding lookup (DQROW RTL), TOPK RTL | defined at M8 start |
| **M9** | resident decode loop (optional; VMASK) | defined at M9 start |

**Performance priority** (spec "performance goal"): keep DDR bandwidth busy (no idle time)
before reducing bytes per token. Speculative decoding is dropped from the plan.

---

## 3. The documents

**EX — execution model (`01-execution-model.md`).** The host never runs programs: it binds the
channel memory (`MEM_EN` latch), loads the firmware's `PT_LOAD` segments into the TCMs through a
CSR loader with a receive CRC and an on-device VERIFY read-back, boots the RISC-V and launches
kernels. A launch is an 8-word descriptor pushed by posted CSR writes into a core-domain launch
queue (LQ, 8 deep); every descriptor gets exactly one 8-word record in a completion ring (LCQ, 8
deep), in order, with a launch STATUS and the first tile error; the runtime contract R1–R10 fixes
the RISC-V side, and cancel / stop watermarks replace v1's ABORT. EX owns the single host CSR map
(§5), the PERF counter space gated by ACTIVE (§6), ISA v2 (v1 tile ops bit-identical, the `x16`
field, the global opcode and unit tables, §7), the error model with the new code 13 OPERAND and
the device-level `SOFT_RESET` (§8), the `ap_ctrl_none` XRT adapter (§9), the transport API (§10),
the DRAM map (§11) and the D16 deletion order: X0 deletes all of v1 in the first M6 commit, and
the v2 replacements follow oracle → traces → host → RTL → platform (§12). The base kernels are
`graph`, `replay` and `selftest`.

**VU — vector unit (`02-vector-unit.md`).** Every VU op works on rows × m elements with a line
stride per operand (0 = broadcast) and a per-row scalar from an s register or an SPM vector, so
one op covers all heads and tokens. Ops: VOP (25 functions, including the CPU-compatible GELU16 /
GEGLU16), segmented RED with fused RCP / RMS post-ops (RMSNorm in two ops), ROPE over T tokens
with ggml-cpu's own angle recurrence (bit-identical angles), SOP, TOPK and DQROW (simulator in M6,
RTL in M8) and VMASK (M9). All numerics are bit-exact algorithms: an integer EXPK (≤ 0.53 ulp,
*model*), correctly rounded reciprocals, an exact 2/π reduction for SIN/COS, the spec's 128-partial
reduction tree with an exact skip rule. The microarchitecture has 16 front lanes, 8 half-rate back
lanes, a reduction unit and an angle unit, reads operand pairs from the even / odd SPM, uses
credits instead of a clock enable and accepts one op ahead. It also executes AT's softmax ops.
Size ≈ 48 K LUT, 366 DSP, 15 RAMB36 in SLR1 (*estimate*).

**AT — attention and KV (`03-attention-kv.md`).** The KV cache holds exactly llama.cpp's q8_0
values in a device layout (16-cell groups round-robin over the channels, K token-major, V
slice-major), so `get/set_tensor` convert losslessly. Attention runs where the bandwidth is: QKᵀ
(ATTQK) and P·V (ATTPV, an exact block-floating-point accumulation in the MAC DSPs) are new modes
of the MVM clusters, KVST stores quantized rows through the DMA, and ASMAX / AMASK / AMERGE /
ANORM do the two-pass-per-chunk softmax and the online merge on the VU. GQA maps (query token,
head) to the MVM's token slots, with head subsets when G > T_MAX (Gemma 4 12B global layers:
two passes). The host reduces each F16 KQ mask to a mask summary, so the device never reads F16
and skips masked bandwidth. The FA tile-op sequence (§6.6) is normative. Estimated DDR busy during
attention ≈ 89 % (32B, 4K context). M6 verifies the ops standalone; device attention is an M7
gate.

**CC — control core (`04-control-core.md`).** The RTL that implements EX's model: unmodified
Ibex (RV32IMC, 2 triggers) with the upstream riscv-dbg module and its system-bus master, 128 KiB
ITCM and 64 KiB DTCM, the RISC-V address map and control window (issue port with credits and
`ISSUE_PIDX`, tickets, queue control, EX's LQ / LCQ / log registers, timer), the issue path
(32-entry CDC issue FIFO, dispatcher, `tsr_chk` refactored from cmdproc with four tagged operand
extents, six queues with registered heads, SYNC, invariants I1–I4, DB's hard hold), the 2R2W
even / odd SPM with per-bank round robin and the DMA LD burst reservation, every clock-domain
crossing (handshakes for the rings, gray counters for single-step counters), the core-domain SR
sequencer for `SOFT_RESET`, the `ap_ctrl_none` adapter, the floorplan C'' with the SLR1 budget
(≈ 62–71 % LUT against a 65 % gate with ordered remedies), the timing gates T1–T4 with the
cycle-identical gather fix, and the RTL verification plan.

**DB — debug (`05-debug.md`).** RISC-V debug through the debug module's DMI in the core CSR
window (gated by `DMI_EN`), with a GDB RSP server built into the device-owning process or the
standalone `tsr-dbg`, memory access through SBA, 2 hardware and unlimited software breakpoints, 2
data watchpoints, halt at boot and at kernel entry. Tile-level debug stops the queues at a
push-index threshold (a hard hold at the queue heads, yielding to QABORT and `SOFT_RESET`), steps
op by op, serialises execution, sets tile breakpoints on opcodes, tickets and SPM / DRAM ranges,
keeps a 512-entry dispatch history with the pushing C++ line, and probes every unit. It fixes the
log record header and the crash / trap contract, the hang triage and a five-level recovery ladder,
and the profiler (PERF slots 61–63, completion records, a hardware trace writer to a DRAM ring on
channel 0, TSR_STATS v2). Its registers are the DBG block at core CSR 0x800–0xAFC.

**SW — software and verification (`06-software-verification.md`).** The simulator stays the
bit-exact oracle and gains an RV32 interpreter (`tsr_rv`, Spike as its oracle), an event-driven
queue model with random schedules, a race detector, the VU / AT references and `tsr_ref_math.c`,
and EX's CSR and ring model. Trace v2 records one launch per directory with a push-ordered
`push.bin`; the v1 trace sets are regenerated or converted. The firmware SDK is freestanding
C++20: crt0, a runtime implementing EX's R1–R10, DB's trap and log contract, the tile/VU header
library and `plan.hpp`, shared with the host. The `graph` kernel executes OPL v1 op lists with
host-computed dependencies; M6 uses it with MUL_MAT records only, and M7 adds the kinds that make a
decode token one launch (R1–R10 of SW §8). The host stack implements EX's transport API with one
protocol layer over sim, mock, vkernel and xrt. SW also fixes the ctest and bench plans, the M6 /
M7 gates and the work breakdown (§10 here).

---

## 4. Where each global table lives

| Table | Single definition | Implemented / used by |
|---|---|---|
| Host CSR map (core CSR port, `s_axi_control` 0x1000 + offset) | EX §5 (DBG block: DB §10.1; `SREG` 0x9C0) | CC §6, SW §4.3 |
| RISC-V address map and control window | CC §4.1, §5 (0x200–0x2FC: EX §4.5; 0x160–0x17C and 0x400–0x7FC: DB §10.2) | SW §4.2, §6 |
| Launch descriptor, records, STATUS precedence, runtime contract | EX §4.3–§4.7 | CC §5.2, SW §6.3–§6.5, DB §6.3 |
| Opcode table | EX §7.2 (encodings: VU §5, AT §5) | CC §7.4, §8.2 |
| Units and queue depths | EX §7.3, CC §7.4 (`Q_DEPTH`) | all |
| Error codes and check order | EX §8.1 | CC §8.2, VU §5.9, AT §5.8, SW §4.3 |
| PERF slot table | EX §6 | CC §6.3, VU §5.10, AT §10, DB §8.1 |
| Log record header | DB §6.1 | EX §4.10, SW §6.6 |
| Op-list format OPL v1 and base kernels | SW §7.1–§7.2 (contracts: EX §4.6) | backend SW §9.3, DB §5.7 |
| Trace format v2 | SW §5 | CC §15.3, DB §5.6 |
| DRAM map | EX §11 (KV planes: AT §4.5; compute heap: SW §8 R6) | SW §9.3 |
| Transport API | EX §10 (SW §9.1 adds conveniences only) | SW, DB |
| Machine-readable source of all of the above | `sw/tables/tsr_tables.py` + `gen_tables.py` (V2-EX-30; created in W0) | RTL package, C header, `mmio/vu/at.hpp`, tb Python |

---

## 5. What the integration changed

The drafts were written in parallel and disagreed in places; two reviewers listed 42 findings.
The integration applied them as follows (each item names the winning definition):

1. **One host CSR map** — EX §5. CC's `CTL_*` block, mailbox registers, `SREG` at 0x0C0, error
   copies at 0x500, free-running PERF with `PERF_CLEAR`, 1,024-word overwrite log and the "TSR2"
   ID are withdrawn (V2-CC-20, V2-CC-21; CC §6 now maps CC's hardware onto EX's offsets).
2. **One launch protocol** — EX's LQ / LCQ. SW's single-slot mailbox, 5-arg LAUNCH, 16-byte image
   id, PANIC / BYE / LOGCTL messages and its transport API are withdrawn (V2-SW-10); panic is a
   TRAP with `mcause` 24 (DB), a stop is the `LQ_STOP` watermark, the log level is `LOG_CFG`.
3. **`SOFT_RESET`** stays EX's device-level sequence; CC adds the core-domain SR sequencer that
   implements it without `clk_rv` (V2-CC-31).
4. **D16 deletion order** — X0, the first M6 commit, deletes every v1-only part at once (V2-EX-28).
5. **Checker and issue contract** — check order with OPERAND, four operand extents tagged SPM /
   DRAM and read / write, I3 ≤ 2 for the VU, no run-time unit errors (V2-CC-11 withdrawn), a
   static generated opcode → unit table instead of `ISSUE_QMAP`.
6. **AT's vector ops** are specified for the VU decoder (VU §5.11) and executed by the VU with a
   multi-pass AMERGE (V2-VU-27); VMASK is deferred to M9 because llama.cpp KV caches are rings.
7. **DQROW** is VU's SPM form on unit 3; its RTL waits for M8 (V2-VU-15, V2-AT-20).
8. **PERF slots** — VU X+13…15, AT X+16…19, CC X+20…21, DB 61–63, recorded in EX §6.
9. **s registers** are read at `SREG` 0x9C0 (host and, through the core window, the RISC-V).
10. **`ISSUE_PIDX`** — a RISC-V-readable push index at control window 0x03C (V2-CC-32).
11. **RED skip rule** — the draft's "a tree value is never −0" was false; the exact rule passes
    z(q) for a skipped pairing (V2-VU-2).
12. **DB's hard hold** yields after a QABORT and during `SOFT_RESET`; `SOFT_RESET` clears the
    stop state; `monitor reset` releases first; QQUIESCENT counts a hard-held head as held.
13. **Ring CDC** — the RISC-V reaches LQ / LCQ / log only by handshakes; no multi-step counter is
    gray-crossed (V2-CC-30).
14. **XRT adapter** — `ap_ctrl_none` with the `MEM_EN` latch in every document (V2-CC-19).
15. **M6 offload path** — the `graph` executor with MUL_MAT-only op lists; `replay` (with SERIAL)
    for traces and SPM dumps; `tsr.oplist_seq` is withdrawn (V2-EX-29).
16. **M6 gate** runs with attention on the CPU; device attention with the KV cache on the device
    is the M7 gate (AT §13.6, VU §10, SW §10.4).
17. **Trace migration** — SW's plan (regenerate m2 / gen / m3, convert the 290 model traces) in
    EX and CC.
18. **SLR1 budget** recomputed with the VU, DB and AT figures: ≈ 124–142 K LUT (62–71 %), with
    ordered remedies; the out-of-context VU run precedes T2 (V2-CC-24).
19. **One table source** — `sw/tables/tsr_tables.py` (V2-EX-30).
20. **One log header** — DB §6.1.
21. **Gemma 4 12B** read from its GGUF and added to VU §3, AT §4.5 / §7.5 / §9, SW §8: global layers
    have G = 16 (two passes) and E = 512 (K meta read-modify-write).
22. **SBA** writes are refused over the whole control and core windows (V2-CC-33).
23. **No read side effects** — a refused `LOAD_DATA` read returns 0; core-window loads of
    `LOAD_DATA` and the DMI window return 0.
24. **Trace writer** on channel 0 with its ring in the ch0 IO region.
25. **SW stale values** — ITCM 128 KiB, queue depths, the `tsr_ref_math` API, VU recipes, clock
    ratio 0.603.
26. **x16 notation** — x16 bit i is w0 bit 16 + i in every document.
27. **CPU-compatible GELU** — GELU16 / GEGLU16 for llama.cpp graphs (V2-VU-26, V2-SW-31).
28. **RoPE angles** by ggml-cpu's recurrence, bit-identical to the CPU (V2-VU-19).
29. **SW attention records** follow AT's argument layout; the backend produces mask summaries and
    preloads them (V2-SW-32).
30. **Debug-aware launch timeout** and DB's recovery ladder in the transport (V2-SW-24).
31. **Debug software packages** — P1.13, P3.7, P3.8 added to the work plan.
32. **Toolchain** — xPack GCC is the supported compiler; `tsr_rv` gets a test bus profile for
    riscv-tests and Spike.
33. **Unselected graph branches** — DUP and flagless nodes are unsupported (V2-SW-33).
34. **Compute heap** sized from the reserve graph instead of fixed 512 MiB windows (V2-SW-26).

Open items that the drafts left and the integration resolved: the 4-extent checker slot, I3 for
the VU, the VU / AT / DB resource figures in the SLR1 budget, whether AT has a separate unit (no),
where AT's softmax runs (on the VU), the PERF slot assignment, the VU debug-state placement (DB's
per-unit probes), the 12B Gemma 4 dimensions, `TRW_CH`, the RISC-V-readable push index, the log
registers, the AT argument layout in SW's op lists, the VU's TANH and segmented reductions (both
exist), the `user_managed` question (not in XRT 2.18; `ap_ctrl_none`), and the PROF bit's meaning
(REC for the runtime and NODE markers; the host timeline comes from the trace writer).

## 6. Decision log

Every V2-* decision of the six documents in its final, integrated form. "Withdrawn" marks a draft
decision replaced by another area's (the id is kept so references stay valid). The documents hold
the alternatives and the full rationale.

### 6.1 EX — execution model, host interface, ISA (`01-execution-model.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-EX-1 | Launch transport | Core-domain launch queue LQ (depth 8) written by posted CSR writes, plus completion ring LCQ (depth 8) read through the CSR window | Posted writes are the cheapest PCIe operation; no DMA per launch; depth 8 lets the host pipeline launches |
| V2-EX-2 | Host/RISC-V channels | LQ/LCQ replace the mailbox entirely; PING, HELLO, TRAP and kernel events are record types | One channel halves the CDC paths, deadlock arguments and tests |
| V2-EX-3 | Descriptor and record formats | 8-word descriptor {TYPE, PROF, KERNEL, ARG0-6}; 8-word records whose r0 = {type, status, first error code, seq[7:0]} | 7 args cover the known kernels; one PCIe read classifies a completion |
| V2-EX-4 | Launch ordering | Exactly one answer per descriptor, in order; hardware sequence numbers; LQ_POP refused while ACTIVE | Sequential launches make ACTIVE exact for PERF, cancel and the current launch; overlap comes from the async issue path |
| V2-EX-5 | Flow control | Host keeps at most LQ_DEPTH launches outstanding; LCQ_DEPTH = LQ_DEPTH = 8 | Neither ring can overflow and the push path needs no read |
| V2-EX-6 | XRT control protocol | ap_ctrl_none, xrt::ip register access, polling; no interrupts on the U200; host_irq_o for other targets | A halted RISC-V under an outstanding xrt::run breaks KDS; XRT 2.18 routes AP_CTRL_NONE to xrt::ip |
| V2-EX-7 | Channel bases | Bases latched and alignment-checked on the rising edge of ADAPTER_CTRL.MEM_EN and locked while set; requests held while MEM_EN = 0 | No run start to latch on; one edge keeps v1's atomic, checked rebinding |
| V2-EX-8 | Firmware load path | CSR loader with receive CRC plus on-device VERIFY read-back CRC | About 10 ms per load; VERIFY checks the TCM contents without 32 K PCIe reads |
| V2-EX-9 | Image identity | 64-bit prefix of a SHA-256 over the PT_LOAD segments, stored in .tsr_id and echoed in HELLO | Identifies the exact build cheaply; integrity is the CRC's job |
| V2-EX-10 | Warm restart | SOFT_RESET, reload the DTCM segments, VERIFY ITCM (reload only on a mismatch), boot | Avoids stale .data, about 10× cheaper than a cold load; ITCM is writable so it is checked |
| V2-EX-11 | CSR map | One host CSR map (EX §5): CTRL/STATUS at 0x020/0x024, CTL_* block dissolved, CTL_PERF merged into PERF, DBG 0x800-0xAFC, 12-bit space kept; CC, DB, VU, AT place registers inside it | Without host runs a second control block would only duplicate; the map fits 4 KiB |
| V2-EX-12 | Identity and version | ID stays 'TSR1'; VERSION 0x0000_0200 (ISA 2.0); ADAPTER_VERSION 2 | v1 open code already refuses a different ISA major, so no new magic is needed |
| V2-EX-13 | PERF semantics | One counter space gated by ACTIVE (time counters also by ¬QHALT), zeroed by SOFT_RESET; MVM_WAIT and MVM_EMPTY; one slot table incl. VU X+13..15, AT X+16..19, CC X+20..21, DB 61-63 | Restores 'PERF unchanged between executions' per launch; splits MVM idle time into issue- and dependency-bound |
| V2-EX-14 | x16 field | w0[31:16] is the op-specific field x16 (x16 bit i = w0 bit 16+i); must be 0 for every v1-encoded op | v1 ops stay bit-identical; VU and AT get 16 bits of encoding space |
| V2-EX-15 | Opcode allocation | Blocks beside each class (VU 0x13, 0x38-0x3F, 0x41-0x47; AT 0x14-0x17, 0x19-0x1B, 0x21-0x23, 0x34-0x37; SYNC 0x0A-0x0F, 0x70-0x7E); assigned KVST 0x14, ATTQK 0x21, ATTPV 0x22, ASMAX-ANORM 0x34-0x37, VMASK 0x38 (M9); retired opcodes never reused | Areas work in parallel without collisions; class decode stays cheap |
| V2-EX-16 | Unit id 4 | Reserved for a future AT attention engine; absent in M6 (AT's ops run on units 0, 2, 3) | CALL never shipped; keeps unit ids dense and the six-unit CDC widths |
| V2-EX-17 | Error codes | New code 13 OPERAND (ERR_INFO = the value); check order ILLEGAL_OP, BAD_DEP, BAD_FMT, K_RANGE, T_RANGE, OPERAND, SPM_RANGE, DRAM_RANGE; 9, 10, 14, 15 reserved | VU/AT value limits are neither encoding nor K errors; the value makes failures diagnosable |
| V2-EX-18 | Launch-level result | STATUS OK, TILE_ERR, BAD_DESC, KERNEL_ERR, TRAP, ABORTED, PLAT_ERR, STOPPED with fixed precedence; first error = smallest ERR_PIDX | Tile failures never read as success; the selection is schedule-independent |
| V2-EX-19 | Platform AXI errors | plat_err_i core input, mirrored in STATUS and reflected in the launch status | Keeps v1's adapter-error guarantee without a PCIe read per launch |
| V2-EX-20 | Cancel and stop | Sequence watermarks: LQ_CANCEL (hard: QABORT of a covered running launch), LQ_STOP (soft: polled by kernels); covered queued launches answered ABORTED | Precise and race-free; also gives M9's graceful stop |
| V2-EX-21 | RISC-V traps | The handler follows DB's trap contract, posts TRAP and parks for GDB; recovery is SOFT_RESET plus a warm restart | A trap may leave .data inconsistent; parking keeps the state for D18 debugging |
| V2-EX-22 | SOFT_RESET | Hardware sequence (CC's SR sequencer): RISC-V reset, drain, clear error states, flush LQ/LCQ/LOG, zero PERF, clear DB's stop state; works without clk_rv; SR_BUSY shows progress | One atomic known-state operation that the RISC-V cannot race and that survives a missing clk_rv |
| V2-EX-23 | Stuck units | No forced unit reset; per-unit progress watchdog (2^L cycles, about 61 ms default) sets UNIT_HUNG; recovery is a card reset | AXI has no cancel; a unit kill would corrupt later transactions |
| V2-EX-24 | Debug log | 256-word core-domain ring read through the CSR window; whole records dropped and counted, optional BLOCK; DB's record header; LOG_CFG.LEVEL filter | Works when queues wedge; never faults or stalls by default |
| V2-EX-25 | Transport structure | One register-level protocol layer for sim, mock, vkernel and xrt; sim advances deterministically on host reads | The protocol itself becomes part of what the oracle checks |
| V2-EX-26 | DRAM map | CMD region becomes OPL (op lists, replay streams, argument blocks, overlays); TRW ring in the ch0 IO region; compute heap between weights and KV planes | The name follows the content; the addresses stay |
| V2-EX-27 | FETCH memory source | Tied off (renamed AUX); source ids not renumbered | Avoids churn on timing-critical arbiter paths |
| V2-EX-28 | v1 deletion order | X0, the first M6 commit, deletes every v1-only part at once; v2 replacements follow oracle, traces, host, RTL, platform (Y1-Y5) | D16 says v1 is deleted now; build/v1 and tag v1-m4 remain the comparison path |
| V2-EX-29 | M6 offload path | The graph executor with MUL_MAT-only op lists, one launch per v1 split; the replay kernel (SERIAL) for traces and SPM dumps; no tsr.oplist_seq | One mechanism from M6 to M7 and no second offload path of about 1 µs per op |
| V2-EX-30 | Table source | sw/tables/tsr_tables.py (plain Python) with SW's gen_tables.py emitting the SV package, the C header, mmio/vu/at.hpp and tb Python; --check in lint | One source for RTL, simulator, firmware and benches, with no PyYAML dependency |

### 6.2 VU — vector unit and approximate math (`02-vector-unit.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-VU-1 | Operand model | rows × m with line-aligned rows, per-operand line stride (0 repeats), per-row scalar c from an s register or an SPM row vector, for every VU op | Every per-head and per-token pattern becomes one op, closing the 18 K ops/token gap |
| V2-VU-2 | Reduction order | Spec's 128 partials and complete binary tree per row; skipping unwritten subtrees passes z(q) (−0 → +0), which is exact | Keeps the spec contract and cheap short rows; z() fixes the −0 case that FTZ creates |
| V2-VU-3 | Segmented RED | Per-row SPM vector (or an s register for one row) with fused post-ops RCP and RMS (exact SQRT and RECIP) | Per-head norms and softmax sums in one op; RMSNorm in 2 ops instead of 7, bit-identical |
| V2-VU-4 | EXP algorithm | Fixed-point integer EXPK with f32 in/out, 2³⁸-scaled log2e constants, 64-entry 2^(j/64) table, cubic Taylor term | At most 0.53 ulp (model), about 600 LUT per lane, exact portable definition |
| V2-VU-5 | Reciprocal | All reciprocals correctly rounded (= tsr_frecip); RECIP_APPROX removed | Exact rounding is cheap with one correction product and needs no table specification |
| V2-VU-6 | SIN/COS | Exact 64-bit 2/π integer reduction, f32 Cephes polynomials, identity path below 2^-12 | Angles reach 2^18 where f32 Cody-Waite loses precision; the reduction runs once per angle |
| V2-VU-7 | Function set | SILU, SWIGLU, SIGMOID, GELU/GEGLU (tanh as x·σ(2u)), GELU16/GEGLU16, TANH, EXP, EXPSUB(C), RECIP, MIN, MULCB/MULCADD/ADDMULC, SETC | Covers Qwen2.5 and Gemma 4 dense with one shared EXP + RC datapath |
| V2-VU-8 | Lane split | 16 front lanes (incl. EXP) at full rate, 8 back lanes for SILU/GELU/SIGMOID/TANH/RECIP at half rate | Half-rate activations cost under 0.15 % of token time; EXP stays full rate for softmax |
| V2-VU-9 | Flow control | Credit-based reads against output-FIFO space, no clock enable in the lanes, fixed latency classes | Avoids a clock enable fanning out to about 45 K FFs at 276 MHz or more |
| V2-VU-10 | SPM access | CC's pair reads (alternating a-pair and b-pair cycles), no operand parity rule, no operand caches | Two sources stream at full rate whatever their alignment |
| V2-VU-11 | s-register access | One bank, written only by VU ops (SOP.MOVI for constants); read-only for the RISC-V and the host at SREG0-15, core CSR 0x9C0 | Writes stay ordered by queues and deps; one read location for every reader |
| V2-VU-12 | s-register reset | +0.0 at hardware reset; kept across SOFT_RESET, RISC-V reset, QABORT and QCLEAR | EX's requirement; kernels initialise what they read |
| V2-VU-13 | Attention mask | VMASK 0x38 specified but deferred to M9 (ILLEGAL_OP before); llama.cpp masks use AT's mask summaries | VMASK assumes cell j holds position k0+j, which llama.cpp's ring-shaped, reorderable caches break |
| V2-VU-14 | TOPK | Fully specified (rows, padding, total order); simulator in M6, RTL in M8 with a running-threshold filter | M6/M7 sample on the host; keeps 4 K LUT out of the first k4 link |
| V2-VU-15 | DQROW | Opcode 0x12 re-specified as SPM to f32 on the VU after an LD of the repacked row; simulator in M6, RTL in M8 (CAPS bit6 = 0 in M6) | The VU has the fp datapath; the embedding lookup stays on the CPU through M7 |
| V2-VU-16 | Error reporting | OPERAND (13) with the offending value for shape limits; enumeration mismatches are ILLEGAL_OP | Follows EX; the values make failures self-explaining |
| V2-VU-17 | Field widths | 24-bit line addresses and m; 16-bit rows, strides, head counts and T; function codes and s indices in x16 | Every op fits one 256-bit instruction with headroom |
| V2-VU-18 | Softmax ops | No generic fused online-softmax VOP; AT's vector ops (ASMAX, AMASK, AMERGE, ANORM) on the VU, two VU ops per chunk | AT fixed the dataflow; its ops fuse what the 7-op composition needed |
| V2-VU-19 | ROPE shape and angles | T tokens with a position vector; angles by ggml-cpu's recurrence θ_{i+1} = FM(θ_i, TS), FD by freq factor, FM by freq_scale; A-table form only for YaRN; head_dim multiple of 16 (≤ 1024), n_dims multiple of 32 (NEOX) or 16 (NORM) | Angles bit-identical to the CPU for ext_factor = 0 (the A-table form differed by up to 2.9·10⁻³ rad at 32 K) |
| V2-VU-20 | Accept-ahead | Up to 2 VU ops held; RAW, WAW and s-register hazards checked by extent intervals (CC's I3 ≤ 2 for unit 3) | Short independent ops overlap their 20-64-cycle fill and drain |
| V2-VU-21 | EXPSUB with −∞ | EXPSUB(−∞, x) = +0, even when x is −∞ or NaN | Fully masked blocks stay finite without an ordering rule |
| V2-VU-22 | MAX/MIN | Spec total order; MAX propagates NaN, MIN returns the non-NaN operand | One comparator serves MAX, MIN and TOPK, consistent with the spec |
| V2-VU-23 | Raw operations | COPY, SETC, MOVI/LDS/STS, VMASK pass-through and the ROPE tail copy bits without fp canonicalisation | COPY serves any 32-bit data; positions and indices pass unchanged |
| V2-VU-24 | Reduction unit size | 8-input subtree (7 FA) plus 2 scheduled FA; rows shorter than 256 elements reduce at half rate | Saves about 5.5 K LUT; the half-rate cases are short rows |
| V2-VU-25 | Accuracy contract | Functions defined by their algorithms (bit-exact); error bounds informative and measured | Bit-exactness between simulator, RTL and card is the verification basis |
| V2-VU-26 | CPU-compatible GELU | GELU16 (0x17) and GEGLU16 (0x18): clamps at ±10, binary16 rounding of input and result; llama.cpp GELU/GEGLU map to them; exact GELU kept for M8/PyTorch | The pinned CPU rounds every GEGLU element to fp16 (up to 3.9·10⁻³ relative); matching its form removes that gap for about 0.6 K LUT |
| V2-VU-27 | AT vector ops on the VU | ASMAX, AMASK, AMERGE, ANORM run on unit 3 with the front lanes, the MAX path and a multi-pass AMERGE (about 4-5 cycles per output line at C = 4) | Reuses the EXPK lanes and pair reads; matches AT's 1,600-cycle merge tail for about 2 K LUT |

### 6.3 AT — attention and the KV cache (`03-attention-kv.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-AT-1 | Where attention computes | QKᵀ and PV in the MVM clusters (one channel each); the VU only does softmax passes and merges | Only the clusters consume the KV cache at the 256 B/cycle DDR rate; one SPM port is 64 B/cycle |
| V2-AT-2 | KV values | Exactly llama.cpp's q8_0 (blocks along each head row, fp16 scale, quantize_row_q8_0_ref rounding); only byte placement changes | Device KV equals the CPU's; state save/load stays lossless |
| V2-AT-3 | Channel distribution | 16-cell groups round-robin over channels (group u in channel u mod C) | Balances channels at any context; one score line per group and row |
| V2-AT-4 | Plane layouts | K token-major in Q8_0 chunks; V slice-major (each 64-value slice of all cells contiguous) | Whole scores per pass for K; one DSP accumulator per slice for V |
| V2-AT-5 | KV store op | KVST (DMA unit) stores QACT A32 output into consecutive cells; partial meta beats by read-modify-write inside the DMA | QACT is already bit-exact quantize_row_q8_0_ref; no write strobes or boundary change |
| V2-AT-6 | PV numerics | Block floating point per 64-cell channel segment: 24-bit integer weights, exact integer sums in the DSP P register, one f32 add per segment | Reuses the 512 MAC DSPs and is at least as precise as the CPU's f32 path |
| V2-AT-7 | PV segment length | 4 local groups = 64 cells per channel | 7.5 conversions per cycle per cluster instead of 30 for 16-cell segments |
| V2-AT-8 | Softmax structure | Two-pass within a chunk (max from ATTQK, EXP by ASMAX); online merge across chunks (AMERGE) | Per-cell rescaling is impossible in DSP accumulation; chunks fit the slot buffers and pipeline |
| V2-AT-9 | Scale placement | Inside ASMAX and AMERGE as EXP((s − m)·scale); ATTQK returns raw scores | One rounding as on the CPU, free in an op that reads every score |
| V2-AT-10 | Row sums | ATTPV computes l with the same block-float rule (LSUM flag) | Saves a read pass over the scores and keeps l consistent with O |
| V2-AT-11 | Channel partials | ATTPV writes C per-channel partials; AMERGE adds them in a fixed pairwise tree; results depend on NUM_CH | Avoids serialising the clusters; bit-exactness is per configuration |
| V2-AT-12 | Row max floor | M_FLOOR = -2^100 (0xF1800000) in ATTQK MAXOUT and AMASK | Keeps merges finite for rows with no visible cell |
| V2-AT-13 | Empty rows | ANORM multiplies by RECIP(max(l, 2^-126)) | Reproduces the CPU's S == 0 ? 0 : 1/S at no cost |
| V2-AT-14 | Mask handling | Host-built mask summary (≤ 2 cell runs per row) with CAUSAL/SLIDE bounds in ATTQK; f32 copy and AMASK only for GENERAL masks | The VU has no f16; the cached op list stays valid; masked bandwidth is skipped |
| V2-AT-15 | GQA and prefill mapping | MVM slots = (query token, head in the KV-head group); head subsets (hg0, GS) when G > T_MAX; T_q = floor(T_MAX/GS) | K streams once per pass and every G works (Gemma 4 12B global layers: two passes) |
| V2-AT-16 | Units and opcodes | KVST 0x14 (DMA), ATTQK 0x21 and ATTPV 0x22 (MVM), ASMAX/AMASK/AMERGE/ANORM 0x34-0x37 (VU); unit 4 absent in M6; no new error codes | The ops use existing datapaths; a separate queue would only add arbitration |
| V2-AT-17 | Hadamard KV rotation | Off (LLAMA_ATTN_ROT_DISABLE=1 everywhere, V2-SW-16); the bit-exact FWHT recorded in Appendix A | Baselines are fixed with rotation off; q8_0 gains little; it can return in M8 |
| V2-AT-18 | Unsupported FA features | No attention softcap, ALiBi, sinks, sparse n_kv_max, DK ≠ DV, D outside 64-512, K-shift in M6/M7 | No target model uses them; each can be added later without a layout change |
| V2-AT-19 | Chunk policy | NCG = min(128 groups, PCAP limit, at least 2 chunks for ≥ 8C groups), segment-aligned; the chunk with this launch's new cells last | Overlaps VU and MVM work; deterministic, so part of the numeric contract |
| V2-AT-20 | DQROW | Per VU §5.6 (SPM to f32 on the VU, RTL in M8); M8 embedding = LD + DQROW; an f32 tok_embd copy for other GGUF types; never used by attention | VU owns the encoding; the lookup stays on the CPU until M8 |
| V2-AT-21 | KV region | Symmetric planes allocated top-down from each channel's end; one plane per (layer, stream, K/V); LG from kv_size | Equal channel shares; KV growth stays away from the weights |

### 6.4 CC — control core, issue path, SPM, CDC, floorplan (`04-control-core.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-CC-1 | Ibex pin and configuration | OpenTitan-locked Ibex commit, unmodified; RV32IMC+Zicsr+Zifencei, RV32MFast, RegFileFPGA, no PMP/ICache/predictor/WB stage, 2 triggers, MHPMCounterNum 0 | A silicon-signed-off state in the verified small configuration |
| V2-CC-2 | Debug module | Upstream riscv-dbg dm_top without its JTAG DTM; DMI from CSR 0xC00-0xDFC; SBA connected | D18 needs a standard DM; the debugger speaks DMI directly |
| V2-CC-3 | TCMs | ITCM 128 KiB + DTCM 64 KiB, RAMB36 32K × 1 columns, read latency 1 (48 RAMB36) | Executor, two model families, M8 and debug fit without overlays |
| V2-CC-4 | SPM line-buffered alias | 0x5000_0000 alias whose loads hit a one-line buffer, invalidated by SPM stores, window accesses, interrupt edges and SPM_LB_INV | Op lists read about 7× faster while keeping the memory model |
| V2-CC-5 | Issue port | MMIO staging (8 words, 2 deps, push with word mask); software checks credits; 65,536-cycle hardware safety net, suspended under dbg_stop | Ibex stays unmodified; a blocking push would block interrupts and halts |
| V2-CC-6 | Unit ids | 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT engine (absent in M6), 5 SYNC; 6-13 reserved, 14 DB marker, 15 none | Keeps six units, six counters and the CDC widths |
| V2-CC-7 | Opcode to unit mapping | Static UNIT_OF generated from EX §7.2; ISSUE_UNITS and Q_DEPTH report built units; no ISSUE_QMAP | Software has the table at compile time; one mapping fewer to keep consistent |
| V2-CC-8 | Queue depths | DMA 8, QACT 4, MVM 4, VU 16, SYNC 4 (unit 4 absent; parameter 8); issue FIFO 32 | Many short VU ops want a deeper VU queue for about 100 LUT |
| V2-CC-9 | Checker | One tsr_chk from cmdproc's checks, 8 cycles per op, EX's check order with OPERAND, four extents per class tagged SPM/DRAM and read/write, shift+DSP MVM extent | 34 M ops/s is at least 3× the issue rate; four-SPM-operand ops exist |
| V2-CC-10 | Dispatch to units | Queue-head command registers drive the units; an entry never starts in the cycle it becomes head | Clock enables stay one flop away; DB gets one cycle to hold |
| V2-CC-11 | Runtime unit errors | Withdrawn: units have no error output; every error is found at dispatch | EX's rule that no op fails after it started keeps the first error schedule-independent |
| V2-CC-12 | SPM second read path | Even/odd line-interleaved banks, 2R2W, per-bank client ports, registered round robin, 6 clients, 128 URAM unchanged, single-port adapter for v1 clients | A second read path without URAM; line pairs always hit both banks |
| V2-CC-13 | DMA LD flow control | Burst reservation with a 512-beat queue (8 RAMB36); rd_rsp_ready constant 1 | SPM contention can never stall a channel's response path |
| V2-CC-14 | s registers | One bank in the VU, read-only at SREG0-15 (core CSR 0x9C0) for the host and the RISC-V; writes only through ops | One location for every reader; writes ordered by queues |
| V2-CC-15 | Core window | Read-only view of the whole CSR map at 0x2000_1000 without snapshot side effects; LOAD_DATA and DMI read 0 there | One mechanism for ID, s registers, PERF and records; cannot disturb loads or the debugger |
| V2-CC-16 | Debugger memory path | DM SBA as a second master on the RISC-V data bus; no host debug SPM client | Reaches TCM and SPM without halting; the SPM keeps six clients |
| V2-CC-17 | Log path | EX's 256-word ring; CC part: LOG_PUSH into a 16-entry CDC FIFO, LOG_FREE from a clk_rv push count and a handshaked LOG_RD | Never stalls, never torn, works with wedged queues |
| V2-CC-18 | RISC-V clock | ap_clk_2 at 166.7 MHz via a v++ clock request, else BUFGCE_DIV ÷ 3 of the 500 MHz platform clock; fully asynchronous CDC | D5/D12; independent of core-clock scaling |
| V2-CC-19 | Kernel control protocol | ap_ctrl_none per EX §9 (fallback: declare ap_ctrl_hs, never run); xrt::ip and polling; MEM_EN latch; no interrupt on the U200 | XRT 2.18 has no user_managed type and routes AP_CTRL_NONE to xrt::ip |
| V2-CC-20 | ID and VERSION | Withdrawn: ID 'TSR1' and VERSION 2.0 per V2-EX-12 | v1 code already refuses a different ISA major |
| V2-CC-21 | PERF counters | Withdrawn: EX §6's counter space; CC adds X+20 SPM_RD_WAIT and X+21 SPM_WR_WAIT | One counter space with protocol-defined gating |
| V2-CC-22 | Recovery and reset | No partial datapath reset and no op cancel; SOFT_RESET per EX via the SR sequencer; a wedged unit needs a card reset | Unit resets with outstanding AXI bursts break the arbiter contracts |
| V2-CC-23 | Timing gate (D13 restated) | T1 control core OOC at 166.7 MHz; T2 k4 core OOC at 3.333 ns after the OOC VU run; T3 k4 link ≥ 276.3 MHz, target 300 MHz; T4 k1 link 300 MHz | Restated D13; 300 MHz is pursued by the cycle-identical gather fix |
| V2-CC-24 | Floorplan | C'': control core, issue path, VU and DB next to the SPM in SLR1; SLR1 ≤ 65 % LUT, else VU shrink knobs, then DB optional blocks off, then VU lanes to SLR0 | The recomputed budget (62-71 %) straddles the gate; every remedy keeps results unchanged |
| V2-CC-25 | Gather job-pipe fix | tsr_rslice FORM 1 (cycle-identical) for the central job-pipe stages plus pinning stage 0; JOB_PIPE 3 only if still short | Removes the measured worst paths without changing any cycle |
| V2-CC-26 | Memory source 0 | Freed by the fetch removal, renamed TSR_SRC_AUX, tied off in M6 | Keeps the arbiter shape; nothing claims it yet |
| V2-CC-27 | CDC verification | Verilator two-clock benches over a ratio sweep, a jitter mode in tsr_sync, report_cdc, zero unconstrained ap_clk/ap_clk_2 paths | Verilator cannot model metastability; jitter exercises skew |
| V2-CC-28 | Register table | CC's rows in sw/tables/tsr_tables.py (V2-EX-30) | One source for every area |
| V2-CC-29 | RV_HALTED detection | Derived from the last instruction-fetch address in the DM range; cross-checked against dmstatus; DMI fallback | Keeps riscv-dbg unmodified |
| V2-CC-30 | LQ/LCQ/log CDC | The RISC-V reaches EX's rings only by tagged handshakes (reads return core-domain values; pop and post commit in the core domain); no gray-coded copies of multi-step counters | Host-written LCQ_RD/LOG_RD and SOFT_RESET flushes jump by more than one step |
| V2-CC-31 | SR sequencer | Core-domain sequencer for EX's SOFT_RESET (RISC-V reset within 2^16 cycles, drain, clear list, SR_BUSY), independent of clk_rv | The host needs an atomic known-state operation even without clk_rv |
| V2-CC-32 | ISSUE_PIDX | clk_rv counter of ticketed pushes at control window 0x03C, equal to the dispatcher's pidx; +1 per ISSUE_PUSH and per QCLEAR only | Exact and timing-independent like ISSUED_SEQ, for about 40 FF |
| V2-CC-33 | SBA write refusal | Every SBA write to the control and core windows refused in hardware, and SBA TCM writes while RV_RUN = 0 | LQ_POP, LCQ_POST, LOG_PUSH and TRACE_MARK have protocol effects; a whole-window rule is simple to verify |

### 6.5 DB — debug and profiling (`05-debug.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-DB-1 | GDB transport | Built-in RSP server (libtsr_dbg) driving DMI through the CSR window, embedded in the device-owning process or standalone tsr-dbg | One CSR access per DMI access; no OpenOCD and no second process |
| V2-DB-2 | DMI read side effects | DBG_CFG.DMI_EN gates the DMI window (reads 0 and writes dropped when off) | Keeps EX's no-read-side-effect rule for every other agent |
| V2-DB-3 | Debugger memory access | SBA for all memory access; the program buffer only for fence.i | Works without halting and never changes hart state |
| V2-DB-4 | D15 enforcement | Hardware refuses every SBA write to the control and core windows (V2-CC-33); the server never writes the window nor stores to it from the program buffer; issuing inferior calls unsupported | No debugger path can push a tile op |
| V2-DB-5 | Halt requests from DB | DB's request ORed into Ibex debug_req_i, one-shot via a toggle, cleared when the hart halts | Tile stops and watchpoints halt the hart without a DMI round trip; no re-halt |
| V2-DB-6 | Breakpoints and watchpoints | 2 Ibex execute triggers, unlimited ITCM software breakpoints, 2 DB data watchpoints on the Ibex data port | Ibex triggers are execute-only; GDB software watchpoints are unusable over PCIe |
| V2-DB-7 | Halt at boot and launch | Boot: DB halt request pending at reset release; launch: runtime hook ebreak selected by __tsr_dbg | Stops exactly at the first instruction or kernel entry for two loads per launch |
| V2-DB-8 | Tile stop mechanism | Push-index threshold T held at the queue heads (dbg_hold); breakpoint matches registered at enqueue | Yields exactly the push-order prefix the sequential simulator reproduces |
| V2-DB-9 | Hard hold | dbg_hold also blocks non-running completions, except after a QABORT (until the next QCLEAR) and during SOFT_RESET, which clears the stop state | Keeps invariant G at stops while every reset still drains |
| V2-DB-10 | Serial mode | Head holds keep only the oldest outstanding entry runnable: one op at a time in push order | Same mechanism as stops; the op stream is untouched |
| V2-DB-11 | Dispatch history | 512-entry ring of full ops, deps, ticket, push PC and time; recording freezes 256 entries after the first error; exportable as push.bin | A hang needs the ops that never completed; the evidence replays |
| V2-DB-12 | Log ring and record header | EX's 256-word ring, whole records dropped, BLOCK mode; the one header: [7:0] N, [11:8] LEVEL 0-5, [15:12] SOURCE, [18:16] TYPE, [19] TS, [27:20] LSEQ, [31:28] 0xA | Never-torn records need no resync; one header for the encoder and every decoder |
| V2-DB-13 | Crash path | With a debugger: restore and ebreak at __tsr_trap_halt (shown as SIGSEGV etc.); without: __tsr_crash, TRAP log record, EX's TRAP record, park; panic = mcause 24 | GDB shows faults natively; the record survives without a debugger |
| V2-DB-14 | Push PC | Ibex current_pc captured at ISSUE_PUSH and carried to the history ring (+32 bits per issue-FIFO entry) | Every tile op maps back to its C++ line |
| V2-DB-15 | PERF semantics | EX's ACTIVE/QHALT gating and counter list; DB adds TRW_RECORDS, TRW_DROPS, DBG_STOP_CYCLES in slots 61-63 | Exact windows; debugged measurements are detectable |
| V2-DB-16 | Timeline transport | Hardware trace writer: 32-byte records to a DRAM ring through a second tsr_mem_arb write source on TRW_CH = 0 (ring in the ch0 IO region); REC stays for firmware | MMIO and firmware streaming are too slow or perturb the measured phases |
| V2-DB-17 | Who runs the server | The device-owning process (embedded service) or tsr-dbg when nothing else owns the device | xrt::ip contexts are exclusive |
| V2-DB-18 | Simulator debug model | Behavioural DM covering the DMI profile, with exact stop, history and trace-writer semantics | Behaviour is identical where the server looks |
| V2-DB-19 | Watchdog | EX's progress watchdog; DB adds HUNG_STOP, a history freeze and per-unit AGE probes | One watchdog is enough |
| V2-DB-20 | Recovery | 5-level ladder: dump, debug release, LQ_CANCEL, SOFT_RESET with warm restart, card reset; DB's evidence survives every level below a card reset | Post-mortem data exists before any destructive step |
| V2-DB-21 | Host access to the s registers | SREG0-15 at 0x9C0-0x9FC (in the DBG range, present whenever the VU is built), the one read location for host and RISC-V | EX uses 0x0C0 for LOAD_ADDR; one location for every reader |
| V2-DB-22 | SPM bulk dump | A replay launch of ST ops, then a bo read; SBA for small reads and the dead-firmware case | D15-compliant with no new kernel and no SPM client |
| V2-DB-23 | Register placement | DBG block at 0x800-0xAFC, RISC-V control window 0x160-0x17C and 0x400, PERF slots 61-63 | Inside the reservations of EX and CC |

### 6.6 SW — software, simulator, verification, work plan (`06-software-verification.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| V2-SW-1 | RISC-V model in the simulator | In-house RV32IMC+Zicsr interpreter tsr_rv in sw/sim; Spike only as a test oracle on a separate test bus profile at 0x8000_0000 | One deterministic C library usable in the backend; Spike anchors ISA correctness |
| V2-SW-2 | Simulator time model | Event-driven on core cycles; fixed schedule by default, random:<seed> for determinism tests; spin and WFI fast-forward | Exposes schedule dependence cheaply; bounds the polling cost |
| V2-SW-3 | Race detector policy | Vector clocks; fatal in ctest, off in model runs, report nightly | The cost matters only on long model runs |
| V2-SW-4 | Launch state | Launches are stateless (image, args, DRAM); SPM_STALE check; fresh-boot replay of every captured trace | Every trace is self-contained and replayable |
| V2-SW-5 | Trace v2 layout | One directory per launch; single push-ordered ops/push.bin (48-byte records, relative tickets); images shared by id | Replay needs the global push order |
| V2-SW-6 | v1 trace migration | Regenerate m2/m3/gen with tsr-trace-gen v2 (same seeds); convert the 290 model traces to serial chains (traces/model-v1conv, replay SERIAL); drop traces/tmp | 254 error expectations must be re-derived anyway; the model traces check v1-to-v2 unit equivalence |
| V2-SW-7 | Toolchain | xPack riscv-none-elf-gcc 14.2.0-3 (with GDB) in user space, the only supported compiler; Spike and dtc from source; Clang optional | No sudo, pinned; distro GCC lacks the rv32imc multilib and libstdc++ headers |
| V2-SW-8 | Firmware language rules | Freestanding C++20, no heap/exceptions/RTTI/libstdc++ headers, -Os, libgcc only; no f32 math in firmware | No soft-float path; the host computes every constant |
| V2-SW-9 | Image identity | EX §3.2: 8-byte .tsr_id = first 8 SHA-256 bytes (little-endian uint64) over PT_LOAD; echoed in HELLO r2/r3 | Identifies the build; CRC and VERIFY prove the transfer |
| V2-SW-10 | Host/RISC-V protocol | Withdrawn: EX's LQ/LCQ protocol with runtime contract R1-R10, up to LQ_DEPTH launches outstanding | EX owns the launch protocol |
| V2-SW-11 | Logging | Binary FMT records (format-string address + args) with DB's header through EX's log ring; whole-record drop with a counter; runtime filter LOG_CFG.LEVEL | No TCM spent on strings; readable when queues wedge |
| V2-SW-12 | Graph execution model | Host builds a pre-resolved ggml-level op list (OPL v1) with host-computed dependencies, cached per graph; firmware kernels per kind | No register-relative addressing needed; the host does the ggml work once per graph |
| V2-SW-13 | Data placement in M7 | Tensors in DRAM between ops; per-op SPM scratch ring with release tickets; bias/residual fused via MVM ACC, RMS_NORM × weight fused | Simple and race-free; the fusions are bit-identical |
| V2-SW-14 | M6 llama.cpp path | The graph executor with MUL_MAT-only op lists, one launch per v1 split | One mechanism from M6 to M7; like-for-like gates |
| V2-SW-15 | Index inputs | Preload KV write indices, row ids, positions and each KQ mask's MSUM at launch start with one join | Avoids a join per SET_ROWS or FA op |
| V2-SW-16 | llama.cpp Hadamard KV rotation | LLAMA_ATTN_ROT_DISABLE=1 for device runs and every baseline; device rotation is a follow-up | The rotation matrix is a graph input that MVM cannot use |
| V2-SW-17 | KV cache buffers | Recognise llama.cpp KV buffers by tensor names; AT's layout in a KV heap; converting get/set/clear; no context shift in M7 | Names are the stable signal; state save/load keeps working |
| V2-SW-18 | Host protocol layer | One protocol implementation of EX §10 over all kdevs; sim is a kdev with idle() | The card's host code path is tested on every sim run |
| V2-SW-19 | Test access to the queues | tsr_sim_push / tsr_sim_qctrl hooks in the simulator library only, never through tsr_transport | Directed tests without firmware; D15 holds for every real path |
| V2-SW-20 | Planning code | Freestanding plan.hpp (tiling, QA_LINES, ROW_BEATS, scratch needs) compiled for host and firmware | supports_op rejects exactly what the firmware cannot plan |
| V2-SW-21 | Test models | Synthetic tiny Qwen2 and Gemma 4 GGUFs generated at build time; real Qwen 0.5B/32B and Gemma 4 12B (31B when available) for gates | Fast ctest coverage of SWA, per-head norms and GEGLU without binaries in git |
| V2-SW-22 | Baseline settings | FA on, KV q8_0/q8_0, rotation off for device and CPU runs from M6; offload_kqv false in M6, true in M7 | D17; baselines stay constant from M6 to M7 |
| V2-SW-23 | Approximate-math verification | Exhaustive 2^32 bit-exact check vs an independent numpy model, ulp bounds vs double, RTL exhaustive opt-in; GELU16 over all 65,536 fp16 inputs vs the pinned table | Unary f32 functions are fully enumerable |
| V2-SW-24 | Launch watchdog | 60 s default, counted only while not halted and no tile stop is active, off in SERIAL; transport mutex released between polls; DB's ladder on timeout | A GDB breakpoint inside llama.cpp must not trigger a device reset (D18) |
| V2-SW-25 | Kernel test builds | Host build (ASan/UBSan) plus RV build on tsr_rv (the reference); ctest runs both | Fast logic checks; the RV build proves control flow and ABI |
| V2-SW-26 | Compute-buffer placement | Per-channel compute heap between weights and KV planes sized from the reserve graph, plus ch0 IO and ch1-3 low 512 MiB; get_max_size ≥ largest tensor | Gemma 4's 512 MiB logits chunk would fill a fixed window and add a launch per token |
| V2-SW-27 | numcheck on v2 | Rely on cb_eval per-node observation (one launch per node); no special backend mode | No second code path to keep correct |
| V2-SW-28 | Card selftest | selftest kernel with numbered sub-tests; tsr-selftest compares with the same image on the in-process simulator | D15 requires kernels; exact expectations without golden files |
| V2-SW-29 | Op-list size cap | At most 4096 ops per launch (32 KiB of DTCM tables); the builder splits larger graphs | Static DTCM use; decode needs about 1.3-2 K ops |
| V2-SW-30 | Timing independence of firmware | Timing-dependent reads never influence pushes, deps, stores or records; ISSUE_PIDX and ISSUED_SEQ are exact | Makes card logits bit-identical to the simulator |
| V2-SW-31 | GELU in llama.cpp graphs | GGML GELU and GEGLU map to the VU's GELU16/GEGLU16; exact tanh GELU for M8/PyTorch | Matches the CPU's fp16-table form |
| V2-SW-32 | Mask summaries | Backend writes the MSUM (and an f32 copy for GENERAL masks) on set_tensor of 'kq_mask' tensors, reserved via get_alloc_size; FA refers to it via the preload table | The cached op list stays valid per token; the device never reads F16 |
| V2-SW-33 | Unselected graph branches | DUP, non-KV SET_ROWS and nodes without GGML_TENSOR_FLAG_COMPUTE unsupported; graph_compute skips flagless nodes | Prevents a second split and launch per token |

---

## 7. Review findings not applied, or applied differently

All 42 findings were evaluated. Two were not applied because another finding asked for an
incompatible fix of the same problem; the others below were applied with a deliberate deviation.

| Finding | Asked for | Integration | Why |
|---|---|---|---|
| [19] (major) | the single table source = SW's `sw/sim/regs/tsr_regs.yaml` + `gen_regs.py` | **not applied**; [29]'s `sw/tables/tsr_tables.py` + `gen_tables.py` chosen (V2-EX-30) | Both findings agree on one source with SW owning the generator. Plain Python needs no PyYAML in the RTL, firmware and bench builds, and `sw/tables/` reflects that RTL, simulator, firmware and benches all consume it |
| [30] (major) | the push index kept in software (a DTCM counter initialised from `Q_PIDX`) | **not applied**; [10]'s hardware `ISSUE_PIDX` at control window 0x03C chosen (V2-CC-32) | A `clk_rv` counter is exact and timing-independent like `ISSUED_SEQ`, costs ≈ 40 FF and stays right even for code that pushes outside the SDK counter. [30]'s normative rule of which entries advance the index (+1 per push and per QCLEAR; not QABORT, QHOLD, TRACE_MARK) is adopted in CC §5.1 |
| [13] (major) | a periodic snapshot handshake (every 64 `clk_rv` cycles) for `LCQ_RD`, `LOG_RD` and the flushed counters | applied with a simpler mechanism | Every RISC-V read of an LQ / LCQ register is itself a tagged handshake returning the core-domain value, and pop / post commit in the core domain; `LOG_FREE` = depth − (push count in `clk_rv` − handshaked `LOG_RD`). Exact, never over-reports, no periodic traffic (V2-CC-30) |
| [12] (major) | `dbg_hold` ignored while `SR_BUSY` or "while `RV_RESET_DONE` is rising"; `SOFT_RESET` clears `SERIAL` | applied with a broader rule | The hard hold yields after any QABORT until the next QCLEAR (covering the RISC-V reset, `LQ_CANCEL`, host QABORT and `SOFT_RESET`) and is forced off while `SR_BUSY`; `SOFT_RESET` clears the stop state but keeps `SERIAL`, which cannot deadlock and is a user setting (`TSR_TILE_SERIAL`) |
| [25], [39] (minor) | meta `"q_depth":[8,4,4,16,8,4]` | `[8,4,4,16,0,4]` | Unit 4 is absent in M6 (V2-AT-16), so `Q_DEPTH[4]` reads 0 |
| [1] (blocker) | insert EX §4.5 into CC verbatim | CC §5.2 points to EX §4.5 and states only CC's implementation | One normative copy cannot drift; both documents are generated into the same table source |
| [6] (major) | copy AT §5.4–§5.7 into VU verbatim | VU §5.11 restates the encodings for the decoder and points to AT for semantics and checks | Same reason: AT stays the single definition of the semantics |
| [6], [40] | delete VMASK or state its use | VMASK kept as an encoding, deferred to M9 (V2-VU-13) | It is exact for the M9 resident loop's linear cache; deferring saves its area now |
| [42] (minor) | alternative: 1 GiB windows on ch1–3 | the compute heap (the finding's first option) | Sized from the reserve graph; no fixed window to outgrow |
| [2] (blocker) | PANIC as kernel event 0xC0 or a TRAP with `mcause` 24 | TRAP with `mcause` 24 | DB's crash contract already defines it; GDB shows it as SIGABRT |
| AT §14, SW §15 | spec and doc edits "applied by the integrator" | not applied now; listed in §11 as W0 | The integration task allows edits inside `docs/v2/` only |

---

## 8. Remaining risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| 1 | SLR1 LUT ≈ 62–71 % (up to 79 % with the VU at + 30 %) against the 65 % gate | k4 link below 276.3 MHz | out-of-context VU run before T2; ordered remedies: VU shrink knobs (−8.7 K), DB `HAS_TRW` / `HIST_DEPTH` off, VU lanes to SLR0 (CC §14.1) |
| 2 | k4 link timing with the control core, VU, AT cluster modes (DSP input mux, capture bank, gather ATT path) and DB | core clock below 276.3 MHz (D13 restated) | OOC runs per block, the cycle-identical gather fix, registered boundaries, half-rate ATTPV fallback (bit-exact) |
| 3 | AT cluster modes add ≈ 19–27 K LUT per cluster in SLR0 / SLR2 (≈ 50–54 % each) | congestion in the cluster SLRs | per-mode floorplan check in the first OOC run; trim the capture bank (numeric contract change before freeze only) |
| 4 | v++ 2024.2 refuses `ap_ctrl_none` with `m_axi` args or the `ap_clk_2` frequency request | no v2 xclbin | EX §9.1 fallback (`ap_ctrl_hs`, never run); BUFGCE_DIV clock; checked in the first smoke link |
| 5 | CDC or flush bugs in the rings, the SR sequencer or the issue FIFO | hangs that need a card reset | handshakes instead of multi-step gray counters, two-clock benches at every ratio with jitter mode, mutation campaigns, `report_cdc` |
| 6 | Ibex / riscv-dbg details differ from the design (`RV_HALTED` heuristic, triggers, `crash_dump_o`, SBA codes) | wrong halt detection or missing debug features | all marked *confirm*; `tb/ctrl` cross-checks against `dmstatus`; DMI fallback |
| 7 | GELU16 differs from the CPU's libm-based table at fp16 boundaries; Gemma 4 FA-on baselines are new | residual Gemma 4 KL gap | exhaustive fp16 enumeration and the KL with and without `GGML_GELU_FP16` before the VU RTL freeze |
| 8 | No Gemma 4 31B GGUF on the build host | M7 Gemma coverage limited to 12B | 12B is confirmed; 31B rows stay *confirm* until a GGUF is read |
| 9 | Issue-bound phases (short ops on small models, the `graph` executor's per-op cost) | DDR idles, mostly for 0.5B | ops cover all heads per op, line-buffered SPM alias, measured with `MVM_EMPTY` and TRACE_MARK; an "issue from SPM line" path later |
| 10 | Attention of D = 64 models is gather-bound (61 %), and the VU's merge tail is exposed per layer | 0.5B attention slower than DDR allows at long context | accepted for the test model; wider gather later |
| 11 | D16: no in-tree working path until v2 runs | no regressions on the card in between | `build/v1` and tag `v1-m4`; the software path runs end to end on the simulator before any RTL (W3) |
| 12 | 4-hour k4 links on a 14 GB host | slow hardware iteration | OOC runs and k1 links first; at most 2–3 k4 links in M6 |
| 13 | Simulator speed for 32B and Gemma runs with RISC-V interpretation | slow sim gates | fast-forward, race detector off in model runs, short prompts for identity runs |
| 14 | Toolchain availability (xPack multilib, Spike dependencies) | blocks W2 | libgcc fallback multilib; Clang via conda; Spike only for `rv_*` tests |
| 15 | A wedged unit has no cancel | only a card reset recovers | progress watchdog, DB evidence and dumps first (DB §7) |
| 16 | PCIe MMIO read latency unknown (spec open item) | launch round trip above 10 µs | still ≤ 1/3 of v1's 35 µs; M7 makes one launch per token |

---

## 9. Open items (measurements and third-party confirmations only)

No design decision is open. These remain:

- **Smoke link:** v++ 2024.2 with `ap_ctrl_none`, `m_axi` args and `sp=` connectivity; the
  `ap_clk_2` frequency option (EX §17, CC §19).
- **Vendoring:** the pinned Ibex and riscv-dbg commits and their parameter and port names, ROM and
  program-buffer addresses, `resethaltreq`, Access Memory, SBA and DMI codes, `crash_dump_o`,
  `tdata1`, `dcsr.stopcount` / `stepie` (CC §19, DB §16).
- **XRT 2.18:** `xrt::ip` with BO bank lookup without an `xrt::kernel`; attaching to a compute unit
  left by a dead process (SW §15, DB §16).
- **Toolchain:** the xPack release and multilib set, Spike's build dependencies, riscv-tests make
  variables (SW §9.6).
- **Measurements on the card:** MMIO read and posted-write latency, launch round trip and pipelined
  rate, cold and warm load times, DMI and SBA times, log drain, TRW host read time, the RISC-V
  run-ahead margin, KVST's scattered-write DDR cost.
- **Measurements in synthesis:** `tsr_ctrl_rv` Fmax (fixes F_RV per D12), the VU and AT cluster
  utilisation.
- **Numerics:** exhaustive accuracy figures for EXPK, SIGMOID, SILU, GELU, TANH, SIN / COS (gate:
  ≤ 2 × the *model* figure); the GELU16 mismatch rate; the Gemma 4 FA-on KL baselines.
- **Models:** Gemma 4 31B dimensions; whether M8 brings back the Hadamard KV rotation and an f32
  `tok_embd` table.
- **Capacity:** re-check `LQ_DEPTH` = 8 against M7 and M8 launch patterns.
- Observation: `git status` shows `third_party/llama.cpp` with modified content; the v2 design
  work made no change there.

---

## 10. Implementation work packages (ordered)

The software path runs end to end on the simulator before any v2 RTL is needed; the RTL follows
the area documents in parallel and is checked against the simulator; the card comes last. Package
numbers P* refer to SW §11 (sizes and dependencies there).

**W0 — contract and clean slate (first M6 commits).**
1. X0 (EX §12, P0): delete every v1-only part of every layer in one commit; remove the dependent
   ctests and benches (they return with their v2 replacements).
2. P1.1: the table source `sw/tables/tsr_tables.py` with every row of EX §5–§8, CC §5, DB §10,
   VU §5, AT §5; `gen_tables.py` emitting the SV package, `tsr_tables.h`, `mmio/vu/at.hpp`, tb
   Python; `--check` in lint.
3. The spec and document edits of §11.

**W1 — simulator and references (P1.2–P1.13).**
1. P1.3 references first: `tsr_ref_math.c` (EXPK, composites incl. GELU16, SIN / COS) with the
   independent numpy model and exhaustive tests (`ref_math_*`, `ref_math_gelu16`); VU references
   (reductions with the z() rule, ROPE with both angle forms, TOPK, DQROW); AT references (plane
   layout, KVST, ATTQK, ATTPV block float, the vector ops, `tsr_ref_fa`).
2. P1.2 EX's CSR model (loader with VERIFY, LQ / LCQ / log rings, watermarks, `SOFT_RESET`,
   `Q_*`, PERF v2); P1.6 the queue model (checker with OPERAND and four extents, `ISSUE_PIDX`,
   I1–I3); P1.4 / P1.5 VU and AT unit semantics with cycle models.
3. P1.7 `tsr_rv` with the firmware and test bus profiles; P1.11 toolchain and Spike harness; P1.8
   scheduler; P1.9 race detector; P1.10 trace v2, `tsr-trace-gen` v2, `tsr-trace-conv` and the set
   regeneration / conversion; P1.12 the DM model; P1.13 DB's simulator models.
   Exit: `sim_basic`, `sim_queue`, `sim_vu`, `sim_at`, `rv_isa`, `rv_random`, `sim_race`,
   `sim_sched_determinism`, `trace_v2_roundtrip` green.

**W2 — firmware SDK and kernels on the simulator (P2.1–P2.7).** Build (`ExternalProject`,
`tsr-elf`), crt0 and the runtime R1–R10 with DB's trap and log contract, the header library and
generated builders, `plan.hpp`, the OPL executor, the M6 kernels (`graph` with MUL_MAT,
`selftest`, `replay`), then the M7 kinds (norms, element ops, GELU16, RoPE, GET_ROWS, SET_ROWS_KV,
FLASH_ATTN per AT §6.6) with their C++ references. Exit: `fw_runtime`, `fw_kernels_host`,
`fw_kernels_rv` (8 schedules, race fatal), `fw_selftest_sim`, `fw_replay_sim`.

**W3 — host, backend and tools on the simulator (P3.1–P3.8).** Transport v2 (EX §10, debug-aware
wait and recovery ladder; sim, mock, vkernel and xrt kdevs), the backend's M6 OPL builder, tools
(`tsr-selftest`, trace tools, `tsr-log`, klcheck / numcheck with FA on), **M6 on sim** (P3.4:
0.5B identity and the 32B KL on the simulator), the M7 backend (supports_op, KV buffers, MSUM
producer, compute heap), **M7 on sim** (P3.6: one launch per token for the tiny and real models),
`libtsr_dbg` / `tsr-dbg` (P3.7) and `tsr-prof trw` / TSR_STATS v2 (P3.8).

**W4 — RTL (P4), each block bit-exact against the simulator.**
1. Core-domain issue path: `tsr_chk`, `tsr_issue` (queues, heads, SYNC, hard hold), `tsr_csr` v2
   (EX map, rings, SR sequencer, loader with VERIFY); `tb/chk`, `tb/issue`.
2. SPM 2R2W with `tsr_spm_cport` and the DMA LD reservation; `tb/spm`, `tb/dma`.
3. Control subsystem: Ibex, riscv-dbg, TCMs, control window, CDC; `tb/ctrl`; gate T1.
4. VU (incl. AT's vector ops) with `tb/vu`; the out-of-context utilisation run fixes the knob
   setting against the SLR1 gate.
5. AT: KVST in the DMA, the cluster ATTQK / ATTPV modes, the gather ATT path; `tb/at`, `tb/tech`.
6. DB block (stop logic, history, trace writer, probes, DMI gating); `tb/dbg`.
7. Gather job-pipe fix (`tsr_rslice` FORM 1); adapter v2 (`ap_ctrl_none`, `MEM_EN`, `ap_clk_2`).

**W5 — integration on Verilator and Vivado (P5.1–P5.6).** Tile-only replay of every v2 trace set
into the issue FIFO; the full-system bench with Ibex RVFI lockstep against `tsr_rv` at every clock
ratio; vkernel v2; riscv-tests on RTL; 0.5B on vkernel bit-identical to sim; Vivado T2 (after the
VU OOC run), T4 (k1 link), T3 (k4 link ≥ 276.3 MHz).

**W6 — card (P6.1–P6.6).** Bring-up (identity, bind, load and VERIFY, HELLO, PING, launch round
trip and rate), `tsr-selftest` with the simulator shadow and the trace sets, the GDB smoke, **the
M6 gate**, then **the M7 gate**, then performance (DDR utilisation U with the H / G / F / E split,
launch costs).

---

## 11. Pending edits outside `docs/v2` (W0)

The integration edited `docs/v2/` only. These edits follow from the design and are part of W0:

| Document | Edit | Source |
|---|---|---|
| `docs/spec.md:323-392` "실행 모델과 제어 인터페이스" | replace with EX §2–§5, §8–§9 | EX header |
| `docs/spec.md:418-444` ISA table | retired rows (EX §7.2); KVST 0x14, ATTQK 0x21, ATTPV 0x22, ASMAX–ANORM 0x34–0x37; VU rows per VU §5; x16 | EX §7, AT §14, VU §2.2 |
| `docs/spec.md:288-321`, `:437-472` VU sections | replace with VU §4–§6 (RECIP_APPROX removed; EXP, SIN, COS, GELU16, RoPE angles defined) | VU §2.2 |
| `docs/spec.md:760-768` "명령 버퍼 경로와의 공존" | delete | EX header |
| `docs/spec.md:756` memory-model rule 5 | replace with EX §2.3 H1–H5 | EX §2.3 |
| `docs/spec.md:69-76` core boundary | `done` removed; `clk_rv`, `rst_rv_n`, `plat_err_i`, `host_irq_o` added | EX §9.3 |
| `docs/spec.md:130-138` DRAM map | CMD → OPL; KV planes top-down; compute heap; TRW ring | EX §11, AT §14 |
| `docs/spec.md:823-830` trace format | trace v2 with `push.bin` | SW §15 |
| `docs/spec.md:489` backend table | transport row | SW §15 |
| `docs/spec.md:493-498`, `:523`, `:599` | attention offload per AT §2.6 (M7), FA only | AT §14 |
| `docs/spec.md:984`, `:989`, `:994`, `:996` open items | KV layout closed (AT §4); ch0 beats exact; PCIe latency closes with the card measurements; unit reset closed (EX §8.6) | AT §14, EX §6, §8.6 |
| `docs/ctrl-riscv.md` §3.3–3.4, §6, §8.3–8.4, §10 | superseded by EX §3–§4 and DB; mark as historical | EX header, DB header |
| `docs/prof-dbg.md` command-path debug | superseded by DB | DB §1 |
| `docs/klcheck.md` settings | FA on, KV q8_0, rotation off, `offload_kqv` false in M6 / true in M7 | AT §14, SW §10.4 |
| `docs/rtl-m3.md` | cluster ATTQK / ATTPV modes | AT §14 |
| `docs/xrt-host.md`, `docs/platform-u200.md` | transport v2 and adapter v2 | EX §9–§10 |

# Tesseract design set (M6–M9)

Status (2026-11-02): **implemented on the C simulator; RTL integrated; card validation pending**.
The original design integration was on 2026-10-30. Six area documents, reviewed by two
reviewers and integrated into one consistent set. The current implementation map is §12;
the active build and remaining defects are in [../handoff.md](../handoff.md).
The locality candidate status and software compatibility plan are in
[../intermediate-locality-plan.md](../intermediate-locality-plan.md).
Every number marked *estimate* is unmeasured;
every third-party fact marked *confirm* is checked when the dependency is pinned or at the first
smoke link. Source-of-truth order: `docs/spec.md` (§11) > ISA simulator
(`sw/sim`) > `docs/design/*` > `docs/ctrl-riscv.md` > `docs/prof-dbg.md`.

---

## 1. Purpose and how to read

Every tile op is issued by an embedded RISC-V control core. The host loads a firmware image and
launches kernels; it never touches the issue port (D14, D15). This directory is the design
contract: M6 hardware and software, and the M7 graph execution that makes one llama.cpp decode
token one launch.

Read in this order: EX (the host-visible model and every global table), then CC (the RTL that
implements EX's model), VU and AT (the vector and attention compute), DB (debug and profiling),
SW (simulator, firmware, host stack, verification, work plan). Section 4 lists where each global
table lives; every other mention of a table is a pointer to that place.

| File | Area | Owns |
|---|---|---|
| `01-execution-model.md` | EX | launch protocol, host CSR map, opcode / unit / error-code / PERF tables, XRT adapter, transport API, DRAM map, build order |
| `02-vector-unit.md` | VU | VU ISA and bit-exact numerics (EXP, RECIP, SIN/COS, GELU, RoPE, reductions), VU microarchitecture, execution of AT's vector ops |
| `03-attention-kv.md` | AT | KV cache layout and values, attention ops and numerics, the normative flash-attention sequence, masks |
| `04-control-core.md` | CC | Ibex and debug module, RISC-V map and control window, issue path, SPM banks, CDC, resets, floorplan and timing |
| `05-debug.md` | DB | GDB over PCIe, tile-level debug, logs and crash records, recovery, profiler, DBG registers |
| `06-software-verification.md` | SW | simulator and references, trace format, firmware SDK and kernels, op-list format, llama.cpp backend, verification and work plan |

---

## 2. User decisions and milestones

**Decisions of 2026-10-30** (`docs/ctrl-riscv.md` D14–D18; they supersede D7–D11 and D13):

- **D14** — the host loads the firmware image and launches kernels that run on the RISC-V; every
  tile op is issued by the RISC-V. One s0–s15 scalar bank.
- **D15** — no host debug access to the issue port. Bring-up, the card selftest and trace replay
  run as RISC-V kernels.
- **D16** — one execution path in the tree. The software path runs end to end on the simulator
  before any RTL (§10).
- **D17** — the device KV cache is q8_0; attention arrives as llama.cpp `FLASH_ATTN_EXT` (FA on;
  the pinned llama.cpp requires FA for a quantized V cache). klcheck / numcheck baselines use
  FA on.
- **D18** — the design has full debug: a RISC-V Debug Spec module, halt / resume / step,
  breakpoints and GDB source-level debugging over PCIe (a GDB RSP server on the DMI window, no
  OpenOCD), plus tile-level debug.

**Still valid:** D1 simple in-order RISC-V without caches or speculation; D2 code, data and
stack in TCMs; D3 asynchronous issue of 256-bit tile instructions (issue and continue, join
later); D4 SPM mapped into the RISC-V address space; D5 a separate RISC-V clock domain; D6
kernels in C++ with stock GCC/Clang and a header library; D12 RISC-V clock planning value
166.7 MHz (250 if Ibex closes there, 125 fallback). Proposal kept: lowRISC Ibex RV32IMC + Zicsr +
Zifencei.

**Technical defaults chosen under the user's delegation** (kept by this design): even / odd
line-interleaved SPM banks for the second read path (CC-12); XRT `ap_ctrl_none` with polling
(EX-6); a host-readable log ring in the core domain (EX-24); D13 restated as "with the
control core the k4 link stays at or above 250 MHz" (CC-23, amended by the user on
2026-11-02); GELU (tanh form) in the VU for
Gemma, plus the CPU-compatible GELU16 for llama.cpp graphs (VU-26).

The current K4 run still requests a 300 MHz core and a 166.7 MHz RISC-V clock. The initial
clock checks verify that request and the separate clock/reset connections. Final T3 acceptance
requires at least 250 MHz with setup, hold, RISC-V timing and CDC checks passing at the selected
frequencies. Do not repeat a successful run only to reach 300 MHz. T1, T2, T4 and D12's rule
for selecting a 250 MHz RISC-V clock are unchanged. Historical 276.3 MHz measurements and
performance calculations retain their original frequencies.

**Milestones:**

| Milestone | Content | Pass criterion |
|---|---|---|
| **M6** | hardware: control core, issue queues, vector unit, attention / KV ops, second SPM read path, debug | Qwen2.5-0.5B Q8_0 logits bit-identical to the simulator and the Qwen2.5-32B Q4_K_M KL criterion met, MVMs issued by RISC-V kernels (attention on the CPU, `offload_kqv` = false, SW §10.4); GDB smoke on the card |
| **M7** | RISC-V kernel library and graph execution: the llama.cpp backend writes each graph split as an op list into DRAM and launches it once | one decode token = one launch for Qwen2.5 and Gemma 4 dense (12B on disk; 31B when available), KV cache and attention on the device, logits bit-identical to sim and the KL criterion (SW §10.4) |
| **M8** | device MoE, PyTorch on the kernels, device embedding lookup (DQROW RTL), TOPK RTL | defined at M8 start |
| **M9** | resident decode loop (optional; VMASK) | defined at M9 start |

**Performance priority** (spec "performance goal"): keep DDR bandwidth busy (no idle time)
before reducing bytes per token. Speculative decoding is planned after the M7 card gate. llama.cpp
proposes draft tokens on the host (n-gram or prompt lookup, a draft model, or a model's MTP head),
and the device verifies the k + 1 positions as one small ubatch (up to T_MAX tokens), which the MVM
clusters stream once per weight pass.

---

## 3. The documents

**EX — execution model (`01-execution-model.md`).** The host never runs programs: it binds the
channel memory (`MEM_EN` latch), loads the firmware's `PT_LOAD` segments into the TCMs through a
CSR loader with a receive CRC and an on-device VERIFY read-back, boots the RISC-V and launches
kernels. A launch is an 8-word descriptor pushed by posted CSR writes into a core-domain launch
queue (LQ, 8 deep); every descriptor gets exactly one 8-word record in a completion ring (LCQ, 8
deep), in order, with a launch STATUS and the first tile error; the runtime contract R1–R10 fixes
the RISC-V side, and cancel / stop watermarks end a launch early. EX owns the single host CSR map
(§5), the PERF counter space gated by ACTIVE (§6), the ISA (the `x16` field, the global opcode
and unit tables, §7), the error model with code 13 OPERAND and the device-level `SOFT_RESET`
(§8), the `ap_ctrl_none` XRT adapter (§9), the transport API (§10), the DRAM map (§11) and the
build order oracle → traces → host → RTL → platform (§12). The base kernels are `graph`, `replay`
and `selftest`.

**VU — vector unit (`02-vector-unit.md`).** Every VU op works on rows × m elements with a line
stride per operand (0 = broadcast) and a per-row scalar from an s register or an SPM vector, so
one op covers all heads and tokens. Ops: VOP (25 functions, including the CPU-compatible GELU16 /
GEGLU16), segmented RED with fused RCP / RMS post-ops (RMSNorm in two ops), ROPE over T tokens
with ggml-cpu's own angle recurrence (bit-identical angles), SOP, optional TOPK and DQROW
(conditional RTL exists; disabled in the U200 configuration), and VMASK (M9). All numerics are
bit-exact algorithms: an integer EXPK (≤ 0.53 ulp,
*model*), correctly rounded reciprocals, an exact 2/π reduction for SIN/COS, the spec's 128-partial
reduction tree with an exact skip rule. The microarchitecture has 16 front lanes, 8 half-rate back
lanes, a reduction unit and an angle unit, reads operand pairs from the even / odd SPM, uses
credits instead of a clock enable and accepts one op ahead. It also executes AT's softmax ops.
Size ≈ 48 K LUT, 366 DSP, 15 RAMB36 in SLR1 (*estimate*).

**AT — attention and KV (`03-attention-kv.md`).** The KV cache holds exactly llama.cpp's q8_0
values in a device layout (16-cell groups round-robin over the channels, K token-major, V
slice-major), so `get/set_tensor` convert losslessly. Attention runs where the bandwidth is: QKᵀ
(ATTQK) and P·V (ATTPV, an exact block-floating-point accumulation in the MAC DSPs) are modes
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
(32-entry CDC issue FIFO, dispatcher, `tsr_chk` with four tagged operand
extents, six queues with registered heads, SYNC, invariants I1–I4, DB's hard hold), the 2R2W
even / odd SPM with per-bank round robin and the DMA LD burst reservation, every clock-domain
crossing (handshakes for the rings, gray counters for single-step counters), the core-domain SR
sequencer for `SOFT_RESET`, the `ap_ctrl_none` adapter, the floorplan C'' with the SLR1 budget
(≈ 62–72 % LUT against a 65 % gate with ordered remedies), the timing gates T1–T4 with the
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
channel 0, TSR_STATS). Its registers are the DBG block at core CSR 0x800–0xAFC.

**SW — software and verification (`06-software-verification.md`).** The simulator is the
bit-exact oracle and includes an RV32 interpreter (`tsr_rv`, Spike as its oracle), an event-driven
queue model with random schedules, a race detector, the VU / AT references and `tsr_ref_math.c`,
and EX's CSR and ring model. The trace format records one launch per directory with a push-ordered
`push.bin`. The firmware SDK is freestanding
C++20: crt0, a runtime implementing EX's R1–R10, DB's trap and log contract, the tile/VU header
library and `plan.hpp`, shared with the host. The `graph` kernel executes op lists with
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
| Op-list format (OPL) and base kernels | SW §7.1–§7.2 (contracts: EX §4.6) | backend SW §9.3, DB §5.7 |
| Trace format | SW §5 | CC §15.3, DB §5.6 |
| DRAM map | EX §11 (KV planes: AT §4.5; compute heap: SW §8 R6) | SW §9.3 |
| Transport API | EX §10 (SW §9.1 adds conveniences only) | SW, DB |
| Machine-readable source of all of the above | `sw/tables/tsr_tables.py` + `gen_tables.py` (EX-30; created in W0) | RTL package, C header, `mmio/vu/at.hpp`, tb Python |

---

## 5. What the integration changed

The drafts were written in parallel and disagreed in places; two reviewers listed 42 findings.
The integration applied them as follows (each item names the winning definition):

1. **One host CSR map** — EX §5. CC's `CTL_*` block, mailbox registers, `SREG` at 0x0C0, error
   copies at 0x500, free-running PERF with `PERF_CLEAR`, 1,024-word overwrite log and the "TSR2"
   ID are withdrawn (CC-20, CC-21; CC §6 now maps CC's hardware onto EX's offsets).
2. **One launch protocol** — EX's LQ / LCQ. SW's single-slot mailbox, 5-arg LAUNCH, 16-byte image
   id, PANIC / BYE / LOGCTL messages and its transport API are withdrawn (SW-10); panic is a
   TRAP with `mcause` 24 (DB), a stop is the `LQ_STOP` watermark, the log level is `LOG_CFG`.
3. **`SOFT_RESET`** stays EX's device-level sequence; CC adds the core-domain SR sequencer that
   implements it without `clk_rv` (CC-31).
4. **Build order** — oracle, traces, host, RTL, platform (EX-28).
5. **Checker and issue contract** — check order with OPERAND, four operand extents tagged SPM /
   DRAM and read / write, I3 ≤ 2 for the VU, no run-time unit errors (CC-11 withdrawn), a
   static generated opcode → unit table instead of `ISSUE_QMAP`.
6. **AT's vector ops** are specified for the VU decoder (VU §5.11) and executed by the VU with a
   multi-pass AMERGE (VU-27); VMASK is deferred to M9 because llama.cpp KV caches are rings.
7. **DQROW** is VU's SPM form on unit 3; its RTL waits for M8 (VU-15, AT-20).
8. **PERF slots** — VU X+13…15, AT X+16…19, CC X+20…21, DB 61–63, recorded in EX §6.
9. **s registers** are read at `SREG` 0x9C0 (host and, through the core window, the RISC-V).
10. **`ISSUE_PIDX`** — a RISC-V-readable push index at control window 0x03C (CC-32).
11. **RED skip rule** — the draft's "a tree value is never −0" was false; the exact rule passes
    z(q) for a skipped pairing (VU-2).
12. **DB's hard hold** yields after a QABORT and during `SOFT_RESET`; `SOFT_RESET` clears the
    stop state; `monitor reset` releases first; QQUIESCENT counts a hard-held head as held.
13. **Ring CDC** — the RISC-V reaches LQ / LCQ / log only by handshakes; no multi-step counter is
    gray-crossed (CC-30).
14. **XRT adapter** — `ap_ctrl_none` with the `MEM_EN` latch in every document (CC-19).
15. **M6 offload path** — the `graph` executor with MUL_MAT-only op lists; `replay` (with SERIAL)
    for traces and SPM dumps; `tsr.oplist_seq` is withdrawn (EX-29).
16. **M6 gate** runs with attention on the CPU; device attention with the KV cache on the device
    is the M7 gate (AT §13.6, VU §10, SW §10.4).
17. **Trace sets** — SW's plan (generate m2 / gen / m3, convert the 290 model traces) in
    EX and CC.
18. **SLR1 budget** recomputed with the VU, DB and AT figures: ≈ 124–142 K LUT (62–71 %), with
    ordered remedies; the out-of-context VU run precedes T2 (CC-24).
19. **One table source** — `sw/tables/tsr_tables.py` (EX-30).
20. **One log header** — DB §6.1.
21. **Gemma 4 12B** read from its GGUF and added to VU §3, AT §4.5 / §7.5 / §9, SW §8: global layers
    have G = 16 (two passes) and E = 512 (K meta read-modify-write).
22. **SBA** writes are refused over the whole control and core windows (CC-33).
23. **No read side effects** — a refused `LOAD_DATA` read returns 0; core-window loads of
    `LOAD_DATA` and the DMI window return 0.
24. **Trace writer** on channel 0 with its ring in the ch0 IO region.
25. **SW stale values** — ITCM 128 KiB, queue depths, the `tsr_ref_math` API, VU recipes, clock
    ratio 0.603.
26. **x16 notation** — x16 bit i is w0 bit 16 + i in every document.
27. **CPU-compatible GELU** — GELU16 / GEGLU16 for llama.cpp graphs (VU-26, SW-31).
28. **RoPE angles** by ggml-cpu's recurrence, bit-identical to the CPU (VU-19).
29. **SW attention records** follow AT's argument layout; the backend produces mask summaries and
    preloads them (SW-32).
30. **Debug-aware launch timeout** and DB's recovery ladder in the transport (SW-24).
31. **Debug software packages** — P1.13, P3.7, P3.8 added to the work plan.
32. **Toolchain** — xPack GCC is the supported compiler; `tsr_rv` gets a test bus profile for
    riscv-tests and Spike.
33. **Unselected graph branches** — DUP and flagless nodes are unsupported (SW-33; the flag rule
    amended by SW-40 at W3b).
34. **Compute heap** sized from the reserve graph instead of fixed 512 MiB windows (SW-26; as built,
    WEIGHT-heap reservations, SW-46).

Open items that the drafts left and the integration resolved: the 4-extent checker slot, I3 for
the VU, the VU / AT / DB resource figures in the SLR1 budget, whether AT has a separate unit (no),
where AT's softmax runs (on the VU), the PERF slot assignment, the VU debug-state placement (DB's
per-unit probes), the 12B Gemma 4 dimensions, `TRW_CH`, the RISC-V-readable push index, the log
registers, the AT argument layout in SW's op lists, the VU's TANH and segmented reductions (both
exist), the `user_managed` question (not in XRT 2.18; `ap_ctrl_none`), and the PROF bit's meaning
(REC for the runtime and NODE markers; the host timeline comes from the trace writer).

## 6. Decision log

Every decision of the six documents in its final, integrated form. "Withdrawn" marks a draft
decision replaced by another area's (the id is kept so references stay valid). The documents hold
the alternatives and the full rationale.

### 6.1 EX — execution model, host interface, ISA (`01-execution-model.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| EX-1 | Launch transport | Core-domain launch queue LQ (depth 8) written by posted CSR writes, plus completion ring LCQ (depth 8) read through the CSR window | Posted writes are the cheapest PCIe operation; no DMA per launch; depth 8 lets the host pipeline launches |
| EX-2 | Host/RISC-V channels | LQ/LCQ replace the mailbox entirely; PING, HELLO, TRAP and kernel events are record types | One channel halves the CDC paths, deadlock arguments and tests |
| EX-3 | Descriptor and record formats | 8-word descriptor {TYPE, PROF, KERNEL, ARG0-6}; 8-word records whose r0 = {type, status, first error code, seq[7:0]} | 7 args cover the known kernels; one PCIe read classifies a completion |
| EX-4 | Launch ordering | Exactly one answer per descriptor, in order; hardware sequence numbers; LQ_POP refused while ACTIVE | Sequential launches make ACTIVE exact for PERF, cancel and the current launch; overlap comes from the async issue path |
| EX-5 | Flow control | Host keeps at most LQ_DEPTH launches outstanding; LCQ_DEPTH = LQ_DEPTH = 8 | Neither ring can overflow and the push path needs no read |
| EX-6 | XRT control protocol | ap_ctrl_none, xrt::ip register access, polling; no interrupts on the U200; host_irq_o for other targets | A halted RISC-V under an outstanding xrt::run breaks KDS; XRT 2.18 routes AP_CTRL_NONE to xrt::ip |
| EX-7 | Channel bases | Bases latched and alignment-checked on the rising edge of ADAPTER_CTRL.MEM_EN and locked while set; requests held while MEM_EN = 0 | No run start to latch on; one edge gives an atomic, checked rebinding |
| EX-8 | Firmware load path | CSR loader with receive CRC plus on-device VERIFY read-back CRC | About 10 ms per load; VERIFY checks the TCM contents without 32 K PCIe reads |
| EX-9 | Image identity | 64-bit prefix of a SHA-256 over the PT_LOAD segments, stored in .tsr_id and echoed in HELLO | Identifies the exact build cheaply; integrity is the CRC's job |
| EX-10 | Warm restart | SOFT_RESET, reload the DTCM segments, VERIFY ITCM (reload only on a mismatch), boot | Avoids stale .data, about 10× cheaper than a cold load; ITCM is writable so it is checked |
| EX-11 | CSR map | One host CSR map (EX §5): CTRL/STATUS at 0x020/0x024, CTL_* block dissolved, CTL_PERF merged into PERF, DBG 0x800-0xAFC, 12-bit space kept; CC, DB, VU, AT place registers inside it | A second control block would only duplicate; the map fits 4 KiB |
| EX-12 | Identity and version | ID stays 'TSR1'; VERSION 0x0000_0100 = Tesseract Core capability 1.0; CTL_ID[7:0] 1 (0 = no control core); ADAPTER_VERSION major 1 (SW-54) | The host open code refuses a different capability major and CTL_ID = 0, so no new magic is needed |
| EX-13 | PERF semantics | One counter space gated by ACTIVE (time counters also by ¬QHALT), zeroed by SOFT_RESET; MVM_WAIT and MVM_EMPTY; one slot table incl. VU X+13..15, AT X+16..19, CC X+20..21, DB 61-63 | Restores 'PERF unchanged between executions' per launch; splits MVM idle time into issue- and dependency-bound |
| EX-14 | x16 field | w0[31:16] is the op-specific field x16 (x16 bit i = w0 bit 16+i); must be 0 for every opcode that does not assign it | VU and AT get 16 bits of encoding space |
| EX-15 | Opcode allocation | Blocks beside each class (VU 0x13, 0x38-0x3F, 0x41-0x47; AT 0x14-0x17, 0x19-0x1B, 0x21-0x23, 0x34-0x37; SYNC 0x0A-0x0F, 0x70-0x7E); assigned KVST 0x14, ATTQK 0x21, ATTPV 0x22, ASMAX-ANORM 0x34-0x37, VMASK 0x38 (M9); retired opcodes never reused | Areas work in parallel without collisions; class decode stays cheap |
| EX-16 | Unit id 4 | Reserved for a future AT attention engine; absent in M6 (AT's ops run on units 0, 2, 3) | Keeps unit ids dense and the six-unit CDC widths |
| EX-17 | Error codes | Code 13 OPERAND (ERR_INFO = the value); check order ILLEGAL_OP, BAD_DEP, BAD_FMT, K_RANGE, T_RANGE, OPERAND, SPM_RANGE, DRAM_RANGE; 9, 10, 14, 15 reserved | VU/AT value limits are neither encoding nor K errors; the value makes failures diagnosable |
| EX-18 | Launch-level result | STATUS OK, TILE_ERR, BAD_DESC, KERNEL_ERR, TRAP, ABORTED, PLAT_ERR, STOPPED with fixed precedence; first error = smallest ERR_PIDX | Tile failures never read as success; the selection is schedule-independent |
| EX-19 | Platform AXI errors | plat_err_i core input, mirrored in STATUS and reflected in the launch status | Adapter errors always reach the launch status without a PCIe read per launch |
| EX-20 | Cancel and stop | Sequence watermarks: LQ_CANCEL (hard: QABORT of a covered running launch), LQ_STOP (soft: polled by kernels); covered queued launches answered ABORTED | Precise and race-free; also gives M9's graceful stop |
| EX-21 | RISC-V traps | The handler follows DB's trap contract, posts TRAP and parks for GDB; recovery is SOFT_RESET plus a warm restart | A trap may leave .data inconsistent; parking keeps the state for D18 debugging |
| EX-22 | SOFT_RESET | Hardware sequence (CC's SR sequencer): RISC-V reset, drain, clear error states, flush LQ/LCQ/LOG, zero PERF, clear DB's stop state; works without clk_rv; SR_BUSY shows progress | One atomic known-state operation that the RISC-V cannot race and that survives a missing clk_rv |
| EX-23 | Stuck units | No forced unit reset; per-unit progress watchdog (2^L cycles, about 61 ms default) sets UNIT_HUNG; recovery is a card reset | AXI has no cancel; a unit kill would corrupt later transactions |
| EX-24 | Debug log | 256-word core-domain ring read through the CSR window; whole records dropped and counted, optional BLOCK; DB's record header; LOG_CFG.LEVEL filter | Works when queues wedge; never faults or stalls by default |
| EX-25 | Transport structure | One register-level protocol layer for sim, mock, vkernel and xrt; sim advances deterministically on host reads | The protocol itself becomes part of what the oracle checks |
| EX-26 | DRAM map | OPL region (op lists, replay streams, argument blocks, overlays); TRW ring in the ch0 IO region; compute heap between weights and KV planes | The name follows the content |
| EX-27 | AUX memory source | Memory source 0 (TSR_SRC_AUX) is tied off; source ids not renumbered | Avoids churn on timing-critical arbiter paths |
| EX-28 | Build order | Oracle, traces, host, RTL, platform (Y1-Y5) | The whole software path runs end to end on the simulator before any RTL (D16); the platform needs the control core to be useful |
| EX-29 | M6 offload path | The graph executor with MUL_MAT-only op lists, one launch per graph split; the replay kernel (SERIAL) for traces and SPM dumps; no tsr.oplist_seq | One mechanism from M6 to M7 and no second offload path of about 1 µs per op |
| EX-30 | Table source | sw/tables/tsr_tables.py (plain Python) with SW's gen_tables.py emitting the SV package, the C header, mmio/vu/at.hpp and tb Python; --check in lint | One source for RTL, simulator, firmware and benches, with no PyYAML dependency |

### 6.2 VU — vector unit and approximate math (`02-vector-unit.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| VU-1 | Operand model | rows × m with line-aligned rows, per-operand line stride (0 repeats), per-row scalar c from an s register or an SPM row vector, for every VU op | Every per-head and per-token pattern becomes one op, closing the 18 K ops/token gap |
| VU-2 | Reduction order | Spec's 128 partials and complete binary tree per row; skipping unwritten subtrees passes z(q) (−0 → +0), which is exact | Keeps the spec contract and cheap short rows; z() fixes the −0 case that FTZ creates |
| VU-3 | Segmented RED | Per-row SPM vector (or an s register for one row) with fused post-ops RCP and RMS (exact SQRT and RECIP) | Per-head norms and softmax sums in one op; RMSNorm in 2 ops instead of 7, bit-identical |
| VU-4 | EXP algorithm | Fixed-point integer EXPK with f32 in/out, 2³⁸-scaled log2e constants, 64-entry 2^(j/64) table, cubic Taylor term | At most 0.53 ulp (model), about 600 LUT per lane, exact portable definition |
| VU-5 | Reciprocal | All reciprocals correctly rounded (= tsr_frecip); RECIP_APPROX removed | Exact rounding is cheap with one correction product and needs no table specification |
| VU-6 | SIN/COS | Exact 64-bit 2/π integer reduction, f32 Cephes polynomials, identity path below 2^-12 | Angles reach 2^18 where f32 Cody-Waite loses precision; the reduction runs once per angle |
| VU-7 | Function set | SILU, SWIGLU, SIGMOID, GELU/GEGLU (tanh as x·σ(2u)), GELU16/GEGLU16, TANH, EXP, EXPSUB(C), RECIP, MIN, MULCB/MULCADD/ADDMULC, SETC | Covers Qwen2.5 and Gemma 4 dense with one shared EXP + RC datapath |
| VU-8 | Lane split | 16 front lanes (incl. EXP) at full rate, 8 back lanes for SILU/GELU/SIGMOID/TANH/RECIP at half rate | Half-rate activations cost under 0.15 % of token time; EXP stays full rate for softmax |
| VU-9 | Flow control | Credit-based reads against output-FIFO space, no clock enable in the lanes, fixed latency classes | Avoids a clock enable fanning out to about 45 K FFs at 276 MHz or more |
| VU-10 | SPM access | CC's pair reads (alternating a-pair and b-pair cycles), no operand parity rule, no operand caches | Two sources stream at full rate whatever their alignment |
| VU-11 | s-register access | One bank, written only by VU ops (SOP.MOVI for constants); read-only for the RISC-V and the host at SREG0-15, core CSR 0x9C0 | Writes stay ordered by queues and deps; one read location for every reader |
| VU-12 | s-register reset | +0.0 at hardware reset; kept across SOFT_RESET, RISC-V reset, QABORT and QCLEAR | EX's requirement; kernels initialise what they read |
| VU-13 | Attention mask | VMASK 0x38 specified but deferred to M9 (ILLEGAL_OP before); llama.cpp masks use AT's mask summaries | VMASK assumes cell j holds position k0+j, which llama.cpp's ring-shaped, reorderable caches break |
| VU-14 | TOPK | Fully specified (rows, padding, total order); simulator in M6, RTL in M8 with a running-threshold filter | M6/M7 sample on the host; keeps 4 K LUT out of the first k4 link |
| VU-15 | DQROW | Opcode 0x12 re-specified as SPM to f32 on the VU after an LD of the repacked row; simulator in M6, RTL in M8 (CAPS bit6 = 0 in M6) | The VU has the fp datapath; the embedding lookup stays on the CPU through M7 |
| VU-16 | Error reporting | OPERAND (13) with the offending value for shape limits; enumeration mismatches are ILLEGAL_OP | Follows EX; the values make failures self-explaining |
| VU-17 | Field widths | 24-bit line addresses and m; 16-bit rows, strides, head counts and T; function codes and s indices in x16 | Every op fits one 256-bit instruction with headroom |
| VU-18 | Softmax ops | No generic fused online-softmax VOP; AT's vector ops (ASMAX, AMASK, AMERGE, ANORM) on the VU, two VU ops per chunk | AT fixed the dataflow; its ops fuse what the 7-op composition needed |
| VU-19 | ROPE shape and angles | T tokens with a position vector; angles by ggml-cpu's recurrence θ_{i+1} = FM(θ_i, TS), FD by freq factor, FM by freq_scale; A-table form only for YaRN; head_dim multiple of 16 (≤ 1024), n_dims multiple of 32 (NEOX) or 16 (NORM) | Angles bit-identical to the CPU for ext_factor = 0 (the A-table form differed by up to 2.9·10⁻³ rad at 32 K) |
| VU-20 | Accept-ahead | Up to 2 VU ops held; RAW, WAW and s-register hazards checked by extent intervals (CC's I3 ≤ 2 for unit 3) | Short independent ops overlap their 20-64-cycle fill and drain |
| VU-21 | EXPSUB with −∞ | EXPSUB(−∞, x) = +0, even when x is −∞ or NaN | Fully masked blocks stay finite without an ordering rule |
| VU-22 | MAX/MIN | Spec total order; MAX propagates NaN, MIN returns the non-NaN operand | One comparator serves MAX, MIN and TOPK, consistent with the spec |
| VU-23 | Raw operations | COPY, SETC, MOVI/LDS/STS, VMASK pass-through and the ROPE tail copy bits without fp canonicalisation | COPY serves any 32-bit data; positions and indices pass unchanged |
| VU-24 | Reduction unit size | 8-input subtree (7 FA) plus 2 scheduled FA; rows shorter than 256 elements reduce at half rate | Saves about 5.5 K LUT; the half-rate cases are short rows |
| VU-25 | Accuracy contract | Functions defined by their algorithms (bit-exact); error bounds informative and measured | Bit-exactness between simulator, RTL and card is the verification basis |
| VU-26 | CPU-compatible GELU | GELU16 (0x17) and GEGLU16 (0x18): clamps at ±10, binary16 rounding of input and result; llama.cpp GELU/GEGLU map to them; exact GELU kept for M8/PyTorch | The pinned CPU rounds every GEGLU element to fp16 (up to 3.9·10⁻³ relative); matching its form removes that gap for about 0.6 K LUT |
| VU-27 | AT vector ops on the VU | ASMAX, AMASK, AMERGE, ANORM run on unit 3 with the front lanes, the MAX path and a multi-pass AMERGE (about 4-5 cycles per output line at C = 4) | Reuses the EXPK lanes and pair reads; matches AT's 1,600-cycle merge tail for about 2 K LUT |

### 6.3 AT — attention and the KV cache (`03-attention-kv.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| AT-1 | Where attention computes | QKᵀ and PV in the MVM clusters (one channel each); the VU only does softmax passes and merges | Only the clusters consume the KV cache at the 256 B/cycle DDR rate; one SPM port is 64 B/cycle |
| AT-2 | KV values | Exactly llama.cpp's q8_0 (blocks along each head row, fp16 scale, quantize_row_q8_0_ref rounding); only byte placement changes | Device KV equals the CPU's; state save/load stays lossless |
| AT-3 | Channel distribution | 16-cell groups round-robin over channels (group u in channel u mod C) | Balances channels at any context; one score line per group and row |
| AT-4 | Plane layouts | K token-major in Q8_0 chunks; V slice-major (each 64-value slice of all cells contiguous) | Whole scores per pass for K; one DSP accumulator per slice for V |
| AT-5 | KV store op | KVST (DMA unit) stores QACT A32 output into consecutive cells; partial meta beats by read-modify-write inside the DMA | QACT is already bit-exact quantize_row_q8_0_ref; no write strobes or boundary change |
| AT-6 | PV numerics | Block floating point per 64-cell channel segment: 24-bit integer weights, exact integer sums in the DSP P register, one f32 add per segment | Reuses the 512 MAC DSPs and is at least as precise as the CPU's f32 path |
| AT-7 | PV segment length | 4 local groups = 64 cells per channel | 7.5 conversions per cycle per cluster instead of 30 for 16-cell segments |
| AT-8 | Softmax structure | Two-pass within a chunk (max from ATTQK, EXP by ASMAX); online merge across chunks (AMERGE) | Per-cell rescaling is impossible in DSP accumulation; chunks fit the slot buffers and pipeline |
| AT-9 | Scale placement | Inside ASMAX and AMERGE as EXP((s − m)·scale); ATTQK returns raw scores | One rounding as on the CPU, free in an op that reads every score |
| AT-10 | Row sums | ATTPV computes l with the same block-float rule (LSUM flag) | Saves a read pass over the scores and keeps l consistent with O |
| AT-11 | Channel partials | ATTPV writes C per-channel partials; AMERGE adds them in a fixed pairwise tree; results depend on NUM_CH | Avoids serialising the clusters; bit-exactness is per configuration |
| AT-12 | Row max floor | M_FLOOR = -2^100 (0xF1800000) in ATTQK MAXOUT and AMASK | Keeps merges finite for rows with no visible cell |
| AT-13 | Empty rows | ANORM multiplies by RECIP(max(l, 2^-126)) | Reproduces the CPU's S == 0 ? 0 : 1/S at no cost |
| AT-14 | Mask handling | Host-built mask summary (≤ 2 cell runs per row) with CAUSAL/SLIDE bounds in ATTQK; f32 copy and AMASK only for GENERAL masks | The VU has no f16; the cached op list stays valid; masked bandwidth is skipped |
| AT-15 | GQA and prefill mapping | MVM slots = (query token, head in the KV-head group); head subsets (hg0, GS) when G > T_MAX; T_q = floor(T_MAX/GS) | K streams once per pass and every G works (Gemma 4 12B global layers: two passes) |
| AT-16 | Units and opcodes | KVST 0x14 (DMA), ATTQK 0x21 and ATTPV 0x22 (MVM), ASMAX/AMASK/AMERGE/ANORM 0x34-0x37 (VU); unit 4 absent in M6; no new error codes | The ops use existing datapaths; a separate queue would only add arbitration |
| AT-17 | Hadamard KV rotation | Off (LLAMA_ATTN_ROT_DISABLE=1 everywhere, SW-16); the bit-exact FWHT recorded in Appendix A | Baselines are fixed with rotation off; q8_0 gains little; it can return in M8 |
| AT-18 | Unsupported FA features | No attention softcap, ALiBi, sinks, sparse n_kv_max, DK ≠ DV, D outside 64-512, K-shift in M6/M7 | No target model uses them; each can be added later without a layout change |
| AT-19 | Chunk policy | NCG = min(128 groups, PCAP limit, at least 2 chunks for ≥ 8C groups), segment-aligned; the chunk with this launch's new cells last | Overlaps VU and MVM work; deterministic, so part of the numeric contract |
| AT-20 | DQROW | Per VU §5.6 (SPM to f32 on the VU, RTL in M8); M8 embedding = LD + DQROW; an f32 tok_embd copy for other GGUF types; never used by attention | VU owns the encoding; the lookup stays on the CPU until M8 |
| AT-21 | KV region | Symmetric planes allocated top-down from each channel's end; one plane per (layer, stream, K/V); LG from kv_size | Equal channel shares; KV growth stays away from the weights |
| AT-22 | FA head subsets and chunk planner (W1a / W2b) | Head subsets inside each chunk; one chunk planner (`tsr_fa_plan.h`) for `tsr_ref_fa` and the firmware, its rules normative (AT §6.6); FA_NEW flags word; MK[4] | G > T_MAX cannot run as outer passes; one planner gives bit-identical outputs by construction; MK[3] raced |

### 6.4 CC — control core, issue path, SPM, CDC, floorplan (`04-control-core.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| CC-1 | Ibex pin and configuration | OpenTitan-locked Ibex commit, unmodified; RV32IMC+Zicsr+Zifencei, RV32MFast, RegFileFPGA, no PMP/ICache/predictor/WB stage, 2 triggers, MHPMCounterNum 0 (pipeline amended by CC-34) | A silicon-signed-off state in the verified small configuration |
| CC-2 | Debug module | Upstream riscv-dbg dm_top without its JTAG DTM; DMI from CSR 0xC00-0xDFC; SBA connected | D18 needs a standard DM; the debugger speaks DMI directly |
| CC-3 | TCMs | ITCM 128 KiB + DTCM 64 KiB, RAMB36 32K × 1 columns, read latency 1 (48 RAMB36) | Executor, two model families, M8 and debug fit without overlays |
| CC-4 | SPM line-buffered alias | 0x5000_0000 alias whose loads hit a one-line buffer, invalidated by SPM stores, window accesses, interrupt edges and SPM_LB_INV | Op lists read about 7× faster while keeping the memory model |
| CC-5 | Issue port | MMIO staging (8 words, 2 deps, push with word mask); software checks credits; 65,536-cycle hardware safety net, suspended under dbg_stop | Ibex stays unmodified; a blocking push would block interrupts and halts |
| CC-6 | Unit ids | 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT engine (absent in M6), 5 SYNC; 6-13 reserved, 14 DB marker, 15 none | Keeps six units, six counters and the CDC widths |
| CC-7 | Opcode to unit mapping | Static UNIT_OF generated from EX §7.2; ISSUE_UNITS and Q_DEPTH report built units; no ISSUE_QMAP | Software has the table at compile time; one mapping fewer to keep consistent |
| CC-8 | Queue depths | DMA 8, QACT 4, MVM 4, VU 16, SYNC 4 (unit 4 absent; parameter 8); issue FIFO 32 | Many short VU ops want a deeper VU queue for about 100 LUT |
| CC-9 | Checker | One tsr_chk, 8 cycles per op, EX's check order with OPERAND, four extents per class tagged SPM/DRAM and read/write, shift+DSP MVM extent | 34 M ops/s is at least 3× the issue rate; four-SPM-operand ops exist |
| CC-10 | Dispatch to units | Queue-head command registers drive the units; an entry never starts in the cycle it becomes head | Clock enables stay one flop away; DB gets one cycle to hold |
| CC-11 | Runtime unit errors | Withdrawn: units have no error output; every error is found at dispatch | EX's rule that no op fails after it started keeps the first error schedule-independent |
| CC-12 | SPM second read path | Even/odd line-interleaved banks, 2R2W, per-bank client ports, registered round robin, 6 clients, 128 URAM unchanged, single-port adapter for single-port clients | A second read path without URAM; line pairs always hit both banks |
| CC-13 | DMA LD flow control | Burst reservation with a 512-beat queue (8 RAMB36); rd_rsp_ready constant 1 | SPM contention can never stall a channel's response path |
| CC-14 | s registers | One bank in the VU, read-only at SREG0-15 (core CSR 0x9C0) for the host and the RISC-V; writes only through ops | One location for every reader; writes ordered by queues |
| CC-15 | Core window | Read-only view of the whole CSR map at 0x2000_1000 without snapshot side effects; LOAD_DATA and DMI read 0 there | One mechanism for ID, s registers, PERF and records; cannot disturb loads or the debugger |
| CC-16 | Debugger memory path | DM SBA as a second master on the RISC-V data bus; no host debug SPM client | Reaches TCM and SPM without halting; the SPM keeps six clients |
| CC-17 | Log path | EX's 256-word ring; CC part: LOG_PUSH into a 16-entry CDC FIFO, LOG_FREE from a clk_rv push count and a handshaked LOG_RD | Never stalls, never torn, works with wedged queues |
| CC-18 | RISC-V clock | ap_clk_2 at 166.7 MHz via a v++ clock request, else BUFGCE_DIV ÷ 3 of the 500 MHz platform clock; fully asynchronous CDC | D5/D12; independent of core-clock scaling |
| CC-19 | Kernel control protocol | ap_ctrl_none per EX §9 (fallback: declare ap_ctrl_hs, never run); xrt::ip and polling; MEM_EN latch; no interrupt on the U200 | XRT 2.18 has no user_managed type and routes AP_CTRL_NONE to xrt::ip |
| CC-20 | ID and VERSION | Withdrawn: ID 'TSR1' and VERSION 0x0000_0100 (Tesseract Core capability 1.0) per EX-12 | The host open code already refuses a different capability major |
| CC-21 | PERF counters | Withdrawn: EX §6's counter space; CC adds X+20 SPM_RD_WAIT and X+21 SPM_WR_WAIT | One counter space with protocol-defined gating |
| CC-22 | Recovery and reset | No partial datapath reset and no op cancel; SOFT_RESET per EX via the SR sequencer; a wedged unit needs a card reset | Unit resets with outstanding AXI bursts break the arbiter contracts |
| CC-23 | Timing gate (D13 restated) | T1 control core OOC at 166.7 MHz; T2 k4 core OOC at 3.333 ns after the OOC VU run; T3 k4 link ≥ 250 MHz with setup/hold, RISC-V timing and CDC checks passing; current request 300 MHz; T4 k1 link 300 MHz | User amendment of 2026-11-02 changes only T3's lower bound; no repeat solely to reach 300 MHz |
| CC-24 | Floorplan | C'': control core, issue path, VU and DB next to the SPM in SLR1; SLR1 ≤ 65 % LUT, else VU shrink knobs, then DB optional blocks off, then VU lanes to SLR0 | The recomputed budget (62-71 %) straddles the gate; every remedy keeps results unchanged |
| CC-25 | Gather job-pipe fix | tsr_rslice FORM 1 (cycle-identical) for the central job-pipe stages plus pinning stage 0; JOB_PIPE 3 only if still short | Removes the measured worst paths without changing any cycle |
| CC-26 | Memory source 0 | TSR_SRC_AUX, tied off in M6 | Keeps the arbiter shape; nothing claims it yet |
| CC-27 | CDC verification | Verilator two-clock benches over a ratio sweep, a jitter mode in tsr_sync, report_cdc, zero unconstrained ap_clk/ap_clk_2 paths | Verilator cannot model metastability; jitter exercises skew |
| CC-28 | Register table | CC's rows in sw/tables/tsr_tables.py (EX-30) | One source for every area |
| CC-29 | RV_HALTED detection | Derived from the last instruction-fetch address in the DM range; cross-checked against dmstatus; DMI fallback | Keeps riscv-dbg unmodified |
| CC-30 | LQ/LCQ/log CDC | The RISC-V reaches EX's rings only by tagged handshakes (reads return core-domain values; pop and post commit in the core domain); no gray-coded copies of multi-step counters | Host-written LCQ_RD/LOG_RD and SOFT_RESET flushes jump by more than one step |
| CC-31 | SR sequencer | Core-domain sequencer for EX's SOFT_RESET (RISC-V reset within 2^16 cycles, drain, clear list, SR_BUSY), independent of clk_rv | The host needs an atomic known-state operation even without clk_rv |
| CC-32 | ISSUE_PIDX | clk_rv counter of ticketed pushes at control window 0x03C, equal to the dispatcher's pidx; +1 per ISSUE_PUSH and per QCLEAR only | Exact and timing-independent like ISSUED_SEQ, for about 40 FF |
| CC-33 | SBA write refusal | Every SBA write to the control and core windows refused in hardware, and SBA TCM writes while RV_RUN = 0 | LQ_POP, LCQ_POST, LOG_PUSH and TRACE_MARK have protocol effects; a whole-window rule is simple to verify |
| CC-34 | Ibex pipeline (amends CC-1) | Ibex "maxperf" pipeline: WritebackStage 1, BranchTargetALU 1, RV32MSingleCycle; TCMs take one data access per cycle; the simulator's default timing model. T1 fallback: RV32MFast, then the configuration with the lower time per op record | −16 to −22 % RISC-V cycles per op record on the simulator for ≈ +0.5–1 K LUT and +2 DSP |

### 6.5 DB — debug and profiling (`05-debug.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| DB-1 | GDB transport | Built-in RSP server (libtsr_dbg) driving DMI through the CSR window, embedded in the device-owning process or standalone tsr-dbg | One CSR access per DMI access; no OpenOCD and no second process |
| DB-2 | DMI read side effects | DBG_CFG.DMI_EN gates the DMI window (reads 0 and writes dropped when off) | Keeps EX's no-read-side-effect rule for every other agent |
| DB-3 | Debugger memory access | SBA for all memory access; the program buffer only for fence.i | Works without halting and never changes hart state |
| DB-4 | D15 enforcement | Hardware refuses every SBA write to the control and core windows (CC-33); the server never writes the window nor stores to it from the program buffer; issuing inferior calls unsupported | No debugger path can push a tile op |
| DB-5 | Halt requests from DB | DB's request ORed into Ibex debug_req_i, one-shot via a toggle, cleared when the hart halts | Tile stops and watchpoints halt the hart without a DMI round trip; no re-halt |
| DB-6 | Breakpoints and watchpoints | 2 Ibex execute triggers, unlimited ITCM software breakpoints, 2 DB data watchpoints on the Ibex data port | Ibex triggers are execute-only; GDB software watchpoints are unusable over PCIe |
| DB-7 | Halt at boot and launch | Boot: DB halt request pending at reset release; launch: runtime hook ebreak selected by __tsr_dbg | Stops exactly at the first instruction or kernel entry for two loads per launch |
| DB-8 | Tile stop mechanism | Push-index threshold T held at the queue heads (dbg_hold); breakpoint matches registered at enqueue | Yields exactly the push-order prefix the sequential simulator reproduces |
| DB-9 | Hard hold | dbg_hold also blocks non-running completions, except after a QABORT (until the next QCLEAR) and during SOFT_RESET, which clears the stop state | Keeps invariant G at stops while every reset still drains |
| DB-10 | Serial mode | Head holds keep only the oldest outstanding entry runnable: one op at a time in push order | Same mechanism as stops; the op stream is untouched |
| DB-11 | Dispatch history | 512-entry ring of full ops, deps, ticket, push PC and time; recording freezes 256 entries after the first error; exportable as push.bin | A hang needs the ops that never completed; the evidence replays |
| DB-12 | Log ring and record header | EX's 256-word ring, whole records dropped, BLOCK mode; the one header: [7:0] N, [11:8] LEVEL 0-5, [15:12] SOURCE, [18:16] TYPE, [19] TS, [27:20] LSEQ, [31:28] 0xA | Never-torn records need no resync; one header for the encoder and every decoder |
| DB-13 | Crash path | With a debugger: restore and ebreak at __tsr_trap_halt (shown as SIGSEGV etc.); without: __tsr_crash, TRAP log record, EX's TRAP record, park; panic = mcause 24 | GDB shows faults natively; the record survives without a debugger |
| DB-14 | Push PC | Ibex current_pc captured at ISSUE_PUSH and carried to the history ring (+32 bits per issue-FIFO entry) | Every tile op maps back to its C++ line |
| DB-15 | PERF semantics | EX's ACTIVE/QHALT gating and counter list; DB adds TRW_RECORDS, TRW_DROPS, DBG_STOP_CYCLES in slots 61-63 | Exact windows; debugged measurements are detectable |
| DB-16 | Timeline transport | Hardware trace writer: 32-byte records to a DRAM ring through a second tsr_mem_arb write source on TRW_CH = 0 (ring in the ch0 IO region); REC stays for firmware | MMIO and firmware streaming are too slow or perturb the measured phases |
| DB-17 | Who runs the server | The device-owning process (embedded service) or tsr-dbg when nothing else owns the device | xrt::ip contexts are exclusive |
| DB-18 | Simulator debug model | Behavioural DM covering the DMI profile, with exact stop, history and trace-writer semantics | Behaviour is identical where the server looks |
| DB-19 | Watchdog | EX's progress watchdog; DB adds HUNG_STOP, a history freeze and per-unit AGE probes | One watchdog is enough |
| DB-20 | Recovery | 5-level ladder: dump, debug release, LQ_CANCEL, SOFT_RESET with warm restart, card reset; DB's evidence survives every level below a card reset | Post-mortem data exists before any destructive step |
| DB-21 | Host access to the s registers | SREG0-15 at 0x9C0-0x9FC (in the DBG range, present whenever the VU is built), the one read location for host and RISC-V | EX uses 0x0C0 for LOAD_ADDR; one location for every reader |
| DB-22 | SPM bulk dump | A replay launch of ST ops, then a bo read; SBA for small reads and the dead-firmware case | D15-compliant with no new kernel and no SPM client |
| DB-23 | Register placement | DBG block at 0x800-0xAFC, RISC-V control window 0x160-0x17C and 0x400, PERF slots 61-63 | Inside the reservations of EX and CC |

### 6.6 SW — software, simulator, verification, work plan (`06-software-verification.md`)

| Id | Topic | Choice | Rationale |
|---|---|---|---|
| SW-1 | RISC-V model in the simulator | In-house RV32IMC+Zicsr interpreter tsr_rv in sw/sim; Spike only as a test oracle on a separate test bus profile at 0x8000_0000 | One deterministic C library usable in the backend; Spike anchors ISA correctness |
| SW-2 | Simulator time model | Event-driven on core cycles; fixed schedule by default, random:<seed> for determinism tests; spin and WFI fast-forward | Exposes schedule dependence cheaply; bounds the polling cost |
| SW-3 | Race detector policy | Vector clocks; fatal in ctest, off in model runs, report nightly | The cost matters only on long model runs |
| SW-4 | Launch state | Launches are stateless (image, args, DRAM); SPM_STALE check; fresh-boot replay of every captured trace | Every trace is self-contained and replayable |
| SW-5 | Trace layout | One directory per launch; single push-ordered ops/push.bin (48-byte records, relative tickets); images shared by id | Replay needs the global push order |
| SW-6 | Trace sets | Generate m2/m3/gen with tsr-trace-gen (fixed seeds); convert the 290 model traces to serial chains (traces/model-v1conv, replay SERIAL) | Generators and seeds exist; the converted model traces are large and error-free and check that the units' results do not change |
| SW-7 | Toolchain | xPack riscv-none-elf-gcc 14.2.0-3 (with GDB) in user space, the only supported compiler; Spike and dtc from source; Clang optional | No sudo, pinned; distro GCC lacks the rv32imc multilib and libstdc++ headers |
| SW-8 | Firmware language rules | Freestanding C++20, no heap/exceptions/RTTI/libstdc++ headers, -Os (hot path -O2, SW-34), libgcc only; no f32 math in firmware | No soft-float path; the host computes every constant |
| SW-9 | Image identity | EX §3.2: 8-byte .tsr_id = first 8 SHA-256 bytes (little-endian uint64) over PT_LOAD; echoed in HELLO r2/r3 | Identifies the build; CRC and VERIFY prove the transfer |
| SW-10 | Host/RISC-V protocol | Withdrawn: EX's LQ/LCQ protocol with runtime contract R1-R10, up to LQ_DEPTH launches outstanding | EX owns the launch protocol |
| SW-11 | Logging | Binary FMT records (format-string address + args) with DB's header through EX's log ring; whole-record drop with a counter; runtime filter LOG_CFG.LEVEL | No TCM spent on strings; readable when queues wedge |
| SW-12 | Graph execution model | Host builds a pre-resolved ggml-level op list (the op list format) with host-computed dependencies, cached per graph; firmware kernels per kind | Addresses are immediates; the host does the ggml work once per graph |
| SW-13 | Data placement in M7 | Tensors in DRAM between ops; per-op SPM scratch ring with release tickets; bias/residual fused via MVM ACC, RMS_NORM × weight fused | Simple and race-free; the fusions are bit-identical |
| SW-14 | M6 llama.cpp path | The graph executor with MUL_MAT-only op lists, one launch per graph split | One mechanism from M6 to M7; like-for-like gates |
| SW-15 | Index inputs | Preload KV write indices, row ids, positions and each KQ mask's MSUM at launch start with one join | Avoids a join per SET_ROWS or FA op |
| SW-16 | llama.cpp Hadamard KV rotation | LLAMA_ATTN_ROT_DISABLE=1 for device runs and every baseline; device rotation is a follow-up | The rotation matrix is a graph input that MVM cannot use |
| SW-17 | KV cache buffers (as built: SW-47) | Recognise llama.cpp KV buffers by tensor names; AT's layout in a KV heap; converting get/set/clear; no context shift in M7 | Names are the stable signal; state save/load keeps working |
| SW-18 | Host protocol layer | One protocol implementation of EX §10 over all kdevs; sim is a kdev with idle() | The card's host code path is tested on every sim run |
| SW-19 | Test access to the queues | tsr_sim_push / tsr_sim_qctrl hooks in the simulator library only, never through tsr_transport | Directed tests without firmware; D15 holds for every real path |
| SW-20 | Planning code | Freestanding plan.hpp (tiling, QA_LINES, ROW_BEATS, scratch needs) compiled for host and firmware | supports_op rejects exactly what the firmware cannot plan |
| SW-21 | Test models | Synthetic tiny Qwen2 and Gemma 4 GGUFs generated at build time; real Qwen 0.5B/32B and Gemma 4 12B (31B when available) for gates | Fast ctest coverage of SWA, per-head norms and GEGLU without binaries in git |
| SW-22 | Baseline settings | FA on, KV q8_0/q8_0, rotation off for device and CPU runs from M6; offload_kqv false in M6, true in M7 | D17; baselines stay constant from M6 to M7 |
| SW-23 | Approximate-math verification | Exhaustive 2^32 bit-exact check vs an independent numpy model, ulp bounds vs double, RTL exhaustive opt-in; GELU16 over all 65,536 fp16 inputs vs the pinned table | Unary f32 functions are fully enumerable |
| SW-24 | Launch watchdog | 60 s default, counted only while not halted and no tile stop is active, off in SERIAL; transport mutex released between polls; DB's ladder on timeout | A GDB breakpoint inside llama.cpp must not trigger a device reset (D18) |
| SW-25 | Kernel test builds | Host build (ASan/UBSan) plus RV build on tsr_rv (the reference); ctest runs both | Fast logic checks; the RV build proves control flow and ABI |
| SW-26 | Compute-buffer placement (amended by SW-46) | Per-channel compute heap between weights and KV planes sized from the reserve graph, plus ch0 IO and ch1-3 low 512 MiB; get_max_size ≥ largest tensor | Gemma 4's 512 MiB logits chunk would fill a fixed window and add a launch per token |
| SW-27 | numcheck | Rely on cb_eval per-node observation (one launch per node); no special backend mode | No second code path to keep correct |
| SW-28 | Card selftest | selftest kernel with numbered sub-tests; tsr-selftest compares with the same image on the in-process simulator | D15 requires kernels; exact expectations without golden files |
| SW-29 | Op-list size cap | At most 4096 ops per launch (32 KiB of DTCM tables); the builder splits larger graphs | Static DTCM use; decode needs about 1.3-2 K ops |
| SW-30 | Timing independence of firmware | Timing-dependent reads never influence pushes, deps, stores or records; ISSUE_PIDX and ISSUED_SEQ are exact | Makes card logits bit-identical to the simulator |
| SW-31 | GELU in llama.cpp graphs | GGML GELU and GEGLU map to the VU's GELU16/GEGLU16; exact tanh GELU for M8/PyTorch | Matches the CPU's fp16-table form |
| SW-32 | Mask summaries (amended by SW-48) | Backend writes the MSUM (and an f32 copy for GENERAL masks) on set_tensor of 'kq_mask' tensors, reserved via get_alloc_size; FA refers to it via the preload table | The cached op list stays valid per token; the device never reads F16 |
| SW-33 | Unselected graph branches (flag rule amended by SW-40) | DUP, non-KV SET_ROWS and nodes without GGML_TENSOR_FLAG_COMPUTE unsupported; graph_compute skips flagless nodes | Prevents a second split and launch per token |
| SW-34 | Firmware optimisation flags (amends SW-8) | -Os for the image; kernels/graph.cpp and kernels/k_*.cpp add -O2 | −1 % per MUL_MAT record, −5 to −7 % per JOIN / NOP record, −23 % RISC-V time per 0.5B layer for +2.2 KiB ITCM |
| SW-35 | Executor hot path | Records read in place after the op's control-window accesses; T = 1 MUL_MAT fast path (leaf, one credit read per op, ring alloc without deps); planning stays in firmware; spm_ring O(1), 128-block table; streamed lists keep W2a's chunk-LD position; the executor's and MUL_MAT's pushes are W2a's except PROF markers, pinned by trace-digest tests | The MVM no longer idles on issue; 1,518 → 387 RISC-V cycles per 0.5B decode record as spin-free work with deep queues (Ibex maxperf; small 1,966 → 485); 401 with U200 queues, 437 by W2a's method |
| SW-36 | The one-tile block (W2b integration; size W2b fix) | One fixed SPM block (1,024 lines) for every single-tile record of a DMA-bracketed kind, fast and general path alike; MUL_MAT and FLASH_ATTN keep the ring (budget 60,144 lines); GLU's larger decode rows take a ring block in the fast path | The DMA queue serialises DMA-bracketed tiles anyway: no dep, record or release; −60 to −80 RISC-V cycles per short record; M6 trace digests unchanged; 1,024 (not 4,096) lines keep MUL_MAT's prefill T-tile of the Qwen2.5 0.5B–32B output heads at 3 |
| SW-37 | M7 dispatch (W2b integration) | A leaf fast entry per short kind in the kind table, called with (args, kflags, d); the handler out of line on "not taken" | −10 to −25 cycles per short record; MUL_MAT's path unchanged |
| SW-38 | ITCM budget (W2b integration; the figure is amended by SW-75) | 80 KiB for the production image (69.7 KB with the M7 kinds), 48 KiB left for M8 / M9; test images bound by the ITCM. SW-75 raised the budget to 96 KiB for the MoE and Gated DeltaNet kinds together (84,056 B), 32 KiB left (SW §7.8) | The per-kind leaf fast paths meet the issue-rate requirement; shared bodies cost 22–29 cycles per record |
| SW-39 | Staged-word reuse and hot entries (W2b fix) | Fast-path pushes skip `ISSUE_W` stores of words the record's previous push staged with the same value (CC §5.1; host builds check each claim) and unused ticket reads; straight-line hot entries for RMS_NORM / ROPE's common decode records; `op_loop` without MARK / SERIAL tests for plain lists | Every short kind ≤ 350 RISC-V cycles at the decode shapes (RMS_NORM 324, ROPE 325, SCALE with a bias 336) without an OPL change |
| SW-40 | Flagless nodes (W3b; amends R1 / SW-33) | supports_op does not test GGML_TENSOR_FLAG_COMPUTE; DUP, non-KV SET_ROWS and the quantized tok_embd GET_ROWS are refused by their own rules; graph_compute skips flagless nodes | test-backend-ops asks before the flag is set; decode graphs are still one CPU + one Tesseract split |
| SW-41 | supports_op of the M7 kinds (W3b) | One lowering (`tsr_lower_m7`) in probe mode + `check_args` + the kind's planner; 64-byte rule (a result that is a view has rows of whole lines; one row of its own may end inside a line); KV views as llama.cpp builds them; llama.cpp's ROPE placement probe planned as a full rotation | The host refuses exactly what the firmware cannot plan (2,850 shapes, 0 disagreements); Gemma 4's rope_freqs stays on the device |
| SW-42 | Fusion conditions (W3b) | MUL_MAT + ADD (bias, residual) and RMS_NORM + MUL (weight) when the first node's whole-graph use count is 1, no view / output, all operands on the device, nothing in between touches the operands or result | Bit-identical to the unfused records; 168 pairs per 0.5B token, 289 per Gemma 4 12B token |
| SW-43 | List cuts and FA_NEW (W3b) | Cut only at 4,096 records or a full 256-line preload area; FA_NEW across cuts as a preload entry of the FA's list; FLASH_ATTN of T > 680 tokens on the CPU (MSUM + FA_NEW indices over 256 lines) | One launch per decode token; outputs independent of cuts (a 3-chunk wrapped ring bit-exact with every record in its own list) |
| SW-44 | Data-dependent FA refusals (W3b) | The mask producer plans every reader's tiles and promotes a refused tile's MSUM to GENERAL with the f32 copy; no CPU fallback | One launch per token; the simulator stays the bit-exact oracle; never triggered at U200 parameters |
| SW-45 | Plan cache (W3b; R7) | Lists cached by `tsr_graph_key` (addresses, shapes, op params; not values) + placement epoch; mask readers re-registered on a hit; LRU of 16 | 0.5B 8 × 64 tokens: 6 builds, 506 hits |
| SW-46 | Compute windows (W3b; amends SW-26) | ch0 IO, else ch1-3 low 512 MiB, else ≥ 64 MiB WEIGHT-heap reservations; get_max_size 512 MiB; get_alloc_size pads only where ggml allows | Logits chunks stay on the device: 0 staged chunks for 0.5B and Gemma 4 12B at -ub 512 |
| SW-47 | KV buffers as built (W3b; SW-17) | Recognised at init_tensor, planes placed top-down at the first data access; converting get/set/memset/clear on any byte range in 8 MiB windows of whole local groups | Usage is known only after init_tensor; state files bit-identical to the CPU's |
| SW-48 | Mask readers and f32 copies (W3b; amends SW-32) | Every FLASH_ATTN registers (scale, E, D, G); one copy per distinct scale, written only when the MSUM is GENERAL or a tile takes the GENERAL path; FA word 13 always names the copy | Cached lists stay valid per token; rule-3 GENERAL tiles need the copy; decode writes none |
| SW-49 | Device kinds per phase (W3b) | TSR_KINDS=m6 (MUL_MAT only), set by klcheck / numcheck for --phase m6; M6 e2e ctests with TSR_KINDS=m6 TSR_COMPUTE_WINDOWS=io | M6 launches once per split (145 launches per 0.5B token) |
| SW-50 | M7 graph tests (W3b; SW-21) | Tiny Qwen2 Q8_0 / pure Q4_K and Gemma 4 Q8_0; launches per llama_decode from klcheck -v markers, splits from GGML_SCHED_DEBUG=2; tiny Gemma 4 with --min-kl 3e-3; top-1 ≥ 0.95 | A Q6_K mix and a 4-layer model's degenerate floor would make the criterion meaningless; the bounds catch injected SWA / RoPE / eps errors |
| SW-51 | klcheck for SWA models (W3b) | --swa-full off (the SWA ring, recorded in the header); token-row scale √n_embd in the embd / floor rows | Long Gemma 4 prompts wrap the ring; the Gemma 4 floor was 8.2 nats without the scale |
| SW-52 | test-backend-ops for the M7 kinds (W3b) | One ctest per op filter with a minimum supported count; FA / SET_ROWS covered by ggml_tesseract_attn | test-backend-ops cannot place K / V / dst in a KV buffer |
| SW-53 | Host memory and the M7 kinds (W3b fix; amends R6) | Only MUL_MAT is staged: M7 ops with operands in host memory (control vectors, LoRA, TSR_FORCE_HOST) stay on the CPU; a compute buffer without a window aborts naming it | Correct runs instead of aborts; staging every kind is not needed at -ub 512 -c 4096 |
| SW-54 | Versions (SW §6.9) | Tesseract Core capability 1.0 in VERSION; Tesseract Basis 0.1.0 (SemVer, tsr/version.h; .note.tsr word 6, HELLO r6); Basis ABI 1; a numerics revision per kind (.tsr_kinds, HELLO r4[31:16]) guarded by fw_kind_revs; backends refuse an older Basis and route only the booted image's kinds | One number per contract (hardware, software, ABI, numerics), as CUDA's compute capability, toolkit, driver ABI; (kind, revision) is a reproducibility key |

---

## 7. Review findings not applied, or applied differently

All 42 findings were evaluated. Two were not applied because another finding asked for an
incompatible fix of the same problem; the others below were applied with a deliberate deviation.

| Finding | Asked for | Integration | Why |
|---|---|---|---|
| [19] (major) | the single table source = SW's `sw/sim/regs/tsr_regs.yaml` + `gen_regs.py` | **not applied**; [29]'s `sw/tables/tsr_tables.py` + `gen_tables.py` chosen (EX-30) | Both findings agree on one source with SW owning the generator. Plain Python needs no PyYAML in the RTL, firmware and bench builds, and `sw/tables/` reflects that RTL, simulator, firmware and benches all consume it |
| [30] (major) | the push index kept in software (a DTCM counter initialised from `Q_PIDX`) | **not applied**; [10]'s hardware `ISSUE_PIDX` at control window 0x03C chosen (CC-32) | A `clk_rv` counter is exact and timing-independent like `ISSUED_SEQ`, costs ≈ 40 FF and stays right even for code that pushes outside the SDK counter. [30]'s normative rule of which entries advance the index (+1 per push and per QCLEAR; not QABORT, QHOLD, TRACE_MARK) is adopted in CC §5.1 |
| [13] (major) | a periodic snapshot handshake (every 64 `clk_rv` cycles) for `LCQ_RD`, `LOG_RD` and the flushed counters | applied with a simpler mechanism | Every RISC-V read of an LQ / LCQ register is itself a tagged handshake returning the core-domain value, and pop / post commit in the core domain; `LOG_FREE` = depth − (push count in `clk_rv` − handshaked `LOG_RD`). Exact, never over-reports, no periodic traffic (CC-30) |
| [12] (major) | `dbg_hold` ignored while `SR_BUSY` or "while `RV_RESET_DONE` is rising"; `SOFT_RESET` clears `SERIAL` | applied with a broader rule | The hard hold yields after any QABORT until the next QCLEAR (covering the RISC-V reset, `LQ_CANCEL`, host QABORT and `SOFT_RESET`) and is forced off while `SR_BUSY`; `SOFT_RESET` clears the stop state but keeps `SERIAL`, which cannot deadlock and is a user setting (`TSR_TILE_SERIAL`) |
| [25], [39] (minor) | meta `"q_depth":[8,4,4,16,8,4]` | `[8,4,4,16,0,4]` | Unit 4 is absent in M6 (AT-16), so `Q_DEPTH[4]` reads 0 |
| [1] (blocker) | insert EX §4.5 into CC verbatim | CC §5.2 points to EX §4.5 and states only CC's implementation | One normative copy cannot drift; both documents are generated into the same table source |
| [6] (major) | copy AT §5.4–§5.7 into VU verbatim | VU §5.11 restates the encodings for the decoder and points to AT for semantics and checks | Same reason: AT stays the single definition of the semantics |
| [6], [40] | delete VMASK or state its use | VMASK kept as an encoding, deferred to M9 (VU-13) | It is exact for the M9 resident loop's linear cache; deferring saves its area now |
| [42] (minor) | alternative: 1 GiB windows on ch1–3 | the compute heap (the finding's first option) | Sized from the reserve graph; no fixed window to outgrow |
| [2] (blocker) | PANIC as kernel event 0xC0 or a TRAP with `mcause` 24 | TRAP with `mcause` 24 | DB's crash contract already defines it; GDB shows it as SIGABRT |
| AT §14, SW §15 | spec and doc edits "applied by the integrator" | not applied now; listed in §11 as W0 | The integration task allows edits inside `docs/design/` only |

---

## 8. Remaining risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| 1 | SLR1 LUT ≈ 62–72 % (up to 79 % with the VU at + 30 %) against the 65 % gate | k4 link below 250 MHz | out-of-context VU run before T2; ordered remedies: VU shrink knobs (−8.7 K), DB `HAS_TRW` / `HIST_DEPTH` off, VU lanes to SLR0 (CC §14.1) |
| 2 | k4 link timing with the control core, VU, AT cluster modes (DSP input mux, capture bank, gather ATT path) and DB | core clock below 250 MHz (D13 restated) | OOC runs per block, the cycle-identical gather fix, registered boundaries, half-rate ATTPV fallback (bit-exact) |
| 3 | AT cluster modes add ≈ 19–27 K LUT per cluster in SLR0 / SLR2 (≈ 50–54 % each) | congestion in the cluster SLRs | per-mode floorplan check in the first OOC run; trim the capture bank (numeric contract change before freeze only) |
| 4 | v++ 2024.2 refuses `ap_ctrl_none` with `m_axi` args or the `ap_clk_2` frequency request | no xclbin with the control core | EX §9.1 fallback (`ap_ctrl_hs`, never run); BUFGCE_DIV clock; checked in the first smoke link |
| 5 | CDC or flush bugs in the rings, the SR sequencer or the issue FIFO | hangs that need a card reset | handshakes instead of multi-step gray counters, two-clock benches at every ratio with jitter mode, mutation campaigns, `report_cdc` |
| 6 | Ibex / riscv-dbg details differ from the design (`RV_HALTED` heuristic, triggers, `crash_dump_o`, SBA codes) | wrong halt detection or missing debug features | all marked *confirm*; `tb/ctrl` cross-checks against `dmstatus`; DMI fallback |
| 7 | GELU16 differs from the CPU's libm-based table in 272 of 65,536 entries (0.415 %; 7 by 2 fp16 ulp, where the table's f32 form cancels; GELU16 is the correctly rounded value in all, VU §6.5); Gemma 4 FA-on baselines are new | residual Gemma 4 KL gap | exhaustive fp16 enumeration (`ref_math_gelu16`, done in W0) and the KL with and without `GGML_GELU_FP16` before the VU RTL freeze |
| 8 | No Gemma 4 31B GGUF on the build host | M7 Gemma coverage limited to 12B | 12B is confirmed; 31B rows stay *confirm* until a GGUF is read |
| 9 | Issue-bound phases (short ops on small models, the `graph` executor's per-op cost) | DDR idles, mostly for 0.5B | ops cover all heads per op, line-buffered SPM alias, measured with `MVM_EMPTY` and TRACE_MARK; an "issue from SPM line" path later. *Measured* (SW §7.8, executor optimisation): M6's 0.5B layer is never issue-bound (the MVM idles 0.0 % on issue; per record 387 RISC-V cycles of spin-free work with deep queues, 401 with U200 queue depths, 437 by W2a's method, Ibex maxperf); the M7 projection loses 6.2 % of a layer to issue with 25 generic short VU records (1.6 % at 250 MHz), so W2b's VU kinds take the MUL_MAT fast path's form; a deeper DMA queue is the hardware fallback. *Measured at the W2b integration* (SW §7.8, `fw_m7_layer_perf`): a real Qwen2.5-0.5B M7 decode layer (15 records) loses 5.4 % of its launch to issue with the U200 queue depths, 2.2 % with only the DMA queue 31 deep, 2.5 % at 250 MHz; the loss comes from the 8-entry DMA queue (the RISC-V blocks on credits and reaches FLASH_ATTN late) and FLASH_ATTN's 16 µs of planning on the critical path, not from the short records (≈ 340 RISC-V cycles each, RMS_NORM 395, ROPE 420). After the W2b fix (every short kind ≤ 350, RMS_NORM 324, ROPE 325): 5.1 % / 2.1 % / 2.5 % |
| 10 | Attention of D = 64 models is gather-bound (61 %), and the VU's merge tail is exposed per layer | 0.5B attention slower than DDR allows at long context | accepted for the test model; wider gather later |
| 11 | D16: the card has no working path with the control core until W6 | no card runs in between | the software path runs end to end on the simulator before any RTL (W3) |
| 12 | 4-hour k4 links on a 14 GB host | slow hardware iteration | OOC runs and k1 links first; at most 2–3 k4 links in M6 |
| 13 | Simulator speed for 32B and Gemma runs with RISC-V interpretation | slow sim gates | fast-forward, race detector off in model runs, short prompts for identity runs |
| 14 | Toolchain availability (xPack multilib, Spike dependencies) | blocks W2 | libgcc fallback multilib; Clang via conda; Spike only for `rv_*` tests |
| 15 | A wedged unit has no cancel | only a card reset recovers | progress watchdog, DB evidence and dumps first (DB §7) |
| 16 | PCIe MMIO read latency unknown (spec open item) | launch round trip above 10 µs | still ≤ 1/3 of the 35 µs measured on the card before the control core was added; M7 makes one launch per token |

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
- **Numerics:** the Gemma 4 FA-on KL baselines. (Measured in W0: the exhaustive accuracy figures
  for EXPK, SIGMOID, SILU, GELU, TANH, SIN / COS, all within 2 × the *model* figure, VU §6.9; the
  GELU16 mismatch rate, 0.415 %, VU §6.5.)
- **Models:** Gemma 4 31B dimensions; whether M8 brings back the Hadamard KV rotation and an f32
  `tok_embd` table.
- **Capacity:** re-check `LQ_DEPTH` = 8 against M7 and M8 launch patterns.
- Observation: `git status` shows `third_party/llama.cpp` with modified content; the design
  work made no change there.

---

## 10. Implementation work packages (ordered)

The software path runs end to end on the simulator before any RTL is needed; the RTL follows
the area documents in parallel and is checked against the simulator; the card comes last. Package
numbers P* refer to SW §11 (sizes and dependencies there).

**W0 — contract and tables (first M6 commits).**
1. P1.1: the table source `sw/tables/tsr_tables.py` with every row of EX §5–§8, CC §5, DB §10,
   VU §5, AT §5; `gen_tables.py` emitting the SV package, `tsr_tables.h`, `mmio/vu/at.hpp`, tb
   Python; `--check` in lint.
2. The spec and document edits of §11.

**W1 — simulator and references (P1.2–P1.13).**
1. P1.3 references first: `tsr_ref_math.c` (EXPK, composites incl. GELU16, SIN / COS) with the
   independent numpy model and exhaustive tests (`ref_math_*`, `ref_math_gelu16`); VU references
   (reductions with the z() rule, ROPE with both angle forms, TOPK, DQROW); AT references (plane
   layout, KVST, ATTQK, ATTPV block float, the vector ops, `tsr_ref_fa`).
2. P1.2 EX's CSR model (loader with VERIFY, LQ / LCQ / log rings, watermarks, `SOFT_RESET`,
   `Q_*`, PERF); P1.6 the queue model (checker with OPERAND and four extents, `ISSUE_PIDX`,
   I1–I3); P1.4 / P1.5 VU and AT unit semantics with cycle models.
3. P1.7 `tsr_rv` with the firmware and test bus profiles; P1.11 toolchain and Spike harness; P1.8
   scheduler; P1.9 race detector; P1.10 the trace format, `tsr-trace-gen`, `tsr-trace-conv` and the set
   generation / conversion; P1.12 the DM model; P1.13 DB's simulator models.
   Exit: `sim_basic`, `sim_queue`, `sim_vu`, `sim_at`, `rv_isa`, `rv_arch`, `rv_random`,
   `sim_race`, `sim_sched_determinism`, `trace_v2_roundtrip` green.

**W2 — firmware SDK and kernels on the simulator (P2.1–P2.7).** Build (`ExternalProject`,
`tsr-elf`), crt0 and the runtime R1–R10 with DB's trap and log contract, the header library and
generated builders, `plan.hpp`, the OPL executor, the M6 kernels (`graph` with MUL_MAT,
`selftest`, `replay`), then the M7 kinds (norms, element ops, GELU16, RoPE, GET_ROWS, SET_ROWS_KV,
FLASH_ATTN per AT §6.6) with their C++ references. Exit: `fw_runtime`, `fw_kernels_host`,
`fw_kernels_rv` (8 schedules, race fatal), `fw_selftest_sim`, `fw_replay_sim`.
*Status (2026-10-31):* W2a (P2.1–P2.5) done (92f0b17, executor optimisation 41eb0da). W2b (P2.6–P2.7)
integrated: the eleven M7 kinds with their planners (`tsr/plan_elem.hpp`, `plan_norm.hpp`,
`plan_attn.hpp`) and references; exit tests green: `fw_kernels_elem_host` / `_rv` / `_rv_errors` /
`_rv_shapes` (slow), `fw_kernels_normrope_host` / `_rv` / `_rv_prefill` / `_rv_refuse` /
`_rv_mutant_race`, `fw_kvfa_plan`, `fw_kvfa_rv_*` (8 schedules, race fatal; 16K cells slow),
`fw_kvfa_host`, `fw_kvfa_kl`, `fw_exec_rv` (sc_m7), `fw_opl_m7`, `fw_plan_cxx17`, and whole decoder
layers: `fw_m7_layer_qwen05b` (Qwen2.5-0.5B layer 0 with its GGUF weights, decode and an 8-token
prefill tile) and `fw_m7_layer_gemma12b` (a Gemma 4 12B SWA layer shape), every record bit-exact
against the composed references under 8 schedules with the race detector fatal, and
`fw_m7_layer_ggml` (the same layers on ggml-cpu: layer-output NMSE 3·10⁻¹⁴ … 5·10⁻⁶, attention
≤ 8·10⁻¹⁰; with the device's KV cache ≤ 5·10⁻¹³). W2b fix (2026-10-31): every short kind ≤ 350
RISC-V cycles of work at the 0.5B decode shapes (`fw_exec_perf_m7 --strict`: RMS_NORM 324, ROPE 325,
SCALE with a bias 336; SW-39); the one-tile block is 1,024 lines (SW-36), so MUL_MAT's
prefill T-tiles are as before W2b. Open: decode records outside the common shapes still above 350
(Gemma 4's per-head norms 405–455, ROPE with a freq table or freq_scale ≠ 1 439–475, records that
change s-register values); the issue loss of SW §7.8 (DMA queue depth, FLASH_ATTN planning).

**W3 — host, backend and tools on the simulator (P3.1–P3.8).** The transport (EX §10, debug-aware
wait and recovery ladder; sim, mock, vkernel and xrt kdevs), the backend's M6 OPL builder, tools
(`tsr-selftest`, trace tools, `tsr-log`, klcheck / numcheck with FA on), **M6 on sim** (P3.4:
0.5B identity and the 32B KL on the simulator), the M7 backend (supports_op, KV buffers, MSUM
producer, compute heap), **M7 on sim** (P3.6: one launch per token for the tiny and real models),
`libtsr_dbg` / `tsr-dbg` (P3.7) and `tsr-prof trw` / TSR_STATS (P3.8).
*Status (2026-10-31):* W3a (P3.1–P3.4) done (83ac601): the transport, the M6 backend and tools; M6 on
sim: 0.5B 8 × 64 bit-identical across sim, replay and mock (KL ratio 1.026, 145 launches
per token), 32B Q4_K_M KL ratio 1.285 (top-1 98.44 %). W3b (P3.5–P3.6) integrated: the M7 backend
(`tsr-build`: one lowering for supports_op and the builder, fusion, byte-range dependencies, the
preload table, FA_NEW, the per-graph plan cache; KV buffers `tsr-kvbuf`; the mask-summary producer
`tsr-msum`; compute windows up to 512 MiB), the tiny Qwen2 / Gemma 4 models and graph tests, per-op
test-backend-ops ctests, `ggml_tesseract_attn` / `_kvbuf` / `_kvstate`; no new llama.cpp patch.
**M7 on sim:** exactly one launch per decode token and one Tesseract split per decode graph for
Qwen2.5-0.5B Q8_0 (504 decodes), Gemma 4 12B Q4_K_M and Q8_0 (64 decodes each, 0 staged chunks at
`-ub 512`) and Qwen2.5-32B Q4_K_M (smoke); KL vs CPU within klcheck's criterion (0.5B 8 × 64 ratio
0.914; Gemma 4 12B 2 × 33: Q4_K_M 0.680, Q8_0 0.843); 0.5B and Gemma 4 12B replays bit-identical,
0.5B also under a random schedule with the race detector fatal; device time per decode token 2.48 M
cycles for 0.5B in one launch (M6: 2.63 M in 145 launches) and 83.5 M (0.30 s at 276.3 MHz) for 32B. Decisions SW-40 … SW-53; the table is SW §10.4. The W3b review fixes: results that are
views need rows of whole lines, KV views as llama.cpp builds them, M7 ops with operands in host memory
(control vectors, `TSR_FORCE_HOST`) and FLASH_ATTN ubatches of more than 680 tokens stay on the CPU
instead of aborting, clear reports for multi-stream KV caches. Open: prefill ubatches
above ≈ 450 (Qwen2.5) / ≈ 240 (Gemma 4) tokens need several launches (256-line preload area), the
full 32B klcheck and the long-prompt runs, P3.7, P3.8.

**W4 — RTL (P4), each block bit-exact against the simulator.**
1. Core-domain issue path: `tsr_chk`, `tsr_issue` (queues, heads, SYNC, hard hold), `tsr_csr`
   (EX map, rings, SR sequencer, loader with VERIFY); `tb/chk`, `tb/issue`.
2. SPM 2R2W with `tsr_spm_cport` and the DMA LD reservation; `tb/spm`, `tb/dma`.
3. Control subsystem: Ibex, riscv-dbg, TCMs, control window, CDC; `tb/ctrl`; gate T1.
4. VU (incl. AT's vector ops) with `tb/vu`; the out-of-context utilisation run fixes the knob
   setting against the SLR1 gate.
5. AT: KVST in the DMA, the cluster ATTQK / ATTPV modes, the gather ATT path; `tb/at`, `tb/tech`.
6. DB block (stop logic, history, trace writer, probes, DMI gating); `tb/dbg`.
7. Gather job-pipe fix (`tsr_rslice` FORM 1); the adapter (`ap_ctrl_none`, `MEM_EN`, `ap_clk_2`).

**W5 — integration on Verilator and Vivado (P5.1–P5.6).** Tile-only replay of every trace set
into the issue FIFO; the full-system bench with Ibex RVFI lockstep against `tsr_rv` at every clock
ratio; the vkernel; riscv-tests on RTL; 0.5B on vkernel bit-identical to sim; Vivado T2 (after the
VU OOC run), T4 (k1 link), T3 (k4 link ≥ 250 MHz with the final timing and CDC checks).

**W6 — card (P6.1–P6.6).** Bring-up (identity, bind, load and VERIFY, HELLO, PING, launch round
trip and rate), `tsr-selftest` with the simulator shadow and the trace sets, the GDB smoke, **the
M6 gate**, then **the M7 gate**, then performance (DDR utilisation U with the H / G / F / E split,
launch costs).

---

## 11. Edits outside `docs/design` (W0)

W0 (2026-10-30) edited `docs/spec.md` and the other documents outside this directory to follow
the design set; AT §14 and SW §15 list the edits.

---

## 12. Implementation map and verification status

Checked on 2026-11-02 against `2dcdadb`, the source corrections in `47832fc`, the earlier
architecture audit and the saved verification logs. Status corrections made during this review
change documentation only. This section maps the repository; it does not change the area contracts.
The active build and its eventual result belong in [handoff](../handoff.md).

### 12.1 Code, contracts and boundaries

| Area | Implementation entry points | Contract and boundary |
|---|---|---|
| Encodings and versions | `sw/tables/tsr_tables.py`, `gen_tables.py`; `sw/ctrl/include/tsr/version.h`, `opl.hpp` | EX §5–§8, SW §6.9. Generated SV/C/C++/Python tables share encodings; Core capability, Basis release, ABI and kind numerics revisions are separate identifiers. |
| Numerical oracle and simulator | `sw/sim/tsr_ref*.c`, `tsr_sim.c`, `tsr_q.c`, `tsr_rv.c`, `tsr_bus.c`, `tsr_sched.c`, `tsr_race.c`, `tsr_trace.c` | SW §4–§5, VU and AT numerics. Runs firmware and architectural unit semantics. Operation cycle estimates do not model every RTL pipeline handshake. |
| llama.cpp integration | `third_party/patches/0001-ggml-tesseract.patch`, `sw/ggml-tesseract/ggml-tesseract.cpp`, `tsr-build`, `tsr-opl`, `tsr-opl-dev`, `tsr-kvbuf`, `tsr-msum` | SW §8–§9.3. The patch registers the backend; implementation stays in `sw/`. Shared shape/planner checks drive both support decisions and lowering. Cached lists, dependencies, weight/KV layout and masks are host responsibilities. |
| PyTorch and direct runtime | `sw/torch/tesseract_torch/`, `sw/torch/csrc/tsr_torch.cpp`, `sw/rt/tsr_rt.cpp` | SW §9.4 and `torch-backend.md` §3–§8. Current partitioning offloads weight matmuls; it does not fuse the whole M7 device graph. CPU fallbacks and weight-cache mutation limits remain. |
| Device lifecycle and protocol | `sw/transport/tsr_transport.c`, `tsr_kdev.cpp`, `tsr_image.cpp`; sim/mock/vkernel/XRT implementations | EX §2–§4, §9–§10; SW §9.1. One protocol handles bind, load, VERIFY, HELLO, launch, LCQ, logging and recovery over raw device access. The host does not push tile instructions. |
| Basis runtime and kernels | `sw/ctrl/rt/`, `kernels/graph.cpp`, `k_*.cpp`, `replay.cpp`, `selftest.cpp`, `include/tsr/plan*.hpp`, `exec.hpp`, `backend_dev.hpp` | SW §6–§7. Freestanding RV32 firmware owns tile issue; shared planners keep host acceptance and device execution aligned. Image kind declarations and revisions describe the handlers actually linked. |
| Control core and crossings | `rtl/ctrl/tsr_ctrl*.sv`, `tsr_ctl_win.sv`, `tsr_rv_bus.sv`, `cdc/`, `VENDOR.md` | CC §3–§6, §11. Real Ibex and riscv-dbg, TCMs, loader, control windows, tagged handshakes, gray counters/FIFOs and reset sequencing. Vendor internals are separate pinned projects. |
| Issue, CSR and memory | `rtl/core/tsr_core*.sv`, `tsr_chk*.sv`, `tsr_issue.sv`, `tsr_csr.sv`, `tsr_spm*.sv`, `tsr_mem_arb.sv` | CC §7–§10; EX §4–§8. Per-unit in-order queues run concurrently when dependencies permit. Software supplies memory dependencies; range checking does not infer them. SPM response space must be reserved before reads. |
| Compute | `tsr_dma*.sv`, `tsr_qact.sv`, `tsr_mvm*.sv`, `tsr_vu*.sv`, `tsr_fp*.sv`, `rtl/tech/` | VU, AT, CC §10. DMA moves activation/result/KV data; each MVM cluster streams its weights directly from its own DDR channel. Attention reuses DMA, MVM and VU queues; queue 4 is absent. |
| Debug and observation | `rtl/core/tsr_dbg*.sv`, `sw/tools/dbg/`, `tsr_dbg.cpp`, `tsr_prof.cpp`, `tsr_log.cpp` | DB §3–§10; `prof-dbg.md` §1. GDB/RSP, embedded service, tile stops/history, traces and stats exist. Sample/apply completion and one diagnostic bit remain incomplete in RTL. |
| Platform and build | `rtl/platform/alveo_xrt/`, `hw/u200/`, `sw/xrt-driver/` | EX §9, CC §13–§14; `platform-u200.md`, `xrt-host.md`. Two clocks, `MEM_EN`, AXI bridges, source-hashed XO packaging, link hooks and floorplan checks. Driver patches and old card measurements do not prove the new firmware design works on the card. |
| Verification and support | `tb/`, CMake tests, `traces/`, `sw/tools/klcheck*`, `numcheck*`; `hw/viz/`, `docs/fig/src/`, `docs/logo/src/` | SW §10. Tests use different real/model boundaries. Trace generators and manifest preserve fixture provenance; visualisation scripts produce explanatory assets, not timing or locality signoff. |

### 12.2 Normal execution and failure

For a supported M7 decode graph, `tsr_backend_graph_compute` uses the plan cache and shared
lowering. `tsr_build_and_run` / `tsr_flush` eventually reach `tsr_run_list`, `tsr_opl_dev::run`
and `tsr_transport_call`. The transport loads and checks the image when needed, writes inputs
and the op list, then posts a launch. Basis dispatches the `graph` kernel, whose handlers use
the shared planners and issue tile operations with dependency tickets. Tested common decode
graphs use one launch per token; unsupported operations, prefill cuts and other shapes can
change that count. PyTorch's current path still launches each weight matmul separately.

A concrete RTL smoke is `tb/core/fw/fw_core.c`: LD activation data, QACT, MVM, ST and FENCE.
QACT waits for LD's completion ticket before reading SPM. MVM broadcasts quantized activations
from SPM to the clusters, while each cluster's `tsr_mvm_ws` reads its own weight rows directly
through the channel arbiter. Weights are not copied through SPM by DMA. The gather assembles
results with byte enables; MVM completes after its required reads, results and writes drain.
ST then waits for all write responses before its ticket completes. The firmware joins work
before posting the LCQ result, and the host reads output bytes after that completion.

Failure recovery is not graph replay. LCQ reads and acknowledgement must succeed before host
answers/events/credits change; the corrected transport preserves a batch across an ambiguous
ack failure. An uncertain reset blocks stale consumption until reset and counter resynchronisation
complete. A warm restart can reuse the image and device allocation, but the failed graph still
returns failure to its caller. Already-written tensor bytes are not rolled back, and the caller
must decide how to rebuild or retry its work.

### 12.3 What the verification establishes

This table records the earlier architecture-audit evidence. The November 6
candidate checks and their limits are in `docs/checkpoint-2026-11-06.md` and
`docs/handoff.md`; older card or timing results do not qualify that candidate.

The C simulator applies a unit's operation atomically at its completion time. RTL instead streams
memory requests and data; the VU can overlap two accepted operations. Oracle agreement establishes
architectural bytes and ordering for the tested inputs. Ready/valid, credits, bank arbitration,
CDC and physical timing need their own RTL and implementation evidence.

| Evidence | Established scope | Limit |
|---|---|---|
| November 2 `build/dev` CTest | 341 registered; 43 labelled slow; 297 executed and passed, one disabled in the recorded non-slow run | LLAMA on, TORCH and XRT off. This is not a fresh PyTorch or real XRT run. Model-dependent registration and opt-in tests have separate prerequisites. |
| November 2 RTL pytest and lint | 290 passed, 115 opt-in skips; the technology-lint node passed separately after its wrapper environment was corrected; 71 clock/floorplan fixtures pass after the clock-calculator and 250 MHz acceptance changes | The original suite had that one execution failure. Small synthesis and fixture checks are not full-K4 routing signoff. |
| `tb/core` / `core_m7` | Real Ibex, production firmware, issue path and VU/attention/KVST integration against the simulator at small shapes | Does not establish final four-channel K4 placement, DDR performance or every model shape. |
| `tb/at` composite flash-attention test | Real MVM attention modes compared with the composed oracle | Some QACT/KVST/VU operations run on the simulator with mirrored memory; that test alone does not exercise the entire RTL chain. |
| GDB on `sim` | Real pinned RISC-V GDB smoke via stdio: boot/symbol stops, registers, memory, step, hardware breakpoint, launch, backtrace, `finish`, history, detach. RSP/embedded/drop tests also pass. | Integrated current RTL/vkernel/card GDB is unverified. RTL DMI and DBG module tests are separate evidence. |
| Corrected K4 link | New XO, BD clock/reset split and linked-net CLKNET check passed: core 3.333 ns, RV 6.023 ns, 4,225 RV FFs on the separate RV clock, separate resets. Placement completed with zero errors and actual SLR locality passed. Matching placed SLL counts fell from 28,506 to 17,205. | SLR1 CLB sites remain 99.84 % occupied. Routing, final setup/hold and frequency signoff, post-route CDC and card results remain pending. The initial 300/166.7 MHz request check is separate from final T3 acceptance at core ≥ 250 MHz. |
| Historical PyTorch, XRT, bandwidth and card records | Evidence for the exact older image/configuration named by each record | Does not transfer to the control-core K4 image. The current two-clock vkernel server is not in the tree. |

### 12.4 Progress and remaining plan

| Work | Current state | Next required evidence |
|---|---|---|
| W0–W2: tables, oracle, Basis runtime/planners/kernels | Implemented on the simulator, including kind revision gates | Preserve numerical/ABI contracts when changing acceptance or output bits. |
| W3: host, backend and tools | M6/M7 simulator paths and GDB tools implemented; trace export/stats implemented | Long prompts and later frontend coverage remain separate from the recorded short model gates. |
| P3.8 and deferred tools | `tsr-prof trw` / `stats` exist | Automatic profiling, `report`, periodic firmware PC sampling and `tsr-dbg diverge` remain inactive or unimplemented. |
| W4: control/VU/AT/DB RTL | Integrated with module and small-system tests | Complete the known sample/apply handshake and diagnostic gap; retain dense-hit/reset regressions. |
| W5: integration and physical implementation | The November 6 T4 / VU 8/8/4 local-intermediate + masked-KVST candidate passed placement and pre-route clock/CDC checks; routing is active at the handoff checkpoint | Qualify legal route, setup/hold and >=250 MHz. See `docs/handoff.md` for the actual checkout and latest state. Two-clock vkernel, current full-system GDB/card model checks and T2–T4 remain incomplete. |
| W6: current image on card | Pending a usable image | Identity/bind, LOAD/VERIFY/HELLO, selftest, launch costs, GDB, M6/M7 logits/KL and launch-count gates, then DDR utilisation and performance. |
| M8/M9 and later | Future product integration | MoE, broader PyTorch fusion, resident decode, speculation, new layer types and multi-card work follow the card gate. |

T1 measured the RV32MFast control core at 6 ns with the required margin. That is not T2–T4:
the earlier whole-core placed WNS was negative, the old K4 route failed, and k1 did not close.
The route-only experiment failed on 2026-11-03. The localized DDR0/3, paired-gather and
balanced full-size VU attempt also failed routing on 2026-11-04, with far fewer overlaps.
Both error checkpoints are preserved. Later T4 and placement experiments are recorded in
`docs/handoff.md`. The November 6 candidate retains eight front/EXP and four back VU lanes,
adds local intermediate paths and masked KVST, and is undergoing native routing. Its placed
SLL count fell, but central congestion persisted and same-stage estimated timing worsened.
A routed checkpoint with negative setup slack can answer routability without passing T3.
Final T3 requires core ≥ 250 MHz and passing setup, hold, RISC-V timing and CDC checks at the
selected frequencies. K4 requests 250 MHz; missing 300 MHz alone is not a
reason to repeat a run that passes those checks at 250 MHz or higher.

Optional TOPK and DQROW have conditional RTL modules and tests, but are disabled in the U200
configuration. Their M8 rollout is not the same as missing source modules. VMASK remains a
reserved future capability with no registered current simulator handler. The `tb/fastreplay`
directory currently contains a memory-model selftest, not a full-system firmware replay server.

Known current defects and future features must stay separate. `RV_SAMPLE` / `RVWP_APPLY` can
acknowledge before CDC completion; `DBG_STATUS.SBA_REFUSED` lacks its RTL setter. The depth-16
watchpoint FIFO covers the supported core-at-least-as-fast-as-RV case, not arbitrary sustained
RV-faster-than-core traffic. These limits are not hidden by passing simulator tests.

### 12.5 Review coverage and reuse

The review classified all 759 tracked entries and reused the earlier architecture audit and
implementation regressions. File-level records distinguish actual body/call-path review,
specific earlier review evidence, interface inspection, generated files, assets and unreviewed
internals. Classification is not a claim that every line was deeply audited or every input tested.
The combined record has 63 body/call-path reviews, 144 reused reviews, 468 interface inspections,
18 generated files, 55 assets, four vendor boundaries and seven test fixtures without body review.
Vendor submodules were checked at their pinned integration boundary; their full internals were
not re-audited. Figure assets were inventoried, not visually revalidated or regenerated.

Local detailed records are under `build/analysis/repository_map_20261102/`: `software.txt`,
`datapath.txt`, `platform.txt`, file-level coverage and the registered CTest inventory. Earlier
reproduction and fix evidence remains in `build/analysis/architecture_audit/` and
`build/analysis/implementation_fix/`. These generated analysis records are ignored by git; this
implementation map and the handoff preserve the durable summary. No additional tests, synthesis,
DCP loads or links were run for this mapping.

### 12.6 MoE and GDN extension checkpoints

**GDN software baseline (2026-11-09, branch `gdn-sw`).** The recurrence and the surrounding operations of the
Qwen3.5 / 3.8 layers are implemented in software on the existing VU: the kinds SOFTPLUS (0x2D), SSM_CONV (0x2E) and
GATED_DELTA_NET (0x2F) (firmware, planners and C oracle, numerics revision 1), the ggml-tesseract lowering of the
pinned llama.cpp graph (no llama.cpp patch), the recurrent state as device tensors, and tests at every layer. No RTL
changed and the card was not used. Qwen3.8 2B (Q4_K_M) runs its whole decode graph on the simulator in one launch
per token at klcheck ratio 1.06 (512 positions, top-1 98.05 %), and the first four layers of the 27B (the 27B's
shapes: 48 value heads, C = 10240) agree with the CPU at cosine 0.99999. The composed recurrence costs 8.6 K core
cycles per head and token: for the 27B (48 layers, 48 heads) 20.4 M cycles per decode token, 102 ms at 200 MHz,
against the 23.8 ms estimated for the fused op (branch `gdn-vu`, VU §7.12). Design, decisions SW-67 to SW-74, measurements and the open
items: `docs/design/06-software-verification.md` §7.10, `docs/klcheck.md`, `docs/performance.md`.

**MoE status (November 8, branch `moe`).** The software stack runs Gemma 4 26B-A4B on the simulator, in the
M6 form (the host issues the matrix products, llama.cpp's CPU ops select the experts) and in the M7 form (the
router, the top-k, the expert products and the combine in one launch per token). No RTL changed and the TOPK RTL
stays off: the selection is integer compares on the RISC-V in the new firmware kind MOE_ROUTE (0x1E); MUL_MAT_ID
(0x1F) runs the expert MVMs at `W_OFF + id × EXP_STRIDE`. Expert tensors are placed per channel with the expert
stride, the router is converted to Q8_0 at load, and the backend fuses llama.cpp's router nodes and the per-expert
scale nodes into one record each. Prefill still reads each token's experts separately; token grouping is designed
(SW-66), not built. Qwen3-30B-A3B (`qwen3moe`) runs on the same kinds with no change (SW-77). The details and decisions SW-60 to SW-66 and SW-77 are in
[SW](06-software-verification.md) §7.9; the measurements are in [klcheck](../klcheck.md) and the handoff. The
paragraphs below are the planning text that preceded it.

The local-intermediate candidate's checked shared window helper now represents
expert base/stride and input/output token origins while preserving original N
and owner packing. The executing dense commands still use expert zero and
identity token order. Explicit token IDs, 3D expert packing, MUL_MAT_ID,
grouped expert execution and combine remain unimplemented. See
`docs/intermediate-locality-plan.md` §0 for the verified scope.

Priority update (November 5): after the active K4 build, MoE is the first model
extension. Local intermediate work is implemented with expert-dependent weight
bases/strides and token-to-output mapping in mind. Preserve independent MVM and
VU arithmetic and common llama.cpp/PyTorch execution semantics. Do not equate a
local-memory capability or fused dense epilogue with implemented MoE support.

Planning decisions retained on 2026-11-02 for work after the M7 card gate. Keep the MVM
clusters and VU as independent arithmetic units, and preserve the four 512-bit weight
streams. A new Basis op-list kind is a firmware handler that issues tile operations;
it does not by itself require a new RTL opcode or a RISC-V ISA extension.

- **MoE:** use the existing MVM clusters with expert-dependent weight addresses. Add
  three-dimensional expert packing and per-channel expert strides to the backend, then
  `MUL_MAT_ID`, router selection and expert-combine handlers/lowering. The current weight
  loader accepts only two-dimensional device weights (`ggml-tesseract.cpp`), and these
  MoE kinds are absent from `tsr/opl.hpp`. Token grouping by expert is needed to recover
  weight reuse in prefill and speculative verification; T_MAX alone does not guarantee it.
- **MoE hardware:** the planned device selection path enables the existing conditional
  `tsr_vu_topk` RTL through the platform configuration and capability reporting. TOPK is
  disabled in the current K4 wrapper, so this needs a new hardware image and integration
  checks. Expert MVMs can retain the existing weight and result paths. DQROW is a separate
  M8 device-embedding feature, not a prerequisite for expert matrix multiplication.
  VU §7.10 records about 5.6 K LUT for TOPK in the earlier VU synthesis. This is not a
  measured K4 placement increment; check local SLR capacity before enabling it.
- **GDN baseline (implemented, SW §7.10):** the FP32 recurrent state stays in DDR and the
  recurrence and the SSM convolution are firmware kinds that compose VOP, RED, LD and ST.
  Head tiles, scratch use and dependencies are defined in SW §7.10 (SW-67 to SW-74).
  For the planned 48-layer, 48-head, 128-by-128 state, one head is 64 KiB, one layer is
  3 MiB and one sequence is 144 MiB, excluding convolution history and snapshots.
  Start with head-sized SPM tiles. Local URAM buffers or fused state-update RTL remain
  performance options after measuring this baseline, not prerequisites already selected.
- **GDN surrounding operations (implemented):** the norms, gates and rotary variant are
  lowered (SW §7.10). SILU, SIGMOID and EXP already have VU functions; SOFTPLUS has none and is
  composed from VU functions in 20 VOPs, within 5 ulp of float64 (SW-69). IMROPE is the NEOX
  rotation for text positions, with a guard. Full model support also needs sequences of
  more than one slot per ubatch, snapshots and MTP, which are not covered. No new wide SLR
  interface is assumed by the baseline.
- **Numerics and cost:** specify rounding, reduction order and state-update order in the
  C oracle, with kind revisions under `docs/versioning.md`. Check long-sequence state and
  logits against llama.cpp, as well as oracle/RTL agreement. The earlier 8 ms GDN estimate
  counts roughly one pass over 37,748,736 state elements at 16 lanes and 300 MHz. Decay,
  prediction, update, output reduction, DMA/SPM contention and command costs must all be
  budgeted. Neither negligible GDN overhead nor a fixed CPU-fallback penalty is established.
  For MoE, also check router precision, ties and the probability-scaling/accumulation order
  of the illustrative kernel in `docs/spec.md` §13.
- **llama.cpp state contract:** use its graph/backend and recurrent-memory interfaces.
  Specify buffer ownership, layout, initialisation, sequence isolation, save/load and
  reset, including convolution history. CPU fallback must respect those buffers rather
  than assuming every unsupported stateful node can move freely. Connect MTP rejection
  to accepted-prefix state restoration; truncating an attention KV index is insufficient.
  The pinned `llama-memory-recurrent.cpp` already has bounded snapshot rollback. Test
  zero, partial and full draft acceptance before claiming speculative support. Decode and
  prefill schedules, snapshot traffic and expert divergence need separate cost checks.

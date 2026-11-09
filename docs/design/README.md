# Design documents

This directory is the design contract of Tesseract Core, the hardware, and Tesseract Basis, the
firmware SDK and host software that drive it. [docs/spec.md](../spec.md) is the master specification
and summarises these documents. The six area documents define the details.

When two sources disagree, the order of authority is `docs/spec.md`, the C simulator in `sw/sim`, the
documents in this directory, then the other documents.

## 1. Purpose and how to read

Every tile op is issued by an embedded RISC-V control core. The host loads a firmware image and
launches kernels; it never touches the issue port. The documents specify the hardware and the
software that runs on it and on the simulator.

Read them in this order: EX (the host-visible model and every global table), CC (the RTL that
implements it), VU and AT (vector and attention compute), DB (debug and profiling), SW (simulator,
firmware, host stack, verification). Section 4 lists where each global table lives. Every other
mention of a table is a pointer to that place.

| File | Short form | Covers |
|---|---|---|
| `01-execution-model.md` | EX | Launch protocol, host CSR map, opcode, unit, error-code and PERF tables, XRT adapter, transport API, DRAM map, build order |
| `02-vector-unit.md` | VU | VU ISA and bit-exact numerics (EXP, RECIP, SIN/COS, GELU, RoPE, reductions), VU microarchitecture, fused Gated DeltaNet op, execution of AT's vector ops |
| `03-attention-kv.md` | AT | KV cache layout, attention ops and numerics, the flash-attention sequence, masks |
| `04-control-core.md` | CC | Ibex and debug module, RISC-V address map and control window, issue path, SPM banks, clock-domain crossings, resets, floorplan and timing |
| `05-debug.md` | DB | GDB over PCIe, tile-level debug, logs and crash records, recovery, profiler, DBG registers |
| `06-software-verification.md` | SW | Simulator and references, trace format, firmware SDK and kernels, op-list format, versions, llama.cpp backend, verification |

## 2. Execution modes and standing decisions

The backend runs a model in one of two execution modes. `tsr-top` shows the mode in its MODE column,
and the backend names it in error messages. A third value, `rt`, marks the direct runtime `tsr_rt`
without a ggml graph.

- M6 runs `MUL_MAT` and `MUL_MAT_ID` on the device and everything else on the CPU, attention included
  (`offload_kqv` is false, `-nkvo`). Each graph split is one launch. It is selected with
  `TSR_KINDS=m6`, or `TSR_KINDS=m6,+MUL_MAT_ID` for a MoE model.
- M7 runs every supported kind on the device, with the KV cache and attention. One decode token is one
  launch. It is the default: the device takes every kind of the booted image.

Both modes use the `graph` kernel and the same op-list format (SW §7). The pass criteria that compare
a mode with the CPU are in SW §10.4.

Standing decisions D1 to D18 are listed in [spec](../spec.md) section 16. The ones that shape these
documents:

- D14: the host loads firmware and launches kernels that run on the RISC-V. Every tile op is issued by
  the RISC-V. There is one s0–s15 scalar bank.
- D15: the host has no debug access to the issue port. Bring-up, the card selftest and trace replay
  are RISC-V kernels.
- D16: there is one execution path in the tree. The software path runs end to end on the simulator
  before any RTL.
- D17: the device KV cache is q8_0. Attention arrives as llama.cpp `FLASH_ATTN_EXT` (flash attention
  on; the pinned llama.cpp requires it for a quantized V cache). The klcheck and numcheck baselines
  use flash attention on.
- D18: the design has full debug: a RISC-V Debug Spec module, halt, resume and step, breakpoints and
  GDB source-level debugging over PCIe (a GDB RSP server on the DMI window, no OpenOCD), and
  tile-level debug.

## 3. Identifiers

Documents are cited by short form and section number: `CC §14` is section 14 of
`04-control-core.md`, and `spec §8` is section 8 of `docs/spec.md`. Section numbers are stable. Code
and other documents cite them, so a section is never renumbered.

Decisions have identifiers made of the short form and a number: EX-1, VU-1, AT-1, CC-1, DB-1, SW-1.
Each is a short entry in the Decisions section of its area document (EX §13, VU §11, AT §11, CC §17,
DB §12, SW §13) with the decision and its reason. A number is never reused. A decision that another
one replaces stays in the table and names its successor.

Numbers marked estimate in the documents come from the design and not from a measurement. Numbers
marked model are the figure of a reference algorithm, which the simulator checks.

## 4. Where each global table lives

| Table | Single definition | Implemented or used by |
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
| Versions: capability, Basis version, ABI, kind revisions | SW §6.9 (capability: EX §7.5) | EX §3, `docs/versioning.md` |
| Machine-readable source of the tables above | `sw/tables/tsr_tables.py` and `gen_tables.py` (EX-30) | RTL package, C header, `mmio/vu/at.hpp`, tb Python |

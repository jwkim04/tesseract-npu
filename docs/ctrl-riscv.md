# Control core (RISC-V)

The control core is an embedded lowRISC Ibex (RV32IMC) in its own clock domain. It runs the firmware kernels that the host
launches, and it issues every tile op asynchronously to per-unit queues. This document gives the hardware and software view of the
control core. The normative text is [CC](design/04-control-core.md), [EX](design/01-execution-model.md) and
[DB](design/05-debug.md). Where this file and `docs/design` differ, `docs/design` wins. Section numbers are cited by code and
other documents.

Decisions D1 to D18 are the top-level decisions of the control core. D7 and D9 are not used.

| ID | Decision | Where |
|---|---|---|
| D1 | An embedded RISC-V control processor: in order, no branch prediction, no superscalar, no out-of-order, no cache | CC §3 |
| D2 | Code, data and stack live in local SRAM (TCMs), so fetch never uses DRAM bandwidth | CC §4.2 |
| D3 | The RISC-V issues the existing 256-bit tile instructions asynchronously: several tile ops in flight on different units, software joins later | §4 |
| D4 | The SPM is mapped into the RISC-V address space through a bus connection | §3.5 |
| D5 | The RISC-V has its own clock domain | CC §11 |
| D6 | Kernels are C++ for stock RISC-V GCC or Clang, with tile and SIMD operations as a header library. A JIT calls precompiled kernels | §8 |
| D8 | Scalar registers: one bank s0 to s15 inside the VU | CC §1 |
| D10 | Decode steps are C++ kernels that issue tile ops directly and compute position-dependent addresses in C++ | the `graph` kernel |
| D11 | One checker, `tsr_chk`, in front of the dispatcher | CC-9 |
| D12 | The RISC-V clock is requested through `v++` as `ap_clk_2`. The release image runs it at 100 MHz with the core at 200 MHz. Software reads `RV_FREQ` and never assumes a ratio | CC §3.3, §11.1 |
| D13 | The control-core RTL is accepted when out-of-context runs meet timing and the linked image meets timing at its clocks, with setup, hold, RISC-V timing and CDC checks passing | CC-23, CC §14.3 |
| D14 | The host does not execute tile ops. It loads the firmware image and launches kernels. The RISC-V issues every tile op | EX §2 to §4 |
| D15 | No host debug access to the issue port. Bring-up, selftest and trace replay run as RISC-V kernels | SW-19 |
| D16 | One execution path in the tree: the software path runs end to end on the simulator before the RTL and the card | `sw/sim` |
| D17 | The device KV cache is q8_0. Attention arrives as llama.cpp `FLASH_ATTN_EXT` | AT |
| D18 | Full debug: a RISC-V Debug Spec module, halt, resume, step, breakpoints, GDB over PCIe through an RSP server on the DMI window, and tile-level debug | §10, DB |

## 1. Datapath facts

A unit takes one command at a time through `cmd_valid`, `cmd_ready`, `cmd` and `done`. Units report no errors, because every check
is made before dispatch, and there is no flush or cancel ([rtl-core.md](rtl-core.md) §3.0). The checks live in `tsr_chk` and the
handshake is at the queue heads (CC §8, §9). Every unit has its own SPM client port (0 DMA, 1 QACT, 2 MVM) and its own memory
source per channel (`tsr_pkg.sv`), with source 0 tied off as AUX (CC-26). The memory arbiter is round robin per burst with
head-of-line response routing, and sources drop their response ready only for reasons local to that source (rule R4), which is the
no-deadlock argument. On the U200 the XRT shell's AXI firewall cuts a control slave that does not answer, and recovery needs a
card reset, so no CSR access may wait on an unbounded event (CC §6.1).

## 2. Core choice and ISA

### 2.1 What the workload needs

A device-resident decode token is a few thousand tile ops, each costing tens of RISC-V cycles (§4.8). The latency-critical points
are device decisions: the TOPK result, expert ids, expert MVMs and sampling. A sustained 50 MIPS is enough, so frequency and IPC
do not decide the choice. Verification maturity does, since the core is part of every device-resident result. So do standard debug
(C++ kernels need GDB), vendor independence, an ASIC path, Verilator support and an integration that leaves the core unmodified.

### 2.2 Candidates

PicoRV32 has no standard traps or debug, SERV is too slow for decision latency, VexRiscv needs a SpinalHDL toolchain, and an
in-house core costs verification and debug work with no gain. Ibex has the most mature open verification (UVM with riscv-dbg and
Spike co-simulation, OpenTitan silicon) and the standard debug spec with an external debug module.

### 2.3 Decision: Ibex

- The configuration is CC §3.2: no `RV32E`, no `RV32B`, `BranchPredictor` 0, `ICache` 0, `PMPEnable` 0, `SecureIbex` 0,
  `MHPMCounterNum` 0, `DbgTriggerEn` 1 with 2 hardware breakpoints, a register file in LUTRAM on Xilinx, and the maxperf pipeline
  (CC-34). [rtl/ctrl/VENDOR.md](../rtl/ctrl/VENDOR.md) lists the parameters of the pinned commit.
- Ibex is SystemVerilog with generic `prim_*` implementations, so the only FPGA-specific piece is the clock gate, mapped through
  `rtl/tech`. Its simple `req`, `gnt` and `rvalid` ports let TCMs and the control window attach without a bus fabric, and the
  issue port is MMIO, so the core stays unmodified. Ibex and riscv-dbg are vendored at pinned commits in `third_party/` and never
  edited (CC §3.1). They keep their own style (asynchronous active-low resets, `prim_*` modules) and their own lint waivers.
- Bus rule for every data-port and instruction-port slave: `gnt` comes from a register (`gnt = ~outstanding_q`), never from the
  address decode. Ibex's `data_addr_o` comes combinationally from the ALU adder and `data_gnt_i` feeds its stall logic in the same
  cycle, so an address-dependent grant would put register file, adder, decode, grant and PC logic on one path. Every wait (credit,
  SPM bridge, core window, debug module) is a late `rvalid`. A TCM access has a fixed next-cycle response and does not set
  `outstanding_q`, so the TCMs take one access per cycle (CC §4.3).

### 2.4 ISA subset and the F question

The ISA is RV32IMC with Zicsr and Zifencei, machine mode only, with no PMP, A, F, D or B. M is for address arithmetic, C makes the
code about a quarter smaller, Zicsr gives traps, interrupts, `mcycle`, `minstret`, WFI and the debug CSRs, and Zifencei gives
`fence.i` after code is written into ITCM. One hart exists, so host handshakes use sequence numbers in the launch queue and
completion ring (EX §4). There is no F. F-capable verified cores are larger, and bit-exactness does not need it: the same C
compiled for RV32 with soft float gives the same bits as on the host. The firmware has no f32 math (SW-8). The host computes every
f32 constant and passes it in records or arguments, and per-element math and the per-head bookkeeping of an online softmax are the
vector unit's (ASMAX, AMERGE, ANORM; AT §6.6).

## 3. Address map

### 3.1 RISC-V view

Addresses are 32-bit byte addresses with no MMU and PMP off. Every unmapped access is an access fault. The map is CC §4.1. The
constants are generated from one table source (`sw/tables/tsr_tables.py`, EX-30).

| Range | Region | Fetch | Load / store |
|---|---|---|---|
| 0x0000_0000 to 0x0001_FFFF | ITCM, 128 KiB. Vector table at `RV_BOOT`, first fetch at `RV_BOOT` + 0x80 | yes | byte, half, word |
| 0x0010_0000 to 0x0010_FFFF | DTCM, 64 KiB: .rodata, .data, .bss, stack | no | byte, half, word |
| 0x1000_0000 to 0x1000_0FFF | debug module memory (halt address +0x800, exception address +0x808) | yes | word |
| 0x2000_0000 to 0x2000_0FFF | control window (§3.2), `clk_rv` registers | no | word |
| 0x2000_1000 to 0x2000_1FFF | core window, a read-only view of the host CSR space at the same offsets (CC §5.3) | no | word loads |
| 0x4000_0000 to 0x403F_FFFF | SPM window, exact (4 MiB): 0x4000_0000 + 64 * line + offset | no | byte, half, word |
| 0x5000_0000 to 0x503F_FFFF | SPM window, line-buffered alias (CC §10.5) | no | byte, half, word |

The two Ibex ports share the debug module's one slave port through a 2:1 arbiter, data first, which stalls the fetch through
`instr_gnt`. The TCMs are `tsr_sram_1r1w`, with the read port serving fetch, data reads and the host's loader and VERIFY reader
(while `RV_RUN` is 0) and the write port taking data stores (while `RV_RUN` is 1) or the host loader. `.rodata` is in DTCM, so
normal code never reads ITCM as data.

### 3.2 Control window (RISC-V side, offsets from 0x2000_0000)

The table is CC §5.1 and §5.2. The groups are the staged words `ISSUE_W0` to `W7`, the issue registers, the per-unit `DONE_SEQ`,
`ISSUED_SEQ` and error records, `QCTRL` and the wait registers, the DB completion records, `SPM_LB_INV`, the launch queue,
completion ring, log ring and runtime state (EX §4.5), `MTIME` and `MTIMECMP`, and DB flags.

- Reads have no side effects (`REC_POP` is a separate write), so a debugger's memory view cannot pop or push anything.
  Control-window writes through the debug module's SBA are refused (D15, CC §4.3).
- The staging set is not reentrant. An interrupt handler, or a debugger running code through the program buffer, that issues
  between the interrupted code's stores corrupts that push and its ticket. Handlers never issue, GDB inferior calls that issue are
  unsupported, and there is no second staging bank.
- The queue heads are held while any of three sources is set: the RISC-V's (`QCTRL`, applied in push order), the host's
  (`CTRL.HOST_HOLD`) and the debug halt's (`HOLD_ON_HALT`). DB adds a per-unit hold. Each source is changed only by its owner (CC
  §9.6).

### 3.3 Host view: core CSRs

EX §5 is the only host CSR map, and CC §6 maps the control core's hardware onto its offsets (`CTL_ID`, `RV_BOOT`, `RV_FREQ`,
`RV_CFG`, the loader, the DMI window). Every host access completes without `clk_rv`: reads and writes use core-domain copies, and
an access that must reach `clk_rv` ends after 4,096 core cycles with a sticky bit, so a stopped or relocking `ap_clk_2` never
hangs the CSR port. `RV_CLK_OK` and `RV_FREQ` tell the host.

### 3.4 Loading the image

EX §3 defines the loader (`LOAD_ADDR`, `LOAD_DATA`, `LOAD_CRC` with a receive CRC and an on-device VERIFY read-back), the 64-bit
image id in `.tsr_id` echoed in HELLO, boot at `RV_BOOT` + 0x80 and the warm restart. The host sequence is CC §6.4, implemented in
the transport ([sw/transport/README.md](../sw/transport/README.md)). A target with no host bus would need a boot ROM
(`HAS_BOOTROM`).

### 3.5 SPM window

The window is little-endian as the SPM layout. Accesses are uncached and non-speculative, stores use `tsr_spm`'s per-byte enables
with no read-modify-write, and Ibex splits a misaligned access into two aligned ones that are not atomic. One access runs at a
time. The bridge answers a store only after `tsr_spm` accepted the write, and the issue logic forwards a push only while no bridge
store is outstanding, so an op pushed after a store sees it (§5.5). The bridge waits at most 4,096 `clk_rv` cycles for the core
side. Then the access ends with a bus error and `BRIDGE_TIMEOUT`, so the debugger can still reach the hart, and a late answer is
discarded by its request tag. The alias serves loads from a one-line buffer. Any store to either alias, any control-window or
core-window access, any rising interrupt line and `SPM_LB_INV` invalidate it, so rule 2(c) of §5.5 holds for the alias too. The
executor reads op-list records in place through it (§8.1).

### 3.6 No DRAM window

The RISC-V has no direct DRAM access. A 4-byte access would cost a full 64-byte beat and share `tsr_mem_arb` with the weight
streams, and it would be a second ordering path into DRAM beside DMA. Data moves only by tile DMA, and DRAM structures shared with
the host (op lists, argument blocks, KV planes) are reached by LD and ST through the SPM.

### 3.7 Interrupts and timer

Ibex's interrupt inputs are level-sensitive, computed or held in `clk_rv` from synchronized values (CC §12.3). `irq_fast[0]` is
the `WAIT_TICKET` ticket done, `[1]` a unit in error state, `[2]` a launch descriptor waiting, `[3]` completion ring room, `[4]`
DB completion records at least half full and `[5]` the `WAIT_CREDIT` unit has a credit. `irq_timer` is MTIME at or above MTIMECMP.
`mtvec` is vectored only, and the runtime installs a vector table at `RV_BOOT`. Joins need no handler, since WFI wakes on an
interrupt pending and enabled in `mie` even with `mstatus.MIE` = 0, and the SDK never sets `mstatus.MIE`. MTIME counts core clock
cycles, so RISC-V timestamps share the time base of the completion records and the PERF counters. The host has no interrupt from
the device (EX-6), and `host_irq_o` is a core boundary output (EX §9.3).

## 4. Asynchronous tile issue

The normative text is CC §7 to §9 and EX §7 and §8. This section records the decisions.

<p align="center"><img src="fig/issue.svg" alt="Upper panel: a kernel on the control core calls issue(op, after(a, b)); the push checks the credit and assigns a ticket, writes the op with up to two dependencies into the 32-entry issue FIFO that crosses from the clk_rv domain to the core clock domain; the checker tsr_chk checks it; the dispatcher appends it to the in-order queue of its unit. Lower panel: the queue contents of one decode layer of flash attention with three KV chunks on the DMA, QACT, MVM and VU units: ATTQK and ATTPV on the MVM, ASMAX and AMERGE on the VU, with ATTPV and AMERGE one chunk behind." width="100%"></p>

### 4.1 Issue port

The port is memory-mapped (§3.2), not a custom instruction, because a custom instruction needs a coprocessor interface that Ibex
does not have. A push stores the words the op uses into `ISSUE_W0` to `W7`, optionally `ISSUE_DEP0` and `DEP1`, then `ISSUE_PUSH`
with the word mask and the REC flag. In one step the issue logic picks the unit from w0's opcode (a static opcode-to-unit table,
CC-7), assigns the ticket {unit, `ISSUED_SEQ[unit]`++}, writes the entry into the 32-entry issue FIFO, resets the deps and sets
`ISSUE_TICKET`. An opcode the issue path does not accept, or whose unit is absent, still gets a ticket on the SYNC unit and
completes with `ILLEGAL_OP`, so every push has a ticket and a completion.

A push needs a free slot of its unit's queue, counted from the push until the unit accepts the op, and a free issue-FIFO entry.
`ISSUE_CREDIT` shows the minimum, at most 15. Software checks the credit before it pushes: `issue()` spins on it, a local read, so
interrupts stay serviceable. The hardware wait is a safety net. A push without a credit completes late when a credit returns, for
at most `ISSUE_WAIT_MAX` = 65,536 `clk_rv` cycles, then it ends with a store access fault and `PUSH_TIMEOUT`. The count does not
advance during a debug stop (DB §3.9). While the store waits, Ibex can take neither an interrupt nor a debug halt, and the wait
can be long in normal operation: with an MVM queue of 4, the sixth outstanding MVM waits for a whole MVM.

### 4.2 Units and queues

The unit table and the depths are CC §7.4. Five units have queues: 0 DMA (LD, ST, KVST; depth 8), 1 QACT (4), 2 MVM (MVM, ATTQK,
ATTPV; 4), 3 VU (VOP, RED, ROPE, SOP and AT's vector ops; 16) and 5 SYNC (NOP, FENCE, QCLEAR; 4). Unit 4 is reserved and absent.
Each queue is in order, and an op starts after the previous op of its unit is done. The depths are the parameters `Q_DEPTH` (EX
§5.1). The issue-FIFO depth is independent of them, because the credit includes the free FIFO entries.

### 4.3 Tickets and joins

A ticket is {unit[31:28], seq[27:0]}. The sequence counts from 0 per unit after a hardware reset and is never reset otherwise, so
tickets stay unique across RISC-V restarts. Ticket (u, s) is done iff (`DONE_SEQ[u]` - s) mod 2^28 is in [1, 2^27). Completions
are in order per unit, so one counter per unit is the whole completion state. A ticket pushed before the last QCLEAR is stale. As
a dependency it is `BAD_DEP`, and `error(t)` on it is undefined. The dispatcher numbers ticketed entries in push order and keeps
the number with each queue entry and error record (`ERR_PIDX`), which orders errors across units. The RISC-V reads the next one as
`ISSUE_PIDX` (CC-32), which is how the executor maps a hardware error back to an op-list record. The joins are `done(t)`,
`join(t)` (spin), `join_irq(t)` (`WAIT_TICKET` = t, then WFI), `group` (the latest ticket per unit) and `join_all()` (pushes a
FENCE and joins it).

### 4.4 Dependencies, SYNC and FENCE

An op has up to 2 dependencies in hardware, and it does not start before both tickets are done. The check is at the head of its
queue against the core-domain completion counters, so a waiting head blocks only its own queue. Deps on the op's own unit are
implied by queue order, deps on one unit keep only the latest, and more than two are joined by a SYNC NOP or a software join.
`BAD_DEP` is detected by the checker in push order: a dep on a ticket not pushed yet, on a reserved unit, or below its unit's
clear point. Dependencies are in hardware because otherwise the RISC-V must join before every dependent push and runs at most one
stage ahead. With them it pushes a whole pipelined loop and the queues hold the schedule (§8.7). `NOP` completes when its deps are
done and is a join point with a ticket. `FENCE` completes when every op pushed before it, on every unit, is done. `QCLEAR` is a
FENCE that then clears the error states (§4.5). The dispatcher takes the snapshot a FENCE waits for when it pops the FIFO in push
order.

Completion rule (invariant I1). Every entry completes only at the head of its queue and only after both its deps are done. That
holds for an entry that runs, a poisoned one, an `ABORTED` one, a `DEP_FAILED` one, a NOP, a FENCE and a QCLEAR. A `BAD_DEP` dep
is ignored. Hence done(t) implies that every op t depends on, transitively, is done. `join_all()` after an error still waits for
every op pushed before it, and the race detector's clock joins along dependency edges (§9.2) are sound.

### 4.5 Errors

`tsr_chk` checks every op at dispatch, in push order, in the spec's order with the issue-path rules first: `ILLEGAL_OP`,
`BAD_DEP`, `BAD_FMT`, `K_RANGE`, `T_RANGE`, `OPERAND`, `SPM_RANGE`, `DRAM_RANGE` (EX §8.1). A failing op enters its queue as a
poisoned entry and changes no state. At the head, after the deps, a poisoned entry completes with its code, an entry of a unit in
error state completes `ABORTED`, an entry with a dep that completed with an error completes `DEP_FAILED`, and otherwise it runs.
Codes 1 to 8 are as in the spec, 11 is `DEP_FAILED`, 12 `BAD_DEP` and 13 `OPERAND`, and 9, 10, 14 and 15 are reserved.

A unit enters error state when one of its entries completes with a nonzero code, at the head, in queue order, never at dispatch.
The first such entry sets `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`, `ERR_W0` and `ERR_PIDX`. While the state holds, every later entry of
the unit completes `ABORTED`, and dependents elsewhere fail in turn. A queue holds a pipeline, so a unit stops on error: running
past a failed LD would compute on stale SPM data silently. `first_error()` is the record with the smallest `ERR_PIDX`. For a valid
program the set of failing entries does not depend on the schedule, so neither does the result.

Errors are visible before completion (invariant I2). `D[u]` does not advance past `ERR_SEQ` until the RISC-V side has acknowledged
unit u's record. `DONE_SEQ` covering a ticket therefore implies that its record is visible, `error(t)` is decidable at any clock
ratio, and a failed op never reads as a success.

QABORT is the RISC-V `QCTRL` bit, taken when the dispatcher pops it, and the host's `CTRL.QABORT`, `LQ_CANCEL`, `RV_RUN` 1 to 0
and `SOFT_RESET`, taken at once. Every entry not started completes `ABORTED`, in order, each after its deps. A running op
completes normally, since units have no cancel. All six unit slots enter error state until a QCLEAR. QCLEAR from the RISC-V is a
SYNC entry with a ticket, in push order, never aborted. It completes when every entry pushed before it has completed, then clears
all error states and records and sets each unit's clear point. From the host it is accepted only when nothing is queued or
running, the issue FIFO is empty and the RISC-V is quiet. Otherwise the answer is `QCLEAR_REFUSED`.

### 4.6 ABORT, SOFT_RESET, resets and halts

The event table is CC §12.1. `RV_RUN` 1 to 0 and the debug module's `ndmreset` run the RISC-V reset sequence (§7.3) with a QABORT.
`SOFT_RESET` is EX's device-level reset, run by a core-domain sequencer that needs no `clk_rv` (CC §12.2, CC-31). A debug halt
continues the queues unless `HOLD_ON_HALT` is set. There is no partial datapath reset and no cancel of a running op (CC-22). A
wedged unit is recovered by a card reset (EX §8.6).

Quiescent state. `QQUIESCENT` means no op is held by a datapath unit, the dispatcher and the issue FIFO are idle, every non-empty
queue is held, and no non-running entry is left to drain. With the RISC-V also halted, the architectural state (RISC-V registers
and CSRs, TCMs, SPM lines written, DRAM) equals the simulator's after the same number of retired RISC-V instructions and the same
completion vector `DONE_SEQ[0..5]`, given that run's co-simulation log (§9.4). With `HOLD_ON_HALT` it is reached at any breakpoint
once the running ops finish. This is the control-core analog of DB §5.8, and random-halt tests check it (§9.3).

### 4.7 Address fields and the scalar bank

Every address field of an op is an immediate. The RISC-V computes absolute addresses and writes them (D14). One s0 to s15 bank
sits inside the VU, read by the host at `SREG` 0x9C0 and by the RISC-V through the core window (CC §1, EX §5.1). Only VU queue ops
write it, in VU queue order, so it follows rules 2 and 4 of §5.5 like SPM bytes.

### 4.8 Issue latency

A push costs 6 to 12 local MMIO accesses plus the instructions that compute the words (CC §7.6). On the core side an idle unit
with done deps starts about 16 to 18 core cycles after the push: FIFO sync 2 to 3, pop 1, checker 8, enqueue 1, head load 1,
evaluation 1, `cmd_valid` 1.

## 5. Concurrency in the core

### 5.1 Block diagram

The block diagram is CC §2.1 and [rtl-core.md](rtl-core.md) §2. `clk_rv` holds Ibex, the TCMs, the debug module and the control
window. The core clock holds the issue FIFO crossing, `tsr_chk`, `tsr_issue` with the unit queues, `tsr_csr`, `tsr_spm` and the
units.

### 5.2 Checker and dispatcher

`tsr_chk` is the decode and check pipeline, one instance serving the dispatcher only. It takes eight core cycles per op and
handles up to four tagged operand extents per op (CC §8.1, §8.2). `tsr_issue` pops the issue FIFO in push order. Control entries
act at the pop, QCLEAR and op entries get the push index and, for FENCE and QCLEAR, the snapshot, and op entries run the checker
and go into their unit queue. Room is guaranteed, because credits count an op from its push until its unit accepts it (CC §8.3).
`S[u]` counts at the accept (credit return). `D[u]` counts at `done` or at the completion of an entry that does not run, at most
one per cycle per unit (CC §9).

### 5.3 SPM arbitration

`tsr_spm` uses round robin, separately for the read and the write port, with a registered grant. The last granted client has the
lowest priority next, so a waiting client is granted after at most `NCL` - 1 other grants. It is built per bank on a 2R2W even and
odd organization (CC §10.1, §10.2, CC-12), which gives the VU its two read ports, since lines L and L + 1 are always in different
banks.

No-deadlock argument (CC §10.3). Every SPM client raises a request only with its payload ready (rule R2) and never waits on memory
while it holds a request. A granted request transfers in that cycle, and round robin bounds the wait. So every SPM request
completes within a bounded number of cycles that does not depend on memory progress. That replaces R4's "local reasons only" with
"bounded, memory-independent reasons", which is what head-of-line routing needs. Under round robin the DMA's beat queue could fill
while other clients write, drop `rd_rsp_ready` and stall weight beats behind a DMA beat. DMA LD burst reservation removes this
(CC-13): an LD issues a read burst only when its beat queue (512 beats, 8 RAMB36) has room for the whole burst, and `rd_rsp_ready`
is constant 1 (CC §10.4).

### 5.4 DRAM channels

An LD or ST that runs beside an MVM takes its beats out of that channel's weight stream, and the slowest channel ends the MVM, so
overlap hides an LD's latency and fixed costs, not its transfer time. A ST between weight bursts on one channel also adds DDR4 bus
turnarounds. If that penalty exceeds the overlap gain, the SDK can make ST depend on the same-channel MVM, with no hardware
change. The DRAM map keeps the op-list, IO and AUX regions in channel 0's low 512 MiB and adds a per-channel compute heap between
the weights and the KV planes (EX §11, SW-26).

### 5.5 Memory model

The race detector (`sw/sim/tsr_race.c`) cites rules 2 and 4 by number. EX §2.3 (H1 to H5) states the host-side rules.

1. Per-op results. Each op's result is defined by the spec's semantics of that instruction and is bit-exact whatever runs beside
   it, if the program is race-free.
2. Order comes only from:
   - (a) queue order: ops on one unit execute and complete in push order, and an op sees every effect of the earlier ops of its
     unit;
   - (b) dependencies: an op with a dep on ticket t starts after t is done and sees its effects;
   - (c) the RISC-V: after it observed t done (a `DONE_SEQ` read, the WAIT interrupt) its later SPM loads see t's effects, and an
     op pushed after a RISC-V store completed sees the store;
   - (d) FENCE and QCLEAR: such a ticket is done only when every op pushed before it is done;
   - (e) completion: every entry completes only after its deps (invariant I1, §4.4), so done(t) implies everything t depends on,
     transitively, is done.

   Two ops on different units with no dependency chain may run in any order or overlap.
3. Visible means the unit's `done`: every SPM write accepted by `tsr_spm` and every DRAM write response received
   ([rtl-core.md](rtl-core.md) §3.0).
4. Race. Two accesses to the same SPM byte, DRAM beat or s register, at least one a write, by two parties (queue ops, the RISC-V,
   the host) not ordered by rules 2 and 5. A racy program is a software bug and its results are undefined. The hardware does not
   detect races. The simulator's race detector does (§9.2).
5. Host. DRAM the host wrote before posting a launch descriptor (the posted writes complete before `LQ_PUSH`) is seen by ops
   pushed after the RISC-V popped that descriptor. DRAM written by ops is seen by the host after it reads the answer record the
   runtime posted after joining those ops (H1 and H2 of EX §2.3).
6. Not used.
7. As-if optimizations inside a unit stay allowed: MVM-to-MVM overlap and VU accept-ahead (invariant I3 of CC §9.4).
8. Scalar registers. The s bank's accesses follow rules 2 and 4 like SPM bytes.

### 5.6 SPM port contention

A long MVM takes a fraction of a percent of the SPM port cycles of an ffn-sized op, and short ops (QACT, LD, ST, VOP) share the
ports roughly as if serialized. The cost an MVM pays for concurrency is the channel beats of concurrent LD and ST and their DDR
effects (§5.4), not SPM ports.

### 5.7 What concurrency buys

Overlap pays where ops are short: the matvec batch of §8.7 overlaps the fixed costs of LD, QACT and ST with the MVM stream,
attention overlaps the LD of KV tile i + 1 with VU work on tile i, and device MoE pushes expert MVMs back to back with no host
round trip per layer. On a dense decode of a large model, non-MVM time is a small share of the token, so the main return is host
independence and latency hiding in short-op phases, not dense-MVM throughput.

## 6. Host and control core

The host does not execute tile ops. It launches kernels, and the RISC-V issues every tile op, so nothing else shares the unit
queues (EX §2 to §4, CC §1).

## 7. Clock domains and CDC

### 7.1 Crossings

The crossing table is CC §11.2. The structures are the issue FIFO (an async FIFO with gray pointers from `clk_rv` to the core
clock, 32 entries of 387 bits in LUTRAM), gray-coded counters with 2-flop synchronizers, tagged handshakes for the error records,
SPM bridge, core window, LQ and LCQ registers, loader and DMI, a CDC FIFO for the log words, and synchronizers for levels and
sticky events. The rules are:

- Gray counters step by at most 1 per source cycle. Queue drains complete at most one entry per cycle per unit, and an assertion
  on every gray counter checks it. A counter that can move by more than one step (the host-written `LCQ_RD` and `LOG_RD`, and
  everything `SOFT_RESET` flushes) never crosses as a gray code. The RISC-V sees it only through a handshake.
- Nothing crosses into a domain in reset. Every core-to-`clk_rv` request path waits for the synchronized "clk_rv out of reset"
  level. Until then, or while `clk_rv` is absent, it answers locally within the CSR timeout: read 0, write dropped, a sticky bit.
  No CSR access waits on `clk_rv` without a timeout (§3.3).

### 7.2 Structures and constraints

The generic, vendor-independent modules are in `rtl/ctrl/cdc/`: `tsr_cdc_fifo` (gray pointers, LUTRAM), `tsr_cdc_gray` and
`tsr_cdc_hs` (handshake, with a variant that carries a 1-bit request tag, used by every request path that can time out). They use
`tsr_sync` and `tsr_rst_sync` from `rtl/tech` (Xilinx: `ASYNC_REG`). `CTRL_SYNC` = 1, for targets where `clk_rv` equals `clk`,
turns the structures into plain register stages. On the U200 the kernel XDC `rtl/ctrl/viv/tsr_ctrl_cdc.xdc`, packaged in the
`.xo`, sets `set_max_delay -datapath_only` of the smaller period, capped at 3.333 ns, on gray buses and handshake data, and
`set_bus_skew` with the same cap. It does not use `set_clock_groups`, which would override the max delays, and it does not rely on
platform clock groups. The gates are `report_cdc` listing only the intended crossings, all safe, and zero unconstrained `ap_clk`
and `ap_clk_2` paths on the linked design (CC §11.3, [platform-u200.md](platform-u200.md) §8.4).

### 7.3 Reset sequencing

CC §12.2 is normative. The decisions recorded here are:

- `ap_rst_n` resets the core domain. The `clk_rv` domain's reset is `ap_rst_n` through `tsr_rst_sync` (asynchronous assert,
  synchronous deassert after 2 to 3 `clk_rv` cycles). `ap_rst_n_2` is not used for CDC endpoints, because an async FIFO or counter
  whose two sides can be reset independently loses pointer agreement. At most it is AND-ed into the Ibex and DM reset as
  `rst_rv_n`. The minimum `ap_rst_n` low time is max(`RST_PIPE` + 1 core cycles, 3 `clk_rv` cycles). Ibex and riscv-dbg use
  asynchronous active-low resets and get theirs from `tsr_rst_sync`, a documented exception to [rtl-core.md](rtl-core.md) §1.
- A RISC-V reset (`RV_RUN` 1 to 0, ndmreset) is a sequence, not a pulse. The core applies QABORT at once. The `clk_rv` sequencer
  stops granting new accesses on both Ibex ports, waits for the outstanding one (the bridge bound at most) and drops a push that
  still waits for a credit. It then asserts the reset of Ibex, the staging registers, deps, `WAIT_*`, the RISC-V hold source and
  the line buffer for at least 3 `clk_rv` cycles, and raises `RV_RESET_DONE`. It does not reset the CDC endpoints, the core-domain
  rings, MTIME, the TCMs and the DMI side of the debug module, so no pointer resynchronization is needed and a debugger stays
  attached.
- Boot order: the core comes out of reset, the host loads the image, `RV_RUN` = 1, the RISC-V reset deasserts 2 to 3 `clk_rv`
  cycles later, and fetch starts at `RV_BOOT` + 0x80. `SOFT_RESET` is not a RISC-V reset. It is the device-level reset of EX §8.5,
  run by the SR sequencer in the core domain: drop `RV_RUN`, wait for QIDLE, flush the rings and PERF, clear error states.

### 7.4 U200 clock, placement and timing

The release image requests `kernel_frequency=0:200|1:100`, so `clk_rv` is half the core clock
([platform-u200.md](platform-u200.md) §8.3). The requirement for `clk_rv` is at least 100 MHz, and the internal BUFGCE_DIV path
remains the fallback design (CC §11.1). The control plane has its own Pblock in SLR0 at the SLR1 boundary, off the clock regions
of the DDR0 controller ([platform-u200.md](platform-u200.md) §8.1, §8.7). Its partners SPM, DMA and QACT are in SLR1, and three
narrow packet links carry the SPM response, the issue entry and the LQ and LCQ requests across the boundary (CC §11.4). The unit
command-register clock enable is a short path, so the queue head's command register is the only command source (CC §9.1: `cmd_q`
and `cmd_v_q`). The timing gates are D13 (CC-23, CC §14.3).

## 8. Software

### 8.1 Device SDK

The SDK is `sw/ctrl` ([sw/ctrl/README.md](../sw/ctrl/README.md); normative: SW §6 and §7). `include/tsr/` holds `kernel.hpp` (op
builders, tickets, issue, joins, `spm_ring`, SPM views), the generated `mmio.hpp`, `vu.hpp` and `at.hpp`, `rt.hpp` (kernel ABI),
`opl.hpp` (op-list format), the executor and planners, and the MMIO and simulator backends (§8.5). `rt/` holds `crt0.S`,
`link.ld`, `runtime.cpp` and `trap.cpp`, and `kernels/` holds `graph.cpp` (the op-list executor), `k_mulmat.cpp` and the other
kind files, `selftest.cpp` and `replay.cpp`. The compiler is xPack `riscv-none-elf-gcc` 14.2.0-3, pinned in
`docs/design/toolchain.lock`, with `-march=rv32imc_zicsr_zifencei -mabi=ilp32`. The language is freestanding C++20 with no heap
and no libstdc++, built with `-Os` except `kernels/graph.cpp` and `kernels/k_*.cpp` with `-O2` (SW-34). The `graph` executor reads
op-list records in place through the line-buffered SPM alias, keeps its ticket and push-index tables (32 KiB) in DTCM and executes
at most 4,096 records per launch (SW-29, SW-35). The image budgets are 98,304 B of ITCM and 57,344 B of DTCM (SW-75).

### 8.2 Header library (API)

The library is `sw/ctrl/include/tsr/kernel.hpp`. The types are `spm` (a line address), `dram` (channel and beat, with the ABI
handle carrying the channel in bits [31:28] and the beat in bits [27:0]), `wref` (a symmetric-allocated weight tensor) and `op`,
one 256-bit tile instruction with a used-word mask. The op builders are generated from the table source: `LD`, `ST`, `QACT`,
`MVM`, the VU and AT ops, `NOP` and `FENCE`. A `ticket` defaults to none, which is always done, and `after(a, b)` builds the
dependency pair and folds same-unit deps. The operations are:

```cpp
ticket   issue(const op&, deps = {}, uint32_t flags = 0);   // async: waits only for a credit; not reentrant
bool     done(ticket);
void     join(ticket);                 // spin; join_irq(ticket) uses WAIT_TICKET + WFI
void     join_all();                   // FENCE + join
uint32_t error(ticket);                // needs done; 0 or its code; undefined for tickets before the last qclear()
uint32_t first_error();                // the error with the smallest push index over all units, 0 if none
ticket   qclear();                     // QCTRL.QCLEAR: a SYNC ticket, done = every earlier entry done, states cleared
```

`issue()` stages the words of `op::used` and the deps that are not none, spins on `ISSUE_CREDIT` for the unit, writes `ISSUE_PUSH`
and returns `ISSUE_TICKET`. It relies on the single hart and on not being reentered (§3.2), so nobody can take the credit between
the check and the push. `error(t)` returns 0 for a ticket that completed before the first error, the code for the failing entry
and `ERR_ABORTED` after it.

### 8.3 Runtime, image format, launch protocol, kernel ABI

The design is SW-10. The image is one static ELF with no relocations (EX §3, SW §6.2): `.vectors` at `RV_BOOT` with `_start` at
+0x80, `.text` in ITCM, and `.rodata`, `.data`, `.bss` and the stack (8 KiB, top of DTCM) in DTCM. The kernel table `.tsr_kernels`
lists the kernels, with `graph`, `selftest` and `replay` as kernels 0, 1 and 2. `.note.tsr` records the Basis ABI, the Tesseract
Core capability, the TCM sizes, the required CAPS bits and the Basis version, and `.tsr_id` holds the 64-bit image id that
`tsr-elf stamp` writes.

The host writes an 8-word launch descriptor into the launch queue LQ (depth 8) by posted CSR writes. The runtime pops it,
validates it, calls the kernel and posts exactly one 8-word record per descriptor, in order, in the completion ring LCQ (depth 8).
The host polls. The runtime steps R1 to R10 are EX §4.7 (`sw/ctrl/rt/runtime.cpp`), and traps and panics are `rt/trap.cpp` and DB
§6.3. A kernel takes up to 7 arguments of 32 bits (SW §6.5): integers, SPM line numbers or DRAM handles with the channel in bits
[31:28]. A kernel that needs more takes an argument block of up to 1 KiB that the runtime loads into the top 16 SPM lines. A
launch is stateless (SW-4): its results depend only on the image, the descriptor arguments and the DRAM contents, and the
simulator reports `SPM_STALE` for reads of SPM lines that nothing wrote since the pop.

### 8.4 Host API (`tsr_transport`)

The host API is `sw/transport/tsr_transport.h` (EX §10.1 plus the conveniences of SW §9.1). One protocol layer serves the `sim`,
`mock`, `vkernel` and `xrt` kinds ([sw/transport/README.md](../sw/transport/README.md), [xrt-host.md](xrt-host.md)).

### 8.5 Host build of the same headers

The same kernel sources compile for the host with `backend_host.hpp` (`TSR_FW_HOST`): `issue` calls the simulator's queue model
through its test hooks, without the RV interpreter. This gives fast unit tests, host GDB and ASan and UBSan on kernel logic. It is
a convenience, not an oracle: the reference is the RV binary on `tsr_rv`. `fw_kernels_host` and its `rv` twins check that the host
build pushes the same op stream as the RISC-V build (SW §6.8).

### 8.6 JIT integration

A JIT is a registry of precompiled kernels launched by name, with a tinygrad or torch.compile backend that maps scheduled kernels
onto it. `tsr_rt`, the PyTorch path, emits op lists (one MUL_MAT record per weight matmul in matmul mode, a region of ops per list in full mode) and launches the `graph` kernel
([sw/rt/README.md](../sw/rt/README.md), SW §9.4). A fixed kernel set covers the LLM hot path (matmul, norms, attention, sampling),
not arbitrary graphs.

### 8.7 Example: matvec batch with prefetch

A kernel has the form `extern "C" int32_t name(const tsr_args * args)` with the declaration `TSR_KERNEL(name, nargs, flags,
blk_lines)` (`rt.hpp`), and `args->blk` points to the argument block in the SPM window. The matvec batch computes y = W x for a
list of jobs with double-buffered X, Q and Y areas. For job i it issues, in this order, QACT i after LD i and MVM i-2, MVM i after
QACT i and ST i-2, the prefetch LD i+1 after QACT i-1, and ST i after MVM i, each through `issue(op, after(a, b))`, then ends with
`join_all()` and returns `first_error()`. It is the push order of the MUL_MAT kind (`sw/ctrl/kernels/k_mulmat.cpp`), and the
`selftest` kernel (case 6, `CONC`) runs it on every unit. In steady state MVM i runs while the DMA does ST i-1 and LD i+1, so the
MVM unit runs back to back, and every op has at most two dependencies, so no SYNC NOP is needed. Without hardware dependencies the
loop would join before each MVM and each ST, and the DMA could not run ahead. The pattern pays where ops are short, and on long
bandwidth-bound MVMs it saves only the fixed costs (§5.4).

## 9. Verification

CC §15 and SW §10 are normative.

### 9.1 Simulator

`sw/sim/tsr_rv.c` is an RV32IMC and Zicsr interpreter that models the chosen Ibex configuration's CSRs and WARL behavior with two
cycle tables (Ibex maxperf and small, `TSR_SIM_RV_CORE`). It is in-house rather than Spike for deterministic co-scheduling with
the tile model, MMIO and trace hooks, speed, and so that the simulator stays one C library. Spike is its oracle (`ctest -L rv`,
skipped without the pinned toolchain). The queue model (`tsr_q.c`) and the scheduler (`tsr_sched.c`) are the control window,
queues and units as an event model on the core time base, with every rule of §4 and §5. Op durations come from the cycle model or
are randomized (`TSR_SIM_SCHED=fixed|random:<seed>[,<min>,<max>]`), and RISC-V time comes from a CPI table scaled by the clock
ratio (`TSR_SIM_RV_RATIO`). An op's effects are applied atomically at its modeled completion, which is invisible for race-free
programs. Device cycles are an informational model and are not bit-matched against hardware counters.

### 9.2 Race detector

Happens-before comes from vector clocks over the parties {the units, the RISC-V, the host}, with the edges of §5.5 (SW §4.6):
queue order, dependencies, RISC-V push, RISC-V observation of completion (a `DONE_SEQ` read returning v joins the clocks of every
completion of that unit below v, and the WAIT interrupt likewise), RISC-V program order, FENCE and QCLEAR, and the launch protocol
(EX §2.3). The joins along dependency edges are sound because of invariant I1. Shadow state is the last writer and the latest read
per party for each SPM line and an interval map per DRAM channel. A report names both tickets (or the RISC-V pc), the ranges and
the ops. The detector also reports reads of never-written SPM (`SPM_STALE`), dependencies on stale tickets and a host QCLEAR that
would be refused. `TSR_SIM_RACE=off|report|fatal` selects the mode, and ctest uses `fatal`. Detection does not depend on the
schedule for valid programs (§9.5), so one run finds every race.

### 9.3 RTL

CC §15 is the normative test list. Ibex itself is a pinned, upstream-verified commit whose own DV is not rerun, and the
integration runs riscv-tests and riscv-arch-test from ITCM under Verilator. The issue path and queues (`tb/ctrl`, `tb/issue`) are
tested for credits, ticket wrap, deps, SYNC NOP, FENCE and QCLEAR, invariants I1 and I2 as assertions, error state, QABORT, hold
sources, RISC-V reset during a bridge access and during a push, the SPM bridge and its bound, every CSR access with `clk_rv`
stopped or held in reset, and a DMI timeout followed by the late answer. A Verilator C++ bench with independent `clk` and `clk_rv`
runs every test at ratios from 0.31 to 1.67 with random phase, and with the `CTRL_SYNC` = 1 build. Random race-free programs run
on RTL and simulator under random memory latency and SPM contention with a bit-exact final state, and `HOLD_ON_HALT` halts at
random points and compares each quiescent state with the simulator (§4.6). The mutation campaign (like [rtl-mvm.md](rtl-mvm.md)
§13.3) drops a dependency compare, completes an aborted entry before its deps, reorders a queue, advances D[u] before the record
acknowledge, steps a gray counter by 2, and returns a credit at `done` instead of accept. The tests must catch each.

### 9.4 Co-simulation and traces

Lockstep. Ibex's RVFI outputs in simulation builds give per retired instruction {pc, insn, rd, value, memory address, data} and
trap and interrupt entries, and `tsr_rv` steps in lockstep. Loads of timing-dependent registers (`DONE_SEQ`, `ISSUE_CREDIT`,
MTIME, `QSTATUS`, `LQ_AVAIL`, the error records) take the RTL value, and trap and interrupt entries are taken at the RTL's retire
position. That log is what the quiescent-state invariant (§4.6) is conditioned on. A tile-only replay pushes a trace's `push.bin`
through the issue path with no RISC-V, which separates core concurrency bugs from RISC-V integration bugs (SW §5).

### 9.5 Bit-exactness for concurrent programs

A program (RV image, inputs and host message script) is valid if it is race-free (§9.2) and its observable behavior does not
depend on the schedule: the op stream per unit, the answer-record stream, the RISC-V's SPM store stream (address, data, order) and
the final SPM and DRAM are the same under every schedule. The check is the determinism test, at least 8 random schedules in the
simulator comparing all four (`sim_sched_determinism`, `fw_kernels_rv` and the other `_rv` tests run 8 schedules with the race
detector fatal). The simulator's schedule space need not cover every RTL timing, so lockstep co-simulation at every clock ratio
(§9.4) covers the rest. A program that passes the 8 schedules but diverges on the RTL shows up as a timing-dependent MMIO value
followed by a different store or push, which is reported against the program, not the RTL.

For a valid program the final state is unique, and the RTL and the hardware reproduce the simulator's exactly: the DRAM regions
written, the SPM lines written where compared, the answer records and the ticket error records, including error paths. Poll
iteration counts, MTIME, `PERF_*` and completion records are not compared. The kernel rule is SW-30: timing-dependent values
(MTIME, credits, poll counts) must not influence data, stores or pushed ops, except in logging that the comparison excludes.

## 10. Debug and profiling

The normative text is DB ([docs/design/05-debug.md](design/05-debug.md)), and the working summary is [prof-dbg.md](prof-dbg.md).
The debugger talks the GDB remote serial protocol to a server built into the device-owning process (or `tsr-dbg`) that drives the
debug module's DMI directly through the CSR window, with no OpenOCD and no software JTAG TAP. Memory is reached through the system
bus access (SBA). Tile-level debug adds stop points, serial mode, tile breakpoints, a 512-entry dispatch history and unit probes,
and the profiler adds a hardware trace writer to a DRAM ring and per-channel rates to the `TSR_STATS` report. The DBG register
block is at core CSR 0x800 to 0xAFC (DB §10.1).

### 10.1 RISC-V debug

The debug module is pulp riscv-dbg (Debug Spec 0.13.2: halt, resume, step, abstract register access, an 8-word program buffer,
ndmreset) in the `clk_rv` domain with `DmBaseAddress` 0x1000_0000 and its two-scratch debug ROM. Ibex's two execute triggers are
the hardware breakpoints, and software breakpoints go in ITCM. The host path is the CSR window `DMI` through a tagged handshake.
SBA is connected and carries all debugger memory access (CC-16, DB-3). ASIC and JTAG targets would reach the DMI through the real
DTM at DMI word address 0x300 (see [prof-dbg.md](prof-dbg.md)).

### 10.2 Profiling

Completion records (128 bits per entry, a 64-entry FIFO behind per-unit holding registers, dropped and counted when full) serve
firmware (DB §8.2). The host timeline comes from a hardware trace writer to a DRAM ring on channel 0 (DB §8.3, DB-16). The PERF
counters are EX's PERF space (EX §6).

## 11. Not provided

- No cache, branch prediction, superscalar or out-of-order execution (D1), no custom RISC-V instructions, no F extension (§2.4),
  no atomics, no second hart and no operating system. The core stays unmodified and issue is MMIO.
- No direct DRAM access from the RISC-V (§3.6) and no instruction fetch from SPM or DRAM.
- No hardware race detection (the simulator does it), no cancellation of a running tile op, and no unit reset. QABORT drops
  entries that have not started, and a wedged unit needs a card reset (CC-22, EX §8.6).
- No second issue staging bank (§3.2), no epoch field on issue-FIFO entries (after an abort every unit stays in error state until
  a QCLEAR that is ordered with the pushes, which gives the same guarantee, §4.5) and no line buffer on the exact SPM window, so
  the exact window keeps its simple ordering (§3.5).
- No scalar-control ops in the tile ISA. The RISC-V does the arithmetic, reads the SPM directly, loops in C++ and polls. The issue
  path refuses opcodes it does not accept with `ILLEGAL_OP` (EX §7.2).


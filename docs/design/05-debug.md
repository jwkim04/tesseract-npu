# DB: debug, GDB over PCIe, tile-level debug, profiler

Area DB specifies the debug and profiling of the device: the use of the RISC-V debug module, the GDB RSP
server, tile-level debug, the log record format, the crash and trap contract, hang triage and recovery, the
profiler, and the DBG register block. It implements D18. Host-visible registers (CSR offsets, `CTRL` and
`STATUS`, PERF, the log ring, the watchdog, `SOFT_RESET`, launch records) are EX's; the RTL hooks inside the
core (queue heads, dispatcher, Ibex bus, debug module instance) are CC's; SDK and tool names are SW's. Numbers
marked estimate are design figures.

---

## 1. Scope

Area DB owns: the debug semantics of the RISC-V (debug module use, halt, step, breakpoint and watchpoint
behaviour, halt-on-boot, halt-on-launch), the GDB RSP server, tile-level debug (stop points, QSTEP, serial
mode, tile breakpoints and watchpoints, dispatch history, unit probes), the log record format, the crash dump
and trap record, hang triage and the recovery ladder, the profiler (completion records, hardware trace writer,
timeline format, TSR_STATS, DB's PERF slots), the debug parts of the simulator and of `tsr-dbg`, and the DBG
register block at core CSR 0x800–0xAFC (EX §5.1).

The mechanisms: the debugger reaches memory through SBA, connected by CC (CC-16, DB-3, §3.6); GDB speaks to an RSP
server that drives DMI directly (DB-1); tile stop points and the divergence search use the stop logic at the
queue heads (§5, §9.4); the dispatch history is a ring of 512 full ops with push PC (§5.6); the firmware log is
EX's core-domain log ring (EX §4.10) with DB's record format (§6.1); PERF gating is EX's ACTIVE and QHALT gates
plus three DB slots (§8.1); the host timeline comes from the hardware trace writer (§8.3, DB-16) while
completion records stay with the firmware (§8.2). ILA debug builds and no JTAG on the U200 ([spec](../spec.md)
§18), Perfetto output and `tsr-prof` ([prof-dbg.md](../prof-dbg.md)) and dumps under `TSR_DUMP_DIR` (§7.4) are
outside the core; every CSR access completes in bounded time.

---

## 2. Overview

```
 host (PCIe; ap_ctrl_none kernel, xrt::ip register access; one owning process)
  │  debug service in the owning process (libtsr_dbg)  ◄── GDB (RSP over TCP / Unix / stdio)
  │        │ DMI ops          │ DBG / STATUS / Q_* / PERF / log            │ XRT bo read
  ▼        ▼                  ▼                                             ▼
 core CSR window ─ 0xC00 DMI ─► (CC: tagged hs) ─► riscv-dbg DM ─► Ibex debug_req, abstract cmds
   │                                                  └─ SBA ─► RISC-V data bus (TCM, SPM, windows)
   ├─ 0x800–0xAFC DBG block (core clock)
   │    stop logic ── dbg_hold[u] / dbg_step[u] ──► queue heads (CC §9)
   │    matcher, history ring ◄── dispatcher enqueue tap (pidx, words, extents, push PC)
   │    unit probes ◄── head info, start / completion strobes, memory-arbiter taps
   │    trace writer ── 32-B records ──► tsr_mem_arb write source (channel TRW_CH = 0) ──► DRAM ring
   │    RISC-V snapshot (crash dump, PC) ◄── hs ── Ibex crash_dump_o
   ├─ 0x400–0x7FC log ring (EX) ◄── LOG_PUSH ◄── firmware records (§6.1)
   └─ PERF (EX) incl. DB slots 61–63
 clk_rv: RV watchpoints on the Ibex data port, halt-request merge, HALT_ON_BOOT, REC FIFO,
         TRACE_MARK, DBG_FLAGS (control-window reservations 0x160–0x17C, 0x400–0x7FC)
```

| Layer | Stops | Observes | Changes |
|---|---|---|---|
| RISC-V debug (§3, §4) | the hart | GPRs, CSRs, TCMs, SPM, windows (SBA) | GPRs, CSRs, TCMs, SPM |
| Tile debug (§5) | queue heads at a push index; one op at a time; all ops serialised | history, unit probes, `Q_*` (EX), s registers | nothing (D15) |
| Log and crash (§6) | | firmware records, trap record, crash dump | |
| Profiler (§8) | | PERF, records, trace ring, PC samples | |

---

## 3. RISC-V debug

### 3.1 Configuration

CC instantiates the parts (CC §3.2); DB fixes what they must do.

| Item | Value | Note |
|---|---|---|
| Debug spec | 0.13.2 (`dmstatus.version` = 2) | riscv-dbg release pinned by CC (CC-2) |
| `dm_top` | `NrHarts` 1, `BusWidth` 32, `DmBaseAddress` 0x1000_0000, two-scratch ROM, `SelectableHarts` 1; `ProgBufSize` 8, `DataCount` 2 | |
| Ibex | `DbgTriggerEn` 1, `DbgHwBreakNum` 2, `DmHaltAddr` 0x1000_0800, `DmExceptionAddr` 0x1000_0808 | triggers match execute addresses only |
| SBA | connected, round robin with Ibex's data port per access, 8, 16 and 32-bit | refusals §3.6 |
| `RV_HALTED` | last granted fetch in 0x1000_0000–0x1000_0FFF (CC-29); host copy `STATUS.RV_HALTED` | DB uses it for holds, PERF, the stop poll |
| `ndmreset` | starts the RISC-V reset sequence; the DM's DMI side follows `ap_rst_n` only | a debugger stays attached across RISC-V resets and `SOFT_RESET` (EX §8.5) |
| Ibex `debug_req_i` | DM request ∨ DB's `dbg_req_rv` (§3.3) | |
| Ibex outputs DB taps | `crash_dump_o` {current_pc, next_pc, last_data_addr, exception_pc, exception_addr}, `core_sleep_o`, `double_fault_seen_o` | |

### 3.2 DMI window

- DM register r (7-bit DMI address) is core CSR 0xC00 + 4r (U200 `s_axi_control` 0x1C00 + 4r). A CSR read is a
  DMI read, a write a DMI write; the crossing, its 4,096-core-cycle bound with `STATUS.DMI_TIMEOUT` and the
  tag-based discard of late answers are CC's (CC §11.2, EX §5.1, §5.3).
- DMI enable (DB-2). Debug-spec reads have side effects (a `sbdata0` read with `sbreadondata` starts a bus
  read; a `data0` read with `abstractauto` starts a command), but EX's rule is that CSR reads have none (EX
  §5.1). The window therefore acts only while `DBG_CFG.DMI_EN` = 1; with 0 a read returns 0 and a write is
  dropped without reaching the DM (a write sets `DBG_STATUS.DMI_BLOCKED`). `DMI_EN` is 0 after a hardware
  reset; only the debug service sets it. Generic dumps read the window as zeros.
- A DMI response code other than 0 sets `DBG_STATUS.DMI_RESP_ERR`; the data is returned as is.

### 3.3 Halt requests from DB

Ibex `debug_req_i` = DM request ∨ `dbg_req_rv`.

- Sources (each only while `DBG_CFG.RV_HALT_EN` = 1): a tile stop that becomes active with
  `DBG_CFG.STOP_HALT_RV` or a matching `DBG_TBPi_CTL.HALT_RV` (§5.2); an RV watchpoint with `HALT` (§3.7);
  `DBG_CFG.HALT_ON_BOOT` at the release of the RISC-V reset (§3.8).
- Mechanism. Core-domain sources send a request toggle (2-flop synchroniser into clk_rv). In clk_rv, `pend` is
  set by a toggle change, an RV-watchpoint hit, or the reset release with HALT_ON_BOOT; it is cleared by the
  clk_rv-local `RV_HALTED` or by `DBG_CTRL.RV_HALT_CANCEL`. `dbg_req_rv` = `pend` ∧ ¬`RV_HALTED`. One request
  gives one halt; resuming never re-halts on a stale request. `DBG_STATUS.RV_HALT_PEND` mirrors `pend`.
- The hart enters debug mode with `dcsr.cause` = 3 (haltreq); the server tells DB's halts from its own by
  `DBG_STATUS` (§4.7).

### 3.4 Halt, resume, step

- Halt latency. Ibex enters debug mode at an instruction boundary after an outstanding data access completes.
  Bounds: a push without credit at most `ISSUE_WAIT_MAX` = 65,536 clk_rv cycles (only if software skipped the
  credit check), SPM bridge or core window at most 4,096, DIV 38 (CC §3.2, §7.3, §10.5). The server waits 5 ms,
  then reports where the hart is stuck (§6.2).
- While halted: no interrupts; MTIME runs; the queues keep running unless `CTRL.HOLD_ON_HALT` (bit2, EX §5.2)
  holds the heads. The server sets HOLD_ON_HALT at attach (`monitor hold off` clears it). Time counters stop
  while QHALT = `RV_HALTED` ∧ `QQUIESCENT` (EX). The server sets `dcsr.stopcount` = 1 if writable, so `mcycle`
  and `minstret` ignore debug-mode instructions.
- Step: `dcsr.step` = 1, resume, wait for the halt, `dcsr.step` = 0. Interrupts stay disabled during a step
  (Ibex `stepie` = 0).
- Resume: if ITCM was written while halted, the server first runs `fence.i` from the program buffer (§3.5),
  then `resumereq`.

### 3.5 Register access and the program buffer

Abstract command Access Register (`cmdtype` 0, `aarsize` 2): GPR x_i regno 0x1000 + i, CSR c regno c,
pc = `dpc` (0x7B1).

| Operation | DMI sequence (numbers are DMI addresses) |
|---|---|
| read x_i | W 0x17 `command` = 0x0022_1000 + i; R 0x16 `abstractcs` until `busy` (bit12) = 0, check `cmderr` [10:8] = 0; R 0x04 `data0` |
| write x_i | W 0x04 = value; W 0x17 = 0x0023_1000 + i; R 0x16 as above |
| read or write CSR c | as GPR with 0x0022_0000 + c and 0x0023_0000 + c |
| run program buffer | W 0x20… `progbuf0…`; W 0x17 = 0x0004_0000 (postexec only); R 0x16 |
| `fence.i` | progbuf0 = 0x0000_100F, progbuf1 = 0x0010_0073 (`ebreak`); run |
| clear `cmderr` | W 0x16 = 0x0000_0700 |
| halt, resume, ndmreset | W 0x10 `dmcontrol` = 0x8000_0001, 0x4000_0001 or 0x0000_0003, poll R 0x11 `dmstatus` (allhalted bit9 or allresumeack bit17), then W 0x10 = 0x0000_0001 |

`abstractauto` is not used. The program buffer is used only for `fence.i`: it never contains a store to the
control window (D15).

### 3.6 Memory access: SBA

All debugger memory access uses SBA (DB-3): it works with the hart running, halted, parked or in reset-halt,
never changes hart state, and reports bad addresses as `sberror`.

| Region | SBA read | SBA write |
|---|---|---|
| ITCM, DTCM | yes | yes while `RV_RUN` = 1; while `RV_RUN` = 0 the hardware refuses SBA TCM writes (CC-33) and the server uses EX's `LOAD_*` (0x0C0–0x0D4) instead |
| DM 0x1000_0000–0x1000_0FFF | refused | refused |
| control window 0x2000_0000–0x2000_0FFF | yes (reads side-effect free) | hardware refuses every write (CC-33); the server refuses them too |
| core window 0x2000_1000–0x2000_1FFF | yes (`LOAD_DATA` and the DMI window read 0, CC §5.3) | refused |
| SPM windows 0x4000_0000, 0x5000_0000 | yes, any time (racy while ops run) | hardware allows; the server allows it only in a stopped tile state (`DBG_STATUS.DRAINED`, or `QQUIESCENT` with the hart halted), else error unless `monitor spm-write unsafe` |

The host sees SPM only through the debug module (EX §2.3).

Sequences (`sbcs` per Debug Spec 0.13: `sbbusyerror` 22 W1C, `sbbusy` 21, `sbreadonaddr` 20, `sbaccess` [19:17],
`sbautoincrement` 16, `sbreadondata` 15, `sberror` [14:12] W1C):

| Operation | DMI sequence |
|---|---|
| read n words at a | W 0x38 `sbcs` = 0x0055_F000 (32-bit, autoincrement, read on address and data, clear errors); W 0x39 `sbaddress0` = a; n − 1 × R 0x3C `sbdata0`; W 0x38 = 0x0045_7000 (no read on data); R 0x3C (last word); R 0x38: `sberror` = `sbbusyerror` = 0 |
| write n words at a | W 0x38 = 0x0045_7000; W 0x39 = a; n × W 0x3C; R 0x38 check |
| 16-bit write | W 0x38 = 0x0042_7000; W 0x39 = a; W 0x3C = value; R 0x38 |

TCM blocks run back to back and SPM writes go one word per `sbbusy` poll. On `sbbusyerror` the server retries
word by word.

### 3.7 Breakpoints and watchpoints

- Software breakpoints: `c.ebreak` (0x9002, kind 2) or `ebreak` (0x0010_0073, kind 4) written to ITCM through
  SBA, original saved by the server; `dcsr.ebreakm` = 1 while attached. Unlimited. Code copied into ITCM at
  run time (overlays) overwrites them; use hardware breakpoints there.
- Hardware breakpoints: Ibex triggers 0–1: `tselect` = i, `tdata1` = 0x2800_1044 (type 2 mcontrol, dmode,
  action = enter debug mode, m, execute), `tdata2` = address; read back.
- RV watchpoints (DB-6): Ibex has no data triggers, so DB adds two comparators on Ibex's data requests (clk_rv;
  SBA accesses are not watched). An access [addr, addr + size) (size from the byte enables) hits watchpoint j
  when `DBG_RVWPj_CTL.EN`, its kind (LOAD or STORE) matches and it overlaps [LO, HI). A hit counts in `HITS`,
  latches `DBG_RV_WPADDR`, sets `DBG_STATUS.RVWP_HIT[j]` and, with `HALT`, raises the halt request. The access
  completes; the hart halts before the next instruction or at most one later (GDB tolerates it).
  Configuration is written in the core domain and transferred by `DBG_CTRL.RVWP_APPLY` through a handshake
  (`RVWP_BUSY`; 4,096-cycle bound → `RVWP_TIMEOUT`).

### 3.8 Reset, halt-on-boot, halt-on-launch

- Halt-on-boot: with `HALT_ON_BOOT` and `RV_HALT_EN`, the request is pending when the RISC-V reset is released,
  so the hart halts before its first instruction (`dpc` = `RV_BOOT` + 0x80). riscv-dbg's `resethaltreq` is not
  relied on.
- `monitor reset halt` and `reset run`: `DBG_CTRL.RELEASE` (so no tile stop holds the reboot's pushes),
  `CTRL.RV_RUN` 1 → 0 (EX: QABORT at once, ANSWERED := TAKEN, EX §8.5), wait `STATUS.RV_RESET_DONE`,
  `CTRL.QCLEAR` when QIDLE (the aborted entries drain because the hard hold yields to a QABORT, §5.2), set or
  clear HALT_ON_BOOT, `RV_RUN` 0 → 1. The TCM image stays; `load` rewrites it.
- Halt-on-launch (DB-7): the runtime calls `tsr_rt_launch_hook(kernel_index, seq)` (noinline) at EX's step R5
  before calling the kernel (EX §4.7); it executes `ebreak` at the label `__tsr_launch_halt` when
  `__tsr_dbg.attached` ∧ `__tsr_dbg.launch_break` ∈ {0xFFFF_FFFF, kernel_index + 1} (§6.3). The hart stops at
  kernel entry with the arguments in a0–a6. Cost when not debugging: two DTCM loads and a branch.

### 3.9 Interaction with the queues and D15

- A RISC-V halt stops only the hart; with HOLD_ON_HALT the heads are held too (CC hold source "debug halt", CC
  §9.6). Running ops finish; then `QQUIESCENT`.
- D15 holds: no CSR maps an issue register; SBA writes to the whole control window are refused in hardware; the
  program buffer never stores to the control window; the server refuses all GDB writes to the control window.
  GDB inferior calls (`call f()`) run in normal mode; if `f` issues, it corrupts the interrupted push's
  staging: unsupported, and the server warns when GDB moves `pc` away from `dpc`.
- While a DB stop or the debug-halt hold is in effect, DB drives the level `dbg_stop` into clk_rv. CC suspends
  the `ISSUE_WAIT_MAX` count while it is 1, so a debug stop never turns a waiting push into a store fault.
  Firmware reads the same level in `DBG_FLAGS` (§10.2) and suspends its own MTIME-based timeouts.

### 3.10 DMI cost over PCIe (estimate)

A DMI access costs one PCIe MMIO read or posted write plus the window's 10 core cycles and the handshake (about
70–110 ns on chip). Halt costs 2 writes and 1–2 reads; reading one GPR or CSR 1 write and 2 reads; GDB `g`
(x0–x31, pc) 33 of those; 64 B through SBA 3 writes and 17 reads (reads) or 18 writes and 1 read (writes); a
poll for a stop (`DBG_STATUS`, no DMI) 1 read; a dump of DTCM 64 KiB 16 K reads. OpenOCD's `remote_bitbang`
needs one socket message per JTAG bit, 50–100 per DMI access: 1–2 orders of magnitude slower, plus a second
process.

---

## 4. GDB RSP server

### 4.1 Processes and ownership

- Library `libtsr_dbg` (`sw/tools/dbg/`): DMI driver (§3.5, §3.6), breakpoint and stop logic, RSP, monitor
  commands, tile commands (§5), log decoder (§6.1), dumps (§7.4). It uses only the transport API (§9.2), so it
  runs on `sim`, `mock`, `vkernel` and `xrt`.
- Embedded (DB-17): one process owns the device, because `xrt::ip` opens the compute unit exclusively (EX
  §2.4). In that process (llama.cpp backend, `tsr_rt`, PyTorch), `TSR_GDB=tcp:<port>|unix:<path>` starts the
  debug service as a transport thread; every device access takes the transport's mutex. `TSR_GDB_WAIT=boot|launch`
  halts at boot or at the first launch and waits for GDB.
- Standalone: `tsr-dbg gdb --device <kind> [--elf fw.elf] [--port N | --unix P | --stdio] [--attach | --load |
  --halt-on-boot]` owns the device (kernel development, selftests). `--attach` opens with `TSR_ATTACH` semantics
  (no reset at open, SW §9.2).
- Launch waits in the transport count timeout time only while neither `STATUS.RV_HALTED` nor
  `DBG_STATUS.STOP_ACTIVE` is set; with `DBG_CFG.SERIAL` the timeout is disabled.
- GDB: `riscv-none-elf-gdb` (SW's xPack toolchain) or `gdb-multiarch`; `target remote :<port>` or
  `target remote | tsr-dbg gdb --stdio --device sim --elf fw.elf`.

### 4.2 Server state machine

```
DETACHED --connect--> ATTACH: DBG_CFG.DMI_EN=1, RV_HALT_EN=1; dmcontrol=dmactive;
                      check dmstatus.version=2, abstractcs.progbufsize>=2, sbcs.sbversion=1 and
                      sbaccess32, CTL_ID bit8 (debug module), DBG_ID != 0; halt;
                      dcsr.ebreakm=1, stopcount=1; CTRL.HOLD_ON_HALT=1; DBG_CFG.STOP_HALT_RV=1;
                      __tsr_dbg.attached=1 if the ELF build id matches   --> HALTED
HALTED   --g G p P m M X Z z qRcmd--> HALTED
         --c / vCont;c--> (fence.i if ITCM was written) resumereq --> RUNNING
         --s / vCont;s--> dcsr.step=1, resumereq --> STEPPING
         --D--> DETACH ;  --k--> KILL ;  --0x03 (Ctrl-C)--> ignored (as gdbserver in all-stop mode)
         --the connection ends or a write fails (EOF, EPIPE, ECONNRESET)--> DETACH
RUNNING  poll DBG_STATUS every 100 µs, backing off to 10 ms; drain the log into O packets
         --0x03 (Ctrl-C)--> haltreq --> HALTED (SIGINT)
         --RV_HALTED--> classify (§4.7) --> HALTED
STEPPING --RV_HALTED--> dcsr.step=0 --> HALTED (SIGTRAP)
         --no halt within 5 ms--> O packet with a crash-dump sample; keep waiting;
           Ctrl-C --> haltreq --> HALTED (SIGINT)
DETACH   remove software breakpoints, triggers, RV watchpoints, tile breakpoints; RELEASE stops;
         dcsr.ebreakm=0; __tsr_dbg.attached=0; RV_HALT_EN=0; HOLD_ON_HALT restored; resume unless
         --detach-halted; DMI_EN=0 --> DETACHED
KILL     standalone: RV reset without HALT_ON_BOOT; embedded: as DETACH (the owner keeps running)
any      writes never raise SIGPIPE (the process that owns an embedded service is the application: a GDB
         that vanishes ends the session, not that process); a failed write is the end of the session
         --STATUS.DMI_TIMEOUT--> DEGRADED: core-domain reads only (status, history, log, probes);
         monitor commands served; DMI retried on every command
```

### 4.3 Packets

| Packet | Support |
|---|---|
| `qSupported` | `PacketSize=4000;qXfer:features:read+;qXfer:memory-map:read+;swbreak+;hwbreak+;vContSupported+;QStartNoAckMode+` |
| `?`, `g`, `G`, `p n`, `P n=v` | yes; register numbers §4.4 |
| `m`, `M`, `X` | yes; routed per §4.5 |
| `c`, `s`, `C`, `S`, `vCont?` (`vCont;c;C;s;S;t`), `vCont` | one thread; signals in `C` and `S` ignored |
| `Z0`–`Z4`, `z0`–`z4` | §4.6 |
| `qXfer:features:read:target.xml`, `qXfer:memory-map:read::` | §4.4, §4.5 |
| `qC`, `qfThreadInfo`, `qsThreadInfo`, `qAttached` | thread 1; `qAttached` = 1 |
| `qRcmd` | monitor commands §4.8, output as `O` packets |
| `qSymbol` | used when no ELF is available (symbols of §6.3) |
| `QStartNoAckMode`, `D`, `k`, 0x03 | yes |
| `vRun`, `R`, `qOffsets`, non-stop mode, `vFile`, tracepoints | no (`monitor reset`) |

While RUNNING the server forwards decoded log records as `O` packets, so firmware output appears in the GDB
console.

### 4.4 Target description

`riscv:rv32`; feature `org.gnu.gdb.riscv.cpu`: x0–x31 (ABI names) regnum 0–31, `pc` regnum 32; feature
`org.gnu.gdb.riscv.csr`, regnum = 65 + CSR number: `mstatus` 833, `misa` 834, `mie` 837, `mtvec` 838,
`mcountinhibit` 865, `mscratch` 897, `mepc` 898, `mcause` 899, `mtval` 900, `mip` 901, `mcycle` 2881,
`minstret` 2883, `mcycleh` 3009, `minstreth` 3011, `mvendorid` 3922, `marchid` 3923, `mimpid` 3924, `mhartid`
3925; feature `org.gnu.gdb.riscv.virtual`: `priv` regnum 4161 (always 3). `pc` is `dpc`. `dcsr`, `dpc`,
`dscratch*`, `tselect` and `tdata*` are not exposed (the server owns them).

### 4.5 Memory map and routing

```xml
<memory-map>
  <memory type="ram" start="0x00000000" length="ITCM_BYTES"/>   <!-- 128 KiB -->
  <memory type="ram" start="0x00100000" length="DTCM_BYTES"/>   <!-- 64 KiB -->
  <memory type="ram" start="0x20000000" length="0x2000"/>       <!-- windows: reads only -->
  <memory type="ram" start="0x40000000" length="SPM_BYTES"/>
  <memory type="ram" start="0x50000000" length="SPM_BYTES"/>
</memory-map>
```

Sizes come from `RV_CFG` (0x030) and `CFG1` at attach. The DM range is not listed. Reads use SBA; writes use SBA
under §3.6 and are refused for the windows. A write to ITCM marks "`fence.i` before resume".

### 4.6 Breakpoint and watchpoint packets

| Packet | Mapping |
|---|---|
| `Z0,a,2` / `Z0,a,4` | software breakpoint (§3.7) |
| `Z1,a,k` | Ibex trigger; `E04` when both are in use |
| `Z2` / `Z3` / `Z4` (write, read, access), a in TCM or window ranges | RV watchpoint 0–1 with `HALT` |
| `Z2`–`Z4`, a in an SPM window | an RV watchpoint for RISC-V accesses and a tile breakpoint of kind SPM on lines ⌊(a − base) / 64⌋ … ⌊(a + len − 1 − base) / 64⌋ with the same access mask, `AFTER` and `HALT_RV`, so tile ops that write the watched SPM stop the target after they complete; `E04` if no TBP is free |

### 4.7 Stop reasons

| Detected | Reply |
|---|---|
| `dcsr.cause` 1, `dpc` at a server breakpoint | `T05thread:1;swbreak:;` |
| `dcsr.cause` 1, `dpc` = `__tsr_launch_halt` | `T05` + `O` "launch <kernel> seq <n>" |
| `dcsr.cause` 1, `dpc` = `__tsr_trap_halt` | crash: `dpc` ← `mepc`; `mstatus.MIE` ← `MPIE`, `MPIE` ← 1 (the effect of `mret`); signal from `mcause`: 0, 4, 6 → SIGBUS (07); 1, 5, 7 → SIGSEGV (0b); 2 → SIGILL (04); 24 (`tsr::panic`) → SIGABRT (06); else SIGTRAP (05). GDB then shows the faulting instruction with the faulting registers |
| `dcsr.cause` 1, other `ebreak` | `T05` |
| `dcsr.cause` 2 | `T05thread:1;hwbreak:;` |
| `dcsr.cause` 4 | `T05` (step) |
| `dcsr.cause` 3 and `DBG_STATUS.RVWP_HIT` | `T05thread:1;watch:<addr>;` (or `rwatch` or `awatch`), addr = `DBG_RV_WPADDR`; the hit is cleared |
| `dcsr.cause` 3 and a tile stop | `T05` + `O` "tile stop at pidx P (<cause>): <disassembly>, pushed at <file:line>" (§5) |
| `dcsr.cause` 3 after HALT_ON_BOOT | `T05` + `O` "halted at boot" |
| `dcsr.cause` 3 after Ctrl-C | `T02` |

Every `T` reply carries `20:` (pc), `02:` (sp), `01:` (ra), `08:` (s0) as expedited registers.

### 4.8 Monitor commands (`qRcmd`; the same set in the `tsr-dbg` CLI, §9.3)

`reset halt|run`; `load` (image rewrite after an RV reset, EX's `LOAD_*` with VERIFY); `launch-break
<kernel>|all|off`; `hold on|off` (HOLD_ON_HALT); `status`; `log [level]`; `crash` (prints `__tsr_crash`);
`crash-view on|off` (`g` returns the saved trap registers so `bt` works for a fault taken without a debugger);
`tile …` (§5); `hist [n]`; `units`; `perf`; `spm x/<n><fmt> <line>` (f32, hex, i8, QA block, Q8_0 and Q4_K
block); `sreg`; `dump <dir>`; `recover [level]`; `spm-write safe|unsafe`.

### 4.9 Error replies

`E01` DMI timeout (clk_rv absent, `STATUS.RV_CLK_OK` = 0); `E1x` abstract command failed, x = `cmderr`; `E2x`
SBA failed, x = `sberror` (8 = `sbbusyerror`); `E04` refused by policy (D15 window, unsafe SPM write, no free
trigger or TBP); `E05` target running; `E06` malformed packet; `E07` transport error. The reason is also
printed on stderr.

---

## 5. Tile-level debug

### 5.1 Principle: stop points in push order

Tile ops have no instruction boundary: up to six units run concurrently, each in queue order (CC §9). The
comparable observation point is the push-order prefix: every entry with push index < T has completed and no
entry with push index ≥ T has started or completed. For a valid program (race-free and schedule-independent,
[spec](../spec.md) §10) the tile-written SPM and DRAM in that state equal the sequential simulator after
entries 0 … T−1 (§5.8). DB reaches it by holding every queue head whose push index is ≥ T (DB-8). Because the
dispatcher enqueues in push order and an entry cannot start earlier than 3 cycles after it is enqueued (enqueue
t, head load t+1, evaluation t+2, `cmd_valid` t+3, CC §7.6, §9.3), a threshold set when an entry is enqueued
(registered at t+1) always holds it in time. No dispatcher gate is needed: pushes continue into the queues
until credits run out; then the RISC-V spins in `issue()`, or halts with `STOP_HALT_RV`.

Push indices compare modularly: ge(a, b) ⇔ (a − b) mod 2^32 < 2^31.

### 5.2 Stop logic

State (core clock): `S` (manual threshold, valid bit `S_VALID`); the match queue `M` (pending match thresholds
in increasing order, 64 × 32-bit LUTRAM; at most the sum of the `Q_DEPTH` (44) can be pending, so it never
overflows); per-unit shadow queues (§8.3). Derived:

- T = the earlier of `S` (if valid) and the head of `M` (if any). `STOP_ACTIVE` ⇔ T exists.
- `dbg_hold[u]` = `head_valid[u]` ∧ ((`STOP_ACTIVE` ∧ ge(`head_pidx[u]`, T)) ∨ (`SERIAL` ∧ `head_pidx[u]` ≠
  `min_out`)), where `min_out` is the smallest push index among dispatched, not completed entries (the oldest
  outstanding entry of each unit, compared by age against `DBG_NEXT_PIDX`). CC implements it as a hard hold (CC
  §9.2–§9.3): `dbg_hold[u]` is evaluated before the non-running completions in `H_EVAL` (poisoned, ABORTED,
  DEP_FAILED, SYNC), so a held entry neither starts nor completes (DB-9); the other hold sources keep CC's
  order. Two exceptions keep every reset finite: after any QABORT (host, RISC-V, `LQ_CANCEL`, RISC-V reset,
  `SOFT_RESET`) and until the next QCLEAR, `dbg_hold` blocks starts only, so aborted entries drain; and while
  `STATUS.SR_BUSY` it is forced to 0. `SOFT_RESET` step 1 also clears `S`, `S_VALID` and the match queue (EX
  §8.5); TBPs, `SERIAL` and the evidence are kept.
- DRAINED = `STOP_ACTIVE` ∧ ge(`DBG_NEXT_PIDX`, T) ∧ ∀u: unit u has no outstanding entry or
  ge(`oldest_pidx[u]`, T). In that state the prefix property holds.

Events (one of STOP_NOW, RELEASE, QSTEP per `DBG_CTRL` write; priority QSTEP > RELEASE > STOP_NOW):

| Event | Action |
|---|---|
| `STOP_NOW` | `S` ← `DBG_NEXT_PIDX` (an earlier valid `S` is kept); cause NOW |
| write `DBG_STOP_AT` = P | `S` ← P, valid; cause NOW |
| tile breakpoint match at the enqueue of entry P (§5.3) | `BEFORE`: push P into `M`; `AFTER`: push P + 1; cause BP; `BP_HIT` records the TBPs |
| a unit enters error state, with `ERR_STOP` | `S` ← `DBG_NEXT_PIDX`; cause ERR |
| `STATUS.UNIT_HUNG` rises (EX's watchdog), with `HUNG_STOP` | `S` ← `DBG_NEXT_PIDX`; history frozen; cause HUNG |
| `RELEASE` | `S` invalid; pop every `M` entry m with ge(T, m); the next `M` entry (if any) becomes T |
| `QSTEP` with `STOP_ACTIVE` | wait for DRAINED; X ← T; `S` ← X + 1 (valid); pop `M` entries ≤ X; if the debug-halt hold (HOLD_ON_HALT ∧ `RV_HALTED`) applies, pulse `dbg_step[u]` once when `head_pidx[u]` = X, only while the host and RISC-V hold sources are clear; `STEP_BUSY` until the completion strobe of X; repeat `DBG_STEP_N` times; cause STEP |
| `QSTEP` without `STOP_ACTIVE`, hart halted with HOLD_ON_HALT | X ← the smallest `head_pidx` among valid heads; pulse `dbg_step[u]` for it; `STEP_BUSY` until X completes |
| `QSTEP` otherwise | acts as `STOP_NOW` |
| a stop becomes active with `STOP_HALT_RV` or a TBP's `HALT_RV` | RISC-V halt request (§3.3) |

`DBG_STEP_PIDX` shows X. A stop activation is recorded as a STOP record in the trace writer (§8.3) and pulses
the internal net `dbg_stop_evt`, the ILA trigger of debug builds.

### 5.3 Tile breakpoints and watchpoints

Four breakpoints `TBP0`–`TBP3` are evaluated on the dispatcher's enqueue tap (§13): push index, ticket, w0,
checker result, and the operand extents the checker computes for its range checks (up to four extents per op,
each tagged SPM or DRAM, CC §8.2), each {start, end exclusive, read or write}. An op with more extents is
matched on its first four in the checker's field order, in the simulator and the RTL alike: a merging AMERGE
lists Orun (written, read), parts, mk, st (written, read), so its `st` vector cannot be watched. Strided
operands are their bounding interval (false hits in the gaps are possible). The match is computed beside
checker stages C6–C7 and registered with the queue write: no extra dispatch cycle.

| `KIND` | Matches when | A / B |
|---|---|---|
| 0 PIDX | (pidx − A) mod 2^32 < (B − A) mod 2^32 | B = A + 1 for one entry |
| 1 OPCODE | ((w0[7:0] ⊕ A[7:0]) ∧ A[15:8]) = 0 ∧ ((w0[11:8] ⊕ A[19:16]) ∧ A[23:20]) = 0 ∧ (B[7:0] = 0 ∨ B[unit]) | opcode and sub value and mask; unit mask |
| 2 SPM | an SPM extent with a mode in {`R`, `W`} overlaps lines [A, B) | |
| 3 DRAM | the DRAM extent with a mode in {`R`, `W`} overlaps beats [A, B) on a channel in `CHMASK` (MVM weights: the extent on every channel) | |
| 4 TICKET | ticket = A | |

Range kinds never match a poisoned entry. `COUNT_ONLY` counts in `HITS` without stopping. A tile watchpoint is
a range breakpoint with `W` (or `R`): `BEFORE` answers "which op is about to overwrite this line", `AFTER` stops
with the write done. RISC-V accesses to the same lines are caught by an RV watchpoint (§4.6).

### 5.4 Serial mode

`DBG_CFG.SERIAL` = 1 holds every head except the one holding the oldest outstanding entry, and that one only
while nothing is running (the running entry is the oldest). Exactly one entry runs at a time and entries start
and complete in push order; CC's MVM overlap (I3) does not occur. For a valid program the final state equals
normal mode; a result that differs between serial and normal mode is a race in the program or a concurrency bug
in the core (the simulator's race detector decides which). No deadlock: every dependency points to an earlier
push index. The cost is the overlap gain. Unlike the OPL `SERIAL` flag (a join after every op, SW §7.2) it
changes neither the op stream nor the firmware. Use: `TSR_TILE_SERIAL=1`, divergence triage, the debug selftest.

### 5.5 Unit probes

EX's `Q_STATE[u]` (entries, RUNNING, DEP_WAIT, HELD, ERR, HUNG), `Q_DONE`, `Q_ISSUED`, `Q_ERR_*` and `Q_PIDX` (EX
§5.4) give the basic per-unit view. DB adds, per unit u (CC ids 0 DMA, 1 QACT, 2 MVM, 3 VU, 4 AT, 5 SYNC; 6–7
read 0):

- `DBG_Uu_STATE`: [1:0] ops accepted and not done (0, 1, 2 for the MVM overlap), [6:4] CC head state (0 EMPTY,
  1 LOAD, 2 WAIT_DEP, 3 WAIT_SNAP, 4 HELD, 5 WAIT_UNIT, 6 ISSUE), [11:8] hold sources at the head (host, RISC-V,
  debug halt, DB), [19:12] outstanding entries.
- `DBG_Uu_OLDEST` and `DBG_Uu_OPIDX`: ticket and push index of the oldest outstanding entry (running op, else
  head); ticket 0xF000_0000 when none.
- `DBG_Uu_AGE`: core cycles since that entry became the oldest (saturating 32 bits).
- Memory probes `DBG_RDOUT0`–`3` and `DBG_WROUT0`–`3`: outstanding read and write bursts per channel (8 bits
  per channel; channel c in byte c mod 4 of word ⌊c/4⌋), taps from `tsr_mem_arb`.
- Watchdog: EX's progress watchdog (`WDOG_CFG`, `Q_STATE.HUNG`, `STATUS.UNIT_HUNG`, EX §8.6). DB adds only
  actions: `HUNG_STOP` (stop at `DBG_NEXT_PIDX`) and the history freeze.

### 5.6 Dispatch history ring

- Recorded at the dispatcher's enqueue tap, in push order, for every ticketed entry (ops, QCLEAR; poisoned
  entries included): 504 bits {push index, ticket, DEP0, DEP1, w0–w7, push PC, dispatch time MTIME[31:0],
  INFO}. `HIST_DEPTH` = 512 on the U200 (7 RAMB36 at 72 bits).
- Slot = push index mod depth. `DBG_HIST_HEAD` = next push index to be recorded. Reading: write `DBG_HIST_IDX` =
  push index, read the fields; `INFO.PRESENT` = 1 iff that slot holds that push index and it was recorded after
  the last `HIST_CLEAR`. A slot that is not `PRESENT` reads 0 in every field (`INFO` included), so a stale or
  power-up slot content is never taken for an entry.
- `HIST_FREEZE_ERR` (on after reset): recording stops `HIST_DEPTH` / 2 = 256 entries after the first entry that
  completes with an error, so the ring keeps 256 entries on each side of it. `UNIT_HUNG` freezes at once;
  `HIST_FREEZE` by hand; `HIST_CLEAR` empties and unfreezes.
- Push PC: CC latches Ibex's `crash_dump_o.current_pc` (the PC in ID/EX, i.e. of the `ISSUE_PUSH` store) at the
  push and carries it in the issue-FIFO entry (+32 bits). `tsr-dbg` maps it to `file:line` and the inline stack
  with the ELF's DWARF: every tile op points back to the C++ line that issued it.
- Status of an entry: done iff (`Q_DONE[u]` − seq) mod 2^28 ∈ [1, 2^27); running iff it is `DBG_Uu_OLDEST` with
  ops accepted; else queued. Codes follow from `Q_ERR_*` and the rules of EX §8.2.
- `tsr-dbg hist --export` writes the ring as SW's `ops/push.bin` records (SW §5.2), so a captured window
  replays on the simulator.

### 5.7 SPM, register and TCM dumps

| What | Path |
|---|---|
| SPM lines | SBA reads of the SPM window (§3.6): one line is 17 DMI reads |
| SPM bulk, firmware alive and idle | a `replay` launch (EX §4.6, SW §7.7) of ST ops into a DRAM scratch area, then a bo read |
| s0–s15 | `SREG0`–`15` (0x9C0, §10.1), a read-only view of the VU's bank: 16 reads |
| GPRs and CSRs | abstract commands (halted) or `__tsr_crash` (after a trap) |
| TCMs | SBA (`RV_RUN` = 1) or `LOAD_DATA` reads (`RV_RUN` = 0) |

### 5.8 Comparability with the ISA simulator

- Invariant G (stop). If DRAINED holds with threshold T, for a valid program the SPM lines and DRAM beats
  written by tile ops, the per-unit completion counts and the error records equal the simulator running the
  same program in its serial schedule and stopped at T. With the RISC-V halted after n retired instructions, its
  registers, TCMs and SPM stores equal the simulator advanced to n retired instructions, given the
  co-simulation log of timing-dependent MMIO values (CC §15.4).
- Invariant Q (step). From a DRAINED state at T, k QSTEPs give the DRAINED state at T + k.
- Invariant S (serial). In serial mode completions occur in push order (checkable in the trace records) and the
  final state equals normal mode.
- Invariant H (history). For a valid program the history entries (words, deps, tickets, push indices, push PCs)
  equal the simulator's push stream; dispatch times are not compared.
- HOLD_ON_HALT alone (no tile stop) gives the completion-vector form of the quiescent state, not a push-order
  prefix.

---

## 6. Log, crash dump, trap record

### 6.1 Log records (format on EX's ring)

Hardware and host protocol are EX's (EX §4.10): the RISC-V writes `LOG_PUSH` (control window 0x280) after
checking `LOG_FREE` (0x284) ≥ 1 + N, so records are never torn; the host reads words [`LOG_RD`, `LOG_WR`) from
0x400 + 4 × (k mod 256) and writes `LOG_RD`; `LOG_CFG` (0x098: ENABLE, BLOCK, LEVEL); `LOG_DROPPED`,
`STATUS.LOG_DROP`. DB keeps the 256-word depth and the drop-whole-record policy (DB-12) and defines the record,
merging EX's framing fields and SW's binary format (SW §6.6):

| Header bits | Field |
|---|---|
| [7:0] | N: payload words after the header (EX's framing) |
| [11:8] | LEVEL: 0 FATAL, 1 ERROR, 2 WARN, 3 INFO, 4 DEBUG, 5 TRACE |
| [15:12] | SOURCE: 0 runtime, 1 kernel, 2 trap handler, 3 debug kernels |
| [18:16] | TYPE: 0 FMT, 1 TEXT, 2 TRAP, 3 EVENT, 4 BLOB; 5–7 invalid |
| [19] | TS: the first payload word is MTIME[31:0] (SW's TS) |
| [27:20] | LSEQ: SEQ[7:0] of the running launch (as LCQ r0, EX §4.4); 0 outside launches |
| [31:28] | SYNC = 0xA (SW's sync nibble) |

| TYPE | Payload after the optional MTIME word |
|---|---|
| FMT | address of the format string in the non-loaded ELF section `.tsr_logfmt` (SW: VMA 0xF000_0000), then 32-bit arguments: `%d %u %x %f` (f32 bits) `%t` (ticket), 64-bit integers as two words (lo, hi), `%s` only for strings in loaded `.rodata` (read from the ELF) |
| TEXT | UTF-8, little-endian in words, zero-padded; one line per record |
| TRAP | the 44 words of `__tsr_crash` (§6.3) |
| EVENT | event code in the marker encoding of §8.4, then arguments |
| BLOB | tag, then raw words |

- A record that does not fit is dropped whole and counted in DTCM; the next record that fits is preceded by an
  EVENT record "dropped n" (SW's rule). With `LOG_CFG.BLOCK` the SDK waits instead (lossless printf debugging
  while a host drains). Interrupt handlers do not log, so records never interleave.
- Decoding checks SYNC, TYPE, LEVEL ≤ 5, N ≤ 255 and, for FMT, that the address lies in `.tsr_logfmt`; a failure
  is reported as a firmware log bug and the reader skips to `LOG_WR`.
- The transport drains the ring every poll when `TSR_LOG` is set (else every 10 ms, SW §9.1) and the debug
  service drains it while RUNNING. Bulk data never goes through the log (trace writer §8.3).

### 6.2 Ibex crash-dump snapshot

`DBG_CTRL.RV_SAMPLE`: the core sends a request toggle; clk_rv latches `crash_dump_o`, `core_sleep_o`, the reset
state and `RV_HALTED` and answers through a tagged handshake; the core copies them into `DBG_RV_PC`, `_NPC`,
`_DADDR`, `_EPC`, `_EADDR` and `DBG_RV_STATE`, incrementing `SAMPLE_SEQ`. The `DBG_CTRL` write completes only
after the copy or after 4,096 core cycles (then `SAMPLE_VALID` = 0 and `RV_SAMPLE_TIMEOUT`), so one following
read sees the sample: 1 write and 1–2 reads. It works while the hart runs (non-intrusive PC sampling, §8.5), is
stuck in a bus access (`DADDR` names it), or is parked. `double_fault_seen_o` sets the sticky
`DBG_STATUS.RV_DFAULT`.

### 6.3 Runtime debug contract (implemented by SW's SDK)

EX's `RT_STATE` and `RT_INFO` registers (EX §4.5, host 0x038 and 0x03C) carry the runtime phase; DB adds two
DTCM structures and two labels.

`__tsr_dbg` (section `.tsr_dbg`, 64 B, words): 0 magic 0x4742_4454 ("TDBG"), 1 version 1, 2 build id (ELF note;
the server compares it with its ELF), 3 `attached` (debugger writes), 4 `launch_break` (0 off, 0xFFFF_FFFF all,
else kernel index + 1), 5 `trap_count`, 6–15 reserved.

`__tsr_crash` (section `.tsr_crash`, 192 B, words): 0 magic 0x5041_5254 ("TRAP"), 1 version 1, 2 `mcause`, 3
`mepc`, 4 `mtval`, 5 `mstatus`, 6–7 MTIME lo and hi, 8 launch SEQ, 9 kernel index, 10 `RT_STATE`, 11 `RT_INFO`,
12–42 x1–x31, 43 `nested`, 44–47 reserved.

Trap handler (synchronous exceptions; `tsr::panic` uses the custom cause 24):

1. Swap `sp` with `mscratch` (a 512 B trap stack in DTCM); increment `nested`; if `nested` > 1 go to step 5.
2. If `__tsr_dbg.attached`: restore every register and execute `ebreak` at `__tsr_trap_halt`; the server
   presents the faulting context (§4.7).
3. Else fill `__tsr_crash`, push a TRAP log record (LEVEL FATAL, SOURCE 2), post EX's TRAP record (type 0x81:
   `mcause`, `mepc`, `mtval`, `RT_STATE`, `RT_INFO`, MTIME, EX §4.4).
4. `RT_STATE` := TRAPPED; WFI loop with interrupts disabled. The host attaches GDB (halt works in WFI) or reads
   `__tsr_crash` through SBA without halting; `monitor crash-view on` shows that frame.
5. Nested trap: `ebreak` if attached, else WFI loop.

The transport turns a TRAP record into a launch failure with a decoded report (cause name, `mepc` → function and
line).

---

## 7. Hang triage and recovery

### 7.1 Always answerable

Every DBG register is a core-domain register mux; reads never wait on a unit, on clk_rv or on memory (the AXI
firewall rule). The only DBG accesses that wait on clk_rv (`RV_SAMPLE`, `RVWP_APPLY`) end within 4,096 core
cycles.

### 7.2 Triage table (`tsr-dbg status` applies it)

| Observation | Reads | Reading | Next |
|---|---|---|---|
| launch wait timed out, `STATUS.RV_HALTED` | `DBG_RV_STATE`, `DBG_STATUS` | stopped by a debugger, breakpoint or trap halt | attach GDB |
| `RT_STATE` = TRAPPED or a TRAP record | `__tsr_crash` (SBA), log | firmware crash | `monitor crash`, `crash-view` |
| PC samples move, `DADDR` = `DONE_SEQ[u]` (0x2000_0040 + 4u) | 3–5 `RV_SAMPLE`s | spinning on a join of unit u | unit u probe |
| PC samples move, `DADDR` = `ISSUE_CREDIT` (0x2000_0030) | | spinning on credits | the full queue's head state |
| PC fixed, `DADDR` = `ISSUE_PUSH` (0x2000_0028) | | push waiting without credit (at most `ISSUE_WAIT_MAX`, then PUSH_TIMEOUT and a store fault) | software skipped the credit check |
| `STATUS.RV_SLEEP` for long | `RT_STATE`, `WAIT_TICKET` (SBA) | WFI waiting for a ticket or a descriptor | the awaited unit or the host |
| `Q_STATE[u]` RUNNING, `DBG_Uu_AGE` grows, `DBG_RDOUT` > 0 on c, `CH_RD_BEATS[c]` frozen | probes, PERF twice 100 ms apart | read stalled on channel c | ILA debug build; card reset |
| `STATUS.UNIT_HUNG` | `Q_STATE`, probes | unit stuck without progress | card reset (EX §8.6) |
| head WAIT_DEP | the head's deps via the history | waiting for a dependency; follow the chain | the dep's unit |
| head HELD | `DBG_Uu_STATE` [11:8], `DBG_STATUS` | host hold, RISC-V `QHOLD`, HOLD_ON_HALT or a DB stop | release that source |
| DMI timeouts, `RV_CLK_OK` = 0, `RV_FREQ` = 0 | | clk_rv absent | platform or clock |
| reads return 0xFFFF_FFFF | | AXI firewall tripped or card lost | card reset |

### 7.3 Recovery ladder (`tsr-dbg recover [--level n]`; the transport's timeout path follows it)

| Level | Action | Destroys | Keeps |
|---|---|---|---|
| 0 | hang dump (§7.4), always first | | everything |
| 1 | debug release: `RELEASE`, `SERIAL` off, HOLD_ON_HALT off, TBPs cleared, hart resumed | debug stops | all state |
| 2 | `LQ_CANCEL` covering the outstanding launches (EX §4.9: QABORT of the running one); wait for DONE ABORTED at most `TSR_ABORT_MS` | not-started ops | running ops finish |
| 3 | `CTRL.SOFT_RESET` + warm restart (EX §3.5, §8.5) | RISC-V state, rings' unread content, PERF, error records | TCM image (verified), SPM, DRAM, DB evidence |
| 4 | card reset (`xrt-smi reset`; `SOFT_RESET` cannot finish with a hung unit) | everything | the level-0 dump |

DB's evidence (history ring, trace-ring pointers and contents, RV snapshot, DBG configuration) is cleared only
by DB's own clear bits or a hardware reset, never by QABORT, QCLEAR, the RISC-V reset or `SOFT_RESET` (EX §8.5).

### 7.4 Hang dump (`TSR_DUMP_DIR`, `tsr-dbg dump <dir>`)

`state.json` (every EX, DBG and PERF register raw and decoded; adapter registers), `hist.txt` (disassembled ring
with status and `file:line`), `log.txt`, five RV samples 1 ms apart, the trace ring tail, `units.txt`,
`__tsr_dbg` and `__tsr_crash` (SBA, if clk_rv answers), DTCM (SBA). SPM only on request. A dump of another
process's device is impossible while that process lives (exclusive `xrt::ip`); the embedded service writes the
dump itself on timeouts and fatal signals. A new process that opens a compute unit left by a dead one uses
`TSR_ATTACH` (no reset).

---

## 8. Profiler

### 8.1 PERF

EX defines the counters and their gates (EX §6): time counters count with ACTIVE (TAKEN ≠ ANSWERED: a launch
is in progress) ∧ ¬QHALT (`RV_HALTED` ∧ `QQUIESCENT`), event counters with ACTIVE; PERF does not move between
launches. DB adopts this (DB-15) and owns three slots at the top of the range, recorded in EX §6's table next
to VU's (X + 13 … X + 15), AT's (X + 16 … X + 19) and CC's (X + 20, X + 21); at `NUM_CH` = 16 the extensions
end at 58, so they never collide:

| Slot | Name | Kind | Counts |
|---|---|---|---|
| 61 | `TRW_RECORDS` | event | trace records written to DRAM |
| 62 | `TRW_DROPS` | event | trace records dropped |
| 63 | `DBG_STOP_CYCLES` | time, not gated by QHALT | core cycles with ACTIVE ∧ (`RV_HALTED` ∨ `STOP_ACTIVE`): a measurement is valid only if it is 0 |

Derived metrics need no new counters: channel c idle (no read outstanding, no beat) = `CYCLES` −
`CH_RD_BEATS[c]` − `CH_WAIT[c]`, the DDR idle time that the performance model targets ([spec](../spec.md) §14);
issue-bound MVM time = `MVM_EMPTY`; dependency-bound = `MVM_WAIT`.

### 8.2 Completion records to the RISC-V (REC)

For firmware-level profiling and self-tuning. Control window (CC's reservation 0x160–0x17C, CC §5.1): 0x160
`REC_COUNT` (R), 0x164–0x170 `REC_DATA0`–`3` (R, head record), 0x174 `REC_POP` (W), 0x178 `TRACE_MARK` (W,
§8.4), 0x17C `REC_CFG` (RW: bit0 ALL = a record for every entry, not only `ISSUE_PUSH.REC` ones; bit1
`irq_fast[4]` while `REC_COUNT` ≥ 32). Record: `REC_DATA0` ticket (unit 14 for markers); `REC_DATA1`
t_end[31:0] (MTIME; markers: dispatch time); `REC_DATA2` RUN cycles (start → end, saturating; markers: the
value); `REC_DATA3` [3:0] code, bit4 ran, bit5 marker, [15:8] opcode, [31:16] push index [15:0]. FIFO 64 × 128 b
core → clk_rv; records never stall execution: when full they are dropped and counted (EX's `REC_DROPPED`). EX's
descriptor bit PROF (EX §4.3) makes the runtime set `REC_CFG.ALL` and emit NODE markers; the host timeline
itself comes from the trace writer.

### 8.3 Trace writer (TRW): the host timeline source

Bulk records must not cross PCIe as MMIO reads or occupy the RISC-V, which is issue-bound in exactly the phases
worth profiling, so a small engine writes them to a host-allocated DRAM ring (DB-16):

- Inputs: the enqueue tap (dispatch time), CC's head info (time an entry becomes head), start and completion
  strobes {seq, pidx, code, ran} with MTIME, TRACE_MARK entries, stop events, `RV_HALTED` edges. Per unit a
  shadow queue {pidx, t_dispatch, t_head, t_start} mirrors CC's queue (depth `Q_DEPTH[u]` + 2; 56 × 128 b
  LUTRAM on the U200); it also gives `oldest_pidx[u]` for §5.2.
- Record, 32 bytes, two per 64-byte beat:

| Word | OP record (KIND 0) |
|---|---|
| 0 | [3:0] KIND (0 OP, 1 MARK, 2 STOP, 3 RVHALT, 4 DROP, 15 PAD), [7:4] unit, [11:8] code, bit12 ran, bit13 REC flag, bit14 TBP hit, [23:16] opcode, [31:24] w0[15:8] |
| 1 | push index |
| 2 | ticket |
| 3 / 4 | t_end [31:0] / [63:32] (MTIME, core cycles) |
| 5 | RUN = t_end − t_start (0 if not run), saturating |
| 6 | QWAIT = t_head − t_dispatch (behind earlier entries of the unit) |
| 7 | HWAIT = t_start − t_head (deps, holds, unit busy); for non-run entries t_end − t_head |

  MARK: word 1 = next push index, 2 = marker value, 3–4 = dispatch time. STOP: 1 = T, 2 = cause bits, 3–4 =
  time, 5 = 1 activated and 0 released. RVHALT: 2 = 1 halted and 0 resumed. DROP: 1 = records dropped since the
  previous DROP record. PAD: ignored.
- Ring: `DBG_TRW_BASE` {ch[31:28], beat[27:0], 16-beat aligned}, `DBG_TRW_LINES` (power of two ≥ 16),
  `DBG_TRW_WPTR` (records whose write response returned), `DBG_TRW_RPTR` (host), `DBG_TRW_DROPS`,
  `DBG_TRW_MASK`. `TRW_STOP_FULL` = 0 overwrites (flight recorder); 1 drops when WPTR − RPTR = capacity. The
  channel field must name the writer's channel `TRW_CH` (0): a ring on another channel is no ring, and its
  records are dropped (counted in `DROPS`) in the simulator and the RTL alike.
- Write path: 64-record LUTRAM staging (two 1 KiB bursts: one in flight while the next fills); a 16-beat burst
  when 32 records are ready; a partial burst on `TRW_FLUSH`, after 65,536 idle cycles, or after a LAUNCH_END
  marker. It is a second write source of `tsr_mem_arb` on channel `TRW_CH` = 0 (ch0 holds the IO region),
  arbitrated per burst with the DMA (CC §13.2). It never stalls a unit: full staging → drop + DROP record.
- Cost: a decode token produces thousands of entries, a few hundred KB of records, far below 0.1 % of one
  channel's write bandwidth. Host: one bo read of the new records per launch. The ring defaults to 64 MiB
  (`TSR_PROF_TRW_MIB`), allocated by the transport from the ch0 IO region (64–320 MiB, EX §11) only while
  profiling.

### 8.4 Markers and the launch timeline

`TRACE_MARK` (0x178) sends a CTRL entry (CC FIFO kind 1, code 5, w1 = value) through the issue FIFO; it is
stamped at dispatch, in push order, and goes to REC (unit 14) and TRW (MARK). Marker value: [31:28] class,
[27:0] payload:

| Class | Meaning | Payload |
|---|---|---|
| 0 | user | any |
| 1 / 2 | LAUNCH_BEGIN / LAUNCH_END | [27:16] kernel index, [15:0] launch SEQ low bits |
| 3 / 4 | NODE_BEGIN / NODE_END (OPL entry; the OPL `MARK` flag) | op-list index |
| 5 | LAYER | layer index |
| 6 | PHASE | kernel-defined id (named in the ELF's `.tsr_marks` table) |
| 7–15 | reserved | |

The runtime emits LAUNCH_BEGIN and LAUNCH_END around every kernel. Host events (launch push, record seen,
return) are recorded by the transport with `CLOCK_MONOTONIC`; device time is aligned by reading `DBG_MTIME` (HI,
LO, HI) at open and every second and fitting offset and drift.

### 8.5 RISC-V profiling

PC sampling through `RV_SAMPLE` (§6.2): `tsr-dbg rv sample` takes samples symbolised with the ELF. `mcycle` and
`minstret` are available through GDB; Ibex `mhpmcounter`s stay off (`MHPMCounterNum` 0, CC). Push stalls: EX's
`PUSH_STALL`; WFI time: EX's `RV_WFI`.

### 8.6 Outputs, `tsr-prof`, TSR_STATS

- `tsr-prof` ([prof-dbg.md](../prof-dbg.md)) has the source `trw`: Perfetto JSON with process `Tesseract0`,
  one thread per unit (slices = RUN; QWAIT and HWAIT as args and as preceding "waiting" slices), a RISC-V
  thread (launch and node slices from markers), counter tracks per channel (GB/s and idle % from PERF per
  launch). `TSR_PROF_EVERY=n` keeps full TRW for every n-th token.
- TSR_STATS (printed at backend free, per decode token; SW adds launches and ops per launch, SW §9.2): wall;
  device time `CYCLES` / f; host overhead = wall − device time; `DBG_STOP_CYCLES` (must be 0); `MVM_CYCLES`,
  `MVM_WAIT`, `MVM_EMPTY` as % of `CYCLES`; per channel read GB/s and idle %; `RV_WFI` %; `PUSH_STALL`;
  `LOG_DROPPED`. It costs about 16 counter LO reads per launch.

### 8.7 Self-checks (`tsr-prof check`, every profiled launch)

- `RETIRED` (EX: Σ of `DONE_SEQ` advances) = number of OP records, when `TRW_DROPS` = 0; per unit, Σ RUN = Δ
  busy counter (`DMA_CYCLES`, `QACT_CYCLES`, `VU_CYCLES`, `ATT_CYCLES`) exactly; MVM: Σ RUN ≥ Δ`MVM_CYCLES`
  (overlap).
- Per channel c: Δ`CH_RD_BEATS[c]` = Σ expected read beats of the launch's LD and MVM records (rows_c ×
  ROW_BEATS) and of VU and AT records per their areas' formulas.
- LAUNCH_END − LAUNCH_BEGIN ≤ Δ`CYCLES` + Δ`DBG_STOP_CYCLES` (the launch window is wider by the dispatch and
  answer steps only); serial-mode runs: records in push order.

---

## 9. Simulator, vkernel, software

### 9.1 Device parity

| Feature | sim (SW's `tsr_rv` + `tsr_q` + `tsr_dm` + `tsr_dbg`) | vkernel | xrt |
|---|---|---|---|
| DMI, DM | behavioural DM (`sw/sim/tsr_dm.c`, SW §3) implementing the DMI profile below; abstract commands act on the interpreter; the program buffer runs in the interpreter in debug mode | RTL riscv-dbg | RTL |
| SBA | the RV bus model with CC's refusals | RTL | RTL |
| Halt, step, triggers, `ebreak`, RV watchpoints | exact (watchpoints halt right after the access) | RTL (at most 1 instruction later) | RTL |
| Stop logic, QSTEP, serial, TBPs | queue model, §5 semantics exactly | RTL | RTL |
| History, REC, TRW, log | exact contents; times from the cycle model | RTL | RTL |
| Unit probes, crash-dump sample | model values (not compared) | RTL | RTL |

DMI profile (what `libtsr_dbg` uses; the simulator models exactly this, other DM registers read 0 there):
`data0`–`1` (0x04–0x05); `dmcontrol` (0x10: haltreq, resumereq, ackhavereset, ndmreset, dmactive); `dmstatus`
(0x11: version, allhalted, anyhalted, allrunning, allresumeack, allhavereset, allnonexistent, allunavail,
authenticated); `hartinfo` (0x12); `abstractcs` (0x16: progbufsize, busy, cmderr, datacount); `command` (0x17:
Access Register as §3.5, postexec-only); `progbuf0`–`7` (0x20–0x27); `sbcs` (0x38); `sbaddress0` (0x39);
`sbdata0` (0x3C).

Stepping: the standalone server runs the simulator in chunks of at most 10 K modelled cycles while polling the
GDB socket; accesses from the embedded service are applied between chunks under the simulator's lock. With a
fixed schedule a GDB session on `sim` is reproducible.

### 9.2 Transport needs (SW)

`tsr_transport_csr_write` (EX §10.1); DMI helpers over the CSR window (`sim` maps them to `tsr_dm`);
`TSR_ATTACH`; launch waits that suspend their timeout during debug stops (§4.1) and drain the log; the loaded
image exposed to the debug service for symbols; allocation of the TRW ring in the ch0 IO region and a bo read of
it; start of the debug service (`TSR_GDB`, `TSR_DEBUGGER`).

### 9.3 `tsr-dbg` CLI

| Command | Does |
|---|---|
| `gdb [--port N\|--unix P\|--stdio] [--elf F] [--attach\|--load\|--halt-on-boot]` | RSP server (§4) |
| `attach <socket>` | CLI into a process's embedded service (`TSR_DEBUGGER=sock:<path>`) |
| `status [--json]` | EX and DBG decode and the triage verdict (§7.2) |
| `log [-f] [--level L]` | drain and decode the log (ELF for FMT); SW's `tsr-log` decodes saved words |
| `rv halt\|resume\|step [n]\|regs\|sample [-n N]\|reset [halt]\|load F` | RISC-V control without GDB |
| `tile stop [now\|at P]`, `tile break <pidx P[-Q]\|op NAME\|spm A B [r\|w\|rw]\|dram CH A B [r\|w]\|ticket T> [after] [halt] [count]`, `tile delete N`, `tile step [n]`, `tile cont`, `tile serial on\|off`, `tile show` | §5 |
| `hist [-n N] [--pidx P] [--export F]` | history with status, deps and `file:line`; export as `push.bin` |
| `units`, `spm x/<n><f> <line>`, `sreg`, `mem x/<n><f> <addr>` | probes and dumps |
| `perf [--delta]`, `prof …` | PERF snapshot; delegates to `tsr-prof` |
| `dump <dir>`, `recover [--level n]`, `diverge <trace> [--device kind]` | §7.4, §7.3, §9.4 |

Batch mode for CI and scripts: `-ex`, `-x script`, `--json` (one JSON object per command).

### 9.4 Divergence search (`tsr-dbg diverge`)

`tsr-dbg diverge` reports "not implemented"; the method it is specified to follow:

1. Run the launch (SW's trace: image, inputs, descriptors) on `sim` in serial schedule: push stream with push
   indices and, per entry, the written SPM lines and DRAM beats.
2. On the device: `SERIAL` on, `STOP_AT` checkpoints at NODE markers or doubling push indices; at each DRAINED
   stop compare the history entries (a word mismatch is a control divergence: the RISC-V computed a different
   op), then the written SPM lines (SBA) and DRAM regions (bo read) with the simulator at the same T.
3. In the first bad interval, QSTEP entry by entry to the first wrong op; report it with its ticket,
   disassembly, `file:line` of its push, the first wrong element (line, f32 index, expected, actual, ulp), and
   for MVM the channel and cluster.
4. `repro/`: the op and its source lines as a SW trace for the `replay` kernel (SW §7.7), replayable on `sim`,
   vkernel and the card.
5. A mismatch that disappears in serial mode is reported as a concurrency defect.

### 9.5 Environment variables

| Variable | Meaning |
|---|---|
| `TSR_GDB=tcp:<port>\|unix:<path>` | start the embedded RSP service |
| `TSR_GDB_WAIT=boot\|launch` | halt at boot or at the first launch and wait for GDB |
| `TSR_DEBUGGER=sock:<path>` | embedded CLI socket for `tsr-dbg attach` |
| `TSR_LOG`, `TSR_LOG_LEVEL` (SW) | log destination; level written to `LOG_CFG.LEVEL` (0 FATAL … 5 TRACE) |
| `TSR_DUMP_DIR` | hang and post-mortem dumps |
| `TSR_TILE_SERIAL=1`, `TSR_TILE_HIST=0\|1` | serial mode; history on or off (default on) |
| `TSR_PROF=<prefix>`, `TSR_PROF_TRW_MIB`, `TSR_PROF_EVERY`, `TSR_PROF_RV_SAMPLE_US` | profiler, TRW ring size, sampling |
| `TSR_STATS=1` | §8.6 |

Names avoid the `TSR_DBG` prefix (the backend's logging macro in `sw/ggml-tesseract/ggml-tesseract.cpp`).

---

## 10. Register maps

### 10.1 DBG block (core CSR 0x800–0xAFC; U200 `s_axi_control` 0x1800–0x1AFC)

Unlisted offsets read 0 and ignore writes. Reads have no side effects. Hardware-reset values in brackets;
nothing here is reset by QABORT, QCLEAR, the RISC-V reset or `SOFT_RESET`.

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x800 | `DBG_ID` | R | [7:0] version 1 (of this register map; SW §6.9); bit8 history; bit9 TRW; bit10 RV watchpoints; bit11 REC; [14:12] TBPs (4); [16:15] RV watchpoints (2); [23:20] log2 `HIST_DEPTH` (9); 0 = block absent |
| 0x804 | `DBG_CTRL` | W, reads 0 | pulses: bit0 STOP_NOW, bit1 RELEASE, bit2 QSTEP (one of bits 0–2 per write: QSTEP > RELEASE > STOP_NOW); bit4 HIST_CLEAR; bit5 RV_SAMPLE (completes after the sample or 4,096 cycles); bit6 RVWP_APPLY; bit7 TRW_FLUSH; bit8 RV_HALT_CANCEL; bit9 MATCHQ_CLEAR |
| 0x808 | `DBG_CFG` | RW [0x30] | bit0 DMI_EN; bit1 SERIAL; bit2 ERR_STOP; bit3 STOP_HALT_RV; bit4 HIST_EN; bit5 HIST_FREEZE_ERR; bit6 HIST_FREEZE; bit7 HUNG_STOP; bit8 HALT_ON_BOOT; bit9 RV_HALT_EN; bit10 TRW_EN; bit11 TRW_STOP_FULL |
| 0x80C | `DBG_STATUS` | R / W1C | levels: bit0 STOP_ACTIVE; bit1 DRAINED; bit2 STEP_BUSY; bit3 a head held by DB; [8:4] cause of the current stop (NOW, BP, STEP, ERR, HUNG); [12:9] BP_HIT; bit15 RV_HALT_PEND; bit16 RVWP_BUSY; bit17 HIST_FROZEN; bit18 RV_HALTED; bit19 RV_SLEEP; bit31 S_VALID. Sticky W1C: [14:13] RVWP_HIT; bit20 TRW_DROP; bit21 REC_DROP; bit22 SBA_REFUSED; bit23 DMI_BLOCKED; bit24 RV_SAMPLE_TIMEOUT; bit25 RV_DFAULT; bit26 PTR_ERR; bit27 RVWP_TIMEOUT; bit28 MATCHQ_OVF (never expected); bit29 DMI_RESP_ERR |
| 0x810 | `DBG_STOP_PIDX` | R | T (valid while STOP_ACTIVE) |
| 0x814 | `DBG_STOP_AT` | RW | write P: `S` ← P, valid; read: `S` (valid iff `S_VALID`) |
| 0x818 | `DBG_STEP_N` | RW [1] | [15:0] entries per QSTEP (0 = 1) |
| 0x81C | `DBG_NEXT_PIDX` | R | push index the dispatcher assigns to its next ticketed entry |
| 0x820 | `DBG_STEP_PIDX` | R | X of the current or last QSTEP |
| 0x824 | `DBG_MATCHQ` | R | [6:0] pending match thresholds |
| 0x828 / 0x82C | `DBG_MTIME_LO` / `HI` | R | live core time (read HI, LO, HI) |
| 0x834 | `DBG_QSTATE` | R | [10:8] dispatcher state (CC encoding); [23:16] outstanding entries (saturating) |
| 0x840 | `DBG_RV_STATE` | R | bit0 halted, bit2 sleep, bit3 in reset, bit4 SAMPLE_VALID, bit5 `dmactive_o`, bit6 `ndmreset_o`, bit7 halt-request level; [15:8] SAMPLE_SEQ; [31:16] debug-mode entries mod 2^16 |
| 0x844–0x854 | `DBG_RV_PC`, `_NPC`, `_DADDR`, `_EPC`, `_EADDR` | R | last crash-dump sample (§6.2) |
| 0x858 | `DBG_RV_WPADDR` | R | address of the last RV-watchpoint hit |
| 0x860 + 16j | `DBG_RVWPj_CTL` | RW | j < 2: bit0 EN, bit1 LOAD, bit2 STORE, bit3 HALT |
| 0x864 / 0x868 + 16j | `DBG_RVWPj_LO` / `_HI` | RW | byte range [LO, HI) |
| 0x86C + 16j | `DBG_RVWPj_HITS` | R / write clears | hits |
| 0x880 + 16i | `DBG_TBPi_CTL` | RW | i < 4: bit0 EN; [3:1] KIND (0 PIDX, 1 OPCODE, 2 SPM, 3 DRAM, 4 TICKET; 5–7 never match); bit4 R; bit5 W; bit6 AFTER; bit7 HALT_RV; bit8 COUNT_ONLY; [31:16] CHMASK |
| 0x884 / 0x888 + 16i | `DBG_TBPi_A` / `_B` | RW | §5.3 |
| 0x88C + 16i | `DBG_TBPi_HITS` | R / write clears | matches |
| 0x8C0–0x8CC | `DBG_RDOUT0`–`3` | R | outstanding read bursts, 8 bits per channel |
| 0x8D0–0x8DC | `DBG_WROUT0`–`3` | R | outstanding write bursts, 8 bits per channel |
| 0x900 + 16u | `DBG_Uu_STATE` | R | u < 8, §5.5 |
| 0x904 / 0x908 / 0x90C + 16u | `DBG_Uu_OLDEST` / `_OPIDX` / `_AGE` | R | §5.5 |
| 0x980 | `DBG_TRW_BASE` | RW | ch [31:28], beat [27:0] (beat [3:0] read 0) |
| 0x984 | `DBG_TRW_LINES` | RW | ring size in 64-B lines, power of two ≥ 16 (else ignored, PTR_ERR) |
| 0x988 / 0x98C | `DBG_TRW_WPTR` / `_RPTR` | R / RW | records written and consumed (mod 2^32); an RPTR beyond WPTR is ignored (PTR_ERR) |
| 0x990 | `DBG_TRW_DROPS` | R | records dropped (saturating) |
| 0x994 | `DBG_TRW_MASK` | RW [0x7FF] | [7:0] units, bit8 markers, bit9 stops, bit10 RV halts |
| 0x9C0–0x9FC | `SREG0`–`15` | R | the VU's s0–s15 (f32 bits), racy while VU ops run. The one read location of the s bank for the host and the RISC-V (EX §5.1, VU §4.4); present whenever the VU is built, also with `HAS_DBG` = 0 |
| 0xA00 | `DBG_HIST_HEAD` | R | next push index to be recorded |
| 0xA04 | `DBG_HIST_IDX` | RW | push index to read |
| 0xA08 | `DBG_HIST_INFO` | R | bit0 PRESENT; [7:4] unit; [11:8] checker code (0 = runs); bit12 REC; [16:13] TBPs matched; bit17 stop point |
| 0xA0C–0xA20 | `DBG_HIST_PIDX`, `_TICKET`, `_DEP0`, `_DEP1`, `_TIME`, `_PC` | R | §5.6 |
| 0xA40–0xA5C | `DBG_HIST_W0`–`W7` | R | op words |

### 10.2 RISC-V control-window registers (CC's reservations 0x160–0x17C and 0x400–0x7FC)

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x160 | `REC_COUNT` | R | records waiting (0–64) |
| 0x164–0x170 | `REC_DATA0`–`3` | R | head record (§8.2) |
| 0x174 | `REC_POP` | W | drop the head record |
| 0x178 | `TRACE_MARK` | W | marker through the issue FIFO (§8.4) |
| 0x17C | `REC_CFG` | RW [0] | bit0 ALL, bit1 half-full interrupt (`irq_fast[4]`) |
| 0x400 | `DBG_FLAGS` | R | bit0 `dbg_stop` (a DB stop or the debug-halt hold in effect), bit1 SERIAL, bit2 TRW_EN, bit3 DMI_EN (a debugger may be attached), bit4 HALT_ON_BOOT |
| 0x404–0x7FC | | | reserved for DB |

`irq_fast[6]` stays reserved.

### 10.3 Parameters

| Parameter | U200 | Meaning |
|---|---|---|
| `HAS_DBG` | 1 | 0: no DBG block (`DBG_ID` = 0), no halt merge, `dbg_hold` = 0 |
| `DBG_NTBP` | 4 | tile breakpoints (0–4) |
| `DBG_NRVWP` | 2 | RV watchpoints (0–2) |
| `HIST_DEPTH` | 512 | 0 = no history |
| `HAS_TRW`, `TRW_CH` | 1, 0 | trace writer and its channel |
| `REC_DEPTH` | 64 | 0 = no REC |

The matcher and stop logic add no cycle to dispatch or start in any build, so `HAS_DBG` 0 and 1 give identical
results and cycle counts with debugging off.

---

## 11. Cost (U200)

The DM, SBA arbiter, DMI crossing and log ring are in CC's and EX's figures (CC §14.1). The DB block of the
release image uses about 11 K LUT, 7 K FF, 7 RAMB36 and 2 RAMB18 (out of context). The parts are: the halt merge,
HALT_ON_BOOT, two RV watchpoints, the crash-dump holding register, REC, TRACE_MARK and `DBG_FLAGS` in clk_rv; the
REC FIFO (64 × 128 b); the stop logic (T, match queue, DRAINED, QSTEP, serial `min_out`); the matcher with 4 TBPs
and four tagged extent overlaps each; the shadow queues (56 × 128 b LUTRAM), record assembly, TRW staging and the
write source; the history ring (512 × 504 b, 7 RAMB36); the unit and memory probes; the DBG CSR read mux (about
110 words, pipelined); three PERF slots; and the second write source in one `tsr_mem_arb`. Every block is flop →
at most 3 LUT levels → flop with registered boundaries; the history ring and staging sit away from the SPM URAM
columns. `HAS_TRW` and `HIST_DEPTH` can drop without changing any contract (`DBG_ID` reports it).

---

## 12. Decisions

| Id | Decision | Reason |
|---|---|---|
| DB-1 | GDB transport: a built-in RSP server (libtsr_dbg) driving DMI through the CSR window, embedded in the device-owning process or standalone `tsr-dbg` | One CSR access per DMI access; no OpenOCD and no second process, which the exclusive `xrt::ip` would forbid anyway |
| DB-2 | `DBG_CFG.DMI_EN` gates the DMI window (reads 0 and writes dropped when off) | Keeps EX's no-read-side-effect rule for every other bus master |
| DB-3 | Debugger memory access: SBA for all memory access; the program buffer only for fence.i | Works without halting and never changes hart state |
| DB-4 | D15 enforcement: hardware refuses every SBA write to the control and core windows (CC-33); the server never writes the window nor stores to it from the program buffer; inferior calls are unsupported | No debugger path can push a tile op |
| DB-5 | Halt requests from DB: ORed into Ibex debug_req_i, one-shot via a toggle, cleared when the hart halts | Tile stops and watchpoints halt the hart without a DMI round trip; no re-halt |
| DB-6 | Breakpoints and watchpoints: 2 Ibex execute triggers, unlimited ITCM software breakpoints, 2 DB data watchpoints on the Ibex data port | Ibex triggers are execute-only; GDB software watchpoints are unusable over PCIe |
| DB-7 | Halt at boot and launch: boot: a DB halt request pending at reset release; launch: a runtime hook `ebreak` selected by `__tsr_dbg` | Stops exactly at the first instruction or kernel entry for two loads per launch |
| DB-8 | Tile stop mechanism: a push-index threshold T held at the queue heads (dbg_hold); breakpoint matches registered at enqueue | Yields exactly the push-order prefix that the sequential simulator reproduces |
| DB-9 | Hard hold: dbg_hold also blocks non-running completions, except after a QABORT (until the next QCLEAR) and during SOFT_RESET, which clears the stop state | Keeps invariant G at stops while every reset still drains |
| DB-10 | Serial mode: head holds keep only the oldest outstanding entry runnable: one op at a time in push order | Same mechanism as stops; the op stream is untouched |
| DB-11 | Dispatch history: a 512-entry ring of full ops, deps, ticket, push PC and time; recording freezes 256 entries after the first error; exportable as push.bin | A hang needs the ops that never completed; the evidence replays |
| DB-12 | Log ring and record header: EX's 256-word ring, whole records dropped, BLOCK mode; the one header [7:0] N, [11:8] LEVEL 0-5, [15:12] SOURCE, [18:16] TYPE, [19] TS, [27:20] LSEQ, [31:28] 0xA | Never-torn records need no resync; one header for the encoder and every decoder |
| DB-13 | Crash path: with a debugger: restore and ebreak at __tsr_trap_halt (shown as SIGSEGV and so on); without: __tsr_crash, TRAP log record, EX's TRAP record, park; panic = mcause 24 | GDB shows faults natively; the record survives without a debugger |
| DB-14 | Push PC: Ibex current_pc captured at ISSUE_PUSH and carried to the history ring (+32 bits per issue-FIFO entry) | Every tile op maps back to its C++ line |
| DB-15 | PERF semantics: EX's ACTIVE and QHALT gating and counter list; DB adds TRW_RECORDS, TRW_DROPS, DBG_STOP_CYCLES in slots 61–63 | Exact windows; debugged measurements are detectable |
| DB-16 | Timeline transport: a hardware trace writer with 32-byte records to a DRAM ring through a second tsr_mem_arb write source on TRW_CH = 0 (ring in the ch0 IO region); REC stays for firmware | MMIO and firmware streaming are too slow or perturb the measured phases |
| DB-17 | Who runs the server: the device-owning process (embedded service) or tsr-dbg when nothing else owns the device | xrt::ip contexts are exclusive |
| DB-18 | Simulator debug model: a behavioural DM covering the DMI profile, with exact stop, history and trace-writer semantics | Behaviour is identical where the server looks |
| DB-19 | Watchdog: EX's progress watchdog; DB adds HUNG_STOP, a history freeze and per-unit AGE probes | One watchdog is enough |
| DB-20 | Recovery: a 5-level ladder: dump, debug release, LQ_CANCEL, SOFT_RESET with warm restart, card reset; DB's evidence survives every level below a card reset | Post-mortem data exists before any destructive step |
| DB-21 | Host access to the s registers: SREG0-15 at 0x9C0-0x9FC (in the DBG range, present whenever the VU is built), the one read location for host and RISC-V | EX uses 0x0C0 for LOAD_ADDR; one location for every reader |
| DB-22 | SPM bulk dump: a replay launch of ST ops, then a bo read; SBA for small reads and the dead-firmware case | D15-compliant with no new kernel and no SPM client |
| DB-23 | Register placement: DBG block at 0x800-0xAFC, RISC-V control window 0x160-0x17C and 0x400, PERF slots 61-63 | Inside the reservations of EX and CC |

---

## 13. Interfaces with other areas

| Area | DB provides | DB assumes from the area |
|---|---|---|
| EX | DBG block contents; the GDB server in the owning process; halt rules on top of `HOLD_ON_HALT`; the SPM access rule through SBA (§3.6); stop, step and serial semantics and invariants G, Q, S, H; the log record format; trap-record content (`__tsr_crash`); marker encoding; PERF slots 61–63 | the CSR map (DBG 0x800–0xAFC, DMI 0xC00–0xDFC, `CTRL`, `STATUS`, `RV_*`, `LOAD_*`, `Q_*`, `RT_*`, log ring, PERF); ACTIVE and QHALT gating; launch records including TRAP 0x81; `SOFT_RESET` keeping the DBG configuration and DB's evidence (history, TRW pointers, RV snapshot); the progress watchdog; `tsr_transport_csr_write`; the valid-program definition; the opcode table (disassembly, OPCODE breakpoints) |
| CC | `dbg_hold[u]`, `dbg_step[u]`, `dbg_stop`, `dbg_req_rv`; record and trace formats; the contents of the control-window reservations (§10.2) | the DM with DMI, SBA with issue-register refusals, `RV_HALTED`, `ndmreset` → reset sequence, HOLD_ON_HALT, head info one cycle before a start, start and done strobes {seq, pidx, code} with MTIME, the REC flag, the reservations. In addition: (1) `dbg_hold` checked before non-running completions in `H_EVAL`, with the QABORT and `SR_BUSY` exceptions of §5.2 (CC §9.2); (2) completion strobes for non-running completions too, with a `ran` bit; (3) the dispatcher enqueue tap {pidx, ticket, unit, code, w0–w7, deps, REC, push PC, four tagged extents}; (4) the push PC in the issue-FIFO entry; (5) TRACE_MARK as a CTRL entry; (6) `ISSUE_WAIT_MAX` suspended under `dbg_stop`; (7) a second write source on `TRW_CH` in `tsr_mem_arb`; (8) per-channel outstanding counts. Without a requirement a feature degrades alone: no push PC → field 0; no hard hold → invariant G holds for SPM and DRAM but not error records; no write source → TRW off; no enqueue tap → no TBPs and no history |
| VU | breakpoints, history and trace records for VU ops (and for AT's vector ops it executes); `SREG` | the VU ops' operand extents through the checker class (four tagged extents, CC §8.2); start and completion through CC's head; the s bank exported (CC-14) |
| AT | the same for AT ops | AT ops' extents; per-op expected DRAM beats for §8.7; its ops make progress that the watchdog sees |
| SW | `libtsr_dbg`, `tsr-dbg`, the `tsr-prof` `trw` source, TSR_STATS items, the DMI profile, the runtime debug contract (`__tsr_dbg`, `__tsr_crash`, trap handler steps, launch hook, log records, markers) | the transport needs of §9.2; `tsr_dm.c` and the simulator models of the stop logic, history, REC, TRW and log; a serial schedule in the simulator; the SDK implementing §6.3; the trace format and `push.bin`; the `replay` kernel; the ctest wiring of §14 |

---

## 14. Verification

All debug features are observers or schedule controls; the proofs are that (a) they never change a valid
program's results, (b) their outputs equal the simulator's, and (c) the stops they produce are the states the
simulator defines.

1. Simulator models (SW implements DB's semantics): unit tests for every stop event and priority, QSTEP
   selection, RELEASE popping, AFTER, ERR_STOP, HUNG_STOP, serial mode, each TBP kind at its boundaries and
   across push-index wrap, history PRESENT and freeze rules, log records of every TYPE with drops and BLOCK, REC
   and TRW ordering and drops, every field of the DMI profile.
2. Invariants on random programs (SW's race-free generator, CC §15.4), on `sim` and on RTL (`tb/opreplay`,
   `tb/ctrl`) at every clock ratio: (G) at each DRAINED of a random `STOP_AT`, SPM lines written, DRAM regions,
   completion counts and error records equal the simulator's serial run stopped at T; (Q) k QSTEPs from a stop
   equal the stop at T + k; (S) serial final state equals the normal final state and TRW completion order equals
   push order; (H) history equals the simulator push stream. After RELEASE the final state equals the run
   without stops.
3. RISC-V debug integration (`tb/ctrl`, DMI through the CSR window): halt, resume and step on every instruction
   class (including during a waiting push, a bridge access and WFI), abstract access, `cmderr` paths, SBA 8, 16
   and 32-bit with CC's refusals, software breakpoints with and without `fence.i`, both triggers, RV
   watchpoints, HALT_ON_BOOT, DB halt one-shot, `ndmreset` while halted, `DMI_EN` = 0, DMI timeout with clk_rv
   stopped.
4. GDB conformance: a subset of riscv-tests `debug/gdbserver.py` adapted to `tsr-dbg gdb` on `sim` per commit
   and on vkernel nightly, plus ctest GDB batch scripts (break at a kernel symbol, `bt`, `finish`, `watch` on a
   DTCM variable and an SPM vector, launch-break, a crash shown as SIGSEGV, `monitor tile step`).
5. Non-interference: the M6-mode identity check (logits bit-identical to `sim`) repeated with TRW on, history
   on, log at TRACE, serial mode, and a GDB session that halts 100 times at random and continues. The profiler's
   §8.7 self-checks run on every profiled run, and TRW records equal simulator records (kinds, order per unit,
   push indices, tickets, codes; times not compared). `HAS_DBG` = 0 and 1 builds give identical results and
   cycle counts with debugging off.
6. Mutation campaign: start a head with pidx = T; let a held poisoned entry complete; pop the wrong `M` entry on
   RELEASE; DRAINED ignoring a running op; QSTEP releasing T + 1; serial mode allowing the MVM overlap; match
   registered one cycle late; history slot off by one; TRW losing a record without a DROP; DMI_EN ignored; halt
   request not one-shot; RV watchpoint missing the last byte of an access; `ISSUE_WAIT_MAX` counting during a
   stop. Items 1–4 must catch each.
7. Card: `tsr-selftest --debug` (SW §9.5) runs the debug checks of items 3–4 in reduced form, tile stops and
   QSTEP against the in-process simulator shadow, history, log and the TRW ring.

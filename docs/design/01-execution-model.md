# EX: execution model, host interface and ISA

Area EX owns the host-visible model and the global tables that the other areas use: the launch
protocol, the core CSR map, the opcode and unit tables, the error codes, the PERF counters, the XRT
adapter and the transport API. It applies the standing decisions D14–D18 and D1–D6, D12
([spec](../spec.md) §16). The other areas are VU (vector unit), AT (attention and KV), CC (control
core), DB (debug) and SW (software and verification). Numbers marked estimate are design figures, not
measurements.

---

## 1. Summary

- The host never runs programs. It binds channel memory, loads the firmware image into the RISC-V's TCMs
  through a CSR loader (verified by an on-device CRC), starts the RISC-V and launches kernels (D14). The host
  has no access to the issue port (D15).
- A launch is an 8-word descriptor written by posted CSR writes into a core-domain launch queue (LQ, depth 8).
  Each descriptor gets exactly one 8-word answer record, in order, in a core-domain completion ring (LCQ,
  depth 8) that the host reads through the CSR window. A launch needs no PCIe DMA and two PCIe reads per
  completion. A 256-word log ring carries the RISC-V's log records (DB §6.1) and works when every tile queue
  is wedged.
- The kernel uses `ap_ctrl_none`; the host uses `xrt::ip` register access and polls, so a halted RISC-V cannot
  break XRT's scheduler.
- One CSR map (§5), one ISA table (§7) and one error model (§8): codes 1–13, a per-unit error record, one result
  per launch with defined status precedence, `SOFT_RESET` as a device-level reset that works without `clk_rv`,
  cancel and stop watermarks, and a progress watchdog whose recovery is a card reset.

---

## 2. Execution model

### 2.1 What the host does

| Step | Host action | Mechanism |
|---|---|---|
| Open | Map the kernel's registers, allocate one buffer per channel, read identity | `xrt::ip`, `xrt::bo` (§9, §10) |
| Reset | Bring the device to a known state, unless attaching | `CTRL.SOFT_RESET`, wait for `STATUS.SR_BUSY` = 0 (§8.5) |
| Bind | Program the channel bases and enable memory | `ch<c>_base`, then `ADAPTER_CTRL.MEM_EN` = 1 (§9.2) |
| Load | Write the image's `PT_LOAD` segments into ITCM / DTCM and verify them | `LOAD_*` (§3) |
| Boot | Start the RISC-V and wait for its HELLO record | `CTRL.RV_RUN` = 1, LCQ (§4) |
| Data | Write weights, inputs and op lists, and read outputs, by PCIe DMA | [xrt-host.md](../xrt-host.md) |
| Launch | Push descriptors and consume answer records | LQ / LCQ (§4) |
| Control | Cancel, soft stop, soft reset, debug | `LQ_CANCEL`, `LQ_STOP`, `CTRL`, DMI (§4.9, §8, DB) |

Units of host work are launches. Nothing the host does starts a tile op directly.

### 2.2 Device states (host view)

| State | Registers | Host may | Leaves by |
|---|---|---|---|
| RESET | after xclbin load: `RV_RUN` = 0, `MEM_EN` = 0, TCM and SPM undefined | bind, load | load + verify → LOADED |
| LOADED | `RV_RUN` = 0, image verified | boot | `RV_RUN` = 1 → BOOTING |
| BOOTING | `RV_RUN` = 1, no HELLO consumed | wait (`TSR_BOOT_MS`) | HELLO → READY; timeout → FAILED |
| READY | HELLO consumed, no launch outstanding | launch | push → BUSY |
| BUSY | 1 to `LQ_DEPTH` launches outstanding | launch (while < `LQ_DEPTH`), wait, cancel, stop | last answer → READY |
| TRAPPED | TRAP record consumed; `RT_STATE` = TRAPPED | read log, debug (DB), soft reset | `SOFT_RESET` + warm restart (§3.5) → BOOTING |
| HUNG | wait timeout, or `STATUS.UNIT_HUNG` | dump `Q_*` and `DBG_*` | `SOFT_RESET` if it completes, else card reset |
| FAILED | no clock (`RV_CLK_OK` = 0), image mismatch, adapter refused | close | none |

### 2.3 Ordering between the host and the device

[spec](../spec.md) section 16 (Memory model) states the general rules.

- H1. DRAM that the host wrote with a completed transfer before it wrote `LQ_PUSH` for launch s is
  seen by every op pushed while s, or a later launch, runs. "Completed" means the DMA call returned.
- H2. DRAM written by ops of launch s is seen by the host after it read the answer record of s. The
  runtime joins every op of s before it answers (§4.7 R6). A join means `done`, which includes the
  DRAM write responses.
- H3. A host access to DRAM that an outstanding launch reads or writes is a race (undefined). The
  exception is a region that a kernel's own protocol defines, for example a token ring.
- H4. SPM is invisible to the host (DB's debugger reads it through the debug module's system bus,
  DB §3.6). The hardware never changes SPM between launches, and `SOFT_RESET` does not change it
  either. SPM is undefined only after a hardware reset. Kernels may keep SPM state across launches;
  the runtime (SW) owns its allocation. SW's kernels do not: a launch's results must not depend on
  SPM contents left by an earlier launch (SW-4; the simulator's race detector reports such a read as
  `SPM_STALE`), so every launch trace replays on a freshly booted device. A persistent SPM region
  would need an explicit declaration that the detector and the trace capture know about.
- H5. Descriptor words are ordered by PCIe posted-write order and by the AXI4-Lite write channel. The
  adapter handles writes in arrival order ([platform-u200.md](../platform-u200.md) §2), so `LQ_PUSH`
  always sees the staging words written before it.

### 2.4 Ownership

One process owns the device: `xrt::ip` opens an exclusive compute-unit context by default. Tools that
must observe a running process (GDB, profiling) run inside it as threads of the transport. DB
designs the in-process GDB RSP server. The transport serializes CSR accesses with one mutex, and XRT
adds its own.

---

## 3. Firmware image

### 3.1 Format

A static RV32 ELF (no relocations), built by the SDK of SW and CC.

- `PT_LOAD` segments lie inside ITCM [0x0000_0000, `ITCM_BYTES`) or DTCM [0x0010_0000,
  0x0010_0000 + `DTCM_BYTES`) of the RISC-V map (CC §4.1). Their physical addresses are word aligned
  and they do not overlap. The host pads `p_filesz` with zeros up to a whole word. The `.bss` part
  (`p_memsz` − `p_filesz`) is not loaded: crt0 zeroes it.
- `.tsr_kernels`: the kernel table. The kernel index in a descriptor is the position in this table
  (0-based). An entry holds the name hash, the entry point, nargs, flags (bit0 = argument block) and
  the name (SW §6.5).
- `.note.tsr`: Basis ABI, the Tesseract Core capability the image needs (`cap_min`, §7.5), minimum ITCM
  and DTCM sizes, required `CAPS` bits, the Tesseract Basis version (SW §6.2, §6.9). The host refuses
  an image whose requirements the device does not meet, before loading it.
- `.tsr_kinds`: the op list kinds the image provides with their numerics revisions (SW §6.9).
- `.tsr_id`: 8 bytes in a DTCM `PT_LOAD` segment, holding the image id (§3.2).

### 3.2 Image id

The image id is a 64-bit value defined as follows:

1. Order the `PT_LOAD` segments by ascending physical address.
2. Hash with SHA-256, for each segment in that order: its physical address as a 32-bit little-endian
   word, its padded file length as a 32-bit little-endian word, then the padded bytes. The 8 bytes of
   `.tsr_id` are taken as zero.
3. The image id is the first 8 bytes of the digest, read as a little-endian `uint64`.

`tsr-elf` (SW) writes the id into `.tsr_id` after linking. The runtime echoes it in HELLO (r2 =
bits [31:0], r3 = bits [63:32]), and the host compares it with its own computation. 64 bits are enough
to tell builds apart; the integrity check is the CRC (§3.3).

### 3.3 Cold load

Preconditions: `RV_RUN` = 0 and `STATUS.SR_BUSY` = 0. Otherwise every `LOAD_DATA` write and every
VERIFY command is refused (`STATUS.LOAD_REFUSED`), and `LOAD_DATA` reads return 0.

```
for each PT_LOAD segment (padded):
    LOAD_ADDR  := segment physical address          # resets LOAD_CRC
    LOAD_DATA  := word[0], word[1], ...              # address += 4 after each accepted word
    check LOAD_CRC == crc32(segment bytes)           # what the loader received
    LOAD_ADDR  := segment physical address
    LOAD_COUNT := number of words
    LOAD_CTRL  := VERIFY                             # loader reads the words back from the TCM
    wait LOAD_STATUS.VERIFY_BUSY == 0
    check LOAD_CRC == crc32(segment bytes)           # what the TCM holds
RV_BOOT := vector table base (256-byte aligned, normally 0)
```

- CRC-32 is zlib's `crc32` (reflected polynomial 0xEDB88320, initial value 0xFFFFFFFF, final XOR
  0xFFFFFFFF). It is computed over the bytes of each word in address order (little-endian). A write to
  `LOAD_ADDR` restarts the CRC, so `LOAD_CRC` reads 0x0000_0000 (the CRC of zero bytes).
- VERIFY: a core-domain sequencer reads `LOAD_COUNT` words from `LOAD_ADDR` through the load port and
  computes `LOAD_CRC` over them. Each word read is a tagged request bounded by 4,096 core cycles. On a
  timeout, `LOAD_TIMEOUT` is set and the sequencer stops with `VERIFY_BUSY` = 0, so the CRC is then
  wrong. VERIFY catches broken TCM writes and address decode faults that the receive CRC cannot see,
  without any PCIe reads.
- `LOAD_DATA` writes follow the CDC rule of CC §11.2: each write goes into a core-side holding
  register, and a write that cannot complete within 4,096 core cycles is dropped with `LOAD_TIMEOUT`
  set. So the CSR port never hangs.
- Time (estimate): 128 KiB is 32 K posted writes. The AXI4-Lite slave accepts about one word per 15–25
  core cycles, so 2–4 ms, plus the host posting rate. VERIFY of 128 KiB takes 0.6–2 ms. Total at most
  10 ms per cold load.

### 3.4 Boot

`CTRL.RV_RUN` := 1. The RISC-V leaves reset and fetches at `RV_BOOT` + 0x80 (CC §12.2). crt0 and the
runtime boot (§4.7 R1) end with a HELLO event. The host consumes it within `TSR_BOOT_MS` (default 2,000
ms) and checks: the image id, the Basis ABI, the kind table signature, the kernel count, the Tesseract
Basis version (SW §6.9), and that no other record came first.

### 3.5 Warm restart

A warm restart is used after a TRAP, a hang of the RISC-V, or a new process that finds the same image
already loaded. Steps: `SOFT_RESET` (§8.5), reload every DTCM segment (`.rodata` and `.data`: a few
KiB), VERIFY every ITCM segment (no ITCM writes when the CRC matches, otherwise a cold load of that
segment), then boot. ITCM is writable by the RISC-V (software breakpoints, overlays), which is why it
is verified and not assumed to be intact. Time at most 2 ms (estimate). The transport uses a warm
restart when `LOAD_CRC` after VERIFY matches for every ITCM segment.

---

## 4. Launch protocol

### 4.1 Structure

```
 host (PCIe posted writes)            core clock domain                       clk_rv domain (RISC-V)
 LQ_W0..7 ──► staging ──LQ_PUSH──► LQ ring, 8 × 8 words  ──(CDC)──►  LQ_HEAD_W0..7, LQ_AVAIL, LQ_POP
 LCQ_WR/RD, ring reads ◄──────────  LCQ ring, 8 × 8 words  ◄──(CDC)──  LCQ_W0..7 staging, LCQ_POST
 LOG_WR/RD, ring reads ◄──────────  LOG ring, 256 words    ◄──(CDC)──  LOG_PUSH
 LQ_CANCEL / LQ_STOP   ──────────►  watermarks (+QABORT)   ──(CDC)──►  LQ_CANCEL / LQ_STOP copies
```

All three rings and all their counters live in the core domain. A host access never waits for
`clk_rv`, and `SOFT_RESET` can flush the rings without `clk_rv`. The RISC-V side reaches them by
tagged handshakes into the core domain (CC §11.2, CC-30), never through gray-coded copies of the
counters, because the host-written counters (`LCQ_RD`, `LOG_RD`) and the flushes of `SOFT_RESET` move
them by more than one step at a time:

- every RISC-V read of `LQ_HEAD_W*`, `LQ_AVAIL`, `LQ_HEAD_SEQ`, `LQ_CANCEL`, `LQ_STOP`, `LQ_FLAGS`,
  `LCQ_FREE` and `LOG_CFG` returns the core-domain value at the moment the core answers the
  handshake;
- `LQ_POP` and `LCQ_POST` are evaluated in the core domain at commit (the empty and full checks
  included), and the store completes after the core domain has changed;
- `LOG_PUSH` words cross through a small CDC FIFO. `LOG_FREE` = `LOG_DEPTH` − (LOG_WR_rv − LOG_RD_snap),
  where LOG_WR_rv counts accepted `LOG_PUSH` stores in `clk_rv` and LOG_RD_snap is the core-domain
  `LOG_RD` fetched by the same handshake as the `LOG_FREE` read;
- the interrupt levels `irq_fast[2]` and `[3]` (§4.5) are synchronized core-domain booleans. A late or
  stale level only causes a late or spurious wake-up, after which the runtime re-reads the register.

So every RISC-V-side value may lag the core-domain value in the safe direction (fewer descriptors,
less free space) but never leads it.

### 4.2 Counters and invariants

All counters are 32-bit, modulo 2^32, and are zeroed only by a hardware reset.

| Counter | Advanced by | Host register |
|---|---|---|
| PUSHED | every accepted `LQ_PUSH` | `LQ_SEQ` (0x064) |
| TAKEN | every `LQ_POP` by the RISC-V, and the LQ flush of `SOFT_RESET` (TAKEN := PUSHED) | `LQ_TAKEN` (0x068) |
| ANSWERED | every `LCQ_POST` with ANSWER = 1, and the RISC-V reset sequence or `SOFT_RESET` (ANSWERED := TAKEN) | `LCQ_ANSWERED` (0x084) |
| LCQ_WR | every `LCQ_POST` | 0x074 |
| LCQ_RD | host writes; `SOFT_RESET` (LCQ_RD := LCQ_WR) | 0x078 |
| LOG_WR, LOG_RD | `LOG_PUSH` (accepted words); host writes; `SOFT_RESET` (LOG_RD := LOG_WR) | 0x090, 0x094 |

- The sequence number of a descriptor is the value of PUSHED when it was pushed. The host learns it
  without reading: it reads `LQ_SEQ` once at open and counts its own pushes after that.
- ACTIVE := TAKEN ≠ ANSWERED. The hardware keeps TAKEN − ANSWERED ∈ {0, 1}: an `LQ_POP` while ACTIVE is
  refused with an access fault. While ACTIVE, the running launch's sequence number is ANSWERED.
- PUSHED − TAKEN ≤ `LQ_DEPTH`. A push when the ring is full is dropped and sets `STATUS.LQ_OVERFLOW`.
  The transport keeps at most `LQ_DEPTH` launches outstanding (pushed and not yet answered), which
  makes an overflow impossible without any read (§4.8).
- LCQ_WR − LCQ_RD ≤ `LCQ_DEPTH`. Record k lives at core CSR 0x300 + 32 × (k mod `LCQ_DEPTH`) until the
  host consumes it. A host write to `LCQ_RD` of a value v is accepted iff
  (v − LCQ_RD) mod 2^32 ≤ (LCQ_WR − LCQ_RD) mod 2^32. Otherwise it is ignored and `STATUS.RING_RD_BAD`
  is set. `LOG_RD` follows the same rule.
- U200 parameters: `LQ_DEPTH` = `LCQ_DEPTH` = 8, `LOG_DEPTH` = 256 words. They are readable in
  `LQ_CFG` (0x07C). The CSR map limits `LCQ_DEPTH` to 8 and `LOG_DEPTH` to 256.

### 4.3 Descriptor (host → RISC-V), 8 words

| Word | Bits | Field |
|---|---|---|
| w0 | [7:0] | TYPE: 0x01 LAUNCH, 0x02 PING; every other value is reserved |
| w0 | [8] | PROF: a profiled launch. The runtime sets `REC_CFG.ALL` for the launch (completion records for firmware self-profiling, DB §8.2) and kernels emit NODE markers (the graph executor behaves as if the op-list flag MARK were set, SW §7.2). The host timeline itself comes from DB's hardware trace writer (DB §8.3), which the host enables separately |
| w0 | [15:9] | reserved, must be 0 |
| w0 | [31:16] | KERNEL: index into `.tsr_kernels` (LAUNCH); must be 0 for PING |
| w1–w7 | [31:0] | ARG0–ARG6: integers, SPM line numbers, or DRAM handles {ch[31:28], beat[27:0]} |

A kernel whose table entry has the argument-block flag gets ARG0 = the DRAM handle of a block of at
most 1 KiB. The runtime stages that block into the SPM argument range before the call. The descriptor
is answered with BAD_DESC when any of these holds: an unknown TYPE, reserved bits set, KERNEL beyond
the table, or an argument-block handle with ch ≥ `NUM_CH` or a block that does not fit inside
`CH_BYTES`.

### 4.4 Record (RISC-V → host), 8 words

r0 has one layout for every record type: [7:0] TYPE, [15:8] STATUS, [23:16] ERR_CODE, [31:24]
SEQ[7:0]. With r0 alone the host can classify a record and match it to a launch.

| Word | DONE (0x01, answer) | PONG (0x02, answer) | HELLO (0x80, event) | TRAP (0x81, answer if ACTIVE, else event) | kernel event (0xC0–0xFF) |
|---|---|---|---|---|---|
| r0 | type, status, first error code, seq[7:0] | type, 0, 0, seq[7:0] | type, 0, 0, 0 | type, 4, 0, seq[7:0] | type, kernel-defined [23:8], seq[7:0] |
| r1 | SEQ of the answered descriptor | SEQ | TAKEN at the post | SEQ (answer) or TAKEN | SEQ of the running launch |
| r2 | RET: kernel return value (int32) | ARG0 echo | image id [31:0] | `mcause` | kernel-defined |
| r3 | first failing ticket; 0xF000_0000 = none | 0 | image id [63:32] | `mepc` | kernel-defined |
| r4 | its `ERR_INFO` | 0 | [15:0] Basis ABI, [31:16] kind table signature (SW §6.9) | `mtval` | kernel-defined |
| r5 | its `ERR_W0` | 0 | kernel count | `RT_STATE` at the trap | kernel-defined |
| r6 | its `ERR_PIDX` | 0 | Tesseract Basis version: [23:16] major, [15:8] minor, [7:0] patch (SW §6.9) | `RT_INFO` | kernel-defined |
| r7 | DURATION: core cycles from the pop to the post, saturating at 2^32 − 1 | same | boot core cycles | MTIME[31:0] | kernel-defined |

Launch STATUS (r0[15:8] of DONE):

| Code | Name | Meaning |
|---|---|---|
| 0 | OK | the kernel returned RET ≥ 0, and no op of the launch failed |
| 1 | TILE_ERR | the first error (smallest `ERR_PIDX`, §8.2) has a code other than ABORTED; r3–r6 describe it |
| 2 | BAD_DESC | the descriptor was refused (§4.3); the kernel did not run |
| 3 | KERNEL_ERR | RET < 0 (the kernel rejected its arguments or failed in software); no tile error |
| 4 | TRAP | used only in TRAP records |
| 5 | ABORTED | canceled: the launch was queued and is answered without running (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8) |
| 6 | PLAT_ERR | `STATUS.PLAT_ERR` was set when the runtime answered: the platform reported an AXI error, so outputs may be wrong |
| 7 | STOPPED | the kernel honored a soft stop (§4.9) and returned early; RET is kernel-defined |
| 8–255 | reserved | |

Precedence when several codes apply: TRAP > BAD_DESC > ABORTED > TILE_ERR > PLAT_ERR > STOPPED >
KERNEL_ERR > OK. ERR_CODE (r0[23:16]) is the code of the first error whatever the status, and 0 when
there is none. A cancel that lands after the kernel's work has completed but before the answer may
still report ABORTED. The host treats the outputs of any canceled launch as undefined.

### 4.5 RISC-V side (control window offsets 0x200–0x2FC, owned by EX)

CC owns the rest of the control window. Every access is 32-bit; reads have no side effects.

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x200–0x21C | `LQ_HEAD_W0`–`W7` | R | the descriptor that the next `LQ_POP` takes (valid while `LQ_AVAIL` ≠ 0) |
| 0x220 | `LQ_AVAIL` | R | PUSHED − TAKEN as seen in `clk_rv` |
| 0x224 | `LQ_HEAD_SEQ` | R | TAKEN = the sequence number of the head |
| 0x228 | `LQ_POP` | W | take the head (TAKEN += 1, ACTIVE := 1). The store completes after TAKEN has changed in the core domain. Access fault if `LQ_AVAIL` = 0 or ACTIVE |
| 0x22C | `LQ_CANCEL` | R | copy of the host's cancel watermark |
| 0x230 | `LQ_STOP` | R | copy of the host's stop watermark |
| 0x234 | `LQ_FLAGS` | R | bit0 CANCEL_VALID, bit1 STOP_VALID, bit2 ACTIVE, bit3 PLAT_ERR (`STATUS.PLAT_ERR`) |
| 0x240–0x25C | `LCQ_W0`–`W7` | RW | record staging; kept after a post |
| 0x260 | `LCQ_POST` | W | wdata bit0 ANSWER. Copies the staged words to slot LCQ_WR mod depth, then LCQ_WR += 1, and with ANSWER also ANSWERED += 1. The store completes after the record is in the core-domain ring. Access fault if `LCQ_FREE` = 0, or if ANSWER = 1 while not ACTIVE |
| 0x264 | `LCQ_FREE` | R | `LCQ_DEPTH` − (LCQ_WR − LCQ_RD) as seen in `clk_rv` (may under-report) |
| 0x280 | `LOG_PUSH` | W | one log word. When the ring is full, the word is dropped and counted (`LOG_DROPPED`); this never faults |
| 0x284 | `LOG_FREE` | R | free words (may under-report) |
| 0x288 | `LOG_CFG` | R | copy of the host's `LOG_CFG` |
| 0x290 | `RT_STATE` | RW | runtime phase, mirrored to host 0x038: [7:0] phase (0 RESET, 1 BOOT, 2 IDLE, 3 DISPATCH, 4 KERNEL, 5 JOIN, 6 ANSWER, 7 TRAPPED), [31:16] kernel index. 0 after a RISC-V reset |
| 0x294 | `RT_INFO` | RW | runtime- or kernel-defined progress word (for example, the op index being pushed), mirrored to host 0x03C |

Interrupts: `irq_fast[2]` is pending while `LQ_AVAIL` ≠ 0 and not ACTIVE; `irq_fast[3]` is pending while
`LCQ_FREE` ≠ 0 (the runtime enables it only while it waits to post).

### 4.6 Base kernels the host relies on

The SW base image provides these three kernels, because bring-up, the backend and verification depend
on them (contracts in SW §7):

- `graph`: the op-list executor (the op list format, SW §7.2–§7.3). It is the only offload path:
  in M6 mode the backend emits op lists with MUL_MAT records only, one launch per graph split (SW-14);
  in M7 mode it emits the other kinds as well, so a decode token is one launch.
- `replay`: replays a tile-only `push.bin` stream (SW §5.2, §7.7). ARG0 = its DRAM handle, ARG1 = the
  record count, ARG2 = flags. With flag bit0 SERIAL every op record depends on the previous op record
  instead of its recorded deps; because of I1 this gives strictly sequential semantics. Opcodes that
  the issue path refuses complete `ILLEGAL_OP` on the SYNC unit, as on the issue path. RET = 0 when no
  op failed, else 1 + the push index of the first error relative to the first replayed record (the
  kernel's own `push.bin` chunk LDs not counted, SW §7.7). The `ops` trace sets, DB's SPM bulk dumps
  (DB §5.7) and the divergence search run on this kernel.
- `selftest`: the card selftest (D15, SW §7.6), launched by `tsr-selftest`.

### 4.7 Runtime dispatcher contract (normative for the SDK runtime)

- R1 Boot. crt0 (SW §6.3), then `qclear()` and a join on it, then `RT_STATE` := IDLE, then post HELLO as
  an event. Descriptors pushed before HELLO stay queued and are served after it.
- R2 Take. Wait for `LQ_AVAIL` ≠ 0 (WFI on `irq_fast[2]`). Copy `LQ_HEAD_W*` and `LQ_HEAD_SEQ` into
  DTCM, then write `LQ_POP`.
- R3 Canceled or stopped before start. If CANCEL_VALID and the sequence is covered by `LQ_CANCEL`, or
  STOP_VALID and it is covered by `LQ_STOP` (§4.9), answer DONE / ABORTED (ERR_CODE 0) without
  running.
- R4 Validate. BAD_DESC → answer. PING → answer PONG. The validation runs before R3's coverage check:
  a descriptor that is both invalid and covered answers BAD_DESC (§4.4's precedence, BAD_DESC above
  ABORTED).
- R5 Run. `RT_STATE` := KERNEL | index. If PROF is set, set `REC_CFG.ALL` and enable NODE markers
  (w0[8] above). Stage the argument block if the table entry asks for one. Read `ISSUE_PIDX` (CC §5.1)
  as the launch's first push index. Call DB's launch hook (DB §3.8), then the kernel; it returns RET
  (int32).
- R6 Join. `join_all()` (FENCE, then join): every op pushed during the launch has completed, including
  aborted ones (I1).
- R7 Result. Find the first error by the smallest `ERR_PIDX` over the units in error state (CC §9.5).
  Read PLAT_ERR from `LQ_FLAGS`. Fill r0–r7 by the precedence of §4.4.
- R8 Answer. Wait until `LCQ_FREE` ≠ 0 (WFI on `irq_fast[3]`), then post with ANSWER = 1.
- R9 Clear. If any unit is in error state, push `qclear()` and join it before R2.
- R10. `RT_STATE` := IDLE; continue at R2.

Further rules for the runtime and for kernels:

- Kernels never pop and never post answers. They may post events (types 0xC0–0xFF, ANSWER = 0).
- Kernels never leave an op running at return: R6 enforces this. This is what makes ACTIVE the exact
  PERF window (§6).
- Long kernels poll `tsr::stop_requested()` (STOP_VALID and their sequence covered) at their own safe
  points.
- Logs go only to the log ring, never to the LCQ.
- On a trap, the handler follows DB §6.3: with a debugger attached it stops at `__tsr_trap_halt`;
  otherwise it fills `__tsr_crash`, logs a TRAP record, posts a TRAP record (ANSWER = 1 if ACTIVE) and
  parks in a WFI loop with `RT_STATE` = TRAPPED. The debugger can attach there. The runtime does not
  unwind, because a trap can leave `.data` inconsistent. `tsr::panic()` (violated firmware invariants)
  enters the same handler with `mcause` = 24 and `mtval` = its code; there is no separate PANIC
  message.
- A soft stop is the host writing `LQ_STOP` (§4.9). There is no STOP or BYE exchange and no runtime
  log-level message: the runtime filters by `LOG_CFG.LEVEL`.

### 4.8 Host procedures

```
launch(d):   while outstanding == LQ_DEPTH: consume()           # host-side credit, no LQ_FREE read
             write the LQ_W words that differ from the host's shadow copy; write LQ_PUSH
             seq = pushed++; outstanding++; return seq
consume():   v = read LCQ_WR
             for k in [rd, v): r0 = read rec(k).r0               # r2..r6 only if status != OK,
                                                                 # the kernel returns data, or debug
                 answer → result[r1] (outstanding--); event → event queue / callback
             write LCQ_RD = v; rd = v
wait(s, t):  until result[s] is present: consume(); then poll with the policy below; on timeout → §8.6
```

- Timeout clock (DB §4.1): the wait timeout counts only while `STATUS.RV_HALTED` = 0 and
  `DBG_STATUS.STOP_ACTIVE` = 0 (both read once per backoff poll); with `DBG_CFG.SERIAL` = 1 it is off.
  The transport releases its mutex between polls, so an in-process debug service (DB §4.1) can use the
  device while a launch waits.
- Every wait loop drains the whole LCQ. A kernel that posts events therefore cannot block the runtime
  while the host waits.
- Poll policy (default; SW may refine it): busy-poll `LCQ_WR` while the elapsed time is below the last
  DURATION of the same kernel. After that, poll every max(10 µs, 1 % of that duration), sleeping
  between polls. Detection latency is then about max(10 µs, 1 % of the launch). Each poll is one PCIe
  read, and polls never touch DRAM bandwidth.
- The host reads `STATUS` only on a timeout or a non-OK status. It reads the log ring only when a
  launch fails, every 100 ms or more during long waits, and at close.

### 4.9 Cancel and stop

Launch q is covered by a watermark s iff (s − q) mod 2^32 < 2^31.

- `LQ_CANCEL` (0x06C) is a hard cancel. Writing s sets the watermark to s and CANCEL_VALID to 1. In
  the same cycle, if ACTIVE and the running launch (sequence ANSWERED) is covered by s, the core
  applies a host QABORT at once (§8.5). Queued launches that are covered are answered ABORTED without
  running (R3). Uncovered launches run normally: the runtime's R9 clears the error states left by the
  abort. This is the operation behind Ctrl-C, the ggml abort callback and the host's timeout path.
- `LQ_STOP` (0x070) is a soft stop. Writing s sets the watermark to s and STOP_VALID to 1. There is no
  hardware action. Covered queued launches are answered ABORTED. A covered running kernel sees
  `stop_requested()` and returns at its next safe point with status STOPPED (for example, a decode
  loop at a token boundary). A kernel that never polls simply runs to the end.
- Both watermarks persist until `SOFT_RESET`, which clears both VALID bits. Writing s = PUSHED − 1
  covers exactly the launches pushed so far.

### 4.10 Log ring

- The RISC-V writes 32-bit words to `LOG_PUSH`. The host reads word k at 0x400 + 4 × (k mod
  `LOG_DEPTH`) for k in [`LOG_RD`, `LOG_WR`), then writes `LOG_RD`.
- Framing, which the SDK follows: a record is one header word followed by N payload words. The header
  is DB's (DB §6.1, the only definition): [7:0] N, [11:8] LEVEL (0 FATAL, 1 ERROR, 2 WARN, 3 INFO,
  4 DEBUG, 5 TRACE), [15:12] SOURCE (0 runtime, 1 kernel, 2 trap handler, 3 debug kernels), [18:16]
  TYPE (FMT, TEXT, TRAP, EVENT, BLOB), [19] TS, [27:20] LSEQ = SEQ[7:0] of the running launch, [31:28]
  SYNC = 0xA. The SDK writes a record only when `LOG_FREE` ≥ 1 + N, so records are never torn. With
  `LOG_CFG.BLOCK` = 1 it waits for space instead of dropping the record.
- `LOG_CFG` (host RW): bit0 ENABLE (0: the SDK drops every record), bit1 BLOCK, [7:4] LEVEL (the SDK
  drops records above this level; this is the only runtime level filter). Words that the hardware
  drops are counted in `LOG_DROPPED` (saturating) and set `STATUS.LOG_DROP`.
- The log ring is for diagnosis, not for bulk data: the host pays one PCIe read per word. Op traces go
  to DRAM through DB's hardware trace writer (DB §8.3).

### 4.11 Latency and throughput targets

Design targets, in estimated costs per step of an empty launch:

| Step | Cost (estimate) |
|---|---|
| Host: 2–9 posted MMIO writes (changed words and `LQ_PUSH`) | 0.2–1 µs |
| PCIe, adapter, CSR port | 0.5–1 µs |
| LQ CDC, WFI wake-up, 8 head words, pop | 0.4–1.2 µs |
| Dispatch, empty kernel, FENCE join, result, post | 0.6–1.5 µs |
| Host detection: 1–2 poll reads plus the r0 read; `LCQ_RD` write | 2–5 µs |
| Round trip of an empty launch | about 4–8 µs; target at most 10 µs |
| Pipelined rate with at least 2 launches outstanding | target at least 100 K launches/s |

Between two launches the device idles for only the dispatch and answer time (about 1–3 µs). That is
negligible for the goal of keeping the DDR busy ([spec](../spec.md) §14).

---

## 5. Core CSR map (host view)

### 5.1 Offsets

Core CSR offsets are 12-bit; on the U200 the window is `s_axi_control` 0x1000 + offset. Every read has
no side effects except the PERF LO snapshot (host port only). A `LOAD_DATA` read while `RV_RUN` = 1 or
`SR_BUSY` returns 0 and sets nothing. The DMI window (0xC00–0xDFC) acts only while `DBG_CFG.DMI_EN` = 1
(DB §3.2), because debug-module reads can have side effects. Undefined offsets read 0 and ignore writes
([spec](../spec.md) §10). Every access completes without `clk_rv`, within a bound (CC §5.3). This is the
only host CSR map; CC, DB, VU and AT place their registers inside it, and the RISC-V reads it through
CC's core window (CC §5.3).

| Offset | Name | Access | Content | Owner |
|---|---|---|---|---|
| 0x000 | `ID` | R | 0x5453_5231 "TSR1" | EX |
| 0x004 | `VERSION` | R | Tesseract Core capability (§7.5): [15:8] major = 1, [7:0] minor = 0 | EX |
| 0x008 | `CFG0` | R | NUM_CH[7:0], LANES[19:8], T_MAX[23:20] | EX |
| 0x00C | `CFG1` | R | SPM size in lines | EX |
| 0x010 | `CFG2` | R | VL[7:0], NSREG[15:8], K_MAX/256[31:16] | EX |
| 0x014 | `CAPS` | R | §7.5 | EX |
| 0x018 | `CFG3` | R | [15:0] defined by VU, [31:16] defined by AT | VU, AT |
| 0x01C | `CTL_ID` | R | [7:0] control-core revision (1; 0 = no control core: the host refuses the device), bit8 debug module, bit9 completion records, bit10 SPM bridge, bit11 log ring, bit12 watchdog, bit13 SBA connected, bit14 SPM line-buffered alias | EX, CC |
| 0x020 | `CTRL` | RW / W1 | §5.2 | EX |
| 0x024 | `STATUS` | R / W1C | §5.3 | EX |
| 0x028 | `RV_BOOT` | RW | boot vector, 256-byte aligned ([7:0] read 0); taken when `RV_RUN` rises. A write while `RV_RUN` = 1 is ignored, so the bus that crosses to `clk_rv` is stable whenever the RISC-V samples it (reset release, and the restart after a debug ndmreset) | CC |
| 0x02C | `RV_FREQ` | R | `clk_rv` cycles in the last 2^16 core cycles; 0 = no `clk_rv` | CC |
| 0x030 | `RV_CFG` | R | [15:0] ITCM KiB, [31:16] DTCM KiB | CC |
| 0x034 | `WDOG_CFG` | RW | [4:0] L: unit watchdog limit of 2^L core cycles without progress; 0 = off; reset value 24 (§8.6) | EX |
| 0x038 | `RT_STATE` | R | copy of the runtime's `RT_STATE` (§4.5) | EX |
| 0x03C | `RT_INFO` | R | copy of the runtime's `RT_INFO` | EX |
| 0x040–0x05C | `LQ_W0`–`LQ_W7` | RW | descriptor staging; kept after a push | EX |
| 0x060 | `LQ_PUSH` | W | push the staged descriptor (value ignored); dropped with `LQ_OVERFLOW` when full | EX |
| 0x064 | `LQ_SEQ` | R | PUSHED | EX |
| 0x068 | `LQ_TAKEN` | R | TAKEN | EX |
| 0x06C | `LQ_CANCEL` | RW | cancel watermark (§4.9) | EX |
| 0x070 | `LQ_STOP` | RW | stop watermark (§4.9) | EX |
| 0x074 | `LCQ_WR` | R | records posted | EX |
| 0x078 | `LCQ_RD` | RW | records consumed by the host (§4.2 write rule) | EX |
| 0x07C | `LQ_CFG` | R | [7:0] `LQ_DEPTH`, [15:8] `LCQ_DEPTH`, [27:16] `LOG_DEPTH`, bit30 STOP_VALID, bit31 CANCEL_VALID | EX |
| 0x080 | `PERF_INFO` | R | [7:0] counters implemented, [15:8] first extension index X = 5 + 2 × `NUM_CH` | EX |
| 0x084 | `LCQ_ANSWERED` | R | ANSWERED | EX |
| 0x088 | `Q_DEPTH` | R | queue depth of unit u at [5u+4:5u], u = 0..5; 0 = unit absent (U200: DMA 8, QACT 4, MVM 4, VU 16, unit 4 absent, SYNC 4) | CC |
| 0x090 | `LOG_WR` | R | log words accepted | EX |
| 0x094 | `LOG_RD` | RW | log words consumed by the host | EX |
| 0x098 | `LOG_CFG` | RW | bit0 ENABLE, bit1 BLOCK, [7:4] LEVEL; reset value 0x0000_00F1 | EX |
| 0x09C | `LOG_DROPPED` | R | words dropped, saturating; cleared by `SOFT_RESET` | EX |
| 0x0C0 | `LOAD_ADDR` | RW | TCM byte address (word aligned); a write restarts `LOAD_CRC`. Refused while `VERIFY_BUSY` = 1 (the write is dropped and sets `LOAD_REFUSED`: it would restart the CRC that VERIFY is folding) | EX |
| 0x0C4 | `LOAD_DATA` | RW | write: store the word, address += 4; read: the word at the address (no increment). Only while `RV_RUN` = 0 and not `SR_BUSY`: otherwise a write is dropped and sets `LOAD_REFUSED`, and a read returns 0 and sets nothing | EX |
| 0x0C8 | `LOAD_CRC` | R | CRC-32 of the words accepted, or of the words read by VERIFY | EX |
| 0x0CC | `LOAD_COUNT` | RW | word count for VERIFY | EX |
| 0x0D0 | `LOAD_CTRL` | W | bit0 VERIFY | EX |
| 0x0D4 | `LOAD_STATUS` | R | bit0 VERIFY_BUSY | EX |
| 0x100–0x2FC | `PERF_0`–`PERF_63` | R | 64-bit counters (§6): LO at 0x100 + 8i, HI at +4; a LO read snapshots the counter for the next HI read ([spec](../spec.md) §21, `PERF_*` hardware meaning) | EX |
| 0x300–0x3FC | `LCQ_REC` | R | completion ring: record k at 0x300 + 32 × (k mod `LCQ_DEPTH`), word i at +4i | EX |
| 0x400–0x7FC | `LOG_RING` | R | log word k at 0x400 + 4 × (k mod `LOG_DEPTH`) | EX |
| 0x800–0xAFC | `DBG_*` | DB-defined | tile-level debug block (768 B, DB §10.1); undefined offsets read 0 | DB |
| 0x9C0–0x9FC | `SREG0`–`SREG15` | R | the VU's s0–s15 (f32 bits), inside the DBG range but present whenever the VU is built, independent of `HAS_DBG` (VU §4.4, DB §10.1). The one read location of the s bank for the host and for the RISC-V (core window 0x2000_1000 + 0x9C0 + 4i) | VU, DB |
| 0xB00–0xBFC | | | reserved | |
| 0xC00–0xDFC | `DMI` | R / W | debug module register r at 0xC00 + 4r; bounded by 4,096 core cycles, `DMI_TIMEOUT` | DB, CC |
| 0xE00–0xF1C | `Q_*` | R | queue status (§5.4) | EX |
| 0xF20–0xFFC, and every gap above | | | reserved | |

### 5.2 `CTRL` (0x020)

| Bit | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `RV_RUN` | RW level | 1 releases the RISC-V from reset. 1 → 0 starts the RISC-V reset sequence (§8.5). Writes are ignored, with `RV_RUN_REFUSED` set, while `SR_BUSY` |
| 1 | `HOST_HOLD` | RW level | the host's hold source for the queue heads (CC §9.6) |
| 2 | `HOLD_ON_HALT` | RW level | a debug halt holds the queue heads |
| 3 | `IRQ_EN` | RW level | `host_irq_o` = `IRQ_EN` ∧ (LCQ_WR ≠ LCQ_RD); the U200 leaves the output unconnected |
| 8 | `QABORT` | W1 pulse | host QABORT, applied at once (§8.5) |
| 9 | `QCLEAR` | W1 pulse | host QCLEAR: accepted only when QIDLE holds and the RISC-V is quiet (reset done, or halted with no push in flight); otherwise `QCLEAR_REFUSED`. "Quiet" is judged on the state before this write's own `RV_RUN` change: a QCLEAR in the same write as `RV_RUN` 1 → 0 is refused (unless the hart is halted), because the reset has not completed; one in the same write as `RV_RUN` 0 → 1 is accepted if the reset had completed |
| 10 | `SOFT_RESET` | W1 pulse | §8.5 |
| others | | | 0 |

Pulse bits read 0, so a read-modify-write of the level bits never re-triggers them. Each level bit has
exactly one owner, the host. Level bits are updated before the pulses of the same write act.

### 5.3 `STATUS` (0x024)

| Bit | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `RV_RUNNING` | level | `RV_RUN` = 1 and the hart is out of reset |
| 1 | `RV_HALTED` | level | debug halt |
| 2 | `RV_SLEEP` | level | WFI |
| 3 | `RV_CLK_OK` | level | `clk_rv` runs and its domain is out of reset |
| 4 | `RV_RESET_DONE` | level | the RISC-V reset sequence has completed (CC §12.2) |
| 5 | `ACTIVE` | level | TAKEN ≠ ANSWERED |
| 6 | `QIDLE` | level | no entry queued or running, issue FIFO empty |
| 7 | `QQUIESCENT` | level | quiescent state (CC §9.6) |
| 8 | `QERR` | level | some unit is in error state |
| 9 | `SR_BUSY` | level | `SOFT_RESET` in progress |
| 10 | `LCQ_PENDING` | level | LCQ_WR ≠ LCQ_RD |
| 11 | `LOG_PENDING` | level | LOG_WR ≠ LOG_RD |
| 12 | `PLAT_ERR` | level | the platform's error input `plat_err_i` (§9.3) |
| 15:13 | `HOLD_SRC` | level | hold sources: host, RISC-V, debug halt |
| 16 | `LOAD_REFUSED` | sticky | a `LOAD_DATA` write or a VERIFY command while `RV_RUN` = 1 or `SR_BUSY`; a `LOAD_DATA` write, a VERIFY command or a `LOAD_ADDR` write while `VERIFY_BUSY` = 1 (reads never set it) |
| 17 | `LOAD_TIMEOUT` | sticky | a load or VERIFY word timed out |
| 18 | `DMI_TIMEOUT` | sticky | |
| 19 | `QCLEAR_REFUSED` | sticky | |
| 20 | `PUSH_TIMEOUT` | sticky | the RISC-V's issue safety net fired (CC §7.1) |
| 21 | `BRIDGE_TIMEOUT` | sticky | |
| 22 | `LQ_OVERFLOW` | sticky | a push while the LQ was full (descriptor dropped) |
| 23 | `RING_RD_BAD` | sticky | an `LCQ_RD` or `LOG_RD` write was refused |
| 24 | `LOG_DROP` | sticky | log words were dropped |
| 25 | `UNIT_HUNG` | sticky | a unit watchdog fired (§8.6) |
| 26 | `SR_RV_TIMEOUT` | sticky | `SOFT_RESET` went on without the RISC-V reset completing (no `clk_rv`) |
| 27 | `RV_RUN_REFUSED` | sticky | |
| 31:28 | | | 0 |

W1C clears the sticky bits only; writes to the level bits are ignored.

### 5.4 Queue status `Q_*` (0xE00–0xF1C)

These are core-domain values: the same records the RISC-V sees, readable even when the RISC-V is dead.
Unit slots u = 0..7; units that are absent read 0. Different registers may be read at different
moments (they are diagnostic only).

| Offset | Name | Content |
|---|---|---|
| 0xE00 + 4u | `Q_DONE[u]` | `DONE_SEQ[u]`, 28 bits |
| 0xE20 + 4u | `Q_ISSUED[u]` | `ISSUED_SEQ[u]`, 28 bits (synchronized copy) |
| 0xE40 + 4u | `Q_STATE[u]` | [4:0] entries in the queue, bit5 RUNNING (the unit holds an op), bit6 DEP_WAIT (the head waits for deps), bit7 HELD, bit8 ERR (error state), bit9 HUNG (watchdog), [31:16] 0 |
| 0xE60 | `Q_FIFO` | [7:0] issue-FIFO entries waiting for the dispatcher |
| 0xE64 | `Q_PIDX` | push index the dispatcher assigns to its next ticketed entry (= `DBG_NEXT_PIDX`); it lags the RISC-V's `ISSUE_PIDX` (CC §5.1) by the issue-FIFO contents, so firmware never uses it |
| 0xE80 + 16u | `Q_ERR_SEQ[u]`, `Q_ERR_CODE[u]` ([7:0]), `Q_ERR_INFO[u]`, `Q_ERR_W0[u]` | first error record of unit u (§8.2) |
| 0xF00 + 4u | `Q_ERR_PIDX[u]` | its push index |

---

## 6. PERF counters

One counter space has two gates:

- ACTIVE (TAKEN ≠ ANSWERED): a launch is in progress, from its pop to its answer. Because of R6, no
  datapath op runs outside ACTIVE, so PERF does not change between launches. When the host does not
  pipeline launches, the difference of two reads is exactly one launch.
- QHALT (`RV_HALTED` ∧ `QQUIESCENT`): the whole device is stopped by the debugger. Time counters do not
  count then.

Time counters count core cycles with ACTIVE ∧ ¬QHALT. Event counters count with ACTIVE. Counters kept
in the RISC-V domain count `clk_rv` cycles with a synchronized ACTIVE. `SOFT_RESET` zeroes every
counter and drops the snapshot. With C = `NUM_CH` and X = 5 + 2C (four channels: X = 13):

| Index | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `CYCLES` | time | core cycles |
| 1 | `RETIRED` | event | queue entries completed, on every unit and with any outcome (= Σ of `DONE_SEQ` advances) |
| 2 | `MVM_CYCLES` | time | the MVM unit holds an op (from the cycle after accept to done) |
| 3 | `VU_CYCLES` | time | the VU holds an op |
| 4 | | | reserved, reads 0 |
| 5 + c | `CH_RD_BEATS[c]` | event | read beats delivered on channel c, including ch0 |
| 5 + C + c | `CH_WAIT[c]` | time | channel c has a read outstanding and delivers no beat |
| X + 0 | `DMA_CYCLES` | time | the DMA unit holds an op |
| X + 1 | `QACT_CYCLES` | time | the QACT unit holds an op |
| X + 2 | `ATT_CYCLES` | time | unit 4 (AT) holds an op; reads 0 without it |
| X + 3 | `MVM_WAIT` | time | the MVM unit is idle while its queue holds an entry that cannot start (dependencies not done, or the head is held) |
| X + 4 | `MVM_EMPTY` | time | the MVM unit is idle and its queue is empty (issue-bound: the RISC-V has not pushed yet) |
| X + 5 | `HOLD_CYCLES` | time | some queue head is held |
| X + 6 | `PUSHES` | event | entries pushed |
| X + 7 | `PUSH_STALL` | RISC-V time | `clk_rv` cycles a push waited for a credit |
| X + 8 | `BRIDGE_ACC` | event | SPM bridge accesses |
| X + 9 | `BRIDGE_WAIT` | RISC-V time | `clk_rv` cycles bridge accesses waited |
| X + 10 | `REC_DROPPED` | event | completion records dropped |
| X + 11 | `LAUNCHES` | event | answers posted |
| X + 12 | `RV_WFI` | time | the RISC-V sleeps in WFI |
| X + 13 | `VU_SPM_WAIT` | time | a VU SPM request is raised and not granted (VU §5.10) |
| X + 14 | `VU_HAZARD` | time | an accepted VU op waits for an earlier VU op (VU §7.8) |
| X + 15 | `VU_LINES` | event | destination lines written by VU ops |
| X + 16 | `ATT_QK_CYCLES` | time | the MVM unit holds an ATTQK (a subset of `MVM_CYCLES`, AT §10) |
| X + 17 | `ATT_PV_CYCLES` | time | the MVM unit holds an ATTPV |
| X + 18 | `ATT_PATH_STALL` | time | a cluster in an ATT op holds its stream because the gather or the p broadcast is behind |
| X + 19 | `KVST_RMW` | event | meta beats read-modify-written by KVST |
| X + 20 | `SPM_RD_WAIT` | time | client-cycles in which a raised SPM read request waits for its bank port (summed over clients, CC §10.2) |
| X + 21 | `SPM_WR_WAIT` | time | the same for write requests |
| X + 22 … 60 | | | free; unassigned slots read 0 (at `NUM_CH` = 16, X + 21 = 58) |
| 61 | `TRW_RECORDS` | event | trace records written to DRAM (DB §8.1) |
| 62 | `TRW_DROPS` | event | trace records dropped |
| 63 | `DBG_STOP_CYCLES` | time, not gated by QHALT | core cycles with ACTIVE ∧ (`RV_HALTED` ∨ `STOP_ACTIVE`); a measurement is valid only if it is 0 |

The remainder `CYCLES` − `MVM_CYCLES` − `MVM_WAIT` − `MVM_EMPTY` counts the cycles in which the MVM unit
is idle while its head could start: the accept latency, and entries that complete without running. The
RTL asserts that this remainder is at most 3 cycles per MVM-queue entry. The three counters split the
spec's G ([spec](../spec.md) §14) into an issue-bound share and a dependency-bound share. The host is
the only PERF reader. The runtime times launches with MTIME (record r7).

---

## 7. ISA

### 7.1 Instruction format

Instructions stay 256 bits (w0–w7, little-endian). w0 = {x16 = w0[31:16], flags[15:12], sub[11:8],
opcode[7:0]}.

- x16 (w0[31:16]) is an op-specific field; x16 bit i is w0 bit 16 + i, and the area documents name its
  bits x16[15:0]. It must be 0 (`ILLEGAL_OP` otherwise) for every opcode whose definition does not
  assign it. That includes LD, ST, QACT, MVM, NOP and FENCE. VU and AT may assign x16 in the ops they
  define.
- Address fields A1, A2 and A3 are the immediate words themselves: A = imm. There is no base register.
  Range checks follow [spec](../spec.md) §21 (Range check), with a = imm.
- NOP and FENCE use no words: w1–w7 and x16 must be 0.

### 7.2 Global opcode table

| Opcode | Mnemonic | Status | Unit | Defined by |
|---|---|---|---|---|
| 0x00 | `NOP` | SYNC join point (completes when its dependencies are done) | 5 SYNC | spec |
| 0x01 | `END` | reserved: `ILLEGAL_OP` | (SYNC) | |
| 0x02 | `LI` | reserved | (SYNC) | |
| 0x03 | `ADDI` | reserved | (SYNC) | |
| 0x04–0x08 | `MULI`, `RLD`, `LOOP`, `WAITM`, `JUMP` | retired, never reused | (SYNC) | |
| 0x09 | `CALL` | retired, never reused | (SYNC) | |
| 0x0A–0x0F | | reserved: SYNC-class extension | (SYNC) | EX |
| 0x10 | `LD` | defined | 0 DMA | spec |
| 0x11 | `ST` | defined | 0 DMA | spec |
| 0x12 | `DQROW` | SPM → f32 (VU §5.6); simulator only, `CAPS` bit6 = 0 in the release image | 3 VU | VU |
| 0x13 | | reserved for VU (DMA class) | | VU |
| 0x14 | `KVST` | store K or V rows into a KV plane (AT §5.1) | 0 DMA | AT |
| 0x15–0x17 | | reserved for AT (DMA class: KV row stores and loads) | | AT |
| 0x18 | `QACT` | defined | 1 QACT | spec |
| 0x19–0x1B | | reserved for AT (quantize class: q8_0 KV rows) | | AT |
| 0x1C–0x1F | | reserved | | |
| 0x20 | `MVM` | defined | 2 MVM | spec |
| 0x21 | `ATTQK` | scores and row maxima over a KV group range (AT §5.2) | 2 MVM | AT |
| 0x22 | `ATTPV` | P·V channel partials and row sums (AT §5.3) | 2 MVM | AT |
| 0x23 | | reserved for AT (MVM class) | | AT |
| 0x24–0x2F | | reserved (MVM class) | | |
| 0x30–0x33 | `VOP`, `RED`, `ROPE`, `TOPK` | defined by VU (TOPK: simulator only, `CAPS` bit3 = 0 in the release image) | 3 VU | VU |
| 0x34–0x37 | `ASMAX`, `AMASK`, `AMERGE`, `ANORM` | softmax and merge steps; semantics AT §5.4–§5.7, executed by the VU (VU §5.11) | 3 VU | AT |
| 0x38 | `VMASK` | position-window mask for caches where cell j holds position k0 + j (VU §5.7); `ILLEGAL_OP` in the release image | 3 VU | VU |
| 0x39 | `GDN` | fused Gated DeltaNet state update of one head (VU §7.12; `CAPS` bit 8) | 3 VU | VU |
| 0x3A–0x3F | | reserved for VU extensions | 3 VU | VU |
| 0x40 | `SOP` | defined | 3 VU | VU |
| 0x41–0x47 | | reserved for VU (scalar extensions) | 3 VU | VU |
| 0x48–0x6F | | reserved (future units) | | |
| 0x70–0x7E | | reserved: SYNC-class extension | | EX |
| 0x7F | `FENCE` | SYNC barrier (completes when every op pushed before it is done) | 5 SYNC | spec |
| 0x80–0xFF | | reserved | | |

- Every reserved, retired or absent opcode is still accepted by the issue port. It gets a ticket on the
  SYNC unit and completes `ILLEGAL_OP` (CC §7.4), so every push has a completion.
- QCLEAR, QABORT, QHOLD and TRACE_MARK are control-window entries, not opcodes. They stay with CC.
- VU and AT fix the encodings, the units and the dispatch checks of their ops within these numbers. Any
  further number they need is requested from EX and recorded here.
- No op may fail after it has started. Every error is found by the dispatch checks (CC §8). Data-dependent
  conditions, such as NaN or a KV length read from SPM, are defined behavior (clamping, numeric rules),
  not run-time errors. VU and AT design their ops under this rule.

### 7.3 Units

| Id | Unit | Opcodes | Notes |
|---|---|---|---|
| 0 | DMA | LD, ST, KVST | |
| 1 | QACT | QACT | |
| 2 | MVM | MVM, ATTQK, ATTPV | |
| 3 | VU | VOP, RED, ROPE, TOPK, SOP, DQROW, GDN, ASMAX, AMASK, AMERGE, ANORM, VMASK | |
| 4 | ATT | reserved for an AT attention engine; absent (`Q_DEPTH[4]` = 0, `ISSUE_UNITS[4]` = 0) | |
| 5 | SYNC | NOP, FENCE, QCLEAR, refused opcodes | |
| 6–13 | | reserved (for example a second DMA engine) | |
| 14 | | software marker records | |
| 15 | | "no dependency" | |

A dependency on unit 6–14, or on a unit that is absent in the bitstream, is `BAD_DEP`. The opcode → unit
table is static: it is generated from the single table source (EX-30) into the RTL (`UNIT_OF`), the
simulator and the SDK. `ISSUE_UNITS` (control window, CC §5.1) and `Q_DEPTH` (host, §5.1) report which
units a bitstream built. There is no `ISSUE_QMAP` register.

### 7.4 Scalar registers

There is one s0–s15 bank (D14). It is written only by VU-queue ops and read by the RISC-V through the
core window after a join. The one read location is `SREG0`–`SREG15` at core CSR 0x9C0–0x9FC (§5.1):
host offset 0x9C0 + 4i, RISC-V address 0x2000_1000 + 0x9C0 + 4i. Its initial value and its behavior
across `SOFT_RESET` are VU's. EX's requirement: `SOFT_RESET` does not change it, and a hardware reset
sets it to +0.0.

### 7.5 `VERSION`, `CAPS`, `CFG3`

- `VERSION` = 0x0000_0100: the Tesseract Core capability 1.0 ([15:8] major, [7:0] minor, [31:16] 0),
  the compute-capability number of the hardware contract (SW §6.9). The host requires the same major
  and accepts any minor at open; an image names the capability it needs in `.note.tsr` (`cap_min`: same
  major, device minor ≥ its minor) and is refused before loading otherwise. The hardware owner bumps it
  in `sw/tables/tsr_tables.py` (`ISA_MAJOR`, `ISA_MINOR`) in the change that alters the RTL and the
  simulator. Major: an incompatible change of the hardware contract (an instruction encoding, a CSR,
  LQ, LCQ or record field the software relies on, a removed opcode, any tile op's result bits for the
  same inputs, a smaller control-core ISA). Minor: a backward-compatible addition present in every
  configuration of that minor (an opcode or unit, a CSR field, a control-core ISA extension the
  compiler may target). An optional per-configuration feature is a `CAPS` bit, never a minor; the
  `CFG0`–`CFG3` parameters are configuration, not capability. It is shown as "Tesseract Core 1.0
  (VU ATT GDN)".
- `CAPS`: bit0 VU (VOP, RED, ROPE, SOP present), bit1 0, bit2 retired, bit3 TOPK, bit4 retired, bit5
  ATT (the seven AT ops are present), bit6 DQROW, bit7 VMASK, bit8 GDN (the fused state update of
  VU §7.12; the RTL builds with `HAS_GDN`), bits 9–11 reserved for VU, bits 12–15 reserved for AT,
  bits 16–31 0. Retired bits are never set and never reused. The simulator reports what its
  configuration implements, so a simulator may set bits that a bitstream does not. The release image
  reads 0x121 (bits 0, 5 and 8).
- `CFG3`: [4:0] VU front lanes with an EXP core, [9:5] VU back lanes, [15:10] 0 (VU §5.10); [31:16]
  PCAP / 16 (AT §5.10). The VU fields are information for cycle models only. PCAP is a configuration
  parameter like T_MAX: it enters AT's checks and chunk policy (AT §5.8, §6.6), so results are defined
  per configuration, as for every op.

---

## 8. Error model

### 8.1 Op error codes

The CSR fields are 8 bits wide. Only 0–15 are used, so the internal width stays 4 bits
(`rtl/core/tsr_pkg.sv`).

| Code | Name | Condition | `ERR_INFO` |
|---|---|---|---|
| 0 | NONE | | |
| 1 | `ILLEGAL_OP` | reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it | 0 |
| 2 | `BAD_FMT` | unsupported weight format | 0 |
| 3 | `QA_MISMATCH` | reserved, never raised | |
| 4 | `SPM_RANGE` | [spec](../spec.md) §21 (Range check) | start address of the failing operand |
| 5 | `DRAM_RANGE` | as above | start address |
| 6 | `K_RANGE` | K > K_MAX or not a multiple of the block size | 0 |
| 7 | `T_RANGE` | T = 0 or T > T_MAX | 0 |
| 8 | `ABORTED` | the entry completed without running: a QABORT (from the host, the RISC-V, a cancel, a RISC-V reset or `SOFT_RESET`), or its unit was in error state | 0 |
| 9, 10 | | reserved (retired `LOOP_NEST`, `WAIT_TIMEOUT`), never raised | |
| 11 | `DEP_FAILED` | a dependency completed with an error | 0 |
| 12 | `BAD_DEP` | a dependency on a ticket not pushed yet, on unit 6–14 or an absent unit, or below its unit's clear point | 0 |
| 13 | `OPERAND` | an operand value outside the range its op defines, other than K, T and the address ranges (for example n = 0 where that is forbidden, a TOPK k above its limit, an unsupported head dimension); used by VU and AT ops only | the offending value |
| 14, 15 | | reserved | |

Check order at dispatch: `ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` → `T_RANGE` → `OPERAND` →
`SPM_RANGE` → `DRAM_RANGE`. `ERR_W0` is the op's w0.

### 8.2 Op lifecycle, records and propagation

- Poisoning. An op that fails a dispatch check is poisoned: it enters its queue, changes no state, and
  completes with its code at the head.
- At the head, after its dependencies: a poisoned entry completes with its code. Otherwise, if its unit
  is in error state, it completes `ABORTED`. Otherwise, if a dependency failed, it completes
  `DEP_FAILED`. Otherwise it runs.
- Unit error state. The first nonzero completion of a unit records `ERR_SEQ`, `ERR_CODE`, `ERR_INFO`,
  `ERR_W0` and `ERR_PIDX` (RISC-V view, plus `Q_ERR_*` for the host). Every later entry of that unit
  completes `ABORTED` until a QCLEAR.
- I1: every entry completes only at its head and only after its dependencies, so done(t) implies that
  everything t depends on is done. I2: `DONE_SEQ[u]` does not pass `ERR_SEQ` before the record is
  visible to the RISC-V.
- First error = the record with the smallest `ERR_PIDX`. It is independent of the schedule for valid
  programs, and it is what the launch result reports.

### 8.3 Launch-level result

See §4.4: one record per descriptor, with status precedence, ERR_CODE, ticket, `ERR_INFO`, `ERR_W0` and
`ERR_PIDX` of the first error. Platform AXI errors reach the launch result through `plat_err_i` (status
PLAT_ERR) without a host read per launch.

### 8.4 Device-level errors

These are the `STATUS` sticky bits (§5.3), `ADAPTER_STATUS` (§9.2), TRAP records, the watchdog
(`UNIT_HUNG`), image errors (receive or VERIFY CRC mismatch, refused `.note.tsr` requirements, HELLO
with a wrong image id) and a missing `clk_rv` (`RV_CLK_OK` = 0). The transport turns each into an error
of the open, boot, launch or wait call with a decoded message (§10).

### 8.5 Control actions

| Action | Queued entries | Running op | RISC-V | LQ / LCQ / LOG | PERF, error state | TCM, SPM, DRAM |
|---|---|---|---|---|---|---|
| Host QABORT (`CTRL` bit 8, or `LQ_CANCEL` covering the running launch) | complete `ABORTED` in order, after their deps; every unit in error state until a QCLEAR | runs to completion (no cancel) | continues; its join returns, R7 reports ABORTED, R9 clears | unchanged | error state set | unchanged |
| RISC-V QABORT (`QCTRL`) | the same, in push order | the same | continues | unchanged | | unchanged |
| RISC-V QCLEAR | applied after every earlier entry has completed | | continues | unchanged | errors cleared, clear points set | unchanged |
| Host QCLEAR | refused unless QIDLE holds and the RISC-V is quiet | | | | | |
| RISC-V reset (`RV_RUN` 1 → 0, or the DM's ndmreset) | host QABORT at once | completes | reset sequence (CC §12.2) | ANSWERED := TAKEN (the running launch gets no record); queued descriptors stay and are served after the next boot | error state set (cleared by the next boot's R1) | unchanged |
| `SOFT_RESET` (`CTRL` bit 10) | sequence below | completes | reset and left in reset (`RV_RUN` = 0) | flushed: TAKEN := PUSHED, ANSWERED := TAKEN, LCQ_RD := LCQ_WR, LOG_RD := LOG_WR; watermarks invalid | PERF zeroed, error states and records cleared, clear points set, sticky bits cleared | unchanged |
| Debug halt | continue, unless `HOLD_ON_HALT` | completes | halted | unchanged | time counters stop at QHALT | unchanged |
| Hardware reset (xclbin load, card reset) | all state reset | | in reset | counters 0 | 0 | TCM and SPM undefined; DRAM kept unless the platform clears it |

`SOFT_RESET` sequence (hardware, started by the write, which itself completes at once; `SR_BUSY` = 1
throughout):

1. Clear `RV_RUN`, which starts the RISC-V reset sequence (QABORT at once). Clear DB's stop state
   (`S_VALID` and the match queue, DB §5.2) and force `dbg_hold[u]` = 0 for the rest of the sequence;
   DB's configuration (TBPs, `SERIAL`, TRW) and evidence are kept. Wait for `RV_RESET_DONE`, at most
   2^16 core cycles. If `clk_rv` is absent, go on and set `SR_RV_TIMEOUT`.
2. Wait until no datapath unit holds an op and every queue and the issue FIFO have drained. Aborted
   entries drain at one per cycle per unit. This step is bounded by the longest running op. A stuck
   unit never finishes it (§8.6).
3. Apply the host QCLEAR effects and every flush and clear listed in the table above.
4. `SR_BUSY` := 0.

CC implements the whole sequence in a core-domain SR sequencer that needs no `clk_rv` (CC §12.2): step 1
waits at most 2^16 core cycles for `RV_RESET_DONE`, step 2 waits for QIDLE with no unit holding an op,
step 3 applies the clear list in the core domain. The RISC-V-side state (LOG_WR_rv, interrupt levels)
reloads from the core domain when its domain leaves reset.

Kept across `SOFT_RESET`: DRAM, SPM, the TCMs, the s bank, the LQ staging registers, the channel bases
and `MEM_EN`, the DBG configuration (TBPs, `SERIAL`, TRW ring) and DB's evidence (history ring, TRW
pointers, RV snapshot), `RV_BOOT`, `HOST_HOLD`, `HOLD_ON_HALT`, `IRQ_EN`, `LOG_CFG`, `WDOG_CFG`, and
every sequence counter: ticket sequences, push indices, PUSHED, LCQ_WR and LOG_WR. The counters are not
reset, so tickets and launch numbers stay unique across restarts. The debug module follows the hardware
reset only, so a debugger stays attached.

### 8.6 Stuck units, watchdog and recovery

- No forced unit reset. An AXI transaction cannot be canceled. A unit reset with bursts outstanding
  would let stale read responses reach a reused id, and leave write bursts that the interconnect still
  expects ([platform-u200.md](../platform-u200.md) §5). Recovering from a truly stuck unit therefore
  needs a hardware reset (xclbin reload or `xrt-smi reset`).
- Progress watchdog (diagnosis only; it changes no execution). For each datapath unit (DMA, QACT, MVM,
  VU, ATT) a counter runs while the unit holds an op and resets on progress. Progress is any SPM request
  handshake of the unit, any memory request, read beat or write beat of the unit, or its `done`. When the
  counter reaches 2^L (L = `WDOG_CFG[4:0]`, reset value 24 = 16.8 M cycles, 84 ms at 200 MHz), the
  hardware sets `Q_STATE[u].HUNG` and `STATUS.UNIT_HUNG`. The longest legitimate op makes progress on
  every beat, so it never trips.

| Symptom | Detection | Transport action |
|---|---|---|
| Tile-op error | DONE TILE_ERR | report record r3–r6 (`Q_ERR_*` for the other units); the runtime has already cleared |
| Kernel rejected its args | DONE KERNEL_ERR | report RET |
| Bad descriptor | DONE BAD_DESC | host bug: report |
| Platform AXI error | DONE PLAT_ERR / `STATUS.PLAT_ERR` | read, report and clear `ADAPTER_STATUS`; outputs invalid |
| RISC-V trap | TRAP record | drain the log; attach GDB if requested (DB); else `SOFT_RESET` + warm restart (§3.5) |
| Timeout, RISC-V progressing (`RT_INFO` changes) | wait timeout (`TSR_LAUNCH_TIMEOUT_MS`, counted only while not halted or stopped, §4.8) | DB's recovery ladder (DB §7.3): level 0 dump, then level 2 `LQ_CANCEL`; wait `TSR_ABORT_MS` for DONE ABORTED |
| Timeout, RISC-V stuck in a loop or join | `RT_STATE` / `RT_INFO` frozen, no unit RUNNING | ladder level 3: `SOFT_RESET` + warm restart; outstanding launches fail |
| Unit hung | `UNIT_HUNG`, `Q_STATE[u].HUNG` | dump `Q_*`, `DBG_*` and the log; `SOFT_RESET` does not complete; refuse further use: "card reset needed" |
| No `clk_rv` | `RV_CLK_OK` = 0 (`RV_FREQ` = 0) | fail the open (bitstream or clock problem) |

---

## 9. XRT platform adapter (`rtl/platform/alveo_xrt`)

### 9.1 Control protocol: `ap_ctrl_none`

- kernel.xml: `hwControlProtocol="ap_ctrl_none"`, `interrupt="false"`. The only args are `ch<c>` (`void*`
  on `m_axi_ch<c>`) at 0x040 + 8c. The `s_axi_control` range is 0x2000.
- XRT 2.18 facts:
  - `xrt::kernel` refuses `ap_ctrl_none` compute units: "AP_CTRL_NONE is only supported by XRT native API
    xrt::ip".
  - The driver does not touch control or interrupt registers of such units and refuses interrupts for
    them.
  - `xrt::ip` register access is an mmap'ed MMIO access.
  - The buffer of channel c is `xrt::bo(device, size, device_only, mem)`, where `mem` is the index from
    `xrt::xclbin::arg::get_mems()` of arg `ch<c>`.

### 9.2 Register map

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x000–0x03C | | | reserved, read 0 |
| 0x040 + 8c | `ch<c>_base` lo / hi | RW | device byte address of the channel-c buffer; 4 KiB aligned. Written while `MEM_EN` = 0; a write while `MEM_EN` = 1 is ignored and sets `BASE_LOCKED` |
| 0x0E0 | `ADAPTER_CTRL` | RW | bit0 `MEM_EN`. Rising edge: if every base is 4 KiB aligned, latch all bases into the bridges and set `MEM_EN`; otherwise stay 0 and set `ADAPTER_STATUS` bit3. While `MEM_EN` = 0, core memory requests are held (not forwarded, not dropped) and set bit4 |
| 0x0F0 | `ADAPTER_STATUS` | R / W1C sticky | bits 0–2 and [31:16] as in [platform-u200.md](../platform-u200.md) §2; bit3 means "base misaligned at `MEM_EN`"; bit4 `REQ_HELD`, bit5 `BASE_LOCKED` |
| 0x0F4 | `ADAPTER_VERSION` | R | {16'h0001, 8'h00, NUM_CH}: adapter major 1 (the host requires it) |
| 0x1000 + off | core CSR window | | off < 0x1000, any time, writes need WSTRB = 0xF |

The transport clears `MEM_EN` only when the core is idle (after `SOFT_RESET`). Responses of transactions
already issued pass either way; the data beats of a write burst whose request was accepted are held,
like a request, if `MEM_EN` falls before the burst is complete (which the idle rule excludes). The
memory bridges and the reset scheme are in [platform-u200.md](../platform-u200.md) §4 and §5.

### 9.3 Core boundary ([spec](../spec.md), "Core boundary interface")

The core has no `done` output. It has `clk_rv` and `rst_rv_n` (CC); `plat_err_i` (a level from the
platform: on the U200, `ADAPTER_STATUS` bits 0–2 or the error count ≠ 0; it shows in
`STATUS.PLAT_ERR` and in `LQ_FLAGS`, and is tied to 0 where there is no adapter); and `host_irq_o`
(`CTRL.IRQ_EN` ∧ LCQ_WR ≠ LCQ_RD, for ports with a host interrupt such as Zynq; not connected on the
U200).

---

## 10. Transport API (`sw/transport`)

### 10.1 C API (`tsr_transport.h`)

The API is `tsr_transport_open(kind)` (`SOFT_RESET` and bind, unless `TSR_ATTACH=1`), `_close`, `_name`,
`_config`, `_write` and `_read` (per-channel DRAM), `_csr_read` and `_csr_write`; `tsr_image_open` (checks the
segments and `.note.tsr`), `_close`, `_kernel` (index or -1), `_id` (64-bit); `tsr_transport_boot(t, img)` (warm
if possible, else cold; HELLO checked); `tsr_transport_launch(t, {type, flags, kernel, arg[7]}, &seq)` (blocks
only at `LQ_DEPTH` outstanding); `tsr_transport_wait(t, seq, timeout_ns, &rec)` (0: answered; negative:
transport failure or timeout, diagnosed per §8.6; the timeout counts only while the RISC-V is not halted and no
tile stop is active, §4.8); `_call` (launch and wait), `_cancel(seq_upto)`, `_stop(seq_upto)`,
`_set_event_cb`, `_soft_reset` (`SOFT_RESET`, wait `SR_BUSY` = 0) and `_log_drain`. A `tsr_record` is eight words
(§4.4; accessors in `tsr_isa.h`). This API is the only transport API; SW §9.1 adds convenience functions on
top of it (default image, statistics), not a second protocol.

### 10.2 Device-access layer (`tsr_kdev.h`)

- A kdev has `reg_read` and `reg_write` (`s_axi_control` offsets), `mem_write` and `mem_read`,
  `num_ch`, `ch_bytes`, `ch_addr` and `describe`. `KREG_ADAPTER_CTRL` (0x0E0) names the adapter control
  register. An optional `advance(ns)` hint lets the sim and vkernel kinds run device time while the host
  polls.
- One protocol layer (`tsr_kdev.cpp`) implements open, bind, load, boot, launch, wait, cancel, stop,
  reset and log for every kind:
  - `sim`: an in-process simulator exposing the core CSR port, with no adapter. A host read of
    `LCQ_WR`, `STATUS` or `LOG_WR` first runs the model until a record or log word appears, or until the
    RISC-V waits for host input (WFI with `LQ_AVAIL` = 0 and no op running). Results are therefore
    deterministic and independent of host timing.
  - `mock`: sim plus the adapter register model.
  - `vkernel`: the Verilated kernel with two clocks.
  - `xrt`: `xrt::ip` plus per-channel BOs.
- Environment: `TSR_DEVICE`, `TSR_XCLBIN`, `TSR_XRT_DEVICE`, `TSR_XRT_KERNEL`, `TSR_XRT_CH_GIB`,
  `TSR_XRT_DMA*`, `TSR_MOCK_*`, `TSR_LAUNCH_TIMEOUT_MS` (default 60,000), `TSR_ABORT_MS` (10,000),
  `TSR_IMAGE` (firmware ELF; default the build's base image), `TSR_BOOT_MS` (2,000), `TSR_RESET_MS`
  (15,000, the `SOFT_RESET` wait), `TSR_ATTACH` (open without reset or load, for diagnosis).

---

## 11. DRAM map (per channel; [spec](../spec.md) §4)

| Region | Range | Use |
|---|---|---|
| OPL | 0–64 MiB, ch0 | op lists, `replay` push streams, argument blocks, RISC-V code overlays |
| IO | 64–320 MiB, ch0 | host to device tensors (inputs, logits, token rings); compute chunks (SW R6) |
| TRW ring (profiling only) | inside IO, ch0 | DB's trace-writer ring (`TSR_PROF_TRW_MIB`, default 64 MiB), allocated by the transport only while profiling; `TRW_CH` = 0 (DB §8.3) |
| AUX | 320–512 MiB, ch0 | small f32 tensors (norm weights, RoPE frequency-factor tables) |
| WEIGHT and heap | 512 MiB to the end, every channel | weights (symmetric allocation, bottom-up from 512 MiB), KV planes (symmetric, top-down from the channel end, AT §4.5), and between them the per-channel compute heap (SW R6, SW-26), sized from the reserve graph |
| low 512 MiB of ch1–3 | | compute chunks and other activations (SW R6); software may spread IO over the channels |

The region constants are `TSR_REGION_OPL_*` (`rtl/core/tsr_pkg.sv`). The core does not fetch from DRAM:
memory source 0 (`TSR_SRC_AUX`) is tied off, and the source ids are not renumbered.

---

## 12. Build order

The control core is part of every core; there is no compatibility layer and no `HAS_CTRL` parameter. The layers
are built in this order, each checked against the one before it: the oracle (`sw/sim`: generated tables,
op-level API, CSR model, rings and PERF, `SOFT_RESET`, RV32 interpreter, queue model, VU and AT ops), traces
(SW §5.6), host software (transport, firmware SDK and base image, op list builder, `tsr-selftest`), core RTL
(`tsr_chk`, dispatcher and queues, CSR block, control core, VU and AT, with tile-only replay benches) and the
platform (adapter, `ap_clk_2`, the two-clock vkernel, `hw/u200` floorplans). The oracle comes first because
everything is checked against it, so the whole software path runs end to end on the simulator before any RTL
exists (D16).

---

## 13. Decisions

| Id | Decision | Reason |
|---|---|---|
| EX-1 | Launch transport: a core-domain launch queue LQ (depth 8) written by posted CSR writes, plus a completion ring LCQ (depth 8) read through the CSR window | Posted writes are the cheapest PCIe operation and a launch needs no DMA; depth 8 lets the host pipeline launches |
| EX-2 | LQ and LCQ carry every host and RISC-V message. PING, HELLO, TRAP and kernel events are record types | One channel halves the CDC paths, deadlock arguments and tests |
| EX-3 | Descriptor and record formats: 8-word descriptor {TYPE, PROF, KERNEL, ARG0–ARG6}; 8-word records whose r0 = {type, status, first error code, seq[7:0]} | Seven args cover the known kernels; one PCIe read classifies a completion |
| EX-4 | Launch ordering: exactly one answer per descriptor, in order; hardware sequence numbers; `LQ_POP` refused while ACTIVE | Sequential launches make ACTIVE exact for PERF, cancel and the current launch; overlap comes from the async issue path |
| EX-5 | Flow control: the host keeps at most `LQ_DEPTH` launches outstanding; `LCQ_DEPTH` = `LQ_DEPTH` = 8 | Neither ring can overflow and the push path needs no read |
| EX-6 | XRT control: `ap_ctrl_none`, `xrt::ip` register access, polling; no interrupts on the U200; `host_irq_o` for other targets | A halted RISC-V under an outstanding `xrt::run` breaks XRT's scheduler; XRT 2.18 routes AP_CTRL_NONE to `xrt::ip` |
| EX-7 | Channel bases are latched and alignment-checked on the rising edge of `ADAPTER_CTRL.MEM_EN` and locked while set; requests are held while `MEM_EN` = 0 | There is no run start to latch on; one edge gives an atomic, checked rebinding |
| EX-8 | Firmware load path: CSR loader with a receive CRC plus an on-device VERIFY read-back CRC | About 10 ms per load; VERIFY checks the TCM contents without 32 K PCIe reads |
| EX-9 | Image identity: the 64-bit prefix of a SHA-256 over the `PT_LOAD` segments, stored in `.tsr_id` and echoed in HELLO | Identifies the exact build cheaply; integrity is the CRC's job |
| EX-10 | Warm restart: `SOFT_RESET`, reload the DTCM segments, VERIFY ITCM (reload only on a mismatch), boot | Avoids stale `.data` at about a tenth of the cost of a cold load; ITCM is writable, so it is checked |
| EX-11 | One host CSR map (§5): CTRL and STATUS at 0x020 and 0x024, DBG at 0x800–0xAFC, 12-bit space; CC, DB, VU and AT place registers inside it | A second control block would only duplicate; the map fits 4 KiB |
| EX-12 | Identity and version: ID 'TSR1'; VERSION 0x0000_0100 = Tesseract Core capability 1.0; `CTL_ID[7:0]` 1 (0 = no control core); `ADAPTER_VERSION` major 1 (SW-54) | The host open code refuses a different capability major and `CTL_ID` = 0, so no new magic is needed |
| EX-13 | PERF semantics: one counter space gated by ACTIVE (time counters also by ¬QHALT), zeroed by `SOFT_RESET`; `MVM_WAIT` and `MVM_EMPTY`; one slot table with VU X+13..15, AT X+16..19, CC X+20..21, DB 61–63 | PERF is unchanged between launches; MVM idle time splits into issue-bound and dependency-bound |
| EX-14 | x16 field: w0[31:16] is the op-specific field (x16 bit i = w0 bit 16+i), 0 for every opcode that does not assign it | VU and AT get 16 bits of encoding space |
| EX-15 | Opcode allocation: blocks beside each class (VU 0x13, 0x38–0x3F, 0x41–0x47; AT 0x14–0x17, 0x19–0x1B, 0x21–0x23, 0x34–0x37; SYNC 0x0A–0x0F, 0x70–0x7E); retired opcodes are never reused | Areas work without collisions; class decode stays cheap |
| EX-16 | Unit id 4 is reserved for an AT attention engine and absent; AT's ops run on units 0, 2 and 3 | Keeps unit ids dense and the six-unit CDC widths |
| EX-17 | Error codes: 13 `OPERAND` (`ERR_INFO` = the value); check order `ILLEGAL_OP`, `BAD_DEP`, `BAD_FMT`, `K_RANGE`, `T_RANGE`, `OPERAND`, `SPM_RANGE`, `DRAM_RANGE`; 9, 10, 14, 15 reserved | VU and AT value limits are neither encoding nor K errors; the value makes failures diagnosable |
| EX-18 | Launch-level result: OK, TILE_ERR, BAD_DESC, KERNEL_ERR, TRAP, ABORTED, PLAT_ERR, STOPPED with fixed precedence; first error = smallest `ERR_PIDX` | Tile failures never read as success; the selection is schedule-independent |
| EX-19 | Platform AXI errors: `plat_err_i` core input, mirrored in STATUS and reflected in the launch status | Adapter errors reach the launch status without a PCIe read per launch |
| EX-20 | Cancel and stop are sequence watermarks: `LQ_CANCEL` (hard: QABORT of a covered running launch), `LQ_STOP` (soft: polled by kernels); covered queued launches are answered ABORTED | Precise and race-free; also gives a graceful stop |
| EX-21 | RISC-V traps: the handler follows DB's trap contract, posts TRAP and parks for GDB; recovery is `SOFT_RESET` plus a warm restart | A trap may leave `.data` inconsistent; parking keeps the state for debugging (D18) |
| EX-22 | `SOFT_RESET` is a hardware sequence (CC's SR sequencer): RISC-V reset, drain, clear error states, flush LQ, LCQ and log, zero PERF, clear DB's stop state; it works without `clk_rv`; `SR_BUSY` shows progress | One atomic known-state operation that the RISC-V cannot race and that survives a missing `clk_rv` |
| EX-23 | Stuck units: no forced unit reset; a per-unit progress watchdog (2^L cycles, 84 ms at 200 MHz by default) sets `UNIT_HUNG`; recovery is a card reset | AXI has no cancel; killing a unit would corrupt later transactions |
| EX-24 | Debug log: a 256-word core-domain ring read through the CSR window; whole records dropped and counted, optional BLOCK; DB's record header; `LOG_CFG.LEVEL` filter | Works when queues wedge; never faults or stalls by default |
| EX-25 | Transport structure: one register-level protocol layer for sim, mock, vkernel and xrt; sim advances deterministically on host reads | The protocol itself becomes part of what the oracle checks |
| EX-26 | DRAM map: OPL region (op lists, replay streams, argument blocks, overlays); TRW ring in the ch0 IO region; compute heap between weights and KV planes | The name follows the content |
| EX-27 | Memory source 0 (`TSR_SRC_AUX`) is tied off; source ids are not renumbered | Avoids churn on timing-critical arbiter paths |
| EX-28 | Build order: oracle, traces, host, RTL, platform (§12) | The software path runs end to end on the simulator before any RTL (D16); the platform needs the control core to be useful |
| EX-29 | Offload path: the `graph` executor runs op lists, one launch per graph split in M6 mode; the `replay` kernel (SERIAL) serves traces and SPM dumps; there is no `tsr.oplist_seq` | One mechanism from M6 to M7 and no second offload path of about 1 µs per op |
| EX-30 | Table source: `sw/tables/tsr_tables.py` (plain Python) with SW's `gen_tables.py` emitting the SV package, the C header, `mmio/vu/at.hpp` and tb Python; `--check` in lint | One source for RTL, simulator, firmware and benches, with no PyYAML dependency |

---

## 14. Interfaces with other areas

| Area | EX provides |
|---|---|
| VU | Opcodes 0x12, 0x13, 0x30–0x33, 0x38, 0x39, 0x3A–0x3F, 0x40, 0x41–0x47; unit 3; the `x16` field; error code 13 and the check order; `CAPS` bits 0, 3, 6, 7, 8 and 9–11; `CFG3[15:0]`; PERF indices 3 and X + 13 … X + 15; one s-register bank read at `SREG` 0x9C0. VU has no run-time errors (§7.2), defines the s-bank initial value, and executes AT's vector ops 0x34–0x37 (VU §5.11) |
| AT | Opcodes 0x14–0x17, 0x19–0x1B, 0x21–0x23, 0x34–0x37; unit id 4 (absent); `CAPS` bits 5 and 12–15; `CFG3[31:16]`; PERF X + 16 … X + 19; the DRAM heap for the KV cache (§11). AT checks its ops fully at dispatch and clamps data-dependent lengths |
| CC | The static opcode → unit table; the RISC-V-side LQ, LCQ, LOG and RT registers at control window 0x200–0x2FC (§4.5) and `irq_fast[2]` and `[3]`; the runtime contract R1–R10; `RV_BOOT`, `RV_FREQ`, `RV_CFG`, `CTL_ID` and `Q_DEPTH` offsets; the `SOFT_RESET`, RISC-V reset and QABORT semantics (§8.5); ACTIVE and the PERF gates; PERF X + 20 and X + 21. CC implements the ring CDC by tagged handshakes (§4.1), the SR sequencer, the issue path, tickets, error records, I1 and I2, `ISSUE_PIDX`, the loader's TCM write and VERIFY read port |
| DB | CSR range 0x800–0xAFC for `DBG_*`; DMI 0xC00–0xDFC; `HOLD_ON_HALT`; QHALT gating; `Q_*` and `RT_*` for post-mortem; the log ring with DB's record header; PERF 61–63; `tsr_transport_csr_write`; the in-process ownership rule (§2.4); the debug-aware wait timeout (§4.8). DB defines `DBG_*` contents, the GDB RSP server, the halt rules, the SPM access rule through SBA and the trap contract |
| SW | The transport API and kdev layer (§10); descriptor and record formats; image format, image id and load procedure; base-kernel contracts (§4.6); the DRAM map; the build order (§12); the table source format (EX-30). SW provides the SDK runtime implementing R1–R10, `tsr-elf`, the base image, the simulator models of every EX register and ring, the OPL format, the trace format, `tsr-selftest` and the table generator |

---

## 15. Verification

The simulator is the oracle: it is bit-exact for everything EX defines, and the RTL and the card must match it.

1. Generated tables: `gen_tables.py --check` in lint, and a directed test that reads every core CSR offset
   after a scripted sequence on sim, mock and vkernel; the values must be identical except for the
   timing-dependent set (`PERF` time counters, `RV_FREQ`, `RV_SLEEP`, record word r7, log payloads marked as
   timing).
2. Protocol conformance (`sw/transport/tests`, one binary, every kind): cold and warm boot, HELLO, image-id
   mismatch, VERIFY with an injected TCM word flip, `.note.tsr` refusal; PING, LAUNCH with 0–7 args, BAD_DESC,
   8 outstanding plus the credit rule, forced `LQ_OVERFLOW` and `RING_RD_BAD`; kernel events, log framing, drop
   and BLOCK; cancel, stop, `SOFT_RESET` and RISC-V reset mid-launch. Records r0–r6, DRAM and the log streams
   must be identical across sim, mock and vkernel.
3. Error model: an `errgen` kernel pushes each error code (1–8, 11–13) and the propagation cases (error state,
   `DEP_FAILED` chains, first error by push index across two units); DONE records and `Q_ERR_*` must equal the
   simulator's under 8 or more random schedules. The `ops` trace corpus (SW §5.6) replays through `replay`
   with SERIAL on sim and vkernel with equal DONE records, error records and expect regions.
4. PERF: sim equals RTL exactly for `RETIRED`, `CH_RD_BEATS[c]`, `PUSHES` and `LAUNCHES`; the RTL asserts the
   bound on the MVM remainder of §6 (`CYCLES` − `MVM_CYCLES` − `MVM_WAIT` − `MVM_EMPTY` ≤ 3 per MVM-queue
   entry); time counters are not compared; the counters must not change while ACTIVE = 0.
5. RTL directed tests (cocotb and Verilator, two clocks at every ratio of CC §15): every CSR access completes
   with `clk_rv` stopped or held in reset; `SOFT_RESET` without `clk_rv`; the cancel watermark at the exact
   ACTIVE edges; the `LQ_POP` and `LCQ_POST` faults; the watchdog; the adapter (`MEM_EN` latch, misaligned
   refusal, `BASE_LOCKED`, `REQ_HELD`, `plat_err_i`); and a mutation campaign (answer without join, flush
   omitted, watermark compare off by one, ACTIVE from the wrong counter, PERF gate inverted, `mutate_adapter.py`).
6. Card: `tsr-selftest` (identity, bind, load and VERIFY, PING, launch round trip and pipelined rate against
   the §4.11 targets, the `selftest` kernel's tests, the trace corpus through `replay`), then llama.cpp end to
   end in M6 mode (SW §10.4): logits bit-identical to `TSR_DEVICE=sim` and the KL criterion.

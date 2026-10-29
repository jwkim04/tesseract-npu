# Tesseract v2 — execution model, host interface and ISA changes (area EX)

Status: **design, not implemented** (2026-10-30). Area EX of the M6 design set in `docs/v2/`. It
owns the host-visible model and the global tables the other areas use: the launch protocol, the core
CSR map, the opcode and unit tables, the error codes, the PERF counters, the XRT adapter and the
transport API. It applies the user decisions D14–D18 (`docs/ctrl-riscv.md:38-56`) and keeps D1–D6
and D12. When adopted, it replaces these v1 texts: spec "실행 모델과 제어 인터페이스"
(`docs/spec.md:323-392`), the retired rows of the ISA table (`docs/spec.md:418-444`), "명령 버퍼
경로와의 공존" (`docs/spec.md:760-768`), and ctrl-riscv §3.3, §3.4, §6, §8.3 (launch protocol) and
§8.4 (`docs/ctrl-riscv.md:326-386`, `:850-925`, `:1160-1234`). Every unmeasured number is marked
*estimate*. Every third-party behaviour that must be checked when a dependency is pinned is marked
*confirm*. Area names: EX (this document), VU (vector unit), AT (attention and KV), CC (control
core), DB (debug), SW (software and verification).

---

## 1. Summary

- The host never runs programs. It binds channel memory, loads the firmware image into the
  RISC-V's TCMs through a CSR loader (verified by an on-device CRC), starts the RISC-V and
  **launches kernels**. There is no START, no command buffer, no `tsr_cmdproc` sequencer, no
  `tsr_fetch`, no R0–R15, LI, ADDI, END or CALL (D14). The host has no access to the issue port
  (D15).
- A launch is an 8-word descriptor written by **posted CSR writes** into a core-domain **launch
  queue** (LQ, depth 8). Each descriptor gets exactly one 8-word answer record, in order, in a
  core-domain **completion ring** (LCQ, depth 8) that the host reads through the CSR window. The
  host needs no PCIe DMA per launch and two PCIe reads per completion. The empty-launch round
  trip target is ≤ 10 µs (*estimate* 4–8 µs). v1 measured 35 µs per exec
  (`docs/platform-u200.md:1049`).
- A 256-word **log ring** in CSR space carries the RISC-V's log records (header format: DB §6.1)
  to the host. It still works when every tile queue is wedged.
- The kernel uses **`ap_ctrl_none`**. The host uses `xrt::ip` register access and polls. No
  `xrt::run` is ever outstanding, so a halted RISC-V cannot break XRT's scheduler. The channel
  bases are latched when `ADAPTER_CTRL.MEM_EN` rises.
- **One CSR map** (§5). Device control moves to `CTRL` / `STATUS` at 0x020 / 0x024. The planned
  `CTL_*` block at 0xB00 is dissolved. `CTL_PERF` merges into `PERF_*`. The PERF counters count
  only while a launch is in progress (`ACTIVE`, a hardware-defined condition).
- **ISA v2** (§7). LD, ST, QACT, MVM, NOP and FENCE keep their encodings bit for bit. The VU ops
  and DQROW keep their numbers and are re-specified by VU. Opcode blocks are reserved for AT and VU.
  Retired opcodes are never reused. The register fields w0[31:16] become an op-specific field
  `x16`, which must be 0 for every v1-encoded op. Unit 4 (was CMD) is reserved for a future AT
  attention engine; in M6 AT's ops run on the DMA, MVM and VU units and unit 4 is absent.
- **Error model** (§8). Error codes 1–12 keep their meaning, 13 `OPERAND` is new. Per-unit error
  records and the invariants I1 and I2 of ctrl-riscv are kept. Each launch has one result with
  defined status precedence. `SOFT_RESET` becomes a device-level reset sequence that works without
  `clk_rv`. A cancel and a stop watermark replace the v1 ABORT. A stuck unit is detected by a
  progress watchdog and is recovered by a card reset only.

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
| Data | Write weights, inputs and op lists, and read outputs, by PCIe DMA | unchanged from v1 (`docs/xrt-host.md` §4.2) |
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
| FAILED | no clock (`RV_CLK_OK` = 0), image mismatch, adapter refused | close | — |

### 2.3 Ordering between the host and the device

These replace memory-model rule 5 (`docs/spec.md:756`).

- **H1.** DRAM that the host wrote with a completed transfer before it wrote `LQ_PUSH` for launch
  s is seen by every op pushed while s, or a later launch, runs. "Completed" means the DMA call
  returned. This is the same guarantee v1 relied on for an exec.
- **H2.** DRAM written by ops of launch s is seen by the host after it read the answer record of
  s. The runtime joins every op of s before it answers (§4.7 R6). A join means `done`, which
  includes the DRAM write responses.
- **H3.** A host access to DRAM that an outstanding launch reads or writes is a race (undefined).
  The exception is a region that a kernel's own protocol defines, for example an M9 token ring.
- **H4.** SPM is invisible to the host (DB's debugger reads it through the debug module's system
  bus, DB §3.6). The hardware never
  changes SPM between launches, and `SOFT_RESET` does not change it either. SPM is undefined only
  after a hardware reset. Kernels may keep SPM state across launches; the runtime (SW) owns its
  allocation.
- **H5.** Descriptor words are ordered by PCIe posted-write order and by the AXI4-Lite write
  channel. The adapter handles writes in arrival order (`docs/platform-u200.md:83-85`), so
  `LQ_PUSH` always sees the staging words written before it.

### 2.4 Ownership

One process owns the device: `xrt::ip` opens an exclusive compute-unit context by default
(`build/xrt-driver/xrt-202420/src/runtime_src/core/common/api/xrt_ip.cpp:131`, `:178`). Tools that
must observe a running process (GDB, profiling) run inside it as threads of the transport. DB
designs the in-process GDB RSP server. The transport serialises CSR accesses with one mutex, and
XRT adds its own (`.../core/pcie/linux/shim.cpp:2343`).

---

## 3. Firmware image

### 3.1 Format

A static RV32 ELF (no relocations), built by the SDK of SW / CC (`docs/ctrl-riscv.md:1068-1088`).

- `PT_LOAD` segments lie inside ITCM [0x0000_0000, `ITCM_BYTES`) or DTCM [0x0010_0000,
  0x0010_0000 + `DTCM_BYTES`) of the RISC-V map (`docs/ctrl-riscv.md:260-267`). Their physical
  addresses are word aligned and they do not overlap. The host pads `p_filesz` with zeros up to a
  whole word. The `.bss` part (`p_memsz` − `p_filesz`) is not loaded: crt0 zeroes it.
- `.tsr_kernels`: the kernel table. The **kernel index** in a descriptor is the position in this
  table (0-based). SW defines the entry format (`docs/ctrl-riscv.md:1165-1167`: name hash, entry,
  nargs, flags with bit0 = argument block, name).
- `.note.tsr`: SDK ABI version, required control-core version (`CTL_ID[7:0]`), minimum ITCM and
  DTCM sizes, required `CAPS` bits. The host refuses an image whose requirements the device does
  not meet, before loading it.
- `.tsr_id`: 8 bytes in a DTCM `PT_LOAD` segment, holding the image id (§3.2).

### 3.2 Image id

The image id is a 64-bit value defined as follows:

1. Order the `PT_LOAD` segments by ascending physical address.
2. Hash with SHA-256, for each segment in that order: its physical address as a 32-bit
   little-endian word, its padded file length as a 32-bit little-endian word, then the padded
   bytes. The 8 bytes of `.tsr_id` are taken as zero.
3. The image id is the first 8 bytes of the digest, read as a little-endian `uint64`.

`tsr-elf` (SW) writes the id into `.tsr_id` after linking. The runtime echoes it in HELLO (r2 =
bits [31:0], r3 = bits [63:32]), and the host compares it with its own computation. The id
replaces ctrl-riscv's full SHA-256 echo (`docs/ctrl-riscv.md:1168-1169`). 64 bits are enough to
tell builds apart, and the integrity check is the CRC (§3.3).

### 3.3 Cold load

Preconditions: `RV_RUN` = 0 and `STATUS.SR_BUSY` = 0. Otherwise every `LOAD_DATA` write and
every VERIFY command is refused (`STATUS.LOAD_REFUSED`), and `LOAD_DATA` reads return 0.

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

- **CRC-32** is zlib's `crc32` (reflected polynomial 0xEDB88320, initial value 0xFFFFFFFF, final
  XOR 0xFFFFFFFF). It is computed over the bytes of each word in address order (little-endian).
  A write to `LOAD_ADDR` restarts the CRC, so `LOAD_CRC` reads 0x0000_0000 (the CRC of zero
  bytes).
- **VERIFY**: a core-domain sequencer reads `LOAD_COUNT` words from `LOAD_ADDR` through the load
  port and computes `LOAD_CRC` over them. Each word read is a tagged request bounded by 4,096 core
  cycles. On a timeout, `LOAD_TIMEOUT` is set and the sequencer stops with `VERIFY_BUSY` = 0, so
  the CRC is then wrong. VERIFY catches broken TCM writes and address decode faults that the
  receive CRC cannot see, without any PCIe reads.
- `LOAD_DATA` writes keep the CDC rule of ctrl-riscv (`docs/ctrl-riscv.md:340`): each write goes
  into a core-side holding register, and a write that cannot complete within 4,096 core cycles is
  dropped with `LOAD_TIMEOUT` set. So the CSR port never hangs.
- **Time** (*estimate*): 128 KiB is 32 K posted writes. The AXI4-Lite slave accepts about one word
  per 15–25 core cycles (window write 10 cycles plus the holding-register handshake), so 2–4 ms,
  plus the host posting rate. VERIFY of 128 KiB takes 0.6–2 ms. Total ≤ 10 ms per cold load.

### 3.4 Boot

`CTRL.RV_RUN` := 1. The RISC-V leaves reset and fetches at `RV_BOOT` + 0x80
(`docs/ctrl-riscv.md:1014-1015`). crt0 and the runtime boot (§4.7 R1) end with a HELLO event. The
host consumes it within `TSR_BOOT_MS` (default 2,000 ms) and checks four things: the image id,
the SDK ABI version, the kernel count, and that no other record came first.

### 3.5 Warm restart

A warm restart is used after a TRAP, a hang of the RISC-V, or a new process that finds the same
image already loaded. Steps: `SOFT_RESET` (§8.5), reload every DTCM segment (`.rodata` and
`.data`: a few KiB, *estimate*), VERIFY every ITCM segment (no ITCM writes when the CRC matches,
otherwise a cold load of that segment), then boot. ITCM is writable by the RISC-V (software
breakpoints, overlays), which is why it is verified and not assumed to be intact. Time ≤ 2 ms
(*estimate*). The transport uses a warm restart when `LOAD_CRC` after VERIFY matches for every
ITCM segment.

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
tagged handshakes into the core domain (CC §11.2, V2-CC-30), never through gray-coded copies of
the counters, because the host-written counters (`LCQ_RD`, `LOG_RD`) and the flushes of
`SOFT_RESET` move them by more than one step at a time:

- every RISC-V read of `LQ_HEAD_W*`, `LQ_AVAIL`, `LQ_HEAD_SEQ`, `LQ_CANCEL`, `LQ_STOP`,
  `LQ_FLAGS`, `LCQ_FREE` and `LOG_CFG` returns the core-domain value at the moment the core
  answers the handshake;
- `LQ_POP` and `LCQ_POST` are evaluated in the core domain at commit (the empty and full checks
  included), and the store completes after the core domain has changed;
- `LOG_PUSH` words cross through a small CDC FIFO. `LOG_FREE` = `LOG_DEPTH` − (LOG_WR_rv −
  LOG_RD_snap), where LOG_WR_rv counts accepted `LOG_PUSH` stores in `clk_rv` and LOG_RD_snap is
  the core-domain `LOG_RD` fetched by the same handshake as the `LOG_FREE` read;
- the interrupt levels `irq_fast[2]` / `[3]` (§4.5) are synchronised core-domain booleans. A late
  or stale level only causes a late or spurious wake-up, after which the runtime re-reads the
  register.

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

- The **sequence number** of a descriptor is the value of PUSHED when it was pushed. The host
  learns it without reading: it reads `LQ_SEQ` once at open and counts its own pushes after that.
- **ACTIVE** := TAKEN ≠ ANSWERED. The hardware keeps TAKEN − ANSWERED ∈ {0, 1}: an `LQ_POP` while
  ACTIVE is refused with an access fault. While ACTIVE, the running launch's sequence number is
  ANSWERED.
- PUSHED − TAKEN ≤ `LQ_DEPTH`. A push when the ring is full is dropped and sets
  `STATUS.LQ_OVERFLOW`. The transport keeps at most `LQ_DEPTH` launches outstanding (pushed and
  not yet answered), which makes an overflow impossible without any read (§4.8).
- LCQ_WR − LCQ_RD ≤ `LCQ_DEPTH`. Record k lives at core CSR 0x300 + 32 × (k mod `LCQ_DEPTH`) until
  the host consumes it. A host write to `LCQ_RD` of a value v is accepted iff
  (v − LCQ_RD) mod 2^32 ≤ (LCQ_WR − LCQ_RD) mod 2^32. Otherwise it is ignored and
  `STATUS.RING_RD_BAD` is set. `LOG_RD` follows the same rule.
- U200 parameters: `LQ_DEPTH` = `LCQ_DEPTH` = 8, `LOG_DEPTH` = 256 words. They are readable in
  `LQ_CFG` (0x07C). The CSR map limits `LCQ_DEPTH` to 8 and `LOG_DEPTH` to 256.

### 4.3 Descriptor (host → RISC-V), 8 words

| Word | Bits | Field |
|---|---|---|
| w0 | [7:0] | TYPE: 0x01 LAUNCH, 0x02 PING; every other value is reserved |
| w0 | [8] | PROF: a profiled launch. The runtime sets `REC_CFG.ALL` for the launch (completion records for firmware self-profiling, DB §8.2) and kernels emit NODE markers (the graph executor behaves as if the op-list flag MARK were set, SW §7.2). The host timeline itself comes from DB's hardware trace writer (DB §8.3), which the host enables separately |
| w0 | [15:9] | reserved, must be 0 |
| w0 | [31:16] | KERNEL: index into `.tsr_kernels` (LAUNCH); must be 0 for PING |
| w1–w7 | [31:0] | ARG0–ARG6: integers, SPM line numbers, or DRAM handles {ch[31:28], beat[27:0]} (`docs/ctrl-riscv.md:1096`) |

A kernel whose table entry has the argument-block flag gets ARG0 = the DRAM handle of a block of
at most 1 KiB. The runtime stages that block into the SPM argument range before the call
(`docs/ctrl-riscv.md:1171-1176`). The descriptor is answered with BAD_DESC when any of these
holds: an unknown TYPE, reserved bits set, KERNEL beyond the table, or an argument-block handle
with ch ≥ `NUM_CH` or a block that does not fit inside `CH_BYTES`.

### 4.4 Record (RISC-V → host), 8 words

r0 has one layout for every record type: [7:0] TYPE, [15:8] STATUS, [23:16] ERR_CODE, [31:24]
SEQ[7:0]. With r0 alone the host can classify a record and match it to a launch.

| Word | DONE (0x01, answer) | PONG (0x02, answer) | HELLO (0x80, event) | TRAP (0x81, answer if ACTIVE, else event) | kernel event (0xC0–0xFF) |
|---|---|---|---|---|---|
| r0 | type, status, first error code, seq[7:0] | type, 0, 0, seq[7:0] | type, 0, 0, 0 | type, 4, 0, seq[7:0] | type, kernel-defined [23:8], seq[7:0] |
| r1 | SEQ of the answered descriptor | SEQ | TAKEN at the post | SEQ (answer) or TAKEN | SEQ of the running launch |
| r2 | RET: kernel return value (int32) | ARG0 echo | image id [31:0] | `mcause` | kernel-defined |
| r3 | first failing ticket; 0xF000_0000 = none | 0 | image id [63:32] | `mepc` | kernel-defined |
| r4 | its `ERR_INFO` | 0 | [15:0] SDK ABI, [31:16] runtime version | `mtval` | kernel-defined |
| r5 | its `ERR_W0` | 0 | kernel count | `RT_STATE` at the trap | kernel-defined |
| r6 | its `ERR_PIDX` | 0 | 0 | `RT_INFO` | kernel-defined |
| r7 | DURATION: core cycles from the pop to the post, saturating at 2^32 − 1 | same | boot core cycles | MTIME[31:0] | kernel-defined |

**Launch STATUS** (r0[15:8] of DONE):

| Code | Name | Meaning |
|---|---|---|
| 0 | OK | the kernel returned RET ≥ 0, and no op of the launch failed |
| 1 | TILE_ERR | the first error (smallest `ERR_PIDX`, §8.2) has a code other than ABORTED; r3–r6 describe it |
| 2 | BAD_DESC | the descriptor was refused (§4.3); the kernel did not run |
| 3 | KERNEL_ERR | RET < 0 (the kernel rejected its arguments or failed in software); no tile error |
| 4 | TRAP | used only in TRAP records |
| 5 | ABORTED | cancelled: the launch was queued and is answered without running (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8) |
| 6 | PLAT_ERR | `STATUS.PLAT_ERR` was set when the runtime answered: the platform reported an AXI error, so outputs may be wrong |
| 7 | STOPPED | the kernel honoured a soft stop (§4.9) and returned early; RET is kernel-defined |
| 8–255 | — | reserved |

Precedence when several codes apply: TRAP > BAD_DESC > ABORTED > TILE_ERR > PLAT_ERR > STOPPED >
KERNEL_ERR > OK. ERR_CODE (r0[23:16]) is the code of the first error whatever the status, and 0
when there is none. A cancel that lands after the kernel's work has completed but before the
answer may still report ABORTED. The host must treat the outputs of any cancelled launch as
undefined.

### 4.5 RISC-V side (control window offsets 0x200–0x2FC, owned by EX)

These replace the H2R / R2H mailbox of ctrl-riscv (`docs/ctrl-riscv.md:304-309`). CC owns the rest
of the control window. Every access is 32-bit; reads have no side effects.

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

Interrupts (replacing ctrl-riscv's `irq_fast[2]` / `[3]` sources): `irq_fast[2]` is pending while
`LQ_AVAIL` ≠ 0 and not ACTIVE; `irq_fast[3]` is pending while `LCQ_FREE` ≠ 0 (the runtime enables
it only while it waits to post).

### 4.6 Base kernels the host relies on

The SW base image must provide these three kernels, because bring-up, the backend and
verification depend on them (contracts in SW §7):

- **`graph`**: the op-list executor (OPL v1, SW §7.2–§7.3). It is the only offload path from M6
  on: in M6 the backend emits op lists with MUL_MAT records only, one launch per v1 split
  (V2-SW-14); M7 adds the other kinds, so a decode token becomes one launch.
- **`replay`**: replays a tile-only `push.bin` stream (SW §5.2, §7.7). ARG0 = its DRAM handle,
  ARG1 = the record count, ARG2 = flags. With flag bit0 SERIAL every op record depends on the
  previous op record instead of its recorded deps; because of I1 this gives v1's strictly
  sequential semantics. Opcodes that the issue path refuses complete `ILLEGAL_OP` on the SYNC unit,
  as on the issue path. RET = 0 when no op failed, else 1 + the push index of the first error
  relative to the launch's first push. The converted `traces/model-v1conv` set, the regenerated
  trace sets, DB's SPM bulk dumps (DB §5.7) and the divergence search run on this kernel.
- **`selftest`**: the card selftest (D15, SW §7.6), launched by `tsr-selftest`.

### 4.7 Runtime dispatcher contract (normative for the SDK runtime)

- **R1 Boot.** crt0 (`docs/ctrl-riscv.md:1177-1181`, without the mailbox steps), then `qclear()`
  and a join on it, then `RT_STATE` := IDLE, then post HELLO as an event. Descriptors pushed before
  HELLO stay queued and are served after it.
- **R2 Take.** Wait for `LQ_AVAIL` ≠ 0 (WFI on `irq_fast[2]`). Copy `LQ_HEAD_W*` and
  `LQ_HEAD_SEQ` into DTCM, then write `LQ_POP`.
- **R3 Cancelled or stopped before start.** If CANCEL_VALID and the sequence is covered by
  `LQ_CANCEL`, or STOP_VALID and it is covered by `LQ_STOP` (§4.9), answer DONE / ABORTED (ERR_CODE
  0) without running.
- **R4 Validate.** BAD_DESC → answer. PING → answer PONG.
- **R5 Run.** `RT_STATE` := KERNEL | index. If PROF is set, set `REC_CFG.ALL` and enable NODE
  markers (w0[8] above). Stage the argument block if the table entry asks for one. Read
  `ISSUE_PIDX` (CC §5.1) as the launch's first push index. Call DB's launch hook (DB §3.8), then
  the kernel; it returns RET (int32).
- **R6 Join.** `join_all()` (FENCE, then join): every op pushed during the launch has completed,
  including aborted ones (I1).
- **R7 Result.** Find the first error by the smallest `ERR_PIDX` over the units in error state
  (`docs/ctrl-riscv.md:577-579`). Read PLAT_ERR from `LQ_FLAGS`. Fill r0–r7 by the precedence of
  §4.4.
- **R8 Answer.** Wait until `LCQ_FREE` ≠ 0 (WFI on `irq_fast[3]`), then post with ANSWER = 1.
- **R9 Clear.** If any unit is in error state, push `qclear()` and join it before R2.
- **R10.** `RT_STATE` := IDLE; continue at R2.

Further rules for the runtime and for kernels:

- Kernels never pop and never post answers. They may post events (types 0xC0–0xFF, ANSWER = 0).
- Kernels never leave an op running at return: R6 enforces this. This is what makes ACTIVE the
  exact PERF window (§6).
- Long kernels poll `tsr::stop_requested()` (STOP_VALID and their sequence covered) at their own
  safe points.
- Logs go only to the log ring, never to the LCQ.
- On a trap, the handler follows DB §6.3: with a debugger attached it stops at
  `__tsr_trap_halt`; otherwise it fills `__tsr_crash`, logs a TRAP record, posts a TRAP record
  (ANSWER = 1 if ACTIVE) and parks in a WFI loop with `RT_STATE` = TRAPPED. The debugger can
  attach there. The runtime does not unwind, because a trap can leave `.data` inconsistent.
  `tsr::panic()` (violated firmware invariants) enters the same handler with `mcause` = 24 and
  `mtval` = its code; there is no separate PANIC message.
- A soft stop replaces a stop message: the host writes `LQ_STOP` (§4.9). There is no STOP / BYE
  exchange and no runtime log-level message: the runtime filters by `LOG_CFG.LEVEL`.

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

- **Timeout clock** (DB §4.1): the wait timeout counts only while `STATUS.RV_HALTED` = 0 and
  `DBG_STATUS.STOP_ACTIVE` = 0 (both read once per backoff poll); with `DBG_CFG.SERIAL` = 1 it is
  off. The transport releases its mutex between polls, so an in-process debug service (DB §4.1)
  can use the device while a launch waits.

- Every wait loop drains the whole LCQ. A kernel that posts events therefore cannot block the
  runtime while the host waits.
- **Poll policy** (default; SW may refine it): busy-poll `LCQ_WR` while the elapsed time is below
  the last DURATION of the same kernel. After that, poll every max(10 µs, 1 % of that duration),
  sleeping between polls. Detection latency is then about max(10 µs, 1 % of the launch). Each poll
  is one PCIe read, and polls never touch DRAM bandwidth.
- The host reads `STATUS` only on a timeout or a non-OK status. It reads the log ring only when a
  launch fails, every ≥ 100 ms during long waits, and at close.

### 4.9 Cancel and stop

Launch q is **covered** by a watermark s iff (s − q) mod 2^32 < 2^31.

- **`LQ_CANCEL`** (0x06C) is a hard cancel. Writing s sets the watermark to s and CANCEL_VALID to
  1. In the same cycle, if ACTIVE and the running launch (sequence ANSWERED) is covered by s, the
  core applies a host QABORT at once (§8.5). Queued launches that are covered are answered ABORTED
  without running (R3). Uncovered launches run normally: the runtime's R9 clears the error states
  left by the abort. This is the operation behind Ctrl-C, the ggml abort callback and the host's
  timeout path.
- **`LQ_STOP`** (0x070) is a soft stop. Writing s sets the watermark to s and STOP_VALID to 1.
  There is no hardware action. Covered queued launches are answered ABORTED. A covered running
  kernel sees `stop_requested()` and returns at its next safe point with status STOPPED (for
  example, the M9 decode loop at a token boundary). A kernel that never polls simply runs to the
  end.
- Both watermarks persist until `SOFT_RESET`, which clears both VALID bits. Writing s = PUSHED − 1
  covers exactly the launches pushed so far.

### 4.10 Log ring

- The RISC-V writes 32-bit words to `LOG_PUSH`. The host reads word k at 0x400 + 4 × (k mod
  `LOG_DEPTH`) for k in [`LOG_RD`, `LOG_WR`), then writes `LOG_RD`.
- **Framing**, which the SDK must follow: a record is one header word followed by N payload words.
  The header is DB's (DB §6.1, the only definition): [7:0] N, [11:8] LEVEL (0 FATAL, 1 ERROR,
  2 WARN, 3 INFO, 4 DEBUG, 5 TRACE), [15:12] SOURCE (0 runtime, 1 kernel, 2 trap handler, 3 debug
  kernels), [18:16] TYPE (FMT, TEXT, TRAP, EVENT, BLOB), [19] TS, [27:20] LSEQ = SEQ[7:0] of the
  running launch, [31:28] SYNC = 0xA. The SDK writes a record only when `LOG_FREE` ≥ 1 + N, so
  records are never torn. With `LOG_CFG.BLOCK` = 1 it waits for space instead of dropping the
  record.
- `LOG_CFG` (host RW): bit0 ENABLE (0: the SDK drops every record), bit1 BLOCK, [7:4] LEVEL (the
  SDK drops records above this level; this is the only runtime level filter). Words that the
  hardware drops are counted in `LOG_DROPPED` (saturating) and set `STATUS.LOG_DROP`.
- The log ring is for diagnosis, not for bulk data: the host pays one PCIe read per word.
  Op traces go to DRAM through DB's hardware trace writer (DB §8.3).

### 4.11 Latency and throughput targets (estimates; the selftest measures them)

| Step | Cost (*estimate*) | Basis |
|---|---|---|
| Host: 2–9 posted MMIO writes (changed words + `LQ_PUSH`) | 0.2–1 µs | XRT register access is a store to an mmap'ed BAR (`.../core/pcie/linux/shim.cpp:2341-2405`) |
| PCIe, adapter, CSR port | 0.5–1 µs | a window write takes 10 kernel cycles (`docs/platform-u200.md:84`) |
| LQ CDC, WFI wake-up, 8 head words, pop | 0.4–1.2 µs (70–200 `clk_rv` cycles at 166.7 MHz) | ctrl-riscv §4.8 costs per MMIO access |
| Dispatch, empty kernel, FENCE join, result, post | 0.6–1.5 µs | as above |
| Host detection: 1–2 poll reads plus the r0 read; `LCQ_RD` write | 2–5 µs | PCIe read latency is not measured (`docs/spec.md:994`); 1–2 µs per read assumed |
| **Round trip of an empty launch** | **≈ 4–8 µs; target ≤ 10 µs** | v1: 35–36 µs per exec, measured (`docs/platform-u200.md:1049`) |
| **Pipelined rate** (≥ 2 launches outstanding) | ≥ 300 K launches/s (device side 1–3 µs each); **target ≥ 100 K/s** | |

Per decode token: M6 still makes one launch per graph split (281 for Gemma 4 12B, 385 for
Qwen2.5-32B, `docs/platform-u200.md:1053`). At 5–10 µs each instead of 35 µs, that saves about
7–12 ms per token (*estimate*) of v1's 10.1 / 13.6 ms exec overhead. M7 makes one launch per token, so the launch costs ≤ 10 µs per token. Between two
launches the device idles for only the dispatch and answer time (≈ 1–3 µs, *estimate*). That is
negligible for the "DDR always busy" goal (`docs/spec.md:565-587`).

---

## 5. Core CSR map v2 (host view)

### 5.1 Offsets

Core CSR offsets are 12-bit; on the U200 the window is `s_axi_control` 0x1000 + offset. Every read
has no side effects except the PERF LO snapshot (host port only). A `LOAD_DATA` read while
`RV_RUN` = 1 or `SR_BUSY` returns 0 and sets nothing. The DMI window (0xC00–0xDFC) acts only while
`DBG_CFG.DMI_EN` = 1 (DB §3.2), because debug-module reads can have side effects. Undefined offsets
read 0 and ignore writes (`docs/spec.md:1058`). Every access completes without `clk_rv`, within a
bound (`docs/ctrl-riscv.md:355-362`). This is the only host CSR map of v2; CC, DB, VU and AT place
their registers inside it, and the RISC-V reads it through CC's core window (CC §5.3).

| Offset | Name | Access | Content | Owner |
|---|---|---|---|---|
| 0x000 | `ID` | R | 0x5453_5231 "TSR1" (unchanged) | EX |
| 0x004 | `VERSION` | R | [15:8] ISA major = **2**, [7:0] minor = 0 | EX |
| 0x008 | `CFG0` | R | unchanged: NUM_CH[7:0], LANES[19:8], T_MAX[23:20] | EX |
| 0x00C | `CFG1` | R | unchanged: SPM size in lines | EX |
| 0x010 | `CFG2` | R | unchanged: VL[7:0], NSREG[15:8], K_MAX/256[31:16] | EX |
| 0x014 | `CAPS` | R | §7.5 | EX |
| 0x018 | `CFG3` | R | [15:0] defined by VU, [31:16] defined by AT; 0 until they are assigned | VU, AT |
| 0x01C | `CTL_ID` | R | [7:0] control-core version (2), bit8 debug module, bit9 completion records, bit10 SPM bridge, bit11 log ring, bit12 watchdog, bit13 SBA connected, bit14 SPM line-buffered alias | EX, CC |
| 0x020 | `CTRL` | RW / W1 | §5.2 | EX |
| 0x024 | `STATUS` | R / W1C | §5.3 | EX |
| 0x028 | `RV_BOOT` | RW | boot vector, 256-byte aligned ([7:0] read 0); taken when `RV_RUN` rises (was `CTL_BOOT`) | CC |
| 0x02C | `RV_FREQ` | R | `clk_rv` cycles in the last 2^16 core cycles; 0 = no `clk_rv` (was `CTL_RV_FREQ`) | CC |
| 0x030 | `RV_CFG` | R | [15:0] ITCM KiB, [31:16] DTCM KiB (was `CTL_CFG`) | CC |
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
| 0x088 | `Q_DEPTH` | R | queue depth of unit u at [5u+4:5u], u = 0..5; 0 = unit absent (U200 M6: DMA 8, QACT 4, MVM 4, VU 16, unit 4 absent, SYNC 4) | CC |
| 0x090 | `LOG_WR` | R | log words accepted | EX |
| 0x094 | `LOG_RD` | RW | log words consumed by the host | EX |
| 0x098 | `LOG_CFG` | RW | bit0 ENABLE, bit1 BLOCK, [7:4] LEVEL; reset value 0x0000_00F1 | EX |
| 0x09C | `LOG_DROPPED` | R | words dropped, saturating; cleared by `SOFT_RESET` | EX |
| 0x0C0 | `LOAD_ADDR` | RW | TCM byte address (word aligned); a write restarts `LOAD_CRC` | EX |
| 0x0C4 | `LOAD_DATA` | RW | write: store the word, address += 4; read: the word at the address (no increment). Only while `RV_RUN` = 0 and not `SR_BUSY`: otherwise a write is dropped and sets `LOAD_REFUSED`, and a read returns 0 and sets nothing | EX |
| 0x0C8 | `LOAD_CRC` | R | CRC-32 of the words accepted, or of the words read by VERIFY | EX |
| 0x0CC | `LOAD_COUNT` | RW | word count for VERIFY | EX |
| 0x0D0 | `LOAD_CTRL` | W | bit0 VERIFY | EX |
| 0x0D4 | `LOAD_STATUS` | R | bit0 VERIFY_BUSY | EX |
| 0x100–0x2FC | `PERF_0`–`PERF_63` | R | 64-bit counters (§6): LO at 0x100 + 8i, HI at +4; a LO read snapshots the counter for the next HI read (`docs/spec.md:1067`) | EX |
| 0x300–0x3FC | `LCQ_REC` | R | completion ring: record k at 0x300 + 32 × (k mod `LCQ_DEPTH`), word i at +4i | EX |
| 0x400–0x7FC | `LOG_RING` | R | log word k at 0x400 + 4 × (k mod `LOG_DEPTH`) | EX |
| 0x800–0xAFC | `DBG_*` | DB-defined | tile-level debug block (768 B, DB §10.1); undefined offsets read 0 | DB |
| 0x9C0–0x9FC | `SREG0`–`SREG15` | R | the VU's s0–s15 (f32 bits), inside the DBG range but present whenever the VU is built, independent of `HAS_DBG` (VU §4.4, DB §10.1). The one read location of the s bank for the host and for the RISC-V (core window 0x2000_1000 + 0x9C0 + 4i) | VU, DB |
| 0xB00–0xBFC | — | | reserved (the planned `CTL_*` block is dissolved) | — |
| 0xC00–0xDFC | `DMI` | R / W | debug module register r at 0xC00 + 4r; bounded by 4,096 core cycles, `DMI_TIMEOUT` (`docs/ctrl-riscv.md:351`) | DB, CC |
| 0xE00–0xF1C | `Q_*` | R | queue status (§5.4) | EX |
| 0xF20–0xFFC, and every gap above | — | | reserved | — |

**Removed from v1** (`docs/spec.md:345-371`, `docs/ctrl-riscv.md:333-351`): `CTRL.START` and
`CTRL.ABORT`; `STATUS.BUSY`, `DONE`, `ERR`, bit3 and `HALTED`; `CMD_ADDR_LO` / `HI`; `CMD_COUNT`;
`ERR_PC`; `ERR_CODE`; `ERR_INFO`; `PC`; `ARG0`–`7`; `TRC_*`; `CTL_PERF_*` (merged into `PERF_*`);
the whole `CTL_*` block (its registers moved to the offsets above); `CTL_H2R_*` / `CTL_R2H_*`
(replaced by LQ / LCQ); `CTL_DONE_SEQ*` / `CTL_ISSUED_SEQ*` (replaced by `Q_*`). The core's `done`
output is removed as well.

### 5.2 `CTRL` (0x020)

| Bit | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `RV_RUN` | RW level | 1 releases the RISC-V from reset. 1 → 0 starts the RISC-V reset sequence (§8.5). Writes are ignored, with `RV_RUN_REFUSED` set, while `SR_BUSY` |
| 1 | `HOST_HOLD` | RW level | the host's hold source for the queue heads (`docs/ctrl-riscv.md:321-324`) |
| 2 | `HOLD_ON_HALT` | RW level | a debug halt holds the queue heads |
| 3 | `IRQ_EN` | RW level | `host_irq_o` = `IRQ_EN` ∧ (LCQ_WR ≠ LCQ_RD); the U200 leaves the output unconnected |
| 8 | `QABORT` | W1 pulse | host QABORT, applied at once (§8.5) |
| 9 | `QCLEAR` | W1 pulse | host QCLEAR: accepted only when QIDLE holds and the RISC-V is quiet (reset done, or halted with no push in flight); otherwise `QCLEAR_REFUSED` |
| 10 | `SOFT_RESET` | W1 pulse | §8.5 |
| others | — | | 0 |

Pulse bits read 0, so a read-modify-write of the level bits never re-triggers them. Each level bit
has exactly one owner, the host. Level bits are updated before the pulses of the same write act.

### 5.3 `STATUS` (0x024)

| Bit | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `RV_RUNNING` | level | `RV_RUN` = 1 and the hart is out of reset |
| 1 | `RV_HALTED` | level | debug halt |
| 2 | `RV_SLEEP` | level | WFI |
| 3 | `RV_CLK_OK` | level | `clk_rv` runs and its domain is out of reset |
| 4 | `RV_RESET_DONE` | level | the RISC-V reset sequence has completed (`docs/ctrl-riscv.md:1000-1008`) |
| 5 | `ACTIVE` | level | TAKEN ≠ ANSWERED |
| 6 | `QIDLE` | level | no entry queued or running, issue FIFO empty |
| 7 | `QQUIESCENT` | level | `docs/ctrl-riscv.md:622-630` |
| 8 | `QERR` | level | some unit is in error state |
| 9 | `SR_BUSY` | level | `SOFT_RESET` in progress |
| 10 | `LCQ_PENDING` | level | LCQ_WR ≠ LCQ_RD |
| 11 | `LOG_PENDING` | level | LOG_WR ≠ LOG_RD |
| 12 | `PLAT_ERR` | level | the platform's error input `plat_err_i` (§9.3) |
| 15:13 | `HOLD_SRC` | level | hold sources: host, RISC-V, debug halt |
| 16 | `LOAD_REFUSED` | sticky | a `LOAD_DATA` write or a VERIFY command while `RV_RUN` = 1 or `SR_BUSY` (reads never set it) |
| 17 | `LOAD_TIMEOUT` | sticky | a load or VERIFY word timed out |
| 18 | `DMI_TIMEOUT` | sticky | |
| 19 | `QCLEAR_REFUSED` | sticky | |
| 20 | `PUSH_TIMEOUT` | sticky | the RISC-V's issue safety net fired (`docs/ctrl-riscv.md:471-480`) |
| 21 | `BRIDGE_TIMEOUT` | sticky | |
| 22 | `LQ_OVERFLOW` | sticky | a push while the LQ was full (descriptor dropped) |
| 23 | `RING_RD_BAD` | sticky | an `LCQ_RD` or `LOG_RD` write was refused |
| 24 | `LOG_DROP` | sticky | log words were dropped |
| 25 | `UNIT_HUNG` | sticky | a unit watchdog fired (§8.6) |
| 26 | `SR_RV_TIMEOUT` | sticky | `SOFT_RESET` went on without the RISC-V reset completing (no `clk_rv`) |
| 27 | `RV_RUN_REFUSED` | sticky | |
| 31:28 | — | | 0 |

W1C clears the sticky bits only; writes to the level bits are ignored.

### 5.4 Queue status `Q_*` (0xE00–0xF1C)

These are core-domain values: the same records the RISC-V sees, readable even when the RISC-V is
dead. Unit slots u = 0..7; units that are absent read 0. Different registers may be read at
different moments (they are diagnostic only).

| Offset | Name | Content |
|---|---|---|
| 0xE00 + 4u | `Q_DONE[u]` | `DONE_SEQ[u]`, 28 bits |
| 0xE20 + 4u | `Q_ISSUED[u]` | `ISSUED_SEQ[u]`, 28 bits (synchronised copy) |
| 0xE40 + 4u | `Q_STATE[u]` | [4:0] entries in the queue, bit5 RUNNING (the unit holds an op), bit6 DEP_WAIT (the head waits for deps), bit7 HELD, bit8 ERR (error state), bit9 HUNG (watchdog), [31:16] 0 |
| 0xE60 | `Q_FIFO` | [7:0] issue-FIFO entries waiting for the dispatcher |
| 0xE64 | `Q_PIDX` | push index the dispatcher assigns to its next ticketed entry (= `DBG_NEXT_PIDX`); it lags the RISC-V's `ISSUE_PIDX` (CC §5.1) by the issue-FIFO contents, so firmware never uses it |
| 0xE80 + 16u | `Q_ERR_SEQ[u]`, `Q_ERR_CODE[u]` ([7:0]), `Q_ERR_INFO[u]`, `Q_ERR_W0[u]` | first error record of unit u (§8.2) |
| 0xF00 + 4u | `Q_ERR_PIDX[u]` | its push index |

---

## 6. PERF counters v2

v1 counted only while a host run was busy (`docs/spec.md:1067`, `:1075`), and ctrl-riscv's
`CTL_PERF` counted "outside host runs" (`docs/ctrl-riscv.md:333`). Neither condition exists in v2.
One counter space replaces both, with two gates:

- **ACTIVE** (TAKEN ≠ ANSWERED): a launch is in progress, from its pop to its answer. Because of
  R6, no datapath op runs outside ACTIVE, so **PERF does not change between launches**. This is
  the v1 property "실행 사이 PERF 불변" restated per launch: when the host does not pipeline
  launches, the difference of two reads is exactly one launch.
- **QHALT** (`RV_HALTED` ∧ `QQUIESCENT`): the whole device is stopped by the debugger. Time
  counters do not count then; this replaces prof-dbg's halted-time rule.

Time counters count core cycles with ACTIVE ∧ ¬QHALT. Event counters count with ACTIVE. Counters
kept in the RISC-V domain count `clk_rv` cycles with a synchronised ACTIVE. `SOFT_RESET` zeroes
every counter and drops the snapshot. With C = `NUM_CH` and X = 5 + 2C (k4: X = 13):

| Index | Name | Kind | Meaning |
|---|---|---|---|
| 0 | `CYCLES` | time | core cycles |
| 1 | `RETIRED` | event | queue entries completed, on every unit and with any outcome (= Σ of `DONE_SEQ` advances) |
| 2 | `MVM_CYCLES` | time | the MVM unit holds an op (from the cycle after accept to done; unchanged) |
| 3 | `VU_CYCLES` | time | the VU holds an op |
| 4 | — | | reserved (was WAITM), reads 0 |
| 5 + c | `CH_RD_BEATS[c]` | event | read beats delivered on channel c. With fetch gone, ch0 is now exact (closes the open item at `docs/spec.md:989`) |
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
| X + 22 … 60 | — | | free; unassigned slots read 0 (at `NUM_CH` = 16, X + 21 = 58) |
| 61 | `TRW_RECORDS` | event | trace records written to DRAM (DB §8.1) |
| 62 | `TRW_DROPS` | event | trace records dropped |
| 63 | `DBG_STOP_CYCLES` | time, not gated by QHALT | core cycles with ACTIVE ∧ (`RV_HALTED` ∨ `STOP_ACTIVE`); a measurement is valid only if it is 0 |

The remainder `CYCLES` − `MVM_CYCLES` − `MVM_WAIT` − `MVM_EMPTY` counts the cycles in which the MVM
unit is idle while its head could start: the accept latency, and entries that complete without
running. The RTL asserts that this remainder is at most 3 cycles per MVM-queue entry. The three
counters split the spec's G (`docs/spec.md:572`) into an issue-bound share and a
dependency-bound share, which is the measurement the "DDR always busy" priority needs. The host is the only PERF reader. The runtime times launches with MTIME (record
r7).

---

## 7. ISA v2

### 7.1 Instruction format

Instructions stay 256 bits (w0–w7, little-endian). w0 = {x16 = w0[31:16], flags[15:12], sub[11:8],
opcode[7:0]}.

- **x16** (w0[31:16], formerly ra, rb, rc and rd) is an op-specific field; x16 bit i is w0 bit
  16 + i, and the area documents name its bits x16[15:0]. It must be 0
  (`ILLEGAL_OP` otherwise) for every opcode whose definition does not assign it. That includes LD,
  ST, QACT, MVM, NOP and FENCE, so their v2 encodings equal their v1 encodings with ra–rd = 0.
  VU and AT may assign x16 in the ops they define.
- **Address fields** A1, A2 and A3 are the immediate words themselves: A = imm. There is no base
  register. Range checks are unchanged (`docs/spec.md:1053`), with a = imm.
- NOP and FENCE use no words: w1–w7 and x16 must be 0.

### 7.2 Global opcode table

| Opcode | Mnemonic | v2 status | Unit | Defined by |
|---|---|---|---|---|
| 0x00 | `NOP` | kept: SYNC join point (completes when its dependencies are done) | 5 SYNC | spec, ctrl-riscv §4.4 |
| 0x01 | `END` | **reserved**: `ILLEGAL_OP` | (SYNC) | — |
| 0x02 | `LI` | **reserved** | (SYNC) | — |
| 0x03 | `ADDI` | **reserved** | (SYNC) | — |
| 0x04–0x08 | `MULI`, `RLD`, `LOOP`, `WAITM`, `JUMP` | retired, never reused | (SYNC) | — |
| 0x09 | `CALL` | retired (never shipped), never reused | (SYNC) | — |
| 0x0A–0x0F | — | reserved: SYNC-class extension | (SYNC) | EX |
| 0x10 | `LD` | kept, encoding unchanged | 0 DMA | spec |
| 0x11 | `ST` | kept, encoding unchanged | 0 DMA | spec |
| 0x12 | `DQROW` | kept number; re-specified as SPM → f32 (VU §5.6); simulator from M6, RTL in M8 (`CAPS` bit6 = 0 in M6 bitstreams) | 3 VU | VU |
| 0x13 | — | reserved for VU (DMA class) | | VU |
| 0x14 | `KVST` | new: store K or V rows into a KV plane (AT §5.1) | 0 DMA | AT |
| 0x15–0x17 | — | reserved for AT (DMA class: KV row stores and loads) | | AT |
| 0x18 | `QACT` | kept, encoding unchanged | 1 QACT | spec |
| 0x19–0x1B | — | reserved for AT (quantize class: q8_0 KV rows) | | AT |
| 0x1C–0x1F | — | reserved | | — |
| 0x20 | `MVM` | kept, encoding unchanged | 2 MVM | spec |
| 0x21 | `ATTQK` | new: scores and row maxima over a KV group range (AT §5.2) | 2 MVM | AT |
| 0x22 | `ATTPV` | new: P·V channel partials and row sums (AT §5.3) | 2 MVM | AT |
| 0x23 | — | reserved for AT (MVM class) | | AT |
| 0x24–0x2F | — | reserved (MVM class) | | — |
| 0x30–0x33 | `VOP`, `RED`, `ROPE`, `TOPK` | kept numbers; re-specified (TOPK: simulator from M6, RTL in M8) | 3 VU | VU |
| 0x34–0x37 | `ASMAX`, `AMASK`, `AMERGE`, `ANORM` | new: softmax and merge steps; semantics AT §5.4–§5.7, executed by the VU (VU §5.11) | 3 VU | AT |
| 0x38 | `VMASK` | new: position-window mask for caches where cell j holds position k0 + j (VU §5.7); reserved until M9: `ILLEGAL_OP` in M6/M7 | 3 VU | VU |
| 0x39–0x3F | — | reserved for VU extensions | 3 VU | VU |
| 0x40 | `SOP` | kept number; re-specified | 3 VU | VU |
| 0x41–0x47 | — | reserved for VU (scalar extensions) | 3 VU | VU |
| 0x48–0x6F | — | reserved (future units) | | — |
| 0x70–0x7E | — | reserved: SYNC-class extension | | EX |
| 0x7F | `FENCE` | kept: SYNC barrier (completes when every op pushed before it is done) | 5 SYNC | spec, ctrl-riscv §4.4 |
| 0x80–0xFF | — | reserved | | — |

- Every reserved, retired or absent opcode is still accepted by the issue port. It gets a ticket on
  the SYNC unit and completes `ILLEGAL_OP` (`docs/ctrl-riscv.md:481-482`), so every push has a
  completion.
- QCLEAR, QABORT, QHOLD and TRACE_MARK are control-window entries, not opcodes. They stay with CC.
- VU and AT fix the encodings, the units and the dispatch checks of their ops within these numbers.
  Any further number they need is requested from EX and recorded here.
- **No op may fail after it has started.** Every error is found by the dispatch checks
  (`docs/ctrl-riscv.md:556-561`). Data-dependent conditions, such as NaN or a KV length read from
  SPM, must be defined behaviour (clamping, numeric rules), not run-time errors. VU and AT design
  their ops under this rule.

### 7.3 Units

| Id | Unit | Opcodes | Notes |
|---|---|---|---|
| 0 | DMA | LD, ST, KVST | |
| 1 | QACT | QACT | |
| 2 | MVM | MVM, ATTQK, ATTPV | |
| 3 | VU | VOP, RED, ROPE, TOPK, SOP, DQROW, ASMAX, AMASK, AMERGE, ANORM (VMASK from M9) | |
| 4 | ATT | reserved for a future AT attention engine; **absent in M6** (`Q_DEPTH[4]` = 0, `ISSUE_UNITS[4]` = 0) | **was CMD (CALL)** |
| 5 | SYNC | NOP, FENCE, QCLEAR, refused opcodes | |
| 6–13 | — | reserved (for example a second DMA engine) | |
| 14 | — | software marker records | |
| 15 | — | "no dependency" | |

A dependency on unit 6–14, or on a unit that is absent in the bitstream, is `BAD_DEP`. The opcode
→ unit table is static: it is generated from the single table source (V2-EX-30) into the RTL
(`UNIT_OF`), the simulator and the SDK. `ISSUE_UNITS` (control window, CC §5.1) and `Q_DEPTH`
(host, §5.1) report which units a bitstream built. There is no `ISSUE_QMAP` register.

### 7.4 Scalar registers

There is one s0–s15 bank (D14). It is written only by VU-queue ops and read by the RISC-V through
the core window after a join (`docs/ctrl-riscv.md:641-653`, bank 1). The one read location is
`SREG0`–`SREG15` at core CSR 0x9C0–0x9FC (§5.1): host offset 0x9C0 + 4i, RISC-V address
0x2000_1000 + 0x9C0 + 4i. Its initial value and its behaviour across `SOFT_RESET` are VU's. EX's
requirement: `SOFT_RESET` does not change it, and a hardware reset sets it to +0.0.

### 7.5 `VERSION`, `CAPS`, `CFG3`

- `VERSION` = 0x0000_0200 (ISA 2.0). A v1 tool fails at the ISA-major check
  (`docs/xrt-host.md:72-74`).
- `CAPS`: bit0 VU (VOP, RED, ROPE, SOP present), bit1 0, bit2 retired, bit3 TOPK (RTL from M8),
  bit4 retired, **bit5 ATT** (the seven AT ops are present), **bit6 DQROW** (RTL from M8), bit7
  VMASK (M9), bits 8–11 reserved for VU, bits 12–15 reserved for AT, bits 16–31 0. Retired bits are
  never set and never reused. The simulator reports what its configuration implements, so a
  simulator may set bits 3 and 6 while an M6 bitstream does not.
- `CFG3`: [4:0] VU front lanes with an EXP core, [9:5] VU back lanes, [15:10] 0 (VU §5.10);
  [31:16] PCAP / 16 (AT §5.10). The VU fields are information for cycle models only. PCAP is a
  configuration parameter like T_MAX: it enters AT's checks and chunk policy (AT §5.8, §6.6), so
  results are defined per configuration, as for every op.

---

## 8. Error model v2

### 8.1 Op error codes

The CSR fields are 8 bits wide. v2 uses only 0–15, so the internal width stays 4 bits
(`rtl/core/tsr_pkg.sv:140`).

| Code | Name | Condition | `ERR_INFO` |
|---|---|---|---|
| 0 | NONE | | |
| 1 | `ILLEGAL_OP` | reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it | 0 |
| 2 | `BAD_FMT` | unsupported weight format | 0 |
| 3 | `QA_MISMATCH` | reserved, never raised (`rtl/core/tsr_pkg.sv:144`) | |
| 4 | `SPM_RANGE` | `docs/spec.md:1053` | start address of the failing operand |
| 5 | `DRAM_RANGE` | as above | start address |
| 6 | `K_RANGE` | K > K_MAX or not a multiple of the block size | 0 |
| 7 | `T_RANGE` | T = 0 or T > T_MAX | 0 |
| 8 | `ABORTED` | the entry completed without running: a QABORT (from the host, the RISC-V, a cancel, a RISC-V reset or `SOFT_RESET`), or its unit was in error state | 0 |
| 9, 10 | — | reserved (retired `LOOP_NEST`, `WAIT_TIMEOUT`), never raised | |
| 11 | `DEP_FAILED` | a dependency completed with an error | 0 |
| 12 | `BAD_DEP` | a dependency on a ticket not pushed yet, on unit 6–14 or an absent unit, or below its unit's clear point | 0 |
| 13 | **`OPERAND`** (new) | an operand value outside the range its op defines, other than K, T and the address ranges (for example n = 0 where that is forbidden, a TOPK k above its limit, an unsupported head dimension); used by VU and AT ops only | the offending value |
| 14, 15 | — | reserved | |

**Check order** at dispatch: `ILLEGAL_OP` → `BAD_DEP` → `BAD_FMT` → `K_RANGE` → `T_RANGE` →
`OPERAND` → `SPM_RANGE` → `DRAM_RANGE`. `ERR_W0` is the op's w0.

### 8.2 Op lifecycle, records and propagation

Kept from ctrl-riscv §4.4–§4.5 (`docs/ctrl-riscv.md:527-602`) with CALL removed:

- **Poisoning.** An op that fails a dispatch check is poisoned: it enters its queue, changes no
  state, and completes with its code at the head.
- **At the head**, after its dependencies: a poisoned entry completes with its code. Otherwise, if
  its unit is in error state, it completes `ABORTED`. Otherwise, if a dependency failed, it
  completes `DEP_FAILED`. Otherwise it runs.
- **Unit error state.** The first nonzero completion of a unit records `ERR_SEQ`, `ERR_CODE`,
  `ERR_INFO`, `ERR_W0` and `ERR_PIDX` (RISC-V view, plus `Q_ERR_*` for the host). Every later entry
  of that unit completes `ABORTED` until a QCLEAR.
- **I1:** every entry completes only at its head and only after its dependencies, so done(t)
  implies that everything t depends on is done. **I2:** `DONE_SEQ[u]` does not pass `ERR_SEQ`
  before the record is visible to the RISC-V.
- **First error** = the record with the smallest `ERR_PIDX`. It is independent of the schedule for
  valid programs, and it is what the launch result reports.

### 8.3 Launch-level result

See §4.4: one record per descriptor, with status precedence, ERR_CODE, ticket, `ERR_INFO`,
`ERR_W0` and `ERR_PIDX` of the first error. Platform AXI errors reach the launch result through
`plat_err_i` (status PLAT_ERR) without a host read per launch.

### 8.4 Device-level errors

These are the `STATUS` sticky bits (§5.3), `ADAPTER_STATUS` (§9.2), TRAP records, the watchdog
(`UNIT_HUNG`), image errors (receive or VERIFY CRC mismatch, refused `.note.tsr` requirements,
HELLO with a wrong image id) and a missing `clk_rv` (`RV_CLK_OK` = 0). The transport turns each
into an error of the open, boot, launch or wait call with a decoded message (§10).

### 8.5 Control actions

| Action | Queued entries | Running op | RISC-V | LQ / LCQ / LOG | PERF, error state | TCM, SPM, DRAM |
|---|---|---|---|---|---|---|
| Host QABORT (`CTRL` bit 8, or `LQ_CANCEL` covering the running launch) | complete `ABORTED` in order, after their deps; every unit in error state until a QCLEAR (`docs/ctrl-riscv.md:580-586`) | runs to completion (no cancel) | continues; its join returns, R7 reports ABORTED, R9 clears | unchanged | error state set | unchanged |
| RISC-V QABORT (`QCTRL`) | the same, in push order | the same | continues | unchanged | | unchanged |
| RISC-V QCLEAR | applied after every earlier entry has completed | — | continues | unchanged | errors cleared, clear points set | unchanged |
| Host QCLEAR | refused unless QIDLE holds and the RISC-V is quiet | | | | | |
| RISC-V reset (`RV_RUN` 1 → 0, or the DM's ndmreset) | host QABORT at once | completes | reset sequence (`docs/ctrl-riscv.md:1000-1013`) | ANSWERED := TAKEN (the running launch gets no record); queued descriptors stay and are served after the next boot | error state set (cleared by the next boot's R1) | unchanged |
| **`SOFT_RESET`** (`CTRL` bit 10) | sequence below | completes | reset and left in reset (`RV_RUN` = 0) | flushed: TAKEN := PUSHED, ANSWERED := TAKEN, LCQ_RD := LCQ_WR, LOG_RD := LOG_WR; watermarks invalid | PERF zeroed, error states and records cleared, clear points set, sticky bits cleared | unchanged |
| Debug halt | continue, unless `HOLD_ON_HALT` | completes | halted | unchanged | time counters stop at QHALT | unchanged |
| Hardware reset (xclbin load, card reset) | all state reset | | in reset | counters 0 | 0 | TCM and SPM undefined; DRAM kept unless the platform clears it |

**`SOFT_RESET` sequence** (hardware, started by the write, which itself completes at once;
`SR_BUSY` = 1 throughout):

1. Clear `RV_RUN`, which starts the RISC-V reset sequence (QABORT at once). Clear DB's stop state
   (`S_VALID` and the match queue, DB §5.2) and force `dbg_hold[u]` = 0 for the rest of the
   sequence; DB's configuration (TBPs, `SERIAL`, TRW) and evidence are kept. Wait for
   `RV_RESET_DONE`, at most 2^16 core cycles. If `clk_rv` is absent, go on and set
   `SR_RV_TIMEOUT`.
2. Wait until no datapath unit holds an op and every queue and the issue FIFO have drained. Aborted
   entries drain at one per cycle per unit. This step is bounded by the longest running op (≈ 10.8
   ms for the 32B output head, `docs/ctrl-riscv.md:881-882`). A stuck unit never finishes it (§8.6).
3. Apply the host QCLEAR effects and every flush and clear listed in the table above.
4. `SR_BUSY` := 0.

CC implements the whole sequence in a core-domain SR sequencer that needs no `clk_rv` (CC §12.2):
step 1 waits at most 2^16 core cycles for `RV_RESET_DONE`, step 2 waits for QIDLE with no unit
holding an op, step 3 applies the clear list in the core domain. The RISC-V-side state (LOG_WR_rv,
interrupt levels) reloads from the core domain when its domain leaves reset.

Kept across `SOFT_RESET`: DRAM, SPM, the TCMs, the s bank, the LQ staging registers, the channel
bases and `MEM_EN`, the DBG configuration (TBPs, `SERIAL`, TRW ring) and DB's evidence (history
ring, TRW pointers, RV snapshot), `RV_BOOT`, `HOST_HOLD`, `HOLD_ON_HALT`, `IRQ_EN`, `LOG_CFG`,
`WDOG_CFG`, and every sequence counter: ticket sequences, push indices, PUSHED, LCQ_WR and LOG_WR.
The counters are not reset, so tickets and launch numbers stay unique across restarts. The debug
module follows the hardware reset only, so a debugger stays attached.

### 8.6 Stuck units, watchdog and recovery

- **No forced unit reset in v2.** An AXI transaction cannot be cancelled. A unit reset with bursts
  outstanding would let stale read responses reach a reused id, and leave write bursts that the
  interconnect still expects (`docs/platform-u200.md:201-212`). Recovering from a truly stuck unit
  therefore needs a hardware reset (xclbin reload or `xrt-smi reset`). This closes the open item
  at `docs/spec.md:996` for v2.
- **Progress watchdog** (diagnosis only; it changes no execution). For each datapath unit (DMA,
  QACT, MVM, VU, ATT) a counter runs while the unit holds an op and resets on progress. Progress
  is any SPM request handshake of the unit, any memory request, read beat or write beat of the
  unit, or its `done`. When the counter reaches 2^L (L = `WDOG_CFG[4:0]`, reset value 24 =
  16.8 M cycles ≈ 61 ms at 276 MHz), the hardware sets `Q_STATE[u].HUNG` and `STATUS.UNIT_HUNG`.
  The longest legitimate op makes progress on every beat, so it never trips (the 10.8 ms output
  head included).

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

## 9. XRT platform adapter v2 (`rtl/platform/alveo_xrt`)

### 9.1 Control protocol: `ap_ctrl_none`

- kernel.xml: `hwControlProtocol="ap_ctrl_none"`, `interrupt="false"`. The only args are `ch<c>`
  (`void*` on `m_axi_ch<c>`) at 0x040 + 8c; `cmd_addr`, `cmd_count` and `arg0`–`7` are removed. The
  `s_axi_control` range stays 0x2000.
- XRT facts (2.18, pinned sources under `build/xrt-driver/xrt-202420/src/runtime_src/`):
  - `xrt::kernel` refuses `ap_ctrl_none` compute units: "AP_CTRL_NONE is only supported by XRT
    native API xrt::ip" (`core/common/api/xrt_kernel.cpp:1560-1561`).
  - The driver does not touch control or interrupt registers of such units
    (`core/common/drv/cu_hls.c:513-518`) and refuses interrupts for them
    (`core/common/drv/xrt_cu.c:764-766`).
  - `xrt::ip` register access is an mmap'ed MMIO access (`core/pcie/linux/shim.cpp:2341-2405`).
  - The buffer of channel c is `xrt::bo(device, size, device_only, mem)`, where `mem` is the index
    from `xrt::xclbin::arg::get_mems()` of arg `ch<c>` (`core/include/experimental/xrt_xclbin.h:235-273`).
- *confirm* in the first v2 smoke link that v++ 2024.2 accepts an `ap_ctrl_none` RTL kernel with
  `s_axi_control` plus `m_axi` args and honours its `sp=` connectivity. **Fallback**, which needs no
  RTL change: declare `ap_ctrl_hs` with `interrupt="false"`, make 0x000 read a constant ap_idle =
  1, and never call `xrt::run`. The host still uses `xrt::ip`.

### 9.2 Register map

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x000–0x03C | — | | reserved, read 0 (v1 `AP_CTRL`, `GIE`, `IER`, `ISR`, `cmd_addr`, `cmd_count`, `arg*` deleted) |
| 0x040 + 8c | `ch<c>_base` lo / hi | RW | device byte address of the channel-c buffer; 4 KiB aligned. Written while `MEM_EN` = 0; a write while `MEM_EN` = 1 is ignored and sets `BASE_LOCKED` |
| 0x0E0 | `ADAPTER_CTRL` | RW | bit0 `MEM_EN`. Rising edge: if every base is 4 KiB aligned, latch all bases into the bridges and set `MEM_EN`; otherwise stay 0 and set `ADAPTER_STATUS` bit3. While `MEM_EN` = 0, core memory requests are held (not forwarded, not dropped) and set bit4 |
| 0x0F0 | `ADAPTER_STATUS` | R / W1C sticky | bits 0–3 and [31:16] as v1 (`docs/platform-u200.md:79`); bit3 now means "base misaligned at `MEM_EN`"; **bit4 `REQ_HELD`**, **bit5 `BASE_LOCKED`** |
| 0x0F4 | `ADAPTER_VERSION` | R | {16'h0002, 8'h00, NUM_CH} |
| 0x1000 + off | core CSR window | | unchanged: off < 0x1000, any time, writes need WSTRB = 0xF |

The transport clears `MEM_EN` only when the core is idle (after `SOFT_RESET`). Outstanding
transactions complete normally either way. Deleted from `tsr_xrt_ctrl.sv`: the ap_ctrl_hs FSM, the
queued-start behaviour, the run sequencer (13 core CSR transfers per start), the `SOFT_RESET`
watch, the interrupt output and the arg registers. The memory bridges (§4) and the reset scheme
(§5) of `docs/platform-u200.md` are unchanged.

### 9.3 Core boundary changes (spec "코어 경계 인터페이스", `docs/spec.md:69-76`)

- **Removed:** the `done` output.
- **Added:** `clk_rv`, `rst_rv_n` (CC); `plat_err_i` (a level from the platform: on the U200,
  `ADAPTER_STATUS` bits 0–2 or the error count ≠ 0; it shows in `STATUS.PLAT_ERR` and in
  `LQ_FLAGS`, and is tied to 0 where there is no adapter); `host_irq_o` (`CTRL.IRQ_EN` ∧
  LCQ_WR ≠ LCQ_RD, for ports with a host interrupt such as Zynq; not connected on the U200).

---

## 10. Transport API v2 (`sw/transport`)

### 10.1 C API (`tsr_transport.h`)

```c
// unchanged: open, close, name, config, write, read, csr_read
tsr_transport *  tsr_transport_open(const char *kind);   // v2: SOFT_RESET + bind, unless TSR_ATTACH=1
int              tsr_transport_csr_write(tsr_transport *t, uint32_t off, uint32_t val);   // new (DB, tools)

typedef struct tsr_image tsr_image;                       // parsed firmware ELF
int         tsr_image_open(const char *elf_path, tsr_image **out);   // checks segments and .note.tsr
void        tsr_image_close(tsr_image *img);
int         tsr_image_kernel(const tsr_image *img, const char *name); // index, -1 if absent
uint64_t    tsr_image_id(const tsr_image *img);

int tsr_transport_boot(tsr_transport *t, const tsr_image *img);       // warm if possible, else cold; HELLO checked

typedef struct tsr_launch { uint8_t type; uint8_t flags; uint16_t kernel; uint32_t arg[7]; } tsr_launch;
typedef struct tsr_record { uint32_t w[8]; } tsr_record;               // §4.4; accessors in tsr_isa.h
int tsr_transport_launch(tsr_transport *t, const tsr_launch *l, uint32_t *seq);  // blocks only at LQ_DEPTH outstanding
int tsr_transport_wait(tsr_transport *t, uint32_t seq, uint64_t timeout_ns, tsr_record *rec);
        // 0: answered (rec->w[0] has the status); <0: transport failure or timeout (device diagnosed, §8.6)
        // the timeout counts only while the RISC-V is not halted and no tile stop is active (§4.8)
int tsr_transport_call(tsr_transport *t, const tsr_launch *l, uint64_t timeout_ns, tsr_record *rec);  // launch + wait
int tsr_transport_cancel(tsr_transport *t, uint32_t seq_upto);
int tsr_transport_stop(tsr_transport *t, uint32_t seq_upto);
int tsr_transport_set_event_cb(tsr_transport *t, void (*cb)(void *, const tsr_record *), void *user);
int tsr_transport_soft_reset(tsr_transport *t);                        // SOFT_RESET, wait SR_BUSY = 0
int tsr_transport_log_drain(tsr_transport *t, uint32_t *words, size_t max, size_t *n);
```

**Removed:** `tsr_transport_exec`, `tsr_exec_result`, the `ARG*` and `CMD_*` handling, and the
per-exec `TSR_TRACE` dump (SW replaces it with a per-launch dump). This API is the only transport
API; SW §9.1 adds convenience functions on top of it (default image, statistics), not a second
protocol.

### 10.2 Device-access layer (`tsr_kdev.h`)

- A kdev keeps `reg_read` / `reg_write` (`s_axi_control` offsets), `mem_write` / `mem_read`,
  `num_ch`, `ch_bytes`, `ch_addr` and `describe`.
- **Removed:** `start`, `wait`, `settle`, `kargs`, `regs_start` / `regs_wait`, `kds_start` /
  `kds_wait`, and `KREG_AP_CTRL` … `KREG_ARG0`.
- **Added:** `KREG_ADAPTER_CTRL` (0x0E0), and an optional `advance(ns)` hint that lets the sim and
  vkernel kinds run device time while the host polls.
- **One protocol layer** (`tsr_kdev.cpp`) implements open, bind, load, boot, launch, wait, cancel,
  stop, reset and log for every kind:
  - `sim`: an in-process simulator exposing the core CSR port, with no adapter. A host read of
    `LCQ_WR`, `STATUS` or `LOG_WR` first runs the model until a record or log word appears, or
    until the RISC-V waits for host input (WFI with `LQ_AVAIL` = 0 and no op running). Results are
    therefore deterministic and independent of host timing.
  - `mock`: sim plus the v2 adapter register model.
  - `vkernel`: the Verilated kernel with two clocks.
  - `xrt`: `xrt::ip` plus per-channel BOs.
- **Environment.** Kept: `TSR_DEVICE`, `TSR_XCLBIN`, `TSR_XRT_DEVICE`, `TSR_XRT_KERNEL`,
  `TSR_XRT_CH_GIB`, `TSR_XRT_DMA*`, `TSR_MOCK_*`. Removed: `TSR_XRT_CTRL`. Renamed:
  `TSR_XRT_TIMEOUT_MS` → `TSR_LAUNCH_TIMEOUT_MS` (default 60,000), `TSR_XRT_ABORT_MS` →
  `TSR_ABORT_MS` (10,000). New: `TSR_IMAGE` (firmware ELF; default the build's base image),
  `TSR_BOOT_MS` (2,000), `TSR_RESET_MS` (15,000, the `SOFT_RESET` wait), `TSR_ATTACH` (open
  without reset or load, for diagnosis).

---

## 11. DRAM map v2 (per channel; `docs/spec.md:130-138`)

| Region | Range | Use |
|---|---|---|
| **OPL** (was CMD) | 0–64 MiB, ch0 | op lists (OPL, M6 and M7), `replay` push streams, argument blocks, RISC-V code overlays |
| IO | 64–320 MiB, ch0 | host ↔ device tensors (inputs, logits, M9 token ring); compute chunks (SW R6) |
| TRW ring (profiling only) | inside IO, ch0 | DB's trace-writer ring (`TSR_PROF_TRW_MIB`, default 64 MiB), allocated by the transport only while profiling; `TRW_CH` = 0 (DB §8.3) |
| AUX | 320–512 MiB, ch0 | small f32 tensors (norm weights, RoPE frequency-factor tables) |
| WEIGHT / heap | 512 MiB to the end, every channel | weights (symmetric allocation, bottom-up from 512 MiB), KV planes (symmetric, top-down from the channel end, AT §4.5), and between them the per-channel compute heap (SW R6, V2-SW-26), sized from the reserve graph |
| low 512 MiB of ch1–3 | | compute chunks and other activations (SW R6); software may spread IO over the channels (`docs/ctrl-riscv.md:765-768`) |

`TSR_REGION_CMD_*` become `TSR_REGION_OPL_*` (`rtl/core/tsr_pkg.sv:42-43`). The core no longer
fetches from DRAM. Memory source 0 (FETCH, `rtl/core/tsr_pkg.sv:76`) is tied off, and the source
ids are not renumbered.

---

## 12. What is deleted from v1, and in which order

There is no compatibility layer and no `HAS_CTRL` parameter: the control core is part of every
v2 core. Tag `v1-m4` and `build/v1` are the only v1 (D16: "v1 is deleted now, not after v2 passes
on the card"). Therefore **all of v1 that has no v2 meaning is deleted in one change, X0, the first
M6 commit**. The v2 replacements then land layer by layer (Y1–Y5). A ctest or bench that depends on
a deleted part is removed in X0 and comes back with its v2 replacement (SW §10.2, §10.3); until
then `build/v1` and tag `v1-m4` are the comparison path.

**X0 (first M6 commit, D16): delete every v1-only part at once.**

| Layer | Deleted in X0 |
|---|---|
| contract | the v1 spec texts listed in the header; ctrl-riscv §3.3–3.4, §6, §8.3–8.4 (replaced by `docs/v2/*`, then the spec v2 rewrite) |
| oracle (`sw/sim`) | `tsr_sim_run`, `tsr_sim_exec`, `START` / `CMD_*` / `ARG*` / `ERR_PC` / `PC` CSR semantics, R0–R15 and `tsr_sim_ireg`, LI / ADDI / END execution, the fetch model and the fetch beats in `CH_RD_BEATS[0]`; the v1 constants of `tsr_isa.h` |
| traces | `traces/m2` and `traces/gen` `cmd.bin` sets in git; `traces/tmp`. `traces/model` stays on disk only as the input of `tsr-trace-conv` (SW §5.6) |
| host software | `tsr_transport_exec`, `tsr_exec_result`, kdev start / wait / settle / kargs, `regs_*`, `kds_*`, `TSR_XRT_CTRL`; the backend's MUL_MAT emission and program cache (`tsr_mm_emit`, `tsr_prog_cache`); `tsr_rt`'s exec calls; `tsr_xrt_selftest`'s program checks |
| core RTL | `tsr_fetch.sv`; `tsr_cmdproc.sv` (its checks are re-written as `tsr_chk` from the tag `v1-m4` source in Y4); `tsr_csr.sv` v1 registers and `done`; the FETCH source use in `tsr_mem_arb`; `tsr_core.sv` keeps only what the surviving unit benches need and does not elaborate as a full core until Y4 |
| benches | `tb/cmdproc`; the command-path parts of `tb/core`, `tb/fastreplay` and `tb/common` (`proggen.py`, `isa.py`, `csr_driver.py`, `replay.py`, the v1 `trace` reader). Unit benches (dma, fp, mvm, mvmback, mvmfront, qact, spm, tech) are kept unchanged |
| platform | `tsr_xrt_ctrl` ap_ctrl_hs FSM, run sequencer, interrupt and arg registers; `kernel_k*.xml` cmd args; the `gen_tops.py` v1 table; the vkernel v1 protocol; `tb/platform` start / done tests. The v2 platform build is disabled until Y5 |

**X0 as done (W0, 2026-10-30).** X0 also deleted code that only the v1 execution path called, and
whole files of which the table names parts. Their v2 users take them back from tag `v1-m4`:

- **Backend** (P3.2, SW §9.3): the lazy conversion of raw-buffer MUL_MAT src0 into the WEIGHT heap
  (`tsr_lazy_*`, `tsr_heap_alloc`'s evict-and-retry), the host-memory staging (`tsr_lower_staged`,
  `tsr_act_loc`) and `tsr_resolve_weight`, from `v1-m4:sw/ggml-tesseract/ggml-tesseract.cpp`.
- **Trace tools** (P1.10, SW §5.6): all of `sw/tools/tsr_trace_gen.c` and the v1 trace library
  `sw/sim/tsr_trace.{c,h}`. `tsr-trace-gen` v2 starts from the tag's generator (random programs,
  the MVM boundary and error generators, the seeds); `tsr-trace-conv` takes its v1 reader from the
  tag's `tsr_trace.c`. The v1 sets are reproduced only with the v1 tools of `build/v1`
  (`traces/*/regen.sh` check them first; m2 and m3 go to `build/v1-traces/`).
- **Platform bench** (Y5): all of `tb/platform/kernel_bench.cpp`, including the directed tests
  that do not depend on the run protocol (regmap: reset values, RW read-back with strobes,
  `ADAPTER_VERSION`, the CSR window; errors: SLVERR / DECERR, `ADAPTER_STATUS` W1C and count
  saturation; misaligned accesses; reset). The Y5 adapter bench carries them over; until then the
  adapter is covered by lint only, and `hw/u200/build_xclbin.sh` refuses k1 / k4.

**Y1–Y5: v2 replacements, in this order.**

| Phase | Layer | Added (with its tests) |
|---|---|---|
| Y1 | oracle (`sw/sim`) | the generated tables (V2-EX-30), the op-level API (check and execute one op under v2 rules), the v2 CSR model (§5), LQ / LCQ / LOG and PERF v2, SOFT_RESET; then SW and CC add the RV32 interpreter, the queue model, the VU and AT ops (SW P1) |
| Y2 | traces | SW §5.6: `traces/m2`, `traces/gen` and `traces/m3` are regenerated as v2 `ops` traces by `tsr-trace-gen` v2 with the same seeds (error expectations re-derived); the 290 `traces/model` traces are converted by `tsr-trace-conv` into `traces/model-v1conv` as serial chains; both replay through the `replay` kernel (§4.6) and are compared with the simulator |
| Y3 | host software | transport v2 (§10), the firmware SDK and base image (SW P2), the backend's OPL builder: the backend emits OPL MUL_MAT records for the `graph` kernel (SW §9.3, V2-SW-14); `tsr-selftest` launches the `selftest` kernel |
| Y4 | core RTL | `tsr_chk` (the checks of the tag's cmdproc, split out), CC's dispatcher and queues, the v2 CSR block (§5), the control core, VU and AT RTL; tile-only replay benches |
| Y5 | platform | adapter v2 (§9), `ap_clk_2`, two-clock vkernel, kernel-bench tests for `MEM_EN` and the bases; `hw/u200/k4_floorplan.tcl` cell lists updated (`u_cmdproc`, `u_fetch` removed at line 120) |

Order rationale: the oracle comes first, because everything is checked against it. Data and the
host follow, so that every RTL step has a replay path and a host: the whole software path runs end
to end on the simulator before any v2 RTL exists (SW P3.4). The platform comes last; it needs the
control core to be useful.

---

## 13. Decisions

Each decision states the choice, the alternatives considered, and the reason. Each was made under
the user's delegation (2026-10-30).

- **V2-EX-1 Launch transport.**
  - *Choice:* a core-domain launch queue written by posted CSR writes, plus a core-domain
    completion ring read through the CSR window.
  - *Alternatives:* ctrl-riscv's single-slot H2R / R2H mailbox; a DRAM launch ring with a
    doorbell; device writes of completions into host memory.
  - *Rationale:* posted writes are the cheapest PCIe operation, and a launch needs no DMA (XDMA
    pays an ioctl and a descriptor setup per transfer). Depth 8 lets the host pipeline launches,
    which the single slot cannot. A DRAM ring would also need an LD into SPM and window reads on
    the device side. Host-memory writes would need a second AXI master path through the shell,
    which v2 does not need for its ≤ 10 µs target.
- **V2-EX-2 One channel for all messages.**
  - *Choice:* LQ / LCQ replace the H2R / R2H mailbox entirely. PING, HELLO, TRAP and kernel events
    are record types.
  - *Alternatives:* keep the mailbox beside the queue.
  - *Rationale:* two channels would double the CDC paths, the deadlock arguments and the tests,
    with no message the queue cannot carry.
- **V2-EX-3 Formats.**
  - *Choice:* 8-word descriptors (type, flags, 16-bit kernel index, 7 args) and 8-word records
    with r0 self-describing.
  - *Alternatives:* 16-word descriptors; ctrl-riscv's 5 args.
  - *Rationale:* 8 words is the size of the existing staging conventions, and 7 args cover the
    known kernels (op list, count, positions, handles). Larger inputs use argument blocks. A
    self-describing r0 lets the host handle a completion with one read.
- **V2-EX-4 In-order answers and one launch at a time.**
  - *Choice:* each descriptor gets exactly one answer, in order. Sequence numbers come from the
    hardware counters. `LQ_POP` is refused while ACTIVE.
  - *Alternatives:* several launches in flight on the device; host-chosen tags.
  - *Rationale:* the runtime is single-threaded. Sequential launches give ACTIVE an exact
    meaning, which the PERF window, cancel and the "current launch" all depend on. Overlap inside
    a launch comes from the asynchronous issue path; overlap across launches costs only the
    1–3 µs dispatch gap (*estimate*).
- **V2-EX-5 Host-side credit.**
  - *Choice:* the transport keeps at most `LQ_DEPTH` launches outstanding. `LCQ_DEPTH` =
    `LQ_DEPTH` = 8.
  - *Alternatives:* read `LQ_FREE` before each push.
  - *Rationale:* every outstanding launch occupies at most one LQ slot and produces exactly one
    answer, so neither ring can overflow and no read is needed on the push path.
- **V2-EX-6 XRT control.**
  - *Choice:* `ap_ctrl_none`, `xrt::ip` register access, polling, no interrupts on the U200. The
    core still offers `host_irq_o` for other targets.
  - *Alternatives:* `ap_ctrl_hs` with one `xrt::run` per launch (as v1 per exec), or a run wrapped
    around a resident runtime.
  - *Rationale:* a debug-halted RISC-V under an outstanding run breaks XRT's scheduler, and an XRT
    run costs ≈ 35 µs (measured). XRT supports `ap_ctrl_none` only through `xrt::ip`
    (`xrt_kernel.cpp:1561`), which is exactly the register-level path needed. A fallback exists
    without RTL changes (§9.1).
- **V2-EX-7 Channel bases.**
  - *Choice:* the host writes the bases, which are latched on the rising edge of `MEM_EN` and
    locked while it is set. Requests while `MEM_EN` = 0 are held.
  - *Alternatives:* latch on every base write; fixed bases from the xclbin.
  - *Rationale:* there is no run start to latch on any more. One explicit edge keeps v1's
    atomicity (all bases change together and are alignment-checked) and lets a new process rebind
    its BOs.
- **V2-EX-8 Image load path.**
  - *Choice:* the CSR loader with a receive CRC plus an on-device read-back CRC (VERIFY).
  - *Alternatives:* a DMA path from DDR into the TCMs; a boot ROM that copies an image from DRAM
    through SPM; host read-back.
  - *Rationale:* the load takes ≤ 10 ms once per process (*estimate*). A TCM DMA path or a boot
    ROM is hardware for no measurable gain on the U200. Host read-back costs about 32 K PCIe reads
    (≈ 30–60 ms, *estimate*), while VERIFY checks the same thing with no reads.
- **V2-EX-9 Image identity.**
  - *Choice:* a 64-bit prefix of a SHA-256 over the loaded segments, stored in `.tsr_id` and
    echoed in HELLO.
  - *Alternatives:* the full SHA-256 computed on the device; a build counter.
  - *Rationale:* the RISC-V cannot hash 128 KiB cheaply, and a counter does not identify content.
    Integrity is the CRC's job.
- **V2-EX-10 Warm restart.**
  - *Choice:* `SOFT_RESET`, reload DTCM, VERIFY ITCM (reload only on a mismatch), boot.
  - *Alternatives:* always cold load; restart with no reload.
  - *Rationale:* restarting without a reload would run with stale `.data`. A cold load costs ≈ 10×
    more. ITCM is writable, so it is checked rather than trusted.
- **V2-EX-11 CSR map.**
  - *Choice:* device control at 0x020 / 0x024. The planned `CTL_*` block is dissolved into
    0x000–0x0FC. `CTL_PERF` is merged into `PERF_*`. The 12-bit CSR space is kept, with the DBG
    range grown to 768 B.
  - *Alternatives:* keep `CTRL` / `STATUS` for v1-style bits and add `CTL_CTRL` / `CTL_STATUS`;
    grow the CSR space to 13 bits.
  - *Rationale:* without host runs, two control registers would only duplicate each other. The
    map fits 4 KiB with 1.25 KiB still reserved, so `s_axi_control` and the CSR port stay as they
    are.
- **V2-EX-12 Identity.**
  - *Choice:* `ID` stays "TSR1"; `VERSION` becomes 2.0; `ADAPTER_VERSION` becomes 2.
  - *Alternatives:* a new magic "TSR2".
  - *Rationale:* `ID` names the core family and `VERSION` carries the ISA. Existing open code
    already fails cleanly on a major mismatch (`docs/xrt-host.md:72-74`).
- **V2-EX-13 PERF semantics.**
  - *Choice:* one counter space, gated by ACTIVE (time counters also by ¬QHALT). `SOFT_RESET`
    zeroes it. New split counters `MVM_WAIT` and `MVM_EMPTY`.
  - *Alternatives:* free-running counters; a software-set gate bit.
  - *Rationale:* ACTIVE restores the v1 property "PERF does not change between executions" per
    launch, and it is defined by the protocol, not by a bit the runtime could forget. The split
    counters measure why the MVM is idle, which is the first-priority metric.
- **V2-EX-14 Register fields.**
  - *Choice:* w0[31:16] becomes the op-specific field `x16`, which must be 0 for v1-encoded ops.
  - *Alternatives:* keep "must be 0" for every op; drop the field.
  - *Rationale:* the check for existing ops is the same as ctrl-riscv's ra–rd = 0 rule, so they
    are bit-identical. VU and AT get 16 bits of encoding space they asked for (strides, head
    counts) without a new instruction width.
- **V2-EX-15 Opcode allocation.**
  - *Choice:* blocks next to the existing classes: VU 0x13, 0x38–0x3F and 0x41–0x47; AT 0x14–0x17,
    0x19–0x1B, 0x21–0x23 and 0x34–0x37; SYNC 0x0A–0x0F and 0x70–0x7E. Retired opcodes, including
    CALL, are never reused. Assigned in M6: KVST 0x14, ATTQK 0x21, ATTPV 0x22, ASMAX–ANORM
    0x34–0x37 (AT), VMASK 0x38 (VU, M9).
  - *Alternatives:* allocate on demand.
  - *Rationale:* fixed blocks let the areas work in parallel without collisions. Keeping each block
    beside its class keeps decode by class cheap.
- **V2-EX-16 Unit id 4.**
  - *Choice:* the CMD unit is gone and id 4 goes to AT, reserved for a future attention engine.
    It is absent in M6: AT's ops run on units 0, 2 and 3 (V2-AT-16).
  - *Alternatives:* leave 4 reserved and use 6.
  - *Rationale:* CALL never shipped, so no ticket with unit 4 exists anywhere. Reusing it keeps the
    unit ids dense (0–5) and the six-unit CDC widths of CC's plan.
- **V2-EX-17 Error codes.**
  - *Choice:* add 13 `OPERAND`. Its check comes after `T_RANGE` and before the address checks.
    Codes 9, 10, 14 and 15 stay reserved.
  - *Alternatives:* fold operand-value errors into `ILLEGAL_OP` or `K_RANGE`.
  - *Rationale:* VU and AT ops have value limits that are neither encoding errors nor K. A distinct
    code with the value in `ERR_INFO` makes failures diagnosable.
- **V2-EX-18 Launch result.**
  - *Choice:* the status codes and precedence of §4.4; the first error is selected by `ERR_PIDX`.
  - *Alternatives:* report only the kernel's return value.
  - *Rationale:* tile failures must never read as success, and the selection must not depend on
    the schedule. That is exactly ctrl-riscv's first-error rule.
- **V2-EX-19 Platform errors.**
  - *Choice:* a `plat_err_i` core input, reflected in `STATUS` and in the launch status.
  - *Alternatives:* the host reads `ADAPTER_STATUS` after every launch, as v1 did per exec.
  - *Rationale:* this keeps v1's guarantee ("an exec fails on an adapter error") without a PCIe
    read per launch.
- **V2-EX-20 Cancel and stop.**
  - *Choice:* sequence watermarks. Hard cancel applies a hardware QABORT to the running launch when
    it is covered; soft stop is polled by kernels.
  - *Alternatives:* host QABORT only; `SOFT_RESET` for every cancel.
  - *Rationale:* a bare QABORT cannot target one launch and leaves queued launches to run. A soft
    reset discards the runtime. Watermarks are precise, race-free (they are compared against the
    hardware sequence), and also give M9's graceful stop.
- **V2-EX-21 Traps.**
  - *Choice:* the handler posts TRAP and parks. Recovery is `SOFT_RESET` plus a warm restart.
  - *Alternatives:* unwind to the dispatcher and continue.
  - *Rationale:* a trap may leave `.data` inconsistent. Parking keeps the state for GDB (D18).
- **V2-EX-22 `SOFT_RESET`.**
  - *Choice:* a hardware sequence: RISC-V reset, drain, clear, flush, PERF zero. It works without
    `clk_rv`.
  - *Alternatives:* keep v1's meaning (only the command-buffer path, which no longer exists); a
    host-scripted sequence.
  - *Rationale:* the host needs one atomic "known state" operation that cannot be raced by the
    RISC-V and does not hang when its clock is absent.
- **V2-EX-23 Stuck units.**
  - *Choice:* no forced unit reset. A progress watchdog is used for diagnosis, and a card reset for
    recovery.
  - *Alternatives:* a per-unit kill.
  - *Rationale:* AXI has no cancel, so a unit kill corrupts later transactions
    (`docs/platform-u200.md:201-212`). No stuck unit has been seen on the card in M3 or M4.
- **V2-EX-24 Log.**
  - *Choice:* a 256-word core-domain ring read through the CSR window. It drops and counts when
    full, with an optional BLOCK mode. Records use DB's header (DB §6.1); the runtime filter is
    `LOG_CFG.LEVEL`.
  - *Alternatives:* a DRAM log ring written by ST; the LCQ.
  - *Rationale:* the log must work when the DMA queue is wedged, which is when it matters most.
    Logging must never fault or stall the device by default. The LCQ is reserved for protocol
    records.
- **V2-EX-25 One protocol implementation.**
  - *Choice:* all transport kinds, including sim, run the same register protocol. The sim advances
    deterministically on host reads.
  - *Alternatives:* a direct API for sim, as in v1.
  - *Rationale:* the protocol itself becomes part of what the oracle checks, and sim, mock,
    vkernel and the card exercise identical host code.
- **V2-EX-26 DRAM map.**
  - *Choice:* CMD becomes OPL; there is no fetch region.
  - *Alternatives:* keep CMD for op lists.
  - *Rationale:* the name follows the content; the addresses stay where they are.
- **V2-EX-27 FETCH memory source.**
  - *Choice:* tied off, with the ids not renumbered.
  - *Alternatives:* reduce `TSR_NSRC` to 2.
  - *Rationale:* `tsr_mem_arb` and the memory id format sit on timing-critical paths. A constant
    input is removed by synthesis with no RTL churn.
- **V2-EX-28 Deletion order.**
  - *Choice:* X0, the first M6 commit, deletes every v1-only part of every layer at once (§12).
    The v2 replacements follow in the order oracle → traces → host → RTL → platform (Y1–Y5); tests
    that depend on deleted parts are removed in X0 and return with their replacements.
  - *Alternatives:* delete each layer's v1 part in the same change as its replacement (the
    draft's choice).
  - *Rationale:* D16 says v1 is deleted now, not after v2 passes on the card. Staged deletion would
    keep `tsr_cmdproc`, `tsr_fetch` and the exec path in the tree, passing CI, until the end of the
    M6 hardware work. `build/v1` and tag `v1-m4` keep a bit-exact comparison path meanwhile.
- **V2-EX-29 M6 offload path.**
  - *Choice:* one mechanism from M6 to M7: the backend emits OPL op lists with MUL_MAT records
    only, one `graph` launch per v1 split (V2-SW-14). Tile-only traces, the converted v1 model
    traces and SPM dumps run on the `replay` kernel; its SERIAL flag gives v1's sequential
    semantics.
  - *Alternatives:* keep v1's lowering and run absolute-address op arrays through a dedicated
    `tsr.oplist_seq` kernel (the draft's choice); M6 C++ kernels per ggml op.
  - *Rationale:* `oplist_seq` costs ≈ 1 µs of RISC-V issue time per op (*estimate*) and would be a
    second offload path to delete in M7. The `graph` executor issues from pre-resolved records with
    host-computed dependencies, so units overlap from M6 on, and M7 only adds kinds.
- **V2-EX-30 One table source.**
  - *Choice:* every CSR offset and field (host map and RISC-V control window), opcode, unit, error
    code and PERF index is generated from one plain-Python data file, `sw/tables/tsr_tables.py`
    (no dependencies). SW owns the generator `sw/tables/gen_tables.py` (P1.1); EX, CC, DB, VU and
    AT own their rows. It emits `rtl/core/tsr_tables_pkg.sv`, `sw/sim/include/tsr_tables.h`
    (included by `tsr_isa.h`), `sw/ctrl/include/tsr/{mmio,vu,at}.hpp` and `tb/common/tables.py`,
    with `--check` in lint.
  - *Alternatives:* hand-kept copies; a YAML table under `sw/sim/regs` (SW's draft); a generator
    under `rtl/core/ctrl` (CC's draft).
  - *Rationale:* v1 kept `tsr_pkg.sv` and `tsr_isa.h` in step by hand, and v2's tables are larger.
    ctrl-riscv already asked for a generated table (`docs/ctrl-riscv.md:313-315`). Plain Python
    needs no PyYAML in the RTL, firmware or bench builds, and a neutral directory reflects that the
    RTL, the simulator, the firmware and the benches all consume it.

---

## 14. Interfaces with other areas

| Area | EX provides | EX assumes |
|---|---|---|
| **VU** | Opcodes 0x12, 0x13, 0x30–0x33, 0x38–0x3F, 0x40, 0x41–0x47; unit 3; the `x16` field; error code 13 `OPERAND` and the check order; `CAPS` bits 0, 3, 6, 7 and 8–11; `CFG3[15:0]`; PERF indices 3 and X + 13 … X + 15; one s-register bank (§7.4) read at `SREG` 0x9C0 | VU fixes encodings, units, dispatch checks and numeric rules inside these numbers; it has no run-time errors (§7.2); DQROW runs on the VU (VU §5.6); it defines the s-bank initial value (hardware reset +0.0, unchanged by `SOFT_RESET`); it executes AT's vector-class ops 0x34–0x37 (VU §5.11) |
| **AT** | Opcodes 0x14–0x17, 0x19–0x1B, 0x21–0x23, 0x34–0x37; unit id 4 reserved (absent in M6); `CAPS` bits 5 and 12–15; `CFG3[31:16]`; PERF X + 16 … X + 19; the DRAM heap for the KV cache (§11); watchdog coverage of the units its ops use | AT fixes encodings, units and checks; its ops are fully checked at dispatch; data-dependent lengths are clamped, not errors; it defines the KV layout and its placement through the backend allocator |
| **CC** | The static opcode → unit table (generated, V2-EX-30); the RISC-V-side LQ / LCQ / LOG / RT registers at control window 0x200–0x2FC (§4.5) and `irq_fast[2]` / `[3]`; the runtime contract R1–R10; `RV_BOOT`, `RV_FREQ`, `RV_CFG`, `CTL_ID` and `Q_DEPTH` offsets; the `SOFT_RESET`, RISC-V reset and QABORT semantics (§8.5); ACTIVE and the PERF gates; PERF X + 20, X + 21 | CC implements the rings' CDC by tagged handshakes (§4.1, lag never leads), the core-domain SR sequencer of `SOFT_RESET` (works without `clk_rv`); the issue path, tickets, dependencies, error records, I1 and I2 as in ctrl-riscv §4; `ISSUE_PIDX`; the RISC-V reset sequence setting ANSWERED := TAKEN; the loader's TCM write and VERIFY read port; Ibex, TCMs, clocks and the bus rules |
| **DB** | CSR range 0x800–0xAFC for `DBG_*`; DMI 0xC00–0xDFC; `HOLD_ON_HALT`; QHALT gating of time counters; `Q_*` and `RT_*` for post-mortem; the log ring with DB's record header; PERF 61–63; `tsr_transport_csr_write`; the in-process ownership rule (§2.4); the debug-aware wait timeout (§4.8) | DB defines `DBG_*` contents (tile-level debug per queue), the GDB RSP server inside the owning process over DMI, the halt rules on top of `HOLD_ON_HALT`, the SPM access rule through SBA, the log record header and the trap contract |
| **SW** | The transport API and kdev layer (§10); descriptor and record formats; image format, image id and load procedure; base-kernel contracts (`graph`, `replay`, `selftest`, §4.6); DRAM map; the D16 deletion order (§12); the table source format (V2-EX-30) | SW provides the SDK runtime implementing R1–R10, `tsr-elf`, the base image, the simulator models of every EX register and ring (deterministic advance), the OPL format, the v2 trace format and the trace migration (SW §5.6), `tsr-selftest`, and the table generator |

---

## 15. Verification

The oracle chain stays: the simulator is bit-exact for everything EX defines, and the RTL and the
card must match it.

1. **Generated tables.** `gen_tables.py --check` in lint. A directed test reads every core CSR
   offset after a scripted sequence on sim, mock and vkernel. The values must be identical except
   for the timing-dependent set: `PERF` time counters, `RV_FREQ`, `RV_SLEEP`, MTIME-derived record
   word r7, and log payloads that a test marks as timing.
2. **Protocol conformance** (`sw/transport/tests`, one binary, every kind). It covers:
   - boot: cold and warm load, HELLO, image-id mismatch, VERIFY with an injected TCM word flip
     (vkernel force), `.note.tsr` refusal;
   - launches: PING, LAUNCH with 0–7 args, BAD_DESC cases, 8 outstanding plus the credit rule,
     forced `LQ_OVERFLOW`, `RING_RD_BAD`;
   - kernel events while the host waits, the log framing, drop and BLOCK;
   - cancel of queued and running launches, stop;
   - `SOFT_RESET` mid-launch, RISC-V reset mid-launch (ANSWERED := TAKEN).

   Records r0–r6, DRAM and the log streams must be identical across sim, mock and vkernel.
3. **Error model.** An `errgen` firmware kernel pushes each error code (1–8, 11–13) and the
   propagation cases (error state, `DEP_FAILED` chains, first error by push index across two
   units). The DONE records and `Q_ERR_*` must be bit-identical to the simulator. CC's random
   schedules (8 or more seeds) check that the first error is schedule-independent.
4. **Trace corpus** (SW §5.6). The 290 `traces/model` traces are converted to
   `traces/model-v1conv` (serial chains) and replayed through `replay` with SERIAL on sim and
   vkernel: the expect regions must equal the v1 expectations, which checks that the units' results
   did not change from v1 to v2. `traces/m2`, `traces/gen` and `traces/m3` are regenerated as v2
   `ops` traces with the same seeds; their DONE records and error records must equal the
   simulator's.
5. **PERF.** Exact sim = RTL for `RETIRED`, `CH_RD_BEATS[c]` (now including ch0), `PUSHES` and
   `LAUNCHES`. The RTL asserts the bound on the MVM remainder of §6 (`CYCLES` − `MVM_CYCLES` −
   `MVM_WAIT` − `MVM_EMPTY` ≤ 3 per MVM-queue entry). Time counters are not compared, as today. The counters must not change while
   ACTIVE = 0.
6. **RTL directed tests** (cocotb / Verilator, two clocks at every ratio of ctrl-riscv §9.3):
   - every CSR access completes with `clk_rv` stopped or held in reset;
   - `SOFT_RESET` without `clk_rv` (`SR_RV_TIMEOUT`);
   - the cancel watermark at the exact ACTIVE edges (before pop, during, after answer);
   - the `LQ_POP` / `LCQ_POST` faults;
   - the watchdog: trip with a stalled memory model, no trip on a 3.2 M-beat MVM;
   - mutation campaign: answer without join, flush omitted, watermark compare off by one, ACTIVE
     from the wrong counter, PERF gate inverted.
7. **Adapter.** Kernel-bench tests for `MEM_EN` (latch, misaligned refusal, `BASE_LOCKED`,
   `REQ_HELD`), `plat_err_i` and the window; `mutate_adapter.py` extended.
8. **Card.** `tsr-selftest`: identity, bind, load and VERIFY, PING, launch round trip and
   pipelined rate (they set the §4.11 targets), the `selftest` kernel's tests, the trace corpus
   through `replay`. Then llama.cpp end to end with `offload_kqv` = false (SW §10.4): 0.5B Q8_0
   logits bit-identical to `TSR_DEVICE=sim` (`klcheck compare --expect-identical`), and the 32B
   KL criterion (M6 gate).

---

## 16. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| v++ rejects `ap_ctrl_none` with `m_axi` args, or ignores its connectivity | no v2 xclbin | the fallback in §9.1 (declare `ap_ctrl_hs`, never run), checked in the first smoke link |
| PCIe read latency is higher than assumed (≥ 3 µs) | round trip above 10 µs | still ≤ 1/3 of v1's exec cost; M7 makes one launch per token; the poll policy reads once per completion; `host_irq_o` remains for platforms with interrupts |
| CDC or flush bugs in the rings or `SOFT_RESET` | a hang that needs a card reset | core-domain storage, two-clock benches at all ratios, mutation campaign, `SOFT_RESET` without `clk_rv` tested |
| Runtime contract violated (answer before join, kernel posts an answer) | wrong PERF windows, H2 broken | hardware faults on illegal pops and posts; debug builds assert QIDLE before the answer; the simulator checks R6 |
| The `graph` executor's issue rate (SW §7.8: ≤ 400 RISC-V cycles per graph op, *estimate*) limits short-op phases | the device idles between short ops | the line-buffered SPM alias (CC §10.5) for record reads; compact records; measured with `MVM_EMPTY` and TRACE_MARK records; CC may add an "issue from SPM line" path later |
| Exclusive context blocks external debuggers | no second process | in-process GDB server (DB) |
| 4 KiB CSR space exhausted by later VU, AT or DB needs | redesign | 1.25 KiB reserved; growing to a 13-bit CSR port is a contained change (VERSION minor) |
| Deleting v1 before v2 works leaves no working path in the repo | no card regression until M6 | D16 accepted it; `build/v1` binaries and tag `v1-m4` run models meanwhile; the software path runs end to end on the simulator first (Y1–Y3) |
| Watchdog false positive on DRAM stalls (refresh, contention) | a spurious `UNIT_HUNG` | progress-based, with a 61 ms default limit, far above any DDR stall; diagnosis only, no action |

---

## 17. Open items

Only measurements and third-party checks remain. Every design choice above is decided.

- *confirm* that v++ 2024.2 accepts `ap_ctrl_none` with `s_axi_control` and `m_axi` args (first v2
  smoke link), else apply the §9.1 fallback.
- Measure on the card: MMIO posted-write and read latency, the empty-launch round trip, the
  pipelined launch rate, cold and warm load times. These replace the estimates of §3.3, §3.5 and
  §4.11. The spec open item on PCIe CSR latency (`docs/spec.md:994`) closes with them.
- Re-check `LQ_DEPTH` (8) against M7's launch pattern and M8's JIT launch rates; the parameter can
  change without a format change, up to the 8 slots of the CSR ring.
- Possible CC follow-up: an "issue from SPM line" path, if the `graph` executor's per-op cost
  limits short-op phases (measured in P3/P6).

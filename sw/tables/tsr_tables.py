"""sw/tables/tsr_tables.py — the single table source of Tesseract v2 (V2-EX-30, work package P1.1).

Plain Python data: literals only, no imports, no code. sw/tables/gen_tables.py reads this file and
emits rtl/core/tsr_tables_pkg.sv, sw/sim/include/tsr_tables.h, sw/ctrl/include/tsr/mmio.hpp,
vu.hpp, at.hpp and tb/common/tables.py; sw/tables/test_tables.py checks it. Change rows here
and regenerate; never edit a generated file.

Owners (docs/v2/README.md §4): EX owns the host CSR map, PERF, opcodes, units, error codes and the
launch formats; CC the RISC-V address map, the control window, queue depths and the issue FIFO;
DB the DBG block, its control-window rows, the log header and the REC / TRW / marker formats; VU
and AT the encodings of their ops; SW the generator. When two documents disagree, the row follows
the owner; the disagreement is reported with the P1.1 change, not resolved here.

Citations: every row carries `src`, the document section it comes from. Areas: EX = docs/v2/
01-execution-model.md, VU = 02-vector-unit.md, AT = 03-attention-kv.md, CC = 04-control-core.md,
DB = 05-debug.md, SW = 06-software-verification.md, spec = docs/spec.md. A group whose rows all come
from one section (a bit-field layout, an encoding, a word table) carries `src` once for the group;
a row may add its own.

Notation, as in the documents:
    bits     "msb:lsb" or "bit"                         (bit-field layouts)
    enc      "w<i>[msb:lsb]", "w<i>[bit]", "w<i>" (whole word), "x16[msb:lsb]" (x16 bit i is
             w0 bit 16 + i, EX §7.1)                     (instruction fields; opcode = w0[7:0])
    access   R, W, RW, R/W1C (sticky bits cleared by writing 1), RW/W1 (level bits + pulse bits),
             R/WC (any write clears)
Names are upper case without the TSR_ prefix; the generator adds the per-language prefix.
"""

DOCS = {
    "EX": "docs/v2/01-execution-model.md",
    "VU": "docs/v2/02-vector-unit.md",
    "AT": "docs/v2/03-attention-kv.md",
    "CC": "docs/v2/04-control-core.md",
    "DB": "docs/v2/05-debug.md",
    "SW": "docs/v2/06-software-verification.md",
    "spec": "docs/spec.md",
}

# =================================================================================================
# Scalar constants: (name, value, src, description)
# =================================================================================================
CONSTANTS = [
    # Identity and versions
    ("ID_MAGIC",              0x5453_5231, "EX §5.1",  'ID register: "TSR1" (unchanged from v1)'),
    ("ISA_MAJOR",             2,           "EX §7.5",  "VERSION[15:8]"),
    ("ISA_MINOR",             0,           "EX §7.5",  "VERSION[7:0]"),
    ("VERSION_VALUE",         0x0000_0200, "EX §7.5",  "VERSION register: ISA 2.0"),
    ("CTL_VERSION",           2,           "EX §5.1",  "CTL_ID[7:0]: control-core version"),
    ("ADAPTER_MAJOR",         2,           "EX §9.2",  "ADAPTER_VERSION[31:16] (ADAPTER_VERSION_MAJOR field)"),
    ("DBG_VERSION",           2,           "DB §10.1", "DBG_ID[7:0]; DBG_ID = 0 means the DBG block is absent"),
    ("CANON_NAN",             0x7FC0_0000, "VU §4.1",  "the only NaN an fp result has"),
    # Instruction format
    ("INSN_WORDS",            8,           "EX §7.1",  "an instruction is 256 bits, w0..w7, little-endian"),
    ("INSN_BYTES",            32,          "EX §7.1",  ""),
    ("INSN_BITS",             256,         "EX §7.1",  ""),
    ("X16_LSB",               16,          "EX §7.1",  "x16 = w0[31:16]; x16 bit i is w0 bit 16 + i; 0 unless the op defines it"),
    ("X16_W",                 16,          "EX §7.1",  ""),
    # Launch protocol
    ("LQ_DEPTH",              8,           "EX §4.2",  "U200 launch-queue depth (LQ_CFG[7:0])"),
    ("LCQ_DEPTH",             8,           "EX §4.2",  "U200 completion-ring depth (LQ_CFG[15:8])"),
    ("LOG_DEPTH",             256,         "EX §4.2",  "U200 log-ring words (LQ_CFG[27:16])"),
    ("LCQ_DEPTH_MAX",         8,           "EX §4.2",  "the CSR map limits LCQ_DEPTH to 8"),
    ("LOG_DEPTH_MAX",         256,         "EX §4.2",  "the CSR map limits LOG_DEPTH to 256"),
    ("LCQ_REC_BYTES",         32,          "EX §4.2",  "record k at LCQ_REC + 32 x (k mod LCQ_DEPTH)"),
    ("ARGBLK_MAX_BYTES",      1024,        "EX §4.3",  "argument block limit (ARG0 handle of a kernel with the argument-block flag)"),
    ("ARGBLK_MAX_LINES",      16,          "SW §6.2",  "argblk_lines of a kernel table entry: 1..16"),
    ("SEQ_HALF",              0x8000_0000, "EX §4.9",  "launch q is covered by watermark s iff (s - q) mod 2^32 < SEQ_HALF"),
    ("PANIC_MCAUSE",          24,          "EX §4.7, DB §6.3", "tsr::panic enters the trap handler with this mcause, mtval = code"),
    # Firmware image, load and boot
    ("BOOT_ENTRY_OFF",        0x80,        "EX §3.4",  "the RISC-V fetches at RV_BOOT + 0x80"),
    ("RV_BOOT_ALIGN",         256,         "EX §5.1",  "RV_BOOT alignment ([7:0] read 0)"),
    ("CRC32_POLY",            0xEDB8_8320, "EX §3.3",  "LOAD_CRC: zlib crc32 (reflected polynomial, init and final XOR 0xFFFFFFFF)"),
    ("LOAD_TIMEOUT_CYCLES",   4096,        "EX §3.3",  "core cycles a LOAD_DATA word or a VERIFY read may take before LOAD_TIMEOUT"),
    ("LOGFMT_VMA",            0xF000_0000, "SW §6.2, DB §6.1", "VMA of the non-loaded .tsr_logfmt section (FMT log records point into it)"),
    # Bounds and watchdog
    ("WDOG_L_RESET",          24,          "EX §5.1, §8.6", "WDOG_CFG reset value: 2^24 core cycles without progress"),
    ("SR_RV_WAIT_CYCLES",     65536,       "EX §8.5",  "SOFT_RESET step 1 waits at most 2^16 core cycles for RV_RESET_DONE"),
    ("DMI_TIMEOUT_CYCLES",    4096,        "EX §5.1",  "core cycles a DMI access may take before DMI_TIMEOUT"),
    ("BRIDGE_TIMEOUT_CYCLES", 4096,        "CC §5.3",  "clk_rv cycles a core-window or SPM-bridge load may take before a load fault"),
    ("ISSUE_WAIT_MAX",        65536,       "CC §7.3",  "clk_rv cycles a push without credit waits before a store fault and PUSH_TIMEOUT"),
    # Tickets, units, queues
    ("TICKET_NONE",           0xF000_0000, "CC §5.1, EX §4.4", "unit field 15: no dependency / no failing ticket"),
    ("NUM_CH_MAX",            16,          "spec ISA", "LD / ST name the channel in the 4-bit sub field (CFG0.NUM_CH, CH_BASE_LO/HI)"),
    ("NUM_UNITS",             6,           "EX §7.3",  "unit ids 0..5 (6..13 reserved, 14 marker, 15 none)"),
    ("UNIT_SLOTS",            8,           "EX §5.4",  "per-unit register slots u = 0..7 (absent units read 0)"),
    ("Q_DEPTH_FIELD_W",       5,           "EX §5.1",  "Q_DEPTH: depth of unit u at [5u+4:5u]"),
    ("CREDIT_FIELD_W",        4,           "CC §5.1",  "ISSUE_CREDIT: unit u at [4u+3:4u], saturating at 15"),
    ("ISSUE_FIFO_DEPTH",      32,          "CC §7.2",  "issue-FIFO entries (LUTRAM, clk_rv -> core)"),
    ("ERR_W",                 4,           "EX §8.1",  "internal error-code width (codes 0..15); CSR fields are 8 bits"),
    ("PERF_SLOTS",            64,          "EX §5.1",  "PERF_0..PERF_63, 64-bit each"),
    ("PERF_CH_BASE",          5,           "EX §6",    "CH_RD_BEATS[c] = 5 + c, CH_WAIT[c] = 5 + C + c, X = 5 + 2C"),
    # RISC-V side
    ("ITCM_BYTES",            128 * 1024,  "CC §4.1",  "U200 ITCM (RV_CFG[15:0] in KiB)"),
    ("DTCM_BYTES",            64 * 1024,   "CC §4.1",  "U200 DTCM (RV_CFG[31:16] in KiB)"),
    ("SPM_BYTES_U200",        4 * 1024 * 1024, "CC §4.1", "U200 SPM size: the SPM windows span SPM_BYTES"),
    # Debug parameters (U200)
    ("DBG_NTBP",              4,           "DB §10.3", "tile breakpoints"),
    ("DBG_NRVWP",             2,           "DB §10.3", "RISC-V data watchpoints"),
    ("HIST_DEPTH",            512,         "DB §10.3", "dispatch-history entries"),
    ("REC_DEPTH",             64,          "DB §10.3", "completion-record FIFO entries"),
    ("REC_IRQ_LEVEL",         32,          "DB §8.2",  "irq_fast[4] while REC_COUNT >= 32 (REC_CFG.IRQ_HALF)"),
    ("MATCHQ_DEPTH",          64,          "DB §5.2",  "tile-breakpoint match queue"),
    ("TRW_CH",                0,           "DB §10.3, §8.3", "channel of the trace-writer ring (in the ch0 IO region)"),
    ("TRW_RECORD_BYTES",      32,          "DB §8.3",  "trace-writer record, two per 64-byte beat"),
    ("TRW_RING_DEFAULT_MIB",  64,          "DB §8.3, EX §11", "TSR_PROF_TRW_MIB default"),
    ("DBG_INFO_MAGIC",        0x4742_4454, "DB §6.3",  '__tsr_dbg word 0: "TDBG"'),
    ("CRASH_MAGIC",           0x5041_5254, "DB §6.3",  '__tsr_crash word 0: "TRAP"'),
    ("CRASH_LOG_WORDS",       44,          "DB §6.1",  "TRAP log record payload: words 0..43 of __tsr_crash"),
    # VU
    ("VL",                    16,          "VU §5.10", "CFG2.VL: elements per SPM line"),
    ("NSREG",                 16,          "VU §4.4",  "s0..s15"),
    ("VU_FRONT_LANES",        16,          "VU §5.10", "CFG3[4:0] (U200)"),
    ("VU_BACK_LANES",         8,           "VU §5.10", "CFG3[9:5] (U200)"),
    ("VU_LINE_FIELD_W",       24,          "VU §5.1",  "SPM line addresses and element counts m are w[23:0] with w[31:24] = 0"),
    ("ROPE_MAX_HEAD_DIM",     1024,        "VU §5.4",  "head_dim is a multiple of 16 in [16, 1024]"),
    ("TOPK_MAX_K",            16,          "VU §5.5",  "k in 1..16"),
    ("VU_ACCEPT_AHEAD",       2,           "VU §7.8, CC §9.4", "VU ops held at once (I3 for unit 3)"),
    # AT
    ("M_FLOOR",               0xF180_0000, "AT §6.3",  "-2^100: row-maximum floor of ATTQK MAXOUT and AMASK"),
    ("ANORM_L_FLOOR",         0x0080_0000, "AT §5.7",  "2^-126: ANORM divides by MAXT(l, 2^-126)"),
    ("ATT_GROUP_CELLS",       16,          "AT §4.2",  "KV cells per group (groups round-robin over the channels)"),
    ("ATT_MAX_GROUPS",        4095,        "AT §5.2",  "n_groups in [1, 4095]"),
    ("ATT_MAX_G",             64,          "AT §5",    "G (heads per KV head) in 1..64"),
    ("ATT_MAX_PASS_ROWS",     128,         "AT §5.8",  "(E/D) x GS x T_q pass rows at most"),
    ("ATT_MAX_R",             1024,        "AT §5.8",  "R in [1, 1024] for ASMAX, AMASK, AMERGE, ANORM"),
    ("ATT_MAX_LINES",         4095,        "AT §5.4",  "ASMAX / AMASK n_lines in [1, 4095]"),
    ("KVST_MAX_ROWS",         64,          "AT §5.1",  "n_rows in 1..64"),
    ("PCAP_U200",             9344,        "AT §5.8",  "p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX) x 16"),
]

# =================================================================================================
# Shared value sets: name -> (src, [(value, NAME, description)])
# =================================================================================================
ENUMS = {
    "FMT":   ("spec ISA, VU §5.6", [(0, "Q8_0", "weight format Q8_0 (MVM, DQROW sub)"),
                                    (1, "Q4_K", "weight format Q4_K; 2..15 BAD_FMT")]),
    "QA":    ("spec ISA",          [(0, "A32", "QACT kind: block_q8_0 rules, block 32"),
                                    (1, "A256", "QACT kind: block_q8_K rules, block 256, with bsums")]),
    "KVFMT": ("AT §5.2",           [(0, "Q8_0", "KV format of ATTQK / ATTPV sub; else BAD_FMT")]),
    "DCODE": ("AT §5",             [(0, "D64", "head dimension D = 64 x 2^dcode"), (1, "D128", ""),
                                    (2, "D256", ""), (3, "D512", "")]),
}

# =================================================================================================
# Bit-field layouts of 32-bit words: name -> {src, desc, fields: [(bits, NAME, desc)], values:
# {FIELD: [(value, NAME, desc)]}, width (default 32), retired: [(bit, NAME, desc)]}.
# A register uses the layout of its own name unless the register row names another one.
# =================================================================================================
LAYOUTS = {
    # ---- instruction word w0 (EX §7.1) -------------------------------------------------------
    "W0": dict(src="EX §7.1", desc="instruction word w0 = {x16, flags, sub, opcode}", fields=[
        ("7:0",   "OPCODE", "opcode"),
        ("11:8",  "SUB",    "op-specific sub code"),
        ("15:12", "FLAGS",  "op-specific flags"),
        ("31:16", "X16",    "op-specific field; must be 0 (ILLEGAL_OP) where the op does not assign it"),
    ]),
    # ---- identity and configuration (EX §5.1, §7.5) -------------------------------------------
    "VERSION": dict(src="EX §5.1, §7.5", desc="ISA version", fields=[
        ("7:0",  "MINOR", "0"), ("15:8", "MAJOR", "2"),
    ]),
    "CFG0": dict(src="EX §5.1", desc="unchanged from v1", fields=[
        ("7:0", "NUM_CH", "channels C"), ("19:8", "LANES", "MAC lanes"), ("23:20", "T_MAX", "token slots"),
    ]),
    "CFG2": dict(src="EX §5.1", desc="unchanged from v1", fields=[
        ("7:0", "VL", "vector length"), ("15:8", "NSREG", "s registers"), ("31:16", "K_MAX_256", "K_MAX / 256"),
    ]),
    "CAPS": dict(src="EX §7.5", desc="capability bits; retired bits are never set and never reused", fields=[
        ("0",     "VU",      "VOP, RED, ROPE and SOP present (VU §5.10)"),
        ("3",     "TOPK",    "TOPK present (RTL from M8; the simulator from M6)"),
        ("5",     "ATT",     "the seven AT ops present (KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM)"),
        ("6",     "DQROW",   "DQROW in the SPM form (RTL from M8; the simulator from M6)"),
        ("7",     "VMASK",   "VMASK (M9)"),
        ("11:8",  "VU_RSVD", "reserved for VU extensions, 0"),
        ("15:12", "AT_RSVD", "reserved for AT, 0 in v2.0"),
    ], retired=[
        (2, "RLD_LOOP",   "v1 RLD / MULI / LOOP"),
        (4, "WAITM_JUMP", "v1 WAITM / JUMP"),
    ]),
    "CFG3": dict(src="EX §7.5, VU §5.10, AT §5.10", desc="[15:0] VU, [31:16] AT", fields=[
        ("4:0",   "VU_FRONT", "VU front lanes with an EXP core (information for cycle models)"),
        ("9:5",   "VU_BACK",  "VU back lanes (information for cycle models)"),
        ("31:16", "PCAP16",   "PCAP / 16 (U200: 584)"),
    ]),
    "CTL_ID": dict(src="EX §5.1", desc="control-core version and features", fields=[
        ("7:0", "VERSION", "control-core version (2)"),
        ("8",   "DM",      "debug module"),
        ("9",   "REC",     "completion records"),
        ("10",  "BRIDGE",  "SPM bridge"),
        ("11",  "LOG",     "log ring"),
        ("12",  "WDOG",    "watchdog"),
        ("13",  "SBA",     "SBA connected"),
        ("14",  "SPM_LB",  "SPM line-buffered alias"),
    ]),
    # ---- device control and status (EX §5.2, §5.3) -------------------------------------------
    "CTRL": dict(src="EX §5.2", desc="pulse bits read 0; level bits act before the pulses of the same write", fields=[
        ("0",  "RV_RUN",       "RW level: 1 releases the RISC-V; 1 -> 0 starts the RISC-V reset sequence; refused while SR_BUSY"),
        ("1",  "HOST_HOLD",    "RW level: the host's hold source for the queue heads"),
        ("2",  "HOLD_ON_HALT", "RW level: a debug halt holds the queue heads"),
        ("3",  "IRQ_EN",       "RW level: host_irq_o = IRQ_EN and LCQ_WR != LCQ_RD"),
        ("8",  "QABORT",       "W1 pulse: host QABORT, at once"),
        ("9",  "QCLEAR",       "W1 pulse: host QCLEAR; only with QIDLE and the RISC-V quiet, else QCLEAR_REFUSED"),
        ("10", "SOFT_RESET",   "W1 pulse: device-level reset sequence (EX §8.5)"),
    ]),
    "STATUS": dict(src="EX §5.3", desc="bits 0..15 levels, 16..27 sticky (W1C)", fields=[
        ("0",     "RV_RUNNING",     "RV_RUN = 1 and the hart is out of reset"),
        ("1",     "RV_HALTED",      "debug halt"),
        ("2",     "RV_SLEEP",       "WFI"),
        ("3",     "RV_CLK_OK",      "clk_rv runs and its domain is out of reset"),
        ("4",     "RV_RESET_DONE",  "the RISC-V reset sequence has completed"),
        ("5",     "ACTIVE",         "TAKEN != ANSWERED"),
        ("6",     "QIDLE",          "no entry queued or running, issue FIFO empty"),
        ("7",     "QQUIESCENT",     "quiescent (CC §9.6)"),
        ("8",     "QERR",           "some unit is in error state"),
        ("9",     "SR_BUSY",        "SOFT_RESET in progress"),
        ("10",    "LCQ_PENDING",    "LCQ_WR != LCQ_RD"),
        ("11",    "LOG_PENDING",    "LOG_WR != LOG_RD"),
        ("12",    "PLAT_ERR",       "the platform's error input plat_err_i"),
        ("15:13", "HOLD_SRC",       "hold sources (layout HOLD_SRC: host, RISC-V, debug halt)"),
        ("16",    "LOAD_REFUSED",   "sticky: LOAD_DATA write or VERIFY while RV_RUN = 1 or SR_BUSY"),
        ("17",    "LOAD_TIMEOUT",   "sticky: a load or VERIFY word timed out"),
        ("18",    "DMI_TIMEOUT",    "sticky"),
        ("19",    "QCLEAR_REFUSED", "sticky"),
        ("20",    "PUSH_TIMEOUT",   "sticky: the RISC-V's issue safety net fired"),
        ("21",    "BRIDGE_TIMEOUT", "sticky"),
        ("22",    "LQ_OVERFLOW",    "sticky: a push while the LQ was full (descriptor dropped)"),
        ("23",    "RING_RD_BAD",    "sticky: an LCQ_RD or LOG_RD write was refused"),
        ("24",    "LOG_DROP",       "sticky: log words were dropped"),
        ("25",    "UNIT_HUNG",      "sticky: a unit watchdog fired"),
        ("26",    "SR_RV_TIMEOUT",  "sticky: SOFT_RESET went on without the RISC-V reset completing"),
        ("27",    "RV_RUN_REFUSED", "sticky"),
    ]),
    "HOLD_SRC": dict(src="EX §5.3, CC §5.1, DB §5.5", width=4,
                     desc="bit order of a hold-source field (STATUS[15:13], QSTATUS[4:2], DBG_U_STATE[11:8])",
                     fields=[("0", "HOST", "host (CTRL.HOST_HOLD)"), ("1", "RV", "RISC-V (QCTRL QHOLD)"),
                             ("2", "HALT", "debug halt with HOLD_ON_HALT"), ("3", "DB", "DB's dbg_hold (DBG_U_STATE only)")]),
    "RV_BOOT": dict(src="EX §5.1", desc="boot vector, 256-byte aligned", fields=[
        ("31:8", "ADDR", "boot vector bits [31:8]; [7:0] read 0"),
    ]),
    "RV_CFG": dict(src="EX §5.1", desc="TCM sizes", fields=[
        ("15:0", "ITCM_KIB", "ITCM KiB"), ("31:16", "DTCM_KIB", "DTCM KiB"),
    ]),
    "WDOG_CFG": dict(src="EX §5.1, §8.6", desc="unit progress watchdog", fields=[
        ("4:0", "L", "limit 2^L core cycles without progress; 0 = off"),
    ]),
    "RT_STATE": dict(src="EX §4.5", desc="runtime phase (RISC-V 0x290, mirrored to host 0x038)", fields=[
        ("7:0",   "PHASE",  "runtime phase"),
        ("31:16", "KERNEL", "kernel index"),
    ], values={"PHASE": [
        (0, "RESET", ""), (1, "BOOT", ""), (2, "IDLE", ""), (3, "DISPATCH", ""), (4, "KERNEL", ""),
        (5, "JOIN", ""), (6, "ANSWER", ""), (7, "TRAPPED", ""),
    ]}),
    "LQ_CFG": dict(src="EX §5.1", desc="ring parameters and watermark valid bits", fields=[
        ("7:0",   "LQ_DEPTH",     ""), ("15:8", "LCQ_DEPTH", ""), ("27:16", "LOG_DEPTH", ""),
        ("30",    "STOP_VALID",   ""), ("31",   "CANCEL_VALID", ""),
    ]),
    "PERF_INFO": dict(src="EX §5.1", desc="PERF layout", fields=[
        ("7:0", "COUNT", "counters implemented"), ("15:8", "X", "first extension index X = 5 + 2 NUM_CH"),
    ]),
    "Q_DEPTH": dict(src="EX §5.1, CC §7.4", desc="queue depth of unit u at [5u+4:5u]; 0 = unit absent", fields=[
        ("4:0", "DMA", "U200 M6: 8"), ("9:5", "QACT", "4"), ("14:10", "MVM", "4"), ("19:15", "VU", "16"),
        ("24:20", "ATT", "0: absent in M6"), ("29:25", "SYNC", "4"),
    ]),
    "LOG_CFG": dict(src="EX §4.10", desc="host RW at 0x098, RISC-V read-only copy at 0x288", fields=[
        ("0",   "ENABLE", "0: the SDK drops every record"),
        ("1",   "BLOCK",  "the SDK waits for space instead of dropping"),
        ("7:4", "LEVEL",  "the SDK drops records above this level (the only runtime filter)"),
    ]),
    "LOAD_CTRL": dict(src="EX §5.1, §3.3", desc="", fields=[("0", "VERIFY", "read LOAD_COUNT words back and CRC them")]),
    "LOAD_STATUS": dict(src="EX §5.1, §3.3", desc="", fields=[("0", "VERIFY_BUSY", "")]),
    "Q_STATE": dict(src="EX §5.4", desc="queue state of unit u", fields=[
        ("4:0", "ENTRIES", "entries in the queue"), ("5", "RUNNING", "the unit holds an op"),
        ("6", "DEP_WAIT", "the head waits for deps"), ("7", "HELD", ""), ("8", "ERR", "error state"),
        ("9", "HUNG", "watchdog"),
    ]),
    "Q_FIFO": dict(src="EX §5.4", desc="", fields=[("7:0", "ENTRIES", "issue-FIFO entries waiting for the dispatcher")]),
    # ---- adapter (EX §9.2) -------------------------------------------------------------------
    "ADAPTER_CTRL": dict(src="EX §9.2", desc="", fields=[
        ("0", "MEM_EN", "rising edge: latch and alignment-check all bases; requests are held while 0"),
    ]),
    "ADAPTER_STATUS": dict(src="EX §9.2, docs/platform-u200.md §2", desc="sticky, W1C", fields=[
        ("0", "RRESP_ERR", "RRESP error"), ("1", "BRESP_ERR", "BRESP error"),
        ("2", "ORPHAN_RESP", "response with no outstanding request for its id"),
        ("3", "BASE_MISALIGNED", "a base was not 4 KiB aligned at the MEM_EN edge"),
        ("4", "REQ_HELD", "a core memory request was held while MEM_EN = 0"),
        ("5", "BASE_LOCKED", "a base write while MEM_EN = 1 was ignored"),
        ("31:16", "ERR_COUNT", "saturating count of error responses"),
    ]),
    "ADAPTER_VERSION": dict(src="EX §9.2", desc="{16'h0002, 8'h00, NUM_CH}", fields=[
        ("7:0", "NUM_CH", ""), ("31:16", "MAJOR", "2"),
    ]),
    # ---- launch formats (EX §4.3, §4.4) ------------------------------------------------------
    "HANDLE": dict(src="EX §4.3", desc="DRAM handle (descriptor args, OPL, DBG_TRW_BASE)", fields=[
        ("27:0", "BEAT", "beat in the channel"), ("31:28", "CH", "channel"),
    ]),
    "TICKET": dict(src="CC §7.5", desc="ticket {unit, seq}; unit 15 = none; (u, s) done iff (D[u] - s) mod 2^28 in [1, 2^27)", fields=[
        ("27:0", "SEQ", "per-unit sequence"), ("31:28", "UNIT", "unit id"),
    ]),
    "DESC": dict(src="EX §4.3", desc="descriptor word w0", fields=[
        ("7:0",   "TYPE",   "descriptor type; other values reserved (BAD_DESC)"),
        ("8",     "PROF",   "profiled launch: REC_CFG.ALL and NODE markers for the launch"),
        ("31:16", "KERNEL", "index into .tsr_kernels (LAUNCH); 0 for PING"),
    ], values={"TYPE": [(0x01, "LAUNCH", ""), (0x02, "PING", "")]}),
    "LCQ_R0": dict(src="EX §4.4", desc="record word r0, one layout for every record type", fields=[
        ("7:0",   "TYPE",     "record type"),
        ("15:8",  "STATUS",   "launch STATUS (DONE), 4 for TRAP; kernel-defined for kernel events"),
        ("23:16", "ERR_CODE", "code of the first error, 0 if none"),
        ("31:24", "SEQ",      "SEQ[7:0] of the descriptor"),
    ], values={
        "TYPE": [(0x01, "DONE", "answer"), (0x02, "PONG", "answer"), (0x80, "HELLO", "event"),
                 (0x81, "TRAP", "answer if ACTIVE, else event"),
                 (0xC0, "EVENT_FIRST", "kernel events 0xC0..0xFF (ANSWER = 0)"), (0xFF, "EVENT_LAST", "")],
        "STATUS": [(0, "OK", "RET >= 0 and no op of the launch failed"),
                   (1, "TILE_ERR", "the first error has a code other than ABORTED; r3..r6 describe it"),
                   (2, "BAD_DESC", "the descriptor was refused; the kernel did not run"),
                   (3, "KERNEL_ERR", "RET < 0, no tile error"),
                   (4, "TRAP", "TRAP records only"),
                   (5, "ABORTED", "cancelled (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8)"),
                   (6, "PLAT_ERR", "STATUS.PLAT_ERR was set when the runtime answered"),
                   (7, "STOPPED", "the kernel honoured a soft stop")],
    }),
    "HELLO_R4": dict(src="EX §4.4", desc="HELLO record word r4", fields=[
        ("15:0", "SDK_ABI", "SDK ABI version"), ("31:16", "RT_VERSION", "runtime version"),
    ]),
    # ---- RISC-V control window (CC §5.1, EX §4.5, DB §10.2) ----------------------------------
    "ISSUE_PUSH": dict(src="CC §5.1", desc="ISSUE_PUSH write data", fields=[
        ("0",    "REC",   "completion record wanted (DB §8.2)"),
        ("15:8", "WMASK", "word i with bit i clear is sent as 0; w0 is always sent"),
    ]),
    "ISSUE_CREDIT": dict(src="CC §5.1", desc="min(free queue slots, free issue-FIFO entries, 15) per unit", fields=[
        ("3:0", "DMA", ""), ("7:4", "QACT", ""), ("11:8", "MVM", ""), ("15:12", "VU", ""),
        ("19:16", "ATT", ""), ("23:20", "SYNC", ""),
    ]),
    "ISSUE_UNITS": dict(src="CC §5.1", desc="bit u = unit u is built (constant)", fields=[
        ("0", "DMA", ""), ("1", "QACT", ""), ("2", "MVM", ""), ("3", "VU", ""), ("4", "ATT", "0 in M6"),
        ("5", "SYNC", ""),
    ]),
    "ISSUE_STATUS": dict(src="CC §5.1", desc="R/W1C", fields=[
        ("0", "PUSH_TIMEOUT", "a push waited ISSUE_WAIT_MAX and faulted"),
        ("1", "ERR_IRQ", "sticky, set on the rising edge of 'some unit in error state' (irq_fast[1])"),
    ]),
    "QCTRL": dict(src="CC §5.1", desc="one bit per write, in push order through the issue FIFO", fields=[
        ("0", "QABORT", ""), ("1", "QHOLD_SET", "set the RISC-V hold source"),
        ("2", "QHOLD_CLR", "clear the RISC-V hold source"), ("3", "QCLEAR", "gets a SYNC ticket (ISSUE_TICKET)"),
    ]),
    "QSTATUS": dict(src="CC §5.1", desc="queue status (RISC-V view)", fields=[
        ("0", "QIDLE", "ISSUED_SEQ = DONE_SEQ for every u and the issue FIFO empty"),
        ("1", "HOLD", "a hold is in effect"), ("2", "HOLD_HOST", "host hold"), ("3", "HOLD_RV", "RISC-V hold"),
        ("4", "HOLD_HALT", "debug-halt hold"), ("13:8", "ERR", "error state of unit u at bit 8 + u"),
        ("21:16", "NONEMPTY", "queue u non-empty at bit 16 + u"),
    ]),
    "WAIT_CREDIT": dict(src="CC §5.1", desc="irq_fast[5] while the unit's credit != 0", fields=[
        ("2:0", "UNIT", ""), ("31", "EN", ""),
    ]),
    "LQ_FLAGS": dict(src="EX §4.5", desc="RISC-V view of the launch state", fields=[
        ("0", "CANCEL_VALID", ""), ("1", "STOP_VALID", ""), ("2", "ACTIVE", ""), ("3", "PLAT_ERR", "STATUS.PLAT_ERR"),
    ]),
    "LCQ_POST": dict(src="EX §4.5", desc="LCQ_POST write data", fields=[
        ("0", "ANSWER", "the record answers the running launch (ANSWERED += 1)"),
    ]),
    "REC_CFG": dict(src="DB §10.2, §8.2", desc="completion-record configuration", fields=[
        ("0", "ALL", "a record for every entry, not only ISSUE_PUSH.REC ones"),
        ("1", "IRQ_HALF", "irq_fast[4] while REC_COUNT >= 32"),
    ]),
    "REC_DATA3": dict(src="DB §8.2", desc="completion record word 3", fields=[
        ("3:0", "CODE", "completion code"), ("4", "RAN", ""), ("5", "MARKER", ""),
        ("15:8", "OPCODE", ""), ("31:16", "PIDX", "push index [15:0]"),
    ]),
    "DBG_FLAGS": dict(src="DB §10.2", desc="RISC-V view of DB's state (control window 0x400)", fields=[
        ("0", "DBG_STOP", "a DB stop or the debug-halt hold is in effect"), ("1", "SERIAL", ""),
        ("2", "TRW_EN", ""), ("3", "DMI_EN", "a debugger may be attached"), ("4", "HALT_ON_BOOT", ""),
    ]),
    "MARK": dict(src="DB §8.4", desc="TRACE_MARK value", fields=[
        ("27:0", "PAYLOAD", ""), ("31:28", "CLASS", ""),
    ], values={"CLASS": [
        (0, "USER", "any payload"), (1, "LAUNCH_BEGIN", "payload: layout MARK_LAUNCH"),
        (2, "LAUNCH_END", "payload: layout MARK_LAUNCH"), (3, "NODE_BEGIN", "payload: op-list index"),
        (4, "NODE_END", "payload: op-list index"), (5, "LAYER", "payload: layer index"),
        (6, "PHASE", "payload: kernel-defined id (.tsr_marks)"),
    ]}),
    "MARK_LAUNCH": dict(src="DB §8.4", desc="payload of LAUNCH_BEGIN / LAUNCH_END markers", fields=[
        ("15:0", "SEQ", "launch SEQ low bits"), ("27:16", "KERNEL", "kernel index"), ("31:28", "CLASS", ""),
    ]),
    # ---- log, trace and debug records (DB §6.1, §8.3) ----------------------------------------
    "LOG_HDR": dict(src="DB §6.1", desc="log record header (the only definition; EX §4.10, SW §6.6)", fields=[
        ("7:0",   "N",      "payload words after the header"),
        ("11:8",  "LEVEL",  ""),
        ("15:12", "SOURCE", ""),
        ("18:16", "TYPE",   "5..7 invalid"),
        ("19",    "TS",     "the first payload word is MTIME[31:0]"),
        ("27:20", "LSEQ",   "SEQ[7:0] of the running launch; 0 outside launches"),
        ("31:28", "SYNC",   "0xA"),
    ], values={
        "LEVEL": [(0, "FATAL", ""), (1, "ERROR", ""), (2, "WARN", ""), (3, "INFO", ""), (4, "DEBUG", ""), (5, "TRACE", "")],
        "SOURCE": [(0, "RUNTIME", ""), (1, "KERNEL", ""), (2, "TRAP", "trap handler"), (3, "DEBUG", "debug kernels")],
        "TYPE": [(0, "FMT", "format-string address in .tsr_logfmt, then arguments"), (1, "TEXT", "UTF-8, zero-padded"),
                 (2, "TRAP", "the 44 words of __tsr_crash"), (3, "EVENT", "event code (marker encoding), arguments"),
                 (4, "BLOB", "tag, raw words")],
        "SYNC": [(0xA, "MAGIC", "")],
    }),
    "TRW_W0": dict(src="DB §8.3", desc="trace-writer record word 0", fields=[
        ("3:0", "KIND", ""), ("7:4", "UNIT", ""), ("11:8", "CODE", ""), ("12", "RAN", ""),
        ("13", "REC", "REC flag"), ("14", "TBP", "tile-breakpoint hit"), ("23:16", "OPCODE", ""),
        ("31:24", "W0_HI", "w0[15:8] of the op"),
    ], values={"KIND": [(0, "OP", ""), (1, "MARK", ""), (2, "STOP", ""), (3, "RVHALT", ""), (4, "DROP", ""),
                        (15, "PAD", "ignored")]}),
    # ---- DBG block (DB §10.1) ----------------------------------------------------------------
    "DBG_ID": dict(src="DB §10.1", desc="0 = block absent", fields=[
        ("7:0", "VERSION", "2"), ("8", "HIST", "history"), ("9", "TRW", "trace writer"),
        ("10", "RVWP", "RV watchpoints"), ("11", "REC", "completion records"), ("14:12", "NTBP", "tile breakpoints"),
        ("16:15", "NRVWP", "RV watchpoints"), ("23:20", "LOG2_HIST", "log2 HIST_DEPTH"),
    ]),
    "DBG_CTRL": dict(src="DB §10.1", desc="pulses; one of bits 0..2 per write (QSTEP > RELEASE > STOP_NOW)", fields=[
        ("0", "STOP_NOW", ""), ("1", "RELEASE", ""), ("2", "QSTEP", ""), ("4", "HIST_CLEAR", ""),
        ("5", "RV_SAMPLE", "completes after the sample or 4,096 cycles"), ("6", "RVWP_APPLY", ""),
        ("7", "TRW_FLUSH", ""), ("8", "RV_HALT_CANCEL", ""), ("9", "MATCHQ_CLEAR", ""),
    ]),
    "DBG_CFG": dict(src="DB §10.1", desc="reset 0x30", fields=[
        ("0", "DMI_EN", "gates the DMI window"), ("1", "SERIAL", ""), ("2", "ERR_STOP", ""),
        ("3", "STOP_HALT_RV", ""), ("4", "HIST_EN", ""), ("5", "HIST_FREEZE_ERR", ""), ("6", "HIST_FREEZE", ""),
        ("7", "HUNG_STOP", ""), ("8", "HALT_ON_BOOT", ""), ("9", "RV_HALT_EN", ""), ("10", "TRW_EN", ""),
        ("11", "TRW_STOP_FULL", ""),
    ]),
    "DBG_STATUS": dict(src="DB §10.1", desc="levels and sticky W1C bits", fields=[
        ("0", "STOP_ACTIVE", ""), ("1", "DRAINED", ""), ("2", "STEP_BUSY", ""), ("3", "DB_HELD", "a head held by DB"),
        ("8:4", "CAUSE", "cause of the current stop (layout DBG_CAUSE)"), ("12:9", "BP_HIT", "TBPs hit"),
        ("14:13", "RVWP_HIT", "sticky"), ("15", "RV_HALT_PEND", ""), ("16", "RVWP_BUSY", ""),
        ("17", "HIST_FROZEN", ""), ("18", "RV_HALTED", ""), ("19", "RV_SLEEP", ""),
        ("20", "TRW_DROP", "sticky"), ("21", "REC_DROP", "sticky"), ("22", "SBA_REFUSED", "sticky"),
        ("23", "DMI_BLOCKED", "sticky"), ("24", "RV_SAMPLE_TIMEOUT", "sticky"), ("25", "RV_DFAULT", "sticky"),
        ("26", "PTR_ERR", "sticky"), ("27", "RVWP_TIMEOUT", "sticky"), ("28", "MATCHQ_OVF", "sticky, never expected"),
        ("29", "DMI_RESP_ERR", "sticky"), ("31", "S_VALID", ""),
    ]),
    "DBG_CAUSE": dict(src="DB §10.1, §5.2", width=5,
                      desc="bits of DBG_STATUS.CAUSE in the documented order (one bit per cause)", fields=[
        ("0", "NOW", ""), ("1", "BP", ""), ("2", "STEP", ""), ("3", "ERR", ""), ("4", "HUNG", ""),
    ]),
    "DBG_STEP_N": dict(src="DB §10.1", desc="", fields=[("15:0", "N", "entries per QSTEP (0 = 1)")]),
    "DBG_MATCHQ": dict(src="DB §10.1", desc="", fields=[("6:0", "PENDING", "pending match thresholds")]),
    "DBG_QSTATE": dict(src="DB §10.1, CC §8.3", desc="dispatcher state", fields=[
        ("10:8", "DISP", "dispatcher state (CC §8.3 order)"), ("23:16", "OUTSTANDING", "outstanding entries, saturating"),
    ], values={"DISP": [(0, "IDLE", "D_IDLE"), (1, "POP", "D_POP"), (2, "CHK", "D_CHK"), (3, "ENQ", "D_ENQ")]}),
    "DBG_RV_STATE": dict(src="DB §10.1", desc="", fields=[
        ("0", "HALTED", ""), ("2", "SLEEP", ""), ("3", "IN_RESET", ""), ("4", "SAMPLE_VALID", ""),
        ("5", "DMACTIVE", "dmactive_o"), ("6", "NDMRESET", "ndmreset_o"), ("7", "HALT_REQ", "halt-request level"),
        ("15:8", "SAMPLE_SEQ", ""), ("31:16", "DEBUG_ENTRIES", "debug-mode entries mod 2^16"),
    ]),
    "DBG_RVWP_CTL": dict(src="DB §10.1", desc="RV watchpoint j < 2", fields=[
        ("0", "EN", ""), ("1", "LOAD", ""), ("2", "STORE", ""), ("3", "HALT", ""),
    ]),
    "DBG_TBP_CTL": dict(src="DB §10.1, §5.3", desc="tile breakpoint i < 4", fields=[
        ("0", "EN", ""), ("3:1", "KIND", "5..7 never match"), ("4", "R", ""), ("5", "W", ""), ("6", "AFTER", ""),
        ("7", "HALT_RV", ""), ("8", "COUNT_ONLY", ""), ("31:16", "CHMASK", "channels of a DRAM breakpoint"),
    ], values={"KIND": [(0, "PIDX", "(pidx - A) mod 2^32 < (B - A) mod 2^32"), (1, "OPCODE", "A: layout TBP_OPC_A, B: TBP_OPC_B"),
                        (2, "SPM", "an SPM extent overlaps lines [A, B)"), (3, "DRAM", "the DRAM extent overlaps beats [A, B)"),
                        (4, "TICKET", "ticket = A")]}),
    "TBP_OPC_A": dict(src="DB §5.3", desc="DBG_TBP_A of an OPCODE breakpoint", fields=[
        ("7:0", "OPC", "opcode value"), ("15:8", "OPC_MASK", "opcode care mask"),
        ("19:16", "SUB", "sub value"), ("23:20", "SUB_MASK", "sub care mask"),
    ]),
    "TBP_OPC_B": dict(src="DB §5.3", desc="DBG_TBP_B of an OPCODE breakpoint", fields=[
        ("7:0", "UNITS", "unit mask; 0 = any unit"),
    ]),
    "DBG_U_STATE": dict(src="DB §5.5", desc="per-unit probe u < 8", fields=[
        ("1:0", "ACCEPTED", "ops accepted and not done"), ("6:4", "HEAD", "head state"),
        ("11:8", "HOLD", "hold sources at the head (layout HOLD_SRC)"), ("19:12", "OUTSTANDING", "outstanding entries"),
    ], values={"HEAD": [(0, "EMPTY", ""), (1, "LOAD", ""), (2, "WAIT_DEP", ""), (3, "WAIT_SNAP", ""), (4, "HELD", ""),
                        (5, "WAIT_UNIT", ""), (6, "ISSUE", "")]}),
    "DBG_TRW_MASK": dict(src="DB §10.1", desc="reset 0x7FF", fields=[
        ("7:0", "UNITS", ""), ("8", "MARKERS", ""), ("9", "STOPS", ""), ("10", "RVHALTS", ""),
    ]),
    "DBG_HIST_INFO": dict(src="DB §10.1", desc="", fields=[
        ("0", "PRESENT", "the slot holds that push index, recorded after the last HIST_CLEAR"), ("7:4", "UNIT", ""),
        ("11:8", "CODE", "checker code (0 = runs)"), ("12", "REC", ""), ("16:13", "TBPS", "TBPs matched"),
        ("17", "STOP", "stop point"),
    ]),
    # ---- issue FIFO (CC §7.2) ----------------------------------------------------------------
    "IFIFO_CTRL": dict(src="CC §7.2", desc="w0 of a CTRL issue-FIFO entry", fields=[("3:0", "CODE", "")],
                       values={"CODE": [(1, "QABORT", ""), (2, "QHOLD_SET", ""), (3, "QHOLD_CLR", ""),
                                        (4, "QCLEAR", "ticketed"), (5, "TRACE_MARK", "w1 = marker value")]}),
}

# Layouts wider than 32 bits (bit positions only): name -> same format, with `width`.
WIDE_LAYOUTS = {
    "IFIFO": dict(src="CC §7.2", width=387, desc="issue-FIFO entry (tsr_cdc_fifo, 32 entries LUTRAM)", fields=[
        ("1:0",     "KIND",   "entry kind"),
        ("257:2",   "W",      "w0..w7 (OP; masked words are 0); CTRL: w0 = layout IFIFO_CTRL, w1 = marker value"),
        ("289:258", "DEP0",   "OP"),
        ("321:290", "DEP1",   "OP"),
        ("322",     "REC",    ""),
        ("354:323", "TICKET", "OP and QCLEAR; the dispatcher asserts it equals its own count"),
        ("386:355", "PC",     "push PC (OP and TRACE_MARK)"),
    ], values={"KIND": [(0, "OP", ""), (1, "CTRL", "")]}),
}

# =================================================================================================
# Register maps. Row: (offset, NAME, access, owner, src, description[, extras])
# extras: n (array elements), stride (bytes between elements, default 4), words (consecutive words
# per element, default 1), layout (bit-field layout, default the layout of the same name), reset,
# block (a reserved or delegated byte range; NAME None = reserved).
# =================================================================================================

# Host CSR map: core CSR port, U200 s_axi_control 0x1000 + offset (EX §5).
HOST_CSR = [
    (0x000, "ID",           "R",      "EX",     "EX §5.1",        '"TSR1" (unchanged)', dict(reset=0x5453_5231)),
    (0x004, "VERSION",      "R",      "EX",     "EX §5.1, §7.5",  "[15:8] ISA major 2, [7:0] minor 0", dict(reset=0x0000_0200)),
    (0x008, "CFG0",         "R",      "EX",     "EX §5.1",        "NUM_CH, LANES, T_MAX (unchanged)"),
    (0x00C, "CFG1",         "R",      "EX",     "EX §5.1",        "SPM size in lines (unchanged)"),
    (0x010, "CFG2",         "R",      "EX",     "EX §5.1",        "VL, NSREG, K_MAX/256 (unchanged)"),
    (0x014, "CAPS",         "R",      "EX",     "EX §5.1, §7.5",  "capability bits"),
    (0x018, "CFG3",         "R",      "VU, AT", "EX §5.1, §7.5",  "[15:0] VU, [31:16] AT"),
    (0x01C, "CTL_ID",       "R",      "EX, CC", "EX §5.1",        "control-core version and features"),
    (0x020, "CTRL",         "RW/W1",  "EX",     "EX §5.1, §5.2",  "device control"),
    (0x024, "STATUS",       "R/W1C",  "EX",     "EX §5.1, §5.3",  "device status; W1C clears the sticky bits only"),
    (0x028, "RV_BOOT",      "RW",     "CC",     "EX §5.1",        "boot vector, taken when RV_RUN rises"),
    (0x02C, "RV_FREQ",      "R",      "CC",     "EX §5.1",        "clk_rv cycles in the last 2^16 core cycles; 0 = no clk_rv"),
    (0x030, "RV_CFG",       "R",      "CC",     "EX §5.1",        "[15:0] ITCM KiB, [31:16] DTCM KiB"),
    (0x034, "WDOG_CFG",     "RW",     "EX",     "EX §5.1, §8.6",  "unit watchdog limit", dict(reset=24)),
    (0x038, "RT_STATE",     "R",      "EX",     "EX §5.1, §4.5",  "copy of the runtime's RT_STATE"),
    (0x03C, "RT_INFO",      "R",      "EX",     "EX §5.1, §4.5",  "copy of the runtime's RT_INFO"),
    (0x040, "LQ_W",         "RW",     "EX",     "EX §5.1, §4.3",  "descriptor staging LQ_W0..LQ_W7; kept after a push", dict(n=8)),
    (0x060, "LQ_PUSH",      "W",      "EX",     "EX §5.1",        "push the staged descriptor (value ignored); dropped with LQ_OVERFLOW when full"),
    (0x064, "LQ_SEQ",       "R",      "EX",     "EX §5.1, §4.2",  "PUSHED"),
    (0x068, "LQ_TAKEN",     "R",      "EX",     "EX §5.1, §4.2",  "TAKEN"),
    (0x06C, "LQ_CANCEL",    "RW",     "EX",     "EX §5.1, §4.9",  "cancel watermark"),
    (0x070, "LQ_STOP",      "RW",     "EX",     "EX §5.1, §4.9",  "stop watermark"),
    (0x074, "LCQ_WR",       "R",      "EX",     "EX §5.1, §4.2",  "records posted"),
    (0x078, "LCQ_RD",       "RW",     "EX",     "EX §5.1, §4.2",  "records consumed by the host (EX §4.2 write rule)"),
    (0x07C, "LQ_CFG",       "R",      "EX",     "EX §5.1",        "ring depths and watermark valid bits"),
    (0x080, "PERF_INFO",    "R",      "EX",     "EX §5.1",        "counters implemented, extension index X"),
    (0x084, "LCQ_ANSWERED", "R",      "EX",     "EX §5.1, §4.2",  "ANSWERED"),
    (0x088, "Q_DEPTH",      "R",      "CC",     "EX §5.1",        "queue depth per unit; 0 = absent"),
    (0x090, "LOG_WR",       "R",      "EX",     "EX §5.1, §4.10", "log words accepted"),
    (0x094, "LOG_RD",       "RW",     "EX",     "EX §5.1, §4.10", "log words consumed by the host"),
    (0x098, "LOG_CFG",      "RW",     "EX",     "EX §5.1, §4.10", "log configuration", dict(reset=0x0000_00F1)),
    (0x09C, "LOG_DROPPED",  "R",      "EX",     "EX §5.1",        "words dropped, saturating; cleared by SOFT_RESET"),
    (0x0C0, "LOAD_ADDR",    "RW",     "EX",     "EX §5.1, §3.3",  "TCM byte address (word aligned); a write restarts LOAD_CRC"),
    (0x0C4, "LOAD_DATA",    "RW",     "EX",     "EX §5.1, §3.3",  "write: store the word, address += 4; read: the word (only while RV_RUN = 0 and not SR_BUSY)"),
    (0x0C8, "LOAD_CRC",     "R",      "EX",     "EX §5.1, §3.3",  "CRC-32 of the words accepted, or read by VERIFY"),
    (0x0CC, "LOAD_COUNT",   "RW",     "EX",     "EX §5.1, §3.3",  "word count for VERIFY"),
    (0x0D0, "LOAD_CTRL",    "W",      "EX",     "EX §5.1, §3.3",  "bit0 VERIFY"),
    (0x0D4, "LOAD_STATUS",  "R",      "EX",     "EX §5.1, §3.3",  "bit0 VERIFY_BUSY"),
    (0x100, "PERF_LO",      "R",      "EX",     "EX §5.1, §6",    "counter i bits [31:0]; a host LO read snapshots the counter for the next HI read", dict(n=64, stride=8)),
    (0x104, "PERF_HI",      "R",      "EX",     "EX §5.1, §6",    "counter i bits [63:32] (the snapshot)", dict(n=64, stride=8)),
    (0x300, "LCQ_REC",      "R",      "EX",     "EX §5.1, §4.2",  "completion ring: record k at 0x300 + 32 (k mod LCQ_DEPTH), word j at +4j", dict(n=8, stride=32, words=8)),
    (0x400, "LOG_RING",     "R",      "EX",     "EX §5.1, §4.10", "log word k at 0x400 + 4 (k mod LOG_DEPTH)", dict(n=256)),
    (0x800, "DBG",          "DB",     "DB",     "EX §5.1, DB §10.1", "DBG block (DBG_CSR rows); undefined offsets read 0", dict(block=0x300)),
    (0xB00, None,           "-",      "-",      "EX §5.1",        "reserved (the planned CTL_* block is dissolved)", dict(block=0x100)),
    (0xC00, "DMI",          "R/W",    "DB, CC", "EX §5.1, DB §3.2", "debug module register r at 0xC00 + 4r; acts only while DBG_CFG.DMI_EN = 1", dict(n=128)),
    (0xE00, "Q_DONE",       "R",      "EX",     "EX §5.4",        "DONE_SEQ[u], 28 bits", dict(n=8)),
    (0xE20, "Q_ISSUED",     "R",      "EX",     "EX §5.4",        "ISSUED_SEQ[u], 28 bits (synchronised copy)", dict(n=8)),
    (0xE40, "Q_STATE",      "R",      "EX",     "EX §5.4",        "queue state of unit u", dict(n=8)),
    (0xE60, "Q_FIFO",       "R",      "EX",     "EX §5.4",        "issue-FIFO entries waiting for the dispatcher"),
    (0xE64, "Q_PIDX",       "R",      "EX",     "EX §5.4",        "push index of the dispatcher's next ticketed entry (= DBG_NEXT_PIDX); firmware never uses it"),
    (0xE80, "Q_ERR_SEQ",    "R",      "EX",     "EX §5.4",        "first error record of unit u: ERR_SEQ", dict(n=8, stride=16)),
    (0xE84, "Q_ERR_CODE",   "R",      "EX",     "EX §5.4",        "ERR_CODE ([7:0])", dict(n=8, stride=16)),
    (0xE88, "Q_ERR_INFO",   "R",      "EX",     "EX §5.4",        "ERR_INFO", dict(n=8, stride=16)),
    (0xE8C, "Q_ERR_W0",     "R",      "EX",     "EX §5.4",        "ERR_W0", dict(n=8, stride=16)),
    (0xF00, "Q_ERR_PIDX",   "R",      "EX",     "EX §5.4",        "push index of unit u's first error", dict(n=8)),
    (0xF20, None,           "-",      "-",      "EX §5.1",        "reserved", dict(block=0xE0)),
]

# DBG block: host CSR offsets 0x800..0xAFC (DB §10.1). Nothing here is reset by QABORT, QCLEAR,
# the RISC-V reset or SOFT_RESET.
DBG_CSR = [
    (0x800, "DBG_ID",          "R",     "DB",     "DB §10.1", "version and features; 0 = block absent"),
    (0x804, "DBG_CTRL",        "W",     "DB",     "DB §10.1", "pulses, reads 0"),
    (0x808, "DBG_CFG",         "RW",    "DB",     "DB §10.1", "configuration", dict(reset=0x30)),
    (0x80C, "DBG_STATUS",      "R/W1C", "DB",     "DB §10.1", "levels and sticky bits"),
    (0x810, "DBG_STOP_PIDX",   "R",     "DB",     "DB §10.1", "T (valid while STOP_ACTIVE)"),
    (0x814, "DBG_STOP_AT",     "RW",    "DB",     "DB §10.1", "write P: S <- P, valid; read: S (valid iff S_VALID)"),
    (0x818, "DBG_STEP_N",      "RW",    "DB",     "DB §10.1", "entries per QSTEP", dict(reset=1)),
    (0x81C, "DBG_NEXT_PIDX",   "R",     "DB",     "DB §10.1", "push index the dispatcher assigns to its next ticketed entry"),
    (0x820, "DBG_STEP_PIDX",   "R",     "DB",     "DB §10.1", "X of the current / last QSTEP"),
    (0x824, "DBG_MATCHQ",      "R",     "DB",     "DB §10.1", "pending match thresholds"),
    (0x828, "DBG_MTIME_LO",    "R",     "DB",     "DB §10.1", "live core time (read HI, LO, HI)"),
    (0x82C, "DBG_MTIME_HI",    "R",     "DB",     "DB §10.1", ""),
    (0x834, "DBG_QSTATE",      "R",     "DB",     "DB §10.1", "dispatcher state, outstanding entries"),
    (0x840, "DBG_RV_STATE",    "R",     "DB",     "DB §10.1", "RISC-V state and sample sequence"),
    (0x844, "DBG_RV_PC",       "R",     "DB",     "DB §10.1, §6.2", "last crash-dump sample: current PC"),
    (0x848, "DBG_RV_NPC",      "R",     "DB",     "DB §10.1, §6.2", "next PC"),
    (0x84C, "DBG_RV_DADDR",    "R",     "DB",     "DB §10.1, §6.2", "last data address"),
    (0x850, "DBG_RV_EPC",      "R",     "DB",     "DB §10.1, §6.2", "exception PC"),
    (0x854, "DBG_RV_EADDR",    "R",     "DB",     "DB §10.1, §6.2", "exception address"),
    (0x858, "DBG_RV_WPADDR",   "R",     "DB",     "DB §10.1", "address of the last RV-watchpoint hit"),
    (0x860, "DBG_RVWP_CTL",    "RW",    "DB",     "DB §10.1", "RV watchpoint j < 2", dict(n=2, stride=16)),
    (0x864, "DBG_RVWP_LO",     "RW",    "DB",     "DB §10.1", "byte range [LO, HI)", dict(n=2, stride=16)),
    (0x868, "DBG_RVWP_HI",     "RW",    "DB",     "DB §10.1", "", dict(n=2, stride=16)),
    (0x86C, "DBG_RVWP_HITS",   "R/WC",  "DB",     "DB §10.1", "hits; a write clears", dict(n=2, stride=16)),
    (0x880, "DBG_TBP_CTL",     "RW",    "DB",     "DB §10.1", "tile breakpoint i < 4", dict(n=4, stride=16)),
    (0x884, "DBG_TBP_A",       "RW",    "DB",     "DB §10.1, §5.3", "", dict(n=4, stride=16)),
    (0x888, "DBG_TBP_B",       "RW",    "DB",     "DB §10.1, §5.3", "", dict(n=4, stride=16)),
    (0x88C, "DBG_TBP_HITS",    "R/WC",  "DB",     "DB §10.1", "matches; a write clears", dict(n=4, stride=16)),
    (0x8C0, "DBG_RDOUT",       "R",     "DB",     "DB §10.1, §5.5", "outstanding read bursts: channel c in byte c mod 4 of word c / 4", dict(n=4)),
    (0x8D0, "DBG_WROUT",       "R",     "DB",     "DB §10.1, §5.5", "outstanding write bursts, same packing", dict(n=4)),
    (0x900, "DBG_U_STATE",     "R",     "DB",     "DB §10.1, §5.5", "unit u < 8", dict(n=8, stride=16)),
    (0x904, "DBG_U_OLDEST",    "R",     "DB",     "DB §10.1, §5.5", "ticket of the oldest outstanding entry; TICKET_NONE if none", dict(n=8, stride=16, layout="TICKET")),
    (0x908, "DBG_U_OPIDX",     "R",     "DB",     "DB §10.1, §5.5", "its push index", dict(n=8, stride=16)),
    (0x90C, "DBG_U_AGE",       "R",     "DB",     "DB §10.1, §5.5", "core cycles since it became the oldest (saturating)", dict(n=8, stride=16)),
    (0x980, "DBG_TRW_BASE",    "RW",    "DB",     "DB §10.1, §8.3", "ring base handle, 16-beat aligned (beat [3:0] read 0)", dict(layout="HANDLE")),
    (0x984, "DBG_TRW_LINES",   "RW",    "DB",     "DB §10.1, §8.3", "ring size in 64-B lines, power of two >= 16 (else ignored, PTR_ERR)"),
    (0x988, "DBG_TRW_WPTR",    "R",     "DB",     "DB §10.1, §8.3", "records written (mod 2^32)"),
    (0x98C, "DBG_TRW_RPTR",    "RW",    "DB",     "DB §10.1, §8.3", "records consumed; beyond WPTR ignored (PTR_ERR)"),
    (0x990, "DBG_TRW_DROPS",   "R",     "DB",     "DB §10.1, §8.3", "records dropped (saturating)"),
    (0x994, "DBG_TRW_MASK",    "RW",    "DB",     "DB §10.1, §8.3", "record classes written", dict(reset=0x7FF)),
    (0x9C0, "SREG",            "R",     "VU, DB", "EX §5.1, §7.4, DB §10.1, VU §4.4", "s0..s15 (f32 bits); present whenever the VU is built, also with HAS_DBG = 0", dict(n=16)),
    (0xA00, "DBG_HIST_HEAD",   "R",     "DB",     "DB §10.1, §5.6", "next push index to be recorded"),
    (0xA04, "DBG_HIST_IDX",    "RW",    "DB",     "DB §10.1, §5.6", "push index to read"),
    (0xA08, "DBG_HIST_INFO",   "R",     "DB",     "DB §10.1, §5.6", ""),
    (0xA0C, "DBG_HIST_PIDX",   "R",     "DB",     "DB §10.1, §5.6", ""),
    (0xA10, "DBG_HIST_TICKET", "R",     "DB",     "DB §10.1, §5.6", "", dict(layout="TICKET")),
    (0xA14, "DBG_HIST_DEP0",   "R",     "DB",     "DB §10.1, §5.6", "", dict(layout="TICKET")),
    (0xA18, "DBG_HIST_DEP1",   "R",     "DB",     "DB §10.1, §5.6", "", dict(layout="TICKET")),
    (0xA1C, "DBG_HIST_TIME",   "R",     "DB",     "DB §10.1, §5.6", "dispatch time MTIME[31:0]"),
    (0xA20, "DBG_HIST_PC",     "R",     "DB",     "DB §10.1, §5.6", "push PC"),
    (0xA40, "DBG_HIST_W",      "R",     "DB",     "DB §10.1, §5.6", "op words w0..w7", dict(n=8)),
]
DBG_BLOCK = (0x800, 0x300)   # base, bytes (EX §5.1, DB §10.1)

# RISC-V control window: offsets from 0x2000_0000, word accesses only, reads have no side effects
# (CC §5). Owners: CC (§5.1, timer), EX (0x200..0x2FC, EX §4.5), DB (0x160..0x17C, 0x400..0x7FC).
CTL_WINDOW = [
    (0x000, "ISSUE_W",      "RW",    "CC", "CC §5.1", "staged instruction words; kept after a push", dict(n=8, reset=0)),
    (0x020, "ISSUE_DEP0",   "RW",    "CC", "CC §5.1", "dependency ticket; reset to TICKET_NONE by every push", dict(layout="TICKET", reset=0xF000_0000)),
    (0x024, "ISSUE_DEP1",   "RW",    "CC", "CC §5.1", "same", dict(layout="TICKET", reset=0xF000_0000)),
    (0x028, "ISSUE_PUSH",   "W",     "CC", "CC §5.1", "push; a push without a credit completes late (safety net)"),
    (0x02C, "ISSUE_TICKET", "R",     "CC", "CC §5.1", "ticket of the last push or QCTRL.QCLEAR", dict(layout="TICKET")),
    (0x030, "ISSUE_CREDIT", "R",     "CC", "CC §5.1", "credit per unit"),
    (0x034, "ISSUE_UNITS",  "R",     "CC", "CC §5.1", "bit u = unit u is built"),
    (0x038, "ISSUE_STATUS", "R/W1C", "CC", "CC §5.1", "push timeout and error interrupt"),
    (0x03C, "ISSUE_PIDX",   "R",     "CC", "CC §5.1", "push index the next ticketed entry will receive (+1 per ISSUE_PUSH and QCLEAR); exact"),
    (0x040, "DONE_SEQ",     "R",     "CC", "CC §5.1", "[27:0] completions of unit u (synchronised copy; may lag, never leads)", dict(n=8)),
    (0x080, "ISSUED_SEQ",   "R",     "CC", "CC §5.1", "[27:0] sequence the next push to unit u gets", dict(n=8)),
    (0x0C0, "ERR_SEQ",      "R",     "CC", "CC §5.1", "first error of unit u since the last clear: ERR_SEQ", dict(n=8, stride=16)),
    (0x0C4, "ERR_CODE",     "R",     "CC", "CC §5.1", "ERR_CODE; 0 = none", dict(n=8, stride=16)),
    (0x0C8, "ERR_INFO",     "R",     "CC", "CC §5.1", "ERR_INFO", dict(n=8, stride=16)),
    (0x0CC, "ERR_W0",       "R",     "CC", "CC §5.1", "ERR_W0", dict(n=8, stride=16)),
    (0x140, "QCTRL",        "W",     "CC", "CC §5.1", "queue control entries, in push order"),
    (0x144, "QSTATUS",      "R",     "CC", "CC §5.1", "queue status"),
    (0x148, "WAIT_TICKET",  "RW",    "CC", "CC §5.1", "irq_fast[0] pending while this ticket is done; unit 15 disarms", dict(layout="TICKET", reset=0xF000_0000)),
    (0x14C, "WAIT_CREDIT",  "RW",    "CC", "CC §5.1", "irq_fast[5] pending while that unit's credit != 0", dict(reset=0)),
    (0x160, "REC_COUNT",    "R",     "DB", "DB §10.2, §8.2", "completion records waiting (0..64)"),
    (0x164, "REC_DATA",     "R",     "DB", "DB §10.2, §8.2", "head record words 0..3 (word table REC)", dict(n=4)),
    (0x174, "REC_POP",      "W",     "DB", "DB §10.2, §8.2", "drop the head record"),
    (0x178, "TRACE_MARK",   "W",     "DB", "DB §10.2, §8.4", "marker through the issue FIFO", dict(layout="MARK")),
    (0x17C, "REC_CFG",      "RW",    "DB", "DB §10.2, §8.2", "completion-record configuration", dict(reset=0)),
    (0x180, "ERR_PIDX",     "R",     "CC", "CC §5.1", "push index of unit u's first error", dict(n=8)),
    (0x1C0, None,           "-",     "CC", "CC §5.1", "reserved (LOG_PUSH moved to 0x280)", dict(block=4)),
    (0x1C4, "SPM_LB_INV",   "W",     "CC", "CC §5.1, §10.5", "invalidate the SPM line buffer"),
    (0x200, "LQ_HEAD_W",    "R",     "EX", "EX §4.5", "the descriptor the next LQ_POP takes (valid while LQ_AVAIL != 0)", dict(n=8)),
    (0x220, "LQ_AVAIL",     "R",     "EX", "EX §4.5", "PUSHED - TAKEN as seen in clk_rv"),
    (0x224, "LQ_HEAD_SEQ",  "R",     "EX", "EX §4.5", "TAKEN = the sequence number of the head"),
    (0x228, "LQ_POP",       "W",     "EX", "EX §4.5", "take the head; access fault if LQ_AVAIL = 0 or ACTIVE"),
    (0x22C, "LQ_CANCEL",    "R",     "EX", "EX §4.5", "copy of the host's cancel watermark"),
    (0x230, "LQ_STOP",      "R",     "EX", "EX §4.5", "copy of the host's stop watermark"),
    (0x234, "LQ_FLAGS",     "R",     "EX", "EX §4.5", "watermark valid bits, ACTIVE, PLAT_ERR"),
    (0x240, "LCQ_W",        "RW",    "EX", "EX §4.5", "record staging; kept after a post", dict(n=8)),
    (0x260, "LCQ_POST",     "W",     "EX", "EX §4.5", "post the staged record; access fault if LCQ_FREE = 0, or ANSWER while not ACTIVE"),
    (0x264, "LCQ_FREE",     "R",     "EX", "EX §4.5", "LCQ_DEPTH - (LCQ_WR - LCQ_RD) as seen in clk_rv (may under-report)"),
    (0x280, "LOG_PUSH",     "W",     "EX", "EX §4.5", "one log word; dropped and counted when the ring is full, never faults"),
    (0x284, "LOG_FREE",     "R",     "EX", "EX §4.5", "free words (may under-report)"),
    (0x288, "LOG_CFG",      "R",     "EX", "EX §4.5", "copy of the host's LOG_CFG"),
    (0x290, "RT_STATE",     "RW",    "EX", "EX §4.5", "runtime phase, mirrored to host 0x038; 0 after a RISC-V reset", dict(reset=0)),
    (0x294, "RT_INFO",      "RW",    "EX", "EX §4.5", "runtime- or kernel-defined progress word, mirrored to host 0x03C"),
    (0x300, "MTIME_LO",     "R",     "CC", "CC §5.2", "core clock cycles since reset (gray-synchronised)"),
    (0x304, "MTIME_HI",     "R",     "CC", "CC §5.2", ""),
    (0x308, "MTIMECMP_LO",  "RW",    "CC", "CC §5.2", "irq_timer while MTIME >= MTIMECMP", dict(reset=0xFFFF_FFFF)),
    (0x30C, "MTIMECMP_HI",  "RW",    "CC", "CC §5.2", "", dict(reset=0xFFFF_FFFF)),
    (0x400, "DBG_FLAGS",    "R",     "DB", "DB §10.2", "DB state for the runtime"),
    (0x404, None,           "-",     "DB", "DB §10.2", "reserved for DB", dict(block=0x3FC)),
]
CTL_EX_RANGE = (0x200, 0x300)                    # EX §4.5: [lo, hi)
CTL_DB_RANGES = [(0x160, 0x180), (0x400, 0x800)]   # DB §10.2

# XRT adapter s_axi_control map (EX §9.2), range 0x2000.
SAXI = [
    (0x000,  None,              "-",     "EX", "EX §9.2", "reserved, read 0 (v1 AP_CTRL, GIE, IER, ISR, cmd_addr, cmd_count, arg* deleted)", dict(block=0x40)),
    (0x040,  "CH_BASE_LO",      "RW",    "EX", "EX §9.2", "device byte address of the channel-c buffer, bits [31:0]; 4 KiB aligned; written while MEM_EN = 0", dict(n=16, stride=8)),
    (0x044,  "CH_BASE_HI",      "RW",    "EX", "EX §9.2", "bits [63:32]", dict(n=16, stride=8)),
    (0x0E0,  "ADAPTER_CTRL",    "RW",    "EX", "EX §9.2", "MEM_EN"),
    (0x0F0,  "ADAPTER_STATUS",  "R/W1C", "EX", "EX §9.2", "sticky adapter errors"),
    (0x0F4,  "ADAPTER_VERSION", "R",     "EX", "EX §9.2", "{16'h0002, 8'h00, NUM_CH}"),
    (0x1000, "CORE",            "-",     "EX", "EX §9.2", "core CSR window: 0x1000 + offset, any time; writes need WSTRB = 0xF", dict(block=0x1000)),
]
SAXI_BYTES = 0x2000   # EX §9.1

# =================================================================================================
# RISC-V address map (CC §4.1): (NAME, base, size, limit, fetch, access, src, description)
# size = the U200 size of the region; limit = end of the reserved range (exclusive).
# =================================================================================================
RV_MAP = [
    ("ITCM",   0x0000_0000, 128 * 1024,      0x0010_0000, True,  "byte/half/word", "CC §4.1", "code; data reads steal the fetch port"),
    ("DTCM",   0x0010_0000, 64 * 1024,       0x0020_0000, False, "byte/half/word", "CC §4.1", ".rodata, .data, .bss, stack"),
    ("DM",     0x1000_0000, 0x1000,          0x1000_1000, True,  "word",           "CC §4.1", "debug module memory (riscv-dbg layout)"),
    ("CTL",    0x2000_0000, 0x1000,          0x2000_1000, False, "word",           "CC §4.1", "control window (CTL_WINDOW)"),
    ("CORE",   0x2000_1000, 0x1000,          0x2000_2000, False, "word loads",     "CC §4.1, §5.3", "core window: read-only view of the host CSR map (same offsets)"),
    ("SPM",    0x4000_0000, 4 * 1024 * 1024, 0x4040_0000, False, "byte/half/word", "CC §4.1", "SPM window, exact: 0x4000_0000 + 64 x line + offset"),
    ("SPM_LB", 0x5000_0000, 4 * 1024 * 1024, 0x5040_0000, False, "byte/half/word", "CC §4.1, §10.5", "SPM line-buffered alias"),
]

# Ibex fast interrupts (CC §12.3): (index, NAME, src, description)
IRQS = [
    (0, "WAIT_TICKET", "CC §12.3", "WAIT_TICKET done"),
    (1, "ERR",         "CC §12.3", "ISSUE_STATUS.ERR_IRQ"),
    (2, "LQ",          "CC §12.3, EX §4.5", "LQ_AVAIL != 0 and not ACTIVE"),
    (3, "LCQ",         "CC §12.3, EX §4.5", "LCQ_FREE != 0"),
    (4, "REC",         "CC §12.3, DB §8.2", "REC_COUNT >= 32 with REC_CFG.IRQ_HALF"),
    (5, "CREDIT",      "CC §12.3", "WAIT_CREDIT"),
    (6, "DB_RSVD",     "CC §12.3, DB §10.2", "reserved for DB"),
]

# DMI registers used by libtsr_dbg and modelled by the simulator (DB §9.1, RISC-V Debug Spec);
# host offset 0xC00 + 4 x address. (address, NAME, src, description[, n])
DMI_REGS = [
    (0x04, "DATA0",      "DB §9.1", ""),
    (0x05, "DATA1",      "DB §9.1", ""),
    (0x10, "DMCONTROL",  "DB §9.1", "haltreq, resumereq, ackhavereset, ndmreset, dmactive"),
    (0x11, "DMSTATUS",   "DB §9.1", ""),
    (0x12, "HARTINFO",   "DB §9.1", ""),
    (0x16, "ABSTRACTCS", "DB §9.1", "progbufsize, busy, cmderr, datacount"),
    (0x17, "COMMAND",    "DB §9.1", "Access Register, postexec-only"),
    (0x20, "PROGBUF0",   "DB §9.1", "progbuf0..7", 8),
    (0x38, "SBCS",       "DB §9.1", ""),
    (0x39, "SBADDRESS0", "DB §9.1", ""),
    (0x3C, "SBDATA0",    "DB §9.1", ""),
]

# DRAM map per channel (EX §11): (NAME, offset, size or None = to the channel end, src, description)
DRAM_REGIONS = [
    ("OPL",    0,                 64 * 1024 * 1024,  "EX §11", "ch0: op lists, replay push streams, argument blocks, code overlays (was CMD)"),
    ("IO",     64 * 1024 * 1024,  256 * 1024 * 1024, "EX §11", "ch0: host <-> device tensors; compute chunks; the TRW ring while profiling"),
    ("AUX",    320 * 1024 * 1024, 192 * 1024 * 1024, "EX §11", "ch0: small f32 tensors (norm weights, RoPE tables)"),
    ("WEIGHT", 512 * 1024 * 1024, None,              "EX §11", "every channel: weights bottom-up, KV planes top-down, compute heap between"),
]

# =================================================================================================
# Units (EX §7.3, CC §7.4): (id, NAME, Q_DEPTH on the U200 M6 bitstream, parameter depth,
# I3 limit (ops accepted and not done, CC §9.4), src, description)
# =================================================================================================
UNITS = [
    (0, "DMA",  8,  8,  1, "EX §7.3, CC §7.4", "LD, ST, KVST"),
    (1, "QACT", 4,  4,  1, "EX §7.3, CC §7.4", "QACT"),
    (2, "MVM",  4,  4,  2, "EX §7.3, CC §7.4, §9.4", "MVM, ATTQK, ATTPV"),
    (3, "VU",   16, 16, 2, "EX §7.3, CC §7.4, §9.4", "VOP, RED, ROPE, TOPK, SOP, DQROW, ASMAX, AMASK, AMERGE, ANORM (VMASK from M9)"),
    (4, "ATT",  0,  8,  1, "EX §7.3, CC §7.4", "reserved for a future AT attention engine; absent in M6 (was CMD)"),
    (5, "SYNC", 4,  4,  1, "EX §7.3, CC §7.4", "NOP, FENCE, QCLEAR and every refused opcode"),
]
UNITS_SPECIAL = [
    (14, "MARKER", "EX §7.3, CC §7.4", "software marker records (REC / TRW); never a valid dependency"),
    (15, "NONE",   "EX §7.3, CC §7.4", "no dependency"),
]
UNITS_RESERVED = (6, 13, "EX §7.3")   # ids 6..13 reserved; a dependency on them is BAD_DEP

# =================================================================================================
# Global opcode table (EX §7.2): (lo, hi, MNEMONIC or None, status, unit, CAPS field, owner, src,
# description). status: kept (encoding unchanged), respec (number kept, re-specified), new,
# reserved, retired. Only kept / respec / new opcodes are legal; every other opcode is accepted
# by the issue port, gets a SYNC ticket and completes ILLEGAL_OP. `unit` of a reserved block is
# its class (information only). An opcode whose CAPS field is 0 in the configuration is refused
# the same way (TOPK and DQROW before M8, VMASK before M9).
# =================================================================================================
OPCODES = [
    (0x00, 0x00, "NOP",    "kept",     "SYNC", None,    "spec", "EX §7.2", "SYNC join point: completes when its dependencies are done"),
    (0x01, 0x01, "END",    "reserved", "SYNC", None,    "-",    "EX §7.2", "v1 END: ILLEGAL_OP"),
    (0x02, 0x02, "LI",     "reserved", "SYNC", None,    "-",    "EX §7.2", "v1 LI: ILLEGAL_OP"),
    (0x03, 0x03, "ADDI",   "reserved", "SYNC", None,    "-",    "EX §7.2", "v1 ADDI: ILLEGAL_OP"),
    (0x04, 0x04, "MULI",   "retired",  "SYNC", None,    "-",    "EX §7.2", "retired, never reused"),
    (0x05, 0x05, "RLD",    "retired",  "SYNC", None,    "-",    "EX §7.2", "retired, never reused"),
    (0x06, 0x06, "LOOP",   "retired",  "SYNC", None,    "-",    "EX §7.2", "retired, never reused"),
    (0x07, 0x07, "WAITM",  "retired",  "SYNC", None,    "-",    "EX §7.2", "retired, never reused"),
    (0x08, 0x08, "JUMP",   "retired",  "SYNC", None,    "-",    "EX §7.2", "retired, never reused"),
    (0x09, 0x09, "CALL",   "retired",  "SYNC", None,    "-",    "EX §7.2", "retired (never shipped), never reused"),
    (0x0A, 0x0F, None,     "reserved", "SYNC", None,    "EX",   "EX §7.2", "SYNC-class extension"),
    (0x10, 0x10, "LD",     "kept",     "DMA",  None,    "spec", "EX §7.2", "DRAM -> SPM, encoding unchanged"),
    (0x11, 0x11, "ST",     "kept",     "DMA",  None,    "spec", "EX §7.2", "SPM -> DRAM, encoding unchanged"),
    (0x12, 0x12, "DQROW",  "respec",   "VU",   "DQROW", "VU",   "EX §7.2, VU §5.6", "SPM -> f32 dequantisation; simulator from M6, RTL in M8"),
    (0x13, 0x13, None,     "reserved", "DMA",  None,    "VU",   "EX §7.2", "reserved for VU (DMA class)"),
    (0x14, 0x14, "KVST",   "new",      "DMA",  "ATT",   "AT",   "EX §7.2, AT §5.1", "store K or V rows into a KV plane"),
    (0x15, 0x17, None,     "reserved", "DMA",  None,    "AT",   "EX §7.2", "reserved for AT (DMA class: KV row stores and loads)"),
    (0x18, 0x18, "QACT",   "kept",     "QACT", None,    "spec", "EX §7.2", "f32 -> QA, encoding unchanged"),
    (0x19, 0x1B, None,     "reserved", "QACT", None,    "AT",   "EX §7.2", "reserved for AT (quantize class: q8_0 KV rows)"),
    (0x1C, 0x1F, None,     "reserved", None,   None,    "-",    "EX §7.2", "reserved"),
    (0x20, 0x20, "MVM",    "kept",     "MVM",  None,    "spec", "EX §7.2", "matrix x activations, encoding unchanged"),
    (0x21, 0x21, "ATTQK",  "new",      "MVM",  "ATT",   "AT",   "EX §7.2, AT §5.2", "scores and row maxima over a KV group range"),
    (0x22, 0x22, "ATTPV",  "new",      "MVM",  "ATT",   "AT",   "EX §7.2, AT §5.3", "P.V channel partials and row sums"),
    (0x23, 0x23, None,     "reserved", "MVM",  None,    "AT",   "EX §7.2", "reserved for AT (MVM class)"),
    (0x24, 0x2F, None,     "reserved", "MVM",  None,    "-",    "EX §7.2", "reserved (MVM class)"),
    (0x30, 0x30, "VOP",    "respec",   "VU",   "VU",    "VU",   "EX §7.2, VU §5.2", "element-wise vector op"),
    (0x31, 0x31, "RED",    "respec",   "VU",   "VU",    "VU",   "EX §7.2, VU §5.3", "segmented reduction with post-ops"),
    (0x32, 0x32, "ROPE",   "respec",   "VU",   "VU",    "VU",   "EX §7.2, VU §5.4", "rotary embedding over T tokens"),
    (0x33, 0x33, "TOPK",   "respec",   "VU",   "TOPK",  "VU",   "EX §7.2, VU §5.5", "top-k per row; simulator from M6, RTL in M8"),
    (0x34, 0x34, "ASMAX",  "new",      "VU",   "ATT",   "AT",   "EX §7.2, AT §5.4", "softmax numerator"),
    (0x35, 0x35, "AMASK",  "new",      "VU",   "ATT",   "AT",   "EX §7.2, AT §5.5", "general mask path"),
    (0x36, 0x36, "AMERGE", "new",      "VU",   "ATT",   "AT",   "EX §7.2, AT §5.6", "channel tree and online merge"),
    (0x37, 0x37, "ANORM",  "new",      "VU",   "ATT",   "AT",   "EX §7.2, AT §5.7", "final normalisation"),
    (0x38, 0x38, "VMASK",  "new",      "VU",   "VMASK", "VU",   "EX §7.2, VU §5.7", "position-window mask; ILLEGAL_OP until M9"),
    (0x39, 0x3F, None,     "reserved", "VU",   None,    "VU",   "EX §7.2", "reserved for VU extensions"),
    (0x40, 0x40, "SOP",    "respec",   "VU",   "VU",    "VU",   "EX §7.2, VU §5.8", "scalar op on s0..s15"),
    (0x41, 0x47, None,     "reserved", "VU",   None,    "VU",   "EX §7.2", "reserved for VU (scalar extensions)"),
    (0x48, 0x6F, None,     "reserved", None,   None,    "-",    "EX §7.2", "reserved (future units)"),
    (0x70, 0x7E, None,     "reserved", "SYNC", None,    "EX",   "EX §7.2", "SYNC-class extension"),
    (0x7F, 0x7F, "FENCE",  "kept",     "SYNC", None,    "spec", "EX §7.2", "SYNC barrier: completes when every op pushed before it is done"),
    (0x80, 0xFF, None,     "reserved", None,   None,    "-",    "EX §7.2", "reserved"),
]

# =================================================================================================
# Error codes (EX §8.1): (code, NAME or None, status, src, condition, ERR_INFO). status: active,
# reserved (defined name, never raised), retired (old v1 name, never raised, never reused).
# =================================================================================================
ERRORS = [
    (0,  "NONE",         "active",   "EX §8.1", "", ""),
    (1,  "ILLEGAL_OP",   "active",   "EX §8.1", "reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it", "0"),
    (2,  "BAD_FMT",      "active",   "EX §8.1", "unsupported weight format", "0"),
    (3,  "QA_MISMATCH",  "reserved", "EX §8.1", "never raised", ""),
    (4,  "SPM_RANGE",    "active",   "EX §8.1", "SPM operand out of range (spec range-check rule)", "start address of the failing operand"),
    (5,  "DRAM_RANGE",   "active",   "EX §8.1", "DRAM operand out of range", "start address"),
    (6,  "K_RANGE",      "active",   "EX §8.1", "K > K_MAX or not a multiple of the block size", "0"),
    (7,  "T_RANGE",      "active",   "EX §8.1", "T = 0 or T > T_MAX", "0"),
    (8,  "ABORTED",      "active",   "EX §8.1", "completed without running: a QABORT, or its unit was in error state", "0"),
    (9,  "LOOP_NEST",    "retired",  "EX §8.1", "v1, never raised", ""),
    (10, "WAIT_TIMEOUT", "retired",  "EX §8.1", "v1, never raised", ""),
    (11, "DEP_FAILED",   "active",   "EX §8.1", "a dependency completed with an error", "0"),
    (12, "BAD_DEP",      "active",   "EX §8.1", "a dependency on a ticket not pushed yet, on unit 6..14 or an absent unit, or below its unit's clear point", "0"),
    (13, "OPERAND",      "active",   "EX §8.1", "an operand value outside the range its op defines, other than K, T and the address ranges (VU and AT ops only)", "the offending value"),
    (14, None,           "reserved", "EX §8.1", "", ""),
    (15, None,           "reserved", "EX §8.1", "", ""),
]
# Dispatch check order (EX §8.1): the first failing check gives the code.
ERR_CHECK_ORDER = ("EX §8.1", ["ILLEGAL_OP", "BAD_DEP", "BAD_FMT", "K_RANGE", "T_RANGE", "OPERAND",
                               "SPM_RANGE", "DRAM_RANGE"])
# Launch STATUS precedence (EX §4.4), highest first.
STATUS_PRECEDENCE = ("EX §4.4", ["TRAP", "BAD_DESC", "ABORTED", "TILE_ERR", "PLAT_ERR", "STOPPED",
                                 "KERNEL_ERR", "OK"])

# =================================================================================================
# PERF slots (EX §6): (slot, NAME, kind, src, description). slot: an index, ("ch", k) for
# 5 + k*C + c (per channel c < C = NUM_CH), ("x", k) for X + k with X = 5 + 2C.
# kind: time (ACTIVE and not QHALT), event (ACTIVE), rv_time (clk_rv cycles, synchronised ACTIVE),
# time_nohalt (ACTIVE, not gated by QHALT), reserved (reads 0).
# =================================================================================================
PERF = [
    (0,         "CYCLES",          "time",        "EX §6", "core cycles"),
    (1,         "RETIRED",         "event",       "EX §6", "queue entries completed on every unit, any outcome"),
    (2,         "MVM_CYCLES",      "time",        "EX §6", "the MVM unit holds an op"),
    (3,         "VU_CYCLES",       "time",        "EX §6, VU §5.10", "the VU holds an op"),
    (4,         None,              "reserved",    "EX §6", "reserved (was WAITM), reads 0"),
    (("ch", 0), "CH_RD_BEATS",     "event",       "EX §6", "read beats delivered on channel c (ch0 exact: no fetch)"),
    (("ch", 1), "CH_WAIT",         "time",        "EX §6", "channel c has a read outstanding and delivers no beat"),
    (("x", 0),  "DMA_CYCLES",      "time",        "EX §6", "the DMA unit holds an op"),
    (("x", 1),  "QACT_CYCLES",     "time",        "EX §6", "the QACT unit holds an op"),
    (("x", 2),  "ATT_CYCLES",      "time",        "EX §6", "unit 4 (AT) holds an op; reads 0 without it"),
    (("x", 3),  "MVM_WAIT",        "time",        "EX §6", "MVM idle while its queue holds an entry that cannot start"),
    (("x", 4),  "MVM_EMPTY",       "time",        "EX §6", "MVM idle and its queue empty (issue-bound)"),
    (("x", 5),  "HOLD_CYCLES",     "time",        "EX §6", "some queue head is held"),
    (("x", 6),  "PUSHES",          "event",       "EX §6", "entries pushed"),
    (("x", 7),  "PUSH_STALL",      "rv_time",     "EX §6", "clk_rv cycles a push waited for a credit"),
    (("x", 8),  "BRIDGE_ACC",      "event",       "EX §6", "SPM bridge accesses"),
    (("x", 9),  "BRIDGE_WAIT",     "rv_time",     "EX §6", "clk_rv cycles bridge accesses waited"),
    (("x", 10), "REC_DROPPED",     "event",       "EX §6", "completion records dropped"),
    (("x", 11), "LAUNCHES",        "event",       "EX §6", "answers posted"),
    (("x", 12), "RV_WFI",          "time",        "EX §6", "the RISC-V sleeps in WFI"),
    (("x", 13), "VU_SPM_WAIT",     "time",        "EX §6, VU §5.10", "a VU SPM request is raised and not granted"),
    (("x", 14), "VU_HAZARD",       "time",        "EX §6, VU §5.10", "an accepted VU op waits for an earlier VU op"),
    (("x", 15), "VU_LINES",        "event",       "EX §6, VU §5.10", "destination lines written by VU ops"),
    (("x", 16), "ATT_QK_CYCLES",   "time",        "EX §6, AT §10", "the MVM unit holds an ATTQK"),
    (("x", 17), "ATT_PV_CYCLES",   "time",        "EX §6, AT §10", "the MVM unit holds an ATTPV"),
    (("x", 18), "ATT_PATH_STALL",  "time",        "EX §6, AT §10", "a cluster in an ATT op waits for the gather or the p broadcast"),
    (("x", 19), "KVST_RMW",        "event",       "EX §6, AT §10", "meta beats read-modify-written by KVST"),
    (("x", 20), "SPM_RD_WAIT",     "time",        "EX §6, CC §6.3", "client-cycles a raised SPM read request waits for its bank port"),
    (("x", 21), "SPM_WR_WAIT",     "time",        "EX §6, CC §6.3", "the same for write requests"),
    (61,        "TRW_RECORDS",     "event",       "EX §6, DB §8.1", "trace records written to DRAM"),
    (62,        "TRW_DROPS",       "event",       "EX §6, DB §8.1", "trace records dropped"),
    (63,        "DBG_STOP_CYCLES", "time_nohalt", "EX §6, DB §8.1", "ACTIVE and (RV_HALTED or STOP_ACTIVE); a measurement is valid only if 0"),
]

# =================================================================================================
# Word tables of fixed-size records and structures: name -> {src, desc, words (size), rows:
# [(word, NAME, description[, count])]}.
# =================================================================================================
WORD_TABLES = {
    "DESC": dict(src="EX §4.3", words=8, desc="launch descriptor (host -> RISC-V)", rows=[
        (0, "HDR",  "layout DESC"),
        (1, "ARG0", "ARG0..ARG6: integers, SPM lines or DRAM handles (layout HANDLE)", 7),
    ]),
    "LCQ_DONE": dict(src="EX §4.4", words=8, desc="DONE record (0x01, answer)", rows=[
        (0, "R0", "layout LCQ_R0"), (1, "SEQ", "SEQ of the answered descriptor"),
        (2, "RET", "kernel return value (int32)"), (3, "ERR_TICKET", "first failing ticket; TICKET_NONE = none"),
        (4, "ERR_INFO", ""), (5, "ERR_W0", ""), (6, "ERR_PIDX", ""),
        (7, "DURATION", "core cycles from the pop to the post, saturating"),
    ]),
    "LCQ_PONG": dict(src="EX §4.4", words=8, desc="PONG record (0x02, answer)", rows=[
        (0, "R0", ""), (1, "SEQ", ""), (2, "ARG0", "ARG0 echo"), (7, "DURATION", ""),
    ]),
    "LCQ_HELLO": dict(src="EX §4.4, SW §6.3", words=8, desc="HELLO record (0x80, event)", rows=[
        (0, "R0", ""), (1, "TAKEN", "TAKEN at the post"), (2, "ID_LO", "image id [31:0]"),
        (3, "ID_HI", "image id [63:32]"), (4, "VERSIONS", "layout HELLO_R4"), (5, "N_KERNELS", "kernel count"),
        (7, "BOOT_CYCLES", "boot core cycles"),
    ]),
    "LCQ_TRAP": dict(src="EX §4.4, DB §6.3", words=8, desc="TRAP record (0x81)", rows=[
        (0, "R0", ""), (1, "SEQ", "SEQ (answer) or TAKEN (event)"), (2, "MCAUSE", ""), (3, "MEPC", ""),
        (4, "MTVAL", ""), (5, "RT_STATE", "at the trap"), (6, "RT_INFO", ""), (7, "MTIME", "MTIME[31:0]"),
    ]),
    "LCQ_EVENT": dict(src="EX §4.4", words=8, desc="kernel event record (0xC0..0xFF)", rows=[
        (0, "R0", "type, kernel-defined [23:8], seq[7:0]"), (1, "SEQ", "SEQ of the running launch"),
        (2, "DATA", "r2..r7 kernel-defined", 6),
    ]),
    "REC": dict(src="DB §8.2", words=4, desc="completion record (REC_DATA0..3)", rows=[
        (0, "TICKET", "ticket (unit 14 for markers)"), (1, "T_END", "MTIME[31:0] at completion (markers: dispatch time)"),
        (2, "RUN", "start -> end cycles, saturating (markers: the value)"), (3, "INFO", "layout REC_DATA3"),
    ]),
    "TRW_OP": dict(src="DB §8.3", words=8, desc="trace-writer OP record", rows=[
        (0, "HDR", "layout TRW_W0"), (1, "PIDX", "push index"), (2, "TICKET", ""),
        (3, "T_END_LO", "MTIME"), (4, "T_END_HI", ""), (5, "RUN", "t_end - t_start (0 if not run), saturating"),
        (6, "QWAIT", "t_head - t_dispatch"), (7, "HWAIT", "t_start - t_head; t_end - t_head for non-run entries"),
    ]),
    "TRW_MARK": dict(src="DB §8.3", words=8, desc="trace-writer MARK record", rows=[
        (0, "HDR", ""), (1, "NEXT_PIDX", "next push index"), (2, "VALUE", "marker value"),
        (3, "T_LO", "dispatch time"), (4, "T_HI", ""),
    ]),
    "TRW_STOP": dict(src="DB §8.3", words=8, desc="trace-writer STOP record", rows=[
        (0, "HDR", ""), (1, "T", "threshold"), (2, "CAUSE", "cause bits (layout DBG_CAUSE)"),
        (3, "T_LO", "time"), (4, "T_HI", ""), (5, "ACTIVATED", "1 activated, 0 released"),
    ]),
    "TRW_RVHALT": dict(src="DB §8.3", words=8, desc="trace-writer RVHALT record", rows=[
        (0, "HDR", ""), (2, "HALTED", "1 halted, 0 resumed"),
    ]),
    "TRW_DROP": dict(src="DB §8.3", words=8, desc="trace-writer DROP record", rows=[
        (0, "HDR", ""), (1, "DROPPED", "records dropped since the previous DROP record"),
    ]),
    "DBG_INFO": dict(src="DB §6.3", words=16, desc="__tsr_dbg (section .tsr_dbg, 64 B)", rows=[
        (0, "MAGIC", "DBG_INFO_MAGIC"), (1, "VERSION", "1"), (2, "BUILD_ID", "ELF note build id"),
        (3, "ATTACHED", "written by the debugger"), (4, "LAUNCH_BREAK", "0 off, 0xFFFFFFFF all, else kernel index + 1"),
        (5, "TRAP_COUNT", ""),
    ]),
    "CRASH": dict(src="DB §6.3", words=48, desc="__tsr_crash (section .tsr_crash, 192 B)", rows=[
        (0, "MAGIC", "CRASH_MAGIC"), (1, "VERSION", "1"), (2, "MCAUSE", ""), (3, "MEPC", ""), (4, "MTVAL", ""),
        (5, "MSTATUS", ""), (6, "MTIME_LO", ""), (7, "MTIME_HI", ""), (8, "SEQ", "launch SEQ"), (9, "KERNEL", "kernel index"),
        (10, "RT_STATE", ""), (11, "RT_INFO", ""), (12, "X1", "x1..x31 at words 12..42", 31), (43, "NESTED", ""),
    ]),
}

# =================================================================================================
# Instruction encodings: op -> {src, owner, fields: [(enc, NAME, description)], values}.
# w0[7:0] is the opcode (OPCODES). Every bit outside the listed fields must be 0 (ILLEGAL_OP);
# function-dependent rules (unused operands must be 0) are in the tables of the functions below.
# `values` maps a field to its value list, or to the name of a shared ENUMS entry.
# =================================================================================================
ENCODINGS = {
    # ---- v1 ops, encodings unchanged (spec ISA; EX §7.1-§7.2: x16 = 0, A = imm) ----------------
    "NOP":   dict(src="spec ISA, EX §7.1", owner="spec", fields=[]),
    "FENCE": dict(src="spec ISA, EX §7.1", owner="spec", fields=[]),
    "LD": dict(src="spec ISA, EX §7.2", owner="spec", fields=[
        ("w0[11:8]", "CH",    "channel"), ("w1", "SPM", "SPM destination line"),
        ("w2", "DRAM",  "DRAM source beat"), ("w3", "LINES", "lines"),
    ]),
    "ST": dict(src="spec ISA, EX §7.2", owner="spec", fields=[
        ("w0[11:8]", "CH",    "channel"), ("w1", "SPM", "SPM source line"),
        ("w2", "DRAM",  "DRAM destination beat"), ("w3", "LINES", "lines"),
    ]),
    "QACT": dict(src="spec ISA, EX §7.2", owner="spec", fields=[
        ("w0[11:8]", "KIND", "QA kind"), ("w1", "DST", "QA destination line"), ("w2", "SRC", "f32 source line"),
        ("w3", "K", ""), ("w4", "T", ""), ("w5", "SRC_STRIDE", "lines; 0 = ceil(4K/64)"),
    ], values={"KIND": "QA"}),
    "MVM": dict(src="spec ISA, EX §7.2", owner="spec", fields=[
        ("w0[11:8]", "FMT", "weight format"), ("w0[12]", "ACC", "y += W.x"), ("w1", "DST", "destination line"),
        ("w2", "X", "QA activation line"), ("w3", "W_OFF", "weight offset (beats)"), ("w4", "N", "rows"),
        ("w5[15:0]", "K", ""), ("w5[19:16]", "T", ""), ("w6", "DST_STRIDE", "lines; 0 = ceil(4N/64)"),
    ], values={"FMT": "FMT"}),
    # ---- VU (VU §5) ---------------------------------------------------------------------------
    "VOP": dict(src="VU §5.2", owner="VU", fields=[
        ("w0[11:8]",  "CMODE",    "per-row scalar c"),
        ("x16[7:0]",  "FUNC",     "function (VOP_FUNCS)"),
        ("x16[11:8]", "SI",       "s register for CMODE SREG, else 0"),
        ("w1[23:0]",  "D",        "destination line"),
        ("w2[23:0]",  "A",        "source a line (0 if the function reads no a)"),
        ("w3[23:0]",  "B",        "source b line (0 if unused)"),
        ("w4[23:0]",  "C",        "row-vector line for CMODE VEC, else 0"),
        ("w5[23:0]",  "M",        "elements per row"),
        ("w6[15:0]",  "ROWS",     "rows"),
        ("w6[31:16]", "D_STRIDE", "lines"),
        ("w7[15:0]",  "A_STRIDE", "lines (0 if a is unused)"),
        ("w7[31:16]", "B_STRIDE", "lines (0 if b is unused)"),
    ], values={"CMODE": [(0, "NONE", "no c"), (1, "SREG", "c = s[SI]"), (2, "VEC", "c_r = word r of the vector at line C")]}),
    "RED": dict(src="VU §5.3", owner="VU", fields=[
        ("w0[11:8]",   "KIND",     ""),
        ("w0[12]",     "TO_S",     "1: result to s[SD] (ROWS must be 1); 0: per-row vector at D"),
        ("x16[3:0]",   "SD",       "destination s register if TO_S, else 0"),
        ("x16[5:4]",   "POST",     "post-op; MAX requires NONE"),
        ("x16[11:8]",  "SK",       "RMS scale register (typically 1/n), else 0"),
        ("x16[15:12]", "SE",       "RMS epsilon register, else 0"),
        ("w1[23:0]",   "D",        "result vector line (rows words); 0 if TO_S"),
        ("w2[23:0]",   "A",        "source line"),
        ("w5[23:0]",   "M",        "elements per row"),
        ("w6[15:0]",   "ROWS",     "rows"),
        ("w7[15:0]",   "A_STRIDE", "lines"),
    ], values={"KIND": [(0, "SUM", ""), (1, "SUMSQ", ""), (2, "MAX", "")],
               "POST": [(0, "NONE", ""), (1, "RCP", "reciprocal"), (2, "RMS", "RC(SQ(FA(FM(y, s[SK]), s[SE]))) (VU §6.2)")]}),
    "ROPE": dict(src="VU §5.4", owner="VU", fields=[
        ("w0[11:8]",  "MODE",       ""),
        ("w0[12]",    "FF",         "recurrence form: the table holds freq factors"),
        ("w0[13]",    "FS",         "recurrence form: multiply by freq_scale in s[SFS]"),
        ("w0[14]",    "ATAB",       "A-table form (YaRN): FF, FS and STS must be 0"),
        ("x16[3:0]",  "SM",         "s register holding M"),
        ("x16[7:4]",  "STS",        "s register holding TS (recurrence form; 0 with ATAB)"),
        ("x16[11:8]", "SFS",        "s register holding freq_scale when FS, else 0"),
        ("w1[23:0]",  "D",          "destination line of token 0"),
        ("w2[23:0]",  "SRC",        "source line of token 0"),
        ("w3[23:0]",  "POS",        "line of the I32 position vector"),
        ("w4[23:0]",  "TABLE",      "line of the f32 table of n_dims/2 entries; 0 when neither FF nor ATAB"),
        ("w5[15:0]",  "HEAD_DIM",   "elements per head"),
        ("w5[31:16]", "N_DIMS",     "rotated elements per head"),
        ("w6[15:0]",  "N_HEADS",    "heads per token"),
        ("w6[31:16]", "T",          "tokens"),
        ("w7[15:0]",  "SRC_STRIDE", "lines between tokens; 0 = H"),
        ("w7[31:16]", "D_STRIDE",   "lines between tokens; 0 = H"),
    ], values={"MODE": [(0, "NORM", "adjacent pairs"), (1, "NEOX", "half-split pairs")]}),
    "TOPK": dict(src="VU §5.5", owner="VU", fields=[
        ("x16[4:0]",  "K",        "1..16"),
        ("w1[23:0]",  "D",        "output line of row 0: k (index, value) pairs"),
        ("w2[23:0]",  "A",        "source line"),
        ("w5[23:0]",  "M",        "elements per row"),
        ("w6[15:0]",  "ROWS",     "rows"),
        ("w6[31:16]", "D_STRIDE", "lines; 0 = ceil(k/8)"),
        ("w7[15:0]",  "A_STRIDE", "lines"),
    ]),
    "DQROW": dict(src="VU §5.6", owner="VU", fields=[
        ("w0[11:8]",  "FMT",        "2..15 BAD_FMT"),
        ("w1[23:0]",  "D",          "f32 output line of row 0"),
        ("w2[23:0]",  "SRC",        "SPM line of row 0's repacked bytes"),
        ("w5",        "K",          "elements per row (K_RANGE checks)"),
        ("w6[15:0]",  "ROWS",       "rows"),
        ("w7[15:0]",  "SRC_STRIDE", "lines; 0 = ROW_BEATS(fmt, K)"),
        ("w7[31:16]", "D_STRIDE",   "lines; 0 = L(K)"),
    ], values={"FMT": "FMT"}),
    "VMASK": dict(src="VU §5.7", owner="VU", fields=[
        ("w0[11:8]",  "SUB",      ""),
        ("w1[23:0]",  "X",        "operand line (in place)"),
        ("w2",        "WIN",      "window size W, u32; 0 = no window"),
        ("w3",        "K0",       "i32 position of column 0"),
        ("w4",        "Q0",       "i32 query position of row group 0"),
        ("w5[23:0]",  "M",        "columns"),
        ("w6[15:0]",  "ROWS",     "rows"),
        ("w6[31:16]", "G",        "rows per query position (>= 1)"),
        ("w7[15:0]",  "X_STRIDE", "lines"),
        ("w7[31:16]", "P",        "row period; 0 = none"),
    ], values={"SUB": [(0, "WINDOW", "position window; others ILLEGAL_OP")]}),
    "SOP": dict(src="VU §5.8", owner="VU", fields=[
        ("w0[11:8]", "SUB",  "function (SOP_FUNCS)"),
        ("w1[3:0]",  "SD",   "destination s register"),
        ("w1[7:4]",  "SA",   "source s register a"),
        ("w1[11:8]", "SB",   "source s register b"),
        ("w2",       "IMM",  "f32 bits"),
        ("w3[27:0]", "ADDR", "SPM word address (LDS / STS)"),
    ]),
    # ---- AT (AT §5; VU §5.11 restates the vector-op encodings for the VU decoder) --------------
    "KVST": dict(src="AT §5.1", owner="AT", fields=[
        ("w0[11:8]",  "KV",         "plane"),
        ("x16[5:0]",  "N_ROWS_M1",  "n_rows - 1 (n_rows 1..64)"),
        ("w1",        "SRC",        "first QA(A32, E) vector (SPM line)"),
        ("w2",        "BASE",       "plane base B (beat, every channel)"),
        ("w3[8:0]",   "E64",        "E / 64"),
        ("w3[31:16]", "LG",         "local groups per channel"),
        ("w4",        "C0",         "first cell"),
        ("w5",        "SRC_STRIDE", "lines; 0 = QA_LINES(A32, E)"),
    ], values={"KV": [(0, "K", "K plane"), (1, "V", "V plane")]}),
    "ATTQK": dict(src="AT §5.2", owner="AT", fields=[
        ("w0[11:8]",  "FMT",      "KV format"),
        ("w0[12]",    "CAUSAL",   ""),
        ("w0[13]",    "SLIDE",    ""),
        ("w0[14]",    "MAXOUT",   "write row maxima at MDST"),
        ("x16[5:0]",  "G_M1",     "G - 1 (heads per KV head)"),
        ("x16[11:6]", "HG0",      "first head of the pass within each KV-head group"),
        ("w1",        "DST",      "score rows, row r at DST + r x RSTRIDE"),
        ("w2",        "X",        "T_q QA(A32, n_head x D) vectors"),
        ("w3",        "BASE",     "K plane base B"),
        ("w4[8:0]",   "E64",      "E / 64 (1..512)"),
        ("w4[10:9]",  "DCODE",    "D = 64 x 2^dcode"),
        ("w4[13:11]", "TQ_M1",    "T_q - 1"),
        ("w4[16:14]", "GS_M1",    "GS - 1"),
        ("w4[31:17]", "RSTRIDE",  "lines; 0 = n_groups"),
        ("w5[15:0]",  "U0",       "first group"),
        ("w5[31:16]", "N_GROUPS", "1..4095"),
        ("w6[15:0]",  "C_LO",     "cell bound relative to 16 u0"),
        ("w6[31:16]", "C_HI",     "cell bound relative to 16 u0"),
        ("w7",        "MDST",     "row-maximum vector line; 0 unless MAXOUT"),
    ], values={"FMT": "KVFMT", "DCODE": "DCODE"}),
    "ATTPV": dict(src="AT §5.3", owner="AT", fields=[
        ("w0[11:8]",  "FMT",      "KV format"),
        ("w0[12]",    "LSUM",     "also row sums"),
        ("x16[5:0]",  "G_M1",     "G - 1"),
        ("x16[11:6]", "HG0",      "first head of the pass"),
        ("w1",        "DST",      "channel partials"),
        ("w2",        "P",        "weight rows, row r at P + r x PSTRIDE"),
        ("w3",        "BASE",     "V plane base B"),
        ("w4[8:0]",   "E64",      "E / 64"),
        ("w4[10:9]",  "DCODE",    ""),
        ("w4[13:11]", "TQ_M1",    "T_q - 1"),
        ("w4[16:14]", "GS_M1",    "GS - 1"),
        ("w4[31:17]", "PSTRIDE",  "lines; 0 = n_groups"),
        ("w5[15:0]",  "U0",       "first group"),
        ("w5[31:16]", "N_GROUPS", ""),
        ("w6[15:0]",  "LG",       "local groups per channel"),
        ("w7",        "CSTRIDE",  "lines; 0 = R x D / 16"),
    ], values={"FMT": "KVFMT", "DCODE": "DCODE"}),
    "ASMAX": dict(src="AT §5.4, VU §5.11", owner="AT", fields=[
        ("w1",        "DST",     ""),
        ("w2",        "SRC",     "may equal DST"),
        ("w3",        "MVEC",    "row maxima, word r"),
        ("w4[15:0]",  "N_LINES", "1..4095"),
        ("w4[31:16]", "STRIDE",  "0 = n_lines"),
        ("w5",        "R",       "1..1024"),
        ("w6",        "SCALE",   "f32, finite, normal, > 0"),
    ]),
    "AMASK": dict(src="AT §5.5, VU §5.11", owner="AT", fields=[
        ("w1",        "DST",     ""),
        ("w2",        "SRC",     "rows at STRIDE"),
        ("w3",        "MASK",    "f32 mask rows"),
        ("w4[15:0]",  "N_LINES", ""),
        ("w4[31:16]", "STRIDE",  ""),
        ("w5[15:0]",  "R",       ""),
        ("w5[31:16]", "RPQ",     "rows per query (= n_head)"),
        ("w6",        "MSTRIDE", "lines between mask rows"),
        ("w7",        "MDST",    "row maxima out"),
    ]),
    "AMERGE": dict(src="AT §5.6, VU §5.11", owner="AT", fields=[
        ("w0[12]",    "INIT",    "copy instead of merge"),
        ("w1",        "ORUN",    "R rows of D/16 lines"),
        ("w2",        "PARTS",   "an ATTPV destination written with LSUM"),
        ("w3",        "MK",      "chunk row maxima"),
        ("w4",        "ST",      "m_run at ST, l_run at ST + ceil(R/16)"),
        ("w5[15:0]",  "R",       ""),
        ("w5[17:16]", "DCODE",   ""),
        ("w6",        "SCALE",   "f32, as ASMAX"),
        ("w7",        "CSTRIDE", "as the producing ATTPV"),
    ], values={"DCODE": "DCODE"}),
    "ANORM": dict(src="AT §5.7, VU §5.11", owner="AT", fields=[
        ("w1",        "OUT",   "row r at OUT + r x D/16"),
        ("w2",        "ORUN",  ""),
        ("w3",        "L_RUN", ""),
        ("w4[15:0]",  "R",     ""),
        ("w4[17:16]", "DCODE", ""),
    ], values={"DCODE": "DCODE"}),
}

# VOP functions (VU §5.2): (code, NAME, reads (subset of "abc"), rate F full / H half, result)
VOP_FUNCS = ("VU §5.2", [
    (0x00, "ADD",     "ab",  "F", "FA(a, b)"),
    (0x01, "SUB",     "ab",  "F", "FS(a, b)"),
    (0x02, "MUL",     "ab",  "F", "FM(a, b)"),
    (0x03, "MAX",     "ab",  "F", "MAXT(a, b)"),
    (0x04, "MULC",    "ac",  "F", "FM(a, c)"),
    (0x05, "ADDC",    "ac",  "F", "FA(a, c)"),
    (0x06, "SILU",    "a",   "H", "SILU(a)"),
    (0x07, "SWIGLU",  "ab",  "H", "FM(SILU(a), b)"),
    (0x08, "EXP",     "a",   "F", "EXPK(a, P1)"),
    (0x09, "COPY",    "a",   "F", "a (raw)"),
    (0x0A, "MIN",     "ab",  "F", "MINT(a, b)"),
    (0x0B, "SUBC",    "ac",  "F", "FS(a, c)"),
    (0x0C, "MULCB",   "abc", "F", "FM(FM(a, c), b)"),
    (0x0D, "MULCADD", "abc", "F", "FA(FM(a, c), b)"),
    (0x0E, "ADDMULC", "abc", "F", "FM(FA(a, b), c)"),
    (0x0F, "EXPSUB",  "ab",  "F", "EXPSUB(a, b)"),
    (0x10, "EXPSUBC", "ac",  "F", "EXPSUB(a, c)"),
    (0x11, "SIGMOID", "a",   "H", "SIG(a)"),
    (0x12, "GELU",    "a",   "H", "GELU(a), exact tanh form"),
    (0x13, "GEGLU",   "ab",  "H", "FM(GELU(a), b)"),
    (0x14, "TANH",    "a",   "H", "TANH(a)"),
    (0x15, "RECIP",   "a",   "H", "RC(a), correctly rounded"),
    (0x16, "SETC",    "c",   "F", "c (raw)"),
    (0x17, "GELU16",  "a",   "H", "ggml-CPU fp16 form (VU §6.5)"),
    (0x18, "GEGLU16", "ab",  "H", "GEGLU16(a, b) (VU §6.5)"),
])

# SOP functions (VU §5.8): (sub, NAME, fields used (others must be 0), effect)
SOP_FUNCS = ("VU §5.8", [
    (0,  "MOVI",  ["SD", "IMM"],        "sd <- imm (raw)"),
    (1,  "ADD",   ["SD", "SA", "SB"],   "sd <- FA(sa, sb)"),
    (2,  "MUL",   ["SD", "SA", "SB"],   "sd <- FM(sa, sb)"),
    (3,  "ADDI",  ["SD", "SA", "IMM"],  "sd <- FA(sa, imm)"),
    (4,  "MULI",  ["SD", "SA", "IMM"],  "sd <- FM(sa, imm)"),
    (5,  "SQRT",  ["SD", "SA"],         "sd <- SQ(sa)"),
    (6,  "RECIP", ["SD", "SA"],         "sd <- RC(sa)"),
    (7,  "LDS",   ["SD", "ADDR"],       "sd <- SPM word (raw)"),
    (8,  "STS",   ["SA", "ADDR"],       "SPM word <- sa (raw; exactly 4 bytes)"),
    (9,  "SUB",   ["SD", "SA", "SB"],   "sd <- FS(sa, sb)"),
    (10, "DIV",   ["SD", "SA", "SB"],   "sd <- FD(sa, sb)"),
    (11, "RSQRT", ["SD", "SA"],         "sd <- RC(SQ(sa)), two roundings"),
    (12, "MAX",   ["SD", "SA", "SB"],   "sd <- MAXT(sa, sb)"),
    (13, "MIN",   ["SD", "SA", "SB"],   "sd <- MINT(sa, sb)"),
    (14, "EXP",   ["SD", "SA"],         "sd <- EXPK(sa, P1)"),
])

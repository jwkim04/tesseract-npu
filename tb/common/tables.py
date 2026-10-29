# GENERATED FILE, DO NOT EDIT: produced by sw/tables/gen_tables.py from sw/tables/tsr_tables.py.
"""Tesseract v2 global tables for the cocotb benches (V2-EX-30).

Same names as sw/sim/include/tsr_tables.h without the TSR_ prefix. Besides the constants:
LAYOUTS (bit-field layouts), ENCODINGS (instruction fields), OP_UNIT / OP_LEGAL / OP_CAPS,
the PERF slot functions, and field_get / field_put / encode / decode.
"""
# ruff: noqa: E501

# -------------------------------------------------------------------------------------------------
# Scalar constants (CONSTANTS)
# -------------------------------------------------------------------------------------------------
ID_MAGIC              = 0x54535231  # EX §5.1: ID register: "TSR1" (unchanged from v1)
ISA_MAJOR             = 2           # EX §7.5: VERSION[15:8]
ISA_MINOR             = 0           # EX §7.5: VERSION[7:0]
VERSION_VALUE         = 0x200       # EX §7.5: VERSION register: ISA 2.0
CTL_VERSION           = 2           # EX §5.1: CTL_ID[7:0]: control-core version
ADAPTER_MAJOR         = 2           # EX §9.2: ADAPTER_VERSION[31:16] (ADAPTER_VERSION_MAJOR field)
DBG_VERSION           = 2           # DB §10.1: DBG_ID[7:0]; DBG_ID = 0 means the DBG block is absent
CANON_NAN             = 0x7FC00000  # VU §4.1: the only NaN an fp result has
INSN_WORDS            = 8           # EX §7.1: an instruction is 256 bits, w0..w7, little-endian
INSN_BYTES            = 32          # EX §7.1
INSN_BITS             = 256         # EX §7.1
X16_LSB               = 16          # EX §7.1: x16 = w0[31:16]; x16 bit i is w0 bit 16 + i; 0 unless the op defines it
X16_W                 = 16          # EX §7.1
LQ_DEPTH              = 8           # EX §4.2: U200 launch-queue depth (LQ_CFG[7:0])
LCQ_DEPTH             = 8           # EX §4.2: U200 completion-ring depth (LQ_CFG[15:8])
LOG_DEPTH             = 256         # EX §4.2: U200 log-ring words (LQ_CFG[27:16])
LCQ_DEPTH_MAX         = 8           # EX §4.2: the CSR map limits LCQ_DEPTH to 8
LOG_DEPTH_MAX         = 256         # EX §4.2: the CSR map limits LOG_DEPTH to 256
LCQ_REC_BYTES         = 32          # EX §4.2: record k at LCQ_REC + 32 x (k mod LCQ_DEPTH)
ARGBLK_MAX_BYTES      = 1024        # EX §4.3: argument block limit (ARG0 handle of a kernel with the argument-block flag)
ARGBLK_MAX_LINES      = 16          # SW §6.2: argblk_lines of a kernel table entry: 1..16
SEQ_HALF              = 0x80000000  # EX §4.9: launch q is covered by watermark s iff (s - q) mod 2^32 < SEQ_HALF
PANIC_MCAUSE          = 24          # EX §4.7, DB §6.3: tsr::panic enters the trap handler with this mcause, mtval = code
BOOT_ENTRY_OFF        = 0x80        # EX §3.4: the RISC-V fetches at RV_BOOT + 0x80
RV_BOOT_ALIGN         = 256         # EX §5.1: RV_BOOT alignment ([7:0] read 0)
CRC32_POLY            = 0xEDB88320  # EX §3.3: LOAD_CRC: zlib crc32 (reflected polynomial, init and final XOR 0xFFFFFFFF)
LOAD_TIMEOUT_CYCLES   = 4096        # EX §3.3: core cycles a LOAD_DATA word or a VERIFY read may take before LOAD_TIMEOUT
LOGFMT_VMA            = 0xF0000000  # SW §6.2, DB §6.1: VMA of the non-loaded .tsr_logfmt section (FMT log records point into it)
WDOG_L_RESET          = 24          # EX §5.1, §8.6: WDOG_CFG reset value: 2^24 core cycles without progress
SR_RV_WAIT_CYCLES     = 0x10000     # EX §8.5: SOFT_RESET step 1 waits at most 2^16 core cycles for RV_RESET_DONE
DMI_TIMEOUT_CYCLES    = 4096        # EX §5.1: core cycles a DMI access may take before DMI_TIMEOUT
BRIDGE_TIMEOUT_CYCLES = 4096        # CC §5.3: clk_rv cycles a core-window or SPM-bridge load may take before a load fault
ISSUE_WAIT_MAX        = 0x10000     # CC §7.3: clk_rv cycles a push without credit waits before a store fault and PUSH_TIMEOUT
TICKET_NONE           = 0xF0000000  # CC §5.1, EX §4.4: unit field 15: no dependency / no failing ticket
NUM_CH_MAX            = 16          # spec ISA: LD / ST name the channel in the 4-bit sub field (CFG0.NUM_CH, CH_BASE_LO/HI)
NUM_UNITS             = 6           # EX §7.3: unit ids 0..5 (6..13 reserved, 14 marker, 15 none)
UNIT_SLOTS            = 8           # EX §5.4: per-unit register slots u = 0..7 (absent units read 0)
Q_DEPTH_FIELD_W       = 5           # EX §5.1: Q_DEPTH: depth of unit u at [5u+4:5u]
CREDIT_FIELD_W        = 4           # CC §5.1: ISSUE_CREDIT: unit u at [4u+3:4u], saturating at 15
ISSUE_FIFO_DEPTH      = 32          # CC §7.2: issue-FIFO entries (LUTRAM, clk_rv -> core)
ERR_W                 = 4           # EX §8.1: internal error-code width (codes 0..15); CSR fields are 8 bits
PERF_SLOTS            = 64          # EX §5.1: PERF_0..PERF_63, 64-bit each
PERF_CH_BASE          = 5           # EX §6: CH_RD_BEATS[c] = 5 + c, CH_WAIT[c] = 5 + C + c, X = 5 + 2C
ITCM_BYTES            = 0x20000     # CC §4.1: U200 ITCM (RV_CFG[15:0] in KiB)
DTCM_BYTES            = 0x10000     # CC §4.1: U200 DTCM (RV_CFG[31:16] in KiB)
SPM_BYTES_U200        = 0x400000    # CC §4.1: U200 SPM size: the SPM windows span SPM_BYTES
DBG_NTBP              = 4           # DB §10.3: tile breakpoints
DBG_NRVWP             = 2           # DB §10.3: RISC-V data watchpoints
HIST_DEPTH            = 512         # DB §10.3: dispatch-history entries
REC_DEPTH             = 64          # DB §10.3: completion-record FIFO entries
REC_IRQ_LEVEL         = 32          # DB §8.2: irq_fast[4] while REC_COUNT >= 32 (REC_CFG.IRQ_HALF)
MATCHQ_DEPTH          = 64          # DB §5.2: tile-breakpoint match queue
TRW_CH                = 0           # DB §10.3, §8.3: channel of the trace-writer ring (in the ch0 IO region)
TRW_RECORD_BYTES      = 32          # DB §8.3: trace-writer record, two per 64-byte beat
TRW_RING_DEFAULT_MIB  = 64          # DB §8.3, EX §11: TSR_PROF_TRW_MIB default
DBG_INFO_MAGIC        = 0x47424454  # DB §6.3: __tsr_dbg word 0: "TDBG"
CRASH_MAGIC           = 0x50415254  # DB §6.3: __tsr_crash word 0: "TRAP"
CRASH_LOG_WORDS       = 44          # DB §6.1: TRAP log record payload: words 0..43 of __tsr_crash
VL                    = 16          # VU §5.10: CFG2.VL: elements per SPM line
NSREG                 = 16          # VU §4.4: s0..s15
VU_FRONT_LANES        = 16          # VU §5.10: CFG3[4:0] (U200)
VU_BACK_LANES         = 8           # VU §5.10: CFG3[9:5] (U200)
VU_LINE_FIELD_W       = 24          # VU §5.1: SPM line addresses and element counts m are w[23:0] with w[31:24] = 0
ROPE_MAX_HEAD_DIM     = 1024        # VU §5.4: head_dim is a multiple of 16 in [16, 1024]
TOPK_MAX_K            = 16          # VU §5.5: k in 1..16
VU_ACCEPT_AHEAD       = 2           # VU §7.8, CC §9.4: VU ops held at once (I3 for unit 3)
M_FLOOR               = 0xF1800000  # AT §6.3: -2^100: row-maximum floor of ATTQK MAXOUT and AMASK
ANORM_L_FLOOR         = 0x800000    # AT §5.7: 2^-126: ANORM divides by MAXT(l, 2^-126)
ATT_GROUP_CELLS       = 16          # AT §4.2: KV cells per group (groups round-robin over the channels)
ATT_MAX_GROUPS        = 4095        # AT §5.2: n_groups in [1, 4095]
ATT_MAX_G             = 64          # AT §5: G (heads per KV head) in 1..64
ATT_MAX_PASS_ROWS     = 128         # AT §5.8: (E/D) x GS x T_q pass rows at most
ATT_MAX_R             = 1024        # AT §5.8: R in [1, 1024] for ASMAX, AMASK, AMERGE, ANORM
ATT_MAX_LINES         = 4095        # AT §5.4: ASMAX / AMASK n_lines in [1, 4095]
KVST_MAX_ROWS         = 64          # AT §5.1: n_rows in 1..64
PCAP_U200             = 9344        # AT §5.8: p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX) x 16

# -------------------------------------------------------------------------------------------------
# Shared value sets (ENUMS)
# -------------------------------------------------------------------------------------------------
FMT_Q8_0   = 0  # spec ISA, VU §5.6: weight format Q8_0 (MVM, DQROW sub)
FMT_Q4_K   = 1  # spec ISA, VU §5.6: weight format Q4_K; 2..15 BAD_FMT
QA_A32     = 0  # spec ISA: QACT kind: block_q8_0 rules, block 32
QA_A256    = 1  # spec ISA: QACT kind: block_q8_K rules, block 256, with bsums
KVFMT_Q8_0 = 0  # AT §5.2: KV format of ATTQK / ATTPV sub; else BAD_FMT
DCODE_D64  = 0  # AT §5: head dimension D = 64 x 2^dcode
DCODE_D128 = 1  # AT §5
DCODE_D256 = 2  # AT §5
DCODE_D512 = 3  # AT §5

# -------------------------------------------------------------------------------------------------
# Field access helpers
# -------------------------------------------------------------------------------------------------

def field_get(v, mask, lsb):
    """Field value from a word, with a generated mask / _LSB pair."""
    return (v & mask) >> lsb


def field_put(mask, lsb, x):
    """Word bits of a field value, with a generated mask / _LSB pair."""
    return (x << lsb) & mask


# -------------------------------------------------------------------------------------------------
# Host CSR map: core CSR port, U200 s_axi_control 0x1000 + offset (EX §5.1)
# -------------------------------------------------------------------------------------------------
CSR_ID             = 0x000       # [R] EX §5.1: "TSR1" (unchanged)
CSR_ID_RESET       = 0x54535231
CSR_VERSION        = 0x004       # [R] EX §5.1, §7.5: [15:8] ISA major 2, [7:0] minor 0
CSR_VERSION_RESET  = 0x200
CSR_CFG0           = 0x008       # [R] EX §5.1: NUM_CH, LANES, T_MAX (unchanged)
CSR_CFG1           = 0x00C       # [R] EX §5.1: SPM size in lines (unchanged)
CSR_CFG2           = 0x010       # [R] EX §5.1: VL, NSREG, K_MAX/256 (unchanged)
CSR_CAPS           = 0x014       # [R] EX §5.1, §7.5: capability bits
CSR_CFG3           = 0x018       # [R] EX §5.1, §7.5: [15:0] VU, [31:16] AT
CSR_CTL_ID         = 0x01C       # [R] EX §5.1: control-core version and features
CSR_CTRL           = 0x020       # [RW/W1] EX §5.1, §5.2: device control
CSR_STATUS         = 0x024       # [R/W1C] EX §5.1, §5.3: device status; W1C clears the sticky bits only
CSR_RV_BOOT        = 0x028       # [RW] EX §5.1: boot vector, taken when RV_RUN rises
CSR_RV_FREQ        = 0x02C       # [R] EX §5.1: clk_rv cycles in the last 2^16 core cycles; 0 = no clk_rv
CSR_RV_CFG         = 0x030       # [R] EX §5.1: [15:0] ITCM KiB, [31:16] DTCM KiB
CSR_WDOG_CFG       = 0x034       # [RW] EX §5.1, §8.6: unit watchdog limit
CSR_WDOG_CFG_RESET = 0x18
CSR_RT_STATE       = 0x038       # [R] EX §5.1, §4.5: copy of the runtime's RT_STATE
CSR_RT_INFO        = 0x03C       # [R] EX §5.1, §4.5: copy of the runtime's RT_INFO
CSR_LQ_W_BASE      = 0x040       # [RW] EX §5.1, §4.3: descriptor staging LQ_W0..LQ_W7; kept after a push
CSR_LQ_W_N         = 8
CSR_LQ_W_STRIDE    = 4


def CSR_LQ_W(i):
    return 0x040 + 4 * i

CSR_LQ_PUSH        = 0x060  # [W] EX §5.1: push the staged descriptor (value ignored); dropped with LQ_OVERFLOW when full
CSR_LQ_SEQ         = 0x064  # [R] EX §5.1, §4.2: PUSHED
CSR_LQ_TAKEN       = 0x068  # [R] EX §5.1, §4.2: TAKEN
CSR_LQ_CANCEL      = 0x06C  # [RW] EX §5.1, §4.9: cancel watermark
CSR_LQ_STOP        = 0x070  # [RW] EX §5.1, §4.9: stop watermark
CSR_LCQ_WR         = 0x074  # [R] EX §5.1, §4.2: records posted
CSR_LCQ_RD         = 0x078  # [RW] EX §5.1, §4.2: records consumed by the host (EX §4.2 write rule)
CSR_LQ_CFG         = 0x07C  # [R] EX §5.1: ring depths and watermark valid bits
CSR_PERF_INFO      = 0x080  # [R] EX §5.1: counters implemented, extension index X
CSR_LCQ_ANSWERED   = 0x084  # [R] EX §5.1, §4.2: ANSWERED
CSR_Q_DEPTH        = 0x088  # [R] EX §5.1: queue depth per unit; 0 = absent
CSR_LOG_WR         = 0x090  # [R] EX §5.1, §4.10: log words accepted
CSR_LOG_RD         = 0x094  # [RW] EX §5.1, §4.10: log words consumed by the host
CSR_LOG_CFG        = 0x098  # [RW] EX §5.1, §4.10: log configuration
CSR_LOG_CFG_RESET  = 0xF1
CSR_LOG_DROPPED    = 0x09C  # [R] EX §5.1: words dropped, saturating; cleared by SOFT_RESET
CSR_LOAD_ADDR      = 0x0C0  # [RW] EX §5.1, §3.3: TCM byte address (word aligned); a write restarts LOAD_CRC
CSR_LOAD_DATA      = 0x0C4  # [RW] EX §5.1, §3.3: write: store the word, address += 4; read: the word (only while RV_RUN = 0 and not SR_BUSY)
CSR_LOAD_CRC       = 0x0C8  # [R] EX §5.1, §3.3: CRC-32 of the words accepted, or read by VERIFY
CSR_LOAD_COUNT     = 0x0CC  # [RW] EX §5.1, §3.3: word count for VERIFY
CSR_LOAD_CTRL      = 0x0D0  # [W] EX §5.1, §3.3: bit0 VERIFY
CSR_LOAD_STATUS    = 0x0D4  # [R] EX §5.1, §3.3: bit0 VERIFY_BUSY
CSR_PERF_LO_BASE   = 0x100  # [R] EX §5.1, §6: counter i bits [31:0]; a host LO read snapshots the counter for the next HI read
CSR_PERF_LO_N      = 64
CSR_PERF_LO_STRIDE = 8


def CSR_PERF_LO(i):
    return 0x100 + 8 * i

CSR_PERF_HI_BASE   = 0x104  # [R] EX §5.1, §6: counter i bits [63:32] (the snapshot)
CSR_PERF_HI_N      = 64
CSR_PERF_HI_STRIDE = 8


def CSR_PERF_HI(i):
    return 0x104 + 8 * i

CSR_LCQ_REC_BASE   = 0x300  # [R] EX §5.1, §4.2: completion ring: record k at 0x300 + 32 (k mod LCQ_DEPTH), word j at +4j
CSR_LCQ_REC_N      = 8
CSR_LCQ_REC_STRIDE = 32
CSR_LCQ_REC_WORDS  = 8


def CSR_LCQ_REC(i, j):
    return 0x300 + 32 * i + 4 * j

CSR_LOG_RING_BASE   = 0x400  # [R] EX §5.1, §4.10: log word k at 0x400 + 4 (k mod LOG_DEPTH)
CSR_LOG_RING_N      = 256
CSR_LOG_RING_STRIDE = 4


def CSR_LOG_RING(i):
    return 0x400 + 4 * i

CSR_DBG_BASE  = 0x800  # [DB] EX §5.1, DB §10.1: DBG block (DBG_CSR rows); undefined offsets read 0
CSR_DBG_BYTES = 768
# 0xB00..0xBFC reserved (EX §5.1)
CSR_DMI_BASE   = 0xC00  # [R/W] EX §5.1, DB §3.2: debug module register r at 0xC00 + 4r; acts only while DBG_CFG.DMI_EN = 1
CSR_DMI_N      = 128
CSR_DMI_STRIDE = 4


def CSR_DMI(i):
    return 0xC00 + 4 * i

CSR_Q_DONE_BASE   = 0xE00  # [R] EX §5.4: DONE_SEQ[u], 28 bits
CSR_Q_DONE_N      = 8
CSR_Q_DONE_STRIDE = 4


def CSR_Q_DONE(i):
    return 0xE00 + 4 * i

CSR_Q_ISSUED_BASE   = 0xE20  # [R] EX §5.4: ISSUED_SEQ[u], 28 bits (synchronised copy)
CSR_Q_ISSUED_N      = 8
CSR_Q_ISSUED_STRIDE = 4


def CSR_Q_ISSUED(i):
    return 0xE20 + 4 * i

CSR_Q_STATE_BASE   = 0xE40  # [R] EX §5.4: queue state of unit u
CSR_Q_STATE_N      = 8
CSR_Q_STATE_STRIDE = 4


def CSR_Q_STATE(i):
    return 0xE40 + 4 * i

CSR_Q_FIFO           = 0xE60  # [R] EX §5.4: issue-FIFO entries waiting for the dispatcher
CSR_Q_PIDX           = 0xE64  # [R] EX §5.4: push index of the dispatcher's next ticketed entry (= DBG_NEXT_PIDX); firmware never uses it
CSR_Q_ERR_SEQ_BASE   = 0xE80  # [R] EX §5.4: first error record of unit u: ERR_SEQ
CSR_Q_ERR_SEQ_N      = 8
CSR_Q_ERR_SEQ_STRIDE = 16


def CSR_Q_ERR_SEQ(i):
    return 0xE80 + 16 * i

CSR_Q_ERR_CODE_BASE   = 0xE84  # [R] EX §5.4: ERR_CODE ([7:0])
CSR_Q_ERR_CODE_N      = 8
CSR_Q_ERR_CODE_STRIDE = 16


def CSR_Q_ERR_CODE(i):
    return 0xE84 + 16 * i

CSR_Q_ERR_INFO_BASE   = 0xE88  # [R] EX §5.4: ERR_INFO
CSR_Q_ERR_INFO_N      = 8
CSR_Q_ERR_INFO_STRIDE = 16


def CSR_Q_ERR_INFO(i):
    return 0xE88 + 16 * i

CSR_Q_ERR_W0_BASE   = 0xE8C  # [R] EX §5.4: ERR_W0
CSR_Q_ERR_W0_N      = 8
CSR_Q_ERR_W0_STRIDE = 16


def CSR_Q_ERR_W0(i):
    return 0xE8C + 16 * i

CSR_Q_ERR_PIDX_BASE   = 0xF00  # [R] EX §5.4: push index of unit u's first error
CSR_Q_ERR_PIDX_N      = 8
CSR_Q_ERR_PIDX_STRIDE = 4


def CSR_Q_ERR_PIDX(i):
    return 0xF00 + 4 * i

# 0xF20..0xFFC reserved (EX §5.1)

# -------------------------------------------------------------------------------------------------
# DBG block: host CSR 0x800..0xAFC (DB §10.1)
# -------------------------------------------------------------------------------------------------
CSR_DBG_ID              = 0x800  # [R] DB §10.1: version and features; 0 = block absent
CSR_DBG_CTRL            = 0x804  # [W] DB §10.1: pulses, reads 0
CSR_DBG_CFG             = 0x808  # [RW] DB §10.1: configuration
CSR_DBG_CFG_RESET       = 0x30
CSR_DBG_STATUS          = 0x80C  # [R/W1C] DB §10.1: levels and sticky bits
CSR_DBG_STOP_PIDX       = 0x810  # [R] DB §10.1: T (valid while STOP_ACTIVE)
CSR_DBG_STOP_AT         = 0x814  # [RW] DB §10.1: write P: S <- P, valid; read: S (valid iff S_VALID)
CSR_DBG_STEP_N          = 0x818  # [RW] DB §10.1: entries per QSTEP
CSR_DBG_STEP_N_RESET    = 0x1
CSR_DBG_NEXT_PIDX       = 0x81C  # [R] DB §10.1: push index the dispatcher assigns to its next ticketed entry
CSR_DBG_STEP_PIDX       = 0x820  # [R] DB §10.1: X of the current / last QSTEP
CSR_DBG_MATCHQ          = 0x824  # [R] DB §10.1: pending match thresholds
CSR_DBG_MTIME_LO        = 0x828  # [R] DB §10.1: live core time (read HI, LO, HI)
CSR_DBG_MTIME_HI        = 0x82C  # [R] DB §10.1
CSR_DBG_QSTATE          = 0x834  # [R] DB §10.1: dispatcher state, outstanding entries
CSR_DBG_RV_STATE        = 0x840  # [R] DB §10.1: RISC-V state and sample sequence
CSR_DBG_RV_PC           = 0x844  # [R] DB §10.1, §6.2: last crash-dump sample: current PC
CSR_DBG_RV_NPC          = 0x848  # [R] DB §10.1, §6.2: next PC
CSR_DBG_RV_DADDR        = 0x84C  # [R] DB §10.1, §6.2: last data address
CSR_DBG_RV_EPC          = 0x850  # [R] DB §10.1, §6.2: exception PC
CSR_DBG_RV_EADDR        = 0x854  # [R] DB §10.1, §6.2: exception address
CSR_DBG_RV_WPADDR       = 0x858  # [R] DB §10.1: address of the last RV-watchpoint hit
CSR_DBG_RVWP_CTL_BASE   = 0x860  # [RW] DB §10.1: RV watchpoint j < 2
CSR_DBG_RVWP_CTL_N      = 2
CSR_DBG_RVWP_CTL_STRIDE = 16


def CSR_DBG_RVWP_CTL(i):
    return 0x860 + 16 * i

CSR_DBG_RVWP_LO_BASE   = 0x864  # [RW] DB §10.1: byte range [LO, HI)
CSR_DBG_RVWP_LO_N      = 2
CSR_DBG_RVWP_LO_STRIDE = 16


def CSR_DBG_RVWP_LO(i):
    return 0x864 + 16 * i

CSR_DBG_RVWP_HI_BASE   = 0x868  # [RW] DB §10.1
CSR_DBG_RVWP_HI_N      = 2
CSR_DBG_RVWP_HI_STRIDE = 16


def CSR_DBG_RVWP_HI(i):
    return 0x868 + 16 * i

CSR_DBG_RVWP_HITS_BASE   = 0x86C  # [R/WC] DB §10.1: hits; a write clears
CSR_DBG_RVWP_HITS_N      = 2
CSR_DBG_RVWP_HITS_STRIDE = 16


def CSR_DBG_RVWP_HITS(i):
    return 0x86C + 16 * i

CSR_DBG_TBP_CTL_BASE   = 0x880  # [RW] DB §10.1: tile breakpoint i < 4
CSR_DBG_TBP_CTL_N      = 4
CSR_DBG_TBP_CTL_STRIDE = 16


def CSR_DBG_TBP_CTL(i):
    return 0x880 + 16 * i

CSR_DBG_TBP_A_BASE   = 0x884  # [RW] DB §10.1, §5.3
CSR_DBG_TBP_A_N      = 4
CSR_DBG_TBP_A_STRIDE = 16


def CSR_DBG_TBP_A(i):
    return 0x884 + 16 * i

CSR_DBG_TBP_B_BASE   = 0x888  # [RW] DB §10.1, §5.3
CSR_DBG_TBP_B_N      = 4
CSR_DBG_TBP_B_STRIDE = 16


def CSR_DBG_TBP_B(i):
    return 0x888 + 16 * i

CSR_DBG_TBP_HITS_BASE   = 0x88C  # [R/WC] DB §10.1: matches; a write clears
CSR_DBG_TBP_HITS_N      = 4
CSR_DBG_TBP_HITS_STRIDE = 16


def CSR_DBG_TBP_HITS(i):
    return 0x88C + 16 * i

CSR_DBG_RDOUT_BASE   = 0x8C0  # [R] DB §10.1, §5.5: outstanding read bursts: channel c in byte c mod 4 of word c / 4
CSR_DBG_RDOUT_N      = 4
CSR_DBG_RDOUT_STRIDE = 4


def CSR_DBG_RDOUT(i):
    return 0x8C0 + 4 * i

CSR_DBG_WROUT_BASE   = 0x8D0  # [R] DB §10.1, §5.5: outstanding write bursts, same packing
CSR_DBG_WROUT_N      = 4
CSR_DBG_WROUT_STRIDE = 4


def CSR_DBG_WROUT(i):
    return 0x8D0 + 4 * i

CSR_DBG_U_STATE_BASE   = 0x900  # [R] DB §10.1, §5.5: unit u < 8
CSR_DBG_U_STATE_N      = 8
CSR_DBG_U_STATE_STRIDE = 16


def CSR_DBG_U_STATE(i):
    return 0x900 + 16 * i

CSR_DBG_U_OLDEST_BASE   = 0x904  # [R] DB §10.1, §5.5: ticket of the oldest outstanding entry; TICKET_NONE if none
CSR_DBG_U_OLDEST_N      = 8
CSR_DBG_U_OLDEST_STRIDE = 16


def CSR_DBG_U_OLDEST(i):
    return 0x904 + 16 * i

CSR_DBG_U_OPIDX_BASE   = 0x908  # [R] DB §10.1, §5.5: its push index
CSR_DBG_U_OPIDX_N      = 8
CSR_DBG_U_OPIDX_STRIDE = 16


def CSR_DBG_U_OPIDX(i):
    return 0x908 + 16 * i

CSR_DBG_U_AGE_BASE   = 0x90C  # [R] DB §10.1, §5.5: core cycles since it became the oldest (saturating)
CSR_DBG_U_AGE_N      = 8
CSR_DBG_U_AGE_STRIDE = 16


def CSR_DBG_U_AGE(i):
    return 0x90C + 16 * i

CSR_DBG_TRW_BASE       = 0x980  # [RW] DB §10.1, §8.3: ring base handle, 16-beat aligned (beat [3:0] read 0)
CSR_DBG_TRW_LINES      = 0x984  # [RW] DB §10.1, §8.3: ring size in 64-B lines, power of two >= 16 (else ignored, PTR_ERR)
CSR_DBG_TRW_WPTR       = 0x988  # [R] DB §10.1, §8.3: records written (mod 2^32)
CSR_DBG_TRW_RPTR       = 0x98C  # [RW] DB §10.1, §8.3: records consumed; beyond WPTR ignored (PTR_ERR)
CSR_DBG_TRW_DROPS      = 0x990  # [R] DB §10.1, §8.3: records dropped (saturating)
CSR_DBG_TRW_MASK       = 0x994  # [RW] DB §10.1, §8.3: record classes written
CSR_DBG_TRW_MASK_RESET = 0x7FF
CSR_SREG_BASE          = 0x9C0  # [R] EX §5.1, §7.4, DB §10.1, VU §4.4: s0..s15 (f32 bits); present whenever the VU is built, also with HAS_DBG = 0
CSR_SREG_N             = 16
CSR_SREG_STRIDE        = 4


def CSR_SREG(i):
    return 0x9C0 + 4 * i

CSR_DBG_HIST_HEAD     = 0xA00  # [R] DB §10.1, §5.6: next push index to be recorded
CSR_DBG_HIST_IDX      = 0xA04  # [RW] DB §10.1, §5.6: push index to read
CSR_DBG_HIST_INFO     = 0xA08  # [R] DB §10.1, §5.6
CSR_DBG_HIST_PIDX     = 0xA0C  # [R] DB §10.1, §5.6
CSR_DBG_HIST_TICKET   = 0xA10  # [R] DB §10.1, §5.6
CSR_DBG_HIST_DEP0     = 0xA14  # [R] DB §10.1, §5.6
CSR_DBG_HIST_DEP1     = 0xA18  # [R] DB §10.1, §5.6
CSR_DBG_HIST_TIME     = 0xA1C  # [R] DB §10.1, §5.6: dispatch time MTIME[31:0]
CSR_DBG_HIST_PC       = 0xA20  # [R] DB §10.1, §5.6: push PC
CSR_DBG_HIST_W_BASE   = 0xA40  # [R] DB §10.1, §5.6: op words w0..w7
CSR_DBG_HIST_W_N      = 8
CSR_DBG_HIST_W_STRIDE = 4


def CSR_DBG_HIST_W(i):
    return 0xA40 + 4 * i


# -------------------------------------------------------------------------------------------------
# RISC-V control window: offsets from RV_CTL_BASE (CC §5, EX §4.5, DB §10.2)
# -------------------------------------------------------------------------------------------------
CW_ISSUE_W_BASE   = 0x000  # [RW] CC §5.1: staged instruction words; kept after a push
CW_ISSUE_W_N      = 8
CW_ISSUE_W_STRIDE = 4


def CW_ISSUE_W(i):
    return 0x000 + 4 * i

CW_ISSUE_W_RESET    = 0x0
CW_ISSUE_DEP0       = 0x020       # [RW] CC §5.1: dependency ticket; reset to TICKET_NONE by every push
CW_ISSUE_DEP0_RESET = 0xF0000000
CW_ISSUE_DEP1       = 0x024       # [RW] CC §5.1: same
CW_ISSUE_DEP1_RESET = 0xF0000000
CW_ISSUE_PUSH       = 0x028       # [W] CC §5.1: push; a push without a credit completes late (safety net)
CW_ISSUE_TICKET     = 0x02C       # [R] CC §5.1: ticket of the last push or QCTRL.QCLEAR
CW_ISSUE_CREDIT     = 0x030       # [R] CC §5.1: credit per unit
CW_ISSUE_UNITS      = 0x034       # [R] CC §5.1: bit u = unit u is built
CW_ISSUE_STATUS     = 0x038       # [R/W1C] CC §5.1: push timeout and error interrupt
CW_ISSUE_PIDX       = 0x03C       # [R] CC §5.1: push index the next ticketed entry will receive (+1 per ISSUE_PUSH and QCLEAR); exact
CW_DONE_SEQ_BASE    = 0x040       # [R] CC §5.1: [27:0] completions of unit u (synchronised copy; may lag, never leads)
CW_DONE_SEQ_N       = 8
CW_DONE_SEQ_STRIDE  = 4


def CW_DONE_SEQ(i):
    return 0x040 + 4 * i

CW_ISSUED_SEQ_BASE   = 0x080  # [R] CC §5.1: [27:0] sequence the next push to unit u gets
CW_ISSUED_SEQ_N      = 8
CW_ISSUED_SEQ_STRIDE = 4


def CW_ISSUED_SEQ(i):
    return 0x080 + 4 * i

CW_ERR_SEQ_BASE   = 0x0C0  # [R] CC §5.1: first error of unit u since the last clear: ERR_SEQ
CW_ERR_SEQ_N      = 8
CW_ERR_SEQ_STRIDE = 16


def CW_ERR_SEQ(i):
    return 0x0C0 + 16 * i

CW_ERR_CODE_BASE   = 0x0C4  # [R] CC §5.1: ERR_CODE; 0 = none
CW_ERR_CODE_N      = 8
CW_ERR_CODE_STRIDE = 16


def CW_ERR_CODE(i):
    return 0x0C4 + 16 * i

CW_ERR_INFO_BASE   = 0x0C8  # [R] CC §5.1: ERR_INFO
CW_ERR_INFO_N      = 8
CW_ERR_INFO_STRIDE = 16


def CW_ERR_INFO(i):
    return 0x0C8 + 16 * i

CW_ERR_W0_BASE   = 0x0CC  # [R] CC §5.1: ERR_W0
CW_ERR_W0_N      = 8
CW_ERR_W0_STRIDE = 16


def CW_ERR_W0(i):
    return 0x0CC + 16 * i

CW_QCTRL             = 0x140       # [W] CC §5.1: queue control entries, in push order
CW_QSTATUS           = 0x144       # [R] CC §5.1: queue status
CW_WAIT_TICKET       = 0x148       # [RW] CC §5.1: irq_fast[0] pending while this ticket is done; unit 15 disarms
CW_WAIT_TICKET_RESET = 0xF0000000
CW_WAIT_CREDIT       = 0x14C       # [RW] CC §5.1: irq_fast[5] pending while that unit's credit != 0
CW_WAIT_CREDIT_RESET = 0x0
CW_REC_COUNT         = 0x160       # [R] DB §10.2, §8.2: completion records waiting (0..64)
CW_REC_DATA_BASE     = 0x164       # [R] DB §10.2, §8.2: head record words 0..3 (word table REC)
CW_REC_DATA_N        = 4
CW_REC_DATA_STRIDE   = 4


def CW_REC_DATA(i):
    return 0x164 + 4 * i

CW_REC_POP         = 0x174  # [W] DB §10.2, §8.2: drop the head record
CW_TRACE_MARK      = 0x178  # [W] DB §10.2, §8.4: marker through the issue FIFO
CW_REC_CFG         = 0x17C  # [RW] DB §10.2, §8.2: completion-record configuration
CW_REC_CFG_RESET   = 0x0
CW_ERR_PIDX_BASE   = 0x180  # [R] CC §5.1: push index of unit u's first error
CW_ERR_PIDX_N      = 8
CW_ERR_PIDX_STRIDE = 4


def CW_ERR_PIDX(i):
    return 0x180 + 4 * i

# 0x1C0..0x1C0 reserved (CC §5.1)
CW_SPM_LB_INV       = 0x1C4  # [W] CC §5.1, §10.5: invalidate the SPM line buffer
CW_LQ_HEAD_W_BASE   = 0x200  # [R] EX §4.5: the descriptor the next LQ_POP takes (valid while LQ_AVAIL != 0)
CW_LQ_HEAD_W_N      = 8
CW_LQ_HEAD_W_STRIDE = 4


def CW_LQ_HEAD_W(i):
    return 0x200 + 4 * i

CW_LQ_AVAIL     = 0x220  # [R] EX §4.5: PUSHED - TAKEN as seen in clk_rv
CW_LQ_HEAD_SEQ  = 0x224  # [R] EX §4.5: TAKEN = the sequence number of the head
CW_LQ_POP       = 0x228  # [W] EX §4.5: take the head; access fault if LQ_AVAIL = 0 or ACTIVE
CW_LQ_CANCEL    = 0x22C  # [R] EX §4.5: copy of the host's cancel watermark
CW_LQ_STOP      = 0x230  # [R] EX §4.5: copy of the host's stop watermark
CW_LQ_FLAGS     = 0x234  # [R] EX §4.5: watermark valid bits, ACTIVE, PLAT_ERR
CW_LCQ_W_BASE   = 0x240  # [RW] EX §4.5: record staging; kept after a post
CW_LCQ_W_N      = 8
CW_LCQ_W_STRIDE = 4


def CW_LCQ_W(i):
    return 0x240 + 4 * i

CW_LCQ_POST          = 0x260       # [W] EX §4.5: post the staged record; access fault if LCQ_FREE = 0, or ANSWER while not ACTIVE
CW_LCQ_FREE          = 0x264       # [R] EX §4.5: LCQ_DEPTH - (LCQ_WR - LCQ_RD) as seen in clk_rv (may under-report)
CW_LOG_PUSH          = 0x280       # [W] EX §4.5: one log word; dropped and counted when the ring is full, never faults
CW_LOG_FREE          = 0x284       # [R] EX §4.5: free words (may under-report)
CW_LOG_CFG           = 0x288       # [R] EX §4.5: copy of the host's LOG_CFG
CW_RT_STATE          = 0x290       # [RW] EX §4.5: runtime phase, mirrored to host 0x038; 0 after a RISC-V reset
CW_RT_STATE_RESET    = 0x0
CW_RT_INFO           = 0x294       # [RW] EX §4.5: runtime- or kernel-defined progress word, mirrored to host 0x03C
CW_MTIME_LO          = 0x300       # [R] CC §5.2: core clock cycles since reset (gray-synchronised)
CW_MTIME_HI          = 0x304       # [R] CC §5.2
CW_MTIMECMP_LO       = 0x308       # [RW] CC §5.2: irq_timer while MTIME >= MTIMECMP
CW_MTIMECMP_LO_RESET = 0xFFFFFFFF
CW_MTIMECMP_HI       = 0x30C       # [RW] CC §5.2
CW_MTIMECMP_HI_RESET = 0xFFFFFFFF
CW_DBG_FLAGS         = 0x400       # [R] DB §10.2: DB state for the runtime
# 0x404..0x7FC reserved (DB §10.2)

# -------------------------------------------------------------------------------------------------
# XRT adapter s_axi_control map (EX §9.2)
# -------------------------------------------------------------------------------------------------
# 0x000..0x03C reserved (EX §9.2)
SAXI_CH_BASE_LO_BASE   = 0x040  # [RW] EX §9.2: device byte address of the channel-c buffer, bits [31:0]; 4 KiB aligned; written while MEM_EN = 0
SAXI_CH_BASE_LO_N      = 16
SAXI_CH_BASE_LO_STRIDE = 8


def SAXI_CH_BASE_LO(i):
    return 0x040 + 8 * i

SAXI_CH_BASE_HI_BASE   = 0x044  # [RW] EX §9.2: bits [63:32]
SAXI_CH_BASE_HI_N      = 16
SAXI_CH_BASE_HI_STRIDE = 8


def SAXI_CH_BASE_HI(i):
    return 0x044 + 8 * i

SAXI_ADAPTER_CTRL    = 0x0E0   # [RW] EX §9.2: MEM_EN
SAXI_ADAPTER_STATUS  = 0x0F0   # [R/W1C] EX §9.2: sticky adapter errors
SAXI_ADAPTER_VERSION = 0x0F4   # [R] EX §9.2: {16'h0002, 8'h00, NUM_CH}
SAXI_CORE_BASE       = 0x1000  # [-] EX §9.2: core CSR window: 0x1000 + offset, any time; writes need WSTRB = 0xF
SAXI_CORE_BYTES      = 4096

# -------------------------------------------------------------------------------------------------
# RISC-V address map (CC §4.1); region sizes are the U200 values
# -------------------------------------------------------------------------------------------------
RV_ITCM_BASE    = 0x0         # CC §4.1: code; data reads steal the fetch port (byte/half/word, fetch)
RV_ITCM_SIZE    = 0x20000
RV_ITCM_LIMIT   = 0x100000    # end of the reserved range (exclusive)
RV_DTCM_BASE    = 0x100000    # CC §4.1: .rodata, .data, .bss, stack (byte/half/word)
RV_DTCM_SIZE    = 0x10000
RV_DTCM_LIMIT   = 0x200000    # end of the reserved range (exclusive)
RV_DM_BASE      = 0x10000000  # CC §4.1: debug module memory (riscv-dbg layout) (word, fetch)
RV_DM_SIZE      = 0x1000
RV_DM_LIMIT     = 0x10001000  # end of the reserved range (exclusive)
RV_CTL_BASE     = 0x20000000  # CC §4.1: control window (CTL_WINDOW) (word)
RV_CTL_SIZE     = 0x1000
RV_CTL_LIMIT    = 0x20001000  # end of the reserved range (exclusive)
RV_CORE_BASE    = 0x20001000  # CC §4.1, §5.3: core window: read-only view of the host CSR map (same offsets) (word loads)
RV_CORE_SIZE    = 0x1000
RV_CORE_LIMIT   = 0x20002000  # end of the reserved range (exclusive)
RV_SPM_BASE     = 0x40000000  # CC §4.1: SPM window, exact: 0x4000_0000 + 64 x line + offset (byte/half/word)
RV_SPM_SIZE     = 0x400000
RV_SPM_LIMIT    = 0x40400000  # end of the reserved range (exclusive)
RV_SPM_LB_BASE  = 0x50000000  # CC §4.1, §10.5: SPM line-buffered alias (byte/half/word)
RV_SPM_LB_SIZE  = 0x400000
RV_SPM_LB_LIMIT = 0x50400000  # end of the reserved range (exclusive)

# -------------------------------------------------------------------------------------------------
# Ibex fast interrupts irq_fast[i] (CC §12.3)
# -------------------------------------------------------------------------------------------------
IRQ_FAST_WAIT_TICKET = 0  # CC §12.3: WAIT_TICKET done
IRQ_FAST_ERR         = 1  # CC §12.3: ISSUE_STATUS.ERR_IRQ
IRQ_FAST_LQ          = 2  # CC §12.3, EX §4.5: LQ_AVAIL != 0 and not ACTIVE
IRQ_FAST_LCQ         = 3  # CC §12.3, EX §4.5: LCQ_FREE != 0
IRQ_FAST_REC         = 4  # CC §12.3, DB §8.2: REC_COUNT >= 32 with REC_CFG.IRQ_HALF
IRQ_FAST_CREDIT      = 5  # CC §12.3: WAIT_CREDIT
IRQ_FAST_DB_RSVD     = 6  # CC §12.3, DB §10.2: reserved for DB

# -------------------------------------------------------------------------------------------------
# DMI registers (DB §9.1); host CSR offset = CSR_DMI(address)
# -------------------------------------------------------------------------------------------------
DMI_DATA0      = 4   # DB §9.1
DMI_DATA1      = 5   # DB §9.1
DMI_DMCONTROL  = 16  # DB §9.1: haltreq, resumereq, ackhavereset, ndmreset, dmactive
DMI_DMSTATUS   = 17  # DB §9.1
DMI_HARTINFO   = 18  # DB §9.1
DMI_ABSTRACTCS = 22  # DB §9.1: progbufsize, busy, cmderr, datacount
DMI_COMMAND    = 23  # DB §9.1: Access Register, postexec-only
DMI_PROGBUF0   = 32  # DB §9.1: progbuf0..7
DMI_PROGBUF0_N = 8
DMI_SBCS       = 56  # DB §9.1
DMI_SBADDRESS0 = 57  # DB §9.1
DMI_SBDATA0    = 60  # DB §9.1

# -------------------------------------------------------------------------------------------------
# DRAM map per channel (EX §11); byte offsets
# -------------------------------------------------------------------------------------------------
REGION_OPL_OFF    = 0x0         # EX §11: ch0: op lists, replay push streams, argument blocks, code overlays (was CMD)
REGION_OPL_SIZE   = 0x4000000
REGION_IO_OFF     = 0x4000000   # EX §11: ch0: host <-> device tensors; compute chunks; the TRW ring while profiling
REGION_IO_SIZE    = 0x10000000
REGION_AUX_OFF    = 0x14000000  # EX §11: ch0: small f32 tensors (norm weights, RoPE tables)
REGION_AUX_SIZE   = 0xC000000
REGION_WEIGHT_OFF = 0x20000000  # EX §11: every channel: weights bottom-up, KV planes top-down, compute heap between

# -------------------------------------------------------------------------------------------------
# Bit-field layouts (in-place masks; _LSB, _W; values unshifted)
# -------------------------------------------------------------------------------------------------
# W0 (EX §7.1): instruction word w0 = {x16, flags, sub, opcode}
W0_OPCODE     = 0xFF        # [7:0] opcode
W0_OPCODE_LSB = 0
W0_OPCODE_W   = 8
W0_SUB        = 0xF00       # [11:8] op-specific sub code
W0_SUB_LSB    = 8
W0_SUB_W      = 4
W0_FLAGS      = 0xF000      # [15:12] op-specific flags
W0_FLAGS_LSB  = 12
W0_FLAGS_W    = 4
W0_X16        = 0xFFFF0000  # [31:16] op-specific field; must be 0 (ILLEGAL_OP) where the op does not assign it
W0_X16_LSB    = 16
W0_X16_W      = 16
# VERSION (EX §5.1, §7.5): ISA version
VERSION_MINOR     = 0xFF    # [7:0] 0
VERSION_MINOR_LSB = 0
VERSION_MINOR_W   = 8
VERSION_MAJOR     = 0xFF00  # [15:8] 2
VERSION_MAJOR_LSB = 8
VERSION_MAJOR_W   = 8
# CFG0 (EX §5.1): unchanged from v1
CFG0_NUM_CH     = 0xFF      # [7:0] channels C
CFG0_NUM_CH_LSB = 0
CFG0_NUM_CH_W   = 8
CFG0_LANES      = 0xFFF00   # [19:8] MAC lanes
CFG0_LANES_LSB  = 8
CFG0_LANES_W    = 12
CFG0_T_MAX      = 0xF00000  # [23:20] token slots
CFG0_T_MAX_LSB  = 20
CFG0_T_MAX_W    = 4
# CFG2 (EX §5.1): unchanged from v1
CFG2_VL            = 0xFF        # [7:0] vector length
CFG2_VL_LSB        = 0
CFG2_VL_W          = 8
CFG2_NSREG         = 0xFF00      # [15:8] s registers
CFG2_NSREG_LSB     = 8
CFG2_NSREG_W       = 8
CFG2_K_MAX_256     = 0xFFFF0000  # [31:16] K_MAX / 256
CFG2_K_MAX_256_LSB = 16
CFG2_K_MAX_256_W   = 16
# CAPS (EX §7.5): capability bits; retired bits are never set and never reused
CAPS_VU          = 0x1     # [0] VOP, RED, ROPE and SOP present (VU §5.10)
CAPS_VU_LSB      = 0
CAPS_VU_W        = 1
CAPS_TOPK        = 0x8     # [3] TOPK present (RTL from M8; the simulator from M6)
CAPS_TOPK_LSB    = 3
CAPS_TOPK_W      = 1
CAPS_ATT         = 0x20    # [5] the seven AT ops present (KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM)
CAPS_ATT_LSB     = 5
CAPS_ATT_W       = 1
CAPS_DQROW       = 0x40    # [6] DQROW in the SPM form (RTL from M8; the simulator from M6)
CAPS_DQROW_LSB   = 6
CAPS_DQROW_W     = 1
CAPS_VMASK       = 0x80    # [7] VMASK (M9)
CAPS_VMASK_LSB   = 7
CAPS_VMASK_W     = 1
CAPS_VU_RSVD     = 0xF00   # [11:8] reserved for VU extensions, 0
CAPS_VU_RSVD_LSB = 8
CAPS_VU_RSVD_W   = 4
CAPS_AT_RSVD     = 0xF000  # [15:12] reserved for AT, 0 in v2.0
CAPS_AT_RSVD_LSB = 12
CAPS_AT_RSVD_W   = 4
# bit 2 retired (v1 RLD / MULI / LOOP), never set, never reused
# bit 4 retired (v1 WAITM / JUMP), never set, never reused
# CFG3 (EX §7.5, VU §5.10, AT §5.10): [15:0] VU, [31:16] AT
CFG3_VU_FRONT     = 0x1F        # [4:0] VU front lanes with an EXP core (information for cycle models)
CFG3_VU_FRONT_LSB = 0
CFG3_VU_FRONT_W   = 5
CFG3_VU_BACK      = 0x3E0       # [9:5] VU back lanes (information for cycle models)
CFG3_VU_BACK_LSB  = 5
CFG3_VU_BACK_W    = 5
CFG3_PCAP16       = 0xFFFF0000  # [31:16] PCAP / 16 (U200: 584)
CFG3_PCAP16_LSB   = 16
CFG3_PCAP16_W     = 16
# CTL_ID (EX §5.1): control-core version and features
CTL_ID_VERSION     = 0xFF    # [7:0] control-core version (2)
CTL_ID_VERSION_LSB = 0
CTL_ID_VERSION_W   = 8
CTL_ID_DM          = 0x100   # [8] debug module
CTL_ID_DM_LSB      = 8
CTL_ID_DM_W        = 1
CTL_ID_REC         = 0x200   # [9] completion records
CTL_ID_REC_LSB     = 9
CTL_ID_REC_W       = 1
CTL_ID_BRIDGE      = 0x400   # [10] SPM bridge
CTL_ID_BRIDGE_LSB  = 10
CTL_ID_BRIDGE_W    = 1
CTL_ID_LOG         = 0x800   # [11] log ring
CTL_ID_LOG_LSB     = 11
CTL_ID_LOG_W       = 1
CTL_ID_WDOG        = 0x1000  # [12] watchdog
CTL_ID_WDOG_LSB    = 12
CTL_ID_WDOG_W      = 1
CTL_ID_SBA         = 0x2000  # [13] SBA connected
CTL_ID_SBA_LSB     = 13
CTL_ID_SBA_W       = 1
CTL_ID_SPM_LB      = 0x4000  # [14] SPM line-buffered alias
CTL_ID_SPM_LB_LSB  = 14
CTL_ID_SPM_LB_W    = 1
# CTRL (EX §5.2): pulse bits read 0; level bits act before the pulses of the same write
CTRL_RV_RUN           = 0x1    # [0] RW level: 1 releases the RISC-V; 1 -> 0 starts the RISC-V reset sequence; refused while SR_BUSY
CTRL_RV_RUN_LSB       = 0
CTRL_RV_RUN_W         = 1
CTRL_HOST_HOLD        = 0x2    # [1] RW level: the host's hold source for the queue heads
CTRL_HOST_HOLD_LSB    = 1
CTRL_HOST_HOLD_W      = 1
CTRL_HOLD_ON_HALT     = 0x4    # [2] RW level: a debug halt holds the queue heads
CTRL_HOLD_ON_HALT_LSB = 2
CTRL_HOLD_ON_HALT_W   = 1
CTRL_IRQ_EN           = 0x8    # [3] RW level: host_irq_o = IRQ_EN and LCQ_WR != LCQ_RD
CTRL_IRQ_EN_LSB       = 3
CTRL_IRQ_EN_W         = 1
CTRL_QABORT           = 0x100  # [8] W1 pulse: host QABORT, at once
CTRL_QABORT_LSB       = 8
CTRL_QABORT_W         = 1
CTRL_QCLEAR           = 0x200  # [9] W1 pulse: host QCLEAR; only with QIDLE and the RISC-V quiet, else QCLEAR_REFUSED
CTRL_QCLEAR_LSB       = 9
CTRL_QCLEAR_W         = 1
CTRL_SOFT_RESET       = 0x400  # [10] W1 pulse: device-level reset sequence (EX §8.5)
CTRL_SOFT_RESET_LSB   = 10
CTRL_SOFT_RESET_W     = 1
# STATUS (EX §5.3): bits 0..15 levels, 16..27 sticky (W1C)
STATUS_RV_RUNNING         = 0x1        # [0] RV_RUN = 1 and the hart is out of reset
STATUS_RV_RUNNING_LSB     = 0
STATUS_RV_RUNNING_W       = 1
STATUS_RV_HALTED          = 0x2        # [1] debug halt
STATUS_RV_HALTED_LSB      = 1
STATUS_RV_HALTED_W        = 1
STATUS_RV_SLEEP           = 0x4        # [2] WFI
STATUS_RV_SLEEP_LSB       = 2
STATUS_RV_SLEEP_W         = 1
STATUS_RV_CLK_OK          = 0x8        # [3] clk_rv runs and its domain is out of reset
STATUS_RV_CLK_OK_LSB      = 3
STATUS_RV_CLK_OK_W        = 1
STATUS_RV_RESET_DONE      = 0x10       # [4] the RISC-V reset sequence has completed
STATUS_RV_RESET_DONE_LSB  = 4
STATUS_RV_RESET_DONE_W    = 1
STATUS_ACTIVE             = 0x20       # [5] TAKEN != ANSWERED
STATUS_ACTIVE_LSB         = 5
STATUS_ACTIVE_W           = 1
STATUS_QIDLE              = 0x40       # [6] no entry queued or running, issue FIFO empty
STATUS_QIDLE_LSB          = 6
STATUS_QIDLE_W            = 1
STATUS_QQUIESCENT         = 0x80       # [7] quiescent (CC §9.6)
STATUS_QQUIESCENT_LSB     = 7
STATUS_QQUIESCENT_W       = 1
STATUS_QERR               = 0x100      # [8] some unit is in error state
STATUS_QERR_LSB           = 8
STATUS_QERR_W             = 1
STATUS_SR_BUSY            = 0x200      # [9] SOFT_RESET in progress
STATUS_SR_BUSY_LSB        = 9
STATUS_SR_BUSY_W          = 1
STATUS_LCQ_PENDING        = 0x400      # [10] LCQ_WR != LCQ_RD
STATUS_LCQ_PENDING_LSB    = 10
STATUS_LCQ_PENDING_W      = 1
STATUS_LOG_PENDING        = 0x800      # [11] LOG_WR != LOG_RD
STATUS_LOG_PENDING_LSB    = 11
STATUS_LOG_PENDING_W      = 1
STATUS_PLAT_ERR           = 0x1000     # [12] the platform's error input plat_err_i
STATUS_PLAT_ERR_LSB       = 12
STATUS_PLAT_ERR_W         = 1
STATUS_HOLD_SRC           = 0xE000     # [15:13] hold sources (layout HOLD_SRC: host, RISC-V, debug halt)
STATUS_HOLD_SRC_LSB       = 13
STATUS_HOLD_SRC_W         = 3
STATUS_LOAD_REFUSED       = 0x10000    # [16] sticky: LOAD_DATA write or VERIFY while RV_RUN = 1 or SR_BUSY
STATUS_LOAD_REFUSED_LSB   = 16
STATUS_LOAD_REFUSED_W     = 1
STATUS_LOAD_TIMEOUT       = 0x20000    # [17] sticky: a load or VERIFY word timed out
STATUS_LOAD_TIMEOUT_LSB   = 17
STATUS_LOAD_TIMEOUT_W     = 1
STATUS_DMI_TIMEOUT        = 0x40000    # [18] sticky
STATUS_DMI_TIMEOUT_LSB    = 18
STATUS_DMI_TIMEOUT_W      = 1
STATUS_QCLEAR_REFUSED     = 0x80000    # [19] sticky
STATUS_QCLEAR_REFUSED_LSB = 19
STATUS_QCLEAR_REFUSED_W   = 1
STATUS_PUSH_TIMEOUT       = 0x100000   # [20] sticky: the RISC-V's issue safety net fired
STATUS_PUSH_TIMEOUT_LSB   = 20
STATUS_PUSH_TIMEOUT_W     = 1
STATUS_BRIDGE_TIMEOUT     = 0x200000   # [21] sticky
STATUS_BRIDGE_TIMEOUT_LSB = 21
STATUS_BRIDGE_TIMEOUT_W   = 1
STATUS_LQ_OVERFLOW        = 0x400000   # [22] sticky: a push while the LQ was full (descriptor dropped)
STATUS_LQ_OVERFLOW_LSB    = 22
STATUS_LQ_OVERFLOW_W      = 1
STATUS_RING_RD_BAD        = 0x800000   # [23] sticky: an LCQ_RD or LOG_RD write was refused
STATUS_RING_RD_BAD_LSB    = 23
STATUS_RING_RD_BAD_W      = 1
STATUS_LOG_DROP           = 0x1000000  # [24] sticky: log words were dropped
STATUS_LOG_DROP_LSB       = 24
STATUS_LOG_DROP_W         = 1
STATUS_UNIT_HUNG          = 0x2000000  # [25] sticky: a unit watchdog fired
STATUS_UNIT_HUNG_LSB      = 25
STATUS_UNIT_HUNG_W        = 1
STATUS_SR_RV_TIMEOUT      = 0x4000000  # [26] sticky: SOFT_RESET went on without the RISC-V reset completing
STATUS_SR_RV_TIMEOUT_LSB  = 26
STATUS_SR_RV_TIMEOUT_W    = 1
STATUS_RV_RUN_REFUSED     = 0x8000000  # [27] sticky
STATUS_RV_RUN_REFUSED_LSB = 27
STATUS_RV_RUN_REFUSED_W   = 1
# HOLD_SRC (EX §5.3, CC §5.1, DB §5.5): bit order of a hold-source field (STATUS[15:13], QSTATUS[4:2], DBG_U_STATE[11:8])
HOLD_SRC_HOST     = 0x1  # [0] host (CTRL.HOST_HOLD)
HOLD_SRC_HOST_LSB = 0
HOLD_SRC_HOST_W   = 1
HOLD_SRC_RV       = 0x2  # [1] RISC-V (QCTRL QHOLD)
HOLD_SRC_RV_LSB   = 1
HOLD_SRC_RV_W     = 1
HOLD_SRC_HALT     = 0x4  # [2] debug halt with HOLD_ON_HALT
HOLD_SRC_HALT_LSB = 2
HOLD_SRC_HALT_W   = 1
HOLD_SRC_DB       = 0x8  # [3] DB's dbg_hold (DBG_U_STATE only)
HOLD_SRC_DB_LSB   = 3
HOLD_SRC_DB_W     = 1
# RV_BOOT (EX §5.1): boot vector, 256-byte aligned
RV_BOOT_ADDR     = 0xFFFFFF00  # [31:8] boot vector bits [31:8]; [7:0] read 0
RV_BOOT_ADDR_LSB = 8
RV_BOOT_ADDR_W   = 24
# RV_CFG (EX §5.1): TCM sizes
RV_CFG_ITCM_KIB     = 0xFFFF      # [15:0] ITCM KiB
RV_CFG_ITCM_KIB_LSB = 0
RV_CFG_ITCM_KIB_W   = 16
RV_CFG_DTCM_KIB     = 0xFFFF0000  # [31:16] DTCM KiB
RV_CFG_DTCM_KIB_LSB = 16
RV_CFG_DTCM_KIB_W   = 16
# WDOG_CFG (EX §5.1, §8.6): unit progress watchdog
WDOG_CFG_L     = 0x1F  # [4:0] limit 2^L core cycles without progress; 0 = off
WDOG_CFG_L_LSB = 0
WDOG_CFG_L_W   = 5
# RT_STATE (EX §4.5): runtime phase (RISC-V 0x290, mirrored to host 0x038)
RT_STATE_PHASE          = 0xFF        # [7:0] runtime phase
RT_STATE_PHASE_LSB      = 0
RT_STATE_PHASE_W        = 8
RT_STATE_KERNEL         = 0xFFFF0000  # [31:16] kernel index
RT_STATE_KERNEL_LSB     = 16
RT_STATE_KERNEL_W       = 16
RT_STATE_PHASE_RESET    = 0
RT_STATE_PHASE_BOOT     = 1
RT_STATE_PHASE_IDLE     = 2
RT_STATE_PHASE_DISPATCH = 3
RT_STATE_PHASE_KERNEL   = 4
RT_STATE_PHASE_JOIN     = 5
RT_STATE_PHASE_ANSWER   = 6
RT_STATE_PHASE_TRAPPED  = 7
# LQ_CFG (EX §5.1): ring parameters and watermark valid bits
LQ_CFG_LQ_DEPTH         = 0xFF        # [7:0]
LQ_CFG_LQ_DEPTH_LSB     = 0
LQ_CFG_LQ_DEPTH_W       = 8
LQ_CFG_LCQ_DEPTH        = 0xFF00      # [15:8]
LQ_CFG_LCQ_DEPTH_LSB    = 8
LQ_CFG_LCQ_DEPTH_W      = 8
LQ_CFG_LOG_DEPTH        = 0xFFF0000   # [27:16]
LQ_CFG_LOG_DEPTH_LSB    = 16
LQ_CFG_LOG_DEPTH_W      = 12
LQ_CFG_STOP_VALID       = 0x40000000  # [30]
LQ_CFG_STOP_VALID_LSB   = 30
LQ_CFG_STOP_VALID_W     = 1
LQ_CFG_CANCEL_VALID     = 0x80000000  # [31]
LQ_CFG_CANCEL_VALID_LSB = 31
LQ_CFG_CANCEL_VALID_W   = 1
# PERF_INFO (EX §5.1): PERF layout
PERF_INFO_COUNT     = 0xFF    # [7:0] counters implemented
PERF_INFO_COUNT_LSB = 0
PERF_INFO_COUNT_W   = 8
PERF_INFO_X         = 0xFF00  # [15:8] first extension index X = 5 + 2 NUM_CH
PERF_INFO_X_LSB     = 8
PERF_INFO_X_W       = 8
# Q_DEPTH (EX §5.1, CC §7.4): queue depth of unit u at [5u+4:5u]; 0 = unit absent
Q_DEPTH_DMA      = 0x1F        # [4:0] U200 M6: 8
Q_DEPTH_DMA_LSB  = 0
Q_DEPTH_DMA_W    = 5
Q_DEPTH_QACT     = 0x3E0       # [9:5] 4
Q_DEPTH_QACT_LSB = 5
Q_DEPTH_QACT_W   = 5
Q_DEPTH_MVM      = 0x7C00      # [14:10] 4
Q_DEPTH_MVM_LSB  = 10
Q_DEPTH_MVM_W    = 5
Q_DEPTH_VU       = 0xF8000     # [19:15] 16
Q_DEPTH_VU_LSB   = 15
Q_DEPTH_VU_W     = 5
Q_DEPTH_ATT      = 0x1F00000   # [24:20] 0: absent in M6
Q_DEPTH_ATT_LSB  = 20
Q_DEPTH_ATT_W    = 5
Q_DEPTH_SYNC     = 0x3E000000  # [29:25] 4
Q_DEPTH_SYNC_LSB = 25
Q_DEPTH_SYNC_W   = 5
# LOG_CFG (EX §4.10): host RW at 0x098, RISC-V read-only copy at 0x288
LOG_CFG_ENABLE     = 0x1   # [0] 0: the SDK drops every record
LOG_CFG_ENABLE_LSB = 0
LOG_CFG_ENABLE_W   = 1
LOG_CFG_BLOCK      = 0x2   # [1] the SDK waits for space instead of dropping
LOG_CFG_BLOCK_LSB  = 1
LOG_CFG_BLOCK_W    = 1
LOG_CFG_LEVEL      = 0xF0  # [7:4] the SDK drops records above this level (the only runtime filter)
LOG_CFG_LEVEL_LSB  = 4
LOG_CFG_LEVEL_W    = 4
# LOAD_CTRL (EX §5.1, §3.3)
LOAD_CTRL_VERIFY     = 0x1  # [0] read LOAD_COUNT words back and CRC them
LOAD_CTRL_VERIFY_LSB = 0
LOAD_CTRL_VERIFY_W   = 1
# LOAD_STATUS (EX §5.1, §3.3)
LOAD_STATUS_VERIFY_BUSY     = 0x1  # [0]
LOAD_STATUS_VERIFY_BUSY_LSB = 0
LOAD_STATUS_VERIFY_BUSY_W   = 1
# Q_STATE (EX §5.4): queue state of unit u
Q_STATE_ENTRIES      = 0x1F   # [4:0] entries in the queue
Q_STATE_ENTRIES_LSB  = 0
Q_STATE_ENTRIES_W    = 5
Q_STATE_RUNNING      = 0x20   # [5] the unit holds an op
Q_STATE_RUNNING_LSB  = 5
Q_STATE_RUNNING_W    = 1
Q_STATE_DEP_WAIT     = 0x40   # [6] the head waits for deps
Q_STATE_DEP_WAIT_LSB = 6
Q_STATE_DEP_WAIT_W   = 1
Q_STATE_HELD         = 0x80   # [7]
Q_STATE_HELD_LSB     = 7
Q_STATE_HELD_W       = 1
Q_STATE_ERR          = 0x100  # [8] error state
Q_STATE_ERR_LSB      = 8
Q_STATE_ERR_W        = 1
Q_STATE_HUNG         = 0x200  # [9] watchdog
Q_STATE_HUNG_LSB     = 9
Q_STATE_HUNG_W       = 1
# Q_FIFO (EX §5.4)
Q_FIFO_ENTRIES     = 0xFF  # [7:0] issue-FIFO entries waiting for the dispatcher
Q_FIFO_ENTRIES_LSB = 0
Q_FIFO_ENTRIES_W   = 8
# ADAPTER_CTRL (EX §9.2)
ADAPTER_CTRL_MEM_EN     = 0x1  # [0] rising edge: latch and alignment-check all bases; requests are held while 0
ADAPTER_CTRL_MEM_EN_LSB = 0
ADAPTER_CTRL_MEM_EN_W   = 1
# ADAPTER_STATUS (EX §9.2, docs/platform-u200.md §2): sticky, W1C
ADAPTER_STATUS_RRESP_ERR           = 0x1         # [0] RRESP error
ADAPTER_STATUS_RRESP_ERR_LSB       = 0
ADAPTER_STATUS_RRESP_ERR_W         = 1
ADAPTER_STATUS_BRESP_ERR           = 0x2         # [1] BRESP error
ADAPTER_STATUS_BRESP_ERR_LSB       = 1
ADAPTER_STATUS_BRESP_ERR_W         = 1
ADAPTER_STATUS_ORPHAN_RESP         = 0x4         # [2] response with no outstanding request for its id
ADAPTER_STATUS_ORPHAN_RESP_LSB     = 2
ADAPTER_STATUS_ORPHAN_RESP_W       = 1
ADAPTER_STATUS_BASE_MISALIGNED     = 0x8         # [3] a base was not 4 KiB aligned at the MEM_EN edge
ADAPTER_STATUS_BASE_MISALIGNED_LSB = 3
ADAPTER_STATUS_BASE_MISALIGNED_W   = 1
ADAPTER_STATUS_REQ_HELD            = 0x10        # [4] a core memory request was held while MEM_EN = 0
ADAPTER_STATUS_REQ_HELD_LSB        = 4
ADAPTER_STATUS_REQ_HELD_W          = 1
ADAPTER_STATUS_BASE_LOCKED         = 0x20        # [5] a base write while MEM_EN = 1 was ignored
ADAPTER_STATUS_BASE_LOCKED_LSB     = 5
ADAPTER_STATUS_BASE_LOCKED_W       = 1
ADAPTER_STATUS_ERR_COUNT           = 0xFFFF0000  # [31:16] saturating count of error responses
ADAPTER_STATUS_ERR_COUNT_LSB       = 16
ADAPTER_STATUS_ERR_COUNT_W         = 16
# ADAPTER_VERSION (EX §9.2): {16'h0002, 8'h00, NUM_CH}
ADAPTER_VERSION_NUM_CH     = 0xFF        # [7:0]
ADAPTER_VERSION_NUM_CH_LSB = 0
ADAPTER_VERSION_NUM_CH_W   = 8
ADAPTER_VERSION_MAJOR      = 0xFFFF0000  # [31:16] 2
ADAPTER_VERSION_MAJOR_LSB  = 16
ADAPTER_VERSION_MAJOR_W    = 16
# HANDLE (EX §4.3): DRAM handle (descriptor args, OPL, DBG_TRW_BASE)
HANDLE_BEAT     = 0xFFFFFFF   # [27:0] beat in the channel
HANDLE_BEAT_LSB = 0
HANDLE_BEAT_W   = 28
HANDLE_CH       = 0xF0000000  # [31:28] channel
HANDLE_CH_LSB   = 28
HANDLE_CH_W     = 4
# TICKET (CC §7.5): ticket {unit, seq}; unit 15 = none; (u, s) done iff (D[u] - s) mod 2^28 in [1, 2^27)
TICKET_SEQ      = 0xFFFFFFF   # [27:0] per-unit sequence
TICKET_SEQ_LSB  = 0
TICKET_SEQ_W    = 28
TICKET_UNIT     = 0xF0000000  # [31:28] unit id
TICKET_UNIT_LSB = 28
TICKET_UNIT_W   = 4
# DESC (EX §4.3): descriptor word w0
DESC_TYPE        = 0xFF        # [7:0] descriptor type; other values reserved (BAD_DESC)
DESC_TYPE_LSB    = 0
DESC_TYPE_W      = 8
DESC_PROF        = 0x100       # [8] profiled launch: REC_CFG.ALL and NODE markers for the launch
DESC_PROF_LSB    = 8
DESC_PROF_W      = 1
DESC_KERNEL      = 0xFFFF0000  # [31:16] index into .tsr_kernels (LAUNCH); 0 for PING
DESC_KERNEL_LSB  = 16
DESC_KERNEL_W    = 16
DESC_TYPE_LAUNCH = 1
DESC_TYPE_PING   = 2
# LCQ_R0 (EX §4.4): record word r0, one layout for every record type
LCQ_R0_TYPE              = 0xFF        # [7:0] record type
LCQ_R0_TYPE_LSB          = 0
LCQ_R0_TYPE_W            = 8
LCQ_R0_STATUS            = 0xFF00      # [15:8] launch STATUS (DONE), 4 for TRAP; kernel-defined for kernel events
LCQ_R0_STATUS_LSB        = 8
LCQ_R0_STATUS_W          = 8
LCQ_R0_ERR_CODE          = 0xFF0000    # [23:16] code of the first error, 0 if none
LCQ_R0_ERR_CODE_LSB      = 16
LCQ_R0_ERR_CODE_W        = 8
LCQ_R0_SEQ               = 0xFF000000  # [31:24] SEQ[7:0] of the descriptor
LCQ_R0_SEQ_LSB           = 24
LCQ_R0_SEQ_W             = 8
LCQ_R0_TYPE_DONE         = 1           # answer
LCQ_R0_TYPE_PONG         = 2           # answer
LCQ_R0_TYPE_HELLO        = 0x80        # event
LCQ_R0_TYPE_TRAP         = 0x81        # answer if ACTIVE, else event
LCQ_R0_TYPE_EVENT_FIRST  = 0xC0        # kernel events 0xC0..0xFF (ANSWER = 0)
LCQ_R0_TYPE_EVENT_LAST   = 0xFF
LCQ_R0_STATUS_OK         = 0           # RET >= 0 and no op of the launch failed
LCQ_R0_STATUS_TILE_ERR   = 1           # the first error has a code other than ABORTED; r3..r6 describe it
LCQ_R0_STATUS_BAD_DESC   = 2           # the descriptor was refused; the kernel did not run
LCQ_R0_STATUS_KERNEL_ERR = 3           # RET < 0, no tile error
LCQ_R0_STATUS_TRAP       = 4           # TRAP records only
LCQ_R0_STATUS_ABORTED    = 5           # cancelled (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8)
LCQ_R0_STATUS_PLAT_ERR   = 6           # STATUS.PLAT_ERR was set when the runtime answered
LCQ_R0_STATUS_STOPPED    = 7           # the kernel honoured a soft stop
# HELLO_R4 (EX §4.4): HELLO record word r4
HELLO_R4_SDK_ABI        = 0xFFFF      # [15:0] SDK ABI version
HELLO_R4_SDK_ABI_LSB    = 0
HELLO_R4_SDK_ABI_W      = 16
HELLO_R4_RT_VERSION     = 0xFFFF0000  # [31:16] runtime version
HELLO_R4_RT_VERSION_LSB = 16
HELLO_R4_RT_VERSION_W   = 16
# ISSUE_PUSH (CC §5.1): ISSUE_PUSH write data
ISSUE_PUSH_REC       = 0x1     # [0] completion record wanted (DB §8.2)
ISSUE_PUSH_REC_LSB   = 0
ISSUE_PUSH_REC_W     = 1
ISSUE_PUSH_WMASK     = 0xFF00  # [15:8] word i with bit i clear is sent as 0; w0 is always sent
ISSUE_PUSH_WMASK_LSB = 8
ISSUE_PUSH_WMASK_W   = 8
# ISSUE_CREDIT (CC §5.1): min(free queue slots, free issue-FIFO entries, 15) per unit
ISSUE_CREDIT_DMA      = 0xF       # [3:0]
ISSUE_CREDIT_DMA_LSB  = 0
ISSUE_CREDIT_DMA_W    = 4
ISSUE_CREDIT_QACT     = 0xF0      # [7:4]
ISSUE_CREDIT_QACT_LSB = 4
ISSUE_CREDIT_QACT_W   = 4
ISSUE_CREDIT_MVM      = 0xF00     # [11:8]
ISSUE_CREDIT_MVM_LSB  = 8
ISSUE_CREDIT_MVM_W    = 4
ISSUE_CREDIT_VU       = 0xF000    # [15:12]
ISSUE_CREDIT_VU_LSB   = 12
ISSUE_CREDIT_VU_W     = 4
ISSUE_CREDIT_ATT      = 0xF0000   # [19:16]
ISSUE_CREDIT_ATT_LSB  = 16
ISSUE_CREDIT_ATT_W    = 4
ISSUE_CREDIT_SYNC     = 0xF00000  # [23:20]
ISSUE_CREDIT_SYNC_LSB = 20
ISSUE_CREDIT_SYNC_W   = 4
# ISSUE_UNITS (CC §5.1): bit u = unit u is built (constant)
ISSUE_UNITS_DMA      = 0x1   # [0]
ISSUE_UNITS_DMA_LSB  = 0
ISSUE_UNITS_DMA_W    = 1
ISSUE_UNITS_QACT     = 0x2   # [1]
ISSUE_UNITS_QACT_LSB = 1
ISSUE_UNITS_QACT_W   = 1
ISSUE_UNITS_MVM      = 0x4   # [2]
ISSUE_UNITS_MVM_LSB  = 2
ISSUE_UNITS_MVM_W    = 1
ISSUE_UNITS_VU       = 0x8   # [3]
ISSUE_UNITS_VU_LSB   = 3
ISSUE_UNITS_VU_W     = 1
ISSUE_UNITS_ATT      = 0x10  # [4] 0 in M6
ISSUE_UNITS_ATT_LSB  = 4
ISSUE_UNITS_ATT_W    = 1
ISSUE_UNITS_SYNC     = 0x20  # [5]
ISSUE_UNITS_SYNC_LSB = 5
ISSUE_UNITS_SYNC_W   = 1
# ISSUE_STATUS (CC §5.1): R/W1C
ISSUE_STATUS_PUSH_TIMEOUT     = 0x1  # [0] a push waited ISSUE_WAIT_MAX and faulted
ISSUE_STATUS_PUSH_TIMEOUT_LSB = 0
ISSUE_STATUS_PUSH_TIMEOUT_W   = 1
ISSUE_STATUS_ERR_IRQ          = 0x2  # [1] sticky, set on the rising edge of 'some unit in error state' (irq_fast[1])
ISSUE_STATUS_ERR_IRQ_LSB      = 1
ISSUE_STATUS_ERR_IRQ_W        = 1
# QCTRL (CC §5.1): one bit per write, in push order through the issue FIFO
QCTRL_QABORT        = 0x1  # [0]
QCTRL_QABORT_LSB    = 0
QCTRL_QABORT_W      = 1
QCTRL_QHOLD_SET     = 0x2  # [1] set the RISC-V hold source
QCTRL_QHOLD_SET_LSB = 1
QCTRL_QHOLD_SET_W   = 1
QCTRL_QHOLD_CLR     = 0x4  # [2] clear the RISC-V hold source
QCTRL_QHOLD_CLR_LSB = 2
QCTRL_QHOLD_CLR_W   = 1
QCTRL_QCLEAR        = 0x8  # [3] gets a SYNC ticket (ISSUE_TICKET)
QCTRL_QCLEAR_LSB    = 3
QCTRL_QCLEAR_W      = 1
# QSTATUS (CC §5.1): queue status (RISC-V view)
QSTATUS_QIDLE         = 0x1       # [0] ISSUED_SEQ = DONE_SEQ for every u and the issue FIFO empty
QSTATUS_QIDLE_LSB     = 0
QSTATUS_QIDLE_W       = 1
QSTATUS_HOLD          = 0x2       # [1] a hold is in effect
QSTATUS_HOLD_LSB      = 1
QSTATUS_HOLD_W        = 1
QSTATUS_HOLD_HOST     = 0x4       # [2] host hold
QSTATUS_HOLD_HOST_LSB = 2
QSTATUS_HOLD_HOST_W   = 1
QSTATUS_HOLD_RV       = 0x8       # [3] RISC-V hold
QSTATUS_HOLD_RV_LSB   = 3
QSTATUS_HOLD_RV_W     = 1
QSTATUS_HOLD_HALT     = 0x10      # [4] debug-halt hold
QSTATUS_HOLD_HALT_LSB = 4
QSTATUS_HOLD_HALT_W   = 1
QSTATUS_ERR           = 0x3F00    # [13:8] error state of unit u at bit 8 + u
QSTATUS_ERR_LSB       = 8
QSTATUS_ERR_W         = 6
QSTATUS_NONEMPTY      = 0x3F0000  # [21:16] queue u non-empty at bit 16 + u
QSTATUS_NONEMPTY_LSB  = 16
QSTATUS_NONEMPTY_W    = 6
# WAIT_CREDIT (CC §5.1): irq_fast[5] while the unit's credit != 0
WAIT_CREDIT_UNIT     = 0x7         # [2:0]
WAIT_CREDIT_UNIT_LSB = 0
WAIT_CREDIT_UNIT_W   = 3
WAIT_CREDIT_EN       = 0x80000000  # [31]
WAIT_CREDIT_EN_LSB   = 31
WAIT_CREDIT_EN_W     = 1
# LQ_FLAGS (EX §4.5): RISC-V view of the launch state
LQ_FLAGS_CANCEL_VALID     = 0x1  # [0]
LQ_FLAGS_CANCEL_VALID_LSB = 0
LQ_FLAGS_CANCEL_VALID_W   = 1
LQ_FLAGS_STOP_VALID       = 0x2  # [1]
LQ_FLAGS_STOP_VALID_LSB   = 1
LQ_FLAGS_STOP_VALID_W     = 1
LQ_FLAGS_ACTIVE           = 0x4  # [2]
LQ_FLAGS_ACTIVE_LSB       = 2
LQ_FLAGS_ACTIVE_W         = 1
LQ_FLAGS_PLAT_ERR         = 0x8  # [3] STATUS.PLAT_ERR
LQ_FLAGS_PLAT_ERR_LSB     = 3
LQ_FLAGS_PLAT_ERR_W       = 1
# LCQ_POST (EX §4.5): LCQ_POST write data
LCQ_POST_ANSWER     = 0x1  # [0] the record answers the running launch (ANSWERED += 1)
LCQ_POST_ANSWER_LSB = 0
LCQ_POST_ANSWER_W   = 1
# REC_CFG (DB §10.2, §8.2): completion-record configuration
REC_CFG_ALL          = 0x1  # [0] a record for every entry, not only ISSUE_PUSH.REC ones
REC_CFG_ALL_LSB      = 0
REC_CFG_ALL_W        = 1
REC_CFG_IRQ_HALF     = 0x2  # [1] irq_fast[4] while REC_COUNT >= 32
REC_CFG_IRQ_HALF_LSB = 1
REC_CFG_IRQ_HALF_W   = 1
# REC_DATA3 (DB §8.2): completion record word 3
REC_DATA3_CODE       = 0xF         # [3:0] completion code
REC_DATA3_CODE_LSB   = 0
REC_DATA3_CODE_W     = 4
REC_DATA3_RAN        = 0x10        # [4]
REC_DATA3_RAN_LSB    = 4
REC_DATA3_RAN_W      = 1
REC_DATA3_MARKER     = 0x20        # [5]
REC_DATA3_MARKER_LSB = 5
REC_DATA3_MARKER_W   = 1
REC_DATA3_OPCODE     = 0xFF00      # [15:8]
REC_DATA3_OPCODE_LSB = 8
REC_DATA3_OPCODE_W   = 8
REC_DATA3_PIDX       = 0xFFFF0000  # [31:16] push index [15:0]
REC_DATA3_PIDX_LSB   = 16
REC_DATA3_PIDX_W     = 16
# DBG_FLAGS (DB §10.2): RISC-V view of DB's state (control window 0x400)
DBG_FLAGS_DBG_STOP         = 0x1   # [0] a DB stop or the debug-halt hold is in effect
DBG_FLAGS_DBG_STOP_LSB     = 0
DBG_FLAGS_DBG_STOP_W       = 1
DBG_FLAGS_SERIAL           = 0x2   # [1]
DBG_FLAGS_SERIAL_LSB       = 1
DBG_FLAGS_SERIAL_W         = 1
DBG_FLAGS_TRW_EN           = 0x4   # [2]
DBG_FLAGS_TRW_EN_LSB       = 2
DBG_FLAGS_TRW_EN_W         = 1
DBG_FLAGS_DMI_EN           = 0x8   # [3] a debugger may be attached
DBG_FLAGS_DMI_EN_LSB       = 3
DBG_FLAGS_DMI_EN_W         = 1
DBG_FLAGS_HALT_ON_BOOT     = 0x10  # [4]
DBG_FLAGS_HALT_ON_BOOT_LSB = 4
DBG_FLAGS_HALT_ON_BOOT_W   = 1
# MARK (DB §8.4): TRACE_MARK value
MARK_PAYLOAD            = 0xFFFFFFF   # [27:0]
MARK_PAYLOAD_LSB        = 0
MARK_PAYLOAD_W          = 28
MARK_CLASS              = 0xF0000000  # [31:28]
MARK_CLASS_LSB          = 28
MARK_CLASS_W            = 4
MARK_CLASS_USER         = 0           # any payload
MARK_CLASS_LAUNCH_BEGIN = 1           # payload: layout MARK_LAUNCH
MARK_CLASS_LAUNCH_END   = 2           # payload: layout MARK_LAUNCH
MARK_CLASS_NODE_BEGIN   = 3           # payload: op-list index
MARK_CLASS_NODE_END     = 4           # payload: op-list index
MARK_CLASS_LAYER        = 5           # payload: layer index
MARK_CLASS_PHASE        = 6           # payload: kernel-defined id (.tsr_marks)
# MARK_LAUNCH (DB §8.4): payload of LAUNCH_BEGIN / LAUNCH_END markers
MARK_LAUNCH_SEQ        = 0xFFFF      # [15:0] launch SEQ low bits
MARK_LAUNCH_SEQ_LSB    = 0
MARK_LAUNCH_SEQ_W      = 16
MARK_LAUNCH_KERNEL     = 0xFFF0000   # [27:16] kernel index
MARK_LAUNCH_KERNEL_LSB = 16
MARK_LAUNCH_KERNEL_W   = 12
MARK_LAUNCH_CLASS      = 0xF0000000  # [31:28]
MARK_LAUNCH_CLASS_LSB  = 28
MARK_LAUNCH_CLASS_W    = 4
# LOG_HDR (DB §6.1): log record header (the only definition; EX §4.10, SW §6.6)
LOG_HDR_N              = 0xFF        # [7:0] payload words after the header
LOG_HDR_N_LSB          = 0
LOG_HDR_N_W            = 8
LOG_HDR_LEVEL          = 0xF00       # [11:8]
LOG_HDR_LEVEL_LSB      = 8
LOG_HDR_LEVEL_W        = 4
LOG_HDR_SOURCE         = 0xF000      # [15:12]
LOG_HDR_SOURCE_LSB     = 12
LOG_HDR_SOURCE_W       = 4
LOG_HDR_TYPE           = 0x70000     # [18:16] 5..7 invalid
LOG_HDR_TYPE_LSB       = 16
LOG_HDR_TYPE_W         = 3
LOG_HDR_TS             = 0x80000     # [19] the first payload word is MTIME[31:0]
LOG_HDR_TS_LSB         = 19
LOG_HDR_TS_W           = 1
LOG_HDR_LSEQ           = 0xFF00000   # [27:20] SEQ[7:0] of the running launch; 0 outside launches
LOG_HDR_LSEQ_LSB       = 20
LOG_HDR_LSEQ_W         = 8
LOG_HDR_SYNC           = 0xF0000000  # [31:28] 0xA
LOG_HDR_SYNC_LSB       = 28
LOG_HDR_SYNC_W         = 4
LOG_HDR_LEVEL_FATAL    = 0
LOG_HDR_LEVEL_ERROR    = 1
LOG_HDR_LEVEL_WARN     = 2
LOG_HDR_LEVEL_INFO     = 3
LOG_HDR_LEVEL_DEBUG    = 4
LOG_HDR_LEVEL_TRACE    = 5
LOG_HDR_SOURCE_RUNTIME = 0
LOG_HDR_SOURCE_KERNEL  = 1
LOG_HDR_SOURCE_TRAP    = 2           # trap handler
LOG_HDR_SOURCE_DEBUG   = 3           # debug kernels
LOG_HDR_TYPE_FMT       = 0           # format-string address in .tsr_logfmt, then arguments
LOG_HDR_TYPE_TEXT      = 1           # UTF-8, zero-padded
LOG_HDR_TYPE_TRAP      = 2           # the 44 words of __tsr_crash
LOG_HDR_TYPE_EVENT     = 3           # event code (marker encoding), arguments
LOG_HDR_TYPE_BLOB      = 4           # tag, raw words
LOG_HDR_SYNC_MAGIC     = 10
# TRW_W0 (DB §8.3): trace-writer record word 0
TRW_W0_KIND        = 0xF         # [3:0]
TRW_W0_KIND_LSB    = 0
TRW_W0_KIND_W      = 4
TRW_W0_UNIT        = 0xF0        # [7:4]
TRW_W0_UNIT_LSB    = 4
TRW_W0_UNIT_W      = 4
TRW_W0_CODE        = 0xF00       # [11:8]
TRW_W0_CODE_LSB    = 8
TRW_W0_CODE_W      = 4
TRW_W0_RAN         = 0x1000      # [12]
TRW_W0_RAN_LSB     = 12
TRW_W0_RAN_W       = 1
TRW_W0_REC         = 0x2000      # [13] REC flag
TRW_W0_REC_LSB     = 13
TRW_W0_REC_W       = 1
TRW_W0_TBP         = 0x4000      # [14] tile-breakpoint hit
TRW_W0_TBP_LSB     = 14
TRW_W0_TBP_W       = 1
TRW_W0_OPCODE      = 0xFF0000    # [23:16]
TRW_W0_OPCODE_LSB  = 16
TRW_W0_OPCODE_W    = 8
TRW_W0_W0_HI       = 0xFF000000  # [31:24] w0[15:8] of the op
TRW_W0_W0_HI_LSB   = 24
TRW_W0_W0_HI_W     = 8
TRW_W0_KIND_OP     = 0
TRW_W0_KIND_MARK   = 1
TRW_W0_KIND_STOP   = 2
TRW_W0_KIND_RVHALT = 3
TRW_W0_KIND_DROP   = 4
TRW_W0_KIND_PAD    = 15          # ignored
# DBG_ID (DB §10.1): 0 = block absent
DBG_ID_VERSION       = 0xFF      # [7:0] 2
DBG_ID_VERSION_LSB   = 0
DBG_ID_VERSION_W     = 8
DBG_ID_HIST          = 0x100     # [8] history
DBG_ID_HIST_LSB      = 8
DBG_ID_HIST_W        = 1
DBG_ID_TRW           = 0x200     # [9] trace writer
DBG_ID_TRW_LSB       = 9
DBG_ID_TRW_W         = 1
DBG_ID_RVWP          = 0x400     # [10] RV watchpoints
DBG_ID_RVWP_LSB      = 10
DBG_ID_RVWP_W        = 1
DBG_ID_REC           = 0x800     # [11] completion records
DBG_ID_REC_LSB       = 11
DBG_ID_REC_W         = 1
DBG_ID_NTBP          = 0x7000    # [14:12] tile breakpoints
DBG_ID_NTBP_LSB      = 12
DBG_ID_NTBP_W        = 3
DBG_ID_NRVWP         = 0x18000   # [16:15] RV watchpoints
DBG_ID_NRVWP_LSB     = 15
DBG_ID_NRVWP_W       = 2
DBG_ID_LOG2_HIST     = 0xF00000  # [23:20] log2 HIST_DEPTH
DBG_ID_LOG2_HIST_LSB = 20
DBG_ID_LOG2_HIST_W   = 4
# DBG_CTRL (DB §10.1): pulses; one of bits 0..2 per write (QSTEP > RELEASE > STOP_NOW)
DBG_CTRL_STOP_NOW           = 0x1    # [0]
DBG_CTRL_STOP_NOW_LSB       = 0
DBG_CTRL_STOP_NOW_W         = 1
DBG_CTRL_RELEASE            = 0x2    # [1]
DBG_CTRL_RELEASE_LSB        = 1
DBG_CTRL_RELEASE_W          = 1
DBG_CTRL_QSTEP              = 0x4    # [2]
DBG_CTRL_QSTEP_LSB          = 2
DBG_CTRL_QSTEP_W            = 1
DBG_CTRL_HIST_CLEAR         = 0x10   # [4]
DBG_CTRL_HIST_CLEAR_LSB     = 4
DBG_CTRL_HIST_CLEAR_W       = 1
DBG_CTRL_RV_SAMPLE          = 0x20   # [5] completes after the sample or 4,096 cycles
DBG_CTRL_RV_SAMPLE_LSB      = 5
DBG_CTRL_RV_SAMPLE_W        = 1
DBG_CTRL_RVWP_APPLY         = 0x40   # [6]
DBG_CTRL_RVWP_APPLY_LSB     = 6
DBG_CTRL_RVWP_APPLY_W       = 1
DBG_CTRL_TRW_FLUSH          = 0x80   # [7]
DBG_CTRL_TRW_FLUSH_LSB      = 7
DBG_CTRL_TRW_FLUSH_W        = 1
DBG_CTRL_RV_HALT_CANCEL     = 0x100  # [8]
DBG_CTRL_RV_HALT_CANCEL_LSB = 8
DBG_CTRL_RV_HALT_CANCEL_W   = 1
DBG_CTRL_MATCHQ_CLEAR       = 0x200  # [9]
DBG_CTRL_MATCHQ_CLEAR_LSB   = 9
DBG_CTRL_MATCHQ_CLEAR_W     = 1
# DBG_CFG (DB §10.1): reset 0x30
DBG_CFG_DMI_EN              = 0x1    # [0] gates the DMI window
DBG_CFG_DMI_EN_LSB          = 0
DBG_CFG_DMI_EN_W            = 1
DBG_CFG_SERIAL              = 0x2    # [1]
DBG_CFG_SERIAL_LSB          = 1
DBG_CFG_SERIAL_W            = 1
DBG_CFG_ERR_STOP            = 0x4    # [2]
DBG_CFG_ERR_STOP_LSB        = 2
DBG_CFG_ERR_STOP_W          = 1
DBG_CFG_STOP_HALT_RV        = 0x8    # [3]
DBG_CFG_STOP_HALT_RV_LSB    = 3
DBG_CFG_STOP_HALT_RV_W      = 1
DBG_CFG_HIST_EN             = 0x10   # [4]
DBG_CFG_HIST_EN_LSB         = 4
DBG_CFG_HIST_EN_W           = 1
DBG_CFG_HIST_FREEZE_ERR     = 0x20   # [5]
DBG_CFG_HIST_FREEZE_ERR_LSB = 5
DBG_CFG_HIST_FREEZE_ERR_W   = 1
DBG_CFG_HIST_FREEZE         = 0x40   # [6]
DBG_CFG_HIST_FREEZE_LSB     = 6
DBG_CFG_HIST_FREEZE_W       = 1
DBG_CFG_HUNG_STOP           = 0x80   # [7]
DBG_CFG_HUNG_STOP_LSB       = 7
DBG_CFG_HUNG_STOP_W         = 1
DBG_CFG_HALT_ON_BOOT        = 0x100  # [8]
DBG_CFG_HALT_ON_BOOT_LSB    = 8
DBG_CFG_HALT_ON_BOOT_W      = 1
DBG_CFG_RV_HALT_EN          = 0x200  # [9]
DBG_CFG_RV_HALT_EN_LSB      = 9
DBG_CFG_RV_HALT_EN_W        = 1
DBG_CFG_TRW_EN              = 0x400  # [10]
DBG_CFG_TRW_EN_LSB          = 10
DBG_CFG_TRW_EN_W            = 1
DBG_CFG_TRW_STOP_FULL       = 0x800  # [11]
DBG_CFG_TRW_STOP_FULL_LSB   = 11
DBG_CFG_TRW_STOP_FULL_W     = 1
# DBG_STATUS (DB §10.1): levels and sticky W1C bits
DBG_STATUS_STOP_ACTIVE           = 0x1         # [0]
DBG_STATUS_STOP_ACTIVE_LSB       = 0
DBG_STATUS_STOP_ACTIVE_W         = 1
DBG_STATUS_DRAINED               = 0x2         # [1]
DBG_STATUS_DRAINED_LSB           = 1
DBG_STATUS_DRAINED_W             = 1
DBG_STATUS_STEP_BUSY             = 0x4         # [2]
DBG_STATUS_STEP_BUSY_LSB         = 2
DBG_STATUS_STEP_BUSY_W           = 1
DBG_STATUS_DB_HELD               = 0x8         # [3] a head held by DB
DBG_STATUS_DB_HELD_LSB           = 3
DBG_STATUS_DB_HELD_W             = 1
DBG_STATUS_CAUSE                 = 0x1F0       # [8:4] cause of the current stop (layout DBG_CAUSE)
DBG_STATUS_CAUSE_LSB             = 4
DBG_STATUS_CAUSE_W               = 5
DBG_STATUS_BP_HIT                = 0x1E00      # [12:9] TBPs hit
DBG_STATUS_BP_HIT_LSB            = 9
DBG_STATUS_BP_HIT_W              = 4
DBG_STATUS_RVWP_HIT              = 0x6000      # [14:13] sticky
DBG_STATUS_RVWP_HIT_LSB          = 13
DBG_STATUS_RVWP_HIT_W            = 2
DBG_STATUS_RV_HALT_PEND          = 0x8000      # [15]
DBG_STATUS_RV_HALT_PEND_LSB      = 15
DBG_STATUS_RV_HALT_PEND_W        = 1
DBG_STATUS_RVWP_BUSY             = 0x10000     # [16]
DBG_STATUS_RVWP_BUSY_LSB         = 16
DBG_STATUS_RVWP_BUSY_W           = 1
DBG_STATUS_HIST_FROZEN           = 0x20000     # [17]
DBG_STATUS_HIST_FROZEN_LSB       = 17
DBG_STATUS_HIST_FROZEN_W         = 1
DBG_STATUS_RV_HALTED             = 0x40000     # [18]
DBG_STATUS_RV_HALTED_LSB         = 18
DBG_STATUS_RV_HALTED_W           = 1
DBG_STATUS_RV_SLEEP              = 0x80000     # [19]
DBG_STATUS_RV_SLEEP_LSB          = 19
DBG_STATUS_RV_SLEEP_W            = 1
DBG_STATUS_TRW_DROP              = 0x100000    # [20] sticky
DBG_STATUS_TRW_DROP_LSB          = 20
DBG_STATUS_TRW_DROP_W            = 1
DBG_STATUS_REC_DROP              = 0x200000    # [21] sticky
DBG_STATUS_REC_DROP_LSB          = 21
DBG_STATUS_REC_DROP_W            = 1
DBG_STATUS_SBA_REFUSED           = 0x400000    # [22] sticky
DBG_STATUS_SBA_REFUSED_LSB       = 22
DBG_STATUS_SBA_REFUSED_W         = 1
DBG_STATUS_DMI_BLOCKED           = 0x800000    # [23] sticky
DBG_STATUS_DMI_BLOCKED_LSB       = 23
DBG_STATUS_DMI_BLOCKED_W         = 1
DBG_STATUS_RV_SAMPLE_TIMEOUT     = 0x1000000   # [24] sticky
DBG_STATUS_RV_SAMPLE_TIMEOUT_LSB = 24
DBG_STATUS_RV_SAMPLE_TIMEOUT_W   = 1
DBG_STATUS_RV_DFAULT             = 0x2000000   # [25] sticky
DBG_STATUS_RV_DFAULT_LSB         = 25
DBG_STATUS_RV_DFAULT_W           = 1
DBG_STATUS_PTR_ERR               = 0x4000000   # [26] sticky
DBG_STATUS_PTR_ERR_LSB           = 26
DBG_STATUS_PTR_ERR_W             = 1
DBG_STATUS_RVWP_TIMEOUT          = 0x8000000   # [27] sticky
DBG_STATUS_RVWP_TIMEOUT_LSB      = 27
DBG_STATUS_RVWP_TIMEOUT_W        = 1
DBG_STATUS_MATCHQ_OVF            = 0x10000000  # [28] sticky, never expected
DBG_STATUS_MATCHQ_OVF_LSB        = 28
DBG_STATUS_MATCHQ_OVF_W          = 1
DBG_STATUS_DMI_RESP_ERR          = 0x20000000  # [29] sticky
DBG_STATUS_DMI_RESP_ERR_LSB      = 29
DBG_STATUS_DMI_RESP_ERR_W        = 1
DBG_STATUS_S_VALID               = 0x80000000  # [31]
DBG_STATUS_S_VALID_LSB           = 31
DBG_STATUS_S_VALID_W             = 1
# DBG_CAUSE (DB §10.1, §5.2): bits of DBG_STATUS.CAUSE in the documented order (one bit per cause)
DBG_CAUSE_NOW      = 0x1   # [0]
DBG_CAUSE_NOW_LSB  = 0
DBG_CAUSE_NOW_W    = 1
DBG_CAUSE_BP       = 0x2   # [1]
DBG_CAUSE_BP_LSB   = 1
DBG_CAUSE_BP_W     = 1
DBG_CAUSE_STEP     = 0x4   # [2]
DBG_CAUSE_STEP_LSB = 2
DBG_CAUSE_STEP_W   = 1
DBG_CAUSE_ERR      = 0x8   # [3]
DBG_CAUSE_ERR_LSB  = 3
DBG_CAUSE_ERR_W    = 1
DBG_CAUSE_HUNG     = 0x10  # [4]
DBG_CAUSE_HUNG_LSB = 4
DBG_CAUSE_HUNG_W   = 1
# DBG_STEP_N (DB §10.1)
DBG_STEP_N_N     = 0xFFFF  # [15:0] entries per QSTEP (0 = 1)
DBG_STEP_N_N_LSB = 0
DBG_STEP_N_N_W   = 16
# DBG_MATCHQ (DB §10.1)
DBG_MATCHQ_PENDING     = 0x7F  # [6:0] pending match thresholds
DBG_MATCHQ_PENDING_LSB = 0
DBG_MATCHQ_PENDING_W   = 7
# DBG_QSTATE (DB §10.1, CC §8.3): dispatcher state
DBG_QSTATE_DISP            = 0x700     # [10:8] dispatcher state (CC §8.3 order)
DBG_QSTATE_DISP_LSB        = 8
DBG_QSTATE_DISP_W          = 3
DBG_QSTATE_OUTSTANDING     = 0xFF0000  # [23:16] outstanding entries, saturating
DBG_QSTATE_OUTSTANDING_LSB = 16
DBG_QSTATE_OUTSTANDING_W   = 8
DBG_QSTATE_DISP_IDLE       = 0         # D_IDLE
DBG_QSTATE_DISP_POP        = 1         # D_POP
DBG_QSTATE_DISP_CHK        = 2         # D_CHK
DBG_QSTATE_DISP_ENQ        = 3         # D_ENQ
# DBG_RV_STATE (DB §10.1)
DBG_RV_STATE_HALTED            = 0x1         # [0]
DBG_RV_STATE_HALTED_LSB        = 0
DBG_RV_STATE_HALTED_W          = 1
DBG_RV_STATE_SLEEP             = 0x4         # [2]
DBG_RV_STATE_SLEEP_LSB         = 2
DBG_RV_STATE_SLEEP_W           = 1
DBG_RV_STATE_IN_RESET          = 0x8         # [3]
DBG_RV_STATE_IN_RESET_LSB      = 3
DBG_RV_STATE_IN_RESET_W        = 1
DBG_RV_STATE_SAMPLE_VALID      = 0x10        # [4]
DBG_RV_STATE_SAMPLE_VALID_LSB  = 4
DBG_RV_STATE_SAMPLE_VALID_W    = 1
DBG_RV_STATE_DMACTIVE          = 0x20        # [5] dmactive_o
DBG_RV_STATE_DMACTIVE_LSB      = 5
DBG_RV_STATE_DMACTIVE_W        = 1
DBG_RV_STATE_NDMRESET          = 0x40        # [6] ndmreset_o
DBG_RV_STATE_NDMRESET_LSB      = 6
DBG_RV_STATE_NDMRESET_W        = 1
DBG_RV_STATE_HALT_REQ          = 0x80        # [7] halt-request level
DBG_RV_STATE_HALT_REQ_LSB      = 7
DBG_RV_STATE_HALT_REQ_W        = 1
DBG_RV_STATE_SAMPLE_SEQ        = 0xFF00      # [15:8]
DBG_RV_STATE_SAMPLE_SEQ_LSB    = 8
DBG_RV_STATE_SAMPLE_SEQ_W      = 8
DBG_RV_STATE_DEBUG_ENTRIES     = 0xFFFF0000  # [31:16] debug-mode entries mod 2^16
DBG_RV_STATE_DEBUG_ENTRIES_LSB = 16
DBG_RV_STATE_DEBUG_ENTRIES_W   = 16
# DBG_RVWP_CTL (DB §10.1): RV watchpoint j < 2
DBG_RVWP_CTL_EN        = 0x1  # [0]
DBG_RVWP_CTL_EN_LSB    = 0
DBG_RVWP_CTL_EN_W      = 1
DBG_RVWP_CTL_LOAD      = 0x2  # [1]
DBG_RVWP_CTL_LOAD_LSB  = 1
DBG_RVWP_CTL_LOAD_W    = 1
DBG_RVWP_CTL_STORE     = 0x4  # [2]
DBG_RVWP_CTL_STORE_LSB = 2
DBG_RVWP_CTL_STORE_W   = 1
DBG_RVWP_CTL_HALT      = 0x8  # [3]
DBG_RVWP_CTL_HALT_LSB  = 3
DBG_RVWP_CTL_HALT_W    = 1
# DBG_TBP_CTL (DB §10.1, §5.3): tile breakpoint i < 4
DBG_TBP_CTL_EN             = 0x1         # [0]
DBG_TBP_CTL_EN_LSB         = 0
DBG_TBP_CTL_EN_W           = 1
DBG_TBP_CTL_KIND           = 0xE         # [3:1] 5..7 never match
DBG_TBP_CTL_KIND_LSB       = 1
DBG_TBP_CTL_KIND_W         = 3
DBG_TBP_CTL_R              = 0x10        # [4]
DBG_TBP_CTL_R_LSB          = 4
DBG_TBP_CTL_R_W            = 1
DBG_TBP_CTL_W              = 0x20        # [5]
DBG_TBP_CTL_W_LSB          = 5
DBG_TBP_CTL_W_W            = 1
DBG_TBP_CTL_AFTER          = 0x40        # [6]
DBG_TBP_CTL_AFTER_LSB      = 6
DBG_TBP_CTL_AFTER_W        = 1
DBG_TBP_CTL_HALT_RV        = 0x80        # [7]
DBG_TBP_CTL_HALT_RV_LSB    = 7
DBG_TBP_CTL_HALT_RV_W      = 1
DBG_TBP_CTL_COUNT_ONLY     = 0x100       # [8]
DBG_TBP_CTL_COUNT_ONLY_LSB = 8
DBG_TBP_CTL_COUNT_ONLY_W   = 1
DBG_TBP_CTL_CHMASK         = 0xFFFF0000  # [31:16] channels of a DRAM breakpoint
DBG_TBP_CTL_CHMASK_LSB     = 16
DBG_TBP_CTL_CHMASK_W       = 16
DBG_TBP_CTL_KIND_PIDX      = 0           # (pidx - A) mod 2^32 < (B - A) mod 2^32
DBG_TBP_CTL_KIND_OPCODE    = 1           # A: layout TBP_OPC_A, B: TBP_OPC_B
DBG_TBP_CTL_KIND_SPM       = 2           # an SPM extent overlaps lines [A, B)
DBG_TBP_CTL_KIND_DRAM      = 3           # the DRAM extent overlaps beats [A, B)
DBG_TBP_CTL_KIND_TICKET    = 4           # ticket = A
# TBP_OPC_A (DB §5.3): DBG_TBP_A of an OPCODE breakpoint
TBP_OPC_A_OPC          = 0xFF      # [7:0] opcode value
TBP_OPC_A_OPC_LSB      = 0
TBP_OPC_A_OPC_W        = 8
TBP_OPC_A_OPC_MASK     = 0xFF00    # [15:8] opcode care mask
TBP_OPC_A_OPC_MASK_LSB = 8
TBP_OPC_A_OPC_MASK_W   = 8
TBP_OPC_A_SUB          = 0xF0000   # [19:16] sub value
TBP_OPC_A_SUB_LSB      = 16
TBP_OPC_A_SUB_W        = 4
TBP_OPC_A_SUB_MASK     = 0xF00000  # [23:20] sub care mask
TBP_OPC_A_SUB_MASK_LSB = 20
TBP_OPC_A_SUB_MASK_W   = 4
# TBP_OPC_B (DB §5.3): DBG_TBP_B of an OPCODE breakpoint
TBP_OPC_B_UNITS     = 0xFF  # [7:0] unit mask; 0 = any unit
TBP_OPC_B_UNITS_LSB = 0
TBP_OPC_B_UNITS_W   = 8
# DBG_U_STATE (DB §5.5): per-unit probe u < 8
DBG_U_STATE_ACCEPTED        = 0x3      # [1:0] ops accepted and not done
DBG_U_STATE_ACCEPTED_LSB    = 0
DBG_U_STATE_ACCEPTED_W      = 2
DBG_U_STATE_HEAD            = 0x70     # [6:4] head state
DBG_U_STATE_HEAD_LSB        = 4
DBG_U_STATE_HEAD_W          = 3
DBG_U_STATE_HOLD            = 0xF00    # [11:8] hold sources at the head (layout HOLD_SRC)
DBG_U_STATE_HOLD_LSB        = 8
DBG_U_STATE_HOLD_W          = 4
DBG_U_STATE_OUTSTANDING     = 0xFF000  # [19:12] outstanding entries
DBG_U_STATE_OUTSTANDING_LSB = 12
DBG_U_STATE_OUTSTANDING_W   = 8
DBG_U_STATE_HEAD_EMPTY      = 0
DBG_U_STATE_HEAD_LOAD       = 1
DBG_U_STATE_HEAD_WAIT_DEP   = 2
DBG_U_STATE_HEAD_WAIT_SNAP  = 3
DBG_U_STATE_HEAD_HELD       = 4
DBG_U_STATE_HEAD_WAIT_UNIT  = 5
DBG_U_STATE_HEAD_ISSUE      = 6
# DBG_TRW_MASK (DB §10.1): reset 0x7FF
DBG_TRW_MASK_UNITS       = 0xFF   # [7:0]
DBG_TRW_MASK_UNITS_LSB   = 0
DBG_TRW_MASK_UNITS_W     = 8
DBG_TRW_MASK_MARKERS     = 0x100  # [8]
DBG_TRW_MASK_MARKERS_LSB = 8
DBG_TRW_MASK_MARKERS_W   = 1
DBG_TRW_MASK_STOPS       = 0x200  # [9]
DBG_TRW_MASK_STOPS_LSB   = 9
DBG_TRW_MASK_STOPS_W     = 1
DBG_TRW_MASK_RVHALTS     = 0x400  # [10]
DBG_TRW_MASK_RVHALTS_LSB = 10
DBG_TRW_MASK_RVHALTS_W   = 1
# DBG_HIST_INFO (DB §10.1)
DBG_HIST_INFO_PRESENT     = 0x1      # [0] the slot holds that push index, recorded after the last HIST_CLEAR
DBG_HIST_INFO_PRESENT_LSB = 0
DBG_HIST_INFO_PRESENT_W   = 1
DBG_HIST_INFO_UNIT        = 0xF0     # [7:4]
DBG_HIST_INFO_UNIT_LSB    = 4
DBG_HIST_INFO_UNIT_W      = 4
DBG_HIST_INFO_CODE        = 0xF00    # [11:8] checker code (0 = runs)
DBG_HIST_INFO_CODE_LSB    = 8
DBG_HIST_INFO_CODE_W      = 4
DBG_HIST_INFO_REC         = 0x1000   # [12]
DBG_HIST_INFO_REC_LSB     = 12
DBG_HIST_INFO_REC_W       = 1
DBG_HIST_INFO_TBPS        = 0x1E000  # [16:13] TBPs matched
DBG_HIST_INFO_TBPS_LSB    = 13
DBG_HIST_INFO_TBPS_W      = 4
DBG_HIST_INFO_STOP        = 0x20000  # [17] stop point
DBG_HIST_INFO_STOP_LSB    = 17
DBG_HIST_INFO_STOP_W      = 1
# IFIFO_CTRL (CC §7.2): w0 of a CTRL issue-FIFO entry
IFIFO_CTRL_CODE            = 0xF  # [3:0]
IFIFO_CTRL_CODE_LSB        = 0
IFIFO_CTRL_CODE_W          = 4
IFIFO_CTRL_CODE_QABORT     = 1
IFIFO_CTRL_CODE_QHOLD_SET  = 2
IFIFO_CTRL_CODE_QHOLD_CLR  = 3
IFIFO_CTRL_CODE_QCLEAR     = 4    # ticketed
IFIFO_CTRL_CODE_TRACE_MARK = 5    # w1 = marker value

# -------------------------------------------------------------------------------------------------
# Wide layouts (bit positions only)
# -------------------------------------------------------------------------------------------------
# IFIFO (CC §7.2): issue-FIFO entry (tsr_cdc_fifo, 32 entries LUTRAM)
IFIFO_BITS       = 387
IFIFO_KIND_LSB   = 0    # [1:0] entry kind
IFIFO_KIND_W     = 2
IFIFO_W_LSB      = 2    # [257:2] w0..w7 (OP; masked words are 0); CTRL: w0 = layout IFIFO_CTRL, w1 = marker value
IFIFO_W_W        = 256
IFIFO_DEP0_LSB   = 258  # [289:258] OP
IFIFO_DEP0_W     = 32
IFIFO_DEP1_LSB   = 290  # [321:290] OP
IFIFO_DEP1_W     = 32
IFIFO_REC_LSB    = 322  # [322]
IFIFO_REC_W      = 1
IFIFO_TICKET_LSB = 323  # [354:323] OP and QCLEAR; the dispatcher asserts it equals its own count
IFIFO_TICKET_W   = 32
IFIFO_PC_LSB     = 355  # [386:355] push PC (OP and TRACE_MARK)
IFIFO_PC_W       = 32
IFIFO_KIND_OP    = 0
IFIFO_KIND_CTRL  = 1

# -------------------------------------------------------------------------------------------------
# Word tables of fixed-size records and structures (word indices)
# -------------------------------------------------------------------------------------------------
# DESC (EX §4.3): launch descriptor (host -> RISC-V)
DESC_NWORDS    = 8
DESC_HDR_WORD  = 0  # layout DESC
DESC_ARG0_WORD = 1  # ARG0..ARG6: integers, SPM lines or DRAM handles (layout HANDLE)
DESC_ARG0_N    = 7
# LCQ_DONE (EX §4.4): DONE record (0x01, answer)
LCQ_DONE_NWORDS          = 8
LCQ_DONE_R0_WORD         = 0  # layout LCQ_R0
LCQ_DONE_SEQ_WORD        = 1  # SEQ of the answered descriptor
LCQ_DONE_RET_WORD        = 2  # kernel return value (int32)
LCQ_DONE_ERR_TICKET_WORD = 3  # first failing ticket; TICKET_NONE = none
LCQ_DONE_ERR_INFO_WORD   = 4
LCQ_DONE_ERR_W0_WORD     = 5
LCQ_DONE_ERR_PIDX_WORD   = 6
LCQ_DONE_DURATION_WORD   = 7  # core cycles from the pop to the post, saturating
# LCQ_PONG (EX §4.4): PONG record (0x02, answer)
LCQ_PONG_NWORDS        = 8
LCQ_PONG_R0_WORD       = 0
LCQ_PONG_SEQ_WORD      = 1
LCQ_PONG_ARG0_WORD     = 2  # ARG0 echo
LCQ_PONG_DURATION_WORD = 7
# LCQ_HELLO (EX §4.4, SW §6.3): HELLO record (0x80, event)
LCQ_HELLO_NWORDS           = 8
LCQ_HELLO_R0_WORD          = 0
LCQ_HELLO_TAKEN_WORD       = 1  # TAKEN at the post
LCQ_HELLO_ID_LO_WORD       = 2  # image id [31:0]
LCQ_HELLO_ID_HI_WORD       = 3  # image id [63:32]
LCQ_HELLO_VERSIONS_WORD    = 4  # layout HELLO_R4
LCQ_HELLO_N_KERNELS_WORD   = 5  # kernel count
LCQ_HELLO_BOOT_CYCLES_WORD = 7  # boot core cycles
# LCQ_TRAP (EX §4.4, DB §6.3): TRAP record (0x81)
LCQ_TRAP_NWORDS        = 8
LCQ_TRAP_R0_WORD       = 0
LCQ_TRAP_SEQ_WORD      = 1  # SEQ (answer) or TAKEN (event)
LCQ_TRAP_MCAUSE_WORD   = 2
LCQ_TRAP_MEPC_WORD     = 3
LCQ_TRAP_MTVAL_WORD    = 4
LCQ_TRAP_RT_STATE_WORD = 5  # at the trap
LCQ_TRAP_RT_INFO_WORD  = 6
LCQ_TRAP_MTIME_WORD    = 7  # MTIME[31:0]
# LCQ_EVENT (EX §4.4): kernel event record (0xC0..0xFF)
LCQ_EVENT_NWORDS    = 8
LCQ_EVENT_R0_WORD   = 0  # type, kernel-defined [23:8], seq[7:0]
LCQ_EVENT_SEQ_WORD  = 1  # SEQ of the running launch
LCQ_EVENT_DATA_WORD = 2  # r2..r7 kernel-defined
LCQ_EVENT_DATA_N    = 6
# REC (DB §8.2): completion record (REC_DATA0..3)
REC_NWORDS      = 4
REC_TICKET_WORD = 0  # ticket (unit 14 for markers)
REC_T_END_WORD  = 1  # MTIME[31:0] at completion (markers: dispatch time)
REC_RUN_WORD    = 2  # start -> end cycles, saturating (markers: the value)
REC_INFO_WORD   = 3  # layout REC_DATA3
# TRW_OP (DB §8.3): trace-writer OP record
TRW_OP_NWORDS        = 8
TRW_OP_HDR_WORD      = 0  # layout TRW_W0
TRW_OP_PIDX_WORD     = 1  # push index
TRW_OP_TICKET_WORD   = 2
TRW_OP_T_END_LO_WORD = 3  # MTIME
TRW_OP_T_END_HI_WORD = 4
TRW_OP_RUN_WORD      = 5  # t_end - t_start (0 if not run), saturating
TRW_OP_QWAIT_WORD    = 6  # t_head - t_dispatch
TRW_OP_HWAIT_WORD    = 7  # t_start - t_head; t_end - t_head for non-run entries
# TRW_MARK (DB §8.3): trace-writer MARK record
TRW_MARK_NWORDS         = 8
TRW_MARK_HDR_WORD       = 0
TRW_MARK_NEXT_PIDX_WORD = 1  # next push index
TRW_MARK_VALUE_WORD     = 2  # marker value
TRW_MARK_T_LO_WORD      = 3  # dispatch time
TRW_MARK_T_HI_WORD      = 4
# TRW_STOP (DB §8.3): trace-writer STOP record
TRW_STOP_NWORDS         = 8
TRW_STOP_HDR_WORD       = 0
TRW_STOP_T_WORD         = 1  # threshold
TRW_STOP_CAUSE_WORD     = 2  # cause bits (layout DBG_CAUSE)
TRW_STOP_T_LO_WORD      = 3  # time
TRW_STOP_T_HI_WORD      = 4
TRW_STOP_ACTIVATED_WORD = 5  # 1 activated, 0 released
# TRW_RVHALT (DB §8.3): trace-writer RVHALT record
TRW_RVHALT_NWORDS      = 8
TRW_RVHALT_HDR_WORD    = 0
TRW_RVHALT_HALTED_WORD = 2  # 1 halted, 0 resumed
# TRW_DROP (DB §8.3): trace-writer DROP record
TRW_DROP_NWORDS       = 8
TRW_DROP_HDR_WORD     = 0
TRW_DROP_DROPPED_WORD = 1  # records dropped since the previous DROP record
# DBG_INFO (DB §6.3): __tsr_dbg (section .tsr_dbg, 64 B)
DBG_INFO_NWORDS            = 16
DBG_INFO_MAGIC_WORD        = 0   # DBG_INFO_MAGIC
DBG_INFO_VERSION_WORD      = 1   # 1
DBG_INFO_BUILD_ID_WORD     = 2   # ELF note build id
DBG_INFO_ATTACHED_WORD     = 3   # written by the debugger
DBG_INFO_LAUNCH_BREAK_WORD = 4   # 0 off, 0xFFFFFFFF all, else kernel index + 1
DBG_INFO_TRAP_COUNT_WORD   = 5
# CRASH (DB §6.3): __tsr_crash (section .tsr_crash, 192 B)
CRASH_NWORDS        = 48
CRASH_MAGIC_WORD    = 0   # CRASH_MAGIC
CRASH_VERSION_WORD  = 1   # 1
CRASH_MCAUSE_WORD   = 2
CRASH_MEPC_WORD     = 3
CRASH_MTVAL_WORD    = 4
CRASH_MSTATUS_WORD  = 5
CRASH_MTIME_LO_WORD = 6
CRASH_MTIME_HI_WORD = 7
CRASH_SEQ_WORD      = 8   # launch SEQ
CRASH_KERNEL_WORD   = 9   # kernel index
CRASH_RT_STATE_WORD = 10
CRASH_RT_INFO_WORD  = 11
CRASH_X1_WORD       = 12  # x1..x31 at words 12..42
CRASH_X1_N          = 31
CRASH_NESTED_WORD   = 43

# -------------------------------------------------------------------------------------------------
# Units (EX §7.3, CC §7.4, §9.4)
# -------------------------------------------------------------------------------------------------
UNIT_DMA          = 0          # EX §7.3, CC §7.4: LD, ST, KVST
UNIT_QACT         = 1          # EX §7.3, CC §7.4: QACT
UNIT_MVM          = 2          # EX §7.3, CC §7.4, §9.4: MVM, ATTQK, ATTPV
UNIT_VU           = 3          # EX §7.3, CC §7.4, §9.4: VOP, RED, ROPE, TOPK, SOP, DQROW, ASMAX, AMASK, AMERGE, ANORM (VMASK from M9)
UNIT_ATT          = 4          # EX §7.3, CC §7.4: reserved for a future AT attention engine; absent in M6 (was CMD)
UNIT_SYNC         = 5          # EX §7.3, CC §7.4: NOP, FENCE, QCLEAR and every refused opcode
UNIT_MARKER       = 14         # EX §7.3, CC §7.4: software marker records (REC / TRW); never a valid dependency
UNIT_NONE         = 15         # EX §7.3, CC §7.4: no dependency
UNIT_RSVD_FIRST   = 6          # EX §7.3: ids 6..13 reserved: a dependency on them is BAD_DEP
UNIT_RSVD_LAST    = 13
QDEPTH_U200_DMA   = 8          # Q_DEPTH of the U200 M6 bitstream
QDEPTH_U200_QACT  = 4
QDEPTH_U200_MVM   = 4
QDEPTH_U200_VU    = 16
QDEPTH_U200_ATT   = 0
QDEPTH_U200_SYNC  = 4
QDEPTH_PARAM_DMA  = 8          # depth parameter when the unit is built
QDEPTH_PARAM_QACT = 4
QDEPTH_PARAM_MVM  = 4
QDEPTH_PARAM_VU   = 16
QDEPTH_PARAM_ATT  = 8
QDEPTH_PARAM_SYNC = 4
ACCEPT_MAX_DMA    = 1          # I3: ops accepted and not done (CC §9.4)
ACCEPT_MAX_QACT   = 1
ACCEPT_MAX_MVM    = 2
ACCEPT_MAX_VU     = 2
ACCEPT_MAX_ATT    = 1
ACCEPT_MAX_SYNC   = 1
QDEPTH_U200_REG   = 0x8081088  # Q_DEPTH register value of the U200 M6 bitstream

# -------------------------------------------------------------------------------------------------
# Opcodes (EX §7.2); reserved and retired opcodes complete ILLEGAL_OP on SYNC
# -------------------------------------------------------------------------------------------------
OP_NOP    = 0x0   # kept, unit SYNC: EX §7.2: SYNC join point: completes when its dependencies are done
OP_END    = 0x1   # reserved, unit SYNC: EX §7.2: v1 END: ILLEGAL_OP
OP_LI     = 0x2   # reserved, unit SYNC: EX §7.2: v1 LI: ILLEGAL_OP
OP_ADDI   = 0x3   # reserved, unit SYNC: EX §7.2: v1 ADDI: ILLEGAL_OP
OP_MULI   = 0x4   # retired, unit SYNC: EX §7.2: retired, never reused
OP_RLD    = 0x5   # retired, unit SYNC: EX §7.2: retired, never reused
OP_LOOP   = 0x6   # retired, unit SYNC: EX §7.2: retired, never reused
OP_WAITM  = 0x7   # retired, unit SYNC: EX §7.2: retired, never reused
OP_JUMP   = 0x8   # retired, unit SYNC: EX §7.2: retired, never reused
OP_CALL   = 0x9   # retired, unit SYNC: EX §7.2: retired (never shipped), never reused
OP_LD     = 0x10  # kept, unit DMA: EX §7.2: DRAM -> SPM, encoding unchanged
OP_ST     = 0x11  # kept, unit DMA: EX §7.2: SPM -> DRAM, encoding unchanged
OP_DQROW  = 0x12  # respec, unit VU, CAPS.DQROW: EX §7.2, VU §5.6: SPM -> f32 dequantisation; simulator from M6, RTL in M8
OP_KVST   = 0x14  # new, unit DMA, CAPS.ATT: EX §7.2, AT §5.1: store K or V rows into a KV plane
OP_QACT   = 0x18  # kept, unit QACT: EX §7.2: f32 -> QA, encoding unchanged
OP_MVM    = 0x20  # kept, unit MVM: EX §7.2: matrix x activations, encoding unchanged
OP_ATTQK  = 0x21  # new, unit MVM, CAPS.ATT: EX §7.2, AT §5.2: scores and row maxima over a KV group range
OP_ATTPV  = 0x22  # new, unit MVM, CAPS.ATT: EX §7.2, AT §5.3: P.V channel partials and row sums
OP_VOP    = 0x30  # respec, unit VU, CAPS.VU: EX §7.2, VU §5.2: element-wise vector op
OP_RED    = 0x31  # respec, unit VU, CAPS.VU: EX §7.2, VU §5.3: segmented reduction with post-ops
OP_ROPE   = 0x32  # respec, unit VU, CAPS.VU: EX §7.2, VU §5.4: rotary embedding over T tokens
OP_TOPK   = 0x33  # respec, unit VU, CAPS.TOPK: EX §7.2, VU §5.5: top-k per row; simulator from M6, RTL in M8
OP_ASMAX  = 0x34  # new, unit VU, CAPS.ATT: EX §7.2, AT §5.4: softmax numerator
OP_AMASK  = 0x35  # new, unit VU, CAPS.ATT: EX §7.2, AT §5.5: general mask path
OP_AMERGE = 0x36  # new, unit VU, CAPS.ATT: EX §7.2, AT §5.6: channel tree and online merge
OP_ANORM  = 0x37  # new, unit VU, CAPS.ATT: EX §7.2, AT §5.7: final normalisation
OP_VMASK  = 0x38  # new, unit VU, CAPS.VMASK: EX §7.2, VU §5.7: position-window mask; ILLEGAL_OP until M9
OP_SOP    = 0x40  # respec, unit VU, CAPS.VU: EX §7.2, VU §5.8: scalar op on s0..s15
OP_FENCE  = 0x7F  # kept, unit SYNC: EX §7.2: SYNC barrier: completes when every op pushed before it is done

# Per opcode (index = opcode): unit of a legal opcode (UNIT_SYNC otherwise), legality, CAPS bits.
OP_UNIT = [
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 3, 5, 0, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5,
    2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5,
    3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
]
OP_LEGAL = [
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
OP_CAPS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 8, 32, 32, 32, 32, 128, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
]
OPCODES = {
    'NOP': 0x00, 'END': 0x01, 'LI': 0x02, 'ADDI': 0x03, 'MULI': 0x04, 'RLD': 0x05, 'LOOP': 0x06,
    'WAITM': 0x07, 'JUMP': 0x08, 'CALL': 0x09, 'LD': 0x10, 'ST': 0x11, 'DQROW': 0x12, 'KVST': 0x14,
    'QACT': 0x18, 'MVM': 0x20, 'ATTQK': 0x21, 'ATTPV': 0x22, 'VOP': 0x30, 'RED': 0x31,
    'ROPE': 0x32, 'TOPK': 0x33, 'ASMAX': 0x34, 'AMASK': 0x35, 'AMERGE': 0x36, 'ANORM': 0x37,
    'VMASK': 0x38, 'SOP': 0x40, 'FENCE': 0x7F,
}


# -------------------------------------------------------------------------------------------------
# Error codes (EX §8.1)
# -------------------------------------------------------------------------------------------------
ERR_NONE        = 0
ERR_ILLEGAL_OP  = 1   # reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it; ERR_INFO = 0
ERR_BAD_FMT     = 2   # unsupported weight format; ERR_INFO = 0
ERR_QA_MISMATCH = 3   # (reserved) never raised
ERR_SPM_RANGE   = 4   # SPM operand out of range (spec range-check rule); ERR_INFO = start address of the failing operand
ERR_DRAM_RANGE  = 5   # DRAM operand out of range; ERR_INFO = start address
ERR_K_RANGE     = 6   # K > K_MAX or not a multiple of the block size; ERR_INFO = 0
ERR_T_RANGE     = 7   # T = 0 or T > T_MAX; ERR_INFO = 0
ERR_ABORTED     = 8   # completed without running: a QABORT, or its unit was in error state; ERR_INFO = 0
ERR_DEP_FAILED  = 11  # a dependency completed with an error; ERR_INFO = 0
ERR_BAD_DEP     = 12  # a dependency on a ticket not pushed yet, on unit 6..14 or an absent unit, or below its unit's clear point; ERR_INFO = 0
ERR_OPERAND     = 13  # an operand value outside the range its op defines, other than K, T and the address ranges (VU and AT ops only); ERR_INFO = the offending value
# code 9 retired (v1 LOOP_NEST), never raised, never reused
# code 10 retired (v1 WAIT_TIMEOUT), never raised, never reused
ERR_CHECK_N = 8  # dispatch check order: ILLEGAL_OP > BAD_DEP > BAD_FMT > K_RANGE > T_RANGE > OPERAND > SPM_RANGE > DRAM_RANGE

ERR_CHECK_ORDER = [
    ERR_ILLEGAL_OP, ERR_BAD_DEP, ERR_BAD_FMT, ERR_K_RANGE, ERR_T_RANGE, ERR_OPERAND, ERR_SPM_RANGE,
    ERR_DRAM_RANGE,
]
LCQ_STATUS_PRECEDENCE = [
    LCQ_R0_STATUS_TRAP, LCQ_R0_STATUS_BAD_DESC, LCQ_R0_STATUS_ABORTED, LCQ_R0_STATUS_TILE_ERR,
    LCQ_R0_STATUS_PLAT_ERR, LCQ_R0_STATUS_STOPPED, LCQ_R0_STATUS_KERNEL_ERR, LCQ_R0_STATUS_OK,
]


def err_check_rank(code):
    """Rank of an error code in the dispatch check order (0 = first); 0xFF if not a check."""
    return ERR_CHECK_ORDER.index(code) if code in ERR_CHECK_ORDER else 0xFF


def lcq_status_rank(status):
    """Rank of a launch STATUS in the precedence of EX §4.4 (0 = highest); 0xFF if reserved."""
    return LCQ_STATUS_PRECEDENCE.index(status) if status in LCQ_STATUS_PRECEDENCE else 0xFF


# -------------------------------------------------------------------------------------------------
# PERF slots (EX §6): C = NUM_CH, X = 5 + 2C
# -------------------------------------------------------------------------------------------------
PERF_CYCLES     = 0  # EX §6: time, core cycles
PERF_RETIRED    = 1  # EX §6: event, queue entries completed on every unit, any outcome
PERF_MVM_CYCLES = 2  # EX §6: time, the MVM unit holds an op
PERF_VU_CYCLES  = 3  # EX §6, VU §5.10: time, the VU holds an op
# slot 4 reserved (was WAITM), reads 0 (EX §6)
PERF_XOFF_DMA_CYCLES     = 0   # EX §6: X + 0, time, the DMA unit holds an op
PERF_XOFF_QACT_CYCLES    = 1   # EX §6: X + 1, time, the QACT unit holds an op
PERF_XOFF_ATT_CYCLES     = 2   # EX §6: X + 2, time, unit 4 (AT) holds an op; reads 0 without it
PERF_XOFF_MVM_WAIT       = 3   # EX §6: X + 3, time, MVM idle while its queue holds an entry that cannot start
PERF_XOFF_MVM_EMPTY      = 4   # EX §6: X + 4, time, MVM idle and its queue empty (issue-bound)
PERF_XOFF_HOLD_CYCLES    = 5   # EX §6: X + 5, time, some queue head is held
PERF_XOFF_PUSHES         = 6   # EX §6: X + 6, event, entries pushed
PERF_XOFF_PUSH_STALL     = 7   # EX §6: X + 7, rv_time, clk_rv cycles a push waited for a credit
PERF_XOFF_BRIDGE_ACC     = 8   # EX §6: X + 8, event, SPM bridge accesses
PERF_XOFF_BRIDGE_WAIT    = 9   # EX §6: X + 9, rv_time, clk_rv cycles bridge accesses waited
PERF_XOFF_REC_DROPPED    = 10  # EX §6: X + 10, event, completion records dropped
PERF_XOFF_LAUNCHES       = 11  # EX §6: X + 11, event, answers posted
PERF_XOFF_RV_WFI         = 12  # EX §6: X + 12, time, the RISC-V sleeps in WFI
PERF_XOFF_VU_SPM_WAIT    = 13  # EX §6, VU §5.10: X + 13, time, a VU SPM request is raised and not granted
PERF_XOFF_VU_HAZARD      = 14  # EX §6, VU §5.10: X + 14, time, an accepted VU op waits for an earlier VU op
PERF_XOFF_VU_LINES       = 15  # EX §6, VU §5.10: X + 15, event, destination lines written by VU ops
PERF_XOFF_ATT_QK_CYCLES  = 16  # EX §6, AT §10: X + 16, time, the MVM unit holds an ATTQK
PERF_XOFF_ATT_PV_CYCLES  = 17  # EX §6, AT §10: X + 17, time, the MVM unit holds an ATTPV
PERF_XOFF_ATT_PATH_STALL = 18  # EX §6, AT §10: X + 18, time, a cluster in an ATT op waits for the gather or the p broadcast
PERF_XOFF_KVST_RMW       = 19  # EX §6, AT §10: X + 19, event, meta beats read-modify-written by KVST
PERF_XOFF_SPM_RD_WAIT    = 20  # EX §6, CC §6.3: X + 20, time, client-cycles a raised SPM read request waits for its bank port
PERF_XOFF_SPM_WR_WAIT    = 21  # EX §6, CC §6.3: X + 21, time, the same for write requests
PERF_TRW_RECORDS         = 61  # EX §6, DB §8.1: event, trace records written to DRAM
PERF_TRW_DROPS           = 62  # EX §6, DB §8.1: event, trace records dropped
PERF_DBG_STOP_CYCLES     = 63  # EX §6, DB §8.1: time_nohalt, ACTIVE and (RV_HALTED or STOP_ACTIVE); a measurement is valid only if 0

def perf_x(C):
    """First extension slot X = 5 + 2C."""
    return 5 + 2 * C


def perf_ext(C, xoff):
    return perf_x(C) + xoff


def perf_ch_rd_beats(c):
    return 5 + c


def perf_ch_wait(C, c):
    return 5 + 1 * C + c


def perf_slots(C):
    """Every named PERF slot for NUM_CH = C: name -> index (per-channel names get [c])."""
    out = {'CYCLES': 0, 'RETIRED': 1, 'MVM_CYCLES': 2, 'VU_CYCLES': 3, 'TRW_RECORDS': 61, 'TRW_DROPS': 62, 'DBG_STOP_CYCLES': 63}
    for c in range(C):
        out['CH_RD_BEATS[%d]' % c] = 5 + 0 * C + c
        out['CH_WAIT[%d]' % c] = 5 + 1 * C + c
    for name, xoff in [('DMA_CYCLES', 0), ('QACT_CYCLES', 1), ('ATT_CYCLES', 2), ('MVM_WAIT', 3), ('MVM_EMPTY', 4), ('HOLD_CYCLES', 5), ('PUSHES', 6), ('PUSH_STALL', 7), ('BRIDGE_ACC', 8), ('BRIDGE_WAIT', 9), ('REC_DROPPED', 10), ('LAUNCHES', 11), ('RV_WFI', 12), ('VU_SPM_WAIT', 13), ('VU_HAZARD', 14), ('VU_LINES', 15), ('ATT_QK_CYCLES', 16), ('ATT_PV_CYCLES', 17), ('ATT_PATH_STALL', 18), ('KVST_RMW', 19), ('SPM_RD_WAIT', 20), ('SPM_WR_WAIT', 21)]:
        out[name] = perf_x(C) + xoff
    return out


# -------------------------------------------------------------------------------------------------
# Instruction encodings of the v1 ops, unchanged (spec ISA, EX §7.2)
# -------------------------------------------------------------------------------------------------
# NOP 0x00 (spec ISA, EX §7.1, owner spec)
NOP_RSV_W0 = 0xFFFFFF00  # must be 0 (ILLEGAL_OP)
NOP_RSV_W1 = 0xFFFFFFFF
NOP_RSV_W2 = 0xFFFFFFFF
NOP_RSV_W3 = 0xFFFFFFFF
NOP_RSV_W4 = 0xFFFFFFFF
NOP_RSV_W5 = 0xFFFFFFFF
NOP_RSV_W6 = 0xFFFFFFFF
NOP_RSV_W7 = 0xFFFFFFFF
NOP_WORDS  = 1           # words holding a field (w0 always)
# FENCE 0x7F (spec ISA, EX §7.1, owner spec)
FENCE_RSV_W0 = 0xFFFFFF00  # must be 0 (ILLEGAL_OP)
FENCE_RSV_W1 = 0xFFFFFFFF
FENCE_RSV_W2 = 0xFFFFFFFF
FENCE_RSV_W3 = 0xFFFFFFFF
FENCE_RSV_W4 = 0xFFFFFFFF
FENCE_RSV_W5 = 0xFFFFFFFF
FENCE_RSV_W6 = 0xFFFFFFFF
FENCE_RSV_W7 = 0xFFFFFFFF
FENCE_WORDS  = 1           # words holding a field (w0 always)
# LD 0x10 (spec ISA, EX §7.2, owner spec)
LD_CH         = 0xF00       # w0[11:8] channel
LD_CH_LSB     = 8
LD_CH_W       = 4
LD_CH_WORD    = 0
LD_CH_BIT     = 8
LD_SPM        = 0xFFFFFFFF  # w1 SPM destination line
LD_SPM_LSB    = 0
LD_SPM_W      = 32
LD_SPM_WORD   = 1
LD_SPM_BIT    = 32
LD_DRAM       = 0xFFFFFFFF  # w2 DRAM source beat
LD_DRAM_LSB   = 0
LD_DRAM_W     = 32
LD_DRAM_WORD  = 2
LD_DRAM_BIT   = 64
LD_LINES      = 0xFFFFFFFF  # w3 lines
LD_LINES_LSB  = 0
LD_LINES_W    = 32
LD_LINES_WORD = 3
LD_LINES_BIT  = 96
LD_RSV_W0     = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
LD_RSV_W1     = 0x0
LD_RSV_W2     = 0x0
LD_RSV_W3     = 0x0
LD_RSV_W4     = 0xFFFFFFFF
LD_RSV_W5     = 0xFFFFFFFF
LD_RSV_W6     = 0xFFFFFFFF
LD_RSV_W7     = 0xFFFFFFFF
LD_WORDS      = 15          # words holding a field (w0 always)
# ST 0x11 (spec ISA, EX §7.2, owner spec)
ST_CH         = 0xF00       # w0[11:8] channel
ST_CH_LSB     = 8
ST_CH_W       = 4
ST_CH_WORD    = 0
ST_CH_BIT     = 8
ST_SPM        = 0xFFFFFFFF  # w1 SPM source line
ST_SPM_LSB    = 0
ST_SPM_W      = 32
ST_SPM_WORD   = 1
ST_SPM_BIT    = 32
ST_DRAM       = 0xFFFFFFFF  # w2 DRAM destination beat
ST_DRAM_LSB   = 0
ST_DRAM_W     = 32
ST_DRAM_WORD  = 2
ST_DRAM_BIT   = 64
ST_LINES      = 0xFFFFFFFF  # w3 lines
ST_LINES_LSB  = 0
ST_LINES_W    = 32
ST_LINES_WORD = 3
ST_LINES_BIT  = 96
ST_RSV_W0     = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
ST_RSV_W1     = 0x0
ST_RSV_W2     = 0x0
ST_RSV_W3     = 0x0
ST_RSV_W4     = 0xFFFFFFFF
ST_RSV_W5     = 0xFFFFFFFF
ST_RSV_W6     = 0xFFFFFFFF
ST_RSV_W7     = 0xFFFFFFFF
ST_WORDS      = 15          # words holding a field (w0 always)
# QACT 0x18 (spec ISA, EX §7.2, owner spec)
QACT_KIND            = 0xF00       # w0[11:8] QA kind
QACT_KIND_LSB        = 8
QACT_KIND_W          = 4
QACT_KIND_WORD       = 0
QACT_KIND_BIT        = 8
QACT_DST             = 0xFFFFFFFF  # w1 QA destination line
QACT_DST_LSB         = 0
QACT_DST_W           = 32
QACT_DST_WORD        = 1
QACT_DST_BIT         = 32
QACT_SRC             = 0xFFFFFFFF  # w2 f32 source line
QACT_SRC_LSB         = 0
QACT_SRC_W           = 32
QACT_SRC_WORD        = 2
QACT_SRC_BIT         = 64
QACT_K               = 0xFFFFFFFF  # w3
QACT_K_LSB           = 0
QACT_K_W             = 32
QACT_K_WORD          = 3
QACT_K_BIT           = 96
QACT_T               = 0xFFFFFFFF  # w4
QACT_T_LSB           = 0
QACT_T_W             = 32
QACT_T_WORD          = 4
QACT_T_BIT           = 128
QACT_SRC_STRIDE      = 0xFFFFFFFF  # w5 lines; 0 = ceil(4K/64)
QACT_SRC_STRIDE_LSB  = 0
QACT_SRC_STRIDE_W    = 32
QACT_SRC_STRIDE_WORD = 5
QACT_SRC_STRIDE_BIT  = 160
# QACT_KIND takes the QA_* values
QACT_RSV_W0 = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
QACT_RSV_W1 = 0x0
QACT_RSV_W2 = 0x0
QACT_RSV_W3 = 0x0
QACT_RSV_W4 = 0x0
QACT_RSV_W5 = 0x0
QACT_RSV_W6 = 0xFFFFFFFF
QACT_RSV_W7 = 0xFFFFFFFF
QACT_WORDS  = 0x3F        # words holding a field (w0 always)
# MVM 0x20 (spec ISA, EX §7.2, owner spec)
MVM_FMT             = 0xF00       # w0[11:8] weight format
MVM_FMT_LSB         = 8
MVM_FMT_W           = 4
MVM_FMT_WORD        = 0
MVM_FMT_BIT         = 8
MVM_ACC             = 0x1000      # w0[12] y += W.x
MVM_ACC_LSB         = 12
MVM_ACC_W           = 1
MVM_ACC_WORD        = 0
MVM_ACC_BIT         = 12
MVM_DST             = 0xFFFFFFFF  # w1 destination line
MVM_DST_LSB         = 0
MVM_DST_W           = 32
MVM_DST_WORD        = 1
MVM_DST_BIT         = 32
MVM_X               = 0xFFFFFFFF  # w2 QA activation line
MVM_X_LSB           = 0
MVM_X_W             = 32
MVM_X_WORD          = 2
MVM_X_BIT           = 64
MVM_W_OFF           = 0xFFFFFFFF  # w3 weight offset (beats)
MVM_W_OFF_LSB       = 0
MVM_W_OFF_W         = 32
MVM_W_OFF_WORD      = 3
MVM_W_OFF_BIT       = 96
MVM_N               = 0xFFFFFFFF  # w4 rows
MVM_N_LSB           = 0
MVM_N_W             = 32
MVM_N_WORD          = 4
MVM_N_BIT           = 128
MVM_K               = 0xFFFF      # w5[15:0]
MVM_K_LSB           = 0
MVM_K_W             = 16
MVM_K_WORD          = 5
MVM_K_BIT           = 160
MVM_T               = 0xF0000     # w5[19:16]
MVM_T_LSB           = 16
MVM_T_W             = 4
MVM_T_WORD          = 5
MVM_T_BIT           = 176
MVM_DST_STRIDE      = 0xFFFFFFFF  # w6 lines; 0 = ceil(4N/64)
MVM_DST_STRIDE_LSB  = 0
MVM_DST_STRIDE_W    = 32
MVM_DST_STRIDE_WORD = 6
MVM_DST_STRIDE_BIT  = 192
# MVM_FMT takes the FMT_* values
MVM_RSV_W0 = 0xFFFFE000  # must be 0 (ILLEGAL_OP)
MVM_RSV_W1 = 0x0
MVM_RSV_W2 = 0x0
MVM_RSV_W3 = 0x0
MVM_RSV_W4 = 0x0
MVM_RSV_W5 = 0xFFF00000
MVM_RSV_W6 = 0x0
MVM_RSV_W7 = 0xFFFFFFFF
MVM_WORDS  = 0x7F        # words holding a field (w0 always)

# -------------------------------------------------------------------------------------------------
# VU encodings (VU §5)
# -------------------------------------------------------------------------------------------------
# VOP 0x30 (VU §5.2, owner VU)
VOP_CMODE         = 0xF00       # w0[11:8] per-row scalar c
VOP_CMODE_LSB     = 8
VOP_CMODE_W       = 4
VOP_CMODE_WORD    = 0
VOP_CMODE_BIT     = 8
VOP_FUNC          = 0xFF0000    # x16[7:0] function (VOP_FUNCS)
VOP_FUNC_LSB      = 16
VOP_FUNC_W        = 8
VOP_FUNC_WORD     = 0
VOP_FUNC_BIT      = 16
VOP_SI            = 0xF000000   # x16[11:8] s register for CMODE SREG, else 0
VOP_SI_LSB        = 24
VOP_SI_W          = 4
VOP_SI_WORD       = 0
VOP_SI_BIT        = 24
VOP_D             = 0xFFFFFF    # w1[23:0] destination line
VOP_D_LSB         = 0
VOP_D_W           = 24
VOP_D_WORD        = 1
VOP_D_BIT         = 32
VOP_A             = 0xFFFFFF    # w2[23:0] source a line (0 if the function reads no a)
VOP_A_LSB         = 0
VOP_A_W           = 24
VOP_A_WORD        = 2
VOP_A_BIT         = 64
VOP_B             = 0xFFFFFF    # w3[23:0] source b line (0 if unused)
VOP_B_LSB         = 0
VOP_B_W           = 24
VOP_B_WORD        = 3
VOP_B_BIT         = 96
VOP_C             = 0xFFFFFF    # w4[23:0] row-vector line for CMODE VEC, else 0
VOP_C_LSB         = 0
VOP_C_W           = 24
VOP_C_WORD        = 4
VOP_C_BIT         = 128
VOP_M             = 0xFFFFFF    # w5[23:0] elements per row
VOP_M_LSB         = 0
VOP_M_W           = 24
VOP_M_WORD        = 5
VOP_M_BIT         = 160
VOP_ROWS          = 0xFFFF      # w6[15:0] rows
VOP_ROWS_LSB      = 0
VOP_ROWS_W        = 16
VOP_ROWS_WORD     = 6
VOP_ROWS_BIT      = 192
VOP_D_STRIDE      = 0xFFFF0000  # w6[31:16] lines
VOP_D_STRIDE_LSB  = 16
VOP_D_STRIDE_W    = 16
VOP_D_STRIDE_WORD = 6
VOP_D_STRIDE_BIT  = 208
VOP_A_STRIDE      = 0xFFFF      # w7[15:0] lines (0 if a is unused)
VOP_A_STRIDE_LSB  = 0
VOP_A_STRIDE_W    = 16
VOP_A_STRIDE_WORD = 7
VOP_A_STRIDE_BIT  = 224
VOP_B_STRIDE      = 0xFFFF0000  # w7[31:16] lines (0 if b is unused)
VOP_B_STRIDE_LSB  = 16
VOP_B_STRIDE_W    = 16
VOP_B_STRIDE_WORD = 7
VOP_B_STRIDE_BIT  = 240
VOP_CMODE_NONE    = 0           # no c
VOP_CMODE_SREG    = 1           # c = s[SI]
VOP_CMODE_VEC     = 2           # c_r = word r of the vector at line C
VOP_FUNC_ADD      = 0           # reads ab, rate F: FA(a, b)
VOP_FUNC_SUB      = 1           # reads ab, rate F: FS(a, b)
VOP_FUNC_MUL      = 2           # reads ab, rate F: FM(a, b)
VOP_FUNC_MAX      = 3           # reads ab, rate F: MAXT(a, b)
VOP_FUNC_MULC     = 4           # reads ac, rate F: FM(a, c)
VOP_FUNC_ADDC     = 5           # reads ac, rate F: FA(a, c)
VOP_FUNC_SILU     = 6           # reads a, rate H: SILU(a)
VOP_FUNC_SWIGLU   = 7           # reads ab, rate H: FM(SILU(a), b)
VOP_FUNC_EXP      = 8           # reads a, rate F: EXPK(a, P1)
VOP_FUNC_COPY     = 9           # reads a, rate F: a (raw)
VOP_FUNC_MIN      = 10          # reads ab, rate F: MINT(a, b)
VOP_FUNC_SUBC     = 11          # reads ac, rate F: FS(a, c)
VOP_FUNC_MULCB    = 12          # reads abc, rate F: FM(FM(a, c), b)
VOP_FUNC_MULCADD  = 13          # reads abc, rate F: FA(FM(a, c), b)
VOP_FUNC_ADDMULC  = 14          # reads abc, rate F: FM(FA(a, b), c)
VOP_FUNC_EXPSUB   = 15          # reads ab, rate F: EXPSUB(a, b)
VOP_FUNC_EXPSUBC  = 0x10        # reads ac, rate F: EXPSUB(a, c)
VOP_FUNC_SIGMOID  = 0x11        # reads a, rate H: SIG(a)
VOP_FUNC_GELU     = 0x12        # reads a, rate H: GELU(a), exact tanh form
VOP_FUNC_GEGLU    = 0x13        # reads ab, rate H: FM(GELU(a), b)
VOP_FUNC_TANH     = 0x14        # reads a, rate H: TANH(a)
VOP_FUNC_RECIP    = 0x15        # reads a, rate H: RC(a), correctly rounded
VOP_FUNC_SETC     = 0x16        # reads c, rate F: c (raw)
VOP_FUNC_GELU16   = 0x17        # reads a, rate H: ggml-CPU fp16 form (VU §6.5)
VOP_FUNC_GEGLU16  = 0x18        # reads ab, rate H: GEGLU16(a, b) (VU §6.5)
VOP_RSV_W0        = 0xF000F000  # must be 0 (ILLEGAL_OP)
VOP_RSV_W1        = 0xFF000000
VOP_RSV_W2        = 0xFF000000
VOP_RSV_W3        = 0xFF000000
VOP_RSV_W4        = 0xFF000000
VOP_RSV_W5        = 0xFF000000
VOP_RSV_W6        = 0x0
VOP_RSV_W7        = 0x0
VOP_WORDS         = 0xFF        # words holding a field (w0 always)
# RED 0x31 (VU §5.3, owner VU)
RED_KIND          = 0xF00       # w0[11:8]
RED_KIND_LSB      = 8
RED_KIND_W        = 4
RED_KIND_WORD     = 0
RED_KIND_BIT      = 8
RED_TO_S          = 0x1000      # w0[12] 1: result to s[SD] (ROWS must be 1); 0: per-row vector at D
RED_TO_S_LSB      = 12
RED_TO_S_W        = 1
RED_TO_S_WORD     = 0
RED_TO_S_BIT      = 12
RED_SD            = 0xF0000     # x16[3:0] destination s register if TO_S, else 0
RED_SD_LSB        = 16
RED_SD_W          = 4
RED_SD_WORD       = 0
RED_SD_BIT        = 16
RED_POST          = 0x300000    # x16[5:4] post-op; MAX requires NONE
RED_POST_LSB      = 20
RED_POST_W        = 2
RED_POST_WORD     = 0
RED_POST_BIT      = 20
RED_SK            = 0xF000000   # x16[11:8] RMS scale register (typically 1/n), else 0
RED_SK_LSB        = 24
RED_SK_W          = 4
RED_SK_WORD       = 0
RED_SK_BIT        = 24
RED_SE            = 0xF0000000  # x16[15:12] RMS epsilon register, else 0
RED_SE_LSB        = 28
RED_SE_W          = 4
RED_SE_WORD       = 0
RED_SE_BIT        = 28
RED_D             = 0xFFFFFF    # w1[23:0] result vector line (rows words); 0 if TO_S
RED_D_LSB         = 0
RED_D_W           = 24
RED_D_WORD        = 1
RED_D_BIT         = 32
RED_A             = 0xFFFFFF    # w2[23:0] source line
RED_A_LSB         = 0
RED_A_W           = 24
RED_A_WORD        = 2
RED_A_BIT         = 64
RED_M             = 0xFFFFFF    # w5[23:0] elements per row
RED_M_LSB         = 0
RED_M_W           = 24
RED_M_WORD        = 5
RED_M_BIT         = 160
RED_ROWS          = 0xFFFF      # w6[15:0] rows
RED_ROWS_LSB      = 0
RED_ROWS_W        = 16
RED_ROWS_WORD     = 6
RED_ROWS_BIT      = 192
RED_A_STRIDE      = 0xFFFF      # w7[15:0] lines
RED_A_STRIDE_LSB  = 0
RED_A_STRIDE_W    = 16
RED_A_STRIDE_WORD = 7
RED_A_STRIDE_BIT  = 224
RED_KIND_SUM      = 0
RED_KIND_SUMSQ    = 1
RED_KIND_MAX      = 2
RED_POST_NONE     = 0
RED_POST_RCP      = 1           # reciprocal
RED_POST_RMS      = 2           # RC(SQ(FA(FM(y, s[SK]), s[SE]))) (VU §6.2)
RED_RSV_W0        = 0xC0E000    # must be 0 (ILLEGAL_OP)
RED_RSV_W1        = 0xFF000000
RED_RSV_W2        = 0xFF000000
RED_RSV_W3        = 0xFFFFFFFF
RED_RSV_W4        = 0xFFFFFFFF
RED_RSV_W5        = 0xFF000000
RED_RSV_W6        = 0xFFFF0000
RED_RSV_W7        = 0xFFFF0000
RED_WORDS         = 0xE7        # words holding a field (w0 always)
# ROPE 0x32 (VU §5.4, owner VU)
ROPE_MODE            = 0xF00       # w0[11:8]
ROPE_MODE_LSB        = 8
ROPE_MODE_W          = 4
ROPE_MODE_WORD       = 0
ROPE_MODE_BIT        = 8
ROPE_FF              = 0x1000      # w0[12] recurrence form: the table holds freq factors
ROPE_FF_LSB          = 12
ROPE_FF_W            = 1
ROPE_FF_WORD         = 0
ROPE_FF_BIT          = 12
ROPE_FS              = 0x2000      # w0[13] recurrence form: multiply by freq_scale in s[SFS]
ROPE_FS_LSB          = 13
ROPE_FS_W            = 1
ROPE_FS_WORD         = 0
ROPE_FS_BIT          = 13
ROPE_ATAB            = 0x4000      # w0[14] A-table form (YaRN): FF, FS and STS must be 0
ROPE_ATAB_LSB        = 14
ROPE_ATAB_W          = 1
ROPE_ATAB_WORD       = 0
ROPE_ATAB_BIT        = 14
ROPE_SM              = 0xF0000     # x16[3:0] s register holding M
ROPE_SM_LSB          = 16
ROPE_SM_W            = 4
ROPE_SM_WORD         = 0
ROPE_SM_BIT          = 16
ROPE_STS             = 0xF00000    # x16[7:4] s register holding TS (recurrence form; 0 with ATAB)
ROPE_STS_LSB         = 20
ROPE_STS_W           = 4
ROPE_STS_WORD        = 0
ROPE_STS_BIT         = 20
ROPE_SFS             = 0xF000000   # x16[11:8] s register holding freq_scale when FS, else 0
ROPE_SFS_LSB         = 24
ROPE_SFS_W           = 4
ROPE_SFS_WORD        = 0
ROPE_SFS_BIT         = 24
ROPE_D               = 0xFFFFFF    # w1[23:0] destination line of token 0
ROPE_D_LSB           = 0
ROPE_D_W             = 24
ROPE_D_WORD          = 1
ROPE_D_BIT           = 32
ROPE_SRC             = 0xFFFFFF    # w2[23:0] source line of token 0
ROPE_SRC_LSB         = 0
ROPE_SRC_W           = 24
ROPE_SRC_WORD        = 2
ROPE_SRC_BIT         = 64
ROPE_POS             = 0xFFFFFF    # w3[23:0] line of the I32 position vector
ROPE_POS_LSB         = 0
ROPE_POS_W           = 24
ROPE_POS_WORD        = 3
ROPE_POS_BIT         = 96
ROPE_TABLE           = 0xFFFFFF    # w4[23:0] line of the f32 table of n_dims/2 entries; 0 when neither FF nor ATAB
ROPE_TABLE_LSB       = 0
ROPE_TABLE_W         = 24
ROPE_TABLE_WORD      = 4
ROPE_TABLE_BIT       = 128
ROPE_HEAD_DIM        = 0xFFFF      # w5[15:0] elements per head
ROPE_HEAD_DIM_LSB    = 0
ROPE_HEAD_DIM_W      = 16
ROPE_HEAD_DIM_WORD   = 5
ROPE_HEAD_DIM_BIT    = 160
ROPE_N_DIMS          = 0xFFFF0000  # w5[31:16] rotated elements per head
ROPE_N_DIMS_LSB      = 16
ROPE_N_DIMS_W        = 16
ROPE_N_DIMS_WORD     = 5
ROPE_N_DIMS_BIT      = 176
ROPE_N_HEADS         = 0xFFFF      # w6[15:0] heads per token
ROPE_N_HEADS_LSB     = 0
ROPE_N_HEADS_W       = 16
ROPE_N_HEADS_WORD    = 6
ROPE_N_HEADS_BIT     = 192
ROPE_T               = 0xFFFF0000  # w6[31:16] tokens
ROPE_T_LSB           = 16
ROPE_T_W             = 16
ROPE_T_WORD          = 6
ROPE_T_BIT           = 208
ROPE_SRC_STRIDE      = 0xFFFF      # w7[15:0] lines between tokens; 0 = H
ROPE_SRC_STRIDE_LSB  = 0
ROPE_SRC_STRIDE_W    = 16
ROPE_SRC_STRIDE_WORD = 7
ROPE_SRC_STRIDE_BIT  = 224
ROPE_D_STRIDE        = 0xFFFF0000  # w7[31:16] lines between tokens; 0 = H
ROPE_D_STRIDE_LSB    = 16
ROPE_D_STRIDE_W      = 16
ROPE_D_STRIDE_WORD   = 7
ROPE_D_STRIDE_BIT    = 240
ROPE_MODE_NORM       = 0           # adjacent pairs
ROPE_MODE_NEOX       = 1           # half-split pairs
ROPE_RSV_W0          = 0xF0008000  # must be 0 (ILLEGAL_OP)
ROPE_RSV_W1          = 0xFF000000
ROPE_RSV_W2          = 0xFF000000
ROPE_RSV_W3          = 0xFF000000
ROPE_RSV_W4          = 0xFF000000
ROPE_RSV_W5          = 0x0
ROPE_RSV_W6          = 0x0
ROPE_RSV_W7          = 0x0
ROPE_WORDS           = 0xFF        # words holding a field (w0 always)
# TOPK 0x33 (VU §5.5, owner VU)
TOPK_K             = 0x1F0000    # x16[4:0] 1..16
TOPK_K_LSB         = 16
TOPK_K_W           = 5
TOPK_K_WORD        = 0
TOPK_K_BIT         = 16
TOPK_D             = 0xFFFFFF    # w1[23:0] output line of row 0: k (index, value) pairs
TOPK_D_LSB         = 0
TOPK_D_W           = 24
TOPK_D_WORD        = 1
TOPK_D_BIT         = 32
TOPK_A             = 0xFFFFFF    # w2[23:0] source line
TOPK_A_LSB         = 0
TOPK_A_W           = 24
TOPK_A_WORD        = 2
TOPK_A_BIT         = 64
TOPK_M             = 0xFFFFFF    # w5[23:0] elements per row
TOPK_M_LSB         = 0
TOPK_M_W           = 24
TOPK_M_WORD        = 5
TOPK_M_BIT         = 160
TOPK_ROWS          = 0xFFFF      # w6[15:0] rows
TOPK_ROWS_LSB      = 0
TOPK_ROWS_W        = 16
TOPK_ROWS_WORD     = 6
TOPK_ROWS_BIT      = 192
TOPK_D_STRIDE      = 0xFFFF0000  # w6[31:16] lines; 0 = ceil(k/8)
TOPK_D_STRIDE_LSB  = 16
TOPK_D_STRIDE_W    = 16
TOPK_D_STRIDE_WORD = 6
TOPK_D_STRIDE_BIT  = 208
TOPK_A_STRIDE      = 0xFFFF      # w7[15:0] lines
TOPK_A_STRIDE_LSB  = 0
TOPK_A_STRIDE_W    = 16
TOPK_A_STRIDE_WORD = 7
TOPK_A_STRIDE_BIT  = 224
TOPK_RSV_W0        = 0xFFE0FF00  # must be 0 (ILLEGAL_OP)
TOPK_RSV_W1        = 0xFF000000
TOPK_RSV_W2        = 0xFF000000
TOPK_RSV_W3        = 0xFFFFFFFF
TOPK_RSV_W4        = 0xFFFFFFFF
TOPK_RSV_W5        = 0xFF000000
TOPK_RSV_W6        = 0x0
TOPK_RSV_W7        = 0xFFFF0000
TOPK_WORDS         = 0xE7        # words holding a field (w0 always)
# DQROW 0x12 (VU §5.6, owner VU)
DQROW_FMT             = 0xF00       # w0[11:8] 2..15 BAD_FMT
DQROW_FMT_LSB         = 8
DQROW_FMT_W           = 4
DQROW_FMT_WORD        = 0
DQROW_FMT_BIT         = 8
DQROW_D               = 0xFFFFFF    # w1[23:0] f32 output line of row 0
DQROW_D_LSB           = 0
DQROW_D_W             = 24
DQROW_D_WORD          = 1
DQROW_D_BIT           = 32
DQROW_SRC             = 0xFFFFFF    # w2[23:0] SPM line of row 0's repacked bytes
DQROW_SRC_LSB         = 0
DQROW_SRC_W           = 24
DQROW_SRC_WORD        = 2
DQROW_SRC_BIT         = 64
DQROW_K               = 0xFFFFFFFF  # w5 elements per row (K_RANGE checks)
DQROW_K_LSB           = 0
DQROW_K_W             = 32
DQROW_K_WORD          = 5
DQROW_K_BIT           = 160
DQROW_ROWS            = 0xFFFF      # w6[15:0] rows
DQROW_ROWS_LSB        = 0
DQROW_ROWS_W          = 16
DQROW_ROWS_WORD       = 6
DQROW_ROWS_BIT        = 192
DQROW_SRC_STRIDE      = 0xFFFF      # w7[15:0] lines; 0 = ROW_BEATS(fmt, K)
DQROW_SRC_STRIDE_LSB  = 0
DQROW_SRC_STRIDE_W    = 16
DQROW_SRC_STRIDE_WORD = 7
DQROW_SRC_STRIDE_BIT  = 224
DQROW_D_STRIDE        = 0xFFFF0000  # w7[31:16] lines; 0 = L(K)
DQROW_D_STRIDE_LSB    = 16
DQROW_D_STRIDE_W      = 16
DQROW_D_STRIDE_WORD   = 7
DQROW_D_STRIDE_BIT    = 240
# DQROW_FMT takes the FMT_* values
DQROW_RSV_W0 = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
DQROW_RSV_W1 = 0xFF000000
DQROW_RSV_W2 = 0xFF000000
DQROW_RSV_W3 = 0xFFFFFFFF
DQROW_RSV_W4 = 0xFFFFFFFF
DQROW_RSV_W5 = 0x0
DQROW_RSV_W6 = 0xFFFF0000
DQROW_RSV_W7 = 0x0
DQROW_WORDS  = 0xE7        # words holding a field (w0 always)
# VMASK 0x38 (VU §5.7, owner VU)
VMASK_SUB           = 0xF00       # w0[11:8]
VMASK_SUB_LSB       = 8
VMASK_SUB_W         = 4
VMASK_SUB_WORD      = 0
VMASK_SUB_BIT       = 8
VMASK_X             = 0xFFFFFF    # w1[23:0] operand line (in place)
VMASK_X_LSB         = 0
VMASK_X_W           = 24
VMASK_X_WORD        = 1
VMASK_X_BIT         = 32
VMASK_WIN           = 0xFFFFFFFF  # w2 window size W, u32; 0 = no window
VMASK_WIN_LSB       = 0
VMASK_WIN_W         = 32
VMASK_WIN_WORD      = 2
VMASK_WIN_BIT       = 64
VMASK_K0            = 0xFFFFFFFF  # w3 i32 position of column 0
VMASK_K0_LSB        = 0
VMASK_K0_W          = 32
VMASK_K0_WORD       = 3
VMASK_K0_BIT        = 96
VMASK_Q0            = 0xFFFFFFFF  # w4 i32 query position of row group 0
VMASK_Q0_LSB        = 0
VMASK_Q0_W          = 32
VMASK_Q0_WORD       = 4
VMASK_Q0_BIT        = 128
VMASK_M             = 0xFFFFFF    # w5[23:0] columns
VMASK_M_LSB         = 0
VMASK_M_W           = 24
VMASK_M_WORD        = 5
VMASK_M_BIT         = 160
VMASK_ROWS          = 0xFFFF      # w6[15:0] rows
VMASK_ROWS_LSB      = 0
VMASK_ROWS_W        = 16
VMASK_ROWS_WORD     = 6
VMASK_ROWS_BIT      = 192
VMASK_G             = 0xFFFF0000  # w6[31:16] rows per query position (>= 1)
VMASK_G_LSB         = 16
VMASK_G_W           = 16
VMASK_G_WORD        = 6
VMASK_G_BIT         = 208
VMASK_X_STRIDE      = 0xFFFF      # w7[15:0] lines
VMASK_X_STRIDE_LSB  = 0
VMASK_X_STRIDE_W    = 16
VMASK_X_STRIDE_WORD = 7
VMASK_X_STRIDE_BIT  = 224
VMASK_P             = 0xFFFF0000  # w7[31:16] row period; 0 = none
VMASK_P_LSB         = 16
VMASK_P_W           = 16
VMASK_P_WORD        = 7
VMASK_P_BIT         = 240
VMASK_SUB_WINDOW    = 0           # position window; others ILLEGAL_OP
VMASK_RSV_W0        = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
VMASK_RSV_W1        = 0xFF000000
VMASK_RSV_W2        = 0x0
VMASK_RSV_W3        = 0x0
VMASK_RSV_W4        = 0x0
VMASK_RSV_W5        = 0xFF000000
VMASK_RSV_W6        = 0x0
VMASK_RSV_W7        = 0x0
VMASK_WORDS         = 0xFF        # words holding a field (w0 always)
# SOP 0x40 (VU §5.8, owner VU)
SOP_SUB       = 0xF00       # w0[11:8] function (SOP_FUNCS)
SOP_SUB_LSB   = 8
SOP_SUB_W     = 4
SOP_SUB_WORD  = 0
SOP_SUB_BIT   = 8
SOP_SD        = 0xF         # w1[3:0] destination s register
SOP_SD_LSB    = 0
SOP_SD_W      = 4
SOP_SD_WORD   = 1
SOP_SD_BIT    = 32
SOP_SA        = 0xF0        # w1[7:4] source s register a
SOP_SA_LSB    = 4
SOP_SA_W      = 4
SOP_SA_WORD   = 1
SOP_SA_BIT    = 36
SOP_SB        = 0xF00       # w1[11:8] source s register b
SOP_SB_LSB    = 8
SOP_SB_W      = 4
SOP_SB_WORD   = 1
SOP_SB_BIT    = 40
SOP_IMM       = 0xFFFFFFFF  # w2 f32 bits
SOP_IMM_LSB   = 0
SOP_IMM_W     = 32
SOP_IMM_WORD  = 2
SOP_IMM_BIT   = 64
SOP_ADDR      = 0xFFFFFFF   # w3[27:0] SPM word address (LDS / STS)
SOP_ADDR_LSB  = 0
SOP_ADDR_W    = 28
SOP_ADDR_WORD = 3
SOP_ADDR_BIT  = 96
SOP_SUB_MOVI  = 0           # sd <- imm (raw) (uses SD, IMM)
SOP_SUB_ADD   = 1           # sd <- FA(sa, sb) (uses SD, SA, SB)
SOP_SUB_MUL   = 2           # sd <- FM(sa, sb) (uses SD, SA, SB)
SOP_SUB_ADDI  = 3           # sd <- FA(sa, imm) (uses SD, SA, IMM)
SOP_SUB_MULI  = 4           # sd <- FM(sa, imm) (uses SD, SA, IMM)
SOP_SUB_SQRT  = 5           # sd <- SQ(sa) (uses SD, SA)
SOP_SUB_RECIP = 6           # sd <- RC(sa) (uses SD, SA)
SOP_SUB_LDS   = 7           # sd <- SPM word (raw) (uses SD, ADDR)
SOP_SUB_STS   = 8           # SPM word <- sa (raw; exactly 4 bytes) (uses SA, ADDR)
SOP_SUB_SUB   = 9           # sd <- FS(sa, sb) (uses SD, SA, SB)
SOP_SUB_DIV   = 10          # sd <- FD(sa, sb) (uses SD, SA, SB)
SOP_SUB_RSQRT = 11          # sd <- RC(SQ(sa)), two roundings (uses SD, SA)
SOP_SUB_MAX   = 12          # sd <- MAXT(sa, sb) (uses SD, SA, SB)
SOP_SUB_MIN   = 13          # sd <- MINT(sa, sb) (uses SD, SA, SB)
SOP_SUB_EXP   = 14          # sd <- EXPK(sa, P1) (uses SD, SA)
SOP_RSV_W0    = 0xFFFFF000  # must be 0 (ILLEGAL_OP)
SOP_RSV_W1    = 0xFFFFF000
SOP_RSV_W2    = 0x0
SOP_RSV_W3    = 0xF0000000
SOP_RSV_W4    = 0xFFFFFFFF
SOP_RSV_W5    = 0xFFFFFFFF
SOP_RSV_W6    = 0xFFFFFFFF
SOP_RSV_W7    = 0xFFFFFFFF
SOP_WORDS     = 15          # words holding a field (w0 always)

VOP_FUNC_N = 25
VOP_READS_A, VOP_READS_B, VOP_READS_C = 1, 2, 4
# VOP function code -> (name, operands read (VOP_READS_*), half rate)
VOP_FUNCS = {
    0x00: ('ADD', 3, 0), 0x01: ('SUB', 3, 0), 0x02: ('MUL', 3, 0), 0x03: ('MAX', 3, 0),
    0x04: ('MULC', 5, 0), 0x05: ('ADDC', 5, 0), 0x06: ('SILU', 1, 1), 0x07: ('SWIGLU', 3, 1),
    0x08: ('EXP', 1, 0), 0x09: ('COPY', 1, 0), 0x0A: ('MIN', 3, 0), 0x0B: ('SUBC', 5, 0),
    0x0C: ('MULCB', 7, 0), 0x0D: ('MULCADD', 7, 0), 0x0E: ('ADDMULC', 7, 0), 0x0F: ('EXPSUB', 3,
    0), 0x10: ('EXPSUBC', 5, 0), 0x11: ('SIGMOID', 1, 1), 0x12: ('GELU', 1, 1), 0x13: ('GEGLU', 3,
    1), 0x14: ('TANH', 1, 1), 0x15: ('RECIP', 1, 1), 0x16: ('SETC', 4, 0), 0x17: ('GELU16', 1, 1),
    0x18: ('GEGLU16', 3, 1),
}
SOP_SUB_N = 15
SOP_USES_SD, SOP_USES_SA, SOP_USES_SB, SOP_USES_IMM, SOP_USES_ADDR = 1, 2, 4, 8, 16
# SOP sub code -> (name, fields used (SOP_USES_*))
SOP_FUNCS = {
    0: ('MOVI', 9), 1: ('ADD', 7), 2: ('MUL', 7), 3: ('ADDI', 11), 4: ('MULI', 11), 5: ('SQRT', 3),
    6: ('RECIP', 3), 7: ('LDS', 17), 8: ('STS', 18), 9: ('SUB', 7), 10: ('DIV', 7), 11: ('RSQRT',
    3), 12: ('MAX', 7), 13: ('MIN', 7), 14: ('EXP', 3),
}


# -------------------------------------------------------------------------------------------------
# AT encodings (AT §5; VU §5.11 restates the vector-op encodings)
# -------------------------------------------------------------------------------------------------
# KVST 0x14 (AT §5.1, owner AT)
KVST_KV              = 0xF00       # w0[11:8] plane
KVST_KV_LSB          = 8
KVST_KV_W            = 4
KVST_KV_WORD         = 0
KVST_KV_BIT          = 8
KVST_N_ROWS_M1       = 0x3F0000    # x16[5:0] n_rows - 1 (n_rows 1..64)
KVST_N_ROWS_M1_LSB   = 16
KVST_N_ROWS_M1_W     = 6
KVST_N_ROWS_M1_WORD  = 0
KVST_N_ROWS_M1_BIT   = 16
KVST_SRC             = 0xFFFFFFFF  # w1 first QA(A32, E) vector (SPM line)
KVST_SRC_LSB         = 0
KVST_SRC_W           = 32
KVST_SRC_WORD        = 1
KVST_SRC_BIT         = 32
KVST_BASE            = 0xFFFFFFFF  # w2 plane base B (beat, every channel)
KVST_BASE_LSB        = 0
KVST_BASE_W          = 32
KVST_BASE_WORD       = 2
KVST_BASE_BIT        = 64
KVST_E64             = 0x1FF       # w3[8:0] E / 64
KVST_E64_LSB         = 0
KVST_E64_W           = 9
KVST_E64_WORD        = 3
KVST_E64_BIT         = 96
KVST_LG              = 0xFFFF0000  # w3[31:16] local groups per channel
KVST_LG_LSB          = 16
KVST_LG_W            = 16
KVST_LG_WORD         = 3
KVST_LG_BIT          = 112
KVST_C0              = 0xFFFFFFFF  # w4 first cell
KVST_C0_LSB          = 0
KVST_C0_W            = 32
KVST_C0_WORD         = 4
KVST_C0_BIT          = 128
KVST_SRC_STRIDE      = 0xFFFFFFFF  # w5 lines; 0 = QA_LINES(A32, E)
KVST_SRC_STRIDE_LSB  = 0
KVST_SRC_STRIDE_W    = 32
KVST_SRC_STRIDE_WORD = 5
KVST_SRC_STRIDE_BIT  = 160
KVST_KV_K            = 0           # K plane
KVST_KV_V            = 1           # V plane
KVST_RSV_W0          = 0xFFC0F000  # must be 0 (ILLEGAL_OP)
KVST_RSV_W1          = 0x0
KVST_RSV_W2          = 0x0
KVST_RSV_W3          = 0xFE00
KVST_RSV_W4          = 0x0
KVST_RSV_W5          = 0x0
KVST_RSV_W6          = 0xFFFFFFFF
KVST_RSV_W7          = 0xFFFFFFFF
KVST_WORDS           = 0x3F        # words holding a field (w0 always)
# ATTQK 0x21 (AT §5.2, owner AT)
ATTQK_FMT           = 0xF00       # w0[11:8] KV format
ATTQK_FMT_LSB       = 8
ATTQK_FMT_W         = 4
ATTQK_FMT_WORD      = 0
ATTQK_FMT_BIT       = 8
ATTQK_CAUSAL        = 0x1000      # w0[12]
ATTQK_CAUSAL_LSB    = 12
ATTQK_CAUSAL_W      = 1
ATTQK_CAUSAL_WORD   = 0
ATTQK_CAUSAL_BIT    = 12
ATTQK_SLIDE         = 0x2000      # w0[13]
ATTQK_SLIDE_LSB     = 13
ATTQK_SLIDE_W       = 1
ATTQK_SLIDE_WORD    = 0
ATTQK_SLIDE_BIT     = 13
ATTQK_MAXOUT        = 0x4000      # w0[14] write row maxima at MDST
ATTQK_MAXOUT_LSB    = 14
ATTQK_MAXOUT_W      = 1
ATTQK_MAXOUT_WORD   = 0
ATTQK_MAXOUT_BIT    = 14
ATTQK_G_M1          = 0x3F0000    # x16[5:0] G - 1 (heads per KV head)
ATTQK_G_M1_LSB      = 16
ATTQK_G_M1_W        = 6
ATTQK_G_M1_WORD     = 0
ATTQK_G_M1_BIT      = 16
ATTQK_HG0           = 0xFC00000   # x16[11:6] first head of the pass within each KV-head group
ATTQK_HG0_LSB       = 22
ATTQK_HG0_W         = 6
ATTQK_HG0_WORD      = 0
ATTQK_HG0_BIT       = 22
ATTQK_DST           = 0xFFFFFFFF  # w1 score rows, row r at DST + r x RSTRIDE
ATTQK_DST_LSB       = 0
ATTQK_DST_W         = 32
ATTQK_DST_WORD      = 1
ATTQK_DST_BIT       = 32
ATTQK_X             = 0xFFFFFFFF  # w2 T_q QA(A32, n_head x D) vectors
ATTQK_X_LSB         = 0
ATTQK_X_W           = 32
ATTQK_X_WORD        = 2
ATTQK_X_BIT         = 64
ATTQK_BASE          = 0xFFFFFFFF  # w3 K plane base B
ATTQK_BASE_LSB      = 0
ATTQK_BASE_W        = 32
ATTQK_BASE_WORD     = 3
ATTQK_BASE_BIT      = 96
ATTQK_E64           = 0x1FF       # w4[8:0] E / 64 (1..512)
ATTQK_E64_LSB       = 0
ATTQK_E64_W         = 9
ATTQK_E64_WORD      = 4
ATTQK_E64_BIT       = 128
ATTQK_DCODE         = 0x600       # w4[10:9] D = 64 x 2^dcode
ATTQK_DCODE_LSB     = 9
ATTQK_DCODE_W       = 2
ATTQK_DCODE_WORD    = 4
ATTQK_DCODE_BIT     = 137
ATTQK_TQ_M1         = 0x3800      # w4[13:11] T_q - 1
ATTQK_TQ_M1_LSB     = 11
ATTQK_TQ_M1_W       = 3
ATTQK_TQ_M1_WORD    = 4
ATTQK_TQ_M1_BIT     = 139
ATTQK_GS_M1         = 0x1C000     # w4[16:14] GS - 1
ATTQK_GS_M1_LSB     = 14
ATTQK_GS_M1_W       = 3
ATTQK_GS_M1_WORD    = 4
ATTQK_GS_M1_BIT     = 142
ATTQK_RSTRIDE       = 0xFFFE0000  # w4[31:17] lines; 0 = n_groups
ATTQK_RSTRIDE_LSB   = 17
ATTQK_RSTRIDE_W     = 15
ATTQK_RSTRIDE_WORD  = 4
ATTQK_RSTRIDE_BIT   = 145
ATTQK_U0            = 0xFFFF      # w5[15:0] first group
ATTQK_U0_LSB        = 0
ATTQK_U0_W          = 16
ATTQK_U0_WORD       = 5
ATTQK_U0_BIT        = 160
ATTQK_N_GROUPS      = 0xFFFF0000  # w5[31:16] 1..4095
ATTQK_N_GROUPS_LSB  = 16
ATTQK_N_GROUPS_W    = 16
ATTQK_N_GROUPS_WORD = 5
ATTQK_N_GROUPS_BIT  = 176
ATTQK_C_LO          = 0xFFFF      # w6[15:0] cell bound relative to 16 u0
ATTQK_C_LO_LSB      = 0
ATTQK_C_LO_W        = 16
ATTQK_C_LO_WORD     = 6
ATTQK_C_LO_BIT      = 192
ATTQK_C_HI          = 0xFFFF0000  # w6[31:16] cell bound relative to 16 u0
ATTQK_C_HI_LSB      = 16
ATTQK_C_HI_W        = 16
ATTQK_C_HI_WORD     = 6
ATTQK_C_HI_BIT      = 208
ATTQK_MDST          = 0xFFFFFFFF  # w7 row-maximum vector line; 0 unless MAXOUT
ATTQK_MDST_LSB      = 0
ATTQK_MDST_W        = 32
ATTQK_MDST_WORD     = 7
ATTQK_MDST_BIT      = 224
# ATTQK_FMT takes the KVFMT_* values
# ATTQK_DCODE takes the DCODE_* values
ATTQK_RSV_W0 = 0xF0008000  # must be 0 (ILLEGAL_OP)
ATTQK_RSV_W1 = 0x0
ATTQK_RSV_W2 = 0x0
ATTQK_RSV_W3 = 0x0
ATTQK_RSV_W4 = 0x0
ATTQK_RSV_W5 = 0x0
ATTQK_RSV_W6 = 0x0
ATTQK_RSV_W7 = 0x0
ATTQK_WORDS  = 0xFF        # words holding a field (w0 always)
# ATTPV 0x22 (AT §5.3, owner AT)
ATTPV_FMT           = 0xF00       # w0[11:8] KV format
ATTPV_FMT_LSB       = 8
ATTPV_FMT_W         = 4
ATTPV_FMT_WORD      = 0
ATTPV_FMT_BIT       = 8
ATTPV_LSUM          = 0x1000      # w0[12] also row sums
ATTPV_LSUM_LSB      = 12
ATTPV_LSUM_W        = 1
ATTPV_LSUM_WORD     = 0
ATTPV_LSUM_BIT      = 12
ATTPV_G_M1          = 0x3F0000    # x16[5:0] G - 1
ATTPV_G_M1_LSB      = 16
ATTPV_G_M1_W        = 6
ATTPV_G_M1_WORD     = 0
ATTPV_G_M1_BIT      = 16
ATTPV_HG0           = 0xFC00000   # x16[11:6] first head of the pass
ATTPV_HG0_LSB       = 22
ATTPV_HG0_W         = 6
ATTPV_HG0_WORD      = 0
ATTPV_HG0_BIT       = 22
ATTPV_DST           = 0xFFFFFFFF  # w1 channel partials
ATTPV_DST_LSB       = 0
ATTPV_DST_W         = 32
ATTPV_DST_WORD      = 1
ATTPV_DST_BIT       = 32
ATTPV_P             = 0xFFFFFFFF  # w2 weight rows, row r at P + r x PSTRIDE
ATTPV_P_LSB         = 0
ATTPV_P_W           = 32
ATTPV_P_WORD        = 2
ATTPV_P_BIT         = 64
ATTPV_BASE          = 0xFFFFFFFF  # w3 V plane base B
ATTPV_BASE_LSB      = 0
ATTPV_BASE_W        = 32
ATTPV_BASE_WORD     = 3
ATTPV_BASE_BIT      = 96
ATTPV_E64           = 0x1FF       # w4[8:0] E / 64
ATTPV_E64_LSB       = 0
ATTPV_E64_W         = 9
ATTPV_E64_WORD      = 4
ATTPV_E64_BIT       = 128
ATTPV_DCODE         = 0x600       # w4[10:9]
ATTPV_DCODE_LSB     = 9
ATTPV_DCODE_W       = 2
ATTPV_DCODE_WORD    = 4
ATTPV_DCODE_BIT     = 137
ATTPV_TQ_M1         = 0x3800      # w4[13:11] T_q - 1
ATTPV_TQ_M1_LSB     = 11
ATTPV_TQ_M1_W       = 3
ATTPV_TQ_M1_WORD    = 4
ATTPV_TQ_M1_BIT     = 139
ATTPV_GS_M1         = 0x1C000     # w4[16:14] GS - 1
ATTPV_GS_M1_LSB     = 14
ATTPV_GS_M1_W       = 3
ATTPV_GS_M1_WORD    = 4
ATTPV_GS_M1_BIT     = 142
ATTPV_PSTRIDE       = 0xFFFE0000  # w4[31:17] lines; 0 = n_groups
ATTPV_PSTRIDE_LSB   = 17
ATTPV_PSTRIDE_W     = 15
ATTPV_PSTRIDE_WORD  = 4
ATTPV_PSTRIDE_BIT   = 145
ATTPV_U0            = 0xFFFF      # w5[15:0] first group
ATTPV_U0_LSB        = 0
ATTPV_U0_W          = 16
ATTPV_U0_WORD       = 5
ATTPV_U0_BIT        = 160
ATTPV_N_GROUPS      = 0xFFFF0000  # w5[31:16]
ATTPV_N_GROUPS_LSB  = 16
ATTPV_N_GROUPS_W    = 16
ATTPV_N_GROUPS_WORD = 5
ATTPV_N_GROUPS_BIT  = 176
ATTPV_LG            = 0xFFFF      # w6[15:0] local groups per channel
ATTPV_LG_LSB        = 0
ATTPV_LG_W          = 16
ATTPV_LG_WORD       = 6
ATTPV_LG_BIT        = 192
ATTPV_CSTRIDE       = 0xFFFFFFFF  # w7 lines; 0 = R x D / 16
ATTPV_CSTRIDE_LSB   = 0
ATTPV_CSTRIDE_W     = 32
ATTPV_CSTRIDE_WORD  = 7
ATTPV_CSTRIDE_BIT   = 224
# ATTPV_FMT takes the KVFMT_* values
# ATTPV_DCODE takes the DCODE_* values
ATTPV_RSV_W0 = 0xF000E000  # must be 0 (ILLEGAL_OP)
ATTPV_RSV_W1 = 0x0
ATTPV_RSV_W2 = 0x0
ATTPV_RSV_W3 = 0x0
ATTPV_RSV_W4 = 0x0
ATTPV_RSV_W5 = 0x0
ATTPV_RSV_W6 = 0xFFFF0000
ATTPV_RSV_W7 = 0x0
ATTPV_WORDS  = 0xFF        # words holding a field (w0 always)
# ASMAX 0x34 (AT §5.4, VU §5.11, owner AT)
ASMAX_DST          = 0xFFFFFFFF  # w1
ASMAX_DST_LSB      = 0
ASMAX_DST_W        = 32
ASMAX_DST_WORD     = 1
ASMAX_DST_BIT      = 32
ASMAX_SRC          = 0xFFFFFFFF  # w2 may equal DST
ASMAX_SRC_LSB      = 0
ASMAX_SRC_W        = 32
ASMAX_SRC_WORD     = 2
ASMAX_SRC_BIT      = 64
ASMAX_MVEC         = 0xFFFFFFFF  # w3 row maxima, word r
ASMAX_MVEC_LSB     = 0
ASMAX_MVEC_W       = 32
ASMAX_MVEC_WORD    = 3
ASMAX_MVEC_BIT     = 96
ASMAX_N_LINES      = 0xFFFF      # w4[15:0] 1..4095
ASMAX_N_LINES_LSB  = 0
ASMAX_N_LINES_W    = 16
ASMAX_N_LINES_WORD = 4
ASMAX_N_LINES_BIT  = 128
ASMAX_STRIDE       = 0xFFFF0000  # w4[31:16] 0 = n_lines
ASMAX_STRIDE_LSB   = 16
ASMAX_STRIDE_W     = 16
ASMAX_STRIDE_WORD  = 4
ASMAX_STRIDE_BIT   = 144
ASMAX_R            = 0xFFFFFFFF  # w5 1..1024
ASMAX_R_LSB        = 0
ASMAX_R_W          = 32
ASMAX_R_WORD       = 5
ASMAX_R_BIT        = 160
ASMAX_SCALE        = 0xFFFFFFFF  # w6 f32, finite, normal, > 0
ASMAX_SCALE_LSB    = 0
ASMAX_SCALE_W      = 32
ASMAX_SCALE_WORD   = 6
ASMAX_SCALE_BIT    = 192
ASMAX_RSV_W0       = 0xFFFFFF00  # must be 0 (ILLEGAL_OP)
ASMAX_RSV_W1       = 0x0
ASMAX_RSV_W2       = 0x0
ASMAX_RSV_W3       = 0x0
ASMAX_RSV_W4       = 0x0
ASMAX_RSV_W5       = 0x0
ASMAX_RSV_W6       = 0x0
ASMAX_RSV_W7       = 0xFFFFFFFF
ASMAX_WORDS        = 0x7F        # words holding a field (w0 always)
# AMASK 0x35 (AT §5.5, VU §5.11, owner AT)
AMASK_DST          = 0xFFFFFFFF  # w1
AMASK_DST_LSB      = 0
AMASK_DST_W        = 32
AMASK_DST_WORD     = 1
AMASK_DST_BIT      = 32
AMASK_SRC          = 0xFFFFFFFF  # w2 rows at STRIDE
AMASK_SRC_LSB      = 0
AMASK_SRC_W        = 32
AMASK_SRC_WORD     = 2
AMASK_SRC_BIT      = 64
AMASK_MASK         = 0xFFFFFFFF  # w3 f32 mask rows
AMASK_MASK_LSB     = 0
AMASK_MASK_W       = 32
AMASK_MASK_WORD    = 3
AMASK_MASK_BIT     = 96
AMASK_N_LINES      = 0xFFFF      # w4[15:0]
AMASK_N_LINES_LSB  = 0
AMASK_N_LINES_W    = 16
AMASK_N_LINES_WORD = 4
AMASK_N_LINES_BIT  = 128
AMASK_STRIDE       = 0xFFFF0000  # w4[31:16]
AMASK_STRIDE_LSB   = 16
AMASK_STRIDE_W     = 16
AMASK_STRIDE_WORD  = 4
AMASK_STRIDE_BIT   = 144
AMASK_R            = 0xFFFF      # w5[15:0]
AMASK_R_LSB        = 0
AMASK_R_W          = 16
AMASK_R_WORD       = 5
AMASK_R_BIT        = 160
AMASK_RPQ          = 0xFFFF0000  # w5[31:16] rows per query (= n_head)
AMASK_RPQ_LSB      = 16
AMASK_RPQ_W        = 16
AMASK_RPQ_WORD     = 5
AMASK_RPQ_BIT      = 176
AMASK_MSTRIDE      = 0xFFFFFFFF  # w6 lines between mask rows
AMASK_MSTRIDE_LSB  = 0
AMASK_MSTRIDE_W    = 32
AMASK_MSTRIDE_WORD = 6
AMASK_MSTRIDE_BIT  = 192
AMASK_MDST         = 0xFFFFFFFF  # w7 row maxima out
AMASK_MDST_LSB     = 0
AMASK_MDST_W       = 32
AMASK_MDST_WORD    = 7
AMASK_MDST_BIT     = 224
AMASK_RSV_W0       = 0xFFFFFF00  # must be 0 (ILLEGAL_OP)
AMASK_RSV_W1       = 0x0
AMASK_RSV_W2       = 0x0
AMASK_RSV_W3       = 0x0
AMASK_RSV_W4       = 0x0
AMASK_RSV_W5       = 0x0
AMASK_RSV_W6       = 0x0
AMASK_RSV_W7       = 0x0
AMASK_WORDS        = 0xFF        # words holding a field (w0 always)
# AMERGE 0x36 (AT §5.6, VU §5.11, owner AT)
AMERGE_INIT         = 0x1000      # w0[12] copy instead of merge
AMERGE_INIT_LSB     = 12
AMERGE_INIT_W       = 1
AMERGE_INIT_WORD    = 0
AMERGE_INIT_BIT     = 12
AMERGE_ORUN         = 0xFFFFFFFF  # w1 R rows of D/16 lines
AMERGE_ORUN_LSB     = 0
AMERGE_ORUN_W       = 32
AMERGE_ORUN_WORD    = 1
AMERGE_ORUN_BIT     = 32
AMERGE_PARTS        = 0xFFFFFFFF  # w2 an ATTPV destination written with LSUM
AMERGE_PARTS_LSB    = 0
AMERGE_PARTS_W      = 32
AMERGE_PARTS_WORD   = 2
AMERGE_PARTS_BIT    = 64
AMERGE_MK           = 0xFFFFFFFF  # w3 chunk row maxima
AMERGE_MK_LSB       = 0
AMERGE_MK_W         = 32
AMERGE_MK_WORD      = 3
AMERGE_MK_BIT       = 96
AMERGE_ST           = 0xFFFFFFFF  # w4 m_run at ST, l_run at ST + ceil(R/16)
AMERGE_ST_LSB       = 0
AMERGE_ST_W         = 32
AMERGE_ST_WORD      = 4
AMERGE_ST_BIT       = 128
AMERGE_R            = 0xFFFF      # w5[15:0]
AMERGE_R_LSB        = 0
AMERGE_R_W          = 16
AMERGE_R_WORD       = 5
AMERGE_R_BIT        = 160
AMERGE_DCODE        = 0x30000     # w5[17:16]
AMERGE_DCODE_LSB    = 16
AMERGE_DCODE_W      = 2
AMERGE_DCODE_WORD   = 5
AMERGE_DCODE_BIT    = 176
AMERGE_SCALE        = 0xFFFFFFFF  # w6 f32, as ASMAX
AMERGE_SCALE_LSB    = 0
AMERGE_SCALE_W      = 32
AMERGE_SCALE_WORD   = 6
AMERGE_SCALE_BIT    = 192
AMERGE_CSTRIDE      = 0xFFFFFFFF  # w7 as the producing ATTPV
AMERGE_CSTRIDE_LSB  = 0
AMERGE_CSTRIDE_W    = 32
AMERGE_CSTRIDE_WORD = 7
AMERGE_CSTRIDE_BIT  = 224
# AMERGE_DCODE takes the DCODE_* values
AMERGE_RSV_W0 = 0xFFFFEF00  # must be 0 (ILLEGAL_OP)
AMERGE_RSV_W1 = 0x0
AMERGE_RSV_W2 = 0x0
AMERGE_RSV_W3 = 0x0
AMERGE_RSV_W4 = 0x0
AMERGE_RSV_W5 = 0xFFFC0000
AMERGE_RSV_W6 = 0x0
AMERGE_RSV_W7 = 0x0
AMERGE_WORDS  = 0xFF        # words holding a field (w0 always)
# ANORM 0x37 (AT §5.7, VU §5.11, owner AT)
ANORM_OUT        = 0xFFFFFFFF  # w1 row r at OUT + r x D/16
ANORM_OUT_LSB    = 0
ANORM_OUT_W      = 32
ANORM_OUT_WORD   = 1
ANORM_OUT_BIT    = 32
ANORM_ORUN       = 0xFFFFFFFF  # w2
ANORM_ORUN_LSB   = 0
ANORM_ORUN_W     = 32
ANORM_ORUN_WORD  = 2
ANORM_ORUN_BIT   = 64
ANORM_L_RUN      = 0xFFFFFFFF  # w3
ANORM_L_RUN_LSB  = 0
ANORM_L_RUN_W    = 32
ANORM_L_RUN_WORD = 3
ANORM_L_RUN_BIT  = 96
ANORM_R          = 0xFFFF      # w4[15:0]
ANORM_R_LSB      = 0
ANORM_R_W        = 16
ANORM_R_WORD     = 4
ANORM_R_BIT      = 128
ANORM_DCODE      = 0x30000     # w4[17:16]
ANORM_DCODE_LSB  = 16
ANORM_DCODE_W    = 2
ANORM_DCODE_WORD = 4
ANORM_DCODE_BIT  = 144
# ANORM_DCODE takes the DCODE_* values
ANORM_RSV_W0 = 0xFFFFFF00  # must be 0 (ILLEGAL_OP)
ANORM_RSV_W1 = 0x0
ANORM_RSV_W2 = 0x0
ANORM_RSV_W3 = 0x0
ANORM_RSV_W4 = 0xFFFC0000
ANORM_RSV_W5 = 0xFFFFFFFF
ANORM_RSV_W6 = 0xFFFFFFFF
ANORM_RSV_W7 = 0xFFFFFFFF
ANORM_WORDS  = 0x1F        # words holding a field (w0 always)

# -------------------------------------------------------------------------------------------------
# Structured tables
# -------------------------------------------------------------------------------------------------
LAYOUTS = {   # layout -> field -> (lsb, width)
    'W0': {'OPCODE': (0, 8), 'SUB': (8, 4), 'FLAGS': (12, 4), 'X16': (16, 16)},
    'VERSION': {'MINOR': (0, 8), 'MAJOR': (8, 8)},
    'CFG0': {'NUM_CH': (0, 8), 'LANES': (8, 12), 'T_MAX': (20, 4)},
    'CFG2': {'VL': (0, 8), 'NSREG': (8, 8), 'K_MAX_256': (16, 16)},
    'CAPS': {'VU': (0, 1), 'TOPK': (3, 1), 'ATT': (5, 1), 'DQROW': (6, 1), 'VMASK': (7, 1), 'VU_RSVD': (8, 4), 'AT_RSVD': (12, 4)},
    'CFG3': {'VU_FRONT': (0, 5), 'VU_BACK': (5, 5), 'PCAP16': (16, 16)},
    'CTL_ID': {'VERSION': (0, 8), 'DM': (8, 1), 'REC': (9, 1), 'BRIDGE': (10, 1), 'LOG': (11, 1), 'WDOG': (12, 1), 'SBA': (13, 1), 'SPM_LB': (14, 1)},
    'CTRL': {'RV_RUN': (0, 1), 'HOST_HOLD': (1, 1), 'HOLD_ON_HALT': (2, 1), 'IRQ_EN': (3, 1), 'QABORT': (8, 1), 'QCLEAR': (9, 1), 'SOFT_RESET': (10, 1)},
    'STATUS': {'RV_RUNNING': (0, 1), 'RV_HALTED': (1, 1), 'RV_SLEEP': (2, 1), 'RV_CLK_OK': (3, 1), 'RV_RESET_DONE': (4, 1), 'ACTIVE': (5, 1), 'QIDLE': (6, 1), 'QQUIESCENT': (7, 1), 'QERR': (8, 1), 'SR_BUSY': (9, 1), 'LCQ_PENDING': (10, 1), 'LOG_PENDING': (11, 1), 'PLAT_ERR': (12, 1), 'HOLD_SRC': (13, 3), 'LOAD_REFUSED': (16, 1), 'LOAD_TIMEOUT': (17, 1), 'DMI_TIMEOUT': (18, 1), 'QCLEAR_REFUSED': (19, 1), 'PUSH_TIMEOUT': (20, 1), 'BRIDGE_TIMEOUT': (21, 1), 'LQ_OVERFLOW': (22, 1), 'RING_RD_BAD': (23, 1), 'LOG_DROP': (24, 1), 'UNIT_HUNG': (25, 1), 'SR_RV_TIMEOUT': (26, 1), 'RV_RUN_REFUSED': (27, 1)},
    'HOLD_SRC': {'HOST': (0, 1), 'RV': (1, 1), 'HALT': (2, 1), 'DB': (3, 1)},
    'RV_BOOT': {'ADDR': (8, 24)},
    'RV_CFG': {'ITCM_KIB': (0, 16), 'DTCM_KIB': (16, 16)},
    'WDOG_CFG': {'L': (0, 5)},
    'RT_STATE': {'PHASE': (0, 8), 'KERNEL': (16, 16)},
    'LQ_CFG': {'LQ_DEPTH': (0, 8), 'LCQ_DEPTH': (8, 8), 'LOG_DEPTH': (16, 12), 'STOP_VALID': (30, 1), 'CANCEL_VALID': (31, 1)},
    'PERF_INFO': {'COUNT': (0, 8), 'X': (8, 8)},
    'Q_DEPTH': {'DMA': (0, 5), 'QACT': (5, 5), 'MVM': (10, 5), 'VU': (15, 5), 'ATT': (20, 5), 'SYNC': (25, 5)},
    'LOG_CFG': {'ENABLE': (0, 1), 'BLOCK': (1, 1), 'LEVEL': (4, 4)},
    'LOAD_CTRL': {'VERIFY': (0, 1)},
    'LOAD_STATUS': {'VERIFY_BUSY': (0, 1)},
    'Q_STATE': {'ENTRIES': (0, 5), 'RUNNING': (5, 1), 'DEP_WAIT': (6, 1), 'HELD': (7, 1), 'ERR': (8, 1), 'HUNG': (9, 1)},
    'Q_FIFO': {'ENTRIES': (0, 8)},
    'ADAPTER_CTRL': {'MEM_EN': (0, 1)},
    'ADAPTER_STATUS': {'RRESP_ERR': (0, 1), 'BRESP_ERR': (1, 1), 'ORPHAN_RESP': (2, 1), 'BASE_MISALIGNED': (3, 1), 'REQ_HELD': (4, 1), 'BASE_LOCKED': (5, 1), 'ERR_COUNT': (16, 16)},
    'ADAPTER_VERSION': {'NUM_CH': (0, 8), 'MAJOR': (16, 16)},
    'HANDLE': {'BEAT': (0, 28), 'CH': (28, 4)},
    'TICKET': {'SEQ': (0, 28), 'UNIT': (28, 4)},
    'DESC': {'TYPE': (0, 8), 'PROF': (8, 1), 'KERNEL': (16, 16)},
    'LCQ_R0': {'TYPE': (0, 8), 'STATUS': (8, 8), 'ERR_CODE': (16, 8), 'SEQ': (24, 8)},
    'HELLO_R4': {'SDK_ABI': (0, 16), 'RT_VERSION': (16, 16)},
    'ISSUE_PUSH': {'REC': (0, 1), 'WMASK': (8, 8)},
    'ISSUE_CREDIT': {'DMA': (0, 4), 'QACT': (4, 4), 'MVM': (8, 4), 'VU': (12, 4), 'ATT': (16, 4), 'SYNC': (20, 4)},
    'ISSUE_UNITS': {'DMA': (0, 1), 'QACT': (1, 1), 'MVM': (2, 1), 'VU': (3, 1), 'ATT': (4, 1), 'SYNC': (5, 1)},
    'ISSUE_STATUS': {'PUSH_TIMEOUT': (0, 1), 'ERR_IRQ': (1, 1)},
    'QCTRL': {'QABORT': (0, 1), 'QHOLD_SET': (1, 1), 'QHOLD_CLR': (2, 1), 'QCLEAR': (3, 1)},
    'QSTATUS': {'QIDLE': (0, 1), 'HOLD': (1, 1), 'HOLD_HOST': (2, 1), 'HOLD_RV': (3, 1), 'HOLD_HALT': (4, 1), 'ERR': (8, 6), 'NONEMPTY': (16, 6)},
    'WAIT_CREDIT': {'UNIT': (0, 3), 'EN': (31, 1)},
    'LQ_FLAGS': {'CANCEL_VALID': (0, 1), 'STOP_VALID': (1, 1), 'ACTIVE': (2, 1), 'PLAT_ERR': (3, 1)},
    'LCQ_POST': {'ANSWER': (0, 1)},
    'REC_CFG': {'ALL': (0, 1), 'IRQ_HALF': (1, 1)},
    'REC_DATA3': {'CODE': (0, 4), 'RAN': (4, 1), 'MARKER': (5, 1), 'OPCODE': (8, 8), 'PIDX': (16, 16)},
    'DBG_FLAGS': {'DBG_STOP': (0, 1), 'SERIAL': (1, 1), 'TRW_EN': (2, 1), 'DMI_EN': (3, 1), 'HALT_ON_BOOT': (4, 1)},
    'MARK': {'PAYLOAD': (0, 28), 'CLASS': (28, 4)},
    'MARK_LAUNCH': {'SEQ': (0, 16), 'KERNEL': (16, 12), 'CLASS': (28, 4)},
    'LOG_HDR': {'N': (0, 8), 'LEVEL': (8, 4), 'SOURCE': (12, 4), 'TYPE': (16, 3), 'TS': (19, 1), 'LSEQ': (20, 8), 'SYNC': (28, 4)},
    'TRW_W0': {'KIND': (0, 4), 'UNIT': (4, 4), 'CODE': (8, 4), 'RAN': (12, 1), 'REC': (13, 1), 'TBP': (14, 1), 'OPCODE': (16, 8), 'W0_HI': (24, 8)},
    'DBG_ID': {'VERSION': (0, 8), 'HIST': (8, 1), 'TRW': (9, 1), 'RVWP': (10, 1), 'REC': (11, 1), 'NTBP': (12, 3), 'NRVWP': (15, 2), 'LOG2_HIST': (20, 4)},
    'DBG_CTRL': {'STOP_NOW': (0, 1), 'RELEASE': (1, 1), 'QSTEP': (2, 1), 'HIST_CLEAR': (4, 1), 'RV_SAMPLE': (5, 1), 'RVWP_APPLY': (6, 1), 'TRW_FLUSH': (7, 1), 'RV_HALT_CANCEL': (8, 1), 'MATCHQ_CLEAR': (9, 1)},
    'DBG_CFG': {'DMI_EN': (0, 1), 'SERIAL': (1, 1), 'ERR_STOP': (2, 1), 'STOP_HALT_RV': (3, 1), 'HIST_EN': (4, 1), 'HIST_FREEZE_ERR': (5, 1), 'HIST_FREEZE': (6, 1), 'HUNG_STOP': (7, 1), 'HALT_ON_BOOT': (8, 1), 'RV_HALT_EN': (9, 1), 'TRW_EN': (10, 1), 'TRW_STOP_FULL': (11, 1)},
    'DBG_STATUS': {'STOP_ACTIVE': (0, 1), 'DRAINED': (1, 1), 'STEP_BUSY': (2, 1), 'DB_HELD': (3, 1), 'CAUSE': (4, 5), 'BP_HIT': (9, 4), 'RVWP_HIT': (13, 2), 'RV_HALT_PEND': (15, 1), 'RVWP_BUSY': (16, 1), 'HIST_FROZEN': (17, 1), 'RV_HALTED': (18, 1), 'RV_SLEEP': (19, 1), 'TRW_DROP': (20, 1), 'REC_DROP': (21, 1), 'SBA_REFUSED': (22, 1), 'DMI_BLOCKED': (23, 1), 'RV_SAMPLE_TIMEOUT': (24, 1), 'RV_DFAULT': (25, 1), 'PTR_ERR': (26, 1), 'RVWP_TIMEOUT': (27, 1), 'MATCHQ_OVF': (28, 1), 'DMI_RESP_ERR': (29, 1), 'S_VALID': (31, 1)},
    'DBG_CAUSE': {'NOW': (0, 1), 'BP': (1, 1), 'STEP': (2, 1), 'ERR': (3, 1), 'HUNG': (4, 1)},
    'DBG_STEP_N': {'N': (0, 16)},
    'DBG_MATCHQ': {'PENDING': (0, 7)},
    'DBG_QSTATE': {'DISP': (8, 3), 'OUTSTANDING': (16, 8)},
    'DBG_RV_STATE': {'HALTED': (0, 1), 'SLEEP': (2, 1), 'IN_RESET': (3, 1), 'SAMPLE_VALID': (4, 1), 'DMACTIVE': (5, 1), 'NDMRESET': (6, 1), 'HALT_REQ': (7, 1), 'SAMPLE_SEQ': (8, 8), 'DEBUG_ENTRIES': (16, 16)},
    'DBG_RVWP_CTL': {'EN': (0, 1), 'LOAD': (1, 1), 'STORE': (2, 1), 'HALT': (3, 1)},
    'DBG_TBP_CTL': {'EN': (0, 1), 'KIND': (1, 3), 'R': (4, 1), 'W': (5, 1), 'AFTER': (6, 1), 'HALT_RV': (7, 1), 'COUNT_ONLY': (8, 1), 'CHMASK': (16, 16)},
    'TBP_OPC_A': {'OPC': (0, 8), 'OPC_MASK': (8, 8), 'SUB': (16, 4), 'SUB_MASK': (20, 4)},
    'TBP_OPC_B': {'UNITS': (0, 8)},
    'DBG_U_STATE': {'ACCEPTED': (0, 2), 'HEAD': (4, 3), 'HOLD': (8, 4), 'OUTSTANDING': (12, 8)},
    'DBG_TRW_MASK': {'UNITS': (0, 8), 'MARKERS': (8, 1), 'STOPS': (9, 1), 'RVHALTS': (10, 1)},
    'DBG_HIST_INFO': {'PRESENT': (0, 1), 'UNIT': (4, 4), 'CODE': (8, 4), 'REC': (12, 1), 'TBPS': (13, 4), 'STOP': (17, 1)},
    'IFIFO_CTRL': {'CODE': (0, 4)},
    'IFIFO': {'KIND': (0, 2), 'W': (2, 256), 'DEP0': (258, 32), 'DEP1': (290, 32), 'REC': (322, 1), 'TICKET': (323, 32), 'PC': (355, 32)},
}
ENCODINGS = {   # op -> opcode, fields: name -> (word, lsb, width), rsv: must-be-zero masks per word
    'NOP': dict(opcode=0x00, unit=5, fields={},
        rsv=[0xFFFFFF00, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    'FENCE': dict(opcode=0x7F, unit=5, fields={},
        rsv=[0xFFFFFF00, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    'LD': dict(opcode=0x10, unit=0, fields={'CH': (0, 8, 4), 'SPM': (1, 0, 32), 'DRAM': (2, 0, 32), 'LINES': (3, 0, 32)},
        rsv=[0xFFFFF000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    'ST': dict(opcode=0x11, unit=0, fields={'CH': (0, 8, 4), 'SPM': (1, 0, 32), 'DRAM': (2, 0, 32), 'LINES': (3, 0, 32)},
        rsv=[0xFFFFF000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    'QACT': dict(opcode=0x18, unit=1, fields={'KIND': (0, 8, 4), 'DST': (1, 0, 32), 'SRC': (2, 0, 32), 'K': (3, 0, 32), 'T': (4, 0, 32), 'SRC_STRIDE': (5, 0, 32)},
        rsv=[0xFFFFF000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF]),
    'MVM': dict(opcode=0x20, unit=2, fields={'FMT': (0, 8, 4), 'ACC': (0, 12, 1), 'DST': (1, 0, 32), 'X': (2, 0, 32), 'W_OFF': (3, 0, 32), 'N': (4, 0, 32), 'K': (5, 0, 16), 'T': (5, 16, 4), 'DST_STRIDE': (6, 0, 32)},
        rsv=[0xFFFFE000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFF00000, 0x00000000, 0xFFFFFFFF]),
    'VOP': dict(opcode=0x30, unit=3, fields={'CMODE': (0, 8, 4), 'FUNC': (0, 16, 8), 'SI': (0, 24, 4), 'D': (1, 0, 24), 'A': (2, 0, 24), 'B': (3, 0, 24), 'C': (4, 0, 24), 'M': (5, 0, 24), 'ROWS': (6, 0, 16), 'D_STRIDE': (6, 16, 16), 'A_STRIDE': (7, 0, 16), 'B_STRIDE': (7, 16, 16)},
        rsv=[0xF000F000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0x00000000, 0x00000000]),
    'RED': dict(opcode=0x31, unit=3, fields={'KIND': (0, 8, 4), 'TO_S': (0, 12, 1), 'SD': (0, 16, 4), 'POST': (0, 20, 2), 'SK': (0, 24, 4), 'SE': (0, 28, 4), 'D': (1, 0, 24), 'A': (2, 0, 24), 'M': (5, 0, 24), 'ROWS': (6, 0, 16), 'A_STRIDE': (7, 0, 16)},
        rsv=[0x00C0E000, 0xFF000000, 0xFF000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFF000000, 0xFFFF0000, 0xFFFF0000]),
    'ROPE': dict(opcode=0x32, unit=3, fields={'MODE': (0, 8, 4), 'FF': (0, 12, 1), 'FS': (0, 13, 1), 'ATAB': (0, 14, 1), 'SM': (0, 16, 4), 'STS': (0, 20, 4), 'SFS': (0, 24, 4), 'D': (1, 0, 24), 'SRC': (2, 0, 24), 'POS': (3, 0, 24), 'TABLE': (4, 0, 24), 'HEAD_DIM': (5, 0, 16), 'N_DIMS': (5, 16, 16), 'N_HEADS': (6, 0, 16), 'T': (6, 16, 16), 'SRC_STRIDE': (7, 0, 16), 'D_STRIDE': (7, 16, 16)},
        rsv=[0xF0008000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0x00000000, 0x00000000, 0x00000000]),
    'TOPK': dict(opcode=0x33, unit=3, fields={'K': (0, 16, 5), 'D': (1, 0, 24), 'A': (2, 0, 24), 'M': (5, 0, 24), 'ROWS': (6, 0, 16), 'D_STRIDE': (6, 16, 16), 'A_STRIDE': (7, 0, 16)},
        rsv=[0xFFE0FF00, 0xFF000000, 0xFF000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFF000000, 0x00000000, 0xFFFF0000]),
    'DQROW': dict(opcode=0x12, unit=3, fields={'FMT': (0, 8, 4), 'D': (1, 0, 24), 'SRC': (2, 0, 24), 'K': (5, 0, 32), 'ROWS': (6, 0, 16), 'SRC_STRIDE': (7, 0, 16), 'D_STRIDE': (7, 16, 16)},
        rsv=[0xFFFFF000, 0xFF000000, 0xFF000000, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFF0000, 0x00000000]),
    'VMASK': dict(opcode=0x38, unit=3, fields={'SUB': (0, 8, 4), 'X': (1, 0, 24), 'WIN': (2, 0, 32), 'K0': (3, 0, 32), 'Q0': (4, 0, 32), 'M': (5, 0, 24), 'ROWS': (6, 0, 16), 'G': (6, 16, 16), 'X_STRIDE': (7, 0, 16), 'P': (7, 16, 16)},
        rsv=[0xFFFFF000, 0xFF000000, 0x00000000, 0x00000000, 0x00000000, 0xFF000000, 0x00000000, 0x00000000]),
    'SOP': dict(opcode=0x40, unit=3, fields={'SUB': (0, 8, 4), 'SD': (1, 0, 4), 'SA': (1, 4, 4), 'SB': (1, 8, 4), 'IMM': (2, 0, 32), 'ADDR': (3, 0, 28)},
        rsv=[0xFFFFF000, 0xFFFFF000, 0x00000000, 0xF0000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
    'KVST': dict(opcode=0x14, unit=0, fields={'KV': (0, 8, 4), 'N_ROWS_M1': (0, 16, 6), 'SRC': (1, 0, 32), 'BASE': (2, 0, 32), 'E64': (3, 0, 9), 'LG': (3, 16, 16), 'C0': (4, 0, 32), 'SRC_STRIDE': (5, 0, 32)},
        rsv=[0xFFC0F000, 0x00000000, 0x00000000, 0x0000FE00, 0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF]),
    'ATTQK': dict(opcode=0x21, unit=2, fields={'FMT': (0, 8, 4), 'CAUSAL': (0, 12, 1), 'SLIDE': (0, 13, 1), 'MAXOUT': (0, 14, 1), 'G_M1': (0, 16, 6), 'HG0': (0, 22, 6), 'DST': (1, 0, 32), 'X': (2, 0, 32), 'BASE': (3, 0, 32), 'E64': (4, 0, 9), 'DCODE': (4, 9, 2), 'TQ_M1': (4, 11, 3), 'GS_M1': (4, 14, 3), 'RSTRIDE': (4, 17, 15), 'U0': (5, 0, 16), 'N_GROUPS': (5, 16, 16), 'C_LO': (6, 0, 16), 'C_HI': (6, 16, 16), 'MDST': (7, 0, 32)},
        rsv=[0xF0008000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    'ATTPV': dict(opcode=0x22, unit=2, fields={'FMT': (0, 8, 4), 'LSUM': (0, 12, 1), 'G_M1': (0, 16, 6), 'HG0': (0, 22, 6), 'DST': (1, 0, 32), 'P': (2, 0, 32), 'BASE': (3, 0, 32), 'E64': (4, 0, 9), 'DCODE': (4, 9, 2), 'TQ_M1': (4, 11, 3), 'GS_M1': (4, 14, 3), 'PSTRIDE': (4, 17, 15), 'U0': (5, 0, 16), 'N_GROUPS': (5, 16, 16), 'LG': (6, 0, 16), 'CSTRIDE': (7, 0, 32)},
        rsv=[0xF000E000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFF0000, 0x00000000]),
    'ASMAX': dict(opcode=0x34, unit=3, fields={'DST': (1, 0, 32), 'SRC': (2, 0, 32), 'MVEC': (3, 0, 32), 'N_LINES': (4, 0, 16), 'STRIDE': (4, 16, 16), 'R': (5, 0, 32), 'SCALE': (6, 0, 32)},
        rsv=[0xFFFFFF00, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF]),
    'AMASK': dict(opcode=0x35, unit=3, fields={'DST': (1, 0, 32), 'SRC': (2, 0, 32), 'MASK': (3, 0, 32), 'N_LINES': (4, 0, 16), 'STRIDE': (4, 16, 16), 'R': (5, 0, 16), 'RPQ': (5, 16, 16), 'MSTRIDE': (6, 0, 32), 'MDST': (7, 0, 32)},
        rsv=[0xFFFFFF00, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    'AMERGE': dict(opcode=0x36, unit=3, fields={'INIT': (0, 12, 1), 'ORUN': (1, 0, 32), 'PARTS': (2, 0, 32), 'MK': (3, 0, 32), 'ST': (4, 0, 32), 'R': (5, 0, 16), 'DCODE': (5, 16, 2), 'SCALE': (6, 0, 32), 'CSTRIDE': (7, 0, 32)},
        rsv=[0xFFFFEF00, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xFFFC0000, 0x00000000, 0x00000000]),
    'ANORM': dict(opcode=0x37, unit=3, fields={'OUT': (1, 0, 32), 'ORUN': (2, 0, 32), 'L_RUN': (3, 0, 32), 'R': (4, 0, 16), 'DCODE': (4, 16, 2)},
        rsv=[0xFFFFFF00, 0x00000000, 0x00000000, 0x00000000, 0xFFFC0000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]),
}
UNITS = {'DMA': 0, 'QACT': 1, 'MVM': 2, 'VU': 3, 'ATT': 4, 'SYNC': 5}
ERRORS = {'NONE': 0, 'ILLEGAL_OP': 1, 'BAD_FMT': 2, 'QA_MISMATCH': 3, 'SPM_RANGE': 4, 'DRAM_RANGE': 5, 'K_RANGE': 6, 'T_RANGE': 7, 'ABORTED': 8, 'DEP_FAILED': 11, 'BAD_DEP': 12, 'OPERAND': 13}


def encode(op, **fields):
    """Instruction words of `op` (an ENCODINGS key); unnamed fields are 0. Raises on a value
    that does not fit its field."""
    e = ENCODINGS[op]
    w = [0] * 8
    w[0] = e['opcode']
    for name, value in fields.items():
        word, lsb, width = e['fields'][name]
        if value < 0 or value >> width:
            raise ValueError('%s.%s = %r does not fit %d bits' % (op, name, value, width))
        w[word] |= value << lsb
    return w


def decode(op, words):
    """Field values of `op` from its instruction words; also checks the opcode."""
    e = ENCODINGS[op]
    if words[0] & 0xFF != e['opcode']:
        raise ValueError('opcode 0x%02X is not %s' % (words[0] & 0xFF, op))
    return {name: (words[word] >> lsb) & ((1 << width) - 1)
            for name, (word, lsb, width) in e['fields'].items()}


def reserved_bits(op, words):
    """Per-word mask of set must-be-zero bits (all 0 for a well-formed instruction)."""
    return [w & r for w, r in zip(words, ENCODINGS[op]['rsv'])]

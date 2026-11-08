// GENERATED FILE, DO NOT EDIT: produced by sw/tables/gen_tables.py from sw/tables/tsr_tables.py.
//
// tsr/mmio.hpp — Tesseract tables for the firmware SDK (EX-30, SW §6):
// the RISC-V and host maps, opcodes, units, errors, PERF, the launch / log / trace
// formats, the base-op encodings and tsr::op.
// Freestanding C++20: no libstdc++ headers, no heap, everything constexpr in namespace tsr.
// Names as tsr_tables.h without the TSR_ prefix (see sw/tables/gen_tables.py).
#ifndef TSR_MMIO_HPP
#define TSR_MMIO_HPP

#include <stdint.h>

namespace tsr {

// -----------------------------------------------------------------------------------------------
// Scalar constants (CONSTANTS)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t ID_MAGIC              = 0x54535231u;  // EX §5.1: ID register: "TSR1"
constexpr uint32_t ISA_MAJOR             = 1u;           // EX §7.5: VERSION[15:8]: Tesseract Core capability major
constexpr uint32_t ISA_MINOR             = 0u;           // EX §7.5: VERSION[7:0]: Tesseract Core capability minor
constexpr uint32_t VERSION_VALUE         = 0x100u;       // EX §7.5: VERSION register: Tesseract Core capability 1.0
constexpr uint32_t CTL_VERSION           = 1u;           // EX §5.1: CTL_ID[7:0]: control-core revision; 0 = no control core
constexpr uint32_t ADAPTER_MAJOR         = 1u;           // EX §9.2: ADAPTER_VERSION[31:16] (ADAPTER_VERSION_MAJOR field)
constexpr uint32_t DBG_VERSION           = 1u;           // DB §10.1: DBG_ID[7:0]: DBG register map; DBG_ID = 0 means the DBG block is absent
constexpr uint32_t CANON_NAN             = 0x7FC00000u;  // VU §4.1: the only NaN an fp result has
constexpr uint32_t INSN_WORDS            = 8u;           // EX §7.1: an instruction is 256 bits, w0..w7, little-endian
constexpr uint32_t INSN_BYTES            = 32u;          // EX §7.1
constexpr uint32_t INSN_BITS             = 256u;         // EX §7.1
constexpr uint32_t X16_LSB               = 16u;          // EX §7.1: x16 = w0[31:16]; x16 bit i is w0 bit 16 + i; 0 unless the op defines it
constexpr uint32_t X16_W                 = 16u;          // EX §7.1
constexpr uint32_t LQ_DEPTH              = 8u;           // EX §4.2: U200 launch-queue depth (LQ_CFG[7:0])
constexpr uint32_t LCQ_DEPTH             = 8u;           // EX §4.2: U200 completion-ring depth (LQ_CFG[15:8])
constexpr uint32_t LOG_DEPTH             = 256u;         // EX §4.2: U200 log-ring words (LQ_CFG[27:16])
constexpr uint32_t LCQ_DEPTH_MAX         = 8u;           // EX §4.2: the CSR map limits LCQ_DEPTH to 8
constexpr uint32_t LOG_DEPTH_MAX         = 256u;         // EX §4.2: the CSR map limits LOG_DEPTH to 256
constexpr uint32_t LCQ_REC_BYTES         = 32u;          // EX §4.2: record k at LCQ_REC + 32 x (k mod LCQ_DEPTH)
constexpr uint32_t ARGBLK_MAX_BYTES      = 1024u;        // EX §4.3: argument block limit (ARG0 handle of a kernel with the argument-block flag)
constexpr uint32_t ARGBLK_MAX_LINES      = 16u;          // SW §6.2: argblk_lines of a kernel table entry: 1..16
constexpr uint32_t SEQ_HALF              = 0x80000000u;  // EX §4.9: launch q is covered by watermark s iff (s - q) mod 2^32 < SEQ_HALF
constexpr uint32_t PANIC_MCAUSE          = 24u;          // EX §4.7, DB §6.3: tsr::panic enters the trap handler with this mcause, mtval = code
constexpr uint32_t BOOT_ENTRY_OFF        = 0x80u;        // EX §3.4: the RISC-V fetches at RV_BOOT + 0x80
constexpr uint32_t RV_BOOT_ALIGN         = 256u;         // EX §5.1: RV_BOOT alignment ([7:0] read 0)
constexpr uint32_t CRC32_POLY            = 0xEDB88320u;  // EX §3.3: LOAD_CRC: zlib crc32 (reflected polynomial, init and final XOR 0xFFFFFFFF)
constexpr uint32_t LOAD_TIMEOUT_CYCLES   = 4096u;        // EX §3.3: core cycles a LOAD_DATA word or a VERIFY read may take before LOAD_TIMEOUT
constexpr uint32_t LOGFMT_VMA            = 0xF0000000u;  // SW §6.2, DB §6.1: VMA of the non-loaded .tsr_logfmt section (FMT log records point into it)
constexpr uint32_t WDOG_L_RESET          = 24u;          // EX §5.1, §8.6: WDOG_CFG reset value: 2^24 core cycles without progress
constexpr uint32_t SR_RV_WAIT_CYCLES     = 0x00010000u;  // EX §8.5: SOFT_RESET step 1 waits at most 2^16 core cycles for RV_RESET_DONE
constexpr uint32_t DMI_TIMEOUT_CYCLES    = 4096u;        // EX §5.1: core cycles a DMI access may take before DMI_TIMEOUT
constexpr uint32_t BRIDGE_TIMEOUT_CYCLES = 4096u;        // CC §5.3: clk_rv cycles a core-window or SPM-bridge load may take before a load fault
constexpr uint32_t ISSUE_WAIT_MAX        = 0x00010000u;  // CC §7.3: clk_rv cycles a push without credit waits before a store fault and PUSH_TIMEOUT
constexpr uint32_t TICKET_NONE           = 0xF0000000u;  // CC §5.1, EX §4.4: unit field 15: no dependency / no failing ticket
constexpr uint32_t NUM_CH_MAX            = 16u;          // spec ISA: LD / ST name the channel in the 4-bit sub field (CFG0.NUM_CH, CH_BASE_LO/HI)
constexpr uint32_t NUM_UNITS             = 6u;           // EX §7.3: unit ids 0..5 (6..13 reserved, 14 marker, 15 none)
constexpr uint32_t UNIT_SLOTS            = 8u;           // EX §5.4: per-unit register slots u = 0..7 (absent units read 0)
constexpr uint32_t Q_DEPTH_FIELD_W       = 5u;           // EX §5.1: Q_DEPTH: depth of unit u at [5u+4:5u]
constexpr uint32_t CREDIT_FIELD_W        = 4u;           // CC §5.1: ISSUE_CREDIT: unit u at [4u+3:4u], saturating at 15
constexpr uint32_t ISSUE_FIFO_DEPTH      = 32u;          // CC §7.2: issue-FIFO entries (LUTRAM, clk_rv -> core)
constexpr uint32_t ERR_W                 = 4u;           // EX §8.1: internal error-code width (codes 0..15); CSR fields are 8 bits
constexpr uint32_t PERF_SLOTS            = 64u;          // EX §5.1: PERF_0..PERF_63, 64-bit each
constexpr uint32_t PERF_CH_BASE          = 5u;           // EX §6: CH_RD_BEATS[c] = 5 + c, CH_WAIT[c] = 5 + C + c, X = 5 + 2C
constexpr uint32_t ITCM_BYTES            = 0x00020000u;  // CC §4.1: U200 ITCM (RV_CFG[15:0] in KiB)
constexpr uint32_t DTCM_BYTES            = 0x00010000u;  // CC §4.1: U200 DTCM (RV_CFG[31:16] in KiB)
constexpr uint32_t SPM_BYTES_U200        = 0x00400000u;  // CC §4.1: U200 SPM size: the SPM windows span SPM_BYTES
constexpr uint32_t DBG_NTBP              = 4u;           // DB §10.3: tile breakpoints
constexpr uint32_t DBG_NRVWP             = 2u;           // DB §10.3: RISC-V data watchpoints
constexpr uint32_t HIST_DEPTH            = 512u;         // DB §10.3: dispatch-history entries
constexpr uint32_t REC_DEPTH             = 64u;          // DB §10.3: completion-record FIFO entries
constexpr uint32_t REC_IRQ_LEVEL         = 32u;          // DB §8.2: irq_fast[4] while REC_COUNT >= 32 (REC_CFG.IRQ_HALF)
constexpr uint32_t MATCHQ_DEPTH          = 64u;          // DB §5.2: tile-breakpoint match queue
constexpr uint32_t TRW_CH                = 0u;           // DB §10.3, §8.3: channel of the trace-writer ring (in the ch0 IO region)
constexpr uint32_t TRW_RECORD_BYTES      = 32u;          // DB §8.3: trace-writer record, two per 64-byte beat
constexpr uint32_t TRW_RING_DEFAULT_MIB  = 64u;          // DB §8.3, EX §11: TSR_PROF_TRW_MIB default
constexpr uint32_t DBG_INFO_MAGIC        = 0x47424454u;  // DB §6.3: __tsr_dbg word 0: "TDBG"
constexpr uint32_t CRASH_MAGIC           = 0x50415254u;  // DB §6.3: __tsr_crash word 0: "TRAP"
constexpr uint32_t CRASH_LOG_WORDS       = 44u;          // DB §6.1: TRAP log record payload: words 0..43 of __tsr_crash
constexpr uint32_t VL                    = 16u;          // VU §5.10: CFG2.VL: elements per SPM line
constexpr uint32_t NSREG                 = 16u;          // VU §4.4: s0..s15
constexpr uint32_t VU_FRONT_LANES        = 16u;          // VU §5.10: CFG3[4:0] (U200)
constexpr uint32_t VU_BACK_LANES         = 8u;           // VU §5.10: CFG3[9:5] (U200)
constexpr uint32_t VU_LINE_FIELD_W       = 24u;          // VU §5.1: SPM line addresses and element counts m are w[23:0] with w[31:24] = 0
constexpr uint32_t ROPE_MAX_HEAD_DIM     = 1024u;        // VU §5.4: head_dim is a multiple of 16 in [16, 1024]
constexpr uint32_t TOPK_MAX_K            = 16u;          // VU §5.5: k in 1..16
constexpr uint32_t VU_ACCEPT_AHEAD       = 2u;           // VU §7.8, CC §9.4: VU ops held at once (I3 for unit 3)
constexpr uint32_t GDN_MIN_M             = 64u;          // VU §7.12.4: GDN d_k in [64, 256]: at least four beats of the widest build (16 lanes) per row
constexpr uint32_t GDN_MAX_M             = 256u;         // VU §7.12.4: GDN d_k at most 256: the 48-bit DSP accumulator bound
constexpr uint32_t GDN_MAX_ROWS          = 1024u;        // VU §7.12.4: GDN d_v in [1, 1024]: the bv RAM
constexpr uint32_t M_FLOOR               = 0xF1800000u;  // AT §6.3: -2^100: row-maximum floor of ATTQK MAXOUT and AMASK
constexpr uint32_t ANORM_L_FLOOR         = 0x00800000u;  // AT §5.7: 2^-126: ANORM divides by MAXT(l, 2^-126)
constexpr uint32_t ATT_GROUP_CELLS       = 16u;          // AT §4.2: KV cells per group (groups round-robin over the channels)
constexpr uint32_t ATT_MAX_GROUPS        = 4095u;        // AT §5.2: n_groups in [1, 4095]
constexpr uint32_t ATT_MAX_G             = 64u;          // AT §5: G (heads per KV head) in 1..64
constexpr uint32_t ATT_MAX_PASS_ROWS     = 128u;         // AT §5.8: (E/D) x GS x T_q pass rows at most
constexpr uint32_t ATT_MAX_R             = 1024u;        // AT §5.8: R in [1, 1024] for ASMAX, AMASK, AMERGE, ANORM
constexpr uint32_t ATT_MAX_LINES         = 4095u;        // AT §5.4: ASMAX / AMASK n_lines in [1, 4095]
constexpr uint32_t KVST_MAX_ROWS         = 64u;          // AT §5.1: n_rows in 1..64
constexpr uint32_t PCAP_U200             = 9344u;        // AT §5.8: p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX) x 16

// -----------------------------------------------------------------------------------------------
// Shared value sets (ENUMS)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t FMT_Q8_0   = 0u;  // spec ISA, VU §5.6: weight format Q8_0 (MVM, DQROW sub)
constexpr uint32_t FMT_Q4_K   = 1u;  // spec ISA, VU §5.6: weight format Q4_K; 2..15 BAD_FMT
constexpr uint32_t QA_A32     = 0u;  // spec ISA: QACT kind: block_q8_0 rules, block 32
constexpr uint32_t QA_A256    = 1u;  // spec ISA: QACT kind: block_q8_K rules, block 256, with bsums
constexpr uint32_t KVFMT_Q8_0 = 0u;  // AT §5.2: KV format of ATTQK / ATTPV sub; else BAD_FMT
constexpr uint32_t DCODE_D64  = 0u;  // AT §5: head dimension D = 64 x 2^dcode
constexpr uint32_t DCODE_D128 = 1u;  // AT §5
constexpr uint32_t DCODE_D256 = 2u;  // AT §5
constexpr uint32_t DCODE_D512 = 3u;  // AT §5

// -----------------------------------------------------------------------------------------------
// Field access helpers
// -----------------------------------------------------------------------------------------------
// Field access with the generated mask / _LSB pairs: field_get(v, CTRL_RV_RUN, CTRL_RV_RUN_LSB).
constexpr uint32_t field_get(uint32_t v, uint32_t mask, uint32_t lsb) { return (v & mask) >> lsb; }
constexpr uint32_t field_put(uint32_t mask, uint32_t lsb, uint32_t x) { return (x << lsb) & mask; }

// One 256-bit tile instruction with its used-word mask (docs/ctrl-riscv.md §8.2): issue() writes
// only the staging words whose bit is set and pushes with WMASK = used (ISSUE_PUSH, CC §5.1);
// words with a clear bit are sent as 0. The generated builders set the bits of the words that hold
// a field (<OP>_WORDS); op_finish sets those of the nonzero words (for ops assembled by hand).
struct op {
    uint32_t w[8];
    uint8_t used;
};
constexpr op op_finish(op o) {
    o.used = 1u;
    for (uint32_t i = 1; i < 8; ++i)
        if (o.w[i] != 0u) o.used = uint8_t(o.used | (1u << i));
    return o;
}

// Field placement of the generated builders (vu.hpp, at.hpp; P2.3). A value that does not fit its
// field is never masked into a different, legal value: in a constant expression it is a compile
// error (the call of the non-constexpr detail::op_field_does_not_fit), at run time the builder
// sets the op's lowest reserved w0 bit (the poison argument), so that the checker answers
// ILLEGAL_OP. The builders set op::used to the encoding's <OP>_WORDS (the words that hold a field:
// a field word that happens to be 0 is written and sent as 0, the same push content) instead of
// testing every word, and build the op in place: no copies on the issue path.
namespace detail {
inline void op_field_does_not_fit() {}
}  // namespace detail
constexpr uint32_t op_put(uint32_t x, uint32_t vmask, uint32_t lsb, bool & bad) {
    if (x > vmask) bad = true;
    return (x & vmask) << lsb;
}
constexpr void op_poison_if(op & o, bool bad, uint32_t poison) {
    if (bad) {
        detail::op_field_does_not_fit();
        o.w[0] |= poison;
    }
}

// -----------------------------------------------------------------------------------------------
// Host CSR map: core CSR port, U200 s_axi_control 0x1000 + offset (EX §5.1)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t CSR_ID             = 0x000u;       // [R] EX §5.1: "TSR1" (unchanged)
constexpr uint32_t CSR_ID_RESET       = 0x54535231u;
constexpr uint32_t CSR_VERSION        = 0x004u;       // [R] EX §5.1, §7.5: Tesseract Core capability: [15:8] major 1, [7:0] minor 0
constexpr uint32_t CSR_VERSION_RESET  = 0x00000100u;
constexpr uint32_t CSR_CFG0           = 0x008u;       // [R] EX §5.1: NUM_CH, LANES, T_MAX (unchanged)
constexpr uint32_t CSR_CFG1           = 0x00Cu;       // [R] EX §5.1: SPM size in lines (unchanged)
constexpr uint32_t CSR_CFG2           = 0x010u;       // [R] EX §5.1: VL, NSREG, K_MAX/256 (unchanged)
constexpr uint32_t CSR_CAPS           = 0x014u;       // [R] EX §5.1, §7.5: capability bits
constexpr uint32_t CSR_CFG3           = 0x018u;       // [R] EX §5.1, §7.5: [15:0] VU, [31:16] AT
constexpr uint32_t CSR_CTL_ID         = 0x01Cu;       // [R] EX §5.1: control-core revision and features; 0 = no control core
constexpr uint32_t CSR_CTRL           = 0x020u;       // [RW/W1] EX §5.1, §5.2: device control
constexpr uint32_t CSR_STATUS         = 0x024u;       // [R/W1C] EX §5.1, §5.3: device status; W1C clears the sticky bits only
constexpr uint32_t CSR_RV_BOOT        = 0x028u;       // [RW] EX §5.1: boot vector, taken when RV_RUN rises
constexpr uint32_t CSR_RV_FREQ        = 0x02Cu;       // [R] EX §5.1: clk_rv cycles in the last 2^16 core cycles; 0 = no clk_rv
constexpr uint32_t CSR_RV_CFG         = 0x030u;       // [R] EX §5.1: [15:0] ITCM KiB, [31:16] DTCM KiB
constexpr uint32_t CSR_WDOG_CFG       = 0x034u;       // [RW] EX §5.1, §8.6: unit watchdog limit
constexpr uint32_t CSR_WDOG_CFG_RESET = 0x00000018u;
constexpr uint32_t CSR_RT_STATE       = 0x038u;       // [R] EX §5.1, §4.5: copy of the runtime's RT_STATE
constexpr uint32_t CSR_RT_INFO        = 0x03Cu;       // [R] EX §5.1, §4.5: copy of the runtime's RT_INFO
constexpr uint32_t CSR_LQ_W_BASE      = 0x040u;       // [RW] EX §5.1, §4.3: descriptor staging LQ_W0..LQ_W7; kept after a push
constexpr uint32_t CSR_LQ_W_N         = 8u;
constexpr uint32_t CSR_LQ_W_STRIDE    = 4u;
constexpr uint32_t CSR_LQ_W(uint32_t i) { return 0x040u + 4u * i; }
constexpr uint32_t CSR_LQ_PUSH        = 0x060u;       // [W] EX §5.1: push the staged descriptor (value ignored); dropped with LQ_OVERFLOW when full
constexpr uint32_t CSR_LQ_SEQ         = 0x064u;       // [R] EX §5.1, §4.2: PUSHED
constexpr uint32_t CSR_LQ_TAKEN       = 0x068u;       // [R] EX §5.1, §4.2: TAKEN
constexpr uint32_t CSR_LQ_CANCEL      = 0x06Cu;       // [RW] EX §5.1, §4.9: cancel watermark
constexpr uint32_t CSR_LQ_STOP        = 0x070u;       // [RW] EX §5.1, §4.9: stop watermark
constexpr uint32_t CSR_LCQ_WR         = 0x074u;       // [R] EX §5.1, §4.2: records posted
constexpr uint32_t CSR_LCQ_RD         = 0x078u;       // [RW] EX §5.1, §4.2: records consumed by the host (EX §4.2 write rule)
constexpr uint32_t CSR_LQ_CFG         = 0x07Cu;       // [R] EX §5.1: ring depths and watermark valid bits
constexpr uint32_t CSR_PERF_INFO      = 0x080u;       // [R] EX §5.1: counters implemented, extension index X
constexpr uint32_t CSR_LCQ_ANSWERED   = 0x084u;       // [R] EX §5.1, §4.2: ANSWERED
constexpr uint32_t CSR_Q_DEPTH        = 0x088u;       // [R] EX §5.1: queue depth per unit; 0 = absent
constexpr uint32_t CSR_LOG_WR         = 0x090u;       // [R] EX §5.1, §4.10: log words accepted
constexpr uint32_t CSR_LOG_RD         = 0x094u;       // [RW] EX §5.1, §4.10: log words consumed by the host
constexpr uint32_t CSR_LOG_CFG        = 0x098u;       // [RW] EX §5.1, §4.10: log configuration
constexpr uint32_t CSR_LOG_CFG_RESET  = 0x000000F1u;
constexpr uint32_t CSR_LOG_DROPPED    = 0x09Cu;       // [R] EX §5.1: words dropped, saturating; cleared by SOFT_RESET
constexpr uint32_t CSR_LOAD_ADDR      = 0x0C0u;       // [RW] EX §5.1, §3.3: TCM byte address (word aligned); a write restarts LOAD_CRC
constexpr uint32_t CSR_LOAD_DATA      = 0x0C4u;       // [RW] EX §5.1, §3.3: write: store the word, address += 4; read: the word (only while RV_RUN = 0 and not SR_BUSY)
constexpr uint32_t CSR_LOAD_CRC       = 0x0C8u;       // [R] EX §5.1, §3.3: CRC-32 of the words accepted, or read by VERIFY
constexpr uint32_t CSR_LOAD_COUNT     = 0x0CCu;       // [RW] EX §5.1, §3.3: word count for VERIFY
constexpr uint32_t CSR_LOAD_CTRL      = 0x0D0u;       // [W] EX §5.1, §3.3: bit0 VERIFY
constexpr uint32_t CSR_LOAD_STATUS    = 0x0D4u;       // [R] EX §5.1, §3.3: bit0 VERIFY_BUSY
constexpr uint32_t CSR_PERF_LO_BASE   = 0x100u;       // [R] EX §5.1, §6: counter i bits [31:0]; a host LO read snapshots the counter for the next HI read
constexpr uint32_t CSR_PERF_LO_N      = 64u;
constexpr uint32_t CSR_PERF_LO_STRIDE = 8u;
constexpr uint32_t CSR_PERF_LO(uint32_t i) { return 0x100u + 8u * i; }
constexpr uint32_t CSR_PERF_HI_BASE   = 0x104u;  // [R] EX §5.1, §6: counter i bits [63:32] (the snapshot)
constexpr uint32_t CSR_PERF_HI_N      = 64u;
constexpr uint32_t CSR_PERF_HI_STRIDE = 8u;
constexpr uint32_t CSR_PERF_HI(uint32_t i) { return 0x104u + 8u * i; }
constexpr uint32_t CSR_LCQ_REC_BASE   = 0x300u;  // [R] EX §5.1, §4.2: completion ring: record k at 0x300 + 32 (k mod LCQ_DEPTH), word j at +4j
constexpr uint32_t CSR_LCQ_REC_N      = 8u;
constexpr uint32_t CSR_LCQ_REC_STRIDE = 32u;
constexpr uint32_t CSR_LCQ_REC_WORDS  = 8u;
constexpr uint32_t CSR_LCQ_REC(uint32_t i, uint32_t j) { return 0x300u + 32u * i + 4u * j; }
constexpr uint32_t CSR_LOG_RING_BASE   = 0x400u;  // [R] EX §5.1, §4.10: log word k at 0x400 + 4 (k mod LOG_DEPTH)
constexpr uint32_t CSR_LOG_RING_N      = 256u;
constexpr uint32_t CSR_LOG_RING_STRIDE = 4u;
constexpr uint32_t CSR_LOG_RING(uint32_t i) { return 0x400u + 4u * i; }
constexpr uint32_t CSR_DBG_BASE  = 0x800u;  // [DB] EX §5.1, DB §10.1: DBG block (DBG_CSR rows); undefined offsets read 0
constexpr uint32_t CSR_DBG_BYTES = 768u;
// 0xB00..0xBFC reserved (EX §5.1)
constexpr uint32_t CSR_DMI_BASE   = 0xC00u;  // [R/W] EX §5.1, DB §3.2: debug module register r at 0xC00 + 4r; acts only while DBG_CFG.DMI_EN = 1
constexpr uint32_t CSR_DMI_N      = 128u;
constexpr uint32_t CSR_DMI_STRIDE = 4u;
constexpr uint32_t CSR_DMI(uint32_t i) { return 0xC00u + 4u * i; }
constexpr uint32_t CSR_Q_DONE_BASE   = 0xE00u;  // [R] EX §5.4: DONE_SEQ[u], 28 bits
constexpr uint32_t CSR_Q_DONE_N      = 8u;
constexpr uint32_t CSR_Q_DONE_STRIDE = 4u;
constexpr uint32_t CSR_Q_DONE(uint32_t i) { return 0xE00u + 4u * i; }
constexpr uint32_t CSR_Q_ISSUED_BASE   = 0xE20u;  // [R] EX §5.4: ISSUED_SEQ[u], 28 bits (synchronised copy)
constexpr uint32_t CSR_Q_ISSUED_N      = 8u;
constexpr uint32_t CSR_Q_ISSUED_STRIDE = 4u;
constexpr uint32_t CSR_Q_ISSUED(uint32_t i) { return 0xE20u + 4u * i; }
constexpr uint32_t CSR_Q_STATE_BASE   = 0xE40u;  // [R] EX §5.4: queue state of unit u
constexpr uint32_t CSR_Q_STATE_N      = 8u;
constexpr uint32_t CSR_Q_STATE_STRIDE = 4u;
constexpr uint32_t CSR_Q_STATE(uint32_t i) { return 0xE40u + 4u * i; }
constexpr uint32_t CSR_Q_FIFO           = 0xE60u;  // [R] EX §5.4: issue-FIFO entries waiting for the dispatcher
constexpr uint32_t CSR_Q_PIDX           = 0xE64u;  // [R] EX §5.4: push index of the dispatcher's next ticketed entry (= DBG_NEXT_PIDX); firmware never uses it
constexpr uint32_t CSR_Q_ERR_SEQ_BASE   = 0xE80u;  // [R] EX §5.4: first error record of unit u: ERR_SEQ
constexpr uint32_t CSR_Q_ERR_SEQ_N      = 8u;
constexpr uint32_t CSR_Q_ERR_SEQ_STRIDE = 16u;
constexpr uint32_t CSR_Q_ERR_SEQ(uint32_t i) { return 0xE80u + 16u * i; }
constexpr uint32_t CSR_Q_ERR_CODE_BASE   = 0xE84u;  // [R] EX §5.4: ERR_CODE ([7:0])
constexpr uint32_t CSR_Q_ERR_CODE_N      = 8u;
constexpr uint32_t CSR_Q_ERR_CODE_STRIDE = 16u;
constexpr uint32_t CSR_Q_ERR_CODE(uint32_t i) { return 0xE84u + 16u * i; }
constexpr uint32_t CSR_Q_ERR_INFO_BASE   = 0xE88u;  // [R] EX §5.4: ERR_INFO
constexpr uint32_t CSR_Q_ERR_INFO_N      = 8u;
constexpr uint32_t CSR_Q_ERR_INFO_STRIDE = 16u;
constexpr uint32_t CSR_Q_ERR_INFO(uint32_t i) { return 0xE88u + 16u * i; }
constexpr uint32_t CSR_Q_ERR_W0_BASE   = 0xE8Cu;  // [R] EX §5.4: ERR_W0
constexpr uint32_t CSR_Q_ERR_W0_N      = 8u;
constexpr uint32_t CSR_Q_ERR_W0_STRIDE = 16u;
constexpr uint32_t CSR_Q_ERR_W0(uint32_t i) { return 0xE8Cu + 16u * i; }
constexpr uint32_t CSR_Q_ERR_PIDX_BASE   = 0xF00u;  // [R] EX §5.4: push index of unit u's first error
constexpr uint32_t CSR_Q_ERR_PIDX_N      = 8u;
constexpr uint32_t CSR_Q_ERR_PIDX_STRIDE = 4u;
constexpr uint32_t CSR_Q_ERR_PIDX(uint32_t i) { return 0xF00u + 4u * i; }
// 0xF20..0xFFC reserved (EX §5.1)

// -----------------------------------------------------------------------------------------------
// DBG block: host CSR 0x800..0xAFC (DB §10.1)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t CSR_DBG_ID              = 0x800u;       // [R] DB §10.1: version and features; 0 = block absent
constexpr uint32_t CSR_DBG_CTRL            = 0x804u;       // [W] DB §10.1: pulses, reads 0
constexpr uint32_t CSR_DBG_CFG             = 0x808u;       // [RW] DB §10.1: configuration
constexpr uint32_t CSR_DBG_CFG_RESET       = 0x00000030u;
constexpr uint32_t CSR_DBG_STATUS          = 0x80Cu;       // [R/W1C] DB §10.1: levels and sticky bits
constexpr uint32_t CSR_DBG_STOP_PIDX       = 0x810u;       // [R] DB §10.1: T (valid while STOP_ACTIVE)
constexpr uint32_t CSR_DBG_STOP_AT         = 0x814u;       // [RW] DB §10.1: write P: S <- P, valid; read: S (valid iff S_VALID)
constexpr uint32_t CSR_DBG_STEP_N          = 0x818u;       // [RW] DB §10.1: entries per QSTEP
constexpr uint32_t CSR_DBG_STEP_N_RESET    = 0x00000001u;
constexpr uint32_t CSR_DBG_NEXT_PIDX       = 0x81Cu;       // [R] DB §10.1: push index the dispatcher assigns to its next ticketed entry
constexpr uint32_t CSR_DBG_STEP_PIDX       = 0x820u;       // [R] DB §10.1: X of the current / last QSTEP
constexpr uint32_t CSR_DBG_MATCHQ          = 0x824u;       // [R] DB §10.1: pending match thresholds
constexpr uint32_t CSR_DBG_MTIME_LO        = 0x828u;       // [R] DB §10.1: live core time (read HI, LO, HI)
constexpr uint32_t CSR_DBG_MTIME_HI        = 0x82Cu;       // [R] DB §10.1
constexpr uint32_t CSR_DBG_QSTATE          = 0x834u;       // [R] DB §10.1: dispatcher state, outstanding entries
constexpr uint32_t CSR_DBG_RV_STATE        = 0x840u;       // [R] DB §10.1: RISC-V state and sample sequence
constexpr uint32_t CSR_DBG_RV_PC           = 0x844u;       // [R] DB §10.1, §6.2: last crash-dump sample: current PC
constexpr uint32_t CSR_DBG_RV_NPC          = 0x848u;       // [R] DB §10.1, §6.2: next PC
constexpr uint32_t CSR_DBG_RV_DADDR        = 0x84Cu;       // [R] DB §10.1, §6.2: last data address
constexpr uint32_t CSR_DBG_RV_EPC          = 0x850u;       // [R] DB §10.1, §6.2: exception PC
constexpr uint32_t CSR_DBG_RV_EADDR        = 0x854u;       // [R] DB §10.1, §6.2: exception address
constexpr uint32_t CSR_DBG_RV_WPADDR       = 0x858u;       // [R] DB §10.1: address of the last RV-watchpoint hit
constexpr uint32_t CSR_DBG_RVWP_CTL_BASE   = 0x860u;       // [RW] DB §10.1: RV watchpoint j < 2
constexpr uint32_t CSR_DBG_RVWP_CTL_N      = 2u;
constexpr uint32_t CSR_DBG_RVWP_CTL_STRIDE = 16u;
constexpr uint32_t CSR_DBG_RVWP_CTL(uint32_t i) { return 0x860u + 16u * i; }
constexpr uint32_t CSR_DBG_RVWP_LO_BASE   = 0x864u;  // [RW] DB §10.1: byte range [LO, HI)
constexpr uint32_t CSR_DBG_RVWP_LO_N      = 2u;
constexpr uint32_t CSR_DBG_RVWP_LO_STRIDE = 16u;
constexpr uint32_t CSR_DBG_RVWP_LO(uint32_t i) { return 0x864u + 16u * i; }
constexpr uint32_t CSR_DBG_RVWP_HI_BASE   = 0x868u;  // [RW] DB §10.1
constexpr uint32_t CSR_DBG_RVWP_HI_N      = 2u;
constexpr uint32_t CSR_DBG_RVWP_HI_STRIDE = 16u;
constexpr uint32_t CSR_DBG_RVWP_HI(uint32_t i) { return 0x868u + 16u * i; }
constexpr uint32_t CSR_DBG_RVWP_HITS_BASE   = 0x86Cu;  // [R/WC] DB §10.1: hits; a write clears
constexpr uint32_t CSR_DBG_RVWP_HITS_N      = 2u;
constexpr uint32_t CSR_DBG_RVWP_HITS_STRIDE = 16u;
constexpr uint32_t CSR_DBG_RVWP_HITS(uint32_t i) { return 0x86Cu + 16u * i; }
constexpr uint32_t CSR_DBG_TBP_CTL_BASE   = 0x880u;  // [RW] DB §10.1: tile breakpoint i < 4
constexpr uint32_t CSR_DBG_TBP_CTL_N      = 4u;
constexpr uint32_t CSR_DBG_TBP_CTL_STRIDE = 16u;
constexpr uint32_t CSR_DBG_TBP_CTL(uint32_t i) { return 0x880u + 16u * i; }
constexpr uint32_t CSR_DBG_TBP_A_BASE   = 0x884u;  // [RW] DB §10.1, §5.3
constexpr uint32_t CSR_DBG_TBP_A_N      = 4u;
constexpr uint32_t CSR_DBG_TBP_A_STRIDE = 16u;
constexpr uint32_t CSR_DBG_TBP_A(uint32_t i) { return 0x884u + 16u * i; }
constexpr uint32_t CSR_DBG_TBP_B_BASE   = 0x888u;  // [RW] DB §10.1, §5.3
constexpr uint32_t CSR_DBG_TBP_B_N      = 4u;
constexpr uint32_t CSR_DBG_TBP_B_STRIDE = 16u;
constexpr uint32_t CSR_DBG_TBP_B(uint32_t i) { return 0x888u + 16u * i; }
constexpr uint32_t CSR_DBG_TBP_HITS_BASE   = 0x88Cu;  // [R/WC] DB §10.1: matches; a write clears
constexpr uint32_t CSR_DBG_TBP_HITS_N      = 4u;
constexpr uint32_t CSR_DBG_TBP_HITS_STRIDE = 16u;
constexpr uint32_t CSR_DBG_TBP_HITS(uint32_t i) { return 0x88Cu + 16u * i; }
constexpr uint32_t CSR_DBG_RDOUT_BASE   = 0x8C0u;  // [R] DB §10.1, §5.5: outstanding read bursts: channel c in byte c mod 4 of word c / 4
constexpr uint32_t CSR_DBG_RDOUT_N      = 4u;
constexpr uint32_t CSR_DBG_RDOUT_STRIDE = 4u;
constexpr uint32_t CSR_DBG_RDOUT(uint32_t i) { return 0x8C0u + 4u * i; }
constexpr uint32_t CSR_DBG_WROUT_BASE   = 0x8D0u;  // [R] DB §10.1, §5.5: outstanding write bursts, same packing
constexpr uint32_t CSR_DBG_WROUT_N      = 4u;
constexpr uint32_t CSR_DBG_WROUT_STRIDE = 4u;
constexpr uint32_t CSR_DBG_WROUT(uint32_t i) { return 0x8D0u + 4u * i; }
constexpr uint32_t CSR_DBG_U_STATE_BASE   = 0x900u;  // [R] DB §10.1, §5.5: unit u < 8
constexpr uint32_t CSR_DBG_U_STATE_N      = 8u;
constexpr uint32_t CSR_DBG_U_STATE_STRIDE = 16u;
constexpr uint32_t CSR_DBG_U_STATE(uint32_t i) { return 0x900u + 16u * i; }
constexpr uint32_t CSR_DBG_U_OLDEST_BASE   = 0x904u;  // [R] DB §10.1, §5.5: ticket of the oldest outstanding entry; TICKET_NONE if none
constexpr uint32_t CSR_DBG_U_OLDEST_N      = 8u;
constexpr uint32_t CSR_DBG_U_OLDEST_STRIDE = 16u;
constexpr uint32_t CSR_DBG_U_OLDEST(uint32_t i) { return 0x904u + 16u * i; }
constexpr uint32_t CSR_DBG_U_OPIDX_BASE   = 0x908u;  // [R] DB §10.1, §5.5: its push index
constexpr uint32_t CSR_DBG_U_OPIDX_N      = 8u;
constexpr uint32_t CSR_DBG_U_OPIDX_STRIDE = 16u;
constexpr uint32_t CSR_DBG_U_OPIDX(uint32_t i) { return 0x908u + 16u * i; }
constexpr uint32_t CSR_DBG_U_AGE_BASE   = 0x90Cu;  // [R] DB §10.1, §5.5: core cycles since it became the oldest (saturating)
constexpr uint32_t CSR_DBG_U_AGE_N      = 8u;
constexpr uint32_t CSR_DBG_U_AGE_STRIDE = 16u;
constexpr uint32_t CSR_DBG_U_AGE(uint32_t i) { return 0x90Cu + 16u * i; }
constexpr uint32_t CSR_DBG_TRW_BASE       = 0x980u;       // [RW] DB §10.1, §8.3: ring base handle, 16-beat aligned (beat [3:0] read 0)
constexpr uint32_t CSR_DBG_TRW_LINES      = 0x984u;       // [RW] DB §10.1, §8.3: ring size in 64-B lines, power of two >= 16 (else ignored, PTR_ERR)
constexpr uint32_t CSR_DBG_TRW_WPTR       = 0x988u;       // [R] DB §10.1, §8.3: records written (mod 2^32)
constexpr uint32_t CSR_DBG_TRW_RPTR       = 0x98Cu;       // [RW] DB §10.1, §8.3: records consumed; beyond WPTR ignored (PTR_ERR)
constexpr uint32_t CSR_DBG_TRW_DROPS      = 0x990u;       // [R] DB §10.1, §8.3: records dropped (saturating)
constexpr uint32_t CSR_DBG_TRW_MASK       = 0x994u;       // [RW] DB §10.1, §8.3: record classes written
constexpr uint32_t CSR_DBG_TRW_MASK_RESET = 0x000007FFu;
constexpr uint32_t CSR_SREG_BASE          = 0x9C0u;       // [R] EX §5.1, §7.4, DB §10.1, VU §4.4: s0..s15 (f32 bits); present whenever the VU is built, also with HAS_DBG = 0
constexpr uint32_t CSR_SREG_N             = 16u;
constexpr uint32_t CSR_SREG_STRIDE        = 4u;
constexpr uint32_t CSR_SREG(uint32_t i) { return 0x9C0u + 4u * i; }
constexpr uint32_t CSR_DBG_HIST_HEAD     = 0xA00u;  // [R] DB §10.1, §5.6: next push index to be recorded
constexpr uint32_t CSR_DBG_HIST_IDX      = 0xA04u;  // [RW] DB §10.1, §5.6: push index to read
constexpr uint32_t CSR_DBG_HIST_INFO     = 0xA08u;  // [R] DB §10.1, §5.6
constexpr uint32_t CSR_DBG_HIST_PIDX     = 0xA0Cu;  // [R] DB §10.1, §5.6
constexpr uint32_t CSR_DBG_HIST_TICKET   = 0xA10u;  // [R] DB §10.1, §5.6
constexpr uint32_t CSR_DBG_HIST_DEP0     = 0xA14u;  // [R] DB §10.1, §5.6
constexpr uint32_t CSR_DBG_HIST_DEP1     = 0xA18u;  // [R] DB §10.1, §5.6
constexpr uint32_t CSR_DBG_HIST_TIME     = 0xA1Cu;  // [R] DB §10.1, §5.6: dispatch time MTIME[31:0]
constexpr uint32_t CSR_DBG_HIST_PC       = 0xA20u;  // [R] DB §10.1, §5.6: push PC
constexpr uint32_t CSR_DBG_HIST_W_BASE   = 0xA40u;  // [R] DB §10.1, §5.6: op words w0..w7
constexpr uint32_t CSR_DBG_HIST_W_N      = 8u;
constexpr uint32_t CSR_DBG_HIST_W_STRIDE = 4u;
constexpr uint32_t CSR_DBG_HIST_W(uint32_t i) { return 0xA40u + 4u * i; }

// -----------------------------------------------------------------------------------------------
// RISC-V control window: offsets from RV_CTL_BASE (CC §5, EX §4.5, DB §10.2)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t CW_ISSUE_W_BASE   = 0x000u;  // [RW] CC §5.1: staged instruction words; kept after a push
constexpr uint32_t CW_ISSUE_W_N      = 8u;
constexpr uint32_t CW_ISSUE_W_STRIDE = 4u;
constexpr uint32_t CW_ISSUE_W(uint32_t i) { return 0x000u + 4u * i; }
constexpr uint32_t CW_ISSUE_W_RESET    = 0x00000000u;
constexpr uint32_t CW_ISSUE_DEP0       = 0x020u;       // [RW] CC §5.1: dependency ticket; reset to TICKET_NONE by every push
constexpr uint32_t CW_ISSUE_DEP0_RESET = 0xF0000000u;
constexpr uint32_t CW_ISSUE_DEP1       = 0x024u;       // [RW] CC §5.1: same
constexpr uint32_t CW_ISSUE_DEP1_RESET = 0xF0000000u;
constexpr uint32_t CW_ISSUE_PUSH       = 0x028u;       // [W] CC §5.1: push; a push without a credit completes late (safety net)
constexpr uint32_t CW_ISSUE_TICKET     = 0x02Cu;       // [R] CC §5.1: ticket of the last push or QCTRL.QCLEAR
constexpr uint32_t CW_ISSUE_CREDIT     = 0x030u;       // [R] CC §5.1: credit per unit
constexpr uint32_t CW_ISSUE_UNITS      = 0x034u;       // [R] CC §5.1: bit u = unit u is built
constexpr uint32_t CW_ISSUE_STATUS     = 0x038u;       // [R/W1C] CC §5.1: push timeout and error interrupt
constexpr uint32_t CW_ISSUE_PIDX       = 0x03Cu;       // [R] CC §5.1: push index the next ticketed entry will receive (+1 per ISSUE_PUSH and QCLEAR); exact
constexpr uint32_t CW_DONE_SEQ_BASE    = 0x040u;       // [R] CC §5.1: [27:0] completions of unit u (synchronised copy; may lag, never leads)
constexpr uint32_t CW_DONE_SEQ_N       = 8u;
constexpr uint32_t CW_DONE_SEQ_STRIDE  = 4u;
constexpr uint32_t CW_DONE_SEQ(uint32_t i) { return 0x040u + 4u * i; }
constexpr uint32_t CW_ISSUED_SEQ_BASE   = 0x080u;  // [R] CC §5.1: [27:0] sequence the next push to unit u gets
constexpr uint32_t CW_ISSUED_SEQ_N      = 8u;
constexpr uint32_t CW_ISSUED_SEQ_STRIDE = 4u;
constexpr uint32_t CW_ISSUED_SEQ(uint32_t i) { return 0x080u + 4u * i; }
constexpr uint32_t CW_ERR_SEQ_BASE   = 0x0C0u;  // [R] CC §5.1: first error of unit u since the last clear: ERR_SEQ
constexpr uint32_t CW_ERR_SEQ_N      = 8u;
constexpr uint32_t CW_ERR_SEQ_STRIDE = 16u;
constexpr uint32_t CW_ERR_SEQ(uint32_t i) { return 0x0C0u + 16u * i; }
constexpr uint32_t CW_ERR_CODE_BASE   = 0x0C4u;  // [R] CC §5.1: ERR_CODE; 0 = none
constexpr uint32_t CW_ERR_CODE_N      = 8u;
constexpr uint32_t CW_ERR_CODE_STRIDE = 16u;
constexpr uint32_t CW_ERR_CODE(uint32_t i) { return 0x0C4u + 16u * i; }
constexpr uint32_t CW_ERR_INFO_BASE   = 0x0C8u;  // [R] CC §5.1: ERR_INFO
constexpr uint32_t CW_ERR_INFO_N      = 8u;
constexpr uint32_t CW_ERR_INFO_STRIDE = 16u;
constexpr uint32_t CW_ERR_INFO(uint32_t i) { return 0x0C8u + 16u * i; }
constexpr uint32_t CW_ERR_W0_BASE   = 0x0CCu;  // [R] CC §5.1: ERR_W0
constexpr uint32_t CW_ERR_W0_N      = 8u;
constexpr uint32_t CW_ERR_W0_STRIDE = 16u;
constexpr uint32_t CW_ERR_W0(uint32_t i) { return 0x0CCu + 16u * i; }
constexpr uint32_t CW_QCTRL             = 0x140u;       // [W] CC §5.1: queue control entries, in push order
constexpr uint32_t CW_QSTATUS           = 0x144u;       // [R] CC §5.1: queue status
constexpr uint32_t CW_WAIT_TICKET       = 0x148u;       // [RW] CC §5.1: irq_fast[0] pending while this ticket is done; unit 15 disarms
constexpr uint32_t CW_WAIT_TICKET_RESET = 0xF0000000u;
constexpr uint32_t CW_WAIT_CREDIT       = 0x14Cu;       // [RW] CC §5.1: irq_fast[5] pending while that unit's credit != 0
constexpr uint32_t CW_WAIT_CREDIT_RESET = 0x00000000u;
constexpr uint32_t CW_REC_COUNT         = 0x160u;       // [R] DB §10.2, §8.2: completion records waiting (0..64)
constexpr uint32_t CW_REC_DATA_BASE     = 0x164u;       // [R] DB §10.2, §8.2: head record words 0..3 (word table REC)
constexpr uint32_t CW_REC_DATA_N        = 4u;
constexpr uint32_t CW_REC_DATA_STRIDE   = 4u;
constexpr uint32_t CW_REC_DATA(uint32_t i) { return 0x164u + 4u * i; }
constexpr uint32_t CW_REC_POP         = 0x174u;       // [W] DB §10.2, §8.2: drop the head record
constexpr uint32_t CW_TRACE_MARK      = 0x178u;       // [W] DB §10.2, §8.4: marker through the issue FIFO
constexpr uint32_t CW_REC_CFG         = 0x17Cu;       // [RW] DB §10.2, §8.2: completion-record configuration
constexpr uint32_t CW_REC_CFG_RESET   = 0x00000000u;
constexpr uint32_t CW_ERR_PIDX_BASE   = 0x180u;       // [R] CC §5.1: push index of unit u's first error
constexpr uint32_t CW_ERR_PIDX_N      = 8u;
constexpr uint32_t CW_ERR_PIDX_STRIDE = 4u;
constexpr uint32_t CW_ERR_PIDX(uint32_t i) { return 0x180u + 4u * i; }
// 0x1C0..0x1C0 reserved (CC §5.1)
constexpr uint32_t CW_SPM_LB_INV       = 0x1C4u;  // [W] CC §5.1, §10.5: invalidate the SPM line buffer
constexpr uint32_t CW_LQ_HEAD_W_BASE   = 0x200u;  // [R] EX §4.5: the descriptor the next LQ_POP takes (valid while LQ_AVAIL != 0)
constexpr uint32_t CW_LQ_HEAD_W_N      = 8u;
constexpr uint32_t CW_LQ_HEAD_W_STRIDE = 4u;
constexpr uint32_t CW_LQ_HEAD_W(uint32_t i) { return 0x200u + 4u * i; }
constexpr uint32_t CW_LQ_AVAIL     = 0x220u;  // [R] EX §4.5: PUSHED - TAKEN as seen in clk_rv
constexpr uint32_t CW_LQ_HEAD_SEQ  = 0x224u;  // [R] EX §4.5: TAKEN = the sequence number of the head
constexpr uint32_t CW_LQ_POP       = 0x228u;  // [W] EX §4.5: take the head; access fault if LQ_AVAIL = 0 or ACTIVE
constexpr uint32_t CW_LQ_CANCEL    = 0x22Cu;  // [R] EX §4.5: copy of the host's cancel watermark
constexpr uint32_t CW_LQ_STOP      = 0x230u;  // [R] EX §4.5: copy of the host's stop watermark
constexpr uint32_t CW_LQ_FLAGS     = 0x234u;  // [R] EX §4.5: watermark valid bits, ACTIVE, PLAT_ERR
constexpr uint32_t CW_LCQ_W_BASE   = 0x240u;  // [RW] EX §4.5: record staging; kept after a post
constexpr uint32_t CW_LCQ_W_N      = 8u;
constexpr uint32_t CW_LCQ_W_STRIDE = 4u;
constexpr uint32_t CW_LCQ_W(uint32_t i) { return 0x240u + 4u * i; }
constexpr uint32_t CW_LCQ_POST          = 0x260u;       // [W] EX §4.5: post the staged record; access fault if LCQ_FREE = 0, or ANSWER while not ACTIVE
constexpr uint32_t CW_LCQ_FREE          = 0x264u;       // [R] EX §4.5: LCQ_DEPTH - (LCQ_WR - LCQ_RD) as seen in clk_rv (may under-report)
constexpr uint32_t CW_LOG_PUSH          = 0x280u;       // [W] EX §4.5: one log word; dropped and counted when the ring is full, never faults
constexpr uint32_t CW_LOG_FREE          = 0x284u;       // [R] EX §4.5: free words (may under-report)
constexpr uint32_t CW_LOG_CFG           = 0x288u;       // [R] EX §4.5: copy of the host's LOG_CFG
constexpr uint32_t CW_RT_STATE          = 0x290u;       // [RW] EX §4.5: runtime phase, mirrored to host 0x038; 0 after a RISC-V reset
constexpr uint32_t CW_RT_STATE_RESET    = 0x00000000u;
constexpr uint32_t CW_RT_INFO           = 0x294u;       // [RW] EX §4.5: runtime- or kernel-defined progress word, mirrored to host 0x03C
constexpr uint32_t CW_MTIME_LO          = 0x300u;       // [R] CC §5.2: core clock cycles since reset (gray-synchronised)
constexpr uint32_t CW_MTIME_HI          = 0x304u;       // [R] CC §5.2
constexpr uint32_t CW_MTIMECMP_LO       = 0x308u;       // [RW] CC §5.2: irq_timer while MTIME >= MTIMECMP
constexpr uint32_t CW_MTIMECMP_LO_RESET = 0xFFFFFFFFu;
constexpr uint32_t CW_MTIMECMP_HI       = 0x30Cu;       // [RW] CC §5.2
constexpr uint32_t CW_MTIMECMP_HI_RESET = 0xFFFFFFFFu;
constexpr uint32_t CW_DBG_FLAGS         = 0x400u;       // [R] DB §10.2: DB state for the runtime
// 0x404..0x7FC reserved (DB §10.2)

// -----------------------------------------------------------------------------------------------
// XRT adapter s_axi_control map (EX §9.2)
// -----------------------------------------------------------------------------------------------
// 0x000..0x03C reserved (EX §9.2)
constexpr uint32_t SAXI_CH_BASE_LO_BASE   = 0x0040u;  // [RW] EX §9.2: device byte address of the channel-c buffer, bits [31:0]; 4 KiB aligned; written while MEM_EN = 0
constexpr uint32_t SAXI_CH_BASE_LO_N      = 16u;
constexpr uint32_t SAXI_CH_BASE_LO_STRIDE = 8u;
constexpr uint32_t SAXI_CH_BASE_LO(uint32_t i) { return 0x040u + 8u * i; }
constexpr uint32_t SAXI_CH_BASE_HI_BASE   = 0x0044u;  // [RW] EX §9.2: bits [63:32]
constexpr uint32_t SAXI_CH_BASE_HI_N      = 16u;
constexpr uint32_t SAXI_CH_BASE_HI_STRIDE = 8u;
constexpr uint32_t SAXI_CH_BASE_HI(uint32_t i) { return 0x044u + 8u * i; }
constexpr uint32_t SAXI_ADAPTER_CTRL    = 0x00E0u;  // [RW] EX §9.2: MEM_EN
constexpr uint32_t SAXI_ADAPTER_STATUS  = 0x00F0u;  // [R/W1C] EX §9.2: sticky adapter errors
constexpr uint32_t SAXI_ADAPTER_VERSION = 0x00F4u;  // [R] EX §9.2: {16'h0001, 8'h00, NUM_CH}
constexpr uint32_t SAXI_CORE_BASE       = 0x1000u;  // [-] EX §9.2: core CSR window: 0x1000 + offset, any time; writes need WSTRB = 0xF
constexpr uint32_t SAXI_CORE_BYTES      = 4096u;

// -----------------------------------------------------------------------------------------------
// RISC-V address map (CC §4.1); region sizes are the U200 values
// -----------------------------------------------------------------------------------------------
constexpr uint32_t RV_ITCM_BASE    = 0x00000000u;  // CC §4.1: code; data reads steal the fetch port (byte/half/word, fetch)
constexpr uint32_t RV_ITCM_SIZE    = 0x00020000u;
constexpr uint32_t RV_ITCM_LIMIT   = 0x00100000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_DTCM_BASE    = 0x00100000u;  // CC §4.1: .rodata, .data, .bss, stack (byte/half/word)
constexpr uint32_t RV_DTCM_SIZE    = 0x00010000u;
constexpr uint32_t RV_DTCM_LIMIT   = 0x00200000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_DM_BASE      = 0x10000000u;  // CC §4.1: debug module memory (riscv-dbg layout) (word, fetch)
constexpr uint32_t RV_DM_SIZE      = 0x00001000u;
constexpr uint32_t RV_DM_LIMIT     = 0x10001000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_CTL_BASE     = 0x20000000u;  // CC §4.1: control window (CTL_WINDOW) (word)
constexpr uint32_t RV_CTL_SIZE     = 0x00001000u;
constexpr uint32_t RV_CTL_LIMIT    = 0x20001000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_CORE_BASE    = 0x20001000u;  // CC §4.1, §5.3: core window: read-only view of the host CSR map (same offsets) (word loads)
constexpr uint32_t RV_CORE_SIZE    = 0x00001000u;
constexpr uint32_t RV_CORE_LIMIT   = 0x20002000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_SPM_BASE     = 0x40000000u;  // CC §4.1: SPM window, exact: 0x4000_0000 + 64 x line + offset (byte/half/word)
constexpr uint32_t RV_SPM_SIZE     = 0x00400000u;
constexpr uint32_t RV_SPM_LIMIT    = 0x40400000u;  // end of the reserved range (exclusive)
constexpr uint32_t RV_SPM_LB_BASE  = 0x50000000u;  // CC §4.1, §10.5: SPM line-buffered alias (byte/half/word)
constexpr uint32_t RV_SPM_LB_SIZE  = 0x00400000u;
constexpr uint32_t RV_SPM_LB_LIMIT = 0x50400000u;  // end of the reserved range (exclusive)

// -----------------------------------------------------------------------------------------------
// Ibex fast interrupts irq_fast[i] (CC §12.3)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t IRQ_FAST_WAIT_TICKET = 0u;  // CC §12.3: WAIT_TICKET done
constexpr uint32_t IRQ_FAST_ERR         = 1u;  // CC §12.3: ISSUE_STATUS.ERR_IRQ
constexpr uint32_t IRQ_FAST_LQ          = 2u;  // CC §12.3, EX §4.5: LQ_AVAIL != 0 and not ACTIVE
constexpr uint32_t IRQ_FAST_LCQ         = 3u;  // CC §12.3, EX §4.5: LCQ_FREE != 0
constexpr uint32_t IRQ_FAST_REC         = 4u;  // CC §12.3, DB §8.2: REC_COUNT >= 32 with REC_CFG.IRQ_HALF
constexpr uint32_t IRQ_FAST_CREDIT      = 5u;  // CC §12.3: WAIT_CREDIT
constexpr uint32_t IRQ_FAST_DB_RSVD     = 6u;  // CC §12.3, DB §10.2: reserved for DB

// -----------------------------------------------------------------------------------------------
// DMI registers (DB §9.1); host CSR offset = CSR_DMI(address)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t DMI_DATA0      = 4u;   // DB §9.1
constexpr uint32_t DMI_DATA1      = 5u;   // DB §9.1
constexpr uint32_t DMI_DMCONTROL  = 16u;  // DB §9.1: haltreq, resumereq, ackhavereset, ndmreset, dmactive
constexpr uint32_t DMI_DMSTATUS   = 17u;  // DB §9.1
constexpr uint32_t DMI_HARTINFO   = 18u;  // DB §9.1
constexpr uint32_t DMI_ABSTRACTCS = 22u;  // DB §9.1: progbufsize, busy, cmderr, datacount
constexpr uint32_t DMI_COMMAND    = 23u;  // DB §9.1: Access Register, postexec-only
constexpr uint32_t DMI_PROGBUF0   = 32u;  // DB §9.1: progbuf0..7
constexpr uint32_t DMI_PROGBUF0_N = 8u;
constexpr uint32_t DMI_SBCS       = 56u;  // DB §9.1
constexpr uint32_t DMI_SBADDRESS0 = 57u;  // DB §9.1
constexpr uint32_t DMI_SBDATA0    = 60u;  // DB §9.1

// -----------------------------------------------------------------------------------------------
// DRAM map per channel (EX §11); byte offsets
// -----------------------------------------------------------------------------------------------
constexpr uint64_t REGION_OPL_OFF    = 0x0ull;         // EX §11: ch0: op lists, replay push streams, argument blocks, code overlays (was CMD)
constexpr uint64_t REGION_OPL_SIZE   = 0x4000000ull;
constexpr uint64_t REGION_IO_OFF     = 0x4000000ull;   // EX §11: ch0: host <-> device tensors; compute chunks; the TRW ring while profiling
constexpr uint64_t REGION_IO_SIZE    = 0x10000000ull;
constexpr uint64_t REGION_AUX_OFF    = 0x14000000ull;  // EX §11: ch0: small f32 tensors (norm weights, RoPE tables)
constexpr uint64_t REGION_AUX_SIZE   = 0xC000000ull;
constexpr uint64_t REGION_WEIGHT_OFF = 0x20000000ull;  // EX §11: every channel: weights bottom-up, KV planes top-down, compute heap between

// -----------------------------------------------------------------------------------------------
// Bit-field layouts (in-place masks; _LSB, _W; values unshifted)
// -----------------------------------------------------------------------------------------------
// W0 (EX §7.1): instruction word w0 = {x16, flags, sub, opcode}
constexpr uint32_t W0_OPCODE     = 0x000000FFu;  // [7:0] opcode
constexpr uint32_t W0_OPCODE_LSB = 0u;
constexpr uint32_t W0_OPCODE_W   = 8u;
constexpr uint32_t W0_SUB        = 0x00000F00u;  // [11:8] op-specific sub code
constexpr uint32_t W0_SUB_LSB    = 8u;
constexpr uint32_t W0_SUB_W      = 4u;
constexpr uint32_t W0_FLAGS      = 0x0000F000u;  // [15:12] op-specific flags
constexpr uint32_t W0_FLAGS_LSB  = 12u;
constexpr uint32_t W0_FLAGS_W    = 4u;
constexpr uint32_t W0_X16        = 0xFFFF0000u;  // [31:16] op-specific field; must be 0 (ILLEGAL_OP) where the op does not assign it
constexpr uint32_t W0_X16_LSB    = 16u;
constexpr uint32_t W0_X16_W      = 16u;
// VERSION (EX §5.1, §7.5): Tesseract Core capability MAJOR.MINOR; [31:16] 0
constexpr uint32_t VERSION_MINOR     = 0x000000FFu;  // [7:0] capability minor (0)
constexpr uint32_t VERSION_MINOR_LSB = 0u;
constexpr uint32_t VERSION_MINOR_W   = 8u;
constexpr uint32_t VERSION_MAJOR     = 0x0000FF00u;  // [15:8] capability major (1)
constexpr uint32_t VERSION_MAJOR_LSB = 8u;
constexpr uint32_t VERSION_MAJOR_W   = 8u;
// CFG0 (EX §5.1): geometry configuration
constexpr uint32_t CFG0_NUM_CH     = 0x000000FFu;  // [7:0] channels C
constexpr uint32_t CFG0_NUM_CH_LSB = 0u;
constexpr uint32_t CFG0_NUM_CH_W   = 8u;
constexpr uint32_t CFG0_LANES      = 0x000FFF00u;  // [19:8] MAC lanes
constexpr uint32_t CFG0_LANES_LSB  = 8u;
constexpr uint32_t CFG0_LANES_W    = 12u;
constexpr uint32_t CFG0_T_MAX      = 0x00F00000u;  // [23:20] token slots
constexpr uint32_t CFG0_T_MAX_LSB  = 20u;
constexpr uint32_t CFG0_T_MAX_W    = 4u;
// CFG2 (EX §5.1): geometry configuration
constexpr uint32_t CFG2_VL            = 0x000000FFu;  // [7:0] vector length
constexpr uint32_t CFG2_VL_LSB        = 0u;
constexpr uint32_t CFG2_VL_W          = 8u;
constexpr uint32_t CFG2_NSREG         = 0x0000FF00u;  // [15:8] s registers
constexpr uint32_t CFG2_NSREG_LSB     = 8u;
constexpr uint32_t CFG2_NSREG_W       = 8u;
constexpr uint32_t CFG2_K_MAX_256     = 0xFFFF0000u;  // [31:16] K_MAX / 256
constexpr uint32_t CFG2_K_MAX_256_LSB = 16u;
constexpr uint32_t CFG2_K_MAX_256_W   = 16u;
// CAPS (EX §7.5): capability bits; retired bits are never set and never reused
constexpr uint32_t CAPS_VU                    = 0x00000001u;  // [0] VOP, RED, ROPE and SOP present (VU §5.10)
constexpr uint32_t CAPS_VU_LSB                = 0u;
constexpr uint32_t CAPS_VU_W                  = 1u;
constexpr uint32_t CAPS_TOPK                  = 0x00000008u;  // [3] TOPK present (RTL from M8; the simulator from M6)
constexpr uint32_t CAPS_TOPK_LSB              = 3u;
constexpr uint32_t CAPS_TOPK_W                = 1u;
constexpr uint32_t CAPS_ATT                   = 0x00000020u;  // [5] the seven AT ops present (KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM)
constexpr uint32_t CAPS_ATT_LSB               = 5u;
constexpr uint32_t CAPS_ATT_W                 = 1u;
constexpr uint32_t CAPS_DQROW                 = 0x00000040u;  // [6] DQROW in the SPM form (RTL from M8; the simulator from M6)
constexpr uint32_t CAPS_DQROW_LSB             = 6u;
constexpr uint32_t CAPS_DQROW_W               = 1u;
constexpr uint32_t CAPS_VMASK                 = 0x00000080u;  // [7] VMASK (M9)
constexpr uint32_t CAPS_VMASK_LSB             = 7u;
constexpr uint32_t CAPS_VMASK_W               = 1u;
constexpr uint32_t CAPS_GDN                   = 0x00000100u;  // [8] GDN present (the fused Gated DeltaNet state update, VU §7.12)
constexpr uint32_t CAPS_GDN_LSB               = 8u;
constexpr uint32_t CAPS_GDN_W                 = 1u;
constexpr uint32_t CAPS_VU_RSVD               = 0x00000E00u;  // [11:9] reserved for VU extensions, 0
constexpr uint32_t CAPS_VU_RSVD_LSB           = 9u;
constexpr uint32_t CAPS_VU_RSVD_W             = 3u;
constexpr uint32_t CAPS_ATT_LOCAL             = 0x00001000u;  // [12] four-owner local FAST attention score and probability slots
constexpr uint32_t CAPS_ATT_LOCAL_LSB         = 12u;
constexpr uint32_t CAPS_ATT_LOCAL_W           = 1u;
constexpr uint32_t CAPS_MVM_LOCAL_SCALE       = 0x00002000u;  // [13] scale MVM output through the owner-local VU before export
constexpr uint32_t CAPS_MVM_LOCAL_SCALE_LSB   = 13u;
constexpr uint32_t CAPS_MVM_LOCAL_SCALE_W     = 1u;
constexpr uint32_t CAPS_MVM_LOCAL_SOFTCAP     = 0x00004000u;  // [14] owner-local FP32 scale, TANH, scale after MVM
constexpr uint32_t CAPS_MVM_LOCAL_SOFTCAP_LSB = 14u;
constexpr uint32_t CAPS_MVM_LOCAL_SOFTCAP_W   = 1u;
constexpr uint32_t CAPS_AT_RSVD               = 0x00008000u;  // [15] reserved for AT, 0
constexpr uint32_t CAPS_AT_RSVD_LSB           = 15u;
constexpr uint32_t CAPS_AT_RSVD_W             = 1u;
// bit 2 retired (RLD / MULI / LOOP), never set, never reused
// bit 4 retired (WAITM / JUMP), never set, never reused
// CFG3 (EX §7.5, VU §5.10, AT §5.10): [15:0] VU, [31:16] AT
constexpr uint32_t CFG3_VU_FRONT     = 0x0000001Fu;  // [4:0] VU front lanes with an EXP core (information for cycle models)
constexpr uint32_t CFG3_VU_FRONT_LSB = 0u;
constexpr uint32_t CFG3_VU_FRONT_W   = 5u;
constexpr uint32_t CFG3_VU_BACK      = 0x000003E0u;  // [9:5] VU back lanes (information for cycle models)
constexpr uint32_t CFG3_VU_BACK_LSB  = 5u;
constexpr uint32_t CFG3_VU_BACK_W    = 5u;
constexpr uint32_t CFG3_PCAP16       = 0xFFFF0000u;  // [31:16] PCAP / 16 (U200: 584)
constexpr uint32_t CFG3_PCAP16_LSB   = 16u;
constexpr uint32_t CFG3_PCAP16_W     = 16u;
// CTL_ID (EX §5.1): control-core revision and features
constexpr uint32_t CTL_ID_VERSION     = 0x000000FFu;  // [7:0] control-core revision (1); 0: no control core
constexpr uint32_t CTL_ID_VERSION_LSB = 0u;
constexpr uint32_t CTL_ID_VERSION_W   = 8u;
constexpr uint32_t CTL_ID_DM          = 0x00000100u;  // [8] debug module
constexpr uint32_t CTL_ID_DM_LSB      = 8u;
constexpr uint32_t CTL_ID_DM_W        = 1u;
constexpr uint32_t CTL_ID_REC         = 0x00000200u;  // [9] completion records
constexpr uint32_t CTL_ID_REC_LSB     = 9u;
constexpr uint32_t CTL_ID_REC_W       = 1u;
constexpr uint32_t CTL_ID_BRIDGE      = 0x00000400u;  // [10] SPM bridge
constexpr uint32_t CTL_ID_BRIDGE_LSB  = 10u;
constexpr uint32_t CTL_ID_BRIDGE_W    = 1u;
constexpr uint32_t CTL_ID_LOG         = 0x00000800u;  // [11] log ring
constexpr uint32_t CTL_ID_LOG_LSB     = 11u;
constexpr uint32_t CTL_ID_LOG_W       = 1u;
constexpr uint32_t CTL_ID_WDOG        = 0x00001000u;  // [12] watchdog
constexpr uint32_t CTL_ID_WDOG_LSB    = 12u;
constexpr uint32_t CTL_ID_WDOG_W      = 1u;
constexpr uint32_t CTL_ID_SBA         = 0x00002000u;  // [13] SBA connected
constexpr uint32_t CTL_ID_SBA_LSB     = 13u;
constexpr uint32_t CTL_ID_SBA_W       = 1u;
constexpr uint32_t CTL_ID_SPM_LB      = 0x00004000u;  // [14] SPM line-buffered alias
constexpr uint32_t CTL_ID_SPM_LB_LSB  = 14u;
constexpr uint32_t CTL_ID_SPM_LB_W    = 1u;
// CTRL (EX §5.2): pulse bits read 0; level bits act before the pulses of the same write
constexpr uint32_t CTRL_RV_RUN           = 0x00000001u;  // [0] RW level: 1 releases the RISC-V; 1 -> 0 starts the RISC-V reset sequence; refused while SR_BUSY
constexpr uint32_t CTRL_RV_RUN_LSB       = 0u;
constexpr uint32_t CTRL_RV_RUN_W         = 1u;
constexpr uint32_t CTRL_HOST_HOLD        = 0x00000002u;  // [1] RW level: the host's hold source for the queue heads
constexpr uint32_t CTRL_HOST_HOLD_LSB    = 1u;
constexpr uint32_t CTRL_HOST_HOLD_W      = 1u;
constexpr uint32_t CTRL_HOLD_ON_HALT     = 0x00000004u;  // [2] RW level: a debug halt holds the queue heads
constexpr uint32_t CTRL_HOLD_ON_HALT_LSB = 2u;
constexpr uint32_t CTRL_HOLD_ON_HALT_W   = 1u;
constexpr uint32_t CTRL_IRQ_EN           = 0x00000008u;  // [3] RW level: host_irq_o = IRQ_EN and LCQ_WR != LCQ_RD
constexpr uint32_t CTRL_IRQ_EN_LSB       = 3u;
constexpr uint32_t CTRL_IRQ_EN_W         = 1u;
constexpr uint32_t CTRL_QABORT           = 0x00000100u;  // [8] W1 pulse: host QABORT, at once
constexpr uint32_t CTRL_QABORT_LSB       = 8u;
constexpr uint32_t CTRL_QABORT_W         = 1u;
constexpr uint32_t CTRL_QCLEAR           = 0x00000200u;  // [9] W1 pulse: host QCLEAR; only with QIDLE and the RISC-V quiet, else QCLEAR_REFUSED
constexpr uint32_t CTRL_QCLEAR_LSB       = 9u;
constexpr uint32_t CTRL_QCLEAR_W         = 1u;
constexpr uint32_t CTRL_SOFT_RESET       = 0x00000400u;  // [10] W1 pulse: device-level reset sequence (EX §8.5)
constexpr uint32_t CTRL_SOFT_RESET_LSB   = 10u;
constexpr uint32_t CTRL_SOFT_RESET_W     = 1u;
// STATUS (EX §5.3): bits 0..15 levels, 16..27 sticky (W1C)
constexpr uint32_t STATUS_RV_RUNNING         = 0x00000001u;  // [0] RV_RUN = 1 and the hart is out of reset
constexpr uint32_t STATUS_RV_RUNNING_LSB     = 0u;
constexpr uint32_t STATUS_RV_RUNNING_W       = 1u;
constexpr uint32_t STATUS_RV_HALTED          = 0x00000002u;  // [1] debug halt
constexpr uint32_t STATUS_RV_HALTED_LSB      = 1u;
constexpr uint32_t STATUS_RV_HALTED_W        = 1u;
constexpr uint32_t STATUS_RV_SLEEP           = 0x00000004u;  // [2] WFI
constexpr uint32_t STATUS_RV_SLEEP_LSB       = 2u;
constexpr uint32_t STATUS_RV_SLEEP_W         = 1u;
constexpr uint32_t STATUS_RV_CLK_OK          = 0x00000008u;  // [3] clk_rv runs and its domain is out of reset
constexpr uint32_t STATUS_RV_CLK_OK_LSB      = 3u;
constexpr uint32_t STATUS_RV_CLK_OK_W        = 1u;
constexpr uint32_t STATUS_RV_RESET_DONE      = 0x00000010u;  // [4] the RISC-V reset sequence has completed
constexpr uint32_t STATUS_RV_RESET_DONE_LSB  = 4u;
constexpr uint32_t STATUS_RV_RESET_DONE_W    = 1u;
constexpr uint32_t STATUS_ACTIVE             = 0x00000020u;  // [5] TAKEN != ANSWERED
constexpr uint32_t STATUS_ACTIVE_LSB         = 5u;
constexpr uint32_t STATUS_ACTIVE_W           = 1u;
constexpr uint32_t STATUS_QIDLE              = 0x00000040u;  // [6] no entry queued or running, issue FIFO empty
constexpr uint32_t STATUS_QIDLE_LSB          = 6u;
constexpr uint32_t STATUS_QIDLE_W            = 1u;
constexpr uint32_t STATUS_QQUIESCENT         = 0x00000080u;  // [7] quiescent (CC §9.6)
constexpr uint32_t STATUS_QQUIESCENT_LSB     = 7u;
constexpr uint32_t STATUS_QQUIESCENT_W       = 1u;
constexpr uint32_t STATUS_QERR               = 0x00000100u;  // [8] some unit is in error state
constexpr uint32_t STATUS_QERR_LSB           = 8u;
constexpr uint32_t STATUS_QERR_W             = 1u;
constexpr uint32_t STATUS_SR_BUSY            = 0x00000200u;  // [9] SOFT_RESET in progress
constexpr uint32_t STATUS_SR_BUSY_LSB        = 9u;
constexpr uint32_t STATUS_SR_BUSY_W          = 1u;
constexpr uint32_t STATUS_LCQ_PENDING        = 0x00000400u;  // [10] LCQ_WR != LCQ_RD
constexpr uint32_t STATUS_LCQ_PENDING_LSB    = 10u;
constexpr uint32_t STATUS_LCQ_PENDING_W      = 1u;
constexpr uint32_t STATUS_LOG_PENDING        = 0x00000800u;  // [11] LOG_WR != LOG_RD
constexpr uint32_t STATUS_LOG_PENDING_LSB    = 11u;
constexpr uint32_t STATUS_LOG_PENDING_W      = 1u;
constexpr uint32_t STATUS_PLAT_ERR           = 0x00001000u;  // [12] the platform's error input plat_err_i
constexpr uint32_t STATUS_PLAT_ERR_LSB       = 12u;
constexpr uint32_t STATUS_PLAT_ERR_W         = 1u;
constexpr uint32_t STATUS_HOLD_SRC           = 0x0000E000u;  // [15:13] hold sources (layout HOLD_SRC: host, RISC-V, debug halt)
constexpr uint32_t STATUS_HOLD_SRC_LSB       = 13u;
constexpr uint32_t STATUS_HOLD_SRC_W         = 3u;
constexpr uint32_t STATUS_LOAD_REFUSED       = 0x00010000u;  // [16] sticky: LOAD_DATA write or VERIFY while RV_RUN = 1 or SR_BUSY
constexpr uint32_t STATUS_LOAD_REFUSED_LSB   = 16u;
constexpr uint32_t STATUS_LOAD_REFUSED_W     = 1u;
constexpr uint32_t STATUS_LOAD_TIMEOUT       = 0x00020000u;  // [17] sticky: a load or VERIFY word timed out
constexpr uint32_t STATUS_LOAD_TIMEOUT_LSB   = 17u;
constexpr uint32_t STATUS_LOAD_TIMEOUT_W     = 1u;
constexpr uint32_t STATUS_DMI_TIMEOUT        = 0x00040000u;  // [18] sticky
constexpr uint32_t STATUS_DMI_TIMEOUT_LSB    = 18u;
constexpr uint32_t STATUS_DMI_TIMEOUT_W      = 1u;
constexpr uint32_t STATUS_QCLEAR_REFUSED     = 0x00080000u;  // [19] sticky
constexpr uint32_t STATUS_QCLEAR_REFUSED_LSB = 19u;
constexpr uint32_t STATUS_QCLEAR_REFUSED_W   = 1u;
constexpr uint32_t STATUS_PUSH_TIMEOUT       = 0x00100000u;  // [20] sticky: the RISC-V's issue safety net fired
constexpr uint32_t STATUS_PUSH_TIMEOUT_LSB   = 20u;
constexpr uint32_t STATUS_PUSH_TIMEOUT_W     = 1u;
constexpr uint32_t STATUS_BRIDGE_TIMEOUT     = 0x00200000u;  // [21] sticky
constexpr uint32_t STATUS_BRIDGE_TIMEOUT_LSB = 21u;
constexpr uint32_t STATUS_BRIDGE_TIMEOUT_W   = 1u;
constexpr uint32_t STATUS_LQ_OVERFLOW        = 0x00400000u;  // [22] sticky: a push while the LQ was full (descriptor dropped)
constexpr uint32_t STATUS_LQ_OVERFLOW_LSB    = 22u;
constexpr uint32_t STATUS_LQ_OVERFLOW_W      = 1u;
constexpr uint32_t STATUS_RING_RD_BAD        = 0x00800000u;  // [23] sticky: an LCQ_RD or LOG_RD write was refused
constexpr uint32_t STATUS_RING_RD_BAD_LSB    = 23u;
constexpr uint32_t STATUS_RING_RD_BAD_W      = 1u;
constexpr uint32_t STATUS_LOG_DROP           = 0x01000000u;  // [24] sticky: log words were dropped
constexpr uint32_t STATUS_LOG_DROP_LSB       = 24u;
constexpr uint32_t STATUS_LOG_DROP_W         = 1u;
constexpr uint32_t STATUS_UNIT_HUNG          = 0x02000000u;  // [25] sticky: a unit watchdog fired
constexpr uint32_t STATUS_UNIT_HUNG_LSB      = 25u;
constexpr uint32_t STATUS_UNIT_HUNG_W        = 1u;
constexpr uint32_t STATUS_SR_RV_TIMEOUT      = 0x04000000u;  // [26] sticky: SOFT_RESET went on without the RISC-V reset completing
constexpr uint32_t STATUS_SR_RV_TIMEOUT_LSB  = 26u;
constexpr uint32_t STATUS_SR_RV_TIMEOUT_W    = 1u;
constexpr uint32_t STATUS_RV_RUN_REFUSED     = 0x08000000u;  // [27] sticky
constexpr uint32_t STATUS_RV_RUN_REFUSED_LSB = 27u;
constexpr uint32_t STATUS_RV_RUN_REFUSED_W   = 1u;
// HOLD_SRC (EX §5.3, CC §5.1, DB §5.5): bit order of a hold-source field (STATUS[15:13], QSTATUS[4:2], DBG_U_STATE[11:8])
constexpr uint32_t HOLD_SRC_HOST     = 0x00000001u;  // [0] host (CTRL.HOST_HOLD)
constexpr uint32_t HOLD_SRC_HOST_LSB = 0u;
constexpr uint32_t HOLD_SRC_HOST_W   = 1u;
constexpr uint32_t HOLD_SRC_RV       = 0x00000002u;  // [1] RISC-V (QCTRL QHOLD)
constexpr uint32_t HOLD_SRC_RV_LSB   = 1u;
constexpr uint32_t HOLD_SRC_RV_W     = 1u;
constexpr uint32_t HOLD_SRC_HALT     = 0x00000004u;  // [2] debug halt with HOLD_ON_HALT
constexpr uint32_t HOLD_SRC_HALT_LSB = 2u;
constexpr uint32_t HOLD_SRC_HALT_W   = 1u;
constexpr uint32_t HOLD_SRC_DB       = 0x00000008u;  // [3] DB's dbg_hold (DBG_U_STATE only)
constexpr uint32_t HOLD_SRC_DB_LSB   = 3u;
constexpr uint32_t HOLD_SRC_DB_W     = 1u;
// RV_BOOT (EX §5.1): boot vector, 256-byte aligned
constexpr uint32_t RV_BOOT_ADDR     = 0xFFFFFF00u;  // [31:8] boot vector bits [31:8]; [7:0] read 0
constexpr uint32_t RV_BOOT_ADDR_LSB = 8u;
constexpr uint32_t RV_BOOT_ADDR_W   = 24u;
// RV_CFG (EX §5.1): TCM sizes
constexpr uint32_t RV_CFG_ITCM_KIB     = 0x0000FFFFu;  // [15:0] ITCM KiB
constexpr uint32_t RV_CFG_ITCM_KIB_LSB = 0u;
constexpr uint32_t RV_CFG_ITCM_KIB_W   = 16u;
constexpr uint32_t RV_CFG_DTCM_KIB     = 0xFFFF0000u;  // [31:16] DTCM KiB
constexpr uint32_t RV_CFG_DTCM_KIB_LSB = 16u;
constexpr uint32_t RV_CFG_DTCM_KIB_W   = 16u;
// WDOG_CFG (EX §5.1, §8.6): unit progress watchdog
constexpr uint32_t WDOG_CFG_L     = 0x0000001Fu;  // [4:0] limit 2^L core cycles without progress; 0 = off
constexpr uint32_t WDOG_CFG_L_LSB = 0u;
constexpr uint32_t WDOG_CFG_L_W   = 5u;
// RT_STATE (EX §4.5): runtime phase (RISC-V 0x290, mirrored to host 0x038)
constexpr uint32_t RT_STATE_PHASE          = 0x000000FFu;  // [7:0] runtime phase
constexpr uint32_t RT_STATE_PHASE_LSB      = 0u;
constexpr uint32_t RT_STATE_PHASE_W        = 8u;
constexpr uint32_t RT_STATE_KERNEL         = 0xFFFF0000u;  // [31:16] kernel index
constexpr uint32_t RT_STATE_KERNEL_LSB     = 16u;
constexpr uint32_t RT_STATE_KERNEL_W       = 16u;
constexpr uint32_t RT_STATE_PHASE_RESET    = 0u;
constexpr uint32_t RT_STATE_PHASE_BOOT     = 1u;
constexpr uint32_t RT_STATE_PHASE_IDLE     = 2u;
constexpr uint32_t RT_STATE_PHASE_DISPATCH = 3u;
constexpr uint32_t RT_STATE_PHASE_KERNEL   = 4u;
constexpr uint32_t RT_STATE_PHASE_JOIN     = 5u;
constexpr uint32_t RT_STATE_PHASE_ANSWER   = 6u;
constexpr uint32_t RT_STATE_PHASE_TRAPPED  = 7u;
// LQ_CFG (EX §5.1): ring parameters and watermark valid bits
constexpr uint32_t LQ_CFG_LQ_DEPTH         = 0x000000FFu;  // [7:0]
constexpr uint32_t LQ_CFG_LQ_DEPTH_LSB     = 0u;
constexpr uint32_t LQ_CFG_LQ_DEPTH_W       = 8u;
constexpr uint32_t LQ_CFG_LCQ_DEPTH        = 0x0000FF00u;  // [15:8]
constexpr uint32_t LQ_CFG_LCQ_DEPTH_LSB    = 8u;
constexpr uint32_t LQ_CFG_LCQ_DEPTH_W      = 8u;
constexpr uint32_t LQ_CFG_LOG_DEPTH        = 0x0FFF0000u;  // [27:16]
constexpr uint32_t LQ_CFG_LOG_DEPTH_LSB    = 16u;
constexpr uint32_t LQ_CFG_LOG_DEPTH_W      = 12u;
constexpr uint32_t LQ_CFG_STOP_VALID       = 0x40000000u;  // [30]
constexpr uint32_t LQ_CFG_STOP_VALID_LSB   = 30u;
constexpr uint32_t LQ_CFG_STOP_VALID_W     = 1u;
constexpr uint32_t LQ_CFG_CANCEL_VALID     = 0x80000000u;  // [31]
constexpr uint32_t LQ_CFG_CANCEL_VALID_LSB = 31u;
constexpr uint32_t LQ_CFG_CANCEL_VALID_W   = 1u;
// PERF_INFO (EX §5.1): PERF layout
constexpr uint32_t PERF_INFO_COUNT     = 0x000000FFu;  // [7:0] counters implemented
constexpr uint32_t PERF_INFO_COUNT_LSB = 0u;
constexpr uint32_t PERF_INFO_COUNT_W   = 8u;
constexpr uint32_t PERF_INFO_X         = 0x0000FF00u;  // [15:8] first extension index X = 5 + 2 NUM_CH
constexpr uint32_t PERF_INFO_X_LSB     = 8u;
constexpr uint32_t PERF_INFO_X_W       = 8u;
// Q_DEPTH (EX §5.1, CC §7.4): queue depth of unit u at [5u+4:5u]; 0 = unit absent
constexpr uint32_t Q_DEPTH_DMA      = 0x0000001Fu;  // [4:0] U200 M6: 8
constexpr uint32_t Q_DEPTH_DMA_LSB  = 0u;
constexpr uint32_t Q_DEPTH_DMA_W    = 5u;
constexpr uint32_t Q_DEPTH_QACT     = 0x000003E0u;  // [9:5] 4
constexpr uint32_t Q_DEPTH_QACT_LSB = 5u;
constexpr uint32_t Q_DEPTH_QACT_W   = 5u;
constexpr uint32_t Q_DEPTH_MVM      = 0x00007C00u;  // [14:10] 4
constexpr uint32_t Q_DEPTH_MVM_LSB  = 10u;
constexpr uint32_t Q_DEPTH_MVM_W    = 5u;
constexpr uint32_t Q_DEPTH_VU       = 0x000F8000u;  // [19:15] 16
constexpr uint32_t Q_DEPTH_VU_LSB   = 15u;
constexpr uint32_t Q_DEPTH_VU_W     = 5u;
constexpr uint32_t Q_DEPTH_ATT      = 0x01F00000u;  // [24:20] 0: absent in M6
constexpr uint32_t Q_DEPTH_ATT_LSB  = 20u;
constexpr uint32_t Q_DEPTH_ATT_W    = 5u;
constexpr uint32_t Q_DEPTH_SYNC     = 0x3E000000u;  // [29:25] 4
constexpr uint32_t Q_DEPTH_SYNC_LSB = 25u;
constexpr uint32_t Q_DEPTH_SYNC_W   = 5u;
// LOG_CFG (EX §4.10): host RW at 0x098, RISC-V read-only copy at 0x288
constexpr uint32_t LOG_CFG_ENABLE     = 0x00000001u;  // [0] 0: the SDK drops every record
constexpr uint32_t LOG_CFG_ENABLE_LSB = 0u;
constexpr uint32_t LOG_CFG_ENABLE_W   = 1u;
constexpr uint32_t LOG_CFG_BLOCK      = 0x00000002u;  // [1] the SDK waits for space instead of dropping
constexpr uint32_t LOG_CFG_BLOCK_LSB  = 1u;
constexpr uint32_t LOG_CFG_BLOCK_W    = 1u;
constexpr uint32_t LOG_CFG_LEVEL      = 0x000000F0u;  // [7:4] the SDK drops records above this level (the only runtime filter)
constexpr uint32_t LOG_CFG_LEVEL_LSB  = 4u;
constexpr uint32_t LOG_CFG_LEVEL_W    = 4u;
// LOAD_CTRL (EX §5.1, §3.3)
constexpr uint32_t LOAD_CTRL_VERIFY     = 0x00000001u;  // [0] read LOAD_COUNT words back and CRC them
constexpr uint32_t LOAD_CTRL_VERIFY_LSB = 0u;
constexpr uint32_t LOAD_CTRL_VERIFY_W   = 1u;
// LOAD_STATUS (EX §5.1, §3.3)
constexpr uint32_t LOAD_STATUS_VERIFY_BUSY     = 0x00000001u;  // [0]
constexpr uint32_t LOAD_STATUS_VERIFY_BUSY_LSB = 0u;
constexpr uint32_t LOAD_STATUS_VERIFY_BUSY_W   = 1u;
// Q_STATE (EX §5.4): queue state of unit u
constexpr uint32_t Q_STATE_ENTRIES      = 0x0000001Fu;  // [4:0] entries in the queue
constexpr uint32_t Q_STATE_ENTRIES_LSB  = 0u;
constexpr uint32_t Q_STATE_ENTRIES_W    = 5u;
constexpr uint32_t Q_STATE_RUNNING      = 0x00000020u;  // [5] the unit holds an op
constexpr uint32_t Q_STATE_RUNNING_LSB  = 5u;
constexpr uint32_t Q_STATE_RUNNING_W    = 1u;
constexpr uint32_t Q_STATE_DEP_WAIT     = 0x00000040u;  // [6] the head waits for deps
constexpr uint32_t Q_STATE_DEP_WAIT_LSB = 6u;
constexpr uint32_t Q_STATE_DEP_WAIT_W   = 1u;
constexpr uint32_t Q_STATE_HELD         = 0x00000080u;  // [7]
constexpr uint32_t Q_STATE_HELD_LSB     = 7u;
constexpr uint32_t Q_STATE_HELD_W       = 1u;
constexpr uint32_t Q_STATE_ERR          = 0x00000100u;  // [8] error state
constexpr uint32_t Q_STATE_ERR_LSB      = 8u;
constexpr uint32_t Q_STATE_ERR_W        = 1u;
constexpr uint32_t Q_STATE_HUNG         = 0x00000200u;  // [9] watchdog
constexpr uint32_t Q_STATE_HUNG_LSB     = 9u;
constexpr uint32_t Q_STATE_HUNG_W       = 1u;
// Q_FIFO (EX §5.4)
constexpr uint32_t Q_FIFO_ENTRIES     = 0x000000FFu;  // [7:0] issue-FIFO entries waiting for the dispatcher
constexpr uint32_t Q_FIFO_ENTRIES_LSB = 0u;
constexpr uint32_t Q_FIFO_ENTRIES_W   = 8u;
// ADAPTER_CTRL (EX §9.2)
constexpr uint32_t ADAPTER_CTRL_MEM_EN     = 0x00000001u;  // [0] rising edge: latch and alignment-check all bases; requests are held while 0
constexpr uint32_t ADAPTER_CTRL_MEM_EN_LSB = 0u;
constexpr uint32_t ADAPTER_CTRL_MEM_EN_W   = 1u;
// ADAPTER_STATUS (EX §9.2, docs/platform-u200.md §2): sticky, W1C
constexpr uint32_t ADAPTER_STATUS_RRESP_ERR           = 0x00000001u;  // [0] RRESP error
constexpr uint32_t ADAPTER_STATUS_RRESP_ERR_LSB       = 0u;
constexpr uint32_t ADAPTER_STATUS_RRESP_ERR_W         = 1u;
constexpr uint32_t ADAPTER_STATUS_BRESP_ERR           = 0x00000002u;  // [1] BRESP error
constexpr uint32_t ADAPTER_STATUS_BRESP_ERR_LSB       = 1u;
constexpr uint32_t ADAPTER_STATUS_BRESP_ERR_W         = 1u;
constexpr uint32_t ADAPTER_STATUS_ORPHAN_RESP         = 0x00000004u;  // [2] response with no outstanding request for its id
constexpr uint32_t ADAPTER_STATUS_ORPHAN_RESP_LSB     = 2u;
constexpr uint32_t ADAPTER_STATUS_ORPHAN_RESP_W       = 1u;
constexpr uint32_t ADAPTER_STATUS_BASE_MISALIGNED     = 0x00000008u;  // [3] a base was not 4 KiB aligned at the MEM_EN edge
constexpr uint32_t ADAPTER_STATUS_BASE_MISALIGNED_LSB = 3u;
constexpr uint32_t ADAPTER_STATUS_BASE_MISALIGNED_W   = 1u;
constexpr uint32_t ADAPTER_STATUS_REQ_HELD            = 0x00000010u;  // [4] a core memory request was held while MEM_EN = 0
constexpr uint32_t ADAPTER_STATUS_REQ_HELD_LSB        = 4u;
constexpr uint32_t ADAPTER_STATUS_REQ_HELD_W          = 1u;
constexpr uint32_t ADAPTER_STATUS_BASE_LOCKED         = 0x00000020u;  // [5] a base write while MEM_EN = 1 was ignored
constexpr uint32_t ADAPTER_STATUS_BASE_LOCKED_LSB     = 5u;
constexpr uint32_t ADAPTER_STATUS_BASE_LOCKED_W       = 1u;
constexpr uint32_t ADAPTER_STATUS_ERR_COUNT           = 0xFFFF0000u;  // [31:16] saturating count of error responses
constexpr uint32_t ADAPTER_STATUS_ERR_COUNT_LSB       = 16u;
constexpr uint32_t ADAPTER_STATUS_ERR_COUNT_W         = 16u;
// ADAPTER_VERSION (EX §9.2): {16'h0001, 8'h00, NUM_CH}
constexpr uint32_t ADAPTER_VERSION_NUM_CH     = 0x000000FFu;  // [7:0]
constexpr uint32_t ADAPTER_VERSION_NUM_CH_LSB = 0u;
constexpr uint32_t ADAPTER_VERSION_NUM_CH_W   = 8u;
constexpr uint32_t ADAPTER_VERSION_MAJOR      = 0xFFFF0000u;  // [31:16] 1
constexpr uint32_t ADAPTER_VERSION_MAJOR_LSB  = 16u;
constexpr uint32_t ADAPTER_VERSION_MAJOR_W    = 16u;
// HANDLE (EX §4.3): DRAM handle (descriptor args, OPL, DBG_TRW_BASE)
constexpr uint32_t HANDLE_BEAT     = 0x0FFFFFFFu;  // [27:0] beat in the channel
constexpr uint32_t HANDLE_BEAT_LSB = 0u;
constexpr uint32_t HANDLE_BEAT_W   = 28u;
constexpr uint32_t HANDLE_CH       = 0xF0000000u;  // [31:28] channel
constexpr uint32_t HANDLE_CH_LSB   = 28u;
constexpr uint32_t HANDLE_CH_W     = 4u;
// TICKET (CC §7.5): ticket {unit, seq}; unit 15 = none; (u, s) done iff (D[u] - s) mod 2^28 in [1, 2^27)
constexpr uint32_t TICKET_SEQ      = 0x0FFFFFFFu;  // [27:0] per-unit sequence
constexpr uint32_t TICKET_SEQ_LSB  = 0u;
constexpr uint32_t TICKET_SEQ_W    = 28u;
constexpr uint32_t TICKET_UNIT     = 0xF0000000u;  // [31:28] unit id
constexpr uint32_t TICKET_UNIT_LSB = 28u;
constexpr uint32_t TICKET_UNIT_W   = 4u;
// DESC (EX §4.3): descriptor word w0
constexpr uint32_t DESC_TYPE        = 0x000000FFu;  // [7:0] descriptor type; other values reserved (BAD_DESC)
constexpr uint32_t DESC_TYPE_LSB    = 0u;
constexpr uint32_t DESC_TYPE_W      = 8u;
constexpr uint32_t DESC_PROF        = 0x00000100u;  // [8] profiled launch: REC_CFG.ALL and NODE markers for the launch
constexpr uint32_t DESC_PROF_LSB    = 8u;
constexpr uint32_t DESC_PROF_W      = 1u;
constexpr uint32_t DESC_KERNEL      = 0xFFFF0000u;  // [31:16] index into .tsr_kernels (LAUNCH); 0 for PING
constexpr uint32_t DESC_KERNEL_LSB  = 16u;
constexpr uint32_t DESC_KERNEL_W    = 16u;
constexpr uint32_t DESC_TYPE_LAUNCH = 1u;
constexpr uint32_t DESC_TYPE_PING   = 2u;
// LCQ_R0 (EX §4.4): record word r0, one layout for every record type
constexpr uint32_t LCQ_R0_TYPE              = 0x000000FFu;  // [7:0] record type
constexpr uint32_t LCQ_R0_TYPE_LSB          = 0u;
constexpr uint32_t LCQ_R0_TYPE_W            = 8u;
constexpr uint32_t LCQ_R0_STATUS            = 0x0000FF00u;  // [15:8] launch STATUS (DONE), 4 for TRAP; kernel-defined for kernel events
constexpr uint32_t LCQ_R0_STATUS_LSB        = 8u;
constexpr uint32_t LCQ_R0_STATUS_W          = 8u;
constexpr uint32_t LCQ_R0_ERR_CODE          = 0x00FF0000u;  // [23:16] code of the first error, 0 if none
constexpr uint32_t LCQ_R0_ERR_CODE_LSB      = 16u;
constexpr uint32_t LCQ_R0_ERR_CODE_W        = 8u;
constexpr uint32_t LCQ_R0_SEQ               = 0xFF000000u;  // [31:24] SEQ[7:0] of the descriptor
constexpr uint32_t LCQ_R0_SEQ_LSB           = 24u;
constexpr uint32_t LCQ_R0_SEQ_W             = 8u;
constexpr uint32_t LCQ_R0_TYPE_DONE         = 1u;           // answer
constexpr uint32_t LCQ_R0_TYPE_PONG         = 2u;           // answer
constexpr uint32_t LCQ_R0_TYPE_HELLO        = 0x80u;        // event
constexpr uint32_t LCQ_R0_TYPE_TRAP         = 0x81u;        // answer if ACTIVE, else event
constexpr uint32_t LCQ_R0_TYPE_EVENT_FIRST  = 0xC0u;        // kernel events 0xC0..0xFF (ANSWER = 0)
constexpr uint32_t LCQ_R0_TYPE_EVENT_LAST   = 0xFFu;
constexpr uint32_t LCQ_R0_STATUS_OK         = 0u;           // RET >= 0 and no op of the launch failed
constexpr uint32_t LCQ_R0_STATUS_TILE_ERR   = 1u;           // the first error has a code other than ABORTED; r3..r6 describe it
constexpr uint32_t LCQ_R0_STATUS_BAD_DESC   = 2u;           // the descriptor was refused; the kernel did not run
constexpr uint32_t LCQ_R0_STATUS_KERNEL_ERR = 3u;           // RET < 0, no tile error
constexpr uint32_t LCQ_R0_STATUS_TRAP       = 4u;           // TRAP records only
constexpr uint32_t LCQ_R0_STATUS_ABORTED    = 5u;           // cancelled (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8)
constexpr uint32_t LCQ_R0_STATUS_PLAT_ERR   = 6u;           // STATUS.PLAT_ERR was set when the runtime answered
constexpr uint32_t LCQ_R0_STATUS_STOPPED    = 7u;           // the kernel honoured a soft stop
// HELLO_R4 (EX §4.4, SW §6.9): HELLO record word r4
constexpr uint32_t HELLO_R4_SDK_ABI       = 0x0000FFFFu;  // [15:0] Basis ABI
constexpr uint32_t HELLO_R4_SDK_ABI_LSB   = 0u;
constexpr uint32_t HELLO_R4_SDK_ABI_W     = 16u;
constexpr uint32_t HELLO_R4_KINDS_SIG     = 0xFFFF0000u;  // [31:16] kind table signature (opl::kinds_sig)
constexpr uint32_t HELLO_R4_KINDS_SIG_LSB = 16u;
constexpr uint32_t HELLO_R4_KINDS_SIG_W   = 16u;
// HELLO_R6 (EX §4.4, SW §6.9): HELLO record word r6: Tesseract Basis version; [31:24] 0
constexpr uint32_t HELLO_R6_PATCH     = 0x000000FFu;  // [7:0]
constexpr uint32_t HELLO_R6_PATCH_LSB = 0u;
constexpr uint32_t HELLO_R6_PATCH_W   = 8u;
constexpr uint32_t HELLO_R6_MINOR     = 0x0000FF00u;  // [15:8]
constexpr uint32_t HELLO_R6_MINOR_LSB = 8u;
constexpr uint32_t HELLO_R6_MINOR_W   = 8u;
constexpr uint32_t HELLO_R6_MAJOR     = 0x00FF0000u;  // [23:16]
constexpr uint32_t HELLO_R6_MAJOR_LSB = 16u;
constexpr uint32_t HELLO_R6_MAJOR_W   = 8u;
// ISSUE_PUSH (CC §5.1): ISSUE_PUSH write data
constexpr uint32_t ISSUE_PUSH_REC       = 0x00000001u;  // [0] completion record wanted (DB §8.2)
constexpr uint32_t ISSUE_PUSH_REC_LSB   = 0u;
constexpr uint32_t ISSUE_PUSH_REC_W     = 1u;
constexpr uint32_t ISSUE_PUSH_WMASK     = 0x0000FF00u;  // [15:8] word i with bit i clear is sent as 0; w0 is always sent
constexpr uint32_t ISSUE_PUSH_WMASK_LSB = 8u;
constexpr uint32_t ISSUE_PUSH_WMASK_W   = 8u;
// ISSUE_CREDIT (CC §5.1): min(free queue slots, free issue-FIFO entries, 15) per unit
constexpr uint32_t ISSUE_CREDIT_DMA      = 0x0000000Fu;  // [3:0]
constexpr uint32_t ISSUE_CREDIT_DMA_LSB  = 0u;
constexpr uint32_t ISSUE_CREDIT_DMA_W    = 4u;
constexpr uint32_t ISSUE_CREDIT_QACT     = 0x000000F0u;  // [7:4]
constexpr uint32_t ISSUE_CREDIT_QACT_LSB = 4u;
constexpr uint32_t ISSUE_CREDIT_QACT_W   = 4u;
constexpr uint32_t ISSUE_CREDIT_MVM      = 0x00000F00u;  // [11:8]
constexpr uint32_t ISSUE_CREDIT_MVM_LSB  = 8u;
constexpr uint32_t ISSUE_CREDIT_MVM_W    = 4u;
constexpr uint32_t ISSUE_CREDIT_VU       = 0x0000F000u;  // [15:12]
constexpr uint32_t ISSUE_CREDIT_VU_LSB   = 12u;
constexpr uint32_t ISSUE_CREDIT_VU_W     = 4u;
constexpr uint32_t ISSUE_CREDIT_ATT      = 0x000F0000u;  // [19:16]
constexpr uint32_t ISSUE_CREDIT_ATT_LSB  = 16u;
constexpr uint32_t ISSUE_CREDIT_ATT_W    = 4u;
constexpr uint32_t ISSUE_CREDIT_SYNC     = 0x00F00000u;  // [23:20]
constexpr uint32_t ISSUE_CREDIT_SYNC_LSB = 20u;
constexpr uint32_t ISSUE_CREDIT_SYNC_W   = 4u;
// ISSUE_UNITS (CC §5.1): bit u = unit u is built (constant)
constexpr uint32_t ISSUE_UNITS_DMA      = 0x00000001u;  // [0]
constexpr uint32_t ISSUE_UNITS_DMA_LSB  = 0u;
constexpr uint32_t ISSUE_UNITS_DMA_W    = 1u;
constexpr uint32_t ISSUE_UNITS_QACT     = 0x00000002u;  // [1]
constexpr uint32_t ISSUE_UNITS_QACT_LSB = 1u;
constexpr uint32_t ISSUE_UNITS_QACT_W   = 1u;
constexpr uint32_t ISSUE_UNITS_MVM      = 0x00000004u;  // [2]
constexpr uint32_t ISSUE_UNITS_MVM_LSB  = 2u;
constexpr uint32_t ISSUE_UNITS_MVM_W    = 1u;
constexpr uint32_t ISSUE_UNITS_VU       = 0x00000008u;  // [3]
constexpr uint32_t ISSUE_UNITS_VU_LSB   = 3u;
constexpr uint32_t ISSUE_UNITS_VU_W     = 1u;
constexpr uint32_t ISSUE_UNITS_ATT      = 0x00000010u;  // [4] 0 in M6
constexpr uint32_t ISSUE_UNITS_ATT_LSB  = 4u;
constexpr uint32_t ISSUE_UNITS_ATT_W    = 1u;
constexpr uint32_t ISSUE_UNITS_SYNC     = 0x00000020u;  // [5]
constexpr uint32_t ISSUE_UNITS_SYNC_LSB = 5u;
constexpr uint32_t ISSUE_UNITS_SYNC_W   = 1u;
// ISSUE_STATUS (CC §5.1): R/W1C
constexpr uint32_t ISSUE_STATUS_PUSH_TIMEOUT     = 0x00000001u;  // [0] a push waited ISSUE_WAIT_MAX and faulted
constexpr uint32_t ISSUE_STATUS_PUSH_TIMEOUT_LSB = 0u;
constexpr uint32_t ISSUE_STATUS_PUSH_TIMEOUT_W   = 1u;
constexpr uint32_t ISSUE_STATUS_ERR_IRQ          = 0x00000002u;  // [1] sticky, set on the rising edge of 'some unit in error state' (irq_fast[1])
constexpr uint32_t ISSUE_STATUS_ERR_IRQ_LSB      = 1u;
constexpr uint32_t ISSUE_STATUS_ERR_IRQ_W        = 1u;
// QCTRL (CC §5.1): one bit per write, in push order through the issue FIFO
constexpr uint32_t QCTRL_QABORT        = 0x00000001u;  // [0]
constexpr uint32_t QCTRL_QABORT_LSB    = 0u;
constexpr uint32_t QCTRL_QABORT_W      = 1u;
constexpr uint32_t QCTRL_QHOLD_SET     = 0x00000002u;  // [1] set the RISC-V hold source
constexpr uint32_t QCTRL_QHOLD_SET_LSB = 1u;
constexpr uint32_t QCTRL_QHOLD_SET_W   = 1u;
constexpr uint32_t QCTRL_QHOLD_CLR     = 0x00000004u;  // [2] clear the RISC-V hold source
constexpr uint32_t QCTRL_QHOLD_CLR_LSB = 2u;
constexpr uint32_t QCTRL_QHOLD_CLR_W   = 1u;
constexpr uint32_t QCTRL_QCLEAR        = 0x00000008u;  // [3] gets a SYNC ticket (ISSUE_TICKET)
constexpr uint32_t QCTRL_QCLEAR_LSB    = 3u;
constexpr uint32_t QCTRL_QCLEAR_W      = 1u;
// QSTATUS (CC §5.1): queue status (RISC-V view)
constexpr uint32_t QSTATUS_QIDLE         = 0x00000001u;  // [0] ISSUED_SEQ = DONE_SEQ for every u and the issue FIFO empty
constexpr uint32_t QSTATUS_QIDLE_LSB     = 0u;
constexpr uint32_t QSTATUS_QIDLE_W       = 1u;
constexpr uint32_t QSTATUS_HOLD          = 0x00000002u;  // [1] a hold is in effect
constexpr uint32_t QSTATUS_HOLD_LSB      = 1u;
constexpr uint32_t QSTATUS_HOLD_W        = 1u;
constexpr uint32_t QSTATUS_HOLD_HOST     = 0x00000004u;  // [2] host hold
constexpr uint32_t QSTATUS_HOLD_HOST_LSB = 2u;
constexpr uint32_t QSTATUS_HOLD_HOST_W   = 1u;
constexpr uint32_t QSTATUS_HOLD_RV       = 0x00000008u;  // [3] RISC-V hold
constexpr uint32_t QSTATUS_HOLD_RV_LSB   = 3u;
constexpr uint32_t QSTATUS_HOLD_RV_W     = 1u;
constexpr uint32_t QSTATUS_HOLD_HALT     = 0x00000010u;  // [4] debug-halt hold
constexpr uint32_t QSTATUS_HOLD_HALT_LSB = 4u;
constexpr uint32_t QSTATUS_HOLD_HALT_W   = 1u;
constexpr uint32_t QSTATUS_ERR           = 0x00003F00u;  // [13:8] error state of unit u at bit 8 + u
constexpr uint32_t QSTATUS_ERR_LSB       = 8u;
constexpr uint32_t QSTATUS_ERR_W         = 6u;
constexpr uint32_t QSTATUS_NONEMPTY      = 0x003F0000u;  // [21:16] queue u non-empty at bit 16 + u
constexpr uint32_t QSTATUS_NONEMPTY_LSB  = 16u;
constexpr uint32_t QSTATUS_NONEMPTY_W    = 6u;
// WAIT_CREDIT (CC §5.1): irq_fast[5] while the unit's credit != 0
constexpr uint32_t WAIT_CREDIT_UNIT     = 0x00000007u;  // [2:0]
constexpr uint32_t WAIT_CREDIT_UNIT_LSB = 0u;
constexpr uint32_t WAIT_CREDIT_UNIT_W   = 3u;
constexpr uint32_t WAIT_CREDIT_EN       = 0x80000000u;  // [31]
constexpr uint32_t WAIT_CREDIT_EN_LSB   = 31u;
constexpr uint32_t WAIT_CREDIT_EN_W     = 1u;
// LQ_FLAGS (EX §4.5): RISC-V view of the launch state
constexpr uint32_t LQ_FLAGS_CANCEL_VALID     = 0x00000001u;  // [0]
constexpr uint32_t LQ_FLAGS_CANCEL_VALID_LSB = 0u;
constexpr uint32_t LQ_FLAGS_CANCEL_VALID_W   = 1u;
constexpr uint32_t LQ_FLAGS_STOP_VALID       = 0x00000002u;  // [1]
constexpr uint32_t LQ_FLAGS_STOP_VALID_LSB   = 1u;
constexpr uint32_t LQ_FLAGS_STOP_VALID_W     = 1u;
constexpr uint32_t LQ_FLAGS_ACTIVE           = 0x00000004u;  // [2]
constexpr uint32_t LQ_FLAGS_ACTIVE_LSB       = 2u;
constexpr uint32_t LQ_FLAGS_ACTIVE_W         = 1u;
constexpr uint32_t LQ_FLAGS_PLAT_ERR         = 0x00000008u;  // [3] STATUS.PLAT_ERR
constexpr uint32_t LQ_FLAGS_PLAT_ERR_LSB     = 3u;
constexpr uint32_t LQ_FLAGS_PLAT_ERR_W       = 1u;
// LCQ_POST (EX §4.5): LCQ_POST write data
constexpr uint32_t LCQ_POST_ANSWER     = 0x00000001u;  // [0] the record answers the running launch (ANSWERED += 1)
constexpr uint32_t LCQ_POST_ANSWER_LSB = 0u;
constexpr uint32_t LCQ_POST_ANSWER_W   = 1u;
// REC_CFG (DB §10.2, §8.2): completion-record configuration
constexpr uint32_t REC_CFG_ALL          = 0x00000001u;  // [0] a record for every entry, not only ISSUE_PUSH.REC ones
constexpr uint32_t REC_CFG_ALL_LSB      = 0u;
constexpr uint32_t REC_CFG_ALL_W        = 1u;
constexpr uint32_t REC_CFG_IRQ_HALF     = 0x00000002u;  // [1] irq_fast[4] while REC_COUNT >= 32
constexpr uint32_t REC_CFG_IRQ_HALF_LSB = 1u;
constexpr uint32_t REC_CFG_IRQ_HALF_W   = 1u;
// REC_DATA3 (DB §8.2): completion record word 3
constexpr uint32_t REC_DATA3_CODE       = 0x0000000Fu;  // [3:0] completion code
constexpr uint32_t REC_DATA3_CODE_LSB   = 0u;
constexpr uint32_t REC_DATA3_CODE_W     = 4u;
constexpr uint32_t REC_DATA3_RAN        = 0x00000010u;  // [4]
constexpr uint32_t REC_DATA3_RAN_LSB    = 4u;
constexpr uint32_t REC_DATA3_RAN_W      = 1u;
constexpr uint32_t REC_DATA3_MARKER     = 0x00000020u;  // [5]
constexpr uint32_t REC_DATA3_MARKER_LSB = 5u;
constexpr uint32_t REC_DATA3_MARKER_W   = 1u;
constexpr uint32_t REC_DATA3_OPCODE     = 0x0000FF00u;  // [15:8]
constexpr uint32_t REC_DATA3_OPCODE_LSB = 8u;
constexpr uint32_t REC_DATA3_OPCODE_W   = 8u;
constexpr uint32_t REC_DATA3_PIDX       = 0xFFFF0000u;  // [31:16] push index [15:0]
constexpr uint32_t REC_DATA3_PIDX_LSB   = 16u;
constexpr uint32_t REC_DATA3_PIDX_W     = 16u;
// DBG_FLAGS (DB §10.2): RISC-V view of DB's state (control window 0x400)
constexpr uint32_t DBG_FLAGS_DBG_STOP         = 0x00000001u;  // [0] a DB stop or the debug-halt hold is in effect
constexpr uint32_t DBG_FLAGS_DBG_STOP_LSB     = 0u;
constexpr uint32_t DBG_FLAGS_DBG_STOP_W       = 1u;
constexpr uint32_t DBG_FLAGS_SERIAL           = 0x00000002u;  // [1]
constexpr uint32_t DBG_FLAGS_SERIAL_LSB       = 1u;
constexpr uint32_t DBG_FLAGS_SERIAL_W         = 1u;
constexpr uint32_t DBG_FLAGS_TRW_EN           = 0x00000004u;  // [2]
constexpr uint32_t DBG_FLAGS_TRW_EN_LSB       = 2u;
constexpr uint32_t DBG_FLAGS_TRW_EN_W         = 1u;
constexpr uint32_t DBG_FLAGS_DMI_EN           = 0x00000008u;  // [3] a debugger may be attached
constexpr uint32_t DBG_FLAGS_DMI_EN_LSB       = 3u;
constexpr uint32_t DBG_FLAGS_DMI_EN_W         = 1u;
constexpr uint32_t DBG_FLAGS_HALT_ON_BOOT     = 0x00000010u;  // [4]
constexpr uint32_t DBG_FLAGS_HALT_ON_BOOT_LSB = 4u;
constexpr uint32_t DBG_FLAGS_HALT_ON_BOOT_W   = 1u;
// MARK (DB §8.4): TRACE_MARK value
constexpr uint32_t MARK_PAYLOAD            = 0x0FFFFFFFu;  // [27:0]
constexpr uint32_t MARK_PAYLOAD_LSB        = 0u;
constexpr uint32_t MARK_PAYLOAD_W          = 28u;
constexpr uint32_t MARK_CLASS              = 0xF0000000u;  // [31:28]
constexpr uint32_t MARK_CLASS_LSB          = 28u;
constexpr uint32_t MARK_CLASS_W            = 4u;
constexpr uint32_t MARK_CLASS_USER         = 0u;           // any payload
constexpr uint32_t MARK_CLASS_LAUNCH_BEGIN = 1u;           // payload: layout MARK_LAUNCH
constexpr uint32_t MARK_CLASS_LAUNCH_END   = 2u;           // payload: layout MARK_LAUNCH
constexpr uint32_t MARK_CLASS_NODE_BEGIN   = 3u;           // payload: op-list index
constexpr uint32_t MARK_CLASS_NODE_END     = 4u;           // payload: op-list index
constexpr uint32_t MARK_CLASS_LAYER        = 5u;           // payload: layer index
constexpr uint32_t MARK_CLASS_PHASE        = 6u;           // payload: kernel-defined id (.tsr_marks)
// MARK_LAUNCH (DB §8.4): payload of LAUNCH_BEGIN / LAUNCH_END markers
constexpr uint32_t MARK_LAUNCH_SEQ        = 0x0000FFFFu;  // [15:0] launch SEQ low bits
constexpr uint32_t MARK_LAUNCH_SEQ_LSB    = 0u;
constexpr uint32_t MARK_LAUNCH_SEQ_W      = 16u;
constexpr uint32_t MARK_LAUNCH_KERNEL     = 0x0FFF0000u;  // [27:16] kernel index
constexpr uint32_t MARK_LAUNCH_KERNEL_LSB = 16u;
constexpr uint32_t MARK_LAUNCH_KERNEL_W   = 12u;
constexpr uint32_t MARK_LAUNCH_CLASS      = 0xF0000000u;  // [31:28]
constexpr uint32_t MARK_LAUNCH_CLASS_LSB  = 28u;
constexpr uint32_t MARK_LAUNCH_CLASS_W    = 4u;
// LOG_HDR (DB §6.1): log record header (the only definition; EX §4.10, SW §6.6)
constexpr uint32_t LOG_HDR_N              = 0x000000FFu;  // [7:0] payload words after the header
constexpr uint32_t LOG_HDR_N_LSB          = 0u;
constexpr uint32_t LOG_HDR_N_W            = 8u;
constexpr uint32_t LOG_HDR_LEVEL          = 0x00000F00u;  // [11:8]
constexpr uint32_t LOG_HDR_LEVEL_LSB      = 8u;
constexpr uint32_t LOG_HDR_LEVEL_W        = 4u;
constexpr uint32_t LOG_HDR_SOURCE         = 0x0000F000u;  // [15:12]
constexpr uint32_t LOG_HDR_SOURCE_LSB     = 12u;
constexpr uint32_t LOG_HDR_SOURCE_W       = 4u;
constexpr uint32_t LOG_HDR_TYPE           = 0x00070000u;  // [18:16] 5..7 invalid
constexpr uint32_t LOG_HDR_TYPE_LSB       = 16u;
constexpr uint32_t LOG_HDR_TYPE_W         = 3u;
constexpr uint32_t LOG_HDR_TS             = 0x00080000u;  // [19] the first payload word is MTIME[31:0]
constexpr uint32_t LOG_HDR_TS_LSB         = 19u;
constexpr uint32_t LOG_HDR_TS_W           = 1u;
constexpr uint32_t LOG_HDR_LSEQ           = 0x0FF00000u;  // [27:20] SEQ[7:0] of the running launch; 0 outside launches
constexpr uint32_t LOG_HDR_LSEQ_LSB       = 20u;
constexpr uint32_t LOG_HDR_LSEQ_W         = 8u;
constexpr uint32_t LOG_HDR_SYNC           = 0xF0000000u;  // [31:28] 0xA
constexpr uint32_t LOG_HDR_SYNC_LSB       = 28u;
constexpr uint32_t LOG_HDR_SYNC_W         = 4u;
constexpr uint32_t LOG_HDR_LEVEL_FATAL    = 0u;
constexpr uint32_t LOG_HDR_LEVEL_ERROR    = 1u;
constexpr uint32_t LOG_HDR_LEVEL_WARN     = 2u;
constexpr uint32_t LOG_HDR_LEVEL_INFO     = 3u;
constexpr uint32_t LOG_HDR_LEVEL_DEBUG    = 4u;
constexpr uint32_t LOG_HDR_LEVEL_TRACE    = 5u;
constexpr uint32_t LOG_HDR_SOURCE_RUNTIME = 0u;
constexpr uint32_t LOG_HDR_SOURCE_KERNEL  = 1u;
constexpr uint32_t LOG_HDR_SOURCE_TRAP    = 2u;           // trap handler
constexpr uint32_t LOG_HDR_SOURCE_DEBUG   = 3u;           // debug kernels
constexpr uint32_t LOG_HDR_TYPE_FMT       = 0u;           // format-string address in .tsr_logfmt, then arguments
constexpr uint32_t LOG_HDR_TYPE_TEXT      = 1u;           // UTF-8, zero-padded
constexpr uint32_t LOG_HDR_TYPE_TRAP      = 2u;           // the 44 words of __tsr_crash
constexpr uint32_t LOG_HDR_TYPE_EVENT     = 3u;           // event code (marker encoding), arguments
constexpr uint32_t LOG_HDR_TYPE_BLOB      = 4u;           // tag, raw words
constexpr uint32_t LOG_HDR_SYNC_MAGIC     = 10u;
// TRW_W0 (DB §8.3): trace-writer record word 0
constexpr uint32_t TRW_W0_KIND        = 0x0000000Fu;  // [3:0]
constexpr uint32_t TRW_W0_KIND_LSB    = 0u;
constexpr uint32_t TRW_W0_KIND_W      = 4u;
constexpr uint32_t TRW_W0_UNIT        = 0x000000F0u;  // [7:4]
constexpr uint32_t TRW_W0_UNIT_LSB    = 4u;
constexpr uint32_t TRW_W0_UNIT_W      = 4u;
constexpr uint32_t TRW_W0_CODE        = 0x00000F00u;  // [11:8]
constexpr uint32_t TRW_W0_CODE_LSB    = 8u;
constexpr uint32_t TRW_W0_CODE_W      = 4u;
constexpr uint32_t TRW_W0_RAN         = 0x00001000u;  // [12]
constexpr uint32_t TRW_W0_RAN_LSB     = 12u;
constexpr uint32_t TRW_W0_RAN_W       = 1u;
constexpr uint32_t TRW_W0_REC         = 0x00002000u;  // [13] REC flag
constexpr uint32_t TRW_W0_REC_LSB     = 13u;
constexpr uint32_t TRW_W0_REC_W       = 1u;
constexpr uint32_t TRW_W0_TBP         = 0x00004000u;  // [14] tile-breakpoint hit
constexpr uint32_t TRW_W0_TBP_LSB     = 14u;
constexpr uint32_t TRW_W0_TBP_W       = 1u;
constexpr uint32_t TRW_W0_OPCODE      = 0x00FF0000u;  // [23:16]
constexpr uint32_t TRW_W0_OPCODE_LSB  = 16u;
constexpr uint32_t TRW_W0_OPCODE_W    = 8u;
constexpr uint32_t TRW_W0_W0_HI       = 0xFF000000u;  // [31:24] w0[15:8] of the op
constexpr uint32_t TRW_W0_W0_HI_LSB   = 24u;
constexpr uint32_t TRW_W0_W0_HI_W     = 8u;
constexpr uint32_t TRW_W0_KIND_OP     = 0u;
constexpr uint32_t TRW_W0_KIND_MARK   = 1u;
constexpr uint32_t TRW_W0_KIND_STOP   = 2u;
constexpr uint32_t TRW_W0_KIND_RVHALT = 3u;
constexpr uint32_t TRW_W0_KIND_DROP   = 4u;
constexpr uint32_t TRW_W0_KIND_PAD    = 15u;          // ignored
// DBG_ID (DB §10.1): 0 = block absent
constexpr uint32_t DBG_ID_VERSION       = 0x000000FFu;  // [7:0] 1
constexpr uint32_t DBG_ID_VERSION_LSB   = 0u;
constexpr uint32_t DBG_ID_VERSION_W     = 8u;
constexpr uint32_t DBG_ID_HIST          = 0x00000100u;  // [8] history
constexpr uint32_t DBG_ID_HIST_LSB      = 8u;
constexpr uint32_t DBG_ID_HIST_W        = 1u;
constexpr uint32_t DBG_ID_TRW           = 0x00000200u;  // [9] trace writer
constexpr uint32_t DBG_ID_TRW_LSB       = 9u;
constexpr uint32_t DBG_ID_TRW_W         = 1u;
constexpr uint32_t DBG_ID_RVWP          = 0x00000400u;  // [10] RV watchpoints
constexpr uint32_t DBG_ID_RVWP_LSB      = 10u;
constexpr uint32_t DBG_ID_RVWP_W        = 1u;
constexpr uint32_t DBG_ID_REC           = 0x00000800u;  // [11] completion records
constexpr uint32_t DBG_ID_REC_LSB       = 11u;
constexpr uint32_t DBG_ID_REC_W         = 1u;
constexpr uint32_t DBG_ID_NTBP          = 0x00007000u;  // [14:12] tile breakpoints
constexpr uint32_t DBG_ID_NTBP_LSB      = 12u;
constexpr uint32_t DBG_ID_NTBP_W        = 3u;
constexpr uint32_t DBG_ID_NRVWP         = 0x00018000u;  // [16:15] RV watchpoints
constexpr uint32_t DBG_ID_NRVWP_LSB     = 15u;
constexpr uint32_t DBG_ID_NRVWP_W       = 2u;
constexpr uint32_t DBG_ID_LOG2_HIST     = 0x00F00000u;  // [23:20] log2 HIST_DEPTH
constexpr uint32_t DBG_ID_LOG2_HIST_LSB = 20u;
constexpr uint32_t DBG_ID_LOG2_HIST_W   = 4u;
// DBG_CTRL (DB §10.1): pulses; one of bits 0..2 per write (QSTEP > RELEASE > STOP_NOW)
constexpr uint32_t DBG_CTRL_STOP_NOW           = 0x00000001u;  // [0]
constexpr uint32_t DBG_CTRL_STOP_NOW_LSB       = 0u;
constexpr uint32_t DBG_CTRL_STOP_NOW_W         = 1u;
constexpr uint32_t DBG_CTRL_RELEASE            = 0x00000002u;  // [1]
constexpr uint32_t DBG_CTRL_RELEASE_LSB        = 1u;
constexpr uint32_t DBG_CTRL_RELEASE_W          = 1u;
constexpr uint32_t DBG_CTRL_QSTEP              = 0x00000004u;  // [2]
constexpr uint32_t DBG_CTRL_QSTEP_LSB          = 2u;
constexpr uint32_t DBG_CTRL_QSTEP_W            = 1u;
constexpr uint32_t DBG_CTRL_HIST_CLEAR         = 0x00000010u;  // [4]
constexpr uint32_t DBG_CTRL_HIST_CLEAR_LSB     = 4u;
constexpr uint32_t DBG_CTRL_HIST_CLEAR_W       = 1u;
constexpr uint32_t DBG_CTRL_RV_SAMPLE          = 0x00000020u;  // [5] completes after the sample or 4,096 cycles
constexpr uint32_t DBG_CTRL_RV_SAMPLE_LSB      = 5u;
constexpr uint32_t DBG_CTRL_RV_SAMPLE_W        = 1u;
constexpr uint32_t DBG_CTRL_RVWP_APPLY         = 0x00000040u;  // [6]
constexpr uint32_t DBG_CTRL_RVWP_APPLY_LSB     = 6u;
constexpr uint32_t DBG_CTRL_RVWP_APPLY_W       = 1u;
constexpr uint32_t DBG_CTRL_TRW_FLUSH          = 0x00000080u;  // [7]
constexpr uint32_t DBG_CTRL_TRW_FLUSH_LSB      = 7u;
constexpr uint32_t DBG_CTRL_TRW_FLUSH_W        = 1u;
constexpr uint32_t DBG_CTRL_RV_HALT_CANCEL     = 0x00000100u;  // [8]
constexpr uint32_t DBG_CTRL_RV_HALT_CANCEL_LSB = 8u;
constexpr uint32_t DBG_CTRL_RV_HALT_CANCEL_W   = 1u;
constexpr uint32_t DBG_CTRL_MATCHQ_CLEAR       = 0x00000200u;  // [9]
constexpr uint32_t DBG_CTRL_MATCHQ_CLEAR_LSB   = 9u;
constexpr uint32_t DBG_CTRL_MATCHQ_CLEAR_W     = 1u;
// DBG_CFG (DB §10.1): reset 0x30
constexpr uint32_t DBG_CFG_DMI_EN              = 0x00000001u;  // [0] gates the DMI window
constexpr uint32_t DBG_CFG_DMI_EN_LSB          = 0u;
constexpr uint32_t DBG_CFG_DMI_EN_W            = 1u;
constexpr uint32_t DBG_CFG_SERIAL              = 0x00000002u;  // [1]
constexpr uint32_t DBG_CFG_SERIAL_LSB          = 1u;
constexpr uint32_t DBG_CFG_SERIAL_W            = 1u;
constexpr uint32_t DBG_CFG_ERR_STOP            = 0x00000004u;  // [2]
constexpr uint32_t DBG_CFG_ERR_STOP_LSB        = 2u;
constexpr uint32_t DBG_CFG_ERR_STOP_W          = 1u;
constexpr uint32_t DBG_CFG_STOP_HALT_RV        = 0x00000008u;  // [3]
constexpr uint32_t DBG_CFG_STOP_HALT_RV_LSB    = 3u;
constexpr uint32_t DBG_CFG_STOP_HALT_RV_W      = 1u;
constexpr uint32_t DBG_CFG_HIST_EN             = 0x00000010u;  // [4]
constexpr uint32_t DBG_CFG_HIST_EN_LSB         = 4u;
constexpr uint32_t DBG_CFG_HIST_EN_W           = 1u;
constexpr uint32_t DBG_CFG_HIST_FREEZE_ERR     = 0x00000020u;  // [5]
constexpr uint32_t DBG_CFG_HIST_FREEZE_ERR_LSB = 5u;
constexpr uint32_t DBG_CFG_HIST_FREEZE_ERR_W   = 1u;
constexpr uint32_t DBG_CFG_HIST_FREEZE         = 0x00000040u;  // [6]
constexpr uint32_t DBG_CFG_HIST_FREEZE_LSB     = 6u;
constexpr uint32_t DBG_CFG_HIST_FREEZE_W       = 1u;
constexpr uint32_t DBG_CFG_HUNG_STOP           = 0x00000080u;  // [7]
constexpr uint32_t DBG_CFG_HUNG_STOP_LSB       = 7u;
constexpr uint32_t DBG_CFG_HUNG_STOP_W         = 1u;
constexpr uint32_t DBG_CFG_HALT_ON_BOOT        = 0x00000100u;  // [8]
constexpr uint32_t DBG_CFG_HALT_ON_BOOT_LSB    = 8u;
constexpr uint32_t DBG_CFG_HALT_ON_BOOT_W      = 1u;
constexpr uint32_t DBG_CFG_RV_HALT_EN          = 0x00000200u;  // [9]
constexpr uint32_t DBG_CFG_RV_HALT_EN_LSB      = 9u;
constexpr uint32_t DBG_CFG_RV_HALT_EN_W        = 1u;
constexpr uint32_t DBG_CFG_TRW_EN              = 0x00000400u;  // [10]
constexpr uint32_t DBG_CFG_TRW_EN_LSB          = 10u;
constexpr uint32_t DBG_CFG_TRW_EN_W            = 1u;
constexpr uint32_t DBG_CFG_TRW_STOP_FULL       = 0x00000800u;  // [11]
constexpr uint32_t DBG_CFG_TRW_STOP_FULL_LSB   = 11u;
constexpr uint32_t DBG_CFG_TRW_STOP_FULL_W     = 1u;
// DBG_STATUS (DB §10.1): levels and sticky W1C bits
constexpr uint32_t DBG_STATUS_STOP_ACTIVE           = 0x00000001u;  // [0]
constexpr uint32_t DBG_STATUS_STOP_ACTIVE_LSB       = 0u;
constexpr uint32_t DBG_STATUS_STOP_ACTIVE_W         = 1u;
constexpr uint32_t DBG_STATUS_DRAINED               = 0x00000002u;  // [1]
constexpr uint32_t DBG_STATUS_DRAINED_LSB           = 1u;
constexpr uint32_t DBG_STATUS_DRAINED_W             = 1u;
constexpr uint32_t DBG_STATUS_STEP_BUSY             = 0x00000004u;  // [2]
constexpr uint32_t DBG_STATUS_STEP_BUSY_LSB         = 2u;
constexpr uint32_t DBG_STATUS_STEP_BUSY_W           = 1u;
constexpr uint32_t DBG_STATUS_DB_HELD               = 0x00000008u;  // [3] a head held by DB
constexpr uint32_t DBG_STATUS_DB_HELD_LSB           = 3u;
constexpr uint32_t DBG_STATUS_DB_HELD_W             = 1u;
constexpr uint32_t DBG_STATUS_CAUSE                 = 0x000001F0u;  // [8:4] cause of the current stop (layout DBG_CAUSE)
constexpr uint32_t DBG_STATUS_CAUSE_LSB             = 4u;
constexpr uint32_t DBG_STATUS_CAUSE_W               = 5u;
constexpr uint32_t DBG_STATUS_BP_HIT                = 0x00001E00u;  // [12:9] TBPs hit
constexpr uint32_t DBG_STATUS_BP_HIT_LSB            = 9u;
constexpr uint32_t DBG_STATUS_BP_HIT_W              = 4u;
constexpr uint32_t DBG_STATUS_RVWP_HIT              = 0x00006000u;  // [14:13] sticky
constexpr uint32_t DBG_STATUS_RVWP_HIT_LSB          = 13u;
constexpr uint32_t DBG_STATUS_RVWP_HIT_W            = 2u;
constexpr uint32_t DBG_STATUS_RV_HALT_PEND          = 0x00008000u;  // [15]
constexpr uint32_t DBG_STATUS_RV_HALT_PEND_LSB      = 15u;
constexpr uint32_t DBG_STATUS_RV_HALT_PEND_W        = 1u;
constexpr uint32_t DBG_STATUS_RVWP_BUSY             = 0x00010000u;  // [16]
constexpr uint32_t DBG_STATUS_RVWP_BUSY_LSB         = 16u;
constexpr uint32_t DBG_STATUS_RVWP_BUSY_W           = 1u;
constexpr uint32_t DBG_STATUS_HIST_FROZEN           = 0x00020000u;  // [17]
constexpr uint32_t DBG_STATUS_HIST_FROZEN_LSB       = 17u;
constexpr uint32_t DBG_STATUS_HIST_FROZEN_W         = 1u;
constexpr uint32_t DBG_STATUS_RV_HALTED             = 0x00040000u;  // [18]
constexpr uint32_t DBG_STATUS_RV_HALTED_LSB         = 18u;
constexpr uint32_t DBG_STATUS_RV_HALTED_W           = 1u;
constexpr uint32_t DBG_STATUS_RV_SLEEP              = 0x00080000u;  // [19]
constexpr uint32_t DBG_STATUS_RV_SLEEP_LSB          = 19u;
constexpr uint32_t DBG_STATUS_RV_SLEEP_W            = 1u;
constexpr uint32_t DBG_STATUS_TRW_DROP              = 0x00100000u;  // [20] sticky
constexpr uint32_t DBG_STATUS_TRW_DROP_LSB          = 20u;
constexpr uint32_t DBG_STATUS_TRW_DROP_W            = 1u;
constexpr uint32_t DBG_STATUS_REC_DROP              = 0x00200000u;  // [21] sticky
constexpr uint32_t DBG_STATUS_REC_DROP_LSB          = 21u;
constexpr uint32_t DBG_STATUS_REC_DROP_W            = 1u;
constexpr uint32_t DBG_STATUS_SBA_REFUSED           = 0x00400000u;  // [22] sticky
constexpr uint32_t DBG_STATUS_SBA_REFUSED_LSB       = 22u;
constexpr uint32_t DBG_STATUS_SBA_REFUSED_W         = 1u;
constexpr uint32_t DBG_STATUS_DMI_BLOCKED           = 0x00800000u;  // [23] sticky
constexpr uint32_t DBG_STATUS_DMI_BLOCKED_LSB       = 23u;
constexpr uint32_t DBG_STATUS_DMI_BLOCKED_W         = 1u;
constexpr uint32_t DBG_STATUS_RV_SAMPLE_TIMEOUT     = 0x01000000u;  // [24] sticky
constexpr uint32_t DBG_STATUS_RV_SAMPLE_TIMEOUT_LSB = 24u;
constexpr uint32_t DBG_STATUS_RV_SAMPLE_TIMEOUT_W   = 1u;
constexpr uint32_t DBG_STATUS_RV_DFAULT             = 0x02000000u;  // [25] sticky
constexpr uint32_t DBG_STATUS_RV_DFAULT_LSB         = 25u;
constexpr uint32_t DBG_STATUS_RV_DFAULT_W           = 1u;
constexpr uint32_t DBG_STATUS_PTR_ERR               = 0x04000000u;  // [26] sticky
constexpr uint32_t DBG_STATUS_PTR_ERR_LSB           = 26u;
constexpr uint32_t DBG_STATUS_PTR_ERR_W             = 1u;
constexpr uint32_t DBG_STATUS_RVWP_TIMEOUT          = 0x08000000u;  // [27] sticky
constexpr uint32_t DBG_STATUS_RVWP_TIMEOUT_LSB      = 27u;
constexpr uint32_t DBG_STATUS_RVWP_TIMEOUT_W        = 1u;
constexpr uint32_t DBG_STATUS_MATCHQ_OVF            = 0x10000000u;  // [28] sticky, never expected
constexpr uint32_t DBG_STATUS_MATCHQ_OVF_LSB        = 28u;
constexpr uint32_t DBG_STATUS_MATCHQ_OVF_W          = 1u;
constexpr uint32_t DBG_STATUS_DMI_RESP_ERR          = 0x20000000u;  // [29] sticky
constexpr uint32_t DBG_STATUS_DMI_RESP_ERR_LSB      = 29u;
constexpr uint32_t DBG_STATUS_DMI_RESP_ERR_W        = 1u;
constexpr uint32_t DBG_STATUS_S_VALID               = 0x80000000u;  // [31]
constexpr uint32_t DBG_STATUS_S_VALID_LSB           = 31u;
constexpr uint32_t DBG_STATUS_S_VALID_W             = 1u;
// DBG_CAUSE (DB §10.1, §5.2): bits of DBG_STATUS.CAUSE in the documented order (one bit per cause)
constexpr uint32_t DBG_CAUSE_NOW      = 0x00000001u;  // [0]
constexpr uint32_t DBG_CAUSE_NOW_LSB  = 0u;
constexpr uint32_t DBG_CAUSE_NOW_W    = 1u;
constexpr uint32_t DBG_CAUSE_BP       = 0x00000002u;  // [1]
constexpr uint32_t DBG_CAUSE_BP_LSB   = 1u;
constexpr uint32_t DBG_CAUSE_BP_W     = 1u;
constexpr uint32_t DBG_CAUSE_STEP     = 0x00000004u;  // [2]
constexpr uint32_t DBG_CAUSE_STEP_LSB = 2u;
constexpr uint32_t DBG_CAUSE_STEP_W   = 1u;
constexpr uint32_t DBG_CAUSE_ERR      = 0x00000008u;  // [3]
constexpr uint32_t DBG_CAUSE_ERR_LSB  = 3u;
constexpr uint32_t DBG_CAUSE_ERR_W    = 1u;
constexpr uint32_t DBG_CAUSE_HUNG     = 0x00000010u;  // [4]
constexpr uint32_t DBG_CAUSE_HUNG_LSB = 4u;
constexpr uint32_t DBG_CAUSE_HUNG_W   = 1u;
// DBG_STEP_N (DB §10.1)
constexpr uint32_t DBG_STEP_N_N     = 0x0000FFFFu;  // [15:0] entries per QSTEP (0 = 1)
constexpr uint32_t DBG_STEP_N_N_LSB = 0u;
constexpr uint32_t DBG_STEP_N_N_W   = 16u;
// DBG_MATCHQ (DB §10.1)
constexpr uint32_t DBG_MATCHQ_PENDING     = 0x0000007Fu;  // [6:0] pending match thresholds
constexpr uint32_t DBG_MATCHQ_PENDING_LSB = 0u;
constexpr uint32_t DBG_MATCHQ_PENDING_W   = 7u;
// DBG_QSTATE (DB §10.1, CC §8.3): dispatcher state
constexpr uint32_t DBG_QSTATE_DISP            = 0x00000700u;  // [10:8] dispatcher state (CC §8.3 order)
constexpr uint32_t DBG_QSTATE_DISP_LSB        = 8u;
constexpr uint32_t DBG_QSTATE_DISP_W          = 3u;
constexpr uint32_t DBG_QSTATE_OUTSTANDING     = 0x00FF0000u;  // [23:16] outstanding entries, saturating
constexpr uint32_t DBG_QSTATE_OUTSTANDING_LSB = 16u;
constexpr uint32_t DBG_QSTATE_OUTSTANDING_W   = 8u;
constexpr uint32_t DBG_QSTATE_DISP_IDLE       = 0u;           // D_IDLE
constexpr uint32_t DBG_QSTATE_DISP_POP        = 1u;           // D_POP
constexpr uint32_t DBG_QSTATE_DISP_CHK        = 2u;           // D_CHK
constexpr uint32_t DBG_QSTATE_DISP_ENQ        = 3u;           // D_ENQ
// DBG_RV_STATE (DB §10.1)
constexpr uint32_t DBG_RV_STATE_HALTED            = 0x00000001u;  // [0]
constexpr uint32_t DBG_RV_STATE_HALTED_LSB        = 0u;
constexpr uint32_t DBG_RV_STATE_HALTED_W          = 1u;
constexpr uint32_t DBG_RV_STATE_SLEEP             = 0x00000004u;  // [2]
constexpr uint32_t DBG_RV_STATE_SLEEP_LSB         = 2u;
constexpr uint32_t DBG_RV_STATE_SLEEP_W           = 1u;
constexpr uint32_t DBG_RV_STATE_IN_RESET          = 0x00000008u;  // [3]
constexpr uint32_t DBG_RV_STATE_IN_RESET_LSB      = 3u;
constexpr uint32_t DBG_RV_STATE_IN_RESET_W        = 1u;
constexpr uint32_t DBG_RV_STATE_SAMPLE_VALID      = 0x00000010u;  // [4]
constexpr uint32_t DBG_RV_STATE_SAMPLE_VALID_LSB  = 4u;
constexpr uint32_t DBG_RV_STATE_SAMPLE_VALID_W    = 1u;
constexpr uint32_t DBG_RV_STATE_DMACTIVE          = 0x00000020u;  // [5] dmactive_o
constexpr uint32_t DBG_RV_STATE_DMACTIVE_LSB      = 5u;
constexpr uint32_t DBG_RV_STATE_DMACTIVE_W        = 1u;
constexpr uint32_t DBG_RV_STATE_NDMRESET          = 0x00000040u;  // [6] ndmreset_o
constexpr uint32_t DBG_RV_STATE_NDMRESET_LSB      = 6u;
constexpr uint32_t DBG_RV_STATE_NDMRESET_W        = 1u;
constexpr uint32_t DBG_RV_STATE_HALT_REQ          = 0x00000080u;  // [7] halt-request level
constexpr uint32_t DBG_RV_STATE_HALT_REQ_LSB      = 7u;
constexpr uint32_t DBG_RV_STATE_HALT_REQ_W        = 1u;
constexpr uint32_t DBG_RV_STATE_SAMPLE_SEQ        = 0x0000FF00u;  // [15:8]
constexpr uint32_t DBG_RV_STATE_SAMPLE_SEQ_LSB    = 8u;
constexpr uint32_t DBG_RV_STATE_SAMPLE_SEQ_W      = 8u;
constexpr uint32_t DBG_RV_STATE_DEBUG_ENTRIES     = 0xFFFF0000u;  // [31:16] debug-mode entries mod 2^16
constexpr uint32_t DBG_RV_STATE_DEBUG_ENTRIES_LSB = 16u;
constexpr uint32_t DBG_RV_STATE_DEBUG_ENTRIES_W   = 16u;
// DBG_RVWP_CTL (DB §10.1): RV watchpoint j < 2
constexpr uint32_t DBG_RVWP_CTL_EN        = 0x00000001u;  // [0]
constexpr uint32_t DBG_RVWP_CTL_EN_LSB    = 0u;
constexpr uint32_t DBG_RVWP_CTL_EN_W      = 1u;
constexpr uint32_t DBG_RVWP_CTL_LOAD      = 0x00000002u;  // [1]
constexpr uint32_t DBG_RVWP_CTL_LOAD_LSB  = 1u;
constexpr uint32_t DBG_RVWP_CTL_LOAD_W    = 1u;
constexpr uint32_t DBG_RVWP_CTL_STORE     = 0x00000004u;  // [2]
constexpr uint32_t DBG_RVWP_CTL_STORE_LSB = 2u;
constexpr uint32_t DBG_RVWP_CTL_STORE_W   = 1u;
constexpr uint32_t DBG_RVWP_CTL_HALT      = 0x00000008u;  // [3]
constexpr uint32_t DBG_RVWP_CTL_HALT_LSB  = 3u;
constexpr uint32_t DBG_RVWP_CTL_HALT_W    = 1u;
// DBG_TBP_CTL (DB §10.1, §5.3): tile breakpoint i < 4
constexpr uint32_t DBG_TBP_CTL_EN             = 0x00000001u;  // [0]
constexpr uint32_t DBG_TBP_CTL_EN_LSB         = 0u;
constexpr uint32_t DBG_TBP_CTL_EN_W           = 1u;
constexpr uint32_t DBG_TBP_CTL_KIND           = 0x0000000Eu;  // [3:1] 5..7 never match
constexpr uint32_t DBG_TBP_CTL_KIND_LSB       = 1u;
constexpr uint32_t DBG_TBP_CTL_KIND_W         = 3u;
constexpr uint32_t DBG_TBP_CTL_R              = 0x00000010u;  // [4]
constexpr uint32_t DBG_TBP_CTL_R_LSB          = 4u;
constexpr uint32_t DBG_TBP_CTL_R_W            = 1u;
constexpr uint32_t DBG_TBP_CTL_W              = 0x00000020u;  // [5]
constexpr uint32_t DBG_TBP_CTL_W_LSB          = 5u;
constexpr uint32_t DBG_TBP_CTL_W_W            = 1u;
constexpr uint32_t DBG_TBP_CTL_AFTER          = 0x00000040u;  // [6]
constexpr uint32_t DBG_TBP_CTL_AFTER_LSB      = 6u;
constexpr uint32_t DBG_TBP_CTL_AFTER_W        = 1u;
constexpr uint32_t DBG_TBP_CTL_HALT_RV        = 0x00000080u;  // [7]
constexpr uint32_t DBG_TBP_CTL_HALT_RV_LSB    = 7u;
constexpr uint32_t DBG_TBP_CTL_HALT_RV_W      = 1u;
constexpr uint32_t DBG_TBP_CTL_COUNT_ONLY     = 0x00000100u;  // [8]
constexpr uint32_t DBG_TBP_CTL_COUNT_ONLY_LSB = 8u;
constexpr uint32_t DBG_TBP_CTL_COUNT_ONLY_W   = 1u;
constexpr uint32_t DBG_TBP_CTL_CHMASK         = 0xFFFF0000u;  // [31:16] channels of a DRAM breakpoint
constexpr uint32_t DBG_TBP_CTL_CHMASK_LSB     = 16u;
constexpr uint32_t DBG_TBP_CTL_CHMASK_W       = 16u;
constexpr uint32_t DBG_TBP_CTL_KIND_PIDX      = 0u;           // (pidx - A) mod 2^32 < (B - A) mod 2^32
constexpr uint32_t DBG_TBP_CTL_KIND_OPCODE    = 1u;           // A: layout TBP_OPC_A, B: TBP_OPC_B
constexpr uint32_t DBG_TBP_CTL_KIND_SPM       = 2u;           // an SPM extent overlaps lines [A, B)
constexpr uint32_t DBG_TBP_CTL_KIND_DRAM      = 3u;           // the DRAM extent overlaps beats [A, B)
constexpr uint32_t DBG_TBP_CTL_KIND_TICKET    = 4u;           // ticket = A
// TBP_OPC_A (DB §5.3): DBG_TBP_A of an OPCODE breakpoint
constexpr uint32_t TBP_OPC_A_OPC          = 0x000000FFu;  // [7:0] opcode value
constexpr uint32_t TBP_OPC_A_OPC_LSB      = 0u;
constexpr uint32_t TBP_OPC_A_OPC_W        = 8u;
constexpr uint32_t TBP_OPC_A_OPC_MASK     = 0x0000FF00u;  // [15:8] opcode care mask
constexpr uint32_t TBP_OPC_A_OPC_MASK_LSB = 8u;
constexpr uint32_t TBP_OPC_A_OPC_MASK_W   = 8u;
constexpr uint32_t TBP_OPC_A_SUB          = 0x000F0000u;  // [19:16] sub value
constexpr uint32_t TBP_OPC_A_SUB_LSB      = 16u;
constexpr uint32_t TBP_OPC_A_SUB_W        = 4u;
constexpr uint32_t TBP_OPC_A_SUB_MASK     = 0x00F00000u;  // [23:20] sub care mask
constexpr uint32_t TBP_OPC_A_SUB_MASK_LSB = 20u;
constexpr uint32_t TBP_OPC_A_SUB_MASK_W   = 4u;
// TBP_OPC_B (DB §5.3): DBG_TBP_B of an OPCODE breakpoint
constexpr uint32_t TBP_OPC_B_UNITS     = 0x000000FFu;  // [7:0] unit mask; 0 = any unit
constexpr uint32_t TBP_OPC_B_UNITS_LSB = 0u;
constexpr uint32_t TBP_OPC_B_UNITS_W   = 8u;
// DBG_U_STATE (DB §5.5): per-unit probe u < 8
constexpr uint32_t DBG_U_STATE_ACCEPTED        = 0x00000003u;  // [1:0] ops accepted and not done
constexpr uint32_t DBG_U_STATE_ACCEPTED_LSB    = 0u;
constexpr uint32_t DBG_U_STATE_ACCEPTED_W      = 2u;
constexpr uint32_t DBG_U_STATE_HEAD            = 0x00000070u;  // [6:4] head state
constexpr uint32_t DBG_U_STATE_HEAD_LSB        = 4u;
constexpr uint32_t DBG_U_STATE_HEAD_W          = 3u;
constexpr uint32_t DBG_U_STATE_HOLD            = 0x00000F00u;  // [11:8] hold sources at the head (layout HOLD_SRC)
constexpr uint32_t DBG_U_STATE_HOLD_LSB        = 8u;
constexpr uint32_t DBG_U_STATE_HOLD_W          = 4u;
constexpr uint32_t DBG_U_STATE_OUTSTANDING     = 0x000FF000u;  // [19:12] outstanding entries
constexpr uint32_t DBG_U_STATE_OUTSTANDING_LSB = 12u;
constexpr uint32_t DBG_U_STATE_OUTSTANDING_W   = 8u;
constexpr uint32_t DBG_U_STATE_HEAD_EMPTY      = 0u;
constexpr uint32_t DBG_U_STATE_HEAD_LOAD       = 1u;
constexpr uint32_t DBG_U_STATE_HEAD_WAIT_DEP   = 2u;
constexpr uint32_t DBG_U_STATE_HEAD_WAIT_SNAP  = 3u;
constexpr uint32_t DBG_U_STATE_HEAD_HELD       = 4u;
constexpr uint32_t DBG_U_STATE_HEAD_WAIT_UNIT  = 5u;
constexpr uint32_t DBG_U_STATE_HEAD_ISSUE      = 6u;
// DBG_TRW_MASK (DB §10.1): reset 0x7FF
constexpr uint32_t DBG_TRW_MASK_UNITS       = 0x000000FFu;  // [7:0]
constexpr uint32_t DBG_TRW_MASK_UNITS_LSB   = 0u;
constexpr uint32_t DBG_TRW_MASK_UNITS_W     = 8u;
constexpr uint32_t DBG_TRW_MASK_MARKERS     = 0x00000100u;  // [8]
constexpr uint32_t DBG_TRW_MASK_MARKERS_LSB = 8u;
constexpr uint32_t DBG_TRW_MASK_MARKERS_W   = 1u;
constexpr uint32_t DBG_TRW_MASK_STOPS       = 0x00000200u;  // [9]
constexpr uint32_t DBG_TRW_MASK_STOPS_LSB   = 9u;
constexpr uint32_t DBG_TRW_MASK_STOPS_W     = 1u;
constexpr uint32_t DBG_TRW_MASK_RVHALTS     = 0x00000400u;  // [10]
constexpr uint32_t DBG_TRW_MASK_RVHALTS_LSB = 10u;
constexpr uint32_t DBG_TRW_MASK_RVHALTS_W   = 1u;
// DBG_HIST_INFO (DB §10.1)
constexpr uint32_t DBG_HIST_INFO_PRESENT     = 0x00000001u;  // [0] the slot holds that push index, recorded after the last HIST_CLEAR
constexpr uint32_t DBG_HIST_INFO_PRESENT_LSB = 0u;
constexpr uint32_t DBG_HIST_INFO_PRESENT_W   = 1u;
constexpr uint32_t DBG_HIST_INFO_UNIT        = 0x000000F0u;  // [7:4]
constexpr uint32_t DBG_HIST_INFO_UNIT_LSB    = 4u;
constexpr uint32_t DBG_HIST_INFO_UNIT_W      = 4u;
constexpr uint32_t DBG_HIST_INFO_CODE        = 0x00000F00u;  // [11:8] checker code (0 = runs)
constexpr uint32_t DBG_HIST_INFO_CODE_LSB    = 8u;
constexpr uint32_t DBG_HIST_INFO_CODE_W      = 4u;
constexpr uint32_t DBG_HIST_INFO_REC         = 0x00001000u;  // [12]
constexpr uint32_t DBG_HIST_INFO_REC_LSB     = 12u;
constexpr uint32_t DBG_HIST_INFO_REC_W       = 1u;
constexpr uint32_t DBG_HIST_INFO_TBPS        = 0x0001E000u;  // [16:13] TBPs matched
constexpr uint32_t DBG_HIST_INFO_TBPS_LSB    = 13u;
constexpr uint32_t DBG_HIST_INFO_TBPS_W      = 4u;
constexpr uint32_t DBG_HIST_INFO_STOP        = 0x00020000u;  // [17] stop point
constexpr uint32_t DBG_HIST_INFO_STOP_LSB    = 17u;
constexpr uint32_t DBG_HIST_INFO_STOP_W      = 1u;
// IFIFO_CTRL (CC §7.2): w0 of a CTRL issue-FIFO entry
constexpr uint32_t IFIFO_CTRL_CODE            = 0x0000000Fu;  // [3:0]
constexpr uint32_t IFIFO_CTRL_CODE_LSB        = 0u;
constexpr uint32_t IFIFO_CTRL_CODE_W          = 4u;
constexpr uint32_t IFIFO_CTRL_CODE_QABORT     = 1u;
constexpr uint32_t IFIFO_CTRL_CODE_QHOLD_SET  = 2u;
constexpr uint32_t IFIFO_CTRL_CODE_QHOLD_CLR  = 3u;
constexpr uint32_t IFIFO_CTRL_CODE_QCLEAR     = 4u;           // ticketed
constexpr uint32_t IFIFO_CTRL_CODE_TRACE_MARK = 5u;           // w1 = marker value

// -----------------------------------------------------------------------------------------------
// Wide layouts (bit positions only)
// -----------------------------------------------------------------------------------------------
// IFIFO (CC §7.2): issue-FIFO entry (tsr_cdc_fifo, 32 entries LUTRAM)
constexpr uint32_t IFIFO_BITS       = 387u;
constexpr uint32_t IFIFO_KIND_LSB   = 0u;    // [1:0] entry kind
constexpr uint32_t IFIFO_KIND_W     = 2u;
constexpr uint32_t IFIFO_W_LSB      = 2u;    // [257:2] w0..w7 (OP; masked words are 0); CTRL: w0 = layout IFIFO_CTRL, w1 = marker value
constexpr uint32_t IFIFO_W_W        = 256u;
constexpr uint32_t IFIFO_DEP0_LSB   = 258u;  // [289:258] OP
constexpr uint32_t IFIFO_DEP0_W     = 32u;
constexpr uint32_t IFIFO_DEP1_LSB   = 290u;  // [321:290] OP
constexpr uint32_t IFIFO_DEP1_W     = 32u;
constexpr uint32_t IFIFO_REC_LSB    = 322u;  // [322]
constexpr uint32_t IFIFO_REC_W      = 1u;
constexpr uint32_t IFIFO_TICKET_LSB = 323u;  // [354:323] OP and QCLEAR; the dispatcher asserts it equals its own count
constexpr uint32_t IFIFO_TICKET_W   = 32u;
constexpr uint32_t IFIFO_PC_LSB     = 355u;  // [386:355] push PC (OP and TRACE_MARK)
constexpr uint32_t IFIFO_PC_W       = 32u;
constexpr uint32_t IFIFO_KIND_OP    = 0u;
constexpr uint32_t IFIFO_KIND_CTRL  = 1u;

// -----------------------------------------------------------------------------------------------
// Word tables of fixed-size records and structures (word indices)
// -----------------------------------------------------------------------------------------------
// DESC (EX §4.3): launch descriptor (host -> RISC-V)
constexpr uint32_t DESC_NWORDS    = 8u;
constexpr uint32_t DESC_HDR_WORD  = 0u;  // layout DESC
constexpr uint32_t DESC_ARG0_WORD = 1u;  // ARG0..ARG6: integers, SPM lines or DRAM handles (layout HANDLE)
constexpr uint32_t DESC_ARG0_N    = 7u;
// LCQ_DONE (EX §4.4): DONE record (0x01, answer)
constexpr uint32_t LCQ_DONE_NWORDS          = 8u;
constexpr uint32_t LCQ_DONE_R0_WORD         = 0u;  // layout LCQ_R0
constexpr uint32_t LCQ_DONE_SEQ_WORD        = 1u;  // SEQ of the answered descriptor
constexpr uint32_t LCQ_DONE_RET_WORD        = 2u;  // kernel return value (int32)
constexpr uint32_t LCQ_DONE_ERR_TICKET_WORD = 3u;  // first failing ticket; TICKET_NONE = none
constexpr uint32_t LCQ_DONE_ERR_INFO_WORD   = 4u;
constexpr uint32_t LCQ_DONE_ERR_W0_WORD     = 5u;
constexpr uint32_t LCQ_DONE_ERR_PIDX_WORD   = 6u;
constexpr uint32_t LCQ_DONE_DURATION_WORD   = 7u;  // core cycles from the pop to the post, saturating
// LCQ_PONG (EX §4.4): PONG record (0x02, answer)
constexpr uint32_t LCQ_PONG_NWORDS        = 8u;
constexpr uint32_t LCQ_PONG_R0_WORD       = 0u;
constexpr uint32_t LCQ_PONG_SEQ_WORD      = 1u;
constexpr uint32_t LCQ_PONG_ARG0_WORD     = 2u;  // ARG0 echo
constexpr uint32_t LCQ_PONG_DURATION_WORD = 7u;
// LCQ_HELLO (EX §4.4, SW §6.3): HELLO record (0x80, event)
constexpr uint32_t LCQ_HELLO_NWORDS           = 8u;
constexpr uint32_t LCQ_HELLO_R0_WORD          = 0u;
constexpr uint32_t LCQ_HELLO_TAKEN_WORD       = 1u;  // TAKEN at the post
constexpr uint32_t LCQ_HELLO_ID_LO_WORD       = 2u;  // image id [31:0]
constexpr uint32_t LCQ_HELLO_ID_HI_WORD       = 3u;  // image id [63:32]
constexpr uint32_t LCQ_HELLO_VERSIONS_WORD    = 4u;  // layout HELLO_R4
constexpr uint32_t LCQ_HELLO_N_KERNELS_WORD   = 5u;  // kernel count
constexpr uint32_t LCQ_HELLO_BASIS_WORD       = 6u;  // Tesseract Basis version, layout HELLO_R6
constexpr uint32_t LCQ_HELLO_BOOT_CYCLES_WORD = 7u;  // boot core cycles
// LCQ_TRAP (EX §4.4, DB §6.3): TRAP record (0x81)
constexpr uint32_t LCQ_TRAP_NWORDS        = 8u;
constexpr uint32_t LCQ_TRAP_R0_WORD       = 0u;
constexpr uint32_t LCQ_TRAP_SEQ_WORD      = 1u;  // SEQ (answer) or TAKEN (event)
constexpr uint32_t LCQ_TRAP_MCAUSE_WORD   = 2u;
constexpr uint32_t LCQ_TRAP_MEPC_WORD     = 3u;
constexpr uint32_t LCQ_TRAP_MTVAL_WORD    = 4u;
constexpr uint32_t LCQ_TRAP_RT_STATE_WORD = 5u;  // at the trap
constexpr uint32_t LCQ_TRAP_RT_INFO_WORD  = 6u;
constexpr uint32_t LCQ_TRAP_MTIME_WORD    = 7u;  // MTIME[31:0]
// LCQ_EVENT (EX §4.4): kernel event record (0xC0..0xFF)
constexpr uint32_t LCQ_EVENT_NWORDS    = 8u;
constexpr uint32_t LCQ_EVENT_R0_WORD   = 0u;  // type, kernel-defined [23:8], seq[7:0]
constexpr uint32_t LCQ_EVENT_SEQ_WORD  = 1u;  // SEQ of the running launch
constexpr uint32_t LCQ_EVENT_DATA_WORD = 2u;  // r2..r7 kernel-defined
constexpr uint32_t LCQ_EVENT_DATA_N    = 6u;
// REC (DB §8.2): completion record (REC_DATA0..3)
constexpr uint32_t REC_NWORDS      = 4u;
constexpr uint32_t REC_TICKET_WORD = 0u;  // ticket (unit 14 for markers)
constexpr uint32_t REC_T_END_WORD  = 1u;  // MTIME[31:0] at completion (markers: dispatch time)
constexpr uint32_t REC_RUN_WORD    = 2u;  // start -> end cycles, saturating (markers: the value)
constexpr uint32_t REC_INFO_WORD   = 3u;  // layout REC_DATA3
// TRW_OP (DB §8.3): trace-writer OP record
constexpr uint32_t TRW_OP_NWORDS        = 8u;
constexpr uint32_t TRW_OP_HDR_WORD      = 0u;  // layout TRW_W0
constexpr uint32_t TRW_OP_PIDX_WORD     = 1u;  // push index
constexpr uint32_t TRW_OP_TICKET_WORD   = 2u;
constexpr uint32_t TRW_OP_T_END_LO_WORD = 3u;  // MTIME
constexpr uint32_t TRW_OP_T_END_HI_WORD = 4u;
constexpr uint32_t TRW_OP_RUN_WORD      = 5u;  // t_end - t_start (0 if not run), saturating
constexpr uint32_t TRW_OP_QWAIT_WORD    = 6u;  // t_head - t_dispatch
constexpr uint32_t TRW_OP_HWAIT_WORD    = 7u;  // t_start - t_head; t_end - t_head for non-run entries
// TRW_MARK (DB §8.3): trace-writer MARK record
constexpr uint32_t TRW_MARK_NWORDS         = 8u;
constexpr uint32_t TRW_MARK_HDR_WORD       = 0u;
constexpr uint32_t TRW_MARK_NEXT_PIDX_WORD = 1u;  // next push index
constexpr uint32_t TRW_MARK_VALUE_WORD     = 2u;  // marker value
constexpr uint32_t TRW_MARK_T_LO_WORD      = 3u;  // dispatch time
constexpr uint32_t TRW_MARK_T_HI_WORD      = 4u;
// TRW_STOP (DB §8.3): trace-writer STOP record
constexpr uint32_t TRW_STOP_NWORDS         = 8u;
constexpr uint32_t TRW_STOP_HDR_WORD       = 0u;
constexpr uint32_t TRW_STOP_T_WORD         = 1u;  // threshold
constexpr uint32_t TRW_STOP_CAUSE_WORD     = 2u;  // cause bits (layout DBG_CAUSE)
constexpr uint32_t TRW_STOP_T_LO_WORD      = 3u;  // time
constexpr uint32_t TRW_STOP_T_HI_WORD      = 4u;
constexpr uint32_t TRW_STOP_ACTIVATED_WORD = 5u;  // 1 activated, 0 released
// TRW_RVHALT (DB §8.3): trace-writer RVHALT record
constexpr uint32_t TRW_RVHALT_NWORDS      = 8u;
constexpr uint32_t TRW_RVHALT_HDR_WORD    = 0u;
constexpr uint32_t TRW_RVHALT_HALTED_WORD = 2u;  // 1 halted, 0 resumed
// TRW_DROP (DB §8.3): trace-writer DROP record
constexpr uint32_t TRW_DROP_NWORDS       = 8u;
constexpr uint32_t TRW_DROP_HDR_WORD     = 0u;
constexpr uint32_t TRW_DROP_DROPPED_WORD = 1u;  // records dropped since the previous DROP record
// DBG_INFO (DB §6.3): __tsr_dbg (section .tsr_dbg, 64 B)
constexpr uint32_t DBG_INFO_NWORDS            = 16u;
constexpr uint32_t DBG_INFO_MAGIC_WORD        = 0u;   // DBG_INFO_MAGIC
constexpr uint32_t DBG_INFO_VERSION_WORD      = 1u;   // 1
constexpr uint32_t DBG_INFO_BUILD_ID_WORD     = 2u;   // ELF note build id
constexpr uint32_t DBG_INFO_ATTACHED_WORD     = 3u;   // written by the debugger
constexpr uint32_t DBG_INFO_LAUNCH_BREAK_WORD = 4u;   // 0 off, 0xFFFFFFFF all, else kernel index + 1
constexpr uint32_t DBG_INFO_TRAP_COUNT_WORD   = 5u;
// CRASH (DB §6.3): __tsr_crash (section .tsr_crash, 192 B)
constexpr uint32_t CRASH_NWORDS        = 48u;
constexpr uint32_t CRASH_MAGIC_WORD    = 0u;   // CRASH_MAGIC
constexpr uint32_t CRASH_VERSION_WORD  = 1u;   // 1
constexpr uint32_t CRASH_MCAUSE_WORD   = 2u;
constexpr uint32_t CRASH_MEPC_WORD     = 3u;
constexpr uint32_t CRASH_MTVAL_WORD    = 4u;
constexpr uint32_t CRASH_MSTATUS_WORD  = 5u;
constexpr uint32_t CRASH_MTIME_LO_WORD = 6u;
constexpr uint32_t CRASH_MTIME_HI_WORD = 7u;
constexpr uint32_t CRASH_SEQ_WORD      = 8u;   // launch SEQ
constexpr uint32_t CRASH_KERNEL_WORD   = 9u;   // kernel index
constexpr uint32_t CRASH_RT_STATE_WORD = 10u;
constexpr uint32_t CRASH_RT_INFO_WORD  = 11u;
constexpr uint32_t CRASH_X1_WORD       = 12u;  // x1..x31 at words 12..42
constexpr uint32_t CRASH_X1_N          = 31u;
constexpr uint32_t CRASH_NESTED_WORD   = 43u;

// -----------------------------------------------------------------------------------------------
// Units (EX §7.3, CC §7.4, §9.4)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t UNIT_DMA          = 0u;           // EX §7.3, CC §7.4: LD, ST, KVST
constexpr uint32_t UNIT_QACT         = 1u;           // EX §7.3, CC §7.4: QACT
constexpr uint32_t UNIT_MVM          = 2u;           // EX §7.3, CC §7.4, §9.4: MVM, ATTQK, ATTPV
constexpr uint32_t UNIT_VU           = 3u;           // EX §7.3, CC §7.4, §9.4: VOP, RED, ROPE, TOPK, SOP, DQROW, ASMAX, AMASK, AMERGE, ANORM (VMASK from M9)
constexpr uint32_t UNIT_ATT          = 4u;           // EX §7.3, CC §7.4: reserved for a future AT attention engine; absent in M6 (was CMD)
constexpr uint32_t UNIT_SYNC         = 5u;           // EX §7.3, CC §7.4: NOP, FENCE, QCLEAR and every refused opcode
constexpr uint32_t UNIT_MARKER       = 14u;          // EX §7.3, CC §7.4: software marker records (REC / TRW); never a valid dependency
constexpr uint32_t UNIT_NONE         = 15u;          // EX §7.3, CC §7.4: no dependency
constexpr uint32_t UNIT_RSVD_FIRST   = 6u;           // EX §7.3: ids 6..13 reserved: a dependency on them is BAD_DEP
constexpr uint32_t UNIT_RSVD_LAST    = 13u;
constexpr uint32_t QDEPTH_U200_DMA   = 8u;           // Q_DEPTH of the U200 M6 bitstream
constexpr uint32_t QDEPTH_U200_QACT  = 4u;
constexpr uint32_t QDEPTH_U200_MVM   = 4u;
constexpr uint32_t QDEPTH_U200_VU    = 16u;
constexpr uint32_t QDEPTH_U200_ATT   = 0u;
constexpr uint32_t QDEPTH_U200_SYNC  = 4u;
constexpr uint32_t QDEPTH_PARAM_DMA  = 8u;           // depth parameter when the unit is built
constexpr uint32_t QDEPTH_PARAM_QACT = 4u;
constexpr uint32_t QDEPTH_PARAM_MVM  = 4u;
constexpr uint32_t QDEPTH_PARAM_VU   = 16u;
constexpr uint32_t QDEPTH_PARAM_ATT  = 8u;
constexpr uint32_t QDEPTH_PARAM_SYNC = 4u;
constexpr uint32_t ACCEPT_MAX_DMA    = 1u;           // I3: ops accepted and not done (CC §9.4)
constexpr uint32_t ACCEPT_MAX_QACT   = 1u;
constexpr uint32_t ACCEPT_MAX_MVM    = 2u;
constexpr uint32_t ACCEPT_MAX_VU     = 2u;
constexpr uint32_t ACCEPT_MAX_ATT    = 1u;
constexpr uint32_t ACCEPT_MAX_SYNC   = 1u;
constexpr uint32_t QDEPTH_U200_REG   = 0x08081088u;  // Q_DEPTH register value of the U200 M6 bitstream

// -----------------------------------------------------------------------------------------------
// Opcodes (EX §7.2); reserved and retired opcodes complete ILLEGAL_OP on SYNC
// -----------------------------------------------------------------------------------------------
constexpr uint32_t OP_NOP    = 0x00u;  // kept, unit SYNC: EX §7.2: SYNC join point: completes when its dependencies are done
constexpr uint32_t OP_END    = 0x01u;  // reserved, unit SYNC: EX §7.2: END: ILLEGAL_OP
constexpr uint32_t OP_LI     = 0x02u;  // reserved, unit SYNC: EX §7.2: LI: ILLEGAL_OP
constexpr uint32_t OP_ADDI   = 0x03u;  // reserved, unit SYNC: EX §7.2: ADDI: ILLEGAL_OP
constexpr uint32_t OP_MULI   = 0x04u;  // retired, unit SYNC: EX §7.2: retired, never reused
constexpr uint32_t OP_RLD    = 0x05u;  // retired, unit SYNC: EX §7.2: retired, never reused
constexpr uint32_t OP_LOOP   = 0x06u;  // retired, unit SYNC: EX §7.2: retired, never reused
constexpr uint32_t OP_WAITM  = 0x07u;  // retired, unit SYNC: EX §7.2: retired, never reused
constexpr uint32_t OP_JUMP   = 0x08u;  // retired, unit SYNC: EX §7.2: retired, never reused
constexpr uint32_t OP_CALL   = 0x09u;  // retired, unit SYNC: EX §7.2: retired (never shipped), never reused
constexpr uint32_t OP_LD     = 0x10u;  // kept, unit DMA: EX §7.2: DRAM -> SPM, encoding unchanged
constexpr uint32_t OP_ST     = 0x11u;  // kept, unit DMA: EX §7.2: SPM -> DRAM, encoding unchanged
constexpr uint32_t OP_DQROW  = 0x12u;  // respec, unit VU, CAPS.DQROW: EX §7.2, VU §5.6: SPM -> f32 dequantisation; simulator from M6, RTL in M8
constexpr uint32_t OP_KVST   = 0x14u;  // new, unit DMA, CAPS.ATT: EX §7.2, AT §5.1: store K or V rows into a KV plane
constexpr uint32_t OP_QACT   = 0x18u;  // kept, unit QACT: EX §7.2: f32 -> QA, encoding unchanged
constexpr uint32_t OP_MVM    = 0x20u;  // kept, unit MVM: EX §7.2: matrix x activations, encoding unchanged
constexpr uint32_t OP_ATTQK  = 0x21u;  // new, unit MVM, CAPS.ATT: EX §7.2, AT §5.2: scores and row maxima over a KV group range
constexpr uint32_t OP_ATTPV  = 0x22u;  // new, unit MVM, CAPS.ATT: EX §7.2, AT §5.3: P.V channel partials and row sums
constexpr uint32_t OP_VOP    = 0x30u;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.2: element-wise vector op
constexpr uint32_t OP_RED    = 0x31u;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.3: segmented reduction with post-ops
constexpr uint32_t OP_ROPE   = 0x32u;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.4: rotary embedding over T tokens
constexpr uint32_t OP_TOPK   = 0x33u;  // respec, unit VU, CAPS.TOPK: EX §7.2, VU §5.5: top-k per row; simulator from M6, RTL in M8
constexpr uint32_t OP_ASMAX  = 0x34u;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.4: softmax numerator
constexpr uint32_t OP_AMASK  = 0x35u;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.5: general mask path
constexpr uint32_t OP_AMERGE = 0x36u;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.6: channel tree and online merge
constexpr uint32_t OP_ANORM  = 0x37u;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.7: final normalisation
constexpr uint32_t OP_VMASK  = 0x38u;  // new, unit VU, CAPS.VMASK: EX §7.2, VU §5.7: position-window mask; ILLEGAL_OP until M9
constexpr uint32_t OP_GDN    = 0x39u;  // new, unit VU, CAPS.GDN: EX §7.2, VU §7.12: fused Gated DeltaNet state update of one head
constexpr uint32_t OP_SOP    = 0x40u;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.8: scalar op on s0..s15
constexpr uint32_t OP_FENCE  = 0x7Fu;  // kept, unit SYNC: EX §7.2: SYNC barrier: completes when every op pushed before it is done
// Unit of a legal opcode; UNIT_SYNC for every reserved or retired opcode. A legal opcode is
// refused the same way when (CAPS & op_caps) != op_caps or its unit is not built.
constexpr uint32_t op_unit(uint32_t opcode) {
    switch (opcode & 0xFFu) {
    case 0x10u: case 0x11u: case 0x14u: return UNIT_DMA;
    case 0x12u: case 0x30u: case 0x31u: case 0x32u: case 0x33u: case 0x34u: case 0x35u: case 0x36u:
    case 0x37u: case 0x38u: case 0x39u: case 0x40u: return UNIT_VU;
    case 0x18u: return UNIT_QACT;
    case 0x20u: case 0x21u: case 0x22u: return UNIT_MVM;
    default: return UNIT_SYNC;
    }
}
constexpr bool op_legal(uint32_t opcode) {
    switch (opcode & 0xFFu) {
    case 0x00u: case 0x10u: case 0x11u: case 0x12u: case 0x14u: case 0x18u: case 0x20u: case 0x21u:
    case 0x22u: case 0x30u: case 0x31u: case 0x32u: case 0x33u: case 0x34u: case 0x35u: case 0x36u:
    case 0x37u: case 0x38u: case 0x39u: case 0x40u: case 0x7Fu: return true;
    default: return false;
    }
}
constexpr uint32_t op_caps(uint32_t opcode) {
    switch (opcode & 0xFFu) {
    case 0x12u: return CAPS_DQROW;
    case 0x14u: case 0x21u: case 0x22u: case 0x34u: case 0x35u: case 0x36u: case 0x37u:
    return CAPS_ATT;
    case 0x30u: case 0x31u: case 0x32u: case 0x40u: return CAPS_VU;
    case 0x33u: return CAPS_TOPK;
    case 0x38u: return CAPS_VMASK;
    case 0x39u: return CAPS_GDN;
    default: return 0u;
    }
}

// -----------------------------------------------------------------------------------------------
// Error codes (EX §8.1)
// -----------------------------------------------------------------------------------------------
constexpr uint32_t ERR_NONE        = 0u;
constexpr uint32_t ERR_ILLEGAL_OP  = 1u;   // reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it; ERR_INFO = 0
constexpr uint32_t ERR_BAD_FMT     = 2u;   // unsupported weight format; ERR_INFO = 0
constexpr uint32_t ERR_QA_MISMATCH = 3u;   // (reserved) never raised
constexpr uint32_t ERR_SPM_RANGE   = 4u;   // SPM operand out of range (spec range-check rule); ERR_INFO = start address of the failing operand
constexpr uint32_t ERR_DRAM_RANGE  = 5u;   // DRAM operand out of range; ERR_INFO = start address
constexpr uint32_t ERR_K_RANGE     = 6u;   // K > K_MAX or not a multiple of the block size; ERR_INFO = 0
constexpr uint32_t ERR_T_RANGE     = 7u;   // T = 0 or T > T_MAX; ERR_INFO = 0
constexpr uint32_t ERR_ABORTED     = 8u;   // completed without running: a QABORT, or its unit was in error state; ERR_INFO = 0
constexpr uint32_t ERR_DEP_FAILED  = 11u;  // a dependency completed with an error; ERR_INFO = 0
constexpr uint32_t ERR_BAD_DEP     = 12u;  // a dependency on a ticket not pushed yet, on unit 6..14 or an absent unit, or below its unit's clear point; ERR_INFO = 0
constexpr uint32_t ERR_OPERAND     = 13u;  // an operand value outside the range its op defines, other than K, T and the address ranges (VU and AT ops only); ERR_INFO = the offending value
// code 9 retired (LOOP_NEST), never raised, never reused
// code 10 retired (WAIT_TIMEOUT), never raised, never reused
constexpr uint32_t ERR_CHECK_N = 8u;  // dispatch check order: ILLEGAL_OP > BAD_DEP > BAD_FMT > K_RANGE > T_RANGE > OPERAND > SPM_RANGE > DRAM_RANGE
constexpr uint32_t ERR_CHECK_ORDER[8] = {ERR_ILLEGAL_OP, ERR_BAD_DEP, ERR_BAD_FMT, ERR_K_RANGE, ERR_T_RANGE, ERR_OPERAND, ERR_SPM_RANGE, ERR_DRAM_RANGE};
constexpr uint32_t LCQ_STATUS_PRECEDENCE[8] = {LCQ_R0_STATUS_TRAP, LCQ_R0_STATUS_BAD_DESC, LCQ_R0_STATUS_ABORTED, LCQ_R0_STATUS_TILE_ERR, LCQ_R0_STATUS_PLAT_ERR, LCQ_R0_STATUS_STOPPED, LCQ_R0_STATUS_KERNEL_ERR, LCQ_R0_STATUS_OK};
constexpr uint32_t err_check_rank(uint32_t code) {
    for (uint32_t i = 0; i < 8; ++i)
        if (ERR_CHECK_ORDER[i] == code) return i;
    return 0xFFu;
}
constexpr uint32_t lcq_status_rank(uint32_t status) {
    for (uint32_t i = 0; i < 8; ++i)
        if (LCQ_STATUS_PRECEDENCE[i] == status) return i;
    return 0xFFu;
}

// -----------------------------------------------------------------------------------------------
// PERF slots (EX §6): C = NUM_CH, X = 5 + 2C
// -----------------------------------------------------------------------------------------------
constexpr uint32_t PERF_CYCLES     = 0u;  // EX §6: time, core cycles
constexpr uint32_t PERF_RETIRED    = 1u;  // EX §6: event, queue entries completed on every unit, any outcome
constexpr uint32_t PERF_MVM_CYCLES = 2u;  // EX §6: time, the MVM unit holds an op
constexpr uint32_t PERF_VU_CYCLES  = 3u;  // EX §6, VU §5.10: time, the VU holds an op
// slot 4 reserved (was WAITM), reads 0 (EX §6)
constexpr uint32_t PERF_XOFF_DMA_CYCLES     = 0u;   // EX §6: X + 0, time, the DMA unit holds an op
constexpr uint32_t PERF_XOFF_QACT_CYCLES    = 1u;   // EX §6: X + 1, time, the QACT unit holds an op
constexpr uint32_t PERF_XOFF_ATT_CYCLES     = 2u;   // EX §6: X + 2, time, unit 4 (AT) holds an op; reads 0 without it
constexpr uint32_t PERF_XOFF_MVM_WAIT       = 3u;   // EX §6: X + 3, time, MVM idle while its queue holds an entry that cannot start
constexpr uint32_t PERF_XOFF_MVM_EMPTY      = 4u;   // EX §6: X + 4, time, MVM idle and its queue empty (issue-bound)
constexpr uint32_t PERF_XOFF_HOLD_CYCLES    = 5u;   // EX §6: X + 5, time, some queue head is held
constexpr uint32_t PERF_XOFF_PUSHES         = 6u;   // EX §6: X + 6, event, entries pushed
constexpr uint32_t PERF_XOFF_PUSH_STALL     = 7u;   // EX §6: X + 7, rv_time, clk_rv cycles a push waited for a credit
constexpr uint32_t PERF_XOFF_BRIDGE_ACC     = 8u;   // EX §6: X + 8, event, SPM bridge accesses
constexpr uint32_t PERF_XOFF_BRIDGE_WAIT    = 9u;   // EX §6: X + 9, rv_time, clk_rv cycles bridge accesses waited
constexpr uint32_t PERF_XOFF_REC_DROPPED    = 10u;  // EX §6: X + 10, event, completion records dropped
constexpr uint32_t PERF_XOFF_LAUNCHES       = 11u;  // EX §6: X + 11, event, answers posted
constexpr uint32_t PERF_XOFF_RV_WFI         = 12u;  // EX §6: X + 12, time, the RISC-V sleeps in WFI
constexpr uint32_t PERF_XOFF_VU_SPM_WAIT    = 13u;  // EX §6, VU §5.10: X + 13, time, a VU SPM request is raised and not granted
constexpr uint32_t PERF_XOFF_VU_HAZARD      = 14u;  // EX §6, VU §5.10: X + 14, time, an accepted VU op waits for an earlier VU op
constexpr uint32_t PERF_XOFF_VU_LINES       = 15u;  // EX §6, VU §5.10: X + 15, event, destination lines written by VU ops
constexpr uint32_t PERF_XOFF_ATT_QK_CYCLES  = 16u;  // EX §6, AT §10: X + 16, time, the MVM unit holds an ATTQK
constexpr uint32_t PERF_XOFF_ATT_PV_CYCLES  = 17u;  // EX §6, AT §10: X + 17, time, the MVM unit holds an ATTPV
constexpr uint32_t PERF_XOFF_ATT_PATH_STALL = 18u;  // EX §6, AT §10: X + 18, time, a cluster in an ATT op waits for the gather or the p broadcast
constexpr uint32_t PERF_XOFF_KVST_RMW       = 19u;  // EX §6, AT §10: X + 19, event, meta beats read-modify-written by KVST
constexpr uint32_t PERF_XOFF_SPM_RD_WAIT    = 20u;  // EX §6, CC §6.3: X + 20, time, client-cycles a raised SPM read request waits for its bank port
constexpr uint32_t PERF_XOFF_SPM_WR_WAIT    = 21u;  // EX §6, CC §6.3: X + 21, time, the same for write requests
constexpr uint32_t PERF_TRW_RECORDS         = 61u;  // EX §6, DB §8.1: event, trace records written to DRAM
constexpr uint32_t PERF_TRW_DROPS           = 62u;  // EX §6, DB §8.1: event, trace records dropped
constexpr uint32_t PERF_DBG_STOP_CYCLES     = 63u;  // EX §6, DB §8.1: time_nohalt, ACTIVE and (RV_HALTED or STOP_ACTIVE); a measurement is valid only if 0
constexpr uint32_t perf_x(uint32_t C) { return 5u + 2u * C; }   // first extension slot
constexpr uint32_t perf_ext(uint32_t C, uint32_t xoff) { return perf_x(C) + xoff; }
constexpr uint32_t perf_ch_rd_beats(uint32_t c) { return 5u + c; }
constexpr uint32_t perf_ch_wait(uint32_t C, uint32_t c) { return 5u + 1u * C + c; }

// -----------------------------------------------------------------------------------------------
// Instruction encodings of the base ops (spec ISA, EX §7.2)
// -----------------------------------------------------------------------------------------------
// NOP 0x00 (spec ISA, EX §7.1, owner spec)
constexpr uint32_t NOP_RSV_W0 = 0xFFFFFF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t NOP_RSV_W1 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W2 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W3 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W4 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W5 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W6 = 0xFFFFFFFFu;
constexpr uint32_t NOP_RSV_W7 = 0xFFFFFFFFu;
constexpr uint32_t NOP_WORDS  = 1u;           // words holding a field (w0 always)
// FENCE 0x7F (spec ISA, EX §7.1, owner spec)
constexpr uint32_t FENCE_RSV_W0 = 0xFFFFFF00u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t FENCE_RSV_W1 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W2 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W3 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W4 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W5 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W6 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_RSV_W7 = 0xFFFFFFFFu;
constexpr uint32_t FENCE_WORDS  = 1u;           // words holding a field (w0 always)
// LD 0x10 (spec ISA, EX §7.2, owner spec)
constexpr uint32_t LD_CH         = 0x00000F00u;  // w0[11:8] channel
constexpr uint32_t LD_CH_LSB     = 8u;
constexpr uint32_t LD_CH_W       = 4u;
constexpr uint32_t LD_CH_WORD    = 0u;
constexpr uint32_t LD_CH_BIT     = 8u;
constexpr uint32_t LD_SPM        = 0xFFFFFFFFu;  // w1 SPM destination line
constexpr uint32_t LD_SPM_LSB    = 0u;
constexpr uint32_t LD_SPM_W      = 32u;
constexpr uint32_t LD_SPM_WORD   = 1u;
constexpr uint32_t LD_SPM_BIT    = 32u;
constexpr uint32_t LD_DRAM       = 0xFFFFFFFFu;  // w2 DRAM source beat
constexpr uint32_t LD_DRAM_LSB   = 0u;
constexpr uint32_t LD_DRAM_W     = 32u;
constexpr uint32_t LD_DRAM_WORD  = 2u;
constexpr uint32_t LD_DRAM_BIT   = 64u;
constexpr uint32_t LD_LINES      = 0xFFFFFFFFu;  // w3 lines
constexpr uint32_t LD_LINES_LSB  = 0u;
constexpr uint32_t LD_LINES_W    = 32u;
constexpr uint32_t LD_LINES_WORD = 3u;
constexpr uint32_t LD_LINES_BIT  = 96u;
constexpr uint32_t LD_RSV_W0     = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t LD_RSV_W1     = 0x00000000u;
constexpr uint32_t LD_RSV_W2     = 0x00000000u;
constexpr uint32_t LD_RSV_W3     = 0x00000000u;
constexpr uint32_t LD_RSV_W4     = 0xFFFFFFFFu;
constexpr uint32_t LD_RSV_W5     = 0xFFFFFFFFu;
constexpr uint32_t LD_RSV_W6     = 0xFFFFFFFFu;
constexpr uint32_t LD_RSV_W7     = 0xFFFFFFFFu;
constexpr uint32_t LD_WORDS      = 15u;          // words holding a field (w0 always)
// ST 0x11 (spec ISA, EX §7.2, owner spec)
constexpr uint32_t ST_CH         = 0x00000F00u;  // w0[11:8] channel
constexpr uint32_t ST_CH_LSB     = 8u;
constexpr uint32_t ST_CH_W       = 4u;
constexpr uint32_t ST_CH_WORD    = 0u;
constexpr uint32_t ST_CH_BIT     = 8u;
constexpr uint32_t ST_SPM        = 0xFFFFFFFFu;  // w1 SPM source line
constexpr uint32_t ST_SPM_LSB    = 0u;
constexpr uint32_t ST_SPM_W      = 32u;
constexpr uint32_t ST_SPM_WORD   = 1u;
constexpr uint32_t ST_SPM_BIT    = 32u;
constexpr uint32_t ST_DRAM       = 0xFFFFFFFFu;  // w2 DRAM destination beat
constexpr uint32_t ST_DRAM_LSB   = 0u;
constexpr uint32_t ST_DRAM_W     = 32u;
constexpr uint32_t ST_DRAM_WORD  = 2u;
constexpr uint32_t ST_DRAM_BIT   = 64u;
constexpr uint32_t ST_LINES      = 0xFFFFFFFFu;  // w3 lines
constexpr uint32_t ST_LINES_LSB  = 0u;
constexpr uint32_t ST_LINES_W    = 32u;
constexpr uint32_t ST_LINES_WORD = 3u;
constexpr uint32_t ST_LINES_BIT  = 96u;
constexpr uint32_t ST_RSV_W0     = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t ST_RSV_W1     = 0x00000000u;
constexpr uint32_t ST_RSV_W2     = 0x00000000u;
constexpr uint32_t ST_RSV_W3     = 0x00000000u;
constexpr uint32_t ST_RSV_W4     = 0xFFFFFFFFu;
constexpr uint32_t ST_RSV_W5     = 0xFFFFFFFFu;
constexpr uint32_t ST_RSV_W6     = 0xFFFFFFFFu;
constexpr uint32_t ST_RSV_W7     = 0xFFFFFFFFu;
constexpr uint32_t ST_WORDS      = 15u;          // words holding a field (w0 always)
// QACT 0x18 (spec ISA, EX §7.2, owner spec)
constexpr uint32_t QACT_KIND            = 0x00000F00u;  // w0[11:8] QA kind
constexpr uint32_t QACT_KIND_LSB        = 8u;
constexpr uint32_t QACT_KIND_W          = 4u;
constexpr uint32_t QACT_KIND_WORD       = 0u;
constexpr uint32_t QACT_KIND_BIT        = 8u;
constexpr uint32_t QACT_DST             = 0xFFFFFFFFu;  // w1 QA destination line
constexpr uint32_t QACT_DST_LSB         = 0u;
constexpr uint32_t QACT_DST_W           = 32u;
constexpr uint32_t QACT_DST_WORD        = 1u;
constexpr uint32_t QACT_DST_BIT         = 32u;
constexpr uint32_t QACT_SRC             = 0xFFFFFFFFu;  // w2 f32 source line
constexpr uint32_t QACT_SRC_LSB         = 0u;
constexpr uint32_t QACT_SRC_W           = 32u;
constexpr uint32_t QACT_SRC_WORD        = 2u;
constexpr uint32_t QACT_SRC_BIT         = 64u;
constexpr uint32_t QACT_K               = 0xFFFFFFFFu;  // w3
constexpr uint32_t QACT_K_LSB           = 0u;
constexpr uint32_t QACT_K_W             = 32u;
constexpr uint32_t QACT_K_WORD          = 3u;
constexpr uint32_t QACT_K_BIT           = 96u;
constexpr uint32_t QACT_T               = 0xFFFFFFFFu;  // w4
constexpr uint32_t QACT_T_LSB           = 0u;
constexpr uint32_t QACT_T_W             = 32u;
constexpr uint32_t QACT_T_WORD          = 4u;
constexpr uint32_t QACT_T_BIT           = 128u;
constexpr uint32_t QACT_SRC_STRIDE      = 0xFFFFFFFFu;  // w5 lines; 0 = ceil(4K/64)
constexpr uint32_t QACT_SRC_STRIDE_LSB  = 0u;
constexpr uint32_t QACT_SRC_STRIDE_W    = 32u;
constexpr uint32_t QACT_SRC_STRIDE_WORD = 5u;
constexpr uint32_t QACT_SRC_STRIDE_BIT  = 160u;
// QACT_KIND takes the QA_* values
constexpr uint32_t QACT_RSV_W0 = 0xFFFFF000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t QACT_RSV_W1 = 0x00000000u;
constexpr uint32_t QACT_RSV_W2 = 0x00000000u;
constexpr uint32_t QACT_RSV_W3 = 0x00000000u;
constexpr uint32_t QACT_RSV_W4 = 0x00000000u;
constexpr uint32_t QACT_RSV_W5 = 0x00000000u;
constexpr uint32_t QACT_RSV_W6 = 0xFFFFFFFFu;
constexpr uint32_t QACT_RSV_W7 = 0xFFFFFFFFu;
constexpr uint32_t QACT_WORDS  = 0x3Fu;        // words holding a field (w0 always)
// MVM 0x20 (spec ISA, EX §7.2, owner spec)
constexpr uint32_t MVM_FMT                = 0x00000F00u;  // w0[11:8] weight format
constexpr uint32_t MVM_FMT_LSB            = 8u;
constexpr uint32_t MVM_FMT_W              = 4u;
constexpr uint32_t MVM_FMT_WORD           = 0u;
constexpr uint32_t MVM_FMT_BIT            = 8u;
constexpr uint32_t MVM_ACC                = 0x00001000u;  // w0[12] y += W.x
constexpr uint32_t MVM_ACC_LSB            = 12u;
constexpr uint32_t MVM_ACC_W              = 1u;
constexpr uint32_t MVM_ACC_WORD           = 0u;
constexpr uint32_t MVM_ACC_BIT            = 12u;
constexpr uint32_t MVM_LOCAL_SCALE        = 0x00002000u;  // w0[13] scale completed FP32 rows before export
constexpr uint32_t MVM_LOCAL_SCALE_LSB    = 13u;
constexpr uint32_t MVM_LOCAL_SCALE_W      = 1u;
constexpr uint32_t MVM_LOCAL_SCALE_WORD   = 0u;
constexpr uint32_t MVM_LOCAL_SCALE_BIT    = 13u;
constexpr uint32_t MVM_LOCAL_SOFTCAP      = 0x00004000u;  // w0[14] scale, TANH, scale before export; dense destination stride
constexpr uint32_t MVM_LOCAL_SOFTCAP_LSB  = 14u;
constexpr uint32_t MVM_LOCAL_SOFTCAP_W    = 1u;
constexpr uint32_t MVM_LOCAL_SOFTCAP_WORD = 0u;
constexpr uint32_t MVM_LOCAL_SOFTCAP_BIT  = 14u;
constexpr uint32_t MVM_DST                = 0xFFFFFFFFu;  // w1 destination line
constexpr uint32_t MVM_DST_LSB            = 0u;
constexpr uint32_t MVM_DST_W              = 32u;
constexpr uint32_t MVM_DST_WORD           = 1u;
constexpr uint32_t MVM_DST_BIT            = 32u;
constexpr uint32_t MVM_X                  = 0xFFFFFFFFu;  // w2 QA activation line
constexpr uint32_t MVM_X_LSB              = 0u;
constexpr uint32_t MVM_X_W                = 32u;
constexpr uint32_t MVM_X_WORD             = 2u;
constexpr uint32_t MVM_X_BIT              = 64u;
constexpr uint32_t MVM_W_OFF              = 0xFFFFFFFFu;  // w3 weight offset (beats)
constexpr uint32_t MVM_W_OFF_LSB          = 0u;
constexpr uint32_t MVM_W_OFF_W            = 32u;
constexpr uint32_t MVM_W_OFF_WORD         = 3u;
constexpr uint32_t MVM_W_OFF_BIT          = 96u;
constexpr uint32_t MVM_N                  = 0xFFFFFFFFu;  // w4 rows
constexpr uint32_t MVM_N_LSB              = 0u;
constexpr uint32_t MVM_N_W                = 32u;
constexpr uint32_t MVM_N_WORD             = 4u;
constexpr uint32_t MVM_N_BIT              = 128u;
constexpr uint32_t MVM_K                  = 0x0000FFFFu;  // w5[15:0]
constexpr uint32_t MVM_K_LSB              = 0u;
constexpr uint32_t MVM_K_W                = 16u;
constexpr uint32_t MVM_K_WORD             = 5u;
constexpr uint32_t MVM_K_BIT              = 160u;
constexpr uint32_t MVM_T                  = 0x000F0000u;  // w5[19:16]
constexpr uint32_t MVM_T_LSB              = 16u;
constexpr uint32_t MVM_T_W                = 4u;
constexpr uint32_t MVM_T_WORD             = 5u;
constexpr uint32_t MVM_T_BIT              = 176u;
constexpr uint32_t MVM_DST_STRIDE         = 0xFFFFFFFFu;  // w6 lines; 0 = ceil(4N/64); with LOCAL_SOFTCAP, first FP32 coefficient
constexpr uint32_t MVM_DST_STRIDE_LSB     = 0u;
constexpr uint32_t MVM_DST_STRIDE_W       = 32u;
constexpr uint32_t MVM_DST_STRIDE_WORD    = 6u;
constexpr uint32_t MVM_DST_STRIDE_BIT     = 192u;
constexpr uint32_t MVM_SCALE              = 0xFFFFFFFFu;  // w7 FP32 coefficient with LOCAL_SCALE; final FP32 coefficient with LOCAL_SOFTCAP
constexpr uint32_t MVM_SCALE_LSB          = 0u;
constexpr uint32_t MVM_SCALE_W            = 32u;
constexpr uint32_t MVM_SCALE_WORD         = 7u;
constexpr uint32_t MVM_SCALE_BIT          = 224u;
// MVM_FMT takes the FMT_* values
constexpr uint32_t MVM_RSV_W0 = 0xFFFF8000u;  // must be 0 (ILLEGAL_OP)
constexpr uint32_t MVM_RSV_W1 = 0x00000000u;
constexpr uint32_t MVM_RSV_W2 = 0x00000000u;
constexpr uint32_t MVM_RSV_W3 = 0x00000000u;
constexpr uint32_t MVM_RSV_W4 = 0x00000000u;
constexpr uint32_t MVM_RSV_W5 = 0xFFF00000u;
constexpr uint32_t MVM_RSV_W6 = 0x00000000u;
constexpr uint32_t MVM_RSV_W7 = 0x00000000u;
constexpr uint32_t MVM_WORDS  = 0xFFu;        // words holding a field (w0 always)

}  // namespace tsr

#endif  // TSR_MMIO_HPP

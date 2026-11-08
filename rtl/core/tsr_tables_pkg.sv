// GENERATED FILE, DO NOT EDIT: produced by sw/tables/gen_tables.py from sw/tables/tsr_tables.py.
//
// tsr_tables_pkg.sv — Tesseract global tables (EX-30) for the RTL: CSR offsets, opcodes,
// units, error codes, PERF indices, bit-field layouts and instruction field positions; the
// opcode -> unit table as functions. C mirror: sw/sim/include/tsr_tables.h (same names).

package tsr_tables_pkg;

  // A catalogue of constants: no module uses all of them, so UNUSEDPARAM is off for this package.
  /* verilator lint_off UNUSEDPARAM */

  // ---------------------------------------------------------------------------------------------
  // Scalar constants (CONSTANTS)
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned TSR_ID_MAGIC              = 32'h5453_5231;  // EX §5.1: ID register: "TSR1"
  localparam int unsigned TSR_ISA_MAJOR             = 1;              // EX §7.5: VERSION[15:8]: Tesseract Core capability major
  localparam int unsigned TSR_ISA_MINOR             = 0;              // EX §7.5: VERSION[7:0]: Tesseract Core capability minor
  localparam int unsigned TSR_VERSION_VALUE         = 32'h0000_0100;  // EX §7.5: VERSION register: Tesseract Core capability 1.0
  localparam int unsigned TSR_CTL_VERSION           = 1;              // EX §5.1: CTL_ID[7:0]: control-core revision; 0 = no control core
  localparam int unsigned TSR_ADAPTER_MAJOR         = 1;              // EX §9.2: ADAPTER_VERSION[31:16] (ADAPTER_VERSION_MAJOR field)
  localparam int unsigned TSR_DBG_VERSION           = 1;              // DB §10.1: DBG_ID[7:0]: DBG register map; DBG_ID = 0 means the DBG block is absent
  localparam int unsigned TSR_CANON_NAN             = 32'h7FC0_0000;  // VU §4.1: the only NaN an fp result has
  localparam int unsigned TSR_INSN_WORDS            = 8;              // EX §7.1: an instruction is 256 bits, w0..w7, little-endian
  localparam int unsigned TSR_INSN_BYTES            = 32;             // EX §7.1
  localparam int unsigned TSR_INSN_BITS             = 256;            // EX §7.1
  localparam int unsigned TSR_X16_LSB               = 16;             // EX §7.1: x16 = w0[31:16]; x16 bit i is w0 bit 16 + i; 0 unless the op defines it
  localparam int unsigned TSR_X16_W                 = 16;             // EX §7.1
  localparam int unsigned TSR_LQ_DEPTH              = 8;              // EX §4.2: U200 launch-queue depth (LQ_CFG[7:0])
  localparam int unsigned TSR_LCQ_DEPTH             = 8;              // EX §4.2: U200 completion-ring depth (LQ_CFG[15:8])
  localparam int unsigned TSR_LOG_DEPTH             = 256;            // EX §4.2: U200 log-ring words (LQ_CFG[27:16])
  localparam int unsigned TSR_LCQ_DEPTH_MAX         = 8;              // EX §4.2: the CSR map limits LCQ_DEPTH to 8
  localparam int unsigned TSR_LOG_DEPTH_MAX         = 256;            // EX §4.2: the CSR map limits LOG_DEPTH to 256
  localparam int unsigned TSR_LCQ_REC_BYTES         = 32;             // EX §4.2: record k at LCQ_REC + 32 x (k mod LCQ_DEPTH)
  localparam int unsigned TSR_ARGBLK_MAX_BYTES      = 1024;           // EX §4.3: argument block limit (ARG0 handle of a kernel with the argument-block flag)
  localparam int unsigned TSR_ARGBLK_MAX_LINES      = 16;             // SW §6.2: argblk_lines of a kernel table entry: 1..16
  localparam int unsigned TSR_SEQ_HALF              = 32'h8000_0000;  // EX §4.9: launch q is covered by watermark s iff (s - q) mod 2^32 < SEQ_HALF
  localparam int unsigned TSR_PANIC_MCAUSE          = 24;             // EX §4.7, DB §6.3: tsr::panic enters the trap handler with this mcause, mtval = code
  localparam int unsigned TSR_BOOT_ENTRY_OFF        = 32'h0000_0080;  // EX §3.4: the RISC-V fetches at RV_BOOT + 0x80
  localparam int unsigned TSR_RV_BOOT_ALIGN         = 256;            // EX §5.1: RV_BOOT alignment ([7:0] read 0)
  localparam int unsigned TSR_CRC32_POLY            = 32'hEDB8_8320;  // EX §3.3: LOAD_CRC: zlib crc32 (reflected polynomial, init and final XOR 0xFFFFFFFF)
  localparam int unsigned TSR_LOAD_TIMEOUT_CYCLES   = 4096;           // EX §3.3: core cycles a LOAD_DATA word or a VERIFY read may take before LOAD_TIMEOUT
  localparam int unsigned TSR_LOGFMT_VMA            = 32'hF000_0000;  // SW §6.2, DB §6.1: VMA of the non-loaded .tsr_logfmt section (FMT log records point into it)
  localparam int unsigned TSR_WDOG_L_RESET          = 24;             // EX §5.1, §8.6: WDOG_CFG reset value: 2^24 core cycles without progress
  localparam int unsigned TSR_SR_RV_WAIT_CYCLES     = 32'h0001_0000;  // EX §8.5: SOFT_RESET step 1 waits at most 2^16 core cycles for RV_RESET_DONE
  localparam int unsigned TSR_DMI_TIMEOUT_CYCLES    = 4096;           // EX §5.1: core cycles a DMI access may take before DMI_TIMEOUT
  localparam int unsigned TSR_BRIDGE_TIMEOUT_CYCLES = 4096;           // CC §5.3: clk_rv cycles a core-window or SPM-bridge load may take before a load fault
  localparam int unsigned TSR_ISSUE_WAIT_MAX        = 32'h0001_0000;  // CC §7.3: clk_rv cycles a push without credit waits before a store fault and PUSH_TIMEOUT
  localparam int unsigned TSR_TICKET_NONE           = 32'hF000_0000;  // CC §5.1, EX §4.4: unit field 15: no dependency / no failing ticket
  localparam int unsigned TSR_NUM_CH_MAX            = 16;             // spec ISA: LD / ST name the channel in the 4-bit sub field (CFG0.NUM_CH, CH_BASE_LO/HI)
  localparam int unsigned TSR_NUM_UNITS             = 6;              // EX §7.3: unit ids 0..5 (6..13 reserved, 14 marker, 15 none)
  localparam int unsigned TSR_UNIT_SLOTS            = 8;              // EX §5.4: per-unit register slots u = 0..7 (absent units read 0)
  localparam int unsigned TSR_Q_DEPTH_FIELD_W       = 5;              // EX §5.1: Q_DEPTH: depth of unit u at [5u+4:5u]
  localparam int unsigned TSR_CREDIT_FIELD_W        = 4;              // CC §5.1: ISSUE_CREDIT: unit u at [4u+3:4u], saturating at 15
  localparam int unsigned TSR_ISSUE_FIFO_DEPTH      = 32;             // CC §7.2: issue-FIFO entries (LUTRAM, clk_rv -> core)
  localparam int unsigned TSR_ERR_W                 = 4;              // EX §8.1: internal error-code width (codes 0..15); CSR fields are 8 bits
  localparam int unsigned TSR_PERF_SLOTS            = 64;             // EX §5.1: PERF_0..PERF_63, 64-bit each
  localparam int unsigned TSR_PERF_CH_BASE          = 5;              // EX §6: CH_RD_BEATS[c] = 5 + c, CH_WAIT[c] = 5 + C + c, X = 5 + 2C
  localparam int unsigned TSR_ITCM_BYTES            = 32'h0002_0000;  // CC §4.1: U200 ITCM (RV_CFG[15:0] in KiB)
  localparam int unsigned TSR_DTCM_BYTES            = 32'h0001_0000;  // CC §4.1: U200 DTCM (RV_CFG[31:16] in KiB)
  localparam int unsigned TSR_SPM_BYTES_U200        = 32'h0040_0000;  // CC §4.1: U200 SPM size: the SPM windows span SPM_BYTES
  localparam int unsigned TSR_DBG_NTBP              = 4;              // DB §10.3: tile breakpoints
  localparam int unsigned TSR_DBG_NRVWP             = 2;              // DB §10.3: RISC-V data watchpoints
  localparam int unsigned TSR_HIST_DEPTH            = 512;            // DB §10.3: dispatch-history entries
  localparam int unsigned TSR_REC_DEPTH             = 64;             // DB §10.3: completion-record FIFO entries
  localparam int unsigned TSR_REC_IRQ_LEVEL         = 32;             // DB §8.2: irq_fast[4] while REC_COUNT >= 32 (REC_CFG.IRQ_HALF)
  localparam int unsigned TSR_MATCHQ_DEPTH          = 64;             // DB §5.2: tile-breakpoint match queue
  localparam int unsigned TSR_TRW_CH                = 0;              // DB §10.3, §8.3: channel of the trace-writer ring (in the ch0 IO region)
  localparam int unsigned TSR_TRW_RECORD_BYTES      = 32;             // DB §8.3: trace-writer record, two per 64-byte beat
  localparam int unsigned TSR_TRW_RING_DEFAULT_MIB  = 64;             // DB §8.3, EX §11: TSR_PROF_TRW_MIB default
  localparam int unsigned TSR_DBG_INFO_MAGIC        = 32'h4742_4454;  // DB §6.3: __tsr_dbg word 0: "TDBG"
  localparam int unsigned TSR_CRASH_MAGIC           = 32'h5041_5254;  // DB §6.3: __tsr_crash word 0: "TRAP"
  localparam int unsigned TSR_CRASH_LOG_WORDS       = 44;             // DB §6.1: TRAP log record payload: words 0..43 of __tsr_crash
  localparam int unsigned TSR_VL                    = 16;             // VU §5.10: CFG2.VL: elements per SPM line
  localparam int unsigned TSR_NSREG                 = 16;             // VU §4.4: s0..s15
  localparam int unsigned TSR_VU_FRONT_LANES        = 16;             // VU §5.10: CFG3[4:0] (U200)
  localparam int unsigned TSR_VU_BACK_LANES         = 8;              // VU §5.10: CFG3[9:5] (U200)
  localparam int unsigned TSR_VU_LINE_FIELD_W       = 24;             // VU §5.1: SPM line addresses and element counts m are w[23:0] with w[31:24] = 0
  localparam int unsigned TSR_ROPE_MAX_HEAD_DIM     = 1024;           // VU §5.4: head_dim is a multiple of 16 in [16, 1024]
  localparam int unsigned TSR_TOPK_MAX_K            = 16;             // VU §5.5: k in 1..16
  localparam int unsigned TSR_VU_ACCEPT_AHEAD       = 2;              // VU §7.8, CC §9.4: VU ops held at once (I3 for unit 3)
  localparam int unsigned TSR_GDN_MIN_M             = 64;             // VU §7.12.4: GDN d_k in [64, 256]: at least four beats of the widest build (16 lanes) per row
  localparam int unsigned TSR_GDN_MAX_M             = 256;            // VU §7.12.4: GDN d_k at most 256: the 48-bit DSP accumulator bound
  localparam int unsigned TSR_GDN_MAX_ROWS          = 1024;           // VU §7.12.4: GDN d_v in [1, 1024]: the bv RAM
  localparam int unsigned TSR_M_FLOOR               = 32'hF180_0000;  // AT §6.3: -2^100: row-maximum floor of ATTQK MAXOUT and AMASK
  localparam int unsigned TSR_ANORM_L_FLOOR         = 32'h0080_0000;  // AT §5.7: 2^-126: ANORM divides by MAXT(l, 2^-126)
  localparam int unsigned TSR_ATT_GROUP_CELLS       = 16;             // AT §4.2: KV cells per group (groups round-robin over the channels)
  localparam int unsigned TSR_ATT_MAX_GROUPS        = 4095;           // AT §5.2: n_groups in [1, 4095]
  localparam int unsigned TSR_ATT_MAX_G             = 64;             // AT §5: G (heads per KV head) in 1..64
  localparam int unsigned TSR_ATT_MAX_PASS_ROWS     = 128;            // AT §5.8: (E/D) x GS x T_q pass rows at most
  localparam int unsigned TSR_ATT_MAX_R             = 1024;           // AT §5.8: R in [1, 1024] for ASMAX, AMASK, AMERGE, ANORM
  localparam int unsigned TSR_ATT_MAX_LINES         = 4095;           // AT §5.4: ASMAX / AMASK n_lines in [1, 4095]
  localparam int unsigned TSR_KVST_MAX_ROWS         = 64;             // AT §5.1: n_rows in 1..64
  localparam int unsigned TSR_PCAP_U200             = 9344;           // AT §5.8: p capacity of one cluster slot column in f32: QA_LINES(A256, K_MAX) x 16

  // ---------------------------------------------------------------------------------------------
  // Shared value sets (ENUMS)
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned TSR_FMT_Q8_0   = 0;  // spec ISA, VU §5.6: weight format Q8_0 (MVM, DQROW sub)
  localparam int unsigned TSR_FMT_Q4_K   = 1;  // spec ISA, VU §5.6: weight format Q4_K; 2..15 BAD_FMT
  localparam int unsigned TSR_QA_A32     = 0;  // spec ISA: QACT kind: block_q8_0 rules, block 32
  localparam int unsigned TSR_QA_A256    = 1;  // spec ISA: QACT kind: block_q8_K rules, block 256, with bsums
  localparam int unsigned TSR_KVFMT_Q8_0 = 0;  // AT §5.2: KV format of ATTQK / ATTPV sub; else BAD_FMT
  localparam int unsigned TSR_DCODE_D64  = 0;  // AT §5: head dimension D = 64 x 2^dcode
  localparam int unsigned TSR_DCODE_D128 = 1;  // AT §5
  localparam int unsigned TSR_DCODE_D256 = 2;  // AT §5
  localparam int unsigned TSR_DCODE_D512 = 3;  // AT §5

  // ---------------------------------------------------------------------------------------------
  // Field access helpers
  // ---------------------------------------------------------------------------------------------

  // ---------------------------------------------------------------------------------------------
  // Host CSR map: core CSR port, U200 s_axi_control 0x1000 + offset (EX §5.1)
  // ---------------------------------------------------------------------------------------------
  localparam logic [11:0] TSR_CSR_ID              = 12'h000;        // [R] EX §5.1: "TSR1" (unchanged)
  localparam logic [31:0] TSR_CSR_ID_RESET        = 32'h5453_5231;
  localparam logic [11:0] TSR_CSR_VERSION         = 12'h004;        // [R] EX §5.1, §7.5: Tesseract Core capability: [15:8] major 1, [7:0] minor 0
  localparam logic [31:0] TSR_CSR_VERSION_RESET   = 32'h0000_0100;
  localparam logic [11:0] TSR_CSR_CFG0            = 12'h008;        // [R] EX §5.1: NUM_CH, LANES, T_MAX (unchanged)
  localparam logic [11:0] TSR_CSR_CFG1            = 12'h00C;        // [R] EX §5.1: SPM size in lines (unchanged)
  localparam logic [11:0] TSR_CSR_CFG2            = 12'h010;        // [R] EX §5.1: VL, NSREG, K_MAX/256 (unchanged)
  localparam logic [11:0] TSR_CSR_CAPS            = 12'h014;        // [R] EX §5.1, §7.5: capability bits
  localparam logic [11:0] TSR_CSR_CFG3            = 12'h018;        // [R] EX §5.1, §7.5: [15:0] VU, [31:16] AT
  localparam logic [11:0] TSR_CSR_CTL_ID          = 12'h01C;        // [R] EX §5.1: control-core revision and features; 0 = no control core
  localparam logic [11:0] TSR_CSR_CTRL            = 12'h020;        // [RW/W1] EX §5.1, §5.2: device control
  localparam logic [11:0] TSR_CSR_STATUS          = 12'h024;        // [R/W1C] EX §5.1, §5.3: device status; W1C clears the sticky bits only
  localparam logic [11:0] TSR_CSR_RV_BOOT         = 12'h028;        // [RW] EX §5.1: boot vector, taken when RV_RUN rises
  localparam logic [11:0] TSR_CSR_RV_FREQ         = 12'h02C;        // [R] EX §5.1: clk_rv cycles in the last 2^16 core cycles; 0 = no clk_rv
  localparam logic [11:0] TSR_CSR_RV_CFG          = 12'h030;        // [R] EX §5.1: [15:0] ITCM KiB, [31:16] DTCM KiB
  localparam logic [11:0] TSR_CSR_WDOG_CFG        = 12'h034;        // [RW] EX §5.1, §8.6: unit watchdog limit
  localparam logic [31:0] TSR_CSR_WDOG_CFG_RESET  = 32'h0000_0018;
  localparam logic [11:0] TSR_CSR_RT_STATE        = 12'h038;        // [R] EX §5.1, §4.5: copy of the runtime's RT_STATE
  localparam logic [11:0] TSR_CSR_RT_INFO         = 12'h03C;        // [R] EX §5.1, §4.5: copy of the runtime's RT_INFO
  localparam logic [11:0] TSR_CSR_LQ_W_BASE       = 12'h040;        // [RW] EX §5.1, §4.3: descriptor staging LQ_W0..LQ_W7; kept after a push
  localparam int unsigned TSR_CSR_LQ_W_N          = 8;
  localparam int unsigned TSR_CSR_LQ_W_STRIDE     = 4;
  localparam logic [11:0] TSR_CSR_LQ_PUSH         = 12'h060;        // [W] EX §5.1: push the staged descriptor (value ignored); dropped with LQ_OVERFLOW when full
  localparam logic [11:0] TSR_CSR_LQ_SEQ          = 12'h064;        // [R] EX §5.1, §4.2: PUSHED
  localparam logic [11:0] TSR_CSR_LQ_TAKEN        = 12'h068;        // [R] EX §5.1, §4.2: TAKEN
  localparam logic [11:0] TSR_CSR_LQ_CANCEL       = 12'h06C;        // [RW] EX §5.1, §4.9: cancel watermark
  localparam logic [11:0] TSR_CSR_LQ_STOP         = 12'h070;        // [RW] EX §5.1, §4.9: stop watermark
  localparam logic [11:0] TSR_CSR_LCQ_WR          = 12'h074;        // [R] EX §5.1, §4.2: records posted
  localparam logic [11:0] TSR_CSR_LCQ_RD          = 12'h078;        // [RW] EX §5.1, §4.2: records consumed by the host (EX §4.2 write rule)
  localparam logic [11:0] TSR_CSR_LQ_CFG          = 12'h07C;        // [R] EX §5.1: ring depths and watermark valid bits
  localparam logic [11:0] TSR_CSR_PERF_INFO       = 12'h080;        // [R] EX §5.1: counters implemented, extension index X
  localparam logic [11:0] TSR_CSR_LCQ_ANSWERED    = 12'h084;        // [R] EX §5.1, §4.2: ANSWERED
  localparam logic [11:0] TSR_CSR_Q_DEPTH         = 12'h088;        // [R] EX §5.1: queue depth per unit; 0 = absent
  localparam logic [11:0] TSR_CSR_LOG_WR          = 12'h090;        // [R] EX §5.1, §4.10: log words accepted
  localparam logic [11:0] TSR_CSR_LOG_RD          = 12'h094;        // [RW] EX §5.1, §4.10: log words consumed by the host
  localparam logic [11:0] TSR_CSR_LOG_CFG         = 12'h098;        // [RW] EX §5.1, §4.10: log configuration
  localparam logic [31:0] TSR_CSR_LOG_CFG_RESET   = 32'h0000_00F1;
  localparam logic [11:0] TSR_CSR_LOG_DROPPED     = 12'h09C;        // [R] EX §5.1: words dropped, saturating; cleared by SOFT_RESET
  localparam logic [11:0] TSR_CSR_LOAD_ADDR       = 12'h0C0;        // [RW] EX §5.1, §3.3: TCM byte address (word aligned); a write restarts LOAD_CRC
  localparam logic [11:0] TSR_CSR_LOAD_DATA       = 12'h0C4;        // [RW] EX §5.1, §3.3: write: store the word, address += 4; read: the word (only while RV_RUN = 0 and not SR_BUSY)
  localparam logic [11:0] TSR_CSR_LOAD_CRC        = 12'h0C8;        // [R] EX §5.1, §3.3: CRC-32 of the words accepted, or read by VERIFY
  localparam logic [11:0] TSR_CSR_LOAD_COUNT      = 12'h0CC;        // [RW] EX §5.1, §3.3: word count for VERIFY
  localparam logic [11:0] TSR_CSR_LOAD_CTRL       = 12'h0D0;        // [W] EX §5.1, §3.3: bit0 VERIFY
  localparam logic [11:0] TSR_CSR_LOAD_STATUS     = 12'h0D4;        // [R] EX §5.1, §3.3: bit0 VERIFY_BUSY
  localparam logic [11:0] TSR_CSR_PERF_LO_BASE    = 12'h100;        // [R] EX §5.1, §6: counter i bits [31:0]; a host LO read snapshots the counter for the next HI read
  localparam int unsigned TSR_CSR_PERF_LO_N       = 64;
  localparam int unsigned TSR_CSR_PERF_LO_STRIDE  = 8;
  localparam logic [11:0] TSR_CSR_PERF_HI_BASE    = 12'h104;        // [R] EX §5.1, §6: counter i bits [63:32] (the snapshot)
  localparam int unsigned TSR_CSR_PERF_HI_N       = 64;
  localparam int unsigned TSR_CSR_PERF_HI_STRIDE  = 8;
  localparam logic [11:0] TSR_CSR_LCQ_REC_BASE    = 12'h300;        // [R] EX §5.1, §4.2: completion ring: record k at 0x300 + 32 (k mod LCQ_DEPTH), word j at +4j
  localparam int unsigned TSR_CSR_LCQ_REC_N       = 8;
  localparam int unsigned TSR_CSR_LCQ_REC_STRIDE  = 32;
  localparam int unsigned TSR_CSR_LCQ_REC_WORDS   = 8;
  localparam logic [11:0] TSR_CSR_LOG_RING_BASE   = 12'h400;        // [R] EX §5.1, §4.10: log word k at 0x400 + 4 (k mod LOG_DEPTH)
  localparam int unsigned TSR_CSR_LOG_RING_N      = 256;
  localparam int unsigned TSR_CSR_LOG_RING_STRIDE = 4;
  localparam logic [11:0] TSR_CSR_DBG_BASE        = 12'h800;        // [DB] EX §5.1, DB §10.1: DBG block (DBG_CSR rows); undefined offsets read 0
  localparam int unsigned TSR_CSR_DBG_BYTES       = 768;
  // 0xB00..0xBFC reserved (EX §5.1)
  localparam logic [11:0] TSR_CSR_DMI_BASE          = 12'hC00;  // [R/W] EX §5.1, DB §3.2: debug module register r at 0xC00 + 4r; acts only while DBG_CFG.DMI_EN = 1
  localparam int unsigned TSR_CSR_DMI_N             = 128;
  localparam int unsigned TSR_CSR_DMI_STRIDE        = 4;
  localparam logic [11:0] TSR_CSR_Q_DONE_BASE       = 12'hE00;  // [R] EX §5.4: DONE_SEQ[u], 28 bits
  localparam int unsigned TSR_CSR_Q_DONE_N          = 8;
  localparam int unsigned TSR_CSR_Q_DONE_STRIDE     = 4;
  localparam logic [11:0] TSR_CSR_Q_ISSUED_BASE     = 12'hE20;  // [R] EX §5.4: ISSUED_SEQ[u], 28 bits (synchronised copy)
  localparam int unsigned TSR_CSR_Q_ISSUED_N        = 8;
  localparam int unsigned TSR_CSR_Q_ISSUED_STRIDE   = 4;
  localparam logic [11:0] TSR_CSR_Q_STATE_BASE      = 12'hE40;  // [R] EX §5.4: queue state of unit u
  localparam int unsigned TSR_CSR_Q_STATE_N         = 8;
  localparam int unsigned TSR_CSR_Q_STATE_STRIDE    = 4;
  localparam logic [11:0] TSR_CSR_Q_FIFO            = 12'hE60;  // [R] EX §5.4: issue-FIFO entries waiting for the dispatcher
  localparam logic [11:0] TSR_CSR_Q_PIDX            = 12'hE64;  // [R] EX §5.4: push index of the dispatcher's next ticketed entry (= DBG_NEXT_PIDX); firmware never uses it
  localparam logic [11:0] TSR_CSR_Q_ERR_SEQ_BASE    = 12'hE80;  // [R] EX §5.4: first error record of unit u: ERR_SEQ
  localparam int unsigned TSR_CSR_Q_ERR_SEQ_N       = 8;
  localparam int unsigned TSR_CSR_Q_ERR_SEQ_STRIDE  = 16;
  localparam logic [11:0] TSR_CSR_Q_ERR_CODE_BASE   = 12'hE84;  // [R] EX §5.4: ERR_CODE ([7:0])
  localparam int unsigned TSR_CSR_Q_ERR_CODE_N      = 8;
  localparam int unsigned TSR_CSR_Q_ERR_CODE_STRIDE = 16;
  localparam logic [11:0] TSR_CSR_Q_ERR_INFO_BASE   = 12'hE88;  // [R] EX §5.4: ERR_INFO
  localparam int unsigned TSR_CSR_Q_ERR_INFO_N      = 8;
  localparam int unsigned TSR_CSR_Q_ERR_INFO_STRIDE = 16;
  localparam logic [11:0] TSR_CSR_Q_ERR_W0_BASE     = 12'hE8C;  // [R] EX §5.4: ERR_W0
  localparam int unsigned TSR_CSR_Q_ERR_W0_N        = 8;
  localparam int unsigned TSR_CSR_Q_ERR_W0_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_Q_ERR_PIDX_BASE   = 12'hF00;  // [R] EX §5.4: push index of unit u's first error
  localparam int unsigned TSR_CSR_Q_ERR_PIDX_N      = 8;
  localparam int unsigned TSR_CSR_Q_ERR_PIDX_STRIDE = 4;
  // 0xF20..0xFFC reserved (EX §5.1)

  // ---------------------------------------------------------------------------------------------
  // DBG block: host CSR 0x800..0xAFC (DB §10.1)
  // ---------------------------------------------------------------------------------------------
  localparam logic [11:0] TSR_CSR_DBG_ID               = 12'h800;        // [R] DB §10.1: version and features; 0 = block absent
  localparam logic [11:0] TSR_CSR_DBG_CTRL             = 12'h804;        // [W] DB §10.1: pulses, reads 0
  localparam logic [11:0] TSR_CSR_DBG_CFG              = 12'h808;        // [RW] DB §10.1: configuration
  localparam logic [31:0] TSR_CSR_DBG_CFG_RESET        = 32'h0000_0030;
  localparam logic [11:0] TSR_CSR_DBG_STATUS           = 12'h80C;        // [R/W1C] DB §10.1: levels and sticky bits
  localparam logic [11:0] TSR_CSR_DBG_STOP_PIDX        = 12'h810;        // [R] DB §10.1: T (valid while STOP_ACTIVE)
  localparam logic [11:0] TSR_CSR_DBG_STOP_AT          = 12'h814;        // [RW] DB §10.1: write P: S <- P, valid; read: S (valid iff S_VALID)
  localparam logic [11:0] TSR_CSR_DBG_STEP_N           = 12'h818;        // [RW] DB §10.1: entries per QSTEP
  localparam logic [31:0] TSR_CSR_DBG_STEP_N_RESET     = 32'h0000_0001;
  localparam logic [11:0] TSR_CSR_DBG_NEXT_PIDX        = 12'h81C;        // [R] DB §10.1: push index the dispatcher assigns to its next ticketed entry
  localparam logic [11:0] TSR_CSR_DBG_STEP_PIDX        = 12'h820;        // [R] DB §10.1: X of the current / last QSTEP
  localparam logic [11:0] TSR_CSR_DBG_MATCHQ           = 12'h824;        // [R] DB §10.1: pending match thresholds
  localparam logic [11:0] TSR_CSR_DBG_MTIME_LO         = 12'h828;        // [R] DB §10.1: live core time (read HI, LO, HI)
  localparam logic [11:0] TSR_CSR_DBG_MTIME_HI         = 12'h82C;        // [R] DB §10.1
  localparam logic [11:0] TSR_CSR_DBG_QSTATE           = 12'h834;        // [R] DB §10.1: dispatcher state, outstanding entries
  localparam logic [11:0] TSR_CSR_DBG_RV_STATE         = 12'h840;        // [R] DB §10.1: RISC-V state and sample sequence
  localparam logic [11:0] TSR_CSR_DBG_RV_PC            = 12'h844;        // [R] DB §10.1, §6.2: last crash-dump sample: current PC
  localparam logic [11:0] TSR_CSR_DBG_RV_NPC           = 12'h848;        // [R] DB §10.1, §6.2: next PC
  localparam logic [11:0] TSR_CSR_DBG_RV_DADDR         = 12'h84C;        // [R] DB §10.1, §6.2: last data address
  localparam logic [11:0] TSR_CSR_DBG_RV_EPC           = 12'h850;        // [R] DB §10.1, §6.2: exception PC
  localparam logic [11:0] TSR_CSR_DBG_RV_EADDR         = 12'h854;        // [R] DB §10.1, §6.2: exception address
  localparam logic [11:0] TSR_CSR_DBG_RV_WPADDR        = 12'h858;        // [R] DB §10.1: address of the last RV-watchpoint hit
  localparam logic [11:0] TSR_CSR_DBG_RVWP_CTL_BASE    = 12'h860;        // [RW] DB §10.1: RV watchpoint j < 2
  localparam int unsigned TSR_CSR_DBG_RVWP_CTL_N       = 2;
  localparam int unsigned TSR_CSR_DBG_RVWP_CTL_STRIDE  = 16;
  localparam logic [11:0] TSR_CSR_DBG_RVWP_LO_BASE     = 12'h864;        // [RW] DB §10.1: byte range [LO, HI)
  localparam int unsigned TSR_CSR_DBG_RVWP_LO_N        = 2;
  localparam int unsigned TSR_CSR_DBG_RVWP_LO_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_DBG_RVWP_HI_BASE     = 12'h868;        // [RW] DB §10.1
  localparam int unsigned TSR_CSR_DBG_RVWP_HI_N        = 2;
  localparam int unsigned TSR_CSR_DBG_RVWP_HI_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_DBG_RVWP_HITS_BASE   = 12'h86C;        // [R/WC] DB §10.1: hits; a write clears
  localparam int unsigned TSR_CSR_DBG_RVWP_HITS_N      = 2;
  localparam int unsigned TSR_CSR_DBG_RVWP_HITS_STRIDE = 16;
  localparam logic [11:0] TSR_CSR_DBG_TBP_CTL_BASE     = 12'h880;        // [RW] DB §10.1: tile breakpoint i < 4
  localparam int unsigned TSR_CSR_DBG_TBP_CTL_N        = 4;
  localparam int unsigned TSR_CSR_DBG_TBP_CTL_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_DBG_TBP_A_BASE       = 12'h884;        // [RW] DB §10.1, §5.3
  localparam int unsigned TSR_CSR_DBG_TBP_A_N          = 4;
  localparam int unsigned TSR_CSR_DBG_TBP_A_STRIDE     = 16;
  localparam logic [11:0] TSR_CSR_DBG_TBP_B_BASE       = 12'h888;        // [RW] DB §10.1, §5.3
  localparam int unsigned TSR_CSR_DBG_TBP_B_N          = 4;
  localparam int unsigned TSR_CSR_DBG_TBP_B_STRIDE     = 16;
  localparam logic [11:0] TSR_CSR_DBG_TBP_HITS_BASE    = 12'h88C;        // [R/WC] DB §10.1: matches; a write clears
  localparam int unsigned TSR_CSR_DBG_TBP_HITS_N       = 4;
  localparam int unsigned TSR_CSR_DBG_TBP_HITS_STRIDE  = 16;
  localparam logic [11:0] TSR_CSR_DBG_RDOUT_BASE       = 12'h8C0;        // [R] DB §10.1, §5.5: outstanding read bursts: channel c in byte c mod 4 of word c / 4
  localparam int unsigned TSR_CSR_DBG_RDOUT_N          = 4;
  localparam int unsigned TSR_CSR_DBG_RDOUT_STRIDE     = 4;
  localparam logic [11:0] TSR_CSR_DBG_WROUT_BASE       = 12'h8D0;        // [R] DB §10.1, §5.5: outstanding write bursts, same packing
  localparam int unsigned TSR_CSR_DBG_WROUT_N          = 4;
  localparam int unsigned TSR_CSR_DBG_WROUT_STRIDE     = 4;
  localparam logic [11:0] TSR_CSR_DBG_U_STATE_BASE     = 12'h900;        // [R] DB §10.1, §5.5: unit u < 8
  localparam int unsigned TSR_CSR_DBG_U_STATE_N        = 8;
  localparam int unsigned TSR_CSR_DBG_U_STATE_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_DBG_U_OLDEST_BASE    = 12'h904;        // [R] DB §10.1, §5.5: ticket of the oldest outstanding entry; TICKET_NONE if none
  localparam int unsigned TSR_CSR_DBG_U_OLDEST_N       = 8;
  localparam int unsigned TSR_CSR_DBG_U_OLDEST_STRIDE  = 16;
  localparam logic [11:0] TSR_CSR_DBG_U_OPIDX_BASE     = 12'h908;        // [R] DB §10.1, §5.5: its push index
  localparam int unsigned TSR_CSR_DBG_U_OPIDX_N        = 8;
  localparam int unsigned TSR_CSR_DBG_U_OPIDX_STRIDE   = 16;
  localparam logic [11:0] TSR_CSR_DBG_U_AGE_BASE       = 12'h90C;        // [R] DB §10.1, §5.5: core cycles since it became the oldest (saturating)
  localparam int unsigned TSR_CSR_DBG_U_AGE_N          = 8;
  localparam int unsigned TSR_CSR_DBG_U_AGE_STRIDE     = 16;
  localparam logic [11:0] TSR_CSR_DBG_TRW_BASE         = 12'h980;        // [RW] DB §10.1, §8.3: ring base handle, 16-beat aligned (beat [3:0] read 0)
  localparam logic [11:0] TSR_CSR_DBG_TRW_LINES        = 12'h984;        // [RW] DB §10.1, §8.3: ring size in 64-B lines, power of two >= 16 (else ignored, PTR_ERR)
  localparam logic [11:0] TSR_CSR_DBG_TRW_WPTR         = 12'h988;        // [R] DB §10.1, §8.3: records written (mod 2^32)
  localparam logic [11:0] TSR_CSR_DBG_TRW_RPTR         = 12'h98C;        // [RW] DB §10.1, §8.3: records consumed; beyond WPTR ignored (PTR_ERR)
  localparam logic [11:0] TSR_CSR_DBG_TRW_DROPS        = 12'h990;        // [R] DB §10.1, §8.3: records dropped (saturating)
  localparam logic [11:0] TSR_CSR_DBG_TRW_MASK         = 12'h994;        // [RW] DB §10.1, §8.3: record classes written
  localparam logic [31:0] TSR_CSR_DBG_TRW_MASK_RESET   = 32'h0000_07FF;
  localparam logic [11:0] TSR_CSR_SREG_BASE            = 12'h9C0;        // [R] EX §5.1, §7.4, DB §10.1, VU §4.4: s0..s15 (f32 bits); present whenever the VU is built, also with HAS_DBG = 0
  localparam int unsigned TSR_CSR_SREG_N               = 16;
  localparam int unsigned TSR_CSR_SREG_STRIDE          = 4;
  localparam logic [11:0] TSR_CSR_DBG_HIST_HEAD        = 12'hA00;        // [R] DB §10.1, §5.6: next push index to be recorded
  localparam logic [11:0] TSR_CSR_DBG_HIST_IDX         = 12'hA04;        // [RW] DB §10.1, §5.6: push index to read
  localparam logic [11:0] TSR_CSR_DBG_HIST_INFO        = 12'hA08;        // [R] DB §10.1, §5.6
  localparam logic [11:0] TSR_CSR_DBG_HIST_PIDX        = 12'hA0C;        // [R] DB §10.1, §5.6
  localparam logic [11:0] TSR_CSR_DBG_HIST_TICKET      = 12'hA10;        // [R] DB §10.1, §5.6
  localparam logic [11:0] TSR_CSR_DBG_HIST_DEP0        = 12'hA14;        // [R] DB §10.1, §5.6
  localparam logic [11:0] TSR_CSR_DBG_HIST_DEP1        = 12'hA18;        // [R] DB §10.1, §5.6
  localparam logic [11:0] TSR_CSR_DBG_HIST_TIME        = 12'hA1C;        // [R] DB §10.1, §5.6: dispatch time MTIME[31:0]
  localparam logic [11:0] TSR_CSR_DBG_HIST_PC          = 12'hA20;        // [R] DB §10.1, §5.6: push PC
  localparam logic [11:0] TSR_CSR_DBG_HIST_W_BASE      = 12'hA40;        // [R] DB §10.1, §5.6: op words w0..w7
  localparam int unsigned TSR_CSR_DBG_HIST_W_N         = 8;
  localparam int unsigned TSR_CSR_DBG_HIST_W_STRIDE    = 4;

  // ---------------------------------------------------------------------------------------------
  // RISC-V control window: offsets from RV_CTL_BASE (CC §5, EX §4.5, DB §10.2)
  // ---------------------------------------------------------------------------------------------
  localparam logic [11:0] TSR_CW_ISSUE_W_BASE      = 12'h000;        // [RW] CC §5.1: staged instruction words; kept after a push
  localparam int unsigned TSR_CW_ISSUE_W_N         = 8;
  localparam int unsigned TSR_CW_ISSUE_W_STRIDE    = 4;
  localparam logic [31:0] TSR_CW_ISSUE_W_RESET     = 32'h0000_0000;
  localparam logic [11:0] TSR_CW_ISSUE_DEP0        = 12'h020;        // [RW] CC §5.1: dependency ticket; reset to TICKET_NONE by every push
  localparam logic [31:0] TSR_CW_ISSUE_DEP0_RESET  = 32'hF000_0000;
  localparam logic [11:0] TSR_CW_ISSUE_DEP1        = 12'h024;        // [RW] CC §5.1: same
  localparam logic [31:0] TSR_CW_ISSUE_DEP1_RESET  = 32'hF000_0000;
  localparam logic [11:0] TSR_CW_ISSUE_PUSH        = 12'h028;        // [W] CC §5.1: push; a push without a credit completes late (safety net)
  localparam logic [11:0] TSR_CW_ISSUE_TICKET      = 12'h02C;        // [R] CC §5.1: ticket of the last push or QCTRL.QCLEAR
  localparam logic [11:0] TSR_CW_ISSUE_CREDIT      = 12'h030;        // [R] CC §5.1: credit per unit
  localparam logic [11:0] TSR_CW_ISSUE_UNITS       = 12'h034;        // [R] CC §5.1: bit u = unit u is built
  localparam logic [11:0] TSR_CW_ISSUE_STATUS      = 12'h038;        // [R/W1C] CC §5.1: push timeout and error interrupt
  localparam logic [11:0] TSR_CW_ISSUE_PIDX        = 12'h03C;        // [R] CC §5.1: push index the next ticketed entry will receive (+1 per ISSUE_PUSH and QCLEAR); exact
  localparam logic [11:0] TSR_CW_DONE_SEQ_BASE     = 12'h040;        // [R] CC §5.1: [27:0] completions of unit u (synchronised copy; may lag, never leads)
  localparam int unsigned TSR_CW_DONE_SEQ_N        = 8;
  localparam int unsigned TSR_CW_DONE_SEQ_STRIDE   = 4;
  localparam logic [11:0] TSR_CW_ISSUED_SEQ_BASE   = 12'h080;        // [R] CC §5.1: [27:0] sequence the next push to unit u gets
  localparam int unsigned TSR_CW_ISSUED_SEQ_N      = 8;
  localparam int unsigned TSR_CW_ISSUED_SEQ_STRIDE = 4;
  localparam logic [11:0] TSR_CW_ERR_SEQ_BASE      = 12'h0C0;        // [R] CC §5.1: first error of unit u since the last clear: ERR_SEQ
  localparam int unsigned TSR_CW_ERR_SEQ_N         = 8;
  localparam int unsigned TSR_CW_ERR_SEQ_STRIDE    = 16;
  localparam logic [11:0] TSR_CW_ERR_CODE_BASE     = 12'h0C4;        // [R] CC §5.1: ERR_CODE; 0 = none
  localparam int unsigned TSR_CW_ERR_CODE_N        = 8;
  localparam int unsigned TSR_CW_ERR_CODE_STRIDE   = 16;
  localparam logic [11:0] TSR_CW_ERR_INFO_BASE     = 12'h0C8;        // [R] CC §5.1: ERR_INFO
  localparam int unsigned TSR_CW_ERR_INFO_N        = 8;
  localparam int unsigned TSR_CW_ERR_INFO_STRIDE   = 16;
  localparam logic [11:0] TSR_CW_ERR_W0_BASE       = 12'h0CC;        // [R] CC §5.1: ERR_W0
  localparam int unsigned TSR_CW_ERR_W0_N          = 8;
  localparam int unsigned TSR_CW_ERR_W0_STRIDE     = 16;
  localparam logic [11:0] TSR_CW_QCTRL             = 12'h140;        // [W] CC §5.1: queue control entries, in push order
  localparam logic [11:0] TSR_CW_QSTATUS           = 12'h144;        // [R] CC §5.1: queue status
  localparam logic [11:0] TSR_CW_WAIT_TICKET       = 12'h148;        // [RW] CC §5.1: irq_fast[0] pending while this ticket is done; unit 15 disarms
  localparam logic [31:0] TSR_CW_WAIT_TICKET_RESET = 32'hF000_0000;
  localparam logic [11:0] TSR_CW_WAIT_CREDIT       = 12'h14C;        // [RW] CC §5.1: irq_fast[5] pending while that unit's credit != 0
  localparam logic [31:0] TSR_CW_WAIT_CREDIT_RESET = 32'h0000_0000;
  localparam logic [11:0] TSR_CW_REC_COUNT         = 12'h160;        // [R] DB §10.2, §8.2: completion records waiting (0..64)
  localparam logic [11:0] TSR_CW_REC_DATA_BASE     = 12'h164;        // [R] DB §10.2, §8.2: head record words 0..3 (word table REC)
  localparam int unsigned TSR_CW_REC_DATA_N        = 4;
  localparam int unsigned TSR_CW_REC_DATA_STRIDE   = 4;
  localparam logic [11:0] TSR_CW_REC_POP           = 12'h174;        // [W] DB §10.2, §8.2: drop the head record
  localparam logic [11:0] TSR_CW_TRACE_MARK        = 12'h178;        // [W] DB §10.2, §8.4: marker through the issue FIFO
  localparam logic [11:0] TSR_CW_REC_CFG           = 12'h17C;        // [RW] DB §10.2, §8.2: completion-record configuration
  localparam logic [31:0] TSR_CW_REC_CFG_RESET     = 32'h0000_0000;
  localparam logic [11:0] TSR_CW_ERR_PIDX_BASE     = 12'h180;        // [R] CC §5.1: push index of unit u's first error
  localparam int unsigned TSR_CW_ERR_PIDX_N        = 8;
  localparam int unsigned TSR_CW_ERR_PIDX_STRIDE   = 4;
  // 0x1C0..0x1C0 reserved (CC §5.1)
  localparam logic [11:0] TSR_CW_SPM_LB_INV        = 12'h1C4;        // [W] CC §5.1, §10.5: invalidate the SPM line buffer
  localparam logic [11:0] TSR_CW_LQ_HEAD_W_BASE    = 12'h200;        // [R] EX §4.5: the descriptor the next LQ_POP takes (valid while LQ_AVAIL != 0)
  localparam int unsigned TSR_CW_LQ_HEAD_W_N       = 8;
  localparam int unsigned TSR_CW_LQ_HEAD_W_STRIDE  = 4;
  localparam logic [11:0] TSR_CW_LQ_AVAIL          = 12'h220;        // [R] EX §4.5: PUSHED - TAKEN as seen in clk_rv
  localparam logic [11:0] TSR_CW_LQ_HEAD_SEQ       = 12'h224;        // [R] EX §4.5: TAKEN = the sequence number of the head
  localparam logic [11:0] TSR_CW_LQ_POP            = 12'h228;        // [W] EX §4.5: take the head; access fault if LQ_AVAIL = 0 or ACTIVE
  localparam logic [11:0] TSR_CW_LQ_CANCEL         = 12'h22C;        // [R] EX §4.5: copy of the host's cancel watermark
  localparam logic [11:0] TSR_CW_LQ_STOP           = 12'h230;        // [R] EX §4.5: copy of the host's stop watermark
  localparam logic [11:0] TSR_CW_LQ_FLAGS          = 12'h234;        // [R] EX §4.5: watermark valid bits, ACTIVE, PLAT_ERR
  localparam logic [11:0] TSR_CW_LCQ_W_BASE        = 12'h240;        // [RW] EX §4.5: record staging; kept after a post
  localparam int unsigned TSR_CW_LCQ_W_N           = 8;
  localparam int unsigned TSR_CW_LCQ_W_STRIDE      = 4;
  localparam logic [11:0] TSR_CW_LCQ_POST          = 12'h260;        // [W] EX §4.5: post the staged record; access fault if LCQ_FREE = 0, or ANSWER while not ACTIVE
  localparam logic [11:0] TSR_CW_LCQ_FREE          = 12'h264;        // [R] EX §4.5: LCQ_DEPTH - (LCQ_WR - LCQ_RD) as seen in clk_rv (may under-report)
  localparam logic [11:0] TSR_CW_LOG_PUSH          = 12'h280;        // [W] EX §4.5: one log word; dropped and counted when the ring is full, never faults
  localparam logic [11:0] TSR_CW_LOG_FREE          = 12'h284;        // [R] EX §4.5: free words (may under-report)
  localparam logic [11:0] TSR_CW_LOG_CFG           = 12'h288;        // [R] EX §4.5: copy of the host's LOG_CFG
  localparam logic [11:0] TSR_CW_RT_STATE          = 12'h290;        // [RW] EX §4.5: runtime phase, mirrored to host 0x038; 0 after a RISC-V reset
  localparam logic [31:0] TSR_CW_RT_STATE_RESET    = 32'h0000_0000;
  localparam logic [11:0] TSR_CW_RT_INFO           = 12'h294;        // [RW] EX §4.5: runtime- or kernel-defined progress word, mirrored to host 0x03C
  localparam logic [11:0] TSR_CW_MTIME_LO          = 12'h300;        // [R] CC §5.2: core clock cycles since reset (gray-synchronised)
  localparam logic [11:0] TSR_CW_MTIME_HI          = 12'h304;        // [R] CC §5.2
  localparam logic [11:0] TSR_CW_MTIMECMP_LO       = 12'h308;        // [RW] CC §5.2: irq_timer while MTIME >= MTIMECMP
  localparam logic [31:0] TSR_CW_MTIMECMP_LO_RESET = 32'hFFFF_FFFF;
  localparam logic [11:0] TSR_CW_MTIMECMP_HI       = 12'h30C;        // [RW] CC §5.2
  localparam logic [31:0] TSR_CW_MTIMECMP_HI_RESET = 32'hFFFF_FFFF;
  localparam logic [11:0] TSR_CW_DBG_FLAGS         = 12'h400;        // [R] DB §10.2: DB state for the runtime
  // 0x404..0x7FC reserved (DB §10.2)

  // ---------------------------------------------------------------------------------------------
  // XRT adapter s_axi_control map (EX §9.2)
  // ---------------------------------------------------------------------------------------------
  // 0x000..0x03C reserved (EX §9.2)
  localparam logic [12:0] TSR_SAXI_CH_BASE_LO_BASE   = 13'h0040;  // [RW] EX §9.2: device byte address of the channel-c buffer, bits [31:0]; 4 KiB aligned; written while MEM_EN = 0
  localparam int unsigned TSR_SAXI_CH_BASE_LO_N      = 16;
  localparam int unsigned TSR_SAXI_CH_BASE_LO_STRIDE = 8;
  localparam logic [12:0] TSR_SAXI_CH_BASE_HI_BASE   = 13'h0044;  // [RW] EX §9.2: bits [63:32]
  localparam int unsigned TSR_SAXI_CH_BASE_HI_N      = 16;
  localparam int unsigned TSR_SAXI_CH_BASE_HI_STRIDE = 8;
  localparam logic [12:0] TSR_SAXI_ADAPTER_CTRL      = 13'h00E0;  // [RW] EX §9.2: MEM_EN
  localparam logic [12:0] TSR_SAXI_ADAPTER_STATUS    = 13'h00F0;  // [R/W1C] EX §9.2: sticky adapter errors
  localparam logic [12:0] TSR_SAXI_ADAPTER_VERSION   = 13'h00F4;  // [R] EX §9.2: {16'h0001, 8'h00, NUM_CH}
  localparam logic [12:0] TSR_SAXI_CORE_BASE         = 13'h1000;  // [-] EX §9.2: core CSR window: 0x1000 + offset, any time; writes need WSTRB = 0xF
  localparam int unsigned TSR_SAXI_CORE_BYTES        = 4096;

  // ---------------------------------------------------------------------------------------------
  // RISC-V address map (CC §4.1); region sizes are the U200 values
  // ---------------------------------------------------------------------------------------------
  localparam logic [31:0] TSR_RV_ITCM_BASE    = 32'h0000_0000;  // CC §4.1: code; data reads steal the fetch port (byte/half/word, fetch)
  localparam logic [31:0] TSR_RV_ITCM_SIZE    = 32'h0002_0000;
  localparam logic [31:0] TSR_RV_ITCM_LIMIT   = 32'h0010_0000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_DTCM_BASE    = 32'h0010_0000;  // CC §4.1: .rodata, .data, .bss, stack (byte/half/word)
  localparam logic [31:0] TSR_RV_DTCM_SIZE    = 32'h0001_0000;
  localparam logic [31:0] TSR_RV_DTCM_LIMIT   = 32'h0020_0000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_DM_BASE      = 32'h1000_0000;  // CC §4.1: debug module memory (riscv-dbg layout) (word, fetch)
  localparam logic [31:0] TSR_RV_DM_SIZE      = 32'h0000_1000;
  localparam logic [31:0] TSR_RV_DM_LIMIT     = 32'h1000_1000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_CTL_BASE     = 32'h2000_0000;  // CC §4.1: control window (CTL_WINDOW) (word)
  localparam logic [31:0] TSR_RV_CTL_SIZE     = 32'h0000_1000;
  localparam logic [31:0] TSR_RV_CTL_LIMIT    = 32'h2000_1000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_CORE_BASE    = 32'h2000_1000;  // CC §4.1, §5.3: core window: read-only view of the host CSR map (same offsets) (word loads)
  localparam logic [31:0] TSR_RV_CORE_SIZE    = 32'h0000_1000;
  localparam logic [31:0] TSR_RV_CORE_LIMIT   = 32'h2000_2000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_SPM_BASE     = 32'h4000_0000;  // CC §4.1: SPM window, exact: 0x4000_0000 + 64 x line + offset (byte/half/word)
  localparam logic [31:0] TSR_RV_SPM_SIZE     = 32'h0040_0000;
  localparam logic [31:0] TSR_RV_SPM_LIMIT    = 32'h4040_0000;  // end of the reserved range (exclusive)
  localparam logic [31:0] TSR_RV_SPM_LB_BASE  = 32'h5000_0000;  // CC §4.1, §10.5: SPM line-buffered alias (byte/half/word)
  localparam logic [31:0] TSR_RV_SPM_LB_SIZE  = 32'h0040_0000;
  localparam logic [31:0] TSR_RV_SPM_LB_LIMIT = 32'h5040_0000;  // end of the reserved range (exclusive)

  // ---------------------------------------------------------------------------------------------
  // Ibex fast interrupts irq_fast[i] (CC §12.3)
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned TSR_IRQ_FAST_WAIT_TICKET = 0;  // CC §12.3: WAIT_TICKET done
  localparam int unsigned TSR_IRQ_FAST_ERR         = 1;  // CC §12.3: ISSUE_STATUS.ERR_IRQ
  localparam int unsigned TSR_IRQ_FAST_LQ          = 2;  // CC §12.3, EX §4.5: LQ_AVAIL != 0 and not ACTIVE
  localparam int unsigned TSR_IRQ_FAST_LCQ         = 3;  // CC §12.3, EX §4.5: LCQ_FREE != 0
  localparam int unsigned TSR_IRQ_FAST_REC         = 4;  // CC §12.3, DB §8.2: REC_COUNT >= 32 with REC_CFG.IRQ_HALF
  localparam int unsigned TSR_IRQ_FAST_CREDIT      = 5;  // CC §12.3: WAIT_CREDIT
  localparam int unsigned TSR_IRQ_FAST_DB_RSVD     = 6;  // CC §12.3, DB §10.2: reserved for DB

  // ---------------------------------------------------------------------------------------------
  // DMI registers (DB §9.1); host CSR offset = CSR_DMI(address)
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned TSR_DMI_DATA0      = 4;   // DB §9.1
  localparam int unsigned TSR_DMI_DATA1      = 5;   // DB §9.1
  localparam int unsigned TSR_DMI_DMCONTROL  = 16;  // DB §9.1: haltreq, resumereq, ackhavereset, ndmreset, dmactive
  localparam int unsigned TSR_DMI_DMSTATUS   = 17;  // DB §9.1
  localparam int unsigned TSR_DMI_HARTINFO   = 18;  // DB §9.1
  localparam int unsigned TSR_DMI_ABSTRACTCS = 22;  // DB §9.1: progbufsize, busy, cmderr, datacount
  localparam int unsigned TSR_DMI_COMMAND    = 23;  // DB §9.1: Access Register, postexec-only
  localparam int unsigned TSR_DMI_PROGBUF0   = 32;  // DB §9.1: progbuf0..7
  localparam int unsigned TSR_DMI_PROGBUF0_N = 8;
  localparam int unsigned TSR_DMI_SBCS       = 56;  // DB §9.1
  localparam int unsigned TSR_DMI_SBADDRESS0 = 57;  // DB §9.1
  localparam int unsigned TSR_DMI_SBDATA0    = 60;  // DB §9.1

  // ---------------------------------------------------------------------------------------------
  // DRAM map per channel (EX §11); byte offsets
  // ---------------------------------------------------------------------------------------------
  localparam longint unsigned TSR_REGION_OPL_OFF    = 64'h0;         // EX §11: ch0: op lists, replay push streams, argument blocks, code overlays (was CMD)
  localparam longint unsigned TSR_REGION_OPL_SIZE   = 64'h4000000;
  localparam longint unsigned TSR_REGION_IO_OFF     = 64'h4000000;   // EX §11: ch0: host <-> device tensors; compute chunks; the TRW ring while profiling
  localparam longint unsigned TSR_REGION_IO_SIZE    = 64'h10000000;
  localparam longint unsigned TSR_REGION_AUX_OFF    = 64'h14000000;  // EX §11: ch0: small f32 tensors (norm weights, RoPE tables)
  localparam longint unsigned TSR_REGION_AUX_SIZE   = 64'hC000000;
  localparam longint unsigned TSR_REGION_WEIGHT_OFF = 64'h20000000;  // EX §11: every channel: weights bottom-up, KV planes top-down, compute heap between

  // ---------------------------------------------------------------------------------------------
  // Bit-field layouts (in-place masks; _LSB, _W; values unshifted)
  // ---------------------------------------------------------------------------------------------
  // W0 (EX §7.1): instruction word w0 = {x16, flags, sub, opcode}
  localparam logic [31:0] TSR_W0_OPCODE     = 32'h0000_00FF;  // [7:0] opcode
  localparam int unsigned TSR_W0_OPCODE_LSB = 0;
  localparam int unsigned TSR_W0_OPCODE_W   = 8;
  localparam logic [31:0] TSR_W0_SUB        = 32'h0000_0F00;  // [11:8] op-specific sub code
  localparam int unsigned TSR_W0_SUB_LSB    = 8;
  localparam int unsigned TSR_W0_SUB_W      = 4;
  localparam logic [31:0] TSR_W0_FLAGS      = 32'h0000_F000;  // [15:12] op-specific flags
  localparam int unsigned TSR_W0_FLAGS_LSB  = 12;
  localparam int unsigned TSR_W0_FLAGS_W    = 4;
  localparam logic [31:0] TSR_W0_X16        = 32'hFFFF_0000;  // [31:16] op-specific field; must be 0 (ILLEGAL_OP) where the op does not assign it
  localparam int unsigned TSR_W0_X16_LSB    = 16;
  localparam int unsigned TSR_W0_X16_W      = 16;
  // VERSION (EX §5.1, §7.5): Tesseract Core capability MAJOR.MINOR; [31:16] 0
  localparam logic [31:0] TSR_VERSION_MINOR     = 32'h0000_00FF;  // [7:0] capability minor (0)
  localparam int unsigned TSR_VERSION_MINOR_LSB = 0;
  localparam int unsigned TSR_VERSION_MINOR_W   = 8;
  localparam logic [31:0] TSR_VERSION_MAJOR     = 32'h0000_FF00;  // [15:8] capability major (1)
  localparam int unsigned TSR_VERSION_MAJOR_LSB = 8;
  localparam int unsigned TSR_VERSION_MAJOR_W   = 8;
  // CFG0 (EX §5.1): geometry configuration
  localparam logic [31:0] TSR_CFG0_NUM_CH     = 32'h0000_00FF;  // [7:0] channels C
  localparam int unsigned TSR_CFG0_NUM_CH_LSB = 0;
  localparam int unsigned TSR_CFG0_NUM_CH_W   = 8;
  localparam logic [31:0] TSR_CFG0_LANES      = 32'h000F_FF00;  // [19:8] MAC lanes
  localparam int unsigned TSR_CFG0_LANES_LSB  = 8;
  localparam int unsigned TSR_CFG0_LANES_W    = 12;
  localparam logic [31:0] TSR_CFG0_T_MAX      = 32'h00F0_0000;  // [23:20] token slots
  localparam int unsigned TSR_CFG0_T_MAX_LSB  = 20;
  localparam int unsigned TSR_CFG0_T_MAX_W    = 4;
  // CFG2 (EX §5.1): geometry configuration
  localparam logic [31:0] TSR_CFG2_VL            = 32'h0000_00FF;  // [7:0] vector length
  localparam int unsigned TSR_CFG2_VL_LSB        = 0;
  localparam int unsigned TSR_CFG2_VL_W          = 8;
  localparam logic [31:0] TSR_CFG2_NSREG         = 32'h0000_FF00;  // [15:8] s registers
  localparam int unsigned TSR_CFG2_NSREG_LSB     = 8;
  localparam int unsigned TSR_CFG2_NSREG_W       = 8;
  localparam logic [31:0] TSR_CFG2_K_MAX_256     = 32'hFFFF_0000;  // [31:16] K_MAX / 256
  localparam int unsigned TSR_CFG2_K_MAX_256_LSB = 16;
  localparam int unsigned TSR_CFG2_K_MAX_256_W   = 16;
  // CAPS (EX §7.5): capability bits; retired bits are never set and never reused
  localparam logic [31:0] TSR_CAPS_VU                    = 32'h0000_0001;  // [0] VOP, RED, ROPE and SOP present (VU §5.10)
  localparam int unsigned TSR_CAPS_VU_LSB                = 0;
  localparam int unsigned TSR_CAPS_VU_W                  = 1;
  localparam logic [31:0] TSR_CAPS_TOPK                  = 32'h0000_0008;  // [3] TOPK present (RTL from M8; the simulator from M6)
  localparam int unsigned TSR_CAPS_TOPK_LSB              = 3;
  localparam int unsigned TSR_CAPS_TOPK_W                = 1;
  localparam logic [31:0] TSR_CAPS_ATT                   = 32'h0000_0020;  // [5] the seven AT ops present (KVST, ATTQK, ATTPV, ASMAX, AMASK, AMERGE, ANORM)
  localparam int unsigned TSR_CAPS_ATT_LSB               = 5;
  localparam int unsigned TSR_CAPS_ATT_W                 = 1;
  localparam logic [31:0] TSR_CAPS_DQROW                 = 32'h0000_0040;  // [6] DQROW in the SPM form (RTL from M8; the simulator from M6)
  localparam int unsigned TSR_CAPS_DQROW_LSB             = 6;
  localparam int unsigned TSR_CAPS_DQROW_W               = 1;
  localparam logic [31:0] TSR_CAPS_VMASK                 = 32'h0000_0080;  // [7] VMASK (M9)
  localparam int unsigned TSR_CAPS_VMASK_LSB             = 7;
  localparam int unsigned TSR_CAPS_VMASK_W               = 1;
  localparam logic [31:0] TSR_CAPS_GDN                   = 32'h0000_0100;  // [8] GDN present (the fused Gated DeltaNet state update, VU §7.12)
  localparam int unsigned TSR_CAPS_GDN_LSB               = 8;
  localparam int unsigned TSR_CAPS_GDN_W                 = 1;
  localparam logic [31:0] TSR_CAPS_VU_RSVD               = 32'h0000_0E00;  // [11:9] reserved for VU extensions, 0
  localparam int unsigned TSR_CAPS_VU_RSVD_LSB           = 9;
  localparam int unsigned TSR_CAPS_VU_RSVD_W             = 3;
  localparam logic [31:0] TSR_CAPS_ATT_LOCAL             = 32'h0000_1000;  // [12] four-owner local FAST attention score and probability slots
  localparam int unsigned TSR_CAPS_ATT_LOCAL_LSB         = 12;
  localparam int unsigned TSR_CAPS_ATT_LOCAL_W           = 1;
  localparam logic [31:0] TSR_CAPS_MVM_LOCAL_SCALE       = 32'h0000_2000;  // [13] scale MVM output through the owner-local VU before export
  localparam int unsigned TSR_CAPS_MVM_LOCAL_SCALE_LSB   = 13;
  localparam int unsigned TSR_CAPS_MVM_LOCAL_SCALE_W     = 1;
  localparam logic [31:0] TSR_CAPS_MVM_LOCAL_SOFTCAP     = 32'h0000_4000;  // [14] owner-local FP32 scale, TANH, scale after MVM
  localparam int unsigned TSR_CAPS_MVM_LOCAL_SOFTCAP_LSB = 14;
  localparam int unsigned TSR_CAPS_MVM_LOCAL_SOFTCAP_W   = 1;
  localparam logic [31:0] TSR_CAPS_AT_RSVD               = 32'h0000_8000;  // [15] reserved for AT, 0
  localparam int unsigned TSR_CAPS_AT_RSVD_LSB           = 15;
  localparam int unsigned TSR_CAPS_AT_RSVD_W             = 1;
  // bit 2 retired (RLD / MULI / LOOP), never set, never reused
  // bit 4 retired (WAITM / JUMP), never set, never reused
  // CFG3 (EX §7.5, VU §5.10, AT §5.10): [15:0] VU, [31:16] AT
  localparam logic [31:0] TSR_CFG3_VU_FRONT     = 32'h0000_001F;  // [4:0] VU front lanes with an EXP core (information for cycle models)
  localparam int unsigned TSR_CFG3_VU_FRONT_LSB = 0;
  localparam int unsigned TSR_CFG3_VU_FRONT_W   = 5;
  localparam logic [31:0] TSR_CFG3_VU_BACK      = 32'h0000_03E0;  // [9:5] VU back lanes (information for cycle models)
  localparam int unsigned TSR_CFG3_VU_BACK_LSB  = 5;
  localparam int unsigned TSR_CFG3_VU_BACK_W    = 5;
  localparam logic [31:0] TSR_CFG3_PCAP16       = 32'hFFFF_0000;  // [31:16] PCAP / 16 (U200: 584)
  localparam int unsigned TSR_CFG3_PCAP16_LSB   = 16;
  localparam int unsigned TSR_CFG3_PCAP16_W     = 16;
  // CTL_ID (EX §5.1): control-core revision and features
  localparam logic [31:0] TSR_CTL_ID_VERSION     = 32'h0000_00FF;  // [7:0] control-core revision (1); 0: no control core
  localparam int unsigned TSR_CTL_ID_VERSION_LSB = 0;
  localparam int unsigned TSR_CTL_ID_VERSION_W   = 8;
  localparam logic [31:0] TSR_CTL_ID_DM          = 32'h0000_0100;  // [8] debug module
  localparam int unsigned TSR_CTL_ID_DM_LSB      = 8;
  localparam int unsigned TSR_CTL_ID_DM_W        = 1;
  localparam logic [31:0] TSR_CTL_ID_REC         = 32'h0000_0200;  // [9] completion records
  localparam int unsigned TSR_CTL_ID_REC_LSB     = 9;
  localparam int unsigned TSR_CTL_ID_REC_W       = 1;
  localparam logic [31:0] TSR_CTL_ID_BRIDGE      = 32'h0000_0400;  // [10] SPM bridge
  localparam int unsigned TSR_CTL_ID_BRIDGE_LSB  = 10;
  localparam int unsigned TSR_CTL_ID_BRIDGE_W    = 1;
  localparam logic [31:0] TSR_CTL_ID_LOG         = 32'h0000_0800;  // [11] log ring
  localparam int unsigned TSR_CTL_ID_LOG_LSB     = 11;
  localparam int unsigned TSR_CTL_ID_LOG_W       = 1;
  localparam logic [31:0] TSR_CTL_ID_WDOG        = 32'h0000_1000;  // [12] watchdog
  localparam int unsigned TSR_CTL_ID_WDOG_LSB    = 12;
  localparam int unsigned TSR_CTL_ID_WDOG_W      = 1;
  localparam logic [31:0] TSR_CTL_ID_SBA         = 32'h0000_2000;  // [13] SBA connected
  localparam int unsigned TSR_CTL_ID_SBA_LSB     = 13;
  localparam int unsigned TSR_CTL_ID_SBA_W       = 1;
  localparam logic [31:0] TSR_CTL_ID_SPM_LB      = 32'h0000_4000;  // [14] SPM line-buffered alias
  localparam int unsigned TSR_CTL_ID_SPM_LB_LSB  = 14;
  localparam int unsigned TSR_CTL_ID_SPM_LB_W    = 1;
  // CTRL (EX §5.2): pulse bits read 0; level bits act before the pulses of the same write
  localparam logic [31:0] TSR_CTRL_RV_RUN           = 32'h0000_0001;  // [0] RW level: 1 releases the RISC-V; 1 -> 0 starts the RISC-V reset sequence; refused while SR_BUSY
  localparam int unsigned TSR_CTRL_RV_RUN_LSB       = 0;
  localparam int unsigned TSR_CTRL_RV_RUN_W         = 1;
  localparam logic [31:0] TSR_CTRL_HOST_HOLD        = 32'h0000_0002;  // [1] RW level: the host's hold source for the queue heads
  localparam int unsigned TSR_CTRL_HOST_HOLD_LSB    = 1;
  localparam int unsigned TSR_CTRL_HOST_HOLD_W      = 1;
  localparam logic [31:0] TSR_CTRL_HOLD_ON_HALT     = 32'h0000_0004;  // [2] RW level: a debug halt holds the queue heads
  localparam int unsigned TSR_CTRL_HOLD_ON_HALT_LSB = 2;
  localparam int unsigned TSR_CTRL_HOLD_ON_HALT_W   = 1;
  localparam logic [31:0] TSR_CTRL_IRQ_EN           = 32'h0000_0008;  // [3] RW level: host_irq_o = IRQ_EN and LCQ_WR != LCQ_RD
  localparam int unsigned TSR_CTRL_IRQ_EN_LSB       = 3;
  localparam int unsigned TSR_CTRL_IRQ_EN_W         = 1;
  localparam logic [31:0] TSR_CTRL_QABORT           = 32'h0000_0100;  // [8] W1 pulse: host QABORT, at once
  localparam int unsigned TSR_CTRL_QABORT_LSB       = 8;
  localparam int unsigned TSR_CTRL_QABORT_W         = 1;
  localparam logic [31:0] TSR_CTRL_QCLEAR           = 32'h0000_0200;  // [9] W1 pulse: host QCLEAR; only with QIDLE and the RISC-V quiet, else QCLEAR_REFUSED
  localparam int unsigned TSR_CTRL_QCLEAR_LSB       = 9;
  localparam int unsigned TSR_CTRL_QCLEAR_W         = 1;
  localparam logic [31:0] TSR_CTRL_SOFT_RESET       = 32'h0000_0400;  // [10] W1 pulse: device-level reset sequence (EX §8.5)
  localparam int unsigned TSR_CTRL_SOFT_RESET_LSB   = 10;
  localparam int unsigned TSR_CTRL_SOFT_RESET_W     = 1;
  // STATUS (EX §5.3): bits 0..15 levels, 16..27 sticky (W1C)
  localparam logic [31:0] TSR_STATUS_RV_RUNNING         = 32'h0000_0001;  // [0] RV_RUN = 1 and the hart is out of reset
  localparam int unsigned TSR_STATUS_RV_RUNNING_LSB     = 0;
  localparam int unsigned TSR_STATUS_RV_RUNNING_W       = 1;
  localparam logic [31:0] TSR_STATUS_RV_HALTED          = 32'h0000_0002;  // [1] debug halt
  localparam int unsigned TSR_STATUS_RV_HALTED_LSB      = 1;
  localparam int unsigned TSR_STATUS_RV_HALTED_W        = 1;
  localparam logic [31:0] TSR_STATUS_RV_SLEEP           = 32'h0000_0004;  // [2] WFI
  localparam int unsigned TSR_STATUS_RV_SLEEP_LSB       = 2;
  localparam int unsigned TSR_STATUS_RV_SLEEP_W         = 1;
  localparam logic [31:0] TSR_STATUS_RV_CLK_OK          = 32'h0000_0008;  // [3] clk_rv runs and its domain is out of reset
  localparam int unsigned TSR_STATUS_RV_CLK_OK_LSB      = 3;
  localparam int unsigned TSR_STATUS_RV_CLK_OK_W        = 1;
  localparam logic [31:0] TSR_STATUS_RV_RESET_DONE      = 32'h0000_0010;  // [4] the RISC-V reset sequence has completed
  localparam int unsigned TSR_STATUS_RV_RESET_DONE_LSB  = 4;
  localparam int unsigned TSR_STATUS_RV_RESET_DONE_W    = 1;
  localparam logic [31:0] TSR_STATUS_ACTIVE             = 32'h0000_0020;  // [5] TAKEN != ANSWERED
  localparam int unsigned TSR_STATUS_ACTIVE_LSB         = 5;
  localparam int unsigned TSR_STATUS_ACTIVE_W           = 1;
  localparam logic [31:0] TSR_STATUS_QIDLE              = 32'h0000_0040;  // [6] no entry queued or running, issue FIFO empty
  localparam int unsigned TSR_STATUS_QIDLE_LSB          = 6;
  localparam int unsigned TSR_STATUS_QIDLE_W            = 1;
  localparam logic [31:0] TSR_STATUS_QQUIESCENT         = 32'h0000_0080;  // [7] quiescent (CC §9.6)
  localparam int unsigned TSR_STATUS_QQUIESCENT_LSB     = 7;
  localparam int unsigned TSR_STATUS_QQUIESCENT_W       = 1;
  localparam logic [31:0] TSR_STATUS_QERR               = 32'h0000_0100;  // [8] some unit is in error state
  localparam int unsigned TSR_STATUS_QERR_LSB           = 8;
  localparam int unsigned TSR_STATUS_QERR_W             = 1;
  localparam logic [31:0] TSR_STATUS_SR_BUSY            = 32'h0000_0200;  // [9] SOFT_RESET in progress
  localparam int unsigned TSR_STATUS_SR_BUSY_LSB        = 9;
  localparam int unsigned TSR_STATUS_SR_BUSY_W          = 1;
  localparam logic [31:0] TSR_STATUS_LCQ_PENDING        = 32'h0000_0400;  // [10] LCQ_WR != LCQ_RD
  localparam int unsigned TSR_STATUS_LCQ_PENDING_LSB    = 10;
  localparam int unsigned TSR_STATUS_LCQ_PENDING_W      = 1;
  localparam logic [31:0] TSR_STATUS_LOG_PENDING        = 32'h0000_0800;  // [11] LOG_WR != LOG_RD
  localparam int unsigned TSR_STATUS_LOG_PENDING_LSB    = 11;
  localparam int unsigned TSR_STATUS_LOG_PENDING_W      = 1;
  localparam logic [31:0] TSR_STATUS_PLAT_ERR           = 32'h0000_1000;  // [12] the platform's error input plat_err_i
  localparam int unsigned TSR_STATUS_PLAT_ERR_LSB       = 12;
  localparam int unsigned TSR_STATUS_PLAT_ERR_W         = 1;
  localparam logic [31:0] TSR_STATUS_HOLD_SRC           = 32'h0000_E000;  // [15:13] hold sources (layout HOLD_SRC: host, RISC-V, debug halt)
  localparam int unsigned TSR_STATUS_HOLD_SRC_LSB       = 13;
  localparam int unsigned TSR_STATUS_HOLD_SRC_W         = 3;
  localparam logic [31:0] TSR_STATUS_LOAD_REFUSED       = 32'h0001_0000;  // [16] sticky: LOAD_DATA write or VERIFY while RV_RUN = 1 or SR_BUSY
  localparam int unsigned TSR_STATUS_LOAD_REFUSED_LSB   = 16;
  localparam int unsigned TSR_STATUS_LOAD_REFUSED_W     = 1;
  localparam logic [31:0] TSR_STATUS_LOAD_TIMEOUT       = 32'h0002_0000;  // [17] sticky: a load or VERIFY word timed out
  localparam int unsigned TSR_STATUS_LOAD_TIMEOUT_LSB   = 17;
  localparam int unsigned TSR_STATUS_LOAD_TIMEOUT_W     = 1;
  localparam logic [31:0] TSR_STATUS_DMI_TIMEOUT        = 32'h0004_0000;  // [18] sticky
  localparam int unsigned TSR_STATUS_DMI_TIMEOUT_LSB    = 18;
  localparam int unsigned TSR_STATUS_DMI_TIMEOUT_W      = 1;
  localparam logic [31:0] TSR_STATUS_QCLEAR_REFUSED     = 32'h0008_0000;  // [19] sticky
  localparam int unsigned TSR_STATUS_QCLEAR_REFUSED_LSB = 19;
  localparam int unsigned TSR_STATUS_QCLEAR_REFUSED_W   = 1;
  localparam logic [31:0] TSR_STATUS_PUSH_TIMEOUT       = 32'h0010_0000;  // [20] sticky: the RISC-V's issue safety net fired
  localparam int unsigned TSR_STATUS_PUSH_TIMEOUT_LSB   = 20;
  localparam int unsigned TSR_STATUS_PUSH_TIMEOUT_W     = 1;
  localparam logic [31:0] TSR_STATUS_BRIDGE_TIMEOUT     = 32'h0020_0000;  // [21] sticky
  localparam int unsigned TSR_STATUS_BRIDGE_TIMEOUT_LSB = 21;
  localparam int unsigned TSR_STATUS_BRIDGE_TIMEOUT_W   = 1;
  localparam logic [31:0] TSR_STATUS_LQ_OVERFLOW        = 32'h0040_0000;  // [22] sticky: a push while the LQ was full (descriptor dropped)
  localparam int unsigned TSR_STATUS_LQ_OVERFLOW_LSB    = 22;
  localparam int unsigned TSR_STATUS_LQ_OVERFLOW_W      = 1;
  localparam logic [31:0] TSR_STATUS_RING_RD_BAD        = 32'h0080_0000;  // [23] sticky: an LCQ_RD or LOG_RD write was refused
  localparam int unsigned TSR_STATUS_RING_RD_BAD_LSB    = 23;
  localparam int unsigned TSR_STATUS_RING_RD_BAD_W      = 1;
  localparam logic [31:0] TSR_STATUS_LOG_DROP           = 32'h0100_0000;  // [24] sticky: log words were dropped
  localparam int unsigned TSR_STATUS_LOG_DROP_LSB       = 24;
  localparam int unsigned TSR_STATUS_LOG_DROP_W         = 1;
  localparam logic [31:0] TSR_STATUS_UNIT_HUNG          = 32'h0200_0000;  // [25] sticky: a unit watchdog fired
  localparam int unsigned TSR_STATUS_UNIT_HUNG_LSB      = 25;
  localparam int unsigned TSR_STATUS_UNIT_HUNG_W        = 1;
  localparam logic [31:0] TSR_STATUS_SR_RV_TIMEOUT      = 32'h0400_0000;  // [26] sticky: SOFT_RESET went on without the RISC-V reset completing
  localparam int unsigned TSR_STATUS_SR_RV_TIMEOUT_LSB  = 26;
  localparam int unsigned TSR_STATUS_SR_RV_TIMEOUT_W    = 1;
  localparam logic [31:0] TSR_STATUS_RV_RUN_REFUSED     = 32'h0800_0000;  // [27] sticky
  localparam int unsigned TSR_STATUS_RV_RUN_REFUSED_LSB = 27;
  localparam int unsigned TSR_STATUS_RV_RUN_REFUSED_W   = 1;
  // HOLD_SRC (EX §5.3, CC §5.1, DB §5.5): bit order of a hold-source field (STATUS[15:13], QSTATUS[4:2], DBG_U_STATE[11:8])
  localparam logic [3:0] TSR_HOLD_SRC_HOST      = 4'h1;  // [0] host (CTRL.HOST_HOLD)
  localparam int unsigned TSR_HOLD_SRC_HOST_LSB = 0;
  localparam int unsigned TSR_HOLD_SRC_HOST_W   = 1;
  localparam logic [3:0] TSR_HOLD_SRC_RV        = 4'h2;  // [1] RISC-V (QCTRL QHOLD)
  localparam int unsigned TSR_HOLD_SRC_RV_LSB   = 1;
  localparam int unsigned TSR_HOLD_SRC_RV_W     = 1;
  localparam logic [3:0] TSR_HOLD_SRC_HALT      = 4'h4;  // [2] debug halt with HOLD_ON_HALT
  localparam int unsigned TSR_HOLD_SRC_HALT_LSB = 2;
  localparam int unsigned TSR_HOLD_SRC_HALT_W   = 1;
  localparam logic [3:0] TSR_HOLD_SRC_DB        = 4'h8;  // [3] DB's dbg_hold (DBG_U_STATE only)
  localparam int unsigned TSR_HOLD_SRC_DB_LSB   = 3;
  localparam int unsigned TSR_HOLD_SRC_DB_W     = 1;
  // RV_BOOT (EX §5.1): boot vector, 256-byte aligned
  localparam logic [31:0] TSR_RV_BOOT_ADDR     = 32'hFFFF_FF00;  // [31:8] boot vector bits [31:8]; [7:0] read 0
  localparam int unsigned TSR_RV_BOOT_ADDR_LSB = 8;
  localparam int unsigned TSR_RV_BOOT_ADDR_W   = 24;
  // RV_CFG (EX §5.1): TCM sizes
  localparam logic [31:0] TSR_RV_CFG_ITCM_KIB     = 32'h0000_FFFF;  // [15:0] ITCM KiB
  localparam int unsigned TSR_RV_CFG_ITCM_KIB_LSB = 0;
  localparam int unsigned TSR_RV_CFG_ITCM_KIB_W   = 16;
  localparam logic [31:0] TSR_RV_CFG_DTCM_KIB     = 32'hFFFF_0000;  // [31:16] DTCM KiB
  localparam int unsigned TSR_RV_CFG_DTCM_KIB_LSB = 16;
  localparam int unsigned TSR_RV_CFG_DTCM_KIB_W   = 16;
  // WDOG_CFG (EX §5.1, §8.6): unit progress watchdog
  localparam logic [31:0] TSR_WDOG_CFG_L     = 32'h0000_001F;  // [4:0] limit 2^L core cycles without progress; 0 = off
  localparam int unsigned TSR_WDOG_CFG_L_LSB = 0;
  localparam int unsigned TSR_WDOG_CFG_L_W   = 5;
  // RT_STATE (EX §4.5): runtime phase (RISC-V 0x290, mirrored to host 0x038)
  localparam logic [31:0] TSR_RT_STATE_PHASE         = 32'h0000_00FF;  // [7:0] runtime phase
  localparam int unsigned TSR_RT_STATE_PHASE_LSB     = 0;
  localparam int unsigned TSR_RT_STATE_PHASE_W       = 8;
  localparam logic [31:0] TSR_RT_STATE_KERNEL        = 32'hFFFF_0000;  // [31:16] kernel index
  localparam int unsigned TSR_RT_STATE_KERNEL_LSB    = 16;
  localparam int unsigned TSR_RT_STATE_KERNEL_W      = 16;
  localparam logic [7:0] TSR_RT_STATE_PHASE_RESET    = 8'd0;
  localparam logic [7:0] TSR_RT_STATE_PHASE_BOOT     = 8'd1;
  localparam logic [7:0] TSR_RT_STATE_PHASE_IDLE     = 8'd2;
  localparam logic [7:0] TSR_RT_STATE_PHASE_DISPATCH = 8'd3;
  localparam logic [7:0] TSR_RT_STATE_PHASE_KERNEL   = 8'd4;
  localparam logic [7:0] TSR_RT_STATE_PHASE_JOIN     = 8'd5;
  localparam logic [7:0] TSR_RT_STATE_PHASE_ANSWER   = 8'd6;
  localparam logic [7:0] TSR_RT_STATE_PHASE_TRAPPED  = 8'd7;
  // LQ_CFG (EX §5.1): ring parameters and watermark valid bits
  localparam logic [31:0] TSR_LQ_CFG_LQ_DEPTH         = 32'h0000_00FF;  // [7:0]
  localparam int unsigned TSR_LQ_CFG_LQ_DEPTH_LSB     = 0;
  localparam int unsigned TSR_LQ_CFG_LQ_DEPTH_W       = 8;
  localparam logic [31:0] TSR_LQ_CFG_LCQ_DEPTH        = 32'h0000_FF00;  // [15:8]
  localparam int unsigned TSR_LQ_CFG_LCQ_DEPTH_LSB    = 8;
  localparam int unsigned TSR_LQ_CFG_LCQ_DEPTH_W      = 8;
  localparam logic [31:0] TSR_LQ_CFG_LOG_DEPTH        = 32'h0FFF_0000;  // [27:16]
  localparam int unsigned TSR_LQ_CFG_LOG_DEPTH_LSB    = 16;
  localparam int unsigned TSR_LQ_CFG_LOG_DEPTH_W      = 12;
  localparam logic [31:0] TSR_LQ_CFG_STOP_VALID       = 32'h4000_0000;  // [30]
  localparam int unsigned TSR_LQ_CFG_STOP_VALID_LSB   = 30;
  localparam int unsigned TSR_LQ_CFG_STOP_VALID_W     = 1;
  localparam logic [31:0] TSR_LQ_CFG_CANCEL_VALID     = 32'h8000_0000;  // [31]
  localparam int unsigned TSR_LQ_CFG_CANCEL_VALID_LSB = 31;
  localparam int unsigned TSR_LQ_CFG_CANCEL_VALID_W   = 1;
  // PERF_INFO (EX §5.1): PERF layout
  localparam logic [31:0] TSR_PERF_INFO_COUNT     = 32'h0000_00FF;  // [7:0] counters implemented
  localparam int unsigned TSR_PERF_INFO_COUNT_LSB = 0;
  localparam int unsigned TSR_PERF_INFO_COUNT_W   = 8;
  localparam logic [31:0] TSR_PERF_INFO_X         = 32'h0000_FF00;  // [15:8] first extension index X = 5 + 2 NUM_CH
  localparam int unsigned TSR_PERF_INFO_X_LSB     = 8;
  localparam int unsigned TSR_PERF_INFO_X_W       = 8;
  // Q_DEPTH (EX §5.1, CC §7.4): queue depth of unit u at [5u+4:5u]; 0 = unit absent
  localparam logic [31:0] TSR_Q_DEPTH_DMA      = 32'h0000_001F;  // [4:0] U200 M6: 8
  localparam int unsigned TSR_Q_DEPTH_DMA_LSB  = 0;
  localparam int unsigned TSR_Q_DEPTH_DMA_W    = 5;
  localparam logic [31:0] TSR_Q_DEPTH_QACT     = 32'h0000_03E0;  // [9:5] 4
  localparam int unsigned TSR_Q_DEPTH_QACT_LSB = 5;
  localparam int unsigned TSR_Q_DEPTH_QACT_W   = 5;
  localparam logic [31:0] TSR_Q_DEPTH_MVM      = 32'h0000_7C00;  // [14:10] 4
  localparam int unsigned TSR_Q_DEPTH_MVM_LSB  = 10;
  localparam int unsigned TSR_Q_DEPTH_MVM_W    = 5;
  localparam logic [31:0] TSR_Q_DEPTH_VU       = 32'h000F_8000;  // [19:15] 16
  localparam int unsigned TSR_Q_DEPTH_VU_LSB   = 15;
  localparam int unsigned TSR_Q_DEPTH_VU_W     = 5;
  localparam logic [31:0] TSR_Q_DEPTH_ATT      = 32'h01F0_0000;  // [24:20] 0: absent in M6
  localparam int unsigned TSR_Q_DEPTH_ATT_LSB  = 20;
  localparam int unsigned TSR_Q_DEPTH_ATT_W    = 5;
  localparam logic [31:0] TSR_Q_DEPTH_SYNC     = 32'h3E00_0000;  // [29:25] 4
  localparam int unsigned TSR_Q_DEPTH_SYNC_LSB = 25;
  localparam int unsigned TSR_Q_DEPTH_SYNC_W   = 5;
  // LOG_CFG (EX §4.10): host RW at 0x098, RISC-V read-only copy at 0x288
  localparam logic [31:0] TSR_LOG_CFG_ENABLE     = 32'h0000_0001;  // [0] 0: the SDK drops every record
  localparam int unsigned TSR_LOG_CFG_ENABLE_LSB = 0;
  localparam int unsigned TSR_LOG_CFG_ENABLE_W   = 1;
  localparam logic [31:0] TSR_LOG_CFG_BLOCK      = 32'h0000_0002;  // [1] the SDK waits for space instead of dropping
  localparam int unsigned TSR_LOG_CFG_BLOCK_LSB  = 1;
  localparam int unsigned TSR_LOG_CFG_BLOCK_W    = 1;
  localparam logic [31:0] TSR_LOG_CFG_LEVEL      = 32'h0000_00F0;  // [7:4] the SDK drops records above this level (the only runtime filter)
  localparam int unsigned TSR_LOG_CFG_LEVEL_LSB  = 4;
  localparam int unsigned TSR_LOG_CFG_LEVEL_W    = 4;
  // LOAD_CTRL (EX §5.1, §3.3)
  localparam logic [31:0] TSR_LOAD_CTRL_VERIFY     = 32'h0000_0001;  // [0] read LOAD_COUNT words back and CRC them
  localparam int unsigned TSR_LOAD_CTRL_VERIFY_LSB = 0;
  localparam int unsigned TSR_LOAD_CTRL_VERIFY_W   = 1;
  // LOAD_STATUS (EX §5.1, §3.3)
  localparam logic [31:0] TSR_LOAD_STATUS_VERIFY_BUSY     = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_LOAD_STATUS_VERIFY_BUSY_LSB = 0;
  localparam int unsigned TSR_LOAD_STATUS_VERIFY_BUSY_W   = 1;
  // Q_STATE (EX §5.4): queue state of unit u
  localparam logic [31:0] TSR_Q_STATE_ENTRIES      = 32'h0000_001F;  // [4:0] entries in the queue
  localparam int unsigned TSR_Q_STATE_ENTRIES_LSB  = 0;
  localparam int unsigned TSR_Q_STATE_ENTRIES_W    = 5;
  localparam logic [31:0] TSR_Q_STATE_RUNNING      = 32'h0000_0020;  // [5] the unit holds an op
  localparam int unsigned TSR_Q_STATE_RUNNING_LSB  = 5;
  localparam int unsigned TSR_Q_STATE_RUNNING_W    = 1;
  localparam logic [31:0] TSR_Q_STATE_DEP_WAIT     = 32'h0000_0040;  // [6] the head waits for deps
  localparam int unsigned TSR_Q_STATE_DEP_WAIT_LSB = 6;
  localparam int unsigned TSR_Q_STATE_DEP_WAIT_W   = 1;
  localparam logic [31:0] TSR_Q_STATE_HELD         = 32'h0000_0080;  // [7]
  localparam int unsigned TSR_Q_STATE_HELD_LSB     = 7;
  localparam int unsigned TSR_Q_STATE_HELD_W       = 1;
  localparam logic [31:0] TSR_Q_STATE_ERR          = 32'h0000_0100;  // [8] error state
  localparam int unsigned TSR_Q_STATE_ERR_LSB      = 8;
  localparam int unsigned TSR_Q_STATE_ERR_W        = 1;
  localparam logic [31:0] TSR_Q_STATE_HUNG         = 32'h0000_0200;  // [9] watchdog
  localparam int unsigned TSR_Q_STATE_HUNG_LSB     = 9;
  localparam int unsigned TSR_Q_STATE_HUNG_W       = 1;
  // Q_FIFO (EX §5.4)
  localparam logic [31:0] TSR_Q_FIFO_ENTRIES     = 32'h0000_00FF;  // [7:0] issue-FIFO entries waiting for the dispatcher
  localparam int unsigned TSR_Q_FIFO_ENTRIES_LSB = 0;
  localparam int unsigned TSR_Q_FIFO_ENTRIES_W   = 8;
  // ADAPTER_CTRL (EX §9.2)
  localparam logic [31:0] TSR_ADAPTER_CTRL_MEM_EN     = 32'h0000_0001;  // [0] rising edge: latch and alignment-check all bases; requests are held while 0
  localparam int unsigned TSR_ADAPTER_CTRL_MEM_EN_LSB = 0;
  localparam int unsigned TSR_ADAPTER_CTRL_MEM_EN_W   = 1;
  // ADAPTER_STATUS (EX §9.2, docs/platform-u200.md §2): sticky, W1C
  localparam logic [31:0] TSR_ADAPTER_STATUS_RRESP_ERR           = 32'h0000_0001;  // [0] RRESP error
  localparam int unsigned TSR_ADAPTER_STATUS_RRESP_ERR_LSB       = 0;
  localparam int unsigned TSR_ADAPTER_STATUS_RRESP_ERR_W         = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_BRESP_ERR           = 32'h0000_0002;  // [1] BRESP error
  localparam int unsigned TSR_ADAPTER_STATUS_BRESP_ERR_LSB       = 1;
  localparam int unsigned TSR_ADAPTER_STATUS_BRESP_ERR_W         = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_ORPHAN_RESP         = 32'h0000_0004;  // [2] response with no outstanding request for its id
  localparam int unsigned TSR_ADAPTER_STATUS_ORPHAN_RESP_LSB     = 2;
  localparam int unsigned TSR_ADAPTER_STATUS_ORPHAN_RESP_W       = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_BASE_MISALIGNED     = 32'h0000_0008;  // [3] a base was not 4 KiB aligned at the MEM_EN edge
  localparam int unsigned TSR_ADAPTER_STATUS_BASE_MISALIGNED_LSB = 3;
  localparam int unsigned TSR_ADAPTER_STATUS_BASE_MISALIGNED_W   = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_REQ_HELD            = 32'h0000_0010;  // [4] a core memory request was held while MEM_EN = 0
  localparam int unsigned TSR_ADAPTER_STATUS_REQ_HELD_LSB        = 4;
  localparam int unsigned TSR_ADAPTER_STATUS_REQ_HELD_W          = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_BASE_LOCKED         = 32'h0000_0020;  // [5] a base write while MEM_EN = 1 was ignored
  localparam int unsigned TSR_ADAPTER_STATUS_BASE_LOCKED_LSB     = 5;
  localparam int unsigned TSR_ADAPTER_STATUS_BASE_LOCKED_W       = 1;
  localparam logic [31:0] TSR_ADAPTER_STATUS_ERR_COUNT           = 32'hFFFF_0000;  // [31:16] saturating count of error responses
  localparam int unsigned TSR_ADAPTER_STATUS_ERR_COUNT_LSB       = 16;
  localparam int unsigned TSR_ADAPTER_STATUS_ERR_COUNT_W         = 16;
  // ADAPTER_VERSION (EX §9.2): {16'h0001, 8'h00, NUM_CH}
  localparam logic [31:0] TSR_ADAPTER_VERSION_NUM_CH     = 32'h0000_00FF;  // [7:0]
  localparam int unsigned TSR_ADAPTER_VERSION_NUM_CH_LSB = 0;
  localparam int unsigned TSR_ADAPTER_VERSION_NUM_CH_W   = 8;
  localparam logic [31:0] TSR_ADAPTER_VERSION_MAJOR      = 32'hFFFF_0000;  // [31:16] 1
  localparam int unsigned TSR_ADAPTER_VERSION_MAJOR_LSB  = 16;
  localparam int unsigned TSR_ADAPTER_VERSION_MAJOR_W    = 16;
  // HANDLE (EX §4.3): DRAM handle (descriptor args, OPL, DBG_TRW_BASE)
  localparam logic [31:0] TSR_HANDLE_BEAT     = 32'h0FFF_FFFF;  // [27:0] beat in the channel
  localparam int unsigned TSR_HANDLE_BEAT_LSB = 0;
  localparam int unsigned TSR_HANDLE_BEAT_W   = 28;
  localparam logic [31:0] TSR_HANDLE_CH       = 32'hF000_0000;  // [31:28] channel
  localparam int unsigned TSR_HANDLE_CH_LSB   = 28;
  localparam int unsigned TSR_HANDLE_CH_W     = 4;
  // TICKET (CC §7.5): ticket {unit, seq}; unit 15 = none; (u, s) done iff (D[u] - s) mod 2^28 in [1, 2^27)
  localparam logic [31:0] TSR_TICKET_SEQ      = 32'h0FFF_FFFF;  // [27:0] per-unit sequence
  localparam int unsigned TSR_TICKET_SEQ_LSB  = 0;
  localparam int unsigned TSR_TICKET_SEQ_W    = 28;
  localparam logic [31:0] TSR_TICKET_UNIT     = 32'hF000_0000;  // [31:28] unit id
  localparam int unsigned TSR_TICKET_UNIT_LSB = 28;
  localparam int unsigned TSR_TICKET_UNIT_W   = 4;
  // DESC (EX §4.3): descriptor word w0
  localparam logic [31:0] TSR_DESC_TYPE       = 32'h0000_00FF;  // [7:0] descriptor type; other values reserved (BAD_DESC)
  localparam int unsigned TSR_DESC_TYPE_LSB   = 0;
  localparam int unsigned TSR_DESC_TYPE_W     = 8;
  localparam logic [31:0] TSR_DESC_PROF       = 32'h0000_0100;  // [8] profiled launch: REC_CFG.ALL and NODE markers for the launch
  localparam int unsigned TSR_DESC_PROF_LSB   = 8;
  localparam int unsigned TSR_DESC_PROF_W     = 1;
  localparam logic [31:0] TSR_DESC_KERNEL     = 32'hFFFF_0000;  // [31:16] index into .tsr_kernels (LAUNCH); 0 for PING
  localparam int unsigned TSR_DESC_KERNEL_LSB = 16;
  localparam int unsigned TSR_DESC_KERNEL_W   = 16;
  localparam logic [7:0] TSR_DESC_TYPE_LAUNCH = 8'd1;
  localparam logic [7:0] TSR_DESC_TYPE_PING   = 8'd2;
  // LCQ_R0 (EX §4.4): record word r0, one layout for every record type
  localparam logic [31:0] TSR_LCQ_R0_TYPE             = 32'h0000_00FF;  // [7:0] record type
  localparam int unsigned TSR_LCQ_R0_TYPE_LSB         = 0;
  localparam int unsigned TSR_LCQ_R0_TYPE_W           = 8;
  localparam logic [31:0] TSR_LCQ_R0_STATUS           = 32'h0000_FF00;  // [15:8] launch STATUS (DONE), 4 for TRAP; kernel-defined for kernel events
  localparam int unsigned TSR_LCQ_R0_STATUS_LSB       = 8;
  localparam int unsigned TSR_LCQ_R0_STATUS_W         = 8;
  localparam logic [31:0] TSR_LCQ_R0_ERR_CODE         = 32'h00FF_0000;  // [23:16] code of the first error, 0 if none
  localparam int unsigned TSR_LCQ_R0_ERR_CODE_LSB     = 16;
  localparam int unsigned TSR_LCQ_R0_ERR_CODE_W       = 8;
  localparam logic [31:0] TSR_LCQ_R0_SEQ              = 32'hFF00_0000;  // [31:24] SEQ[7:0] of the descriptor
  localparam int unsigned TSR_LCQ_R0_SEQ_LSB          = 24;
  localparam int unsigned TSR_LCQ_R0_SEQ_W            = 8;
  localparam logic [7:0] TSR_LCQ_R0_TYPE_DONE         = 8'd1;           // answer
  localparam logic [7:0] TSR_LCQ_R0_TYPE_PONG         = 8'd2;           // answer
  localparam logic [7:0] TSR_LCQ_R0_TYPE_HELLO        = 8'd128;         // event
  localparam logic [7:0] TSR_LCQ_R0_TYPE_TRAP         = 8'd129;         // answer if ACTIVE, else event
  localparam logic [7:0] TSR_LCQ_R0_TYPE_EVENT_FIRST  = 8'd192;         // kernel events 0xC0..0xFF (ANSWER = 0)
  localparam logic [7:0] TSR_LCQ_R0_TYPE_EVENT_LAST   = 8'd255;
  localparam logic [7:0] TSR_LCQ_R0_STATUS_OK         = 8'd0;           // RET >= 0 and no op of the launch failed
  localparam logic [7:0] TSR_LCQ_R0_STATUS_TILE_ERR   = 8'd1;           // the first error has a code other than ABORTED; r3..r6 describe it
  localparam logic [7:0] TSR_LCQ_R0_STATUS_BAD_DESC   = 8'd2;           // the descriptor was refused; the kernel did not run
  localparam logic [7:0] TSR_LCQ_R0_STATUS_KERNEL_ERR = 8'd3;           // RET < 0, no tile error
  localparam logic [7:0] TSR_LCQ_R0_STATUS_TRAP       = 8'd4;           // TRAP records only
  localparam logic [7:0] TSR_LCQ_R0_STATUS_ABORTED    = 8'd5;           // cancelled (ERR_CODE 0), or the first error is ABORTED (ERR_CODE 8)
  localparam logic [7:0] TSR_LCQ_R0_STATUS_PLAT_ERR   = 8'd6;           // STATUS.PLAT_ERR was set when the runtime answered
  localparam logic [7:0] TSR_LCQ_R0_STATUS_STOPPED    = 8'd7;           // the kernel honoured a soft stop
  // HELLO_R4 (EX §4.4, SW §6.9): HELLO record word r4
  localparam logic [31:0] TSR_HELLO_R4_SDK_ABI       = 32'h0000_FFFF;  // [15:0] Basis ABI
  localparam int unsigned TSR_HELLO_R4_SDK_ABI_LSB   = 0;
  localparam int unsigned TSR_HELLO_R4_SDK_ABI_W     = 16;
  localparam logic [31:0] TSR_HELLO_R4_KINDS_SIG     = 32'hFFFF_0000;  // [31:16] kind table signature (opl::kinds_sig)
  localparam int unsigned TSR_HELLO_R4_KINDS_SIG_LSB = 16;
  localparam int unsigned TSR_HELLO_R4_KINDS_SIG_W   = 16;
  // HELLO_R6 (EX §4.4, SW §6.9): HELLO record word r6: Tesseract Basis version; [31:24] 0
  localparam logic [31:0] TSR_HELLO_R6_PATCH     = 32'h0000_00FF;  // [7:0]
  localparam int unsigned TSR_HELLO_R6_PATCH_LSB = 0;
  localparam int unsigned TSR_HELLO_R6_PATCH_W   = 8;
  localparam logic [31:0] TSR_HELLO_R6_MINOR     = 32'h0000_FF00;  // [15:8]
  localparam int unsigned TSR_HELLO_R6_MINOR_LSB = 8;
  localparam int unsigned TSR_HELLO_R6_MINOR_W   = 8;
  localparam logic [31:0] TSR_HELLO_R6_MAJOR     = 32'h00FF_0000;  // [23:16]
  localparam int unsigned TSR_HELLO_R6_MAJOR_LSB = 16;
  localparam int unsigned TSR_HELLO_R6_MAJOR_W   = 8;
  // ISSUE_PUSH (CC §5.1): ISSUE_PUSH write data
  localparam logic [31:0] TSR_ISSUE_PUSH_REC       = 32'h0000_0001;  // [0] completion record wanted (DB §8.2)
  localparam int unsigned TSR_ISSUE_PUSH_REC_LSB   = 0;
  localparam int unsigned TSR_ISSUE_PUSH_REC_W     = 1;
  localparam logic [31:0] TSR_ISSUE_PUSH_WMASK     = 32'h0000_FF00;  // [15:8] word i with bit i clear is sent as 0; w0 is always sent
  localparam int unsigned TSR_ISSUE_PUSH_WMASK_LSB = 8;
  localparam int unsigned TSR_ISSUE_PUSH_WMASK_W   = 8;
  // ISSUE_CREDIT (CC §5.1): min(free queue slots, free issue-FIFO entries, 15) per unit
  localparam logic [31:0] TSR_ISSUE_CREDIT_DMA      = 32'h0000_000F;  // [3:0]
  localparam int unsigned TSR_ISSUE_CREDIT_DMA_LSB  = 0;
  localparam int unsigned TSR_ISSUE_CREDIT_DMA_W    = 4;
  localparam logic [31:0] TSR_ISSUE_CREDIT_QACT     = 32'h0000_00F0;  // [7:4]
  localparam int unsigned TSR_ISSUE_CREDIT_QACT_LSB = 4;
  localparam int unsigned TSR_ISSUE_CREDIT_QACT_W   = 4;
  localparam logic [31:0] TSR_ISSUE_CREDIT_MVM      = 32'h0000_0F00;  // [11:8]
  localparam int unsigned TSR_ISSUE_CREDIT_MVM_LSB  = 8;
  localparam int unsigned TSR_ISSUE_CREDIT_MVM_W    = 4;
  localparam logic [31:0] TSR_ISSUE_CREDIT_VU       = 32'h0000_F000;  // [15:12]
  localparam int unsigned TSR_ISSUE_CREDIT_VU_LSB   = 12;
  localparam int unsigned TSR_ISSUE_CREDIT_VU_W     = 4;
  localparam logic [31:0] TSR_ISSUE_CREDIT_ATT      = 32'h000F_0000;  // [19:16]
  localparam int unsigned TSR_ISSUE_CREDIT_ATT_LSB  = 16;
  localparam int unsigned TSR_ISSUE_CREDIT_ATT_W    = 4;
  localparam logic [31:0] TSR_ISSUE_CREDIT_SYNC     = 32'h00F0_0000;  // [23:20]
  localparam int unsigned TSR_ISSUE_CREDIT_SYNC_LSB = 20;
  localparam int unsigned TSR_ISSUE_CREDIT_SYNC_W   = 4;
  // ISSUE_UNITS (CC §5.1): bit u = unit u is built (constant)
  localparam logic [31:0] TSR_ISSUE_UNITS_DMA      = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_ISSUE_UNITS_DMA_LSB  = 0;
  localparam int unsigned TSR_ISSUE_UNITS_DMA_W    = 1;
  localparam logic [31:0] TSR_ISSUE_UNITS_QACT     = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_ISSUE_UNITS_QACT_LSB = 1;
  localparam int unsigned TSR_ISSUE_UNITS_QACT_W   = 1;
  localparam logic [31:0] TSR_ISSUE_UNITS_MVM      = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_ISSUE_UNITS_MVM_LSB  = 2;
  localparam int unsigned TSR_ISSUE_UNITS_MVM_W    = 1;
  localparam logic [31:0] TSR_ISSUE_UNITS_VU       = 32'h0000_0008;  // [3]
  localparam int unsigned TSR_ISSUE_UNITS_VU_LSB   = 3;
  localparam int unsigned TSR_ISSUE_UNITS_VU_W     = 1;
  localparam logic [31:0] TSR_ISSUE_UNITS_ATT      = 32'h0000_0010;  // [4] 0 in M6
  localparam int unsigned TSR_ISSUE_UNITS_ATT_LSB  = 4;
  localparam int unsigned TSR_ISSUE_UNITS_ATT_W    = 1;
  localparam logic [31:0] TSR_ISSUE_UNITS_SYNC     = 32'h0000_0020;  // [5]
  localparam int unsigned TSR_ISSUE_UNITS_SYNC_LSB = 5;
  localparam int unsigned TSR_ISSUE_UNITS_SYNC_W   = 1;
  // ISSUE_STATUS (CC §5.1): R/W1C
  localparam logic [31:0] TSR_ISSUE_STATUS_PUSH_TIMEOUT     = 32'h0000_0001;  // [0] a push waited ISSUE_WAIT_MAX and faulted
  localparam int unsigned TSR_ISSUE_STATUS_PUSH_TIMEOUT_LSB = 0;
  localparam int unsigned TSR_ISSUE_STATUS_PUSH_TIMEOUT_W   = 1;
  localparam logic [31:0] TSR_ISSUE_STATUS_ERR_IRQ          = 32'h0000_0002;  // [1] sticky, set on the rising edge of 'some unit in error state' (irq_fast[1])
  localparam int unsigned TSR_ISSUE_STATUS_ERR_IRQ_LSB      = 1;
  localparam int unsigned TSR_ISSUE_STATUS_ERR_IRQ_W        = 1;
  // QCTRL (CC §5.1): one bit per write, in push order through the issue FIFO
  localparam logic [31:0] TSR_QCTRL_QABORT        = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_QCTRL_QABORT_LSB    = 0;
  localparam int unsigned TSR_QCTRL_QABORT_W      = 1;
  localparam logic [31:0] TSR_QCTRL_QHOLD_SET     = 32'h0000_0002;  // [1] set the RISC-V hold source
  localparam int unsigned TSR_QCTRL_QHOLD_SET_LSB = 1;
  localparam int unsigned TSR_QCTRL_QHOLD_SET_W   = 1;
  localparam logic [31:0] TSR_QCTRL_QHOLD_CLR     = 32'h0000_0004;  // [2] clear the RISC-V hold source
  localparam int unsigned TSR_QCTRL_QHOLD_CLR_LSB = 2;
  localparam int unsigned TSR_QCTRL_QHOLD_CLR_W   = 1;
  localparam logic [31:0] TSR_QCTRL_QCLEAR        = 32'h0000_0008;  // [3] gets a SYNC ticket (ISSUE_TICKET)
  localparam int unsigned TSR_QCTRL_QCLEAR_LSB    = 3;
  localparam int unsigned TSR_QCTRL_QCLEAR_W      = 1;
  // QSTATUS (CC §5.1): queue status (RISC-V view)
  localparam logic [31:0] TSR_QSTATUS_QIDLE         = 32'h0000_0001;  // [0] ISSUED_SEQ = DONE_SEQ for every u and the issue FIFO empty
  localparam int unsigned TSR_QSTATUS_QIDLE_LSB     = 0;
  localparam int unsigned TSR_QSTATUS_QIDLE_W       = 1;
  localparam logic [31:0] TSR_QSTATUS_HOLD          = 32'h0000_0002;  // [1] a hold is in effect
  localparam int unsigned TSR_QSTATUS_HOLD_LSB      = 1;
  localparam int unsigned TSR_QSTATUS_HOLD_W        = 1;
  localparam logic [31:0] TSR_QSTATUS_HOLD_HOST     = 32'h0000_0004;  // [2] host hold
  localparam int unsigned TSR_QSTATUS_HOLD_HOST_LSB = 2;
  localparam int unsigned TSR_QSTATUS_HOLD_HOST_W   = 1;
  localparam logic [31:0] TSR_QSTATUS_HOLD_RV       = 32'h0000_0008;  // [3] RISC-V hold
  localparam int unsigned TSR_QSTATUS_HOLD_RV_LSB   = 3;
  localparam int unsigned TSR_QSTATUS_HOLD_RV_W     = 1;
  localparam logic [31:0] TSR_QSTATUS_HOLD_HALT     = 32'h0000_0010;  // [4] debug-halt hold
  localparam int unsigned TSR_QSTATUS_HOLD_HALT_LSB = 4;
  localparam int unsigned TSR_QSTATUS_HOLD_HALT_W   = 1;
  localparam logic [31:0] TSR_QSTATUS_ERR           = 32'h0000_3F00;  // [13:8] error state of unit u at bit 8 + u
  localparam int unsigned TSR_QSTATUS_ERR_LSB       = 8;
  localparam int unsigned TSR_QSTATUS_ERR_W         = 6;
  localparam logic [31:0] TSR_QSTATUS_NONEMPTY      = 32'h003F_0000;  // [21:16] queue u non-empty at bit 16 + u
  localparam int unsigned TSR_QSTATUS_NONEMPTY_LSB  = 16;
  localparam int unsigned TSR_QSTATUS_NONEMPTY_W    = 6;
  // WAIT_CREDIT (CC §5.1): irq_fast[5] while the unit's credit != 0
  localparam logic [31:0] TSR_WAIT_CREDIT_UNIT     = 32'h0000_0007;  // [2:0]
  localparam int unsigned TSR_WAIT_CREDIT_UNIT_LSB = 0;
  localparam int unsigned TSR_WAIT_CREDIT_UNIT_W   = 3;
  localparam logic [31:0] TSR_WAIT_CREDIT_EN       = 32'h8000_0000;  // [31]
  localparam int unsigned TSR_WAIT_CREDIT_EN_LSB   = 31;
  localparam int unsigned TSR_WAIT_CREDIT_EN_W     = 1;
  // LQ_FLAGS (EX §4.5): RISC-V view of the launch state
  localparam logic [31:0] TSR_LQ_FLAGS_CANCEL_VALID     = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_LQ_FLAGS_CANCEL_VALID_LSB = 0;
  localparam int unsigned TSR_LQ_FLAGS_CANCEL_VALID_W   = 1;
  localparam logic [31:0] TSR_LQ_FLAGS_STOP_VALID       = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_LQ_FLAGS_STOP_VALID_LSB   = 1;
  localparam int unsigned TSR_LQ_FLAGS_STOP_VALID_W     = 1;
  localparam logic [31:0] TSR_LQ_FLAGS_ACTIVE           = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_LQ_FLAGS_ACTIVE_LSB       = 2;
  localparam int unsigned TSR_LQ_FLAGS_ACTIVE_W         = 1;
  localparam logic [31:0] TSR_LQ_FLAGS_PLAT_ERR         = 32'h0000_0008;  // [3] STATUS.PLAT_ERR
  localparam int unsigned TSR_LQ_FLAGS_PLAT_ERR_LSB     = 3;
  localparam int unsigned TSR_LQ_FLAGS_PLAT_ERR_W       = 1;
  // LCQ_POST (EX §4.5): LCQ_POST write data
  localparam logic [31:0] TSR_LCQ_POST_ANSWER     = 32'h0000_0001;  // [0] the record answers the running launch (ANSWERED += 1)
  localparam int unsigned TSR_LCQ_POST_ANSWER_LSB = 0;
  localparam int unsigned TSR_LCQ_POST_ANSWER_W   = 1;
  // REC_CFG (DB §10.2, §8.2): completion-record configuration
  localparam logic [31:0] TSR_REC_CFG_ALL          = 32'h0000_0001;  // [0] a record for every entry, not only ISSUE_PUSH.REC ones
  localparam int unsigned TSR_REC_CFG_ALL_LSB      = 0;
  localparam int unsigned TSR_REC_CFG_ALL_W        = 1;
  localparam logic [31:0] TSR_REC_CFG_IRQ_HALF     = 32'h0000_0002;  // [1] irq_fast[4] while REC_COUNT >= 32
  localparam int unsigned TSR_REC_CFG_IRQ_HALF_LSB = 1;
  localparam int unsigned TSR_REC_CFG_IRQ_HALF_W   = 1;
  // REC_DATA3 (DB §8.2): completion record word 3
  localparam logic [31:0] TSR_REC_DATA3_CODE       = 32'h0000_000F;  // [3:0] completion code
  localparam int unsigned TSR_REC_DATA3_CODE_LSB   = 0;
  localparam int unsigned TSR_REC_DATA3_CODE_W     = 4;
  localparam logic [31:0] TSR_REC_DATA3_RAN        = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_REC_DATA3_RAN_LSB    = 4;
  localparam int unsigned TSR_REC_DATA3_RAN_W      = 1;
  localparam logic [31:0] TSR_REC_DATA3_MARKER     = 32'h0000_0020;  // [5]
  localparam int unsigned TSR_REC_DATA3_MARKER_LSB = 5;
  localparam int unsigned TSR_REC_DATA3_MARKER_W   = 1;
  localparam logic [31:0] TSR_REC_DATA3_OPCODE     = 32'h0000_FF00;  // [15:8]
  localparam int unsigned TSR_REC_DATA3_OPCODE_LSB = 8;
  localparam int unsigned TSR_REC_DATA3_OPCODE_W   = 8;
  localparam logic [31:0] TSR_REC_DATA3_PIDX       = 32'hFFFF_0000;  // [31:16] push index [15:0]
  localparam int unsigned TSR_REC_DATA3_PIDX_LSB   = 16;
  localparam int unsigned TSR_REC_DATA3_PIDX_W     = 16;
  // DBG_FLAGS (DB §10.2): RISC-V view of DB's state (control window 0x400)
  localparam logic [31:0] TSR_DBG_FLAGS_DBG_STOP         = 32'h0000_0001;  // [0] a DB stop or the debug-halt hold is in effect
  localparam int unsigned TSR_DBG_FLAGS_DBG_STOP_LSB     = 0;
  localparam int unsigned TSR_DBG_FLAGS_DBG_STOP_W       = 1;
  localparam logic [31:0] TSR_DBG_FLAGS_SERIAL           = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_DBG_FLAGS_SERIAL_LSB       = 1;
  localparam int unsigned TSR_DBG_FLAGS_SERIAL_W         = 1;
  localparam logic [31:0] TSR_DBG_FLAGS_TRW_EN           = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_FLAGS_TRW_EN_LSB       = 2;
  localparam int unsigned TSR_DBG_FLAGS_TRW_EN_W         = 1;
  localparam logic [31:0] TSR_DBG_FLAGS_DMI_EN           = 32'h0000_0008;  // [3] a debugger may be attached
  localparam int unsigned TSR_DBG_FLAGS_DMI_EN_LSB       = 3;
  localparam int unsigned TSR_DBG_FLAGS_DMI_EN_W         = 1;
  localparam logic [31:0] TSR_DBG_FLAGS_HALT_ON_BOOT     = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_DBG_FLAGS_HALT_ON_BOOT_LSB = 4;
  localparam int unsigned TSR_DBG_FLAGS_HALT_ON_BOOT_W   = 1;
  // MARK (DB §8.4): TRACE_MARK value
  localparam logic [31:0] TSR_MARK_PAYLOAD           = 32'h0FFF_FFFF;  // [27:0]
  localparam int unsigned TSR_MARK_PAYLOAD_LSB       = 0;
  localparam int unsigned TSR_MARK_PAYLOAD_W         = 28;
  localparam logic [31:0] TSR_MARK_CLASS             = 32'hF000_0000;  // [31:28]
  localparam int unsigned TSR_MARK_CLASS_LSB         = 28;
  localparam int unsigned TSR_MARK_CLASS_W           = 4;
  localparam logic [3:0] TSR_MARK_CLASS_USER         = 4'd0;           // any payload
  localparam logic [3:0] TSR_MARK_CLASS_LAUNCH_BEGIN = 4'd1;           // payload: layout MARK_LAUNCH
  localparam logic [3:0] TSR_MARK_CLASS_LAUNCH_END   = 4'd2;           // payload: layout MARK_LAUNCH
  localparam logic [3:0] TSR_MARK_CLASS_NODE_BEGIN   = 4'd3;           // payload: op-list index
  localparam logic [3:0] TSR_MARK_CLASS_NODE_END     = 4'd4;           // payload: op-list index
  localparam logic [3:0] TSR_MARK_CLASS_LAYER        = 4'd5;           // payload: layer index
  localparam logic [3:0] TSR_MARK_CLASS_PHASE        = 4'd6;           // payload: kernel-defined id (.tsr_marks)
  // MARK_LAUNCH (DB §8.4): payload of LAUNCH_BEGIN / LAUNCH_END markers
  localparam logic [31:0] TSR_MARK_LAUNCH_SEQ        = 32'h0000_FFFF;  // [15:0] launch SEQ low bits
  localparam int unsigned TSR_MARK_LAUNCH_SEQ_LSB    = 0;
  localparam int unsigned TSR_MARK_LAUNCH_SEQ_W      = 16;
  localparam logic [31:0] TSR_MARK_LAUNCH_KERNEL     = 32'h0FFF_0000;  // [27:16] kernel index
  localparam int unsigned TSR_MARK_LAUNCH_KERNEL_LSB = 16;
  localparam int unsigned TSR_MARK_LAUNCH_KERNEL_W   = 12;
  localparam logic [31:0] TSR_MARK_LAUNCH_CLASS      = 32'hF000_0000;  // [31:28]
  localparam int unsigned TSR_MARK_LAUNCH_CLASS_LSB  = 28;
  localparam int unsigned TSR_MARK_LAUNCH_CLASS_W    = 4;
  // LOG_HDR (DB §6.1): log record header (the only definition; EX §4.10, SW §6.6)
  localparam logic [31:0] TSR_LOG_HDR_N             = 32'h0000_00FF;  // [7:0] payload words after the header
  localparam int unsigned TSR_LOG_HDR_N_LSB         = 0;
  localparam int unsigned TSR_LOG_HDR_N_W           = 8;
  localparam logic [31:0] TSR_LOG_HDR_LEVEL         = 32'h0000_0F00;  // [11:8]
  localparam int unsigned TSR_LOG_HDR_LEVEL_LSB     = 8;
  localparam int unsigned TSR_LOG_HDR_LEVEL_W       = 4;
  localparam logic [31:0] TSR_LOG_HDR_SOURCE        = 32'h0000_F000;  // [15:12]
  localparam int unsigned TSR_LOG_HDR_SOURCE_LSB    = 12;
  localparam int unsigned TSR_LOG_HDR_SOURCE_W      = 4;
  localparam logic [31:0] TSR_LOG_HDR_TYPE          = 32'h0007_0000;  // [18:16] 5..7 invalid
  localparam int unsigned TSR_LOG_HDR_TYPE_LSB      = 16;
  localparam int unsigned TSR_LOG_HDR_TYPE_W        = 3;
  localparam logic [31:0] TSR_LOG_HDR_TS            = 32'h0008_0000;  // [19] the first payload word is MTIME[31:0]
  localparam int unsigned TSR_LOG_HDR_TS_LSB        = 19;
  localparam int unsigned TSR_LOG_HDR_TS_W          = 1;
  localparam logic [31:0] TSR_LOG_HDR_LSEQ          = 32'h0FF0_0000;  // [27:20] SEQ[7:0] of the running launch; 0 outside launches
  localparam int unsigned TSR_LOG_HDR_LSEQ_LSB      = 20;
  localparam int unsigned TSR_LOG_HDR_LSEQ_W        = 8;
  localparam logic [31:0] TSR_LOG_HDR_SYNC          = 32'hF000_0000;  // [31:28] 0xA
  localparam int unsigned TSR_LOG_HDR_SYNC_LSB      = 28;
  localparam int unsigned TSR_LOG_HDR_SYNC_W        = 4;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_FATAL    = 4'd0;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_ERROR    = 4'd1;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_WARN     = 4'd2;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_INFO     = 4'd3;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_DEBUG    = 4'd4;
  localparam logic [3:0] TSR_LOG_HDR_LEVEL_TRACE    = 4'd5;
  localparam logic [3:0] TSR_LOG_HDR_SOURCE_RUNTIME = 4'd0;
  localparam logic [3:0] TSR_LOG_HDR_SOURCE_KERNEL  = 4'd1;
  localparam logic [3:0] TSR_LOG_HDR_SOURCE_TRAP    = 4'd2;           // trap handler
  localparam logic [3:0] TSR_LOG_HDR_SOURCE_DEBUG   = 4'd3;           // debug kernels
  localparam logic [2:0] TSR_LOG_HDR_TYPE_FMT       = 3'd0;           // format-string address in .tsr_logfmt, then arguments
  localparam logic [2:0] TSR_LOG_HDR_TYPE_TEXT      = 3'd1;           // UTF-8, zero-padded
  localparam logic [2:0] TSR_LOG_HDR_TYPE_TRAP      = 3'd2;           // the 44 words of __tsr_crash
  localparam logic [2:0] TSR_LOG_HDR_TYPE_EVENT     = 3'd3;           // event code (marker encoding), arguments
  localparam logic [2:0] TSR_LOG_HDR_TYPE_BLOB      = 3'd4;           // tag, raw words
  localparam logic [3:0] TSR_LOG_HDR_SYNC_MAGIC     = 4'd10;
  // TRW_W0 (DB §8.3): trace-writer record word 0
  localparam logic [31:0] TSR_TRW_W0_KIND       = 32'h0000_000F;  // [3:0]
  localparam int unsigned TSR_TRW_W0_KIND_LSB   = 0;
  localparam int unsigned TSR_TRW_W0_KIND_W     = 4;
  localparam logic [31:0] TSR_TRW_W0_UNIT       = 32'h0000_00F0;  // [7:4]
  localparam int unsigned TSR_TRW_W0_UNIT_LSB   = 4;
  localparam int unsigned TSR_TRW_W0_UNIT_W     = 4;
  localparam logic [31:0] TSR_TRW_W0_CODE       = 32'h0000_0F00;  // [11:8]
  localparam int unsigned TSR_TRW_W0_CODE_LSB   = 8;
  localparam int unsigned TSR_TRW_W0_CODE_W     = 4;
  localparam logic [31:0] TSR_TRW_W0_RAN        = 32'h0000_1000;  // [12]
  localparam int unsigned TSR_TRW_W0_RAN_LSB    = 12;
  localparam int unsigned TSR_TRW_W0_RAN_W      = 1;
  localparam logic [31:0] TSR_TRW_W0_REC        = 32'h0000_2000;  // [13] REC flag
  localparam int unsigned TSR_TRW_W0_REC_LSB    = 13;
  localparam int unsigned TSR_TRW_W0_REC_W      = 1;
  localparam logic [31:0] TSR_TRW_W0_TBP        = 32'h0000_4000;  // [14] tile-breakpoint hit
  localparam int unsigned TSR_TRW_W0_TBP_LSB    = 14;
  localparam int unsigned TSR_TRW_W0_TBP_W      = 1;
  localparam logic [31:0] TSR_TRW_W0_OPCODE     = 32'h00FF_0000;  // [23:16]
  localparam int unsigned TSR_TRW_W0_OPCODE_LSB = 16;
  localparam int unsigned TSR_TRW_W0_OPCODE_W   = 8;
  localparam logic [31:0] TSR_TRW_W0_W0_HI      = 32'hFF00_0000;  // [31:24] w0[15:8] of the op
  localparam int unsigned TSR_TRW_W0_W0_HI_LSB  = 24;
  localparam int unsigned TSR_TRW_W0_W0_HI_W    = 8;
  localparam logic [3:0] TSR_TRW_W0_KIND_OP     = 4'd0;
  localparam logic [3:0] TSR_TRW_W0_KIND_MARK   = 4'd1;
  localparam logic [3:0] TSR_TRW_W0_KIND_STOP   = 4'd2;
  localparam logic [3:0] TSR_TRW_W0_KIND_RVHALT = 4'd3;
  localparam logic [3:0] TSR_TRW_W0_KIND_DROP   = 4'd4;
  localparam logic [3:0] TSR_TRW_W0_KIND_PAD    = 4'd15;          // ignored
  // DBG_ID (DB §10.1): 0 = block absent
  localparam logic [31:0] TSR_DBG_ID_VERSION       = 32'h0000_00FF;  // [7:0] 1
  localparam int unsigned TSR_DBG_ID_VERSION_LSB   = 0;
  localparam int unsigned TSR_DBG_ID_VERSION_W     = 8;
  localparam logic [31:0] TSR_DBG_ID_HIST          = 32'h0000_0100;  // [8] history
  localparam int unsigned TSR_DBG_ID_HIST_LSB      = 8;
  localparam int unsigned TSR_DBG_ID_HIST_W        = 1;
  localparam logic [31:0] TSR_DBG_ID_TRW           = 32'h0000_0200;  // [9] trace writer
  localparam int unsigned TSR_DBG_ID_TRW_LSB       = 9;
  localparam int unsigned TSR_DBG_ID_TRW_W         = 1;
  localparam logic [31:0] TSR_DBG_ID_RVWP          = 32'h0000_0400;  // [10] RV watchpoints
  localparam int unsigned TSR_DBG_ID_RVWP_LSB      = 10;
  localparam int unsigned TSR_DBG_ID_RVWP_W        = 1;
  localparam logic [31:0] TSR_DBG_ID_REC           = 32'h0000_0800;  // [11] completion records
  localparam int unsigned TSR_DBG_ID_REC_LSB       = 11;
  localparam int unsigned TSR_DBG_ID_REC_W         = 1;
  localparam logic [31:0] TSR_DBG_ID_NTBP          = 32'h0000_7000;  // [14:12] tile breakpoints
  localparam int unsigned TSR_DBG_ID_NTBP_LSB      = 12;
  localparam int unsigned TSR_DBG_ID_NTBP_W        = 3;
  localparam logic [31:0] TSR_DBG_ID_NRVWP         = 32'h0001_8000;  // [16:15] RV watchpoints
  localparam int unsigned TSR_DBG_ID_NRVWP_LSB     = 15;
  localparam int unsigned TSR_DBG_ID_NRVWP_W       = 2;
  localparam logic [31:0] TSR_DBG_ID_LOG2_HIST     = 32'h00F0_0000;  // [23:20] log2 HIST_DEPTH
  localparam int unsigned TSR_DBG_ID_LOG2_HIST_LSB = 20;
  localparam int unsigned TSR_DBG_ID_LOG2_HIST_W   = 4;
  // DBG_CTRL (DB §10.1): pulses; one of bits 0..2 per write (QSTEP > RELEASE > STOP_NOW)
  localparam logic [31:0] TSR_DBG_CTRL_STOP_NOW           = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_DBG_CTRL_STOP_NOW_LSB       = 0;
  localparam int unsigned TSR_DBG_CTRL_STOP_NOW_W         = 1;
  localparam logic [31:0] TSR_DBG_CTRL_RELEASE            = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_DBG_CTRL_RELEASE_LSB        = 1;
  localparam int unsigned TSR_DBG_CTRL_RELEASE_W          = 1;
  localparam logic [31:0] TSR_DBG_CTRL_QSTEP              = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_CTRL_QSTEP_LSB          = 2;
  localparam int unsigned TSR_DBG_CTRL_QSTEP_W            = 1;
  localparam logic [31:0] TSR_DBG_CTRL_HIST_CLEAR         = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_DBG_CTRL_HIST_CLEAR_LSB     = 4;
  localparam int unsigned TSR_DBG_CTRL_HIST_CLEAR_W       = 1;
  localparam logic [31:0] TSR_DBG_CTRL_RV_SAMPLE          = 32'h0000_0020;  // [5] completes after the sample or 4,096 cycles
  localparam int unsigned TSR_DBG_CTRL_RV_SAMPLE_LSB      = 5;
  localparam int unsigned TSR_DBG_CTRL_RV_SAMPLE_W        = 1;
  localparam logic [31:0] TSR_DBG_CTRL_RVWP_APPLY         = 32'h0000_0040;  // [6]
  localparam int unsigned TSR_DBG_CTRL_RVWP_APPLY_LSB     = 6;
  localparam int unsigned TSR_DBG_CTRL_RVWP_APPLY_W       = 1;
  localparam logic [31:0] TSR_DBG_CTRL_TRW_FLUSH          = 32'h0000_0080;  // [7]
  localparam int unsigned TSR_DBG_CTRL_TRW_FLUSH_LSB      = 7;
  localparam int unsigned TSR_DBG_CTRL_TRW_FLUSH_W        = 1;
  localparam logic [31:0] TSR_DBG_CTRL_RV_HALT_CANCEL     = 32'h0000_0100;  // [8]
  localparam int unsigned TSR_DBG_CTRL_RV_HALT_CANCEL_LSB = 8;
  localparam int unsigned TSR_DBG_CTRL_RV_HALT_CANCEL_W   = 1;
  localparam logic [31:0] TSR_DBG_CTRL_MATCHQ_CLEAR       = 32'h0000_0200;  // [9]
  localparam int unsigned TSR_DBG_CTRL_MATCHQ_CLEAR_LSB   = 9;
  localparam int unsigned TSR_DBG_CTRL_MATCHQ_CLEAR_W     = 1;
  // DBG_CFG (DB §10.1): reset 0x30
  localparam logic [31:0] TSR_DBG_CFG_DMI_EN              = 32'h0000_0001;  // [0] gates the DMI window
  localparam int unsigned TSR_DBG_CFG_DMI_EN_LSB          = 0;
  localparam int unsigned TSR_DBG_CFG_DMI_EN_W            = 1;
  localparam logic [31:0] TSR_DBG_CFG_SERIAL              = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_DBG_CFG_SERIAL_LSB          = 1;
  localparam int unsigned TSR_DBG_CFG_SERIAL_W            = 1;
  localparam logic [31:0] TSR_DBG_CFG_ERR_STOP            = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_CFG_ERR_STOP_LSB        = 2;
  localparam int unsigned TSR_DBG_CFG_ERR_STOP_W          = 1;
  localparam logic [31:0] TSR_DBG_CFG_STOP_HALT_RV        = 32'h0000_0008;  // [3]
  localparam int unsigned TSR_DBG_CFG_STOP_HALT_RV_LSB    = 3;
  localparam int unsigned TSR_DBG_CFG_STOP_HALT_RV_W      = 1;
  localparam logic [31:0] TSR_DBG_CFG_HIST_EN             = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_DBG_CFG_HIST_EN_LSB         = 4;
  localparam int unsigned TSR_DBG_CFG_HIST_EN_W           = 1;
  localparam logic [31:0] TSR_DBG_CFG_HIST_FREEZE_ERR     = 32'h0000_0020;  // [5]
  localparam int unsigned TSR_DBG_CFG_HIST_FREEZE_ERR_LSB = 5;
  localparam int unsigned TSR_DBG_CFG_HIST_FREEZE_ERR_W   = 1;
  localparam logic [31:0] TSR_DBG_CFG_HIST_FREEZE         = 32'h0000_0040;  // [6]
  localparam int unsigned TSR_DBG_CFG_HIST_FREEZE_LSB     = 6;
  localparam int unsigned TSR_DBG_CFG_HIST_FREEZE_W       = 1;
  localparam logic [31:0] TSR_DBG_CFG_HUNG_STOP           = 32'h0000_0080;  // [7]
  localparam int unsigned TSR_DBG_CFG_HUNG_STOP_LSB       = 7;
  localparam int unsigned TSR_DBG_CFG_HUNG_STOP_W         = 1;
  localparam logic [31:0] TSR_DBG_CFG_HALT_ON_BOOT        = 32'h0000_0100;  // [8]
  localparam int unsigned TSR_DBG_CFG_HALT_ON_BOOT_LSB    = 8;
  localparam int unsigned TSR_DBG_CFG_HALT_ON_BOOT_W      = 1;
  localparam logic [31:0] TSR_DBG_CFG_RV_HALT_EN          = 32'h0000_0200;  // [9]
  localparam int unsigned TSR_DBG_CFG_RV_HALT_EN_LSB      = 9;
  localparam int unsigned TSR_DBG_CFG_RV_HALT_EN_W        = 1;
  localparam logic [31:0] TSR_DBG_CFG_TRW_EN              = 32'h0000_0400;  // [10]
  localparam int unsigned TSR_DBG_CFG_TRW_EN_LSB          = 10;
  localparam int unsigned TSR_DBG_CFG_TRW_EN_W            = 1;
  localparam logic [31:0] TSR_DBG_CFG_TRW_STOP_FULL       = 32'h0000_0800;  // [11]
  localparam int unsigned TSR_DBG_CFG_TRW_STOP_FULL_LSB   = 11;
  localparam int unsigned TSR_DBG_CFG_TRW_STOP_FULL_W     = 1;
  // DBG_STATUS (DB §10.1): levels and sticky W1C bits
  localparam logic [31:0] TSR_DBG_STATUS_STOP_ACTIVE           = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_DBG_STATUS_STOP_ACTIVE_LSB       = 0;
  localparam int unsigned TSR_DBG_STATUS_STOP_ACTIVE_W         = 1;
  localparam logic [31:0] TSR_DBG_STATUS_DRAINED               = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_DBG_STATUS_DRAINED_LSB           = 1;
  localparam int unsigned TSR_DBG_STATUS_DRAINED_W             = 1;
  localparam logic [31:0] TSR_DBG_STATUS_STEP_BUSY             = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_STATUS_STEP_BUSY_LSB         = 2;
  localparam int unsigned TSR_DBG_STATUS_STEP_BUSY_W           = 1;
  localparam logic [31:0] TSR_DBG_STATUS_DB_HELD               = 32'h0000_0008;  // [3] a head held by DB
  localparam int unsigned TSR_DBG_STATUS_DB_HELD_LSB           = 3;
  localparam int unsigned TSR_DBG_STATUS_DB_HELD_W             = 1;
  localparam logic [31:0] TSR_DBG_STATUS_CAUSE                 = 32'h0000_01F0;  // [8:4] cause of the current stop (layout DBG_CAUSE)
  localparam int unsigned TSR_DBG_STATUS_CAUSE_LSB             = 4;
  localparam int unsigned TSR_DBG_STATUS_CAUSE_W               = 5;
  localparam logic [31:0] TSR_DBG_STATUS_BP_HIT                = 32'h0000_1E00;  // [12:9] TBPs hit
  localparam int unsigned TSR_DBG_STATUS_BP_HIT_LSB            = 9;
  localparam int unsigned TSR_DBG_STATUS_BP_HIT_W              = 4;
  localparam logic [31:0] TSR_DBG_STATUS_RVWP_HIT              = 32'h0000_6000;  // [14:13] sticky
  localparam int unsigned TSR_DBG_STATUS_RVWP_HIT_LSB          = 13;
  localparam int unsigned TSR_DBG_STATUS_RVWP_HIT_W            = 2;
  localparam logic [31:0] TSR_DBG_STATUS_RV_HALT_PEND          = 32'h0000_8000;  // [15]
  localparam int unsigned TSR_DBG_STATUS_RV_HALT_PEND_LSB      = 15;
  localparam int unsigned TSR_DBG_STATUS_RV_HALT_PEND_W        = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RVWP_BUSY             = 32'h0001_0000;  // [16]
  localparam int unsigned TSR_DBG_STATUS_RVWP_BUSY_LSB         = 16;
  localparam int unsigned TSR_DBG_STATUS_RVWP_BUSY_W           = 1;
  localparam logic [31:0] TSR_DBG_STATUS_HIST_FROZEN           = 32'h0002_0000;  // [17]
  localparam int unsigned TSR_DBG_STATUS_HIST_FROZEN_LSB       = 17;
  localparam int unsigned TSR_DBG_STATUS_HIST_FROZEN_W         = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RV_HALTED             = 32'h0004_0000;  // [18]
  localparam int unsigned TSR_DBG_STATUS_RV_HALTED_LSB         = 18;
  localparam int unsigned TSR_DBG_STATUS_RV_HALTED_W           = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RV_SLEEP              = 32'h0008_0000;  // [19]
  localparam int unsigned TSR_DBG_STATUS_RV_SLEEP_LSB          = 19;
  localparam int unsigned TSR_DBG_STATUS_RV_SLEEP_W            = 1;
  localparam logic [31:0] TSR_DBG_STATUS_TRW_DROP              = 32'h0010_0000;  // [20] sticky
  localparam int unsigned TSR_DBG_STATUS_TRW_DROP_LSB          = 20;
  localparam int unsigned TSR_DBG_STATUS_TRW_DROP_W            = 1;
  localparam logic [31:0] TSR_DBG_STATUS_REC_DROP              = 32'h0020_0000;  // [21] sticky
  localparam int unsigned TSR_DBG_STATUS_REC_DROP_LSB          = 21;
  localparam int unsigned TSR_DBG_STATUS_REC_DROP_W            = 1;
  localparam logic [31:0] TSR_DBG_STATUS_SBA_REFUSED           = 32'h0040_0000;  // [22] sticky
  localparam int unsigned TSR_DBG_STATUS_SBA_REFUSED_LSB       = 22;
  localparam int unsigned TSR_DBG_STATUS_SBA_REFUSED_W         = 1;
  localparam logic [31:0] TSR_DBG_STATUS_DMI_BLOCKED           = 32'h0080_0000;  // [23] sticky
  localparam int unsigned TSR_DBG_STATUS_DMI_BLOCKED_LSB       = 23;
  localparam int unsigned TSR_DBG_STATUS_DMI_BLOCKED_W         = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT     = 32'h0100_0000;  // [24] sticky
  localparam int unsigned TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT_LSB = 24;
  localparam int unsigned TSR_DBG_STATUS_RV_SAMPLE_TIMEOUT_W   = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RV_DFAULT             = 32'h0200_0000;  // [25] sticky
  localparam int unsigned TSR_DBG_STATUS_RV_DFAULT_LSB         = 25;
  localparam int unsigned TSR_DBG_STATUS_RV_DFAULT_W           = 1;
  localparam logic [31:0] TSR_DBG_STATUS_PTR_ERR               = 32'h0400_0000;  // [26] sticky
  localparam int unsigned TSR_DBG_STATUS_PTR_ERR_LSB           = 26;
  localparam int unsigned TSR_DBG_STATUS_PTR_ERR_W             = 1;
  localparam logic [31:0] TSR_DBG_STATUS_RVWP_TIMEOUT          = 32'h0800_0000;  // [27] sticky
  localparam int unsigned TSR_DBG_STATUS_RVWP_TIMEOUT_LSB      = 27;
  localparam int unsigned TSR_DBG_STATUS_RVWP_TIMEOUT_W        = 1;
  localparam logic [31:0] TSR_DBG_STATUS_MATCHQ_OVF            = 32'h1000_0000;  // [28] sticky, never expected
  localparam int unsigned TSR_DBG_STATUS_MATCHQ_OVF_LSB        = 28;
  localparam int unsigned TSR_DBG_STATUS_MATCHQ_OVF_W          = 1;
  localparam logic [31:0] TSR_DBG_STATUS_DMI_RESP_ERR          = 32'h2000_0000;  // [29] sticky
  localparam int unsigned TSR_DBG_STATUS_DMI_RESP_ERR_LSB      = 29;
  localparam int unsigned TSR_DBG_STATUS_DMI_RESP_ERR_W        = 1;
  localparam logic [31:0] TSR_DBG_STATUS_S_VALID               = 32'h8000_0000;  // [31]
  localparam int unsigned TSR_DBG_STATUS_S_VALID_LSB           = 31;
  localparam int unsigned TSR_DBG_STATUS_S_VALID_W             = 1;
  // DBG_CAUSE (DB §10.1, §5.2): bits of DBG_STATUS.CAUSE in the documented order (one bit per cause)
  localparam logic [4:0] TSR_DBG_CAUSE_NOW       = 5'h01;  // [0]
  localparam int unsigned TSR_DBG_CAUSE_NOW_LSB  = 0;
  localparam int unsigned TSR_DBG_CAUSE_NOW_W    = 1;
  localparam logic [4:0] TSR_DBG_CAUSE_BP        = 5'h02;  // [1]
  localparam int unsigned TSR_DBG_CAUSE_BP_LSB   = 1;
  localparam int unsigned TSR_DBG_CAUSE_BP_W     = 1;
  localparam logic [4:0] TSR_DBG_CAUSE_STEP      = 5'h04;  // [2]
  localparam int unsigned TSR_DBG_CAUSE_STEP_LSB = 2;
  localparam int unsigned TSR_DBG_CAUSE_STEP_W   = 1;
  localparam logic [4:0] TSR_DBG_CAUSE_ERR       = 5'h08;  // [3]
  localparam int unsigned TSR_DBG_CAUSE_ERR_LSB  = 3;
  localparam int unsigned TSR_DBG_CAUSE_ERR_W    = 1;
  localparam logic [4:0] TSR_DBG_CAUSE_HUNG      = 5'h10;  // [4]
  localparam int unsigned TSR_DBG_CAUSE_HUNG_LSB = 4;
  localparam int unsigned TSR_DBG_CAUSE_HUNG_W   = 1;
  // DBG_STEP_N (DB §10.1)
  localparam logic [31:0] TSR_DBG_STEP_N_N     = 32'h0000_FFFF;  // [15:0] entries per QSTEP (0 = 1)
  localparam int unsigned TSR_DBG_STEP_N_N_LSB = 0;
  localparam int unsigned TSR_DBG_STEP_N_N_W   = 16;
  // DBG_MATCHQ (DB §10.1)
  localparam logic [31:0] TSR_DBG_MATCHQ_PENDING     = 32'h0000_007F;  // [6:0] pending match thresholds
  localparam int unsigned TSR_DBG_MATCHQ_PENDING_LSB = 0;
  localparam int unsigned TSR_DBG_MATCHQ_PENDING_W   = 7;
  // DBG_QSTATE (DB §10.1, CC §8.3): dispatcher state
  localparam logic [31:0] TSR_DBG_QSTATE_DISP            = 32'h0000_0700;  // [10:8] dispatcher state (CC §8.3 order)
  localparam int unsigned TSR_DBG_QSTATE_DISP_LSB        = 8;
  localparam int unsigned TSR_DBG_QSTATE_DISP_W          = 3;
  localparam logic [31:0] TSR_DBG_QSTATE_OUTSTANDING     = 32'h00FF_0000;  // [23:16] outstanding entries, saturating
  localparam int unsigned TSR_DBG_QSTATE_OUTSTANDING_LSB = 16;
  localparam int unsigned TSR_DBG_QSTATE_OUTSTANDING_W   = 8;
  localparam logic [2:0] TSR_DBG_QSTATE_DISP_IDLE        = 3'd0;           // D_IDLE
  localparam logic [2:0] TSR_DBG_QSTATE_DISP_POP         = 3'd1;           // D_POP
  localparam logic [2:0] TSR_DBG_QSTATE_DISP_CHK         = 3'd2;           // D_CHK
  localparam logic [2:0] TSR_DBG_QSTATE_DISP_ENQ         = 3'd3;           // D_ENQ
  // DBG_RV_STATE (DB §10.1)
  localparam logic [31:0] TSR_DBG_RV_STATE_HALTED            = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_DBG_RV_STATE_HALTED_LSB        = 0;
  localparam int unsigned TSR_DBG_RV_STATE_HALTED_W          = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_SLEEP             = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_RV_STATE_SLEEP_LSB         = 2;
  localparam int unsigned TSR_DBG_RV_STATE_SLEEP_W           = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_IN_RESET          = 32'h0000_0008;  // [3]
  localparam int unsigned TSR_DBG_RV_STATE_IN_RESET_LSB      = 3;
  localparam int unsigned TSR_DBG_RV_STATE_IN_RESET_W        = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_SAMPLE_VALID      = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_DBG_RV_STATE_SAMPLE_VALID_LSB  = 4;
  localparam int unsigned TSR_DBG_RV_STATE_SAMPLE_VALID_W    = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_DMACTIVE          = 32'h0000_0020;  // [5] dmactive_o
  localparam int unsigned TSR_DBG_RV_STATE_DMACTIVE_LSB      = 5;
  localparam int unsigned TSR_DBG_RV_STATE_DMACTIVE_W        = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_NDMRESET          = 32'h0000_0040;  // [6] ndmreset_o
  localparam int unsigned TSR_DBG_RV_STATE_NDMRESET_LSB      = 6;
  localparam int unsigned TSR_DBG_RV_STATE_NDMRESET_W        = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_HALT_REQ          = 32'h0000_0080;  // [7] halt-request level
  localparam int unsigned TSR_DBG_RV_STATE_HALT_REQ_LSB      = 7;
  localparam int unsigned TSR_DBG_RV_STATE_HALT_REQ_W        = 1;
  localparam logic [31:0] TSR_DBG_RV_STATE_SAMPLE_SEQ        = 32'h0000_FF00;  // [15:8]
  localparam int unsigned TSR_DBG_RV_STATE_SAMPLE_SEQ_LSB    = 8;
  localparam int unsigned TSR_DBG_RV_STATE_SAMPLE_SEQ_W      = 8;
  localparam logic [31:0] TSR_DBG_RV_STATE_DEBUG_ENTRIES     = 32'hFFFF_0000;  // [31:16] debug-mode entries mod 2^16
  localparam int unsigned TSR_DBG_RV_STATE_DEBUG_ENTRIES_LSB = 16;
  localparam int unsigned TSR_DBG_RV_STATE_DEBUG_ENTRIES_W   = 16;
  // DBG_RVWP_CTL (DB §10.1): RV watchpoint j < 2
  localparam logic [31:0] TSR_DBG_RVWP_CTL_EN        = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_DBG_RVWP_CTL_EN_LSB    = 0;
  localparam int unsigned TSR_DBG_RVWP_CTL_EN_W      = 1;
  localparam logic [31:0] TSR_DBG_RVWP_CTL_LOAD      = 32'h0000_0002;  // [1]
  localparam int unsigned TSR_DBG_RVWP_CTL_LOAD_LSB  = 1;
  localparam int unsigned TSR_DBG_RVWP_CTL_LOAD_W    = 1;
  localparam logic [31:0] TSR_DBG_RVWP_CTL_STORE     = 32'h0000_0004;  // [2]
  localparam int unsigned TSR_DBG_RVWP_CTL_STORE_LSB = 2;
  localparam int unsigned TSR_DBG_RVWP_CTL_STORE_W   = 1;
  localparam logic [31:0] TSR_DBG_RVWP_CTL_HALT      = 32'h0000_0008;  // [3]
  localparam int unsigned TSR_DBG_RVWP_CTL_HALT_LSB  = 3;
  localparam int unsigned TSR_DBG_RVWP_CTL_HALT_W    = 1;
  // DBG_TBP_CTL (DB §10.1, §5.3): tile breakpoint i < 4
  localparam logic [31:0] TSR_DBG_TBP_CTL_EN             = 32'h0000_0001;  // [0]
  localparam int unsigned TSR_DBG_TBP_CTL_EN_LSB         = 0;
  localparam int unsigned TSR_DBG_TBP_CTL_EN_W           = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_KIND           = 32'h0000_000E;  // [3:1] 5..7 never match
  localparam int unsigned TSR_DBG_TBP_CTL_KIND_LSB       = 1;
  localparam int unsigned TSR_DBG_TBP_CTL_KIND_W         = 3;
  localparam logic [31:0] TSR_DBG_TBP_CTL_R              = 32'h0000_0010;  // [4]
  localparam int unsigned TSR_DBG_TBP_CTL_R_LSB          = 4;
  localparam int unsigned TSR_DBG_TBP_CTL_R_W            = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_W              = 32'h0000_0020;  // [5]
  localparam int unsigned TSR_DBG_TBP_CTL_W_LSB          = 5;
  localparam int unsigned TSR_DBG_TBP_CTL_W_W            = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_AFTER          = 32'h0000_0040;  // [6]
  localparam int unsigned TSR_DBG_TBP_CTL_AFTER_LSB      = 6;
  localparam int unsigned TSR_DBG_TBP_CTL_AFTER_W        = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_HALT_RV        = 32'h0000_0080;  // [7]
  localparam int unsigned TSR_DBG_TBP_CTL_HALT_RV_LSB    = 7;
  localparam int unsigned TSR_DBG_TBP_CTL_HALT_RV_W      = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_COUNT_ONLY     = 32'h0000_0100;  // [8]
  localparam int unsigned TSR_DBG_TBP_CTL_COUNT_ONLY_LSB = 8;
  localparam int unsigned TSR_DBG_TBP_CTL_COUNT_ONLY_W   = 1;
  localparam logic [31:0] TSR_DBG_TBP_CTL_CHMASK         = 32'hFFFF_0000;  // [31:16] channels of a DRAM breakpoint
  localparam int unsigned TSR_DBG_TBP_CTL_CHMASK_LSB     = 16;
  localparam int unsigned TSR_DBG_TBP_CTL_CHMASK_W       = 16;
  localparam logic [2:0] TSR_DBG_TBP_CTL_KIND_PIDX       = 3'd0;           // (pidx - A) mod 2^32 < (B - A) mod 2^32
  localparam logic [2:0] TSR_DBG_TBP_CTL_KIND_OPCODE     = 3'd1;           // A: layout TBP_OPC_A, B: TBP_OPC_B
  localparam logic [2:0] TSR_DBG_TBP_CTL_KIND_SPM        = 3'd2;           // an SPM extent overlaps lines [A, B)
  localparam logic [2:0] TSR_DBG_TBP_CTL_KIND_DRAM       = 3'd3;           // the DRAM extent overlaps beats [A, B)
  localparam logic [2:0] TSR_DBG_TBP_CTL_KIND_TICKET     = 3'd4;           // ticket = A
  // TBP_OPC_A (DB §5.3): DBG_TBP_A of an OPCODE breakpoint
  localparam logic [31:0] TSR_TBP_OPC_A_OPC          = 32'h0000_00FF;  // [7:0] opcode value
  localparam int unsigned TSR_TBP_OPC_A_OPC_LSB      = 0;
  localparam int unsigned TSR_TBP_OPC_A_OPC_W        = 8;
  localparam logic [31:0] TSR_TBP_OPC_A_OPC_MASK     = 32'h0000_FF00;  // [15:8] opcode care mask
  localparam int unsigned TSR_TBP_OPC_A_OPC_MASK_LSB = 8;
  localparam int unsigned TSR_TBP_OPC_A_OPC_MASK_W   = 8;
  localparam logic [31:0] TSR_TBP_OPC_A_SUB          = 32'h000F_0000;  // [19:16] sub value
  localparam int unsigned TSR_TBP_OPC_A_SUB_LSB      = 16;
  localparam int unsigned TSR_TBP_OPC_A_SUB_W        = 4;
  localparam logic [31:0] TSR_TBP_OPC_A_SUB_MASK     = 32'h00F0_0000;  // [23:20] sub care mask
  localparam int unsigned TSR_TBP_OPC_A_SUB_MASK_LSB = 20;
  localparam int unsigned TSR_TBP_OPC_A_SUB_MASK_W   = 4;
  // TBP_OPC_B (DB §5.3): DBG_TBP_B of an OPCODE breakpoint
  localparam logic [31:0] TSR_TBP_OPC_B_UNITS     = 32'h0000_00FF;  // [7:0] unit mask; 0 = any unit
  localparam int unsigned TSR_TBP_OPC_B_UNITS_LSB = 0;
  localparam int unsigned TSR_TBP_OPC_B_UNITS_W   = 8;
  // DBG_U_STATE (DB §5.5): per-unit probe u < 8
  localparam logic [31:0] TSR_DBG_U_STATE_ACCEPTED        = 32'h0000_0003;  // [1:0] ops accepted and not done
  localparam int unsigned TSR_DBG_U_STATE_ACCEPTED_LSB    = 0;
  localparam int unsigned TSR_DBG_U_STATE_ACCEPTED_W      = 2;
  localparam logic [31:0] TSR_DBG_U_STATE_HEAD            = 32'h0000_0070;  // [6:4] head state
  localparam int unsigned TSR_DBG_U_STATE_HEAD_LSB        = 4;
  localparam int unsigned TSR_DBG_U_STATE_HEAD_W          = 3;
  localparam logic [31:0] TSR_DBG_U_STATE_HOLD            = 32'h0000_0F00;  // [11:8] hold sources at the head (layout HOLD_SRC)
  localparam int unsigned TSR_DBG_U_STATE_HOLD_LSB        = 8;
  localparam int unsigned TSR_DBG_U_STATE_HOLD_W          = 4;
  localparam logic [31:0] TSR_DBG_U_STATE_OUTSTANDING     = 32'h000F_F000;  // [19:12] outstanding entries
  localparam int unsigned TSR_DBG_U_STATE_OUTSTANDING_LSB = 12;
  localparam int unsigned TSR_DBG_U_STATE_OUTSTANDING_W   = 8;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_EMPTY       = 3'd0;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_LOAD        = 3'd1;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_WAIT_DEP    = 3'd2;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_WAIT_SNAP   = 3'd3;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_HELD        = 3'd4;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_WAIT_UNIT   = 3'd5;
  localparam logic [2:0] TSR_DBG_U_STATE_HEAD_ISSUE       = 3'd6;
  // DBG_TRW_MASK (DB §10.1): reset 0x7FF
  localparam logic [31:0] TSR_DBG_TRW_MASK_UNITS       = 32'h0000_00FF;  // [7:0]
  localparam int unsigned TSR_DBG_TRW_MASK_UNITS_LSB   = 0;
  localparam int unsigned TSR_DBG_TRW_MASK_UNITS_W     = 8;
  localparam logic [31:0] TSR_DBG_TRW_MASK_MARKERS     = 32'h0000_0100;  // [8]
  localparam int unsigned TSR_DBG_TRW_MASK_MARKERS_LSB = 8;
  localparam int unsigned TSR_DBG_TRW_MASK_MARKERS_W   = 1;
  localparam logic [31:0] TSR_DBG_TRW_MASK_STOPS       = 32'h0000_0200;  // [9]
  localparam int unsigned TSR_DBG_TRW_MASK_STOPS_LSB   = 9;
  localparam int unsigned TSR_DBG_TRW_MASK_STOPS_W     = 1;
  localparam logic [31:0] TSR_DBG_TRW_MASK_RVHALTS     = 32'h0000_0400;  // [10]
  localparam int unsigned TSR_DBG_TRW_MASK_RVHALTS_LSB = 10;
  localparam int unsigned TSR_DBG_TRW_MASK_RVHALTS_W   = 1;
  // DBG_HIST_INFO (DB §10.1)
  localparam logic [31:0] TSR_DBG_HIST_INFO_PRESENT     = 32'h0000_0001;  // [0] the slot holds that push index, recorded after the last HIST_CLEAR
  localparam int unsigned TSR_DBG_HIST_INFO_PRESENT_LSB = 0;
  localparam int unsigned TSR_DBG_HIST_INFO_PRESENT_W   = 1;
  localparam logic [31:0] TSR_DBG_HIST_INFO_UNIT        = 32'h0000_00F0;  // [7:4]
  localparam int unsigned TSR_DBG_HIST_INFO_UNIT_LSB    = 4;
  localparam int unsigned TSR_DBG_HIST_INFO_UNIT_W      = 4;
  localparam logic [31:0] TSR_DBG_HIST_INFO_CODE        = 32'h0000_0F00;  // [11:8] checker code (0 = runs)
  localparam int unsigned TSR_DBG_HIST_INFO_CODE_LSB    = 8;
  localparam int unsigned TSR_DBG_HIST_INFO_CODE_W      = 4;
  localparam logic [31:0] TSR_DBG_HIST_INFO_REC         = 32'h0000_1000;  // [12]
  localparam int unsigned TSR_DBG_HIST_INFO_REC_LSB     = 12;
  localparam int unsigned TSR_DBG_HIST_INFO_REC_W       = 1;
  localparam logic [31:0] TSR_DBG_HIST_INFO_TBPS        = 32'h0001_E000;  // [16:13] TBPs matched
  localparam int unsigned TSR_DBG_HIST_INFO_TBPS_LSB    = 13;
  localparam int unsigned TSR_DBG_HIST_INFO_TBPS_W      = 4;
  localparam logic [31:0] TSR_DBG_HIST_INFO_STOP        = 32'h0002_0000;  // [17] stop point
  localparam int unsigned TSR_DBG_HIST_INFO_STOP_LSB    = 17;
  localparam int unsigned TSR_DBG_HIST_INFO_STOP_W      = 1;
  // IFIFO_CTRL (CC §7.2): w0 of a CTRL issue-FIFO entry
  localparam logic [31:0] TSR_IFIFO_CTRL_CODE           = 32'h0000_000F;  // [3:0]
  localparam int unsigned TSR_IFIFO_CTRL_CODE_LSB       = 0;
  localparam int unsigned TSR_IFIFO_CTRL_CODE_W         = 4;
  localparam logic [3:0] TSR_IFIFO_CTRL_CODE_QABORT     = 4'd1;
  localparam logic [3:0] TSR_IFIFO_CTRL_CODE_QHOLD_SET  = 4'd2;
  localparam logic [3:0] TSR_IFIFO_CTRL_CODE_QHOLD_CLR  = 4'd3;
  localparam logic [3:0] TSR_IFIFO_CTRL_CODE_QCLEAR     = 4'd4;           // ticketed
  localparam logic [3:0] TSR_IFIFO_CTRL_CODE_TRACE_MARK = 4'd5;           // w1 = marker value

  // ---------------------------------------------------------------------------------------------
  // Wide layouts (bit positions only)
  // ---------------------------------------------------------------------------------------------
  // IFIFO (CC §7.2): issue-FIFO entry (tsr_cdc_fifo, 32 entries LUTRAM)
  localparam int unsigned TSR_IFIFO_BITS       = 387;
  localparam int unsigned TSR_IFIFO_KIND_LSB   = 0;     // [1:0] entry kind
  localparam int unsigned TSR_IFIFO_KIND_W     = 2;
  localparam int unsigned TSR_IFIFO_W_LSB      = 2;     // [257:2] w0..w7 (OP; masked words are 0); CTRL: w0 = layout IFIFO_CTRL, w1 = marker value
  localparam int unsigned TSR_IFIFO_W_W        = 256;
  localparam int unsigned TSR_IFIFO_DEP0_LSB   = 258;   // [289:258] OP
  localparam int unsigned TSR_IFIFO_DEP0_W     = 32;
  localparam int unsigned TSR_IFIFO_DEP1_LSB   = 290;   // [321:290] OP
  localparam int unsigned TSR_IFIFO_DEP1_W     = 32;
  localparam int unsigned TSR_IFIFO_REC_LSB    = 322;   // [322]
  localparam int unsigned TSR_IFIFO_REC_W      = 1;
  localparam int unsigned TSR_IFIFO_TICKET_LSB = 323;   // [354:323] OP and QCLEAR; the dispatcher asserts it equals its own count
  localparam int unsigned TSR_IFIFO_TICKET_W   = 32;
  localparam int unsigned TSR_IFIFO_PC_LSB     = 355;   // [386:355] push PC (OP and TRACE_MARK)
  localparam int unsigned TSR_IFIFO_PC_W       = 32;
  localparam logic [1:0] TSR_IFIFO_KIND_OP     = 2'd0;
  localparam logic [1:0] TSR_IFIFO_KIND_CTRL   = 2'd1;

  // ---------------------------------------------------------------------------------------------
  // Word tables of fixed-size records and structures (word indices)
  // ---------------------------------------------------------------------------------------------
  // DESC (EX §4.3): launch descriptor (host -> RISC-V)
  localparam int unsigned TSR_DESC_NWORDS    = 8;
  localparam int unsigned TSR_DESC_HDR_WORD  = 0;  // layout DESC
  localparam int unsigned TSR_DESC_ARG0_WORD = 1;  // ARG0..ARG6: integers, SPM lines or DRAM handles (layout HANDLE)
  localparam int unsigned TSR_DESC_ARG0_N    = 7;
  // LCQ_DONE (EX §4.4): DONE record (0x01, answer)
  localparam int unsigned TSR_LCQ_DONE_NWORDS          = 8;
  localparam int unsigned TSR_LCQ_DONE_R0_WORD         = 0;  // layout LCQ_R0
  localparam int unsigned TSR_LCQ_DONE_SEQ_WORD        = 1;  // SEQ of the answered descriptor
  localparam int unsigned TSR_LCQ_DONE_RET_WORD        = 2;  // kernel return value (int32)
  localparam int unsigned TSR_LCQ_DONE_ERR_TICKET_WORD = 3;  // first failing ticket; TICKET_NONE = none
  localparam int unsigned TSR_LCQ_DONE_ERR_INFO_WORD   = 4;
  localparam int unsigned TSR_LCQ_DONE_ERR_W0_WORD     = 5;
  localparam int unsigned TSR_LCQ_DONE_ERR_PIDX_WORD   = 6;
  localparam int unsigned TSR_LCQ_DONE_DURATION_WORD   = 7;  // core cycles from the pop to the post, saturating
  // LCQ_PONG (EX §4.4): PONG record (0x02, answer)
  localparam int unsigned TSR_LCQ_PONG_NWORDS        = 8;
  localparam int unsigned TSR_LCQ_PONG_R0_WORD       = 0;
  localparam int unsigned TSR_LCQ_PONG_SEQ_WORD      = 1;
  localparam int unsigned TSR_LCQ_PONG_ARG0_WORD     = 2;  // ARG0 echo
  localparam int unsigned TSR_LCQ_PONG_DURATION_WORD = 7;
  // LCQ_HELLO (EX §4.4, SW §6.3): HELLO record (0x80, event)
  localparam int unsigned TSR_LCQ_HELLO_NWORDS           = 8;
  localparam int unsigned TSR_LCQ_HELLO_R0_WORD          = 0;
  localparam int unsigned TSR_LCQ_HELLO_TAKEN_WORD       = 1;  // TAKEN at the post
  localparam int unsigned TSR_LCQ_HELLO_ID_LO_WORD       = 2;  // image id [31:0]
  localparam int unsigned TSR_LCQ_HELLO_ID_HI_WORD       = 3;  // image id [63:32]
  localparam int unsigned TSR_LCQ_HELLO_VERSIONS_WORD    = 4;  // layout HELLO_R4
  localparam int unsigned TSR_LCQ_HELLO_N_KERNELS_WORD   = 5;  // kernel count
  localparam int unsigned TSR_LCQ_HELLO_BASIS_WORD       = 6;  // Tesseract Basis version, layout HELLO_R6
  localparam int unsigned TSR_LCQ_HELLO_BOOT_CYCLES_WORD = 7;  // boot core cycles
  // LCQ_TRAP (EX §4.4, DB §6.3): TRAP record (0x81)
  localparam int unsigned TSR_LCQ_TRAP_NWORDS        = 8;
  localparam int unsigned TSR_LCQ_TRAP_R0_WORD       = 0;
  localparam int unsigned TSR_LCQ_TRAP_SEQ_WORD      = 1;  // SEQ (answer) or TAKEN (event)
  localparam int unsigned TSR_LCQ_TRAP_MCAUSE_WORD   = 2;
  localparam int unsigned TSR_LCQ_TRAP_MEPC_WORD     = 3;
  localparam int unsigned TSR_LCQ_TRAP_MTVAL_WORD    = 4;
  localparam int unsigned TSR_LCQ_TRAP_RT_STATE_WORD = 5;  // at the trap
  localparam int unsigned TSR_LCQ_TRAP_RT_INFO_WORD  = 6;
  localparam int unsigned TSR_LCQ_TRAP_MTIME_WORD    = 7;  // MTIME[31:0]
  // LCQ_EVENT (EX §4.4): kernel event record (0xC0..0xFF)
  localparam int unsigned TSR_LCQ_EVENT_NWORDS    = 8;
  localparam int unsigned TSR_LCQ_EVENT_R0_WORD   = 0;  // type, kernel-defined [23:8], seq[7:0]
  localparam int unsigned TSR_LCQ_EVENT_SEQ_WORD  = 1;  // SEQ of the running launch
  localparam int unsigned TSR_LCQ_EVENT_DATA_WORD = 2;  // r2..r7 kernel-defined
  localparam int unsigned TSR_LCQ_EVENT_DATA_N    = 6;
  // REC (DB §8.2): completion record (REC_DATA0..3)
  localparam int unsigned TSR_REC_NWORDS      = 4;
  localparam int unsigned TSR_REC_TICKET_WORD = 0;  // ticket (unit 14 for markers)
  localparam int unsigned TSR_REC_T_END_WORD  = 1;  // MTIME[31:0] at completion (markers: dispatch time)
  localparam int unsigned TSR_REC_RUN_WORD    = 2;  // start -> end cycles, saturating (markers: the value)
  localparam int unsigned TSR_REC_INFO_WORD   = 3;  // layout REC_DATA3
  // TRW_OP (DB §8.3): trace-writer OP record
  localparam int unsigned TSR_TRW_OP_NWORDS        = 8;
  localparam int unsigned TSR_TRW_OP_HDR_WORD      = 0;  // layout TRW_W0
  localparam int unsigned TSR_TRW_OP_PIDX_WORD     = 1;  // push index
  localparam int unsigned TSR_TRW_OP_TICKET_WORD   = 2;
  localparam int unsigned TSR_TRW_OP_T_END_LO_WORD = 3;  // MTIME
  localparam int unsigned TSR_TRW_OP_T_END_HI_WORD = 4;
  localparam int unsigned TSR_TRW_OP_RUN_WORD      = 5;  // t_end - t_start (0 if not run), saturating
  localparam int unsigned TSR_TRW_OP_QWAIT_WORD    = 6;  // t_head - t_dispatch
  localparam int unsigned TSR_TRW_OP_HWAIT_WORD    = 7;  // t_start - t_head; t_end - t_head for non-run entries
  // TRW_MARK (DB §8.3): trace-writer MARK record
  localparam int unsigned TSR_TRW_MARK_NWORDS         = 8;
  localparam int unsigned TSR_TRW_MARK_HDR_WORD       = 0;
  localparam int unsigned TSR_TRW_MARK_NEXT_PIDX_WORD = 1;  // next push index
  localparam int unsigned TSR_TRW_MARK_VALUE_WORD     = 2;  // marker value
  localparam int unsigned TSR_TRW_MARK_T_LO_WORD      = 3;  // dispatch time
  localparam int unsigned TSR_TRW_MARK_T_HI_WORD      = 4;
  // TRW_STOP (DB §8.3): trace-writer STOP record
  localparam int unsigned TSR_TRW_STOP_NWORDS         = 8;
  localparam int unsigned TSR_TRW_STOP_HDR_WORD       = 0;
  localparam int unsigned TSR_TRW_STOP_T_WORD         = 1;  // threshold
  localparam int unsigned TSR_TRW_STOP_CAUSE_WORD     = 2;  // cause bits (layout DBG_CAUSE)
  localparam int unsigned TSR_TRW_STOP_T_LO_WORD      = 3;  // time
  localparam int unsigned TSR_TRW_STOP_T_HI_WORD      = 4;
  localparam int unsigned TSR_TRW_STOP_ACTIVATED_WORD = 5;  // 1 activated, 0 released
  // TRW_RVHALT (DB §8.3): trace-writer RVHALT record
  localparam int unsigned TSR_TRW_RVHALT_NWORDS      = 8;
  localparam int unsigned TSR_TRW_RVHALT_HDR_WORD    = 0;
  localparam int unsigned TSR_TRW_RVHALT_HALTED_WORD = 2;  // 1 halted, 0 resumed
  // TRW_DROP (DB §8.3): trace-writer DROP record
  localparam int unsigned TSR_TRW_DROP_NWORDS       = 8;
  localparam int unsigned TSR_TRW_DROP_HDR_WORD     = 0;
  localparam int unsigned TSR_TRW_DROP_DROPPED_WORD = 1;  // records dropped since the previous DROP record
  // DBG_INFO (DB §6.3): __tsr_dbg (section .tsr_dbg, 64 B)
  localparam int unsigned TSR_DBG_INFO_NWORDS            = 16;
  localparam int unsigned TSR_DBG_INFO_MAGIC_WORD        = 0;   // DBG_INFO_MAGIC
  localparam int unsigned TSR_DBG_INFO_VERSION_WORD      = 1;   // 1
  localparam int unsigned TSR_DBG_INFO_BUILD_ID_WORD     = 2;   // ELF note build id
  localparam int unsigned TSR_DBG_INFO_ATTACHED_WORD     = 3;   // written by the debugger
  localparam int unsigned TSR_DBG_INFO_LAUNCH_BREAK_WORD = 4;   // 0 off, 0xFFFFFFFF all, else kernel index + 1
  localparam int unsigned TSR_DBG_INFO_TRAP_COUNT_WORD   = 5;
  // CRASH (DB §6.3): __tsr_crash (section .tsr_crash, 192 B)
  localparam int unsigned TSR_CRASH_NWORDS        = 48;
  localparam int unsigned TSR_CRASH_MAGIC_WORD    = 0;   // CRASH_MAGIC
  localparam int unsigned TSR_CRASH_VERSION_WORD  = 1;   // 1
  localparam int unsigned TSR_CRASH_MCAUSE_WORD   = 2;
  localparam int unsigned TSR_CRASH_MEPC_WORD     = 3;
  localparam int unsigned TSR_CRASH_MTVAL_WORD    = 4;
  localparam int unsigned TSR_CRASH_MSTATUS_WORD  = 5;
  localparam int unsigned TSR_CRASH_MTIME_LO_WORD = 6;
  localparam int unsigned TSR_CRASH_MTIME_HI_WORD = 7;
  localparam int unsigned TSR_CRASH_SEQ_WORD      = 8;   // launch SEQ
  localparam int unsigned TSR_CRASH_KERNEL_WORD   = 9;   // kernel index
  localparam int unsigned TSR_CRASH_RT_STATE_WORD = 10;
  localparam int unsigned TSR_CRASH_RT_INFO_WORD  = 11;
  localparam int unsigned TSR_CRASH_X1_WORD       = 12;  // x1..x31 at words 12..42
  localparam int unsigned TSR_CRASH_X1_N          = 31;
  localparam int unsigned TSR_CRASH_NESTED_WORD   = 43;

  // ---------------------------------------------------------------------------------------------
  // Units (EX §7.3, CC §7.4, §9.4)
  // ---------------------------------------------------------------------------------------------
  localparam logic [3:0] TSR_UNIT_DMA           = 4'd0;           // EX §7.3, CC §7.4: LD, ST, KVST
  localparam logic [3:0] TSR_UNIT_QACT          = 4'd1;           // EX §7.3, CC §7.4: QACT
  localparam logic [3:0] TSR_UNIT_MVM           = 4'd2;           // EX §7.3, CC §7.4, §9.4: MVM, ATTQK, ATTPV
  localparam logic [3:0] TSR_UNIT_VU            = 4'd3;           // EX §7.3, CC §7.4, §9.4: VOP, RED, ROPE, TOPK, SOP, DQROW, ASMAX, AMASK, AMERGE, ANORM (VMASK from M9)
  localparam logic [3:0] TSR_UNIT_ATT           = 4'd4;           // EX §7.3, CC §7.4: reserved for a future AT attention engine; absent in M6 (was CMD)
  localparam logic [3:0] TSR_UNIT_SYNC          = 4'd5;           // EX §7.3, CC §7.4: NOP, FENCE, QCLEAR and every refused opcode
  localparam logic [3:0] TSR_UNIT_MARKER        = 4'd14;          // EX §7.3, CC §7.4: software marker records (REC / TRW); never a valid dependency
  localparam logic [3:0] TSR_UNIT_NONE          = 4'd15;          // EX §7.3, CC §7.4: no dependency
  localparam logic [3:0] TSR_UNIT_RSVD_FIRST    = 4'd6;           // EX §7.3: ids 6..13 reserved: a dependency on them is BAD_DEP
  localparam logic [3:0] TSR_UNIT_RSVD_LAST     = 4'd13;
  localparam int unsigned TSR_QDEPTH_U200_DMA   = 8;              // Q_DEPTH of the U200 M6 bitstream
  localparam int unsigned TSR_QDEPTH_U200_QACT  = 4;
  localparam int unsigned TSR_QDEPTH_U200_MVM   = 4;
  localparam int unsigned TSR_QDEPTH_U200_VU    = 16;
  localparam int unsigned TSR_QDEPTH_U200_ATT   = 0;
  localparam int unsigned TSR_QDEPTH_U200_SYNC  = 4;
  localparam int unsigned TSR_QDEPTH_PARAM_DMA  = 8;              // depth parameter when the unit is built
  localparam int unsigned TSR_QDEPTH_PARAM_QACT = 4;
  localparam int unsigned TSR_QDEPTH_PARAM_MVM  = 4;
  localparam int unsigned TSR_QDEPTH_PARAM_VU   = 16;
  localparam int unsigned TSR_QDEPTH_PARAM_ATT  = 8;
  localparam int unsigned TSR_QDEPTH_PARAM_SYNC = 4;
  localparam int unsigned TSR_ACCEPT_MAX_DMA    = 1;              // I3: ops accepted and not done (CC §9.4)
  localparam int unsigned TSR_ACCEPT_MAX_QACT   = 1;
  localparam int unsigned TSR_ACCEPT_MAX_MVM    = 2;
  localparam int unsigned TSR_ACCEPT_MAX_VU     = 2;
  localparam int unsigned TSR_ACCEPT_MAX_ATT    = 1;
  localparam int unsigned TSR_ACCEPT_MAX_SYNC   = 1;
  localparam logic [31:0] TSR_QDEPTH_U200_REG   = 32'h0808_1088;  // Q_DEPTH register value of the U200 M6 bitstream

  // ---------------------------------------------------------------------------------------------
  // Opcodes (EX §7.2); reserved and retired opcodes complete ILLEGAL_OP on SYNC
  // ---------------------------------------------------------------------------------------------
  localparam logic [7:0] TSR_OP_NOP    = 8'h00;  // kept, unit SYNC: EX §7.2: SYNC join point: completes when its dependencies are done
  localparam logic [7:0] TSR_OP_END    = 8'h01;  // reserved, unit SYNC: EX §7.2: END: ILLEGAL_OP
  localparam logic [7:0] TSR_OP_LI     = 8'h02;  // reserved, unit SYNC: EX §7.2: LI: ILLEGAL_OP
  localparam logic [7:0] TSR_OP_ADDI   = 8'h03;  // reserved, unit SYNC: EX §7.2: ADDI: ILLEGAL_OP
  localparam logic [7:0] TSR_OP_MULI   = 8'h04;  // retired, unit SYNC: EX §7.2: retired, never reused
  localparam logic [7:0] TSR_OP_RLD    = 8'h05;  // retired, unit SYNC: EX §7.2: retired, never reused
  localparam logic [7:0] TSR_OP_LOOP   = 8'h06;  // retired, unit SYNC: EX §7.2: retired, never reused
  localparam logic [7:0] TSR_OP_WAITM  = 8'h07;  // retired, unit SYNC: EX §7.2: retired, never reused
  localparam logic [7:0] TSR_OP_JUMP   = 8'h08;  // retired, unit SYNC: EX §7.2: retired, never reused
  localparam logic [7:0] TSR_OP_CALL   = 8'h09;  // retired, unit SYNC: EX §7.2: retired (never shipped), never reused
  localparam logic [7:0] TSR_OP_LD     = 8'h10;  // kept, unit DMA: EX §7.2: DRAM -> SPM, encoding unchanged
  localparam logic [7:0] TSR_OP_ST     = 8'h11;  // kept, unit DMA: EX §7.2: SPM -> DRAM, encoding unchanged
  localparam logic [7:0] TSR_OP_DQROW  = 8'h12;  // respec, unit VU, CAPS.DQROW: EX §7.2, VU §5.6: SPM -> f32 dequantisation; simulator from M6, RTL in M8
  localparam logic [7:0] TSR_OP_KVST   = 8'h14;  // new, unit DMA, CAPS.ATT: EX §7.2, AT §5.1: store K or V rows into a KV plane
  localparam logic [7:0] TSR_OP_QACT   = 8'h18;  // kept, unit QACT: EX §7.2: f32 -> QA, encoding unchanged
  localparam logic [7:0] TSR_OP_MVM    = 8'h20;  // kept, unit MVM: EX §7.2: matrix x activations, encoding unchanged
  localparam logic [7:0] TSR_OP_ATTQK  = 8'h21;  // new, unit MVM, CAPS.ATT: EX §7.2, AT §5.2: scores and row maxima over a KV group range
  localparam logic [7:0] TSR_OP_ATTPV  = 8'h22;  // new, unit MVM, CAPS.ATT: EX §7.2, AT §5.3: P.V channel partials and row sums
  localparam logic [7:0] TSR_OP_VOP    = 8'h30;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.2: element-wise vector op
  localparam logic [7:0] TSR_OP_RED    = 8'h31;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.3: segmented reduction with post-ops
  localparam logic [7:0] TSR_OP_ROPE   = 8'h32;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.4: rotary embedding over T tokens
  localparam logic [7:0] TSR_OP_TOPK   = 8'h33;  // respec, unit VU, CAPS.TOPK: EX §7.2, VU §5.5: top-k per row; simulator from M6, RTL in M8
  localparam logic [7:0] TSR_OP_ASMAX  = 8'h34;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.4: softmax numerator
  localparam logic [7:0] TSR_OP_AMASK  = 8'h35;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.5: general mask path
  localparam logic [7:0] TSR_OP_AMERGE = 8'h36;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.6: channel tree and online merge
  localparam logic [7:0] TSR_OP_ANORM  = 8'h37;  // new, unit VU, CAPS.ATT: EX §7.2, AT §5.7: final normalisation
  localparam logic [7:0] TSR_OP_VMASK  = 8'h38;  // new, unit VU, CAPS.VMASK: EX §7.2, VU §5.7: position-window mask; ILLEGAL_OP until M9
  localparam logic [7:0] TSR_OP_GDN    = 8'h39;  // new, unit VU, CAPS.GDN: EX §7.2, VU §7.12: fused Gated DeltaNet state update of one head
  localparam logic [7:0] TSR_OP_SOP    = 8'h40;  // respec, unit VU, CAPS.VU: EX §7.2, VU §5.8: scalar op on s0..s15
  localparam logic [7:0] TSR_OP_FENCE  = 8'h7F;  // kept, unit SYNC: EX §7.2: SYNC barrier: completes when every op pushed before it is done

  // Unit of a legal opcode; TSR_UNIT_SYNC for every reserved or retired opcode. A legal opcode
  // is refused the same way when (CAPS & tsr_op_caps) != tsr_op_caps or its unit is not built.
  function automatic logic [3:0] tsr_op_unit(input logic [7:0] opcode);
    case (opcode)
      8'h10, 8'h11, 8'h14: return TSR_UNIT_DMA;
      8'h12, 8'h30, 8'h31, 8'h32, 8'h33, 8'h34, 8'h35, 8'h36, 8'h37, 8'h38, 8'h39, 8'h40:
      return TSR_UNIT_VU;
      8'h18: return TSR_UNIT_QACT;
      8'h20, 8'h21, 8'h22: return TSR_UNIT_MVM;
      default: return TSR_UNIT_SYNC;
    endcase
  endfunction

  function automatic logic tsr_op_legal(input logic [7:0] opcode);
    case (opcode)
      8'h00, 8'h10, 8'h11, 8'h12, 8'h14, 8'h18, 8'h20, 8'h21, 8'h22, 8'h30, 8'h31, 8'h32, 8'h33,
      8'h34, 8'h35, 8'h36, 8'h37, 8'h38, 8'h39, 8'h40, 8'h7F: return 1'b1;
      default: return 1'b0;
    endcase
  endfunction

  function automatic logic [31:0] tsr_op_caps(input logic [7:0] opcode);
    case (opcode)
      8'h12: return TSR_CAPS_DQROW;
      8'h14, 8'h21, 8'h22, 8'h34, 8'h35, 8'h36, 8'h37: return TSR_CAPS_ATT;
      8'h30, 8'h31, 8'h32, 8'h40: return TSR_CAPS_VU;
      8'h33: return TSR_CAPS_TOPK;
      8'h38: return TSR_CAPS_VMASK;
      8'h39: return TSR_CAPS_GDN;
      default: return 32'h0;
    endcase
  endfunction

  // ---------------------------------------------------------------------------------------------
  // Error codes (EX §8.1)
  // ---------------------------------------------------------------------------------------------
  localparam logic [3:0] TSR_ERR_NONE        = 4'd0;
  localparam logic [3:0] TSR_ERR_ILLEGAL_OP  = 4'd1;   // reserved, retired or absent opcode; undefined sub, flags or function code; nonzero reserved field or bit, including x16 where the op does not define it; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_BAD_FMT     = 4'd2;   // unsupported weight format; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_QA_MISMATCH = 4'd3;   // (reserved) never raised
  localparam logic [3:0] TSR_ERR_SPM_RANGE   = 4'd4;   // SPM operand out of range (spec range-check rule); ERR_INFO = start address of the failing operand
  localparam logic [3:0] TSR_ERR_DRAM_RANGE  = 4'd5;   // DRAM operand out of range; ERR_INFO = start address
  localparam logic [3:0] TSR_ERR_K_RANGE     = 4'd6;   // K > K_MAX or not a multiple of the block size; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_T_RANGE     = 4'd7;   // T = 0 or T > T_MAX; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_ABORTED     = 4'd8;   // completed without running: a QABORT, or its unit was in error state; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_DEP_FAILED  = 4'd11;  // a dependency completed with an error; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_BAD_DEP     = 4'd12;  // a dependency on a ticket not pushed yet, on unit 6..14 or an absent unit, or below its unit's clear point; ERR_INFO = 0
  localparam logic [3:0] TSR_ERR_OPERAND     = 4'd13;  // an operand value outside the range its op defines, other than K, T and the address ranges (VU and AT ops only); ERR_INFO = the offending value
  // code 9 retired (LOOP_NEST), never raised, never reused
  // code 10 retired (WAIT_TIMEOUT), never raised, never reused
  localparam int unsigned TSR_ERR_CHECK_N = 8;  // dispatch check order: ILLEGAL_OP > BAD_DEP > BAD_FMT > K_RANGE > T_RANGE > OPERAND > SPM_RANGE > DRAM_RANGE

  localparam logic [3:0] TSR_ERR_CHECK_ORDER [8] = '{TSR_ERR_ILLEGAL_OP, TSR_ERR_BAD_DEP, TSR_ERR_BAD_FMT, TSR_ERR_K_RANGE, TSR_ERR_T_RANGE, TSR_ERR_OPERAND, TSR_ERR_SPM_RANGE, TSR_ERR_DRAM_RANGE};

  function automatic int unsigned tsr_err_check_rank(input logic [3:0] code);
    for (int unsigned i = 0; i < 8; i++)
      if (TSR_ERR_CHECK_ORDER[i] == code) return i;
    return 32'hFF;
  endfunction

  // ---------------------------------------------------------------------------------------------
  // PERF slots (EX §6): C = NUM_CH, X = 5 + 2C
  // ---------------------------------------------------------------------------------------------
  localparam int unsigned TSR_PERF_CYCLES     = 0;  // EX §6: time, core cycles
  localparam int unsigned TSR_PERF_RETIRED    = 1;  // EX §6: event, queue entries completed on every unit, any outcome
  localparam int unsigned TSR_PERF_MVM_CYCLES = 2;  // EX §6: time, the MVM unit holds an op
  localparam int unsigned TSR_PERF_VU_CYCLES  = 3;  // EX §6, VU §5.10: time, the VU holds an op
  // slot 4 reserved (was WAITM), reads 0 (EX §6)
  localparam int unsigned TSR_PERF_XOFF_DMA_CYCLES     = 0;   // EX §6: X + 0, time, the DMA unit holds an op
  localparam int unsigned TSR_PERF_XOFF_QACT_CYCLES    = 1;   // EX §6: X + 1, time, the QACT unit holds an op
  localparam int unsigned TSR_PERF_XOFF_ATT_CYCLES     = 2;   // EX §6: X + 2, time, unit 4 (AT) holds an op; reads 0 without it
  localparam int unsigned TSR_PERF_XOFF_MVM_WAIT       = 3;   // EX §6: X + 3, time, MVM idle while its queue holds an entry that cannot start
  localparam int unsigned TSR_PERF_XOFF_MVM_EMPTY      = 4;   // EX §6: X + 4, time, MVM idle and its queue empty (issue-bound)
  localparam int unsigned TSR_PERF_XOFF_HOLD_CYCLES    = 5;   // EX §6: X + 5, time, some queue head is held
  localparam int unsigned TSR_PERF_XOFF_PUSHES         = 6;   // EX §6: X + 6, event, entries pushed
  localparam int unsigned TSR_PERF_XOFF_PUSH_STALL     = 7;   // EX §6: X + 7, rv_time, clk_rv cycles a push waited for a credit
  localparam int unsigned TSR_PERF_XOFF_BRIDGE_ACC     = 8;   // EX §6: X + 8, event, SPM bridge accesses
  localparam int unsigned TSR_PERF_XOFF_BRIDGE_WAIT    = 9;   // EX §6: X + 9, rv_time, clk_rv cycles bridge accesses waited
  localparam int unsigned TSR_PERF_XOFF_REC_DROPPED    = 10;  // EX §6: X + 10, event, completion records dropped
  localparam int unsigned TSR_PERF_XOFF_LAUNCHES       = 11;  // EX §6: X + 11, event, answers posted
  localparam int unsigned TSR_PERF_XOFF_RV_WFI         = 12;  // EX §6: X + 12, time, the RISC-V sleeps in WFI
  localparam int unsigned TSR_PERF_XOFF_VU_SPM_WAIT    = 13;  // EX §6, VU §5.10: X + 13, time, a VU SPM request is raised and not granted
  localparam int unsigned TSR_PERF_XOFF_VU_HAZARD      = 14;  // EX §6, VU §5.10: X + 14, time, an accepted VU op waits for an earlier VU op
  localparam int unsigned TSR_PERF_XOFF_VU_LINES       = 15;  // EX §6, VU §5.10: X + 15, event, destination lines written by VU ops
  localparam int unsigned TSR_PERF_XOFF_ATT_QK_CYCLES  = 16;  // EX §6, AT §10: X + 16, time, the MVM unit holds an ATTQK
  localparam int unsigned TSR_PERF_XOFF_ATT_PV_CYCLES  = 17;  // EX §6, AT §10: X + 17, time, the MVM unit holds an ATTPV
  localparam int unsigned TSR_PERF_XOFF_ATT_PATH_STALL = 18;  // EX §6, AT §10: X + 18, time, a cluster in an ATT op waits for the gather or the p broadcast
  localparam int unsigned TSR_PERF_XOFF_KVST_RMW       = 19;  // EX §6, AT §10: X + 19, event, meta beats read-modify-written by KVST
  localparam int unsigned TSR_PERF_XOFF_SPM_RD_WAIT    = 20;  // EX §6, CC §6.3: X + 20, time, client-cycles a raised SPM read request waits for its bank port
  localparam int unsigned TSR_PERF_XOFF_SPM_WR_WAIT    = 21;  // EX §6, CC §6.3: X + 21, time, the same for write requests
  localparam int unsigned TSR_PERF_TRW_RECORDS         = 61;  // EX §6, DB §8.1: event, trace records written to DRAM
  localparam int unsigned TSR_PERF_TRW_DROPS           = 62;  // EX §6, DB §8.1: event, trace records dropped
  localparam int unsigned TSR_PERF_DBG_STOP_CYCLES     = 63;  // EX §6, DB §8.1: time_nohalt, ACTIVE and (RV_HALTED or STOP_ACTIVE); a measurement is valid only if 0

  function automatic int unsigned tsr_perf_x(input int unsigned num_ch);
    return 5 + 2 * num_ch;
  endfunction

  function automatic int unsigned tsr_perf_ext(input int unsigned num_ch, input int unsigned xoff);
    return tsr_perf_x(num_ch) + xoff;
  endfunction

  function automatic int unsigned tsr_perf_ch_rd_beats(input int unsigned c);
    return 5 + c;
  endfunction

  function automatic int unsigned tsr_perf_ch_wait(input int unsigned num_ch, input int unsigned c);
    return 5 + 1 * num_ch + c;
  endfunction

  // ---------------------------------------------------------------------------------------------
  // Instruction encodings of the base ops (spec ISA, EX §7.2)
  // ---------------------------------------------------------------------------------------------
  // NOP 0x00 (spec ISA, EX §7.1, owner spec)
  localparam logic [31:0] TSR_NOP_RSV_W0 = 32'hFFFF_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_NOP_RSV_W1 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W2 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W3 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W4 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W5 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W6 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_NOP_RSV_W7 = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_NOP_WORDS   = 8'd1;           // words holding a field (w0 always)
  // FENCE 0x7F (spec ISA, EX §7.1, owner spec)
  localparam logic [31:0] TSR_FENCE_RSV_W0 = 32'hFFFF_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_FENCE_RSV_W1 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W2 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W3 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W4 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W5 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W6 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_FENCE_RSV_W7 = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_FENCE_WORDS   = 8'd1;           // words holding a field (w0 always)
  // LD 0x10 (spec ISA, EX §7.2, owner spec)
  localparam logic [31:0] TSR_LD_CH         = 32'h0000_0F00;  // w0[11:8] channel
  localparam int unsigned TSR_LD_CH_LSB     = 8;
  localparam int unsigned TSR_LD_CH_W       = 4;
  localparam int unsigned TSR_LD_CH_WORD    = 0;
  localparam int unsigned TSR_LD_CH_BIT     = 8;
  localparam logic [31:0] TSR_LD_SPM        = 32'hFFFF_FFFF;  // w1 SPM destination line
  localparam int unsigned TSR_LD_SPM_LSB    = 0;
  localparam int unsigned TSR_LD_SPM_W      = 32;
  localparam int unsigned TSR_LD_SPM_WORD   = 1;
  localparam int unsigned TSR_LD_SPM_BIT    = 32;
  localparam logic [31:0] TSR_LD_DRAM       = 32'hFFFF_FFFF;  // w2 DRAM source beat
  localparam int unsigned TSR_LD_DRAM_LSB   = 0;
  localparam int unsigned TSR_LD_DRAM_W     = 32;
  localparam int unsigned TSR_LD_DRAM_WORD  = 2;
  localparam int unsigned TSR_LD_DRAM_BIT   = 64;
  localparam logic [31:0] TSR_LD_LINES      = 32'hFFFF_FFFF;  // w3 lines
  localparam int unsigned TSR_LD_LINES_LSB  = 0;
  localparam int unsigned TSR_LD_LINES_W    = 32;
  localparam int unsigned TSR_LD_LINES_WORD = 3;
  localparam int unsigned TSR_LD_LINES_BIT  = 96;
  localparam logic [31:0] TSR_LD_RSV_W0     = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_LD_RSV_W1     = 32'h0000_0000;
  localparam logic [31:0] TSR_LD_RSV_W2     = 32'h0000_0000;
  localparam logic [31:0] TSR_LD_RSV_W3     = 32'h0000_0000;
  localparam logic [31:0] TSR_LD_RSV_W4     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_LD_RSV_W5     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_LD_RSV_W6     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_LD_RSV_W7     = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_LD_WORDS       = 8'd15;          // words holding a field (w0 always)
  // ST 0x11 (spec ISA, EX §7.2, owner spec)
  localparam logic [31:0] TSR_ST_CH         = 32'h0000_0F00;  // w0[11:8] channel
  localparam int unsigned TSR_ST_CH_LSB     = 8;
  localparam int unsigned TSR_ST_CH_W       = 4;
  localparam int unsigned TSR_ST_CH_WORD    = 0;
  localparam int unsigned TSR_ST_CH_BIT     = 8;
  localparam logic [31:0] TSR_ST_SPM        = 32'hFFFF_FFFF;  // w1 SPM source line
  localparam int unsigned TSR_ST_SPM_LSB    = 0;
  localparam int unsigned TSR_ST_SPM_W      = 32;
  localparam int unsigned TSR_ST_SPM_WORD   = 1;
  localparam int unsigned TSR_ST_SPM_BIT    = 32;
  localparam logic [31:0] TSR_ST_DRAM       = 32'hFFFF_FFFF;  // w2 DRAM destination beat
  localparam int unsigned TSR_ST_DRAM_LSB   = 0;
  localparam int unsigned TSR_ST_DRAM_W     = 32;
  localparam int unsigned TSR_ST_DRAM_WORD  = 2;
  localparam int unsigned TSR_ST_DRAM_BIT   = 64;
  localparam logic [31:0] TSR_ST_LINES      = 32'hFFFF_FFFF;  // w3 lines
  localparam int unsigned TSR_ST_LINES_LSB  = 0;
  localparam int unsigned TSR_ST_LINES_W    = 32;
  localparam int unsigned TSR_ST_LINES_WORD = 3;
  localparam int unsigned TSR_ST_LINES_BIT  = 96;
  localparam logic [31:0] TSR_ST_RSV_W0     = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ST_RSV_W1     = 32'h0000_0000;
  localparam logic [31:0] TSR_ST_RSV_W2     = 32'h0000_0000;
  localparam logic [31:0] TSR_ST_RSV_W3     = 32'h0000_0000;
  localparam logic [31:0] TSR_ST_RSV_W4     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_ST_RSV_W5     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_ST_RSV_W6     = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_ST_RSV_W7     = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_ST_WORDS       = 8'd15;          // words holding a field (w0 always)
  // QACT 0x18 (spec ISA, EX §7.2, owner spec)
  localparam logic [31:0] TSR_QACT_KIND            = 32'h0000_0F00;  // w0[11:8] QA kind
  localparam int unsigned TSR_QACT_KIND_LSB        = 8;
  localparam int unsigned TSR_QACT_KIND_W          = 4;
  localparam int unsigned TSR_QACT_KIND_WORD       = 0;
  localparam int unsigned TSR_QACT_KIND_BIT        = 8;
  localparam logic [31:0] TSR_QACT_DST             = 32'hFFFF_FFFF;  // w1 QA destination line
  localparam int unsigned TSR_QACT_DST_LSB         = 0;
  localparam int unsigned TSR_QACT_DST_W           = 32;
  localparam int unsigned TSR_QACT_DST_WORD        = 1;
  localparam int unsigned TSR_QACT_DST_BIT         = 32;
  localparam logic [31:0] TSR_QACT_SRC             = 32'hFFFF_FFFF;  // w2 f32 source line
  localparam int unsigned TSR_QACT_SRC_LSB         = 0;
  localparam int unsigned TSR_QACT_SRC_W           = 32;
  localparam int unsigned TSR_QACT_SRC_WORD        = 2;
  localparam int unsigned TSR_QACT_SRC_BIT         = 64;
  localparam logic [31:0] TSR_QACT_K               = 32'hFFFF_FFFF;  // w3
  localparam int unsigned TSR_QACT_K_LSB           = 0;
  localparam int unsigned TSR_QACT_K_W             = 32;
  localparam int unsigned TSR_QACT_K_WORD          = 3;
  localparam int unsigned TSR_QACT_K_BIT           = 96;
  localparam logic [31:0] TSR_QACT_T               = 32'hFFFF_FFFF;  // w4
  localparam int unsigned TSR_QACT_T_LSB           = 0;
  localparam int unsigned TSR_QACT_T_W             = 32;
  localparam int unsigned TSR_QACT_T_WORD          = 4;
  localparam int unsigned TSR_QACT_T_BIT           = 128;
  localparam logic [31:0] TSR_QACT_SRC_STRIDE      = 32'hFFFF_FFFF;  // w5 lines; 0 = ceil(4K/64)
  localparam int unsigned TSR_QACT_SRC_STRIDE_LSB  = 0;
  localparam int unsigned TSR_QACT_SRC_STRIDE_W    = 32;
  localparam int unsigned TSR_QACT_SRC_STRIDE_WORD = 5;
  localparam int unsigned TSR_QACT_SRC_STRIDE_BIT  = 160;
  // QACT_KIND takes the QA_* values
  localparam logic [31:0] TSR_QACT_RSV_W0 = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_QACT_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_QACT_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_QACT_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_QACT_RSV_W4 = 32'h0000_0000;
  localparam logic [31:0] TSR_QACT_RSV_W5 = 32'h0000_0000;
  localparam logic [31:0] TSR_QACT_RSV_W6 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_QACT_RSV_W7 = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_QACT_WORDS   = 8'd63;          // words holding a field (w0 always)
  // MVM 0x20 (spec ISA, EX §7.2, owner spec)
  localparam logic [31:0] TSR_MVM_FMT                = 32'h0000_0F00;  // w0[11:8] weight format
  localparam int unsigned TSR_MVM_FMT_LSB            = 8;
  localparam int unsigned TSR_MVM_FMT_W              = 4;
  localparam int unsigned TSR_MVM_FMT_WORD           = 0;
  localparam int unsigned TSR_MVM_FMT_BIT            = 8;
  localparam logic [31:0] TSR_MVM_ACC                = 32'h0000_1000;  // w0[12] y += W.x
  localparam int unsigned TSR_MVM_ACC_LSB            = 12;
  localparam int unsigned TSR_MVM_ACC_W              = 1;
  localparam int unsigned TSR_MVM_ACC_WORD           = 0;
  localparam int unsigned TSR_MVM_ACC_BIT            = 12;
  localparam logic [31:0] TSR_MVM_LOCAL_SCALE        = 32'h0000_2000;  // w0[13] scale completed FP32 rows before export
  localparam int unsigned TSR_MVM_LOCAL_SCALE_LSB    = 13;
  localparam int unsigned TSR_MVM_LOCAL_SCALE_W      = 1;
  localparam int unsigned TSR_MVM_LOCAL_SCALE_WORD   = 0;
  localparam int unsigned TSR_MVM_LOCAL_SCALE_BIT    = 13;
  localparam logic [31:0] TSR_MVM_LOCAL_SOFTCAP      = 32'h0000_4000;  // w0[14] scale, TANH, scale before export; dense destination stride
  localparam int unsigned TSR_MVM_LOCAL_SOFTCAP_LSB  = 14;
  localparam int unsigned TSR_MVM_LOCAL_SOFTCAP_W    = 1;
  localparam int unsigned TSR_MVM_LOCAL_SOFTCAP_WORD = 0;
  localparam int unsigned TSR_MVM_LOCAL_SOFTCAP_BIT  = 14;
  localparam logic [31:0] TSR_MVM_DST                = 32'hFFFF_FFFF;  // w1 destination line
  localparam int unsigned TSR_MVM_DST_LSB            = 0;
  localparam int unsigned TSR_MVM_DST_W              = 32;
  localparam int unsigned TSR_MVM_DST_WORD           = 1;
  localparam int unsigned TSR_MVM_DST_BIT            = 32;
  localparam logic [31:0] TSR_MVM_X                  = 32'hFFFF_FFFF;  // w2 QA activation line
  localparam int unsigned TSR_MVM_X_LSB              = 0;
  localparam int unsigned TSR_MVM_X_W                = 32;
  localparam int unsigned TSR_MVM_X_WORD             = 2;
  localparam int unsigned TSR_MVM_X_BIT              = 64;
  localparam logic [31:0] TSR_MVM_W_OFF              = 32'hFFFF_FFFF;  // w3 weight offset (beats)
  localparam int unsigned TSR_MVM_W_OFF_LSB          = 0;
  localparam int unsigned TSR_MVM_W_OFF_W            = 32;
  localparam int unsigned TSR_MVM_W_OFF_WORD         = 3;
  localparam int unsigned TSR_MVM_W_OFF_BIT          = 96;
  localparam logic [31:0] TSR_MVM_N                  = 32'hFFFF_FFFF;  // w4 rows
  localparam int unsigned TSR_MVM_N_LSB              = 0;
  localparam int unsigned TSR_MVM_N_W                = 32;
  localparam int unsigned TSR_MVM_N_WORD             = 4;
  localparam int unsigned TSR_MVM_N_BIT              = 128;
  localparam logic [31:0] TSR_MVM_K                  = 32'h0000_FFFF;  // w5[15:0]
  localparam int unsigned TSR_MVM_K_LSB              = 0;
  localparam int unsigned TSR_MVM_K_W                = 16;
  localparam int unsigned TSR_MVM_K_WORD             = 5;
  localparam int unsigned TSR_MVM_K_BIT              = 160;
  localparam logic [31:0] TSR_MVM_T                  = 32'h000F_0000;  // w5[19:16]
  localparam int unsigned TSR_MVM_T_LSB              = 16;
  localparam int unsigned TSR_MVM_T_W                = 4;
  localparam int unsigned TSR_MVM_T_WORD             = 5;
  localparam int unsigned TSR_MVM_T_BIT              = 176;
  localparam logic [31:0] TSR_MVM_DST_STRIDE         = 32'hFFFF_FFFF;  // w6 lines; 0 = ceil(4N/64); with LOCAL_SOFTCAP, first FP32 coefficient
  localparam int unsigned TSR_MVM_DST_STRIDE_LSB     = 0;
  localparam int unsigned TSR_MVM_DST_STRIDE_W       = 32;
  localparam int unsigned TSR_MVM_DST_STRIDE_WORD    = 6;
  localparam int unsigned TSR_MVM_DST_STRIDE_BIT     = 192;
  localparam logic [31:0] TSR_MVM_SCALE              = 32'hFFFF_FFFF;  // w7 FP32 coefficient with LOCAL_SCALE; final FP32 coefficient with LOCAL_SOFTCAP
  localparam int unsigned TSR_MVM_SCALE_LSB          = 0;
  localparam int unsigned TSR_MVM_SCALE_W            = 32;
  localparam int unsigned TSR_MVM_SCALE_WORD         = 7;
  localparam int unsigned TSR_MVM_SCALE_BIT          = 224;
  // MVM_FMT takes the FMT_* values
  localparam logic [31:0] TSR_MVM_RSV_W0 = 32'hFFFF_8000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_MVM_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_MVM_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_MVM_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_MVM_RSV_W4 = 32'h0000_0000;
  localparam logic [31:0] TSR_MVM_RSV_W5 = 32'hFFF0_0000;
  localparam logic [31:0] TSR_MVM_RSV_W6 = 32'h0000_0000;
  localparam logic [31:0] TSR_MVM_RSV_W7 = 32'h0000_0000;
  localparam logic [7:0] TSR_MVM_WORDS   = 8'd255;         // words holding a field (w0 always)

  // ---------------------------------------------------------------------------------------------
  // VU encodings (VU §5)
  // ---------------------------------------------------------------------------------------------
  // VOP 0x30 (VU §5.2, owner VU)
  localparam logic [31:0] TSR_VOP_CMODE         = 32'h0000_0F00;  // w0[11:8] per-row scalar c
  localparam int unsigned TSR_VOP_CMODE_LSB     = 8;
  localparam int unsigned TSR_VOP_CMODE_W       = 4;
  localparam int unsigned TSR_VOP_CMODE_WORD    = 0;
  localparam int unsigned TSR_VOP_CMODE_BIT     = 8;
  localparam logic [31:0] TSR_VOP_FUNC          = 32'h00FF_0000;  // x16[7:0] function (VOP_FUNCS)
  localparam int unsigned TSR_VOP_FUNC_LSB      = 16;
  localparam int unsigned TSR_VOP_FUNC_W        = 8;
  localparam int unsigned TSR_VOP_FUNC_WORD     = 0;
  localparam int unsigned TSR_VOP_FUNC_BIT      = 16;
  localparam logic [31:0] TSR_VOP_SI            = 32'h0F00_0000;  // x16[11:8] s register for CMODE SREG, else 0
  localparam int unsigned TSR_VOP_SI_LSB        = 24;
  localparam int unsigned TSR_VOP_SI_W          = 4;
  localparam int unsigned TSR_VOP_SI_WORD       = 0;
  localparam int unsigned TSR_VOP_SI_BIT        = 24;
  localparam logic [31:0] TSR_VOP_D             = 32'h00FF_FFFF;  // w1[23:0] destination line
  localparam int unsigned TSR_VOP_D_LSB         = 0;
  localparam int unsigned TSR_VOP_D_W           = 24;
  localparam int unsigned TSR_VOP_D_WORD        = 1;
  localparam int unsigned TSR_VOP_D_BIT         = 32;
  localparam logic [31:0] TSR_VOP_A             = 32'h00FF_FFFF;  // w2[23:0] source a line (0 if the function reads no a)
  localparam int unsigned TSR_VOP_A_LSB         = 0;
  localparam int unsigned TSR_VOP_A_W           = 24;
  localparam int unsigned TSR_VOP_A_WORD        = 2;
  localparam int unsigned TSR_VOP_A_BIT         = 64;
  localparam logic [31:0] TSR_VOP_B             = 32'h00FF_FFFF;  // w3[23:0] source b line (0 if unused)
  localparam int unsigned TSR_VOP_B_LSB         = 0;
  localparam int unsigned TSR_VOP_B_W           = 24;
  localparam int unsigned TSR_VOP_B_WORD        = 3;
  localparam int unsigned TSR_VOP_B_BIT         = 96;
  localparam logic [31:0] TSR_VOP_C             = 32'h00FF_FFFF;  // w4[23:0] row-vector line for CMODE VEC, else 0
  localparam int unsigned TSR_VOP_C_LSB         = 0;
  localparam int unsigned TSR_VOP_C_W           = 24;
  localparam int unsigned TSR_VOP_C_WORD        = 4;
  localparam int unsigned TSR_VOP_C_BIT         = 128;
  localparam logic [31:0] TSR_VOP_M             = 32'h00FF_FFFF;  // w5[23:0] elements per row
  localparam int unsigned TSR_VOP_M_LSB         = 0;
  localparam int unsigned TSR_VOP_M_W           = 24;
  localparam int unsigned TSR_VOP_M_WORD        = 5;
  localparam int unsigned TSR_VOP_M_BIT         = 160;
  localparam logic [31:0] TSR_VOP_ROWS          = 32'h0000_FFFF;  // w6[15:0] rows
  localparam int unsigned TSR_VOP_ROWS_LSB      = 0;
  localparam int unsigned TSR_VOP_ROWS_W        = 16;
  localparam int unsigned TSR_VOP_ROWS_WORD     = 6;
  localparam int unsigned TSR_VOP_ROWS_BIT      = 192;
  localparam logic [31:0] TSR_VOP_D_STRIDE      = 32'hFFFF_0000;  // w6[31:16] lines
  localparam int unsigned TSR_VOP_D_STRIDE_LSB  = 16;
  localparam int unsigned TSR_VOP_D_STRIDE_W    = 16;
  localparam int unsigned TSR_VOP_D_STRIDE_WORD = 6;
  localparam int unsigned TSR_VOP_D_STRIDE_BIT  = 208;
  localparam logic [31:0] TSR_VOP_A_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines (0 if a is unused)
  localparam int unsigned TSR_VOP_A_STRIDE_LSB  = 0;
  localparam int unsigned TSR_VOP_A_STRIDE_W    = 16;
  localparam int unsigned TSR_VOP_A_STRIDE_WORD = 7;
  localparam int unsigned TSR_VOP_A_STRIDE_BIT  = 224;
  localparam logic [31:0] TSR_VOP_B_STRIDE      = 32'hFFFF_0000;  // w7[31:16] lines (0 if b is unused)
  localparam int unsigned TSR_VOP_B_STRIDE_LSB  = 16;
  localparam int unsigned TSR_VOP_B_STRIDE_W    = 16;
  localparam int unsigned TSR_VOP_B_STRIDE_WORD = 7;
  localparam int unsigned TSR_VOP_B_STRIDE_BIT  = 240;
  localparam logic [3:0] TSR_VOP_CMODE_NONE     = 4'd0;           // no c
  localparam logic [3:0] TSR_VOP_CMODE_SREG     = 4'd1;           // c = s[SI]
  localparam logic [3:0] TSR_VOP_CMODE_VEC      = 4'd2;           // c_r = word r of the vector at line C
  localparam logic [7:0] TSR_VOP_FUNC_ADD       = 8'd0;           // reads ab, rate F: FA(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_SUB       = 8'd1;           // reads ab, rate F: FS(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_MUL       = 8'd2;           // reads ab, rate F: FM(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_MAX       = 8'd3;           // reads ab, rate F: MAXT(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_MULC      = 8'd4;           // reads ac, rate F: FM(a, c)
  localparam logic [7:0] TSR_VOP_FUNC_ADDC      = 8'd5;           // reads ac, rate F: FA(a, c)
  localparam logic [7:0] TSR_VOP_FUNC_SILU      = 8'd6;           // reads a, rate H: SILU(a)
  localparam logic [7:0] TSR_VOP_FUNC_SWIGLU    = 8'd7;           // reads ab, rate H: FM(SILU(a), b)
  localparam logic [7:0] TSR_VOP_FUNC_EXP       = 8'd8;           // reads a, rate F: EXPK(a, P1)
  localparam logic [7:0] TSR_VOP_FUNC_COPY      = 8'd9;           // reads a, rate F: a (raw)
  localparam logic [7:0] TSR_VOP_FUNC_MIN       = 8'd10;          // reads ab, rate F: MINT(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_SUBC      = 8'd11;          // reads ac, rate F: FS(a, c)
  localparam logic [7:0] TSR_VOP_FUNC_MULCB     = 8'd12;          // reads abc, rate F: FM(FM(a, c), b)
  localparam logic [7:0] TSR_VOP_FUNC_MULCADD   = 8'd13;          // reads abc, rate F: FA(FM(a, c), b)
  localparam logic [7:0] TSR_VOP_FUNC_ADDMULC   = 8'd14;          // reads abc, rate F: FM(FA(a, b), c)
  localparam logic [7:0] TSR_VOP_FUNC_EXPSUB    = 8'd15;          // reads ab, rate F: EXPSUB(a, b)
  localparam logic [7:0] TSR_VOP_FUNC_EXPSUBC   = 8'd16;          // reads ac, rate F: EXPSUB(a, c)
  localparam logic [7:0] TSR_VOP_FUNC_SIGMOID   = 8'd17;          // reads a, rate H: SIG(a)
  localparam logic [7:0] TSR_VOP_FUNC_GELU      = 8'd18;          // reads a, rate H: GELU(a), exact tanh form
  localparam logic [7:0] TSR_VOP_FUNC_GEGLU     = 8'd19;          // reads ab, rate H: FM(GELU(a), b)
  localparam logic [7:0] TSR_VOP_FUNC_TANH      = 8'd20;          // reads a, rate H: TANH(a)
  localparam logic [7:0] TSR_VOP_FUNC_RECIP     = 8'd21;          // reads a, rate H: RC(a), correctly rounded
  localparam logic [7:0] TSR_VOP_FUNC_SETC      = 8'd22;          // reads c, rate F: c (raw)
  localparam logic [7:0] TSR_VOP_FUNC_GELU16    = 8'd23;          // reads a, rate H: ggml-CPU fp16 form (VU §6.5)
  localparam logic [7:0] TSR_VOP_FUNC_GEGLU16   = 8'd24;          // reads ab, rate H: GEGLU16(a, b) (VU §6.5)
  localparam logic [31:0] TSR_VOP_RSV_W0        = 32'hF000_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_VOP_RSV_W1        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VOP_RSV_W2        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VOP_RSV_W3        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VOP_RSV_W4        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VOP_RSV_W5        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VOP_RSV_W6        = 32'h0000_0000;
  localparam logic [31:0] TSR_VOP_RSV_W7        = 32'h0000_0000;
  localparam logic [7:0] TSR_VOP_WORDS          = 8'd255;         // words holding a field (w0 always)
  // RED 0x31 (VU §5.3, owner VU)
  localparam logic [31:0] TSR_RED_KIND          = 32'h0000_0F00;  // w0[11:8]
  localparam int unsigned TSR_RED_KIND_LSB      = 8;
  localparam int unsigned TSR_RED_KIND_W        = 4;
  localparam int unsigned TSR_RED_KIND_WORD     = 0;
  localparam int unsigned TSR_RED_KIND_BIT      = 8;
  localparam logic [31:0] TSR_RED_TO_S          = 32'h0000_1000;  // w0[12] 1: result to s[SD] (ROWS must be 1); 0: per-row vector at D
  localparam int unsigned TSR_RED_TO_S_LSB      = 12;
  localparam int unsigned TSR_RED_TO_S_W        = 1;
  localparam int unsigned TSR_RED_TO_S_WORD     = 0;
  localparam int unsigned TSR_RED_TO_S_BIT      = 12;
  localparam logic [31:0] TSR_RED_SD            = 32'h000F_0000;  // x16[3:0] destination s register if TO_S, else 0
  localparam int unsigned TSR_RED_SD_LSB        = 16;
  localparam int unsigned TSR_RED_SD_W          = 4;
  localparam int unsigned TSR_RED_SD_WORD       = 0;
  localparam int unsigned TSR_RED_SD_BIT        = 16;
  localparam logic [31:0] TSR_RED_POST          = 32'h0030_0000;  // x16[5:4] post-op; MAX requires NONE
  localparam int unsigned TSR_RED_POST_LSB      = 20;
  localparam int unsigned TSR_RED_POST_W        = 2;
  localparam int unsigned TSR_RED_POST_WORD     = 0;
  localparam int unsigned TSR_RED_POST_BIT      = 20;
  localparam logic [31:0] TSR_RED_SK            = 32'h0F00_0000;  // x16[11:8] RMS scale register (typically 1/n), else 0
  localparam int unsigned TSR_RED_SK_LSB        = 24;
  localparam int unsigned TSR_RED_SK_W          = 4;
  localparam int unsigned TSR_RED_SK_WORD       = 0;
  localparam int unsigned TSR_RED_SK_BIT        = 24;
  localparam logic [31:0] TSR_RED_SE            = 32'hF000_0000;  // x16[15:12] RMS epsilon register, else 0
  localparam int unsigned TSR_RED_SE_LSB        = 28;
  localparam int unsigned TSR_RED_SE_W          = 4;
  localparam int unsigned TSR_RED_SE_WORD       = 0;
  localparam int unsigned TSR_RED_SE_BIT        = 28;
  localparam logic [31:0] TSR_RED_D             = 32'h00FF_FFFF;  // w1[23:0] result vector line (rows words); 0 if TO_S
  localparam int unsigned TSR_RED_D_LSB         = 0;
  localparam int unsigned TSR_RED_D_W           = 24;
  localparam int unsigned TSR_RED_D_WORD        = 1;
  localparam int unsigned TSR_RED_D_BIT         = 32;
  localparam logic [31:0] TSR_RED_A             = 32'h00FF_FFFF;  // w2[23:0] source line
  localparam int unsigned TSR_RED_A_LSB         = 0;
  localparam int unsigned TSR_RED_A_W           = 24;
  localparam int unsigned TSR_RED_A_WORD        = 2;
  localparam int unsigned TSR_RED_A_BIT         = 64;
  localparam logic [31:0] TSR_RED_M             = 32'h00FF_FFFF;  // w5[23:0] elements per row
  localparam int unsigned TSR_RED_M_LSB         = 0;
  localparam int unsigned TSR_RED_M_W           = 24;
  localparam int unsigned TSR_RED_M_WORD        = 5;
  localparam int unsigned TSR_RED_M_BIT         = 160;
  localparam logic [31:0] TSR_RED_ROWS          = 32'h0000_FFFF;  // w6[15:0] rows
  localparam int unsigned TSR_RED_ROWS_LSB      = 0;
  localparam int unsigned TSR_RED_ROWS_W        = 16;
  localparam int unsigned TSR_RED_ROWS_WORD     = 6;
  localparam int unsigned TSR_RED_ROWS_BIT      = 192;
  localparam logic [31:0] TSR_RED_A_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines
  localparam int unsigned TSR_RED_A_STRIDE_LSB  = 0;
  localparam int unsigned TSR_RED_A_STRIDE_W    = 16;
  localparam int unsigned TSR_RED_A_STRIDE_WORD = 7;
  localparam int unsigned TSR_RED_A_STRIDE_BIT  = 224;
  localparam logic [3:0] TSR_RED_KIND_SUM       = 4'd0;
  localparam logic [3:0] TSR_RED_KIND_SUMSQ     = 4'd1;
  localparam logic [3:0] TSR_RED_KIND_MAX       = 4'd2;
  localparam logic [1:0] TSR_RED_POST_NONE      = 2'd0;
  localparam logic [1:0] TSR_RED_POST_RCP       = 2'd1;           // reciprocal
  localparam logic [1:0] TSR_RED_POST_RMS       = 2'd2;           // RC(SQ(FA(FM(y, s[SK]), s[SE]))) (VU §6.2)
  localparam logic [31:0] TSR_RED_RSV_W0        = 32'h00C0_E000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_RED_RSV_W1        = 32'hFF00_0000;
  localparam logic [31:0] TSR_RED_RSV_W2        = 32'hFF00_0000;
  localparam logic [31:0] TSR_RED_RSV_W3        = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_RED_RSV_W4        = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_RED_RSV_W5        = 32'hFF00_0000;
  localparam logic [31:0] TSR_RED_RSV_W6        = 32'hFFFF_0000;
  localparam logic [31:0] TSR_RED_RSV_W7        = 32'hFFFF_0000;
  localparam logic [7:0] TSR_RED_WORDS          = 8'd231;         // words holding a field (w0 always)
  // ROPE 0x32 (VU §5.4, owner VU)
  localparam logic [31:0] TSR_ROPE_MODE            = 32'h0000_0F00;  // w0[11:8]
  localparam int unsigned TSR_ROPE_MODE_LSB        = 8;
  localparam int unsigned TSR_ROPE_MODE_W          = 4;
  localparam int unsigned TSR_ROPE_MODE_WORD       = 0;
  localparam int unsigned TSR_ROPE_MODE_BIT        = 8;
  localparam logic [31:0] TSR_ROPE_FF              = 32'h0000_1000;  // w0[12] recurrence form: the table holds freq factors
  localparam int unsigned TSR_ROPE_FF_LSB          = 12;
  localparam int unsigned TSR_ROPE_FF_W            = 1;
  localparam int unsigned TSR_ROPE_FF_WORD         = 0;
  localparam int unsigned TSR_ROPE_FF_BIT          = 12;
  localparam logic [31:0] TSR_ROPE_FS              = 32'h0000_2000;  // w0[13] recurrence form: multiply by freq_scale in s[SFS]
  localparam int unsigned TSR_ROPE_FS_LSB          = 13;
  localparam int unsigned TSR_ROPE_FS_W            = 1;
  localparam int unsigned TSR_ROPE_FS_WORD         = 0;
  localparam int unsigned TSR_ROPE_FS_BIT          = 13;
  localparam logic [31:0] TSR_ROPE_ATAB            = 32'h0000_4000;  // w0[14] A-table form (YaRN): FF, FS and STS must be 0
  localparam int unsigned TSR_ROPE_ATAB_LSB        = 14;
  localparam int unsigned TSR_ROPE_ATAB_W          = 1;
  localparam int unsigned TSR_ROPE_ATAB_WORD       = 0;
  localparam int unsigned TSR_ROPE_ATAB_BIT        = 14;
  localparam logic [31:0] TSR_ROPE_SM              = 32'h000F_0000;  // x16[3:0] s register holding M
  localparam int unsigned TSR_ROPE_SM_LSB          = 16;
  localparam int unsigned TSR_ROPE_SM_W            = 4;
  localparam int unsigned TSR_ROPE_SM_WORD         = 0;
  localparam int unsigned TSR_ROPE_SM_BIT          = 16;
  localparam logic [31:0] TSR_ROPE_STS             = 32'h00F0_0000;  // x16[7:4] s register holding TS (recurrence form; 0 with ATAB)
  localparam int unsigned TSR_ROPE_STS_LSB         = 20;
  localparam int unsigned TSR_ROPE_STS_W           = 4;
  localparam int unsigned TSR_ROPE_STS_WORD        = 0;
  localparam int unsigned TSR_ROPE_STS_BIT         = 20;
  localparam logic [31:0] TSR_ROPE_SFS             = 32'h0F00_0000;  // x16[11:8] s register holding freq_scale when FS, else 0
  localparam int unsigned TSR_ROPE_SFS_LSB         = 24;
  localparam int unsigned TSR_ROPE_SFS_W           = 4;
  localparam int unsigned TSR_ROPE_SFS_WORD        = 0;
  localparam int unsigned TSR_ROPE_SFS_BIT         = 24;
  localparam logic [31:0] TSR_ROPE_D               = 32'h00FF_FFFF;  // w1[23:0] destination line of token 0
  localparam int unsigned TSR_ROPE_D_LSB           = 0;
  localparam int unsigned TSR_ROPE_D_W             = 24;
  localparam int unsigned TSR_ROPE_D_WORD          = 1;
  localparam int unsigned TSR_ROPE_D_BIT           = 32;
  localparam logic [31:0] TSR_ROPE_SRC             = 32'h00FF_FFFF;  // w2[23:0] source line of token 0
  localparam int unsigned TSR_ROPE_SRC_LSB         = 0;
  localparam int unsigned TSR_ROPE_SRC_W           = 24;
  localparam int unsigned TSR_ROPE_SRC_WORD        = 2;
  localparam int unsigned TSR_ROPE_SRC_BIT         = 64;
  localparam logic [31:0] TSR_ROPE_POS             = 32'h00FF_FFFF;  // w3[23:0] line of the I32 position vector
  localparam int unsigned TSR_ROPE_POS_LSB         = 0;
  localparam int unsigned TSR_ROPE_POS_W           = 24;
  localparam int unsigned TSR_ROPE_POS_WORD        = 3;
  localparam int unsigned TSR_ROPE_POS_BIT         = 96;
  localparam logic [31:0] TSR_ROPE_TABLE           = 32'h00FF_FFFF;  // w4[23:0] line of the f32 table of n_dims/2 entries; 0 when neither FF nor ATAB
  localparam int unsigned TSR_ROPE_TABLE_LSB       = 0;
  localparam int unsigned TSR_ROPE_TABLE_W         = 24;
  localparam int unsigned TSR_ROPE_TABLE_WORD      = 4;
  localparam int unsigned TSR_ROPE_TABLE_BIT       = 128;
  localparam logic [31:0] TSR_ROPE_HEAD_DIM        = 32'h0000_FFFF;  // w5[15:0] elements per head
  localparam int unsigned TSR_ROPE_HEAD_DIM_LSB    = 0;
  localparam int unsigned TSR_ROPE_HEAD_DIM_W      = 16;
  localparam int unsigned TSR_ROPE_HEAD_DIM_WORD   = 5;
  localparam int unsigned TSR_ROPE_HEAD_DIM_BIT    = 160;
  localparam logic [31:0] TSR_ROPE_N_DIMS          = 32'hFFFF_0000;  // w5[31:16] rotated elements per head
  localparam int unsigned TSR_ROPE_N_DIMS_LSB      = 16;
  localparam int unsigned TSR_ROPE_N_DIMS_W        = 16;
  localparam int unsigned TSR_ROPE_N_DIMS_WORD     = 5;
  localparam int unsigned TSR_ROPE_N_DIMS_BIT      = 176;
  localparam logic [31:0] TSR_ROPE_N_HEADS         = 32'h0000_FFFF;  // w6[15:0] heads per token
  localparam int unsigned TSR_ROPE_N_HEADS_LSB     = 0;
  localparam int unsigned TSR_ROPE_N_HEADS_W       = 16;
  localparam int unsigned TSR_ROPE_N_HEADS_WORD    = 6;
  localparam int unsigned TSR_ROPE_N_HEADS_BIT     = 192;
  localparam logic [31:0] TSR_ROPE_T               = 32'hFFFF_0000;  // w6[31:16] tokens
  localparam int unsigned TSR_ROPE_T_LSB           = 16;
  localparam int unsigned TSR_ROPE_T_W             = 16;
  localparam int unsigned TSR_ROPE_T_WORD          = 6;
  localparam int unsigned TSR_ROPE_T_BIT           = 208;
  localparam logic [31:0] TSR_ROPE_SRC_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines between tokens; 0 = H
  localparam int unsigned TSR_ROPE_SRC_STRIDE_LSB  = 0;
  localparam int unsigned TSR_ROPE_SRC_STRIDE_W    = 16;
  localparam int unsigned TSR_ROPE_SRC_STRIDE_WORD = 7;
  localparam int unsigned TSR_ROPE_SRC_STRIDE_BIT  = 224;
  localparam logic [31:0] TSR_ROPE_D_STRIDE        = 32'hFFFF_0000;  // w7[31:16] lines between tokens; 0 = H
  localparam int unsigned TSR_ROPE_D_STRIDE_LSB    = 16;
  localparam int unsigned TSR_ROPE_D_STRIDE_W      = 16;
  localparam int unsigned TSR_ROPE_D_STRIDE_WORD   = 7;
  localparam int unsigned TSR_ROPE_D_STRIDE_BIT    = 240;
  localparam logic [3:0] TSR_ROPE_MODE_NORM        = 4'd0;           // adjacent pairs
  localparam logic [3:0] TSR_ROPE_MODE_NEOX        = 4'd1;           // half-split pairs
  localparam logic [31:0] TSR_ROPE_RSV_W0          = 32'hF000_8000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ROPE_RSV_W1          = 32'hFF00_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W2          = 32'hFF00_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W3          = 32'hFF00_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W4          = 32'hFF00_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W5          = 32'h0000_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W6          = 32'h0000_0000;
  localparam logic [31:0] TSR_ROPE_RSV_W7          = 32'h0000_0000;
  localparam logic [7:0] TSR_ROPE_WORDS            = 8'd255;         // words holding a field (w0 always)
  // TOPK 0x33 (VU §5.5, owner VU)
  localparam logic [31:0] TSR_TOPK_K             = 32'h001F_0000;  // x16[4:0] 1..16
  localparam int unsigned TSR_TOPK_K_LSB         = 16;
  localparam int unsigned TSR_TOPK_K_W           = 5;
  localparam int unsigned TSR_TOPK_K_WORD        = 0;
  localparam int unsigned TSR_TOPK_K_BIT         = 16;
  localparam logic [31:0] TSR_TOPK_D             = 32'h00FF_FFFF;  // w1[23:0] output line of row 0: k (index, value) pairs
  localparam int unsigned TSR_TOPK_D_LSB         = 0;
  localparam int unsigned TSR_TOPK_D_W           = 24;
  localparam int unsigned TSR_TOPK_D_WORD        = 1;
  localparam int unsigned TSR_TOPK_D_BIT         = 32;
  localparam logic [31:0] TSR_TOPK_A             = 32'h00FF_FFFF;  // w2[23:0] source line
  localparam int unsigned TSR_TOPK_A_LSB         = 0;
  localparam int unsigned TSR_TOPK_A_W           = 24;
  localparam int unsigned TSR_TOPK_A_WORD        = 2;
  localparam int unsigned TSR_TOPK_A_BIT         = 64;
  localparam logic [31:0] TSR_TOPK_M             = 32'h00FF_FFFF;  // w5[23:0] elements per row
  localparam int unsigned TSR_TOPK_M_LSB         = 0;
  localparam int unsigned TSR_TOPK_M_W           = 24;
  localparam int unsigned TSR_TOPK_M_WORD        = 5;
  localparam int unsigned TSR_TOPK_M_BIT         = 160;
  localparam logic [31:0] TSR_TOPK_ROWS          = 32'h0000_FFFF;  // w6[15:0] rows
  localparam int unsigned TSR_TOPK_ROWS_LSB      = 0;
  localparam int unsigned TSR_TOPK_ROWS_W        = 16;
  localparam int unsigned TSR_TOPK_ROWS_WORD     = 6;
  localparam int unsigned TSR_TOPK_ROWS_BIT      = 192;
  localparam logic [31:0] TSR_TOPK_D_STRIDE      = 32'hFFFF_0000;  // w6[31:16] lines; 0 = ceil(k/8)
  localparam int unsigned TSR_TOPK_D_STRIDE_LSB  = 16;
  localparam int unsigned TSR_TOPK_D_STRIDE_W    = 16;
  localparam int unsigned TSR_TOPK_D_STRIDE_WORD = 6;
  localparam int unsigned TSR_TOPK_D_STRIDE_BIT  = 208;
  localparam logic [31:0] TSR_TOPK_A_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines
  localparam int unsigned TSR_TOPK_A_STRIDE_LSB  = 0;
  localparam int unsigned TSR_TOPK_A_STRIDE_W    = 16;
  localparam int unsigned TSR_TOPK_A_STRIDE_WORD = 7;
  localparam int unsigned TSR_TOPK_A_STRIDE_BIT  = 224;
  localparam logic [31:0] TSR_TOPK_RSV_W0        = 32'hFFE0_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_TOPK_RSV_W1        = 32'hFF00_0000;
  localparam logic [31:0] TSR_TOPK_RSV_W2        = 32'hFF00_0000;
  localparam logic [31:0] TSR_TOPK_RSV_W3        = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_TOPK_RSV_W4        = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_TOPK_RSV_W5        = 32'hFF00_0000;
  localparam logic [31:0] TSR_TOPK_RSV_W6        = 32'h0000_0000;
  localparam logic [31:0] TSR_TOPK_RSV_W7        = 32'hFFFF_0000;
  localparam logic [7:0] TSR_TOPK_WORDS          = 8'd231;         // words holding a field (w0 always)
  // DQROW 0x12 (VU §5.6, owner VU)
  localparam logic [31:0] TSR_DQROW_FMT             = 32'h0000_0F00;  // w0[11:8] 2..15 BAD_FMT
  localparam int unsigned TSR_DQROW_FMT_LSB         = 8;
  localparam int unsigned TSR_DQROW_FMT_W           = 4;
  localparam int unsigned TSR_DQROW_FMT_WORD        = 0;
  localparam int unsigned TSR_DQROW_FMT_BIT         = 8;
  localparam logic [31:0] TSR_DQROW_D               = 32'h00FF_FFFF;  // w1[23:0] f32 output line of row 0
  localparam int unsigned TSR_DQROW_D_LSB           = 0;
  localparam int unsigned TSR_DQROW_D_W             = 24;
  localparam int unsigned TSR_DQROW_D_WORD          = 1;
  localparam int unsigned TSR_DQROW_D_BIT           = 32;
  localparam logic [31:0] TSR_DQROW_SRC             = 32'h00FF_FFFF;  // w2[23:0] SPM line of row 0's repacked bytes
  localparam int unsigned TSR_DQROW_SRC_LSB         = 0;
  localparam int unsigned TSR_DQROW_SRC_W           = 24;
  localparam int unsigned TSR_DQROW_SRC_WORD        = 2;
  localparam int unsigned TSR_DQROW_SRC_BIT         = 64;
  localparam logic [31:0] TSR_DQROW_K               = 32'hFFFF_FFFF;  // w5 elements per row (K_RANGE checks)
  localparam int unsigned TSR_DQROW_K_LSB           = 0;
  localparam int unsigned TSR_DQROW_K_W             = 32;
  localparam int unsigned TSR_DQROW_K_WORD          = 5;
  localparam int unsigned TSR_DQROW_K_BIT           = 160;
  localparam logic [31:0] TSR_DQROW_ROWS            = 32'h0000_FFFF;  // w6[15:0] rows
  localparam int unsigned TSR_DQROW_ROWS_LSB        = 0;
  localparam int unsigned TSR_DQROW_ROWS_W          = 16;
  localparam int unsigned TSR_DQROW_ROWS_WORD       = 6;
  localparam int unsigned TSR_DQROW_ROWS_BIT        = 192;
  localparam logic [31:0] TSR_DQROW_SRC_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines; 0 = ROW_BEATS(fmt, K)
  localparam int unsigned TSR_DQROW_SRC_STRIDE_LSB  = 0;
  localparam int unsigned TSR_DQROW_SRC_STRIDE_W    = 16;
  localparam int unsigned TSR_DQROW_SRC_STRIDE_WORD = 7;
  localparam int unsigned TSR_DQROW_SRC_STRIDE_BIT  = 224;
  localparam logic [31:0] TSR_DQROW_D_STRIDE        = 32'hFFFF_0000;  // w7[31:16] lines; 0 = L(K)
  localparam int unsigned TSR_DQROW_D_STRIDE_LSB    = 16;
  localparam int unsigned TSR_DQROW_D_STRIDE_W      = 16;
  localparam int unsigned TSR_DQROW_D_STRIDE_WORD   = 7;
  localparam int unsigned TSR_DQROW_D_STRIDE_BIT    = 240;
  // DQROW_FMT takes the FMT_* values
  localparam logic [31:0] TSR_DQROW_RSV_W0 = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_DQROW_RSV_W1 = 32'hFF00_0000;
  localparam logic [31:0] TSR_DQROW_RSV_W2 = 32'hFF00_0000;
  localparam logic [31:0] TSR_DQROW_RSV_W3 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_DQROW_RSV_W4 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_DQROW_RSV_W5 = 32'h0000_0000;
  localparam logic [31:0] TSR_DQROW_RSV_W6 = 32'hFFFF_0000;
  localparam logic [31:0] TSR_DQROW_RSV_W7 = 32'h0000_0000;
  localparam logic [7:0] TSR_DQROW_WORDS   = 8'd231;         // words holding a field (w0 always)
  // VMASK 0x38 (VU §5.7, owner VU)
  localparam logic [31:0] TSR_VMASK_SUB           = 32'h0000_0F00;  // w0[11:8]
  localparam int unsigned TSR_VMASK_SUB_LSB       = 8;
  localparam int unsigned TSR_VMASK_SUB_W         = 4;
  localparam int unsigned TSR_VMASK_SUB_WORD      = 0;
  localparam int unsigned TSR_VMASK_SUB_BIT       = 8;
  localparam logic [31:0] TSR_VMASK_X             = 32'h00FF_FFFF;  // w1[23:0] operand line (in place)
  localparam int unsigned TSR_VMASK_X_LSB         = 0;
  localparam int unsigned TSR_VMASK_X_W           = 24;
  localparam int unsigned TSR_VMASK_X_WORD        = 1;
  localparam int unsigned TSR_VMASK_X_BIT         = 32;
  localparam logic [31:0] TSR_VMASK_WIN           = 32'hFFFF_FFFF;  // w2 window size W, u32; 0 = no window
  localparam int unsigned TSR_VMASK_WIN_LSB       = 0;
  localparam int unsigned TSR_VMASK_WIN_W         = 32;
  localparam int unsigned TSR_VMASK_WIN_WORD      = 2;
  localparam int unsigned TSR_VMASK_WIN_BIT       = 64;
  localparam logic [31:0] TSR_VMASK_K0            = 32'hFFFF_FFFF;  // w3 i32 position of column 0
  localparam int unsigned TSR_VMASK_K0_LSB        = 0;
  localparam int unsigned TSR_VMASK_K0_W          = 32;
  localparam int unsigned TSR_VMASK_K0_WORD       = 3;
  localparam int unsigned TSR_VMASK_K0_BIT        = 96;
  localparam logic [31:0] TSR_VMASK_Q0            = 32'hFFFF_FFFF;  // w4 i32 query position of row group 0
  localparam int unsigned TSR_VMASK_Q0_LSB        = 0;
  localparam int unsigned TSR_VMASK_Q0_W          = 32;
  localparam int unsigned TSR_VMASK_Q0_WORD       = 4;
  localparam int unsigned TSR_VMASK_Q0_BIT        = 128;
  localparam logic [31:0] TSR_VMASK_M             = 32'h00FF_FFFF;  // w5[23:0] columns
  localparam int unsigned TSR_VMASK_M_LSB         = 0;
  localparam int unsigned TSR_VMASK_M_W           = 24;
  localparam int unsigned TSR_VMASK_M_WORD        = 5;
  localparam int unsigned TSR_VMASK_M_BIT         = 160;
  localparam logic [31:0] TSR_VMASK_ROWS          = 32'h0000_FFFF;  // w6[15:0] rows
  localparam int unsigned TSR_VMASK_ROWS_LSB      = 0;
  localparam int unsigned TSR_VMASK_ROWS_W        = 16;
  localparam int unsigned TSR_VMASK_ROWS_WORD     = 6;
  localparam int unsigned TSR_VMASK_ROWS_BIT      = 192;
  localparam logic [31:0] TSR_VMASK_G             = 32'hFFFF_0000;  // w6[31:16] rows per query position (>= 1)
  localparam int unsigned TSR_VMASK_G_LSB         = 16;
  localparam int unsigned TSR_VMASK_G_W           = 16;
  localparam int unsigned TSR_VMASK_G_WORD        = 6;
  localparam int unsigned TSR_VMASK_G_BIT         = 208;
  localparam logic [31:0] TSR_VMASK_X_STRIDE      = 32'h0000_FFFF;  // w7[15:0] lines
  localparam int unsigned TSR_VMASK_X_STRIDE_LSB  = 0;
  localparam int unsigned TSR_VMASK_X_STRIDE_W    = 16;
  localparam int unsigned TSR_VMASK_X_STRIDE_WORD = 7;
  localparam int unsigned TSR_VMASK_X_STRIDE_BIT  = 224;
  localparam logic [31:0] TSR_VMASK_P             = 32'hFFFF_0000;  // w7[31:16] row period; 0 = none
  localparam int unsigned TSR_VMASK_P_LSB         = 16;
  localparam int unsigned TSR_VMASK_P_W           = 16;
  localparam int unsigned TSR_VMASK_P_WORD        = 7;
  localparam int unsigned TSR_VMASK_P_BIT         = 240;
  localparam logic [3:0] TSR_VMASK_SUB_WINDOW     = 4'd0;           // position window; others ILLEGAL_OP
  localparam logic [31:0] TSR_VMASK_RSV_W0        = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_VMASK_RSV_W1        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W2        = 32'h0000_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W3        = 32'h0000_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W4        = 32'h0000_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W5        = 32'hFF00_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W6        = 32'h0000_0000;
  localparam logic [31:0] TSR_VMASK_RSV_W7        = 32'h0000_0000;
  localparam logic [7:0] TSR_VMASK_WORDS          = 8'd255;         // words holding a field (w0 always)
  // GDN 0x39 (VU §7.12.4, owner VU)
  localparam logic [31:0] TSR_GDN_SA        = 32'h000F_0000;  // x16[3:0] s register holding alpha
  localparam int unsigned TSR_GDN_SA_LSB    = 16;
  localparam int unsigned TSR_GDN_SA_W      = 4;
  localparam int unsigned TSR_GDN_SA_WORD   = 0;
  localparam int unsigned TSR_GDN_SA_BIT    = 16;
  localparam logic [31:0] TSR_GDN_SB        = 32'h00F0_0000;  // x16[7:4] s register holding beta
  localparam int unsigned TSR_GDN_SB_LSB    = 20;
  localparam int unsigned TSR_GDN_SB_W      = 4;
  localparam int unsigned TSR_GDN_SB_WORD   = 0;
  localparam int unsigned TSR_GDN_SB_BIT    = 20;
  localparam logic [31:0] TSR_GDN_D         = 32'h00FF_FFFF;  // w1[23:0] S' tile, line of row 0 (may equal A: in place)
  localparam int unsigned TSR_GDN_D_LSB     = 0;
  localparam int unsigned TSR_GDN_D_W       = 24;
  localparam int unsigned TSR_GDN_D_WORD    = 1;
  localparam int unsigned TSR_GDN_D_BIT     = 32;
  localparam logic [31:0] TSR_GDN_A         = 32'h00FF_FFFF;  // w2[23:0] S tile, line of row 0
  localparam int unsigned TSR_GDN_A_LSB     = 0;
  localparam int unsigned TSR_GDN_A_W       = 24;
  localparam int unsigned TSR_GDN_A_WORD    = 2;
  localparam int unsigned TSR_GDN_A_BIT     = 64;
  localparam logic [31:0] TSR_GDN_K         = 32'h00FF_FFFF;  // w3[23:0] k vector line (m words)
  localparam int unsigned TSR_GDN_K_LSB     = 0;
  localparam int unsigned TSR_GDN_K_W       = 24;
  localparam int unsigned TSR_GDN_K_WORD    = 3;
  localparam int unsigned TSR_GDN_K_BIT     = 96;
  localparam logic [31:0] TSR_GDN_Q         = 32'h00FF_FFFF;  // w4[23:0] q vector line (m words)
  localparam int unsigned TSR_GDN_Q_LSB     = 0;
  localparam int unsigned TSR_GDN_Q_W       = 24;
  localparam int unsigned TSR_GDN_Q_WORD    = 4;
  localparam int unsigned TSR_GDN_Q_BIT     = 128;
  localparam logic [31:0] TSR_GDN_V         = 32'h00FF_FFFF;  // w5[23:0] v vector line (rows words)
  localparam int unsigned TSR_GDN_V_LSB     = 0;
  localparam int unsigned TSR_GDN_V_W       = 24;
  localparam int unsigned TSR_GDN_V_WORD    = 5;
  localparam int unsigned TSR_GDN_V_BIT     = 160;
  localparam logic [31:0] TSR_GDN_O         = 32'h00FF_FFFF;  // w6[23:0] o vector line (rows words, tail rule of VU §4.2)
  localparam int unsigned TSR_GDN_O_LSB     = 0;
  localparam int unsigned TSR_GDN_O_W       = 24;
  localparam int unsigned TSR_GDN_O_WORD    = 6;
  localparam int unsigned TSR_GDN_O_BIT     = 192;
  localparam logic [31:0] TSR_GDN_M         = 32'h0000_FFFF;  // w7[15:0] d_k: a multiple of 16
  localparam int unsigned TSR_GDN_M_LSB     = 0;
  localparam int unsigned TSR_GDN_M_W       = 16;
  localparam int unsigned TSR_GDN_M_WORD    = 7;
  localparam int unsigned TSR_GDN_M_BIT     = 224;
  localparam logic [31:0] TSR_GDN_ROWS      = 32'hFFFF_0000;  // w7[31:16] d_v rows of the tile
  localparam int unsigned TSR_GDN_ROWS_LSB  = 16;
  localparam int unsigned TSR_GDN_ROWS_W    = 16;
  localparam int unsigned TSR_GDN_ROWS_WORD = 7;
  localparam int unsigned TSR_GDN_ROWS_BIT  = 240;
  localparam logic [31:0] TSR_GDN_RSV_W0    = 32'hFF00_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_GDN_RSV_W1    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W2    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W3    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W4    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W5    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W6    = 32'hFF00_0000;
  localparam logic [31:0] TSR_GDN_RSV_W7    = 32'h0000_0000;
  localparam logic [7:0] TSR_GDN_WORDS      = 8'd255;         // words holding a field (w0 always)
  // SOP 0x40 (VU §5.8, owner VU)
  localparam logic [31:0] TSR_SOP_SUB       = 32'h0000_0F00;  // w0[11:8] function (SOP_FUNCS)
  localparam int unsigned TSR_SOP_SUB_LSB   = 8;
  localparam int unsigned TSR_SOP_SUB_W     = 4;
  localparam int unsigned TSR_SOP_SUB_WORD  = 0;
  localparam int unsigned TSR_SOP_SUB_BIT   = 8;
  localparam logic [31:0] TSR_SOP_SD        = 32'h0000_000F;  // w1[3:0] destination s register
  localparam int unsigned TSR_SOP_SD_LSB    = 0;
  localparam int unsigned TSR_SOP_SD_W      = 4;
  localparam int unsigned TSR_SOP_SD_WORD   = 1;
  localparam int unsigned TSR_SOP_SD_BIT    = 32;
  localparam logic [31:0] TSR_SOP_SA        = 32'h0000_00F0;  // w1[7:4] source s register a
  localparam int unsigned TSR_SOP_SA_LSB    = 4;
  localparam int unsigned TSR_SOP_SA_W      = 4;
  localparam int unsigned TSR_SOP_SA_WORD   = 1;
  localparam int unsigned TSR_SOP_SA_BIT    = 36;
  localparam logic [31:0] TSR_SOP_SB        = 32'h0000_0F00;  // w1[11:8] source s register b
  localparam int unsigned TSR_SOP_SB_LSB    = 8;
  localparam int unsigned TSR_SOP_SB_W      = 4;
  localparam int unsigned TSR_SOP_SB_WORD   = 1;
  localparam int unsigned TSR_SOP_SB_BIT    = 40;
  localparam logic [31:0] TSR_SOP_IMM       = 32'hFFFF_FFFF;  // w2 f32 bits
  localparam int unsigned TSR_SOP_IMM_LSB   = 0;
  localparam int unsigned TSR_SOP_IMM_W     = 32;
  localparam int unsigned TSR_SOP_IMM_WORD  = 2;
  localparam int unsigned TSR_SOP_IMM_BIT   = 64;
  localparam logic [31:0] TSR_SOP_ADDR      = 32'h0FFF_FFFF;  // w3[27:0] SPM word address (LDS / STS)
  localparam int unsigned TSR_SOP_ADDR_LSB  = 0;
  localparam int unsigned TSR_SOP_ADDR_W    = 28;
  localparam int unsigned TSR_SOP_ADDR_WORD = 3;
  localparam int unsigned TSR_SOP_ADDR_BIT  = 96;
  localparam logic [3:0] TSR_SOP_SUB_MOVI   = 4'd0;           // sd <- imm (raw) (uses SD, IMM)
  localparam logic [3:0] TSR_SOP_SUB_ADD    = 4'd1;           // sd <- FA(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_MUL    = 4'd2;           // sd <- FM(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_ADDI   = 4'd3;           // sd <- FA(sa, imm) (uses SD, SA, IMM)
  localparam logic [3:0] TSR_SOP_SUB_MULI   = 4'd4;           // sd <- FM(sa, imm) (uses SD, SA, IMM)
  localparam logic [3:0] TSR_SOP_SUB_SQRT   = 4'd5;           // sd <- SQ(sa) (uses SD, SA)
  localparam logic [3:0] TSR_SOP_SUB_RECIP  = 4'd6;           // sd <- RC(sa) (uses SD, SA)
  localparam logic [3:0] TSR_SOP_SUB_LDS    = 4'd7;           // sd <- SPM word (raw) (uses SD, ADDR)
  localparam logic [3:0] TSR_SOP_SUB_STS    = 4'd8;           // SPM word <- sa (raw; exactly 4 bytes) (uses SA, ADDR)
  localparam logic [3:0] TSR_SOP_SUB_SUB    = 4'd9;           // sd <- FS(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_DIV    = 4'd10;          // sd <- FD(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_RSQRT  = 4'd11;          // sd <- RC(SQ(sa)), two roundings (uses SD, SA)
  localparam logic [3:0] TSR_SOP_SUB_MAX    = 4'd12;          // sd <- MAXT(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_MIN    = 4'd13;          // sd <- MINT(sa, sb) (uses SD, SA, SB)
  localparam logic [3:0] TSR_SOP_SUB_EXP    = 4'd14;          // sd <- EXPK(sa, P1) (uses SD, SA)
  localparam logic [31:0] TSR_SOP_RSV_W0    = 32'hFFFF_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_SOP_RSV_W1    = 32'hFFFF_F000;
  localparam logic [31:0] TSR_SOP_RSV_W2    = 32'h0000_0000;
  localparam logic [31:0] TSR_SOP_RSV_W3    = 32'hF000_0000;
  localparam logic [31:0] TSR_SOP_RSV_W4    = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_SOP_RSV_W5    = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_SOP_RSV_W6    = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_SOP_RSV_W7    = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_SOP_WORDS      = 8'd15;          // words holding a field (w0 always)

  localparam int unsigned TSR_VOP_FUNC_N = 25;
  localparam int unsigned TSR_SOP_SUB_N = 15;

  // Operands a VOP function reads: bit0 a, bit1 b, bit2 c (VU §5.2).
  function automatic logic [2:0] tsr_vop_func_reads(input logic [7:0] func);
    case (func)
      8'h00, 8'h01, 8'h02, 8'h03, 8'h07, 8'h0A, 8'h0F, 8'h13, 8'h18: return 3'd3;
      8'h04, 8'h05, 8'h0B, 8'h10: return 3'd5;
      8'h06, 8'h08, 8'h09, 8'h11, 8'h12, 8'h14, 8'h15, 8'h17: return 3'd1;
      8'h0C, 8'h0D, 8'h0E: return 3'd7;
      8'h16: return 3'd4;
      default: return 3'd0;
    endcase
  endfunction

  // Half-rate (back-lane) VOP functions (VU §5.2, §7.4).
  function automatic logic tsr_vop_func_half(input logic [7:0] func);
    case (func)
      8'h06, 8'h07, 8'h11, 8'h12, 8'h13, 8'h14, 8'h15, 8'h17, 8'h18: return 1'b1;
      default: return 1'b0;
    endcase
  endfunction

  // Fields an SOP function uses: bit0 SD, bit1 SA, bit2 SB, bit3 IMM, bit4 ADDR (VU §5.8).
  function automatic logic [4:0] tsr_sop_uses(input logic [3:0] sub);
    case (sub)
      4'd0: return 5'd9;
      4'd1, 4'd2, 4'd9, 4'd10, 4'd12, 4'd13: return 5'd7;
      4'd3, 4'd4: return 5'd11;
      4'd5, 4'd6, 4'd11, 4'd14: return 5'd3;
      4'd7: return 5'd17;
      4'd8: return 5'd18;
      default: return 5'd0;
    endcase
  endfunction

  // ---------------------------------------------------------------------------------------------
  // AT encodings (AT §5; VU §5.11 restates the vector-op encodings)
  // ---------------------------------------------------------------------------------------------
  // KVST 0x14 (AT §5.1, owner AT)
  localparam logic [31:0] TSR_KVST_KV              = 32'h0000_0F00;  // w0[11:8] plane
  localparam int unsigned TSR_KVST_KV_LSB          = 8;
  localparam int unsigned TSR_KVST_KV_W            = 4;
  localparam int unsigned TSR_KVST_KV_WORD         = 0;
  localparam int unsigned TSR_KVST_KV_BIT          = 8;
  localparam logic [31:0] TSR_KVST_N_ROWS_M1       = 32'h003F_0000;  // x16[5:0] n_rows - 1 (n_rows 1..64)
  localparam int unsigned TSR_KVST_N_ROWS_M1_LSB   = 16;
  localparam int unsigned TSR_KVST_N_ROWS_M1_W     = 6;
  localparam int unsigned TSR_KVST_N_ROWS_M1_WORD  = 0;
  localparam int unsigned TSR_KVST_N_ROWS_M1_BIT   = 16;
  localparam logic [31:0] TSR_KVST_SRC             = 32'hFFFF_FFFF;  // w1 first QA(A32, E) vector (SPM line)
  localparam int unsigned TSR_KVST_SRC_LSB         = 0;
  localparam int unsigned TSR_KVST_SRC_W           = 32;
  localparam int unsigned TSR_KVST_SRC_WORD        = 1;
  localparam int unsigned TSR_KVST_SRC_BIT         = 32;
  localparam logic [31:0] TSR_KVST_BASE            = 32'hFFFF_FFFF;  // w2 plane base B (beat, every channel)
  localparam int unsigned TSR_KVST_BASE_LSB        = 0;
  localparam int unsigned TSR_KVST_BASE_W          = 32;
  localparam int unsigned TSR_KVST_BASE_WORD       = 2;
  localparam int unsigned TSR_KVST_BASE_BIT        = 64;
  localparam logic [31:0] TSR_KVST_E64             = 32'h0000_01FF;  // w3[8:0] E / 64
  localparam int unsigned TSR_KVST_E64_LSB         = 0;
  localparam int unsigned TSR_KVST_E64_W           = 9;
  localparam int unsigned TSR_KVST_E64_WORD        = 3;
  localparam int unsigned TSR_KVST_E64_BIT         = 96;
  localparam logic [31:0] TSR_KVST_LG              = 32'hFFFF_0000;  // w3[31:16] local groups per channel
  localparam int unsigned TSR_KVST_LG_LSB          = 16;
  localparam int unsigned TSR_KVST_LG_W            = 16;
  localparam int unsigned TSR_KVST_LG_WORD         = 3;
  localparam int unsigned TSR_KVST_LG_BIT          = 112;
  localparam logic [31:0] TSR_KVST_C0              = 32'hFFFF_FFFF;  // w4 first cell
  localparam int unsigned TSR_KVST_C0_LSB          = 0;
  localparam int unsigned TSR_KVST_C0_W            = 32;
  localparam int unsigned TSR_KVST_C0_WORD         = 4;
  localparam int unsigned TSR_KVST_C0_BIT          = 128;
  localparam logic [31:0] TSR_KVST_SRC_STRIDE      = 32'hFFFF_FFFF;  // w5 lines; 0 = QA_LINES(A32, E)
  localparam int unsigned TSR_KVST_SRC_STRIDE_LSB  = 0;
  localparam int unsigned TSR_KVST_SRC_STRIDE_W    = 32;
  localparam int unsigned TSR_KVST_SRC_STRIDE_WORD = 5;
  localparam int unsigned TSR_KVST_SRC_STRIDE_BIT  = 160;
  localparam logic [3:0] TSR_KVST_KV_K             = 4'd0;           // K plane
  localparam logic [3:0] TSR_KVST_KV_V             = 4'd1;           // V plane
  localparam logic [31:0] TSR_KVST_RSV_W0          = 32'hFFC0_F000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_KVST_RSV_W1          = 32'h0000_0000;
  localparam logic [31:0] TSR_KVST_RSV_W2          = 32'h0000_0000;
  localparam logic [31:0] TSR_KVST_RSV_W3          = 32'h0000_FE00;
  localparam logic [31:0] TSR_KVST_RSV_W4          = 32'h0000_0000;
  localparam logic [31:0] TSR_KVST_RSV_W5          = 32'h0000_0000;
  localparam logic [31:0] TSR_KVST_RSV_W6          = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_KVST_RSV_W7          = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_KVST_WORDS            = 8'd63;          // words holding a field (w0 always)
  // ATTQK 0x21 (AT §5.2, owner AT)
  localparam logic [31:0] TSR_ATTQK_FMT             = 32'h0000_0F00;  // w0[11:8] KV format
  localparam int unsigned TSR_ATTQK_FMT_LSB         = 8;
  localparam int unsigned TSR_ATTQK_FMT_W           = 4;
  localparam int unsigned TSR_ATTQK_FMT_WORD        = 0;
  localparam int unsigned TSR_ATTQK_FMT_BIT         = 8;
  localparam logic [31:0] TSR_ATTQK_CAUSAL          = 32'h0000_1000;  // w0[12]
  localparam int unsigned TSR_ATTQK_CAUSAL_LSB      = 12;
  localparam int unsigned TSR_ATTQK_CAUSAL_W        = 1;
  localparam int unsigned TSR_ATTQK_CAUSAL_WORD     = 0;
  localparam int unsigned TSR_ATTQK_CAUSAL_BIT      = 12;
  localparam logic [31:0] TSR_ATTQK_SLIDE           = 32'h0000_2000;  // w0[13]
  localparam int unsigned TSR_ATTQK_SLIDE_LSB       = 13;
  localparam int unsigned TSR_ATTQK_SLIDE_W         = 1;
  localparam int unsigned TSR_ATTQK_SLIDE_WORD      = 0;
  localparam int unsigned TSR_ATTQK_SLIDE_BIT       = 13;
  localparam logic [31:0] TSR_ATTQK_MAXOUT          = 32'h0000_4000;  // w0[14] write row maxima at MDST
  localparam int unsigned TSR_ATTQK_MAXOUT_LSB      = 14;
  localparam int unsigned TSR_ATTQK_MAXOUT_W        = 1;
  localparam int unsigned TSR_ATTQK_MAXOUT_WORD     = 0;
  localparam int unsigned TSR_ATTQK_MAXOUT_BIT      = 14;
  localparam logic [31:0] TSR_ATTQK_G_M1            = 32'h003F_0000;  // x16[5:0] G - 1 (heads per KV head)
  localparam int unsigned TSR_ATTQK_G_M1_LSB        = 16;
  localparam int unsigned TSR_ATTQK_G_M1_W          = 6;
  localparam int unsigned TSR_ATTQK_G_M1_WORD       = 0;
  localparam int unsigned TSR_ATTQK_G_M1_BIT        = 16;
  localparam logic [31:0] TSR_ATTQK_HG0             = 32'h0FC0_0000;  // x16[11:6] first head of the pass within each KV-head group
  localparam int unsigned TSR_ATTQK_HG0_LSB         = 22;
  localparam int unsigned TSR_ATTQK_HG0_W           = 6;
  localparam int unsigned TSR_ATTQK_HG0_WORD        = 0;
  localparam int unsigned TSR_ATTQK_HG0_BIT         = 22;
  localparam logic [31:0] TSR_ATTQK_LOCAL_SLOT      = 32'h3000_0000;  // w0[29:28] 0 = SPM, 1..3 = local FAST attention slot
  localparam int unsigned TSR_ATTQK_LOCAL_SLOT_LSB  = 28;
  localparam int unsigned TSR_ATTQK_LOCAL_SLOT_W    = 2;
  localparam int unsigned TSR_ATTQK_LOCAL_SLOT_WORD = 0;
  localparam int unsigned TSR_ATTQK_LOCAL_SLOT_BIT  = 28;
  localparam logic [31:0] TSR_ATTQK_DST             = 32'hFFFF_FFFF;  // w1 score rows, row r at DST + r x RSTRIDE
  localparam int unsigned TSR_ATTQK_DST_LSB         = 0;
  localparam int unsigned TSR_ATTQK_DST_W           = 32;
  localparam int unsigned TSR_ATTQK_DST_WORD        = 1;
  localparam int unsigned TSR_ATTQK_DST_BIT         = 32;
  localparam logic [31:0] TSR_ATTQK_X               = 32'hFFFF_FFFF;  // w2 T_q QA(A32, n_head x D) vectors
  localparam int unsigned TSR_ATTQK_X_LSB           = 0;
  localparam int unsigned TSR_ATTQK_X_W             = 32;
  localparam int unsigned TSR_ATTQK_X_WORD          = 2;
  localparam int unsigned TSR_ATTQK_X_BIT           = 64;
  localparam logic [31:0] TSR_ATTQK_BASE            = 32'hFFFF_FFFF;  // w3 K plane base B
  localparam int unsigned TSR_ATTQK_BASE_LSB        = 0;
  localparam int unsigned TSR_ATTQK_BASE_W          = 32;
  localparam int unsigned TSR_ATTQK_BASE_WORD       = 3;
  localparam int unsigned TSR_ATTQK_BASE_BIT        = 96;
  localparam logic [31:0] TSR_ATTQK_E64             = 32'h0000_01FF;  // w4[8:0] E / 64 (1..512)
  localparam int unsigned TSR_ATTQK_E64_LSB         = 0;
  localparam int unsigned TSR_ATTQK_E64_W           = 9;
  localparam int unsigned TSR_ATTQK_E64_WORD        = 4;
  localparam int unsigned TSR_ATTQK_E64_BIT         = 128;
  localparam logic [31:0] TSR_ATTQK_DCODE           = 32'h0000_0600;  // w4[10:9] D = 64 x 2^dcode
  localparam int unsigned TSR_ATTQK_DCODE_LSB       = 9;
  localparam int unsigned TSR_ATTQK_DCODE_W         = 2;
  localparam int unsigned TSR_ATTQK_DCODE_WORD      = 4;
  localparam int unsigned TSR_ATTQK_DCODE_BIT       = 137;
  localparam logic [31:0] TSR_ATTQK_TQ_M1           = 32'h0000_3800;  // w4[13:11] T_q - 1
  localparam int unsigned TSR_ATTQK_TQ_M1_LSB       = 11;
  localparam int unsigned TSR_ATTQK_TQ_M1_W         = 3;
  localparam int unsigned TSR_ATTQK_TQ_M1_WORD      = 4;
  localparam int unsigned TSR_ATTQK_TQ_M1_BIT       = 139;
  localparam logic [31:0] TSR_ATTQK_GS_M1           = 32'h0001_C000;  // w4[16:14] GS - 1
  localparam int unsigned TSR_ATTQK_GS_M1_LSB       = 14;
  localparam int unsigned TSR_ATTQK_GS_M1_W         = 3;
  localparam int unsigned TSR_ATTQK_GS_M1_WORD      = 4;
  localparam int unsigned TSR_ATTQK_GS_M1_BIT       = 142;
  localparam logic [31:0] TSR_ATTQK_RSTRIDE         = 32'hFFFE_0000;  // w4[31:17] lines; 0 = n_groups
  localparam int unsigned TSR_ATTQK_RSTRIDE_LSB     = 17;
  localparam int unsigned TSR_ATTQK_RSTRIDE_W       = 15;
  localparam int unsigned TSR_ATTQK_RSTRIDE_WORD    = 4;
  localparam int unsigned TSR_ATTQK_RSTRIDE_BIT     = 145;
  localparam logic [31:0] TSR_ATTQK_U0              = 32'h0000_FFFF;  // w5[15:0] first group
  localparam int unsigned TSR_ATTQK_U0_LSB          = 0;
  localparam int unsigned TSR_ATTQK_U0_W            = 16;
  localparam int unsigned TSR_ATTQK_U0_WORD         = 5;
  localparam int unsigned TSR_ATTQK_U0_BIT          = 160;
  localparam logic [31:0] TSR_ATTQK_N_GROUPS        = 32'hFFFF_0000;  // w5[31:16] 1..4095
  localparam int unsigned TSR_ATTQK_N_GROUPS_LSB    = 16;
  localparam int unsigned TSR_ATTQK_N_GROUPS_W      = 16;
  localparam int unsigned TSR_ATTQK_N_GROUPS_WORD   = 5;
  localparam int unsigned TSR_ATTQK_N_GROUPS_BIT    = 176;
  localparam logic [31:0] TSR_ATTQK_C_LO            = 32'h0000_FFFF;  // w6[15:0] cell bound relative to 16 u0
  localparam int unsigned TSR_ATTQK_C_LO_LSB        = 0;
  localparam int unsigned TSR_ATTQK_C_LO_W          = 16;
  localparam int unsigned TSR_ATTQK_C_LO_WORD       = 6;
  localparam int unsigned TSR_ATTQK_C_LO_BIT        = 192;
  localparam logic [31:0] TSR_ATTQK_C_HI            = 32'hFFFF_0000;  // w6[31:16] cell bound relative to 16 u0
  localparam int unsigned TSR_ATTQK_C_HI_LSB        = 16;
  localparam int unsigned TSR_ATTQK_C_HI_W          = 16;
  localparam int unsigned TSR_ATTQK_C_HI_WORD       = 6;
  localparam int unsigned TSR_ATTQK_C_HI_BIT        = 208;
  localparam logic [31:0] TSR_ATTQK_MDST            = 32'hFFFF_FFFF;  // w7 row-maximum vector line; 0 unless MAXOUT
  localparam int unsigned TSR_ATTQK_MDST_LSB        = 0;
  localparam int unsigned TSR_ATTQK_MDST_W          = 32;
  localparam int unsigned TSR_ATTQK_MDST_WORD       = 7;
  localparam int unsigned TSR_ATTQK_MDST_BIT        = 224;
  // ATTQK_FMT takes the KVFMT_* values
  // ATTQK_DCODE takes the DCODE_* values
  localparam logic [31:0] TSR_ATTQK_RSV_W0 = 32'hC000_8000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ATTQK_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W4 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W5 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W6 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTQK_RSV_W7 = 32'h0000_0000;
  localparam logic [7:0] TSR_ATTQK_WORDS   = 8'd255;         // words holding a field (w0 always)
  // ATTPV 0x22 (AT §5.3, owner AT)
  localparam logic [31:0] TSR_ATTPV_FMT             = 32'h0000_0F00;  // w0[11:8] KV format
  localparam int unsigned TSR_ATTPV_FMT_LSB         = 8;
  localparam int unsigned TSR_ATTPV_FMT_W           = 4;
  localparam int unsigned TSR_ATTPV_FMT_WORD        = 0;
  localparam int unsigned TSR_ATTPV_FMT_BIT         = 8;
  localparam logic [31:0] TSR_ATTPV_LSUM            = 32'h0000_1000;  // w0[12] also row sums
  localparam int unsigned TSR_ATTPV_LSUM_LSB        = 12;
  localparam int unsigned TSR_ATTPV_LSUM_W          = 1;
  localparam int unsigned TSR_ATTPV_LSUM_WORD       = 0;
  localparam int unsigned TSR_ATTPV_LSUM_BIT        = 12;
  localparam logic [31:0] TSR_ATTPV_G_M1            = 32'h003F_0000;  // x16[5:0] G - 1
  localparam int unsigned TSR_ATTPV_G_M1_LSB        = 16;
  localparam int unsigned TSR_ATTPV_G_M1_W          = 6;
  localparam int unsigned TSR_ATTPV_G_M1_WORD       = 0;
  localparam int unsigned TSR_ATTPV_G_M1_BIT        = 16;
  localparam logic [31:0] TSR_ATTPV_HG0             = 32'h0FC0_0000;  // x16[11:6] first head of the pass
  localparam int unsigned TSR_ATTPV_HG0_LSB         = 22;
  localparam int unsigned TSR_ATTPV_HG0_W           = 6;
  localparam int unsigned TSR_ATTPV_HG0_WORD        = 0;
  localparam int unsigned TSR_ATTPV_HG0_BIT         = 22;
  localparam logic [31:0] TSR_ATTPV_LOCAL_SLOT      = 32'h3000_0000;  // w0[29:28] 0 = SPM, 1..3 = local FAST attention slot
  localparam int unsigned TSR_ATTPV_LOCAL_SLOT_LSB  = 28;
  localparam int unsigned TSR_ATTPV_LOCAL_SLOT_W    = 2;
  localparam int unsigned TSR_ATTPV_LOCAL_SLOT_WORD = 0;
  localparam int unsigned TSR_ATTPV_LOCAL_SLOT_BIT  = 28;
  localparam logic [31:0] TSR_ATTPV_DST             = 32'hFFFF_FFFF;  // w1 channel partials
  localparam int unsigned TSR_ATTPV_DST_LSB         = 0;
  localparam int unsigned TSR_ATTPV_DST_W           = 32;
  localparam int unsigned TSR_ATTPV_DST_WORD        = 1;
  localparam int unsigned TSR_ATTPV_DST_BIT         = 32;
  localparam logic [31:0] TSR_ATTPV_P               = 32'hFFFF_FFFF;  // w2 weight rows, row r at P + r x PSTRIDE
  localparam int unsigned TSR_ATTPV_P_LSB           = 0;
  localparam int unsigned TSR_ATTPV_P_W             = 32;
  localparam int unsigned TSR_ATTPV_P_WORD          = 2;
  localparam int unsigned TSR_ATTPV_P_BIT           = 64;
  localparam logic [31:0] TSR_ATTPV_BASE            = 32'hFFFF_FFFF;  // w3 V plane base B
  localparam int unsigned TSR_ATTPV_BASE_LSB        = 0;
  localparam int unsigned TSR_ATTPV_BASE_W          = 32;
  localparam int unsigned TSR_ATTPV_BASE_WORD       = 3;
  localparam int unsigned TSR_ATTPV_BASE_BIT        = 96;
  localparam logic [31:0] TSR_ATTPV_E64             = 32'h0000_01FF;  // w4[8:0] E / 64
  localparam int unsigned TSR_ATTPV_E64_LSB         = 0;
  localparam int unsigned TSR_ATTPV_E64_W           = 9;
  localparam int unsigned TSR_ATTPV_E64_WORD        = 4;
  localparam int unsigned TSR_ATTPV_E64_BIT         = 128;
  localparam logic [31:0] TSR_ATTPV_DCODE           = 32'h0000_0600;  // w4[10:9]
  localparam int unsigned TSR_ATTPV_DCODE_LSB       = 9;
  localparam int unsigned TSR_ATTPV_DCODE_W         = 2;
  localparam int unsigned TSR_ATTPV_DCODE_WORD      = 4;
  localparam int unsigned TSR_ATTPV_DCODE_BIT       = 137;
  localparam logic [31:0] TSR_ATTPV_TQ_M1           = 32'h0000_3800;  // w4[13:11] T_q - 1
  localparam int unsigned TSR_ATTPV_TQ_M1_LSB       = 11;
  localparam int unsigned TSR_ATTPV_TQ_M1_W         = 3;
  localparam int unsigned TSR_ATTPV_TQ_M1_WORD      = 4;
  localparam int unsigned TSR_ATTPV_TQ_M1_BIT       = 139;
  localparam logic [31:0] TSR_ATTPV_GS_M1           = 32'h0001_C000;  // w4[16:14] GS - 1
  localparam int unsigned TSR_ATTPV_GS_M1_LSB       = 14;
  localparam int unsigned TSR_ATTPV_GS_M1_W         = 3;
  localparam int unsigned TSR_ATTPV_GS_M1_WORD      = 4;
  localparam int unsigned TSR_ATTPV_GS_M1_BIT       = 142;
  localparam logic [31:0] TSR_ATTPV_PSTRIDE         = 32'hFFFE_0000;  // w4[31:17] lines; 0 = n_groups
  localparam int unsigned TSR_ATTPV_PSTRIDE_LSB     = 17;
  localparam int unsigned TSR_ATTPV_PSTRIDE_W       = 15;
  localparam int unsigned TSR_ATTPV_PSTRIDE_WORD    = 4;
  localparam int unsigned TSR_ATTPV_PSTRIDE_BIT     = 145;
  localparam logic [31:0] TSR_ATTPV_U0              = 32'h0000_FFFF;  // w5[15:0] first group
  localparam int unsigned TSR_ATTPV_U0_LSB          = 0;
  localparam int unsigned TSR_ATTPV_U0_W            = 16;
  localparam int unsigned TSR_ATTPV_U0_WORD         = 5;
  localparam int unsigned TSR_ATTPV_U0_BIT          = 160;
  localparam logic [31:0] TSR_ATTPV_N_GROUPS        = 32'hFFFF_0000;  // w5[31:16]
  localparam int unsigned TSR_ATTPV_N_GROUPS_LSB    = 16;
  localparam int unsigned TSR_ATTPV_N_GROUPS_W      = 16;
  localparam int unsigned TSR_ATTPV_N_GROUPS_WORD   = 5;
  localparam int unsigned TSR_ATTPV_N_GROUPS_BIT    = 176;
  localparam logic [31:0] TSR_ATTPV_LG              = 32'h0000_FFFF;  // w6[15:0] local groups per channel
  localparam int unsigned TSR_ATTPV_LG_LSB          = 0;
  localparam int unsigned TSR_ATTPV_LG_W            = 16;
  localparam int unsigned TSR_ATTPV_LG_WORD         = 6;
  localparam int unsigned TSR_ATTPV_LG_BIT          = 192;
  localparam logic [31:0] TSR_ATTPV_CSTRIDE         = 32'hFFFF_FFFF;  // w7 lines; 0 = R x D / 16
  localparam int unsigned TSR_ATTPV_CSTRIDE_LSB     = 0;
  localparam int unsigned TSR_ATTPV_CSTRIDE_W       = 32;
  localparam int unsigned TSR_ATTPV_CSTRIDE_WORD    = 7;
  localparam int unsigned TSR_ATTPV_CSTRIDE_BIT     = 224;
  // ATTPV_FMT takes the KVFMT_* values
  // ATTPV_DCODE takes the DCODE_* values
  localparam logic [31:0] TSR_ATTPV_RSV_W0 = 32'hC000_E000;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ATTPV_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W4 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W5 = 32'h0000_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W6 = 32'hFFFF_0000;
  localparam logic [31:0] TSR_ATTPV_RSV_W7 = 32'h0000_0000;
  localparam logic [7:0] TSR_ATTPV_WORDS   = 8'd255;         // words holding a field (w0 always)
  // ASMAX 0x34 (AT §5.4, VU §5.11, owner AT)
  localparam logic [31:0] TSR_ASMAX_DST             = 32'hFFFF_FFFF;  // w1
  localparam int unsigned TSR_ASMAX_DST_LSB         = 0;
  localparam int unsigned TSR_ASMAX_DST_W           = 32;
  localparam int unsigned TSR_ASMAX_DST_WORD        = 1;
  localparam int unsigned TSR_ASMAX_DST_BIT         = 32;
  localparam logic [31:0] TSR_ASMAX_SRC             = 32'hFFFF_FFFF;  // w2 may equal DST
  localparam int unsigned TSR_ASMAX_SRC_LSB         = 0;
  localparam int unsigned TSR_ASMAX_SRC_W           = 32;
  localparam int unsigned TSR_ASMAX_SRC_WORD        = 2;
  localparam int unsigned TSR_ASMAX_SRC_BIT         = 64;
  localparam logic [31:0] TSR_ASMAX_MVEC            = 32'hFFFF_FFFF;  // w3 row maxima, word r
  localparam int unsigned TSR_ASMAX_MVEC_LSB        = 0;
  localparam int unsigned TSR_ASMAX_MVEC_W          = 32;
  localparam int unsigned TSR_ASMAX_MVEC_WORD       = 3;
  localparam int unsigned TSR_ASMAX_MVEC_BIT        = 96;
  localparam logic [31:0] TSR_ASMAX_N_LINES         = 32'h0000_FFFF;  // w4[15:0] 1..4095
  localparam int unsigned TSR_ASMAX_N_LINES_LSB     = 0;
  localparam int unsigned TSR_ASMAX_N_LINES_W       = 16;
  localparam int unsigned TSR_ASMAX_N_LINES_WORD    = 4;
  localparam int unsigned TSR_ASMAX_N_LINES_BIT     = 128;
  localparam logic [31:0] TSR_ASMAX_STRIDE          = 32'hFFFF_0000;  // w4[31:16] 0 = n_lines
  localparam int unsigned TSR_ASMAX_STRIDE_LSB      = 16;
  localparam int unsigned TSR_ASMAX_STRIDE_W        = 16;
  localparam int unsigned TSR_ASMAX_STRIDE_WORD     = 4;
  localparam int unsigned TSR_ASMAX_STRIDE_BIT      = 144;
  localparam logic [31:0] TSR_ASMAX_R               = 32'hFFFF_FFFF;  // w5 1..1024
  localparam int unsigned TSR_ASMAX_R_LSB           = 0;
  localparam int unsigned TSR_ASMAX_R_W             = 32;
  localparam int unsigned TSR_ASMAX_R_WORD          = 5;
  localparam int unsigned TSR_ASMAX_R_BIT           = 160;
  localparam logic [31:0] TSR_ASMAX_SCALE           = 32'hFFFF_FFFF;  // w6 f32, finite, normal, > 0
  localparam int unsigned TSR_ASMAX_SCALE_LSB       = 0;
  localparam int unsigned TSR_ASMAX_SCALE_W         = 32;
  localparam int unsigned TSR_ASMAX_SCALE_WORD      = 6;
  localparam int unsigned TSR_ASMAX_SCALE_BIT       = 192;
  localparam logic [31:0] TSR_ASMAX_LOCAL_SLOT      = 32'h0000_0003;  // w7[1:0] 0 = SPM, 1..3 = local FAST attention slot
  localparam int unsigned TSR_ASMAX_LOCAL_SLOT_LSB  = 0;
  localparam int unsigned TSR_ASMAX_LOCAL_SLOT_W    = 2;
  localparam int unsigned TSR_ASMAX_LOCAL_SLOT_WORD = 7;
  localparam int unsigned TSR_ASMAX_LOCAL_SLOT_BIT  = 224;
  localparam logic [31:0] TSR_ASMAX_LOCAL_U0        = 32'h0003_FFFC;  // w7[17:2] first group when LOCAL_SLOT is set
  localparam int unsigned TSR_ASMAX_LOCAL_U0_LSB    = 2;
  localparam int unsigned TSR_ASMAX_LOCAL_U0_W      = 16;
  localparam int unsigned TSR_ASMAX_LOCAL_U0_WORD   = 7;
  localparam int unsigned TSR_ASMAX_LOCAL_U0_BIT    = 226;
  localparam logic [31:0] TSR_ASMAX_RSV_W0          = 32'hFFFF_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ASMAX_RSV_W1          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W2          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W3          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W4          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W5          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W6          = 32'h0000_0000;
  localparam logic [31:0] TSR_ASMAX_RSV_W7          = 32'hFFFC_0000;
  localparam logic [7:0] TSR_ASMAX_WORDS            = 8'd255;         // words holding a field (w0 always)
  // AMASK 0x35 (AT §5.5, VU §5.11, owner AT)
  localparam logic [31:0] TSR_AMASK_DST          = 32'hFFFF_FFFF;  // w1
  localparam int unsigned TSR_AMASK_DST_LSB      = 0;
  localparam int unsigned TSR_AMASK_DST_W        = 32;
  localparam int unsigned TSR_AMASK_DST_WORD     = 1;
  localparam int unsigned TSR_AMASK_DST_BIT      = 32;
  localparam logic [31:0] TSR_AMASK_SRC          = 32'hFFFF_FFFF;  // w2 rows at STRIDE
  localparam int unsigned TSR_AMASK_SRC_LSB      = 0;
  localparam int unsigned TSR_AMASK_SRC_W        = 32;
  localparam int unsigned TSR_AMASK_SRC_WORD     = 2;
  localparam int unsigned TSR_AMASK_SRC_BIT      = 64;
  localparam logic [31:0] TSR_AMASK_MASK         = 32'hFFFF_FFFF;  // w3 f32 mask rows
  localparam int unsigned TSR_AMASK_MASK_LSB     = 0;
  localparam int unsigned TSR_AMASK_MASK_W       = 32;
  localparam int unsigned TSR_AMASK_MASK_WORD    = 3;
  localparam int unsigned TSR_AMASK_MASK_BIT     = 96;
  localparam logic [31:0] TSR_AMASK_N_LINES      = 32'h0000_FFFF;  // w4[15:0]
  localparam int unsigned TSR_AMASK_N_LINES_LSB  = 0;
  localparam int unsigned TSR_AMASK_N_LINES_W    = 16;
  localparam int unsigned TSR_AMASK_N_LINES_WORD = 4;
  localparam int unsigned TSR_AMASK_N_LINES_BIT  = 128;
  localparam logic [31:0] TSR_AMASK_STRIDE       = 32'hFFFF_0000;  // w4[31:16]
  localparam int unsigned TSR_AMASK_STRIDE_LSB   = 16;
  localparam int unsigned TSR_AMASK_STRIDE_W     = 16;
  localparam int unsigned TSR_AMASK_STRIDE_WORD  = 4;
  localparam int unsigned TSR_AMASK_STRIDE_BIT   = 144;
  localparam logic [31:0] TSR_AMASK_R            = 32'h0000_FFFF;  // w5[15:0]
  localparam int unsigned TSR_AMASK_R_LSB        = 0;
  localparam int unsigned TSR_AMASK_R_W          = 16;
  localparam int unsigned TSR_AMASK_R_WORD       = 5;
  localparam int unsigned TSR_AMASK_R_BIT        = 160;
  localparam logic [31:0] TSR_AMASK_RPQ          = 32'hFFFF_0000;  // w5[31:16] rows per query (= n_head)
  localparam int unsigned TSR_AMASK_RPQ_LSB      = 16;
  localparam int unsigned TSR_AMASK_RPQ_W        = 16;
  localparam int unsigned TSR_AMASK_RPQ_WORD     = 5;
  localparam int unsigned TSR_AMASK_RPQ_BIT      = 176;
  localparam logic [31:0] TSR_AMASK_MSTRIDE      = 32'hFFFF_FFFF;  // w6 lines between mask rows
  localparam int unsigned TSR_AMASK_MSTRIDE_LSB  = 0;
  localparam int unsigned TSR_AMASK_MSTRIDE_W    = 32;
  localparam int unsigned TSR_AMASK_MSTRIDE_WORD = 6;
  localparam int unsigned TSR_AMASK_MSTRIDE_BIT  = 192;
  localparam logic [31:0] TSR_AMASK_MDST         = 32'hFFFF_FFFF;  // w7 row maxima out
  localparam int unsigned TSR_AMASK_MDST_LSB     = 0;
  localparam int unsigned TSR_AMASK_MDST_W       = 32;
  localparam int unsigned TSR_AMASK_MDST_WORD    = 7;
  localparam int unsigned TSR_AMASK_MDST_BIT     = 224;
  localparam logic [31:0] TSR_AMASK_RSV_W0       = 32'hFFFF_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_AMASK_RSV_W1       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W2       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W3       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W4       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W5       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W6       = 32'h0000_0000;
  localparam logic [31:0] TSR_AMASK_RSV_W7       = 32'h0000_0000;
  localparam logic [7:0] TSR_AMASK_WORDS         = 8'd255;         // words holding a field (w0 always)
  // AMERGE 0x36 (AT §5.6, VU §5.11, owner AT)
  localparam logic [31:0] TSR_AMERGE_INIT         = 32'h0000_1000;  // w0[12] copy instead of merge
  localparam int unsigned TSR_AMERGE_INIT_LSB     = 12;
  localparam int unsigned TSR_AMERGE_INIT_W       = 1;
  localparam int unsigned TSR_AMERGE_INIT_WORD    = 0;
  localparam int unsigned TSR_AMERGE_INIT_BIT     = 12;
  localparam logic [31:0] TSR_AMERGE_ORUN         = 32'hFFFF_FFFF;  // w1 R rows of D/16 lines
  localparam int unsigned TSR_AMERGE_ORUN_LSB     = 0;
  localparam int unsigned TSR_AMERGE_ORUN_W       = 32;
  localparam int unsigned TSR_AMERGE_ORUN_WORD    = 1;
  localparam int unsigned TSR_AMERGE_ORUN_BIT     = 32;
  localparam logic [31:0] TSR_AMERGE_PARTS        = 32'hFFFF_FFFF;  // w2 an ATTPV destination written with LSUM
  localparam int unsigned TSR_AMERGE_PARTS_LSB    = 0;
  localparam int unsigned TSR_AMERGE_PARTS_W      = 32;
  localparam int unsigned TSR_AMERGE_PARTS_WORD   = 2;
  localparam int unsigned TSR_AMERGE_PARTS_BIT    = 64;
  localparam logic [31:0] TSR_AMERGE_MK           = 32'hFFFF_FFFF;  // w3 chunk row maxima
  localparam int unsigned TSR_AMERGE_MK_LSB       = 0;
  localparam int unsigned TSR_AMERGE_MK_W         = 32;
  localparam int unsigned TSR_AMERGE_MK_WORD      = 3;
  localparam int unsigned TSR_AMERGE_MK_BIT       = 96;
  localparam logic [31:0] TSR_AMERGE_ST           = 32'hFFFF_FFFF;  // w4 m_run at ST, l_run at ST + ceil(R/16)
  localparam int unsigned TSR_AMERGE_ST_LSB       = 0;
  localparam int unsigned TSR_AMERGE_ST_W         = 32;
  localparam int unsigned TSR_AMERGE_ST_WORD      = 4;
  localparam int unsigned TSR_AMERGE_ST_BIT       = 128;
  localparam logic [31:0] TSR_AMERGE_R            = 32'h0000_FFFF;  // w5[15:0]
  localparam int unsigned TSR_AMERGE_R_LSB        = 0;
  localparam int unsigned TSR_AMERGE_R_W          = 16;
  localparam int unsigned TSR_AMERGE_R_WORD       = 5;
  localparam int unsigned TSR_AMERGE_R_BIT        = 160;
  localparam logic [31:0] TSR_AMERGE_DCODE        = 32'h0003_0000;  // w5[17:16]
  localparam int unsigned TSR_AMERGE_DCODE_LSB    = 16;
  localparam int unsigned TSR_AMERGE_DCODE_W      = 2;
  localparam int unsigned TSR_AMERGE_DCODE_WORD   = 5;
  localparam int unsigned TSR_AMERGE_DCODE_BIT    = 176;
  localparam logic [31:0] TSR_AMERGE_SCALE        = 32'hFFFF_FFFF;  // w6 f32, as ASMAX
  localparam int unsigned TSR_AMERGE_SCALE_LSB    = 0;
  localparam int unsigned TSR_AMERGE_SCALE_W      = 32;
  localparam int unsigned TSR_AMERGE_SCALE_WORD   = 6;
  localparam int unsigned TSR_AMERGE_SCALE_BIT    = 192;
  localparam logic [31:0] TSR_AMERGE_CSTRIDE      = 32'hFFFF_FFFF;  // w7 as the producing ATTPV
  localparam int unsigned TSR_AMERGE_CSTRIDE_LSB  = 0;
  localparam int unsigned TSR_AMERGE_CSTRIDE_W    = 32;
  localparam int unsigned TSR_AMERGE_CSTRIDE_WORD = 7;
  localparam int unsigned TSR_AMERGE_CSTRIDE_BIT  = 224;
  // AMERGE_DCODE takes the DCODE_* values
  localparam logic [31:0] TSR_AMERGE_RSV_W0 = 32'hFFFF_EF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_AMERGE_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W4 = 32'h0000_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W5 = 32'hFFFC_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W6 = 32'h0000_0000;
  localparam logic [31:0] TSR_AMERGE_RSV_W7 = 32'h0000_0000;
  localparam logic [7:0] TSR_AMERGE_WORDS   = 8'd255;         // words holding a field (w0 always)
  // ANORM 0x37 (AT §5.7, VU §5.11, owner AT)
  localparam logic [31:0] TSR_ANORM_OUT        = 32'hFFFF_FFFF;  // w1 row r at OUT + r x D/16
  localparam int unsigned TSR_ANORM_OUT_LSB    = 0;
  localparam int unsigned TSR_ANORM_OUT_W      = 32;
  localparam int unsigned TSR_ANORM_OUT_WORD   = 1;
  localparam int unsigned TSR_ANORM_OUT_BIT    = 32;
  localparam logic [31:0] TSR_ANORM_ORUN       = 32'hFFFF_FFFF;  // w2
  localparam int unsigned TSR_ANORM_ORUN_LSB   = 0;
  localparam int unsigned TSR_ANORM_ORUN_W     = 32;
  localparam int unsigned TSR_ANORM_ORUN_WORD  = 2;
  localparam int unsigned TSR_ANORM_ORUN_BIT   = 64;
  localparam logic [31:0] TSR_ANORM_L_RUN      = 32'hFFFF_FFFF;  // w3
  localparam int unsigned TSR_ANORM_L_RUN_LSB  = 0;
  localparam int unsigned TSR_ANORM_L_RUN_W    = 32;
  localparam int unsigned TSR_ANORM_L_RUN_WORD = 3;
  localparam int unsigned TSR_ANORM_L_RUN_BIT  = 96;
  localparam logic [31:0] TSR_ANORM_R          = 32'h0000_FFFF;  // w4[15:0]
  localparam int unsigned TSR_ANORM_R_LSB      = 0;
  localparam int unsigned TSR_ANORM_R_W        = 16;
  localparam int unsigned TSR_ANORM_R_WORD     = 4;
  localparam int unsigned TSR_ANORM_R_BIT      = 128;
  localparam logic [31:0] TSR_ANORM_DCODE      = 32'h0003_0000;  // w4[17:16]
  localparam int unsigned TSR_ANORM_DCODE_LSB  = 16;
  localparam int unsigned TSR_ANORM_DCODE_W    = 2;
  localparam int unsigned TSR_ANORM_DCODE_WORD = 4;
  localparam int unsigned TSR_ANORM_DCODE_BIT  = 144;
  // ANORM_DCODE takes the DCODE_* values
  localparam logic [31:0] TSR_ANORM_RSV_W0 = 32'hFFFF_FF00;  // must be 0 (ILLEGAL_OP)
  localparam logic [31:0] TSR_ANORM_RSV_W1 = 32'h0000_0000;
  localparam logic [31:0] TSR_ANORM_RSV_W2 = 32'h0000_0000;
  localparam logic [31:0] TSR_ANORM_RSV_W3 = 32'h0000_0000;
  localparam logic [31:0] TSR_ANORM_RSV_W4 = 32'hFFFC_0000;
  localparam logic [31:0] TSR_ANORM_RSV_W5 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_ANORM_RSV_W6 = 32'hFFFF_FFFF;
  localparam logic [31:0] TSR_ANORM_RSV_W7 = 32'hFFFF_FFFF;
  localparam logic [7:0] TSR_ANORM_WORDS   = 8'd31;          // words holding a field (w0 always)

  /* verilator lint_on UNUSEDPARAM */

endpackage

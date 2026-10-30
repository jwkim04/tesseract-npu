// tsr_pkg.sv — Tesseract core: architectural constants, ISA encoding, CSR map, error codes,
// configuration defaults, helper functions and the internal-interface structs of the core.
//
// C mirror: sw/sim/include/tsr_isa.h (encoding, CSR map, error codes) and tsr_ref.h (QA layout
// helpers). Contract: docs/spec.md and docs/design (EX §5, §7, §11); RTL architecture and interface
// rules: docs/rtl-m2.md, docs/rtl-m3.md. D14 / D16: there is no host command
// path (no instruction fetch, no run registers, no integer registers); LI / ADDI / END are
// reserved opcodes. The offsets, opcodes and units come from the generated tsr_tables_pkg
// (EX-30, Y4); this package keeps what the remaining blocks use.
// Every constant below must match tsr_isa.h exactly; tb/common/isa.py mirrors this file for the
// testbenches and tb/common/test_common.py cross-checks the struct widths.
//
// Width policy: values that the ISA defines as 32-bit (immediate addresses A = imm, lengths,
// strides) are carried as 32-bit fields in the dispatch structs, independent of the elaboration
// parameters. The dispatch checks (tsr_chk, Y4) guarantee that the upper bits a unit does not need
// (e.g. above the SPM line-address width) are zero; units take the low bits they need.

package tsr_pkg;

  // This package is the shared catalogue of ISA/CSR constants: no single module uses all of them,
  // so Verilator's UNUSEDPARAM is disabled for this package only (the only waiver in rtl/core).
  /* verilator lint_off UNUSEDPARAM */

  // -------------------------------------------------------------------------------------------
  // Fixed architecture constants (tsr_isa.h)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_BEAT_BYTES      = 64;   // DRAM beat
  localparam int unsigned TSR_LINE_BYTES      = 64;   // SPM line
  localparam int unsigned TSR_LINE_BITS       = 512;
  localparam int unsigned TSR_INSN_BYTES      = 32;   // one instruction = w0..w7
  localparam int unsigned TSR_INSN_WORDS      = 8;
  localparam int unsigned TSR_INSN_BITS       = 256;
  localparam int unsigned TSR_ROW_PARTIALS    = 16;   // MVM row accumulation partial sums
  localparam int unsigned TSR_RED_PARTIALS    = 128;  // VU reduction partial sums
  localparam int unsigned TSR_MAX_BURST_BEATS = 64;   // a burst never crosses a 4 KiB boundary
  localparam int unsigned TSR_BURST_ALIGN     = 64;   // 4 KiB / 64 B: bursts stay in one 64-beat block
  localparam int unsigned TSR_T_MAX_ARCH      = 8;
  localparam int unsigned TSR_NUM_CH_MAX      = 16;   // LD/ST channel field is 4 bits
  localparam logic [31:0] TSR_ID_MAGIC        = 32'h5453_5231;  // "TSR1"
  // VERSION = the Tesseract Core capability MAJOR.MINOR (EX §7.5), from the generated package.
  localparam logic [7:0]  TSR_ISA_MAJOR       = 8'(tsr_tables_pkg::TSR_ISA_MAJOR);
  localparam logic [7:0]  TSR_ISA_MINOR       = 8'(tsr_tables_pkg::TSR_ISA_MINOR);
  localparam logic [31:0] TSR_CANON_NAN       = 32'h7FC0_0000;

  // Channel DRAM map (byte offsets inside a channel; OPL/IO/AUX on ch0 only; docs/design EX §11).
  localparam longint unsigned TSR_MIB               = 64'd1048576;
  localparam longint unsigned TSR_REGION_OPL_OFF    = 64'd0;          // op lists (was CMD)
  localparam longint unsigned TSR_REGION_OPL_SIZE   = 64 * TSR_MIB;
  localparam longint unsigned TSR_REGION_IO_OFF     = 64 * TSR_MIB;
  localparam longint unsigned TSR_REGION_IO_SIZE    = 256 * TSR_MIB;
  localparam longint unsigned TSR_REGION_AUX_OFF    = 320 * TSR_MIB;
  localparam longint unsigned TSR_REGION_AUX_SIZE   = 192 * TSR_MIB;
  localparam longint unsigned TSR_REGION_WEIGHT_OFF = 512 * TSR_MIB;

  // -------------------------------------------------------------------------------------------
  // Elaboration parameter defaults: U200 (spec 구성 파라미터, TSR_CONFIG_U200).
  // tsr_core takes each of these as a parameter; these are only the defaults.
  // -------------------------------------------------------------------------------------------
  localparam int unsigned     TSR_DEF_NUM_CH         = 4;
  localparam int unsigned     TSR_DEF_BEAT_BITS      = 512;
  localparam int unsigned     TSR_DEF_LANES          = 128;
  localparam int unsigned     TSR_DEF_T_MAX          = 1;
  localparam int unsigned     TSR_DEF_K_MAX          = 32768;
  localparam int unsigned     TSR_DEF_SPM_BYTES      = 4 * 1024 * 1024;
  localparam int unsigned     TSR_DEF_VL             = 16;
  localparam int unsigned     TSR_DEF_NSREG          = 16;
  localparam int unsigned     TSR_DEF_RD_OUTSTANDING = 16;
  localparam longint unsigned TSR_DEF_CH_BYTES       = 64'd17179869184;  // 16 GiB
  localparam int unsigned     TSR_DEF_ADDR_BEAT_W    = 28;
  localparam int unsigned     TSR_DEF_SPM_RD_LAT     = 2;   // tsr_spm client-visible read latency

  // -------------------------------------------------------------------------------------------
  // Memory port encoding (core boundary and tsr_mem_arb source ports; docs/rtl-m2.md §4)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_LEN_W    = 7;   // len_beats as a count, 1..64 (0 never issued)
  localparam int unsigned TSR_SRC_ID_W = 2;   // id inside one source (source-local)
  localparam int unsigned TSR_MEM_ID_W = 4;   // memory-side id = {source[1:0], local[1:0]}
  localparam int unsigned TSR_NSRC     = 3;   // read sources per channel arbiter
  localparam int unsigned TSR_SRC_AUX   = 0;  // freed by the fetch removal; tied off (CC-26)
  localparam int unsigned TSR_SRC_DMA   = 1;  // LD/ST (reads and writes)
  localparam int unsigned TSR_SRC_WS    = 2;  // M3 MVM weight stream (tied off in M2)

  // -------------------------------------------------------------------------------------------
  // SPM client ports of tsr_spm (docs/rtl-m2.md §5)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_SPM_NCL     = 3;
  localparam int unsigned TSR_SPM_CL_DMA  = 0;
  localparam int unsigned TSR_SPM_CL_QACT = 1;
  localparam int unsigned TSR_SPM_CL_MVM  = 2;  // M3 (tied off in M2)
  localparam int unsigned TSR_SPM_BE_W    = TSR_LINE_BYTES;

  // -------------------------------------------------------------------------------------------
  // CSR map (32-bit registers, byte offsets; tsr_isa.h enum tsr_csr). Offsets: docs/design
  // EX §5.1. CTRL / STATUS bits and the rest of the map: EX §5.2-§5.4, generated tsr_tables_pkg (Y4).
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_CSR_AW = 12;
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_ID          = 12'h000;
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_VERSION     = 12'h004;  // major[15:8] minor[7:0]
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_CFG0        = 12'h008;  // NUM_CH[7:0] LANES[19:8] T_MAX[23:20]
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_CFG1        = 12'h00C;  // SPM size in lines
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_CFG2        = 12'h010;  // VL[7:0] NSREG[15:8] K_MAX/256[31:16]
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_CAPS        = 12'h014;
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_CTRL        = 12'h020;  // bits: EX §5.2
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_STATUS      = 12'h024;  // bits: EX §5.3
  localparam logic [TSR_CSR_AW-1:0] TSR_CSR_PERF_BASE   = 12'h100;  // counter i: LO +8i, HI +8i+4

  localparam logic [31:0] TSR_CAPS_VU         = 32'h1;   // bit0
  localparam logic [31:0] TSR_CAPS_TOPK       = 32'h8;   // bit3 (bits 2 and 4: retired, EX §7.5)

  // PERF_* counter slots (tsr_isa.h enum tsr_perf_idx). Per-channel counters take NUM_CH slots.
  localparam int unsigned TSR_PERF_CYCLES       = 0;
  localparam int unsigned TSR_PERF_RETIRED      = 1;
  localparam int unsigned TSR_PERF_MVM_CYCLES   = 2;
  localparam int unsigned TSR_PERF_VU_CYCLES    = 3;
  localparam int unsigned TSR_PERF_RSVD4        = 4;   // reserved (was WAITM), reads 0 (EX §6)
  localparam int unsigned TSR_PERF_CH_RD_BEATS  = 5;   // + ch; CH_WAIT = 5 + NUM_CH + ch

  function automatic int unsigned tsr_perf_count(int unsigned num_ch);
    return TSR_PERF_CH_RD_BEATS + 2 * num_ch;
  endfunction

  // -------------------------------------------------------------------------------------------
  // Error codes (ERR_CODE; the CSR reads the 4-bit code zero-extended)
  // -------------------------------------------------------------------------------------------
  localparam int unsigned TSR_ERR_W = 4;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_NONE         = 4'd0;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_ILLEGAL_OP   = 4'd1;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_BAD_FMT      = 4'd2;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_QA_MISMATCH  = 4'd3;  // reserved: never raised
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_SPM_RANGE    = 4'd4;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_DRAM_RANGE   = 4'd5;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_K_RANGE      = 4'd6;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_T_RANGE      = 4'd7;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_ABORTED      = 4'd8;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_LOOP_NEST    = 4'd9;
  localparam logic [TSR_ERR_W-1:0] TSR_ERR_WAIT_TIMEOUT = 4'd10;

  // -------------------------------------------------------------------------------------------
  // Opcodes and sub-codes (tsr_isa.h; docs/design EX §7.2). 0x01 END, 0x02 LI, 0x03 ADDI: reserved
  // (ILLEGAL_OP); 0x04..0x08 MULI, RLD, LOOP, WAITM, JUMP and 0x09 CALL: retired, never reused.
  // -------------------------------------------------------------------------------------------
  localparam logic [7:0] TSR_OP_NOP   = 8'h00;
  localparam logic [7:0] TSR_OP_LD    = 8'h10;
  localparam logic [7:0] TSR_OP_ST    = 8'h11;
  localparam logic [7:0] TSR_OP_DQROW = 8'h12;
  localparam logic [7:0] TSR_OP_QACT  = 8'h18;
  localparam logic [7:0] TSR_OP_MVM   = 8'h20;
  localparam logic [7:0] TSR_OP_VOP   = 8'h30;
  localparam logic [7:0] TSR_OP_RED   = 8'h31;
  localparam logic [7:0] TSR_OP_ROPE  = 8'h32;
  localparam logic [7:0] TSR_OP_TOPK  = 8'h33;
  localparam logic [7:0] TSR_OP_SOP   = 8'h40;
  localparam logic [7:0] TSR_OP_FENCE = 8'h7F;

  localparam logic [3:0] TSR_FMT_Q8_0 = 4'd0;   // MVM / DQROW sub; 2..15 reserved (BAD_FMT)
  localparam logic [3:0] TSR_FMT_Q4_K = 4'd1;
  localparam logic [3:0] TSR_QA_A32   = 4'd0;   // QACT sub; >= 2 is ILLEGAL_OP
  localparam logic [3:0] TSR_QA_A256  = 4'd1;
  localparam logic [3:0] TSR_MVM_FLAG_ACC = 4'h1;

  // The per-opcode reserved-field rules (x16 = w0[31:16] zero for every opcode that does not
  // assign it, unused words zero; EX §7.1) are checked at dispatch (tsr_chk, Y4).

  // -------------------------------------------------------------------------------------------
  // Instructions (256 bits, w0..w7 little-endian; EX §7.1). insn[i] is word wi; insn[0][7:0] is
  // byte 0 of the 32-byte instruction. w0 = {x16[31:16], flags[15:12], sub[11:8], opcode[7:0]}.
  // -------------------------------------------------------------------------------------------
  typedef logic [TSR_INSN_WORDS-1:0][31:0] tsr_insn_t;

  function automatic logic [7:0]  tsr_w0_opcode(logic [31:0] w0); return 8'(w0 >> 0);   endfunction
  function automatic logic [3:0]  tsr_w0_sub(logic [31:0] w0);    return 4'(w0 >> 8);   endfunction
  function automatic logic [3:0]  tsr_w0_flags(logic [31:0] w0);  return 4'(w0 >> 12);  endfunction
  function automatic logic [15:0] tsr_w0_x16(logic [31:0] w0);    return 16'(w0 >> 16); endfunction

  // 1 iff words w[from..7] are all zero.
  function automatic logic tsr_words_zero_from(tsr_insn_t insn, int unsigned from);
    logic z;
    z = 1'b1;
    for (int unsigned i = 0; i < TSR_INSN_WORDS; i++) begin
      if (i >= from && insn[i] != 32'd0) z = 1'b0;
    end
    return z;
  endfunction

  // -------------------------------------------------------------------------------------------
  // Configuration helpers (elaboration-time)
  // -------------------------------------------------------------------------------------------
  // Addressable beats per channel: min(CH_BYTES / 64, 2^ADDR_BEAT_W) (spec 구현 명확화, 범위 검사).
  function automatic longint unsigned tsr_dram_limit(longint unsigned ch_bytes, int unsigned addr_beat_w);
    longint unsigned by_bytes, by_width;
    by_bytes = ch_bytes / 64;
    by_width = (addr_beat_w >= 63) ? 64'h7FFF_FFFF_FFFF_FFFF : (64'd1 << addr_beat_w);
    return (by_bytes < by_width) ? by_bytes : by_width;
  endfunction

  function automatic int unsigned tsr_spm_lines(int unsigned spm_bytes);
    return spm_bytes / TSR_LINE_BYTES;
  endfunction

  // Line-address width of an SPM with `lines` lines (at least 1 bit).
  function automatic int unsigned tsr_spm_aw(int unsigned lines);
    return (lines > 1) ? $clog2(lines) : 1;
  endfunction

  // -------------------------------------------------------------------------------------------
  // QA layout (spec 양자화 활성값 레이아웃; tsr_ref.c tsr_qa_*). kind: 0 = A32, 1 = A256.
  // K must be a multiple of the block size (the dispatch checks K_RANGE first); K <= 65280.
  // -------------------------------------------------------------------------------------------
  function automatic logic [31:0] tsr_qa_block_size(logic kind);
    return kind ? 32'd256 : 32'd32;
  endfunction
  function automatic logic [31:0] tsr_qa_qs_lines(logic [31:0] k);             // ceil(K / 64)
    return (k + 32'd63) >> 6;
  endfunction
  function automatic logic [31:0] tsr_qa_d_lines(logic kind, logic [31:0] k);  // ceil(4 * nblk / 64)
    logic [31:0] nblk;
    nblk = kind ? (k >> 8) : (k >> 5);
    return (nblk + 32'd15) >> 4;
  endfunction
  function automatic logic [31:0] tsr_qa_bs_lines(logic kind, logic [31:0] k); // A256: ceil(2 * K/16 / 64)
    return kind ? (((k >> 4) + 32'd31) >> 5) : 32'd0;
  endfunction
  function automatic logic [31:0] tsr_qa_lines(logic kind, logic [31:0] k);    // QA_LINES(kind, K)
    return tsr_qa_qs_lines(k) + tsr_qa_d_lines(kind, k) + tsr_qa_bs_lines(kind, k);
  endfunction
  // Lines of one f32 source vector of length K (K % 16 == 0 for every legal QACT K).
  function automatic logic [31:0] tsr_f32_lines(logic [31:0] k);
    return (k + 32'd15) >> 4;
  endfunction

  // ROW_BEATS(fmt, K) (spec 행 크기; tsr_ref.c tsr_row_beats). fmt: 0 = Q8_0, 1 = Q4_K. M3.
  function automatic logic [31:0] tsr_row_beats(logic fmt, logic [31:0] k);
    logic [31:0] nb, full, rem;
    if (!fmt) begin               // Q8_0: S = 32 blocks, chunk = 1 + 16 beats; partial 1 + ceil(rem/2)
      nb   = k >> 5;
      full = nb >> 5;
      rem  = nb & 32'd31;
      return full * 32'd17 + ((rem != 0) ? (32'd1 + ((rem + 32'd1) >> 1)) : 32'd0);
    end else begin                // Q4_K: S = 4 sb, chunk = 1 + 8 beats; partial 1 + 2*rem
      nb   = k >> 8;
      full = nb >> 2;
      rem  = nb & 32'd3;
      return full * 32'd9 + ((rem != 0) ? (32'd1 + (rem << 1)) : 32'd0);
    end
  endfunction

  // -------------------------------------------------------------------------------------------
  // Internal interfaces (docs/rtl-m2.md §3). Packed: the first member is the MSB.
  // tb/common/isa.py keeps the same field order and widths (checked by tb/common/test_common.py).
  // The dispatcher (tsr_chk + tsr_issue, CC §13, Y4) offers them to the units; address operands
  // are the immediates (A1 = w1, A2 = w2, A3 = w3; EX §7.1).
  // -------------------------------------------------------------------------------------------

  // dispatch -> tsr_dma: LD / ST with every operand checked and resolved.
  typedef struct packed {
    logic        is_st;      // 0: LD DRAM -> SPM, 1: ST SPM -> DRAM
    logic [3:0]  ch;         // channel (< NUM_CH)
    logic [31:0] spm_line;   // A1 = w1; spm_line + n_lines <= SPM lines
    logic [31:0] dram_beat;  // A2 = w2; dram_beat + n_lines <= DRAM limit
    logic [31:0] n_lines;    // w3, >= 1 (n_lines == 0 completes without a dispatch)
  } tsr_dma_cmd_t;           // 101 bits

  // dispatch -> tsr_qact: QACT with every operand checked and resolved.
  typedef struct packed {
    logic        kind;        // 0: A32, 1: A256
    logic [31:0] dst_line;    // A1; dst_line + t * qa_lines <= SPM lines
    logic [31:0] src_line;    // A2; src_line + (t-1) * src_stride + k/16 <= SPM lines
    logic [15:0] k;           // 0 < K <= K_MAX, multiple of 32 (A32) / 256 (A256)
    logic [3:0]  t;           // 1 <= T <= T_MAX
    logic [31:0] src_stride;  // lines between token sources; w5 == 0 already replaced by k/16
    logic [15:0] qa_lines;    // QA_LINES(kind, K): dst stride between tokens
  } tsr_qact_cmd_t;           // 133 bits

  // dispatch -> MVM unit (M3; HAS_MVM = 0 builds have no MVM unit).
  typedef struct packed {
    logic        fmt;         // 0: Q8_0 (A32 QA), 1: Q4_K (A256 QA)
    logic        acc;         // flags.ACC
    logic [31:0] dst_line;    // A1
    logic [31:0] x_line;      // A2
    logic [31:0] w_off;       // A3 (beat offset, same in every channel)
    logic [31:0] n;           // N rows (0 legal: no traffic, nothing written)
    logic [15:0] k;
    logic [3:0]  t;
    logic [31:0] dst_stride;  // resolved (w6 == 0 -> y_lines)
    logic [31:0] y_lines;     // ceil(4N / 64)
    logic [31:0] rows_per_ch; // R = ceil(N / NUM_CH)
    logic [31:0] row_beats;   // ROW_BEATS(fmt, K)
    logic [15:0] qa_lines;    // QA_LINES(kind(fmt), K): x stride between tokens
  } tsr_mvm_cmd_t;            // 294 bits

  // ===========================================================================================
  // M3: MVM unit (docs/rtl-m3.md). Everything below is additive; M2 users are unaffected.
  // ===========================================================================================

  // ---- architecture constants of the MVM datapath (spec 디바이스 가중치 레이아웃, 수치 규칙) ----
  localparam int unsigned TSR_MVM_LANES       = 128;  // weight lanes per beat (= LANES, U200)
  localparam int unsigned TSR_MVM_GROUPS      = 4;    // 32-lane groups per beat (one sum each)
  localparam int unsigned TSR_MVM_GROUP_LANES = 32;   // lanes summed per group (block / sub-block)
  localparam int unsigned TSR_MVM_SUM_W       = 21;   // signed group sum: |sum| <= 32*128*128 = 2^19
  localparam int unsigned TSR_MVM_ISUM_W      = 26;   // signed Q4_K isum / msum (any SPM content:
                                                      // |msum| <= 8*63*2*32768 < 2^25)
  localparam int unsigned TSR_Q8_0_CHUNK_BLOCKS = 32; // S
  localparam int unsigned TSR_Q8_0_CHUNK_BEATS  = 17; // 1 meta + 16 quant
  localparam int unsigned TSR_Q4_K_CHUNK_SB     = 4;  // S
  localparam int unsigned TSR_Q4_K_CHUNK_BEATS  = 9;  // 1 meta + 8 quant
  localparam int unsigned TSR_MVM_BEATS_W     = 40;   // per-channel weight beats (<= 2^40 - 1, docs/rtl-m3.md §3.2)
  localparam int unsigned TSR_MVM_QS_LINE_W   = 10;   // QS line index inside one QA image (K <= 65280)
  localparam int unsigned TSR_MVM_BLK_W       = 11;   // block / superblock index inside a row

  // ---- implementation-parameter defaults of the MVM unit (docs/rtl-m3.md §1) ----
  localparam int unsigned TSR_DEF_WS_FIFO_BEATS = 512; // weight FIFO per cluster (power of two >= 64)
  localparam int unsigned TSR_DEF_MVM_BC_PIPE   = 1;   // broadcast pipe stages top -> cluster
  localparam int unsigned TSR_DEF_MVM_RES_PIPE  = 1;   // result pipe stages cluster -> top (rslices)
  localparam int unsigned TSR_DEF_ACT_RD_LAT    = 2;   // activation buffer read latency
  localparam int unsigned TSR_DEF_MAC_LAT       = 3;   // tsr_mac_lane latency
  localparam int unsigned TSR_DEF_FMUL_LAT      = 4;   // tsr_fp_mul latency (its 300 MHz default)
  localparam int unsigned TSR_DEF_FADD_LAT      = 5;   // tsr_fp_add latency (= its default, <= 8; tb/fp test_fp_default_lats)
  localparam int unsigned TSR_DEF_WS_RD_LAT     = 2;   // weight FIFO RAM read latency (tsr_mvm_ws, 1..4;
                                                       // 2 = BRAM output register at 300 MHz)

  // ---- layout helpers (sw/sim/tsr_ref.c tsr_qa_*, tsr_row_beats, tsr_rows_per_ch) ----
  // Activation buffer geometry for K_MAX (docs/rtl-m3.md §5): QS words of 1024 bits (two QS
  // lines) per token, and D/BS lines of 512 bits per token (max over both QA kinds).
  function automatic int unsigned tsr_actbuf_qs_words(int unsigned k_max);
    return (k_max + 127) / 128;
  endfunction
  function automatic int unsigned tsr_actbuf_dbs_lines(int unsigned k_max);
    int unsigned a32, a256;
    a32  = (4 * (k_max / 32) + 63) / 64;                                  // A32 D lines
    a256 = (4 * (k_max / 256) + 63) / 64 + (2 * (k_max / 16) + 63) / 64;  // A256 D + BS lines
    return (a32 > a256) ? a32 : a256;
  endfunction
  // Rows of channel c: max(0, min((c+1)R, N) - cR) for first = c*R (64-bit: c*R may exceed 2^32).
  function automatic logic [31:0] tsr_mvm_rows_c(logic [31:0] n, logic [31:0] r, logic [63:0] first);
    logic [63:0] left;
    if ({32'd0, n} <= first) return 32'd0;
    left = {32'd0, n} - first;
    return (left < {32'd0, r}) ? left[31:0] : r;
  endfunction

  // ---- internal interfaces of the MVM unit (docs/rtl-m3.md §3). Packed: first member = MSB ----

  // tsr_mvm -> every cluster (on the broadcast pipe, once per MVM with N >= 1, before its lines):
  // the MVM fields every cluster needs. Each cluster derives its own rows from n / rows_per_ch
  // and its channel index (first row CH*R, rows_c = tsr_mvm_rows_c).
  typedef struct packed {
    logic        fmt;          // 0: Q8_0 (A32 QA), 1: Q4_K (A256 QA)
    logic [15:0] k;            // K: multiple of the block size, <= K_MAX
    logic [31:0] n;            // N >= 1
    logic [31:0] rows_per_ch;  // R = ceil(N / NUM_CH) >= 1
    logic [15:0] row_beats;    // ROW_BEATS(fmt, K) (<= 1084)
    logic [31:0] w_off;        // first beat of the tensor, same offset in every channel
    logic [7:0]  d_lines;      // QA D-region lines ceil(4*(K/bs)/64): BS lines follow them
  } tsr_mvm_ccmd_t;            // 137 bits

  // tsr_mvm -> every cluster (broadcast pipe): one line of the QA images, in SPM order
  // (token 0 lines 0..QA_LINES-1, token 1, ...). The activation buffer stores it at
  // (tok, dbs, addr); `last` marks the final line of the MVM's broadcast.
  typedef struct packed {
    logic         last;        // last broadcast line of this MVM
    logic [2:0]   tok;         // token t < T
    logic         dbs;         // 0: QS line, 1: D or BS line
    logic [9:0]   addr;        // QS: line l < ceil(K/64); D/BS: l - ceil(K/64) (D lines, then BS)
    logic [511:0] data;        // the SPM line
  } tsr_mvm_bcline_t;          // 527 bits

  // cluster wrapper -> front end (one per MVM, only when the channel has rows).
  typedef struct packed {
    logic        fmt;          // 0: Q8_0, 1: Q4_K
    logic [15:0] k;            // K
    logic [15:0] row_beats;    // ROW_BEATS(fmt, K)
    logic [31:0] w_off;        // first beat of row 0 of this channel (= MVM W_OFF)
    logic [31:0] rows;         // rows_c >= 1
    logic [TSR_MVM_BEATS_W-1:0] beats;  // rows_c * ROW_BEATS: exactly the beats to request
  } tsr_mvm_fcmd_t;            // 137 bits

  // front end -> back end: one QUANT beat, unpacked and tagged (META beats are absorbed by the
  // front end). Term indices are inside the row: Q8_0 block b, Q4_K superblock s.
  typedef struct packed {
    logic        last;         // last beat of this cluster's MVM (implies row_last)
    logic        row_last;     // last beat of a row
    logic [TSR_MVM_BLK_W-1:0] blk;  // Q8_0: b of the beat's first block (even), terms b (and b+1);
                                    // Q4_K: superblock s
    logic        two;          // Q8_0: block b+1 is present (0 only in the last beat of a partial
                               //   chunk with an odd block count); Q4_K: 0
    logic        half;         // Q4_K: 0 = sub-blocks 0..3 of s, 1 = sub-blocks 4..7 (term of s
                               //   completes); Q8_0: 0
    logic [31:0] d_a;          // f32 (exact fp16 -> f32): Q8_0 d of block b; Q4_K d of s
    logic [31:0] d_b;          // f32: Q8_0 d of block b+1 (+0 if !two); Q4_K dmin of s
    logic [TSR_MVM_GROUPS-1:0][5:0] sc;  // Q4_K: sc_(4*half+g) of lane group g; Q8_0: 0
    logic [7:0][5:0]  m;       // Q4_K: m_0..m_7 of s (same in both halves); Q8_0: 0
    logic [TSR_MVM_LANES-1:0][7:0] w;    // signed weight lanes, lane L <-> K position
                               //   Q8_0: 32b + L (L < 64; lanes 32..63 = 0 if !two; 64..127 = 0)
                               //   Q4_K: 256s + 128*half + L (values 0..15)
  } tsr_mvm_beat_t;            // 1175 bits

  /* verilator lint_on UNUSEDPARAM */

endpackage

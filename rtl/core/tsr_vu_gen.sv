// tsr_vu_gen.sv — the VU's slot generator: expands one op at a time into its slots (one output line
// position each) and pseudo-slots (c-line loads, ROPE table / position loads), in the order the
// injector consumes them, with the credits each slot takes.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §4.2 operand model, §5 encodings, §7.7-§7.8 sequencer;
// AT §5.4-§5.7 for AT's vector ops on the VU).
// Interface:
//   start / start_tag / start_w   begin an op (the top's hazard check passed); the s bank is sampled in
//                                 that cycle (VU §4.4: after every earlier VU op's s write).
//   opk[tag]                      the op's constants for the injector and tsr_vu_red, written at start.
//   rope_*                        the ROPE configuration for tsr_vu_angle, valid from rope_start.
//   slot_valid / slot_ready / slot   R2: the slot register holds until taken. A slot is loaded only with
//                                 the credits it needs (oc_free: output FIFO lines, rc_free: RED FIFO
//                                 lines); slot_take pulses in the loading cycle with slot_oc / slot_rc /
//                                 slot_cnt (the SPM / s writes it causes, for the top's done count) /
//                                 slot_lmw (it writes the local memory).
//   gen_done / gen_done_tag       the op's last slot has been loaded (pulse; ops without slots too).
//   lm_idle                       no local-memory write in flight: the barrier between dependent passes
//                                 (ANORM: the 1/l prepass before the main pass, VU §7.4).
// Slot order per op (r = row, j = line in the row; every address is a counter, no multiplier on the
// slot path):
//   VOP    rows × L(m): [c-line of rows r..r+15 (cmode 2) at j = 0, r mod 16 = 0], slot (r, j): a + r·as + j,
//          b + r·bs + j, d + r·ds + j, tail = m mod 16 on the last line
//   RED    rows × L(m) slots into the partials (row token at j = L - 1); TO_S with m = 0: one empty-row
//          slot. Output: one RED-FIFO line per 16 rows, or one s write
//   ROPE   [table lines (FF / ATAB)], position lines 0 and 1, then per token t: [position line t/16 + 1 at
//          t mod 16 = 0], per head h: NEOX pair slots j < n_dims/32 (two output lines each: j and
//          j + n_dims/32), NORM slots o < n_dims/16, raw copies of the lines past n_dims (VU §6.7)
//   SOP    one slot: lane 0 functions with a = k1, b = k2 (constants), MOVI as SETC, LDS / STS through
//          COPY; SQRT, RSQRT, DIV as a scalar-path pseudo-slot (tsr_vu_red's post unit, the divider)
//   ASMAX  R × n_lines with the mvec c-line per 16 rows (AT §5.4)
//   AMASK  R × n_lines, b = mask row floor(r / rpq), MAX into the partials from M_FLOOR (AT §5.5)
//   ANORM  1/l prepass (ceil(R/16) RCPMAX slots into the local memory), barrier, R × D/16 MULC slots with
//          the 1/l c-line from the local memory (AT §5.7)
//   TOPK   rows × L(m) lines to tsr_vu_topk (ceil(k/8) result lines per row from there)
//   DQROW  per row, per chunk of 64 output lines: the chunk's meta line (SK_CL2 with dq, starts
//          tsr_vu_dq), then per output line j the quant line holding it (Q8_0 chunk line 1 + (j mod 64)/4,
//          bytes 16·(j mod 4); Q4_K line 1 + (j mod 64)/8, bytes 16·{j[2], j[0]}, nibble j[1]); chunk
//          bases advance by 17 (Q8_0) / 9 (Q4_K) lines (tsr_row_beats)
//   GDN    CFG slot (m, rows, o line), the lines of k, q and v (vector slots, 4 injection cycles each), the m / 16
//          lines of k again (the pseudo row), then rows × m / 16 lines of S (row slots, one cycle each; a row's first
//          line takes a row slot of the engine); the writes (S' lines, o lines) come from the engine (VU §7.12.5)
//   AMERGE: see the op FSM below.

module tsr_vu_gen
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter int unsigned FRONT_LANES = 16,
  parameter int unsigned EXP_LANES  = 16,
  parameter int unsigned BACK_LANES = 8,
  parameter int unsigned NUM_CH     = 4,
  parameter int unsigned LM_X       = 1,         // local memory: 512·LM_X lines (2 for NUM_CH > 8)
  parameter bit          HAS_TOPK   = 1'b0,      // TOPK (CAPS bit3; RTL from M8, VU §5.5)
  parameter bit          HAS_DQROW  = 1'b0,      // DQROW (CAPS bit6; RTL from M8, VU §5.6)
  parameter bit          HAS_GDN    = 1'b0       // GDN (CAPS bit8, VU §7.12)
) (
  input  logic                clk,
  input  logic                rst_n,
  // op start
  input  logic                start,
  input  logic                start_tag,
  input  tsr_insn_t           start_w,
  input  logic [15:0][31:0]   sreg,
  output logic                busy,
  output logic                gen_done,
  output logic                gen_done_tag,
  output tsr_vu_opk_t [1:0]   opk,
  // ROPE configuration (tsr_vu_angle)
  output logic                rope_start,
  output logic                rope_neox,
  output logic                rope_ff,
  output logic                rope_fs,
  output logic                rope_atab,
  output logic [31:0]         rope_m,
  output logic [31:0]         rope_ts,
  output logic [31:0]         rope_fsv,
  output logic [9:0]          rope_na,     // n_dims / 2 angles per token
  output logic [15:0]         rope_t,      // tokens T'
  // slots
  output logic                slot_valid,
  input  logic                slot_ready,
  output tsr_vu_slot_t        slot,
  output logic                slot_take,
  output logic                slot_take_tag,   // tag of the slot being loaded
  output logic [1:0]          slot_oc,
  output logic                slot_rc,
  output logic [1:0]          slot_cnt,
  output logic                slot_lmw,
  input  logic [7:0]          oc_free,     // output FIFO lines free (saturated)
  input  logic [3:0]          rc_free,     // RED FIFO lines free (saturated)
  input  logic                lm_idle
);
  import tsr_tables_pkg::*;

  localparam int unsigned RF = 16 / FRONT_LANES;
  localparam int unsigned RB = 16 / BACK_LANES;   // sub-slots of a back-lane function
  localparam int unsigned RE = 16 / EXP_LANES;    // sub-slots of an EXPK front function
  localparam logic [TSR_VU_LW-1:0] LM_RINV = '0;  // ANORM: 1/l vector at local-memory line 0
  // AMERGE local-memory layout (lines): a, b and m' vectors (<= 64 lines each, R <= 1024), and the
  // channel-tree scratch: prepass slots of 64 lines, main-phase slots of NB lines.
  localparam logic [TSR_VU_LW-1:0] LM_A   = TSR_VU_LW'(0);
  localparam logic [TSR_VU_LW-1:0] LM_B   = TSR_VU_LW'(64);
  localparam logic [TSR_VU_LW-1:0] LM_M   = TSR_VU_LW'(128);
  localparam logic [TSR_VU_LW-1:0] LM_SCR = TSR_VU_LW'(256);
  localparam int unsigned NSLOT = (NUM_CH / 2 > 0) ? NUM_CH / 2 : 1;      // scratch slots of a level
  localparam int unsigned NB    = ((256 * LM_X) / NSLOT > 64) ? 64 : (256 * LM_X) / NSLOT;
  localparam int unsigned PSL   = (NUM_CH > 8) ? 128 : 64;               // prepass slot lines (LM_X = 2)

  // CTREE over NUM_CH channels (AT §5.6: pairwise by levels, left to right, the unpaired last item
  // carried): pass p adds items a and b (channel blocks in the SPM or scratch slots) into slot d; bar
  // marks the first pass of a level >= 1, which waits for the previous level's writes.
  typedef struct packed {
    logic       a_scr;
    logic [3:0] a_i;
    logic       b_scr;
    logic [3:0] b_i;
    logic [3:0] d;
    logic       bar;
  } ctp_t;
  localparam int unsigned NP = NUM_CH - 1;
  function automatic logic [15:0][14:0] ct_plan();
    logic [15:0]      sc;
    logic [15:0][3:0] ix;
    logic [15:0][14:0] pl;
    int n, p, lvl;
    pl = '0;
    for (int k = 0; k < 16; k++) begin
      sc[k] = 1'b0;
      ix[k] = 4'(k);
    end
    n = NUM_CH;
    p = 0;
    lvl = 0;
    while (n > 1) begin
      for (int k = 0; k < n / 2; k++) begin
        pl[p] = {sc[2*k], ix[2*k], sc[2*k+1], ix[2*k+1], 4'(k), (lvl > 0) && (k == 0)};
        p = p + 1;
      end
      if (n % 2 == 1) begin
        sc[n/2] = sc[n-1];
        ix[n/2] = ix[n-1];
      end
      for (int k = 0; k < n / 2; k++) begin
        sc[k] = 1'b1;
        ix[k] = 4'(k);
      end
      n = (n + 1) / 2;
      lvl = lvl + 1;
    end
    return pl;
  endfunction
  localparam logic [15:0][14:0] CTP = ct_plan();
  localparam int unsigned CTF_I = 0;                     // the tree's result: scratch slot 0 (NUM_CH >= 2)

  if (!((EXP_LANES == 16 || EXP_LANES == 8) && (BACK_LANES == 8 || BACK_LANES == 4) &&
        BACK_LANES <= EXP_LANES && EXP_LANES <= FRONT_LANES &&
        (FRONT_LANES == 16 || FRONT_LANES == 8))) begin : g_chk_lanes
    $fatal(1, "tsr_vu_gen: EXP_LANES 16 or 8, BACK_LANES 8 or 4 (VU §7.10 knobs)");
  end

  function automatic logic [2:0] nsub_of(input tsr_vu_uop_e u);
    if (tsr_vu_uop_back(u)) return 3'(RB);
    if (tsr_vu_uop_exp(u))  return 3'(RE);
    return 3'(RF);
  endfunction

  // -------------------------------------------------------------------------------------------
  // Op FSM state
  // -------------------------------------------------------------------------------------------
  typedef enum logic [3:0] {
    G_IDLE, G_SETUP, G_PASS, G_SOP, G_REDE, G_RTBL, G_RPOS0, G_RPOS1, G_RTOK, G_RROT, G_BAR, G_FIN, G_AM,
    G_GSEQ, G_GSET2
  } gst_e;
  // AMERGE sub-states (AT §5.6): prepass tree over the LSUM vectors, m', a, b, l_run, m_run, then batches of
  // the main tree and the combine
  typedef enum logic [3:0] {
    AM_PRE, AM_M, AM_BAR1, AM_A, AM_B, AM_BAR2, AM_L, AM_MR, AM_BARI, AM_LI, AM_MI, AM_MT, AM_MBAR, AM_MC
  } am_e;
  am_e         am_q;
  logic [3:0]  amp_q;                  // tree pass
  logic [15:0] ami_q;                  // line (prepass: vector line; main: output line)
  logic [15:0] amb_q;                  // main-phase batch start
  logic [15:0] ambe_q;                 // its last line: min(amb + NB, R·D/16) - 1 (set with amb_q)
  logic [1:0]  amcl_q;                 // c-lines loaded for the current line / group (0, 1: CL1, 2: both)
  logic [15:0] amg_q;                  // group of the loaded c-lines (main phase), 16'hFFFF none
  logic [15:0] am_R, am_Rl, am_NL;     // R, ceil(R/16), R·D/16
  logic [2:0]  am_dsh;                 // log2(D/16)
  logic [31:0] am_cs;                  // resolved cstride
  logic [TSR_VU_LW-1:0] am_orun, am_parts, am_lparts, am_mk, am_st;
  logic        am_init;
  // AMERGE pass bases (registered from CTP[amp_q], valid 3 cycles after amp_q or the op changes):
  // prepass items a / b and destination, main-phase items a / b and destination
  logic [14:0]          pe_q;
  logic [3:0]           amp_d_q;
  logic [1:0]           bset_q;
  logic                 bases_ok;        // the AMERGE pass bases are settled (assigned with them below)
  logic [TSR_VU_LW-1:0] pm_a_q, pm_b_q, pp_a_q, pp_b_q;        // products i·am_Rl (prepass), i·am_cs (main)
  logic [TSR_VU_LW-1:0] bpa_q, bpb_q, bpd_q, bma_q, bmb_q, bmd_q;
  typedef enum logic [3:0] { P_VOP, P_RED, P_ASMAX, P_AMASK, P_ANORM1, P_ANORM2, P_TOPK, P_DQ, P_GPS, P_GROW } pass_e;

  gst_e        st_q;
  gst_e        setup_next;             // the state after G_SETUP (decoded below)
  pass_e       pk_q;
  logic        tag_q;
  tsr_insn_t   w_q;
  logic [7:0]  op_q;

  // pass geometry
  logic [15:0] rows_q;                 // rows of the pass (>= 1 in G_PASS)
  logic [23:0] nl_q;                   // lines per row (>= 1)
  logic [4:0]  tail_q;                 // words of the last line of a row
  logic [15:0] r_q;
  logic [23:0] j_q;
  logic [TSR_VU_LW-1:0] ab_q, bb_q, db_q, cb_q;   // row bases, c-line base
  logic [15:0] as_q, bs_q, ds_q;       // row strides (lines)
  logic [31:0] bsl_q;                  // AMASK mask row stride (32-bit field)
  logic [15:0] rpq_q, qc_q;            // AMASK rows per query, count in the current query
  tsr_vu_uop_e uop_q;
  logic [2:0]  nsub_q;
  logic        ra_q, rbq_q;            // the pass reads a / b
  tsr_vu_csel_e c1s_q, c2s_q;
  logic        cl_q;                   // a c-line per 16 rows
  tsr_vu_src_e cls_q;                  // its source
  logic [3:0]  ng_q;                   // RED groups - 1
  logic        cl_done_q;              // the c-line of the current 16-row group was emitted
  tsr_vu_dst_e dst_q;
  logic        dqf_q;                  // DQROW format: 1 Q4_K

  // GDN: the head command and vector slots (G_GSEQ), then the pseudo row and the rows (G_PASS)
  logic [1:0]  gp_q;                   // phase: 0 CFG, 1 k, 2 q, 3 v
  logic [15:0] gj_q;                   // line in the phase

  // ROPE
  logic [15:0] hd_q, nd_q, nh_q, tp_q; // head_dim, n_dims, n_heads, T'
  logic [15:0] t_q, h_q;
  logic [23:0] o_q;                    // line / pair index in the head
  logic [31:0] hl_q;                   // lines per head: head_dim / 16
  logic [31:0] H_q, ssp_q, dsp_q;      // lines per token, resolved strides
  logic [TSR_VU_LW-1:0] sbt_q, dbt_q, sbh_q, dbh_q;   // token / head bases
  logic [15:0] tbl_n_q, tbl_i_q;       // table lines
  logic [15:0] posl_q;                 // position lines ceil(T'/16)
  logic [23:0] nrot_q, npm1_q;         // rotation slots per head; slots per head - 1 (decoded at setup)

  // -------------------------------------------------------------------------------------------
  // Field decode of the captured words (VU §5, AT §5)
  // -------------------------------------------------------------------------------------------
  wire [3:0]  sub = w_q[0][11:8];
  wire [15:0] x16 = w_q[0][31:16];

  // -------------------------------------------------------------------------------------------
  // Next slot (combinational from the state) and whether it is the op's last
  // -------------------------------------------------------------------------------------------
  tsr_vu_slot_t nxt;
  logic         nxt_v, nxt_last;
  logic [1:0]   nxt_oc, nxt_cnt;
  logic         nxt_rc, nxt_lmw;
  logic         nxt_cl;                // the next slot is the c-line pseudo-slot of its group

  wire last_line = (j_q == nl_q - 24'd1);
  wire last_row  = (r_q == rows_q - 16'd1);

  always_comb begin
    logic [15:0] k;
    logic        neox, rot;
    logic [15:0] i_in_b, g, rr;
    logic [TSR_VU_LW-1:0] la, lb, lk, ok;
    i_in_b   = '0;
    g        = '0;
    rr       = '0;
    la       = '0;
    lb       = '0;
    lk       = '0;
    ok       = '0;
    k        = '0;
    neox     = 1'b0;
    rot      = 1'b0;
    nxt      = '0;
    nxt.tag  = tag_q;
    nxt_v    = 1'b0;
    nxt_last = 1'b0;
    nxt_oc   = 2'd0;
    nxt_cnt  = 2'd0;
    nxt_rc   = 1'b0;
    nxt_lmw  = 1'b0;
    nxt_cl   = 1'b0;
    case (st_q)
      G_PASS: begin
        nxt_v = 1'b1;
        if (cl_q && !cl_done_q && j_q == 24'd0 && r_q[3:0] == 4'd0) begin
          // c-line of rows r..r+15 (VOP cmode 2: c + r/16; ASMAX mvec; ANORM 1/l in the local memory)
          nxt_cl     = 1'b1;
          nxt.kind   = SK_CL1;
          nxt.asrc   = cls_q;
          nxt.aline  = cb_q + TSR_VU_LW'(r_q[15:4]);
        end else if (pk_q == P_DQ && !cl_done_q && j_q[5:0] == 6'd0) begin
          // DQROW: the chunk's meta line
          nxt_cl    = 1'b1;
          nxt.kind  = SK_CL2;
          nxt.asrc  = SRC_SPM;
          nxt.aline = cb_q;
          nxt.dq    = {1'b1, dqf_q, 4'd0};
        end else if (pk_q == P_DQ) begin
          // DQROW output line j of row r (VU §6.8, tsr_vu_dq)
          nxt.kind  = SK_LANE;
          nxt.uop   = uop_q;
          nxt.nsub  = 3'(RF);
          nxt.asrc  = SRC_SPM;
          nxt.aline = cb_q + TSR_VU_LW'(1) + (dqf_q ? TSR_VU_LW'(j_q[5:3]) : TSR_VU_LW'(j_q[5:2]));
          nxt.dst   = DS_SPM;
          nxt.d0    = db_q + TSR_VU_LW'(j_q);
          nxt.tail  = 5'd16;
          nxt.dq    = {1'b1, dqf_q, (dqf_q ? {j_q[2], j_q[0]} : j_q[1:0]), j_q[1], 1'b0};
          nxt.dqb   = j_q[5:1];
          nxt_oc    = 2'd1;
          nxt_cnt   = 2'd1;
          nxt_last  = last_line && last_row;
        end else if (HAS_GDN && (pk_q == P_GPS || pk_q == P_GROW)) begin
          // GDN rows (VU §7.12.5): the lines of the pseudo row (k again) and of S go to the engine, which writes S'
          // and o itself. A line takes no output place here (the engine takes them when a row is launched); its
          // S' line counts as a write of the op, and the first row of each 16 also counts the o line.
          nxt.kind    = SK_GDN;
          nxt.gsub    = GS_ROW;
          nxt.nsub    = 3'd1;                                  // a whole line to the engine
          nxt.asrc    = SRC_SPM;
          nxt.aline   = ab_q + TSR_VU_LW'(j_q);
          nxt.d0      = db_q + TSR_VU_LW'(j_q);
          nxt.tail    = 5'd16;
          nxt.r_first = (j_q == 24'd0);
          nxt_cnt     = (pk_q == P_GROW) ? ((j_q == 24'd0 && r_q[3:0] == 4'd0) ? 2'd2 : 2'd1) : 2'd0;
          nxt_last    = (pk_q == P_GROW) && last_line && last_row;
        end else if (pk_q == P_TOPK) begin
          // TOPK (VU §6.8): the source lines to tsr_vu_topk; ceil(k/8) result lines per row
          nxt.kind    = SK_TOPK;
          nxt.asrc    = SRC_SPM;
          nxt.aline   = ab_q + TSR_VU_LW'(j_q);
          nxt.d0      = db_q;
          nxt.d1      = TSR_VU_LW'(j_q) << 4;
          nxt.tail    = last_line ? tail_q : 5'd16;
          nxt.r_first = (j_q == 24'd0);
          nxt.r_last  = last_line;
          nxt.r_row   = r_q;
          nxt_cnt     = (j_q == 24'd0) ? 2'((opk[tag_q].k + 5'd7) >> 3) : 2'd0;
          nxt_last    = last_line && last_row;
        end else begin
          nxt.kind   = SK_LANE;
          nxt.uop    = uop_q;
          nxt.nsub   = nsub_q;
          nxt.exitb  = tsr_vu_uop_back(uop_q);
          nxt.asrc   = ra_q ? SRC_SPM : SRC_NONE;
          nxt.aline  = ab_q + TSR_VU_LW'(j_q);
          nxt.bsrc   = rbq_q ? SRC_SPM : SRC_NONE;
          nxt.bline  = bb_q + TSR_VU_LW'(j_q);
          nxt.c1sel  = c1s_q;
          nxt.c2sel  = c2s_q;
          nxt.cword  = r_q[3:0];
          nxt.dst    = dst_q;
          nxt.d0     = db_q + TSR_VU_LW'(j_q);
          nxt.tail   = last_line ? tail_q : 5'd16;
          nxt.red    = (pk_q == P_RED) || (pk_q == P_AMASK);
          nxt.r_first = (j_q == 24'd0);
          nxt.r_last  = last_line;
          nxt.r_li    = j_q[2:0];
          nxt.r_fresh = (j_q < 24'd8);
          nxt.r_ng    = ng_q;
          nxt.r_row   = r_q;
          nxt.r_lrow  = last_row;
          if (pk_q == P_ANORM1) nxt.d0 = LM_RINV + TSR_VU_LW'(j_q);
          if (pk_q == P_RED) begin
            // one result line per 16 rows (vector) or the s write (TO_S), at the row's first line
            if (j_q == 24'd0 && r_q[3:0] == 4'd0) begin
              nxt_cnt = 2'd1;
              nxt_rc  = !opk[tag_q].to_s;
            end
          end else if (pk_q == P_AMASK) begin
            nxt_oc  = 2'd1;
            nxt_cnt = (j_q == 24'd0 && r_q[3:0] == 4'd0) ? 2'd2 : 2'd1;
            nxt_rc  = (j_q == 24'd0 && r_q[3:0] == 4'd0);
          end else if (dst_q == DS_SPM) begin
            nxt_oc  = 2'd1;
            nxt_cnt = 2'd1;
          end else if (dst_q == DS_LM) begin
            nxt_lmw = 1'b1;
          end
          nxt_last = last_line && last_row && (pk_q != P_ANORM1);
        end
      end
      G_GSEQ: begin                      // GDN: the head command, then the lines of k, q and v (VU §7.12.5)
        nxt_v     = 1'b1;
        nxt.kind  = SK_GDN;
        nxt.nsub  = 3'd4;
        nxt.tail  = 5'd16;
        case (gp_q)
          2'd0: begin
            nxt.gsub  = GS_CFG;
            nxt.bline = TSR_VU_LW'(w_q[7][15:0]);          // m
            nxt.r_row = w_q[7][31:16];                     // rows
            nxt.d1    = TSR_VU_LW'(w_q[6]);                // o line
          end
          2'd1: begin
            nxt.gsub  = GS_VK;
            nxt.asrc  = SRC_SPM;
            nxt.aline = TSR_VU_LW'(w_q[3]) + TSR_VU_LW'(gj_q);
          end
          2'd2: begin
            nxt.gsub  = GS_VQ;
            nxt.asrc  = SRC_SPM;
            nxt.aline = TSR_VU_LW'(w_q[4]) + TSR_VU_LW'(gj_q);
          end
          default: begin
            nxt.gsub  = GS_VV;
            nxt.asrc  = SRC_SPM;
            nxt.aline = TSR_VU_LW'(w_q[5]) + TSR_VU_LW'(gj_q);
            if (gj_q == ((w_q[7][31:16] + 16'd15) >> 4) - 16'd1 && w_q[7][19:16] != 4'd0) nxt.tail = 5'(w_q[7][19:16]);
          end
        endcase
      end
      G_REDE: begin                      // RED TO_S with m = 0: the empty-row result (VU §6.2)
        nxt_v       = 1'b1;
        nxt.kind    = SK_REDE;
        nxt.r_row   = 16'd0;
        nxt.r_lrow  = 1'b1;
        nxt_cnt     = 2'd1;
        nxt_last    = 1'b1;
      end
      G_SOP: begin
        nxt_v    = 1'b1;
        nxt_last = 1'b1;
        nxt_cnt  = 2'd1;
        nxt.kind  = SK_LANE;
        nxt.nsub  = 3'(RF);
        nxt.asrc  = SRC_K;
        nxt.bsrc  = SRC_K;
        nxt.dst   = DS_SREG;
        nxt.d0    = TSR_VU_LW'(w_q[1][3:0]);                  // sd
        nxt.tail  = 5'd16;
        case (sub)
          TSR_SOP_SUB_MOVI:  begin nxt.uop = U_SETC; nxt.c1sel = C_K; end
          TSR_SOP_SUB_ADD, TSR_SOP_SUB_ADDI: nxt.uop = U_ADD;
          TSR_SOP_SUB_MUL, TSR_SOP_SUB_MULI: nxt.uop = U_MUL;
          TSR_SOP_SUB_SUB:   nxt.uop = U_SUB;
          TSR_SOP_SUB_MAX:   nxt.uop = U_MAX;
          TSR_SOP_SUB_MIN:   nxt.uop = U_MIN;
          TSR_SOP_SUB_EXP:   begin nxt.uop = U_EXP; nxt.nsub = 3'(RE); end
          TSR_SOP_SUB_RECIP: begin nxt.uop = U_RECIP; nxt.nsub = 3'(RB); nxt.exitb = 1'b1; end
          TSR_SOP_SUB_LDS: begin
            nxt.uop   = U_COPY;
            nxt.asrc  = SRC_SPM;
            nxt.aline = TSR_VU_LW'(w_q[3][27:4]);
            nxt.dword = w_q[3][3:0];
          end
          TSR_SOP_SUB_STS: begin
            nxt.uop   = U_COPY;
            nxt.dst   = DS_SPM;
            nxt.d0    = TSR_VU_LW'(w_q[3][27:4]);
            nxt.dword = w_q[3][3:0];
            nxt.sts   = 1'b1;
            nxt_oc    = 2'd1;
          end
          default: begin                 // SQRT, RSQRT, DIV: the scalar path
            nxt.kind = SK_SOPX;
            nxt.dst  = DS_SREG;
          end
        endcase
      end
      G_RTBL: begin                      // ROPE: table line i -> table RAM line i
        nxt_v     = 1'b1;
        nxt.kind  = SK_TBL;
        nxt.asrc  = SRC_SPM;
        nxt.aline = TSR_VU_LW'(w_q[4][23:0]) + TSR_VU_LW'(tbl_i_q);
        nxt.d0    = TSR_VU_LW'(tbl_i_q);
      end
      G_RPOS0, G_RPOS1, G_RTOK: begin     // ROPE: position line k -> pos buffer half k mod 2
        nxt_v     = 1'b1;
        nxt.kind  = SK_POS;
        nxt.asrc  = SRC_SPM;
        begin
          k = (st_q == G_RPOS0) ? 16'd0 : (st_q == G_RPOS1) ? 16'd1 : (16'(t_q[15:4]) + 16'd1);
          nxt.aline = TSR_VU_LW'(w_q[3][23:0]) + TSR_VU_LW'(k);
          nxt.d0    = TSR_VU_LW'(k);
        end
      end
      G_RROT: begin
        // rotation slots per head: NEOX n_dims/32 pairs, NORM n_dims/16
        neox = (sub == TSR_ROPE_MODE_NEOX);
        rot  = (o_q < nrot_q);
        nxt_v      = 1'b1;
        nxt.kind   = SK_LANE;
        nxt.asrc   = SRC_SPM;
        nxt.dst    = DS_SPM;
        nxt.tail   = 5'd16;
        nxt.ang_h  = t_q[0];
        nxt.c1sel  = C_ANG;
        nxt.c2sel  = C_ANG;
        if (rot && neox) begin
          nxt.uop   = U_ROT;
          nxt.rope  = 2'd1;
          nxt.nsub  = 3'(2*RF);
          nxt.aline = sbh_q + TSR_VU_LW'(o_q);
          nxt.bsrc  = SRC_SPM;
          nxt.bline = sbh_q + TSR_VU_LW'(o_q) + TSR_VU_LW'(nd_q[15:5]);
          nxt.d0    = dbh_q + TSR_VU_LW'(o_q);
          nxt.d1    = dbh_q + TSR_VU_LW'(o_q) + TSR_VU_LW'(nd_q[15:5]);
          nxt.ang_e = o_q[4:0];
          nxt_oc    = 2'd2;
          nxt_cnt   = 2'd2;
        end else if (rot) begin
          nxt.uop    = U_ROT;
          nxt.rope   = 2'd2;
          nxt.nsub   = 3'(RF);
          nxt.aline  = sbh_q + TSR_VU_LW'(o_q);
          nxt.d0     = dbh_q + TSR_VU_LW'(o_q);
          nxt.ang_e  = o_q[5:1];
          nxt.ang_hi = o_q[0];
          nxt_oc     = 2'd1;
          nxt_cnt    = 2'd1;
        end else begin                   // raw copy of a line past n_dims (VU-23)
          nxt.uop   = U_COPY;
          nxt.nsub  = 3'(RF);
          nxt.c1sel = C_ZERO;
          nxt.c2sel = C_ZERO;
          nxt.aline = sbh_q + TSR_VU_LW'(o_q) + (neox ? TSR_VU_LW'(nd_q[15:5]) : TSR_VU_LW'(0));
          nxt.d0    = dbh_q + TSR_VU_LW'(o_q) + (neox ? TSR_VU_LW'(nd_q[15:5]) : TSR_VU_LW'(0));
          nxt_oc    = 2'd1;
          nxt_cnt   = 2'd1;
        end
        begin
          nxt.tok_last = (o_q == npm1_q) && (h_q == nh_q - 16'd1);
          nxt_last     = nxt.tok_last && (t_q == tp_q - 16'd1);
        end
      end
      G_AM: begin
        i_in_b = ami_q - amb_q;
        rr = ami_q >> am_dsh;                            // row of an output line
        g  = rr >> 4;
        // item locations (prepass: vector lines of the LSUM partials; main: the partial blocks)
        la = bpa_q + TSR_VU_LW'(ami_q);
        lb = bpb_q + TSR_VU_LW'(ami_q);
        lk = (NUM_CH == 1) ? am_lparts + TSR_VU_LW'(ami_q)
                           : LM_SCR + TSR_VU_LW'(CTF_I) * TSR_VU_LW'(PSL) + TSR_VU_LW'(ami_q);
        ok = (NUM_CH == 1) ? am_parts + TSR_VU_LW'(ami_q)
                           : LM_SCR + TSR_VU_LW'(CTF_I) * TSR_VU_LW'(NB) + TSR_VU_LW'(i_in_b);
        nxt.kind  = SK_LANE;
        nxt.nsub  = 3'(RF);
        nxt.tail  = 5'd16;
        nxt.dst   = DS_SPM;
        case (am_q)
          AM_PRE: begin                  // lk tree pass amp_q on vector line ami_q
            nxt_v     = !(pc.bar && ami_q == 16'd0 && !lm_idle) && bases_ok;
            nxt.uop   = U_ADD;
            nxt.asrc  = pc.a_scr ? SRC_LM : SRC_SPM;
            nxt.aline = la;
            nxt.bsrc  = pc.b_scr ? SRC_LM : SRC_SPM;
            nxt.bline = lb;
            nxt.dst   = DS_LM;
            nxt.d0    = bpd_q + TSR_VU_LW'(ami_q);
            nxt_lmw   = 1'b1;
          end
          AM_M: begin                    // m' = MAXT(m_run, mk)
            nxt_v     = 1'b1;
            nxt.uop   = U_MAX;
            nxt.asrc  = SRC_SPM;
            nxt.aline = am_st + TSR_VU_LW'(ami_q);
            nxt.bsrc  = SRC_SPM;
            nxt.bline = am_mk + TSR_VU_LW'(ami_q);
            nxt.dst   = DS_LM;
            nxt.d0    = LM_M + TSR_VU_LW'(ami_q);
            nxt_lmw   = 1'b1;
          end
          AM_A, AM_B: begin              // a = EXP(FM(FS(m_run, m'), scale)), b likewise from mk
            nxt_v     = 1'b1;
            nxt.uop   = U_ESCB;
            nxt.nsub  = nsub_of(U_ESCB);
            nxt.asrc  = SRC_SPM;
            nxt.aline = ((am_q == AM_A) ? am_st : am_mk) + TSR_VU_LW'(ami_q);
            nxt.bsrc  = SRC_LM;
            nxt.bline = LM_M + TSR_VU_LW'(ami_q);
            nxt.c2sel = C_K;
            nxt.dst   = DS_LM;
            nxt.d0    = ((am_q == AM_A) ? LM_A : LM_B) + TSR_VU_LW'(ami_q);
            nxt_lmw   = 1'b1;
          end
          AM_L: begin                    // l_run = FA(FM(l_run, a), FM(lk, b)) (a, b element-wise)
            nxt_v = 1'b1;
            if (amcl_q != 2'd2) begin
              nxt_cl    = 1'b1;
              nxt.kind  = (amcl_q == 2'd0) ? SK_CL1 : SK_CL2;
              nxt.asrc  = SRC_LM;
              nxt.aline = ((amcl_q == 2'd0) ? LM_A : LM_B) + TSR_VU_LW'(ami_q);
            end else begin
              nxt.uop   = U_ROT;
              nxt.asrc  = SRC_SPM;
              nxt.aline = am_st + TSR_VU_LW'(am_Rl) + TSR_VU_LW'(ami_q);
              nxt.bsrc  = (NUM_CH == 1) ? SRC_SPM : SRC_LM;
              nxt.bline = lk;
              nxt.c1sel = C_CLE;
              nxt.c2sel = C_CLE;
              nxt.d0    = am_st + TSR_VU_LW'(am_Rl) + TSR_VU_LW'(ami_q);
              nxt.keep  = 1'b1;
              nxt.tail  = (ami_q == am_Rl - 16'd1) ? 5'(am_R - 16'(16) * (am_Rl - 16'd1)) : 5'd16;
              nxt_oc    = 2'd1;
              nxt_cnt   = 2'd1;
            end
          end
          AM_MR, AM_LI, AM_MI: begin     // raw copies into the running state (the words past R keep)
            nxt_v     = 1'b1;
            nxt.uop   = U_COPY;
            nxt.asrc  = (am_q == AM_MR) ? SRC_LM : (am_q == AM_LI && NUM_CH > 1) ? SRC_LM : SRC_SPM;
            nxt.aline = (am_q == AM_MR) ? LM_M + TSR_VU_LW'(ami_q) : (am_q == AM_LI) ? lk : am_mk + TSR_VU_LW'(ami_q);
            nxt.d0    = am_st + ((am_q == AM_LI) ? TSR_VU_LW'(am_Rl) : TSR_VU_LW'(0)) + TSR_VU_LW'(ami_q);
            nxt.keep  = 1'b1;
            nxt.tail  = (ami_q == am_Rl - 16'd1) ? 5'(am_R - 16'(16) * (am_Rl - 16'd1)) : 5'd16;
            nxt_oc    = 2'd1;
            nxt_cnt   = 2'd1;
          end
          AM_MT: begin                   // main tree pass amp_q on output line ami_q of the batch
            nxt_v     = !(pc.bar && ami_q == amb_q && !lm_idle) && bases_ok;
            nxt.uop   = U_ADD;
            nxt.asrc  = pc.a_scr ? SRC_LM : SRC_SPM;
            nxt.aline = bma_q + (pc.a_scr ? TSR_VU_LW'(i_in_b) : TSR_VU_LW'(ami_q));
            nxt.bsrc  = pc.b_scr ? SRC_LM : SRC_SPM;
            nxt.bline = bmb_q + (pc.b_scr ? TSR_VU_LW'(i_in_b) : TSR_VU_LW'(ami_q));
            nxt.dst   = DS_LM;
            nxt.d0    = bmd_q + TSR_VU_LW'(i_in_b);
            nxt_lmw   = 1'b1;
          end
          AM_MC: begin                   // Orun = FA(FM(Orun, a_r), FM(Ok, b_r)), or Ok (INIT)
            nxt_v = 1'b1;
            if (!am_init && (amg_q != g || amcl_q != 2'd2)) begin
              nxt_cl    = 1'b1;
              nxt.kind  = (amg_q != g || amcl_q == 2'd0) ? SK_CL1 : SK_CL2;
              nxt.asrc  = SRC_LM;
              nxt.aline = ((amg_q != g || amcl_q == 2'd0) ? LM_A : LM_B) + TSR_VU_LW'(g);
            end else begin
              nxt.uop   = am_init ? U_COPY : U_ROT;
              nxt.asrc  = am_init ? ((NUM_CH == 1) ? SRC_SPM : SRC_LM) : SRC_SPM;
              nxt.aline = am_init ? ok : am_orun + TSR_VU_LW'(ami_q);
              nxt.bsrc  = am_init ? SRC_NONE : (NUM_CH == 1) ? SRC_SPM : SRC_LM;
              nxt.bline = ok;
              nxt.c1sel = am_init ? C_ZERO : C_CLW;
              nxt.c2sel = am_init ? C_ZERO : C_CLW;
              nxt.cword = rr[3:0];
              nxt.d0    = am_orun + TSR_VU_LW'(ami_q);
              nxt_oc    = 2'd1;
              nxt_cnt   = 2'd1;
              nxt_last  = (ami_q == am_NL - 16'd1);
            end
          end
          default: ;                     // barriers
        endcase
      end
      default: ;
    endcase
  end

  // AMERGE pass bases: CTP[amp_q] -> products -> bases (3 stages; bases_ok when settled)
  assign bases_ok = (bset_q == 2'd0) && (amp_d_q == amp_q);
  ctp_t pc;                              // the current pass entry
  assign pc = ctp_t'(pe_q);
  always_ff @(posedge clk) begin
    amp_d_q <= amp_q;
    pe_q    <= CTP[amp_q];
    pp_a_q  <= TSR_VU_LW'(pc.a_i) * TSR_VU_LW'(am_Rl);
    pp_b_q  <= TSR_VU_LW'(pc.b_i) * TSR_VU_LW'(am_Rl);
    pm_a_q  <= TSR_VU_LW'(pc.a_i) * TSR_VU_LW'(am_cs);
    pm_b_q  <= TSR_VU_LW'(pc.b_i) * TSR_VU_LW'(am_cs);
    bpa_q   <= pc.a_scr ? LM_SCR + TSR_VU_LW'(pc.a_i) * TSR_VU_LW'(PSL) : am_lparts + pp_a_q;
    bpb_q   <= pc.b_scr ? LM_SCR + TSR_VU_LW'(pc.b_i) * TSR_VU_LW'(PSL) : am_lparts + pp_b_q;
    bpd_q   <= LM_SCR + TSR_VU_LW'(pc.d) * TSR_VU_LW'(PSL);
    bma_q   <= pc.a_scr ? LM_SCR + TSR_VU_LW'(pc.a_i) * TSR_VU_LW'(NB) : am_parts + pm_a_q;
    bmb_q   <= pc.b_scr ? LM_SCR + TSR_VU_LW'(pc.b_i) * TSR_VU_LW'(NB) : am_parts + pm_b_q;
    bmd_q   <= LM_SCR + TSR_VU_LW'(pc.d) * TSR_VU_LW'(NB);
    if (!rst_n || st_q == G_SETUP) bset_q <= 2'd3;
    else if (amp_d_q != amp_q)     bset_q <= 2'd2;
    else if (bset_q != 2'd0)       bset_q <= bset_q - 2'd1;
  end

  // credits of the next slot
  wire nxt_ok = (32'(oc_free) >= 32'(nxt_oc)) && (!nxt_rc || rc_free != 4'd0);
  wire load   = nxt_v && (!slot_valid || slot_ready) && nxt_ok;

  // -------------------------------------------------------------------------------------------
  // Slot register
  // -------------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      slot_valid <= 1'b0;
    end else if (load) begin
      slot_valid <= 1'b1;
    end else if (slot_ready) begin
      slot_valid <= 1'b0;
    end
  end
  always_ff @(posedge clk) if (load) slot <= nxt;
  assign slot_take = load;
  assign slot_take_tag = tag_q;
  assign slot_oc   = nxt_oc;
  assign slot_rc   = nxt_rc;
  assign slot_cnt  = nxt_cnt;
  assign slot_lmw  = nxt_lmw;

  // -------------------------------------------------------------------------------------------
  // FSM
  // -------------------------------------------------------------------------------------------
  assign busy = (st_q != G_IDLE);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st_q         <= G_IDLE;
      gen_done     <= 1'b0;
      gen_done_tag <= 1'b0;
      rope_start   <= 1'b0;
      cl_done_q    <= 1'b0;
    end else begin
      gen_done   <= 1'b0;
      rope_start <= 1'b0;
      case (st_q)
        G_IDLE: begin
          if (start) begin
            st_q  <= G_SETUP;
            tag_q <= start_tag;
            w_q   <= start_w;
            op_q  <= start_w[0][7:0];
          end
        end
        G_SETUP: begin
          // decode the op into the pass registers
          r_q       <= '0;
          j_q       <= '0;
          qc_q      <= '0;
          t_q       <= '0;
          h_q       <= '0;
          o_q       <= '0;
          tbl_i_q   <= '0;
          ra_q      <= 1'b0;
          rbq_q     <= 1'b0;
          c1s_q     <= C_ZERO;
          c2s_q     <= C_ZERO;
          cl_q      <= 1'b0;
          cls_q     <= SRC_SPM;
          dst_q     <= DS_SPM;
          ng_q      <= '0;
          case (op_q)
            tsr_tables_pkg::TSR_OP_VOP: begin
              logic [1:0] rd3;
              logic [4:0] fn;
              rd3 = 2'(tsr_vop_func_reads(x16[7:0]));
              fn  = x16[4:0];
              pk_q   <= P_VOP;
              rows_q <= w_q[6][15:0];
              nl_q   <= (w_q[5][23:0] + 24'd15) >> 4;
              tail_q <= (w_q[5][3:0] == 4'd0) ? 5'd16 : 5'(w_q[5][3:0]);
              ab_q   <= TSR_VU_LW'(w_q[2]);
              bb_q   <= TSR_VU_LW'(w_q[3]);
              cb_q   <= TSR_VU_LW'(w_q[4]);
              db_q   <= TSR_VU_LW'(w_q[1]);
              ds_q   <= w_q[6][31:16];
              as_q   <= w_q[7][15:0];
              bs_q   <= w_q[7][31:16];
              uop_q  <= tsr_vu_uop_of_func(fn);
              nsub_q <= nsub_of(tsr_vu_uop_of_func(fn));
              ra_q   <= rd3[0];
              rbq_q  <= rd3[1];
              c1s_q  <= (sub == TSR_VOP_CMODE_SREG) ? C_K : (sub == TSR_VOP_CMODE_VEC) ? C_CLW : C_ZERO;
              cl_q   <= (sub == TSR_VOP_CMODE_VEC);
              cls_q  <= SRC_SPM;
            end
            tsr_tables_pkg::TSR_OP_RED: begin
              logic [23:0] g;
              g = (w_q[5][23:0] + 24'd7) >> 3;
              pk_q   <= P_RED;
              rows_q <= w_q[6][15:0];
              nl_q   <= (w_q[5][23:0] + 24'd15) >> 4;
              tail_q <= (w_q[5][3:0] == 4'd0) ? 5'd16 : 5'(w_q[5][3:0]);
              ab_q   <= TSR_VU_LW'(w_q[2]);
              as_q   <= w_q[7][15:0];
              db_q   <= '0;
              ds_q   <= '0;
              uop_q  <= (sub == TSR_RED_KIND_SUM) ? U_RSUM : (sub == TSR_RED_KIND_SUMSQ) ? U_RSUMSQ : U_RMAX;
              nsub_q <= 3'(RF);
              ra_q   <= 1'b1;
              dst_q  <= DS_NONE;
              ng_q   <= (g >= 24'd16) ? 4'd15 : 4'(g - 24'd1);
            end
            TSR_OP_ASMAX, TSR_OP_AMASK: begin
              logic [15:0] nl, st;
              nl = w_q[4][15:0];
              st = (w_q[4][31:16] != 16'd0) ? w_q[4][31:16] : w_q[4][15:0];
              pk_q   <= (op_q == TSR_OP_ASMAX) ? P_ASMAX : P_AMASK;
              rows_q <= w_q[5][15:0];
              nl_q   <= 24'(nl);
              tail_q <= 5'd16;
              ab_q   <= TSR_VU_LW'(w_q[2]);
              db_q   <= TSR_VU_LW'(w_q[1]);
              as_q   <= st;
              ds_q   <= st;
              bb_q   <= TSR_VU_LW'(w_q[3]);
              bsl_q  <= w_q[6];
              rpq_q  <= w_q[5][31:16];
              cb_q   <= TSR_VU_LW'(w_q[3]);
              ra_q   <= 1'b1;
              if (op_q == TSR_OP_ASMAX) begin
                uop_q  <= U_ASMAX;
                nsub_q <= nsub_of(U_ASMAX);
                c1s_q  <= C_CLW;
                c2s_q  <= C_K;
                cl_q   <= 1'b1;
              end else begin
                uop_q  <= U_AMASK;
                nsub_q <= 3'(RF);
                rbq_q  <= 1'b1;
                ng_q   <= (nl >= 16'd8) ? 4'd15 : 4'(2 * nl - 16'd1);
              end
            end
            TSR_OP_ANORM: begin
              // prepass: 1/l = RC(MAXT(l_run, 2^-126)) for ceil(R/16) lines into the local memory
              pk_q   <= P_ANORM1;
              rows_q <= 16'd1;
              nl_q   <= (24'(w_q[4][15:0]) + 24'd15) >> 4;
              tail_q <= 5'd16;
              ab_q   <= TSR_VU_LW'(w_q[3]);
              uop_q  <= U_RCPMAX;
              nsub_q <= nsub_of(U_RCPMAX);
              ra_q   <= 1'b1;
              dst_q  <= DS_LM;
            end
            tsr_tables_pkg::TSR_OP_TOPK: begin
              logic [15:0] kl;
              kl = (16'(w_q[0][20:16]) + 16'd7) >> 3;
              pk_q   <= P_TOPK;
              rows_q <= w_q[6][15:0];
              nl_q   <= (w_q[5][23:0] + 24'd15) >> 4;
              tail_q <= (w_q[5][3:0] == 4'd0) ? 5'd16 : 5'(w_q[5][3:0]);
              ab_q   <= TSR_VU_LW'(w_q[2]);
              as_q   <= w_q[7][15:0];
              db_q   <= TSR_VU_LW'(w_q[1]);
              ds_q   <= (w_q[6][31:16] != 16'd0) ? w_q[6][31:16] : kl;
              ra_q   <= 1'b1;
              dst_q  <= DS_NONE;
            end
            tsr_tables_pkg::TSR_OP_GDN: begin
              gp_q <= 2'd0;
              gj_q <= '0;
            end
            tsr_tables_pkg::TSR_OP_DQROW: begin
              logic [15:0] lk, rb;
              lk = 16'(w_q[5] >> 4);                                       // L(K), K = 32·n
              rb = 16'(tsr_row_beats(sub[0], w_q[5]));
              pk_q   <= P_DQ;
              rows_q <= w_q[6][15:0];
              nl_q   <= 24'(lk);
              tail_q <= 5'd16;
              ab_q   <= TSR_VU_LW'(w_q[2]);
              cb_q   <= TSR_VU_LW'(w_q[2]);
              db_q   <= TSR_VU_LW'(w_q[1]);
              ds_q   <= (w_q[7][31:16] != 16'd0) ? w_q[7][31:16] : lk;
              as_q   <= (w_q[7][15:0] != 16'd0) ? w_q[7][15:0] : rb;
              dqf_q  <= sub[0];
              uop_q  <= sub[0] ? U_ROT : U_ADDMULC;
              nsub_q <= 3'(RF);
              ra_q   <= 1'b1;
              dst_q  <= DS_SPM;
            end
            TSR_OP_AMERGE: begin
              logic [15:0] R, rl16;
              logic [31:0] nl, cs;
              R    = w_q[5][15:0];
              rl16 = (R + 16'd15) >> 4;
              nl   = 32'(R) << (2 + w_q[5][17:16]);
              cs   = (w_q[7] != 32'd0) ? w_q[7] : nl;
              am_R      <= R;
              am_Rl     <= rl16;
              am_NL     <= 16'(nl);
              am_dsh    <= 3'(2 + w_q[5][17:16]);
              am_cs     <= cs;
              am_orun   <= TSR_VU_LW'(w_q[1]);
              am_parts  <= TSR_VU_LW'(w_q[2]);
              am_lparts <= TSR_VU_LW'(w_q[2]) + TSR_VU_LW'(32'(NUM_CH) * cs);
              am_mk     <= TSR_VU_LW'(w_q[3]);
              am_st     <= TSR_VU_LW'(w_q[4]);
              am_init   <= w_q[0][12];
              am_q      <= (NUM_CH == 1) ? (w_q[0][12] ? AM_BARI : AM_M) : AM_PRE;
              amp_q     <= '0;
              ami_q     <= '0;
              amb_q     <= '0;
              amcl_q    <= 2'd0;
              amg_q     <= 16'hFFFF;
            end
            tsr_tables_pkg::TSR_OP_ROPE: begin
              logic [31:0] H;
              logic [15:0] tp;
              H  = (32'(w_q[6][15:0]) * 32'(w_q[5][15:0])) >> 4;
              tp = (w_q[6][15:0] != 16'd0) ? w_q[6][31:16] : 16'd0;
              hd_q    <= w_q[5][15:0];
              nd_q    <= w_q[5][31:16];
              nh_q    <= w_q[6][15:0];
              tp_q    <= tp;
              hl_q    <= 32'(w_q[5][15:4]);
              H_q     <= H;
              ssp_q   <= (w_q[7][15:0] != 16'd0) ? 32'(w_q[7][15:0]) : H;
              dsp_q   <= (w_q[7][31:16] != 16'd0) ? 32'(w_q[7][31:16]) : H;
              sbt_q   <= TSR_VU_LW'(w_q[2]);
              dbt_q   <= TSR_VU_LW'(w_q[1]);
              sbh_q   <= TSR_VU_LW'(w_q[2]);
              dbh_q   <= TSR_VU_LW'(w_q[1]);
              tbl_n_q <= (w_q[5][31:16] + 16'd31) >> 5;
              // NEOX: n_dims/32 pair slots + (head_dim - n_dims)/16 copies; NORM: head_dim/16 slots
              nrot_q  <= (sub == TSR_ROPE_MODE_NEOX) ? 24'(w_q[5][31:21]) : 24'(w_q[5][31:20]);
              npm1_q  <= ((sub == TSR_ROPE_MODE_NEOX) ? 24'(w_q[5][15:4]) - 24'(w_q[5][31:21])
                                                      : 24'(w_q[5][15:4])) - 24'd1;
              posl_q  <= (tp + 16'd15) >> 4;
            end
            default: ;
          endcase
    
          st_q      <= setup_next;
          cl_done_q <= 1'b0;
          if (setup_next == G_FIN) begin
            gen_done     <= 1'b1;
            gen_done_tag <= tag_q;
            st_q         <= G_IDLE;
          end
          if (op_q == tsr_tables_pkg::TSR_OP_ROPE && setup_next != G_FIN) rope_start <= 1'b1;
        end
        G_PASS: begin
          if (load) begin
            if (nxt_cl) begin
              cl_done_q <= 1'b1;
            end else begin
              cl_done_q <= cl_done_q;
              if (!last_line) begin
                j_q <= j_q + 24'd1;
                if (pk_q == P_DQ && j_q[5:0] == 6'd63) begin          // DQROW: next chunk
                  cl_done_q <= 1'b0;
                  cb_q      <= cb_q + (dqf_q ? TSR_VU_LW'(9) : TSR_VU_LW'(17));
                end
              end else begin
                j_q <= '0;
                cl_done_q <= 1'b0;
                if (!last_row) begin
                  r_q  <= r_q + 16'd1;
                  ab_q <= ab_q + TSR_VU_LW'(as_q);
                  db_q <= db_q + TSR_VU_LW'(ds_q);
                  if (pk_q == P_DQ) cb_q <= ab_q + TSR_VU_LW'(as_q);
                  if (pk_q == P_AMASK) begin
                    if (qc_q == rpq_q - 16'd1) begin
                      qc_q <= '0;
                      bb_q <= bb_q + TSR_VU_LW'(bsl_q);
                    end else begin
                      qc_q <= qc_q + 16'd1;
                    end
                  end else begin
                    bb_q <= bb_q + TSR_VU_LW'(bs_q);
                  end
                end else if (pk_q == P_ANORM1) begin
                  st_q <= G_BAR;
                end else if (HAS_GDN && pk_q == P_GPS) begin
                  st_q <= G_GSET2;
                end else begin
                  st_q         <= G_IDLE;
                  gen_done     <= 1'b1;
                  gen_done_tag <= tag_q;
                end
              end
            end
          end
        end
        G_BAR: begin                     // ANORM: wait for the 1/l prepass, then the main pass
          if (lm_idle && !slot_valid) begin
            st_q   <= G_PASS;
            pk_q   <= P_ANORM2;
            rows_q <= 16'(w_q[4][15:0]);
            nl_q   <= 24'(16'd4 << w_q[4][17:16]);          // D/16
            r_q    <= '0;
            j_q    <= '0;
            ab_q   <= TSR_VU_LW'(w_q[2]);
            db_q   <= TSR_VU_LW'(w_q[1]);
            as_q   <= 16'(16'd4 << w_q[4][17:16]);
            ds_q   <= 16'(16'd4 << w_q[4][17:16]);
            uop_q  <= U_MULC;
            nsub_q <= 3'(RF);
            ra_q   <= 1'b1;
            rbq_q  <= 1'b0;
            c1s_q  <= C_CLW;
            c2s_q  <= C_ZERO;
            cl_q   <= 1'b1;
            cls_q  <= SRC_LM;
            cb_q   <= LM_RINV;
            dst_q  <= DS_SPM;
            tail_q <= 5'd16;
            cl_done_q <= 1'b0;
          end
        end
        G_SOP, G_REDE: begin
          if (load) begin
            st_q         <= G_IDLE;
            gen_done     <= 1'b1;
            gen_done_tag <= tag_q;
          end
        end
        G_GSEQ: if (HAS_GDN && load) begin
          gj_q <= gj_q + 16'd1;
          case (gp_q)
            2'd0: begin gp_q <= 2'd1; gj_q <= '0; end
            2'd1: if (gj_q == 16'(w_q[7][15:4]) - 16'd1) begin gp_q <= 2'd2; gj_q <= '0; end
            2'd2: if (gj_q == 16'(w_q[7][15:4]) - 16'd1) begin gp_q <= 2'd3; gj_q <= '0; end
            default: if (gj_q == ((w_q[7][31:16] + 16'd15) >> 4) - 16'd1) begin
              // the pseudo row: the lines of k once more, as one row
              st_q      <= G_PASS;
              pk_q      <= P_GPS;
              rows_q    <= 16'd1;
              nl_q      <= 24'(w_q[7][15:4]);
              tail_q    <= 5'd16;
              r_q       <= '0;
              j_q       <= '0;
              ab_q      <= TSR_VU_LW'(w_q[3]);
              as_q      <= '0;
              db_q      <= '0;
              ds_q      <= '0;
              nsub_q    <= 3'(RF);
              ra_q      <= 1'b1;
              rbq_q     <= 1'b0;
              cl_q      <= 1'b0;
              cl_done_q <= 1'b0;
              dst_q     <= DS_NONE;
            end
          endcase
        end
        G_GSET2: begin                   // GDN: the rows of S
          st_q      <= G_PASS;
          pk_q      <= P_GROW;
          rows_q    <= w_q[7][31:16];
          nl_q      <= 24'(w_q[7][15:4]);
          r_q       <= '0;
          j_q       <= '0;
          ab_q      <= TSR_VU_LW'(w_q[2]);
          as_q      <= 16'(w_q[7][15:4]);
          db_q      <= TSR_VU_LW'(w_q[1]);
          ds_q      <= 16'(w_q[7][15:4]);
          cl_done_q <= 1'b0;
        end
        G_RTBL: begin
          if (load) begin
            if (tbl_i_q == tbl_n_q - 16'd1) st_q <= G_RPOS0;
            tbl_i_q <= tbl_i_q + 16'd1;
          end
        end
        G_RPOS0: if (load) st_q <= (posl_q > 16'd1) ? G_RPOS1 : G_RROT;
        G_RPOS1: if (load) st_q <= G_RROT;
        G_RTOK:  if (load) st_q <= G_RROT;             // prefetch of position line t/16 + 1
        G_RROT: begin
          if (load) begin
            if (nxt.tok_last) begin
              o_q <= '0;
              h_q <= '0;
              if (nxt_last) begin
                st_q         <= G_IDLE;
                gen_done     <= 1'b1;
                gen_done_tag <= tag_q;
              end else begin
                t_q   <= t_q + 16'd1;
                sbt_q <= sbt_q + TSR_VU_LW'(ssp_q);
                dbt_q <= dbt_q + TSR_VU_LW'(dsp_q);
                sbh_q <= sbt_q + TSR_VU_LW'(ssp_q);
                dbh_q <= dbt_q + TSR_VU_LW'(dsp_q);
                // token t+1 = 16k (k >= 1): prefetch position line k + 1 if it exists
                if (t_q[3:0] == 4'd15 && (32'(t_q[15:4]) + 32'd2) < 32'(posl_q))
                  st_q <= G_RTOK;
              end
            end else begin
              begin
                if (o_q == npm1_q) begin
                  o_q   <= '0;
                  h_q   <= h_q + 16'd1;
                  sbh_q <= sbh_q + TSR_VU_LW'(hl_q);
                  dbh_q <= dbh_q + TSR_VU_LW'(hl_q);
                end else begin
                  o_q <= o_q + 24'd1;
                end
              end
            end
          end
        end
        G_AM: begin
          logic [15:0] bend;               // the last line of the batch starting after line ami_q
          bend = ((am_NL - ami_q - 16'd1) <= 16'(NB)) ? (am_NL - 16'd1) : (ami_q + 16'(NB));
          case (am_q)
            AM_PRE: if (load) begin
              if (ami_q == am_Rl - 16'd1) begin
                ami_q <= '0;
                if (amp_q == 4'(NP - 1)) am_q <= am_init ? AM_BARI : AM_M;
                amp_q <= amp_q + 4'd1;
              end else begin
                ami_q <= ami_q + 16'd1;
              end
            end
            AM_M: if (load) begin
              ami_q <= (ami_q == am_Rl - 16'd1) ? 16'd0 : ami_q + 16'd1;
              if (ami_q == am_Rl - 16'd1) am_q <= AM_BAR1;
            end
            AM_BAR1: if (lm_idle) am_q <= AM_A;
            AM_A: if (load) begin
              ami_q <= (ami_q == am_Rl - 16'd1) ? 16'd0 : ami_q + 16'd1;
              if (ami_q == am_Rl - 16'd1) am_q <= AM_B;
            end
            AM_B: if (load) begin
              ami_q <= (ami_q == am_Rl - 16'd1) ? 16'd0 : ami_q + 16'd1;
              if (ami_q == am_Rl - 16'd1) am_q <= AM_BAR2;
            end
            AM_BAR2: if (lm_idle) begin
              am_q   <= AM_L;
              amcl_q <= 2'd0;
            end
            AM_L: if (load) begin
              if (amcl_q != 2'd2) begin
                amcl_q <= amcl_q + 2'd1;
              end else begin
                amcl_q <= 2'd0;
                ami_q  <= (ami_q == am_Rl - 16'd1) ? 16'd0 : ami_q + 16'd1;
                if (ami_q == am_Rl - 16'd1) am_q <= AM_MR;
              end
            end
            AM_BARI: if (lm_idle) am_q <= AM_LI;
            AM_LI, AM_MI, AM_MR: if (load) begin
              if (ami_q == am_Rl - 16'd1) begin
                ami_q <= '0;
                am_q  <= (am_q == AM_LI) ? AM_MI : (NUM_CH == 1) ? AM_MC : AM_MT;
                amp_q <= '0;
                amb_q <= '0;
                ambe_q <= (am_NL <= 16'(NB)) ? am_NL - 16'd1 : 16'(NB - 1);
                amg_q <= 16'hFFFF;
                amcl_q <= 2'd0;
              end else begin
                ami_q <= ami_q + 16'd1;
              end
            end
            AM_MT: if (load) begin
              if (ami_q == ambe_q) begin
                ami_q <= amb_q;
                if (amp_q == 4'(NP - 1)) am_q <= AM_MBAR;
                amp_q <= amp_q + 4'd1;
              end else begin
                ami_q <= ami_q + 16'd1;
              end
            end
            AM_MBAR: if (lm_idle) begin
              am_q   <= AM_MC;
              amcl_q <= (amg_q == 16'hFFFF) ? 2'd0 : amcl_q;
            end
            AM_MC: if (load) begin
              if (nxt_cl) begin
                if (nxt.kind == SK_CL1) begin
                  amg_q  <= (ami_q >> am_dsh) >> 4;
                  amcl_q <= 2'd1;
                end else begin
                  amcl_q <= 2'd2;
                end
              end else if (ami_q == am_NL - 16'd1) begin
                st_q         <= G_IDLE;
                gen_done     <= 1'b1;
                gen_done_tag <= tag_q;
              end else if (ami_q == ambe_q) begin
                ami_q <= ami_q + 16'd1;
                amb_q <= ami_q + 16'd1;
                ambe_q <= bend;
                amp_q <= '0;
                am_q  <= (NUM_CH == 1) ? AM_MC : AM_MT;
              end else begin
                ami_q <= ami_q + 16'd1;
              end
            end
            default: am_q <= AM_PRE;
          endcase
        end
        default: st_q <= G_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------------------------
  // Setup: decode the op into the pass registers and opk (G_SETUP)
  // -------------------------------------------------------------------------------------------
  always_comb begin
    logic [23:0] m;
    setup_next = G_FIN;
    m = w_q[5][23:0];
    case (op_q)
      tsr_tables_pkg::TSR_OP_VOP:  setup_next = (w_q[6][15:0] != 16'd0 && m != 24'd0) ? G_PASS : G_FIN;
      tsr_tables_pkg::TSR_OP_RED:  setup_next = (w_q[0][12] && m == 24'd0) ? G_REDE :
                                (w_q[6][15:0] != 16'd0 && m != 24'd0) ? G_PASS : G_FIN;
      tsr_tables_pkg::TSR_OP_SOP:  setup_next = G_SOP;
      tsr_tables_pkg::TSR_OP_ROPE: setup_next = (w_q[6][15:0] != 16'd0 && w_q[6][31:16] != 16'd0) ?
                                ((w_q[0][12] || w_q[0][14]) ? G_RTBL : G_RPOS0) : G_FIN;
      TSR_OP_ASMAX, TSR_OP_AMASK: setup_next = G_PASS;     // R >= 1, n_lines >= 1 (AT §5.8)
      TSR_OP_ANORM: setup_next = G_PASS;
      TSR_OP_AMERGE: setup_next = G_AM;
      tsr_tables_pkg::TSR_OP_TOPK: setup_next = (HAS_TOPK && w_q[6][15:0] != 16'd0 && m != 24'd0) ? G_PASS : G_FIN;
      tsr_tables_pkg::TSR_OP_DQROW: setup_next = (HAS_DQROW && w_q[6][15:0] != 16'd0) ? G_PASS : G_FIN;
      tsr_tables_pkg::TSR_OP_GDN:   setup_next = HAS_GDN ? G_GSEQ : G_FIN;
      default:     setup_next = G_FIN;
    endcase
  end

  always_ff @(posedge clk) begin
    if (st_q == G_IDLE && start) begin
      // constants of the op (s registers sampled at the start, VU §4.4)
      opk[start_tag] <= '0;
      opk[start_tag].k1 <= sreg[start_w[0][27:24]];                  // VOP cmode 1: s[si]
      opk[start_tag].rkind <= 2'(start_w[0][9:8]);
      opk[start_tag].post  <= start_w[0][21:20];
      opk[start_tag].sk    <= sreg[start_w[0][27:24]];
      opk[start_tag].se    <= sreg[start_w[0][31:28]];
      opk[start_tag].to_s  <= start_w[0][12];
      opk[start_tag].sd    <= start_w[0][19:16];
      opk[start_tag].rd    <= TSR_VU_LW'(start_w[1]);
      opk[start_tag].rows  <= start_w[6][15:0];
      opk[start_tag].k     <= start_w[0][20:16];
      case (start_w[0][7:0])
        tsr_tables_pkg::TSR_OP_SOP: begin
          // a = s[sa]; b = s[sb], or imm for ADDI / MULI; MOVI: c1 = imm
          opk[start_tag].k1 <= (start_w[0][11:8] == TSR_SOP_SUB_MOVI) ? start_w[2] : sreg[start_w[1][7:4]];
          opk[start_tag].k2 <= (start_w[0][11:8] == TSR_SOP_SUB_ADDI || start_w[0][11:8] == TSR_SOP_SUB_MULI) ?
                               start_w[2] : sreg[start_w[1][11:8]];
          opk[start_tag].sd <= start_w[1][3:0];
          opk[start_tag].post <= start_w[0][11:8] == TSR_SOP_SUB_SQRT ? 2'd1 :
                                 start_w[0][11:8] == TSR_SOP_SUB_RSQRT ? 2'd2 : 2'd3;   // scalar path mode
        end
        TSR_OP_ASMAX, TSR_OP_AMERGE: begin
          opk[start_tag].k2 <= start_w[6];                            // scale
        end
        tsr_tables_pkg::TSR_OP_GDN: begin
          opk[start_tag].k1 <= sreg[start_w[0][19:16]];               // alpha
          opk[start_tag].k2 <= sreg[start_w[0][23:20]];               // beta
        end
        TSR_OP_AMASK: begin
          opk[start_tag].rkind <= 2'd3;
          opk[start_tag].post  <= 2'd0;
          opk[start_tag].to_s  <= 1'b0;
          opk[start_tag].rd    <= TSR_VU_LW'(start_w[7]);             // mdst
          opk[start_tag].keep  <= 1'b1;
          opk[start_tag].rows  <= start_w[5][15:0];
        end
        default: ;
      endcase
    end

  end

  // ROPE configuration for the angle unit (from the captured words and the s bank at the start)
  logic [31:0] rm_q, rts_q, rfs_q;
  always_ff @(posedge clk) begin
    if (st_q == G_IDLE && start) begin
      rm_q  <= sreg[start_w[0][19:16]];
      rts_q <= sreg[start_w[0][23:20]];
      rfs_q <= sreg[start_w[0][27:24]];
    end
  end
  assign rope_neox = (sub == TSR_ROPE_MODE_NEOX);
  assign rope_ff   = w_q[0][12];
  assign rope_fs   = w_q[0][13];
  assign rope_atab = w_q[0][14];
  assign rope_m    = rm_q;
  assign rope_ts   = rts_q;
  assign rope_fsv  = rfs_q;
  assign rope_na   = 10'(w_q[5][31:17]);
  assign rope_t    = tp_q;

  wire unused_ok = &{1'b0, hd_q, H_q, x16[15:8], NUM_CH[0], bsl_q[31:24], nd_q[4:0], hl_q[31:24], ssp_q[31:24],
                     dsp_q[31:24], am_cs[31:24]};

endmodule

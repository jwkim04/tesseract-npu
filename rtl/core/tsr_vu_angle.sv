// tsr_vu_angle.sv — the VU's ROPE angle unit: per token the n_dims/2 rotation constants c_i = FM(M, COS(φ_i)),
// s_i = FM(M, SIN(φ_i)) into one half of the double-buffered angle buffer, while the lanes rotate the
// previous token.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §6.7 angles, §7.6 angle unit and ROPE engine).
// Angles (VU §6.7): Pf = i32_to_f32(pos[t]);
//   recurrence form: θ_0 = Pf, θ_{i+1} = FM(θ_i, TS) (one FM chain: one angle per 4 cycles); φ_i = FF ?
//                    FD(θ_i, ff_i) : θ_i (two iterative tsr_fp_div taking angles in turn); φ_i <- FS ?
//                    FM(freq_scale, φ_i) : φ_i
//   A-table form:    φ_i = FM(Pf, A_i) (one per cycle)
// then tsr_vu_sincos and the two FM by M, written to bank i mod 16, entry {half, i / 16}. The table
// (ff_i or A_i, n_dims/2 words) is loaded once per op from the table lines (SK_TBL); the positions from
// position lines (SK_POS) into a two-line buffer. Token t uses half t mod 2; the half is released by the
// injector after the token's last rotation slot (ang_rel). ang_ready[h]: half h holds the angles of
// the next token to rotate in it.
// The read port is combinational: all 16 banks at (half, entry).

module tsr_vu_angle
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(parameter bit LOCAL_ANGLE = 0) (
  input  logic                       clk,
  input  logic                       rst_n,
  // configuration (tsr_vu_gen, at the op's start)
  input  logic                       cfg_start,
  input  logic                       cfg_neox,
  input  logic                       cfg_ff,
  input  logic                       cfg_fs,
  input  logic                       cfg_atab,
  input  logic [31:0]                cfg_m,
  input  logic [31:0]                cfg_ts,
  input  logic [31:0]                cfg_fsv,
  input  logic [9:0]                 cfg_na,
  input  logic [15:0]                cfg_t,
  // table / position lines (tsr_vu_inj)
  input  logic                       tbl_we,
  input  logic [4:0]                 tbl_line,
  input  logic                       pos_we,
  input  logic [15:0]                pos_idx,
  input  logic [TSR_LINE_BITS-1:0]   pa_data,
  // angle buffer
  output logic [1:0]                 ang_ready,
  input  logic                       ang_rel,
  input  logic                       ang_rel_h,
  input  logic                       rd_h,
  input  logic [4:0]                 rd_e,
  output logic [15:0][63:0]          rd_cs,
  output tsr_vu_bank_angle_t [1:0]    bank_angle,
  output logic                       idle
);
  import tsr_fp_pkg::*;

  localparam int unsigned FML = TSR_VU_FMUL_LAT, SCL = TSR_VU_SC_LAT;

  logic        ff_q, fs_q, atab_q;
  logic [31:0] m_q, ts_q, fsv_q;
  logic [9:0]  na_q;
  logic [15:0] nt_q, t_q;                                // tokens, the token being generated
  logic        gen_q;                                    // generating φ of token t_q
  logic        act_q;                                    // the op is active (tokens left)
  logic [1:0]  hfull_q;                                  // half holds a complete token
  logic [1:0][9:0] wcnt_q;                               // angles written per half
  logic [9:0]  i_q;                                      // next angle index
  logic [31:0] th_q;                                     // θ_i (recurrence)
  logic        thv_q;                                    // θ_i valid in th_q
  logic [2:0]  fmw_q;                                    // cycles until θ_{i+1}
  logic [31:0] pf_q;
  logic        pfv_q;                                    // Pf computed for t_q


  // -------------------------------------------------------------------------------------------
  // Table RAM (16 word banks x 32 entries) and the position buffer
  // -------------------------------------------------------------------------------------------
  // one LUTRAM per word bank (32 x 32, one write, one asynchronous read)
  logic [5:0]        tbl_n_q;                            // lines loaded
  logic [15:0][31:0] tbl_e;                              // the 16 banks at entry i / 16
  logic [31:0]       tbl_w;                              // word i
  for (genvar b = 0; b < 16; b++) begin : g_tbl
    logic [31:0] mem [32];
    always_ff @(posedge clk) begin
      if (tbl_we) mem[tbl_line] <= pa_data[32*b +: 32];
    end
    assign tbl_e[b] = mem[i_q[8:4]];
  end
  assign tbl_w = tbl_e[i_q[3:0]];
  logic [1:0][15:0][31:0] pos_q;
  logic [1:0][15:0]       posl_q;                        // line index held by each half
  logic [1:0]             posv_q;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      posv_q  <= '0;
      tbl_n_q <= '0;
    end else begin
      if (cfg_start) begin
        posv_q  <= '0;
        tbl_n_q <= '0;
      end
      if (pos_we) begin
        posv_q[pos_idx[0]] <= 1'b1;
      end
      if (tbl_we) tbl_n_q <= tbl_n_q + 6'd1;
    end
    if (pos_we) begin
      pos_q[pos_idx[0]]  <= pa_data;
      posl_q[pos_idx[0]] <= pos_idx;
    end
  end

  // -------------------------------------------------------------------------------------------
  // Token sequencing
  // -------------------------------------------------------------------------------------------
  logic        wr_v, wr_h;                               // angle write (end of the pipeline)
  logic [9:0]  wr_i;
  wire [5:0] tbl_need = 6'((32'(cfg_na) * 2 + 31) / 32);
  logic [5:0] tbl_need_q;
  wire        h      = t_q[0];
  wire        pos_ok = posv_q[t_q[4]] && (posl_q[t_q[4]] == {4'd0, t_q[15:4]});
  wire        tbl_ok = !(ff_q || atab_q) || (tbl_n_q >= tbl_need_q);
  wire        can_start = act_q && !gen_q && !hfull_q[h] && (wcnt_q[h] == 10'd0) && pos_ok && tbl_ok;

  // Pf = i32_to_f32(pos[t]): the position word is registered first (pw_q), then converted in two stages
  wire [31:0] pos_w = pos_q[t_q[4]][t_q[3:0]];
  logic [31:0] pw_q;
  logic        pwv_q, pwv2_q;
  logic        pc_neg_q;
  logic [31:0] pc_mag_q;                                 // |pos|; INT32_MIN -> 2^31
  logic [4:0]  pc_lz_q;
  always_ff @(posedge clk) begin
    logic [31:0] mag;
    logic [4:0]  lz;
    mag = pw_q[31] ? (32'd0 - pw_q) : pw_q;
    lz  = 5'd31;
    for (int i = 0; i < 32; i++) if (mag[i]) lz = 5'(31 - i);
    pc_neg_q <= pw_q[31];
    pc_mag_q <= mag;
    pc_lz_q  <= lz;
  end
  logic [31:0] pf_pack;
  always_comb begin
    logic [30:0] nrm;
    logic [23:0] fr;
    logic        inc;
    nrm = 31'(pc_mag_q << pc_lz_q);
    inc = nrm[7] & ((nrm[6:0] != 7'd0) | nrm[8]);
    fr  = {1'b0, nrm[30:8]} + {23'd0, inc};
    pf_pack = (pc_mag_q == 32'd0) ? 32'd0 : {pc_neg_q, 8'(8'd158 - {3'd0, pc_lz_q} + {7'd0, fr[23]}), fr[22:0]};
  end

  // φ producer
  logic        ph_v;                                     // φ_i (before FS) valid this cycle
  logic [31:0] ph_x;
  logic [9:0]  ph_i;
  logic        ph_h;
  logic        fm_v;
  logic [31:0] fm_a, fm_b, fm_y;
  logic        fmo;
  tsr_fp_mul #(.LAT(FML)) u_fmr (.clk, .rst_n, .ce(1'b1), .in_valid(fm_v), .a(fm_a), .b(fm_b), .out_valid(fmo), .y(fm_y));

  // dividers (FF): angle i goes to divider i mod 2
  logic [1:0]       dv_in_v, dv_in_rdy, dv_out_v;
  logic [1:0][31:0] dv_y;
  logic [1:0][9:0]  dv_i;
  logic [1:0]       dv_h;
  for (genvar d = 0; d < 2; d++) begin : g_dv
    tsr_fp_div u_dv (
      .clk, .rst_n, .in_valid(dv_in_v[d]), .in_ready(dv_in_rdy[d]), .a(th_q), .b(tbl_w),
      .out_valid(dv_out_v[d]), .out_ready(1'b1), .y(dv_y[d])
    );
    always_ff @(posedge clk) begin
      if (dv_in_v[d] && dv_in_rdy[d]) begin
        dv_i[d] <= i_q;
        dv_h[d] <= h;
      end
    end
  end

  // ATAB pipeline tags (FM(Pf, A_i), one per cycle)
  logic [FML-1:0]       at_v;
  logic [FML-1:0][9:0]  at_i;
  logic [FML-1:0]       at_h;

  always_comb begin
    fm_v    = 1'b0;
    fm_a    = th_q;
    fm_b    = ts_q;
    dv_in_v = '0;
    if (gen_q && atab_q) begin
      fm_v = 1'b1;
      fm_a = pf_q;
      fm_b = tbl_w;
    end else if (gen_q && thv_q) begin
      if (ff_q) dv_in_v[i_q[0]] = 1'b1;
      fm_v = !ff_q || dv_in_rdy[i_q[0]];                 // θ_{i+1} once θ_i is taken
    end
  end
  wire th_take = gen_q && !atab_q && thv_q && (!ff_q || dv_in_rdy[i_q[0]]);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      act_q   <= 1'b0;
      gen_q   <= 1'b0;
      pfv_q   <= 1'b0;
      pwv_q   <= 1'b0;
      pwv2_q  <= 1'b0;
      thv_q   <= 1'b0;
      hfull_q <= '0;
      wcnt_q  <= '0;
      fmw_q   <= '0;
      at_v    <= '0;
    end else begin
      if (cfg_start) begin
        act_q      <= (cfg_t != 16'd0);
        nt_q       <= cfg_t;
        t_q        <= '0;
        ff_q       <= cfg_ff;
        fs_q       <= cfg_fs;
        atab_q     <= cfg_atab;
        m_q        <= cfg_m;
        ts_q       <= cfg_ts;
        fsv_q      <= cfg_fsv;
        na_q       <= cfg_na;
        tbl_need_q <= tbl_need;
        hfull_q    <= '0;
        pfv_q      <= 1'b0;
      end
      // Pf of the next token (three stages: the position word, sign / magnitude / leading zeros, the
      // rounded f32 = tsr_i32_to_f32)
      pwv_q  <= act_q && !gen_q && !pfv_q && pos_ok;
      pwv2_q <= pwv_q && act_q && !gen_q && !pfv_q;
      if (pwv2_q && act_q && !gen_q && !pfv_q) begin
        pf_q  <= pf_pack;
        pfv_q <= 1'b1;
      end
      // recurrence: θ_i taken -> θ_{i+1} after the FM
      if (th_take) begin
        thv_q <= 1'b0;
        fmw_q <= 3'(FML);
        if (i_q == na_q - 10'd1) begin
          gen_q <= 1'b0;
          pfv_q <= 1'b0;
          t_q   <= t_q + 16'd1;
          if (t_q == nt_q - 16'd1) act_q <= 1'b0;
        end
        i_q <= i_q + 10'd1;
      end
      if (fmw_q != 3'd0 && !th_take) begin
        fmw_q <= fmw_q - 3'd1;
        if (fmw_q == 3'd1 && gen_q) begin
          th_q  <= fm_y;
          thv_q <= 1'b1;
        end
      end
      // a new token (after the previous token's last θ was taken; its FM result is dropped)
      if (can_start && pfv_q) begin
        gen_q <= 1'b1;
        i_q   <= '0;
        th_q  <= pf_q;
        thv_q <= 1'b1;
        fmw_q <= '0;
      end
      // A-table form: one angle per cycle
      at_v <= {at_v[FML-2:0], gen_q && atab_q};
      if (gen_q && atab_q) begin
        if (i_q == na_q - 10'd1) begin
          gen_q <= 1'b0;
          pfv_q <= 1'b0;
          t_q   <= t_q + 16'd1;
          if (t_q == nt_q - 16'd1) act_q <= 1'b0;
        end
        i_q <= i_q + 10'd1;
      end
      // completion per half; release by the injector
      for (int hh = 0; hh < 2; hh++) begin
        if (wr_v && wr_h == 1'(hh)) begin
          if (wcnt_q[hh] == na_q - 10'd1) begin
            wcnt_q[hh]  <= '0;
            hfull_q[hh] <= 1'b1;
          end else begin
            wcnt_q[hh] <= wcnt_q[hh] + 10'd1;
          end
        end
        if (ang_rel && ang_rel_h == 1'(hh)) hfull_q[hh] <= 1'b0;
      end
    end
    at_i <= {at_i[FML-2:0], i_q};
    at_h <= {at_h[FML-2:0], h};
    pw_q <= pos_w;
  end

  // φ before FS: the recurrence's θ_i (no FF), a divider output (FF), or the A-table product
  logic        rc_v;                                     // θ_i taken without FF: φ_i = θ_i
  logic [31:0] rc_x;
  logic [9:0]  rc_i;
  logic        rc_h;
  always_ff @(posedge clk) begin
    if (!rst_n) rc_v <= 1'b0;
    else        rc_v <= th_take && !ff_q;
    rc_x <= th_q;
    rc_i <= i_q;
    rc_h <= h;
  end
  always_comb begin
    ph_v = 1'b0;
    ph_x = rc_x;
    ph_i = rc_i;
    ph_h = rc_h;
    if (rc_v) ph_v = 1'b1;
    for (int d = 0; d < 2; d++) begin
      if (dv_out_v[d]) begin
        ph_v = 1'b1;
        ph_x = dv_y[d];
        ph_i = dv_i[d];
        ph_h = dv_h[d];
      end
    end
    if (at_v[FML-1]) begin
      ph_v = 1'b1;
      ph_x = fm_y;
      ph_i = at_i[FML-1];
      ph_h = at_h[FML-1];
    end
  end

  // FS: φ <- FM(freq_scale, φ) (bypassed with the same latency)
  logic [31:0] fs_y, ph_d;
  logic        ofs;
  tsr_fp_mul #(.LAT(FML)) u_fs (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(fsv_q), .b(ph_x), .out_valid(ofs), .y(fs_y));
  tsr_vu_dly #(.W(32), .D(FML)) d_phd (.clk, .d(ph_x), .q(ph_d));
  logic [FML-1:0] f_v;
  logic [9:0]     f_i;
  logic           f_h;
  always_ff @(posedge clk) begin
    if (!rst_n) f_v <= '0;
    else        f_v <= {f_v[FML-2:0], ph_v};
  end
  tsr_vu_dly #(.W(11), .D(FML)) d_fih (.clk, .d({ph_i, ph_h}), .q({f_i, f_h}));
  wire [31:0] phi = fs_q ? fs_y : ph_d;

  // SIN / COS, then c = FM(M, COS), s = FM(M, SIN)
  logic [31:0] sn, cs, mc, ms;
  logic        omc, oms;
  tsr_vu_sincos u_sc (.clk, .rst_n, .t(phi), .s(sn), .c(cs));
  tsr_fp_mul #(.LAT(FML)) u_mc (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(m_q), .b(cs), .out_valid(omc), .y(mc));
  tsr_fp_mul #(.LAT(FML)) u_ms (.clk, .rst_n, .ce(1'b1), .in_valid(1'b1), .a(m_q), .b(sn), .out_valid(oms), .y(ms));
  localparam int unsigned WL = SCL + FML;
  logic [WL-1:0] w_sr;
  always_ff @(posedge clk) begin
    if (!rst_n) w_sr <= '0;
    else        w_sr <= {w_sr[WL-2:0], f_v[FML-1]};
  end
  assign wr_v = w_sr[WL-1];
  tsr_vu_dly #(.W(11), .D(WL)) d_wih (.clk, .d({f_i, f_h}), .q({wr_i, wr_h}));

  if (LOCAL_ANGLE) begin : g_bank_writes
    logic neox_q;
    always_ff @(posedge clk) begin
      if (cfg_start) neox_q <= cfg_neox;
    end
    wire owner = neox_q ? wr_i[1] : wr_i[0];
    wire [2:0] idx = {wr_i[3:2], neox_q ? wr_i[0] : wr_i[1]};
    for (genvar bank = 0; bank < 2; bank++) begin : g_packet
      assign bank_angle[bank].valid = wr_v && owner == 1'(bank);
      assign bank_angle[bank].addr = {wr_h, wr_i[8:4], idx};
      assign bank_angle[bank].data = {mc, ms};
    end
    assign rd_cs = '0;
    wire unused_rd = &{1'b0, rd_h, rd_e};
  end else begin : g_central_buffer
    assign bank_angle = '0;
    wire unused_neox = &{1'b0, cfg_neox};
    // One LUTRAM per angle index, with an asynchronous read.
    for (genvar b = 0; b < 16; b++) begin : g_ab
      logic [63:0] mem [64];
      always_ff @(posedge clk) begin
        if (wr_v && wr_i[3:0] == 4'(b)) mem[{wr_h, wr_i[8:4]}] <= {mc, ms};
      end
      assign rd_cs[b] = mem[{rd_h, rd_e}];
    end
  end

  assign ang_ready = hfull_q;
  assign idle = !act_q && !gen_q && (f_v == '0) && (w_sr == '0) && !rc_v && (dv_out_v == '0) && (at_v == '0);

  wire unused_ok = &{1'b0, fmo, ofs, omc, oms, wr_i[9], i_q[9], f_i[9], dv_i[0][9], dv_i[1][9], pos_idx[15:1]};

endmodule

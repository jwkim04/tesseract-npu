// tsr_vu_lanes.sv — the VU's 16 lanes behind registered pipes: the lane array as one block that can sit in
// another SLR than the SPM-facing part of the unit (reader, injector, writer, reduction).
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §7.10 placement, remedy 3 of CC §14.1; floorplan C'' of
// hw/u200/k4_floorplan.tcl).
// Contract: with PIPE = 0 the lanes are wired straight to the ports, exactly as tsr_vu used to instantiate them.
// With PIPE = p >= 1, the injector's lane inputs (l_*), the partial drain address (dr_addr) and the reset
// pass p register stages on the way in, and the lane exits f_y / b_y and the drain read data dr_data p
// stages on the way out: every lane result leaves 2p cycles later than the lane's own latency
// (TSR_VU_LF / TSR_VU_LB), a drain read returns 2p cycles after its address, and nothing else changes.
// The consumers allow for it (tsr_vu_inj and tsr_vu_wr: LANE_X = 2p, tsr_vu_red: DRL = 2p); the lanes
// have no back-pressure, so no handshake needs to cross.
// Why the reset is piped like the inputs: a lane leaves reset p cycles after the injector does, which is
// exactly when the first element of a later op can arrive, so the reset stays in step with the data.
// Placement (hw/u200/k4_floorplan.tcl, PIPE >= 2): the first stage of the input pipes and the last stage
// of the output pipes are pinned beside the injector / writer, the other stages and the lanes in the lane
// SLR, so each SLR crossing is one register-to-register hop. The stage registers carry
// shreg_extract = "no" (no SRL).

module tsr_vu_lanes
  import tsr_vu_pkg::*;
#(
  parameter int unsigned EXP_LANES  = 16,
  parameter int unsigned BACK_LANES = 8,
  parameter int unsigned PIPE       = 0          // register stages on each side of the lanes, >= 0
) (
  input  logic              clk,
  input  logic              rst_n,
  // element and reduction controls of the injector (registered there)
  input  logic [15:0]       l_v,
  input  tsr_vu_uop_e       l_uop,
  input  logic [15:0][31:0] l_a,
  input  logic [15:0][31:0] l_b,
  input  logic [15:0][31:0] l_c1,
  input  logic [15:0][31:0] l_c2,
  input  logic [15:0]       l_neg,
  input  logic [15:0]       l_lv,
  input  logic              l_fresh,
  input  logic [3:0]        l_pa,
  // partial drain (tsr_vu_red): address in, data PIPE + PIPE cycles later
  input  logic [3:0]        dr_addr,
  output logic [15:0][31:0] dr_data,
  // lane exits
  output logic [15:0][31:0] f_y,
  output logic [15:0][31:0] b_y
);

  // everything that goes into the lane array, and everything that comes out of it
  typedef struct packed {
    logic [15:0]       v;
    logic [5:0]        uop;
    logic [15:0][31:0] a, b, c1, c2;
    logic [15:0]       neg, lv;
    logic              fresh;
    logic [3:0]        pa;
    logic [3:0]        dr_addr;
  } lin_t;
  typedef struct packed {
    logic [15:0][31:0] dr_data, f_y, b_y;
  } lout_t;

  localparam int unsigned IW = $bits(lin_t);
  localparam int unsigned OW = $bits(lout_t);

  lin_t  li, lo_in;                  // lane array input: the ports / after the input pipe
  lout_t lq, lp;                     // lane array output: the lanes' / after the output pipe
  logic  lane_rst_n;

  assign li.v       = l_v;
  assign li.uop     = 6'(l_uop);
  assign li.a       = l_a;
  assign li.b       = l_b;
  assign li.c1      = l_c1;
  assign li.c2      = l_c2;
  assign li.neg     = l_neg;
  assign li.lv      = l_lv;
  assign li.fresh   = l_fresh;
  assign li.pa      = l_pa;
  assign li.dr_addr = dr_addr;

  if (PIPE == 0) begin : g_wire
    assign lo_in      = li;
    assign lane_rst_n = rst_n;
    assign lp         = lq;
  end else begin : g_pipe
    // input side, p stages; the reset rides along (its flops are not reset themselves: during a reset
    // pulse of >= p cycles the chain fills with zeros, and a flop starts at 0 on the device)
    (* shreg_extract = "no" *) logic [PIPE-1:0][IW-1:0] ip_q;
    (* shreg_extract = "no" *) logic [PIPE-1:0]         rp_q;
    always_ff @(posedge clk) begin
      ip_q[0] <= li;
      rp_q[0] <= rst_n;
      for (int unsigned i = 1; i < PIPE; i++) begin
        ip_q[i] <= ip_q[i-1];
        rp_q[i] <= rp_q[i-1];
      end
    end
    assign lo_in      = lin_t'(ip_q[PIPE-1]);
    assign lane_rst_n = rp_q[PIPE-1];
    // output side, p stages
    (* shreg_extract = "no" *) logic [PIPE-1:0][OW-1:0] op_q;
    always_ff @(posedge clk) begin
      op_q[0] <= lq;
      for (int unsigned i = 1; i < PIPE; i++) op_q[i] <= op_q[i-1];
    end
    assign lp = lout_t'(op_q[PIPE-1]);
  end

  for (genvar l = 0; l < 16; l++) begin : g_lane
    tsr_vu_lane #(.HAS_EXP(l < EXP_LANES), .HAS_BACK(l < BACK_LANES)) u_lane (
      .clk, .rst_n(lane_rst_n), .in_v(lo_in.v[l]), .in_uop(tsr_vu_uop_e'(lo_in.uop)), .in_a(lo_in.a[l]),
      .in_b(lo_in.b[l]), .in_c1(lo_in.c1[l]), .in_c2(lo_in.c2[l]), .in_neg(lo_in.neg[l]), .acc_lv(lo_in.lv[l]),
      .acc_fresh(lo_in.fresh), .acc_pa(lo_in.pa), .dr_addr(lo_in.dr_addr), .dr_data(lq.dr_data[l]),
      .f_y(lq.f_y[l]), .b_y(lq.b_y[l])
    );
  end

  assign dr_data = lp.dr_data;
  assign f_y     = lp.f_y;
  assign b_y     = lp.b_y;

endmodule

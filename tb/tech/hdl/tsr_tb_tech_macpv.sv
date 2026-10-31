// tsr_tb_tech_macpv.sv — tb/tech toplevel (W4-B at): tsr_mac_lane with PV = 1 (ATTPV's accumulate
// mode, AT §8.2) in every configuration the attention clusters can build, side by side on shared
// inputs. Verilated twice (rtl/tech/generic and rtl/tech/xilinx, models Vgen / Vxil);
// tb/tech/cpp/macpv_equiv.cpp drives both and compares ov / yy (the MVM mode of the same lanes)
// and pva (the PV accumulators) in every cycle.
//
// Configurations (index c, the order macpv_equiv.cpp rebuilds):
//   N 32: T {1, 2, 3, 8} x LAT {3, 4, 5, 6, 7, 9, 11}      c = 0 .. 27
//   N 5 : T 3 x LAT {3, 6}                                 c = 28, 29
// yy[c]: y in its low T x SUM_W bits; pva[c]: pv_acc in its low NS x N x 38 bits.

module tsr_tb_tech_macpv #(
  localparam int unsigned NC  = 30,
  localparam int unsigned YW  = 8 * 21,
  localparam int unsigned PW  = 4 * 32 * 38
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    ce,
  input  logic                    in_valid,
  input  logic [31:0][7:0]        w,
  input  logic [7:0][31:0][7:0]   x,
  input  logic                    pv_en,
  input  logic                    pv_valid,
  input  logic                    pv_first,
  input  logic [3:0][24:0]        pv_a,
  output logic [NC-1:0]           ov,
  output logic [NC-1:0][YW-1:0]   yy,
  output logic [NC-1:0][PW-1:0]   pva
);

  localparam int unsigned TS [4] = '{1, 2, 3, 8};
  localparam int unsigned LS [7] = '{3, 4, 5, 6, 7, 9, 11};

  for (genvar c = 0; c < NC; c++) begin : g_c
    localparam int unsigned N   = (c < 28) ? 32 : 5;
    localparam int unsigned T   = (c < 28) ? TS[c / 7] : 3;
    localparam int unsigned LAT = (c < 28) ? LS[c % 7] : ((c == 28) ? 3 : 6);
    localparam int unsigned SW  = 16 + $clog2(N);
    localparam int unsigned NS  = (T + 1) / 2;

    logic [N-1:0][7:0]          wi;
    logic [T-1:0][N-1:0][7:0]   xi;
    logic [T-1:0][SW-1:0]       yo;
    logic [NS-1:0][N-1:0][37:0] po;
    assign wi = w[N-1:0];
    for (genvar t = 0; t < T; t++) begin : g_t
      assign xi[t] = x[t][N-1:0];
    end

    tsr_mac_lane #(.N(N), .T(T), .LAT(LAT), .PV(1'b1)) u_mac (
      .clk, .rst_n, .ce, .in_valid, .w(wi), .x(xi), .out_valid(ov[c]), .y(yo),
      .pv_en, .pv_valid, .pv_first, .pv_a(pv_a[NS-1:0]), .pv_acc(po)
    );

    assign yy[c]  = YW'(yo);
    assign pva[c] = PW'(po);
  end

  wire unused_ok = &{1'b0, w, x, pv_a};

endmodule

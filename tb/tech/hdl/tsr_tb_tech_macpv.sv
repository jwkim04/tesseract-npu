// tsr_tb_tech_macpv.sv — tb/tech toplevel: tsr_mac_lane with PV = 1 (ATTPV's accumulate
// mode, AT §8.2) in every configuration the attention clusters can build, side by side on shared
// inputs. Verilated twice (rtl/tech/generic and rtl/tech/xilinx, models Vgen / Vxil);
// tb/tech/cpp/macpv_equiv.cpp drives both and compares ov / yy (the MVM mode of the same lanes)
// and pva (the PV accumulators) in every cycle.
//
// Configurations (index c, the order macpv_equiv.cpp rebuilds):
//   N 32: T {1, 2, 3, 8} x LAT {3, 4, 5, 6, 7, 9, 11}      c = 0 .. 27
//   N 5 : T 3 x LAT {3, 6}                                 c = 28, 29
//   N 32, dual weight stream (TH = T / 2; the second weight vector equals the first, as in the PV mode of tsr_mvm_back):
//         T 4 x LAT 9, T 2 x LAT 5, T 6 x LAT 6                c = 30, 31, 32
//   PVR 1 (pv_en, the accumulator select and the P enable term through registers, as FAST_CLK builds the lanes):
//         N 32 T 2 x LAT 3, N 32 T 8 x LAT 9, N 5 T 3 x LAT 6, N 32 T 4 (TH 2) x LAT 9, N 32 T 3 x LAT 4
//                                                              c = 33 .. 37
// yy[c]: y in its low T x SUM_W bits; pva[c]: pv_acc in its low NS x N x 38 bits.

module tsr_tb_tech_macpv #(
  localparam int unsigned NC  = 38,
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
    localparam int unsigned N   = (c < 28 || (c >= 30 && c != 35)) ? 32 : 5;
    localparam int unsigned T   = (c < 28) ? TS[c / 7] : (c < 30) ? 3 : (c == 30) ? 4 : (c == 31) ? 2 : (c == 32) ? 6
                                : (c == 33) ? 2 : (c == 34) ? 8 : (c == 35) ? 3 : (c == 36) ? 4 : 3;
    localparam int unsigned LAT = (c < 28) ? LS[c % 7] : (c < 30) ? ((c == 28) ? 3 : 6) : (c == 30) ? 9 : (c == 31) ? 5 : (c == 32) ? 6
                                : (c == 33) ? 3 : (c == 34) ? 9 : (c == 35) ? 6 : (c == 36) ? 9 : 4;
    localparam int unsigned TH  = (c < 30 || c == 33 || c == 34 || c == 35 || c == 37) ? 0 : T / 2;
    localparam bit          PVR = (c >= 33);
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

    tsr_mac_lane #(.N(N), .T(T), .LAT(LAT), .PV(1'b1), .PVR(PVR), .TH(TH)) u_mac (
      .clk, .rst_n, .ce, .in_valid, .w(wi), .wb(wi), .x(xi), .out_valid(ov[c]), .y(yo),
      .pv_en, .pv_valid, .pv_first, .pv_a(pv_a[NS-1:0]), .pv_acc(po)
    );

    assign yy[c]  = YW'(yo);
    assign pva[c] = PW'(po);
  end

  wire unused_ok = &{1'b0, w, x, pv_a};

endmodule

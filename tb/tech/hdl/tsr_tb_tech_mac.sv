// tsr_tb_tech_mac.sv — tb/tech toplevel: every tsr_mac_lane configuration of the technology
// equivalence test side by side on shared inputs. Verilated twice, once with rtl/tech/generic and
// once with rtl/tech/xilinx (tb/tech/test_tech.py, models Vgen / Vxil); tb/tech/cpp/mac_equiv.cpp
// drives both with the same stimulus and compares ov / yy in every cycle.
//
// Configurations (index c, the order mac_equiv.cpp rebuilds):
//   N 32: T 1..8 x LAT 1..11      c = 0 .. 87      (every T_MAX of the core, every register
//                                                   allocation of the xilinx lane + 2 extras)
//   N 5 : T 1..3 x LAT 1..6       c = 88 .. 105
//   N 1 : T 1..3 x LAT 1..4       c = 106 .. 117
//   N 2 : T 1..3 x LAT 1..4       c = 118 .. 129
//   N 3 : T 3    x LAT 1..6       c = 130 .. 135
//   dual weight stream (TH = T / 2, tokens >= TH take wb), N 32: T {2, 4, 6, 8} x LAT {1, 2, 3, 5, 6, 9}   c = 136 .. 159
// yy[c] holds y of configuration c in its low T x SUM_W bits (zero above).

module tsr_tb_tech_mac #(
  localparam int unsigned NC = 160,
  localparam int unsigned YW = 8 * 21
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    ce,
  input  logic                    in_valid,
  input  logic [31:0][7:0]        w,
  input  logic [31:0][7:0]        wb,
  input  logic [7:0][31:0][7:0]   x,
  output logic [NC-1:0]           ov,
  output logic [NC-1:0][YW-1:0]   yy
);

  for (genvar c = 0; c < NC; c++) begin : g_c
    localparam int unsigned DT [4] = '{2, 4, 6, 8};
    localparam int unsigned DL [6] = '{1, 2, 3, 5, 6, 9};
    localparam int unsigned N   = (c < 88) ? 32 : (c < 106) ? 5 : (c < 118) ? 1 : (c < 130) ? 2 : (c < 136) ? 3 : 32;
    localparam int unsigned T   = (c < 88)  ? c / 11 + 1 :
                                  (c < 106) ? (c - 88) / 6 + 1 :
                                  (c < 118) ? (c - 106) / 4 + 1 :
                                  (c < 130) ? (c - 118) / 4 + 1 :
                                  (c < 136) ? 3 : DT[(c - 136) / 6];
    localparam int unsigned LAT = (c < 88)  ? c % 11 + 1 :
                                  (c < 106) ? (c - 88) % 6 + 1 :
                                  (c < 118) ? (c - 106) % 4 + 1 :
                                  (c < 130) ? (c - 118) % 4 + 1 :
                                  (c < 136) ? c - 130 + 1 : DL[(c - 136) % 6];
    localparam int unsigned TH  = (c < 136) ? 0 : T / 2;
    localparam int unsigned SW  = 16 + $clog2(N);

    logic [N-1:0][7:0]        wi, wbi;
    logic [T-1:0][N-1:0][7:0] xi;
    logic [T-1:0][SW-1:0]     yo;
    assign wi  = w[N-1:0];
    assign wbi = wb[N-1:0];
    for (genvar t = 0; t < T; t++) begin : g_t
      assign xi[t] = x[t][N-1:0];
    end

    logic [(T+1)/2-1:0][N-1:0][37:0] pvo;          // PV mode off here (tsr_tb_tech_macpv)
    tsr_mac_lane #(.N(N), .T(T), .LAT(LAT), .TH(TH)) u_mac (
      .clk, .rst_n, .ce, .in_valid, .w(wi), .wb(wbi), .x(xi), .out_valid(ov[c]), .y(yo),
      .pv_en(1'b0), .pv_valid(1'b0), .pv_first(1'b0), .pv_a('0), .pv_acc(pvo)
    );
    wire unused_pv = &{1'b0, pvo};

    if (T * SW < YW) begin : g_pad
      assign yy[c] = {{(YW - T * SW){1'b0}}, yo};
    end else begin : g_full
      assign yy[c] = yo;
    end
  end

  // lanes / tokens beyond a configuration's N / T are not used by it
  wire unused_ok = &{1'b0, w, wb, x};

endmodule

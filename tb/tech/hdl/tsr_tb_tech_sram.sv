// tsr_tb_tech_sram.sv — tb/tech toplevel: tsr_sram_1r1w configurations of the technology
// equivalence test side by side on shared inputs. Verilated twice, once with rtl/tech/generic and
// once with rtl/tech/xilinx (tb/tech/test_tech.py, models Vgen / Vxil); tb/tech/cpp/sram_equiv.cpp
// drives both with the same stimulus and compares rd (every cycle; URAM sizes: every cycle with a
// read result).
//
// Configurations (DEPTH x WIDTH, RD_LAT; xilinx RAM style), index c as in sram_equiv.cpp:
//    0  16384 x 512, 1   ultra        whole-line URAM cascade (the SPM bank before its 64-bit
//    1  16384 x 512, 2   ultra          column slices, configs 16..20)
//    2  16384 x 512, 3   ultra
//    3   4096 x 512, 2   ultra
//    4   4096 x   8, 3   block        narrow (tb/spm edge configuration; < one URAM of bits)
//    5   1024 x 512, 3   block        weight FIFO, largest
//    6    512 x 512, 2   block        weight FIFO, U200 default (WS_RD_LAT 2)
//    7    256 x 512, 2   block        activation buffer QS bank, K_MAX 32768
//    8     72 x 512, 2   block        activation buffer DBS, K_MAX 32768
//    9     64 x 512, 1   block        QACT line buffer
//   10     64 x 512, 4   block        weight FIFO 64, WS_RD_LAT 4
//   11    100 x 512, 1   block        (tb/spm edge configuration)
//   12     37 x   8, 4   block
//   13     32 x 512, 2   distributed  (largest LUTRAM)
//   14      2 x 512, 1   distributed  activation buffer at K_MAX 256
//   15      1 x  72, 2   distributed
//   16  16384 x  64, 1   ultra        SPM column slice, SPM_RD_LAT 2 (IN_REG + RAM_LAT 1)
//   17  16384 x  64, 2   ultra        SPM slice, SPM_RD_LAT 3 / 4 (U200 4: IN_REG + 2 + OUT_REG)
//   18  16384 x  64, 3   ultra        SPM slice, SPM_RD_LAT 5
//   19   4096 x  64, 1   ultra        256 KiB SPM slice (one bank), SPM_RD_LAT 2
//   20   4096 x  64, 3   ultra        256 KiB SPM slice, SPM_RD_LAT 4 (one bank: no OUT_REG)
// Addresses are taken modulo DEPTH (identical in both builds), data / byte enables truncated.

module tsr_tb_tech_sram #(
  localparam int unsigned NC = 21
) (
  input  logic                    clk,
  input  logic                    we,
  input  logic [13:0]             waddr,
  input  logic [511:0]            wdata,
  input  logic [63:0]             wbe,
  input  logic                    re,
  input  logic [13:0]             raddr,
  output logic [NC-1:0][511:0]    rd
);

  localparam int unsigned CD [NC] = '{16384, 16384, 16384, 4096, 4096, 1024, 512, 256, 72, 64, 64, 100, 37, 32, 2, 1,
                                       16384, 16384, 16384, 4096, 4096};
  localparam int unsigned CW [NC] = '{512, 512, 512, 512, 8, 512, 512, 512, 512, 512, 512, 512, 8, 512, 512, 72,
                                       64, 64, 64, 64, 64};
  localparam int unsigned CL [NC] = '{1, 2, 3, 2, 3, 3, 2, 2, 2, 1, 4, 1, 4, 2, 1, 2,
                                       1, 2, 3, 1, 3};

  for (genvar c = 0; c < NC; c++) begin : g_c
    localparam int unsigned D  = CD[c];
    localparam int unsigned W  = CW[c];
    localparam int unsigned AW = (D > 1) ? $clog2(D) : 1;
    logic [W-1:0] q;

    tsr_sram_1r1w #(.DEPTH(D), .WIDTH(W), .RD_LAT(CL[c])) u_ram (
      .clk,
      .we, .waddr(AW'(32'(waddr) % D)), .wdata(wdata[W-1:0]), .wbe(wbe[W/8-1:0]),
      .re, .raddr(AW'(32'(raddr) % D)), .rdata(q)
    );

    if (W < 512) begin : g_pad
      assign rd[c] = {{(512 - W){1'b0}}, q};
    end else begin : g_full
      assign rd[c] = q;
    end
  end

endmodule

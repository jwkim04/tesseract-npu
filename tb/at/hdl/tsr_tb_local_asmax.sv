// Drive the real ASMAX scheduler, single-cycle score RAM and four VU lanes.
module tsr_tb_local_asmax
  import tsr_att_pkg::*;
  import tsr_vu_pkg::*;
(
  input logic clk, rst_n,
  input tsr_local_asmax_cmd_t cmd,
  output logic done,
  input logic load_valid, load_owner,
  input logic [TSR_LOCAL_SCORE_AW-1:0] load_address,
  input logic [511:0] load_data,
  output logic [1:0] wr,
  output logic [1:0][TSR_LOCAL_SCORE_AW-1:0] wr_address,
  output logic [1:0][511:0] wr_data
);
  tsr_local_score_req_t [1:0] mem_req;
  logic [1:0][511:0] data;
  tsr_vu_local_req_t req;
  tsr_vu_local_rsp_t rsp;

  tsr_local_asmax u_asmax (
    .clk, .rst_n, .cmd, .done, .mem_req, .mem_data(data), .lane_req(req), .lane_rsp(rsp)
  );
  tsr_vu_lane_bank #(
    .FRONT_LANES(8), .EXP_LANES(8), .BACK_LANES(4), .PIPE(2), .HAS_LOCAL(1'b1)
  ) u_bank (
    .clk, .rst_n, .l_v('0), .l_uop(U_NOP), .l_a('0), .l_b('0), .l_c1('0), .l_c2('0),
    .l_coeff('0), .angle('0), .l_neg('0), .l_lv('0), .l_fresh(1'b0), .l_pa('0), .l_ctx(1'b0),
    .back_valid(1'b0), .dr_addr('0), .dr_sel(1'b0), .dr_empty(1'b0), .dr_mx(1'b0),
    .dr_data(), .dr_l1(), .lane_y(), .local_req(req), .local_rsp(rsp)
  );

  for (genvar owner = 0; owner < 2; owner++) begin : g_owner
    wire load = load_valid && load_owner == 1'(owner);
    tsr_sram_1r1w #(.DEPTH(TSR_LOCAL_SCORE_LINES), .WIDTH(512)) u_ram (
      .clk, .we(mem_req[owner].wr || load), .waddr(load ? load_address : mem_req[owner].wa),
      .wdata(load ? load_data : mem_req[owner].data), .wbe('1), .re(mem_req[owner].rd),
      .raddr(mem_req[owner].ra), .rdata(data[owner])
    );
    assign wr[owner] = mem_req[owner].wr;
    assign wr_address[owner] = mem_req[owner].wa;
    assign wr_data[owner] = mem_req[owner].data;
  end
endmodule

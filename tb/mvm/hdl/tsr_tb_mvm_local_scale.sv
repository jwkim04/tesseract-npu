// Exercise the real scale queue and the four existing VU lanes together.
module tsr_tb_mvm_local_scale
  import tsr_vu_pkg::*;
(
  input logic clk, rst_n,
  input logic [31:0] scale, post_scale,
  input logic softcap,
  input logic s_valid,
  output logic s_ready,
  input logic [561:0] s_data,
  output logic m_valid,
  input logic m_ready,
  output logic [561:0] m_data
);
  tsr_vu_local_req_t req;
  tsr_vu_local_rsp_t rsp;
  tsr_mvm_local_scale u_queue (
    .clk, .rst_n, .scale, .post_scale, .softcap, .s_valid, .s_ready, .s_data, .m_valid, .m_ready, .m_data,
    .lane_req(req), .lane_rsp(rsp)
  );
  tsr_vu_lane_bank #(
    .FRONT_LANES(8), .EXP_LANES(8), .BACK_LANES(4), .PIPE(2), .HAS_LOCAL(1'b1)
  ) u_bank (
    .clk, .rst_n, .l_v('0), .l_uop(U_NOP), .l_a('0), .l_b('0), .l_c1('0), .l_c2('0),
    .l_coeff('0), .angle('0), .l_neg('0), .l_lv('0), .l_fresh(1'b0), .l_pa('0), .l_ctx(1'b0),
    .back_valid(1'b0), .dr_addr('0), .dr_sel(1'b0), .dr_empty(1'b0), .dr_mx(1'b0),
    .dr_data(), .dr_l1(), .lane_y(), .local_req(req), .local_rsp(rsp)
  );
endmodule

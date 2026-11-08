// tsr_vu_xc.sv — tsr_chk's external-class port: the VU class (tsr_vu_chk, inside) and the AT class
// (tsr_chk_at, outside through the at_* ports), each request routed to the class of its opcode
// (tsr_xc_pkg::tsr_xc_class) and the answering class's response returned.
//
// Owner: VU (W4-B task "vu"; CC §8.2 "opcodes from VU / AT plug in as classes"; contract: tsr_xc_pkg).
// Wiring (tsr_core_dp): xc_* <-> tsr_chk's xc_* port (via tsr_issue); at_* <-> tsr_chk_at's class side
// (req_valid / req_insn in, rsp_valid / rsp_code / rsp_info / rsp_ext / rsp_cmd out). With HAS_AT = 0
// the AT ports are unused (tie at_rsp_valid to 0): no AT opcode reaches the port then (tsr_chk refuses
// opcodes whose CAPS bits are 0). Combinational between the two sides (no cycle added): the classes'
// outputs are registers and tsr_chk registers what it takes.

module tsr_vu_xc
  import tsr_pkg::*;
  import tsr_xc_pkg::*;
#(
  parameter bit          LOCAL_INTERMEDIATES = 1'b0,
  parameter bit          HAS_GDN  = 1'b0,
  parameter int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  parameter int unsigned K_MAX     = TSR_DEF_K_MAX,
  parameter int unsigned NUM_CH    = TSR_DEF_NUM_CH
) (
  input  logic                         clk,
  input  logic                         rst_n,
  // tsr_chk side
  input  logic                         xc_req_valid,
  input  tsr_insn_t                    xc_req_insn,
  output logic                         xc_rsp_valid,
  output logic [3:0]                   xc_rsp_code,
  output logic [31:0]                  xc_rsp_info,
  output tsr_ext_t [TSR_EXT_N-1:0]     xc_rsp_ext,
  output logic [TSR_XC_CMD_W-1:0]      xc_rsp_cmd,
  // AT class (tsr_chk_at)
  output logic                         at_req_valid,
  output tsr_insn_t                    at_req_insn,
  input  logic                         at_rsp_valid,
  input  logic [3:0]                   at_rsp_code,
  input  logic [31:0]                  at_rsp_info,
  input  tsr_ext_t [TSR_EXT_N-1:0]     at_rsp_ext,
  input  logic [TSR_XC_CMD_W-1:0]      at_rsp_cmd
);
  wire [1:0] cls = tsr_xc_class(xc_req_insn[0][7:0]);

  logic                     vu_rsp_valid;
  logic [3:0]               vu_rsp_code;
  logic [31:0]              vu_rsp_info;
  tsr_ext_t [TSR_EXT_N-1:0] vu_rsp_ext;
  logic [TSR_XC_CMD_W-1:0]  vu_rsp_cmd;
  tsr_vu_chk #(.LOCAL_INTERMEDIATES(LOCAL_INTERMEDIATES), .HAS_GDN(HAS_GDN), .SPM_BYTES(SPM_BYTES), .K_MAX(K_MAX), .NUM_CH(NUM_CH)) u_vu (
    .clk, .rst_n, .req_valid(xc_req_valid && cls == TSR_XC_VU), .req_insn(xc_req_insn),
    .rsp_valid(vu_rsp_valid), .rsp_code(vu_rsp_code), .rsp_info(vu_rsp_info), .rsp_ext(vu_rsp_ext),
    .rsp_cmd(vu_rsp_cmd)
  );

  assign at_req_valid = xc_req_valid && (cls == TSR_XC_AT);
  assign at_req_insn  = xc_req_insn;

  assign xc_rsp_valid = vu_rsp_valid || at_rsp_valid;
  assign xc_rsp_code  = vu_rsp_valid ? vu_rsp_code : at_rsp_code;
  assign xc_rsp_info  = vu_rsp_valid ? vu_rsp_info : at_rsp_info;
  assign xc_rsp_ext   = vu_rsp_valid ? vu_rsp_ext  : at_rsp_ext;
  assign xc_rsp_cmd   = vu_rsp_valid ? vu_rsp_cmd  : at_rsp_cmd;

`ifndef SYNTHESIS
  logic chk_on_q;
  initial chk_on_q = 1'b0;
  always @(posedge clk) if (!rst_n) chk_on_q <= 1'b1;
  always_ff @(posedge clk) begin
    if (rst_n && chk_on_q) begin
      assert (!(vu_rsp_valid && at_rsp_valid)) else $error("tsr_vu_xc: both classes answered");
      assert (!(xc_req_valid && cls == TSR_XC_NONE))
        else $error("tsr_vu_xc: external-class request for opcode %02x of no class", xc_req_insn[0][7:0]);
    end
  end
`endif
endmodule

// tsr_xc_pkg.sv — shared types and rules of the external checker classes on tsr_chk's xc_* port
// (docs/design/04-control-core.md "CC" §8.2, §16: "opcodes from VU / AT plug in as classes").
//
// Owner: VU (W4-B task "vu"); the AT branch is the AT builder's. Both classes follow this contract,
// and the integrator (tsr_core_dp) joins them on tsr_chk's single xc port.
//
// Split of the external opcodes (EX §7.2; tsr_tables_pkg TSR_OP_*), by the unit that runs them:
//   VU class (tsr_vu_chk)   unit 3: DQROW 0x12, VOP 0x30, RED 0x31, ROPE 0x32, TOPK 0x33, GDN 0x39, SOP 0x40, and AT's
//                           vector ops ASMAX 0x34, AMASK 0x35, AMERGE 0x36, ANORM 0x37 (AT §5.8's checks,
//                           VU §5.11: the VU executes them)
//   AT class (tsr_chk_at)   KVST 0x14 (unit 0), ATTQK 0x21, ATTPV 0x22 (unit 2)
// tsr_vu_xc routes each request to its class and returns the answering class's response.
//   (VMASK 0x38 and the other reserved VU / AT codes are refused by tsr_chk itself while their CAPS
//   bits are 0, so no class sees them.)
//
// Port contract of one class (every signal named from the class's side):
//   req_valid  in   tsr_chk.xc_req_valid: raised one cycle after C1 decoded an external opcode, held
//                   (with req_insn) until the cycle in which some class answers.
//   req_insn   in   tsr_chk.xc_req_insn: the eight words w0..w7.
//   rsp_valid  out  one-cycle strobe, raised by exactly one class: the one whose opcode set
//                   (tsr_xc_class) holds req_insn[0][7:0]. A class answers each request once: it
//                   starts on req_valid && !busy and stays busy until it has seen req_valid low
//                   again (req_valid is still high in the cycle of rsp_valid).
//   rsp_code   out  the EX §8.1 code of the class's checks, in EX's order without BAD_DEP (the
//                   dispatcher merges it): ILLEGAL_OP -> BAD_FMT -> K_RANGE -> T_RANGE -> OPERAND ->
//                   SPM_RANGE -> DRAM_RANGE.
//   rsp_info   out  ERR_INFO of rsp_code (0 for code 0 and ILLEGAL_OP).
//   rsp_ext    out  tsr_pkg::tsr_ext_t [3:0]: the SPM / DRAM operand extents of a passing op in
//                   operand field order, empty operands skipped, compacted into slots 0..3 (the
//                   simulator's tsr_ext_add order; its s-register entries have no RTL slot).
//   rsp_cmd    out  TSR_XC_CMD_W bits: the payload tsr_issue stores in the unit's queue. For every
//                   unit-3 opcode it is the eight instruction words unchanged (tsr_insn_t, w0 in the
//                   low word): the VU decodes its commands itself (VU §7.8). The AT class chooses the
//                   payload of its unit-0 / unit-2 ops (KVST, ATTQK, ATTPV) within TSR_XC_CMD_W bits.
//   A class sees req_valid only for its own opcodes (tsr_vu_xc gates it); tsr_vu_chk also ignores
//   other opcodes itself and holds its rsp_* outputs at 0 while rsp_valid is 0.

package tsr_xc_pkg;

  import tsr_pkg::*;

  /* verilator lint_off UNUSEDPARAM */
  localparam int unsigned TSR_XC_CMD_W = 256;          // = tsr_chk / tsr_issue XC_CMD_W

  // Class of an external opcode.
  localparam logic [1:0] TSR_XC_NONE = 2'd0;
  localparam logic [1:0] TSR_XC_VU   = 2'd1;
  localparam logic [1:0] TSR_XC_AT   = 2'd2;
  /* verilator lint_on UNUSEDPARAM */

  function automatic logic [1:0] tsr_xc_class(input logic [7:0] op);
    case (op)
      8'h12, 8'h30, 8'h31, 8'h32, 8'h33, 8'h34, 8'h35, 8'h36, 8'h37, 8'h39, 8'h40: return TSR_XC_VU;
      8'h14, 8'h21, 8'h22: return TSR_XC_AT;
      default: return TSR_XC_NONE;
    endcase
  endfunction

  // One class's answer (all fields 0 unless valid).
  typedef struct packed {
    logic                       valid;
    logic [3:0]                 code;
    logic [31:0]                info;
    tsr_ext_t [TSR_EXT_N-1:0]   ext;
    logic [TSR_XC_CMD_W-1:0]    cmd;
  } tsr_xc_rsp_t;

  // Join of two classes' answers (each is all-zero unless it answers).
  function automatic tsr_xc_rsp_t tsr_xc_or(input tsr_xc_rsp_t a, input tsr_xc_rsp_t b);
    return tsr_xc_rsp_t'(a | b);
  endfunction

endpackage

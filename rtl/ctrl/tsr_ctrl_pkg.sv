// tsr_ctrl_pkg.sv — types and constants of the RISC-V control subsystem (CC §2-§5, §7, §11).
//
// Owner: ctrl agent (W4 item 3). The register offsets, the address map, the issue-FIFO layout and
// the opcode -> unit map are generated (tsr_tables_pkg, EX-30, CC §13.4); this package only adds
// the crossing formats of tsr_ctrl_cdc and the bus-internal types, which are this subsystem's own.
//
// Crossing formats (the core-domain side of tsr_ctrl / tsr_ctrl_cdc presents them unpacked; the
// widths are the exact ones of the implementation, CC §11.2 gives the planning widths):
//   error record of unit u (core -> clk_rv, 128 b, CC §5.1, §9.5):
//     [27:0] ERR_SEQ  [31:28] ERR_CODE (EX §8.1 codes fit 4 bits; the window shows it zero-extended)
//     [63:32] ERR_INFO  [95:64] ERR_W0  [127:96] ERR_PIDX
//   bridge / core-window request (clk_rv -> core, 64 b): XK_SPM_RD / XK_SPM_WR / XK_CORE_RD, the SPM
//     byte offset or the CSR offset, byte enables, write data; response 513 b {err, line[511:0]}
//     (a core-window read answers in line[31:0]).
//   LQ / LCQ request (clk_rv -> core, 301 b): {lcq_w[7:0][31:0], wdata, we, off[11:0]}; response
//     33 b {fault, rdata}.
//   LOAD (core -> clk_rv, 65 b): {we, addr, wdata}; response 33 b {err, rdata}.
//   DMI (core -> clk_rv, 41 b): {addr[6:0], op[1:0], data[31:0]} (dm::dmi_req_t); response 34 b
//     {data[31:0], resp[1:0]} (dm::dmi_resp_t).
//   RT mirror (clk_rv -> core, 64 b): {RT_INFO, RT_STATE}.

package tsr_ctrl_pkg;

  // A catalogue: not every module uses every constant (docs/rtl-core.md §1 package waiver).
  /* verilator lint_off UNUSEDPARAM */

  // Error record fields (128 b).
  localparam int unsigned ERR_REC_W = 128;
  typedef struct packed {
    logic [31:0] pidx;
    logic [31:0] w0;
    logic [31:0] info;
    logic [3:0]  code;
    logic [27:0] seq;
  } err_rec_t;

  // Bridge / core-window crossing (CC §10.5, §5.3).
  typedef enum logic [1:0] {
    XK_SPM_RD  = 2'd0,   // read the whole SPM line holding addr (exact window or line-buffer fill)
    XK_SPM_WR  = 2'd1,   // write wdata into the line at addr with byte enables be (word lane addr[5:2])
    XK_CORE_RD = 2'd2    // core-window read of host CSR offset addr[11:0] (tsr_csr requester 2)
  } xkind_e;

  typedef struct packed {
    xkind_e      kind;
    logic [3:0]  be;
    logic [25:0] addr;    // SPM byte offset (SPM_BYTES <= 64 MiB) or CSR offset
    logic [31:0] wdata;
  } xreq_t;               // 64 b
  localparam int unsigned XREQ_W = $bits(xreq_t);
  localparam int unsigned XRSP_W = 513;   // {err, line[511:0]}

  // LQ / LCQ crossing (EX §4.1, §4.5, CC-30).
  typedef struct packed {
    logic [7:0][31:0] lcq_w;   // LCQ_W0..7 (LCQ_POST), lcq_w[i] = word i
    logic [31:0]      wdata;
    logic             we;
    logic [11:0]      off;     // control-window offset 0x200-0x2FC
  } lqreq_t;                   // 301 b
  localparam int unsigned LQREQ_W = $bits(lqreq_t);
  localparam int unsigned LQRSP_W = 33;   // {fault, rdata}

  // LOAD_* (EX §3.3, CC §6.4).
  typedef struct packed {
    logic        we;
    logic [31:0] addr;    // RISC-V byte address in ITCM or DTCM, word aligned
    logic [31:0] wdata;
  } ldreq_t;              // 65 b
  localparam int unsigned LDREQ_W = $bits(ldreq_t);
  localparam int unsigned LDRSP_W = 33;   // {err, rdata}

  localparam int unsigned DMIREQ_W = 41;
  localparam int unsigned DMIRSP_W = 34;

  // RISC-V bus regions (CC §4.1).
  typedef enum logic [2:0] {
    RG_NONE  = 3'd0,
    RG_ITCM  = 3'd1,
    RG_DTCM  = 3'd2,
    RG_DM    = 3'd3,
    RG_CW    = 3'd4,
    RG_CORE  = 3'd5,
    RG_SPM   = 3'd6,
    RG_SPMLB = 3'd7
  } region_e;

  // Data-side masters (CC §4.3): Ibex's data port, the DM's SBA master, the host load port
  // (EX's LOAD_* while the hart is held in reset, CC §4.2).
  typedef enum logic [1:0] {
    MS_IBEX = 2'd0,
    MS_SBA  = 2'd1,
    MS_LD   = 2'd2
  } master_e;

  /* verilator lint_on UNUSEDPARAM */

endpackage

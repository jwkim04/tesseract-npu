// tsr_vu_rd.sv — the VU's SPM read side: per-bank read requests on both bank ports of SPM client 4, the
// response FIFOs per (bank, operand), and the slot FIFO towards the injector.
//
// Owner: VU (docs/design/02-vector-unit.md "VU" §7.7 SPM access; CC §10.2 client contract).
// A slot is taken from the generator when every SPM line it reads has a reserved place: its A line in the
// response FIFO (bank(A), A), its B line in (bank(B), B), and the slot a place in the slot FIFO (rule R4:
// a read is issued only when its response can be stored; the SPM answers without backpressure). The
// line requests go into a queue per bank in slot order (A before B), so each bank port works on its own
// and two operands stream at one line each per cycle whatever the parity of their lines (consecutive
// lines of an operand alternate banks; VU-10: no placement rule). Each bank queue's head is the bank
// port's request (R2: it changes only when accepted). Responses arrive exactly RD_LAT cycles after the
// accepting edge (CC §10.2) and are steered into the FIFO of their operand by a tag pipeline per bank;
// per (bank, operand) they stay in slot order, which is the injector's consumption order.
// The injector pops the slot FIFO and, per slot, the heads of the FIFOs its lines are in (f_pop).

module tsr_vu_rd
  import tsr_pkg::*;
  import tsr_vu_pkg::*;
#(
  parameter  int unsigned SPM_BYTES = TSR_DEF_SPM_BYTES,
  parameter  int unsigned RD_LAT    = TSR_DEF_SPM_RD_LAT,
  parameter  int unsigned FD        = 16,     // response FIFO depth per (bank, operand)
  parameter  int unsigned SD        = 16,     // slot FIFO depth
  localparam int unsigned LINES     = SPM_BYTES / TSR_LINE_BYTES,
  localparam int unsigned AW        = (LINES > 1) ? $clog2(LINES) : 1
) (
  input  logic                           clk,
  input  logic                           rst_n,
  // slots from tsr_vu_gen
  input  logic                           s_valid,
  output logic                           s_ready,
  input  tsr_vu_slot_t                   s_slot,
  // SPM client 4, read side
  output logic [1:0]                     rd_req_valid,
  input  logic [1:0]                     rd_req_ready,
  output logic [1:0][AW-1:0]             rd_req_line,
  input  logic [1:0]                     rd_rsp_valid,
  input  logic [1:0][TSR_LINE_BITS-1:0]  rd_rsp_data,
  // to tsr_vu_inj
  output logic                           q_valid,
  output tsr_vu_slot_t                   q_slot,
  input  logic                           q_pop,
  output logic [1:0][1:0]                f_valid,     // [bank][0 A, 1 B]
  output logic [1:0][1:0][TSR_LINE_BITS-1:0] f_data,
  input  logic [1:0][1:0]                f_pop,
  output logic                           rd_wait       // a read request is raised and not granted
);

  localparam int unsigned QD  = 4;                    // per-bank request queue
  localparam int unsigned FPW = $clog2(FD);
  localparam int unsigned SPW = $clog2(SD);

  if (FD < 4 || (FD & (FD - 1)) != 0 || SD < 4 || (SD & (SD - 1)) != 0 || RD_LAT < 1) begin : g_chk
    $fatal(1, "tsr_vu_rd: FD and SD powers of two >= 4, RD_LAT >= 1");
  end

  // -------------------------------------------------------------------------------------------
  // Admission of a slot
  // -------------------------------------------------------------------------------------------
  wire        na   = (s_slot.asrc == SRC_SPM);
  wire        nb   = (s_slot.bsrc == SRC_SPM);
  wire        pa   = s_slot.aline[0];
  wire        pb   = s_slot.bline[0];

  logic [1:0][1:0]   res_full;                        // (bank, operand) has no free place
  logic [1:0][2:0]   qcnt;                            // bank queue occupancy
  logic [SPW:0]      scnt;
  logic [1:0][2:0]   need_q;                          // queue entries the slot takes per bank
  always_comb begin
    need_q[0] = 3'(na && !pa) + 3'(nb && !pb);
    need_q[1] = 3'(na && pa) + 3'(nb && pb);
  end
  assign s_ready = (32'(scnt) < SD) &&
                   (32'(qcnt[0]) + 32'(need_q[0]) <= QD) && (32'(qcnt[1]) + 32'(need_q[1]) <= QD) &&
                   !(na && res_full[pa][0]) && !(nb && res_full[pb][1]);
  wire take = s_valid && s_ready;

  // -------------------------------------------------------------------------------------------
  // Per-bank request queues {line, operand}; the head is the bank port's request
  // -------------------------------------------------------------------------------------------
  logic [1:0][QD-1:0][AW:0] qm;
  logic [1:0][1:0]          qrp, qwp;
  logic [1:0]               acc;
  for (genvar b = 0; b < 2; b++) begin : g_bq
    wire wa = take && na && (pa == 1'(b));
    wire wb = take && nb && (pb == 1'(b));
    assign acc[b] = rd_req_valid[b] && rd_req_ready[b];
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        qrp[b]  <= '0;
        qwp[b]  <= '0;
        qcnt[b] <= '0;
      end else begin
        qrp[b]  <= qrp[b] + 2'(acc[b]);
        qwp[b]  <= qwp[b] + 2'(wa) + 2'(wb);
        qcnt[b] <= qcnt[b] + 3'(wa) + 3'(wb) - 3'(acc[b]);
      end
    end
    always_ff @(posedge clk) begin
      if (wa) qm[b][qwp[b]] <= {1'b0, AW'(s_slot.aline)};
      if (wb) qm[b][qwp[b] + 2'(wa)] <= {1'b1, AW'(s_slot.bline)};
    end
    assign rd_req_valid[b] = (qcnt[b] != 3'd0);
    assign rd_req_line[b]  = qm[b][qrp[b]][AW-1:0];

    // tag pipeline: the operand of the request accepted RD_LAT cycles ago
    logic tv [RD_LAT];
    logic tf [RD_LAT];
    always_ff @(posedge clk) begin
      if (!rst_n) begin
        for (int i = 0; i < RD_LAT; i++) tv[i] <= 1'b0;
      end else begin
        tv[0] <= acc[b];
        for (int i = 1; i < RD_LAT; i++) tv[i] <= tv[i-1];
      end
      tf[0] <= qm[b][qrp[b]][AW];
      for (int i = 1; i < RD_LAT; i++) tf[i] <= tf[i-1];
    end
    wire rv = tv[RD_LAT-1];
    wire rf = tf[RD_LAT-1];

    // response FIFOs of this bank, one per operand
    for (genvar f = 0; f < 2; f++) begin : g_f
      logic [TSR_LINE_BITS-1:0] fm [FD];
      logic [FPW-1:0] frp, fwp;
      logic [FPW:0]   fcnt, fres;                     // stored; reserved (queued + in flight + stored)
      wire  fw = rv && (rf == 1'(f));
      wire  fr = f_pop[b][f];
      wire  ft = take && ((f == 0) ? (na && pa == 1'(b)) : (nb && pb == 1'(b)));
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          frp  <= '0;
          fwp  <= '0;
          fcnt <= '0;
          fres <= '0;
        end else begin
          frp  <= frp + FPW'(fr);
          fwp  <= fwp + FPW'(fw);
          fcnt <= fcnt + (FPW+1)'(fw) - (FPW+1)'(fr);
          fres <= fres + (FPW+1)'(ft) - (FPW+1)'(fr);
        end
      end
      always_ff @(posedge clk) if (fw) fm[fwp] <= rd_rsp_data[b];
      assign f_valid[b][f] = (fcnt != '0);
      assign f_data[b][f]  = fm[frp];
      assign res_full[b][f] = (32'(fres) >= FD);
`ifndef SYNTHESIS
      always_ff @(posedge clk) begin
        if (rst_n) begin
          assert (!(fr && fcnt == '0)) else $error("tsr_vu_rd: pop of an empty response FIFO (%0d, %0d)", b, f);
        end
      end
`endif
    end
`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
      if (rst_n) begin
        assert (rd_rsp_valid[b] == rv)
          else $error("tsr_vu_rd: bank %0d response valid %0b, expected %0b (RD_LAT %0d)", b, rd_rsp_valid[b], rv, RD_LAT);
      end
    end
`endif
  end

  // -------------------------------------------------------------------------------------------
  // Slot FIFO: a registered head (the injector's decision starts from flip-flops) and SD entries
  // behind it in distributed RAM; scnt counts both
  // -------------------------------------------------------------------------------------------
  tsr_vu_slot_t     sm [SD];
  logic [SPW-1:0]   srp, swp;
  logic [SPW:0]     mcnt;                             // entries in sm
  logic             hv_q;
  tsr_vu_slot_t     hd_q;
  wire              hfree = !hv_q || q_pop;           // the head register takes a slot this cycle
  wire              byp   = take && hfree && (mcnt == '0);
  wire              mw    = take && !byp;
  wire              mr    = hfree && (mcnt != '0);
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      srp  <= '0;
      swp  <= '0;
      mcnt <= '0;
      scnt <= '0;
      hv_q <= 1'b0;
    end else begin
      srp  <= srp + SPW'(mr);
      swp  <= swp + SPW'(mw);
      mcnt <= mcnt + (SPW+1)'(mw) - (SPW+1)'(mr);
      scnt <= scnt + (SPW+1)'(take) - (SPW+1)'(q_pop);
      hv_q <= mr || byp || (hv_q && !q_pop);
    end
    if (mr)       hd_q <= sm[srp];
    else if (byp) hd_q <= s_slot;
  end
  always_ff @(posedge clk) if (mw) sm[swp] <= s_slot;
  assign q_valid = hv_q;
  assign q_slot  = hd_q;

  assign rd_wait = |(rd_req_valid & ~rd_req_ready);

endmodule

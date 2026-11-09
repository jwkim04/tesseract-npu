// tsr_att_pkg.sv — area AT (attention and the KV cache, docs/design/03-attention-kv.md "AT"):
// the dispatch payloads of KVST / ATTQK / ATTPV, the MVM unit's internal attention command, and the
// small combinational helpers the AT blocks share (bit-exact numerics of AT §6, the constant
// divider by NUM_CH, the MAC lane's PV latency).
//
// Spec: tsr_chk_at, the KVST engine of tsr_dma, the attention modes of tsr_mvm*.
// The payloads travel in the issue path's 256-bit external-class payload (tsr_chk xc_rsp_cmd ->
// tsr_issue <u>_xcmd, CC §8.2); their layout is AT's: tsr_chk_at builds them, the DMA (KVST) and
// the MVM unit (ATTQK, ATTPV) consume them. Every field is checked and resolved at dispatch (AT
// §5.8; strides of 0 replaced by their defaults), so the units assume legal values.
// Packed: the first member is the MSB (tb/at/at_isa.py keeps the same order and widths).

package tsr_att_pkg;
  import tsr_pkg::*;

  /* verilator lint_off UNUSEDPARAM */

  // -------------------------------------------------------------------------------------------
  // Dispatch payloads (tsr_chk_at -> tsr_issue -> unit; low bits of the 256-bit payload)
  // -------------------------------------------------------------------------------------------

  localparam int unsigned TSR_LOCAL_SCORE_LINES = 4480;
  localparam int unsigned TSR_LOCAL_SCORE_AW = $clog2(TSR_LOCAL_SCORE_LINES);
  localparam int unsigned TSR_LOCAL_MAX_ROWS = 224;

  typedef struct packed {
    logic [1:0] slot;
    logic rd;
    logic wr;
    logic [TSR_LOCAL_SCORE_AW-1:0] ra;
    logic [TSR_LOCAL_SCORE_AW-1:0] wa;
    logic [TSR_LINE_BITS-1:0] data;
  } tsr_local_score_req_t;

  typedef struct packed {
    logic        start;
    logic [1:0]  slot;
    logic [1:0]  first_owner;
    logic [7:0]  groups;
    logic        max_valid;
    logic [7:0]  row;        // START: row count; MAX: row index
    logic [31:0] value;      // START: scale; MAX: row maximum
  } tsr_local_asmax_cmd_t;

  // KVST 0x14 (AT §5.1) -> tsr_dma (dma_cmd_xc = 1).
  typedef struct packed {
    logic        kv;        // 0: K plane (token-major, AT §4.3), 1: V plane (slice-major, AT §4.4)
    logic [6:0]  n_rows;    // 1..64: cells c0 .. c0 + n_rows - 1
    logic [31:0] src;       // w1: SPM line of QA(A32, E) vector 0
    logic [31:0] stride;    // w5 resolved: lines between the source vectors (0 -> QA_LINES(A32, E))
    logic [31:0] base;      // w2: plane base beat B (the same in every channel)
    logic [8:0]  e64;       // w3[8:0]: E / 64, 1..511
    logic [15:0] lg;        // w3[31:16]: local groups per channel, >= 1
    logic [31:0] c0;        // w4: first cell; c0 + n_rows <= 16 C LG
  } tsr_kvst_cmd_t;         // 161 bits

  // ATTQK 0x21 / ATTPV 0x22 (AT §5.2, §5.3) -> tsr_mvm (mvm_cmd_xc = 1).
  typedef struct packed {
    logic [1:0]  local_slot;
    logic        pv;        // 0: ATTQK, 1: ATTPV
    logic [2:0]  flags;     // ATTQK {MAXOUT, SLIDE, CAUSAL}; ATTPV {0, 0, LSUM} (w0[14:12])
    logic [8:0]  e64;       // w4[8:0]: E / 64 (E % D == 0)
    logic [1:0]  dcode;     // w4[10:9]: D = 64 << dcode
    logic [5:0]  g_m1;      // x16[5:0]: G - 1
    logic [5:0]  hg0;       // x16[11:6]: first head of the pass (hg0 + GS <= G)
    logic [2:0]  gs_m1;     // w4[16:14]: GS - 1
    logic [2:0]  tq_m1;     // w4[13:11]: T_q - 1 (T_q GS <= T_MAX)
    logic [31:0] dst;       // w1: ATTQK score rows, ATTPV channel partials
    logic [31:0] x;         // w2: ATTQK the T_q QA(A32, G E) vectors, ATTPV the p rows
    logic [31:0] base;      // w3: plane base beat B
    logic [14:0] rs;        // w4[31:17] resolved: rstride / pstride (0 -> n_groups), >= n_groups
    logic [15:0] u0;        // w5[15:0]: first group
    logic [11:0] n;         // w5[31:16]: n_groups, 1..4095
    logic [15:0] aux0;      // w6[15:0]: ATTQK c_lo, ATTPV LG
    logic [15:0] aux1;      // w6[31:16]: ATTQK c_hi, ATTPV 0
    logic [31:0] w7;        // ATTQK mdst (MAXOUT), ATTPV cstride resolved (0 -> R D / 16)
  } tsr_att_cmd_t;          // 238 bits

  localparam int unsigned TSR_ATT_FL_CAUSAL = 0;   // flags bit (ATTQK)
  localparam int unsigned TSR_ATT_FL_SLIDE  = 1;
  localparam int unsigned TSR_ATT_FL_MAXOUT = 2;
  localparam int unsigned TSR_ATT_FL_LSUM   = 0;   // flags bit (ATTPV)

  // -------------------------------------------------------------------------------------------
  // MVM unit internal: tsr_mvm -> every cluster (on the broadcast pipe, overlaid on the line
  // payload like tsr_mvm_ccmd_t, before the op's lines). Everything a cluster needs to derive its
  // own groups (AT §4.2: group u in channel u mod C at local index u / C).
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    tsr_att_cmd_t c;        // the dispatch payload
    logic [8:0]   nseg;     // E / D (head segments per cell row)
    logic [14:0]  n_head;   // G E / D
    logic [17:0]  r;        // R = T_q n_head
    logic [3:0]   tact;     // T_q GS: active slots
    logic [7:0]   prows;    // pass rows (E / D) GS T_q <= 128
    logic [15:0]  q0;       // u0 / C
    logic [3:0]   m0;       // u0 mod C
    logic [16:0]  qn;       // (u0 + n) / C
    logic [3:0]   mn;       // (u0 + n) mod C
    logic [13:0]  rl;       // ceil(R / 16) (ATTPV LSUM vector lines per channel)
  } tsr_att_ccmd_t;         // 238 + 109 = 347 bits

  // -------------------------------------------------------------------------------------------
  // MVM unit internal: the cluster -> back end configuration of an attention op (stable while its
  // beats flow, like cfg_fmt; docs/rtl-mvm.md §3.2), and the PV unit's activation-buffer read.
  // -------------------------------------------------------------------------------------------
  typedef struct packed {
    logic        att;       // ATTQK: segment trees, per-token activation addresses (AT §8.1)
    logic        pv;        // ATTPV: MAC lanes in PV mode, activation reads from the PV unit (AT §8.2)
    logic [1:0]  dcode;     // D = 64 << dcode (segment = D/32 blocks = D/64 beats)
    logic [9:0]  nb;        // E / 32: blocks per cell row
    logic [10:0] hstep;     // G D / 32: block distance of the next head segment's head
  } tsr_att_bcfg_t;         // 25 bits

  typedef struct packed {
    logic        v;         // read in this cycle (all tokens, one address)
    logic        dbs;       // 0: QS word `word` (both banks), 1: DBS line `line`
    logic [9:0]  word;
    logic [10:0] line;
  } tsr_att_rd_t;           // 23 bits

  // -------------------------------------------------------------------------------------------
  // Opcode helpers
  // -------------------------------------------------------------------------------------------
  function automatic logic tsr_att_is_op(logic [7:0] op);   // the AT ops of the DMA / MVM units
    return (op == tsr_tables_pkg::TSR_OP_KVST) || (op == tsr_tables_pkg::TSR_OP_ATTQK) ||
           (op == tsr_tables_pkg::TSR_OP_ATTPV);
  endfunction

  // QA_LINES(A32, K) for K a multiple of 64: K / 64 QS lines + ceil(K / 512) D lines.
  function automatic logic [31:0] tsr_att_qa32_lines(logic [31:0] k);
    return (k >> 6) + ((k + 32'd511) >> 9);
  endfunction

  // Exact x / C for 0 <= x < 2^21 and 1 <= C <= 16 by multiplication with M = ceil(2^25 / C):
  // with e = M C - 2^25 < C, x e < 2^25, so floor(x M / 2^25) = floor(x / C). Callers register the
  // operand and the result (one 21 x 26 product between flops: a DSP48E2 pair or a short LUT
  // multiplier; for a power-of-two C it is a shift).
  function automatic logic [20:0] tsr_att_divc(logic [20:0] x, int unsigned C);
    logic [25:0] m;                                  // C = 1: 2^25
    logic [46:0] p;
    m = 26'((64'd33554432 + 64'(C) - 64'd1) / 64'(C));
    p = 47'(x) * 47'(m);
    return 21'(p >> 25);
  endfunction

  // -------------------------------------------------------------------------------------------
  // Numerics (AT §6; tsr_ref_at.c): bit for bit
  // -------------------------------------------------------------------------------------------
  // DAZ of an f32 (subnormal -> signed zero), as every fp input of the spec.
  function automatic logic [31:0] tsr_att_daz(logic [31:0] u);
    return (u[30:23] == 8'd0) ? {u[31], 31'd0} : u;
  endfunction

  // MAXT key (VU §6.8, tsr_att_maxt): the total order -0 < +0, NaN largest. key = {nan, k} with
  // k = sign ? ~u : u | 2^31 of the DAZ'd value; a larger key is the larger value. The value
  // that goes with a key: the DAZ'd input, or the canonical NaN.
  function automatic logic [32:0] tsr_att_maxkey(logic [31:0] a);
    logic [31:0] u;
    u = tsr_att_daz(a);
    if (u[30:23] == 8'hFF && u[22:0] != 23'd0) return {1'b1, 32'hFFFF_FFFF};
    return {1'b0, u[31] ? ~u : (u | 32'h8000_0000)};
  endfunction
  function automatic logic [31:0] tsr_att_maxval(logic [31:0] a);
    logic [31:0] u;
    u = tsr_att_daz(a);
    return (u[30:23] == 8'hFF && u[22:0] != 23'd0) ? TSR_CANON_NAN : u;
  endfunction
  // key -> value (inverse of tsr_att_maxkey on its image)
  function automatic logic [31:0] tsr_att_keyval(logic [32:0] k);
    if (k[32]) return TSR_CANON_NAN;
    return k[31] ? (k[31:0] & 32'h7FFF_FFFF) : ~k[31:0];
  endfunction

  // bf_int (AT §6.4 step 5, tsr_att_bf_int): RNE_int(w 2^(23 - E_S)) for an f32 w with
  // eps(w) <= E_S, as a 25-bit two's complement value (|result| < 2^24, or 0 for a zero or
  // subnormal w). es is E_S + 127 (biased, 1..254). Non-finite w never reaches here (callers
  // mask the block, AT §6.4 step 3).
  function automatic logic [24:0] tsr_att_bf_int(logic [31:0] w, logic [7:0] es);
    logic [7:0]  e;
    logic [8:0]  sh;
    logic [23:0] m, v, rem, half;
    logic        up;
    e = w[30:23];
    if (e == 8'd0) return 25'd0;
    sh = 9'(es) - 9'(e);
    m  = {1'b1, w[22:0]};
    if (sh >= 9'd25) return 25'd0;                 // m < 2^24 <= half: rounds to 0
    v    = m >> sh;
    rem  = (sh == 9'd0) ? 24'd0 : (m & ((24'd1 << sh) - 24'd1));
    half = (sh == 9'd0) ? 24'd0 : (24'd1 << (sh - 9'd1));
    up   = (sh != 9'd0) && ((rem > half) || (rem == half && v[0]));
    v    = v + 24'(up);
    return w[31] ? -25'({1'b0, v}) : 25'({1'b0, v});
  endfunction

  // F(A, k) (AT §6.4 step 7, tsr_att_bf_f) for k = es - 127 - 23 (es biased E_S): A rounded to 24
  // significant bits (RNE), times 2^k; below 2^-126 -> signed zero (sign of A; A = 0 -> +0),
  // >= 2^128 -> signed infinity. AW-bit two's complement A.
  function automatic logic [31:0] tsr_att_bf_f(logic [39:0] a, logic [7:0] es);
    logic        s;
    logic [39:0] mag, nrm;
    int          msb;
    logic [23:0] m;
    logic        g, st, up;
    logic [24:0] mr;
    int          ex;
    if (a == 40'd0) return 32'd0;
    s   = a[39];
    mag = s ? (~a + 40'd1) : a;
    msb = 0;
    for (int i = 0; i < 40; i++) if (mag[i]) msb = i;
    nrm = mag << (39 - msb);                         // nrm[39] = 1
    m   = nrm[39:16];
    g   = nrm[15];
    st  = |nrm[14:0];
    up  = g & (st | m[0]);
    mr  = {1'b0, m} + 25'(up);
    ex  = msb + int'(es) - 23 + (mr[24] ? 1 : 0);    // biased exponent of the result
    if (ex < 1) return {s, 31'd0};
    if (ex > 254) return {s, 8'hFF, 23'd0};
    return {s, 8'(ex), mr[24] ? mr[23:1] : mr[22:0]};
  endfunction

  // -------------------------------------------------------------------------------------------
  // MAC lane PV mode (AT §8.2): cycles from the PV operands at tsr_mac_lane's inputs to the
  // accumulator holding their product (the register allocation of the Xilinx lane: M, P, a first
  // tree cut, AD, IN, ...; the generic lane copies it for PV). Needs LAT >= 3 (M and P).
  // -------------------------------------------------------------------------------------------
  function automatic int unsigned tsr_mac_pv_lat(int unsigned lat, int unsigned n);
    int unsigned lv, tcmx, nreg, ra0, tc1, ra1, rad, ra2, rin;
    lv   = (n > 1) ? $clog2(n) : 0;
    tcmx = (lv > 1) ? lv - 1 : 0;
    nreg = lat - 1;
    ra0  = (nreg > 2) ? nreg - 2 : 0;
    tc1  = (ra0 >= 1 && tcmx >= 1) ? 1 : 0;
    ra1  = ra0 - tc1;
    rad  = (ra1 >= 1) ? 1 : 0;
    ra2  = ra1 - rad;
    rin  = (ra2 >= 1) ? 1 : 0;
    return rin + rad + 2;                            // IN, AD, M, P
  endfunction

  /* verilator lint_on UNUSEDPARAM */

endpackage

"""Figure `cluster`: the MVM unit datapath of one cluster, DDR4 weight beats to the SPM write.

Content follows docs/rtl-mvm.md §2 (block diagram), §3 (internal interfaces), §4 (weight-stream
protocol), §5 (activation buffer), §6 (flow, broadcast, gather), §7 (back end), §15 (dual weight
stream) and docs/spec.md §8 (MVM cluster microarchitecture): tsr_mvm (shared by all clusters:
broadcast reader, gather, ACC read-modify-write), the per-cluster tsr_vpipe / tsr_rpipe and row
assemblers of the gather, and one tsr_mvm_cluster: front end A and front end B (stages 1-4:
tsr_mvm_ws, chunk parser, meta + unpack), activation buffer, back end (stages 5-9: 4 x
tsr_mac_lane, integer scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials, tree16),
DDR4 channel c. The two SPM ports (read port, write port of the same scratchpad) are drawn as one
block.

Facts with their sources:
- one tsr_vpipe and one tsr_rpipe per cluster: rtl/core/tsr_mvm.sv (both inside the g_cl generate
  loop), docs/rtl-mvm.md §3.1.
- the tile op's setup goes first through the broadcast pipe and starts the cluster's weight
  requests (the cluster wrapper sends the front command to tsr_mvm_ws), and only the QA lines fill
  the activation buffer: docs/rtl-mvm.md §3.1, §3.2, §3.4, §6.1. The arrow into the buffer is
  therefore named QA LINES and the setup is named in the bottom callout.
- bursts of up to 64 beats, never across a 64-beat block: docs/rtl-mvm.md §4.
- the ACC read uses the SPM read port, every write the write port: docs/rtl-mvm.md §3.6, §6.3.
- QS / BS / Dx on the activation-buffer reads: docs/spec.md §6, docs/rtl-mvm.md §3.4, §7.1.
- 32 int8 products and an exact sum per MAC lane: docs/rtl-mvm.md §7.1, §7.2.
- "all T tokens read with one address": docs/rtl-mvm.md §5.
- "rows_c x ROW_BEATS beats, one contiguous range": docs/spec.md §5 (device weight layout).
- callouts: docs/rtl-mvm.md §6.1-§6.2 (weights requested at once, DRAM latency hidden behind the
  broadcast; beats enter the back end only once the activation buffer is complete), §7.4 (the tree
  finishes one row while the next accumulates), §14.2 (MVM_BC_PIPE, MVM_RES_PIPE >= 2 for clusters
  in another SLR). The leader of the last one ends on the job pipe, the arrow from the row
  assemblers to the gather (JOB_PIPE = RES_PIPE slices), and not on tsr_rpipe: each assembler sits
  next to its cluster, so the result pipe stays inside one SLR and only the assembled jobs cross to
  the central part: §14.4 items 2 and 6, docs/platform-u200.md §8.1, rtl/core/tsr_mvm_gather.sv.
The attention modes ATTQK (scores and row maxima) and ATTPV (P.V partial outputs and row sums)
reuse the front end and the MAC lanes (docs/design/03-attention-kv.md §8, docs/rtl-mvm.md §6.5). In
tsr_mvm they add the p broadcast and the gather ATT path (a line buffer per pass row, full-line SPM
writes, a max tracker per row; AT §8.1, §8.2); in the cluster they add per-slot activation read
addresses, the w-unit (16 fmul per cycle), the capture bank (512 x 38 bit) and the convert pipes
(AT §8.2). The two blocks are drawn where there is room; the arrows into them are not drawn because
the modes use the paths that exist.

The dual weight stream (CAPS bit 16, MVM_DUAL; docs/rtl-mvm.md §15, rtl/core/tsr_mvm_cluster.sv):
- when: a weight MVM (not ATTQK or ATTPV) with T <= T_MAX / 2 (TH) on a core with CAPS bit 16 and
  the run-time enable on; every other MVM runs with stream A alone: §15.1 (the note at the foot of
  the figure says it).
- two front ends in the cluster, each a tsr_mvm_front (stream requester tsr_mvm_ws with its
  512-beat FIFO, chunk parser, meta + unpack): u_front on the channel's main port, u_front_b on the
  read-only port B. A has the rows [first, first + rA), rA = ceil(rows_c / 2), B the rest, from beat
  w_off + rA * row_beats: §15.2. The titles of the front end blocks end in A or B.
- the port of stream B is read-only, goes to the same DDR4 channel and does not go through
  tsr_mem_arb (the main port, stream A, does): §15.5, docs/platform-u200.md §4 (both ports connect
  to the same bank). The cluster figure shows no arbiter, so the names of the two arrows say it.
- the back end takes a beat only when both fronts offer one while B has beats (use_b): §15.2. Token
  slots t < TH work on A's data, slots TH + t on B's (second weight placement register, second
  scale delay lines, a MAC operand wb for the upper half): §15.3. The beats of A and B enter the
  MAC lanes side by side; B's arrow bends because its front end stands to the right of A's.
- blocks that every MVM uses name the mode where they state a dual-stream fact. In a single-stream
  MVM stream A reads all rows_c rows (rA = rows_c in tsr_mvm_cluster.sv) and, when T > TH, the slots
  at and above TH hold real tokens on A's data (§15.3 "Single beats"), so MAC LANE and STREAM
  REQUESTER A say "dual". Front end B exists only in a dual MVM, so its rows need no mode.
- the broadcast writes each QA line of token t also into token t + TH of the activation buffer
  (the write token mask of tsr_mvm_actbuf): §15.3.
- results: one item per row pair (A's row r and B's row r with its valid and last flags) through
  tsr_rpipe: §15.4. The gather has a second row assembler (u_asm_b) per cluster next to the first,
  and the jobs of the two are merged 2:1 before the cluster's job path: §15.4. tsr_mvm_gather is
  one assembler per cluster plus a central part (docs/rtl-mvm.md §14.4 item 2); the figure draws
  the assemblers beside the per-cluster pipes and the central part in the shared panel, with the
  job pipe between them.

Helper defined here (not in tsrfig): stacked_panel() draws offset copies of a panel behind it
for a unit repeated NUM_CH times (8 up and right per copy, as tsrfig box(stack=) does).

Run: build/tools/svgvenv/bin/python docs/fig/src/cluster.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import CUT_PANEL, INK_SOFT, PANEL, Fig, chamfer, wrap  # noqa: E402

W = 1600
PL, PR = 56, 1360                     # panels; callouts in the right margin
RX = (PR + W) / 2                     # callout center

# back-end columns (row B), left to right
GAP = 56
CW = [232, 272, 136, 192, 192]        # MAC, integer scale, i2f, block term, accumulate
CX = [88]
for w_ in CW[:-1]:
    CX.append(CX[-1] + w_ + GAP)
MID = [x + w_ / 2 for x, w_ in zip(CX, CW)]
MAC, ISC, I2F, FT, ACU = range(5)

# front-end columns: A at the left (stream A, main port), B to its right (stream B, port B)
AX = 160                              # axis of column A: arrows of its chain, and A's beat arrow
BEAT_B = AX + 92                      # B's beat arrow enters the MAC lane beside A's
B_COL = 560                           # left edge of column B
BX = B_COL + 72                       # axis of column B
REQ_W = 416                           # stream requester (the widest line is the row range of B)
PARSE_W, UNPACK_W = 216, 176
DDR_X, DDR_W = 88, 672

# rows (y)
SPM_Y, SPM_H = 56, 88
SH_Y, SH_B = 216, 432                 # shared tsr_mvm panel
SHB_Y, SHB_H = 248, 152               # its blocks
PIPE_Y, PIPE_H = 472, 152             # per-cluster tsr_vpipe, row assemblers, tsr_rpipe
CL_Y = 680                            # cluster panel top (room for the name of the result arrow)
BE_Y = CL_Y + 20                      # back-end band
RA_Y, RA_H = BE_Y + 60, 176           # row A: activation buffer, fmul d*Dx, tree16
RB_Y, RB_H = RA_Y + RA_H + 64, 168    # row B: MAC .. accumulate
RB_B = RB_Y + RB_H
BE_B = RB_B + 24
FE_Y = BE_B + 16                      # front-end band
YG = RB_B + 60                        # centre line of the horizontal run of B's beat arrow
RC_Y, RC_H = YG + 52, 176             # row C: meta + unpack A and B
RD_Y, RD_H = RC_Y + RC_H + 64, 152    # row D: chunk parser A and B
RE_Y, RE_H = RD_Y + RD_H + 64, 168    # row E: stream requester A and B
FE_B = RE_Y + RE_H + 24
CL_B = FE_B + 16
DDR_Y, DDR_H = CL_B + 104, 120
LEG_Y = DDR_Y + DDR_H + 56

NOTE_SPM = ("TSR_SPM is one scratchpad: the broadcast and ACC reads use its read port, the gather "
            "and ACC write through its write port.")
NOTE_DUAL = ("Dual weight stream: a weight MVM with T ≤ T_MAX / 2 on a core with CAPS bit 16 and "
             "the run-time enable on runs as two streams: A reads the rows [first, first + rA), "
             "rA = ceil(rows_c / 2), and B reads the rest through the read-only port B. While B "
             "has beats, the back end takes a beat of each in lockstep: token slots t < TH take "
             "A's, slots TH + t take B's, TH = T_MAX / 2. Every other MVM runs with stream A "
             "alone.")
note_spm, note_dual = wrap(NOTE_SPM, 1440), wrap(NOTE_DUAL, 1440)
NOTE_LH = 17 * 1.35                   # line pitch of the "note" role
NOTE_Y0 = LEG_Y + 38                  # first baseline of the first note
NOTE_Y1 = NOTE_Y0 + NOTE_LH * (1 + note_spm.count("\n")) + 8     # first baseline of the second
H = int(NOTE_Y1 + NOTE_LH * note_dual.count("\n")) + 6 + 56       # descenders, outer padding of 56
H = (H + 7) // 8 * 8

f = Fig("cluster", W, H,
        title="MVM unit: one cluster, DDR4 weight streams A and B to the SPM write",
        desc="tsr_mvm, shared by all clusters, reads the QA lines from the SPM read port and "
             "broadcasts the setup of the MVM tile op, which starts the cluster's weight "
             "requests, then the lines through each cluster's own tsr_vpipe into the cluster's "
             "activation buffer. DDR4 channel c streams weight beats into the front end: with the "
             "dual weight stream (CAPS bit 16) the cluster has two front ends, A on the "
             "channel's main port and B on the channel's read-only port B, each with a stream "
             "requester (512-beat FIFO), a chunk parser and meta + unpack. A reads the rows "
             "[first, first + rA) of the cluster and B the rest. The back end takes the beats of "
             "A and B in lockstep, the token slots below TH = T_MAX / 2 on A's beat and the "
             "slots from TH on B's, once the activation buffer is complete: 4 MAC lanes, integer "
             "scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials per row, tree16. "
             "One result item per row pair returns through the cluster's tsr_rpipe to the "
             "gather, where the cluster's row assemblers A and B build the lines that the gather "
             "writes with 16 rows per SPM line, or with ACC reads the old line and adds first. "
             "This is how a weight MVM with T <= T_MAX / 2 runs on a core with CAPS bit 16 and "
             "the run-time enable on; every other MVM runs with stream A alone. "
             "The attention modes ATTQK and ATTPV add the p broadcast and gather ATT path in "
             "tsr_mvm and the w-unit, capture bank and convert pipes in the cluster.")


def stacked_panel(fig, x, y, w, h, n, d=8):
    """Offset copies of a panel behind it (up and right, like tsrfig box(stack=)), for a unit
    repeated NUM_CH times. The name tags are drawn later, so they stay on top."""
    for i in range(n, 0, -1):
        fig.poly(chamfer(x + d * i, y - d * i, w, h, CUT_PANEL), PANEL)


# ---- SPM ---------------------------------------------------------------------------------------
f.box(PL, SPM_Y, PR - PL, SPM_H, "memory", "TSR_SPM", sub="4 MiB · 1 read port + 1 write port")

# ---- shared tsr_mvm ----------------------------------------------------------------------------
f.panel(PL, SH_Y, PR - PL, SH_B - SH_Y, label="TSR_MVM")
ab_x0, ab_x1 = CX[MAC], CX[I2F] + CW[I2F]            # activation buffer spans MAC .. i2f
# Columns of the shared row, from the right: ACC over the accumulate column, the gather over the
# row assemblers (left of tsr_rpipe), then the attention path and the broadcast reader, 48 apart.
# The broadcast column follows from the reader, a little left of the activation buffer's middle.
RP_W = 216                                           # per-cluster pipes and assemblers
RP_X0 = MID[ACU] - RP_W / 2                          # tsr_rpipe
ASM_W = 224
ASM_X1 = RP_X0 - GAP                                 # row assemblers, left of tsr_rpipe
ASM_X0 = ASM_X1 - ASM_W
GX = (ASM_X0 + ASM_X1) / 2                           # gather column: above the row assemblers
GW, ATT_W, BCR_W, SH_GAP = 184, 192, 280, 48
ATT_X = GX - GW / 2 - SH_GAP - ATT_W
BC_X = ATT_X - SH_GAP - BCR_W / 2                    # broadcast column
f.box(BC_X - BCR_W / 2, SHB_Y, BCR_W, SHB_H, "io", "BROADCAST READER",
      sub="T × QA_LINES lines\none per cycle")
f.text(PL + 36, SHB_Y + SHB_H / 2, "SHARED BY\nALL CLUSTERS", "panel", valign="middle",
       color=INK_SOFT)                                  # left of the broadcast reader
f.box(ATT_X, SHB_Y, ATT_W, SHB_H, "io", "ATT PATH", sub="p broadcast,\nline buffers,\nmax tracker",
      label_role="label_s")
f.box(GX - GW / 2, SHB_Y, GW, SHB_H, "io", "GATHER", sub="16 rows per\nSPM line,\nbyte enables")
f.box(CX[ACU], SHB_Y, CW[ACU], SHB_H, "compute", "ACC", sub="read old line\n16 × fadd")

# ---- cluster panel, stacked x NUM_CH -----------------------------------------------------------
stacked_panel(f, PL, CL_Y, PR - PL, CL_B - CL_Y, 2)
f.panel(PL, CL_Y, PR - PL, CL_B - CL_Y, label="TSR_MVM_CLUSTER · CHANNEL c")
f.tag(BC_X + 56, CL_Y, "× NUM_CH", anchor="start")            # right of the broadcast arrow
f.band(PL + 12, BE_Y, PR - PL - 24, BE_B - BE_Y, label="BACK END · STAGES 5–9",
       label_xy=(CX[MAC], BE_Y + 36))
f.band(PL + 12, FE_Y, PR - PL - 24, FE_B - FE_Y, label="FRONT END · STAGES 1–4",
       label_xy=(PR - 36, FE_B - 20), anchor="end")

# arrows that pass through blocks drawn later (the broadcast pipe)
f.arrow([(BC_X, SHB_Y + SHB_H - 12), (BC_X, RA_Y + 10)])                     # broadcast

# per-cluster pipes and row assemblers between tsr_mvm and the cluster (one of each per cluster)
pw = 216
f.box(BC_X - pw / 2, PIPE_Y, pw, PIPE_H, "io", "TSR_VPIPE", sub="BC_PIPE stages\none per cluster",
      label_role="label_s")
f.box(ASM_X0, PIPE_Y, ASM_W, PIPE_H, "io")           # the gather's row assemblers of this cluster
f.label_block(ASM_X0, PIPE_Y + 24, ASM_W, 45, "ROW ASSEMBLERS", "per cluster", label_role="label_s")
cell_w, cell_gap, cell_y = 80, 16, PIPE_Y + 88
cell_x = GX - (2 * cell_w + cell_gap) / 2
f.box(cell_x, cell_y, cell_w, 48, "neutral", "A", label_role="label_s", cut=10)
f.box(cell_x + cell_w + cell_gap, cell_y, cell_w, 48, "neutral", "B", label_role="label_s",
      cut=10)
f.box(RP_X0, PIPE_Y, RP_W, PIPE_H, "io", "TSR_RPIPE",
      sub="RES_PIPE stages\none per cluster\nrow-pair items", label_role="label_s")

# row A: activation buffer, fmul d*Dx, tree16
f.box(ab_x0, RA_Y, ab_x1 - ab_x0, RA_H, "memory", "ACTIVATION BUFFER",
      sub="tsr_mvm_actbuf · QS, D, BS per token\nall T tokens read with one address\n"
          "dual stream: each line also into token t + TH,\nTH = T_MAX / 2")
f.box(CX[FT], RA_Y, CW[FT], RA_H, "compute", "FMUL",
      sub="d·Dx\nQ4_K: dmin·Dx\nd, dmin from\nthe meta")
f.box(CX[ACU], RA_Y, CW[ACU], RA_H, "compute", "TREE16", sub="8 + 4 + 2 + 1\nfadds")

# row B: back end
f.box(CX[MAC], RB_Y, CW[MAC], RB_H, "compute", "MAC LANE",
      sub="32 int8 products\nexact sum\ndual: slots < TH\non A, ≥ TH on B", stack=2)
f.tag(CX[MAC] + CW[MAC] + 16, RB_Y - 16, "× 4", anchor="end")   # on the back copy, clear of QS
f.box(CX[ISC], RB_Y, CW[ISC], RB_H, "compute", "INTEGER SCALE",
      sub="Q4_K:\nisum = Σ sc·S\nmsum = Σ m·(BS pair)")
f.box(CX[I2F], RB_Y, CW[I2F], RB_H, "compute", "2 × I2F", sub="int32\nto f32", label_role="label_s")
f.box(CX[FT], RB_Y, CW[FT], RB_H, "compute", "BLOCK TERM",
      sub="× f32(isum)\nQ4_K: fsub\n− msum term")
f.box(CX[ACU], RB_Y, CW[ACU], RB_H, "compute", "ACCUMULATE", sub="16 partials\nper row")

# front end A (left column) and B (right column): the weight stream runs upward, DDR4 -> requester
# -> chunk parser -> meta + unpack -> the MAC lanes
for tag, col, ax, rows in (("A", CX[MAC], AX, "dual: rows [first, first + rA)"),
                           ("B", B_COL, BX, "rows [first + rA, first + rows_c)")):
    f.box(col, RC_Y, UNPACK_W, RC_H, "control", f"META +\nUNPACK {tag}",
          sub="fp16 to f32\nQ4_K scales\nint8 lanes")
    f.box(col, RD_Y, PARSE_W, RD_H, "control", f"CHUNK\nPARSER {tag}",
          sub="META or quant\nrow, block tags")
    f.box(col, RE_Y, REQ_W, RE_H, "io", f"STREAM REQUESTER {tag}",
          sub="tsr_mvm_ws · ≤ 64-beat bursts\n512-beat FIFO: 1 beat/cycle\n"
              f"up to ~440 cycles round trip\n{rows}")

# attention modes: right of the front ends, in the front-end band's free space
f.box(CX[ACU] - 32, RC_Y, CW[ACU] + 32, RC_H, "compute", "ATTQK · ATTPV",
      sub="per-slot reads\nw-unit, capture\nconvert pipes", label_role="label_s")

# DDR4 channel c: the main port (stream A) and the read-only port B (stream B)
f.box(DDR_X, DDR_Y, DDR_W, DDR_H, "memory", "DDR4 CH c",
      sub="rows_c × ROW_BEATS beats\none contiguous range")

# ---- arrows --------------------------------------------------------------------------------------
spm_b = SPM_Y + SPM_H
f.arrow([(BC_X, spm_b - 12), (BC_X, SHB_Y + 10)])                              # read port
f.arrow([(MID[ACU], spm_b - 12), (MID[ACU], SHB_Y + 10)], tail=True)          # ACC r/w
f.arrow([(GX, SHB_Y + 12), (GX, spm_b - 10)])                                  # plain write
f.arrow([(GX + GW / 2 - 12, SHB_Y + SHB_H / 2), (CX[ACU] + 10, SHB_Y + SHB_H / 2)])   # to ACC

pipe_mid = PIPE_Y + PIPE_H / 2
f.arrow([(MID[ACU], RA_Y + 12), (MID[ACU], PIPE_Y + PIPE_H - 10)])             # row results
f.arrow([(RP_X0 + 8, pipe_mid), (ASM_X1 - 10, pipe_mid)])                      # row-pair items
f.arrow([(GX, PIPE_Y + 12), (GX, SHB_Y + SHB_H - 10)])                         # assembled jobs: job pipe

ra_b, rb_mid = RA_Y + RA_H, RB_Y + RB_H / 2
f.arrow([(MID[MAC], ra_b - 10), (MID[MAC], RB_Y + 10)])                        # QS
f.arrow([(MID[ISC], ra_b - 10), (MID[ISC], RB_Y + 10)])                        # BS
f.arrow([(ab_x1 - 12, RA_Y + RA_H / 2), (CX[FT] + 10, RA_Y + RA_H / 2)])       # Dx
f.arrow([(MID[FT], ra_b - 8), (MID[FT], RB_Y + 10)])                           # d*Dx
for a, b in [(MAC, ISC), (ISC, I2F), (I2F, FT), (FT, ACU)]:
    f.arrow([(CX[a] + CW[a] - 12, rb_mid), (CX[b] + 10, rb_mid)])
f.arrow([(MID[ACU], RB_Y + 12), (MID[ACU], ra_b - 10)])                        # to the tree

# the beat pair into the MAC lanes: A straight, B bends because its front end is to the right
f.arrow([(AX, RC_Y + 12), (AX, RB_B - 10)], fat=True)                          # tagged beat A
f.arrow([(BX, RC_Y + 12), (BX, YG), (BEAT_B, YG), (BEAT_B, RB_B - 10)], fat=True)   # tagged beat B
for ax, col in ((AX, CX[MAC]), (BX, B_COL)):                                   # the two chains
    f.arrow([(ax, RD_Y + 12), (ax, RC_Y + RC_H - 10)], fat=True)
    f.arrow([(ax, RE_Y + 12), (ax, RD_Y + RD_H - 10)], fat=True)
    f.arrow([(ax, DDR_Y + 8), (ax, RE_Y + RE_H - 10)], fat=True)               # weight beats

# arrow names
f.text(BC_X + 44, (spm_b + SHB_Y) / 2, "READ PORT", "callout", valign="middle")
f.text(GX + 44, (spm_b + SHB_Y) / 2, "PLAIN", "callout", valign="middle")
f.text(BC_X + pw / 2 + 24, pipe_mid, "QA LINES", "callout", valign="middle")
job_y = (SH_B + PIPE_Y) / 2           # middle of the free band between the shared panel and the pipes
f.text(GX - 44, job_y, "JOB PIPE", "callout", anchor="end", valign="middle")
# middle of the free band between the pipes and the top copy of the cluster panel
f.text(MID[ACU] - 44, (PIPE_Y + PIPE_H + CL_Y - 16) / 2, "ROW RESULTS", "callout", anchor="end",
       valign="middle")
gy = (ra_b + RB_Y) / 2
f.text(MID[MAC] - 32, gy, "QS", "callout", anchor="end", valign="middle")
f.text(MID[ISC] + 32, gy, "BS", "callout", valign="middle")
f.text((ab_x1 + CX[FT]) / 2 - 6, RA_Y + RA_H / 2 - 44, "Dx", "callout", anchor="middle",
       valign="middle")
f.text(AX - 32, RB_B + 60, "A", "callout", anchor="end", valign="middle")
f.text(BEAT_B + 40, YG + 38, "B · LOCKSTEP WITH A", "callout", valign="middle")
port_y = (CL_B + DDR_Y) / 2
f.text(AX + 44, port_y, "PORT A\nVIA tsr_mem_arb", "callout", valign="middle")
f.text(BX - 44, port_y, "PORT B\nREAD-ONLY,\nNO tsr_mem_arb", "callout", anchor="end",
       valign="middle")

# ---- callouts (right margin) ------------------------------------------------------------------
f.callout("MVM_BC_PIPE AND\nMVM_RES_PIPE:\n≥ 2 STAGES FOR\nA CLUSTER IN\nANOTHER SLR",
          (RX, job_y), target=(GX, job_y))      # on the job pipe: the assembled jobs cross the SLR
f.callout("THE TREE\nFINISHES ONE\nROW WHILE THE\nNEXT ROW\nACCUMULATES",
          (RX, gy), target=(MID[ACU], gy))
f.callout("TAGGED BEATS\nENTER THE BACK\nEND ONLY ONCE\nTHE ACTIVATION\nBUFFER IS\nCOMPLETE",
          (RX, YG), target=(BX, YG))
f.callout("AFTER SETUP,\nWEIGHT REQUESTS\nSTART AT ONCE:\nDRAM LATENCY\nHIDES BEHIND\nTHE BROADCAST",
          (RX, port_y + 16), target=(BX, port_y + 16))

# ---- legend and footnotes ----------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                        ("io", "DATA MOVEMENT")])
f.text(W / 2, NOTE_Y0, note_spm, "note", anchor="middle", color=INK_SOFT)
f.text(W / 2, NOTE_Y1, note_dual, "note", anchor="middle", color=INK_SOFT)

f.save()

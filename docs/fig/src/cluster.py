"""Figure `cluster`: the MVM unit datapath of one cluster, DDR4 weight beats to the SPM write.

Content follows docs/rtl-m3.md §2 (block diagram), §3 (internal interfaces), §4 (weight-stream
protocol), §5 (activation buffer), §6 (flow, broadcast, gather), §7 (back end) and docs/spec.md §8
(MVM cluster microarchitecture): tsr_mvm (shared by all clusters: broadcast reader, gather, ACC
read-modify-write), the per-cluster tsr_vpipe / tsr_rpipe, and one tsr_mvm_cluster: front end
(stages 1-4: tsr_mvm_ws, chunk parser, meta + unpack), activation buffer, back end (stages 5-9:
4 x tsr_mac_lane, integer scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials,
tree16), DDR4 channel c. The two SPM ports (read port, write port of the same scratchpad) are drawn
as one block. The datapath is solid (rtl/core; ran on the card).

Facts with their sources:
- one tsr_vpipe and one tsr_rpipe per cluster: rtl/core/tsr_mvm.sv (both inside the g_cl generate
  loop), docs/rtl-m3.md §3.1.
- the tile op's setup goes first through the broadcast pipe and starts the cluster's weight
  requests (the cluster wrapper sends the front command to tsr_mvm_ws), and only the QA lines fill
  the activation buffer: docs/rtl-m3.md §3.1, §3.2, §3.4, §6.1. The arrow into the buffer is
  therefore named QA LINES and the setup is named in the bottom callout.
- bursts of up to 64 beats, never across a 64-beat block: docs/rtl-m3.md §4.
- the ACC read uses the SPM read port, every write the write port: docs/rtl-m3.md §3.6, §6.3.
- QS / BS / Dx on the activation-buffer reads: docs/spec.md §6, docs/rtl-m3.md §3.4, §7.1.
- 32 int8 products and an exact sum per MAC lane: docs/rtl-m3.md §7.1, §7.2.
- "all T tokens read with one address": docs/rtl-m3.md §5.
- "rows_c x ROW_BEATS beats, one contiguous range": docs/spec.md §5 (device weight layout).
- callouts: docs/rtl-m3.md §6.1-§6.2 (weights requested at once, DRAM latency hidden behind the
  broadcast; beats enter the back end only once the activation buffer is complete), §7.4 (the tree
  finishes one row while the next accumulates), §14.2 (MVM_BC_PIPE, MVM_RES_PIPE >= 2 for clusters
  in another SLR).
Planned, drawn dashed (docs/design/03-attention-kv.md §8, docs/rtl-m3.md §6.5; no RTL yet): the
attention modes ATTQK (scores and row maxima) and ATTPV (P.V partial outputs and row sums) reuse
the front end and the MAC lanes. In tsr_mvm they add the p broadcast and the gather ATT path (a
line buffer per pass row, full-line SPM writes, a max tracker per row; AT §8.1, §8.2); in the
cluster they add per-slot activation read addresses, the w-unit (16 fmul per cycle), the capture
bank (512 x 38 bit) and the convert pipes (AT §8.2). The two blocks are drawn where there is room;
the arrows into them are not drawn because the modes use the paths that exist.

Helper defined here (not in tsrfig): stacked_panel() draws offset copies of a panel behind it
for a unit repeated NUM_CH times (8 up and right per copy, as tsrfig box(stack=) does).

Run: build/tools/svgvenv/bin/python docs/fig/src/cluster.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import CUT_PANEL, INK_SOFT, PANEL, Fig, chamfer, wrap  # noqa: E402

W = 1600
PL, PR = 56, 1360                     # panels; callouts in the right margin
RX = (PR + W) / 2                     # callout centre

# back-end columns (row B), left to right
GAP = 56
CW = [200, 272, 136, 224, 192]        # MAC, integer scale, i2f, block term, accumulate
CX = [88]
for w_ in CW[:-1]:
    CX.append(CX[-1] + w_ + GAP)
MID = [x + w_ / 2 for x, w_ in zip(CX, CW)]
MAC, ISC, I2F, FT, ACU = range(5)

# rows (y)
SPM_Y, SPM_H = 56, 88
SH_Y, SH_B = 216, 432                 # shared tsr_mvm panel
SHB_Y, SHB_H = 248, 152               # its blocks
PIPE_Y, PIPE_H = 464, 112             # per-cluster tsr_vpipe / tsr_rpipe
CL_Y = 624                            # cluster panel top
BE_Y, BE_B = 644, 1120                # back-end band
RA_Y, RA_H = 704, 176                 # row A: activation buffer, fmul d*Dx, tree16
RB_Y, RB_H = 944, 152                 # row B: MAC .. accumulate
FE_Y, FE_B = 1160, 1400               # front-end band
RC_Y, RC_H = 1184, 168                # row C: meta + unpack, parser, stream requester
CL_B = 1416
DDR_Y, DDR_H = 1496, 120
LEG_Y = 1672
H = LEG_Y + 96

f = Fig("cluster", W, H,
        title="MVM unit: one cluster, DDR4 weight beats to the SPM write",
        desc="tsr_mvm, shared by all clusters, reads the QA lines from the SPM read port and "
             "broadcasts the setup of the MVM tile op, which starts the cluster's weight "
             "requests, then the "
             "lines through each cluster's own tsr_vpipe into the cluster's activation buffer. "
             "DDR4 channel c streams weight beats into the "
             "front end (stream requester with a 512-beat FIFO, chunk parser, meta + unpack); "
             "tagged beats enter the back end once the activation buffer is complete: 4 MAC "
             "lanes, integer scale, 2 x i2f, fmul d*Dx, block term, accumulate into 16 partials "
             "per row, tree16. Row results return through the cluster's tsr_rpipe to the shared "
             "gather, which "
             "writes 16 rows per SPM line, or with ACC reads the old line and adds first. "
             "Planned, drawn dashed: the attention modes ATTQK and ATTPV, with the p broadcast "
             "and gather ATT path in tsr_mvm and the w-unit, capture bank and convert pipes in "
             "the cluster.")


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
BC_X = (ab_x0 + ab_x1) / 2                           # broadcast column
bcr_w = 288
f.box(BC_X - bcr_w / 2, SHB_Y, bcr_w, SHB_H, "io", "BROADCAST READER",
      sub="T × QA_LINES lines\none per cycle")
f.box(CX[FT], SHB_Y, CW[FT], SHB_H, "compute", "ACC", sub="read old line\n16 × fadd")
f.box(CX[ACU], SHB_Y, CW[ACU], SHB_H, "io", "GATHER", sub="16 rows per\nSPM line,\nbyte enables")
f.text(PL + 36, SHB_Y + SHB_H / 2, "SHARED BY\nALL CLUSTERS", "panel", valign="middle",
       color=INK_SOFT)                                  # left of the broadcast reader
f.box(624, SHB_Y, 208, SHB_H, "io", "ATT PATH", sub="p broadcast,\nline buffers,\nmax tracker",
      planned=True, tag="PLANNED · M6", label_role="label_s")

# ---- cluster panel, stacked x NUM_CH -----------------------------------------------------------
stacked_panel(f, PL, CL_Y, PR - PL, CL_B - CL_Y, 2)
f.panel(PL, CL_Y, PR - PL, CL_B - CL_Y, label="TSR_MVM_CLUSTER · CHANNEL c")
f.tag(MID[ACU] - 40, CL_Y, "× NUM_CH", anchor="end")          # left of the row-result arrow
f.band(PL + 12, BE_Y, PR - PL - 24, BE_B - BE_Y, label="BACK END · STAGES 5–9",
       label_xy=(CX[MAC], BE_Y + 36))
ws_x, ws_w = CX[I2F], CX[FT] + CW[FT] - CX[I2F]      # stream requester spans i2f .. block term
f.band(PL + 12, FE_Y, PR - PL - 24, FE_B - FE_Y, label="FRONT END · STAGES 1–4",
       label_xy=(CX[MAC], FE_B - 20))

# arrows that pass through blocks drawn later (pipes)
f.arrow([(BC_X, SHB_Y + SHB_H - 12), (BC_X, RA_Y + 10)])                     # broadcast
f.arrow([(MID[ACU], RA_Y + 12), (MID[ACU], SHB_Y + SHB_H - 10)])             # row results

# per-cluster pipes between tsr_mvm and the cluster (one of each per cluster)
pw = 216
f.box(BC_X - pw / 2, PIPE_Y, pw, PIPE_H, "io", "TSR_VPIPE", sub="BC_PIPE stages\none per cluster",
      label_role="label_s")
f.box(MID[ACU] - pw / 2, PIPE_Y, pw, PIPE_H, "io", "TSR_RPIPE",
      sub="RES_PIPE stages\none per cluster", label_role="label_s")

# row A: activation buffer, fmul d*Dx, tree16
f.box(ab_x0, RA_Y, ab_x1 - ab_x0, RA_H, "memory", "ACTIVATION BUFFER",
      sub="tsr_mvm_actbuf · QS, D, BS per token\nall T tokens read with one address")
f.box(CX[FT], RA_Y, CW[FT], RA_H, "compute", "FMUL",
      sub="d·Dx\nQ4_K: dmin·Dx\nd, dmin from\nthe meta")
f.box(CX[ACU], RA_Y, CW[ACU], RA_H, "compute", "TREE16", sub="8 + 4 + 2 + 1\nfadds")

# row B: back end
f.box(CX[MAC], RB_Y, CW[MAC], RB_H, "compute", "MAC LANE", sub="32 int8\nproducts\nexact sum",
      stack=2)
f.tag(CX[MAC] + CW[MAC] + 16, RB_Y - 16, "× 4", anchor="end")   # on the back copy, clear of QS
f.box(CX[ISC], RB_Y, CW[ISC], RB_H, "compute", "INTEGER SCALE",
      sub="Q4_K:\nisum = Σ sc·S\nmsum = Σ m·(BS pair)")
f.box(CX[I2F], RB_Y, CW[I2F], RB_H, "compute", "2 × I2F", sub="int32\nto f32", label_role="label_s")
f.box(CX[FT], RB_Y, CW[FT], RB_H, "compute", "BLOCK TERM",
      sub="× f32(isum)\nQ4_K: fsub\n− msum term")
f.box(CX[ACU], RB_Y, CW[ACU], RB_H, "compute", "ACCUMULATE", sub="16 partials\nper row")

# row C: front end (the weight stream runs right to left, then up into the MAC lanes)
f.box(CX[MAC], RC_Y, CW[MAC], RC_H, "control", "META +\nUNPACK",
      sub="fp16 to f32\nQ4_K scales\nint8 lanes")
f.box(CX[ISC], RC_Y, CW[ISC], RC_H, "control", "CHUNK\nPARSER",
      sub="META or quant\nrow, block tags")
f.box(ws_x, RC_Y, ws_w, RC_H, "io", "STREAM REQUESTER",
      sub="tsr_mvm_ws · ≤ 64-beat bursts\n512-beat FIFO: 1 beat/cycle\nup to ~440 cycles round trip")

# planned attention modes: right of the stream requester, in the front-end band's free space
f.box(CX[ACU] - 32, RC_Y, CW[ACU] + 32, RC_H, "compute", "ATTQK · ATTPV",
      sub="per-slot reads\nw-unit, capture\nconvert pipes", planned=True,
      tag="PLANNED · M6", label_role="label_s")

# DDR4 channel c
ws_mid = ws_x + ws_w / 2
ddr_w = 336
f.box(ws_mid - ddr_w / 2, DDR_Y, ddr_w, DDR_H, "memory", "DDR4 CH c",
      sub="rows_c × ROW_BEATS beats\none contiguous range")

# ---- arrows --------------------------------------------------------------------------------------
spm_b = SPM_Y + SPM_H
f.arrow([(BC_X, spm_b - 12), (BC_X, SHB_Y + 10)])                              # read port
f.arrow([(MID[FT], spm_b - 12), (MID[FT], SHB_Y + 10)], tail=True)            # ACC r/w
f.arrow([(MID[ACU], SHB_Y + 12), (MID[ACU], spm_b - 10)])                      # plain write
f.arrow([(CX[ACU] + 12, SHB_Y + SHB_H / 2), (CX[FT] + CW[FT] - 10, SHB_Y + SHB_H / 2)])

ra_b, rb_mid = RA_Y + RA_H, RB_Y + RB_H / 2
f.arrow([(MID[MAC], ra_b - 10), (MID[MAC], RB_Y + 10)])                        # QS
f.arrow([(MID[ISC], ra_b - 10), (MID[ISC], RB_Y + 10)])                        # BS
f.arrow([(ab_x1 - 12, RA_Y + RA_H / 2), (CX[FT] + 10, RA_Y + RA_H / 2)])       # Dx
f.arrow([(MID[FT], ra_b - 8), (MID[FT], RB_Y + 10)])                           # d*Dx
for a, b in [(MAC, ISC), (ISC, I2F), (I2F, FT), (FT, ACU)]:
    f.arrow([(CX[a] + CW[a] - 12, rb_mid), (CX[b] + 10, rb_mid)])
f.arrow([(MID[ACU], RB_Y + 12), (MID[ACU], ra_b - 10)])                        # to the tree

rc_mid = RC_Y + RC_H / 2
f.arrow([(MID[MAC], RC_Y + 12), (MID[MAC], RB_Y + RB_H - 10)], fat=True)       # tagged beat
f.arrow([(CX[ISC] + 12, rc_mid), (CX[MAC] + CW[MAC] - 10, rc_mid)], fat=True)
f.arrow([(ws_x + 12, rc_mid), (CX[ISC] + CW[ISC] - 10, rc_mid)], fat=True)
f.arrow([(ws_mid, DDR_Y + 8), (ws_mid, RC_Y + RC_H - 10)], fat=True)           # weight beats

# arrow names
f.text(BC_X + 44, (spm_b + SHB_Y) / 2, "READ PORT", "callout", valign="middle")
f.text(MID[ACU] + 44, (spm_b + SHB_Y) / 2, "PLAIN", "callout", valign="middle")
f.text(BC_X + pw / 2 + 24, PIPE_Y + PIPE_H / 2, "QA LINES", "callout", valign="middle")
f.text(MID[ACU] - pw / 2 - 24, PIPE_Y + PIPE_H / 2, "ROW\nRESULTS", "callout", anchor="end",
       valign="middle")
gy = (ra_b + RB_Y) / 2
f.text(MID[MAC] - 32, gy, "QS", "callout", anchor="end", valign="middle")
f.text(MID[ISC] + 32, gy, "BS", "callout", valign="middle")
f.text((ab_x1 + CX[FT]) / 2, RA_Y + RA_H / 2 - 40, "Dx", "callout", anchor="middle",
       valign="middle")

# ---- callouts (right margin) ------------------------------------------------------------------
f.callout("MVM_BC_PIPE AND\nMVM_RES_PIPE:\n≥ 2 STAGES FOR\nA CLUSTER IN\nANOTHER SLR",
          (RX, PIPE_Y + PIPE_H / 2), target=(MID[ACU] + pw / 2 - 8, PIPE_Y + PIPE_H / 2))
f.callout("THE TREE\nFINISHES ONE\nROW WHILE THE\nNEXT ROW\nACCUMULATES",
          (RX, gy), target=(MID[ACU], gy))
gap_y = (BE_B + FE_Y) / 2
f.callout("TAGGED BEATS\nENTER THE BACK\nEND ONLY ONCE\nTHE ACTIVATION\nBUFFER IS\nCOMPLETE",
          (RX, gap_y), target=(MID[MAC], gap_y))
ws_y = (CL_B + DDR_Y) / 2
f.callout("AFTER SETUP,\nWEIGHT REQUESTS\nSTART AT ONCE:\nDRAM LATENCY\nHIDES BEHIND\nTHE BROADCAST",
          (RX, ws_y + 16), target=(ws_mid, ws_y + 16))

# ---- legend and footnote -----------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                        ("io", "DATA MOVEMENT"), ("neutral", "PLANNED", True)])
f.text(W / 2, LEG_Y + 38, wrap(
    "TSR_SPM is one scratchpad: the broadcast and ACC reads use its read port, the gather and ACC "
    "write through its write port. Solid blocks are the datapath (RTL, ran on the card); the "
    "dashed blocks are the attention modes (planned).", 1440), "note", anchor="middle",
       color=INK_SOFT)

f.save()

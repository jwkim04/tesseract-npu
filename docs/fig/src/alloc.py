"""Figure `alloc`: symmetric allocation of one weight tensor over the four DDR4 channels.

Content follows docs/spec.md §4 (Memory model: symmetric allocation, the per-channel map) and §5
(Device weight layout): a weight tensor of N rows x K is split into four contiguous row ranges of
R = ceil(N/4) rows (ch0: rows 0 .. R-1, ch1: R .. 2R-1, ch2: 2R .. 3R-1, ch3: 3R .. N-1); every
range sits at the same offset W_OFF of its channel, and cluster c streams only channel c.
Additions, with their sources:
- the per-channel map (docs/design/01-execution-model.md §11, the DRAM map): OPL, IO and AUX in the
  low 512 MiB of ch0; the low 512 MiB of ch1-ch3 hold compute chunks and other activations; the
  WEIGHT heap runs from 512 MiB up (the KV planes of docs/design/03-attention-kv.md §4.5 grow down
  from the end of the channel and are not drawn);
- row r of channel c starts at W_OFF + (r - c*R) x ROW_BEATS(fmt, K) (docs/spec.md §4); rows
  start on 64 B beat boundaries (docs/spec.md §5);
- the host converts (unless Q8_0 or Q4_K) and repacks row by row before writing (docs/spec.md
  §12, ggml backend integration: weight placement and conversion);
- one MVM names the tensor with one offset; each cluster works on its own range (docs/spec.md §4).
The inset on the right magnifies the ch3 slice (rows 3R .. N-1 back to back from W_OFF); its top
edge is the W_OFF line itself and a dashed zoom line joins the slice's bottom to the inset's.
Row ranges are written "0 TO R-1" because the ellipsis glyph is too small at README width.

Run: build/tools/svgvenv/bin/python docs/fig/src/alloc.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import BAND, INK, INK_SOFT, Fig  # noqa: E402

W = 1600
COLW, GAP = 200, 32
X0 = 192                                           # first channel column
CX = [X0 + c * (COLW + GAP) for c in range(4)]     # 192, 424, 656, 888
CC = [x + COLW / 2 for x in CX]                    # column centres
XR = CX[3] + COLW                                  # 1088
PX0, PX1 = X0 - 40, XR + 40                        # host / DRAM / core panels: 152 .. 1128
IX0, IX1 = 1168, 1544                              # zoom inset

# rows (y)
HOST_Y, HOST_B = 56, 248
T_Y, T_H = 120, 96                                 # tensor sections
DRAM_Y = 288
HD_Y, HD_H = 320, 56                               # "DDR4 CHc" headers
LO_Y, LO_H = 400, 88                               # 0 - 512 MiB
WT_Y = 512                                         # WEIGHT heap band
WOFF = 640                                         # the slice top: W_OFF
SL_H = 144
SL_B = WOFF + SL_H
WT_B = SL_B + 64
DRAM_B = WT_B + 24
CORE_Y = DRAM_B + 104
CL_Y, CL_H = CORE_Y + 40, 88
CORE_B = CL_Y + CL_H + 32
NOTE_Y = CORE_B + 52
H = NOTE_Y + 44                                    # card height on the 8-unit grid

RANGES = ["0 TO R−1", "R TO 2R−1", "2R TO 3R−1", "3R TO N−1"]

f = Fig("alloc", W, H,
        title="Symmetric allocation of a weight tensor",
        desc="A weight tensor of N rows by K is split into four row ranges of R = ceil(N/4) "
             "rows. The host repacks the rows and writes range c into DDR4 channel c, at the "
             "same offset W_OFF in every channel, where rows sit back to back: row r of "
             "channel c starts at W_OFF + (r - c*R) x ROW_BEATS(fmt, K). Cluster c streams only "
             "channel c, so one MVM names the tensor with one offset.")

# ---- host: the tensor --------------------------------------------------------------------------
f.panel(PX0, HOST_Y, PX1 - PX0, HOST_B - HOST_Y, label="HOST PC", label_at="middle")
f.text((X0 + XR) / 2, T_Y - 28, "WEIGHT TENSOR · N ROWS × K", "callout", anchor="middle")
for c in range(4):
    f.box(CX[c], T_Y, COLW, T_H, "host", "ROWS\n" + RANGES[c], label_role="label_s")

# ---- card DRAM -----------------------------------------------------------------------------------
f.panel(PX0, DRAM_Y, PX1 - PX0, DRAM_B - DRAM_Y, label="CARD DRAM", label_at="middle")

# W_OFF: one dashed line through all four channels into the inset (drawn under the blocks)
IX_BLK = IX0 + 24                                  # inset row block
f.hline(PX0 - 8, IX_BLK + 8, WOFF, dashed=True, color=INK, sw=1.5)

for c in range(4):
    x = CX[c]
    # WEIGHT heap band, 512 MiB to the end of the channel
    f.band(x, WT_Y, COLW, WT_B - WT_Y, fill=BAND)
    f.text(x + COLW / 2, WT_Y + 34, "WEIGHT", "panel", anchor="middle", color=INK_SOFT)
    f.text(x + COLW / 2, WT_Y + 64, "512 MiB – end", "sub", anchor="middle", color=INK_SOFT)
    # low region
    if c == 0:
        f.box(x, LO_Y, COLW, LO_H, "neutral", "OPL, IO, AUX", sub="0 – 512 MiB",
              label_role="label_s")
    else:
        f.band(x, LO_Y, COLW, LO_H, fill=BAND)
        f.text(x + COLW / 2, LO_Y + LO_H / 2, "compute chunks,\nactivations\n0 – 512 MiB", "sub",
               anchor="middle", valign="middle", color=INK_SOFT)
    f.box(x, WOFF, COLW, SL_H, "memory", "ROWS\n" + RANGES[c], label_role="label_s")
    f.box(x, HD_Y, COLW, HD_H, "memory", f"DDR4 CH{c}", label_role="label_s", cut=12)
    # repacked rows written into channel c
    f.arrow([(CC[c], T_Y + T_H - 12), (CC[c], HD_Y + 10)])

# W_OFF tag on the left end of the line
f.tag(PX0 - 12, WOFF, "W_OFF", anchor="end")

# ---- tsr_core: the clusters ----------------------------------------------------------------------
f.panel(PX0, CORE_Y, PX1 - PX0, CORE_B - CORE_Y, label="TSR_CORE", label_at="middle")
for c in range(4):
    # weight stream: drawn after the TSR_CORE panel so its head shows, over the slice's bottom
    # edge (12 inside the slice) and under the cluster (tip 12 inside it)
    f.arrow([(CC[c], SL_B - 12), (CC[c], CL_Y + 12)], fat=True)
    f.box(CX[c], CL_Y, COLW, CL_H, "compute", f"CLUSTER {c}", label_role="label_s")

# ---- inset: the rows of channel 3, back to back ---------------------------------------------------
rows = [("ROW 3R", "W_OFF"), ("ROW 3R+1", "+ 1 × ROW_BEATS"), ("ROW 3R+2", "+ 2 × ROW_BEATS"),
        ("", ""), ("ROW N−1", "")]
RH = 48
BW_IN = 128
IN_Y = LO_Y + 48
BLK_B = WOFF + RH * len(rows)
IN_B = BLK_B + 32
f.panel(IX0, IN_Y, IX1 - IX0, IN_B - IN_Y, label="CH3 · ROWS BACK TO BACK", label_at="middle")
f.text(IX0 + 28, IN_Y + 44, "ROW r OF CHANNEL c\nSTARTS AT BEAT\nW_OFF + (r − c·R)\n× ROW_BEATS(fmt, K)",
       "callout", valign="top")
# zoom line from the bottom of the ch3 slice to the bottom of the inset block (dashed); the W_OFF
# line is the top one
f.line([(XR, SL_B), (IX_BLK, BLK_B)], end=False, dashed=True, sw=1.5)
f.box(IX_BLK, WOFF, BW_IN, BLK_B - WOFF, "memory", cut=10)
for i, (lab, addr) in enumerate(rows):
    y = WOFF + i * RH
    if i:
        f.hline(IX_BLK, IX_BLK + BW_IN, y, dashed=False, color=INK, sw=1.5)
    if lab:
        f.text(IX_BLK + BW_IN / 2, y + RH / 2, lab, "label_s", anchor="middle", valign="middle")
    else:                                          # elided rows
        for k in (-14, 0, 14):
            f.dot(IX_BLK + BW_IN / 2 + k, y + RH / 2, r=3)
    if addr:
        f.hline(IX_BLK + BW_IN, IX_BLK + BW_IN + 16, y, dashed=False, color=INK, sw=1.5)
        f.text(IX_BLK + BW_IN + 24, y, addr, "sub", valign="middle")

f.text(IX_BLK + BW_IN + 24, BLK_B - RH - 12, "rows start on\n64 B beat\nboundaries", "sub",
       valign="top", color=INK_SOFT)

# ---- callouts ----------------------------------------------------------------------------------
RX = (PX1 + W) / 2 + 24
f.callout("R = ⌈N/4⌉ ROWS\nPER CHANNEL", (RX, T_Y + T_H / 2), target=(XR - 24, T_Y + T_H / 2))
up_y = (HOST_B + DRAM_Y) / 2
f.callout("CONVERTED IF\nNEEDED, REPACKED\nROW BY ROW", (RX, up_y), target=(CC[3], up_y))
st_y = (DRAM_B + CORE_Y) / 2
f.callout("WEIGHT\nSTREAM", ((56 + PX0) / 2 + 8, st_y), target=(CC[0], st_y))

f.text(W / 2, NOTE_Y, "Cluster c streams only channel c. All ranges sit at the same W_OFF, so "
       "one MVM names the whole tensor with one offset.", "note", anchor="middle", color=INK_SOFT)

f.save()

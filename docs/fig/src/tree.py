"""Figure `tree`: deterministic row summation, block terms to the SPM write.

Content follows docs/spec.md §7 (Numeric rules: block terms, the row accumulation, the tree
formula, the 16 partials) and docs/rtl-mvm.md §7.3-§7.4 (accumulators and the tree): block term i of
a row goes to partial p_(i mod 16); the 16 partials start at +0.0 for every row; y is the fixed
tree16 sum (8 + 4 + 2 + 1 fadds, pairs taken from the left); the ACC flag of the MVM selects
y_out = y or y_out = y_old + y.

Additions, with their sources:
- block-term formulas, one term per 32 (Q8_0) or 256 (Q4_K) weights, no FMA: docs/spec.md §7
  (block terms) and docs/rtl-mvm.md §7.1.
- module names tsr_mvm_back (stage 9, accumulator and tree) and tsr_mvm_gather (stage 10,
  output, ACC read-modify-write): docs/rtl-mvm.md §2, §6.3, §7.
- "ACC: one fadd, old value first": docs/spec.md §21 (Implementation clarifications, "ACC
  addition"), docs/rtl-mvm.md §6.3.
- SPM placement of row n, token t (line dst + t·stride + n/16, lane n mod 16): docs/rtl-mvm.md
  §6.3 (gather).
- callouts: docs/spec.md §7 and docs/numcheck.md ("16 partials" is an architectural constant; two
  terms of one partial are at least 8 cycles apart; a sequential-sum mutant is wrong on 67-87 % of
  the elements, so the order is verified bit-exactly; every product and sum rounded separately).
The VU has its own bit-exact rules (VU §6).

Helpers defined here (not in tsrfig): join() draws the bracket that joins two tree inputs into
one fadd node.

Run: TSR_FIG_PNG_DIR=... build/tools/svgvenv/bin/python docs/fig/src/tree.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import BAND, INK, INK_SOFT, Fig, fmt  # noqa: E402

W = 1600
PX0, PX1 = 224, 1376                  # panels; callouts in both 224 margins
LX, RX = PX0 / 2, (PX1 + W) / 2       # callout centers

# 16 lanes, one per partial
PITCH, LANE_W = 68, 56
LANE0 = 256 + (PITCH - LANE_W) / 2
LANE = [LANE0 + PITCH * j for j in range(16)]          # lane x (left)
LC = [x + LANE_W / 2 for x in LANE]                    # lane centers
LV = [LC]                                              # node centers per tree level
for _ in range(4):
    prev = LV[-1]
    LV.append([(prev[2 * k] + prev[2 * k + 1]) / 2 for k in range(len(prev) // 2)])

# rows (y)
TOP_Y = 56
HD_Y, HD_H = 96, 120                  # block-term header
LN_Y = 288                            # lanes start
CHIP_H, CHIP_P = 40, 56               # term chips: height, pitch
CH_Y = LN_Y + 24
ROWS = 3
DOTS_Y = CH_Y + ROWS * CHIP_P + 8     # vertical ellipsis: more terms follow
P_Y, P_H = DOTS_Y + 56, 56            # partials
LN_B = P_Y + P_H + 16
NODE_W, NODE_H = 56, 48
LVL_Y = [None] + [LN_B + 64 + (k - 1) * 112 for k in range(1, 5)]   # node tops, levels 1..4
TOP_B = LVL_Y[4] + NODE_H + 88
GA_Y = TOP_B + 64                     # gather panel
G1_Y, G1_H = GA_Y + 48, 120
SPM_Y, SPM_H = G1_Y + G1_H + 80, 88
GA_B = SPM_Y + SPM_H + 32
FOOT_Y = GA_B + 48
H = FOOT_Y + 48

f = Fig("tree", W, H,
        title="Deterministic row summation",
        desc="Block term i of a row (Q8_0: one per 32 weights, Q4_K: one per 256) is added to "
             "partial p_(i mod 16); the 16 partials start at +0.0 for every row. The row result "
             "y is the fixed tree16 sum: level 1 adds p0+p1 ... p14+p15, then 4, 2 and 1 fadds, "
             "pairs taken from the left. The gather writes y_out = y, or with the MVM's ACC flag "
             "y_out = y_old + y, to lane n mod 16 of SPM line dst + t*stride + n/16.")


def join(fig, xa, xb, y0, xm, y1):
    """Bracket: inputs at (xa, y0) and (xb, y0) meet at mid height, then drop into the fadd
    node whose top center is (xm, y1)."""
    ym = (y0 + y1) / 2
    fig.raw(f'<polyline points="{fmt(xa)},{fmt(y0)} {fmt(xa)},{fmt(ym)} {fmt(xb)},{fmt(ym)} '
            f'{fmt(xb)},{fmt(y0)}" fill="none" stroke="{INK}" stroke-width="2" '
            f'stroke-linejoin="miter"/>')
    fig.line([(xm, ym), (xm, y1)])


# ---- tsr_mvm_back panel ----------------------------------------------------------------------
f.panel(PX0, TOP_Y, PX1 - PX0, TOP_B - TOP_Y, label="TSR_MVM_BACK · ONE ROW, ONE TOKEN")

f.box(256, HD_Y, 1088, HD_H, "compute", "BLOCK TERM i OF THE ROW",
      sub="Q8_0:  t = (d·Dx) · f32(isum)                       one per 32 weights\n"
          "Q4_K:  t = (d·Dx) · f32(isum) − (dmin·Dx) · f32(msum)   one per 256 weights")

# lanes
for j in range(16):
    f.band(LANE[j], LN_Y, LANE_W, LN_B - LN_Y, fill=BAND)
f.arrow([(800, HD_Y + HD_H - 12), (800, LN_Y + 8)])
f.text(800 + 44, (HD_Y + HD_H + LN_Y) / 2, "TERMS IN ORDER OF i", "callout",
       valign="middle")

# term chips: term i in lane i mod 16
for r in range(ROWS):
    for j in range(16):
        f.box(LANE[j], CH_Y + r * CHIP_P, LANE_W, CHIP_H, "neutral", sub=f"t{16 * r + j}",
              cut=8)
for j in range(16):
    for d in (-9, 0, 9):                                   # vertical ellipsis: more terms
        f.dot(LC[j], DOTS_Y + d, r=2.5, color=INK_SOFT)
    f.arrow([(LC[j], DOTS_Y + 20), (LC[j], P_Y + 6)], w=16, hw=36, hl=16)
    f.box(LANE[j], P_Y, LANE_W, P_H, "compute", f"p{j}", label_role="label_s", cut=8)

# tree16: level k joins pairs of level k-1 (partials for k = 1)
for k in range(1, 5):
    y0 = (P_Y + P_H) if k == 1 else LVL_Y[k - 1] + NODE_H
    for n, xm in enumerate(LV[k]):
        join(f, LV[k - 1][2 * n], LV[k - 1][2 * n + 1], y0, xm, LVL_Y[k])
    for xm in LV[k]:
        f.box(xm - NODE_W / 2, LVL_Y[k], NODE_W, NODE_H, "compute", "+", cut=8)

# ---- gather panel --------------------------------------------------------------------------------
f.panel(PX0, GA_Y, PX1 - PX0, GA_B - GA_Y, label="TSR_MVM_GATHER")
y_root = LVL_Y[4] + NODE_H
fl_w, side_w = 240, 304
f.arrow([(800, y_root - 8), (800, G1_Y + 10)])
f.text(800 + 44, (TOP_B + GA_Y) / 2, "ROW RESULT y", "callout", valign="middle")
f.box(800 - fl_w / 2, G1_Y, fl_w, G1_H, "control", "ACC FLAG", sub="of the MVM")
pl_x, ac_x = 256, 1344 - side_w
f.box(pl_x, G1_Y, side_w, G1_H, "io", "PLAIN", sub="y_out = y")
f.box(ac_x, G1_Y, side_w, G1_H, "compute", "ACC", sub="y_out = y_old + y\none fadd")
g1m = G1_Y + G1_H / 2
f.arrow([(800 - fl_w / 2 + 12, g1m), (pl_x + side_w - 10, g1m)])
f.arrow([(800 + fl_w / 2 - 12, g1m), (ac_x + 10, g1m)])
f.text((800 - fl_w / 2 + pl_x + side_w) / 2, g1m - 36, "CLEAR", "callout", anchor="middle",
       valign="middle")
f.text((800 + fl_w / 2 + ac_x) / 2, g1m - 36, "SET", "callout", anchor="middle", valign="middle")
f.box(256, SPM_Y, 1088, SPM_H, "memory", "TSR_SPM",
      sub="row n, token t: line dst + t·stride + n/16, lane n mod 16")
f.arrow([(pl_x + side_w / 2, G1_Y + G1_H - 8), (pl_x + side_w / 2, SPM_Y + 10)])
f.arrow([(ac_x + side_w / 2, G1_Y + G1_H - 8), (ac_x + side_w / 2, SPM_Y + 10)], tail=True)

# ---- callouts --------------------------------------------------------------------------------------
f.callout("NO FMA: EVERY\nPRODUCT AND\nSUM IS ROUNDED\nSEPARATELY", (RX, HD_Y + HD_H / 2),
          target=(1344 - 24, HD_Y + HD_H / 2))
gy = CH_Y + CHIP_P + CHIP_H / 2
f.callout("TERM i IS\nADDED TO\nPARTIAL\ni mod 16", (LX, gy), target=(LANE[0], gy))
f.callout("ALL 16 START\nAT +0.0 FOR\nEVERY ROW", (LX, P_Y + P_H / 2),
          target=(LANE[0], P_Y + P_H / 2))
f.callout("16 IS AN\nARCHITECTURAL\nCONSTANT: TERMS\nOF ONE PARTIAL\nARE ≥ 8 CYCLES\nAPART",
          (RX, P_Y + P_H / 2), target=(LANE[15] + LANE_W, P_Y + P_H / 2))
for k, txt in [(1, "8 FADDS"), (2, "4 FADDS"), (3, "2 FADDS"), (4, "1 FADD")]:
    yy = LVL_Y[k] + NODE_H / 2
    f.callout(txt, (LX, yy), target=(LV[k][0] - NODE_W / 2, yy))
f.callout("FIXED PAIR\nORDER, CHECKED\nBIT-EXACT: A\nSEQUENTIAL SUM\nIS WRONG ON\n67–87 % OF\n"
          "ELEMENTS", (RX, LVL_Y[2] + NODE_H / 2),
          target=(LV[2][3] + NODE_W / 2, LVL_Y[2] + NODE_H / 2))

# ---- footnote: the tree as a formula ---------------------------------------------------------------
f.text(W / 2, FOOT_Y, "y = (((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)))+"
       "(((p8+p9)+(p10+p11))+((p12+p13)+(p14+p15)))", "note", anchor="middle", color=INK_SOFT)

f.save()

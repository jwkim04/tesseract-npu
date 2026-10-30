"""Figure `hero`: the Tesseract v2 NPU core as a stylised die, with one MVM cluster zoomed.

The picture for the top of the README, in the manner of Tenstorrent's chip-and-tile art: tsr_core
drawn as a die, the four DDR4 channels off-die, and one MVM cluster opened up by dashed zoom
lines. It is a stylised view, not a floorplan, and the figure says so.

Status drawing rule (STYLE.md "Planned and unconfirmed things"): solid blocks exist in rtl/ and ran
on the card with the v1 core (tag v1-m4); the v2 additions exist in the simulator and the firmware
only, so they are dashed and pale with a PLANNED · M6 tag, or, where a solid v1 block gains a mode,
carry a dashed "+ ..." tag (DMA + KVST, SPM + even / odd banks, MAC lanes + ATTQK / ATTPV).

Die content, with sources:
- four MVM clusters of 128 MACs (NUM_CH = 4, LANES = 128), cluster c next to DDR4 channel c and
  fed only by it, through one tsr_mem_arb per channel (docs/spec.md §2, §3; docs/rtl-m2.md §2);
- SPM, 4 MiB in 64 B lines, in the centre, with the x broadcast / y gather of tsr_mvm between it
  and the clusters (docs/rtl-m3.md §2, §3); v2: even / odd line-interleaved banks, 2R2W
  (docs/v2/04-control-core.md §10.1, V2-CC-12), planned;
- tsr_csr on the die edge: the core boundary is a CSR port; v2 adds the launch queue LQ, the
  completion ring LCQ, the log ring and the firmware loader to it (docs/v2/01-execution-model.md
  §4, §5, §3.3), planned; the platform adapter maps the port to s_axi_control of the XRT shell;
- planned: the control core (Ibex RV32IMC, ITCM 128 KiB, DTCM 64 KiB; docs/v2/04-control-core.md
  §2.1, §3, V2-CC-3), the issue path (checker tsr_chk, dispatcher, one in-order queue per unit;
  CC §7, §8), the debug module (riscv-dbg, GDB over the DMI window; docs/v2/05-debug.md §3), the
  vector unit (VL 16: norms, RoPE, activations, reductions, softmax parts; docs/v2/
  02-vector-unit.md §1, §7), KVST in the DMA (docs/v2/03-attention-kv.md §5.1, §8.3);
- DMA (LD, ST) and QACT (f32 -> QA) are solid (rtl/core/tsr_dma.sv, tsr_qact.sv).
Zoomed cluster (docs/rtl-m3.md §2, §7; docs/spec.md §8): weight streamer tsr_mvm_ws (64-beat
bursts, 512-beat FIFO), parse + unpack (tsr_mvm_front: chunk parser, meta, Q4_K scales, int8
lanes), activation buffer tsr_mvm_actbuf (QS, D, BS per token, filled by the x broadcast), 128 MAC
lanes (4 x tsr_mac_lane, int8 x int8), integer scale + fp term + accumulate (16 partials per row),
tree16 (8 + 4 + 2 + 1 fadds), y gather back to the SPM. Weight beats: 64 B per cycle. Planned on
the MAC lanes: the ATTQK and ATTPV modes (docs/v2/03-attention-kv.md §8.1, §8.2).
"FP ACCUMULATE" stands for the stages integer scale, fp term, accumulator of docs/rtl-m3.md §7.
Left out on purpose: the platform adapter and the XRT shell (the system figure shows them) and
the paths between the units and the arbiters. The footnote says so. The die places clusters 1 and
2 at opposite corners for symmetry. The U200 SLR placement (floorplan C') is in the system figure;
the die makes no SLR claim. The upper clusters start lower than the die's top edge so that all four
clusters have the same height around the SPM collar.

Run: build/tools/svgvenv/bin/python docs/fig/src/hero.py        (TSR_FIG_THEME=dark: hero-dark.svg)
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import DASH, INK, INK_SOFT, Fig, chamfer, fmt, logo_box, wrap  # noqa: E402

W = 1600


# ---- local helpers (not in tsrfig) ---------------------------------------------------------------
def outline(f, x, y, w, h, cut=18, sw=4):
    """Heavy ink outline around a block (the zoomed tile in ref2)."""
    pts = " ".join(f"{fmt(px)},{fmt(py)}" for px, py in chamfer(x, y, w, h, cut))
    f.raw(f'<polygon points="{pts}" fill="none" stroke="{INK}" stroke-width="{sw}" '
          f'stroke-linejoin="miter"/>')


def zoom_line(f, a, b, sw=2):
    """Dashed straight line from the zoomed tile to the zoom panel (may be diagonal)."""
    f.raw(f'<line x1="{fmt(a[0])}" y1="{fmt(a[1])}" x2="{fmt(b[0])}" y2="{fmt(b[1])}" '
          f'stroke="{INK}" stroke-width="{sw}" stroke-dasharray="{DASH}"/>')


# ---- geometry --------------------------------------------------------------------------------------
# die (x)
DIE_X0, DIE_X1 = 72, 1008
CLW = 152
CL_L, CL_R = 104, DIE_X1 - 32 - CLW                  # cluster columns: 104..256, 824..976
CEN_X0, CEN_X1 = CL_L + CLW + 48, CL_R - 48          # centre column: 304..776
COL_X0, COL_X1 = CL_L + CLW + 80, CL_R - 80          # x broadcast / y gather collar: 336..744
EDGE_W, EDGE_H = 136, 48                             # tiles on the die edge (arbiters, CSR)
# die (y)
DDR_T_Y, DDR_H = 64, 80
DIE_Y0 = 216
Y0 = 296                                              # interior top
A1_H = 152                                            # row 1: control core, issue path
A2_Y, A2_H = Y0 + A1_H + 56, 112                      # row 2: debug, vector unit (504..616)
COL_Y0, COL_H = A2_Y + A2_H + 48, 240                 # collar 664..904
COL_Y1 = COL_Y0 + COL_H
R2_Y, R2_H = COL_Y1 + 48, 112                         # DMA, QACT (952..1064)
Y1 = R2_Y + R2_H                                      # interior bottom 1064
MID = (COL_Y0 + COL_Y1) // 2                          # 784: the gap between the cluster rows
CL_H = Y1 - (MID + 24)                                # 256: all four clusters
DIE_Y1 = Y1 + 56
DDR_B_Y = DIE_Y1 + 64
# zoom panel
ZX0, ZX1 = 1064, 1544
ZY0, ZY1 = DIE_Y0, DIE_Y1
LEG_Y = DDR_B_Y + DDR_H + 64
H = LEG_Y + 144

f = Fig("hero", W, H,
        title="The Tesseract NPU core",
        desc="Stylised die of tsr_core: four MVM clusters of 128 MACs at the corners, each next "
             "to its own DDR4 channel and memory arbiter; the 4 MiB SPM in the centre with the x "
             "broadcast / y gather to the clusters; DMA, QACT and the CSR block on the v1 core; "
             "the v2 additions drawn dashed: control core (Ibex RV32IMC, ITCM and DTCM), debug "
             "module, issue path with unit queues, vector unit, KVST in the DMA, even and odd SPM "
             "banks. The host reaches the CSR port over PCIe and XRT with a launch queue and a "
             "completion ring. One cluster is opened up: weight streamer, unpack, activation "
             "buffer, 128 MAC lanes with planned attention modes, fp accumulate and tree16.")

# ---- the die, DDR4 channels and weight streams ---------------------------------------------------------
f.panel(DIE_X0, DIE_Y0, DIE_X1 - DIE_X0, DIE_Y1 - DIE_Y0)
cl_x = {0: CL_L, 1: CL_R, 2: CL_R, 3: CL_L}
cl_y = {0: MID - 24 - CL_H, 1: MID - 24 - CL_H, 2: MID + 24, 3: MID + 24}
top = {0: True, 1: True, 2: False, 3: False}
zc = 1                                                # the cluster opened up on the right
hx, hy, hw, hh = cl_x[zc] - 8, cl_y[zc] - 8, CLW + 16, CL_H + 16
for c in range(4):
    x = cl_x[c]
    f.box(x, DDR_T_Y if top[c] else DDR_B_Y, CLW, DDR_H, "memory", f"DDR4 CH{c}", cut=12,
          label_role="label_s")
    f.box(x, cl_y[c], CLW, CL_H, "compute", f"CLUSTER\n{c}", sub="128 MACs")
outline(f, hx, hy, hw, hh)                            # under the arrows, like ref2's marked tile
for c in range(4):
    x, cx = cl_x[c], cl_x[c] + CLW / 2
    # weight stream: DDR4 ch c -> tsr_mem_arb c -> cluster c (the arbiter is drawn over it)
    if top[c]:
        f.arrow([(cx, DDR_T_Y + DDR_H - 8), (cx, cl_y[c] + 12)], fat=True)
    else:
        f.arrow([(cx, DDR_B_Y + 8), (cx, cl_y[c] + CL_H - 12)], fat=True)
    ey = DIE_Y0 if top[c] else DIE_Y1
    f.box(cx - EDGE_W / 2, ey - EDGE_H / 2, EDGE_W, EDGE_H, "io", "MEM ARB",
          label_role="label_s", cut=10)

# centre column, row 1: control core -> issue path (planned)
cc_w, is_w = 216, 200
cc_x, is_x = CEN_X0, CEN_X1 - is_w
f.box(cc_x, Y0, cc_w, A1_H, "control", "CONTROL CORE",
      sub="Ibex RV32IMC\nITCM 128 KiB\nDTCM 64 KiB", planned=True, tag="PLANNED",
      label_role="label_s")
f.box(is_x, Y0, is_w, A1_H, "control", "ISSUE PATH", sub="checker\nunit queues", planned=True,
      tag="PLANNED", label_role="label_s")
f.arrow([(cc_x + cc_w - 12, Y0 + A1_H / 2), (is_x + 12, Y0 + A1_H / 2)])
# row 2: debug module under the control core, vector unit under the issue path (planned)
f.box(cc_x, A2_Y, cc_w, A2_H, "control", "DEBUG", sub="debug module\nGDB over PCIe",
      planned=True, tag="PLANNED", label_role="label_s")
f.box(is_x, A2_Y, is_w, A2_H, "compute", "VECTOR UNIT", sub="VL 16 · norms\nRoPE · softmax",
      planned=True, tag="PLANNED", label_role="label_s")
# arrows meet the top edges 48 right of the block's left edge, clear of the right-hand PLANNED tags
cc_c, is_c = cc_x + 48, is_x + 48
f.arrow([(cc_c, Y0 + A1_H - 8), (cc_c, A2_Y + 12)], tail=True)       # control core <-> debug
f.arrow([(is_c, Y0 + A1_H - 8), (is_c, A2_Y + 12)], dashed=True)     # tile ops to the unit
f.text(is_c - 40, Y0 + A1_H + 28, "TILE OPS", "callout", anchor="end", valign="middle")

# host interface: the CSR port, through the platform adapter and XRT to the host
csr_cx = cc_c
f.arrow([(csr_cx, 104), (csr_cx, DIE_Y0 - 12)], tail=True)
f.arrow([(csr_cx, DIE_Y0 + EDGE_H / 2 - 8), (csr_cx, Y0 + 12)], tail=True)   # LQ, LCQ
f.box(csr_cx - EDGE_W / 2, DIE_Y0 - EDGE_H / 2, EDGE_W, EDGE_H, "control", "CSR",
      label_role="label_s", cut=10)
f.text(csr_cx - 40, 76, "HOST · PCIe · XRT", "callout", valign="middle")
f.text(csr_cx + 40, 150, "CSR PORT ·\nLAUNCH QUEUE ·\nCOMPLETION RING", "callout", valign="middle")
f.tag(csr_cx + EDGE_W / 2 + 32, DIE_Y0, "TSR_CORE", anchor="start")

# SPM inside the x broadcast / y gather collar
f.box(COL_X0, COL_Y0, COL_X1 - COL_X0, COL_Y1 - COL_Y0, "io", "")
f.box(COL_X0 + 24, COL_Y0 + 32, COL_X1 - COL_X0 - 48, COL_Y1 - COL_Y0 - 104, "memory", "SPM",
      sub="4 MiB · 64 B lines", tag="+ EVEN / ODD BANKS", tag_style="planned")
f.text((COL_X0 + COL_X1) / 2, COL_Y1 - 36, "X BROADCAST · Y GATHER", "callout",
       anchor="middle", valign="middle")
up_y, dn_y = COL_Y0 + 48, COL_Y1 - 72
for y in (up_y, dn_y):
    f.arrow([(CL_L + CLW - 12, y), (COL_X0 + 12, y)], tail=True)
    f.arrow([(CL_R + 12, y), (COL_X1 - 12, y)], tail=True)

# units: the vector unit, DMA and QACT use the SPM
f.arrow([(is_c, A2_Y + A2_H - 8), (is_c, COL_Y0 + 12)], tail=True)
half = (CEN_X1 - CEN_X0 - 48) / 2
dma_x, qa_x = CEN_X0, CEN_X1 - half
f.box(dma_x, R2_Y, half, R2_H, "io", "DMA", sub="LD · ST", tag="+ KVST", tag_style="planned")
f.box(qa_x, R2_Y, half, R2_H, "compute", "QACT", sub="f32 to QA")
for x in (dma_x + 48, qa_x + 48):                      # left of the "+ KVST" tag
    f.arrow([(x, COL_Y1 - 8), (x, R2_Y + 12)], tail=True)

# ---- zoom: cluster 1 ---------------------------------------------------------------------------------
zoom_line(f, (hx + hw, hy + 18), (ZX0, ZY0 + 20))
zoom_line(f, (hx + hw, hy + hh - 18), (ZX0, ZY1 - 20))

ZL0, ZR1 = ZX0 + 24, ZX1 - 24                         # 1088..1520
ZL1 = ZL0 + 188                                       # left column: weight path, accumulate
ZR0 = ZL1 + 56                                        # right column: activation buffer, tree16
zcw, zrw = ZL1 - ZL0, ZR1 - ZR0
lcx, rcx = (ZL0 + ZL1) / 2, (ZR0 + ZR1) / 2
GAP = 104
WS_Y, WS_H = ZY0 + 32, 128
UN_Y, UN_H = WS_Y + WS_H + GAP, 128
MAC_Y, MAC_H = UN_Y + UN_H + GAP, 112
ACC_Y = MAC_Y + MAC_H + GAP
ACC_H = ZY1 - 40 - ACC_Y
AB_Y0, AB_Y1 = WS_Y, UN_Y + UN_H

# external connections of the cluster (drawn before the panel: they cross its edge)
f.arrow([(lcx, 112), (lcx, WS_Y + 12)], fat=True)
f.arrow([(rcx, 112), (rcx, AB_Y0 + 12)])
f.text(lcx, 80, "DDR4 CH1\n64 B PER CYCLE", "callout", anchor="middle", valign="middle")
f.text(rcx, 80, "X BROADCAST\nFROM SPM", "callout", anchor="middle", valign="middle")
f.arrow([(rcx, ACC_Y + ACC_H - 12), (rcx, ZY1 + 72)])
f.text(rcx, ZY1 + 104, "Y GATHER TO SPM", "callout", anchor="middle", valign="middle")

f.panel(ZX0, ZY0, ZX1 - ZX0, ZY1 - ZY0, label="CLUSTER 1", label_at="middle")
f.box(ZL0, WS_Y, zcw, WS_H, "io", "WEIGHT\nSTREAMER", sub="512-beat FIFO",
      label_role="label_s")
f.box(ZL0, UN_Y, zcw, UN_H, "control", "UNPACK", sub="chunk parse,\nto int8 lanes",
      label_role="label_s")
f.box(ZR0, AB_Y0, zrw, AB_Y1 - AB_Y0, "memory", "ACTIVATION\nBUFFER", sub="QS, D, BS\nper token",
      label_role="label_s")
f.box(ZL0, MAC_Y, ZR1 - ZL0, MAC_H, "compute", "128 MAC LANES",
      sub="4 × tsr_mac_lane · int8 × int8")
f.tag(ZR1 - 22, MAC_Y + MAC_H, "+ ATTQK · ATTPV", anchor="end", style="planned")
f.box(ZL0, ACC_Y, zcw, ACC_H, "compute", "FP\nACCUMULATE", sub="16 partials\nper row",
      label_role="label_s")
f.box(ZR0, ACC_Y, zrw, ACC_H, "compute", "TREE16", sub="8 + 4 + 2 + 1\nfadds",
      label_role="label_s")
f.arrow([(lcx, WS_Y + WS_H - 6), (lcx, UN_Y + 12)], fat=True)    # start 6 inside: clear of text
f.arrow([(lcx, UN_Y + UN_H - 6), (lcx, MAC_Y + 12)], fat=True)
f.arrow([(rcx, AB_Y1 - 12), (rcx, MAC_Y + 12)])
f.arrow([(lcx, MAC_Y + MAC_H - 6), (lcx, ACC_Y + 12)])
acc_cy = ACC_Y + ACC_H / 2
f.arrow([(ZL1 - 12, acc_cy), (ZR0 + 8, acc_cy)])      # tip 8 inside: clear of "8 + 4 + 2 + 1"

# ---- legend and footnote -----------------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                        ("io", "DATA MOVEMENT"), ("neutral", "PLANNED", True)])
_lg = {"key": "tesseract-core", "h": 40, "backing": True, "pad": 6}
_lw, _lh = logo_box(**_lg)
f.logo(x=W - 72 - _lw, y=LEG_Y - _lh + 6, **_lg)             # name of the configuration, bottom right
f.text(W / 2, LEG_Y + 38, wrap(
    "Stylised view of tsr_core, not a floorplan. Solid blocks: the v1 core (run on the card). "
    "Dashed blocks and tags: the v2 additions (milestone M6), verified in the simulator and in "
    "firmware; their RTL is planned (W4). The issue path dispatches tile "
    "ops to the unit queues of the DMA, QACT, MVM and VU. Cluster c streams weights only from "
    "DDR4 channel c. The platform adapter and the XRT shell are left out.", 1440),
    "note", anchor="middle", color=INK_SOFT)

f.save(scale=2)          # preview PNG at 2x for checking; _gh.png at 880 px

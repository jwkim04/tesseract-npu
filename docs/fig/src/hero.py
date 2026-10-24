"""Figure `hero`: the Tesseract NPU core as a stylised die, with one MVM cluster zoomed.

A picture for the top of README "Architecture", in the manner of Tenstorrent's chip-and-tile art:
tsr_core drawn as a die, the four DDR4 channels off-die, and one MVM cluster opened up by dashed
zoom lines. It is a stylised view, not a floorplan, and the figure says so.

Die content (README "System on the Alveo U200", "Inside tsr_core", "Core boundary"):
- four MVM clusters of 128 MACs (NUM_CH = 4, LANES = 128), cluster c next to DDR4 channel c and
  fed only by it, through tsr_mem_arb channel c (one arbiter per channel, every memory-port
  output registered there);
- SPM, 4 MiB in 64 B lines, in the centre; the x broadcast / y gather of tsr_mvm between the SPM
  and the clusters (README system Mermaid: SPM <-> x broadcast / y gather <-> clusters 0..3);
- command processor (fetch, decode, R0-R15), CSR block (tsr_csr), DMA (tsr_dma: LD, ST), QACT
  (tsr_qact: f32 -> QA);
- vector + scalar unit, planned for M6 (README milestone table);
- host interface: the core boundary is a CSR port and `done` (plus the memory ports); the
  platform adapter maps them to s_axi_control / ap_done of the XRT shell, which the host
  reaches over PCIe (README system Mermaid and bullets).
Zoomed cluster (README "MVM unit": Mermaid, stage table; docs/rtl-m3.md §2, §7; spec "MVM
클러스터 마이크로아키텍처"): weight streamer tsr_mvm_ws (64-beat bursts, 512-beat FIFO), parse +
unpack (tsr_mvm_front: chunk parser, meta, Q4_K scales, int8 lanes), activation buffer
tsr_mvm_actbuf (QS, D, BS per token, filled by the x broadcast), 128 MAC lanes (4 x tsr_mac_lane,
int8 x int8), integer scale + fp term + accumulate (16 partials per row), tree16 (8 + 4 + 2 + 1
fadds), y gather back to the SPM. Weight beats: 64 B per cycle (README design principles).
"FP ACCUMULATE" stands for spec stages 7-9 (integer scale, fp term, accumulator), as the spec's
own one-line summary of a cluster does ("스트림·언팩, MAC 128, fp 누산").
The platform adapter (control bridge, AXI4 bridges) and the XRT shell are left out on purpose;
the footnote says so and points to the system figure. The DMA <-> arbiter and fetch paths are not
drawn either; the footnote names them (README tsr_mem_arb row: "ch0 sources: FETCH, DMA, WS0 · ch c
sources: DMA, WS c"). The weight streamer is lavender (data movement), as the stream requester in
the cluster figure. The die places clusters 1 and 2 at opposite corners for symmetry. The U200 SLR
placement (floorplan C': clusters 0 and 2 in SLR0, 1 and 3 in SLR2, the rest of the core in SLR1;
hw/u200/k4_floorplan.tcl, docs/rtl-m3.md §14.5) is in the system figure, and the footnote points
there; the die makes no SLR claim.

Run: build/tools/svgvenv/bin/python docs/fig/src/hero.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import DASH, INK, INK_SOFT, Fig, chamfer, fmt  # noqa: E402

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
CLW = 176
CL_L, CL_R = 104, DIE_X1 - 32 - CLW                  # cluster columns: 104..280, 800..976
CEN_X0, CEN_X1 = CL_L + CLW + 48, CL_R - 48          # centre rows r0, r2: 328..752
COL_X0, COL_X1 = CL_L + CLW + 80, CL_R - 80          # x broadcast / y gather collar: 360..720
EDGE_W, EDGE_H = 136, 48                             # tiles on the die edge (arbiters, CSR)
# die (y)
DDR_T_Y, DDR_H = 64, 80
DIE_Y0 = 216
Y0 = 280                                              # interior top
R0_H = 112                                            # command processor, vector unit
COL_Y0, COL_Y1 = Y0 + R0_H + 48, Y0 + R0_H + 48 + 256  # collar 440..696
R2_Y, R2_H = COL_Y1 + 48, 112                         # DMA, QACT
Y1 = R2_Y + R2_H                                      # interior bottom 840
CL_H = (Y1 - Y0 - 48) / 2
DIE_Y1 = Y1 + 56
DDR_B_Y = DIE_Y1 + 64
# zoom panel
ZX0, ZX1 = 1064, 1544
ZY0, ZY1 = DIE_Y0, DIE_Y1
LEG_Y = DDR_B_Y + DDR_H + 64
H = LEG_Y + 136

f = Fig("hero", W, H,
        title="The Tesseract NPU core",
        desc="Stylised die of tsr_core: four MVM clusters of 128 MACs at the corners, each next "
             "to its own DDR4 channel and memory arbiter; the 4 MiB SPM in the centre with the x "
             "broadcast / y gather to the clusters; command processor, CSR, DMA, QACT and a "
             "planned vector + scalar unit (M6); the host reaches the CSR port over PCIe and XRT. "
             "One cluster is opened up: weight streamer, parse and unpack, activation buffer, 128 "
             "MAC lanes, fp accumulate and tree16.")

# ---- the die, DDR4 channels and weight streams ---------------------------------------------------------
f.panel(DIE_X0, DIE_Y0, DIE_X1 - DIE_X0, DIE_Y1 - DIE_Y0)
cl_x = {0: CL_L, 1: CL_R, 2: CL_R, 3: CL_L}
cl_y = {0: Y0, 1: Y0, 2: Y0 + CL_H + 48, 3: Y0 + CL_H + 48}
top = {0: True, 1: True, 2: False, 3: False}
zc = 1                                                # the cluster opened up on the right
hx, hy, hw, hh = cl_x[zc] - 8, cl_y[zc] - 8, CLW + 16, CL_H + 16
for c in range(4):
    x = cl_x[c]
    f.box(x, DDR_T_Y if top[c] else DDR_B_Y, CLW, DDR_H, "memory", f"DDR4 CH{c}", cut=12)
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

# command processor, planned vector + scalar unit
cp_w = 200
cp_x = CEN_X0
vu_x = cp_x + cp_w + 48
f.box(cp_x, Y0, cp_w, R0_H, "control", "COMMAND\nPROCESSOR", sub="R0–R15")
f.box(vu_x, Y0, CEN_X1 - vu_x, R0_H, "compute", "VECTOR +\nSCALAR UNIT", planned=True,
      tag="PLANNED · M6", label_role="label_s")

# host interface: CSR port and done, through the platform adapter and XRT to the host
csr_cx = cp_x + cp_w / 2
f.arrow([(csr_cx, 104), (csr_cx, DIE_Y0 - 12)], tail=True)
f.box(csr_cx - EDGE_W / 2, DIE_Y0 - EDGE_H / 2, EDGE_W, EDGE_H, "control", "CSR",
      label_role="label_s", cut=10)
f.text(csr_cx, 76, "HOST · PCIe · XRT", "callout", anchor="middle", valign="middle")
f.text(csr_cx + 40, 156, "CSR PORT · DONE", "callout", valign="middle")
f.tag(csr_cx + EDGE_W / 2 + 32, DIE_Y0, "TSR_CORE", anchor="start")

# SPM inside the x broadcast / y gather collar
f.box(COL_X0, COL_Y0, COL_X1 - COL_X0, COL_Y1 - COL_Y0, "io", "")
f.box(COL_X0 + 24, COL_Y0 + 24, COL_X1 - COL_X0 - 48, COL_Y1 - COL_Y0 - 96, "memory", "SPM",
      sub="4 MiB · 64 B lines")
f.text((COL_X0 + COL_X1) / 2, COL_Y1 - 36, "X BROADCAST · Y GATHER", "callout",
       anchor="middle", valign="middle")
up_y, dn_y = COL_Y0 + 40, COL_Y1 - 64
for y in (up_y, dn_y):
    f.arrow([(CL_L + CLW - 12, y), (COL_X0 + 12, y)], tail=True)
    f.arrow([(CL_R + 12, y), (COL_X1 - 12, y)], tail=True)

# DMA, QACT
half = (CEN_X1 - CEN_X0 - 48) / 2
f.box(CEN_X0, R2_Y, half, R2_H, "io", "DMA", sub="LD, ST")
f.box(CEN_X1 - half, R2_Y, half, R2_H, "compute", "QACT", sub="f32 to QA")

# ---- zoom: cluster 1 ---------------------------------------------------------------------------------
zoom_line(f, (hx + hw, hy + 18), (ZX0, ZY0 + 20))
zoom_line(f, (hx + hw, hy + hh - 18), (ZX0, ZY1 - 20))

ZL0, ZR1 = ZX0 + 24, ZX1 - 24                         # 1088..1520
ZL1 = ZL0 + 188                                       # left column: weight path, accumulate
ZR0 = ZL1 + 56                                        # right column: activation buffer, tree16
zcw, zrw = ZL1 - ZL0, ZR1 - ZR0
lcx, rcx = (ZL0 + ZL1) / 2, (ZR0 + ZR1) / 2
WS_Y, WS_H = ZY0 + 32, 112
UN_Y, UN_H = WS_Y + WS_H + 56, 112
MAC_Y, MAC_H = UN_Y + UN_H + 56, 96
ACC_Y = MAC_Y + MAC_H + 56
ACC_H = ZY1 - 32 - ACC_Y
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
f.text(W / 2, LEG_Y + 38, "Stylised view of tsr_core, not a floorplan. Cluster c streams weights "
       "only from DDR4 channel c.\nEach MEM ARB also carries the "
       "DMA's LD/ST, and ch0's carries instruction fetch. The platform adapter and the XRT shell\n"
       "are left out; the system figure shows them and the U200 SLR placement.", "note",
       anchor="middle", color=INK_SOFT)

f.save(scale=2)          # preview PNG at 2x for checking; _gh.png at 880 px

"""Figure `clocks`: the clock domains of Tesseract Core and every clock-domain crossing between them.

Drawn the way clock-domain crossings are usually shown: colour is the clock domain, each domain is one
tinted region with the blocks that share its clock, and every crossing is a synchroniser cell on the
boundary between the two regions.

Content follows docs/design/04-control-core.md §2.1-§2.2 (block diagram, domains), §11.1 (clocks),
§11.2 (crossings and widths), §11.3 (constraints), §12.2 (reset synchronisers); the RTL in
rtl/ctrl (`tsr_ctrl_cdc.sv` header and instances, `tsr_ctrl.sv` for the reset synchronisers,
`tsr_ctrl_rv.sv` for the sources and sinks on the clk_rv side, `tsr_ctrl_pkg.sv` for the widths,
`cdc/tsr_cdc_fifo.sv`, `cdc/tsr_cdc_gray.sv`, `cdc/tsr_cdc_hs.sv`, `cdc/tsr_sync.sv`,
`cdc/tsr_rst_sync.sv`, `viv/tsr_ctrl_cdc.xdc` for the constraints), `rtl/core/tsr_core.sv` and
`rtl/core/tsr_core_dp.sv` (the core-domain endpoints of every crossing: tsr_issue, tsr_csr,
tsr_spm_bridge), `rtl/platform/alveo_xrt/tsr_xrt_kernel.sv` (the adapter runs on ap_clk) and
docs/platform-u200.md §1 and §8.3 (ap_clk, ap_clk_2):
- XRT shell: one vendor box at the top. PCIe and the DDR4 controllers run on their own clocks, which the
  shell converts; it provides the kernel clocks ap_clk and ap_clk_2 (and their resets) and carries the
  CSR port (s_axi_control) and the memory ports into the core clock domain. Nothing of its inside is
  drawn;
- core clock domain (`ap_clk`, 200 MHz in the release image): the platform adapter (control bridge
  and AXI4 bridges, `tsr_xrt_kernel`), `tsr_csr`, `tsr_issue` with `tsr_chk` and the unit queues,
  `tsr_spm_bridge`, DMA, QACT, the MVM clusters, `tsr_spm` and one `tsr_mem_arb` per channel;
- control core domain (`clk_rv` from `ap_clk_2`, 100 MHz in the release image, CC §11.1): Ibex, the debug
  module (riscv-dbg), ITCM, DTCM, the RISC-V bus `tsr_rv_bus` and the control window `tsr_ctl_win`
  (all in `tsr_ctrl_rv`);
- crossings (`tsr_ctrl_cdc`), one pipe and one cell per bundle, the cell on the boundary. The cell is the
  primitive of the instance (async FIFO `tsr_cdc_fifo`, gray counter `tsr_cdc_gray`, handshake
  `tsr_cdc_hs`, 2-flop `tsr_sync`); the pipe changes colour at the cell, and a head marks the side that
  receives (two heads: request one way, response data back). Widths are the ones of `tsr_ctrl_pkg.sv` and
  the instances (a handshake's `a / b` is request / response). Sources and sinks are the modules that
  `tsr_ctrl_rv` and `tsr_core_dp` connect: control window to tsr_issue (issue FIFO), to tsr_csr (log
  FIFO, counters, RT mirrors, LQ / LCQ, sticky events and RV state) and to tsr_spm_bridge (bridge and
  core window, the core-window read is tsr_csr's); tsr_issue to the control window (S, D, error
  records); tsr_csr to the control window (MTIME, levels), to the TCM load port through the bus (LOAD_*)
  and to the debug module (DMI). Levels and RV state are attached to the control window, the nearest
  certain block of `tsr_ctrl_rv`;
- resets: ap_rst_n reaches the core domain registered (`tsr_xrt_reset`, inside the adapter); the clk_rv
  domain gets its reset from the core reset through `tsr_rst_sync` (2 stages, asynchronous assert,
  synchronous release; `u_rst_rv`) and ap_rst_n_2 through a second `tsr_rst_sync` (`u_rst_ext`, into the
  Ibex and debug-module resets only);
- constraints (legend): the three rules of `tsr_ctrl_cdc.xdc` (ASYNC_REG, `set_max_delay -datapath_only`
  = the smaller period, `set_bus_skew` on the gray buses).
The clocks are the request of the release image, `kernel_frequency=0:200|1:100`
(docs/platform-u200.md §8.3). Every block, pipe and cell is drawn solid.

Run: build/tools/svgvenv/bin/python docs/fig/src/clocks.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import (CARD, INK, INK_SOFT, ROLES, WHITE, Fig, chamfer, fmt, measure,  # noqa: E402
                    tint, wrap)

W = 1600
CX = 800                              # the boundary between the two domains

# ---- domain colour families (colour = clock domain) ----------------------------------------------
CORE = dict(acc="#1F5FA6", blk="#9CC4EC", band="#DDEAF7", pipe="#C8DBEE", cell="#7FB0E3")
RV = dict(acc="#A44A0C", blk="#F5BC8A", band="#FBE9D8", pipe="#F4D7C0", cell="#EFA570")
SHELL = ROLES["vendor"]

# ---- columns (x) ---------------------------------------------------------------------------------
CR0, CR1 = 56, 792                    # core clock region
RR0, RR1 = 808, 1544                  # clk_rv region
LC0, LC1 = 80, 280                    # core: left column (blocks without a crossing)
FC0, FC1 = 336, 520                   # core: column facing the boundary
RF0, RF1 = 1080, 1264                 # clk_rv: column facing the boundary
BS0, BS1 = 1292, 1320                 # clk_rv: RISC-V bus
RC0, RC1 = 1348, 1520                 # clk_rv: right column
OVL = 8                               # a pipe starts this far inside its block

# ---- rows (y) ------------------------------------------------------------------------------------
SH_Y, SH_H = 56, 128                  # XRT shell strip
TAB_Y, TAB_H = 248, 104                # domain header tabs
REG_Y = 288                           # region top edge (the tabs straddle it)
CT = 392                              # content top
ROWS = dict(dmi=436, load=508, log=580, rvc=652, rt=728, rvl=804, mtime=880, lvl=956, lq=1032,
            iss=1116, sd=1188, err=1260, spm=1344)
CSR_B = ROWS["lq"] + 34
ISS_T, ISS_B = ROWS["iss"] - 34, ROWS["err"] + 34
BR_T, BR_B = ISS_B + 16, ISS_B + 16 + 68
WIN_T = ROWS["log"] - 36
BOT = BR_B                            # lowest block edge
REG_B = BOT + 48                      # region bottom edge

# ---- crossing cells ------------------------------------------------------------------------------
CELL_W = dict(ff=128, fifo=128, gray=112, hs=112, rst=64)
CELL_H = dict(ff=44, fifo=60, gray=44, hs=44, rst=44)


def text_ok(text, role, maxw, what):
    w, _ = measure(text, role)
    assert w <= maxw, f"{what}: {text!r} is {w:.0f} wide, room for {maxw:.0f}"


# ---- the figure ----------------------------------------------------------------------------------
LEG1_Y = REG_B + 64
LEG2_Y = LEG1_Y + 56
LEG3_Y = LEG2_Y + 52
FOOT_Y = LEG3_Y + 44
FOOT_W = 1440
foot = wrap(
    "A pipe changes colour at its cell and its head marks the receiving side (two heads: request one "
    "way, response data back). Widths: a / b is request / response.", FOOT_W)
H = int((FOOT_Y + 24 * (foot.count("\n") + 1) + 40 + 7) // 8 * 8)

f = Fig("clocks", W, H,
        title="Clock domains and clock-domain crossings",
        desc="Two clock domains side by side, each one tinted region with its own colour. The core clock "
             "domain (ap_clk, 200 MHz, blue) holds the platform adapter, the CSR "
             "block, the issue path with the checker and the unit queues, the SPM bridge, the DMA, QACT, "
             "the MVM clusters, the scratchpad and one memory arbiter per channel. The control core "
             "domain (clk_rv from ap_clk_2, 100 MHz, orange) holds Ibex, the debug "
             "module, ITCM, DTCM, the RISC-V bus and the control window. The XRT shell, a grey box at "
             "the top, supplies both kernel clocks and resets, each drawn as a line in its domain "
             "colour, and carries the CSR port and the memory ports into the core domain. Between the "
             "two regions, on the boundary, tsr_ctrl_cdc has one synchroniser cell per bundle of "
             "crossings, drawn as a pipe that changes colour at the cell. From clk_rv to the core "
             "clock: the issue FIFO (32 entries of 387 bits, gray pointers), the log FIFO (16 entries "
             "of 32 bits), the gray counters ISSUED_SEQ, clk_rv cycles and PERF, the RT_STATE and "
             "RT_INFO handshake, and the RV levels and sticky events through two-flop synchronisers. "
             "From the core clock to clk_rv: the S, D and MTIME gray counters, the error records (6 of "
             "128 bits, handshake), and the levels. Both ways: the SPM bridge and core window "
             "handshake, the LQ and LCQ register handshake, the LOAD handshake into the TCM load "
             "port, and the debug-module interface handshake. The reset of the core domain reaches "
             "clk_rv through a reset synchroniser. The constraints are ASYNC_REG on the synchroniser "
             "flops, a maximum delay equal to the smaller clock period, and a bus skew limit on the "
             "gray buses.")


# ---- small drawing helpers -----------------------------------------------------------------------
def poly_line(pts, color=INK, sw=1.5):
    f.raw(f'<polyline points="{" ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)}" fill="none" '
          f'stroke="{color}" stroke-width="{fmt(sw)}" stroke-linejoin="miter"/>')


def mini_arrow(x1, y1, x2, y2, color=INK, sw=1.5, hl=7, hw=6):
    """A short arrow along an axis, drawn small enough for a cell symbol."""
    d = 1 if x2 > x1 else -1
    poly_line([(x1, y1), (x2 - d * hl * 0.6, y2)], color, sw)
    f.poly([(x2, y2), (x2 - d * hl, y2 - hw / 2), (x2 - d * hl, y2 + hw / 2)], color, stroke=color,
           sw=0.5)


def flop(x, y, w=22, h=26, color=WHITE):
    """A flip-flop symbol: a square with the clock wedge on its left edge."""
    f.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], color, sw=1.5)
    cy = y + h - 7
    poly_line([(x, cy - 5), (x + 6, cy), (x, cy + 5)], INK, 1.2)


def block(x, y, w, h, fam, label, sub="", label_role="label_s", text_h=None, pad=14, stack=0):
    """A block in a domain colour family. A tall block keeps its text near the top (text_h)."""
    col = fam["blk"]
    for i in range(stack, 0, -1):
        d = 8 * i
        f.poly(chamfer(x + d, y - d, w, h), tint(col, 0.35 * i))
    f.poly(chamfer(x, y, w, h), col)
    f.label_block(x, y, w, text_h or h, label, sub, label_role, "sub", "middle", pad)


def halves(cx, cy, cw, ch, c1, c2, vertical=False):
    """The two halves of a crossing cell: c1 left (top), c2 right (bottom)."""
    x0, x1, y0, y1, k = cx - cw / 2, cx + cw / 2, cy - ch / 2, cy + ch / 2, 6
    if not vertical:
        f.poly([(x0 + k, y0), (cx, y0), (cx, y1), (x0 + k, y1), (x0, y1 - k), (x0, y0 + k)], c1)
        f.poly([(cx, y0), (x1 - k, y0), (x1, y0 + k), (x1, y1 - k), (x1 - k, y1), (cx, y1)], c2)
        f.vline(cx, y0, y1, dashed=False, color=INK, sw=1.5)
    else:
        f.poly([(x0 + k, y0), (x1 - k, y0), (x1, y0 + k), (x1, cy), (x0, cy), (x0, y0 + k)], c1)
        f.poly([(x0, cy), (x1, cy), (x1, y1 - k), (x1 - k, y1), (x0 + k, y1), (x0, y1 - k)], c2)
        f.hline(x0, x1, cy, dashed=False, color=INK, sw=1.5)


def cell(kind, cx, cy, src="l", dest="r", fam_l=CORE, fam_r=RV):
    """A synchroniser cell on the boundary at (cx, cy). The left half is in the left domain's colour
    and the right half in the right domain's colour; `src` is the side of the source domain (the
    write half of a FIFO), `dest` the side of the destination (where the 2-flop synchroniser sits)."""
    cw, ch = CELL_W[kind], CELL_H[kind]
    halves(cx, cy, cw, ch, fam_l["cell"], fam_r["cell"])
    if kind == "ff":                               # two flops in the destination domain
        sgn = 1 if dest == "r" else -1
        fx = [cx + 10, cx + 38] if dest == "r" else [cx - 10 - 20, cx - 38 - 20]
        poly_line([(cx - sgn * 18, cy), (fx[0] if dest == "r" else fx[0] + 20, cy)], INK, 1.5)
        flop(fx[0], cy - 13, 20, 26)
        flop(fx[1], cy - 13, 20, 26)
        a = fx[0] + 20 if dest == "r" else fx[1] + 20
        b = fx[1] if dest == "r" else fx[0]
        poly_line([(a, cy), (b, cy)], INK, 1.5)
    elif kind == "gray":
        f.poly(chamfer(cx - 29, cy - 12, 58, 24, 4), WHITE, sw=1.5)
        f.text(cx, cy + 5.5, "GRAY", "tag", anchor="middle")
    elif kind == "hs":
        f.poly(chamfer(cx - 33, cy - 15, 66, 30, 4), WHITE, sw=1.5)
        mini_arrow(cx - 22, cy - 6, cx + 22, cy - 6)
        mini_arrow(cx + 22, cy + 6, cx - 22, cy + 6)
    elif kind == "fifo":                           # write half in the source colour, read half in the destination's
        f.text(cx + (-32 if src == "l" else 32), cy - 17, "WR", "tag", anchor="middle")
        f.text(cx + (32 if src == "l" else -32), cy - 17, "RD", "tag", anchor="middle")
        f.poly(chamfer(cx - 46, cy + 1, 92, 24, 4), WHITE, sw=1.5)
        f.text(cx, cy + 18.5, "gray ptr", "note", anchor="middle")
    elif kind == "rst":                            # two flops with the asynchronous reset pins
        for fx in (cx - 24, cx + 6):
            flop(fx, cy - 2, 18, 20)
            poly_line([(fx + 9, cy - 16), (fx + 9, cy - 2)], INK, 1.2)
            f.dot(fx + 9, cy - 16, r=2.2)
        poly_line([(cx - 6, cy + 8), (cx + 6, cy + 8)], INK, 1.5)


def pipe(row, kind, name, detail, flow, thick=36, lface=FC1, rface=RF0, rend=None):
    """One crossing: a pipe from the source block through the cell to the destination block.
    flow: 'l' = clk_rv to core (head at the core block), 'r' = core to clk_rv, 'b' = both (request
    one way and response data back)."""
    y = ROWS[row]
    cw = CELL_W[kind]
    hw, hl = thick + 16, 24
    lx0 = lface - OVL
    rx1 = (rend if rend is not None else rface + OVL)
    head_l, head_r = flow in ("l", "b"), flow in ("r", "b")
    cl, cr = CX - cw / 2 + 6, CX + cw / 2 - 6
    f.arrow([(cl, y), (lx0, y)] if head_l else [(lx0, y), (cl, y)], w=thick, hw=hw, hl=hl,
            head=head_l, fill=CORE["pipe"])
    f.arrow([(cr, y), (rx1, y)] if head_r else [(rx1, y), (cr, y)], w=thick, hw=hw, hl=hl,
            head=head_r, fill=RV["pipe"])
    cell(kind, CX, y, src="r" if flow == "l" else "l", dest="l" if flow == "l" else "r")
    # the name sits in the core half, the detail in the clk_rv half (reading order left to right)
    nx0 = (lx0 + hl + 10) if head_l else (lface + 12)
    nx1 = CX - cw / 2 - 8
    dx0 = CX + cw / 2 + 10
    dx1 = (rx1 - hl - 8) if head_r else (rface - 12)
    for s in name.split("\n"):
        text_ok(s, "callout", nx1 - nx0, f"{row} name")
    for s in detail.split("\n"):
        text_ok(s, "note", dx1 - dx0, f"{row} detail")
    f.text(nx0, y, name, "callout", valign="middle")
    f.text(dx0, y, detail, "note", valign="middle")


# =================================================================================================
# XRT shell (one vendor box) with the kernel clock and reset sources
# =================================================================================================
f.box(56, SH_Y, 1488, SH_H, "vendor", "XRT SHELL",
      sub="PCIe · DDR4 controllers\neach on its own clocks,\nconverted by the shell", align="start", pad=32)


def source(cx, fam, clk, rst):
    """Kernel clock and reset source in the shell, one per domain, with the clock icon."""
    x, y, w, h = cx - 108, 112, 216, 56
    f.poly(chamfer(x, y, w, h, 10), fam["blk"])
    ix, iy = x + 16, y + h / 2
    poly_line([(ix, iy + 9), (ix + 8, iy + 9), (ix + 8, iy - 9), (ix + 20, iy - 9), (ix + 20, iy + 9),
               (ix + 32, iy + 9), (ix + 32, iy - 9), (ix + 44, iy - 9)], fam["acc"], 2.5)
    f.text(x + 72, y + 22, clk, "label_s", valign="middle")
    f.text(x + 72, y + 42, rst, "sub", valign="middle")


source(540, CORE, "AP_CLK", "AP_RST_N")
source(1060, RV, "AP_CLK_2", "AP_RST_N_2")

# =================================================================================================
# The two regions, their tabs and the boundary
# =================================================================================================
f.panel(CR0, REG_Y, CR1 - CR0, REG_B - REG_Y, fill=CORE["band"], stroke=CORE["acc"])
f.panel(RR0, REG_Y, RR1 - RR0, REG_B - REG_Y, fill=RV["band"], stroke=RV["acc"])


def tab(x0, x1, fam, title, clk, note):
    f.poly(chamfer(x0, TAB_Y, x1 - x0, TAB_H, 14), fam["acc"])
    f.text((x0 + x1) / 2, TAB_Y + 38, title, "title", anchor="middle", color=CARD)
    f.text((x0 + x1) / 2, TAB_Y + 70, clk, "label", anchor="middle", color=CARD)
    f.text((x0 + x1) / 2, TAB_Y + 94, note, "sub", anchor="middle", color=CARD)


T0, T1 = 336, 744                     # core tab
U0, U1 = 856, 1264                    # clk_rv tab
tab(T0, T1, CORE, "CORE DOMAIN", "AP_CLK · 200 MHz", "core clock")
tab(U0, U1, RV, "CONTROL CORE DOMAIN", "AP_CLK_2 · 100 MHz", "clk_rv")

# clock tree and reset lines from the sources into the tabs (one pair per domain, in the domain colour)
for cx, fam, rname in ((540, CORE, "tsr_xrt_reset · registered"), (1060, RV, "tsr_rst_sync")):
    xc, xr = cx - 52, cx + 52
    f.line([(xc, 168), (xc, TAB_Y)], color=fam["acc"], sw=3)
    f.line([(xr, 168), (xr, TAB_Y)], color=fam["acc"], sw=2, dashed=True)
    halves(xr, 208, 40, 40, SHELL, fam["cell"], vertical=True)
    for fx in (xr - 15, xr + 2):
        flop(fx, 202, 13, 18)
    f.text(xr + 30, 208, rname, "note", valign="middle", color=INK)

# the reset of the core domain into clk_rv (u_rst_rv): between the two tabs, on the boundary
ry = TAB_Y + TAB_H // 2
f.line([(T1, ry), (CX - 32, ry)], color=CORE["acc"], sw=2, dashed=True, end=False)
f.line([(CX + 32, ry), (U0, ry)], color=RV["acc"], sw=2, dashed=True)
cell("rst", CX, ry)
f.text(CX, ry - 34, "RESET", "note", anchor="middle", color=INK)

# crossings tag on the boundary, and the module names of the two regions
f.tag(CX, 376, "CROSSINGS · TSR_CTRL_CDC")
f.tag(CR0 + 36, REG_B, "TSR_CORE_DP", anchor="start")
f.tag(RR0 + 96, REG_B, "TSR_CTRL_RV", anchor="start")

# =================================================================================================
# Core clock domain: blocks
# =================================================================================================
AD_Y, AD_H = CT, 120
ARB_Y, ARB_H = 568, 120
DMA_Y, QA_Y, MV_Y, UH = 736, 912, 1088, 128
SPM_Y = BR_B - 112
block(LC0, AD_Y, LC1 - LC0, AD_H, CORE, "PLATFORM\nADAPTER", sub="s_axi_control\nm_axi × NUM_CH", pad=12)
block(LC0, ARB_Y, LC1 - LC0 - 16, ARB_H, CORE, "TSR_MEM_ARB", sub="one per channel", stack=2, pad=12)
f.tag(LC1, ARB_Y - 16, "× NUM_CH", anchor="end")
block(LC0, DMA_Y, LC1 - LC0, UH, CORE, "TSR_DMA", sub="LD · ST\nall channels", pad=12)
block(LC0, QA_Y, LC1 - LC0, UH, CORE, "TSR_QACT", sub="quantizes f32\nto the QA formats", pad=12)
block(LC0, MV_Y, LC1 - LC0, UH, CORE, "MVM CLUSTERS", sub="tsr_mvm\nNUM_CH clusters", pad=12)
block(LC0, SPM_Y, LC1 - LC0, 112, CORE, "TSR_SPM", sub="4 MiB · 2R2W\neven/odd banks", pad=12)

block(FC0, CT, FC1 - FC0, CSR_B - CT, CORE, "TSR_CSR", sub="host CSR port\nID · PERF\nLQ·LCQ·log",
      label_role="label", text_h=132, pad=12)
block(FC0, ISS_T, FC1 - FC0, ISS_B - ISS_T, CORE, "ISSUE PATH", sub="tsr_issue\ntsr_chk\nunit queues",
      label_role="label", text_h=ISS_B - ISS_T, pad=12)
block(FC0, BR_T, FC1 - FC0, BR_B - BR_T, CORE, "SPM BRIDGE")

# =================================================================================================
# clk_rv domain: blocks and the RISC-V bus
# =================================================================================================
f.poly(chamfer(BS0, 380, BS1 - BS0, BOT - 380, 10), RV["pipe"])
f.tag((BS0 + BS1) / 2, 380, "RISC-V BUS")
block(RF0, CT, RF1 - RF0, 88, RV, "DEBUG MODULE", sub="riscv-dbg", pad=10)
block(RF0, WIN_T, RF1 - RF0, BOT - WIN_T, RV, "CONTROL\nWINDOW", sub="tsr_ctl_win\nissue port\nLQ · LCQ\nlog · timer",
      label_role="label", text_h=180, pad=12)
RH = 200
rc = [CT, CT + (BOT - CT - RH) // 2 // 8 * 8, BOT - RH]
for ry0, (lab, sub) in zip(rc, [("IBEX", "RV32IMC"), ("ITCM", "128 KiB"), ("DTCM", "64 KiB")]):
    block(RC0, ry0, RC1 - RC0, RH, RV, lab, sub=sub, pad=12)
    f.line([(BS1 + 2, ry0 + RH / 2), (RC0 + 2, ry0 + RH / 2)], start=True)
for yy in (CT + 44, WIN_T + 90):
    f.line([(RF1 - 2, yy), (BS0 + 2, yy)], start=True)

# =================================================================================================
# Shell interfaces and the arrows inside the core domain
# =================================================================================================
f.arrow([(104, SH_Y + SH_H - 8), (104, AD_Y + 8)], tail=True)
f.text(136, 224, "CSR PORT", "callout", valign="middle")
f.arrow([(272, SH_Y + SH_H - 8), (272, AD_Y + 8)], tail=True, fat=True)
f.text(316, 224, "MEMORY PORTS", "callout", valign="middle")
f.arrow([(LC1 - 8, AD_Y + 60), (FC0 + 8, AD_Y + 60)], tail=True)
f.arrow([(136, AD_Y + AD_H - 8), (136, ARB_Y + 8)], tail=True)
f.arrow([(LC1 - 8, ROWS["spm"]), (FC0 + 8, ROWS["spm"])], tail=True)

# =================================================================================================
# The crossings, one pipe and one cell each
# =================================================================================================
pipe("dmi", "hs", "DMI", "41 b / 34 b", "b")
pipe("load", "hs", "LOAD_* · TCM", "65 b / 33 b", "b", rend=BS0 + 8)
pipe("log", "fifo", "LOG FIFO", "16 × 32 b", "l")
pipe("rvc", "gray", "ISSUED_SEQ\nCYCLES · PERF", "6 × 28 b\n16 b · 3 × 32 b", "l", thick=52)
pipe("rt", "hs", "RT_STATE/INFO", "64 b", "l")
pipe("rvl", "ff", "RV LEVELS\nSTICKY EVENTS", "RV_HALTED · SLEEP\n2 timeout toggles", "l", thick=52)
pipe("mtime", "gray", "MTIME", "64 b", "r")
pipe("lvl", "ff", "LEVELS\n+ RV_BOOT", "RV_RUN · holds\nirq · dbg_stop", "r", thick=52)
pipe("lq", "hs", "LQ / LCQ", "301 b / 33 b", "b")
pipe("iss", "fifo", "ISSUE FIFO", "32 × 387 b", "l")
pipe("sd", "gray", "S[u] · D[u]", "12 × 28 b", "r")
pipe("err", "hs", "ERROR RECORDS", "6 × 128 b", "r")
pipe("spm", "hs", "SPM · CORE WIN", "64 b / 513 b", "b")

# =================================================================================================
# Legend
# =================================================================================================
# row 1: domain colours and the two kinds of line from the shell
row1 = [("sw", CORE["blk"], "CORE · AP_CLK · 200 MHz"), ("sw", RV["blk"], "CONTROL CORE · CLK_RV · 100 MHz"),
        ("sw", SHELL, "XRT SHELL"),
        ("line", False, "CLOCK TREE"), ("line", True, "RESET")]
sample_w = {"sw": 30, "line": 48}
total = sum(sample_w[k] + 12 + measure(t, "note")[0] for k, _, t in row1) + 32 * (len(row1) - 1)
lx = CX - total / 2
for k, v, t in row1:
    if k == "sw":
        f.poly(chamfer(lx, LEG1_Y - 10, 30, 20, 5), v, sw=1.5)
    else:
        for i, fam in enumerate((CORE, RV)):
            f.raw(f'<line x1="{fmt(lx + i * 24)}" y1="{fmt(LEG1_Y)}" x2="{fmt(lx + i * 24 + 22)}" '
                  f'y2="{fmt(LEG1_Y)}" stroke="{fam["acc"]}" stroke-width="{2 if v else 3}"'
                  f'{" stroke-dasharray=\"9 6\"" if v else ""}/>')
    f.text(lx + sample_w[k] + 12, LEG1_Y, t, "note", valign="middle")
    lx += sample_w[k] + 12 + measure(t, "note")[0] + 32
# row 2: the crossing cells
items = [("ff", "2-FLOP SYNC"), ("fifo", "ASYNC FIFO"), ("hs", "HANDSHAKE"), ("gray", "GRAY COUNTER"),
         ("rst", "RESET SYNC")]
tw = [measure(t, "note")[0] for _, t in items]
total = sum(CELL_W[k] + 14 + w for (k, _), w in zip(items, tw)) + 36 * (len(items) - 1)
lx = CX - total / 2
for (k, t), w in zip(items, tw):
    cell(k, lx + CELL_W[k] / 2, LEG2_Y)
    f.text(lx + CELL_W[k] + 14, LEG2_Y, t, "note", valign="middle")
    lx += CELL_W[k] + 14 + w + 36
f.text(CX, LEG3_Y, "ASYNC_REG on the synchroniser flops · set_max_delay -datapath_only = the smaller "
       "period · set_bus_skew on the gray buses", "note", anchor="middle", valign="middle")
f.text(CX, FOOT_Y, foot, "note", anchor="middle", color=INK_SOFT)

f.save(scale=1)

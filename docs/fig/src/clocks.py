"""Figure `clocks`: the clock domains of tsr_core and every clock-domain crossing between them.

Content follows docs/design/04-control-core.md §2.1-§2.2 (block diagram, domains), §11.1 (clocks),
§11.2 (crossings), §11.3 (constraints), §12.2 (reset synchronisers), §14.3 (T1); the control-core RTL
in rtl/ctrl (`tsr_ctrl_cdc.sv` header and instances, `cdc/tsr_cdc_fifo.sv`, `cdc/tsr_cdc_gray.sv`,
`cdc/tsr_cdc_hs.sv`, `cdc/tsr_sync.sv`, `cdc/tsr_rst_sync.sv`, `tsr_ctrl.sv`, `tsr_ctrl_pkg.sv` for
the widths, `tsr_rv_bus.sv`, `tsr_ctl_win.sv`), `rtl/ctrl/viv/tsr_ctrl_cdc.xdc` (constraints),
`rtl/core/tsr_core.sv` (core boundary: clk, rst_n, clk_rv, rst_rv_n) and docs/platform-u200.md §3
and §14.1 (ap_clk, ap_clk_2):
- XRT shell: one vendor box. PCIe and the DDR4 controllers run on their own clocks, which the shell
  converts; it provides the kernel clocks ap_clk and ap_clk_2 and carries the CSR port and the
  memory ports into the core clock domain. Nothing of its inside is drawn;
- core clock domain (`ap_clk`, 276.3 MHz in the measured k4 build): `tsr_core_dp` with `tsr_csr`,
  `tsr_issue` + `tsr_chk` with the unit queues, DMA, QACT, the MVM clusters, `tsr_spm` and one
  `tsr_mem_arb` per channel;
- `clk_rv` domain (from `ap_clk_2`, 166.7 MHz planning value, CC §11.1): `tsr_ctrl_rv` with Ibex,
  the debug module, ITCM, DTCM, the RISC-V bus `tsr_rv_bus` and the control window `tsr_ctl_win`;
- crossings (`tsr_ctrl_cdc`): one cell per crossing of CC §11.2, grouped by direction (the arrow
  points to the receiving domain), with its primitive and width. Widths are the ones in
  `tsr_ctrl_pkg.sv` and the instances in `tsr_ctrl_cdc.sv`, the same as in the table of CC §11.2
  (a handshake's request width includes its control bits: LQ / LCQ 301 b, LOAD 65 b). The reset
  synchronisers of `tsr_ctrl` (2 stages, asynchronous assert, synchronous release) are drawn as a
  crossing into clk_rv;
- constraints legend: the three rules of `tsr_ctrl_cdc.xdc` (ASYNC_REG, `set_max_delay
  -datapath_only` = the smaller period, `set_bus_skew` on the gray buses).
Status (STYLE.md): solid blocks ran on the card; the control core, the issue path and the crossings
exist in rtl/ and in simulation and are planned for the card (M6), so their regions are dashed and
carry one PLANNED tag, and the blocks inside are dashed without their own tag. A solid block that
gains a mode carries a dashed tag with a `+`.
The 166.7 MHz of clk_rv is a planning value (T1: the control core met 6.0 ns out of context; at
4.0 ns it missed, so 250 MHz is out of reach), named as such in the footnote.

Run: build/tools/svgvenv/bin/python docs/fig/src/clocks.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK, INK_SOFT, Fig, measure, wrap  # noqa: E402

W = 1600
SX0, SX1 = 128, 1472                  # shell box and tsr_core panel
BX0, BX1 = 152, 1448                  # the two clock-domain bands and the crossings region
IX0, IX1 = 176, 1424                  # content inside the bands
CX = 800


def fits(text, role, maxw):
    w, _ = measure(text, role)
    assert w <= maxw, f"{text!r} is {w:.0f} wide, room for {maxw}"


# ---- rows (y) ----------------------------------------------------------------------------------
SH_Y, SH_H = 56, 104                  # XRT shell
PY0 = 200                             # tsr_core panel top
C_Y = PY0 + 24                        # core clock band
RA_Y, RA_H = C_Y + 24, 112            # core row A: tsr_csr, issue path, memory arbiters
RB_Y, RB_H = RA_Y + RA_H + 40, 112    # core row B: DMA, QACT, MVM, SPM
CL_Y = RB_Y + RB_H + 32               # core band label row (centre)
C_B = CL_Y + 32
X_Y = C_B + 16                        # crossings region top
ROW0 = X_Y + 88                       # first cell row
PITCH = 96
X_B = ROW0 + 5 * PITCH + 24
V_Y = X_B + 16                        # clk_rv band
VA_Y, VA_H = V_Y + 24, 112            # clk_rv blocks
BUS_Y, BUS_H = VA_Y + VA_H + 40, 64
VL_Y = BUS_Y + BUS_H + 36             # clk_rv band label row (centre)
V_B = VL_Y + 32
PY1 = V_B + 24                        # tsr_core panel bottom
LEG_Y = PY1 + 56
FOOT_Y = LEG_Y + 44
FOOT_W = 1440
foot = wrap(
    "Each arrow points to the receiving domain. Solid blocks ran on the card; dashed blocks and "
    "tags are planned for the card (M6). 166.7 MHz is the planning value of clk_rv: the control "
    "core met 6.0 ns out of context (T1) and did not close 4.0 ns (250 MHz).", FOOT_W)
H = int((FOOT_Y + 24 * (foot.count("\n") + 1) + 40 + 7) // 8 * 8)

f = Fig("clocks", W, H,
        title="Clock domains and clock-domain crossings",
        desc="The XRT shell, drawn as one box, gives the kernel the clocks ap_clk and ap_clk_2 and "
             "carries the CSR port and the memory ports into tsr_core. The core clock domain "
             "(ap_clk, 276.3 MHz in the k4 build) holds the CSR block, the issue path with the "
             "checker and the unit queues, the DMA, QACT, the MVM clusters, the scratchpad and one "
             "memory arbiter per channel. The clk_rv domain (ap_clk_2, 166.7 MHz planning value) "
             "holds Ibex, the debug module, the instruction and data memories, the RISC-V bus and "
             "the control window. Between them, tsr_ctrl_cdc carries every crossing, each with its "
             "primitive and width. From clk_rv to the core clock: the issue FIFO (gray pointers, "
             "32 entries of 387 bits), the log FIFO, the gray counters ISSUED_SEQ, clk_rv cycles "
             "and PERF, the RT_STATE and RT_INFO handshake, and the sticky-event toggles. From the "
             "core clock to clk_rv: the gray counters S, D and MTIME, the error records, the LOAD "
             "handshake and the reset synchronisers. In both directions: the SPM bridge and core "
             "window handshake, the LQ and LCQ register handshake, the debug-module interface "
             "handshake and the level synchronisers. The constraints are ASYNC_REG on the "
             "synchroniser flops, a maximum delay equal to the smaller clock period, and a bus "
             "skew limit on the gray buses.")

# ---- XRT shell ---------------------------------------------------------------------------------
f.box(SX0, SH_Y, SX1 - SX0, SH_H, "vendor", "XRT SHELL · HOST SIDE",
      sub="PCIe to the host PC · DDR4 controllers · each on its own clock, converted by the shell\n"
          "provides the kernel clocks ap_clk and ap_clk_2")

# ---- the tsr_core panel and its three regions -----------------------------------------------------
f.panel(SX0, PY0, SX1 - SX0, PY1 - PY0)
f.tag(CX, PY0, "TSR_CORE", anchor="middle")

# core clock band
f.band(BX0, C_Y, BX1 - BX0, C_B - C_Y,
       label="AP_CLK · CORE CLOCK DOMAIN · TSR_CORE_DP · 276.3 MHz (k4 build)",
       label_xy=(BX0 + 40, CL_Y + 7))
CSR_W = 296
ARB_W = 280
ISS_X0 = IX0 + CSR_W + 48
ARB_X = IX1 - 16 - ARB_W              # room for the two stacked copies behind the arbiter
ISS_W = ARB_X - 48 - ISS_X0
f.box(IX0, RA_Y, CSR_W, RA_H, "control", "TSR_CSR", sub="host CSR port\nID · PERF counters",
      tag="+ LQ · LCQ · LOG", tag_style="planned")
f.box(ISS_X0, RA_Y, ISS_W, RA_H, "control", "ISSUE PATH",
      sub="tsr_issue: dispatcher, unit queues\ntsr_chk: checks every tile op",
      planned=True, plan_tag=False, label_role="label")
f.box(ARB_X, RA_Y, ARB_W, RA_H, "io", "TSR_MEM_ARB", sub="one per DDR4 channel\nround robin per burst",
      stack=2)
f.tag(ARB_X + ARB_W + 16, RA_Y - 16, "× NUM_CH", anchor="end")
UW, UG = 264, 64
ux = [IX0 + i * (UW + UG) for i in range(4)]
f.box(ux[0], RB_Y, UW, RB_H, "io", "TSR_DMA", sub="LD · ST\nall channels")
f.box(ux[1], RB_Y, UW, RB_H, "compute", "TSR_QACT", sub="quantizes f32\nto the QA formats")
f.box(ux[2], RB_Y, UW, RB_H, "compute", "MVM CLUSTERS", sub="tsr_mvm\nNUM_CH clusters")
f.box(ux[3], RB_Y, UW, RB_H, "memory", "TSR_SPM", sub="4 MiB · 64 B lines\neven / odd banks",
      tag="+ BRIDGE · 2R2W", tag_style="planned")

# crossings region (outline only, planned as a whole)
f.panel(BX0, X_Y, BX1 - BX0, X_B - X_Y, fill=None, dashed=True)
f.tag(BX0 + 36, X_Y, "CLOCK-DOMAIN CROSSINGS · TSR_CTRL_CDC", anchor="start")
f.tag(BX1 - 36, X_Y, "PLANNED", anchor="end", planned=True)

COLW = (IX1 - IX0) // 3
TXT_DX, TXT_W = 84, COLW - 84 - 12
heads = ["CLK_RV TO CORE", "CORE TO CLK_RV", "BOTH WAYS"]
cols = [
    ("up", [
        ("ISSUE FIFO", "tsr_cdc_fifo · gray pointers", "32 entries × 387 b"),
        ("LOG FIFO", "tsr_cdc_fifo · gray pointers", "16 entries × 32 b"),
        ("ISSUED_SEQ · CYCLES · PERF", "tsr_cdc_gray · gray counters", "6 × 28 b · 16 b · 3 × 32 b"),
        ("RT_STATE · RT_INFO", "tsr_cdc_hs · tagged handshake", "64 b"),
        ("STICKY EVENTS", "toggles through tsr_sync", "PUSH_TIMEOUT · BRIDGE_TIMEOUT"),
    ]),
    ("down", [
        ("S[u] · D[u] · MTIME", "tsr_cdc_gray · gray counters", "12 × 28 b · 64 b"),
        ("ERROR RECORDS", "tsr_cdc_hs · one per unit", "6 × 128 b"),
        ("LOAD_*", "tsr_cdc_hs · tagged handshake", "65 b req / 33 b rsp"),
        ("RESETS INTO CLK_RV", "tsr_rst_sync · 2 stages", "async assert, sync release"),
    ]),
    ("both", [
        ("SPM BRIDGE · CORE WINDOW", "tsr_cdc_hs · one shared slot", "64 b req / 513 b rsp"),
        ("LQ / LCQ REGISTERS", "tsr_cdc_hs · own slot", "301 b req / 33 b rsp"),
        ("DMI", "tsr_cdc_hs · tagged handshake", "41 b req / 34 b rsp"),
        ("LEVELS", "tsr_sync · 2 flops per bit", "RV_RUN, holds, irq, dbg_stop"),
    ]),
]
for ci, (kind, cells) in enumerate(cols):
    cx0 = IX0 + ci * COLW
    f.text(cx0 + 12, X_Y + 52, heads[ci], "callout", valign="middle")
    f.hline(cx0 + 8, cx0 + COLW - 24, X_Y + 72, dashed=False, color=INK_SOFT)
    ax = cx0 + 36
    for ri, (name, prim, width) in enumerate(cells):
        fits(name, "callout", TXT_W)
        fits(prim, "note", TXT_W)
        fits(width, "note", TXT_W)
        top = ROW0 + ri * PITCH
        a0, a1 = top + 4, top + 76
        if kind == "up":
            f.arrow([(ax, a1), (ax, a0)])
        elif kind == "down":
            f.arrow([(ax, a0), (ax, a1)])
        else:
            f.arrow([(ax, a0), (ax, a1)], tail=True)
        tx = cx0 + TXT_DX
        f.text(tx, top + 22, name, "callout")
        f.text(tx, top + 48, prim, "note", color=INK)
        f.text(tx, top + 70, width, "note", color=INK_SOFT)

# constraints legend in the free cells of the last row
KX0, KX1 = IX0 + COLW + 16, IX1
KY = ROW0 + 4 * PITCH + 2
f.band(KX0, KY, KX1 - KX0, 88)
f.text(KX0 + 20, KY + 36, "CONSTRAINTS", "panel", color=INK_SOFT, valign="middle")
f.text(KX0 + 20, KY + 60, "tsr_ctrl_cdc.xdc", "note", color=INK_SOFT, valign="middle")
for i, line in enumerate(["ASYNC_REG on every synchroniser flop",
                          "set_max_delay -datapath_only = the smaller clock period",
                          "set_bus_skew on the gray buses"]):
    fits(line, "note", KX1 - KX0 - 216 - 12)
    f.text(KX0 + 216, KY + 20 + i * 24, line, "note", valign="middle")

# clk_rv band
f.band(BX0, V_Y, BX1 - BX0, V_B - V_Y,
       label="AP_CLK_2 · CLK_RV DOMAIN · TSR_CTRL_RV · 166.7 MHz (planning value)",
       label_xy=(BX0 + 40, VL_Y + 7))
f.tag(BX1 - 24, VL_Y, "PLANNED", anchor="end", planned=True)
widths = [192, 192, 176, 176, 320]
gap = (IX1 - IX0 - sum(widths)) / 4
vx, x = [], IX0
for w in widths:
    vx.append(x)
    x += w + gap
blocks = [
    ("control", "DEBUG\nMODULE", "riscv-dbg"),
    ("control", "IBEX", "RV32IMC"),
    ("memory", "ITCM", "128 KiB"),
    ("memory", "DTCM", "64 KiB"),
    ("control", "CONTROL WINDOW", "issue port · LQ · LCQ\nlog · timer"),
]
for (role, lab, sub), bx, bw in zip(blocks, vx, widths):
    f.box(bx, VA_Y, bw, VA_H, role, lab, sub=sub, planned=True, plan_tag=False, label_role="label_s")
    f.line([(bx + bw / 2, VA_Y + VA_H - 6), (bx + bw / 2, BUS_Y + 6)], start=True)
f.box(IX0, BUS_Y, IX1 - IX0, BUS_H, "io", "RISC-V BUS", sub="tsr_rv_bus", planned=True,
      plan_tag=False, label_role="label_s")

# ---- the arrows between the shell and the core clock domain -----------------------------------
csr_ax = IX0 + 44
f.arrow([(csr_ax, SH_Y + SH_H - 8), (csr_ax, RA_Y + 12)], tail=True)
f.text(csr_ax + 40, (SH_Y + SH_H + PY0) / 2, "CSR PORT", "callout", valign="middle")
mem_ax = ARB_X + 72
f.arrow([(mem_ax, SH_Y + SH_H - 8), (mem_ax, RA_Y + 12)], tail=True, fat=True)
f.text(mem_ax - 64, (SH_Y + SH_H + PY0) / 2, "MEMORY PORTS", "callout",
       anchor="end", valign="middle")

# ---- the clock lines: from the shell down the margin to the label of each band ------------------
f.line([(SX0, SH_Y + 40), (76, SH_Y + 40), (76, VL_Y), (BX0 + 14, VL_Y)])      # ap_clk_2
f.line([(SX0, SH_Y + 72), (100, SH_Y + 72), (100, CL_Y), (BX0 + 14, CL_Y)])    # ap_clk

# ---- legend ------------------------------------------------------------------------------------
f.legend(800, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                      ("io", "DATA MOVEMENT"), ("vendor", "VENDOR-SPECIFIC"),
                      ("neutral", "PLANNED", True)])
f.text(800, FOOT_Y, foot, "note", anchor="middle", color=INK_SOFT)

f.save(scale=1)

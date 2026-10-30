"""Figure `layers`: the four layers, what is shared and what each target writes.

Content follows docs/spec.md §2 (System structure, "Portability: four layers and targets": the
layer table and the per-target table for the Alveo U200, Zynq / Kria and SoC / ASIC), the
adapter in docs/platform-u200.md §1 and the porting notes in docs/rtl-m2.md §1 and docs/rtl-m3.md
§1. Visual grammar: three target lanes; a block that spans all three lanes is the same on every
target, a tile inside a lane is written per target.
- Software (sw/, shared): the ggml-tesseract backend (op lists), tsr_rt, the firmware SDK (sw/ctrl:
  runtime, kernels, header library; runs on the control core on every target) and the simulator
  with libtsr_ref (the `sim` transport runs it in process).
- Transport (sw/transport, per target): xrt (U200), Zynq AXI, SoC driver (docs/xrt-host.md,
  docs/design/01-execution-model.md §10).
- Platform adapter (rtl/platform/TARGET, per target): U200 alveo_xrt = AXI4-Lite to the CSR port
  and one AXI4 bridge per DDR4 channel, `ap_ctrl_none` (EX §9); Zynq / SoC: to be written.
- Core (rtl/core, shared, parameters only): control core with the issue path and the vector unit
  (planned, M6: docs/design/04-control-core.md §2.1, docs/design/02-vector-unit.md §7), DMA, QACT, SPM,
  MVM clusters x NUM_CH. Parameters: U200 NUM_CH 4, BEAT_BITS 512; Zynq NUM_CH 1-2, BEAT_BITS about
  128 (needs RTL work: the current RTL requires BEAT_BITS = 512 and LANES = 128, docs/spec.md §2,
  §3); SoC to be decided.
- Tech wrappers (rtl/tech/TARGET, per target): U200 xilinx = URAM, BRAM, DSP48E2 (docs/rtl-m3.md
  §14); Zynq to be written; SoC SRAM macros (for example on an open PDK). rtl/tech/generic:
  behavioural wrappers for simulation and lint on any target (shared).
- Interfaces between the layers (edge labels): tsr_transport API; CSR access, DRAM buffers; CSR
  port, memory ports and the two clocks (docs/design/04-control-core.md §13.1); tsr_sram_1r1w,
  tsr_mac_lane.
Status (STYLE.md): the U200 transport, adapter and tech wrappers ran on the card;
Zynq / Kria and SoC / ASIC are ideas, not scheduled (docs/spec.md §15 roadmap), drawn dashed with an
IDEA tag; the planned core blocks carry PLANNED.

Run: build/tools/svgvenv/bin/python docs/fig/src/layers.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK, INK_SOFT, Fig, measure, wrap  # noqa: E402

W = 1600
PX0, PX1 = 224, 1376                  # panel; row headers left of it, callouts right of it
RX = (PX1 + W) / 2                    # callout centre
HX = 204                              # right edge of the row headers
LANE_X = [248, 624, 1000]             # target lanes (bands)
LANE_W = 352
LC = [x + LANE_W / 2 for x in LANE_X]           # lane centres: 424, 800, 1176
TX = [x + 24 for x in LANE_X]                   # tile x
TW = LANE_W - 48                                # tile width 304
SX0, SX1 = LANE_X[0], LANE_X[2] + LANE_W        # shared slabs bridge all lanes: 248..1352
IDEA = [False, True, True]                      # lanes 1 and 2 are ideas, not scheduled

# rows (y)
PY0 = 56
BAND_Y = PY0 + 24
HEAD_Y, HEAD_H = BAND_Y + 16, 64
SW_Y, SW_H = HEAD_Y + HEAD_H + 24, 168          # software slab
TR_Y, TR_H = SW_Y + SW_H + 88, 112              # transport tiles
PA_Y, PA_H = TR_Y + TR_H + 88, 160              # platform adapter tiles
CO_Y = PA_Y + PA_H + 96                         # core slab
CU_Y, CU_H = CO_Y + 40, 112                     # core units
CP_Y, CP_H = CU_Y + CU_H + 32, 112              # core parameters (per lane)
CO_H = CP_Y + CP_H + 24 - CO_Y
TE_Y, TE_H = CO_Y + CO_H + 88, 120              # tech wrapper tiles
GE_Y, GE_H = TE_Y + TE_H + 24, 80               # rtl/tech/generic
BAND_B = GE_Y + GE_H + 24
PY1 = BAND_B + 24
LEG_Y = PY1 + 48
H = LEG_Y + 104

f = Fig("layers", W, H,
        title="Tesseract layers and targets",
        desc="Four layers: software, platform adapter, core and tech wrappers. The software "
             "(ggml-tesseract, tsr_rt, the firmware SDK and the simulator), the core RTL and the "
             "generic simulation wrappers are the same on every target. Each target writes its "
             "own transport, platform adapter and tech wrappers and sets the core parameters: "
             "Alveo U200 (xrt, alveo_xrt, xilinx, NUM_CH 4, BEAT_BITS 512), Zynq / Kria and SoC "
             "/ ASIC (ideas). The control core, issue path and vector unit are planned.")


def row_head(y_mid, name="", path="", tag=""):
    """Row header left of the panel, right-aligned at HX: name, path, a SHARED / PER TARGET tag."""
    parts = []                                       # (height, draw)
    if name:
        role = "label" if measure(name, "label")[0] <= 150 else "label_s"
        parts.append((measure(name, role)[1], lambda y, n=name, r=role:
                      f.text(HX, y, n, r, anchor="end", valign="top")))
    if path:
        parts.append((measure(path, "sub")[1], lambda y, p=path:
                      f.text(HX, y, p, "sub", anchor="end", valign="top", color=INK_SOFT)))
    if tag:
        parts.append((27, lambda y, t=tag: f.tag(HX, y + 13.5, t, anchor="end")))
    gap = 14
    total = sum(h for h, _ in parts) + gap * (len(parts) - 1)
    y = y_mid - total / 2
    for h, draw in parts:
        draw(y)
        y += h + gap


# ---- panel and target lanes ------------------------------------------------------------------------
f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0)
for x in LANE_X:
    f.band(x, BAND_Y, LANE_W, BAND_B - BAND_Y)
heads = [("ALVEO U200", "ran on the card"), ("ZYNQ / KRIA", "idea, not scheduled"),
         ("SOC / ASIC", "idea, not scheduled")]
for (name, sub), x in zip(heads, LANE_X):
    f.label_block(x, HEAD_Y, LANE_W, HEAD_H, name, sub)

# ---- arrows between the layers (drawn first, blocks cover their ends) -------------------------------
for i, x in enumerate(LC):
    d = IDEA[i]
    f.arrow([(x, SW_Y + SW_H - 12), (x, TR_Y + 10)], dashed=d)
    f.arrow([(x, TR_Y + TR_H - 10), (x, PA_Y + 10)], dashed=d)
    f.arrow([(x, PA_Y + PA_H - 10), (x, CO_Y + 10)], tail=True, dashed=d)
    f.arrow([(x, CO_Y + CO_H - 12), (x, TE_Y + 10)], dashed=d)

# ---- software (shared) ------------------------------------------------------------------------------
f.panel(SX0, SW_Y, SX1 - SX0, SW_H)
sb_y, sb_h = SW_Y + 32, SW_H - 64
x = SX0 + 24
for w, lab, sub in ((240, "GGML-TESSERACT", "backend\nop lists"),
                    (240, "TSR_RT", "weight upload\nlinear"),
                    (240, "FIRMWARE SDK", "sw/ctrl\nruntime · kernels"),
                    (240, "SIMULATOR", "libtsr_ref\nTSR_DEVICE=sim")):
    f.box(x, sb_y, w, sb_h, "host", lab, sub=sub, label_role="label_s")
    x += w + 32

# ---- transport, platform adapter (per target) --------------------------------------------------------
tr = [("XRT", "TSR_DEVICE=xrt"), ("ZYNQ AXI", ""), ("SOC DRIVER", "")]
pa = [("ALVEO_XRT", "AXI4-Lite to CSR port,\none AXI4 bridge per\nDDR4 channel"),
      ("TO BE\nWRITTEN", ""), ("TO BE\nWRITTEN", "")]
for i in range(3):
    tag = "IDEA" if IDEA[i] else ""
    f.box(TX[i], TR_Y, TW, TR_H, "host", tr[i][0], sub=tr[i][1], planned=IDEA[i], tag=tag)
    f.box(TX[i], PA_Y, TW, PA_H, "adapter", pa[i][0], sub=pa[i][1], planned=IDEA[i], tag=tag)

# ---- core (shared RTL, parameters per target) ---------------------------------------------------------
f.panel(SX0, CO_Y, SX1 - SX0, CO_H)
f.tag(SX0 + 24, CO_Y, "TSR_CORE", anchor="start")      # clear of the lane 0 core <-> adapter arrow
x = SX0 + 24
for w, role, lab, sub, kw in ((224, "control", "CONTROL CORE\n+ ISSUE PATH", "",
                               dict(planned=True, tag="PLANNED")),
                              (120, "io", "DMA", "", {}),
                              (120, "compute", "QACT", "", {}),
                              (120, "memory", "SPM", "", {}),
                              (184, "compute", "MVM\nCLUSTERS", "",
                               dict(stack=2, tag="× NUM_CH")),
                              (168, "compute", "VECTOR\nUNIT", "",
                               dict(planned=True, tag="PLANNED"))):
    bw = w - 16 if kw.get("stack") else w
    f.box(x, CU_Y, bw, CU_H, role, lab, sub=sub, label_role="label_s", **kw)
    x += w + 20
params = ["NUM_CH 4\nBEAT_BITS 512", "NUM_CH 1–2\nBEAT_BITS about 128\nneeds RTL work",
          "to be decided"]
ptags = ["", "IDEA", "IDEA"]
for i in range(3):
    f.box(TX[i], CP_Y, TW, CP_H, "neutral", sub=params[i], planned=IDEA[i], tag=ptags[i], cut=10)

# ---- tech wrappers (per target) and the generic ones (shared) -----------------------------------------
te = [("XILINX", "rtl/tech/xilinx\nURAM, BRAM, DSP48E2"), ("TO BE\nWRITTEN", ""),
      ("SRAM MACROS", "for example on\nan open PDK")]
for i in range(3):
    f.box(TX[i], TE_Y, TW, TE_H, "vendor", te[i][0], sub=te[i][1], planned=IDEA[i],
          tag="IDEA" if IDEA[i] else "")
f.box(SX0 + 24, GE_Y, SX1 - SX0 - 48, GE_H, "neutral", "GENERIC",
      sub="rtl/tech/generic · behavioural wrappers for simulation and lint on any target",
      label_role="label_s", cut=12)

# ---- row headers ----------------------------------------------------------------------------------------
row_head(SW_Y + SW_H / 2, "SOFTWARE", "sw/", "SHARED")
row_head(TR_Y + TR_H / 2, "TRANSPORT", "sw/transport", "PER TARGET")
row_head(PA_Y + PA_H / 2, "PLATFORM\nADAPTER", "rtl/platform/", "PER TARGET")
row_head(CU_Y + CU_H / 2, "CORE", "rtl/core", "SHARED")
row_head(CP_Y + CP_H / 2, "PARAMETERS", "", "PER TARGET")
row_head(TE_Y + TE_H / 2, "TECH\nWRAPPERS", "rtl/tech/", "PER TARGET")
row_head(GE_Y + GE_H / 2, "", "", "ANY TARGET")

# ---- interface callouts (dots on the rightmost lane's arrows) --------------------------------------------
for y0, y1, text in ((SW_Y + SW_H, TR_Y, "tsr_transport\nAPI"),
                     (TR_Y + TR_H, PA_Y, "CSR ACCESS,\nDRAM BUFFERS"),
                     (PA_Y + PA_H, CO_Y, "CSR PORT, TWO\nCLOCKS, MEMORY\nPORTS"),
                     (CO_Y + CO_H, TE_Y, "tsr_sram_1r1w,\ntsr_mac_lane")):
    y = (y0 + y1) / 2
    f.callout(text, (RX, y), target=(LC[2], y))

# ---- legend and footnote -----------------------------------------------------------------------------------
f.legend(800, LEG_Y, [("host", "HOST SOFTWARE"), ("adapter", "PLATFORM ADAPTER"),
                      ("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                      ("io", "DATA MOVEMENT"), ("vendor", "VENDOR-SPECIFIC"),
                      ("neutral", "PLANNED / IDEA", True)], gap=28)
f.text(800, LEG_Y + 38, wrap(
    "A block across all three columns is the same on every target. The U200 transport, platform "
    "adapter and tech wrappers ran on the card. The dashed core blocks run on the simulator and "
    "are planned for the card (M6). Zynq / Kria and SoC / ASIC are ideas, not scheduled.", 1440),
       "note", anchor="middle", color=INK_SOFT)

f.save(scale=2)

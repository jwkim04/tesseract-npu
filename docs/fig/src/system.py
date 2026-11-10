"""Figure `system`: the Tesseract NPU core on the Alveo U200, host to DDR4.

Content follows docs/platform-u200.md §1 (system structure, adapter, weight traffic) and
docs/spec.md §2 (System structure): host stack, PCIe + XRT shell, platform adapter (control
bridge, one AXI4 bridge per channel), tsr_core (control core and issue path, SPM, x broadcast /
y gather, four MVM clusters, vector unit), DDR4 ch0..ch3 each feeding its own cluster.
The host chain is llama.cpp -> ggml-tesseract -> tsr_transport; the firmware image
(`tsr-fw.elf`, docs/design/01-execution-model.md §3) is loaded, booted and launched by the transport
over the same PCIe path (EX §2.1, §4). The adapter is `ap_ctrl_none` (EX §9.1): the control
bridge maps `s_axi_control` to the core's CSR port and there is no done / ap_done path.
SLR tags follow the Pblock plan of the release image tsr_u200_k4 (hw/u200/k4_major_blocks.tcl,
docs/platform-u200.md §8.1, docs/design/04-control-core.md §14.2). DDR4 channels from platforminfo:
bank0 SLR0, bank1 and bank2 SLR1, bank3 SLR2. Clusters 0 and 2 with their result assemblers in
SLR0, 1 and 3 in SLR2; the DDR0 bridge and arbiter in SLR0 and the DDR3 bridge and arbiter in SLR2,
beside their DDR controllers. The control plane (control core with TCMs and debug module, issue
path with its checkers, CSR block) is in its own Pblock inside SLR0; the whole vector unit in its
own Pblock inside SLR2; the SPM, the DMA, QACT and the arbiters of channels 1 and 2 in SLR1. The
XRT shell, the control bridge (the adapter's control logic), the bridges of channels 1 and 2 and
the x broadcast / y gather bus have no single region and carry no tag; the footnote says so.
Every crossing is a register pipe: per cluster the broadcast, weight-stream request / response and
gather job / start pipes (MVM_BC_PIPE, MVM_RES_PIPE = 2 on the U200, docs/rtl-mvm.md §14.2), the
reset copies, and the AXI slices of the outer channels. The weight-response pipe carries one beat
per cycle without backpressure, so the crossing keeps the channel's full rate
(docs/platform-u200.md §8.1).
Clocks: the core clock is 200 MHz and clk_rv 100 MHz, from ap_clk_2, the second platform clock
(docs/platform-u200.md §8.3, CC §11.1). The weight stream is 64 B per cycle: 64 B x 200 MHz =
12.8 GB/s per channel (docs/spec.md §1, design principles). DDR4 ch0 holds the op lists (OPL), the IO window and the
AUX tensors besides its quarter of the weights (EX §11).
Layout: the arrows of each channel column run 32 left of its center, so the SLR tag on the right of
a block's top edge stays clear of them.

Run: build/tools/svgvenv/bin/python docs/fig/src/system.py      (TSR_FIG_THEME=dark: system-dark.svg)
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 224, 1376                  # host and board panels; callouts in the 224 margins
CX0, CX1 = 256, 1344                  # shell strip and tsr_core region
SX0, COLW = 320, 240                  # per-channel slices: 4 columns of 240, x 320..1280
BW = 196                              # slice block width
COL = [SX0 + COLW * c + COLW / 2 for c in range(4)]          # 440, 680, 920, 1160
AX = [x - 32 for x in COL]            # arrows of column c (left of center, clear of the SLR tag)
LX, RX = PX0 / 2, (PX1 + W) / 2       # callout centers in the left / right margins

# SLR of each block under the Pblock plan (hw/u200/k4_major_blocks.tcl; DDR4 from platforminfo)
SLR_CL = ["SLR0", "SLR2", "SLR0", "SLR2"]       # clusters 0 and 2 in SLR0, 1 and 3 in SLR2
SLR_DDR = ["SLR0", "SLR1", "SLR1", "SLR2"]      # DDR[0] SLR0, DDR[1] / DDR[2] SLR1, DDR[3] SLR2
SLR_BR = ["SLR0", "", "", "SLR2"]               # bridges of channels 0 and 3 beside their DDR
SLR_CTRL, SLR_HUB, SLR_VU = "SLR0", "SLR1", "SLR2"   # control Pblock, SPM + DMA + QACT, VU Pblock

# rows (y)
HOST_Y, HOST_H = 56, 176
HB_Y, HB_H = 88, 112                  # host blocks
BOARD_Y = 296
SHELL_Y, SHELL_H = 328, 88
CB_Y, CB_H = 488, 112                 # control bridge (72 below the shell: AXI4-LITE clears its tag)
CORE_Y = 648
MID_Y, MID_H = 696, 152               # control core + issue path, SPM, vector unit
BUS_Y, BUS_H = 928, 72
CL_Y, CL_H = 1088, 112                # clusters
CORE_B = CL_Y + CL_H + 24
BR_Y, BR_H = 1296, 80                 # AXI4 bridges (72 below tsr_core: room for a leader)
DDR_Y, DDR_H = 1440, 120
BOARD_B = DDR_Y + DDR_H + 40
LEG_Y = BOARD_B + 48
H = LEG_Y + 96

f = Fig("system", W, H,
        title="Tesseract NPU core on the Alveo U200",
        desc="Host PC (llama.cpp, ggml-tesseract backend, tsr_transport, firmware image) talks "
             "over PCIe to the XRT shell of an Alveo U200. The platform adapter (control bridge "
             "and one AXI4 bridge per DDR4 channel) connects the shell and DDR4 to tsr_core: "
             "control core with issue path, 4 MiB SPM with DMA and QACT, x broadcast / y gather "
             "bus, vector unit and four MVM clusters of 128 MACs. Cluster c streams weights "
             "only from DDR4 channel c. Tags give each block's SLR under the Pblock plan of the "
             "release image: DDR4 ch0 SLR0, ch1 and ch2 SLR1, ch3 SLR2; clusters 0 and 2 SLR0, "
             "clusters 1 and 3 SLR2; control core and issue path SLR0; SPM, DMA and QACT SLR1; "
             "vector unit SLR2. Every SLR crossing is a register pipe that keeps the full "
             "weight-stream rate. The core clock is 200 MHz and clk_rv 100 MHz.")

# ---- host --------------------------------------------------------------------------------------
f.panel(PX0, HOST_Y, PX1 - PX0, HOST_H, label="HOST PC")
hb_w = [208, 232, 216, 208]
hb_gap = 72
hx0 = 800 - (sum(hb_w) + 3 * hb_gap) / 2
hx = [hx0]
for w_ in hb_w[:-1]:
    hx.append(hx[-1] + w_ + hb_gap)
hmid = HB_Y + HB_H / 2
f.box(hx[0], HB_Y, hb_w[0], HB_H, "host", "LLAMA.CPP", sub="graph, CPU ops",
      label_role="label_s")
f.box(hx[1], HB_Y, hb_w[1], HB_H, "host", "GGML-TESSERACT", sub="op lists, cache",
      label_role="label_s")
f.box(hx[2], HB_Y, hb_w[2], HB_H, "host", "TSR_TRANSPORT", sub="sim, mock, xrt",
      label_role="label_s")
f.box(hx[3], HB_Y, hb_w[3], HB_H, "host", "FIRMWARE\nIMAGE", sub="tsr-fw.elf",
      label_role="label_s")
for i in range(2):
    f.arrow([(hx[i] + hb_w[i] - 12, hmid), (hx[i + 1] + 12, hmid)])
f.arrow([(hx[3] + 12, hmid), (hx[2] + hb_w[2] - 12, hmid)])       # the transport reads the image
f.text(hx[2] + hb_w[2] + hb_gap / 2, hmid - 36, "LOAD", "callout", anchor="middle")

# ---- board ---------------------------------------------------------------------------------------
f.panel(PX0, BOARD_Y, PX1 - PX0, BOARD_B - BOARD_Y, label="ALVEO U200")

# XRT shell (spans the device; no SLR tag)
f.box(CX0, SHELL_Y, CX1 - CX0, SHELL_H, "vendor", "XRT SHELL",
      sub="registers · PCIe DMA to all four DDR4 channels")

# tsr_core region (outline only)
f.panel(CX0, CORE_Y, CX1 - CX0, CORE_B - CORE_Y, label="TSR_CORE", fill=None)

# control bridge, above the control core
cp_x, cp_w = SX0, 296
f.box(cp_x, CB_Y, cp_w, CB_H, "adapter", "CONTROL BRIDGE",
      sub="s_axi_control <-> CSR\nap_ctrl_none", label_role="label_s")

# control core with the issue path, SPM, vector unit
vu_x = SX0 + 4 * COLW - cp_w
spm_w = 232
spm_x = (cp_x + cp_w + vu_x) / 2 - spm_w / 2
f.box(cp_x, MID_Y, cp_w, MID_H, "control", "CONTROL CORE +\nISSUE PATH",
      sub="Ibex RV32IMC · CSR\nchecker · unit queues", tag=SLR_CTRL)
f.box(spm_x, MID_Y, spm_w, MID_H, "memory", "SPM", sub="4 MiB\nDMA · QACT", tag=SLR_HUB)
f.box(vu_x, MID_Y, cp_w, MID_H, "compute", "VECTOR UNIT",
      sub="norms, RoPE, softmax,\nactivations", tag=SLR_VU)

# x broadcast / y gather (the central part; each cluster's assembler sits with its cluster)
bus_x0, bus_x1 = SX0, SX0 + 4 * COLW
f.box(bus_x0, BUS_Y, bus_x1 - bus_x0, BUS_H, "io", "X BROADCAST · Y GATHER", cut=12)   # no tag: assemblers sit with their clusters

# clusters, AXI4 bridges, DDR4 channels
ddr_sub = ["OPL · IO · AUX\n1/4 of weights", "1/4 of weights", "1/4 of weights", "1/4 of weights"]
for c in range(4):
    x = COL[c] - BW / 2
    f.box(x, CL_Y, BW, CL_H, "compute", f"CLUSTER {c}", sub="128 MACs", tag=SLR_CL[c])
    f.box(x, DDR_Y, BW, DDR_H, "memory", f"DDR4 CH{c}", sub=ddr_sub[c], tag=SLR_DDR[c])
    # weight stream: DDR4 ch c -> AXI4 bridge c -> cluster c (the bridge is drawn over it)
    f.arrow([(AX[c], DDR_Y + 8), (AX[c], CL_Y + CL_H - 12)], fat=True)
    f.box(x, BR_Y, BW, BR_H, "adapter", "AXI4 BRIDGE", sub=f"m_axi_ch{c}", label_role="label_s",
          tag=SLR_BR[c])
    # broadcast down, gather up
    f.arrow([(AX[c], BUS_Y + BUS_H - 8), (AX[c], CL_Y + 12)], tail=True)

# arrows of the control and data paths
mx = cp_x + cp_w / 2
tx = hx[2] + hb_w[2] / 2
f.arrow([(tx, HB_Y + HB_H - 12), (tx, SHELL_Y + 12)], tail=True)               # PCIe
f.arrow([(mx, SHELL_Y + SHELL_H - 12), (mx, CB_Y + 8)], tail=True)         # AXI4-Lite
f.arrow([(mx, CB_Y + CB_H - 8), (mx, MID_Y + 8)], tail=True)              # CSR port, done
f.arrow([(cp_x + cp_w - 12, MID_Y + MID_H / 2), (spm_x + 12, MID_Y + MID_H / 2)], tail=True)
f.arrow([(vu_x + 12, MID_Y + MID_H / 2), (spm_x + spm_w - 12, MID_Y + MID_H / 2)], tail=True)
f.arrow([(spm_x + spm_w / 2, MID_Y + MID_H - 12), (spm_x + spm_w / 2, BUS_Y + 12)], tail=True)

# arrow names
f.text(mx + 44, SHELL_Y + SHELL_H + 28, "AXI4-LITE", "callout", valign="middle")
f.text(mx + 44, (CB_Y + CB_H + CORE_Y) / 2 - 2, "CSR PORT", "callout", valign="middle")

# callouts
pcie_y = (HOST_Y + HOST_H + BOARD_Y) / 2
f.callout("PCIe:\nLOAD · BOOT ·\nLAUNCH,\nWEIGHTS,\nOP LISTS", (RX, pcie_y), target=(tx, pcie_y))
xing_y = (BUS_Y + BUS_H + CL_Y) / 2   # on cluster 3's broadcast / gather arrow
f.callout("EVERY SLR\nCROSSING IS A\nREGISTER PIPE:\nMVM_BC_PIPE,\nMVM_RES_PIPE\n= 2 STAGES",
          (RX, xing_y), target=(AX[3], xing_y))
ws_y = (CORE_B + BR_Y) / 2 - 8        # on ch3's bridge -> cluster stretch (SLR2)
f.callout("WEIGHT STREAM:\n64 B × 200 MHz\n= 12.8 GB/s\nPER CHANNEL;\nSLR CROSSINGS\nKEEP FULL RATE",
          (RX, ws_y), target=(AX[3], ws_y))
clk_y = MID_Y + MID_H / 2
f.callout("CONTROL CORE:\nclk_rv 100 MHz\nFROM ap_clk_2;\nCORE CLOCK\n200 MHz", (LX, clk_y),
          target=(cp_x + 8, clk_y), via=())
ax_y = (BR_Y + BR_H + DDR_Y) / 2      # on ch0's DDR4 -> bridge stretch
f.callout("CH0, CH3: BRIDGE\nAND ARBITER SIT\nBESIDE THEIR\nDDR CONTROLLER", (LX, ax_y),
          target=(AX[0], ax_y))

# legend and footnote
f.legend(800, LEG_Y, [("host", "HOST SOFTWARE"), ("vendor", "VENDOR-SPECIFIC"),
                      ("adapter", "PLATFORM ADAPTER"), ("control", "CONTROL"),
                      ("compute", "COMPUTE"), ("memory", "MEMORY"), ("io", "DATA MOVEMENT")],
         gap=32)
f.text(800, LEG_Y + 38, wrap(
    "SLR tags follow the Pblock plan of the k4 image (hw/u200/k4_major_blocks.tcl): control "
    "core and issue path in the control Pblock of SLR0, vector unit in the VU Pblock of SLR2, "
    "SPM with DMA, QACT and the arbiters of channels 1 and 2 in SLR1. No tag: left to Vivado.",
    1440), "note", anchor="middle", color=INK_SOFT)

f.save()

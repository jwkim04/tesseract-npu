"""Figure `core`: inside tsr_core, two clock domains, the issue path and the units.

Content follows docs/design/04-control-core.md §2.1 (block diagram), §2.2 (hierarchy, domains), §3
(Ibex and the debug module), §4 (TCMs and buses), §5 (control window), §7 (issue path: push, issue
FIFO, queues and depths), §8 (dispatcher and checker), §10 (SPM banks and arbitration), §11 (clock domains
and CDC), §13.2 (module table); docs/rtl-core.md §2 for tsr_csr, tsr_dma, tsr_qact, tsr_spm and
tsr_mem_arb, and docs/rtl-mvm.md §2 for tsr_mvm:
- clk_rv band: Ibex RV32IMC with ITCM 128 KiB, DTCM 64 KiB, the debug module (upstream riscv-dbg,
  reached through the DMI window) and the control window `tsr_ctl_win` (issue port with credits
  and tickets, queue control, LQ / LCQ registers, log, timer; CC §5). Clock 100 MHz in the
  release image (CC §11.1);
- clock-domain crossings (`tsr_ctrl_cdc`, CC §11.2): the issue FIFO (32 entries), gray-coded
  counters, tagged handshakes, so the host and the SPM bridge never wait on clk_rv;
- core clock band (200 MHz in the release image, `docs/platform-u200.md` §8.3): `tsr_csr` (host CSR
  port with LQ, LCQ, the log ring and the image loader, CC §13.2), the issue FIFO, the checker
  `tsr_chk` (EX §8.1 order), the dispatcher with one in-order queue per unit (CC §7.4: DMA 8,
  QACT 4, MVM 4, VU 16, SYNC 4; unit 4 is reserved and absent), the units DMA (LD, ST, KVST),
  QACT, MVM (NUM_CH clusters; attention modes ATTQK, ATTPV), VU, and `tsr_spm` (4 MiB, 64 B
  lines; even / odd banks, 2R2W, with the RISC-V bridge as client 3, CC §10); one
  `tsr_mem_arb` per channel (sources DMA and weight stream; channel 0 also the trace writer,
  CC §13.2);
- the dashed bus under the dispatcher stands for the unit queues feeding every unit (CC §7.4).
Every block is drawn solid: the figure shows what the tree contains.
Arrow direction: weights flow from the arbiter into the clusters (as the `system` figure does).

Run: build/tools/svgvenv/bin/python docs/fig/src/core.py        (TSR_FIG_THEME=dark: core-dark.svg)
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK, INK_SOFT, Fig  # noqa: E402

W = 1600
PX0, PX1 = 224, 1376                  # tsr_core panel; callouts in the 224 margins
LX, RX = PX0 / 2, (PX1 + W) / 2       # callout centres
X0, X1 = 264, 1336                    # content inside the panel
BX0, BX1 = 248, 1352                  # the clock-domain bands
CX = 800                              # centre line

# ---- rows (y) ----------------------------------------------------------------------------------
PY0 = 72
A_Y = PY0 + 40                        # clk_rv band top
DBG_Y, DBG_H = A_Y + 56, 80
TCM_Y, TCM_H = DBG_Y + DBG_H + 40, 112
WIN_Y, WIN_H = TCM_Y + TCM_H + 40, 104
A_B = WIN_Y + WIN_H + 24
CDC_Y, CDC_H = A_B + 40, 64
B_Y = CDC_Y + CDC_H + 40              # core clock band top
B1_Y, B1_H = B_Y + 64, 128
BUS_GAP = 64
U_Y = B1_Y + B1_H + BUS_GAP           # units row top
UQ_H = 112                            # QACT, VU
SPM_Y, SPM_H = U_Y + UQ_H + 56, 136
U_B = SPM_Y + SPM_H                   # DMA, MVM stand from U_Y to here
ARB_Y, ARB_H = U_B + 96, 144
B_B = ARB_Y + ARB_H + 40
PY1 = B_B + 24
PORT_B = PY1 + 72
LEG_Y = PORT_B + 72
H = LEG_Y + 64

f = Fig("core", W, H,
        title="Inside tsr_core",
        desc="tsr_core in two clock domains. In the clk_rv domain (100 MHz): the Ibex "
             "RV32IMC control core with ITCM, DTCM, a debug module and the control window. Clock "
             "domain crossings connect it to the core clock domain (200 MHz): tsr_csr with the "
             "launch queue, completion ring and log ring, the issue FIFO, the checker tsr_chk "
             "and the dispatcher with one in-order queue per unit. The units are the DMA (LD, "
             "ST, KVST), QACT, the MVM clusters (with ATTQK and ATTPV) and the vector unit; all "
             "use the 4 MiB tsr_spm. One tsr_mem_arb per channel merges the DMA and "
             "weight-stream traffic onto that channel's memory port.")

f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0, label="TSR_CORE")

# ---- band A: clk_rv -------------------------------------------------------------------
f.band(BX0, A_Y, BX1 - BX0, A_B - A_Y, label="CLK_RV DOMAIN · CONTROL CORE · 100 MHz",
       label_xy=(BX0 + 24, A_Y + 36))
IB_W = 240
IB_X = CX - IB_W / 2
f.box(IB_X, DBG_Y, IB_W, DBG_H, "control", "DEBUG MODULE", sub="riscv-dbg · DMI", label_role="label_s")
f.box(IB_X, TCM_Y, IB_W, TCM_H, "control", "IBEX RV32IMC", sub="Zicsr · Zifencei\nno caches",
      label_role="label_s")
TCM_W = 256
f.box(IB_X - 96 - TCM_W, TCM_Y, TCM_W, TCM_H, "memory", "ITCM", sub="128 KiB\ninstructions",
      label_role="label_s")
f.box(IB_X + IB_W + 96, TCM_Y, TCM_W, TCM_H, "memory", "DTCM", sub="64 KiB\ndata, stack",
      label_role="label_s")
WIN_W = 480
f.box(CX - WIN_W / 2, WIN_Y, WIN_W, WIN_H, "control", "CONTROL WINDOW",
      sub="issue port · tickets · queue control\nLQ · LCQ · log · timer", label_role="label_s")
f.arrow([(CX, DBG_Y + DBG_H - 8), (CX, TCM_Y + 12)], tail=True)                 # debug <-> core
f.arrow([(IB_X + 12, TCM_Y + TCM_H / 2), (IB_X - 96 - 12, TCM_Y + TCM_H / 2)], tail=True)
f.arrow([(IB_X + IB_W - 12, TCM_Y + TCM_H / 2), (IB_X + IB_W + 96 + 12, TCM_Y + TCM_H / 2)], tail=True)
f.arrow([(CX, TCM_Y + TCM_H - 2), (CX, WIN_Y + 12)], tail=True)                 # core <-> window

# ---- clock-domain crossings ---------------------------------------------------------------------
f.box(X0, CDC_Y, X1 - X0, CDC_H, "io", "CLOCK-DOMAIN CROSSINGS", sub="",
      label_role="label_s", cut=12)
f.text(X0 + 24, CDC_Y + CDC_H / 2, "tsr_ctrl_cdc", "sub", valign="middle", color=INK)
f.arrow([(CX, WIN_Y + WIN_H - 8), (CX, CDC_Y + 12)], tail=True)

# ---- band B: core clock -------------------------------------------------------------------------
B_FULL = B_B - B_Y
f.band(BX0, B_Y, BX1 - BX0, B_FULL, label="CORE CLOCK DOMAIN · 200 MHz",
       label_xy=(BX1 - 24, B_Y + 36), anchor="end")
# row 1: tsr_csr, issue FIFO, checker, dispatcher
CSR_W, FIFO_W, CHK_W, DSP_W = 296, 192, 192, 248
csr_x = X0
fifo_x = csr_x + CSR_W + 48
chk_x = fifo_x + FIFO_W + 48
dsp_x = chk_x + CHK_W + 48
f.box(csr_x, B1_Y, CSR_W, B1_H, "control", "TSR_CSR",
      sub="host CSR port · ID\nLQ · LCQ · log · PERF")
f.box(fifo_x, B1_Y, FIFO_W, B1_H, "io", "ISSUE FIFO", sub="32 entries\ntile ops", label_role="label_s")
f.box(chk_x, B1_Y, CHK_W, B1_H, "control", "TSR_CHK", sub="checks every\ntile op", label_role="label_s")
f.box(dsp_x, B1_Y, DSP_W, B1_H, "control", "DISPATCHER", sub="one in-order queue\nper unit", label_role="label_s")
f.arrow([(csr_x + 44, CDC_Y + CDC_H - 8), (csr_x + 44, B1_Y + 12)], tail=True)       # LQ, LCQ, log
f.arrow([(fifo_x + FIFO_W / 2, CDC_Y + CDC_H - 8), (fifo_x + FIFO_W / 2, B1_Y + 12)])
f.arrow([(fifo_x + FIFO_W - 12, B1_Y + B1_H / 2), (chk_x + 12, B1_Y + B1_H / 2)])
f.arrow([(chk_x + CHK_W - 12, B1_Y + B1_H / 2), (dsp_x + 12, B1_Y + B1_H / 2)])

# row 2: units. DMA | QACT, VU over SPM | MVM
DMA_W, MVM_W, GAP = 232, 232, 64
dma_x = X0
mid_x0 = dma_x + DMA_W + GAP
mvm_x = X1 - MVM_W
mid_x1 = mvm_x - GAP
mid_w = mid_x1 - mid_x0
half = (mid_w - 48) / 2
qa_x, vu_x = mid_x0, mid_x1 - half
f.box(dma_x, U_Y, DMA_W, U_B - U_Y, "io", "TSR_DMA", sub="LD, ST\none engine for\nall channels")
f.box(dma_x + 24, U_B - 24 - 64, DMA_W - 48, 64, "io", "KVST", sub="",
      label_role="label_s", cut=10)
f.box(qa_x, U_Y, half, UQ_H, "compute", "TSR_QACT", sub="quantizes f32\nto QA (A32, A256)",
      label_role="label_s")
f.box(vu_x, U_Y, half, UQ_H, "compute", "VECTOR UNIT", sub="norms · RoPE ·\nsoftmax parts",
      label_role="label_s")
f.box(mid_x0, SPM_Y, mid_w, SPM_H, "memory", "TSR_SPM",
      sub="4 MiB · 64 B lines · 2R2W banks\nclients: DMA, QACT, MVM, VU\nand the RISC-V bridge")
# tsr_mvm: label at the top, the per-channel clusters as a stacked chip at the bottom
f.box(mvm_x, U_Y, MVM_W, U_B - U_Y, "compute")
f.label_block(mvm_x, U_Y + 20, MVM_W, 96, "TSR_MVM", "reads QA lines,\nwrites y lines")
f.box(mvm_x + 12, U_Y + 114, MVM_W - 24, 44, "compute", "ATTQK · ATTPV", sub="",
      label_role="label_s", cut=10)
CL_W, CL_H = 176, 80
CL_X, CL_Y = mvm_x + (MVM_W - CL_W) / 2 - 8, U_B - 28 - CL_H
f.box(CL_X, CL_Y, CL_W, CL_H, "neutral", "CLUSTER", sub="128 MACs", stack=2, label_role="label_s", cut=12)
f.tag(CL_X + CL_W + 16, CL_Y - 16, "× NUM_CH", anchor="end")

# the unit queues: a dashed bus from the dispatcher to every unit
bus_y = B1_Y + B1_H + BUS_GAP / 2
dsp_c = dsp_x + DSP_W / 2
ucs = [dma_x + 56, qa_x + half / 2, vu_x + 56, mvm_x + 56]
f.line([(dsp_c, B1_Y + B1_H - 6), (dsp_c, bus_y)], end=False, dashed=True)
f.hline(ucs[0], max(ucs[-1], dsp_c), bus_y, dashed=True, color=INK, sw=2)
for ux in ucs:
    f.line([(ux, bus_y), (ux, U_Y + 8)], dashed=True)
f.text(dsp_c - 24, bus_y - 16, "TILE OPS", "callout", anchor="end", valign="middle")

# units and the SPM, the DMA and the clusters and the arbiters
spm_m = SPM_Y + SPM_H / 2
f.arrow([(qa_x + half / 2, U_Y + UQ_H - 8), (qa_x + half / 2, SPM_Y + 12)], tail=True)
f.arrow([(vu_x + half / 2, U_Y + UQ_H - 8), (vu_x + half / 2, SPM_Y + 12)], tail=True)
f.arrow([(dma_x + DMA_W - 12, spm_m), (mid_x0 + 12, spm_m)], tail=True)
f.arrow([(mvm_x + 12, spm_m), (mid_x1 - 12, spm_m)], tail=True)

# ---- arbiters ------------------------------------------------------------------------------------
ARB_X1 = X1 - 16                      # room for the two stacked copies behind it
f.box(X0, ARB_Y, ARB_X1 - X0, ARB_H, "io", "TSR_MEM_ARB", stack=2,
      sub="one per channel · round robin per burst · ≤ 16 reads in flight\n"
          "sources: DMA, weight stream (cluster c) · channel 0 also the trace writer")
f.tag(CX, ARB_Y - 16, "× NUM_CH")
dma_c = dma_x + DMA_W / 2
f.arrow([(dma_c, U_B - 8), (dma_c, ARB_Y + 12)], tail=True)                     # LD / ST
ws_x = CL_X + CL_W / 2
f.arrow([(ws_x, ARB_Y + 12), (ws_x, CL_Y + CL_H - 12)], fat=True)               # weight streams
f.arrow([(CX, ARB_Y + ARB_H - 12), (CX, PORT_B)], tail=True, fat=True)           # memory ports
f.text(dma_c + 44, (U_B + ARB_Y) / 2, "DMA: LD · ST", "callout", valign="middle")
f.text(CX + 64, (PY1 + PORT_B) / 2 - 12, "NUM_CH MEMORY PORTS", "callout", valign="middle")
f.text(CX + 64, (PY1 + PORT_B) / 2 + 16, "five valid/ready streams each · 512-bit data",
       "note", valign="middle", color=INK_SOFT)

# ---- external CSR port, left margin; callouts ------------------------------------------------------
csr_y = B1_Y + B1_H / 2
f.line([(LX + 8, csr_y), (csr_x, csr_y)], start=True)
f.text(LX, csr_y - 36, "CSR PORT", "callout", anchor="middle", valign="middle")
f.callout("CODE AND DATA\nIN TCMs;\nNO CACHES", (LX, TCM_Y + TCM_H / 2), target=(IB_X - 96 - TCM_W + 8, TCM_Y + TCM_H / 2))
f.callout("QUEUE DEPTHS:\nDMA 8, QACT 4,\nMVM 4, VU 16,\nSYNC 4", (RX, B1_Y + B1_H / 2),
          target=(dsp_x + DSP_W - 8, B1_Y + B1_H / 2))
f.callout("ONLY THE CONTROL\nCORE ISSUES\nTILE OPS; THE\nHOST LAUNCHES\nKERNELS", (RX, WIN_Y + WIN_H / 2),
          target=(CX + WIN_W / 2 - 8, WIN_Y + WIN_H / 2))
wy = (U_B + ARB_Y) / 2
f.text(ws_x - 44, wy, "WS", "callout", anchor="end", valign="middle")
f.callout("WEIGHT STREAMS:\nCLUSTER c READS\nONLY CHANNEL c", (RX, wy), target=(ws_x, wy))
ay = ARB_Y + ARB_H / 2
f.callout("RESPONSES\nROUTED BY ID;\nEVERY STREAM\nTHROUGH A\nREGISTER SLICE", (LX, ay),
          target=(X0 + 72, ay))

# ---- legend ------------------------------------------------------------------------------------------
f.legend(800, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                      ("io", "DATA MOVEMENT")])

f.save(scale=1)

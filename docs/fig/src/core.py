"""Figure `core`: inside tsr_core, the blocks of the vendor-independent core and their paths.

Content follows README "Inside tsr_core" (the Mermaid block and the block table), "Core boundary"
and the dispatch protocol paragraph:
- core boundary: CSR port <-> tsr_csr, tsr_csr -> done, tsr_mem_arb <-> NUM_CH memory ports
  (valid/ready; README "Core boundary"),
- tsr_csr (CTRL, STATUS, ERR, ARGs, 64-bit PERF counters) <-> tsr_cmdproc,
- tsr_fetch (prefetch from ch0, 32 instructions) -> tsr_cmdproc (instructions),
- tsr_cmdproc (decode, all checks, R0-R15, LI, ADDI, one instruction in flight) dispatches to
  tsr_dma (LD, ST), tsr_qact (f32 -> QA) and tsr_mvm (NUM_CH clusters, reads QA lines, writes y
  lines),
- tsr_spm (4 MiB, 1R + 1W port, clients DMA, QACT, MVM) <-> each of the three units,
- tsr_mem_arb, one per channel (ch0 sources FETCH, DMA, WS0; ch c sources DMA, WS c): DMA <->
  arbiter, weight streams -> MVM, fetch reads -> tsr_fetch.
Arrow direction: the Mermaid block draws requests (MVM ==> ARB, FET -.-> ARB); this figure draws
the data direction instead (weights flow from the arbiter into the clusters, instructions from the
arbiter into tsr_fetch), as the `system` figure does. LD/ST is two-way because LD reads and ST
writes DRAM (README tsr_dma row).
Additions beyond the Mermaid block, with their sources: "two instructions per beat" (README
tsr_fetch row), "one engine for LD/ST on all channels" (README tsr_dma row), A32 / A256 (README ISA
table, QACT sub 0 / 1), "64 B lines" (README At a glance), "round robin per burst" and "at most
RD_OUTSTANDING (16) reads in flight per channel, summed over all sources" (README tsr_mem_arb row
and memory port rules), "cluster c reads weights only from channel c" (README design principles),
the dispatch rule "one unit at a time, the next instruction only after the unit's done" (README
dispatch protocol), "five valid/ready streams" per memory port and its 512-bit data (README Core
boundary), the platform adapter as the other side of the ports (README Four layers), 128 MACs per
cluster (README At a glance; the current RTL requires LANES = 128, README porting notes).
RTL check: rtl/core/tsr_core.sv (instances u_csr, u_cmdproc, u_fetch, u_dma, u_qact, u_mvm,
u_spm, g_ch[c].u_arb; SPM clients 0 DMA, 1 QACT, 2 MVM; FETCH source on channel 0 only).

Run: build/tools/svgvenv/bin/python docs/fig/src/core.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig  # noqa: E402

W = 1600
PX0, PX1 = 224, 1376                  # tsr_core panel; callouts in the 224 margins
LX, RX = PX0 / 2, (PX1 + W) / 2       # callout centres
X0, X1 = 264, 1336                    # content inside the panel
FX = 288                              # fetch path, left of every block

# columns of the unit row: DMA | QACT over SPM | MVM
DMA_X, DMA_W = 320, 232
MID_X, MID_W = 632, 336
MVM_X, MVM_W = 1048, 288
DMA_C, MID_C, MVM_C = DMA_X + DMA_W / 2, MID_X + MID_W / 2, MVM_X + MVM_W / 2

# rows (y)
PY0 = 72
TOP_Y, TOP_H = 128, 152               # tsr_fetch, tsr_csr
CP_Y, CP_H = TOP_Y + TOP_H + 80, 112  # tsr_cmdproc
U_Y = CP_Y + CP_H + 96                # unit row top
QA_H = 120                            # tsr_qact
SPM_Y, SPM_H = U_Y + QA_H + 80, 152   # tsr_spm
U_B = SPM_Y + SPM_H                   # unit row bottom (DMA, MVM are tall)
ARB_Y, ARB_H = U_B + 112, 144
PY1 = ARB_Y + ARB_H + 40
PORT_B = PY1 + 72
LEG_Y = PORT_B + 72
H = LEG_Y + 56

f = Fig("core", W, H,
        title="Inside tsr_core",
        desc="tsr_core has three external interfaces: the CSR port, done and one memory port per "
             "channel. tsr_csr holds the CSRs and PERF counters and starts tsr_cmdproc, which "
             "decodes the instructions that tsr_fetch prefetches from channel 0, runs every check "
             "and dispatches one instruction at a time to tsr_dma, tsr_qact or tsr_mvm. All three "
             "units read and write the 4 MiB tsr_spm. One tsr_mem_arb per channel merges the fetch "
             "(channel 0 only), DMA and weight-stream sources onto that channel's memory port.")

# ---- panel ---------------------------------------------------------------------------------
f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0, label="TSR_CORE")

# ---- top row: fetch, CSR; command processor ----------------------------------------------------
FE_X, FE_W = DMA_X, 280
CSR_X, CSR_W = MVM_X, MVM_W
f.box(FE_X, TOP_Y, FE_W, TOP_H, "control", "TSR_FETCH", sub="prefetch from ch0\n32 instructions")
f.box(CSR_X, TOP_Y, CSR_W, TOP_H, "control", "TSR_CSR",
      sub="CTRL, STATUS, ERR\nARG0–ARG7\n64-bit PERF counters")
f.box(DMA_X, CP_Y, X1 - DMA_X, CP_H, "control", "TSR_CMDPROC",
      sub="decode · all checks · R0–R15, LI, ADDI · one instruction in flight")

# ---- unit row ------------------------------------------------------------------------------------
f.box(DMA_X, U_Y, DMA_W, U_B - U_Y, "io", "TSR_DMA", sub="LD, ST\none engine for\nall channels")
f.box(MID_X, U_Y, MID_W, QA_H, "compute", "TSR_QACT", sub="quantizes f32\nto QA (A32, A256)")
f.box(MID_X, SPM_Y, MID_W, SPM_H, "memory", "TSR_SPM",
      sub="4 MiB, 64 B lines\n1R + 1W port\nclients: DMA, QACT, MVM")
# tsr_mvm: label at the top, the per-channel clusters as a stacked chip at the bottom
f.box(MVM_X, U_Y, MVM_W, U_B - U_Y, "compute")
f.label_block(MVM_X, U_Y + 24, MVM_W, 112, "TSR_MVM", "reads QA lines,\nwrites y lines")
CL_W, CL_H = 192, 96
CL_X, CL_Y = MVM_C - CL_W / 2 - 8, U_B - 40 - CL_H
f.box(CL_X, CL_Y, CL_W, CL_H, "neutral", "CLUSTER", sub="128 MACs", stack=2, label_role="label_s",
      cut=12)
f.tag(CL_X + CL_W + 16, CL_Y - 16, "× NUM_CH", anchor="end")

# ---- arbiters --------------------------------------------------------------------------------------
ARB_X1 = X1 - 16                      # room for the two stacked copies behind it
f.box(X0, ARB_Y, ARB_X1 - X0, ARB_H, "io", "TSR_MEM_ARB", stack=2,
      sub="one per channel · round robin per burst · ≤ 16 reads in flight\n"
          "ch0 sources: FETCH, DMA, WS0 · ch c sources: DMA, WS c")
f.tag(MID_C, ARB_Y - 16, "× NUM_CH")   # on the back copy, clear of the incoming arrows

# ---- arrows ----------------------------------------------------------------------------------------
fe_c = FE_X + FE_W / 2
f.arrow([(fe_c, TOP_Y + TOP_H - 8), (fe_c, CP_Y + 12)])                        # instructions
f.arrow([(MVM_C, TOP_Y + TOP_H - 8), (MVM_C, CP_Y + 12)], tail=True)           # run control
for x, y1 in ((DMA_C, U_Y), (MID_C, U_Y), (MVM_C, U_Y)):                      # dispatch
    f.arrow([(x, CP_Y + CP_H - 12), (x, y1 + 12)])
spm_m = SPM_Y + SPM_H / 2
f.arrow([(DMA_X + DMA_W - 12, spm_m), (MID_X + 12, spm_m)], tail=True)        # DMA <-> SPM
f.arrow([(MVM_X + 12, spm_m), (MID_X + MID_W - 12, spm_m)], tail=True)        # MVM <-> SPM
f.arrow([(MID_C, U_Y + QA_H - 12), (MID_C, SPM_Y + 12)], tail=True)           # QACT <-> SPM
f.arrow([(DMA_C, U_B - 12), (DMA_C, ARB_Y + 12)], tail=True)                  # LD / ST
ws_x = CL_X + CL_W / 2
f.arrow([(ws_x, ARB_Y + 12), (ws_x, CL_Y + CL_H - 12)], fat=True)             # weight streams
fe_m = TOP_Y + TOP_H / 2
f.line([(FX, ARB_Y - 16), (FX, fe_m), (FE_X, fe_m)])                       # fetch reads
f.arrow([(800, ARB_Y + ARB_H - 12), (800, PORT_B)], tail=True, fat=True)       # memory ports

# arrow names
f.text(fe_c + 44, (TOP_Y + TOP_H + CP_Y) / 2, "INSTRUCTIONS", "callout", valign="middle")
f.text(DMA_C + 44, (U_B + ARB_Y) / 2, "DMA: LD · ST", "callout", valign="middle")
f.text(FX + 20, (U_B + ARB_Y) / 2, "FETCH", "callout", valign="middle")
f.text(800 + 64, (PY1 + PORT_B) / 2 - 12, "NUM_CH MEMORY PORTS", "callout", valign="middle")
f.text(800 + 64, (PY1 + PORT_B) / 2 + 16, "five valid/ready streams each · 512-bit data",
       "note", valign="middle", color=INK_SOFT)

# ---- external CSR port and done, on the right margin ------------------------------------------------
csr_p = TOP_Y + 44
done_y = TOP_Y + TOP_H - 44
f.line([(RX - 64, csr_p), (CSR_X + CSR_W, csr_p)], start=True)
f.text(RX - 48, csr_p, "CSR PORT", "callout", valign="middle")
f.line([(CSR_X + CSR_W, done_y), (RX - 64, done_y)])
f.text(RX - 48, done_y, "DONE", "callout", valign="middle")

# ---- callouts ----------------------------------------------------------------------------------------
dy = (CP_Y + CP_H + U_Y) / 2
f.callout("DISPATCH:\nONE UNIT AT A\nTIME, NEXT ONLY\nAFTER ITS DONE", (RX, dy), target=(MVM_C, dy))
wy = (U_B + ARB_Y) / 2
f.text(ws_x - 44, wy, "WS", "callout", anchor="end", valign="middle")
f.callout("WEIGHT STREAMS:\nCLUSTER c READS\nONLY CHANNEL c", (RX, wy), target=(ws_x, wy))
fy = (U_Y + U_B) / 2
f.callout("INSTRUCTION\nFETCH: CH0\nONLY, TWO PER\n64 B BEAT", (LX, fy), target=(FX, fy))
ay = ARB_Y + ARB_H / 2
f.callout("RESPONSES\nROUTED BY ID;\nEVERY STREAM\nTHROUGH A\nREGISTER SLICE", (LX, ay),
          target=(X0 + 72, ay))           # inside the arbiter, clear of the FETCH line

# ---- legend ------------------------------------------------------------------------------------------
f.legend(800, LEG_Y, [("control", "CONTROL"), ("compute", "COMPUTE"), ("memory", "MEMORY"),
                      ("io", "DATA MOVEMENT")])

f.save(scale=2)

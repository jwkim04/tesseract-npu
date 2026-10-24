"""Figure `verify`: the verification oracle chain, from the references to the RTL and the card.

Content follows README "Verification" (the oracle-chain Mermaid block, its caption, the stage /
oracle / pass-condition table and the mutation-testing bullets) and "Trace sets and RTL replay":
- references: the ggml CPU backend, libtsr_ref (the one C reference of the numerics) and the
  numpy spec model (written from the spec text only);
- libtsr_ref -> ISA simulator -> ggml-tesseract on TSR_DEVICE=sim;
- checks: test-backend-ops (NMSE vs CPU per op) and klcheck (model KL vs the CPU floor) compare
  the backend with the CPU backend; numcheck compares it bit-exactly with libtsr_ref and the numpy
  model and checks a proven fp32 error bound;
- traces: tsr-trace-gen (directed, random; results from the simulator) and TSR_TRACE dumps of
  real runs -> trace sets -> bit-exact RTL replay (cocotb + Verilator block and core tests,
  fastreplay C++ Verilator replay); mutation campaigns check the RTL tests (solid: they already
  run; the Mermaid block drew this edge dotted);
- the U200 hardware replays the same trace sets later (dashed, pending the card).
Additions beyond the Mermaid block, with their sources (README "Verification"): the pass
conditions in the check blocks (stage table: "mean KL <= 3 x floor KL, top-1 agreement >= 80 %"),
the four memory regimes and the 0 failed runs of every replayed set (RTL replay results table),
"every test must be shown to fail on at least one deliberate mutant" and the M3 core campaign
counts (31 mutants, 28 killed, 3 equivalent), and "trace replay bit-exact, same token sequence"
for the hardware (stage table, XRT host runner pending), and the names of the trace sets
(m2, gen, m3, model, tbo; trace-set table). The cocotb box separates core trace replay from the
block tests, whose oracle is libtsr_ref (stage table: "RTL block | shared C reference
functions"); the "every trace, all four regimes" callout is the fastreplay result (README "RTL
replay results (tb/fastreplay, every trace in all four memory regimes)").
Colours: the references are white (neutral), so lavender keeps its data-movement meaning from the
architecture figures; planned is a dashed pale swatch, as in every figure.

Run: build/tools/svgvenv/bin/python docs/fig/src/verify.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig  # noqa: E402

W = 1600
PX0, PX1 = 56, 1376                   # panels; callouts in the right margin
RX = (PX1 + W) / 2                    # callout centre, right margin

# columns of the upper panel (x)
SIM_X, SIM_W = 168, 232
NC_X, NC_W = 456, 288
KL_X, KL_W = 800, 192
TB_X, TB_W = 1040, 280
REF_X, REF_W = 168, 384
NPY_X, NPY_W = 600, 192
CPU_X, CPU_W = 840, 480
BE_X = 280
GEN_AX = 216                          # SIM -> tsr-trace-gen, straight down left of the backend
BE_AX = 344                           # SIM -> backend
# columns of the lower panel (x)
GEN_X, GEN_W = 88, 256
DMP_X, DMP_W = 448, 304
SET_X, SET_W = 88, 664
RTL_X, RTL_W = 856, 464
MUT_X, MUT_W = 1048, 272

# rows (y)
P1_Y = 56
R_Y, R_H = 104, 136                   # references
C_Y, C_H = 312, 144                   # checks and the simulator
B_Y, B_H = C_Y + C_H + 80, 104        # backend
P1_B = B_Y + B_H + 40
P2_Y = P1_B + 56
T_Y, T_H = P2_Y + 56, 120             # trace producers, mutation campaigns
S_Y, S_H = T_Y + T_H + 80, 120        # trace sets; the RTL replay region starts here
V_Y, V_H = S_Y + S_H + 80, 120        # hardware, under the trace sets
RTL_B = V_Y + V_H
P2_B = RTL_B + 48
LEG_Y = P2_B + 48
H = LEG_Y + 48

f = Fig("verify", W, H,
        title="Tesseract verification oracle chain",
        desc="References (ggml CPU backend, libtsr_ref, numpy spec model) check the ISA "
             "simulator and the ggml-tesseract backend on TSR_DEVICE=sim through "
             "test-backend-ops, klcheck and numcheck. tsr-trace-gen and TSR_TRACE dumps make "
             "trace sets whose results come from the simulator; the RTL replays them bit-exactly "
             "in cocotb + Verilator and fastreplay, mutation campaigns check those tests, and the "
             "U200 hardware will replay the same traces once the card is installed.")

# ---- upper panel: the simulator against the references ------------------------------------------
f.panel(PX0, P1_Y, PX1 - PX0, P1_B - P1_Y, label="SIMULATOR VS REFERENCES")

f.box(REF_X, R_Y, REF_W, R_H, "neutral", "LIBTSR_REF", sub="the one C reference\nof the numerics")
f.box(NPY_X, R_Y, NPY_W, R_H, "neutral", "NUMPY SPEC\nMODEL", sub="spec text only")
f.box(CPU_X, R_Y, CPU_W, R_H, "neutral", "GGML CPU BACKEND")

f.box(SIM_X, C_Y, SIM_W, C_H, "host", "ISA\nSIMULATOR")
f.box(NC_X, C_Y, NC_W, C_H, "control", "NUMCHECK",
      sub="bit-exact vs\nlibtsr_ref and numpy,\nerror bound")
f.box(KL_X, C_Y, KL_W, C_H, "control", "KLCHECK", sub="model KL vs\nCPU floor")
f.box(TB_X, C_Y, TB_W, C_H, "control", "TEST-BACKEND-OPS", sub="NMSE vs CPU\nper op")

f.box(BE_X, B_Y, TB_X + TB_W - BE_X, B_H, "host", "GGML-TESSERACT", sub="TSR_DEVICE=sim")

sim_cx = SIM_X + SIM_W / 2
nc_a = NC_X + 48                                   # libtsr_ref -> numcheck
npy_cx = NPY_X + NPY_W / 2                         # numpy -> numcheck
nc_cx = NC_X + NC_W / 2
kl_cx, tb_cx = KL_X + KL_W / 2, TB_X + TB_W / 2
R_B, C_B = R_Y + R_H, C_Y + C_H
f.arrow([(sim_cx, R_B - 12), (sim_cx, C_Y + 12)])                 # libtsr_ref -> simulator
f.arrow([(nc_a, R_B - 12), (nc_a, C_Y + 12)])
f.arrow([(npy_cx, R_B - 12), (npy_cx, C_Y + 12)])
f.arrow([(kl_cx, R_B - 12), (kl_cx, C_Y + 12)])                   # CPU -> klcheck
f.arrow([(tb_cx, R_B - 12), (tb_cx, C_Y + 12)])                   # CPU -> test-backend-ops
f.arrow([(BE_AX, C_B - 12), (BE_AX, B_Y + 12)])                   # simulator -> backend
for x in (nc_cx, kl_cx, tb_cx):                                     # backend -> checks
    f.arrow([(x, B_Y + 12), (x, C_B - 12)])

# ---- lower panel: the RTL against the simulator ---------------------------------------------------
f.panel(PX0, P2_Y, PX1 - PX0, P2_B - P2_Y)
f.tag(GEN_AX + 40, P2_Y, "RTL VS SIMULATOR", anchor="start")   # clear of the sim -> gen shaft

f.box(GEN_X, T_Y, GEN_W, T_H, "host", "TSR-TRACE-GEN", sub="directed, random")
f.box(DMP_X, T_Y, DMP_W, T_H, "memory", "TSR_TRACE DUMPS", sub="of real runs")
f.box(SET_X, S_Y, SET_W, S_H, "memory", "TRACE SETS",
      sub="results from the simulator\nm2 · gen · m3 · model · tbo")
f.box(SET_X, V_Y, SET_W, V_H, "neutral", "U200 HARDWARE",
      sub="trace replay bit-exact, same token\nsequence · XRT host runner",
      planned=True, tag="PENDING THE CARD")

f.panel(RTL_X, S_Y, RTL_W, RTL_B - S_Y, label="RTL REPLAY · BIT-EXACT", fill=None, cut=16)
in_x, in_w, in_h = RTL_X + 40, RTL_W - 80, 104
f.box(in_x, S_Y + 40, in_w, in_h, "compute", "COCOTB + VERILATOR",
      sub="core: trace replay\nblocks: vs libtsr_ref")
f.box(in_x, RTL_B - 32 - in_h, in_w, in_h, "compute", "FASTREPLAY",
      sub="C++ Verilator replay")
f.box(MUT_X, T_Y, MUT_W, T_H, "control", "MUTATION\nCAMPAIGNS")

# arrows
s_cy = S_Y + S_H / 2
f.arrow([(GEN_AX, C_B - 12), (GEN_AX, T_Y + 12)])                 # simulator -> tsr-trace-gen
f.arrow([(nc_cx, B_Y + B_H - 12), (nc_cx, T_Y + 12)])             # backend -> TSR_TRACE dumps
f.arrow([(GEN_AX, T_Y + T_H - 12), (GEN_AX, S_Y + 12)])            # gen -> sets
f.arrow([(nc_cx, T_Y + T_H - 12), (nc_cx, S_Y + 12)])              # dumps -> sets
f.arrow([(SET_X + SET_W - 12, s_cy), (RTL_X + 12, s_cy)], fat=True)  # sets -> RTL replay
mut_cx = MUT_X + MUT_W / 2
f.arrow([(mut_cx, T_Y + T_H - 12), (mut_cx, S_Y + 12)])            # mutants -> RTL tests
set_cx = SET_X + SET_W / 2
f.arrow([(set_cx, S_Y + S_H - 12), (set_cx, V_Y + 12)], dashed=True)  # sets -> hardware

# callouts
f.callout("THE GGML CPU\nBACKEND\nVALIDATES THE\nSIMULATOR", (RX, R_Y + R_H / 2),
          target=(CPU_X + CPU_W, R_Y + R_H / 2))
f.callout("THE BACKEND\nUSES THE\nSIMULATOR AS\nITS DEVICE", (RX, B_Y + B_H / 2),
          target=(TB_X + TB_W, B_Y + B_H / 2))
f.callout("EVERY TEST\nMUST FAIL ON\n≥ 1 MUTANT", (RX, T_Y + T_H / 2),
          target=(MUT_X + MUT_W, T_Y + T_H / 2))
fr_cy = RTL_B - 32 - in_h / 2                     # the fastreplay box
f.callout("FASTREPLAY:\nEVERY TRACE, ALL\nFOUR MEMORY\nREGIMES, 0 FAILED", (RX, fr_cy),
          target=(in_x + in_w, fr_cy))

# legend
f.legend(W / 2, LEG_Y, [("neutral", "REFERENCE"), ("host", "HOST SOFTWARE"),
                        ("control", "CHECK"), ("memory", "TRACES"), ("compute", "RTL"),
                        ("neutral", "PLANNED", True)])

f.save(scale=2)          # preview PNG at 2x for checking; _gh.png at 880 px

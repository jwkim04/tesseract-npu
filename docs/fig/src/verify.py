"""Figure `verify`: the verification oracle chain, from the references to the RTL and the card.

Content follows docs/spec.md §17 (Verification strategy: stages, oracles, trace format, block-level
coverage) and docs/design/06-software-verification.md §10.1 (oracle chain), §10.2 (ctest plan), §10.4
(pass criteria), §5 (trace format). Three panels.
1. Simulator and backend against the references (docs/design/06-software-verification.md §4.2, §4.5,
   §10.1; docs/numcheck.md, docs/klcheck.md):
   - references: Spike (oracle of the RV32 interpreter `tsr_rv`: riscv-tests, riscv-arch-test, a
     retire-by-retire comparison of random programs), libtsr_ref (the one C reference of the
     numerics, with tsr_ref_vu and tsr_ref_at), the numpy spec models (written from the spec text
     only) and the ggml CPU backend;
   - libtsr_ref -> simulator; Spike -> its RV32 interpreter; the numpy models check libtsr_ref and
     numcheck (bit-exact, plus a proven fp32 error bound); the ggml CPU backend is the floor of
     klcheck and test-backend-ops;
   - the ggml-tesseract backend runs on TSR_DEVICE=sim, so the simulator is its device.
2. Firmware kernels (SW §6.8, §7, §10.1; ctests fw_*): the C++ references of the composite kernels
   (sw/ctrl/ref) and the ggml CPU backend (layer tests) check the RISC-V build running on the
   simulator, and the host build of the same kernels; the firmware tests run 8 schedules with the
   race detector fatal; `tsr-selftest` compares a device (sim, mock or the card) with the
   in-process simulator shadow (SW §9.5).
3. Traces (SW §5, §5.4): tsr-trace-gen (directed, random) and TSR_TRACE dumps of real runs make
   trace sets (m2, gen, m3, vu, at, ctrl, model-v1conv; traces/README.md) whose results come
   from the simulator; they replay on the simulator kinds (`tsr-trace-replay`, full or tile-only,
   N schedules) and through the `replay` kernel (SW §7.7), also on the card (`tsr-selftest --device
   xrt --traces`, docs/xrt-host.md §5). The RTL checks: the tile-only replay of `push.bin` into the
   issue FIFO under Verilator (`tb/fastreplay`, `tb/core`; docs/design/04-control-core.md §15.3,
   §15.4), the Ibex lockstep against the RV32 interpreter, and the mutation campaigns of the RTL
   tests (docs/design/04-control-core.md §15.5, docs/rtl-mvm.md §13.3).
Colors: the references are white (neutral), so lavender keeps its data-movement meaning from the
architecture figures; code under test is host-software butter; checks are mint, traces salmon, the
RTL sky (the key below). Every block is drawn solid.

Run: build/tools/svgvenv/bin/python docs/fig/src/verify.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import Fig  # noqa: E402

W = 1600
PX0, PX1 = 56, 1376                   # panels; callouts in the right margin
RX = (PX1 + W) / 2                    # callout center, right margin

# ---- panel 1: the simulator and the backend against the references -------------------------------
P1_Y = 56
R_Y, R_H = 104, 136                   # references
C_Y, C_H = 312, 152                   # simulator and checks
B_Y, B_H = C_Y + C_H + 80, 104        # backend
P1_B = B_Y + B_H + 40
SPK_X, SPK_W = 88, 232
REF_X, REF_W = 360, 400
NPY_X, NPY_W = 800, 232
CPU_X, CPU_W = 1072, 272
SIM_X, SIM_W = 88, 432
NC_X, NC_W = 568, 312
KL_X, KL_W = 920, 200
TB_X, TB_W = 1160, 184

# ---- panel 2: the firmware kernels ---------------------------------------------------------------
P2_Y = P1_B + 56
FR_Y, FR_H = P2_Y + 56, 120           # references of the kernels
FT_Y, FT_H = FR_Y + FR_H + 64, 144    # firmware tests
FB_Y, FB_H = FT_Y + FT_H + 64, 112    # the two builds
P2_B = FB_Y + FB_H + 40

# ---- panel 3: traces, replays, RTL, card ----------------------------------------------------------
P3_Y = P2_B + 56
T_Y, T_H = P3_Y + 56, 120             # trace producers, mutation campaigns
S_Y, S_H = T_Y + T_H + 72, 120        # trace sets
RP_Y, RP_H = S_Y + S_H + 64, 120      # replays on the simulator kinds and the replay kernel
V_Y, V_H = RP_Y + RP_H + 64, 112      # the card
P3_B = V_Y + V_H + 40
GEN_X, GEN_W = 88, 256
DMP_X, DMP_W = 448, 304
SET_X, SET_W = 88, 664
RTL_X, RTL_W = 856, 488
MUT_X, MUT_W = 1048, 296

LEG_Y = P3_B + 56
H = LEG_Y + 64

f = Fig("verify", W, H,
        title="Tesseract verification oracle chain",
        desc="References (Spike, libtsr_ref with the VU and AT references, the numpy spec models, "
             "the ggml CPU backend) check the simulator, its RV32 interpreter and the "
             "ggml-tesseract backend on TSR_DEVICE=sim through numcheck, klcheck and "
             "test-backend-ops. The firmware kernels, built for RISC-V and for the host, are "
             "checked against C++ references and the ggml CPU backend in firmware tests under 8 "
             "schedules with the race detector; tsr-selftest compares a device with the in-process "
             "simulator. tsr-trace-gen and TSR_TRACE dumps make trace sets whose results come from "
             "the simulator; they replay on the simulator kinds and through the replay kernel. "
             "The RTL replay (tile-only replay into the issue FIFO and the Ibex lockstep) and the "
             "RTL mutation campaigns check the RTL, and the traces replay on the U200 card "
             "through the replay kernel.")

# ---- panel 1 -----------------------------------------------------------------------------------
f.panel(PX0, P1_Y, PX1 - PX0, P1_B - P1_Y, label="SIMULATOR AND BACKEND VS REFERENCES")
f.box(SPK_X, R_Y, SPK_W, R_H, "neutral", "SPIKE", sub="oracle of the\nRV32 interpreter")
f.box(REF_X, R_Y, REF_W, R_H, "neutral", "LIBTSR_REF",
      sub="the C reference of numerics\ntsr_ref_vu · tsr_ref_at")
f.box(NPY_X, R_Y, NPY_W, R_H, "neutral", "NUMPY SPEC\nMODELS", sub="spec text only")
f.box(CPU_X, R_Y, CPU_W, R_H, "neutral", "GGML CPU BACKEND")

f.box(SIM_X, C_Y, SIM_W, C_H, "host", "SIMULATOR",
      sub="units · issue queues · VU, AT\nRV32 interpreter · race detector")
f.box(NC_X, C_Y, NC_W, C_H, "control", "NUMCHECK",
      sub="bit-exact vs libtsr_ref\nand numpy, every weight\nMUL_MAT; error bound")
f.box(KL_X, C_Y, KL_W, C_H, "control", "KLCHECK", sub="model KL vs\nCPU floor")
f.box(TB_X, C_Y, TB_W, C_H, "control", "TEST-\nBACKEND-OPS", sub="NMSE vs CPU\nper op", label_role="label_s")
f.box(SIM_X, B_Y, TB_X + TB_W - SIM_X, B_H, "host", "GGML-TESSERACT",
      sub="the llama.cpp backend on TSR_DEVICE=sim: the simulator is its device")

R_B, C_B = R_Y + R_H, C_Y + C_H
f.arrow([(SPK_X + SPK_W / 2, R_B - 12), (SPK_X + SPK_W / 2, C_Y + 12)])          # Spike -> sim
f.arrow([(REF_X + 100, R_B - 12), (REF_X + 100, C_Y + 12)])                      # libtsr_ref -> sim
f.arrow([(REF_X + 320, R_B - 12), (REF_X + 320, C_Y + 12)])                      # libtsr_ref -> numcheck
f.arrow([(NPY_X + 50, R_B - 12), (NPY_X + 50, C_Y + 12)])                        # numpy -> numcheck
f.arrow([(KL_X + 170, R_B - 12), (KL_X + 170, C_Y + 12)])                         # CPU -> klcheck
f.arrow([(TB_X + 92, R_B - 12), (TB_X + 92, C_Y + 12)])                          # CPU -> test-backend-ops
f.arrow([(SIM_X + 180, C_B - 12), (SIM_X + 180, B_Y + 12)])                      # simulator -> backend
for x in (NC_X + 156, KL_X + 100, TB_X + 92):                                    # backend -> checks
    f.arrow([(x, B_Y + 12), (x, C_B - 12)])

# ---- panel 2 -----------------------------------------------------------------------------------
f.panel(PX0, P2_Y, PX1 - PX0, P2_B - P2_Y, label="FIRMWARE KERNELS")
KR_X, KR_W = 88, 336
f.box(KR_X, FR_Y, KR_W, FR_H, "neutral", "C++ REFERENCES", sub="sw/ctrl/ref\nof the composite kernels")
f.box(472, FR_Y, 336, FR_H, "neutral", "GGML CPU BACKEND", sub="layer tests: norms, RoPE,\nflash attention")
f.box(KR_X, FT_Y, 808 - KR_X, FT_H, "control", "FIRMWARE TESTS",
      sub="bit-exact: RISC-V build = host build = reference\n8 schedules · race detector fatal · fw_* ctests")
f.box(KR_X, FB_Y, KR_W, FB_H, "host", "RISC-V BUILD", sub="tsr-fw.elf on the RV32\ninterpreter (simulator)")
f.box(472, FB_Y, 336, FB_H, "host", "HOST BUILD", sub="the same kernels, the same\ntile-op stream")
for x in (KR_X + KR_W / 2, 472 + 168):
    f.arrow([(x, FR_Y + FR_H - 12), (x, FT_Y + 12)])
    f.arrow([(x, FB_Y + 12), (x, FT_Y + FT_H - 12)])
f.box(900, FT_Y, 444, FT_H, "control", "TSR-SELFTEST",
      sub="a device against the in-process\nsimulator shadow · load, VERIFY,\nHELLO, launch rate, traces")
f.box(900, FB_Y, 444, FB_H, "host", "SIM · MOCK · XRT KINDS", sub="the same checks on the card")
f.arrow([(1122, FB_Y + 12), (1122, FT_Y + FT_H - 12)])

# ---- panel 3 -----------------------------------------------------------------------------------
f.panel(PX0, P3_Y, PX1 - PX0, P3_B - P3_Y, label="TRACES, REPLAYS, RTL AND CARD")
f.box(GEN_X, T_Y, GEN_W, T_H, "host", "TSR-TRACE-GEN", sub="directed, random", label_role="label_s")
f.box(DMP_X, T_Y, DMP_W, T_H, "memory", "TSR_TRACE DUMPS", sub="of real runs")
f.box(SET_X, S_Y, SET_W, S_H, "memory", "TRACE SETS",
      sub="results from the simulator, byte-identical\nm2 · gen · m3 · vu · at · ctrl · model-v1conv")
f.box(SET_X, RP_Y, 320, RP_H, "host", "SIMULATOR KINDS", sub="tsr-trace-replay: full or\ntile-only, N schedules")
f.box(SET_X + 344, RP_Y, 320, RP_H, "control", "REPLAY KERNEL", sub="tile-only push.bin\nthrough the issue path")
f.box(SET_X, V_Y, SET_W, V_H, "compute", "U200 CARD",
      sub="traces through the replay kernel")
f.box(MUT_X, T_Y, MUT_W, T_H, "control", "MUTATION\nCAMPAIGNS", sub="of the RTL tests")
f.panel(RTL_X, S_Y, RTL_W, V_Y + V_H - S_Y, label="RTL REPLAY · BIT-EXACT", fill=None, cut=16)
in_x, in_w = RTL_X + 40, RTL_W - 80
f.box(in_x, S_Y + 40, in_w, 112, "compute", "ISSUE-FIFO REPLAY",
      sub="tb/fastreplay: tile-only\npush.bin into the issue FIFO")
f.box(in_x, S_Y + 40 + 112 + 40, in_w, 112, "compute", "IBEX LOCKSTEP",
      sub="RTL core vs the RV32\ninterpreter, retire by retire")

s_cy = S_Y + S_H / 2
f.arrow([(GEN_X + GEN_W / 2, T_Y + T_H - 12), (GEN_X + GEN_W / 2, S_Y + 12)])      # gen -> sets
f.arrow([(DMP_X + DMP_W / 2, T_Y + T_H - 12), (DMP_X + DMP_W / 2, S_Y + 12)])      # dumps -> sets
f.arrow([(SET_X + 160, S_Y + S_H - 12), (SET_X + 160, RP_Y + 12)])
f.arrow([(SET_X + 504, S_Y + S_H - 12), (SET_X + 504, RP_Y + 12)])
f.arrow([(SET_X + SET_W - 12, s_cy), (RTL_X + 12, s_cy)], fat=True)  # sets -> RTL replay
f.arrow([(SET_X + 408, RP_Y + RP_H - 12), (SET_X + 408, V_Y + 12)])  # replay kernel -> card
mut_cx = MUT_X + MUT_W / 2
f.line([(mut_cx, T_Y + T_H), (mut_cx, S_Y + 8)])                       # mutants -> RTL tests

# ---- callouts ----------------------------------------------------------------------------------
f.callout("THE GGML CPU\nBACKEND\nVALIDATES THE\nREFERENCES", (RX, R_Y + R_H / 2),
          target=(CPU_X + CPU_W, R_Y + R_H / 2))
f.callout("THE BACKEND\nUSES THE\nSIMULATOR AS\nITS DEVICE", (RX, B_Y + B_H / 2),
          target=(TB_X + TB_W, B_Y + B_H / 2))
f.callout("EVERY TEST\nOF THE RTL\nMUST FAIL ON\n≥ 1 MUTANT", (RX, T_Y + T_H / 2),
          target=(MUT_X + MUT_W, T_Y + T_H / 2))

# ---- legend ------------------------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("neutral", "REFERENCE"), ("host", "CODE UNDER TEST"),
                        ("control", "CHECK"), ("memory", "TRACES"), ("compute", "RTL")])

f.save(scale=1)

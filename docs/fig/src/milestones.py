"""Figure `milestones`: milestones M0 to M9 with their work, their gate and their status.

Replaces the Korean JPG of the spec. Source: docs/spec.md §19 (Repository layout and milestones:
the milestone table), docs/v2/README.md §2 (milestones M6 to M9 and their pass criteria) and §10
(work packages W0 to W6: which part of each milestone exists). One row per milestone: its name
(left, with a status tag on the top edge), its work (two lines) and its gate (two lines, right).
Status words (STYLE.md "Planned and unconfirmed things"): DONE · CARD (V1) = ran on the Alveo U200 with
the v1 core (tag v1-m4: M0, M3, M4); DONE · SIMULATOR (M1) and DONE · RTL SIM (M2) = passed in the
simulator or in RTL simulation, not on the card; MERGED INTO M8 (M5: host-assisted MoE assumed the
v1 command-buffer path, which v2 removes, D14, docs/v2/README.md §2); IN DEVELOPMENT (M6: the
simulator and the firmware are done, the RTL follows in W4 and W5, the card in W6); PLANNED (M7:
the kernels exist on the simulator, the backend that emits one launch per token is W3b; M8, M9).
The gates are the pass criteria of the spec; numbers of measured results are in
docs/performance.md and are not repeated here. Dark badge = on the card, outlined badge = on the
simulator or in RTL simulation, dashed badge and dashed blocks = planned (STYLE.md).

Run: build/tools/svgvenv/bin/python docs/fig/src/milestones.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544
MX, MW = 88, 296                      # milestone block
WX, WW = 424, 540                     # work
GX, GW = 1016, 480                    # gate (right edge 1496)
ROW_GAP = 24

CARD = ("DONE · CARD (V1)", "dark")
SIM = ("DONE · SIMULATOR", "outline")
RTLSIM = ("DONE · RTL SIMULATION", "outline")
DEV = ("IN DEVELOPMENT", "outline")
PLAN = ("PLANNED", "planned")
MERGED = ("MERGED INTO M8", "planned")

# (id, name, note, status, work, gate)
ROWS = [
    ("M0", "Shell bring-up", "", CARD,
     "DMA test kernel: per-channel and\n4-channel concurrent bandwidth",
     "Bandwidth recorded as the baseline\nfor performance prediction"),
    ("M1", "Simulator + backend", "", SIM,
     "TSR_DEVICE=sim, MUL_MAT offload\nonly, no hardware",
     "test-backend-ops; the 32B KL\ncheck in the simulator"),
    ("M2", "RTL core skeleton", "", RTLSIM,
     "v1 core: command processor,\nDMA (LD, ST), QACT",
     "Verilator trace replay,\nbit-exact"),
    ("M3", "MVM clusters, Q8_0", "", CARD,
     "One cluster per DDR4 channel:\n1 channel, then 4",
     "Qwen2.5-0.5B Q8_0 on the card: logits\nbit-identical to the simulator, KL met"),
    ("M4", "Q4_K, 4 channels", "the v1 goal", CARD,
     "Q4_K unpack on 4 channels; other\ntypes converted to Q8_0 at load",
     "Qwen2.5-32B Q4_K_M on the card\nat ≥ 1 tok/s"),
    ("M5", "Host-assisted MoE", "", MERGED,
     "Needed the v1 command-buffer path,\nwhich v2 removes",
     "-"),
    ("M6", "Core v2 hardware", "sim + firmware done\nRTL W4, W5 · card W6", DEV,
     "RISC-V control core, issue path,\nvector unit, attention and KV ops,\nSPM banks, GDB debug",
     "0.5B Q8_0 logits bit-identical to the\nsimulator, 32B KL criterion, GDB smoke"),
    ("M7", "Kernel library", "kernels on the sim;\nbackend: W3b", PLAN,
     "Kernels for norms, RoPE, activations,\nflash attention; one launch per token",
     "Qwen2.5 and Gemma 4 dense: one launch\nper token; DDR utilisation measured"),
    ("M8", "Device MoE, PyTorch", "", PLAN,
     "MUL_MAT_ID, TOPK, expert placement;\nPyTorch calls the same kernels",
     "Qwen3-30B-A3B on hardware"),
    ("M9", "Resident decode loop", "optional", PLAN,
     "On-device sampling and an\noutput-token ring buffer",
     "The host only polls the\nring buffer"),
]


def row_h(name, note):
    """Rows grow with the lines of the milestone block (name + note): 104, 128 or 152."""
    lines = 1 + (note.count("\n") + 1 if note else 0)
    return 104 + 24 * (lines - 1)


def work_lines(work):
    return work.count("\n") + 1


# ---- layout pass ----------------------------------------------------------------------------------
PY0 = 56
Y = PY0 + 56
ys = []
for i, (mid_, name, note, st, work, gate) in enumerate(ROWS):
    if mid_ == "M0":
        P_V1 = Y
        Y += 40
    if mid_ == "M5":
        Y += 16
    if mid_ == "M6":
        P_V2 = Y
        Y += 56
    h_ = max(row_h(name, note), 104 + 24 * (work_lines(work) - 2))
    ys.append((Y, h_))
    Y += h_ + ROW_GAP
PY1 = Y - ROW_GAP + 40
LEG_Y = PY1 + 56
H = LEG_Y + 112

f = Fig("milestones", W, H,
        title="Tesseract milestones M0 to M9",
        desc="Ten milestones as rows with name, work, gate and status. M0 shell bring-up, M3 MVM "
             "clusters with Q8_0 and M4 Q4_K on four channels ran on the Alveo U200 with the v1 "
             "core; M1 the simulator and backend and M2 the RTL core skeleton passed in the "
             "simulator and in RTL simulation; M5 host-assisted MoE is merged into M8; M6 core v2 "
             "hardware is in development (simulator and firmware done, RTL next); M7 kernels and "
             "graph execution, M8 device MoE and PyTorch, and M9 a resident decode loop are "
             "planned.")

f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0)


def phase(y, text):
    f.hline(MX, GX + GW, y, dashed=True, color=INK_SOFT, sw=1.5)
    f.tag(MX + 16, y, text, anchor="start")


phase(P_V1, "V1 · HOST COMMAND BUFFERS · TAG v1-m4")
phase(P_V2, "V2 · CONTROL CORE, ISSUE PATH, VECTOR UNIT, ATTENTION · docs/v2")

for (mid_, name, note, st, work, gate), (y, h) in zip(ROWS, ys):
    planned = st in (PLAN, MERGED)
    role = "control" if st in (CARD, SIM, RTLSIM) else ("host" if st == DEV else "neutral")
    f.box(MX, y, MW, h, role, mid_, sub=name + ("\n" + note if note else ""), planned=planned,
          tag=st[0], tag_style=st[1], plan_tag=True, label_role="label")
    f.box(WX, y, WW, h, "neutral", "", sub=work, planned=planned, plan_tag=False, cut=12)
    if gate != "-":
        f.box(GX, y, GW, h, "neutral", "", sub=gate, planned=planned, plan_tag=False, cut=12)
        f.arrow([(WX + WW - 12, y + h / 2), (GX + 12, y + h / 2)])
    else:
        f.box(GX, y, GW, h, "neutral", "", sub="no gate: dropped", planned=True, plan_tag=False,
              cut=12)
f.text((WX + WW + GX) / 2, ys[0][0] - 18, "GATE", "callout", anchor="middle", valign="middle")
f.text(WX + WW / 2, ys[0][0] - 18, "WORK", "callout", anchor="middle", valign="middle")

f.legend(W / 2, LEG_Y, [("control", "DONE"), ("host", "IN DEVELOPMENT"),
                        ("neutral", "PLANNED", True)], gap=48)
f.text(W / 2, LEG_Y + 38, wrap(
    "Dark badge: ran on the Alveo U200 with the v1 core. Outlined badge: passed in the simulator or "
    "in RTL simulation, or implemented on the simulator. Dashed: planned. Measured results are in "
    "docs/performance.md; the gates are those of docs/spec.md §19 and docs/v2/README.md §2.",
    1440), "note", anchor="middle", color=INK_SOFT)

f.save(scale=1)

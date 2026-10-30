"""Figure `launch`: from a ggml graph to device work, as a swimlane timeline (time runs down).

Lanes: llama.cpp, ggml-tesseract, tsr_transport, card DRAM, the control core (firmware) and a thin
lane for the units. Content follows docs/design/01-execution-model.md §2.1 (what the host does), §3.3
(cold load), §3.4 (boot), §4.1-§4.8 (launch protocol, runtime contract R1-R10, host procedures),
§11 (DRAM map) and docs/design/06-software-verification.md §7.2-§7.3 (the op list format, the
`graph` kernel), §8 (one launch per token), §9.3 (the backend); docs/spec.md §10 and §12:
- model load: supports_op probe with 512 tokens (llama.cpp's placement probe), set_tensor of a
  weight, conversion to Q8_0 if needed and repacking of rows into chunks, write of every channel's
  rows at W_OFF (the weight path: docs/spec.md §12, docs/xrt-host.md);
  firmware image: PT_LOAD segments written through the CSR loader with a receive CRC, VERIFY
  (read-back CRC of the TCMs), boot (CTRL.RV_RUN = 1), HELLO record (EX §3.3, §3.4);
- each launch: activations to the IO window of channel 0; the backend lowers one graph split to
  an op list (in M6 MUL_MAT records only; a cache hit needs no write) in the OPL region of
  channel 0; the transport writes the 8-word descriptor by posted CSR writes (LQ_W0-7) and
  LQ_PUSH; the runtime pops it, validates it and calls the kernel `graph` (EX §4.7 R2-R5); the
  executor reads the op list and issues the tile ops of every record to the unit queues without
  waiting (asynchronous issue with host-computed dependencies and joins, SW §7.3); cluster c
  streams its weights from channel c; the runtime joins every op (R6) and posts one answer
  record {STATUS, first error} to the completion ring (R7, R8); the host polls the ring (EX §4.8,
  `ap_ctrl_none`, no interrupt) and reads the result back from IO; the CPU continues with what the
  device did not take: in M6 attention, norms and RoPE (D17, SW §10.4 `offload_kqv` = false);
- the dashed note names M7 (one launch per token, SW §8), which is planned (work package W3b).
The control core is drawn planned (no RTL yet); the flow runs today on the simulator, where the
firmware executes on the RV32 interpreter (docs/design/06-software-verification.md §4.2).
The loop panel is the per-record loop of the executor; its arrow back is the next record. The
units lane stands for the five unit queues (DMA, QACT, MVM, VU, SYNC; CC §7.4).

Run: build/tools/svgvenv/bin/python docs/fig/src/launch.py      (TSR_FIG_THEME=dark: launch-dark.svg)
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import BAND, INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544                   # the inner panel that holds the lanes
LW, LG = 240, 12                      # lane width, gap between lanes
UW = 196                              # the units lane is thinner
LX = [PX0 + 16 + i * (LW + LG) for i in range(5)]    # 72, 324, 576, 828, 1080
LX.append(LX[4] + LW + LG)                            # 1332
LWS = [LW] * 5 + [UW]
BX = [x + 12 for x in LX]                              # block x in each lane
BWS = [w - 24 for w in LWS]                            # block widths 216 / 172
LC = [x + w / 2 for x, w in zip(LX, LWS)]              # lane centres
L, B, T, M, C, U = range(6)
LANE_R = LX[U] + UW                                    # 1528


class Timeline:
    """Rows of a swimlane: each row has a centre line; blocks are centred on it, so the hollow
    arrows between lanes run straight along it."""

    def __init__(self, y):
        self.y = y

    def row(self, h, gap=24):
        cy = self.y + h / 2
        self.y += h + gap
        return cy

    def skip(self, d):
        self.y += d


def block(f, lane, cy, h, role, label, sub="", **kw):
    """A block in `lane`, centred on the row line cy."""
    x = kw.pop("x", BX[lane])
    w = kw.pop("w", BWS[lane])
    f.box(x, cy - h / 2, w, h, role, label, sub=sub, label_role=kw.pop("label_role", "label_s"),
          **kw)


def hop(f, a, b, cy, tail=False, fat=False, dashed=False):
    """Hollow arrow from lane a to lane b along the row line (12 units inside each block)."""
    if a < b:
        x0, x1 = BX[a] + BWS[a] - 12, BX[b] + 12
    else:
        x0, x1 = BX[a] + 12, BX[b] + BWS[b] - 12
    f.arrow([(x0, cy), (x1, cy)], tail=tail, fat=fat, dashed=dashed)


def phase(f, y, text):
    """Phase rule across all lanes with a dark tag on its left end."""
    f.hline(LX[0], LANE_R, y, dashed=True, color=INK_SOFT, sw=1.5)
    f.tag(LX[0] + 16, y, text, anchor="start")


# ---- layout pass: row centres ------------------------------------------------------------------
PY0 = 56
HEAD_Y, HEAD_H = PY0 + 24, 88
tl = Timeline(HEAD_Y + HEAD_H + 40)
P1 = tl.y
tl.skip(40)
R_PROBE = tl.row(104)
R_WLOAD = tl.row(112, gap=48)
R_IMAGE = tl.row(104)
R_BOOT = tl.row(104, gap=40)
P2 = tl.y
tl.skip(40)
R_ACT = tl.row(80)
R_BUILD = tl.row(128)
R_LAUNCH = tl.row(128)
R_TAKE = tl.row(128, gap=40)
LOOP_Y = tl.y
tl.skip(56)
R_L1 = tl.row(136)                                     # issue; the units block starts here
R_L2 = tl.row(104, gap=32)                             # weights and IO to the units
tl.skip(-8)
LOOP_B = tl.y
tl.skip(32)
R_JOIN = tl.row(96, gap=56)
R_ANS = tl.row(128, gap=48)
R_READ = tl.row(80, gap=56)
R_CPU = tl.row(104, gap=40)
R_NOTE = tl.row(96, gap=24)
BAND_B = tl.y
PY1 = BAND_B + 24
LEG_Y = PY1 + 40
H = LEG_Y + 96

f = Fig("launch", W, H,
        title="From a ggml graph to device work",
        desc="Swimlane timeline, time running down, with lanes llama.cpp, ggml-tesseract, "
             "tsr_transport, card DRAM, the control core (firmware) and the units. Model load: "
             "supports_op probe, set_tensor of weights, conversion and repacking, rows written at "
             "W_OFF on every channel; the firmware image is loaded through the CSR loader, "
             "verified by read-back CRC, booted, and answers with HELLO. Each launch: "
             "activations written to the IO window, the split lowered to an op list in the OPL "
             "region (cache hit: no write), the 8-word launch descriptor pushed by posted CSR "
             "writes; the control core pops it, validates it and runs the graph kernel, which "
             "issues the tile ops of every record to the unit queues; cluster c streams weights "
             "from channel c; one completion record with status and first error is posted; the "
             "host polls, reads the result back and the CPU continues with attention, norms and "
             "RoPE. A dashed note marks M7, one launch per token, as planned.")

# ---- lanes and headers ---------------------------------------------------------------------------
f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0)
for i in range(6):
    f.band(LX[i], HEAD_Y + HEAD_H / 2, LWS[i], BAND_B - HEAD_Y - HEAD_H / 2, fill=BAND)
heads = [("host", "LLAMA.CPP", "graph, CPU ops"), ("host", "GGML-\nTESSERACT", "backend"),
         ("host", "TSR_\nTRANSPORT", "sim, mock, xrt"), ("memory", "CARD DRAM", "DDR4 ch0–ch3"),
         ("control", "CONTROL CORE", "firmware"), ("compute", "UNITS", "unit queues")]
for i, (role, lab, sub) in enumerate(heads):
    f.box(LX[i], HEAD_Y, LWS[i], HEAD_H, role, lab, sub=sub, label_role="label_s",
          planned=(i == C), tag="PLANNED · M6" if i == C else "")

# ---- 1. model load -------------------------------------------------------------------------------
phase(f, P1, "MODEL LOAD · ONCE")
block(f, L, R_PROBE, 104, "host", "SUPPORTS_OP", "MUL_MAT probe,\n512 tokens")
block(f, B, R_PROBE, 104, "host", "TRUE", "for loadable\n2D weights")
hop(f, L, B, R_PROBE, tail=True)

block(f, L, R_WLOAD, 112, "host", "SET_TENSOR", "a weight")
block(f, B, R_WLOAD, 112, "host", "CONVERT ·\nREPACK", "to Q8_0 if needed")
block(f, T, R_WLOAD, 112, "host", "WRITE", "rows of channel\nc at W_OFF")
block(f, M, R_WLOAD, 112, "memory", "WEIGHT", "same W_OFF on\nevery channel")
for a in (L, B, T):
    hop(f, a, a + 1, R_WLOAD)

block(f, T, R_IMAGE, 104, "host", "LOAD IMAGE", "CSR loader,\nreceive CRC")
block(f, C, R_IMAGE, 104, "control", "VERIFY", "read-back CRC\nof ITCM, DTCM", planned=True,
      plan_tag=False)
hop(f, T, C, R_IMAGE)
f.text(LC[M], R_IMAGE - 44, "PT_LOAD\nSEGMENTS", "callout", anchor="middle", valign="middle")

block(f, T, R_BOOT, 104, "host", "BOOT", "RV_RUN = 1,\nwait for HELLO")
block(f, C, R_BOOT, 104, "control", "HELLO", "runtime ready,\nimage id", planned=True,
      plan_tag=False)
hop(f, T, C, R_BOOT, tail=True)

# ---- 2. each launch ------------------------------------------------------------------------------
phase(f, P2, "EACH LAUNCH · M6: ONE GRAPH SPLIT")
block(f, L, R_ACT, 80, "host", "SET_TENSOR", "activations")
block(f, B, R_ACT, 80, "host", "IO WINDOW", "write(ch0, IO)")
block(f, T, R_ACT, 80, "host", "WRITE", "ch0, IO")
block(f, M, R_ACT, 80, "memory", "IO · CH0", "inputs")
for a in (L, B, T):
    hop(f, a, a + 1, R_ACT)

block(f, L, R_BUILD, 104, "host", "GRAPH_\nCOMPUTE", "one split")
# the backend lowers the split to an op list and hands it to the transport: a tall block
bt, bb = R_BUILD - 64, R_LAUNCH + 64
f.box(BX[B], bt, BWS[B], bb - bt, "host", "BUILD\nOP LIST", sub="MUL_MATs as\nrecords, with\ndependencies;\ncache lookup",
      label_role="label_s")
block(f, T, R_BUILD, 104, "host", "WRITE", "op list, only on\na cache miss")
block(f, M, R_BUILD, 104, "memory", "OPL · CH0", "op list")
hop(f, L, B, R_BUILD)
hop(f, B, T, R_BUILD)
hop(f, T, M, R_BUILD)

block(f, T, R_LAUNCH, 128, "host", "LAUNCH", "kernel graph,\n8 words, posted\nCSR writes")
hop(f, B, T, R_LAUNCH)
block(f, C, R_TAKE, 128, "control", "TAKE", "pop, validate,\ncall kernel\ngraph", planned=True,
      plan_tag=False)
# LQ_PUSH: transport to control core, one row lower (the arrow is drawn with an elbow)
f.arrow([(BX[T] + BWS[T] / 2, R_LAUNCH + 64 - 8), (BX[T] + BWS[T] / 2, R_TAKE),
         (BX[C] + 12, R_TAKE)])
f.text(LC[M], R_TAKE - 44, "LQ_W0–7,\nLQ_PUSH", "callout", anchor="middle", valign="middle")

# ---- loop: each record -------------------------------------------------------------------------------
lp_x0, lp_x1 = LX[M] - 8, LANE_R + 8
f.panel(lp_x0, LOOP_Y, lp_x1 - lp_x0, LOOP_B - LOOP_Y, label="LOOP · EACH OP-LIST RECORD",
        fill=None)
block(f, M, R_L1, 104, "memory", "OPL · CH0", "next record")
block(f, C, R_L1, 136, "control", "ISSUE", "tile ops LD,\nQACT, MVM, ST;\ndo not wait", planned=True,
      plan_tag=False)
hop(f, M, C, R_L1)
ub_t, ub_b = R_L1 - 68, R_L2 + 52
f.box(BX[U], ub_t, BWS[U], ub_b - ub_t, "compute", "UNIT\nQUEUES", sub="in order per\nunit; tickets,\njoins",
      label_role="label_s")
hop(f, C, U, R_L1)
block(f, M, R_L2, 104, "memory", "WEIGHT · IO", "cluster c reads\nchannel c")
f.arrow([(BX[M] + BWS[M] - 12, R_L2), (BX[U] + 12, R_L2)], fat=True)

# ---- end: join, answer, readback ------------------------------------------------------------------
block(f, C, R_JOIN, 96, "control", "JOIN", "every op done,\nDRAM written", planned=True,
      plan_tag=False)
block(f, C, R_ANS, 128, "control", "ANSWER", "LCQ record:\nstatus, first\nerror", planned=True,
      plan_tag=False)
f.arrow([(LC[C], R_JOIN + 48 - 2), (LC[C], R_ANS - 64 + 6)])
block(f, T, R_ANS, 128, "host", "POLL", "LCQ_WR, then\nrecord r0")
hop(f, C, T, R_ANS)
f.text(LC[M], R_ANS - 44, "NO INTERRUPT:\nHOST POLLS", "callout", anchor="middle", valign="middle")

block(f, L, R_READ, 80, "host", "GET_TENSOR", "result")
block(f, B, R_READ, 80, "host", "READ BACK", "f32 rows")
block(f, T, R_READ, 80, "host", "READ", "ch0, IO")
block(f, M, R_READ, 80, "memory", "IO · CH0", "result bytes")
for a in (L, B, T):
    hop(f, a, a + 1, R_READ, tail=True)

block(f, L, R_CPU, 104, "host", "CPU CONTINUES", "M6: attention,\nnorms, RoPE")
f.arrow([(LC[L], R_READ + 40 - 8), (LC[L], R_CPU - 52 + 8)])
f.box(BX[B], R_NOTE - 48, BX[T] + BWS[T] - BX[B], 96, "host", "M7: ONE LAUNCH PER TOKEN",
      sub="attention, norms, RoPE on the device", planned=True, tag="PLANNED · W3b",
      label_role="label_s")

# ---- legend --------------------------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("host", "HOST SOFTWARE"), ("memory", "MEMORY"), ("control", "CONTROL"),
                        ("compute", "COMPUTE"), ("neutral", "PLANNED", True)])
f.text(W / 2, LEG_Y + 36, wrap(
    "The flow runs on the simulator today; the control core is planned RTL (W4), so its blocks are "
    "dashed.", 1440), "note", anchor="middle", color=INK_SOFT)

f.save()

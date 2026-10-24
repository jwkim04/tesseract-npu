"""Figure `exec`: one graph execution as a swimlane timeline (time runs down).

Content follows README "From a llama.cpp graph to device work" (the sequence Mermaid block),
step for step: model load (supports_op probe with 512 tokens, set_tensor of a weight, convert to
Q8_0 if needed and repack rows into chunks, write each channel's rows at W_OFF); then for each
graph and each Tesseract split: set_tensor of activations into an IO window and write(ch0, IO);
graph_compute, lowering of MUL_MATs to LD, QACT, MVM, ST and the program-cache lookup; the program
+ END written into the idle CMD half; exec(cmd_addr, cmd_count, args), CMD_ADDR, CMD_COUNT then
CTRL.START; instruction prefetch from ch0; per MUL_MAT and token tile LD, QACT, MVM (cluster c
streams its rows from channel c; the y rows are gathered into the SPM as part of the MVM), ST; END
sets STATUS.DONE and done; return, or abort with ERR_CODE, ERR_PC, ERR_INFO; get_tensor,
read(ch0, IO), f32 rows back and the CPU continues with norms, RoPE and attention.
The loop shows exactly the four instructions of a tile (README "Lowering and program cache": "Each
tile emits four instructions: LD, QACT, MVM, ST"); the gather is inside the MVM step because
tsr_mvm's done waits for it (docs/rtl-m3.md §6.4: done = broadcast complete and gather done).
Additions beyond the Mermaid block, with their sources:
- exec blocks until DONE or ERR (README "Transport layer"), drawn as one bar in the transport lane;
- at START R1-R7 = 0 and R8-R15 = ARG0-ARG7, and the core prefetches and executes strictly in
  order; on an error it stops at the faulting instruction and STATUS becomes DONE | ERR, without
  reaching END (README "Execution model, CSRs and errors", steps 2-5);
- the loop's blocks use the unit colours of the core figure: LD and ST lavender (tsr_dma, data
  movement), QACT and MVM sky (compute);
- the lanes' subtitles repeat the system figure (README system Mermaid block); tsr_core is "the NPU
  core" (the project's positioning);
- compute buffers are one window in the ch0 IO region (README "Weight placement and conversion");
- weights are converted unless already Q8_0 or Q4_K (README "Weight placement and conversion").

Run: build/tools/svgvenv/bin/python docs/fig/src/exec.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import BAND, INK_SOFT, Fig  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544                   # the inner panel that holds the lanes
LW, LG = 272, 20                      # lane width, gap between lanes
LX0 = PX0 + 24
LX = [LX0 + i * (LW + LG) for i in range(5)]        # 80, 372, 664, 956, 1248
BX = [x + 16 for x in LX]                            # block x in each lane
BW = LW - 32                                         # block width 240
LC = [x + LW / 2 for x in LX]                        # lane centres
L, B, T, M, C = range(5)
LANE_R = LX[4] + LW                                  # 1520


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
    w = kw.pop("w", BW)
    f.box(x, cy - h / 2, w, h, role, label, sub=sub, label_role=kw.pop("label_role", "label_s"),
          **kw)
    return x, w


def hop(f, a, b, cy, tail=False, fat=False, xa=None, xb=None):
    """Hollow arrow from lane a to lane b along the row line (12 units inside each block)."""
    if a < b:
        x0 = (xa if xa is not None else BX[a] + BW) - 12
        x1 = (xb if xb is not None else BX[b]) + 12
    else:
        x0 = (xa if xa is not None else BX[a]) + 12
        x1 = (xb if xb is not None else BX[b] + BW) - 12
    f.arrow([(x0, cy), (x1, cy)], tail=tail, fat=fat)


def phase(f, y, text):
    """Phase rule across all lanes with a dark tag on its left end."""
    f.hline(LX0, LANE_R, y, dashed=True, color=INK_SOFT, sw=1.5)
    f.tag(LX0 + 16, y, text, anchor="start")


# ---- layout pass: row centres ------------------------------------------------------------------
PY0 = 56
HEAD_Y, HEAD_H = PY0 + 24, 80
tl = Timeline(HEAD_Y + HEAD_H + 40)
P1 = tl.y
tl.skip(40)
R_PROBE = tl.row(104)
R_WLOAD = tl.row(104, gap=40)
P2 = tl.y
tl.skip(40)
R_ACT = tl.row(80)
R_LOWER = tl.row(104, gap=40)
R_EXEC = tl.row(128)
R_FETCH = tl.row(80, gap=40)
LOOP_Y = tl.y
tl.skip(56)
LOOP_ROWS = [("LD", "x tile into SPM", 88), ("QACT", "f32 rows to QA", 88),
             ("MVM", "cluster c streams\nchannel c,\ny rows gathered\ninto the SPM", 160),
             ("ST", "y tile to IO", 88)]
R_LOOP = [tl.row(h, gap=40) for _, _, h in LOOP_ROWS]
tl.skip(-40 + 32)
LOOP_B = tl.y
tl.skip(32)
R_END = tl.row(128)
R_READ = tl.row(80, gap=56)
R_CPU = tl.row(104, gap=32)
BAND_B = tl.y
PY1 = BAND_B + 24
LEG_Y = PY1 + 40
H = LEG_Y + 48

f = Fig("exec", W, H,
        title="One graph execution, from llama.cpp to tsr_core",
        desc="Swimlane timeline, time running down, with lanes llama.cpp, ggml-tesseract, "
             "tsr_transport, card DRAM and tsr_core. Model load: supports_op probe, set_tensor "
             "of weights, conversion and repacking, rows written at W_OFF on every channel. Each "
             "split: activations written to ch0 IO; MUL_MATs lowered with a program cache; "
             "program + END written to the idle CMD half; exec writes CMD_ADDR, CMD_COUNT and "
             "CTRL.START and blocks; the core prefetches from ch0 and runs LD, QACT, MVM (which "
             "gathers the y rows into the SPM) and ST per MUL_MAT and token tile; END sets DONE, "
             "or an error stops the core with DONE | ERR; the host reads "
             "the result back from ch0 IO and the CPU continues.")

# ---- lanes and headers ---------------------------------------------------------------------------
f.panel(PX0, PY0, PX1 - PX0, PY1 - PY0)
for i in range(5):
    f.band(LX[i], HEAD_Y + HEAD_H / 2, LW, BAND_B - HEAD_Y - HEAD_H / 2, fill=BAND)
heads = [("host", "LLAMA.CPP", "graph, CPU ops"), ("host", "GGML-TESSERACT", "repack, lower, cache"),
         ("host", "TSR_TRANSPORT", "sim or xrt"), ("memory", "CARD DRAM", "DDR4 ch0–ch3"),
         ("control", "TSR_CORE", "the NPU core")]
for i, (role, lab, sub) in enumerate(heads):
    f.box(LX[i], HEAD_Y, LW, HEAD_H, role, lab, sub=sub)

# ---- 1. model load -------------------------------------------------------------------------------
phase(f, P1, "MODEL LOAD")
block(f, L, R_PROBE, 104, "host", "SUPPORTS_OP", "MUL_MAT probe,\n512 tokens")
block(f, B, R_PROBE, 104, "host", "TRUE", "for loadable\n2D weights")
hop(f, L, B, R_PROBE, tail=True)

block(f, L, R_WLOAD, 104, "host", "SET_TENSOR", "weight, into a\nWEIGHTS buffer")
block(f, B, R_WLOAD, 104, "host", "CONVERT, REPACK", "to Q8_0 if needed\nrows into chunks")
block(f, T, R_WLOAD, 104, "host", "WRITE", "rows of channel c\nat W_OFF")
block(f, M, R_WLOAD, 104, "memory", "WEIGHT", "same W_OFF on\nevery channel")
for a in (L, B, T):
    hop(f, a, a + 1, R_WLOAD)

# ---- 2. each graph, each Tesseract split -----------------------------------------------------------
phase(f, P2, "EACH GRAPH · EACH TESSERACT SPLIT")
block(f, L, R_ACT, 80, "host", "SET_TENSOR", "activations")
block(f, B, R_ACT, 80, "host", "IO WINDOW", "write(ch0, IO)")
block(f, T, R_ACT, 80, "host", "WRITE", "ch0, IO")
block(f, M, R_ACT, 80, "memory", "IO · CH0", "activations")
for a in (L, B, T):
    hop(f, a, a + 1, R_ACT)

block(f, L, R_LOWER, 80, "host", "GRAPH_COMPUTE", "one split")
# the backend lowers, writes the program and calls exec: one tall block over two rows
lb_top, lb_bot = R_LOWER - 52, R_EXEC + 64
f.box(BX[B], lb_top, BW, lb_bot - lb_top, "host", "LOWER +\nPROGRAM CACHE",
      sub="MUL_MATs to LD,\nQACT, MVM, ST", label_role="label_s")
block(f, T, R_LOWER, 104, "host", "WRITE", "program + END,\nidle CMD half")
block(f, M, R_LOWER, 104, "memory", "CMD · CH0", "idle half")
hop(f, L, B, R_LOWER)
hop(f, B, T, R_LOWER)
hop(f, T, M, R_LOWER)

# exec: the transport blocks from START until DONE or ERR (one bar down to the END row)
ex_top, ex_bot = R_EXEC - 64, R_END + 64
f.box(BX[T], ex_top, BW, ex_bot - ex_top, "host")
f.label_block(BX[T], ex_top, BW, 128, "EXEC", "cmd_addr,\ncmd_count, args", label_role="label_s")
f.text(LC[T], (R_FETCH + R_END) / 2, "blocks until\nDONE or ERR", "sub", anchor="middle",
       valign="middle", color=INK_SOFT)
f.label_block(BX[T], R_END - 64, BW, 128, "EXEC RETURNS", label_role="label_s")
hop(f, B, T, R_EXEC)
block(f, C, R_EXEC, 128, "control", "START", "R1–R7 = 0\nR8–R15 =\nARG0–ARG7")
hop(f, T, C, R_EXEC)
f.text(LC[M], R_EXEC - 26 - 19 * 1.4, "CMD_ADDR, CMD_COUNT,\nTHEN CTRL.START", "callout",
       anchor="middle")

block(f, M, R_FETCH, 80, "memory", "CMD · CH0", "the program")
block(f, C, R_FETCH, 80, "control", "PREFETCH", "in order")
hop(f, M, C, R_FETCH)

# ---- loop: each MUL_MAT, each token tile ------------------------------------------------------------
lp_x0, lp_x1 = LX[M] - 8, LANE_R + 8
f.panel(lp_x0, LOOP_Y, lp_x1 - lp_x0, LOOP_B - LOOP_Y, label="LOOP · EACH MUL_MAT, EACH TOKEN TILE",
        fill=None)
CW = 232
roles = {"LD": "io", "QACT": "compute", "MVM": "compute", "ST": "io"}   # as in the core figure
for (lab, sub, h), cy in zip(LOOP_ROWS, R_LOOP):
    block(f, C, cy, h, roles[lab], lab, sub, w=CW)
for i in range(len(LOOP_ROWS) - 1):
    y0 = R_LOOP[i] + LOOP_ROWS[i][2] / 2
    y1 = R_LOOP[i + 1] - LOOP_ROWS[i + 1][2] / 2
    f.arrow([(BX[C] + CW / 2, y0 - 8), (BX[C] + CW / 2, y1 + 8)])
# back to LD for the next tile
rx = BX[C] + CW + 16
f.line([(BX[C] + CW, R_LOOP[-1]), (rx, R_LOOP[-1]), (rx, R_LOOP[0]), (BX[C] + CW, R_LOOP[0])])
block(f, M, R_LOOP[0], 80, "memory", "IO · CH0", "x tile")
hop(f, M, C, R_LOOP[0])
block(f, M, R_LOOP[2], 104, "memory", "WEIGHT", "ch c, at W_OFF")             # the MVM row
hop(f, M, C, R_LOOP[2], fat=True)
block(f, M, R_LOOP[-1], 80, "memory", "IO · CH0", "y tile")
hop(f, C, M, R_LOOP[-1])

# ---- end, readback -------------------------------------------------------------------------------
block(f, C, R_END, 128, "control", "END OR ERROR", "END: STATUS.DONE\nerror: stops,\nDONE | ERR")
hop(f, C, T, R_END)
f.text(LC[M], R_END - 26, "DONE", "callout", anchor="middle")
block(f, B, R_END, 128, "host", "RETURN", "or abort with\nERR_CODE, ERR_PC,\nERR_INFO")
hop(f, T, B, R_END)

block(f, L, R_READ, 80, "host", "GET_TENSOR", "result")
block(f, B, R_READ, 80, "host", "READ BACK", "f32 rows")
block(f, T, R_READ, 80, "host", "READ", "ch0, IO")
block(f, M, R_READ, 80, "memory", "IO · CH0", "result bytes")
for a in (L, B, T):
    hop(f, a, a + 1, R_READ, tail=True)

block(f, L, R_CPU, 104, "host", "CPU CONTINUES", "norms, RoPE,\nattention")
f.arrow([(LC[L], R_READ + 40 - 8), (LC[L], R_CPU - 52 + 8)])

# ---- legend -----------------------------------------------------------------------------------------
f.legend(W / 2, LEG_Y, [("host", "HOST SOFTWARE"), ("memory", "MEMORY"), ("control", "CONTROL"),
                        ("compute", "COMPUTE"), ("io", "DATA MOVEMENT")])

f.save()

"""Figure `dataflow`: how data and control flow for one decode token, from the host to the hardware
and back, as seven numbered steps over three bands (host, firmware, hardware).

Content follows docs/design/01-execution-model.md §2.1, §4 (launch protocol, runtime contract),
docs/design/06-software-verification.md §7 (op list, `graph` kernel), §8, §9.3 (backend) and
docs/spec.md §2 (four MVM clusters of 128 MAC lanes, 4 MiB SPM, four DDR4 channels):
1. llama.cpp builds the ggml graph; ggml-tesseract lowers it to an op list (cached after the first
   token); the weights already sit in the card DDR (written at model load);
2. tsr_transport writes the token inputs to DDR and pushes one launch descriptor over PCIe;
3. the runtime on the control core takes the launch; the `graph` kernel of Basis walks the op list;
4. the kernel issues tile ops through the issue path (checker, unit queues) to the units;
5. the DMA moves activations and KV rows between the DDR and the SPM; each MVM cluster streams its
   weights from its own DDR4 channel; the vector unit and the attention ops work in the SPM; the KV
   cache stays in DDR (docs/spec.md §2, docs/rtl-core.md §3.2, docs/rtl-mvm.md §4);
6. results go back to DDR; the runtime posts an answer record;
7. the host reads the answer and the logits.
The bus under the issue path stands for the unit queues. Every block is drawn solid. The SPM is a
bar under the four units, which all read and write it; the weight stream skips the SPM, so its
arrow passes behind the bar.

Run: build/tools/svgvenv/bin/python docs/fig/src/dataflow.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import Fig, logo_box, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544
BW, GAP = 336, 32
CX0 = 80
X = [CX0 + i * (BW + GAP) for i in range(4)]           # block x per column
C = [x + BW / 2 for x in X]                             # column centers
BH = 136

# vertical layout
HOST_Y = 56
HB = HOST_Y + 56                                        # host blocks
HOST_B = HB + BH + 32
FW_Y = HOST_B + 120
FB = FW_Y + 56
FW_B = FB + BH + 32
HW_Y = FW_B + 72
RA = HW_Y + 56                                          # row A: issue path
BUS = RA + BH + 40
RB = BUS + 32                                           # row B: the four units
RB_H = 128
RS = RB + RB_H + 88                                     # SPM bar under the units
RS_H = 72
RC = RS + RS_H + 88                                     # row C: DDR4
RC_H = 104
HW_B = RC + RC_H + 32
STEPS_Y = HW_B + 72

STEPS = [
    "llama.cpp builds the ggml graph; ggml-tesseract turns it into an op list, cached after the "
    "first token. The weights already sit in card DDR.",
    "The transport writes the token inputs to DDR and pushes one launch descriptor over PCIe.",
    "The runtime on the control core takes the launch; the Basis graph kernel walks the op list.",
    "The kernel issues tile ops through the issue path (checker, unit queues) to the units.",
    "The DMA moves activations and KV rows between DDR and the 4 MiB SPM. Each MVM cluster "
    "streams weights from its own DDR4 channel. The vector unit and attention ops work in the "
    "SPM; the KV cache stays in DDR.",
    "Results go back to DDR and the runtime posts an answer record.",
    "The host reads the answer and the logits.",
]
STEP_COLW = 700
STEP_LINES = [wrap(s_, STEP_COLW - 44, "sub") for s_ in STEPS]
STEP_H = [26 * (t.count("\n") + 1) + 28 for t in STEP_LINES]
H = int((STEPS_Y + max(sum(STEP_H[:4]), sum(STEP_H[4:])) + 40 + 7) // 8 * 8)

f = Fig("dataflow", W, H,
        title="One decode token, from the host to the hardware and back",
        desc="Three bands, host, firmware on the RISC-V control core and hardware, with seven "
             "numbered steps. 1: llama.cpp builds the ggml graph and ggml-tesseract turns it into "
             "a cached op list; the weights already sit in card DDR. 2: the transport writes the "
             "token inputs to DDR and pushes one launch descriptor over PCIe. 3: the runtime takes "
             "the launch and the Basis graph kernel walks the op list. 4: the kernel issues tile "
             "ops through the issue path and unit queues. 5: the DMA moves activations and KV rows "
             "between DDR and the 4 MiB SPM, each of four MVM clusters streams weights from its own "
             "DDR4 channel, the vector unit and attention ops work in the SPM, and the KV cache "
             "stays in DDR. 6: results go back to DDR and the runtime posts an "
             "answer record. 7: the host reads the answer and the logits.")


def blk(col, y, role, label, sub, steps="", h=BH, **kw):
    f.box(X[col], y, kw.pop("w", BW), h, role, label, sub=sub, label_role="label_s", **kw)
    if steps:
        f.tag(X[col] + 22, y, steps, anchor="start")


def hop(a, b, cy, **kw):
    """Arrow between columns a and b along cy, 12 units inside each block."""
    if a < b:
        pts = [(X[a] + BW - 12, cy), (X[b] + 12, cy)]
    else:
        pts = [(X[a] + 12, cy), (X[b] + BW - 12, cy)]
    f.arrow(pts, **kw)


f.panel(PX0, HOST_Y, PX1 - PX0, HOST_B - HOST_Y, label="HOST")
f.panel(PX0, FW_Y, PX1 - PX0, FW_B - FW_Y, label="FIRMWARE · RISC-V CONTROL CORE")
f.panel(PX0, HW_Y, PX1 - PX0, HW_B - HW_Y, label="HARDWARE · TESSERACT CORE AND CARD")
for _key, _y in (("tesseract-basis", FW_Y), ("tesseract-core", HW_Y)):   # sub-brand lockups on the panel edge
    _lg = {"key": _key, "h": 36, "backing": True, "pad": 4}
    _lw, _lh = logo_box(**_lg)
    f.logo(x=PX1 - 48 - _lw, y=_y - _lh / 2, **_lg)

# ---- host ------------------------------------------------------------------------------------------
cy = HB + BH / 2
blk(0, HB, "host", "LLAMA.CPP", "builds the ggml\ngraph for the token", "1")
blk(1, HB, "host", "GGML-\nTESSERACT", "op list, cached\nafter token 1", "1")
blk(2, HB, "host", "TSR_TRANSPORT", "writes inputs, pushes\nthe launch, reads the\nanswer", "2 · 7")
hop(0, 1, cy)
hop(1, 2, cy)

# host <-> firmware: launch down, answer up (both at column 2)
y0, y1 = HB + BH - 6, FB + 6
f.arrow([(C[2] - 56, y0), (C[2] - 56, y1)])
f.arrow([(C[2] + 56, y1), (C[2] + 56, y0)])
f.text(C[2] - 56 - 28, (y0 + y1) / 2, "LAUNCH\nOVER PCIE", "callout", anchor="end", valign="middle")
f.text(C[2] + 56 + 28, (y0 + y1) / 2, "ANSWER\nRECORD", "callout", anchor="start", valign="middle")

# ---- firmware ----------------------------------------------------------------------------------------
cyf = FB + BH / 2
blk(2, FB, "control", "RUNTIME", "takes the launch;\nposts the answer", "3 · 6")
blk(1, FB, "control", "BASIS GRAPH\nKERNEL", "walks the op list", "3")
hop(2, 1, cyf, tail=True)

# kernel -> issue path (column 1)
f.arrow([(C[1], FB + BH - 6), (C[1], RA + 6)])

# ---- hardware ----------------------------------------------------------------------------------------
blk(1, RA, "control", "ISSUE PATH", "checker, unit queues;\ntile ops in order", "4")
# bus to the four units of row B
f.line([(C[1], RA + BH), (C[1], BUS)], end=False, dashed=True)
f.line([(C[0], BUS), (C[3], BUS)], end=False, dashed=True)
for c in range(4):
    f.line([(C[c], BUS), (C[c], RB - 2)], dashed=True)
f.text(C[2], BUS - 20, "UNIT QUEUES", "callout", anchor="middle", valign="middle")

blk(0, RB, "io", "DMA", "LD · ST · KVST\nDDR <-> SPM", "5", h=RB_H)
blk(1, RB, "compute", "MVM CLUSTERS x4", "128 MAC lanes each;\nstream weights", "5", h=RB_H)
blk(2, RB, "compute", "VECTOR UNIT", "norms, RoPE,\nsoftmax parts", "5", h=RB_H)
blk(3, RB, "compute", "QACT", "quantizes f32\nto QA", "5", h=RB_H)

f.box(X[0], RC, X[3] + BW - X[0], RC_H, "memory", "CARD DDR · FOUR DDR4 CHANNELS",
      sub="weights, KV cache, op list, token inputs, results", label_role="label_s")
f.tag(X[0] + 22, RC, "1 · 6", anchor="start")
# the weight stream skips the SPM: the arrow runs behind the SPM bar
wx = C[1] + 96
f.arrow([(wx, RC + 12), (wx, RB + RB_H - 12)], fat=True)
f.box(X[0], RS, X[3] + BW - X[0], RS_H, "memory", "SPM · 4 MiB",
      sub="activations, results and attention scratch, read and written by every unit",
      label_role="label_s", cut=12)
for c in range(4):
    cx = C[c] - 96 if c == 1 else C[c]
    f.arrow([(cx, RB + RB_H - 12), (cx, RS + 12)], tail=True)
f.arrow([(C[0], RS + RS_H - 8), (C[0], RC + 12)], tail=True)
f.text(C[0] + 52, (RS + RS_H + RC) / 2, "LD · ST", "callout", valign="middle")
f.text(wx + 64, (RS + RS_H + RC) / 2, "WEIGHTS", "callout", valign="middle")

# ---- steps -------------------------------------------------------------------------------------------
colw = STEP_COLW
ys = [STEPS_Y, STEPS_Y]
for i, txt in enumerate(STEP_LINES):
    col = 0 if i < 4 else 1
    x = 80 + col * (colw + 48)
    y = ys[col]
    f.tag(x + 16, y + 8, str(i + 1), anchor="middle")
    f.text(x + 44, y + 2, txt, "sub", valign="top")
    ys[col] += STEP_H[i]

f.save()

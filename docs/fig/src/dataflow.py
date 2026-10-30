"""Figure `dataflow`: how data and control flow for one decode token, from the host to the hardware
and back, as seven numbered steps over three bands (host, firmware, hardware).

Content follows docs/v2/01-execution-model.md §2.1, §4 (launch protocol, runtime contract),
docs/v2/06-software-verification.md §7 (op list, `graph` kernel), §8, §9.3 (backend) and
docs/spec.md §2 (four MVM clusters of 128 MAC lanes, 4 MiB SPM, four DDR4 channels):
1. llama.cpp builds the ggml graph; ggml-tesseract lowers it to an op list (cached after the first
   token); the weights already sit in the card DDR (written at model load);
2. tsr_transport writes the token inputs to DDR and pushes one launch descriptor over PCIe;
3. the runtime on the control core takes the launch; the `graph` kernel of Basis walks the op list;
4. the kernel issues tile ops through the issue path (checker, unit queues) to the units;
5. DMA streams weights from the DDR4 channels into the four MVM clusters; the vector unit and the
   attention ops work in the SPM; the KV cache stays in DDR;
6. results go back to DDR; the runtime posts an answer record;
7. the host reads the answer and the logits.
The bus under the issue path stands for the unit queues. Control core and units are planned RTL;
the flow runs on the simulator today (docs/v2/06-software-verification.md §4.2).

Run: build/tools/svgvenv/bin/python docs/fig/src/dataflow.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544
BW, GAP = 336, 32
CX0 = 80
X = [CX0 + i * (BW + GAP) for i in range(4)]           # block x per column
C = [x + BW / 2 for x in X]                             # column centres
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
RB = BUS + 32                                           # row B: units and SPM
RB_H = 128
RC = RB + RB_H + 56                                     # row C: DDR4
RC_H = 104
HW_B = RC + RC_H + 32
STEPS_Y = HW_B + 72
H = STEPS_Y + 408

STEPS = [
    "llama.cpp builds the ggml graph; ggml-tesseract turns it into an op list, cached after the "
    "first token. The weights already sit in card DDR.",
    "The transport writes the token inputs to DDR and pushes one launch descriptor over PCIe.",
    "The runtime on the control core takes the launch; the Basis graph kernel walks the op list.",
    "The kernel issues tile ops through the issue path (checker, unit queues) to the units.",
    "DMA streams weights from the four DDR4 channels into the four MVM clusters; the vector unit "
    "and attention ops work in the 4 MiB SPM; the KV cache stays in DDR.",
    "Results go back to DDR and the runtime posts an answer record.",
    "The host reads the answer and the logits.",
]

f = Fig("dataflow", W, H,
        title="One decode token, from the host to the hardware and back",
        desc="Three bands, host, firmware on the RISC-V control core and hardware, with seven "
             "numbered steps. 1: llama.cpp builds the ggml graph and ggml-tesseract turns it into "
             "a cached op list; the weights already sit in card DDR. 2: the transport writes the "
             "token inputs to DDR and pushes one launch descriptor over PCIe. 3: the runtime takes "
             "the launch and the Basis graph kernel walks the op list. 4: the kernel issues tile "
             "ops through the issue path and unit queues. 5: DMA streams weights from four DDR4 "
             "channels into four MVM clusters; the vector unit and attention ops work in the 4 MiB "
             "SPM; the KV cache stays in DDR. 6: results go back to DDR and the runtime posts an "
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
f.panel(PX0, HW_Y, PX1 - PX0, HW_B - HW_Y, label="HARDWARE · NPU CORE AND CARD")

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
blk(2, FB, "control", "RUNTIME", "takes the launch;\nposts the answer", "3 · 6", planned=True, plan_tag=False)
blk(1, FB, "control", "BASIS GRAPH\nKERNEL", "walks the op list", "3", planned=True, plan_tag=False)
hop(2, 1, cyf, tail=True)

# kernel -> issue path (column 1)
f.arrow([(C[1], FB + BH - 6), (C[1], RA + 6)])

# ---- hardware ----------------------------------------------------------------------------------------
blk(1, RA, "control", "ISSUE PATH", "checker, unit queues;\ntile ops in order", "4", planned=True, plan_tag=False)
# bus to the units of row B (columns 0, 1, 2)
f.line([(C[1], RA + BH), (C[1], BUS)], end=False, dashed=True)
f.line([(C[0], BUS), (C[2], BUS)], end=False, dashed=True)
for c in (0, 1, 2):
    f.line([(C[c], BUS), (C[c], RB - 2)], dashed=True)
f.text(C[2] + 40, BUS, "UNIT QUEUES", "callout", valign="middle")

blk(0, RB, "compute", "DMA", "streams weights\nfrom DDR4", "5", h=RB_H, planned=True, plan_tag=False)
blk(1, RB, "compute", "MVM CLUSTERS x4", "128 MAC lanes\neach", "5", h=RB_H, planned=True, plan_tag=False)
blk(2, RB, "compute", "VECTOR UNIT", "norms, softmax,\nattention ops", "5", h=RB_H, planned=True, plan_tag=False)
blk(3, RB, "memory", "SPM · 4 MiB", "activations and\nattention scratch", "5", h=RB_H)
cyb = RB + RB_H / 2
hop(0, 1, cyb, fat=True)
hop(1, 2, cyb)
hop(2, 3, cyb, tail=True)

f.box(X[0], RC, X[3] + BW - X[0], RC_H, "memory", "CARD DDR · FOUR DDR4 CHANNELS",
      sub="weights, KV cache, op list, token inputs, results", label_role="label_s")
f.tag(X[0] + 22, RC, "1 · 6", anchor="start")
f.arrow([(C[0], RC + 12), (C[0], RB + RB_H - 12)], fat=True)
f.arrow([(C[3], RB + RB_H - 12), (C[3], RC + 12)])
f.text(C[0] + 80, (RC + RB + RB_H) / 2, "WEIGHTS", "callout", valign="middle")
f.text(C[3] + 40, (RC + RB + RB_H) / 2, "RESULTS", "callout", valign="middle")

# ---- steps -------------------------------------------------------------------------------------------
colw = 700
ys = [STEPS_Y, STEPS_Y]
for i, s_ in enumerate(STEPS):
    col = 0 if i < 4 else 1
    x = 80 + col * (colw + 48)
    y = ys[col]
    txt = wrap(s_, colw - 44, "sub")
    f.tag(x + 16, y + 8, str(i + 1), anchor="middle")
    f.text(x + 44, y + 2, txt, "sub", valign="top")
    ys[col] += 26 * (txt.count("\n") + 1) + 28
H_END = max(ys)

f.text(W / 2, H - 40, "The flow runs on the simulator today; the control core and units are planned RTL, "
       "so their blocks are dashed.", "note", anchor="middle", color=INK_SOFT)
f.save()

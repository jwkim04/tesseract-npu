"""Figure `issue`: from a kernel's `issue()` to a unit, and the queues of one flash-attention layer.

Upper panel: the issue path of one tile op, left to right (docs/v2/04-control-core.md §7 issue
path, §8 dispatcher and checker, §11.1 clocks; docs/v2/01-execution-model.md §7-§8; the firmware
side in `sw/ctrl/include/tsr/kernel.hpp`): the kernel calls `issue(op, after(a, b))` on the
control core (clk_rv); the push checks the unit's credit (queue depth, FIFO space), assigns the
ticket {unit, sequence} and writes the 256-bit op with up to two dependencies into the 32-entry
issue FIFO, which crosses into the core clock domain; `tsr_chk` checks every op in the order of
EX §8.1; the dispatcher appends it to the in-order queue of its unit; a queue head waits until its
dependencies are done and then starts on the unit. Credit, ticket and dependencies are what let
the firmware issue ahead and join later (D3).
Lower panel: the queue contents of the normative flash-attention sequence for a decode layer with
three KV chunks (docs/v2/03-attention-kv.md §6.6 step 4 and §7.1, the queue orders listed there):
DMA: KVST K, KVST V; QACT: the query rows; MVM: ATTQK 0, ATTQK 1, ATTPV 0, ATTQK 2, ATTPV 1,
ATTPV 2; VU: ASMAX 0, ASMAX 1, AMERGE 0, ASMAX 2, AMERGE 1, AMERGE 2, ANORM. Dependencies (AT §7.1
code): ATTQK k after QACT, the last ATTQK also after KVST K; ASMAX k after ATTQK k; ATTPV k after
ASMAX k, the last also after KVST V; AMERGE k after ATTPV k; ANORM after the last AMERGE. ATTPV
and AMERGE run one chunk behind ATTQK and ASMAX, so the MVM and the VU work on different chunks at
the same time. Queue depths are those of CC §7.4 (DMA 8, QACT 4, MVM 4, VU 16). The columns group
ops that can run at the same time; they are not a clock. The sequence is the numerical contract
of the kernel (AT §6.6); the order in which the firmware issues across chunks is not.
Status (STYLE.md): the issue path, the vector unit and the attention modes are planned RTL; the
sequence runs on the simulator (`sw/ctrl/kernels/k_fattn.cpp` and the simulator's queue model).
QACT exists in RTL and is solid; every other chip is a planned op and dashed.

Run: build/tools/svgvenv/bin/python docs/fig/src/issue.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544

# ---- upper panel: the issue path -----------------------------------------------------------------
A_Y = 56
BOX_W, BOX_G, BOX_H = 232, 52, 136
BX = [88 + i * (BOX_W + BOX_G) for i in range(5)]
BOX_Y = A_Y + 72
A_B = BOX_Y + BOX_H + 88

# ---- lower panel: the queues ---------------------------------------------------------------------
B_Y = A_B + 88
LANE_Y0 = B_Y + 96
LANE_H, LANE_G = 80, 24
CH_W, CH_G = 120, 14
CX0 = 288                             # first chip column
NCOL = 9
LANE_X0 = 72
B_LANES = LANE_Y0 + 4 * LANE_H + 3 * LANE_G
CAP_Y = B_LANES + 40
B_B = CAP_Y + 3 * 26 + 8
LEG_Y = B_B + 56
H = LEG_Y + 96

f = Fig("issue", W, H,
        title="From issue() to a unit, and the queues of one flash-attention layer",
        desc="Upper panel: a kernel on the control core calls issue(op, after(a, b)); the push "
             "checks the credit and assigns a ticket, writes the op with up to two dependencies "
             "into the 32-entry issue FIFO that crosses from the clk_rv domain to the core clock "
             "domain; the checker tsr_chk checks it; the dispatcher appends it to the in-order "
             "queue of its unit. Lower panel: the queue contents of one decode layer of flash "
             "attention with three KV chunks on the DMA, QACT, MVM and VU units: ATTQK and ATTPV "
             "on the MVM, ASMAX and AMERGE on the VU, with ATTPV and AMERGE one chunk behind.")

# ---- upper panel -------------------------------------------------------------------------------
f.panel(PX0, A_Y, PX1 - PX0, A_B - A_Y, label="ISSUE PATH · ONE TILE OP")
b1_x1 = BX[1] + BOX_W + 26
f.band(72, BOX_Y - 24, b1_x1 - 72, BOX_H + 24 + 56, label="CLK_RV · CONTROL CORE",
       label_xy=(88, BOX_Y + BOX_H + 40))
f.band(b1_x1 + 26, BOX_Y - 24, BX[4] + BOX_W + 16 - (b1_x1 + 26), BOX_H + 24 + 56,
       label="CORE CLOCK", label_xy=(b1_x1 + 26 + 16, BOX_Y + BOX_H + 40))
stages = [("control", "KERNEL", "issue(op, deps)\nreturns a ticket", "firmware"),
          ("control", "ISSUE PORT", "credit check\nticket per push", "push"),
          ("io", "ISSUE FIFO", "CDC · 32 entries\n2 dependencies", "FIFO"),
          ("control", "TSR_CHK", "checks every op\nEX §8.1 order", "checker"),
          ("control", "DISPATCHER", "in-order queue\nper unit", "queues")]
for (role, lab, sub, _), x in zip(stages, BX):
    f.box(x, BOX_Y, BOX_W, BOX_H, role, lab, sub=sub, planned=(lab != "KERNEL"), plan_tag=False,
          label_role="label_s")
for a, b in zip(BX, BX[1:]):
    f.arrow([(a + BOX_W - 12, BOX_Y + BOX_H / 2), (b + 12, BOX_Y + BOX_H / 2)])

# ---- lower panel -------------------------------------------------------------------------------
f.panel(PX0, B_Y, PX1 - PX0, B_B - B_Y,
        label="UNIT QUEUES · FLASH ATTENTION, ONE DECODE LAYER, THREE KV CHUNKS")
# the dispatcher feeds the queues of the lower panel (drawn after the panel, so its head shows)
f.arrow([(BX[4] + BOX_W / 2, A_B - 12), (BX[4] + BOX_W / 2, B_Y + 40)])
f.text(CX0, LANE_Y0 - 32, "QUEUE ORDER, LEFT TO RIGHT · SAME COLUMN: CAN RUN AT THE SAME TIME",
       "callout", valign="middle")

lanes = [("io", "DMA", "queue 8"), ("compute", "QACT", "queue 4"), ("compute", "MVM", "queue 4"),
         ("compute", "VU", "queue 16")]
# (lane, column, op, detail)
chips = [(0, 0, "KVST", "K row"), (0, 1, "KVST", "V row"),
         (1, 0, "QACT", "Q rows"),
         (2, 1, "ATTQK", "chunk 0"), (2, 2, "ATTQK", "chunk 1"), (2, 3, "ATTPV", "chunk 0"),
         (2, 4, "ATTQK", "chunk 2"), (2, 5, "ATTPV", "chunk 1"), (2, 6, "ATTPV", "chunk 2"),
         (3, 2, "ASMAX", "chunk 0"), (3, 3, "ASMAX", "chunk 1"), (3, 4, "AMERGE", "chunk 0"),
         (3, 5, "ASMAX", "chunk 2"), (3, 6, "AMERGE", "chunk 1"), (3, 7, "AMERGE", "chunk 2"),
         (3, 8, "ANORM", "output")]
lane_y = [LANE_Y0 + i * (LANE_H + LANE_G) for i in range(4)]
for (role, name, sub), y in zip(lanes, lane_y):
    f.band(LANE_X0, y - 8, CX0 + NCOL * (CH_W + CH_G) - LANE_X0, LANE_H + 16)
    f.box(88, y, 152, LANE_H, role, name, sub=sub, label_role="label_s", cut=10)
for lane, col, op, det in chips:
    x = CX0 + col * (CH_W + CH_G)
    solid = op == "QACT"
    f.box(x, lane_y[lane], CH_W, LANE_H, "compute" if lane in (1, 2, 3) else "io", op, sub=det,
          planned=not solid, plan_tag=False, label_role="label_s", cut=10)

f.text(CX0, CAP_Y, wrap(
    "Dependencies: ATTQK k after QACT (the last ATTQK also after KVST K) · ASMAX k after ATTQK k · "
    "ATTPV k after ASMAX k (the last also after KVST V) · AMERGE k after ATTPV k · ANORM after the "
    "last AMERGE. ATTPV and AMERGE run one chunk behind ATTQK and ASMAX.", 1180, "note"),
       "note", color=INK_SOFT)

f.legend(W / 2, LEG_Y, [("control", "CONTROL"), ("io", "DATA MOVEMENT"), ("compute", "COMPUTE"),
                        ("neutral", "PLANNED", True)])
f.text(W / 2, LEG_Y + 36, wrap(
    "The issue path, the vector unit and the attention modes are planned RTL (M6); the kernel and "
    "the queue model run on the simulator. QACT exists in RTL and is solid.", 1440),
       "note", anchor="middle", color=INK_SOFT)

f.save(scale=1)

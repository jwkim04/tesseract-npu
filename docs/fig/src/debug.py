"""Figure `debug`: debugging and observing the device over PCIe.

Content follows docs/design/05-debug.md §2 (overview diagram and the layer table), §3 (RISC-V debug:
DMI window, halt / resume / step, SBA memory access, breakpoints and watchpoints, halt at boot and
at launch), §4 (GDB RSP server: the debug service in the device-owning process, or the standalone
`tsr-dbg`, no OpenOCD), §5 (tile-level debug: stop at a push index, step op by op, serial mode,
tile breakpoints, the 512-entry dispatch history with the pushing C++ line, unit probes), §6 (log
ring and crash record), §8 (PERF slots 61-63, the hardware trace writer to a DRAM ring on channel
0, `tsr-prof`), §10.1 (register map: DMI 0xC00, DBG block 0x800-0xAFC, log ring 0x400-0x7FC in the
core CSR window) and docs/design/01-execution-model.md §2.4 (one process owns the device). The host
reaches every register through the PCIe CSR window and the transport; there is no JTAG on the
U200 (docs/platform-u200.md §1, docs/prof-dbg.md). The tile-level block observes and stops the
queue heads and changes nothing (D15).
Status (STYLE.md): GDB is third-party and solid. Everything else is planned: the debug service,
`tsr-dbg` and `tsr-prof` are tools still to write (W3b and later), and the debug module, the DBG
block and the trace writer are planned RTL; the simulator has models of the debug module and of
the DBG block (`sw/sim/tsr_dm.c`, `sw/sim/tsr_dbg.c`). The group panels carry no status tag of
their own: the card panel carries PLANNED once and the blocks inside are dashed.

Run: build/tools/svgvenv/bin/python docs/fig/src/debug.py
"""
import sys

sys.dont_write_bytecode = True        # keep docs/fig/src free of __pycache__
from tsrfig import INK_SOFT, Fig, wrap  # noqa: E402

W = 1600
PX0, PX1 = 56, 1544
X0, X1 = 88, 1512

# ---- host panel ---------------------------------------------------------------------------------
H_Y = 56
HB_Y, HB_H = H_Y + 56, 120
H_B = HB_Y + HB_H + 40
HOST = [("GDB", "RISC-V GDB", "neutral", False, 232),
        ("DEBUG SERVICE", "libtsr_dbg in the owning\nprocess, or tsr-dbg", "host", True, 360),
        ("TSR_TRANSPORT", "one mutex per device,\nthe only CSR path", "host", False, 304),
        ("TSR-PROF · TSR-LOG", "timeline, counters,\nfirmware logs", "host", True, 304)]
HG = 72
hx = [X0]
for _, _, _, _, w_ in HOST[:-1]:
    hx.append(hx[-1] + w_ + HG)

# ---- card panel ---------------------------------------------------------------------------------
C_Y = H_B + 88
WIN_Y, WIN_H = C_Y + 56, 96
G_Y = WIN_Y + WIN_H + 72              # group panels
GB_H = 104
GB_G = 48
G_H = 64 + 3 * GB_H + 2 * GB_G + 32
C_B = G_Y + G_H + 40
GW, GG = 440, 48
GX = [X0 + i * (GW + GG) for i in range(3)]
LEG_Y = C_B + 56
H = LEG_Y + 96

f = Fig("debug", W, H,
        title="Debugging and observing the device over PCIe",
        desc="On the host, GDB talks the remote serial protocol to a debug service in the process "
             "that owns the device, or to the standalone tsr-dbg; the service and the profiling "
             "and log tools use tsr_transport, which reaches the core CSR window over PCIe. In "
             "the window: the DMI at 0xC00 leads to the RISC-V debug module, the Ibex control "
             "core (halt, step, breakpoints) and the system bus access to the TCMs and the SPM; "
             "the DBG block at 0x800 to 0xAFC holds the stop logic for the queue heads, the "
             "512-entry dispatch history and the unit probes; the log ring, the performance "
             "counters and the trace writer to a DRAM ring serve observation. All of it is "
             "planned.")

# ---- host --------------------------------------------------------------------------------------
f.panel(PX0, H_Y, PX1 - PX0, H_B - H_Y, label="HOST PC")
for (lab, sub, role, planned, w_), x in zip(HOST, hx):
    f.box(x, HB_Y, w_, HB_H, role, lab, sub=sub, planned=planned, plan_tag=False,
          label_role="label_s")
mid = HB_Y + HB_H / 2
f.arrow([(hx[0] + HOST[0][4] - 12, mid), (hx[1] + 12, mid)], tail=True)
f.text((hx[0] + HOST[0][4] + hx[1]) / 2, mid - 40, "RSP", "callout", anchor="middle")
f.arrow([(hx[1] + HOST[1][4] - 12, mid), (hx[2] + 12, mid)])
f.arrow([(hx[2] + HOST[2][4] - 12, mid), (hx[3] + 12, mid)], dashed=True)

# ---- card --------------------------------------------------------------------------------------
f.panel(PX0, C_Y, PX1 - PX0, C_B - C_Y, label="ALVEO U200 · TSR_CORE")
f.tag(PX1 - 48, C_Y, "PLANNED", anchor="end", planned=True)
f.box(X0, WIN_Y, X1 - X0, WIN_H, "control", "CORE CSR WINDOW",
      sub="DMI 0xC00 · DBG block 0x800–0xAFC · log ring 0x400–0x7FC · PERF · Q_*",
      planned=True, plan_tag=False, label_role="label_s")
t_x = hx[2] + HOST[2][4] / 2
f.arrow([(t_x, HB_Y + HB_H - 12), (t_x, WIN_Y + 12)], tail=True)
f.text(t_x + 40, (H_B + C_Y) / 2 + 4, "PCIe · XRT", "callout", valign="middle")

groups = [
    ("RISC-V DEBUG", [
        ("control", "DEBUG MODULE", "riscv-dbg, reached by DMI"),
        ("control", "IBEX", "halt · resume · step\nbreakpoints · 2 watchpoints"),
        ("memory", "SBA", "TCMs · SPM · windows\nall debugger memory access")]),
    ("TILE-LEVEL DEBUG", [
        ("control", "STOP LOGIC", "stop at a push index · step op\nserial mode · tile breakpoints"),
        ("control", "QUEUE HEADS", "held by the stop logic;\nthe debugger changes nothing"),
        ("compute", "HISTORY · PROBES", "512 ops with the pushing\nC++ line · unit probes")]),
    ("OBSERVE", [
        ("memory", "LOG RING", "256 words: firmware log\nrecords, trap and crash record"),
        ("control", "PERF", "counters incl. slots 61–63\nper launch"),
        ("memory", "TRACE WRITER", "32-byte records to a\nDRAM ring on channel 0")]),
]
for (gname, blocks), gx in zip(groups, GX):
    f.panel(gx, G_Y, GW, G_H, label=gname, fill=None, cut=16)
    by = G_Y + 48
    for role, lab, sub in blocks:
        f.box(gx + 24, by, GW - 48, GB_H, role, lab, sub=sub, planned=True, plan_tag=False,
              label_role="label_s", cut=12)
        by += GB_H + GB_G
    # one arrow from the window into the group, right of the group's name tag
    ax = gx + GW - 96
    f.arrow([(ax, WIN_Y + WIN_H - 8), (ax, G_Y + 12)], tail=True)
    if gname != "OBSERVE":                         # the first block drives the second
        yy = G_Y + 48 + GB_H
        f.arrow([(ax, yy - 8), (ax, yy + GB_G + 8)], dashed=True)

f.legend(W / 2, LEG_Y, [("neutral", "THIRD-PARTY"), ("host", "HOST SOFTWARE"),
                        ("control", "CONTROL"), ("memory", "MEMORY"), ("compute", "COMPUTE"),
                        ("neutral", "PLANNED", True)], gap=36)
f.text(W / 2, LEG_Y + 36, wrap(
    "Designed in docs/design/05-debug.md. The simulator models the debug module and the DBG block; "
    "the RSP server, tsr-dbg, tsr-prof and the RTL are planned (M6).", 1440),
       "note", anchor="middle", color=INK_SOFT)

f.save(scale=1)

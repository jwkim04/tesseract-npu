#!/usr/bin/env python3
"""rtl/platform/alveo_xrt/gen_tops.py — generate the U200 Vitis kernel tops, their kernel.xml and
the testbench wrappers from one table (docs/platform-u200.md §6).

    rtl/platform/alveo_xrt/gen_tops.py           write the files below
    rtl/platform/alveo_xrt/gen_tops.py --check   exit 1 if a generated file is missing or stale

Kernels (the lead's kernel contract): module name = kernel name.
    tsr_u200_k1   NUM_CH 1, T_MAX 1   1-channel bring-up bitstream (m_axi_ch0)
    tsr_u200_k4   NUM_CH 4, T_MAX 8   the target (m_axi_ch0 .. m_axi_ch3)
Both: the U200 bitstream parameters of docs/rtl-m3.md §14.2 (SPM 4 MiB with SPM_RD_LAT 4,
MVM_BC_PIPE 2, MVM_RES_PIPE 2, MVM_MAC_LAT 9, RST_PIPE 3: the core's reset stages with per-block
copies), CH_BYTES 16 GiB, the other tsr_core parameters at the U200 defaults (tsr_pkg TSR_DEF_*),
AXI_SLICES 2, RST_STAGES 3. With RST_PIPE 3, ap_rst_n must stay low for >= 4 cycles
(docs/platform-u200.md §5).

The kernels are `ap_ctrl_none` with interrupt="false"
and only the ch<c> buffer args (EX §9.1); two clocks: ap_clk / ap_rst_n (the core and the adapter)
and ap_clk_2 / ap_rst_n_2 (the control core, tsr_core clk_rv / rst_rv_n; CC §11.1, §13.3). The
platform build stays disabled until Y5 (hw/u200/build_xclbin.sh), whose first smoke link confirms
ap_ctrl_none (fallback: EX §9.1); the link configurations already request both clocks.

Outputs:
    rtl/platform/alveo_xrt/<kernel>.v         named-port Verilog-2001 top (ap_clk, ap_rst_n,
                                              s_axi_control_*, m_axi_ch<c>_*) around tsr_xrt_kernel
    rtl/platform/alveo_xrt/kernel_<k>.xml     Vitis kernel description (ch<c> args at 0x040 + 8c,
                                              control range 0x2000, ap_ctrl_none, no interrupt)
    tb/platform/sv/kwrap_<kernel>.sv          simulation wrapper: the named top with the packed
                                              port list of tsr_xrt_kernel, so one C++ testbench
                                              drives the generic body and both packaged tops
                                              (linted now; the kernel bench returns in Y5)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TBSV = REPO / "tb" / "platform" / "sv"

KERNELS = {
    # VU lane knobs (VU §7.10, the measured out-of-context areas of tb/vu/viv): k4 16 / 8, k1 8 / 4.
    # VU_LANE_PIPE (CC §14.1 remedy 3): k4 puts the 16 lanes in SLR0 behind 2 register stages on each side
    # (hw/u200/k4_floorplan.tcl), k1 keeps the whole VU in SLR1
    "tsr_u200_k1": dict(short="k1", NUM_CH=1, T_MAX=1, VU_EXP_LANES=8, VU_BACK_LANES=4, VU_LANE_PIPE=0),
    "tsr_u200_k4": dict(short="k4", NUM_CH=4, T_MAX=8, VU_EXP_LANES=16, VU_BACK_LANES=8, VU_LANE_PIPE=2),
}
# The U200 bitstream parameters (docs/rtl-m3.md §14.2; tb/fastreplay/run.py PACKAGED carries the same)
COMMON = dict(SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9, RST_PIPE=3, AXI_SLICES=2, RST_STAGES=3)
FIXED_DOC = ("SPM 4 MiB (default) with SPM_RD_LAT 4, MVM_BC_PIPE 2, MVM_RES_PIPE 2,\n// MVM_MAC_LAT 9, "
             "RST_PIPE 3 (U200 bitstream parameters, docs/rtl-m3.md §14.2), CH_BYTES 16 GiB (default),\n"
             "// the other tsr_core parameters at the U200 defaults, AXI_SLICES 2, RST_STAGES 3;\n"
             "// ap_rst_n low for >= 4 cycles (RST_PIPE + 1, docs/platform-u200.md §5)")

BEAT = 512
# (signal, direction seen from the kernel, width)
AXI = [
    ("awvalid", "out", 1), ("awready", "in", 1), ("awaddr", "out", 64), ("awlen", "out", 8),
    ("awsize", "out", 3), ("awburst", "out", 2), ("awlock", "out", 1), ("awcache", "out", 4),
    ("awprot", "out", 3), ("awqos", "out", 4), ("awid", "out", 4),
    ("wvalid", "out", 1), ("wready", "in", 1), ("wdata", "out", BEAT), ("wstrb", "out", BEAT // 8),
    ("wlast", "out", 1),
    ("bvalid", "in", 1), ("bready", "out", 1), ("bresp", "in", 2), ("bid", "in", 4),
    ("arvalid", "out", 1), ("arready", "in", 1), ("araddr", "out", 64), ("arlen", "out", 8),
    ("arsize", "out", 3), ("arburst", "out", 2), ("arlock", "out", 1), ("arcache", "out", 4),
    ("arprot", "out", 3), ("arqos", "out", 4), ("arid", "out", 4),
    ("rvalid", "in", 1), ("rready", "out", 1), ("rdata", "in", BEAT), ("rresp", "in", 2),
    ("rlast", "in", 1), ("rid", "in", 4),
]
AXIL = [
    ("awvalid", "in", 1), ("awready", "out", 1), ("awaddr", "in", 13),
    ("wvalid", "in", 1), ("wready", "out", 1), ("wdata", "in", 32), ("wstrb", "in", 4),
    ("bvalid", "out", 1), ("bready", "in", 1), ("bresp", "out", 2),
    ("arvalid", "in", 1), ("arready", "out", 1), ("araddr", "in", 13),
    ("rvalid", "out", 1), ("rready", "in", 1), ("rdata", "out", 32), ("rresp", "out", 2),
]


def _w(n: int) -> str:
    return "" if n == 1 else "[%d:0]" % (n - 1)


def _dir(d: str) -> str:
    return "input  logic" if d == "in" else "output logic"


def _vw(n: int) -> str:
    return "" if n == 1 else "[%d:0]" % (n - 1)


def gen_top(name: str, k: dict) -> str:
    """Verilog-2001 top (like the RTL Kernel Wizard's): named ports, tsr_xrt_kernel inside."""
    C = k["NUM_CH"]
    params = dict(NUM_CH=C, T_MAX=k["T_MAX"], VU_EXP_LANES=k["VU_EXP_LANES"], VU_BACK_LANES=k["VU_BACK_LANES"],
                  VU_LANE_PIPE=k["VU_LANE_PIPE"], **COMMON)
    L = []
    L.append("// %s.v — GENERATED by rtl/platform/alveo_xrt/gen_tops.py; do not edit." % name)
    L.append("//")
    L.append("// Vitis RTL kernel `%s` (module name = kernel name): tsr_xrt_kernel with NUM_CH %d, T_MAX %d," % (
        name, C, k["T_MAX"]))
    L.append("// VU lanes %d / %d (front lanes with an EXP core / back lanes, VU §7.10) behind %d pipe stages on each side," % (
        k["VU_EXP_LANES"], k["VU_BACK_LANES"], k["VU_LANE_PIPE"]))
    L.append("// %s." % FIXED_DOC)
    L.append("// Ports: ap_clk, ap_rst_n (active low, synchronous: the core and the adapter), ap_clk_2,")
    L.append("// ap_rst_n_2 (the control core's clock and reset), s_axi_control (AXI4-Lite, 32-bit")
    L.append("// data, 13-bit address), m_axi_ch<c> for c < %d (AXI4, 512-bit data, 64-bit address, 4-bit ID)." % C)
    L.append("// Register map and protocol: docs/design EX §9 (ap_ctrl_none); packaging: docs/platform-u200.md;")
    L.append("// kernel_%s.xml describes the args." % k["short"])
    L.append("// Plain Verilog-2001 (the IP packager and IP integrator expect a Verilog top); the body is SystemVerilog.")
    L.append("")
    L.append("`default_nettype none")
    L.append("")
    L.append("module %s (" % name)
    ports = ["  input  wire          ap_clk", "  input  wire          ap_rst_n", "  input  wire          ap_clk_2",
             "  input  wire          ap_rst_n_2"]
    for s_, d, w in AXIL:
        ports.append("  %-6s wire %-8s s_axi_control_%s" % ("input" if d == "in" else "output", _vw(w), s_))
    for c in range(C):
        for s_, d, w in AXI:
            ports.append("  %-6s wire %-8s m_axi_ch%d_%s" % ("input" if d == "in" else "output", _vw(w), c, s_))
    L.append(",\n".join(ports))
    L.append(");")
    L.append("")
    L.append("  // per-channel signals packed as tsr_xrt_kernel's [NUM_CH][W] ports (channel c = bits [c*W +: W])")
    for s_, d, w in AXI:
        L.append("  wire [%d:0] %s;" % (C * w - 1, s_))
    L.append("")
    for s_, d, w in AXI:
        for c in range(C):
            if d == "out":
                L.append("  assign m_axi_ch%d_%s = %s[%d +: %d];" % (c, s_, s_, c * w, w))
            else:
                L.append("  assign %s[%d +: %d] = m_axi_ch%d_%s;" % (s_, c * w, w, c, s_))
    L.append("")
    L.append("  tsr_xrt_kernel #(")
    L.append(",\n".join("    .%s(%s)" % (p, v) for p, v in params.items()))
    L.append("  ) u_kernel (")
    conns = ["    .ap_clk(ap_clk)", "    .ap_rst_n(ap_rst_n)", "    .ap_clk_2(ap_clk_2)", "    .ap_rst_n_2(ap_rst_n_2)"]
    conns += ["    .s_axi_control_%s(s_axi_control_%s)" % (s_, s_) for s_, _, _ in AXIL]
    conns += ["    .m_axi_%s(%s)" % (s_, s_) for s_, _, _ in AXI]
    L.append(",\n".join(conns))
    L.append("  );")
    L.append("")
    L.append("endmodule")
    L.append("")
    L.append("`default_nettype wire")
    return "\n".join(L) + "\n"


def gen_wrap(name: str, k: dict) -> str:
    """Simulation wrapper: the port list of tsr_xrt_kernel (packed per channel) around the named top."""
    C = k["NUM_CH"]
    L = []
    L.append("// kwrap_%s.sv — GENERATED by rtl/platform/alveo_xrt/gen_tops.py; do not edit." % name)
    L.append("//")
    L.append("// Testbench-only wrapper: the packaged kernel %s with the packed port list of" % name)
    L.append("// tsr_xrt_kernel (NUM_CH = %d), so one bench (tb/platform) drives the generic" % C)
    L.append("// body and the named top. Not a synthesis source.")
    L.append("")
    L.append("module kwrap_%s #(" % name)
    L.append("  parameter int unsigned NUM_CH = %d   // fixed by the kernel; present for the bench" % C)
    L.append(") (")
    ports = ["  input  logic        ap_clk", "  input  logic        ap_rst_n", "  input  logic        ap_clk_2",
             "  input  logic        ap_rst_n_2"]
    for s, d, w in AXIL:
        ports.append("  %s %-8s s_axi_control_%s" % (_dir(d), _w(w), s))
    for s, d, w in AXI:
        ports.append("  %s [NUM_CH-1:0]%s m_axi_%s" % (_dir(d), "" if w == 1 else "[%d:0]" % (w - 1), s))
    L.append(",\n".join(ports))
    L.append(");")
    L.append("")
    L.append("  if (NUM_CH != %d) begin : g_chk_ch" % C)
    L.append("    $fatal(1, \"kwrap_%s: NUM_CH is %d\");" % (name, C))
    L.append("  end")
    L.append("")
    L.append("  %s u_top (" % name)
    conns = ["    .ap_clk", "    .ap_rst_n", "    .ap_clk_2", "    .ap_rst_n_2"]
    conns += ["    .s_axi_control_%s" % s for s, _, _ in AXIL]
    for c in range(C):
        conns += ["    .m_axi_ch%d_%s(m_axi_%s[%d])" % (c, s, s, c) for s, _, _ in AXI]
    L.append(",\n".join(conns))
    L.append("  );")
    L.append("")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def gen_xml(name: str, k: dict) -> str:
    C = k["NUM_CH"]
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!-- kernel_%s.xml — GENERATED by rtl/platform/alveo_xrt/gen_tops.py; do not edit.' % k["short"],
         '     Vitis RTL kernel %s: s_axi_control register map = docs/design EX §9.2 (ap_ctrl_none: the host' % name,
         '     uses register access only, xrt::ip). The only args are the ch<c> buffers; the core CSR',
         '     window at 0x1000..0x1FFF has no args. -->',
         '<root versionMajor="1" versionMinor="6">',
         '  <kernel name="%s" language="ip_c" vlnv="tesseract:kernel:%s:1.0" attributes="" '
         'preferredWorkGroupSizeMultiple="0" workGroupSize="1" interrupt="false" hwControlProtocol="ap_ctrl_none">'
         % (name, name),
         '    <ports>',
         '      <port name="s_axi_control" mode="slave" range="0x2000" dataWidth="32" portType="addressable" base="0x0"/>']
    for c in range(C):
        L.append('      <port name="m_axi_ch%d" mode="master" range="0xFFFFFFFFFFFFFFFF" dataWidth="%d" '
                 'portType="addressable" base="0x0"/>' % (c, BEAT))
    L.append('    </ports>')
    L.append('    <args>')
    args = [("ch%d" % c, 1, "m_axi_ch%d" % c, 8, 0x040 + 8 * c, "void*") for c in range(C)]
    for i, (n, aq, port, size, off, ty) in enumerate(args):
        L.append('      <arg name="%s" addressQualifier="%d" id="%d" port="%s" size="0x%x" offset="0x%03X" '
                 'type="%s" hostOffset="0x0" hostSize="0x%x"/>' % (n, aq, i, port, size, off, ty, size))
    L.append('    </args>')
    L.append('  </kernel>')
    L.append('</root>')
    return "\n".join(L) + "\n"


def outputs() -> dict:
    out = {}
    for name, k in KERNELS.items():
        out[HERE / ("%s.v" % name)] = gen_top(name, k)
        out[HERE / ("kernel_%s.xml" % k["short"])] = gen_xml(name, k)
        out[TBSV / ("kwrap_%s.sv" % name)] = gen_wrap(name, k)
    return out


def main(argv) -> int:
    check = "--check" in argv
    bad = 0
    for path, text in outputs().items():
        if check:
            if not path.is_file() or path.read_text() != text:
                print("gen_tops.py: %s is missing or stale (run rtl/platform/alveo_xrt/gen_tops.py)" % path)
                bad += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file() or path.read_text() != text:
                path.write_text(text)
                print("wrote %s" % path.relative_to(REPO))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

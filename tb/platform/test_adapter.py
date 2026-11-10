"""tb/platform — the Alveo XRT adapter (rtl/platform/alveo_xrt) against the mock adapter model
(sw/transport/tsr_kdev_mock.cpp, TSR_DEVICE=mock; docs/design/01-execution-model.md "EX" §9,
docs/platform-u200.md §2-§5).

  test_adapter_regs    tsr_xrt_ctrl (s_axi_control, AXI4-Lite) against the mock model: the same random
                       script of register accesses (reserved 0x000-0x03C: no AP_CTRL, ap_ctrl_none;
                       ch<c>_base lo / hi; ADAPTER_CTRL.MEM_EN with aligned and misaligned bases;
                       base writes while MEM_EN (BASE_LOCKED); ADAPTER_STATUS W1C and the error count;
                       ADAPTER_VERSION; unmapped offsets; the core CSR window) runs on the mock
                       (tb/platform/adapter/mock_regs.cpp, linked against build/dev's transport and
                       simulator libraries) and on the RTL, whose CSR port is answered by a simulator
                       core (as the mock's window is) with plat_err_i driven by the RTL's plat_err;
                       every read must match. RTL-only: REQ_HELD, byte strobes, latched bases.
  test_adapter_kernel  tsr_xrt_kernel (reduced core; XSLR_REG 0 and 1): ap_clk_2 is the control core's clock (RV_FREQ,
                       STATUS.RV_CLK_OK), MEM_EN latches the bases into the bridges, the window, the
                       reserved range (no AP_CTRL), ADAPTER_VERSION from tsr_tables_pkg.
  test_adapter_kernel_dual  the same kernel with MVM_DUAL 1: CAPS bit 16, DUAL_DIS to the core, no B traffic without an MVM.
  test_adapter_portb   tsr_xrt_kernel with MVM_DUAL 1 and a stub core (tb/platform/sv/stub_core_portb.sv) whose memory ports
                       the bench drives: read traffic on the main port and on the read-only port B of every channel.
                       Per channel: AR on m_axib_*[c] only (m_axi_chb<c> of the tops), with the base of channel c; R data
                       and RRESP back to channel c; MEM_EN low holds the request (nothing leaves, REQ_HELD) and MEM_EN
                       high lets it through; an RRESP error
                       and an unexpected R beat reach ADAPTER_STATUS; ap_rst_n clears a bridge that holds an AR and a
                       count of outstanding bursts. Every check runs on the main port too: port B must behave as it does.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "issue"))

from common import tables as T                            # noqa: E402
from common.runner import REPO, build_root, rtl_args, run   # noqa: E402

M32 = 0xFFFFFFFF
NUM_CH = 4


# =================================================================================================
# The access script and the mock model
# =================================================================================================
WINDOW = [0x000, 0x004, 0x024, 0x028, 0x034, 0x040, 0x044, 0x05C, 0x098]       # core CSR offsets
WINDOW_RW = [0x028, 0x034, 0x040, 0x044, 0x05C, 0x098]


def script(seed, num_ch):
    """A random sequence of s_axi_control accesses: ('W', off, val) / ('R', off) / ('E', bits, cnt)."""
    r = random.Random(seed)
    ops = []
    bases = [0x40 + 8 * c for c in range(num_ch)]
    for i in range(500):
        x = r.random()
        if x < 0.22:
            c = r.randrange(num_ch)
            hi = r.random() < 0.3
            v = r.getrandbits(32) if hi else (r.getrandbits(20) << 12 if r.random() < 0.8 else r.getrandbits(32))
            ops.append(("W", bases[c] + 4 * hi, v))
        elif x < 0.32:
            ops.append(("W", 0x0E0, r.choice([0, 1, 1, 1, 0xFFFFFFFE, 3])))
        elif x < 0.40:
            ops.append(("W", 0x0F0, r.choice([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x3F, 0x10000, 0xFFFF0000, M32,
                                              r.getrandbits(32)])))
        elif x < 0.47:
            k = r.randrange(3)
            ops.append(("E", 1 << k, 0 if k == 2 else r.randint(1, 3)))
        elif x < 0.52:
            ops.append(("W", r.choice([0x000, 0x010, 0x03C, 0x0E4, 0x0F4, 0x0F8, 0x0FC, 0x0A0, 0x0C0, 0x100, 0x104, 0x118, 0x17C]),
                        r.getrandbits(32)))
        elif x < 0.60:
            ops.append(("W", 0x1000 + r.choice(WINDOW_RW), r.getrandbits(32)))
        elif x < 0.75:
            ops.append(("R", 0x1000 + r.choice(WINDOW)))
        else:
            ops.append(("R", r.choice([0x000, 0x004, 0x03C, 0x0E0, 0x0F0, 0x0F4, 0x0E4, 0x0FC, 0x0A0, 0x100, 0x118, 0x17C] +
                                      [b + 4 * h for b in bases for h in (0, 1)] + [0x40 + 8 * num_ch])))
    for b in bases:
        ops += [("R", b), ("R", b + 4)]
    ops += [("R", 0x0E0), ("R", 0x0F0), ("R", 0x1024)]
    # plat_err from the count alone (bits cleared, count kept), then nothing
    ops += [("E", 1, 2), ("W", 0x0F0, 0x7), ("R", 0x0F0), ("R", 0x1024), ("W", 0x0F0, 0xFFFF0000), ("R", 0x0F0),
            ("R", 0x1024)]
    return ops


def mock_build():
    """Builds tb/platform/adapter/mock_regs against build/dev's static libraries (or $TSR_MOCK_REGS)."""
    exe = os.environ.get("TSR_MOCK_REGS")
    if exe:
        return Path(exe)
    b = REPO / "build" / "dev"
    libs = [b / "sw/transport/libtsr_transport.a", b / "sw/ctrl/host/libtsr_elf.a", b / "sw/sim/libtsr_sim.a",
            b / "sw/sim/libtsr_rv.a", b / "sw/sim/libtsr_ref.a"]
    missing = [str(x) for x in libs if not x.exists()]
    if missing:
        pytest.skip("build/dev libraries missing (%s): cmake --build build/dev" % missing[0])
    out = build_root() / "adapter_mock" / "mock_regs"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = HERE / "adapter" / "mock_regs.cpp"
    newest = max(x.stat().st_mtime for x in libs + [src])
    if not out.exists() or out.stat().st_mtime < newest:
        cmd = ["c++", "-std=c++17", "-O1", "-I", str(REPO / "sw/transport"), "-I", str(REPO / "sw/sim/include"),
               "-I", str(REPO / "sw/sim"), str(src)] + [str(x) for x in libs] + ["-lpthread", "-lm", "-o", str(out)]
        subprocess.run(cmd, check=True)
    return out


def mock_run(ops, num_ch):
    exe = mock_build()
    lines = []
    for op in ops:
        if op[0] == "W":
            lines.append("W %x %x" % (op[1], op[2] & M32))
        elif op[0] == "R":
            lines.append("R %x" % op[1])
        else:
            lines.append("E %x %x" % (op[1], op[2]))
    r = subprocess.run([str(exe), str(num_ch)], input="\n".join(lines) + "\n", capture_output=True, text=True,
                       check=True, env=dict(os.environ, TSR_SIM_THREADS="1"))
    return [int(ln.split()[1], 16) for ln in r.stdout.splitlines() if not ln.startswith("bind_errors")]


# =================================================================================================
# AXI4-Lite master and the CSR-port responder (cocotb)
# =================================================================================================
class Axil:
    def __init__(self, dut, clk, pfx, rnd):
        self.d, self.clk, self.p, self.r = dut, clk, pfx, rnd

    def _s(self, n):
        return getattr(self.d, self.p + n)

    async def write(self, addr, data, strb=0xF):
        r = self.r
        await FallingEdge(self.clk)
        aw_done = w_done = False
        self._s("awvalid").value = 1
        self._s("awaddr").value = addr
        delay_w = r.random() < 0.3
        if not delay_w:
            self._s("wvalid").value = 1
        self._s("wdata").value = data & M32
        self._s("wstrb").value = strb
        self._s("bready").value = int(r.random() < 0.7)
        n = 0
        while not (aw_done and w_done):
            await RisingEdge(self.clk)
            if int(self._s("awvalid").value) and int(self._s("awready").value):
                aw_done = True
            if int(self._s("wvalid").value) and int(self._s("wready").value):
                w_done = True
            await FallingEdge(self.clk)
            if aw_done:
                self._s("awvalid").value = 0
            if w_done:
                self._s("wvalid").value = 0
            elif delay_w and n >= 2:
                self._s("wvalid").value = 1
            n += 1
            assert n < 2000, "AXI4-Lite write handshake stuck"
        while True:
            self._s("bready").value = int(r.random() < 0.7)
            await RisingEdge(self.clk)
            if int(self._s("bvalid").value) and int(self._s("bready").value):
                resp = int(self._s("bresp").value)
                break
            await FallingEdge(self.clk)
        await FallingEdge(self.clk)
        self._s("bready").value = 0
        return resp

    async def read(self, addr):
        r = self.r
        await FallingEdge(self.clk)
        self._s("arvalid").value = 1
        self._s("araddr").value = addr
        while True:
            await RisingEdge(self.clk)
            if int(self._s("arready").value):
                break
        await FallingEdge(self.clk)
        self._s("arvalid").value = 0
        while True:
            self._s("rready").value = int(r.random() < 0.7)
            await RisingEdge(self.clk)
            if int(self._s("rvalid").value) and int(self._s("rready").value):
                v, resp = int(self._s("rdata").value), int(self._s("rresp").value)
                break
            await FallingEdge(self.clk)
        await FallingEdge(self.clk)
        self._s("rready").value = 0
        return v, resp


async def csr_responder(dut, sim, log):
    """The core CSR port (docs/rtl-core.md §6): a registered one-cycle csr_ready with csr_rdata,
    answered by a simulator core (tsr_sim_csr_read / _write), plat_err_i = the RTL's plat_err."""
    import issue_sim as S                                  # noqa: F401
    pe = -1
    while True:
        await FallingEdge(dut.clk)
        p = int(dut.plat_err.value)
        if p != pe:
            sim.L.tsr_sim_set_plat_err(sim.sim, p)
            pe = p
        dut.csr_ready.value = 0
        if int(dut.csr_valid.value):
            for _ in range(random.randint(0, 3)):
                await FallingEdge(dut.clk)
            a, we, wd = int(dut.csr_addr.value), int(dut.csr_we.value), int(dut.csr_wdata.value)
            if we:
                sim.csr_write(a, wd)
                v = 0
            else:
                v = sim.csr_read(a)
            log.append((we, a, wd))
            dut.csr_rdata.value = v
            dut.csr_ready.value = 1
            await FallingEdge(dut.clk)
            dut.csr_ready.value = 0


async def pulse(dut, name, n):
    for _ in range(n):
        await FallingEdge(dut.clk)
        getattr(dut, name).value = 1
        await FallingEdge(dut.clk)
        getattr(dut, name).value = 0


@cocotb.test()
async def adapter_regs(dut):
    """tsr_xrt_ctrl against the mock adapter model: every read of the script must match."""
    import issue_sim as S
    c = json.loads(os.environ["TSR_ADAPTER_CASE"])
    ops = [tuple(x) for x in c["ops"]]
    exp = c["exp"]
    num_ch = c["num_ch"]
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for n in ("s_awvalid", "s_wvalid", "s_bready", "s_arvalid", "s_rready", "s_awaddr", "s_araddr", "s_wdata",
              "s_wstrb", "csr_ready", "csr_rdata", "ev_rresp_err", "ev_bresp_err", "ev_unexpected", "ev_req_held",
              "ev2_rresp_err", "ev2_unexpected"):
        getattr(dut, n).value = 0
    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    sim = S.Sim(S.config(num_ch=num_ch, t_max=1, ch_bytes=1 << 30, caps=S.lib().tsr_sim_caps_implemented()))
    log = []
    cocotb.start_soon(csr_responder(dut, sim, log))
    rng = random.Random(int(os.environ.get("TSR_SEED", "1")))
    m = Axil(dut, dut.clk, "s_", rng)
    bad = []
    k = 0
    ne = 0
    for op in ops:
        if op[0] == "W":
            resp = await m.write(op[1], op[2])
            if resp:
                bad.append("write %#x BRESP %d" % (op[1], resp))
        elif op[0] == "E":
            name = {1: "ev_rresp_err", 2: "ev_bresp_err", 4: "ev_unexpected"}[op[1]]
            ne += 1
            if op[1] != 2 and ne % 2 == 0:          # every other R event comes from the read-only bridges (ev2_*)
                name = name.replace("ev_", "ev2_")
            await pulse(dut, name, max(1, op[2]))
            for _ in range(4):
                await RisingEdge(dut.clk)
        else:
            v, resp = await m.read(op[1])
            if resp or v != exp[k]:
                bad.append("read #%d %#x: RTL %#x (RRESP %d), mock %#x" % (k, op[1], v, resp, exp[k]))
            k += 1
    # RTL-only: ADAPTER_CTRL.DUAL_DIS drives the dual_en output (reset 1; a write of 1 to bit 0 only keeps it 1)
    await m.write(0x0E0, 0)
    await m.write(0x0E0, 2)
    if int(dut.dual_en.value):
        bad.append("dual_en high after DUAL_DIS = 1")
    v, _ = await m.read(0x0E0)
    if v & 2 == 0:
        bad.append("ADAPTER_CTRL.DUAL_DIS reads back 0: %#x" % v)
    await m.write(0x0E0, 0)
    if not int(dut.dual_en.value):
        bad.append("dual_en low after DUAL_DIS = 0")
    # RTL-only: the events of the read-only bridges (ev2_*) set the same status bits and count like the main bridges' own
    await m.write(0x0F0, M32)
    v0, _ = await m.read(0x0F0)
    await pulse(dut, "ev2_rresp_err", 1)
    for _ in range(4):
        await RisingEdge(dut.clk)
    v, _ = await m.read(0x0F0)
    if not v & T.ADAPTER_STATUS_RRESP_ERR or (v >> 16) != (v0 >> 16) + 1:
        bad.append("ev2_rresp_err: ADAPTER_STATUS %#x (before %#x)" % (v, v0))
    await m.write(0x0F0, M32)
    await pulse(dut, "ev2_unexpected", 1)
    for _ in range(4):
        await RisingEdge(dut.clk)
    v, _ = await m.read(0x0F0)
    if not v & T.ADAPTER_STATUS_ORPHAN_RESP:
        bad.append("ev2_unexpected: ADAPTER_STATUS %#x" % v)
    # RTL-only: REQ_HELD (the kernel's gate event), byte strobes on a base, a partial window write
    await m.write(0x0F0, M32)
    await pulse(dut, "ev_req_held", 1)
    for _ in range(3):
        await RisingEdge(dut.clk)
    v, _ = await m.read(0x0F0)
    if not v & T.ADAPTER_STATUS_REQ_HELD:
        bad.append("REQ_HELD not set: %#x" % v)
    await m.write(0x0E0, 0)
    await m.write(0x040, 0x12345000)
    await m.write(0x040, 0xFFFFFFFF, strb=0b0010)
    v, _ = await m.read(0x040)
    if v != 0x1234FF00:
        bad.append("byte strobes on ch0_base lo: %#x" % v)
    n0 = len(log)
    await m.write(0x1028, 0xABCD0000, strb=0x3)
    if len(log) != n0:
        bad.append("a partial window write reached the CSR port")
    # latched bases: the bridges see the bases of the MEM_EN edge, not later writes
    for c_ in range(num_ch):
        await m.write(0x40 + 8 * c_, 0x1000 * (c_ + 1))
        await m.write(0x44 + 8 * c_, c_)
    await m.write(0x0E0, 1)
    await m.write(0x040, 0x7000)                          # ignored (BASE_LOCKED)
    for c_ in range(num_ch):
        got = int(dut.ch_base.value) >> (64 * c_) & ((1 << 64) - 1)
        if got != (c_ << 32) | 0x1000 * (c_ + 1):
            bad.append("ch%d latched base %#x" % (c_, got))
    if not int(dut.mem_en.value):
        bad.append("MEM_EN output low after an aligned MEM_EN write")
    sim.close()
    for x in bad[:20]:
        dut._log.error(x)
    assert not bad, "%d mismatches with the mock adapter model" % len(bad)


@cocotb.test()
async def adapter_kernel(dut):
    """tsr_xrt_kernel: ap_clk_2 drives the control core (RV_FREQ, STATUS.RV_CLK_OK), MEM_EN latches the
    bases, the reserved range reads 0 (no AP_CTRL), ADAPTER_VERSION, the window."""
    cocotb.start_soon(Clock(dut.ap_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.ap_clk_2, 16, unit="ns").start())
    pfx = "s_axi_control_"
    for n in ("awvalid", "wvalid", "bready", "arvalid", "rready", "awaddr", "araddr", "wdata", "wstrb"):
        getattr(dut, pfx + n).value = 0
    dual = os.environ.get("TSR_KDUAL") == "1"
    for n in ("m_axi_awready", "m_axi_wready", "m_axi_bvalid", "m_axi_arready", "m_axi_rvalid", "m_axi_bresp",
              "m_axi_bid", "m_axi_rdata", "m_axi_rresp", "m_axi_rlast", "m_axi_rid"):
        getattr(dut, n).value = 0
    if dual:
        for n in ("m_axib_arready", "m_axib_rvalid", "m_axib_rresp", "m_axib_rlast", "m_axib_rid", "m_axib_rdata"):
            getattr(dut, n).value = 0
    dut.ap_rst_n.value = 0
    dut.ap_rst_n_2.value = 0
    for _ in range(20):
        await RisingEdge(dut.ap_clk)
    dut.ap_rst_n.value = 1
    dut.ap_rst_n_2.value = 1
    m = Axil(dut, dut.ap_clk, pfx, random.Random(1))
    bad = []
    v, _ = await m.read(0x000)
    if v != 0:
        bad.append("0x000 (no AP_CTRL) reads %#x" % v)
    v, _ = await m.read(0x0F4)
    if v != (T.ADAPTER_MAJOR << 16) | int(os.environ.get("TSR_KCH", "1")):
        bad.append("ADAPTER_VERSION %#x" % v)
    v, _ = await m.read(0x1000)
    if v != 0x54535231:
        bad.append("window ID %#x" % v)
    for _ in range(70000):
        await RisingEdge(dut.ap_clk)
    st, _ = await m.read(0x1000 + T.CSR_STATUS)
    fq, _ = await m.read(0x1000 + T.CSR_RV_FREQ)
    want = 65536 * 10 // 16
    if not st & T.STATUS_RV_CLK_OK or abs(fq - want) > 64:
        bad.append("ap_clk_2: STATUS %#x RV_FREQ %d (about %d expected)" % (st, fq, want))
    # bases and MEM_EN
    await m.write(0x040, 0x1234)                         # misaligned: MEM_EN refused
    await m.write(0x0E0, 1)
    v, _ = await m.read(0x0E0)
    s, _ = await m.read(0x0F0)
    if v != 0 or not s & T.ADAPTER_STATUS_BASE_MISALIGNED:
        bad.append("misaligned base: ADAPTER_CTRL %#x STATUS %#x" % (v, s))
    await m.write(0x040, 0x40000)
    await m.write(0x044, 0x2)
    await m.write(0x0E0, 1)
    v, _ = await m.read(0x0E0)
    br = int(dut.u_ctrl.ch_base.value) & ((1 << 64) - 1)
    if v != 1 or br != 0x2_0004_0000:
        bad.append("MEM_EN %#x, latched base %#x" % (v, br))
    if dual:
        # the dual weight stream (MVM_DUAL 1): CAPS bit 16, the run-time enable ADAPTER_CTRL.DUAL_DIS reaches the core
        # (dual_en: reset 1, a write of bit 1 clears it, a write without it sets it again), and the read-only
        # ports stay idle without an MVM
        cp, _ = await m.read(0x1000 + T.CSR_CAPS)
        if not cp & T.CAPS_MVM_DUAL:
            bad.append("CAPS %#x lacks MVM_DUAL" % cp)
        if int(dut.dual_en.value) != 1:
            bad.append("dual_en low after reset")
        await m.write(0x0E0, 3)
        if int(dut.dual_en.value) != 0:
            bad.append("dual_en high with ADAPTER_CTRL.DUAL_DIS = 1")
        v, _ = await m.read(0x0E0)
        if v != 3:
            bad.append("ADAPTER_CTRL %#x after a write of 3" % v)
        await m.write(0x0E0, 1)
        if int(dut.dual_en.value) != 1:
            bad.append("dual_en low with ADAPTER_CTRL.DUAL_DIS = 0")
        for _ in range(50):
            await RisingEdge(dut.ap_clk)
            if int(dut.m_axib_arvalid.value):
                bad.append("m_axib_arvalid without an MVM")
                break
        # args chb<c> at 0x100 + 8c: no register behind them
        for off in (0x100, 0x104, 0x108, 0x118):
            await m.write(off, 0xFFFFFFFF)
            v, _ = await m.read(off)
            if v != 0:
                bad.append("%#x reads %#x (no register behind the chb<c> args)" % (off, v))
    for x in bad:
        dut._log.error(x)
    assert not bad, "%d kernel checks failed" % len(bad)


# =================================================================================================
# The read ports of the kernel: main port A and read-only port B (tsr_xrt_kernel with the stub core)
# =================================================================================================
BEAT = 512
ID_W = 4
KMUL = (0x9E3779B97F4A7C15F39CC0605CEDC8341082276BF3A27251F86C6A11D0C18E95 << 256 |
        0xD1B54A32D192ED03A0761D6478BD642F7FEB352D5851F42D4C957F2D1B6A2D3B) | 1
A_ACTRL, A_ASTATUS = 0x0E0, 0x0F0
# per port: the core-side net prefix of the kernel, the AXI port prefix, the kernel's event nets, the stub's variable prefix
PORTS = {"A": dict(core="rd_", axi="m_axi_", ev=("ev_rresp_err", "ev_unexpected"), stub="t_a_"),
         "B": dict(core="rb_", axi="m_axib_", ev=("ev2_rresp_err", "ev2_unexpected"), stub="t_b_")}


class Vec:
    """A packed [NUM_CH] input: set(c, v) drives field c and keeps the others."""

    def __init__(self, handle, n):
        self.h, self.w, self.q = handle, len(handle) // n, 0
        handle.value = 0

    def set(self, c, v):
        m = ((1 << self.w) - 1) << (self.w * c)
        q = (self.q & ~m) | ((v << (self.w * c)) & m)
        if q != self.q:
            self.q = q
            self.h.value = q


class Port:
    """The read path of one port of one channel c: kind "A" is the main port m_axi_*[c] (m_axi_ch<c> of the tops), kind "B"
    the read-only port m_axib_*[c] (m_axi_chb<c>). The bench is the core on one side (requests through the stub core, a
    response sink) and the memory on the other (an AXI slave that answers every AR with its beats in order, with error beats
    and stray beats on request). Every handshake in between is checked."""

    def __init__(self, tb, kind, c):
        k = PORTS[kind]
        self.tb, self.kind, self.c = tb, kind, c
        self.name = "%s%d" % (kind, c)
        self.stub, self.axi = k["stub"], k["axi"]
        self.n = {s: k["core"] + s for s in ("req_valid", "req_ready", "req_addr", "req_len", "req_id", "rsp_valid", "rsp_ready",
                                              "rsp_data", "rsp_id", "rsp_last")}
        self.n.update({s: k["axi"] + s for s in ("arvalid", "arready", "araddr", "arlen", "arsize", "arburst", "arlock",
                                                  "arcache", "arprot", "arqos", "arid", "rvalid", "rready", "rdata", "rresp",
                                                  "rlast", "rid")})
        self.n["ev_err"], self.n["ev_unx"] = k["ev"]
        self.ar_p = self.rv_p = self.rsp_p = 1.0     # probability per cycle of ARREADY, a new R beat, the sink's ready
        self.n_req = self.n_ar = self.n_rsp = self.n_beats = self.n_ev_err = self.n_ev_unx = 0
        self.ar_valid = 0
        self.clear()

    def clear(self):
        """The core and the memory forget everything (a reset)."""
        self.req = deque()           # requests the core still has to offer: (addr, len, id)
        self.cur, self.drop = None, False
        self.exp_ar = deque()        # accepted at the core side, AR not yet seen at the AXI side
        self.bursts = deque()        # AR seen, beats to send: dict(id, beats [(data, resp)], sent)
        self.exp_rsp = deque()       # beats the core side has to see: (data, id, last)
        self.r_hold, self.r_stall = None, False
        self.stray = deque()         # ids of R beats to send with no burst outstanding
        self.err_beats = 0           # the next beats of the next AR(s) get an error response

    def counts_of(self):
        return (self.n_req, self.n_ar, self.n_rsp, self.n_beats, self.n_ev_err, self.n_ev_unx)

    def idle(self):
        return not (self.req or self.cur or self.exp_ar or self.bursts or self.exp_rsp or self.r_hold or self.stray)

    def withdraw(self):
        """The core stops offering what it has not yet issued."""
        self.req.clear()
        self.drop = True

    def make_burst(self, a, ln, i):
        beats = []
        for k in range(ln):
            resp = 0
            if self.err_beats > 0:
                resp = self.tb.r.choice((2, 3))
                self.err_beats -= 1
            beats.append((self.tb.data(self.kind, self.c, a // 64 + k), resp))
            self.exp_rsp.append((beats[-1][0], i, int(k == ln - 1)))
        self.n_beats += ln
        self.bursts.append(dict(id=i, beats=beats, sent=0))

    def sample(self, x):
        """The cycle's handshakes (inputs were driven after the last rising edge)."""
        tb, c, n = self.tb, self.c, self.n

        def bit(s):
            return (x[n[s]] >> c) & 1

        def fl(s, w):
            return (x[n[s]] >> (w * c)) & ((1 << w) - 1)

        self.ar_valid = bit("arvalid")
        if tb.quiet:
            if tb.strict and self.ar_valid:
                tb.err("%s: ARVALID high while ap_rst_n is low or just released" % self.name)
            return
        self.n_ev_err += bit("ev_err")
        self.n_ev_unx += bit("ev_unx")
        # core side request
        rv, rr = bit("req_valid"), bit("req_ready")
        if rv and rr:
            if tb.gate_closed:
                tb.err("%s: the core-side request was accepted with MEM_EN low" % self.name)
            if self.cur is None or (fl("req_addr", tb.aw), fl("req_len", tb.lw), fl("req_id", tb.iw)) != self.cur:
                tb.err("%s: the core-side request bus differs from the request offered" % self.name)
            else:
                self.n_req += 1
                self.exp_ar.append(self.cur)
                self.cur = None
        # AR
        if self.ar_valid and tb.gate_closed:
            tb.err("%s: ARVALID high with MEM_EN low" % self.name)
        if self.ar_valid and not self.exp_ar:
            tb.err("%s: ARVALID high with no request accepted from the core side" % self.name)
        elif self.ar_valid and bit("arready"):
            ea, el, ei = self.exp_ar.popleft()
            a, ln, i = fl("araddr", 64), fl("arlen", 8) + 1, fl("arid", ID_W)
            if (a, ln, i) != (tb.bases[c] + (ea << 6), el, ei):
                tb.err("%s: AR %#x len %d id %d, expected %#x len %d id %d (base %#x)" %
                       (self.name, a, ln, i, tb.bases[c] + (ea << 6), el, ei, tb.bases[c]))
            if (fl("arsize", 3), fl("arburst", 2), fl("arlock", 1), fl("arcache", 4), fl("arprot", 3), fl("arqos", 4)) != \
                    (6, 1, 0, 3, 0, 0):
                tb.err("%s: AR attributes" % self.name)
            self.n_ar += 1
            self.make_burst(a, ln, i)
        # R at the AXI side
        if self.r_hold is not None and bit("rvalid") and bit("rready"):
            if self.r_hold[4]:
                b = self.bursts[0]
                b["sent"] += 1
                if b["sent"] == len(b["beats"]):
                    self.bursts.popleft()
            self.r_hold = None
        # R at the core side
        if bit("rsp_valid") and bit("rsp_ready"):
            got = (fl("rsp_data", BEAT), fl("rsp_id", ID_W), bit("rsp_last"))
            if not self.exp_rsp:
                tb.err("%s: a beat reached the core side with nothing outstanding (id %d)" % (self.name, got[1]))
            else:
                if got != self.exp_rsp.popleft():
                    tb.err("%s: R beat at the core side differs (id %d)" % (self.name, got[1]))
                self.n_rsp += 1

    def pick(self):
        r = self.tb.r
        if self.r_stall or r.random() >= self.rv_p:
            return None
        if self.bursts:
            b = self.bursts[0]
            d, resp = b["beats"][b["sent"]]
            return (d, resp, b["id"], int(b["sent"] == len(b["beats"]) - 1), True)
        if self.stray and not (self.exp_ar or self.exp_rsp):
            return (r.getrandbits(BEAT), 0, self.stray.popleft(), 1, False)
        return None

    def drive(self):
        """The inputs of the next cycle (after a rising edge)."""
        tb, c, r = self.tb, self.c, self.tb.r
        out, s, a = tb.out, self.stub, self.axi
        q = tb.quiet
        if self.drop:
            self.cur, self.drop = None, False
        if not q and self.cur is None and self.req:
            self.cur = self.req.popleft()
        out(s + "req_valid", c, int(self.cur is not None))
        if self.cur is not None:
            out(s + "req_addr", c, self.cur[0])
            out(s + "req_len", c, self.cur[1])
            out(s + "req_id", c, self.cur[2])
        out(s + "rsp_ready", c, int(not q and r.random() < self.rsp_p))
        out(a + "arready", c, int(not q and r.random() < self.ar_p))
        if not q and self.r_hold is None:
            self.r_hold = self.pick()
        h = None if q else self.r_hold
        out(a + "rvalid", c, int(h is not None))
        if h is not None:
            out(a + "rdata", c, h[0])
            out(a + "rresp", c, h[1])
            out(a + "rid", c, h[2])
            out(a + "rlast", c, h[3])


class PortBench:
    """Runs the ports of every channel in the background (run) and holds the checks that span them."""

    def __init__(self, dut, nch, rnd):
        self.d, self.n, self.r = dut, nch, rnd
        self.bad = []
        self.quiet = True            # ap_rst_n low, or released less than 8 cycles ago: nothing is checked
        self.strict = False          # ... except that no AR may be offered
        self.gate_closed = True      # MEM_EN is low (and has been for a few cycles): nothing may leave
        self.bases = [0] * nch       # the bases the bridges hold
        self.aw, self.lw, self.iw = (len(getattr(dut, "rd_req_" + s)) // nch for s in ("addr", "len", "id"))
        self.ports = [Port(self, k, c) for k in "AB" for c in range(nch)]
        self.nets = sorted({v for p in self.ports for v in p.n.values()})
        self.vecs = {}
        for p in self.ports:
            for s in ("arready", "rvalid", "rdata", "rresp", "rlast", "rid"):
                self.vecs.setdefault(p.axi + s, Vec(getattr(dut, p.axi + s), nch))
            for s in ("req_valid", "req_addr", "req_len", "req_id", "rsp_ready"):
                self.vecs.setdefault(p.stub + s, Vec(getattr(dut.u_core, p.stub + s), nch))

    def ports_of(self, kind):
        return [p for p in self.ports if p.kind in kind]

    def out(self, name, c, v):
        self.vecs[name].set(c, v)

    def err(self, msg):
        if msg not in self.bad and len(self.bad) < 60:
            self.bad.append(msg)

    def data(self, kind, c, abs_beat):
        return (((abs_beat + 1) ^ (((ord(kind) << 8) | c) << 40)) * KMUL) & ((1 << BEAT) - 1)

    def rand_req(self):
        ln = self.r.choice([1, 1, 2, 3, 8, 16, 31, 63, 64, self.r.randint(1, 64)])
        return ((self.r.randrange(1 << 14) << 6) + self.r.randint(0, 64 - ln), ln, self.r.randrange(1 << ID_W))

    async def run(self):
        d = self.d
        while True:
            await FallingEdge(d.ap_clk)
            x = {nm: int(getattr(d, nm).value) for nm in self.nets}
            for p in self.ports:
                p.sample(x)
            await RisingEdge(d.ap_clk)
            for p in self.ports:
                p.drive()

    async def idle(self, n):
        for _ in range(n):
            await RisingEdge(self.d.ap_clk)

    async def until(self, cond, what, limit=4000):
        for _ in range(limit):
            await RisingEdge(self.d.ap_clk)
            if cond():
                return
        self.err("timeout: %s" % what)

    async def drain(self, ports, limit=60000):
        """Waits until the ports have nothing left to do (every request answered) for 6 cycles."""
        ok = 0
        for _ in range(limit):
            await RisingEdge(self.d.ap_clk)
            ok = ok + 1 if all(p.idle() for p in ports) else 0
            if ok >= 6:
                return
        self.err("drain: %s not idle after %d cycles" % ([p.name for p in ports if not p.idle()], limit))

    async def reset(self, n):
        """ap_rst_n low for n cycles; the core and the memory forget what was in flight."""
        d = self.d
        await FallingEdge(d.ap_clk)
        self.quiet, self.gate_closed = True, True
        for p in self.ports:
            p.clear()
        d.ap_rst_n.value = 0
        d.ap_rst_n_2.value = 0
        for i in range(n):
            await FallingEdge(d.ap_clk)
            self.strict = i >= 3                 # the bridges' AXI outputs are low from the third cycle on
        d.ap_rst_n.value = 1
        d.ap_rst_n_2.value = 1
        await self.idle(8)
        self.quiet = self.strict = False

    async def bind(self, m, bases):
        """Writes the channel bases and sets MEM_EN: the bridges latch the bases and the gate opens."""
        for c, b in enumerate(bases[:self.n]):
            await m.write(0x40 + 8 * c, b & M32)
            await m.write(0x44 + 8 * c, b >> 32)
        self.bases = list(bases[:self.n])
        self.gate_closed = False
        await m.write(A_ACTRL, 1)
        v, _ = await m.read(A_ACTRL)
        if v != 1:
            self.err("ADAPTER_CTRL %#x after the MEM_EN write" % v)
        await self.idle(4)

    async def unbind(self, m):
        await m.write(A_ACTRL, 0)
        await self.idle(4)
        self.gate_closed = True

    async def status(self, m, what, bits, cnt):
        """Reads ADAPTER_STATUS (after the events of the last cycles have reached it) and compares bits [5:0] and the count."""
        await self.idle(12)
        v, _ = await m.read(A_ASTATUS)
        if (v & 0x3F, v >> 16) != (bits, cnt):
            self.err("%s: ADAPTER_STATUS %#x, expected bits %#x and error count %d" % (what, v, bits, cnt))

    async def clear_status(self, m):
        await m.write(A_ASTATUS, M32)
        await self.idle(10)

    def counts(self):
        return {p.name: p.counts_of() for p in self.ports}


async def portb_hold(tb, m, kind):
    """MEM_EN low: a request of one channel waits at the gate (nothing leaves, REQ_HELD), then requests of every channel wait
    together and go through when MEM_EN is set."""
    ps = tb.ports_of(kind)
    for p in ps:
        await tb.clear_status(m)
        await tb.status(m, "%s, no request" % p.name, 0, 0)
        n0 = tb.counts()
        p.req.append(tb.rand_req())
        await tb.idle(40)
        await tb.status(m, "%s, request waiting" % p.name, T.ADAPTER_STATUS_REQ_HELD, 0)
        if tb.counts() != n0:
            tb.err("%s: a request passed the closed gate" % p.name)
        p.withdraw()
        await tb.idle(10)
    await tb.clear_status(m)
    n0 = tb.counts()
    for p in ps:
        p.req.extend([tb.rand_req(), tb.rand_req()])
    await tb.idle(40)
    await tb.status(m, "%s, every channel waiting" % kind, T.ADAPTER_STATUS_REQ_HELD, 0)
    if tb.counts() != n0:
        tb.err("%s: requests passed the closed gate" % kind)
    await tb.bind(m, BASES)
    await tb.drain(ps)
    n1 = tb.counts()
    for p in ps:
        if n1[p.name][:2] != (n0[p.name][0] + 2, n0[p.name][1] + 2):
            tb.err("%s: the waiting requests did not go through after MEM_EN" % p.name)
    await tb.status(m, "%s, after MEM_EN" % kind, T.ADAPTER_STATUS_REQ_HELD, 0)


async def portb_traffic(tb, m, kind, nreq):
    """Random reads on every channel of the ports in `kind`, the other ports idle: the AR and R paths against the AXI slave."""
    ps = tb.ports_of(kind)
    await tb.clear_status(m)
    n0 = tb.counts()
    for p in ps:
        p.ar_p, p.rv_p, p.rsp_p = (tb.r.choice((0.3, 0.6, 1.0)) for _ in range(3))
        p.req.extend(tb.rand_req() for _ in range(nreq))
    await tb.drain(ps)
    await tb.status(m, "traffic on %s" % kind, 0, 0)
    n1 = tb.counts()
    for p in tb.ports:
        d = tuple(b - a for a, b in zip(n0[p.name], n1[p.name]))
        want = (nreq, nreq, d[3], d[3], 0, 0) if p in ps else (0,) * 6
        if d != want or (p in ps and d[3] < nreq):
            tb.err("%s: counts of the traffic run (requests, AR, beats expected, beats seen, error events, stray events) %s, "
                   "expected %s" % (p.name, d, want))


async def portb_errors(tb, m, kind):
    """Per channel: two error beats of one burst, then an R beat nothing waits for; both reach ADAPTER_STATUS."""
    for p in tb.ports_of(kind):
        await tb.clear_status(m)
        n0 = tb.counts()
        p.err_beats = 2
        p.req.append((tb.r.randrange(1 << 14) << 6, 5, tb.r.randrange(1 << ID_W)))
        await tb.drain([p])
        await tb.status(m, "%s, two RRESP errors" % p.name, T.ADAPTER_STATUS_RRESP_ERR, 2)
        if not int(tb.d.plat_err.value):
            tb.err("%s: plat_err low after RRESP errors" % p.name)
        n1 = tb.counts()
        for q in tb.ports:
            if n1[q.name][4:] != (n0[q.name][4] + (2 if q is p else 0), n0[q.name][5]):
                tb.err("%s: error events of %s: %s -> %s" % (p.name, q.name, n0[q.name][4:], n1[q.name][4:]))
        await tb.clear_status(m)
        n0 = tb.counts()
        p.stray.append(tb.r.randrange(1 << ID_W))
        await tb.drain([p])
        await tb.status(m, "%s, an unexpected R beat" % p.name, T.ADAPTER_STATUS_ORPHAN_RESP, 0)
        if not int(tb.d.plat_err.value):
            tb.err("%s: plat_err low after an unexpected R beat" % p.name)
        n1 = tb.counts()
        for q in tb.ports:
            if n1[q.name][4:] != (n0[q.name][4], n0[q.name][5] + (1 if q is p else 0)):
                tb.err("%s: unexpected-beat events of %s: %s -> %s" % (p.name, q.name, n0[q.name][4:], n1[q.name][4:]))


async def portb_reset(tb, m, kind):
    """ap_rst_n while a bridge holds an AR at the AXI port, requests in its slices and the count of an outstanding burst:
    the AR is gone, the burst is forgotten (its late R beat is dropped and reported), the base and MEM_EN are cleared, and the
    port works again after the bases are written."""
    ps = tb.ports_of(kind)
    n0 = tb.counts()
    for p in ps:
        p.ar_p = p.rv_p = p.rsp_p = 1.0
        p.r_stall = True
        p.req.append((0x40, 4, 3))                  # taken, its R beats withheld
    await tb.until(lambda: all(p.n_ar == n0[p.name][1] + 1 for p in ps), "%s: first AR" % kind)
    for p in ps:
        p.ar_p = 0.0
        p.req.extend([(0x80, 2, 5), (0xC0, 2, 6), (0x100, 1, 7), (0x140, 1, 8)])
    await tb.idle(30)
    for p in ps:
        if not p.ar_valid:
            tb.err("%s: no AR waiting at the AXI port before the reset" % p.name)
    await tb.reset(6)
    n1 = tb.counts()
    for p in ps:
        p.stray.append(3)                           # the late beat of the burst of the earlier epoch
    await tb.drain(ps)
    await tb.status(m, "%s, after the reset" % kind, T.ADAPTER_STATUS_ORPHAN_RESP, 0)
    n2 = tb.counts()
    for p in ps:
        if n2[p.name][:4] != n1[p.name][:4] or n2[p.name][5] != n1[p.name][5] + 1:
            tb.err("%s: after the reset %s -> %s (the late beat must give one unexpected event and nothing else)" %
                   (p.name, n1[p.name], n2[p.name]))
    for p in ps:
        p.ar_p = 0.7
    await tb.bind(m, BASES2)
    for p in ps:
        p.req.extend(tb.rand_req() for _ in range(3))
    await tb.drain(ps)
    n3 = tb.counts()
    for p in ps:
        if n3[p.name][:2] != (n2[p.name][0] + 3, n2[p.name][1] + 3):
            tb.err("%s: requests after the reset: %s -> %s" % (p.name, n2[p.name], n3[p.name]))


# 4 KiB aligned bases that differ in both words, per channel; the second set is for after a reset
BASES = [((c + 1) << 33) + 0x1000 * (7 * c + 5) for c in range(16)]
BASES2 = [b + 0x1000 * (0x100 + c) for c, b in enumerate(BASES)]


@cocotb.test()
async def adapter_portb(dut):
    """The read ports of tsr_xrt_kernel (MVM_DUAL 1, stub core): see the module header, test_adapter_portb."""
    nch = len(dut.m_axib_arvalid)
    rnd = random.Random(int(os.environ.get("TSR_SEED", "1")))
    cocotb.start_soon(Clock(dut.ap_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.ap_clk_2, 16, unit="ns").start())
    pfx = "s_axi_control_"
    for n in ("awvalid", "wvalid", "bready", "arvalid", "rready", "awaddr", "araddr", "wdata", "wstrb"):
        getattr(dut, pfx + n).value = 0
    for n in ("m_axi_awready", "m_axi_wready", "m_axi_bvalid", "m_axi_bresp", "m_axi_bid"):
        getattr(dut, n).value = 0
    dut.ap_rst_n.value = 0
    dut.ap_rst_n_2.value = 0
    tb = PortBench(dut, nch, rnd)
    cocotb.start_soon(tb.run())
    m = Axil(dut, dut.ap_clk, pfx, random.Random(3))
    await tb.reset(20)
    await portb_hold(tb, m, "B")                    # MEM_EN is low after the reset
    await portb_traffic(tb, m, "B", 30)
    await portb_traffic(tb, m, "A", 30)
    await portb_traffic(tb, m, "AB", 30)
    await portb_errors(tb, m, "B")
    await portb_errors(tb, m, "A")
    await tb.unbind(m)
    await portb_hold(tb, m, "A")
    await portb_reset(tb, m, "B")
    await portb_reset(tb, m, "A")
    for k in "AB":
        c = [sum(p.counts_of()[i] for p in tb.ports_of(k)) for i in range(6)]
        dut._log.info("port %s: %d requests, %d AR, %d beats to the core, %d beats expected, %d RRESP error events, "
                      "%d unexpected events", k, *c)
    for x in tb.bad:
        dut._log.error(x)
    assert not tb.bad, "%d port B checks failed" % len(tb.bad)


# =================================================================================================
# pytest launchers
# =================================================================================================
def _src_root():
    """Sources: the tree (default) or $TSR_PLATFORM_SRC_ROOT (a snapshot with the same layout)."""
    return Path(os.environ.get("TSR_PLATFORM_SRC_ROOT", str(REPO)))


def _regs(ctrl_src=None, name="adapter_regs"):
    seed = int(os.environ.get("TSR_SEED", "1"))
    ops = script(seed, NUM_CH)
    exp = mock_run(ops, NUM_CH)
    root = _src_root()
    srcs = [root / "rtl/core/tsr_tables_pkg.sv", root / "rtl/core/tsr_pkg.sv",
            Path(ctrl_src) if ctrl_src else root / "rtl/platform/alveo_xrt/tsr_xrt_ctrl.sv"]
    run("tsr_xrt_ctrl", __name__, parameters=dict(NUM_CH=NUM_CH), sources=srcs, name=name,
        extra_env={"TSR_ADAPTER_CASE": json.dumps(dict(ops=ops, exp=exp, num_ch=NUM_CH))}, testcase=["adapter_regs"])


def test_adapter_regs():
    """tsr_xrt_ctrl against the mock adapter model (register map, MEM_EN, ADAPTER_STATUS, window)."""
    _regs()


# (id, original text of tsr_xrt_ctrl.sv, mutated text, what it breaks: EX §9.2-§9.3)
ADAPTER_MUTANTS = [
    ("no_base_locked", "base_wr && mem_en_q, held_q", "1'b0, held_q", "BASE_LOCKED not reported"),
    ("base_write_while_en", "      if (reg_wr && !mem_en_q) begin", "      if (reg_wr) begin",
     "a base write while MEM_EN = 1 not ignored"),
    ("no_align_check", "      if (base_reg_q[c][11:0] != 12'd0) all_aligned = 1'b0;",
     "      if (base_reg_q[c][3:0] != 4'd0) all_aligned = 1'b0;", "the 4 KiB alignment check at MEM_EN"),
    ("w1c_low_bits", "ast_d = ast_d & ~ax_wdata_q[5:0];", "ast_d = ast_d & ~{3'd0, ax_wdata_q[2:0]};",
     "W1C of bits 3..5"),
    ("plat_err_no_count", "plat_err_q <= (ast_d[2:0] != 3'd0) || (acnt_d != 16'd0);",
     "plat_err_q <= (ast_d[2:0] != 3'd0);", "plat_err_i ignoring the error count"),
    ("no_latch", "          base_lat_q <= base_reg_q;", "", "the bases not latched at MEM_EN"),
    ("dual_polarity", "        dual_en_q <= ~ax_wdata_q[1];", "        dual_en_q <= ax_wdata_q[1];",
     "DUAL_DIS: the run-time enable of the dual weight stream has the wrong polarity"),
    ("dual_readback", "A_ACTRL:    reg_rdata = {30'd0, ~dual_en_q, mem_en_q};", "A_ACTRL:    reg_rdata = {30'd0, 1'b0, mem_en_q};",
     "ADAPTER_CTRL.DUAL_DIS reads back 0"),
    ("ev2_no_count", "      anyr_q <= |evr_q | |evr2_q;", "      anyr_q <= |evr_q;",
     "R errors of the read-only bridges not in ADAPTER_STATUS bit 0"),
]


@pytest.mark.slow
@pytest.mark.parametrize("m", ADAPTER_MUTANTS, ids=[m[0] for m in ADAPTER_MUTANTS])
def test_adapter_mutants(m):
    """Adapter mutation spot checks: test_adapter_regs must fail on each. Opt-in: TSR_ADAPTER_MUTANTS=1."""
    if os.environ.get("TSR_ADAPTER_MUTANTS") != "1" and "mutants" not in " ".join(sys.argv):
        pytest.skip("opt-in: TSR_ADAPTER_MUTANTS=1")
    mid, old, new, what = m
    src = (_src_root() / "rtl/platform/alveo_xrt/tsr_xrt_ctrl.sv").read_text()
    assert src.count(old) == 1, "mutant %s: original text not found once" % mid
    d = build_root() / "adapter_mutants" / mid
    d.mkdir(parents=True, exist_ok=True)
    f = d / "tsr_xrt_ctrl.sv"
    f.write_text(src.replace(old, new))
    killed = False
    try:
        _regs(ctrl_src=f, name="adapter_mut_" + mid)
    except BaseException:                    # a failing cocotb run raises SystemExit under pytest
        killed = True
    assert killed, "adapter mutant %s (%s) survived" % (mid, what)


def test_adapter_kernel_dual():
    """tsr_xrt_kernel with MVM_DUAL 1 (2 channels, T_MAX 2, the MVM unit in): the read-only ports m_axib_*, CAPS bit 16,
    ADAPTER_CTRL.DUAL_DIS to the core's cfg_dual_en, the chb<c> args without registers."""
    root = _src_root()
    from common.runner import parse_filelist
    srcs = parse_filelist(root / "rtl/filelist.f")[0] + parse_filelist(root / "rtl/platform/alveo_xrt/filelist.f")[0]
    params = dict(HAS_MVM=1, MVM_DUAL=1, NUM_CH=2, T_MAX=2, SPM_BYTES=65536, CH_BYTES="64'd1073741824", K_MAX=4096,
                  ADDR_BEAT_W=22, XSLR_REG=1)
    run("tsr_xrt_kernel", __name__, parameters=params, sources=srcs, name="adapter_kernel_dual",
        extra_env={"TSR_KCH": "2", "TSR_KDUAL": "1"}, testcase=["adapter_kernel"])


def test_adapter_portb():
    """tsr_xrt_kernel with MVM_DUAL 1 and the stub core of tb/platform/sv (3 channels, XSLR_REG 1): traffic on the read-only
    port B and on the main port, the MEM_EN gate, error and unexpected-beat events, reset."""
    root = _src_root()
    plat = root / "rtl/platform/alveo_xrt"
    srcs = [root / "rtl/core/tsr_tables_pkg.sv", root / "rtl/core/tsr_pkg.sv", root / "rtl/core/tsr_mem_arb.sv"]
    srcs += [plat / f for f in ("tsr_xrt_reset.sv", "tsr_xrt_slices.sv", "tsr_xrt_axi_bridge.sv", "tsr_xrt_ctrl.sv",
                                "tsr_xrt_kernel.sv")]
    srcs.append(HERE / "sv" / "stub_core_portb.sv")
    params = dict(MVM_DUAL=1, NUM_CH=3, ADDR_BEAT_W=22, XSLR_REG=1)
    run("tsr_xrt_kernel", __name__, parameters=params, sources=srcs, name="adapter_portb", testcase=["adapter_portb"])


@pytest.mark.parametrize("xslr", [0, 1])
def test_adapter_kernel(xslr):
    """tsr_xrt_kernel (HAS_MVM 0, 1 channel): the second clock, MEM_EN, the window; XSLR_REG 1 as in the K4 top."""
    root = _src_root()
    from common.runner import parse_filelist
    srcs = parse_filelist(root / "rtl/filelist.f")[0] + parse_filelist(root / "rtl/platform/alveo_xrt/filelist.f")[0]
    params = dict(HAS_MVM=0, NUM_CH=1, SPM_BYTES=65536, CH_BYTES="64'd1073741824", K_MAX=4096, ADDR_BEAT_W=22,
                  XSLR_REG=xslr)
    run("tsr_xrt_kernel", __name__, parameters=params, sources=srcs, name="adapter_kernel_x%d" % xslr,
        extra_env={"TSR_KCH": "1"}, testcase=["adapter_kernel"])

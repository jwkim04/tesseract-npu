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
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
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

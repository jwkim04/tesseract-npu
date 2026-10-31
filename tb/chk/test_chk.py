"""tb/chk — tsr_chk against the simulator's checker classes (docs/design/04-control-core.md "CC" §8.1-§8.2,
§15.2; EX §7.1, §8.1; spec §21 range rule).

For every op: the unit, the error code (ILLEGAL_OP, BAD_FMT, K_RANGE, T_RANGE, OPERAND, SPM_RANGE,
DRAM_RANGE in their order; BAD_DEP is the dispatcher's), ERR_INFO, the four tagged extents (kind, direction,
channel, [lo, hi)), the LD / ST n = 0 flag, the SYNC kind, and for a passing op every field of its unit
command (docs/rtl-m2.md §3.2-§3.4) are compared with tsr_sim_accept_op + the handler's check
(sw/sim/tsr_sim.c chk_*; tb/issue/issue_sim.py) and the command fields the simulator's units use. The
VU / AT opcodes go through the external-class port (answered with the simulator's VU / AT checks).
Stimulus: every opcode with random words, and per class edge-heavy operands (limits +- small deltas,
0, 2^31, 2^32 - 1, T and stride products that overflow 32 bits, reserved bits one at a time). The
checker's latency (8 cycles, + 33 for an MVM with odd NUM_CH) is measured. chk_directed adds the ops
of directed_ops(): half-block K values (odd multiples of 128 for Q4_K / A256, of 16 for Q8_0 / A32),
weight extents of 2^14 beats and more at the DRAM limit, destination / source extents of 2^32 lines and
more; the same list is the netlist regression of tb/chk/viv/net_chk.sh (the post-synthesis netlist
answers like the simulator; found by the W4-A review: a variable with two drivers in synthesis).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import isa                                    # noqa: E402
from common.runner import REPO, run                       # noqa: E402

import issue_sim as S                                     # noqa: E402
import issue_tb as E                                      # noqa: E402

def src():
    d = os.environ.get("TSR_MUT_DIR")                      # a mutated copy (tb/issue/test_issue_mutants.py)
    chk = Path(d) / "tsr_chk.sv" if d and (Path(d) / "tsr_chk.sv").exists() else REPO / "rtl/core/tsr_chk.sv"
    return [REPO / "rtl/core/tsr_tables_pkg.sv", REPO / "rtl/core/tsr_pkg.sv", chk]
M32 = 0xFFFFFFFF


def cfg_from_env():
    import json
    c = json.loads(os.environ.get("TSR_CHK_CFG", "{}"))
    cfg = S.config(num_ch=c.get("num_ch", 4), t_max=c.get("t_max", 8), k_max=c.get("k_max", 32768),
                   spm_bytes=c.get("spm_bytes", 4 << 20), ch_bytes=c.get("ch_bytes", 16 << 30),
                   addr_beat_w=c.get("addr_beat_w", 28), caps=c.get("caps", 0))
    return cfg, tuple(c.get("qd", S.U200_QD)), c.get("n_ops", 4000)


class OpGen:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.r = random.Random(seed)
        self.C = cfg["num_ch"]
        self.L = cfg["spm_bytes"] // 64
        self.DL = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])

    def near(self, lim, ext):
        """A start address around lim - ext (in range by 0..2, out by 1..2) or anywhere."""
        r = self.r
        x = r.random()
        if x < 0.35:
            return max(0, lim - ext + r.randint(-2, 2)) & M32
        if x < 0.45:
            return r.choice([0, lim, lim - 1, 1 << 31, M32]) & M32
        return r.randint(0, max(0, min(lim, 1 << 32) - 1))

    def val32(self, typical):
        r = self.r
        return r.choice([typical, typical, typical, 0, 1, 1 << 31, M32, r.getrandbits(32), typical + 1,
                         max(0, typical - 1)]) & M32

    def ldst(self):
        r = self.r
        n = r.choice([0, 1, r.randint(1, 64), r.randint(1, 4096), self.L, M32])
        ch = r.choice([r.randrange(self.C)] * 4 + [r.randint(0, 15)])
        w = (isa.enc_st if r.random() < 0.5 else isa.enc_ld)(ch, self.near(self.L, n), self.near(self.DL, n), n & M32)
        return w

    def qact(self):
        r = self.r
        kind = r.choice([0, 1, 0, 1, r.randint(2, 15)])
        bs = 256 if kind == 1 else 32
        K = r.choice([bs * r.randint(1, max(1, self.cfg["k_max"] // bs)), self.cfg["k_max"], self.cfg["k_max"] + bs,
                      0, bs * r.randint(1, 8) + 16, 1 << 31, M32 & ~31, 65536 + bs,
                      (bs // 2) * (2 * r.randint(0, max(0, self.cfg["k_max"] // bs - 1)) + 1),   # half a block
                      32 * r.randint(1, max(1, self.cfg["k_max"] // 32))])    # a multiple of 32, not of the block
        Tt = r.choice([r.randint(1, self.cfg["t_max"]), self.cfg["t_max"], 0, self.cfg["t_max"] + 1, 0x80000001, M32])
        ql = isa.qa_lines(kind & 1, K & 0xFFFF) if (K & 0xFFFF) else 1
        xl = (K >> 4) & M32
        stride = r.choice([0, 0, xl, xl + r.randint(1, 64), 1 << 31, M32, r.getrandbits(32)])
        tl = Tt if Tt <= 8 else 1
        e_dst = tl * ql
        e_src = (tl - 1) * (stride or xl) + xl
        return isa.enc_qact(kind, self.near(self.L, e_dst), self.near(self.L, e_src), K & M32, Tt & M32, stride & M32)

    def mvm(self):
        r = self.r
        fmt = r.choice([0, 1, 0, 1, r.randint(2, 15)])
        bs = 256 if fmt == 1 else 32
        K = r.choice([bs * r.randint(1, max(1, self.cfg["k_max"] // bs)), self.cfg["k_max"], 0, bs + 16,
                      (self.cfg["k_max"] + bs) & 0xFFFF, 0xFFFF & ~31,
                      (bs // 2) * (2 * r.randint(0, max(0, self.cfg["k_max"] // bs - 1)) + 1),   # half a block
                      32 * r.randint(1, max(1, self.cfg["k_max"] // 32))])    # a multiple of 32, not of the block
        Tt = r.choice([r.randint(1, self.cfg["t_max"]), 0, min(15, self.cfg["t_max"] + 1), 15])
        N = r.choice([0, 1, self.C - 1, self.C, self.C + 1, r.randint(1, 4096), 16 * self.L, 16 * self.L + 1,
                      r.getrandbits(32), M32, r.choice([4864, 8960, 9728, 151936])])   # weights extent >= 2^14 beats
        yl = (N + 15) >> 4
        ds = r.choice([0, 0, yl, yl + 3, M32, r.getrandbits(32)])
        f1 = fmt & 1
        ql = isa.qa_lines(f1, K) if K else 1
        tl = max(1, Tt)
        e_dst = (tl - 1) * (ds or yl) + yl
        e_x = tl * ql
        R = (N + self.C - 1) // self.C
        e_w = R * (isa.row_beats(f1, K) if K else 0)
        w = isa.enc_mvm(fmt, self.near(self.L, e_dst), self.near(self.L, e_x), self.near(self.DL, e_w), N & M32,
                        K, Tt, ds & M32, acc=r.randint(0, 1))
        return w

    def sync(self):
        return isa.enc_fence() if self.r.random() < 0.5 else isa.enc_nop()

    def any_op(self):
        r = self.r
        op = r.randrange(256)
        return [op | (r.getrandbits(24) << 8 if r.random() < 0.5 else 0)] + \
               [r.choice([0, r.getrandbits(32), r.getrandbits(8)]) for _ in range(7)]

    def reserved(self, w):
        """One reserved bit or word set (EX §7.1: ILLEGAL_OP), or a sub code changed."""
        r = self.r
        w = list(w)
        j = r.randrange(8)
        w[j] ^= 1 << r.randrange(32)
        return w

    def op(self):
        r = self.r
        k = r.choices(["ldst", "qact", "mvm", "sync", "any"], [4, 4, 4, 1, 1.5])[0]
        w = getattr(self, k if k != "any" else "any_op")()
        if r.random() < 0.12:
            w = self.reserved(w)
        return w


def directed_ops(cfg):
    """The directed ops of the W4-A review (answers computed by the simulator): half-block K values, weight
    extents of 2^14 beats and more at the DRAM limit, destination / source extents of 2^32 lines and more,
    LD / ST counts near 2^32. Also the netlist regression's stimulus (tb/chk/viv/net_chk.sh)."""
    L = cfg["spm_bytes"] // 64
    DL = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
    C = cfg["num_ch"]
    v = []
    for K in (128, 384, 640, 32640, 256, 32768):                     # Q4_K / A256: odd multiples of 128, 256 itself
        v.append(isa.enc_mvm(1, 0, 4096, 0, 64, K, 1, 0))
        v.append(isa.enc_qact(1, 0, 4096, K, 1, 0))
    for K in (16, 48, 32, 1008):                                     # Q8_0 / A32: odd multiples of 16, 32 itself
        v.append(isa.enc_mvm(0, 0, 4096, 0, 64, K, 1, 0))
        v.append(isa.enc_qact(0, 0, 4096, K, 1, 0))
    # MVM weights extent >= 2^14 beats (0.5B FFN shape N 4864, K 896, Q8_0) at the DRAM limit
    for (N, K, fmt) in ((4864, 896, 0), (151936, 896, 0), (4864, 4864, 1)):
        R = (N + C - 1) // C
        ew = R * isa.row_beats(fmt, K)
        for woff in (DL - ew, DL - ew + 1, 0):
            if 0 <= woff <= M32:
                v.append(isa.enc_mvm(fmt, 0, L - 64, woff, N, K, 1, 0) if N <= 16 * (L - 64) else
                         isa.enc_mvm(fmt, 0, 0, woff, 1024, K, 1, 0))
    # MVM dst extent >= 2^32 lines: T 2, dst stride 2^32 - 1 (SPM_RANGE), and ACC
    v.append(isa.enc_mvm(0, 0, 0, 0, 16, 32, 2, M32))
    v.append(isa.enc_mvm(0, 0, 0, 0, 16, 32, 2, M32, acc=1))
    v.append(isa.enc_mvm(0, 1, 0, 0, 16, 32, 2, (1 << 31)))
    # QACT src extent >= 2^32: T 8, stride 2^31
    v.append(isa.enc_qact(0, 0, 0, 64, 8, 1 << 31))
    # LD / ST n near 2^32
    v.append(isa.enc_ld(0, 0, 0, M32))
    v.append(isa.enc_st(0, 1, 1, L - 1))
    return [[x & M32 for x in w] for w in v]


class DirectedGen(OpGen):
    """Replays directed_ops(cfg) in order (check_ops(..., gen=DirectedGen) with n_ops = its length)."""

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.list, self.k = directed_ops(cfg), 0

    def op(self):
        w = self.list[self.k % len(self.list)]
        self.k += 1
        return w


async def check_ops(dut, cfg, qd, n_ops, seed, gen=None):
    sim = S.Sim(cfg, qd)
    g = (gen or OpGen)(cfg, seed)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.req_valid.value = 0
    dut.req_insn.value = 0
    dut.xc_rsp_valid.value = 0
    dut.xc_rsp_code.value = 0
    dut.xc_rsp_info.value = 0
    dut.xc_rsp_ext.value = 0
    dut.xc_rsp_cmd.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    bad = []
    hist = {}
    lat = {}
    for i in range(n_ops):
        w = g.op()
        w = [x & M32 for x in w]
        unit, code, info, exts = sim.check(w)
        hist[code] = hist.get(code, 0) + 1
        await RisingEdge(dut.clk)            # the previous rsp_valid cycle (busy) is over
        dut.req_insn.value = isa.insn_int(w)
        dut.req_valid.value = 1
        await RisingEdge(dut.clk)
        dut.req_valid.value = 0
        n = 0
        while True:
            await FallingEdge(dut.clk)
            n += 1
            if int(dut.xc_req_valid.value):
                await RisingEdge(dut.clk)
                dut.xc_rsp_valid.value = 1
                dut.xc_rsp_code.value = code
                dut.xc_rsp_info.value = info
                dut.xc_rsp_ext.value = E.pack_ext(E.sim_ext_to_rtl(exts))
                dut.xc_rsp_cmd.value = isa.insn_int(w)
                await RisingEdge(dut.clk)
                dut.xc_rsp_valid.value = 0
                continue
            if int(dut.rsp_valid.value):
                break
            if n > 200:
                raise TimeoutError("no rsp_valid for %s" % [hex(x) for x in w])
        op = w[0] & 0xFF
        lat.setdefault(op, set()).add(n - 1)     # edges from the accepting edge to rsp_valid
        got = dict(unit=int(dut.rsp_unit.value), code=int(dut.rsp_code.value), info=int(dut.rsp_info.value),
                   ext=E.unpack_ext(int(dut.rsp_ext.value)), null=int(dut.rsp_null.value), kind=int(dut.rsp_kind.value),
                   xc=int(dut.rsp_xc.value))
        exp = dict(unit=unit, code=code, info=info, ext=E.sim_ext_to_rtl(exts)[:4],
                   null=int(op in (isa.OP_LD, isa.OP_ST) and w[3] == 0 and code == 0 and unit == 0),
                   kind=(2 if op == isa.OP_FENCE else 1) if (unit == 5 and op in (isa.OP_NOP, isa.OP_FENCE)) else 0)
        is_core = op in (isa.OP_NOP, isa.OP_FENCE, isa.OP_LD, isa.OP_ST, isa.OP_QACT, isa.OP_MVM)
        exp_xc = int(unit != 5 and not is_core)
        if code == 0 and unit != 5 and not exp_xc:
            k, v = E.expected_cmd(unit, w, cfg["num_ch"])
            gotv = int({"dma": dut.rsp_dma, "qact": dut.rsp_qact, "mvm": dut.rsp_mvm}[k].value)
            if gotv != v:
                st = {"dma": isa.DMA_CMD_T, "qact": isa.QACT_CMD_T, "mvm": isa.MVM_CMD_T}[k]
                bad.append("op %s: %s command %s, expected %s" % ([hex(x) for x in w], k, st.unpack(gotv), st.unpack(v)))
        if exp_xc and code == 0 and int(dut.rsp_xcmd.value) != isa.insn_int(w):
            bad.append("op %s: external payload not passed through" % [hex(x) for x in w])
        if got["xc"] != exp_xc:
            bad.append("op %s: xc %d, expected %d" % ([hex(x) for x in w], got["xc"], exp_xc))
        del got["xc"]
        if got != exp:
            bad.append("op %s: RTL %s, simulator %s" % ([hex(x) for x in w], got, exp))
        if len(bad) > 20:
            break
    sim.close()
    dut._log.info("%d ops, codes %s, latency per opcode %s", n_ops, dict(sorted(hist.items())),
                  {hex(k): sorted(v) for k, v in sorted(lat.items()) if k in (0, 0x10, 0x11, 0x18, 0x20, 0x7F)})
    return bad, lat


@cocotb.test()
async def chk_directed(dut):
    cfg, qd, _ = cfg_from_env()
    bad, _ = await check_ops(dut, cfg, qd, len(directed_ops(cfg)), 1, gen=DirectedGen)
    for b in bad[:20]:
        dut._log.error(b)
    assert not bad, "%d mismatches" % len(bad)


@cocotb.test()
async def chk_random(dut):
    cfg, qd, n = cfg_from_env()
    bad, lat = await check_ops(dut, cfg, qd, n, int(os.environ.get("TSR_SEED", "1")))
    for b in bad[:20]:
        dut._log.error(b)
    assert not bad, "%d mismatches" % len(bad)
    pow2 = (cfg["num_ch"] & (cfg["num_ch"] - 1)) == 0
    for op in (0x10, 0x11, 0x18, 0x00, 0x7F) + ((0x20,) if pow2 and qd[2] else ()):
        if op in lat:
            assert lat[op] == {8}, "opcode %#x latency %s, CC §8.2 says 8" % (op, lat[op])


def _run(name, cfg, qd=S.U200_QD, n_ops=4000):
    import json
    units = sum(1 << u for u in range(6) if qd[u])
    c = dict(num_ch=cfg["num_ch"], t_max=cfg["t_max"], k_max=cfg["k_max"], spm_bytes=cfg["spm_bytes"],
             ch_bytes=cfg["ch_bytes"], addr_beat_w=cfg["addr_beat_w"], caps=cfg["caps"], qd=list(qd), n_ops=n_ops)
    run("tsr_chk", __name__, sources=src(), name="chk_" + name, extra_env={"TSR_CHK_CFG": json.dumps(c)},
        parameters=dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                        CH_BYTES="64'd%d" % cfg["ch_bytes"], ADDR_BEAT_W=cfg["addr_beat_w"],
                        CAPS="32'd%d" % cfg["caps"], UNITS="6'd%d" % units))


def test_chk_u200():
    """U200: 4 channels, 4 MiB SPM, 16 GiB channels, T_MAX 8, K_MAX 32768."""
    _run("u200", S.config(num_ch=4, t_max=8), n_ops=6000)


def test_chk_odd():
    """3 channels (serial divider for R), 256 KiB SPM, 1 GiB channels with ADDR_BEAT_W 22 (limit 2^22)."""
    _run("odd", S.config(num_ch=3, t_max=3, k_max=4096, spm_bytes=256 << 10, ch_bytes=1 << 30, addr_beat_w=22))


def test_chk_wide():
    """16 channels, K_MAX 65280, ADDR_BEAT_W 36 (limit CH_BYTES / 64), T_MAX 8; and 1 channel, T_MAX 1,
    64 KiB SPM, a 4 KiB channel (DRAM limit 64 beats), no MVM unit."""
    _run("wide", S.config(num_ch=16, t_max=8, k_max=65280, addr_beat_w=36))
    _run("tiny", S.config(num_ch=1, t_max=1, k_max=256, spm_bytes=64 << 10, ch_bytes=4096, addr_beat_w=40),
         qd=(8, 4, 0, 16, 0, 4))


def test_chk_xc():
    """CAPS VU + ATT: VU / AT opcodes through the external-class port, 7 channels."""
    _run("xc", S.config(num_ch=7, t_max=8, caps=33), n_ops=3000)


def test_chk_netlist(request):
    """Opt-in (needs Vivado, about 3 minutes; TSR_CHK_VIVADO=1 or -k netlist): the Vivado-synthesised netlist of
    tsr_chk answers the directed ops like the simulator, and its synthesis has no multi-driven net (W4-A review:
    two always_ff blocks wrote the operand extents; Verilator simulated it correctly, Vivado did not)."""
    import subprocess
    import pytest
    from common.runner import build_root
    vivado = Path(os.environ.get("TSR_VIVADO", "/tool/Xilinx/Vivado/2024.2/bin/vivado"))
    wanted = os.environ.get("TSR_CHK_VIVADO") == "1" or "netlist" in (request.config.getoption("keyword") or "") or \
        any("test_chk_netlist" in a for a in request.config.args)
    if not wanted:
        pytest.skip("opt-in Vivado run: set TSR_CHK_VIVADO=1 or select it with -k netlist")
    if not vivado.exists():
        pytest.skip("Vivado not found (%s)" % vivado)
    out = build_root() / "chk_net"
    r = subprocess.run(["bash", str(HERE / "viv/net_chk.sh"), str(out)], capture_output=True, text=True,
                       env=dict(os.environ, TSR_PYTHON=sys.executable))
    print(r.stdout, r.stderr)
    assert r.returncode == 0, "netlist regression failed:\n%s\n%s" % (r.stdout[-2000:], r.stderr[-2000:])

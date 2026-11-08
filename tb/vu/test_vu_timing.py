"""tb/vu timing — the cycles of VU op streams on the RTL unit (tsr_vu on a real tsr_spm, no other client) against the
simulator's VU cycle model (sw/sim/tsr_vu.c; docs/design/02-vector-unit.md "VU" §7.9, SW §4.5), per op.

vu_stream      pushes the ops of a stream file (TSR_VU_STREAM: one op per line, eight hex words; the unit-3 pushes of a
               simulator run, `TSR_SIM_LOG=w,<file>`) as fast as the unit accepts them and writes the cycle of every done
               pulse, with the hazard and SPM-wait cycles so far, to TSR_VU_STREAM_OUT (json). The data is not checked (tb/vu
               test_vu_unit does that); the timing is the point.
vu_gdn_step    the ops of one Gated DeltaNet head step (sw/ctrl/kernels/gdn_kinds.cpp: S_v 128, L 8, 13 VU ops) on tiles of
               a 4 MiB SPM, repeated; no external file. The test asserts the model against the RTL (see _TOL).
vu_gdn_op      the fused GDN op (VU §7.12, opcode 0x39; the unit built with HAS_GDN 1), one op and four dependent in-place
               repetitions of it on a 128 x 128 head: the model's cycles of the op (sw/sim/tsr_vu.c cyc_gdn, SW-78)
               against the RTL's, at the front lanes of the configuration (2,064 beats at 8 lanes).

The RTL configuration is the card image (VU 8 / 8 / 4 lanes behind two lane-pipe stages, SPM_RD_LAT 4, four channels, no
TOPK / DQROW) unless TSR_VU_TIMING_CFG names the U200 lanes ("u200": 16 / 16 / 8, no lane pipe). The simulator side is
configured the same way (TSR_SIM_VU_TIMING, tsr_sim.h).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for _p in (HERE.parent / "issue",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.runner import REPO, run       # noqa: E402

import vu_ops as V                         # noqa: E402
from vu_env import Rtl                     # noqa: E402
import test_vu_unit as U                   # noqa: E402

CFGS = {
    # name: (front, exp, back, lane_pipe)
    "k4": (8, 8, 4, 2),
    "u200": (16, 16, 8, 0),
    # the other two corners of the lane-count / lane-pipe square (the model's fit; no image has them)
    "k4p0": (8, 8, 4, 0),
    "u200p2": (16, 16, 8, 2),
}
SPM_BYTES = 4 << 20


def timing_cfg():
    return CFGS[os.environ.get("TSR_VU_TIMING_CFG", "k4")]


class Timer:
    """The RTL unit alone on a 4 MiB SPM: push ops, keep the done log."""

    def __init__(self, dut):
        self.dut = dut
        self.lines = SPM_BYTES // 64
        self.rtl = Rtl(dut, self.lines)

    async def start(self):
        cocotb.start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        await self.rtl.reset(noise=False, seed=1)
        cocotb.start_soon(self.rtl.monitor())
        # the SPM content is arbitrary for timing; leave what the memory model starts with

    async def run(self, ops):
        """Push every op; returns [(done cycle, hazard cycles, SPM-wait cycles)] relative to the first push, and the
        push cycles."""
        r = self.rtl
        n0 = len(r.done_log)
        c0 = r.cycles
        h0, w0 = r.hazard_cycles, r.wait_cycles
        pushed = []
        for w in ops:
            await r.push(w)
            pushed.append(r.cycles - c0)
        await r.wait_done(n0 + len(ops), limit=50_000_000)
        log = [(c - c0, h - h0, wt - w0) for (c, h, wt) in r.done_log[n0:n0 + len(ops)]]
        return log, pushed


def read_stream(path):
    ops = []
    for line in Path(path).read_text().split("\n"):
        line = line.strip()
        if line:
            ops.append([int(x, 16) for x in line.split()])
    return ops


@cocotb.test()
async def vu_stream(dut):
    """The stream of TSR_VU_STREAM on the RTL unit; the per-op done cycles go to TSR_VU_STREAM_OUT."""
    ops = read_stream(os.environ["TSR_VU_STREAM"])
    lim = int(os.environ.get("TSR_VU_STREAM_LIMIT", "0"))
    if lim:
        ops = ops[:lim]
    t = Timer(dut)
    await t.start()
    log, pushed = await t.run(ops)
    out = dict(config=dict(zip(("front", "exp", "back", "lane_pipe"), timing_cfg())), n=len(ops), done=log, pushed=pushed)
    Path(os.environ["TSR_VU_STREAM_OUT"]).write_text(json.dumps(out) + "\n")
    dut._log.info("vu_stream: %d ops, %d cycles, hazard %d, SPM wait %d", len(ops), log[-1][0], log[-1][1], log[-1][2])


# ---------------------------------------------------------------------------------------------
# The GDN head step
# ---------------------------------------------------------------------------------------------
S_V, LROW = 128, 8
SZ = S_V * LROW            # lines of a state tile


def gdn_steps(heads: int):
    """The VU ops of `heads` head steps of GATED_DELTA_NET (T = 1, S_v 128) as plan_gdn / gdn_kinds.cpp emit them, with
    the two state tiles alternating between heads."""
    St = [0, SZ]
    P = 2 * SZ
    sl = 3 * SZ                 # token slot: k q v g b u w (5 L + 2 lines)
    k, q, v = sl, sl + LROW, sl + 2 * LROW
    g, b, u = sl + 3 * LROW, sl + 3 * LROW + 1, sl + 3 * LROW + 2
    w = sl + 4 * LROW + 2
    ops = []
    S_G, S_SCALE, S_BETA, S_ALPHA = 12, 13, 14, 15
    for h in range(heads):
        S = St[h & 1]
        ops += [V.enc_sop(7, sd=S_G, addr=16 * g),                       # LDS g
                V.enc_sop(14, sd=S_ALPHA, sa=S_G),                       # EXP alpha
                V.enc_sop(7, sd=S_BETA, addr=16 * b),                    # LDS beta
                V.enc_vop(V.VOP_BY_NAME["MULC"], S, S, 0, 0, S_V, S_V, LROW, LROW, 0, 1, S_ALPHA),   # S *= alpha
                V.enc_vop(V.VOP_BY_NAME["MUL"], P, S, k, 0, S_V, S_V, LROW, LROW, 0, 0),           # P = S * k
                V.enc_red(0, u, P, S_V, S_V, LROW),                                                # u = rowsums
                V.enc_vop(V.VOP_BY_NAME["SUB"], u, v, u, 0, S_V, 1, 0, 0, 0, 0),
                V.enc_vop(V.VOP_BY_NAME["MULC"], u, u, 0, 0, S_V, 1, 0, 0, 0, 1, S_BETA),
                V.enc_vop(V.VOP_BY_NAME["MULCADD"], S, k, S, u, S_V, S_V, LROW, 0, LROW, 2),       # S = k delta + S
                V.enc_vop(V.VOP_BY_NAME["MUL"], P, S, q, 0, S_V, S_V, LROW, LROW, 0, 0),           # P = S * q
                V.enc_red(0, w, P, S_V, S_V, LROW),                                                # w = rowsums
                V.enc_vop(V.VOP_BY_NAME["MULC"], w, w, 0, 0, S_V, 1, 0, 0, 0, 1, S_SCALE)]         # o = w * scale
    return ops


def sim_done_times(ops, vu_timing=None, caps=0x21):
    """The simulator's done time of every VU op (one VU queue, no other unit running), relative to the first push."""
    import issue_sim as S
    cfg = S.config(num_ch=4, t_max=4, spm_bytes=SPM_BYTES, caps=caps)
    if vu_timing in CFGS:
        vu_timing = ",".join(str(x) for x in CFGS[vu_timing])     # the simulator knows "u200" and "k4" by name only
    sim = S.Sim(cfg, (8, 4, 4, 16, 0, 4), vu_timing=vu_timing)
    try:
        sim.listen()
        t0 = sim.now()
        for w in ops:
            sim.push(w)          # waits for a queue credit like the hardware (time advances)
        sim.run_until_idle()
        return [e["time"] - t0 for e in sim.events if e["kind"] == S.QEV_DONE and e["unit"] == 3]
    finally:
        sim.close()


@cocotb.test()
async def vu_gdn_step(dut):
    heads = int(os.environ.get("TSR_VU_GDN_HEADS", "3"))
    t = Timer(dut)
    await t.start()
    ops = gdn_steps(heads)
    log, _ = await t.run(ops)
    per = len(ops) // heads
    gaps = [log[(i + 1) * per - 1][0] - log[i * per - 1][0] for i in range(1, heads)]
    out = os.environ.get("TSR_VU_TIMING_OUT")
    if out:
        Path(out).write_text(json.dumps(dict(heads=heads, per_step=per, done=log, gaps=gaps)) + "\n")
    dut._log.info("vu_gdn_step: %d heads, %d ops, head-step gaps %s cycles, hazard %d, SPM wait %d", heads, len(ops), gaps,
                  log[-1][1], log[-1][2])


GDN_S, GDN_K, GDN_Q, GDN_V, GDN_O = 0, 5000, 5040, 5080, 5120     # lines of the tile (up to 256 rows x 16 lines) and the vectors of the GDN op test


GDN_SHAPES = [(128, 128), (64, 128), (256, 128), (128, 16), (128, 64), (128, 256), (64, 16), (256, 256), (256, 16), (192, 100)]


@cocotb.test()
async def vu_gdn_op(dut):
    """TSR_VU_TIMING_OUT: a list of {"m", "rows", "one": T(1), "four": T(4)} of the GDN op in place on one tile (so each
    repetition waits for the one before), in RTL cycles from the first push to the last done, for the shapes of GDN_SHAPES."""
    t = Timer(dut)
    await t.start()
    await t.run([V.enc_sop(0, sd=i, imm=0x3F800000) for i in range(16)])
    res = []
    for m, rows in GDN_SHAPES:
        op = V.enc_gdn(1, 2, GDN_S, GDN_S, GDN_K, GDN_Q, GDN_V, GDN_O, m, rows)
        r = dict(m=m, rows=rows)
        for name, n in (("one", 1), ("four", 4)):
            log, _ = await t.run([op] * n)
            r[name] = log[-1][0]
        res.append(r)
        dut._log.info("vu_gdn_op: m %d rows %d: one op %d cycles, four dependent ops %d", m, rows, r["one"], r["four"])
    Path(os.environ["TSR_VU_TIMING_OUT"]).write_text(json.dumps(res) + "\n")


# ---------------------------------------------------------------------------------------------
# Single-op sweeps: the marginal cost of one more op of a shape, dependent (the next op reads or overwrites the previous
# one's result, so it waits for it) and independent (disjoint regions, the unit overlaps their fill and drain)
# ---------------------------------------------------------------------------------------------
def _f(name):
    return V.VOP_BY_NAME[name]


def sweep_cases():
    """[(name, make(rep, chained) -> words)]: one op of the shape; `rep` moves the regions apart when not chained."""
    cases = []
    R = 3300                                           # lines between the regions of independent repetitions

    def vop(name, m, rows=1, cm=0, two=True, tile=False):
        f = _f(name)
        reads = V.VOP_FUNCS[f][1]
        L = V.L(m)
        ds = L if rows > 1 else 0

        def make(rep, chained):
            o = 0 if chained else rep * R
            X, Y, C = 16 + o, 16 + 1100 + o, 16 + 2200 + o
            return V.enc_vop(f, X, X if "a" in reads else 0, Y if "b" in reads else 0, C if cm == 2 else 0, m, rows, ds,
                             ds if "a" in reads else 0, 0 if tile else (ds if "b" in reads else 0), cm, 12 if cm == 1 else 0)
        return make

    for name, ms in (("ADD", (16, 32, 48, 128, 144, 1024)), ("COPY", (128, 1024)), ("MULC", (128, 1024)),
                     ("EXP", (128, 1024)), ("EXPSUB", (128, 1024)), ("SILU", (128, 1024)), ("SIGMOID", (128,)),
                     ("SWIGLU", (128, 1024)), ("RECIP", (128,)), ("GELU16", (128, 1024))):
        for m in ms:
            cm = 1 if name in ("MULC",) else 0
            cases.append(("VOP.%s L%d" % (name, V.L(m)), vop(name, m, cm=cm)))
    for name, rows, ls in (("ADD", 4, (1, 3, 8, 9)), ("ADD", 16, (1, 3, 8, 9)), ("ADD", 128, (1, 3, 8, 9)), ("COPY", 16, (1, 8)),
                           ("SILU", 16, (1, 8)), ("EXP", 16, (1, 8)), ("SWIGLU", 16, (1, 3))):
        for L in ls:
            cases.append(("VOP.%s rows%d L%d" % (name, rows, L), vop(name, 16 * L, rows=rows)))
    cases.append(("VOP.MUL tile 128x8 (b broadcast)", vop("MUL", 128, rows=128, tile=True)))
    cases.append(("VOP.MULC tile 128x8 (s reg)", vop("MULC", 128, rows=128, cm=1)))
    cases.append(("VOP.MULCADD tile 128x8 (vec c)", vop("MULCADD", 128, rows=128, cm=2, tile=True)))

    def red(kind, m, rows, post=0, to_s=0):
        L = V.L(m)

        def make(rep, chained):
            o = 0 if chained else rep * R
            return V.enc_red(kind, 0 if to_s else 16 + o, 16 + 1100 + o, m, rows, L, to_s, 0, post, 0, 1 if post == 2 else 0)
        return make

    for m, rows in ((128, 1), (256, 1), (2048, 1), (4096, 1), (128, 16), (256, 16), (128, 128), (64, 128)):
        cases.append(("RED.SUM m%d rows%d" % (m, rows), red(0, m, rows)))
    for m in (2048, 4096):
        cases.append(("RED.SUMSQ.rms m%d" % m, red(1, m, 1, post=2)))
    cases.append(("RED.MAX m256", red(2, 256, 1)))
    cases.append(("RED.SUM.to_s m256", red(0, 256, 1, to_s=1)))

    def rope(hd, nd, nh, T, form):
        H = nh * hd // 16
        ff, atab = int(form == "ff"), int(form == "atab")

        def make(rep, chained):
            o = 0 if chained else rep * R
            d = 16 + o
            src = d if chained else 16 + 1100 + o
            return V.enc_rope(1, d, src, 16 + 2200 + o, hd, nd, nh, T, (16 + 2300 + o) if (ff or atab) else 0, 0, 0, ff, 0, atab,
                              1, 0 if atab else 2, 0)
        return make

    for hd, nd in ((64, 64), (128, 128), (256, 64), (256, 256), (128, 64)):
        for nh in (1, 4, 16):
            for T in (1, 2):
                cases.append(("ROPE rec hd%d nd%d nh%d T%d" % (hd, nd, nh, T), rope(hd, nd, nh, T, "rec")))
    for hd, nd, nh in ((256, 64, 8), (256, 64, 6), (256, 64, 12)):   # rotation as long as the angles (Gemma prefill)
        for T in (3, 4, 7, 8):
            cases.append(("ROPE rec hd%d nd%d nh%d T%d" % (hd, nd, nh, T), rope(hd, nd, nh, T, "rec")))
    for hd, nd in ((128, 128), (256, 256)):
        for nh in (1, 4):
            cases.append(("ROPE ff hd%d nd%d nh%d T1" % (hd, nd, nh), rope(hd, nd, nh, 1, "ff")))
            cases.append(("ROPE atab hd%d nd%d nh%d T1" % (hd, nd, nh), rope(hd, nd, nh, 1, "atab")))

    def sop(sub, **kw):
        return lambda rep, chained: V.enc_sop(sub, **kw)
    cases.append(("SOP.MOVI", sop(0, sd=1, imm=0x3F800000)))
    cases.append(("SOP.ADD", sop(1, sd=2, sa=1, sb=1)))
    cases.append(("SOP.LDS", sop(7, sd=3, addr=16 * 20)))
    cases.append(("SOP.EXP", sop(14, sd=3, sa=3)))
    cases.append(("SOP.RSQRT", sop(11, sd=4, sa=1)))
    cases.append(("SOP.DIV", sop(10, sd=5, sa=1, sb=1)))
    return cases


@cocotb.test()
async def vu_sweep(dut):
    """TSR_VU_SWEEP_OUT: {name: {"chain": [T(1), T(4)], "free": [T(1), T(4)]}} in RTL cycles (first push to last done)."""
    t = Timer(dut)
    await t.start()
    # s registers: a defined value
    await t.run([V.enc_sop(0, sd=i, imm=0x3F800000) for i in range(16)])
    res = {}
    for name, make in sweep_cases():
        r = {}
        for mode, chained in (("chain", True), ("free", False)):
            r[mode] = []
            for n in (1, 4):
                log, _ = await t.run([make(i, chained) for i in range(n)])
                r[mode].append(log[-1][0])
        res[name] = r
        dut._log.info("%-36s chain %s free %s", name, r["chain"], r["free"])
    Path(os.environ["TSR_VU_SWEEP_OUT"]).write_text(json.dumps(res, indent=1) + "\n")


@cocotb.test()
async def vu_sweep_words(dut):
    """TSR_VU_WORDS (ops, one per line), TSR_VU_SWEEP_OUT: for every distinct op, T(1) and T(4) of the same op repeated
    (each repetition overwrites or reads the previous one's result: dependent), in RTL cycles."""
    t = Timer(dut)
    await t.start()
    await t.run([V.enc_sop(0, sd=i, imm=0x3F800000) for i in range(16)])
    res = {}
    for w in unique_ops(read_stream(os.environ["TSR_VU_WORDS"])):
        r = []
        for n in (1, 4):
            log, _ = await t.run([w] * n)
            r.append(log[-1][0])
        res[" ".join("%08x" % x for x in w)] = r
    Path(os.environ["TSR_VU_SWEEP_OUT"]).write_text(json.dumps(res, indent=1) + "\n")


def unique_ops(ops):
    """The distinct ops, in order; every AMERGE also with the opposite INIT flag (a merging one is not in the decode
    streams of one group)."""
    seen, out = set(), []
    for w in ops:
        for v in ([w] if (w[0] & 0xFF) != 0x36 else [w, [w[0] ^ 0x1000] + w[1:]]):
            k = tuple(v)
            if k not in seen:
                seen.add(k)
                out.append(v)
    return out


def sim_sweep(vu_timing=None):
    res = {}
    for name, make in sweep_cases():
        r = {}
        for mode, chained in (("chain", True), ("free", False)):
            r[mode] = []
            for n in (1, 4):
                t = sim_done_times([make(i, chained) for i in range(n)], vu_timing)
                r[mode].append(t[-1])
        res[name] = r
    return res


def sweep_table(rtl, sim):
    """Rows: name, RTL and simulator marginal cycles per op, dependent and independent."""
    rows = []
    for name in rtl:
        row = [name]
        for mode in ("chain", "free"):
            a, b = rtl[name][mode], sim[name][mode]
            row += [(a[1] - a[0]) / 3.0, (b[1] - b[0]) / 3.0]
        row += [rtl[name]["chain"][0], sim[name]["chain"][0]]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# pytest
# ---------------------------------------------------------------------------------------------
def _run(name, testcase, extra_env=None, gdn=False):
    f, e, b, p = timing_cfg()
    env = {"TSR_VU_TIMING_CFG": os.environ.get("TSR_VU_TIMING_CFG", "k4")}
    env.update(extra_env or {})
    run("tsr_vu_tb", __name__, sources=U.vu_sources(), name="vu_timing_" + os.environ.get("TSR_VU_TIMING_CFG", "k4") + ("_gdn" if gdn else ""),
        testcase=testcase, extra_env=env,
        parameters=dict(SPM_BYTES=SPM_BYTES, SPM_RD_LAT=4, NUM_CH=4, FRONT_LANES=f, EXP_LANES=e, BACK_LANES=b, LANE_PIPE=p,
                        HAS_TOPK=0, HAS_DQROW=0, **(dict(HAS_GDN=1) if gdn else {})))


_TOL = 0.02     # the model must be within 2 % of the RTL on a head step (0.8 % today: it does not overlap the independent SOPs)


@pytest.mark.parametrize("cfg", ["k4", "u200"])
def test_vu_timing_gdn_step(tmp_path, monkeypatch, cfg):
    """One GDN head step repeated on the RTL unit: the model (TSR_SIM_VU_TIMING = the RTL's configuration) must take what
    the RTL takes, within _TOL, per head step in the steady state."""
    monkeypatch.setenv("TSR_VU_TIMING_CFG", cfg)
    out = tmp_path / "gdn.json"
    heads = 4
    _run("gdn", "vu_gdn_step", {"TSR_VU_TIMING_OUT": str(out), "TSR_VU_GDN_HEADS": str(heads)})
    r = json.loads(out.read_text())
    per = r["per_step"]
    rtl = r["gaps"]
    ops = gdn_steps(heads)
    t = sim_done_times(ops, os.environ.get("TSR_VU_TIMING_CFG", "k4"))
    sim = [t[(i + 1) * per - 1] - t[i * per - 1] for i in range(1, heads)]
    print("head step cycles: RTL %s, simulator %s" % (rtl, sim))
    for a, b in zip(rtl, sim):
        assert abs(a - b) <= _TOL * a, "head step: RTL %d, simulator %d cycles" % (a, b)


@pytest.mark.parametrize("cfg", ["k4", "u200"])
def test_vu_timing_gdn_op(tmp_path, monkeypatch, cfg):
    """The fused GDN op on the RTL unit (HAS_GDN 1) against the model, as the marginal cost of a dependent in-place
    repetition, for the shapes of GDN_SHAPES (SW-78: cyc_gdn follows the front lanes of TSR_SIM_VU_TIMING). The 128 x 128 head
    of Qwen3.8 must agree within _GDN_TOL (_GDN_TOL_U200), every shape within _GDN_TOL_ALL."""
    monkeypatch.setenv("TSR_VU_TIMING_CFG", cfg)
    out = tmp_path / "gdnop.json"
    _run("gdnop", "vu_gdn_op", {"TSR_VU_TIMING_OUT": str(out)}, gdn=True)
    movi = [V.enc_sop(0, sd=i, imm=0x3F800000) for i in range(16)]
    bad = []
    print("GDN op, %s: %5s %5s | %9s %9s | %7s" % (cfg, "m", "rows", "RTL marg", "sim marg", "RTL one"))
    for r in json.loads(out.read_text()):
        op = V.enc_gdn(1, 2, GDN_S, GDN_S, GDN_K, GDN_Q, GDN_V, GDN_O, r["m"], r["rows"])
        t1 = sim_done_times(movi + [op], cfg, caps=0x121)
        t4 = sim_done_times(movi + [op] * 4, cfg, caps=0x121)
        sim_marg = (t4[-1] - t1[-1]) / 3.0
        rtl_marg = (r["four"] - r["one"]) / 3.0
        print("GDN op, %s: %5d %5d | %9.1f %9.1f | %7d" % (cfg, r["m"], r["rows"], rtl_marg, sim_marg, r["one"]))
        tol = (_GDN_TOL if cfg == "k4" else _GDN_TOL_U200) if (r["m"], r["rows"]) == (128, 128) else _GDN_TOL_ALL
        if abs(rtl_marg - sim_marg) > tol * rtl_marg:
            bad.append("m %d rows %d: RTL %.1f, simulator %.1f cycles per dependent op" % (r["m"], r["rows"], rtl_marg, sim_marg))
    assert not bad, "GDN op: " + "; ".join(bad)


_GDN_TOL = 0.005       # the 128 x 128 head of the Qwen3.8 models at the K4 lanes (the model is within 2 cycles of 2,914)
_GDN_TOL_U200 = 0.015   # the same head at the U200 lanes (1,898 against 1,917)
_GDN_TOL_ALL = 0.03     # every other shape, both lane configurations


def test_vu_timing_stream():
    """Opt-in: a stream file of a simulator run (TSR_VU_STREAM, TSR_VU_STREAM_OUT)."""
    if "TSR_VU_STREAM" not in os.environ:
        pytest.skip("set TSR_VU_STREAM (and TSR_VU_STREAM_OUT) to time a stream file")
    _run("stream", "vu_stream")


def test_vu_timing_sweep(tmp_path):
    """Opt-in (TSR_VU_SWEEP=1): the single-op sweep on the RTL unit and in the simulator; prints the table and writes it
    to TSR_VU_SWEEP_TABLE (json) when set."""
    if not os.environ.get("TSR_VU_SWEEP"):
        pytest.skip("set TSR_VU_SWEEP=1 to run the single-op sweep")
    out = tmp_path / "sweep.json"
    _run("sweep", "vu_sweep", {"TSR_VU_SWEEP_OUT": str(out)})
    rtl = json.loads(out.read_text())
    sim = sim_sweep(os.environ.get("TSR_VU_TIMING_CFG", "k4"))
    rows = sweep_table(rtl, sim)
    print("%-36s %9s %9s | %9s %9s | %7s %7s" % ("op (marginal cycles)", "RTL dep", "sim dep", "RTL free", "sim free",
                                                 "RTL one", "sim one"))
    for r in rows:
        print("%-36s %9.1f %9.1f | %9.1f %9.1f | %7d %7d" % tuple(r))
    if os.environ.get("TSR_VU_SWEEP_TABLE"):
        Path(os.environ["TSR_VU_SWEEP_TABLE"]).write_text(json.dumps(rows, indent=1) + "\n")


def test_vu_timing_words(tmp_path):
    """Opt-in (TSR_VU_WORDS=<file>): the distinct ops of a stream file, each repeated four times, RTL against the
    simulator (marginal cycles of a dependent repetition); TSR_VU_SWEEP_TABLE gets the rows."""
    if "TSR_VU_WORDS" not in os.environ:
        pytest.skip("set TSR_VU_WORDS to a stream file")
    out = tmp_path / "words.json"
    _run("words", "vu_sweep_words", {"TSR_VU_SWEEP_OUT": str(out)})
    rtl = json.loads(out.read_text())
    rows = []
    cfg = os.environ.get("TSR_VU_TIMING_CFG", "k4")
    for key, (a1, a4) in rtl.items():
        w = [int(x, 16) for x in key.split()]
        s1 = sim_done_times([w], cfg)[-1]
        s4 = sim_done_times([w] * 4, cfg)[-1]
        rows.append([key, (a4 - a1) / 3.0, (s4 - s1) / 3.0])
    for r in rows:
        print("%s  RTL %8.1f  sim %8.1f" % tuple(r))
    if os.environ.get("TSR_VU_SWEEP_TABLE"):
        Path(os.environ["TSR_VU_SWEEP_TABLE"]).write_text(json.dumps(rows, indent=1) + "\n")

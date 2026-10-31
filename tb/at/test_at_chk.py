"""tb/at test_at_chk — tsr_chk_at (the AT checker classes of tsr_chk's external-class port) against the
simulator's chk_kvst / chk_attqk / chk_attpv (sw/sim/tsr_at.c; AT §5.8, EX §8.1).

For every op: the error code (ILLEGAL_OP, BAD_FMT, K_RANGE, T_RANGE, OPERAND, SPM_RANGE, DRAM_RANGE in their
order), ERR_INFO, the tagged extents (kind, direction, channel, [lo, hi)) and, for a passing op, the unit
payload (tsr_att_pkg tsr_kvst_cmd_t / tsr_att_cmd_t, at_isa.payload) are compared with
tsr_sim_accept_op + the handler's check (tb/issue/issue_sim.py Sim.check). The latency (request to
rsp_valid) is 8 cycles for every op.
Stimulus: random words of the three opcodes, and per op edge-heavy operands: E/64 = 0 / D multiples / not,
G E around K_MAX, T_q GS around T_MAX, hg0 + GS around G, n_groups 0 / 1 / 4095 / 4096 / 65535, c_lo / c_hi
around 16 n, pass rows around 128, strides below / at n, l_max around LG, PCAP, extents ending at the SPM and
DRAM limits +- 1, reserved bits one at a time.
Configurations: U200 (4 channels, T_MAX 8, K_MAX 32768, 4 MiB SPM, 16 GiB channels), 3 channels (the
divider by a non-power of two; small SPM, ADDR_BEAT_W 22), 16 channels with K_MAX 65280 (PCAP), 1 channel T_MAX 1.
"""
from __future__ import annotations

import json
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

from common import isa                                   # noqa: E402
from common.runner import REPO, run                      # noqa: E402

import at_isa as A                                       # noqa: E402
import issue_sim as S                                    # noqa: E402
import issue_tb as E                                     # noqa: E402

M32 = 0xFFFFFFFF


def src():
    d = os.environ.get("TSR_MUT_DIR")
    f = Path(d) / "tsr_chk_at.sv" if d and (Path(d) / "tsr_chk_at.sv").exists() else REPO / "rtl/core/tsr_chk_at.sv"
    return [REPO / "rtl/core/tsr_tables_pkg.sv", REPO / "rtl/core/tsr_pkg.sv", REPO / "rtl/core/tsr_att_pkg.sv", f]


def cfg_from_env():
    c = json.loads(os.environ.get("TSR_ATCHK_CFG", "{}"))
    cfg = S.config(num_ch=c.get("num_ch", 4), t_max=c.get("t_max", 8), k_max=c.get("k_max", 32768),
                   spm_bytes=c.get("spm_bytes", 4 << 20), ch_bytes=c.get("ch_bytes", 16 << 30),
                   addr_beat_w=c.get("addr_beat_w", 28), caps=0x20)
    return cfg, c.get("n_ops", 3000)


class Gen:
    def __init__(self, cfg, seed):
        self.cfg, self.r = cfg, random.Random(seed)
        self.C = cfg["num_ch"]
        self.spm = cfg["spm_bytes"] // 64
        self.dram = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])

    def pick(self, *vals):
        return self.r.choice(vals)

    def near(self, lim, ext):
        """A start address so that start + ext lands at lim - 2 .. lim + 2 (often), else random."""
        r = self.r
        if r.random() < 0.5 and ext <= lim + 2:
            return max(0, lim - ext + r.randint(-2, 2)) & M32
        return r.randrange(0, max(1, min(lim, 1 << 32)))

    def rsv(self, w, op):
        masks = {A.OP_KVST: "KVST", A.OP_ATTQK: "ATTQK", A.OP_ATTPV: "ATTPV"}[op]
        from common import tables as T
        r = self.r
        if r.random() < 0.06:
            i = r.randrange(8)
            m = getattr(T, "%s_RSV_W%d" % (masks, i))
            if m:
                bits = [b for b in range(32) if (m >> b) & 1]
                w[i] |= 1 << r.choice(bits)
        return w

    def kvst(self):
        r = self.r
        e64 = self.pick(0, 1, 2, 3, 8, 16, 17, 31, 64, 255, 511, r.randint(1, 511))
        nr = self.pick(1, 2, 15, 16, 17, 63, 64, r.randint(1, 64))
        lg = self.pick(0, 1, 2, 3, 7, 1000, 65535, r.randint(1, 4096))
        cells = 16 * self.C * lg
        c0 = self.pick(0, max(0, cells - nr), max(0, cells - nr + 1), cells, r.randrange(0, max(1, cells + 2)),
                       r.getrandbits(32))
        qa = e64 + (e64 + 7) // 8
        stride = self.pick(0, 0, qa, qa + 1, 1, r.getrandbits(12), r.getrandbits(32))
        es = (nr - 1) * (stride if stride else qa) + qa
        src_ = self.near(self.spm, es)
        base = self.near(self.dram, lg * 17 * e64)
        kv = self.pick(0, 1, 0, 1, 2, 15) if r.random() < 0.1 else r.randint(0, 1)
        return self.rsv(A.enc_kvst(kv, nr, src_, base, e64, lg, c0, stride), A.OP_KVST)

    def shape(self):
        r = self.r
        tm = self.cfg["t_max"]
        dcode = r.randint(0, 3)
        D = 64 << dcode
        if r.random() < 0.8:
            nseg = self.pick(1, 1, 2, 3, 4, 8, 16, r.randint(1, 64))
            e64 = nseg * (D // 64)
            if e64 > 511:
                e64 = (511 // (D // 64)) * (D // 64)
        else:
            e64 = self.pick(0, r.randint(1, 511))
        G = self.pick(1, 2, 4, 5, 7, 8, 16, 32, 64, r.randint(1, 64))
        GS = self.pick(1, 2, 4, 8, min(G, 8), r.randint(1, 8))
        tq = self.pick(1, max(1, tm // GS), max(1, tm // GS) + 1, r.randint(1, 8))
        hg0 = self.pick(0, 0, max(0, G - GS), max(0, G - GS + 1), r.randint(0, 63))
        if r.random() < 0.1:
            G = max(1, self.cfg["k_max"] // max(1, 64 * e64)) + self.pick(0, 1)
            G = min(G, 64)
        return e64, dcode, G, hg0, GS, tq

    def groups(self):
        r = self.r
        n = self.pick(1, 1, 2, 3, 4, 5, 16, 128, 4095, 4096, 0, 65535, r.randint(1, 4095), r.randint(1, 300))
        u0 = self.pick(0, 1, 3, 65535, 65535 - n, r.randint(0, 65535), r.randint(0, 64))
        return u0 & 0xFFFF, n

    def attqk(self):
        r = self.r
        e64, dcode, G, hg0, GS, tq = self.shape()
        u0, n = self.groups()
        D = 64 << dcode
        nseg = (64 * e64) // D if e64 else 0
        n_head = G * nseg
        maxrow = (tq - 1) * n_head + max(0, nseg - 1) * G + hg0 + GS - 1
        rs = self.pick(0, 0, 0, n, max(0, n - 1), n + 1, r.randint(0, 32767))
        ed = maxrow * (rs if rs else n) + n
        c_hi = self.pick(16 * n, 16 * n + 1, 0, r.randint(0, 16 * n + 2) if n < 4096 else 0, r.getrandbits(16))
        c_lo = self.pick(0, c_hi, c_hi + 1, r.randint(0, max(0, c_hi)), r.getrandbits(16))
        maxout = r.randint(0, 1)
        mdst = self.near(self.spm, (maxrow + 16) // 16) if maxout else self.pick(0, 0, 0, 0, 1)
        GE = G * 64 * e64
        x = self.near(self.spm, tq * A.qa32_lines(GE))
        ep = ((u0 + n - 1) // self.C + 1) * 17 * e64 if n else 17 * e64
        w = A.enc_attqk(dst=self.near(self.spm, ed), x=x, base=self.near(self.dram, ep), e64=e64, dcode=dcode, G=G,
                        hg0=hg0, GS=GS, tq=tq, u0=u0, n=n, c_lo=c_lo, c_hi=c_hi, rs=rs, mdst=mdst,
                        causal=r.randint(0, 1), slide=r.randint(0, 1), maxout=maxout,
                        fmt=self.pick(0, 0, 0, 0, 0, 0, 0, 1, 15))
        return self.rsv(w, A.OP_ATTQK)

    def attpv(self):
        r = self.r
        e64, dcode, G, hg0, GS, tq = self.shape()
        u0, n = self.groups()
        D = 64 << dcode
        nseg = (64 * e64) // D if e64 else 0
        n_head = G * nseg
        R = tq * n_head
        maxrow = (tq - 1) * n_head + max(0, nseg - 1) * G + hg0 + GS - 1
        lmax = (u0 + n - 1) // self.C if n else 0
        LG = self.pick(lmax + 1, lmax, lmax + 2, 0, 1, 65535, r.randint(0, 65535))
        rs = self.pick(0, 0, 0, n, max(0, n - 1), n + 1, r.randint(0, 32767))
        lsum = r.randint(0, 1)
        rl = R * D // 16
        cs = self.pick(0, 0, rl, rl + 1, r.getrandbits(12), r.getrandbits(32))
        csr = cs if cs else rl
        ed = (self.C - 1) * csr + rl
        if lsum:
            ed = max(ed, self.C * csr + self.C * ((R + 15) // 16))
        ep = maxrow * (rs if rs else n) + n
        ev = ((max(0, e64 - 1)) * LG + lmax + 1) * 17
        w = A.enc_attpv(dst=self.near(self.spm, ed), p=self.near(self.spm, ep), base=self.near(self.dram, ev),
                        e64=e64, dcode=dcode, G=G, hg0=hg0, GS=GS, tq=tq, u0=u0, n=n, LG=LG, rs=rs, cstride=cs,
                        lsum=lsum, fmt=self.pick(0, 0, 0, 0, 0, 0, 0, 1, 15))
        return self.rsv(w, A.OP_ATTPV)

    def op(self):
        r = self.r.random()
        if r < 0.03:
            op = self.pick(A.OP_KVST, A.OP_ATTQK, A.OP_ATTPV)
            return [op | (self.r.getrandbits(24) << 8)] + [self.r.getrandbits(32) for _ in range(7)]
        if r < 0.3:
            return self.kvst()
        if r < 0.65:
            return self.attqk()
        return self.attpv()


async def check_ops(dut, cfg, n_ops, seed):
    sim = S.Sim(cfg, (8, 4, 4, 16, 0, 4))
    g = Gen(cfg, seed)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.req_valid.value = 0
    dut.req_insn.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    bad, hist, lat = [], {}, set()
    for _ in range(n_ops):
        w = [x & M32 for x in g.op()]
        unit, code, info, exts = sim.check(w)
        hist[code] = hist.get(code, 0) + 1
        dut.req_insn.value = isa.insn_int(w)
        dut.req_valid.value = 1
        n = 0
        while True:
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            n += 1
            if int(dut.rsp_valid.value):
                break
            # tsr_xc_pkg: a class's rsp_* are 0 outside its rsp_valid cycle (the answers are ORed)
            if int(dut.rsp_code.value) or int(dut.rsp_info.value) or int(dut.rsp_ext.value) or int(dut.rsp_cmd.value):
                bad.append("op %s: rsp_* not 0 before rsp_valid" % [hex(x) for x in w])
            if n > 50:
                raise TimeoutError("no rsp_valid for %s" % [hex(x) for x in w])
        lat.add(n - 1)                       # edges after the accepting edge (tb/chk's count)
        got = dict(code=int(dut.rsp_code.value), info=int(dut.rsp_info.value),
                   ext=E.unpack_ext(int(dut.rsp_ext.value)))
        exp = dict(code=code, info=info, ext=E.sim_ext_to_rtl(exts)[:4])
        if got != exp:
            bad.append("op %s: RTL %s, simulator %s" % ([hex(x) for x in w], got, exp))
        elif code == 0:
            pw = A.ATT_CMD_T.width if (w[0] & 0xFF) != A.OP_KVST else A.KVST_CMD_T.width
            gp = int(dut.rsp_cmd.value) & ((1 << pw) - 1)
            ep = A.payload(w)
            if gp != ep:
                st = A.KVST_CMD_T if (w[0] & 0xFF) == A.OP_KVST else A.ATT_CMD_T
                bad.append("op %s: payload %s, expected %s" % ([hex(x) for x in w], st.unpack(gp), st.unpack(ep)))
        # one answer per request: req_valid may stay high for a while after rsp_valid (tsr_xc_pkg)
        hold = g.r.choice([0, 0, 0, 1, 2, 12])
        for _ in range(1 + hold):
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            if int(dut.rsp_valid.value):
                bad.append("op %s: a second rsp_valid while req_valid stays high" % [hex(x) for x in w])
            if int(dut.rsp_code.value) or int(dut.rsp_info.value) or int(dut.rsp_ext.value) or int(dut.rsp_cmd.value):
                bad.append("op %s: rsp_* not 0 after rsp_valid" % [hex(x) for x in w])
        dut.req_valid.value = 0
        await RisingEdge(dut.clk)
        if len(bad) > 20:
            break
    sim.close()
    dut._log.info("%d ops, codes %s, latency %s", n_ops, dict(sorted(hist.items())), sorted(lat))
    return bad, hist, lat


@cocotb.test()
async def at_chk_random(dut):
    cfg, n = cfg_from_env()
    bad, hist, lat = await check_ops(dut, cfg, n, int(os.environ.get("TSR_SEED", "1")))
    for b in bad[:20]:
        dut._log.error(b)
    assert not bad, "%d mismatches" % len(bad)
    assert lat == {8}, "latency %s, expected 8" % sorted(lat)
    for c in (0, 1, 2, 4, 5, 6, 7, 13):
        assert hist.get(c, 0) > 0, "code %d never produced (%s)" % (c, hist)


def _run(name, cfg, n_ops=3000):
    c = dict(num_ch=cfg["num_ch"], t_max=cfg["t_max"], k_max=cfg["k_max"], spm_bytes=cfg["spm_bytes"],
             ch_bytes=cfg["ch_bytes"], addr_beat_w=cfg["addr_beat_w"], n_ops=n_ops)
    run("tsr_chk_at", __name__, sources=src(), name="atchk_" + name, extra_env={"TSR_ATCHK_CFG": json.dumps(c)},
        parameters=dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                        CH_BYTES="64'd%d" % cfg["ch_bytes"], ADDR_BEAT_W=cfg["addr_beat_w"]))


def test_at_chk_u200():
    _run("u200", S.config(num_ch=4, t_max=8), n_ops=5000)


def test_at_chk_odd():
    _run("odd", S.config(num_ch=3, t_max=3, k_max=4096, spm_bytes=256 << 10, ch_bytes=1 << 30, addr_beat_w=22))


def test_at_chk_wide():
    _run("wide", S.config(num_ch=16, t_max=8, k_max=65280, addr_beat_w=36))
    _run("one", S.config(num_ch=1, t_max=1, k_max=1024, spm_bytes=64 << 10, ch_bytes=1 << 20, addr_beat_w=40))

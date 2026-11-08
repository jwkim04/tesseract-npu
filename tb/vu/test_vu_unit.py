"""tb/vu unit — tsr_vu (rtl/core/tsr_vu.sv) on a real tsr_spm against the simulator's VU, bit for bit
(docs/design/02-vector-unit.md "VU" §10 item 4: "VU top with an SPM model: random streams of dependent and
independent ops, compared with the simulator on SPM bytes, s registers, completion order and error codes;
random ready on both bank ports; hazard cases (RAW, WAW and s RAW across accept-ahead)").

Every op passes the simulator's check first (codes are tb/vu test_vu_chk's subject); the RTL gets the words,
the simulator executes them in push order (vu_env.Oracle), the RTL with its accept-ahead overlap. After each
batch every line of every write extent and the s bank are compared; the done count must equal the ops pushed.
A contender client reads random lines (random grant delays on both bank ports) unless a test switches it off.
Tests (cocotb, one build per parameter set):
  vu_smoke          one op of each kind, no contention
  vu_vop_funcs      every VOP function x cmode x shape class (rows, m mod 16, strides incl. 0, in place)
  vu_red            SUM / SUMSQ / MAX, NONE / RCP / RMS, TO_S, m = 0 .. 400 incl. the -0 subtree cases
  vu_sop            the 15 SOP functions, LDS / STS
  vu_rope           NEOX / NORM, partial rotation, recurrence / FF / FS / ATAB, T up to 40 (position lines)
  vu_at             ASMAX, AMASK, ANORM shapes
  vu_stream         random mixed streams with RAW / WAW / s dependencies between neighbours
  vu_decode         the decode-record shapes of the W2b kernels (sw/ctrl tests: RMS_NORM, ROPE, GLU, ADD,
                    SCALE for Qwen2.5-0.5B / 32B and Gemma 4)
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import struct
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.runner import REPO, run       # noqa: E402

import vu_ops as V                         # noqa: E402
from vu_env import Oracle, Rtl, rnd_f32, rnd_line, M32   # noqa: E402

sys.path.insert(0, str(REPO / "tb" / "gdn"))
import gdn_data as GD                      # noqa: E402

CORE = REPO / "rtl" / "core"
VU_SOURCES = [CORE / s for s in (
    "tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_fp_pkg.sv", "tsr_att_pkg.sv", "tsr_xc_pkg.sv", "tsr_vu_pkg.sv")] + \
    [REPO / "rtl/tech/generic/tsr_sram_1r1w.sv"] + [CORE / s for s in (
        "tsr_fp_mul.sv", "tsr_fp_add.sv", "tsr_fp_div.sv", "tsr_spm.sv", "tsr_spm_cport.sv", "tsr_vu_dly.sv",
        "tsr_vu_exp.sv", "tsr_vu_rcp.sv", "tsr_vu_sqrt.sv", "tsr_vu_sincos.sv", "tsr_vu_lane.sv",
        "tsr_vu_lane_share.sv", "tsr_vu_lanes.sv",
        "tsr_vu_gen.sv",
        "tsr_vu_rd.sv", "tsr_vu_dq.sv", "tsr_vu_inj.sv", "tsr_vu_wr.sv", "tsr_vu_red.sv", "tsr_vu_angle.sv",
        "tsr_vu_topk.sv", "tsr_local_asmax.sv",
        "tsr_vu_gdn_ram.sv", "tsr_vu_gdn_fx2f.sv", "tsr_vu_gdn_dlane.sv", "tsr_vu_gdn_mac.sv", "tsr_vu_gdn_dot.sv",
        "tsr_vu_gdn_upd.sv", "tsr_vu_gdn_coef.sv", "tsr_vu_gdn.sv", "tsr_vu.sv")] + \
    [HERE / "hdl" / "tsr_vu_tb.sv"]


def vu_sources():
    """VU_SOURCES, with the mutated copies of $TSR_MUT_DIR in place of their originals (tb/vu/test_vu_mutants.py)."""
    d = os.environ.get("TSR_MUT_DIR")
    return [Path(d) / s.name if d and (Path(d) / s.name).exists() else s for s in VU_SOURCES]


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def cfg():
    c = json.loads(os.environ.get("TSR_VU_UNIT_CFG", "{}"))
    return dict(spm_bytes=c.get("spm_bytes", 262144), num_ch=c.get("num_ch", 4), n=c.get("n", 1.0),
                topk=c.get("topk", 1), dqrow=c.get("dqrow", 1),
                front_lanes=c.get("front_lanes", 16), exp_lanes=c.get("exp_lanes", 16),
                back_lanes=c.get("back_lanes", 8), lane_pipe=c.get("lane_pipe", 0), gdn=c.get("gdn", 0))


# ---------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------
class Bench:
    def __init__(self, dut, seed, noise=True):
        c = cfg()
        self.dut = dut
        self.spm_bytes = c["spm_bytes"]
        self.L = self.spm_bytes // 64
        self.scale = c["n"]
        self.gdn = c["gdn"]
        self.o = Oracle(self.spm_bytes, c["num_ch"], caps=0x21 | 0x8 | 0x40 | (0x100 if c["gdn"] else 0))
        self.rtl = Rtl(dut, self.L)
        self.r = random.Random(seed)
        self.noise = noise
        self.bad = []
        self.n_ops = 0
        self.cycle_records = []

    async def start(self):
        cocotb.start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        await self.rtl.reset(noise=self.noise, seed=self.r.getrandbits(32))
        cocotb.start_soon(self.rtl.monitor())
        img = [rnd_line(self.r) for _ in range(self.L)]
        for i, b in enumerate(img):
            self.o.set_line(i, b)
        await self.rtl.write_lines(list(enumerate(img)))
        if os.environ.get("TSR_VU_VERBOSE"):
            self.dut._log.info("SPM loaded (%d lines, %d cycles)", self.L, self.rtl.cycles)

    async def prewrite(self, items):
        """Set SPM lines on both sides (before a batch's ops are pushed)."""
        for (ln, b) in items:
            self.o.set_line(ln, b)
        await self.rtl.write_lines(items)

    async def batch(self, ops, what="", extra_wr=()):
        """Push ops (each must pass the simulator's check), wait for their done, compare. extra_wr: lines written
        that the extent list of the checks does not carry (a GDN op lists four of its six operands: not v, o)."""
        import time
        t0, c0 = time.time(), self.rtl.cycles
        wr = set(extra_wr)
        accepted = []
        for w in ops:
            w = [x & M32 for x in w]
            unit, code, info, exts = self.o.check(w)
            assert unit == 3 and code == 0, "bench op fails the check: code %d info %d %s" % (code, info, [hex(x) for x in w])
            for (k, write, ch, lo, hi) in exts:
                if k == 1 and write:
                    wr.update(range(lo, hi))
            self.o.exec(w)
            accepted.append(w)
        for w in accepted:
            await self.rtl.push(w)
        await self.rtl.wait_done(self.rtl.pushed)
        self.cycle_records.append(dict(workload=what, cycles=self.rtl.cycles-c0, ops=accepted))
        for _ in range(4):
            await cocotb.triggers.RisingEdge(self.dut.clk)
        assert self.rtl.done == self.rtl.pushed, "done %d, pushed %d" % (self.rtl.done, self.rtl.pushed)
        lines = sorted(wr)
        got = await self.rtl.read_lines(lines) if lines else {}
        nbad = 0
        for ln in lines:
            e = self.o.line(ln)
            if got[ln] != e:
                nbad += 1
                if len(self.bad) < 12:
                    gw = struct.unpack("<16I", got[ln])
                    ew = struct.unpack("<16I", e)
                    diff = [(i, hex(gw[i]), hex(ew[i])) for i in range(16) if gw[i] != ew[i]]
                    self.bad.append("%s line %d: words (i, rtl, sim) %s; ops %s" % (
                        what, ln, diff[:6], [[hex(x) for x in w] for w in accepted][:3]))
        sg, se = self.rtl.sreg(), list(self.o.sreg)
        if sg != se:
            self.bad.append("%s s bank: rtl %s sim %s" % (what, [hex(x) for x in sg], [hex(x) for x in se]))
        self.n_ops += len(accepted)
        if os.environ.get("TSR_VU_VERBOSE"):
            self.dut._log.info("batch %s: %d ops, %d cycles, %.1f s", what, len(accepted), self.rtl.cycles - c0,
                               time.time() - t0)
        return nbad

    def finish(self, name):
        report = os.environ.get("TSR_VU_CYCLE_REPORT")
        if report:
            digest = hashlib.sha256(bytes(self.o.spm)+struct.pack("<16I", *list(self.o.sreg))).hexdigest()
            Path(report).write_text(json.dumps(dict(config=cfg(), records=self.cycle_records,
                                                   output_digest=digest), indent=2)+"\n")
        self.o.close()
        log = self.dut._log
        log.info("%s: %d ops, %d cycles, hazard-wait cycles %d, SPM-wait cycles %d", name, self.n_ops,
                 self.rtl.cycles, self.rtl.hazard_cycles, self.rtl.wait_cycles)
        for b in self.bad[:12]:
            log.error(b)
        assert not self.bad, "%d mismatching lines / banks" % len(self.bad)

    # ---- region helpers ----
    def region(self, n, avoid=()):
        """A start line for n lines that overlaps none of the [lo, hi) in avoid."""
        n = max(1, n)
        for _ in range(1000):
            s = self.r.randrange(0, self.L - n + 1)
            if all(s + n <= lo or s >= hi for (lo, hi) in avoid):
                return s
        raise RuntimeError("no free region of %d lines" % n)

    def movi(self, i, v):
        return V.enc_sop(0, sd=i, imm=v)


def ext(n, s, ln):
    return 0 if n == 0 or ln == 0 else (n - 1) * s + ln


# ---------------------------------------------------------------------------------------------
# Op generators (legal ops: sources equal to the destination exactly, or disjoint from it, VU §4.5)
# ---------------------------------------------------------------------------------------------
def gen_vop(b: Bench, func=None, cmode=None, m=None, rows=None):
    r = b.r
    func = r.choice(list(V.VOP_FUNCS)) if func is None else func
    reads = V.VOP_FUNCS[func][1]
    if "c" in reads:
        cm = r.choice([1, 2]) if cmode is None else cmode
    else:
        cm = 0
    m = r.choice([1, 5, 15, 16, 17, 31, 32, 33, 64, 100, 128, 200, 256, 300]) if m is None else m
    rows = r.choice([1, 1, 2, 3, 5, 8, 17]) if rows is None else rows
    Lm = V.L(m)
    ds = Lm + r.choice([0, 0, 1, 3])
    D = (b.region(ext(rows, ds, Lm)), )
    d = D[0]
    dext = (d, d + ext(rows, ds, Lm))
    def src():
        if r.random() < 0.15:
            return d, ds                                       # in place
        st = r.choice([0, Lm, Lm, Lm + 2])
        return b.region(ext(rows, st, Lm), [dext]), st
    a, as_ = src() if "a" in reads else (0, 0)
    bb, bs = src() if "b" in reads else (0, 0)
    c = b.region(V.L(rows), [dext]) if cm == 2 else 0
    si = r.randrange(16) if cm == 1 else 0
    return V.enc_vop(func, d, a, bb, c, m, rows, ds if rows > 1 else r.choice([ds, 0, 7]), as_, bs, cm, si)


def gen_red(b: Bench, kind=None, post=None, to_s=None, m=None, rows=None):
    r = b.r
    kind = r.choice([0, 1, 2]) if kind is None else kind
    post = (0 if kind == 2 else r.choice([0, 1, 2])) if post is None else post
    to_s = (r.random() < 0.3) if to_s is None else to_s
    rows = 1 if to_s else (r.choice([1, 2, 3, 16, 17, 40]) if rows is None else rows)
    m = r.choice([0, 1, 2, 7, 8, 9, 15, 16, 17, 24, 64, 100, 127, 128, 129, 200, 256, 257, 400]) if m is None else m
    Lm = V.L(m)
    as_ = Lm + r.choice([0, 0, 2]) if rows > 1 else 0
    a = b.region(max(1, ext(rows, as_, Lm)))
    aext = (a, a + ext(rows, as_, Lm))
    d = 0 if to_s else b.region(V.L(rows), [aext])
    sk, se = (r.randrange(16), r.randrange(16)) if post == 2 else (0, 0)
    return V.enc_red(kind, d, a, m, rows, as_, int(to_s), r.randrange(16) if to_s else 0, post, sk, se)


def gen_sop(b: Bench, sub=None):
    r = b.r
    sub = r.choice(list(V.SOP_SUBS)) if sub is None else sub
    uses = V.SOP_USES[sub].split()
    sd = r.randrange(16) if "d" in uses else 0
    sa = r.randrange(16) if "a" in uses else 0
    sb = r.randrange(16) if "b" in uses else 0
    imm = rnd_f32(r) if "i" in uses else 0
    addr = r.randrange(b.L * 16) if "m" in uses else 0
    return V.enc_sop(sub, sd, sa, sb, imm, addr)


def s_init(b: Bench, vals=None):
    """SOP MOVI into all 16 s registers (typical constants and random patterns)."""
    r = b.r
    out = []
    for i in range(16):
        v = vals[i] if vals and i in vals else r.choice([f2u(1.0 / r.choice([64, 128, 896, 5120])), f2u(1e-6),
                                                         f2u(r.uniform(0.01, 4.0)), rnd_f32(r)])
        out.append(b.movi(i, v))
    return out


def gen_rope(b: Bench, mode=None, hd=None, nd=None, nh=None, T=None, form=None):
    """A ROPE op, its s-register setup and the pre-written pos / table lines."""
    r = b.r
    neox = (r.random() < 0.7) if mode is None else (mode == 1)
    hd = r.choice([32, 64, 128] if neox else [16, 32, 64, 128]) if hd is None else hd   # NEOX: n_dims >= 32
    if nd is None:
        q = 32 if neox else 16
        nd = r.choice([hd, hd, max(q, (hd // 2) // q * q)])
        nd = max(q, nd // q * q)
    nh = r.choice([1, 2, 3]) if nh is None else nh
    T = r.choice([1, 1, 2, 3, 17, 33]) if T is None else T
    form = r.choice(["rec", "rec", "ff", "fs", "ffs", "atab"]) if form is None else form
    ff, fs, atab = int(form in ("ff", "ffs")), int(form in ("fs", "ffs")), int(form == "atab")
    H = nh * hd // 16
    ss = r.choice([0, H, H + 1])
    dsr = r.choice([0, H, H + 2])
    sext = ext(T, ss or H, H)
    dext_n = ext(T, dsr or H, H)
    inplace = r.random() < 0.2 and ss == dsr
    d = b.region(dext_n)
    dext = (d, d + dext_n)
    src = d if inplace else b.region(sext, [dext])
    sx = (src, src + sext)
    pos = b.region(V.L(T), [dext, sx])
    px = (pos, pos + V.L(T))
    table = b.region((nd + 31) // 32, [dext, sx, px]) if (ff or atab) else 0
    pre = []
    positions = [r.choice([0, 1, 7, 100, 4095, 32767, 131071, r.randrange(1 << 17), -5, r.getrandbits(31)]) & M32
                 for _ in range(16 * V.L(T))]
    for i in range(V.L(T)):
        pre.append((pos + i, struct.pack("<16I", *positions[16 * i: 16 * i + 16])))
    if ff or atab:
        na = nd // 2
        if atab:
            vals = [f2u(r.uniform(1e-6, 1.5)) for _ in range(16 * ((nd + 31) // 32))]
        else:
            vals = [r.choice([f2u(1.0), f2u(1e30), f2u(r.uniform(0.5, 40.0))]) for _ in range(16 * ((nd + 31) // 32))]
        for i in range((nd + 31) // 32):
            pre.append((table + i, struct.pack("<16I", *vals[16 * i: 16 * i + 16])))
    base = r.choice([1e4, 1e6, 5e5])
    ts = f2u(math.pow(base, -2.0 / nd))
    sm, sts, sfs = 1, 2, 3
    sops = [b.movi(sm, f2u(r.choice([1.0, 1.0, 0.75, 1.1]))), b.movi(sts, ts), b.movi(sfs, f2u(r.choice([0.25, 1.0, 0.5])))]
    w = V.enc_rope(1 if neox else 0, d, src, pos, hd, nd, nh, T, table, ss, dsr, ff, fs, atab, sm,
                   0 if atab else sts, sfs if fs else 0)
    return pre, sops + [w]


# ---------------------------------------------------------------------------------------------
# GDN (VU §7.12): one head = SOP MOVI of alpha and beta, then the op; the tile, the vectors and o in disjoint regions
# ---------------------------------------------------------------------------------------------
def gen_gdn(b: Bench, kind=None, m=None, rows=None, inplace=None, avoid=(), sa=None, sb=None):
    """Returns (pre, ops, extra_wr, regions): pre-written lines (the head's data), the MOVI ops and the GDN op, the
    o lines (not in the check's extent list), and the regions used (for the next head to avoid)."""
    r = b.r
    kind = r.choice(GD.KINDS + ["model", "model"]) if kind is None else kind
    m = r.choice([64, 128, 128, 256, 96, 160]) if m is None else m
    rows = r.choice([1, 2, 7, 16, 17, 33, 64, 40]) if rows is None else rows
    nl = m // 16
    inplace = (r.random() < 0.4) if inplace is None else inplace
    k, q, v, S, alpha, beta = GD.head_data(r, kind, m, rows)
    te = rows * nl
    a = b.region(te, avoid)
    ax = (a, a + te)
    d = a if inplace else b.region(te, list(avoid) + [ax])
    tx = [ax] + ([] if inplace else [(d, d + te)])
    kl = b.region(nl, list(avoid) + tx)
    ql = b.region(nl, list(avoid) + tx + [(kl, kl + nl)])
    vl = b.region(V.L(rows), list(avoid) + tx + [(kl, kl + nl), (ql, ql + nl)])
    ol = b.region(V.L(rows), list(avoid) + tx + [(kl, kl + nl), (ql, ql + nl), (vl, vl + V.L(rows))])
    def lines(base, words):
        words = list(words) + [0] * (-len(words) % 16)
        return [(base + i, struct.pack("<16I", *words[16 * i:16 * i + 16])) for i in range(len(words) // 16)]
    pre = lines(a, S) + lines(kl, k) + lines(ql, q) + lines(vl, v)
    sa = r.randrange(16) if sa is None else sa
    sb = r.choice([x for x in range(16) if x != sa]) if sb is None else sb
    ops = [b.movi(sa, alpha), b.movi(sb, beta), V.enc_gdn(sa, sb, d, a, kl, ql, vl, ol, m, rows)]
    used = tx + [(kl, kl + nl), (ql, ql + nl), (vl, vl + V.L(rows)), (ol, ol + V.L(rows))]
    return pre, ops, list(range(ol, ol + V.L(rows))), used


def gen_asmax(b: Bench):
    r = b.r
    nl = r.choice([1, 2, 3, 5])
    R = r.choice([1, 2, 5, 16, 17, 20])
    st = r.choice([0, nl, nl + 1])
    e = ext(R, st or nl, nl)
    dst = b.region(e)
    dx = (dst, dst + e)
    src = dst if r.random() < 0.3 else b.region(e, [dx])
    mvec = b.region(V.L(R), [dx])
    scale = f2u(r.choice([1.0, 0.125, 0.08838835, r.uniform(0.01, 2.0)]))
    return V.enc_asmax(dst, src, mvec, nl, R, scale, st)


def gen_amask(b: Bench):
    r = b.r
    nl = r.choice([1, 2, 3, 4, 9])
    R = r.choice([1, 2, 5, 16, 17, 20])
    rpq = r.choice([1, 2, 4, 7])
    st = r.choice([0, nl, nl + 2])
    e = ext(R, st or nl, nl)
    dst = b.region(e)
    dx = (dst, dst + e)
    src = dst if r.random() < 0.3 else b.region(e, [dx])
    sx = (src, src + e)
    nq = (R + rpq - 1) // rpq
    ms = r.choice([nl, nl + 1, 0]) if nq > 1 else r.choice([0, nl])
    em = ext(nq, ms, nl)
    mask = b.region(em, [dx])
    mx = (mask, mask + em)
    mdst = b.region(V.L(R), [dx, sx, mx])
    return V.enc_amask(dst, src, mask, nl, R, rpq, ms, mdst, st)


def gen_anorm(b: Bench):
    r = b.r
    R = r.choice([1, 2, 5, 16, 17, 33])
    dcode = r.choice([0, 0, 1])
    rl = R * (64 << dcode) // 16
    out = b.region(rl)
    ox = (out, out + rl)
    orun = out if r.random() < 0.2 else b.region(rl, [ox])
    lrun = b.region(V.L(R), [ox])
    return V.enc_anorm(out, orun, lrun, R, dcode)


def gen_amerge(b: Bench, R=None, dcode=None, init=None):
    """AMERGE (AT §5.6) with disjoint Orun / parts / mk / st regions; C = the bench's NUM_CH."""
    r = b.r
    C = b.o.cfg["num_ch"]
    R = r.choice([1, 5, 16, 17, 40]) if R is None else R
    dcode = r.choice([0, 0, 1, 2] if R <= 17 else [0, 1]) if dcode is None else dcode
    init = r.randint(0, 1) if init is None else init
    rl = R * (64 << dcode) // 16
    cs = r.choice([0, rl, rl + 3])
    ep = C * (cs or rl) + C * V.L(R)
    orun = b.region(rl)
    ox = (orun, orun + rl)
    parts = b.region(ep, [ox])
    px = (parts, parts + ep)
    mk = b.region(V.L(R), [ox, px])
    mx = (mk, mk + V.L(R))
    st = b.region(2 * V.L(R), [ox, px, mx])
    scale = f2u(r.choice([1.0, 0.125, 0.08838835, r.uniform(0.05, 2.0)]))
    w = V.enc_amerge(orun, parts, mk, st, R, dcode, scale, cs, init)
    # running maxima / sums and chunk maxima: finite values (with M_FLOOR, -inf, NaN now and then)
    pre = []
    def vec(n, f):
        out = []
        for i in range(n):
            out.append((None, struct.pack("<16I", *[f() for _ in range(16)])))
        return out
    mval = lambda: r.choice([f2u(r.uniform(-8, 8)), f2u(r.uniform(-8, 8)), 0xF1800000, 0xFF800000, 0x7FC00000])  # noqa
    lval = lambda: r.choice([f2u(r.uniform(0, 30)), f2u(r.uniform(0, 30)), 0, 0x00800000])  # noqa
    for i, (_, bb) in enumerate(vec(V.L(R), mval)):
        pre.append((mk + i, bb))
    for i, (_, bb) in enumerate(vec(V.L(R), mval)):
        pre.append((st + i, bb))
    for i, (_, bb) in enumerate(vec(V.L(R), lval)):
        pre.append((st + V.L(R) + i, bb))
    lp = parts + C * (cs or rl)
    for c in range(C):
        for i, (_, bb) in enumerate(vec(V.L(R), lval)):
            pre.append((lp + c * V.L(R) + i, bb))
    return pre, w


def gen_topk(b: Bench, k=None, m=None, rows=None):
    """TOPK with tie-heavy rows: values drawn from a small pool (ties), NaNs, signed zeros, subnormals."""
    r = b.r
    k = r.randint(1, 16) if k is None else k
    m = r.choice([1, 3, 15, 16, 17, 40, 100, 300]) if m is None else m
    rows = r.choice([1, 2, 5]) if rows is None else rows
    Lm = V.L(m)
    as_ = r.choice([Lm, Lm + 1, 0]) if rows > 1 else 0
    kl = (k + 7) // 8
    ds = r.choice([0, kl, kl + 1])
    a = b.region(ext(rows, as_, Lm))
    d = b.region(ext(rows, ds or kl, kl), [(a, a + ext(rows, as_, Lm))])
    pool = [rnd_f32(r) for _ in range(6)] + [0, 0x80000000, 0x7FC00000, 0xFFC00000, 0x00000001, 0x80000001, 0x7F800000]
    pre = []
    for rr in range(rows):
        for j in range(Lm):
            vals = [r.choice(pool) if r.random() < 0.6 else rnd_f32(r) for _ in range(16)]
            pre.append((a + rr * as_ + j, struct.pack("<16I", *vals)))
    return dict(pre).items(), V.enc_topk(k, d, a, m, rows, ds, as_)


def row_beats(fmt, K):
    nb = K // (32 if fmt == 0 else 256)
    S, CB = (32, 17) if fmt == 0 else (4, 9)
    full, rem = divmod(nb, S)
    return full * CB + ((1 + ((rem + 1) // 2 if fmt == 0 else 2 * rem)) if rem else 0)


def rnd_fp16(r):
    """fp16 scale bits: mostly normal moderate values, sometimes zeros, subnormals, inf, NaN."""
    if r.random() < 0.15:
        return r.choice([0x0000, 0x8000, 0x0001, 0x03FF, 0x8200, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFBFF, 0x7BFF,
                         r.getrandbits(16)])
    return (r.getrandbits(1) << 15) | (r.randint(3, 20) << 10) | r.getrandbits(10)


def gen_dqrow(b: Bench, fmt=None, K=None, rows=None):
    """DQROW over random repacked rows (any bytes are a valid row; the meta lines get fp16 scales from
    rnd_fp16). Returns (prewrite items, op)."""
    r = b.r
    fmt = r.randint(0, 1) if fmt is None else fmt
    if K is None:
        K = 32 * r.choice([1, 2, 3, 31, 32, 33, 64, 65]) if fmt == 0 else 256 * r.choice([1, 2, 3, 4, 5, 9])
    rows = r.choice([1, 2, 3]) if rows is None else rows
    rb, lk = row_beats(fmt, K), K // 16
    ss = r.choice([0, rb, rb + 1]) if rows > 1 else r.choice([0, rb])
    ds = r.choice([0, lk, lk + 2]) if rows > 1 else 0
    ssp, dsp = ss or rb, ds or lk
    src = b.region(ext(rows, ssp, rb))
    d = b.region(ext(rows, dsp, lk), [(src, src + ext(rows, ssp, rb))])
    S, CB = (32, 17) if fmt == 0 else (4, 9)
    nb = K // (32 if fmt == 0 else 256)
    pre = {}
    for rr in range(rows):
        base = src + rr * ssp
        for i in range(rb):
            pre[base + i] = bytes(r.getrandbits(8) for _ in range(64))
        for c in range((nb + S - 1) // S):
            ml = bytearray(pre[base + c * CB])
            for k in range(min(S, nb - c * S)):
                if fmt == 0:
                    struct.pack_into("<H", ml, 2 * k, rnd_fp16(r))
                else:
                    struct.pack_into("<HH", ml, 16 * k, rnd_fp16(r), rnd_fp16(r))
            pre[base + c * CB] = bytes(ml)
    return list(pre.items()), V.enc_dqrow(fmt, d, src, K, rows, ss, ds)


def mask_prewrite(b: Bench, w):
    """AMASK: masks with -inf cells (a typical causal pattern) over the mask rows."""
    r = b.r
    nl = w[4] & 0xFFFF
    R, rpq = w[5] & 0xFFFF, w[5] >> 16
    nq = (R + rpq - 1) // rpq
    ms = w[6]
    out = []
    for q in range(nq):
        for j in range(nl):
            vals = [0xFF800000 if r.random() < 0.3 else r.choice([0, f2u(r.uniform(-3, 0))]) for _ in range(16)]
            out.append((w[3] + q * ms + j, struct.pack("<16I", *vals)))
    return dict(out).items()


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
def seed():
    return int(os.environ.get("TSR_SEED", "1"), 0)


def nscale(n):
    return max(1, int(n * cfg()["n"]))


@cocotb.test()
async def vu_smoke(dut):
    b = Bench(dut, seed(), noise=False)
    await b.start()
    await b.batch(s_init(b), "s init")
    await b.batch([V.enc_vop(0x00, 100, 200, 300, 0, 16, 1)], "VOP ADD 16")
    await b.batch([gen_vop(b, 0x09, m=40, rows=2)], "VOP COPY")
    await b.batch([gen_vop(b, 0x06, m=33, rows=2)], "VOP SILU")
    await b.batch([gen_red(b, 0, 0, False, 40, 2)], "RED SUM")
    await b.batch([gen_red(b, 1, 2, True, 896)], "RED SUMSQ RMS TO_S")
    await b.batch([gen_sop(b, 1), gen_sop(b, 6), gen_sop(b, 5), gen_sop(b, 10)], "SOP")
    pre, ops = gen_rope(b, 1, 64, 64, 2, 1, "rec")
    await b.prewrite(pre)
    await b.batch(ops, "ROPE NEOX")
    b.finish("vu_smoke")


@cocotb.test()
async def vu_vop_funcs(dut):
    b = Bench(dut, seed() + 1)
    await b.start()
    await b.batch(s_init(b), "s init")
    for f in V.VOP_FUNCS:
        ops = [gen_vop(b, f) for _ in range(nscale(6))]
        await b.batch(ops, "VOP %s" % V.VOP_FUNCS[f][0])
    b.finish("vu_vop_funcs")


@cocotb.test()
async def vu_red(dut):
    b = Bench(dut, seed() + 2)
    await b.start()
    await b.batch(s_init(b, {2: f2u(1.0 / 896), 3: f2u(1e-6)}), "s init")
    # -0 subtree cases (VU §6.2, §10 item 4): partials summing to -0 at m = 8k + r
    for m in (8, 9, 15, 16, 17, 24, 25, 33, 41, 120, 127, 128, 129, 136):
        a = b.region(V.L(m) * 2)
        vals = []
        for i in range(m):
            vals.append(0x80C00000 if (i // 8) % 2 == 0 and i % 2 == 0 else 0x00800000 if (i // 8) % 2 == 0 else
                        b.r.choice([0x80000000, 0x80C00000, 0x00800000]))
        vals += [0] * (16 * V.L(m) - m)
        await b.prewrite([(a + i, struct.pack("<16I", *vals[16 * i: 16 * i + 16])) for i in range(V.L(m))])
        d = b.region(1, [(a, a + V.L(m))])
        await b.batch([V.enc_red(0, d, a, m, 1), V.enc_red(1, 0, a, m, 1, 0, 1, 4, 2, 2, 3),
                       V.enc_red(2, 0, a, m, 1, 0, 1, 5)], "RED -0 m %d" % m)
    for _ in range(nscale(40)):
        await b.batch([gen_red(b) for _ in range(3)], "RED random")
    b.finish("vu_red")


@cocotb.test()
async def vu_sop(dut):
    b = Bench(dut, seed() + 3)
    await b.start()
    await b.batch(s_init(b), "s init")
    for _ in range(nscale(12)):
        await b.batch([gen_sop(b, s) for s in range(15)], "SOP all")
    for _ in range(nscale(10)):
        await b.batch([gen_sop(b) for _ in range(8)], "SOP random")
    b.finish("vu_sop")


@cocotb.test()
async def vu_rope(dut):
    b = Bench(dut, seed() + 4)
    await b.start()
    for (mode, hd, nd, nh, T, form) in ((1, 64, 64, 2, 1, "rec"), (0, 64, 64, 2, 1, "rec"), (1, 128, 64, 1, 2, "rec"),
                                        (0, 64, 16, 1, 1, "fs"), (1, 64, 64, 1, 3, "ff"), (1, 64, 64, 2, 2, "atab"),
                                        (1, 32, 32, 3, 33, "rec"), (0, 32, 32, 1, 20, "ffs")):
        pre, ops = gen_rope(b, mode, hd, nd, nh, T, form)
        await b.prewrite(pre)
        await b.batch(ops, "ROPE %s %d/%d x%d T%d %s" % ("NEOX" if mode else "NORM", hd, nd, nh, T, form))
    for _ in range(nscale(12)):
        pre, ops = gen_rope(b)
        await b.prewrite(pre)
        await b.batch(ops, "ROPE random")
    b.finish("vu_rope")


@cocotb.test()
async def vu_at(dut):
    b = Bench(dut, seed() + 5)
    await b.start()
    for _ in range(nscale(10)):
        await b.batch([gen_asmax(b) for _ in range(2)], "ASMAX")
    for _ in range(nscale(10)):
        w = gen_amask(b)
        await b.prewrite(mask_prewrite(b, w))
        await b.batch([w], "AMASK")
    for _ in range(nscale(10)):
        await b.batch([gen_anorm(b) for _ in range(2)], "ANORM")
    for (R, dcode, init) in ((1, 0, 1), (1, 0, 0), (5, 1, 0), (16, 0, 1), (17, 1, 0), (40, 1, 0), (20, 2, 1), (8, 3, 0)):
        pre, w = gen_amerge(b, R, dcode, init)
        await b.prewrite(pre)
        await b.batch([w], "AMERGE R %d dcode %d init %d" % (R, dcode, init))
    for _ in range(nscale(6)):
        pre, w = gen_amerge(b)
        await b.prewrite(pre)
        await b.batch([w, gen_asmax(b)], "AMERGE random")
    b.finish("vu_at")


@cocotb.test()
async def vu_topk(dut):
    """TOPK (VU §6.8): k = 1 .. 16, m < k (padding), ties, NaN, signed zeros, subnormals, ascending rows (the
    worst case of the running-threshold filter)."""
    if not cfg().get("topk", 1):
        return
    b = Bench(dut, seed() + 8)
    await b.start()
    for k in (1, 2, 7, 8, 9, 16):
        for m in (1, 5, 16, 33):
            pre, w = gen_topk(b, k, m, 2)
            await b.prewrite(pre)
            await b.batch([w], "TOPK k %d m %d" % (k, m))
    a = b.region(8)
    d = b.region(2, [(a, a + 8)])
    await b.prewrite([(a + i, struct.pack("<16I", *[f2u(float(16 * i + j)) for j in range(16)])) for i in range(8)])
    await b.batch([V.enc_topk(16, d, a, 128, 1)], "TOPK ascending")
    for _ in range(nscale(25)):
        pre, w = gen_topk(b)
        await b.prewrite(pre)
        await b.batch([w, gen_vop(b)], "TOPK random")
    b.finish("vu_topk")


@cocotb.test()
async def vu_dqrow(dut):
    """DQROW (VU §5.6, §6.8): Q8_0 and Q4_K rows of one block up to several chunks with a partial last chunk,
    strides, fp16 scale specials (zero, subnormal, inf, NaN), and VOPs reading the result (RAW)."""
    if not cfg().get("dqrow", 1):
        return
    b = Bench(dut, seed() + 9)
    await b.start()
    for fmt, Ks in ((0, (32, 64, 96, 1024, 1056, 2080)), (1, (256, 512, 768, 1024, 1280, 2304))):
        for K in Ks:
            pre, w = gen_dqrow(b, fmt, K, 2)
            await b.prewrite(pre)
            await b.batch([w], "DQROW fmt %d K %d" % (fmt, K))
    for _ in range(nscale(25)):
        pre, w = gen_dqrow(b)
        await b.prewrite(pre)
        ops = [w]
        if b.r.random() < 0.5:
            # VOP ADD reading the dequantised rows (same shape)
            K, rows, dsp = w[5], w[6] & 0xFFFF, (w[7] >> 16) or (w[5] // 16)
            o = b.region(ext(rows, V.L(K), V.L(K)), [(w[1], w[1] + ext(rows, dsp, K // 16)),
                                                    (w[2], w[2] + ext(rows, (w[7] & 0xFFFF) or row_beats(w[0] >> 8 & 15, K), row_beats(w[0] >> 8 & 15, K)))])
            ops.append(V.enc_vop(0x00, o, w[1], w[1], 0, K, rows, ds=V.L(K), as_=dsp, bs=dsp))
        ops.append(gen_vop(b))
        await b.batch(ops, "DQROW random")
    b.finish("vu_dqrow")


@cocotb.test()
async def vu_gdn(dut):
    """GDN heads of every data class, shape and layout (in place or not), one at a time and several back to back
    (the next head's setup under the rows of the previous one), against the simulator's tsr_ref_gdn on SPM bytes."""
    b = Bench(dut, 0x6D4, noise=True)
    if not b.gdn:
        b.o.close()
        return
    await b.start()
    for kind in GD.KINDS:
        for m, rows in ((64, 5), (128, 20), (256, 9)):
            pre, ops, ex, _ = gen_gdn(b, kind=kind, m=m, rows=rows)
            await b.prewrite(pre)
            await b.batch(ops, "GDN %s m %d rows %d" % (kind, m, rows), extra_wr=ex)
    for it in range(6):
        pre, ops, ex, used = [], [], [], []
        r_n = b.r.choice([2, 3, 4])
        for h in range(r_n):
            p, o_, e, u = gen_gdn(b, m=b.r.choice([64, 128, 96]), rows=b.r.choice([5, 16, 17, 24]), avoid=used)
            pre += p
            ops += o_
            ex += e
            used += u
        await b.prewrite(pre)
        await b.batch(ops, "GDN %d heads back to back" % r_n, extra_wr=ex)
    b.finish("vu_gdn")


@cocotb.test()
async def vu_gdn_mixed(dut):
    """GDN between other ops: VOP / RED on the head's results (RAW across the op), the same tile in two consecutive
    heads (the state of head 2 is the S' of head 1: RAW and WAW), and a VOP overwriting a vector that GDN read."""
    b = Bench(dut, 0x6D5, noise=True)
    if not b.gdn:
        b.o.close()
        return
    await b.start()
    r = b.r

    def lines(base, words):
        words = list(words) + [0] * (-len(words) % 16)
        return [(base + i, struct.pack("<16I", *words[16 * i:16 * i + 16])) for i in range(len(words) // 16)]

    for it in range(8):
        m, rows = r.choice([(64, 16), (128, 17), (128, 33), (256, 8)])
        nl, te, lr = m // 16, rows * (m // 16), V.L(rows)
        pre, ops, ex, used = gen_gdn(b, kind="model", m=m, rows=rows, inplace=True)
        a, o1 = ops[-1][2], ops[-1][6]                 # the tile (in place), the o vector of head 1
        seq = list(ops)
        dst = b.region(te, used)                       # RAW: a VOP COPY of the new state, one of o
        used.append((dst, dst + te))
        seq.append(V.enc_vop(0x09, dst, a, 0, 0, m, rows, nl, nl, 0))
        dst2 = b.region(lr, used)
        used.append((dst2, dst2 + lr))
        seq.append(V.enc_vop(0x09, dst2, o1, 0, 0, rows, 1))
        k2, q2, v2, _, al2, be2 = GD.head_data(r, "model", m, rows)    # head 2 on the same tile
        kl2 = b.region(nl, used); used.append((kl2, kl2 + nl))
        ql2 = b.region(nl, used); used.append((ql2, ql2 + nl))
        vl2 = b.region(lr, used); used.append((vl2, vl2 + lr))
        ol2 = b.region(lr, used); used.append((ol2, ol2 + lr))
        pre += lines(kl2, k2) + lines(ql2, q2) + lines(vl2, v2)
        seq += [b.movi(1, al2), b.movi(2, be2), V.enc_gdn(1, 2, a, a, kl2, ql2, vl2, ol2, m, rows)]
        seq.append(V.enc_vop(0x09, vl2, dst2, 0, 0, rows, 1))      # overwrites v of head 2 afterwards
        seq.append(V.enc_red(0, 0, ol2, rows, 1, 0, to_s=1, sd=7))   # a RED over the final o
        await b.prewrite(pre)
        await b.batch(seq, "GDN mixed %d (m %d rows %d)" % (it, m, rows), extra_wr=ex + list(range(ol2, ol2 + lr)))
    b.finish("vu_gdn_mixed")


@cocotb.test()
async def vu_stream(dut):
    """Random streams: each batch mixes op kinds; a third of the ops read the previous op's destination
    (RAW) or write it (WAW), s registers are written by SOP / RED TO_S and read by the next ops."""
    b = Bench(dut, seed() + 6)
    await b.start()
    await b.batch(s_init(b), "s init")
    r = b.r
    for it in range(nscale(30)):
        ops = []
        for _ in range(r.randint(2, 8)):
            k = r.choices(["vop", "red", "sop", "asmax", "anorm"], [6, 3, 2, 1, 1])[0]
            w = {"vop": gen_vop, "red": gen_red, "sop": gen_sop, "asmax": gen_asmax, "anorm": gen_anorm}[k](b)
            if ops and r.random() < 0.35 and (w[0] & 0xFF) == 0x30 and (ops[-1][0] & 0xFF) == 0x30:
                # chain: read the previous VOP's destination as a (same shape), or write it again
                p = ops[-1]
                if r.random() < 0.5 and (V.VOP_FUNCS[(w[0] >> 16) & 0xFF][1].startswith("a")):
                    w[2] = p[1]
                    w[5] = p[5]
                    w[6] = (w[6] & 0xFFFF0000) | (p[6] & 0xFFFF)
                    w[7] = (w[7] & 0xFFFF0000) | (p[6] >> 16)
                    if w[1] == w[2]:
                        continue
                    if not (b.o.check(w)[1] == 0):
                        continue
            if b.o.check(w)[1] != 0:
                continue
            ops.append(w)
        await b.batch(ops, "stream %d" % it)
    b.finish("vu_stream")


@cocotb.test()
async def vu_decode(dut):
    """Decode-record shapes of the W2b kernels (sw/ctrl/tests test_fw_normrope / test_fw_elem tables): RMS_NORM
    (RED.SUMSQ TO_S RMS + VOP.MULCB, VU §8.1) n 896 / 5120, Gemma per-head norms (16 / 8 rows of 256 with and
    without weight), ROPE Qwen 0.5B q 14 x 64 (theta 1e6), 32B k 8 x 128, Gemma SWA 256 (1e4) and global 512 with
    rope_freqs (1e30 factors), GLU SWIGLU 4864 / GEGLU16 15360, ADD 896 (residual, bias with stride 0), SCALE
    (MULC by s, by a per-row c vector)."""
    b = Bench(dut, seed() + 7)
    await b.start()
    r = b.r
    for n in (896, 5120):
        a = b.region(V.L(n))
        wt = b.region(V.L(n), [(a, a + V.L(n))])
        d = b.region(V.L(n), [(a, a + V.L(n)), (wt, wt + V.L(n))])
        await b.batch([b.movi(2, f2u(1.0 / n)), b.movi(3, f2u(1e-6)),
                       V.enc_red(1, 0, a, n, 1, 0, 1, 1, 2, 2, 3),
                       V.enc_vop(0x0C, d, a, wt, 0, n, 1, cmode=1, si=1)], "RMS_NORM %d" % n)
    for rows, weight in ((16, True), (8, False), (32, True)):
        n = 256
        a = b.region(rows * 16)
        vec = b.region(2, [(a, a + rows * 16)])
        wt = b.region(16, [(a, a + rows * 16), (vec, vec + 2)])
        ops = [b.movi(2, f2u(1.0 / n)), b.movi(3, f2u(1e-6)), V.enc_red(1, vec, a, n, rows, 16, 0, 0, 2, 2, 3)]
        if weight:
            ops.append(V.enc_vop(0x0C, a, a, wt, vec, n, rows, 16, 16, 0, cmode=2))
        else:
            ops.append(V.enc_vop(0x04, a, a, 0, vec, n, rows, 16, 16, 0, cmode=2))
        await b.batch(ops, "per-head norm %d x %d" % (rows, n))
    for (hd, nd, nh, base, form) in ((64, 64, 14, 1e6, "rec"), (128, 128, 8, 1e6, "rec"), (256, 256, 2, 1e4, "rec"),
                                     (512, 512, 1, 1e6, "ff")):
        pre, ops = gen_rope(b, 1, hd, nd, nh, 1, form)
        pos = ops[-1][3]
        pre = [p for p in pre if p[0] != pos] + [(pos, struct.pack("<16I", *([r.randrange(1 << 15)] + [0] * 15)))]
        if form == "ff":
            tbl = ops[-1][4]
            vals = [f2u(1.0) if i < 64 else f2u(1e30) for i in range(nd // 2)]
            pre = [p for p in pre if not (tbl <= p[0] < tbl + nd // 32)]
            pre += [(tbl + i, struct.pack("<16I", *vals[16 * i: 16 * i + 16])) for i in range(nd // 32)]
        ops[1] = b.movi(2, f2u(math.pow(base, -2.0 / nd)))
        ops[2] = b.movi(3, f2u(1.0))
        await b.prewrite(pre)
        await b.batch(ops, "ROPE decode %d/%d x%d" % (hd, nd, nh))
    for (f, n) in ((0x07, 4864), (0x18, 15360), (0x13, 1024)):
        a = b.region(V.L(n))
        bb = b.region(V.L(n), [(a, a + V.L(n))])
        await b.batch([V.enc_vop(f, a, a, bb, 0, n, 1)], "GLU %s %d" % (V.VOP_FUNCS[f][0], n))
    a = b.region(56 * 3)
    bias = b.region(56, [(a, a + 56 * 3)])
    await b.batch([V.enc_vop(0x00, a, a, bias, 0, 896, 3, 56, 56, 0)], "ADD bias stride 0")
    await b.batch([b.movi(7, f2u(30.0)), V.enc_vop(0x04, a, a, 0, 0, 896, 3, 56, 56, cmode=1, si=7)], "SCALE s")
    vec = b.region(1, [(a, a + 56 * 3)])
    await b.batch([V.enc_vop(0x04, a, a, 0, vec, 896, 3, 56, 56, cmode=2)], "SCALE per-row c")
    await b.batch([V.enc_vop(0x14, a, a, 0, 0, 896, 3, 56, 56)], "TANH softcap")
    b.finish("vu_decode")


# ---------------------------------------------------------------------------------------------
@cocotb.test()
async def vu_width_contexts(dut):
    """Distinct upper/lower histories, ring wrap, tails and ordered mixed exits."""
    b = Bench(dut, 0x1684, noise=True)
    await b.start()
    for m in (16*9+1, 16*17+9, 16*33+15, 16*65+7):
        lines = V.L(m)
        src = 100
        values = [f2u((-1.0 if (j//16+j%8)%2 else 1.0) *
                       2.0**((24 if j%16 < 8 else -10) + (j//128)%3))
                  for j in range(lines*16)]
        await b.prewrite([(src+i, struct.pack("<16I", *values[16*i:16*i+16])) for i in range(lines)])
        await b.batch([V.enc_red(kind, 10+kind, src, m) for kind in (0, 1, 2)], "logical contexts %d" % m)
    # First touch must initialize both contexts, even when tail masks one half.
    for m in (1, 7, 8, 9, 15, 16, 17, 129):
        await b.batch([V.enc_red(0, 12, 100, m), V.enc_red(2, 13, 100, m)], "fresh/tail %d" % m)
    for _ in range(4):
        w = gen_amask(b)
        await b.prewrite(mask_prewrite(b, w))
        await b.batch([w], "AMASK contexts/freshness")
    # Earliest scheduling must preserve back->front and front->back exit order.
    await b.batch([V.enc_vop(f, 700+i*64, 100, 200 if "b" in V.VOP_FUNCS[f][1] else 0, 0, 513, 1)
                   for i, f in enumerate((0x06, 0x00, 0x13, 0x18, 0x09, 0x06))], "mixed 2/4-part exits")
    for mode in (0, 1):
        pre, ops = gen_rope(b, mode, 128, 64, 3, 33, "rec")
        await b.prewrite(pre)
        await b.batch(ops, "ROPE four subslots, partial rotation, angle reuse %d" % mode)
    b.finish("vu_width_contexts")


@cocotb.test()
async def vu_width_reset(dut):
    """Reset while a NEOX pair or reduction row token is in flight."""
    from cocotb.triggers import FallingEdge
    b = Bench(dut, 0x8451, noise=False)
    await b.start()

    async def clean_reset():
        await b.rtl.reset(noise=False)
        b.rtl.done = b.rtl.pushed = 0
        for _ in range(160):
            await FallingEdge(dut.clk)
            assert not int(dut.probe_write_pending.value), "stale write after reset"
        assert b.rtl.done == 0, "stale completion after reset"
        b.o.close()
        b.o = Oracle(b.spm_bytes, cfg()["num_ch"], caps=0x21 | 0x8 | 0x40)
        await b.prewrite([(i, rnd_line(b.r)) for i in range(b.L)])

    pre, ops = gen_rope(b, 1, 64, 64, 2, 3, "rec")
    await b.prewrite(pre)
    await b.batch(ops[:-1], "rope setup before abort")
    await b.rtl.push(ops[-1])
    target = 2 if cfg().get("front_lanes", 16) == 8 else 1
    for _ in range(20000):
        await FallingEdge(dut.clk)
        if int(dut.probe_i_valid.value) and int(dut.probe_i_rope.value)==1 and int(dut.probe_i_part.value)==target:
            break
    else:
        assert False, "did not reach occupied NEOX subslot"
    await clean_reset()
    pre, ops = gen_rope(b, 1, 64, 64, 2, 3, "rec")
    await b.prewrite(pre)
    await b.batch(ops, "NEOX after active reset")
    await b.rtl.push(V.enc_red(0, 8, 100, 16*25+9))
    for _ in range(20000):
        await FallingEdge(dut.clk)
        if int(dut.probe_row_token.value):
            break
    else:
        assert False, "did not reach queued reduction row token"
    await clean_reset()
    await b.batch([V.enc_red(0, 8, 100, 16*25+9), V.enc_red(2, 9, 100, 129)], "RED after queued-token reset")
    b.finish("vu_width_reset")


@cocotb.test()
async def vu_width_cycles(dut):
    """Matched no-contention operation latency; excludes readback and setup."""
    b = Bench(dut, 0xC1C1, noise=False)
    await b.start()
    for name, f, n in (("ADD896", 0x00, 896), ("COPY4096", 0x09, 4096),
                       ("EXPK1024", 0x08, 1024), ("SILU4864", 0x06, 4864)):
        await b.batch([V.enc_vop(f, 1000, 100, 500 if "b" in V.VOP_FUNCS[f][1] else 0, 0, n, 1)], name)
    await b.batch([V.enc_red(1, 10, 100, 5120)], "SUMSQ5120")
    for mode in (0, 1):
        pre, ops = gen_rope(b, mode, 128, 64, 2, 2, "rec")
        await b.prewrite(pre)
        await b.batch(ops[:-1], "rope setup %d" % mode)
        await b.batch(ops[-1:], "NORM128" if mode==0 else "NEOX128")
    b.finish("vu_width_cycles")


# pytest
# ---------------------------------------------------------------------------------------------
def _run(name, spm_bytes=262144, rd_lat=4, exp_lanes=16, back_lanes=8, num_ch=4, n=1.0, testcase=None, topk=1,
         dqrow=1, lane_pipe=0, front_lanes=16, cycle_report=None, gdn=0):
    c = dict(spm_bytes=spm_bytes, num_ch=num_ch, n=n, topk=topk, dqrow=dqrow, front_lanes=front_lanes, exp_lanes=exp_lanes, back_lanes=back_lanes, lane_pipe=lane_pipe, gdn=gdn)
    run("tsr_vu_tb", __name__, sources=vu_sources(), name="vu_unit_" + name, testcase=testcase,
        extra_env={"TSR_VU_UNIT_CFG": json.dumps(c), **({"TSR_VU_CYCLE_REPORT": str(cycle_report)} if cycle_report else {})},
        parameters=dict(SPM_BYTES=spm_bytes, SPM_RD_LAT=rd_lat, NUM_CH=num_ch, FRONT_LANES=front_lanes, EXP_LANES=exp_lanes,
                        BACK_LANES=back_lanes, LANE_PIPE=lane_pipe, HAS_TOPK=topk, HAS_DQROW=dqrow, HAS_GDN=gdn))


def test_vu_unit_u200():
    """U200 lanes (16 EXPK, 8 back), SPM_RD_LAT 4 (the U200 bitstreams), 256 KiB SPM, 4 channels."""
    _run("u200")


def test_vu_unit_knobs():
    """The shrink knobs VU_EXP_LANES 8 and VU_BACK_LANES 4 (VU §7.10): same results, SPM_RD_LAT 2."""
    _run("knobs", rd_lat=2, exp_lanes=8, back_lanes=4, n=0.5)


def test_vu_unit_lane_pipe():
    """The lane array behind VU_LANE_PIPE register stages on each side (CC §14.1 remedy 3, the k4 bitstream's 2):
    exits and partial drain reads 2p cycles later, results unchanged (U200 lanes, SPM_RD_LAT 4; and the shrink knobs
    with 3 stages, SPM_RD_LAT 2: every test of the unit)."""
    _run("pipe2", lane_pipe=2)
    _run("pipe3_knobs", rd_lat=2, exp_lanes=8, back_lanes=4, n=0.5, lane_pipe=3)


def test_vu_unit_ch16():
    """16 channels (AMERGE's channel tree over 16 blocks, local memory 1024 lines), 3 channels (an odd tree), the
    units without TOPK / DQROW (HAS_TOPK = HAS_DQROW = 0: the M6 bitstream; those tests skip), SPM_RD_LAT 1."""
    _run("ch16", spm_bytes=1 << 20, num_ch=16, rd_lat=1, n=0.5, testcase="vu_smoke|vu_at|vu_stream")
    _run("ch3", num_ch=3, n=0.5, testcase="vu_smoke|vu_at|vu_topk|vu_dqrow", topk=0, dqrow=0)


def test_vu_unit_half_width():
    """Eight physical fronts retain all sixteen logical reduction histories."""
    _run("half8_pipe2", front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=2)


def test_vu_half_cycle_comparison():
    out = REPO / "build/analysis/k4_vu_half_20261104"
    out.mkdir(parents=True, exist_ok=True)
    full, half = out/"cycles_front16.json", out/"cycles_front8.json"
    _run("cycles_full16", lane_pipe=2, testcase="vu_width_cycles", cycle_report=full)
    _run("cycles_half8", lane_pipe=2, front_lanes=8, exp_lanes=8, back_lanes=4,
         testcase="vu_width_cycles", cycle_report=half)
    a, b = json.loads(full.read_text()), json.loads(half.read_text())
    assert a["output_digest"] == b["output_digest"]
    records = []
    for x, y in zip(a["records"], b["records"]):
        assert x["workload"] == y["workload"] and x["ops"] == y["ops"]
        records.append(dict(workload=x["workload"], front16_cycles=x["cycles"],
                            front8_cycles=y["cycles"], ratio=y["cycles"]/x["cycles"]))
    (out/"cycles.json").write_text(json.dumps(records, indent=2)+"\n")


def test_vu_unit_half_transport():
    """The same eight physical lanes and two contexts with zero/three pipe stages."""
    _run("half8_wire", front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=0)
    _run("half8_pipe3", front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=3)


def test_vu_unit_gdn_k4():
    """HAS_GDN = 1 on the k4 shape (8 front lanes, two lane-pipe stages): the GDN tests and every existing test of
    the unit, which must pass unchanged with the engine present."""
    _run("gdn_half8_pipe2", front_lanes=8, exp_lanes=8, back_lanes=4, lane_pipe=2, gdn=1)


def test_vu_unit_gdn_front16():
    """HAS_GDN = 1 with 16 front lanes (the unit at 16 elements per cycle), no lane pipe and two."""
    _run("gdn_front16", gdn=1, testcase="vu_gdn|vu_gdn_mixed|vu_smoke|vu_stream")
    _run("gdn_front16_pipe2", gdn=1, lane_pipe=2, testcase="vu_gdn|vu_gdn_mixed")

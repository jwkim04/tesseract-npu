"""tb/at test_at_mvm — ATTQK 0x21 and ATTPV 0x22 (AT §5.2, §5.3, §6.2-§6.5, §8.1, §8.2) on the MVM unit's
attention modes, end to end with the AT checker class, against the simulator's AT unit (tsr_at.c ->
tsr_ref_attqk / tsr_ref_attpv) for the same SPM and DRAM contents.

DUT: tb/at/hdl/tsr_tb_at_mvm.sv = tsr_chk_at + tsr_mvm (HAS_ATT = 1: clusters with the SEG trees, the score
assembler, the PV unit, the gather's attention path) + one tsr_mem_arb per channel (weight-stream source);
SPM port on a Python model (at_tb.SpmPort, random ready), memory ports on common.mem_model.MemModel.
Per op: the words go through tsr_chk_at (code / info == tsr_sim_op's), a passing op runs on the MVM unit with
the checker's payload, then on the simulator; after `done`:
  - the whole SPM image equals the simulator's (scores, -inf bounds, MAXOUT words, channel partials, row sums,
    and every byte the op must not touch: rows of other passes, words past R);
  - the SPM writes stay inside the op's write extents; the weight-stream beats per channel equal the
    simulator's CH_RD_BEATS (whole local groups / slices of the channel);
  - the unit's PERF levels: perf_att_qk / perf_att_pv only while it holds the op.
Stimulus: the target models' attention shapes (decode and prefill tiles, AT §6.6 pass rule):
  Qwen2.5-0.5B D 64 (E 128, G 7), Qwen2.5-32B D 128 (E 1024, G 5), Gemma 4 12B SWA D 256 (E 2048, G 2) and its
  global layers D 512 (E 512, G 16: two head subsets), with CAUSAL / SLIDE bounds of prefill tiles, wrapped-SWA
  sub-ranges (c_lo / c_hi inside a group, u0 anywhere), MAXOUT, LSUM, strides and cstride; random shapes (D
  64..512, G 1..64, GS / T_q / hg0, n_groups 1..80, u0, odd channel counts); q8_0 K / V with fp16 scales over a
  wide range plus zero / subnormal / inf / NaN, Q / p with -0, subnormals, inf, NaN (block NaN rules, skipped
  cells, empty rows). Plain MVMs in between (the weight path of the same clusters).
at_fa (test_at_fa_t8 / _t4): the AT §6.6 flash-attention tile sequence op by op -- QACT / KVST / ASMAX / AMERGE /
  ANORM on the simulator (its SPM and planes mirrored into the RTL's models), every ATTQK / ATTPV on the RTL and
  compared with the simulator on the whole SPM, the chunk plan from tsr_fa_plan, the layer output against
  tsr_ref_fa bit for bit: decode (G 4 D 64, Qwen2.5-32B, Gemma 4 global G 16 in two head subsets, two passes over a
  wrapped SWA ring) and prefill tiles (T 2 causal, Gemma 4 SWA T 4 with window 512).
Knobs (debug / measurement): TSR_SEED; TSR_ATMVM_DIRECTED (JSON list of att() shapes) with TSR_ATMVM_ONLY=1;
  TSR_ATMVM_CFGX (JSON overrides, e.g. {"p_wr": 1.0, "mem": {"p_rd_rsp_valid": 1.0}}); TSR_ATMVM_VERBOSE (per-op
  cycles against the simulator's cycle model); TSR_ATMVM_DUMP (pickle of a failing op's state).
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
from pathlib import Path

import ctypes

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge

try:
    import pytest
except ImportError:      # inside the simulator
    pytest = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import isa, tables as T                       # noqa: E402
from common.mem_model import MemModel                     # noqa: E402
from common.runner import run, rtl_sources, mutated_sources, RTL   # noqa: E402

import at_isa as A                                        # noqa: E402
import at_tb as B                                         # noqa: E402

M32 = 0xFFFFFFFF


def cfg_env():
    return json.loads(os.environ.get("TSR_ATMVM_CFG", "{}"))


def f2b(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h2b(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


class Data:
    """Random operand contents with special values at controlled rates."""

    def __init__(self, r: random.Random, p_sp: float):
        self.r, self.p_sp = r, p_sp

    def fp16(self):
        r = self.r
        if r.random() < self.p_sp:
            return r.choice([0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x0001, 0x03FF, 0x7BFF, 0xFBFF, r.getrandbits(16)])
        e = r.randint(-9, 3)
        return h2b(r.choice((-1.0, 1.0)) * r.uniform(1.0, 2.0) * 2.0 ** e)

    def f32_scale(self):
        r = self.r
        if r.random() < self.p_sp:
            return r.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF,
                             0x00800000, 0x7F7FFFFF, r.getrandbits(32)])
        return f2b(r.choice((-1.0, 1.0)) * r.uniform(1.0, 2.0) * 2.0 ** r.randint(-12, 2))

    def p(self):
        r = self.r
        x = r.random()
        if x < self.p_sp:
            return r.choice([0x80000000, 0x7F800000, 0x7FC00000, 0x00000001, 0x007FFFFF, 0xFF800000,
                             f2b(-0.5), f2b(3.0e38), f2b(1.0e-38), r.getrandbits(32)])
        if x < 0.25:
            return 0
        return f2b(math.exp(-r.uniform(0.0, 12.0)))

    def plane(self, beats: int) -> bytes:
        """Q8_0 chunk stream: a meta beat (32 fp16) every 17 beats from the plane base, else quants."""
        r = self.r
        out = bytearray(r.randbytes(64 * beats))
        for k in range(0, beats, 17):
            out[64 * k:64 * k + 64] = b"".join(struct.pack("<H", self.fp16()) for _ in range(32))
        return bytes(out)

    def qa32(self, K: int) -> bytes:
        """A QA(A32, K) image: QS K int8, D K/32 f32 (each region padded to whole lines)."""
        r = self.r
        qs = r.randbytes(K) + bytes(64 * ((K + 63) // 64) - K)
        nb = K // 32
        d = b"".join(struct.pack("<I", self.f32_scale()) for _ in range(nb))
        dl = (nb + 15) // 16
        return qs + d + bytes(64 * dl - 4 * nb)


class Bench:
    def __init__(self, dut, cfg, seed):
        self.dut, self.cfg = dut, cfg
        self.r = random.Random(seed)
        self.C = cfg["num_ch"]
        self.tmax = cfg["t_max"]
        self.kmax = cfg["k_max"]
        self.spm_bytes = cfg["spm_bytes"]
        self.lines = self.spm_bytes // 64
        self.limit = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.sim = B.SimAt(num_ch=self.C, t_max=self.tmax, k_max=self.kmax, spm_bytes=self.spm_bytes,
                           ch_bytes=cfg["ch_bytes"], addr_beat_w=cfg["addr_beat_w"], caps=T.CAPS_VU | T.CAPS_ATT)
        self.spm = bytearray(self.r.randbytes(self.spm_bytes))
        self.sim.spm_write(0, self.spm)
        self.port = B.SpmPort(dut, self.spm, prefix="spm_", rd_lat=cfg["spm_rd_lat"], rnd=random.Random(seed ^ 0x50414952) if cfg.get("isolate_spm_rng") else self.r,
                              p_rd=cfg.get("p_rd", 0.85), p_wr=cfg.get("p_wr", 0.85))
        self.mem = MemModel(dut, num_ch=self.C, addr_w=cfg["addr_beat_w"], rd_outstanding=cfg["rd_out"],
                            limit_beats=self.limit, seed=seed, rd_lat=tuple(cfg.get("rd_lat", (1, 40))),
                            **cfg.get("mem", {}))                 # measurement: p_rd_rsp_valid 1, p_stall 0, ...
        self.d = Data(self.r, cfg.get("p_sp", 0.03))
        self.stats = dict(qk=0, pv=0, fail=0, mvm=0, beats=0, cycles=0)
        self.perf_bad = []
        self.stall_cycles = 0               # cycles with perf_att_stall high (the XSLR_REG test needs some)

    # ---- memory images ----
    def load_plane(self, B0, beats):
        self.last_plane = (B0, beats)
        for c in range(self.C):
            data = self.d.plane(beats)
            self.mem.load(c, 64 * B0, data)
            self.sim.dram_write(c, 64 * B0, data)

    def spm_put(self, line, data: bytes):
        self.spm[64 * line:64 * line + len(data)] = data
        self.sim.spm_write(64 * line, data)

    async def perf_mon(self):
        while True:
            await FallingEdge(self.dut.clk)
            busy = int(self.dut.perf_mvm_busy.value)
            qk, pv = int(self.dut.perf_att_qk.value), int(self.dut.perf_att_pv.value)
            if (qk or pv) and not busy:
                self.perf_bad.append("perf_att_* outside perf_mvm_busy")
            if qk and pv:
                self.perf_bad.append("perf_att_qk and perf_att_pv together")
            self.stall_cycles += int(self.dut.perf_att_stall.value)

    async def mvm_cmd(self, xc, xcmd=0, cmd=0, timeout=2000000):
        dut = self.dut
        await FallingEdge(dut.clk)
        while not int(dut.cmd_ready.value):
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
        dut.cmd.value = cmd
        dut.cmd_xc.value = xc
        dut.xcmd.value = xcmd
        dut.cmd_valid.value = 1
        await RisingEdge(dut.clk)
        dut.cmd_valid.value = 0
        for n in range(timeout):
            await FallingEdge(dut.clk)
            if int(dut.done.value):
                return n
            await RisingEdge(dut.clk)
        raise TimeoutError("MVM unit did not finish")

    def alloc(self, sizes):
        """Disjoint SPM regions of the given line counts, in random order with random gaps."""
        total = sum(sizes)
        if total + 8 * len(sizes) > self.lines:
            return None
        order = list(range(len(sizes)))
        self.r.shuffle(order)
        slack = self.lines - total
        out = [0] * len(sizes)
        pos = 0
        for i in order:
            gap = self.r.randint(0, min(64, slack // (2 * len(sizes)) + 1))
            pos += gap
            slack -= gap
            out[i] = pos
            pos += sizes[i]
        return out

    async def att(self, *, pv, E, D, G, GS, tq, hg0, u0, n, flags=0, c_lo=0, c_hi=None, rs=0, cs=0,
                  LG=None, tag=""):
        r = self.r
        dcode = {64: 0, 128: 1, 256: 2, 512: 3}[D]
        e64, nseg = E // 64, E // D
        n_head = G * nseg
        R = tq * n_head
        maxrow = (tq - 1) * n_head + (nseg - 1) * G + hg0 + GS - 1
        lmax = (u0 + n - 1) // self.C
        if LG is None:
            LG = lmax + 1 + r.choice([0, 0, 1, 3])
        GB = 17 * e64
        if pv:
            plane = ((e64 - 1) * LG + lmax + 1) * 17
        else:
            plane = (lmax + 1) * GB
        B0 = r.randint(0, max(0, min(self.limit, 1 << 22) - plane))
        if pv:
            rl = R * D // 16
            csr = cs if cs else rl
            ed = (self.C - 1) * csr + rl
            if flags & 1:
                ed = max(ed, self.C * csr + self.C * ((R + 15) // 16))
            pst = rs if rs else n
            ep = maxrow * pst + n
            regs = self.alloc([ed, ep])
            if regs is None:
                return
            dst, x = regs
            w = A.enc_attpv(dst=dst, p=x, base=B0, e64=e64, dcode=dcode, G=G, hg0=hg0, GS=GS, tq=tq, u0=u0, n=n,
                            LG=LG, rs=rs, cstride=cs, lsum=flags & 1)
            self.load_plane(B0, ((e64 - 1) * LG + lmax + 1) * 17 if e64 > 0 else 17)
            # p rows: every row r at x + r pstride, n lines (pass rows matter; fill the region)
            self.spm_put(x, b"".join(struct.pack("<I", self.d.p()) for _ in range(16 * ep)))
        else:
            if c_hi is None:
                c_hi = 16 * n
            rst = rs if rs else n
            ed = maxrow * rst + n
            K = G * E
            qa = A.qa32_lines(K)
            ex = tq * qa
            em = (maxrow + 16) // 16
            regs = self.alloc([ed, ex, em])
            if regs is None:
                return
            dst, x, mdst = regs
            w = A.enc_attqk(dst=dst, x=x, base=B0, e64=e64, dcode=dcode, G=G, hg0=hg0, GS=GS, tq=tq, u0=u0, n=n,
                            c_lo=c_lo, c_hi=c_hi, rs=rs, mdst=mdst if flags & 4 else 0,
                            causal=flags & 1, slide=(flags >> 1) & 1, maxout=(flags >> 2) & 1)
            self.load_plane(B0, (lmax + 1) * GB)
            for q in range(tq):
                self.spm_put(x + q * qa, self.d.qa32(K))
        spm_before = bytes(self.spm)
        if os.environ.get("TSR_ATMVM_VERBOSE"):
            self.dut._log.info("%s %s E %d D %d G %d GS %d tq %d hg0 %d u0 %d n %d fl %d c %d..%s rs %d cs %d",
                               "ATTPV" if pv else "ATTQK", tag, E, D, G, GS, tq, hg0, u0, n, flags, c_lo, c_hi, rs, cs)
        code, info, _, xcmd = await B.run_check(self.dut, w)
        self.mem.clear_tracking()
        self.port.clear_log()
        scode, sinfo = self.sim.op_with_ch_beats(w, self.C)
        assert (code, info) == (scode[0], sinfo), "%s %s: checker (%d, %#x), simulator (%d, %#x)" % (
            "ATTPV" if pv else "ATTQK", tag, code, info, scode[0], sinfo)
        if code:
            self.stats["fail"] += 1
            return
        sim_beats = scode[1]
        cyc = await self.mvm_cmd(1, xcmd)
        for _ in range(4):
            await RisingEdge(self.dut.clk)
        self.mem.check(idle=True)
        assert not self.port.errors, self.port.errors[:3]
        assert not self.perf_bad, self.perf_bad[:3]
        exp = self.sim.spm_read(0, self.spm_bytes)
        got = bytes(self.spm)
        name = "ATTPV" if pv else "ATTQK"
        if got != exp and os.environ.get("TSR_ATMVM_DUMP"):
            import pickle
            B0, nb = self.last_plane
            planes = [self.sim.dram_read(c, 64 * B0, 64 * nb) for c in range(self.C)]
            with open(os.environ["TSR_ATMVM_DUMP"], "wb") as fh:
                pickle.dump(dict(w=w, spm=spm_before, exp=exp, got=got, B0=B0, planes=planes, C=self.C), fh)
        if got != exp:
            dw = [(i // 16, i % 16) for i in range(self.spm_bytes // 4) if got[4 * i:4 * i + 4] != exp[4 * i:4 * i + 4]]
            rst = rs if rs else n
            byrow = {}
            for (ln, wd) in dw:
                off = ln - w[1]
                if off < 0:
                    key = ("other", ln)
                elif pv:
                    rl_ = R * D // 16
                    csr_ = cs if cs else rl_
                    if off < self.C * csr_:            # (ch, row, lane)
                        key = ("ch", off // csr_, (off % csr_) // (D // 16), (off % csr_) % (D // 16) * 16 + wd)
                    else:
                        o2 = off - self.C * csr_
                        key = ("lsum ch", o2 // ((R + 15) // 16), 16 * (o2 % ((R + 15) // 16)) + wd)
                else:
                    key = (off // rst, wd)
                byrow[key] = byrow.get(key, 0) + 1
            self.dut._log.warning("%s %s: %d words differ, first (line, word) %s; (row, word): count %s", name, tag,
                                  len(dw), dw[:8], sorted(byrow.items(), key=str)[:40])
            for ln in range(self.lines):
                if got[64 * ln:64 * ln + 64] != exp[64 * ln:64 * ln + 64]:
                    gw = struct.unpack("<16I", got[64 * ln:64 * ln + 64])
                    xw = struct.unpack("<16I", exp[64 * ln:64 * ln + 64])
                    raise AssertionError("%s %s %s: SPM line %d differs (dst %d x %d):\n RTL %s\n sim %s" % (
                        name, tag, [hex(v) for v in w], ln, w[1], w[2], [hex(v) for v in gw], [hex(v) for v in xw]))
        # writes inside the write extents
        if pv:
            lo_hi = [(w[1], w[1] + ed)]
        else:
            lo_hi = [(w[1], w[1] + ed)] + ([(w[7], w[7] + em)] if flags & 4 else [])
        for (ln, _) in self.port.writes:
            assert any(lo <= ln < hi for lo, hi in lo_hi), "%s %s: SPM write of line %d outside %s" % (name, tag, ln, lo_hi)
        got_b = [x[2] for x in self.mem.src_beats]
        assert got_b == sim_beats, "%s %s: weight-stream beats per channel %s, simulator %s" % (name, tag, got_b, sim_beats)
        self.stats["pv" if pv else "qk"] += 1
        self.stats["beats"] += sum(got_b)
        self.stats["cycles"] += cyc
        prows = (E // D) * GS * tq
        model = 350 + max(prows * n, max(got_b))          # the simulator's cycle model (tsr_at.c att_cycles)
        (self.dut._log.info if os.environ.get("TSR_ATMVM_VERBOSE") else self.dut._log.debug)(
            "%s %s: %d cycles (model %d: AT_FILL 350 + max(pass rows x n %d, beats %d)), beats %s", name, tag, cyc,
            model, prows * n, max(got_b), got_b)

    # ---- shape generators ----
    def pass_rule(self, G):
        GS = min(G, self.tmax)
        return GS, max(1, self.tmax // GS)

    async def model_shapes(self):
        """The target models' layers (AT §6.6 pass rule), decode and prefill tiles."""
        r = self.r
        shapes = [("qwen05b", 128, 64, 7), ("qwen32b", 1024, 128, 5), ("gemma_swa", 2048, 256, 2),
                  ("gemma_glob", 512, 512, 16)]
        for name, E, D, G in shapes:
            if G * E > self.kmax:
                continue
            GS, TQ = self.pass_rule(G)
            for hg0 in range(0, G, GS):
                gs = min(GS, G - hg0)
                for tq in sorted({1, TQ}):
                    if tq * gs > self.tmax:
                        continue
                    n = r.choice([1, 2, 3, 5, 8, 13]) * max(1, self.C) + r.randint(0, self.C)
                    u0 = r.randint(0, 40)
                    tag = "%s hg0 %d tq %d n %d u0 %d" % (name, hg0, tq, n, u0)
                    # decode: full range; prefill: a CAUSAL tile at the end, SLIDE window
                    if tq == 1:
                        await self.att(pv=0, E=E, D=D, G=G, GS=gs, tq=1, hg0=hg0, u0=u0, n=n, flags=4, tag=tag)
                    else:
                        ch = 16 * n - r.randint(0, 15)
                        await self.att(pv=0, E=E, D=D, G=G, GS=gs, tq=tq, hg0=hg0, u0=u0, n=n, flags=4 | 1,
                                       c_hi=ch, tag=tag + " causal")
                        await self.att(pv=0, E=E, D=D, G=G, GS=gs, tq=tq, hg0=hg0, u0=u0, n=n, flags=4 | 2 | 1,
                                       c_lo=r.randint(0, 20), c_hi=ch, tag=tag + " slide")
                    await self.att(pv=1, E=E, D=D, G=G, GS=gs, tq=tq, hg0=hg0, u0=u0, n=n, flags=1, tag=tag)
            # wrapped SWA ring: two sub-ranges with bounds inside groups (decode)
            gs = min(GS, G)
            n = 3 * self.C + 1
            await self.att(pv=0, E=E, D=D, G=G, GS=gs, tq=1, hg0=0, u0=7, n=n, flags=4, c_lo=5, c_hi=16 * n,
                           tag=name + " ring a")
            await self.att(pv=0, E=E, D=D, G=G, GS=gs, tq=1, hg0=0, u0=0, n=2, flags=4, c_lo=0, c_hi=9,
                           tag=name + " ring b")

    async def random_op(self):
        r = self.r
        pv = r.randint(0, 1)
        D = r.choice([64, 64, 128, 128, 256, 512])
        nseg = r.choice([1, 1, 2, 3, 4, 8])
        E = D * nseg
        if E > 32704:
            E = D
        G = r.choice([1, 2, 3, 4, 5, 7, 8, 16, 33, 64])
        while G * E > self.kmax and G > 1:
            G = max(1, G // 2)
        if G * E > self.kmax:
            E, D = 64, 64
            G = 1
            nseg = 1
        GS = r.randint(1, min(8, G, self.tmax))
        tq = r.randint(1, max(1, self.tmax // GS))
        if nseg * GS * tq > 128:
            tq = 1
            GS = max(1, min(GS, 128 // nseg))
        hg0 = r.randint(0, G - GS)
        n = r.choice([1, 2, 3, 4, 5, 7, 9, 16, 33, r.randint(1, 80)])
        u0 = r.choice([0, 1, 2, 3, r.randint(0, 200)])
        if pv:
            # PCAP bound: (E/D) 16 ceil(n/C) <= QA_LINES(A256, K_MAX) 16
            qa256 = self.kmax // 64 + (4 * (self.kmax // 256) + 63) // 64 + (2 * (self.kmax // 16) + 63) // 64
            while nseg * 16 * ((n + self.C - 1) // self.C) > 16 * qa256 and n > 1:
                n //= 2
            await self.att(pv=1, E=E, D=D, G=G, GS=GS, tq=tq, hg0=hg0, u0=u0, n=n, flags=r.randint(0, 1),
                           rs=r.choice([0, 0, n, n + r.randint(1, 5)]),
                           cs=r.choice([0, 0, R_cs(tq, G, nseg, D) + r.randint(0, 9)]), tag="random")
        else:
            # prefill tiles (T_q > 1): CAUSAL / SLIDE in most ops (their bounds differ per query)
            flags = (r.choice([1, 1, 3, 3, 0, 2]) | r.choice([0, 4])) if tq > 1 else (r.choice([0, 4]) | r.choice([0, 1, 2, 3]))
            c_hi = r.choice([16 * n, 16 * n, r.randint(0, 16 * n)])
            c_lo = r.choice([0, 0, r.randint(0, c_hi)])
            if r.random() < 0.15:                   # an empty range: every row fully masked (MAXOUT = M_FLOOR)
                c_lo = c_hi
            await self.att(pv=0, E=E, D=D, G=G, GS=GS, tq=tq, hg0=hg0, u0=u0, n=n, flags=flags, c_lo=c_lo, c_hi=c_hi,
                           rs=r.choice([0, 0, n, n + r.randint(1, 5)]), tag="random")

    async def plain_mvm(self):
        """A plain Q8_0 MVM through the same clusters (the weight path in the HAS_ATT build)."""
        r = self.r
        K = r.choice([32, 64, 512, 1024, 1056])
        if K > self.kmax:
            K = 32
        N = r.randint(1, 40)
        t = r.randint(1, self.tmax)
        rb = isa.row_beats(0, K)
        R = (N + self.C - 1) // self.C
        qa = isa.qa_lines(0, K)
        yl = (N + 15) // 16
        regs = self.alloc([(t - 1) * yl + yl, t * qa])
        if regs is None:
            return
        dst, x = regs
        woff = r.randint(0, 1 << 16)
        for c in range(self.C):
            data = self.d.plane(R * rb)
            self.mem.load(c, 64 * woff, data)
            self.sim.dram_write(c, 64 * woff, data)
        for k in range(t):
            self.spm_put(x + k * qa, self.d.qa32(K))
        w = [isa.OP_MVM, dst, x, woff, N, K | (t << 16), 0, 0]
        code, _ = self.sim.op(w)
        assert code == 0
        cmd = isa.MVM_CMD_T.pack(fmt=0, acc=0, dst_line=dst, x_line=x, w_off=woff, n=N, k=K, t=t, dst_stride=yl,
                                 y_lines=yl, rows_per_ch=R, row_beats=rb, qa_lines=qa)
        if os.environ.get("TSR_ATMVM_VERBOSE"):
            self.dut._log.info("MVM K %d N %d t %d", K, N, t)
        cycles = await self.mvm_cmd(0, cmd=cmd)
        if os.environ.get("TSR_ATMVM_VERBOSE"):
            self.dut._log.info("MVM K %d N %d t %d: %d cycles", K, N, t, cycles)
        for _ in range(4):
            await RisingEdge(self.dut.clk)
        assert bytes(self.spm) == self.sim.spm_read(0, self.spm_bytes), "plain MVM differs from the simulator"
        self.stats["mvm"] += 1


def R_cs(tq, G, nseg, D):
    return tq * G * nseg * D // 16


async def mvm_run(dut):
    cfg = cfg_env()
    b = Bench(dut, cfg, int(os.environ.get("TSR_SEED", "1")))
    for n in ("chk_req_valid", "chk_req_insn", "cmd_valid", "cmd", "cmd_xc", "xcmd"):
        getattr(dut, n).value = 0
    b.port.start()
    b.mem.start()
    await B.clock_reset(dut)
    for _ in range(cfg.get("rst_local", 0)):
        await RisingEdge(dut.clk)
    cocotb.start_soon(b.perf_mon())
    for d in json.loads(os.environ.get("TSR_ATMVM_DIRECTED", "[]")) + cfg.get("directed", []):
        await b.att(tag="directed", **d)
    if cfg.get("models", True):
        await b.model_shapes()
    for i in range(cfg.get("n_ops", 40)):
        await b.random_op()
        if b.r.random() < 0.1:
            await b.plain_mvm()
    if cfg.get("bc_all_valids"):
        await b.plain_mvm()
        assert int(dut.o_bc_seen.value) == 7, "line / MVM command / ATT command broadcast coverage missing"
    b.port.stop()
    b.mem.stop()
    b.sim.close()
    dut._log.info("ATT: %s", b.stats)
    dut._log.info("perf_att_stall high in %d cycles", b.stall_cycles)
    if cfg.get("require_stall"):
        # tsr_mvm checks att_stall_q against the clusters' strobes delayed by 3 cycles (XSLR_REG); that needs strobes
        assert b.stall_cycles > 0, "no attention stall in the run: the XSLR_REG stage was not exercised"


@cocotb.test()
async def at_mvm(dut):
    await mvm_run(dut)

# ---------------------------------------------------------------------------------------------------
# The AT §6.6 flash-attention tile sequence, op by op: ATTQK / ATTPV on the RTL (each compared with the
# simulator on the whole SPM), QACT / KVST / ASMAX / AMERGE / ANORM on the simulator (its SPM and planes
# mirrored into the RTL's models), the chunk plan from tsr_fa_plan, the result against tsr_ref_fa.
# ---------------------------------------------------------------------------------------------------
class _AttCfg(ctypes.Structure):
    _fields_ = [("C", ctypes.c_uint32), ("t_max", ctypes.c_uint32), ("pcap", ctypes.c_uint32), ("k_max", ctypes.c_uint32)]


class _Msum(ctypes.Structure):
    _fields_ = [("n_q", ctypes.c_uint32), ("n_kv", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("rows", ctypes.c_void_p)]


class _FaChunk(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("u0", "n_groups", "c_lo", "c_hi", "flags", "general")]


class _FaStats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("tiles", "passes", "chunks", "general_chunks", "ops_attqk", "ops_attpv",
                                               "ops_asmax", "ops_amask", "ops_amerge", "ops_anorm", "ops_qact")]


class _FaArgs(ctypes.Structure):
    _fields_ = [("T", ctypes.c_uint32), ("D", ctypes.c_uint32), ("G", ctypes.c_uint32), ("scale", ctypes.c_float),
                ("q", ctypes.c_void_p), ("E", ctypes.c_uint32), ("LG", ctypes.c_uint32), ("kplane", ctypes.c_void_p),
                ("vplane", ctypes.c_void_p), ("msum", ctypes.c_void_p), ("mask_f32", ctypes.c_void_p),
                ("new_runs", ctypes.c_void_p), ("n_new", ctypes.c_uint32), ("out", ctypes.c_void_p),
                ("stats", ctypes.c_void_p)]


def _fa_lib(L):
    if not hasattr(L, "_fa_bound"):
        L.tsr_fa_plan.restype = ctypes.c_long
        L.tsr_fa_plan.argtypes = [ctypes.POINTER(_AttCfg), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_Msum),
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                                  ctypes.POINTER(_FaChunk), ctypes.c_size_t]
        L.tsr_msum_view.restype = ctypes.c_int
        L.tsr_msum_view.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_Msum)]
        L.tsr_fa_pass_rule.restype = None
        L.tsr_fa_pass_rule.argtypes = [ctypes.POINTER(_AttCfg), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                       ctypes.POINTER(ctypes.c_uint32)]
        L.tsr_kv_lg_min.restype = ctypes.c_uint32
        L.tsr_kv_lg_min.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        L.tsr_ref_fa.restype = ctypes.c_int
        L.tsr_ref_fa.argtypes = [ctypes.POINTER(_AttCfg), ctypes.POINTER(_FaArgs)]
        L._fa_bound = True
    return L


MSUM_MAGIC = 0x4D55534D
TT = T                       # the tables module (T is a tile's token count below)


class FaBench(Bench):
    """One FA layer call (AT §6.6): a query tile of T tokens over n_kv cells, the tile's new K / V rows
    stored first (KVST), runs[q] the visible cells [lo, hi) of query row q (one or two runs)."""

    def sim_only(self, w):
        code, info = self.sim.op(w)
        assert code == 0, "simulator op %#x: code %d info %#x" % (w[0] & 0xFF, code, info)
        self.spm[:] = self.sim.spm_read(0, self.spm_bytes)

    async def rtl_att(self, w, tag):
        name = "ATTPV" if (w[0] & 0xFF) == A.OP_ATTPV else "ATTQK"
        code, info, _, xcmd = await B.run_check(self.dut, w)
        self.mem.clear_tracking()
        self.port.clear_log()
        (scode, sbeats), sinfo = self.sim.op_with_ch_beats(w, self.C)
        assert (code, info) == (scode, sinfo) and code == 0, "%s %s: checker (%d, %#x), simulator (%d, %#x)" % (
            name, tag, code, info, scode, sinfo)
        cyc = await self.mvm_cmd(1, xcmd)
        for _ in range(4):
            await RisingEdge(self.dut.clk)
        self.mem.check(idle=True)
        exp = self.sim.spm_read(0, self.spm_bytes)
        if bytes(self.spm) != exp:
            dw = [(i // 16, i % 16) for i in range(self.spm_bytes // 4) if self.spm[4 * i:4 * i + 4] != exp[4 * i:4 * i + 4]]
            raise AssertionError("%s %s %s: %d SPM words differ from the simulator, first %s" % (
                name, tag, [hex(v) for v in w], len(dw), dw[:8]))
        got_b = [x[2] for x in self.mem.src_beats]
        assert got_b == sbeats, "%s %s: weight-stream beats %s, simulator %s" % (name, tag, got_b, sbeats)
        self.stats["pv" if name == "ATTPV" else "qk"] += 1
        self.stats["cycles"] += cyc
        self.stats["beats"] += sum(got_b)
        return cyc

    def fa_plane(self, B0, beats):
        """Q8_0 chunks with moderate scales (meta every 17 beats from B0), in the simulator and the RTL's memory."""
        r = self.r
        for c in range(self.C):
            out = bytearray(r.randbytes(64 * beats))
            for k in range(0, beats, 17):
                out[64 * k:64 * k + 64] = b"".join(
                    struct.pack("<H", h2b(r.choice((-1.0, 1.0)) * r.uniform(0.5, 1.0) * 2.0 ** -7)) for _ in range(32))
            self.sim.dram_write(c, 64 * B0, out)
            self.mem.load(c, 64 * B0, bytes(out))

    def mirror_plane(self, B0, beats):
        for c in range(self.C):
            self.mem.load(c, 64 * B0, self.sim.dram_read(c, 64 * B0, 64 * beats))

    def f32_rows(self, line, vals):
        data = struct.pack("<%df" % len(vals), *vals)
        self.spm_put(line, data)

    async def fa_layer(self, name, *, E, D, G, T, n_kv, new0, runs, scale):
        L = _fa_lib(self.sim.L)
        r = self.r
        C = self.C
        e64, nseg, dc = E // 64, E // D, {64: 0, 128: 1, 256: 2, 512: 3}[D]
        n_head = G * nseg
        K = n_head * D
        R = T * n_head
        Rl = (R + 15) // 16
        cs = R * D // 16
        LG = L.tsr_kv_lg_min(n_kv, C)
        acfg = _AttCfg(C, self.tmax, isa.qa_lines(TT.QA_A256, self.kmax) * 16, self.kmax)
        GS, TQ = ctypes.c_uint32(), ctypes.c_uint32()
        L.tsr_fa_pass_rule(ctypes.byref(acfg), G, ctypes.byref(GS), ctypes.byref(TQ))
        GS, TQ = GS.value, TQ.value
        assert T <= TQ, "%s: the tile's T %d > T_q %d of the pass rule" % (name, T, TQ)
        BK, BV = 4096, 200000
        pbeats = 17 * e64 * LG
        self.fa_plane(BK, pbeats)
        self.fa_plane(BV, pbeats)
        # f32 Q (T tokens of n_head D), K and V rows of the T new cells
        QF = 64
        KF = QF + T * K // 16
        VF = KF + T * E // 16
        QA = VF + T * E // 16
        qa_k = isa.qa_lines(TT.QA_A32, K)
        qa_e = isa.qa_lines(TT.QA_A32, E)
        KQA = QA + T * qa_k
        VQA = KQA + T * qa_e
        nxt = VQA + T * qa_e
        q_host = [r.gauss(0.0, 1.5) for _ in range(T * K)]
        self.f32_rows(QF, q_host)
        self.f32_rows(KF, [r.gauss(0.0, 0.7) for _ in range(T * E)])
        self.f32_rows(VF, [r.gauss(0.0, 0.7) for _ in range(T * E)])
        q_host = struct.unpack("<%df" % (T * K), struct.pack("<%df" % (T * K), *q_host))   # f32-rounded
        # mask summary and chunk plan
        words = [MSUM_MAGIC, T, n_kv, 0, 0, 0, 0, 0]
        for q in range(T):
            rr = list(runs[q]) + [0, 0, 0, 0]
            words += rr[:4]
        warr = (ctypes.c_uint32 * len(words))(*words)
        ms = _Msum()
        assert L.tsr_msum_view(warr, len(words), ctypes.byref(ms)) == 0, name
        new_runs = (ctypes.c_uint32 * 2)(new0, new0 + T)
        chunks = (_FaChunk * 64)()
        nch = L.tsr_fa_plan(ctypes.byref(acfg), E, D, ctypes.byref(ms), 0, T, new_runs, 1, chunks, 64)
        assert 1 <= nch <= 64, "%s: %d chunks" % (name, nch)
        # the kernel's op stream
        def qact(dst, src, k, t):
            return [TT.OP_QACT | (TT.QA_A32 << 8), dst, src, k, t, 0, 0, 0]
        self.sim_only(qact(KQA, KF, E, T))
        self.sim_only(qact(VQA, VF, E, T))
        self.sim_only(A.enc_kvst(0, T, KQA, BK, e64, LG, new0))
        self.sim_only(A.enc_kvst(1, T, VQA, BV, e64, LG, new0))
        self.mirror_plane(BK, pbeats)
        self.mirror_plane(BV, pbeats)
        self.sim_only(qact(QA, QF, K, T))
        ORUN = nxt
        ST = ORUN + cs
        OUT = ST + 2 * Rl
        nxt = OUT + cs
        cyc = dict(qk=0, pv=0)
        for k in range(nch):
            c = chunks[k]
            assert not c.general, "%s: FAST chunks only" % name
            n = c.n_groups
            S = nxt
            MK = S + R * n
            P = MK + Rl
            nxt = P + C * cs + C * Rl
            assert nxt < self.lines, "%s: SPM layout (%d lines)" % (name, nxt)
            for hg0 in range(0, G, GS):
                gs = min(GS, G - hg0)
                w = A.enc_attqk(dst=S, x=QA, base=BK, e64=e64, dcode=dc, G=G, hg0=hg0, GS=gs, tq=T, u0=c.u0, n=n,
                                c_lo=c.c_lo, c_hi=c.c_hi, mdst=MK, causal=c.flags & 1, slide=(c.flags >> 1) & 1, maxout=1)
                cyc["qk"] += await self.rtl_att(w, "%s chunk %d hg0 %d" % (name, k, hg0))
            self.sim_only([TT.OP_ASMAX, S, S, MK, n, R, scale, 0])
            for hg0 in range(0, G, GS):
                gs = min(GS, G - hg0)
                w = A.enc_attpv(dst=P, p=S, base=BV, e64=e64, dcode=dc, G=G, hg0=hg0, GS=gs, tq=T, u0=c.u0, n=n, LG=LG,
                                cstride=cs, lsum=1)
                cyc["pv"] += await self.rtl_att(w, "%s chunk %d hg0 %d" % (name, k, hg0))
            self.sim_only([TT.OP_AMERGE | ((1 if k == 0 else 0) << 12), ORUN, P, MK, ST, R | (dc << 16), scale, cs])
        self.sim_only([TT.OP_ANORM, OUT, ORUN, ST + Rl, R | (dc << 16), 0, 0, 0])
        # tsr_ref_fa on the simulator's planes (with the new cells) and the same Q
        kp = (ctypes.c_void_p * C)(*[self.sim.dram_ptr(ch) + 64 * BK for ch in range(C)])
        vp = (ctypes.c_void_p * C)(*[self.sim.dram_ptr(ch) + 64 * BV for ch in range(C)])
        qbuf = (ctypes.c_float * (T * K))(*q_host)
        obuf = (ctypes.c_float * (R * D))()
        st = _FaStats()
        fa = _FaArgs(T, D, G, struct.unpack("<f", struct.pack("<I", scale))[0], ctypes.cast(qbuf, ctypes.c_void_p), E, LG,
                     ctypes.cast(kp, ctypes.c_void_p), ctypes.cast(vp, ctypes.c_void_p),
                     ctypes.cast(ctypes.pointer(ms), ctypes.c_void_p), None, ctypes.cast(new_runs, ctypes.c_void_p), 1,
                     ctypes.cast(obuf, ctypes.c_void_p), ctypes.cast(ctypes.pointer(st), ctypes.c_void_p))
        rc = L.tsr_ref_fa(ctypes.byref(acfg), ctypes.byref(fa))
        assert rc == 0, "%s: tsr_ref_fa %d" % (name, rc)
        assert st.chunks == nch, "%s: tsr_ref_fa %d chunks, plan %d" % (name, st.chunks, nch)
        ref = bytes(obuf)
        if self.tmax == 4:
            # Slot capacity changes head/query partitioning, not the KV format
            # or arithmetic. Check the same planes/Q/bounds against T8 as well.
            peer_cfg = _AttCfg(C, 8, acfg.pcap, self.kmax)
            peer_out = (ctypes.c_float * (R * D))()
            peer_stats = _FaStats()
            peer = _FaArgs.from_buffer_copy(bytes(fa))
            peer.out = ctypes.cast(peer_out, ctypes.c_void_p)
            peer.stats = ctypes.cast(ctypes.pointer(peer_stats), ctypes.c_void_p)
            assert L.tsr_ref_fa(ctypes.byref(peer_cfg), ctypes.byref(peer)) == 0, name
            assert bytes(peer_out) == ref, "%s: T4/T8 oracle output differs" % name
            self.dut._log.info("%s: T4/T8 oracle bit-identical (%d words; passes %d/%d)",
                               name, R * D, st.passes, peer_stats.passes)
        got = bytes(self.spm[64 * OUT:64 * OUT + 4 * R * D])
        bad = sum(1 for i in range(R * D) if got[4 * i:4 * i + 4] != ref[4 * i:4 * i + 4])
        fin = all(math.isfinite(v) for v in struct.unpack("<%df" % (R * D), got))
        assert bad == 0, "%s: FA output differs from tsr_ref_fa in %d of %d words" % (name, bad, R * D)
        assert fin, "%s: FA output not finite" % name
        self.dut._log.info("FA %s: T %d R %d, %d chunks, ATTQK %d / ATTPV %d cycles, output = tsr_ref_fa (%d words)",
                           name, T, R, nch, cyc["qk"], cyc["pv"], R * D)
        self.stats["fa"] = self.stats.get("fa", 0) + 1


FA_CASES = {
    8: [
        # one pass (GS = G = 4), 38 groups, the new cell in the first sub-range
        dict(name="decode G4 D64", E=128, D=64, G=4, T=1, n_kv=600, new0=100, runs=[(0, 600)], scale=0x3E000000),
        # Qwen2.5-32B: 8 KV heads of D 128, G 5
        dict(name="decode Qwen2.5-32B", E=1024, D=128, G=5, T=1, n_kv=300, new0=299, runs=[(0, 300)], scale=0x3DB504F3),
        # Gemma 4 global layer: one KV head of D 512, G 16 > T_MAX: two head subsets
        dict(name="decode Gemma 4 global", E=512, D=512, G=16, T=1, n_kv=520, new0=519, runs=[(0, 520)],
             scale=0x3F800000),
        # prefill tile T 2 (GS = G = 4): causal rows, the tile's two cells written in this launch
        dict(name="prefill T2 causal", E=256, D=128, G=4, T=2, n_kv=300, new0=298, runs=[(0, 299), (0, 300)],
             scale=0x3DB504F3),
        # Gemma 4 SWA layer prefill tile T 4 (GS = G = 2): window 512 (CAUSAL + SLIDE bounds)
        dict(name="prefill T4 SWA 512", E=2048, D=256, G=2, T=4, n_kv=700, new0=696,
             runs=[(696 + q - 511, 697 + q) for q in range(4)], scale=0x3D800000),
    ],
    4: [
        dict(name="prefill T4 causal", E=128, D=64, G=1, T=4, n_kv=100, new0=96,
             runs=[(0, 97 + q) for q in range(4)], scale=0x3E000000),
        dict(name="prefill T2 heads", E=256, D=128, G=2, T=2, n_kv=130, new0=128,
             runs=[(0, 129), (0, 130)], scale=0x3DB504F3),
        # G = 8 > T_MAX = 4: two head subsets; wrapped SWA ring: two runs, two FAST chunks with bounds
        dict(name="decode 2 passes, SWA ring", E=128, D=128, G=8, T=1, n_kv=400, new0=149, runs=[(0, 150, 300, 400)],
             scale=0x3DB504F3),
    ],
}


async def fa_run(dut):
    cfg = cfg_env()
    b = FaBench(dut, cfg, int(os.environ.get("TSR_SEED", "1")))
    for n in ("chk_req_valid", "chk_req_insn", "cmd_valid", "cmd", "cmd_xc", "xcmd"):
        getattr(dut, n).value = 0
    b.port.start()
    b.mem.start()
    await B.clock_reset(dut)
    cocotb.start_soon(b.perf_mon())
    for case in FA_CASES[cfg["t_max"]]:
        case = dict(case)
        await b.fa_layer(case.pop("name"), **case)
    assert not b.perf_bad, b.perf_bad[:3]
    b.port.stop()
    b.mem.stop()
    b.sim.close()
    dut._log.info("FA: %s", b.stats)


@cocotb.test()
async def at_fa(dut):
    await fa_run(dut)



def _run(name, cfg, testcase="at_mvm", rand_reset=None, mutant=None):
    if os.environ.get("TSR_ATMVM_ONLY"):          # debug: only the directed ops of TSR_ATMVM_DIRECTED
        cfg = dict(cfg, models=False, n_ops=0)
    cfg.update(json.loads(os.environ.get("TSR_ATMVM_CFGX", "{}")))   # measurement: e.g. rd_lat, p_rd, p_wr
    params = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                  CH_BYTES="64'd%d" % cfg["ch_bytes"], SPM_RD_LAT=cfg["spm_rd_lat"], ADDR_BEAT_W=cfg["addr_beat_w"],
                  RD_OUTSTANDING=cfg["rd_out"], BC_PIPE=cfg.get("bc_pipe", 1), RES_PIPE=cfg.get("res_pipe", 1),
                  ACT_RD_LAT=cfg.get("act_rd_lat", 2), MAC_LAT=cfg.get("mac_lat", 3), MUL_LAT=cfg.get("mul_lat", 4),
                  ADD_LAT=cfg.get("add_lat", 5), WS_FIFO_BEATS=cfg.get("ws_fifo", 512),
                  RST_LOCAL=cfg.get("rst_local", 0), GATHER_PAIR=cfg.get("gather_pair", 0), GATHER_FLITS=cfg.get("gather_flits", 1),
                  XSLR_REG=cfg.get("xslr_reg", 0), FAST_CLK=cfg.get("fast_clk", 0))
    srcs = rtl_sources(RTL / ("filelist_xilinx.f" if cfg.get("tech") == "xilinx" else "filelist.f"))
    if mutant:
        srcs = mutated_sources(srcs, *mutant)
    run("tsr_tb_at_mvm", __name__, parameters=params, name="at_mvm_" + name + ("_mut" if mutant else ""),
        testcase=testcase,
        sources=srcs + [HERE.parent / "mvm" / "hdl" / "tsr_mvm_bc_observer.sv", HERE / "hdl" / "tsr_tb_at_mvm.sv"],
        rand_reset=rand_reset,
        extra_env={"TSR_ATMVM_CFG": json.dumps(cfg), "TSR_ATMVM_DIRECTED": os.environ.get("TSR_ATMVM_DIRECTED", "[]")})


def test_at_mvm_u200():
    """U200-like: 4 channels, T_MAX 4, K_MAX 32768, SPM_RD_LAT 4, pipes 2, MAC_LAT 9 (the bitstream's)."""
    _run("u200", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                      addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                      rst_local=2, gather_pair=1, n_ops=40))


def test_at_mvm_u200_flits2():
    """The U200 shape with the paired two-flit gather (MVM_GATHER_FLITS 2, the full K4 image's). The attention results
    and the SPM image are the simulator's."""
    _run("u200_flits2", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                             addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                             rst_local=2, gather_pair=1, gather_flits=2, n_ops=40))


def test_at_mvm_u200_xilinx():
    """The full K4 shape (two-flit gather, XSLR_REG) with the Xilinx technology wrappers of the card image
    (rtl/filelist_xilinx.f: inferred BRAM / URAM / LUTRAM, DSP48E2-packed MAC lanes) instead of the generic ones."""
    _run("u200_xilinx", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                             addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                             rst_local=2, gather_pair=1, gather_flits=2, xslr_reg=1, n_ops=40, tech="xilinx"))


def test_at_mvm_fast_clk_xilinx():
    """FAST_CLK 1 with the Xilinx technology wrappers of the card image: the DSP lanes take pv_en through register copies
    (PVR). tsr_mac_lane asserts that pv_en does not change within a cycle of an operand."""
    _run("u200_fast_xilinx", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                                  addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9, rst_local=2, gather_pair=1,
                                  gather_flits=2, xslr_reg=1, fast_clk=1, n_ops=40, tech="xilinx"))


# tsr_mvm_pv before the fix: the output stage was reset to `ng == 0`, a value that is 1 after configuration
PV_FIX = "ob_zero_q  <= rst_n & (ng == 13'd0);"
PV_OLD = "ob_zero_q  <= (ng == 13'd0);"


def _zero_state(mutant=None, fast_clk=0):
    _run("u200_flits2" if not fast_clk else "u200_fast", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                             addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                             rst_local=2, gather_pair=1, gather_flits=2, models=False, n_ops=0, fast_clk=fast_clk,
                             directed=[dict(pv=False, E=64, D=64, G=1, GS=1, tq=4, hg0=0, u0=0, n=3, flags=4 | 1,
                                            c_hi=40),
                                       dict(pv=True, E=64, D=64, G=1, GS=1, tq=4, hg0=0, u0=0, n=3, flags=1)]),
         rand_reset=0, mutant=mutant)


def test_at_mvm_fast_clk():
    """The full K4 shape with FAST_CLK 1 (the 250 MHz stages, docs/platform-u200.md §8.10): the three-stage row-vector
    emitter of the PV unit and of the gather (MAXOUT and LSUM lines 3 cycles later). Results and SPM image are the
    simulator's."""
    _run("u200_fast", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                           addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9, rst_local=2, gather_pair=1,
                           gather_flits=2, xslr_reg=1, fast_clk=1, n_ops=40))


def test_at_mvm_zero_state():
    """The first ATTPV after configuration. Every flop and RAM word starts at zero, as on the FPGA after the image
    load (rand_reset 0; the usual random initial values hide this). The ops are those of `selftest` id 5 (n_kv 40,
    T_q 4 on four channels: three groups, so cluster 3 has none): a causal ATTQK with MAXOUT, then ATTPV with LSUM.
    With the output stage of tsr_mvm_pv reset to `ng == 0`, every cluster offered the zero lines of an ATTPV that had
    never started to the gather as soon as the first ATTPV switched it to PV mode. They were written to lines of the
    setup registers (here the scores of the ATTQK before it) and the SPM image differed from the simulator's. The
    shape is that of test_at_mvm_u200_flits2 (the full image's paired two-flit gather); the packaged core with the
    Xilinx wrappers is tb/core test_k4_selftest_first_run."""
    _zero_state()


def test_at_mvm_zero_state_fast():
    """The same first ATTPV at zero state with FAST_CLK 1 (the PV drain with the extra stages, the emitters in three stages)."""
    _zero_state(fast_clk=1)


def test_at_mvm_zero_state_mutant():
    """The old reset of the output stage must fail the same run (the assertion of tsr_mvm_pv stops the simulation
    after reset; without it the SPM image differs from the simulator's)."""
    with pytest.raises(SystemExit):
        _zero_state(("tsr_mvm_pv.sv", PV_FIX, PV_OLD, "at_zero_state_pv_reset"))


if pytest is not None:
    test_at_mvm_zero_state_mutant = pytest.mark.slow(test_at_mvm_zero_state_mutant)


def test_at_mvm_xslr():
    """The U200 shape with XSLR_REG: two register stages per cluster on the attention-stall strobe. The results and
    the SPM image are the simulator's; tsr_mvm asserts that att_stall_q equals the clusters' strobes of 3 cycles before."""
    _run("xslr", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                      addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                      rst_local=2, gather_pair=1, n_ops=24, xslr_reg=1, require_stall=True))


def test_at_mvm_bc_reset():
    """Production BC_PIPE 2 / local reset 2: all three broadcast kinds against old-pipe histories."""
    _run("bc_rst2", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30,
                         spm_rd_lat=4, addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2,
                         mac_lat=9, rst_local=2, models=False, n_ops=0, bc_all_valids=True,
                         directed=[dict(pv=False, E=128, D=64, G=2, GS=2, tq=1, hg0=0, u0=0, n=3),
                                   dict(pv=True, E=128, D=64, G=2, GS=2, tq=1, hg0=0, u0=0, n=3)]))


def test_at_mvm_odd():
    """3 channels, T_MAX 3, K_MAX 4096, latencies at their minimum, pipes 0, ADD_LAT 8."""
    _run("odd", dict(num_ch=3, t_max=3, k_max=4096, spm_bytes=256 << 10, ch_bytes=1 << 30, spm_rd_lat=1,
                     addr_beat_w=24, rd_out=4, bc_pipe=0, res_pipe=0, act_rd_lat=1, mac_lat=3, mul_lat=1, add_lat=8,
                     n_ops=40, p_sp=0.1))


def test_at_mvm_one():
    """1 channel, T_MAX 1 (the k1 bitstream), K_MAX 2048."""
    _run("one", dict(num_ch=1, t_max=1, k_max=2048, spm_bytes=256 << 10, ch_bytes=1 << 28, spm_rd_lat=2,
                     addr_beat_w=22, rd_out=16, n_ops=30))


def test_at_fa_t8():
    """AT §6.6 tile sequences at T_MAX 8 (U200 shape, SPM 4 MiB): decode, prefill, SWA, two head subsets."""
    _run("fa8", dict(num_ch=4, t_max=8, k_max=32768, spm_bytes=4 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                     addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9), testcase="at_fa")


def test_at_fa_t4():
    """AT §6.6 at T_MAX 4: G 8 in two head subsets over a wrapped SWA ring (two FAST chunks with bounds)."""
    _run("fa4", dict(num_ch=4, t_max=4, k_max=32768, spm_bytes=4 << 20, ch_bytes=16 << 30, spm_rd_lat=4,
                     addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2, mac_lat=9,
                     rst_local=2, gather_pair=1), testcase="at_fa")


def test_at_mvm_pair_old():
    _pair_run(0)


def test_at_mvm_pair_new():
    _pair_run(1)


def _pair_run(enabled):
    """Same seeded real MVM/QK/PV/MAXOUT/LSUM streams, random SPM stalls."""
    _run("pair_%d" % enabled,
         dict(num_ch=4, t_max=8, k_max=32768, spm_bytes=1 << 20, ch_bytes=16 << 30,
              spm_rd_lat=4, addr_beat_w=28, rd_out=16, bc_pipe=2, res_pipe=2,
              mac_lat=9, rst_local=2, gather_pair=enabled, isolate_spm_rng=True, models=False, n_ops=16,
              directed=[dict(pv=False, E=128, D=64, G=4, GS=4, tq=2, hg0=0, u0=0, n=8),
                        dict(pv=True, E=128, D=64, G=4, GS=4, tq=2, hg0=0, u0=0, n=8),
                        dict(pv=False, E=128, D=64, G=1, GS=1, tq=1, hg0=0, u0=0, n=1),
                        dict(pv=True, E=128, D=64, G=1, GS=1, tq=1, hg0=0, u0=0, n=1)]))

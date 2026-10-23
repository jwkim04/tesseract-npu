"""Random Tesseract programs for simulator-lockstep tests (moved from tb/cmdproc/test_cmdproc.py;
tb/cmdproc and tb/core import it from here).

    gen = ProgGen(rnd, cfg, has_mvm)              # M2 behaviour (unchanged random stream)
    gen = ProgGen(rnd, cfg, True, m3=True)        # + execution-oriented MVMs, no undefined overlaps
    data = gen.init_data()                        # (ch, byte offset, bytes) regions to preload
    prog, cmd, count, args = gen.program()        # random program
    prog, cmd, count, args = gen.mvm_program()    # m3: LD -> QACT -> MVM (-> ACC MVM) -> ST chain

Programs exercise register-relative operands (with 32-bit wrap) aimed at interesting addresses
(inside, exact fit, just over, wrapped) of the SPM and the DRAM limit, K / T / stride / channel /
sub values inside and outside their ranges, reserved-field flips and undefined opcodes, so runs end
in END, in every error kind at any position, past CMD_COUNT or in a fetch range error. DRAM writes
never hit the command buffer (self-modifying programs are undefined).

m3=True (docs/rtl-m3.md §10.1) changes the generator for M3 lockstep tests:
  - MVM: both fmts, ACC, dst_stride 0 / gaps, T up to T_MAX, N edge cases (0, 1, < NUM_CH, R % 16
    != 0, multiples of 16 +- 1), K over block multiples incl. partial chunks, x preferably the
    output of an earlier QACT of the program (realistic QA) else any SPM lines (raw QA), weights
    from a "sane fp16" window (every aligned halfword is a normal fp16 of 2^-14..2^-2, so every d
    / dmin is sane), sometimes from a raw-byte window, register-relative A1..A3;
  - undefined overlaps are never generated for instructions that pass their checks: MVM dst vs x,
    MVM token dst ranges (dst_stride >= ceil(N/16) when T > 1), QACT dst vs src (spec 구현
    명확화 "겹치는 피연산자");
  - init_data(): per channel a weight window (sane fp16 halfwords), an activation window (f32 of
    magnitude 2^-6..2^3) and a raw window (random bytes).
"""
from __future__ import annotations

import struct

from . import isa
from .tsr_ref import ref

M32 = 0xFFFFFFFF


def prog_bytes(insns) -> bytes:
    return b"".join(isa.insn_bytes(w) for w in insns)


def raw(*w):
    return list(w) + [0] * (8 - len(w))


def enc_mvm(fmt, flags, dst, x, woff, N, K, T, dstride, ra=0, rb=0, rc=0):
    """MVM words (tb/cmdproc argument order: flags = w0[15:12]); isa.enc_mvm is the keyword form."""
    return [isa.w0(isa.OP_MVM, sub=fmt, flags=flags, ra=ra, rb=rb, rc=rc), dst & M32, x & M32,
            woff & M32, N & M32, (K & 0xFFFF) | ((T & 0xF) << 16), dstride & M32, 0]


def with_base(w, ra=None, rb=None, rc=None):
    w = list(w)
    f = isa.w0_fields(w[0])
    w[0] = isa.w0(f["opcode"], f["sub"], f["flags"], f["ra"] if ra is None else ra,
                  f["rb"] if rb is None else rb, f["rc"] if rc is None else rc, f["rd"])
    return w


OPC, SUB, RA, RB, RC, RD = 0xFF, 0xF00, 0xF0000, 0xF00000, 0xF000000, 0xF0000000


def _overlap(a, ea, b, eb) -> bool:
    return a < b + eb and b < a + ea


class ProgGen:
    """Random programs over NOP FENCE END LI ADDI LD ST QACT (+ MVM). See the module docstring.
    With m3=False the generated programs are exactly those of the M2 generator for every seed."""

    def __init__(self, rnd, cfg, has_mvm, *, m3=False):
        self.r = rnd
        self.cfg, self.has_mvm = cfg, has_mvm
        self.m3 = bool(m3)
        self.SL = cfg["spm_bytes"] // 64
        self.lim = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.nch, self.kmax, self.tmax = cfg["num_ch"], cfg["k_max"], cfg["t_max"]
        L = self.lim
        w = max(1, L // 4)
        self.data_base, self.data_n = L // 4, min(512, w)   # random data (LD sources, weights)
        self.wr_base, self.wr_n = L // 2, min(4096, w)      # preferred ST targets
        self.cmd_base, self.cmd_n = (3 * L) // 4, min(4096, w)   # command buffers
        self.cmd_lo = self.cmd_hi = 0
        self.e = 1.0                                        # error intensity of the current program
        if self.m3:
            # data window = [weights (sane fp16) | activations (f32) | raw bytes]
            self.data_n = min(8192, w)
            self.wwin_base, self.wwin_n = self.data_base, max(1, self.data_n // 2)
            self.awin_base, self.awin_n = self.data_base + self.wwin_n, max(1, self.data_n // 4)
            self.rwin_base = self.awin_base + self.awin_n
            self.rwin_n = max(0, self.data_base + self.data_n - self.rwin_base)
        self.qa_regions = []                                # m3: QACT outputs of the current program

    # ------------------------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------------------------
    def init_data(self):
        """(ch, byte offset, data) regions with random bytes: the data window of every channel
        (m3: weight / activation / raw sub-windows, see the module docstring)."""
        if not self.m3:
            return [(c, self.data_base * 64, self.r.randbytes(64 * self.data_n)) for c in range(self.nch)]
        out = []
        r = self.r
        for c in range(self.nch):
            hw = [((r.getrandbits(1) << 15) | (r.randint(1, 13) << 10) | r.getrandbits(10))
                  for _ in range(32 * self.wwin_n)]
            wts = struct.pack("<%dH" % len(hw), *hw)
            fw = [((r.getrandbits(1) << 31) | ((127 + r.randint(-6, 3)) << 23) | r.getrandbits(23))
                  for _ in range(16 * self.awin_n)]
            act = struct.pack("<%dI" % len(fw), *fw)
            out.append((c, self.data_base * 64, wts + act + r.randbytes(64 * self.rwin_n)))
        return out

    # ------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------
    def w(self, choices):
        tot = sum(p for _, p in choices)
        x = self.r.random() * tot
        for v, p in choices:
            x -= p
            if x < 0:
                return v
        return choices[-1][0]

    def bad(self, p):
        """True with probability p scaled by the program's error intensity."""
        return self.r.random() < p * self.e

    def spm_start(self, ext):
        r, SL = self.r, self.SL
        p = r.random() if self.bad(1.0) else 0.0
        if ext <= SL and p < 0.72:
            return r.randrange(0, SL - ext + 1)
        if ext <= SL and p < 0.82:
            return SL - ext
        if p < 0.90:
            return (SL - min(ext, SL) + r.randint(1, 4)) & M32
        return r.choice([SL, SL + 1, M32, 0x80000000, M32 - 3, r.getrandbits(32), 0])

    def dram_start(self, ext, write=False):
        # The limit as the 32-bit start field sees it: beyond M32 + ext it is out of reach and the
        # top of the field is the edge (DRAM limit above 2^32 beats).
        r, L = self.r, min(self.lim, M32 + ext)
        p = r.random() if self.bad(1.0) else r.random() * 0.80
        if p < 0.55 and ext <= (self.wr_n if write else self.data_n):
            a = (self.wr_base + r.randrange(self.wr_n - ext + 1)) if write else \
                (self.data_base + r.randrange(self.data_n - ext + 1))
        elif p < 0.70 and ext <= L:
            a = r.randrange(0, L - ext + 1)
        elif p < 0.80 and ext <= L:
            a = L - ext
        elif p < 0.90:
            a = (L - min(ext, L) + r.randint(1, 4)) & M32
        else:
            a = r.choice([L, L + 1, M32, 0x80000000, r.getrandbits(32), 0])
        a &= M32                                      # the address the operand encodes
        if write and a < self.cmd_hi and a + ext > self.cmd_lo:
            a = self.wr_base                          # keep the command buffer intact
            if a < self.cmd_hi and a + ext > self.cmd_lo:
                a = self.cmd_hi                       # (may be out of range: DRAM_RANGE, no write)
        return a

    def operand(self, regs, target):
        """(base register, immediate) with R[base] + imm == target (mod 2^32)."""
        if self.r.random() < 0.45:
            return 0, target & M32
        k = self.r.randrange(16)
        return k, (target - regs[k]) & M32

    def flip(self, w, mask_used):
        """Flip one random bit outside mask_used (8 words) -> reserved-field violation."""
        while True:
            wi = self.r.randrange(8)
            b = self.r.randrange(32)
            if not (mask_used[wi] >> b) & 1:
                w = list(w)
                w[wi] ^= 1 << b
                return w

    def k_value(self, bs, kmax_soft):
        r = self.r
        p = r.random() if self.bad(1.0) else 0.0
        if p < 0.80:
            return bs * r.randint(1, max(1, min(self.kmax, kmax_soft) // bs))
        if p < 0.86:
            return self.kmax - (self.kmax % bs)
        return r.choice([0, bs // 2, bs + 16, self.kmax + bs, 0x80000000, M32, 0x10000 + bs])

    def t_value(self, field_max=M32):
        r = self.r
        if not self.bad(0.15):
            return r.randint(1, self.tmax)
        return r.choice([0, min(self.tmax + 1, field_max), min(15, field_max), min(0x80000001, field_max),
                         field_max])

    def place_apart(self, ext, avoid, start=None):
        """An SPM start for an operand of `ext` lines that does not overlap the in-range ranges
        `avoid` [(start, ext), ...]: `start` if it already qualifies, else a random free spot, else
        an out-of-range start (the instruction then ends in SPM_RANGE and has no effect)."""
        avoid = [(a, e) for a, e in avoid if e > 0 and a + e <= self.SL]
        ok = lambda s: s + ext > self.SL or not any(_overlap(s, max(ext, 1), a, e) for a, e in avoid)  # noqa: E731
        if start is not None and ok(start):
            return start
        if ext <= self.SL:
            for _ in range(24):
                s = self.r.randrange(0, self.SL - ext + 1)
                if ok(s):
                    return s
            for a, e in sorted(avoid):            # just after / before a range
                for s in (a + e, a - ext):
                    if 0 <= s <= self.SL - ext and ok(s):
                        return s
        return self.SL + self.r.randint(0, 3)

    # ------------------------------------------------------------------------------------------
    # One instruction
    # ------------------------------------------------------------------------------------------
    def insn(self, regs):
        if self.m3:
            return self._insn_m3(regs)
        ops = [("nop", 3), ("fence", 2), ("li", 9), ("addi", 9), ("ld", 14), ("st", 10), ("qact", 12),
               ("bad", 3 * self.e)]
        if self.has_mvm:
            ops.append(("mvm", 9))
        op = self.w(ops)
        w, used, upd = self._gen(op, regs)
        return self._finish(op, w, used, upd, regs)

    def _finish(self, op, w, used, upd, regs):
        if op != "bad" and self.bad(0.03):
            w = self.flip(w, used)
            upd = None
        if upd is not None and upd[0] != 0:
            regs[upd[0]] = upd[1]
        return w

    def _gen(self, op, regs):
        """(words, used-bit masks, register update) of one random instruction of kind op (the M2
        generator; random calls in the original order)."""
        r = self.r
        upd = None
        if op in ("nop", "fence"):
            w = isa.enc_nop() if op == "nop" else isa.enc_fence()
            used = [OPC, 0, 0, 0, 0, 0, 0, 0]
        elif op == "li":
            rd = r.randrange(16)
            imm = r.choice([r.getrandbits(32), r.randrange(64), self.SL - r.randrange(8), self.lim - r.randrange(8),
                            M32, 0x80000000, self.data_base, self.wr_base, r.randrange(self.SL)]) & M32
            w = isa.enc_li(rd, imm)
            used = [OPC | RD, M32, 0, 0, 0, 0, 0, 0]
            upd = (rd, imm)
        elif op == "addi":
            rd, ra = r.randrange(16), r.randrange(16)
            imm = r.choice([r.getrandbits(32), r.randrange(-64, 64) & M32, M32, 1])
            w = isa.enc_addi(rd, ra, imm)
            used = [OPC | RA | RD, M32, 0, 0, 0, 0, 0, 0]
            upd = (rd, (regs[ra] + imm) & M32)
        elif op in ("ld", "st"):
            ch = r.randrange(16) if self.bad(0.05) else r.randrange(self.nch)
            ns = [(0, 5), (r.randint(1, 8), 40), (r.randint(9, 64), 30), (r.randint(65, 300), 15),
                  (r.choice([self.SL + 1, M32, 0x80000000, self.lim + 1]), 10 * self.e)]
            top_n = self.lim - M32               # > 0: limit above 2^32, reached from the field top
            if 0 < top_n < self.SL:
                ns.append((top_n + r.randint(-2, 3), 8))
            n = self.w(ns)
            spm = self.spm_start(n)
            dram = self.dram_start(n, write=(op == "st"))
            ra, w1 = self.operand(regs, spm)
            rb, w2 = self.operand(regs, dram)
            enc = isa.enc_ld if op == "ld" else isa.enc_st
            w = enc(ch, w1, w2, n, ra=ra, rb=rb)
            used = [OPC | SUB | RA | RB, M32, M32, M32, 0, 0, 0, 0]
        elif op == "qact":
            kind = r.randrange(2, 16) if self.bad(0.05) else r.randrange(2)
            bs = 256 if kind == 1 else 32
            K = self.k_value(bs, 2048)
            T = self.t_value()
            kv = 0 < K <= self.kmax and K % bs == 0
            xl = K // 16 if kv else 4
            stride = self.w([(0, 60), (xl + r.randint(0, 8), 30),
                             (r.choice([1, M32, 0x80000000, xl - 1 if xl > 1 else 1]), 10 * self.e)])
            ql = isa.qa_lines(kind & 1, K) if kv else 4
            tt = T if 1 <= T <= 8 else 1
            st_eff = stride or xl
            dst = self.spm_start(tt * ql)
            src = self.spm_start((tt - 1) * st_eff + xl)
            for _ in range(4):                       # keep dst and src apart (overlap is undefined)
                if dst + tt * ql <= src or src + (tt - 1) * st_eff + xl <= dst:
                    break
                src = self.spm_start((tt - 1) * st_eff + xl)
            ra, w1 = self.operand(regs, dst)
            rb, w2 = self.operand(regs, src)
            w = isa.enc_qact(kind, w1, w2, K, T, stride, ra=ra, rb=rb)
            used = [OPC | SUB | RA | RB, M32, M32, M32, M32, M32, 0, 0]
        elif op == "mvm":
            fmt = r.randrange(2, 16) if self.bad(0.05) else r.randrange(2)
            acc = r.randrange(2)
            bs = 256 if fmt == 1 else 32
            K = self.k_value(bs, 512) & 0xFFFF
            T = self.t_value(15)
            N = self.w([(r.randint(0, 40), 80), (r.choice([0, 1, self.nch - 1, self.nch + 1]), 10),
                        (r.choice([M32, self.SL * 16 + 1, 1 << 20]), 10 * self.e)])
            kv = 0 < K <= self.kmax and K % bs == 0
            yl = (N + 15) // 16
            ds = self.w([(0, 60), (yl + r.randint(0, 3), 30), (r.choice([1, M32, 0x80000000]), 10 * self.e)])
            ql = isa.qa_lines(fmt & 1, K) if kv else 4
            tt = T if 1 <= T <= 8 else 1
            dst = self.spm_start((tt - 1) * (ds or yl) + yl)
            x = self.spm_start(tt * ql)
            Rr = -(-N // self.nch)
            rbeats = ref().row_beats(fmt & 1, K) if kv else 1
            ext = Rr * rbeats
            if not self.bad(0.25) and ext <= self.data_n:
                woff = self.data_base + r.randrange(self.data_n - ext + 1)
            else:
                woff = self.dram_start(min(ext, M32))
            ra, w1 = self.operand(regs, dst)
            rb, w2 = self.operand(regs, x)
            rc, w3 = self.operand(regs, woff)
            w = enc_mvm(fmt, acc, w1, w2, w3, N, K, T, ds, ra=ra, rb=rb, rc=rc)
            used = [OPC | SUB | 0x1000 | RA | RB | RC, M32, M32, M32, M32, 0x000FFFFF, M32, 0]
        else:   # bad: undefined / unimplemented opcode, or a reserved bit of a valid instruction
            if r.random() < 0.5:
                legal = set(isa.M2_OPS) | ({isa.OP_MVM} if self.has_mvm else set())
                opc = r.choice([o for o in range(256) if o not in legal])
                w = raw(isa.w0(opc, r.randrange(16), 0, r.randrange(16), 0, 0, r.randrange(16)), r.getrandbits(32))
            else:
                w = self.flip(isa.enc_nop(), [0xFF, 0, 0, 0, 0, 0, 0, 0])
            used = [M32] * 8
        return w, used, upd

    # ------------------------------------------------------------------------------------------
    # M3 generation
    # ------------------------------------------------------------------------------------------
    def _insn_m3(self, regs):
        ops = [("nop", 2), ("fence", 1), ("li", 7), ("addi", 7), ("ld", 12), ("st", 9), ("qact", 12),
               ("bad", 3 * self.e)]
        if self.has_mvm:
            ops.append(("mvm", 18))
        op = self.w(ops)
        nq = len(self.qa_regions)
        if op == "qact":
            w, used = self.qact_m3(regs)
            upd = None
        elif op == "mvm":
            w, used = self.mvm_m3(regs)
            upd = None
        else:
            w, used, upd = self._gen(op, regs)
        if op != "bad" and self.bad(0.03):
            w = self.flip(w, used)
            upd = None
            del self.qa_regions[nq:]                  # a flipped QACT does not execute
        if upd is not None and upd[0] != 0:
            regs[upd[0]] = upd[1]
        return w

    def qact_m3(self, regs, kind=None, K=None, T=None, src=None, src_stride=None):
        """QACT with dst and src kept apart whenever both are in range. Returns (words, used).
        Records the output as a QA region (for later MVMs) when the instruction will execute."""
        r = self.r
        if kind is None:
            kind = r.randrange(2, 16) if self.bad(0.05) else r.randrange(2)
        bs = 256 if kind == 1 else 32
        if K is None:
            K = self.k_value(bs, 2048 if self.SL >= 4096 else 512)
        if T is None:
            T = self.t_value()
        kv = 0 < K <= self.kmax and K % bs == 0 and kind < 2
        xl = K // 16 if kv else 4
        if src_stride is None:
            src_stride = self.w([(0, 60), (xl + r.randint(0, 8), 30),
                                 (r.choice([1, M32, 0x80000000, xl - 1 if xl > 1 else 1]), 10 * self.e)])
        ql = isa.qa_lines(kind & 1, K) if kv else 4
        tt = T if 1 <= T <= 8 else 1
        st_eff = src_stride or xl
        src_ext = (tt - 1) * st_eff + xl
        if src is None:
            src = self.spm_start(src_ext)
        dst = self.place_apart(tt * ql, [(src, src_ext)], self.spm_start(tt * ql))
        ra, w1 = self.operand(regs, dst)
        rb, w2 = self.operand(regs, src)
        w = isa.enc_qact(kind, w1, w2, K, T, src_stride, ra=ra, rb=rb)
        executes = kv and 1 <= T <= self.tmax and dst + tt * ql <= self.SL and src + src_ext <= self.SL
        if executes:
            self.qa_regions.append((dst, kind, K, T))
        return w, [OPC | SUB | RA | RB, M32, M32, M32, M32, M32, 0, 0]

    def mvm_k(self, fmt):
        """Interesting K values (blocks, partial chunks, chunk boundaries), or an illegal K."""
        r = self.r
        bs = 256 if fmt == 1 else 32
        if self.bad(0.06):
            return self.k_value(bs, 512) & 0xFFFF
        soft = min(self.kmax, 4096 if self.SL >= 2048 else 512)
        if fmt == 1:
            c = [256, 512, 768, 1024, 1280, 2048, 2304, bs * r.randint(1, max(1, soft // bs))]
        else:
            c = [32, 64, 96, 480, 992, 1024, 1056, 2048, 3104, bs * r.randint(1, max(1, soft // bs))]
        K = r.choice([k for k in c if k <= soft] or [bs])
        return K

    def mvm_n(self, rbeats):
        """N: edge cases, bounded so that R * ROW_BEATS fits the weight window."""
        r, nch = self.r, self.nch
        nmax = nch * max(1, self.wwin_n // max(1, rbeats))
        nmax = min(nmax, 4 * self.SL)                 # y keeps <= 1/4 of the SPM: most MVMs execute
        R16 = 16 * nch
        c = [0, 1, max(1, nch - 1), nch, nch + 1, 15, 16, 17, 31, 33, R16 - 1, R16 + 1, 3 * nch + 5,
             r.randint(1, max(1, nmax))]
        return min(r.choice(c), nmax)

    def mvm_m3(self, regs, fmt=None, x=None, K=None, T=None, acc=None, avoid=()):
        """One MVM: legal shapes most of the time (errors scaled by self.e), x preferably an earlier
        QACT output, no undefined overlaps when it executes. Returns (words, used)."""
        r = self.r
        if fmt is None:
            fmt = r.randrange(2, 16) if self.bad(0.05) else r.randrange(2)
        if acc is None:
            acc = r.randrange(2)
        bs = 256 if fmt == 1 else 32
        if x is None and K is None and fmt < 2:
            cands = [q for q in self.qa_regions if q[1] == fmt]
            if cands and r.random() < 0.65:
                q = r.choice(cands)
                x, K = q[0], q[2]
                T = r.randint(1, q[3]) if T is None else T
        if K is None:
            K = self.mvm_k(fmt & 1)
        if T is None:
            T = self.t_value(15)
        kv = 0 < K <= self.kmax and K % bs == 0 and fmt < 2
        rbeats = ref().row_beats(fmt & 1, K) if (0 < K and K % bs == 0) else 1
        N = self.mvm_n(rbeats)
        if self.bad(0.04):
            N = r.choice([M32, self.SL * 16 + 1, 1 << 20, self.nch * (self.wwin_n + 1)])
        yl = (N + 15) // 16
        tt = T if 1 <= T <= 8 else 1
        ds = self.w([(0, 55), (yl + r.randint(0, 3), 35), (r.choice([1, M32, 0x80000000]), 10 * self.e)])
        if tt > 1 and ds and ds < yl and not self.bad(0.3):
            ds = yl + r.randint(0, 2)                 # token dst ranges must not overlap
        if tt > 1 and ds and ds < yl:
            N = min(N, 16 * ds)                       # still keep token ranges apart
            yl = (N + 15) // 16
        ds_eff = ds or yl
        ql = isa.qa_lines(fmt & 1, K) if kv else 4
        x_ext = tt * ql
        d_ext = (tt - 1) * ds_eff + yl
        if x is None:
            x = self.spm_start(x_ext)
        dst = self.place_apart(d_ext, [(x, x_ext)] + list(avoid), self.spm_start(d_ext))
        Rr = -(-N // self.nch)
        ext = Rr * rbeats
        p = r.random()
        if not self.bad(0.2) and ext <= self.wwin_n and p < 0.85:
            woff = self.wwin_base + r.randrange(self.wwin_n - ext + 1)
        elif not self.bad(0.2) and 0 < ext <= self.rwin_n:
            woff = self.rwin_base + r.randrange(self.rwin_n - ext + 1)
        else:
            woff = self.dram_start(min(ext, M32))
        ra, w1 = self.operand(regs, dst)
        rb, w2 = self.operand(regs, x)
        rc, w3 = self.operand(regs, woff)
        w = isa.enc_mvm(fmt, w1, w2, w3, N, K, T, ds, acc=acc, ra=ra, rb=rb, rc=rc)
        return w, [OPC | SUB | 0x1000 | RA | RB | RC, M32, M32, M32, M32, 0x000FFFFF, M32, 0]

    # ------------------------------------------------------------------------------------------
    # Programs
    # ------------------------------------------------------------------------------------------
    def _cmd_placement(self, n, n_max):
        r = self.r
        p = r.random() if self.bad(0.5) else 0.0
        if p < 0.85:
            cmd = self.cmd_base + r.randrange(max(1, self.cmd_n - n_max))
        elif p < 0.93:
            cmd = self.lim - r.randint(0, (n + 1) // 2 + 1)    # crosses the DRAM limit
        elif p < 0.97:
            cmd = self.lim + r.randrange(4)
        else:
            cmd = r.choice([1 << 32, (1 << 32) + 7, (1 << 40) + 3, (1 << 64) - 1])
        self.cmd_lo, self.cmd_hi = cmd, cmd + n_max
        return cmd

    def program(self, n_max=20):
        r = self.r
        self.e = r.choice([0.0, 0.03, 0.15, 0.5, 1.0])     # clean ... error-heavy
        self.qa_regions = []
        args = [r.choice([r.getrandbits(32), r.randrange(self.SL), self.lim - r.randrange(4), 0, M32])
                for _ in range(8)]
        regs = [0] * 8 + list(args)
        n = r.randint(1, n_max)
        cmd = self._cmd_placement(n, n_max)
        insns = [self.insn(regs) for _ in range(n)]
        if not self.bad(0.3):
            insns.append(isa.enc_end())
        prog = prog_bytes(insns)
        count = len(insns)
        p = r.random() if self.bad(1.0) else 1.0
        if p < 0.05:
            count = r.randint(0, len(insns))
        elif p < 0.08:
            count = len(insns) + r.randint(1, 3)
        return prog, cmd, count, args

    def mvm_program(self, n_max=16, e=None):
        """m3: a structured chain on one SPM layout — LD f32 activations, QACT, MVM (sometimes a
        second MVM with ACC into the same dst, sometimes with the other fmt on the same x), ST
        of y, END — with register-relative operands; e (error intensity) defaults to clean-ish."""
        assert self.m3, "mvm_program needs ProgGen(..., m3=True)"
        r = self.r
        self.e = r.choice([0.0, 0.0, 0.0, 0.03, 0.15]) if e is None else e
        self.qa_regions = []
        args = [r.choice([r.getrandbits(32), r.randrange(self.SL), self.wwin_base, 0]) for _ in range(8)]
        regs = [0] * 8 + list(args)
        cmd = self._cmd_placement(n_max, n_max + 2)  # before any ST target is chosen
        insns = []
        for _ in range(r.randint(0, 2)):                # set up some base registers
            rd = r.randrange(1, 16)
            v = r.choice([r.randrange(self.SL), self.wwin_base + r.randrange(self.wwin_n), r.getrandbits(32)])
            insns.append(isa.enc_li(rd, v))
            regs[rd] = v
        fmt = r.randrange(2)
        kind = fmt
        bs = 256 if fmt else 32
        K = self.mvm_k(fmt)
        if not (0 < K <= self.kmax and K % bs == 0):
            K = bs
        T = r.randint(1, self.tmax)
        xl = K // 16
        # shrink until the activations and their QA images fit the SPM comfortably
        while (T * xl + T * isa.qa_lines(kind, K) + 8 > self.SL // 2 or T * xl > self.awin_n) and (K > bs or T > 1):
            if K > bs:
                K = max(bs, (K // 2) // bs * bs)
            else:
                T -= 1
            xl = K // 16
        src_stride = r.choice([0, xl + r.randint(0, 2)]) if T > 1 else 0
        st_eff = src_stride or xl
        src_ext = (T - 1) * st_eff + xl
        src = r.randrange(0, max(1, self.SL - src_ext + 1)) if src_ext <= self.SL else 0
        n_ld = src_ext
        if n_ld <= self.SL and n_ld <= self.awin_n:
            ra, w1 = self.operand(regs, src)
            dram = self.awin_base + r.randrange(self.awin_n - n_ld + 1)
            rb, w2 = self.operand(regs, dram)
            insns.append(isa.enc_ld(r.randrange(self.nch), w1, w2, n_ld, ra=ra, rb=rb))
        w, _ = self.qact_m3(regs, kind=kind, K=K, T=T, src=src, src_stride=src_stride)
        insns.append(w)
        if self.qa_regions:
            x = self.qa_regions[-1][0]
            avoid = []
            wm, _ = self.mvm_m3(regs, fmt=fmt, x=x, K=K, T=r.randint(1, T), acc=r.randrange(2))
            insns.append(wm)
            f = isa.w0_fields(wm[0])
            dst = (regs[f["ra"]] + wm[1]) & M32
            Tm, N = (wm[5] >> 16) & 0xF, wm[4]
            yl = (N + 15) // 16
            ds = wm[6] or yl
            d_ext = (Tm - 1) * ds + yl
            if r.random() < 0.4 and dst + d_ext <= self.SL:   # accumulate a second product into y
                rbeats = ref().row_beats(fmt, K)
                Rr = -(-N // self.nch) if N else 0
                if Rr * rbeats <= self.wwin_n:
                    woff = self.wwin_base + r.randrange(self.wwin_n - Rr * rbeats + 1)
                    ra, w1 = self.operand(regs, dst)
                    rb, w2 = self.operand(regs, x)
                    rc, w3 = self.operand(regs, woff)
                    insns.append(isa.enc_mvm(fmt, w1, w2, w3, N, K, Tm, wm[6], acc=1, ra=ra, rb=rb, rc=rc))
            if 0 < d_ext <= self.SL and dst + d_ext <= self.SL:
                ra, w1 = self.operand(regs, dst)
                wr = self.dram_start(d_ext, write=True) if self.bad(0.2) else (
                    self.wr_base + r.randrange(max(1, self.wr_n - d_ext + 1)) if d_ext <= self.wr_n else self.wr_base)
                if wr < self.cmd_hi and wr + d_ext > self.cmd_lo:
                    wr = self.cmd_hi                  # never overwrite the command buffer
                rb, w2 = self.operand(regs, wr)
                insns.append(isa.enc_st(r.randrange(self.nch), w1, w2, d_ext, ra=ra, rb=rb))
            avoid.append((dst, d_ext))
        while len(insns) < n_max - 1 and r.random() < 0.3:
            insns.append(self.insn(regs))
        insns.append(isa.enc_end())
        return prog_bytes(insns), cmd, len(insns), args


# =============================================================================================
# Undefined-overlap check of an executed program prefix
# =============================================================================================
def executed_overlaps(prog: bytes, n_exec: int, args, cfg) -> list:
    """Undefined operand overlaps (spec 구현 명확화 "겹치는 피연산자") among the first n_exec
    instructions of prog, which executed successfully (n_exec = ERR_PC for an error ending, the
    index of END + 1 otherwise). Register values follow LI / ADDI exactly. Returns a list of
    strings: QACT dst vs src, MVM dst vs x, MVM token dst ranges (T > 1, stride < ceil(N/16))."""
    R = [0] * 8 + [a & M32 for a in (list(args) + [0] * 8)[:8]]
    out = []
    for pc in range(n_exec):
        w = isa.insn_words(prog[32 * pc:32 * pc + 32])
        f = isa.w0_fields(w[0])
        op = f["opcode"]
        if op == isa.OP_LI and f["rd"]:
            R[f["rd"]] = w[1]
        elif op == isa.OP_ADDI and f["rd"]:
            R[f["rd"]] = (R[f["ra"]] + w[1]) & M32
        elif op == isa.OP_QACT:
            kind, K, T = f["sub"], w[3], w[4]
            xl = K // 16
            st = w[5] or xl
            dst, src = (R[f["ra"]] + w[1]) & M32, (R[f["rb"]] + w[2]) & M32
            if _overlap(dst, T * isa.qa_lines(kind, K), src, (T - 1) * st + xl):
                out.append("pc %d: QACT dst %d and src %d overlap" % (pc, dst, src))
        elif op == isa.OP_MVM:
            fmt, N, K, T = f["sub"], w[4], w[5] & 0xFFFF, (w[5] >> 16) & 0xF
            yl = (N + 15) // 16
            ds = w[6] or yl
            dst, x = (R[f["ra"]] + w[1]) & M32, (R[f["rb"]] + w[2]) & M32
            if N and _overlap(dst, (T - 1) * ds + yl, x, T * isa.qa_lines(fmt, K)):
                out.append("pc %d: MVM dst %d and x %d overlap" % (pc, dst, x))
            if N and T > 1 and ds < yl:
                out.append("pc %d: MVM token dst ranges overlap (stride %d < %d lines)" % (pc, ds, yl))
    return out


def executed_count(prog: bytes, final: dict, count: int) -> int:
    """Instructions that executed successfully, from a run's final state (simulator or RTL)."""
    if final["err_code"]:
        return min(final["err_pc"], count)
    for pc in range(count):
        if isa.insn_words(prog[32 * pc:32 * pc + 32])[0] & 0xFF == isa.OP_END:
            return pc + 1
    return count

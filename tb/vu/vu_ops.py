"""VU op encoders and random op generators for tb/vu (docs/design/02-vector-unit.md "VU" §5, AT §5.4-§5.7).

Encoders return the eight words w0..w7 (ints). Field layout: VU §5.2-§5.8, VU §5.11 (AT's vector ops);
the generated tables (tsr_tables_pkg / tsr_tables.h) are the source the simulator decodes with.
"""
from __future__ import annotations

import random
import struct

M32 = 0xFFFFFFFF

OP_DQROW, OP_VOP, OP_RED, OP_ROPE, OP_TOPK, OP_SOP = 0x12, 0x30, 0x31, 0x32, 0x33, 0x40
OP_ASMAX, OP_AMASK, OP_AMERGE, OP_ANORM = 0x34, 0x35, 0x36, 0x37
OP_GDN = 0x39

# VOP functions (VU §5.2): code -> (name, reads 'a' 'b' 'c', half rate)
VOP_FUNCS = {
    0x00: ("ADD", "ab", 0), 0x01: ("SUB", "ab", 0), 0x02: ("MUL", "ab", 0), 0x03: ("MAX", "ab", 0),
    0x04: ("MULC", "ac", 0), 0x05: ("ADDC", "ac", 0), 0x06: ("SILU", "a", 1), 0x07: ("SWIGLU", "ab", 1),
    0x08: ("EXP", "a", 0), 0x09: ("COPY", "a", 0), 0x0A: ("MIN", "ab", 0), 0x0B: ("SUBC", "ac", 0),
    0x0C: ("MULCB", "abc", 0), 0x0D: ("MULCADD", "abc", 0), 0x0E: ("ADDMULC", "abc", 0),
    0x0F: ("EXPSUB", "ab", 0), 0x10: ("EXPSUBC", "ac", 0), 0x11: ("SIGMOID", "a", 1), 0x12: ("GELU", "a", 1),
    0x13: ("GEGLU", "ab", 1), 0x14: ("TANH", "a", 1), 0x15: ("RECIP", "a", 1), 0x16: ("SETC", "c", 0),
    0x17: ("GELU16", "a", 1), 0x18: ("GEGLU16", "ab", 1),
}
VOP_BY_NAME = {v[0]: k for k, v in VOP_FUNCS.items()}
SOP_SUBS = {0: "MOVI", 1: "ADD", 2: "MUL", 3: "ADDI", 4: "MULI", 5: "SQRT", 6: "RECIP", 7: "LDS", 8: "STS",
            9: "SUB", 10: "DIV", 11: "RSQRT", 12: "MAX", 13: "MIN", 14: "EXP"}
# fields an SOP sub uses: sd sa sb imm addr (tsr_sop_uses)
SOP_USES = {0: "d i", 1: "d a b", 2: "d a b", 3: "d a i", 4: "d a i", 5: "d a", 6: "d a", 7: "d m", 8: "a m",
            9: "d a b", 10: "d a b", 11: "d a", 12: "d a b", 13: "d a b", 14: "d a"}


def L(n: int) -> int:
    return (n + 15) // 16


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & M32))[0]


# ---------------------------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------------------------
def enc_vop(func, d, a=0, b=0, c=0, m=0, rows=1, ds=0, as_=0, bs=0, cmode=0, si=0):
    w0 = OP_VOP | (cmode & 0xF) << 8 | (func & 0xFF) << 16 | (si & 0xF) << 24
    return [w0, d, a, b, c, m, (rows & 0xFFFF) | (ds & 0xFFFF) << 16, (as_ & 0xFFFF) | (bs & 0xFFFF) << 16]


def enc_red(kind, d, a, m, rows=1, as_=0, to_s=0, sd=0, post=0, sk=0, se=0):
    w0 = OP_RED | (kind & 0xF) << 8 | (to_s & 1) << 12 | (sd & 0xF) << 16 | (post & 3) << 20 | \
        (sk & 0xF) << 24 | (se & 0xF) << 28
    return [w0, d, a, 0, 0, m, rows & 0xFFFF, as_ & 0xFFFF]


def enc_rope(mode, d, src, pos, hd, nd, nh, T=1, table=0, ss=0, ds=0, ff=0, fs=0, atab=0, sm=0, sts=0, sfs=0):
    w0 = OP_ROPE | (mode & 0xF) << 8 | (ff & 1) << 12 | (fs & 1) << 13 | (atab & 1) << 14 | (sm & 0xF) << 16 | \
        (sts & 0xF) << 20 | (sfs & 0xF) << 24
    return [w0, d, src, pos, table, (hd & 0xFFFF) | (nd & 0xFFFF) << 16, (nh & 0xFFFF) | (T & 0xFFFF) << 16,
            (ss & 0xFFFF) | (ds & 0xFFFF) << 16]


def enc_topk(k, d, a, m, rows=1, ds=0, as_=0):
    return [OP_TOPK | (k & 0x1F) << 16, d, a, 0, 0, m, (rows & 0xFFFF) | (ds & 0xFFFF) << 16, as_ & 0xFFFF]


def enc_sop(sub, sd=0, sa=0, sb=0, imm=0, addr=0):
    return [OP_SOP | (sub & 0xF) << 8, (sd & 0xF) | (sa & 0xF) << 4 | (sb & 0xF) << 8, imm & M32, addr & M32,
            0, 0, 0, 0]


def enc_gdn(sa, sb, d, a, k, q, v, o, m, rows):
    """GDN 0x39 (VU §7.12.4): alpha in s[sa], beta in s[sb]; S' tile d, S tile a, vectors k q v, o; m, rows."""
    return [OP_GDN | (sa & 0xF) << 16 | (sb & 0xF) << 20, d, a, k, q, v, o, (m & 0xFFFF) | (rows & 0xFFFF) << 16]


def enc_dqrow(fmt, d, src, K, rows=1, ss=0, ds=0):
    return [OP_DQROW | (fmt & 0xF) << 8, d, src, 0, 0, K & M32, rows & 0xFFFF, (ss & 0xFFFF) | (ds & 0xFFFF) << 16]


def enc_asmax(dst, src, mvec, n_lines, R, scale, stride=0):
    return [OP_ASMAX, dst, src, mvec, (n_lines & 0xFFFF) | (stride & 0xFFFF) << 16, R, scale & M32, 0]


def enc_amask(dst, src, mask, n_lines, R, rpq, mstride, mdst, stride=0):
    return [OP_AMASK, dst, src, mask, (n_lines & 0xFFFF) | (stride & 0xFFFF) << 16,
            (R & 0xFFFF) | (rpq & 0xFFFF) << 16, mstride & M32, mdst]


def enc_amerge(orun, parts, mk, st, R, dcode, scale, cstride=0, init=0):
    return [OP_AMERGE | (init & 1) << 12, orun, parts, mk, st, (R & 0xFFFF) | (dcode & 3) << 16, scale & M32,
            cstride & M32]


def enc_anorm(out, orun, lrun, R, dcode):
    return [OP_ANORM, out, orun, lrun, (R & 0xFFFF) | (dcode & 3) << 16, 0, 0, 0]


# ---------------------------------------------------------------------------------------------
# Random ops for the checker (edge-heavy; legal and illegal)
# ---------------------------------------------------------------------------------------------
class ChkGen:
    """Random VU-class words: shapes near the limits, every enumeration value, unused fields set,
    reserved bits flipped one at a time (the simulator decides what is legal)."""

    def __init__(self, spm_lines: int, k_max: int, seed: int):
        self.r = random.Random(seed)
        self.Ls = spm_lines
        self.k_max = k_max

    def line(self, ext=0):
        r = self.r
        x = r.random()
        if x < 0.3:
            return max(0, self.Ls - ext + r.randint(-2, 2)) & 0xFFFFFF
        if x < 0.4:
            return r.choice([0, self.Ls, self.Ls - 1, 0xFFFFFF, 1 << 23])
        return r.randrange(max(1, self.Ls))

    def n16(self, typical):
        r = self.r
        return r.choice([typical, typical, 0, 1, 2, 0xFFFF, r.getrandbits(16), typical + 1, max(0, typical - 1)]) & 0xFFFF

    def vop(self):
        r = self.r
        func = r.choice(list(VOP_FUNCS) * 6 + [0x19, 0x1F, 0xFF, r.getrandbits(8)])
        reads = VOP_FUNCS.get(func, ("", "abc", 0))[1]
        cmode = (2 if r.random() < 0.5 else 1) if "c" in reads else 0
        if r.random() < 0.08:
            cmode = r.randint(0, 15)
        m = r.choice([0, 1, 15, 16, 17, 255, 256, 896, 5120, r.randint(1, 4096), r.getrandbits(24), 0xFFFFFF])
        rows = self.n16(r.choice([1, 2, 16, 40]))
        Lm = L(m)
        ds = r.choice([0, Lm, Lm + 1, max(0, Lm - 1), r.getrandbits(16)]) & 0xFFFF
        as_ = r.choice([0, Lm, r.getrandbits(16)]) if "a" in reads or r.random() < 0.05 else 0
        bs = r.choice([0, Lm, r.getrandbits(16)]) if "b" in reads or r.random() < 0.05 else 0
        ext = (max(rows, 1) - 1) * ds + Lm
        a = self.line(ext) if "a" in reads or r.random() < 0.05 else 0
        b = self.line(ext) if "b" in reads or r.random() < 0.05 else 0
        c = self.line(L(rows)) if cmode == 2 or r.random() < 0.05 else 0
        si = r.randrange(16) if cmode == 1 or r.random() < 0.05 else 0
        return enc_vop(func, self.line(ext), a, b, c, m, rows, ds, as_ & 0xFFFF, bs & 0xFFFF, cmode, si)

    def red(self):
        r = self.r
        kind = r.choice([0, 1, 2] * 4 + [3, 15])
        to_s = r.randint(0, 1)
        post = r.choice([0, 0, 1, 2, 2, 3]) if kind != 2 or r.random() < 0.1 else 0
        rows = 1 if to_s and r.random() < 0.8 else self.n16(r.choice([1, 16, 32, 40]))
        m = r.choice([0, 1, 7, 8, 9, 16, 127, 128, 129, 256, 896, r.randint(1, 4096), r.getrandbits(24)])
        as_ = r.choice([0, L(m), L(m) + 3, r.getrandbits(16)]) & 0xFFFF
        sd = r.randrange(16) if to_s or r.random() < 0.05 else 0
        d = 0 if to_s and r.random() < 0.9 else self.line(L(rows))
        sk, se = (r.randrange(16), r.randrange(16)) if post == 2 or r.random() < 0.05 else (0, 0)
        return enc_red(kind, d, self.line((max(rows, 1) - 1) * as_ + L(m)), m, rows, as_, to_s, sd, post, sk, se)

    def rope(self):
        r = self.r
        mode = r.choice([0, 1, 1, 1, 2])
        atab = int(r.random() < 0.2)
        ff = int(r.random() < 0.3) if not atab or r.random() < 0.1 else 0
        fs = int(r.random() < 0.3) if not atab or r.random() < 0.1 else 0
        hd = r.choice([16, 64, 128, 256, 512, 1024, 0, 24, 1040, r.randint(0, 1100)])
        nd = r.choice([hd, hd // 2, 32, 16, 48, 0, hd + 32, r.randint(0, 1100)]) & 0xFFFF
        nh = self.n16(r.choice([1, 2, 14, 16, 48]))
        T = self.n16(r.choice([1, 2, 3, 17]))
        H = nh * hd // 16
        ss = r.choice([0, 0, H, H + 1, r.getrandbits(16)]) & 0xFFFF
        ds = r.choice([0, 0, H, max(0, H - 1), r.getrandbits(16)]) & 0xFFFF
        sts = r.randrange(16) if not atab or r.random() < 0.1 else 0
        sfs = r.randrange(16) if fs or r.random() < 0.05 else 0
        table = self.line(L(nd) // 2 + 1) if ff or atab or r.random() < 0.05 else 0
        ext = (max(T, 1) - 1) * (ds or H) + H
        return enc_rope(mode, self.line(ext), self.line(ext), self.line(L(T)), hd, nd, nh, T, table, ss, ds, ff, fs,
                        atab, r.randrange(16), sts, sfs)

    def topk(self):
        r = self.r
        k = r.choice(list(range(0, 18)) + [31])
        m = r.choice([0, 1, 16, 100, 262144 & 0xFFFFFF, r.getrandbits(24)])
        rows = self.n16(r.choice([1, 2, 8]))
        kl = (k + 7) // 8
        ds = r.choice([0, kl, max(0, kl - 1), r.getrandbits(16)]) & 0xFFFF
        as_ = r.choice([0, L(m), r.getrandbits(16)]) & 0xFFFF
        return enc_topk(k, self.line((max(rows, 1) - 1) * (ds or kl) + kl), self.line(L(m)), m, rows, ds, as_)

    def sop(self):
        r = self.r
        sub = r.choice(list(range(15)) * 3 + [15])
        uses = SOP_USES.get(sub, "d a b i m").split()
        f = lambda key, gen: gen() if key in uses or r.random() < 0.04 else 0   # noqa: E731
        wl = self.Ls * 16
        addr = f("m", lambda: r.choice([0, wl - 1, wl, wl + 1, r.randrange(wl), (1 << 28) - 1]))
        return enc_sop(sub, f("d", lambda: r.randrange(16)), f("a", lambda: r.randrange(16)),
                       f("b", lambda: r.randrange(16)), f("i", lambda: r.getrandbits(32)), addr)

    def dqrow(self):
        r = self.r
        fmt = r.choice([0, 1, 0, 1, 2, 15])
        bs = 256 if fmt == 1 else 32
        K = r.choice([bs * r.randint(1, max(1, self.k_max // bs)), self.k_max, self.k_max + bs, 0, bs + 16,
                      896, 4864, r.getrandbits(32)]) & M32
        rows = self.n16(r.choice([1, 2, 4]))
        Lk = L(K) & 0xFFFF
        ds = r.choice([0, Lk, max(0, Lk - 1), r.getrandbits(16)]) & 0xFFFF
        ss = r.choice([0, r.getrandbits(16)]) & 0xFFFF
        return enc_dqrow(fmt, self.line((max(rows, 1) - 1) * (ds or Lk) + Lk), self.line(64), K, rows, ss, ds)

    def at_line(self, ext=0):
        r = self.r
        return r.choice([self.line(ext), self.line(ext), (1 << 31) | r.getrandbits(8), M32, self.Ls])

    def scale(self):
        r = self.r
        return r.choice([0x3F800000, 0x3DB504F3, 0, 0x80000000, 0xBF800000, 0x7F800000, 0x7FC00000, 0x00400000,
                         0x00800000, r.getrandbits(32)])

    def asmax(self):
        r = self.r
        nl = r.choice([1, 2, 8, 4095, 4096, 0, r.getrandbits(16)])
        R = r.choice([1, 2, 16, 40, 1024, 1025, 0, r.getrandbits(32)])
        st = r.choice([0, nl, nl + 1, r.getrandbits(16)]) & 0xFFFF
        e = (max(min(R, 1024), 1) - 1) * (st or nl) + nl
        return enc_asmax(self.at_line(e), self.at_line(e), self.at_line(L(min(R, 1024))), nl & 0xFFFF,
                         R & M32, self.scale(), st)

    def amask(self):
        r = self.r
        nl = r.choice([1, 2, 9, 4095, 4096, 0])
        R = r.choice([1, 2, 16, 40, 1024, 1025, 0])
        rpq = r.choice([1, 2, 7, 0, 1000, 0xFFFF])
        st = r.choice([0, nl, r.getrandbits(16)]) & 0xFFFF
        ms = r.choice([0, nl, nl + 1, r.getrandbits(32), r.getrandbits(12)])
        e = (max(R, 1) - 1) * (st or nl) + nl
        return enc_amask(self.at_line(e), self.at_line(e), self.at_line(64), nl, R, rpq, ms, self.at_line(L(R)), st)

    def amerge(self):
        r = self.r
        R = r.choice([1, 2, 16, 40, 1024, 1025, 0])
        dcode = r.randrange(4)
        rl = R * (64 << dcode) // 16
        cs = r.choice([0, rl, rl + 5, r.getrandbits(20), r.getrandbits(32)])
        return enc_amerge(self.at_line(rl), self.at_line(5 * rl), self.at_line(L(R)), self.at_line(2 * L(R)), R,
                          dcode, self.scale(), cs, r.randint(0, 1))

    def anorm(self):
        r = self.r
        R = r.choice([1, 2, 16, 40, 1024, 1025, 0])
        dcode = r.randrange(4)
        rl = R * (64 << dcode) // 16
        return enc_anorm(self.at_line(rl), self.at_line(rl), self.at_line(L(R)), R, dcode)

    def op(self):
        r = self.r
        k = r.choices(["vop", "red", "rope", "topk", "sop", "dqrow", "asmax", "amask", "amerge", "anorm"],
                      [5, 3, 3, 1.5, 2, 1.5, 1, 1, 1, 1])[0]
        w = getattr(self, k)()
        if r.random() < 0.1:
            j = r.randrange(8)
            w[j] ^= 1 << r.randrange(8 if j == 0 else 0, 32)     # never the opcode byte
        return [x & M32 for x in w]

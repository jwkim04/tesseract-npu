"""AT op encodings (docs/design/03-attention-kv.md "AT" §5.1-§5.3) and the RTL payload structs of
rtl/core/tsr_att_pkg.sv (same field order and widths, MSB first) for the tb/at benches.

  enc_kvst / enc_attqk / enc_attpv   the 8 instruction words (generated field positions, tables.py)
  KVST_CMD_T / ATT_CMD_T / ATT_CCMD_T payload StructDefs (tsr_att_pkg)
  kvst_payload / att_payload        the payload tsr_chk_at must build for a passing op
"""
from __future__ import annotations

from common import isa, tables as T
from common.isa import StructDef

OP_KVST, OP_ATTQK, OP_ATTPV = T.OP_KVST, T.OP_ATTQK, T.OP_ATTPV
M32 = 0xFFFFFFFF

KVST_CMD_T = StructDef("tsr_kvst_cmd_t", (("kv", 1), ("n_rows", 7), ("src", 32), ("stride", 32), ("base", 32),
                                          ("e64", 9), ("lg", 16), ("c0", 32)))
ATT_CMD_T = StructDef("tsr_att_cmd_t", (("local_slot", 2), ("pv", 1), ("flags", 3), ("e64", 9), ("dcode", 2), ("g_m1", 6), ("hg0", 6),
                                        ("gs_m1", 3), ("tq_m1", 3), ("dst", 32), ("x", 32), ("base", 32),
                                        ("rs", 15), ("u0", 16), ("n", 12), ("aux0", 16), ("aux1", 16),
                                        ("w7", 32)))
ATT_CCMD_T = StructDef("tsr_att_ccmd_t", (("c", ATT_CMD_T.width), ("nseg", 9), ("n_head", 15), ("r", 18),
                                          ("tact", 4), ("prows", 8), ("q0", 16), ("m0", 4), ("qn", 17),
                                          ("mn", 4), ("rl", 14)))


def qa32_lines(k: int) -> int:
    return k // 64 + (k + 511) // 512


def enc_kvst(kv, n_rows, src, base, e64, lg, c0, stride=0, rsv=None):
    w = [0] * 8
    w[0] = OP_KVST | (kv << 8) | (((n_rows - 1) & 0x3F) << 16)
    w[1] = src & M32
    w[2] = base & M32
    w[3] = (e64 & 0x1FF) | ((lg & 0xFFFF) << 16)
    w[4] = c0 & M32
    w[5] = stride & M32
    if rsv:
        for i, m in rsv:
            w[i] |= m
    return w


def _att_w4(e64, dcode, tq, gs, rs):
    return (e64 & 0x1FF) | ((dcode & 3) << 9) | (((tq - 1) & 7) << 11) | (((gs - 1) & 7) << 14) | ((rs & 0x7FFF) << 17)


def enc_attqk(*, dst, x, base, e64, dcode, G, hg0, GS, tq, u0, n, c_lo, c_hi, rs=0, mdst=0, causal=0, slide=0,
              maxout=0, fmt=0):
    w = [0] * 8
    w[0] = OP_ATTQK | ((fmt & 0xF) << 8) | (causal << 12) | (slide << 13) | (maxout << 14) | \
        (((G - 1) & 0x3F) << 16) | ((hg0 & 0x3F) << 22)
    w[1], w[2], w[3] = dst & M32, x & M32, base & M32
    w[4] = _att_w4(e64, dcode, tq, GS, rs)
    w[5] = (u0 & 0xFFFF) | ((n & 0xFFFF) << 16)
    w[6] = (c_lo & 0xFFFF) | ((c_hi & 0xFFFF) << 16)
    w[7] = mdst & M32
    return w


def enc_attpv(*, dst, p, base, e64, dcode, G, hg0, GS, tq, u0, n, LG, rs=0, cstride=0, lsum=0, fmt=0):
    w = [0] * 8
    w[0] = OP_ATTPV | ((fmt & 0xF) << 8) | (lsum << 12) | (((G - 1) & 0x3F) << 16) | ((hg0 & 0x3F) << 22)
    w[1], w[2], w[3] = dst & M32, p & M32, base & M32
    w[4] = _att_w4(e64, dcode, tq, GS, rs)
    w[5] = (u0 & 0xFFFF) | ((n & 0xFFFF) << 16)
    w[6] = LG & 0xFFFF
    w[7] = cstride & M32
    return w


def kvst_payload(w) -> int:
    e64 = w[3] & 0x1FF
    qa = e64 + (e64 + 7) // 8
    return KVST_CMD_T.pack(kv=(w[0] >> 8) & 1, n_rows=((w[0] >> 16) & 0x3F) + 1, src=w[1],
                           stride=w[5] if w[5] else qa, base=w[2], e64=e64, lg=w[3] >> 16, c0=w[4])


def att_shape(w):
    """Decoded ATTQK / ATTPV fields (AT §5 table)."""
    e64 = w[4] & 0x1FF
    d = dict(e64=e64, E=64 * e64, dcode=(w[4] >> 9) & 3, tq=((w[4] >> 11) & 7) + 1, GS=((w[4] >> 14) & 7) + 1,
             rs=w[4] >> 17, G=((w[0] >> 16) & 0x3F) + 1, hg0=(w[0] >> 22) & 0x3F, u0=w[5] & 0xFFFF, n=w[5] >> 16,
             flags=(w[0] >> 12) & 7)
    d["D"] = 64 << d["dcode"]
    d["nseg"] = d["E"] // d["D"]
    d["n_head"] = d["G"] * d["nseg"]
    d["R"] = d["tq"] * d["n_head"]
    d["tact"] = d["tq"] * d["GS"]
    d["prows"] = d["nseg"] * d["tact"]
    return d


def att_payload(w) -> int:
    op = w[0] & 0xFF
    s = att_shape(w)
    pv = int(op == OP_ATTPV)
    rs = s["rs"] if s["rs"] else s["n"]
    if pv:
        rl = s["R"] * s["D"] // 16
        w7 = w[7] if w[7] else rl
        aux1 = 0
    else:
        w7 = w[7]
        aux1 = w[6] >> 16
    return ATT_CMD_T.pack(pv=pv, flags=(s["flags"] & 1) if pv else s["flags"], e64=s["e64"], dcode=s["dcode"],
                          g_m1=s["G"] - 1, hg0=s["hg0"], gs_m1=s["GS"] - 1, tq_m1=s["tq"] - 1, dst=w[1], x=w[2],
                          base=w[3], rs=rs & 0x7FFF, u0=s["u0"], n=s["n"] & 0xFFF, aux0=w[6] & 0xFFFF, aux1=aux1,
                          w7=w7 & M32)


def att_ccmd(w, num_ch) -> int:
    """tsr_att_ccmd_t that tsr_mvm sends the clusters for a passing ATTQK / ATTPV."""
    s = att_shape(w)
    u0, n = s["u0"], s["n"]
    return ATT_CCMD_T.pack(c=att_payload(w), nseg=s["nseg"], n_head=s["n_head"], r=s["R"], tact=s["tact"],
                           prows=s["prows"], q0=u0 // num_ch, m0=u0 % num_ch, qn=(u0 + n) // num_ch,
                           mn=(u0 + n) % num_ch, rl=(s["R"] + 15) // 16)


def payload(w) -> int:
    op = w[0] & 0xFF
    return kvst_payload(w) if op == OP_KVST else att_payload(w)


__all__ = ["OP_KVST", "OP_ATTQK", "OP_ATTPV", "KVST_CMD_T", "ATT_CMD_T", "ATT_CCMD_T", "enc_kvst", "enc_attqk",
           "enc_attpv", "kvst_payload", "att_payload", "att_ccmd", "att_shape", "payload", "qa32_lines", "isa"]

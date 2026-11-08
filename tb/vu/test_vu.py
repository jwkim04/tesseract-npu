"""tb/vu — the vector unit against the simulator (docs/design/02-vector-unit.md "VU" §10 item 4; SW §10.3).

Tests (pytest functions at the end of the file):
  test_vu_chk_*      tsr_vu_chk (the VU checker class, rtl/core/tsr_vu_chk.sv) against the simulator's
                     VU checks (sw/sim/tsr_vu.c chk_*, through tsr_sim_accept_op + the handler's check):
                     code, ERR_INFO and the SPM extents of random and edge-heavy DQROW / VOP / RED /
                     ROPE / TOPK / SOP words, and the command payload (the words unchanged).
The arithmetic sweeps (Verilator C++ against libtsr_ref), the lane and the whole-unit benches are in
tb/vu/test_vu_arith.py and tb/vu/test_vu_unit.py.
"""
from __future__ import annotations

import json
import os
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
import vu_ops as V                                        # noqa: E402

M32 = 0xFFFFFFFF
CAPS_VU, CAPS_TOPK, CAPS_DQROW, CAPS_ATT, CAPS_GDN = 0x1, 0x8, 0x40, 0x20, 0x100


def chk_sources():
    d = os.environ.get("TSR_MUT_DIR")
    f = Path(d) / "tsr_vu_chk.sv" if d and (Path(d) / "tsr_vu_chk.sv").exists() else REPO / "rtl/core/tsr_vu_chk.sv"
    return [REPO / "rtl/core/tsr_tables_pkg.sv", REPO / "rtl/core/tsr_pkg.sv",
            REPO / "rtl/core/tsr_fp_pkg.sv", REPO / "rtl/core/tsr_att_pkg.sv",
            REPO / "rtl/core/tsr_xc_pkg.sv", f]


GDN = json.loads(os.environ.get("TSR_VU_CHK_CFG", "{}")).get("gdn", 0)


def chk_cfg():
    c = json.loads(os.environ.get("TSR_VU_CHK_CFG", "{}"))
    return S.config(num_ch=c.get("num_ch", 4), t_max=8, k_max=c.get("k_max", 32768),
                    spm_bytes=c.get("spm_bytes", 4 << 20),
                    caps=CAPS_VU | CAPS_TOPK | CAPS_DQROW | CAPS_ATT | (CAPS_GDN if c.get("gdn") else 0)), \
        c.get("n_ops", 3000)


def directed_chk_ops(cfg, gdn=False):
    """Edge cases the random stream may miss: extents at the SPM end, 32-bit products, every VU §5.9
    OPERAND row, SOP word addresses at the limit."""
    Ls = cfg["spm_bytes"] // 64
    W = cfg["spm_bytes"] // 4
    v = []
    v.append(V.enc_vop(0x00, Ls - 56, 0, 0, 0, 896, 1))                 # d + L(896) = Ls exactly: a at 0
    v.append(V.enc_vop(0x00, Ls - 55, 0, 0, 0, 896, 1))                 # one past: SPM_RANGE d
    v.append(V.enc_vop(0x04, 0, Ls - 1, 0, 0, 16, 1, cmode=1, si=3))    # a in range (1 line)
    v.append(V.enc_vop(0x04, 0, Ls, 0, 0, 16, 1, cmode=1, si=3))        # a out of range
    v.append(V.enc_vop(0x0C, 0, 100, 200, Ls - 3, 16, 40, 1, 1, 0, cmode=2))   # c: ceil(40/16) = 3 lines
    v.append(V.enc_vop(0x0C, 0, 100, 200, Ls - 2, 16, 40, 1, 1, 0, cmode=2))
    v.append(V.enc_vop(0x00, 0, 0, 0, 0, 0xFFFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF))   # 2^32-scale extents
    v.append(V.enc_vop(0x00, 0, 0, 0, 0, 17, 2, 1))                     # d_stride 1 < L 2: OPERAND
    v.append(V.enc_vop(0x09, 0, 0, 0, 0, 0, 0))                         # m = 0, rows = 0
    v.append(V.enc_red(1, 0, 0, 896, 1, 0, to_s=1, sd=1, post=2, sk=2, se=3))
    v.append(V.enc_red(1, 0, 0, 896, 2, 56, to_s=1, sd=1, post=2, sk=2, se=3))  # TO_S rows 2: OPERAND
    v.append(V.enc_red(2, 5, 0, 0, 0, 0))                               # rows 0, m 0
    v.append(V.enc_red(0, Ls - 3, 0, 16, 40, 1))                        # d: 3 lines at the end
    v.append(V.enc_red(0, Ls - 2, 0, 16, 40, 1))
    for hd, nd, mode in ((64, 64, 1), (128, 64, 1), (64, 16, 0), (24, 16, 0), (1040, 32, 1), (0, 0, 1), (64, 48, 1),
                         (64, 80, 1), (64, 8, 0), (1024, 1024, 1)):
        v.append(V.enc_rope(mode, 0, 4096, 8192, hd, nd, 16, 3, sm=1, sts=2))
    v.append(V.enc_rope(1, 0, 4096, 8192, 64, 64, 16, 3, ds=63, sm=1, sts=2))     # d_stride' < H: OPERAND
    v.append(V.enc_rope(1, 0, 4096, 8192, 64, 64, 0, 3, ds=1, sm=1, sts=2))       # n_heads 0: no extents
    v.append(V.enc_rope(1, 0, 4096, 8192, 512, 512, 16, 1, table=9000, ff=1, sm=1, sts=2))
    v.append(V.enc_rope(1, 0, 4096, 8192, 512, 512, 16, 1, table=9000, atab=1, sm=1, sts=0))
    v.append(V.enc_rope(1, 0, 4096, 8192, 512, 512, 16, 1, table=9000, atab=1, ff=1, sm=1))   # ATAB + FF: ILLEGAL
    v.append(V.enc_rope(1, 0, 4096, 8192, 512, 512, 16, 1, table=9000, sm=1, sts=2))         # table w/o FF: ILLEGAL
    for k in (0, 1, 8, 9, 16, 17):
        v.append(V.enc_topk(k, 0, 100, 1000, 3, 0, 63))
    v.append(V.enc_topk(16, 0, 100, 1000, 3, 1, 63))                   # d_stride 1 < 2: OPERAND
    for addr in (0, W - 1, W, W + 1):
        v.append(V.enc_sop(7, sd=1, addr=addr))
        v.append(V.enc_sop(8, sa=1, addr=addr))
    v.append(V.enc_sop(0, sd=1, imm=0x3F800000))
    v.append(V.enc_sop(15))
    for K in (0, 32, 48, 256, cfg["k_max"], cfg["k_max"] + 256):
        v.append(V.enc_dqrow(0, 0, 1000, K, 2))
        v.append(V.enc_dqrow(1, 0, 1000, K, 2))
    v.append(V.enc_dqrow(2, 0, 1000, 256, 2))
    v.append(V.enc_dqrow(0, 0, 1000, 256, 2, 0, 1))                     # d_stride 1 < 16: OPERAND
    # AT's vector ops (AT §5.8): limits of n_lines, R, scale, rpq; the extents at the SPM end
    one = 0x3F800000
    for nl, R in ((1, 1), (4095, 1), (4096, 1), (0, 1), (1, 1024), (1, 1025), (1, 0), (2, 40)):
        v.append(V.enc_asmax(0, 64, Ls - 3, nl, R, one))
        v.append(V.enc_amask(0, 64, 128, nl, R, 4, 2, Ls - 3))
    for sc in (0, 0x80000000, 0xBF800000, 0x7F800000, 0x7FC00000, 0x00400000, 0x00800000, 0x7F7FFFFF):
        v.append(V.enc_asmax(0, 64, 128, 1, 16, sc))
        v.append(V.enc_amerge(0, 4096, 8192, 8200, 40, 1, sc))
    v.append(V.enc_amask(0, 64, 128, 1, 16, 0, 2, 300))                  # rpq 0: OPERAND (0)
    v.append(V.enc_amask(0, 64, Ls - 4, 2, 40, 7, 1, 300))                # mask rows ceil(40/7) = 6: 5 + 2 lines
    v.append(V.enc_amask(0, 64, Ls - 6, 2, 40, 7, 1, 300))
    for R, dcode in ((1, 0), (40, 1), (1024, 3), (1025, 0), (0, 2)):
        v.append(V.enc_amerge(0, 4096, 8192, 8200, R, dcode, one, 0, 0))
        v.append(V.enc_amerge(0, 4096, 8192, 8200, R, dcode, one, 0, 1))
        v.append(V.enc_anorm(0, 4096, 8192, R, dcode))
    v.append(V.enc_amerge(Ls - 40 * 8, 0, 9000, 9010, 40, 1, one))       # Orun at the SPM end
    v.append(V.enc_amerge(Ls - 40 * 8 + 1, 0, 9000, 9010, 40, 1, one))
    v.append(V.enc_amerge(0, Ls - 4 * 1000 - 4 * 3, 9000, 9010, 40, 1, one, 1000))   # parts C·cs + C·ceil(R/16)
    v.append(V.enc_amerge(0, Ls - 4 * 1000 - 4 * 3 + 1, 9000, 9010, 40, 1, one, 1000))
    v.append(V.enc_anorm(0, 4096, Ls - 3, 40, 1))
    v.append(V.enc_anorm(0, 4096, Ls - 2, 40, 1))
    v.append(V.enc_asmax(1 << 31, 64, 128, 1, 1, one))                    # a 32-bit line address
    if gdn:
        # GDN (VU §7.12.4): m and rows at their limits, d / a / o overlaps, each range operand at the SPM end
        base = (3, 5, 40000, 20000, 100, 200, 300, 400)
        g = lambda **kw: V.enc_gdn(*[kw.get(n, b) for n, b in zip(("sa", "sb", "d", "a", "k", "q", "v", "o"), base)],  # noqa: E731
                                   kw.get("m", 128), kw.get("rows", 20))
        v.append(g())
        v.append(g(d=20000))                                         # in place
        for m in (0, 16, 32, 48, 63, 64, 72, 128, 240, 256, 272, 0xFFF0, 0xFFFF):
            v.append(g(m=m))
        for rows in (0, 1, 1023, 1024, 1025, 0xFFFF):
            v.append(g(rows=rows, m=64))
        v.append(g(m=72, rows=0))                                   # m is checked before rows
        for d in (20000 - 160, 20000 - 159, 20000 - 1, 20000 + 1, 20000 + 159, 20000 + 160):
            v.append(g(d=d))                                         # tile 160 lines
        for o in (40000, 40000 + 159, 40000 + 160, 20000 + 159, 20000 - 1, 20000 - 2, 20000 - 1 + 160):
            v.append(g(o=o))
        for name in ("d", "a", "k", "q", "v", "o"):
            for off in (0, 1, 2, 8, 9, 160, 161):
                v.append(g(**{name: Ls - off}))
        v.append(g(d=Ls - 160, a=20000, v=Ls - 2, o=Ls - 1))
        v.append(g(d=Ls - 160, a=Ls - 320, k=Ls - 8, q=Ls - 16, v=Ls - 2, o=Ls - 3))
    return [[x & M32 for x in w] for w in v]


async def chk_run(dut, cfg, ops):
    sim = S.Sim(cfg, (8, 4, 4, 16, 0, 4))
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.req_valid.value = 0
    dut.req_insn.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    bad, hist = [], {}
    for w in ops:
        unit, code, info, exts = sim.check(w)
        assert unit == 3, "not a unit-3 op: %s" % [hex(x) for x in w]
        hist[code] = hist.get(code, 0) + 1
        dut.req_insn.value = isa.insn_int(w)
        dut.req_valid.value = 1
        n = 0
        while True:
            await FallingEdge(dut.clk)
            n += 1
            if int(dut.rsp_valid.value):
                break
            assert n < 50, "no rsp_valid"
        got = dict(code=int(dut.rsp_code.value), info=int(dut.rsp_info.value), ext=E.unpack_ext(int(dut.rsp_ext.value)))
        exp = dict(code=code, info=info, ext=E.sim_ext_to_rtl(exts)[:4])
        if got != exp:
            bad.append("op %s: RTL %s, simulator %s" % ([hex(x) for x in w], got, exp))
        if int(dut.rsp_cmd.value) != isa.insn_int(w):
            bad.append("op %s: command payload is not the words" % [hex(x) for x in w])
        await RisingEdge(dut.clk)
        dut.req_valid.value = 0                       # tsr_chk drops req_valid after the answer
        await RisingEdge(dut.clk)
        if len(bad) > 20:
            break
    sim.close()
    dut._log.info("%d ops, codes %s", len(ops), dict(sorted(hist.items())))
    return bad


@cocotb.test()
async def vu_chk_directed(dut):
    cfg, _ = chk_cfg()
    bad = await chk_run(dut, cfg, directed_chk_ops(cfg, GDN))
    for b in bad[:20]:
        dut._log.error(b)
    assert not bad, "%d mismatches" % len(bad)


@cocotb.test()
async def vu_chk_random(dut):
    cfg, n = chk_cfg()
    g = V.ChkGen(cfg["spm_bytes"] // 64, cfg["k_max"], int(os.environ.get("TSR_SEED", "1")), GDN)
    bad = await chk_run(dut, cfg, [g.op() for _ in range(n)])
    for b in bad[:20]:
        dut._log.error(b)
    assert not bad, "%d mismatches" % len(bad)


def _run_chk(name, spm_bytes, k_max, n_ops=3000, num_ch=4, gdn=0):
    c = dict(spm_bytes=spm_bytes, k_max=k_max, n_ops=n_ops, num_ch=num_ch, gdn=gdn)
    run("tsr_vu_chk", __name__, sources=chk_sources(), name="vu_chk_" + name,
        extra_env={"TSR_VU_CHK_CFG": json.dumps(c)},
        parameters=dict(SPM_BYTES=spm_bytes, K_MAX=k_max, NUM_CH=num_ch, HAS_GDN=gdn))


def test_vu_chk_u200():
    """U200: 4 MiB SPM, K_MAX 32768, 4 channels (AMERGE's C)."""
    _run_chk("u200", 4 << 20, 32768, 6000)


def test_vu_chk_small():
    """64 KiB SPM (1024 lines), K_MAX 4096, 3 channels: more operands at the SPM end."""
    _run_chk("small", 64 << 10, 4096, 4000, num_ch=3)


def test_vu_chk_gdn():
    """With the GDN class (HAS_GDN = 1, CAPS bit 8): GDN words of every shape and overlap beside the other ops, on the
    U200 SPM and on 64 KiB (the six range operands at the SPM end)."""
    _run_chk("gdn_u200", 4 << 20, 32768, 6000, gdn=1)
    _run_chk("gdn_small", 64 << 10, 4096, 4000, num_ch=3, gdn=1)

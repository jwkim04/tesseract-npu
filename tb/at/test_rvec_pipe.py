"""tb/at test_rvec_pipe — the three-stage row-vector emitter (tsr_mvm_rvec PIPE 1, FAST_CLK) against the single-stage one
(PIPE 0): random op descriptions (T_q, GS, G, T_q GS, head offset, line base), a random value table and an independent random
job_ready for each emitter; the jobs {last, line, own, 0, y} and the done pulses must be equal in order and content.

DUT: tb/at/hdl/tsr_tb_rvec.sv. Both emitters run the same op at the same time; PIPE 1 offers its first job 3 cycles later and
done 3 cycles later (docs/platform-u200.md §8.9). The value table does not change while an op runs (the callers' contract).
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

from common.runner import run

HERE = Path(__file__).resolve().parent


async def one_op(dut, rng, spm_aw, cfg, tbl, stall):
    """Run one op on both emitters; returns the job lists and the done counts (bounded wait)."""
    dut.tq.value = cfg["tq"]
    dut.gs.value = cfg["gs"]
    dut.tact.value = cfg["tact"]
    dut.nseg.value = cfg["nseg"]
    dut.g_heads.value = cfg["g_heads"]
    dut.hg0.value = cfg["hg0"]
    dut.n_head.value = cfg["n_head"]
    dut.base.value = cfg["base"]
    dut.last_en.value = cfg["last_en"]
    dut.tbl.value = sum(v << (32 * i) for i, v in enumerate(tbl))
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0
    jobs = [[], []]
    dones = [0, 0]
    cyc_done = [None, None]
    jw = 1 + spm_aw + 32 + 512
    for cyc in range(200000):
        r = 0
        for p in range(2):
            if rng.random() >= stall[p]:
                r |= 1 << p
        dut.job_ready.value = r
        # the registered outputs of this cycle are what the next rising edge sees together with r
        jv = int(dut.job_valid.value)
        dn = int(dut.done.value)
        jb = int(dut.job.value)
        for p in range(2):
            if (jv >> p) & 1 and (r >> p) & 1:
                jobs[p].append((jb >> (p * jw)) & ((1 << jw) - 1))
            if (dn >> p) & 1:
                dones[p] += 1
                cyc_done[p] = cyc
        await FallingEdge(dut.clk)
        if dones[0] and dones[1]:
            break
    # a few quiet cycles: no extra job or done
    for _ in range(8):
        dut.job_ready.value = 3
        await FallingEdge(dut.clk)
        jv = int(dut.job_valid.value)
        dn = int(dut.done.value)
        assert jv == 0 and dn == 0, "activity after done: valid %x done %x" % (jv, dn)
    return jobs, dones, cyc_done


@cocotb.test()
async def rvec_pipe_equal(dut):
    spm_aw = int(dut.SPM_AW.value)
    rng = random.Random(int(__import__("os").environ.get("TSR_SEED", "1"), 0))
    cocotb.start_soon(Clock(dut.clk, 4, unit="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.job_ready.value = 3
    dut.tbl.value = 0
    for _ in range(6):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    nops = int(__import__("os").environ.get("TB_RVEC_OPS", "120"))
    for n in range(nops):
        tq = rng.randint(1, 4)
        gs = rng.choice([1, 1, 2, 3, 4, 8, 15])
        gs = min(gs, 15)
        tact = min(15, tq * gs)
        nseg = rng.choice([1, 1, 2, 3, 5, 16, 40])
        cfg = dict(tq=tq, gs=gs, tact=tact, nseg=nseg,
                   g_heads=rng.choice([gs, gs + rng.randint(0, 20), rng.randint(1, 127)]),
                   hg0=rng.randint(0, 63), n_head=rng.randint(0, 3000) if rng.random() < 0.8 else rng.randint(0, 32767),
                   base=rng.randrange(1 << spm_aw), last_en=rng.randint(0, 1))
        # table entries addressed: e = g tact + q gs + i (mod 128)
        tbl = [rng.getrandbits(32) for _ in range(128)]
        stall = (rng.choice([0.0, 0.0, 0.3, 0.7, 0.95]), rng.choice([0.0, 0.0, 0.3, 0.7, 0.95]))
        jobs, dones, cd = await one_op(dut, rng, spm_aw, cfg, tbl, stall)
        assert dones == [1, 1], ("op %d %r: done counts %r" % (n, cfg, dones))
        def brief(js):
            return [((j >> 544) & ((1 << spm_aw) - 1), (j >> 528) & 0xFFFF, j >> (544 + spm_aw)) for j in js]
        assert jobs[0] == jobs[1], ("op %d %r: jobs differ (%d vs %d)\n%s\n%s" % (
            n, cfg, len(jobs[0]), len(jobs[1]), brief(jobs[0])[:30], brief(jobs[1])[:30]))
        assert jobs[0], "op %d produced no job" % n


def test_rvec_pipe():
    for aw in (16, 11):
        run("tsr_tb_rvec", __name__, parameters={"SPM_AW": aw}, extra_sources=[HERE / "hdl" / "tsr_tb_rvec.sv"],
            name="rvec_pipe_aw%d" % aw, testcase=["rvec_pipe_equal"])


@pytest.mark.parametrize("old,new", [
    ("wire              rd_same = pv_q & (", "wire              rd_same = ~pv_q & ("),                    # same flag inverted
    ("rd_l   = base + SPM_AW'(a_r >> 4);", "rd_l   = base + SPM_AW'(a_r >> 3);"),                         # line address
    ("if (h_last) fin_q <= 1'b1;", "if (h_last && h_same) fin_q <= 1'b1;"),                              # final line
])
def test_rvec_pipe_mutants(old, new):
    """Three mutants of the PIPE 1 emitter that the equivalence test must reject (the flat emitter is the reference)."""
    from common.runner import mutated_sources, rtl_sources
    srcs = mutated_sources(rtl_sources(), "tsr_mvm_att.sv", old, new, "rvec_" + format(zlib.crc32(old.encode()), "08x"))
    with pytest.raises((SystemExit, RuntimeError)):
        run("tsr_tb_rvec", __name__, parameters={"SPM_AW": 16}, sources=srcs, extra_sources=[HERE / "hdl" / "tsr_tb_rvec.sv"],
            name="rvec_pipe_mut_" + format(zlib.crc32(old.encode()), "08x"), testcase=["rvec_pipe_equal"], extra_env={"TB_RVEC_OPS": "40"})

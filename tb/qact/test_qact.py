"""tsr_qact testbench (docs/rtl-core.md §3.3, §8.1; block (e)).

Every QA image is compared bit for bit with tsr_qact() of the reference library (libtsr_ref.so,
through common.tsr_ref), token by token, together with the unit's SPM traffic (each source line
read once and in order, each destination line written once with full byte enables, nothing else
touched), the done timing (one-cycle pulse right after the edge that accepted the last write),
cmd_ready, and the valid/ready rules (R2 on both SPM request streams, R5 by probing). Details of
the vectors and checks: tb/qact/qact_tb.py.

pytest functions:
  test_qact[cfg]        tsr_qact alone with a behavioural SPM client port (exact SPM_RD_LAT, random /
                        bursty ready, garbage on rd_rsp_data between responses):
                          qact_directed    the spec / test_ref_ggml.c special-value vectors, then
                                           every block generator on its own, both kinds, several K
                                           (and T up to T_MAX)
                          qact_random      random commands: kind, K, T, stride gaps, the T = 1
                                           stride field ignored, operands at the SPM ends,
                                           random ready policies, mixed special-value blocks
                          qact_kmax        K = K_MAX for both kinds (T as large as fits)
                          qact_throughput  always-ready SPM: cycles per line / block within the
                                           documented bounds; back-to-back commands with
                                           cmd_valid raised before done
                          qact_reset       synchronous reset in the middle of commands, reset state
                                           of every output, then fully checked commands
                          qact_comb_probe  no combinational input -> output path (R5)
                        configurations: u200 (exactly as tsr_core instantiates it: K_MAX 32768,
                        T_MAX 1, 4 MiB SPM, SPM_RD_LAT 2, default implementation parameters),
                        t8 (T_MAX 8, K_MAX 4096, 256 KiB SPM, SPM_RD_LAT 1, MUL_LAT 5), tight
                        (T_MAX 3, 32 KiB SPM, SPM_RD_LAT 4, MUL_LAT 1, minimum buffer / queues:
                        BUF_LINES 16, AQ_DEPTH 2, PQ_DEPTH 2), k65280 (K_MAX 65280, T_MAX 2,
                        1 MiB SPM, SPM_RD_LAT 3, MUL_LAT 3)
  test_qact_rtl_spm     tb/qact/hdl/tsr_tb_qact.sv: tsr_qact on client 1 of the real tsr_spm, the
                        testbench on client 0 (preload, contention traffic that takes the read and
                        write grants away from QACT, readback); the SPM's own R2 / range
                        assertions and tsr_qact's response-latency assertion are live
  test_qact_time0_edge[build-div]  Verilator's first evaluation may clock submodules with rst_n
                        and all state random (see "First evaluation" below), so every simulation
                        check must be armed by a reset edge. qact_time0_edge / rtl_spm_time0_edge,
                        each alone in a fresh simulation, force the worst case deterministically
                        (spurious rising edge in every clocked instance, rst_n high, check-violating
                        state in tsr_qact / tsr_spm / tsr_fp_div), then reset and run normally.
                        div=busy (tsr_fp_div in_ready high while busy; the 0 ns failure
                        at seeds 5 / 10 that arming tsr_fp_div.sv's assertion with a reset
                        edge fixed); builds u200, tight, rtl_spm (the latter also clocks tsr_qact
                        and tsr_spm themselves through their clk copies)
  test_qact_lint        verilator --lint-only -Wall of tsr_qact (simulation and SYNTHESIS views)
                        in edge configurations, and of the testbench wrapper
  test_qact_ooc_timing  opt-in (Vivado): out-of-context place & route at 300 MHz on xcu200-2,
                        WNS / WHS >= 0 (TSR_QACT_VIVADO=1 or -k ooc)
Seeded-bug check: TSR_QACT_RTL=<modified tsr_qact.sv> runs the bench on another source (the
bench must fail on it).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.handle import Immediate
from cocotb.simtime import get_sim_time
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import isa
from common.runner import REPO, rtl_sources, run
from common.tbutil import CLK_PERIOD_NS
from common.tbutil import rng as tb_rng
from common.tsr_ref import QA_A32, QA_A256, ref

import qact_tb
from qact_tb import QactDriver, Ready, SpmModel, block_size, directed_vectors, gen_vector

MIB = 1024 * 1024
TB_SV = REPO / "tb" / "qact" / "hdl" / "tsr_tb_qact.sv"

DEFAULTS = dict(K_MAX=32768, T_MAX=1, SPM_BYTES=4 * MIB, SPM_RD_LAT=2, MUL_LAT=4, BUF_LINES=64,
                AQ_DEPTH=4, PQ_DEPTH=16)
ALL_TESTS = ["qact_directed", "qact_random", "qact_kmax", "qact_throughput", "qact_reset",
             "qact_comb_probe"]
CFGS = {
    "u200":   (dict(), ALL_TESTS, 250),
    "t8":     (dict(K_MAX=4096, T_MAX=8, SPM_BYTES=256 * 1024, SPM_RD_LAT=1, MUL_LAT=5),
               ["qact_directed", "qact_random", "qact_kmax", "qact_throughput", "qact_reset"], 150),
    "tight":  (dict(K_MAX=2048, T_MAX=3, SPM_BYTES=32 * 1024, SPM_RD_LAT=4, MUL_LAT=1, BUF_LINES=16,
                    AQ_DEPTH=2, PQ_DEPTH=2), ALL_TESTS, 300),
    # the card image of the fourth link: T_MAX 4, SPM_RD_LAT 4 (tb/qact qact_timing)
    "k4":     (dict(T_MAX=4, SPM_RD_LAT=4), ["qact_timing"], 1),
    "k65280": (dict(K_MAX=65280, T_MAX=2, SPM_BYTES=1 * MIB, SPM_RD_LAT=3, MUL_LAT=3),
               ["qact_kmax", "qact_random"], 16),
}


def full_config(over: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(over)
    cfg["LINES"] = cfg["SPM_BYTES"] // 64
    return cfg


def qact_sources(names=("tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_fp_pkg.sv", "tsr_sram_1r1w.sv", "tsr_fp_mul.sv",
                         "tsr_fp_div.sv", "tsr_qact.sv")):
    """The files tsr_qact needs, in filelist order. $TSR_QACT_RTL replaces rtl/core/tsr_qact.sv
    (used to check that the bench catches seeded bugs)."""
    srcs = [s for s in rtl_sources() if s.name in names]
    assert [s.name for s in srcs] == list(names), srcs
    alt = os.environ.get("TSR_QACT_RTL")
    if alt:
        srcs = [Path(alt).resolve() if s.name == "tsr_qact.sv" else s for s in srcs]
    return srcs


# =============================================================================================
# pytest side
# =============================================================================================
def run_cfg(cfg: str, tests=None, extra_env: dict | None = None):
    over, cfg_tests, n_random = CFGS[cfg]
    full = full_config(over)
    full["N_RANDOM"] = n_random
    run("tsr_qact", __name__, parameters=over, sources=qact_sources(), name="qact_" + cfg,
        testcase=tests or cfg_tests,
        extra_env=dict(extra_env or {}, TSR_QACT_CFG=json.dumps(full)))


@pytest.mark.parametrize("cfg", list(CFGS))
def test_qact(cfg):
    run_cfg(cfg)


RTL_SPM_CFG = dict(K_MAX=2048, T_MAX=4, SPM_BYTES=64 * 1024, SPM_RD_LAT=2, BANK_LINES=256)


def run_rtl_spm(tests=("qact_rtl_spm",), extra_env: dict | None = None):
    srcs = qact_sources(("tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_fp_pkg.sv", "tsr_sram_1r1w.sv", "tsr_fp_mul.sv",
                         "tsr_fp_div.sv", "tsr_spm.sv", "tsr_spm_cport.sv", "tsr_qact.sv"))   # + cport
    full = dict(RTL_SPM_CFG)
    full["LINES"] = full["SPM_BYTES"] // 64
    run("tsr_tb_qact", __name__, parameters=RTL_SPM_CFG, sources=srcs, extra_sources=[TB_SV],
        name="qact_rtl_spm", testcase=list(tests),
        extra_env=dict(extra_env or {}, TSR_QACT_CFG=json.dumps(full)))


def test_qact_rtl_spm():
    run_rtl_spm()


# tsr_fp_div.sv's simulation assertion is armed by a reset edge like tsr_spm's / tsr_qact's. Gated
# with rst_n only it fired at 0 ns (seeds 5 / 10); the 'busy' variant reproduces that case
# deterministically, so both variants must pass.
@pytest.mark.parametrize("div", ["idle", "busy"])
@pytest.mark.parametrize("build", ["u200", "tight", "rtl_spm"])
def test_qact_time0_edge(build, div):
    """qact_time0_edge / rtl_spm_time0_edge alone in a fresh simulation (they must see the first
    evaluation); reuses the test_qact / test_qact_rtl_spm builds."""
    env = {"TSR_QACT_T0_DIV": div}
    if build == "rtl_spm":
        run_rtl_spm(["rtl_spm_time0_edge"], extra_env=env)
    else:
        run_cfg(build, ["qact_time0_edge"], extra_env=env)


LINT_CONFIGS = [
    {},
    {"MUL_LAT": 1},
    {"MUL_LAT": 2, "BUF_LINES": 16, "AQ_DEPTH": 2, "PQ_DEPTH": 2},
    {"MUL_LAT": 3, "K_MAX": 256, "SPM_BYTES": 64, "SPM_RD_LAT": 1},
    {"MUL_LAT": 5, "K_MAX": 65280, "T_MAX": 8, "SPM_RD_LAT": 4},
    {"SPM_BYTES": 65536, "K_MAX": 4096, "T_MAX": 2},
    {"SPM_BYTES": 128, "K_MAX": 32768},
    {"SPM_BYTES": 64 * 1000, "BUF_LINES": 256, "AQ_DEPTH": 16, "PQ_DEPTH": 64},
    {"SPM_BYTES": 256 * MIB, "SPM_RD_LAT": 7},
]


def test_qact_lint():
    fl = REPO / "rtl" / "filelist.f"
    fails = []
    runs = [(["--top-module", "tsr_qact"] + ["-G%s=%s" % kv for kv in p.items()], str(p))
            for p in LINT_CONFIGS]
    runs.append((["--top-module", "tsr_tb_qact", str(TB_SV)], "tsr_tb_qact"))
    for args, name in runs:
        for view in ([], ["-DSYNTHESIS"]):
            cmd = ["verilator", "--lint-only", "-Wall", "-F", str(fl)] + view + args
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                fails.append("%s %s:\n%s" % (name, view, p.stderr[-3000:]))
    assert not fails, "\n".join(fails)


VIVADO = Path(os.environ.get("TSR_VIVADO", "/tool/Xilinx/Vivado/2024.2/bin/vivado"))


def test_qact_ooc_timing(tmp_path, request):
    """Opt-in (needs Vivado, ~15 min): out-of-context implementation of tsr_qact (U200 defaults)
    at 300 MHz on xcu200-2 (tb/qact/viv/qact_ooc.tcl); requires WNS >= 0 and WHS >= 0.
    Runs only with TSR_QACT_VIVADO=1 or when selected by -k / node id."""
    selected = os.environ.get("TSR_QACT_VIVADO") == "1" or \
        "ooc" in (request.config.getoption("keyword") or "") or \
        any("test_qact_ooc_timing" in a for a in request.config.args)
    if not selected:
        pytest.skip("opt-in Vivado run: set TSR_QACT_VIVADO=1 or select it with -k ooc")
    if not VIVADO.exists():
        pytest.skip("Vivado not found (%s)" % VIVADO)
    tcl = REPO / "tb" / "qact" / "viv" / "qact_ooc.tcl"
    p = subprocess.run([str(VIVADO), "-mode", "batch", "-nojournal", "-nolog", "-source", str(tcl),
                        "-tclargs", str(REPO), "-", str(tmp_path / "out")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=7200)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("QOOC")]
    print("\n".join(lines))
    assert p.returncode == 0, p.stdout[-3000:]
    wns = [ln for ln in lines if ln.startswith("QOOC wns")]
    assert wns, p.stdout[-3000:]
    _, _, w, _, h = wns[0].split()
    assert float(w) >= 0.0 and float(h) >= 0.0, "300 MHz not met: %s" % wns[0]


# =============================================================================================
# cocotb side
# =============================================================================================
def _cfg() -> dict:
    return json.loads(os.environ["TSR_QACT_CFG"])


async def _setup(dut, salt: int, clock: Clock | None = None):
    cfg = _cfg()
    rnd = tb_rng(salt)
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd.value = 0
    clock = clock or Clock(dut.clk, CLK_PERIOD_NS, unit="ns")
    clock.start()
    spm = SpmModel(dut, rd_lat=cfg["SPM_RD_LAT"], rnd=rnd, lines=cfg["LINES"])
    cocotb.start_soon(spm.run())
    await ClockCycles(dut.clk, 5)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    drv = QactDriver(dut, spm, rnd, cfg)
    return cfg, rnd, spm, drv, clock


def _set_ready(spm: SpmModel, rnd: random.Random, mode: str | None = None):
    mode = mode or rnd.choice(("always", "random", "bursty", "random_slow"))
    if mode == "random_slow":
        spm.rd_ready = Ready(rnd, "random", p=0.3)
        spm.wr_ready = Ready(rnd, "random", p=0.3)
    else:
        spm.rd_ready = Ready(rnd, mode, p=rnd.uniform(0.4, 0.95))
        spm.wr_ready = Ready(rnd, mode, p=rnd.uniform(0.4, 0.95))
    return mode


def _ks(cfg: dict, kind: int, rnd: random.Random, n: int, kmax_fit: int) -> list:
    bs = block_size(kind)
    hi = min(cfg["K_MAX"], kmax_fit) // bs
    small = [k for k in (1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33) if k <= hi]
    return small + [rnd.randint(1, hi) for _ in range(n)]


def _kfit(cfg: dict, kind: int, T: int) -> int:
    """Largest K (multiple of the block size) whose T tokens + images fit the SPM."""
    R = ref()
    bs = block_size(kind)
    k = cfg["K_MAX"] // bs * bs
    while k >= bs:
        if T * (k // 16) + T * R.qa_lines(kind, k) <= cfg["LINES"]:
            return k
        k -= bs
    return 0


@cocotb.test()
async def qact_directed(dut):
    cfg, rnd, spm, drv, _ = await _setup(dut, 11)
    await drv.idle_check(4)
    # Fixed vectors (one or two blocks), each with T = 1 and, when allowed, replicated over tokens.
    for name, kind, K, x in directed_vectors():
        if K > cfg["K_MAX"]:
            continue
        _set_ready(spm, rnd)
        src, dst, stride = drv.place(kind, K, 1)
        await drv.qact(kind, K, 1, [x], src, dst, stride)
        T = cfg["T_MAX"]
        if T > 1 and _kfit(cfg, kind, T) >= K:
            xs = [x] + [gen_vector(rnd, kind, K) for _ in range(T - 1)]
            rnd.shuffle(xs)
            src, dst, stride = drv.place(kind, K, T, gap=rnd.randint(0, 3))
            await drv.qact(kind, K, T, xs, src, dst, stride)
    # Every generator on its own, both kinds, a few K each.
    for gname in qact_tb.GENERATORS:
        for kind in (QA_A32, QA_A256):
            bs = block_size(kind)
            for nb in (1, 2, rnd.randint(3, 9)):
                K = nb * bs
                if K > _kfit(cfg, kind, 1):
                    continue
                _set_ready(spm, rnd)
                T = rnd.randint(1, cfg["T_MAX"])
                while T > 1 and _kfit(cfg, kind, T) < K:
                    T -= 1
                xs = [gen_vector(rnd, kind, K, gname) for _ in range(T)]
                src, dst, stride = drv.place(kind, K, T, gap=rnd.choice((0, 0, 1, 5)))
                await drv.qact(kind, K, T, xs, src, dst, stride)
    await drv.idle_check(8)
    dut._log.info("qact_directed: %d commands; coverage: %s", drv.n_cmds, drv.cov.report())
    drv.cov.require(qact_tb.COVER_KEYS)


@cocotb.test()
async def qact_random(dut):
    cfg, rnd, spm, drv, _ = await _setup(dut, 12)
    n = cfg["N_RANDOM"]
    lines = cfg["LINES"]
    for i in range(n):
        kind = rnd.choice((QA_A32, QA_A256))
        bs = block_size(kind)
        T = rnd.randint(1, cfg["T_MAX"])
        kf = _kfit(cfg, kind, T)
        while kf == 0:
            T -= 1
            kf = _kfit(cfg, kind, T)
        cap = min(kf, 8192 if i % 4 else kf)
        K = bs * (rnd.randint(1, cap // bs) if rnd.random() < 0.7 else rnd.randint(1, min(8, cap // bs)))
        n_l = K // 16
        qa = ref().qa_lines(kind, K)
        room = lines - T * n_l - T * qa
        gap = 0 if T == 1 or room <= 0 or rnd.random() < 0.4 else rnd.randint(1, min(room // max(1, T - 1), 40))
        at_end = rnd.choice((None, None, None, "src", "dst", "dst0"))
        try:
            src, dst, stride = drv.place(kind, K, T, gap=gap, at_end=at_end)
        except ValueError:
            src, dst, stride = drv.place(kind, K, T, gap=0)
        # T == 1: the resolved stride is whatever w5 was (the dispatch passes it); the unit ignores it.
        sfield = rnd.choice((stride, 0xFFFFFFFF, rnd.getrandbits(32))) if T == 1 else None
        mode = rnd.choice(("mix", "mix", "mix", "normal", "bits"))
        xs = [gen_vector(rnd, kind, K, mode) for _ in range(T)]
        rm = _set_ready(spm, rnd)
        cyc = await drv.qact(kind, K, T, xs, src, dst, stride, stride_field=sfield)
        dut._log.info("random %d: kind %d K %d T %d gap %d ready %s: %d cycles", i, kind, K, T, gap, rm, cyc)
        if rnd.random() < 0.3:
            await drv.idle_check(rnd.randint(1, 5))
    dut._log.info("qact_random: %d commands; coverage: %s", drv.n_cmds, drv.cov.report())


@cocotb.test()
async def qact_kmax(dut):
    cfg, rnd, spm, drv, _ = await _setup(dut, 13)
    for kind in (QA_A256, QA_A32):
        T = cfg["T_MAX"]
        while T > 1 and _kfit(cfg, kind, T) < cfg["K_MAX"]:
            T -= 1
        K = _kfit(cfg, kind, T)
        assert K == cfg["K_MAX"], "K_MAX does not fit the SPM of this configuration"
        for mode in ("normal", "mix"):
            xs = [gen_vector(rnd, kind, K, mode) for _ in range(T)]
            _set_ready(spm, rnd, "always" if mode == "normal" else "bursty")
            src, dst, stride = drv.place(kind, K, T)
            cyc = await drv.qact(kind, K, T, xs, src, dst, stride)
            dut._log.info("kmax kind %d K %d T %d (%s): %d cycles", kind, K, T, mode, cyc)


@cocotb.test()
async def qact_timing(dut):
    """Accept to done of QACT ops with an always-ready SPM, by kind, K and T (the table behind the simulator's cycle model,
    SW-80). TSR_QACT_TIMING_OUT gets the json rows [kind, K, T, cycles]."""
    cfg, rnd, spm, drv, _ = await _setup(dut, 15)
    _set_ready(spm, rnd, "always")
    spm.garbage = False
    rows = []
    for kind in (QA_A32, QA_A256):
        for T in range(1, cfg["T_MAX"] + 1):
            for K in (256, 512, 704, 1024, 2048, 2816, 4096, 6144, 8192):
                if K % block_size(kind) or K > _kfit(cfg, kind, T):
                    continue
                xs = [gen_vector(rnd, kind, K, "normal") for _ in range(T)]
                src, dst, stride = drv.place(kind, K, T)
                cyc = await drv.qact(kind, K, T, xs, src, dst, stride)
                rows.append([kind, K, T, cyc])
                dut._log.info("qact_timing kind %d K %d T %d: %d cycles", kind, K, T, cyc)
                # the simulator's model (sw/sim/tsr_sim.c cyc_qact, SW-80) is these plus 4 cycles of the core's done path:
                # A32 16 cycles per 32-block + 40 (divider bound), A256 17 per 256-block + 68
                nblk = T * (K // block_size(kind))
                want, tol = (16 * nblk + 40, 1) if kind == QA_A32 else (17 * nblk + 68, 8)
                assert abs(cyc - want) <= tol, "kind %d K %d T %d: %d cycles, the model says %d" % (kind, K, T, cyc, want)
    out = os.environ.get("TSR_QACT_TIMING_OUT")
    if out:
        Path(out).write_text(json.dumps(rows) + "\n")


@cocotb.test()
async def qact_throughput(dut):
    cfg, rnd, spm, drv, _ = await _setup(dut, 14)
    _set_ready(spm, rnd, "always")
    spm.garbage = False
    res = {}
    for kind in (QA_A256, QA_A32):
        K = min(4096, _kfit(cfg, kind, 1))
        x = gen_vector(rnd, kind, K, "normal")
        src, dst, stride = drv.place(kind, K, 1)
        res[kind] = (K, await drv.qact(kind, K, 1, [x], src, dst, stride))
    k256, c256 = res[QA_A256]
    k32, c32 = res[QA_A32]
    lines256 = k256 // 16
    blocks32 = k32 // 32
    dut._log.info("throughput: A256 K %d: %d cycles (%.2f / line); A32 K %d: %d cycles (%.2f / block)",
                  k256, c256, c256 / lines256, k32, c32, c32 / blocks32)
    # Documented bounds (tsr_qact.sv header): A256 ~1 line / cycle with the default buffer (2 with
    # the minimum one), A32 ~15 cycles / block (divider bound), + pipeline fill.
    per_line = 1.25 if cfg["BUF_LINES"] >= 64 else 4.5
    assert c256 <= per_line * lines256 + 160, "A256 throughput: %d cycles for %d lines" % (c256, lines256)
    assert c32 <= 16 * blocks32 + 120, "A32 throughput: %d cycles for %d blocks" % (c32, blocks32)
    # Latency of a single block (pipeline fill): accept -> done.
    for kind in (QA_A256, QA_A32):
        K = block_size(kind)
        src, dst, stride = drv.place(kind, K, 1)
        lat = await drv.qact(kind, K, 1, [gen_vector(rnd, kind, K, "normal")], src, dst, stride)
        dut._log.info("latency: one %s block: %d cycles", "A256" if kind else "A32", lat)
    # Back-to-back commands: the next one is offered while the previous one runs, so it is
    # accepted at the edge that ends the previous done cycle.
    await back_to_back(dut, cfg, rnd, spm, drv, 8)


async def back_to_back(dut, cfg, rnd, spm, drv, n: int):
    prev = None
    for i in range(n):
        avoid = QactDriver.regions(prev) if prev else ()
        while True:   # a command small enough to sit next to the running one (small SPMs)
            kind = rnd.choice((QA_A32, QA_A256))
            T = rnd.randint(1, cfg["T_MAX"])
            K = block_size(kind) * rnd.randint(1, 6)
            if T * (K // 16 + ref().qa_lines(kind, K)) > cfg["LINES"] // 3:
                continue
            try:
                src, dst, stride = drv.place(kind, K, T, avoid=avoid)
                break
            except ValueError:
                continue
        xs = [gen_vector(rnd, kind, K) for _ in range(T)]
        task = cocotb.start_soon(drv.start(kind, K, T, xs, src, dst, stride))
        if prev:
            await drv.finish(prev)
        ctx = await task
        if prev:
            assert ctx["t_acc"] == prev_done_cycle(spm, prev) + 1, \
                "back-to-back: accepted at %d, previous done in %d" % (ctx["t_acc"], prev_done_cycle(spm, prev))
        prev = ctx
    await drv.finish(prev)


def prev_done_cycle(spm, ctx) -> int:
    return [c for c in spm.done_cycles if c >= ctx["t_acc"]][0]


@cocotb.test()
async def qact_reset(dut):
    cfg, rnd, spm, drv, _ = await _setup(dut, 15)
    for rep in range(4):
        kind = rnd.choice((QA_A32, QA_A256))
        T = rnd.randint(1, cfg["T_MAX"])
        while T > 1 and _kfit(cfg, kind, T) < block_size(kind) * 4:
            T -= 1
        K = min(_kfit(cfg, kind, T), block_size(kind) * rnd.randint(2, 8))
        xs = [gen_vector(rnd, kind, K) for _ in range(T)]
        src, dst, stride = drv.place(kind, K, T)
        _set_ready(spm, rnd)
        # Start a command and reset it after a random number of cycles.
        for t in range(T):
            for i, ln in enumerate(qact_tb.lines_of(xs[t])):
                spm.mem[src + t * stride + i] = ln
        for ln in range(dst, dst + T * ref().qa_lines(kind, K)):
            spm.mem[ln] = 0
        await RisingEdge(dut.clk)
        dut.cmd.value = isa.QACT_CMD_T.pack(kind=kind, dst_line=dst, src_line=src, k=K, t=T,
                                            src_stride=stride, qa_lines=ref().qa_lines(kind, K))
        dut.cmd_valid.value = 1
        while True:
            await FallingEdge(dut.clk)
            if int(dut.cmd_ready.value):
                break
        await RisingEdge(dut.clk)
        dut.cmd_valid.value = 0
        await ClockCycles(dut.clk, rnd.randint(1, 60 * (rep + 1)))
        await FallingEdge(dut.clk)
        dut.rst_n.value = 0
        await ClockCycles(dut.clk, rnd.randint(1, 3))
        await FallingEdge(dut.clk)
        # Reset state of every output (registered: visible after the first reset edge): idle.
        assert int(dut.cmd_ready.value) == 1 and int(dut.done.value) == 0, "reset: cmd_ready / done"
        assert int(dut.spm_rd_req_valid.value) == 0 and int(dut.spm_wr_valid.value) == 0, "reset: SPM requests"
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        await drv.idle_check(3)
        # A fully checked command after the reset.
        _set_ready(spm, rnd)
        xs = [gen_vector(rnd, kind, K) for _ in range(T)]
        src, dst, stride = drv.place(kind, K, T)
        await drv.qact(kind, K, T, xs, src, dst, stride)


PROBE_INPUTS = ["rst_n", "cmd_valid", "cmd", "spm_rd_req_ready", "spm_rd_rsp_valid", "spm_rd_rsp_data",
                "spm_wr_ready"]
PROBE_OUTPUTS = ["cmd_ready", "done", "spm_rd_req_valid", "spm_rd_req_line", "spm_wr_valid", "spm_wr_line",
                 "spm_wr_data", "spm_wr_be"]


async def comb_probe(dut, clock: Clock, rnd: random.Random, tries: int = 2) -> int:
    """Stop the clock right after a falling edge, set every input (one at a time) to random values
    and check that no output changes; restore the inputs and restart the clock. Without a clock
    edge only combinational logic can react: an output change is an input->output path."""
    await FallingEdge(dut.clk)
    clock.stop()
    await Timer(1, "ns")
    snap = {n: getattr(dut, n).value for n in PROBE_OUTPUTS}
    bad = set()
    pokes = 0
    for name in PROBE_INPUTS:
        sig = getattr(dut, name)
        saved = sig.value
        for _ in range(tries):
            sig.value = rnd.getrandbits(len(sig))
            pokes += 1
            await Timer(1, "ns")
            for n in PROBE_OUTPUTS:
                if getattr(dut, n).value != snap[n]:
                    bad.add((name, n))
        sig.value = saved
        await Timer(1, "ns")
    clock.start(start_high=True)
    assert not bad, "combinational input->output paths: %s" % sorted(bad)
    return pokes


@cocotb.test()
async def qact_comb_probe(dut):
    clock = Clock(dut.clk, CLK_PERIOD_NS, unit="ns")
    cfg, rnd, spm, drv, clock = await _setup(dut, 16, clock)
    probes = {"n": 0, "stop": False}

    async def prober():
        while not probes["stop"]:
            await ClockCycles(dut.clk, rnd.randint(3, 40))
            if probes["stop"]:
                break
            await comb_probe(dut, clock, rnd)
            probes["n"] += 1

    task = cocotb.start_soon(prober())
    for i in range(6):
        kind = (QA_A32, QA_A256)[i % 2]
        K = min(_kfit(cfg, kind, 1), block_size(kind) * rnd.randint(2, 6))
        xs = [gen_vector(rnd, kind, K)]
        src, dst, stride = drv.place(kind, K, 1)
        _set_ready(spm, rnd)
        await drv.qact(kind, K, 1, xs, src, dst, stride)
    probes["stop"] = True
    await ClockCycles(dut.clk, 60)
    task.cancel()
    assert probes["n"] >= 5, "too few probes (%d)" % probes["n"]
    dut._log.info("comb probe: %d probes", probes["n"])


# =============================================================================================
# First evaluation (time 0)
# =============================================================================================
# Verilator (--x-initial unique) gives every submodule its own randomly initialised copy of clk,
# and seeds each edge detector from that copy in the first evaluation, before the continuous
# assignment from the top-level clk is applied. A copy that starts at 0 under a top-level clk of 1
# therefore takes a rising edge at 0 ns, with rst_n and all state still random (rtl/core/
# tsr_spm.sv, chk_on_q). cocotb's deferred `.value =` writes and the clock reach the model only
# after that evaluation. Every simulation check must therefore be armed by a reset edge.
# These tests force the worst case. The first test's code up to its first await runs before that
# evaluation and `Immediate` writes take effect at once, so it sets top clk = 1, rst_n = 1, every
# clocked instance's clk copy = 0 (a spurious rising edge in each of them) and, in each module
# with checks, state that violates one of them. With top = tsr_qact, tsr_qact's own processes use
# the top-level clk and get no such edge; the rtl_spm build covers them (and tsr_spm).
FP_DIV_S_IDLE, FP_DIV_S_ITER = 0, 2


def _clocked(q) -> list:
    """The clocked instances inside a tsr_qact (each has its own clk copy)."""
    return [q.u_buf, q.u_da, q.u_db] + [q.g_mul[j].u_mul for j in range(16)]


async def _time0_worst_case(dut, q, others=()):
    assert get_sim_time("step") == 0, "must be the first test of a simulation"
    div = os.environ.get("TSR_QACT_T0_DIV", "busy")
    dut.clk.value = Immediate(1)
    dut.rst_n.value = Immediate(1)
    for inst in list(others) + _clocked(q):
        inst.clk.value = Immediate(0)
    # tsr_qact: buf_res above BUF_LINES ("line buffer over-reserved").
    q.buf_res.value = Immediate((1 << len(q.buf_res)) - 1)
    # tsr_fp_div: 'busy' = in_ready high in S_ITER ("in_ready while busy"), the power-up state of
    # the seed-5 / seed-10 failures; 'idle' = a consistent state (tests the rest of the hierarchy).
    for d in (q.u_da, q.u_db):
        d.in_ready.value = Immediate(1)
        d.state.value = Immediate(FP_DIV_S_ITER if div == "busy" else FP_DIV_S_IDLE)
    await Timer(1, "ns")      # the first evaluation has run: no check may have fired
    dut._log.info("time 0: forced first-evaluation edge with rst_n high survived (dividers %s)", div)


@cocotb.test()
async def qact_time0_edge(dut):
    await _time0_worst_case(dut, dut)
    cfg, rnd, spm, drv, _ = await _setup(dut, 18)
    await drv.idle_check(3)
    for kind in (QA_A32, QA_A256):           # then the unit works normally
        K = min(_kfit(cfg, kind, 1), 4 * block_size(kind))
        src, dst, stride = drv.place(kind, K, 1)
        await drv.qact(kind, K, 1, [gen_vector(rnd, kind, K)], src, dst, stride)


# =============================================================================================
# Real tsr_spm (tb/qact/hdl/tsr_tb_qact.sv)
# =============================================================================================
class TbClient:
    """Client 0 of the real tsr_spm: preload writes, readback reads, random contention traffic.
    Requests obey R2 (held until ready); each response pops the oldest accepted read."""

    def __init__(self, dut, rnd: random.Random):
        self.dut, self.rnd = dut, rnd
        self.wq = []            # testbench writes (line, data)
        self.rq = []            # testbench reads (line)
        self.rd_data = []       # testbench read data, in request order
        self.contend = None     # (lo, hi) inclusive scratch line range for random traffic
        self.inflight = []      # per accepted read: True = testbench read, False = contention
        self.rv = self.wv = False
        self.qact_rd_stall = 0
        self.qact_wr_stall = 0

    async def run(self):
        dut = self.dut
        for sig in (dut.x_rd_req_valid, dut.x_wr_valid, dut.x_rd_req_line, dut.x_wr_line, dut.x_wr_data):
            sig.value = 0
        r_tb = w_tb = False
        q = dut.u_qact
        while True:
            await FallingEdge(dut.clk)
            if not int(dut.rst_n.value):
                continue              # tsr_spm's state is arbitrary until the first reset edge
            rd_x = self.rv and int(dut.x_rd_req_ready.value)
            wr_x = self.wv and int(dut.x_wr_ready.value)
            got = int(dut.x_rd_rsp_valid.value)
            rsp = int(dut.x_rd_rsp_data.value) if got else None
            if int(q.spm_rd_req_valid.value) and not int(q.spm_rd_req_ready.value):
                self.qact_rd_stall += 1
            if int(q.spm_wr_valid.value) and not int(q.spm_wr_ready.value):
                self.qact_wr_stall += 1
            await RisingEdge(dut.clk)
            if got:
                assert self.inflight, "client 0 read response without a request"
                if self.inflight.pop(0):
                    self.rd_data.append(rsp)
            if rd_x:
                self.inflight.append(r_tb)
                if r_tb:
                    self.rq.pop(0)
                self.rv = False
            if wr_x:
                if w_tb:
                    self.wq.pop(0)
                self.wv = False
            if not self.rv:
                if self.rq:
                    self.rv, r_tb = True, True
                    dut.x_rd_req_line.value = self.rq[0]
                elif self.contend and self.rnd.random() < 0.35:
                    self.rv, r_tb = True, False
                    dut.x_rd_req_line.value = self.rnd.randint(*self.contend)
            if not self.wv:
                if self.wq:
                    self.wv, w_tb = True, True
                    dut.x_wr_line.value = self.wq[0][0]
                    dut.x_wr_data.value = self.wq[0][1]
                elif self.contend and self.rnd.random() < 0.35:
                    self.wv, w_tb = True, False
                    dut.x_wr_line.value = self.rnd.randint(*self.contend)
                    dut.x_wr_data.value = self.rnd.getrandbits(512)
            dut.x_rd_req_valid.value = int(self.rv)
            dut.x_wr_valid.value = int(self.wv)

    async def drain(self):
        while self.wq or self.rq or self.inflight or self.rv or self.wv:
            await RisingEdge(self.dut.clk)


async def _rtl_spm_setup(dut, rnd: random.Random) -> TbClient:
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd.value = 0
    Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start()
    tb = TbClient(dut, rnd)
    cocotb.start_soon(tb.run())
    await ClockCycles(dut.clk, 5)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    return tb


@cocotb.test()
async def qact_rtl_spm(dut):
    cfg = _cfg()
    rnd = tb_rng(17)
    tb = await _rtl_spm_setup(dut, rnd)
    R = ref()
    lines = cfg["LINES"]
    n_cmds = 0
    for i in range(24):
        kind = rnd.choice((QA_A32, QA_A256))
        bs = block_size(kind)
        T = rnd.randint(1, cfg["T_MAX"])
        K = bs * rnd.randint(1, cfg["K_MAX"] // bs)
        n = K // 16
        qa = R.qa_lines(kind, K)
        gap = rnd.choice((0, 0, 3))
        stride = n + gap
        src_sz, dst_sz = (T - 1) * stride + n, T * qa
        scratch = 64
        if src_sz + dst_sz + scratch > lines:
            continue
        # Layout: [src][dst][scratch] at a random base, or rotated.
        base = rnd.randrange(0, lines - (src_sz + dst_sz + scratch) + 1)
        if rnd.random() < 0.5:
            src, dst, scr = base, base + src_sz, base + src_sz + dst_sz
        else:
            dst, src, scr = base, base + dst_sz, base + src_sz + dst_sz
        xs = [gen_vector(rnd, kind, K) for _ in range(T)]
        for t in range(T):
            for j, ln in enumerate(qact_tb.lines_of(xs[t])):
                tb.wq.append((src + t * stride + j, ln))
        for ln in range(dst, dst + dst_sz):
            tb.wq.append((ln, rnd.getrandbits(512)))      # garbage: every byte must be overwritten
        await tb.drain()
        tb.contend = (scr, scr + scratch - 1) if i % 3 else None
        await RisingEdge(dut.clk)
        dut.cmd.value = isa.QACT_CMD_T.pack(kind=kind, dst_line=dst, src_line=src, k=K, t=T,
                                            src_stride=stride, qa_lines=qa)
        dut.cmd_valid.value = 1
        while True:
            await FallingEdge(dut.clk)
            if int(dut.cmd_ready.value):
                break
        await RisingEdge(dut.clk)
        dut.cmd_valid.value = 0
        for c in range(200000):
            await FallingEdge(dut.clk)
            if int(dut.done.value):
                break
        else:
            raise AssertionError("no done")
        tb.contend = None
        await tb.drain()
        tb.rd_data = []
        tb.rq = list(range(dst, dst + dst_sz))
        await tb.drain()
        assert len(tb.rd_data) == dst_sz
        for t in range(T):
            want = R.qact(kind, xs[t], K)
            got = b"".join(v.to_bytes(64, "little") for v in tb.rd_data[t * qa:(t + 1) * qa])
            if got != want:
                raise AssertionError("rtl_spm cmd %d token %d: %s" % (
                    i, t, qact_tb.describe_mismatch(kind, K, xs[t], got, want)))
        n_cmds += 1
    assert n_cmds >= 15
    assert tb.qact_rd_stall > 0 and tb.qact_wr_stall > 0, "contention did not stall QACT (%d, %d)" % (
        tb.qact_rd_stall, tb.qact_wr_stall)
    dut._log.info("rtl_spm: %d commands, QACT read stalls %d, write stalls %d", n_cmds,
                  tb.qact_rd_stall, tb.qact_wr_stall)


@cocotb.test()
async def rtl_spm_time0_edge(dut):
    """_time0_worst_case on the real-SPM wrapper: tsr_qact and tsr_spm (and its banks' RAMs) also
    get the spurious edge here; tsr_spm's grant registers are forced non-one-hot."""
    cfg = _cfg()
    spm = dut.u_spm
    # tsr_spm builds each bank from 8 column slices of 64 bits (g_bank[b].g_col[k].u_ram)
    banks = [spm.g_bank[b].g_col[k].u_ram
             for b in range(-(-cfg["LINES"] // cfg["BANK_LINES"])) for k in range(8)]
    spm.rd_gnt_q.value = Immediate((1 << len(spm.rd_gnt_q)) - 1)
    await _time0_worst_case(dut, dut.u_qact, [dut.u_qact, spm] + banks)
    rnd = tb_rng(20)
    tb = await _rtl_spm_setup(dut, rnd)
    for _ in range(3):
        await FallingEdge(dut.clk)
        assert int(dut.cmd_ready.value) == 1 and int(dut.done.value) == 0, "QACT not idle after reset"
    ln, data = rnd.randrange(cfg["LINES"]), rnd.getrandbits(512)
    tb.wq.append((ln, data))
    await tb.drain()
    tb.rq.append(ln)
    await tb.drain()
    assert tb.rd_data == [data], "SPM write / readback after the start-up"

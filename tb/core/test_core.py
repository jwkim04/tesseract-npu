"""Core-level verification of tsr_core (docs/rtl-m2.md block (f); spec 검증 전략 "RTL 코어 전체":
the ISA simulator is the oracle, trace replay must be bit-exact).

This module holds the M2 suite: every build here pins HAS_MVM = 0 (tsr_core's default is 1 since
the M3 integration, docs/rtl-m3.md §13), so MVM is ILLEGAL_OP and the M2 results stay those of
the M2 core. The HAS_MVM = 1 suite (MVM traces, MVM random programs, MVM directed tests) is
tb/core/test_core_m3.py, which reuses the harness below (CoreH, regimes, helpers).

The DUT is the real tsr_core (CSR port, NUM_CH memory ports, done) with every block inside it:
tsr_csr, tsr_cmdproc, tsr_fetch, tsr_dma, tsr_qact (+ fp), tsr_spm, tsr_mem_arb. Its boundary is
driven only through common.csr_driver.CsrDriver and common.mem_model.MemModel (random latency,
backpressure, stalls, write-ordering modes, protocol assertions).

pytest functions (one Verilator build per configuration; all cocotb tests of a build run in one
simulator invocation, back to back on one DUT without reset):
  test_core_traces[<sets>]  one build per distinct trace configuration under $TSR_TRACES
                            (default traces/m2; pytest --tsr-traces DIR), elaborated with that
                            configuration's parameters (Trace.core_parameters()). Tests:
                            core_csr_consts, core_traces (every trace of the configuration under
                            TSR_CORE_PASSES memory regimes, default 3 of rand / fast / slow /
                            ordered, rotating per trace; seeds derived from --tsr-seed),
                            core_random (random programs vs the simulator at this configuration,
                            TSR_CORE_RANDOM programs).
  test_core_impl[<variant>] the same trace replay + core_random with implementation parameters
                            that must not change results: SPM_RD_LAT 1 / 3 / 4, FETCH_BEATS 2 /
                            4 / 64, RD_OUTSTANDING 1 / 2 / 4.
  test_core_directed[<cfg>] reduced configurations (fast builds, c2 / tiny): every directed test
                            below plus core_random; u200: U200 v1 parameters (throughput, a few
                            directed tests).
cocotb tests:
  core_csr_consts     ID / VERSION / CFG0..2 / CAPS match the elaboration parameters; STATUS 0,
                      done 0 after reset
  core_csr_map        every 12-bit CSR offset (aligned and unaligned) vs the simulator after
                      reset and after writing random values to every offset
  core_traces         replay.run_trace for every trace (STATUS, ERR_CODE, ERR_PC, ERR_INFO,
                      expect regions bit-exact, written-beat set == expect beats, reads inside
                      init / command beats, quiescent port at DONE), plus PERF deltas: retired
                      == trace perf, read beats of channels >= 1 == trace perf (LD beats; spec
                      성능 카운터), channel-0 counter == beats seen at the port
  core_status_w1c     STATUS W1C semantics (DONE / ERR only), done output follows STATUS.DONE,
                      START clears ERR_*, START while BUSY ignored
  core_soft_reset     SOFT_RESET idle / with START / while BUSY (P3: no DONE, all cleared,
                      port quiet), CMD_* / ARG* kept, R1..R15 cleared
  core_hw_reset       rst_n at random points of a run, pulses of >= RST_PIPE + 1 cycles (P10):
                      outputs quiet from RST_PIPE cycles after rst_n falls until the next CSR
                      access, every CSR equal to a fresh simulator's afterwards, the next
                      program matches (RST_PIPE from TSR_CORE "rst_pipe", default 0)
  core_abort          START|ABORT, ABORT while idle, ABORT at random cycle offsets inside long
                      programs: ERR_PC = first instruction not executed, PC unchanged, effects
                      == simulator run of the executed prefix (DRAM, registers, SPM via dump)
  core_back_to_back   programs back to back without reset, alternating channels and units,
                      repeated program gives the same result
  core_mvm_illegal    HAS_MVM = 0: MVM at any position (valid and odd fields) is ILLEGAL_OP with
                      the effects of the preceding instructions
  core_perf           PERF counters: cycles, retired, per-channel read beats vs the simulator
                      and the port, wait cycles bounded, MVM / VU / WAITM 0, LO/HI snapshot
  core_throughput     cycles to done with an ideal memory: LD / ST beats per cycle, QACT cycles
                      per source line at K_MAX, cycles per instruction (logged, loose floors)
  core_random         ProgGen (tb/common) random programs on the RTL and the simulator in
                      lockstep (SPM initialised to the simulator's 0xA5 fill through an LD);
                      final CSRs, PC, R1..R15, retired, read beats, DRAM (replay.compare) and the
                      whole SPM (ST dumps) compared; half of the runs with random CSR reads
                      (PERF LO/HI, unmapped, unaligned) while BUSY. Programs whose executed QACT
                      overlaps dst and src (undefined, spec 겹치는 피연산자) are skipped.
Environment: TSR_CORE_PASSES (memory regimes per trace), TSR_CORE_RANDOM (random programs per
build, default per configuration), TSR_CORE_SPM_EVERY=N (core_random: also compare the whole SPM
after every N programs), TSR_CORE_TESTS=a,b (subset of cocotb tests),
TSR_CORE_TRACE_FILTER=substring (only traces whose path contains it).
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import isa, replay, trace  # noqa: E402
from common.csr_driver import CsrDriver  # noqa: E402
from common.mem_model import MemModel  # noqa: E402
from common.runner import REPO, run  # noqa: E402
from common.tbutil import seed as tb_seed  # noqa: E402
from common.tbutil import start_clock_and_reset, wait_level  # noqa: E402

from common import simlib  # noqa: E402  (ctypes wrapper of the ISA simulator)
from common.proggen import ProgGen  # noqa: E402  (random program generator)

M32 = 0xFFFFFFFF
DONE, DONEERR = isa.STATUS_DONE, isa.STATUS_DONE | isa.STATUS_ERR

ALL_TESTS = ["core_csr_consts", "core_csr_map", "core_status_w1c", "core_soft_reset", "core_hw_reset",
             "core_abort", "core_back_to_back", "core_mvm_illegal", "core_perf", "core_throughput",
             "core_random"]
TRACE_TESTS = ["core_csr_consts", "core_traces", "core_random"]

# Memory-port regimes for trace replay (MemModel keyword arguments).
REGIMES = {
    "rand": {},
    "fast": dict(rd_lat=(0, 2), wr_lat=(0, 2), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0,
                 p_wr_req_ready=1.0, p_wr_dat_ready=1.0, p_wr_rsp_valid=1.0, p_stall=0.0,
                 wr_order="any"),
    "slow": dict(rd_lat=(20, 160), wr_lat=(10, 120), p_rd_req_ready=0.3, p_rd_rsp_valid=0.45,
                 p_wr_req_ready=0.3, p_wr_dat_ready=0.45, p_wr_rsp_valid=0.4, p_stall=0.02,
                 stall_len=(5, 60)),
    "ordered": dict(interleave=False, wr_order="req_first", p_rd_rsp_valid=0.9),
}
REGIME_ORDER = ["rand", "fast", "slow", "ordered"]


def _cfg(**kw):
    c = dict(isa.U200_V1)
    c.update(kw)
    return c


# Reduced configurations for the directed tests (fast Verilator builds).
DIRECTED = {
    # 2 channels, T_MAX 4, 64 KiB SPM, 1 GiB channels with a 2^22-beat address (limit < CH_BYTES),
    # RD_OUTSTANDING 4 so that the limit is reached by the short LDs of the directed tests.
    "c2": dict(cfg=_cfg(num_ch=2, t_max=4, k_max=4096, spm_bytes=65536, ch_bytes=1 << 30,
                        addr_beat_w=22, rd_outstanding=4), tests=ALL_TESTS, n_random=300),
    # Limits everywhere: 32-beat channels (below one 4 KiB block), 64-line SPM, K_MAX 256.
    "tiny": dict(cfg=_cfg(num_ch=2, t_max=4, k_max=256, spm_bytes=4096, ch_bytes=4096, addr_beat_w=5),
                 tests=["core_csr_consts", "core_csr_map", "core_random"], n_random=500),
    # U200 v1 parameters: throughput figures and the directed tests that are cheap at 4 MiB SPM.
    "u200": dict(cfg=_cfg(), tests=["core_csr_consts", "core_throughput", "core_status_w1c",
                                    "core_mvm_illegal", "core_perf"], n_random=0),
}


def core_params(cfg: dict, has_mvm: int = 0) -> dict:
    """tsr_core parameters of a tsr_trace.h config dict. HAS_MVM is always passed explicitly
    (the RTL default is 1): 0 for the M2 suite of this module, 1 for test_core_m3.py."""
    return {"NUM_CH": cfg["num_ch"], "BEAT_BITS": cfg["beat_bits"], "LANES": cfg["lanes"],
            "T_MAX": cfg["t_max"], "K_MAX": cfg["k_max"], "SPM_BYTES": cfg["spm_bytes"], "VL": cfg["vl"],
            "NIREG": cfg["nireg"], "NSREG": cfg["nsreg"], "LOOP_DEPTH": cfg["loop_depth"],
            "RD_OUTSTANDING": cfg["rd_outstanding"], "CH_BYTES": "64'd%d" % cfg["ch_bytes"],
            "ADDR_BEAT_W": cfg["addr_beat_w"], "HAS_MVM": int(has_mvm)}


def needs_mvm(path) -> int | None:
    """1 when the trace executes an MVM (or faults at one with a code a HAS_MVM = 0 core cannot
    produce), 0 when its MVM must be ILLEGAL_OP (HAS_MVM = 0 only), None when indifferent
    (tb/fastreplay/run.py trace_info rule)."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    count, err, err_pc = int(meta["cmd_count"]), int(meta["err_code"]), int(meta["err_pc"])
    cf = path / "cmd.bin"
    cmd = cf.read_bytes() if cf.is_file() else b""
    if len(cmd) < 32 * count:
        return None
    executed = min(err_pc, count) if err else next(
        (pc + 1 for pc in range(count) if cmd[32 * pc] == isa.OP_END), count)
    if any(cmd[32 * pc] == isa.OP_MVM for pc in range(executed)):
        return 1
    if err and err_pc < count and cmd[32 * err_pc] == isa.OP_MVM:
        if err != isa.ERR_ILLEGAL_OP:
            return 1
        w = isa.insn_words(cmd[32 * err_pc:32 * err_pc + 32])
        reserved = ((w[0] >> 28) & 0xF) or ((w[0] >> 13) & 7) or (w[5] >> 20) or w[7]
        return None if reserved else 0
    return None


def executed_mvms(tr) -> int:
    """MVM instructions a trace (or a simulator capture) executed (dispatched to the unit)."""
    return sum(1 for pc in range(tr.executed_count()) if tr.insn(pc)[0] & 0xFF == isa.OP_MVM)


def _selected(tests):
    sel = os.environ.get("TSR_CORE_TESTS")
    if not sel:
        return list(tests)
    sel = sel.split(",")
    return [t for t in tests if t in sel]


# =============================================================================================
# pytest functions
# =============================================================================================
def trace_groups(roots, want_mvm: int, prefix_root: bool = False) -> list:
    """[(id, config dict, [trace paths])] for every distinct configuration of the traces under
    `roots` that a HAS_MVM = want_mvm core can replay (needs_mvm() is None or want_mvm). The id
    joins the set (parent directory) names, prefixed by the root's name when prefix_root."""
    groups = collections.OrderedDict()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in trace.find(root):
            nm = needs_mvm(p)
            if nm is not None and nm != want_mvm:
                continue
            meta = json.loads((p / "meta.json").read_text())
            key = json.dumps(meta["config"], sort_keys=True)
            groups.setdefault(key, []).append((root, p))
    out = []
    for key, items in groups.items():
        sets = sorted({(root.name + "." if prefix_root else "") + (p.parent.name if p.parent != root else p.name)
                       for root, p in items})
        out.append(("+".join(sets)[:80], json.loads(key), [str(p) for _, p in items]))
    return out


# M2 suite: traces under $TSR_TRACES (default traces/m2) that a HAS_MVM = 0 core can replay.
TRACE_GROUPS = trace_groups([Path(os.environ.get("TSR_TRACES", REPO / "traces" / "m2"))], 0)


@pytest.mark.traces
@pytest.mark.parametrize("gid,cfg,paths", TRACE_GROUPS, ids=[g[0] for g in TRACE_GROUPS])
def test_core_traces(gid, cfg, paths):
    tests = _selected(TRACE_TESTS)
    if not tests:
        pytest.skip("no selected test")
    n_random = 150 if cfg["spm_bytes"] <= 1 << 20 else 100
    run("tsr_core", __name__, parameters=core_params(cfg), name="core_" + gid.replace("+", "_"),
        testcase=tests,
        extra_env={"TSR_CORE": json.dumps(dict(cfg=cfg, name=gid, traces=paths, n_random=n_random))})


# Implementation-parameter variants (not part of a trace's configuration, so results must not
# change): SPM read latency (M4 expects 4 at U200), fetch buffer depth, read-outstanding limit
# (1 serialises fetch and DMA reads on channel 0). Each replays the trace sets named by `sets`.
IMPL = {
    "lat4_u200": dict(sets="errors+tmax1", params=dict(SPM_RD_LAT=4), passes=1, n_random=60),
    "lat3_t8": dict(sets="errors8+tmax8", params=dict(SPM_RD_LAT=3, FETCH_BEATS=64, RD_OUTSTANDING=4),
                    passes=1, n_random=60),
    "lat1_small": dict(sets="small1ch", params=dict(SPM_RD_LAT=1, FETCH_BEATS=2, RD_OUTSTANDING=1),
                       passes=3, n_random=150),
    "lat4_ch3": dict(sets="ch3", params=dict(SPM_RD_LAT=4, FETCH_BEATS=4, RD_OUTSTANDING=2),
                     passes=3, n_random=150),
}


@pytest.mark.traces
@pytest.mark.parametrize("name", list(IMPL))
def test_core_impl(name):
    v = IMPL[name]
    grp = [g for g in TRACE_GROUPS if g[0] == v["sets"]]
    if not grp:
        pytest.skip("trace set %s not found under $TSR_TRACES" % v["sets"])
    gid, cfg, paths = grp[0]
    tests = _selected(TRACE_TESTS)
    if not tests:
        pytest.skip("no selected test")
    params = core_params(cfg)
    params.update(v["params"])
    env = dict(cfg=cfg, name=name, traces=paths, n_random=v["n_random"], passes=v["passes"])
    if "RD_OUTSTANDING" in v["params"]:
        env["rd_outstanding"] = v["params"]["RD_OUTSTANDING"]
    run("tsr_core", __name__, parameters=params, name="core_" + name, testcase=tests,
        extra_env={"TSR_CORE": json.dumps(env)})


@pytest.mark.parametrize("name", list(DIRECTED))
def test_core_directed(name):
    c = DIRECTED[name]
    tests = _selected(c["tests"])
    if not tests:
        pytest.skip("no selected test")
    run("tsr_core", __name__, parameters=core_params(c["cfg"]), name="core_" + name, testcase=tests,
        extra_env={"TSR_CORE": json.dumps(dict(cfg=c["cfg"], name=name, traces=[],
                                               n_random=c["n_random"]))})


def core_env() -> dict:
    return json.loads(os.environ["TSR_CORE"])


# =============================================================================================
# Harness
# =============================================================================================
def prog_bytes(insns) -> bytes:
    return b"".join(isa.insn_bytes(w) for w in insns)


def raw(*w):
    return list(w) + [0] * (8 - len(w))


def enc_mvm(fmt, flags, dst, x, woff, N, K, T, dstride, ra=0, rb=0, rc=0):
    return [isa.w0(isa.OP_MVM, sub=fmt, flags=flags, ra=ra, rb=rb, rc=rc), dst & M32, x & M32,
            woff & M32, N & M32, (K & 0xFFFF) | ((T & 0xF) << 16), dstride & M32, 0]


MEM_INPUTS = ("m_rd_req_ready", "m_rd_rsp_valid", "m_rd_rsp_data", "m_rd_rsp_id", "m_rd_rsp_last",
              "m_wr_req_ready", "m_wr_dat_ready", "m_wr_rsp_valid", "m_wr_rsp_id")


async def reset_core(dut):
    """Drive the memory-side inputs and the CSR port to 0, then clock + synchronous reset. Without
    this the inputs keep Verilator's random initial values until the first MemModel drives them
    (e.g. a random wr_rsp_valid right after reset trips tsr_dma's assertion)."""
    for n in MEM_INPUTS:
        getattr(dut, n).value = 0
    dut.csr_valid.value = 0
    dut.csr_we.value = 0
    dut.csr_addr.value = 0
    dut.csr_wdata.value = 0
    await start_clock_and_reset(dut)


class Outcome:
    def __init__(self, final, retired, regs, cycles, perf):
        self.final, self.retired, self.regs, self.cycles, self.perf = final, retired, regs, cycles, perf


class CoreH:
    """tsr_core + MemModel + CsrDriver (+ an ISA simulator in lockstep for check())."""

    def __init__(self, dut, cfg: dict, seed: int | None = None, mem_kw: dict | None = None):
        self.dut, self.cfg = dut, dict(cfg)
        self.num_ch = cfg["num_ch"]
        self.spm_lines = cfg["spm_bytes"] // 64
        self.lim = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.seed = tb_seed() if seed is None else seed
        self.rnd = random.Random(self.seed * 104729 + 3)
        self.mem_kw = dict(mem_kw or {})
        self.rd_out = int(core_env().get("rd_outstanding", cfg["rd_outstanding"]))   # RTL limit
        self.has_mvm = int(core_env().get("has_mvm", 0))
        self.rst_pipe = int(core_env().get("rst_pipe", 0))   # tsr_core RST_PIPE of the build (P10)
        self.mem = None
        self.new_mem()
        self.csr = CsrDriver(dut, timeout_cycles=200)
        self.sim = None
        self.n_perf = isa.perf_count(self.num_ch)

    @classmethod
    async def create(cls, dut, **kw):
        await reset_core(dut)
        env = core_env()
        return cls(dut, env["cfg"], **kw)

    def new_sim(self):
        if self.sim is not None:
            self.sim.close()
        self.sim = simlib.Sim(self.cfg)
        return self.sim

    def new_mem(self, **kw):
        """Fresh memory model (empty DRAM)."""
        if self.mem is not None:
            self.mem.stop()
        k = dict(seed=self.rnd.getrandbits(31))
        k.update(self.mem_kw)
        k.update(kw)
        self.mem = MemModel(self.dut, num_ch=self.num_ch, addr_w=self.cfg["addr_beat_w"],
                            rd_outstanding=self.rd_out, limit_beats=self.lim, **k).start()
        return self.mem

    def retime_mem(self, **kw):
        """Change the memory model's timing knobs, keeping its DRAM content (between runs)."""
        old = self.mem
        k = dict(seed=self.rnd.getrandbits(31))
        k.update(self.mem_kw)
        k.update(kw)
        old.stop()
        new = MemModel(self.dut, num_ch=self.num_ch, addr_w=self.cfg["addr_beat_w"],
                       rd_outstanding=self.rd_out, limit_beats=self.lim, **k)
        for c in range(self.num_ch):
            new.ch[c].mem = old.ch[c].mem
        self.mem = new.start()
        return new

    # ---- CSR / state helpers ----
    async def rd(self, addr):
        return await self.csr.read(addr)

    async def wr(self, addr, val):
        await self.csr.write(addr, val & M32)

    async def final(self):
        return await self.csr.final_state()

    async def perf(self) -> list:
        return [await self.csr.read64(isa.perf_csr(i)) for i in range(self.n_perf)]

    def regs(self) -> list:
        v = int(self.dut.u_cmdproc.rf_q.value)
        return [0] + [(v >> (32 * (i - 1))) & M32 for i in range(1, 16)]

    def busy(self) -> int:
        return int(self.dut.u_csr.busy_q.value)

    def outputs_active(self) -> int:
        d = self.dut
        return (int(d.m_rd_req_valid.value) | int(d.m_wr_req_valid.value) | int(d.m_wr_dat_valid.value)
                | int(d.csr_ready.value) | int(d.done.value))

    async def hw_reset(self, rnd, what: str, minimal: bool = False):
        """Hardware reset through rst_n (synchronous; the MemModel resets with it, as the platform
        resets the memory side together with the core). Reset contract of docs/rtl-m2.md P10 with
        P = RST_PIPE: rst_n low for >= P + 1 rising edges (here exactly P + 1 when `minimal`, else
        P + 2 .. P + 7); the outputs (memory requests, write data, csr_ready, done) are quiet from
        the falling edge after the (P + 1)-th low edge on, through the rest of the pulse and the
        P + 5 cycles after the release (the core leaves reset P cycles after it; no CSR access
        here). Returns at the end of those P + 5 cycles."""
        dut, p = self.dut, self.rst_pipe
        n_low = p + 1 if minimal else rnd.randint(p + 2, p + 7)
        await RisingEdge(dut.clk)
        dut.rst_n.value = 0
        for k in range(n_low + p + 5):            # rising edge k + 1 after rst_n fell
            await RisingEdge(dut.clk)
            if k == n_low - 1:
                dut.rst_n.value = 1               # rst_n was low at edges 1 .. n_low
            await FallingEdge(dut.clk)
            if k >= p:
                assert self.outputs_active() == 0, "output active %s reset (%s, cycle %d, rst_n low %d " \
                    "cycles, RST_PIPE %d)" % ("during" if k < n_low else "after", what, k, n_low, p)

    # ---- runs ----
    def load(self, prog: bytes, cmd_addr: int, init=()):
        if prog and cmd_addr * 64 < self.cfg["ch_bytes"]:
            n = min(len(prog), self.cfg["ch_bytes"] - cmd_addr * 64)
            self.mem.load(0, cmd_addr * 64, prog[:n])
        for ch, off, data in init:
            self.mem.load(ch, off, data)

    async def setup_run(self, cmd_addr, count, args=None):
        args = [(a & M32) for a in (list(args or []) + [0] * 8)[:8]]
        await self.wr(isa.CSR_CMD_ADDR_LO, cmd_addr & M32)
        await self.wr(isa.CSR_CMD_ADDR_HI, (cmd_addr >> 32) & M32)
        await self.wr(isa.CSR_CMD_COUNT, count)
        for i, a in enumerate(args):
            await self.wr(isa.CSR_ARG0 + 4 * i, a)

    async def launch(self, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START):
        """Load, write the run CSRs, then CTRL. Returns the PERF values before the run."""
        count = len(prog) // 32 if count is None else count
        self.load(prog, cmd_addr, init)
        self.mem.clear_tracking()
        p0 = await self.perf()
        await self.setup_run(cmd_addr, count, args)
        await self.wr(isa.CSR_CTRL, ctrl)
        return p0

    async def finish(self, p0, timeout=2_000_000) -> Outcome:
        cycles = await wait_level(self.dut.clk, self.dut.done, 1, timeout)
        self.mem.check(idle=True)          # DONE implies a quiescent memory port (P4)
        final = await self.final()
        p1 = await self.perf()
        d = [b - a for a, b in zip(p0, p1)]
        return Outcome(final, d[isa.PERF_RETIRED], self.regs(), cycles, d)

    async def run(self, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START,
                  during=None, timeout=2_000_000) -> Outcome:
        p0 = await self.launch(prog, cmd_addr, args, count, init, ctrl)
        if during is not None:
            await during()
        return await self.finish(p0, timeout)

    def compare(self, out: Outcome, res, what: str, prog=b"", regs=True, final=True, sources=None,
                n_mvm=None) -> list:
        """sources: also check the read beats per memory-side source at the port (DMA == LD
        beats, WS == rows_c x ROW_BEATS per executed MVM; replay.compare check_sources). Default:
        on for HAS_MVM = 1 builds. Needs res.trace to hold the executed program. n_mvm: executed
        MVMs (default: derived from res.trace, which must then hold the executed program)."""
        errs = []
        if final:
            for k in ("status", "err_code", "err_pc", "err_info", "pc"):
                if out.final[k] != res.final[k]:
                    errs.append("%s = 0x%x, simulator 0x%x" % (k, out.final[k], res.final[k]))
            if out.retired != res.retired:
                errs.append("retired %d, simulator %d" % (out.retired, res.retired))
        if regs:
            for i in range(1, 16):
                if out.regs[i] != res.regs[i]:
                    errs.append("R%d = 0x%x, simulator 0x%x" % (i, out.regs[i], res.regs[i]))
        # LD beats per channel must equal the simulator's (spec 성능 카운터); channel 0 also
        # carries instruction fetch, which may differ.
        for c in range(1, self.num_ch):
            got = out.perf[isa.perf_ch_rd_beats(c)]
            want = res.perf[isa.perf_ch_rd_beats(c)]
            if got != want:
                errs.append("PERF_CH_RD_BEATS[%d] delta %d, simulator %d" % (c, got, want))
        for i in (isa.PERF_VU_CYCLES, isa.PERF_WAITM_CYCLES) + (() if self.has_mvm else (isa.PERF_MVM_CYCLES,)):
            if out.perf[i]:
                errs.append("PERF counter %d delta %d (no VU / WAITM; no MVM in M2)" % (i, out.perf[i]))
        if self.has_mvm:
            # PERF_MVM_CYCLES (docs/rtl-m3.md §3.0): cycles the MVM unit holds an instruction, so
            # > 0 exactly when an MVM was dispatched; never more than the run's BUSY cycles.
            n_mvm = executed_mvms(res.trace) if n_mvm is None else n_mvm
            mc = out.perf[isa.PERF_MVM_CYCLES]
            if (mc > 0) != (n_mvm > 0) or mc > out.perf[isa.PERF_CYCLES]:
                errs.append("PERF_MVM_CYCLES %d with %d executed MVM(s), PERF_CYCLES %d"
                            % (mc, n_mvm, out.perf[isa.PERF_CYCLES]))
        fin = out.final if final else dict(res.final)
        errs += replay.compare(res.trace, self.mem, fin,
                               check_sources=bool(self.has_mvm) if sources is None else sources)
        if errs and prog:
            n = min(len(prog) // 32, 64)
            self.dut._log.error("%s program:\n%s", what, "\n".join(
                "%4d  %s" % (i, isa.disasm(isa.insn_words(prog[32 * i:32 * i + 32]))) for i in range(n)))
        return errs

    def sim_run(self, prog, cmd_addr=0, args=None, init=(), count=None, ctrl=isa.CTRL_START):
        sim = self.sim
        p0 = [sim.perf(i) for i in range(self.n_perf)]
        res = sim.run(prog, cmd_addr, args, init, count, ctrl)
        res.perf = [sim.perf(i) - a for i, a in enumerate(p0)]
        return res

    async def check(self, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START,
                    what="run", expect=None, during=None) -> Outcome:
        """Run on the simulator and on the RTL; compare final CSRs, PC, retired, R1..R15, LD beats
        of channels >= 1, and the DRAM trace rules (replay.compare)."""
        res = self.sim_run(prog, cmd_addr, args, init, count, ctrl)
        out = await self.run(prog, cmd_addr, args, count, init, ctrl, during=during)
        errs = self.compare(out, res, what, prog)
        for k, v in (expect or {}).items():
            if res.final[k] != v:
                errs.append("test expectation %s = 0x%x, simulator gives 0x%x" % (k, v, res.final[k]))
        assert not errs, "%s: %s" % (what, "; ".join(errs))
        return out

    # ---- SPM lockstep with the simulator ----
    def _spm_chunk(self):
        """(DRAM base beat, chunk lines, command beat) for SPM fill / dump runs on channel 0."""
        chunk = min(self.spm_lines, 4096, max(1, (self.lim - 1) // 2))
        return 0, chunk, chunk

    async def spm_fill(self):
        """Fill the RTL SPM with the simulator's initial 0xA5 pattern (LD from a 0xA5 area that is
        loaded into both DRAMs), so programs that read unwritten SPM stay comparable."""
        base, chunk, cmd = self._spm_chunk()
        fill = b"\xA5" * (64 * chunk)
        self.sim.load(0, base * 64, fill)
        self.mem.load(0, base * 64, fill)
        insns, line = [], 0
        while line < self.spm_lines:
            n = min(chunk, self.spm_lines - line)
            insns.append(isa.enc_ld(0, line, base, n))
            line += n
        insns.append(isa.enc_end())
        await self.check(prog_bytes(insns), cmd, what="SPM fill", expect=dict(status=DONE))

    async def spm_compare(self, what="SPM dump"):
        """ST the whole SPM (chunk by chunk) on both and compare the written DRAM."""
        base, chunk, cmd = self._spm_chunk()
        line = 0
        while line < self.spm_lines:
            n = min(chunk, self.spm_lines - line)
            await self.check(prog_bytes([isa.enc_st(0, line, base, n), isa.enc_end()]), cmd,
                             what="%s lines %d..%d" % (what, line, line + n - 1), expect=dict(status=DONE))
            line += n


def qact_overlap(prog: bytes, args, n_exec: int):
    """pc of the first executed QACT whose destination overlaps its source, else None. Such a
    QACT is undefined (spec 구현 명확화 "겹치는 피연산자"): the simulator reads and writes token by
    token, the RTL pipelines tokens. n_exec = number of instructions the run executes (registers
    are a function of the program and ARGs only, so they are tracked here)."""
    R = [0] * 8 + [a & M32 for a in (list(args or []) + [0] * 8)[:8]]
    for pc in range(min(n_exec, len(prog) // 32)):
        w = isa.insn_words(prog[32 * pc:32 * pc + 32])
        f = isa.w0_fields(w[0])
        op = f["opcode"]
        if op == isa.OP_LI and f["rd"]:
            R[f["rd"]] = w[1]
        elif op == isa.OP_ADDI and f["rd"]:
            R[f["rd"]] = (R[f["ra"]] + w[1]) & M32
        elif op == isa.OP_QACT:
            kind, K, T = f["sub"] & 1, w[3], w[4]
            dst, src = (R[f["ra"]] + w[1]) & M32, (R[f["rb"]] + w[2]) & M32
            stride = w[5] or K // 16
            d_end = dst + T * isa.qa_lines(kind, K)
            s_end = src + (T - 1) * stride + K // 16
            if dst < s_end and src < d_end:
                return pc
    return None


POLL_CSRS = [isa.CSR_ID, isa.CSR_VERSION, isa.CSR_CFG0, isa.CSR_CFG1, isa.CSR_CFG2, isa.CSR_CAPS,
             isa.CSR_CTRL, isa.CSR_STATUS, isa.CSR_CMD_ADDR_LO, isa.CSR_CMD_ADDR_HI, isa.CSR_CMD_COUNT,
             isa.CSR_ERR_PC, isa.CSR_ERR_CODE, isa.CSR_ERR_INFO, isa.CSR_PC, isa.CSR_ARG0 + 12, 0x0FC,
             0x7FE]


async def poll_csrs(h, rnd):
    """Read random CSRs (incl. PERF LO/HI, unmapped, unaligned) until the run ends; check the
    values that cannot change during a run."""
    perf = [isa.perf_csr(i) + 4 * hi for i in range(h.n_perf) for hi in (0, 1)]
    while not int(h.dut.done.value) and h.busy():
        off = rnd.choice(POLL_CSRS + perf)
        v = await h.rd(off)
        if off == isa.CSR_ID:
            assert v == isa.ID_MAGIC
        elif off == isa.CSR_STATUS:
            assert v in (isa.STATUS_BUSY, DONE, DONEERR), "STATUS 0x%x during a run" % v
        elif off in (isa.CSR_CTRL, 0x0FC, 0x7FE):
            assert v == 0, "CSR 0x%03x reads 0x%x" % (off, v)
        if rnd.random() < 0.3:
            await ClockCycles(h.dut.clk, rnd.randint(1, 50))


def expect_final(out, what, status=None, code=None, pc=None, info=None, cur_pc=None, retired=None):
    f = out.final
    want = dict(status=status, err_code=code, err_pc=pc, err_info=info, pc=cur_pc)
    bad = ["%s 0x%x (want 0x%x)" % (k, f[k], v) for k, v in want.items() if v is not None and f[k] != v]
    if retired is not None and out.retired != retired:
        bad.append("retired %d (want %d)" % (out.retired, retired))
    assert not bad, "%s: %s" % (what, ", ".join(bad))


def io_beat(h):
    """A DRAM beat for test data that leaves room below the limit on every configuration."""
    return min(isa.REGION_IO_OFF // 64, h.lim // 4)


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def core_csr_consts(dut):
    """ID / VERSION / CFG0..2 / CAPS equal the elaboration parameters (spec CSR map); STATUS,
    ERR_*, PC 0 and done low after reset."""
    h = await CoreH.create(dut)
    cfg = h.cfg
    assert int(dut.done.value) == 0
    want = {
        isa.CSR_ID: isa.ID_MAGIC,
        isa.CSR_VERSION: (isa.ISA_MAJOR << 8) | isa.ISA_MINOR,
        isa.CSR_CFG0: cfg["num_ch"] | (cfg["lanes"] << 8) | (cfg["t_max"] << 20),
        isa.CSR_CFG1: cfg["spm_bytes"] // 64,
        isa.CSR_CFG2: cfg["vl"] | (cfg["nsreg"] << 8) | ((cfg["k_max"] // 256) << 16),
        isa.CSR_CAPS: 0,
        isa.CSR_STATUS: 0, isa.CSR_ERR_CODE: 0, isa.CSR_ERR_PC: 0, isa.CSR_ERR_INFO: 0, isa.CSR_PC: 0,
    }
    sim = simlib.Sim(cfg)
    bad = []
    for off, v in want.items():
        got = await h.rd(off)
        if got != v or got != sim.csr_read(off):
            bad.append("0x%03x = 0x%08x, expected 0x%08x (simulator 0x%08x)" % (off, got, v, sim.csr_read(off)))
    sim.close()
    assert not bad, "; ".join(bad)
    for v in await h.perf():
        assert v < 64, "PERF counter not reset (%d)" % v      # PERF_CYCLES counts BUSY cycles only
    dut._log.info("CSR constants ok for %s", core_env()["name"])
    h.mem.stop()


@cocotb.test()
async def core_csr_map(dut):
    """Every 12-bit CSR offset vs the simulator, after reset and after writing every offset
    (except CTRL) with random values (RO / unmapped / unaligned writes ignored)."""
    h = await CoreH.create(dut)
    sim = h.new_sim()

    async def sweep(tag):
        bad = []
        for off in range(1 << isa.CSR_AW):
            got, want = await h.rd(off), sim.csr_read(off)
            if got != want:
                bad.append("0x%03x: 0x%08x (simulator 0x%08x)" % (off, got, want))
        assert not bad, "%s: %d CSR offsets differ, first %s" % (tag, len(bad), bad[:6])

    await sweep("after reset")
    rnd = random.Random(h.seed + 99)
    for off in range(1 << isa.CSR_AW):
        if off == isa.CSR_CTRL:
            continue
        v = rnd.getrandbits(32)
        await h.wr(off, v)
        sim.csr_write(off, v)
    await sweep("after writing every offset")
    assert int(dut.done.value) == 0 and not h.busy()
    h.mem.stop()


@cocotb.test()
async def core_traces(dut):
    """Every trace of this configuration, replayed back to back without reset under several
    memory regimes; plus PERF deltas against the trace's perf record."""
    env = core_env()
    await reset_core(dut)
    filt = os.environ.get("TSR_CORE_TRACE_FILTER")
    paths = [p for p in env["traces"] if not filt or filt in p]
    passes = int(os.environ.get("TSR_CORE_PASSES", env.get("passes", 3)))
    rd_out = env.get("rd_outstanding")          # RTL RD_OUTSTANDING when it differs from the trace
    csr = CsrDriver(dut, timeout_cycles=200)
    num_ch = env["cfg"]["num_ch"]
    n_perf = isa.perf_count(num_ch)
    seed = tb_seed()
    stats = collections.Counter()
    cyc_total, t0 = 0, time.time()
    worst = (0, "")
    max_out = 0
    for k, path in enumerate(paths):
        tr = trace.load(path)
        assert tr.config == env["cfg"], "%s: configuration differs from the build" % path
        if rd_out is not None:                  # the memory model checks the RTL's limit
            tr.config = dict(tr.config, rd_outstanding=rd_out)
        for ps in range(passes):
            regime = REGIME_ORDER[(k + ps) % len(REGIME_ORDER)] if passes < len(REGIME_ORDER) \
                else REGIME_ORDER[ps % len(REGIME_ORDER)]
            s = (seed * 1000003 + k * 7919 + ps * 104729) & 0x7FFFFFFF
            p0 = [await csr.read64(isa.perf_csr(i)) for i in range(n_perf)]
            try:
                res = await replay.run_trace(dut, tr, csr=csr, seed=s, mem_kwargs=REGIMES[regime])
            except Exception:
                dut._log.error("trace %s failed under regime %s (seed %d):\n%s", path, regime, s, tr.disasm())
                raise
            st = res["final"]
            p1 = [await csr.read64(isa.perf_csr(i)) for i in range(n_perf)]
            d = [b - a for a, b in zip(p0, p1)]
            errs = []
            if tr.perf is not None:
                if d[isa.PERF_RETIRED] != tr.perf["retired"]:
                    errs.append("retired %d, trace %d" % (d[isa.PERF_RETIRED], tr.perf["retired"]))
                for c in range(1, num_ch):
                    if d[isa.perf_ch_rd_beats(c)] != tr.perf["ch_rd_beats"][c]:
                        errs.append("PERF_CH_RD_BEATS[%d] %d, trace %d"
                                    % (c, d[isa.perf_ch_rd_beats(c)], tr.perf["ch_rd_beats"][c]))
            mstats = res["mem_stats"]
            for c in range(num_ch):
                if d[isa.perf_ch_rd_beats(c)] != mstats[c]["rd_beats"]:
                    errs.append("PERF_CH_RD_BEATS[%d] %d, port beats %d"
                                % (c, d[isa.perf_ch_rd_beats(c)], mstats[c]["rd_beats"]))
                if d[isa.perf_ch_wait(num_ch, c)] > d[isa.PERF_CYCLES]:
                    errs.append("PERF_CH_WAIT[%d] > PERF_CYCLES" % c)
            has_mvm = int(env.get("has_mvm", 0))
            for i in (isa.PERF_VU_CYCLES, isa.PERF_WAITM_CYCLES) + (() if has_mvm else (isa.PERF_MVM_CYCLES,)):
                if d[i]:
                    errs.append("PERF counter %d = %d (no VU / WAITM; no MVM in M2)" % (i, d[i]))
            if has_mvm:
                n_mvm = executed_mvms(tr)
                if (d[isa.PERF_MVM_CYCLES] > 0) != (n_mvm > 0) or d[isa.PERF_MVM_CYCLES] > d[isa.PERF_CYCLES]:
                    errs.append("PERF_MVM_CYCLES %d with %d executed MVM(s), PERF_CYCLES %d"
                                % (d[isa.PERF_MVM_CYCLES], n_mvm, d[isa.PERF_CYCLES]))
                stats["mvm"] += n_mvm
                stats["mvm_cycles"] += d[isa.PERF_MVM_CYCLES]
            if not (res["cycles"] <= d[isa.PERF_CYCLES] <= res["cycles"] + 400):
                errs.append("PERF_CYCLES %d vs %d cycles to done" % (d[isa.PERF_CYCLES], res["cycles"]))
            assert not errs, "%s (%s): %s" % (path, regime, "; ".join(errs))
            stats[regime] += 1
            stats["err" if st["err_code"] else "end"] += 1
            max_out = max([max_out] + [m["max_outstanding"] for m in mstats.values()])
            cyc_total += res["cycles"]
            if res["cycles"] > worst[0]:
                worst = (res["cycles"], tr.name)
        if (k + 1) % 20 == 0:
            dut._log.info("core_traces: %d/%d traces (%.0f s)", k + 1, len(paths), time.time() - t0)
    dut._log.info("core_traces %s: %d traces x %d passes ok %s, %d cycles total, longest %d (%s), "
                  "max reads outstanding on a channel %d, %.0f s", env["name"], len(paths), passes,
                  dict(stats), cyc_total, worst[0], worst[1], max_out, time.time() - t0)
    assert len(paths) > 0 or filt


@cocotb.test()
async def core_status_w1c(dut):
    """STATUS W1C: DONE / ERR only, BUSY / WAITING writes ignored; done output == STATUS.DONE;
    START clears ERR_*; START while BUSY is ignored."""
    h = await CoreH.create(dut)
    h.new_sim()
    io = io_beat(h)
    ch = min(1, h.num_ch - 1)
    p = prog_bytes([isa.enc_ld(ch, 10, io, 3), isa.enc_st(ch, 10, io + 64, 3), isa.enc_end()])
    init = [(ch, io * 64, bytes(range(192)))]
    out = await h.check(p, 0, init=init, what="LD ST END", expect=dict(status=DONE))
    expect_final(out, "LD ST END", status=DONE, code=0, pc=0, info=0, cur_pc=2, retired=3)
    assert int(dut.done.value) == 1
    await h.wr(isa.CSR_STATUS, isa.STATUS_BUSY | isa.STATUS_WAITING)
    assert await h.rd(isa.CSR_STATUS) == DONE and int(dut.done.value) == 1
    await h.wr(isa.CSR_STATUS, isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == DONE and int(dut.done.value) == 1
    await h.wr(isa.CSR_STATUS, isa.STATUS_DONE)
    assert await h.rd(isa.CSR_STATUS) == 0 and int(dut.done.value) == 0
    # Error run: DONE|ERR; clearing ERR alone keeps DONE and ERR_CODE; DONE alone keeps ERR.
    out = await h.check(prog_bytes([isa.enc_nop(), raw(0xEE)]), 32, what="illegal opcode",
                        expect=dict(status=DONEERR, err_code=isa.ERR_ILLEGAL_OP, err_pc=1))
    await h.wr(isa.CSR_STATUS, isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == DONE and await h.rd(isa.CSR_ERR_CODE) == isa.ERR_ILLEGAL_OP
    assert int(dut.done.value) == 1
    out = await h.check(prog_bytes([raw(0xEE)]), 33, what="illegal opcode 2")
    await h.wr(isa.CSR_STATUS, isa.STATUS_DONE)
    assert await h.rd(isa.CSR_STATUS) == isa.STATUS_ERR and int(dut.done.value) == 0
    await h.wr(isa.CSR_STATUS, isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == 0 and await h.rd(isa.CSR_ERR_PC) == 0
    # START clears ERR_* (and the stale DONE|ERR) at once.
    out = await h.check(prog_bytes([isa.enc_ld(0, 0, h.lim, 1)]), 34, what="DRAM range",
                        expect=dict(err_code=isa.ERR_DRAM_RANGE))
    assert await h.rd(isa.CSR_ERR_INFO) == h.lim & M32
    out = await h.check(prog_bytes([isa.enc_end()]), 35, what="START clears ERR_*")
    expect_final(out, "START clears ERR_*", status=DONE, code=0, pc=0, info=0, cur_pc=0, retired=1)

    # START while BUSY is ignored: the run keeps its CMD_COUNT; a slow memory keeps it busy.
    h.retime_mem(rd_lat=(300, 300), p_stall=0.0)
    p = prog_bytes([isa.enc_ld(ch, 0, io, 2), isa.enc_nop(), isa.enc_end()])
    p0 = await h.launch(p, 64, init=[(ch, io * 64, bytes(128))], count=2)
    await ClockCycles(dut.clk, 30)
    assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY and int(dut.done.value) == 0
    await h.wr(isa.CSR_CMD_COUNT, 3)
    await h.wr(isa.CSR_CTRL, isa.CTRL_START)
    await h.wr(isa.CSR_STATUS, isa.STATUS_BUSY | isa.STATUS_DONE | isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY
    out = await h.finish(p0)
    expect_final(out, "START while BUSY", status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=2, info=0,
                 cur_pc=2, retired=2)
    assert await h.rd(isa.CSR_CMD_COUNT) == 3
    h.retime_mem()
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_soft_reset(dut):
    """SOFT_RESET idle (STATUS, ERR_*, PC, PERF, R1..R15 := 0; CMD_* / ARG* kept), SOFT_RESET|START,
    and SOFT_RESET while BUSY at random points (P3: abort at the next boundary, no DONE, then the
    reset); runs afterwards match the simulator."""
    h = await CoreH.create(dut)
    sim = h.new_sim()
    io = io_beat(h)
    p = prog_bytes([isa.enc_li(1, 11), isa.enc_li(2, 22), isa.enc_ld(0, 0, io, 4), raw(0xEE)])
    out = await h.check(p, 0, args=[5, 1001], init=[(0, io * 64, bytes(range(256)))], what="error run")
    expect_final(out, "error run", status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=3, cur_pc=3, retired=3)
    assert out.regs[1] == 11 and out.regs[8] == 5 and out.regs[9] == 1001
    perf = await h.perf()
    assert perf[isa.PERF_RETIRED] >= 3 and perf[isa.PERF_CYCLES] > 0 and perf[isa.perf_ch_rd_beats(0)] >= 4
    await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
    sim.csr_write(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
    st = await h.final()
    assert st == dict(status=0, err_code=0, err_pc=0, err_info=0, pc=0), st
    assert all(v == 0 for v in await h.perf()), "PERF not cleared"
    assert h.regs() == [0] * 16 and int(dut.done.value) == 0
    assert await h.rd(isa.CSR_CMD_COUNT) == 4 and await h.rd(isa.CSR_ARG0 + 4) == 1001
    for off in range(0, 0x80, 4):
        assert await h.rd(off) == sim.csr_read(off), "CSR 0x%03x after SOFT_RESET" % off

    # SOFT_RESET | START when idle: reset, then a normal run.
    out = await h.check(prog_bytes([isa.enc_li(3, 33), isa.enc_end()]), 64,
                        ctrl=isa.CTRL_SOFT_RESET | isa.CTRL_START, what="SOFT_RESET|START")
    assert out.regs[3] == 33 and await h.csr.read64(isa.perf_csr(isa.PERF_RETIRED)) == 2

    # SOFT_RESET while BUSY at random points of a long program (and with START in the same write).
    rnd = random.Random(h.seed * 5 + 1)
    n_lines = min(256, h.spm_lines // 2, h.lim // 4)
    body = [isa.enc_li(4, 44), isa.enc_ld(0, 0, io, n_lines), isa.enc_st(0, 0, io + n_lines, n_lines),
            isa.enc_ld(h.num_ch - 1, n_lines, io, n_lines), isa.enc_li(5, 55),
            isa.enc_st(h.num_ch - 1, n_lines, io + 2 * n_lines, n_lines), isa.enc_end()]
    prog = prog_bytes(body)
    init = [(c, io * 64, rnd.randbytes(64 * n_lines)) for c in range(h.num_ch)]
    while_busy = 0
    n_sr = 24
    for i in range(n_sr):
        ctrl = isa.CTRL_SOFT_RESET | (isa.CTRL_START if i % 3 == 2 else 0)
        h.retime_mem(**(REGIMES["slow"] if i % 2 else {}))
        await h.launch(prog, 128, args=[7, 8], init=init)
        await ClockCycles(dut.clk, rnd.choice([rnd.randint(0, 40), rnd.randint(0, 600), rnd.randint(0, 3000)]))
        while_busy += h.busy()
        # From the end of this CTRL write on, done must stay low: an idle SOFT_RESET clears
        # STATUS at once, one while BUSY ends the run without DONE (START in it ignored).
        await h.wr(isa.CSR_CTRL, ctrl)
        restarted = False
        for _ in range(400000):
            await FallingEdge(dut.clk)
            if int(dut.done.value):
                # Only legal when the run had ended before the write and SOFT_RESET|START then
                # reset the core and started the program again: counters restart from 0.
                st = await h.final()
                assert ctrl & isa.CTRL_START and st["status"] == DONE and st["err_code"] == 0, \
                    "done rose on a run ended by SOFT_RESET (#%d, ctrl %d): %s" % (i, ctrl, st)
                assert await h.csr.read64(isa.perf_csr(isa.PERF_RETIRED)) == len(body)
                restarted = True
                break
            if not h.busy():
                break
        assert not h.busy(), "run did not end after SOFT_RESET"
        await ClockCycles(dut.clk, 10)
        h.mem.check(idle=True)
        if restarted:
            while_busy -= 1
            await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
        assert int(dut.done.value) == 0
        st = await h.final()
        assert st == dict(status=0, err_code=0, err_pc=0, err_info=0, pc=0), (i, st)
        assert all(v == 0 for v in await h.perf()), "PERF not cleared"
        assert h.regs() == [0] * 16, "registers not cleared"
        assert await h.rd(isa.CSR_ARG0) == 7 and await h.rd(isa.CSR_CMD_COUNT) == len(body)
        # A self-contained program afterwards matches a fresh simulator (SPM written before read).
        sim = h.new_sim()
        sim.csr_write(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
        h.mem.clear_tracking()
        h.retime_mem()
        p2 = prog_bytes([isa.enc_li(6, 66), isa.enc_ld(0, 3, io, 5), isa.enc_st(0, 3, io + 3 * n_lines + 8, 5),
                         isa.enc_end()])
        # Fresh simulator: give it the DRAM the program reads.
        out = await h.check(p2, 256, args=[7], init=[(0, io * 64, init[0][2][:320])],
                            what="run after SOFT_RESET while BUSY #%d" % i, expect=dict(status=DONE))
        assert out.regs[6] == 66 and out.regs[8] == 7
    dut._log.info("core_soft_reset: %d of %d SOFT_RESETs written while BUSY", while_busy, n_sr)
    assert while_busy >= n_sr // 2
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_hw_reset(dut):
    """Hardware reset (rst_n, synchronous, together with the memory side as the platform does) at
    random points of a run, with the RST_PIPE contract of docs/rtl-m2.md P10 (CoreH.hw_reset:
    pulses of >= RST_PIPE + 1 cycles, a third of them exactly that): from RST_PIPE cycles after
    rst_n falls no memory request, write data, csr_ready or done until the next CSR access; after
    it every CSR equals a fresh simulator's (CMD_* / ARG* / PERF / STATUS 0) and a following
    program matches a fresh simulator (SPM content is undefined after reset, P7: the program writes
    the SPM before reading it)."""
    h = await CoreH.create(dut)
    io = io_beat(h)
    rnd = random.Random(h.seed * 19 + 4)
    n_lines = min(256, h.spm_lines // 2, h.lim // 4)
    body = [isa.enc_li(4, 44), isa.enc_ld(0, 0, io, n_lines), isa.enc_st(0, 0, io + n_lines, n_lines),
            isa.enc_qact(isa.QA_A32, n_lines, 0, min(h.cfg["k_max"], 16 * (n_lines // 2)) // 32 * 32, 1),
            isa.enc_ld(h.num_ch - 1, 0, io, n_lines), isa.enc_st(h.num_ch - 1, 0, io + 2 * n_lines, n_lines),
            isa.enc_end()]
    prog = prog_bytes(body)
    init = [(c, io * 64, rnd.randbytes(64 * n_lines)) for c in range(h.num_ch)]
    mid = 0
    for i in range(12):
        h.retime_mem(**REGIMES[REGIME_ORDER[i % 3]])
        await h.launch(prog, 128, args=[7, 8], init=init)
        await ClockCycles(dut.clk, rnd.choice([rnd.randint(0, 30), rnd.randint(0, 400), rnd.randint(0, 3000)]))
        mid += h.busy()
        await h.hw_reset(rnd, "reset #%d" % i, minimal=(i % 3 == 2))
        h.mem.check(idle=True)
        sim = h.new_sim()
        bad = []
        for off in range(0, 0x200, 4):
            got, want = await h.rd(off), sim.csr_read(off)
            if got != want:
                bad.append("0x%03x = 0x%x (fresh simulator 0x%x)" % (off, got, want))
        assert not bad, "after reset #%d: %s" % (i, bad[:6])
        assert h.regs() == [0] * 16 and int(dut.done.value) == 0
        p2 = prog_bytes([isa.enc_li(6, 66), isa.enc_ld(0, 3, io, 5), isa.enc_st(0, 3, io + 3 * n_lines + 8, 5),
                         isa.enc_end()])
        h.retime_mem()
        out = await h.check(p2, 256, args=[9], init=[(0, io * 64, init[0][2][:320])],
                            what="run after reset #%d" % i, expect=dict(status=DONE))
        assert out.regs[6] == 66 and out.regs[8] == 9
    dut._log.info("core_hw_reset: %d of 12 resets during a run", mid)
    assert mid >= 6
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_abort(dut):
    """ABORT: START|ABORT aborts before pc 0; ABORT while idle has no effect on the next START;
    ABORT at random cycle offsets inside long programs stops at the next instruction boundary with
    ERR_PC = first instruction not executed, ERR_INFO 0, PC unchanged; the executed prefix has
    exactly the simulator's effects (DRAM writes, registers, SPM)."""
    h = await CoreH.create(dut)
    h.new_sim()
    await h.spm_fill()
    p = prog_bytes([isa.enc_li(1, 9), isa.enc_end()])
    out = await h.run(p, 0, ctrl=isa.CTRL_START | isa.CTRL_ABORT)
    expect_final(out, "START|ABORT", status=DONEERR, code=isa.ERR_ABORTED, pc=0, info=0, cur_pc=0, retired=0)
    assert out.regs[1] == 0
    await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)                        # idle: cleared by START
    out = await h.check(p, 0, what="ABORT while idle", expect=dict(status=DONE))
    assert out.regs[1] == 9

    rnd = random.Random(h.seed * 17 + 5)
    gen = ProgGen(random.Random(h.seed * 31 + 11), h.cfg, False)
    data = gen.init_data()
    await h.check(prog_bytes([isa.enc_end()]), gen.cmd_base, init=data, what="load data")
    aborted = finished = 0
    for i in range(60):
        # Long clean program (no errors), aborted somewhere.
        gen.e = 0.0
        args = [0] * 8
        n = rnd.randint(4, 24)
        gen.cmd_lo, gen.cmd_hi = gen.cmd_base, gen.cmd_base + 16
        while True:
            regs = [0] * 16
            insns = [gen.insn(regs) for _ in range(n)] + [isa.enc_end()]
            prog = prog_bytes(insns)
            if qact_overlap(prog, args, len(insns)) is None:    # overlap is undefined
                break
        cmd = gen.cmd_base
        regime = REGIME_ORDER[i % 3]
        h.retime_mem(**REGIMES[regime])
        p0 = await h.launch(prog, cmd, args)
        when = rnd.choice([rnd.randint(0, 60), rnd.randint(0, 400), rnd.randint(0, 2000), rnd.randint(0, 6000)])
        await ClockCycles(dut.clk, when)
        await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)
        out = await h.finish(p0)
        f = out.final
        if f["err_code"] == isa.ERR_ABORTED:
            k = f["err_pc"]
            aborted += 1
            assert f["status"] == DONEERR and f["err_info"] == 0, f
            assert out.retired == k, "retired %d, ERR_PC %d" % (out.retired, k)
            assert f["pc"] == (k - 1 if k > 0 else 0), "PC %d after ABORT at %d" % (f["pc"], k)
            # The simulator executes the prefix (k instructions, then END in place of pc k).
            sp = prog_bytes(insns[:k] + [isa.enc_end()])
            res = h.sim_run(sp, cmd, args)
            n_mvm = executed_mvms(res.trace)
            res.trace.cmd = prog            # the RTL may prefetch the whole command buffer
            errs = h.compare(out, res, "abort #%d" % i, prog, final=False, sources=False, n_mvm=n_mvm)
            assert res.final["err_code"] == 0, "prefix fails on the simulator: %s" % res.final
            assert not errs, "abort #%d (ERR_PC %d, %s, after %d cycles): %s" % (i, k, regime, when, "; ".join(errs))
        else:
            finished += 1
            res = h.sim_run(prog, cmd, args)
            errs = h.compare(out, res, "abort-late #%d" % i, prog)
            assert not errs, "abort-late #%d: %s" % (i, "; ".join(errs))
        # The run after an ABORT starts clean (abort flag cleared by START).
        out = await h.check(prog_bytes([isa.enc_li(1, i), isa.enc_end()]), cmd + 32, what="after abort #%d" % i,
                            expect=dict(status=DONE))
    h.retime_mem()
    await h.spm_compare("SPM after aborts")
    dut._log.info("core_abort: %d aborted mid-program, %d finished before the ABORT", aborted, finished)
    assert aborted >= 20
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_back_to_back(dut):
    """Programs back to back without reset: every unit and channel alternately, a program
    repeated gives identical results, errors in between do not disturb the next run."""
    h = await CoreH.create(dut)
    h.new_sim()
    await h.spm_fill()
    io = io_beat(h)
    rnd = random.Random(h.seed * 3 + 2)
    K = min(512, h.cfg["k_max"])
    xl = K // 16
    init = [(c, io * 64, rnd.randbytes(64 * 2 * xl)) for c in range(h.num_ch)]
    progs = []
    for c in range(h.num_ch):
        progs.append(prog_bytes([
            isa.enc_ld(c, 0, io, 2 * xl),
            isa.enc_qact(isa.QA_A32, 2 * xl, 0, K, 1),
            isa.enc_qact(isa.QA_A256, 3 * xl, xl, K - K % 256 or 256, 1),
            isa.enc_st(c, 2 * xl, io + 4 * xl, 2 * xl),
            isa.enc_end()]))
    progs.append(prog_bytes([isa.enc_ld(0, 0, h.lim, 1)]))                  # DRAM range
    progs.append(prog_bytes([isa.enc_li(1, 5), isa.enc_end()]))
    progs.append(prog_bytes([isa.enc_nop()] * 3))                            # past CMD_COUNT
    outs = {}
    for rep in range(3):
        for j, p in enumerate(progs):
            out = await h.check(p, 64 + 16 * j, init=init if rep == 0 else (), what="b2b %d.%d" % (rep, j))
            if rep and j < h.num_ch:
                assert out.final == outs[j], "repeat of program %d differs" % j
            outs[j] = out.final
    await h.spm_compare()
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_mvm_illegal(dut):
    """HAS_MVM = 0 (M2): MVM is ILLEGAL_OP at any position, valid fields or not; the preceding
    instructions take effect, nothing after it; ERR_INFO 0."""
    h = await CoreH.create(dut)
    h.new_sim()
    await h.spm_fill()
    io = io_beat(h)
    rnd = random.Random(h.seed * 7 + 3)
    init = [(0, io * 64, rnd.randbytes(64 * 64))]
    mvms = [enc_mvm(0, 0, 100, 0, io, 16, 32, 1, 0),                      # a legal M3 MVM
            enc_mvm(1, 1, 0, 0, 0, 0, 256, 1, 0),                         # N = 0, ACC
            enc_mvm(5, 0, 0, 0, 0, 1, 33, 9, 0),                          # BAD_FMT / K / T at M3
            enc_mvm(0, 0, M32, M32, M32, M32, 32, 1, 0),                  # range errors at M3
            [isa.w0(isa.OP_MVM, ra=1, rb=2, rc=3)] + [0] * 7,
            raw(isa.OP_MVM | 0xF0000000, 1, 2, 3, 4, 5, 6, 7)]            # reserved bits too
    for i, m in enumerate(mvms):
        pre = [isa.enc_li(1, 10 + i), isa.enc_ld(0, 0, io, 8), isa.enc_st(0, 0, io + 128, 8 - i % 3)][:i % 4]
        prog = prog_bytes(pre + [m, isa.enc_li(2, 99), isa.enc_end()])
        # Oracle: the simulator implements MVM (M1), so it runs the same program with the MVM
        # replaced by an undefined opcode (0x21): ILLEGAL_OP at the same pc, same prefix effects.
        sp = prog_bytes(pre + [raw(0x21), isa.enc_li(2, 99), isa.enc_end()])
        res = h.sim_run(sp, 64, init=init)
        out = await h.run(prog, 64, init=init)
        errs = h.compare(out, res, "MVM #%d" % i, prog)
        assert not errs, "MVM #%d: %s" % (i, "; ".join(errs))
        expect_final(out, "MVM #%d" % i, status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=len(pre), info=0,
                     cur_pc=len(pre), retired=len(pre))
        assert out.regs[2] == 0
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_perf(dut):
    """PERF counters (spec 성능 카운터, P5): PERF_CYCLES counts BUSY cycles, PERF_RETIRED and the
    LD beats per channel equal the simulator, channel-0 beats include fetch, wait cycles <=
    cycles, MVM / VU / WAITM 0; LO read snapshots HI."""
    h = await CoreH.create(dut)
    h.new_sim()
    io = io_beat(h)
    rnd = random.Random(h.seed * 11)
    n = min(200, h.lim // 8, h.spm_lines // 2)
    init = [(c, io * 64, rnd.randbytes(64 * n)) for c in range(h.num_ch)]
    insns = []
    for c in range(h.num_ch):
        insns += [isa.enc_ld(c, 0, io, n), isa.enc_st(c, 0, io + n, n)]
    prog = prog_bytes(insns + [isa.enc_end()])
    h.retime_mem(**REGIMES["slow"])
    out = await h.check(prog, 0, init=init, what="perf program", expect=dict(status=DONE))
    d = out.perf
    assert d[isa.PERF_RETIRED] == len(insns) + 1
    assert out.cycles <= d[isa.PERF_CYCLES] <= out.cycles + 400, (out.cycles, d[isa.PERF_CYCLES])
    fetch_beats = (len(insns) + 2) // 2         # ceil(CMD_COUNT / 2): fetch reads each beat once
    assert d[isa.perf_ch_rd_beats(0)] == n + fetch_beats, (d, n, fetch_beats)
    for c in range(h.num_ch):
        if c:
            assert d[isa.perf_ch_rd_beats(c)] == n
        w = d[isa.perf_ch_wait(h.num_ch, c)]
        assert 0 < w <= d[isa.PERF_CYCLES], (c, w)
    for i in (isa.PERF_MVM_CYCLES, isa.PERF_VU_CYCLES, isa.PERF_WAITM_CYCLES):
        assert d[i] == 0
    # Snapshot: HI read after LO returns the HI sampled with that LO.
    lo = await h.rd(isa.perf_csr(isa.PERF_CYCLES))
    hi = await h.rd(isa.perf_csr(isa.PERF_CYCLES) + 4)
    assert hi == 0 and lo > 0
    h.retime_mem()
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_throughput(dut):
    """Cycles from START to done with an ideal memory (always ready, latency 0..1): LD / ST
    streaming rate, QACT A32 / A256 at K_MAX, instruction overhead. Logged, with loose floors
    that catch performance regressions (exact values are not part of the contract)."""
    h = await CoreH.create(dut, mem_kw=dict(rd_lat=(0, 1), wr_lat=(0, 1), p_rd_req_ready=1.0,
                                            p_rd_rsp_valid=1.0, p_wr_req_ready=1.0,
                                            p_wr_dat_ready=1.0, p_wr_rsp_valid=1.0, p_stall=0.0,
                                            wr_order="any"))
    h.new_sim()
    io = io_beat(h)
    rnd = random.Random(h.seed)
    K = h.cfg["k_max"]
    n = min(4096, h.spm_lines // 4)
    xl = K // 16
    h.mem.load(0, io * 64, rnd.randbytes(64 * max(n, xl)))
    h.sim.load(0, io * 64, h.mem.read_bytes(0, io * 64, 64 * max(n, xl)))
    rows = []

    async def timed(what, insns, units, unit_name, floor=None, ceil=None):
        prog = prog_bytes(insns + [isa.enc_end()])
        out = await h.check(prog, 0, what=what, expect=dict(status=DONE))
        c = out.perf[isa.PERF_CYCLES]
        per = c / units
        rows.append("%-34s %8d cycles  %8.3f cycles/%s" % (what, c, per, unit_name))
        if floor is not None:
            assert units / c >= floor, "%s: %.3f %s/cycle < %.3f" % (what, units / c, unit_name, floor)
        if ceil is not None:
            assert per <= ceil, "%s: %.3f cycles/%s > %.3f" % (what, per, unit_name, ceil)

    big = n >= 2048                       # floors only where the per-instruction overhead is small
    await timed("LD %d lines" % n, [isa.enc_ld(0, 0, io, n)], n, "beat", floor=0.95 if big else None)
    await timed("ST %d lines" % n, [isa.enc_st(0, 0, io + 2 * n, n)], n, "beat", floor=0.95 if big else None)
    await timed("LD %d lines on every channel" % n, [isa.enc_ld(c, 0, io, n) for c in range(h.num_ch)],
                n * h.num_ch, "beat", floor=0.9 if big else None)
    await timed("LD x (K=%d, %d lines)" % (K, xl), [isa.enc_ld(0, 0, io, xl)], xl, "beat")
    await timed("QACT A256 K=%d" % K, [isa.enc_qact(isa.QA_A256, xl, 0, K, 1)], xl, "src line", ceil=1.5)
    await timed("QACT A32 K=%d" % K, [isa.enc_qact(isa.QA_A32, xl, 0, K, 1)], xl, "src line", ceil=9.0)
    await timed("64 x NOP", [isa.enc_nop()] * 64, 65, "insn", ceil=6.0)
    await timed("64 x LI/ADDI", [isa.enc_li(1, 5), isa.enc_addi(2, 1, 7)] * 32, 65, "insn", ceil=6.0)
    await timed("32 x LD 1 line", [isa.enc_ld(0, i, io + i, 1) for i in range(32)], 32, "insn")
    await timed("32 x ST 1 line", [isa.enc_st(0, i, io + 3 * n + i, 1) for i in range(32)], 32, "insn")
    dut._log.info("core_throughput %s (ideal memory):\n%s", core_env()["name"], "\n".join(rows))
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def core_random(dut):
    """Random programs (tb/common ProgGen: every M2 instruction, register-relative operands,
    reserved-field flips, illegal opcodes incl. MVM, range edges, fetch beyond the limit, missing
    END) on the RTL and the simulator in lockstep; SPM compared through ST dumps."""
    env = core_env()
    h = await CoreH.create(dut)
    h.new_sim()
    n = int(os.environ.get("TSR_CORE_RANDOM", env.get("n_random", 200)))
    spm_every = int(os.environ.get("TSR_CORE_SPM_EVERY", "0"))     # SPM dump compare period
    if n == 0:
        return
    t0 = time.time()
    await h.spm_fill()
    gen = ProgGen(random.Random(h.seed * 31 + 7), h.cfg, False)
    data = gen.init_data()
    await h.check(prog_bytes([isa.enc_end()]), gen.cmd_base, init=data, what="load data")
    codes = collections.Counter()
    retired = cycles = skipped = 0
    probe = simlib.Sim(h.cfg)          # control flow only (never data-dependent in M2)
    rpoll = random.Random(h.seed * 13 + 1)
    for i in range(n):
        if i % 50 == 0:
            regime = REGIME_ORDER[(i // 50) % len(REGIME_ORDER)]
            h.retime_mem(**REGIMES[regime])
        while True:
            prog, cmd, count, args = gen.program()
            pf = probe.run(prog, cmd, args, (), count).final
            n_exec = pf["err_pc"] if pf["err_code"] else pf["pc"] + 1
            if qact_overlap(prog, args, n_exec) is None:
                break
            skipped += 1                   # undefined program: QACT dst overlaps its src
        # Half of the runs: random CSR reads while the program runs (no side effect allowed).
        during = (lambda: poll_csrs(h, rpoll)) if i % 2 else None
        out = await h.check(prog, cmd, args, count=count, what="random program %d" % i, during=during)
        codes[out.final["err_code"]] += 1
        retired += out.retired
        cycles += out.perf[isa.PERF_CYCLES]
        if spm_every and (i + 1) % spm_every == 0:
            try:
                await h.spm_compare("SPM after random program %d" % i)
            except AssertionError:
                dut._log.error("SPM differs after random program %d (CMD_ADDR 0x%x, count %d, args %s, %s):\n%s",
                               i, cmd, count, [hex(a) for a in args], out.final, "\n".join(
                                   "%4d  %s" % (j, isa.disasm(isa.insn_words(prog[32 * j:32 * j + 32])))
                                   for j in range(min(len(prog) // 32, 64))))
                raise
    h.retime_mem()
    await h.spm_compare("SPM after random programs")
    probe.close()
    dut._log.info("core_random %s: %d programs, %d instructions retired, %d cycles, endings %s, "
                  "%d undefined programs (QACT dst/src overlap) skipped, %.0f s",
                  env["name"], n, retired, cycles,
                  {isa.ERR_NAMES[k]: v for k, v in sorted(codes.items())}, skipped, time.time() - t0)
    # Ending coverage (K_RANGE / T_RANGE endings are ~2 % of the programs: required only for
    # long runs, where missing them would be a generator problem rather than chance).
    need = []
    if n >= 100:
        need += [0, isa.ERR_ILLEGAL_OP, isa.ERR_SPM_RANGE, isa.ERR_DRAM_RANGE]
    if n >= 250:
        need += [isa.ERR_K_RANGE, isa.ERR_T_RANGE]
    for code in need:
        assert codes[code] > 0, "random programs never ended in %s" % isa.ERR_NAMES[code]
    h.sim.close()
    h.mem.stop()

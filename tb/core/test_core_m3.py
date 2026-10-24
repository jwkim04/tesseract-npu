"""Core-level verification of tsr_core with the MVM unit (HAS_MVM = 1; docs/rtl-m3.md §10.4, §13;
spec 검증 전략 "RTL 코어 전체": the ISA simulator is the oracle, trace replay must be bit-exact).

The DUT is the whole tsr_core with HAS_MVM = 1 (tsr_mvm: broadcast, NUM_CH clusters with weight
streamers, activation buffers, back ends, the gather) driven only through its boundary
(common.csr_driver.CsrDriver, common.mem_model.MemModel). The harness (CoreH, memory regimes,
helpers) is tb/core/test_core.py's; the M2 cocotb tests imported below run unchanged on the MVM
core (M2 behaviour with HAS_MVM = 1). tsr_core's simulation checks I1..I4 (one instruction in
flight, SPM / weight-stream traffic only while the unit holds the instruction, MVM done =>
quiescent and exactly rows_c x ROW_BEATS weight beats per channel, PERF_MVM_CYCLES meaning) are
armed in every run here: any violation fails the test.

pytest functions (one Verilator build each; all cocotb tests of a build run back to back):
  test_core_m3_traces[<sets>]   one build per distinct configuration of the traces under
                                $TSR_CORE_M3_TRACES (default traces/m3, traces/gen, traces/m2),
                                every trace a HAS_MVM = 1 core can replay: core_csr_consts,
                                core_traces (TSR_CORE_PASSES memory regimes per trace, default 4 =
                                every regime on every trace, 2 rotating for T_MAX 8 whose build
                                simulates at ~6 K cycles/s; replay.run_trace incl. the per-source
                                beat check, PERF deltas), m3_random (ProgGen m3 lockstep)
  test_core_m3_fastreplay       tb/fastreplay (C++): every traces/m3 + traces/gen trace under all
                                four regimes on the HAS_MVM = 1 core
  test_core_m3_impl[<variant>]  implementation parameters that must not change results (MVM
                                pipes 0 / 3, WS FIFO 64 / 1024, SPM_RD_LAT 1 / 4, RD_OUTSTANDING
                                1 / 2, MVM latencies 1 / 8 ...): a trace subset + m3_random
  test_core_m3_directed[<cfg>]  directed tests (below) on reduced / U200 configurations;
                                u200_rst3: the U200 v1 core at the bitstream parameters
                                (docs/rtl-m3.md §14.2, RST_PIPE 3) on rtl/filelist_xilinx.f
cocotb tests (new here):
  m3_random      ProgGen(m3=True) program() and mvm_program() on the RTL and the simulator in
                 lockstep: final CSRs, PC, R1..R15, retired, read beats per channel and per
                 source (DMA == LD beats, WS == rows_c x ROW_BEATS), DRAM, PERF_MVM_CYCLES > 0
                 iff an MVM executed; the whole SPM compared (ST dumps) after the programs (and
                 every TSR_CORE_SPM_EVERY programs); executed_overlaps() must stay empty
  m3_perf        PERF_MVM_CYCLES == the cycles the MVM unit holds instructions (counted at
                 mvm_cmd handshake .. mvm_done), N = 0 MVM: >= 1 cycle and no traffic, a faulting
                 MVM: 0; per-channel weight beats at the port == rows_c x ROW_BEATS (source WS);
                 PERF_CH_RD_BEATS[c] == LD + WS beats (+ fetch on channel 0); wait <= cycles
  m3_throughput  ideal memory on every channel at once: steady-state cycles per weight beat
                 (two-point: R and 3R rows per channel) for Q8_0 / Q4_K at T = 1 and T = T_MAX
                 with ROW_BEATS >= RB_MIN, asserted <= TSR_CORE_M3_RATE (default 1.02); the M3 gate
                 shapes (Qwen2.5-0.5B Q8_0: K 896 x N 896 / 4864 / 151936, K 4864 x N 896) timed
                 (logged; the 151936 output head on 4-channel T_MAX 1 builds unless
                 TSR_CORE_M3_HEAD=0); at least 4 shapes with ROW_BEATS >= RB_MIN must fit
  m3_mvm_st      back-to-back dependencies through the SPM: MVM -> ST of y, MVM -> MVM ACC into the
                 same y, MVM whose x is the previous MVM's y (raw QA), LD -> MVM, MVM -> QACT of y,
                 under every memory regime, against the simulator
  m3_abort       ABORT at random cycles of MVM programs (during the weight stream, the broadcast,
                 the gather): the running MVM completes, ERR_PC = first instruction not executed,
                 effects == the simulator's run of the executed prefix (DRAM, registers, SPM)
  m3_soft_reset  SOFT_RESET while BUSY inside MVM programs (P3: acts as ABORT, then the reset), the
                 next MVM program matches a fresh simulator
  m3_hw_reset    rst_n at random points of MVM programs (also mid weight stream with reads in
                 flight; the memory side is reset with the core), pulses of >= RST_PIPE + 1
                 cycles: outputs quiet from RST_PIPE cycles after rst_n falls (docs/rtl-m2.md
                 P10), CSRs == a fresh simulator, the next MVM program matches
Environment: TSR_CORE_PASSES, TSR_CORE_RANDOM, TSR_CORE_SPM_EVERY, TSR_CORE_TESTS,
TSR_CORE_TRACE_FILTER (as in test_core.py), TSR_CORE_M3_TRACES (colon-separated trace roots),
TSR_CORE_M3_RATE, TSR_CORE_M3_HEAD, TSR_CORE_M3_INTERRUPTS (runs per abort / reset test),
TSR_CORE_FILELIST (RTL filelist instead of rtl/filelist.f, e.g. a mutated scratch copy).
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
from cocotb.triggers import ClockCycles, FallingEdge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import isa, simlib  # noqa: E402
from common.proggen import ProgGen, executed_count, executed_overlaps  # noqa: E402
from common.runner import REPO, rtl_sources  # noqa: E402
from common.runner import run as _run  # noqa: E402
from common.tsr_ref import ref  # noqa: E402

import test_core as tc  # noqa: E402  (harness: CoreH, regimes, helpers)
from test_core import (DONE, DONEERR, M32, REGIME_ORDER, REGIMES, CoreH, core_env, core_params,  # noqa: E402
                       executed_mvms, expect_final, io_beat, prog_bytes, trace_groups)
# M2 cocotb tests reused on the HAS_MVM = 1 core (discovered by cocotb in this module's namespace).
from test_core import (core_abort, core_back_to_back, core_csr_consts, core_csr_map,  # noqa: E402,F401
                       core_hw_reset, core_soft_reset, core_status_w1c, core_traces)

M3_ROOTS = [Path(p) for p in os.environ.get(
    "TSR_CORE_M3_TRACES", ":".join(str(REPO / "traces" / s) for s in ("m3", "gen", "m2"))).split(":") if p]
M3_GROUPS = trace_groups(M3_ROOTS, 1, prefix_root=True)


def run(*a, **kw):
    """common.runner.run with the RTL of $TSR_CORE_FILELIST when set (tb/core/mutate_core.py runs
    the suite on mutated scratch copies of rtl/)."""
    fl = os.environ.get("TSR_CORE_FILELIST")
    if fl:
        kw["sources"] = rtl_sources(fl)
    return _run(*a, **kw)


def _selected(tests):
    sel = os.environ.get("TSR_CORE_TESTS")
    if not sel:
        return list(tests)
    sel = sel.split(",")
    return [t for t in tests if t in sel]


def _cfg(**kw):
    c = dict(isa.U200_V1)
    c.update(kw)
    return c


# Implementation parameters of the U200 bitstream (docs/rtl-m3.md §14.2; with rtl/filelist_xilinx.f).
U200_BITSTREAM = dict(SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9, RST_PIPE=3)


def m3_params(cfg: dict, **impl) -> dict:
    p = core_params(cfg, has_mvm=1)
    p.update(impl)
    return p


def _gid(g):
    return "m3_" + g[0].replace("+", "_")


# =============================================================================================
# pytest functions
# =============================================================================================
M3_TRACE_TESTS = ["core_csr_consts", "core_traces", "m3_random"]


@pytest.mark.traces
@pytest.mark.parametrize("gid,cfg,paths", M3_GROUPS, ids=[g[0] for g in M3_GROUPS])
def test_core_m3_traces(gid, cfg, paths):
    tests = _selected(M3_TRACE_TESTS)
    if not tests:
        pytest.skip("no selected test")
    # random programs: cheap on few channels; the Python memory model makes 16 channels ~10x slower
    n_random = (300 if cfg["spm_bytes"] <= 1 << 20 else 150) if cfg["num_ch"] <= 4 else 60
    run("tsr_core", __name__, parameters=m3_params(cfg), name="core_m3_" + gid.replace("+", "_")[:40],
        testcase=tests,
        extra_env={"TSR_CORE": json.dumps(dict(cfg=cfg, name=gid, traces=paths, n_random=n_random, has_mvm=1,
                                               passes=4 if cfg["t_max"] <= 4 else 2))})


# =============================================================================================
# Helpers
# =============================================================================================
def mvm_insns(prog: bytes, n: int) -> list:
    return [pc for pc in range(n) if prog[32 * pc] == isa.OP_MVM]


def disasm_prog(prog: bytes, n: int = 64) -> str:
    return "\n".join("%4d  %s" % (i, isa.disasm(isa.insn_words(prog[32 * i:32 * i + 32])))
                     for i in range(min(len(prog) // 32, n)))


class M3H(CoreH):
    """CoreH for HAS_MVM = 1 builds: lockstep runs that also check the executed program for
    undefined overlaps and count MVMs / weight beats."""

    def __init__(self, dut, cfg, **kw):
        super().__init__(dut, cfg, **kw)
        assert self.has_mvm == 1, "test_core_m3 needs has_mvm = 1 in TSR_CORE"
        self.n_mvm = 0
        self.ws_beats = 0

    async def check_m3(self, prog, cmd, args=None, count=None, init=(), what="run", during=None,
                       expect=None, allow_overlap=False):
        """check() plus: the executed prefix has no undefined overlap (the generator promises
        none; spec 겹치는 피연산자), statistics."""
        res = self.sim_run(prog, cmd, args, init, count)
        n_exec = executed_count(prog, res.final, len(prog) // 32 if count is None else count)
        ov = executed_overlaps(prog, n_exec, args or [0] * 8, self.cfg)
        assert allow_overlap or not ov, "%s: undefined overlap in the executed program: %s\n%s" % (
            what, ov, disasm_prog(prog))
        out = await self.run(prog, cmd, args, count, init, during=during)
        errs = self.compare(out, res, what, prog)
        for k, v in (expect or {}).items():
            if res.final[k] != v:
                errs.append("test expectation %s = 0x%x, simulator gives 0x%x" % (k, v, res.final[k]))
        assert not errs, "%s: %s" % (what, "; ".join(errs))
        self.n_mvm += executed_mvms(res.trace)
        self.ws_beats += sum(b[isa.SRC_WS] for b in res.trace.expected_read_beats())
        return out, res


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def m3_random(dut):
    """ProgGen(m3=True) programs (program(): every instruction incl. MVM with errors; mvm_program():
    LD -> QACT -> MVM (-> ACC MVM) -> ST chains) on the RTL and the simulator in lockstep; the
    whole SPM compared at the end (ST dumps)."""
    env = core_env()
    h = await M3H.create(dut)
    h.new_sim()
    n = int(os.environ.get("TSR_CORE_RANDOM", env.get("n_random", 40)))
    spm_every = int(os.environ.get("TSR_CORE_SPM_EVERY", "0"))
    if n == 0:
        return
    t0 = time.time()
    await h.spm_fill()
    gen = ProgGen(random.Random(h.seed * 37 + 5), h.cfg, True, m3=True)
    data = gen.init_data()
    await h.check(prog_bytes([isa.enc_end()]), gen.cmd_base, init=data, what="load data")
    codes = collections.Counter()
    kinds = collections.Counter()
    retired = cycles = mvm_cycles = 0
    rpoll = random.Random(h.seed * 13 + 2)
    for i in range(n):
        if i % 10 == 0:
            regime = REGIME_ORDER[(i // 10) % len(REGIME_ORDER)]
            h.retime_mem(**REGIMES[regime])
        kind = "mvm_program" if i % 2 else "program"
        prog, cmd, count, args = gen.mvm_program() if i % 2 else gen.program()
        kinds[kind] += 1
        during = (lambda: tc.poll_csrs(h, rpoll)) if i % 4 == 3 else None
        try:
            out, res = await h.check_m3(prog, cmd, args, count=count, what="random %s %d" % (kind, i),
                                        during=during)
        except AssertionError:
            dut._log.error("random %s %d (CMD_ADDR 0x%x, count %d, args %s):\n%s", kind, i, cmd, count,
                           [hex(a) for a in args], disasm_prog(prog))
            raise
        codes[out.final["err_code"]] += 1
        retired += out.retired
        cycles += out.perf[isa.PERF_CYCLES]
        mvm_cycles += out.perf[isa.PERF_MVM_CYCLES]
        if spm_every and (i + 1) % spm_every == 0:
            await h.spm_compare("SPM after random program %d" % i)
    h.retime_mem()
    await h.spm_compare("SPM after random programs")
    dut._log.info("m3_random %s: %d programs %s, %d instructions retired, %d MVMs executed (%d weight "
                  "beats), %d cycles (%d MVM cycles), endings %s, %.0f s", env["name"], n, dict(kinds), retired,
                  h.n_mvm, h.ws_beats, cycles, mvm_cycles,
                  {isa.ERR_NAMES[k]: v for k, v in sorted(codes.items())}, time.time() - t0)
    assert h.n_mvm >= n // 2, "too few MVMs executed (%d in %d programs)" % (h.n_mvm, n)
    h.sim.close()
    h.mem.stop()


# =============================================================================================
# Directed-test helpers
# =============================================================================================
def sane_weights(rnd, n_beats: int) -> bytes:
    """Weight bytes whose every aligned halfword is a normal fp16 of 2^-14..2^-2 (sane d / dmin,
    ProgGen m3 weight window); quant bytes are then random too."""
    import struct
    hw = [((rnd.getrandbits(1) << 15) | (rnd.randint(1, 13) << 10) | rnd.getrandbits(10))
          for _ in range(32 * n_beats)]
    return struct.pack("<%dH" % len(hw), *hw)


def f32_acts(rnd, n_lines: int) -> bytes:
    import struct
    fw = [((rnd.getrandbits(1) << 31) | ((127 + rnd.randint(-6, 3)) << 23) | rnd.getrandbits(23))
          for _ in range(16 * n_lines)]
    return struct.pack("<%dI" % len(fw), *fw)


class Chain:
    """A self-contained MVM program on fixed SPM / DRAM layouts (writes every SPM line it reads):
    LD f32 activations -> QACT -> MVM -> [MVM ACC into the same y] -> ST y (one ST per token when
    dst_stride leaves gaps, which the MVM does not write) -> END, plus the DRAM init regions.
    Weights per channel at `wbeat` (rows_c x ROW_BEATS beats used)."""

    def __init__(self, h, rnd, fmt, K, N, T=1, acc2=True, dstride=0, st_ch=None, ld_ch=0):
        cfg = h.cfg
        self.fmt, self.K, self.N, self.T = fmt, K, N, T
        xl = K // 16
        ql = isa.qa_lines(fmt, K)
        yl = (N + 15) // 16
        ds = dstride or yl
        self.src, self.qa = 0, T * xl
        self.y = self.qa + T * ql
        self.y_ext = (T - 1) * ds + yl
        assert self.y + self.y_ext <= h.spm_lines, "chain does not fit the SPM"
        io = io_beat(h)
        R = -(-N // h.num_ch)
        rb = isa.row_beats(fmt, K)
        self.wbeat = io + T * xl + 64
        self.w2beat = self.wbeat + R * rb
        self.wr = self.w2beat + R * rb + 64
        assert self.wr + self.y_ext <= h.lim, "chain does not fit the DRAM limit"
        st_ch = h.num_ch - 1 if st_ch is None else st_ch
        insns = [isa.enc_ld(ld_ch, self.src, io, T * xl),
                 isa.enc_qact(fmt, self.qa, self.src, K, T),
                 isa.enc_mvm(fmt, self.y, self.qa, self.wbeat, N, K, T, dstride)]
        if acc2:
            insns.append(isa.enc_mvm(fmt, self.y, self.qa, self.w2beat, N, K, T, dstride, acc=1))
        if T > 1 and ds > yl:
            insns += [isa.enc_st(st_ch, self.y + t * ds, self.wr + t * ds, yl) for t in range(T)]
        else:
            insns.append(isa.enc_st(st_ch, self.y, self.wr, self.y_ext))
        insns.append(isa.enc_end())
        self.insns = insns
        self.prog = prog_bytes(insns)
        self.init = [(ld_ch, io * 64, f32_acts(rnd, T * xl))]
        for c in range(h.num_ch):
            self.init.append((c, self.wbeat * 64, sane_weights(rnd, 2 * R * rb)))
        self.ws_beats = R * rb


class MvmMon:
    """Monitor of the MVM dispatch port inside tsr_core (sampled at falling edges): per dispatched
    MVM the cycles it is held (accept + 1 .. done, = PERF_MVM_CYCLES), its SPM client-2 read
    requests / writes accepted, its WS beats requested per channel; traffic outside any MVM."""

    def __init__(self, dut, num_ch):
        self.dut, self.nch = dut, num_ch
        self.recs = []
        self.outside = collections.Counter()
        self._task = cocotb.start_soon(self._run())

    def stop(self):
        self._task.cancel()

    async def _run(self):
        d = self.dut
        cur = None
        nsrc, lw = 3, isa.LEN_W            # TSR_NSRC read sources per arbiter
        while True:
            await FallingEdge(d.clk)
            if not int(d.rst_n.value):
                cur = None
                continue
            take = int(d.mvm_cmd_valid.value) & int(d.mvm_cmd_ready.value)
            done = int(d.mvm_done.value)
            rv = (int(d.spm_rd_req_valid.value) >> 2) & 1
            rr = (int(d.spm_rd_req_ready.value) >> 2) & 1
            wv = (int(d.spm_wr_valid.value) >> 2) & 1
            wr = (int(d.spm_wr_ready.value) >> 2) & 1
            qv = int(d.s_rd_req_valid.value)
            ws = []
            if qv:
                qr = int(d.s_rd_req_ready.value)
                ln = int(d.s_rd_req_len.value)
                for c in range(self.nch):
                    b = c * nsrc + isa.SRC_WS
                    if (qv >> b) & (qr >> b) & 1:
                        ws.append((c, (ln >> (b * lw)) & ((1 << lw) - 1)))
                    if (qv >> b) & 1 and cur is None:
                        self.outside["ws_offer"] += 1
            if cur is not None:
                cur["cycles"] += 1
                cur["spm_rd"] += rv & rr
                cur["spm_wr"] += wv & wr
                for c, ln in ws:
                    cur["ws"][c] += ln
                if done:
                    self.recs.append(cur)
                    cur = None
            else:
                self.outside["spm"] += rv | wv
                self.outside["done"] += done
            if take:
                cur = dict(cmd=isa.MVM_CMD_T.unpack(int(d.mvm_cmd.value)), cycles=0, spm_rd=0, spm_wr=0,
                           ws=[0] * self.nch)


def expect_ws(cmd: dict, num_ch: int) -> list:
    N, R = cmd["n"], cmd["rows_per_ch"]
    return [max(0, min(R, N - c * R)) * cmd["row_beats"] for c in range(num_ch)]


def src_beats_of(prog: bytes, n_exec: int, num_ch: int) -> list:
    """Per channel {DMA: LD beats, WS: rows_c x ROW_BEATS} of the first n_exec instructions."""
    out = [{isa.SRC_DMA: 0, isa.SRC_WS: 0} for _ in range(num_ch)]
    for pc in range(n_exec):
        w = isa.insn_words(prog[32 * pc:32 * pc + 32])
        op = w[0] & 0xFF
        if op == isa.OP_LD:
            out[(w[0] >> 8) & 0xF][isa.SRC_DMA] += w[3]
        elif op == isa.OP_MVM:
            fmt, N, K = (w[0] >> 8) & 0xF, w[4], w[5] & 0xFFFF
            rb = isa.row_beats(fmt, K)
            R = -(-N // num_ch)
            for c in range(num_ch):
                out[c][isa.SRC_WS] += max(0, min(R, N - c * R)) * rb
    return out


def check_src_beats(h, prog, n_exec, what):
    want = src_beats_of(prog, n_exec, h.num_ch)
    got = h.mem.src_beats
    bad = ["ch%d %s %d (want %d)" % (c, name, got[c][s], want[c][s])
           for c in range(h.num_ch) for s, name in ((isa.SRC_DMA, "DMA"), (isa.SRC_WS, "WS"))
           if got[c][s] != want[c][s]]
    assert not bad, "%s: read beats per source: %s" % (what, ", ".join(bad))


async def wait_mvm_accept(dut, nth: int, limit: int = 2_000_000) -> bool:
    """Wait (falling edges) until the nth MVM dispatch (1-based) of the current run is accepted;
    False when the run ends first."""
    seen = 0
    for _ in range(limit):
        await FallingEdge(dut.clk)
        if int(dut.mvm_cmd_valid.value) & int(dut.mvm_cmd_ready.value):
            seen += 1
            if seen >= nth:
                return True
        if int(dut.done.value):
            return False
    return False


def chain_shapes(h, rnd, n: int, long=False):
    """Random (fmt, K, N, T, dstride) chain shapes that fit this configuration."""
    out = []
    kmax = h.cfg["k_max"]
    for _ in range(n * 4):
        if len(out) >= n:
            break
        fmt = rnd.randrange(2)
        bs = 256 if fmt else 32
        ks = [k for k in ([256, 768, 1024, 2048, 2304, 4096] if fmt else [32, 96, 480, 896, 1056, 2048, 4096])
              if k <= kmax]
        K = rnd.choice(ks)
        T = rnd.randint(1, h.cfg["t_max"])
        rb = isa.row_beats(fmt, K)
        nmax = max(1, (6000 if long else 1500) // rb) * h.num_ch
        N = rnd.choice([1, h.num_ch + 1, 16 * h.num_ch + 5, rnd.randint(1, nmax), nmax])
        yl = (N + 15) // 16
        ds = rnd.choice([0, yl + rnd.randint(0, 3)]) if T > 1 else 0
        xl, ql = K // 16, isa.qa_lines(fmt, K)
        if T * xl + T * ql + (T - 1) * (ds or yl) + yl > h.spm_lines or K % bs:
            continue
        R = -(-N // h.num_ch)
        if io_beat(h) + T * xl + 2 * R * rb + 200 + (T - 1) * (ds or yl) + yl > h.lim:
            continue
        out.append((fmt, K, N, T, ds))
    return out


# =============================================================================================
# Directed cocotb tests
# =============================================================================================
@cocotb.test()
async def m3_perf(dut):
    """PERF_MVM_CYCLES == the cycles the MVM unit holds instructions; per MVM: weight beats per
    channel == rows_c x ROW_BEATS, SPM reads == T x QA_LINES (+ one per gather job with ACC),
    SPM writes == gather jobs, nothing outside a held MVM; N = 0: >= 1 cycle, no traffic;
    faulting MVM: 0 cycles; PERF_CH_RD_BEATS == port beats (LD + WS, + fetch on channel 0)."""
    from common import mvm_model
    h = await M3H.create(dut)
    h.new_sim()
    rnd = random.Random(h.seed * 41 + 3)
    nch = h.num_ch
    mon = MvmMon(dut, nch)
    kmax = h.cfg["k_max"]
    runs = [Chain(h, rnd, 0, min(1024, kmax), 16 * nch + 5, acc2=True),
            Chain(h, rnd, 1, min(2048, kmax // 256 * 256), nch + 1, acc2=True),
            Chain(h, rnd, 0, 32, 1, T=h.cfg["t_max"], acc2=True),
            Chain(h, rnd, 1, 256, 3 * nch - 1, T=h.cfg["t_max"], acc2=False,
                  dstride=((3 * nch + 14) // 16) + 2)]
    rows = []
    for k, ch in enumerate(runs):
        h.retime_mem(**REGIMES[REGIME_ORDER[k % 4]])
        n0 = len(mon.recs)
        out, res = await h.check_m3(ch.prog, 64 + 32 * k, init=ch.init, what="perf chain %d" % k,
                                    expect=dict(status=DONE))
        recs = mon.recs[n0:]
        d = out.perf
        assert len(recs) == executed_mvms(res.trace) == sum(1 for w in ch.insns if w[0] & 0xFF == isa.OP_MVM)
        assert d[isa.PERF_MVM_CYCLES] == sum(r["cycles"] for r in recs), (d[isa.PERF_MVM_CYCLES], recs)
        for r in recs:
            c = r["cmd"]
            assert r["ws"] == expect_ws(c, nch), ("WS beats per channel", r["ws"], expect_ws(c, nch))
            stride = c["dst_stride"]
            jobs = len(mvm_model.gather_jobs(c["n"], nch, c["t"], c["dst_line"], stride))
            want_rd = c["t"] * c["qa_lines"] + (jobs if c["acc"] else 0)
            assert r["spm_rd"] == want_rd and r["spm_wr"] == jobs, \
                "SPM reads %d (want %d) writes %d (want %d jobs)" % (r["spm_rd"], want_rd, r["spm_wr"], jobs)
            rows.append("fmt %d K %5d N %5d T %d acc %d: %6d cycles, %5d WS beats/ch0, %4d SPM rd, %4d wr"
                        % (c["fmt"], c["k"], c["n"], c["t"], c["acc"], r["cycles"], r["ws"][0], r["spm_rd"],
                           r["spm_wr"]))
        fetch = (len(ch.insns) + 1) // 2
        sb = h.mem.src_beats
        assert sb[0][isa.SRC_FETCH] == fetch, (sb[0], fetch)
        for c in range(nch):
            want = sum(sb[c])
            assert d[isa.perf_ch_rd_beats(c)] == want, (c, d[isa.perf_ch_rd_beats(c)], sb[c])
            assert d[isa.perf_ch_wait(nch, c)] <= d[isa.PERF_CYCLES]
        assert d[isa.PERF_MVM_CYCLES] < d[isa.PERF_CYCLES]
    # N = 0 (with ACC): held >= 1 cycle, no SPM access, no weight traffic; then a faulting MVM.
    n0 = len(mon.recs)
    p = prog_bytes([isa.enc_mvm(0, 10, 20, 30, 0, 32, 1, acc=1), isa.enc_mvm(1, 10, 20, 30, 0, 256, 1),
                    isa.enc_end()])
    out, _ = await h.check_m3(p, 300, what="N = 0", expect=dict(status=DONE))
    recs = mon.recs[n0:]
    assert len(recs) == 2 and all(1 <= r["cycles"] <= 3 and r["spm_rd"] == r["spm_wr"] == 0 and not any(r["ws"])
                                  for r in recs), recs
    assert out.perf[isa.PERF_MVM_CYCLES] == sum(r["cycles"] for r in recs)
    rows.append("N = 0: held %s cycles, no SPM / weight traffic" % [r["cycles"] for r in recs])
    n0 = len(mon.recs)
    p = prog_bytes([isa.enc_mvm(0, 10, 20, 30, 16, 33, 1), isa.enc_end()])       # K_RANGE
    out, _ = await h.check_m3(p, 310, what="faulting MVM", expect=dict(err_code=isa.ERR_K_RANGE))
    assert len(mon.recs) == n0 and out.perf[isa.PERF_MVM_CYCLES] == 0
    assert mon.outside["spm"] == 0 and mon.outside["done"] == 0 and mon.outside["ws_offer"] == 0, mon.outside
    mon.stop()
    dut._log.info("m3_perf %s:\n%s", core_env()["name"], "\n".join(rows))
    h.sim.close()
    h.mem.stop()


IDEAL = dict(rd_lat=(0, 1), wr_lat=(0, 1), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_wr_req_ready=1.0,
             p_wr_dat_ready=1.0, p_wr_rsp_valid=1.0, p_stall=0.0, wr_order="any")
RB_MIN = 9          # docs/rtl-m3.md §7.5 (achieved bound of tsr_mvm_back)


@cocotb.test()
async def m3_throughput(dut):
    """Ideal memory on every channel at once: steady-state cycles per weight beat (two-point,
    R and 3R rows per channel) for Q8_0 / Q4_K at T = 1 and T = T_MAX with ROW_BEATS >= RB_MIN
    (asserted <= TSR_CORE_M3_RATE), fixed overhead per MVM, and the M3 gate shapes timed."""
    h = await M3H.create(dut, mem_kw=IDEAL)
    nch, tmax, kmax = h.num_ch, h.cfg["t_max"], h.cfg["k_max"]
    rate_max = float(os.environ.get("TSR_CORE_M3_RATE", "1.02"))
    rows, worst = [], 0.0

    async def mvm_cycles(fmt, K, N, T, acc=0):
        ql = isa.qa_lines(fmt, K)
        dst = T * ql + 1
        prog = prog_bytes([isa.enc_mvm(fmt, dst, 0, 64, N, K, T, acc=acc), isa.enc_end()])
        out = await h.run(prog, 16)
        assert out.final["status"] == DONE, out.final
        R = -(-N // nch)
        want = [max(0, min(R, N - c * R)) * isa.row_beats(fmt, K) for c in range(nch)]
        got = [b[isa.SRC_WS] for b in h.mem.src_beats]
        assert got == want, (got, want)
        return out.perf[isa.PERF_MVM_CYCLES], max(want)

    shapes = [(0, 896), (0, 4096), (0, 4864), (1, 1024), (1, 2304), (1, 4096)]
    measured = 0
    for T in sorted({1, tmax}):
        for fmt, K in shapes:
            rb = isa.row_beats(fmt, K)
            r1 = max(8, -(-800 // rb))
            y_ext = (T - 1) * ((nch * 3 * r1 + 15) // 16) + (nch * 3 * r1 + 15) // 16
            if K > kmax or T * isa.qa_lines(fmt, K) + 1 + y_ext > h.spm_lines:
                continue
            c1, b1 = await mvm_cycles(fmt, K, nch * r1, T)
            c2, b2 = await mvm_cycles(fmt, K, nch * 3 * r1, T)
            rate = (c2 - c1) / (b2 - b1)
            ovh = c1 - rate * b1
            rows.append("%s K %5d T %d (%2d-beat rows): %5d / %5d cycles for %5d / %5d beats/ch -> %.4f cycles/beat, "
                        "fixed %.1f cycles (T x QA_LINES = %d)" % ("Q4_K" if fmt else "Q8_0", K, T, rb, c1, c2, b1, b2,
                                                                  rate, ovh, T * isa.qa_lines(fmt, K)))
            if rb >= RB_MIN:
                worst = max(worst, rate)
                measured += 1
                assert rate <= rate_max, rows[-1]
    assert measured >= 4, "only %d shapes with ROW_BEATS >= RB_MIN fit this configuration" % measured
    # M3 gate (Qwen2.5-0.5B Q8_0, 1 token): K 896 x N 896 / 4864 / 151936 (output head), K 4864 x N 896.
    gate = [(0, 896, 896), (0, 896, 4864), (0, 4864, 896)]
    if os.environ.get("TSR_CORE_M3_HEAD", "1") != "0" and nch == 4 and kmax >= 4864 and tmax == 1:
        gate.append((0, 896, 151936))
    for fmt, K, N in gate:
        if K > kmax or (N + 15) // 16 + isa.qa_lines(fmt, K) + 2 > h.spm_lines:
            continue
        t0 = time.time()
        c, b = await mvm_cycles(fmt, K, N, 1)
        model = 64 + isa.qa_lines(fmt, K) + b
        rows.append("gate Q8_0 K %4d N %6d: %7d cycles for %6d beats/ch -> %.4f cycles/beat (overhead %d; simulator "
                    "cycle model %d) = %.1f us at 300 MHz [%.0f s]" % (K, N, c, b, c / b, c - b, model, c / 300.0,
                                                                     time.time() - t0))
    dut._log.info("m3_throughput %s (ideal memory, every channel at once; worst steady-state rate for "
                  "ROW_BEATS >= %d: %.4f):\n%s", core_env()["name"], RB_MIN, worst, "\n".join(rows))
    h.mem.stop()


@cocotb.test()
async def m3_mvm_st(dut):
    """Back-to-back dependencies through the SPM against the simulator, rotating memory regimes:
    chains LD -> QACT -> MVM -> MVM ACC -> ST; MVM whose x is the previous MVM's y (raw QA, the
    read must see the gather's writes); LD of raw bytes -> MVM on them; MVM -> QACT of y -> ST;
    an MVM pair with the second one's dst = the first one's dst (plain overwrite) -> ST."""
    h = await M3H.create(dut)
    h.new_sim()
    await h.spm_fill()                 # SPM == the simulator's 0xA5 fill (whole-SPM compare at the end)
    rnd = random.Random(h.seed * 43 + 7)
    nch, kmax, tmax = h.num_ch, h.cfg["k_max"], h.cfg["t_max"]
    io = io_beat(h)
    n_prog = 0
    shapes = chain_shapes(h, rnd, 8)
    for k, (fmt, K, N, T, ds) in enumerate(shapes):
        h.retime_mem(**REGIMES[REGIME_ORDER[k % 4]])
        ch = Chain(h, rnd, fmt, K, N, T=T, acc2=bool(k % 2 == 0), dstride=ds, st_ch=k % nch, ld_ch=(k + 1) % nch)
        await h.check_m3(ch.prog, 64 + 32 * (k % 8), init=ch.init, what="chain %d %s" % (k, (fmt, K, N, T, ds)),
                         expect=dict(status=DONE))
        n_prog += 1
    # MVM2 reads MVM1's y as raw QA (x2 = y1); then ST y2. Q8_0 K = 32 needs 2 QA lines per token.
    for k in range(4):
        h.retime_mem(**REGIMES[REGIME_ORDER[(k + 1) % 4]])
        fmt1 = k % 2
        K1 = 256 if fmt1 else 64
        N1 = rnd.choice([40, 16 * nch + 7, 64 * nch])
        ch = Chain(h, rnd, fmt1, K1, N1, T=1, acc2=False)
        y1_lines = (N1 + 15) // 16
        fmt2 = rnd.randrange(2)
        K2 = 256 if fmt2 else 32 * max(1, min(kmax // 32, (y1_lines * 64) // 36 // 32 * 32 // 32))
        while isa.qa_lines(fmt2, K2) > y1_lines and K2 > (256 if fmt2 else 32):
            K2 -= 256 if fmt2 else 32
        if isa.qa_lines(fmt2, K2) > y1_lines:
            continue
        N2 = rnd.choice([1, 17, 3 * nch + 1])
        y2 = ch.y + y1_lines + 2
        if y2 + (N2 + 15) // 16 > h.spm_lines:
            continue
        w2 = ch.wr + 512
        R2 = -(-N2 // nch)
        insns = ch.insns[:3] + [isa.enc_mvm(fmt2, y2, ch.y, w2, N2, K2, 1, acc=k // 2),
                                isa.enc_st(nch - 1, y2, ch.wr, (N2 + 15) // 16), isa.enc_end()]
        init = ch.init + [(c, w2 * 64, sane_weights(rnd, R2 * isa.row_beats(fmt2, K2))) for c in range(nch)]
        if k // 2:   # ACC on y2: define y2 first through an LD of zeros
            insns = [isa.enc_ld(0, y2, ch.wr + 4096, (N2 + 15) // 16)] + insns
            init.append((0, (ch.wr + 4096) * 64, bytes(64 * ((N2 + 15) // 16))))
        await h.check_m3(prog_bytes(insns), 32, init=init, what="x = previous y #%d" % k, expect=dict(status=DONE))
        n_prog += 1
    # LD raw bytes -> MVM directly on them (raw QA incl. -128 quants, any D / BS) -> ST.
    for k in range(4):
        h.retime_mem(**REGIMES[REGIME_ORDER[(k + 2) % 4]])
        fmt = k % 2
        K = rnd.choice([256, 512, 768]) if fmt else rnd.choice([32, 160, 480, 1024])
        K = min(K, kmax)
        T = rnd.randint(1, tmax)
        ql = isa.qa_lines(fmt, K)
        N = rnd.choice([5, 16 * nch + 3, 100])
        yl = (N + 15) // 16
        x, y = 3, 3 + T * ql + 1
        R = -(-N // nch)
        wb = io + 2048
        insns = [isa.enc_ld(k % nch, x, io, T * ql), isa.enc_mvm(fmt, y, x, wb, N, K, T),
                 isa.enc_st(0, y, io + 1024, T * yl), isa.enc_end()]
        init = [(k % nch, io * 64, rnd.randbytes(64 * T * ql))]
        init += [(c, wb * 64, sane_weights(rnd, R * isa.row_beats(fmt, K))) for c in range(nch)]
        await h.check_m3(prog_bytes(insns), 40, init=init, what="LD -> MVM raw #%d" % k, expect=dict(status=DONE))
        n_prog += 1
    # MVM -> QACT of y (A32 over 16 * y_lines f32: needs an even y_lines) -> ST of the QA image.
    for k in range(2):
        h.retime_mem(**REGIMES[REGIME_ORDER[k % 4]])
        ch = Chain(h, rnd, k, 1024 if kmax >= 1024 else 256, 32 * nch, T=1, acc2=bool(k))
        yl = (32 * nch) // 16
        Kq = 16 * yl
        qd = ch.y + yl + 1
        qal = isa.qa_lines(0, Kq)
        insns = ch.insns[:-2] + [isa.enc_qact(0, qd, ch.y, Kq, 1), isa.enc_st(1 % nch, qd, ch.wr, qal),
                                 isa.enc_end()]
        await h.check_m3(prog_bytes(insns), 48, init=ch.init, what="MVM -> QACT of y #%d" % k,
                         expect=dict(status=DONE))
        n_prog += 1
    h.retime_mem()
    if h.spm_lines <= 4096:
        await h.spm_compare("SPM after the dependency programs")
    dut._log.info("m3_mvm_st %s: %d programs, %d MVMs, %d weight beats", core_env()["name"], n_prog, h.n_mvm,
                  h.ws_beats)
    h.sim.close()
    h.mem.stop()


async def _interrupt_runs(h, dut, rnd, kind: str, n_runs: int):
    """ABORT / SOFT_RESET / hardware reset at random points of MVM chain programs (half of them
    timed inside a dispatched MVM). Returns counters."""
    stats = collections.Counter()
    shapes = chain_shapes(h, rnd, max(4, n_runs // 3), long=True)
    for i in range(n_runs):
        fmt, K, N, T, ds = shapes[i % len(shapes)]
        ch = Chain(h, rnd, fmt, K, N, T=T, acc2=True, dstride=ds, st_ch=i % h.num_ch)
        regime = REGIME_ORDER[i % 3]                    # rand / fast / slow
        h.retime_mem(**REGIMES[regime])
        # abort: one simulator follows every executed prefix (SPM compared at the end); resets:
        # a fresh simulator after each reset (chains write every SPM line they read)
        p0 = await h.launch(ch.prog, 128 + 16 * (i % 4), init=ch.init)
        if i % 2:
            nth = rnd.randint(1, 2)
            got = await wait_mvm_accept(dut, nth)
            est = ch.ws_beats * (3 if regime == "slow" else 1.3) + 200
            await ClockCycles(dut.clk, rnd.randint(0, int(est)))
            stats["targeted"] += got
        else:
            await ClockCycles(dut.clk, rnd.choice([rnd.randint(0, 100), rnd.randint(0, 2000),
                                                  rnd.randint(0, 3 * ch.ws_beats + 2000)]))
        busy = h.busy()
        in_mvm = int(dut.perf_mvm_busy.value)
        stats["busy"] += busy
        stats["in_mvm"] += in_mvm
        if kind == "abort":
            await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)
            out = await h.finish(p0)
            f = out.final
            if f["err_code"] == isa.ERR_ABORTED:
                k = f["err_pc"]
                stats["aborted"] += 1
                assert f["status"] == DONEERR and f["err_info"] == 0 and out.retired == k, (f, out.retired)
                assert f["pc"] == (k - 1 if k > 0 else 0), f
                sp = prog_bytes(ch.insns[:k] + [isa.enc_end()])
                res = h.sim_run(sp, 128 + 16 * (i % 4), init=ch.init)
                assert res.final["err_code"] == 0
                check_src_beats(h, ch.prog, k, "abort #%d" % i)
                n_mvm = executed_mvms(res.trace)
                res.trace.cmd = ch.prog          # the RTL may prefetch the whole command buffer
                errs = h.compare(out, res, "abort #%d" % i, ch.prog, final=False, sources=False, n_mvm=n_mvm)
                assert not errs, "abort #%d (ERR_PC %d): %s" % (i, k, "; ".join(errs))
                stats["aborted_after_mvm"] += sum(1 for w in ch.insns[:k] if w[0] & 0xFF == isa.OP_MVM) > 0
            else:
                res = h.sim_run(ch.prog, 128 + 16 * (i % 4), init=ch.init)
                errs = h.compare(out, res, "abort-late #%d" % i, ch.prog)
                assert not errs, "abort-late #%d: %s" % (i, "; ".join(errs))
                stats["finished"] += 1
            # the next run starts clean (abort flag cleared by START)
            await h.check(prog_bytes([isa.enc_li(1, i), isa.enc_end()]), 120, what="after abort #%d" % i,
                          expect=dict(status=DONE))
            continue
        if kind == "soft_reset":
            await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
            for _ in range(4_000_000):
                await FallingEdge(dut.clk)
                assert not int(dut.done.value), "done rose on a run ended by SOFT_RESET (#%d)" % i
                if not h.busy():
                    break
            assert not h.busy(), "run did not end after SOFT_RESET"
            await ClockCycles(dut.clk, 10)
            h.mem.check(idle=True)
            st = await h.final()
            assert st == dict(status=0, err_code=0, err_pc=0, err_info=0, pc=0), (i, st)
            assert all(v == 0 for v in await h.perf()), "PERF not cleared"
            assert h.regs() == [0] * 16
            sim = h.new_sim()
            sim.csr_write(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
        else:   # hardware reset (memory side together with the core; RST_PIPE contract, P10)
            await h.hw_reset(rnd, "#%d" % i, minimal=(i % 3 == 2))
            h.mem.check(idle=True)
            sim = h.new_sim()
            bad = []
            for off in range(0, 0x200, 4):
                got, want = await h.rd(off), sim.csr_read(off)
                if got != want:
                    bad.append("0x%03x = 0x%x (fresh simulator 0x%x)" % (off, got, want))
            assert not bad, "after reset #%d: %s" % (i, bad[:6])
            assert h.regs() == [0] * 16 and int(dut.done.value) == 0
        # The next MVM program (self-contained) matches a fresh simulator.
        h.mem.clear_tracking()
        h.retime_mem()
        fmt2, K2, N2, T2, ds2 = shapes[(i + 1) % len(shapes)]
        ch2 = Chain(h, rnd, fmt2, K2, N2, T=T2, acc2=True, dstride=ds2)
        await h.check_m3(ch2.prog, 256, init=ch2.init, what="MVM program after %s #%d" % (kind, i),
                         expect=dict(status=DONE))
        stats["after_ok"] += 1
    return stats


@cocotb.test()
async def m3_abort(dut):
    """ABORT at random cycles of MVM chain programs (half timed inside a dispatched MVM): the
    running MVM completes, ERR_PC = first instruction not executed, effects and read beats per
    source == the executed prefix; the SPM equals the simulator's afterwards."""
    h = await M3H.create(dut)
    h.new_sim()
    rnd = random.Random(h.seed * 47 + 9)
    n = int(os.environ.get("TSR_CORE_M3_INTERRUPTS", "24"))
    await h.spm_fill()
    st = await _interrupt_runs(h, dut, rnd, "abort", n)
    h.retime_mem()
    if h.spm_lines <= 4096:
        await h.spm_compare("SPM after aborts")
    dut._log.info("m3_abort %s: %s", core_env()["name"], dict(st))
    assert st["aborted"] >= n // 3 and st["in_mvm"] >= n // 4, st
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def m3_soft_reset(dut):
    """SOFT_RESET while BUSY at random points of MVM chain programs (half inside a dispatched MVM):
    no DONE, every CSR / PERF / register cleared, the memory port quiet; the next MVM program
    matches a fresh simulator."""
    h = await M3H.create(dut)
    rnd = random.Random(h.seed * 53 + 1)
    n = int(os.environ.get("TSR_CORE_M3_INTERRUPTS", "24"))
    st = await _interrupt_runs(h, dut, rnd, "soft_reset", n)
    dut._log.info("m3_soft_reset %s: %s", core_env()["name"], dict(st))
    assert st["busy"] >= n // 2 and st["in_mvm"] >= n // 4 and st["after_ok"] == n, st
    h.sim.close()
    h.mem.stop()


@cocotb.test()
async def m3_hw_reset(dut):
    """Hardware reset (memory side together) at random points of MVM chain programs (half inside
    a dispatched MVM, weight reads in flight), pulses of >= RST_PIPE + 1 cycles (CoreH.hw_reset,
    docs/rtl-m2.md P10): outputs quiet from RST_PIPE cycles after rst_n falls, CSRs == a fresh
    simulator, the next MVM program matches a fresh simulator."""
    h = await M3H.create(dut)
    rnd = random.Random(h.seed * 59 + 2)
    n = int(os.environ.get("TSR_CORE_M3_INTERRUPTS", "24"))
    st = await _interrupt_runs(h, dut, rnd, "hw_reset", n)
    dut._log.info("m3_hw_reset %s: %s", core_env()["name"], dict(st))
    assert st["busy"] >= n // 2 and st["in_mvm"] >= n // 4 and st["after_ok"] == n, st
    h.sim.close()
    h.mem.stop()


# =============================================================================================
# pytest functions: implementation-parameter variants and directed configurations
# =============================================================================================
def _group(sub: str):
    g = [x for x in M3_GROUPS if sub in x[0]]
    return g[0] if g else None


# Implementation parameters (results must not change; docs/rtl-m3.md §1): each variant replays the
# traces of one M3 group (path filter `only`, if given) under `passes` rotating regimes, then
# m3_random. "u200_bitstream" is the expected U200 v1 build (SPM_RD_LAT 4 for URAM timing, pipe
# depth 2 for the SLR crossings, docs/rtl-m2.md §12, docs/rtl-m3.md §9.2).
M3_IMPL = {
    "u200_bitstream": dict(group="m3.tmax1", only=["/m3/tmax1/", "/gen/tmax1/"], passes=1, n_random=16,
                           params=dict(SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2)),
    "pipe0_fifo64_lat1": dict(group="small1ch", passes=2, n_random=40, rd_outstanding=1,
                              params=dict(MVM_BC_PIPE=0, MVM_RES_PIPE=0, MVM_WS_FIFO=64, SPM_RD_LAT=1,
                                          RD_OUTSTANDING=1, MVM_WS_RD_LAT=1, FETCH_BEATS=2)),
    "pipe3_fifo1024_lat4": dict(group="ch3", passes=2, n_random=40, rd_outstanding=2,
                                params=dict(MVM_BC_PIPE=3, MVM_RES_PIPE=3, MVM_WS_FIFO=1024, SPM_RD_LAT=4,
                                            RD_OUTSTANDING=2, MVM_WS_RD_LAT=4, FETCH_BEATS=64)),
    "fast_latencies": dict(group="ch3", passes=2, n_random=40,
                           params=dict(MVM_ACT_RD_LAT=1, MVM_MAC_LAT=1, MVM_MUL_LAT=1, MVM_ADD_LAT=1,
                                       MVM_WS_RD_LAT=1, SPM_RD_LAT=2)),
    "slow_latencies": dict(group="m3.ch16", passes=2, n_random=30,
                           params=dict(MVM_ACT_RD_LAT=4, MVM_MAC_LAT=8, MVM_MUL_LAT=8, MVM_ADD_LAT=8,
                                       MVM_WS_RD_LAT=4, MVM_WS_FIFO=128, SPM_RD_LAT=3)),
}


@pytest.mark.traces
@pytest.mark.parametrize("name", list(M3_IMPL))
def test_core_m3_impl(name):
    v = M3_IMPL[name]
    g = _group(v["group"])
    if g is None:
        pytest.skip("trace group %s not found" % v["group"])
    gid, cfg, paths = g
    if v.get("only"):
        paths = [p for p in paths if any(o in p for o in v["only"])]
    tests = _selected(M3_TRACE_TESTS)
    if not tests:
        pytest.skip("no selected test")
    env = dict(cfg=cfg, name=name, traces=paths, n_random=v["n_random"], passes=v["passes"], has_mvm=1)
    if v.get("rd_outstanding"):
        env["rd_outstanding"] = v["rd_outstanding"]
    run("tsr_core", __name__, parameters=m3_params(cfg, **v["params"]), name="core_m3i_" + name, testcase=tests,
        extra_env={"TSR_CORE": json.dumps(env)})


M3_DIRECTED = {
    # 2 channels, T_MAX 4, 64 KiB SPM, 1 GiB channels with a 2^22-beat limit, RD_OUTSTANDING 4.
    "c2": dict(cfg=_cfg(num_ch=2, t_max=4, k_max=4096, spm_bytes=65536, ch_bytes=1 << 30, addr_beat_w=22,
                        rd_outstanding=4),
               tests=["core_csr_consts", "core_csr_map", "core_status_w1c", "core_soft_reset", "core_hw_reset",
                      "core_abort", "core_back_to_back", "m3_perf", "m3_mvm_st", "m3_abort", "m3_soft_reset",
                      "m3_hw_reset", "m3_random"], n_random=100),
    # 3 channels (rows_c = 0 clusters for small N, shared y lines), T_MAX 3.
    "c3t3": dict(cfg=_cfg(num_ch=3, t_max=3, spm_bytes=262144), tests=["m3_perf", "m3_mvm_st", "m3_abort",
                                                                         "m3_hw_reset", "m3_throughput"],
                 n_random=0),
    # U200 v1: throughput incl. the M3 gate shapes, PERF, dependencies.
    "u200": dict(cfg=_cfg(), tests=["core_csr_consts", "m3_throughput", "m3_perf", "m3_mvm_st"], n_random=0),
    # U200 v1 with T_MAX 8: throughput at T = 1 and T = 8, PERF.
    "u200t8": dict(cfg=_cfg(t_max=8), tests=["m3_throughput", "m3_perf"], n_random=0),
    # U200 v1 (4 channels, T_MAX 1: the k4 T_MAX 1 kernel's core) at the U200 bitstream parameters
    # on the Xilinx technology wrappers: the RST_PIPE 3 reset contract (docs/rtl-m2.md P10) in the
    # hardware-reset tests, SOFT_RESET / ABORT, throughput and PERF at these latencies.
    "u200_rst3": dict(cfg=_cfg(), params=U200_BITSTREAM, filelist=REPO / "rtl" / "filelist_xilinx.f",
                      tests=["core_csr_consts", "core_status_w1c", "core_soft_reset", "core_hw_reset",
                             "m3_throughput", "m3_perf", "m3_mvm_st", "m3_abort", "m3_soft_reset",
                             "m3_hw_reset"], n_random=0),
}


@pytest.mark.parametrize("name", list(M3_DIRECTED))
def test_core_m3_directed(name):
    c = M3_DIRECTED[name]
    tests = _selected(c["tests"])
    if not tests:
        pytest.skip("no selected test")
    params = c.get("params", {})
    kw = dict(sources=rtl_sources(c["filelist"])) if c.get("filelist") else {}
    run("tsr_core", __name__, parameters=m3_params(c["cfg"], **params), name="core_m3d_" + name, testcase=tests,
        extra_env={"TSR_CORE": json.dumps(dict(cfg=c["cfg"], name=name, traces=[], n_random=c["n_random"],
                                               has_mvm=1, rst_pipe=params.get("RST_PIPE", 0)))}, **kw)


# =============================================================================================
# Fast C++ replay smoke (tb/fastreplay, harness agent): every trace of traces/m3 and traces/gen on
# the HAS_MVM = 1 core under all four memory regimes (the full campaigns over traces/model and
# traces/tmp are run by hand, docs/rtl-m3.md §13).
# =============================================================================================
@pytest.mark.traces
def test_core_m3_fastreplay():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fastreplay_run_m3", REPO / "tb" / "fastreplay" / "run.py")
    fr = importlib.util.module_from_spec(spec)
    sys.modules["fastreplay_run_m3"] = fr
    spec.loader.exec_module(fr)
    roots = [REPO / "traces" / "m3", REPO / "traces" / "gen"]
    procs = int(os.environ.get("TSR_CORE_FR_PROCS", "4"))
    fl = os.environ.get("TSR_CORE_FILELIST")
    rep = fr.replay(roots, regimes=["all"], has_mvm="1", procs=procs, filelist=fl, log=print, echo=None)
    for g in rep.groups:
        print(fr.format_group(g))
    n = sum(1 for _ in fr.find_traces(roots))
    assert rep.runs == 4 * n, (rep.runs, n)
    assert rep.ok, [f["errors"][:2] for g in rep.groups for f in g.failures[:5]]

"""Block (a) testbench: tsr_csr + tsr_cmdproc + tsr_fetch (tb/cmdproc/hdl/tsr_cmdproc_tb.sv).

Environment of the DUT:
  - fetch runs over the real channel-0 memory port against common.mem_model.MemModel (random
    latency, backpressure, stalls, protocol assertions incl. the DRAM limit);
  - the DMA / QACT / MVM units are Python models on the dispatch ports (UnitStub): they check the
    dispatch protocol (docs/rtl-m2.md §3.0) and every operand guarantee of §3.2-§3.4, and execute
    LD / ST / QACT / MVM functionally (FuncUnits: Python SPM, DRAM = the MemModel memory, numerics
    from libtsr_ref through common.tsr_ref), with random ready / done timing.
Oracle: the ISA simulator itself (simlib.Sim, ctypes). Harness.check() runs a program on both and
compares STATUS, ERR_CODE, ERR_PC, ERR_INFO, PC, R1..R15, the retired count, the whole SPM, and the
trace rules of replay.compare() (written DRAM beats and data, reads inside init / command beats).
After every run it also checks that the memory port is quiescent (DONE implies no transaction in
flight) and that fetch read only command beats of instructions below CMD_COUNT.

pytest functions (one Verilator build per configuration, all cocotb tests of that build):
  test_cmdproc[u200]   U200 v1, T_MAX 1, HAS_MVM 0 (MVM opcode is ILLEGAL_OP); traces tmax1, errors
  test_cmdproc[t8mvm]  U200 v1, T_MAX 8, HAS_MVM 1 (M3 MVM check path); traces tmax8, errors8
  test_cmdproc[small]  1 ch, 64 KiB SPM, 1 GiB ch, ADDR_BEAT_W 22, K_MAX 4096, T_MAX 2, 4-beat
                       fetch buffer; traces small1ch
  test_cmdproc[ch3]    3 ch, T_MAX 3, 256 KiB SPM, HAS_MVM 1 (ceil(N/3)), 8-beat fetch buffer;
                       traces ch3
  test_cmdproc[tiny]   2 ch of 32 beats (ADDR_BEAT_W 5), 64-line SPM, K_MAX 256, T_MAX 4, HAS_MVM 1,
                       2-beat fetch buffer: limits everywhere (simulator-compared tests only)
  test_cmdproc[big]    2 ch, DRAM limit 2^32 + 1000 beats (ADDR_BEAT_W 36), 64 KiB SPM, HAS_MVM 1,
                       64-beat fetch buffer: CMD_ADDR_HI != 0 in range, fetch across beat 2^32,
                       DRAM range errors only through long operands at the top of the 32-bit fields
cocotb tests: cp_csr_map (every CSR offset vs the simulator), cp_run_control, cp_soft_reset,
cp_abort, cp_regs_flow, cp_fetch, cp_ldst, cp_reserved, cp_errors (test_sim_basic.c cases),
cp_perf, cp_throughput (cycles per instruction), cp_random (ProgGen vs the simulator), cp_traces,
cp_stray_rsp (read responses with nothing outstanding, e.g. after a reset mid-run).
Expectations that depend on the DRAM limit are derived from it (configurations with a limit
above 2^32 beats, where the 32-bit operand fields cannot reach it, are legal).
Environment: TSR_CP_TESTS=a,b (subset), TSR_CP_RANDOM=N (random programs per build).
"""
from __future__ import annotations

import json
import os
import random
import sys
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
from common.runner import REPO, rtl_sources, run  # noqa: E402
from common.tbutil import seed as tb_seed  # noqa: E402
from common.tbutil import start_clock_and_reset, wait_level  # noqa: E402
from common.tsr_ref import ref  # noqa: E402

import simlib  # noqa: E402

HDL = HERE / "hdl" / "tsr_cmdproc_tb.sv"
# Only the files of this block (filelist order), so work in progress elsewhere cannot break it.
BLOCK_FILES = ("tsr_pkg.sv", "tsr_fetch.sv", "tsr_csr.sv", "tsr_cmdproc.sv")


def block_sources() -> list:
    return [p for p in rtl_sources() if p.name in BLOCK_FILES]
TRACES = REPO / "traces" / "m2"
M32 = 0xFFFFFFFF
DONE, DONEERR = isa.STATUS_DONE, isa.STATUS_DONE | isa.STATUS_ERR
IO_BEAT = isa.REGION_IO_OFF // 64
WOFF = isa.REGION_WEIGHT_OFF // 64


# =============================================================================================
# Configurations
# =============================================================================================
def _cfg(**kw):
    c = dict(isa.U200_V1)
    c.update(kw)
    return c


CONFIGS = {
    "u200": dict(cfg=_cfg(), has_mvm=0, fetch_beats=16, traces=["tmax1", "errors"], n_random=500),
    "t8mvm": dict(cfg=_cfg(t_max=8), has_mvm=1, fetch_beats=16, traces=["tmax8", "errors8"],
                  n_random=500),
    "small": dict(cfg=_cfg(num_ch=1, t_max=2, k_max=4096, spm_bytes=65536, ch_bytes=1 << 30,
                           addr_beat_w=22), has_mvm=0, fetch_beats=4, traces=["small1ch"],
                  n_random=400),
    "ch3": dict(cfg=_cfg(num_ch=3, t_max=3, spm_bytes=256 * 1024), has_mvm=1, fetch_beats=8,
                traces=["ch3"], n_random=400),
    # Limits everywhere: 32-beat channels (ADDR_BEAT_W 5 < one 4 KiB block), 64-line SPM.
    "tiny": dict(cfg=_cfg(num_ch=2, t_max=4, k_max=256, spm_bytes=4096, ch_bytes=4096, addr_beat_w=5),
                 has_mvm=1, fetch_beats=2, traces=[], n_random=1000,
                 tests=["cp_csr_map", "cp_fetch", "cp_random"]),
    # DRAM limit just above 2^32 beats (ADDR_BEAT_W 36, limit set by CH_BYTES).
    "big": dict(cfg=_cfg(num_ch=2, t_max=2, k_max=4096, spm_bytes=65536, ch_bytes=(1 << 38) + 64 * 1000,
                         addr_beat_w=36), has_mvm=1, fetch_beats=64, traces=[], n_random=300,
                tests=["cp_csr_map", "cp_regs_flow", "cp_fetch", "cp_ldst", "cp_errors", "cp_random",
                       "cp_stray_rsp"]),
}

COCOTB_TESTS = ["cp_csr_map", "cp_run_control", "cp_soft_reset", "cp_abort", "cp_regs_flow",
                "cp_fetch", "cp_ldst", "cp_reserved", "cp_errors", "cp_perf", "cp_throughput",
                "cp_random", "cp_traces", "cp_stray_rsp"]
# Tests that need simulator plusargs: each runs in its own simulator invocation (same build).
TEST_PLUSARGS = {"cp_stray_rsp": ["+tsr_allow_stray_rsp"]}


def _params(c: dict) -> dict:
    cfg = c["cfg"]
    return {"NUM_CH": cfg["num_ch"], "LANES": cfg["lanes"], "T_MAX": cfg["t_max"],
            "K_MAX": cfg["k_max"], "SPM_BYTES": cfg["spm_bytes"], "VL": cfg["vl"],
            "NSREG": cfg["nsreg"], "CH_BYTES": "64'd%d" % cfg["ch_bytes"],
            "ADDR_BEAT_W": cfg["addr_beat_w"], "HAS_MVM": c["has_mvm"],
            "FETCH_BEATS": c["fetch_beats"]}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_cmdproc(name):
    c = CONFIGS[name]
    tests = os.environ.get("TSR_CP_TESTS")
    tests = tests.split(",") if tests else COCOTB_TESTS
    tests = [t for t in tests if t in c.get("tests", COCOTB_TESTS)]
    if not tests:
        pytest.skip("no selected test applies to configuration %s" % name)
    groups = [([t for t in tests if t not in TEST_PLUSARGS], [])]
    groups += [([t], TEST_PLUSARGS[t]) for t in tests if t in TEST_PLUSARGS]
    for tcs, plusargs in groups:
        if tcs:
            run("tsr_cmdproc_tb", __name__, parameters=_params(c), sources=block_sources(),
                extra_sources=[HDL], name="cmdproc_" + name, testcase=tcs, plusargs=plusargs,
                extra_env={"TSR_CP": json.dumps(dict(c, name=name))})


def cp_env() -> dict:
    return json.loads(os.environ["TSR_CP"])


# =============================================================================================
# Unit models
# =============================================================================================
class UnitStub:
    """Python model of one dispatched unit (docs/rtl-m2.md §3.0). Drives <p>_cmd_ready (from its
    own state only), accepts a command at the handshake, executes it (`execute(cmd)`, a
    functional model), and pulses <p>_done once, `delay` cycles after the accepting edge (0 = the
    cycle right after it), or when released while `hold` is set. Asserts that cmd_valid and the
    payload stay stable until the handshake and that no command is offered while one is in flight."""

    def __init__(self, dut, prefix, sdef, rnd, execute=None, p_ready=0.6, delay=(0, 12)):
        self.clk = dut.clk
        self.v = getattr(dut, prefix + "_cmd_valid")
        self.r = getattr(dut, prefix + "_cmd_ready")
        self.c = getattr(dut, prefix + "_cmd")
        self.d = getattr(dut, prefix + "_done")
        self.name, self.sdef, self.rnd, self.execute = prefix, sdef, rnd, execute
        self.p_ready, self.delay = p_ready, delay
        self.cmds = []
        self.hold = False
        self.busy = False
        self.r.value = 0
        self.d.value = 0
        self.task = cocotb.start_soon(self._run())

    def _fail(self, msg):
        raise AssertionError("%s unit model: %s" % (self.name, msg))

    async def _run(self):
        ready, left, pend = 0, 0, None
        while True:
            await FallingEdge(self.clk)
            v = int(self.v.value)
            payload = int(self.c.value) if v else None
            await RisingEdge(self.clk)
            if pend is not None and (not v or payload != pend):
                self._fail("cmd_valid dropped or payload changed before the handshake")
            if v and self.busy:
                self._fail("command offered while one is in flight")
            take = v and ready
            pend = payload if (v and not ready) else None
            self.d.value = 0
            if take:
                cmd = self.sdef.unpack(payload)
                self.cmds.append(cmd)
                if self.execute is not None:
                    self.execute(cmd)
                ready = 0
                self.busy = True
                left = self.rnd.randint(*self.delay)
                if left == 0 and not self.hold:
                    self.d.value = 1
                    self.busy = False
            elif self.busy:
                left -= 1
                if left <= 0 and not self.hold:
                    self.d.value = 1
                    self.busy = False
            else:
                ready = int(self.rnd.random() < self.p_ready)
            self.r.value = ready


class FuncUnits:
    """Functional LD / ST / QACT / MVM on a Python SPM and the MemModel's DRAM, exactly as
    tsr_sim.c ex_ldst / ex_qact / ex_mvm, after asserting the operand guarantees of the dispatch
    structs (docs/rtl-m2.md §3.2-§3.4)."""

    def __init__(self, h):
        self.h = h
        self.spm = bytearray(b"\xA5" * h.cfg["spm_bytes"])
        self.reads = [set() for _ in range(h.num_ch)]

    def begin_run(self):
        self.reads = [set() for _ in range(self.h.num_ch)]

    def _line(self, i):
        return self.spm[64 * i:64 * i + 64]

    def _rd(self, ch, beat):
        c = self.h.mem.ch[ch]
        c.read.add(beat)
        self.reads[ch].add(beat)
        if beat not in c.written:
            c.read_unwritten.add(beat)
        return c.mem.get(beat, 0)

    def dma(self, c):
        h = self.h
        ch, s, b, n = c["ch"], c["spm_line"], c["dram_beat"], c["n_lines"]
        assert n >= 1 and ch < h.num_ch and s + n <= h.spm_lines and b + n <= h.lim, "illegal DMA command %s" % c
        chan = h.mem.ch[ch]
        if c["is_st"]:
            for i in range(n):
                v = int.from_bytes(self._line(s + i), "little")
                if v:
                    chan.mem[b + i] = v
                else:
                    chan.mem.pop(b + i, None)
                chan.written.add(b + i)
        else:
            for i in range(n):
                self.spm[64 * (s + i):64 * (s + i + 1)] = self._rd(ch, b + i).to_bytes(64, "little")

    def qact(self, c):
        h = self.h
        kind, dst, src, K, T = c["kind"], c["dst_line"], c["src_line"], c["k"], c["t"]
        st, ql = c["src_stride"], c["qa_lines"]
        bs = isa.qa_block_size(kind)
        assert 0 < K <= h.cfg["k_max"] and K % bs == 0 and 1 <= T <= h.cfg["t_max"], "QACT K/T %s" % c
        assert ql == isa.qa_lines(kind, K) and st != 0, "QACT derived fields %s" % c
        assert dst + T * ql <= h.spm_lines and src + (T - 1) * st + K // 16 <= h.spm_lines, "QACT range %s" % c
        for t in range(T):
            s0 = 64 * (src + t * st)
            img = ref().qact(kind, bytes(self.spm[s0:s0 + 4 * K]), K)
            d0 = 64 * (dst + t * ql)
            self.spm[d0:d0 + len(img)] = img

    def mvm(self, c):
        h, r = self.h, ref()
        fmt, acc, dst, x, woff = c["fmt"], c["acc"], c["dst_line"], c["x_line"], c["w_off"]
        N, K, T, ds, yl, R, rb, ql = (c["n"], c["k"], c["t"], c["dst_stride"], c["y_lines"],
                                      c["rows_per_ch"], c["row_beats"], c["qa_lines"])
        bs = 256 if fmt else 32
        assert 0 < K <= h.cfg["k_max"] and K % bs == 0 and 1 <= T <= h.cfg["t_max"], "MVM K/T %s" % c
        assert yl == (N + 15) // 16 and R == -(-N // h.num_ch), "MVM y_lines / rows_per_ch %s" % c
        assert rb == r.row_beats(fmt, K) and ql == isa.qa_lines(fmt, K), "MVM derived %s" % c
        assert ds != 0 or yl == 0, "MVM dst_stride not resolved %s" % c      # N = 0: 0 is resolved
        assert dst + (T - 1) * ds + yl <= h.spm_lines and x + T * ql <= h.spm_lines, "MVM SPM range %s" % c
        assert woff + R * rb <= h.lim, "MVM DRAM range %s" % c
        rows = []
        for n in range(N):
            ch = n // R
            off = woff + (n - ch * R) * rb
            rows.append(b"".join(self._rd(ch, off + i).to_bytes(64, "little") for i in range(rb)))
        ys = []
        for t in range(T):
            qa = bytes(self.spm[64 * (x + t * ql):64 * (x + t * ql + ql)])
            ys.append([r.dot_row(fmt, rows[n], K, qa) for n in range(N)])
        for t in range(T):
            base = 64 * (dst + t * ds)
            for n in range(N):
                v = ys[t][n]
                if acc:
                    v = r.fadd(int.from_bytes(self.spm[base + 4 * n:base + 4 * n + 4], "little"), v)
                self.spm[base + 4 * n:base + 4 * n + 4] = v.to_bytes(4, "little")
            self.spm[base + 4 * N:base + 64 * yl] = bytes(64 * yl - 4 * N)


# =============================================================================================
# Harness
# =============================================================================================
class Outcome:
    def __init__(self, final, retired, regs, cycles):
        self.final, self.retired, self.regs, self.cycles = final, retired, regs, cycles


class Harness:
    def __init__(self, dut, env=None, seed=None, mem_kw=None, p_ready=0.6, delay=(0, 12)):
        env = env or cp_env()
        self.dut, self.env = dut, env
        self.cfg = env["cfg"]
        self.has_mvm = bool(env["has_mvm"])
        self.num_ch = self.cfg["num_ch"]
        self.spm_lines = self.cfg["spm_bytes"] // 64
        self.lim = isa.dram_limit(self.cfg["ch_bytes"], self.cfg["addr_beat_w"])
        self.seed = tb_seed() if seed is None else seed
        self.rnd = random.Random(self.seed * 7919 + 13)
        self.mem = None
        self.mem_kw = dict(mem_kw or {})
        self.new_mem()
        self.csr = CsrDriver(dut, timeout_cycles=200)
        self.func = FuncUnits(self)
        self.dma = UnitStub(dut, "dma", isa.DMA_CMD_T, self.rnd, self.func.dma, p_ready, delay)
        self.qact = UnitStub(dut, "qact", isa.QACT_CMD_T, self.rnd, self.func.qact, p_ready, delay)
        self.mvm = UnitStub(dut, "mvm", isa.MVM_CMD_T, self.rnd, self.func.mvm, p_ready, delay)
        self.units = (self.dma, self.qact, self.mvm)
        dut.tb_perf_rd_beat.value = 0
        dut.tb_perf_wait.value = 0
        dut.tb_perf_mvm_busy.value = 0
        self._csr_cache = {}

    @classmethod
    async def create(cls, dut, **kw):
        await start_clock_and_reset(dut)
        return cls(dut, **kw)

    def new_mem(self, **kw):
        if self.mem is not None:
            self.mem.stop()
        k = dict(seed=self.rnd.getrandbits(31))
        k.update(self.mem_kw)
        k.update(kw)
        self.mem = MemModel(self.dut, num_ch=self.num_ch, addr_w=self.cfg["addr_beat_w"],
                            rd_outstanding=self.cfg["rd_outstanding"], limit_beats=self.lim, **k).start()
        return self.mem

    # ---- CSR helpers ----
    async def wr(self, addr, val, cache=False):
        val &= M32
        if cache and self._csr_cache.get(addr) == val:
            return
        await self.csr.write(addr, val)
        if cache:
            self._csr_cache[addr] = val

    async def rd(self, addr):
        return await self.csr.read(addr)

    async def setup_run(self, cmd_addr, count, args=None):
        args = [(a & M32) for a in (list(args or []) + [0] * 8)[:8]]
        await self.wr(isa.CSR_CMD_ADDR_LO, cmd_addr & M32, True)
        await self.wr(isa.CSR_CMD_ADDR_HI, (cmd_addr >> 32) & M32, True)
        await self.wr(isa.CSR_CMD_COUNT, count, True)
        for i, a in enumerate(args):
            await self.wr(isa.CSR_ARG0 + 4 * i, a, True)

    async def final(self):
        return await self.csr.final_state()

    async def retired(self):
        return await self.csr.read64(isa.perf_csr(isa.PERF_RETIRED))

    def regs(self) -> list:
        v = int(self.dut.u_cmdproc.rf_q.value)
        return [0] + [(v >> (32 * (i - 1))) & M32 for i in range(1, 16)]

    def busy(self) -> int:
        return int(self.dut.u_csr.busy_q.value)

    async def wait_cmds(self, unit, n, timeout=20000):
        for _ in range(timeout):
            await FallingEdge(self.dut.clk)
            if len(unit.cmds) >= n:
                return
        raise TimeoutError("%s: %d command(s) not dispatched" % (unit.name, n))

    # ---- runs ----
    def load(self, prog: bytes, cmd_addr: int, init=()):
        if prog:
            self.mem.load(0, cmd_addr * 64, prog)
        for ch, off, data in init:
            self.mem.load(ch, off, data)

    def begin(self):
        self.mem.clear_tracking()
        self.func.begin_run()
        for u in self.units:
            u.cmds.clear()

    async def launch(self, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START):
        """Load, write the run CSRs and CTRL; returns the PERF_RETIRED value before the run."""
        count = len(prog) // 32 if count is None else count
        self.load(prog, cmd_addr, init)
        self.begin()
        r0 = await self.retired()
        await self.setup_run(cmd_addr, count, args)
        await self.csr.write(isa.CSR_CTRL, ctrl)
        return r0

    async def finish(self, r0, cmd_addr, count, timeout=400000):
        cycles = await wait_level(self.dut.clk, self.dut.done, 1, timeout)
        await ClockCycles(self.dut.clk, 2)
        final = await self.final()
        retired = await self.retired() - r0
        self.check_quiet(cmd_addr, count)
        return Outcome(final, retired, self.regs(), cycles)

    async def run(self, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START,
                  timeout=400000, during=None) -> Outcome:
        """during: optional coroutine function awaited between the START write and the wait for
        DONE (stimulus in the middle of a run)."""
        count = len(prog) // 32 if count is None else count
        r0 = await self.launch(prog, cmd_addr, args, count, init, ctrl)
        if during is not None:
            await during()
        return await self.finish(r0, cmd_addr, count, timeout)

    def check_quiet(self, cmd_addr, count):
        """DONE implies a quiescent memory port; fetch reads only command beats below CMD_COUNT
        and below the DRAM limit; no other port traffic (writes, other channels)."""
        self.mem.check(idle=True)
        hi = min(cmd_addr + (count + 1) // 2, self.lim)
        fetched = self.mem.read[0] - self.func.reads[0]
        extra = sorted(b for b in fetched if not cmd_addr <= b < hi)
        assert not extra, "fetch read beats outside the command buffer: %s" % [hex(b) for b in extra[:8]]
        for c, st in self.mem.stats().items():
            assert st["wr_bursts"] == 0, "write traffic on ch%d" % c
            if c:
                assert st["rd_bursts"] == 0, "read traffic on ch%d" % c

    async def check(self, sim, prog, cmd_addr=0, args=None, count=None, init=(), ctrl=isa.CTRL_START,
                    what="", expect=None, spm=True, during=None) -> Outcome:
        """Run on the simulator and on the RTL; compare everything (see module doc)."""
        res = sim.run(prog, cmd_addr, args, init, count, ctrl)
        out = await self.run(prog, cmd_addr, args, count, init, ctrl, during=during)
        errs = []
        for k in ("status", "err_code", "err_pc", "err_info", "pc"):
            if out.final[k] != res.final[k]:
                errs.append("%s = 0x%x, simulator 0x%x" % (k, out.final[k], res.final[k]))
        if out.retired != res.retired:
            errs.append("retired %d, simulator %d" % (out.retired, res.retired))
        for i in range(1, 16):
            if out.regs[i] != res.regs[i]:
                errs.append("R%d = 0x%x, simulator 0x%x" % (i, out.regs[i], res.regs[i]))
        if spm and bytes(self.func.spm) != res.spm:
            d = next(i for i in range(len(res.spm)) if self.func.spm[i] != res.spm[i])
            errs.append("SPM differs from line %d (byte 0x%x)" % (d // 64, d))
        errs += replay.compare(res.trace, self.mem, out.final)
        if expect:
            for k, v in expect.items():
                if res.final[k] != v:
                    errs.append("test expectation %s = 0x%x, but the simulator gives 0x%x" % (k, v, res.final[k]))
        if errs:
            n = min(res.trace.cmd_count, len(prog) // 32, 64)
            self.dut._log.error("program @0x%x count %d args %s:\n%s", cmd_addr, res.trace.cmd_count,
                                [hex(a) for a in (args or [])],
                                "\n".join("%4d  %s" % (i, isa.disasm(isa.insn_words(prog[32 * i:32 * i + 32])))
                                          for i in range(n)))
        assert not errs, "%s: %s" % (what or "run", "; ".join(errs))
        return out

    def expect(self, out, what, status=None, code=None, pc=None, info=None, retired=None):
        f = out.final
        want = dict(status=status, err_code=code, err_pc=pc, err_info=info)
        bad = ["%s 0x%x (want 0x%x)" % (k, f[k], v) for k, v in want.items() if v is not None and f[k] != v]
        if retired is not None and out.retired != retired:
            bad.append("retired %d (want %d)" % (out.retired, retired))
        assert not bad, "%s: %s" % (what, ", ".join(bad))


class Counter3(list):
    """Element-wise accumulating [dma, qact, mvm] counter."""

    def __init__(self):
        super().__init__([0, 0, 0])

    def __iadd__(self, other):
        for i, v in enumerate(other):
            self[i] += v
        return self


# Program helpers and the random program generator live in tb/common/proggen.py (shared with
# tb/core and the M3 testbenches); ProgGen(m3=False) generates exactly the programs it did here.
from common.proggen import ProgGen, enc_mvm, prog_bytes, raw, with_base  # noqa: E402,F401


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def cp_csr_map(dut):
    """Full CSR map vs the simulator (every 12-bit offset, aligned and unaligned): reset values,
    constants, RW registers, ignored writes to RO / unmapped / unaligned offsets."""
    h = await Harness.create(dut)
    sim = simlib.Sim(h.cfg)
    cfg = h.cfg
    assert int(dut.done.value) == 0
    # Spot checks against the spec (test_sim_basic.c test_csr).
    assert await h.rd(isa.CSR_ID) == isa.ID_MAGIC
    assert await h.rd(isa.CSR_VERSION) == 0x0100
    assert await h.rd(isa.CSR_CFG0) == cfg["num_ch"] | (cfg["lanes"] << 8) | (cfg["t_max"] << 20)
    assert await h.rd(isa.CSR_CFG1) == cfg["spm_bytes"] // 64
    assert await h.rd(isa.CSR_CFG2) == cfg["vl"] | (cfg["nsreg"] << 8) | ((cfg["k_max"] // 256) << 16)
    assert await h.rd(isa.CSR_CAPS) == 0
    assert await h.rd(isa.CSR_STATUS) == 0 and await h.rd(isa.CSR_CTRL) == 0

    async def sweep(tag):
        bad = []
        for off in range(1 << isa.CSR_AW):
            got = await h.rd(off)
            want = sim.csr_read(off)
            if got != want:
                bad.append("0x%03x: 0x%08x (simulator 0x%08x)" % (off, got, want))
        assert not bad, "%s: %d CSR offsets differ, first %s" % (tag, len(bad), bad[:6])

    await sweep("after reset")
    rnd = random.Random(h.seed)
    for off in range(1 << isa.CSR_AW):
        if off == isa.CSR_CTRL:
            continue
        v = rnd.getrandbits(32)
        await h.wr(off, v)
        sim.csr_write(off, v)
    await sweep("after writing every offset")
    assert await h.rd(isa.CSR_ID) == isa.ID_MAGIC and await h.rd(isa.CSR_ERR_CODE) == 0
    sim.close()


@cocotb.test()
async def cp_run_control(dut):
    """test_sim_basic.c test_csr: a run, PC, STATUS W1C, done output, error run, START clears
    ERR_*, START while BUSY ignored (DMA held), BUSY not W1C."""
    h = await Harness.create(dut)
    sim = simlib.Sim(h.cfg)
    ch = min(1, h.num_ch - 1)
    p = prog_bytes([isa.enc_ld(ch, 10, IO_BEAT, 3), isa.enc_nop(), isa.enc_end()])
    out = await h.check(sim, p, what="LD NOP END", expect=dict(status=DONE))
    h.expect(out, "LD NOP END", status=DONE, code=0, pc=0, info=0, retired=3)
    assert h.dma.cmds == [dict(is_st=0, ch=ch, spm_line=10, dram_beat=IO_BEAT, n_lines=3)], h.dma.cmds
    assert out.final["pc"] == 2 and int(dut.done.value) == 1
    # STATUS W1C: BUSY not writable, DONE cleared by writing 1; done output follows STATUS.DONE.
    await h.wr(isa.CSR_STATUS, isa.STATUS_BUSY)
    assert await h.rd(isa.CSR_STATUS) == DONE and int(dut.done.value) == 1
    await h.wr(isa.CSR_STATUS, isa.STATUS_DONE)
    assert await h.rd(isa.CSR_STATUS) == 0 and int(dut.done.value) == 0
    # Error run: DONE | ERR; ERR W1C keeps ERR_CODE; the next START clears ERR_*.
    out = await h.check(sim, prog_bytes([raw(0xEE)]), what="illegal opcode",
                        expect=dict(status=DONEERR, err_code=isa.ERR_ILLEGAL_OP))
    await h.wr(isa.CSR_STATUS, isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == DONE and await h.rd(isa.CSR_ERR_CODE) == isa.ERR_ILLEGAL_OP
    out = await h.check(sim, prog_bytes([isa.enc_end()]), what="START clears ERR_*")
    h.expect(out, "START clears ERR_*", status=DONE, code=0, pc=0, info=0, retired=1)
    sim.close()

    # START while BUSY is ignored (the first latched CMD_COUNT is used); BUSY is not W1C.
    p = prog_bytes([isa.enc_ld(0, 0, IO_BEAT, 1), isa.enc_nop(), isa.enc_end()])
    h.dma.hold = True
    r0 = await h.launch(p, 0, count=2)
    await h.wait_cmds(h.dma, 1)
    assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY and int(dut.done.value) == 0
    await h.wr(isa.CSR_CMD_COUNT, 3)
    await h.wr(isa.CSR_CTRL, isa.CTRL_START)
    await h.wr(isa.CSR_STATUS, isa.STATUS_BUSY | isa.STATUS_DONE | isa.STATUS_ERR)
    assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY
    h._csr_cache.pop(isa.CSR_CMD_COUNT, None)
    h.dma.hold = False
    out = await h.finish(r0, 0, 2)
    h.expect(out, "START while busy", status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=2, info=0, retired=2)
    assert len(h.dma.cmds) == 1


@cocotb.test()
async def cp_soft_reset(dut):
    """SOFT_RESET: idle (clears STATUS, ERR_*, PC, PERF, R1..R15; keeps CMD_*, ARG*), with START in
    the same write, and while BUSY (abort at the next boundary, reset applied at the run end: no
    DONE; START in the same write ignored)."""
    h = await Harness.create(dut)
    p = prog_bytes([isa.enc_li(1, 11), isa.enc_li(2, 22), raw(0xEE)])
    out = await h.run(p, 0, args=[5, 1001])
    h.expect(out, "error run", status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=2, retired=2)
    assert out.regs[1] == 11 and out.regs[8] == 5 and out.regs[9] == 1001
    perf = await h.csr.perf(h.num_ch)
    assert perf[isa.PERF_RETIRED] >= 2 and perf[isa.PERF_CYCLES] > 0
    await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
    st = await h.final()
    assert st == dict(status=0, err_code=0, err_pc=0, err_info=0, pc=0), st
    assert all(v == 0 for v in await h.csr.perf(h.num_ch))
    assert h.regs() == [0] * 16 and int(dut.done.value) == 0
    assert await h.rd(isa.CSR_CMD_COUNT) == 3 and await h.rd(isa.CSR_ARG0 + 4) == 1001

    # SOFT_RESET | START when idle: reset, then a normal run (counters start from 0).
    p = prog_bytes([isa.enc_li(3, 33), isa.enc_end()])
    await h.run(prog_bytes([isa.enc_nop(), isa.enc_end()]), 0)
    out = await h.run(p, 64, ctrl=isa.CTRL_SOFT_RESET | isa.CTRL_START)
    h.expect(out, "SOFT_RESET|START", status=DONE, code=0)
    assert await h.retired() == 2 and out.regs[3] == 33

    # SOFT_RESET while BUSY (and with START in the same write): the instruction in flight
    # completes, the next one is not executed, no DONE, everything reset; then a normal run.
    for ctrl in (isa.CTRL_SOFT_RESET, isa.CTRL_SOFT_RESET | isa.CTRL_START):
        p = prog_bytes([isa.enc_li(4, 44), isa.enc_ld(0, 0, IO_BEAT, 2), isa.enc_st(0, 0, IO_BEAT + 8, 1),
                        isa.enc_end()])
        h.dma.hold = True
        await h.launch(p, 128, args=[7])
        await h.wait_cmds(h.dma, 1)
        await h.wr(isa.CSR_CTRL, ctrl)
        assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY
        await ClockCycles(dut.clk, 30)
        assert await h.rd(isa.CSR_STATUS) == isa.STATUS_BUSY
        h.dma.hold = False
        for _ in range(2000):
            await FallingEdge(dut.clk)
            assert int(dut.done.value) == 0, "done rose on a run ended by SOFT_RESET"
            if not h.busy():
                break
        await ClockCycles(dut.clk, 20)
        assert int(dut.done.value) == 0
        st = await h.final()
        assert st == dict(status=0, err_code=0, err_pc=0, err_info=0, pc=0), st
        assert all(v == 0 for v in await h.csr.perf(h.num_ch)), "PERF not cleared"
        assert h.regs() == [0] * 16 and len(h.dma.cmds) == 1, "instruction after SOFT_RESET executed"
        h.check_quiet(128, 4)
        out = await h.run(prog_bytes([isa.enc_li(6, 66), isa.enc_end()]), 256, args=[7])
        h.expect(out, "run after SOFT_RESET", status=DONE, code=0, retired=2)
        assert out.regs[6] == 66 and out.regs[8] == 7      # ARG0 kept by SOFT_RESET (not rewritten)


@cocotb.test()
async def cp_abort(dut):
    """ABORT (test_sim_basic.c test_abort + RTL timing cases): START|ABORT aborts before pc 0;
    ABORT while idle is cleared by START; ABORT while a unit runs stops at the next boundary
    (ERR_PC = first instruction not executed, PC unchanged); ABORT while waiting for the fetch;
    DONE only after outstanding fetch reads drained."""
    h = await Harness.create(dut)
    p = prog_bytes([isa.enc_li(1, 9), isa.enc_end()])
    out = await h.run(p, 0, ctrl=isa.CTRL_START | isa.CTRL_ABORT)
    h.expect(out, "START|ABORT", status=DONEERR, code=isa.ERR_ABORTED, pc=0, info=0, retired=0)
    assert out.regs[1] == 0 and out.final["pc"] == 0
    await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)                      # idle: no effect on the next run
    out = await h.run(p, 0)
    h.expect(out, "ABORT while idle", status=DONE, code=0, retired=2)
    assert out.regs[1] == 9
    out = await h.run(b"", 0, count=0, ctrl=isa.CTRL_START | isa.CTRL_ABORT)
    h.expect(out, "START|ABORT, CMD_COUNT 0", status=DONEERR, code=isa.ERR_ABORTED, pc=0, info=0, retired=0)
    out = await h.run(p, 0, ctrl=isa.CTRL_SOFT_RESET | isa.CTRL_START | isa.CTRL_ABORT)
    h.expect(out, "SOFT_RESET|START|ABORT", status=DONEERR, code=isa.ERR_ABORTED, pc=0)
    assert await h.retired() == 0                                 # counters were reset, nothing retired

    # ABORT while the DMA runs: the LD completes, the next instruction is not executed.
    p = prog_bytes([isa.enc_li(1, 5), isa.enc_ld(0, 3, IO_BEAT, 4), isa.enc_li(2, 7), isa.enc_end()])
    h.dma.hold = True
    r0 = await h.launch(p, 64)
    await h.wait_cmds(h.dma, 1)
    await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)
    await ClockCycles(dut.clk, 25)
    assert h.busy() and await h.rd(isa.CSR_PC) == 1
    h.dma.hold = False
    out = await h.finish(r0, 64, 4)
    h.expect(out, "ABORT during LD", status=DONEERR, code=isa.ERR_ABORTED, pc=2, info=0, retired=2)
    assert out.final["pc"] == 1 and out.regs[1] == 5 and out.regs[2] == 0

    # ABORT during the last instruction of the buffer: ABORTED (checked before the end of the
    # buffer), ERR_PC = CMD_COUNT.
    p = prog_bytes([isa.enc_li(1, 6), isa.enc_ld(0, 3, IO_BEAT, 1)])
    h.dma.hold = True
    r0 = await h.launch(p, 96)
    await h.wait_cmds(h.dma, 1)
    await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)
    h.dma.hold = False
    out = await h.finish(r0, 96, 2)
    h.expect(out, "ABORT at CMD_COUNT", status=DONEERR, code=isa.ERR_ABORTED, pc=2, info=0, retired=2)
    assert out.final["pc"] == 1

    # ABORT while cmdproc waits for an instruction (slow memory): ERR_PC = that pc; DONE only
    # after the outstanding fetch reads came back.
    h.new_mem(rd_lat=(600, 600), p_stall=0.0)
    p = prog_bytes([isa.enc_nop()] * 6 + [isa.enc_end()])
    r0 = await h.launch(p, 1024)
    await ClockCycles(dut.clk, 80)
    assert h.busy()
    await h.wr(isa.CSR_CTRL, isa.CTRL_ABORT)
    await ClockCycles(dut.clk, 40)
    assert h.busy() and int(dut.done.value) == 0, "finished before the fetch drained"
    out = await h.finish(r0, 1024, 7)
    h.expect(out, "ABORT while fetching", status=DONEERR, code=isa.ERR_ABORTED, pc=0, retired=0)
    assert out.cycles > 300
    h.new_mem()


class _NoReset:
    """rst_n stand-in for a MemModel that is not reset together with the core."""
    value = 1


@cocotb.test()
async def cp_stray_rsp(dut):
    """Read responses with nothing outstanding are dropped by tsr_fetch: they neither enter the
    buffer nor move its outstanding-beat count. They reach the core only if it is reset while the
    memory side keeps bursts in flight (outside the reset contract); the core must still work.
      A  hardware reset in the middle of a run, memory model not reset: the rest of the fetch
         bursts arrive after the reset;
      B  stray beats while idle, more than the fetch buffer holds;
      C  stray beats in F_RUN with nothing outstanding: fetch waiting for buffer space behind a
         held LD (pushed, they would run as instructions or overwrite unread ones), and a
         completely prefetched program (the run end waits for the outstanding count to reach 0).
    The stray beats hold LI R5..R7 instructions. After every case fetch must be idle with nothing
    counted as outstanding, and runs must match the simulator with a quiet memory port.
    Runs with +tsr_allow_stray_rsp (tsr_fetch's protocol assertion off; see TEST_PLUSARGS).
    Environment: TSR_CP_STRAY = subset of the cases (default ABC)."""
    cases = os.environ.get("TSR_CP_STRAY", "ABC")
    h = await Harness.create(dut, delay=(0, 3))
    sim = simlib.Sim(h.cfg)
    rnd = random.Random(h.seed + 17)
    fb = cp_env()["fetch_beats"]
    f = dut.u_fetch
    n_long = 4 * fb + 7                                   # several bursts, more than the buffer
    assert h.lim > 4096 + n_long, "configuration too small for cp_stray_rsp"
    init = [(0, isa.REGION_IO_OFF, rnd.randbytes(64 * 8))]

    def li_prog(n, nreg=4):
        return prog_bytes([isa.enc_li(1 + i % nreg, rnd.getrandbits(32)) for i in range(n - 1)] + [isa.enc_end()])

    def stray_beat(i):
        a = isa.insn_bytes(isa.enc_li(5 + i % 3, 0xDEAD0000 + i))
        b = isa.insn_bytes(isa.enc_li(5 + (i + 1) % 3, 0xBEEF0000 + i))
        return int.from_bytes(a + b, "little")

    def fetch_state():
        return dict(st=int(f.st_q.value), infl=int(f.infl_q.value), fcnt=int(f.fcnt_q.value))

    async def until(cond, what, timeout=20000):
        for _ in range(timeout):
            await FallingEdge(dut.clk)
            if cond():
                return
        raise TimeoutError("%s: not reached; fetch %s, req_left %d, memory idle %s"
                           % (what, fetch_state(), int(f.req_left_q.value), h.mem.idle()))

    async def inject(n):
        """Stop the (idle) memory model and drive n stray beats with the fetch id on channel 0."""
        assert h.mem.idle()
        h.mem.stop()
        dut.m_rd_req_ready.value = 0
        k = 0
        while k < n:
            await RisingEdge(dut.clk)
            v = int(rnd.random() < 0.75)
            dut.m_rd_rsp_valid.value = v
            dut.m_rd_rsp_id.value = isa.SRC_FETCH << isa.SRC_ID_W
            dut.m_rd_rsp_data.value = stray_beat(k)
            dut.m_rd_rsp_last.value = int(k % 4 == 3)
            k += v
        await RisingEdge(dut.clk)
        dut.m_rd_rsp_valid.value = 0
        dut.m_rd_rsp_last.value = 0
        h.mem._last_out.clear()                           # the model re-drives all of its outputs
        h.mem.start()
        await FallingEdge(dut.clk)
        assert fetch_state()["infl"] == 0, "stray beats counted: %s" % fetch_state()

    async def after(what):
        for i in range(3):
            n = rnd.choice([1, 2, 5, n_long])
            await h.check(sim, li_prog(n, 7), rnd.randrange(4096), what="%s, run %d" % (what, i))

    # ---- A: hardware reset in the middle of a run; the memory side keeps its bursts ----
    stray_a, n_a = 0, 0
    for trial in range(8 if "A" in cases else 0):
        h.new_mem(rst_n=_NoReset(), rd_lat=(20, 120), p_rd_rsp_valid=0.5)
        c0 = h.mem.ch[0]
        await h.launch(li_prog(4 * n_long), rnd.randrange(4096))
        await ClockCycles(dut.clk, rnd.randrange(0, 60))
        await until(lambda: int(dut.done.value) or (c0.rd_bursts and not int(dut.m_rd_req_valid.value) & 1),
                    "fetch bursts in flight")
        if int(dut.done.value):
            continue
        dut.rst_n.value = 0                               # sampled at the next rising edge
        await ClockCycles(dut.clk, rnd.randint(1, 4))
        dut.rst_n.value = 1
        h._csr_cache.clear()                              # the reset cleared CMD_* and ARG*
        b0 = c0.stats["rd_beats"]
        await until(lambda: not c0.rd_bursts, "memory side drained")
        await ClockCycles(dut.clk, 2)
        stray_a += c0.stats["rd_beats"] - b0
        n_a += 1
        assert fetch_state() == dict(st=0, infl=0, fcnt=0), "after the reset: %s" % fetch_state()
        assert int(dut.done.value) == 0 and not h.busy()
        h.new_mem()
        await after("reset mid-run (trial %d)" % trial)
    dut._log.info("cp_stray_rsp A: %d resets mid-run, %d stray beats after the reset", n_a, stray_a)
    assert "A" not in cases or (n_a >= 4 and stray_a > 0)

    # ---- B: stray beats while idle ----
    for trial in range(3 if "B" in cases else 0):
        await inject(rnd.randint(1, 3 * fb + 5))
        assert fetch_state() == dict(st=0, infl=0, fcnt=0), fetch_state()
        await after("stray beats while idle (trial %d)" % trial)

    # ---- C: stray beats in F_RUN with nothing outstanding ----
    async def c_full():
        await h.wait_cmds(h.dma, 1)
        for _ in range(50):                               # stable: fetch waits for buffer space
            await until(lambda: fetch_state()["infl"] == 0 and int(f.req_left_q.value) and h.mem.idle(),
                        "fetch waiting for buffer space")
            s0 = fetch_state()
            await ClockCycles(dut.clk, 10)
            if fetch_state() == s0:
                break
        assert s0["st"] == 3 and s0["infl"] == 0 and s0["fcnt"] > 0 and fetch_state() == s0, s0   # F_RUN
        await inject(rnd.randint(1, 2 * fb))
        assert fetch_state() == s0, (s0, fetch_state())
        h.dma.hold = False

    async def c_prefetched():
        await h.wait_cmds(h.dma, 1)
        await until(lambda: fetch_state()["infl"] == 0 and int(f.req_left_q.value) == 0 and h.mem.idle(),
                    "program prefetched")
        s0 = fetch_state()
        assert s0["st"] == 3, s0
        await inject(rnd.randint(1, 2 * fb))
        assert fetch_state() == s0, (s0, fetch_state())
        h.dma.hold = False

    for trial in range(3 if "C" in cases else 0):
        h.dma.hold = True
        p = prog_bytes([isa.enc_ld(0, trial, IO_BEAT, 1)] +
                       [isa.enc_li(1 + i % 4, rnd.getrandbits(32)) for i in range(n_long)] + [isa.enc_end()])
        await h.check(sim, p, rnd.randrange(4096), init=init, what="stray beats, fetch waiting (%d)" % trial,
                      during=c_full)
        h.dma.hold = True
        p = prog_bytes([isa.enc_ld(0, 8 + trial, IO_BEAT + 1, 1), isa.enc_li(1, trial), isa.enc_end()])
        await h.check(sim, p, rnd.randrange(4096), init=init, what="stray beats, prefetched (%d)" % trial,
                      during=c_prefetched)
        await after("stray beats in F_RUN (trial %d)" % trial)
    sim.close()


@cocotb.test()
async def cp_regs_flow(dut):
    """test_sim_basic.c test_regs_and_flow: LI / ADDI (wrap, R0 discarded), ARG -> R8..R15, FENCE,
    NOP, nothing after END, R1..R7 cleared by START; odd command address / count; past CMD_COUNT;
    CMD_COUNT 0; fetch beyond the DRAM limit (mid-program, at CMD_ADDR, CMD_ADDR >= 2^32)."""
    h = await Harness.create(dut)
    sim = simlib.Sim(h.cfg)
    args = [10, 11, 12, 13, 14, 15, 16, 0xFFFFFFF0]
    p = prog_bytes([isa.enc_li(1, 0x7FFFFFFF), isa.enc_addi(2, 1, 1), isa.enc_addi(3, 2, M32),
                    isa.enc_li(0, 1234), isa.enc_addi(0, 1, 5), isa.enc_addi(4, 0, 77),
                    isa.enc_addi(5, 8, 1), isa.enc_addi(6, 15, 0), isa.enc_fence(), isa.enc_nop(),
                    isa.enc_end(), isa.enc_li(7, 99)])
    out = await h.check(sim, p, 0, args, what="regs", expect=dict(status=DONE))
    assert out.regs[1:8] == [0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 77, 11, 0xFFFFFFF0, 0], out.regs
    assert out.regs[8:] == args and out.retired == 11
    out = await h.check(sim, prog_bytes([isa.enc_end()]), 0, what="regs cleared by START")
    assert out.regs == [0] * 16
    for cmd, n in ((12345, 7), (63, 2), (64 * 64 - 1, 5), (31, 1), (1, 9), (64 * 3 - 3, 12)):
        p = prog_bytes([isa.enc_addi(1, 1, 1)] * (n - 1) + [isa.enc_end()])
        out = await h.check(sim, p, cmd, what="cmd at beat %d, %d insns" % (cmd, n))
        assert out.regs[1] == n - 1
    p = prog_bytes([isa.enc_li(1, 5), isa.enc_nop()])
    await h.check(sim, p, 0, what="past CMD_COUNT", expect=dict(err_code=isa.ERR_ILLEGAL_OP, err_pc=2))
    await h.check(sim, b"", 0, count=0, what="CMD_COUNT 0", expect=dict(err_code=isa.ERR_ILLEGAL_OP, err_pc=0))
    lim = h.lim
    for k in range(0, 4):                    # program starting k beats below the limit
        p = prog_bytes([isa.enc_nop()] * (2 * k + 3))
        await h.check(sim, p, lim - k, what="fetch crosses the limit (k=%d)" % k,
                      expect=dict(err_code=isa.ERR_DRAM_RANGE, err_pc=2 * k, err_info=lim & M32))
    # CMD_ADDR compared in 64 bits: CMD_ADDR_HI != 0 is beyond a limit below 2^32 beats but may be
    # inside a larger one (then the NOP runs and the run ends past CMD_COUNT).
    for cmd in (lim, lim + 5, (1 << 32) - 1, 1 << 32, (1 << 32) + 3, (1 << 40) + 3, (1 << 64) - 1, 1 << 63):
        exp = (dict(err_code=isa.ERR_DRAM_RANGE, err_pc=0, err_info=cmd & M32) if cmd >= lim else
               dict(err_code=isa.ERR_ILLEGAL_OP, err_pc=1, err_info=0))
        await h.check(sim, prog_bytes([isa.enc_nop()]), cmd, count=1, what="CMD_ADDR 0x%x" % cmd, expect=exp)
    # A program across beat 2^32 (carry into CMD_ADDR_HI): DRAM_RANGE at the first pc whose beat
    # is at or beyond the limit, else it runs to END.
    for cmd in ((1 << 32) - 1, (1 << 32) - 3):
        n = 9
        k = 2 * (lim - cmd) if cmd < lim else 0
        exp = (dict(err_code=isa.ERR_DRAM_RANGE, err_pc=k, err_info=(cmd + k // 2) & M32) if k < n else
               dict(status=DONE))
        await h.check(sim, prog_bytes([isa.enc_addi(1, 1, 1)] * (n - 1) + [isa.enc_end()]), cmd,
                      what="program across beat 2^32 at 0x%x" % cmd, expect=exp)
    sim.close()


@cocotb.test()
async def cp_fetch(dut):
    """Prefetch under stress: long straight-line programs across 4 KiB blocks with random memory
    latency, odd / even starts and counts; programs much shorter than CMD_COUNT (prefetch stops
    at CMD_COUNT); huge CMD_COUNT with an early END; programs ending exactly at the DRAM limit."""
    h = await Harness.create(dut, delay=(0, 3))
    sim = simlib.Sim(h.cfg)
    rnd = random.Random(h.seed + 5)
    for it in range(10):
        n = rnd.choice([1, 2, 3, 31, 32, 33, 64, 127, 128, 129, 300, 600])
        cmd = rnd.choice([0, 1, 63, 62, 64 * 7 - 5, rnd.randrange(1 << 20)])
        insns = []
        for i in range(n - 1):
            insns.append(rnd.choice([isa.enc_addi(1 + i % 7, 1 + (i + 3) % 7, i), isa.enc_li(1 + i % 15, i),
                                     isa.enc_nop(), isa.enc_fence()]))
        insns.append(isa.enc_end())
        count = n + rnd.choice([0, 0, 1, 2, 100])
        await h.check(sim, prog_bytes(insns), cmd, count=count, what="straight line %d @%d" % (n, cmd))
    # Huge CMD_COUNT with END early: prefetch stays bounded by the buffer.
    await h.check(sim, prog_bytes([isa.enc_nop(), isa.enc_end()]), 4096, count=M32, what="count 2^32-1")
    # A program whose last beat is the last beat below the limit.
    lim = h.lim
    for n in (1, 2, 3, 4, 40):
        p = prog_bytes([isa.enc_nop()] * (n - 1) + [isa.enc_end()])
        await h.check(sim, p, lim - (n + 1) // 2, what="ends at the limit (%d)" % n, expect=dict(status=DONE))
    # Long programs across beat 2^32 (DRAM limit above it): bursts on both sides of the carry.
    if lim > (1 << 32) + 400:
        for k in (1, 37, 64, 200):
            p = prog_bytes([isa.enc_li(1 + i % 15, i) for i in range(599)] + [isa.enc_end()])
            await h.check(sim, p, (1 << 32) - k, what="across beat 2^32 (-%d)" % k, expect=dict(status=DONE))
    # Slow, bursty memory (fetch buffer drains between responses).
    h.new_mem(rd_lat=(30, 120), p_rd_rsp_valid=0.3)
    for it in range(3):
        n = rnd.choice([20, 70, 150])
        p = prog_bytes([isa.enc_li(1 + i % 15, rnd.getrandbits(32)) for i in range(n)] + [isa.enc_end()])
        await h.check(sim, p, rnd.randrange(1 << 16), what="slow memory %d" % n)
    h.new_mem()
    sim.close()


@cocotb.test()
async def cp_ldst(dut):
    """test_sim_basic.c test_ldst: dispatch fields, register-relative operands with 32-bit wrap,
    zero length (no dispatch), exact fits at the SPM end and the DRAM limit, every channel."""
    h = await Harness.create(dut)
    sim = simlib.Sim(h.cfg)
    rnd = random.Random(h.seed + 9)
    init = [(c, isa.REGION_IO_OFF, rnd.randbytes(64 * 1024)) for c in range(h.num_ch)]
    c2, c3, c1 = (2 % h.num_ch), (3 % h.num_ch), (1 % h.num_ch)
    insns = [isa.enc_ld(c2, 100, IO_BEAT + 7, 5), isa.enc_st(c3, 100, IO_BEAT + 512, 5),
             isa.enc_li(1, M32), isa.enc_li(2, IO_BEAT + 600),
             with_base(isa.enc_ld(0, 201, 0xFFFFFFF0, 2), ra=1, rb=2),
             isa.enc_li(4, 200), with_base(isa.enc_st(c1, 0, 5, 2), ra=4, rb=2),
             isa.enc_ld(0, 300, IO_BEAT, 0), isa.enc_end()]
    out = await h.check(sim, prog_bytes(insns), 0, init=init, what="ldst", expect=dict(status=DONE))
    want = [dict(is_st=0, ch=c2, spm_line=100, dram_beat=IO_BEAT + 7, n_lines=5),
            dict(is_st=1, ch=c3, spm_line=100, dram_beat=IO_BEAT + 512, n_lines=5),
            dict(is_st=0, ch=0, spm_line=200, dram_beat=IO_BEAT + 600 - 16, n_lines=2),
            dict(is_st=1, ch=c1, spm_line=200, dram_beat=IO_BEAT + 605, n_lines=2)]
    assert h.dma.cmds == want, h.dma.cmds
    assert out.retired == 9
    SL, lim = h.spm_lines, h.lim
    top = min(lim, 1 << 32)                  # highest DRAM end the 32-bit operand fields reach
    for ins, what in ((isa.enc_ld(0, SL - 2, top - 2, 2), "LD exact fit"),
                      (isa.enc_st(h.num_ch - 1, SL - 1, top - 1, 1), "ST exact fit"),
                      (isa.enc_ld(0, SL, lim, 0), "zero length at the limits")):
        await h.check(sim, prog_bytes([ins, isa.enc_end()]), 0, what=what, expect=dict(status=DONE))
    for ch in range(h.num_ch):
        await h.check(sim, prog_bytes([isa.enc_ld(ch, ch, IO_BEAT + ch, 1), isa.enc_st(ch, ch, IO_BEAT + 4096, 1),
                                       isa.enc_end()]), 0, what="channel %d" % ch)
        assert [c["ch"] for c in h.dma.cmds] == [ch, ch]
    sim.close()


@cocotb.test()
async def cp_reserved(dut):
    """test_sim_basic.c test_reserved_fields: every bit an instruction does not use -> ILLEGAL_OP
    (ERR_INFO 0, nothing retired, no dispatch); sub codes (LD/ST channel >= NUM_CH, QACT kind >= 2,
    MVM fmt >= 2 -> BAD_FMT); every opcode outside the implemented set; MVM without HAS_MVM."""
    h = await Harness.create(dut, delay=(0, 2))
    sim = simlib.Sim(h.cfg)
    OPC, SUB, RA, RB, RC, RD = 0xFF, 0xF00, 0xF0000, 0xF00000, 0xF000000, 0xF0000000
    cases = [("NOP", isa.enc_nop(), [OPC, 0, 0, 0, 0, 0, 0, 0]),
             ("END", isa.enc_end(), [OPC, 0, 0, 0, 0, 0, 0, 0]),
             ("FENCE", isa.enc_fence(), [OPC, 0, 0, 0, 0, 0, 0, 0]),
             ("LI", isa.enc_li(1, 5), [OPC | RD, M32, 0, 0, 0, 0, 0, 0]),
             ("ADDI", isa.enc_addi(1, 2, 5), [OPC | RA | RD, M32, 0, 0, 0, 0, 0, 0]),
             ("LD", isa.enc_ld(0, 0, IO_BEAT, 1), [OPC | SUB | RA | RB, M32, M32, M32, 0, 0, 0, 0]),
             ("ST", isa.enc_st(0, 0, IO_BEAT, 1), [OPC | SUB | RA | RB, M32, M32, M32, 0, 0, 0, 0]),
             ("QACT", isa.enc_qact(0, 0, 100, 32, 1, 0), [OPC | SUB | RA | RB, M32, M32, M32, M32, M32, 0, 0])]
    if h.has_mvm:
        cases.append(("MVM", enc_mvm(0, 0, 0, 100, WOFF, 4, 32, 1, 0),
                      [OPC | SUB | 0x1000 | RA | RB | RC, M32, M32, M32, M32, 0x000FFFFF, M32, 0]))
    for name, base, used in cases:
        if name != "END":
            await h.check(sim, prog_bytes([base, isa.enc_end()]), 0, what=name, expect=dict(status=DONE))
        n = 0
        for wi in range(8):
            for b in range(32):
                if (used[wi] >> b) & 1:
                    continue
                bad = list(base)
                bad[wi] ^= 1 << b
                out = await h.check(sim, prog_bytes([bad, isa.enc_end()]), 0, what="%s w%d bit %d" % (name, wi, b),
                                    expect=dict(err_code=isa.ERR_ILLEGAL_OP, err_pc=0, err_info=0))
                assert out.retired == 0 and not any(u.cmds for u in h.units)
                n += 1
        assert n > 0
    for sub in range(16):
        for name, ins, ok, err in (
                ("LD ch", isa.enc_ld(sub, 0, IO_BEAT, 1), sub < h.num_ch, isa.ERR_ILLEGAL_OP),
                ("ST ch", isa.enc_st(sub, 0, IO_BEAT, 1), sub < h.num_ch, isa.ERR_ILLEGAL_OP),
                ("QACT kind", isa.enc_qact(sub, 0, 100, 256, 1, 0), sub < 2, isa.ERR_ILLEGAL_OP),
                ("MVM fmt", enc_mvm(sub, 0, 0, 100, WOFF, 4, 256, 1, 0), sub < 2, isa.ERR_BAD_FMT)):
            if name == "MVM fmt" and not h.has_mvm:
                continue
            exp = dict(status=DONE) if ok else dict(err_code=err, err_pc=0, err_info=0)
            await h.check(sim, prog_bytes([ins, isa.enc_end()]), 0, what="%s %d" % (name, sub), expect=exp)
    impl = set(isa.M2_OPS) | ({isa.OP_MVM} if h.has_mvm else set())
    for op in range(256):
        if op in impl:
            continue
        p = prog_bytes([raw(op), isa.enc_end()])
        if op == isa.OP_MVM:          # HAS_MVM = 0: absent from this bitstream (the simulator has it)
            out = await h.run(p, 0)
        else:
            out = await h.check(sim, p, 0, what="opcode 0x%02x" % op)
        h.expect(out, "opcode 0x%02x" % op, status=DONEERR, code=isa.ERR_ILLEGAL_OP, pc=0, info=0, retired=0)
    for w, what in ((raw(isa.w0(isa.OP_MULI, ra=2, rd=1), 3), "MULI"),
                    (raw(isa.w0(isa.OP_SOP, sub=0), 1, 0x3F800000), "SOP MOVI"),
                    (raw(isa.w0(isa.OP_LOOP, ra=1), 4, 2), "LOOP"),
                    (raw(isa.w0(isa.OP_JUMP), 0, 4), "JUMP")):
        await h.check(sim, prog_bytes([w, isa.enc_end()]), 0, what=what,
                      expect=dict(err_code=isa.ERR_ILLEGAL_OP, err_pc=0, err_info=0))
    sim.close()


@cocotb.test()
async def cp_errors(dut):
    """test_sim_basic.c test_errors: every M2 error code with ERR_PC / ERR_INFO and its priority,
    register-relative operands, 32-bit wrap, 64-bit extents; MVM checks when HAS_MVM."""
    h = await Harness.create(dut)
    sim = simlib.Sim(h.cfg)
    rnd = random.Random(h.seed + 3)
    init = [(c, isa.REGION_IO_OFF, rnd.randbytes(64 * 64)) for c in range(h.num_ch)]
    await h.check(sim, prog_bytes([isa.enc_end()]), 0, init=init)
    SL, LIM, T = h.spm_lines, h.lim, h.cfg["t_max"]
    KM = h.cfg["k_max"]
    ch3 = min(3, h.num_ch - 1)
    setup = [isa.enc_li(1, SL - 3), isa.enc_li(2, M32), isa.enc_li(3, LIM - 10)]
    E = isa

    async def err(ins, code, info, what, pre=()):
        p = prog_bytes(list(pre) + [ins, isa.enc_end()])
        out = await h.check(sim, p, 0, what=what, expect=dict(err_code=code, err_pc=len(pre), err_info=info))
        assert out.retired == len(pre)
        return out

    async def ok(ins, what, pre=()):
        await h.check(sim, prog_bytes(list(pre) + [ins, isa.enc_end()]), 0, what=what, expect=dict(status=DONE))

    def dram(a, e):
        """Expected (code, info) of a case whose only possible error is the DRAM range rule
        a + e > LIM (64-bit), a = the operand's 32-bit start address. A limit above 2^32 beats is
        out of reach of short operands: those cases must pass."""
        a &= M32
        return (E.ERR_DRAM_RANGE, a) if a + e > LIM else (0, 0)

    async def case(ins, exp, what, pre=()):
        if exp[0]:
            await err(ins, exp[0], exp[1], what, pre)
        else:
            await ok(ins, what + " (in range)", pre)

    # LD / ST
    await err(E.enc_ld(0, SL - 1, IO_BEAT, 2), E.ERR_SPM_RANGE, SL - 1, "LD SPM")
    await case(E.enc_ld(1 % h.num_ch, 0, LIM - 1, 2), dram(LIM - 1, 2), "LD DRAM")
    await err(E.enc_st(2 % h.num_ch, SL, IO_BEAT, 1), E.ERR_SPM_RANGE, SL, "ST SPM")
    await case(E.enc_st(ch3, 0, M32, 1), dram(M32, 1), "ST DRAM")
    await err(E.enc_st(ch3, 0, IO_BEAT, M32), E.ERR_SPM_RANGE, 0, "ST huge n")
    await err(E.enc_ld(0, SL + 1, LIM, 0), E.ERR_SPM_RANGE, SL + 1, "LD n=0 start past SPM end")
    await case(E.enc_ld(0, 0, LIM + 1, 0), dram(LIM + 1, 0), "LD n=0 start past DRAM limit")
    # The top of the 32-bit DRAM field: with a limit above 2^32 beats only n = LIM - a + 1 lines
    # reach it (if the SPM holds them).
    n_top = LIM - M32 + 1
    if 2 <= n_top <= SL:
        await err(E.enc_ld(0, 0, M32, n_top), E.ERR_DRAM_RANGE, M32, "LD DRAM at the top of the 32-bit field")
        await ok(E.enc_st(ch3, 0, M32, n_top - 1), "ST exact fit at the top of the 32-bit field")
    if h.num_ch < 16:
        await err(E.enc_ld(h.num_ch, SL, LIM, 1), E.ERR_ILLEGAL_OP, 0, "LD ch+SPM+DRAM")
    await err(E.enc_ld(0, SL, LIM, 1), E.ERR_SPM_RANGE, SL, "LD SPM+DRAM")
    await err(with_base(E.enc_ld(0, 2, 0, 2), ra=1), E.ERR_SPM_RANGE, SL - 1, "LD reg SPM", setup)
    await case(with_base(E.enc_st(0, 0, 9, 2), rb=3), dram(LIM - 10 + 9, 2), "ST reg DRAM", setup)
    await case(with_base(E.enc_st(0, 0, 8, 2), rb=3), dram(LIM - 10 + 8, 2), "ST reg exact fit", setup)
    await err(with_base(E.enc_ld(0, 0, 0, 1), ra=2), E.ERR_SPM_RANGE, M32, "LD SPM 0xFFFFFFFF", setup)
    await case(with_base(E.enc_ld(0, 5, 0, 1), ra=2, rb=2), dram(M32, 1), "LD wrap SPM ok, DRAM 0xFFFFFFFF", setup)
    # QACT
    await err(E.enc_qact(0, 0, 100, 0, 1, 0), E.ERR_K_RANGE, 0, "QACT K=0")
    await err(E.enc_qact(0, 0, 100, 48, 1, 0), E.ERR_K_RANGE, 0, "QACT A32 K%32")
    await err(E.enc_qact(1, 0, 100, 128, 1, 0), E.ERR_K_RANGE, 0, "QACT A256 K%256")
    await err(E.enc_qact(0, 0, 100, KM + 32, 1, 0), E.ERR_K_RANGE, 0, "QACT K>K_MAX")
    await err(E.enc_qact(1, 0, 100, 0x80000000 + 256, 1, 0), E.ERR_K_RANGE, 0, "QACT K>=2^31")
    big = 2 * isa.qa_lines(1, KM) * T + 64
    if big + (T - 1) * (KM // 16) + KM // 16 <= SL:
        await ok(E.enc_qact(1, 0, big, KM, T, 0), "QACT K=K_MAX T=T_MAX")
    await err(E.enc_qact(0, 0, 100, 32, 0, 0), E.ERR_T_RANGE, 0, "QACT T=0")
    await err(E.enc_qact(0, 0, 100, 32, T + 1, 0), E.ERR_T_RANGE, 0, "QACT T>T_MAX")
    await err(E.enc_qact(0, 0, 100, 32, 0x80000001, 0), E.ERR_T_RANGE, 0, "QACT T=2^31+1")
    await err(E.enc_qact(0, SL - 1, 0, 32, 1, 0), E.ERR_SPM_RANGE, SL - 1, "QACT dst")
    await err(E.enc_qact(0, 0, SL - 1, 32, 1, 0), E.ERR_SPM_RANGE, SL - 1, "QACT src")
    if T >= 2:
        await err(E.enc_qact(0, 0, SL - 3, 32, 2, 3), E.ERR_SPM_RANGE, SL - 3, "QACT src stride")
        await ok(E.enc_qact(0, 0, SL - 4, 32, 2, 2), "QACT src stride exact fit")
        await ok(E.enc_qact(0, SL - 4, 0, 32, 2, 0), "QACT dst exact fit (2 x 2 lines)")
        await err(E.enc_qact(0, 0, 100, 32, 2, M32), E.ERR_SPM_RANGE, 100, "QACT (T-1)*stride >= 2^32")
        await err(E.enc_qact(1, 0, 100, 256, T, 0x80000000), E.ERR_SPM_RANGE, 100, "QACT (T-1)*2^31")
    await ok(E.enc_qact(1, SL - isa.qa_lines(1, 256), 0, 256, 1, 0), "QACT A256 dst exact fit")
    await err(E.enc_qact(1, SL - isa.qa_lines(1, 256) + 1, 0, 256, 1, 0), E.ERR_SPM_RANGE,
              SL - isa.qa_lines(1, 256) + 1, "QACT A256 dst over by one")
    await err(E.enc_qact(5, SL, SL, 33, 0, 0), E.ERR_ILLEGAL_OP, 0, "QACT kind+K+T+SPM")
    await err(E.enc_qact(0, SL, SL, 33, 0, 0), E.ERR_K_RANGE, 0, "QACT K+T+SPM")
    await err(E.enc_qact(0, SL, SL, 32, 0, 0), E.ERR_T_RANGE, 0, "QACT T+SPM")
    await err(E.enc_qact(0, SL, SL + 7, 32, 1, 0), E.ERR_SPM_RANGE, SL, "QACT dst before src")
    await err(with_base(E.enc_qact(0, 3, 0, 32, 1, 0), ra=1), E.ERR_SPM_RANGE, SL, "QACT reg dst", setup)
    if not h.has_mvm:
        sim.close()
        return
    # MVM (HAS_MVM = 1): N = 9, K = 64 (Q8_0) / 256 (Q4_K)
    winit = [(c, isa.REGION_WEIGHT_OFF, rnd.randbytes(64 * 256)) for c in range(h.num_ch)]
    await h.check(sim, prog_bytes([isa.enc_end()]), 0, init=winit)
    rb8 = ref().row_beats(0, 64)
    R9 = -(-9 // h.num_ch)
    T2 = min(2, T)
    qa8 = isa.qa_lines(0, 64)
    await ok(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, T2, 0), "MVM ok")
    await ok(enc_mvm(1, 1, 0, 100, WOFF, 9, 256, T2, 0), "MVM Q4_K ACC ok")
    await err(enc_mvm(0, 2, 0, 100, WOFF, 9, 64, T2, 0), E.ERR_ILLEGAL_OP, 0, "MVM flag bit1")
    await err(enc_mvm(3, 0, 0, 100, WOFF, 9, 64, T2, 0), E.ERR_BAD_FMT, 0, "MVM fmt 3")
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 0, T2, 0), E.ERR_K_RANGE, 0, "MVM K=0")
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 48, T2, 0), E.ERR_K_RANGE, 0, "MVM Q8_0 K%32")
    await err(enc_mvm(1, 0, 0, 100, WOFF, 9, 64, T2, 0), E.ERR_K_RANGE, 0, "MVM Q4_K K%256")
    if KM + 256 <= 0xFFFF:
        await err(enc_mvm(1, 0, 0, 100, WOFF, 9, KM + 256, T2, 0), E.ERR_K_RANGE, 0, "MVM K>K_MAX")
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 0, 0), E.ERR_T_RANGE, 0, "MVM T=0")
    if T < 15:
        await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, T + 1, 0), E.ERR_T_RANGE, 0, "MVM T>T_MAX")
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 15, 0), E.ERR_T_RANGE, 0, "MVM T=15")
    await err(enc_mvm(0, 0, SL - T2 + 1, 100, WOFF, 9, 64, T2, 0), E.ERR_SPM_RANGE, SL - T2 + 1, "MVM dst")
    await ok(enc_mvm(0, 0, SL - T2, 100, WOFF, 9, 64, T2, 0), "MVM dst exact fit")
    if T2 == 2:
        await err(enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 10), E.ERR_SPM_RANGE, SL - 10, "MVM dst stride")
        await ok(enc_mvm(0, 0, SL - 10, 100, WOFF, 9, 64, 2, 9), "MVM dst stride exact fit")
        await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 2, M32), E.ERR_SPM_RANGE, 0, "MVM dst stride 2^32-1")
    await err(enc_mvm(0, 0, 0, SL - T2 * qa8 + 1, WOFF, 9, 64, T2, 0), E.ERR_SPM_RANGE, SL - T2 * qa8 + 1, "MVM x")
    await ok(enc_mvm(0, 0, 0, SL - T2 * qa8, WOFF, 9, 64, T2, 0), "MVM x exact fit")
    e9, w = R9 * rb8, LIM - R9 * rb8
    await case(enc_mvm(0, 0, 0, 100, w + 1, 9, 64, T2, 0), dram(w + 1, e9), "MVM W_OFF")
    await case(enc_mvm(0, 0, 0, 100, w, 9, 64, T2, 0), dram(w, e9), "MVM W_OFF exact fit")
    rb4 = ref().row_beats(1, 512)            # Q4_K extent (differs from Q8_0's for the same K)
    e9, w = R9 * rb4, LIM - R9 * rb4
    await case(enc_mvm(1, 1, 0, 100, w + 1, 9, 512, T2, 0), dram(w + 1, e9), "MVM Q4_K W_OFF")
    await case(enc_mvm(1, 1, 0, 100, w, 9, 512, T2, 0), dram(w, e9), "MVM Q4_K W_OFF exact fit")
    R5 = -(-5 // h.num_ch)                   # N = 5: another rows-per-channel value
    e5, w = R5 * rb8, LIM - R5 * rb8
    await case(enc_mvm(0, 0, 0, 100, w + 1, 5, 64, 1, 0), dram(w + 1, e5), "MVM N=5 W_OFF")
    await case(enc_mvm(0, 0, 0, 100, w, 5, 64, 1, 0), dram(w, e5), "MVM N=5 W_OFF exact fit")
    await case(enc_mvm(0, 0, 0, 100, LIM + 1, 0, 64, 1, 0), dram(LIM + 1, 0), "MVM N=0 W_OFF past end")
    await case(enc_mvm(0, 0, 0, 100, LIM, 0, 64, 1, 0), dram(LIM, 0), "MVM N=0 W_OFF at the limit")
    await err(enc_mvm(0, 0, 0, 100, WOFF, M32, 64, 1, 0), E.ERR_SPM_RANGE, 0, "MVM huge N")
    await err(enc_mvm(7, 8, SL, SL, LIM, 9, 33, 0, 0), E.ERR_ILLEGAL_OP, 0, "MVM all")
    await err(enc_mvm(7, 0, SL, SL, LIM, 9, 33, 0, 0), E.ERR_BAD_FMT, 0, "MVM fmt+K+T+SPM+DRAM")
    await err(enc_mvm(0, 0, SL, SL, LIM, 9, 33, 0, 0), E.ERR_K_RANGE, 0, "MVM K+T+SPM+DRAM")
    await err(enc_mvm(0, 0, SL, SL, LIM, 9, 32, 0, 0), E.ERR_T_RANGE, 0, "MVM T+SPM+DRAM")
    await err(enc_mvm(0, 0, SL, SL + 1, LIM, 9, 32, 1, 0), E.ERR_SPM_RANGE, SL, "MVM dst first")
    await err(enc_mvm(0, 0, 0, SL + 1, LIM, 9, 32, 1, 0), E.ERR_SPM_RANGE, SL + 1, "MVM x before DRAM")
    await case(with_base(enc_mvm(0, 0, 0, 100, 10, 9, 64, 1, 0), rc=3), dram(LIM, R9 * rb8), "MVM reg W_OFF", setup)
    await err(with_base(enc_mvm(0, 1, 3, 100, WOFF, 9, 64, 1, 0), ra=1), E.ERR_SPM_RANGE, SL, "MVM reg dst (ACC)", setup)
    await ok(with_base(enc_mvm(0, 1, 2, 100, WOFF, 9, 64, 1, 0), ra=1), "MVM reg dst ACC fits", setup)
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 1, 0)[:7] + [1], E.ERR_ILLEGAL_OP, 0, "MVM w7")
    await err(enc_mvm(0, 0, 0, 100, WOFF, 9, 64, 1, 0)[:5] + [64 | (1 << 16) | (1 << 20), 0, 0],
              E.ERR_ILLEGAL_OP, 0, "MVM w5 reserved")
    # Large N (dst fills the SPM): the weight extent R * ROW_BEATS (serial divide / multiply)
    # decides; the fitting cases are not executed (too slow in Python).
    for fmt in (0, 1):
        K = KM - KM % 256
        ql = isa.qa_lines(fmt, K)
        N = 16 * (SL - ql)
        ext = -(-N // h.num_ch) * ref().row_beats(fmt, K)
        for woff in ((LIM - ext + 1) & M32, (LIM - ext + 77) & M32, M32 - 3, 1 << 31):
            if woff + ext <= LIM:
                continue
            await err(enc_mvm(fmt, 0, 0, SL - ql, woff, N, K, 1, 0), E.ERR_DRAM_RANGE, woff,
                      "MVM N=%d K=%d weights 0x%x beats @0x%x" % (N, K, ext, woff))
    sim.close()


@cocotb.test()
async def cp_perf(dut):
    """PERF_*: CYCLES counts BUSY cycles, RETIRED, per-channel read beats (incl. fetch) and wait
    cycles, MVM cycles; VU / WAITM read 0; not cleared by START, cleared by SOFT_RESET; 64-bit
    reads are tear-free (LO read snapshots HI)."""
    h = await Harness.create(dut)
    n = h.num_ch
    assert all(v == 0 for v in await h.csr.perf(n))
    # Event inputs: each counter counts exactly its events.
    pattern = []
    for i in range(64):
        rd_ = h.rnd.getrandbits(n)
        wt = h.rnd.getrandbits(n)
        mv = h.rnd.getrandbits(1)
        pattern.append((rd_, wt, mv))
    for rd_, wt, mv in pattern:
        await RisingEdge(dut.clk)
        dut.tb_perf_rd_beat.value = rd_
        dut.tb_perf_wait.value = wt
        dut.tb_perf_mvm_busy.value = mv
    await RisingEdge(dut.clk)
    dut.tb_perf_rd_beat.value = 0
    dut.tb_perf_wait.value = 0
    dut.tb_perf_mvm_busy.value = 0
    perf = await h.csr.perf(n)
    for c in range(n):
        assert perf[isa.perf_ch_rd_beats(c)] == sum((p[0] >> c) & 1 for p in pattern), ("rd", c)
        assert perf[isa.perf_ch_wait(n, c)] == sum((p[1] >> c) & 1 for p in pattern), ("wait", c)
    assert perf[isa.PERF_MVM_CYCLES] == sum(p[2] for p in pattern)
    assert perf[isa.PERF_VU_CYCLES] == 0 and perf[isa.PERF_WAITM_CYCLES] == 0
    assert perf[isa.PERF_CYCLES] == 0 and perf[isa.PERF_RETIRED] == 0
    # CYCLES == cycles with BUSY; RETIRED; fetch beats counted on ch0.
    base = await h.csr.perf(n)
    busy_cycles = 0
    p = prog_bytes([isa.enc_li(1, 1), isa.enc_ld(0, 0, IO_BEAT, 3), isa.enc_nop(), isa.enc_end()])

    async def count_busy():
        nonlocal busy_cycles
        while True:
            await FallingEdge(dut.clk)
            busy_cycles += h.busy()
    t = cocotb.start_soon(count_busy())
    beats0 = h.mem.stats()[0]["rd_beats"]
    out = await h.run(p, 0)
    t.cancel()
    perf = await h.csr.perf(n)
    assert out.retired == 4 and perf[isa.PERF_RETIRED] - base[isa.PERF_RETIRED] == 4
    assert perf[isa.PERF_CYCLES] - base[isa.PERF_CYCLES] == busy_cycles, (perf[0], base[0], busy_cycles)
    assert perf[isa.perf_ch_rd_beats(0)] - base[isa.perf_ch_rd_beats(0)] == h.mem.stats()[0]["rd_beats"] - beats0
    # Not cleared by START; cleared by SOFT_RESET.
    out = await h.run(prog_bytes([isa.enc_end()]), 0)
    assert await h.retired() == perf[isa.PERF_RETIRED] + 1
    await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
    assert all(v == 0 for v in await h.csr.perf(n))
    # Tear-free 64-bit read across a 2^32 carry: a LO read snapshots HI.
    idx = isa.perf_ch_wait(n, 0)
    dut.u_csr.perf_q[idx].value = (5 << 32) | (M32 - 40)
    await RisingEdge(dut.clk)
    dut.tb_perf_wait.value = 1
    lo = await h.rd(isa.perf_csr(idx))
    await ClockCycles(dut.clk, 60)                       # the counter carries into HI meanwhile
    hi = await h.rd(isa.perf_csr(idx) + 4)
    assert hi == 5 and lo > M32 - 40, (hex(hi), hex(lo))
    hi2 = await h.rd(isa.perf_csr(idx) + 4)               # snapshot consumed: live HI
    assert hi2 == 6, hex(hi2)
    v1 = await h.csr.read64(isa.perf_csr(idx))
    v2 = await h.csr.read64(isa.perf_csr(idx))
    assert (6 << 32) < v1 < v2, (hex(v1), hex(v2))
    await RisingEdge(dut.clk)
    dut.tb_perf_wait.value = 0
    await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
    assert await h.csr.read64(isa.perf_csr(idx)) == 0


@cocotb.test()
async def cp_throughput(dut):
    """Instruction overhead with an ideal memory and zero-latency units (PERF_CYCLES per run):
    NOP / LI / ADDI 4 cycles, LD / ST / QACT ~10 cycles + unit time; prefetch keeps up."""
    h = await Harness.create(dut, p_ready=1.0, delay=(0, 0),
                             mem_kw=dict(rd_lat=(1, 1), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0))
    n = 256
    res = {}
    for name, ins in (("NOP", isa.enc_nop()), ("LI", isa.enc_li(3, 7)), ("ADDI", isa.enc_addi(3, 3, 1)),
                      ("LD", isa.enc_ld(0, 0, IO_BEAT, 1)), ("ST n=0", isa.enc_st(0, 0, IO_BEAT, 0)),
                      ("QACT", isa.enc_qact(0, 0, 64, 32, 1, 0))):
        await h.wr(isa.CSR_CTRL, isa.CTRL_SOFT_RESET)
        out = await h.run(prog_bytes([ins] * n + [isa.enc_end()]), 0)
        assert out.final["status"] == DONE, (name, out.final)
        cyc = await h.csr.read64(isa.perf_csr(isa.PERF_CYCLES))
        res[name] = cyc / (n + 1)
    dut._log.info("cp_throughput: cycles per instruction %s", {k: round(v, 2) for k, v in res.items()})
    for name in ("NOP", "LI", "ADDI"):
        assert res[name] < 4.3, res
    for name in ("LD", "QACT"):
        assert res[name] < 11.5, res
    assert res["ST n=0"] < 8.3, res


@cocotb.test()
async def cp_random(dut):
    """Random programs (ProgGen) on the RTL and on the simulator; everything compared."""
    env = cp_env()
    h = await Harness.create(dut, p_ready=0.5, delay=(0, 15))
    sim = simlib.Sim(h.cfg)
    gen = ProgGen(random.Random(h.seed * 31 + 7), h.cfg, h.has_mvm)
    data = gen.init_data()
    await h.check(sim, prog_bytes([isa.enc_end()]), gen.cmd_base, init=data, what="load data")
    n = int(os.environ.get("TSR_CP_RANDOM", env["n_random"]))
    codes = {}
    retired = 0
    disp = Counter3()
    for i in range(n):
        prog, cmd, count, args = gen.program()
        out = await h.check(sim, prog, cmd, args, count=count, what="random program %d" % i)
        codes[out.final["err_code"]] = codes.get(out.final["err_code"], 0) + 1
        retired += out.retired
        disp += [len(u.cmds) for u in h.units]
    dut._log.info("cp_random: %d programs, %d instructions retired, dispatched DMA/QACT/MVM %s, "
                  "endings %s", n, retired, list(disp), {isa.ERR_NAMES[k]: v for k, v in sorted(codes.items())})
    for code in (0, isa.ERR_ILLEGAL_OP, isa.ERR_SPM_RANGE, isa.ERR_DRAM_RANGE, isa.ERR_K_RANGE, isa.ERR_T_RANGE):
        assert codes.get(code, 0) > 0, "random programs never ended in %s" % isa.ERR_NAMES[code]
    sim.close()


@cocotb.test()
async def cp_traces(dut):
    """traces/m2 sets of this configuration replayed through the block (units functional in Python)."""
    env = cp_env()
    h = await Harness.create(dut)
    keys = ("num_ch", "t_max", "k_max", "spm_bytes", "ch_bytes", "addr_beat_w")
    n = 0
    for s in env["traces"]:
        for tr in trace.load_all(TRACES / s):
            assert all(tr.config[k] == h.cfg[k] for k in keys), (tr.name, tr.config)
            h.new_mem()
            h.mem.load_trace(tr)
            out = await h.run(b"", tr.cmd_addr_beat, tr.args, count=tr.cmd_count)
            errs = replay.compare(tr, h.mem, out.final)
            assert not errs, "%s/%s: %s" % (s, tr.name, errs)
            n += 1
    dut._log.info("cp_traces: %d traces replayed", n)
    assert n > 0 or not env["traces"]

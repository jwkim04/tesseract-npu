"""Shared testbench components for tb/dma (tsr_dma, tsr_mem_arb, tb/dma/hdl/tsr_tb_dma.sv).

clock_reset()   start a pausable clock (cocotb Clock, kept so it can be stopped) and reset
comb_probe()    stop the clock in the low phase and wiggle every input: no output may change
                (rules R5/M8 of docs/rtl-m2.md: every output is a flop; no input->output path)
SpmModel        Python model of one tsr_spm client port (docs/rtl-m2.md §5): random ready, read
                data exactly RD_LAT cycles after acceptance (or later, still in order, with
                var_lat), read-before-write in one cycle, R2 checks on the client's requests
ReadSources     random read bursts on a group of read-source ports packed into one signal set
                (tsr_mem_arb s_rd_* sources, or the wrapper's f_/w_ background sources); checks
                data, per-id order, `last`, response ids
WriteSource     random write bursts on a write-source port (request and data offered
                independently with random gaps); checks write responses; keeps a shadow
BurstMonitor    log of request handshakes on a memory-style port (packed per channel)
PerfMonitor     counts perf_rd_beat / perf_rd_wait pulses per channel
SpmContender    wrapper with RTL_SPM = 1: a second tsr_spm write client (x_wr_*, client QACT, lower
                priority than the DMA) that rewrites lines with their current content, in bursts,
                outside an excluded line range, to force write-grant handovers during DMA ops
LdReadyMonitor  wrapper with RTL_SPM = 1: while an LD runs, the DMA's rd_rsp_ready must stay 1
                (D2 with tsr_spm: the DMA is SPM write client 0, so its SPM write stalls are short
                enough never to fill its beat queue); counts the DMA's SPM write stalls and the
                contender's writes during LDs
DmaDriver       dispatch of tsr_dma_cmd_t commands with cmd_ready/done protocol checks
split()         the D1 burst split: len = min(left, 64 - addr % 64)

Timing discipline (as tb/common): drive right after a rising edge, sample at the falling edge.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict, deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import isa

LINE_MASK = (1 << 512) - 1


def cfg_from_env() -> dict:
    return json.loads(os.environ["TSR_DMA_CFG"])


def split(addr: int, n: int):
    """docs/rtl-m2.md D1: bursts of <= 64 beats that never cross a 64-beat (4 KiB) block."""
    out = []
    while n:
        ln = min(n, isa.BURST_ALIGN - addr % isa.BURST_ALIGN)
        out.append((addr, ln))
        addr += ln
        n -= ln
    return out


def fld(v: int, i: int, w: int) -> int:
    return (v >> (i * w)) & ((1 << w) - 1)


class _Out:
    """Write-on-change cache for TB-driven signals (fewer simulator writes)."""

    def __init__(self):
        self.last = {}

    def put(self, sig, val):
        k = id(sig)
        if self.last.get(k) != val:
            sig.value = val
            self.last[k] = val

    def clear(self):
        self.last.clear()


# =============================================================================================
# Clock / reset / combinational-path probe
# =============================================================================================
async def clock_reset(dut, cycles: int = 5, period_ns: int = 10) -> Clock:
    """rst_n is low before the first clock edge (the initial state of a Verilator build with
    X randomisation is random, including rst_n)."""
    dut.rst_n.value = 0
    dut.clk.value = 0
    await Timer(1, "ns")
    clk = Clock(dut.clk, period_ns, unit="ns")
    clk.start(start_high=True)
    await ClockCycles(dut.clk, cycles)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    return clk


async def comb_probe(dut, clock: Clock, inputs, outputs, rnd: random.Random, tries: int = 2):
    """Stop the clock right after a falling edge, then set every input (one at a time) to random
    values and check that no output changes; restore the inputs and restart the clock (the next
    event is a rising edge). Without a clock edge only combinational logic can react, so any
    output change is an input->output combinational path. Returns the number of pokes."""
    await FallingEdge(dut.clk)
    clock.stop()
    await Timer(1, "ns")
    outs = [(n, getattr(dut, n)) for n in outputs]
    snap = {n: s.value for n, s in outs}
    bad = set()
    pokes = 0
    for name in inputs:
        sig = getattr(dut, name)
        saved = sig.value
        for _ in range(tries):
            sig.value = rnd.getrandbits(len(sig))
            pokes += 1
            await Timer(1, "ns")
            for n, s in outs:
                if s.value != snap[n]:
                    bad.add((name, n))
        sig.value = saved
        await Timer(1, "ns")
    for n, s in outs:
        assert s.value == snap[n], "output %s did not return to its value after the probe" % n
    clock.start(start_high=True)
    assert not bad, "combinational input->output paths: %s" % sorted(bad)
    return pokes


# =============================================================================================
# SPM client model
# =============================================================================================
class SpmModel:
    def __init__(self, dut, *, lines: int, rd_lat: int, rnd: random.Random, prefix: str = "spm_",
                 p_rd_ready: float = 0.85, p_wr_ready: float = 0.85, var_lat: int = 0,
                 p_stall: float = 0.01, stall_len=(1, 20)):
        self.dut, self.lines, self.rd_lat, self.rnd = dut, lines, rd_lat, rnd
        self.p_rd_ready, self.p_wr_ready, self.var_lat = p_rd_ready, p_wr_ready, var_lat
        self.p_stall, self.stall_len = p_stall, stall_len
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.rq_v, self.rq_r, self.rq_l = g("rd_req_valid"), g("rd_req_ready"), g("rd_req_line")
        self.rs_v, self.rs_d = g("rd_rsp_valid"), g("rd_rsp_data")
        self.wr_v, self.wr_r, self.wr_l = g("wr_valid"), g("wr_ready"), g("wr_line")
        self.wr_d, self.wr_b = g("wr_data"), g("wr_be")
        self.mem = {}
        self.reads, self.writes = [], []
        self.errors = []
        self.cycle = 0
        self._o = _Out()
        self._task = None
        self._seed = rnd.getrandbits(32)
        self._idle()

    def line(self, i: int) -> int:
        """Current content of line i (undefined lines get a deterministic random value)."""
        if i not in self.mem:
            self.mem[i] = random.Random((self._seed << 24) ^ i).getrandbits(512)
        return self.mem[i]

    def start(self):
        if self._task is None:
            self._task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def clear_log(self):
        self.reads.clear()
        self.writes.clear()

    def _fail(self, msg):
        msg = "spm cycle %d: %s" % (self.cycle, msg)
        self.errors.append(msg)
        raise AssertionError(msg)

    def _idle(self):
        self._o.clear()
        for s in (self.rq_r, self.rs_v, self.rs_d, self.wr_r):
            s.value = 0

    async def _run(self):
        clk, rst = self.dut.clk, self.dut.rst_n
        rd_rdy = wr_rdy = 0
        pend_rq = pend_wr = None
        pipe = deque()
        last_due = -1
        stall = 0
        while True:
            await FallingEdge(clk)
            in_rst = int(rst.value) == 0
            rqv = int(self.rq_v.value)
            rql = int(self.rq_l.value) if rqv else None
            wv = int(self.wr_v.value)
            wp = (int(self.wr_l.value), int(self.wr_d.value), int(self.wr_b.value)) if wv else None
            await RisingEdge(clk)
            self.cycle += 1
            if in_rst:
                pipe.clear()
                pend_rq = pend_wr = None
                rd_rdy = wr_rdy = 0
                self._idle()
                continue
            # ---- read request handshake (R2 checks while not accepted) ----
            if rqv:
                if pend_rq is not None and pend_rq != rql:
                    self._fail("rd_req_line changed before the handshake: %d -> %d" % (pend_rq, rql))
                if rd_rdy:
                    pend_rq = None
                    if rql >= self.lines:
                        self._fail("read of line %d >= %d lines" % (rql, self.lines))
                    due = self.cycle + self.rd_lat - 1
                    if self.var_lat:
                        due += self.rnd.randint(0, self.var_lat)
                    due = max(due, last_due + 1)
                    last_due = due
                    pipe.append((due, self.line(rql)))   # data as of acceptance (read-first)
                    self.reads.append(rql)
                else:
                    pend_rq = rql
            elif pend_rq is not None:
                self._fail("rd_req_valid dropped before the handshake")
            # ---- write handshake ----
            if wv:
                if pend_wr is not None and pend_wr != wp:
                    self._fail("write payload changed before the handshake (line %d)" % wp[0])
                if wr_rdy:
                    pend_wr = None
                    wl, wd, wb = wp
                    if wl >= self.lines:
                        self._fail("write of line %d >= %d lines" % (wl, self.lines))
                    if wb != (1 << 64) - 1:
                        self._fail("write of line %d with byte enables 0x%x (not all ones)" % (wl, wb))
                    self.mem[wl] = wd
                    self.writes.append(wl)
                else:
                    pend_wr = wp
            elif pend_wr is not None:
                self._fail("wr_valid dropped before the handshake")
            # ---- drive: read response (no backpressure), readies ----
            if pipe and pipe[0][0] == self.cycle:
                _, data = pipe.popleft()
                self._o.put(self.rs_v, 1)
                self._o.put(self.rs_d, data)
            else:
                self._o.put(self.rs_v, 0)
            if stall > 0:
                stall -= 1
            elif self.rnd.random() < self.p_stall:
                stall = self.rnd.randint(*self.stall_len)
            rd_rdy = int(stall == 0 and self.rnd.random() < self.p_rd_ready)
            wr_rdy = int(stall == 0 and self.rnd.random() < self.p_wr_ready)
            self._o.put(self.rq_r, rd_rdy)
            self._o.put(self.wr_r, wr_rdy)


# =============================================================================================
# Read sources (arbiter sources / background traffic)
# =============================================================================================
class ReadSources:
    """`lanes` read-source ports packed into one signal set <prefix>rd_req_* / <prefix>rd_rsp_*.

    shared_rsp: rd_rsp_data/last/id are one shared vector qualified by the per-lane valid
    (tsr_mem_arb source side) instead of per-lane fields (the wrapper's w_* group).
    expect(lane, beat) -> expected 512-bit data. Requests come from .plan[lane] (explicit) or,
    while .auto is set, are generated at random inside .region = (lo, hi) of beats."""

    def __init__(self, dut, *, prefix: str, lanes: int, aw: int, rnd: random.Random, expect,
                 region, shared_rsp: bool, id_w: int = isa.SRC_ID_W, p_req: float = 0.4,
                 p_rsp_ready: float = 0.8, lens=None, name: str = "src"):
        self.dut, self.lanes, self.aw, self.rnd, self.expect = dut, lanes, aw, rnd, expect
        self.region, self.shared, self.id_w = region, shared_rsp, id_w
        self.p_req, self.p_rsp_ready = p_req, p_rsp_ready
        self.lens = lens or [1, 1, 2, 3, 8, 16, 31, 32, 63, 64]
        self.name = name
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.rq_v, self.rq_r = g("rd_req_valid"), g("rd_req_ready")
        self.rq_a, self.rq_l, self.rq_i = g("rd_req_addr"), g("rd_req_len"), g("rd_req_id")
        self.rs_v, self.rs_r = g("rd_rsp_valid"), g("rd_rsp_ready")
        self.rs_d, self.rs_l, self.rs_i = g("rd_rsp_data"), g("rd_rsp_last"), g("rd_rsp_id")
        self.plan = [deque() for _ in range(lanes)]
        self.auto = False
        self.p_req_lane = {}                    # lane -> p_req override (auto mode)
        self.p_rsp_lane = {}                    # lane -> p_rsp_ready override
        self.cur = [None] * lanes
        self.out = [defaultdict(deque) for _ in range(lanes)]
        self.n_out = [0] * lanes
        self.beats = [0] * lanes
        self.grants = [0] * lanes
        self.grant_log = []                     # (cycle, lane)
        self.cycle = 0
        self._o = _Out()
        self._task = None
        for s in (self.rq_v, self.rq_a, self.rq_l, self.rq_i, self.rs_r):
            s.value = 0

    def start(self):
        if self._task is None:
            self._task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def busy(self) -> bool:
        return any(self.plan[k] or self.cur[k] is not None or self.n_out[k] for k in range(self.lanes))

    def _reset_state(self):
        """DUT reset: everything in flight is gone (requests, outstanding reads, the plan)."""
        for k in range(self.lanes):
            self.plan[k].clear()
            self.cur[k] = None
            self.out[k].clear()
            self.n_out[k] = 0

    async def drain(self, timeout: int = 200000):
        self.auto = False
        for _ in range(timeout):
            if not self.busy():
                return
            await RisingEdge(self.dut.clk)
        raise TimeoutError("%s: outstanding reads did not complete" % self.name)

    def rand_req(self):
        """Random legal burst inside .region (1..64 beats, inside one 64-beat block)."""
        lo, hi = self.region
        for _ in range(100):
            ln = min(self.rnd.choice(self.lens), hi - lo, 64)
            a = self.rnd.randrange(lo, hi - ln + 1)
            if a % 64 + ln > 64:
                a = a - a % 64 + 64 - ln          # slide back inside the block
            if a >= lo:
                return a, ln, self.rnd.randrange(1 << self.id_w)
        return lo, 1, 0

    async def _run(self):
        d = self.dut
        n, aw, iw = self.lanes, self.aw, self.id_w
        in_rst = False
        while True:
            # ---- drive (right after the rising edge) ----
            v = a = ln = i = r = 0
            rdy = [0] * n
            for k in range(n):
                if in_rst:
                    continue
                if self.cur[k] is None:
                    if self.plan[k]:
                        self.cur[k] = self.plan[k].popleft()
                    elif self.auto and self.rnd.random() < self.p_req_lane.get(k, self.p_req):
                        self.cur[k] = self.rand_req()
                if self.cur[k] is not None:
                    ca, cl, ci = self.cur[k]
                    v |= 1 << k
                    a |= ca << (k * aw)
                    ln |= cl << (k * isa.LEN_W)
                    i |= ci << (k * iw)
                rdy[k] = int(self.rnd.random() < self.p_rsp_lane.get(k, self.p_rsp_ready))
                r |= rdy[k] << k
            self._o.put(self.rq_v, v)
            if v:
                self._o.put(self.rq_a, a)
                self._o.put(self.rq_l, ln)
                self._o.put(self.rq_i, i)
            self._o.put(self.rs_r, r)
            # ---- sample (falling edge) ----
            await FallingEdge(d.clk)
            in_rst = int(d.rst_n.value) == 0
            s_rqr = int(self.rq_r.value)
            s_rsv = int(self.rs_v.value)
            if s_rsv:
                s_rsd, s_rsl, s_rsi = int(self.rs_d.value), int(self.rs_l.value), int(self.rs_i.value)
            await RisingEdge(d.clk)
            self.cycle += 1
            if in_rst:
                self._reset_state()
                continue
            # ---- update ----
            for k in range(n):
                if self.cur[k] is not None and (s_rqr >> k) & 1:
                    ca, cl, ci = self.cur[k]
                    self.out[k][ci].append([ca, cl, 0])
                    self.n_out[k] += 1
                    self.grants[k] += 1
                    self.grant_log.append((self.cycle, k))
                    self.cur[k] = None
                if (s_rsv >> k) & 1 and rdy[k]:
                    if self.shared:
                        data, last, rid = s_rsd, s_rsl & 1, s_rsi
                    else:
                        data, last, rid = fld(s_rsd, k, 512), (s_rsl >> k) & 1, fld(s_rsi, k, iw)
                    q = self.out[k].get(rid)
                    assert q, "%s lane %d: read beat for id %d with nothing outstanding" % (self.name, k, rid)
                    b = q[0]
                    beat = b[0] + b[2]
                    exp = self.expect(k, beat)
                    assert data == exp, "%s lane %d id %d beat 0x%x: data mismatch" % (self.name, k, rid, beat)
                    b[2] += 1
                    assert last == int(b[2] == b[1]), "%s lane %d: rd_rsp_last=%d on beat %d of %d" % (
                        self.name, k, last, b[2], b[1])
                    self.beats[k] += 1
                    if b[2] == b[1]:
                        q.popleft()
                        self.n_out[k] -= 1


# =============================================================================================
# Write source (arbiter test)
# =============================================================================================
class WriteSource:
    """Random write bursts on <prefix>wr_req_* / wr_dat_* / wr_rsp_* (one port). Request and data
    channels are driven independently with random gaps. shadow: beat -> last written data."""

    def __init__(self, dut, *, prefix: str, aw: int, rnd: random.Random, region, p_req: float = 0.5,
                 p_dat: float = 0.7, p_rsp_ready: float = 0.7, name: str = "wr"):
        self.dut, self.aw, self.rnd, self.region = dut, aw, rnd, region
        self.p_req, self.p_dat, self.p_rsp_ready, self.name = p_req, p_dat, p_rsp_ready, name
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.wq_v, self.wq_r = g("wr_req_valid"), g("wr_req_ready")
        self.wq_a, self.wq_l, self.wq_i = g("wr_req_addr"), g("wr_req_len"), g("wr_req_id")
        self.wd_v, self.wd_r, self.wd_d, self.wd_l = g("wr_dat_valid"), g("wr_dat_ready"), g("wr_dat_data"), g("wr_dat_last")
        self.ws_v, self.ws_r, self.ws_i = g("wr_rsp_valid"), g("wr_rsp_ready"), g("wr_rsp_id")
        self.reqs, self.dats = deque(), deque()
        self.pend = defaultdict(int)
        self.n_pend = 0
        self.shadow = {}
        self.bursts = 0
        self._o = _Out()
        self._task = None
        for s in (self.wq_v, self.wq_a, self.wq_l, self.wq_i, self.wd_v, self.wd_d, self.wd_l, self.ws_r):
            s.value = 0

    def start(self):
        if self._task is None:
            self._task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def plan_random(self, count: int):
        lo, hi = self.region
        for _ in range(count):
            ln = self.rnd.choice([1, 2, 5, 16, 33, 64, self.rnd.randint(1, 64)])
            blk = self.rnd.randrange(lo // 64, hi // 64)
            off = self.rnd.randint(0, 64 - ln)
            self.plan(blk * 64 + off, ln, self.rnd.randrange(4))

    def plan(self, addr: int, ln: int, id_: int):
        beats = [self.rnd.getrandbits(512) for _ in range(ln)]
        self.reqs.append((addr, ln, id_))
        for k, b in enumerate(beats):
            self.dats.append((b, int(k == ln - 1)))
            self.shadow[addr + k] = b

    def busy(self) -> bool:
        return bool(self.reqs or self.dats or self.n_pend)

    async def drain(self, timeout: int = 200000):
        for _ in range(timeout):
            if not self.busy():
                return
            await RisingEdge(self.dut.clk)
        raise TimeoutError("%s: writes did not complete" % self.name)

    def _reset_state(self):
        self.reqs.clear()
        self.dats.clear()
        self.pend.clear()
        self.n_pend = 0
        self.shadow.clear()

    async def _run(self):
        d = self.dut
        cur_q = cur_d = None
        in_rst = False
        while True:
            if in_rst:
                cur_q = cur_d = None
            if cur_q is None and self.reqs and self.rnd.random() < self.p_req:
                cur_q = self.reqs.popleft()
            if cur_d is None and self.dats and self.rnd.random() < self.p_dat:
                cur_d = self.dats.popleft()
            self._o.put(self.wq_v, int(cur_q is not None))
            if cur_q is not None:
                self._o.put(self.wq_a, cur_q[0])
                self._o.put(self.wq_l, cur_q[1])
                self._o.put(self.wq_i, cur_q[2])
            self._o.put(self.wd_v, int(cur_d is not None))
            if cur_d is not None:
                self._o.put(self.wd_d, cur_d[0])
                self._o.put(self.wd_l, cur_d[1])
            rr = int(self.rnd.random() < self.p_rsp_ready)
            self._o.put(self.ws_r, rr)
            await FallingEdge(d.clk)
            in_rst = int(d.rst_n.value) == 0
            qr, dr = int(self.wq_r.value), int(self.wd_r.value)
            sv = int(self.ws_v.value)
            si = int(self.ws_i.value) if sv else 0
            await RisingEdge(d.clk)
            if in_rst:
                self._reset_state()
                self._o.put(self.wq_v, 0)
                self._o.put(self.wd_v, 0)
                continue
            if cur_q is not None and qr:
                self.pend[cur_q[2]] += 1
                self.n_pend += 1
                self.bursts += 1
                cur_q = None
            if cur_d is not None and dr:
                cur_d = None
            if sv and rr:
                assert self.pend[si] > 0, "%s: write response id %d with nothing outstanding" % (self.name, si)
                self.pend[si] -= 1
                self.n_pend -= 1


# =============================================================================================
# Monitors
# =============================================================================================
class BurstMonitor:
    """Request handshakes on <prefix>{rd,wr}_req_* (packed per channel): .log entries
    (cycle, kind, ch, addr, len, id)."""

    def __init__(self, dut, *, prefix: str, num_ch: int, aw: int, id_w: int):
        self.dut, self.n, self.aw, self.iw = dut, num_ch, aw, id_w
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.sig = {k: (g(k + "_req_valid"), g(k + "_req_ready"), g(k + "_req_addr"), g(k + "_req_len"),
                        g(k + "_req_id")) for k in ("rd", "wr")}
        self.log = []
        self.cycle = 0
        self._task = cocotb.start_soon(self._run())

    def stop(self):
        self._task.cancel()

    async def _run(self):
        while True:
            await FallingEdge(self.dut.clk)
            self.cycle += 1
            for kind, (v, r, a, ln, i) in self.sig.items():
                hs = int(v.value) & int(r.value)
                if hs:
                    av, lv, iv = int(a.value), int(ln.value), int(i.value)
                    for c in range(self.n):
                        if (hs >> c) & 1:
                            self.log.append((self.cycle, kind, c, fld(av, c, self.aw), fld(lv, c, isa.LEN_W),
                                             fld(iv, c, self.iw)))


class PerfMonitor:
    def __init__(self, dut, num_ch: int):
        self.dut, self.n = dut, num_ch
        self.beats = [0] * num_ch
        self.wait = [0] * num_ch
        self._task = cocotb.start_soon(self._run())

    def stop(self):
        self._task.cancel()

    async def _run(self):
        while True:
            await FallingEdge(self.dut.clk)
            b, w = int(self.dut.perf_rd_beat.value), int(self.dut.perf_rd_wait.value)
            assert not (b & w), "perf_rd_beat and perf_rd_wait both set (0x%x)" % (b & w)
            for c in range(self.n):
                self.beats[c] += (b >> c) & 1
                self.wait[c] += (w >> c) & 1


# =============================================================================================
# Real-SPM integration helpers (wrapper with RTL_SPM = 1)
# =============================================================================================
class SpmContender:
    """Writes on the wrapper's x_wr_* port (tsr_spm client TSR_SPM_CL_QACT). Every write stores
    content(line), which must be the line's current content (None: line unknown, skipped), so the
    SPM contents never change; lines in .excl (a range being written by an LD) are never written.
    Valid comes in on/off phases of random length (many grant handovers). R2: a write stays valid
    with an unchanged payload until accepted. Use window() to change .excl: it stops issuing and
    waits until the pending write (if any) has been accepted, so no write chosen under the old
    window can land later."""

    def __init__(self, dut, *, lines: int, rnd: random.Random, content, on_len=(1, 12), off_len=(0, 8)):
        self.dut, self.lines, self.rnd, self.content = dut, lines, rnd, content
        self.on_len, self.off_len = on_len, off_len
        self.v, self.r, self.l, self.d = dut.x_wr_valid, dut.x_wr_ready, dut.x_wr_line, dut.x_wr_data
        self.enabled = False
        self.excl = range(0)
        self.pending = None
        self.fires = 0
        self._phase = 0                          # > 0: on cycles left, < 0: off cycles left
        self._task = None
        self.v.value = 0
        self.l.value = 0
        self.d.value = 0

    def start(self):
        if self._task is None:
            self._task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def window(self, excl, enabled: bool = True, timeout: int = 1000):
        self.enabled = False
        for _ in range(timeout):
            if self.pending is None:
                break
            await RisingEdge(self.dut.clk)
        else:
            raise TimeoutError("SPM contender: pending write to line %d never accepted" % self.pending)
        self.excl = excl
        self.enabled = enabled

    def _pick(self):
        for _ in range(8):
            ln = self.rnd.randrange(self.lines)
            if ln not in self.excl:
                c = self.content(ln)
                if c is not None:
                    return ln, c
        return None

    async def _run(self):
        clk, rst = self.dut.clk, self.dut.rst_n
        while True:
            await FallingEdge(clk)
            in_rst = int(rst.value) == 0
            acc = self.pending is not None and int(self.r.value) == 1
            await RisingEdge(clk)
            if in_rst:
                self.pending = None
                self.v.value = 0
                continue
            if acc:
                self.fires += 1
                self.pending = None
            if self.pending is not None:
                continue                         # R2: hold
            if self._phase == 0:
                self._phase = self.rnd.randint(*self.on_len) if self.rnd.random() < 0.5 \
                    else -self.rnd.randint(*self.off_len)
            on = self._phase > 0
            self._phase += -1 if self._phase > 0 else 1
            w = self._pick() if (self.enabled and on) else None
            if w is None:
                self.v.value = 0
                continue
            self.pending = w[0]
            self.l.value = w[0]
            self.d.value = w[1]
            self.v.value = 1


class LdReadyMonitor:
    """While .ld_active (set by the test around an LD): the DMA's rd_rsp_ready (u_dma port) must
    be all ones. Counts LD cycles, the DMA's SPM write stalls (spm_wr_valid && !spm_wr_ready) and
    the contender's accepted writes during LDs, and records the highest beat-queue occupancy
    (u_dma.q_cnt_q, information only)."""

    def __init__(self, dut, num_ch: int):
        self.dut, self.full = dut, (1 << num_ch) - 1
        self.ld_active = False
        self.cycles = self.dma_wr_stalls = self.x_writes = 0
        self.q_max = 0                           # highest beat-queue occupancy seen during LDs (info)
        self._task = cocotb.start_soon(self._run())

    def stop(self):
        self._task.cancel()

    async def _run(self):
        d, u = self.dut, self.dut.u_dma
        while True:
            await FallingEdge(d.clk)
            if not self.ld_active or int(d.rst_n.value) == 0:
                continue
            self.cycles += 1
            rr = int(u.rd_rsp_ready.value)
            assert rr == self.full, "LD in progress: DMA rd_rsp_ready = 0x%x (beat queue full: the SPM " \
                "write port stalled the DMA too long)" % rr
            if int(u.spm_wr_valid.value) and not int(u.spm_wr_ready.value):
                self.dma_wr_stalls += 1
            self.q_max = max(self.q_max, int(u.q_cnt_q.value))
            if int(d.x_wr_valid.value) and int(d.x_wr_ready.value):
                self.x_writes += 1


# =============================================================================================
# DMA command driver
# =============================================================================================
class DmaDriver:
    """Drives cmd_valid/cmd and checks the dispatch protocol (docs/rtl-m2.md §3.0):
    cmd_ready = idle; one `done` pulse per command, never in the accepting cycle, with cmd_ready
    back high in the done cycle; cmd_ready low while busy."""

    def __init__(self, dut, rnd: random.Random):
        self.dut, self.rnd = dut, rnd
        dut.cmd_valid.value = 0
        dut.cmd.value = 0
        self.busy = False
        self.dones = 0
        self.cycles = 0
        self._task = cocotb.start_soon(self._mon())

    def stop(self):
        self._task.cancel()

    def reset(self):
        """After a DUT reset: nothing in flight."""
        self.dut.cmd_valid.value = 0
        self.busy = False

    async def _mon(self):
        prev = 0
        while True:
            await FallingEdge(self.dut.clk)
            dn = int(self.dut.done.value)
            if dn:
                assert prev == 0, "done held for more than one cycle"
                assert self.busy, "done without a command in flight"
                assert int(self.dut.cmd_ready.value) == 1, "cmd_ready not back with done"
                self.dones += 1
            elif self.busy:
                assert int(self.dut.cmd_ready.value) == 0, "cmd_ready high while busy"
            prev = dn

    async def op(self, is_st: int, ch: int, spm_line: int, dram_beat: int, n: int, gap: int = 0,
                 timeout: int | None = None, on_done=None) -> int:
        """Dispatch one command, wait for done. Returns right after the edge that ends the done
        cycle (a next op can be offered immediately: back-to-back as cmdproc can do it).
        on_done() runs in the done cycle (falling edge)."""
        d = self.dut
        for _ in range(gap):
            await RisingEdge(d.clk)
        d.cmd.value = isa.DMA_CMD_T.pack(is_st=is_st, ch=ch, spm_line=spm_line, dram_beat=dram_beat,
                                         n_lines=n)
        d.cmd_valid.value = 1
        while True:
            await FallingEdge(d.clk)
            rdy = int(d.cmd_ready.value)
            assert int(d.done.value) == 0, "done in the cycle of the accepting handshake"
            await RisingEdge(d.clk)
            if rdy:
                break
        self.busy = True
        d.cmd_valid.value = 0
        d.cmd.value = self.rnd.getrandbits(isa.DMA_CMD_T.width)    # the DMA must have latched it
        timeout = timeout if timeout is not None else 60 * n + 20000
        cyc = 0
        while True:
            await FallingEdge(d.clk)
            if int(d.done.value):
                break
            cyc += 1
            if cyc > timeout:
                raise TimeoutError("DMA op (st=%d ch=%d spm=%d dram=0x%x n=%d) did not finish in %d cycles"
                                   % (is_st, ch, spm_line, dram_beat, n, timeout))
        if on_done is not None:
            on_done()
        await RisingEdge(d.clk)
        self.busy = False
        self.cycles += cyc + 1
        return cyc + 1

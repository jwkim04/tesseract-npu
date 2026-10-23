"""tsr_mvm_front / tsr_mvm_ws tests (docs/rtl-m3.md §3.3, §4, §10.2 "front").

DUT: tb/mvmfront/hdl/tsr_tb_mvm_front.sv = tsr_mvm_front on read source WS (id {2, 0}) of a real
tsr_mem_arb, whose channel memory port is served by common.mem_model.MemModel (random latency,
backpressure, stall phases, protocol assertions: burst length / 4 KiB blocks / channel limit,
RD_OUTSTANDING, payload stability). The arbiter's DMA and write sources are tied off; FETCH is
exposed and either idle or driven by a background reader (BgFetch: random bursts, random
response ready -> head-of-line blocking of the shared response channel).

Golden model: tb/common/mvm_model.py front_beats() (itself verified against libtsr_ref in
tb/common: every term recomputed from the beat stream equals tsr_term_*). The commands are those
of a cluster: fcmd(fmt, K, N, num_ch, c, w_off) of mvm_model (rows_c = 0 channels get none).

Checks on every command:
  - EVERY tagged beat (the whole 1175-bit tsr_mvm_beat_t) equals pack_beat(front_beats(...)), in
    order; on a mismatch the differing fields are reported (unpacked lanes incl. Q8_0 -128 and Q4_K
    nibbles in ggml order, d_a / d_b f32 bits, sc / m from get_scale_min_k4 — also checked directly
    against libtsr_ref's tsr_get_scale_min_k4 —, blk, two, half, row_last, last);
  - the WS source requests: exactly the bursts of rows * ROW_BEATS beats from w_off (BurstQueue:
    len = min(64 - addr % 64, left), address order, id 0); MemModel.src_beats: WS beats = sum of
    rows * ROW_BEATS, DMA 0, FETCH = background beats; rd_rsp_ready of the WS source is 1 whenever
    a WS beat is offered;
  - R2 on the beat output (valid held, payload stable while not ready);
  - cmd_ready == idle; cmd_ready = 1 only when every beat of the accepted commands was delivered
    and no WS read is in flight at the memory (DONE => quiescent);
  - simulation assertions in the RTL (tsr_mvm_ws / tsr_mvm_front / tsr_mem_arb) fail the test.

cocotb tests (each configuration runs all of them):
  front_directed     K = block size, partial chunks (Q8_0 odd / even remainders, Q4_K rem 1..3),
                     full chunks, K_MAX; rows 1..3; W_OFF at 4 KiB block edges and ending exactly at
                     the DRAM limit; data: raw random bytes (also in unused meta / tails), repacked
                     rows with special fp16 d, all-0x80 / 0x7F / 0xFF quant beats, 0xFF scales
  front_random       random fmt / K / N / NUM_CH / channel / W_OFF (block edges, limit, 2^32
                     crossing), random memory regimes, random beat_ready, background FETCH traffic
  front_fp16_all     (u200 only) every fp16 pattern as a Q8_0 d: d_a / d_b == tsr_fp16_to_f32
  front_throughput   ideal memory, beat_ready = 1: cycles per weight beat (META beats included)
                     for both formats across chunk / row / command boundaries; 1.000 required in
                     the configurations with enough FIFO / outstanding budget; first-beat latency
  front_fifo_sizing  (u200) fixed memory latency 400 -> still 1 beat/cycle; 520 -> stalls
                     (proves the measurement discriminates; docs/rtl-m3.md §4 FIFO sizing)
  front_reset        synchronous reset in the middle of a command: reset values of the outputs,
                     then fully checked commands
  front_max_count    the largest weight-beat count of a legal MVM ending at the DRAM limit (wide:
                     2^40 - 1 = every bit of tsr_mvm_fcmd_t.beats): complete in small / tiny,
                     prefix-checked (bursts, first rows, FIFO fill) then reset in u200 / wide
  front_comb_probe   no combinational input -> output path (clock stopped, inputs wiggled)
Configurations: u200 (K_MAX 32768, ADDR_BEAT_W 28, FIFO 512, RAM latency 2, RD_OUTSTANDING 16),
small (4096, 13, FIFO 64, latency 1, RD_OUTSTANDING 1), wide (65280, 40 = the maximum: streams
crossing beat 2^32, DRAM limit 2^40, FIFO 1024, latency 3, RD_OUTSTANDING 3), tiny (256, 5: a
32-beat channel, FIFO 64, latency 4, RD_OUTSTANDING 2).
Environment: TSR_FRONT_TESTS=a,b (subset of the cocotb tests), TSR_FRONT_SCALE (random test size
factor, default 1.0).
"""
from __future__ import annotations

import json
import os
import random
from collections import deque

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import isa
from common import mvm_model as mm
from common.mem_model import MemModel
from common.runner import REPO, run
from common.tbutil import seed as tb_seed
from common.tsr_ref import ref

TB_SV = REPO / "tb" / "mvmfront" / "hdl" / "tsr_tb_mvm_front.sv"

CFGS = {
    "u200":  dict(K_MAX=32768, ADDR_BEAT_W=28, WS_FIFO_BEATS=512, WS_RD_LAT=2, RD_OUTSTANDING=16),
    "small": dict(K_MAX=4096, ADDR_BEAT_W=13, WS_FIFO_BEATS=64, WS_RD_LAT=1, RD_OUTSTANDING=1),
    "wide":  dict(K_MAX=65280, ADDR_BEAT_W=40, WS_FIFO_BEATS=1024, WS_RD_LAT=3, RD_OUTSTANDING=3),
    "tiny":  dict(K_MAX=256, ADDR_BEAT_W=5, WS_FIFO_BEATS=64, WS_RD_LAT=4, RD_OUTSTANDING=2),
}
# Configurations whose WS budget sustains 1 beat/cycle with an ideal memory (FIFO >= 2 bursts and
# >= 2 outstanding bursts; tiny: every stream is a single burst of <= 32 beats).
FULL_RATE = {"u200": True, "small": False, "wide": True, "tiny": True}

TESTS = ["front_directed", "front_random", "front_fp16_all", "front_throughput", "front_fifo_sizing",
         "front_reset", "front_max_count", "front_comb_probe"]

INPUTS = ["cmd_valid", "cmd", "beat_ready", "f_rd_req_valid", "f_rd_req_addr", "f_rd_req_len",
          "f_rd_req_id", "f_rd_rsp_ready", "m_rd_req_ready", "m_rd_rsp_valid", "m_rd_rsp_data",
          "m_rd_rsp_id", "m_rd_rsp_last", "m_wr_req_ready", "m_wr_dat_ready", "m_wr_rsp_valid",
          "m_wr_rsp_id"]
OUTPUTS = ["cmd_ready", "beat_valid", "beat", "idle", "ws_req_valid", "ws_req_ready", "ws_req_addr",
           "ws_req_len", "ws_req_id", "ws_rsp_valid", "ws_rsp_ready", "ws_idle", "f_rd_req_ready", "f_rd_rsp_valid",
           "f_rd_rsp_data", "f_rd_rsp_last", "f_rd_rsp_id", "m_rd_req_valid", "m_rd_req_addr",
           "m_rd_req_len", "m_rd_req_id", "m_rd_rsp_ready", "m_wr_req_valid", "m_wr_req_addr",
           "m_wr_req_len", "m_wr_req_id", "m_wr_dat_valid", "m_wr_dat_data", "m_wr_dat_last",
           "m_wr_rsp_ready", "perf_rd_beat", "perf_rd_wait"]

WS_ID = isa.SRC_WS << isa.SRC_ID_W      # memory-side id {2, 0} = 8
F16_SPECIAL = (0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
               0x7E00, 0x7C01, 0xFE01, 0x7FFF, 0x3C00)


# =============================================================================================
# pytest side
# =============================================================================================
@pytest.mark.parametrize("cfg", list(CFGS))
def test_mvm_front(cfg):
    p = CFGS[cfg]
    sel = os.environ.get("TSR_FRONT_TESTS")
    tests = [t for t in TESTS if not sel or t in sel.split(",")]
    run("tsr_tb_mvm_front", __name__, parameters=p, extra_sources=[TB_SV], testcase=tests,
        extra_env={"TSR_FRONT_CFG": json.dumps(dict(p, name=cfg))})


# =============================================================================================
# Helpers
# =============================================================================================
def cfg_from_env() -> dict:
    return json.loads(os.environ["TSR_FRONT_CFG"])


def scale() -> float:
    return float(os.environ.get("TSR_FRONT_SCALE", "1.0"))


class BurstQueue:
    """The WS bursts still expected, in order: per accepted command its beats from w_off as
    bursts of min(64 - addr % 64, left) beats in address order (docs/rtl-m3.md §4), produced
    lazily because a legal command may have up to 2^40 - 1 beats (front_max_count)."""

    def __init__(self):
        self.q = deque()            # [addr, beats left] per command

    def add(self, addr: int, n: int):
        if n:
            self.q.append([addr, n])

    def __bool__(self) -> bool:
        return bool(self.q)

    def clear(self):
        self.q.clear()

    def popleft(self):
        e = self.q[0]
        ln = min(e[1], isa.BURST_ALIGN - e[0] % isa.BURST_ALIGN)
        out = (e[0], ln)
        e[0] += ln
        e[1] -= ln
        if not e[1]:
            self.q.popleft()
        return out

    def head(self, k: int = 3) -> list:
        """The next k bursts (not consumed), for messages."""
        out = []
        for a, n in self.q:
            while n and len(out) < k:
                ln = min(n, isa.BURST_ALIGN - a % isa.BURST_ALIGN)
                out.append((a, ln))
                a += ln
                n -= ln
        return out


def beat_diff(got: int, exp: int) -> str:
    g, e = mm.unpack_beat(got), mm.unpack_beat(exp)
    out = []
    for k in mm.BEAT_KEYS:
        if g[k] != e[k]:
            if k in ("w", "sc", "m"):
                lanes = [i for i in range(len(e[k])) if g[k][i] != e[k][i]]
                out.append("%s lanes %s: got %s exp %s" % (k, lanes[:8], [g[k][i] for i in lanes[:8]],
                                                           [e[k][i] for i in lanes[:8]]))
            else:
                out.append("%s: got 0x%x exp 0x%x" % (k, g[k], e[k]))
    return "; ".join(out)


async def clock_reset(dut, cycles: int = 5) -> Clock:
    """rst_n low before the first edge (random initial state), released after `cycles` edges."""
    dut.rst_n.value = 0
    dut.clk.value = 0
    dut.cmd_valid.value = 0
    dut.cmd.value = 0
    dut.beat_ready.value = 0
    dut.f_rd_req_valid.value = 0
    dut.f_rd_req_addr.value = 0
    dut.f_rd_req_len.value = 0
    dut.f_rd_req_id.value = 0
    dut.f_rd_rsp_ready.value = 1
    await Timer(1, "ns")
    clk = Clock(dut.clk, 10, unit="ns")
    clk.start(start_high=True)
    await ClockCycles(dut.clk, cycles)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    return clk


async def comb_probe(dut, clock: Clock, rnd: random.Random, tries: int = 2) -> int:
    """Stop the clock right after a falling edge, wiggle every input (one at a time) and check that
    no output changes (any change is an input -> output combinational path); restore and restart."""
    await FallingEdge(dut.clk)
    clock.stop()
    await Timer(1, "ns")
    outs = [(n, getattr(dut, n)) for n in OUTPUTS]
    snap = {n: s.value for n, s in outs}
    bad, pokes = set(), 0
    for name in INPUTS:
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
# Data generation
# =============================================================================================
def k_choices(fmt: int, k_max: int):
    bs = mm.block_size(fmt)
    ks = {bs, 2 * bs, 3 * bs, 1024 - bs, 1024, 1024 + bs, 2048 - bs, 2048, 2048 + 3 * bs, k_max,
          k_max - bs, (k_max // 1024) * 1024 + bs}
    return sorted(k for k in ks if 0 < k <= k_max and k % bs == 0)


def special_row(fmt: int, K: int, rnd: random.Random, kind: int) -> bytes:
    """Repacked-layout row image with extreme content: kind 0 all quant bytes 0x80 (-128), 1 0x7F,
    2 0xFF (Q4_K nibbles 15), 3 0x00; meta: special fp16 d / dmin, Q4_K scales 0xFF / random."""
    rb = mm.row_beats(fmt, K)
    img = bytearray()
    for c, nblk in enumerate(mm.row_chunks(fmt, K)):
        meta = bytearray(rnd.randbytes(64))
        if fmt == mm.FMT_Q8_0:
            for b in range(32):
                h = rnd.choice(F16_SPECIAL)
                meta[2 * b:2 * b + 2] = h.to_bytes(2, "little")
        else:
            for s in range(4):
                meta[16 * s:16 * s + 2] = rnd.choice(F16_SPECIAL).to_bytes(2, "little")
                meta[16 * s + 2:16 * s + 4] = rnd.choice(F16_SPECIAL).to_bytes(2, "little")
                if kind % 2 == 0:
                    meta[16 * s + 4:16 * s + 16] = b"\xff" * 12
        img += meta
        fill = (0x80, 0x7F, 0xFF, 0x00)[kind % 4]
        for _ in range(mm.quant_beats(fmt, nblk)):
            img += bytes([fill]) * 64
    assert len(img) == 64 * rb
    return bytes(img)


def gen_rows(fmt: int, K: int, n: int, rnd: random.Random, mode: str = "mix") -> list:
    rows = []
    for i in range(n):
        m = mode if mode != "mix" else rnd.choice(("raw", "raw", "pack", "pack_special", "special"))
        if m == "raw":
            r = mm.rand_row(fmt, K, rnd, raw=True)
        elif m == "pack":
            r = mm.rand_row(fmt, K, rnd)
        elif m == "pack_special":
            r = mm.rand_row(fmt, K, rnd, special=0.2)
        else:
            r = special_row(fmt, K, rnd, rnd.randrange(4))
        if rnd.random() < 0.3:          # sprinkle -128 weights into the quant beats
            r = bytearray(r)
            for _ in range(8):
                p = rnd.randrange(len(r))
                if p % (64 * (17 if fmt == 0 else 9)) >= 64:
                    r[p] = 0x80
            r = bytes(r)
        rows.append(r)
    return rows


class Cmd:
    """One front-end command with its rows and the expected beat stream. With n_rows > len(rows)
    the command covers n_rows rows of which only the first len(rows) are loaded and expected
    (a prefix check of a stream too long to simulate: front_max_count)."""

    def __init__(self, fmt, K, rows, w_off, tag="", n_rows=None):
        self.fmt, self.K, self.rows, self.w_off, self.tag = fmt, K, rows, w_off, tag
        self.n_rows = len(rows) if n_rows is None else n_rows
        assert self.n_rows >= len(rows)
        self.rb = mm.row_beats(fmt, K)
        self.beats = self.n_rows * self.rb
        self.fc = dict(fmt=fmt, k=K, row_beats=self.rb, w_off=w_off, rows=self.n_rows, beats=self.beats)
        self.exp = None

    def expected(self) -> list:
        if self.exp is None:
            ds = mm.front_beats(self.fmt, self.K, self.rows)
            if self.n_rows > len(self.rows):    # prefix of a longer command: its last beat is later
                for d in ds:
                    d["last"] = 0
            r = ref()
            for d in ds:            # sc / m straight from libtsr_ref's get_scale_min_k4
                if self.fmt == mm.FMT_Q4_K:
                    row = self.rows[d["row"]]
                    meta0 = 64 * (d["beat"] - (d["beat"] - 1) % 9 - 1)
                    j = (d["beat"] - meta0 // 64 - 1) // 2
                    sc12 = row[meta0 + 16 * j + 4:meta0 + 16 * j + 16]
                    scm = [r.get_scale_min_k4(k, sc12) for k in range(8)]
                    assert d["sc"] == [scm[4 * d["half"] + g][0] for g in range(4)], "model sc"
                    assert d["m"] == [scm[k][1] for k in range(8)], "model m"
            self.exp = [mm.pack_beat(d) for d in ds]
            self.info = [(d["row"], d["chunk"], d["beat"]) for d in ds]
        return self.exp

    def data(self) -> bytes:
        return b"".join(self.rows)

    def __repr__(self):
        return "Cmd(%s fmt %d K %d rows %d w_off 0x%x beats %d)" % (self.tag, self.fmt, self.K, self.n_rows,
                                                                    self.w_off, self.beats)


# =============================================================================================
# Background FETCH reader
# =============================================================================================
class BgFetch:
    """Random read bursts on the arbiter's FETCH source (ids 0..3) with random response ready."""

    def __init__(self, dut, rnd: random.Random, limit: int, p_req: float = 0.08, p_ready: float = 0.75):
        self.dut, self.rnd, self.limit = dut, rnd, limit
        self.p_req, self.p_ready = p_req, p_ready
        self.enabled = False
        self.req = None                       # (addr, len, id) offered
        self.pend = {i: deque() for i in range(4)}   # per id: beats left of each burst
        self.beats = 0
        self.bursts = 0
        self.task = None

    def start(self):
        self.task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

    def busy(self) -> bool:
        return self.req is not None or any(self.pend[i] for i in range(4))

    async def drain(self, timeout: int = 200000):
        self.enabled = False
        for _ in range(timeout):
            if not self.busy():
                return
            await RisingEdge(self.dut.clk)
        raise TimeoutError("background FETCH reads did not drain")

    async def _run(self):
        dut, rnd = self.dut, self.rnd
        while True:
            await FallingEdge(dut.clk)
            if int(dut.rst_n.value) == 0:
                self.req = None
                self.pend = {i: deque() for i in range(4)}
                await RisingEdge(dut.clk)
                dut.f_rd_req_valid.value = 0
                continue
            fire = self.req is not None and int(dut.f_rd_req_ready.value)
            rsp = int(dut.f_rd_rsp_valid.value) and int(dut.f_rd_rsp_ready.value)
            if rsp:
                i = int(dut.f_rd_rsp_id.value)
                assert self.pend[i], "FETCH response for id %d with nothing outstanding" % i
                self.pend[i][0] -= 1
                last = self.pend[i][0] == 0
                assert int(dut.f_rd_rsp_last.value) == int(last), "FETCH rsp_last wrong"
                if last:
                    self.pend[i].popleft()
                self.beats += 1
            await RisingEdge(dut.clk)
            if fire:
                a, ln, i = self.req
                self.pend[i].append(ln)
                self.bursts += 1
                self.req = None
                dut.f_rd_req_valid.value = 0
            if self.req is None and self.enabled and rnd.random() < self.p_req:
                a = rnd.randrange(self.limit)
                ln = rnd.randint(1, min(64 - a % 64, self.limit - a))
                self.req = (a, ln, rnd.randrange(4))
                dut.f_rd_req_addr.value = a
                dut.f_rd_req_len.value = ln
                dut.f_rd_req_id.value = self.req[2]
                dut.f_rd_req_valid.value = 1
            dut.f_rd_rsp_ready.value = int(rnd.random() < self.p_ready)


# =============================================================================================
# Environment
# =============================================================================================
class Env:
    def __init__(self, dut, cfg, rnd, *, mem_kw=None):
        self.dut, self.cfg, self.rnd = dut, cfg, rnd
        self.aw = cfg["ADDR_BEAT_W"]
        self.limit = 1 << self.aw
        self.kmax = cfg["K_MAX"]
        self.mem = MemModel(dut, num_ch=1, prefix="m_", addr_w=self.aw, rd_outstanding=cfg["RD_OUTSTANDING"],
                            limit_beats=self.limit, seed=rnd.getrandbits(31), **(mem_kw or {}))
        self.bg = BgFetch(dut, rnd, self.limit)
        self.cyc = 0
        self.ready_mode = ("always",)
        # expectations
        self.exp = deque()          # (packed beat, cmd, index)
        self.exp_bursts = BurstQueue()  # (addr, len) of the WS source, in order (accepted commands)
        self.next_bursts = None     # (w_off, beats, tagged beats) of the command being offered
                                    # (armed at its handshake)
        self.fired = 0              # tagged beats of all accepted commands
        self.got = 0                # beats delivered
        self.ws_beats = 0           # WS response beats at the source port
        self.out_cyc = []           # cycle of every delivered beat (since clear_timing)
        self.rsp_cyc = []           # cycle of every WS response beat at the source port
        self.fire_cyc = []          # cycle of every command handshake
        self.prev_out = None        # (valid, ready, beat) of the previous cycle (R2)
        self.prev_req = None
        self.cmd_ready_prev = None
        self.cov = dict(q8_m128=0, q4_n0=0, q4_n15=0, d_nan=0, d_inf=0, d_sub=0, d_zero=0, two0=0,
                        row_last=0, last=0, q4_half1=0, sc63=0)
        self.ws_idle_seen = 0       # WS beats received when tsr_mvm_ws was last seen idle
        self.mon_task = self.rdy_task = None
        self.errors = []

    # ------------------------------------------------------------------------------------------
    def start(self, bg: bool = False):
        self.mem.start()
        self.mon_task = cocotb.start_soon(self._mon())
        self.rdy_task = cocotb.start_soon(self._ready())
        self.bg.start()
        self.bg.enabled = bg
        return self

    def stop(self):
        self.mem.stop()
        self.bg.stop()
        for t in (self.mon_task, self.rdy_task):
            if t is not None:
                t.cancel()

    def clear_timing(self):
        self.out_cyc.clear()
        self.rsp_cyc.clear()
        self.fire_cyc.clear()

    def reset_tracking(self):
        """After a DUT reset: nothing is expected any more."""
        self.exp.clear()
        self.exp_bursts.clear()
        self.next_bursts = None
        self.fired = self.got = 0
        self.prev_out = self.prev_req = None

    # ------------------------------------------------------------------------------------------
    async def _ready(self):
        dut, rnd = self.dut, self.rnd
        on, left = 1, 0
        while True:
            await RisingEdge(dut.clk)
            m = self.ready_mode
            if m[0] == "always":
                v = 1
            elif m[0] == "random":
                v = int(rnd.random() < m[1])
            elif m[0] == "upto":        # ready until m[1] beats were delivered in total, then 0
                v = int(self.got < m[1])
            else:                       # ("bursty", max_on, max_off)
                if left <= 0:
                    on = 1 - on
                    left = rnd.randint(1, m[1] if on else m[2])
                left -= 1
                v = on
            dut.beat_ready.value = v

    async def _mon(self):
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            self.cyc += 1
            if int(dut.rst_n.value) == 0:
                self.prev_out = self.prev_req = None
                self.cmd_ready_prev = None
                continue
            try:
                self._sample()
            except AssertionError as e:
                self.errors.append(str(e))
                raise

    def _sample(self):
        dut = self.dut
        # tsr_mvm_ws idle = no command, nothing outstanding, FIFO empty (checked before a command
        # handshake of this cycle arms new bursts)
        if int(dut.ws_idle.value):
            assert not self.exp_bursts and self.prev_req is None and not int(dut.ws_req_valid.value), \
                "tsr_mvm_ws idle with WS bursts still to request"
            busy = [(b.id, b.addr, b.len, b.sent) for b in self.mem.ch[0].rd_bursts if (b.id >> 2) == isa.SRC_WS]
            assert not busy, "tsr_mvm_ws idle with WS reads in flight at the memory: %s" % busy[:3]
            assert not int(dut.ws_rsp_valid.value), "tsr_mvm_ws idle while a WS beat arrives"
            self.ws_idle_seen = self.ws_beats
        # ---- command handshake / idle ----
        cr = int(dut.cmd_ready.value)
        assert int(dut.idle.value) == cr, "idle != cmd_ready"
        if cr:
            assert self.got == self.fired, "cmd_ready = 1 with %d of %d beats delivered" % (self.got, self.fired)
            assert not self.exp_bursts, "cmd_ready = 1 with WS bursts not yet requested: %s" % self.exp_bursts.head()
            busy = [(b.id, b.addr, b.len, b.sent) for b in self.mem.ch[0].rd_bursts if (b.id >> 2) == isa.SRC_WS]
            assert not busy, "cmd_ready = 1 with WS reads in flight at the memory: %s" % busy[:3]
            if int(dut.cmd_valid.value):
                fc = isa.MVM_FCMD_T.unpack(int(dut.cmd.value))
                assert self.next_bursts is not None, "command handshake without an offered command"
                nw, nb, ntag = self.next_bursts
                assert (nw, nb) == (fc["w_off"], fc["beats"]), "offered command differs from the driven one"
                self.fired += ntag
                self.fire_cyc.append(self.cyc)
                self.exp_bursts.add(nw, nb)
                self.next_bursts = None
        # ---- WS source port: requests and rd_rsp_ready ----
        if int(dut.ws_req_valid.value):
            req = (int(dut.ws_req_addr.value), int(dut.ws_req_len.value), int(dut.ws_req_id.value))
            if self.prev_req is not None:
                assert req == self.prev_req, "WS request changed before the handshake: %s -> %s" % (self.prev_req, req)
            if int(dut.ws_req_ready.value):
                self.prev_req = None
                assert self.exp_bursts, "unexpected WS request %s" % (req,)
                ea, el = self.exp_bursts.popleft()
                assert req == (ea, el, 0), "WS request (addr 0x%x, len %d, id %d), expected (0x%x, %d, 0)" % (
                    req[0], req[1], req[2], ea, el)
            else:
                self.prev_req = req
        else:
            assert self.prev_req is None, "WS request valid dropped before the handshake"
        if int(dut.ws_rsp_valid.value):
            assert int(dut.ws_rsp_ready.value) == 1, "WS rd_rsp_ready = 0 while a WS beat is offered"
            self.ws_beats += 1
            self.rsp_cyc.append(self.cyc)
        # ---- beat output ----
        v, r = int(dut.beat_valid.value), int(dut.beat_ready.value)
        bt = int(dut.beat.value) if v else None
        if self.prev_out is not None:
            pv, pr, pb = self.prev_out
            if pv and not pr:
                assert v, "beat_valid dropped without a handshake (R2)"
                assert bt == pb, "beat payload changed without a handshake (R2)"
        self.prev_out = (v, r, bt)
        if v and r:
            assert self.exp, "beat delivered with no beat expected: %s" % mm.unpack_beat(bt)
            e, cmd, i = self.exp.popleft()
            if bt != e:
                raise AssertionError("%s beat %d (row, chunk, row beat) = %s: %s" % (
                    cmd, i, cmd.info[i], beat_diff(bt, e)))
            self.got += 1
            self.out_cyc.append(self.cyc)
            self._cover(cmd.fmt, bt)

    def _cover(self, fmt, bt):
        d = mm.unpack_beat(bt)
        c = self.cov
        if fmt == mm.FMT_Q8_0:
            c["q8_m128"] += d["w"].count(-128)
            c["two0"] += d["two"] == 0
        else:
            c["q4_n0"] += d["w"].count(0) > 0
            c["q4_n15"] += d["w"].count(15) > 0
            c["q4_half1"] += d["half"]
            c["sc63"] += 63 in d["sc"] or 63 in d["m"]
        for x in (d["d_a"], d["d_b"]):
            e, m = (x >> 23) & 0xFF, x & 0x7FFFFF
            c["d_nan"] += e == 0xFF and m != 0
            c["d_inf"] += e == 0xFF and m == 0
            c["d_zero"] += e == 0 and m == 0
            # fp16 subnormals become normal f32 with exponents 103..112
            c["d_sub"] += 103 <= e <= 112
        c["row_last"] += d["row_last"]
        c["last"] += d["last"]

    # ------------------------------------------------------------------------------------------
    async def send(self, cmd: Cmd, gap: int = 0):
        """Wait until idle, place the rows at W_OFF (host write: no WS read is in flight), queue
        the expectations and hand the command over (valid held until the handshake)."""
        dut = self.dut
        exp = cmd.expected()
        for _ in range(gap):
            await RisingEdge(dut.clk)
        for n in range(1000000):
            await FallingEdge(dut.clk)
            if self.errors:
                raise AssertionError(self.errors[0])
            if int(dut.cmd_ready.value):
                break
        else:
            raise TimeoutError("cmd_ready never rose")
        assert cmd.w_off + cmd.beats <= self.limit, cmd
        self.mem.load(0, 64 * cmd.w_off, cmd.data())
        self.exp.extend((e, cmd, i) for i, e in enumerate(exp))
        self.next_bursts = (cmd.w_off, cmd.beats, len(exp))
        await RisingEdge(dut.clk)
        dut.cmd.value = isa.MVM_FCMD_T.pack(**cmd.fc)
        dut.cmd_valid.value = 1
        while True:
            await FallingEdge(dut.clk)
            fire = int(dut.cmd_ready.value)
            await RisingEdge(dut.clk)
            if fire:
                break
        dut.cmd_valid.value = 0

    async def finish(self, timeout: int = 2000000):
        """Wait until every expected beat was delivered and the DUT is idle; final checks."""
        dut = self.dut
        for n in range(timeout):
            await FallingEdge(dut.clk)
            if self.errors:
                raise AssertionError(self.errors[0])
            if not self.exp and int(dut.cmd_ready.value) and self.got == self.fired:
                break
        else:
            raise TimeoutError("front end did not finish: %d beats still expected, got %d of %d" % (
                len(self.exp), self.got, self.fired))
        await self.bg.drain()
        await ClockCycles(dut.clk, 8)
        assert not self.exp_bursts, "WS bursts never requested: %s" % self.exp_bursts.head(4)
        self.mem.check(idle=True)

    def check_src_beats(self, ws_beats: int):
        sb = self.mem.src_beats[0]
        assert sb[isa.SRC_WS] == ws_beats, "MemModel WS beats %d, expected %d" % (sb[isa.SRC_WS], ws_beats)
        assert sb[isa.SRC_DMA] == 0, "DMA beats %d on a tied-off source" % sb[isa.SRC_DMA]
        assert sb[isa.SRC_FETCH] == self.bg.beats, "FETCH beats %d, background reader saw %d" % (
            sb[isa.SRC_FETCH], self.bg.beats)
        assert self.ws_beats == ws_beats, "WS beats at the source port %d, expected %d" % (self.ws_beats, ws_beats)


async def setup(dut, *, salt: int, mem_kw=None, bg=False):
    cfg = cfg_from_env()
    rnd = random.Random(tb_seed() * 1000003 + salt)
    clk = await clock_reset(dut)
    env = Env(dut, cfg, rnd, mem_kw=mem_kw).start(bg=bg)
    return cfg, rnd, env, clk


def ideal_mem(lat: int = 1) -> dict:
    return dict(rd_lat=(lat, lat), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0)


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def front_directed(dut):
    """Every K edge, rows 1..3, W_OFF at 4 KiB edges and at the DRAM limit, special data."""
    cfg, rnd, env, _ = await setup(dut, salt=1)
    limit = env.limit
    cmds = []
    total = 0
    ri = 0
    for fmt in (mm.FMT_Q8_0, mm.FMT_Q4_K):
        for K in k_choices(fmt, env.kmax):
            rb = mm.row_beats(fmt, K)
            for nrows in (1, 2, 3):
                if nrows * rb > limit:
                    continue
                beats = nrows * rb
                spots = [0, 63, 62, 1, 64 - beats % 64, 128 - 1, limit - beats]
                w = spots[ri % len(spots)] % (limit - beats + 1)
                w = min(w, (1 << 32) - 1)
                ri += 1
                mode = ("raw", "pack_special", "special")[ri % 3]
                cmds.append(Cmd(fmt, K, gen_rows(fmt, K, nrows, rnd, mode), w, "dir"))
                total += beats
    # explicit: ending exactly at the DRAM limit, each format, partial chunk
    for fmt in (mm.FMT_Q8_0, mm.FMT_Q4_K):
        K = mm.block_size(fmt) * 3 if mm.block_size(fmt) * 3 <= env.kmax else mm.block_size(fmt)
        rb = mm.row_beats(fmt, K)
        nrows = max(1, min(5, limit // rb))
        if (limit - nrows * rb) < (1 << 32):
            cmds.append(Cmd(fmt, K, gen_rows(fmt, K, nrows, rnd, "special"), limit - nrows * rb, "limit"))
            total += nrows * rb
    modes = [("always",), ("random", 0.5), ("bursty", 30, 30), ("random", 0.9)]
    for i, c in enumerate(cmds):
        env.ready_mode = modes[i % len(modes)]
        await env.send(c, gap=rnd.choice((0, 0, 3)))
    await env.finish()
    env.check_src_beats(total)
    cov = env.cov
    dut._log.info("directed: %d commands, %d weight beats, %d tagged beats, coverage %s", len(cmds), total,
                  env.got, cov)
    for k in ("q8_m128", "q4_n0", "q4_n15", "d_nan", "d_inf", "d_sub", "d_zero", "row_last", "last",
              "q4_half1", "sc63"):
        assert cov[k] > 0, "directed coverage: no %s" % k
    if any(mm.row_chunks(0, K)[-1] % 2 for K in k_choices(0, env.kmax)):
        assert cov["two0"] > 0, "directed coverage: no Q8_0 beat with two = 0"
    env.stop()


def rand_cmds(env, rnd: random.Random, n: int, budget: int):
    """Commands of random clusters: fmt, K, N, NUM_CH, channel c -> fcmd (rows_c = 0: skipped)."""
    limit, kmax = env.limit, env.kmax
    out, skipped = [], 0
    while len(out) < n:
        fmt = rnd.randrange(2)
        bs = mm.block_size(fmt)
        r = rnd.random()
        if r < 0.15:
            K = bs
        elif r < 0.25:
            K = kmax
        elif r < 0.6:
            K = rnd.randint(1, kmax // bs) * bs
        else:
            K = min(kmax, rnd.randint(0, max(0, kmax // 1024 - 1)) * 1024 + rnd.randint(1, 1024 // bs) * bs)
        rb = mm.row_beats(fmt, K)
        num_ch = rnd.choice((1, 2, 3, 4, 4, 5, 7, 16))
        maxrows = max(1, min(budget, limit) // rb)
        r = rnd.random()
        if r < 0.1:
            N = 0
        elif r < 0.25:
            N = 1
        elif r < 0.4:
            N = rnd.randint(1, num_ch)                     # some channels empty
        elif r < 0.6:
            N = 2 * rnd.randint(0, maxrows * num_ch // 2) + 1   # odd
        else:
            N = rnd.randint(1, maxrows * num_ch)
        c = rnd.randrange(num_ch)
        rc = mm.rows_c(N, num_ch, c)
        if rc == 0 or rc * rb > min(budget, limit):
            skipped += 1
            continue
        beats = rc * rb
        hi = min(limit - beats, (1 << 32) - 1)
        r = rnd.random()
        if r < 0.3:
            w = rnd.randint(0, hi)
        elif r < 0.6:                                     # around a 4 KiB block edge
            w = max(0, min(hi, 64 * rnd.randint(0, hi // 64 + 1) + rnd.randint(-3, 3)))
        elif r < 0.75:
            w = limit - beats if limit - beats < (1 << 32) else hi    # ends exactly at the limit
        elif r < 0.85 and limit > (1 << 32):
            w = (1 << 32) - rnd.randint(1, min(beats, 200))           # crosses beat 2^32
        else:
            w = rnd.randint(0, min(hi, 4096))
        fc = mm.fcmd(fmt, K, N, num_ch, c, w)
        assert fc["rows"] == rc and fc["beats"] == beats
        out.append(Cmd(fmt, K, gen_rows(fmt, K, rc, rnd), w, "N=%d C=%d c=%d" % (N, num_ch, c)))
    return out, skipped


@cocotb.test()
async def front_random(dut):
    """Random cluster commands under random memory regimes, backpressure and background reads."""
    cfg, rnd, env, _ = await setup(dut, salt=2, bg=True)
    s = scale()
    budget = {"u200": 2500, "small": 1200, "wide": 3000, "tiny": 32}[cfg["name"]]
    ncmd = int({"u200": 70, "small": 70, "wide": 45, "tiny": 200}[cfg["name"]] * s)
    cmds, skipped = rand_cmds(env, rnd, ncmd, budget)
    regimes = [
        dict(rd_lat=(1, 40)),
        dict(rd_lat=(1, 3), p_rd_req_ready=1.0, p_rd_rsp_valid=1.0, p_stall=0.0),
        dict(rd_lat=(60, 250), p_rd_rsp_valid=0.95),
        dict(rd_lat=(1, 20), p_rd_req_ready=0.3, p_rd_rsp_valid=0.4, p_stall=0.05),
    ]
    modes = [("always",), ("random", 0.7), ("random", 0.25), ("bursty", 50, 200), ("bursty", 5, 3)]
    total = 0
    for i, c in enumerate(cmds):
        if i % 8 == 0:
            reg = regimes[(i // 8) % len(regimes)]
            for k, v in {**dict(rd_lat=(1, 40), p_rd_req_ready=0.7, p_rd_rsp_valid=0.8, p_stall=0.01), **reg}.items():
                setattr(env.mem, k, v)
            env.bg.enabled = (i // 8) % 3 != 1
        env.ready_mode = rnd.choice(modes)
        await env.send(c, gap=rnd.choice((0, 0, 0, 1, 7)))
        total += c.beats
    await env.finish()
    env.check_src_beats(total)
    dut._log.info("random: %d commands (%d empty channels skipped), %d weight beats, %d tagged beats, "
                  "%d background FETCH beats in %d bursts, %d cycles; coverage %s", len(cmds), skipped, total,
                  env.got, env.bg.beats, env.bg.bursts, env.cyc, env.cov)
    for k in ("q8_m128", "q4_n0", "q4_n15", "row_last", "last", "q4_half1"):
        assert env.cov[k] > 0, "random coverage: no %s" % k
    env.stop()


@cocotb.test()
async def front_fp16_all(dut):
    """Every fp16 pattern as a Q8_0 d (d_a and d_b): the f32 the back end gets is tsr_fp16_to_f32."""
    cfg, rnd, env, _ = await setup(dut, salt=3, mem_kw=ideal_mem(2))
    if cfg["name"] != "u200":
        dut._log.info("fp16_all: only in the u200 configuration (the conversion is parameter-free)")
        env.stop()
        return
    K = 32768
    rb = mm.row_beats(0, K)
    vals = list(range(1 << 16))
    rnd.shuffle(vals)
    rows = []
    per_row = K // 32
    for r0 in range(0, len(vals), per_row):
        img = bytearray(rnd.randbytes(64 * rb))
        for c in range(32):
            for b in range(32):
                h = vals[r0 + 32 * c + b]
                img[64 * 17 * c + 2 * b:64 * 17 * c + 2 * b + 2] = h.to_bytes(2, "little")
        rows.append(bytes(img))
    env.ready_mode = ("always",)
    cmd = Cmd(0, K, rows, 1000, "fp16")
    await env.send(cmd)
    await env.finish()
    env.check_src_beats(cmd.beats)
    # the expected stream (compared beat by beat above) carries every fp16 value exactly once as
    # d_a or d_b, each converted by libtsr_ref's tsr_fp16_to_f32
    r = ref()
    seen = {}
    for x in cmd.exp:
        d = mm.unpack_beat(x)
        seen[d["d_a"]] = seen.get(d["d_a"], 0) + 1
        seen[d["d_b"]] = seen.get(d["d_b"], 0) + 1
    want = {}
    for h in range(1 << 16):
        f = r.fp16_to_f32(h)
        want[f] = want.get(f, 0) + 1
    assert seen == want, "the fp16 test stream does not carry every fp16 value exactly once"
    dut._log.info("fp16_all: %d rows, %d beats: all 65536 fp16 values as d_a / d_b == tsr_fp16_to_f32",
                  len(rows), cmd.beats)
    env.stop()


async def measure(env, cmds, label: str, *, require_full: bool, log):
    """Run the commands back to back with beat_ready = 1; per command: cycles between the first and
    the last tagged beat vs the weight beats after the first META (1.000 cycles per weight beat =
    one weight beat per cycle sustained across chunk and row boundaries)."""
    env.ready_mode = ("always",)
    res = []
    for c in cmds:
        env.clear_timing()
        await env.send(c)
        await env.finish()
        n = len(c.expected())
        span = env.out_cyc[-1] - env.out_cyc[0] + 1
        need = c.beats - 1
        lat = env.out_cyc[0] - env.fire_cyc[0]
        rsp_span = env.rsp_cyc[-1] - env.rsp_cyc[0] + 1
        cpb = span / need if need else 1.0
        res.append((c, cpb, lat))
        log.info("%s %s: %d weight beats (%d tagged), output span %d cycles for %d beats -> %.4f cycles/beat; "
                 "WS port span %d cycles for %d beats; first tagged beat %d cycles after the command",
                 label, c, c.beats, n, span, need, cpb, rsp_span, c.beats, lat)
        if require_full:
            assert span == need, "%s %s: %d cycles for %d weight beats (not 1 beat/cycle)" % (label, c, span, need)
    return res


@cocotb.test()
async def front_throughput(dut):
    """Ideal memory, beat_ready = 1: one weight beat per cycle (META beats included)."""
    cfg, rnd, env, _ = await setup(dut, salt=4, mem_kw=ideal_mem(1))
    kmax, limit = env.kmax, env.limit
    cmds = []
    for fmt, K in ((0, 1056), (1, 1280), (0, 32), (1, 256), (0, kmax), (1, kmax), (0, 5120), (1, 27648)):
        if K > kmax:
            continue
        rb = mm.row_beats(fmt, K)
        nrows = max(1, min(4000 // rb, limit // rb))
        if nrows * rb > limit:
            continue
        w = 63 if 63 + nrows * rb <= limit else 0
        cmds.append(Cmd(fmt, K, gen_rows(fmt, K, nrows, rnd, "raw"), w, "tp"))
    res = await measure(env, cmds, "throughput", require_full=FULL_RATE[cfg["name"]], log=dut._log)
    worst = max(r[1] for r in res)
    dut._log.info("throughput summary (%s): worst %.4f cycles per weight beat, first-beat latency %s cycles",
                  cfg["name"], worst, sorted({r[2] for r in res}))
    env.stop()


@cocotb.test()
async def front_fifo_sizing(dut):
    """docs/rtl-m3.md §4: 1 beat/cycle up to a round trip of ~FIFO_BEATS - 64 - internal latency."""
    cfg, rnd, env, _ = await setup(dut, salt=5, mem_kw=ideal_mem(400))
    if cfg["name"] != "u200":
        dut._log.info("fifo_sizing: only in the u200 configuration (FIFO 512, RD_OUTSTANDING 16)")
        env.stop()
        return
    K = 27648
    rb = mm.row_beats(1, K)
    c1 = Cmd(1, K, gen_rows(1, K, 16, rnd, "raw"), 5, "lat400")
    await measure(env, [c1], "fifo sizing, memory latency 400", require_full=True, log=dut._log)
    env.mem.rd_lat = (520, 520)
    c2 = Cmd(1, K, gen_rows(1, K, 16, rnd, "raw"), 9000, "lat520")
    res = await measure(env, [c2], "fifo sizing, memory latency 520", require_full=False, log=dut._log)
    assert res[0][1] > 1.02, "latency 520 should exceed the FIFO budget (got %.4f cycles/beat)" % res[0][1]
    env.stop()


@cocotb.test()
async def front_reset(dut):
    """Synchronous reset in the middle of a command, then fully checked commands."""
    cfg, rnd, env, _ = await setup(dut, salt=6)
    kmax, limit = env.kmax, env.limit
    for rep in range(3):
        fmt = rep % 2
        K = min(kmax, 2048 if fmt == 0 else 2304)
        rb = mm.row_beats(fmt, K)
        nrows = max(1, min(20, limit // rb))
        c = Cmd(fmt, K, gen_rows(fmt, K, nrows, rnd), 0, "pre-reset")
        env.ready_mode = ("random", 0.5)
        await env.send(c)
        await ClockCycles(dut.clk, rnd.randint(1, max(2, nrows * rb // 2)))
        await RisingEdge(dut.clk)
        dut.rst_n.value = 0
        dut.cmd_valid.value = 0
        await ClockCycles(dut.clk, 3)
        await FallingEdge(dut.clk)
        # reset values (registered outputs)
        assert int(dut.cmd_ready.value) == 1 and int(dut.idle.value) == 1, "cmd_ready / idle after reset"
        assert int(dut.beat_valid.value) == 0, "beat_valid after reset"
        assert int(dut.ws_req_valid.value) == 0 and int(dut.m_rd_req_valid.value) == 0, "requests after reset"
        await RisingEdge(dut.clk)
        dut.rst_n.value = 1
        env.reset_tracking()
        env.mem.clear_tracking()
        env.ws_beats = 0
        total = 0
        for i in range(3):
            fmt2 = rnd.randrange(2)
            K2 = min(kmax, rnd.choice((mm.block_size(fmt2), 1024 + mm.block_size(fmt2), kmax)))
            rb2 = mm.row_beats(fmt2, K2)
            n2 = max(1, min(rnd.randint(1, 6), limit // rb2))
            w2 = rnd.randint(0, min(limit - n2 * rb2, (1 << 32) - 1))
            c2 = Cmd(fmt2, K2, gen_rows(fmt2, K2, n2, rnd), w2, "post-reset")
            await env.send(c2)
            total += c2.beats
        await env.finish()
        env.check_src_beats(total)
    dut._log.info("reset: 3 mid-command resets, then checked commands")
    env.stop()


# Largest N of a legal MVM: the dst operand check (T - 1) * stride + ceil(N / 16) <= SPM lines
# (spec 구현 명확화 "범위 검사", checked before DRAM_RANGE) with SPM_BYTES < 2^32 (tsr_core
# `int unsigned`, tsr_config.spm_bytes uint32_t), i.e. at most 2^26 - 1 SPM lines.
N_MAX = 16 * (((1 << 32) - isa.LINE_BYTES) // isa.LINE_BYTES)      # 2^30 - 16


def max_count(kmax: int, limit: int):
    """The largest per-channel weight-beat count rows_c * ROW_BEATS of any legal MVM with
    K <= K_MAX and DRAM limit `limit` (NUM_CH = 1, rows_c = N <= min(N_MAX, limit // ROW_BEATS)).
    Ties: larger ROW_BEATS, then larger K. Returns (beats, fmt, K, rows)."""
    best = None
    for fmt in (mm.FMT_Q8_0, mm.FMT_Q4_K):
        bs = mm.block_size(fmt)
        for K in range(bs, kmax + 1, bs):
            rb = mm.row_beats(fmt, K)
            rows = min(N_MAX, limit // rb)
            key = (rows * rb, rb, K)
            if rows and (best is None or key > best[0]):
                best = (key, fmt, rows)
    (beats, _, K), fmt, rows = best
    return beats, fmt, K, rows


@cocotb.test()
async def front_max_count(dut):
    """The largest weight-beat count of any legal MVM in the configuration, ending exactly at the
    DRAM limit. wide (ADDR_BEAT_W 40, K_MAX 65280): 2^40 - 1 beats (Q8_0 K 61696, ROW_BEATS 1025,
    1072694271 rows, W_OFF 1) - every bit of tsr_mvm_fcmd_t.beats (TSR_MVM_BEATS_W = 40) is used,
    and 2^40 itself is not a legal count (it would need ROW_BEATS a power of two <= 1024 and
    rows_c >= 2^30 > N_MAX). Counts that fit the budget (small, tiny) run to completion, fully
    checked; longer ones (u200 2^28, wide) are checked on a prefix - the exact burst list, the
    first rows' tagged beats, the FIFO filling up behind a stalled consumer (no hang) - and are
    then reset mid-command, followed by a fully checked command."""
    cfg, rnd, env, _ = await setup(dut, salt=8)
    kmax, limit = env.kmax, env.limit
    beats, fmt, K, rows = max_count(kmax, limit)
    rb = mm.row_beats(fmt, K)
    assert beats < 1 << isa.MVM_BEATS_W, "a legal count does not fit tsr_mvm_fcmd_t.beats"
    if limit == 1 << 40:
        assert (beats, fmt, K, rows) == ((1 << 40) - 1, mm.FMT_Q8_0, 61696, 1072694271), (beats, fmt, K, rows)
    w_off = limit - beats if limit - beats < (1 << 32) else 0
    if beats <= 20000:
        c = Cmd(fmt, K, gen_rows(fmt, K, rows, rnd), w_off, "max")
        env.ready_mode = ("random", 0.8)
        await env.send(c)
        await env.finish()
        env.check_src_beats(beats)
        dut._log.info("max count: %s run to completion (%d tagged beats)", c, env.got)
        env.stop()
        return
    nload = 3
    c = Cmd(fmt, K, gen_rows(fmt, K, nload, rnd), w_off, "max-prefix", n_rows=rows)
    n_exp = len(c.expected())
    env.ready_mode = ("upto", n_exp)
    await env.send(c)
    # prefix rows consumed + a weight FIFO filled up to the reservation granularity (the next
    # burst, <= 64 beats, is requested only when it fits whole)
    fill = nload * rb + cfg["WS_FIFO_BEATS"] - (isa.MAX_BURST_BEATS - 1)
    for n in range(400000):
        await FallingEdge(dut.clk)
        if env.errors:
            raise AssertionError(env.errors[0])
        if env.got == n_exp and env.ws_beats >= fill:
            break
    else:
        raise TimeoutError("max count %s: %d of %d prefix beats, %d WS beats (expected >= %d)" % (
            c, env.got, n_exp, env.ws_beats, fill))
    await ClockCycles(dut.clk, 50)
    assert not env.errors, env.errors[0]
    assert not int(dut.cmd_ready.value) and not int(dut.ws_idle.value), "front end idle inside the command"
    assert env.got == n_exp and env.exp_bursts, "prefix check"
    dut._log.info("max count: %s: %d tagged beats of %d rows checked, %d WS beats requested and received "
                  "in order (next burst %s); reset mid-command", c, n_exp, nload, env.ws_beats,
                  env.exp_bursts.head(1))
    # reset in the middle of the command, then a fully checked command
    await RisingEdge(dut.clk)
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    await ClockCycles(dut.clk, 3)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    env.ready_mode = ("random", 0.7)
    env.reset_tracking()
    env.mem.clear_tracking()
    env.ws_beats = 0
    c2 = Cmd(fmt, K, gen_rows(fmt, K, 2, rnd), rnd.randint(0, min(limit - 2 * rb, (1 << 32) - 1)), "post-reset")
    await env.send(c2)
    await env.finish()
    env.check_src_beats(c2.beats)
    env.stop()


@cocotb.test()
async def front_comb_probe(dut):
    """No combinational path from any input to any output (rules R5/M8), probed mid-stream."""
    cfg, rnd, env, clk = await setup(dut, salt=7)
    kmax, limit = env.kmax, env.limit
    fmt = 1
    K = min(kmax, 2304)
    rb = mm.row_beats(fmt, K)
    nrows = max(1, min(30, limit // rb))
    c = Cmd(fmt, K, gen_rows(fmt, K, nrows, rnd), 0, "probe")
    env.ready_mode = ("random", 0.6)
    await env.send(c)
    pokes = 0
    for _ in range(4):
        await ClockCycles(dut.clk, rnd.randint(3, 20))
        pokes += await comb_probe(dut, clk, rnd)
    # the probe restores every input: the stream must still complete exactly
    await env.finish()
    env.check_src_beats(c.beats)
    dut._log.info("comb probe: %d input pokes, no input->output path", pokes)
    env.stop()

"""cocotb environment shared by the tb/at benches (KVST on the DMA, ATTQK / ATTPV on the MVM unit).

  SpmPort    Python model of one unit's SPM client port (docs/rtl-core.md §5: reads accepted when
             ready, the response exactly SPM_RD_LAT cycles after the accepting edge, no
             backpressure; writes with 64 byte enables). Several ports share one bytearray (the
             SPM image), so the bench can copy it to / from the simulator in one move. Random
             ready (p_rd / p_wr, stall phases); every read / write line is logged.
  SimAt      a tsr_sim instance with CAPS.ATT (sw/sim through libtsr_sim.so, tb/spm/spm_sim.py):
             tsr_sim_op executes one op at once (AT §5 semantics: tsr_at.c calls tsr_ref_at.c),
             with direct SPM / DRAM access and PERF slots.
  clock_reset, wait_done, rand helpers (QA images with fp16-exact or arbitrary scales, raw bytes).
"""
from __future__ import annotations

import ctypes as C
import random
import struct
import sys
from collections import deque
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "spm", HERE.parent / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spm_sim as SS                                   # noqa: E402

LINE = 64


class SpmPort:
    def __init__(self, dut, mem: bytearray, *, prefix: str, rd_lat: int, rnd: random.Random, p_rd: float = 0.85,
                 p_wr: float = 0.85, p_stall: float = 0.01, stall_len=(1, 16), name: str = "spm"):
        self.dut, self.mem, self.rd_lat, self.rnd = dut, mem, rd_lat, rnd
        self.p_rd, self.p_wr, self.p_stall, self.stall_len = p_rd, p_wr, p_stall, stall_len
        self.name = name
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.rq_v, self.rq_r, self.rq_l = g("rd_req_valid"), g("rd_req_ready"), g("rd_req_line")
        self.rs_v, self.rs_d = g("rd_rsp_valid"), g("rd_rsp_data")
        self.wr_v, self.wr_r, self.wr_l = g("wr_valid"), g("wr_ready"), g("wr_line")
        self.wr_d, self.wr_b = g("wr_data"), g("wr_be")
        self.lines = len(mem) // LINE
        self.reads, self.writes = [], []        # (line) / (line, be)
        self.errors = []
        self.cycle = 0
        self._task = None
        for s in (self.rq_r, self.rs_v, self.rs_d, self.wr_r):
            s.value = 0

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

    def line(self, i: int) -> int:
        return int.from_bytes(self.mem[LINE * i:LINE * i + LINE], "little")

    def _fail(self, msg):
        msg = "%s cycle %d: %s" % (self.name, self.cycle, msg)
        self.errors.append(msg)
        raise AssertionError(msg)

    async def _run(self):
        clk, rst = self.dut.clk, self.dut.rst_n
        rd_rdy = wr_rdy = 0
        pend_rq = pend_wr = None
        pipe = deque()
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
                for s in (self.rq_r, self.rs_v, self.wr_r):
                    s.value = 0
                continue
            if rqv:
                if pend_rq is not None and pend_rq != rql:
                    self._fail("rd_req_line changed before the handshake: %d -> %d" % (pend_rq, rql))
                if rd_rdy:
                    pend_rq = None
                    if rql >= self.lines:
                        self._fail("read of line %d >= %d lines" % (rql, self.lines))
                    pipe.append((self.cycle + self.rd_lat - 1, self.line(rql)))   # read-first
                    self.reads.append(rql)
                else:
                    pend_rq = rql
            elif pend_rq is not None:
                self._fail("rd_req_valid dropped before the handshake")
            if wv:
                if pend_wr is not None and pend_wr != wp:
                    self._fail("write payload changed before the handshake (line %d)" % wp[0])
                if wr_rdy:
                    pend_wr = None
                    wl, wd, wb = wp
                    if wl >= self.lines:
                        self._fail("write of line %d >= %d lines" % (wl, self.lines))
                    old = bytearray(self.mem[LINE * wl:LINE * wl + LINE])
                    new = wd.to_bytes(LINE, "little")
                    for j in range(LINE):
                        if (wb >> j) & 1:
                            old[j] = new[j]
                    self.mem[LINE * wl:LINE * wl + LINE] = old
                    self.writes.append((wl, wb))
                else:
                    pend_wr = wp
            elif pend_wr is not None:
                self._fail("wr_valid dropped before the handshake")
            if pipe and pipe[0][0] == self.cycle:
                _, data = pipe.popleft()
                self.rs_v.value = 1
                self.rs_d.value = data
            else:
                self.rs_v.value = 0
            if stall > 0:
                stall -= 1
            elif self.rnd.random() < self.p_stall:
                stall = self.rnd.randint(*self.stall_len)
            rd_rdy = int(stall == 0 and self.rnd.random() < self.p_rd)
            wr_rdy = int(stall == 0 and self.rnd.random() < self.p_wr)
            self.rq_r.value = rd_rdy
            self.wr_r.value = wr_rdy


class SimAt:
    """tsr_sim with CAPS.ATT; op-level API (tsr_sim_op) plus SPM / DRAM / PERF access."""

    def __init__(self, **cfg):
        L = SS.lib()
        if not hasattr(L, "_at_bound"):
            L.tsr_sim_perf.restype = C.c_uint64
            L.tsr_sim_perf.argtypes = [C.c_void_p, C.c_uint32]
            L._at_bound = True
        cfg.setdefault("caps", 0x20)
        self.cfg = SS.u200_cfg(**cfg)
        self.sim = L.tsr_sim_create(C.byref(SS.TsrConfig(**self.cfg)), None)
        assert self.sim, self.cfg
        self.L = L

    def close(self):
        if self.sim:
            self.L.tsr_sim_destroy(self.sim)
            self.sim = None

    def dram_ptr(self, ch: int) -> int:
        return self.L.tsr_sim_dram(self.sim, ch)

    def dram_write(self, ch: int, off: int, data: bytes):
        C.memmove(self.dram_ptr(ch) + off, bytes(data), len(data))

    def dram_read(self, ch: int, off: int, n: int) -> bytes:
        return C.string_at(self.dram_ptr(ch) + off, n)

    def spm_write(self, off: int, data):
        C.memmove(self.L.tsr_sim_spm(self.sim) + off, bytes(data), len(data))

    def spm_read(self, off: int, n: int) -> bytes:
        return C.string_at(self.L.tsr_sim_spm(self.sim) + off, n)

    def op(self, words):
        info = C.c_uint32(0)
        code = self.L.tsr_sim_op(self.sim, (C.c_uint32 * 8)(*[w & 0xFFFFFFFF for w in words]), C.byref(info))
        return code, info.value

    def perf(self, idx: int) -> int:
        return self.L.tsr_sim_perf(self.sim, idx)

    def op_with_ch_beats(self, words, num_ch: int):
        """tsr_sim_op plus the op's PERF CH_RD_BEATS[c] increments: ((code, [beats per channel]), info)."""
        before = [self.perf(5 + c) for c in range(num_ch)]
        code, info = self.op(words)
        return (code, [self.perf(5 + c) - before[c] for c in range(num_ch)]), info


async def clock_reset(dut, cycles: int = 5):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_check(dut, words, timeout=100):
    """Present words to the bench's checker class (chk_*), return (code, info, ext int, cmd int)."""
    from common import isa
    dut.chk_req_insn.value = isa.insn_int(words)
    dut.chk_req_valid.value = 1
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        if int(dut.chk_rsp_valid.value):
            r = (int(dut.chk_rsp_code.value), int(dut.chk_rsp_info.value), int(dut.chk_rsp_ext.value),
                 int(dut.chk_rsp_cmd.value))
            await RisingEdge(dut.clk)
            dut.chk_req_valid.value = 0
            await RisingEdge(dut.clk)        # the class sees req_valid low before the next request
            return r
    raise TimeoutError("checker did not answer")


def f32b(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rand_fp16_exact(r: random.Random) -> int:
    """An f32 bit pattern that is exactly representable in fp16 (QACT D values are)."""
    h = r.getrandbits(16)
    e = (h >> 10) & 0x1F
    if e == 0x1F:
        h &= ~(1 << 14)            # avoid inf / NaN most of the time
    s, e, m = h >> 15, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        if m == 0:
            return s << 31
        # subnormal fp16 -> normal f32
        k = m.bit_length() - 1
        return (s << 31) | ((k - 24 + 127) << 23) | ((m << (23 - k)) & 0x7FFFFF)
    return (s << 31) | ((e - 15 + 127) << 23) | (m << 13)


def rand_scale(r: random.Random, p_any: float = 0.15) -> int:
    """Mostly fp16-exact scales; sometimes arbitrary f32 (rounding, overflow, NaN, subnormal)."""
    if r.random() < p_any:
        return r.choice([r.getrandbits(32), 0x7F800000, 0xFF800000, 0x7FC00001, 0x00000001, 0x80000000,
                         0x477FF000, 0x477FF001, 0x33000000, 0x33000001, 0x387FC000, r.getrandbits(32)])
    return rand_fp16_exact(r)

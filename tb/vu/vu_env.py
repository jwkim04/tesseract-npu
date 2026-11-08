"""tb/vu environment: the simulator's VU as the oracle and the cocotb drivers of tb/vu/hdl/tsr_vu_tb.sv.

Oracle (docs/design/06-software-verification.md "SW" §10.3: benches compare through the simulator's shared
library): one tsr_sim instance (libtsr_sim.so) with the VU and ATT capabilities. Every op is checked with the
handler's check (tsr_sim_accept_op + check; it must pass) and executed with the handler's exec on the
simulator's own SPM image and s bank (sw/sim/tsr_vu.c ex_*, tsr_at.c ex_* on tsr_ref_vu.c / tsr_ref_at.c),
in push order: the sequential semantics the RTL must reproduce with its accept-ahead overlap.
RTL: the same SPM image is loaded through the bench port (client 0 of tsr_spm); after a batch the lines of
every write extent are read back and compared byte for byte, and the s bank is compared.
"""
from __future__ import annotations

import ctypes
import random
import struct
import sys
from pathlib import Path

from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import issue_sim as S       # noqa: E402

M32 = 0xFFFFFFFF
EXEC_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))


class Oracle:
    def __init__(self, spm_bytes: int, num_ch: int, caps: int = 0x21):
        self.cfg = S.config(num_ch=num_ch, t_max=8, spm_bytes=spm_bytes, caps=caps)
        self.sim = S.Sim(self.cfg, (8, 4, 4, 16, 0, 4))
        L = self.L = S.lib()
        L.tsr_sim_spm.restype = ctypes.POINTER(ctypes.c_uint8)
        L.tsr_sim_spm.argtypes = [ctypes.c_void_p]
        L.tsr_sim_sreg.restype = ctypes.POINTER(ctypes.c_uint32)
        L.tsr_sim_sreg.argtypes = [ctypes.c_void_p]
        p = L.tsr_sim_spm(self.sim.sim)
        self.spm = (ctypes.c_uint8 * spm_bytes).from_address(ctypes.addressof(p.contents))
        q = L.tsr_sim_sreg(self.sim.sim)
        self.sreg = (ctypes.c_uint32 * 16).from_address(ctypes.addressof(q.contents))
        self.spm_bytes = spm_bytes

    def close(self):
        self.sim.close()

    def check(self, w):
        return self.sim.check([x & M32 for x in w])

    def exec(self, w):
        h = self.L.tsr_sim_accept_op(self.sim.sim, w[0] & 0xFF)
        assert h, "opcode %#x not accepted" % (w[0] & 0xFF)
        EXEC_FN(h.contents.exec)(self.sim.sim, (ctypes.c_uint32 * 8)(*[x & M32 for x in w]))

    def line(self, i: int) -> bytes:
        return bytes(self.spm[64 * i: 64 * i + 64])

    def set_line(self, i: int, b: bytes):
        ctypes.memmove(ctypes.addressof(self.spm) + 64 * i, b, 64)

    def words(self, line: int, n: int):
        return list(struct.unpack_from("<%dI" % n, bytes(self.spm[64 * line: 64 * line + 4 * n])))


class Rtl:
    """Drivers of tsr_vu_tb: bench port (load / read back), command push, done counting."""

    def __init__(self, dut, lines: int):
        self.dut = dut
        self.lines = lines
        self.done = 0
        self.pushed = 0
        self.cycles = 0
        self.hazard_cycles = 0
        self.wait_cycles = 0
        self.done_log = []           # (cycle, hazard cycles, SPM-wait cycles so far) at every done pulse (timing benches)

    async def reset(self, noise=False, seed=1):
        d = self.dut
        d.rst_n.value = 0
        d.cmd_valid.value = 0
        d.cmd.value = 0
        d.t_rd_valid.value = 0
        d.t_wr_valid.value = 0
        d.t_rd_line.value = 0
        d.t_wr_line.value = 0
        d.t_wr_data.value = 0
        d.noise_en.value = 1 if noise else 0
        d.noise_seed.value = seed & M32
        for _ in range(6):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        await RisingEdge(d.clk)

    async def monitor(self):
        d = self.dut
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            self.cycles += 1
            if int(d.perf_hazard.value):
                self.hazard_cycles += 1
            if int(d.perf_spm_wait.value):
                self.wait_cycles += 1
            if int(d.done.value):
                self.done += 1
                self.done_log.append((self.cycles, self.hazard_cycles, self.wait_cycles))

    LIMIT = 200_000      # cycles any single handshake or wait may take before the bench declares a hang

    async def write_lines(self, items):
        """items: iterable of (line, 64 bytes). Inputs change at the falling edge; ready (a grant flop
        through tsr_spm_cport's bank select) is read in the same cycle, the transfer is at the next
        rising edge."""
        d = self.dut
        await FallingEdge(d.clk)
        for (ln, b) in items:
            d.t_wr_valid.value = 1
            d.t_wr_line.value = ln
            d.t_wr_data.value = int.from_bytes(b, "little")
            k = 0
            while True:
                await ReadOnly()
                rdy = int(d.t_wr_ready.value)
                await FallingEdge(d.clk)
                if rdy:
                    break
                k += 1
                if k > self.LIMIT:
                    raise TimeoutError("bench write of line %d not granted in %d cycles" % (ln, self.LIMIT))
        d.t_wr_valid.value = 0

    async def read_lines(self, lines):
        d = self.dut
        lines = list(lines)
        out = []
        i = 0
        k = 0
        await FallingEdge(d.clk)
        while len(out) < len(lines):
            k += 1
            if k > self.LIMIT + 8 * len(lines):
                raise TimeoutError("bench read of %d lines incomplete after %d cycles" % (len(lines), k))
            if i < len(lines):
                d.t_rd_valid.value = 1
                d.t_rd_line.value = lines[i]
            else:
                d.t_rd_valid.value = 0
            await ReadOnly()
            if i < len(lines) and int(d.t_rd_ready.value):
                i += 1                                   # accepted at the next rising edge
            if int(d.t_rsp_valid.value):
                out.append(int(d.t_rsp_data.value).to_bytes(64, "little"))
            await FallingEdge(d.clk)
        d.t_rd_valid.value = 0
        return dict(zip(lines, out))

    async def push(self, w):
        d = self.dut
        v = 0
        for i, x in enumerate(w):
            v |= (x & M32) << (32 * i)
        await FallingEdge(d.clk)
        d.cmd.value = v
        d.cmd_valid.value = 1
        k = 0
        while True:
            await ReadOnly()
            rdy = int(d.cmd_ready.value)
            await FallingEdge(d.clk)
            if rdy:
                break
            k += 1
            if k > self.LIMIT:
                raise TimeoutError("command not accepted in %d cycles (%d of %d ops done)" % (self.LIMIT, self.done,
                                                                                            self.pushed))
        d.cmd_valid.value = 0
        self.pushed += 1

    async def wait_done(self, n, limit=200_000):
        d = self.dut
        k = 0
        while self.done < n:
            await RisingEdge(d.clk)
            k += 1
            if k > limit:
                raise TimeoutError("%d of %d ops done after %d cycles" % (self.done, n, limit))

    def sreg(self):
        v = int(self.dut.sreg.value)
        return [(v >> (32 * i)) & M32 for i in range(16)]


def rnd_f32(r: random.Random) -> int:
    """Edge-heavy f32 bit patterns."""
    x = r.random()
    if x < 0.08:
        return r.choice([0, 0x80000000, 1, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F800000, 0xFF800000, 0x7FC00000,
                         0xFFC00001, 0x7F7FFFFF, 0x3F800000, 0xBF800000, 0x41200000, 0xC1200000, 0x80C00000])
    if x < 0.2:
        return r.getrandbits(32)
    if x < 0.6:
        return struct.unpack("<I", struct.pack("<f", r.gauss(0.0, 3.0)))[0]
    if x < 0.8:
        return struct.unpack("<I", struct.pack("<f", r.uniform(-12.0, 12.0)))[0]
    return (r.getrandbits(1) << 31) | (r.randint(100, 150) << 23) | r.getrandbits(23)


def rnd_line(r: random.Random) -> bytes:
    return struct.pack("<16I", *[rnd_f32(r) for _ in range(16)])

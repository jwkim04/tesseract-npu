"""tb/at test_at_kvst — KVST 0x14 (AT §5.1, §8.3) on tsr_dma's KVST engine, end to end with the AT checker
class, against the simulator's ex_kvst (tsr_ref_kvst) for the same SPM and DRAM contents.

DUT: tb/at/hdl/tsr_tb_at_dma.sv = tsr_chk_at + tsr_dma (HAS_KVST = 1) + one tsr_mem_arb per channel;
the SPM port on a Python model (at_tb.SpmPort, random ready), the memory ports on common.mem_model.MemModel
(random latency, backpressure and write orderings).
For every op: the instruction goes through tsr_chk_at (its code / info must equal tsr_sim_op's); a passing op
then runs on the DMA with the checker's payload (cmd_xc = 1). After `done`:
  - every byte of the plane (B .. B + LG GB on every channel, random before the op) equals the simulator's;
  - the beats written are exactly the op's meta and quant beats (AT §4.3 / §4.4), the beats read exactly the
    partially covered meta beats (read-modify-write), and perf_kvst_rmw pulsed once per such beat (KVST_RMW);
  - the SPM is unchanged and was only read inside the source vectors;
  - LD / ST still work between KVSTs (a few plain LD / ST ops through the same DMA, checked against the
    simulator), and the memory protocol checks of MemModel hold (bursts, 4 KiB rule, M5 orderings).
Stimulus: decode (n_rows 1) and prefill (up to 64 rows across groups and channels) shapes of the target
models (E = 128 Qwen2.5-0.5B, 1024 Qwen2.5-32B, 512 / 2048 Gemma 4 12B global / SWA), small and odd E/64 (rows
sharing / spanning chunks: E = 64, 192, 960, 1088), V groups written fully and partly, random strides,
arbitrary f32 scales (fp16 rounding, overflow, NaN), random checker failures (OPERAND, ranges, reserved bits).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import isa                                    # noqa: E402
from common.mem_model import MemModel                     # noqa: E402
from common.runner import REPO, run, rtl_sources          # noqa: E402

import at_isa as A                                        # noqa: E402
import at_tb as B                                         # noqa: E402

M32 = 0xFFFFFFFF


def cfg_env():
    return json.loads(os.environ.get("TSR_KVST_CFG", "{}"))


def plane_sets(kv, E, C, LG, B0, c0, n):
    """(written beats per channel, rmw meta beats per channel) of a KVST (AT §4.3 / §4.4)."""
    nb, e64, GB = E // 32, E // 64, 17 * (E // 64)
    wr = [set() for _ in range(C)]
    cov = {}                                     # (ch, meta beat) -> covered block slots
    for cell in range(c0, c0 + n):
        u, r = cell >> 4, cell & 15
        ch, l = u % C, u // C
        if kv == 0:
            for b in range(nb):
                beta = r * nb + b
                kap, o = beta >> 5, beta & 31
                meta = B0 + l * GB + 17 * kap
                wr[ch].add(meta)
                wr[ch].add(meta + 1 + (o >> 1))
                cov.setdefault((ch, meta), set()).add(o)
        else:
            for s in range(e64):
                chunk = B0 + (s * LG + l) * 17
                wr[ch].add(chunk)
                wr[ch].add(chunk + 1 + r)
                cov.setdefault((ch, chunk), set()).update((2 * r, 2 * r + 1))
    rmw = [set() for _ in range(C)]
    for (ch, meta), slots in cov.items():
        if len(slots) != 32:
            rmw[ch].add(meta)
    return wr, rmw


class Bench:
    def __init__(self, dut, cfg, seed):
        self.dut, self.cfg = dut, cfg
        self.r = random.Random(seed)
        self.C = cfg["num_ch"]
        self.spm_bytes = cfg["spm_bytes"]
        self.lines = self.spm_bytes // 64
        self.limit = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.sim = B.SimAt(num_ch=self.C, t_max=cfg["t_max"], k_max=cfg["k_max"], spm_bytes=self.spm_bytes,
                           ch_bytes=cfg["ch_bytes"], addr_beat_w=cfg["addr_beat_w"])
        self.spm = bytearray(self.r.getrandbits(8) for _ in range(self.spm_bytes))
        self.sim.spm_write(0, self.spm)
        self.port = B.SpmPort(dut, self.spm, prefix="spm_", rd_lat=cfg["spm_rd_lat"], rnd=self.r,
                              p_rd=cfg.get("p_rd", 0.8), p_wr=0.8)
        self.mem = MemModel(dut, num_ch=self.C, addr_w=cfg["addr_beat_w"], rd_outstanding=cfg["rd_out"],
                            limit_beats=self.limit, seed=seed, rd_lat=tuple(cfg.get("rd_lat", (1, 60))))
        self.rmw_pulses = 0
        self.stats = dict(ops=0, passed=0, rmw=0, beats=0)

    async def perf_mon(self):
        while True:
            await RisingEdge(self.dut.clk)
            await FallingEdge(self.dut.clk)
            if int(self.dut.perf_kvst_rmw.value):
                self.rmw_pulses += 1

    def load_plane(self, B0, beats):
        for c in range(self.C):
            data = self.r.randbytes(64 * beats)
            self.mem.load(c, 64 * B0, data)
            self.sim.dram_write(c, 64 * B0, data)

    def fill_src(self, src, e64, n, stride):
        """QA(A32, E) source vectors: random QS bytes, D words mostly fp16-exact (QACT-like)."""
        qa = e64 + (e64 + 7) // 8
        st = stride if stride else qa
        for i in range(n):
            base = (src + i * st) * 64
            img = bytearray(self.r.randbytes(qa * 64))
            for b in range(2 * e64):
                v = B.rand_scale(self.r, self.cfg.get("p_any", 0.15))
                img[e64 * 64 + 4 * b:e64 * 64 + 4 * b + 4] = v.to_bytes(4, "little")
            self.spm[base:base + qa * 64] = img
        self.sim.spm_write(0, self.spm)

    async def dma(self, cmd=0, xc=0, xcmd=0, timeout=400000):
        dut = self.dut
        await FallingEdge(dut.clk)
        while not int(dut.cmd_ready.value):
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
        dut.cmd.value = cmd
        dut.cmd_xc.value = xc
        dut.xcmd.value = xcmd
        dut.cmd_valid.value = 1
        await RisingEdge(dut.clk)
        dut.cmd_valid.value = 0
        for n in range(timeout):
            await FallingEdge(dut.clk)
            if int(dut.done.value):
                return n
            await RisingEdge(dut.clk)
        kv = dut.u_dma.g_kv.u_kv
        dut._log.error("op %s mem errors %s stats %s", [hex(x) for x in getattr(self, "last_w", [])], self.mem.errors[:3],
                       self.mem.stats())
        raise TimeoutError("DMA command did not finish (KVST st %s, wq_v %s, dq rp/wp %s/%s, wr_pend %s, rb %s/%s, kq %s/%s)" % (
            kv.st_q.value, kv.wq_v_q.value, kv.dq_rp_q.value, kv.dq_wp_q.value, kv.wr_pend_q.value,
            kv.rb_cnt_q.value, kv.rb_inf_q.value, kv.kq_wp_q.value, kv.kq_inf_q.value))

    async def kvst(self, kv, e64, LG, c0, n, stride=None, src=None, B0=None, expect_pass=True):
        r = self.r
        E = 64 * e64
        qa = e64 + (e64 + 7) // 8
        if stride is None:
            stride = r.choice([0, 0, qa, qa + r.randint(1, 5)])
        st = stride if stride else qa
        if expect_pass and (n - 1) * st + qa > self.lines:       # fit the bench's SPM
            stride, st = 0, qa
            n = min(n, (self.lines - qa) // qa + 1)
            c0 = min(c0, 16 * self.C * LG - n)
        ext = (n - 1) * st + qa
        if src is None:
            src = r.randint(0, max(0, self.lines - ext))
        plane = LG * 17 * e64
        if B0 is None:
            B0 = r.randint(0, max(0, min(self.limit, 1 << 20) - plane))
        w = A.enc_kvst(kv, n, src, B0, e64, LG, c0, stride)
        if ext <= self.lines and B0 + plane <= self.limit:
            self.fill_src(src, e64, n, stride)
            self.load_plane(B0, plane)
        code, info, _, xcmd = await B.run_check(self.dut, w)
        scode, sinfo = self.sim.op(w)
        self.stats["ops"] += 1
        assert (code, info) == (scode, sinfo), "KVST %s: checker (%d, %#x), simulator (%d, %#x)" % (
            [hex(x) for x in w], code, info, scode, sinfo)
        if code:
            assert not expect_pass, "KVST %s failed with code %d (info %#x)" % ([hex(x) for x in w], code, info)
            return
        self.stats["passed"] += 1
        self.mem.clear_tracking()
        self.port.clear_log()
        p0 = self.rmw_pulses
        self.last_w = w
        await self.dma(xc=1, xcmd=xcmd)
        for _ in range(4):
            await RisingEdge(self.dut.clk)
        self.mem.check(idle=True)
        assert not self.port.errors, self.port.errors[:3]
        # plane bytes == simulator's
        for c in range(self.C):
            got = self.mem.read_bytes(c, 64 * B0, 64 * plane)
            exp = self.sim.dram_read(c, 64 * B0, 64 * plane)
            if got != exp:
                for k in range(plane):
                    if got[64 * k:64 * k + 64] != exp[64 * k:64 * k + 64]:
                        raise AssertionError("KVST %s: channel %d beat %d (plane +%d) differs:\n RTL %s\n sim %s" % (
                            [hex(x) for x in w], c, B0 + k, k, got[64 * k:64 * k + 64].hex(),
                            exp[64 * k:64 * k + 64].hex()))
        wr, rmw = plane_sets(kv, E, self.C, LG, B0, c0, n)
        for c in range(self.C):
            assert set(self.mem.written[c]) == wr[c], "KVST %s: channel %d wrote %s, expected %s" % (
                [hex(x) for x in w], c, sorted(set(self.mem.written[c]) ^ wr[c])[:8], sorted(wr[c])[:8])
            assert set(self.mem.read[c]) == rmw[c], "KVST %s: channel %d read %s, expected the RMW beats %s" % (
                [hex(x) for x in w], c, sorted(self.mem.read[c])[:8], sorted(rmw[c])[:8])
        nrmw = sum(len(x) for x in rmw)
        assert self.rmw_pulses - p0 == nrmw, "perf_kvst_rmw %d, expected %d" % (self.rmw_pulses - p0, nrmw)
        assert not self.port.writes, "KVST wrote the SPM"
        lo, hi = src, src + ext
        bad = [x for x in self.port.reads if not lo <= x < hi]
        assert not bad, "SPM reads outside the sources: %s" % bad[:4]
        assert bytes(self.spm) == self.sim.spm_read(0, self.spm_bytes)
        self.stats["rmw"] += nrmw
        self.stats["beats"] += sum(len(x) for x in wr)

    async def ldst(self):
        """A plain LD or ST between KVSTs (the LD / ST engine after KVST)."""
        r = self.r
        n = r.randint(1, 80)
        ch = r.randrange(self.C)
        line = r.randint(0, self.lines - n)
        beat = r.randint(0, min(self.limit, 1 << 20) - n)
        st = r.randint(0, 1)
        data = r.randbytes(64 * n)
        if st:
            self.spm[64 * line:64 * (line + n)] = data
            self.sim.spm_write(64 * line, data)
        else:
            self.mem.load(ch, 64 * beat, data)
            self.sim.dram_write(ch, 64 * beat, data)
        w = [(isa.OP_ST if st else isa.OP_LD) | (ch << 8), line, beat, n, 0, 0, 0, 0]
        assert self.sim.op(w)[0] == 0
        cmd = isa.DMA_CMD_T.pack(is_st=st, ch=ch, spm_line=line, dram_beat=beat, n_lines=n)
        await self.dma(cmd=cmd)
        for _ in range(4):
            await RisingEdge(self.dut.clk)
        if st:
            assert self.mem.read_bytes(ch, 64 * beat, 64 * n) == self.sim.dram_read(ch, 64 * beat, 64 * n)
        else:
            assert bytes(self.spm[64 * line:64 * (line + n)]) == self.sim.spm_read(64 * line, 64 * n)

    def rand_shape(self):
        r = self.r
        e64 = r.choice([1, 2, 3, 4, 8, 15, 16, 17, 32, 2, 1, r.randint(1, 40)])
        LG = r.choice([1, 2, 3, 5, 8])
        cells = 16 * self.C * LG
        n = r.choice([1, 1, 2, 3, 15, 16, 17, 31, 33, 64, r.randint(1, 64)])
        n = min(n, cells)
        c0 = r.choice([0, 16 * r.randint(0, cells // 16 - 1), r.randint(0, cells - n), cells - n])
        c0 = min(c0, cells - n)
        return e64, LG, c0, n


async def kvst_run(dut, n_ops):
    cfg = cfg_env()
    b = Bench(dut, cfg, int(os.environ.get("TSR_SEED", "1")))
    dut.chk_req_valid.value = 0
    dut.chk_req_insn.value = 0
    dut.cmd_valid.value = 0
    dut.cmd.value = 0
    dut.cmd_xc.value = 0
    dut.xcmd.value = 0
    b.port.start()
    b.mem.start()
    await B.clock_reset(dut)
    cocotb.start_soon(b.perf_mon())
    r = b.r
    # directed: the target models' shapes, decode and prefill, both planes
    lg = 3
    for kv in (0, 1):
        for e64 in (2, 16, 8, 32, 1, 3, 15, 17):            # E 128, 1024, 512, 2048, 64, 192, 960, 1088
            cells = 16 * b.C * lg
            for (c0, n) in ((0, 1), (5, 1), (15, 1), (16, 1), (cells - 1, 1), (0, 16), (16, 16), (3, 64 if cells >= 67 else 1),
                            (cells - min(cells, 64), min(cells, 64)), (7, min(cells - 7, 40))):
                await b.kvst(kv, e64, lg, c0, n)
        await b.ldst()
    # random, with checker failures mixed in
    for i in range(n_ops):
        e64, LG, c0, n = b.rand_shape()
        if r.random() < 0.12:
            # an op that fails at the checker (OPERAND / range): nothing may move
            c0 = 16 * b.C * LG - n + r.randint(1, 3)
            await b.kvst(r.randint(0, 1), e64, LG, c0, n, expect_pass=False)
        else:
            await b.kvst(r.randint(0, 1), e64, LG, c0, n)
        if r.random() < 0.08:
            await b.ldst()
    b.port.stop()
    b.mem.stop()
    b.sim.close()
    dut._log.info("KVST: %s", b.stats)


@cocotb.test()
async def at_kvst(dut):
    await kvst_run(dut, cfg_env().get("n_ops", 150))


def _run(name, cfg):
    params = dict(NUM_CH=cfg["num_ch"], T_MAX=cfg["t_max"], K_MAX=cfg["k_max"], SPM_BYTES=cfg["spm_bytes"],
                  CH_BYTES="64'd%d" % cfg["ch_bytes"], SPM_RD_LAT=cfg["spm_rd_lat"], ADDR_BEAT_W=cfg["addr_beat_w"],
                  RD_OUTSTANDING=cfg["rd_out"])
    run("tsr_tb_at_dma", __name__, parameters=params, name="at_kvst_" + name,
        sources=rtl_sources() + [HERE / "hdl" / "tsr_tb_at_dma.sv"], extra_env={"TSR_KVST_CFG": json.dumps(cfg)})


def test_at_kvst_u200():
    """U200: 4 channels, 256 KiB SPM (the bench's), 16 GiB channels, SPM_RD_LAT 4."""
    _run("u200", dict(num_ch=4, t_max=8, k_max=32768, spm_bytes=256 << 10, ch_bytes=16 << 30, spm_rd_lat=4,
                      addr_beat_w=28, rd_out=16, n_ops=150))


def test_at_kvst_odd():
    """3 channels (division by a non-power of two), SPM_RD_LAT 1, RD_OUTSTANDING 2, slow memory."""
    _run("odd", dict(num_ch=3, t_max=3, k_max=4096, spm_bytes=128 << 10, ch_bytes=1 << 30, spm_rd_lat=1,
                     addr_beat_w=22, rd_out=2, n_ops=120, rd_lat=[20, 200]))


def test_at_kvst_one():
    """1 channel (every group in channel 0), SPM_RD_LAT 2, arbitrary scales often."""
    _run("one", dict(num_ch=1, t_max=1, k_max=2048, spm_bytes=64 << 10, ch_bytes=1 << 28, spm_rd_lat=2,
                     addr_beat_w=22, rd_out=16, n_ops=100, p_any=0.5))

"""Self-test of the shared testbench infrastructure (tb/common).

pytest functions (test_*):
  test_ref_*         libtsr_ref.so wrapper against spec vectors and the pure-Python layouts
  test_isa_structs   struct pack/unpack round trips, unit-instruction encodings
  test_loop_sim      cocotb on tb/common/hdl/tsr_tb_loop.sv: struct layouts vs tsr_pkg.sv,
                     MemModel under random legal traffic from a Python master (data, ordering,
                     outstanding limit, write tracking) and its protocol checks on deliberately
                     illegal traffic
M3 (docs/rtl-m3.md §10.1):
  test_ref_m3_*      tsr_ref wrappers (repack / unrepack, row_location, terms, dequant, scales)
  test_mvm_model_*   golden cluster model: tagged beats vs the reference row walk (all terms
                     recomputed from the beat stream == tsr_term_*), struct packing, integer
                     widths, actbuf / broadcast images, gather jobs (every y byte written once,
                     any job order and ACC interleaving == the ex_mvm write loop), the symmetric
                     weight layout == tsr_row_location, whole MVMs vs the ISA simulator's op-level
                     API (tsr_sim_op through libtsr_sim.so, CMake target tsr_sim_shared)
  test_pipes_sim     cocotb on tb/common/hdl/tsr_tb_pipes.sv: tsr_vpipe / tsr_rpipe at DEPTH 0/1/3
cocotb tests (loop_*, pipes_*) are not collected by pytest directly; the runner selects them.
Planned with the trace format and the dispatcher (docs/design SW §10.3): trace loading and
replay comparison tests, loop_csr (CsrDriver), the core skeleton and pycore trace tests.
test_mvm_model_vs_simulator issues the LD and MVM ops one at a time through tsr_sim_op.
"""
from __future__ import annotations

import collections
import ctypes
import os
import random
import struct
from collections import deque

import cocotb
import pytest
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from common import isa, mvm_model
from common.mem_model import MemModel, ProtocolError
from common.runner import REPO, run
from common.tbutil import rng as tb_rng
from common.tbutil import start_clock_and_reset
from common.tsr_ref import f32_bits, ref

LOOP_SV = REPO / "tb" / "common" / "hdl" / "tsr_tb_loop.sv"
PIPES_SV = REPO / "tb" / "common" / "hdl" / "tsr_tb_pipes.sv"
M32 = 0xFFFFFFFF

# =============================================================================================
# Pure-Python tests
# =============================================================================================


def test_ref_fp_vectors():
    r = ref()
    # FTZ window (spec, Implementation clarifications, "FTZ decision point") — also required in the RTL fp block tests.
    assert r.fmul(0x3F7FFFFF, 0x00800000) == 0x00800000
    assert r.fmul(0x3F1FA0BA, 0x00CD4708) == 0x00800000
    assert r.fdiv(0x00FFFFFF, f32_bits(2.0)) == 0x00800000
    assert r.fmul(0x3F7FFFFE, 0x00800000) == 0x00000000
    assert r.fmul(0x7F800000, 0) == isa.CANON_NAN
    assert r.fdiv(0, 0) == isa.CANON_NAN
    assert r.fdiv(f32_bits(1.0), 0x80000000) == 0xFF800000
    assert r.frecip(f32_bits(4.0)) == f32_bits(0.25)
    assert r.daz(0x807FFFFF) == 0x80000000
    assert r.daz(0x7F800001) == 0x7F800001          # sNaN passes the ctypes boundary unchanged
    assert r.fp16_to_f32(0x0001) == f32_bits(2.0 ** -24)
    assert r.fp16_to_f32(0x7C01) == 0x7FC02000      # NaN: payload kept, quiet bit set
    assert r.f32_to_fp16(f32_bits(65520.0)) == 0x7C00
    assert r.f32_to_fp16(0x7F800001) == 0x7E00
    assert r.i32_to_f32(16777217) == f32_bits(16777216.0)
    assert r.i32_to_f32(-1) == f32_bits(-1.0)


def test_ref_q8_rounding():
    r = ref()
    cases = {2.5: (3, 2), -2.5: (-3, -2), 0.5: (1, 0), 1.5: (2, 2), 127.5: (127, 127),
             -128.5: (-128, -128), 300.0: (127, 127), -1e30: (-128, -128), 0.49999997: (0, 0)}
    for x, (away, even) in cases.items():
        assert r.q8_round_away(f32_bits(x)) == away, x
        assert r.q8_round_even(f32_bits(x)) == even, x
    for nan in (0x7FC00000, 0xFFC00001, 0x7F800001):
        assert r.q8_round_away(nan) == 0 and r.q8_round_even(nan) == 0
    assert r.q8_round_away(0xFF800000) == -128 and r.q8_round_even(0x7F800000) == 127


def test_ref_qa_layout_matches_python():
    r = ref()
    for kind in (isa.QA_A32, isa.QA_A256):
        bs = isa.qa_block_size(kind)
        for K in list(range(bs, 4096 + 1, bs)) + [27648, 32768, 65280 // bs * bs]:
            assert r.qa_lines(kind, K) == isa.qa_lines(kind, K), (kind, K)
            assert r.qa_d_lines(kind, K) == isa.qa_d_lines(kind, K), (kind, K)
            assert r.qa_bs_lines(kind, K) == isa.qa_bs_lines(kind, K), (kind, K)
    assert r.qa_lines(isa.QA_A256, 27648) == 493          # spec example
    assert r.qa_lines(isa.QA_A256, 32768) == 584 and r.qa_lines(isa.QA_A32, 32768) == 576
    assert r.row_beats(isa.FMT_Q4_K, 27648) == 243 and r.row_beats(isa.FMT_Q8_0, 5120) == 85


def test_ref_qact_examples():
    r = ref()
    # Spec QA section example: x = {127, 2.5, -0.5, 0, ...} -> A32 q = {127, 3, -1, 0}.
    x = [f32_bits(127.0), f32_bits(2.5), f32_bits(-0.5)] + [0] * 29
    qa = r.qact(isa.QA_A32, x, 32)
    v = r.qa_split(isa.QA_A32, 32, qa)
    assert v["qs"][:4] == [127, 3, -1, 0] and v["d"] == [f32_bits(1.0)]
    assert len(qa) == 128 and qa[32:64] == bytes(32) and qa[68:] == bytes(60)   # QS / D tails are zero
    # A256 all-zero block: d = +0, q = 0, BS = 0.
    qa = r.qact(isa.QA_A256, [0] * 256, 256)
    assert qa == bytes(len(qa))
    # A256 first-occurrence rule: max = -4 (first of the |4| ties) -> iscale = +31.75, q = {-127, 127, 32}.
    x = [f32_bits(-4.0), f32_bits(4.0)] + [f32_bits(1.0)] * 254
    v = r.qa_split(isa.QA_A256, 256, r.qact(isa.QA_A256, x, 256))
    assert v["qs"][:3] == [-127, 127, 32] and v["d"][0] == r.frecip(r.fdiv(f32_bits(-127.0), f32_bits(-4.0)))
    assert v["bs"][0] == -127 + 127 + 32 * 14


def test_isa_structs():
    rnd = random.Random(5)
    for sd in isa.STRUCTS:
        for _ in range(50):
            vals = {n: rnd.getrandbits(w) for n, w in sd.fields}
            assert sd.unpack(sd.pack(**vals)) == vals
    assert isa.DMA_CMD_T.width == 101 and isa.QACT_CMD_T.width == 133
    assert isa.MVM_CMD_T.width == 294
    assert isa.MVM_CCMD_T.width == 137 and isa.MVM_BCLINE_T.width == 527
    assert isa.MVM_FCMD_T.width == 137 and isa.MVM_BEAT_T.width == 1175
    assert isa.unpack_array(isa.pack_array([-1, 5, -128, 127], 8), 4, 8, signed=True) == [-1, 5, -128, 127]
    assert isa.insn_bytes(isa.enc_mvm(1, 2, 3, 4, 5, 256, 3, 7, acc=1)) == struct.pack(
        "<8I", 0x1120, 2, 3, 4, 5, 256 | (3 << 16), 7, 0)              # x16 = 0 (EX §7.1)
    assert isa.insn_bytes(isa.enc_ld(3, 2, 7, 1)) == struct.pack("<8I", 0x0310, 2, 7, 1, 0, 0, 0, 0)
    assert isa.w0_fields(isa.w0(isa.OP_ST, sub=2, flags=1, x16=0xBEEF)) == dict(
        opcode=isa.OP_ST, sub=2, flags=1, x16=0xBEEF)
    assert isa.insn_bytes(isa.enc_qact(1, 2, 3, 256, 1, 0)) == struct.pack(
        "<8I", isa.w0(isa.OP_QACT, sub=1), 2, 3, 256, 1, 0, 0, 0)


# =============================================================================================
# Simulation tests (pytest side)
# =============================================================================================
LOOP_PARAMS = {"NUM_CH": 2, "ADDR_BEAT_W": 20}


def test_loop_sim():
    run("tsr_tb_loop", __name__, parameters=LOOP_PARAMS, extra_sources=[LOOP_SV],
        testcase=["loop_structs", "loop_mem_random", "loop_mem_violations"])


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def loop_structs(dut):
    """isa.py struct layouts == tsr_pkg.sv (widths and field positions)."""
    await start_clock_and_reset(dut)
    assert int(dut.bits_dma_cmd.value) == isa.DMA_CMD_T.width
    assert int(dut.bits_qact_cmd.value) == isa.QACT_CMD_T.width
    assert int(dut.bits_mvm_cmd.value) == isa.MVM_CMD_T.width
    assert int(dut.qa_lines_a256_27648.value) == isa.qa_lines(isa.QA_A256, 27648)
    assert int(dut.row_beats_q4k_27648.value) == ref().row_beats(isa.FMT_Q4_K, 27648)
    rnd = tb_rng(1)
    for _ in range(20):
        d = {n: rnd.getrandbits(w) for n, w in isa.DMA_CMD_T.fields}
        q = {n: rnd.getrandbits(w) for n, w in isa.QACT_CMD_T.fields}
        m = {n: rnd.getrandbits(w) for n, w in isa.MVM_CMD_T.fields}
        dut.dma_in.value = isa.DMA_CMD_T.pack(**d)
        dut.qact_in.value = isa.QACT_CMD_T.pack(**q)
        dut.mvm_in.value = isa.MVM_CMD_T.pack(**m)
        await FallingEdge(dut.clk)
        for n, _ in isa.DMA_CMD_T.fields:
            assert int(getattr(dut, "dma_" + n).value) == d[n], n
        for n, _ in isa.QACT_CMD_T.fields:
            assert int(getattr(dut, "qact_" + n).value) == q[n], n
        assert int(dut.mvm_n.value) == m["n"] and int(dut.mvm_row_beats.value) == m["row_beats"]
    # M3 structs: widths, field positions, package helpers (docs/rtl-m3.md §3, §5)
    assert int(dut.bits_ccmd.value) == isa.MVM_CCMD_T.width
    assert int(dut.bits_bcline.value) == isa.MVM_BCLINE_T.width
    assert int(dut.bits_fcmd.value) == isa.MVM_FCMD_T.width
    assert int(dut.bits_beat.value) == isa.MVM_BEAT_T.width
    assert int(dut.actbuf_qs_words_32768.value) == 256
    assert int(dut.actbuf_dbs_lines_32768.value) == 72          # A256: 8 D + 64 BS lines
    assert int(dut.actbuf_dbs_lines_256.value) == 2
    for _ in range(40):
        c = {n: rnd.getrandbits(w) for n, w in isa.MVM_CCMD_T.fields}
        bl = {n: rnd.getrandbits(w) for n, w in isa.MVM_BCLINE_T.fields}
        fc = {n: rnd.getrandbits(w) for n, w in isa.MVM_FCMD_T.fields}
        bt = {n: rnd.getrandbits(w) for n, w in isa.MVM_BEAT_T.fields}
        c["d_lines"] = rnd.randrange(16)                       # channel index for the rows probe
        c["rows_per_ch"] = rnd.choice([1, 10, 16, rnd.getrandbits(20) + 1, 0xFFFFFFFF])
        c["n"] = rnd.choice([c["n"], rnd.randrange(64), c["rows_per_ch"] * c["d_lines"] + rnd.randrange(40)]) & M32
        dut.ccmd_in.value = isa.MVM_CCMD_T.pack(**c)
        dut.bcline_in.value = isa.MVM_BCLINE_T.pack(**bl)
        dut.fcmd_in.value = isa.MVM_FCMD_T.pack(**fc)
        dut.beat_in.value = isa.MVM_BEAT_T.pack(**bt)
        await FallingEdge(dut.clk)
        assert int(dut.ccmd_k.value) == c["k"] and int(dut.ccmd_rows_per_ch.value) == c["rows_per_ch"]
        assert int(dut.ccmd_row_beats.value) == c["row_beats"] and int(dut.ccmd_d_lines.value) == c["d_lines"]
        first = c["d_lines"] * c["rows_per_ch"]
        assert int(dut.rows_c_probe.value) == max(0, min(c["rows_per_ch"], c["n"] - first)), c
        for n in ("last", "tok", "dbs", "addr"):
            assert int(getattr(dut, "bcline_" + n).value) == bl[n], n
        assert int(dut.bcline_data0.value) == bl["data"] & M32
        for n in ("row_beats", "rows", "beats"):
            assert int(getattr(dut, "fcmd_" + n).value) == fc[n], n
        for n in ("last", "row_last", "blk", "two", "half", "d_a", "d_b"):
            assert int(getattr(dut, "beat_" + n).value) == bt[n], n
        assert int(dut.beat_sc3.value) == (bt["sc"] >> 18) & 63 and int(dut.beat_m5.value) == (bt["m"] >> 30) & 63
        assert int(dut.beat_w0.value) == bt["w"] & 255 and int(dut.beat_w127.value) == bt["w"] >> 1016


class LoopMaster:
    """Random legal traffic on the src_* side of tsr_tb_loop (all channels, several ids)."""

    def __init__(self, dut, num_ch, rnd, limit, rd_outstanding, shadow):
        self.dut, self.n, self.rnd, self.limit, self.rdo = dut, num_ch, rnd, limit, rd_outstanding
        self.shadow = shadow                    # per channel: beat -> int (expected memory)
        self.rd_todo = [deque() for _ in range(num_ch)]     # (addr, len, id)
        self.wr_todo = [deque() for _ in range(num_ch)]     # (addr, len, id, [beats])
        self.rd_out = [{} for _ in range(num_ch)]           # id -> deque of [addr, len, got]
        self.rd_cnt = [0] * num_ch
        self.wdat = [deque() for _ in range(num_ch)]        # data beats (data, last) ready to send
        self.wr_out = [{} for _ in range(num_ch)]           # id -> count of responses expected
        self.max_out = [0] * num_ch
        self.beats_read = 0
        self.rsp_ready_p = 0.7

    def plan_read(self, c, addr, ln, id_):
        self.rd_todo[c].append((addr, ln, id_))

    def plan_write(self, c, addr, ln, id_):
        beats = [self.rnd.getrandbits(512) for _ in range(ln)]
        self.wr_todo[c].append((addr, ln, id_, beats))
        for i, b in enumerate(beats):
            self.wdat[c].append((b, int(i == ln - 1)))

    def busy(self):
        return any(self.rd_todo[c] or self.rd_cnt[c] or self.wr_todo[c] or self.wdat[c]
                   or sum(self.wr_out[c].values()) for c in range(self.n))

    async def run(self, max_cycles=200000):
        d, n = self.dut, self.n
        cur_rq = [None] * n
        cur_wq = [None] * n
        cur_wd = [None] * n
        rsp_rdy = [0] * n
        wrsp_rdy = [0] * n
        for _ in range(max_cycles):
            # drive (after the rising edge)
            rqv = rqa = rql = rqi = wqv = wqa = wql = wqi = wdv = wdd = wdl = rr = wr = 0
            for c in range(n):
                if cur_rq[c] is None and self.rd_todo[c] and self.rd_cnt[c] < self.rdo and self.rnd.random() < 0.8:
                    cur_rq[c] = self.rd_todo[c].popleft()
                if cur_rq[c] is not None:
                    a, ln, i = cur_rq[c]
                    rqv |= 1 << c
                    rqa |= a << (c * 20)
                    rql |= ln << (c * 7)
                    rqi |= i << (c * 4)
                if cur_wq[c] is None and self.wr_todo[c] and self.rnd.random() < 0.8:
                    cur_wq[c] = self.wr_todo[c].popleft()
                if cur_wq[c] is not None:
                    a, ln, i, _ = cur_wq[c]
                    wqv |= 1 << c
                    wqa |= a << (c * 20)
                    wql |= ln << (c * 7)
                    wqi |= i << (c * 4)
                if cur_wd[c] is None and self.wdat[c] and self.rnd.random() < 0.8:
                    cur_wd[c] = self.wdat[c].popleft()
                if cur_wd[c] is not None:
                    wdv |= 1 << c
                    wdd |= cur_wd[c][0] << (c * 512)
                    wdl |= cur_wd[c][1] << c
                rsp_rdy[c] = int(self.rnd.random() < self.rsp_ready_p)
                wrsp_rdy[c] = int(self.rnd.random() < 0.7)
                rr |= rsp_rdy[c] << c
                wr |= wrsp_rdy[c] << c
            d.src_rd_req_valid.value, d.src_rd_req_addr.value = rqv, rqa
            d.src_rd_req_len.value, d.src_rd_req_id.value = rql, rqi
            d.src_wr_req_valid.value, d.src_wr_req_addr.value = wqv, wqa
            d.src_wr_req_len.value, d.src_wr_req_id.value = wql, wqi
            d.src_wr_dat_valid.value, d.src_wr_dat_data.value, d.src_wr_dat_last.value = wdv, wdd, wdl
            d.src_rd_rsp_ready.value, d.src_wr_rsp_ready.value = rr, wr
            await FallingEdge(d.clk)
            s_rqr = int(d.src_rd_req_ready.value)
            s_wqr = int(d.src_wr_req_ready.value)
            s_wdr = int(d.src_wr_dat_ready.value)
            s_rsv = int(d.src_rd_rsp_valid.value)
            s_wsv = int(d.src_wr_rsp_valid.value)
            s_rsd = int(d.src_rd_rsp_data.value) if s_rsv else 0
            s_rsi = int(d.src_rd_rsp_id.value) if s_rsv else 0
            s_rsl = int(d.src_rd_rsp_last.value) if s_rsv else 0
            s_wsi = int(d.src_wr_rsp_id.value) if s_wsv else 0
            await RisingEdge(d.clk)
            for c in range(n):
                if cur_rq[c] is not None and (s_rqr >> c) & 1:
                    a, ln, i = cur_rq[c]
                    self.rd_out[c].setdefault(i, deque()).append([a, ln, 0])
                    self.rd_cnt[c] += 1
                    self.max_out[c] = max(self.max_out[c], self.rd_cnt[c])
                    cur_rq[c] = None
                if (s_rsv >> c) & 1 and rsp_rdy[c]:
                    i = (s_rsi >> (4 * c)) & 15
                    q = self.rd_out[c].get(i)
                    assert q, "ch%d: read beat for id %d with nothing outstanding" % (c, i)
                    b = q[0]
                    data = (s_rsd >> (512 * c)) & ((1 << 512) - 1)
                    exp = self.shadow[c].get(b[0] + b[2], 0)
                    assert data == exp, "ch%d beat 0x%x: data mismatch" % (c, b[0] + b[2])
                    b[2] += 1
                    last = (s_rsl >> c) & 1
                    assert last == int(b[2] == b[1]), "ch%d: rd_rsp_last wrong" % c
                    self.beats_read += 1
                    if b[2] == b[1]:
                        q.popleft()
                        self.rd_cnt[c] -= 1
                if cur_wq[c] is not None and (s_wqr >> c) & 1:
                    a, ln, i, beats = cur_wq[c]
                    for k, v in enumerate(beats):
                        self.shadow[c][a + k] = v
                    self.wr_out[c][i] = self.wr_out[c].get(i, 0) + 1
                    cur_wq[c] = None
                if cur_wd[c] is not None and (s_wdr >> c) & 1:
                    cur_wd[c] = None
                if (s_wsv >> c) & 1 and wrsp_rdy[c]:
                    i = (s_wsi >> (4 * c)) & 15
                    assert self.wr_out[c].get(i, 0) > 0, "ch%d: unexpected write response id %d" % (c, i)
                    self.wr_out[c][i] -= 1
            if not self.busy() and all(x is None for x in cur_rq + cur_wq + cur_wd):
                d.src_rd_req_valid.value = 0
                d.src_wr_req_valid.value = 0
                d.src_wr_dat_valid.value = 0
                return
        raise TimeoutError("loop master did not finish")


def _rand_burst(rnd, limit, region_lo, region_hi):
    while True:
        ln = rnd.choice([1, 1, 2, 3, 7, 16, 31, 63, 64, rnd.randint(1, 64)])
        a = rnd.randrange(region_lo, region_hi - ln)
        if (a % 64) + ln <= 64 and a + ln <= limit:
            return a, ln


@cocotb.test()
async def loop_mem_random(dut):
    num_ch, limit, rdo = LOOP_PARAMS["NUM_CH"], 1 << LOOP_PARAMS["ADDR_BEAT_W"], 4
    await start_clock_and_reset(dut)
    rnd = tb_rng(3)
    mem = MemModel(dut, num_ch=num_ch, addr_w=LOOP_PARAMS["ADDR_BEAT_W"], rd_outstanding=rdo,
                   limit_beats=limit, seed=rnd.getrandbits(31), rd_lat=(0, 25), wr_lat=(0, 25))
    shadow = [dict() for _ in range(num_ch)]
    # Preload a read-only region [0, 4096) with random data, including the last beat (limit - 1).
    for c in range(num_ch):
        blob = bytes(rnd.getrandbits(8) for _ in range(64 * 256))
        mem.load(c, 0, blob)
        mem.load(c, (limit - 64) * 64, blob[:64 * 64])
        for i in range(256):
            shadow[c][i] = int.from_bytes(blob[64 * i:64 * i + 64], "little")
        for i in range(64):
            shadow[c][limit - 64 + i] = shadow[c][i]
    mem.start()
    m = LoopMaster(dut, num_ch, rnd, limit, rdo, shadow)
    written = [set() for _ in range(num_ch)]
    for c in range(num_ch):
        for _ in range(60):
            a, ln = _rand_burst(rnd, limit, 0, 256)
            m.plan_read(c, a, ln, rnd.randrange(16))
        m.plan_read(c, limit - 64, 64, 3)        # last block of the channel
        for _ in range(30):
            a, ln = _rand_burst(rnd, limit, 4096, 8192)
            m.plan_write(c, a, ln, rnd.randrange(16))
            written[c].update(range(a, a + ln))
    await m.run()
    await ClockCycles(dut.clk, 5)
    mem.check(idle=True)
    # Phase 2: read back the written area (shadow holds the last write per beat).
    for c in range(num_ch):
        for a in range(4096, 8192, 64):
            m.plan_read(c, a, 64, 5)
    await m.run()
    await ClockCycles(dut.clk, 2)        # let the model process the last handshake edge
    mem.check(idle=True)
    assert [mem.written[c] for c in range(num_ch)] == written
    assert max(m.max_out) <= rdo and max(m.max_out) >= 2, m.max_out   # the limit was exercised
    st = mem.stats()
    dut._log.info("loop_mem_random: %d beats read, stats %s", m.beats_read, st)


async def _issue_read(dut, addr, ln, id_=0):
    """Offer one read request on channel 0 and hold it until the handshake."""
    await RisingEdge(dut.clk)
    dut.src_rd_req_addr.value = addr
    dut.src_rd_req_len.value = ln
    dut.src_rd_req_id.value = id_
    dut.src_rd_req_valid.value = 1
    while True:
        await FallingEdge(dut.clk)
        rdy = int(dut.src_rd_req_ready.value) & 1
        await RisingEdge(dut.clk)
        if rdy:
            break
    dut.src_rd_req_valid.value = 0


@cocotb.test()
async def loop_mem_violations(dut):
    """Each illegal request is reported by the model (strict=False collects instead of raising)."""
    num_ch, aw = LOOP_PARAMS["NUM_CH"], LOOP_PARAMS["ADDR_BEAT_W"]
    limit = 1 << aw
    await start_clock_and_reset(dut)
    dut.src_rd_rsp_ready.value = (1 << num_ch) - 1
    cases = [("4 KiB", 60, 5, "crosses a 4 KiB"), ("len 0", 0, 0, "length 0 not in"),
             ("len 65", 0, 65, "length 65 not in"), ("at limit", limit - 2, 2, None),
             ("beyond", limit - 1, 2, "beyond the channel limit")]
    for name, addr, ln, msg in cases:
        mem = MemModel(dut, num_ch=num_ch, addr_w=aw, rd_outstanding=4, limit_beats=limit, seed=7,
                       strict=False, p_stall=0.0).start()
        await _issue_read(dut, addr, ln)
        await ClockCycles(dut.clk, 120)
        mem.stop()
        if msg is None:
            assert not mem.errors, (name, mem.errors)
            assert mem.idle(), name
        else:
            assert any(msg in e for e in mem.errors), (name, mem.errors)
    # Outstanding limit: 5 requests while rd_outstanding = 4 and responses are held back.
    mem = MemModel(dut, num_ch=num_ch, addr_w=aw, rd_outstanding=4, limit_beats=limit, seed=7,
                   strict=False, p_stall=0.0, rd_lat=(1000, 1000)).start()
    for i in range(5):
        await _issue_read(dut, i, 1)
    await ClockCycles(dut.clk, 2)        # let the model process the last handshake edge
    mem.stop()
    assert any("RD_OUTSTANDING" in e for e in mem.errors), mem.errors
    assert sum(1 for e in mem.errors if "RD_OUTSTANDING" in e) == 1, mem.errors
    # Valid dropped before the handshake (memory never ready).
    mem = MemModel(dut, num_ch=num_ch, addr_w=aw, rd_outstanding=4, limit_beats=limit, seed=7,
                   strict=False, p_rd_req_ready=0.0, p_stall=0.0).start()
    await RisingEdge(dut.clk)
    dut.src_rd_req_valid.value = 1
    dut.src_rd_req_addr.value = 0
    dut.src_rd_req_len.value = 1
    await ClockCycles(dut.clk, 3)
    dut.src_rd_req_valid.value = 0
    await ClockCycles(dut.clk, 3)
    mem.stop()
    assert any("dropped before the handshake" in e for e in mem.errors), mem.errors
    with pytest.raises(ProtocolError):
        mem.check()


# =============================================================================================
# M3: reference wrappers, golden MVM model, pipes
# =============================================================================================
def test_ref_m3_wrappers():
    r = ref()
    rnd = random.Random(31)
    assert (r.fmt_block_size(0), r.fmt_block_size(1)) == (32, 256)
    assert (r.fmt_chunk_blocks(0), r.fmt_chunk_blocks(1)) == (32, 4)
    assert (r.fmt_quant_bytes(0), r.fmt_quant_bytes(1)) == (32, 128)
    assert (r.fmt_ggml_block_bytes(0), r.fmt_ggml_block_bytes(1)) == (34, 144)
    assert r.fmt_valid(1) and not r.fmt_valid(2)
    assert r.fmt_k_valid(1, 256, 32768) and not r.fmt_k_valid(1, 288, 32768) and not r.fmt_k_valid(0, 32800, 32768)
    assert (r.qa_kind_for_fmt(0), r.qa_kind_for_fmt(1)) == (0, 1)
    for fmt in (0, 1):
        bs = 256 if fmt else 32
        for K in [bs * k for k in range(1, 40)] + [27648 // bs * bs, 65280 // bs * bs]:
            assert r.row_beats(fmt, K) == mvm_model.row_beats(fmt, K), (fmt, K)
        for K in (bs, 5 * bs, 1024, 1024 + bs, 4096 + 3 * bs):
            blocks = mvm_model.rand_blocks(fmt, K, rnd, special=0.3)
            row = r.repack_row(fmt, blocks, K)
            assert len(row) == 64 * r.row_beats(fmt, K)
            assert r.unrepack_row(fmt, row, K) == blocks
            deq = r.dequant_row(fmt, row, K)
            assert len(deq) == K
            if fmt == 0:   # Q8_0: y = fmul(d, f32(q)) for the first block
                d = r.fp16_to_f32(blocks[0] | (blocks[1] << 8))
                assert deq[:32] == [r.fmul(d, r.i32_to_f32(mvm_model.s8(q))) for q in blocks[2:34]]
    # spec examples (Row size) and the symmetric allocation (Symmetric allocation)
    assert r.row_beats(1, 27648) == 243 and r.row_beats(0, 5120) == 85 and r.row_beats(0, 896) == 15
    for N, C in ((40, 4), (5, 4), (1, 16), (17, 3), (100, 7)):
        R = r.rows_per_ch(N, C)
        assert R == -(-N // C) and sum(mvm_model.rows_c(N, C, c) for c in range(C)) == N
        for n in range(N):
            ch, off = r.row_location(1, N, 512, C, n)
            assert ch == n // R and off == (n - ch * R) * r.row_beats(1, 512)
        assert r.tensor_ch_beats(0, N, 1024, C) == R * 17
    # 6-bit scales: every byte value in every position
    for j in range(8):
        for v in range(256):
            q = bytes([(v + 17 * i) & 255 for i in range(12)])
            assert r.get_scale_min_k4(j, q) == mvm_model.get_scale_min_k4(j, q)
    # term wrappers == the definition from values (Q8_0 block 0)
    K = 64
    row = mvm_model.rand_row(0, K, rnd)
    qa = mvm_model.rand_qa(0, K, rnd, "raw")
    v = mvm_model.QaView.of(0, K, qa)
    d0 = r.fp16_to_f32(row[0] | (row[1] << 8))
    isum = sum(mvm_model.s8(row[64 + i]) * v.qs[i] for i in range(32))
    assert r.term_q8_0(row, K, 0, qa) == mvm_model.term_q8_0_values(d0, v.d[0], isum)
    p = [rnd.getrandbits(32) for _ in range(16)]
    assert r.tree16(p) == mvm_model.tree16(p)


def test_mvm_model_cluster():
    """front_beats / Cluster against libtsr_ref over formats, partial chunks, raw data and special
    values (Cluster.check recomputes every term from the beat stream: == tsr_term_*, and every
    row result: tree16(partials) == tsr_dot_row)."""
    rnd = random.Random(32)
    n_beats = 0
    for fmt in (0, 1):
        bs = 256 if fmt else 32
        Ks = ([32, 64, 96, 992, 1024, 1056, 2048 + 96, 3104] if fmt == 0 else [256, 512, 768, 1024, 1280, 2304, 4352])
        for K in Ks:
            for mode in ("real", "raw"):
                rows = [mvm_model.rand_row(fmt, K, rnd, special=0.15, raw=(mode == "raw")) for _ in range(3)]
                qas = [mvm_model.rand_qa(fmt, K, rnd, rnd.choice(["qact", "raw"]), special=0.15)
                       for _ in range(rnd.randint(1, 3))]
                cl = mvm_model.Cluster(fmt, K, rows, qas)
                nq = sum(mvm_model.quant_beats(fmt, n) for n in mvm_model.row_chunks(fmt, K))
                assert len(cl.beats) == 3 * nq
                assert sum(b["row_last"] for b in cl.beats) == 3 and cl.beats[-1]["last"] == 1
                assert sum(b["last"] for b in cl.beats) == 1
                for b in cl.beats:
                    assert mvm_model.unpack_beat(mvm_model.pack_beat(b)) == mvm_model.beat_fields(b)
                    if fmt == 0:
                        assert b["w"][64:] == [0] * 64 and b["blk"] % 2 == 0 and b["half"] == 0
                        assert b["two"] or (b["w"][32:64] == [0] * 32 and b["d_b"] == 0)
                    else:
                        assert all(0 <= x <= 15 for x in b["w"]) and b["two"] == 0
                    for t, view in enumerate(cl.views):          # integer widths (§7.6)
                        m = mvm_model.beat_math(fmt, b, view)
                        assert all(abs(x) <= 1 << 19 for x in m["S"])
                        if fmt == 1:
                            assert abs(m["msum"]) < 1 << 25
                n_beats += len(cl.beats)
                for ri in range(len(rows)):
                    for t in range(len(qas)):
                        assert all(abs(x.isum) < 1 << 25 for x in cl.terms[ri][t])
                ab = cl.actbuf(t_max=4)
                assert ab[0]["qs_lines"] == (K + 63) // 64 and len(ab[0]["dbs"]) == isa.qa_lines(fmt, K) - (K + 63) // 64
                bl = mvm_model.bc_lines(fmt, K, qas)
                assert len(bl) == len(qas) * isa.qa_lines(fmt, K) and bl[-1]["last"] == 1
                assert sum(x["last"] for x in bl) == 1
    # msum extreme (arbitrary BS): 8 * 63 * 2 * -32768 needs 26 bits (docs/rtl-m3.md §7.6)
    b = dict(blk=0, half=1, sc=[63] * 4, m=[63] * 8, w=[15] * 128, two=0)
    view = mvm_model.QaView(1, 256, b"", qs=[-128] * 256, d=[0x3F800000], bs=[-32768] * 16)
    m = mvm_model.beat_math(1, b, view)
    assert m["msum"] == -8 * 63 * 65536 and -(1 << 25) <= m["msum"] < (1 << 25)
    assert n_beats > 1000


def _apply_jobs(spm, jobs, ys, acc, rnd):
    """Partial-line writes of docs/rtl-m3.md §6.3 in a random interleaving: every ACC job reads
    its line at some point before its own write; reads and writes of different jobs interleave
    arbitrarily. Returns the byte-write counts per SPM byte."""
    r = ref()
    pend = list(range(len(jobs)))
    rnd.shuffle(pend)
    reads = {}
    count = collections.Counter()
    while pend:
        k = pend[rnd.randrange(len(pend))]
        j = jobs[k]
        if acc and k not in reads:
            reads[k] = bytes(spm[64 * j.line:64 * j.line + 64])
            continue
        pend.remove(k)
        for ln in range(16):
            o = 64 * j.line + 4 * ln
            if j.lo <= ln <= j.hi:
                n = 16 * (j.rows[0] // 16) + ln
                v = ys[j.tok][n]
                if acc:
                    v = r.fadd(int.from_bytes(reads[k][4 * ln:4 * ln + 4], "little"), v)
                spm[o:o + 4] = v.to_bytes(4, "little")
            elif j.tail and ln > j.hi:
                spm[o:o + 4] = bytes(4)
            else:
                continue
            for b_ in range(o, o + 4):
                count[b_] += 1
    return count


def test_mvm_model_gather_jobs():
    """The partial-line / byte-enable / ACC scheme of docs/rtl-m3.md §6.3 equals the ex_mvm write
    loop for any job order and read/write interleaving, writes every y byte exactly once and
    nothing outside the y lines; Job.be() covers exactly the written bytes."""
    rnd = random.Random(33)
    for it in range(150):
        nch = rnd.choice([1, 2, 3, 4, 7, 16])
        N = rnd.choice([0, 1, 2, nch - 1 or 1, nch, nch + 1, 15, 16, 17, 33, 40, rnd.randint(1, 300)])
        T = rnd.randint(1, 8)
        yl = (N + 15) // 16
        stride = rnd.choice([0, yl + rnd.randint(0, 3)]) if T > 1 else rnd.choice([0, yl + 2])
        ds = stride or yl
        dst = rnd.randrange(0, 8)
        lines = dst + (T - 1) * ds + yl + 8
        acc = bool(rnd.getrandbits(1))
        old = bytearray(rnd.randbytes(64 * lines))
        ys = [[mvm_model.rand_f32(rnd, 0.1) for _ in range(N)] for _ in range(T)]
        want = bytearray(old)
        mvm_model.apply_mvm(want, dst, ds, N, T, acc, ys)
        jobs = mvm_model.gather_jobs(N, nch, T, dst, ds)
        got = bytearray(old)
        cnt = _apply_jobs(got, jobs, ys, acc, rnd)
        assert got == want, (it, N, nch, T, stride, acc)
        ybytes = {64 * (dst + t * ds) + i for t in range(T) for i in range(64 * yl)}
        assert set(cnt) == ybytes and all(v == 1 for v in cnt.values()), (it, N, nch, T)
        for j in jobs:
            be = j.be()
            assert be and all(((be >> (4 * ln)) & 0xF) in (0, 0xF) for ln in range(16))
        assert sum(bin(j.be()).count("1") for j in jobs) == len(ybytes)


def test_mvm_model_layout():
    """layout_channels (the symmetric allocation the block tests load into DRAM) == tsr_row_location
    of libtsr_ref, for both fmts and N edge cases (whole MVMs: test_mvm_model_vs_simulator)."""
    rnd = random.Random(34)
    for it in range(24):
        fmt = it % 2
        bs = 256 if fmt else 32
        K = rnd.choice([bs, 3 * bs, 1024, 1024 + bs] if fmt else [32, 96, 1024, 1056, 2080])
        N = rnd.choice([0, 1, 2, 3, 4, 17, 40, rnd.randint(1, 80)])
        nch = rnd.choice([1, 3, 4])
        rows = [mvm_model.rand_row(fmt, K, rnd, special=0.05, raw=rnd.random() < 0.3) for _ in range(N)]
        per_ch = mvm_model.layout_channels(fmt, K, rows, nch)
        rbytes = 64 * mvm_model.row_beats(fmt, K)
        for n in range(N):                     # symmetric allocation == tsr_row_location
            ch, off = ref().row_location(fmt, N, K, nch, n)
            assert per_ch[ch][64 * off:64 * off + rbytes] == rows[n], (n, ch, off)
        assert sum(len(d) for d in per_ch) == N * rbytes
        for c in range(nch):
            assert len(per_ch[c]) == mvm_model.rows_c(N, nch, c) * rbytes, (it, c)


SIM_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_sim.so"


class _TsrConfig(ctypes.Structure):       # tsr_config (sw/sim/include/tsr_isa.h)
    _fields_ = [(n, ctypes.c_uint32) for n in ("num_ch", "beat_bits", "lanes", "t_max", "k_max", "spm_bytes",
                                               "vl", "nsreg", "rd_outstanding")] + [
               ("ch_bytes", ctypes.c_uint64), ("addr_beat_w", ctypes.c_uint32), ("caps", ctypes.c_uint32)]


class _OpSim:
    """The simulator's op-level API (sw/sim/include/tsr_sim.h) through ctypes: libtsr_sim.so of the
    CMake target tsr_sim_shared, $TSR_SIM_LIB or else build/dev/sw/sim/libtsr_sim.so."""

    def __init__(self, cfg: dict):
        path = os.environ.get("TSR_SIM_LIB", str(SIM_LIB))
        if not os.path.exists(path):
            raise FileNotFoundError("%s not found: build it with `cmake --build build/dev --target tsr_sim_shared "
                                    "-j4` or set TSR_SIM_LIB" % path)
        L = self.lib = ctypes.CDLL(path)
        vp, u32, u64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        for name, res, args in (("tsr_sim_create", vp, [ctypes.POINTER(_TsrConfig), ctypes.c_char_p]),
                                ("tsr_sim_destroy", None, [vp]),
                                ("tsr_sim_dram", vp, [vp, u32]),
                                ("tsr_sim_spm", vp, [vp]),
                                ("tsr_sim_op", u32, [vp, ctypes.POINTER(u32), ctypes.POINTER(u32)]),
                                ("tsr_sim_perf", u64, [vp, u32])):
            getattr(L, name).restype = res
            getattr(L, name).argtypes = args
        self.cfg = cfg
        self.sim = L.tsr_sim_create(ctypes.byref(_TsrConfig(**cfg)), None)
        assert self.sim, cfg

    def write(self, ch: int, off: int, data: bytes):
        ctypes.memmove(self.lib.tsr_sim_dram(self.sim, ch) + off, data, len(data))

    def spm(self) -> bytes:
        return ctypes.string_at(self.lib.tsr_sim_spm(self.sim), self.cfg["spm_bytes"])

    def op(self, words) -> int:
        info = ctypes.c_uint32(0)
        return self.lib.tsr_sim_op(self.sim, (ctypes.c_uint32 * 8)(*words), ctypes.byref(info))

    def perf(self, idx: int) -> int:
        return self.lib.tsr_sim_perf(self.sim, idx)

    def close(self):
        self.lib.tsr_sim_destroy(self.sim)
        self.sim = None


def test_mvm_model_vs_simulator():
    """mvm_model.mvm_reference (the golden MVM of the tb/mvm benches) == the ISA simulator executing
    the same MVM through tsr_sim_op (whole SPM compared), for both fmts, ACC, T, strides and N edge
    cases; per-channel weight beats (PERF CH_RD_BEATS) == rows_c x ROW_BEATS."""
    rnd = random.Random(34)
    cfg = dict(isa.U200, t_max=8, num_ch=3, spm_bytes=256 * 1024)
    sim = _OpSim(cfg)
    woff, io = isa.REGION_WEIGHT_OFF // 64, isa.REGION_IO_OFF // 64
    n_runs = 0
    for it in range(24):
        fmt = it % 2
        bs = 256 if fmt else 32
        K = rnd.choice([bs, 3 * bs, 1024, 1024 + bs] if fmt else [32, 96, 1024, 1056, 2080])
        N = rnd.choice([0, 1, 2, 3, 4, 17, 40, rnd.randint(1, 80)])
        T = rnd.randint(1, 4)
        acc = it % 3 == 0
        rows = [mvm_model.rand_row(fmt, K, rnd, special=0.05, raw=rnd.random() < 0.3) for _ in range(N)]
        qas = [mvm_model.rand_qa(fmt, K, rnd, rnd.choice(["qact", "raw"]), special=0.05) for _ in range(T)]
        per_ch = mvm_model.layout_channels(fmt, K, rows, 3)
        ql = isa.qa_lines(fmt, K)
        yl = (N + 15) // 16
        stride = rnd.choice([0, yl + 1]) if T > 1 else 0
        ds = stride or yl
        x, dst = 64, 64 + T * ql + rnd.randint(0, 5)
        y_old = rnd.randbytes(64 * ((T - 1) * ds + yl))
        for c, d in enumerate(per_ch):
            sim.write(c, woff * 64, d)
        sim.write(0, io * 64, b"".join(qas))
        sim.write(0, (io + 4096) * 64, y_old)
        assert sim.op(isa.enc_ld(0, x, io, T * ql)) == 0
        if y_old:
            assert sim.op(isa.enc_ld(0, dst, io + 4096, len(y_old) // 64)) == 0
        spm = bytearray(sim.spm())
        beats0 = [sim.perf(isa.perf_ch_rd_beats(c)) for c in range(3)]
        err = sim.op(isa.enc_mvm(fmt, dst, x, woff, N, K, T, stride, acc=acc))
        assert err == 0, (it, isa.ERR_NAMES.get(err, err))
        read = lambda c, b, n: per_ch[c][64 * (b - woff):64 * (b - woff + n)]  # noqa: E731
        mvm_model.mvm_reference(spm, read, fmt=fmt, acc=acc, dst=dst, x=x, w_off=woff, N=N, K=K, T=T,
                                dst_stride=stride, num_ch=3)
        assert bytes(spm) == sim.spm(), (it, fmt, K, N, T, acc)
        rb = mvm_model.row_beats(fmt, K)
        for c in range(3):
            got = sim.perf(isa.perf_ch_rd_beats(c)) - beats0[c]
            assert got == mvm_model.rows_c(N, 3, c) * rb, (it, c, got)
        n_runs += 1
    assert n_runs == 24
    sim.close()


def test_pipes_sim():
    run("tsr_tb_pipes", __name__, extra_sources=[PIPES_SV], testcase=["pipes_vpipe", "pipes_rpipe"])


PIPE_DEPTHS = (0, 1, 3)


@cocotb.test()
async def pipes_vpipe(dut):
    """tsr_vpipe: outputs == inputs delayed by exactly DEPTH cycles (valid reset to 0)."""
    await start_clock_and_reset(dut)
    rnd = tb_rng(41)
    hist = []
    for _ in range(400):
        vv = [rnd.getrandbits(2) for _ in range(3)]
        dd = [rnd.getrandbits(40) for _ in range(3)]
        dut.v_in_valid.value = sum(v << (2 * i) for i, v in enumerate(vv))
        dut.v_in_data.value = sum(d << (40 * i) for i, d in enumerate(dd))
        hist.append((vv, dd))
        await FallingEdge(dut.clk)
        ov, od = int(dut.v_out_valid.value), int(dut.v_out_data.value)
        for i, D in enumerate(PIPE_DEPTHS):
            want_v, want_d = (hist[-1 - D][0][i], hist[-1 - D][1][i]) if len(hist) > D else (0, None)
            assert (ov >> (2 * i)) & 3 == want_v, (i, D)
            if want_v:
                assert (od >> (40 * i)) & ((1 << 40) - 1) == want_d, (i, D)
        await RisingEdge(dut.clk)


@cocotb.test()
async def pipes_rpipe(dut):
    """tsr_rpipe: in-order lossless transfer under random valid / ready, R2 on the output, full
    throughput and latency DEPTH when both sides are always ready."""
    await start_clock_and_reset(dut)
    rnd = tb_rng(42)
    M40 = (1 << 40) - 1
    sent = [[] for _ in range(3)]
    got = [[] for _ in range(3)]
    cur = [None] * 3
    pend = [None] * 3          # output offered but not taken at the last sample
    for phase, (p_v, p_r, cycles) in enumerate(((0.6, 0.5, 600), (1.0, 1.0, 200), (0.9, 0.2, 300))):
        first_out = [None] * 3
        n0 = [len(g) for g in got]
        for cyc in range(cycles):
            sv = sd = rr = 0
            for i in range(3):
                if cur[i] is None and rnd.random() < p_v:
                    cur[i] = rnd.getrandbits(40)
                if cur[i] is not None:
                    sv |= 1 << i
                    sd |= cur[i] << (40 * i)
                rr |= int(rnd.random() < p_r) << i
            dut.r_s_valid.value, dut.r_s_data.value, dut.r_m_ready.value = sv, sd, rr
            await FallingEdge(dut.clk)
            s_rdy, m_v, m_d = int(dut.r_s_ready.value), int(dut.r_m_valid.value), int(dut.r_m_data.value)
            for i in range(3):
                mv, md = (m_v >> i) & 1, (m_d >> (40 * i)) & M40
                if pend[i] is not None:
                    assert mv and md == pend[i], "R2 violated on rpipe %d" % i
                pend[i] = None
                if mv:
                    if (rr >> i) & 1:
                        got[i].append(md)
                        if first_out[i] is None:
                            first_out[i] = cyc
                    else:
                        pend[i] = md
                if cur[i] is not None and (s_rdy >> i) & 1:
                    sent[i].append(cur[i])
                    cur[i] = None
            await RisingEdge(dut.clk)
        if phase == 1:   # always valid and ready: latency DEPTH, then one item per cycle
            for i, D in enumerate(PIPE_DEPTHS):
                assert first_out[i] is not None and first_out[i] <= D + 2, (i, first_out)
                assert len(got[i]) - n0[i] >= cycles - D - 3, (i, len(got[i]) - n0[i], cycles)
    for i in range(3):          # finish the item being offered, then drain
        if cur[i] is not None:
            sent[i].append(cur[i])
    dut.r_s_valid.value = sum(1 << i for i in range(3) if cur[i] is not None)
    dut.r_s_data.value = sum(cur[i] << (40 * i) for i in range(3) if cur[i] is not None)
    dut.r_m_ready.value = 7
    offered = [cur[i] is not None for i in range(3)]
    for _ in range(12):
        await FallingEdge(dut.clk)
        s_rdy, m_v, m_d = int(dut.r_s_ready.value), int(dut.r_m_valid.value), int(dut.r_m_data.value)
        for i in range(3):
            if (m_v >> i) & 1:
                got[i].append((m_d >> (40 * i)) & M40)
        await RisingEdge(dut.clk)
        for i in range(3):
            if offered[i] and (s_rdy >> i) & 1:
                offered[i] = False
        dut.r_s_valid.value = sum(1 << i for i in range(3) if offered[i])
    for i in range(3):
        assert got[i] == sent[i], (i, len(sent[i]), len(got[i]))
        assert len(got[i]) > 400

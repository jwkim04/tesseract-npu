"""Self-test of the shared testbench infrastructure (tb/common).

pytest functions (test_*):
  test_ref_*         libtsr_ref.so wrapper against spec vectors and the pure-Python layouts
  test_isa_structs   struct pack/unpack round trips
  test_traces_load   every trace under traces/ loads and is self-consistent
  test_replay_compare_detects_mismatches  replay.compare() flags each kind of mismatch
  test_loop_sim      cocotb on tb/common/hdl/tsr_tb_loop.sv: struct layouts vs tsr_pkg.sv,
                     CsrDriver, MemModel under random legal traffic from a Python master
                     (data, ordering, outstanding limit, write tracking) and its protocol checks
                     on deliberately illegal traffic
  test_core_skeleton_sim  cocotb on tsr_core (whatever its state): elaborates with the default and
                     a reduced configuration, MemModel/CsrDriver attach to its boundary
  test_pycore_traces cocotb on tsr_tb_loop: pycore.PyCore executes traces/m2 programs, the MVM
                     traces of traces/gen and (when present) traces/m3 through the memory-port
                     protocol against MemModel; replay.compare must pass (proves the trace-replay
                     harness end to end, incl. the MVM weight streams, and re-checks the traces)
M3 (docs/rtl-m3.md §10.1):
  test_ref_m3_*      tsr_ref wrappers (repack / unrepack, row_location, terms, dequant, scales)
  test_mvm_model_*   golden cluster model: tagged beats vs the reference row walk (all terms
                     recomputed from the beat stream == tsr_term_*), struct packing, integer
                     widths, actbuf / broadcast images, gather jobs (every y byte written once,
                     any job order and ACC interleaving == the ex_mvm write loop), whole MVMs vs
                     the ISA simulator
  test_proggen_*     M2 stream unchanged (digest), m3 programs on the simulator: executed MVMs of
                     both fmts with ACC and T > 1, no undefined overlaps
  test_simlib_keyed_build  content-keyed, race-free build of the simulator library
  test_pipes_sim     cocotb on tb/common/hdl/tsr_tb_pipes.sv: tsr_vpipe / tsr_rpipe at DEPTH 0/1/3
cocotb tests (loop_*, core_*, pipes_*) are not collected by pytest directly; the runner selects them.
"""
from __future__ import annotations

import collections
import random
import struct
from collections import deque
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from common import isa, mvm_model, replay, trace
from common.csr_driver import CsrDriver, CsrTimeout
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
    # FTZ window (spec 구현 명확화 "FTZ 판정 시점") — also required in the RTL fp block tests.
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
    assert isa.FETCH_INSN_T.width == 289 and isa.MVM_CMD_T.width == 294
    assert isa.MVM_CCMD_T.width == 137 and isa.MVM_BCLINE_T.width == 527
    assert isa.MVM_FCMD_T.width == 137 and isa.MVM_BEAT_T.width == 1175
    assert isa.unpack_array(isa.pack_array([-1, 5, -128, 127], 8), 4, 8, signed=True) == [-1, 5, -128, 127]
    assert isa.insn_bytes(isa.enc_mvm(1, 2, 3, 4, 5, 256, 3, 7, acc=1, ra=1, rb=2, rc=3)) == struct.pack(
        "<8I", isa.w0(isa.OP_MVM, sub=1, flags=1, ra=1, rb=2, rc=3), 2, 3, 4, 5, 256 | (3 << 16), 7, 0)
    assert isa.insn_bytes(isa.enc_qact(1, 2, 3, 256, 1, 0)) == struct.pack(
        "<8I", isa.w0(isa.OP_QACT, sub=1), 2, 3, 256, 1, 0, 0, 0)


def test_traces_load():
    roots = [REPO / "traces" / "m2", REPO / "traces" / "gen", REPO / "traces" / "m3"]
    n = 0
    for root in roots:
        if not root.exists():
            continue
        for tr in trace.load_all(root):
            n += 1
            assert len(tr.cmd) == 32 * tr.cmd_count
            for r in tr.init:   # cmd.bin agrees with init where they overlap (ch0)
                c = tr.cmd_region()
                if r.ch == 0 and c is not None:
                    lo, hi = max(r.offset, c.offset), min(r.offset + r.size, c.offset + c.size)
                    if lo < hi:
                        assert r.data[lo - r.offset:hi - r.offset] == c.data[lo - c.offset:hi - c.offset]
            assert tr.core_parameters()["NUM_CH"] == tr.num_ch
            assert (tr.status == isa.STATUS_DONE) == (tr.err_code == 0)
            if root.name == "m2":
                assert tr.opcodes() <= set(isa.M2_OPS) | _illegal_opcodes(tr), tr.path
            if root.name == "m3":
                assert tr.opcodes() <= set(isa.M3_OPS) | _any_illegal(tr), tr.path
    if n == 0:
        pytest.skip("no traces found")


class _FakeMem:
    """Just enough of MemModel for replay.compare(): the state after a perfect run of tr."""

    def __init__(self, tr):
        self.mem = {}
        self.src_beats = [[0] * 4 for _ in range(tr.num_ch)]
        self.written = tr.expect_beats()
        self.read_unwritten = [set() for _ in range(tr.num_ch)]
        for r in tr.expect:
            for i, b in enumerate(r.beats()):
                self.mem[(r.ch, b)] = r.data[64 * i:64 * i + 64]

    def read_bytes(self, ch, off, size):
        return b"".join(self.mem.get((ch, off // 64 + i), bytes(64)) for i in range(size // 64))


def test_replay_compare_detects_mismatches():
    trs = [t for t in trace.load_all(REPO / "traces" / "m2" / "tmax1") if t.expect and t.err_info is not None]
    if not trs:
        pytest.skip("no traces/m2/tmax1")
    tr = next(t for t in trs if t.is_error) if any(t.is_error for t in trs) else trs[0]
    final = dict(status=tr.status, err_code=tr.err_code, err_pc=tr.err_pc, err_info=tr.err_info)
    assert replay.compare(tr, _FakeMem(tr), final) == []
    for k in final:                                        # every final CSR is compared
        bad = dict(final, **{k: final[k] ^ 1})
        assert any(k in e for e in replay.compare(tr, _FakeMem(tr), bad)), k
    m = _FakeMem(tr)                                       # one flipped data byte
    key = next(iter(m.mem))
    m.mem[key] = bytes([m.mem[key][0] ^ 0x40]) + m.mem[key][1:]
    assert any("expect mismatch" in e for e in replay.compare(tr, m, final))
    m = _FakeMem(tr)                                       # extra written beat
    m.written[0].add(1 << 27)
    assert any("outside the expect" in e for e in replay.compare(tr, m, final))
    m = _FakeMem(tr)                                       # missing written beat
    r0 = tr.expect[0]
    m.written[r0.ch].discard(r0.offset // 64)
    assert any("not written" in e for e in replay.compare(tr, m, final))
    m = _FakeMem(tr)                                       # read of uninitialised DRAM
    m.read_unwritten[0].add((1 << 27) + 5)
    assert any("does not initialise" in e for e in replay.compare(tr, m, final))
    m = _FakeMem(tr)                                       # read beats per source (LD / MVM)
    m.src_beats = [[0, b[isa.SRC_DMA], b[isa.SRC_WS], 0] for b in tr.expected_read_beats()]
    assert replay.compare(tr, m, final, check_sources=True) == []
    if any(b[isa.SRC_DMA] for b in tr.expected_read_beats()):
        c = next(i for i, b in enumerate(tr.expected_read_beats()) if b[isa.SRC_DMA])
        m.src_beats[c][isa.SRC_WS] += 1
        assert any("source WS" in e for e in replay.compare(tr, m, final, check_sources=True))


def _any_illegal(tr):
    """The faulting opcode of an ILLEGAL_OP ending (any opcode may be the deliberate error)."""
    if tr.err_code == isa.ERR_ILLEGAL_OP and tr.err_pc < tr.cmd_count:
        return {tr.insn(tr.err_pc)[0] & 0xFF}
    return set()


def _illegal_opcodes(tr):
    """Opcodes allowed in an M2 trace only as its faulting (last executed) instruction."""
    if tr.err_code == isa.ERR_ILLEGAL_OP and tr.err_pc < tr.cmd_count:
        op = tr.insn(tr.err_pc)[0] & 0xFF
        return {op} if op != isa.OP_MVM else set()
    return set()


# =============================================================================================
# Simulation tests (pytest side)
# =============================================================================================
LOOP_PARAMS = {"NUM_CH": 2, "ADDR_BEAT_W": 20}


def test_loop_sim():
    run("tsr_tb_loop", __name__, parameters=LOOP_PARAMS, extra_sources=[LOOP_SV],
        testcase=["loop_structs", "loop_csr", "loop_mem_random", "loop_mem_violations"])


@pytest.mark.parametrize("cfg", ["default", "small"])
def test_core_skeleton_sim(cfg):
    params = {} if cfg == "default" else {
        "NUM_CH": 1, "SPM_BYTES": 65536, "CH_BYTES": "64'd1073741824", "K_MAX": 4096, "T_MAX": 8}
    run("tsr_core", __name__, parameters=params, testcase="core_skeleton_attach",
        extra_env={"TSR_CORE_NUM_CH": str(params.get("NUM_CH", 4))})


PYCORE_SETS = [("m2/errors", 40), ("m2/errors8", 32), ("m2/small1ch", 16), ("m2/ch3", 8), ("m2/tmax8", 6),
               ("m2/tmax1", 12), ("gen/tmax1", 24), ("gen/tmax8", 8)]


def _m3_sets():
    """Every directory of traces/m3 that directly holds traces (one configuration each)."""
    root = REPO / "traces" / "m3"
    if not root.is_dir():
        return []
    sets = sorted({p.parent.relative_to(REPO / "traces").as_posix() for p in trace.find(root)})
    # every short error trace; the long MVM sets are capped (the RTL core replays all of them)
    cap = {"m3/errors": 100, "m3/errors8": 100, "m3/tmax1": 12, "m3/tmax8": 8}
    return [(s_, cap.get(s_, 16)) for s_ in sets]


PYCORE_SETS += _m3_sets()


@pytest.mark.parametrize("tset,limit", PYCORE_SETS, ids=[s.replace("/", "-") for s, _ in PYCORE_SETS])
def test_pycore_traces(tset, limit):
    root = REPO / "traces" / tset
    trs = trace.find(root)
    if not trs:
        pytest.skip("no traces in %s" % root)
    cfg = trace.load(trs[0]).config
    run("tsr_tb_loop", __name__, parameters={"NUM_CH": cfg["num_ch"], "ADDR_BEAT_W": cfg["addr_beat_w"]},
        extra_sources=[LOOP_SV], testcase="loop_pycore_traces",
        extra_env={"TSR_PYCORE_SET": str(root), "TSR_PYCORE_MAX": str(limit)})


# =============================================================================================
# cocotb tests
# =============================================================================================
@cocotb.test()
async def loop_structs(dut):
    """isa.py struct layouts == tsr_pkg.sv (widths and field positions)."""
    await start_clock_and_reset(dut)
    assert int(dut.bits_fetch_insn.value) == isa.FETCH_INSN_T.width
    assert int(dut.bits_dma_cmd.value) == isa.DMA_CMD_T.width
    assert int(dut.bits_qact_cmd.value) == isa.QACT_CMD_T.width
    assert int(dut.bits_mvm_cmd.value) == isa.MVM_CMD_T.width
    assert int(dut.qa_lines_a256_27648.value) == isa.qa_lines(isa.QA_A256, 27648)
    assert int(dut.row_beats_q4k_27648.value) == ref().row_beats(isa.FMT_Q4_K, 27648)
    rnd = tb_rng(1)
    for _ in range(20):
        d = {n: rnd.getrandbits(w) for n, w in isa.DMA_CMD_T.fields}
        q = {n: rnd.getrandbits(w) for n, w in isa.QACT_CMD_T.fields}
        f = {n: rnd.getrandbits(w) for n, w in isa.FETCH_INSN_T.fields}
        m = {n: rnd.getrandbits(w) for n, w in isa.MVM_CMD_T.fields}
        dut.dma_in.value = isa.DMA_CMD_T.pack(**d)
        dut.qact_in.value = isa.QACT_CMD_T.pack(**q)
        dut.fetch_in.value = isa.FETCH_INSN_T.pack(**f)
        dut.mvm_in.value = isa.MVM_CMD_T.pack(**m)
        await FallingEdge(dut.clk)
        for n, _ in isa.DMA_CMD_T.fields:
            assert int(getattr(dut, "dma_" + n).value) == d[n], n
        for n, _ in isa.QACT_CMD_T.fields:
            assert int(getattr(dut, "qact_" + n).value) == q[n], n
        assert int(dut.fetch_range_err.value) == f["range_err"] and int(dut.fetch_pc.value) == f["pc"]
        assert int(dut.fetch_w0.value) == f["insn"] & 0xFFFFFFFF
        assert int(dut.fetch_w7.value) == f["insn"] >> 224
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


@cocotb.test()
async def loop_csr(dut):
    await start_clock_and_reset(dut)
    csr = CsrDriver(dut)
    vals = [tb_rng(2).getrandbits(32) for _ in range(8)]
    for i, v in enumerate(vals):
        await csr.write(4 * i, v)
    for i, v in enumerate(vals):
        assert await csr.read(4 * i) == v
    assert await csr.read(0x020) == 0 and await csr.read(0x002) == 0
    assert await csr.read64(0x008) == (vals[3] << 32) | vals[2]


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


@cocotb.test()
async def core_skeleton_attach(dut):
    """tsr_core elaborates; MemModel and CsrDriver attach to its boundary (works for the skeleton
    and for the finished core: with a finished core the CSR ID read returns the magic)."""
    import os
    num_ch = int(os.environ.get("TSR_CORE_NUM_CH", "4"))
    await start_clock_and_reset(dut)
    mem = MemModel(dut, num_ch=num_ch, addr_w=len(dut.m_rd_req_addr) // num_ch, seed=11).start()
    csr = CsrDriver(dut, timeout_cycles=50)
    try:
        ident = await csr.read(isa.CSR_ID)
        assert ident == isa.ID_MAGIC, hex(ident)
    except CsrTimeout:
        dut._log.warning("CSR port does not answer (skeleton tsr_csr)")
    await ClockCycles(dut.clk, 20)
    assert int(dut.done.value) == 0
    mem.check(idle=True)
    mem.stop()


@cocotb.test()
async def loop_pycore_traces(dut):
    """PyCore runs each trace over the memory-port protocol; replay.compare() must be clean."""
    import os
    from common.pycore import PyCore, SeqMaster
    trs = trace.load_all(os.environ["TSR_PYCORE_SET"])[:int(os.environ.get("TSR_PYCORE_MAX", "1000"))]
    await start_clock_and_reset(dut)
    n_err = 0
    for k, tr in enumerate(trs):
        mem = MemModel.for_trace(dut, tr, seed=tb_rng(k).getrandbits(31)).start()
        core = PyCore(tr.config, SeqMaster(dut, tr.num_ch, tr.config["addr_beat_w"]))
        final = await core.run(tr.cmd_addr_beat, tr.cmd_count, tr.args)
        await ClockCycles(dut.clk, 3)
        mem.check(idle=True)
        errs = replay.compare(tr, mem, final, check_sources=True)
        mem.stop()
        assert not errs, "%s: %s" % (tr.name, errs)
        assert [b[isa.SRC_WS] for b in tr.expected_read_beats()] == core.mvm_beats, tr.name
        n_err += final["err_code"] != 0
    dut._log.info("PyCore: %d traces replayed (%d end in an error)", len(trs), n_err)


# =============================================================================================
# M3: reference wrappers, golden MVM model, program generator, simlib, pipes
# =============================================================================================
def _cfg(**kw):
    c = dict(isa.U200_V1)
    c.update(kw)
    return c


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
    # spec examples (행 크기) and the symmetric allocation (대칭 할당)
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


def test_mvm_model_vs_simulator():
    """mvm_model.mvm_reference (golden MVM semantics used by the block tests) == the ISA simulator
    executing the same MVM (whole SPM compared), for both fmts, ACC, T, strides, N edge cases."""
    from common.simlib import Sim
    from common.proggen import prog_bytes
    rnd = random.Random(34)
    cfg = _cfg(t_max=8, num_ch=3, spm_bytes=256 * 1024)
    sim = Sim(cfg)
    lim = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
    woff, io, cmd = isa.REGION_WEIGHT_OFF // 64, isa.REGION_IO_OFF // 64, 0x100
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
        rbytes = 64 * mvm_model.row_beats(fmt, K)
        for n in range(N):                     # symmetric allocation == tsr_row_location
            ch, off = ref().row_location(fmt, N, K, 3, n)
            assert per_ch[ch][64 * off:64 * off + rbytes] == rows[n], (n, ch, off)
        assert sum(len(d) for d in per_ch) == N * rbytes
        ql = isa.qa_lines(fmt, K)
        yl = (N + 15) // 16
        stride = rnd.choice([0, yl + 1]) if T > 1 else 0
        ds = stride or yl
        x, dst = 64, 64 + T * ql + rnd.randint(0, 5)
        y_old = rnd.randbytes(64 * ((T - 1) * ds + yl))
        init = [(c, woff * 64, d) for c, d in enumerate(per_ch) if d]
        init += [(0, io * 64, b"".join(qas)), (0, (io + 4096) * 64, y_old)]
        insns = [isa.enc_ld(0, x, io, T * ql)]
        if y_old:
            insns.append(isa.enc_ld(0, dst, io + 4096, len(y_old) // 64))
        insns += [isa.enc_end()]
        sim.run(prog_bytes(insns), cmd, init=init)
        spm = bytearray(sim.spm())
        res = sim.run(prog_bytes([isa.enc_mvm(fmt, dst, x, woff, N, K, T, stride, acc=acc), isa.enc_end()]), cmd)
        assert res.final["err_code"] == 0, res.final
        dram = {c: per_ch[c] for c in range(3)}
        read = lambda c, b, n: dram[c][64 * (b - woff):64 * (b - woff + n)]  # noqa: E731
        mvm_model.mvm_reference(spm, read, fmt=fmt, acc=acc, dst=dst, x=x, w_off=woff, N=N, K=K, T=T,
                                dst_stride=stride, num_ch=3)
        assert bytes(spm) == res.spm, (it, fmt, K, N, T, acc)
        R = -(-N // 3) if N else 0
        rb = mvm_model.row_beats(fmt, K)
        for c in range(3):   # PERF: MVM weight beats per channel = rows_c * ROW_BEATS (+ 1 fetch beat on ch0)
            want = (mvm_model.rows_c(N, 3, c) * rb if N else 0) + (1 if c == 0 else 0)
            assert res.perf[isa.perf_ch_rd_beats(c)] == want, (c, R, res.perf)
        n_runs += 1
    assert n_runs == 24 and lim > woff
    sim.close()


# Digest of the M2 program stream (ProgGen m3=False) for fixed seeds: generated by the original
# tb/cmdproc generator before the move (verified identical over 6000 programs in 6 configurations).
PROGGEN_M2_DIGEST_VALUE = "2f46bb92027ec5433481c604db52ae0e7db2bf5bd1d0d365e7d1b51280505707"


def _proggen_digest():
    import hashlib
    from common.proggen import ProgGen
    h = hashlib.sha256()
    for c in (_cfg(), _cfg(t_max=8), _cfg(num_ch=2, t_max=4, k_max=256, spm_bytes=4096, ch_bytes=4096, addr_beat_w=5)):
        for has_mvm in (False, True):
            for seed in range(4):
                g = ProgGen(random.Random(seed), c, has_mvm)
                h.update(g.init_data()[0][2][:64])
                for _ in range(10):
                    prog, cmd, count, args = g.program()
                    h.update(prog + repr((cmd, count, args)).encode())
    return h.hexdigest()


def test_proggen_m2_stream_digest():
    """ProgGen(m3=False) keeps generating the M2 programs (tb/cmdproc and tb/core rely on it)."""
    assert _proggen_digest() == PROGGEN_M2_DIGEST_VALUE


def test_proggen_m3_simulator():
    """ProgGen(m3=True) programs on the simulator: MVMs execute (both fmts, ACC, T > 1, N edge
    cases), error endings still occur, and no executed instruction has an undefined overlap."""
    from common.proggen import ProgGen, executed_count, executed_overlaps, prog_bytes
    from common.simlib import Sim
    for name, cfg in (("t8", _cfg(t_max=8)), ("ch3", _cfg(num_ch=3, t_max=3, spm_bytes=256 * 1024)),
                      ("tiny", _cfg(num_ch=2, t_max=4, k_max=256, spm_bytes=4096, ch_bytes=4096, addr_beat_w=5))):
        sim = Sim(cfg)
        g = ProgGen(random.Random(35), cfg, True, m3=True)
        sim.run(prog_bytes([isa.enc_end()]), g.cmd_base, init=g.init_data())
        codes, fmts, Ts, acc, Ns = collections.Counter(), set(), set(), 0, set()
        for i in range(120):
            prog, cmd, count, args = g.mvm_program() if i % 2 else g.program()
            res = sim.run(prog, cmd, args, count=count)
            codes[res.final["err_code"]] += 1
            n = executed_count(prog, res.final, count)
            assert not executed_overlaps(prog, n, args, cfg), (name, i)
            for pc in range(n):
                w = isa.insn_words(prog[32 * pc:32 * pc + 32])
                if w[0] & 0xFF == isa.OP_MVM:
                    fmts.add((w[0] >> 8) & 0xF)
                    Ts.add((w[5] >> 16) & 0xF)
                    acc += (w[0] >> 12) & 1
                    Ns.add(min(w[4], 17))
        assert fmts == {0, 1} and acc > 10 and codes[0] > 40, (name, fmts, acc, codes)
        assert len(Ts) == cfg["t_max"] and {0, 1} <= Ns, (name, Ts, Ns)
        assert sum(v for k, v in codes.items() if k) > 5, (name, codes)
        sim.close()


def _build_simlib(root):
    from common import simlib
    return str(simlib._lib_path(Path(root)))


def test_simlib_keyed_build(tmp_path):
    """Two processes building the simulator library into one root at the same time end with one
    content-keyed library that loads, and no temporary build directories are left behind."""
    import multiprocessing as mp
    from common import simlib
    with mp.get_context("spawn").Pool(2) as pool:
        paths = pool.map(_build_simlib, [str(tmp_path)] * 2)
    assert paths[0] == paths[1], paths
    p = Path(paths[0])
    assert p.exists() and p.parent.parent == tmp_path and p.name == simlib.LIB_NAME
    assert not simlib._missing_symbols(p)
    assert [d.name for d in tmp_path.iterdir()] == [p.parent.name]
    assert [f.name for f in p.parent.iterdir()] == [simlib.LIB_NAME]
    assert simlib._lib_path(tmp_path) == p                 # cached: same content -> same path


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

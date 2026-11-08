"""tsr_mvm_gather alone (docs/rtl-m3.md §6.3): the gather is the toplevel; Python drives the
per-cluster result streams and models SPM client 2 (fixed read latency SPM_RD_LAT, no read
backpressure, random request / write acceptance, read-first on same-cycle collisions, byte
enables), so the gather is checked at its own ports.

Reference: common.mvm_model.apply_mvm (the tsr_sim.c ex_mvm write loop; ACC = libtsr_ref fadd)
on the same initial SPM image. Checks per MVM:
  - final SPM image == reference (every token's y lines incl. zero tails; nothing else changed);
  - every byte of the y ranges written exactly once, no other byte (byte enables);
  - writes == mvm_model.gather_jobs (count) when the token ranges do not overlap;
  - ACC: reads only of y lines, every read's response arrives (R4: the gather never stalls a
    response), each lane's old value is read before the only write of that lane;
  - done: one registered pulse, after the last write was accepted, never in the start cycle; no
    request after done; res_ready = 0 while idle;
  - R2 on the SPM request ports (held, payload stable until accepted).
cocotb tests:
  gather_random     random N / NUM_CH channel split / T / ACC / strides / old values, random
                    result order across clusters (each cluster in row order), random valid gaps,
                    random SPM read / write acceptance
  gather_special    ACC with every pair of special old / new values (NaN payloads, Inf - Inf,
                    subnormals (DAZ / FTZ), signed zeros, overflow, cancellation) in every lane
  gather_perf       results offered every cycle on every cluster, SPM always ready: SPM line
                    writes per cycle (plain and ACC), result-to-write latency, and fairness when
                    the write port is oversubscribed (clusters finish together: round robin)
  gather_comb       no combinational path from any input to any output (clock stopped, inputs
                    wiggled one at a time)
Configurations: NUM_CH 1 / 3 / 4 / 16, T_MAX 1 / 2 / 3 / 8, SPM_RD_LAT 1..4, ADD_LAT 1 / 3 / 5 / 8.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer

from common import mvm_model as mm
from common.runner import RTL, run
from common.tbutil import seed as tb_seed
from common.tsr_ref import ref

CFGS = {
    "g4t8_old": dict(NUM_CH=4, T_MAX=8, SPM_BYTES=262144, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1),
    "g4t8_pair": dict(NUM_CH=4, T_MAX=8, SPM_BYTES=262144, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1),
    "g4t8_narrow": dict(NUM_CH=4, T_MAX=8, SPM_BYTES=262144, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1, GATHER_FLITS=2),
    "g4t4_narrow": dict(NUM_CH=4, T_MAX=4, SPM_BYTES=4194304, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1, GATHER_FLITS=2),
    "g4t4_pair": dict(NUM_CH=4, T_MAX=4, SPM_BYTES=4194304, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1),
    # XSLR_REG: one tsr_rpipe stage on each pair link (2 slots of a whole job, or 2 flits)
    "g4t8_pair_x": dict(NUM_CH=4, T_MAX=8, SPM_BYTES=262144, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1, XSLR_REG=1),
    "g4t4_narrow_x": dict(NUM_CH=4, T_MAX=4, SPM_BYTES=4194304, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1, GATHER_FLITS=2, XSLR_REG=1),
    "g4t4_narrow_odd": dict(NUM_CH=4, T_MAX=4, SPM_BYTES=524288, SPM_RD_LAT=4, ADD_LAT=8, JOB_PIPE=2, HAS_ATT=1, GATHER_PAIR=1, GATHER_FLITS=2),
    "g4t1":  dict(NUM_CH=4, T_MAX=1, SPM_BYTES=262144, SPM_RD_LAT=2, ADD_LAT=5),
    "g1t3":  dict(NUM_CH=1, T_MAX=3, SPM_BYTES=65536, SPM_RD_LAT=1, ADD_LAT=1),
    "g3t2":  dict(NUM_CH=3, T_MAX=2, SPM_BYTES=131072, SPM_RD_LAT=3, ADD_LAT=3),
    "g16t8": dict(NUM_CH=16, T_MAX=8, SPM_BYTES=262144, SPM_RD_LAT=4, ADD_LAT=8),
}
TESTS = ["gather_random", "gather_special", "gather_perf", "gather_comb", "gather_pair_jobs"]

INPUTS = ["start", "acc", "t", "n", "dst_line", "dst_stride", "expect_ch", "res_valid", "res_row",
          "res_last", "res_y", "spm_rd_req_ready", "spm_rd_rsp_valid", "spm_rd_rsp_data",
          "spm_wr_ready"]
OUTPUTS = ["done", "res_ready", "spm_rd_req_valid", "spm_rd_req_line", "spm_wr_valid", "spm_wr_line",
           "spm_wr_data", "spm_wr_be"]


@pytest.mark.parametrize("cfg", list(CFGS))
def test_mvm_gather(cfg):
    p = CFGS[cfg]
    sel = os.environ.get("TSR_GATHER_TESTS")
    tests = [t for t in TESTS if not sel or t in sel.split(",")]
    extra = os.environ.get("TSR_MVM_BUILD_ARGS", "").split()   # e.g. -DSYNTHESIS (no RTL assertions)
    run("tsr_mvm_gather", __name__, parameters=p, testcase=tests, build_args=extra,
        sources=[RTL / name for name in (
            "core/tsr_tables_pkg.sv", "core/tsr_pkg.sv", "core/tsr_fp_pkg.sv",
            "core/tsr_att_pkg.sv", "core/tsr_vu_pkg.sv", "../third_party/common_cells/src/fifo_v3.sv",
            "tech/generic/tsr_sram_1r1w.sv", "core/tsr_mem_arb.sv",
            "core/tsr_vpipe.sv", "core/tsr_rpipe.sv", "core/tsr_fp_add.sv",
            "core/tsr_mvm_att.sv", "core/tsr_mvm_local_scale.sv", "core/tsr_mvm_gather.sv")],
        name="tsr_mvm_gather_%s%s" % (cfg, "_x" if extra else ""),
        extra_env={"TSR_GATHER_CFG": json.dumps(dict(p, name=cfg))})


def scale() -> float:
    return float(os.environ.get("TSR_MVM_SCALE", "1.0"))


def save_metrics(kind, measurements):
    Path("gather_%s_metrics.json" % kind).write_text(json.dumps({
        "configuration": json.loads(os.environ["TSR_GATHER_CFG"]),
        "measurements": measurements,
    }, indent=2) + "\n")


SPECIAL = (0x7FC00000, 0xFFC00001, 0x7F800001, 0x7FBFFFFF, 0x7F800000, 0xFF800000, 0x00000000,
           0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
           0x3F800000, 0xBF800000, 0x00FFFFFF, 0x80FFFFFF, 0x7F000000, 0x33800000)


class Gtb:
    def __init__(self, dut, p: dict):
        self.dut, self.p = dut, p
        self.nch, self.tmax = p["NUM_CH"], p["T_MAX"]
        self.L = p["SPM_BYTES"] // 64
        self.lat = p["SPM_RD_LAT"]
        self.spm = bytearray(64 * self.L)
        self.rsp = []                 # [due edge, data]
        self.edge = 0
        self.p_rd = 1.0
        self.p_wr = 1.0
        self.rng = random.Random(tb_seed() * 31 + 7)
        self.wr_log = []              # (edge, line, be)
        self.rd_log = []              # (edge, line)
        self.done_edges = []
        self.errors = []
        self.pend_rd = None
        self.pend_wr = None
        self.active = False

    async def start(self):
        d = self.dut
        d.rst_n.value = 0
        for n in ("start", "acc", "t", "n", "dst_line", "dst_stride", "expect_ch", "res_valid",
                  "res_row", "res_last", "res_y", "spm_rd_req_ready", "spm_rd_rsp_valid",
                  "spm_rd_rsp_data", "spm_wr_ready", "att", "att_maxout", "att_tq", "att_gs", "att_tact", "att_nseg",
                  "att_g", "att_hg0", "att_nhead", "att_mdst", "aj_valid", "aj",
                  "scale_en", "scale", "scale_rsp"):
            getattr(d, n).value = 0
        await Timer(1, "ns")
        self.clock = Clock(d.clk, 10, unit="ns")
        self.clock.start(start_high=True)
        cocotb.start_soon(self.spm_task())
        await ClockCycles(d.clk, 5)
        await RisingEdge(d.clk)
        d.rst_n.value = 1
        await ClockCycles(d.clk, 2)

    def fail(self, msg):
        self.errors.append("edge %d: %s" % (self.edge, msg))
        raise AssertionError(self.errors[-1])

    async def spm_task(self):
        """SPM client model: sample at the falling edge, act at the rising edge, drive after it."""
        d = self.dut
        rd_ready = wr_ready = 0
        while True:
            await FallingEdge(d.clk)
            rst = int(d.rst_n.value) == 0
            rv = int(d.spm_rd_req_valid.value)
            rl = int(d.spm_rd_req_line.value)
            wv = int(d.spm_wr_valid.value)
            if wv:
                w = (int(d.spm_wr_line.value), int(d.spm_wr_be.value), int(d.spm_wr_data.value))
            done = int(d.done.value)
            await RisingEdge(d.clk)
            self.edge += 1
            if rst:
                self.rsp.clear()
                self.pend_rd = self.pend_wr = None
                rd_ready = wr_ready = 0
                d.spm_rd_req_ready.value = 0
                d.spm_wr_ready.value = 0
                d.spm_rd_rsp_valid.value = 0
                continue
            # R2 on both request ports
            if self.pend_rd is not None and (not rv or rl != self.pend_rd):
                self.fail("read request dropped / changed before acceptance")
            if self.pend_wr is not None and (not wv or w != self.pend_wr):
                self.fail("write request dropped / changed before acceptance")
            if done:
                self.done_edges.append(self.edge)
            # reads first (same-edge read of a line written at this edge returns old data)
            if rv and rd_ready:
                if not 0 <= rl < self.L:
                    self.fail("read line %d out of range" % rl)
                self.rsp.append([self.edge + self.lat - 1, int.from_bytes(self.spm[64 * rl:64 * rl + 64], "little")])
                self.rd_log.append((self.edge, rl))
                self.pend_rd = None
            elif rv:
                self.pend_rd = rl
            if wv and wr_ready:
                ln, be, data = w
                if not 0 <= ln < self.L:
                    self.fail("write line %d out of range" % ln)
                db = data.to_bytes(64, "little")
                for b in range(64):
                    if be >> b & 1:
                        self.spm[64 * ln + b] = db[b]
                self.wr_log.append((self.edge, ln, be))
                self.pend_wr = None
            elif wv:
                self.pend_wr = w
            # drive for the next cycle
            if self.rsp and self.rsp[0][0] == self.edge:
                _, data = self.rsp.pop(0)
                d.spm_rd_rsp_valid.value = 1
                d.spm_rd_rsp_data.value = data
            else:
                d.spm_rd_rsp_valid.value = 0
                d.spm_rd_rsp_data.value = self.rng.getrandbits(512)   # don't-care data
            if self.rsp and self.rsp[0][0] < self.edge:
                self.fail("read response overdue")
            rd_ready = int(self.rng.random() < self.p_rd)
            wr_ready = int(self.rng.random() < self.p_wr)
            d.spm_rd_req_ready.value = rd_ready
            d.spm_wr_ready.value = wr_ready

    async def run_mvm(self, *, N: int, T: int, acc: bool, dst: int, stride: int, ys, p_gap: float = 0.3,
                      timeout: int = 200000, order: str = "rand") -> dict:
        """One gather run: ys[t][n] f32 bits. Returns stats."""
        d, nch = self.dut, self.nch
        R = -(-N // nch)
        expect = sum(1 << c for c in range(nch) if c * R < N)
        queues = []
        for c in range(nch):
            rc = mm.rows_c(N, nch, c)
            queues.append([(c * R + r, r == rc - 1) for r in range(rc)])
        await RisingEdge(d.clk)
        d.acc.value = int(acc)
        d.t.value = T
        d.n.value = N
        d.dst_line.value = dst
        d.dst_stride.value = stride
        d.expect_ch.value = expect
        d.start.value = 1
        e0 = self.edge
        await RisingEdge(d.clk)
        d.start.value = 0
        n_done0 = len(self.done_edges)
        wr0, rd0 = len(self.wr_log), len(self.rd_log)
        head = [0] * nch
        cur = [None] * nch        # item presented per cluster
        first_res = last_res = None
        last_c = [None] * nch     # edge of each cluster's last accepted result
        cyc = 0
        rnd = self.rng
        while len(self.done_edges) == n_done0:
            # drive (after the rising edge)
            vmask, rows, lasts, yv = 0, 0, 0, 0
            for c in range(nch):
                if cur[c] is None and head[c] < len(queues[c]) and rnd.random() >= p_gap:
                    if order == "rand" or c == min((k for k in range(nch) if head[k] < len(queues[k])), default=c):
                        cur[c] = queues[c][head[c]]
                if cur[c] is not None:
                    row, last = cur[c]
                    vmask |= 1 << c
                    rows |= row << (32 * c)
                    lasts |= int(last) << c
                    for t in range(self.tmax):
                        v = ys[t][row] if t < T else rnd.getrandbits(32)
                        yv |= v << (32 * (c * self.tmax + t))
            d.res_valid.value = vmask
            d.res_row.value = rows
            d.res_last.value = lasts
            d.res_y.value = yv
            await FallingEdge(d.clk)
            rr = int(d.res_ready.value)
            for c in range(nch):
                if cur[c] is not None and rr >> c & 1:
                    head[c] += 1
                    cur[c] = None
                    last_res = self.edge + 1
                    last_c[c] = self.edge + 1
                    if first_res is None:
                        first_res = self.edge + 1
            await RisingEdge(d.clk)
            cyc += 1
            if cyc > timeout:
                raise AssertionError("gather did not finish (N %d T %d acc %d): heads %s" % (N, T, acc, head))
        d.res_valid.value = 0
        assert all(head[c] == len(queues[c]) for c in range(nch)), "done before every result was taken"
        de = self.done_edges[-1]
        wr = self.wr_log[wr0:]
        rd = self.rd_log[rd0:]
        assert de > e0 + 1, "done in the start cycle"
        assert not wr or wr[-1][0] < de, "SPM write accepted at / after done"
        # quiet after done
        await ClockCycles(d.clk, 3)
        assert len(self.wr_log) == wr0 + len(wr) and len(self.rd_log) == rd0 + len(rd), "traffic after done"
        assert int(d.res_ready.value) == 0, "res_ready while idle"
        return dict(wr=wr, rd=rd, done=de, first_res=first_res, last_res=last_res, start=e0, last_c=last_c)

    def check(self, *, N, T, acc, dst, stride, ys, before: bytes, st: dict):
        nch = self.nch
        ds = stride
        yl = (N + 15) // 16
        want = bytearray(before)
        mm.apply_mvm(want, dst, ds, N, T, acc, ys)
        if bytes(self.spm) != bytes(want):
            bad = [i for i in range(0, len(want), 4) if self.spm[i:i + 4] != want[i:i + 4]]
            i = bad[0]
            raise AssertionError("SPM mismatch (N %d T %d acc %d dst %d stride %d): %d words differ, first line %d "
                                 "lane %d got 0x%08x want 0x%08x" % (
                                     N, T, acc, dst, ds, len(bad), i // 64, (i % 64) // 4,
                                     int.from_bytes(self.spm[i:i + 4], "little"), int.from_bytes(want[i:i + 4], "little")))
        ylines = {dst + t * ds + i for t in range(T) for i in range(yl)}
        overlap = T > 1 and ds < yl
        if not overlap:
            cnt = {}
            for _, ln, be in st["wr"]:
                for b in range(64):
                    if be >> b & 1:
                        cnt[64 * ln + b] = cnt.get(64 * ln + b, 0) + 1
            ybytes = {64 * ln + b for ln in ylines for b in range(64)}
            assert set(cnt) == ybytes and all(v == 1 for v in cnt.values()), "byte coverage wrong"
            assert len(st["wr"]) == len(mm.gather_jobs(N, nch, T, dst, ds)), "write count != gather_jobs"
        if acc:
            assert all(ln in ylines for _, ln in st["rd"]), "ACC read outside y"
            assert {ln for _, ln in st["rd"]} == ylines
        else:
            assert not st["rd"], "non-ACC gather read the SPM"


async def new_gtb(dut) -> Gtb:
    p = json.loads(os.environ["TSR_GATHER_CFG"])
    g = Gtb(dut, p)
    await g.start()
    return g


def rand_ys(rnd: random.Random, T: int, N: int, special: float = 0.1):
    return [[mm.rand_f32(rnd, special, -30, 30) for _ in range(N)] for _ in range(T)]


@cocotb.test()
async def gather_random(dut):
    g = await new_gtb(dut)
    rnd = random.Random(tb_seed() * 13 + 1)
    nch, TM, L = g.nch, g.tmax, g.L
    n_mvm = max(5, int(80 * scale()))
    for it in range(n_mvm):
        N = rnd.choice([1, 2, nch - 1 or 1, nch, nch + 1, 15, 16, 17, 31, 33, 16 * nch, 17 * nch + 3,
                        rnd.randint(1, 300), rnd.randint(1, 2000)])
        T = rnd.randint(1, TM)
        yl = (N + 15) // 16
        overlap = T > 1 and rnd.random() < 0.08
        if T > 1:
            stride = rnd.randint(1, yl - 1) if overlap and yl > 1 else yl + rnd.choice([0, 0, 1, rnd.randint(0, 30)])
        else:
            stride = rnd.choice([yl, rnd.randint(1, 1 << 30), 0xFFFFFFFF])
        span = (T - 1) * stride + yl if T > 1 else yl
        if span > L:
            continue
        dst = rnd.choice([0, L - span, rnd.randrange(0, L - span + 1)])
        acc = bool(rnd.getrandbits(1))
        g.p_rd = rnd.choice([1.0, 1.0, 0.7, 0.3])
        g.p_wr = rnd.choice([1.0, 1.0, 0.7, 0.3])
        # SPM content: random bytes (old values with specials for ACC)
        for ln in range(dst, dst + span):
            g.spm[64 * ln:64 * ln + 64] = b"".join(mm.rand_f32(rnd, 0.2, -30, 30).to_bytes(4, "little")
                                                  for _ in range(16)) if acc else rnd.randbytes(64)
        before = bytes(g.spm)
        ys = rand_ys(rnd, T, N, special=rnd.choice([0.0, 0.1, 0.5]))
        st = await g.run_mvm(N=N, T=T, acc=acc, dst=dst, stride=stride, ys=ys,
                             p_gap=rnd.choice([0.0, 0.3, 0.8]), order=rnd.choice(["rand", "rand", "seq"]))
        if overlap:
            continue                        # undefined result; finishing normally is the check
        g.check(N=N, T=T, acc=acc, dst=dst, stride=stride if T > 1 else yl, ys=ys, before=before, st=st)
    dut._log.info("gather_random: %d MVMs", n_mvm)


@cocotb.test()
async def gather_special(dut):
    g = await new_gtb(dut)
    rnd = random.Random(tb_seed() * 13 + 2)
    nch, TM, L = g.nch, g.tmax, g.L
    r = ref()
    pairs = [(a, b) for a in SPECIAL for b in SPECIAL]
    rnd.shuffle(pairs)
    # every pair in some lane: N rows of T tokens per MVM
    while pairs:
        T = rnd.randint(1, TM)
        N = rnd.choice([16 * nch, 33, 47, 16 * nch + 5])
        yl = (N + 15) // 16
        dst = rnd.randrange(0, L - T * yl + 1)
        ys = [[0] * N for _ in range(T)]
        for t in range(T):
            for n in range(N):
                a, b = pairs.pop() if pairs else (rnd.choice(SPECIAL), rnd.choice(SPECIAL))
                ln, lane = dst + t * yl + n // 16, n % 16
                g.spm[64 * ln + 4 * lane:64 * ln + 4 * lane + 4] = a.to_bytes(4, "little")
                ys[t][n] = b
        for t in range(T):
            ln = dst + t * yl + yl - 1
            for lane in range(16):
                if 16 * (yl - 1) + lane >= N:
                    g.spm[64 * ln + 4 * lane:64 * ln + 4 * lane + 4] = rnd.choice(SPECIAL).to_bytes(4, "little")
        before = bytes(g.spm)
        st = await g.run_mvm(N=N, T=T, acc=True, dst=dst, stride=yl, ys=ys, p_gap=0.2)
        g.check(N=N, T=T, acc=True, dst=dst, stride=yl, ys=ys, before=before, st=st)
    # canonical NaN and FTZ spot checks against the reference directly
    assert r.fadd(0x7F800001, 0x3F800000) == 0x7FC00000
    assert r.fadd(0x00000001, 0x80000000) in (0x00000000, 0x80000000)
    dut._log.info("gather_special: %d special pairs", len(SPECIAL) ** 2)


@cocotb.test()
async def gather_perf(dut):
    g = await new_gtb(dut)
    rnd = random.Random(tb_seed() * 13 + 3)
    nch, TM, L = g.nch, g.tmax, g.L
    g.p_rd = g.p_wr = 1.0
    out = []
    metrics = []
    for acc in (False, True):
        for T in sorted({1, TM}):
            N = min(16 * 64 * max(1, 16 // nch) * nch // max(1, T), 16 * (L // (2 * T)))
            N -= N % (16 * nch)
            N = max(N, 16 * nch)
            yl = (N + 15) // 16
            ys = rand_ys(rnd, T, N, 0.0)
            before = bytes(g.spm)
            st = await g.run_mvm(N=N, T=T, acc=acc, dst=0, stride=yl, ys=ys, p_gap=0.0)
            g.check(N=N, T=T, acc=acc, dst=0, stride=yl, ys=ys, before=before, st=st)
            wr = st["wr"]
            span = wr[-1][0] - wr[0][0] + 1
            # source-limited rate: nch rows per cycle -> nch * T / 16 lines per cycle
            src = min(1.0, nch * T / 16)
            lat = wr[0][0] - st["first_res"]
            tail = st["done"] - st["last_res"]
            metrics.append(dict(acc=acc, tokens=T, rows=N, writes=len(wr), span=span,
                                first_write_latency=lat, completion_tail=tail,
                                issue_to_done=st["done"] - st["start"],
                                lines_per_cycle=len(wr) / span))
            out.append("%s T %d N %d: %d writes in %d cycles = %.3f / cycle (source bound %.3f); first "
                       "write %d cycles after the first result, done %d cycles after the last result" % (
                           "ACC  " if acc else "plain", T, N, len(wr), span, len(wr) / span, src, lat, tail))
            if src >= 1.0:
                assert len(wr) / span > 0.97, out[-1]
            if src > 1.0 and nch > 1:
                # oversubscribed write port: round robin keeps every cluster progressing, so the
                # clusters' last results are close together (fixed priority would starve some)
                lc = [x for x in st["last_c"] if x is not None]
                spread = (max(lc) - min(lc)) / (st["last_res"] - st["first_res"])
                out.append("  per-cluster completion spread %.3f of the run" % spread)
                assert spread < 0.2, "unfair job arbitration: cluster completion spread %.3f" % spread
    for ln in out:
        dut._log.info("gather_perf: %s", ln)
    save_metrics("dense", metrics)


@cocotb.test()
async def gather_comb(dut):
    g = await new_gtb(dut)
    rnd = random.Random(tb_seed() * 13 + 4)
    nch, TM = g.nch, g.tmax
    # bring the gather into an active state with partial work in flight, then probe
    N = 16 * nch + 3
    ys = rand_ys(rnd, TM, N)
    d = dut
    await RisingEdge(d.clk)
    d.acc.value = 1
    d.t.value = TM
    d.n.value = N
    d.dst_line.value = 0
    d.dst_stride.value = (N + 15) // 16
    d.expect_ch.value = (1 << nch) - 1
    d.start.value = 1
    await RisingEdge(d.clk)
    d.start.value = 0
    pokes = 0
    for probe in range(3):
        await ClockCycles(d.clk, rnd.randint(1, 5))
        await FallingEdge(d.clk)
        g.clock.stop()
        await Timer(1, "ns")
        outs = [(n, getattr(d, n)) for n in OUTPUTS]
        snap = {n: s.value for n, s in outs}
        bad = set()
        for name in INPUTS:
            sig = getattr(d, name)
            saved = sig.value
            for _ in range(2):
                sig.value = rnd.getrandbits(len(sig))
                pokes += 1
                await Timer(1, "ns")
                for n, s in outs:
                    if s.value != snap[n]:
                        bad.add((name, n))
            sig.value = saved
            await Timer(1, "ns")
        g.clock.start(start_high=True)
        assert not bad, "combinational input -> output paths: %s" % sorted(bad)
    # reset away the half-done MVM (the SPM model is reset too)
    d.res_valid.value = 0
    d.rst_n.value = 0
    await ClockCycles(d.clk, 3)
    await RisingEdge(d.clk)
    d.rst_n.value = 1
    await ClockCycles(d.clk, 2)
    dut._log.info("gather_comb: %d input pokes, no combinational path", pokes)


@cocotb.test()
async def gather_pair_jobs(dut):
    """Complete ATT records isolate pair bandwidth, origin completion and reset flush.

    Both old/new configurations execute identical records and compare every byte. The
    integrated AT bench separately checks real score/MAXOUT/PV arithmetic against C.
    """
    p = json.loads(os.environ["TSR_GATHER_CFG"])
    if not p.get("HAS_ATT") or p["NUM_CH"] != 4:
        return
    g = await new_gtb(dut)
    d = dut
    for name in ("att", "att_maxout", "att_tq", "att_gs", "att_tact", "att_nseg", "att_g",
                 "att_hg0", "att_nhead", "att_mdst", "aj_valid", "aj"):
        getattr(d, name).value = 0
    aw = (g.L - 1).bit_length()
    jw = 1 + aw + 32 + 512
    metrics = []

    async def jobs(active, count=128, stall=False, abort=False, abort_after=80):
        g.p_wr = 0.0 if abort else 1.0
        before = bytes(g.spm)
        expected = bytearray(before)
        await RisingEdge(d.clk)
        d.att.value = 1
        d.expect_ch.value = sum(1 << c for c in active)
        d.start.value = 1
        await RisingEdge(d.clk)
        d.start.value = 0
        heads = [0] * 4
        accepted = 0
        wr0, done0, e0 = len(g.wr_log), len(g.done_edges), g.edge
        while len(g.done_edges) == done0:
            v, records = 0, 0
            for c in active:
                k = heads[c]
                if k < count:
                    ln = 128 + c * count + k
                    payload = int.from_bytes(bytes([(c * 41 + k) % 256]) * 64, "little")
                    record = (((int(k == count - 1) << aw) | ln) << 32 | 0xffff0000) << 512 | payload
                    records |= record << (c * jw)
                    v |= 1 << c
                    expected[64 * ln:64 * (ln + 1)] = payload.to_bytes(64, "little")
            d.aj_valid.value, d.aj.value = v, records
            await FallingEdge(d.clk)
            ready = int(d.aj_ready.value)
            for c in active:
                if v & ready & (1 << c):
                    heads[c] += 1
                    accepted += 1
            if len(active) > 1 and not stall and not abort:
                assert max(heads[c] for c in active) - min(heads[c] for c in active) <= 16, "unfair source arbitration"
            await RisingEdge(d.clk)
            elapsed = g.edge - e0
            if abort and elapsed == abort_after:
                if abort_after >= 80:
                    wqd = 1 << (p["SPM_RD_LAT"] + p["ADD_LAT"] + 5).bit_length()
                    # XSLR_REG: the link stage holds two flits more per pair (2 / GATHER_FLITS jobs)
                    xs = 2 * (2 // p.get("GATHER_FLITS", 1)) if p.get("XSLR_REG") else 0
                    assert accepted == wqd + 16 + xs, "must retain eight transport slots per pair under a full stall"
                    assert int(d.spm_wr_valid.value), "stalled output must be occupied at reset"
                d.aj_valid.value = 0
                d.rst_n.value = 0
                await ClockCycles(d.clk, 4)
                d.rst_n.value = 1
                g.p_wr = 1.0
                await ClockCycles(d.clk, 12)
                assert len(g.done_edges) == done0, "aborted operation completed after reset"
                assert len(g.wr_log) == wr0, "pre-reset jobs escaped after reset"
                return
            if stall:
                g.p_wr = 0.0 if 30 <= elapsed % 100 < 65 else 1.0
            assert elapsed < 10000, "tagged jobs failed to complete"
        d.aj_valid.value = 0
        assert all(heads[c] == count for c in active), "done before all source finals accepted"
        assert bytes(g.spm) == expected, "tag/data lost, duplicated or reordered within source"
        writes = g.wr_log[wr0:]
        assert len(writes) == len(active) * count
        assert len({ln for _, ln, _ in writes}) == len(writes), "duplicate write"
        for c in active:
            first = 128 + c * count
            assert [ln for _, ln, _ in writes if first <= ln < first + count] == list(range(first, first + count)), "source job order changed"
        span = writes[-1][0] - writes[0][0] + 1
        rate = len(writes) / span
        metrics.append(dict(active=list(active), stall=stall, writes=len(writes),
                            span=span, cycles=g.edge-e0, lines_per_cycle=rate))
        d._log.info("pair_jobs active=%s stall=%s writes=%d span=%d rate=%.4f cycles=%d",
                    active, stall, len(writes), span, rate, g.edge - e0)
        if not stall:
            # Two flits halve a single pair's peak. Both physical pairs together
            # still feed the one-line/cycle central writer. This is a bus-rate
            # assertion, not an end-to-end performance claim for skewed work.
            pairs = len({c % 2 for c in active})
            bound = min(1.0, pairs / p.get("GATHER_FLITS", 1))
            assert rate > 0.97 * bound, "pair link missed its service-rate bound"
        await ClockCycles(d.clk, 4)
        assert len(g.wr_log) == wr0 + len(writes), "writes after done"

    for active in ((0,), (1,), (2,), (3,), (0, 2), (1, 3), (0, 1, 2, 3)):
        await jobs(active)
    await jobs((0, 1, 2, 3), stall=True)
    # Reset across successive early transport phases as well as a fully blocked
    # queue. In a two-flit link this includes partially assembled packets.
    for edge in (5, 6, 7, 8, 9, 80):
        await jobs((0, 1, 2, 3), abort=True, abort_after=edge)
    await jobs((3,), count=19)
    save_metrics("jobs", metrics)

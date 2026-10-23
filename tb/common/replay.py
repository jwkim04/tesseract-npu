"""Replay a trace on a tsr_core DUT and compare bit-exactly (spec 검증 전략: RTL 코어 전체).

    from common import replay, trace
    @cocotb.test()
    async def core_traces(dut):
        await tbutil.start_clock_and_reset(dut)
        for tr in trace.load_all(TRACE_DIR):          # all with the DUT's configuration
            res = await replay.run_trace(dut, tr)

run_trace follows tsr_trace.h's replay rules (also implemented by tsr_trace_replay()):
  1. fresh MemModel with the trace's initial DRAM (cmd.bin, then init regions; zero elsewhere)
  2. CMD_ADDR, CMD_COUNT, ARG0..7, CTRL.START through the CSR port; wait for `done`
  3. STATUS, ERR_CODE, ERR_PC (and ERR_INFO when recorded) must equal the trace
  4. no memory traffic in flight once done is high (DONE implies a quiescent memory port)
  5. every expect region bit-exact; the written beat set equals the union of expect regions
  6. every beat read before being written lies in an init region or the command buffer
  7. (check_sources, default on) the port delivered exactly the LD beats to source DMA and exactly
     rows_c * ROW_BEATS per executed MVM to source WS, per channel (docs/rtl-m3.md §4)
The DUT must expose the tsr_core boundary (clk, rst_n, csr_*, done, m_*). Several traces may run
back to back on one DUT (START re-initialises the run state; SPM content is irrelevant because
programs never read SPM they did not write).
"""
from __future__ import annotations

import logging

from . import isa
from .csr_driver import CsrDriver
from .mem_model import MemModel
from .tbutil import wait_level

log = logging.getLogger("tsr.replay")


class TraceMismatch(AssertionError):
    pass


def estimate_cycles(tr) -> int:
    """Generous run-time bound for a trace (cycles), used as the default timeout."""
    cyc = 20000 + 200 * tr.cmd_count
    for pc in range(tr.executed_count()):          # the faulting instruction does no work
        w = tr.insn(pc)
        op = w[0] & 0xFF
        if op in (isa.OP_LD, isa.OP_ST):
            cyc += 64 * min(w[3], 1 << 20) // 8 + 8 * min(w[3], 1 << 20)
        elif op == isa.OP_QACT:
            K, T = min(w[3], 65280), min(w[4], 8)
            cyc += T * (K // 16 + (K // 32) * 120 + 400)
        elif op == isa.OP_MVM:
            # Weight-bound (docs/rtl-m3.md §9.1): rows_c * ROW_BEATS beats per channel at a slow
            # memory's pace, plus the broadcast, short-row tree stalls and the gather.
            fmt, K, T, N = (w[0] >> 8) & 1, min(w[5] & 0xFFFF, 65280), min((w[5] >> 16) & 0xF, 8), w[4]
            bs = 256 if fmt else 32
            if K == 0 or K % bs or T == 0:
                continue
            N = min(N, 1 << 22)
            R = -(-N // tr.num_ch)
            rbeats = isa.row_beats(fmt, K)
            cyc += 4000 + 8 * T * isa.qa_lines(fmt, K) + R * (24 * rbeats + 200) + 40 * T * (N // 16 + 1)
    return cyc


def compare(tr, mem: MemModel, final: dict, *, check_reads: bool = True, check_sources: bool = False) -> list:
    """Every tsr_trace.h replay comparison for a finished run. Returns a list of error strings.
    final: dict with status, err_code, err_pc, err_info (CsrDriver.final_state() or a model).
    check_sources: the memory port delivered exactly the LD beats to source DMA and exactly the
    MVM weight beats (rows_c * ROW_BEATS per executed MVM) to source WS on every channel
    (Trace.expected_read_beats; needs a model whose reads went through the MemModel port)."""
    errs = []
    if check_sources:
        got = mem.src_beats
        for c, want in enumerate(tr.expected_read_beats()):
            for src, n in want.items():
                if got[c][src] != n:
                    errs.append("ch%d: %d read beats for source %s, expected %d" % (
                        c, got[c][src], {isa.SRC_DMA: "DMA", isa.SRC_WS: "WS"}[src], n))
    want = dict(status=tr.status, err_code=tr.err_code, err_pc=tr.err_pc)
    if tr.err_info is not None:
        want["err_info"] = tr.err_info
    for k, v in want.items():
        if final[k] != v:
            errs.append("%s = %d (0x%x), expected %d (0x%x)" % (k, final[k], final[k], v, v))
    exp_beats = tr.expect_beats()
    for c in range(tr.num_ch):
        extra = sorted(mem.written[c] - exp_beats[c])
        missing = sorted(exp_beats[c] - mem.written[c])
        if extra:
            errs.append("ch%d: %d beat(s) written outside the expect regions, first beat 0x%x"
                        % (c, len(extra), extra[0]))
        if missing:
            errs.append("ch%d: %d expected beat(s) not written, first beat 0x%x" % (c, len(missing), missing[0]))
    for r in tr.expect:
        got = mem.read_bytes(r.ch, r.offset, r.size)
        if got != r.data:
            i = next(i for i in range(r.size) if got[i] != r.data[i])
            nd = sum(1 for j in range(i, r.size) if got[j] != r.data[j])
            errs.append("expect mismatch ch%d byte 0x%x: got 0x%02x expected 0x%02x (%d bytes differ)"
                        % (r.ch, r.offset + i, got[i], r.data[i], nd))
    if check_reads:
        allowed = tr.allowed_read_beats()
        for c in range(tr.num_ch):
            bad = sorted(mem.read_unwritten[c] - allowed[c])
            if bad:
                errs.append("ch%d: read %d beat(s) the trace does not initialise, first 0x%x" % (c, len(bad), bad[0]))
    return errs


async def run_trace(dut, tr, *, csr: CsrDriver | None = None, mem_kwargs: dict | None = None,
                    seed: int = 1, timeout_cycles: int | None = None, check_reads: bool = True,
                    check_sources: bool = True, stop_model: bool = True) -> dict:
    kw = dict(seed=seed)
    kw.update(mem_kwargs or {})
    mem = MemModel.for_trace(dut, tr, **kw).start()
    csr = csr or CsrDriver(dut)
    try:
        await csr.start_run(tr.cmd_addr_beat, tr.cmd_count, tr.args)
        timeout = timeout_cycles or estimate_cycles(tr)
        cycles = await wait_level(dut.clk, dut.done, 1, timeout)
        st = await csr.final_state()
        mem.check(idle=True)
        errs = compare(tr, mem, st, check_reads=check_reads, check_sources=check_sources)
        if errs:
            raise TraceMismatch("%s: %s" % (tr.name, "; ".join(errs)))
        log.info("%s ok: %d insns, status %d err %s, %d cycles", tr.name, tr.cmd_count, st["status"],
                 isa.ERR_NAMES.get(st["err_code"]), cycles)
        return dict(cycles=cycles, final=st, mem_stats=mem.stats())
    finally:
        if stop_model:
            mem.stop()

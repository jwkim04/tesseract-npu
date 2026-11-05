"""Helpers of the tb/core bench (W4-A): the firmware build, the scenario (DRAM contents, the firmware's
arguments, the regions to compare), the host CSR sequence (EX §3 image load with VERIFY, boot, the
wait for the runtime's DONE, the final register snapshot) and its execution on the simulator
(libtsr_sim.so through tb/issue/issue_sim.py: the built-in hart runs the same image).

The same host sequence runs on the RTL (test_core.py, cocotb, tsr_core through its CSR port with
tb/common/mem_model.py on the memory ports); everything compared is schedule-independent: DRAM
regions, the SPM dump (written to DRAM by the firmware's last op), the LCQ record, the log words,
RT_STATE / RT_INFO, STATUS's architectural bits, Q_DONE / Q_ISSUED / Q_ERR_CODE per unit, Q_PIDX and
the identification registers.
"""
from __future__ import annotations

import ctypes
import os
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TB = REPO / "tb"
for p in (TB, TB / "ctrl", TB / "issue"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ctrl_tb                      # noqa: E402  (tb/ctrl: toolchain, ELF helpers)
import issue_sim as S               # noqa: E402  (tb/issue: the simulator bridge)
from common import isa, mvm_model   # noqa: E402
from common import tables as TT     # noqa: E402

FW_DIR = HERE / "fw"
CTRL_FW = TB / "ctrl" / "fw"
RT_DONE = 0xD0E

# The bench configuration: a reduced core (2 channels, 256 KiB SPM) with the MVM, the VU and attention
# (W4-B): CAPS VU | ATT (TOPK and DQROW are M8 RTL and stay off), the U200 queue depths (VU 16, unit 4
# reserved).
CAPS_U200 = 0x21
NCH = int(os.environ.get("TB_CORE_NCH", "2"))
TMAX = int(os.environ.get("TB_CORE_TMAX", "2"))
CFG = S.config(num_ch=NCH, t_max=TMAX, k_max=4096, spm_bytes=256 * 1024, ch_bytes=1 << 30, addr_beat_w=24,
               caps=CAPS_U200)
QD = (8, 4, 4, 16, 0, 4)
PARAMS = {"NUM_CH": NCH, "T_MAX": TMAX, "K_MAX": 4096, "SPM_BYTES": 256 * 1024, "CH_BYTES": "64'd1073741824",
          "ADDR_BEAT_W": 24}

ARG_NAMES = ("K", "T", "N1", "N2", "X_BEAT", "W1_OFF", "W2_OFF", "X_LINE", "QA1_LINE", "QA2_LINE", "Y1_LINE",
             "Y2_LINE", "X_LINES", "Y1_LINES", "Y2_LINES", "Y1_DST", "Y2_DST", "DUMP_DST", "DUMP_LINES",
             "SCRATCH_LINE", "BG_SRC", "SPM_LINES")


def build_fw(out_dir: Path, src: Path | None = None) -> Path:
    """tb/core/fw/fw_core.c with tb/ctrl's crt0.S / link.ld and the pinned xPack GCC."""
    src = src or FW_DIR / "fw_core.c"
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = out_dir / (src.stem + ".elf")
    deps = [CTRL_FW / "crt0.S", CTRL_FW / "link.ld", src]
    if elf.exists() and all(elf.stat().st_mtime >= s.stat().st_mtime for s in deps):
        return elf
    cmd = [str(ctrl_tb.TOOLS) + "gcc"] + ctrl_tb.FW_FLAGS + [
        "-I" + str(REPO / "sw" / "sim" / "include"), "-T", str(CTRL_FW / "link.ld"), str(CTRL_FW / "crt0.S"),
        str(src), "-lgcc", "-o", str(elf)]
    subprocess.run(cmd, check=True)
    return elf


# ---------------------------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------------------------
def scenario(seed: int, K: int = 512, T: int = 2, N1: int = 40, N2: int = 24) -> dict:
    """DRAM writes [(ch, byte_off, bytes)], the firmware arguments, the regions to compare."""
    rnd = random.Random(seed)
    nch = CFG["num_ch"]
    outer = nch - 1  # channel 3 in the four-channel locality bench; channel 1 otherwise
    fq8, fq4 = isa.FMT_Q8_0, isa.FMT_Q4_K
    xl = T * K * 4 // 64
    ql1, ql2 = isa.qa_lines(isa.QA_A32, K), isa.qa_lines(isa.QA_A256, K)
    yl1, yl2 = (N1 + 15) // 16, (N2 + 15) // 16
    X = 0
    QA1 = X + xl
    QA2 = QA1 + T * ql1
    Y1 = QA2 + T * ql2 + 1                 # one line gap (keeps the background bytes)
    Y2 = Y1 + T * yl1
    SCR = Y2 + T * yl2
    dump = SCR + 2
    assert dump * 64 <= CFG["spm_bytes"]
    io, wo = isa.REGION_IO_OFF // 64, isa.REGION_WEIGHT_OFF // 64
    # beats inside the 1 GiB channel (ADDR_BEAT_W 24: 2^24 beats)
    x_beat, bg_beat = io, io + 2048
    w1_off, w2_off = wo, wo + 4096
    y1_beat, y2_beat, dump_beat = io + 8192, io + 8192 + 512, io + 16384
    # activations: f32, normal values of moderate range
    x = struct.pack("<%df" % (T * K), *[rnd.gauss(0.0, 1.5) for _ in range(T * K)])
    rows1 = [mvm_model.rand_row(fq8, K, rnd) for _ in range(N1)]
    rows2 = [mvm_model.rand_row(fq4, K, rnd) for _ in range(N2)]
    writes = [(0, x_beat * 64, x), (outer, bg_beat * 64, rnd.randbytes(64 * dump))]
    for c, d in enumerate(mvm_model.layout_channels(fq8, K, rows1, nch)):
        if d:
            writes.append((c, w1_off * 64, d))
    for c, d in enumerate(mvm_model.layout_channels(fq4, K, rows2, nch)):
        if d:
            writes.append((c, w2_off * 64, d))
    # old contents of the store targets (the stores must overwrite exactly their lines)
    writes += [(outer, y1_beat * 64, rnd.randbytes(64 * (T * yl1 + 2))), (0, y2_beat * 64, rnd.randbytes(64 * (T * yl2 + 2))),
               (outer, dump_beat * 64, rnd.randbytes(64 * (dump + 2)))]
    a = dict(K=K, T=T, N1=N1, N2=N2, X_BEAT=x_beat, W1_OFF=w1_off, W2_OFF=w2_off, X_LINE=X, QA1_LINE=QA1,
             QA2_LINE=QA2, Y1_LINE=Y1, Y2_LINE=Y2, X_LINES=xl, Y1_LINES=T * yl1, Y2_LINES=T * yl2,
             Y1_DST=(outer << 28) | y1_beat, Y2_DST=(0 << 28) | y2_beat, DUMP_DST=(outer << 28) | dump_beat,
             DUMP_LINES=dump, SCRATCH_LINE=SCR, BG_SRC=(outer << 28) | bg_beat,
             SPM_LINES=CFG["spm_bytes"] // 64)          # the failing LD's line (SPM_RANGE)
    regions = [(outer, y1_beat * 64, 64 * (T * yl1 + 2)), (0, y2_beat * 64, 64 * (T * yl2 + 2)),
               (outer, dump_beat * 64, 64 * (dump + 2)), (0, x_beat * 64, len(x))]
    return dict(writes=writes, args=[a[n] for n in ARG_NAMES], regions=regions, spm_lines=dump)


# ---------------------------------------------------------------------------------------------
# Host CSR sequence (EX §3.3 - §3.4): a list of steps run by an executor (simulator or RTL)
# ---------------------------------------------------------------------------------------------
ID_REGS = (TT.CSR_ID, TT.CSR_VERSION, TT.CSR_CFG0, TT.CSR_CFG1, TT.CSR_CFG2, TT.CSR_CAPS, TT.CSR_CFG3,
           TT.CSR_CTL_ID, TT.CSR_RV_CFG, TT.CSR_Q_DEPTH, TT.CSR_LQ_CFG, TT.CSR_PERF_INFO, TT.CSR_LOG_CFG)
# STATUS bits that do not depend on the schedule once everything is idle (RV_SLEEP / RV_HALTED and
# the clock bits are levels of the implementation; ACTIVE, QIDLE, QERR, SR_BUSY, the pending bits,
# PLAT_ERR, HOLD_SRC and the sticky errors are architectural)
STATUS_MASK = 0xFFFFFFFF & ~(TT.STATUS_RV_SLEEP | TT.STATUS_RV_CLK_OK | TT.STATUS_QQUIESCENT)


def load_words(elf, args_addr, args):
    """[(addr, [words])] runs: the PT_LOAD segments, then the argument block."""
    runs = []
    for pa, data in ctrl_tb.elf_segments(elf):
        runs.append((pa, list(struct.unpack("<%dI" % (len(data) // 4), data))))
    runs.append((args_addr, [v & 0xFFFFFFFF for v in args]))
    return runs


def final_regs():
    """CSR offsets read after the run (compared with the simulator)."""
    offs = [TT.CSR_STATUS, TT.CSR_RT_STATE, TT.CSR_RT_INFO, TT.CSR_LCQ_WR, TT.CSR_LCQ_RD, TT.CSR_LOG_WR,
            TT.CSR_LOG_RD, TT.CSR_LOG_DROPPED, TT.CSR_LQ_SEQ, TT.CSR_LQ_TAKEN, TT.CSR_LCQ_ANSWERED, TT.CSR_Q_PIDX,
            TT.CSR_Q_FIFO]
    for u in range(6):
        offs += [TT.CSR_Q_DONE_BASE + 4 * u, TT.CSR_Q_ISSUED_BASE + 4 * u, TT.CSR_Q_ERR_CODE_BASE + 16 * u]
    offs += [TT.CSR_LCQ_REC_BASE + 4 * j for j in range(8)]
    offs += [TT.CSR_LOG_RING_BASE + 4 * j for j in range(3)]
    return offs


def expected_id(off: int, simulator_value: int) -> int:
    # The simulator keeps the default timing model. Hardware lane-count fields
    # describe the selected implementation; all other identity bits still match.
    if off == TT.CSR_CFG3 and "TB_CORE_VU_FRONT" in os.environ:
        front = int(os.environ["TB_CORE_VU_FRONT"])
        back = int(os.environ["TB_CORE_VU_BACK"])
        return (simulator_value & ~0x3FF) | front | (back << 5)
    return simulator_value


def check_ids(vals: dict, sim_vals: dict, errs: list):
    for o in ID_REGS:
        want = expected_id(o, sim_vals[o])
        if vals[o] != want:
            errs.append("CSR %#05x: RTL %#010x / expected %#010x" % (o, vals[o], want))


def compare_final(rtl: dict, sim: dict, errs: list):
    for o in final_regs():
        a, b = rtl[o], sim[o]
        if o == TT.CSR_STATUS:
            a, b = a & STATUS_MASK, b & STATUS_MASK
        if a != b:
            errs.append("CSR %#05x after the run: RTL %#010x / simulator %#010x" % (o, a, b))


# ---------------------------------------------------------------------------------------------
# Simulator executor
# ---------------------------------------------------------------------------------------------
_EXTRA = False


def _lib():
    global _EXTRA
    L = S.lib()
    if not _EXTRA:
        for name, res, argt in (
                ("tsr_sim_dram_write", ctypes.c_int, [S.vp, S.u32, S.u64, ctypes.c_char_p, S.u64]),
                ("tsr_sim_dram_read", ctypes.c_int, [S.vp, S.u32, S.u64, ctypes.c_void_p, S.u64]),
                ("tsr_sim_spm", ctypes.POINTER(ctypes.c_uint8), [S.vp]),
                ("tsr_sim_rv_instret", S.u64, [S.vp]),
                ("tsr_sim_rv_pc", S.u32, [S.vp])):
            f = getattr(L, name)
            f.restype = res
            f.argtypes = argt
        _EXTRA = True
    return L


def run_sim(sc: dict, elf: Path, max_cycles: int = 50_000_000) -> dict:
    """The host sequence on the simulator; returns its observations."""
    L = _lib()
    sim = S.Sim(CFG, QD)
    try:
        for ch, off, data in sc["writes"]:
            assert L.tsr_sim_dram_write(sim.sim, ch, off, data, len(data)) == 0
        out = {"ids": {o: sim.csr_read(o) for o in ID_REGS}}
        syms = ctrl_tb.elf_symbols(elf)
        crcs = []
        for addr, words in load_words(elf, syms["args"], sc["args"]):
            sim.csr_write(TT.CSR_LOAD_ADDR, addr)
            for w in words:
                sim.csr_write(TT.CSR_LOAD_DATA, w)
            crc_rx = sim.csr_read(TT.CSR_LOAD_CRC)
            sim.csr_write(TT.CSR_LOAD_ADDR, addr)
            sim.csr_write(TT.CSR_LOAD_COUNT, len(words))
            sim.csr_write(TT.CSR_LOAD_CTRL, 1)
            n = 0
            while sim.csr_read(TT.CSR_LOAD_STATUS) & 1:
                sim.advance(64)
                n += 1
                assert n < 100000, "VERIFY did not finish"
            crcs.append((crc_rx, sim.csr_read(TT.CSR_LOAD_CRC), zlib.crc32(struct.pack("<%dI" % len(words), *words))))
        out["crcs"] = crcs
        out["status_loaded"] = sim.csr_read(TT.CSR_STATUS)
        sim.csr_write(TT.CSR_RV_BOOT, 0)
        sim.csr_write(TT.CSR_CTRL, TT.CTRL_RV_RUN)
        t = 0
        while sim.csr_read(TT.CSR_RT_STATE) != RT_DONE:
            sim.advance(2000)
            t += 2000
            if t > max_cycles:
                raise AssertionError("simulator: RT_STATE %#x after %d cycles (pc %#x)" % (
                    sim.csr_read(TT.CSR_RT_STATE), t, L.tsr_sim_rv_pc(sim.sim)))
        assert sim.run_until_idle(1 << 30)
        sim.advance(5000)
        out["regs"] = {o: sim.csr_read(o) for o in final_regs()}
        out["dram"] = {}
        for ch, off, size in sc["regions"]:
            buf = ctypes.create_string_buffer(size)
            assert L.tsr_sim_dram_read(sim.sim, ch, off, buf, size) == 0
            out["dram"][(ch, off)] = buf.raw
        spm = L.tsr_sim_spm(sim.sim)
        out["spm"] = bytes(spm[:64 * sc["spm_lines"]])
        out["instret"] = L.tsr_sim_rv_instret(sim.sim)
        out["cycles"] = sim.now()
        return out
    finally:
        sim.close()


# ---------------------------------------------------------------------------------------------
# M7 layer scenario (W4-B): the production image (build/dev/fw/tsr-fw.elf) runs an op list of the M7 kinds
# (RMS_NORM, MUL_MAT, ROPE, SET_ROWS_KV, FLASH_ATTN, GLU) through the graph kernel on the RTL core; the
# reference is the simulator's run of the same launch, dumped by sw/ctrl/tests/test_fw_m7_layer.cpp
# (mode `dump`, the `tiny` group: a decoder layer of 128 / 2 heads of 64 / 1 KV head / n_ff 256 on a 2-channel
# device with T_MAX 2 and 1 MiB of SPM, CAPS VU | ATT).
# ---------------------------------------------------------------------------------------------
M7_CASES = ("tiny-decode", "tiny-prefill")
M7_BIN = REPO / "build" / "dev" / "sw" / "ctrl" / "host" / "test_fw_m7_layer"
M7_ELF = REPO / "build" / "dev" / "fw" / "tsr-fw.elf"
M7_OUT = REPO / "build" / "cocotb" / "core_m7"


def m7_scenario(case: str) -> Path:
    """The scenario directory of a case (generated by the simulator run when missing or older than the
    test binary or the image)."""
    d = M7_OUT / case
    meta = d / "meta.txt"
    if meta.exists() and all(meta.stat().st_mtime >= p.stat().st_mtime for p in (M7_BIN, M7_ELF)):
        return d
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(M7_BIN), str(M7_ELF), "dump", case, str(d)], check=True)
    return d


def parse_m7(d: Path) -> dict:
    sc = dict(ins=[], outs=[], planes=[], csrs=[], ids=[], done=[], launch=None, cfg=None, dbg=None, hist=[],
              trw_cfg=None)
    for line in (d / "meta.txt").read_text().splitlines():
        f = line.split()
        v = [int(x) for x in f[1:]]
        if f[0] == "cfg":
            sc["cfg"] = dict(num_ch=v[0], t_max=v[1], k_max=v[2], spm_bytes=v[3], ch_bytes=v[4], addr_beat_w=v[5],
                             caps=v[6])
        elif f[0] == "launch":
            sc["launch"] = v
        elif f[0] in ("in", "out", "plane", "csr", "id"):
            sc[{"in": "ins", "out": "outs", "plane": "planes", "csr": "csrs", "id": "ids"}[f[0]]].append(v)
        elif f[0] == "done":
            sc["done"] = v
        elif f[0] == "dbg":                       # HIST_HEAD, TRW_WPTR, TRW_DROPS, PERF TRW_RECORDS, PERF TRW_DROPS
            sc["dbg"] = v
        elif f[0] == "hist":                      # push index, then the DBG_HIST_* registers (HIST_OFFS)
            sc["hist"].append(v[1:])
        elif f[0] == "trw":
            sc["trw_cfg"] = v
    sc["data"] = (d / "data.bin").read_bytes()
    sc["expect"] = (d / "expect.bin").read_bytes()
    raw = (d / "trw.bin").read_bytes()
    sc["trw"] = [[int.from_bytes(raw[32 * k + 4 * i:32 * k + 4 * i + 4], "little") for i in range(8)]
                 for k in range(len(raw) // 32)]
    return sc


# ---------------------------------------------------------------------------------------------
# DB block at core level (W4-B): the fw_core program with the DBG block configured through the host port.
# One host program (db_prog) runs unchanged on the simulator (SimHost) and on the RTL (test_core.py's Host),
# so every observation has the same definition on both sides.
# ---------------------------------------------------------------------------------------------
DB_RING_BEAT = (isa.REGION_IO_OFF // 64) + 0x20000     # the trace ring: channel 0, 64 lines, 16-beat aligned
DB_RING_LINES = 64
DB_WAIT = int(os.environ.get("TB_DB_WAIT", "8000"))     # polls of 500 core cycles for the stops / the firmware
DB_TBP_OPC = 0x20                                       # tile breakpoint 0: the first MVM, before it starts
STOP_REGS = [TT.CSR_DBG_ID, TT.CSR_DBG_CFG, TT.CSR_DBG_STATUS, TT.CSR_DBG_STOP_PIDX, TT.CSR_DBG_STOP_AT,
             TT.CSR_DBG_STEP_N, TT.CSR_DBG_NEXT_PIDX, TT.CSR_DBG_STEP_PIDX, TT.CSR_DBG_MATCHQ, TT.CSR_DBG_HIST_HEAD,
             TT.CSR_DBG_TRW_BASE, TT.CSR_DBG_TRW_LINES, TT.CSR_DBG_TRW_MASK, TT.CSR_Q_PIDX]
STOP_REGS += [TT.CSR_DBG_TBP_CTL(0), TT.CSR_DBG_TBP_A(0), TT.CSR_DBG_TBP_B(0), TT.CSR_DBG_TBP_HITS(0)]
STOP_REGS += [TT.CSR_DBG_U_OLDEST(u) for u in range(8)] + [TT.CSR_DBG_U_OPIDX(u) for u in range(8)]
for _u in range(6):
    STOP_REGS += [TT.CSR_Q_DONE(_u), TT.CSR_Q_ERR_SEQ(_u), TT.CSR_Q_ERR_CODE(_u)]
HIST_OFFS = [TT.CSR_DBG_HIST_INFO, TT.CSR_DBG_HIST_PIDX, TT.CSR_DBG_HIST_TICKET, TT.CSR_DBG_HIST_DEP0,
             TT.CSR_DBG_HIST_DEP1, TT.CSR_DBG_HIST_PC] + [TT.CSR_DBG_HIST_W_BASE + 4 * i for i in range(8)]
# DBG_STATUS: RVWP_BUSY is the RTL's handshake in flight; RV_HALTED / RV_SLEEP are levels of the hart
DB_STATUS_MASK = 0xFFFFFFFF & ~(TT.DBG_STATUS_RVWP_BUSY | TT.DBG_STATUS_RV_SLEEP | TT.DBG_STATUS_RV_HALTED)
U_STATE_MASK = TT.DBG_U_STATE_ACCEPTED | TT.DBG_U_STATE_HOLD | TT.DBG_U_STATE_OUTSTANDING


def run_sync(coro):
    """Runs a coroutine that never suspends (SimHost's) to its end."""
    try:
        while True:
            coro.send(None)
    except StopIteration as e:
        return e.value


class SimHost:
    """The host port of a simulator instance with the interface of test_core.py's Host (async rd / wr, idle,
    mem_read), for db_prog."""

    def __init__(self, sim):
        self.sim = sim

    async def rd(self, off):
        return self.sim.csr_read(off)

    async def wr(self, off, val):
        self.sim.csr_write(off, val)

    async def idle(self, n):
        self.sim.advance(n)

    async def mem_read(self, ch, off, size):
        buf = ctypes.create_string_buffer(size)
        assert _lib().tsr_sim_dram_read(self.sim.sim, ch, off, buf, size) == 0
        return buf.raw


async def db_prog(h, sc, elf, errs=None):
    """The host sequence of core_db. Returns the observations (everything schedule-independent)."""
    o = {}
    syms = ctrl_tb.elf_symbols(elf)
    crcs = []
    for addr, words in load_words(elf, syms["args"], sc["args"]):
        await h.wr(TT.CSR_LOAD_ADDR, addr)
        for w in words:
            await h.wr(TT.CSR_LOAD_DATA, w)
        crc_rx = await h.rd(TT.CSR_LOAD_CRC)
        crcs.append((crc_rx, zlib.crc32(struct.pack("<%dI" % len(words), *words))))
    o["crcs"] = crcs
    # the DBG block: identification, the DMI gate (DB-2), history, the trace writer, one tile breakpoint
    o["dbg_id"] = await h.rd(TT.CSR_DBG_ID)
    o["cfg_reset"] = await h.rd(TT.CSR_DBG_CFG)
    await h.wr(TT.CSR_DBG_CFG, TT.DBG_CFG_HIST_EN | TT.DBG_CFG_TRW_EN)           # DMI_EN = 0
    await h.wr(TT.CSR_DMI_BASE + 4 * 0x10, 1)                                    # dmcontrol: dropped, DMI_BLOCKED
    await h.idle(200)
    o["blocked"] = await h.rd(TT.CSR_DBG_STATUS) & TT.DBG_STATUS_DMI_BLOCKED
    await h.wr(TT.CSR_DBG_STATUS, TT.DBG_STATUS_DMI_BLOCKED)                     # W1C
    o["cleared"] = await h.rd(TT.CSR_DBG_STATUS) & TT.DBG_STATUS_DMI_BLOCKED
    await h.wr(TT.CSR_DBG_CFG, TT.DBG_CFG_HIST_EN | TT.DBG_CFG_TRW_EN | TT.DBG_CFG_DMI_EN)
    await h.idle(400)
    o["dmcontrol_dropped"] = await h.rd(TT.CSR_DMI_BASE + 4 * 0x10)                # the blocked write did not arrive
    await h.wr(TT.CSR_DMI_BASE + 4 * 0x10, 1)                                    # dmcontrol.dmactive
    await h.idle(400)
    o["dmcontrol"] = await h.rd(TT.CSR_DMI_BASE + 4 * 0x10)
    o["dmstatus"] = await h.rd(TT.CSR_DMI_BASE + 4 * 0x11)
    await h.wr(TT.CSR_DBG_TRW_BASE, DB_RING_BEAT)
    await h.wr(TT.CSR_DBG_TRW_LINES, DB_RING_LINES)
    await h.wr(TT.CSR_DBG_TBP_A(0), TT.TBP_OPC_A_OPC_MASK | DB_TBP_OPC)
    await h.wr(TT.CSR_DBG_TBP_B(0), 0)
    await h.wr(TT.CSR_DBG_TBP_CTL(0), TT.DBG_TBP_CTL_EN | (TT.DBG_TBP_CTL_KIND_OPCODE << 1))
    await h.wr(TT.CSR_RV_BOOT, 0)
    await h.wr(TT.CSR_CTRL, TT.CTRL_RV_RUN)
    # the stops at the two MVMs (the breakpoint matches both): at every DRAINED stop observe, RELEASE, until the
    # firmware is done
    o["stops"] = []
    last_t = None
    n = 0
    while True:
        st = await h.rd(TT.CSR_DBG_STATUS)
        if st & TT.DBG_STATUS_STOP_ACTIVE and st & TT.DBG_STATUS_DRAINED and \
                (await h.rd(TT.CSR_DBG_STOP_PIDX)) != last_t:
            snap = {off: await h.rd(off) for off in STOP_REGS}
            snap[TT.CSR_DBG_STATUS] &= DB_STATUS_MASK
            snap["u_state"] = [await h.rd(TT.CSR_DBG_U_STATE(u)) & U_STATE_MASK for u in range(8)]
            snap["hist"] = []
            for p in range(snap[TT.CSR_DBG_HIST_HEAD]):
                await h.wr(TT.CSR_DBG_HIST_IDX, p)
                snap["hist"].append([await h.rd(x) for x in HIST_OFFS])
            o["stops"].append(snap)
            last_t = snap[TT.CSR_DBG_STOP_PIDX]
            await h.wr(TT.CSR_DBG_CTRL, TT.DBG_CTRL_RELEASE)
        elif (await h.rd(TT.CSR_RT_STATE)) == RT_DONE:
            break
        await h.idle(500)
        n += 1
        assert n < DB_WAIT, "RT_STATE %#x, DBG_STATUS %#x after %d stops" % (await h.rd(TT.CSR_RT_STATE), st, len(o["stops"]))
    n = 0
    while not ((await h.rd(TT.CSR_STATUS)) & TT.STATUS_QIDLE):
        await h.idle(100)
        n += 1
        assert n < 10000, "QIDLE not reached"
    await h.idle(3000)
    await h.wr(TT.CSR_DBG_CTRL, TT.DBG_CTRL_TRW_FLUSH)
    await h.idle(3000)
    o["regs"] = {off: await h.rd(off) for off in final_regs()}
    # PERF event counters of the DB block (TRW_RECORDS 61, TRW_DROPS 62) and RETIRED
    o["perf"] = {i: await h.rd(TT.CSR_PERF_LO(i)) for i in (1, 61, 62)}
    o["db_final"] = {off: await h.rd(off) for off in (TT.CSR_DBG_STATUS, TT.CSR_DBG_NEXT_PIDX, TT.CSR_DBG_HIST_HEAD,
                                                      TT.CSR_DBG_TRW_WPTR, TT.CSR_DBG_TRW_RPTR, TT.CSR_DBG_TRW_DROPS,
                                                      TT.CSR_DBG_TBP_HITS(0), TT.CSR_DBG_STOP_PIDX)}
    o["db_final"][TT.CSR_DBG_STATUS] &= DB_STATUS_MASK
    head = o["db_final"][TT.CSR_DBG_HIST_HEAD]
    o["hist"] = []
    for p in range(head):
        await h.wr(TT.CSR_DBG_HIST_IDX, p)
        o["hist"].append([await h.rd(x) for x in HIST_OFFS])
    wptr = o["db_final"][TT.CSR_DBG_TRW_WPTR]
    assert 0 < wptr <= 2 * DB_RING_LINES, "trace ring: %d records (the check does not handle a wrapped ring)" % wptr
    raw = await h.mem_read(0, DB_RING_BEAT * 64, wptr * 32 + (-wptr * 32) % 64)
    o["trw"] = [[int.from_bytes(raw[32 * k + 4 * i:32 * k + 4 * i + 4], "little") for i in range(8)]
                for k in range(wptr)]
    o["dram"] = {(ch, off): await h.mem_read(ch, off, size) for ch, off, size in sc["regions"]}
    return o


def trw_view(recs):
    """Trace records without their times: OP records per unit, MARK, STOP, DROP, RVHALT (DB §8.3)."""
    out = dict(op=[[] for _ in range(8)], mark=[], stop=[], drop=[], rvhalt=[], other=[])
    for w in recs:
        k = w[0] & 0xF
        if k == TT.TRW_W0_KIND_OP:
            out["op"][(w[0] >> 4) & 0xF].append((w[0], w[1], w[2]))
        elif k == TT.TRW_W0_KIND_MARK:
            out["mark"].append((w[1], w[2]))
        elif k == TT.TRW_W0_KIND_STOP:
            out["stop"].append((w[1], w[2], w[5]))
        elif k == TT.TRW_W0_KIND_DROP:
            out["drop"].append(w[1])
        elif k == TT.TRW_W0_KIND_RVHALT:
            out["rvhalt"].append(w[2])
        else:
            out["other"].append(tuple(w))
    return out


def compare_db(rtl: dict, sim: dict, errs: list):
    for k in ("crcs", "dbg_id", "cfg_reset", "blocked", "cleared", "dmcontrol_dropped", "dmcontrol", "dmstatus"):
        if rtl[k] != sim[k]:
            errs.append("%s: RTL %s / simulator %s" % (k, rtl[k], sim[k]))
    if len(rtl["stops"]) != len(sim["stops"]):
        errs.append("%d stops RTL / %d simulator" % (len(rtl["stops"]), len(sim["stops"])))
    for i, (a, b) in enumerate(zip(rtl["stops"], sim["stops"])):
        for off in STOP_REGS:
            if a[off] != b[off]:
                errs.append("stop %d: CSR %#05x RTL %#010x / simulator %#010x" % (i, off, a[off], b[off]))
        if a["u_state"] != b["u_state"]:
            errs.append("stop %d: DBG_Uu_STATE RTL %s / simulator %s" % (i, a["u_state"], b["u_state"]))
        if len(a["hist"]) != len(b["hist"]):
            errs.append("stop %d: %d history entries RTL / %d simulator" % (i, len(a["hist"]), len(b["hist"])))
        for p, (x, y) in enumerate(zip(a["hist"], b["hist"])):
            if x != y:
                errs.append("stop %d: hist[%d]: RTL %s / simulator %s" % (i, p, [hex(v) for v in x], [hex(v) for v in y]))
    key = "hist"
    if len(rtl[key]) != len(sim[key]):
        errs.append("%s: %d entries RTL / %d simulator" % (key, len(rtl[key]), len(sim[key])))
    for p, (a, b) in enumerate(zip(rtl[key], sim[key])):
        if a != b:
            errs.append("%s[%d]: RTL %s / simulator %s" % (key, p, [hex(x) for x in a], [hex(x) for x in b]))
    for off, v in rtl["db_final"].items():
        if v != sim["db_final"][off]:
            errs.append("after the run: CSR %#05x RTL %#010x / simulator %#010x" % (off, v, sim["db_final"][off]))
    compare_final(rtl["regs"], sim["regs"], errs)
    if rtl["perf"] != sim["perf"]:
        errs.append("PERF RETIRED / TRW_RECORDS / TRW_DROPS: RTL %s / simulator %s" % (rtl["perf"], sim["perf"]))
    for key, want in sim["dram"].items():
        if rtl["dram"][key] != want:
            errs.append("DRAM ch%d [%#x, +%#x): RTL differs from the simulator" % (key[0], key[1], len(want)))
    a, b = trw_view(rtl["trw"]), trw_view(sim["trw"])
    for k in a:
        if a[k] != b[k]:
            errs.append("trace records (%s) differ: RTL %s / simulator %s" % (k, a[k], b[k]))


def run_sim_db(sc: dict, elf: Path) -> dict:
    L = _lib()
    sim = S.Sim(CFG, QD)
    try:
        for ch, off, data in sc["writes"]:
            assert L.tsr_sim_dram_write(sim.sim, ch, off, data, len(data)) == 0
        o = run_sync(db_prog(SimHost(sim), sc, elf))
        sim.run_until_idle(1 << 30)
        return o
    finally:
        sim.close()

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

# The bench configuration: a reduced core (2 channels, 256 KiB SPM) with the MVM unit; VU / AT
# absent (CAPS 0, depth 0), as tsr_core builds them in W4-A.
CFG = S.config(num_ch=2, t_max=2, k_max=4096, spm_bytes=256 * 1024, ch_bytes=1 << 30, addr_beat_w=24, caps=0)
QD = (8, 4, 4, 0, 0, 4)
PARAMS = {"NUM_CH": 2, "T_MAX": 2, "K_MAX": 4096, "SPM_BYTES": 256 * 1024, "CH_BYTES": "64'd1073741824",
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
    writes = [(0, x_beat * 64, x), (1, bg_beat * 64, rnd.randbytes(64 * dump))]
    for c, d in enumerate(mvm_model.layout_channels(fq8, K, rows1, nch)):
        if d:
            writes.append((c, w1_off * 64, d))
    for c, d in enumerate(mvm_model.layout_channels(fq4, K, rows2, nch)):
        if d:
            writes.append((c, w2_off * 64, d))
    # old contents of the store targets (the stores must overwrite exactly their lines)
    writes += [(1, y1_beat * 64, rnd.randbytes(64 * (T * yl1 + 2))), (0, y2_beat * 64, rnd.randbytes(64 * (T * yl2 + 2))),
               (1, dump_beat * 64, rnd.randbytes(64 * (dump + 2)))]
    a = dict(K=K, T=T, N1=N1, N2=N2, X_BEAT=x_beat, W1_OFF=w1_off, W2_OFF=w2_off, X_LINE=X, QA1_LINE=QA1,
             QA2_LINE=QA2, Y1_LINE=Y1, Y2_LINE=Y2, X_LINES=xl, Y1_LINES=T * yl1, Y2_LINES=T * yl2,
             Y1_DST=(1 << 28) | y1_beat, Y2_DST=(0 << 28) | y2_beat, DUMP_DST=(1 << 28) | dump_beat,
             DUMP_LINES=dump, SCRATCH_LINE=SCR, BG_SRC=(1 << 28) | bg_beat,
             SPM_LINES=CFG["spm_bytes"] // 64)          # the failing LD's line (SPM_RANGE)
    regions = [(1, y1_beat * 64, 64 * (T * yl1 + 2)), (0, y2_beat * 64, 64 * (T * yl2 + 2)),
               (1, dump_beat * 64, 64 * (dump + 2)), (0, x_beat * 64, len(x))]
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


def check_ids(vals: dict, sim_vals: dict, errs: list):
    for o in ID_REGS:
        if vals[o] != sim_vals[o]:
            errs.append("CSR %#05x: RTL %#010x / simulator %#010x" % (o, vals[o], sim_vals[o]))


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

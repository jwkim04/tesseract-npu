"""ctypes wrapper of the ISA simulator (sw/sim/tsr_sim.c) — the oracle (spec 작업 규칙 2) — for the
testbenches that run programs on the simulator and on the RTL and compare the outcomes
(tb/cmdproc, tb/core). Moved here from tb/cmdproc (tb/cmdproc/simlib.py re-exports it).

Library: $TSR_SIM_LIB, else a shared library linked on first use from the CMake-built static
libraries build/dev/sw/sim/libtsr_sim.a + libtsr_ref.a (the code tsr-trace-gen / tsr-trace-replay
link), so no simulator source is compiled or touched here. The library goes into a private
directory keyed by the content of the two archives:
    <build root>/simlib/<sha256 of both archives>/libtsr_sim_ctypes.so
built from private copies of the archives in a temporary directory, verified (every symbol used
here must resolve) and atomically renamed into place. Concurrent test processes (several agents,
pytest workers) therefore never see a half-written library, never overwrite one another's library
with different content, and a rebuild of build/dev while a test runs cannot change a library that
is already loaded. An archive caught in the middle of a rewrite (size / mtime change while it is
read, or a failed link) is retried.

    sim = Sim(cfg)                       # cfg: tsr_trace.h config dict (isa.U200_V1 keys)
    res = sim.run(prog_bytes, cmd_addr, args, init=[(ch, byte_off, data), ...])
    res.final  -> dict(status, err_code, err_pc, err_info, pc)
    res.trace  -> common.trace.Trace (captured init/expect regions; replay.compare() accepts it)
    res.regs, res.spm, res.retired, res.perf (all PERF_* counter deltas of the run)
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import isa, trace
from .runner import REPO, build_root

SIM_A = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_sim.a"
REF_A = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_ref.a"
LIB_NAME = "libtsr_sim_ctypes.so"

u32, u64, i32 = C.c_uint32, C.c_uint64, C.c_int32


class TsrConfig(C.Structure):
    _fields_ = [("num_ch", u32), ("beat_bits", u32), ("lanes", u32), ("t_max", u32), ("k_max", u32),
                ("spm_bytes", u32), ("vl", u32), ("nireg", u32), ("nsreg", u32), ("loop_depth", u32),
                ("rd_outstanding", u32), ("ch_bytes", u64), ("addr_beat_w", u32), ("caps", u32)]


class TsrRegion(C.Structure):
    _fields_ = [("ch", u32), ("byte_off", u64), ("size", u64), ("data", C.POINTER(C.c_uint8))]


class TsrCapture(C.Structure):
    _fields_ = [("init", C.POINTER(TsrRegion)), ("n_init", C.c_size_t),
                ("expect", C.POINTER(TsrRegion)), ("n_expect", C.c_size_t)]


_SIG = {
    "tsr_sim_create": (C.c_void_p, [C.POINTER(TsrConfig), C.c_char_p]),
    "tsr_sim_destroy": (None, [C.c_void_p]),
    "tsr_sim_dram": (C.POINTER(C.c_uint8), [C.c_void_p, u32]),
    "tsr_sim_spm": (C.POINTER(C.c_uint8), [C.c_void_p]),
    "tsr_sim_dram_limit_beats": (u64, [C.c_void_p]),
    "tsr_sim_csr_read": (u32, [C.c_void_p, u32]),
    "tsr_sim_csr_write": (None, [C.c_void_p, u32, u32]),
    "tsr_sim_run": (u32, [C.c_void_p]),
    "tsr_sim_ireg": (i32, [C.c_void_p, u32]),
    "tsr_sim_perf": (u64, [C.c_void_p, u32]),
    "tsr_sim_capture_begin": (None, [C.c_void_p]),
    "tsr_sim_capture_end": (None, [C.c_void_p, C.POINTER(TsrCapture)]),
    "tsr_capture_free": (None, [C.POINTER(TsrCapture)]),
}


def _read_stable(p: Path) -> bytes:
    """Read a file whose size and mtime are the same before and after the read (else raise)."""
    st0 = p.stat()
    data = p.read_bytes()
    st1 = p.stat()
    if (st0.st_size, st0.st_mtime_ns) != (st1.st_size, st1.st_mtime_ns) or len(data) != st1.st_size:
        raise OSError("%s changed while being read" % p)
    return data


def _missing_symbols(so: Path) -> list:
    """Symbols of _SIG that a library does not export (loads it under a private handle)."""
    lib = C.CDLL(str(so), mode=getattr(os, "RTLD_LOCAL", 0))
    return [s for s in _SIG if not hasattr(lib, s)]


def _lib_path(root: Path | None = None, attempts: int = 6) -> Path:
    env = os.environ.get("TSR_SIM_LIB")
    if env:
        return Path(env)
    for a in (SIM_A, REF_A):
        if not a.exists():
            raise FileNotFoundError("%s not found: build the simulator (cmake --build build/dev "
                                    "--target tsr_sim -j4) or set TSR_SIM_LIB" % a)
    base = Path(root) if root is not None else build_root() / "simlib"
    err = None
    for k in range(attempts):
        try:
            sim_b, ref_b = _read_stable(SIM_A), _read_stable(REF_A)
        except OSError as e:          # archive being rewritten right now
            err = e
            time.sleep(0.5 * (k + 1))
            continue
        key = hashlib.sha256(b"tsr_sim\0" + sim_b + b"\0tsr_ref\0" + ref_b).hexdigest()[:20]
        out = base / key / LIB_NAME
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".build_", dir=out.parent))
        try:
            (tmp / "libtsr_sim.a").write_bytes(sim_b)
            (tmp / "libtsr_ref.a").write_bytes(ref_b)
            so = tmp / LIB_NAME
            res = subprocess.run(["gcc", "-shared", "-o", str(so), "-Wl,--whole-archive",
                                  str(tmp / "libtsr_sim.a"), str(tmp / "libtsr_ref.a"),
                                  "-Wl,--no-whole-archive", "-lpthread", "-lm"],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                err = RuntimeError("linking %s failed: %s" % (LIB_NAME, res.stderr.strip()[-400:]))
                time.sleep(0.5 * (k + 1))
                continue
            miss = _missing_symbols(so)
            if miss:
                err = RuntimeError("%s lacks %s (archives mid-rebuild?)" % (LIB_NAME, miss))
                time.sleep(0.5 * (k + 1))
                continue
            os.replace(so, out)       # atomic; a concurrent builder of the same key wrote the same bytes
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError("could not build %s: %s" % (LIB_NAME, err))


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(str(_lib_path()))
        for name, (res, args) in _SIG.items():
            fn = getattr(L, name)
            fn.restype, fn.argtypes = res, args
        _lib = L
    return _lib


@dataclass
class SimResult:
    final: dict          # status, err_code, err_pc, err_info, pc
    trace: trace.Trace   # captured run as a trace (init/expect regions, expected final state)
    regs: list           # R0..R15 (uint32)
    spm: bytes
    retired: int
    perf: dict = field(default_factory=dict)   # PERF_* index -> delta over this run


class Sim:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        c = TsrConfig(**{k: int(self.cfg[k]) for k, _ in TsrConfig._fields_})
        self._c = c
        self.h = lib().tsr_sim_create(C.byref(c), None)
        if not self.h:
            raise ValueError("tsr_sim_create rejected the config %s" % self.cfg)
        self.limit = int(lib().tsr_sim_dram_limit_beats(self.h))
        self.n_perf = isa.perf_count(self.cfg["num_ch"])

    def close(self):
        if self.h:
            lib().tsr_sim_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:   # interpreter shutdown
            pass

    # ---- host access ----
    def load(self, ch: int, byte_off: int, data: bytes):
        """Host write (clipped at the channel end)."""
        ch_bytes = self.cfg["ch_bytes"]
        if byte_off >= ch_bytes or not data:
            return
        n = min(len(data), ch_bytes - byte_off)
        base = C.addressof(lib().tsr_sim_dram(self.h, ch).contents)
        C.memmove(base + byte_off, bytes(data[:n]), n)

    def dram(self, ch: int, byte_off: int, size: int) -> bytes:
        base = C.addressof(lib().tsr_sim_dram(self.h, ch).contents)
        return C.string_at(base + byte_off, size)

    def spm(self) -> bytes:
        return C.string_at(C.addressof(lib().tsr_sim_spm(self.h).contents), self.cfg["spm_bytes"])

    def csr_read(self, off: int) -> int:
        return int(lib().tsr_sim_csr_read(self.h, off))

    def csr_write(self, off: int, val: int):
        lib().tsr_sim_csr_write(self.h, off, val & 0xFFFFFFFF)

    def regs(self) -> list:
        return [int(lib().tsr_sim_ireg(self.h, i)) & 0xFFFFFFFF for i in range(16)]

    def perf(self, idx: int) -> int:
        return int(lib().tsr_sim_perf(self.h, idx))

    # ---- one run, captured ----
    def run(self, prog: bytes, cmd_addr: int, args=None, init=(), count: int | None = None,
            ctrl: int = isa.CTRL_START) -> SimResult:
        """Load prog at ch0 beat cmd_addr (clipped at the channel end) and the init regions (later
        ones win), write CMD_ADDR/CMD_COUNT/ARGs and CTRL, run with access capture."""
        L = lib()
        count = len(prog) // 32 if count is None else count
        args = [(a & 0xFFFFFFFF) for a in (list(args or []) + [0] * 8)[:8]]
        self.load(0, cmd_addr * 64, prog)
        for ch, off, data in init:
            self.load(ch, off, data)
        perf0 = [self.perf(i) for i in range(self.n_perf)]
        self.csr_write(isa.CSR_CMD_ADDR_LO, cmd_addr & 0xFFFFFFFF)
        self.csr_write(isa.CSR_CMD_ADDR_HI, cmd_addr >> 32)
        self.csr_write(isa.CSR_CMD_COUNT, count)
        for i, a in enumerate(args):
            self.csr_write(isa.CSR_ARG0 + 4 * i, a)
        L.tsr_sim_capture_begin(self.h)
        self.csr_write(isa.CSR_CTRL, ctrl)
        L.tsr_sim_run(self.h)
        cap = TsrCapture()
        L.tsr_sim_capture_end(self.h, C.byref(cap))
        try:
            regs_init = [trace.Region(int(cap.init[i].ch), int(cap.init[i].byte_off),
                                      C.string_at(cap.init[i].data, int(cap.init[i].size)))
                         for i in range(cap.n_init)]
            regs_exp = [trace.Region(int(cap.expect[i].ch), int(cap.expect[i].byte_off),
                                     C.string_at(cap.expect[i].data, int(cap.expect[i].size)))
                        for i in range(cap.n_expect)]
        finally:
            L.tsr_capture_free(C.byref(cap))
        final = dict(status=self.csr_read(isa.CSR_STATUS), err_code=self.csr_read(isa.CSR_ERR_CODE),
                     err_pc=self.csr_read(isa.CSR_ERR_PC), err_info=self.csr_read(isa.CSR_ERR_INFO),
                     pc=self.csr_read(isa.CSR_PC))
        tr = trace.Trace(path=Path("sim_run"), config=dict(self.cfg), cmd_addr_beat=cmd_addr,
                         cmd_count=count, args=args, status=final["status"],
                         err_code=final["err_code"], err_pc=final["err_pc"],
                         err_info=final["err_info"], init=regs_init, expect=regs_exp,
                         cmd=bytes(prog[:32 * count]).ljust(min(32 * count, len(prog) + 32 * 256), b"\0"))
        perf = {i: self.perf(i) - perf0[i] for i in range(self.n_perf)}
        return SimResult(final=final, trace=tr, regs=self.regs(), spm=self.spm(),
                         retired=perf[isa.PERF_RETIRED], perf=perf)

"""The ISA simulator's SPM semantics for the tb/spm and tb/dma benches, through ctypes.

libtsr_sim.so (CMake target tsr_sim_shared; $TSR_SIM_LIB, else build/dev/sw/sim/libtsr_sim.so):
  SimSpm      a tsr_sim instance of the bench's SPM size; its SPM bytes (tsr_sim_spm) are the
              reference memory of the benches: line writes with byte enables, line reads, and the
              RISC-V's SPM window (byte / half / word loads and stores at 0x4000_0000 + byte
              offset) through the simulator's bus model (tsr_bus_create_fw on the same bytes,
              tsr_bus_sba_read / tsr_bus_sba_write: the window semantics of CC §4.1 / §10.5 that the
              firmware runs on, little-endian, accesses beyond SPM_BYTES fault). The bench tracks
              which bytes it has written (`known`): the RTL SPM is undefined at start (the
              simulator's starts as 0xA5), so bytes never written are not compared.
  SimOps      tsr_sim_op on a tsr_sim instance (LD / ST of EX §7: the DMA's architectural
              semantics), with DRAM writes / reads and the SPM view.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_sim.so"

SPM_BASE = 0x4000_0000          # TSR_RV_SPM_BASE (CC §4.1)
SPM_WIN = 0x0040_0000           # TSR_RV_SPM_SIZE
LINE = 64


def mut_sources(srcs):
    """Mutation spot-checks: $TSR_MUT = "name.sv=/path/to/mutant.sv[;...]" replaces RTL sources by
    basename (the benches of tb/spm and tb/dma pass their source lists through this), so mutants
    are built from copies and the tree is never edited."""
    spec = os.environ.get("TSR_MUT", "")
    if not spec:
        return list(srcs)
    rep = dict(x.split("=", 1) for x in spec.split(";") if x)
    return [Path(rep[s.name]).resolve() if s.name in rep else s for s in srcs]


class TsrConfig(C.Structure):       # tsr_config (sw/sim/include/tsr_isa.h)
    _fields_ = [(n, C.c_uint32) for n in ("num_ch", "beat_bits", "lanes", "t_max", "k_max", "spm_bytes",
                                          "vl", "nsreg", "rd_outstanding")] + [
               ("ch_bytes", C.c_uint64), ("addr_beat_w", C.c_uint32), ("caps", C.c_uint32)]


class BusEnv(C.Structure):          # tsr_bus_env (sw/sim/include/tsr_bus.h); NULL callbacks
    _fields_ = [("ctx", C.c_void_p), ("cw", C.c_void_p), ("core_read", C.c_void_p), ("spm", C.c_void_p),
                ("spm_bytes", C.c_uint32), ("spm_access", C.c_void_p), ("irq", C.c_void_p),
                ("yield_", C.c_void_p), ("rv_run", C.c_void_p), ("xlat", C.c_void_p)]


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("TSR_SIM_LIB", str(SIM_LIB))
        if not os.path.exists(path):
            raise FileNotFoundError("%s not found: build it with `cmake --build build/dev --target tsr_sim_shared "
                                    "-j2` or set TSR_SIM_LIB" % path)
        L = C.CDLL(path)
        vp, u32, u64 = C.c_void_p, C.c_uint32, C.c_uint64
        for name, res, args in (("tsr_sim_create", vp, [C.POINTER(TsrConfig), C.c_char_p]),
                                ("tsr_sim_destroy", None, [vp]),
                                ("tsr_sim_dram", vp, [vp, u32]),
                                ("tsr_sim_spm", vp, [vp]),
                                ("tsr_sim_op", u32, [vp, C.POINTER(u32), C.POINTER(u32)]),
                                ("tsr_bus_create_fw", vp, [vp, C.POINTER(BusEnv)]),
                                ("tsr_bus_destroy", None, [vp]),
                                ("tsr_bus_sba_read", C.c_int, [vp, u32, u32, C.POINTER(u32)]),
                                ("tsr_bus_sba_write", C.c_int, [vp, u32, u32, u32])):
            getattr(L, name).restype = res
            getattr(L, name).argtypes = args
        _LIB = L
    return _LIB


def u200_cfg(**over) -> dict:
    cfg = dict(num_ch=4, beat_bits=512, lanes=128, t_max=1, k_max=32768, spm_bytes=4 << 20, vl=16, nsreg=16,
               rd_outstanding=16, ch_bytes=16 << 30, addr_beat_w=28, caps=0)
    cfg.update(over)
    return cfg


class SimSpm:
    """Reference SPM: the simulator's SPM bytes. Line granularity for the units' ports, byte /
    half / word for the RISC-V window (the simulator's bus model)."""

    def __init__(self, spm_bytes: int):
        L = lib()
        self.cfg = u200_cfg(spm_bytes=spm_bytes, num_ch=1, ch_bytes=1 << 20)
        self.sim = L.tsr_sim_create(C.byref(TsrConfig(**self.cfg)), None)
        assert self.sim, self.cfg
        self.base = L.tsr_sim_spm(self.sim)
        self.bytes = spm_bytes
        self.lines = spm_bytes // LINE
        self._env = BusEnv(spm=self.base, spm_bytes=spm_bytes)
        self.bus = L.tsr_bus_create_fw(None, C.byref(self._env))
        assert self.bus
        self.known = {}             # line -> bit mask of the bytes written by the bench (512 bits)

    def close(self):
        L = lib()
        if self.bus:
            L.tsr_bus_destroy(self.bus)
            self.bus = None
        if self.sim:
            L.tsr_sim_destroy(self.sim)
            self.sim = None

    # ---- line level (unit ports) -------------------------------------------------------------
    def read_line(self, line: int):
        """(data, known mask) of one line, data as a 512-bit int (byte j = bits 8j+7..8j)."""
        assert 0 <= line < self.lines
        data = int.from_bytes(C.string_at(self.base + LINE * line, LINE), "little")
        return data, self.known.get(line, 0)

    def write_line(self, line: int, data: int, be: int):
        """Bytes j with be[j] = 1 take data's byte j."""
        assert 0 <= line < self.lines
        if be == 0:
            return
        m = be_mask(be)
        old = int.from_bytes(C.string_at(self.base + LINE * line, LINE), "little")
        new = (old & ~m) | (data & m)
        C.memmove(self.base + LINE * line, new.to_bytes(LINE, "little"), LINE)
        self.known[line] = self.known.get(line, 0) | m

    # ---- the RISC-V's SPM window (simulator bus model, debug-module system-bus path) -------------
    def rv_load(self, off: int, size: int):
        """Load of `size` bytes at SPM byte offset off; None on an access fault."""
        v = C.c_uint32(0)
        r = lib().tsr_bus_sba_read(self.bus, SPM_BASE + off, size, C.byref(v))
        return None if r else v.value

    def rv_store(self, off: int, size: int, val: int) -> bool:
        """Store; False on an access fault (nothing written)."""
        r = lib().tsr_bus_sba_write(self.bus, SPM_BASE + off, size, val & 0xFFFFFFFF)
        if r == 0:
            line, b0 = off // LINE, off % LINE
            m = ((1 << (8 * size)) - 1) << (8 * b0)
            self.known[line] = self.known.get(line, 0) | m
        return r == 0


def be_mask(be: int) -> int:
    """64-bit byte enable -> 512-bit bit mask."""
    m = 0
    b = 0
    while be:
        if be & 1:
            m |= 0xFF << (8 * b)
        be >>= 1
        b += 1
    return m


class SimOps:
    """tsr_sim_op on a simulator instance (op-level API, sw/sim/include/tsr_sim.h)."""

    def __init__(self, **cfg):
        L = lib()
        self.cfg = u200_cfg(**cfg)
        self.sim = L.tsr_sim_create(C.byref(TsrConfig(**self.cfg)), None)
        assert self.sim, self.cfg

    def dram_write(self, ch: int, off: int, data: bytes):
        C.memmove(lib().tsr_sim_dram(self.sim, ch) + off, data, len(data))

    def dram_read(self, ch: int, off: int, n: int) -> bytes:
        return C.string_at(lib().tsr_sim_dram(self.sim, ch) + off, n)

    def spm_read(self, off: int, n: int) -> bytes:
        return C.string_at(lib().tsr_sim_spm(self.sim) + off, n)

    def spm_write(self, off: int, data: bytes):
        C.memmove(lib().tsr_sim_spm(self.sim) + off, data, len(data))

    def op(self, words) -> int:
        info = C.c_uint32(0)
        return lib().tsr_sim_op(self.sim, (C.c_uint32 * 8)(*words), C.byref(info))

    def close(self):
        if self.sim:
            lib().tsr_sim_destroy(self.sim)
            self.sim = None

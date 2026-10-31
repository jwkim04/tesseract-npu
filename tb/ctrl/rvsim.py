"""The simulator side of tb/ctrl: tsr_rv (the ISA simulator's RV32 interpreter, SW §4.2) running a
firmware ELF on a firmware-profile tsr_bus whose control window, core window, SPM and interrupt
levels are a tsr_sim's (tsr_q.c / tsr_sim.c: the simulator's own implementation of CC §5, §7 and
EX §4.5), plus the simulator's debug-module model (tsr_dm.c) for the DMI smoke.

Everything goes through build/dev/sw/sim/libtsr_sim.so (target tsr_sim_shared; $TSR_SIM_LIB
overrides) with ctypes. The structures below mirror sw/sim/include/tsr_rv.h, tsr_bus.h, tsr_isa.h,
tsr_sim.h and sw/sim/tsr_sim_int.h (queue events); keep them in step with those headers.

Lockstep (CC §15.4): run(..., ref=rtl_records) gives the interpreter the RTL's value for every load
of a timing- or environment-dependent register (TIMING_CW below, every core-window offset except the
two the bus answers itself) and leaves everything else to the simulator: staging, tickets,
ISSUE_PIDX, ISSUED_SEQ, waits, faults, LCQ commits and the rest of tsr_q.c's control window.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_sim.so"

u8, u16, u32, u64, vp = C.c_uint8, C.c_uint16, C.c_uint32, C.c_uint64, C.c_void_p

TSR_CAPS_VU, TSR_CAPS_ATT = 0x1, 0x20
TICKET_NONE = 0xF0000000
ENV_OK, ENV_FAULT, ENV_WAIT = 0, 1, 2
STOP_BUDGET, STOP_WFI, STOP_HALTED, STOP_REQUEST, STOP_WAIT, STOP_YIELD = range(6)
QEV_PUSH, QEV_QCTRL, QEV_MARK_PUSH = 5, 6, 7
IFIFO_KIND_OP, IFIFO_KIND_CTRL = 0, 1
CTRL_CODES = {0: 1, 1: 2, 2: 3, 3: 4}     # QCTRL bit -> IFIFO CTRL code (QABORT, QHOLD set / clear, QCLEAR)
CODE_TRACE_MARK = 5
PC_HOOK = 0xFFFFFFFD


def timing_cw(off: int) -> bool:
    """Control-window offsets whose value depends on time or on the core environment (CC §15.4)."""
    if off in (0x030, 0x038, 0x144, 0x300, 0x304):          # CREDIT, STATUS(ERR_IRQ), QSTATUS, MTIME
        return True
    if 0x040 <= off < 0x060:                                  # DONE_SEQ
        return True
    if 0x0C0 <= off < 0x140 or 0x180 <= off < 0x1A0:          # error records, ERR_PIDX
        return True
    if (0x200 <= off <= 0x234 and off != 0x228) or off in (0x264, 0x284, 0x288):   # LQ / LCQ / log reads
        return True
    return False


def core_masked(off: int) -> bool:
    return off == 0x0C4 or 0xC00 <= off < 0xE00


class TsrConfig(C.Structure):
    _fields_ = [("num_ch", u32), ("beat_bits", u32), ("lanes", u32), ("t_max", u32), ("k_max", u32),
                ("spm_bytes", u32), ("vl", u32), ("nsreg", u32), ("rd_outstanding", u32),
                ("ch_bytes", u64), ("addr_beat_w", u32), ("caps", u32)]


class SimOpts(C.Structure):
    _fields_ = [("dram_dir", C.c_char_p), ("threads", u32), ("sched", C.c_char_p), ("rv_ratio", C.c_double),
                ("race", u32), ("log", vp), ("q_depth", C.POINTER(u8)), ("itcm_bytes", u32),
                ("dtcm_bytes", u32), ("log_flags", u32), ("trace_dir", C.c_char_p), ("rv_core", C.c_char_p)]


CW_FN = C.CFUNCTYPE(C.c_int, vp, C.c_int, u32, C.POINTER(u32))
CORE_FN = C.CFUNCTYPE(C.c_int, vp, u32, C.POINTER(u32))
SPMACC_FN = C.CFUNCTYPE(None, vp, u32, u32, C.c_int)
IRQ_FN = C.CFUNCTYPE(u32, vp)
INT_FN = C.CFUNCTYPE(C.c_int, vp)
XLAT_FN = C.CFUNCTYPE(u32, vp, u32, u32)


class BusEnv(C.Structure):
    _fields_ = [("ctx", vp), ("cw", CW_FN), ("core_read", CORE_FN), ("spm", C.POINTER(u8)), ("spm_bytes", u32),
                ("spm_access", SPMACC_FN), ("irq", IRQ_FN), ("yield_", INT_FN), ("rv_run", INT_FN),
                ("xlat", XLAT_FN)]


class BusFwCfg(C.Structure):
    _fields_ = [("itcm", C.POINTER(u8)), ("itcm_bytes", u32), ("dtcm", C.POINTER(u8)), ("dtcm_bytes", u32)]


class Retire(C.Structure):
    _fields_ = [("order", u64), ("pc", u32), ("next_pc", u32), ("insn", u32), ("len", u8), ("trap", u8),
                ("intr", u8), ("dbg", u8), ("priv", u8), ("rd", u8), ("mem_size", u8), ("mem_write", u8),
                ("rd_val", u32), ("mem_addr", u32), ("mem_val", u32), ("cause", u32), ("csr", u16),
                ("csr_written", u8), ("rsvd", u8), ("csr_val", u32)]


class QEvent(C.Structure):
    _fields_ = [("kind", u32), ("unit", u32), ("ticket", u32), ("pidx", u32), ("code", u32), ("ran", u32),
                ("rec", u32), ("dep", u32 * 2), ("time", u64), ("w", C.POINTER(u32)), ("ext", vp),
                ("push", u32), ("pc", u32)]


class Elf(C.Structure):
    _fields_ = [("entry", u32), ("lo", u32), ("hi", u32), ("nseg", u32), ("seg", vp), ("buf", vp),
                ("size", C.c_size_t), ("symoff", u32), ("symnum", u32), ("symentsz", u32), ("stroff", u32),
                ("strsize", u32)]


RETIRE_FN = C.CFUNCTYPE(C.c_int, vp, C.POINTER(Retire))
QLIS_FN = C.CFUNCTYPE(None, vp, vp, C.POINTER(QEvent))


def load_lib(path=None):
    path = Path(path or os.environ.get("TSR_SIM_LIB", DEFAULT_LIB))
    if not path.exists():
        raise FileNotFoundError("%s not found: build it with `cmake --build build/dev --target tsr_sim_shared`" % path)
    lib = C.CDLL(str(path))
    sig = {
        "tsr_sim_caps_implemented": (u32, []),
        "tsr_sim_create_opts": (vp, [C.POINTER(TsrConfig), C.POINTER(SimOpts)]),
        "tsr_sim_destroy": (None, [vp]),
        "tsr_sim_cw_access_pc": (C.c_int, [vp, C.c_int, u32, C.POINTER(u32), u32]),
        "tsr_sim_corewin_read": (u32, [vp, u32]),
        "tsr_sim_spm": (C.POINTER(u8), [vp]),
        "tsr_sim_rv_irq": (u32, [vp]),
        "tsr_sim_advance": (u32, [vp, u64]),
        "tsr_sim_now": (u64, [vp]),
        "tsr_sim_csr_read": (u32, [vp, u32]),
        "tsr_sim_csr_write": (None, [vp, u32, u32]),
        "tsr_sim_add_listener": (C.c_int, [vp, QLIS_FN, vp]),
        "tsr_sim_dtcm": (C.POINTER(u8), [vp, C.POINTER(u32)]),
        "tsr_sim_itcm": (C.POINTER(u8), [vp, C.POINTER(u32)]),
        "tsr_bus_create_fw": (vp, [C.POINTER(BusFwCfg), C.POINTER(BusEnv)]),
        "tsr_bus_destroy": (None, [vp]),
        "tsr_bus_iface": (vp, [vp]),
        "tsr_bus_attach": (None, [vp, vp]),
        "tsr_bus_mem": (C.POINTER(u8), [vp, u32, u32]),
        "tsr_bus_set_rv_run": (None, [vp, C.c_int]),
        "tsr_bus_sba_read": (C.c_int, [vp, u32, u32, C.POINTER(u32)]),
        "tsr_bus_sba_write": (C.c_int, [vp, u32, u32, u32]),
        "tsr_rv_create": (vp, [vp]),
        "tsr_rv_destroy": (None, [vp]),
        "tsr_rv_reset": (None, [vp, u32]),
        "tsr_rv_run": (u32, [vp, u64, u64]),
        "tsr_rv_idle": (None, [vp, u64]),
        "tsr_rv_pc": (u32, [vp]),
        "tsr_rv_reg": (u32, [vp, C.c_uint]),
        "tsr_rv_sleeping": (C.c_int, [vp]),
        "tsr_rv_waiting": (C.c_int, [vp]),
        "tsr_rv_set_retire_hook": (None, [vp, RETIRE_FN, vp]),
        "tsr_rv_elf_read": (C.c_int, [C.c_char_p, C.POINTER(Elf)]),
        "tsr_rv_elf_load": (C.c_int, [vp, C.POINTER(Elf)]),
        "tsr_rv_elf_free": (None, [C.POINTER(Elf)]),
        "tsr_rv_elf_symbol": (C.c_int, [C.POINTER(Elf), C.c_char_p, C.POINTER(u32)]),
        "tsr_dm_attach": (None, [vp]),
        "tsr_sim_dm": (vp, [vp]),
        "tsr_dm_dmi_read": (u32, [vp, u32]),
        "tsr_dm_dmi_write": (None, [vp, u32, u32]),
        "tsr_sim_rv_reg": (u32, [vp, C.c_uint]),
        "tsr_sim_rv_pc": (u32, [vp]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    return lib


def u200_config(caps: int) -> TsrConfig:
    return TsrConfig(num_ch=4, beat_bits=512, lanes=128, t_max=1, k_max=32768, spm_bytes=4 * 1024 * 1024,
                     vl=16, nsreg=16, rd_outstanding=16, ch_bytes=1 << 30, addr_beat_w=28, caps=caps)


def elf_symbol(lib, elf_path, name):
    e = Elf()
    if lib.tsr_rv_elf_read(str(elf_path).encode(), C.byref(e)) != 0:
        raise RuntimeError("cannot read %s" % elf_path)
    v = u32()
    rc = lib.tsr_rv_elf_symbol(C.byref(e), name.encode(), C.byref(v))
    lib.tsr_rv_elf_free(C.byref(e))
    if rc != 0:
        raise KeyError(name)
    return v.value


class SimRun:
    """One firmware run of tsr_rv on a firmware bus bound to a fresh tsr_sim."""

    def __init__(self, elf_path, caps=TSR_CAPS_VU | TSR_CAPS_ATT, lib=None):
        self.lib = lib or load_lib()
        L = self.lib
        if (L.tsr_sim_caps_implemented() & caps) != caps:
            raise RuntimeError("libtsr_sim lacks CAPS %#x" % caps)
        self.cfg = u200_config(caps)
        self.opts = SimOpts()
        self.opts.sched = b"fixed"
        self.sim = L.tsr_sim_create_opts(C.byref(self.cfg), C.byref(self.opts))
        if not self.sim:
            raise RuntimeError("tsr_sim_create_opts failed")
        self.records = []
        self.pushes = []          # issue-FIFO entries in push order (dict per entry)
        self.ref = None
        self.injected = 0
        self.cw_log = []
        # bus environment: the simulator's control window, core window, SPM and interrupt levels
        self._cw = CW_FN(self._cw_cb)
        self._core = CORE_FN(self._core_cb)
        self._irq = IRQ_FN(lambda ctx: L.tsr_sim_rv_irq(self.sim))
        self.env = BusEnv()
        self.env.ctx = None
        self.env.cw = self._cw
        self.env.core_read = self._core
        self.env.spm = L.tsr_sim_spm(self.sim)
        self.env.spm_bytes = self.cfg.spm_bytes
        self.env.irq = self._irq
        self.bus = L.tsr_bus_create_fw(None, C.byref(self.env))
        L.tsr_bus_set_rv_run(self.bus, 1)
        self.rv = L.tsr_rv_create(L.tsr_bus_iface(self.bus))
        L.tsr_bus_attach(self.bus, self.rv)
        self._ret = RETIRE_FN(self._retire_cb)
        L.tsr_rv_set_retire_hook(self.rv, self._ret, None)
        self._lis = QLIS_FN(self._qev_cb)
        L.tsr_sim_add_listener(self.sim, self._lis, None)
        e = Elf()
        if L.tsr_rv_elf_read(str(elf_path).encode(), C.byref(e)) != 0:
            raise RuntimeError("cannot read %s" % elf_path)
        if L.tsr_rv_elf_load(self.bus, C.byref(e)) != 0:
            raise RuntimeError("cannot load %s" % elf_path)
        end = u32()
        L.tsr_rv_elf_symbol(C.byref(e), b"__fw_end", C.byref(end))
        self.fw_end = end.value
        L.tsr_rv_elf_free(C.byref(e))
        L.tsr_rv_reset(self.rv, 0)

    def close(self):
        L = self.lib
        if self.rv:
            L.tsr_rv_destroy(self.rv)
            self.rv = None
        if self.bus:
            L.tsr_bus_destroy(self.bus)
            self.bus = None
        if self.sim:
            L.tsr_sim_destroy(self.sim)
            self.sim = None

    # ---------------------------------------------------------------- callbacks
    def _ref_rd(self):
        """The RTL's value for the load being executed (record index = records so far)."""
        k = len(self.records)
        if self.ref is None or k >= len(self.ref):
            return None
        self.injected += 1
        return self.ref[k]["rd_val"]

    def _cw_cb(self, ctx, is_write, off, val):
        L = self.lib
        pc = L.tsr_rv_pc(self.rv)
        rc = L.tsr_sim_cw_access_pc(self.sim, is_write, off, val, pc)
        if not is_write and rc == ENV_OK and timing_cw(off):
            v = self._ref_rd()
            if v is not None:
                val[0] = v
        self.cw_log.append((is_write, off, val[0], rc))
        return rc

    def _core_cb(self, ctx, off, val):
        v = self._ref_rd() if self.ref is not None else None
        val[0] = self.lib.tsr_sim_corewin_read(self.sim, off) if v is None else v
        return ENV_OK

    def _retire_cb(self, ctx, r):
        x = r.contents
        self.records.append({"order": x.order, "pc": x.pc, "next_pc": x.next_pc, "insn": x.insn, "len": x.len,
                             "trap": x.trap, "intr": x.intr, "dbg": x.dbg, "rd": x.rd, "rd_val": x.rd_val,
                             "mem_size": x.mem_size, "mem_write": x.mem_write, "mem_addr": x.mem_addr,
                             "mem_val": x.mem_val, "cause": x.cause})
        return 0

    def _qev_cb(self, ctx, sim, ev):
        e = ev.contents
        if e.kind == QEV_PUSH:
            w = [e.w[i] for i in range(8)]
            self.pushes.append({"kind": IFIFO_KIND_OP, "w": w, "dep0": e.dep[0], "dep1": e.dep[1], "rec": e.rec,
                                "ticket": e.ticket, "pc": e.pc})
        elif e.kind == QEV_QCTRL:
            bits = e.w[0] & 0xF
            ticket = e.ticket
            for b in range(4):
                if bits >> b & 1:
                    code = CTRL_CODES[b]
                    self.pushes.append({"kind": IFIFO_KIND_CTRL, "w": [code] + [0] * 7, "dep0": TICKET_NONE,
                                        "dep1": TICKET_NONE, "rec": 0,
                                        "ticket": ticket if code == 4 else TICKET_NONE, "pc": e.pc})
        elif e.kind == QEV_MARK_PUSH:
            self.pushes.append({"kind": IFIFO_KIND_CTRL, "w": [CODE_TRACE_MARK, e.w[0]] + [0] * 6,
                                "dep0": TICKET_NONE, "dep1": TICKET_NONE, "rec": 0, "ticket": TICKET_NONE,
                                "pc": e.pc})

    # ---------------------------------------------------------------- execution
    def run(self, ref=None, max_insns=200000):
        """Runs to __fw_end (one instruction at a time; the simulator's time advances while a
        store waits or the hart sleeps). With ref (the RTL's RVFI records) every executed
        instruction is checked against it as it retires; returns the list of mismatches."""
        L = self.lib
        self.ref = ref
        stalls = 0
        while len(self.records) < max_insns:
            if L.tsr_rv_pc(self.rv) == self.fw_end:
                break
            n0 = len(self.records)
            r = L.tsr_rv_run(self.rv, 1, 1 << 62)
            if r in (STOP_WAIT, STOP_WFI, STOP_YIELD):
                L.tsr_sim_advance(self.sim, 64)
                L.tsr_rv_idle(self.rv, 32)
                stalls += 1
                if stalls > 200000:
                    raise RuntimeError("simulator run stuck at pc %#x (stop %d)" % (L.tsr_rv_pc(self.rv), r))
                continue
            if len(self.records) > n0:
                L.tsr_sim_advance(self.sim, 2)     # time passes with the program (schedule only)
        # let queued work finish so the simulator's rings and counters settle
        L.tsr_sim_advance(self.sim, 100000)
        return self.records

    # ---------------------------------------------------------------- results
    def dtcm(self, n=64 * 1024):
        p = self.lib.tsr_bus_mem(self.bus, 0x00100000, n)
        return bytes(C.string_at(p, n))

    def spm(self, off, n):
        p = self.lib.tsr_sim_spm(self.sim)
        return bytes(C.string_at(C.addressof(p.contents) + off, n))

    def csr(self, off):
        return self.lib.tsr_sim_csr_read(self.sim, off)

    def lcq_records(self):
        n = self.csr(0x074)
        return [[self.csr(0x300 + 32 * (k % 8) + 4 * j) for j in range(8)] for k in range(n)]

    def log_words(self):
        n = self.csr(0x090)
        return [self.csr(0x400 + 4 * (k % 256)) for k in range(n)]


class DmSim:
    """The simulator's debug-module model (tsr_dm.c) on the simulator's built-in hart, for the DMI
    smoke: the same DMI sequence as on the RTL, compared read by read."""

    def __init__(self, elf_path, caps=TSR_CAPS_VU | TSR_CAPS_ATT, lib=None, patches=None):
        self.lib = lib or load_lib()
        L = self.lib
        self.cfg = u200_config(caps)
        self.opts = SimOpts()
        self.opts.sched = b"fixed"
        self.sim = L.tsr_sim_create_opts(C.byref(self.cfg), C.byref(self.opts))
        L.tsr_dm_attach(self.sim)
        self.dm = L.tsr_sim_dm(self.sim)
        if not self.dm:
            raise RuntimeError("libtsr_sim has no DM model")
        # load the image into the simulator's TCMs (as the host loader would) and boot
        e = Elf()
        L.tsr_rv_elf_read(str(elf_path).encode(), C.byref(e))
        n = u32()
        itcm = L.tsr_sim_itcm(self.sim, C.byref(n))
        itcm_n = n.value
        dtcm = L.tsr_sim_dtcm(self.sim, C.byref(n))
        dtcm_n = n.value
        # tsr_rv_elf_seg = {paddr, filesz, memsz, data*}: on 64-bit hosts data* is 8 bytes after 3 u32
        class Seg(C.Structure):
            _fields_ = [("paddr", u32), ("filesz", u32), ("memsz", u32), ("data", C.POINTER(u8))]
        seg = (Seg * e.nseg).from_address(e.seg)
        for s in seg:
            data = C.string_at(s.data, s.filesz) if s.filesz else b""
            if s.paddr < itcm_n:
                C.memmove(C.addressof(itcm.contents) + s.paddr, data, len(data))
            else:
                C.memmove(C.addressof(dtcm.contents) + s.paddr - 0x00100000, data, len(data))
        L.tsr_rv_elf_free(C.byref(e))
        for a, v in (patches or {}).items():
            C.memmove(C.addressof(dtcm.contents) + a - 0x00100000, v.to_bytes(4, "little"), 4)
        L.tsr_sim_csr_write(self.sim, 0x028, 0)          # RV_BOOT
        L.tsr_sim_csr_write(self.sim, 0x020, 1)          # CTRL.RV_RUN

    def advance(self, n):
        self.lib.tsr_sim_advance(self.sim, n)

    def dmi_read(self, a):
        return self.lib.tsr_dm_dmi_read(self.dm, a)

    def dmi_write(self, a, v):
        self.lib.tsr_dm_dmi_write(self.dm, a, v)

    def close(self):
        if self.sim:
            self.lib.tsr_sim_destroy(self.sim)
            self.sim = None

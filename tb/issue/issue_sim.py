"""ctypes bridge to the simulator (sw/sim, libtsr_sim.so of the CMake target tsr_sim_shared) for the
issue-path benches: tb/chk (checker classes), tb/issue (queue model) and tb/csr (host CSR port,
control window, rings, loader, SOFT_RESET).

The simulator is the oracle (docs/design/06-software-verification.md "SW" §4; CC §15): the same
push stream goes through tsr_sim's issue port (tsr_sim_push / tsr_sim_qctrl / TRACE_MARK) and
through the RTL; the queue listener records every enqueue, start and completion with its code.
Library: $TSR_SIM_LIB, else build/dev/sw/sim/libtsr_sim.so.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIM_LIB = REPO / "build" / "dev" / "sw" / "sim" / "libtsr_sim.so"

u8, u32, u64, vp = ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p


class TsrConfig(ctypes.Structure):          # tsr_config (sw/sim/include/tsr_isa.h)
    _fields_ = [(n, u32) for n in ("num_ch", "beat_bits", "lanes", "t_max", "k_max", "spm_bytes",
                                   "vl", "nsreg", "rd_outstanding")] + [
               ("ch_bytes", u64), ("addr_beat_w", u32), ("caps", u32)]


class TsrSimOpts(ctypes.Structure):         # tsr_sim_opts (sw/sim/include/tsr_sim.h)
    _fields_ = [("dram_dir", ctypes.c_char_p), ("threads", u32), ("sched", ctypes.c_char_p),
                ("rv_ratio", ctypes.c_double), ("race", u32), ("log", vp), ("q_depth", ctypes.POINTER(u8)),
                ("itcm_bytes", u32), ("dtcm_bytes", u32), ("log_flags", u32), ("trace_dir", ctypes.c_char_p),
                ("rv_core", ctypes.c_char_p), ("vu_timing", ctypes.c_char_p)]


class TsrExtent(ctypes.Structure):          # tsr_extent (sw/sim/tsr_sim_int.h §1)
    _fields_ = [("kind", u8), ("write", u8), ("ch", u8), ("rsvd", u8), ("lo", u64), ("hi", u64)]


class TsrExtents(ctypes.Structure):
    _fields_ = [("n", u32), ("overflow", u32), ("e", TsrExtent * 7)]


CHECK_FN = ctypes.CFUNCTYPE(u32, vp, ctypes.POINTER(u32), ctypes.POINTER(TsrExtents), ctypes.POINTER(u32))


class TsrOpHandler(ctypes.Structure):       # tsr_op_handler (tsr_sim_int.h §2)
    _fields_ = [("unit", u8), ("check", CHECK_FN), ("exec", vp), ("cycles", vp)]


class TsrQEvent(ctypes.Structure):          # tsr_q_event (tsr_sim_int.h §5)
    _fields_ = [("kind", u32), ("unit", u32), ("ticket", u32), ("pidx", u32), ("code", u32), ("ran", u32),
                ("rec", u32), ("dep", u32 * 2), ("time", u64), ("w", ctypes.POINTER(u32)),
                ("ext", ctypes.POINTER(TsrExtents)), ("push", u32), ("pc", u32)]


LISTENER_FN = ctypes.CFUNCTYPE(None, vp, vp, ctypes.POINTER(TsrQEvent))
HOOK_U32 = ctypes.CFUNCTYPE(u32, vp)


class TsrDbgHooks(ctypes.Structure):        # tsr_sim_dbg_hooks (tsr_sim_int.h §5)
    _fields_ = [("ctx", vp), ("csr_read", vp), ("csr_write", vp), ("cw_read", vp), ("cw_write", vp),
                ("unit_hold", HOOK_U32), ("soft_reset", vp), ("unit_step", HOOK_U32), ("stop_active", vp),
                ("rv_irq", vp), ("rv_state", vp)]


QEV_ENQ, QEV_START, QEV_DONE, QEV_MARK, QEV_PUSH, QEV_QCTRL, QEV_MARK_PUSH = 1, 2, 3, 4, 5, 6, 7
EXT_SPM, EXT_DRAM, EXT_SREG = 1, 2, 3
CW_OK, CW_FAULT, CW_WAIT = 0, 1, 2
TICKET_NONE = 0xF0000000
U200_QD = (8, 4, 4, 16, 0, 4)

_LIB = None


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = os.environ.get("TSR_SIM_LIB", str(SIM_LIB))
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found: build it with `cmake --build build/dev --target tsr_sim_shared`"
                                " or set TSR_SIM_LIB" % path)
    L = ctypes.CDLL(path)
    for name, res, args in (
            ("tsr_sim_create_opts", vp, [ctypes.POINTER(TsrConfig), ctypes.POINTER(TsrSimOpts)]),
            ("tsr_sim_destroy", None, [vp]),
            ("tsr_sim_csr_read", u32, [vp, u32]),
            ("tsr_sim_csr_write", None, [vp, u32, u32]),
            ("tsr_sim_corewin_read", u32, [vp, u32]),
            ("tsr_sim_cw_read", u32, [vp, u32]),
            ("tsr_sim_cw_write", None, [vp, u32, u32]),
            ("tsr_sim_cw_access", ctypes.c_int, [vp, ctypes.c_int, u32, ctypes.POINTER(u32)]),
            ("tsr_sim_push", u32, [vp, ctypes.POINTER(u32), u32, u32, u32]),
            ("tsr_sim_qctrl", u32, [vp, u32]),
            ("tsr_sim_advance", u32, [vp, u64]),
            ("tsr_sim_run_until_idle", ctypes.c_int, [vp, u64]),
            ("tsr_sim_now", u64, [vp]),
            ("tsr_sim_perf", u64, [vp, u32]),
            ("tsr_sim_add_listener", ctypes.c_int, [vp, LISTENER_FN, vp]),
            ("tsr_sim_set_dbg_hooks", None, [vp, ctypes.POINTER(TsrDbgHooks)]),
            ("tsr_sim_rv_set_state", None, [vp, u32]),
            ("tsr_sim_rv_ndmreset", None, [vp, ctypes.c_int]),
            ("tsr_sim_itcm", vp, [vp, ctypes.POINTER(u32)]),
            ("tsr_sim_dtcm", vp, [vp, ctypes.POINTER(u32)]),
            ("tsr_sim_accept_op", ctypes.POINTER(TsrOpHandler), [vp, u32]),
            ("tsr_sim_set_plat_err", None, [vp, ctypes.c_int]),
            ("tsr_sim_sreg", vp, [vp]),
            ("tsr_sim_cw_push_timeout", None, [vp]),
            ("tsr_sim_caps_implemented", u32, [])):
        f = getattr(L, name)
        f.restype = res
        f.argtypes = args
    _LIB = L
    return L


def config(num_ch=4, t_max=8, k_max=32768, spm_bytes=4 << 20, ch_bytes=16 << 30, addr_beat_w=28, caps=0,
           lanes=128, vl=16, nsreg=16, rd_outstanding=16) -> dict:
    return dict(num_ch=num_ch, beat_bits=512, lanes=lanes, t_max=t_max, k_max=k_max, spm_bytes=spm_bytes,
                vl=vl, nsreg=nsreg, rd_outstanding=rd_outstanding, ch_bytes=ch_bytes, addr_beat_w=addr_beat_w,
                caps=caps)


class Sim:
    """One simulator instance (tsr_sim.h): host CSR port, the RISC-V control window through the test
    hooks, the queue listener, DB's per-unit hold / step through the debug hooks (CC §9.2)."""

    def __init__(self, cfg: dict, q_depth=U200_QD, sched: str = "fixed", rv_ratio: float = 0.0, vu_timing: str = None):
        L = self.L = lib()
        self.cfg = dict(cfg)
        self.qd = tuple(q_depth)
        self._qd_arr = (u8 * 6)(*self.qd)
        self._sched = sched.encode()
        opts = TsrSimOpts(dram_dir=None, threads=1, sched=self._sched, rv_ratio=rv_ratio, race=0, log=None,
                          q_depth=ctypes.cast(self._qd_arr, ctypes.POINTER(u8)), itcm_bytes=0, dtcm_bytes=0,
                          log_flags=0, trace_dir=None, rv_core=None,
                          vu_timing=vu_timing.encode() if vu_timing else None)
        self.sim = L.tsr_sim_create_opts(ctypes.byref(TsrConfig(**self.cfg)), ctypes.byref(opts))
        if not self.sim:
            raise RuntimeError("tsr_sim_create_opts refused %r q_depth %r" % (cfg, q_depth))
        self.events = []
        self._cbs = []
        self.hold_mask = 0
        self.step_mask = 0

    def close(self):
        if self.sim:
            self.L.tsr_sim_destroy(self.sim)
            self.sim = None

    # ---- host CSR port and RISC-V control window ----
    def csr_read(self, off):
        return self.L.tsr_sim_csr_read(self.sim, off)

    def csr_write(self, off, val):
        self.L.tsr_sim_csr_write(self.sim, off, val & 0xFFFFFFFF)

    def corewin_read(self, off):
        return self.L.tsr_sim_corewin_read(self.sim, off)

    def cw_read(self, off):
        return self.L.tsr_sim_cw_read(self.sim, off)

    def cw_write(self, off, val):
        self.L.tsr_sim_cw_write(self.sim, off, val & 0xFFFFFFFF)

    def cw_access(self, is_write, off, val=0):
        """tsr_sim_cw_access: (rc, value) with rc CW_OK / CW_FAULT / CW_WAIT."""
        v = u32(val & 0xFFFFFFFF)
        rc = self.L.tsr_sim_cw_access(self.sim, 1 if is_write else 0, off, ctypes.byref(v))
        return rc, v.value

    # ---- issue port (test hooks, CC §7.1) ----
    def push(self, words, dep0=TICKET_NONE, dep1=TICKET_NONE, push_val=0xFF00):
        return self.L.tsr_sim_push(self.sim, (u32 * 8)(*[w & 0xFFFFFFFF for w in words]), dep0, dep1, push_val)

    def qctrl(self, bits):
        return self.L.tsr_sim_qctrl(self.sim, bits)

    def advance(self, n):
        return self.L.tsr_sim_advance(self.sim, n)

    def run_until_idle(self, n=1 << 40):
        return bool(self.L.tsr_sim_run_until_idle(self.sim, n))

    def now(self):
        return self.L.tsr_sim_now(self.sim)

    def perf(self, idx):
        return self.L.tsr_sim_perf(self.sim, idx)

    def rv_set_state(self, bits):
        self.L.tsr_sim_rv_set_state(self.sim, bits)

    def ndmreset(self, level):
        self.L.tsr_sim_rv_ndmreset(self.sim, 1 if level else 0)

    def tcm(self, which: str):
        n = u32(0)
        p = (self.L.tsr_sim_itcm if which == "itcm" else self.L.tsr_sim_dtcm)(self.sim, ctypes.byref(n))
        return p, n.value

    # ---- queue listener ----
    def listen(self):
        """Records every queue event as a dict in self.events (kind, unit, ticket, pidx, code, ran,
        dep, w, ext, time, pc, push, rec)."""
        def cb(_ctx, _sim, evp):
            e = evp.contents
            ext = []
            if e.ext:
                x = e.ext.contents
                ext = [(x.e[i].kind, x.e[i].write, x.e[i].ch, x.e[i].lo, x.e[i].hi) for i in range(x.n)]
            w = [e.w[i] for i in range(8)] if e.w else [0] * 8
            self.events.append(dict(kind=e.kind, unit=e.unit, ticket=e.ticket, pidx=e.pidx, code=e.code,
                                    ran=e.ran, rec=e.rec, dep=(e.dep[0], e.dep[1]), time=e.time, w=w,
                                    ext=ext, push=e.push, pc=e.pc))
        f = LISTENER_FN(cb)
        self._cbs.append(f)
        rc = self.L.tsr_sim_add_listener(self.sim, f, None)
        assert rc >= 0

    def dbg_hooks(self):
        """DB's dbg_hold[u] / dbg_step[u] from self.hold_mask / self.step_mask (replaces the DB model's
        hooks: its DBG block then reads 0)."""
        hold = HOOK_U32(lambda _c: self.hold_mask)
        step = HOOK_U32(lambda _c: self.step_mask)
        self._cbs += [hold, step]
        self._hooks = TsrDbgHooks(ctx=None, unit_hold=hold, unit_step=step)
        self.L.tsr_sim_set_dbg_hooks(self.sim, ctypes.byref(self._hooks))

    # ---- checker classes (tsr_sim_accept_op + the handler's check) ----
    def check(self, words):
        """(unit, code, info, extents) of the issue path's checker for the 8 words: a refused opcode is
        (SYNC, ILLEGAL_OP, 0, []); BAD_DEP is the dispatcher's (not here)."""
        h = self.L.tsr_sim_accept_op(self.sim, words[0] & 0xFF)
        if not h:
            return 5, 1, 0, []
        hh = h.contents
        ext = TsrExtents()
        info = u32(0)
        code = hh.check(self.sim, (u32 * 8)(*[w & 0xFFFFFFFF for w in words]), ctypes.byref(ext), ctypes.byref(info))
        exts = [(ext.e[i].kind, ext.e[i].write, ext.e[i].ch, ext.e[i].lo, ext.e[i].hi) for i in range(ext.n)] \
            if code == 0 else []
        return hh.unit, code, (info.value if code else 0), exts

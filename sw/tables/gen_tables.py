#!/usr/bin/env python3
"""sw/tables/gen_tables.py — generate the v2 tables from sw/tables/tsr_tables.py (V2-EX-30, P1.1).

    sw/tables/gen_tables.py           write the generated files below (only those that changed)
    sw/tables/gen_tables.py --check   regenerate in memory; exit 1 if a file is missing or stale

Outputs (each starts with a do-not-edit banner naming the source):
    rtl/core/tsr_tables_pkg.sv        SystemVerilog package tsr_tables_pkg
    sw/sim/include/tsr_tables.h       C header (C11 and C++): macros and static inline lookups
    sw/ctrl/include/tsr/mmio.hpp      freestanding C++20: maps, opcodes, units, errors, PERF, launch
                                      and debug formats, the v1-op encodings, tsr::op
    sw/ctrl/include/tsr/vu.hpp        freestanding C++20: VU encodings and constexpr op builders
    sw/ctrl/include/tsr/at.hpp        freestanding C++20: AT encodings and constexpr op builders
    tb/common/tables.py               Python constants, tables and encode / decode for the benches

Naming (C and SV carry the TSR_ prefix; C++ (namespace tsr) and Python drop it):
    CSR_<REG>                 host CSR offset (EX §5, DB §10.1); arrays: _BASE, _N, _STRIDE and a
                              function CSR_<REG>(i) (two arguments (i, j) for multi-word elements)
    CW_<REG>                  RISC-V control-window offset (CC §5, EX §4.5, DB §10.2)
    SAXI_<REG>                s_axi_control offset (EX §9.2)
    RV_<REGION>_BASE / _SIZE / _LIMIT                RISC-V address map (CC §4.1)
    <LAYOUT>_<F>              in-place mask of bit field F; _LSB, _W; values <LAYOUT>_<F>_<VALUE>
    <OP>_<F>                  instruction field F: in-place mask in its word; _LSB, _W, _WORD and
                              _BIT (position in the 256-bit instruction); <OP>_RSV_W<i> = bits that
                              must be 0; <OP>_WORDS = words that hold a field (bit 0 always)
    OP_<MNEMONIC>, UNIT_<U>, ERR_<E>, PERF_<P>, IRQ_FAST_<I>, DMI_<R>, REGION_<R>_OFF / _SIZE,
    <TABLE>_<ROW>_WORD (word tables), <ENUM>_<VALUE> (shared value sets)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = "sw/tables/tsr_tables.py"
GENERATOR = "sw/tables/gen_tables.py"

OUT_SV = REPO / "rtl/core/tsr_tables_pkg.sv"
OUT_C = REPO / "sw/sim/include/tsr_tables.h"
OUT_MMIO = REPO / "sw/ctrl/include/tsr/mmio.hpp"
OUT_VU = REPO / "sw/ctrl/include/tsr/vu.hpp"
OUT_AT = REPO / "sw/ctrl/include/tsr/at.hpp"
OUT_PY = REPO / "tb/common/tables.py"

BANNER = "GENERATED FILE, DO NOT EDIT: produced by %s from %s." % (GENERATOR, SOURCE)

# Constants printed in hex whatever their size (bit patterns, addresses, magic words).
HEX_SUFFIXES = ("_MAGIC", "_NAN", "_OFF", "_VMA", "_VALUE", "_NONE", "_HALF", "_FLOOR", "_POLY")


# ================================================================================================
# Loading and normalising the table source
# ================================================================================================
def load_source(path: Path = HERE / "tsr_tables.py"):
    """Import tsr_tables.py by path (no sys.path change)."""
    spec = importlib.util.spec_from_file_location("tsr_tables_source", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_bits(s: str) -> tuple[int, int]:
    """'15:13' -> (lsb 13, width 3); '7' -> (7, 1)."""
    if ":" in s:
        hi, lo = (int(x) for x in s.split(":"))
        if hi < lo:
            raise ValueError("bit range %r: msb < lsb" % s)
        return lo, hi - lo + 1
    b = int(s)
    return b, 1


def parse_enc(s: str) -> tuple[int, int, int]:
    """'w1[23:0]' -> (word 1, lsb 0, width 24); 'w5' -> (5, 0, 32); 'x16[7:0]' -> (0, 16, 8)."""
    if s.startswith("x16[") and s.endswith("]"):
        lsb, width = parse_bits(s[4:-1])
        if lsb + width > 16:
            raise ValueError("x16 field %r beyond x16[15:0]" % s)
        return 0, 16 + lsb, width
    if s.startswith("w"):
        if "[" in s:
            word = int(s[1:s.index("[")])
            lsb, width = parse_bits(s[s.index("[") + 1:-1])
        else:
            word, lsb, width = int(s[1:]), 0, 32
        return word, lsb, width
    raise ValueError("bad encoding field %r" % s)


class Field:
    def __init__(self, name, lsb, width, desc="", word=None, src=None, spec=""):
        self.name, self.lsb, self.width, self.desc = name, lsb, width, desc
        self.word, self.src, self.spec = word, src, spec

    @property
    def vmask(self) -> int:        # value mask (not shifted)
        return (1 << self.width) - 1

    @property
    def mask(self) -> int:         # in-place mask
        return self.vmask << self.lsb

    @property
    def bit(self) -> int:          # position in the 256-bit instruction
        return 32 * self.word + self.lsb


class Layout:
    def __init__(self, name, d, enums, wide=False):
        self.name, self.wide = name, wide
        self.src, self.desc = d["src"], d.get("desc", "")
        self.width = d.get("width", 32)
        self.fields = []
        for row in d["fields"]:
            bits, fname, fdesc = row[0], row[1], row[2]
            lsb, width = parse_bits(bits)
            self.fields.append(Field(fname, lsb, width, fdesc, src=row[3] if len(row) > 3 else None, spec=bits))
        self.values = {f: resolve_values(v, enums) for f, v in d.get("values", {}).items()}
        self.retired = d.get("retired", [])

    def field(self, name):
        return next(f for f in self.fields if f.name == name)


def resolve_values(v, enums):
    if isinstance(v, str):
        return enums[v][1]
    return v


class Reg:
    def __init__(self, row, space):
        self.off, self.name, self.acc, self.owner, self.src, self.desc = row[:6]
        ex = row[6] if len(row) > 6 else {}
        self.space = space
        self.n = ex.get("n", 1)
        self.stride = ex.get("stride", 4)
        self.words = ex.get("words", 1)
        self.layout = ex.get("layout")
        self.reset = ex.get("reset")
        self.block = ex.get("block")

    @property
    def is_array(self) -> bool:
        return self.n > 1 or self.words > 1

    def word_offsets(self) -> list[int]:
        if self.block is not None:
            return list(range(self.off, self.off + self.block, 4))
        return [self.off + i * self.stride + 4 * j for i in range(self.n) for j in range(self.words)]


class Enc:
    def __init__(self, name, d, opcode, enums, funcs=None, func_field=None):
        self.name, self.opcode = name, opcode
        self.src, self.owner = d["src"], d["owner"]
        self.fields = []
        for spec, fname, fdesc in d["fields"]:
            word, lsb, width = parse_enc(spec)
            self.fields.append(Field(fname, lsb, width, fdesc, word=word, spec=spec))
        self.values = {}       # inline value lists (emitted per op)
        self.value_refs = {}   # field -> shared ENUMS name (emitted once, under ENUMS)
        for f, v in d.get("values", {}).items():
            if isinstance(v, str):
                self.value_refs[f] = v
            else:
                self.values[f] = v
        if funcs is not None:
            self.values[func_field] = [(code, fname, desc) for (code, fname, desc) in funcs]
        self.all_values = dict(self.values)
        for f, ename in self.value_refs.items():
            self.all_values[f] = enums[ename][1]
        covered = [0] * 8
        covered[0] = 0xFF
        for f in self.fields:
            covered[f.word] |= f.mask
        self.rsv = [(~c) & 0xFFFFFFFF for c in covered]
        self.words_mask = 1
        for f in self.fields:
            self.words_mask |= 1 << f.word


class Model:
    """Normalised view of tsr_tables.py, shared by the emitters and test_tables.py."""

    def __init__(self, src):
        self.src = src
        self.consts = list(src.CONSTANTS)
        self.enums = dict(src.ENUMS)
        self.layouts = {n: Layout(n, d, self.enums) for n, d in src.LAYOUTS.items()}
        self.wide = {n: Layout(n, d, self.enums, wide=True) for n, d in src.WIDE_LAYOUTS.items()}
        self.host = [Reg(r, "CSR") for r in src.HOST_CSR]
        self.dbg = [Reg(r, "CSR") for r in src.DBG_CSR]
        self.cw = [Reg(r, "CW") for r in src.CTL_WINDOW]
        self.saxi = [Reg(r, "SAXI") for r in src.SAXI]
        for r in self.host + self.dbg + self.cw + self.saxi:
            if r.layout is None and r.name in self.layouts:
                r.layout = r.name
        self.units = list(src.UNITS)
        self.unit_id = {u[1]: u[0] for u in src.UNITS}
        for uid, uname, *_ in src.UNITS_SPECIAL:
            self.unit_id[uname] = uid
        # Opcodes, one entry per value.
        self.ops = [None] * 256
        for lo, hi, mnem, status, unit, caps, owner, osrc, desc in src.OPCODES:
            for code in range(lo, hi + 1):
                if self.ops[code] is not None:
                    raise ValueError("opcode 0x%02X listed twice" % code)
                self.ops[code] = dict(code=code, mnem=mnem, status=status, unit=unit, caps=caps,
                                      owner=owner, src=osrc, desc=desc, range=(lo, hi),
                                      legal=status in ("kept", "respec", "new"))
        self.opcode = {o["mnem"]: o["code"] for o in self.ops if o is not None and o["mnem"]}
        vop_src, vop_funcs = src.VOP_FUNCS
        sop_src, sop_funcs = src.SOP_FUNCS
        self.vop_funcs = vop_funcs
        self.sop_funcs = sop_funcs
        self.encs = {}
        for name, d in src.ENCODINGS.items():
            funcs = field = None
            if name == "VOP":
                funcs, field = [(c, n, "reads %s, rate %s: %s" % (rd, rt, res)) for c, n, rd, rt, res in vop_funcs], "FUNC"
            elif name == "SOP":
                funcs, field = [(c, n, "%s (uses %s)" % (eff, ", ".join(u))) for c, n, u, eff in sop_funcs], "SUB"
            self.encs[name] = Enc(name, d, self.opcode[name], self.enums, funcs, field)
        self.errors = list(src.ERRORS)
        self.err_code = {e[1]: e[0] for e in src.ERRORS if e[1]}
        self.check_order = src.ERR_CHECK_ORDER[1]
        self.status_prec = src.STATUS_PRECEDENCE[1]
        self.perf = list(src.PERF)
        self.word_tables = dict(src.WORD_TABLES)

    # ---- derived tables --------------------------------------------------------------------------
    def op_unit(self, code: int) -> int:
        o = self.ops[code]
        return self.unit_id[o["unit"]] if o["legal"] else self.unit_id["SYNC"]

    def op_caps(self, code: int) -> int:
        o = self.ops[code]
        if not o["legal"] or o["caps"] is None:
            return 0
        return self.layouts["CAPS"].field(o["caps"]).mask

    def perf_slot(self, slot, num_ch: int) -> int:
        if isinstance(slot, int):
            return slot
        kind, k = slot
        if kind == "ch":
            raise ValueError("per-channel slot needs a channel")
        return 5 + 2 * num_ch + k

    def perf_slots(self, num_ch: int) -> dict:
        """name -> slot for C = num_ch (per-channel names get a [c] suffix)."""
        out = {}
        for slot, name, kind, psrc, desc in self.perf:
            if name is None:
                continue
            if isinstance(slot, tuple) and slot[0] == "ch":
                for c in range(num_ch):
                    out["%s[%d]" % (name, c)] = 5 + slot[1] * num_ch + c
            else:
                out[name] = self.perf_slot(slot, num_ch)
        return out

    def q_depth_value(self) -> int:
        v = 0
        lay = self.layouts["Q_DEPTH"]
        for uid, uname, qd, *_ in self.units:
            v |= qd << lay.field(uname).lsb
        return v


# ================================================================================================
# Formatting helpers
# ================================================================================================
def one_line(s: str) -> str:
    return " ".join(str(s).split())


def comment(src: str, desc: str) -> str:
    d = one_line(desc)
    return "%s: %s" % (src, d) if d else src


def const_is_hex(name: str, value: int) -> bool:
    return value > 0xFFFF or name.endswith(HEX_SUFFIXES)


def sv_hex(v: int, width: int) -> str:
    digits = (width + 3) // 4
    h = "%0*X" % (digits, v)
    if digits > 4:   # group by 4 hex digits from the right, as the RTL style does
        groups = []
        while h:
            groups.insert(0, h[-4:])
            h = h[:-4]
        h = "_".join(groups)
    return "%d'h%s" % (width, h)


def rows_aligned(rows, sep=" "):
    """rows: list of (left, right, comment or None) -> lines with aligned columns."""
    if not rows:
        return []
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    out = []
    for left, right, cmt in rows:
        line = left.ljust(w0) + sep + (right.ljust(w1) if cmt else right)
        if cmt:
            line += "  " + cmt
        out.append(line.rstrip())
    return out


def bits_str(f: Field) -> str:
    return "[%d]" % f.lsb if f.width == 1 else "[%d:%d]" % (f.lsb + f.width - 1, f.lsb)


# ================================================================================================
# Language back ends. Each back end turns a list of abstract items into text; the walk over the
# model (what is emitted, in which order) is shared, so every language carries the same names.
# Items:
#   ("section", title)                    a heading comment
#   ("note", text)                        a comment line
#   ("const", name, value, kind, cmt)     kind: u32 (default), off12, off13, addr, u64, op, unit,
#                                         err, mask32, mask(width), val(width), count
#   ("func_arr", name, base, stride, words, word_stride, cmt)   offset function of an array
#   ("func_src", name, {lang: source})    a function written per language
# ================================================================================================
def walk(m: Model, part: str):
    """Yield the items of one part: 'all' (C, SV, Python), 'mmio', 'vu', 'at' (C++ headers)."""
    items = []
    add = items.append
    want = (lambda p: True) if part == "all" else (lambda p: p == part)

    if want("mmio"):
        add(("section", "Scalar constants (CONSTANTS)"))
        for name, value, csrc, desc in m.consts:
            add(("const", name, value, "hex" if const_is_hex(name, value) else "u32", comment(csrc, desc)))

        add(("section", "Shared value sets (ENUMS)"))
        for ename, (esrc, vals) in m.enums.items():
            for v, vname, vdesc in vals:
                add(("const", "%s_%s" % (ename, vname), v, "u32", comment(esrc, vdesc)))

        add(("section", "Field access helpers"))
        add(("func_src", "field_helpers", FIELD_HELPERS))

        # ---- register maps --------------------------------------------------------------------
        for title, regs, prefix, kind in (
                ("Host CSR map: core CSR port, U200 s_axi_control 0x1000 + offset (EX §5.1)", m.host, "CSR_", "off12"),
                ("DBG block: host CSR 0x800..0xAFC (DB §10.1)", m.dbg, "CSR_", "off12"),
                ("RISC-V control window: offsets from RV_CTL_BASE (CC §5, EX §4.5, DB §10.2)", m.cw, "CW_", "off12"),
                ("XRT adapter s_axi_control map (EX §9.2)", m.saxi, "SAXI_", "off13")):
            add(("section", title))
            for r in regs:
                if r.name is None:
                    add(("note", "0x%03X..0x%03X reserved (%s)" % (r.off, r.off + r.block - 4, r.src)))
                    continue
                cmt = "[%s] %s: %s" % (r.acc, r.src, one_line(r.desc)) if r.desc else "[%s] %s" % (r.acc, r.src)
                if r.block is not None:
                    add(("const", prefix + r.name + "_BASE", r.off, kind, cmt))
                    add(("const", prefix + r.name + "_BYTES", r.block, "count", None))
                elif r.is_array:
                    add(("const", prefix + r.name + "_BASE", r.off, kind, cmt))
                    add(("const", prefix + r.name + "_N", r.n, "count", None))
                    add(("const", prefix + r.name + "_STRIDE", r.stride, "count", None))
                    if r.words > 1:
                        add(("const", prefix + r.name + "_WORDS", r.words, "count", None))
                    add(("func_arr", prefix + r.name, r.off, r.stride, r.words, kind))
                else:
                    add(("const", prefix + r.name, r.off, kind, cmt))
                if r.reset is not None:
                    add(("const", prefix + r.name + "_RESET", r.reset, "mask32", None))

        add(("section", "RISC-V address map (CC §4.1); region sizes are the U200 values"))
        for name, base, size, limit, fetch, acc, rsrc, desc in m.src.RV_MAP:
            add(("const", "RV_%s_BASE" % name, base, "addr", comment(rsrc, "%s (%s%s)" % (desc, acc, ", fetch" if fetch else ""))))
            add(("const", "RV_%s_SIZE" % name, size, "addr", None))
            add(("const", "RV_%s_LIMIT" % name, limit, "addr", "end of the reserved range (exclusive)"))

        add(("section", "Ibex fast interrupts irq_fast[i] (CC §12.3)"))
        for idx, name, isrc, desc in m.src.IRQS:
            add(("const", "IRQ_FAST_" + name, idx, "count", comment(isrc, desc)))

        add(("section", "DMI registers (DB §9.1); host CSR offset = CSR_DMI(address)"))
        for row in m.src.DMI_REGS:
            addr, name, dsrc, desc = row[:4]
            add(("const", "DMI_" + name, addr, "count", comment(dsrc, desc)))
            if len(row) > 4:
                add(("const", "DMI_%s_N" % name, row[4], "count", None))

        add(("section", "DRAM map per channel (EX §11); byte offsets"))
        for name, off, size, dsrc, desc in m.src.DRAM_REGIONS:
            add(("const", "REGION_%s_OFF" % name, off, "u64", comment(dsrc, desc)))
            if size is not None:
                add(("const", "REGION_%s_SIZE" % name, size, "u64", None))

        # ---- bit-field layouts ----------------------------------------------------------------
        add(("section", "Bit-field layouts (in-place masks; _LSB, _W; values unshifted)"))
        for lay in m.layouts.values():
            add(("note", "%s (%s)%s" % (lay.name, lay.src, ": " + one_line(lay.desc) if lay.desc else "")))
            for f in lay.fields:
                add(("const", "%s_%s" % (lay.name, f.name), f.mask, "mask%d" % lay.width,
                     "%s %s" % (bits_str(f), one_line(f.desc)) if f.desc else bits_str(f)))
                add(("const", "%s_%s_LSB" % (lay.name, f.name), f.lsb, "count", None))
                add(("const", "%s_%s_W" % (lay.name, f.name), f.width, "count", None))
            for fname, vals in lay.values.items():
                fw = lay.field(fname).width
                for v, vname, vdesc in vals:
                    add(("const", "%s_%s_%s" % (lay.name, fname, vname), v, "val%d" % fw, one_line(vdesc) or None))
            for bit, rname, rdesc in lay.retired:
                add(("note", "bit %d retired (%s), never set, never reused" % (bit, rdesc)))

        add(("section", "Wide layouts (bit positions only)"))
        for lay in m.wide.values():
            add(("note", "%s (%s): %s" % (lay.name, lay.src, one_line(lay.desc))))
            add(("const", "%s_BITS" % lay.name, lay.width, "count", None))
            for f in lay.fields:
                add(("const", "%s_%s_LSB" % (lay.name, f.name), f.lsb, "count", "%s %s" % (bits_str(f), one_line(f.desc))))
                add(("const", "%s_%s_W" % (lay.name, f.name), f.width, "count", None))
            for fname, vals in lay.values.items():
                fw = lay.field(fname).width
                for v, vname, vdesc in vals:
                    add(("const", "%s_%s_%s" % (lay.name, fname, vname), v, "val%d" % fw, one_line(vdesc) or None))

        add(("section", "Word tables of fixed-size records and structures (word indices)"))
        for tname, t in m.word_tables.items():
            add(("note", "%s (%s): %s" % (tname, t["src"], one_line(t["desc"]))))
            add(("const", "%s_NWORDS" % tname, t["words"], "count", None))
            for row in t["rows"]:
                w, rname, rdesc = row[:3]
                add(("const", "%s_%s_WORD" % (tname, rname), w, "count", one_line(rdesc) or None))
                if len(row) > 3:
                    add(("const", "%s_%s_N" % (tname, rname), row[3], "count", None))

        # ---- ISA tables ---------------------------------------------------------------------------
        add(("section", "Units (EX §7.3, CC §7.4, §9.4)"))
        for uid, name, qd, pd, acc, usrc, desc in m.units:
            add(("const", "UNIT_" + name, uid, "unit", comment(usrc, desc)))
        for uid, name, usrc, desc in m.src.UNITS_SPECIAL:
            add(("const", "UNIT_" + name, uid, "unit", comment(usrc, desc)))
        lo, hi, rsrc = m.src.UNITS_RESERVED
        add(("const", "UNIT_RSVD_FIRST", lo, "unit", comment(rsrc, "ids %d..%d reserved: a dependency on them is BAD_DEP" % (lo, hi))))
        add(("const", "UNIT_RSVD_LAST", hi, "unit", None))
        for uid, name, qd, pd, acc, usrc, desc in m.units:
            add(("const", "QDEPTH_U200_" + name, qd, "count", "Q_DEPTH of the U200 M6 bitstream" if uid == 0 else None))
        for uid, name, qd, pd, acc, usrc, desc in m.units:
            add(("const", "QDEPTH_PARAM_" + name, pd, "count", "depth parameter when the unit is built" if uid == 0 else None))
        for uid, name, qd, pd, acc, usrc, desc in m.units:
            add(("const", "ACCEPT_MAX_" + name, acc, "count", "I3: ops accepted and not done (CC §9.4)" if uid == 0 else None))
        add(("const", "QDEPTH_U200_REG", m.q_depth_value(), "mask32", "Q_DEPTH register value of the U200 M6 bitstream"))

        add(("section", "Opcodes (EX §7.2); reserved and retired opcodes complete ILLEGAL_OP on SYNC"))
        for o in m.ops:
            if o["mnem"] and o["code"] == o["range"][0]:
                cmt = "%s, unit %s%s: %s" % (o["status"], o["unit"], ", CAPS." + o["caps"] if o["caps"] else "",
                                             comment(o["src"], o["desc"]))
                add(("const", "OP_" + o["mnem"], o["code"], "op", cmt))
        add(("func_src", "op_tables", gen_op_funcs(m)))

        add(("section", "Error codes (EX §8.1)"))
        for code, name, status, esrc, cond, info in m.errors:
            if name is None or status == "retired":
                continue
            c = one_line(cond)
            if info:
                c += "; ERR_INFO = " + info
            add(("const", "ERR_" + name, code, "err", ("%s %s" % ("(reserved)" if status == "reserved" else "", c)).strip() or None))
        for code, name, status, esrc, cond, info in m.errors:
            if status == "retired":
                add(("note", "code %d retired (v1 %s), never raised, never reused" % (code, name)))
        add(("const", "ERR_CHECK_N", len(m.check_order), "count", "dispatch check order: " + " > ".join(m.check_order)))
        add(("func_src", "err_tables", gen_err_funcs(m)))

        add(("section", "PERF slots (EX §6): C = NUM_CH, X = 5 + 2C"))
        for slot, name, kind, psrc, desc in m.perf:
            if name is None:
                add(("note", "slot %s %s (%s)" % (slot, one_line(desc), psrc)))
            elif isinstance(slot, int):
                add(("const", "PERF_" + name, slot, "count", comment(psrc, "%s, %s" % (kind, desc))))
            elif slot[0] == "x":
                add(("const", "PERF_XOFF_" + name, slot[1], "count", comment(psrc, "X + %d, %s, %s" % (slot[1], kind, desc))))
        add(("func_src", "perf", gen_perf_funcs(m)))

        add(("section", "Instruction encodings of the v1 ops, unchanged (spec ISA, EX §7.2)"))
        for e in m.encs.values():
            if e.owner == "spec":
                emit_enc(m, e, add)

    if want("vu"):
        add(("section", "VU encodings (VU §5)"))
        for e in m.encs.values():
            if e.owner == "VU":
                emit_enc(m, e, add)
        add(("func_src", "vu_funcs", gen_vu_funcs(m)))
    if want("at"):
        add(("section", "AT encodings (AT §5; VU §5.11 restates the vector-op encodings)"))
        for e in m.encs.values():
            if e.owner == "AT":
                emit_enc(m, e, add)
    return items


def emit_enc(m: Model, e: Enc, add):
    add(("note", "%s 0x%02X (%s, owner %s)" % (e.name, e.opcode, e.src, e.owner)))
    for f in e.fields:
        add(("const", "%s_%s" % (e.name, f.name), f.mask, "mask32", "%s %s" % (f.spec, one_line(f.desc)) if f.desc else f.spec))
        add(("const", "%s_%s_LSB" % (e.name, f.name), f.lsb, "count", None))
        add(("const", "%s_%s_W" % (e.name, f.name), f.width, "count", None))
        add(("const", "%s_%s_WORD" % (e.name, f.name), f.word, "count", None))
        add(("const", "%s_%s_BIT" % (e.name, f.name), f.bit, "count", None))
    for fname, vals in e.values.items():
        fw = next(f for f in e.fields if f.name == fname).width
        for v, vname, vdesc in vals:
            add(("const", "%s_%s_%s" % (e.name, fname, vname), v, "val%d" % fw, one_line(vdesc) or None))
    for fname, ename in e.value_refs.items():
        add(("note", "%s_%s takes the %s_* values" % (e.name, fname, ename)))
    for i in range(8):
        add(("const", "%s_RSV_W%d" % (e.name, i), e.rsv[i], "mask32", "must be 0 (ILLEGAL_OP)" if i == 0 else None))
    add(("const", "%s_WORDS" % e.name, e.words_mask, "val8", "words holding a field (w0 always)"))
    if e.owner in ("VU", "AT"):
        add(("builder", e))


# ---- per-language function sources ---------------------------------------------------------------
FIELD_HELPERS = {
    "c": """\
// Field access. F is a field macro name: TSR_GET(v, TSR_CTRL_RV_RUN), TSR_PUT(TSR_RT_STATE_PHASE, 2).
#define TSR_GET(v, F)          ((((uint32_t)(v)) & (F)) >> F##_LSB)
#define TSR_PUT(F, x)          ((((uint32_t)(x)) << F##_LSB) & (F))
// Instruction fields: w is the word array of an instruction, F an encoding field (TSR_VOP_D).
#define TSR_INSN_GET(w, F)     ((((w)[F##_WORD]) & (F)) >> F##_LSB)
#define TSR_INSN_SET(w, F, x)  ((w)[F##_WORD] = ((w)[F##_WORD] & ~(uint32_t)(F)) | ((((uint32_t)(x)) << F##_LSB) & (F)))""",
    "cpp": """\
// Field access with the generated mask / _LSB pairs: field_get(v, CTRL_RV_RUN, CTRL_RV_RUN_LSB).
constexpr uint32_t field_get(uint32_t v, uint32_t mask, uint32_t lsb) { return (v & mask) >> lsb; }
constexpr uint32_t field_put(uint32_t mask, uint32_t lsb, uint32_t x) { return (x << lsb) & mask; }

// One 256-bit tile instruction with its used-word mask (docs/ctrl-riscv.md §8.2): issue() writes
// only the staging words whose bit is set and pushes with WMASK = used (ISSUE_PUSH, CC §5.1);
// words with a clear bit are sent as 0. The generated builders set the bit of every nonzero word.
struct op {
    uint32_t w[8];
    uint8_t used;
};
constexpr op op_finish(op o) {
    o.used = 1u;
    for (uint32_t i = 1; i < 8; ++i)
        if (o.w[i] != 0u) o.used = uint8_t(o.used | (1u << i));
    return o;
}""",
    "py": '''\
def field_get(v, mask, lsb):
    """Field value from a word, with a generated mask / _LSB pair."""
    return (v & mask) >> lsb


def field_put(mask, lsb, x):
    """Word bits of a field value, with a generated mask / _LSB pair."""
    return (x << lsb) & mask''',
    "sv": "",
}


def case_lines(labels, tail, indent="    ", width=100):
    """Switch-case labels wrapped to `width`, the last line ending with `tail`."""
    lines, cur = [], indent
    for lab in labels:
        if len(cur) + len(lab) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += lab + " "
    if len(cur) + len(tail) > width and cur.strip():
        lines.append(cur.rstrip())
        cur = indent
    lines.append(cur + tail)
    return lines


def sv_case_lines(labels, tail, indent="    ", width=100):
    """SV case item: comma-separated labels wrapped to `width`, then ': ' + tail."""
    lines, cur = [], indent
    for i, lab in enumerate(labels):
        piece = lab + ("," if i < len(labels) - 1 else ":")
        if len(cur) + len(piece) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece + " "
    if len(cur) + len(tail) > width and cur.strip():
        lines.append(cur.rstrip())
        cur = indent
    lines.append(cur + tail)
    return lines


def _cases(pairs):
    """pairs: list of (value, key) -> {key: [values]} keeping order of first appearance."""
    out = {}
    for v, k in pairs:
        out.setdefault(k, []).append(v)
    return out


def gen_op_funcs(m: Model) -> dict:
    legal = [o["code"] for o in m.ops if o["legal"]]
    by_unit = _cases([(c, m.op_unit(c)) for c in legal if m.op_unit(c) != m.unit_id["SYNC"]])
    by_caps = _cases([(c, o["caps"]) for c, o in ((c, m.ops[c]) for c in legal) if o["caps"]])
    unit_name = {v: k for k, v in m.unit_id.items()}
    # C
    c = ["// Unit of a legal opcode; UNIT_SYNC for every reserved or retired opcode (they complete",
         "// ILLEGAL_OP on SYNC). A legal opcode is refused the same way when (CAPS & op_caps) != op_caps",
         "// or its unit is not built (ISSUE_UNITS, Q_DEPTH = 0), EX §7.2-§7.3, CC §7.4.",
         "static inline uint32_t tsr_op_unit(uint32_t opcode) {", "    switch (opcode & 0xFFu) {"]
    for u, codes in by_unit.items():
        c += case_lines(["case 0x%02Xu:" % x for x in codes], "return TSR_UNIT_%s;" % unit_name[u])
    c += ["    default: return TSR_UNIT_SYNC;", "    }", "}",
          "// 1 for kept, re-specified and new opcodes; 0 for reserved and retired ones.",
          "static inline uint32_t tsr_op_legal(uint32_t opcode) {", "    switch (opcode & 0xFFu) {",
          *case_lines(["case 0x%02Xu:" % x for x in legal], "return 1u;"),
          "    default: return 0u;", "    }", "}",
          "// CAPS bits that must all be set for a legal opcode to be accepted (0 = always).",
          "static inline uint32_t tsr_op_caps(uint32_t opcode) {", "    switch (opcode & 0xFFu) {"]
    for cap, codes in by_caps.items():
        c += case_lines(["case 0x%02Xu:" % x for x in codes], "return TSR_CAPS_%s;" % cap)
    c += ["    default: return 0u;", "    }", "}"]
    # C++
    cpp = ["// Unit of a legal opcode; UNIT_SYNC for every reserved or retired opcode. A legal opcode is",
           "// refused the same way when (CAPS & op_caps) != op_caps or its unit is not built.",
           "constexpr uint32_t op_unit(uint32_t opcode) {", "    switch (opcode & 0xFFu) {"]
    for u, codes in by_unit.items():
        cpp += case_lines(["case 0x%02Xu:" % x for x in codes], "return UNIT_%s;" % unit_name[u])
    cpp += ["    default: return UNIT_SYNC;", "    }", "}",
            "constexpr bool op_legal(uint32_t opcode) {", "    switch (opcode & 0xFFu) {",
            *case_lines(["case 0x%02Xu:" % x for x in legal], "return true;"),
            "    default: return false;", "    }", "}",
            "constexpr uint32_t op_caps(uint32_t opcode) {", "    switch (opcode & 0xFFu) {"]
    for cap, codes in by_caps.items():
        cpp += case_lines(["case 0x%02Xu:" % x for x in codes], "return CAPS_%s;" % cap)
    cpp += ["    default: return 0u;", "    }", "}"]
    # SV
    sv = ["// Unit of a legal opcode; TSR_UNIT_SYNC for every reserved or retired opcode. A legal opcode",
          "// is refused the same way when (CAPS & tsr_op_caps) != tsr_op_caps or its unit is not built.",
          "function automatic logic [3:0] tsr_op_unit(input logic [7:0] opcode);", "  case (opcode)"]
    for u, codes in by_unit.items():
        sv += sv_case_lines(["8'h%02X" % x for x in codes], "return TSR_UNIT_%s;" % unit_name[u])
    sv += ["    default: return TSR_UNIT_SYNC;", "  endcase", "endfunction", "",
           "function automatic logic tsr_op_legal(input logic [7:0] opcode);", "  case (opcode)",
           *sv_case_lines(["8'h%02X" % x for x in legal], "return 1'b1;"),
           "    default: return 1'b0;", "  endcase", "endfunction", "",
           "function automatic logic [31:0] tsr_op_caps(input logic [7:0] opcode);", "  case (opcode)"]
    for cap, codes in by_caps.items():
        sv += sv_case_lines(["8'h%02X" % x for x in codes], "return TSR_CAPS_%s;" % cap)
    sv += ["    default: return 32'h0;", "  endcase", "endfunction"]
    # Python
    py = ["# Per opcode (index = opcode): unit of a legal opcode (UNIT_SYNC otherwise), legality, CAPS bits.",
          "OP_UNIT = [%s]" % ", ".join(str(m.op_unit(x)) for x in range(256)),
          "OP_LEGAL = [%s]" % ", ".join("1" if m.ops[x]["legal"] else "0" for x in range(256)),
          "OP_CAPS = [%s]" % ", ".join(str(m.op_caps(x)) for x in range(256)),
          "OPCODES = {%s}" % ", ".join("%r: 0x%02X" % (o["mnem"], o["code"]) for o in m.ops
                                       if o["mnem"] and o["code"] == o["range"][0])]
    return {"c": "\n".join(c), "cpp": "\n".join(cpp), "sv": "\n".join(sv), "py": "\n".join(_wrap_py(py))}


def _wrap_py(lines, width=100):
    """Wrap long Python list / dict literals after commas (keeps the module readable)."""
    out = []
    for line in lines:
        if len(line) <= width or line.startswith("#"):
            out.append(line)
            continue
        head, _, rest = line.partition("= ")
        opener = rest[0]
        closer = {"[": "]", "{": "}", "(": ")"}[opener]
        body = rest[1:-1]
        parts = [p.strip() for p in body.split(", ")]
        out.append(head + "= " + opener)
        cur = "    "
        for p in parts:
            if len(cur) + len(p) + 2 > width:
                out.append(cur.rstrip())
                cur = "    "
            cur += p + ", "
        out.append(cur.rstrip())
        out.append(closer)
    return out


def gen_err_funcs(m: Model) -> dict:
    order = m.check_order
    prec = m.status_prec
    c = ["// Rank of an error code in the dispatch check order (0 = checked first); 0xFF if not a check.",
         "static inline uint32_t tsr_err_check_rank(uint32_t code) {", "    switch (code) {"]
    c += ["    case TSR_ERR_%s: return %du;" % (n, i) for i, n in enumerate(order)]
    c += ["    default: return 0xFFu;", "    }", "}",
          "// Rank of a launch STATUS in the precedence of EX §4.4 (0 = highest); 0xFF if reserved.",
          "static inline uint32_t tsr_lcq_status_rank(uint32_t status) {", "    switch (status) {"]
    c += ["    case TSR_LCQ_R0_STATUS_%s: return %du;" % (n, i) for i, n in enumerate(prec)]
    c += ["    default: return 0xFFu;", "    }", "}"]
    cpp = ["constexpr uint32_t ERR_CHECK_ORDER[%d] = {%s};" % (len(order), ", ".join("ERR_" + n for n in order)),
           "constexpr uint32_t LCQ_STATUS_PRECEDENCE[%d] = {%s};" % (len(prec), ", ".join("LCQ_R0_STATUS_" + n for n in prec)),
           "constexpr uint32_t err_check_rank(uint32_t code) {",
           "    for (uint32_t i = 0; i < %d; ++i)" % len(order),
           "        if (ERR_CHECK_ORDER[i] == code) return i;", "    return 0xFFu;", "}",
           "constexpr uint32_t lcq_status_rank(uint32_t status) {",
           "    for (uint32_t i = 0; i < %d; ++i)" % len(prec),
           "        if (LCQ_STATUS_PRECEDENCE[i] == status) return i;", "    return 0xFFu;", "}"]
    sv = ["localparam logic [3:0] TSR_ERR_CHECK_ORDER [%d] = '{%s};" % (len(order), ", ".join("TSR_ERR_" + n for n in order)),
          "",
          "function automatic int unsigned tsr_err_check_rank(input logic [3:0] code);",
          "  for (int unsigned i = 0; i < %d; i++)" % len(order),
          "    if (TSR_ERR_CHECK_ORDER[i] == code) return i;",
          "  return 32'hFF;", "endfunction"]
    py = ["ERR_CHECK_ORDER = [%s]" % ", ".join("ERR_" + n for n in order),
          "LCQ_STATUS_PRECEDENCE = [%s]" % ", ".join("LCQ_R0_STATUS_" + n for n in prec),
          "", "",
          "def err_check_rank(code):",
          '    """Rank of an error code in the dispatch check order (0 = first); 0xFF if not a check."""',
          "    return ERR_CHECK_ORDER.index(code) if code in ERR_CHECK_ORDER else 0xFF",
          "", "",
          "def lcq_status_rank(status):",
          '    """Rank of a launch STATUS in the precedence of EX §4.4 (0 = highest); 0xFF if reserved."""',
          "    return LCQ_STATUS_PRECEDENCE.index(status) if status in LCQ_STATUS_PRECEDENCE else 0xFF"]
    return {"c": "\n".join(c), "cpp": "\n".join(cpp), "sv": "\n".join(sv), "py": "\n".join(_wrap_py(py))}


def gen_perf_funcs(m: Model) -> dict:
    chs = {name: slot[1] for slot, name, *_ in m.perf if isinstance(slot, tuple) and slot[0] == "ch"}
    xs = [(name, slot[1]) for slot, name, *_ in m.perf if isinstance(slot, tuple) and slot[0] == "x"]
    c = ["#define TSR_PERF_X(C)              (5u + 2u * (C))          // first extension slot",
         "#define TSR_PERF_EXT(C, xoff)      (TSR_PERF_X(C) + (xoff))"]
    for name, k in chs.items():
        if k == 0:
            c.append("#define TSR_PERF_%s(c) (5u + (c))" % name)
        else:
            c.append("#define TSR_PERF_%s(C, c) (5u + %du * (C) + (c))" % (name, k))
    for name, k in xs:
        c.append("#define TSR_PERF_%s(C) (TSR_PERF_X(C) + %du)" % (name, k))
    cpp = ["constexpr uint32_t perf_x(uint32_t C) { return 5u + 2u * C; }   // first extension slot",
           "constexpr uint32_t perf_ext(uint32_t C, uint32_t xoff) { return perf_x(C) + xoff; }"]
    for name, k in chs.items():
        if k == 0:
            cpp.append("constexpr uint32_t perf_%s(uint32_t c) { return 5u + c; }" % name.lower())
        else:
            cpp.append("constexpr uint32_t perf_%s(uint32_t C, uint32_t c) { return 5u + %du * C + c; }" % (name.lower(), k))
    sv = ["function automatic int unsigned tsr_perf_x(input int unsigned num_ch);",
          "  return 5 + 2 * num_ch;", "endfunction", "",
          "function automatic int unsigned tsr_perf_ext(input int unsigned num_ch, input int unsigned xoff);",
          "  return tsr_perf_x(num_ch) + xoff;", "endfunction"]
    for name, k in chs.items():
        if k == 0:
            sv += ["", "function automatic int unsigned tsr_perf_%s(input int unsigned c);" % name.lower(),
                   "  return 5 + c;", "endfunction"]
        else:
            sv += ["", "function automatic int unsigned tsr_perf_%s(input int unsigned num_ch, input int unsigned c);" % name.lower(),
                   "  return 5 + %d * num_ch + c;" % k, "endfunction"]
    py = ["def perf_x(C):", '    """First extension slot X = 5 + 2C."""', "    return 5 + 2 * C", "", "",
          "def perf_ext(C, xoff):", "    return perf_x(C) + xoff"]
    for name, k in chs.items():
        py += ["", ""]
        if k == 0:
            py += ["def perf_%s(c):" % name.lower(), "    return 5 + c"]
        else:
            py += ["def perf_%s(C, c):" % name.lower(), "    return 5 + %d * C + c" % k]
    py += ["", "",
           "def perf_slots(C):",
           '    """Every named PERF slot for NUM_CH = C: name -> index (per-channel names get [c])."""',
           "    out = {%s}" % ", ".join("%r: %d" % (name, slot) for slot, name, *_ in m.perf
                                         if name is not None and isinstance(slot, int)),
           "    for c in range(C):"]
    for name, k in chs.items():
        py.append("        out['%s[%%d]' %% c] = 5 + %d * C + c" % (name, k))
    py += ["    for name, xoff in [%s]:" % ", ".join("(%r, %d)" % (n, k) for n, k in xs),
           "        out[name] = perf_x(C) + xoff",
           "    return out"]
    return {"c": "\n".join(c), "cpp": "\n".join(cpp), "sv": "\n".join(sv), "py": "\n".join(py)}


def gen_vu_funcs(m: Model) -> dict:
    reads = [(c, sum({"a": 1, "b": 2, "c": 4}[x] for x in rd)) for c, n, rd, rt, res in m.vop_funcs]
    half = [c for c, n, rd, rt, res in m.vop_funcs if rt == "H"]
    sop_bits = {"SD": 1, "SA": 2, "SB": 4, "IMM": 8, "ADDR": 16}
    uses = [(c, sum(sop_bits[u] for u in us)) for c, n, us, eff in m.sop_funcs]
    nvop, nsop = len(m.vop_funcs), len(m.sop_funcs)
    by_reads = _cases([(c, r) for c, r in reads])
    by_uses = _cases([(c, u) for c, u in uses])
    c = ["#define TSR_VOP_FUNC_N     %du    // function codes 0..%d are defined; others ILLEGAL_OP" % (nvop, nvop - 1),
         "#define TSR_VOP_READS_A    1u",
         "#define TSR_VOP_READS_B    2u",
         "#define TSR_VOP_READS_C    4u",
         "// Operands a VOP function reads (TSR_VOP_READS_*); an operand it does not read must be 0, and",
         "// a function that reads c needs CMODE != NONE, one that does not needs CMODE NONE (VU §5.2).",
         "static inline uint32_t tsr_vop_func_reads(uint32_t func) {", "    switch (func) {"]
    for r, codes in by_reads.items():
        c += case_lines(["case 0x%02Xu:" % x for x in codes], "return %du;" % r)
    c += ["    default: return 0u;", "    }", "}",
          "// 1 for the half-rate (back-lane) functions, VU §5.2, §7.4.",
          "static inline uint32_t tsr_vop_func_half(uint32_t func) {", "    switch (func) {",
          *case_lines(["case 0x%02Xu:" % x for x in half], "return 1u;"),
          "    default: return 0u;", "    }", "}",
          "#define TSR_SOP_SUB_N      %du    // sub codes 0..%d are defined; %d ILLEGAL_OP" % (nsop, nsop - 1, nsop),
          "#define TSR_SOP_USES_SD    1u", "#define TSR_SOP_USES_SA    2u", "#define TSR_SOP_USES_SB    4u",
          "#define TSR_SOP_USES_IMM   8u", "#define TSR_SOP_USES_ADDR  16u",
          "// Fields an SOP function uses (TSR_SOP_USES_*); the others must be 0 (VU §5.8).",
          "static inline uint32_t tsr_sop_uses(uint32_t sub) {", "    switch (sub) {"]
    for u, codes in by_uses.items():
        c += case_lines(["case %du:" % x for x in codes], "return %du;" % u)
    c += ["    default: return 0u;", "    }", "}"]
    cpp = ["constexpr uint32_t VOP_FUNC_N = %du;   // function codes 0..%d are defined" % (nvop, nvop - 1),
           "constexpr uint32_t VOP_READS_A = 1u, VOP_READS_B = 2u, VOP_READS_C = 4u;",
           "// Operands a VOP function reads; an operand it does not read must be 0 (VU §5.2).",
           "constexpr uint32_t vop_func_reads(uint32_t func) {", "    switch (func) {"]
    for r, codes in by_reads.items():
        cpp += case_lines(["case 0x%02Xu:" % x for x in codes], "return %du;" % r)
    cpp += ["    default: return 0u;", "    }", "}",
            "constexpr bool vop_func_half(uint32_t func) {", "    switch (func) {",
            *case_lines(["case 0x%02Xu:" % x for x in half], "return true;"),
            "    default: return false;", "    }", "}",
            "constexpr uint32_t SOP_SUB_N = %du;" % nsop,
            "constexpr uint32_t SOP_USES_SD = 1u, SOP_USES_SA = 2u, SOP_USES_SB = 4u, SOP_USES_IMM = 8u, SOP_USES_ADDR = 16u;",
            "// Fields an SOP function uses; the others must be 0 (VU §5.8).",
            "constexpr uint32_t sop_uses(uint32_t sub) {", "    switch (sub) {"]
    for u, codes in by_uses.items():
        cpp += case_lines(["case %du:" % x for x in codes], "return %du;" % u)
    cpp += ["    default: return 0u;", "    }", "}"]
    sv = ["localparam int unsigned TSR_VOP_FUNC_N = %d;" % nvop,
          "localparam int unsigned TSR_SOP_SUB_N = %d;" % nsop,
          "",
          "// Operands a VOP function reads: bit0 a, bit1 b, bit2 c (VU §5.2).",
          "function automatic logic [2:0] tsr_vop_func_reads(input logic [7:0] func);", "  case (func)"]
    for r, codes in by_reads.items():
        sv += sv_case_lines(["8'h%02X" % x for x in codes], "return 3'd%d;" % r)
    sv += ["    default: return 3'd0;", "  endcase", "endfunction", "",
           "// Half-rate (back-lane) VOP functions (VU §5.2, §7.4).",
           "function automatic logic tsr_vop_func_half(input logic [7:0] func);", "  case (func)",
           *sv_case_lines(["8'h%02X" % x for x in half], "return 1'b1;"),
           "    default: return 1'b0;", "  endcase", "endfunction", "",
           "// Fields an SOP function uses: bit0 SD, bit1 SA, bit2 SB, bit3 IMM, bit4 ADDR (VU §5.8).",
           "function automatic logic [4:0] tsr_sop_uses(input logic [3:0] sub);", "  case (sub)"]
    for u, codes in by_uses.items():
        sv += sv_case_lines(["4'd%d" % x for x in codes], "return 5'd%d;" % u)
    sv += ["    default: return 5'd0;", "  endcase", "endfunction"]
    py = ["VOP_FUNC_N = %d" % nvop,
          "VOP_READS_A, VOP_READS_B, VOP_READS_C = 1, 2, 4",
          "# VOP function code -> (name, operands read (VOP_READS_*), half rate)",
          "VOP_FUNCS = {%s}" % ", ".join("0x%02X: (%r, %d, %d)" % (c_, n, dict(reads)[c_], 1 if rt == "H" else 0)
                                         for c_, n, rd, rt, res in m.vop_funcs),
          "SOP_SUB_N = %d" % nsop,
          "SOP_USES_SD, SOP_USES_SA, SOP_USES_SB, SOP_USES_IMM, SOP_USES_ADDR = 1, 2, 4, 8, 16",
          "# SOP sub code -> (name, fields used (SOP_USES_*))",
          "SOP_FUNCS = {%s}" % ", ".join("%d: (%r, %d)" % (c_, n, dict(uses)[c_]) for c_, n, us, eff in m.sop_funcs)]
    return {"c": "\n".join(c), "cpp": "\n".join(cpp), "sv": "\n".join(sv), "py": "\n".join(_wrap_py(py))}


# ================================================================================================
# C back end
# ================================================================================================
def c_value(value: int, kind: str) -> str:
    if kind == "off12":
        return "0x%03Xu" % value
    if kind == "off13":
        return "0x%04Xu" % value
    if kind in ("addr", "mask32") or kind.startswith("mask"):
        return "0x%08Xu" % value
    if kind == "u64":
        return "0x%Xull" % value
    if kind == "op":
        return "0x%02Xu" % value
    if kind == "hex":
        return "0x%08Xu" % value if value > 0xFFFF else "0x%Xu" % value
    if kind.startswith("val"):
        return ("0x%Xu" % value) if value > 15 else "%du" % value
    return "%du" % value


def gen_c(m: Model) -> str:
    out = ["// " + BANNER,
           "//",
           "// tsr_tables.h — Tesseract v2 global tables (V2-EX-30): host CSR map and DBG block, RISC-V",
           "// address map and control window, adapter map, bit-field layouts, launch / log / trace",
           "// formats, opcodes and the opcode -> unit table, units and queue depths, error codes and",
           "// the check order, PERF slots, and every instruction encoding (v1 ops, VU, AT).",
           "// Included by tsr_isa.h. Names: see sw/tables/gen_tables.py. Usable from C11 and C++.",
           "#ifndef TSR_TABLES_H",
           "#define TSR_TABLES_H",
           "",
           "#include <stdint.h>",
           "",
           "#ifdef __cplusplus",
           'extern "C" {',
           "#endif"]
    block = []

    def flush():
        out.extend(rows_aligned(block))
        block.clear()

    for it in walk(m, "all"):
        kind = it[0]
        if kind == "section":
            flush()
            out += ["", "// " + "-" * 95, "// " + it[1], "// " + "-" * 95]
        elif kind == "note":
            flush()
            out.append("// " + it[1])
        elif kind == "const":
            _, name, value, vk, cmt = it
            block.append(("#define TSR_" + name, c_value(value, vk), "// " + cmt if cmt else None))
        elif kind == "func_arr":
            _, name, base, stride, words, vk = it
            flush()
            if words > 1:
                out.append("#define TSR_%s(i, j) (0x%03Xu + %du * (i) + 4u * (j))" % (name, base, stride))
            else:
                out.append("#define TSR_%s(i) (0x%03Xu + %du * (i))" % (name, base, stride))
        elif kind == "func_src":
            flush()
            src = it[2].get("c", "")
            if src:
                out.append(src)
        elif kind == "builder":
            pass
    flush()
    out += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif  // TSR_TABLES_H", ""]
    return "\n".join(out)


# ================================================================================================
# C++ back end (freestanding C++20; only <stdint.h>)
# ================================================================================================
def cpp_value(value: int, kind: str) -> str:
    return c_value(value, kind)


def cpp_type(kind: str) -> str:
    return "uint64_t" if kind == "u64" else "uint32_t"


def gen_cpp(m: Model, part: str) -> str:
    title = {
        "mmio": ("mmio.hpp", "the RISC-V and host maps, opcodes, units, errors, PERF, the launch / log / trace\n"
                 "// formats, the v1-op encodings and tsr::op"),
        "vu": ("vu.hpp", "the VU encodings (VU §5) and their constexpr builders"),
        "at": ("at.hpp", "the AT encodings (AT §5) and their constexpr builders"),
    }[part]
    out = ["// " + BANNER,
           "//",
           "// tsr/%s — Tesseract v2 tables for the firmware SDK (V2-EX-30, SW §6):" % title[0],
           "// %s." % title[1],
           "// Freestanding C++20: no libstdc++ headers, no heap, everything constexpr in namespace tsr.",
           "// Names as tsr_tables.h without the TSR_ prefix (see sw/tables/gen_tables.py)."]
    if part != "mmio":
        example = {
            "vu": "VOP({.cmode = VOP_CMODE_SREG, .func = VOP_FUNC_MULC, .si = 3, .d = 100, .a = 200, .m = 896, .rows = 1})",
            "at": "ASMAX({.dst = 100, .src = 100, .mvec = 300, .n_lines = 4, .r = 14, .scale = 0x3E000000u})",
        }[part]
        out += ["// Builders take a field struct in encoding order (C++20 designated initializers; omitted",
                "// fields are 0, as every unused field must be), mask each value to its field width and",
                "// mark every nonzero word in op::used:",
                "//     " + example]
    guard = "TSR_%s_HPP" % part.upper()
    out += ["#ifndef " + guard, "#define " + guard, "", "#include <stdint.h>"]
    if part != "mmio":
        out += ['', '#include "mmio.hpp"']
    out += ["", "namespace tsr {"]
    block = []

    def flush():
        out.extend(rows_aligned(block))
        block.clear()

    for it in walk(m, part):
        kind = it[0]
        if kind == "section":
            flush()
            out += ["", "// " + "-" * 95, "// " + it[1], "// " + "-" * 95]
        elif kind == "note":
            flush()
            out.append("// " + it[1])
        elif kind == "const":
            _, name, value, vk, cmt = it
            block.append(("constexpr %s %s" % (cpp_type(vk), name), "= " + cpp_value(value, vk) + ";", "// " + cmt if cmt else None))
        elif kind == "func_arr":
            _, name, base, stride, words, vk = it
            flush()
            if words > 1:
                out.append("constexpr uint32_t %s(uint32_t i, uint32_t j) { return 0x%03Xu + %du * i + 4u * j; }" % (name, base, stride))
            else:
                out.append("constexpr uint32_t %s(uint32_t i) { return 0x%03Xu + %du * i; }" % (name, base, stride))
        elif kind == "func_src":
            flush()
            src = it[2].get("cpp", "")
            if src:
                out.append(src)
        elif kind == "builder":
            flush()
            out += gen_cpp_builder(it[1])
    flush()
    out += ["", "}  // namespace tsr", "", "#endif  // " + guard, ""]
    return "\n".join(out)


def gen_cpp_builder(e: Enc) -> list[str]:
    sname = e.name.lower() + "_f"
    members = [("    uint32_t %s = 0;" % f.name.lower(), "// %s %s" % (f.spec, one_line(f.desc)) if f.desc else "// " + f.spec)
               for f in e.fields]
    w = max(len(a) for a, _ in members) if members else 0
    lines = ["struct %s {" % sname] + ["%s  %s" % (a.ljust(w), b) for a, b in members] + ["};"]
    lines.append("constexpr op %s(const %s& f) {" % (e.name, sname))
    lines.append("    op o{};")
    lines.append("    o.w[0] = OP_%s;" % e.name)
    for f in e.fields:
        if f.width == 32:
            lines.append("    o.w[%d] |= f.%s;" % (f.word, f.name.lower()))
        else:
            lines.append("    o.w[%d] |= (f.%s & 0x%Xu) << %d;" % (f.word, f.name.lower(), f.vmask, f.lsb))
    lines.append("    return op_finish(o);")
    lines.append("}")
    return lines


# ================================================================================================
# SystemVerilog back end
# ================================================================================================
def sv_decl(name: str, value: int, kind: str) -> str:
    if kind == "off12":
        return "localparam logic [11:0] TSR_%s = %s;" % (name, sv_hex(value, 12))
    if kind == "off13":
        return "localparam logic [12:0] TSR_%s = %s;" % (name, sv_hex(value, 13))
    if kind in ("addr", "mask32"):
        return "localparam logic [31:0] TSR_%s = %s;" % (name, sv_hex(value, 32))
    if kind.startswith("mask"):
        w = int(kind[4:])
        return "localparam logic [%d:0] TSR_%s = %s;" % (w - 1, name, sv_hex(value, w))
    if kind.startswith("val"):
        w = int(kind[3:])
        return "localparam logic [%d:0] TSR_%s = %d'd%d;" % (w - 1, name, w, value)
    if kind == "op":
        return "localparam logic [7:0] TSR_%s = %s;" % (name, sv_hex(value, 8))
    if kind == "unit":
        return "localparam logic [3:0] TSR_%s = 4'd%d;" % (name, value)
    if kind == "err":
        return "localparam logic [3:0] TSR_%s = 4'd%d;" % (name, value)
    if kind == "u64":
        return "localparam longint unsigned TSR_%s = 64'h%X;" % (name, value)
    if kind == "hex":
        return "localparam int unsigned TSR_%s = %s;" % (name, sv_hex(value, 32))
    return "localparam int unsigned TSR_%s = %d;" % (name, value)


def gen_sv(m: Model) -> str:
    out = ["// " + BANNER,
           "//",
           "// tsr_tables_pkg.sv — Tesseract v2 global tables (V2-EX-30) for the RTL: CSR offsets, opcodes,",
           "// units, error codes, PERF indices, bit-field layouts and instruction field positions; the",
           "// opcode -> unit table as functions. C mirror: sw/sim/include/tsr_tables.h (same names).",
           "",
           "package tsr_tables_pkg;",
           "",
           "  // A catalogue of constants: no module uses all of them, so UNUSEDPARAM is off for this package.",
           "  /* verilator lint_off UNUSEDPARAM */"]
    block = []

    def flush():
        out.extend("  " + ln for ln in rows_aligned(block))
        block.clear()

    for it in walk(m, "all"):
        kind = it[0]
        if kind == "section":
            flush()
            out += ["", "  // " + "-" * 93, "  // " + it[1], "  // " + "-" * 93]
        elif kind == "note":
            flush()
            out.append("  // " + it[1])
        elif kind == "const":
            _, name, value, vk, cmt = it
            decl = sv_decl(name, value, vk)
            left, _, right = decl.partition(" = ")
            block.append((left, "= " + right, "// " + cmt if cmt else None))
        elif kind == "func_arr":
            pass   # _BASE / _N / _STRIDE are enough for the RTL
        elif kind == "func_src":
            flush()
            src = it[2].get("sv", "")
            if src:
                out.append("")
                out += [("  " + ln) if ln else "" for ln in src.split("\n")]
        elif kind == "builder":
            pass
    flush()
    out += ["", "  /* verilator lint_on UNUSEDPARAM */", "", "endpackage", ""]
    return "\n".join(out)


# ================================================================================================
# Python back end (tb/common/tables.py)
# ================================================================================================
def py_value(value: int, kind: str) -> str:
    if kind in ("off12", "off13"):
        return "0x%03X" % value
    if kind in ("addr", "mask32", "hex", "op") or kind.startswith("mask"):
        return "0x%X" % value
    if kind == "u64":
        return "0x%X" % value
    if kind.startswith("val"):
        return "0x%X" % value if value > 15 else "%d" % value
    return "%d" % value


def gen_py(m: Model) -> str:
    out = ["# " + BANNER,
           '"""Tesseract v2 global tables for the cocotb benches (V2-EX-30).',
           "",
           "Same names as sw/sim/include/tsr_tables.h without the TSR_ prefix. Besides the constants:",
           "LAYOUTS (bit-field layouts), ENCODINGS (instruction fields), OP_UNIT / OP_LEGAL / OP_CAPS,",
           "the PERF slot functions, and field_get / field_put / encode / decode.",
           '"""',
           "# ruff: noqa: E501"]
    block = []

    def flush():
        out.extend(rows_aligned(block))
        block.clear()

    for it in walk(m, "all"):
        kind = it[0]
        if kind == "section":
            flush()
            out += ["", "# " + "-" * 97, "# " + it[1], "# " + "-" * 97]
        elif kind == "note":
            flush()
            out.append("# " + it[1])
        elif kind == "const":
            _, name, value, vk, cmt = it
            block.append((name, "= " + py_value(value, vk), "# " + cmt if cmt else None))
        elif kind == "func_arr":
            _, name, base, stride, words, vk = it
            flush()
            out += ["", ""]
            if words > 1:
                out += ["def %s(i, j):" % name, "    return 0x%03X + %d * i + 4 * j" % (base, stride)]
            else:
                out += ["def %s(i):" % name, "    return 0x%03X + %d * i" % (base, stride)]
            out.append("")
        elif kind == "func_src":
            flush()
            src = it[2].get("py", "")
            if src:
                out += ["", src, ""]
        elif kind == "builder":
            pass
    flush()
    # Structured tables and helpers.
    out += ["", "# " + "-" * 97, "# Structured tables", "# " + "-" * 97]
    lay = ["LAYOUTS = {   # layout -> field -> (lsb, width)"]
    for l in list(m.layouts.values()) + list(m.wide.values()):
        lay.append("    %r: {%s}," % (l.name, ", ".join("%r: (%d, %d)" % (f.name, f.lsb, f.width) for f in l.fields)))
    lay.append("}")
    out += lay
    enc = ["ENCODINGS = {   # op -> opcode, fields: name -> (word, lsb, width), rsv: must-be-zero masks per word"]
    for e in m.encs.values():
        enc.append("    %r: dict(opcode=0x%02X, unit=%d, fields={%s}," % (
            e.name, e.opcode, m.op_unit(e.opcode),
            ", ".join("%r: (%d, %d, %d)" % (f.name, f.word, f.lsb, f.width) for f in e.fields)))
        enc.append("        rsv=[%s])," % ", ".join("0x%08X" % r for r in e.rsv))
    enc.append("}")
    out += enc
    out += ["UNITS = {%s}" % ", ".join("%r: %d" % (n, i) for i, n, *_ in m.units),
            "ERRORS = {%s}" % ", ".join("%r: %d" % (e[1], e[0]) for e in m.errors if e[1] and e[2] != "retired"),
            "", "",
            "def encode(op, **fields):",
            '    """Instruction words of `op` (an ENCODINGS key); unnamed fields are 0. Raises on a value',
            '    that does not fit its field."""',
            "    e = ENCODINGS[op]",
            "    w = [0] * 8",
            "    w[0] = e['opcode']",
            "    for name, value in fields.items():",
            "        word, lsb, width = e['fields'][name]",
            "        if value < 0 or value >> width:",
            "            raise ValueError('%s.%s = %r does not fit %d bits' % (op, name, value, width))",
            "        w[word] |= value << lsb",
            "    return w",
            "", "",
            "def decode(op, words):",
            '    """Field values of `op` from its instruction words; also checks the opcode."""',
            "    e = ENCODINGS[op]",
            "    if words[0] & 0xFF != e['opcode']:",
            "        raise ValueError('opcode 0x%02X is not %s' % (words[0] & 0xFF, op))",
            "    return {name: (words[word] >> lsb) & ((1 << width) - 1)",
            "            for name, (word, lsb, width) in e['fields'].items()}",
            "", "",
            "def reserved_bits(op, words):",
            '    """Per-word mask of set must-be-zero bits (all 0 for a well-formed instruction)."""',
            "    return [w & r for w, r in zip(words, ENCODINGS[op]['rsv'])]",
            ""]
    return "\n".join(out)


# ================================================================================================
# Driver
# ================================================================================================
def outputs(model: Model | None = None) -> dict:
    m = model or Model(load_source())
    return {
        OUT_SV: gen_sv(m),
        OUT_C: gen_c(m),
        OUT_MMIO: gen_cpp(m, "mmio"),
        OUT_VU: gen_cpp(m, "vu"),
        OUT_AT: gen_cpp(m, "at"),
        OUT_PY: gen_py(m),
    }


def check(outs: dict | None = None) -> list:
    """Paths whose file is missing or differs from the regenerated text."""
    outs = outs or outputs()
    return [p for p, text in outs.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]


def main(argv) -> int:
    outs = outputs()
    if "--check" in argv:
        stale = check(outs)
        for p in stale:
            print("gen_tables.py: %s is missing or stale (run %s)" % (p.relative_to(REPO), GENERATOR))
        return 1 if stale else 0
    for path, text in outs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            print("wrote %s" % path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

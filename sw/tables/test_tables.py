#!/usr/bin/env python3
"""sw/tables/test_tables.py — self-consistency checks of sw/tables/tsr_tables.py (V2-EX-30, P1.1).

    sw/tables/test_tables.py          run every check, print one line per check, exit 1 on a failure
    pytest sw/tables/test_tables.py   the same checks as pytest tests

Checks: every row cites a document section; no overlapping CSR / control-window / adapter offsets
or ranges; the DBG block holds the DBG rows; no overlapping RISC-V regions; bit fields fit their
32-bit word (or the 256-bit instruction) and do not overlap; opcodes cover 0x00..0xFF exactly once,
are unique, keep the EX §7.2 assignments, and the retired v1 opcodes stay unassigned; unit ids are
dense; error codes and the check order match EX §8.1; PERF slots are unique for every NUM_CH 1..16;
every name emitted into the generated files is unique per language; the generated files are
current. Expected values below are typed from the documents, not read from the table source.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load("tsr_gen_tables", HERE / "gen_tables.py")
SRC = GEN.load_source()
M = GEN.Model(SRC)

# ---- expectations typed from the documents ---------------------------------------------------------
AREAS = ("EX", "VU", "AT", "CC", "DB", "SW", "spec", "docs/")
# EX §7.2 / §7.3: assigned opcodes and their units.
ASSIGNED = {
    0x00: ("NOP", "SYNC"), 0x10: ("LD", "DMA"), 0x11: ("ST", "DMA"), 0x12: ("DQROW", "VU"),
    0x14: ("KVST", "DMA"), 0x18: ("QACT", "QACT"), 0x20: ("MVM", "MVM"), 0x21: ("ATTQK", "MVM"),
    0x22: ("ATTPV", "MVM"), 0x30: ("VOP", "VU"), 0x31: ("RED", "VU"), 0x32: ("ROPE", "VU"),
    0x33: ("TOPK", "VU"), 0x34: ("ASMAX", "VU"), 0x35: ("AMASK", "VU"), 0x36: ("AMERGE", "VU"),
    0x37: ("ANORM", "VU"), 0x38: ("VMASK", "VU"), 0x40: ("SOP", "VU"), 0x7F: ("FENCE", "SYNC"),
}
# EX §7.2: v1 opcodes that are reserved or retired and never reused.
RETIRED_V1 = {0x01: "END", 0x02: "LI", 0x03: "ADDI", 0x04: "MULI", 0x05: "RLD", 0x06: "LOOP",
              0x07: "WAITM", 0x08: "JUMP", 0x09: "CALL"}
# V2-EX-15: opcode blocks per owner.
BLOCKS = {
    "VU": [0x13, *range(0x38, 0x40), *range(0x41, 0x48)],
    "AT": [*range(0x14, 0x18), *range(0x19, 0x1C), *range(0x21, 0x24), *range(0x34, 0x38)],
    "EX": [*range(0x0A, 0x10), *range(0x70, 0x7F)],
}
UNIT_IDS = {"DMA": 0, "QACT": 1, "MVM": 2, "VU": 3, "ATT": 4, "SYNC": 5, "MARKER": 14, "NONE": 15}
Q_DEPTH_U200 = [8, 4, 4, 16, 0, 4]                                      # EX §5.1, SW §5.1 meta
CHECK_ORDER = ["ILLEGAL_OP", "BAD_DEP", "BAD_FMT", "K_RANGE", "T_RANGE", "OPERAND", "SPM_RANGE",
               "DRAM_RANGE"]                                            # EX §8.1
ERR_CODES = {"NONE": 0, "ILLEGAL_OP": 1, "BAD_FMT": 2, "QA_MISMATCH": 3, "SPM_RANGE": 4,
             "DRAM_RANGE": 5, "K_RANGE": 6, "T_RANGE": 7, "ABORTED": 8, "DEP_FAILED": 11,
             "BAD_DEP": 12, "OPERAND": 13}                              # EX §8.1
STATUS_ORDER = ["TRAP", "BAD_DESC", "ABORTED", "TILE_ERR", "PLAT_ERR", "STOPPED", "KERNEL_ERR", "OK"]
PERF_EXT = {"DMA_CYCLES": 0, "MVM_WAIT": 3, "MVM_EMPTY": 4, "RV_WFI": 12, "VU_SPM_WAIT": 13,
            "VU_LINES": 15, "ATT_QK_CYCLES": 16, "KVST_RMW": 19, "SPM_RD_WAIT": 20, "SPM_WR_WAIT": 21}
HOST_SPOT = {"ID": 0x000, "CTRL": 0x020, "STATUS": 0x024, "RV_BOOT": 0x028, "LQ_PUSH": 0x060,
             "LCQ_RD": 0x078, "LQ_CFG": 0x07C, "Q_DEPTH": 0x088, "LOG_CFG": 0x098, "LOAD_ADDR": 0x0C0,
             "LOAD_STATUS": 0x0D4, "PERF_LO": 0x100, "LCQ_REC": 0x300, "LOG_RING": 0x400, "DBG": 0x800,
             "DMI": 0xC00, "Q_DONE": 0xE00, "Q_PIDX": 0xE64, "Q_ERR_SEQ": 0xE80, "Q_ERR_PIDX": 0xF00,
             "SREG": 0x9C0, "DBG_CTRL": 0x804, "DBG_NEXT_PIDX": 0x81C, "DBG_HIST_W": 0xA40}
CW_SPOT = {"ISSUE_PUSH": 0x028, "ISSUE_CREDIT": 0x030, "ISSUE_PIDX": 0x03C, "DONE_SEQ": 0x040,
           "ERR_SEQ": 0x0C0, "QCTRL": 0x140, "REC_COUNT": 0x160, "TRACE_MARK": 0x178, "ERR_PIDX": 0x180,
           "SPM_LB_INV": 0x1C4, "LQ_POP": 0x228, "LCQ_POST": 0x260, "LOG_PUSH": 0x280,
           "RT_STATE": 0x290, "MTIME_LO": 0x300, "DBG_FLAGS": 0x400}


def _all_srcs():
    """(where, src) for every row or group that must cite a document section."""
    out = [("CONSTANTS " + c[0], c[2]) for c in SRC.CONSTANTS]
    out += [("ENUMS " + n, e[0]) for n, e in SRC.ENUMS.items()]
    out += [("LAYOUTS " + n, d["src"]) for n, d in {**SRC.LAYOUTS, **SRC.WIDE_LAYOUTS}.items()]
    for tab in ("HOST_CSR", "DBG_CSR", "CTL_WINDOW", "SAXI"):
        out += [("%s 0x%03X" % (tab, r[0]), r[4]) for r in getattr(SRC, tab)]
    out += [("RV_MAP " + r[0], r[6]) for r in SRC.RV_MAP]
    out += [("IRQS " + r[1], r[2]) for r in SRC.IRQS]
    out += [("DMI_REGS " + r[1], r[2]) for r in SRC.DMI_REGS]
    out += [("DRAM_REGIONS " + r[0], r[3]) for r in SRC.DRAM_REGIONS]
    out += [("UNITS " + r[1], r[5]) for r in SRC.UNITS]
    out += [("UNITS_SPECIAL " + r[1], r[2]) for r in SRC.UNITS_SPECIAL]
    out += [("OPCODES 0x%02X" % r[0], r[7]) for r in SRC.OPCODES]
    out += [("ERRORS %d" % r[0], r[3]) for r in SRC.ERRORS]
    out += [("PERF %s" % (r[1],), r[3]) for r in SRC.PERF]
    out += [("WORD_TABLES " + n, d["src"]) for n, d in SRC.WORD_TABLES.items()]
    out += [("ENCODINGS " + n, d["src"]) for n, d in SRC.ENCODINGS.items()]
    out += [("ERR_CHECK_ORDER", SRC.ERR_CHECK_ORDER[0]), ("STATUS_PRECEDENCE", SRC.STATUS_PRECEDENCE[0]),
            ("VOP_FUNCS", SRC.VOP_FUNCS[0]), ("SOP_FUNCS", SRC.SOP_FUNCS[0]), ("UNITS_RESERVED", SRC.UNITS_RESERVED[2])]
    return out


# ================================================================================================
def test_citations():
    for where, src in _all_srcs():
        assert isinstance(src, str) and src.startswith(AREAS), "%s: no document citation (%r)" % (where, src)
        assert "§" in src or "ISA" in src, "%s: citation without a section (%r)" % (where, src)


def _check_map(regs, space_bytes, what):
    seen = {}
    names = set()
    for r in regs:
        assert r.off % 4 == 0, "%s 0x%X: not word aligned" % (what, r.off)
        if r.name is not None:
            assert r.name not in names, "%s: name %s twice" % (what, r.name)
            names.add(r.name)
        if r.block is not None:
            assert r.block % 4 == 0 and r.block > 0, "%s %s: bad block size" % (what, r.name)
        assert r.stride % 4 == 0 and r.stride >= 4 * r.words, "%s %s: stride below the element size" % (what, r.name)
        for w in r.word_offsets():
            assert 0 <= w < space_bytes, "%s %s: word 0x%X outside the map" % (what, r.name, w)
            assert w not in seen, "%s: 0x%03X used by %s and %s" % (what, w, seen[w], r.name)
            seen[w] = r.name
        if r.reset is not None:
            assert 0 <= r.reset < 1 << 32, "%s %s: reset value" % (what, r.name)
            if r.layout:
                lay = M.layouts[r.layout]
                fields = 0
                for f in lay.fields:
                    fields |= f.mask
                assert r.reset & ~fields == 0, "%s %s: reset 0x%X sets undefined bits" % (what, r.name, r.reset)
        if r.layout is not None:
            assert r.layout in M.layouts, "%s %s: unknown layout %s" % (what, r.name, r.layout)
    return seen


def test_host_csr_map():
    seen = _check_map(M.host, 0x1000, "HOST_CSR")
    dbg_base, dbg_bytes = SRC.DBG_BLOCK
    assert seen.get(dbg_base) == "DBG" and seen.get(dbg_base + dbg_bytes - 4) == "DBG", "DBG block row"
    dseen = _check_map(M.dbg, 0x1000, "DBG_CSR")
    for w, name in dseen.items():
        assert dbg_base <= w < dbg_base + dbg_bytes, "DBG_CSR %s at 0x%03X outside the DBG block" % (name, w)
    host_names = {r.name for r in M.host if r.name}
    dbg_names = {r.name for r in M.dbg}
    assert not host_names & dbg_names, "names in both HOST_CSR and DBG_CSR: %s" % (host_names & dbg_names)
    offs = {r.name: r.off for r in M.host + M.dbg if r.name}
    for name, off in HOST_SPOT.items():
        assert offs.get(name) == off, "host %s at 0x%03X, expected 0x%03X" % (name, offs.get(name, -1), off)
    sreg = next(r for r in M.dbg if r.name == "SREG")
    assert (sreg.off, sreg.n, sreg.stride) == (0x9C0, 16, 4), "SREG0..15 at 0x9C0..0x9FC"
    perf_lo = next(r for r in M.host if r.name == "PERF_LO")
    assert perf_lo.n == SRC_CONST("PERF_SLOTS"), "PERF_LO count = PERF_SLOTS"
    lcq = next(r for r in M.host if r.name == "LCQ_REC")
    assert lcq.n == SRC_CONST("LCQ_DEPTH_MAX") and lcq.stride == SRC_CONST("LCQ_REC_BYTES"), "LCQ_REC shape"
    log = next(r for r in M.host if r.name == "LOG_RING")
    assert log.n == SRC_CONST("LOG_DEPTH_MAX"), "LOG_RING words"
    # every per-unit array has UNIT_SLOTS elements
    for r in M.host + M.dbg + M.cw:
        if r.name and (r.name.startswith(("Q_DONE", "Q_ISSUED", "Q_STATE", "Q_ERR_", "DBG_U_")) or
                       r.name in ("DONE_SEQ", "ISSUED_SEQ", "ERR_SEQ", "ERR_CODE", "ERR_INFO", "ERR_W0", "ERR_PIDX")):
            assert r.n == SRC_CONST("UNIT_SLOTS"), "%s: %d unit slots" % (r.name, r.n)


def SRC_CONST(name):
    return next(c[1] for c in SRC.CONSTANTS if c[0] == name)


def test_control_window():
    _check_map(M.cw, 0x1000, "CTL_WINDOW")
    ex_lo, ex_hi = SRC.CTL_EX_RANGE
    for r in M.cw:
        words = r.word_offsets()
        in_ex = all(ex_lo <= w < ex_hi for w in words)
        in_db = any(all(lo <= w < hi for w in words) for lo, hi in SRC.CTL_DB_RANGES)
        if r.owner == "EX":
            assert in_ex, "CTL_WINDOW %s (EX) outside 0x%03X..0x%03X" % (r.name, ex_lo, ex_hi - 4)
        elif r.owner == "DB":
            assert in_db, "CTL_WINDOW %s (DB) outside DB's reservations" % r.name
        else:
            assert not any(ex_lo <= w < ex_hi for w in words), "CTL_WINDOW %s (CC) in EX's range" % r.name
            assert not any(lo <= w < hi for lo, hi in SRC.CTL_DB_RANGES for w in words), "CTL_WINDOW %s (CC) in DB's range" % r.name
    offs = {r.name: r.off for r in M.cw if r.name}
    for name, off in CW_SPOT.items():
        assert offs.get(name) == off, "control window %s at 0x%03X, expected 0x%03X" % (name, offs.get(name, -1), off)


def test_adapter_map():
    _check_map(M.saxi, SRC.SAXI_BYTES, "SAXI")
    core = next(r for r in M.saxi if r.name == "CORE")
    assert core.off == 0x1000 and core.block == 0x1000, "core CSR window at 0x1000 + offset"


def test_rv_map():
    regions = sorted(SRC.RV_MAP, key=lambda r: r[1])
    for name, base, size, limit, fetch, acc, rsrc, desc in regions:
        assert size <= limit - base, "RV_MAP %s: size beyond its reserved range" % name
        assert base % (limit - base) == 0, "RV_MAP %s: base not aligned to its range" % name
    for a, b in zip(regions, regions[1:]):
        assert a[3] <= b[1], "RV_MAP %s overlaps %s" % (a[0], b[0])
    rv = {r[0]: r for r in SRC.RV_MAP}
    assert rv["ITCM"][2] == SRC_CONST("ITCM_BYTES") and rv["DTCM"][2] == SRC_CONST("DTCM_BYTES"), "TCM sizes"
    assert rv["CORE"][1] == rv["CTL"][1] + 0x1000, "core window directly after the control window"
    assert rv["CORE"][2] == 0x1000, "core window spans the 12-bit CSR space"


def _check_fields(fields, width, what, values=None):
    used = 0
    names = set()
    for f in fields:
        assert f.width >= 1 and f.lsb >= 0 and f.lsb + f.width <= width, "%s.%s does not fit %d bits" % (what, f.name, width)
        assert f.mask & used == 0, "%s.%s overlaps another field" % (what, f.name)
        assert f.name not in names, "%s: field %s twice" % (what, f.name)
        used |= f.mask
        names.add(f.name)
    for fname, vals in (values or {}).items():
        assert fname in names, "%s: values for unknown field %s" % (what, fname)
        fw = next(f for f in fields if f.name == fname).width
        assert len({v for v, *_ in vals}) == len(vals), "%s.%s: duplicate values" % (what, fname)
        assert len({n for _, n, *_ in vals}) == len(vals), "%s.%s: duplicate value names" % (what, fname)
        for v, vname, *_ in vals:
            assert 0 <= v < 1 << fw, "%s.%s = %s (%d) does not fit %d bits" % (what, fname, vname, v, fw)
    return used


def test_layouts():
    for lay in M.layouts.values():
        assert lay.width <= 32, "layout %s wider than 32 bits" % lay.name
        used = _check_fields(lay.fields, lay.width, lay.name, lay.values)
        for bit, rname, rdesc in lay.retired:
            assert not used >> bit & 1, "%s: retired bit %d reused" % (lay.name, bit)
    for lay in M.wide.values():
        _check_fields(lay.fields, lay.width, lay.name, lay.values)
        assert sum(f.width for f in lay.fields) == lay.width, "%s: fields do not cover the entry" % lay.name
    # STATUS: levels 0..15, sticky 16..27 (EX §5.3)
    st = M.layouts["STATUS"]
    sticky = sum(f.mask for f in st.fields if f.lsb >= 16)
    assert sticky == 0x0FFF0000, "STATUS sticky bits 16..27 (0x%08X)" % sticky
    # Per-unit fields in unit-id order
    for lname, fw in (("Q_DEPTH", SRC_CONST("Q_DEPTH_FIELD_W")), ("ISSUE_CREDIT", SRC_CONST("CREDIT_FIELD_W")), ("ISSUE_UNITS", 1)):
        lay = M.layouts[lname]
        for uid, uname, *_ in SRC.UNITS:
            f = lay.field(uname)
            assert (f.lsb, f.width) == (fw * uid, fw), "%s.%s at [%d+:%d]" % (lname, uname, f.lsb, f.width)
    for fname, lsb in (("ERR", 8), ("NONEMPTY", 16)):
        f = M.layouts["QSTATUS"].field(fname)
        assert (f.lsb, f.width) == (lsb, SRC_CONST("NUM_UNITS")), "QSTATUS.%s: one bit per unit" % fname
    # CAPS: retired bits 2 and 4 (EX §7.5); every opcode CAPS gate names a CAPS field
    caps = M.layouts["CAPS"]
    assert {b for b, *_ in caps.retired} == {2, 4}, "CAPS retired bits"
    assert caps.field("ATT").lsb == 5 and caps.field("DQROW").lsb == 6 and caps.field("VMASK").lsb == 7
    # Hold-source bit order shared by STATUS, QSTATUS and DBG_U_STATE
    hs = M.layouts["HOLD_SRC"]
    assert [f.name for f in sorted(hs.fields, key=lambda f: f.lsb)] == ["HOST", "RV", "HALT", "DB"]
    assert M.layouts["QSTATUS"].field("HOLD_HOST").lsb == 2 and M.layouts["QSTATUS"].field("HOLD_HALT").lsb == 4
    # Log header (DB §6.1)
    lh = M.layouts["LOG_HDR"]
    assert [(f.name, f.lsb, f.width) for f in lh.fields] == [
        ("N", 0, 8), ("LEVEL", 8, 4), ("SOURCE", 12, 4), ("TYPE", 16, 3), ("TS", 19, 1), ("LSEQ", 20, 8), ("SYNC", 28, 4)]


def test_encodings():
    for e in M.encs.values():
        used = [0] * 8
        used[0] = 0xFF
        names = set()
        for f in e.fields:
            assert 0 <= f.word < 8, "%s.%s: word %d" % (e.name, f.name, f.word)
            assert f.lsb + f.width <= 32, "%s.%s beyond its word" % (e.name, f.name)
            assert used[f.word] & f.mask == 0, "%s.%s overlaps the opcode or another field" % (e.name, f.name)
            assert f.name not in names, "%s: field %s twice" % (e.name, f.name)
            used[f.word] |= f.mask
            names.add(f.name)
            if f.spec.startswith("x16"):
                assert f.word == 0 and f.lsb >= 16, "%s.%s: x16 field outside w0[31:16]" % (e.name, f.name)
        for i in range(8):
            assert e.rsv[i] == (~used[i]) & 0xFFFFFFFF, "%s: RSV_W%d" % (e.name, i)
        for fname, vals in e.all_values.items():
            assert fname in names, "%s: values for unknown field %s" % (e.name, fname)
            fw = next(f for f in e.fields if f.name == fname).width
            for v, vname, *_ in vals:
                assert 0 <= v < 1 << fw, "%s.%s = %s does not fit" % (e.name, fname, vname)
        o = M.ops[e.opcode]
        assert o["legal"] and o["mnem"] == e.name, "%s: not a legal opcode" % e.name
        assert o["owner"] == e.owner, "%s: encoding owner %s, opcode owner %s" % (e.name, e.owner, o["owner"])
    legal = {o["mnem"] for o in M.ops if o["legal"]}
    assert legal == set(M.encs), "legal opcodes without an encoding: %s" % (legal - set(M.encs))
    # v1 encodings bit-identical with ra..rd = 0 (EX §7.1): v1 reserved masks plus the old register fields
    assert M.encs["LD"].rsv[0] == 0xFFFFF000 and M.encs["ST"].rsv[0] == 0xFFFFF000
    assert M.encs["QACT"].rsv[0] == 0xFFFFF000 and M.encs["MVM"].rsv[0] == 0xFFFFE000
    assert M.encs["MVM"].rsv[5] == 0xFFF00000 and M.encs["MVM"].rsv[7] == 0xFFFFFFFF
    assert M.encs["NOP"].rsv == [0xFFFFFF00] + [0xFFFFFFFF] * 7 == M.encs["FENCE"].rsv
    assert M.encs["LD"].words_mask == 0x0F and M.encs["QACT"].words_mask == 0x3F and M.encs["MVM"].words_mask == 0x7F
    # VU §5.1: line addresses and m are w[23:0]; rows / strides / head counts / T are 16 bits
    for op in ("VOP", "RED", "ROPE", "TOPK", "DQROW", "VMASK"):
        for f in M.encs[op].fields:
            if f.name in ("D", "A", "B", "C", "M", "SRC", "POS", "TABLE", "X"):
                assert (f.lsb, f.width) == (0, 24), "%s.%s: 24-bit line / element field" % (op, f.name)
            if f.name in ("ROWS", "D_STRIDE", "A_STRIDE", "B_STRIDE", "SRC_STRIDE", "X_STRIDE", "N_HEADS", "T"):
                assert f.width == 16, "%s.%s: 16-bit field" % (op, f.name)
    # VOP functions dense 0..24, SOP subs dense 0..14
    codes = [c for c, *_ in M.vop_funcs]
    assert codes == list(range(len(codes))) and len(codes) == 25, "VOP function codes 0..0x18"
    assert all(set(rd) <= set("abc") and rd and rt in "FH" for c, n, rd, rt, res in M.vop_funcs)
    assert [n for c, n, *_ in M.vop_funcs][:10] == ["ADD", "SUB", "MUL", "MAX", "MULC", "ADDC", "SILU", "SWIGLU",
                                                    "EXP", "COPY"], "VOP codes 0..9 keep the spec numbers"
    subs = [c for c, *_ in M.sop_funcs]
    assert subs == list(range(15)), "SOP subs 0..14"
    sop_fields = {f.name for f in M.encs["SOP"].fields} - {"SUB"}
    assert all(set(u) <= sop_fields for c, n, u, eff in M.sop_funcs)
    # AT shape fields shared by ATTQK and ATTPV (AT §5)
    for op in ("ATTQK", "ATTPV"):
        f = {x.name: (x.word, x.lsb, x.width) for x in M.encs[op].fields}
        assert f["E64"] == (4, 0, 9) and f["DCODE"] == (4, 9, 2) and f["TQ_M1"] == (4, 11, 3) and f["GS_M1"] == (4, 14, 3)
        assert f["G_M1"] == (0, 16, 6) and f["HG0"] == (0, 22, 6), "%s: x16 G-1 / hg0" % op


def test_opcodes():
    assert all(o is not None for o in M.ops), "opcodes not covered: %s" % [i for i, o in enumerate(M.ops) if o is None]
    mnems = [r[2] for r in SRC.OPCODES if r[2]]
    assert len(mnems) == len(set(mnems)), "duplicate mnemonics"
    for code, (mnem, unit) in ASSIGNED.items():
        o = M.ops[code]
        assert o["mnem"] == mnem and o["legal"], "0x%02X: expected %s" % (code, mnem)
        assert M.op_unit(code) == UNIT_IDS[unit], "%s: unit %d, expected %s" % (mnem, M.op_unit(code), unit)
    legal = {o["code"] for o in M.ops if o["legal"]}
    assert legal == set(ASSIGNED), "legal opcodes differ from EX §7.2: %s" % (legal ^ set(ASSIGNED))
    for code, mnem in RETIRED_V1.items():
        o = M.ops[code]
        assert o["mnem"] == mnem and o["status"] in ("reserved", "retired") and not o["legal"], \
            "retired v1 opcode 0x%02X (%s) reused" % (code, mnem)
        assert M.op_unit(code) == UNIT_IDS["SYNC"], "0x%02X must complete ILLEGAL_OP on SYNC" % code
    for owner, codes in BLOCKS.items():
        for code in codes:
            assert M.ops[code]["owner"] == owner, "0x%02X: owner %s, V2-EX-15 says %s" % (code, M.ops[code]["owner"], owner)
    for o in M.ops:
        if not o["legal"]:
            assert o["caps"] is None and M.op_unit(o["code"]) == UNIT_IDS["SYNC"], "0x%02X" % o["code"]
        elif o["caps"]:
            assert o["caps"] in {f.name for f in M.layouts["CAPS"].fields}, "0x%02X: CAPS.%s" % (o["code"], o["caps"])
    # CAPS gates (EX §7.5, VU §5.10, AT §5.10)
    gate = {o["mnem"]: o["caps"] for o in M.ops if o["legal"]}
    assert gate["TOPK"] == "TOPK" and gate["DQROW"] == "DQROW" and gate["VMASK"] == "VMASK"
    assert all(gate[x] == "ATT" for x in ("KVST", "ATTQK", "ATTPV", "ASMAX", "AMASK", "AMERGE", "ANORM"))
    assert all(gate[x] == "VU" for x in ("VOP", "RED", "ROPE", "SOP"))
    assert all(gate[x] is None for x in ("NOP", "FENCE", "LD", "ST", "QACT", "MVM"))


def test_units():
    ids = [u[0] for u in SRC.UNITS]
    assert ids == list(range(len(ids))) == list(range(SRC_CONST("NUM_UNITS"))), "unit ids not dense"
    for uid, name, *_ in SRC.UNITS:
        assert UNIT_IDS[name] == uid, "unit %s id %d" % (name, uid)
    for uid, name, *_ in SRC.UNITS_SPECIAL:
        assert UNIT_IDS[name] == uid
    lo, hi, _ = SRC.UNITS_RESERVED
    assert (lo, hi) == (len(ids), 13), "reserved unit ids 6..13"
    assert [u[2] for u in SRC.UNITS] == Q_DEPTH_U200, "U200 M6 queue depths"
    for uid, name, qd, pd, acc, *_ in SRC.UNITS:
        assert 0 <= qd < 1 << SRC_CONST("Q_DEPTH_FIELD_W") and 0 < pd < 1 << SRC_CONST("Q_DEPTH_FIELD_W")
        assert acc == (2 if name in ("MVM", "VU") else 1), "I3 limit of %s" % name
    assert next(u[4] for u in SRC.UNITS if u[1] == "VU") == SRC_CONST("VU_ACCEPT_AHEAD")
    expect = sum(d << 5 * u for u, d in enumerate(Q_DEPTH_U200))
    assert M.q_depth_value() == expect, "Q_DEPTH register value"
    assert SRC_CONST("UNIT_SLOTS") >= len(ids)


def test_errors():
    codes = [e[0] for e in SRC.ERRORS]
    assert codes == list(range(16)), "error codes 0..15, once each"
    assert max(codes) < 1 << SRC_CONST("ERR_W")
    named = {e[1]: e[0] for e in SRC.ERRORS if e[1] and e[2] != "retired"}
    assert named == ERR_CODES, "error codes differ from EX §8.1"
    assert {e[0] for e in SRC.ERRORS if e[2] == "retired"} == {9, 10}, "9, 10 retired"
    assert SRC.ERR_CHECK_ORDER[1] == CHECK_ORDER, "check order differs from EX §8.1"
    for name in CHECK_ORDER:
        assert next(e for e in SRC.ERRORS if e[1] == name)[2] == "active"


def test_status_precedence():
    assert SRC.STATUS_PRECEDENCE[1] == STATUS_ORDER, "STATUS precedence differs from EX §4.4"
    vals = {n: v for v, n, _ in M.layouts["LCQ_R0"].values["STATUS"]}
    assert set(vals) == set(STATUS_ORDER) and vals["TRAP"] == 4 and vals["OK"] == 0 and vals["STOPPED"] == 7


def test_perf():
    ext_top = max(s[1] for s, *_ in SRC.PERF if isinstance(s, tuple) and s[0] == "x")
    db = sorted(s for s, *_ in SRC.PERF if isinstance(s, int) and s >= 32)
    assert db == [61, 62, 63], "DB slots 61..63"
    reserved = [s for s, name, *_ in SRC.PERF if name is None]
    assert reserved == [4], "slot 4 reserved (was WAITM)"
    xoff = {name: s[1] for s, name, *_ in SRC.PERF if isinstance(s, tuple) and s[0] == "x"}
    for name, k in PERF_EXT.items():
        assert xoff[name] == k, "PERF %s at X + %d, expected X + %d" % (name, xoff[name], k)
    assert sorted(xoff.values()) == list(range(len(xoff))), "extension slots dense from X"
    for C in range(1, SRC_CONST("NUM_CH_MAX") + 1):
        slots = M.perf_slots(C)
        vals = list(slots.values())
        assert len(vals) == len(set(vals)), "C=%d: PERF slots collide" % C
        assert all(0 <= v < SRC_CONST("PERF_SLOTS") for v in vals), "C=%d: slot beyond 63" % C
        x = 5 + 2 * C
        assert x + ext_top < 61, "C=%d: extensions reach DB's slots" % C
        assert 4 not in vals
        for c in range(C):
            assert slots["CH_RD_BEATS[%d]" % c] == 5 + c and slots["CH_WAIT[%d]" % c] == 5 + C + c
    assert M.perf_slots(4)["DMA_CYCLES"] == 13 and M.perf_slots(16)["SPM_WR_WAIT"] == 58  # EX §6 examples


def test_word_tables():
    for name, t in SRC.WORD_TABLES.items():
        seen = {}
        for row in t["rows"]:
            w, rname = row[0], row[1]
            n = row[3] if len(row) > 3 else 1
            assert 0 <= w and w + n <= t["words"], "%s.%s beyond %d words" % (name, rname, t["words"])
            for i in range(w, w + n):
                assert i not in seen, "%s: word %d used by %s and %s" % (name, i, seen[i], rname)
                seen[i] = rname
    assert SRC.WORD_TABLES["CRASH"]["words"] * 4 == 192 and SRC.WORD_TABLES["DBG_INFO"]["words"] * 4 == 64
    assert SRC.WORD_TABLES["DESC"]["words"] * 4 == SRC_CONST("LCQ_REC_BYTES")


def _names(text, pattern):
    return re.findall(pattern, text, flags=re.M)


def _dups(names):
    seen, dup = set(), set()
    for n in names:
        (dup if n in seen else seen).add(n)
    return sorted(dup)


def test_generated_names_unique():
    outs = {p.name: t for p, t in GEN.outputs(M).items()}
    c = _names(outs["tsr_tables.h"], r"^#define (TSR_\w+)")
    assert not _dups(c), "C: duplicate macros %s" % _dups(c)
    sv = _names(outs["tsr_tables_pkg.sv"], r"^\s*localparam [^=]*?\b(TSR_\w+)\s*(?:\[\d+\]\s*)?=")
    sv += _names(outs["tsr_tables_pkg.sv"], r"^\s*function automatic [^(]*?\b(tsr_\w+)\(")
    assert not _dups(sv), "SV: duplicate names %s" % _dups(sv)
    cpp_text = outs["mmio.hpp"] + outs["vu.hpp"] + outs["at.hpp"]
    cpp = _names(cpp_text, r"^constexpr \w+ (\w+)(?:\[\d+\])? =")
    cpp += _names(cpp_text, r"^constexpr [\w ]+? (\w+)\(")
    cpp += _names(cpp_text, r"^struct (\w+) \{")
    assert not _dups(cpp), "C++: duplicate names %s" % _dups(cpp)
    py = _names(outs["tables.py"], r"^([A-Za-z_]\w*)\s*=") + _names(outs["tables.py"], r"^def (\w+)\(")
    assert not _dups(py), "Python: duplicate names %s" % _dups(py)
    for name, text in outs.items():
        assert GEN.BANNER in text.split("\n", 1)[0], "%s: no do-not-edit banner on line 1" % name
    for name in ("mmio.hpp", "vu.hpp", "at.hpp"):
        incs = _names(outs[name], r'^#include [<"]([^>"]+)[>"]')
        assert set(incs) <= {"stdint.h", "mmio.hpp"}, "%s includes %s (freestanding: <stdint.h> only)" % (name, incs)


def test_python_module():
    py = _load("tsr_tables_tb", GEN.OUT_PY)
    w = py.encode("ATTQK", CAUSAL=1, G_M1=7, HG0=3, E64=2, DCODE=1, U0=5, N_GROUPS=9, MDST=12, MAXOUT=1)
    assert w[0] == 0x21 | 1 << 12 | 1 << 14 | 7 << 16 | 3 << 22 and w[5] == 5 | 9 << 16 and w[7] == 12
    assert py.decode("ATTQK", w)["HG0"] == 3 and py.reserved_bits("ATTQK", w) == [0] * 8
    assert py.OP_UNIT[0x22] == py.UNIT_MVM and py.OP_UNIT[0x09] == py.UNIT_SYNC and py.OP_LEGAL[0x09] == 0
    assert py.perf_slots(4)["VU_LINES"] == 28 and py.CSR_SREG(1) == 0x9C4 and py.CW_ERR_PIDX(2) == 0x188
    assert py.err_check_rank(py.ERR_OPERAND) == 5 and py.VOP_FUNCS[0x17][0] == "GELU16"
    try:
        py.encode("VOP", ROWS=1 << 16)
    except ValueError:
        pass
    else:
        raise AssertionError("encode accepted a value wider than its field")


def test_generated_files_current():
    stale = GEN.check()
    assert not stale, "stale generated files (run %s): %s" % (GEN.GENERATOR, [str(p.relative_to(REPO)) for p in stale])


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (name, e))
    print("test_tables.py: %d of %d checks passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

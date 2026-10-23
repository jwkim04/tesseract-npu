"""Read Tesseract trace directories (spec 트레이스 형식; sw/sim/include/tsr_trace.h).

    <dir>/meta.json                   config, cmd_addr_beat, cmd_count, args, expected final state,
                                      init / expect region lists, perf (informational)
    <dir>/cmd.bin                     cmd_count * 32 bytes
    <dir>/init/ch<c>_<offset>.bin     DRAM content before execution
    <dir>/expect/ch<c>_<offset>.bin   DRAM content after execution of the ST-written ranges

Replay semantics (tsr_trace.h): DRAM outside the init regions is zero; load cmd.bin at ch0 byte
cmd_addr_beat*64 (clipped at the channel end; not loaded when the command buffer starts beyond the
channel), then every init region (init wins). Run with CMD_ADDR/CMD_COUNT/ARG0..7, compare STATUS,
ERR_CODE, ERR_PC, ERR_INFO (if present) and every expect region; the set of written beats must
equal the union of the expect regions; every beat read must lie in an init region, in the command
buffer (whole beats), or have been written earlier by the run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import isa


@dataclass
class Region:
    ch: int
    offset: int          # bytes, multiple of 64
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def beats(self):
        b0 = self.offset // 64
        return range(b0, b0 + self.size // 64)


@dataclass
class Trace:
    path: Path
    config: dict
    cmd_addr_beat: int
    cmd_count: int
    args: list
    status: int
    err_code: int
    err_pc: int
    err_info: int | None          # None: not recorded (not compared)
    init: list = field(default_factory=list)
    expect: list = field(default_factory=list)
    cmd: bytes = b""
    perf: dict | None = None

    # ---- derived ----
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def num_ch(self) -> int:
        return self.config["num_ch"]

    @property
    def dram_limit(self) -> int:
        return isa.dram_limit(self.config["ch_bytes"], self.config["addr_beat_w"])

    @property
    def is_error(self) -> bool:
        return self.err_code != 0

    def insn(self, pc: int) -> list:
        return isa.insn_words(self.cmd[32 * pc:32 * pc + 32])

    def opcodes(self) -> set:
        return {self.insn(pc)[0] & 0xFF for pc in range(self.cmd_count)}

    def disasm(self) -> str:
        return "\n".join("%6d  %s" % (pc, isa.disasm(self.insn(pc))) for pc in range(self.cmd_count))

    def core_parameters(self) -> dict:
        """tsr_core elaboration parameters for this trace's config (cocotb runner `parameters`)."""
        c = self.config
        return {
            "NUM_CH": c["num_ch"], "BEAT_BITS": c["beat_bits"], "LANES": c["lanes"],
            "T_MAX": c["t_max"], "K_MAX": c["k_max"], "SPM_BYTES": c["spm_bytes"], "VL": c["vl"],
            "NIREG": c["nireg"], "NSREG": c["nsreg"], "LOOP_DEPTH": c["loop_depth"],
            "RD_OUTSTANDING": c["rd_outstanding"], "CH_BYTES": "64'd%d" % c["ch_bytes"],
            "ADDR_BEAT_W": c["addr_beat_w"],
        }

    def config_key(self) -> tuple:
        """Hashable config identity: traces with equal keys can share one RTL build."""
        return tuple(sorted(self.config.items()))

    def cmd_region(self) -> Region | None:
        """cmd.bin as it is loaded into ch0 (clipped at the channel end), or None."""
        ch_bytes = self.config["ch_bytes"]
        lo = self.cmd_addr_beat * 64
        if not self.cmd or self.cmd_count == 0 or self.cmd_addr_beat >= ch_bytes // 64:
            return None
        n = min(len(self.cmd), ch_bytes - lo)
        return Region(0, lo, self.cmd[:n])

    def initial_dram(self) -> list:
        """Regions to load before the run, in load order (cmd first, init regions win)."""
        out = []
        c = self.cmd_region()
        if c is not None:
            out.append(c)
        out.extend(self.init)
        return out

    def allowed_read_beats(self) -> list:
        """Per channel: set of beats a run may read before writing them (init + command beats)."""
        allowed = [set() for _ in range(self.num_ch)]
        for r in self.init:
            allowed[r.ch].update(r.beats())
        c = self.cmd_region()
        if c is not None:
            b0 = c.offset // 64
            allowed[0].update(range(b0, b0 + (c.size + 63) // 64))
        return allowed

    def executed_count(self) -> int:
        """Instructions that executed successfully: ERR_PC for an error ending, else up to END."""
        if self.err_code:
            return min(self.err_pc, self.cmd_count)
        for pc in range(self.cmd_count):
            if self.insn(pc)[0] & 0xFF == isa.OP_END:
                return pc + 1
        return self.cmd_count

    def expected_read_beats(self) -> list:
        """Per channel: {source: beats} the run must read through the memory port, derived from
        the program alone (lengths, channels, N, K and fmt are immediates): source DMA = LD
        n_lines, source WS = rows_c * ROW_BEATS of every executed MVM (spec 구현 명확화 "성능
        카운터"; docs/rtl-m3.md §4). Instruction fetch beats are implementation-defined (not here)."""
        out = [{isa.SRC_DMA: 0, isa.SRC_WS: 0} for _ in range(self.num_ch)]
        for pc in range(self.executed_count()):
            w = self.insn(pc)
            op = w[0] & 0xFF
            if op == isa.OP_LD:
                out[(w[0] >> 8) & 0xF][isa.SRC_DMA] += w[3]
            elif op == isa.OP_MVM:
                fmt, N, K = (w[0] >> 8) & 0xF, w[4], w[5] & 0xFFFF
                rb = isa.row_beats(fmt, K)
                R = -(-N // self.num_ch)
                for c in range(self.num_ch):
                    out[c][isa.SRC_WS] += max(0, min(R, N - c * R)) * rb
        return out

    def expect_beats(self) -> list:
        """Per channel: set of beats the run must write (union of the expect regions)."""
        out = [set() for _ in range(self.num_ch)]
        for r in self.expect:
            out[r.ch].update(r.beats())
        return out


def _region_list(base: Path, items) -> list:
    out = []
    for it in items:
        data = (base / it["file"]).read_bytes()
        if len(data) != it["size"]:
            raise ValueError("%s: size %d, meta.json says %d" % (it["file"], len(data), it["size"]))
        if it["offset"] % 64 or it["size"] % 64:
            raise ValueError("%s: region not 64-byte aligned" % it["file"])
        out.append(Region(int(it["ch"]), int(it["offset"]), data))
    return out


def load(path) -> Trace:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != "tesseract-trace" or meta.get("version") != 1:
        raise ValueError("%s: not a tesseract-trace v1" % path)
    cmd_count = int(meta["cmd_count"])
    cmd_file = path / "cmd.bin"
    cmd = cmd_file.read_bytes() if cmd_file.exists() else b""
    if cmd and len(cmd) != 32 * cmd_count:
        raise ValueError("%s: cmd.bin has %d bytes, expected %d" % (path, len(cmd), 32 * cmd_count))
    tr = Trace(
        path=path, config=dict(meta["config"]), cmd_addr_beat=int(meta["cmd_addr_beat"]),
        cmd_count=cmd_count, args=[int(a) for a in meta["args"]], status=int(meta["status"]),
        err_code=int(meta["err_code"]), err_pc=int(meta["err_pc"]),
        err_info=int(meta["err_info"]) if "err_info" in meta else None,
        init=_region_list(path, meta.get("init", [])),
        expect=_region_list(path, meta.get("expect", [])),
        cmd=cmd, perf=meta.get("perf"))
    if not tr.cmd:
        # tsr_trace.h: cmd.bin missing -> derive from the init regions (zero elsewhere).
        buf = bytearray(32 * cmd_count)
        lo = tr.cmd_addr_beat * 64
        for r in tr.init:
            if r.ch != 0:
                continue
            a, b = max(lo, r.offset), min(lo + len(buf), r.offset + r.size)
            if a < b:
                buf[a - lo:b - lo] = r.data[a - r.offset:b - r.offset]
        tr.cmd = bytes(buf)
    return tr


def find(root) -> list:
    """Trace directories under root (root itself if it holds meta.json), sorted by path."""
    root = Path(root)
    if (root / "meta.json").is_file():
        return [root]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "meta.json" in filenames:
            out.append(Path(dirpath))
            dirnames[:] = []
    return sorted(out)


def load_all(root) -> list:
    return [load(p) for p in find(root)]

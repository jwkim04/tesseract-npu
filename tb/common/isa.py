"""Python mirror of rtl/core/tsr_pkg.sv (and sw/sim/include/tsr_isa.h) for the testbenches.

Constants (opcodes, CSR offsets, error codes, memory-port encoding), instruction encoders that
match tsr_isa.h's tsr_enc_*, and the internal-interface struct layouts of tsr_pkg.sv with
pack/unpack helpers (packed SV structs: the first member is the most significant).
test_common.py checks the struct widths and field order against the RTL.
"""
from __future__ import annotations

import struct as _struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------------------------
BEAT_BYTES = 64
LINE_BYTES = 64
LINE_BITS = 512
INSN_BYTES = 32
MAX_BURST_BEATS = 64
BURST_ALIGN = 64          # bursts stay inside one 64-beat (4 KiB) block
T_MAX_ARCH = 8
NUM_CH_MAX = 16
ID_MAGIC = 0x54535231
ISA_MAJOR, ISA_MINOR = 1, 0
CANON_NAN = 0x7FC00000

MIB = 1024 * 1024
REGION_CMD_OFF, REGION_CMD_SIZE = 0, 64 * MIB
REGION_IO_OFF, REGION_IO_SIZE = 64 * MIB, 256 * MIB
REGION_AUX_OFF, REGION_AUX_SIZE = 320 * MIB, 192 * MIB
REGION_WEIGHT_OFF = 512 * MIB

# U200 v1 defaults (tsr_pkg TSR_DEF_*), keyed like tsr_trace.h's meta.json "config".
U200_V1 = dict(num_ch=4, beat_bits=512, lanes=128, t_max=1, k_max=32768, spm_bytes=4 * MIB,
               vl=16, nireg=16, nsreg=16, loop_depth=4, rd_outstanding=16, ch_bytes=16 << 30,
               addr_beat_w=28, caps=0)

# ---------------------------------------------------------------------------------------------
# Memory port encoding
# ---------------------------------------------------------------------------------------------
LEN_W = 7
SRC_ID_W = 2
MEM_ID_W = 4
SRC_FETCH, SRC_DMA, SRC_WS = 0, 1, 2

SPM_CL_DMA, SPM_CL_QACT, SPM_CL_MVM = 0, 1, 2
SPM_NCL = 3

# ---------------------------------------------------------------------------------------------
# CSR map
# ---------------------------------------------------------------------------------------------
CSR_AW = 12
CSR_ID = 0x000
CSR_VERSION = 0x004
CSR_CFG0 = 0x008
CSR_CFG1 = 0x00C
CSR_CFG2 = 0x010
CSR_CAPS = 0x014
CSR_CTRL = 0x020
CSR_STATUS = 0x024
CSR_CMD_ADDR_LO = 0x028
CSR_CMD_ADDR_HI = 0x02C
CSR_CMD_COUNT = 0x030
CSR_ERR_PC = 0x040
CSR_ERR_CODE = 0x044
CSR_ERR_INFO = 0x048
CSR_PC = 0x050
CSR_ARG0 = 0x060
CSR_PERF_BASE = 0x100

CTRL_START, CTRL_SOFT_RESET, CTRL_ABORT = 1, 2, 4
STATUS_BUSY, STATUS_DONE, STATUS_ERR, STATUS_WAITING = 1, 2, 4, 8
CAPS_VU, CAPS_RLD_LOOP, CAPS_TOPK, CAPS_WAITM_JUMP = 0x1, 0x4, 0x8, 0x10

PERF_CYCLES, PERF_RETIRED, PERF_MVM_CYCLES, PERF_VU_CYCLES, PERF_WAITM_CYCLES = 0, 1, 2, 3, 4
PERF_CH_RD_BEATS = 5


def perf_ch_rd_beats(ch: int) -> int:
    return PERF_CH_RD_BEATS + ch


def perf_ch_wait(num_ch: int, ch: int) -> int:
    return PERF_CH_RD_BEATS + num_ch + ch


def perf_count(num_ch: int) -> int:
    return PERF_CH_RD_BEATS + 2 * num_ch


def perf_csr(idx: int) -> int:
    """Byte offset of the LO half of 64-bit counter idx (HI at +4)."""
    return CSR_PERF_BASE + 8 * idx


# ---------------------------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------------------------
ERR_NONE, ERR_ILLEGAL_OP, ERR_BAD_FMT, ERR_QA_MISMATCH = 0, 1, 2, 3
ERR_SPM_RANGE, ERR_DRAM_RANGE, ERR_K_RANGE, ERR_T_RANGE = 4, 5, 6, 7
ERR_ABORTED, ERR_LOOP_NEST, ERR_WAIT_TIMEOUT = 8, 9, 10
ERR_NAMES = {0: "NONE", 1: "ILLEGAL_OP", 2: "BAD_FMT", 3: "QA_MISMATCH", 4: "SPM_RANGE",
             5: "DRAM_RANGE", 6: "K_RANGE", 7: "T_RANGE", 8: "ABORTED", 9: "LOOP_NEST",
             10: "WAIT_TIMEOUT"}

# ---------------------------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------------------------
OP_NOP, OP_END, OP_LI, OP_ADDI, OP_MULI, OP_RLD, OP_LOOP, OP_WAITM, OP_JUMP = (
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)
OP_LD, OP_ST, OP_DQROW, OP_QACT, OP_MVM = 0x10, 0x11, 0x12, 0x18, 0x20
OP_VOP, OP_RED, OP_ROPE, OP_TOPK, OP_SOP, OP_FENCE = 0x30, 0x31, 0x32, 0x33, 0x40, 0x7F
OP_NAMES = {OP_NOP: "NOP", OP_END: "END", OP_LI: "LI", OP_ADDI: "ADDI", OP_MULI: "MULI",
            OP_RLD: "RLD", OP_LOOP: "LOOP", OP_WAITM: "WAITM", OP_JUMP: "JUMP", OP_LD: "LD",
            OP_ST: "ST", OP_DQROW: "DQROW", OP_QACT: "QACT", OP_MVM: "MVM", OP_VOP: "VOP",
            OP_RED: "RED", OP_ROPE: "ROPE", OP_TOPK: "TOPK", OP_SOP: "SOP", OP_FENCE: "FENCE"}
M2_OPS = (OP_NOP, OP_END, OP_LI, OP_ADDI, OP_LD, OP_ST, OP_QACT, OP_FENCE)
M3_OPS = M2_OPS + (OP_MVM,)

FMT_Q8_0, FMT_Q4_K = 0, 1
QA_A32, QA_A256 = 0, 1
MVM_FLAG_ACC = 1

# Reserved w0 bits / first unused word per instruction (tsr_pkg TSR_W0_RSV_*, TSR_WFREE_*).
W0_RSV = {OP_NOP: 0xFFFFFF00, OP_END: 0xFFFFFF00, OP_FENCE: 0xFFFFFF00, OP_LI: 0x0FFFFF00,
          OP_ADDI: 0x0FF0FF00, OP_LD: 0xFF00F000, OP_ST: 0xFF00F000, OP_QACT: 0xFF00F000,
          OP_MVM: 0xF000E000}
WFREE = {OP_NOP: 1, OP_END: 1, OP_FENCE: 1, OP_LI: 2, OP_ADDI: 2, OP_LD: 4, OP_ST: 4,
         OP_QACT: 6, OP_MVM: 7}


def w0(opcode, sub=0, flags=0, ra=0, rb=0, rc=0, rd=0) -> int:
    return ((opcode & 0xFF) | ((sub & 0xF) << 8) | ((flags & 0xF) << 12) | ((ra & 0xF) << 16)
            | ((rb & 0xF) << 20) | ((rc & 0xF) << 24) | ((rd & 0xF) << 28))


def w0_fields(w: int) -> dict:
    return dict(opcode=w & 0xFF, sub=(w >> 8) & 0xF, flags=(w >> 12) & 0xF, ra=(w >> 16) & 0xF,
                rb=(w >> 20) & 0xF, rc=(w >> 24) & 0xF, rd=(w >> 28) & 0xF)


def insn_bytes(words) -> bytes:
    """8 words -> the 32-byte little-endian instruction."""
    words = list(words) + [0] * (8 - len(words))
    return _struct.pack("<8I", *[x & 0xFFFFFFFF for x in words])


def insn_words(b: bytes) -> list:
    return list(_struct.unpack("<8I", bytes(b[:32])))


def enc_nop():
    return [w0(OP_NOP)] + [0] * 7


def enc_end():
    return [w0(OP_END)] + [0] * 7


def enc_fence():
    return [w0(OP_FENCE)] + [0] * 7


def enc_li(rd, imm):
    return [w0(OP_LI, rd=rd), imm & 0xFFFFFFFF] + [0] * 6


def enc_addi(rd, ra, imm):
    return [w0(OP_ADDI, ra=ra, rd=rd), imm & 0xFFFFFFFF] + [0] * 6


def enc_ld(ch, spm_line, dram_beat, n_lines, ra=0, rb=0):
    return [w0(OP_LD, sub=ch, ra=ra, rb=rb), spm_line & 0xFFFFFFFF, dram_beat & 0xFFFFFFFF,
            n_lines & 0xFFFFFFFF] + [0] * 4


def enc_st(ch, spm_line, dram_beat, n_lines, ra=0, rb=0):
    return [w0(OP_ST, sub=ch, ra=ra, rb=rb), spm_line & 0xFFFFFFFF, dram_beat & 0xFFFFFFFF,
            n_lines & 0xFFFFFFFF] + [0] * 4


def enc_qact(kind, dst_line, src_line, K, T, src_stride=0, ra=0, rb=0):
    return [w0(OP_QACT, sub=kind, ra=ra, rb=rb), dst_line & 0xFFFFFFFF, src_line & 0xFFFFFFFF,
            K & 0xFFFFFFFF, T & 0xFFFFFFFF, src_stride & 0xFFFFFFFF] + [0] * 2


def enc_mvm(fmt, dst_line, x_line, w_off, N, K, T, dst_stride=0, acc=0, ra=0, rb=0, rc=0):
    """MVM: A1 = dst (w1), A2 = x (w2), A3 = W_OFF (w3), w4 N, w5 K[15:0] | T[19:16], w6 dst_stride,
    flags bit0 = ACC (spec ISA; tsr_sim.c ex_mvm)."""
    return [w0(OP_MVM, sub=fmt, flags=MVM_FLAG_ACC if acc else 0, ra=ra, rb=rb, rc=rc),
            dst_line & 0xFFFFFFFF, x_line & 0xFFFFFFFF, w_off & 0xFFFFFFFF, N & 0xFFFFFFFF,
            (K & 0xFFFF) | ((T & 0xF) << 16), dst_stride & 0xFFFFFFFF, 0]


def disasm(words) -> str:
    w = list(words)
    f = w0_fields(w[0])
    name = OP_NAMES.get(f["opcode"], "op%02x" % f["opcode"])
    extra = " ".join("%08x" % x for x in w[1:])
    return "%-5s sub=%d fl=%d ra=%d rb=%d rc=%d rd=%d  %s" % (
        name, f["sub"], f["flags"], f["ra"], f["rb"], f["rc"], f["rd"], extra)


# ---------------------------------------------------------------------------------------------
# QA layout helpers (pure Python; tsr_ref.py exposes the C versions for cross-checks)
# ---------------------------------------------------------------------------------------------
def qa_block_size(kind):
    return 256 if kind == QA_A256 else 32


def qa_qs_lines(K):
    return (K + 63) // 64


def qa_d_lines(kind, K):
    return (4 * (K // qa_block_size(kind)) + 63) // 64


def qa_bs_lines(kind, K):
    return (2 * (K // 16) + 63) // 64 if kind == QA_A256 else 0


def qa_lines(kind, K):
    return qa_qs_lines(K) + qa_d_lines(kind, K) + qa_bs_lines(kind, K)


def row_beats(fmt, K):
    """ROW_BEATS(fmt, K) (spec 행 크기; tsr_ref.c tsr_row_beats): full chunks of 1 meta + 16 (Q8_0)
    or 1 + 8 (Q4_K) quant beats, then a partial chunk of 1 + ceil(rem / 2) or 1 + 2 rem beats."""
    if fmt == FMT_Q4_K:
        nb = K // 256
        return (nb // 4) * 9 + ((1 + 2 * (nb % 4)) if nb % 4 else 0)
    nb = K // 32
    return (nb // 32) * 17 + ((1 + (nb % 32 + 1) // 2) if nb % 32 else 0)


def dram_limit(ch_bytes, addr_beat_w):
    return min(ch_bytes // 64, 1 << addr_beat_w)


# ---------------------------------------------------------------------------------------------
# Internal-interface structs (tsr_pkg.sv). Fields MSB-first, as declared in SV.
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StructDef:
    name: str
    fields: tuple   # ((name, width), ...) MSB first

    @property
    def width(self) -> int:
        return sum(w for _, w in self.fields)

    def pack(self, **vals) -> int:
        v = 0
        for name, wd in self.fields:
            x = int(vals.pop(name, 0))
            if x < 0 or x >= (1 << wd):
                raise ValueError("%s.%s = %d does not fit %d bits" % (self.name, name, x, wd))
            v = (v << wd) | x
        if vals:
            raise KeyError("%s: unknown fields %s" % (self.name, sorted(vals)))
        return v

    def unpack(self, v: int) -> dict:
        out = {}
        for name, wd in reversed(self.fields):
            out[name] = v & ((1 << wd) - 1)
            v >>= wd
        return dict(reversed(list(out.items())))


FETCH_INSN_T = StructDef("tsr_fetch_insn_t", (("range_err", 1), ("pc", 32), ("insn", 256)))
DMA_CMD_T = StructDef("tsr_dma_cmd_t", (("is_st", 1), ("ch", 4), ("spm_line", 32),
                                        ("dram_beat", 32), ("n_lines", 32)))
QACT_CMD_T = StructDef("tsr_qact_cmd_t", (("kind", 1), ("dst_line", 32), ("src_line", 32),
                                          ("k", 16), ("t", 4), ("src_stride", 32),
                                          ("qa_lines", 16)))
MVM_CMD_T = StructDef("tsr_mvm_cmd_t", (("fmt", 1), ("acc", 1), ("dst_line", 32), ("x_line", 32),
                                        ("w_off", 32), ("n", 32), ("k", 16), ("t", 4),
                                        ("dst_stride", 32), ("y_lines", 32), ("rows_per_ch", 32),
                                        ("row_beats", 32), ("qa_lines", 16)))
# ---- M3: MVM unit internal interfaces (tsr_pkg.sv; docs/rtl-m3.md §3) ----
MVM_LANES, MVM_GROUPS, MVM_GROUP_LANES = 128, 4, 32
MVM_BEATS_W, MVM_BLK_W = 40, 11

# tsr_mvm -> clusters (broadcast pipe), once per MVM with N >= 1.
MVM_CCMD_T = StructDef("tsr_mvm_ccmd_t", (("fmt", 1), ("k", 16), ("n", 32), ("rows_per_ch", 32),
                                          ("row_beats", 16), ("w_off", 32), ("d_lines", 8)))
# tsr_mvm -> clusters: one SPM line of the QA images.
MVM_BCLINE_T = StructDef("tsr_mvm_bcline_t", (("last", 1), ("tok", 3), ("dbs", 1), ("addr", 10),
                                              ("data", 512)))
# cluster wrapper -> front end.
MVM_FCMD_T = StructDef("tsr_mvm_fcmd_t", (("fmt", 1), ("k", 16), ("row_beats", 16), ("w_off", 32),
                                          ("rows", 32), ("beats", MVM_BEATS_W)))
# front end -> back end: one tagged QUANT beat. sc / m / w are packed arrays: element i at bits
# [i*width +: width] of the field (logic [N-1:0][W-1:0]).
MVM_BEAT_T = StructDef("tsr_mvm_beat_t", (("last", 1), ("row_last", 1), ("blk", MVM_BLK_W), ("two", 1),
                                          ("half", 1), ("d_a", 32), ("d_b", 32), ("sc", 4 * 6),
                                          ("m", 8 * 6), ("w", MVM_LANES * 8)))
STRUCTS = (FETCH_INSN_T, DMA_CMD_T, QACT_CMD_T, MVM_CMD_T, MVM_CCMD_T, MVM_BCLINE_T, MVM_FCMD_T,
           MVM_BEAT_T)


def pack_array(vals, width: int) -> int:
    """Packed SV array logic [N-1:0][width-1:0] from a list (element i at bits [i*width +: width]);
    negative values are two's complement."""
    m = (1 << width) - 1
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & m) << (i * width)
    return v


def unpack_array(v: int, n: int, width: int, signed: bool = False) -> list:
    m = (1 << width) - 1
    out = []
    for i in range(n):
        x = (v >> (i * width)) & m
        if signed and x >> (width - 1):
            x -= 1 << width
        out.append(x)
    return out


def insn_int(words) -> int:
    """8 instruction words -> the 256-bit tsr_insn_t value (w0 in bits [31:0])."""
    v = 0
    for i, x in enumerate(words):
        v |= (x & 0xFFFFFFFF) << (32 * i)
    return v

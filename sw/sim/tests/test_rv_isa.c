// test_rv_isa.c — hand-encoded instruction sequences for tsr_rv and tsr_bus (ctest rv_unit).
//
// Covers every RV32I / M / C instruction class, Zicsr and the CSR set with its WARL rules, traps
// (illegal, ecall, ebreak, access faults incl. split accesses and the second fetch half), mret and
// U-mode, interrupts (vectoring, priority, enables, WFI), counters, debug mode (halt request,
// step, ebreak, triggers, program buffer, dret), the firmware address map and its access rules
// (CC §4.1), the SPM line buffer (CC §10.5), waiting / yielding stores (CC §7.3), SBA (CC §4.3),
// the test profile with HTIF and the ELF loader. Expected values come from C arithmetic on the
// same inputs or from the documents (docs/v2/04-control-core.md, 06-software-verification.md
// §4.2); Ibex-specific expectations are the *confirm* items of tsr_rv.c.
//
// No toolchain needed. `test_rv_isa --dump-enc` prints the encoder outputs for a cross-check
// against the GNU assembler (sw/sim/rv.cmake runs it as rv_unit_enc when riscv-none-elf-as is
// found).

#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_rv.h"

#include "test_util.h"

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------
static uint32_t eR(uint32_t f7, uint32_t rs2, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t eI(int32_t imm, uint32_t rs1, uint32_t f3, uint32_t rd, uint32_t op) {
    return (((uint32_t)imm & 0xFFFu) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t eS(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 5) & 0x7Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((u & 0x1Fu) << 7) | 0x23u;
}
static uint32_t eB(int32_t imm, uint32_t rs2, uint32_t rs1, uint32_t f3) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 12) & 1u) << 31) | (((u >> 5) & 0x3Fu) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           (((u >> 1) & 0xFu) << 8) | (((u >> 11) & 1u) << 7) | 0x63u;
}
static uint32_t eJ(int32_t imm, uint32_t rd) {
    uint32_t u = (uint32_t)imm;
    return (((u >> 20) & 1u) << 31) | (((u >> 1) & 0x3FFu) << 21) | (((u >> 11) & 1u) << 20) |
           (((u >> 12) & 0xFFu) << 12) | (rd << 7) | 0x6Fu;
}
#define LUI(rd, imm)        (((uint32_t)(imm) & 0xFFFFF000u) | ((rd) << 7) | 0x37u)
#define AUIPC(rd, imm)      (((uint32_t)(imm) & 0xFFFFF000u) | ((rd) << 7) | 0x17u)
#define JAL(rd, off)        eJ(off, rd)
#define JALR(rd, rs1, imm)  eI(imm, rs1, 0, rd, 0x67u)
#define BEQ(a, b, off)      eB(off, b, a, 0)
#define BNE(a, b, off)      eB(off, b, a, 1)
#define BLT(a, b, off)      eB(off, b, a, 4)
#define BGE(a, b, off)      eB(off, b, a, 5)
#define BLTU(a, b, off)     eB(off, b, a, 6)
#define BGEU(a, b, off)     eB(off, b, a, 7)
#define LB(rd, rs1, imm)    eI(imm, rs1, 0, rd, 0x03u)
#define LH(rd, rs1, imm)    eI(imm, rs1, 1, rd, 0x03u)
#define LW(rd, rs1, imm)    eI(imm, rs1, 2, rd, 0x03u)
#define LBU(rd, rs1, imm)   eI(imm, rs1, 4, rd, 0x03u)
#define LHU(rd, rs1, imm)   eI(imm, rs1, 5, rd, 0x03u)
#define SB(rs2, rs1, imm)   eS(imm, rs2, rs1, 0)
#define SH(rs2, rs1, imm)   eS(imm, rs2, rs1, 1)
#define SW(rs2, rs1, imm)   eS(imm, rs2, rs1, 2)
#define ADDI(rd, rs1, imm)  eI(imm, rs1, 0, rd, 0x13u)
#define SLTI(rd, rs1, imm)  eI(imm, rs1, 2, rd, 0x13u)
#define SLTIU(rd, rs1, imm) eI(imm, rs1, 3, rd, 0x13u)
#define XORI(rd, rs1, imm)  eI(imm, rs1, 4, rd, 0x13u)
#define ORI(rd, rs1, imm)   eI(imm, rs1, 6, rd, 0x13u)
#define ANDI(rd, rs1, imm)  eI(imm, rs1, 7, rd, 0x13u)
#define SLLI(rd, rs1, sh)   eI((int32_t)(sh), rs1, 1, rd, 0x13u)
#define SRLI(rd, rs1, sh)   eI((int32_t)(sh), rs1, 5, rd, 0x13u)
#define SRAI(rd, rs1, sh)   eI((int32_t)(0x400u | (sh)), rs1, 5, rd, 0x13u)
#define ADD(rd, a, b)       eR(0x00u, b, a, 0, rd, 0x33u)
#define SUB(rd, a, b)       eR(0x20u, b, a, 0, rd, 0x33u)
#define SLL(rd, a, b)       eR(0x00u, b, a, 1, rd, 0x33u)
#define SLT(rd, a, b)       eR(0x00u, b, a, 2, rd, 0x33u)
#define SLTU(rd, a, b)      eR(0x00u, b, a, 3, rd, 0x33u)
#define XOR(rd, a, b)       eR(0x00u, b, a, 4, rd, 0x33u)
#define SRL(rd, a, b)       eR(0x00u, b, a, 5, rd, 0x33u)
#define SRA(rd, a, b)       eR(0x20u, b, a, 5, rd, 0x33u)
#define OR(rd, a, b)        eR(0x00u, b, a, 6, rd, 0x33u)
#define AND(rd, a, b)       eR(0x00u, b, a, 7, rd, 0x33u)
#define MOP(f3, rd, a, b)   eR(0x01u, b, a, f3, rd, 0x33u)
#define FENCE               0x0FF0000Fu
#define FENCE_I             0x0000100Fu
#define ECALL               0x00000073u
#define EBREAK              0x00100073u
#define MRET                0x30200073u
#define DRET                0x7B200073u
#define WFI                 0x10500073u
#define NOP                 0x00000013u
#define CSRRW(rd, c, rs1)   eI((int32_t)(c), rs1, 1, rd, 0x73u)
#define CSRRS(rd, c, rs1)   eI((int32_t)(c), rs1, 2, rd, 0x73u)
#define CSRRC(rd, c, rs1)   eI((int32_t)(c), rs1, 3, rd, 0x73u)
#define CSRRWI(rd, c, u)    eI((int32_t)(c), u, 5, rd, 0x73u)
#define CSRRSI(rd, c, u)    eI((int32_t)(c), u, 6, rd, 0x73u)
#define CSRRCI(rd, c, u)    eI((int32_t)(c), u, 7, rd, 0x73u)

// Compressed (registers 8..15 for the primed forms).
static uint32_t c_addi4spn(uint32_t rd, uint32_t imm) {
    return (((imm >> 4) & 3u) << 11) | (((imm >> 6) & 0xFu) << 7) | (((imm >> 2) & 1u) << 6) | (((imm >> 3) & 1u) << 5) | ((rd - 8u) << 2);
}
static uint32_t c_lsw(uint32_t f3, uint32_t r, uint32_t rs1, uint32_t imm) {
    return (f3 << 13) | (((imm >> 3) & 7u) << 10) | ((rs1 - 8u) << 7) | (((imm >> 2) & 1u) << 6) | (((imm >> 6) & 1u) << 5) | ((r - 8u) << 2);
}
#define C_LW(rd, rs1, imm)  c_lsw(2, rd, rs1, imm)
#define C_SW(rs2, rs1, imm) c_lsw(6, rs2, rs1, imm)
static uint32_t c_ci(uint32_t f3, uint32_t rd, int32_t imm, uint32_t op) {
    uint32_t u = (uint32_t)imm;
    return (f3 << 13) | (((u >> 5) & 1u) << 12) | (rd << 7) | ((u & 0x1Fu) << 2) | op;
}
#define C_ADDI(rd, imm)     c_ci(0, rd, imm, 1)
#define C_LI(rd, imm)       c_ci(2, rd, imm, 1)
#define C_LUI(rd, imm6)     c_ci(3, rd, imm6, 1)      // rd = sext(imm6) << 12
#define C_SLLI(rd, sh)      c_ci(0, rd, (int32_t)(sh), 2)
static uint32_t c_addi16sp(int32_t imm) {
    uint32_t u = (uint32_t)imm;
    return (3u << 13) | (((u >> 9) & 1u) << 12) | (2u << 7) | (((u >> 4) & 1u) << 6) | (((u >> 6) & 1u) << 5) |
           (((u >> 7) & 3u) << 3) | (((u >> 5) & 1u) << 2) | 1u;
}
static uint32_t c_cj(uint32_t f3, int32_t imm) {
    uint32_t u = (uint32_t)imm;
    return (f3 << 13) | (((u >> 11) & 1u) << 12) | (((u >> 4) & 1u) << 11) | (((u >> 8) & 3u) << 9) | (((u >> 10) & 1u) << 8) |
           (((u >> 6) & 1u) << 7) | (((u >> 7) & 1u) << 6) | (((u >> 1) & 7u) << 3) | (((u >> 5) & 1u) << 2) | 1u;
}
#define C_JAL(imm)          c_cj(1, imm)
#define C_J(imm)            c_cj(5, imm)
static uint32_t c_shift(uint32_t f2, uint32_t rd, uint32_t sh) {
    return (4u << 13) | (((sh >> 5) & 1u) << 12) | (f2 << 10) | ((rd - 8u) << 7) | ((sh & 0x1Fu) << 2) | 1u;
}
#define C_SRLI(rd, sh)      c_shift(0, rd, sh)
#define C_SRAI(rd, sh)      c_shift(1, rd, sh)
static uint32_t c_andi(uint32_t rd, int32_t imm) {
    uint32_t u = (uint32_t)imm;
    return (4u << 13) | (((u >> 5) & 1u) << 12) | (2u << 10) | ((rd - 8u) << 7) | ((u & 0x1Fu) << 2) | 1u;
}
static uint32_t c_arith(uint32_t sel, uint32_t rd, uint32_t rs2) {
    return (4u << 13) | (3u << 10) | ((rd - 8u) << 7) | (sel << 5) | ((rs2 - 8u) << 2) | 1u;
}
#define C_SUB(rd, rs2)      c_arith(0, rd, rs2)
#define C_XOR(rd, rs2)      c_arith(1, rd, rs2)
#define C_OR(rd, rs2)       c_arith(2, rd, rs2)
#define C_AND(rd, rs2)      c_arith(3, rd, rs2)
static uint32_t c_br(uint32_t f3, uint32_t rs1, int32_t imm) {
    uint32_t u = (uint32_t)imm;
    return (f3 << 13) | (((u >> 8) & 1u) << 12) | (((u >> 3) & 3u) << 10) | ((rs1 - 8u) << 7) | (((u >> 6) & 3u) << 5) |
           (((u >> 1) & 3u) << 3) | (((u >> 5) & 1u) << 2) | 1u;
}
#define C_BEQZ(rs1, imm)    c_br(6, rs1, imm)
#define C_BNEZ(rs1, imm)    c_br(7, rs1, imm)
static uint32_t c_lwsp(uint32_t rd, uint32_t imm) {
    return (2u << 13) | (((imm >> 5) & 1u) << 12) | (rd << 7) | (((imm >> 2) & 7u) << 4) | (((imm >> 6) & 3u) << 2) | 2u;
}
static uint32_t c_swsp(uint32_t rs2, uint32_t imm) {
    return (6u << 13) | (((imm >> 2) & 0xFu) << 9) | (((imm >> 6) & 3u) << 7) | (rs2 << 2) | 2u;
}
#define C_JR(rs1)           ((4u << 13) | ((rs1) << 7) | 2u)
#define C_MV(rd, rs2)       ((4u << 13) | ((rd) << 7) | ((rs2) << 2) | 2u)
#define C_EBREAK            0x9002u
#define C_JALR(rs1)         ((4u << 13) | (1u << 12) | ((rs1) << 7) | 2u)
#define C_ADD(rd, rs2)      ((4u << 13) | (1u << 12) | ((rd) << 7) | ((rs2) << 2) | 2u)
#define C_NOP               0x0001u

// --dump-enc: one "<hex> <assembly>" line per encoder case (8 hex digits = 32-bit, 4 = 16-bit).
// --check-enc BIN: compares the same cases with BIN, the .text of that assembly as assembled by
// the GNU assembler (rv.cmake's rv_unit_enc builds it from the --dump-enc text).
static const uint8_t * ENC_BIN;
static size_t ENC_N, ENC_O;
static int ENC_BAD;
static void enc_item(uint32_t w, unsigned bytes, const char * text) {
    if (ENC_BIN == NULL) {
        printf(bytes == 4u ? "%08x %s\n" : "%04x %s\n", (unsigned)w, text);
        return;
    }
    uint32_t v = 0;
    for (unsigned i = 0; i < bytes && ENC_O + i < ENC_N; ++i) v |= (uint32_t)ENC_BIN[ENC_O + i] << (8u * i);
    if (ENC_O + bytes > ENC_N || v != w) {
        fprintf(stderr, "encoding mismatch: %s: test 0x%0*x, assembler 0x%0*x\n", text, 2 * (int)bytes, (unsigned)w, 2 * (int)bytes, (unsigned)v);
        ++ENC_BAD;
    }
    ENC_O += bytes;
}
static int dump_enc(void) {
#define D32(w, s) enc_item(w, 4, s)
#define D16(w, s) enc_item(w, 2, s)
    D32(LUI(5, 0x12345000), "lui t0, 0x12345");
    D32(AUIPC(6, 0xFFFFF000), "auipc t1, 0xfffff");
    D32(JAL(1, 2048), "jal ra, .+2048");
    D32(JAL(0, -1048576), "jal zero, .-1048576");
    D32(JALR(5, 6, -3), "jalr t0, -3(t1)");
    D32(BEQ(1, 2, -4096), "beq ra, sp, .-4096");
    D32(BNE(3, 4, 4094), "bne gp, tp, .+4094");
    D32(BLT(5, 6, 8), "blt t0, t1, .+8");
    D32(BGE(7, 8, -2), "bge t2, s0, .-2");
    D32(BLTU(9, 10, 2050), "bltu s1, a0, .+2050");
    D32(BGEU(11, 12, -16), "bgeu a1, a2, .-16");
    D32(LB(1, 2, -2048), "lb ra, -2048(sp)");
    D32(LH(3, 4, 2047), "lh gp, 2047(tp)");
    D32(LW(5, 6, 4), "lw t0, 4(t1)");
    D32(LBU(7, 8, -1), "lbu t2, -1(s0)");
    D32(LHU(9, 10, 6), "lhu s1, 6(a0)");
    D32(SB(1, 2, -2048), "sb ra, -2048(sp)");
    D32(SH(3, 4, 2047), "sh gp, 2047(tp)");
    D32(SW(5, 6, -33), "sw t0, -33(t1)");
    D32(ADDI(1, 2, -1), "addi ra, sp, -1");
    D32(SLTI(3, 4, 5), "slti gp, tp, 5");
    D32(SLTIU(5, 6, -7), "sltiu t0, t1, -7");
    D32(XORI(7, 8, 0x555), "xori t2, s0, 0x555");
    D32(ORI(9, 10, -2048), "ori s1, a0, -2048");
    D32(ANDI(11, 12, 0xF0), "andi a1, a2, 0xf0");
    D32(SLLI(13, 14, 31), "slli a3, a4, 31");
    D32(SRLI(15, 16, 1), "srli a5, a6, 1");
    D32(SRAI(17, 18, 17), "srai a7, s2, 17");
    D32(ADD(1, 2, 3), "add ra, sp, gp");
    D32(SUB(4, 5, 6), "sub tp, t0, t1");
    D32(SLL(7, 8, 9), "sll t2, s0, s1");
    D32(SLT(10, 11, 12), "slt a0, a1, a2");
    D32(SLTU(13, 14, 15), "sltu a3, a4, a5");
    D32(XOR(16, 17, 18), "xor a6, a7, s2");
    D32(SRL(19, 20, 21), "srl s3, s4, s5");
    D32(SRA(22, 23, 24), "sra s6, s7, s8");
    D32(OR(25, 26, 27), "or s9, s10, s11");
    D32(AND(28, 29, 30), "and t3, t4, t5");
    D32(MOP(0, 1, 2, 3), "mul ra, sp, gp");
    D32(MOP(1, 4, 5, 6), "mulh tp, t0, t1");
    D32(MOP(2, 7, 8, 9), "mulhsu t2, s0, s1");
    D32(MOP(3, 10, 11, 12), "mulhu a0, a1, a2");
    D32(MOP(4, 13, 14, 15), "div a3, a4, a5");
    D32(MOP(5, 16, 17, 18), "divu a6, a7, s2");
    D32(MOP(6, 19, 20, 21), "rem s3, s4, s5");
    D32(MOP(7, 22, 23, 31), "remu s6, s7, t6");
    D32(FENCE, "fence");
    D32(FENCE_I, "fence.i");
    D32(ECALL, "ecall");
    D32(EBREAK, "ebreak");
    D32(MRET, "mret");
    D32(DRET, "dret");
    D32(WFI, "wfi");
    D32(CSRRW(1, 0x340, 2), "csrrw ra, mscratch, sp");
    D32(CSRRS(3, 0x300, 0), "csrrs gp, mstatus, zero");
    D32(CSRRC(4, 0x304, 5), "csrrc tp, mie, t0");
    D32(CSRRWI(6, 0x305, 31), "csrrwi t1, mtvec, 31");
    D32(CSRRSI(7, 0x7B0, 1), "csrrsi t2, dcsr, 1");
    D32(CSRRCI(8, 0xB00, 4), "csrrci s0, mcycle, 4");
    D16(c_addi4spn(8, 1020), "c.addi4spn s0, sp, 1020");
    D16(C_LW(9, 10, 124), "c.lw s1, 124(a0)");
    D16(C_SW(11, 12, 64), "c.sw a1, 64(a2)");
    D16(C_NOP, "c.nop");
    D16(C_ADDI(5, -32), "c.addi t0, -32");
    D16(C_JAL(-2048), "c.jal .-2048");
    D16(C_LI(6, 31), "c.li t1, 31");
    D16(c_addi16sp(-512), "c.addi16sp sp, -512");
    D16(c_addi16sp(496), "c.addi16sp sp, 496");
    D16(C_LUI(7, -32), "c.lui t2, 0xfffe0");
    D16(C_LUI(8, 1), "c.lui s0, 1");
    D16(C_SRLI(13, 31), "c.srli a3, 31");
    D16(C_SRAI(14, 1), "c.srai a4, 1");
    D16(c_andi(15, -1), "c.andi a5, -1");
    D16(C_SUB(8, 9), "c.sub s0, s1");
    D16(C_XOR(10, 11), "c.xor a0, a1");
    D16(C_OR(12, 13), "c.or a2, a3");
    D16(C_AND(14, 15), "c.and a4, a5");
    D16(C_J(2046), "c.j .+2046");
    D16(C_BEQZ(8, -256), "c.beqz s0, .-256");
    D16(C_BNEZ(15, 254), "c.bnez a5, .+254");
    D16(C_SLLI(31, 31), "c.slli t6, 31");
    D16(c_lwsp(1, 252), "c.lwsp ra, 252(sp)");
    D16(C_JR(1), "c.jr ra");
    D16(C_MV(5, 6), "c.mv t0, t1");
    D16(C_EBREAK, "c.ebreak");
    D16(C_JALR(7), "c.jalr t2");
    D16(C_ADD(8, 9), "c.add s0, s1");
    D16(c_swsp(31, 128), "c.swsp t6, 128(sp)");
#undef D32
#undef D16
    if (ENC_BIN == NULL) return 0;
    if (ENC_O != ENC_N) { fprintf(stderr, "encoding check: %zu bytes compared, %zu assembled\n", ENC_O, ENC_N); ++ENC_BAD; }
    if (ENC_BAD) { fprintf(stderr, "rv_unit_enc: %d mismatches\n", ENC_BAD); return 1; }
    printf("rv_unit_enc: %zu bytes of encodings identical to the assembler's\n", ENC_O);
    return 0;
}

static int check_enc(const char * path) {
    static uint8_t buf[4096];
    FILE * f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "cannot read %s\n", path); return 1; }
    ENC_N = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    ENC_BIN = buf;
    return dump_enc();
}

// ---------------------------------------------------------------------------
// Stub environment of the firmware profile (control window, core window, SPM, irq)
// ---------------------------------------------------------------------------
#define SPM_TEST_BYTES (64u * 1024u)
typedef struct acc { char kind; int write; uint32_t off, n, val; } acc;
static struct {
    uint32_t cw[1024];
    uint32_t credit;        // ISSUE_PUSH waits while 0
    uint32_t lq_avail;      // LQ_POP faults while 0
    uint32_t pushes;
    uint32_t irq;
    int      yield_n;       // the next yield_n calls of yield() return 1
    int      yields;
    uint8_t  spm[SPM_TEST_BYTES];
    acc      log[64];
    int      nlog;
} E;

static void logacc(char k, int w, uint32_t off, uint32_t n, uint32_t v) {
    if (E.nlog < 64) { acc a = { k, w, off, n, v }; E.log[E.nlog++] = a; }
}
static int st_cw(void * ctx, int is_write, uint32_t off, uint32_t * val) {
    (void)ctx;
    logacc('c', is_write, off, 4, is_write ? *val : 0);
    if (is_write) {
        if (off == TSR_CW_ISSUE_PUSH) {
            if (E.credit == 0u) return TSR_BUS_ENV_WAIT;
            --E.credit; ++E.pushes;
            return TSR_BUS_ENV_OK;
        }
        if (off == TSR_CW_LQ_POP) {
            if (E.lq_avail == 0u) return TSR_BUS_ENV_FAULT;
            --E.lq_avail;
            return TSR_BUS_ENV_OK;
        }
        E.cw[off / 4u] = *val;
        return TSR_BUS_ENV_OK;
    }
    *val = E.cw[off / 4u];
    return TSR_BUS_ENV_OK;
}
static int st_core(void * ctx, uint32_t off, uint32_t * val) {
    (void)ctx;
    logacc('k', 0, off, 4, 0);
    *val = 0xC0DE0000u | off;
    return TSR_BUS_ENV_OK;
}
static void st_spm(void * ctx, uint32_t off, uint32_t n, int w) { (void)ctx; logacc('s', w, off, n, 0); }
static uint32_t st_irq(void * ctx) { (void)ctx; return E.irq; }
static int st_yield(void * ctx) {
    (void)ctx;
    if (E.yield_n > 0) { --E.yield_n; ++E.yields; return 1; }
    return 0;
}

// ---------------------------------------------------------------------------
// Harness: a firmware-profile bus with the stub, one hart, an assembly cursor
// ---------------------------------------------------------------------------
static tsr_bus * B;
static tsr_rv * RV;
static uint32_t PC;
#define VEC   0x1000u            // mtvec base used by the tests (vectored)
#define DTCM  TSR_RV_DTCM_BASE
#define CW    TSR_RV_CTL_BASE
#define CORE  TSR_RV_CORE_BASE
#define SPM   TSR_RV_SPM_BASE
#define SPMLB TSR_RV_SPM_LB_BASE
#define DM    TSR_RV_DM_BASE

static tsr_rv_retire RET[64];
static int NRET;
static int retire_cb(void * ctx, const tsr_rv_retire * r) {
    (void)ctx;
    if (NRET < 64) RET[NRET++] = *r;
    return 0;
}
static uint32_t DBG_EV[16], DBG_ARG[16];
static int NDBG;
static void debug_cb(void * ctx, uint32_t ev, uint32_t arg) {
    (void)ctx;
    if (NDBG < 16) { DBG_EV[NDBG] = ev; DBG_ARG[NDBG] = arg; ++NDBG; }
}

static uint8_t * mem(uint32_t a, uint32_t n) {
    uint8_t * p = tsr_bus_mem(B, a, n);
    if (p == NULL) { fprintf(stderr, "test: no memory at 0x%08x\n", a); exit(1); }
    return p;
}
static void w32(uint32_t a, uint32_t v) { uint8_t * p = mem(a, 4); p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static uint32_t r32(uint32_t a) { uint8_t * p = mem(a, 4); return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void org(uint32_t a) { PC = a; }
static void i32(uint32_t w) { w32(PC, w); PC += 4; }   // also used for data words
static void i16(uint32_t h) { uint8_t * p = mem(PC, 2); p[0] = (uint8_t)h; p[1] = (uint8_t)(h >> 8); PC += 2; }

static uint32_t csr(uint32_t c) {
    uint32_t v = 0xBAD0BAD0u;
    CHECK(tsr_rv_csr_read(RV, c, &v) == 0, "csr 0x%03x readable", c);
    return v;
}
static void setcsr(uint32_t c, uint32_t v) { CHECK(tsr_rv_csr_write(RV, c, v) == 0, "csr 0x%03x writable", c); }
static uint32_t X(unsigned i) { return tsr_rv_reg(RV, i); }
static void setx(unsigned i, uint32_t v) { tsr_rv_set_reg(RV, i, v); }

static void fresh(void) {
    if (RV) tsr_rv_destroy(RV);
    if (B) tsr_bus_destroy(B);
    memset(&E, 0, sizeof(E));
    tsr_bus_env env;
    memset(&env, 0, sizeof(env));
    env.cw = st_cw; env.core_read = st_core; env.spm = E.spm; env.spm_bytes = SPM_TEST_BYTES;
    env.spm_access = st_spm; env.irq = st_irq; env.yield = st_yield;
    B = tsr_bus_create_fw(NULL, &env);
    RV = tsr_rv_create(tsr_bus_iface(B));
    tsr_bus_attach(B, RV);
    tsr_rv_set_retire_hook(RV, retire_cb, NULL);
    tsr_rv_set_debug_hook(RV, debug_cb, NULL);
    NRET = 0; NDBG = 0;
    // Vector area full of NOPs; tests run from 0x80 (reset entry) or set the pc.
    org(VEC);
    for (int i = 0; i < 64; ++i) i32(NOP);
    setcsr(TSR_RV_CSR_MTVEC, VEC);
}

// Executes the instruction(s) placed at `at` for n instructions from there.
static uint32_t run_at(uint32_t at, uint64_t n) { tsr_rv_set_pc(RV, at); return tsr_rv_step(RV, n); }
static uint32_t one32(uint32_t w) { org(0x400); i32(w); return run_at(0x400, 1); }
static uint32_t one16(uint32_t h) { org(0x400); i16(h); return run_at(0x400, 1); }

static void expect_trap(const char * what, uint32_t cause, uint32_t epc, uint32_t tval) {
    CHECK(csr(TSR_RV_CSR_MCAUSE) == cause, "%s: mcause 0x%x, want 0x%x", what, csr(TSR_RV_CSR_MCAUSE), cause);
    CHECK(csr(TSR_RV_CSR_MEPC) == epc, "%s: mepc 0x%x, want 0x%x", what, csr(TSR_RV_CSR_MEPC), epc);
    CHECK(csr(TSR_RV_CSR_MTVAL) == tval, "%s: mtval 0x%x, want 0x%x", what, csr(TSR_RV_CSR_MTVAL), tval);
    CHECK(tsr_rv_pc(RV) == VEC + ((cause & TSR_RV_MCAUSE_IRQ) ? 4u * (cause & 31u) : 0u),
          "%s: pc 0x%x after the trap", what, tsr_rv_pc(RV));
}

// ---------------------------------------------------------------------------
// RV32I
// ---------------------------------------------------------------------------
static void test_alu(void) {
    fresh();
    const uint32_t a = 0x12345678u, m1 = 0xFFFFFFFFu, h = 0x80000000u;
    org(0x80);
    i32(LUI(1, 0x12345000)); i32(ADDI(1, 1, 0x678));
    i32(ADDI(2, 0, -1));
    i32(SLTI(3, 2, 0)); i32(SLTIU(4, 2, 1)); i32(SLTIU(5, 0, 1));
    i32(XORI(6, 1, -1)); i32(ORI(7, 0, 0x7FF)); i32(ANDI(8, 1, 0xF0));
    i32(SLLI(9, 1, 4)); i32(SRLI(10, 2, 28)); i32(SRAI(11, 2, 28));
    i32(LUI(12, 0x80000000)); i32(SRAI(13, 12, 31)); i32(SRLI(14, 12, 31));
    i32(ADD(15, 1, 1)); i32(SUB(16, 0, 1)); i32(SLL(17, 14, 10));
    i32(SLT(18, 12, 0)); i32(SLTU(19, 0, 12)); i32(XOR(20, 1, 2));
    i32(SRL(21, 12, 10)); i32(SRA(22, 12, 10)); i32(OR(23, 1, 12)); i32(AND(24, 1, 9));
    uint32_t auipc_pc = PC;
    i32(AUIPC(25, 0x00001000)); i32(ADDI(0, 0, 5)); i32(LUI(0, 0x1000));
    i32(SLTI(26, 0, -2048)); i32(SLTIU(27, 0, -1)); i32(SRA(28, 2, 7)); i32(SLL(29, 1, 2));
    uint32_t end = PC;
    CHECK(tsr_rv_step(RV, (end - 0x80) / 4) == TSR_RV_STOP_BUDGET, "budget stop");
    CHECK(tsr_rv_pc(RV) == end, "pc at end 0x%x", tsr_rv_pc(RV));
    const uint32_t exp[30] = {
        0, a, m1, 1, 0, 1, ~a, 0x7FF, a & 0xF0u, a << 4, 0xFu, m1, h, m1, 1,
        a + a, 0u - a, 1u << 15, 1, 1, a ^ m1, h >> 15, 0xFFFF0000u, a | h, a & (a << 4),
        auipc_pc + 0x1000u, 0, 1, m1, a << 31
    };
    for (unsigned i = 0; i < 30; ++i) CHECK(X(i) == exp[i], "x%u = 0x%08x, want 0x%08x", i, X(i), exp[i]);
    CHECK(tsr_rv_retired(RV) == (end - 0x80) / 4, "retired");
}

static void test_branches(void) {
    fresh();
    static const uint32_t vals[] = { 0, 1, 2, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0xFFFFFFFEu };
    const unsigned nv = sizeof(vals) / sizeof(vals[0]);
    for (unsigned f = 0; f < 6; ++f) {
        static const uint32_t f3s[6] = { 0, 1, 4, 5, 6, 7 };
        for (unsigned i = 0; i < nv; ++i) for (unsigned j = 0; j < nv; ++j) {
            uint32_t x = vals[i], y = vals[j];
            int t;
            switch (f3s[f]) {
            case 0: t = x == y; break;
            case 1: t = x != y; break;
            case 4: t = (int32_t)x < (int32_t)y; break;
            case 5: t = (int32_t)x >= (int32_t)y; break;
            case 6: t = x < y; break;
            default: t = x >= y; break;
            }
            int32_t off = ((i + j) & 1u) ? -8 : 16;
            setx(5, x); setx(6, y);
            one32(eB(off, 6, 5, f3s[f]));
            uint32_t want = t ? (uint32_t)(0x400 + off) : 0x404u;
            CHECK(tsr_rv_pc(RV) == want, "branch f3 %u 0x%x 0x%x: pc 0x%x want 0x%x", f3s[f], x, y, tsr_rv_pc(RV), want);
        }
    }
    // Branch cost: taken 3, not taken 1 (SW §4.2 table).
    setx(5, 1); setx(6, 1);
    uint64_t c0 = tsr_rv_cycles(RV);
    one32(BEQ(5, 6, 8));
    CHECK(tsr_rv_cycles(RV) - c0 == 3u, "taken branch cycles %llu", (unsigned long long)(tsr_rv_cycles(RV) - c0));
    c0 = tsr_rv_cycles(RV);
    one32(BNE(5, 6, 8));
    CHECK(tsr_rv_cycles(RV) - c0 == 1u, "not-taken branch cycles");
}

static void test_jumps(void) {
    fresh();
    one32(JAL(1, 12));
    CHECK(X(1) == 0x404u && tsr_rv_pc(RV) == 0x40Cu, "jal +12");
    one32(JAL(1, -0x400));
    CHECK(X(1) == 0x404u && tsr_rv_pc(RV) == 0x0u, "jal -0x400");
    one32(JAL(0, 2048));
    CHECK(X(0) == 0u && tsr_rv_pc(RV) == 0xC00u, "j +2048");
    setx(6, 0x203);
    one32(JALR(5, 6, 0));
    CHECK(X(5) == 0x404u && tsr_rv_pc(RV) == 0x202u, "jalr clears bit 0");
    setx(6, 0x200);
    one32(JALR(6, 6, 4));
    CHECK(X(6) == 0x404u && tsr_rv_pc(RV) == 0x204u, "jalr rd == rs1 uses the old rs1");
    setx(7, 0x1000);
    one32(JALR(1, 7, -16));
    CHECK(tsr_rv_pc(RV) == 0xFF0u, "jalr negative offset");
    one16(C_JAL(-16));
    CHECK(X(1) == 0x402u && tsr_rv_pc(RV) == 0x3F0u, "c.jal links pc + 2");
    one16(C_J(64));
    CHECK(tsr_rv_pc(RV) == 0x440u, "c.j");
    setx(9, 0x601);
    one16(C_JALR(9));
    CHECK(X(1) == 0x402u && tsr_rv_pc(RV) == 0x600u, "c.jalr");
    one16(C_JR(9));
    CHECK(tsr_rv_pc(RV) == 0x600u, "c.jr");
}

static void test_loads_stores(void) {
    fresh();
    setx(10, DTCM);
    setx(1, 0xF1E2D3C4u); setx(7, 0x55); setx(8, 0xFFFFABCDu);
    org(0x80);
    i32(SW(1, 10, 0)); i32(LB(2, 10, 0)); i32(LBU(3, 10, 0)); i32(LH(4, 10, 2)); i32(LHU(5, 10, 2));
    i32(LW(6, 10, 0)); i32(SB(7, 10, 1)); i32(SH(8, 10, 2)); i32(LW(9, 10, 0));
    i32(ADDI(11, 10, 16)); i32(SW(1, 11, -12)); i32(LW(12, 11, -12)); i32(LB(13, 11, -9));
    i32(LW(14, 0, 0x80));   // ITCM data read: the first instruction's bits
    uint32_t end = PC;
    tsr_rv_step(RV, (end - 0x80) / 4);
    CHECK(X(2) == 0xFFFFFFC4u && X(3) == 0xC4u, "lb / lbu");
    CHECK(X(4) == 0xFFFFF1E2u && X(5) == 0xF1E2u, "lh / lhu");
    CHECK(X(6) == 0xF1E2D3C4u, "lw");
    CHECK(X(9) == 0xABCD55C4u && r32(DTCM) == 0xABCD55C4u, "sb / sh merge: 0x%08x", X(9));
    CHECK(X(12) == 0xF1E2D3C4u && r32(DTCM + 4) == 0xF1E2D3C4u, "negative offset");
    CHECK(X(13) == 0xFFFFFFF1u, "lb sign");
    CHECK(X(14) == SW(1, 10, 0), "ITCM data read");
    // Load cost: lsu 1 + TCM 1 (SW §4.2: 2).
    uint64_t c0 = tsr_rv_cycles(RV);
    one32(LW(6, 10, 0));
    CHECK(tsr_rv_cycles(RV) - c0 == 2u, "TCM load cycles");
    // A load to x0 still accesses the bus (here: the control window).
    E.nlog = 0;
    setx(10, CW);
    one32(LW(0, 10, TSR_CW_QSTATUS));
    CHECK(X(0) == 0u && E.nlog == 1 && E.log[0].off == TSR_CW_QSTATUS, "load to x0 accesses the bus");
    // Self-modifying code: store an instruction into ITCM, fence.i, execute it.
    setx(10, 0x500); setx(11, ADDI(20, 0, 77));
    org(0x480); i32(SW(11, 10, 0)); i32(FENCE_I); i32(FENCE); i32(JAL(0, 0x500 - 0x48C));
    run_at(0x480, 5);
    CHECK(X(20) == 77u, "store to ITCM, fence.i, execute");
}

static void test_m(void) {
    fresh();
    static const uint32_t vals[] = { 0, 1, 2, 3, 7, 0xFFFFFFF9u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x12345678u, 0xFEDCBA98u };
    const unsigned nv = sizeof(vals) / sizeof(vals[0]);
    for (uint32_t f3 = 0; f3 < 8; ++f3) for (unsigned i = 0; i < nv; ++i) for (unsigned j = 0; j < nv; ++j) {
        uint32_t a = vals[i], b = vals[j], want;
        int64_t sa = (int32_t)a, sb = (int32_t)b;
        switch (f3) {
        case 0: want = (uint32_t)((uint64_t)a * b); break;
        case 1: want = (uint32_t)((uint64_t)(sa * sb) >> 32); break;
        case 2: want = (uint32_t)((uint64_t)(sa * (int64_t)b) >> 32); break;
        case 3: want = (uint32_t)(((uint64_t)a * b) >> 32); break;
        case 4: want = b == 0u ? 0xFFFFFFFFu : (uint32_t)(int32_t)(sa / sb); break;   // int64 division: no overflow
        case 5: want = b == 0u ? 0xFFFFFFFFu : a / b; break;
        case 6: want = b == 0u ? a : (uint32_t)(int32_t)(sa % sb); break;
        default: want = b == 0u ? a : a % b; break;
        }
        setx(11, a); setx(12, b);
        one32(MOP(f3, 10, 11, 12));
        CHECK(X(10) == want, "M f3 %u: 0x%x op 0x%x = 0x%x, want 0x%x", f3, a, b, X(10), want);
    }
    uint64_t c0 = tsr_rv_cycles(RV);
    one32(MOP(0, 10, 11, 12));
    CHECK(tsr_rv_cycles(RV) - c0 == 3u, "mul cycles");
    c0 = tsr_rv_cycles(RV);
    one32(MOP(5, 10, 11, 12));
    CHECK(tsr_rv_cycles(RV) - c0 == 37u, "divu cycles");
}

static void test_compressed(void) {
    fresh();
    setx(2, DTCM + 0x100);
    one16(c_addi4spn(8, 1020));   CHECK(X(8) == DTCM + 0x100 + 1020, "c.addi4spn");
    setx(10, DTCM); w32(DTCM + 124, 0xCAFEF00Du);
    one16(C_LW(9, 10, 124));      CHECK(X(9) == 0xCAFEF00Du, "c.lw");
    setx(11, 0x1234ABCDu);
    one16(C_SW(11, 10, 64));      CHECK(r32(DTCM + 64) == 0x1234ABCDu, "c.sw");
    setx(5, 100);
    one16(C_ADDI(5, -32));        CHECK(X(5) == 68u, "c.addi");
    one16(C_NOP);                 CHECK(tsr_rv_pc(RV) == 0x402u, "c.nop");
    one16(C_LI(6, -1));           CHECK(X(6) == 0xFFFFFFFFu, "c.li");
    setx(2, 0x1000);
    one16(c_addi16sp(-512));      CHECK(X(2) == 0x1000u - 512u, "c.addi16sp -");
    one16(c_addi16sp(496));       CHECK(X(2) == 0x1000u - 16u, "c.addi16sp +");
    one16(C_LUI(7, -32));         CHECK(X(7) == 0xFFFE0000u, "c.lui negative");
    one16(C_LUI(8, 1));           CHECK(X(8) == 0x1000u, "c.lui");
    setx(13, 0x80000000u);
    one16(C_SRLI(13, 31));        CHECK(X(13) == 1u, "c.srli");
    setx(14, 0x80000000u);
    one16(C_SRAI(14, 4));         CHECK(X(14) == 0xF8000000u, "c.srai");
    setx(15, 0x12345678u);
    one16(c_andi(15, -16));       CHECK(X(15) == 0x12345670u, "c.andi");
    setx(8, 10); setx(9, 3);
    one16(C_SUB(8, 9));           CHECK(X(8) == 7u, "c.sub");
    setx(10, 0xF0F0); setx(11, 0xFF00);
    one16(C_XOR(10, 11));         CHECK(X(10) == 0x0FF0u, "c.xor");
    setx(12, 0xF0); setx(13, 0x0F);
    one16(C_OR(12, 13));          CHECK(X(12) == 0xFFu, "c.or");
    setx(14, 0xF0); setx(15, 0x3C);
    one16(C_AND(14, 15));         CHECK(X(14) == 0x30u, "c.and");
    setx(8, 0);
    one16(C_BEQZ(8, -256));       CHECK(tsr_rv_pc(RV) == 0x300u, "c.beqz taken");
    one16(C_BNEZ(8, 254));        CHECK(tsr_rv_pc(RV) == 0x402u, "c.bnez not taken");
    setx(15, 5);
    one16(C_BNEZ(15, 254));       CHECK(tsr_rv_pc(RV) == 0x4FEu, "c.bnez taken");
    setx(31, 3);
    one16(C_SLLI(31, 30));        CHECK(X(31) == 0xC0000000u, "c.slli");
    setx(2, DTCM); w32(DTCM + 252, 0x600DD00Du);
    one16(c_lwsp(1, 252));        CHECK(X(1) == 0x600DD00Du, "c.lwsp");
    setx(31, 0x5EED5EEDu);
    one16(c_swsp(31, 128));       CHECK(r32(DTCM + 128) == 0x5EED5EEDu, "c.swsp");
    setx(6, 0xABCD);
    one16(C_MV(5, 6));            CHECK(X(5) == 0xABCDu, "c.mv");
    setx(8, 5); setx(9, 6);
    one16(C_ADD(8, 9));           CHECK(X(8) == 11u, "c.add");
    // Hints are legal no-ops: c.addi x0, c.li x0, c.mv x0, c.add x0, c.slli x0, c.lui x0.
    const uint32_t hints[] = { C_ADDI(0, 5), C_LI(0, 3), C_MV(0, 5), C_ADD(0, 5), C_SLLI(0, 3), C_LUI(0, 1), C_SRLI(8, 0) };
    for (unsigned i = 0; i < sizeof(hints) / sizeof(hints[0]); ++i) {
        setx(8, 0x77);
        one16(hints[i]);
        CHECK(tsr_rv_pc(RV) == 0x402u && X(0) == 0u && X(8) == 0x77u, "hint 0x%04x is a no-op", hints[i]);
    }
    // Mixed 16 / 32-bit stream with a 32-bit instruction at a 2-byte offset.
    org(0x500); i16(C_LI(10, 1)); i32(ADDI(10, 10, 2)); i16(C_ADDI(10, 4)); i32(ADDI(10, 10, 8));
    run_at(0x500, 4);
    CHECK(X(10) == 15u && tsr_rv_pc(RV) == 0x50Cu, "mixed-length stream");
}

static void test_illegal(void) {
    fresh();
    const uint32_t ill32[] = {
        0xFFFFFFFFu,                          // >= 48-bit encoding
        eI(0, 1, 3, 2, 0x03u),                // LD
        eI(0, 1, 6, 2, 0x03u),                // LWU
        eI(0, 1, 7, 2, 0x03u),
        eS(0, 1, 2, 3),                       // SD
        eI(0x020 | 3, 1, 1, 2, 0x13u),        // SLLI with funct7 != 0 (shamt[5] / Zbb)
        eI(0x200 | 3, 1, 5, 2, 0x13u),        // SRLI funct7 0x10
        eR(0x20u, 3, 2, 1, 1, 0x33u),         // OP funct7 0x20 with SLL
        eR(0x02u, 3, 2, 0, 1, 0x33u),
        eR(0x04u, 3, 2, 4, 1, 0x33u),         // Zbb-like
        eB(8, 1, 2, 2), eB(8, 1, 2, 3),       // BRANCH funct3 2, 3
        eI(0, 1, 1, 2, 0x67u),                // JALR funct3 1
        0x0000202Fu,                          // AMO (lr.w)
        0x00002007u,                          // FLW
        0x0000003Bu, 0x0000001Bu,             // OP-32, OP-IMM-32
        0x0000200Fu,                          // MISC-MEM funct3 2
        0x000000F3u,                          // ECALL with rd != 0
        0x00108073u,                          // EBREAK with rs1 != 0
        0x10200073u, 0x00200073u,             // SRET, URET
        0x12000073u,                          // SFENCE.VMA
        0x30500073u,                          // unknown funct12
        0x00004073u,                          // SYSTEM funct3 4
        CSRRS(1, 0x3A0, 0),                   // pmpcfg0 (PMPEnable 0)
        CSRRS(1, 0x3B0, 0),                   // pmpaddr0
        CSRRS(1, 0x747, 0),                   // mseccfg
        CSRRS(1, 0xC00, 0), CSRRS(1, 0xC02, 0), CSRRS(1, 0xC01, 0),   // cycle, instret, time
        CSRRS(1, 0x180, 0), CSRRS(1, 0x100, 0), CSRRS(1, 0x7B0, 0),   // satp, sstatus, dcsr outside debug
        CSRRS(1, 0x7B1, 0), CSRRS(1, 0x7B2, 0), CSRRS(1, 0xB01, 0), CSRRS(1, 0x7A4, 0),
        CSRRW(0, 0xF14, 1), CSRRS(0, 0xF11, 1), CSRRCI(0, 0xF12, 1),  // writes to read-only CSRs
        0x00000000u | 0x7Fu,                  // opcode 0x7F
        0x0000005Bu,                          // custom-2
    };
    for (unsigned i = 0; i < sizeof(ill32) / sizeof(ill32[0]); ++i) {
        setx(1, 0x5A5A); setx(2, 0x77);
        uint64_t ret0 = tsr_rv_retired(RV);
        one32(ill32[i]);
        expect_trap("illegal32", TSR_RV_EXC_ILLEGAL, 0x400, ill32[i]);
        CHECK(X(1) == 0x5A5Au && X(2) == 0x77u, "illegal 0x%08x wrote a register", ill32[i]);
        CHECK(tsr_rv_retired(RV) == ret0, "illegal 0x%08x retired", ill32[i]);
    }
    static const uint32_t ill16[] = {
        0x0000u,              // all zero
        0x0004u,              // C.ADDI4SPN nzuimm 0 (rd != 0)
        0x2000u, 0x6000u, 0xA000u, 0xE000u,   // C.FLD C.FLW C.FSD C.FSW
        0x8000u, 0x8400u,     // quadrant 0 funct3 100 (Zcb)
        0x6101u,              // C.ADDI16SP nzimm 0
        0x6281u,              // C.LUI nzimm 0
        0x9001u, 0x9401u,     // C.SRLI / C.SRAI shamt[5] = 1
        0x9C01u, 0x9C21u,     // C.SUBW / C.ADDW
        0x1006u,              // C.SLLI shamt[5] = 1
        0x4002u,              // C.LWSP rd 0
        0x8002u,              // C.JR rs1 0
        0x2002u, 0x6002u, 0xA002u, 0xE002u,   // C.FLDSP C.FLWSP C.FSDSP C.FSWSP
    };
    for (unsigned i = 0; i < sizeof(ill16) / sizeof(ill16[0]); ++i) {
        one16(ill16[i]);
        expect_trap("illegal16", TSR_RV_EXC_ILLEGAL, 0x400, ill16[i]);
    }
    // Legal oddities: FENCE with any fields, FENCE.I, csrrs to read-only CSRs with rs1 = 0.
    const uint32_t legal[] = { 0x8330000Fu, FENCE_I, CSRRS(1, 0xF14, 0), CSRRSI(1, 0xF12, 0), CSRRCI(1, 0xF13, 0) };
    for (unsigned i = 0; i < sizeof(legal) / sizeof(legal[0]); ++i) {
        one32(legal[i]);
        CHECK(tsr_rv_pc(RV) == 0x404u, "0x%08x is legal", legal[i]);
    }
}

// ---------------------------------------------------------------------------
// Zicsr and the CSR set
// ---------------------------------------------------------------------------
static void test_csr(void) {
    fresh();
    setx(2, 0xDEADBEEFu); setx(5, 0xFF);
    one32(CSRRW(1, 0x340, 2));   CHECK(X(1) == 0u && csr(0x340) == 0xDEADBEEFu, "csrrw");
    one32(CSRRS(3, 0x340, 0));   CHECK(X(3) == 0xDEADBEEFu && csr(0x340) == 0xDEADBEEFu, "csrrs x0: read only");
    one32(CSRRC(4, 0x340, 5));   CHECK(X(4) == 0xDEADBEEFu && csr(0x340) == 0xDEADBE00u, "csrrc");
    one32(CSRRS(4, 0x340, 5));   CHECK(csr(0x340) == 0xDEADBEFFu, "csrrs");
    one32(CSRRWI(6, 0x340, 0x1F)); CHECK(X(6) == 0xDEADBEFFu && csr(0x340) == 0x1Fu, "csrrwi");
    one32(CSRRCI(6, 0x340, 0x3)); CHECK(csr(0x340) == 0x1Cu, "csrrci");
    one32(CSRRSI(6, 0x340, 0x2)); CHECK(X(6) == 0x1Cu && csr(0x340) == 0x1Eu, "csrrsi");
    one32(CSRRSI(6, 0x340, 0));  CHECK(X(6) == 0x1Eu, "csrrsi 0");
    // Identification.
    CHECK(csr(0x301) == 0x40101104u, "misa");
    CHECK(csr(0xF11) == 0u && csr(0xF12) == 22u && csr(0xF13) == 0u && csr(0xF14) == 0u && csr(0xF15) == 0u, "ids");
    one32(CSRRW(0, 0x301, 2));   CHECK(tsr_rv_pc(RV) == 0x404u && csr(0x301) == 0x40101104u, "misa write ignored");
    // WARL fields.
    setcsr(0x305, 0x12345u);     CHECK(csr(0x305) == 0x12301u, "mtvec vectored, base [31:8]");
    setcsr(0x341, 0x1235u);      CHECK(csr(0x341) == 0x1234u, "mepc bit 0");
    setcsr(0x342, 0xFFFFFFFFu);  CHECK(csr(0x342) == 0x8000001Fu, "mcause bits");
    setcsr(0x300, 0xFFFFFFFFu);  CHECK(csr(0x300) == 0x00221888u, "mstatus fields");
    setcsr(0x300, 0x00001000u);  CHECK(csr(0x300) == 0u, "mstatus MPP 2 -> U");
    setcsr(0x300, 0x00000800u);  CHECK(csr(0x300) == 0u, "mstatus MPP 1 -> U");
    setcsr(0x304, 0xFFFFFFFFu);  CHECK(csr(0x304) == 0x7FFF0888u, "mie mask");
    E.irq = 0xFFFFFFFFu;
    one32(NOP);                  // irq sampled at the boundary (mstatus.MIE = 0: not taken)
    CHECK(csr(0x344) == 0x7FFF0888u, "mip = lines");
    one32(CSRRW(0, 0x344, 0));   CHECK(csr(0x344) == 0x7FFF0888u, "mip write ignored");
    E.irq = 0; setcsr(0x304, 0);
    setcsr(0x320, 0xFFFFFFFFu);  CHECK(csr(0x320) == 5u, "mcountinhibit CY, IR");
    setcsr(0x320, 0);
    // Present, read 0, writes ignored.
    static const uint32_t zero_csrs[] = { 0x306, 0x30A, 0x31A, 0x323, 0x33F, 0xB03, 0xB1F, 0xB83, 0xB9F, 0x7A3, 0x7A8, 0x7AA, 0x7C1 };
    for (unsigned i = 0; i < sizeof(zero_csrs) / sizeof(zero_csrs[0]); ++i) {
        one32(CSRRW(7, zero_csrs[i], 2));
        CHECK(tsr_rv_pc(RV) == 0x404u && X(7) == 0u && csr(zero_csrs[i]) == 0u, "csr 0x%03x reads 0", zero_csrs[i]);
    }
    setcsr(0x7C0, 0xFFu);        CHECK(csr(0x7C0) == 0xC0u, "cpuctrlsts writable bits");
    setcsr(0x7C0, 0);
    // Triggers: M-mode writes are ignored (dmode = 1); debug-mode writes (the API) work.
    setx(2, 1);
    one32(CSRRW(0, 0x7A0, 2));   CHECK(csr(0x7A0) == 0u, "tselect M-mode write ignored");
    setx(2, 0x28001044u);
    one32(CSRRW(0, 0x7A1, 2));   CHECK(csr(0x7A1) == 0x28001040u, "tdata1 M-mode write ignored");
    setcsr(0x7A0, 5);            CHECK(csr(0x7A0) == 1u, "tselect clamps to 1");
    setcsr(0x7A1, 0x28001044u);  CHECK(csr(0x7A1) == 0x28001044u, "tdata1 execute enable");
    setcsr(0x7A2, 0x1234u);      CHECK(csr(0x7A2) == 0x1234u, "tdata2");
    setcsr(0x7A0, 0);            CHECK(csr(0x7A1) == 0x28001040u && csr(0x7A2) == 0u, "trigger 0 separate");
    setcsr(0x7A0, 1); setcsr(0x7A1, 0);
    // dcsr / dpc / dscratch through the debug API.
    CHECK(csr(0x7B0) == 0x40000003u, "dcsr reset");
    setcsr(0x7B0, 0xFFFFFFFFu);  CHECK(csr(0x7B0) == 0x40009807u, "dcsr writable fields 0x%08x", csr(0x7B0));
    setcsr(0x7B0, 0x00000002u);  CHECK(csr(0x7B0) == 0x40000003u, "dcsr prv 2 -> M");
    setcsr(0x7B0, 0x00000000u);  CHECK(csr(0x7B0) == 0x40000000u, "dcsr prv U");
    setcsr(0x7B0, 0x40000003u);
    setcsr(0x7B1, 0x777u);       CHECK(csr(0x7B1) == 0x776u, "dpc bit 0");
    setcsr(0x7B2, 1); setcsr(0x7B3, 2); CHECK(csr(0x7B2) == 1u && csr(0x7B3) == 2u, "dscratch");
    CHECK(tsr_rv_csr_read(RV, 0x3A0, NULL) != 0 && tsr_rv_csr_write(RV, 0xF14, 1) != 0 && tsr_rv_csr_read(RV, 0x1000, NULL) != 0,
          "debug API refuses absent / read-only CSRs");
}

static void test_counters(void) {
    fresh();
    org(0x80);
    i32(CSRRS(1, 0xB02, 0)); i32(NOP); i32(NOP); i32(NOP); i32(CSRRS(2, 0xB02, 0));
    i32(CSRRS(3, 0xB00, 0)); i32(NOP); i32(NOP); i32(NOP); i32(CSRRS(4, 0xB00, 0));
    i32(CSRRW(0, 0xB02, 5)); i32(CSRRS(6, 0xB02, 0));
    setx(5, 100);
    tsr_rv_step(RV, 12);
    CHECK(X(2) - X(1) == 4u, "minstret delta %u", X(2) - X(1));
    CHECK(X(4) - X(3) == 4u, "mcycle delta %u", X(4) - X(3));
    CHECK(X(6) == 100u, "minstret write suppresses its own increment: %u", X(6));
    // Wrap into the high half.
    setcsr(0xB02, 0xFFFFFFFFu); setcsr(0xB82, 0);
    one32(NOP);
    CHECK(csr(0xB02) == 0u && csr(0xB82) == 1u, "minstret carry");
    setcsr(0xB00, 0xFFFFFFFEu); setcsr(0xB80, 7);
    one32(NOP); one32(NOP);
    CHECK(csr(0xB80) == 8u && csr(0xB00) == 0u, "mcycle carry");
    // Inhibit.
    setcsr(0x320, 4);
    uint32_t i0 = csr(0xB02), c0 = csr(0xB00);
    one32(NOP);
    CHECK(csr(0xB02) == i0 && csr(0xB00) == c0 + 1u, "IR inhibit");
    setcsr(0x320, 1);
    i0 = csr(0xB02); c0 = csr(0xB00);
    one32(NOP);
    tsr_rv_idle(RV, 10);
    CHECK(csr(0xB02) == i0 + 1u && csr(0xB00) == c0, "CY inhibit");
    setcsr(0x320, 0);
    c0 = csr(0xB00);
    tsr_rv_idle(RV, 10);
    CHECK(csr(0xB00) == c0 + 10u, "idle cycles count in mcycle");
    // Trapped instructions do not retire.
    i0 = csr(0xB02);
    one32(ECALL);
    CHECK(csr(0xB02) == i0, "ecall does not retire");
}

// ---------------------------------------------------------------------------
// Traps, mret, U-mode
// ---------------------------------------------------------------------------
static void test_traps(void) {
    fresh();
    setcsr(0x300, TSR_RV_MSTATUS_MIE);
    one32(ECALL);
    expect_trap("ecall", TSR_RV_EXC_ECALL_M, 0x400, 0);
    CHECK(csr(0x300) == (TSR_RV_MSTATUS_MPIE | TSR_RV_MSTATUS_MPP), "mstatus after trap 0x%08x", csr(0x300));
    one32(EBREAK);   expect_trap("ebreak", TSR_RV_EXC_BREAKPOINT, 0x400, 0);
    one16(C_EBREAK); expect_trap("c.ebreak", TSR_RV_EXC_BREAKPOINT, 0x400, 0);
    setx(10, 0x30000000u); setx(11, 0x1111);
    one32(LW(11, 10, 4)); expect_trap("load fault", TSR_RV_EXC_LOAD_FAULT, 0x400, 0x30000004u);
    CHECK(X(11) == 0x1111u, "faulting load writes no register");
    setx(10, TSR_RV_ITCM_BASE + TSR_RV_ITCM_SIZE);
    one32(SW(11, 10, 0)); expect_trap("store fault (ITCM reserved range)", TSR_RV_EXC_STORE_FAULT, 0x400, TSR_RV_ITCM_SIZE);
    run_at(DTCM, 1); expect_trap("fetch from DTCM", TSR_RV_EXC_FETCH_FAULT, DTCM, DTCM);
    run_at(CW, 1);   expect_trap("fetch from the control window", TSR_RV_EXC_FETCH_FAULT, CW, CW);
    run_at(SPM, 1);  expect_trap("fetch from the SPM window", TSR_RV_EXC_FETCH_FAULT, SPM, SPM);
    run_at(0x30000000u, 1); expect_trap("fetch unmapped", TSR_RV_EXC_FETCH_FAULT, 0x30000000u, 0x30000000u);
    // Second half of an uncompressed instruction outside ITCM: mtval = pc + 2.
    org(TSR_RV_ITCM_SIZE - 2u); i16(ADDI(1, 0, 1) & 0xFFFFu);
    run_at(TSR_RV_ITCM_SIZE - 2u, 1);
    expect_trap("fetch second half", TSR_RV_EXC_FETCH_FAULT, TSR_RV_ITCM_SIZE - 2u, TSR_RV_ITCM_SIZE);
    // A compressed instruction in the last parcel is fine.
    org(TSR_RV_ITCM_SIZE - 2u); i16(C_LI(1, 9));
    run_at(TSR_RV_ITCM_SIZE - 2u, 1);
    CHECK(X(1) == 9u && tsr_rv_pc(RV) == TSR_RV_ITCM_SIZE, "c.li in the last parcel");
    // mret: back to M with MIE restored.
    setcsr(0x341, 0x600); setcsr(0x300, TSR_RV_MSTATUS_MPIE | TSR_RV_MSTATUS_MPP);
    one32(MRET);
    CHECK(tsr_rv_pc(RV) == 0x600u && tsr_rv_priv(RV) == TSR_RV_PRIV_M, "mret to M");
    CHECK(csr(0x300) == (TSR_RV_MSTATUS_MIE | TSR_RV_MSTATUS_MPIE), "mret: MIE = MPIE, MPIE = 1, MPP = U: 0x%08x", csr(0x300));
    // mret to U (MPP = U after the previous mret), MPRV cleared.
    setcsr(0x341, 0x600); setcsr(0x300, TSR_RV_MSTATUS_MPRV);
    one32(MRET);
    CHECK(tsr_rv_priv(RV) == TSR_RV_PRIV_U && csr(0x300) == TSR_RV_MSTATUS_MPIE, "mret to U, MPRV cleared");
    one32(CSRRS(1, 0x340, 0));
    expect_trap("csr in U", TSR_RV_EXC_ILLEGAL, 0x400, CSRRS(1, 0x340, 0));
    CHECK(tsr_rv_priv(RV) == TSR_RV_PRIV_M && (csr(0x300) & TSR_RV_MSTATUS_MPP) == 0u, "trap from U: MPP = U");
    setcsr(0x341, 0x600); one32(MRET);
    one32(ECALL);    expect_trap("ecall from U", TSR_RV_EXC_ECALL_U, 0x400, 0);
    setcsr(0x341, 0x600); one32(MRET);
    one32(MRET);     expect_trap("mret in U", TSR_RV_EXC_ILLEGAL, 0x400, MRET);
    setcsr(0x341, 0x600); one32(MRET);
    one32(DRET);     expect_trap("dret outside debug mode", TSR_RV_EXC_ILLEGAL, 0x400, DRET);
    setcsr(0x341, 0x600); setcsr(0x300, TSR_RV_MSTATUS_TW); one32(MRET);
    one32(WFI);      expect_trap("wfi in U with TW", TSR_RV_EXC_ILLEGAL, 0x400, WFI);
    setcsr(0x341, 0x600); setcsr(0x300, 0); one32(MRET);
    one32(WFI);
    CHECK(tsr_rv_sleeping(RV) && tsr_rv_pc(RV) == 0x404u, "wfi in U without TW sleeps");
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_WFI, "asleep");
    // Double-fault status bits (cpuctrlsts), cleared sync_exc_seen by mret.
    fresh();
    one32(ECALL);
    CHECK(csr(0x7C0) == 0x40u, "sync_exc_seen");
    one32(ECALL);
    CHECK(csr(0x7C0) == 0xC0u, "double_fault_seen");
    setcsr(0x341, 0x600); one32(MRET);
    CHECK(csr(0x7C0) == 0x80u, "mret clears sync_exc_seen");
}

// ---------------------------------------------------------------------------
// Misaligned accesses (split into two word transactions, Ibex) and access rules (CC §4.1)
// ---------------------------------------------------------------------------
static void test_misaligned(void) {
    fresh();
    uint8_t * d = mem(DTCM, 32);
    for (int i = 0; i < 32; ++i) d[i] = (uint8_t)i;
    setx(10, DTCM);
    one32(LW(1, 10, 1)); CHECK(X(1) == 0x04030201u, "lw +1");
    one32(LW(2, 10, 3)); CHECK(X(2) == 0x06050403u, "lw +3");
    one32(LH(3, 10, 3)); CHECK(X(3) == 0x0403u, "lh +3");
    one32(LHU(4, 10, 1)); CHECK(X(4) == 0x0201u, "lhu +1 (inside a word)");
    setx(5, 0xAABBCCDDu);
    one32(SW(5, 10, 6)); CHECK(d[6] == 0xDD && d[7] == 0xCC && d[8] == 0xBB && d[9] == 0xAA && d[5] == 5 && d[10] == 10, "sw +6");
    setx(6, 0x1122);
    one32(SH(6, 10, 11)); CHECK(d[11] == 0x22 && d[12] == 0x11, "sh +11");
    d[15] = 0x80; d[16] = 0xFF;
    one32(LH(7, 10, 15)); CHECK(X(7) == 0xFFFFFF80u, "lh split sign 0x%08x", X(7));
    // Cost: split = lsu 1 + 1 + 1 (two TCM transactions) + 1.
    uint64_t c0 = tsr_rv_cycles(RV);
    one32(LW(1, 10, 1));
    CHECK(tsr_rv_cycles(RV) - c0 == 4u, "split load cycles %llu", (unsigned long long)(tsr_rv_cycles(RV) - c0));
    // Two SPM transactions in address order.
    for (int i = 0; i < 8; ++i) E.spm[i] = (uint8_t)(0x10 + i);
    E.nlog = 0; NRET = 0;
    setx(10, SPM);
    one32(LW(8, 10, 1));
    CHECK(X(8) == 0x14131211u, "SPM split load");
    CHECK(E.nlog == 2 && E.log[0].off == 1u && E.log[0].n == 3u && E.log[1].off == 4u && E.log[1].n == 1u,
          "SPM split transactions (%d)", E.nlog);
    CHECK(NRET == 1 && RET[0].mem_addr == SPM + 1u && RET[0].mem_size == 4u && RET[0].mem_val == 0x14131211u, "retire record of a split load");
    // Misaligned word to the control window: the partial transactions fault; the env is not called.
    E.nlog = 0;
    setx(10, CW);
    one32(LW(9, 10, 2)); expect_trap("cw misaligned", TSR_RV_EXC_LOAD_FAULT, 0x400, CW + 2u);
    CHECK(E.nlog == 0, "no control-window access");
    // Second half faults: mtval = its word address; a store's first half is written (not atomic).
    uint32_t end = DTCM + TSR_RV_DTCM_SIZE;
    w32(end - 4u, 0x01020304u);
    setx(10, end - 2u); setx(11, 0xA1B2C3D4u); setx(12, 0x1234);
    one32(SW(11, 10, 0)); expect_trap("split store, second half faults", TSR_RV_EXC_STORE_FAULT, 0x400, end);
    CHECK(r32(end - 4u) == 0xC3D40304u, "first half written: 0x%08x", r32(end - 4u));
    one32(LW(12, 10, 0)); expect_trap("split load, second half faults", TSR_RV_EXC_LOAD_FAULT, 0x400, end);
    CHECK(X(12) == 0x1234u, "no register write");
    // First half faults: mtval = the original address; the second half is still issued.
    w32(DTCM, 0);
    setx(10, DTCM - 2u);
    one32(SW(11, 10, 0)); expect_trap("split store, first half faults", TSR_RV_EXC_STORE_FAULT, 0x400, DTCM - 2u);
    CHECK(r32(DTCM) == 0x0000A1B2u, "second half written: 0x%08x", r32(DTCM));
}

static void test_access(void) {
    fresh();
    // Control window: word only; reads / writes reach the env with the offset.
    setx(10, CW); setx(11, 0x99);
    const uint32_t bad_cw[] = { LB(1, 10, 0), LH(1, 10, 0), LBU(1, 10, 3), LHU(1, 10, 2), SB(11, 10, 0), SH(11, 10, 2) };
    for (unsigned i = 0; i < sizeof(bad_cw) / sizeof(bad_cw[0]); ++i) {
        E.nlog = 0;
        one32(bad_cw[i]);
        CHECK(csr(0x342) == (i < 4 ? TSR_RV_EXC_LOAD_FAULT : TSR_RV_EXC_STORE_FAULT) && E.nlog == 0, "cw partial access %u faults", i);
    }
    E.nlog = 0;
    one32(SW(11, 10, 0x140)); one32(LW(1, 10, 0x140));
    CHECK(E.nlog == 2 && E.log[0].write && E.log[0].off == 0x140u && E.log[0].val == 0x99u && !E.log[1].write && X(1) == 0x99u, "cw word access");
    one32(SW(11, 10, TSR_CW_LQ_POP)); expect_trap("LQ_POP fault", TSR_RV_EXC_STORE_FAULT, 0x400, CW + TSR_CW_LQ_POP);
    E.lq_avail = 1;
    one32(SW(11, 10, TSR_CW_LQ_POP)); CHECK(tsr_rv_pc(RV) == 0x404u && E.lq_avail == 0u, "LQ_POP ok");
    // Core window: word loads only; LOAD_DATA and DMI read 0 without reaching the core.
    setx(10, CORE);
    E.nlog = 0;
    setx(13, CORE + 0x800u);
    one32(LW(1, 13, 0x1C0)); CHECK(X(1) == (0xC0DE0000u | 0x9C0u) && E.nlog == 1, "core window read (SREG0)");
    one32(LW(1, 10, 0x0C4)); CHECK(X(1) == 0u, "LOAD_DATA reads 0");
    setx(12, CORE + 0xC00u);
    one32(LW(1, 12, 0)); CHECK(X(1) == 0u, "DMI reads 0");
    one32(LW(1, 12, 0x1FC)); CHECK(X(1) == 0u, "DMI end reads 0");
    one32(LW(1, 12, 0x200)); CHECK(X(1) == (0xC0DE0000u | 0xE00u), "Q_* reads through");
    CHECK(E.nlog == 2, "only two core-window calls (%d)", E.nlog);
    one32(SW(11, 10, 0)); expect_trap("core window store", TSR_RV_EXC_STORE_FAULT, 0x400, CORE);
    one32(LH(1, 10, 0)); expect_trap("core window half", TSR_RV_EXC_LOAD_FAULT, 0x400, CORE);
    // Debug module memory: word data, fetch allowed.
    setx(10, DM + 0x300u); setx(11, 0xD00DFEEDu);
    one32(SW(11, 10, 0)); one32(LW(1, 10, 0)); CHECK(X(1) == 0xD00DFEEDu, "DM word access");
    one32(LB(1, 10, 0)); expect_trap("DM byte", TSR_RV_EXC_LOAD_FAULT, 0x400, DM + 0x300u);
    w32(DM + 0x400u, ADDI(13, 0, 42));
    run_at(DM + 0x400u, 1); CHECK(X(13) == 42u, "fetch from the DM region");
    // SPM exact window: byte / half / word; beyond SPM_BYTES faults.
    setx(10, SPM); setx(11, 0x89ABCDEFu);
    one32(SW(11, 10, 64)); one32(LBU(1, 10, 65)); one32(LH(2, 10, 66)); one32(SB(11, 10, 70)); one32(SH(11, 10, 72));
    CHECK(X(1) == 0xCDu && X(2) == 0xFFFF89ABu && E.spm[70] == 0xEF && E.spm[72] == 0xEF && E.spm[73] == 0xCD, "SPM window");
    setx(12, SPM + SPM_TEST_BYTES);
    one32(LW(1, 12, 0)); expect_trap("beyond SPM_BYTES", TSR_RV_EXC_LOAD_FAULT, 0x400, SPM + SPM_TEST_BYTES);
    setx(12, SPMLB + SPM_TEST_BYTES);
    one32(SW(1, 12, 0)); expect_trap("beyond SPM_BYTES (alias)", TSR_RV_EXC_STORE_FAULT, 0x400, SPMLB + SPM_TEST_BYTES);
    // Holes of the map.
    static const uint32_t holes[] = { TSR_RV_ITCM_SIZE, TSR_RV_DTCM_LIMIT - 4u, DTCM + TSR_RV_DTCM_SIZE, DM + TSR_RV_DM_SIZE,
                                      TSR_RV_CORE_LIMIT, 0x30000000u, SPM - 4u, SPMLB - 4u, 0xFFFFFFFCu };
    for (unsigned i = 0; i < sizeof(holes) / sizeof(holes[0]); ++i) {
        setx(12, holes[i]);
        one32(LW(1, 12, 0)); expect_trap("hole", TSR_RV_EXC_LOAD_FAULT, 0x400, holes[i]);
    }
    CHECK(tsr_bus_mem(B, TSR_RV_ITCM_SIZE - 4u, 4) != NULL && tsr_bus_mem(B, TSR_RV_ITCM_SIZE - 4u, 8) == NULL &&
          tsr_bus_mem(B, CW, 4) == NULL, "tsr_bus_mem bounds");
}

static void test_line_buffer(void) {
    fresh();
    setx(10, SPMLB + 3u * 64u);
    for (int i = 0; i < 64; ++i) E.spm[3 * 64 + i] = (uint8_t)i;
    E.nlog = 0;
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x07060504u && E.nlog == 1 && E.log[0].off == 192u && E.log[0].n == 64u, "line fill");
    E.spm[3 * 64 + 4] = 0xEE;                   // an op writes the line behind the hart's back
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x07060504u && E.nlog == 1, "hit returns the buffered (stale) line");
    uint64_t c0 = tsr_rv_cycles(RV);
    one32(LW(1, 10, 8));
    CHECK(tsr_rv_cycles(RV) - c0 == 2u, "hit latency");
    setx(11, CW);
    one32(LW(2, 11, TSR_CW_DONE_SEQ(0)));      // observing a completion invalidates (CC §10.5)
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x070605EEu, "control-window access invalidates: 0x%08x", X(1));
    E.spm[3 * 64 + 5] = 0xDD;
    setx(12, SPM); setx(13, 0);
    one32(SB(13, 12, 0));                       // any store to an SPM alias invalidates
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x0706DDEEu, "store invalidates");
    E.spm[3 * 64 + 6] = 0xCC;
    E.irq = TSR_RV_LINE_FAST(0);                // rising edge of an interrupt line (mie = 0: not taken)
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x07CCDDEEu, "irq rising edge invalidates");
    E.spm[3 * 64 + 7] = 0xBB;
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0x07CCDDEEu, "a held level does not");
    setx(13, 0x11223344u);
    one32(SW(13, 10, 8));                       // store through the alias: write-through
    CHECK(E.spm[3 * 64 + 8] == 0x44, "alias store writes through");
    one32(LW(1, 10, 4));
    CHECK(X(1) == 0xBBCCDDEEu, "alias store invalidated the buffer");
}

// ---------------------------------------------------------------------------
// Interrupts and WFI
// ---------------------------------------------------------------------------
static void test_interrupts(void) {
    fresh();
    setcsr(0x304, 0x7FFF0888u);
    setcsr(0x300, TSR_RV_MSTATUS_MIE);
    org(0x500); for (int i = 0; i < 8; ++i) i32(NOP);
    struct { uint32_t lines, cause; } pr[] = {
        { TSR_RV_LINE_MTI, TSR_RV_IRQ_MTI },
        { TSR_RV_LINE_MTI | TSR_RV_LINE_MSI, TSR_RV_IRQ_MSI },
        { TSR_RV_LINE_MTI | TSR_RV_LINE_MSI | TSR_RV_LINE_MEI, TSR_RV_IRQ_MEI },
        { TSR_RV_LINE_MEI | TSR_RV_LINE_FAST(14), TSR_RV_IRQ_FAST(14) },
        { TSR_RV_LINE_FAST(3) | TSR_RV_LINE_FAST(1) | TSR_RV_LINE_MEI, TSR_RV_IRQ_FAST(1) },
        { TSR_RV_LINE_FAST(0) | TSR_RV_LINE_FAST(5), TSR_RV_IRQ_FAST(0) },
    };
    for (unsigned i = 0; i < sizeof(pr) / sizeof(pr[0]); ++i) {
        setcsr(0x300, TSR_RV_MSTATUS_MIE);
        E.irq = pr[i].lines; NRET = 0;
        run_at(0x504, 1);
        uint32_t entry = VEC + 4u * pr[i].cause;
        CHECK(csr(0x342) == (TSR_RV_MCAUSE_IRQ | pr[i].cause), "irq %u: mcause 0x%08x", i, csr(0x342));
        CHECK(csr(0x341) == 0x504u && csr(0x343) == 0u, "irq %u: mepc / mtval", i);
        CHECK(NRET == 1 && RET[0].pc == entry && RET[0].intr == 1, "irq %u: first handler instruction at 0x%x", i, NRET ? RET[0].pc : 0);
        CHECK(csr(0x300) == (TSR_RV_MSTATUS_MPIE | TSR_RV_MSTATUS_MPP), "irq %u: mstatus", i);
    }
    // Masked: mstatus.MIE = 0, or the mie bit clear.
    E.irq = TSR_RV_LINE_FAST(0);
    setcsr(0x300, 0);
    run_at(0x504, 1); CHECK(tsr_rv_pc(RV) == 0x508u, "MIE = 0: not taken");
    setcsr(0x300, TSR_RV_MSTATUS_MIE); setcsr(0x304, TSR_RV_LINE_FAST(1));
    run_at(0x504, 1); CHECK(tsr_rv_pc(RV) == 0x508u, "mie bit clear: not taken");
    // U-mode: taken whatever mstatus.MIE.
    setcsr(0x304, TSR_RV_LINE_FAST(0)); setcsr(0x300, 0); setcsr(0x341, 0x504);
    one32(MRET);
    CHECK(tsr_rv_priv(RV) == TSR_RV_PRIV_U, "in U");
    NRET = 0;
    tsr_rv_step(RV, 1);
    CHECK(csr(0x342) == (TSR_RV_MCAUSE_IRQ | 16u) && tsr_rv_priv(RV) == TSR_RV_PRIV_M && csr(0x341) == 0x504u, "U-mode interrupt with MIE = 0");
    // WFI: sleeps until (mip & mie) != 0 whatever MIE; continues after the WFI with MIE = 0.
    fresh();
    org(0x600); i32(WFI); i32(ADDI(1, 0, 1)); i32(NOP);
    setcsr(0x304, TSR_RV_LINE_FAST(2));
    run_at(0x600, 1);
    CHECK(tsr_rv_sleeping(RV) && tsr_rv_pc(RV) == 0x604u, "wfi sleeps");
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_WFI && X(1) == 0u, "stays asleep");
    E.irq = TSR_RV_LINE_FAST(3);
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_WFI, "a disabled line does not wake");
    E.irq = TSR_RV_LINE_FAST(2);
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_BUDGET && X(1) == 1u && !tsr_rv_sleeping(RV), "wakes with MIE = 0, no trap");
    // With MIE = 1 the interrupt is taken with mepc = the instruction after the WFI.
    E.irq = 0; setcsr(0x300, TSR_RV_MSTATUS_MIE);
    run_at(0x600, 1);
    E.irq = TSR_RV_LINE_FAST(2);
    tsr_rv_step(RV, 1);
    CHECK(csr(0x342) == (TSR_RV_MCAUSE_IRQ | 18u) && csr(0x341) == 0x604u, "wfi wake into the handler");
    // WFI with an enabled interrupt already pending does not sleep (MIE = 0).
    setcsr(0x300, 0);
    run_at(0x600, 2);
    CHECK(X(1) == 1u && tsr_rv_pc(RV) == 0x608u, "wfi falls through");
}

// ---------------------------------------------------------------------------
// Debug mode (DB §3, Debug Spec 0.13.2 as Ibex implements it)
// ---------------------------------------------------------------------------
static void test_debug(void) {
    fresh();
    org(0x700);
    for (int i = 0; i < 8; ++i) i32(ADDI(1, 1, 1));
    run_at(0x700, 2);
    tsr_rv_set_debug_req(RV, 1);
    CHECK(tsr_rv_step(RV, 5) == TSR_RV_STOP_HALTED, "halt request");
    CHECK(tsr_rv_halted(RV) && tsr_rv_debug_mode(RV) && X(1) == 2u, "parked after two instructions");
    CHECK(csr(0x7B1) == 0x708u && (csr(0x7B0) & 0x1C3u) == ((3u << 6) | 3u), "dpc, dcsr.cause haltreq, prv M");
    CHECK(tsr_rv_pc(RV) == TSR_RV_DM_HALT_ADDR, "pc in the debug ROM");
    CHECK(NDBG == 1 && DBG_EV[0] == TSR_RV_DBG_EV_ENTER && DBG_ARG[0] == 3u, "debug event ENTER");
    CHECK(tsr_rv_step(RV, 5) == TSR_RV_STOP_HALTED && X(1) == 2u, "stays parked");
    tsr_rv_set_debug_req(RV, 0);
    // Abstract register access.
    tsr_rv_set_reg(RV, 1, 100);
    CHECK(tsr_rv_debug_resume(RV) == 0 && !tsr_rv_debug_mode(RV), "resume");
    tsr_rv_step(RV, 1);
    CHECK(X(1) == 101u && tsr_rv_pc(RV) == 0x70Cu, "resumed at dpc");
    CHECK(tsr_rv_debug_resume(RV) == -1 && tsr_rv_debug_exec(RV, DM) == -1, "resume / exec need a parked hart");
    // Single step: one instruction, then cause step.
    tsr_rv_set_debug_req(RV, 1); tsr_rv_step(RV, 1); tsr_rv_set_debug_req(RV, 0);
    setcsr(0x7B0, csr(0x7B0) | (1u << 2));
    tsr_rv_debug_resume(RV);
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_HALTED && X(1) == 102u, "one step");
    CHECK(csr(0x7B1) == 0x710u && ((csr(0x7B0) >> 6) & 7u) == 4u, "step: dpc, cause 4");
    // Step over an ecall: the trap is taken, dpc = the handler.
    w32(0x710, ECALL);
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 10);
    CHECK(csr(0x7B1) == VEC && csr(0x342) == TSR_RV_EXC_ECALL_M && csr(0x341) == 0x710u, "step into a trap");
    // Interrupts stay disabled while stepping.
    w32(0x710, ADDI(1, 1, 1));
    setcsr(0x7B1, 0x710); setcsr(0x304, TSR_RV_LINE_FAST(0)); setcsr(0x300, TSR_RV_MSTATUS_MIE);
    E.irq = TSR_RV_LINE_FAST(0);
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 10);
    CHECK(csr(0x7B1) == 0x714u && X(1) == 103u && csr(0x342) == TSR_RV_EXC_ECALL_M, "no interrupt during a step");
    setcsr(0x7B0, csr(0x7B0) & ~(1u << 2));
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 1);
    CHECK(csr(0x342) == (TSR_RV_MCAUSE_IRQ | 16u) && csr(0x341) == 0x714u, "interrupt after the resume");
    E.irq = 0;
    // ebreak with dcsr.ebreakm: debug entry, dpc = the ebreak; no M-mode trap.
    fresh();
    setcsr(0x7B0, 0x40008003u);
    uint32_t mc = csr(0x342);
    one32(EBREAK);
    CHECK(tsr_rv_halted(RV) && csr(0x7B1) == 0x400u && ((csr(0x7B0) >> 6) & 7u) == 1u && csr(0x342) == mc, "ebreakm");
    tsr_rv_debug_resume(RV);
    one16(C_EBREAK);
    CHECK(tsr_rv_halted(RV) && csr(0x7B1) == 0x400u, "c.ebreak with ebreakm");
    // ebreaku in U-mode; ebreakm does not apply there.
    setcsr(0x7B0, 0x40008000u);     // ebreakm, prv U
    setcsr(0x7B1, 0x400);
    w32(0x400, EBREAK);
    tsr_rv_debug_resume(RV);
    CHECK(tsr_rv_priv(RV) == TSR_RV_PRIV_U, "resumed into U");
    tsr_rv_step(RV, 1);
    CHECK(!tsr_rv_debug_mode(RV) && csr(0x342) == TSR_RV_EXC_BREAKPOINT, "ebreak in U without ebreaku traps");
    setcsr(0x7B0, 0x40001000u);
    tsr_rv_set_debug_req(RV, 1); tsr_rv_step(RV, 1); tsr_rv_set_debug_req(RV, 0);
    setcsr(0x7B0, 0x40001000u); setcsr(0x7B1, 0x400);
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 1);
    CHECK(tsr_rv_halted(RV) && (csr(0x7B0) & 3u) == 0u && ((csr(0x7B0) >> 6) & 7u) == 1u, "ebreaku: debug entry, dcsr.prv U");
    // Execute trigger: halts before the matching instruction.
    fresh();
    org(0x800); for (int i = 0; i < 4; ++i) i32(ADDI(2, 2, 1));
    setcsr(0x7A0, 1); setcsr(0x7A2, 0x808); setcsr(0x7A1, 0x28001044u);
    tsr_rv_set_pc(RV, 0x800);
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_HALTED && X(2) == 2u, "trigger halts before 0x808");
    CHECK(csr(0x7B1) == 0x808u && ((csr(0x7B0) >> 6) & 7u) == 2u, "trigger: dpc, cause 2");
    // Program buffer: runs in debug mode until ebreak; GPR write; dpc / dcsr unchanged.
    org(DM + 0x360u); i32(FENCE_I); i32(ADDI(5, 0, 42)); i32(EBREAK);
    NDBG = 0;
    CHECK(tsr_rv_debug_exec(RV, DM + 0x360u) == 0 && !tsr_rv_halted(RV) && tsr_rv_debug_mode(RV), "progbuf started");
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_HALTED && X(5) == 42u, "progbuf ran");
    CHECK(NDBG == 1 && DBG_EV[0] == TSR_RV_DBG_EV_PARK && DBG_ARG[0] == 0u, "PARK event");
    CHECK(csr(0x7B1) == 0x808u && ((csr(0x7B0) >> 6) & 7u) == 2u, "ebreak in debug mode leaves dpc / dcsr");
    // Exception in the program buffer: parks, no CSR changes.
    org(DM + 0x380u); i32(0);
    uint32_t mepc = csr(0x341);
    mc = csr(0x342);
    NDBG = 0;
    tsr_rv_debug_exec(RV, DM + 0x380u);
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_HALTED && NDBG == 1 && DBG_ARG[0] == 1u, "exception in debug mode");
    CHECK(csr(0x341) == mepc && csr(0x342) == mc, "no CSR update");
    // The trigger does not fire in debug mode; dret from the program buffer resumes at dpc.
    org(DM + 0x3A0u); i32(DRET);
    tsr_rv_debug_exec(RV, DM + 0x3A0u);
    tsr_rv_step(RV, 1);
    CHECK(!tsr_rv_debug_mode(RV) && tsr_rv_pc(RV) == 0x808u, "dret");
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_HALTED, "trigger fires again outside debug mode");
    setcsr(0x7A1, 0);
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 1);
    CHECK(X(2) == 3u, "trigger disabled");
    // Halt request wakes a sleeping hart (dpc = after the WFI).
    org(0x900); i32(WFI); i32(NOP);
    run_at(0x900, 1);
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_WFI, "asleep");
    tsr_rv_set_debug_req(RV, 1);
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_HALTED && csr(0x7B1) == 0x904u, "halt from WFI");
    tsr_rv_set_debug_req(RV, 0);
    tsr_rv_debug_resume(RV);
    // WFI is a no-op in debug mode.
    org(DM + 0x3C0u); i32(WFI); i32(ADDI(6, 0, 6)); i32(EBREAK);
    tsr_rv_set_debug_req(RV, 1); tsr_rv_step(RV, 1); tsr_rv_set_debug_req(RV, 0);
    tsr_rv_debug_exec(RV, DM + 0x3C0u);
    CHECK(tsr_rv_step(RV, 10) == TSR_RV_STOP_HALTED && X(6) == 6u, "wfi no-op in debug mode");
    // Debug-mode instructions retire with dbg = 1 in their records.
    NRET = 0;
    tsr_rv_debug_exec(RV, DM + 0x3C0u);
    tsr_rv_step(RV, 10);
    CHECK(NRET == 3 && RET[1].dbg == 1 && RET[1].rd == 6u, "debug-mode records");
}

// ---------------------------------------------------------------------------
// Waiting and yielding stores (CC §7.3), retire records
// ---------------------------------------------------------------------------
static void test_wait_yield(void) {
    fresh();
    setx(10, CW); setx(11, 0x1);
    org(0xA00); i32(ADDI(1, 0, 1)); i32(SW(11, 10, TSR_CW_ISSUE_PUSH)); i32(ADDI(2, 0, 2));
    setcsr(0x304, TSR_RV_LINE_FAST(0)); setcsr(0x300, TSR_RV_MSTATUS_MIE);
    NRET = 0;
    CHECK(run_at(0xA00, 5) == TSR_RV_STOP_WAIT, "push without a credit waits");
    CHECK(tsr_rv_waiting(RV) && tsr_rv_pc(RV) == 0xA04u && X(1) == 1u && X(2) == 0u && NRET == 1, "nothing retired by the waiting store");
    E.irq = TSR_RV_LINE_FAST(0);
    tsr_rv_set_debug_req(RV, 1);
    CHECK(tsr_rv_step(RV, 5) == TSR_RV_STOP_WAIT && !tsr_rv_debug_mode(RV) && csr(0x342) == 0u, "no interrupt or halt while waiting");
    tsr_rv_idle(RV, 100);
    CHECK(tsr_rv_wait_cycles(RV) == 100u, "wait cycles");
    tsr_rv_freeze_wait(RV, 1);
    tsr_rv_idle(RV, 50);
    tsr_rv_freeze_wait(RV, 0);
    CHECK(tsr_rv_wait_cycles(RV) == 100u, "frozen wait count (dbg_stop)");
    E.credit = 1;
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_BUDGET && E.pushes == 1u && !tsr_rv_waiting(RV) && tsr_rv_pc(RV) == 0xA08u, "push completes");
    CHECK(NRET == 2 && RET[1].mem_write && RET[1].mem_addr == CW + TSR_CW_ISSUE_PUSH, "one record for the push");
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_HALTED && ((csr(0x7B0) >> 6) & 7u) == 3u, "halt request first after the push");
    tsr_rv_set_debug_req(RV, 0);
    tsr_rv_debug_resume(RV);
    tsr_rv_step(RV, 1);
    CHECK(csr(0x342) == (TSR_RV_MCAUSE_IRQ | 16u) && csr(0x341) == 0xA08u, "then the interrupt");
    E.irq = 0;
    // YIELD: the instruction is re-executed by the next step; nothing happened before.
    E.yield_n = 1; E.nlog = 0;
    setx(12, 0);
    org(0xB00); i32(LW(12, 10, TSR_CW_QSTATUS)); i32(NOP);
    E.cw[TSR_CW_QSTATUS / 4] = 0x55;
    CHECK(run_at(0xB00, 2) == TSR_RV_STOP_YIELD && tsr_rv_pc(RV) == 0xB00u && E.nlog == 0 && X(12) == 0u, "yield");
    CHECK(tsr_rv_step(RV, 1) == TSR_RV_STOP_BUDGET && X(12) == 0x55u && E.nlog == 1, "retried");
}

static void test_retire(void) {
    fresh();
    setx(10, DTCM); setx(11, 0xABCD1234u);
    org(0xC00);
    i32(ADDI(1, 0, 7)); i32(SW(11, 10, 8)); i32(LH(2, 10, 8)); i32(CSRRWI(3, 0x340, 9)); i32(ECALL);
    NRET = 0;
    run_at(0xC00, 5);
    CHECK(NRET == 5, "five records");
    CHECK(RET[0].rd == 1u && RET[0].rd_val == 7u && RET[0].len == 4u && RET[0].next_pc == 0xC04u && RET[0].priv == 3u, "alu record");
    CHECK(RET[1].mem_write == 1u && RET[1].mem_size == 4u && RET[1].mem_addr == DTCM + 8u && RET[1].mem_val == 0xABCD1234u && RET[1].rd == 0u, "store record");
    CHECK(RET[2].mem_write == 0u && RET[2].mem_size == 2u && RET[2].mem_val == 0x1234u && RET[2].rd == 2u && RET[2].rd_val == 0x1234u, "load record");
    CHECK(RET[3].csr_written && RET[3].csr == 0x340u && RET[3].csr_val == 9u && RET[3].rd == 3u, "csr record");
    CHECK(RET[4].trap && RET[4].cause == TSR_RV_EXC_ECALL_M && RET[4].next_pc == VEC && RET[4].insn == ECALL, "trap record");
    for (int i = 0; i < 5; ++i) CHECK(RET[i].order == RET[0].order + (uint64_t)i, "order");
}

// ---------------------------------------------------------------------------
// SBA (CC §4.3)
// ---------------------------------------------------------------------------
static void test_sba(void) {
    fresh();
    uint32_t v = 0;
    tsr_bus_set_rv_run(B, 0);
    CHECK(tsr_bus_sba_write(B, 0x100, 4, 1) == -1 && tsr_bus_sba_write(B, DTCM, 4, 1) == -1, "TCM writes refused while RV_RUN = 0");
    w32(0x100, 0x12345678u);
    CHECK(tsr_bus_sba_read(B, 0x100, 4, &v) == 0 && v == 0x12345678u, "TCM read");
    CHECK(tsr_bus_sba_read(B, 0x102, 2, &v) == 0 && v == 0x1234u, "half read");
    tsr_bus_set_rv_run(B, 1);
    CHECK(tsr_bus_sba_write(B, DTCM + 4, 4, 0xFEEDu) == 0 && r32(DTCM + 4) == 0xFEEDu, "TCM write while RV_RUN = 1");
    CHECK(tsr_bus_sba_write(B, DTCM + 1, 1, 0xAB) == 0 && r32(DTCM) == 0xAB00u, "byte write");
    CHECK(tsr_bus_sba_read(B, DM, 4, &v) == -1 && tsr_bus_sba_write(B, DM, 4, 0) == -1, "DM range refused");
    E.nlog = 0;
    CHECK(tsr_bus_sba_write(B, CW + TSR_CW_ISSUE_W(0), 4, 1) == -1 && tsr_bus_sba_write(B, CW + TSR_CW_QCTRL, 4, 1) == -1 && E.nlog == 0,
          "control-window writes refused (D15)");
    E.cw[TSR_CW_ISSUE_PIDX / 4] = 77;
    CHECK(tsr_bus_sba_read(B, CW + TSR_CW_ISSUE_PIDX, 4, &v) == 0 && v == 77u, "control-window read");
    CHECK(tsr_bus_sba_write(B, CORE, 4, 0) == -1, "core-window write refused");
    CHECK(tsr_bus_sba_read(B, CORE + TSR_CSR_LOAD_DATA, 4, &v) == 0 && v == 0u, "core-window LOAD_DATA reads 0");
    CHECK(tsr_bus_sba_write(B, SPM + 8, 4, 0xA5A5A5A5u) == 0 && E.spm[8] == 0xA5, "SPM write");
    CHECK(tsr_bus_sba_read(B, SPMLB + 8, 2, &v) == 0 && v == 0xA5A5u, "SPM alias read");
    CHECK(tsr_bus_sba_read(B, 0x101, 2, &v) == -1 && tsr_bus_sba_read(B, 0x100, 3, &v) == -1 && tsr_bus_sba_read(B, 0x30000000u, 4, &v) == -1,
          "misaligned / bad size / unmapped");
    // An SBA store to the SPM invalidates the hart's line buffer.
    setx(10, SPMLB);
    one32(LW(1, 10, 0));
    E.spm[0] = 0x5A;
    tsr_bus_sba_write(B, SPM + 32, 1, 0);
    one32(LW(1, 10, 0));
    CHECK((X(1) & 0xFFu) == 0x5Au, "SBA store invalidates the line buffer");
}

// Data-transaction observer (DB §3.7 RV watchpoints): every transaction once, with the pc of the
// accessing instruction; a halt request from it stops the hart right after that instruction.
static struct { uint32_t addr, n, val, pc; int w, fault; } WL[8];
static int NWL, WATCH_HALT;
static void watch_cb(void * ctx, uint32_t addr, uint32_t n, int is_write, uint32_t val, int fault) {
    (void)ctx;
    if (NWL < 8) { WL[NWL].addr = addr; WL[NWL].n = n; WL[NWL].w = is_write; WL[NWL].val = val; WL[NWL].fault = fault; WL[NWL].pc = tsr_rv_pc(RV); ++NWL; }
    if (WATCH_HALT && is_write) tsr_rv_set_debug_req(RV, 1);
}

static void test_watch(void) {
    fresh();
    tsr_bus_set_watch(B, watch_cb, NULL);
    setx(10, DTCM); setx(11, 0x11223344u); setx(12, CW);
    org(0xD00); i32(LW(1, 10, 0)); i32(SW(11, 10, 6)); i32(LW(2, 12, 2)); i32(NOP);
    NWL = 0;
    run_at(0xD00, 3);
    CHECK(NWL == 5, "five transactions (%d)", NWL);
    CHECK(WL[0].addr == DTCM && WL[0].n == 4 && !WL[0].w && WL[0].pc == 0xD00u, "load");
    CHECK(WL[1].addr == DTCM + 6u && WL[1].n == 2 && WL[1].w && WL[1].val == 0x3344u && WL[1].pc == 0xD04u, "split store, first half");
    CHECK(WL[2].addr == DTCM + 8u && WL[2].n == 2 && WL[2].w && WL[2].val == 0x1122u, "split store, second half");
    CHECK(WL[3].addr == CW + 2u && WL[3].n == 2 && WL[3].fault && WL[4].addr == CW + 4u && WL[4].fault,
          "both halves of a faulting split access are issued and reported");
    // Halt right after the watched store.
    WATCH_HALT = 1; NWL = 0;
    run_at(0xD04, 5);
    WATCH_HALT = 0;
    CHECK(tsr_rv_halted(RV) && csr(0x7B1) == 0xD08u && ((csr(0x7B0) >> 6) & 7u) == 3u, "halted after the store, dpc 0x%x", csr(0x7B1));
    tsr_rv_set_debug_req(RV, 0);
    tsr_bus_set_watch(B, NULL, NULL);
}

// ---------------------------------------------------------------------------
// Test profile: HTIF, ELF loader
// ---------------------------------------------------------------------------
static void test_htif(void) {
    tsr_bus * t = tsr_bus_create_test(0x80000000u, 0x10000);
    tsr_rv * r = tsr_rv_create(tsr_bus_iface(t));
    tsr_bus_attach(t, r);
    tsr_bus_set_htif(t, 0x80001000u, 0x80001040u);
    tsr_rv_reset(r, 0);
    tsr_rv_set_pc(r, 0x80000000u);
    uint8_t * m = tsr_bus_mem(t, 0x80000000u, 0x2000);
    CHECK(m != NULL && tsr_bus_mem(t, 0x7FFFFFFCu, 4) == NULL && tsr_bus_mem(t, 0x80010000u, 1) == NULL, "test RAM bounds");
    uint32_t prog[] = {
        LUI(5, 0x80001000u),                 // t0 = tohost
        LUI(10, 0x80000000u), ADDI(10, 10, 0x100),   // a0 = syscall block
        SW(10, 5, 0), SW(0, 5, 4),           // tohost = a0: SYS_write
        ADDI(6, 0, (5 << 1) | 1),            // exit code 5
        SW(6, 5, 0), SW(0, 5, 4),
        JAL(0, 0),
    };
    for (unsigned i = 0; i < sizeof(prog) / sizeof(prog[0]); ++i) {
        m[4 * i] = (uint8_t)prog[i]; m[4 * i + 1] = (uint8_t)(prog[i] >> 8); m[4 * i + 2] = (uint8_t)(prog[i] >> 16); m[4 * i + 3] = (uint8_t)(prog[i] >> 24);
    }
    // Syscall block at 0x80000100: SYS_write(1, 0x80000200, 0) (zero length: prints nothing).
    uint8_t * sb = m + 0x100;
    memset(sb, 0, 64);
    sb[0] = 64; sb[8] = 1; sb[16] = 0x00; sb[17] = 0x02; sb[19] = 0x80;
    uint32_t why = tsr_rv_run(r, 1000, UINT64_MAX);
    uint32_t code = 99;
    CHECK(why == TSR_RV_STOP_REQUEST && tsr_bus_htif_exited(t, &code) && code == 5u, "HTIF exit 5 (why %u, code %u)", why, code);
    CHECK(sb[0] == 0 && m[0x1040] == 1 && tsr_rv_pc(r) == 0x8000001Cu, "syscall served, fromhost = 1, stopped after the store");
    tsr_rv_destroy(r);
    tsr_bus_destroy(t);
}

static void put32(uint8_t * p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void put16(uint8_t * p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }

static void test_elf(const char * dir) {
    // A minimal executable: one PT_LOAD at 0x80000000 (8 bytes of code + 8 of bss), a .symtab
    // with tohost and begin_signature.
    uint8_t f[512];
    memset(f, 0, sizeof(f));
    memcpy(f, "\177ELF", 4); f[4] = 1; f[5] = 1; f[6] = 1;
    put16(f + 16, 2); put16(f + 18, 243); put32(f + 20, 1); put32(f + 24, 0x80000004u);
    put32(f + 28, 52); put32(f + 32, 200);
    put16(f + 40, 52); put16(f + 42, 32); put16(f + 44, 1); put16(f + 46, 40); put16(f + 48, 3);
    uint8_t * ph = f + 52;     // PT_LOAD
    put32(ph, 1); put32(ph + 4, 100); put32(ph + 8, 0x80000000u); put32(ph + 12, 0x80000000u);
    put32(ph + 16, 8); put32(ph + 20, 16); put32(ph + 24, 5); put32(ph + 28, 4);
    put32(f + 100, ADDI(1, 0, 1)); put32(f + 104, 0x00000013u);
    // strtab at 120: "\0tohost\0begin_signature\0"
    const char strs[] = "\0tohost\0begin_signature";
    memcpy(f + 120, strs, sizeof(strs));
    // symtab at 160: null, tohost = 0x80001000 (shndx 1), begin_signature undefined (shndx 0)
    uint8_t * sy = f + 160;
    put32(sy + 16, 1); put32(sy + 20, 0x80001000u); put16(sy + 30, 1);
    put32(sy + 32, 8); put32(sy + 36, 0x80002000u); put16(sy + 46, 0);
    // Section headers at 200: [0] null, [1] symtab (link 2), [2] strtab
    uint8_t * sh1 = f + 240, * sh2 = f + 280;
    put32(sh1 + 4, 2); put32(sh1 + 16, 160); put32(sh1 + 20, 48); put32(sh1 + 24, 2); put32(sh1 + 36, 16);
    put32(sh2 + 4, 3); put32(sh2 + 16, 120); put32(sh2 + 20, sizeof(strs));
    char path[1024];
    snprintf(path, sizeof(path), "%s/test_rv_isa.elf", dir);
    FILE * fp = fopen(path, "wb");
    CHECK(fp != NULL, "write %s", path);
    if (fp == NULL) return;
    fwrite(f, 1, 320, fp);
    fclose(fp);
    tsr_rv_elf e;
    CHECK(tsr_rv_elf_read(path, &e) == 0, "elf read");
    CHECK(e.entry == 0x80000004u && e.nseg == 1u && e.lo == 0x80000000u && e.hi == 0x80000010u, "elf header");
    uint32_t v = 0;
    CHECK(tsr_rv_elf_symbol(&e, "tohost", &v) == 0 && v == 0x80001000u, "symbol");
    CHECK(tsr_rv_elf_symbol(&e, "begin_signature", &v) == -1 && tsr_rv_elf_symbol(&e, "tohos", &v) == -1, "undefined / prefix");
    tsr_bus * t = tsr_bus_create_test(0x80000000u, 0x1000);
    uint8_t * m = tsr_bus_mem(t, 0x80000000u, 16);
    memset(m, 0xEE, 16);
    CHECK(tsr_rv_elf_load(t, &e) == 0 && m[0] == 0x93 && m[8] == 0 && m[15] == 0, "load with bss");
    tsr_bus_destroy(t);
    t = tsr_bus_create_test(0x90000000u, 0x1000);
    CHECK(tsr_rv_elf_load(t, &e) == -1, "segment outside RAM");
    tsr_bus_destroy(t);
    tsr_rv_elf_free(&e);
    remove(path);
    f[18] = 62;   // EM_X86_64
    fp = fopen(path, "wb");
    if (fp) { fwrite(f, 1, 320, fp); fclose(fp); }
    CHECK(tsr_rv_elf_read(path, &e) == -1, "wrong machine rejected");
    remove(path);
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--dump-enc") == 0) return dump_enc();
    if (argc > 2 && strcmp(argv[1], "--check-enc") == 0) return check_enc(argv[2]);
    const char * dir = argc > 1 ? argv[1] : ".";
    test_alu();
    test_branches();
    test_jumps();
    test_loads_stores();
    test_m();
    test_compressed();
    test_illegal();
    test_csr();
    test_counters();
    test_traps();
    test_misaligned();
    test_access();
    test_line_buffer();
    test_interrupts();
    test_debug();
    test_wait_yield();
    test_retire();
    test_sba();
    test_watch();
    test_htif();
    test_elf(dir);
    if (RV) tsr_rv_destroy(RV);
    if (B) tsr_bus_destroy(B);
    return test_finish("rv_unit");
}

#!/usr/bin/env python3
"""rv_random.py - random RV32IMC + Zicsr programs, Spike vs tsr_rv retire by retire (P1.11).

docs/v2/06-software-verification.md §10.1 (tsr_rv oracle: "random programs (in-house Python
generator, placed at 0x8000_0000 on both) compared retire by retire with spike --log-commits"),
§4.2 (test bus profile), §10.2 (rv_random). Stdlib only; profiles, runner contract and log
format: rvcommon.py.

Program shape (one seed -> one program, bit-identical text for a given seed and options):
  _start: j reset; trap_vector (256-byte aligned, installed as trap_vector | 1, i.e. the vectored
  mode the profile's mtvec has): logs mcause and mepc to a trap log in memory, steps mepc over
  the faulting instruction (2 or 4 bytes, from its low bits) and returns with mret.
  reset: mtvec, mie = 0, data pointers, every free register loaded with a random or special
  value (0, +-1, 0x8000_0000, 0x7FFF_FFFF, ...), then N random instruction units:
    RV32I register and immediate ALU ops, lui / auipc, M (div / rem by zero and overflow come
    from the special values), byte / half / word loads and stores inside a 2 KiB data region
    (20 % misaligned: the profile does them in hardware), RVC (c.li, c.lui, c.addi, c.mv,
    c.add, c.sub / xor / or / and, c.andi, c.slli / srli / srai, c.addi4spn, c.lw / c.sw,
    c.lwsp / c.swsp, c.nop), forward branches (32-bit and c.beqz / c.bnez), jal / c.j / c.jal,
    jalr / c.jr / c.jalr through la, bounded counted loops, CSR ops on mscratch, mepc (WARL bit
    0), reads of mtvec and mhartid, fence / fence.i, and traps (ecall, ebreak, c.ebreak, c.unimp,
    an OP-FP word the profile does not have);
  end: x1..x30 stored to the signature, then tohost = 1.
  The signature (begin_signature..end_signature) = registers, trap log and the whole data
  region, so a run without commit logs still compares the final state.
Never generated: interrupts, wfi, mret outside the handler, CSRs whose value is implementation
defined (misa, mvendorid / marchid / mimpid, mstatus) or timing dependent (mcycle, minstret),
accesses outside the data region (the profile's RAM size differs from Spike's).

Register use: x2 and x8 are data pointers (c.lwsp / c.swsp and c.lw / c.sw bases), x28 the trap
log pointer, x29 the loop counter, x30 / x31 trap-handler and jump scratch; they are read freely
but never written by random instructions. Every other register is a random destination.

Subcommands:
  gen      --seed S [--n N] [-o FILE]        write one program
  run      --out DIR [--seed S] [--count C] [--n N] [--runner CMD] [--require-env VAR]
           compile and run C programs: Spike profile "test" alone (each must end through tohost),
           or Spike and tsr_rv compared retire by retire up to the store to tohost, then the
           signatures. Logs of passing seeds are deleted unless --keep.
  selftest [--out DIR]                       parser / comparator / arch-case unit checks, generator
           determinism, and (when the compiler is installed) assembling and linking programs.
Exit: 0 pass, 1 failure, 77 skipped (tools or runner missing).
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvcommon as rc  # noqa: E402

RESERVED = {0, 2, 8, 28, 29, 30, 31}
FREE = [i for i in range(32) if i not in RESERVED]          # random destinations
CFREE = [i for i in range(9, 16)]                           # compressed rd' destinations
CREG = list(range(8, 16))                                   # compressed rs' sources
SPECIAL = [0, 1, 2, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000000, 0x7FFFFFFF, 0x80000001, 0x0000FFFF,
           0x00008000, 0x00007FFF, 31, 32, 0x55555555, 0xAAAAAAAA, 0xFFFF8000]
DATA_SIZE = 2048            # bytes of random data (inside the signature)
SP_OFF, S0_OFF = 512, 1024  # x2 / x8 point here inside the data region
MARCH = "rv32imc_zicsr_zifencei"

KINDS = (("alu_rr", 16), ("alu_ri", 14), ("upper", 3), ("muldiv", 10), ("load", 9), ("store", 8),
         ("c_alu", 12), ("c_mem", 6), ("branch", 7), ("c_branch", 3), ("jump", 4), ("csr", 3),
         ("fence", 1), ("trap", 2), ("loop", 1))
STRAIGHT = {"alu_rr", "alu_ri", "upper", "muldiv", "load", "store", "c_alu", "c_mem", "csr",
            "fence"}


class Rng:
    """Draws built only on getrandbits, so a seed gives the same program on every Python 3."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def bits(self, n):
        return self.r.getrandbits(n)

    def below(self, n):
        k = max(1, (n - 1).bit_length())
        while True:
            v = self.r.getrandbits(k)
            if v < n:
                return v

    def range(self, lo, hi):            # inclusive
        return lo + self.below(hi - lo + 1)

    def pick(self, seq):
        return seq[self.below(len(seq))]

    def chance(self, num, den):
        return self.below(den) < num

    def weighted(self, pairs):
        x = self.below(sum(w for _, w in pairs))
        for v, w in pairs:
            if x < w:
                return v
            x -= w
        raise AssertionError


def rvc(*lines):
    return [".option push", ".option rvc"] + list(lines) + [".option pop"]


class Gen:
    def __init__(self, seed, n=1500):
        self.seed, self.n = seed, n
        self.g = Rng(seed)
        self.out, self.pending = [], []
        self.nlabel = self.ntraps = 0

    # operand helpers
    def rd(self):
        return self.g.pick(FREE)

    def rs(self):
        return self.g.below(32)

    def imm12(self):
        if self.g.chance(1, 4):
            return self.g.pick([0, 1, -1, 2047, -2048, 0x7F, -0x80, 0x7FF])
        return self.g.range(-2048, 2047)

    def label(self, lo, hi):
        self.nlabel += 1
        name = ".Lr%d" % self.nlabel
        self.pending.append([name, self.g.range(lo, hi)])
        return name

    def mem_off(self, base_off, size, lo=None, hi=None):
        """Offset from a data pointer at base_off for an access of size bytes inside the data
        region; 20 % misaligned."""
        lo = -base_off if lo is None else lo
        hi = DATA_SIZE - base_off - size if hi is None else hi
        off = self.g.range(lo, hi)
        if size > 1 and not self.g.chance(1, 5):
            off -= (base_off + off) % size
            if off < lo:
                off += size
        return off

    # instruction units
    def u_alu_rr(self):
        op = self.g.pick(["add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"])
        return ["%s x%d, x%d, x%d" % (op, self.rd(), self.rs(), self.rs())]

    def u_alu_ri(self):
        if self.g.chance(1, 3):
            op = self.g.pick(["slli", "srli", "srai"])
            return ["%s x%d, x%d, %d" % (op, self.rd(), self.rs(), self.g.below(32))]
        op = self.g.pick(["addi", "slti", "sltiu", "xori", "ori", "andi"])
        return ["%s x%d, x%d, %d" % (op, self.rd(), self.rs(), self.imm12())]

    def u_upper(self):
        return ["%s x%d, 0x%x" % (self.g.pick(["lui", "auipc"]), self.rd(), self.g.bits(20))]

    def u_muldiv(self):
        op = self.g.pick(["mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"])
        return ["%s x%d, x%d, x%d" % (op, self.rd(), self.rs(), self.rs())]

    def u_load(self):
        op, size = self.g.pick([("lb", 1), ("lbu", 1), ("lh", 2), ("lhu", 2), ("lw", 4)])
        base, boff = self.g.pick([(2, SP_OFF), (8, S0_OFF)])
        return ["%s x%d, %d(x%d)" % (op, self.rd(), self.mem_off(boff, size), base)]

    def u_store(self):
        op, size = self.g.pick([("sb", 1), ("sh", 2), ("sw", 4)])
        base, boff = self.g.pick([(2, SP_OFF), (8, S0_OFF)])
        return ["%s x%d, %d(x%d)" % (op, self.rs(), self.mem_off(boff, size), base)]

    def u_c_alu(self):
        g = self.g
        k = g.below(13)
        nz6 = g.pick([v for v in range(-32, 32) if v])
        if k == 0:
            return rvc("c.li x%d, %d" % (self.rd(), g.range(-32, 31)))
        if k == 1:
            imm = g.pick([g.range(1, 31), g.range(0xFFFE0, 0xFFFFF)])
            return rvc("c.lui x%d, 0x%x" % (self.rd(), imm))
        if k == 2:
            return rvc("c.addi x%d, %d" % (self.rd(), nz6))
        if k == 3:
            return rvc("c.mv x%d, x%d" % (self.rd(), g.range(1, 31)))
        if k == 4:
            return rvc("c.add x%d, x%d" % (self.rd(), g.range(1, 31)))
        if k == 5:
            op = g.pick(["c.sub", "c.xor", "c.or", "c.and"])
            return rvc("%s x%d, x%d" % (op, g.pick(CFREE), g.pick(CREG)))
        if k == 6:
            return rvc("c.andi x%d, %d" % (g.pick(CFREE), g.range(-32, 31)))
        if k == 7:
            return rvc("c.slli x%d, %d" % (self.rd(), g.range(1, 31)))
        if k == 8:
            op = g.pick(["c.srli", "c.srai"])
            return rvc("%s x%d, %d" % (op, g.pick(CFREE), g.range(1, 31)))
        if k == 9:
            return rvc("c.addi4spn x%d, x2, %d" % (g.pick(CFREE), 4 * g.range(1, 255)))
        if k == 10:
            return rvc("c.nop")
        # 32-bit forms the assembler would otherwise compress, kept 32-bit on purpose
        if k == 11:
            return ["addi x%d, x%d, %d" % (self.rd(), self.rd(), nz6)]
        return ["add x%d, x%d, x%d" % (self.rd(), self.rs(), self.rs())]

    def u_c_mem(self):
        g = self.g
        k = g.below(4)
        if k == 0:
            return rvc("c.lw x%d, %d(x8)" % (g.pick(CFREE), 4 * g.below(32)))
        if k == 1:
            return rvc("c.sw x%d, %d(x8)" % (g.pick(CREG), 4 * g.below(32)))
        if k == 2:
            return rvc("c.lwsp x%d, %d(x2)" % (self.rd(), 4 * g.below(64)))
        return rvc("c.swsp x%d, %d(x2)" % (self.rs(), 4 * g.below(64)))

    def u_branch(self):
        op = self.g.pick(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
        return ["%s x%d, x%d, %s" % (op, self.rs(), self.rs(), self.label(2, 9))]

    def u_c_branch(self):
        op = self.g.pick(["c.beqz", "c.bnez"])
        return rvc("%s x%d, %s" % (op, self.g.pick(CREG), self.label(2, 7)))

    def u_jump(self):
        g = self.g
        k = g.below(6)
        lab = self.label(2, 7)
        if k == 0:
            return ["jal x%d, %s" % (g.pick(FREE + [0]), lab)]
        if k == 1:
            return rvc("c.j %s" % lab)
        if k == 2:
            return rvc("c.jal %s" % lab)
        la = ["la x31, %s" % lab]
        if k == 3:
            return la + ["jalr x%d, 0(x31)" % g.pick(FREE + [0])]
        if k == 4:
            return la + rvc("c.jr x31")
        return la + rvc("c.jalr x31")

    def u_csr(self):
        g = self.g
        k = g.below(6)
        rd = g.pick(FREE + [0])
        if k == 0:
            op = g.pick(["csrrw", "csrrs", "csrrc"])
            return ["%s x%d, mscratch, x%d" % (op, rd, self.rs())]
        if k == 1:
            op = g.pick(["csrrwi", "csrrsi", "csrrci"])
            return ["%s x%d, mscratch, %d" % (op, rd, g.below(32))]
        if k == 2:
            return ["csrw mepc, x%d" % self.rs(), "csrr x%d, mepc" % self.rd()]
        if k == 3:
            return ["csrr x%d, mtvec" % self.rd()]
        if k == 4:
            return ["csrr x%d, mhartid" % self.rd()]
        return ["csrr x%d, mscratch" % self.rd()]

    def u_fence(self):
        return self.g.pick([["fence"], ["fence rw, rw"], ["fence.i"], ["nop"], rvc("c.nop")])

    def u_trap(self):
        self.ntraps += 1
        return self.g.pick([["ecall"], ["ebreak"], rvc("c.ebreak"),
                            [".half 0x0000    # c.unimp: illegal"],
                            [".word 0x00000053    # OP-FP (fadd.s): illegal without F"]])

    def u_loop(self):
        self.flush()
        self.nlabel += 1
        lab = ".Ll%d" % self.nlabel
        body = ["li x29, %d" % self.g.range(1, 5), lab + ":"]
        for _ in range(self.g.range(2, 8)):
            kind = self.g.weighted([(k, w) for k, w in KINDS if k in STRAIGHT])
            body += getattr(self, "u_" + kind)()
        return body + ["addi x29, x29, -1", "bnez x29, %s" % lab]

    def tick(self):
        keep = []
        for p in self.pending:
            p[1] -= 1
            if p[1] <= 0:
                self.out.append(p[0] + ":")
            else:
                keep.append(p)
        self.pending = keep

    def flush(self):
        for p in self.pending:
            self.out.append(p[0] + ":")
        self.pending = []

    def program(self):
        g = self.g
        o = self.out
        o += ["# rv_random.py seed=%d n=%d (generated; docs/v2/06-software-verification.md §10.1)"
              % (self.seed, self.n),
              '\t.section .text.init, "ax", @progbits', "\t.globl _start", "\t.option norvc",
              "_start:", "j reset",
              "\t.align 8", "trap_vector:",
              "csrr x30, mcause", "sw x30, 0(x28)", "csrr x31, mepc", "sw x31, 4(x28)",
              "addi x28, x28, 8", "lhu x30, 0(x31)", "andi x30, x30, 3", "addi x31, x31, 2",
              "addi x30, x30, -3", "bnez x30, 1f", "addi x31, x31, 2",
              "1:", "csrw mepc, x31", "mret",
              "reset:", "la x31, trap_vector", "ori x31, x31, 1", "csrw mtvec, x31",
              "csrwi mie, 0", "la x2, rnd_data + %d" % SP_OFF, "la x8, rnd_data + %d" % S0_OFF,
              "la x28, trap_log", "li x29, 0"]
        for r in FREE + [30, 31]:
            v = g.pick(SPECIAL) if g.chance(3, 10) else g.bits(32)
            o.append("li x%d, 0x%08x" % (r, v))
        o.append("# body")
        for _ in range(self.n):
            kind = g.weighted(KINDS)
            o += getattr(self, "u_" + kind)()
            self.tick()
        self.flush()
        o += ["# end: registers to the signature, then tohost = 1", "la x31, begin_signature"]
        o += ["sw x%d, %d(x31)" % (r, 4 * (r - 1)) for r in range(1, 31)]
        o += ["li x30, 1", "la x31, tohost", "sw x30, 0(x31)", "2:", "j 2b",
              '\t.section .tohost, "aw", @progbits', "\t.align 6", "\t.globl tohost",
              "tohost: .dword 0", "\t.align 6", "\t.globl fromhost", "fromhost: .dword 0",
              "\t.data", "\t.align 6", "\t.globl begin_signature", "begin_signature:",
              "\t.space %d" % (4 * 30), "trap_log:", "\t.space %d" % (8 * (self.ntraps + 1)),
              "\t.align 4", "rnd_data:"]
        data = [g.bits(8) for _ in range(DATA_SIZE)]
        for i in range(0, DATA_SIZE, 16):
            o.append("\t.byte " + ", ".join("0x%02x" % b for b in data[i:i + 16]))
        o += ["\t.globl end_signature", "end_signature:", ""]
        return "\n".join(x if x.startswith(("\t", "#", ".L")) or x.endswith(":") else "\t" + x
                         for x in o)


def generate(seed, n):
    return Gen(seed, n).program()


def compile_prog(tools, src, elf):
    cmd = [tools.gcc(), "-march=" + MARCH, "-mabi=ilp32", "-static", "-nostdlib", "-nostartfiles",
           "-T" + str(rc.HERE / "random.ld"), str(src), "-o", str(elf)]
    code, so, se = rc.run(cmd, timeout=120)
    return code == 0, "$ %s\n%s%s" % (" ".join(cmd), so, se)


# --------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------

def run_seed(a, tools, runner, seed):
    d = Path(a.out) / ("seed_%d" % seed)
    d.mkdir(parents=True, exist_ok=True)
    src, elf = d / "prog.S", d / "prog.elf"
    src.write_text(generate(seed, a.n))
    ok, msg = compile_prog(tools, src, elf)
    if not ok:
        return False, "compile failed\n" + msg
    tohost = rc.elf_symbols(elf)["tohost"]
    slog, ssig = d / "spike.log", d / "spike.sig"
    code, so, se = rc.run(rc.spike_cmd(tools, "test", elf, log=slog, signature=ssig), a.timeout,
                          env=tools.spike_env())
    if code != 0:
        return False, "spike: exit %s\n%s%s" % (code, so, se)
    with open(slog, errors="replace") as f:
        ref = rc.until_store(rc.parse_commit_log(f), tohost)
    if not ref or not any(s[0] == tohost for s in ref[-1].stores):
        return False, "spike: no store to tohost in the commit log"
    res = "spike %d retires" % len(ref)
    if runner:
        tlog, tsig = d / "tsr.log", d / "tsr.sig"
        for p in (tlog, tsig):
            if p.exists():
                p.unlink()
        code, so, se = rc.run(rc.runner_cmd(runner, elf, log=tlog, signature=tsig,
                                            max_insns=10 * len(ref) + 10000), a.timeout)
        if code != 0:
            return False, "tsr_rv: exit %s\n%s%s" % (code, so, se)
        with open(tlog, errors="replace") as f:
            dut = rc.until_store(rc.parse_commit_log(f), tohost)
        ok, msg = rc.compare_retires(ref, dut, compare_csrs=a.compare_csrs)
        if not ok:
            return False, "lockstep: " + msg
        ok, msg2 = rc.compare_signatures(ssig, tsig, rc.elf_symbols(elf)["begin_signature"])
        if not ok:
            return False, msg2
        res = msg + ", " + msg2
    if not a.keep:
        for p in d.glob("*.log"):
            p.unlink()
    return True, res


def cmd_run(a):
    if a.require_env and os.environ.get(a.require_env) != "1":
        rc.skip("long sweep; runs only with %s=1 in the environment" % a.require_env)
    tools = rc.Tools.from_args(a)
    runner = rc.parse_runner(a.runner)
    if a.runner is not None and runner is None:
        rc.skip("no tsr_rv runner (P1.7; contract in sw/sim/tests/rvtests/rvcommon.py)")
    if not tools.have_gcc():
        rc.skip("RISC-V compiler %s not found; %s" % (tools.gcc(), rc.INSTALL_HINT))
    if not tools.have_spike():
        rc.skip("Spike %s not found; %s" % (tools.spike, rc.INSTALL_HINT))
    seeds = list(range(a.seed, a.seed + a.count))
    results = rc.run_all(a.jobs, lambda s: run_seed(a, tools, runner, s), seeds)
    fails = []
    for s, (ok, msg) in zip(seeds, results):
        print("%s seed %d: %s" % ("PASS" if ok else "FAIL", s, msg.splitlines()[0]))
        if not ok:
            print("    " + msg.replace("\n", "\n    "))
            fails.append(s)
    print("summary: %d passed, %d failed%s (n=%d, %s)" % (
        len(seeds) - len(fails), len(fails),
        (": seeds " + " ".join(map(str, fails))) if fails else "", a.n,
        "spike vs tsr_rv" if runner else "spike only"))
    return 1 if fails else 0


# --------------------------------------------------------------------------------------------
# selftest (no Spike needed)
# --------------------------------------------------------------------------------------------

SPIKE_SAMPLE = """\
warning: tohost and fromhost symbols not in ELF; can't communicate with target
core   0: 3 0x80000000 (0x0040006f)
core   0: 3 0x80000100 (0x34202f73) x30 0x00000002
core   0: 3 0x8000010a (0x4581) x11 0x00000000
core   0: 3 0x80000110 (0x0002a303) x6  0x00000013 mem 0x80001001
core   0: 3 0x80000114 (0x0062a023) mem 0x80001000 0x00000013
core   0: 3 0x80000118 (0x00629023) mem 0x80001002 0x0013
core   0: 3 0x8000011c (0x340292f3) x5  0x00000005 c832_mscratch 0x00000007
core   0: 3 0x80000120 (0x0062a023) mem 0x80002000 0x00000001
core   0: 3 0x80000124 (0x0000006f)
"""


def selftest(a):
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)
        print("%s %s" % ("ok  " if cond else "FAIL", what))

    r = list(rc.parse_commit_log(SPIKE_SAMPLE.splitlines()))
    check(len(r) == 9, "parser: 9 commit lines, non-commit lines ignored")
    check(r[0].pc == 0x80000000 and r[0].insn == 0x0040006F and r[0].ilen == 32 and not r[0].xw,
          "parser: pc, 32-bit insn, no rd write")
    check(r[2].ilen == 16 and r[2].insn == 0x4581 and r[2].xw == [(11, 0)],
          "parser: 16-bit insn and x11 write")
    check(r[3].xw == [(6, 0x13)] and r[3].loads == [0x80001001] and not r[3].stores,
          "parser: load address (misaligned)")
    check(r[4].stores == [(0x80001000, 4, 0x13)] and r[5].stores == [(0x80001002, 2, 0x13)],
          "parser: word and halfword store data widths")
    check(r[6].xw == [(5, 5)] and r[6].csrw == [(832, 7)], "parser: x write and CSR write")
    u = rc.until_store(r, 0x80002000)
    check(len(u) == 8, "until_store: stops at the tohost store")
    ok, _ = rc.compare_retires(u, list(rc.parse_commit_log(SPIKE_SAMPLE.splitlines()))[:8])
    check(ok, "compare: identical logs")
    bad = SPIKE_SAMPLE.replace("x11 0x00000000", "x11 0x00000001")
    ok, msg = rc.compare_retires(r, list(rc.parse_commit_log(bad.splitlines())))
    check(not ok and "retire 2" in msg, "compare: rd value divergence reported at retire 2")
    bad = SPIKE_SAMPLE.replace("c832_mscratch 0x00000007", "c832_mscratch 0x00000006")
    r2 = list(rc.parse_commit_log(bad.splitlines()))
    check(rc.compare_retires(r, r2)[0] and not rc.compare_retires(r, r2, compare_csrs=True)[0],
          "compare: CSR writes ignored unless --compare-csrs")
    check(not rc.compare_retires(r, r[:-1])[0], "compare: retire count difference reported")

    # riscv-arch-test case selection and -march (rvtests.py)
    import rvtests
    t = ('RVTEST_ISA("RV32I_Zicsr")\nRVTEST_CASE(0,"//check ISA:=regex(.*32.*);check ISA:=regex('
         '.*I.*); check hw_data_misaligned_support:=True; def rvtest_mtrap_routine=True;def '
         'TEST_CASE_1=True;",misalign-lw)\nRVTEST_CASE(1,"//check ISA:=regex(.*32.*);check ISA:='
         'regex(.*I.*Zicsr.*); check hw_data_misaligned_support:=False; def TEST_CASE_1=True;",x)')
    check(rvtests.select_case(t) == (["rvtest_mtrap_routine=True", "TEST_CASE_1=True"], None),
          "arch-test: case 0 chosen (hardware misaligned support)")
    check(rvtests.select_case('RVTEST_CASE(0,"//check ISA:=regex(.*I.*Zca.*Zcb.*);def T=1;",c)')[0]
          is None, "arch-test: Zcb case not applicable")
    check(rvtests.isa_to_march("RV32IC") == "rv32ic_zicsr_zifencei" and
          rvtests.isa_to_march("RV32I_Zifencei") == "rv32i_zicsr_zifencei",
          "arch-test: -march from RVTEST_ISA")

    # generator determinism and static constraints
    p1, p2, p3 = generate(1, 400), generate(1, 400), generate(2, 400)
    check(p1 == p2 and p1 != p3, "generator: same seed same text, other seed differs")
    # Mnemonics whose first operand is a destination register; la x31 (jump scratch) and the
    # loop counter x29 (li / addi) are the generator's own and checked separately.
    writes = {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and", "addi", "slti",
              "sltiu", "xori", "ori", "andi", "slli", "srli", "srai", "lui", "auipc", "mul", "mulh",
              "mulhsu", "mulhu", "div", "divu", "rem", "remu", "lb", "lbu", "lh", "lhu", "lw",
              "jal", "jalr", "csrrw", "csrrs", "csrrc", "csrrwi", "csrrsi", "csrrci", "csrr",
              "c.li", "c.lui", "c.addi", "c.mv", "c.add", "c.sub", "c.xor", "c.or", "c.and",
              "c.andi", "c.slli", "c.srli", "c.srai", "c.addi4spn", "c.lw", "c.lwsp", "li"}
    bad, own = [], 0
    body = p1.split("# body", 1)[1].split("# end", 1)[0]
    for ln in body.splitlines():
        f = ln.replace(",", " ").split()
        if len(f) < 2 or f[0] not in writes or not f[1].startswith("x"):
            continue
        r = int(f[1][1:])
        if f[0] in ("li", "addi") and r == 29:
            own += 1
        elif r in RESERVED:
            bad.append(ln.strip())
    check(not bad, "generator: random destinations avoid x0, x2, x8, x28-x31 %s" % bad[:3])
    check(own > 0, "generator: bounded loops present (li / addi x29)")

    tools = rc.Tools.from_args(a)
    if not tools.have_gcc():
        print("note: %s not found, assembling skipped (%s)" % (tools.gcc(), rc.INSTALL_HINT))
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for seed in range(a.seed, a.seed + a.count):
            src, elf = out / ("self_%d.S" % seed), out / ("self_%d.elf" % seed)
            src.write_text(generate(seed, a.n))
            ok, msg = compile_prog(tools, src, elf)
            check(ok, "assemble + link seed %d" % seed)
            if not ok:
                print(msg)
                continue
            sy = rc.elf_symbols(elf)
            data = elf.read_bytes()
            entry = int.from_bytes(data[0x18:0x1C], "little")
            check(entry == 0x80000000 and sy["_start"] == 0x80000000 and
                  sy["trap_vector"] % 256 == 0 and sy["tohost"] % 64 == 0 and
                  sy["begin_signature"] < sy["rnd_data"] < sy["end_signature"],
                  "layout seed %d: entry 0x80000000, trap_vector 256-aligned, signature" % seed)
    print("selftest: %d failures" % len(fails))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--n", type=int, default=1500)
    g.add_argument("-o", "--output", default=None)
    for name in ("run", "selftest"):
        p = sub.add_parser(name)
        rc.Tools.add_args(p)
        p.add_argument("--out", required=(name == "run"),
                       default=None if name == "run" else "rv_random_selftest")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--count", type=int, default=20 if name == "run" else 4)
        p.add_argument("--n", type=int, default=1500 if name == "run" else 600)
        p.add_argument("-j", "--jobs", type=int, default=4)
        p.add_argument("--timeout", type=float, default=rc.DEFAULT_TIMEOUT)
        p.add_argument("--runner", default=None)
        p.add_argument("--compare-csrs", action="store_true")
        p.add_argument("--keep", action="store_true", help="keep commit logs of passing seeds")
        p.add_argument("--require-env", default=None,
                       help="skip (77) unless this environment variable is 1 (slow sweeps)")
    a = ap.parse_args()
    if a.cmd == "gen":
        text = generate(a.seed, a.n)
        if a.output:
            Path(a.output).write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    return selftest(a)


if __name__ == "__main__":
    sys.exit(main())

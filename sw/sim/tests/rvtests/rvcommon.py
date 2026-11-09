"""Shared helpers of the RV32 conformance harness (docs/design/06-software-verification.md
§4.2 test bus profile, §9.6 toolchain, §10.1 oracle chain Spike -> tsr_rv).

Stdlib only (Python >= 3.8). Used by rvtests.py (riscv-tests, riscv-arch-test) and rv_random.py
(random lockstep programs).

Spike profiles (SPIKE_PROFILES):
  upstream  riscv-tests exactly as published (env/p) on a stock Spike: the sanity check of the
            installed compiler and Spike. ISA rv32imc_zicsr_zifencei plus the two extensions
            upstream's own `make run` adds for these suites (zicntr: rv32mi-zicntr; zicclsm:
            rv32ui-ma_data); default privilege modes (MSU), 16 PMP regions, 4 triggers, and the
            generated device tree, so Spike runs `dtc` from PATH (run_spike puts the installed
            dtc first).
  test      the tsr_rv test bus profile (SW §4.2, CC §3.2): RV32IMC + Zicsr + Zifencei, misaligned
            loads/stores done in hardware like Ibex (zicclsm), M and U modes (Ibex always has U:
            tsr_rv's misa is 0x4010_1104; the firmware itself runs in M only), no S-mode, no
            PMP, no Zicntr, 2 triggers, no device tree (--disable-dtb: reset PC = ELF entry, all
            x registers 0, no boot ROM, no CLINT, no dtc). RAM at 0x8000_0000, HTIF tohost /
            fromhost. This is the oracle configuration of rv_isa and rv_random.

tsr_rv runner contract (implemented by tsr_rv_run; the harness calls it as)
    <runner> [<TSR_RV_RUNNER_ARGS>...] [--log-commits FILE] [--signature FILE] [--max-insns N] ELF
  * loads the ELF on the test bus profile: RAM at 0x8000_0000 covering the ELF's PT_LOAD
    segments, reset PC = e_entry, x1..x31 = 0, HTIF `tohost`/`fromhost` found by symbol;
  * exits 0 iff the program stored 1 to the low word of `tohost`; a store of (n << 1) | 1 exits
    with n (like Spike); anything else (budget exhausted, simulator error) exits nonzero;
  * --log-commits FILE: one line per retired instruction in Spike's --log-commits format
        core   0: 3 0x80000000 (0x00000297) x5  0x80000000
        core   0: 3 0x80000010 (0x4581) x11 0x00000000
        core   0: 3 0x80000014 (0x0002a303) x6  0x00000013 mem 0x80001000
        core   0: 3 0x80000018 (0x0062a023) mem 0x80001000 0x00000013
    i.e. "core%4d: " hart, privilege digit, pc as 0x%08x, the instruction bits as 0x%04x (16-bit)
    or 0x%08x (32-bit) in parentheses, then the x-register write (" x%-2d " + 0x%08x; none for
    rd = x0), then one " mem 0x%08x" per load (original address of the access, misaligned
    included) and one " mem 0x%08x 0x<data>" per store (data with 2 hex digits per byte stored).
    A trapping instruction does not retire and prints no line. CSR writes (" c<num>_<name>
    0x%08x") may be printed and are compared only with --compare-csrs. The log may stop right
    after the store to `tohost`; Spike runs on for up to ~5000 more instructions until HTIF polls
    it, so comparisons end at the first store to `tohost` on each side;
  * --signature FILE: the words from `begin_signature` to `end_signature`, one per line as
    %08x (little-endian word value), i.e. Spike's +signature-granularity=4 format;
  * --max-insns N: stop with a nonzero exit after N retired instructions.
"""

import os
import re
import shlex
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]                       # sw/sim/tests/rvtests -> repository root
LOCK_FILE = REPO / "docs" / "design" / "toolchain.lock"
SKIP = 77                                    # ctest SKIP_RETURN_CODE

INSTALL_HINT = ("install the pinned tools with sw/ctrl/tools/install-toolchain.sh "
                "(docs/design/06-software-verification.md §9.6; pins in docs/design/toolchain.lock)")

# Spike command-line options per profile (see the module docstring).
SPIKE_PROFILES = {
    "upstream": ["--isa=rv32imc_zicsr_zifencei_zicntr_zicclsm"],
    "test": ["--isa=rv32imc_zicsr_zifencei_zicclsm", "--priv=mu", "--pmpregions=0",
             "--triggers=2", "--disable-dtb"],
}

# The ISA string riscv-arch-test RVTEST_CASE conditions are matched against (riscof's isa yaml
# spelling), and the profile's answer to "check hw_data_misaligned_support".
PROFILE_ISA = "RV32IMCUZicsr_Zifencei"
PROFILE_MISALIGNED = True

# Data, not timing: a riscv-tests program retires < 100 K instructions; the budget catches hangs.
DEFAULT_TIMEOUT = 20.0


def read_lock(path=LOCK_FILE):
    """docs/design/toolchain.lock as {key: value} ("key = value" lines, '#' comments)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or " = " not in line:
                continue
            k, v = line.split(" = ", 1)
            out[k.strip()] = v.strip()
    return out


class Tools:
    """Compiler, Spike and dtc locations, by default the install script's layout under
    <tools-prefix> (TSR_TOOLS_PREFIX or build/tools/riscv)."""

    def __init__(self, gcc_prefix, spike, dtc_dir):
        self.gcc_prefix = gcc_prefix
        self.spike = spike
        self.dtc_dir = dtc_dir

    @staticmethod
    def add_args(ap):
        ap.add_argument("--tools-prefix", default=os.environ.get(
            "TSR_TOOLS_PREFIX", str(REPO / "build" / "tools" / "riscv")),
            help="install root of sw/ctrl/tools/install-toolchain.sh (default: %(default)s)")
        ap.add_argument("--gcc-prefix", default=None,
                        help="compiler prefix, e.g. <dir>/bin/riscv-none-elf- (TSR_RISCV_PREFIX)")
        ap.add_argument("--spike", default=None, help="spike executable (TSR_SPIKE)")

    @classmethod
    def from_args(cls, a):
        lock = read_lock()
        tp = Path(a.tools_prefix)
        gcc_prefix = a.gcc_prefix or os.environ.get("TSR_RISCV_PREFIX") or str(
            tp / lock["xpack_gcc.dir"] / "bin" / lock["xpack_gcc.prefix"])
        spike = a.spike or os.environ.get("TSR_SPIKE") or str(tp / "spike" / "bin" / "spike")
        return cls(gcc_prefix, spike, str(tp / "dtc" / "bin"))

    def gcc(self):
        return self.gcc_prefix + "gcc"

    def have_gcc(self):
        return os.access(self.gcc(), os.X_OK)

    def have_spike(self):
        return os.access(self.spike, os.X_OK)

    def spike_env(self):
        env = dict(os.environ)
        if os.path.isdir(self.dtc_dir):
            env["PATH"] = self.dtc_dir + os.pathsep + env.get("PATH", "")
        return env


def skip(msg):
    """Report a missing prerequisite and exit with ctest's skip code."""
    print("SKIPPED: " + msg, file=sys.stderr)
    sys.exit(SKIP)


def run(cmd, timeout=DEFAULT_TIMEOUT, env=None, cwd=None):
    """Run cmd; returns (returncode or None on timeout, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                           env=env, cwd=cwd)
        return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace")
        err = (e.stderr or b"").decode(errors="replace")
        return None, out, err + "\n[timeout after %.0f s]" % timeout


def spike_cmd(tools, profile, elf, log=None, signature=None, extra=()):
    cmd = [tools.spike] + SPIKE_PROFILES[profile] + list(extra)
    if log is not None:
        cmd += ["--log-commits", "--log=" + str(log)]
    if signature is not None:
        cmd += ["+signature=" + str(signature), "+signature-granularity=4"]
    return cmd + [str(elf)]


def runner_cmd(runner, elf, log=None, signature=None, max_insns=None):
    """runner: argv list (executable + TSR_RV_RUNNER_ARGS). Contract: module docstring."""
    cmd = list(runner)
    if log is not None:
        cmd += ["--log-commits", str(log)]
    if signature is not None:
        cmd += ["--signature", str(signature)]
    if max_insns is not None:
        cmd += ["--max-insns", str(max_insns)]
    return cmd + [str(elf)]


def parse_runner(s):
    """--runner value -> argv list, or None when empty (rv_isa / rv_random then skip)."""
    if s is None or not s.strip():
        return None
    return shlex.split(s)


# --------------------------------------------------------------------------------------------
# ELF32 symbol table (little-endian RISC-V executables only)
# --------------------------------------------------------------------------------------------

def elf_symbols(path):
    """{name: value} of the ELF's .symtab (global and local), without binutils."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1 or data[5] != 1:
        raise ValueError("%s: not an ELF32 little-endian file" % path)
    e_shoff, = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x2E)
    secs = [struct.unpack_from("<IIIIIIIIII", data, e_shoff + i * e_shentsize)
            for i in range(e_shnum)]
    syms = {}
    for s in secs:
        if s[1] != 2:                                  # SHT_SYMTAB
            continue
        stroff = secs[s[6]][4]                         # sh_link -> string table offset
        for off in range(s[4], s[4] + s[5], 16):
            st_name, st_value = struct.unpack_from("<II", data, off)
            if st_name:
                end = data.index(b"\0", stroff + st_name)
                syms[data[stroff + st_name:end].decode()] = st_value
    return syms


# --------------------------------------------------------------------------------------------
# Spike commit logs
# --------------------------------------------------------------------------------------------

_LINE = re.compile(r"^core\s+(\d+):\s+(\d)\s+0x([0-9a-fA-F]+)\s+\(0x([0-9a-fA-F]+)\)(.*)$")
_REG = re.compile(r"^([xfv])(\d+)$")
_CSR = re.compile(r"^c(\d+)(?:_\S*)?$")


class Retire:
    """One retired instruction of a commit log."""
    __slots__ = ("lineno", "text", "pc", "insn", "ilen", "xw", "csrw", "loads", "stores", "other")

    def __init__(self, lineno, text, pc, insn, ilen):
        self.lineno, self.text, self.pc, self.insn, self.ilen = lineno, text, pc, insn, ilen
        self.xw, self.csrw, self.loads, self.stores, self.other = [], [], [], [], []

    def key(self, compare_csrs=False, ignore_csrs=()):
        csr = tuple(sorted(c for c in self.csrw if c[0] not in ignore_csrs)) if compare_csrs else ()
        return (self.pc, self.insn, tuple(sorted(self.xw)), csr, tuple(self.loads),
                tuple(self.stores))


def parse_commit_log(lines):
    """Iterate Retire records of Spike --log-commits text; lines that are not commit lines
    (warnings, HTIF output) are ignored. Raises ValueError on a malformed commit line."""
    for n, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        m = _LINE.match(line)
        if not m:
            continue
        ins = m.group(4)
        r = Retire(n, line, int(m.group(3), 16), int(ins, 16), len(ins) * 4)
        tok = m.group(5).split()
        i = 0
        while i < len(tok):
            t = tok[i]
            if t == "mem":
                if i + 1 >= len(tok):
                    raise ValueError("line %d: 'mem' without address: %s" % (n, line))
                addr = int(tok[i + 1], 16)
                if i + 2 < len(tok) and tok[i + 2].startswith("0x"):
                    d = tok[i + 2][2:]
                    r.stores.append((addr, len(d) // 2, int(d, 16)))
                    i += 3
                else:
                    r.loads.append(addr)
                    i += 2
                continue
            if i + 1 >= len(tok):
                raise ValueError("line %d: dangling token %r: %s" % (n, t, line))
            g = _REG.match(t)
            c = _CSR.match(t)
            if g and g.group(1) == "x":
                r.xw.append((int(g.group(2)), int(tok[i + 1], 16)))
            elif c:
                r.csrw.append((int(c.group(1)), int(tok[i + 1], 16)))
            else:
                r.other.append((t, tok[i + 1]))
            i += 2
        yield r


def until_store(retires, addr):
    """Retires up to and including the first one that stores to addr (None: all)."""
    out = []
    for r in retires:
        out.append(r)
        if addr is not None and any(s[0] == addr for s in r.stores):
            break
    return out


def compare_retires(ref, dut, compare_csrs=False, ignore_csrs=(), context=3):
    """Compare two Retire lists retire by retire. Returns (ok, message)."""
    n = min(len(ref), len(dut))
    for i in range(n):
        if ref[i].key(compare_csrs, ignore_csrs) != dut[i].key(compare_csrs, ignore_csrs):
            lo = max(0, i - context)
            msg = ["first divergence at retire %d (ref line %d, dut line %d):" %
                   (i, ref[i].lineno, dut[i].lineno)]
            for j in range(lo, i + 1):
                mark = ">>" if j == i else "  "
                msg.append("%s ref: %s" % (mark, ref[j].text))
                msg.append("%s dut: %s" % (mark, dut[j].text))
            return False, "\n".join(msg)
    if len(ref) != len(dut):
        longer, name = (ref, "ref") if len(ref) > len(dut) else (dut, "dut")
        return False, ("retire counts differ: ref %d, dut %d; next %s retire: %s" %
                       (len(ref), len(dut), name, longer[n].text))
    return True, "%d retires identical" % n


def read_signature(path):
    with open(path, encoding="ascii") as f:
        return [ln.strip().lower() for ln in f if ln.strip()]


def compare_signatures(ref_path, dut_path, base=None):
    ref, dut = read_signature(ref_path), read_signature(dut_path)
    for i, (a, b) in enumerate(zip(ref, dut)):
        if a != b:
            where = " (0x%08x)" % (base + 4 * i) if base is not None else ""
            return False, "signature word %d%s: ref %s, dut %s" % (i, where, a, b)
    if len(ref) != len(dut):
        return False, "signature lengths differ: ref %d words, dut %d" % (len(ref), len(dut))
    if not ref:
        return False, "empty signature"
    return True, "%d signature words identical" % len(ref)


# --------------------------------------------------------------------------------------------
# Incremental parallel compilation (one gcc call per ELF, gcc -MD dependency files)
# --------------------------------------------------------------------------------------------

def _deps(dfile):
    try:
        txt = Path(dfile).read_text()
    except OSError:
        return None
    txt = txt.replace("\\\n", " ")
    _, _, rhs = txt.partition(":")
    return [p for p in rhs.split() if p != "\\"]


def up_to_date(out, cmd):
    """True if out exists, was built by the same command, and is newer than every dependency."""
    out = Path(out)
    stamp = out.with_name(out.name + ".cmd")
    if not out.exists() or not stamp.exists() or stamp.read_text() != "\0".join(cmd):
        return False
    deps = _deps(out.with_name(out.name + ".d"))
    if deps is None:
        return False
    t = out.stat().st_mtime
    try:
        return all(os.stat(d).st_mtime <= t for d in deps)
    except OSError:
        return False


def compile_all(jobs, items):
    """items: list of (out_path, cmd) where cmd already contains -o out and -MD -MF out.d.
    Returns a list of (out_path, error text) for failures."""
    def one(item):
        out, cmd = item
        if up_to_date(out, cmd):
            return None
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        rc, so, se = run(cmd, timeout=300)
        if rc != 0:
            return (out, "$ %s\n%s%s" % (" ".join(map(str, cmd)), so, se))
        Path(str(out) + ".cmd").write_text("\0".join(cmd))
        return None

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return [r for r in ex.map(one, items) if r is not None]


def run_all(jobs, fn, items):
    """Run fn over items with a thread pool; results in item order."""
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return list(ex.map(fn, items))

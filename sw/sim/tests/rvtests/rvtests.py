#!/usr/bin/env python3
"""rvtests.py - riscv-tests and riscv-arch-test on Spike and on tsr_rv (P1.11).

docs/design/06-software-verification.md §4.2 (test bus profile), §9.6 (tools), §10.1 (oracle chain:
Spike -> tsr_rv), §10.2 (rv_isa). Stdlib only. Profiles, the tsr_rv runner contract and the
commit-log format: rvcommon.py.

Suites (sources pinned in docs/design/toolchain.lock, fetched by sw/ctrl/tools/install-toolchain.sh):
  riscv-tests  rv32ui, rv32um, rv32uc, rv32mi in the physical-memory "-p" environment, built here
               (not with the suite's Makefile) with the pinned xPack compiler:
                 -march=rv32im_zicsr_zifencei (rv32uc: rv32imc_zicsr_zifencei) -mabi=ilp32 -static
                 -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles, linked at 0x8000_0000
               (upstream builds rv32 with -march=rv32g; the core's ISA is used instead so that the
               assembler rejects anything the profile lacks; neither adds C to non-rvc suites).
               --env upstream: the suite's own env/p, run on Spike profile "upstream" (the tools'
               sanity check). --env tsr (default): sw/sim/tests/rvtests/env/riscv_test.h (vectored
               mtvec, no probes of absent CSRs), run on Spike profile "test" and on tsr_rv.
               Self-checking: pass = exit 0 (tohost = 1).
  arch-test    riscv-arch-test 3.10.0 rv32i_m/{I,M,C,Zifencei,privilege}, compiled as riscof does
               (-march from RVTEST_ISA, -DXLEN=32, the "def" macros of the RVTEST_CASE whose checks
               the profile ISA RV32IMCUZicsr_Zifencei with hardware misaligned support satisfies),
               with sw/sim/tests/rvtests/arch/{model_test.h,link.ld}. The result is the signature;
               tsr_rv must reproduce Spike's word for word (Spike is the oracle; the suite ships
               no reference signatures).

Modes:
  no --runner   Spike only (profile per suite): riscv-tests must pass; arch-test programs must end
                through tohost with a non-empty signature.
  --runner CMD  every test on tsr_rv (contract: rvcommon.py) and on Spike profile "test":
                riscv-tests must pass on both (rv_isa), arch-test signatures must be equal
                (rv_arch). --lockstep also compares the riscv-tests commit logs retire by retire
                (rv_lockstep; the tests that read implementation-defined CSRs, LOCKSTEP_EXCLUDE,
                are left out).
  --spike-optional  riscv-tests suite with --runner, without --lockstep (rv_isa): when Spike is
                absent the self-checking riscv-tests run on tsr_rv alone; the output says so.
                Without it a missing Spike skips the run (exit 77), as it always does for the
                arch-test suite, whose signatures only Spike provides (rv_arch).
Exit: 0 pass, 1 failure, 77 skipped (tools, sources or runner missing; the reason is printed).
"""

import argparse
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True                              # no __pycache__ in the source tree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rvcommon as rc  # noqa: E402

RISCV_TEST_SUITES = ("rv32ui", "rv32um", "rv32uc", "rv32mi")
ARCH_SUITES = ("I", "M", "C", "Zifencei", "privilege")

# Tests outside the test bus profile (both Spike profile "test" and tsr_rv would fail them).
PROFILE_EXCLUDE = {
    "rv32mi-p-zicntr": "the profile has no Zicntr (no cycle/instret/time; CC §3.2, SW §4.2 list "
                       "mcycle/minstret only)",
    "rv32mi-p-pmpaddr": "the profile has no PMP (CC §3.2: PMPEnable = 0)",
}
# Tests run on Spike but not on tsr_rv (rv_isa), with the reason.
RUNNER_EXCLUDE = {
    "rv32mi-p-breakpoint": "guards 'csrs tcontrol' with a 4-byte aligned mtvec; tcontrol is not in "
                           "the profile's CSR list (SW §4.2) and a vectored-only mtvec cannot hold "
                           "that label, so the trap lands at the 256-byte base",
}


# riscv-tests left out of --lockstep: they pass self-checking on both models, but read CSR values
# that are implementation-defined, so the retire logs legitimately differ.
LOCKSTEP_EXCLUDE = {
    "rv32mi-p-mcsr": "reads marchid (Spike 5, Ibex 22) and mimpid",
    "rv32mi-p-ma_fetch": "clears misa.C: Spike lets software clear C, Ibex's misa is read-only",
}


def common_args(ap):
    rc.Tools.add_args(ap)
    ap.add_argument("--riscv-tests", default=None,
                    help="riscv-tests source (default <tools-prefix>/src/riscv-tests)")
    ap.add_argument("--arch-test", default=None,
                    help="riscv-arch-test source (default <tools-prefix>/src/riscv-arch-test)")
    ap.add_argument("--out", required=True, help="build and output directory")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=rc.DEFAULT_TIMEOUT, help="seconds per run")
    ap.add_argument("--runner", default=None,
                    help="tsr_rv runner command (executable + TSR_RV_RUNNER_ARGS); empty: skip")
    ap.add_argument("--lockstep", action="store_true",
                    help="riscv-tests: also compare commit logs (needs --runner and Spike)")
    ap.add_argument("--spike-optional", action="store_true",
                    help="with --runner: run the riscv-tests on tsr_rv alone when Spike is absent")
    ap.add_argument("--compare-csrs", action="store_true",
                    help="with --lockstep, also compare CSR writes in the commit logs")
    ap.add_argument("--only", default=None, help="regex: run only matching tests")
    ap.add_argument("--build-only", action="store_true", help="compile, do not run")
    ap.add_argument("--list", action="store_true", help="list the selected tests and exit")
    ap.add_argument("--env", choices=("tsr", "upstream"), default="tsr",
                    help="riscv-tests environment (default tsr)")
    ap.add_argument("--spike-profile", choices=sorted(rc.SPIKE_PROFILES), default=None,
                    help="Spike profile (default: upstream for --env upstream, else test)")


# --------------------------------------------------------------------------------------------
# riscv-tests
# --------------------------------------------------------------------------------------------

def makefrag_tests(src, suite):
    """Test names of <suite>_sc_tests in isa/<suite>/Makefrag."""
    txt = (Path(src) / "isa" / suite / "Makefrag").read_text().replace("\\\n", " ")
    m = re.search(r"^%s_sc_tests\s*=\s*(.*)$" % suite, txt, re.M)
    if not m:
        raise SystemExit("%s/isa/%s/Makefrag: no %s_sc_tests" % (src, suite, suite))
    return m.group(1).split()


def riscv_tests_items(a, tools, src):
    env_inc = ([str(rc.HERE / "env")] if a.env == "tsr" else []) + [str(src / "env" / "p")]
    out = Path(a.out) / ("riscv-tests-" + a.env)
    items = []
    for suite in RISCV_TEST_SUITES:
        march = "rv32imc_zicsr_zifencei" if suite == "rv32uc" else "rv32im_zicsr_zifencei"
        for t in makefrag_tests(src, suite):
            name = "%s-p-%s" % (suite, t)
            elf = out / name
            cmd = [tools.gcc(), "-march=" + march, "-mabi=ilp32", "-static", "-mcmodel=medany",
                   "-fvisibility=hidden", "-nostdlib", "-nostartfiles"]
            cmd += ["-I" + d for d in env_inc]
            cmd += ["-I" + str(src / "isa" / "macros" / "scalar"),
                    "-T" + str(src / "env" / "p" / "link.ld"),
                    str(src / "isa" / suite / (t + ".S")), "-o", str(elf), "-MD", "-MF",
                    str(elf) + ".d"]
            items.append((name, elf, cmd))
    return items


# --------------------------------------------------------------------------------------------
# riscv-arch-test
# --------------------------------------------------------------------------------------------

_ISA = re.compile(r'RVTEST_ISA\(\s*"([^"]+)"\s*\)')
_CASE = re.compile(r'RVTEST_CASE\(\s*\d+\s*,\s*"([^"]*)"')


def isa_to_march(isa):
    """riscof ISA string (e.g. RV32I_Zicsr, RV32IC) -> gcc -march, always with zicsr and zifencei
    (they only let the assembler accept CSR / fence.i; C is added only where the test has it)."""
    m = re.match(r"^RV32([A-Z]+)((?:_Z[A-Za-z0-9]+)*)$", isa.replace(" ", ""))
    if not m:
        raise ValueError("unsupported RVTEST_ISA %r" % isa)
    exts = {e.lower() for e in m.group(2).split("_") if e} | {"zicsr", "zifencei"}
    return "rv32" + m.group(1).lower() + "".join("_" + e for e in sorted(exts))


def select_case(text):
    """(defines, None) for the first RVTEST_CASE whose checks the profile satisfies, or
    (None, reason) when the test does not apply to the profile."""
    reasons = []
    for case in _CASE.findall(text):
        ok, defs, why = True, [], ""
        for item in case.replace("//", " ").split(";"):
            item = item.strip()
            if not item:
                continue
            if item.startswith("check "):
                key, _, val = item[6:].partition(":=")
                key, val = key.strip(), val.strip()
                if key == "ISA":
                    rm = re.match(r"^regex\((.*)\)$", val)
                    if not rm or not re.match(rm.group(1), rc.PROFILE_ISA):
                        ok, why = False, "ISA %s" % val
                elif key == "hw_data_misaligned_support":
                    if (val == "True") != rc.PROFILE_MISALIGNED:
                        ok, why = False, "hw_data_misaligned_support:=%s" % val
                else:
                    ok, why = False, "unknown check %r" % item
            elif item.startswith("def "):
                defs.append(item[4:].strip())
            if not ok:
                break
        if ok:
            return defs, None
        reasons.append(why)
    return None, "; ".join(reasons) or "no RVTEST_CASE"


def arch_test_items(a, tools, src, skipped):
    env = src / "riscv-test-suite" / "env"
    out = Path(a.out) / "arch-test"
    items = []
    for suite in ARCH_SUITES:
        d = src / "riscv-test-suite" / "rv32i_m" / suite / "src"
        for s in sorted(d.glob("*.S")):
            name = "%s/%s" % (suite, s.stem)
            text = s.read_text(errors="replace")
            isa = _ISA.search(text)
            defs, why = select_case(text)
            if isa is None or defs is None:
                skipped.append((name, why or "no RVTEST_ISA"))
                continue
            elf = out / suite / (s.stem + ".elf")
            cmd = [tools.gcc(), "-march=" + isa_to_march(isa.group(1)), "-mabi=ilp32", "-static",
                   "-mcmodel=medany", "-fvisibility=hidden", "-nostdlib", "-nostartfiles", "-g",
                   "-T" + str(rc.HERE / "arch" / "link.ld"), "-I" + str(rc.HERE / "arch"),
                   "-I" + str(env), "-DXLEN=32"] + ["-D" + x for x in defs]
            cmd += [str(s), "-o", str(elf), "-MD", "-MF", str(elf) + ".d"]
            items.append((name, elf, cmd))
    return items


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------

def run_riscv_test(a, tools, runner, profile, item):
    name, elf, _ = item
    res = []
    if tools.have_spike():
        log = Path(str(elf) + ".spike.log") if a.lockstep and runner else None
        code, so, se = rc.run(rc.spike_cmd(tools, profile, elf, log=log), a.timeout,
                              env=tools.spike_env())
        if code != 0:
            return False, "spike: exit %s\n%s%s" % (code, so, se)
        res.append("spike")
    if runner and name not in RUNNER_EXCLUDE:
        log = Path(str(elf) + ".tsr.log") if a.lockstep else None
        code, so, se = rc.run(rc.runner_cmd(runner, elf, log=log), a.timeout)
        if code != 0:
            return False, "tsr_rv: exit %s\n%s%s" % (code, so, se)
        res.append("tsr_rv")
        if a.lockstep:
            tohost = rc.elf_symbols(elf).get("tohost")
            with open(str(elf) + ".spike.log", errors="replace") as f:
                ref = rc.until_store(rc.parse_commit_log(f), tohost)
            with open(str(elf) + ".tsr.log", errors="replace") as f:
                dut = rc.until_store(rc.parse_commit_log(f), tohost)
            ok, msg = rc.compare_retires(ref, dut, compare_csrs=a.compare_csrs)
            if not ok:
                return False, "lockstep: " + msg
            res.append("lockstep %d" % len(ref))
    if not res:                     # no Spike and excluded on tsr_rv (RUNNER_EXCLUDE)
        return True, "not run: " + RUNNER_EXCLUDE.get(name, "no Spike")
    return True, ", ".join(res)


def run_arch_test(a, tools, runner, profile, item):
    name, elf, _ = item
    ref = Path(str(elf) + ".spike.sig")
    code, so, se = rc.run(rc.spike_cmd(tools, profile, elf, signature=ref), a.timeout,
                          env=tools.spike_env())
    if code != 0:
        return False, "spike: exit %s\n%s%s" % (code, so, se)
    if not ref.exists() or not rc.read_signature(ref):
        return False, "spike: no signature"
    if not runner:
        return True, "spike, %d signature words" % len(rc.read_signature(ref))
    dut = Path(str(elf) + ".tsr.sig")
    if dut.exists():
        dut.unlink()
    code, so, se = rc.run(rc.runner_cmd(runner, elf, signature=dut), a.timeout)
    if code != 0:
        return False, "tsr_rv: exit %s\n%s%s" % (code, so, se)
    if not dut.exists():
        return False, "tsr_rv: no signature file"
    ok, msg = rc.compare_signatures(ref, dut, rc.elf_symbols(elf).get("begin_signature"))
    return ok, msg


def do_suite(a, tools, runner, suite):
    """Returns (n_pass, failures list, n_run)."""
    tp = Path(a.tools_prefix)
    skipped = []
    if suite == "riscv-tests":
        src = Path(a.riscv_tests or tp / "src" / "riscv-tests")
        if not (src / "isa" / "rv32ui" / "Makefrag").exists() or not (src / "env" / "p").exists():
            rc.skip("riscv-tests sources not found at %s; %s" % (src, rc.INSTALL_HINT))
        items = riscv_tests_items(a, tools, src)
        for t, why in PROFILE_EXCLUDE.items():
            if a.env == "tsr":
                skipped.append((t, why))
        if a.env == "tsr":
            items = [i for i in items if i[0] not in PROFILE_EXCLUDE]
        if runner:
            skipped += [(t, "tsr_rv: " + why) for t, why in RUNNER_EXCLUDE.items()]
        if runner and not tools.have_spike():   # --spike-optional: nothing would run them
            items = [i for i in items if i[0] not in RUNNER_EXCLUDE]
        if a.lockstep:
            skipped += [(t, "lockstep: " + why) for t, why in LOCKSTEP_EXCLUDE.items()]
            items = [i for i in items if i[0] not in LOCKSTEP_EXCLUDE and i[0] not in RUNNER_EXCLUDE]
        profile = a.spike_profile or ("upstream" if a.env == "upstream" else "test")
        if runner and profile != "test":
            raise SystemExit("--runner needs the Spike profile 'test'")
        fn = run_riscv_test
    else:
        src = Path(a.arch_test or tp / "src" / "riscv-arch-test")
        if not (src / "riscv-test-suite" / "env" / "arch_test.h").exists():
            rc.skip("riscv-arch-test sources not found at %s; %s" % (src, rc.INSTALL_HINT))
        items = arch_test_items(a, tools, src, skipped)
        profile = a.spike_profile or "test"
        fn = run_arch_test
    if a.only:
        items = [i for i in items if re.search(a.only, i[0])]
    if a.list:
        for i in items:
            print(i[0], " ".join(i[2]))
        for t, why in skipped:
            print("# not run: %s (%s)" % (t, why))
        return len(items), [], 0
    errs = rc.compile_all(a.jobs, [(i[1], i[2]) for i in items])
    if errs:
        for out, e in errs:
            print("BUILD FAIL %s\n%s" % (out, e))
        return 0, ["build: %d failures" % len(errs)], len(items)
    print("%s: %d programs built in %s (%d not applicable to the profile)" %
          (suite, len(items), a.out, len(skipped)))
    for t, why in skipped:
        print("  not run: %s: %s" % (t, why))
    if a.build_only:
        return len(items), [], 0
    results = rc.run_all(a.jobs, lambda it: fn(a, tools, runner, profile, it), items)
    fails = []
    for it, (ok, msg) in zip(items, results):
        print("%s %s (%s)" % ("PASS" if ok else "FAIL", it[0], (msg.splitlines() or [""])[0] if ok else ""))
        if not ok:
            print("    " + msg.replace("\n", "\n    "))
            fails.append(it[0])
    return len(items) - len(fails), fails, len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("suite", choices=("riscv-tests", "arch-test", "all"))
    common_args(ap)
    a = ap.parse_args()
    tools = rc.Tools.from_args(a)
    runner = rc.parse_runner(a.runner)
    if a.runner is not None and runner is None:
        rc.skip("no tsr_rv runner (P1.7; contract in sw/sim/tests/rvtests/rvcommon.py)")
    if not tools.have_gcc():
        rc.skip("RISC-V compiler %s not found; %s" % (tools.gcc(), rc.INSTALL_HINT))
    if a.lockstep and not runner:
        raise SystemExit("--lockstep needs --runner")
    if a.spike_optional and a.suite != "riscv-tests":
        raise SystemExit("--spike-optional applies to the riscv-tests suite only (the arch-test "
                         "signatures are compared with Spike's: run that suite without it)")
    needs_spike = not (a.build_only or a.list)
    if needs_spike and not tools.have_spike():
        if not (a.spike_optional and runner and not a.lockstep):
            rc.skip("Spike %s not found; %s" % (tools.spike, rc.INSTALL_HINT))
        print("Spike %s not found: the riscv-tests run on tsr_rv only (self-checking); %s" %
              (tools.spike, rc.INSTALL_HINT))
    suites = ("riscv-tests", "arch-test") if a.suite == "all" else (a.suite,)
    total_fail, total_pass = [], 0
    for s in suites:
        n_pass, fails, _ = do_suite(a, tools, runner, s)
        total_pass += n_pass
        total_fail += fails
    if not (a.list or a.build_only):
        print("summary: %d passed, %d failed%s" % (total_pass, len(total_fail),
              (": " + " ".join(total_fail)) if total_fail else ""))
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())

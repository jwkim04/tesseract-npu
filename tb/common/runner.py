"""Build and run a cocotb test on Verilator (cocotb_tools.runner) with the project defaults.

    from common.runner import run
    def test_dma():                                   # pytest function
        run("tsr_dma", __name__, parameters={"NUM_CH": 2, "SPM_BYTES": 65536})

Defaults:
  - sources: every file of rtl/filelist.f (packages, generic tech, the control subsystem through
    its nested rtl/ctrl/filelist.f, core) plus `extra_sources`; the filelist's +incdir / +define
    options and waiver files (.vlt) go into every build's arguments (rtl_args)
  - build directory: $TSR_SIM_BUILD_ROOT (else <repo>/build/cocotb) / <name>_<hash of
    toplevel, parameters, sources, args>; reused across runs (Verilator rebuilds what changed)
  - make -j$TSR_BUILD_JOBS (default 4), --assert, X randomisation (--x-assign/--x-initial unique,
    +verilator+rand+reset+2 seeded with the test seed; run(rand_reset=0) or TSR_RAND_RESET=0 starts everything
    at zero, as the FPGA after configuration, 1 at ones),
    default Verilator warnings fatal
  - waves: run(..., waves=True) or WAVES=1 -> VCD (dump.vcd in the build directory)
  - seed: $TSR_SEED (pytest --tsr-seed sets it) is passed to the simulation as TSR_SEED and
    COCOTB_RANDOM_SEED
Under pytest a failing cocotb test makes runner.test() raise SystemExit (pytest reports a
failure); outside pytest run() raises RuntimeError.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RTL = REPO / "rtl"
TB = REPO / "tb"
if str(TB) not in sys.path:          # the simulator's Python inherits sys.path as PYTHONPATH
    sys.path.insert(0, str(TB))


def parse_filelist(filelist: Path | str = RTL / "filelist.f"):
    """(sources, args) of a Verilator filelist (// comments; paths relative to the directory of the
    list that names them). Nested lists (-F / -f) are expanded in place. sources: the HDL files in
    order; args: the options to pass before them, in order: +incdir+<abs> and +define+ lines and the
    Verilator configuration files (*.vlt, e.g. rtl/ctrl/waivers/vendor.vlt, which must precede every
    source; cocotb_tools does not classify .vlt sources). rtl/filelist.f includes rtl/ctrl/filelist.f
    this way (W4-A)."""
    filelist = Path(filelist)
    srcs, args = [], []
    for line in filelist.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-F ", "-f ")):
            s2, a2 = parse_filelist((filelist.parent / line[3:].strip()).resolve())
            srcs += s2
            args += [a for a in a2 if a not in args]
        elif line.startswith("+incdir+"):
            a = "+incdir+" + str((filelist.parent / line[len("+incdir+"):]).resolve())
            if a not in args:
                args.append(a)
        elif line.startswith("+define+"):
            if line not in args:
                args.append(line)
        elif line.endswith(".vlt"):
            a = str((filelist.parent / line).resolve())
            if a not in args:
                args.append(a)
        else:
            srcs.append((filelist.parent / line).resolve())
    return srcs, args


def rtl_sources(filelist: Path | str = RTL / "filelist.f") -> list:
    """HDL source files of a filelist, nested lists expanded (parse_filelist)."""
    return parse_filelist(filelist)[0]


def rtl_args(filelist: Path | str = RTL / "filelist.f") -> list:
    """Verilator options of a filelist: include directories, defines, configuration files
    (parse_filelist); run() passes those of rtl/filelist.f to every build."""
    return parse_filelist(filelist)[1]


def mutated_sources(sources, name: str, old: str, new: str, tag: str) -> list:
    """`sources` with one text replacement in the file called `name`: a mutant that a regression test must reject. The
    changed copy goes to <build root>/mutants/<tag>; `old` must occur exactly once."""
    d = build_root() / "mutants" / tag
    d.mkdir(parents=True, exist_ok=True)
    out = []
    hit = 0
    for src in sources:
        src = Path(src)
        if src.name == name:
            txt = src.read_text()
            assert txt.count(old) == 1, (name, old, txt.count(old))
            dst = d / name
            changed = txt.replace(old, new)
            if not dst.exists() or dst.read_text() != changed:
                dst.write_text(changed)
            src = dst
            hit += 1
        out.append(src)
    assert hit == 1, (name, hit)
    return out


def build_root() -> Path:
    return Path(os.environ.get("TSR_SIM_BUILD_ROOT", REPO / "build" / "cocotb"))


def seed_from_env() -> int:
    return int(os.environ.get("TSR_SEED", "1"), 0)


def run(toplevel: str, test_module: str, *, parameters: dict | None = None, extra_sources=(),
        sources=None, name: str | None = None, waves: bool | None = None, testcase=None,
        seed: int | None = None, extra_env: dict | None = None, build_args=(), plusargs=(),
        x_random: bool = True, rand_reset: int | None = None, timescale=("1ns", "1ps")) -> Path:
    from cocotb_tools import runner as ct_runner

    params = dict(parameters or {})
    srcs = [Path(s).resolve() for s in (sources if sources is not None else rtl_sources())]
    srcs += [Path(s).resolve() for s in extra_sources]
    seed = seed_from_env() if seed is None else seed
    if waves is None:
        waves = bool(int(os.environ.get("WAVES", "0")))
    # the filelist's include directories, defines and waiver files first (the vendored IP of
    # rtl/ctrl needs them; they change nothing for a build that does not use it)
    args = rtl_args() + ["--assert"] + (["--x-assign", "unique", "--x-initial", "unique"] if x_random else [])
    if waves:
        args += ["--trace-structs"]
    args += list(build_args)
    key = hashlib.sha1(repr((toplevel, sorted((k, str(v)) for k, v in params.items()),
                             [str(s) for s in srcs], args)).encode()).hexdigest()[:10]
    bdir = build_root() / ("%s_%s" % (name or toplevel, key))

    ct_runner.MAX_PARALLEL_BUILD_JOBS = int(os.environ.get("TSR_BUILD_JOBS", "4"))
    r = ct_runner.get_runner("verilator")
    r.build(sources=srcs, hdl_toplevel=toplevel, parameters=params, build_args=args,
            build_dir=bdir, waves=waves, timescale=timescale)
    env = {"TSR_SEED": str(seed)}
    env.update(extra_env or {})
    pa = list(plusargs)
    if x_random:
        # rand_reset (else $TSR_RAND_RESET): 0 starts every variable at zero (the FPGA after configuration), 1 at all
        # ones, 2 (default) at random values
        rr = rand_reset if rand_reset is not None else int(os.environ.get("TSR_RAND_RESET", "2"))
        pa += ["+verilator+rand+reset+%d" % rr,
               "+verilator+seed+%d" % (seed & 0x7FFFFFFF or 1)]
    results = r.test(test_module=test_module, hdl_toplevel=toplevel, build_dir=bdir,
                     test_dir=bdir, testcase=testcase, seed=seed, extra_env=env, plusargs=pa,
                     waves=waves, parameters=params)
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        from cocotb_tools.check_results import get_results
        n, failed = get_results(results)
        if failed:
            raise RuntimeError("%d of %d cocotb tests failed (%s)" % (failed, n, results))
    return results

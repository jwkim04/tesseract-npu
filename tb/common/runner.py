"""Build and run a cocotb test on Verilator (cocotb_tools.runner) with the project defaults.

    from common.runner import run
    def test_dma():                                   # pytest function
        run("tsr_dma", __name__, parameters={"NUM_CH": 2, "SPM_BYTES": 65536})

Defaults:
  - sources: every file of rtl/filelist.f (packages, generic tech, core) plus `extra_sources`
  - build directory: $TSR_SIM_BUILD_ROOT (else <repo>/build/cocotb) / <name>_<hash of
    toplevel, parameters, sources, args>; reused across runs (Verilator rebuilds what changed)
  - make -j$TSR_BUILD_JOBS (default 4), --assert, X randomisation (--x-assign/--x-initial unique,
    +verilator+rand+reset+2 seeded with the test seed), default Verilator warnings fatal
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


def rtl_sources(filelist: Path | str = RTL / "filelist.f") -> list:
    """Source files of a filelist (paths relative to the filelist's directory; // comments)."""
    filelist = Path(filelist)
    out = []
    for line in filelist.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if line:
            out.append((filelist.parent / line).resolve())
    return out


def build_root() -> Path:
    return Path(os.environ.get("TSR_SIM_BUILD_ROOT", REPO / "build" / "cocotb"))


def seed_from_env() -> int:
    return int(os.environ.get("TSR_SEED", "1"), 0)


def run(toplevel: str, test_module: str, *, parameters: dict | None = None, extra_sources=(),
        sources=None, name: str | None = None, waves: bool | None = None, testcase=None,
        seed: int | None = None, extra_env: dict | None = None, build_args=(), plusargs=(),
        x_random: bool = True, timescale=("1ns", "1ps")) -> Path:
    from cocotb_tools import runner as ct_runner

    params = dict(parameters or {})
    srcs = [Path(s).resolve() for s in (sources if sources is not None else rtl_sources())]
    srcs += [Path(s).resolve() for s in extra_sources]
    seed = seed_from_env() if seed is None else seed
    if waves is None:
        waves = bool(int(os.environ.get("WAVES", "0")))
    args = ["--assert"] + (["--x-assign", "unique", "--x-initial", "unique"] if x_random else [])
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
        pa += ["+verilator+rand+reset+2", "+verilator+seed+%d" % (seed & 0x7FFFFFFF or 1)]
    results = r.test(test_module=test_module, hdl_toplevel=toplevel, build_dir=bdir,
                     test_dir=bdir, testcase=testcase, seed=seed, extra_env=env, plusargs=pa,
                     waves=waves, parameters=params)
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        from cocotb_tools.check_results import get_results
        n, failed = get_results(results)
        if failed:
            raise RuntimeError("%d of %d cocotb tests failed (%s)" % (failed, n, results))
    return results

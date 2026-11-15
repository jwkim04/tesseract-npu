"""tb/qact tsr_qact_scale — the divider-free A32 scales of QACT against libtsr_ref, bit for bit.

For every amax a block can have (|DAZ(x)| bits, 0 .. 0x7F800000, never NaN) the pipeline's
  d = tsr_fdiv(amax, 127), scale = (d != +-0) ? tsr_frecip(d) : +0, h16 = tsr_f32_to_fp16(d), zero = (amax == 0)
are compared with the reference library, together with out_valid (fixed latency, one block per cycle, idle lanes
and cycles).

Harness: tb/qact/cpp/qact_scale_sweep.cpp on tb/qact/hdl/tsr_qact_scale_tb.sv (64 lanes of tsr_qact_scale), linked
with libtsr_ref.so; built like tb/vu/test_vu_arith.py's.
  test_qact_scale_quick  every mantissa at the exponents 1, 2, 6, 7, 8, 9, 126, 127, 128, 133, 134, 135, 253, 254,
                         amax 0 and Inf, the mantissas 0, 1, 0x400000, max at every exponent, 2^22 random values
                         with idle lanes and cycles
  test_qact_scale_full   opt-in (-m slow, TSR_QACT_SLOW=1, -k scale_full): every amax pattern from 0 to
                         0x7F800000 (2,139,095,041 patterns), split into ranges over TSR_QACT_SWEEP_PROCS processes
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from common.runner import REPO, build_root, rtl_sources
from common.tsr_ref import TsrRef

HERE = Path(__file__).resolve().parent
SLOW_ENV = "TSR_QACT_SLOW"
NAMES = ("tsr_tables_pkg.sv", "tsr_pkg.sv", "tsr_fp_pkg.sv", "tsr_vu_pkg.sv", "tsr_vu_rcp.sv", "tsr_qact_scale.sv")
QUICK_EXPS = [1, 2, 6, 7, 8, 9, 126, 127, 128, 133, 134, 135, 253, 254]
AMAX_END = 0x7F800001


def slow_requested(config, name: str) -> bool:
    if os.environ.get(SLOW_ENV, "0") not in ("", "0"):
        return True
    if re.search(r"\bslow\b", config.getoption("markexpr", "") or ""):
        return True
    short = name[len("test_qact_"):] if name.startswith("test_qact_") else name
    if short in (config.getoption("keyword", "") or ""):
        return True
    return any(("::" + name) in str(a) for a in config.args)


def build_harness() -> Path:
    lib = Path(TsrRef().path).resolve()
    by_name = {s.name: s for s in rtl_sources()}
    srcs = [by_name[n] for n in NAMES]
    alt = os.environ.get("TSR_QACT_SCALE_RTL")      # seeded-bug check: the bench must fail on another source
    if alt:
        srcs = [Path(alt).resolve() if s.name == "tsr_qact_scale.sv" else s for s in srcs]
    bdir = build_root() / ("qact_scale_sweep_" + ("mut" if alt else "ref"))
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = os.environ.get("TSR_BUILD_JOBS", "4")
    cmd = ["verilator", "--cc", "--exe", "--build", "-O3", "-j", jobs, "--top-module", "tsr_qact_scale_tb",
           "-Wno-fatal", "-CFLAGS", "-O2 -I%s" % (REPO / "sw" / "sim" / "include"),
           "-LDFLAGS", "-L%s -ltsr_ref -Wl,-rpath,%s" % (lib.parent, lib.parent),
           "--Mdir", str(bdir), "-o", "qact_scale_sweep"]
    cmd += [str(s) for s in srcs] + [str(HERE / "hdl" / "tsr_qact_scale_tb.sv"), str(HERE / "cpp" / "qact_scale_sweep.cpp")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "qact_scale_sweep build failed:\n%s\n%s" % (p.stdout[-4000:], p.stderr[-4000:])
    return bdir / "qact_scale_sweep"


def run_jobs(exe: Path, jobs, procs: int):
    pending, running, failed = list(jobs), [], []
    while pending or running:
        while pending and len(running) < procs:
            job = pending.pop(0)
            running.append((job, subprocess.Popen([str(exe)] + [str(j) for j in job], stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, text=True)))
        job, p = running.pop(0)
        out = p.communicate()[0]
        print("%s %s:\n%s" % (exe.name, " ".join(str(j) for j in job), out.strip()))
        if p.returncode != 0:
            failed.append((job, out[-3000:]))
    assert not failed, "sweep failures:\n%s" % "\n".join("%s:\n%s" % f for f in failed)


def test_qact_scale_quick():
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    exe = build_harness()
    jobs = [("specials",), ("rand", 1 << 22, seed)]
    jobs += [("exps", *QUICK_EXPS[i::4]) for i in range(4)]
    run_jobs(exe, jobs, int(os.environ.get("TSR_QACT_SWEEP_PROCS", "4")))


@pytest.mark.slow
def test_qact_scale_full(request):
    if not slow_requested(request.config, request.node.originalname):
        pytest.skip("slow test, opt-in: -m slow, %s=1, -k scale_full or its node id" % SLOW_ENV)
    exe = build_harness()
    chunk = 1 << 24
    jobs = [("range", a, min(a + chunk, AMAX_END)) for a in range(0, AMAX_END, chunk)]
    run_jobs(exe, jobs, int(os.environ.get("TSR_QACT_SWEEP_PROCS", "8")))

"""tb/vu arithmetic — the VU's fixed-latency arithmetic blocks against libtsr_ref, bit for bit
(docs/design/02-vector-unit.md "VU" §10 item 4: "tsr_vu_exp and tsr_vu_rcp exhaustive on a Verilator C++
bench (opt-in; RC over all 2^23 mantissas x exponent edges in the default run)").

Harness: tb/vu/cpp/vu_sweep.cpp on tb/vu/hdl/tsr_vu_arith_tb.sv (16 lanes of tsr_vu_exp, tsr_vu_rcp,
tsr_vu_sqrt), linked with libtsr_ref.so; tb/vu/cpp/vu_lane_sweep.cpp on tb/vu/hdl/tsr_vu_lane_tb.sv (the lane
functions, incl. GELU16 over all 65,536 fp16 inputs).
  test_vu_arith_quick   EXPK: every κ on 2^24 strided points + 2^20 boundary-biased randoms + specials; RC
                        and SQ: every mantissa at the exponent edges (1, 2, 126, 127, 128, 252, 253, 254) and
                        both signs, 2^22 randoms, specials (about 40 s)
  test_vu_arith_full    opt-in (-m slow, TSR_VU_SLOW=1, -k arith_full): EXPK over all 2^32 inputs for each κ,
                        RC and SQ over all 2^32 inputs
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from common.runner import REPO, build_root
from common.tsr_ref import TsrRef

HERE = Path(__file__).resolve().parent
SLOW_ENV = "TSR_VU_SLOW"
CORE = REPO / "rtl" / "core"
ARITH_SOURCES = [REPO / "rtl/core/tsr_fp_mul.sv", REPO / "rtl/core/tsr_fp_add.sv", CORE / "tsr_vu_pkg.sv",
                 CORE / "tsr_vu_dly.sv", CORE / "tsr_vu_exp.sv", CORE / "tsr_vu_rcp.sv", CORE / "tsr_vu_sqrt.sv",
                 CORE / "tsr_vu_sincos.sv"]


def slow_requested(config, name: str) -> bool:
    if os.environ.get(SLOW_ENV, "0") not in ("", "0"):
        return True
    if re.search(r"\bslow\b", config.getoption("markexpr", "") or ""):
        return True
    short = name[len("test_vu_"):] if name.startswith("test_vu_") else name
    if short in (config.getoption("keyword", "") or ""):
        return True
    return any(("::" + name) in str(a) for a in config.args)


def build_harness(top: str, sources, cpp: Path, exe: str, defines=()) -> Path:
    lib = Path(TsrRef().path).resolve()
    vl = os.environ.get("VERILATOR", "verilator")
    d = os.environ.get("TSR_MUT_DIR")
    srcs = [Path(d) / s.name if d and (Path(d) / s.name).exists() else s for s in sources]
    tag = "mut" if d else "ref"
    bdir = build_root() / ("%s_%s" % (exe, tag))
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = os.environ.get("TSR_BUILD_JOBS", "4")
    cmd = [vl, "--cc", "--exe", "--build", "-O3", "-j", jobs, "--top-module", top, "-Wno-fatal",
           "-CFLAGS", "-O2 -I%s %s" % (REPO / "sw" / "sim" / "include", " ".join("-D%s" % x for x in defines)),
           "-LDFLAGS", "-L%s -ltsr_ref -Wl,-rpath,%s" % (lib.parent, lib.parent),
           "--Mdir", str(bdir), "-o", exe]
    cmd += [str(REPO / "rtl/core/tsr_tables_pkg.sv"), str(REPO / "rtl/core/tsr_pkg.sv"),
            str(REPO / "rtl/core/tsr_fp_pkg.sv")]
    cmd += [str(s) for s in srcs] + [str(cpp)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "%s build failed:\n%s\n%s" % (exe, p.stdout[-4000:], p.stderr[-4000:])
    return bdir / exe


def run_jobs(exe: Path, jobs, procs=None):
    procs = procs or int(os.environ.get("TSR_VU_SWEEP_PROCS", "2"))
    pending, running, failed, outs = list(jobs), [], [], []
    while pending or running:
        while pending and len(running) < procs:
            job = pending.pop(0)
            running.append((job, subprocess.Popen([str(exe)] + [str(j) for j in job], stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, text=True)))
        job, p = running.pop(0)
        out = p.communicate()[0]
        outs.append(out.strip())
        print("%s %s:\n%s" % (exe.name, " ".join(str(j) for j in job), out.strip()))
        if p.returncode != 0:
            failed.append((job, out[-3000:]))
    assert not failed, "sweep failures:\n%s" % "\n".join("%s:\n%s" % f for f in failed)
    return outs


EDGE_EXPS = ["1", "2", "126", "127", "128", "252", "253", "254"]
LANE_SOURCES = [REPO / "rtl/core/tsr_fp_mul.sv", REPO / "rtl/core/tsr_fp_add.sv", CORE / "tsr_vu_pkg.sv",
                CORE / "tsr_vu_dly.sv", CORE / "tsr_vu_exp.sv", CORE / "tsr_vu_rcp.sv", CORE / "tsr_vu_lane.sv"]


def lane_exe():
    return build_harness("tsr_vu_lane_tb", LANE_SOURCES + [HERE / "hdl" / "tsr_vu_lane_tb.sv"],
                         HERE / "cpp" / "vu_lane_sweep.cpp", "vu_lane_sweep")


def test_vu_lane_quick():
    """Every lane function against the references (2^17 elements each), GELU16 / GEGLU16 over all fp16
    inputs and their neighbors, AT's micro-ops, and 3000 rows into the partials."""
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    exe = lane_exe()
    run_jobs(exe, [("vop", 1 << 17, seed), ("gelu16", seed), ("at", 1 << 17, seed), ("red", 3000, seed)])


def arith_exe():
    return build_harness("tsr_vu_arith_tb", ARITH_SOURCES + [HERE / "hdl" / "tsr_vu_arith_tb.sv"],
                         HERE / "cpp" / "vu_sweep.cpp", "vu_sweep")


def test_vu_arith_quick():
    seed = int(os.environ.get("TSR_SEED", "1"), 0)
    exe = arith_exe()
    run_jobs(exe, [("specials",), ("exp", 4, 0, 1 << 32, 1 << 8), ("exp_rand", 1 << 20, seed),
                   ("rcp", *EDGE_EXPS[:4]), ("rcp", *EDGE_EXPS[4:]), ("sqrt", *EDGE_EXPS[:4]),
                   ("sqrt", *EDGE_EXPS[4:]), ("rand", 1 << 22, seed),
                   ("sincos", 0, 0x48800001, 1 << 9), ("sincos_rand", 1 << 20, seed)])


@pytest.mark.slow
def test_vu_arith_full(request):
    if not slow_requested(request.config, request.node.originalname):
        pytest.skip("slow test, opt-in: -m slow, %s=1, -k arith_full or its node id" % SLOW_ENV)
    exe = arith_exe()
    jobs = [("exp", k, a << 30, (a + 1) << 30, 1) for k in range(4) for a in range(4)]
    jobs += [("rcp", *[str(e) for e in range(0, 256, 1) if e % 4 == q]) for q in range(4)]
    jobs += [("sqrt", *[str(e) for e in range(0, 256, 1) if e % 4 == q]) for q in range(4)]
    run_jobs(exe, jobs, procs=int(os.environ.get("TSR_VU_SWEEP_PROCS", "4")))

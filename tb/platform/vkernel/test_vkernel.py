"""tb/platform/vkernel: the host software against the Verilated kernel (TSR_DEVICE=vkernel,
docs/platform-u200.md §13), and a kernel-level fastreplay smoke at the U200 bitstream parameters.

  test_vkernel_selftest[<ctrl>-<regime>]  tsr_xrt_selftest (compiled here from sw/tools, sw/transport
                          and sw/sim, no CMake) over TSR_DEVICE=vkernel: the real protocol layer
                          (tsr_kdev.cpp) and the vkernel client against the server
                          (vkernel_server.cpp) built for the ch3 configuration (3 channels, T_MAX 3,
                          SPM 256 KiB) with the U200 implementation parameters on
                          rtl/filelist_xilinx.f. Identity, memory ranges, the 9 programs against the
                          ISA simulator, traces/m3/ch3 + m2/ch3; once per control mode (kds = the XRT
                          scheduler's sequence with server-side polling, irq = interrupt mode, regs =
                          TSR_XRT_CTRL=regs) in a different memory regime each; the server must
                          report 0 findings (AXI4 / AXI4-Lite checks, memory-model rules, RTL
                          assertions, quiescence at every completion)
  test_vkernel_adapter_errors  SLVERR / DECERR injected on R and B by the AXI4 memories: the
                          protocol layer must report ADAPTER_STATUS and the selftest must fail
  test_vkernel_packaged_k1  the generated top tsr_u200_k1 (kwrap_tsr_u200_k1, its gen_tops.py
                          parameters) behind the server: identity, memory and the 9 programs
  test_kernel_u200_smoke[t1|t8]  tb/fastreplay/run.py --top kernel with the U200 bitstream
                          parameters (SPM_RD_LAT 4, MVM_BC_PIPE 2, MVM_RES_PIPE 2, MVM_MAC_LAT 9,
                          RST_PIPE 3, AXI_SLICES 2, RST_STAGES 3) on rtl/filelist_xilinx.f:
                          traces/m3/tmax1 + m3/errors + gen/tmax1 (4 channels, T_MAX 1, regimes
                          rotating, 2 passes per trace) and traces/m3/tmax8 + m3/errors8 (T_MAX 8,
                          regimes rotating, 1 pass per trace)
Builds: the server under $TSR_SIM_BUILD_ROOT/vkernel (vkernel.py), run.py's cache for the smoke,
the selftest under the server's build root (host/).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TR = REPO / "traces"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vk = _load("vkernel_build", HERE / "vkernel.py")
fr = vk.fr


def _log(*a):
    print(*a)


def _selftest_binary() -> Path:
    """tsr_xrt_selftest from sources (the transport with the vkernel client), cached by content."""
    import hashlib
    c_srcs = [REPO / "sw" / "sim" / f for f in ("tsr_ref.c", "tsr_sim.c", "tsr_trace.c")] + \
             [REPO / "sw" / "transport" / f for f in ("tsr_transport.c", "tsr_transport_sim.c")]
    cxx_srcs = [REPO / "sw" / "tools" / "tsr_xrt_selftest.cpp"] + \
               [REPO / "sw" / "transport" / f for f in ("tsr_kdev.cpp", "tsr_kdev_mock.cpp", "tsr_kdev_vkernel.cpp")]
    hdrs = sorted((REPO / "sw" / "sim" / "include").glob("*.h")) + sorted((REPO / "sw" / "transport").glob("*.h"))
    h = hashlib.sha256()
    for f in c_srcs + cxx_srcs + hdrs:
        h.update(str(f).encode())
        h.update(f.read_bytes())
    out = vk.build_root() / "host"
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "tsr_xrt_selftest"
    stamp = out / "stamp"
    if binary.is_file() and stamp.is_file() and stamp.read_text() == h.hexdigest():
        return binary
    inc = ["-I%s" % (REPO / "sw" / "sim" / "include"), "-I%s" % (REPO / "sw" / "transport")]
    fp = ["-O2", "-ffp-contract=off", "-fno-fast-math"]
    objs = []
    for s in c_srcs:
        o = out / (s.stem + ".o")
        subprocess.run(["gcc", "-std=c11"] + fp + inc + ["-c", str(s), "-o", str(o)], check=True)
        objs.append(str(o))
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra"] + fp + inc + [str(s) for s in cxx_srcs] + objs +
                   ["-lpthread", "-lm", "-o", str(binary)], check=True)
    stamp.write_text(h.hexdigest())
    return binary


def _run_selftest(tmp_path, ctrl, regime, extra_server=(), traces=("m3/ch3", "m2/ch3"), config="ch3", top="body"):
    server = vk.build(config, top, log=_log)
    selftest = _selftest_binary()
    summary = tmp_path / "summary.json"
    runlog = tmp_path / "runs.jsonl"
    cmd = [str(server), "--regime", regime, "--seed", os.environ.get("TSR_SEED", "1"), "--summary", str(summary),
           "--runlog", str(runlog)] + list(extra_server)
    env = dict(os.environ, TSR_DEVICE="vkernel", TSR_VKERNEL_SERVER=" ".join(cmd), TSR_VKERNEL_CTRL=ctrl,
               TSR_XRT_TIMEOUT_MS="0")
    args = [str(selftest), "--device", "vkernel", "--mem-mib", "1"]
    for t in traces:
        args += ["--traces", str(TR / t)]
    r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=3600)
    print(r.stdout[-4000:], r.stderr[-4000:])
    s = json.loads(summary.read_text()) if summary.is_file() else None
    runs = [json.loads(l) for l in runlog.read_text().splitlines() if l.strip()] if runlog.is_file() else []
    return r, s, runs


@pytest.mark.traces
@pytest.mark.parametrize("ctrl,regime", [("kds", "fast"), ("irq", "rand"), ("regs", "slow")],
                         ids=["kds-fast", "irq-rand", "regs-slow"])
def test_vkernel_selftest(ctrl, regime, tmp_path):
    r, s, runs = _run_selftest(tmp_path, ctrl, regime)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "-> PASS" in r.stdout
    assert "traces: 24 passed, 0 failed" in r.stdout, r.stdout[-3000:]
    assert s is not None and s["findings"] == 0, s
    assert s["runs"] == len(runs) == 9 + 24, (s, len(runs))
    assert s["bad_reads"] == 0, s
    assert max(j["max_t"] for j in runs) == 3          # T = T_MAX MVMs ran


@pytest.mark.traces
def test_vkernel_adapter_errors(tmp_path):
    r, s, _ = _run_selftest(tmp_path, "kds", "rand", ["--mem", "p_rresp_err=0.01", "--mem", "p_bresp_err=0.1"],
                            traces=("m3/ch3",))
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ADAPTER_STATUS" in r.stderr and "error response" in r.stderr, r.stderr[-3000:]
    assert s is not None and s["findings"] == 0, s     # injected errors are not findings: the adapter reports them


def test_vkernel_packaged_k1(tmp_path):
    r, s, runs = _run_selftest(tmp_path, "irq", "rand", traces=(), config="k1", top="packaged")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "kwrap_tsr_u200_k1" in r.stdout and "-> PASS" in r.stdout, r.stdout[-3000:]
    assert s is not None and s["findings"] == 0 and s["runs"] == len(runs) == 9, s


U200 = dict(SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9, RST_PIPE=3, AXI_SLICES=2, RST_STAGES=3)


@pytest.mark.traces
@pytest.mark.parametrize("sets,rotate", [
    (["m3/tmax1", "m3/errors", "gen/tmax1"], 2),
    (["m3/tmax8", "m3/errors8"], 1),
], ids=["t1", "t8"])
def test_kernel_u200_smoke(sets, rotate):
    assert U200 == vk.U200_PARAMS
    rep = fr.replay([TR / s for s in sets], top="kernel", params=dict(U200), filelist=REPO / "rtl" / "filelist_xilinx.f",
                    regimes=["all"], rotate=rotate, log=_log, echo=None)
    for g in rep.groups:
        print(fr.format_group(g))
    assert rep.runs > 0 and rep.failed == 0, [f["errors"][:2] for g in rep.groups for f in g.failures[:5]]
    assert all("MVM_MAC_LAT=9" in g.desc and "RST_PIPE=3" in g.desc for g in rep.groups), [g.desc for g in rep.groups]

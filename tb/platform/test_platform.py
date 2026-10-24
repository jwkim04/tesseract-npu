"""pytest entry of the Alveo XRT platform adapter (tb/platform: Verilator C++ kernel bench, no cocotb).

  test_platform_lint              rtl/platform/alveo_xrt/lint.sh: gen_tops.py --check and the lint
                                  runs of the adapter modules, kernel configurations and tops
  test_kernel_directed[<cfg>]     kernel_bench --directed all (regmap, apctrl, window, errors,
                                  misaligned, reset) on the kernel body built for 1 channel
                                  (small1ch), 3 channels (ch3) and the U200 v1 configuration
  test_kernel_replay[<sets>]      trace replay through the kernel (run.py --top kernel): every
                                  replay check plus the kernel checks (AXI4 / AXI4-Lite protocol,
                                  ap_ctrl_hs state after each run, ADAPTER_STATUS, random bases)
  test_kernel_error_injection     SLVERR / DECERR on R and B during replays: results unchanged,
                                  ADAPTER_STATUS == injected after every run
  test_packaged_k4 / _k1          the packaged tops (tb/platform/sv/kwrap_*): tsr_u200_k4 on
                                  traces/m3/tmax8, tsr_u200_k1 on 1-channel U200 traces generated
                                  with build/dev/bin/tsr-trace-gen (skipped when it is not built)
  test_build_xclbin_no_platform   hw/u200/build_xclbin.sh stops (exit 3) without the platform
  test_build_xclbin_platform      with PLATFORM_REPO_PATHS unset the script finds the extracted
                                  development platform (build/platforms/...) and its platforminfo
                                  shows the DDR bank SLRs of hw/u200/k1.cfg / k4.cfg (skipped
                                  without the extracted platform or Vitis 2024.2)
  test_adapter_mutants            (slow, opt-in: -m slow, TSR_PLATFORM_SLOW=1, -k test_adapter_mutants
                                  or its node id) adapter mutants of tb/platform/mutate_adapter.py are killed
Builds: run.py's cache ($TSR_FASTREPLAY_BUILD / $TSR_SIM_BUILD_ROOT/fastreplay / build/fastreplay).
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TR = REPO / "traces"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fr = _load("fastreplay_run_platform", REPO / "tb" / "fastreplay" / "run.py")


def _log(*a):
    print(*a)


def _cfg(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "meta.json").read_text())["config"]


def _first(set_dir: Path, n: int = 3) -> list:
    return sorted(p for p in set_dir.iterdir() if (p / "meta.json").is_file())[:n]


def test_platform_lint():
    r = subprocess.run([str(REPO / "rtl" / "platform" / "alveo_xrt" / "lint.sh")], capture_output=True, text=True,
                       timeout=1800)
    print(r.stdout[-3000:])
    assert r.returncode == 0, r.stdout[-3000:]


@pytest.mark.traces
@pytest.mark.parametrize("set_dir", ["m3/small1ch", "m3/ch3", "m3/tmax1"], ids=["c1", "c3", "u200"])
def test_kernel_directed(set_dir, tmp_path):
    work = _first(TR / set_dir)
    b = fr.build(_cfg(work[0]), 1, {}, filelist=REPO / "rtl" / "filelist.f", top="kernel", log=_log)
    out = tmp_path / "directed.jsonl"
    r = subprocess.run([str(b.binary), "--directed", "all", "--seed", os.environ.get("TSR_SEED", "1"),
                        "--out", str(out)] + [str(p) for p in work], capture_output=True, text=True, timeout=3600)
    print(r.stdout[-2000:], r.stderr[-4000:])
    assert r.returncode == 0, r.stderr[-4000:]
    res = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [j["directed"] for j in res] == ["regmap", "apctrl", "window", "errors", "misaligned", "reset"]
    assert all(j["ok"] and j["checks"] > 10 for j in res), res


@pytest.mark.traces
@pytest.mark.parametrize("sets,kw", [
    (["m3/small1ch", "m2/small1ch"], dict(regimes=["all"], has_mvm="1")),
    (["m3/ch3", "m2/ch3"], dict(regimes=["all"], has_mvm="1")),
    (["m3/tmax1", "m3/errors", "gen/tmax1"], dict(regimes=["all"], rotate=2)),
], ids=["small1ch", "ch3", "u200"])
def test_kernel_replay(sets, kw):
    rep = fr.replay([TR / s for s in sets], top="kernel", log=_log, echo=None, **kw)
    for g in rep.groups:
        print(fr.format_group(g))
    assert rep.runs > 0 and rep.failed == 0, [f["errors"][:2] for g in rep.groups for f in g.failures[:5]]
    modes = {j["kernel"]["done"] for g in rep.groups for j in g.results}
    assert modes == {"irq", "poll"}, modes


@pytest.mark.traces
def test_kernel_error_injection():
    rep = fr.replay([TR / "m3" / "small1ch", TR / "m2" / "small1ch"], top="kernel", has_mvm="1",
                    regimes=["rand", "slow"], mem=["p_rresp_err=0.02", "p_bresp_err=0.2"], log=_log, echo=None)
    g = rep.groups[0]
    print(fr.format_group(g))
    assert g.failed == 0, [f["errors"][:2] for f in g.failures[:5]]
    inj = sum(j["kernel"]["rresp_err"] + j["kernel"]["bresp_err"] for j in g.results)
    assert inj > 100, inj


@pytest.mark.traces
def test_packaged_k4():
    rep = fr.replay([TR / "m3" / "tmax8"], top="tsr_u200_k4", regimes=["rand", "fast"], max_traces=6,
                    log=_log, echo=None)
    g = rep.groups[0]
    print(fr.format_group(g))
    assert g.runs == 12 and g.failed == 0, [f["errors"][:2] for f in g.failures[:5]]
    assert "tsr_u200_k4" in g.desc


@pytest.mark.traces
def test_packaged_k1(tmp_path):
    gen = REPO / "build" / "dev" / "bin" / "tsr-trace-gen"
    if not gen.is_file():
        pytest.skip("build/dev/bin/tsr-trace-gen not built (1-channel U200 traces for tsr_u200_k1)")
    out = tmp_path / "k1"
    subprocess.run([str(gen), "-q", "--ops", "m3", "--prefix", "k1", "-o", str(out), "-n", "8", "-s", "61",
                    "--errors", "10", "--num-ch", "1", "--max-wbeats", "8192"], check=True, timeout=600)
    rep = fr.replay([out], top="tsr_u200_k1", regimes=["all"], log=_log, echo=None)
    g = rep.groups[0]
    print(fr.format_group(g))
    assert g.runs == 32 and g.failed == 0, [f["errors"][:2] for f in g.failures[:5]]


def test_build_xclbin_no_platform(tmp_path):
    env = dict(os.environ, PLATFORM_REPO_PATHS=str(tmp_path))
    env.pop("TSR_U200_PLATFORM", None)
    r = subprocess.run([str(REPO / "hw" / "u200" / "build_xclbin.sh"), "k1"], capture_output=True, text=True,
                       env=env, timeout=60)
    print(r.stderr)
    if Path("/opt/xilinx/platforms/xilinx_u200_gen3x16_xdma_2_202110_1").exists():
        pytest.skip("the U200 platform is installed")
    assert r.returncode == 3
    assert "is not installed" in r.stderr


def test_build_xclbin_platform():
    plat = REPO / "build" / "platforms" / "root" / "opt" / "xilinx" / "platforms"
    xilinx = Path(os.environ.get("TSR_XILINX", "/tool/Xilinx"))
    if not (plat / "xilinx_u200_gen3x16_xdma_2_202110_1").is_dir():
        pytest.skip("the U200 development platform is not extracted under build/platforms")
    if not (xilinx / "Vitis" / "2024.2" / "settings64.sh").is_file():
        pytest.skip("Vitis 2024.2 not found (TSR_XILINX)")
    env = dict(os.environ, TSR_U200_PLATFORMINFO="1")
    env.pop("PLATFORM_REPO_PATHS", None)
    env.pop("TSR_U200_PLATFORM", None)
    r = subprocess.run([str(REPO / "hw" / "u200" / "build_xclbin.sh"), "k4"], capture_output=True, text=True,
                       env=env, timeout=600)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "PLATFORM_REPO_PATHS unset: using %s" % plat in r.stdout
    banks = dict(re.findall(r"SP Tag:\s+(bank\d)\s+SLR:\s+(SLR\d)", r.stdout))
    assert banks == {"bank0": "SLR0", "bank1": "SLR1", "bank2": "SLR1", "bank3": "SLR2"}, banks


def _require_slow(request):
    """Slow tests are opt-in by tb/fp's rule (tb/fp/test_fp.py slow_requested: an -m expression
    naming `slow`, -k with the test's name, its node id) or TSR_PLATFORM_SLOW=1; a plain `pytest tb`
    / tb/run_all.sh reports them skipped (tb/pytest.ini does not deselect `slow`)."""
    fp = sys.modules.get("tsr_fp_slow_gate") or _load("tsr_fp_slow_gate", REPO / "tb" / "fp" / "test_fp.py")
    if not fp.slow_requested(request.config, request.node.originalname,
                             env={fp.SLOW_ENV: os.environ.get("TSR_PLATFORM_SLOW", "0")}):
        pytest.skip("slow test, opt-in: -m slow, TSR_PLATFORM_SLOW=1, -k %s or its node id"
                    % request.node.originalname)


@pytest.mark.slow
@pytest.mark.traces
def test_adapter_mutants(request, tmp_path):
    _require_slow(request)
    names ="NONE,B_no_base,B_filter_r_off,C_no_cor,C_srst_poll_off,C_base_live"
    r = subprocess.run([sys.executable, str(HERE / "mutate_adapter.py"), str(tmp_path), "--only", names],
                       capture_output=True, text=True, timeout=7200)
    print(r.stdout)
    res = {j["name"]: j for j in json.loads((tmp_path / "results.json").read_text())}
    assert not res["NONE"]["killed"], res["NONE"]
    surv = [n for n in names.split(",")[1:] if not res[n]["killed"]]
    assert not surv, surv

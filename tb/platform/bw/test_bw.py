"""pytest entry of the M0 DMA bandwidth kernel (tb/platform/bw: Verilator C++ bench, no cocotb;
docs/m0-bw.md §5).

  test_bw_lint                   rtl/platform/alveo_xrt/bw/lint.sh: gen_bw.py --check and the lint runs
  test_bw_body                   tsr_bw_kernel (NUM_CH 4, U200 parameters): every test in every memory
                                 regime (bw_bench --test all --regime all)
  test_bw_packaged               the packaged top tsr_bw_k4 (tb/platform/bw/sv/kwrap_tsr_bw_k4.sv), all tests
  test_bw_params[<cfg>]          other parameter sets: 1 / 2 / 3 channels, AXI_SLICES 1 / 3 / 4, CTRL_PIPE
                                 1 / 4, ADDR_BEAT_W 20 / 22 (the range limit inside the bench's memory)
  test_bw_ideal_overhead         ideal memory, 64 outstanding: read / write cycles = beats + 2 x AXI_SLICES
                                 + 3 / + 4 for every burst length, alone and on all channels (measured)
  test_bw_xclbin_no_platform     hw/u200/bw/build_bw_xclbin.sh stops (exit 3) without the platform
  test_bw_tool_mock              sw/tools/tsr_bw --mock (and four injected faults) when a tsr_bw binary
                                 exists ($TSR_BW_BIN or build/*/bin/tsr_bw); skipped otherwise
  test_bw_tool_noop              tb/platform/bw/noop/noop_smoke.sh: tsr_bw --dry-run on XRT's no-op shim
                                 (needs the tsr_bw binary and build/xrt-local); skipped otherwise
  test_bw_mutants                (slow, opt-in: -m slow, TSR_PLATFORM_SLOW=1, -k test_bw_mutants or its
                                 node id) tb/platform/bw/mutate_bw.py: every mutant killed except the
                                 three equivalent ones
Builds: run_bw.py's cache ($TSR_SIM_BUILD_ROOT/bw or build/bw).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("tsr_run_bw", HERE / "run_bw.py")
SEED = os.environ.get("TSR_SEED", "1")


def _bench(binary: Path, args: list, tmp_path: Path) -> list:
    out = tmp_path / "bw.jsonl"
    r = subprocess.run([str(binary), "--seed", SEED, "--out", str(out)] + args, capture_output=True, text=True,
                       timeout=3600)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, r.stdout[-3000:]
    res = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert res and all(j["ok"] and j["failed"] == 0 for j in res), [j for j in res if not j["ok"]]
    return res


def test_bw_lint():
    r = subprocess.run([str(REPO / "rtl" / "platform" / "alveo_xrt" / "bw" / "lint.sh")], capture_output=True,
                       text=True, timeout=900)
    print(r.stdout[-3000:])
    assert r.returncode == 0, r.stdout[-3000:]


def test_bw_body(tmp_path):
    res = _bench(rb.build("body"), ["--test", "all", "--regime", "all"], tmp_path)
    assert {j["test"] for j in res} == {"regmap", "single", "all4", "combos", "ideal", "cfgerr", "errors", "apctrl",
                                        "reset", "big"}
    assert sum(j["runs"] for j in res) > 800


def test_bw_packaged(tmp_path):
    _bench(rb.build("tsr_bw_k4"), ["--test", "all", "--regime", "fast,rand,slow"], tmp_path)


@pytest.mark.parametrize("params", [
    dict(NUM_CH=1, AXI_SLICES=1, CTRL_PIPE=1),
    dict(NUM_CH=3, ADDR_BEAT_W=20, AXI_SLICES=3, CTRL_PIPE=4),
    dict(NUM_CH=2, ADDR_BEAT_W=22, AXI_SLICES=4, CTRL_PIPE=1),
], ids=["c1_s1_p1", "c3_aw20_s3_p4", "c2_aw22_s4_p1"])
def test_bw_params(params, tmp_path):
    _bench(rb.build("body", params), ["--test", "regmap,single,combos,ideal,cfgerr,errors,apctrl,reset",
                                      "--regime", "ideal,rand,ordered", "--runs", "20"], tmp_path)


@pytest.mark.parametrize("slices", [1, 2, 4])
def test_bw_ideal_overhead(slices, tmp_path):
    res = _bench(rb.build("body", dict(AXI_SLICES=slices)), ["--test", "ideal"], tmp_path)
    rate = [x for x in res[0]["ideal"] if x["what"] == "rate"]
    assert rate
    for x in rate:
        if x["rd_cycles"]:
            assert x["rd_cycles"] - x["beats"] == 2 * slices + 3, x
        if x["wr_cycles"]:
            assert x["wr_cycles"] - x["beats"] == 2 * slices + 4, x


def test_bw_xclbin_no_platform(tmp_path):
    # an empty PLATFORM_REPO_PATHS (unset, the script would use the extracted development platform)
    if Path("/opt/xilinx/platforms/xilinx_u200_gen3x16_xdma_2_202110_1").exists():
        pytest.skip("the U200 platform is installed: the script would build")
    env = dict(os.environ, PLATFORM_REPO_PATHS=str(tmp_path))
    env.pop("TSR_U200_PLATFORM", None)
    r = subprocess.run([str(REPO / "hw" / "u200" / "bw" / "build_bw_xclbin.sh")], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 3, r.stderr
    assert "is not installed" in r.stderr


def _tsr_bw() -> Path | None:
    p = os.environ.get("TSR_BW_BIN")
    if p and Path(p).is_file():
        return Path(p)
    c = sorted(glob.glob(str(REPO / "build" / "*" / "bin" / "tsr_bw")))
    return Path(c[0]) if c else None


def test_bw_tool_mock(tmp_path):
    b = _tsr_bw()
    if b is None:
        pytest.skip("no tsr_bw binary (build with -DTSR_WITH_XRT=ON, or set TSR_BW_BIN)")
    args = [str(b), "--mock", "--sizes", "64K,1M,4M", "--bursts", "1,16,64", "--reps", "2", "--pcie",
            "--sweep-outstanding", "--csv", str(tmp_path / "bw.csv")]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:])
    assert r.returncode == 0 and "0 failed their checks" in r.stdout, r.stdout[-3000:]
    for fault in ("csum", "write", "status", "bytes"):
        env = dict(os.environ, TSR_BW_MOCK_FAULT=fault)
        r = subprocess.run(args[:6] + ["--reps", "1"], capture_output=True, text=True, timeout=600, env=env)
        assert r.returncode == 1 and "FAIL" in r.stdout, (fault, r.stdout[-2000:])


def test_bw_tool_noop(tmp_path):
    b = _tsr_bw()
    xrt = REPO / "build" / "xrt-local" / "opt" / "xilinx" / "xrt"
    if b is None or not (xrt / "lib" / "libxrt_noop.so").exists() or shutil.which("g++") is None:
        pytest.skip("needs a tsr_bw binary and the unpacked XRT (docs/xrt-host.md §7)")
    r = subprocess.run([str(HERE / "noop" / "noop_smoke.sh"), str(b), str(xrt), str(tmp_path)], capture_output=True,
                       text=True, timeout=600)
    print(r.stdout)
    assert r.returncode == 0, r.stdout


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
def test_bw_mutants(request, tmp_path):
    _require_slow(request)
    mb =_load("tsr_mutate_bw", HERE / "mutate_bw.py")
    r = subprocess.run([sys.executable, str(HERE / "mutate_bw.py"), "--work", str(tmp_path), "--procs", "2"],
                       capture_output=True, text=True, timeout=7200)
    print(r.stdout[-4000:])
    assert r.returncode == 0
    res = json.loads((tmp_path / "results.json").read_text())
    survivors = {m[0] for m in mb.MUTANTS if not res[m[0]]["killed"]}
    # equivalent (docs/m0-bw.md §5.3): W before AW is legal AXI; the snapshot matters only inside a
    # window the AXI4-Lite slave cannot reach at CTRL_PIPE 2; the checksum drains before any read
    assert survivors <= {"E_w_no_ahead", "C_snapshot_live", "E_cs_nodrain"}, survivors

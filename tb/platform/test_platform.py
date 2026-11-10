"""pytest entry of the Alveo XRT platform adapter (tb/platform; no cocotb).

  test_platform_lint              rtl/platform/alveo_xrt/lint.sh: gen_tops.py --check and the lint
                                  runs of the adapter modules, kernel configurations and tops
  test_build_xclbin_args          hw/u200/build_xclbin.sh: an unknown kernel or target stops (exit 2) before any
                                  tool call
  test_build_xclbin_no_platform   TSR_U200_PLATFORMINFO=1: the script stops (exit 3) without the
                                  platform
  test_build_xclbin_platform      TSR_U200_PLATFORMINFO=1 with PLATFORM_REPO_PATHS unset: the script
                                  finds the extracted development platform (build/platforms/...) and
                                  its platforminfo shows the DDR bank SLRs of hw/u200/k1.cfg / k4.cfg
                                  (skipped without the extracted platform or Vitis 2024.2)
The adapter bench is tb/platform/test_adapter.py (tsr_xrt_ctrl against the mock adapter model: register
map, MEM_EN, ADAPTER_STATUS, window; tsr_xrt_kernel: ap_clk_2, MEM_EN; the read ports of the kernel body with a stub core).
tb/platform/bw (the bandwidth kernel) is independent of the core.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD_XCLBIN = REPO / "hw" / "u200" / "build_xclbin.sh"


def test_platform_lint():
    r = subprocess.run([str(REPO / "rtl" / "platform" / "alveo_xrt" / "lint.sh")], capture_output=True, text=True,
                       timeout=1800)
    print(r.stdout[-3000:])
    assert r.returncode == 0, r.stdout[-3000:]


@pytest.mark.parametrize("args", [[], ["k2"], ["k1", "sim"]])
def test_build_xclbin_args(args):
    env = dict(os.environ)
    env.pop("TSR_U200_PLATFORMINFO", None)
    r = subprocess.run([str(BUILD_XCLBIN), *args], capture_output=True, text=True, env=env, timeout=60)
    print(r.stderr)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "usage:" in r.stderr or "target must be" in r.stderr
    assert r.stdout == ""


def test_build_xclbin_no_platform(tmp_path):
    env = dict(os.environ, PLATFORM_REPO_PATHS=str(tmp_path), TSR_U200_PLATFORMINFO="1")
    env.pop("TSR_U200_PLATFORM", None)
    r = subprocess.run([str(BUILD_XCLBIN), "k1"], capture_output=True, text=True, env=env, timeout=60)
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
    r = subprocess.run([str(BUILD_XCLBIN), "k4"], capture_output=True, text=True, env=env, timeout=600)
    print(r.stdout[-3000:], r.stderr[-2000:])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "PLATFORM_REPO_PATHS unset: using %s" % plat in r.stdout
    banks = dict(re.findall(r"SP Tag:\s+(bank\d)\s+SLR:\s+(SLR\d)", r.stdout))
    assert banks == {"bank0": "SLR0", "bank1": "SLR1", "bank2": "SLR1", "bank3": "SLR2"}, banks

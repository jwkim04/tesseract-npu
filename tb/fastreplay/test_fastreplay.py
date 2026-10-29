"""pytest entry of tb/fastreplay (C++, no cocotb).

  test_memmodel_selftest          memmodel.h against selftest.cpp (legal traffic, every protocol
                                  assertion P1..P6 fires on its violation, ordering, tracking)

memmodel.h is also the memory model of the M0 bandwidth bench (tb/platform/bw). X0 (docs/v2 EX §12,
SW §10.3) deleted the v1 replay of cmd.bin traces on tsr_core (run.py, fastreplay.cpp,
fr_common.h, mutate.py) and its tests test_fastreplay_m2, test_fastreplay_impl,
test_fastreplay_detects_corrupt and test_fastreplay_detects_mutants; the bench is rewritten as the
C++ tile-only replay of v2 `push.bin` traces, then the full-system bench with Ibex and RVFI lockstep.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_memmodel_selftest(tmp_path):
    exe = tmp_path / "selftest"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-Wall", "-Wextra", "-I", str(HERE),
                    str(HERE / "selftest.cpp"), "-o", str(exe)], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    assert r.returncode == 0, r.stdout
    assert "0 check(s) failed" in r.stdout

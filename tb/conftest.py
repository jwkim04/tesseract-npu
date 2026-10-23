"""pytest configuration for the Tesseract RTL testbenches (tb/).

Layout: tb/common/ is the shared infrastructure (import as `from common import ...`); each block
has its own directory tb/<block>/ with a uniquely named test_<block>.py that holds both the cocotb
tests and the pytest function(s) calling common.runner.run(). Do not add __init__.py to the block
directories (pytest then imports test files by their unique basename).

Options:
  --tsr-seed N      seed for every randomised testbench (exported as TSR_SEED; default 1)
  --waves           dump waves (exported as WAVES=1; VCD in the build directory)
  --tsr-traces DIR  trace root for trace-replay tests (exported as TSR_TRACES; default traces/m2)
Environment: TSR_SIM_BUILD_ROOT (build directory root, default <repo>/build/cocotb),
TSR_BUILD_JOBS (make -j, default 4), TSR_REF_LIB (libtsr_ref.so path).
"""
import os
import sys
from pathlib import Path

TB = Path(__file__).resolve().parent
if str(TB) not in sys.path:
    sys.path.insert(0, str(TB))


def pytest_addoption(parser):
    parser.addoption("--tsr-seed", action="store", default=None, help="testbench random seed")
    parser.addoption("--waves", action="store_true", default=False, help="dump waveforms")
    parser.addoption("--tsr-traces", action="store", default=None, help="trace root directory")


def pytest_configure(config):
    seed = config.getoption("--tsr-seed")
    if seed is not None:
        os.environ["TSR_SEED"] = str(int(seed, 0))
    if config.getoption("--waves"):
        os.environ["WAVES"] = "1"
    traces = config.getoption("--tsr-traces")
    if traces is not None:
        os.environ["TSR_TRACES"] = str(Path(traces).resolve())
    os.environ.setdefault("TSR_TRACES", str(TB.parent / "traces" / "m2"))

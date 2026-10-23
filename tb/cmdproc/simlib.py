"""Compatibility shim: the simulator ctypes wrapper lives in tb/common/simlib.py (moved there for
M3 so tb/cmdproc, tb/core and the M3 testbenches share one copy). `import simlib` keeps working."""
from common.simlib import (LIB_NAME, REF_A, SIM_A, Sim, SimResult, TsrCapture,  # noqa: F401
                           TsrConfig, TsrRegion, lib)

"""Directed checks for the RV island links and the datapath's FIFO drain gate."""

import os
import subprocess
from pathlib import Path

from common.runner import REPO, build_root


def test_rv_local_transport():
    verilator = os.environ.get("VERILATOR", "verilator")
    jobs = os.environ.get("TSR_BUILD_JOBS", "8")
    seed = os.environ.get("TSR_SEED", "1")
    sv = Path(__file__).with_name("sv")
    transport_sources = ["rtl/core/tsr_tables_pkg.sv", "rtl/core/tsr_rv_local_transport.sv"]
    drain_sources = ["-Wno-PINMISSING", "-F", "rtl/filelist.f"]
    # The links run twice: without and with the SPM link end's XSLR_REG request stage.
    for top, sources, tag, extra in (
        ("tb_rv_local_transport", transport_sources, "", []),
        ("tb_rv_local_transport", transport_sources, "_xslr", ["-GXSLR_REG=1"]),
        ("tb_rv_local_idle", drain_sources, "", []),
    ):
        obj = (build_root() / (top + tag)).resolve()
        obj.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [verilator, "--binary", "--timing", "--timescale", "1ns/1ps", "--assert",
             "-j", jobs, "--top-module", top, "--Mdir", str(obj), *extra, *sources,
             str(sv / (top + ".sv"))],
            cwd=REPO,
            check=True,
        )
        subprocess.run([str(obj / ("V" + top)), f"+verilator+seed+{seed}"], cwd=obj, check=True)

"""tb/tech: the Xilinx technology wrappers (rtl/tech/xilinx, rtl/filelist_xilinx.f) against the generic
ones (rtl/tech/generic), which every other testbench verifies against the contracts.

  test_tech_mac_equiv      tsr_mac_lane: xilinx vs generic cycle by cycle (tb/tech/cpp/mac_equiv.cpp
                           on tb/tech/hdl/tsr_tb_tech_mac.sv: N 32 x T 1..8 x LAT 1..11 plus N 5 / 1 /
                           2 / 3; corner sweeps of {-128, -127, -1, 0, 1, 127}^3 over every packed DSP
                           slot, uniform extreme sums, random beats with ce / in_valid / reset), both
                           models also against the reference sum and the valid pipeline model
  test_tech_macpv_equiv    tsr_mac_lane's PV mode (ATTPV, AT §8.2): xilinx vs generic vs the
                           accumulate reference (tb/tech/cpp/macpv_equiv.cpp on tsr_tb_tech_macpv.sv)
  test_tech_sram_equiv     tsr_sram_1r1w: xilinx vs generic cycle by cycle (tb/tech/cpp/sram_equiv.cpp
                           on tb/tech/hdl/tsr_tb_tech_sram.sv: the URAM / BRAM / LUTRAM sizes of the
                           core incl. the SPM's 64-bit column slices, RD_LAT 1..4; collisions,
                           read-after-write, byte enables), both models also against the contract
                           model; every cycle for BRAM / LUTRAM sizes, every cycle with a read result
                           for URAM sizes (their output stages hold between reads,
                           rtl/tech/xilinx/tsr_sram_1r1w.sv)
  test_tech_sram_contract  the cocotb contract testbench of tb/spm/test_sram.py ("the contract every
                           target variant must meet") on the xilinx tsr_sram_1r1w: its four
                           configurations plus URAM (4096 x 512, 16 K x 512 RD_LAT 4, the SPM slices
                           16 K x 64 RD_LAT 2 and 4 K x 64 RD_LAT 1) and LUTRAM sizes
  test_tech_spm[...]       tb/spm's cocotb tests (tb/spm/test_spm.py, exact SPM contract model) on
                           tsr_spm built from the xilinx tsr_sram_1r1w: 4 MiB at SPM_RD_LAT 4 (U200)
                           and 2 (default), 256 KiB at 4 (one bank of URAM slices), 1000 lines at 4
  test_tech_lint           rtl/tech/xilinx/lint.sh (Verilator -Wall of the xilinx target)
  test_tech_netlist[...]   opt-in, needs Vivado: the Vivado-synthesised netlist of a xilinx wrapper
                           (DSP48E2 / URAM288 / RAMB36E2 / LUTRAM unisim models) against the generic RTL
                           in xsim (tb/tech/viv/net_sim.sh); TSR_TECH_VIVADO=1 or -k netlist
Both equivalence harnesses link two Verilator models of one toplevel (--prefix Vgen / Vxil) into one
executable. Environment: TSR_TECH_CYCLES (random cycles of the equivalence runs, default 200000 MAC /
300000 SRAM), TSR_TECH_NETSIM_CYCLES (default 20000), TSR_VIVADO_LOCK (flock file wrapped around
every Vivado tool call), TSR_BUILD_JOBS, TSR_SIM_BUILD_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from common.runner import REPO, build_root, run, seed_from_env

TECH = REPO / "tb" / "tech"
TOPS = {
    "mac": ("tsr_tb_tech_mac", "tsr_mac_lane.sv", TECH / "cpp" / "mac_equiv.cpp", 200000),
    # the PV (ATTPV accumulate) mode of tsr_mac_lane, AT §8.2 / §13.5
    "macpv": ("tsr_tb_tech_macpv", "tsr_mac_lane.sv", TECH / "cpp" / "macpv_equiv.cpp", 100000),
    "sram": ("tsr_tb_tech_sram", "tsr_sram_1r1w.sv", TECH / "cpp" / "sram_equiv.cpp", 300000),
}
VARIANTS = {"gen": "generic", "xil": "xilinx"}


def _sh(cmd, cwd=None, log=None):
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(p.stdout + p.stderr)
    assert p.returncode == 0, "command failed: %s\n%s\n%s" % (" ".join(map(str, cmd)), p.stdout[-3000:],
                                                              p.stderr[-3000:])
    return p


def build_equiv(kind: str) -> Path:
    """Two Verilator models of the kind's toplevel (generic / xilinx tech file) linked with its harness."""
    top, tech, cpp, _ = TOPS[kind]
    vl = os.environ.get("VERILATOR", "verilator")
    jobs = os.environ.get("TSR_BUILD_JOBS", "4")
    hdl = TECH / "hdl" / (top + ".sv")
    srcs = {v: [REPO / "rtl" / "tech" / d / tech, hdl] for v, d in VARIANTS.items()}
    vflags = ["-O3", "--x-assign", "unique", "--x-initial", "unique"]
    h = hashlib.sha256()
    h.update(subprocess.run([vl, "--version"], capture_output=True, text=True).stdout.encode())
    for f in [cpp] + srcs["gen"] + srcs["xil"]:
        h.update(str(f).encode())
        h.update(Path(f).read_bytes())
    h.update(repr(vflags).encode())
    stamp = h.hexdigest()
    bdir = build_root() / ("tech_%s_%s" % (kind, stamp[:12]))
    exe = bdir / (kind + "_equiv")
    if exe.is_file() and (bdir / "stamp").is_file() and (bdir / "stamp").read_text() == stamp:
        return exe
    bdir.mkdir(parents=True, exist_ok=True)
    root = subprocess.run([vl, "--getenv", "VERILATOR_ROOT"], capture_output=True, text=True).stdout.strip()
    for v in VARIANTS:
        mdir = bdir / v
        _sh([vl, "--cc"] + vflags + ["--prefix", "V" + v, "--top-module", top, "--Mdir", mdir] + srcs[v],
            log=bdir / ("verilate_%s.log" % v))
        targets = ["libV%s.a" % v] + (["libverilated.a"] if v == "gen" else [])
        _sh(["make", "-j", jobs, "-C", mdir, "-f", "V%s.mk" % v] + targets, log=bdir / ("make_%s.log" % v))
    cxx = os.environ.get("CXX", "g++")
    _sh([cxx, "-O2", "-std=c++17", "-I%s/include" % root, "-I%s/include/vltstd" % root, "-I", bdir / "gen",
         "-I", bdir / "xil", cpp, bdir / "gen" / "libVgen.a", bdir / "xil" / "libVxil.a",
         bdir / "gen" / "libverilated.a", "-pthread", "-latomic", "-o", exe], log=bdir / "link.log")
    (bdir / "stamp").write_text(stamp)
    return exe


def _run_equiv(kind: str):
    exe = build_equiv(kind)
    cycles = int(os.environ.get("TSR_TECH_CYCLES", TOPS[kind][3]))
    p = subprocess.run([str(exe), str(cycles), str(seed_from_env())], capture_output=True, text=True)
    print(p.stdout)
    assert p.returncode == 0, "%s equivalence failed:\n%s%s" % (kind, p.stdout[-4000:], p.stderr[-2000:])
    m = re.search(r"mismatches (\d+)", p.stdout)
    assert m and int(m.group(1)) == 0, p.stdout[-4000:]


def test_tech_mac_equiv():
    """tsr_mac_lane xilinx == generic == reference, every configuration, every cycle."""
    _run_equiv("mac")


def test_tech_macpv_equiv():
    """tsr_mac_lane PV mode (ATTPV, AT §8.2): xilinx == generic == the accumulate reference, every
    cycle, every attention configuration (N 32 x T 1 / 2 / 3 / 8 x LAT 3..11), with the MVM mode in between."""
    _run_equiv("macpv")


def test_tech_sram_equiv():
    """tsr_sram_1r1w xilinx == generic == contract model, every configuration, every cycle."""
    _run_equiv("sram")


SRAM_CONTRACT = {
    "d100_w512_lat1": dict(DEPTH=100, WIDTH=512, RD_LAT=1),        # tb/spm/test_sram.py CONFIGS
    "d1024_w512_lat3": dict(DEPTH=1024, WIDTH=512, RD_LAT=3),
    "d1_w72_lat2": dict(DEPTH=1, WIDTH=72, RD_LAT=2),
    "d37_w8_lat4": dict(DEPTH=37, WIDTH=8, RD_LAT=4),
    "d4096_w512_lat2": dict(DEPTH=4096, WIDTH=512, RD_LAT=2),      # URAM
    "d16384_w512_lat4": dict(DEPTH=16384, WIDTH=512, RD_LAT=4),    # URAM cascade, 3 output stages
    "d16384_w64_lat2": dict(DEPTH=16384, WIDTH=64, RD_LAT=2),      # SPM column slice at U200 (RAM_LAT 2)
    "d4096_w64_lat1": dict(DEPTH=4096, WIDTH=64, RD_LAT=1),        # 256 KiB SPM slice, one URAM
    "d32_w512_lat2": dict(DEPTH=32, WIDTH=512, RD_LAT=2),          # LUTRAM
}


@pytest.mark.parametrize("cfg", list(SRAM_CONTRACT))
def test_tech_sram_contract(cfg):
    """tb/spm's contract tests (sram_sweep, sram_random) on rtl/tech/xilinx/tsr_sram_1r1w.sv."""
    spm = str(REPO / "tb" / "spm")
    if spm not in sys.path:      # the simulator's Python imports test_sram from sys.path
        sys.path.insert(0, spm)
    p = SRAM_CONTRACT[cfg]
    run("tsr_sram_1r1w", "test_sram", parameters=p, sources=[REPO / "rtl" / "tech" / "xilinx" / "tsr_sram_1r1w.sv"],
        name="xsram_" + cfg, testcase=["sram_sweep", "sram_random"], extra_env={"TSR_SRAM_CFG": json.dumps(p)})


SPM_XIL = {
    # name -> (tsr_spm parameter overrides, cocotb tests of tb/spm/test_spm.py, random cycles)
    "full_lat4": (dict(RD_LAT=4), ["spm_reset_state", "spm_directed", "spm_random"], 4000),
    "full": (dict(), ["spm_reset_state", "spm_directed", "spm_random"], 4000),
    "k256_lat4": (dict(SPM_BYTES=262144, RD_LAT=4),
                  ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random"], 8000),
    "odd_lat4": (dict(SPM_BYTES=64 * 1000, BANK_LINES=256, RD_LAT=4),
                 ["spm_reset_state", "spm_directed", "spm_sweep", "spm_random", "spm_reset_midflight"],
                 12000),
}


@pytest.mark.parametrize("cfg", list(SPM_XIL))
def test_tech_spm(cfg):
    """tb/spm's tsr_spm tests on the xilinx SRAM wrapper (URAM column slices whose output stages
    hold between reads, BRAM for the small configuration)."""
    spm = str(REPO / "tb" / "spm")
    if spm not in sys.path:      # the simulator's Python imports test_spm from sys.path
        sys.path.insert(0, spm)
    import test_spm                # noqa: E402 (tb/spm on sys.path)
    over, tests, cycles = SPM_XIL[cfg]
    srcs = [REPO / "rtl" / "core" / "tsr_tables_pkg.sv", REPO / "rtl" / "core" / "tsr_pkg.sv",
            REPO / "rtl" / "tech" / "xilinx" / "tsr_sram_1r1w.sv",
            REPO / "rtl" / "core" / "tsr_spm.sv"]
    run("tsr_spm", "test_spm", parameters=over, sources=srcs, name="xspm_" + cfg, testcase=tests,
        extra_env={"TSR_SPM_CFG": json.dumps(test_spm.full_config(over)), "TSR_SPM_CYCLES": str(cycles)})


def test_tech_lint():
    """rtl/tech/xilinx/lint.sh: the xilinx target (filelist, wrappers, whole core) lint clean."""
    p = subprocess.run([str(REPO / "rtl" / "tech" / "xilinx" / "lint.sh")], capture_output=True, text=True)
    print(p.stdout[-6000:])
    assert p.returncode == 0, p.stdout[-6000:] + p.stderr[-2000:]


# ---------------------------------------------------------------------------------------------
# opt-in: post-synthesis netlists (Vivado + xsim)
# ---------------------------------------------------------------------------------------------
VIVADO = Path(os.environ.get("TSR_VIVADO", "/tool/Xilinx/Vivado/2024.2/bin/vivado"))
NETSIM = {
    # MAC lanes: T 1 (single products), 2 / 3 / 8 (packed pairs, odd T with a single slot), the
    # U200 latency 5 (AD M P | 2 | 3), the generic default 3, 6 (input registers), 1 (none), N 5
    "mac_t1_l5": ("mac", dict(N=32, T=1, LAT=5)),
    "mac_t2_l5": ("mac", dict(N=32, T=2, LAT=5)),
    "mac_t3_l5": ("mac", dict(N=32, T=3, LAT=5)),
    "mac_t8_l5": ("mac", dict(N=32, T=8, LAT=5)),
    "mac_t8_l3": ("mac", dict(N=32, T=8, LAT=3)),
    "mac_t7_l6": ("mac", dict(N=32, T=7, LAT=6)),
    "mac_t4_l1": ("mac", dict(N=32, T=4, LAT=1)),
    "mac_n5_t3_l4": ("mac", dict(N=5, T=3, LAT=4)),
    # SRAMs: URAM cascades (the SPM's 64-bit column slices 16 K x 64 / 4 K x 64, 512-bit cascades),
    # BRAM (weight FIFO, QACT line buffer, DBS, deep FIFO with fabric stages), LUTRAM (32 deep),
    # narrow odd depth
    "sram_16k_w64_l2": ("sram", dict(DEPTH=16384, WIDTH=64, RD_LAT=2)),
    "sram_4k_w64_l1": ("sram", dict(DEPTH=4096, WIDTH=64, RD_LAT=1)),
    "sram_16k_l2": ("sram", dict(DEPTH=16384, WIDTH=512, RD_LAT=2)),
    "sram_16k_l1": ("sram", dict(DEPTH=16384, WIDTH=512, RD_LAT=1)),
    "sram_4k_l3": ("sram", dict(DEPTH=4096, WIDTH=512, RD_LAT=3)),
    "sram_512_l2": ("sram", dict(DEPTH=512, WIDTH=512, RD_LAT=2)),
    "sram_64_l1": ("sram", dict(DEPTH=64, WIDTH=512, RD_LAT=1)),
    "sram_72_l2": ("sram", dict(DEPTH=72, WIDTH=512, RD_LAT=2)),
    "sram_1024_l4": ("sram", dict(DEPTH=1024, WIDTH=512, RD_LAT=4)),
    "sram_32_l2": ("sram", dict(DEPTH=32, WIDTH=512, RD_LAT=2)),
    "sram_37_l4": ("sram", dict(DEPTH=37, WIDTH=8, RD_LAT=4)),
}


@pytest.mark.parametrize("cfg", list(NETSIM))
def test_tech_netlist(cfg, request):
    """Opt-in (needs Vivado, ~1-5 min each): synthesised netlist == generic RTL in xsim."""
    selected = os.environ.get("TSR_TECH_VIVADO") == "1" or \
        "netlist" in (request.config.getoption("keyword") or "") or \
        any("test_tech_netlist" in a for a in request.config.args)
    if not selected:
        pytest.skip("opt-in Vivado run: set TSR_TECH_VIVADO=1 or select it with -k netlist")
    if not VIVADO.exists():
        pytest.skip("Vivado not found (%s)" % VIVADO)
    kind, gens = NETSIM[cfg]
    env = dict(os.environ)
    env.setdefault("NETSIM_CYCLES", os.environ.get("TSR_TECH_NETSIM_CYCLES", "20000"))
    out = build_root() / ("tech_netsim_" + cfg)      # Vivado / xsim work directory
    p = subprocess.run([str(TECH / "viv" / "net_sim.sh"), kind, str(out)] +
                       ["%s=%s" % kv for kv in gens.items()], capture_output=True, text=True, env=env,
                       timeout=7200)
    print(p.stdout)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
    assert re.search(r"NETSIM .* mismatches 0$", p.stdout, re.M), p.stdout[-4000:]

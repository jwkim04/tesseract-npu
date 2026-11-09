"""Opt-in (needs Vivado, about 3 min per form): the synthesised netlist of the ATTPV drain's lane select against the
selection itself (tb/at/viv/lsel_net.sh).

A card run (the index-expression rule of docs/platform-u200.md §8.8) showed every ATTPV result with lanes 32 to 63 equal to lanes 0 to
31. Verilator and the RTL benches were right: Vivado 2024.2 synthesised `automatic int L = 8 * int'(dr_l_q) + k` used as
`cb_a[t][6'(L)]` without the top bit of dr_l_q. The core reads through tsr_mvm_pv_lsel (a lane concatenation). The
test simulates the netlist of that module (it must match every case) and, as a control, the netlist of the old integer
form (it must not: if it ever matches, this Vivado no longer has the fault and the control can go).

Select it with TSR_AT_VIVADO=1 or -k netlist.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VIVADO = Path(os.environ.get("TSR_VIVADO", "/tool/Xilinx/Vivado/2024.2/bin/vivado"))


def _selected(request):
    return os.environ.get("TSR_AT_VIVADO") == "1" or "netlist" in (request.config.getoption("keyword") or "") or \
        any("test_at_pv_netlist" in a for a in request.config.args)


def _run(form, request):
    if not _selected(request):
        pytest.skip("opt-in Vivado run: set TSR_AT_VIVADO=1 or select it with -k netlist")
    if not VIVADO.exists():
        pytest.skip("Vivado not found (%s)" % VIVADO)
    from common.runner import build_root
    env = dict(os.environ)
    env.setdefault("TSR_VIVADO_WRAP", str(REPO / "hw" / "u200" / "vivado_slot.sh"))
    p = subprocess.run([str(HERE / "viv" / "lsel_net.sh"), form, str(build_root() / ("at_lsel_net_" + form))],
                       capture_output=True, text=True, env=env, timeout=3600)
    print(p.stdout)
    m = re.search(r"NETSIM lsel (\w+) cases (\d+) mismatches (\d+)$", p.stdout, re.M)
    assert m, p.stdout[-3000:] + p.stderr[-2000:]
    return p, int(m.group(2)), int(m.group(3))


def test_at_pv_lsel_netlist_rtl(request):
    """tsr_mvm_pv_lsel: every drain step selects lane 8 l + k of slot t and the flag of block lane / 32."""
    p, cases, bad = _run("rtl", request)
    assert cases > 0 and bad == 0 and p.returncode == 0, p.stdout[-3000:]


def test_at_pv_lsel_netlist_legacy_control(request):
    """The integer form the RTL had: the bench must see Vivado's fault (mismatches on steps with l >= 4)."""
    _, cases, bad = _run("legacy", request)
    assert cases > 0 and bad > 0, "no mismatch: this Vivado synthesises the integer form correctly; drop the control"

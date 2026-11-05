"""The Pblock hook of the reduced first card image (hw/u200/k4_major_blocks.tcl) against a mocked Vivado.

Only the hierarchy lookups are mocked: get_cells returns the path asked for, get_pblocks the pblock name, and
add_cells_to_pblock prints the assignment. No Xilinx tool, checkpoint or placement is involved. The test checks which
whole block goes into which SLR pblock, that nothing below a block (a LUT, register, stage or site) is named, that
the control plane follows TSR_K4_CTRL, and that a missing block stops the hook.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "hw/u200/k4_major_blocks.tcl"

MOCK = r"""
proc current_instance {args} {}
proc get_cells {args} {
    if {[lsearch -exact $args -filter] >= 0} {return top/tsr_k4/inst}
    set p [lindex $args end]
    if {[info exists ::env(MOCK_MISSING)] && [string first $::env(MOCK_MISSING) $p] >= 0} {return {}}
    return $p
}
proc get_property {key obj} {
    if {$key eq "IS_PRIMITIVE"} {return 0}
    return $obj
}
proc get_pblocks {args} {return [lindex $args end]}
proc add_cells_to_pblock {pb obj} {puts [join [list assignment [string index $pb end] $obj] "\t"]}
if {[catch {source $::env(MOCK_HOOK)} err]} {
    puts stderr $err
    exit 1
}
"""


def run_hook(ctrl=None, core=None, missing=None, body=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TSR_K4", "MOCK_"))}
    env["MOCK_HOOK"] = str(HOOK)
    if ctrl is not None:
        env["TSR_K4_CTRL"] = ctrl
    if core is not None:
        env["TSR_K4_FP_CORE"] = core
    if body is not None:
        env["TSR_K4_FP_BODY"] = body
    if missing is not None:
        env["MOCK_MISSING"] = missing
    r = subprocess.run(["tclsh"], input=MOCK, text=True, capture_output=True, env=env, timeout=30)
    got = {}
    for line in r.stdout.splitlines():
        if line.startswith("assignment\t"):
            _, slr, path = line.split("\t")
            got[path] = int(slr)
    return r, got


def expected(prefix, kernel, ctrl_slr):
    core = prefix + "u_core/" if kernel else prefix
    dp = core + "u_dp/"
    mvm = dp + "g_mvm.u_mvm/"
    want = {}
    for cl, slr in ((0, 0), (1, 2), (2, 0), (3, 2)):
        want[f"{mvm}g_cl[{cl}].u_cl"] = slr
        want[f"{mvm}u_gather/g_asm[{cl}].u_asm"] = slr
    if kernel:
        want[f"{prefix}g_ch[0].u_br"] = 0
        want[f"{prefix}g_ch[3].u_br"] = 2
    want[core + "u_ctrl"] = 0
    want[core + "g_db.u_dbg_rv"] = 0
    for e in ("u_spm_rx", "u_issue_tx", "u_lq_tx"):
        want[core + "g_rv_local." + e] = 0
    want[core + "g_rv_local.u_spm_tx"] = 1
    for e in ("u_issue_rx", "u_lq_rx"):
        want[core + "g_rv_local." + e] = ctrl_slr
    for b in ("u_spm", "u_bridge", "u_dma", "u_qact"):
        want[dp + b] = 1
    for c in range(4):
        want[f"{dp}g_ch[{c}].u_arb"] = 1
    for b in ("u_issue", "u_csr", "g_db.u_dbg"):
        want[dp + b] = ctrl_slr
    return want


@pytest.mark.parametrize("ctrl,slr", [(None, 0), ("slr1", 1), ("slr0", 0)])
def test_kernel_map(ctrl, slr):
    r, got = run_hook(ctrl=ctrl)
    assert r.returncode == 0, r.stderr
    assert got == expected("top/tsr_k4/inst/u_kernel/", True, slr)


def test_core_map():
    r, got = run_hook(core="")
    assert r.returncode == 0, r.stderr
    assert got == expected("", False, 0)


def test_free_control_plane_leaves_issue_csr_db_unassigned():
    r, got = run_hook(ctrl="free")
    assert r.returncode == 0, r.stderr
    assert not [p for p in got if p.endswith(("/u_issue", "/u_csr", "/g_db.u_dbg"))]
    assert got["top/tsr_k4/inst/u_kernel/u_core/g_rv_local.u_issue_rx"] == 1


def test_only_whole_blocks_are_named():
    r, got = run_hook()
    assert r.returncode == 0, r.stderr
    leaf = re.compile(r"/(\w+_reg|\w+_q|LUT\d|FD\w+|SLICE_\w+)(\[\d+\])*$")
    assert not [p for p in got if leaf.search(p)]
    # no vector unit, attention or trace block is named: the elaboration does not have them
    assert not [p for p in got if any(x in p for x in ("g_vu", "u_vu", "g_att", "kvst", "trw", "g_local_att"))]


@pytest.mark.parametrize("missing", ["g_cl[2].u_cl", "u_qact", "g_rv_local.u_lq_rx", "g_ch[3].u_br", "u_gather/g_asm[1]"])
def test_missing_block_stops_the_hook(missing):
    r, got = run_hook(missing=missing)
    assert r.returncode != 0
    assert "K4 major blocks: missing hierarchy" in r.stderr


def test_bad_control_plane_value():
    r, _ = run_hook(ctrl="slr2")
    assert r.returncode != 0
    assert "TSR_K4_CTRL must be slr0, slr1 or free" in r.stderr

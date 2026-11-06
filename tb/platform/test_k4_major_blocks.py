"""The Pblock hook of the reduced first card image (hw/u200/k4_major_blocks.tcl) against a mocked Vivado.

Only the Pblock and hierarchy lookups are mocked: get_cells returns the path asked for, get_pblocks the platform
pblocks and the ones the hook created, create_pblock / resize_pblock / set_property record the control Pblock, get_sites
returns a counted list, and add_cells_to_pblock prints the assignment. No Xilinx tool, checkpoint or placement is
involved. The test checks which whole block goes into which pblock, that nothing below a block (a LUT, register, stage
or site) is named, that the control plane follows TSR_K4_CTRL, that a missing block stops the hook, and the control
Pblock itself: a child of the SLR0 platform pblock, at the SLR1 boundary, off the DDR0 controller, large enough.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "hw/u200/k4_major_blocks.tcl"
CTRL = "pblock_tsr_ctrl_plane"

MOCK = r"""
proc current_instance {args} {}
set ::pblocks {pblock_dynamic_SLR0 pblock_dynamic_SLR1 pblock_dynamic_SLR2}
array set ::parent {}
array set ::ranges {}
proc get_cells {args} {
    if {[lsearch -exact $args -filter] >= 0} {return top/tsr_k4/inst}
    set p [lindex $args end]
    if {[info exists ::env(MOCK_MISSING)] && [string first $::env(MOCK_MISSING) $p] >= 0} {return {}}
    return $p
}
proc get_pblocks {args} {
    set n [lindex $args end]
    if {[info exists ::env(MOCK_NO_PBLOCK)] && $n eq $::env(MOCK_NO_PBLOCK)} {return {}}
    if {[lsearch -exact $::pblocks $n] >= 0} {return $n}
    return {}
}
proc create_pblock {n} {lappend ::pblocks $n; set ::ranges($n) {}; return $n}
proc resize_pblock {pb args} {
    if {[lindex $args 0] ne "-add"} {error "mock: only resize_pblock -add"}
    lappend ::ranges($pb) [lindex $args 1]
}
proc set_property {key value obj} {
    if {$key ne "PARENT"} {error "mock: unexpected set_property $key"}
    if {[info exists ::env(MOCK_PARENT)]} {set value $::env(MOCK_PARENT)}
    set ::parent($obj) $value
}
proc get_property {key obj} {
    if {$key eq "IS_PRIMITIVE"} {return 0}
    if {$key eq "PARENT"} {return [expr {[info exists ::parent($obj)] ? $::parent($obj) : {}}]}
    return $obj
}
proc get_sites {args} {
    set filter [lindex $args end]
    set n 0
    if {[string match {*SLICE_**} $filter]} {set n [expr {[info exists ::env(MOCK_SLICES)] ? $::env(MOCK_SLICES) : 14000}]}
    if {[string match {*RAMB36_**} $filter]} {set n [expr {[info exists ::env(MOCK_RAMB36)] ? $::env(MOCK_RAMB36) : 250}]}
    set out {}
    for {set i 0} {$i < $n} {incr i} {lappend out site$i}
    return $out
}
proc add_cells_to_pblock {pb obj} {puts [join [list assignment $pb $obj] "\t"]}
if {[catch {source $::env(MOCK_HOOK)} err]} {
    puts stderr $err
    exit 1
}
foreach n $::pblocks {
    if {[info exists ::parent($n)]} {puts [join [list pblock $n $::parent($n) $::ranges($n)] "\t"]}
}
"""


def run_hook(ctrl=None, core=None, missing=None, body=None, **mock):
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
    env.update({"MOCK_" + k.upper(): str(v) for k, v in mock.items()})
    r = subprocess.run(["tclsh"], input=MOCK, text=True, capture_output=True, env=env, timeout=30)
    got = {}
    children = {}
    for line in r.stdout.splitlines():
        if line.startswith("assignment\t"):
            _, pblock, path = line.split("\t")
            got[path] = pblock
        elif line.startswith("pblock\t"):
            _, name, parent, *ranges = line.split("\t")
            children[name] = (parent, ranges[0].split() if ranges else [])
    return r, got, children


def slr(n):
    return f"pblock_dynamic_SLR{n}"


def expected(prefix, kernel, ctrl_where):
    core = prefix + "u_core/" if kernel else prefix
    dp = core + "u_dp/"
    mvm = dp + "g_mvm.u_mvm/"
    where = CTRL if ctrl_where == "ctrl" else slr(ctrl_where)
    want = {}
    for cl, n in ((0, 0), (1, 2), (2, 0), (3, 2)):
        want[f"{mvm}g_cl[{cl}].u_cl"] = slr(n)
        want[f"{mvm}u_gather/g_asm[{cl}].u_asm"] = slr(n)
    if kernel:
        want[f"{prefix}g_ch[0].u_br"] = slr(0)
        want[f"{prefix}g_ch[3].u_br"] = slr(2)
    want[core + "u_ctrl"] = CTRL
    want[core + "g_db.u_dbg_rv"] = CTRL
    for e in ("u_spm_rx", "u_issue_tx", "u_lq_tx"):
        want[core + "g_rv_local." + e] = CTRL
    want[core + "g_rv_local.u_spm_tx"] = slr(1)
    for e in ("u_issue_rx", "u_lq_rx"):
        want[core + "g_rv_local." + e] = where
    for b in ("u_spm", "u_bridge", "u_dma", "u_qact"):
        want[dp + b] = slr(1)
    for c in range(4):
        want[f"{dp}g_ch[{c}].u_arb"] = slr(1)
    for b in ("u_issue", "u_csr", "g_db.u_dbg"):
        want[dp + b] = where
    return want


@pytest.mark.parametrize("ctrl,where", [(None, "ctrl"), ("slr1", 1), ("slr0", "ctrl")])
def test_kernel_map(ctrl, where):
    r, got, _ = run_hook(ctrl=ctrl)
    assert r.returncode == 0, r.stderr
    assert got == expected("top/tsr_k4/inst/u_kernel/", True, where)


def test_core_map():
    r, got, _ = run_hook(core="")
    assert r.returncode == 0, r.stderr
    assert got == expected("", False, "ctrl")


def test_free_control_plane_leaves_issue_csr_db_unassigned():
    r, got, _ = run_hook(ctrl="free")
    assert r.returncode == 0, r.stderr
    assert not [p for p in got if p.endswith(("/u_issue", "/u_csr", "/g_db.u_dbg"))]
    assert got["top/tsr_k4/inst/u_kernel/u_core/g_rv_local.u_issue_rx"] == slr(1)
    # the control core still has its Pblock
    assert got["top/tsr_k4/inst/u_kernel/u_core/u_ctrl"] == CTRL


def test_only_whole_blocks_are_named():
    r, got, _ = run_hook()
    assert r.returncode == 0, r.stderr
    leaf = re.compile(r"/(\w+_reg|\w+_q|LUT\d|FD\w+|SLICE_\w+)(\[\d+\])*$")
    assert not [p for p in got if leaf.search(p)]
    # no vector unit, attention or trace block is named: the elaboration does not have them
    assert not [p for p in got if any(x in p for x in ("g_vu", "u_vu", "g_att", "kvst", "trw", "g_local_att"))]


@pytest.mark.parametrize("missing", ["g_cl[2].u_cl", "u_qact", "g_rv_local.u_lq_rx", "g_ch[3].u_br", "u_gather/g_asm[1]"])
def test_missing_block_stops_the_hook(missing):
    r, got, _ = run_hook(missing=missing)
    assert r.returncode != 0
    assert "K4 major blocks: missing hierarchy" in r.stderr


def test_bad_control_plane_value():
    r, _, _ = run_hook(ctrl="slr2")
    assert r.returncode != 0
    assert "TSR_K4_CTRL must be slr0, slr1 or free" in r.stderr


# ---- the control Pblock ----

# CLBs per clock region that the control Pblock holds in the platform of the first link's checkpoint (the part's clock
# regions less the platform's pipeline Pblocks in row Y4; measured by tb/platform/viv/k4_pblock_check.tcl, 2026-11-06).
# Only the clock regions the control Pblock holds are listed; their sum is 16,300.
CR_SLICES = {
    "X0Y3": 1860, "X1Y3": 1440, "X0Y4": 1620, "X1Y4": 1200, "X2Y4": 1740,
    "X3Y3": 1440, "X4Y3": 1800, "X5Y3": 1620, "X3Y4": 1060, "X4Y4": 1320, "X5Y4": 1200,
}
# CLBs touched by the control plane in the first link (u_ctrl, u_csr, u_issue, g_db.u_dbg, the RV debug and the link
# ends; the union of their slices), and the platform's own CLBs (memory subsystem, shell) inside the Pblock's regions.
COMPLEX_SLICES = 7584
PLATFORM_SLICES_IN_REGION = 2647


def clock_regions(ranges):
    out = set()
    for r in ranges:
        m = re.fullmatch(r"CLOCKREGION_X(\d+)Y(\d+):CLOCKREGION_X(\d+)Y(\d+)", r)
        assert m, r
        x0, y0, x1, y1 = map(int, m.groups())
        assert x0 <= x1 and y0 <= y1, r
        out |= {f"X{x}Y{y}" for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    return out


def control_regions():
    r, _, children = run_hook()
    assert r.returncode == 0, r.stderr
    assert list(children) == [CTRL]
    parent, ranges = children[CTRL]
    assert parent == slr(0)
    return clock_regions(ranges)


def test_control_pblock_is_a_child_of_the_slr0_pblock():
    r, _, children = run_hook()
    assert r.returncode == 0, r.stderr
    assert children[CTRL][0] == slr(0)
    assert "K4 major blocks: pblock_tsr_ctrl_plane: 14000 SLICE, 250 RAMB36 sites" in r.stdout


def test_control_pblock_stays_in_slr0_and_off_the_ddr0_controller():
    regions = control_regions()
    rows = {int(re.search(r"Y(\d+)", n).group(1)) for n in regions}
    assert rows <= set(range(0, 5)), "SLR0 is clock-region rows Y0 to Y4"
    # the two rows at the SLR1 boundary, without the DDR0 controller's clock regions (docs/platform-u200.md §8.7)
    assert rows == {3, 4}
    assert not regions & {"X2Y1", "X2Y2", "X2Y3"}


def test_control_pblock_reaches_the_slr1_boundary_under_slr1s_pblock():
    regions = control_regions()
    # SLR1's dynamic Pblock covers X0 to X2 (SPM, DMA and QACT are there): the Pblock holds the whole of row Y4 at
    # X0 to X2, so the issue path and the CSR block can sit right across the boundary from them
    assert {"X0Y4", "X1Y4", "X2Y4"} <= regions


def test_control_pblock_is_one_connected_region():
    regions = control_regions()
    seen, todo = set(), [sorted(regions)[0]]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        x, y = map(int, re.fullmatch(r"X(\d+)Y(\d+)", n).groups())
        todo += [m for m in (f"X{x+1}Y{y}", f"X{x-1}Y{y}", f"X{x}Y{y+1}", f"X{x}Y{y-1}") if m in regions]
    assert seen == regions


def test_control_pblock_is_large_enough_for_the_plane_at_85_percent():
    regions = control_regions()
    assert regions == set(CR_SLICES), "update CR_SLICES with the clock regions of the Pblock"
    total = sum(CR_SLICES.values())
    # the plane and the platform's logic in the regions fill at most 85 % of them, with every other block moved out
    assert (COMPLEX_SLICES + PLATFORM_SLICES_IN_REGION) / total <= 0.85


def test_control_pblock_names_clock_regions_only():
    r, _, children = run_hook()
    _, ranges = children[CTRL]
    assert ranges and all(x.startswith("CLOCKREGION_") for x in ranges)
    assert not re.search(r"SLICE|RAMB|DSP|BEL", " ".join(ranges))


@pytest.mark.parametrize("mock,message", [
    ({"no_pblock": "pblock_dynamic_SLR0"}, "missing platform pblock pblock_dynamic_SLR0"),
    ({"parent": "pblock_dynamic_SLR1"}, "is not a child of pblock_dynamic_SLR0"),
])
def test_control_pblock_without_its_parent_stops_the_hook_before_any_assignment(mock, message):
    r, got, _ = run_hook(**mock)
    assert r.returncode != 0
    assert message in r.stderr
    assert not got, "no cell is assigned to a missing or wrong Pblock"


@pytest.mark.parametrize("mock,message", [
    ({"slices": 0}, "has no SLICE sites"),
    ({"ramb36": 63}, "has 63 RAMB36 sites, the control plane needs 64"),
])
def test_control_pblock_capacity_is_checked_after_vivado_fitted_it_to_the_parent(mock, message):
    # With SNAPPING_MODE NESTED Vivado resizes the child at the first assignment, so the hook counts at the end.
    r, got, _ = run_hook(**mock)
    assert r.returncode != 0
    assert message in r.stderr
    assert got, "the capacity check follows the assignments"


def test_control_pblock_with_the_minimum_ramb36_passes():
    r, _, _ = run_hook(ramb36=64)
    assert r.returncode == 0, r.stderr

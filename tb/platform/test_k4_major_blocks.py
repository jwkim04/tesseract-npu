"""The Pblock hook of the full-feature card image (hw/u200/k4_major_blocks.tcl) against a mocked Vivado.

Only the Pblock and hierarchy lookups are mocked: get_cells returns the path asked for, get_pblocks the platform
pblocks and the ones the hook created, create_pblock / resize_pblock / set_property record the two child Pblocks,
get_sites returns a counted list, and add_cells_to_pblock prints the assignment. No Xilinx tool, checkpoint or
placement is involved. The test checks which whole block goes into which pblock, that nothing below a block (a LUT,
register, stage or site) is named, that the control plane follows TSR_K4_CTRL and the vector unit TSR_K4_VU, that a
missing block stops the hook (or is skipped with TSR_K4_SKIP_MISSING), and the two child Pblocks themselves: the
control Pblock, a child of the SLR0 platform pblock at the SLR1 boundary and off the DDR0 controller, and the VU
Pblock, a child of the SLR2 platform pblock next to the SPM and off the DDR3 controller, both large enough.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "hw/u200/k4_major_blocks.tcl"
CTRL = "pblock_tsr_ctrl_plane"
VU = "pblock_tsr_vu"

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
    set of [lsearch -exact $args -of_objects]
    if {$of >= 0 && [lindex $args [expr {$of + 1}]] eq "pblock_tsr_vu" && [string match {*RAMB36_**} $filter] && [info exists ::env(MOCK_VU_RAMB36)]} {
        set n $::env(MOCK_VU_RAMB36)
    }
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


def run_hook(ctrl=None, core=None, missing=None, body=None, vu=None, skip_missing=False, **mock):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TSR_K4", "MOCK_"))}
    env["MOCK_HOOK"] = str(HOOK)
    if ctrl is not None:
        env["TSR_K4_CTRL"] = ctrl
    if vu is not None:
        env["TSR_K4_VU"] = vu
    if skip_missing:
        env["TSR_K4_SKIP_MISSING"] = "1"
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


def expected(prefix, kernel, ctrl_where, vu_where="vu"):
    core = prefix + "u_core/" if kernel else prefix
    dp = core + "u_dp/"
    mvm = dp + "g_mvm.u_mvm/"
    where = CTRL if ctrl_where == "ctrl" else slr(ctrl_where)
    want = {}
    for cl, n in ((0, 0), (1, 2), (2, 0), (3, 2)):
        want[f"{mvm}g_cl[{cl}].u_cl"] = slr(n)
        want[f"{mvm}u_gather/g_asm[{cl}].u_asm"] = slr(n)
    for pair, n in ((0, 0), (1, 2)):
        want[f"{mvm}u_gather/g_pairs.g_pair[{pair}].u_source"] = slr(n)
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
    for c, n in ((0, 0), (1, 1), (2, 1), (3, 2)):
        want[f"{dp}g_ch[{c}].u_arb"] = slr(n)
    for b in ("u_issue", "u_csr", "g_db.u_dbg", "g_xc.u_xc", "g_xc.g_at.u_chk_at"):
        want[dp + b] = where
    want[dp + "g_vu.u_vu"] = VU if vu_where == "vu" else slr(2)
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


def test_vu_on_the_platform_pblock_of_slr2():
    r, got, children = run_hook(vu="slr2")
    assert r.returncode == 0, r.stderr
    assert got == expected("top/tsr_k4/inst/u_kernel/", True, "ctrl", vu_where="slr2")
    assert VU not in children, "the VU Pblock is not made when the vector unit takes the platform pblock"


def test_bad_vu_value():
    r, _, _ = run_hook(vu="slr1")
    assert r.returncode != 0
    assert "TSR_K4_VU must be pblock or slr2" in r.stderr


def test_every_block_is_named_once():
    r, got, _ = run_hook()
    assert r.returncode == 0, r.stderr
    assert len(got) == len(set(got))
    assert all(v in {slr(0), slr(1), slr(2), CTRL, VU} for v in got.values())


def test_free_control_plane_leaves_issue_csr_db_unassigned():
    r, got, _ = run_hook(ctrl="free")
    assert r.returncode == 0, r.stderr
    assert not [p for p in got if p.endswith(("/u_issue", "/u_csr", "/g_db.u_dbg"))]
    assert got["top/tsr_k4/inst/u_kernel/u_core/g_rv_local.u_issue_rx"] == slr(1)
    # the checker classes go with the issue path
    assert not [p for p in got if p.endswith(("/g_xc.u_xc", "/g_xc.g_at.u_chk_at"))]
    # the control core still has its Pblock
    assert got["top/tsr_k4/inst/u_kernel/u_core/u_ctrl"] == CTRL


def test_only_whole_blocks_are_named():
    r, got, _ = run_hook()
    assert r.returncode == 0, r.stderr
    leaf = re.compile(r"/(\w+_reg|\w+_q|LUT\d|FD\w+|SLICE_\w+)(\[\d+\])*$")
    assert not [p for p in got if leaf.search(p)]
    # no local-intermediate bank, lane bank or VU front-end part is named: the whole vector unit goes at once
    assert not [p for p in got if any(x in p for x in ("g_local_att", "u_bank", "u_lanes", "u_gen", "u_inj"))]
    assert [p for p in got if p.endswith("g_vu.u_vu")]


@pytest.mark.parametrize("missing", ["g_cl[2].u_cl", "u_qact", "g_rv_local.u_lq_rx", "g_ch[3].u_br", "u_gather/g_asm[1]",
                                     "g_vu.u_vu", "g_xc.u_xc", "g_xc.g_at.u_chk_at", "g_pairs.g_pair[1].u_source"])
def test_missing_block_stops_the_hook(missing):
    r, got, _ = run_hook(missing=missing)
    assert r.returncode != 0
    assert "K4 major blocks: missing hierarchy" in r.stderr


@pytest.mark.parametrize("missing", ["g_vu.u_vu", "g_xc.u_xc", "g_xc.g_at.u_chk_at", "g_pairs.g_pair[1].u_source"])
def test_skip_missing_reports_and_skips_the_block(missing):
    r, got, children = run_hook(missing=missing, skip_missing=True)
    assert r.returncode == 0, r.stderr
    assert f"K4 major blocks: skipped, not in this netlist: " in r.stdout
    assert not [p for p in got if missing in p]
    want = {p: v for p, v in expected("top/tsr_k4/inst/u_kernel/", True, "ctrl").items() if missing not in p}
    assert got == want
    assert set(children) == {CTRL, VU}, "both child Pblocks are still made and checked"


def test_bad_control_plane_value():
    r, _, _ = run_hook(ctrl="slr2")
    assert r.returncode != 0
    assert "TSR_K4_CTRL must be slr0, slr1 or free" in r.stderr


# ---- the control Pblock ----

# CLBs per clock region that the control Pblock holds in the platform of the first link's checkpoint (the part's clock
# regions less the platform's pipeline Pblocks in row Y4; measured by tb/platform/viv/k4_pblock_check.tcl).
# Only the clock regions the control Pblock holds are listed; their sum is 16,300.
CR_SLICES = {
    "X0Y3": 1860, "X1Y3": 1440, "X0Y4": 1620, "X1Y4": 1200, "X2Y4": 1740,
    "X3Y3": 1440, "X4Y3": 1800, "X5Y3": 1620, "X3Y4": 1060, "X4Y4": 1320, "X5Y4": 1200,
}
# CLBs touched by the control plane in the first link of the reduced image (u_ctrl, u_csr, u_issue, g_db.u_dbg, the RV
# debug and the link ends; the union of their slices), plus the SLR1 study's estimate for the two checker classes and the
# full DB block (0.8 K and 0.6 K CLBs), and the platform's own CLBs (memory subsystem, shell) inside the Pblock's regions.
COMPLEX_SLICES = 7584 + 800 + 600
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


def child_regions(name, parent_slr):
    r, _, children = run_hook()
    assert r.returncode == 0, r.stderr
    assert set(children) == {CTRL, VU}
    parent, ranges = children[name]
    assert parent == slr(parent_slr)
    return clock_regions(ranges)


def control_regions():
    return child_regions(CTRL, 0)


def vu_regions():
    return child_regions(VU, 2)


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


def connected(regions):
    seen, todo = set(), [sorted(regions)[0]]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        x, y = map(int, re.fullmatch(r"X(\d+)Y(\d+)", n).groups())
        todo += [m for m in (f"X{x+1}Y{y}", f"X{x-1}Y{y}", f"X{x}Y{y+1}", f"X{x}Y{y-1}") if m in regions]
    return seen == regions


def test_control_pblock_is_one_connected_region():
    assert connected(control_regions())


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


# ---- the VU Pblock ----

# CLBs per clock region that the platform's SLR2 pblock allows (tb/platform/viv/k4_pblock_check.tcl on the routed first
# link). Only the clock regions the VU Pblock holds are listed; their sum is 25,320.
VU_CR_SLICES = {
    "X0Y10": 1620, "X1Y10": 1200, "X2Y10": 1740,
    "X0Y11": 1860, "X1Y11": 1440, "X2Y11": 1980,
    "X0Y12": 1860, "X1Y12": 1440, "X2Y12": 1980,
    "X0Y13": 1860, "X1Y13": 1440, "X2Y13": 1980,
    "X0Y14": 1740, "X1Y14": 1320, "X2Y14": 1860,
}
# The vector unit's CLBs: 60,104 LUT cells (out-of-context synthesis of the core, docs/platform-u200.md §8.8) at the
# pessimistic 0.229 CLBs per cell of a packed SLR
VU_SLICES = 13800
# RAMB36 of the vector unit (16, and 13 RAMB18) and the hook's minimum for its Pblock
VU_RAMB36_MIN = 32
# The SLR2 controller (DDR[3]) of the platform: its logic is in these clock regions (docs/platform-u200.md §8.7)
MIG2_REGIONS = {"X3Y10", "X3Y11", "X4Y10", "X4Y11", "X4Y12", "X4Y13"}


def test_vu_pblock_is_a_child_of_the_slr2_pblock():
    r, _, children = run_hook()
    assert r.returncode == 0, r.stderr
    assert children[VU][0] == slr(2)
    assert "K4 major blocks: pblock_tsr_vu: 14000 SLICE, 250 RAMB36 sites" in r.stdout


def test_vu_pblock_stays_in_slr2_next_to_the_spm_and_off_the_ddr3_controller():
    regions = vu_regions()
    rows = {int(re.search(r"Y(\d+)", n).group(1)) for n in regions}
    assert rows == set(range(10, 15)), "SLR2 is clock-region rows Y10 to Y14"
    cols = {int(re.match(r"X(\d+)", n).group(1)) for n in regions}
    assert cols == {0, 1, 2}, "SPM, DMA and QACT of SLR1 are at X0 to X2"
    assert not regions & MIG2_REGIONS


def test_vu_pblock_is_one_connected_region_and_holds_the_whole_vu_with_room():
    regions = vu_regions()
    assert connected(regions)
    assert regions == set(VU_CR_SLICES), "update VU_CR_SLICES with the clock regions of the Pblock"
    # the vector unit fills at most 60 % of the region at the pessimistic density: the clusters 1 and 3 and the shell
    # slices share it
    assert VU_SLICES / sum(VU_CR_SLICES.values()) <= 0.6


def test_vu_pblock_names_clock_regions_only():
    _, _, children = run_hook()
    _, ranges = children[VU]
    assert ranges and all(x.startswith("CLOCKREGION_") for x in ranges)
    assert not re.search(r"SLICE|RAMB|DSP|BEL", " ".join(ranges))


def test_the_two_child_pblocks_do_not_overlap():
    assert not control_regions() & vu_regions()


def test_vu_pblock_without_its_parent_stops_the_hook_before_any_assignment():
    r, got, _ = run_hook(no_pblock="pblock_dynamic_SLR2")
    assert r.returncode != 0
    assert "missing platform pblock pblock_dynamic_SLR2" in r.stderr
    assert not got


@pytest.mark.parametrize("ramb36,ok", [(VU_RAMB36_MIN - 1, False), (VU_RAMB36_MIN, True)])
def test_vu_pblock_ramb36_minimum(ramb36, ok):
    r, _, _ = run_hook(vu_ramb36=ramb36)
    assert (r.returncode == 0) == ok
    if not ok:
        assert f"pblock_tsr_vu has {ramb36} RAMB36 sites, the vector unit needs {VU_RAMB36_MIN}" in r.stderr

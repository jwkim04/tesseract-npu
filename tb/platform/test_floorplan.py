"""Run the real k4 floorplan against a small synthesized-object fixture.

The fixture includes the reparented cones found in the failed checkpoint.
It checks physical ownership and preserves the two ends of crossing pipes;
no Xilinx tool or checkpoint is needed.
"""
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("floorplan_fixture.tcl")
SCRIPT = REPO / "hw/u200/k4_floorplan.tcl"


def run_fixture(mode, case):
    return subprocess.run(["tclsh", str(FIXTURE), str(SCRIPT), mode, case],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("mode", ["core", "kernel"])
@pytest.mark.parametrize("case", ["normal", "placed", "membership", "post_normal",
                                  "stall_reparented", "stall_existing_parent", "repeat",
                                  "shared_bc", "shared_bc_deep", "shared_bc_repeat",
                                  "post_shared_bc", "shared_vu", "post_shared_vu"])
def test_k4_floorplan_locality(mode, case):
    result = run_fixture(mode, case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIXTURE PASS" in result.stdout


@pytest.mark.parametrize("case,diagnostic", [
    ("missing_selector", "selector"),
    ("unknown_lane_leaf", "unknown"),
    ("missing_att_pipe", "u_ajp"),
    ("late_selector", "a_att_q_reg_replica_99"),
    ("misplaced", "is not placed in SLR1"),
    ("post_late_selector", "a_att_q_reg_replica_99"),
    ("post_misplaced", "is not placed in SLR1"),
    ("post_missing_proc", "locality check unavailable"),
    ("stall_shared", "stall aggregation"),
    ("stall_orphan", "stall aggregation"),
    ("stall_unknown", "stall aggregation"),
    ("stall_missing", "att_stall_q_reg"),
    ("stall_unknown_name", "unknown stall aggregation"),
    ("stall_outside_scope", "unknown stall aggregation"),
    ("post_stall_misplaced", "is not placed in SLR1"),
    ("post_stall_replica", "is not assigned"),
    ("post_stall_source_misplaced", "is not placed in SLR0"),
    ("post_stall_source_replica", "is not assigned"),
    ("shared_bc_empty_head", "shared broadcast head is empty"),
    ("shared_bc_unknown_head", "unknown shared broadcast head leaf"),
    ("shared_bc_missing_tail", "fewer than 1 register stages"),
    ("post_shared_bc_head_misplaced", "is not placed in SLR1"),
    ("post_shared_bc_tail_misplaced", "is not placed in SLR0"),
    ("post_shared_bc_replica", "is not assigned"),
    ("post_shared_vu_misplaced", "is not placed in SLR0"),
    ("post_shared_vu_missing", "empty locality check"),
])
def test_k4_floorplan_rejects_missing_or_unowned_cells(case, diagnostic):
    result = run_fixture("core", case)
    assert result.returncode != 0, result.stdout
    assert diagnostic in result.stderr, result.stdout + result.stderr

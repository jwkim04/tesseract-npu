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
                                  "post_shared_bc", "shared_vu", "post_shared_vu",
                                  "stall_comp_numbered", "post_stall_comp_numbered",
                                  "banked", "post_banked", "banked_repeat", "paired", "paired_repeat", "post_paired", "banked_paired", "banked_arithmetic", "banked_arithmetic_repeat", "post_banked_arithmetic", "banked_drain", "banked_drain_repeat", "post_banked_drain"])
def test_k4_floorplan_locality(mode, case):
    result = run_fixture(mode, case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIXTURE PASS" in result.stdout


@pytest.mark.parametrize("case,diagnostic", [
    ("banked_drain_local_stage", "noncentral or mismapped return"),
    ("banked_drain_crossbank", "noncentral or mismapped return"),
    ("banked_drain_wrong_bit", "noncentral or mismapped return"),
    ("banked_drain_unrelated_control", "unknown input ownership"),
    ("banked_drain_unknown_source", "unknown input ownership"),
    ("banked_drain_missing_source", "missing return/control provenance"),
    ("banked_drain_missing_control", "missing return/control provenance"),
    ("banked_drain_no_consumer", "no reduction consumer"),
    ("banked_drain_shared_sink", "wrong reduction consumer"),
    ("banked_drain_wrong_consumer_bit", "wrong reduction consumer"),
    ("banked_drain_bad_type", "unknown central drain helper"),
    ("banked_drain_unknown_family", "unknown central drain helper"),
    ("post_banked_drain_misplaced", "is not placed in SLR1"),
    ("post_banked_drain_return_misplaced", "is not placed in SLR1"),
    ("post_banked_drain_unowned", "is not assigned"),
    ("banked_arithmetic_external_source", "nonlocal or unknown endpoint"),
    ("banked_arithmetic_crossbank", "nonlocal or unknown endpoint"),
    ("banked_arithmetic_central_stage", "nonlocal or unknown endpoint"),
    ("banked_arithmetic_crosslane", "spans multiple lanes"),
    ("banked_arithmetic_no_source", "no local lane anchor"),
    ("banked_arithmetic_no_sink", "no local lane anchor"),
    ("banked_arithmetic_bad_type", "arithmetic primitive"),
    ("banked_arithmetic_external_sink", "nonlocal or unknown endpoint"),
    ("post_banked_arithmetic_misplaced", "is not placed in SLR2"),
    ("post_banked_arithmetic_unowned", "is not assigned"),
    ("ddr_unknown_boundary", "unknown DMA boundary ownership"),
    ("ddr_missing_boundary", "not found"),
    ("post_ddr_boundary_misplaced", "is not placed in SLR1"),
    ("post_ddr_boundary_wrongowner", "is not assigned"),
    ("post_ddr_boundary_unowned", "is not assigned"),
    ("paired_unknown_scope", "unknown paired gather ownership"),
    ("paired_missing_control", "missing gather pair control"),
    ("post_paired_misplaced", "is not placed in SLR2"),
    ("post_paired_unowned", "is not assigned"),
    ("banked_missing_control", "not found"),
    ("banked_missing_token", "not found"),
    ("banked_missing_bank", "not found"),
    ("banked_missing_reset", "fewer than 2 stages"),
    ("post_banked_misplaced", "is not placed in SLR2"),
    ("post_banked_unowned", "is not assigned"),
    ("post_ddr_unowned", "is not assigned"),
    ("post_ddr_misplaced", "is not placed in SLR0"),
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
    ("stall_comp_bad_suffix", "unknown stall aggregation"),
    ("post_stall_comp_numbered_unowned", "is not assigned"),
    ("post_stall_comp_numbered_misplaced", "is not placed in SLR1"),
    ("post_stall_comp_numbered_shared", "disconnected or shared stall aggregation"),
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


@pytest.mark.parametrize("mode", ["core", "kernel"])
@pytest.mark.parametrize("case", ["banked", "post_banked", "banked_repeat", "banked_paired",
                                  "banked_arithmetic", "post_banked_arithmetic",
                                  "banked_drain", "post_banked_drain", "banked_drain_repeat"])
def test_k4_half_vu_floorplan(mode, case):
    result = run_fixture(mode, "half_"+case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIXTURE PASS" in result.stdout


@pytest.mark.parametrize("case,diagnostic", [
    ("banked_lane_gap", "lane count"),
    ("banked_phantom_lane", "unknown banked VU lane ownership"),
    ("post_banked_width_mismatch", "widths differ"),
    ("banked_width_mismatch", "widths differ"),
    ("banked_missing_control", "not found"),
    ("banked_drain_crossbank", "noncentral or mismapped return"),
    ("banked_drain_wrong_bit", "noncentral or mismapped return"),
    ("banked_drain_shared_sink", "wrong reduction consumer"),
    ("post_banked_drain_return_misplaced", "is not placed in SLR1"),
    ("banked_arithmetic_crossbank", "nonlocal or unknown endpoint"),
    ("banked_arithmetic_crosslane", "spans multiple lanes"),
])
def test_k4_half_vu_floorplan_rejects_bad_shapes_and_ownership(case, diagnostic):
    result = run_fixture("core", "half_"+case)
    assert result.returncode != 0, result.stdout
    assert diagnostic in result.stderr, result.stdout + result.stderr

"""The native K4 diagnostic checkpoint hook; Tcl mocks only, no CAD."""
from pathlib import Path
import subprocess

import pytest

HOOK = Path(__file__).resolve().parents[2] / "hw/u200/k4_post_opt.tcl"


@pytest.mark.parametrize("write_fails", [False, True])
def test_post_opt_checkpoint_path_and_failure(tmp_path, write_fails):
    body = 'error {checkpoint write failed}' if write_fails else (
        'set f [open [lindex $args end] w]; puts $f {actual optimized design}; close $f')
    script = tmp_path / "native_opt_post.tcl"
    script.write_text("proc write_checkpoint {args} {\n"
                      "  if {$args ne [list -force [file join [pwd] k4_post_opt.dcp]]} {error {wrong checkpoint path}}\n"
                      f"  {body}\n"
                      "}\n"
                      f"source {{{HOOK}}}\n"
                      'puts {native downstream hook continues}\n')
    result = subprocess.run(["tclsh", str(script)], cwd=tmp_path, text=True, capture_output=True)
    checkpoint = tmp_path / "k4_post_opt.dcp"
    if write_fails:
        assert result.returncode != 0
        assert "checkpoint write failed" in result.stderr
        assert "native downstream hook continues" not in result.stdout
        assert not checkpoint.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert checkpoint.read_text().strip() == "actual optimized design"
        assert "K4 POST-OPT CHECKPOINT COMPLETE" in result.stdout
        assert "native downstream hook continues" in result.stdout

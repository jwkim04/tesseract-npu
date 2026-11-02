"""Render the real U200 build script with isolated fake tools; never launch CAD."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("build_jobs,synth_jobs,expected", [
    (None, None, (8, 8, 8)),
    ("8", "1", (1, 8, 8)),
    ("4", "1", (1, 4, 4)),
    ("8", "16", (16, 8, 8)),
    ("8", "0", None),
    ("8", "9", (9, 8, 8)),
    ("8", "text", None),
    ("4", "5", (5, 4, 4)),
    ("8", "-1", None),
])
def test_link_job_rendering(tmp_path, build_jobs, synth_jobs, expected):
    root = tmp_path / "repo"
    hw = root / "hw/u200"
    hw.mkdir(parents=True)
    script = hw / "build_xclbin.sh"
    shutil.copy2(REPO / "hw/u200/build_xclbin.sh", script)
    config = (REPO / "hw/u200/k4.cfg").read_text()
    (hw / "k4.cfg").write_text(config)
    hook_paths = []
    for line in config.splitlines():
        if line.startswith(("param=", "prop=")) and line.endswith(".tcl"):
            relative = line.rsplit("=", 1)[1].removeprefix("../../../")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, target)
            hook_paths.append(str(target))

    tools = tmp_path / "tools"
    bin_dir = tools / "bin"
    bin_dir.mkdir(parents=True)
    trace = tmp_path / "calls.jsonl"
    for product in ("Vivado", "Vitis"):
        settings = tools / product / "2024.2/settings64.sh"
        settings.parent.mkdir(parents=True)
        settings.write_text('export PATH="$TSR_TEST_BIN:/usr/bin:/bin"\n'
                            'printf \'{"tool":"settings"}\\n\' >> "$TSR_TEST_TRACE"\n')
    fake_link = bin_dir / "v++"
    fake_link.write_text(f"#!{sys.executable}\n" + '''
from pathlib import Path
import json, os, sys
args = sys.argv[1:]
with Path(os.environ["TSR_TEST_TRACE"]).open("a") as f:
    f.write(json.dumps({"tool": "v++", "argv": args}) + "\\n")
assert "--link" in args
Path(args[args.index("-o") + 1]).write_text("fake xclbin; no CAD executed\\n")
''')
    fake_link.chmod(0o755)
    fake_vivado = bin_dir / "vivado"
    fake_vivado.write_text('#!/bin/bash\nprintf \'{"tool":"vivado"}\\n\' >> "$TSR_TEST_TRACE"\nexit 97\n')
    fake_vivado.chmod(0o755)
    platform = tmp_path / "fake.xpfm"
    platform.touch()
    xo = root / "build/u200/xo/tsr_u200_k4.xo"
    if expected is not None:
        xo.parent.mkdir(parents=True)
        xo.write_text("isolated fake XO\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        xo.with_suffix(".sources.sha256").write_text(f"{digest}  hw/u200/build_xclbin.sh\n")
    # Invalid values must stop even when an absent XO would require packaging.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TSR_") and k != "PLATFORM_REPO_PATHS"}
    out = tmp_path / "out"
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin", TSR_XILINX=str(tools),
               TSR_U200_PLATFORM=str(platform), TSR_U200_OUT=str(out),
               TSR_TEST_BIN=str(bin_dir), TSR_TEST_TRACE=str(trace))
    if build_jobs is not None:
        env["TSR_BUILD_JOBS"] = build_jobs
    if synth_jobs is not None:
        env["TSR_U200_SYNTH_JOBS"] = synth_jobs
    result = subprocess.run([str(script), "k4"], cwd=root, env=env,
                            text=True, capture_output=True, timeout=30)
    calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    rendered = out / "k4.link.cfg"
    if expected is None:
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert "TSR_U200_SYNTH_JOBS must be a positive integer" in result.stderr
        assert not calls
        assert not rendered.exists()
        assert not xo.parent.exists()
        return

    assert result.returncode == 0, (result.stdout, result.stderr)
    lines = rendered.read_text().splitlines()
    for key, value in zip(("synth.jobs", "impl.jobs", "param=general.maxThreads"), expected):
        assert [line for line in lines if line.startswith(key + "=")] == [f"{key}={value}"]
    assert "kernel_frequency=0:250|1:166.666667" in lines
    assert hook_paths
    assert [line.rsplit("=", 1)[1] for line in lines
            if line.startswith(("param=", "prop=")) and line.endswith(".tcl")] == hook_paths
    assert [call["tool"] for call in calls] == ["settings", "v++"]
    args = calls[-1]["argv"]
    assert args[args.index("--config") + 1] == str(rendered)
    assert args[-1] == str(xo)
    assert (out / "tsr_u200_k4.xclbin").is_file()
    assert not xo.with_suffix(".package.log").exists()

#!/usr/bin/env python3
"""tb/platform/bw/run_bw.py — build (Verilator, cached) and run the bandwidth kernel bench
(tb/platform/bw/bw_bench.cpp; docs/bandwidth-kernel.md §5).

    run_bw.py [--top body|tsr_bw_k4] [--param K=V]... [--jobs N] [--waves] [--build-only] [-- BENCH ARGS]

  --top body        tsr_bw_kernel with the parameters (default NUM_CH=4 ADDR_BEAT_W=28 AXI_SLICES=2
                    CTRL_PIPE=2; --param overrides)
  --top tsr_bw_k4   the packaged top through tb/platform/bw/sv/kwrap_tsr_bw_k4.sv (fixed parameters)
  BENCH ARGS        e.g. --test all --regime all --seed 1 --runs 40 --out results.jsonl
Prints the binary path with --build-only. Exit status: the bench's (number of failed test groups).
Build root: $TSR_SIM_BUILD_ROOT/bw, else build/bw (one directory per configuration, reused while
the sources and flags are unchanged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RTL_BW = REPO / "rtl" / "platform" / "alveo_xrt" / "bw"
FILELIST = RTL_BW / "filelist.f"
WRAP = HERE / "sv" / "kwrap_tsr_bw_k4.sv"
BENCH = HERE / "bw_bench.cpp"
MEMMODEL_DIR = REPO / "tb" / "fastreplay"
DEFAULTS = dict(NUM_CH=4, ADDR_BEAT_W=28, AXI_SLICES=2, CTRL_PIPE=2)
PACKAGED = dict(NUM_CH=4, ADDR_BEAT_W=28, AXI_SLICES=2, CTRL_PIPE=2)     # gen_bw.py PARAMS


def build_root() -> Path:
    if os.environ.get("TSR_SIM_BUILD_ROOT"):
        return Path(os.environ["TSR_SIM_BUILD_ROOT"]) / "bw"
    return REPO / "build" / "bw"


def rtl_sources(filelist: Path = FILELIST) -> list:
    out = []
    for line in filelist.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if line:
            out.append((filelist.parent / line).resolve())
    return out


def _sh(cmd, cwd, log: Path) -> None:
    with open(log, "a") as f:
        f.write("$ %s\n" % " ".join(shlex.quote(str(c)) for c in cmd))
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError("command failed (%d), see %s:\n%s" % (r.returncode, log, tail))


def build(top: str = "body", params: dict | None = None, jobs: int = 4, waves: bool = False, log=print,
          filelist: Path = FILELIST, wno_fatal: bool = False, root: Path | None = None) -> Path:
    """Build the bench (cached); filelist / wno_fatal: mutation campaigns (a copy of the RTL)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    if top == "tsr_bw_k4":
        if params and any(PACKAGED.get(k) != int(v) for k, v in params.items()):
            raise SystemExit("run_bw.py: tsr_bw_k4 has fixed parameters %s" % PACKAGED)
        p = dict(PACKAGED)
    elif top != "body":
        raise SystemExit("run_bw.py: --top body | tsr_bw_k4")
    filelist = Path(filelist).resolve()
    rtl_dir = filelist.parent
    srcs = rtl_sources(filelist) + ([WRAP] if top == "tsr_bw_k4" else [])
    vflags = ["-O3", "--assert", "--x-assign", "unique", "--x-initial", "unique",
              "-Werror-USERFATAL", "-Werror-USERERROR"]
    if wno_fatal:
        vflags.append("-Wno-fatal")
    if waves:
        vflags += ["--trace"]
    vtop = "tsr_bw_kernel" if top == "body" else "kwrap_tsr_bw_k4"
    desc = "%s %s" % (vtop if top == "body" else "tsr_bw_k4 (packaged)",
                      " ".join("%s=%s" % kv for kv in sorted(p.items())))
    defs = ["-DBW_NUM_CH=%d" % int(p["NUM_CH"]), "-DBW_AW=%d" % int(p["ADDR_BEAT_W"]),
            "-DBW_SLICES=%d" % int(p["AXI_SLICES"]), "-DBW_DESC='\"%s\"'" % desc, "-DVL_USER_STOP"]
    ident = dict(top=top, params=sorted((k, str(v)) for k, v in p.items()), vflags=vflags, defs=defs,
                 srcs=[str(s) for s in srcs])
    key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    bdir = (Path(root) if root else build_root()) / ("%s_c%d_%s" % ("k4" if top == "tsr_bw_k4" else "body", int(p["NUM_CH"]), key))
    binary = bdir / "obj" / "bw_bench"
    h = hashlib.sha256()
    try:
        h.update(subprocess.run(["verilator", "--version"], capture_output=True, text=True).stdout.encode())
    except FileNotFoundError:
        raise SystemExit("run_bw.py: verilator not found on PATH")
    h.update(json.dumps(ident, sort_keys=True).encode())
    for f in srcs + [BENCH, rtl_dir / "tsr_bw_regs.h", MEMMODEL_DIR / "memmodel.h"]:
        h.update(str(f).encode())
        h.update(Path(f).read_bytes())
    stamp = h.hexdigest()
    stamp_file = bdir / "stamp"
    if binary.is_file() and stamp_file.is_file() and stamp_file.read_text() == stamp:
        return binary
    t0 = time.time()
    bdir.mkdir(parents=True, exist_ok=True)
    if stamp_file.exists():
        stamp_file.unlink()
    blog = bdir / "build.log"
    blog.write_text("")
    log("build %s -> %s" % (desc, bdir))
    cmd = ["verilator", "--cc", "--exe", "--build", "-j", str(jobs), "--top-module", vtop, "--prefix", "Vbw",
           "--Mdir", bdir / "obj", "-o", "bw_bench"] + vflags
    if top == "body":
        cmd += ["-G%s=%s" % (k, v) for k, v in sorted(p.items())]
    cmd += ["-CFLAGS", "-std=gnu++17 -I%s -I%s %s" % (MEMMODEL_DIR, rtl_dir, " ".join(defs)),
            "-MAKEFLAGS", "OPT_FAST=-O2", "-MAKEFLAGS", "OPT_SLOW=-O1", "-MAKEFLAGS", "OPT_GLOBAL=-O2"]
    cmd += srcs + [BENCH]
    _sh(cmd, bdir, blog)
    if not binary.is_file():
        raise RuntimeError("verilator did not produce %s (see %s)" % (binary, blog))
    for gch in (bdir / "obj").glob("*.gch"):
        gch.unlink()
    stamp_file.write_text(stamp)
    log("built in %.0f s" % (time.time() - t0))
    return binary


def main(argv) -> int:
    if "--" in argv:
        i = argv.index("--")
        own, rest = argv[:i], argv[i + 1:]
    else:
        own, rest = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", default="body")
    ap.add_argument("--param", action="append", default=[], help="K=V (body only)")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("TSR_BUILD_JOBS", "4")))
    ap.add_argument("--waves", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args(own)
    params = {}
    for kv in a.param:
        k, v = kv.split("=", 1)
        params[k] = int(v, 0)
    binary = build(a.top, params, a.jobs, a.waves)
    if a.build_only:
        print(binary)
        return 0
    return subprocess.run([str(binary)] + rest).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

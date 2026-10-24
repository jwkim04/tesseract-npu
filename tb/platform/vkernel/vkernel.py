#!/usr/bin/env python3
"""tb/platform/vkernel/vkernel.py — build and run the Verilated kernel server for TSR_DEVICE=vkernel
(docs/platform-u200.md §13).

    vkernel.py build [--config C] [--top body|packaged] [--filelist F] [--param K=V]...
    vkernel.py env   [build options] [server options]          # prints the environment to export
    vkernel.py run   [build options] [server options] -- CMD [ARGS...]

Configurations (the U200 kernels with the bitstream parameters of docs/rtl-m3.md §14.2: SPM 4 MiB,
SPM_RD_LAT 4, MVM_BC_PIPE 2, MVM_RES_PIPE 2, MVM_MAC_LAT 9, RST_PIPE 3, AXI_SLICES 2, RST_STAGES 3):
  k1    NUM_CH 1, T_MAX 1   (tsr_u200_k1, the bring-up kernel)
  k4t1  NUM_CH 4, T_MAX 1
  k4    NUM_CH 4, T_MAX 8   (tsr_u200_k4, the target kernel)
  ch3   NUM_CH 3, T_MAX 3, SPM 256 KiB (the configuration of traces/m3/ch3; quick protocol tests)
--top body (default) builds the kernel body tsr_xrt_kernel with these parameters; --top packaged
builds the generated top (tsr_u200_k1 for k1, tsr_u200_k4 for k4) behind its kwrap_* wrapper with
the parameters fixed in rtl/platform/alveo_xrt/gen_tops.py. --filelist: the core filelist (default
rtl/filelist_xilinx.f, the .xo sources; rtl/filelist.f is behaviourally identical).

Server options (passed to vkernel_server): --regime fast|rand|slow|ordered (default fast), --mem K=V,
--dut K=V (kernel bench options: base=rand|zero|0xHEX, axil=fast, p_* error injection via --mem),
--seed N, --runlog FILE, --summary FILE, --max-run-cycles N.
Client options: --ctrl kds|irq|regs (TSR_VKERNEL_CTRL, default kds), --poll N (TSR_VKERNEL_POLL).

`run` builds the server if needed, exports TSR_DEVICE=vkernel, TSR_VKERNEL_SERVER (the server
command line; the client spawns it), TSR_VKERNEL_CTRL / _POLL and TSR_XRT_TIMEOUT_MS=0 (unless set:
simulated runs take seconds to minutes of wall time; the server's --max-run-cycles bounds a hang),
runs CMD, waits for the server's summary (--summary, default <build>/last_summary.json) and prints
it. Exit status: CMD's, or 1 if the server reported findings.

Build root: $TSR_VKERNEL_BUILD, else $TSR_SIM_BUILD_ROOT/vkernel, else build/vkernel; one directory
per configuration, rebuilt when a source (RTL, bench, server, sw/sim, protocol header) changes.
make -j$TSR_BUILD_JOBS (default 4).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PLATFORM = REPO / "tb" / "platform"
FASTREPLAY = REPO / "tb" / "fastreplay"
TRANSPORT = REPO / "sw" / "transport"
SERVER_SRC = HERE / "vkernel_server.cpp"
VERSION = 1                       # bump when the build recipe changes


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fr = _load("fastreplay_run_vkernel", FASTREPLAY / "run.py")

U200_PARAMS = dict(SPM_RD_LAT=4, MVM_BC_PIPE=2, MVM_RES_PIPE=2, MVM_MAC_LAT=9, RST_PIPE=3, AXI_SLICES=2,
                   RST_STAGES=3)
CONFIGS = {
    "k1": dict(cfg=dict(fr.U200_CFG, num_ch=1, t_max=1), packaged="tsr_u200_k1"),
    "k4t1": dict(cfg=dict(fr.U200_CFG, num_ch=4, t_max=1), packaged=None),
    "k4": dict(cfg=dict(fr.U200_CFG, num_ch=4, t_max=8), packaged="tsr_u200_k4"),
    "ch3": dict(cfg=dict(fr.U200_CFG, num_ch=3, t_max=3, spm_bytes=256 * 1024), packaged=None),
}


def build_root() -> Path:
    if os.environ.get("TSR_VKERNEL_BUILD"):
        return Path(os.environ["TSR_VKERNEL_BUILD"])
    if os.environ.get("TSR_SIM_BUILD_ROOT"):
        return Path(os.environ["TSR_SIM_BUILD_ROOT"]) / "vkernel"
    return REPO / "build" / "vkernel"


def parse_params(kvs) -> dict:
    out = {}
    for kv in kvs or []:
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit("vkernel.py: --param needs K=V, got %r" % kv)
        out[k] = int(v, 0)
    return out


def build(config: str, top: str = "body", filelist=None, overrides=None, jobs=None, log=print) -> Path:
    """Build (or reuse) the server binary of a configuration; returns its path."""
    if config not in CONFIGS:
        raise SystemExit("vkernel.py: unknown configuration %s (%s)" % (config, ", ".join(CONFIGS)))
    c = CONFIGS[config]
    cfg = c["cfg"]
    filelist = Path(filelist or REPO / "rtl" / "filelist_xilinx.f").resolve()
    params = fr.core_params(cfg)
    params["HAS_MVM"] = 1
    if top == "body":
        params.update(U200_PARAMS)
        params.update(overrides or {})
        vtop = "tsr_xrt_kernel"
    elif top == "packaged":
        if not c["packaged"]:
            raise SystemExit("vkernel.py: configuration %s has no packaged top" % config)
        if overrides:
            raise SystemExit("vkernel.py: the packaged tops fix their parameters (--param needs --top body)")
        params.update(fr.PACKAGED[c["packaged"]]["params"])
        vtop = "kwrap_%s" % c["packaged"]
    else:
        raise SystemExit("vkernel.py: --top body or packaged")
    impl = {k: v for k, v in sorted(params.items()) if k not in fr.CONFIG_PARAMS or k == "RD_OUTSTANDING"}
    desc = "vkernel %s %s NUM_CH=%d T_MAX=%d SPM_BYTES=%d CH_BYTES=%d %s (%s)" % (
        config, vtop, cfg["num_ch"], cfg["t_max"], cfg["spm_bytes"], cfg["ch_bytes"],
        " ".join("%s=%s" % kv for kv in impl.items()), filelist.name)
    srcs = fr.rtl_sources(filelist) + fr.rtl_sources(fr.PLATFORM_FL)
    if top == "packaged":
        srcs.append(PLATFORM / "sv" / ("%s.sv" % vtop))
    tb = [SERVER_SRC, PLATFORM / "kernel_dut.h", FASTREPLAY / "fr_common.h", FASTREPLAY / "memmodel.h",
          TRANSPORT / "tsr_vkernel_proto.h"]
    vflags = ["-O3", "--assert", "--x-assign", "unique", "--x-initial", "unique", "-Werror-USERFATAL",
              "-Werror-USERERROR"]
    ident = dict(v=VERSION, params=sorted((k, str(v)) for k, v in params.items()), filelist=str(filelist),
                 srcs=[str(s) for s in srcs], vflags=vflags, top=vtop)
    key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    bdir = build_root() / ("%s_%s_%s" % (config, top, key))
    binary = bdir / "obj" / "vkernel_server"
    cfg_h = fr.fr_config_h(cfg, params, desc)
    h = hashlib.sha256()
    vver = subprocess.run(["verilator", "--version"], capture_output=True, text=True).stdout
    h.update(vver.encode())
    h.update(cfg_h.encode())
    h.update(json.dumps(ident, sort_keys=True).encode())
    hdrs = sorted((fr.SW_SIM / "include").glob("*.h"))
    for f in list(srcs) + tb + [fr.SW_SIM / s for s in fr.C_SOURCES] + hdrs:
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
    log("vkernel: build %s -> %s" % (desc, bdir))
    (bdir / "fr_config.h").write_text(cfg_h)
    swo = bdir / "sw"
    swo.mkdir(exist_ok=True)
    cc = os.environ.get("CC", "gcc")
    objs = []
    for s in fr.C_SOURCES:
        o = swo / (Path(s).stem + ".o")
        fr._sh([cc, "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I", fr.SW_SIM / "include", "-c",
                fr.SW_SIM / s, "-o", o], bdir, blog)
        objs.append(o)
    lib = bdir / "libtsrsw.a"
    if lib.exists():
        lib.unlink()
    fr._sh(["ar", "rcs", lib] + objs, bdir, blog)
    jobs = jobs or int(os.environ.get("TSR_BUILD_JOBS", "4"))
    cmd = ["verilator", "--cc", "--exe", "--build", "-j", str(jobs), "--top-module", vtop, "--Mdir", bdir / "obj",
           "-o", "vkernel_server", "--prefix", "Vkernel"] + vflags
    if top == "body":
        cmd += ["-G%s=%s" % (k, fr._pval(k, v)) for k, v in sorted(params.items())]
    incs = [bdir, FASTREPLAY, PLATFORM, TRANSPORT, fr.SW_SIM / "include"]
    cmd += ["-CFLAGS", "-std=gnu++17 %s -DVL_USER_STOP" % " ".join("-I%s" % i for i in incs),
            "-LDFLAGS", "%s -lpthread -lm" % lib,
            "-MAKEFLAGS", "OPT_FAST=-O2", "-MAKEFLAGS", "OPT_SLOW=-O1", "-MAKEFLAGS", "OPT_GLOBAL=-O2"]
    cmd += srcs + [SERVER_SRC]
    fr._sh(cmd, bdir, blog)
    if not binary.is_file():
        raise RuntimeError("verilator did not produce %s (see %s)" % (binary, blog))
    for gch in (bdir / "obj").glob("*.gch"):
        gch.unlink()
    stamp_file.write_text(stamp)
    log("vkernel: built in %.0f s" % (time.time() - t0))
    return binary


def server_cmdline(binary: Path, a) -> list:
    cmd = [str(binary), "--regime", a.regime, "--seed", str(a.seed)]
    for kv in a.mem or []:
        cmd += ["--mem", kv]
    for kv in a.dut or []:
        cmd += ["--dut", kv]
    if a.runlog:
        cmd += ["--runlog", str(Path(a.runlog).resolve())]
    if a.summary:
        cmd += ["--summary", str(Path(a.summary).resolve())]
    if a.max_run_cycles:
        cmd += ["--max-run-cycles", str(a.max_run_cycles)]
    for w in cmd:
        if any(ch.isspace() for ch in w):
            raise SystemExit("vkernel.py: TSR_VKERNEL_SERVER is split at blanks; %r contains one" % w)
    return cmd


def client_env(binary: Path, a) -> dict:
    env = {"TSR_DEVICE": "vkernel", "TSR_VKERNEL_SERVER": " ".join(server_cmdline(binary, a)),
           "TSR_VKERNEL_CTRL": a.ctrl}
    if a.poll:
        env["TSR_VKERNEL_POLL"] = str(a.poll)
    if "TSR_XRT_TIMEOUT_MS" not in os.environ:
        env["TSR_XRT_TIMEOUT_MS"] = "0"
    return env


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["build", "env", "run"])
    p.add_argument("--config", default="k4t1", choices=list(CONFIGS))
    p.add_argument("--top", default="body", choices=["body", "packaged"])
    p.add_argument("--filelist")
    p.add_argument("--param", action="append")
    p.add_argument("--jobs", type=int)
    p.add_argument("--regime", default="fast")
    p.add_argument("--mem", action="append")
    p.add_argument("--dut", action="append")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runlog")
    p.add_argument("--summary")
    p.add_argument("--max-run-cycles", type=int)
    p.add_argument("--ctrl", default="kds", choices=["kds", "irq", "regs"])
    p.add_argument("--poll", type=int)
    a = p.parse_args(argv)
    binary = build(a.config, a.top, a.filelist, parse_params(a.param), a.jobs,
                   log=lambda *x: print(*x, file=sys.stderr))
    if a.action == "build":
        print(binary)
        return 0
    if a.action == "run" and not a.summary:
        a.summary = str(binary.parent.parent / "last_summary.json")
    env = client_env(binary, a)
    if a.action == "env":
        for k, v in env.items():
            print("export %s=%s" % (k, shlex.quote(v)))
        return 0
    if not cmd:
        raise SystemExit("vkernel.py run: give the command after --")
    summ = Path(a.summary)
    if summ.exists():
        summ.unlink()
    t0 = time.time()
    rc = subprocess.run(cmd, env=dict(os.environ, **env)).returncode
    wall = time.time() - t0
    for _ in range(600):              # the server writes its summary when the client is gone
        if summ.exists():
            break
        time.sleep(0.1)
    if not summ.exists():
        print("vkernel.py: no server summary at %s" % summ, file=sys.stderr)
        return rc or 1
    s = json.loads(summ.read_text())
    print("vkernel.py: command exit %d, wall %.1f s; server: %d runs, %d findings, %d kernel cycles, "
          "PERF_CYCLES %d, PERF_MVM_CYCLES %d, %.0f cycles/s" % (
              rc, wall, s["runs"], s["findings"], s["kernel_cycles"], s["perf_cycles"], s["perf_mvm_cycles"],
              s["cycles_per_s"]), file=sys.stderr)
    if s["findings"]:
        return rc or 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

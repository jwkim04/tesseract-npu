#!/usr/bin/env python3
"""tb/fastreplay/run.py — fast RTL trace replay of tsr_core (Verilator C++ testbench, no cocotb).

    tb/fastreplay/run.py [options] [TRACE_DIR | TREE | GLOB ...]       (default: traces/m2)

Every trace directory (meta.json, tsr_trace.h) found under the arguments is replayed on the RTL
core by tb/fastreplay/fastreplay.cpp with the checks of tb/common/replay.run_trace plus the PERF
and per-source checks of tb/core core_traces (fastreplay.cpp header). Traces are grouped by their
meta.json "config"; each group gets one Verilator build of tsr_core elaborated with that
configuration (Trace.core_parameters()) plus HAS_MVM and implementation parameters, cached under
the build root by a key of the parameters and source paths, and rebuilt only when the content of
a source (RTL, testbench, sw/sim) changed.

Options (see --help): --regimes rand,fast,slow,ordered|all  --rotate P  --seed N  --param K=V
(SPM_RD_LAT=4, FETCH_BEATS=64, RD_OUTSTANDING=2, MVM_WS_FIFO=64, ...)  --has-mvm auto|0|1
--mem key=value (memory model overrides, e.g. rd_lat=400:400)  --procs N (parallel replay
processes per build)  --filelist F (e.g. a scratch copy of rtl/filelist.f with a mutant)
--wno-fatal (Verilator warnings not fatal; elaboration $fatal always is)  --filter / --exclude
SUBSTR  --max-traces N  --timeout-mult X  --reset-each  --check-sim  --waves (VCD-capable build;
then --extra=--vcd=FILE)  --vthreads N (Verilator --threads)  --extra ARG (passed to the binary,
e.g. --extra=--max-cycles=100000)  --json FILE  --build-only  --estimate CYC_PER_S.

Examples:
    tb/fastreplay/run.py traces/m2 --regimes all                      # M2 gate: 172 x 4 runs
    tb/fastreplay/run.py traces/m3 traces/gen --regimes all           # M3 (HAS_MVM auto = 1)
    tb/fastreplay/run.py traces/model --regimes fast,rand --procs 4    # real-model traces
    tb/fastreplay/run.py 'traces/tmp/tbo/0001*' --param MVM_WS_FIFO=64 --mem rd_lat=300:400
    tb/fastreplay/run.py traces/m2/tmax1/m2_s21_0003 --waves --extra=--vcd=/tmp/x.vcd
    <build>/obj/fastreplay --regimes slow --run-seed S TRACE_DIR     # rerun one failing run
Speed (dev server, quiet): U200 M2 core ~0.9 M simulated cycles/s per process (one thread);
--procs scales across cores. --vthreads helps big MVM builds (T_MAX 8), not the M2 core.

HAS_MVM auto: 1 when a trace of the group executes an MVM or faults at one with a code other than
ILLEGAL_OP (or ILLEGAL_OP with reserved MVM fields set: indifferent), 0 when an MVM must be
ILLEGAL_OP (M2 traces); a group needing both is an error (choose --has-mvm, or split the run).

Build root: $TSR_FASTREPLAY_BUILD, else $TSR_SIM_BUILD_ROOT/fastreplay, else build/fastreplay.
make -j$TSR_BUILD_JOBS (default 4). Exit status 0 when every run passed.

Summary: per group traces / runs / failures, simulated cycles, simulation wall time and
simulated cycles per wall second (per process, and aggregate over --procs), slowest trace;
failures with the command line that reproduces one run (--run-seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARNESS_VERSION = 4          # bump when the build recipe changes (part of the build key)
SW_SIM = REPO / "sw" / "sim"
C_SOURCES = ["tsr_ref.c", "tsr_sim.c", "tsr_trace.c"]
TB_SOURCES = [HERE / "fastreplay.cpp", HERE / "memmodel.h"]
REGIMES = ["rand", "fast", "slow", "ordered"]
OP_END, OP_MVM = 0x01, 0x20
ERR_ILLEGAL_OP = 1

# tsr_core parameters that come from the trace configuration (overriding them is an error,
# except RD_OUTSTANDING, which the memory model then checks with the RTL's value).
CONFIG_PARAMS = ("NUM_CH", "BEAT_BITS", "LANES", "T_MAX", "K_MAX", "SPM_BYTES", "VL", "NIREG", "NSREG",
                 "LOOP_DEPTH", "RD_OUTSTANDING", "CH_BYTES", "ADDR_BEAT_W")


# =============================================================================================
# Traces
# =============================================================================================
def _walk(root: Path) -> list:
    if (root / "meta.json").is_file():
        return [root]
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "meta.json" in filenames:
            out.append(Path(dirpath))
            dirnames[:] = []
    return sorted(out)


def find_traces(specs, filters=(), excludes=()) -> list:
    """Trace directories under every spec (directory, tree or glob), in order, deduplicated."""
    import glob as _glob
    out, seen = [], set()
    for spec in specs:
        spec = str(spec)
        roots = sorted(_glob.glob(spec, recursive=True)) if any(c in spec for c in "*?[") else [spec]
        if not roots:
            raise SystemExit("run.py: nothing matches %s" % spec)
        for r in roots:
            r = Path(r)
            if not r.exists():
                raise SystemExit("run.py: %s does not exist" % r)
            for p in _walk(r):
                p = p.resolve()
                s = str(p)
                if p in seen or (filters and not any(f in s for f in filters)) or any(x in s for x in excludes):
                    continue
                seen.add(p)
                out.append(p)
    return out


def insn_words(cmd: bytes, pc: int) -> list:
    return [int.from_bytes(cmd[32 * pc + 4 * i:32 * pc + 4 * i + 4], "little") for i in range(8)]


@dataclass
class TraceInfo:
    path: Path
    meta: dict
    needs_mvm: int | None         # 1 / 0 / None (indifferent)

    @property
    def config(self) -> dict:
        return self.meta["config"]

    @property
    def sim_cycles(self) -> int:
        return int(self.meta.get("perf", {}).get("cycles", 0))


def trace_info(p: Path) -> TraceInfo:
    meta = json.loads((p / "meta.json").read_text())
    count, err, err_pc = int(meta["cmd_count"]), int(meta["err_code"]), int(meta["err_pc"])
    cf = p / "cmd.bin"
    cmd = cf.read_bytes() if cf.is_file() else b""
    need = None
    if len(cmd) >= 32 * count:
        if err:
            executed = min(err_pc, count)
        else:
            executed = count
            for pc in range(count):
                if cmd[32 * pc] == OP_END:
                    executed = pc + 1
                    break
        if any(cmd[32 * pc] == OP_MVM for pc in range(executed)):
            need = 1
        elif err and err_pc < count and cmd[32 * err_pc] == OP_MVM:
            if err != ERR_ILLEGAL_OP:
                need = 1
            else:
                w = insn_words(cmd, err_pc)
                reserved = ((w[0] >> 28) & 0xF) or ((w[0] >> 13) & 7) or (w[5] >> 20) or w[7]
                need = None if reserved else 0
    return TraceInfo(p, meta, need)


def core_params(cfg: dict) -> dict:
    """tsr_core parameters of a tsr_trace.h config dict (tb/core core_params)."""
    return {"NUM_CH": cfg["num_ch"], "BEAT_BITS": cfg["beat_bits"], "LANES": cfg["lanes"],
            "T_MAX": cfg["t_max"], "K_MAX": cfg["k_max"], "SPM_BYTES": cfg["spm_bytes"], "VL": cfg["vl"],
            "NIREG": cfg["nireg"], "NSREG": cfg["nsreg"], "LOOP_DEPTH": cfg["loop_depth"],
            "RD_OUTSTANDING": cfg["rd_outstanding"], "CH_BYTES": cfg["ch_bytes"],
            "ADDR_BEAT_W": cfg["addr_beat_w"]}


def group_name(paths) -> str:
    sets = sorted({p.parent.name for p in paths})
    s = "+".join(sets)
    return s if len(s) <= 48 else s[:40] + "+%d" % len(sets)


# =============================================================================================
# Build
# =============================================================================================
def build_root() -> Path:
    if os.environ.get("TSR_FASTREPLAY_BUILD"):
        return Path(os.environ["TSR_FASTREPLAY_BUILD"])
    if os.environ.get("TSR_SIM_BUILD_ROOT"):
        return Path(os.environ["TSR_SIM_BUILD_ROOT"]) / "fastreplay"
    return REPO / "build" / "fastreplay"


def rtl_sources(filelist: Path) -> list:
    out = []
    for line in filelist.read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if line:
            out.append((filelist.parent / line).resolve())
    return out


def _pval(k, v) -> str:
    if k == "CH_BYTES":
        return "64'd%d" % int(v)
    return str(v)


@dataclass
class Build:
    binary: Path
    desc: str
    params: dict
    has_mvm: int
    built: bool            # False: reused from the cache
    seconds: float


def fr_config_h(cfg: dict, params: dict, desc: str) -> str:
    lines = ["// generated by tb/fastreplay/run.py — configuration of this fastreplay build", "#pragma once"]
    d = dict(FR_NUM_CH=cfg["num_ch"], FR_BEAT_BITS=cfg["beat_bits"], FR_LANES=cfg["lanes"], FR_T_MAX=cfg["t_max"],
             FR_K_MAX=cfg["k_max"], FR_SPM_BYTES="%du" % cfg["spm_bytes"], FR_VL=cfg["vl"], FR_NIREG=cfg["nireg"],
             FR_NSREG=cfg["nsreg"], FR_LOOP_DEPTH=cfg["loop_depth"], FR_CFG_RD_OUTSTANDING=cfg["rd_outstanding"],
             FR_CH_BYTES="%dull" % cfg["ch_bytes"], FR_ADDR_BEAT_W=cfg["addr_beat_w"],
             FR_RD_OUTSTANDING=int(params["RD_OUTSTANDING"]), FR_HAS_MVM=int(params["HAS_MVM"]))
    for k, v in d.items():
        lines.append("#define %s %s" % (k, v))
    lines.append("#define FR_BUILD_DESC %s" % json.dumps(desc))
    return "\n".join(lines) + "\n"


def _sh(cmd, cwd, log: Path, env=None) -> None:
    with open(log, "a") as f:
        f.write("$ %s\n" % " ".join(shlex.quote(str(c)) for c in cmd))
        f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError("command failed (%d), see %s:\n%s" % (r.returncode, log, tail))


def build(cfg: dict, has_mvm: int, overrides: dict, *, filelist: Path, jobs: int = 4, waves: bool = False,
          vthreads: int = 0, wno_fatal: bool = False, log=print) -> Build:
    params = core_params(cfg)
    params["HAS_MVM"] = int(has_mvm)
    for k, v in overrides.items():
        if k in CONFIG_PARAMS and k != "RD_OUTSTANDING":
            raise SystemExit("run.py: %s comes from the trace configuration; it cannot be overridden" % k)
        if k == "HAS_MVM":
            raise SystemExit("run.py: use --has-mvm")
        params[k] = v
    impl = {k: v for k, v in sorted(params.items()) if k not in CONFIG_PARAMS or k == "RD_OUTSTANDING"}
    desc = "NUM_CH=%d T_MAX=%d K_MAX=%d SPM_BYTES=%d CH_BYTES=%d ADDR_BEAT_W=%d %s" % (
        cfg["num_ch"], cfg["t_max"], cfg["k_max"], cfg["spm_bytes"], cfg["ch_bytes"], cfg["addr_beat_w"],
        " ".join("%s=%s" % kv for kv in impl.items()))
    filelist = Path(filelist).resolve()
    srcs = rtl_sources(filelist)
    # Verilator default warnings are fatal (the RTL is lint clean, like the cocotb runner);
    # wno_fatal (mutation campaigns: a mutant may add e.g. a WIDTH warning) keeps them warnings.
    # Elaboration $fatal / $error (parameter checks) always stop the build.
    vflags = ["-O3", "--assert", "--x-assign", "unique", "--x-initial", "unique",
              "-Werror-USERFATAL", "-Werror-USERERROR"]
    if wno_fatal:
        vflags.append("-Wno-fatal")
    if waves:
        vflags += ["--trace", "--trace-structs"]
    if vthreads > 1:
        vflags += ["--threads", str(vthreads)]
    ident = dict(v=HARNESS_VERSION, params=sorted((k, str(v)) for k, v in params.items()), filelist=str(filelist),
                 srcs=[str(s) for s in srcs], vflags=vflags)
    key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    name = "c%d_t%d_m%d" % (cfg["num_ch"], cfg["t_max"], has_mvm)
    bdir = build_root() / ("%s_%s" % (name, key))
    binary = bdir / "obj" / "fastreplay"
    cfg_h = fr_config_h(cfg, params, desc)
    h = hashlib.sha256()
    try:
        vver = subprocess.run(["verilator", "--version"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        raise SystemExit("run.py: verilator not found on PATH")
    h.update(vver.encode())
    h.update(cfg_h.encode())
    h.update(json.dumps(ident, sort_keys=True).encode())
    hdrs = sorted((SW_SIM / "include").glob("*.h"))
    for f in list(srcs) + TB_SOURCES + [SW_SIM / c for c in C_SOURCES] + hdrs:
        h.update(str(f).encode())
        h.update(Path(f).read_bytes())
    stamp = h.hexdigest()
    stamp_file = bdir / "stamp"
    if binary.is_file() and stamp_file.is_file() and stamp_file.read_text() == stamp:
        return Build(binary, desc, params, has_mvm, False, 0.0)
    t0 = time.time()
    bdir.mkdir(parents=True, exist_ok=True)
    if stamp_file.exists():
        stamp_file.unlink()
    blog = bdir / "build.log"
    blog.write_text("")
    log("build %s -> %s" % (desc, bdir))
    (bdir / "fr_config.h").write_text(cfg_h)
    # sw/sim C sources (trace parsing, ISA simulator for --check-sim, reference library), compiled
    # privately so that a concurrent rebuild of build/dev cannot change this build.
    swo = bdir / "sw"
    swo.mkdir(exist_ok=True)
    cc = os.environ.get("CC", "gcc")
    objs = []
    for c in C_SOURCES:
        o = swo / (Path(c).stem + ".o")
        _sh([cc, "-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I", SW_SIM / "include", "-c",
             SW_SIM / c, "-o", o], bdir, blog)
        objs.append(o)
    lib = bdir / "libtsrsw.a"
    if lib.exists():
        lib.unlink()
    _sh(["ar", "rcs", lib] + objs, bdir, blog)
    cmd = ["verilator", "--cc", "--exe", "--build", "-j", str(jobs), "--top-module", "tsr_core",
           "--Mdir", bdir / "obj", "-o", "fastreplay"] + vflags
    cmd += ["-G%s=%s" % (k, _pval(k, v)) for k, v in sorted(params.items())]
    cmd += ["-CFLAGS", "-std=gnu++17 -I%s -I%s -I%s -DVL_USER_STOP" % (bdir, HERE, SW_SIM / "include"),
            "-LDFLAGS", "%s -lpthread -lm" % lib,
            "-MAKEFLAGS", "OPT_FAST=-O2", "-MAKEFLAGS", "OPT_SLOW=-O1", "-MAKEFLAGS", "OPT_GLOBAL=-O2"]
    cmd += srcs + [HERE / "fastreplay.cpp"]
    _sh(cmd, bdir, blog)
    if not binary.is_file():
        raise RuntimeError("verilator did not produce %s (see %s)" % (binary, blog))
    for gch in (bdir / "obj").glob("*.gch"):     # precompiled headers: ~70 MB each, rebuilt on demand
        gch.unlink()
    stamp_file.write_text(stamp)
    dt = time.time() - t0
    log("built in %.0f s" % dt)
    return Build(binary, desc, params, has_mvm, True, dt)


# =============================================================================================
# Run
# =============================================================================================
@dataclass
class GroupReport:
    name: str
    desc: str
    traces: int = 0
    runs: int = 0
    failed: int = 0
    crashed: int = 0
    sim_cycles: int = 0          # every simulated cycle (CSR traffic included)
    run_cycles: int = 0          # START .. done
    sim_wall: float = 0.0        # seconds inside runs, summed over processes
    wall: float = 0.0            # elapsed seconds of the replay (all processes)
    build_s: float = 0.0
    slowest: tuple = (0.0, "")
    failures: list = field(default_factory=list)
    results: list = field(default_factory=list)

    @property
    def cps(self) -> float:      # per process
        return self.sim_cycles / self.sim_wall if self.sim_wall else 0.0

    @property
    def cps_total(self) -> float:
        return self.sim_cycles / self.wall if self.wall else 0.0


@dataclass
class Report:
    groups: list = field(default_factory=list)

    @property
    def runs(self):
        return sum(g.runs for g in self.groups)

    @property
    def failed(self):
        return sum(g.failed for g in self.groups)

    @property
    def ok(self):
        return self.failed == 0 and self.runs > 0

    def to_json(self) -> dict:
        return dict(ok=self.ok, runs=self.runs, failed=self.failed, groups=[dict(
            name=g.name, build=g.desc, traces=g.traces, runs=g.runs, failed=g.failed, crashed=g.crashed,
            sim_cycles=g.sim_cycles, run_cycles=g.run_cycles, sim_wall=g.sim_wall, wall=g.wall,
            cycles_per_s=g.cps, cycles_per_s_total=g.cps_total, build_s=g.build_s, slowest=list(g.slowest),
            failures=g.failures) for g in self.groups])


def _split(infos, n):
    """n balanced lists (greedy on the simulator's cycle estimate), each in the original order."""
    if n <= 1:
        return [list(infos)]
    order = sorted(range(len(infos)), key=lambda i: -max(infos[i].sim_cycles, 1000))
    load, parts = [0] * n, [[] for _ in range(n)]
    for i in order:
        j = load.index(min(load))
        parts[j].append(i)
        load[j] += max(infos[i].sim_cycles, 1000) + 2000
    return [[infos[i] for i in sorted(p)] for p in parts if p]


def _pump(proc, log_path: Path, echo):
    with open(log_path, "a") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if echo is not None:
                echo(line.rstrip("\n"))


def run_group(b: Build, infos, *, regimes, rotate, seed, mem, procs, timeout_mult, reset_each, check_sim,
              stop_on_fail, verbose, extra, run_dir: Path, log=print, echo=None) -> GroupReport:
    gname = group_name([i.path for i in infos])
    g = GroupReport(gname, b.desc, traces=len(infos), build_s=b.seconds)
    per_trace = rotate if rotate else len(regimes)
    run_dir.mkdir(parents=True, exist_ok=True)
    common = ["--regimes", ",".join(regimes), "--seed", str(seed), "--timeout-mult", str(timeout_mult)]
    if rotate:
        common += ["--rotate", str(rotate)]
    for kv in mem:
        common += ["--mem", kv]
    if reset_each:
        common.append("--reset-each")
    if check_sim:
        common.append("--check-sim")
    if stop_on_fail:
        common.append("--stop-on-fail")
    if verbose:
        common.append("-v")
    common += list(extra)
    t0 = time.time()
    pending = [list(p) for p in _split(infos, procs)]
    results = []
    attempt = 0
    while pending:
        attempt += 1
        running = []
        for i, part in enumerate(pending):
            tag = "%s:%d.%d" % (gname[:24], i, attempt)
            lst = run_dir / ("list_%d_%d.txt" % (i, attempt))
            lst.write_text("".join("%s\n" % t.path for t in part))
            out = run_dir / ("out_%d_%d.jsonl" % (i, attempt))
            logf = run_dir / ("log_%d_%d.txt" % (i, attempt))
            cmd = [str(b.binary), "--list", str(lst), "--out", str(out), "--tag", tag] + common
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            th = threading.Thread(target=_pump, args=(p, logf, echo), daemon=True)
            th.start()
            running.append((p, th, part, out, logf))
        nxt = []
        for p, th, part, out, logf in running:
            rc = p.wait()
            th.join()
            got = []
            fatal = None
            if out.is_file():
                for line in out.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "trace" in j:
                        j["log"] = str(logf)
                        got.append(j)
                    elif "fatal" in j:
                        fatal = j["fatal"]
            results += got
            if fatal is not None:
                raise RuntimeError("fastreplay %s: %s (log %s)" % (b.binary, fatal, logf))
            if rc in (0, 1) or stop_on_fail:
                continue
            # crashed: the first trace without all its runs is the culprit; resume after it
            counts = {}
            for j in got:
                counts[j["trace"]] = counts.get(j["trace"], 0) + 1
            for k, t in enumerate(part):
                if counts.get(str(t.path), 0) < per_trace:
                    tail = "\n".join(logf.read_text(errors="replace").splitlines()[-12:])
                    results.append(dict(trace=str(t.path), regime="?", seed=-1, ok=False, cycles=0, total_cycles=0,
                                        wall=0.0, crashed=True, log=str(logf),
                                        errors=["replay process died (exit %d) on this trace; log tail:\n%s"
                                                % (rc, tail)]))
                    g.crashed += 1
                    if part[k + 1:] and attempt < 50:
                        nxt.append(part[k + 1:])
                    break
        pending = nxt
    g.wall = time.time() - t0
    for j in results:
        g.runs += 1
        g.sim_cycles += int(j.get("total_cycles", 0))
        g.run_cycles += int(j.get("cycles", 0))
        g.sim_wall += float(j.get("wall", 0.0))
        if float(j.get("wall", 0.0)) > g.slowest[0]:
            g.slowest = (float(j["wall"]), "%s (%s, %d cycles)" % (j["trace"], j.get("regime"), j.get("cycles", 0)))
        if not j.get("ok"):
            g.failed += 1
            g.failures.append(dict(trace=j["trace"], regime=j.get("regime"), seed=j.get("seed"),
                                   errors=j.get("errors", []), log=j.get("log"),
                                   repro=("%s %s %s" % (b.binary, " ".join(common), j["trace"])) if j.get("crashed")
                                   else "%s --regimes %s --run-seed %s %s" % (
                                       b.binary, j.get("regime"), j.get("seed"), j["trace"])))
    g.results = results
    return g


def replay(specs, *, regimes=("rand",), rotate=0, seed=1, has_mvm="auto", params=None, mem=(), procs=1,
           filelist=None, jobs=None, timeout_mult=1.0, reset_each=False, check_sim=False, waves=False,
           vthreads=0, stop_on_fail=False, verbose=False, filters=(), excludes=(), max_traces=0,
           build_only=False, extra=(), wno_fatal=False, log=print, echo=print) -> Report:
    regimes = list(REGIMES) if "all" in regimes else list(regimes)
    for r in regimes:
        if r not in REGIMES:
            raise SystemExit("run.py: unknown regime %s (%s)" % (r, ", ".join(REGIMES)))
    filelist = Path(filelist or REPO / "rtl" / "filelist.f")
    jobs = int(jobs or os.environ.get("TSR_BUILD_JOBS", "4"))
    paths = find_traces(specs, filters, excludes)
    if max_traces:
        step = max(1, len(paths) // max_traces)
        paths = paths[::step][:max_traces]
    if not paths:
        raise SystemExit("run.py: no trace directory found")
    groups = {}
    for p in paths:
        ti = trace_info(p)
        groups.setdefault(json.dumps(ti.config, sort_keys=True), []).append(ti)
    rep = Report()
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-%d" % os.getpid()
    for gi, (key, infos) in enumerate(groups.items()):
        cfg = json.loads(key)
        if has_mvm == "auto":
            needs = {t.needs_mvm for t in infos} - {None}
            if len(needs) > 1:
                need1 = [t.path for t in infos if t.needs_mvm == 1][:2]
                need0 = [t.path for t in infos if t.needs_mvm == 0][:2]
                raise SystemExit("run.py: configuration group %s mixes traces that need HAS_MVM=1 (%s) and "
                                 "HAS_MVM=0 (%s); pass --has-mvm or run them separately"
                                 % (group_name([t.path for t in infos]), need1, need0))
            hm = needs.pop() if needs else 0
        else:
            hm = int(has_mvm)
        b = build(cfg, hm, dict(params or {}), filelist=filelist, jobs=jobs, waves=waves, vthreads=vthreads,
                  wno_fatal=wno_fatal, log=log)
        if build_only:
            rep.groups.append(GroupReport(group_name([t.path for t in infos]), b.desc, traces=len(infos),
                                          build_s=b.seconds))
            continue
        run_dir = build_root() / "runs" / stamp / ("g%d" % gi)
        g = run_group(b, infos, regimes=regimes, rotate=rotate, seed=seed, mem=mem, procs=procs,
                      timeout_mult=timeout_mult, reset_each=reset_each, check_sim=check_sim,
                      stop_on_fail=stop_on_fail, verbose=verbose, extra=extra, run_dir=run_dir, log=log, echo=echo)
        rep.groups.append(g)
        log(format_group(g))
        if stop_on_fail and g.failed:
            break
    return rep


def format_group(g: GroupReport) -> str:
    s = ("%-28s %4d traces %5d runs %4d failed%s | %11d cycles in %7.1f s run wall: %.3g cycles/s per process, "
         "%.3g aggregate | build %.0f s | %s" % (
             g.name, g.traces, g.runs, g.failed, (" (%d crashed)" % g.crashed) if g.crashed else "", g.sim_cycles,
             g.wall, g.cps, g.cps_total, g.build_s, g.desc))
    if g.slowest[1]:
        s += "\n%28s slowest run %.2f s: %s" % ("", g.slowest[0], g.slowest[1])
    for f in g.failures[:20]:
        s += "\n  FAIL %s [%s seed %s]\n       %s\n       repro: %s\n       log: %s" % (
            f["trace"], f["regime"], f["seed"], "\n       ".join(f["errors"][:4]), f["repro"], f.get("log"))
    if len(g.failures) > 20:
        s += "\n  ... %d more failures" % (len(g.failures) - 20)
    return s


# =============================================================================================
# Estimate (no simulation): replay time of trace sets at a given simulation speed
# =============================================================================================
def estimate(specs, cps: float, factors: dict, overhead: int = 400, filters=(), excludes=()) -> str:
    """Predicted replay time per set and regime: RTL cycles ~ factor[regime] x the simulator's
    cycle model (meta perf.cycles, 1 weight beat per cycle per cluster) + `overhead` CSR cycles
    per run; time = cycles / cps (simulated cycles per second of one process)."""
    paths = find_traces(specs, filters, excludes)
    sets = {}
    for p in paths:
        m = json.loads((p / "meta.json").read_text())
        s = sets.setdefault(str(p.parent.relative_to(REPO) if REPO in p.parents else p.parent), [0, 0, 0])
        s[0] += 1
        s[1] += int(m.get("perf", {}).get("cycles", 0))
        s[2] = max(s[2], int(m.get("perf", {}).get("cycles", 0)))
    out = ["%-40s %6s %13s %10s  %s" % ("set", "traces", "sim cycles", "max", "  ".join(
        "%9s" % r for r in factors))]
    tot = {r: 0.0 for r in factors}
    for name, (n, cyc, mx) in sorted(sets.items()):
        row = []
        for r, f in factors.items():
            t = (cyc * f + n * overhead) / cps
            tot[r] += t
            row.append("%8.0fs" % t)
        out.append("%-40s %6d %13d %10d  %s" % (name, n, cyc, mx, "  ".join(row)))
    out.append("%-40s %6s %13s %10s  %s" % ("total (one process)", "", "", "", "  ".join(
        "%8.0fs" % tot[r] for r in factors)))
    return "\n".join(out)


# =============================================================================================
# CLI
# =============================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", default=None, help="trace directories, trees or globs (default traces/m2)")
    ap.add_argument("--regimes", default="rand", help="comma list of rand,fast,slow,ordered or 'all' (default rand)")
    ap.add_argument("--rotate", type=int, default=0, help="P passes per trace, regimes rotating (tb/core style)")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=int(os.environ.get("TSR_SEED", "1"), 0))
    ap.add_argument("--has-mvm", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--param", action="append", default=[], metavar="K=V", help="tsr_core parameter override")
    ap.add_argument("--mem", action="append", default=[], metavar="K=V", help="memory model override")
    ap.add_argument("--procs", type=int, default=1, help="parallel replay processes per build (default 1)")
    ap.add_argument("--filelist", default=None, help="RTL filelist (default rtl/filelist.f)")
    ap.add_argument("--jobs", type=int, default=None, help="make -j for builds (default $TSR_BUILD_JOBS or 4)")
    ap.add_argument("--filter", action="append", default=[], help="only traces whose path contains this")
    ap.add_argument("--exclude", action="append", default=[], help="skip traces whose path contains this")
    ap.add_argument("--max-traces", type=int, default=0, help="evenly spaced subset of at most N traces")
    ap.add_argument("--timeout-mult", type=float, default=1.0)
    ap.add_argument("--reset-each", action="store_true", help="hardware reset before every trace")
    ap.add_argument("--check-sim", action="store_true", help="also replay every trace on the ISA simulator")
    ap.add_argument("--waves", action="store_true", help="VCD-capable build (pass --extra=--vcd=FILE)")
    ap.add_argument("--extra", action="append", default=[], help="extra fastreplay argument (e.g. --extra=--vcd=x.vcd)")
    ap.add_argument("--vthreads", type=int, default=0, help="Verilator --threads")
    ap.add_argument("--wno-fatal", action="store_true", help="Verilator warnings are not fatal (mutants)")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="no per-process output")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--estimate", type=float, default=0.0, metavar="CYC_PER_S",
                    help="no simulation: predicted replay time at this many simulated cycles/s")
    ap.add_argument("--factors", default="fast=1.05,rand=1.6,slow=3.5",
                    help="--estimate: RTL cycles / simulator perf cycles per regime")
    a = ap.parse_args(argv)
    specs = a.traces or [str(REPO / "traces" / "m2")]
    if a.estimate:
        factors = {k: float(v) for k, v in (kv.split("=") for kv in a.factors.split(","))}
        print(estimate(specs, a.estimate, factors, filters=a.filter, excludes=a.exclude))
        return 0
    params = {}
    for kv in a.param:
        k, _, v = kv.partition("=")
        params[k.strip()] = int(v, 0) if v.strip().lstrip("-").isdigit() or v.startswith("0x") else v
    extra = []
    for e in a.extra:
        extra += e.split("=", 1) if e.startswith("--") and "=" in e else [e]
    t0 = time.time()
    try:
        rep = replay(specs, regimes=a.regimes.split(","), rotate=a.rotate, seed=a.seed, has_mvm=a.has_mvm,
                     params=params, mem=a.mem, procs=a.procs, filelist=a.filelist, jobs=a.jobs,
                     timeout_mult=a.timeout_mult, reset_each=a.reset_each, check_sim=a.check_sim, waves=a.waves,
                     vthreads=a.vthreads, stop_on_fail=a.stop_on_fail, verbose=a.verbose, filters=a.filter,
                     excludes=a.exclude, max_traces=a.max_traces, build_only=a.build_only, extra=extra,
                     wno_fatal=a.wno_fatal, echo=None if a.quiet else print)
    except RuntimeError as e:          # build failure (Verilator / compiler output in the message)
        print("run.py: %s" % e)
        return 2
    if a.json:
        Path(a.json).write_text(json.dumps(rep.to_json(), indent=1))
    if not a.build_only:
        sc = sum(g.sim_cycles for g in rep.groups)
        sw = sum(g.sim_wall for g in rep.groups)
        print("TOTAL: %d traces, %d runs, %d failed | %d simulated cycles, %.1f s replay wall, %.3g cycles/s per "
              "process | %.0f s elapsed incl. builds -> %s" % (
                  sum(g.traces for g in rep.groups), rep.runs, rep.failed, sc,
                  sum(g.wall for g in rep.groups), sc / sw if sw else 0.0, time.time() - t0,
                  "PASS" if rep.ok else "FAIL"))
        return 0 if rep.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

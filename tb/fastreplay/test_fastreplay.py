"""pytest entry of the fast RTL trace replayer (tb/fastreplay; Verilator C++ testbench, no cocotb).

  test_memmodel_selftest          memmodel.h against selftest.cpp (legal traffic, every protocol
                                  assertion P1..P6 fires on its violation, ordering, tracking)
  test_fastreplay_m2[<sets>]      every traces/m2 trace of one configuration on the M2 core
                                  (HAS_MVM = 0) under all four memory regimes (rand, fast, slow,
                                  ordered): STATUS / ERR_*, expect data, written set, read set,
                                  PERF deltas, per-source beats, quiescence, RTL assertions
  test_fastreplay_impl            implementation parameters that must not change results
                                  (SPM_RD_LAT 1 / 4, FETCH_BEATS 2 / 64, RD_OUTSTANDING 1 / 2)
  test_fastreplay_detects_corrupt corrupted copies of one trace (expect byte, final state,
                                  ERR_INFO, PERF retired / read beats, missing init region, extra /
                                  missing expect region) must each fail with the right message;
                                  a forced hang (--max-cycles) fails and the next trace still
                                  passes (hardware-reset recovery); --check-sim passes;
                                  --derive-only fails on the corrupted perf record only
  test_fastreplay_detects_mutants M2 RTL mutants in scratch copies (QACT tail not zeroed, arbiter
                                  RD_OUTSTANDING + 1, ST done before its write responses, write
                                  request waiting for data ready) must fail (slow, opt-in: -m slow,
                                  TSR_FASTREPLAY_SLOW=1, -k test_fastreplay_detects_mutants or its
                                  node id)
Builds go to $TSR_FASTREPLAY_BUILD / $TSR_SIM_BUILD_ROOT/fastreplay / build/fastreplay (cached).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
fr = _load("fastreplay_run", HERE / "run.py")
sys.modules.setdefault("run", fr)          # mutate.py imports run
mut = _load("fastreplay_mutate", HERE / "mutate.py")

M2 = REPO / "traces" / "m2"


def _groups():
    groups = {}
    for p in fr.find_traces([M2]):
        cfg = json.loads((p / "meta.json").read_text())["config"]
        groups.setdefault(json.dumps(cfg, sort_keys=True), []).append(p)
    return [(fr.group_name(ps), [str(p) for p in ps]) for ps in groups.values()]


GROUPS = _groups()


def _log(*a):
    print(*a)


def test_memmodel_selftest(tmp_path):
    exe = tmp_path / "selftest"
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O1", "-Wall", "-Wextra", "-I", str(HERE),
                    str(HERE / "selftest.cpp"), "-o", str(exe)], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    assert r.returncode == 0, r.stdout
    assert "0 check(s) failed" in r.stdout


@pytest.mark.traces
@pytest.mark.parametrize("name,paths", GROUPS, ids=[g[0] for g in GROUPS])
def test_fastreplay_m2(name, paths):
    rep = fr.replay(paths, regimes=["all"], has_mvm="0", log=_log, echo=None)
    g = rep.groups[0]
    print(fr.format_group(g))
    assert g.runs == 4 * len(paths)
    assert g.failed == 0, [f["errors"][:2] for f in g.failures[:5]]
    regs = {j["regime"] for j in g.results}
    assert regs == set(fr.REGIMES)


@pytest.mark.traces
@pytest.mark.parametrize("sets,params", [
    ("small1ch", dict(SPM_RD_LAT=1, FETCH_BEATS=2, RD_OUTSTANDING=1)),
    ("ch3", dict(SPM_RD_LAT=4, FETCH_BEATS=64, RD_OUTSTANDING=2)),
], ids=["lat1_fb2_ro1", "lat4_fb64_ro2"])
def test_fastreplay_impl(sets, params):
    rep = fr.replay([M2 / sets], regimes=["all"], has_mvm="0", params=params, log=_log, echo=None)
    g = rep.groups[0]
    print(fr.format_group(g))
    assert g.runs > 0 and g.failed == 0, [f["errors"][:2] for f in g.failures[:5]]


# ---------------------------------------------------------------------------------------------
# The harness must fail on wrong results: corrupted copies of a trace
# ---------------------------------------------------------------------------------------------
BASE = M2 / "tmax1" / "m2_s21_0003"   # LD ch1 + ch3, ST ch0 + ch3, QACT; status DONE
BIG = M2 / "tmax1" / "m2_s21_0036"    # the longest tmax1 trace (~30 K cycles in slow)


def _copy(tmp: Path, name: str, src: Path = BASE) -> Path:
    d = tmp / "set" / name
    shutil.copytree(src, d)
    return d


def _meta(d: Path, fn):
    m = json.loads((d / "meta.json").read_text())
    fn(m)
    (d / "meta.json").write_text(json.dumps(m))


def _corrupt(tmp: Path) -> dict:
    """{trace dir: expected error substring} (None: must pass)."""
    out = {_copy(tmp, "orig"): None}
    d = _copy(tmp, "expect_byte")
    m = json.loads((d / "meta.json").read_text())
    f = d / m["expect"][0]["file"]
    b = bytearray(f.read_bytes())
    b[100] ^= 0x01
    f.write_bytes(bytes(b))
    out[d] = "expect mismatch"
    d = _copy(tmp, "status")
    _meta(d, lambda m: m.update(status=6, err_code=5, err_pc=2))
    out[d] = "STATUS ="
    d = _copy(tmp, "err_info")
    _meta(d, lambda m: m.update(err_info=7))
    out[d] = "ERR_INFO ="
    d = _copy(tmp, "retired")
    _meta(d, lambda m: m["perf"].update(retired=m["perf"]["retired"] + 1))
    out[d] = "PERF_RETIRED"
    d = _copy(tmp, "rd_beats")
    _meta(d, lambda m: m["perf"]["ch_rd_beats"].__setitem__(1, m["perf"]["ch_rd_beats"][1] + 1))
    out[d] = "PERF_CH_RD_BEATS[1]"

    def drop_init_ch1(m):
        r = [x for x in m["init"] if x["ch"] == 1][0]
        m["init"].remove(r)
        (d / r["file"]).unlink()
    d = _copy(tmp, "no_init")
    _meta(d, drop_init_ch1)
    out[d] = "does not initialise"

    def add_expect(m):
        (d / "expect" / "ch2_0000000000.bin").write_bytes(bytes(64))
        m["expect"].append(dict(ch=2, offset=0, size=64, file="expect/ch2_0000000000.bin"))
    d = _copy(tmp, "extra_expect")
    _meta(d, add_expect)
    out[d] = "not written"

    def drop_expect(m):
        r = [x for x in m["expect"] if x["ch"] == 3][-1]
        m["expect"].remove(r)
        (d / r["file"]).unlink()
    d = _copy(tmp, "drop_expect")
    _meta(d, drop_expect)
    out[d] = "outside the expect regions"
    return out


@pytest.mark.traces
def test_fastreplay_detects_corrupt(tmp_path):
    cases = _corrupt(tmp_path)
    rep = fr.replay([tmp_path / "set"], regimes=["rand", "slow"], has_mvm="0", log=_log, echo=None)
    g = rep.groups[0]
    by = {}
    for j in g.results:
        by.setdefault(Path(j["trace"]).name, []).append(j)
    for d, want in cases.items():
        runs = by[d.name]
        assert len(runs) == 2, d.name
        for j in runs:
            if want is None:
                assert j["ok"], (d.name, j["errors"])
            else:
                assert not j["ok"], "%s passed although corrupted" % d.name
                assert any(want in e for e in j["errors"]), (d.name, want, j["errors"])
    # the ISA simulator replays the original inside the same binary (--check-sim)
    rep = fr.replay([tmp_path / "set" / "orig"], regimes=["fast"], has_mvm="0", check_sim=True, log=_log, echo=None)
    assert rep.ok
    # --derive-only (no simulation): program-derived LD + MVM beats vs the perf record
    rep = fr.replay([tmp_path / "set" / "orig", tmp_path / "set" / "rd_beats"], has_mvm="0",
                    extra=["--derive-only"], log=_log, echo=None)
    res = {Path(j["trace"]).name: j for j in rep.groups[0].results}
    assert res["orig"]["ok"] and not res["rd_beats"]["ok"]
    assert any("derived" in e for e in res["rd_beats"]["errors"]), res["rd_beats"]["errors"]
    # forced hang: the long trace times out, the bench resets the core and the next trace passes
    hang = tmp_path / "hang"
    shutil.copytree(BIG, hang / "a_big")
    shutil.copytree(BASE, hang / "b_small")
    rep = fr.replay([hang], regimes=["slow"], has_mvm="0", extra=["--max-cycles", "6000"], log=_log, echo=None)
    res = {Path(j["trace"]).name: j for j in rep.groups[0].results}
    assert not res["a_big"]["ok"] and any("no done within 6000 cycles" in e for e in res["a_big"]["errors"])
    assert res["b_small"]["ok"], res["b_small"]["errors"]


def _require_slow(request):
    """Slow tests are opt-in by tb/fp's rule (tb/fp/test_fp.py slow_requested: an -m expression
    naming `slow`, -k with the test's name, its node id) or TSR_FASTREPLAY_SLOW=1; a plain
    `pytest tb` / tb/run_all.sh reports them skipped (tb/pytest.ini does not deselect `slow`)."""
    fp = sys.modules.get("tsr_fp_slow_gate") or _load("tsr_fp_slow_gate", REPO / "tb" / "fp" / "test_fp.py")
    if not fp.slow_requested(request.config, request.node.originalname,
                             env={fp.SLOW_ENV: os.environ.get("TSR_FASTREPLAY_SLOW", "0")}):
        pytest.skip("slow test, opt-in: -m slow, TSR_FASTREPLAY_SLOW=1, -k %s or its node id"
                    % request.node.originalname)


@pytest.mark.slow
@pytest.mark.traces
def test_fastreplay_detects_mutants(request, tmp_path):
    _require_slow(request)
    names = {"qact_qs_tail", "arb_outstanding_plus1", "dma_st_done_early", "dma_wr_req_waits_dat_ready"}
    res = mut.rtl_campaign(tmp_path, names, log=_log)
    assert {r[0] for r in res} == names
    surv = [r for r in res if not r[1].startswith("killed")]
    assert not surv, surv

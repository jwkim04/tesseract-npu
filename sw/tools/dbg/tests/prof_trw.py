#!/usr/bin/env python3
"""prof_trw.py — tsr-prof on the sim kind (docs/design/05-debug.md "DB" §8.3, §8.6, §8.7).

1. tsr-prof trw: two rounds of rt_copy (LD + ST) and rt_nop with the trace writer on, --check (the
   DB §8.7 self-checks must pass) and --stats (the DB §8.6 summary). The JSON timeline is checked:
   the process name, one LD and one ST slice per rt_copy on the DMA thread with RUN as the duration,
   four launch slices (B / E pairs on the RISC-V thread), per-channel counter tracks.
2. tsr-prof trw --from-raw of the saved records gives the same slices.
3. tsr-prof stats prints the summary alone; DBG_STOP_CYCLES is 0.
"""
import argparse
import json
import os
import subprocess
import sys

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
    print("$ " + " ".join(cmd) + "\n" + p.stdout)
    return p.returncode, p.stdout


def slices(js):
    ev = json.load(open(js))["traceEvents"]
    return ev, [e for e in ev if e.get("ph") == "X" and e.get("tid") == 0 and not e["name"].startswith("waiting")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof", required=True)
    ap.add_argument("--elf", required=True)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    js, raw, js2 = (os.path.join(a.scratch, n) for n in ("trace.json", "ring.bin", "trace_raw.json"))
    copy = "rt_copy:0x100000,0x100100,4,0"
    rc, out = run([a.prof, "trw", "--device", "sim", "--image", a.elf, "--launch", copy, "--launch", "rt_nop", "--repeat", "2",
                   "--ring-mib", "1", "--out", js, "--raw", raw, "--check", "--stats"])
    check(rc == 0, "tsr-prof trw exits 0")
    check("ok: RETIRED" in out and "FAIL" not in out, "the DB §8.7 self-checks pass")
    check("PERF (DB §8.6)" in out and "DBG_STOP_CYCLES 0;" in out and "read GB/s and idle %" in out, "the DB §8.6 summary")
    ev, dma = slices(js)
    check(any(e.get("ph") == "M" and e.get("args", {}).get("name") == "Tesseract0" for e in ev), "process Tesseract0")
    names = [e["name"] for e in dma]
    check(names.count("LD") == 2 and names.count("ST") == 2, "two LD and two ST slices on the DMA thread: %s" % names)
    check(all(e["dur"] > 0 and e["args"]["run"] > 0 for e in dma), "RUN as the slice duration")
    begins = [e for e in ev if e.get("ph") == "B" and e["name"].startswith("launch ")]
    ends = [e for e in ev if e.get("ph") == "E" and e["name"].startswith("launch ")]
    check(len(begins) == 4 and len(ends) == 4, "four launch slices (%d B, %d E)" % (len(begins), len(ends)))
    check(any("rt_copy" in e["name"] for e in begins) and any("rt_nop" in e["name"] for e in begins), "launch slices named by kernel")
    check(any(e.get("ph") == "C" and e["name"] == "ch0 read" for e in ev), "per-channel counter tracks")
    # Markers are stamped at dispatch in push order (DB §8.4): an op of a launch has a push index in
    # [LAUNCH_BEGIN.next_pidx, LAUNCH_END.next_pidx) and starts after LAUNCH_BEGIN (it may end after
    # LAUNCH_END, which is dispatched while the last ops run).
    for s in dma:
        p = s["args"]["pidx"]
        check(any(b["args"]["next_pidx"] <= p < e["args"]["next_pidx"] and b["ts"] <= s["ts"] for b, e in zip(begins, ends)),
              "%s pidx %d in its launch's push-index window, started after LAUNCH_BEGIN" % (s["name"], p))
    rc, out = run([a.prof, "trw", "--from-raw", raw, "--image", a.elf, "--out", js2])
    check(rc == 0, "tsr-prof trw --from-raw exits 0")
    _, dma2 = slices(js2)
    check([(e["name"], e["ts"], e["dur"]) for e in dma2] == [(e["name"], e["ts"], e["dur"]) for e in dma], "--from-raw gives the same slices")
    rc, out = run([a.prof, "stats", "--device", "sim", "--image", a.elf, "--launch", copy, "--repeat", "3"])
    check(rc == 0 and "3 launches" in out and "DBG_STOP_CYCLES 0;" in out, "tsr-prof stats")
    print("%s: %d failure(s)" % ("FAIL" if FAILS else "PASS", len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

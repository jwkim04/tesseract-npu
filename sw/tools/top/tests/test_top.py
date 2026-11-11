#!/usr/bin/env python3
# test_top.py — tests of tsr-top (docs/prof-dbg.md §7, SW-87); python standard library only.
#
#   layout  TEST_TSR_MON         the record layout of tsr_top.py is the one the C header has
#   synth                        a record written here (this process as the publisher): the reader, the rates, the
#                                COMPUTE, MEMORY and MEM BW meters, the two lines of the time series box and the U split against
#                                hand-computed numbers, the text frame at several sizes, a record that changes while tsr-top
#                                samples it, stale and torn files, and a made-up sysfs tree (TSR_TOP_SYSFS) for the clocks, fan,
#                                temperature and power, with xbutil as the fallback
#   live    LLAMA_COMPLETION MODEL TSR_MON_PROBE
#                                a tiny model on the simulator with the publisher on: the record while it runs
#                                (never torn, counters never go back), the record it leaves with TSR_MON_KEEP,
#                                tokens, model name, the JSON snapshot, TSR_MON=0 makes no file, the probe on sim
#   identity KLCHECK MODEL PROMPTS
#                                klcheck 1 prompt x 16 tokens on the simulator with TSR_MON=0 and with the publisher on,
#                                M6 and M7: every logits row bit-identical, and the second run did publish
#   curses  TMUX                 the curses screen in a pseudo-terminal (tmux): the frame, F2, F6, +/-, resize, q
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.join(os.path.dirname(HERE), "tsr_top.py")
sys.path.insert(0, os.path.dirname(HERE))
import tsr_top  # noqa: E402

failed = 0
checks = 0


def check(cond, msg):
    global failed, checks
    checks += 1
    if not cond:
        failed += 1
        print("FAIL: " + msg)


def run_top(args, env=None, timeout=60):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([sys.executable, TOP] + args, capture_output=True, text=True, timeout=timeout, env=e)
    return r


def make_page(**kw):
    """A page of layout 1 with the given fields (the rest zero)."""
    page = bytearray(tsr_top.PAGE)
    page[0:8] = b"TSRMON\0\0"
    vals = dict(layout=1, size=tsr_top.PAGE, used=tsr_top.USED)
    vals.update(kw)
    for name, kind, count, off in tsr_top.TABLE:
        if name not in vals or name == "magic":
            continue
        v = vals[name]
        if kind == "c":
            b = v.encode()[:count - 1]
            page[off:off + len(b)] = b
        elif kind == "B":
            for i, x in enumerate(v):
                page[off + i] = x
        else:
            if count == 1:
                struct.pack_into("<" + kind, page, off, v)
            else:
                struct.pack_into("<%d%s" % (count, kind), page, off, *(list(v) + [0] * (count - len(v))))
    return bytes(page)


def write_record(path, page):
    tmp = path + ".w"
    with open(tmp, "wb") as f:
        f.write(page)
    os.replace(tmp, path)


def my_ticks():
    return tsr_top.proc_start_ticks(os.getpid())


def synth_perf(k, hz, ch=(0.8, 0.8, 0.8, 0.8)):
    """The PERF array after k intervals of 0.1 s of a device that is active 60 % of the time: the MVM holds an op in
    90 % of the active cycles, the VU 10 %, DMA 20 %, QACT 5 %; channel c streams in the share ch[c] of them (80 %)."""
    C, X = 4, 13
    perf = [0] * 64
    d_cyc = int(0.06 * hz)
    perf[tsr_top.P_CYCLES] = k * d_cyc
    perf[tsr_top.P_MVM] = int(k * d_cyc * 0.9)
    perf[tsr_top.P_VU] = int(k * d_cyc * 0.1)
    for c in range(C):
        perf[5 + c] = int(k * d_cyc * ch[c])        # read beats
        perf[5 + C + c] = int(k * d_cyc * 0.01)     # wait
    perf[X + tsr_top.X_DMA] = int(k * d_cyc * 0.2)
    perf[X + tsr_top.X_QACT] = int(k * d_cyc * 0.05)
    perf[X + tsr_top.X_MVM_EMPTY] = int(k * d_cyc * 0.02)
    perf[X + tsr_top.X_MVM_WAIT] = int(k * d_cyc * 0.03)
    return perf


def synth_page(k, hz, ch=(0.8, 0.8, 0.8, 0.8), **kw):
    """The fields of the record after k intervals (the deltas are the last interval's)."""
    perf = synth_perf(k, hz, ch)
    dperf = [a - b for a, b in zip(perf, synth_perf(k - 1, hz, ch))] if k > 0 else [0] * 64
    d = dict(pid=os.getpid(), uid=os.getuid(), proc_start_ticks=my_ticks(), flags=tsr_top.F_FITTED, core_hz=int(hz), rv_hz=int(hz / 2), num_ch=4, perf_x=13,
             perf=perf, dperf=dperf, transport="xrt", owner="graph", mode="M7", quant="q4_K", model="synthetic-Q4_K_M.gguf",
             cmdline="llama-completion -m synthetic-Q4_K_M.gguf", comm="llama-completion", image_id=0x1234567890abcdef, caps=0x21,
             n_kinds=22, core_ver=0x100, t_max=4, mem_used=12 * 2 ** 30, mem_total=32 * 2 ** 30, ch_bytes=8 * 2 ** 30,
             device_info="device xilinx_u200_gen3x16_xdma_base_2 (0000:ff:00.1), xclbin /x/tsr_u200_k4.xclbin (uuid u), kernel tsr_u200_k4, compute unit cu",
             kinds="NOP JOIN MUL_MAT")
    d.update(kw)
    return d


def make_sysfs(root, bdf="0000:ff:00.1", clocks="250\n125\n0\n0\n", hwmon=True, xmc_fan=False, icap=True):
    """A made-up sysfs tree with one card (the layout of the real one: sysfs of 0000:01:00.1 on the U200 host)."""
    dev = os.path.join(root, "bus/pci/devices", bdf)
    os.makedirs(os.path.join(root, "bus/pci/drivers/xocl", bdf), exist_ok=True)
    os.makedirs(dev, exist_ok=True)

    def put(rel, text):
        path = os.path.join(dev, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    put("vendor", "0x10ee\n")
    put("board_name", "u200\n")
    put("current_link_speed", "8.0 GT/s PCIe\n")
    put("current_link_width", "16\n")
    if icap:
        put("icap.u.22020096/clock_freqs", clocks)
        put("icap.u.22020096/clock_freq_topology", "DATA_CLK")
    if hwmon:
        put("hwmon/hwmon8/name", "xilinx_u200_gen3x16_xdma_base_2_user\n")
        put("hwmon/hwmon8/temp1_label", "PCB TOP FRONT\n")
        put("hwmon/hwmon8/temp1_input", "51000\n")
        put("hwmon/hwmon8/temp4_label", "FPGA TEMP\n")
        put("hwmon/hwmon8/temp4_input", "58000\n")
        put("hwmon/hwmon8/power1_input", "51170562\n")
        put("hwmon/hwmon8/fan1_input", "1264\n")
    if xmc_fan:
        put("xmc.u.18874368/xmc_fan_rpm", "1500\n")
        put("xmc.u.18874368/fan_presence", "P\n")
        put("xmc.u.18874368/xmc_fpga_temp", "61\n")
    return dev


def fake_xbutil(path, temp="60", watts="40.5", max_w="225", fan=None):
    """A script that writes the JSON that xbutil examine -r thermal electrical mechanical -f JSON -o FILE writes."""
    fans = '[{"speed_rpm":"%s","is_present":"true"}]' % fan if fan else "[]"
    doc = ('{"devices":[{"thermals":[{"location_id":"fpga0","temp_C":"%s","is_present":"true"}],'
           '"electrical":{"power_consumption_watts":"%s","power_consumption_max_watts":"%s"},"mechanical":{"fans":%s}}]}') % (temp, watts, max_w, fans)
    with open(path, "w") as f:
        f.write("#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ \"$1\" = -o ]; then out=\"$2\"; fi; shift; done\ncat > \"$out\" <<'EOF'\n%s\nEOF\n" % doc)
    os.chmod(path, 0o755)


def meter_text(txt, label):
    """What stands between the brackets of the meter LABEL[...] in a frame; None when it has no such meter."""
    m = re.search(re.escape(label) + r"\[([^\]]*)\]", txt)
    return m.group(1) if m else None


def meter_spans(line):
    """The summary meters of one frame line: [(label, column, width)]."""
    return [(m.group(1), m.start(), m.end() - m.start()) for m in re.finditer(r"(COMPUTE|MEMORY|MEM BW)\[[^\]]*\]", line)]


def meters_layout(txt):
    """'row' when MEM BW[ is on the line of COMPUTE[, 'below' when it is on the next line, else None."""
    lines = txt.splitlines()
    for i, x in enumerate(lines):
        if "COMPUTE[" in x:
            if "MEM BW[" in x:
                return "row"
            return "below" if i + 1 < len(lines) and "MEM BW[" in lines[i + 1] else None
    return None


def plot_rows(lines):
    """The time series box of a text frame: the row of each mark of its axis, the interior rows with a heavy line glyph (the
    memory line of a text frame) and those with a light one (the COMPUTE line; the vertical glyph is left out: it looks like
    the border)."""
    mark, heavy, light = {}, [], []
    for i, x in enumerate(lines):
        if len(x) > 5 and x[3] == "\u2502":
            if x[:3].strip() in ("100", "75", "50", "25", "0"):
                mark[x[:3].strip()] = i
            body = x[4:-1]
            if any(c in body for c in "\u2501\u2503\u251b\u250f\u2513\u2517"):
                heavy.append(i)
            if any(c in body for c in "\u2500\u256f\u256d\u256e\u2570"):
                light.append(i)
    return mark, heavy, light


def check_meters(size, txt, gbs_total):
    """The summary meters of a text frame of the synthetic record: where they are, how wide, and what MEM BW says."""
    w = int(size.split("x")[0])
    lines = txt.splitlines()
    ci = next((i for i, x in enumerate(lines) if "COMPUTE[" in x), None)
    check(ci is not None and txt.count("MEM BW[") == 1, "%s: the frame has COMPUTE[ and one MEM BW[" % size)
    if ci is None:
        return
    if w >= 96:                     # three meters of one width in one row
        bw = (w - 4) // 3
        got = meter_spans(lines[ci])
        check(meters_layout(txt) == "row" and got == [("COMPUTE", 1, bw), ("MEMORY", 2 + bw, bw), ("MEM BW", 3 + 2 * bw, bw)],
              "%s: three meters of width %d in one row: %s" % (size, bw, got))
        field = bw - len("MEM BW") - 2
    else:                           # COMPUTE and MEMORY in one row, MEM BW as wide as the two below them
        bw = (w - 3) // 2
        below = lines[ci + 1] if ci + 1 < len(lines) else ""
        check(meters_layout(txt) == "below" and meter_spans(lines[ci]) == [("COMPUTE", 1, bw), ("MEMORY", 2 + bw, bw)] and
              meter_spans(below) == [("MEM BW", 1, 2 * bw + 1)] and lines[ci].rfind("]") == below.rfind("]"),
              "%s: MEM BW directly below, its right bracket under the one of MEMORY: '%s' '%s'" % (size, lines[ci], below))
        field = 2 * bw + 1 - len("MEM BW") - 2
    mb = meter_text(txt, "MEM BW") or ""
    check("43% 30.7GB/s" in mb, "%s: MEM BW is the read rate of all channels against 4 x 18 GB/s: '%s'" % (size, mb))
    check(abs(mb.count("|") - gbs_total / 72.0 * field) <= 0.51, "%s: the MEM BW bar fills 43 %% of %d cells: %d" % (size, field, mb.count("|")))


def check_series(size, txt):
    """The time series box of a text frame of the synthetic record: its legend, and its two lines: COMPUTE at 54 % (light)
    and the memory, 12.0 of 32.0 GiB = 37.5 % (heavy, a text frame has no colors)."""
    check("COMPUTE  54% MVM" in txt, "%s: the COMPUTE legend" % size)
    check("MEMORY   38% (heavy)" in txt, "%s: the memory legend (37.5 %% prints as 38; heavy in a text frame)" % size)
    mark, heavy, light = plot_rows(txt.splitlines())
    check(len(mark) == 5 and heavy and all(mark["50"] < i < mark["25"] for i in heavy),
          "%s: the memory line stands at 37.5 %%, between the 50 and the 25 mark: %s, rows %s" % (size, mark, heavy))
    check(light and all(mark["75"] < i <= mark["50"] for i in light), "%s: the COMPUTE line stands at 54 %%: %s, rows %s" % (size, mark, light))


def test_layout(test_tsr_mon):
    r = subprocess.run([test_tsr_mon, "layout"], capture_output=True, text=True, timeout=60)
    check(r.returncode == 0, "test-tsr-mon layout runs")
    c = json.loads(r.stdout)
    p = tsr_top.layout_json()
    check(c["layout"] == p["layout"] == 1, "layout number")
    check(c["size"] == p["size"], "page size")
    check(c["used"] == p["used"], "bytes in use: C %s, Python %s" % (c["used"], p["used"]))
    cf = {f["name"]: f for f in c["fields"]}
    pf = {f["name"]: f for f in p["fields"]}
    check(set(cf) == set(pf), "field names differ: %s" % sorted(set(cf) ^ set(pf)))
    for n in cf:
        if n in pf:
            check(cf[n] == pf[n], "field %s: C %s, Python %s" % (n, cf[n], pf[n]))
    check([f["name"] for f in c["fields"]] == [f["name"] for f in p["fields"]], "field order")
    out = subprocess.run([sys.executable, TOP, "--layout"], capture_output=True, text=True).stdout
    check(json.loads(out) == p, "tsr-top --layout")


def test_synth():
    d = tempfile.mkdtemp(prefix="tsr-top-synth-")
    try:
        hz = 250e6
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        k = 10
        pg = synth_page(k, hz, perf_mono_ns=now, d_wall_ns=100_000_000, d_launches=3, launches=100, perf_launches=100,
                        update_mono_ns=now, tok_decode=50, tok_prefill=7, passes=51)
        path = os.path.join(d, "tsr-mon-%d" % os.getpid())
        write_record(path, make_page(**pg))
        # a file of another dead pid, a name that is no record, and a torn page
        write_record(os.path.join(d, "tsr-mon-2147483000"), make_page(**dict(pg, pid=2147483000, proc_start_ticks=5)))
        write_record(os.path.join(d, "tsr-mon-abc"), b"junk")
        torn = bytearray(make_page(**dict(pg, pid=2147483001)))
        struct.pack_into("<Q", torn, 16, 7)         # odd seq: the writer is in the middle of a store
        write_record(os.path.join(d, "tsr-mon-2147483001"), bytes(torn))

        sysfs = os.path.join(d, "sys")
        make_sysfs(sysfs)
        fake_xbutil(os.path.join(d, "xbutil"))
        senv = {"TSR_TOP_SYSFS": sysfs}
        sargs = ["--xbutil", os.path.join(d, "xbutil")]

        def fresh(**kw):
            """The record's fields with a PERF read of this moment (rates show while the read is at most 2.5 s old); kw changes them."""
            n = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            args = dict(perf_mono_ns=n, d_wall_ns=100_000_000, d_launches=3, launches=100, perf_launches=100, update_mono_ns=n,
                        tok_decode=50, tok_prefill=7, passes=51)
            args.update(kw)
            return synth_page(k, hz, **args)

        def frame(size, **kw):
            """The --once text frame of the record, changed by kw as in fresh()."""
            write_record(path, make_page(**fresh(**kw)))
            r = run_top(["--once", "--window", "0", "--dir", d, "--size", size] + sargs, env=senv)
            check(r.returncode == 0, "--once %s exit %d: %s" % (size, r.returncode, r.stderr[-300:]))
            return r.stdout

        def hist_of(dirpath, polls=2):
            """The history that tsr-top's Monitor keeps (device id -> samples) over a few polls of a record directory."""
            saved = tsr_top.SYSFS
            tsr_top.SYSFS = sysfs                                        # the made-up tree: find_cards and the link read it at every call
            try:
                mon = tsr_top.Monitor(tsr_top.parse_args(["--dir", dirpath, "--no-sensors", "--no-background"]))
                for _ in range(polls):
                    mon.poll()
                return {name: list(samples) for name, samples in mon.hist.items()}
            finally:
                tsr_top.SYSFS = saved

        card_id = "card:0000:ff:00.1"
        recs = tsr_top.scan_records(d)
        check([r["pid"] for r in recs] == [os.getpid()], "only the live publisher is listed: %s" % [r["pid"] for r in recs])
        check(not os.path.exists(os.path.join(d, "tsr-mon-2147483000")), "the stale record of a dead pid is tidied")
        r0 = recs[0]
        check(r0["model"] == "synthetic-Q4_K_M.gguf" and r0["perf"][0] == k * int(0.06 * hz) and r0["image_id"] == 0x1234567890abcdef,
              "record fields read back")
        rr, st = tsr_top.read_record(os.path.join(d, "tsr-mon-2147483001"))
        check(rr is None and st == "busy", "a page with an odd seq is never returned (%s)" % st)
        write_record(os.path.join(d, "tsr-mon-2147483002"), make_page(**dict(pg, layout=2)))
        # layout 2 is written with layout=2: read_record must refuse it
        page2 = bytearray(make_page(**pg))
        struct.pack_into("<I", page2, 8, 2)
        write_record(os.path.join(d, "tsr-mon-2147483002"), bytes(page2))
        rr, st = tsr_top.read_record(os.path.join(d, "tsr-mon-2147483002"))
        check(rr is None and st == "layout", "another layout number is refused (%s)" % st)
        os.unlink(os.path.join(d, "tsr-mon-2147483002"))

        # one record, the record's own deltas (--window 0)
        r = run_top(["--json", "--window", "0", "--dir", d] + sargs, env=senv)
        check(r.returncode == 0, "--json exit %d: %s" % (r.returncode, r.stderr[-300:]))
        snap = json.loads(r.stdout)
        procs = [p for p in snap["processes"] if p["publisher"]]
        check(len(procs) == 1 and procs[0]["pid"] == os.getpid(), "one publisher process")
        rates = procs[0]["rates"]
        check(rates is not None, "rates from the record's last interval")
        if rates:
            check(abs(rates["window_s"] - 0.1) < 1e-9, "window %.3f" % rates["window_s"])
            check(abs(rates["active"] - 0.6) < 0.01, "active %.3f" % rates["active"])
            check(abs(rates["busy"]["mvm"] - 0.9) < 0.01 and abs(rates["busy"]["vu"] - 0.1) < 0.01, "busy %s" % rates["busy"])
            check(abs(rates["busy"]["dma"] - 0.2) < 0.01 and abs(rates["busy"]["qact"] - 0.05) < 0.01, "dma / qact")
            check(abs(rates["mvm_empty"] - 0.02) < 0.01 and abs(rates["mvm_wait"] - 0.03) < 0.01, "empty / wait")
            # 0.06 * 250e6 cycles per 0.1 s, 80 % of them beats of 64 B: 7.68 GB/s per channel over the wall time
            want = 0.06 * hz * 0.8 * 64 / 0.1 / 1e9
            check(all(abs(g - want) < 0.05 for g in rates["ddr"]["ch_gbs"]), "channel GB/s %s, want %.2f" % (rates["ddr"]["ch_gbs"], want))
            check(abs(rates["ddr"]["total_gbs"] - 4 * want) < 0.2, "total GB/s")
            check(abs(rates["ddr"]["stream_limit_gbs"] - 16.0) < 0.01, "stream limit 64 B x 250 MHz = 16 GB/s")
            # COMPUTE: the busiest compute unit (not the DMA) as a share of wall time; MEMORY: the busiest channel against 18 GB/s
            check(rates["compute"]["unit"] == "MVM" and abs(rates["compute"]["busy_active"] - 0.9) < 0.01 and
                  abs(rates["compute"]["pct"] - 0.6 * 0.9) < 0.01, "compute %s" % rates["compute"])
            check(0 <= rates["memory"]["channel"] < 4 and abs(rates["memory"]["gbs"] - want) < 0.05 and
                  abs(rates["memory"]["pct"] - want / 18.0) < 0.003, "memory %s" % rates["memory"])
            u = rates["util"]
            # U = bytes / (wall x 4 x 18 GB/s); (1-H) 0.6; (1-G) 0.9; (1-F) = 0.8 / 0.9
            check(abs(u["U"] - 4 * want / 72.0) < 0.01, "U %.3f" % u["U"])
            check(abs(u["H"] - 0.4) < 0.01 and abs(u["G"] - 0.1) < 0.01 and abs(u["F"] - (1 - 0.8 / 0.9)) < 0.01, "H G F %s" % u)
            prod = 0.6 * 0.9 * (0.8 / 0.9)
            check(abs(u["E"] - u["U"] / prod) < 0.01, "E closes the product: %s" % u)
        card = [d for d in snap["devices"] if d.get("publisher") == os.getpid()]
        check(len(card) == 1 and card[0]["kind"] == "card" and card[0]["bdf"] == "0000:ff:00.1", "the record is on the made-up card")
        if card:
            c = card[0]
            check(c["clocks"] == {"core_mhz": 250, "rv_mhz": 125, "source": "sysfs"}, "clocks from sysfs: %s" % c["clocks"])
            sn = c["sensors"]
            check(sn and sn["ok"] and abs(sn["temp_c"] - 58.0) < 0.01 and abs(sn["power_w"] - 51.170562) < 1e-6 and sn["fan_rpm"] == 1264.0 and
                  sn["power_max_w"] == 225.0 and sn["source"] == {"temp_c": "sysfs", "power_w": "sysfs", "fan_rpm": "sysfs"},
                  "hwmon temperature, power and fan, xbutil for the limit: %s" % sn)
            check(c["link"]["gen"] == 3 and c["link"]["width"] == 16, "PCIe link")
        # the text frame at several sizes: 56x34 is the width of the pane in the project's demo video, 96 and 95 are the two sides of
        # the width from which the three meters share a row
        gbs_total = 4 * 0.06 * hz * 0.8 * 64 / 0.1 / 1e9                 # 30.72 GB/s: 43 % of 4 x 18
        for size in ("80x24", "100x36", "60x20", "132x50", "40x12", "56x34", "96x30", "95x30"):
            w, h = map(int, size.split("x"))
            write_record(path, make_page(**fresh()))
            r = run_top(["--once", "--window", "0", "--dir", d, "--size", size] + sargs, env=senv)
            check(r.returncode == 0, "--once %s exit %d: %s" % (size, r.returncode, r.stderr[-300:]))
            lines = r.stdout.splitlines()
            check(len(lines) <= h, "%s: %d lines" % (size, len(lines)))
            check(all(len(x) <= w for x in lines), "%s: a line is wider than the screen: %d" % (size, max(len(x) for x in lines)))
            check_meters(size, r.stdout, gbs_total)
            if h >= 20:                                                  # room for the time series box
                check_series(size, r.stdout)
            if w >= 80 and h >= 24:
                txt = r.stdout
                for want_s in ("Device", "CORE 250MHz", "RV 125MHz", "TEMP  58", "FAN 1264rpm", "POW  51 / 225 W", "12.0Gi/32.0Gi",
                               "COMPUTE[", "MEMORY[", "% MVM", "COMPUTE ", "PID", "TOK/S", "synth",
                               "M7", str(os.getpid()), "F2Setup", "F6Sort", "F10Quit"):
                    check(want_s in txt, "%s: the frame shows '%s'" % (size, want_s))
                for gone in ("IMG ", "CAPS", "KINDS", "EMPTY[", "WAIT[", "CH0[", "(1-H)", "stream limit", "prefill/s", "VU[", "DMA["):
                    check(gone not in txt, "%s: the frame does not show '%s' by default" % (size, gone))
                if h >= 24:
                    check(len(lines) >= 20, "%s: the frame fills the screen (%d lines)" % (size, len(lines)))
        write_record(path, make_page(**fresh()))
        r = run_top(["--once", "--window", "0", "--dir", d, "--size", "80x24", "--ascii"] + sargs, env=senv)
        check(r.returncode == 0 and all(ord(c) < 128 for c in r.stdout), "--ascii output is ASCII")
        check("MEM BW[" in r.stdout and "43% 30.7GB/s" in r.stdout and "MEMORY   38%" in r.stdout and r.stdout.count("*") >= 1,
              "--ascii: the meters and the legend are there, the memory line is drawn with '*'")

        # MEM BW is the total over all channels against NUM_CH x 18 GB/s, not the busiest channel: channels that stream in 80, 40, 20
        # and 0 % of the active cycles give 7.68 + 3.84 + 1.92 GB/s = 13.44 of 72 (the busiest channel alone would be 43 % of 18)
        mb = meter_text(frame("100x36", ch=(0.8, 0.4, 0.2, 0.0)), "MEM BW") or ""
        check("19% 13.4GB/s" in mb and abs(mb.count("|") - 13.44 / 72.0 * 24) <= 0.51, "MEM BW of unequal channels: '%s'" % mb)
        # where the two lines coincide (the memory in use is 54 %, as the COMPUTE share) the COMPUTE line is drawn over the memory line
        txt = frame("100x36", mem_used=int(0.54 * 32 * 2 ** 30))
        mark, heavy, light = plot_rows(txt.splitlines())
        check("MEMORY   54% (heavy)" in txt and light and not heavy, "COMPUTE is drawn over MEMORY: heavy rows %s, light rows %s" % (heavy, light))
        # no PERF read yet (a model load: no launches) and a read older than 2.5 s: COMPUTE and MEM BW are n/a, MEMORY has its value,
        # and the memory line stands at 37.5 % while the COMPUTE line lies on 0
        no_rates = (("no PERF read", dict(perf_mono_ns=0, d_wall_ns=0, perf=[0] * 64, dperf=[0] * 64)),
                    ("a stale PERF read", dict(perf_mono_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC) - 10_000_000_000)))
        for what, kw in no_rates:
            txt = frame("100x36", **kw)
            check([(meter_text(txt, x) or "?").strip() for x in ("COMPUTE", "MEM BW")] == ["n/a", "n/a"] and
                  "12.0Gi/32.0Gi" in (meter_text(txt, "MEMORY") or ""), "%s: COMPUTE[ and MEM BW[ show n/a, MEMORY[ its value" % what)
            check("COMPUTE n/a" in txt and "MEMORY   38% (heavy)" in txt, "%s: the legend: COMPUTE n/a, MEMORY 38 %%" % what)
            mark, heavy, light = plot_rows(txt.splitlines())
            check(len(mark) == 5 and heavy and all(mark["50"] < i < mark["25"] for i in heavy), "%s: the memory line stands at 37.5 %%" % what)
            check(light and all(i == mark["0"] for i in light), "%s: the COMPUTE line lies on 0 %%" % what)
        # an owner whose memory total is not known (mem_total 0, no capacity record): MEMORY is n/a and its line is not drawn
        txt = frame("100x36", mem_total=0)
        mark, heavy, light = plot_rows(txt.splitlines())
        check((meter_text(txt, "MEMORY") or "").strip() == "n/a" and "MEMORY  n/a" in txt and light and not heavy,
              "no memory total: MEMORY[ is n/a, the legend says n/a, the memory line is not drawn (an unknown sample draws nothing)")
        # the history behind the time series: (compute, busiest channel, DDR GB/s, memory in use), the last of them whether or not
        # the rates are fresh
        write_record(path, make_page(**fresh()))
        hs = hist_of(d).get(card_id, [])
        check(len(hs) == 2 and all(len(x) == 4 for x in hs), "the history samples are 4-tuples: %s" % hs)
        if hs:
            check(abs(hs[-1][0] - 0.54) < 0.01 and abs(hs[-1][1] - 7.68 / 18.0) < 0.003 and abs(hs[-1][2] - 30.72) < 0.2 and hs[-1][3] == 0.375,
                  "the sample of fresh rates ends with the memory in use, 12.0 of 32.0 GiB: %s" % (hs[-1],))
        for what, kw in no_rates:
            write_record(path, make_page(**fresh(**kw)))
            hs = hist_of(d).get(card_id, [])
            check(hs and hs[-1] == (0.0, 0.0, 0.0, 0.375), "%s: the sample is 0 and the memory in use: %s" % (what, hs[-1:]))
        write_record(path, make_page(**pg))

        # an idle card (no publisher): the MEMORY meter measures against the capacity the transport recorded for it
        di = os.path.join(d, "idle")
        os.makedirs(di)
        cap = os.path.join(di, "tsr-cap-0000:ff:00.1")

        def idle_card():
            r = run_top(["--json", "--window", "0", "--dir", di] + sargs, env=senv)
            dv = [x for x in json.loads(r.stdout)["devices"] if x["kind"] == "card"][0]
            t = run_top(["--once", "--window", "0", "--dir", di, "--size", "100x36"] + sargs, env=senv).stdout
            return dv, t
        dv, txt = idle_card()
        check(dv["state"] == "idle" and dv["capacity"] is None and "MEMORY[" in txt and "Gi/" not in txt and "64.0Gi" not in txt,
              "idle card, no capacity record: no memory total (not a made-up 64 GiB): %s" % dv["capacity"])
        check((meter_text(txt, "MEM BW") or "").strip() == "n/a" and "MEMORY  n/a" in txt and "COMPUTE n/a" in txt,
              "idle card, no capacity record: MEM BW[ is n/a, the legend says MEMORY  n/a")
        check(plot_rows(txt.splitlines())[2] and not plot_rows(txt.splitlines())[1],
              "idle card, no capacity record: the memory line is not drawn (an unknown sample draws nothing), the COMPUTE line is")
        hs = hist_of(di).get(card_id, [])
        check(hs and hs[-1] == (0.0, 0.0, 0.0, None), "idle card, no capacity record: the memory in use is unknown in the history: %s" % hs[-1:])
        with open(cap, "w") as f:
            f.write("tsr-cap 1\nbdf 0000:ff:00.1\nnum_ch 4\nch_bytes %d\n" % (8 * 2 ** 30))
        check(tsr_top.card_capacity(di, "0000:ff:00.1") == 32 * 2 ** 30, "card_capacity reads the record")
        dv, txt = idle_card()
        check(dv["capacity"] == 32 * 2 ** 30 and "0.0Mi/32.0Gi" in txt and "64.0Gi" not in txt,
              "idle card with a capacity record shows 0.0Mi/32.0Gi (%s)" % dv["capacity"])
        check((meter_text(txt, "MEM BW") or "").strip() == "n/a" and "MEMORY    0%" in txt and meters_layout(txt) == "row",
              "idle card with a capacity record: MEM BW[ is n/a, the memory legend is at 0 %")
        txt = run_top(["--once", "--window", "0", "--dir", di, "--size", "56x34"] + sargs, env=senv).stdout
        check((meter_text(txt, "MEM BW") or "").strip() == "n/a" and "MEMORY    0%" in txt and meters_layout(txt) == "below" and
              len(txt.splitlines()) <= 34 and all(len(x) <= 56 for x in txt.splitlines()),
              "idle card at 56x34: MEM BW[ n/a below the other two meters, the memory legend at 0 %, the frame fits")
        hs = hist_of(di).get(card_id, [])
        check(hs and hs[-1] == (0.0, 0.0, 0.0, 0.0), "idle card with a capacity record: 0 memory in use in the history: %s" % hs[-1:])
        for bad in ("tsr-cap 2\nbdf 0000:ff:00.1\nnum_ch 4\nch_bytes 8589934592\n", "tsr-cap 1\nbdf 0000:ff:00.0\nnum_ch 4\nch_bytes 8589934592\n",
                    "tsr-cap 1\nbdf 0000:ff:00.1\nnum_ch 4\nch_bytes x\n", "tsr-cap 1\nbdf 0000:ff:00.1\nnum_ch 0\nch_bytes 8589934592\n", ""):
            with open(cap, "w") as f:
                f.write(bad)
            check(tsr_top.card_capacity(di, "0000:ff:00.1") is None, "a bad capacity record is ignored: %r" % bad[:30])
        check(tsr_top.card_capacity(di, "../x") is None and tsr_top.card_capacity(di, "") is None, "a bad address is refused")
        with open(cap, "w") as f:
            f.write("tsr-cap 1\nbdf 0000:ff:00.1\nnum_ch 4\nch_bytes %d\n" % (16 * 2 ** 30))
        check(tsr_top.device_mem_total({"capacity": None, "probe": {"ok": True, "ch_bytes": 16 * 2 ** 30, "num_ch": 4}}) == 64 * 2 ** 30,
              "without a record the idle probe's (topology) size is used")
        check(tsr_top.device_mem_total({"capacity": 32 * 2 ** 30, "probe": {"ok": True, "ch_bytes": 16 * 2 ** 30, "num_ch": 4}}) == 32 * 2 ** 30,
              "the recorded capacity wins over the probe")
        check(tsr_top.device_mem_total({"capacity": None, "probe": None}) is None, "nothing known: None")
        g = 2 ** 30
        check(tsr_top.device_mem_use({"rec": {"mem_used": 12 * g, "mem_total": 32 * g}, "state": "busy"}) == (12 * g, 32 * g, 0.375),
              "memory in use: the owner's record gives the use and the total")
        check(tsr_top.device_mem_use({"rec": None, "state": "idle", "capacity": 32 * g}) == (0, 32 * g, 0.0), "an idle card has nothing in use")
        check(tsr_top.device_mem_use({"rec": None, "state": "held", "capacity": 32 * g}) == (None, 32 * g, None),
              "a holder that does not publish leaves the use unknown")
        check(tsr_top.device_mem_use({"rec": None, "state": "idle"}) == (0, None, None), "no total, no share")
        check(tsr_top.device_mem_use({"rec": {"mem_used": 40 * g, "mem_total": 32 * g}})[2] == 1.0, "the share is at most 1")
        os.unlink(cap)

        # the other sources of the clocks and the sensors
        sysfs2 = os.path.join(d, "sys2")
        make_sysfs(sysfs2, hwmon=False, xmc_fan=True, icap=False)           # no hwmon, no icap: xmc for the fan, xbutil for the rest
        fake_xbutil(os.path.join(d, "xbutil2"), temp="61", watts="40.5", max_w="225", fan="999")
        r = run_top(["--json", "--window", "0", "--dir", d, "--xbutil", os.path.join(d, "xbutil2")], env={"TSR_TOP_SYSFS": sysfs2})
        c = [x for x in json.loads(r.stdout)["devices"] if x.get("publisher") == os.getpid()][0]
        check(c["clocks"] is None, "no icap node: no sysfs clocks (%s)" % c["clocks"])
        sn = c["sensors"]
        check(sn["fan_rpm"] == 1500.0 and sn["temp_c"] == 61.0 and sn["power_w"] == 40.5, "xmc fan, xbutil for what hwmon lacks: %s" % sn)
        r = run_top(["--once", "--window", "0", "--dir", d, "--size", "80x24", "--xbutil", os.path.join(d, "xbutil2")], env={"TSR_TOP_SYSFS": sysfs2})
        check("CORE 250MHz" in r.stdout and "RV 125MHz" in r.stdout, "without sysfs clocks the owner's fitted clocks are shown")
        sysfs3 = os.path.join(d, "sys3")
        make_sysfs(sysfs3, hwmon=False, icap=False)
        r = run_top(["--json", "--window", "0", "--dir", d, "--xbutil", os.path.join(d, "xbutil2")], env={"TSR_TOP_SYSFS": sysfs3})
        c = [x for x in json.loads(r.stdout)["devices"] if x.get("publisher") == os.getpid()][0]
        check(c["sensors"]["fan_rpm"] == 999.0 and c["sensors"]["source"]["fan_rpm"] == "xbutil", "xbutil is the last fallback for the fan: %s" % c["sensors"])
        r = run_top(["--once", "--window", "0", "--dir", d, "--size", "80x24", "--xbutil", "/nonexistent"], env={"TSR_TOP_SYSFS": sysfs3})
        check(r.returncode == 0 and "FAN n/a" in r.stdout and "TEMP n/a" in r.stdout, "all sensors failing shows n/a, not an error")
        # the clock sysfs gives replaces the owner's nominal clock in the rates when the owner could not fit one
        write_record(path, make_page(**synth_page(k, hz, perf_mono_ns=now, d_wall_ns=100_000_000, d_launches=3, launches=100, perf_launches=100,
                                                 update_mono_ns=now, tok_decode=50, tok_prefill=7, passes=51, flags=0, core_hz=int(tsr_top.CORE_HZ_NOMINAL))))
        r = run_top(["--json", "--window", "0", "--dir", d] + sargs, env=senv)
        p = [x for x in json.loads(r.stdout)["processes"] if x["publisher"]][0]
        check(abs(p["rates"]["core_hz"] - 250e6) < 1, "a clock that was not fitted: the rates use sysfs's 250 MHz (%s)" % p["rates"]["core_hz"])
        write_record(path, make_page(**pg))

        # a record that changes while tsr-top samples it: 100 ms steps, 10 tokens/s
        stop = threading.Event()

        def writer():
            kk = k
            tok = 50
            while not stop.is_set():
                kk += 1
                tok += 1
                n = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                write_record(path, make_page(**synth_page(kk, hz, perf_mono_ns=n, d_wall_ns=100_000_000, d_launches=3,
                                                          launches=100 + 3 * (kk - k), perf_launches=100 + 3 * (kk - k), update_mono_ns=n,
                                                          tok_decode=tok, tok_prefill=7, passes=tok + 1)))
                time.sleep(0.1)

        th = threading.Thread(target=writer)
        th.start()
        try:
            time.sleep(0.3)
            r = run_top(["--json", "--window", "1.0", "--dir", d] + sargs, env=senv)
        finally:
            stop.set()
            th.join()
        check(r.returncode == 0, "--json while the record changes: exit %d" % r.returncode)
        snap = json.loads(r.stdout)
        p = [p for p in snap["processes"] if p["publisher"]][0]
        rt = p["rates"]
        check(rt is not None and 0.7 < rt["window_s"] < 1.6, "window of two reads: %s" % (rt and rt["window_s"]))
        if rt:
            check(abs(rt["active"] - 0.6) < 0.15, "active %.3f over the sampled window" % rt["active"])
            check(abs(rt["busy"]["mvm"] - 0.9) < 0.02, "MVM busy of active cycles %.3f" % rt["busy"]["mvm"])
        check(5.0 < p["tok_s"] < 14.0, "tok/s %.2f (10 expected)" % p["tok_s"])
        check(p["launch_s"] > 10.0, "launches/s %.2f" % p["launch_s"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def wait_for(path_glob_dir, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for f in os.listdir(path_glob_dir):
            if re.fullmatch(r"tsr-mon-\d+", f):
                return os.path.join(path_glob_dir, f)
        time.sleep(0.005)
    return None


def test_live(llama, model, probe):
    d = tempfile.mkdtemp(prefix="tsr-top-live-")
    try:
        n_pred = 40
        prompt = "the cat sat on the mat and the dog ran in the park"
        env = dict(os.environ, TSR_DEVICE="sim", TSR_MON_DIR=d, TSR_MON_KEEP="1", TSR_SIM_THREADS="2")
        env.pop("TSR_MON", None)
        cmd = [llama, "-m", model, "-ngl", "99", "-p", prompt, "-n", str(n_pred), "--ignore-eos", "-no-cnv", "--temp", "0",
               "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "512"]
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = wait_for(d)
        check(path is not None, "the record appears while the model loads")
        reads = 0
        last = None
        seen_flight = False
        while proc.poll() is None and path:
            rec, st = tsr_top.read_record(path)
            if rec is not None:
                reads += 1
                if last is not None:
                    check(rec["launches"] >= last["launches"] and rec["update_mono_ns"] >= last["update_mono_ns"] and
                          rec["perf"][0] >= last["perf"][0] and rec["tok_decode"] >= last["tok_decode"],
                          "counters never go back (launches %d -> %d)" % (last["launches"], rec["launches"]))
                seen_flight = seen_flight or rec["launch_begin_mono_ns"] != 0
                last = rec
            else:
                check(st != "bad" or not os.path.exists(path), "a record is readable: %s" % st)
            time.sleep(0.002)
        rc = proc.wait()
        check(rc == 0, "llama-completion exit %d" % rc)
        check(reads > 5, "the record was read %d times while the run was alive" % reads)
        check(seen_flight, "a launch in flight was seen")
        # the record that TSR_MON_KEEP left; its pid is dead, so tsr-top skips it unless told
        rec, st = tsr_top.read_record(path)
        check(rec is not None, "the kept record reads back (%s)" % st)
        recs = tsr_top.scan_records(d, include_dead=True)
        check(len(recs) == 1 and not recs[0]["alive"], "a kept record of a dead pid is listed only with --include-dead")
        rec = recs[0]
        check(rec["pid"] == proc.pid, "pid %d" % rec["pid"])
        check(rec["transport"] == "sim" and rec["owner"] == "graph" and (rec["flags"] & 1) and (rec["flags"] & 4), "transport %s, flags %d" % (rec["transport"], rec["flags"]))
        check(rec["model"] == os.path.basename(model), "model '%s'" % rec["model"])
        check(rec["quant"].lower().startswith("q4_k"), "quant '%s'" % rec["quant"])
        check(rec["mode"] == "M7", "mode %s" % rec["mode"])
        check("MUL_MAT" in rec["kinds"].split(), "kinds: %s" % rec["kinds"])
        check(rec["image_path"].endswith(".elf") and rec["image_id"] != 0 and rec["n_kinds"] > 0, "image %s" % rec["image_path"])
        check("llama-completion" in rec["cmdline"] and "-m" in rec["cmdline"].split(), "cmdline '%s'" % rec["cmdline"][:80])
        check(rec["comm"].startswith("llama-complet"), "comm %s" % rec["comm"])
        check(rec["num_ch"] == 4 and rec["mem_total"] == 4 * rec["ch_bytes"] and rec["mem_used"] > 0, "memory %d of %d" % (rec["mem_used"], rec["mem_total"]))
        # llama.cpp decodes the prompt in one pass after a warm-up pass; n_pred tokens need n_pred - 1 passes of one token
        check(rec["tok_decode"] == n_pred - 1, "tokens decoded %d, expected %d" % (rec["tok_decode"], n_pred - 1))
        check(rec["tok_prefill"] >= len(prompt) // 3 and rec["passes"] == n_pred + 1, "prefill %d tokens, %d passes (expected %d)" %
              (rec["tok_prefill"], rec["passes"], n_pred + 1))
        check(rec["launches"] > 0 and rec["ops"] >= rec["launches"] and rec["launch_ns"] > 0, "launches %d, ops %d" % (rec["launches"], rec["ops"]))
        check(rec["perf_reads"] >= 2 and rec["perf"][0] > 0 and rec["perf"][2] > 0 and rec["perf"][5] > 0, "PERF totals: cycles %d, MVM %d" % (rec["perf"][0], rec["perf"][2]))
        check(rec["perf_launches"] <= rec["launches"] and rec["core_hz"] > 1e6, "core_hz %d" % rec["core_hz"])
        check(rec["launch_begin_mono_ns"] == 0, "no launch in flight at the end")
        # the sum of the channel beats and the unit counters are sane
        X = rec["perf_x"]
        check(rec["perf"][2] <= rec["perf"][0] and rec["perf"][X + 0] <= rec["perf"][0], "unit busy cycles do not exceed the cycles")
        # the JSON snapshot
        r = run_top(["--json", "--window", "0", "--dir", d, "--include-dead", "--no-sensors", "--no-background"])
        check(r.returncode == 0, "--json exit %d %s" % (r.returncode, r.stderr[-200:]))
        snap = json.loads(r.stdout)
        ps = [p for p in snap["processes"] if p["publisher"]]
        check(len(ps) == 1 and ps[0]["pid"] == proc.pid and ps[0]["model"] == os.path.basename(model)[:-5] and ps[0]["mode"] == "M7", "snapshot process")
        check(ps[0]["record"]["tok_decode"] == n_pred - 1 and ps[0]["record"]["perf"][0] == rec["perf"][0], "snapshot record equals the file")
        check([d["kind"] for d in snap["devices"] if d.get("publisher") == proc.pid] == ["sim"], "the publisher is on a sim device")
        rates = ps[0]["rates"]
        check(rates is not None and rates["busy"]["mvm"] > 0.0 and rates["ddr"]["total_gbs"] > 0, "rates from the last PERF interval: %s" % (rates and rates["busy"]))
        r = run_top(["--once", "--window", "0", "--dir", d, "--include-dead", "--no-sensors", "--size", "80x24"])
        check(r.returncode == 0 and os.path.basename(model)[:5] in r.stdout and "M7" in r.stdout, "--once frame of the kept record")
        # without --include-dead the dead pid shows nothing (and the kept file is tidied)
        r = run_top(["--json", "--window", "0", "--dir", d, "--no-sensors", "--no-background"])
        snap = json.loads(r.stdout)
        check([p for p in snap["processes"] if p["publisher"]] == [], "a dead pid's record is ignored")
        check(not os.path.exists(path), "tsr-top tidied the dead pid's record")

        # TSR_MON=0: no file, same run
        d2 = tempfile.mkdtemp(prefix="tsr-top-off-")
        env2 = dict(env, TSR_MON="0", TSR_MON_DIR=d2)
        r = subprocess.run(cmd[:cmd.index("-n")] + ["-n", "4"] + cmd[cmd.index("-n") + 2:], env=env2, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=300)
        check(r.returncode == 0 and os.listdir(d2) == [], "TSR_MON=0 makes no record (%s)" % os.listdir(d2))
        shutil.rmtree(d2, ignore_errors=True)
        # without TSR_MON_KEEP the record is gone after a clean exit
        d3 = tempfile.mkdtemp(prefix="tsr-top-rm-")
        env3 = dict(env, TSR_MON_DIR=d3)
        env3.pop("TSR_MON_KEEP", None)
        r = subprocess.run(cmd[:cmd.index("-n")] + ["-n", "4"] + cmd[cmd.index("-n") + 2:], env=env3, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=300)
        check(r.returncode == 0 and os.listdir(d3) == [], "the record is removed at a clean exit (%s)" % os.listdir(d3))
        shutil.rmtree(d3, ignore_errors=True)
        # the probe on sim: attach-mode open, JSON on stdout
        if probe and os.path.exists(probe):
            r = subprocess.run([probe, "--device", "sim", "--settle-ms", "50"], capture_output=True, text=True, timeout=120,
                               env=dict(os.environ, TSR_DEVICE="sim"))
            try:
                pj = json.loads(r.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                pj = {}
            check(r.returncode == 0 and pj.get("ok") is True and pj.get("num_ch") == 4 and pj.get("transport") == "sim", "tsr-mon-probe on sim: %s %s" % (r.stdout[-200:], r.stderr[-200:]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_identity(klcheck, model, prompts):
    d = tempfile.mkdtemp(prefix="tsr-top-ident-")
    try:
        for phase in ("m6", "m7"):
            outs = {}
            for mon in ("0", "1"):
                md = os.path.join(d, "mon-" + phase)
                os.makedirs(md, exist_ok=True)
                # TSR_MON_PERF_MS=0: a PERF read after every launch. With the default interval the run of a quiet host ends
                # before the first read after the attach is due (100 ms), and the record it leaves holds no PERF counts.
                env = dict(os.environ, TSR_DEVICE="sim", TSR_MON=mon, TSR_MON_DIR=md, TSR_MON_KEEP="1", TSR_SIM_THREADS="2",
                           TSR_MON_PERF_MS="0")
                out = os.path.join(d, "%s-mon%s.klc" % (phase, mon))
                cmd = [klcheck, "run", "-m", model, "--device", "sim", "--phase", phase, "--n-prompts", "1", "-n", "16", "--out", out]
                if prompts:
                    cmd += ["--prompts", prompts]
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
                check(r.returncode == 0, "klcheck run %s TSR_MON=%s exit %d: %s" % (phase, mon, r.returncode, r.stderr[-300:]))
                outs[mon] = out
            r = subprocess.run([klcheck, "compare", "--ref", outs["0"], "--dev", outs["1"], "--expect-identical"], capture_output=True, text=True, timeout=300)
            check(r.returncode == 0, "%s: TSR_MON on / off, every logits row identical: %s" % (phase, (r.stdout + r.stderr)[-300:]))
            recs = tsr_top.scan_records(os.path.join(d, "mon-" + phase), include_dead=True)
            check(len(recs) == 1 and recs[0]["launches"] > 0 and recs[0]["perf_launches"] == recs[0]["launches"] and recs[0]["perf"][0] > 0,
                  "%s: the TSR_MON=1 run published, with a PERF read after every launch (%d records)" % (phase, len(recs)))
            check(recs and recs[0]["mode"] == ("M6" if phase == "m6" else "M7"), "%s: mode %s" % (phase, recs[0]["mode"] if recs else "-"))
            a = open(outs["0"], "rb").read()
            b = open(outs["1"], "rb").read()
            check(len(a) == len(b), "%s: the two .klc files have the same size" % phase)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def tmux_capture(tmux, sock, e=False):
    args = [tmux, "-L", sock, "capture-pane", "-p"] + (["-e"] if e else [])
    return subprocess.run(args, capture_output=True, text=True).stdout


def test_curses(tmux):
    d = tempfile.mkdtemp(prefix="tsr-top-ui-")
    sock = "tsrtop-ui-%d" % os.getpid()

    def keys(*k):
        subprocess.run([tmux, "-L", sock, "send-keys"] + list(k))

    def screen(cond, timeout=45.0):
        """The screen once cond(text) holds (a loaded machine is slow), else the last one."""
        t0 = time.time()
        scr = tmux_capture(tmux, sock)
        while not cond(scr) and time.time() - t0 < timeout:
            time.sleep(0.2)
            scr = tmux_capture(tmux, sock)
        return scr

    try:
        hz = 250e6
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        path = os.path.join(d, "tsr-mon-%d" % os.getpid())
        stop = threading.Event()

        def writer():
            kk = 10
            tok = 0
            while not stop.is_set():
                kk += 1
                tok += 1
                n = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                write_record(path, make_page(**synth_page(kk, hz, perf_mono_ns=n, d_wall_ns=100_000_000, d_launches=3, launches=3 * kk,
                                                          perf_launches=3 * kk, update_mono_ns=n, tok_decode=tok, tok_prefill=7, passes=tok)))
                time.sleep(0.1)

        write_record(path, make_page(**synth_page(10, hz, perf_mono_ns=now, d_wall_ns=100_000_000, launches=30, perf_launches=30)))
        th = threading.Thread(target=writer)
        th.start()
        sysfs = os.path.join(d, "sys")
        make_sysfs(sysfs)
        cmdline = "TSR_MON_DIR=%s TSR_TOP_SYSFS=%s LANG=C.UTF-8 LC_ALL=C.UTF-8 %s %s --xbutil /nonexistent -d 0.3" % (d, sysfs, sys.executable, TOP)
        subprocess.run([tmux, "-L", sock, "kill-server"], capture_output=True)
        r = subprocess.run([tmux, "-L", sock, "new-session", "-d", "-x", "80", "-y", "24", cmdline], capture_output=True, text=True)
        check(r.returncode == 0, "tmux new-session: %s" % r.stderr)
        scr = screen(lambda t: "COMPUTE " in t and "synth" in t and "FAN 1264rpm" in t and "CORE 250MHz" in t, timeout=60.0)
        check(len(scr.splitlines()) == 24, "24 rows")
        for want in ("Device", "CORE 250MHz", "RV 125MHz", "FAN 1264rpm", "COMPUTE[", "MEMORY[", "MEM BW[", "COMPUTE ", "% MVM", "PID", "synth",
                     "F2Setup", "F6Sort", "F10Quit"):
            check(want in scr, "the screen shows '%s'" % want)
        for gone in ("IMG ", "KINDS", "MVM[", "EMPTY[", "CH0[", "(1-H)", "tok/s", "prefill/s"):
            check(gone not in scr, "the screen does not show '%s' by default" % gone)
        check("\u250c" in scr and "\u2514" in scr and any(c in scr for c in "\u2500\u2501\u256d\u2570\u250f\u2517"), "box drawing and solid plot lines")
        check(any(x in scr for x in (" 53%", " 54%", " 55%")), "the COMPUTE meter shows about 54 % (the MVM holds an op in 90 % of the 60 % active time)")
        check(meters_layout(scr) == "below", "80x24: MEM BW[ is on the line below COMPUTE[ and MEMORY[")
        m = re.search(r"(\d+)% ([\d.]+)GB/s", meter_text(scr, "MEM BW") or "")
        # the writer moves 12e6 beats per channel every 0.1 s or a little more: just under 30.72 GB/s, 43 % of 4 x 18
        check(m is not None and 20.0 <= float(m.group(2)) <= 31.0 and abs(int(m.group(1)) - 100 * float(m.group(2)) / 72.0) <= 1.0,
              "MEM BW shows the read rate of the four channels, about 30 GB/s and 42 or 43 %%: '%s'" % (meter_text(scr, "MEM BW"),))
        check("MEMORY   38%" in scr and "(heavy)" not in scr, "the memory legend says 38 %, without '(heavy)': the screen has colors")
        esc = tmux_capture(tmux, sock, e=True)
        cyan, yellow = r"\x1b\[(?:\d+;)*36m", r"\x1b\[(?:\d+;)*33m"
        check(re.search(cyan + "COMPUTE ", esc) and re.search(yellow + "MEMORY ", esc), "the legend: COMPUTE in cyan, MEMORY in yellow")
        check(re.search(cyan + "\u2500", esc) and re.search(yellow + "\u2500", esc),
              "the time series box has a cyan line (COMPUTE) and a yellow one (the memory in use)")
        # setup menu: F2 opens it, Down and Space switch the time series panel off, Esc closes it
        keys("F2")
        scr = screen(lambda t: "[x] Compute, memory and memory bandwidth meters" in t)
        check("Setup" in scr and "[x] Compute, memory and memory bandwidth meters" in scr and "[x] Compute and memory use time series" in scr and
              "[ ] Unit meters" in scr and "[ ] Image line" in scr and "Esc" in scr,
              "F2 opens the setup menu; the unit meters, the DDR channels, U and the image line are off")
        keys("Down", "Space")
        scr = screen(lambda t: "[ ] Compute and memory use time series" in t)
        check("[ ] Compute and memory use time series" in scr, "Space switches the time series panel off in the menu")
        keys("Escape")
        scr = screen(lambda t: "[x] Compute" not in t and "COMPUTE " not in t)
        check("[x] Compute" not in scr and "COMPUTE " not in scr and "MEMORY  " not in scr and "(heavy)" not in scr and
              meters_layout(scr) == "below" and "MEMORY[" in scr,
              "Esc closes the menu, the time series panel is off (no legend lines), the three meters stay")
        keys("F2", "Down", "Space", "Escape")
        scr = screen(lambda t: "COMPUTE " in t and "MEMORY   38%" in t)
        check("COMPUTE " in scr and "MEMORY   38%" in scr, "the panel comes back with both legend lines")
        # the panels that are off by default: unit meters, DDR channels and U (items 5 to 7), then the image line (item 4)
        keys("F2", "Down", "Down", "Down", "Down", "Down", "Space", "Down", "Space", "Down", "Space", "Escape")
        scr = screen(lambda t: "EMPTY[" in t and "CH0[" in t and "(1-H)" in t)
        for want in ("MVM[", "VU[", "EMPTY[", "WAIT[", "CH0[", "CH3[", "rd ", "stream limit", "(1-H)"):
            check(want in scr, "the panels switched on in setup show '%s'" % want)
        keys("F2", "Down", "Down", "Down", "Down", "Space", "Escape")
        scr = screen(lambda t: "IMG tsr_u200_k4" in t or "IMG " in t)
        check("IMG tsr_u200_k4" in scr and "CAPS 0x21" in scr and "KINDS 22:" in scr, "the image line is behind the setup menu")
        # sort menu
        keys("F6")
        scr = screen(lambda t: "Sort by" in t)
        check("Sort by" in scr and "LAUNCH/s" in scr, "F6 opens the sort menu")
        keys("Down", "Enter")
        scr = screen(lambda t: "Sort by" not in t)
        check("Sort by" not in scr, "Enter picks a column and closes the menu")
        # refresh interval
        keys("+")
        check("Refresh 0.4s" in screen(lambda t: "Refresh 0.4s" in t), "+ raises the refresh interval")
        keys("-")
        check("Refresh 0.3s" in screen(lambda t: "Refresh 0.3s" in t), "- lowers it")
        # resize: wider, then narrow
        subprocess.run([tmux, "-L", sock, "resize-window", "-x", "132", "-y", "40"])
        scr = screen(lambda t: len(t.splitlines()) == 40 and "Device" in t and "COMPUTE[" in t and "COMMAND" in t and meters_layout(t) == "row")
        check(len(scr.splitlines()) == 40 and "COMPUTE[" in scr and "COMMAND" in scr and meters_layout(scr) == "row" and "MEMORY[" in scr,
              "resize to 132x40: the three meters share one line")
        subprocess.run([tmux, "-L", sock, "resize-window", "-x", "80", "-y", "24"])
        scr = screen(lambda t: len(t.splitlines()) == 24 and "Device" in t and "F2Setup" in t and "COMPUTE[" in t and meters_layout(t) == "below")
        check(len(scr.splitlines()) == 24 and "F2Setup" in scr and meters_layout(scr) == "below", "back to 80x24: MEM BW[ below the other two meters")
        subprocess.run([tmux, "-L", sock, "resize-window", "-x", "50", "-y", "14"])
        scr = screen(lambda t: len(t.splitlines()) == 14 and "Device" in t and "F2Setup" in t and "MEM BW[" in t)
        check("Device" in scr and "F2Setup" in scr and "MEM BW[" in scr, "a 50x14 terminal still shows the header, the meters and the keys:\n" + scr)
        # quit
        keys("q")
        t0 = time.time()
        while time.time() - t0 < 20:
            r = subprocess.run([tmux, "-L", sock, "ls"], capture_output=True, text=True)
            if r.returncode != 0 or "tsrtop" not in r.stdout:
                break
            time.sleep(0.2)
        check(r.returncode != 0 or "tsrtop" not in r.stdout, "q ends tsr-top (and the session with it)")
        stop.set()
        th.join()
    finally:
        stop.set()
        subprocess.run([tmux, "-L", sock, "kill-server"], capture_output=True)
        shutil.rmtree(d, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    which = sys.argv[1]
    if which == "layout":
        test_layout(sys.argv[2])
    elif which == "synth":
        test_synth()
    elif which == "live":
        if not os.path.exists(sys.argv[3]):
            print("tiny model %s not built: skipped" % sys.argv[3])
            return 77
        test_live(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif which == "identity":
        if not os.path.exists(sys.argv[3]):
            print("model %s not built: skipped" % sys.argv[3])
            return 77
        test_identity(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif which == "curses":
        tmux = (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None) or shutil.which("tmux")
        if not tmux or not os.path.exists(tmux):
            print("tmux not found: skipped")
            return 77
        test_curses(tmux)
    else:
        return 2
    print("%d checks, %d failed" % (checks, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

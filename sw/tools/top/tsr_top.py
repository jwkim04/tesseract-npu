#!/usr/bin/env python3
# tsr-top — a live terminal monitor for the Tesseract card, laid out like nvtop
# (docs/prof-dbg.md §7, decision SW-87).
#
# The XRT transport holds the compute unit exclusively (EX §2.4), so while one process runs on the card
# nobody else can read its CSRs. The process that owns the card therefore publishes a small record in
# shared memory (/dev/shm/tsr-mon-<pid>, sw/transport/tsr_mon.h) after every launch, and this tool reads
# those records. It never touches the compute unit while a publisher is alive. Board sensors come from
# xbutil (the management path), the PCIe link from sysfs, host CPU and memory from /proc.
#
#   tsr-top                  the live screen (q quits, F2 setup, F6 sort, +/- refresh, Tab next device)
#   tsr-top --once           one frame as plain text, no curses (for logs and tests)
#   tsr-top --json           one snapshot as JSON
#   tsr-top --probe          also open an idle card briefly in attach mode (see --help)
#
# Standard library only (curses, json, struct, threading). Run it with a Python that has curses.
import argparse
import collections
import curses
import datetime
import glob
import json
import locale
import math
import os
import pwd
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time

VERSION = 1
LAYOUT = 1
PAGE = 4096
MON_DIR_DEFAULT = "/dev/shm"
DRAM_GBS = 18.0                      # per channel: what the DRAM gives, as tsr_bw measures it (docs/bandwidth-kernel.md §7)
XBUTIL = "/opt/xilinx/xrt/bin/xbutil"
SYSFS = os.environ.get("TSR_TOP_SYSFS", "/sys")       # the tests point it at a made-up tree

F_SIM, F_FITTED, F_EXITING = 1, 2, 4

# PERF slot indices (EX §6); channel and extension slots depend on NUM_CH (perf_x = 5 + 2 NUM_CH)
P_CYCLES, P_RETIRED, P_MVM, P_VU = 0, 1, 2, 3
X_DMA, X_QACT, X_ATT, X_MVM_WAIT, X_MVM_EMPTY = 0, 1, 2, 3, 4
X_LAUNCHES, X_RV_WFI, X_ATT_QK, X_ATT_PV = 11, 12, 16, 17

# ---------------------------------------------------------------------------------------------
# The record layout 1 (sw/transport/tsr_mon.h). (name, kind, count): kind c = char array, B = u8,
# I = u32, Q = u64. Offsets follow C's natural alignment; the ctest tools_top_layout compares them with
# the table the C side prints (`test-tsr-mon layout`).
# ---------------------------------------------------------------------------------------------
FIELDS = [
    ("magic", "c", 8), ("layout", "I", 1), ("size", "I", 1), ("seq", "Q", 1), ("flags", "I", 1), ("used", "I", 1),
    ("pid", "I", 1), ("uid", "I", 1), ("proc_start_ticks", "Q", 1), ("start_real_ns", "Q", 1),
    ("update_real_ns", "Q", 1), ("update_mono_ns", "Q", 1), ("perf_mono_ns", "Q", 1),
    ("launches", "Q", 1), ("launch_ns", "Q", 1), ("ops", "Q", 1), ("last_launch_ns", "Q", 1),
    ("launch_begin_mono_ns", "Q", 1), ("perf_launches", "Q", 1), ("d_launches", "Q", 1), ("d_wall_ns", "Q", 1),
    ("tok_decode", "Q", 1), ("tok_prefill", "Q", 1), ("passes", "Q", 1), ("perf_ns", "Q", 1), ("hook_ns", "Q", 1), ("mem_used", "Q", 1),
    ("mem_total", "Q", 1), ("core_hz", "Q", 1), ("rv_hz", "Q", 1), ("rv_freq_raw", "I", 1), ("perf_reads", "I", 1),
    ("num_ch", "I", 1), ("t_max", "I", 1), ("lanes", "I", 1), ("spm_bytes", "I", 1), ("k_max", "I", 1),
    ("caps", "I", 1), ("core_ver", "I", 1), ("core_min", "I", 1), ("basis", "I", 1), ("abi", "I", 1),
    ("n_kinds", "I", 1), ("kinds_sig", "I", 1), ("ch_bytes", "Q", 1), ("image_id", "Q", 1), ("kind_mask", "Q", 1),
    ("perf_x", "I", 1), ("reserved1", "I", 1), ("kind_rev", "B", 64), ("perf", "Q", 64), ("dperf", "Q", 64),
    ("transport", "c", 16), ("owner", "c", 16), ("mode", "c", 16), ("quant", "c", 24), ("model", "c", 128),
    ("image_path", "c", 192), ("xclbin", "c", 192), ("device_info", "c", 192), ("kinds", "c", 320),
    ("cmdline", "c", 384), ("comm", "c", 16),
]
_ELEM = {"c": 1, "B": 1, "I": 4, "Q": 8}


def _build_offsets():
    off, table = 0, []
    for name, kind, count in FIELDS:
        a = _ELEM[kind]
        off = (off + a - 1) // a * a
        table.append((name, kind, count, off))
        off += a * count
    return table, off


TABLE, USED = _build_offsets()


def layout_json():
    """The table in the form `test-tsr-mon layout` prints."""
    return {"layout": LAYOUT, "size": PAGE, "used": USED,
            "fields": [{"name": n, "off": o, "elem": _ELEM[k], "count": c} for n, k, c, o in TABLE]}


def parse_page(page):
    """A page (4096 bytes) to a dict; None when it is no record of layout 1."""
    if len(page) < USED or page[:8] != b"TSRMON\0\0":
        return None
    rec = {}
    for name, kind, count, off in TABLE:
        if kind == "c":
            rec[name] = page[off:off + count].split(b"\0", 1)[0].decode("utf-8", "replace")
        elif kind == "B":
            rec[name] = list(page[off:off + count])
        else:
            fmt = "<%d%s" % (count, kind)
            vals = struct.unpack_from(fmt, page, off)
            rec[name] = vals[0] if count == 1 else list(vals)
    rec["magic"] = "TSRMON"
    return rec


def read_record(path):
    """Copy a record under its seqlock. Returns (rec, status): status 'ok', 'layout' (another layout
    number), 'bad' (no record) or 'busy' (the writer never left the page consistent)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None, "bad"
    try:
        for _ in range(2000):
            page = os.pread(fd, PAGE, 0)
            tail = os.pread(fd, 8, 16)
            if len(page) < PAGE or len(tail) < 8:
                return None, "bad"
            s1 = struct.unpack_from("<Q", page, 16)[0]
            s2 = struct.unpack("<Q", tail)[0]
            if s1 & 1 or s1 != s2:
                continue
            if page[:8] != b"TSRMON\0\0":
                return None, "bad"
            if struct.unpack_from("<I", page, 8)[0] != LAYOUT:
                return None, "layout"
            return parse_page(page), "ok"
        return None, "busy"
    finally:
        os.close(fd)


def proc_start_ticks(pid):
    try:
        with open("/proc/%d/stat" % pid, "rb") as f:
            s = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    rp = s.rfind(")")
    parts = s[rp + 2:].split()
    try:
        return int(parts[19])            # field 22, the 20th after the comm field
    except (IndexError, ValueError):
        return None


def mon_dir(opt=None):
    return opt or os.environ.get("TSR_MON_DIR") or MON_DIR_DEFAULT


def scan_records(directory, include_dead=False):
    """Records of the live publishers. A file of a dead pid (or of a reused pid) is ignored."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "tsr-mon-*"))):
        base = os.path.basename(path)
        if not re.fullmatch(r"tsr-mon-\d+", base):
            continue
        rec, st = read_record(path)
        if rec is None:
            continue
        ticks = proc_start_ticks(rec["pid"])
        alive = ticks is not None and (rec["proc_start_ticks"] in (0, ticks))
        rec["alive"] = alive
        rec["path"] = path
        if alive or include_dead:
            out.append(rec)
        else:
            try:                                   # a record that its owner could not remove (killed): ours to tidy
                if os.stat(path).st_uid == os.getuid():
                    os.unlink(path)
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------------------------
# Host side: /proc, sysfs, xbutil
# ---------------------------------------------------------------------------------------------
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SZ = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def read_text(path, default=""):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def user_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def proc_sample(pid):
    """(cpu ticks, rss bytes, uid, cmdline) of a pid, None if it is gone."""
    try:
        with open("/proc/%d/stat" % pid, "rb") as f:
            s = f.read().decode("utf-8", "replace")
        rp = s.rfind(")")
        p = s[rp + 2:].split()
        ticks = int(p[11]) + int(p[12])
        with open("/proc/%d/statm" % pid) as f:
            rss = int(f.read().split()[1]) * PAGE_SZ
        uid = os.stat("/proc/%d" % pid).st_uid
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        if not cmd:
            cmd = "[" + read_text("/proc/%d/comm" % pid) + "]"
    except (OSError, ValueError, IndexError):
        return None
    return ticks, rss, uid, cmd


def find_cards():
    """The Alveo user functions that the xocl driver has: [{'bdf', 'render', 'board'}]."""
    cards = []
    for p in sorted(glob.glob(SYSFS + "/bus/pci/drivers/xocl/0000:*")):
        bdf = os.path.basename(p)
        base = SYSFS + "/bus/pci/devices/" + bdf
        if read_text(base + "/vendor") != "0x10ee":
            continue
        render = None
        link = "/dev/dri/by-path/pci-%s-render" % bdf
        if os.path.islink(link):
            render = os.path.realpath(link)
        board = read_text(base + "/board_name")
        shell = ""
        for n in glob.glob(base + "/hwmon/hwmon*/name"):
            shell = read_text(n)
            break
        shell = re.sub(r"_user$", "", shell)
        cards.append({"bdf": bdf, "render": render, "board": board, "shell": shell})
    return cards


def pcie_link(bdf):
    base = SYSFS + "/bus/pci/devices/" + bdf
    speed = read_text(base + "/current_link_speed")
    width = read_text(base + "/current_link_width")
    m = re.match(r"([\d.]+)", speed)
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}.get(m.group(1) if m else "", None)
    try:
        w = int(width)
    except ValueError:
        w = None
    return {"gen": gen, "width": w, "speed": speed}


def read_int(path):
    try:
        return int(read_text(path).split()[0])
    except (ValueError, IndexError):
        return None


def sysfs_clocks(bdf):
    """The card's kernel clocks, from the icap node of its user function: `clock_freqs` holds one MHz value per clock
    ID (0 = the core, 1 = the RISC-V control core's clock; 0 = not used). None if the node is not there."""
    for p in sorted(glob.glob(SYSFS + "/bus/pci/devices/%s/icap.u.*/clock_freqs" % bdf)):
        vals = []
        for tok in read_text(p).split():
            try:
                vals.append(int(tok))
            except ValueError:
                break
        if len(vals) >= 2 and vals[0] > 0:
            return {"core_mhz": vals[0], "rv_mhz": vals[1] or None, "source": "sysfs"}
    return None


def sysfs_sensors(bdf):
    """Temperature, power and fan from the hwmon directory of the card's user function (and the xmc node for the fan):
    plain file reads, no process, no access to the compute unit. {} when nothing is there."""
    base = SYSFS + "/bus/pci/devices/" + bdf
    out = {}
    for h in sorted(glob.glob(base + "/hwmon/hwmon*")):
        for i in range(1, 32):
            lab = read_text("%s/temp%d_label" % (h, i))
            if lab.upper().startswith("FPGA"):
                v = read_int("%s/temp%d_input" % (h, i))
                if v is not None:
                    out["temp_c"] = v / 1000.0
        v = read_int(h + "/power1_input")
        if v is not None:
            out["power_w"] = v / 1e6                      # microwatts
        v = read_int(h + "/fan1_input")
        if v is not None:
            out["fan_rpm"] = float(v)
        if out:
            break
    if "fan_rpm" not in out:
        for x in sorted(glob.glob(base + "/xmc.u.*")):
            v = read_int(x + "/xmc_fan_rpm")
            if v is not None and read_text(x + "/fan_presence") in ("P", ""):
                out["fan_rpm"] = float(v)
                break
    if "temp_c" not in out:
        for x in sorted(glob.glob(base + "/xmc.u.*")):
            v = read_int(x + "/xmc_fpga_temp")
            if v is not None:
                out["temp_c"] = float(v)
                break
    return out


def card_holders(render):
    """Pids that have the card's render node open (their own fds only: other users' are not readable)."""
    pids = []
    if not render:
        return pids
    for d in glob.glob("/proc/[0-9]*/fd"):
        try:
            for fd in os.listdir(d):
                try:
                    if os.readlink(os.path.join(d, fd)) == render:
                        pids.append(int(d.split("/")[2]))
                        break
                except OSError:
                    pass
        except OSError:
            pass
    return pids


def parse_sensors(doc):
    """xbutil examine JSON (thermal, electrical, mechanical) to {'temp_c', 'power_w', 'power_max_w', 'fan_rpm'}."""
    dev = doc["devices"][0]
    out = {"temp_c": None, "power_w": None, "power_max_w": None, "fan_rpm": None}
    temps = {}
    for t in dev.get("thermals", []):
        if t.get("is_present") == "true":
            try:
                temps[t.get("location_id", "")] = float(t["temp_C"])
            except (KeyError, ValueError):
                pass
    if "fpga0" in temps:
        out["temp_c"] = temps["fpga0"]
    elif temps:
        out["temp_c"] = max(temps.values())
    out["temps"] = temps
    el = dev.get("electrical", {})
    try:
        out["power_w"] = float(el["power_consumption_watts"])
        out["power_max_w"] = float(el["power_consumption_max_watts"])
    except (KeyError, ValueError):
        pass
    for fan in dev.get("mechanical", {}).get("fans", []):
        if fan.get("is_present") == "true":
            try:
                out["fan_rpm"] = float(fan["speed_rpm"])
            except (KeyError, ValueError):
                pass
    return out


class Background(threading.Thread):
    """Board sensors (xbutil, the management path: it does not touch the compute unit), the card's
    users (render node holders) and, when asked, the idle probe. One thread, never blocks the UI."""

    def __init__(self, cards, opts):
        super().__init__(daemon=True)
        self.cards = cards
        self.opts = opts
        self.sensors = {}            # bdf -> dict (replaced whole)
        self.clocks = {}             # bdf -> {'core_mhz', 'rv_mhz', 'source'}
        self._xb = {}                # bdf -> the last xbutil result (the power limit, and the values when sysfs has none)
        self.holders = {}            # bdf -> [pids]
        self.probe = {}              # bdf -> dict
        self.platform = {}           # bdf -> vbnv
        self._stop = threading.Event()
        self.first_pass = threading.Event()      # set when sensors, clocks and holders have been read once
        self._last_xb = {}
        self._last_probe = {}
        self._probe_lock = threading.Lock()
        self.publishers_alive = lambda: True       # set by the monitor: any live publisher means no probe

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            now = time.monotonic()
            for c in self.cards:
                bdf = c["bdf"]
                self.holders[bdf] = card_holders(c["render"])
                self.clocks[bdf] = sysfs_clocks(bdf)
                if not self.opts.no_sensors:
                    if now - self._last_xb.get(bdf, -1e9) >= self.opts.sensor_interval:
                        self._last_xb[bdf] = now
                        self._xb[bdf] = self.read_xbutil(bdf)
                    self.sensors[bdf] = self.merge_sensors(bdf)
                if self.opts.probe and now - self._last_probe.get(bdf, -1e9) >= self.opts.probe_interval:
                    self._last_probe[bdf] = now
                    self.maybe_probe(c)
            self.first_pass.set()
            self._stop.wait(0.5)

    def merge_sensors(self, bdf):
        """sysfs first (a few file reads, every half second), xbutil for what sysfs does not have and for the power limit."""
        sy = sysfs_sensors(bdf)
        xb = self._xb.get(bdf) or {}
        out = {"temp_c": None, "power_w": None, "power_max_w": xb.get("power_max_w"), "fan_rpm": None, "source": {}}
        for k in ("temp_c", "power_w", "fan_rpm"):
            if sy.get(k) is not None:
                out[k], out["source"][k] = sy[k], "sysfs"
            elif xb.get("ok") and xb.get(k) is not None:
                out[k], out["source"][k] = xb[k], "xbutil"
        out["ok"] = any(out[k] is not None for k in ("temp_c", "power_w", "fan_rpm"))
        if not out["ok"]:
            out["error"] = xb.get("error", "no sensors")
        out["t"] = time.time()
        return out

    def read_xbutil(self, bdf):
        exe = self.opts.xbutil
        if not os.path.exists(exe):
            return {"ok": False, "error": "no xbutil", "t": time.time()}
        tdir = tempfile.mkdtemp(prefix="tsr-top-")
        tmp = os.path.join(tdir, "sensors.json")          # xbutil will not write over an existing file
        try:
            r = subprocess.run([exe, "examine", "-d", bdf, "-r", "thermal", "electrical", "mechanical", "-f", "JSON", "-o", tmp],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
            if r.returncode != 0:
                return {"ok": False, "error": "xbutil exit %d" % r.returncode, "t": time.time()}
            with open(tmp) as f:
                doc = json.load(f)
            s = parse_sensors(doc)
            s.update(ok=True, t=time.time())
            return s
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as e:
            return {"ok": False, "error": str(e)[:80], "t": time.time()}
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                os.rmdir(tdir)
            except OSError:
                pass

    def xclbin_matches(self, c):
        """The xclbin named by TSR_XCLBIN is the one on the card (its UUID, read from the file's header and
        from sysfs): opening with another would load it and reprogram the card."""
        path = self.opts.xclbin
        if not path or not os.path.exists(path):
            return False, "no xclbin (set TSR_XCLBIN or --xclbin)"
        try:
            with open(path, "rb") as f:
                head = f.read(512)
        except OSError:
            return False, "cannot read the xclbin"
        if head[:8] != b"xclbin2\0":
            return False, "not an xclbin"
        off = 8 + 4 + 28 + 256 + 8 + 24 + 2 + 1 + 1 + 4 + 16 + 64      # axlf header: m_uuid
        import uuid
        file_uuid = str(uuid.UUID(bytes=head[off:off + 16]))
        card_uuid = read_text(SYSFS + "/bus/pci/devices/%s/xclbinuuid" % c["bdf"]).lower()
        if file_uuid != card_uuid:
            return False, "the xclbin on the card is another one"
        return True, ""

    def maybe_probe(self, c):
        bdf = c["bdf"]
        if self.publishers_alive():
            self.probe[bdf] = {"ok": False, "skipped": "a publisher is alive"}
            return
        if self.holders.get(bdf):
            self.probe[bdf] = {"ok": False, "skipped": "the card is held by pid %s" % ",".join(map(str, self.holders[bdf]))}
            return
        ok, why = self.xclbin_matches(c)
        if not ok:
            self.probe[bdf] = {"ok": False, "skipped": why}
            return
        exe = self.opts.probe_bin
        if not exe or not os.path.exists(exe):
            self.probe[bdf] = {"ok": False, "skipped": "no tsr-mon-probe"}
            return
        # look once more right before opening: the window is the span between this check and the open
        if self.holders.get(bdf) or card_holders(c["render"]) or self.publishers_alive():
            self.probe[bdf] = {"ok": False, "skipped": "the card was taken"}
            return
        env = dict(os.environ, TSR_XCLBIN=self.opts.xclbin, TSR_MON="0")
        env.setdefault("TSR_XRT_DEVICE", bdf)
        try:
            r = subprocess.run([exe, "--device", "xrt"], env=env, capture_output=True, text=True, timeout=60, check=False)
            doc = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"ok": False, "error": "no output"}
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            doc = {"ok": False, "error": str(e)[:80]}
        doc["t"] = time.time()
        self.probe[bdf] = doc


# ---------------------------------------------------------------------------------------------
# Rates from the counters
# ---------------------------------------------------------------------------------------------
def perf_layout(rec):
    C = rec["num_ch"]
    X = rec["perf_x"] or (5 + 2 * C)
    return C, X


def rates_between(a, b, rec, hz=None):
    """Rates over the PERF reads a and b (dicts with 'pm' (CLOCK_MONOTONIC ns of the read), 'perf', 'launches',
    'tok_dec', 'tok_pre'), for the record's configuration. None when the reads are not in order."""
    wall = (b["pm"] - a["pm"]) / 1e9
    if wall <= 0:
        return None
    C, X = perf_layout(rec)
    hz = float(hz or rec["core_hz"]) or 276.3e6
    d = [y - x for x, y in zip(a["perf"], b["perf"])]
    cyc = d[P_CYCLES]
    r = {"window_s": wall, "core_hz": hz}
    r["active"] = min(1.0, cyc / (hz * wall)) if hz > 0 else 0.0
    if cyc > 0:
        r["busy"] = {
            "mvm": d[P_MVM] / cyc, "vu": d[P_VU] / cyc, "dma": d[X + X_DMA] / cyc, "qact": d[X + X_QACT] / cyc,
            "att": d[X + X_ATT] / cyc, "att_qk": d[X + X_ATT_QK] / cyc, "att_pv": d[X + X_ATT_PV] / cyc,
        }
        r["mvm_empty"] = d[X + X_MVM_EMPTY] / cyc
        r["mvm_wait"] = d[X + X_MVM_WAIT] / cyc
    else:
        r["busy"] = {k: 0.0 for k in ("mvm", "vu", "dma", "qact", "att", "att_qk", "att_pv")}
        r["mvm_empty"] = r["mvm_wait"] = 0.0
    r["mvm_wall"] = r["active"] * r["busy"]["mvm"]
    # COMPUTE: the busiest compute unit (DMA moves data, it is not compute), as a share of wall time; the share of the
    # active cycles is r["busy"][unit]. ATT_QK / ATT_PV are the MVM's own attention cycles, so they cannot beat MVM.
    names = (("MVM", "mvm"), ("VU", "vu"), ("QACT", "qact"), ("ATT", "att"), ("ATT_QK", "att_qk"), ("ATT_PV", "att_pv"))
    unit, key = max(names, key=lambda nk: r["busy"][nk[1]])
    r["compute"] = {"unit": unit, "busy_active": r["busy"][key], "pct": r["active"] * r["busy"][key]}
    beats = [d[5 + c] for c in range(C)]
    wait = [d[5 + C + c] for c in range(C)]
    gbs = [x * 64.0 / wall / 1e9 for x in beats]
    stream = 64.0 * hz / 1e9
    top = max(range(C), key=lambda c: gbs[c]) if C else 0
    r["memory"] = {"channel": top, "gbs": gbs[top] if C else 0.0, "pct": (gbs[top] / DRAM_GBS) if C else 0.0}
    r["ddr"] = {
        "ch_gbs": gbs, "ch_wait_pct": [(w / cyc if cyc > 0 else 0.0) for w in wait],
        "total_gbs": sum(gbs), "stream_limit_gbs": stream, "dram_limit_gbs": DRAM_GBS,
        "pct_stream": sum(gbs) / (C * stream) if C else 0.0, "pct_dram": sum(gbs) / (C * DRAM_GBS) if C else 0.0,
    }
    # U ~ (1-H)(1-G)(1-F)E (docs/prof-dbg.md §3). H: no launch active; G: the MVM does not hold an op;
    # F: the busiest channel does not stream while the MVM holds an op (clamped at 0); E is what is left of U.
    one_h = r["active"]
    one_g = r["busy"]["mvm"]
    maxb = max(beats) if beats else 0
    one_f = min(1.0, max(0.0, maxb / d[P_MVM])) if d[P_MVM] > 0 else 0.0
    u = r["ddr"]["pct_dram"]
    prod = one_h * one_g * one_f
    r["util"] = {"U": u, "H": 1.0 - one_h, "G": 1.0 - one_g, "F": 1.0 - one_f, "E": (u / prod) if prod > 0 else None}
    r["launches_per_s"] = (b["launches"] - a["launches"]) / wall
    r["tok_s"] = (b["tok_dec"] - a["tok_dec"]) / wall
    r["pp_s"] = (b["tok_pre"] - a["tok_pre"]) / wall
    return r


class Tracker:
    """The samples of one publisher: a new PERF read is a new sample; rates come from the newest sample
    and the newest one at least `window` seconds before it."""

    def __init__(self):
        self.reads = collections.deque(maxlen=600)
        self.last = None
        self.cpu_prev = None
        self.cpu_pct = 0.0
        self.tok_ticks = collections.deque(maxlen=600)     # (reader mono, tok_dec, tok_pre, launches)

    def add(self, rec, now):
        pm = rec["perf_mono_ns"]
        if pm and (not self.reads or self.reads[-1]["pm"] != pm):
            self.reads.append({"pm": pm, "perf": list(rec["perf"]), "launches": rec["perf_launches"],
                               "tok_dec": rec["tok_decode"], "tok_pre": rec["tok_prefill"]})
        self.tok_ticks.append((now, rec["tok_decode"], rec["tok_prefill"], rec["launches"]))
        self.last = rec

    def rates(self, window, mono_now_ns, hz=None):
        rec = self.last
        if rec is None or not self.reads:
            return None
        b = self.reads[-1]
        a = None
        for s in reversed(self.reads):
            if s is b:
                continue
            a = s
            if b["pm"] - s["pm"] >= window * 1e9:
                break
        if a is None:
            # one read only: the record's own deltas between its last two reads
            if not rec["d_wall_ns"]:
                return None
            a = {"pm": b["pm"] - rec["d_wall_ns"], "perf": [x - y for x, y in zip(b["perf"], rec["dperf"])],
                 "launches": b["launches"] - rec["d_launches"], "tok_dec": b["tok_dec"], "tok_pre": b["tok_pre"]}
        r = rates_between(a, b, rec, hz)
        if r is None:
            return None
        age = (mono_now_ns - b["pm"]) / 1e9
        in_flight = rec["launch_begin_mono_ns"] != 0
        r["age_s"] = age
        r["stale"] = (not in_flight) and age > max(2.5, 2.0 * r["window_s"])
        r["in_flight"] = in_flight
        return r

    def token_rates(self, window):
        """tok/s and prefill tok/s over the reader's own samples (they move at every pass, not at every PERF read)."""
        if len(self.tok_ticks) < 2:
            return 0.0, 0.0, 0.0
        b = self.tok_ticks[-1]
        a = self.tok_ticks[0]
        for s in reversed(self.tok_ticks):
            a = s
            if b[0] - s[0] >= window:
                break
        dt = b[0] - a[0]
        if dt <= 0:
            return 0.0, 0.0, 0.0
        return (b[1] - a[1]) / dt, (b[2] - a[2]) / dt, (b[3] - a[3]) / dt


# ---------------------------------------------------------------------------------------------
# The monitor: records, host data, history, snapshot
# ---------------------------------------------------------------------------------------------
BDF_RE = re.compile(r"\(([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\)")
QUANT_RE = re.compile(r"(?<![A-Za-z0-9])((?:I?Q\d(?:_[A-Z0-9]+)*)|F16|BF16|F32)(?![A-Za-z0-9])", re.I)


def kernel_name(rec):
    m = re.search(r"kernel (\S+?),", rec["device_info"])
    if m:
        return m.group(1)
    if rec["xclbin"]:
        return re.sub(r"\.xclbin$", "", os.path.basename(rec["xclbin"]))
    return ""


def model_name(rec):
    """The model file's name without .gguf: the record's, else the argument of -m / --model of the command line."""
    name = rec["model"]
    if not name:
        args = rec["cmdline"].split()
        for i, a in enumerate(args[:-1]):
            if a in ("-m", "--model", "-hf", "--hf-repo"):
                name = os.path.basename(args[i + 1])
                break
    return re.sub(r"\.gguf$", "", name) if name else "?"


def shorten_mid(s, w):
    """Cut a long name in the middle, so that its end (the quantization) stays."""
    if len(s) <= w or w <= 4:
        return s[:w]
    tail = min(8, (w - 1) // 2)
    return s[:w - 1 - tail] + "~" + s[len(s) - tail:]


def model_quant(rec):
    m = QUANT_RE.search(rec["model"])
    return m.group(1).upper() if m else rec["quant"]


def clock_text(rec, hz):
    """A clock that was measured (fitted from DBG_MTIME, or a simulated one); the nominal value of a card that has not
    run a launch for half a second is not shown as a measurement."""
    return fmt_hz(hz) if hz and (rec["flags"] & (F_FITTED | F_SIM)) else "n/a"


def fmt_hz(hz):
    if not hz:
        return "n/a"
    return "%.0fMHz" % (hz / 1e6)


class Monitor:
    def __init__(self, opts):
        self.opts = opts
        self.dir = mon_dir(opts.dir)
        self.cards = find_cards()
        self.bg = Background(self.cards, opts)
        self.bg.publishers_alive = lambda: bool(self.live_publishers)
        self.live_publishers = []
        self.trk = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=720))   # device id -> (compute, memory, DDR GB/s)
        self.procs = {}
        self.snap = None
        self.interval = max(0.1, opts.delay) if opts.delay else 1.0
        self.last_proc = {}
        self.warned = []

    def start(self):
        if not self.opts.no_background:
            self.bg.start()

    def stop(self):
        self.bg.stop()

    def poll(self):
        now = time.monotonic()
        mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        recs = scan_records(self.dir, self.opts.include_dead)
        self.live_publishers = [r for r in recs if r["alive"]]
        seen = set()
        for r in recs:
            pid = r["pid"]
            seen.add(pid)
            self.trk.setdefault(pid, Tracker()).add(r, now)
        for pid in list(self.trk):
            if pid not in seen:
                del self.trk[pid]
        window = max(self.interval, self.opts.window_min)
        devices = self.build_devices(recs, mono_ns)
        procs = self.build_processes(recs, devices, now, window, mono_ns)
        for dev in devices:
            pr = [p for p in procs if p["device"] == dev["id"] and p["publisher"]]
            rr = None
            for p in pr:
                rr = p["rates"]
            dev["rates"] = rr
            dev["state"] = self.device_state(dev)
            h = self.hist[dev["id"]]
            if rr is not None and not rr["stale"]:
                h.append((rr["compute"]["pct"], rr["memory"]["pct"], rr["ddr"]["total_gbs"]))
            else:
                h.append((0.0, 0.0, 0.0))
        self.snap = {
            "tsr_top": VERSION, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "interval_s": self.interval, "mon_dir": self.dir, "devices": devices, "processes": procs,
        }
        return self.snap

    def build_devices(self, recs, mono_ns):
        devices = []
        by_bdf = {}
        for c in self.cards:
            d = {"id": "card:" + c["bdf"], "kind": "card", "bdf": c["bdf"],
                 "board": ("Alveo " + c["board"].upper()) if c["board"] else "Alveo card", "shell": c["shell"],
                 "link": pcie_link(c["bdf"]), "sensors": self.bg.sensors.get(c["bdf"]),
                 "holders": list(self.bg.holders.get(c["bdf"], [])), "probe": self.bg.probe.get(c["bdf"]),
                 "clocks": self.bg.clocks.get(c["bdf"]), "publisher": None}
            devices.append(d)
            by_bdf[c["bdf"].lower()] = d
        for r in recs:
            if not r["alive"] and not self.opts.include_dead:
                continue
            dev = None
            if not (r["flags"] & F_SIM):
                m = BDF_RE.search(r["device_info"])
                if m:
                    dev = by_bdf.get(m.group(1).lower())
                if dev is None and len(devices) == 1 and devices[0]["kind"] == "card":
                    dev = devices[0]
                if dev is None:
                    dev = {"id": "xrt:%d" % r["pid"], "kind": "card", "bdf": m.group(1) if m else "", "board": "Tesseract card",
                           "shell": "", "link": {"gen": None, "width": None, "speed": ""}, "sensors": None, "holders": [],
                           "probe": None, "clocks": None, "publisher": None}
                    devices.append(dev)
            else:
                dev = {"id": "sim:%d" % r["pid"], "kind": "sim", "bdf": "", "board": "Tesseract %s (pid %d)" % (r["transport"], r["pid"]),
                       "shell": "", "link": {"gen": None, "width": None, "speed": ""}, "sensors": None, "holders": [],
                       "probe": None, "clocks": None, "publisher": None}
                devices.append(dev)
            dev["publisher"] = r["pid"]
            dev["rec"] = r
            dev["in_flight_s"] = (mono_ns - r["launch_begin_mono_ns"]) / 1e9 if r["launch_begin_mono_ns"] else None
        return devices

    def device_state(self, dev):
        r = dev.get("rec")
        if r is None:
            if dev["holders"]:
                return "held"
            return "idle"
        if not r["alive"]:
            return "exited"
        if (dev.get("in_flight_s") or 0.0) > 1.0:
            return "launch"                       # a long launch: the counters are those of the last one
        rr = dev.get("rates")
        if r["launch_begin_mono_ns"] != 0 or (rr is not None and not rr["stale"] and rr["active"] > 0.005):
            return "busy"
        return "owned"

    def build_processes(self, recs, devices, now, window, mono_ns):
        procs = []
        dev_of_pid = {d["publisher"]: d for d in devices if d.get("publisher")}
        for r in recs:
            pid = r["pid"]
            t = self.trk[pid]
            s = proc_sample(pid)
            cpu, rss, user, cmd = 0.0, 0, user_name(r["uid"]), r["cmdline"]
            if s is not None:
                ticks, rss, uid, cmd = s
                user = user_name(uid)
                prev = self.last_proc.get(pid)
                if prev is not None and now > prev[1]:
                    cpu = max(0.0, (ticks - prev[0]) / CLK_TCK / (now - prev[1]) * 100.0)
                else:
                    cpu = t.cpu_pct
                self.last_proc[pid] = (ticks, now)
                t.cpu_pct = cpu
            dev = dev_of_pid.get(pid)
            rates = t.rates(window, mono_ns, core_hz_of(r, dev))
            tok, pp, lps = t.token_rates(max(window, self.opts.token_window))      # tokens come in steps: a longer window smooths them
            if rates is not None and rates["stale"]:
                tok = pp = lps = 0.0
            procs.append({
                "pid": pid, "user": user, "model": model_name(r), "quant": model_quant(r), "mode": r["mode"],
                "tok_s": tok, "pp_s": pp, "launch_s": lps, "cpu_pct": cpu, "rss": rss, "cmd": cmd or r["cmdline"],
                "device": dev["id"] if dev else None, "publisher": True, "rates": rates, "alive": r["alive"],
                "record": record_json(r),
            })
        for dev in devices:
            for pid in dev["holders"]:
                if pid in dev_of_pid and dev_of_pid[pid]["id"] == dev["id"]:
                    continue
                if any(p["pid"] == pid for p in procs):
                    continue
                s = proc_sample(pid)
                if s is None:
                    continue
                ticks, rss, uid, cmd = s
                cpu = 0.0
                prev = self.last_proc.get(pid)
                if prev is not None and now > prev[1]:
                    cpu = max(0.0, (ticks - prev[0]) / CLK_TCK / (now - prev[1]) * 100.0)
                self.last_proc[pid] = (ticks, now)
                procs.append({"pid": pid, "user": user_name(uid), "model": "-", "quant": "", "mode": "?", "tok_s": None, "pp_s": None,
                              "launch_s": None, "cpu_pct": cpu, "rss": rss, "cmd": cmd, "device": dev["id"], "publisher": False,
                              "rates": None, "alive": True, "record": None})
        return procs


def core_hz_of(rec, dev):
    """The core clock for the rates: the one the owner fitted from DBG_MTIME, or a simulator's; else what sysfs says the card runs at
    (an image without the debug block cannot be fitted, and the owner then only has the nominal clock)."""
    if rec["flags"] & (F_FITTED | F_SIM):
        return rec["core_hz"]
    clk = dev.get("clocks") if dev else None
    return clk["core_mhz"] * 1e6 if clk else rec["core_hz"]


def device_clocks(dev):
    """(core, rv) clock texts of the header: sysfs for a card, else the owner's fitted (or the simulator's) clock, else n/a."""
    r = dev.get("rec")
    clk = dev.get("clocks")
    if dev["kind"] == "card" and clk:
        return "%dMHz" % clk["core_mhz"], ("%dMHz" % clk["rv_mhz"]) if clk["rv_mhz"] else "n/a"
    if r is not None:
        return clock_text(r, r["core_hz"]), clock_text(r, r["rv_hz"])
    probe = dev.get("probe")
    if probe and probe.get("ok"):
        return fmt_hz(probe.get("core_hz")), fmt_hz(probe.get("rv_hz"))
    return "n/a", "n/a"


def record_json(r):
    """The record for --json: everything, the kind table decoded, the PERF arrays trimmed to what the device has."""
    C, X = perf_layout(r)
    n = min(64, X + 22)
    out = {k: v for k, v in r.items() if k not in ("perf", "dperf", "kind_rev", "magic", "path")}
    out["perf"] = r["perf"][:n]
    out["dperf"] = r["dperf"][:n]
    out["kind_rev"] = {str(i): v for i, v in enumerate(r["kind_rev"]) if v}
    out["kernel"] = kernel_name(r)
    out["image_id_hex"] = "%016x" % r["image_id"]
    out["caps_hex"] = "0x%x" % r["caps"]
    return out


# ---------------------------------------------------------------------------------------------
# Frame: a grid of cells with styles. The curses screen and the text output both come from it.
# ---------------------------------------------------------------------------------------------
(S_DEF, S_LABEL, S_VAL, S_BAR, S_BAR2, S_BAR3, S_MAG, S_DIM, S_REV, S_S1, S_S2, S_S3, S_WARN, S_ERR, S_OK, S_BOX, S_HEAD, S_SEL) = range(18)


class Frame:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.ch = [[" "] * w for _ in range(h)]
        self.st = [[S_DEF] * w for _ in range(h)]

    def put(self, y, x, s, st=S_DEF):
        if y < 0 or y >= self.h:
            return
        for ch in s:
            if 0 <= x < self.w:
                self.ch[y][x] = ch
                self.st[y][x] = st
            x += 1
            if x >= self.w:
                break

    def segs(self, y, x, parts):
        for text, st in parts:
            self.put(y, x, text, st)
            x += len(text)
        return x

    def fill(self, y, x, w, ch, st=S_DEF):
        self.put(y, x, ch * max(0, w), st)

    def lines(self):
        return [("".join(r)).rstrip() for r in self.ch]

    def text(self):
        ls = self.lines()
        while ls and not ls[-1]:
            ls.pop()
        return "\n".join(ls)


class Charset:
    def __init__(self, utf8):
        self.utf8 = utf8
        if utf8:
            self.h, self.v = "─", "│"
            self.tl, self.tr, self.bl, self.br = "┌", "┐", "└", "┘"
            self.deg = "°"
        else:
            self.h, self.v = "-", "|"
            self.tl = self.tr = self.bl = self.br = "+"
            self.deg = "C"      # "59 C" reads as 59 degrees


# Line glyphs of a plot, as nvtop draws them: flat, vertical, then the four turns (from the left going up, the top of a
# rise, from the left going down, the bottom of a fall). The heavy set tells a series apart when there is no color.
LINE_LIGHT = ("─", "│", "╯", "╭", "╮", "╰")
LINE_HEAVY = ("━", "┃", "┛", "┏", "┓", "┗")


def fmt_gib(b):
    if b < 2 ** 30:
        return "%.1fMi" % (b / 2.0 ** 20)
    return "%.1fGi" % (b / 2.0 ** 30)


def fmt_mem(b):
    if b is None:
        return "-"
    for unit, div in (("GiB", 2.0 ** 30), ("MiB", 2.0 ** 20), ("KiB", 1024.0)):
        if b >= div:
            return "%.0f%s" % (b / div, unit) if b / div >= 100 or unit != "GiB" else "%.1f%s" % (b / div, unit)
    return "%dB" % b


def draw_bar(f, y, x, w, label, frac, text, style=S_BAR, na=False):
    """nvtop's meter: LABEL[|||||    87%]; the text sits at the right end of the field."""
    if w < len(label) + 6:
        return
    inner = w - len(label) - 2
    x0 = x + len(label) + 1
    f.put(y, x, label, S_LABEL)
    f.put(y, x + len(label), "[", S_DEF)
    f.put(y, x + w - 1, "]", S_DEF)
    if na or frac is None:
        f.put(y, x0 + inner - 3, "n/a", S_DIM)
        return
    n = int(round(max(0.0, min(1.0, frac)) * inner))
    f.fill(y, x0, n, "|", style)
    t = text[:inner]
    f.put(y, x0 + inner - len(t), t, S_VAL)


def pct(v):
    return "%3.0f%%" % (100.0 * v)


def draw_graph(f, cs, y, x, h, w, series, legend, left_labels, right_labels, window_s):
    """One box with line plots, one sample per cell. series: list of (values, vmax, style, heavy). The box occupies
    columns x..x+w-1 and rows y..y+h-1; labels go in the three columns outside it."""
    if h < 4 or w < 8:
        return
    bx, bw = x, w
    f.put(y, bx, cs.tl + cs.h * (bw - 2) + cs.tr, S_BOX)
    f.put(y + h - 1, bx, cs.bl + cs.h * (bw - 2) + cs.br, S_BOX)
    for r in range(1, h - 1):
        f.put(y + r, bx, cs.v, S_BOX)
        f.put(y + r, bx + bw - 1, cs.v, S_BOX)
    pw, ph = bw - 2, h - 2                    # plot cells
    # time axis in the bottom border: seconds back from now at the quartiles
    if bw >= 24:
        for i in range(5):
            sec = window_s * (1.0 - i / 4.0)
            lab = "%ds" % round(sec)
            px = bx + 1 + int(round(i * (pw - 1) / 4.0))
            if i == 0:
                px = bx + 1
            elif i == 4:
                px = bx + bw - 1 - len(lab)
            else:
                px = px - len(lab) // 2
            f.put(y + h - 1, px, lab, S_BOX)
    # axis labels
    for r, lab in left_labels:
        f.put(y + 1 + int(round(r * (ph - 1))), x - len(lab), lab, S_DEF)
    for r, lab in right_labels:
        f.put(y + 1 + int(round(r * (ph - 1))), x + bw, lab, S_DEF)
    if cs.utf8:
        for vals, vmax, st, heavy in series:                       # later series are drawn over earlier ones
            flat, vert, up_from, up_to, down_from, down_to = LINE_HEAVY if heavy else LINE_LIGHT
            vals = list(vals)[-pw:]
            vals = [None] * (pw - len(vals)) + vals                # no sample yet: nothing drawn
            prev = None
            for cx, v in enumerate(vals):
                if v is None:
                    prev = None
                    continue
                frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax))
                r = ph - 1 - int(round(frac * (ph - 1)))           # row in the plot, 0 = top
                col = bx + 1 + cx
                if prev is None or r == prev:
                    f.put(y + 1 + r, col, flat, st)
                elif r < prev:                                     # a rise: turn up at the old level, right at the new
                    f.put(y + 1 + prev, col, up_from, st)
                    for rr in range(r + 1, prev):
                        f.put(y + 1 + rr, col, vert, st)
                    f.put(y + 1 + r, col, up_to, st)
                else:                                              # a fall: turn down at the old level, right at the new
                    f.put(y + 1 + prev, col, down_from, st)
                    for rr in range(prev + 1, r):
                        f.put(y + 1 + rr, col, vert, st)
                    f.put(y + 1 + r, col, down_to, st)
                prev = r
    else:
        for si, (vals, vmax, st, heavy) in enumerate(series):
            vals = list(vals)[-pw:]
            vals = [None] * (pw - len(vals)) + vals
            for cx, v in enumerate(vals):
                if v is None:
                    continue
                frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax))
                r = ph - 1 - int(round(frac * (ph - 1)))
                f.put(y + 1 + r, bx + 1 + cx, "*+#"[si % 3], st)
    # legend over the plot, as nvtop does
    for i, (text, st) in enumerate(legend):
        f.put(y + 1 + i, bx + 1, text, st)


# ---------------------------------------------------------------------------------------------
# Screen composition
# ---------------------------------------------------------------------------------------------
SORT_KEYS = [("PID", "pid"), ("USER", "user"), ("MODEL", "model"), ("MODE", "mode"), ("TOK/S", "tok_s"),
             ("LAUNCH/s", "launch_s"), ("CPU", "cpu_pct"), ("HOST MEM", "rss"), ("COMMAND", "cmd")]
PANELS = [("summary", "Compute and memory meters"), ("graph", "Utilization time series"), ("table", "Process table"),
          ("others", "Card users without tsr-mon"), ("image", "Image line (IMG, ID, CAPS, kinds, mode)"),
          ("meters", "Unit meters (MVM VU DMA QACT ATT EMPTY WAIT)"), ("ddr", "DDR channel bars and read rate"),
          ("util", "Utilization split U")]
PANELS_OFF = ("image", "meters", "ddr", "util")            # what --json has and the screen shows only when asked


class View:
    """What the user chose: panels, sort, device; and what the screen is showing."""

    def __init__(self, opts):
        self.panels = {k: k not in PANELS_OFF for k, _ in PANELS}
        self.sort = "tok_s"
        self.desc = True
        self.dev = 0
        self.dev_manual = False     # Tab was pressed: stop following the device that has a publisher
        self.menu = None            # None, 'setup', 'sort'
        self.sel = 0
        self.color = not opts.no_color
        self.utf8 = not opts.ascii
        self.mono = not self.color            # set from the theme once curses knows whether the terminal has colors


def device_header(f, cs, y, w, dev, idx, n_dev, v):
    r = dev.get("rec")
    gen = dev["link"].get("gen")
    wid = dev["link"].get("width")
    # line 1
    name = dev["board"] + (" " + dev["shell"] if dev["shell"] and dev["kind"] == "card" else "")
    st = dev["state"]
    st_txt, st_sty = {"launch": ("LAUNCH %.1fs" % min(9999.0, dev.get("in_flight_s") or 0.0), S_OK), "busy": ("BUSY", S_OK), "owned": ("OWNED", S_DEF), "held": ("HELD", S_WARN), "idle": ("IDLE", S_DIM),
                      "exited": ("EXITED", S_DIM)}[st]
    right = []
    if dev["kind"] == "card" and gen:
        right += [("PCIe ", S_LABEL), ("GEN ", S_MAG), ("%s@%2sx" % (gen, wid if wid else "?"), S_DEF)]
    right += [("  [", S_DEF), (st_txt, st_sty), ("]", S_DEF)]
    rlen = sum(len(t) for t, _ in right)
    left = [(" Device %d " % idx, S_LABEL), ("[%s]" % name, S_DEF)]
    if n_dev > 1:
        left.append((" (Tab: %d/%d)" % (idx + 1, n_dev), S_DIM))
    llen = sum(len(t) for t, _ in left)
    if llen + rlen + 1 > w:
        keep = max(10, w - rlen - 1 - len(" Device %d [" % idx) - 1)
        left = [(" Device %d " % idx, S_LABEL), ("[%s]" % name[:keep], S_DEF)]
    f.segs(y, 0, left)
    f.segs(y, max(0, w - rlen - 1), right)
    # line 2: the clocks (the image line, when asked for, follows them)
    probe = dev.get("probe")
    core, rv = device_clocks(dev)
    parts = [(" CORE ", S_LABEL), (core, S_DEF), ("  RV ", S_LABEL), (rv, S_DEF)]
    if v.panels["image"] and r is not None:
        img = kernel_name(r) or r["transport"]
        parts += [("   IMG ", S_LABEL), (img, S_DEF), (" ID ", S_LABEL), ("%08x" % (r["image_id"] & 0xFFFFFFFF), S_DEF),
                  (" CAPS ", S_LABEL), ("0x%x" % r["caps"], S_DEF), (" KINDS ", S_LABEL),
                  ("%d:%04x" % (r["n_kinds"], r["kinds_sig"] & 0xFFFF), S_DEF), (" ", S_DEF), (r["mode"], S_VAL)]
    elif r is None:
        if probe and probe.get("ok"):
            parts += [("   idle, lifetime launches ", S_DIM), (str(probe.get("launches", 0)), S_DIM)]
        elif st == "idle":
            parts += [("   no process owns the card", S_DIM)]
        else:
            parts += [("   held by pid %s, which does not publish" % ",".join(map(str, dev["holders"])), S_DIM)]
    elif dev["kind"] == "sim":
        parts += [("   simulated time", S_DIM)]
    f.segs(y + 1, 0, parts)
    # line 3: sensors and memory
    s = dev.get("sensors")
    sp = [(" ", S_DEF)]
    if dev["kind"] == "sim":
        sp += [("simulator: counters are in simulated time", S_DIM)]
    else:
        ok = bool(s and s.get("ok"))
        t = s.get("temp_c") if ok else None
        sp += [("TEMP ", S_LABEL)]
        if t is None:
            sp += [("n/a", S_DIM)]
        else:
            sp += [("%3.0f%s" % (t, cs.deg) + ("C" if cs.utf8 else ""), S_OK if t < 70 else (S_WARN if t < 85 else S_ERR))]
        sp += [(" FAN ", S_LABEL)]
        fan = s.get("fan_rpm") if ok else None
        sp += [("n/a" if fan is None else "%.0frpm" % fan, S_DIM if fan is None else S_DEF)]
        sp += [(" POW ", S_LABEL)]
        pw_, pm = (s.get("power_w"), s.get("power_max_w")) if ok else (None, None)
        if pw_ is None:
            sp += [("n/a", S_DIM)]
        else:
            sp += [("%3.0f / %.0f W" % (pw_, pm) if pm else "%3.0f W" % pw_, S_DEF)]
    x = f.segs(y + 2, 0, sp)
    used = r["mem_used"] if r else None
    total = r["mem_total"] if r else (probe["ch_bytes"] * probe["num_ch"] if probe and probe.get("ok") else 64 * 2 ** 30)
    bw = w - x - 2
    if bw >= 22:
        txt = "n/a" if used is None else "%s/%s" % (fmt_gib(used), fmt_gib(total))
        frac = None if used is None else used / total if total else 0.0
        draw_bar(f, y + 2, x + 2, bw, "DDR", frac, txt, S_BAR, na=used is None)


def summary_meters(f, y, w, dev):
    """COMPUTE (the busiest compute unit) and MEMORY (the busiest DDR channel against its 18 GB/s), side by side: one row."""
    rr = dev.get("rates")
    have = rr is not None and not rr.get("stale")
    bw = (w - 3) // 2
    if have:
        c, m = rr["compute"], rr["memory"]
        draw_bar(f, y, 1, bw, "COMPUTE", c["pct"], "%3.0f%% %s" % (100 * c["pct"], c["unit"]), S_BAR)
        draw_bar(f, y, 2 + bw, bw, "MEMORY", m["pct"], "%3.0f%% ch%d %.1f GB/s" % (100 * m["pct"], m["channel"], m["gbs"]), S_BAR)
    else:
        draw_bar(f, y, 1, bw, "COMPUTE", None, "", na=True)
        draw_bar(f, y, 2 + bw, bw, "MEMORY", None, "", na=True)
    return 1


def meters(f, cs, y, w, dev, v):
    """The unit bars, the channel bars, the read rate and the U line, each behind its own setup switch. Returns the rows used."""
    rr = dev.get("rates")
    rec = dev.get("rec")
    rows = 0
    have = rr is not None and not rr.get("stale")
    colw = 19 if w >= 76 else max(14, w // 4)
    ncol = max(1, w // colw)
    colw = w // ncol
    groups = []                      # lists of (label, fraction or None, text); each starts a new row

    def share(cells, label, x):
        cells.append((label, x, "" if x is None else pct(x)))

    if v.panels["meters"]:
        cells = []
        b = rr["busy"] if have else {}
        share(cells, "ACT", rr["active"] if have else None)
        for label, key in (("MVM", "mvm"), ("VU", "vu"), ("DMA", "dma"), ("QACT", "qact"), ("ATT", "att")):
            share(cells, label, b.get(key))
        share(cells, "EMPTY", rr["mvm_empty"] if have else None)
        share(cells, "WAIT", rr["mvm_wait"] if have else None)
        groups.append(cells)
    if v.panels["ddr"]:
        cells = []
        C = rec["num_ch"] if rec else 4
        for c in range(C):
            g = rr["ddr"]["ch_gbs"][c] if have and c < len(rr["ddr"]["ch_gbs"]) else None
            text = "" if g is None else "%.1fG" % g
            if g is not None and colw - 1 - len("CH%d" % c) - 2 >= 16:          # room: add the share of the stream limit
                text += " %.0f%%" % (100.0 * g / rr["ddr"]["stream_limit_gbs"])
            cells.append(("CH%d" % c, None if g is None else g / DRAM_GBS, text))
        groups.append(cells)
    for cells in groups:
        for i in range(0, len(cells), ncol):
            for j, (label, frac, text) in enumerate(cells[i:i + ncol]):
                draw_bar(f, y + rows, 1 + j * colw, colw - 1, label, frac, text, S_BAR)
            rows += 1
    if v.panels["ddr"]:
        if have:
            d = rr["ddr"]
            txt = " rd %.1f GB/s: %.0f%% of stream limit %.1f (64 B x %s), %.0f%% of DRAM %.1f" % (
                d["total_gbs"], 100 * d["pct_stream"], d["stream_limit_gbs"] * len(d["ch_gbs"]), fmt_hz(rr["core_hz"]),
                100 * d["pct_dram"], d["dram_limit_gbs"] * len(d["ch_gbs"]))
        else:
            txt = " rd n/a GB/s"
        f.segs(y + rows, 0, [(txt[:1], S_DEF), ("rd", S_LABEL), (txt[3:], S_DEF)])
        rows += 1
    if v.panels["util"]:
        if have and rr["util"]["E"] is not None:
            u = rr["util"]
            f.segs(y + rows, 0, [(" U ", S_LABEL), ("%3.0f%%" % (100 * u["U"]), S_VAL), (" ~ (1-H) ", S_LABEL),
                                 ("%.0f%%" % (100 * (1 - u["H"])), S_DEF), (" x (1-G) ", S_LABEL), ("%.0f%%" % (100 * (1 - u["G"])), S_DEF),
                                 (" x (1-F) ", S_LABEL), ("%.0f%%" % (100 * (1 - u["F"])), S_DEF), (" x E ", S_LABEL),
                                 ("%.0f%%" % (100 * u["E"]), S_DEF)])
        elif have:
            f.segs(y + rows, 0, [(" U ", S_LABEL), ("%3.0f%%" % (100 * rr["util"]["U"]), S_VAL), ("  (no MVM time in the window)", S_DIM)])
        else:
            f.segs(y + rows, 0, [(" U ", S_LABEL), ("n/a", S_DIM)])
        rows += 1
    return rows


def graph_panel(f, cs, y, h, w, dev, hist, v, interval):
    """One box over the full width: the COMPUTE share (cyan) and the MEMORY share (yellow) of wall time, 0 to 100 %, the last
    two minutes (less when the box is narrow)."""
    if h < 5 or w < 24:
        return
    box = w - 4
    pw = box - 2
    ncols = pw                                                 # one sample per cell, as nvtop
    window = min(120.0, ncols * interval)
    samples = list(hist)
    cols = []
    for c in range(ncols):
        age = (ncols - 1 - c) * window / ncols
        i = len(samples) - 1 - int(round(age / interval))
        cols.append(samples[i] if i >= 0 else None)
    comp = [None if x is None else x[0] for x in cols]
    mem = [None if x is None else x[1] for x in cols]
    rr = dev.get("rates")
    have = rr is not None and not rr.get("stale")
    if have:
        legend = [("COMPUTE %3.0f%% %s" % (100 * rr["compute"]["pct"], rr["compute"]["unit"]), S_S1),
                  ("MEMORY  %3.0f%% ch%d %.1f GB/s%s" % (100 * rr["memory"]["pct"], rr["memory"]["channel"], rr["memory"]["gbs"],
                                                       " (heavy)" if v.mono else ""), S_S2)]
    else:
        legend = [("COMPUTE n/a", S_S1), ("MEMORY  n/a%s" % (" (heavy)" if v.mono else ""), S_S2)]
    left = [(0.0, "100"), (0.25, " 75"), (0.5, " 50"), (0.75, " 25"), (1.0, "  0")]
    draw_graph(f, cs, y, 3, h, box, [(mem, 1.0, S_S2, v.mono), (comp, 1.0, S_S1, False)], legend, left, [], window)


def table(f, cs, y, h, w, procs, v):
    if h < 2:
        return
    cols = [("PID", 7, "r"), ("USER", 8, "l"), ("MODEL", 14, "l"), ("MODE", 4, "l"), ("TOK/S", 7, "r"), ("LAUNCH/s", 8, "r"),
            ("CPU", 4, "r"), ("HOST MEM", 8, "r")]
    cmd_min = 12
    widths = [c[1] for c in cols]
    while sum(widths) + len(widths) + 1 + cmd_min > w:
        for i in (2, 1, 5, 7, 6):                                  # shrink MODEL, USER, then the numbers
            if widths[i] > max(4, len(cols[i][0]) - (0 if i != 5 else 2)):
                widths[i] -= 1
                break
        else:
            break
    cmd_w = max(0, w - sum(widths) - len(widths) - 1)
    longest = max([len(p["model"]) for p in procs if p["publisher"]] + [5])
    grow = min(max(0, cmd_w - 30), max(0, min(longest, 32) - widths[2]))      # a wide screen gives the model name room
    widths[2] += grow
    cmd_w -= grow
    f.fill(y, 0, w, " ", S_HEAD)
    x = 1
    sortname = dict((k, n) for n, k in SORT_KEYS)[v.sort]
    for i, (name, _, al) in enumerate(cols):
        label = name if i != 5 or widths[i] >= 8 else "LCH/s"
        if name == sortname:
            label = (label + (" v" if v.desc else " ^"))[:widths[i]] if len(label) + 2 <= widths[i] else label
        f.put(y, x, label.rjust(widths[i]) if al == "r" else label.ljust(widths[i]), S_HEAD)
        x += widths[i] + 1
    f.put(y, x, "COMMAND" + (" v" if v.sort == "cmd" and v.desc else " ^" if v.sort == "cmd" else ""), S_HEAD)

    rows = sorted(procs, key=lambda p: (p.get(v.sort) is None, p.get(v.sort) if p.get(v.sort) is not None else 0), reverse=False)
    if v.desc:
        rows = sorted(procs, key=lambda p: (p.get(v.sort) is not None, p.get(v.sort) if p.get(v.sort) is not None else 0), reverse=True)
    avail = h - 1
    for i, p in enumerate(rows[:avail]):
        pub = p["publisher"]
        if pub:
            tok = ("pp %.1f" % p["pp_s"]) if p["pp_s"] > 0.05 and p["tok_s"] < 0.05 else "%.1f" % p["tok_s"]
            vals = [str(p["pid"]), p["user"], p["model"], p["mode"], tok, "%.1f" % p["launch_s"], "%.0f%%" % p["cpu_pct"], fmt_mem(p["rss"])]
        else:
            vals = [str(p["pid"]), p["user"], "-", "?", "-", "-", "%.0f%%" % p["cpu_pct"], fmt_mem(p["rss"])]
        x = 1
        for j, (name, _, al) in enumerate(cols):
            s = vals[j]
            if len(s) > widths[j]:
                s = shorten_mid(s, widths[j]) if j == 2 else (s[:widths[j] - 1] + "~" if widths[j] > 3 else s[:widths[j]])
            f.put(y + 1 + i, x, s.rjust(widths[j]) if al == "r" else s.ljust(widths[j]), S_DEF if pub else S_DIM)
            x += widths[j] + 1
        cmd = p["cmd"] if pub else p["cmd"] + "  (no tsr-mon)"
        f.put(y + 1 + i, x, cmd[:cmd_w], S_DEF if pub else S_DIM)
    if len(rows) > avail and avail > 0:
        f.put(y + h - 1, max(0, w - 12), " +%d more " % (len(rows) - avail), S_DIM)


def footer(f, y, w, v, interval, msg=""):
    x = 0
    keys = [("F2", "Setup"), ("F6", "Sort"), ("F10", "Quit"), ("+/-", "Refresh %.1fs" % interval), ("Tab", "Device")]
    if v.menu:
        keys = [("Esc", "Close"), ("Enter", "Select"), ("Up/Down", "Move")]
    for k, label in keys:
        x = f.segs(y, x, [(k, S_DEF), (label, S_REV), ("  ", S_DEF)])
    if msg:
        f.put(y, max(x, w - len(msg) - 1), msg, S_DIM)


def menu(f, cs, h, w, v):
    if v.menu == "setup":
        items = [("[%s] %s" % ("x" if v.panels[k] else " ", name)) for k, name in PANELS]
        items.append("[%s] Braille graphs (UTF-8)" % ("x" if v.utf8 else " "))
        title = " Setup "
    else:
        items = [("%s %s" % (name, ("v" if v.desc else "^") if key == v.sort else " ")) for name, key in SORT_KEYS]
        title = " Sort by "
    bw = max(len(title), max(len(i) for i in items)) + 4
    bh = len(items) + 2
    x0, y0 = max(0, (w - bw) // 2), max(0, (h - bh) // 2)
    f.put(y0, x0, cs.tl + title + cs.h * (bw - 2 - len(title)) + cs.tr, S_BOX)
    for i, it in enumerate(items):
        f.put(y0 + 1 + i, x0, cs.v + " " + it.ljust(bw - 3) + cs.v, S_SEL if i == v.sel else S_DEF)
    f.put(y0 + bh - 1, x0, cs.bl + cs.h * (bw - 2) + cs.br, S_BOX)


def compose(snap, hist, v, h, w, interval, msg=""):
    """The whole screen for a snapshot."""
    cs = Charset(v.utf8)
    f = Frame(h, w)
    devs = snap["devices"] if snap else []
    y = 0
    footer_y = h - 1
    if not devs:
        f.put(0, 1, "No Tesseract device: no xocl card found and no process publishes a tsr-mon record in %s." % (snap or {}).get("mon_dir", "/dev/shm"), S_DIM)
        f.put(2, 1, "Start a run with TSR_DEVICE=sim or on the card; the owner of the card publishes after every launch.", S_DIM)
        footer(f, footer_y, w, v, interval, msg)
        if v.menu:
            menu(f, cs, h, w, v)
        return f
    if not v.dev_manual:
        v.dev = next((i for i, d in enumerate(devs) if d.get("publisher")), 0)
    v.dev = max(0, min(v.dev, len(devs) - 1))
    dev = devs[v.dev]
    device_header(f, cs, 0, w, dev, v.dev, len(devs), v)
    y = 3
    procs = [p for p in snap["processes"] if p["device"] == dev["id"] and (p["publisher"] or v.panels["others"])]
    if v.panels["summary"]:
        y += summary_meters(f, y, w, dev)
    if v.panels["meters"] or v.panels["ddr"] or v.panels["util"]:
        y += meters(f, cs, y, w, dev, v)
    remaining = footer_y - y
    table_rows = max(len(procs), 1) + 1 if v.panels["table"] else 0
    table_h = min(table_rows, max(3, remaining // 3)) if v.panels["table"] else 0
    graph_h = remaining - table_h
    if v.panels["graph"] and graph_h >= 6:
        graph_panel(f, cs, y, graph_h, w, dev, hist.get(dev["id"], []), v, interval)
        y += graph_h
    else:
        table_h = min(table_rows, remaining)
    if v.panels["table"] and table_h > 0:
        table(f, cs, y, table_h, w, procs, v)
    footer(f, footer_y, w, v, interval, msg)
    if v.menu:
        menu(f, cs, h, w, v)
    return f


# ---------------------------------------------------------------------------------------------
# curses
# ---------------------------------------------------------------------------------------------
class Theme:
    def __init__(self, color):
        self.attr = {}
        self.colored = bool(color and curses.has_colors())
        if self.colored:
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = fg_default = -1
            except curses.error:
                bg, fg_default = curses.COLOR_BLACK, curses.COLOR_WHITE
            spec = {S_LABEL: (curses.COLOR_CYAN, 0), S_VAL: (fg_default, curses.A_BOLD), S_BAR: (curses.COLOR_GREEN, 0),
                    S_BAR2: (curses.COLOR_YELLOW, 0), S_BAR3: (curses.COLOR_RED, 0), S_MAG: (curses.COLOR_MAGENTA, 0),
                    S_S1: (curses.COLOR_CYAN, 0), S_S2: (curses.COLOR_YELLOW, 0), S_S3: (curses.COLOR_GREEN, 0),
                    S_WARN: (curses.COLOR_YELLOW, 0), S_ERR: (curses.COLOR_RED, curses.A_BOLD), S_OK: (curses.COLOR_GREEN, 0),
                    S_DIM: (fg_default, curses.A_DIM)}
            for i, (st, (fg, extra)) in enumerate(spec.items(), start=1):
                try:
                    curses.init_pair(i, fg, bg)
                    self.attr[st] = curses.color_pair(i) | extra
                except curses.error:
                    pass
            self.attr[S_HEAD] = curses.A_REVERSE
            self.attr[S_REV] = curses.A_REVERSE
            self.attr[S_SEL] = curses.A_REVERSE
        else:
            self.attr = {S_LABEL: curses.A_BOLD, S_VAL: curses.A_BOLD, S_REV: curses.A_REVERSE, S_HEAD: curses.A_REVERSE,
                         S_SEL: curses.A_REVERSE, S_DIM: curses.A_DIM, S_ERR: curses.A_BOLD | curses.A_REVERSE, S_WARN: curses.A_BOLD,
                         S_MAG: curses.A_BOLD}

    def get(self, st):
        return self.attr.get(st, curses.A_NORMAL)


def blit(scr, theme, f):
    scr.erase()
    for y in range(f.h):
        x = 0
        row, sty = f.ch[y], f.st[y]
        while x < f.w:
            x2 = x
            while x2 < f.w and sty[x2] == sty[x]:
                x2 += 1
            run = "".join(row[x:x2])
            if run.strip() or sty[x] in (S_HEAD, S_REV, S_SEL):
                try:
                    scr.addstr(y, x, run, theme.get(sty[x]))
                except curses.error:
                    pass                              # the bottom right cell
            x = x2
    scr.refresh()


def handle_key(k, v, mon, state):
    """Returns False to quit."""
    if v.menu:
        n = len(PANELS) + 1 if v.menu == "setup" else len(SORT_KEYS)
        if k in (27, ord("q"), curses.KEY_F2 if v.menu == "setup" else curses.KEY_F6):
            v.menu = None
        elif k in (curses.KEY_UP, ord("k")):
            v.sel = (v.sel - 1) % n
        elif k in (curses.KEY_DOWN, ord("j")):
            v.sel = (v.sel + 1) % n
        elif k in (10, 13, ord(" "), curses.KEY_ENTER):
            if v.menu == "setup":
                if v.sel < len(PANELS):
                    key = PANELS[v.sel][0]
                    v.panels[key] = not v.panels[key]
                else:
                    v.utf8 = not v.utf8
            else:
                key = SORT_KEYS[v.sel][1]
                if key == v.sort:
                    v.desc = not v.desc
                else:
                    v.sort, v.desc = key, key in ("tok_s", "launch_s", "cpu_pct", "rss")
                v.menu = None
        return True
    if k in (ord("q"), ord("Q"), curses.KEY_F10):
        return False
    if k in (curses.KEY_F2, ord("S"), ord("s")):
        v.menu, v.sel = "setup", 0
    elif k in (curses.KEY_F6, ord("o"), ord("O")):
        v.menu = "sort"
        v.sel = [key for _, key in SORT_KEYS].index(v.sort)
    elif k in (ord("+"), ord("=")):
        mon.interval = min(10.0, round(mon.interval + (0.5 if mon.interval >= 1.0 else 0.1), 2))
    elif k in (ord("-"), ord("_")):
        mon.interval = max(0.1, round(mon.interval - (0.5 if mon.interval > 1.0 else 0.1), 2))
    elif k in (9, ord("d")):
        v.dev_manual = True
        v.dev += 1
        if mon.snap and v.dev >= len(mon.snap["devices"]):
            v.dev = 0
    return True


def run_curses(stdscr, mon, v, opts):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    theme = Theme(v.color)
    v.mono = not theme.colored
    stdscr.keypad(True)
    stdscr.timeout(100)
    last = -1e9
    snap = None
    while True:
        now = time.monotonic()
        if now - last >= mon.interval:
            snap = mon.poll()
            last = now
            dirty = True
        else:
            dirty = False
        h, w = stdscr.getmaxyx()
        if dirty or state_changed(v):
            blit(stdscr, theme, compose(snap, mon.hist, v, h, w, mon.interval))
        k = stdscr.getch()
        if k == -1:
            continue
        if k == curses.KEY_RESIZE:
            curses.update_lines_cols()
            stdscr.erase()
            blit(stdscr, theme, compose(snap, mon.hist, v, *stdscr.getmaxyx(), mon.interval))
            continue
        if not handle_key(k, v, mon, None):
            return
        snap = mon.snap
        h, w = stdscr.getmaxyx()
        blit(stdscr, theme, compose(snap, mon.hist, v, h, w, mon.interval))
        if k in (ord("+"), ord("="), ord("-"), ord("_")):
            last = min(last, now - mon.interval)       # take effect at once


_last_view = [None]


def state_changed(v):
    key = (v.menu, v.sel, tuple(sorted(v.panels.items())), v.sort, v.desc, v.dev, v.utf8)
    if key != _last_view[0]:
        _last_view[0] = key
        return True
    return False


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(prog="tsr-top", description="Live monitor for the Tesseract card, laid out like nvtop. "
                                 "Reads the records that the process owning the card publishes (TSR_MON=0 turns the publisher off).")
    ap.add_argument("--once", action="store_true", help="print one frame as plain text and exit (no curses)")
    ap.add_argument("--json", action="store_true", help="print one snapshot as JSON and exit")
    ap.add_argument("--size", default=None, metavar="WxH", help="frame size for --once (default: the terminal, else 100x36)")
    ap.add_argument("--window", type=float, default=1.0, metavar="S", help="seconds between the two samples of --once / --json (default 1)")
    ap.add_argument("-d", "--delay", type=float, default=None, metavar="S", help="refresh interval in seconds (default 1)")
    ap.add_argument("--dir", default=None, help="record directory (default $TSR_MON_DIR or /dev/shm)")
    ap.add_argument("--include-dead", action="store_true", help="also show records of dead pids (TSR_MON_KEEP leftovers; tests)")
    ap.add_argument("--no-sensors", action="store_true", help="do not read the board sensors (sysfs, xbutil)")
    ap.add_argument("--sensor-interval", type=float, default=10.0, metavar="S",
                    help="seconds between xbutil runs, which give the power limit and what sysfs lacks (default 10)")
    ap.add_argument("--xbutil", default=os.environ.get("TSR_TOP_XBUTIL", XBUTIL), help="path of xbutil")
    ap.add_argument("--probe", action="store_true", help="open an idle card briefly in attach mode (tsr-mon-probe) when no process holds it "
                    "and the xclbin on the card is TSR_XCLBIN; off by default because a run that starts during the probe cannot open the card")
    ap.add_argument("--probe-interval", type=float, default=15.0, metavar="S", help="seconds between probes (default 15)")
    ap.add_argument("--probe-bin", default=None, help="path of tsr-mon-probe (default: next to tsr-top)")
    ap.add_argument("--xclbin", default=os.environ.get("TSR_XCLBIN"), help="the xclbin on the card, for --probe (default $TSR_XCLBIN)")
    ap.add_argument("--no-color", action="store_true", help="monochrome")
    ap.add_argument("--ascii", action="store_true", help="no UTF-8 box drawing or line glyphs")
    ap.add_argument("--no-background", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--layout", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--window-min", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--token-window", type=float, default=3.0, metavar="S", help="seconds the tokens/s and launches/s rates average over (default 3)")
    o = ap.parse_args(argv)
    if o.probe_bin is None:
        d = os.environ.get("TSR_TOP_BIN_DIR") or os.path.dirname(os.path.abspath(__file__))
        o.probe_bin = os.path.join(d, "tsr-mon-probe")
    return o


def main(argv=None):
    opts = parse_args(argv if argv is not None else sys.argv[1:])
    if opts.layout:
        print(json.dumps(layout_json()))
        return 0
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        pass
    utf8 = "UTF-8" in (locale.getpreferredencoding(False) or "").upper() or "utf" in (locale.getpreferredencoding(False) or "").lower()
    if not utf8:
        opts.ascii = True
    mon = Monitor(opts)
    if opts.delay:
        mon.interval = max(0.1, opts.delay)
    v = View(opts)
    if opts.once or opts.json:
        mon.opts.no_background = False
        mon.start()
        if mon.bg.is_alive():
            mon.bg.first_pass.wait(30)                  # sysfs, xbutil and the card's users, once
        mon.poll()
        time.sleep(max(0.0, opts.window))
        snap = mon.poll()
        if opts.probe or not opts.no_sensors:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                snap = mon.poll()
                need = False
                for d in snap["devices"]:
                    if d["kind"] == "card" and not opts.no_sensors and d["sensors"] is None:
                        need = True
                    if d["kind"] == "card" and opts.probe and d["probe"] is None:
                        need = True
                if not need:
                    break
                time.sleep(0.3)
        mon.stop()
        if opts.json:
            out = dict(snap, devices=[{k: v for k, v in d.items() if k != "rec"} for d in snap["devices"]])   # the record is in the process
            print(json.dumps(out, indent=1, default=str))
            return 0
        w, h = 100, 36
        if opts.size:
            m = re.fullmatch(r"(\d+)x(\d+)", opts.size)
            if not m:
                print("tsr-top: --size must be WxH", file=sys.stderr)
                return 2
            w, h = int(m.group(1)), int(m.group(2))
        elif sys.stdout.isatty():
            w, h = os.get_terminal_size()
        v.mono = True                                   # a text frame has no colors: the memory series is heavy
        print(compose(snap, mon.hist, v, h, w, mon.interval).text())
        return 0
    if not sys.stdout.isatty():
        print("tsr-top: stdout is not a terminal; use --once or --json", file=sys.stderr)
        return 2
    mon.start()
    os.environ.setdefault("ESCDELAY", "25")           # Esc closes a menu at once
    try:
        curses.wrapper(run_curses, mon, v, opts)
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

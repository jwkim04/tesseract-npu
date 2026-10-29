"""cocotb model of the core's per-channel memory ports (spec 코어 경계 인터페이스; docs/rtl-m2.md §4).

One MemModel serves all NUM_CH channels of a DUT whose memory ports follow the tsr_core naming
(packed per-channel vectors, channel c at slice [c]):

    <p>rd_req_valid/ready, <p>rd_req_addr, <p>rd_req_len, <p>rd_req_id        (core -> mem)
    <p>rd_rsp_valid/ready, <p>rd_rsp_data, <p>rd_rsp_id, <p>rd_rsp_last       (mem -> core)
    <p>wr_req_valid/ready, <p>wr_req_addr, <p>wr_req_len, <p>wr_req_id        (core -> mem)
    <p>wr_dat_valid/ready, <p>wr_dat_data, <p>wr_dat_last                     (core -> mem)
    <p>wr_rsp_valid/ready, <p>wr_rsp_id                                       (mem -> core)

Behaviour (seeded, per channel): random request acceptance (backpressure on rd_req, wr_req,
wr_dat), random response latency per burst, random gaps in rd_rsp/wr_rsp valid, read data in
order within one id and interleaved at beat granularity across ids, write responses in order per
id and reordered across ids, random stall phases. Write ordering modes per channel ("wr_order"):
  any        request and data accepted independently
  req_first  data beats are accepted only for requests already accepted
  data_first a request is accepted only after at least one of its data beats was accepted
  random     a mode is drawn per channel every few hundred cycles
req_first / data_first deadlock a core that waits for one channel's handshake before offering the
other (forbidden by docs/rtl-m2.md rule M5) and turn that into a timeout.

Protocol assertions (raise AssertionError in the model task, which fails the cocotb test; all
violations are also collected in .errors):
  - read/write burst length 1..64, inside one 64-beat (4 KiB) block, inside the channel limit
  - at most rd_outstanding read requests outstanding per channel (accept -> last beat accepted)
  - valid stays high and the payload stable until the handshake (core-driven channels)
  - write data: exactly len beats per request, wr_dat_last on (and only on) the last beat
Tracking: .written[c] (beats written), .read[c] (beats read), .read_unwritten[c] (beats read
before any write by this run), .src_beats[c][s] (read beats per source id[3:2]: 0 AUX, 1 DMA,
2 WS; the unit benches check DMA = LD beats and WS = MVM weight beats).

Timing discipline: the model samples DUT outputs at the falling clock edge and updates/drives its
own outputs right after the rising edge. Drive DUT inputs the same way in other TB components.
"""
from __future__ import annotations

import logging
import random
from collections import deque

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge

from . import isa

_MODES = ("any", "req_first", "data_first")


class ProtocolError(AssertionError):
    pass


class _Burst:
    __slots__ = ("id", "addr", "len", "sent", "due", "seq")

    def __init__(self, id_, addr, len_, due, seq):
        self.id, self.addr, self.len, self.sent, self.due, self.seq = id_, addr, len_, 0, due, seq


class _Chan:
    def __init__(self, idx: int, rng: random.Random):
        self.idx = idx
        self.rng = rng
        self.mem = {}                       # beat -> 512-bit int
        # read side
        self.rd_bursts = []                 # accepted, unfinished, in acceptance order
        self.rsp_burst = None               # burst whose beat is presented (rsp_valid = 1)
        self.rsp_valid = 0
        self.rsp_data = 0
        self.rsp_id = 0
        self.rsp_last = 0
        self.outstanding = 0
        self.max_outstanding = 0
        # write side
        self.wreqs = deque()                # accepted write requests [id, addr, len]
        self.wdata = deque()                # accepted data beats (data, last)
        self.wrsp = []                      # pending write responses _Burst-like (id, due, seq)
        self.wrsp_cur = None
        self.wrsp_valid = 0
        # ready outputs for the coming edge
        self.rd_req_ready = 0
        self.wr_req_ready = 0
        self.wr_dat_ready = 0
        # stability tracking of DUT-driven channels: payload offered without handshake
        self.pend_rd_req = None
        self.pend_wr_req = None
        self.pend_wr_dat = None
        # tracking
        self.written = set()
        self.read = set()
        self.read_unwritten = set()
        self.stall = 0
        self.mode = "any"
        self.mode_left = 0
        self.stats = dict(rd_bursts=0, rd_beats=0, wr_bursts=0, wr_beats=0)
        self.src_beats = [0, 0, 0, 0]       # read beats per source id[3:2] (AUX, DMA, WS, -)


class MemModel:
    def __init__(self, dut, *, num_ch: int, clk=None, rst_n=None, prefix: str = "m_",
                 beat_bits: int = 512, addr_w: int = 28, id_w: int = isa.MEM_ID_W,
                 len_w: int = isa.LEN_W, rd_outstanding: int = 16, limit_beats: int | None = None,
                 seed: int = 1, rd_lat=(1, 40), wr_lat=(1, 40), p_rd_req_ready: float = 0.7,
                 p_rd_rsp_valid: float = 0.8, p_wr_req_ready: float = 0.7,
                 p_wr_dat_ready: float = 0.7, p_wr_rsp_valid: float = 0.8,
                 p_stall: float = 0.01, stall_len=(1, 30), interleave: bool = True,
                 wr_order: str = "random", strict: bool = True, name: str = "mem"):
        self.dut = dut
        self.clk = clk if clk is not None else dut.clk
        self.rst_n = rst_n if rst_n is not None else getattr(dut, "rst_n", None)
        self.num_ch = num_ch
        self.beat_bits = beat_bits
        self.addr_w, self.id_w, self.len_w = addr_w, id_w, len_w
        self.rd_outstanding = rd_outstanding
        self.limit = limit_beats if limit_beats is not None else (1 << addr_w)
        self.rd_lat, self.wr_lat = rd_lat, wr_lat
        self.p_rd_req_ready, self.p_rd_rsp_valid = p_rd_req_ready, p_rd_rsp_valid
        self.p_wr_req_ready, self.p_wr_dat_ready = p_wr_req_ready, p_wr_dat_ready
        self.p_wr_rsp_valid = p_wr_rsp_valid
        self.p_stall, self.stall_len = p_stall, stall_len
        self.interleave = interleave
        if wr_order not in _MODES + ("random",):
            raise ValueError("wr_order must be one of %s or 'random'" % (_MODES,))
        self.wr_order = wr_order
        self.strict = strict
        self.log = logging.getLogger("tsr.%s" % name)
        self.ch = [_Chan(c, random.Random((seed * 1000003) ^ (c * 7919 + 17))) for c in range(num_ch)]
        self.errors = []
        self.cycle = 0
        self._seq = 0
        self._task = None
        self._last_out = {}
        g = lambda n: getattr(dut, prefix + n)  # noqa: E731
        self.s_rd_req_valid, self.s_rd_req_ready = g("rd_req_valid"), g("rd_req_ready")
        self.s_rd_req_addr, self.s_rd_req_len, self.s_rd_req_id = g("rd_req_addr"), g("rd_req_len"), g("rd_req_id")
        self.s_rd_rsp_valid, self.s_rd_rsp_ready = g("rd_rsp_valid"), g("rd_rsp_ready")
        self.s_rd_rsp_data, self.s_rd_rsp_id, self.s_rd_rsp_last = g("rd_rsp_data"), g("rd_rsp_id"), g("rd_rsp_last")
        self.s_wr_req_valid, self.s_wr_req_ready = g("wr_req_valid"), g("wr_req_ready")
        self.s_wr_req_addr, self.s_wr_req_len, self.s_wr_req_id = g("wr_req_addr"), g("wr_req_len"), g("wr_req_id")
        self.s_wr_dat_valid, self.s_wr_dat_ready = g("wr_dat_valid"), g("wr_dat_ready")
        self.s_wr_dat_data, self.s_wr_dat_last = g("wr_dat_data"), g("wr_dat_last")
        self.s_wr_rsp_valid, self.s_wr_rsp_ready, self.s_wr_rsp_id = g("wr_rsp_valid"), g("wr_rsp_ready"), g("wr_rsp_id")
        self._drive_idle()

    # -------------------------------------------------------------------------------------------
    # Host-side access
    # -------------------------------------------------------------------------------------------
    def load(self, ch: int, byte_off: int, data: bytes):
        """Host write of bytes into channel ch (any alignment; partial beats are merged).
        Not tracked as a write of the run."""
        m = self.ch[ch].mem
        pos, end = byte_off, byte_off + len(data)
        while pos < end:
            beat, lo = divmod(pos, 64)
            n = min(64 - lo, end - pos)
            chunk = data[pos - byte_off:pos - byte_off + n]
            if n == 64:
                v = int.from_bytes(chunk, "little")
            else:
                old = bytearray(m.get(beat, 0).to_bytes(64, "little"))
                old[lo:lo + n] = chunk
                v = int.from_bytes(old, "little")
            if v:
                m[beat] = v
            else:
                m.pop(beat, None)
            pos += n

    def read_bytes(self, ch: int, byte_off: int, size: int) -> bytes:
        if byte_off % 64 or size % 64:
            raise ValueError("read_bytes: unaligned region")
        m = self.ch[ch].mem
        b0 = byte_off // 64
        return b"".join(m.get(b0 + i, 0).to_bytes(64, "little") for i in range(size // 64))

    def beat(self, ch: int, beat: int) -> int:
        return self.ch[ch].mem.get(beat, 0)

    @property
    def written(self):
        return [c.written for c in self.ch]

    @property
    def read(self):
        return [c.read for c in self.ch]

    @property
    def read_unwritten(self):
        return [c.read_unwritten for c in self.ch]

    def clear_tracking(self):
        for c in self.ch:
            c.written.clear()
            c.read.clear()
            c.read_unwritten.clear()
            c.src_beats = [0, 0, 0, 0]

    @property
    def src_beats(self):
        """Per channel: read beats delivered per memory-side source (id[3:2]: 0 AUX, 1 DMA, 2 WS)
        since the model was created or clear_tracking()."""
        return [list(c.src_beats) for c in self.ch]

    # -------------------------------------------------------------------------------------------
    # Run control
    # -------------------------------------------------------------------------------------------
    def start(self):
        if self._task is None:
            self._task = cocotb.start_soon(self._run())
        return self

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def idle(self) -> bool:
        """No read or write transaction in flight on any channel."""
        return all(not c.rd_bursts and not c.wreqs and not c.wdata and not c.wrsp and not c.wrsp_valid
                   for c in self.ch)

    def check(self, idle: bool = True):
        """Raise if any protocol violation was seen (and, with idle=True, if traffic is in flight)."""
        if self.errors:
            raise ProtocolError("%d memory protocol violation(s), first: %s" % (len(self.errors), self.errors[0]))
        if idle and not self.idle():
            busy = ["ch%d: %d rd bursts %s, %d wr reqs, %d wr beats, %d wr rsps" % (
                c.idx, len(c.rd_bursts), ["id%d@0x%x+%d sent %d" % (b.id, b.addr, b.len, b.sent)
                                          for b in c.rd_bursts[:4]],
                len(c.wreqs), len(c.wdata), len(c.wrsp) + c.wrsp_valid)
                for c in self.ch if c.rd_bursts or c.wreqs or c.wdata or c.wrsp or c.wrsp_valid]
            raise ProtocolError("memory traffic still in flight: " + "; ".join(busy))

    def stats(self) -> dict:
        return {c.idx: dict(c.stats, max_outstanding=c.max_outstanding) for c in self.ch}

    # -------------------------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------------------------
    def _fail(self, msg: str):
        msg = "cycle %d: %s" % (self.cycle, msg)
        self.errors.append(msg)
        self.log.error(msg)
        if self.strict:
            raise ProtocolError(msg)

    @staticmethod
    def _bit(v: int, i: int) -> int:
        return (v >> i) & 1

    def _field(self, v: int, i: int, w: int) -> int:
        return (v >> (i * w)) & ((1 << w) - 1)

    def _check_burst(self, kind: str, c: int, addr: int, ln: int):
        if ln < 1 or ln > isa.MAX_BURST_BEATS:
            self._fail("ch%d %s burst length %d not in 1..64 (addr 0x%x)" % (c, kind, ln, addr))
        if (addr % isa.BURST_ALIGN) + ln > isa.BURST_ALIGN:
            self._fail("ch%d %s burst [0x%x, +%d) crosses a 4 KiB boundary" % (c, kind, addr, ln))
        if addr + ln > self.limit:
            self._fail("ch%d %s burst [0x%x, +%d) beyond the channel limit 0x%x" % (c, kind, addr, ln, self.limit))

    def _sample(self) -> dict:
        s = {"rst": self.rst_n is not None and int(self.rst_n.value) == 0}
        s["rqv"] = int(self.s_rd_req_valid.value)
        if s["rqv"]:
            s["rqa"] = int(self.s_rd_req_addr.value)
            s["rql"] = int(self.s_rd_req_len.value)
            s["rqi"] = int(self.s_rd_req_id.value)
        s["rsr"] = int(self.s_rd_rsp_ready.value)
        s["wqv"] = int(self.s_wr_req_valid.value)
        if s["wqv"]:
            s["wqa"] = int(self.s_wr_req_addr.value)
            s["wql"] = int(self.s_wr_req_len.value)
            s["wqi"] = int(self.s_wr_req_id.value)
        s["wdv"] = int(self.s_wr_dat_valid.value)
        if s["wdv"]:
            s["wdd"] = int(self.s_wr_dat_data.value)
            s["wdl"] = int(self.s_wr_dat_last.value)
        s["wsr"] = int(self.s_wr_rsp_ready.value)
        return s

    async def _run(self):
        clk = self.clk
        while True:
            await FallingEdge(clk)
            s = self._sample()
            await RisingEdge(clk)
            self.cycle += 1
            if s["rst"]:
                self._drive_idle()
                continue
            for ch in self.ch:
                self._update(ch, s)
            for ch in self.ch:
                self._choose(ch)
            self._drive()

    def _update(self, ch: _Chan, s: dict):
        c = ch.idx
        aw, lw, iw, bb = self.addr_w, self.len_w, self.id_w, self.beat_bits
        # ---- read request ----
        if self._bit(s["rqv"], c):
            p = (self._field(s["rqa"], c, aw), self._field(s["rql"], c, lw), self._field(s["rqi"], c, iw))
            if ch.pend_rd_req is not None and p != ch.pend_rd_req:
                self._fail("ch%d rd_req payload changed before the handshake: %s -> %s" % (c, ch.pend_rd_req, p))
            if ch.rd_req_ready:
                ch.pend_rd_req = None
                addr, ln, id_ = p
                self._check_burst("read", c, addr, ln)
                if ch.outstanding >= self.rd_outstanding:
                    self._fail("ch%d read request beyond RD_OUTSTANDING=%d" % (c, self.rd_outstanding))
                self._seq += 1
                ch.rd_bursts.append(_Burst(id_, addr, max(ln, 1), self.cycle + ch.rng.randint(*self.rd_lat), self._seq))
                ch.outstanding += 1
                ch.max_outstanding = max(ch.max_outstanding, ch.outstanding)
                ch.stats["rd_bursts"] += 1
            else:
                ch.pend_rd_req = p
        elif ch.pend_rd_req is not None:
            self._fail("ch%d rd_req_valid dropped before the handshake" % c)
            ch.pend_rd_req = None
        # ---- read response beat ----
        if ch.rsp_valid and self._bit(s["rsr"], c):
            b = ch.rsp_burst
            beat = b.addr + b.sent
            ch.read.add(beat)
            if beat not in ch.written:
                ch.read_unwritten.add(beat)
            b.sent += 1
            ch.stats["rd_beats"] += 1
            ch.src_beats[(b.id >> 2) & 3] += 1
            if b.sent == b.len:
                ch.rd_bursts.remove(b)
                ch.outstanding -= 1
            ch.rsp_valid = 0
            ch.rsp_burst = None
        # ---- write request ----
        if self._bit(s["wqv"], c):
            p = (self._field(s["wqa"], c, aw), self._field(s["wql"], c, lw), self._field(s["wqi"], c, iw))
            if ch.pend_wr_req is not None and p != ch.pend_wr_req:
                self._fail("ch%d wr_req payload changed before the handshake: %s -> %s" % (c, ch.pend_wr_req, p))
            if ch.wr_req_ready:
                ch.pend_wr_req = None
                self._check_burst("write", c, p[0], p[1])
                ch.wreqs.append([p[2], p[0], max(p[1], 1)])
                ch.stats["wr_bursts"] += 1
            else:
                ch.pend_wr_req = p
        elif ch.pend_wr_req is not None:
            self._fail("ch%d wr_req_valid dropped before the handshake" % c)
            ch.pend_wr_req = None
        # ---- write data ----
        if self._bit(s["wdv"], c):
            p = (self._field(s["wdd"], c, bb), self._bit(s["wdl"], c))
            if ch.pend_wr_dat is not None and p != ch.pend_wr_dat:
                self._fail("ch%d wr_dat payload changed before the handshake" % c)
            if ch.wr_dat_ready:
                ch.pend_wr_dat = None
                ch.wdata.append(p)
                ch.stats["wr_beats"] += 1
            else:
                ch.pend_wr_dat = p
        elif ch.pend_wr_dat is not None:
            self._fail("ch%d wr_dat_valid dropped before the handshake" % c)
            ch.pend_wr_dat = None
        # ---- commit complete write bursts ----
        while ch.wreqs and len(ch.wdata) >= ch.wreqs[0][2]:
            id_, addr, ln = ch.wreqs.popleft()
            for i in range(ln):
                data, last = ch.wdata.popleft()
                if last != (i == ln - 1):
                    self._fail("ch%d write burst at 0x%x len %d: wr_dat_last=%d on beat %d" % (c, addr, ln, last, i))
                beat = addr + i
                if beat < self.limit:
                    if data:
                        ch.mem[beat] = data
                    else:
                        ch.mem.pop(beat, None)
                    ch.written.add(beat)
            self._seq += 1
            ch.wrsp.append(_Burst(id_, addr, ln, self.cycle + ch.rng.randint(*self.wr_lat), self._seq))
        # ---- write response ----
        if ch.wrsp_valid and self._bit(s["wsr"], c):
            ch.wrsp.remove(ch.wrsp_cur)
            ch.wrsp_cur = None
            ch.wrsp_valid = 0

    def _choose(self, ch: _Chan):
        rng = self.rng_of(ch)
        # stall phases: everything quiet for a while
        if ch.stall > 0:
            ch.stall -= 1
        elif rng.random() < self.p_stall:
            ch.stall = rng.randint(*self.stall_len)
        stalled = ch.stall > 0
        # write ordering mode
        if self.wr_order == "random":
            if ch.mode_left <= 0:
                ch.mode = rng.choice(_MODES)
                ch.mode_left = rng.randint(50, 400)
            ch.mode_left -= 1
        else:
            ch.mode = self.wr_order
        ch.rd_req_ready = int(not stalled and rng.random() < self.p_rd_req_ready)
        need = sum(r[2] for r in ch.wreqs) - len(ch.wdata)
        if ch.mode == "req_first":
            ch.wr_req_ready = int(not stalled and rng.random() < self.p_wr_req_ready)
            ch.wr_dat_ready = int(not stalled and need > 0 and rng.random() < self.p_wr_dat_ready)
        elif ch.mode == "data_first":
            ch.wr_req_ready = int(not stalled and need < 0 and rng.random() < self.p_wr_req_ready)
            ch.wr_dat_ready = int(not stalled and rng.random() < self.p_wr_dat_ready)
        else:
            ch.wr_req_ready = int(not stalled and rng.random() < self.p_wr_req_ready)
            ch.wr_dat_ready = int(not stalled and rng.random() < self.p_wr_dat_ready)
        # read response: hold the presented beat until accepted
        if not ch.rsp_valid and not stalled and ch.rd_bursts and rng.random() < self.p_rd_rsp_valid:
            cand = self._eligible(ch.rd_bursts)
            if not self.interleave:
                started = [b for b in cand if b.sent > 0]
                cand = started or cand
            if cand:
                b = rng.choice(cand)
                beat = b.addr + b.sent
                ch.rsp_burst = b
                ch.rsp_valid = 1
                ch.rsp_data = ch.mem.get(beat, 0) if beat < self.limit else 0
                ch.rsp_id = b.id
                ch.rsp_last = int(b.sent == b.len - 1)
        # write response
        if not ch.wrsp_valid and not stalled and ch.wrsp and rng.random() < self.p_wr_rsp_valid:
            cand = self._eligible(ch.wrsp)
            if cand:
                ch.wrsp_cur = rng.choice(cand)
                ch.wrsp_valid = 1

    def rng_of(self, ch: _Chan) -> random.Random:
        return ch.rng

    def _eligible(self, items):
        """Oldest pending item of each id whose latency has elapsed (same-id order kept)."""
        seen, out = set(), []
        for b in items:
            if b.id in seen:
                continue
            seen.add(b.id)
            if b.due <= self.cycle:
                out.append(b)
        return out

    def _put(self, sig, key, val):
        if self._last_out.get(key) != val:
            sig.value = val
            self._last_out[key] = val

    def _drive(self):
        rqr = rsv = rsl = wqr = wdr = wsv = 0
        rsd = rsi = wsi = 0
        for ch in self.ch:
            c = ch.idx
            rqr |= ch.rd_req_ready << c
            wqr |= ch.wr_req_ready << c
            wdr |= ch.wr_dat_ready << c
            if ch.rsp_valid:
                rsv |= 1 << c
                rsl |= ch.rsp_last << c
                rsd |= ch.rsp_data << (c * self.beat_bits)
                rsi |= ch.rsp_id << (c * self.id_w)
            if ch.wrsp_valid:
                wsv |= 1 << c
                wsi |= ch.wrsp_cur.id << (c * self.id_w)
        self._put(self.s_rd_req_ready, "rqr", rqr)
        self._put(self.s_rd_rsp_valid, "rsv", rsv)
        self._put(self.s_rd_rsp_last, "rsl", rsl)
        self._put(self.s_rd_rsp_data, "rsd", rsd)
        self._put(self.s_rd_rsp_id, "rsi", rsi)
        self._put(self.s_wr_req_ready, "wqr", wqr)
        self._put(self.s_wr_dat_ready, "wdr", wdr)
        self._put(self.s_wr_rsp_valid, "wsv", wsv)
        self._put(self.s_wr_rsp_id, "wsi", wsi)

    def _drive_idle(self):
        """Reset: drop every transaction in flight and drive all model outputs low."""
        for ch in self.ch:
            ch.rd_bursts.clear()
            ch.outstanding = 0
            ch.wreqs.clear()
            ch.wdata.clear()
            ch.wrsp.clear()
            ch.rd_req_ready = ch.wr_req_ready = ch.wr_dat_ready = 0
            ch.rsp_valid = 0
            ch.rsp_burst = None
            ch.wrsp_valid = 0
            ch.wrsp_cur = None
            ch.pend_rd_req = ch.pend_wr_req = ch.pend_wr_dat = None
        self._last_out.clear()
        for sig in (self.s_rd_req_ready, self.s_rd_rsp_valid, self.s_rd_rsp_last, self.s_rd_rsp_data,
                    self.s_rd_rsp_id, self.s_wr_req_ready, self.s_wr_dat_ready, self.s_wr_rsp_valid,
                    self.s_wr_rsp_id):
            sig.value = 0

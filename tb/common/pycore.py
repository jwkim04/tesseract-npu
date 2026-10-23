"""Python reference of the core's architectural behaviour, executing over a real memory port.

Purpose: (1) prove the trace-replay harness (trace loading, MemModel, replay.compare) end to end
before the RTL core exists, (2) give the block agents an executable statement of docs/rtl-m2.md
§3.1, §3.2, §7 (fetch rules, burst splitting, every architectural check and its order) and of
docs/rtl-m3.md §4 / §6.3 (MVM weight streams: id {WS, 0}, bursts, exactly rows_c * ROW_BEATS
beats per channel; all y computed first, then written per token with ACC and zero tails).
It is not an oracle — the simulator is — and it is checked against the traces like the RTL.

SeqMaster drives the master side of a memory port (tb/common/hdl/tsr_tb_loop.sv src_* signals)
one burst at a time; PyCore fetches and executes NOP END LI ADDI LD ST QACT FENCE and (has_mvm,
the default) MVM with it. has_mvm=False decodes MVM as ILLEGAL_OP (an M2 core, HAS_MVM = 0).
"""
from __future__ import annotations

from cocotb.triggers import FallingEdge, RisingEdge

from . import isa
from .tsr_ref import ref


class SeqMaster:
    """Sequential bursts on a tsr_core-style memory port (prefix `src_`, packed per channel)."""

    def __init__(self, dut, num_ch: int, addr_w: int, prefix: str = "src_", beat_bits: int = 512):
        self.dut, self.n, self.aw, self.bb = dut, num_ch, addr_w, beat_bits
        g = lambda name: getattr(dut, prefix + name)  # noqa: E731
        self.rq_v, self.rq_r, self.rq_a, self.rq_l, self.rq_i = (g("rd_req_valid"), g("rd_req_ready"),
                                                                 g("rd_req_addr"), g("rd_req_len"), g("rd_req_id"))
        self.rs_v, self.rs_r, self.rs_d, self.rs_i, self.rs_l = (g("rd_rsp_valid"), g("rd_rsp_ready"),
                                                                 g("rd_rsp_data"), g("rd_rsp_id"), g("rd_rsp_last"))
        self.wq_v, self.wq_r, self.wq_a, self.wq_l, self.wq_i = (g("wr_req_valid"), g("wr_req_ready"),
                                                                 g("wr_req_addr"), g("wr_req_len"), g("wr_req_id"))
        self.wd_v, self.wd_r, self.wd_d, self.wd_l = g("wr_dat_valid"), g("wr_dat_ready"), g("wr_dat_data"), g("wr_dat_last")
        self.ws_v, self.ws_r, self.ws_i = g("wr_rsp_valid"), g("wr_rsp_ready"), g("wr_rsp_id")
        for s in (self.rq_v, self.rs_r, self.wq_v, self.wd_v, self.ws_r):
            s.value = 0
        self.beats_read = [0] * num_ch

    @staticmethod
    def bursts(addr: int, n: int):
        """docs/rtl-m2.md D1: <= 64 beats, never across a 64-beat (4 KiB) block."""
        while n:
            ln = min(n, isa.BURST_ALIGN - addr % isa.BURST_ALIGN)
            yield addr, ln
            addr += ln
            n -= ln

    async def read(self, ch: int, addr: int, n: int, id_: int = 4) -> list:
        out = []
        for a, ln in self.bursts(addr, n):
            await RisingEdge(self.dut.clk)
            self.rq_a.value = a << (ch * self.aw)
            self.rq_l.value = ln << (ch * isa.LEN_W)
            self.rq_i.value = id_ << (ch * isa.MEM_ID_W)
            self.rq_v.value = 1 << ch
            self.rs_r.value = 1 << ch                     # space is reserved: always ready
            got = 0
            req_done = False
            while got < ln:
                await FallingEdge(self.dut.clk)
                rq_ok = not req_done and (int(self.rq_r.value) >> ch) & 1
                rs_ok = (int(self.rs_v.value) >> ch) & 1
                data = (int(self.rs_d.value) >> (ch * self.bb)) & ((1 << self.bb) - 1) if rs_ok else 0
                last = (int(self.rs_l.value) >> ch) & 1 if rs_ok else 0
                rid = (int(self.rs_i.value) >> (ch * isa.MEM_ID_W)) & 0xF if rs_ok else 0
                await RisingEdge(self.dut.clk)
                if rq_ok:
                    req_done = True
                    self.rq_v.value = 0
                if rs_ok:
                    assert rid == id_, "read response id %d, expected %d" % (rid, id_)
                    got += 1
                    assert last == (got == ln), "rd_rsp_last on beat %d of %d" % (got, ln)
                    out.append(data)
            self.rs_r.value = 0
            self.beats_read[ch] += ln
        return out

    async def write(self, ch: int, addr: int, beats: list, id_: int = 4):
        i = 0
        for a, ln in self.bursts(addr, len(beats)):
            await RisingEdge(self.dut.clk)
            self.wq_a.value = a << (ch * self.aw)
            self.wq_l.value = ln << (ch * isa.LEN_W)
            self.wq_i.value = id_ << (ch * isa.MEM_ID_W)
            self.wq_v.value = 1 << ch                    # request and data offered independently (M5)
            k = 0
            self.wd_d.value = beats[i] << (ch * self.bb)
            self.wd_l.value = int(ln == 1) << ch
            self.wd_v.value = 1 << ch
            self.ws_r.value = 1 << ch
            req_done = rsp_done = False
            while not (req_done and k == ln and rsp_done):
                await FallingEdge(self.dut.clk)
                rq_ok = not req_done and (int(self.wq_r.value) >> ch) & 1
                wd_ok = k < ln and (int(self.wd_r.value) >> ch) & 1
                ws_ok = (int(self.ws_v.value) >> ch) & 1
                await RisingEdge(self.dut.clk)
                if rq_ok:
                    req_done = True
                    self.wq_v.value = 0
                if wd_ok:
                    k += 1
                    if k < ln:
                        self.wd_d.value = beats[i + k] << (ch * self.bb)
                        self.wd_l.value = int(k == ln - 1) << ch
                    else:
                        self.wd_v.value = 0
                if ws_ok:
                    assert req_done and k == ln, "write response before the burst was complete"
                    rsp_done = True
            self.ws_r.value = 0
            i += ln


class PyCore:
    """Architectural model of the core (one instruction at a time), simulator semantics."""

    WS_ID = isa.SRC_WS << isa.SRC_ID_W          # memory-side id of a weight stream: {2, 0} = 8

    def __init__(self, cfg: dict, master: SeqMaster, has_mvm: bool = True):
        self.cfg, self.m = cfg, master
        self.has_mvm = has_mvm
        self.spm_lines = cfg["spm_bytes"] // 64
        self.lim = isa.dram_limit(cfg["ch_bytes"], cfg["addr_beat_w"])
        self.spm = bytearray(b"\xA5" * cfg["spm_bytes"])
        self.R = [0] * 16
        self.retired = 0
        self.mvm_beats = [0] * cfg["num_ch"]    # weight beats read per channel (PERF reference)

    async def run(self, cmd_addr: int, cmd_count: int, args) -> dict:
        R = self.R
        R[:] = [0] * 8 + [a & 0xFFFFFFFF for a in args]
        cur, beat_data = None, 0
        pc = 0
        while True:
            if pc >= cmd_count:
                return self._fin(isa.ERR_ILLEGAL_OP, cmd_count, 0)
            b = cmd_addr + pc // 2
            if cmd_addr >= self.lim or b >= self.lim:
                return self._fin(isa.ERR_DRAM_RANGE, pc, b & 0xFFFFFFFF)
            if b != cur:
                beat_data = (await self.m.read(0, b, 1, id_=0))[0]
                cur = b
            w = [(beat_data >> (256 * (pc % 2) + 32 * k)) & 0xFFFFFFFF for k in range(8)]
            code, info, end = await self.execute(w)
            if code:
                return self._fin(code, pc, info)
            self.retired += 1
            if end:
                return dict(status=isa.STATUS_DONE, err_code=0, err_pc=0, err_info=0)
            pc += 1

    @staticmethod
    def _fin(code, pc, info):
        return dict(status=isa.STATUS_DONE | isa.STATUS_ERR, err_code=code, err_pc=pc & 0xFFFFFFFF,
                    err_info=info & 0xFFFFFFFF)

    def _spm_ok(self, a, e):
        return a + e <= self.spm_lines

    async def execute(self, w):
        """Returns (err_code, err_info, is_end). Checks exactly as docs/rtl-m2.md §7."""
        R, w0 = self.R, w[0]
        f = isa.w0_fields(w0)
        op = f["opcode"]
        zero_from = lambda k: all(x == 0 for x in w[k:])  # noqa: E731
        if op in (isa.OP_NOP, isa.OP_FENCE, isa.OP_END):
            if w0 & 0xFFFFFF00 or not zero_from(1):
                return isa.ERR_ILLEGAL_OP, 0, False
            return 0, 0, op == isa.OP_END
        if op == isa.OP_LI:
            if w0 & 0x0FFFFF00 or not zero_from(2):
                return isa.ERR_ILLEGAL_OP, 0, False
            if f["rd"]:
                R[f["rd"]] = w[1]
            return 0, 0, False
        if op == isa.OP_ADDI:
            if w0 & 0x0FF0FF00 or not zero_from(2):
                return isa.ERR_ILLEGAL_OP, 0, False
            if f["rd"]:
                R[f["rd"]] = (R[f["ra"]] + w[1]) & 0xFFFFFFFF
            return 0, 0, False
        if op in (isa.OP_LD, isa.OP_ST):
            if w0 & 0xFF00F000 or not zero_from(4) or f["sub"] >= self.cfg["num_ch"]:
                return isa.ERR_ILLEGAL_OP, 0, False
            ch = f["sub"]
            spm = (R[f["ra"]] + w[1]) & 0xFFFFFFFF
            dram = (R[f["rb"]] + w[2]) & 0xFFFFFFFF
            n = w[3]
            if not self._spm_ok(spm, n):
                return isa.ERR_SPM_RANGE, spm, False
            if dram + n > self.lim:
                return isa.ERR_DRAM_RANGE, dram, False
            if n:
                if op == isa.OP_LD:
                    beats = await self.m.read(ch, dram, n, id_=(isa.SRC_DMA << 2))
                    for i, v in enumerate(beats):
                        self.spm[64 * (spm + i):64 * (spm + i + 1)] = v.to_bytes(64, "little")
                else:
                    beats = [int.from_bytes(self.spm[64 * (spm + i):64 * (spm + i + 1)], "little") for i in range(n)]
                    await self.m.write(ch, dram, beats, id_=(isa.SRC_DMA << 2))
            return 0, 0, False
        if op == isa.OP_QACT:
            if w0 & 0xFF00F000 or not zero_from(6) or f["sub"] >= 2:
                return isa.ERR_ILLEGAL_OP, 0, False
            kind, K, T = f["sub"], w[3], w[4]
            bs = isa.qa_block_size(kind)
            if K == 0 or K % bs or K > self.cfg["k_max"]:
                return isa.ERR_K_RANGE, 0, False
            if T == 0 or T > self.cfg["t_max"]:
                return isa.ERR_T_RANGE, 0, False
            xl = K // 16
            st = w[5] or xl
            ql = isa.qa_lines(kind, K)
            dst = (R[f["ra"]] + w[1]) & 0xFFFFFFFF
            src = (R[f["rb"]] + w[2]) & 0xFFFFFFFF
            if not self._spm_ok(dst, T * ql):
                return isa.ERR_SPM_RANGE, dst, False
            if not self._spm_ok(src, (T - 1) * st + xl):
                return isa.ERR_SPM_RANGE, src, False
            for t in range(T):
                s0 = 64 * (src + t * st)
                img = ref().qact(kind, bytes(self.spm[s0:s0 + 4 * K]), K)
                d0 = 64 * (dst + t * ql)
                self.spm[d0:d0 + len(img)] = img
            return 0, 0, False
        if op == isa.OP_MVM and self.has_mvm:
            return await self.mvm(w, f)
        return isa.ERR_ILLEGAL_OP, 0, False      # MVM with has_mvm=False and every non-M3 opcode

    async def mvm(self, w, f):
        """tsr_sim.c ex_mvm: checks in order, weight streams per channel (docs/rtl-m3.md §4), all y
        first, then per token: ACC = fadd(old, y), zero tail (docs/rtl-m3.md §6.3)."""
        R, r = self.R, ref()
        if w[0] & 0xF000E000 or w[5] & 0xFFF00000 or w[7]:
            return isa.ERR_ILLEGAL_OP, 0, False
        fmt = f["sub"]
        if fmt >= 2:
            return isa.ERR_BAD_FMT, 0, False
        acc = f["flags"] & isa.MVM_FLAG_ACC
        N, K, T = w[4], w[5] & 0xFFFF, (w[5] >> 16) & 0xF
        bs = 256 if fmt else 32
        if K == 0 or K % bs or K > self.cfg["k_max"]:
            return isa.ERR_K_RANGE, 0, False
        if T == 0 or T > self.cfg["t_max"]:
            return isa.ERR_T_RANGE, 0, False
        ql = isa.qa_lines(fmt, K)                 # kind == fmt (Q8_0 -> A32, Q4_K -> A256)
        yl = (N + 15) // 16
        ds = w[6] or yl
        dst = (R[f["ra"]] + w[1]) & 0xFFFFFFFF
        x = (R[f["rb"]] + w[2]) & 0xFFFFFFFF
        woff = (R[f["rc"]] + w[3]) & 0xFFFFFFFF
        if not self._spm_ok(dst, (T - 1) * ds + yl):
            return isa.ERR_SPM_RANGE, dst, False
        if not self._spm_ok(x, T * ql):
            return isa.ERR_SPM_RANGE, x, False
        nch = self.cfg["num_ch"]
        Rr = -(-N // nch)
        rb = r.row_beats(fmt, K)
        if woff + Rr * rb > self.lim:
            return isa.ERR_DRAM_RANGE, woff, False
        # weight streams: channel c reads its rows_c * ROW_BEATS beats from W_OFF (one id, bursts)
        rows = []
        for c in range(nch):
            rc = max(0, min(Rr, N - c * Rr))
            if rc == 0:
                continue
            beats = await self.m.read(c, woff, rc * rb, id_=self.WS_ID)
            self.mvm_beats[c] += rc * rb
            blob = b"".join(v.to_bytes(64, "little") for v in beats)
            rows += [blob[64 * rb * i:64 * rb * (i + 1)] for i in range(rc)]
        assert len(rows) == N
        qas = [bytes(self.spm[64 * (x + t * ql):64 * (x + (t + 1) * ql)]) for t in range(T)]
        ys = [[r.dot_row(fmt, rows[n], K, qas[t]) for n in range(N)] for t in range(T)]
        for t in range(T):
            base = 64 * (dst + t * ds)
            for n in range(N):
                v = ys[t][n]
                if acc:
                    v = r.fadd(int.from_bytes(self.spm[base + 4 * n:base + 4 * n + 4], "little"), v)
                self.spm[base + 4 * n:base + 4 * n + 4] = v.to_bytes(4, "little")
            self.spm[base + 4 * N:base + 64 * yl] = bytes(64 * yl - 4 * N)
        return 0, 0, False

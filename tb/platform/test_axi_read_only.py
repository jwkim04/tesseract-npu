"""The read-only AXI bridge of the second weight stream (tsr_xrt_axi_bridge with READ_ONLY = 1, docs/platform-u200.md §4).

read_only_bridge  random read traffic through the bridge against a random AXI slave: every AR is base + 64 * addr, LEN - 1,
                  SIZE 6, INCR, ID, cache 4'b0011; the slave returns the bursts' beats interleaved between IDs (in order per
                  ID) with random gaps, some with RRESP = SLVERR; the core side sees every beat with its ID and LAST, in
                  order, and ev_rresp_err pulses once per error beat. R beats for an ID with no outstanding burst are not
                  forwarded and give one ev_unexpected each. The write half does not exist: the write request, data and
                  response ports never accept or answer, AWVALID and WVALID stay 0 whatever the write inputs do (they are
                  driven randomly), and ev_bresp_err stays 0.

Run for SLICES 1, 2 and 4 (the kernel's AXI_SLICES values) and two seeds.
"""
import os
import random
from collections import defaultdict, deque

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from common.runner import REPO, run

BEAT = 512
ID_W = 4
NID = 1 << ID_W
KMUL = (0x9E3779B97F4A7C15F39CC0605CEDC8341082276BF3A27251F86C6A11D0C18E95 << 256 |
        0xD1B54A32D192ED03A0761D6478BD642F7FEB352D5851F42D4C957F2D1B6A2D3B) | 1


def beat_data(abs_beat):
    return ((abs_beat + 1) * KMUL) & ((1 << BEAT) - 1)


@cocotb.test()
async def read_only_bridge(dut):
    seed = int(os.environ.get("TSR_SEED", "1"))
    rnd = random.Random(seed)
    base = 0x1_2345_6000
    nreq = 300
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    zero = ("c_rd_req_valid", "c_rd_req_addr", "c_rd_req_len", "c_rd_req_id", "c_rd_rsp_ready", "m_axi_arready",
            "m_axi_rvalid", "m_axi_rdata", "m_axi_rresp", "m_axi_rlast", "m_axi_rid", "c_wr_req_valid", "c_wr_req_addr",
            "c_wr_req_len", "c_wr_req_id", "c_wr_dat_valid", "c_wr_dat_data", "c_wr_dat_mask", "c_wr_dat_last",
            "c_wr_rsp_ready", "m_axi_awready", "m_axi_wready", "m_axi_bvalid", "m_axi_bresp", "m_axi_bid")
    for n in zero:
        getattr(dut, n).value = 0
    dut.base.value = base
    resets = ("rst_n", "rst_axi_n", "rst_rd_n", "rst_wr_n", "rst_rdc_n", "rst_wrc_n")
    for n in resets:
        getattr(dut, n).value = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
    for n in resets:
        getattr(dut, n).value = (1 << len(getattr(dut, n))) - 1 if n.endswith(("rdc_n", "wrc_n")) else 1
    for _ in range(4):
        await RisingEdge(dut.clk)

    reqs = deque()                         # (id, addr, len) still to issue, in order
    for _ in range(nreq):
        ln = rnd.choice([1, 1, 2, 3, 8, 16, 31, 63, 64, rnd.randint(1, 64)])
        addr = (rnd.randrange(1 << 14) << 6) + rnd.randint(0, 64 - ln)     # inside one 64-beat block
        reqs.append((rnd.randrange(NID), addr, ln))
    issued = []                            # core-side handshakes
    pending_ar = []                        # (id, abs first beat, len) accepted at AXI, in AR order
    bursts = []                            # slave: dict(id, beats [(data, resp)], sent)
    exp_beats = defaultdict(deque)         # per ID: (data, last, resp) expected at the core side
    got = defaultdict(list)
    n_err_fwd = 0
    ev_err = ev_unx = 0
    inject = []                            # ids to inject as unexpected R beats once idle
    injected = 0
    bad = []
    r_hold = None                          # an R beat on the bus not yet accepted by the bridge
    r_beat_kind = r_prev_kind = r_burst = None
    cycles = 0
    quiet = 0
    while cycles < 200000:
        cycles += 1
        await FallingEdge(dut.clk)
        # ---- sample (inputs were driven after the last rising edge) ----
        if int(dut.ev_rresp_err.value):
            ev_err += 1
        if int(dut.ev_unexpected.value):
            ev_unx += 1
        if int(dut.ev_bresp_err.value):
            bad.append("ev_bresp_err pulsed")
        if int(dut.m_axi_awvalid.value) or int(dut.m_axi_wvalid.value):
            bad.append("AWVALID / WVALID high on the read-only bridge")
        if int(dut.c_wr_req_ready.value) or int(dut.c_wr_dat_ready.value) or int(dut.c_wr_rsp_valid.value):
            bad.append("the core-side write port accepted or answered")
        if int(dut.c_rd_req_valid.value) and int(dut.c_rd_req_ready.value):
            rid, addr, ln = reqs.popleft()
            issued.append((rid, addr, ln))
        if int(dut.m_axi_arvalid.value) and int(dut.m_axi_arready.value):
            a, ln, rid = int(dut.m_axi_araddr.value), int(dut.m_axi_arlen.value) + 1, int(dut.m_axi_arid.value)
            if not issued:
                bad.append("AR without a core request")
            else:
                eid, eaddr, eln = issued.pop(0)
                if (a, ln, rid) != (base + 64 * eaddr, eln, eid):
                    bad.append("AR %#x len %d id %d, expected %#x len %d id %d" % (a, ln, rid, base + 64 * eaddr, eln, eid))
                if (int(dut.m_axi_arsize.value), int(dut.m_axi_arburst.value), int(dut.m_axi_arlock.value),
                        int(dut.m_axi_arcache.value), int(dut.m_axi_arprot.value), int(dut.m_axi_arqos.value)) != (6, 1, 0, 3, 0, 0):
                    bad.append("AR attributes")
                beats = []
                for k in range(ln):
                    resp = 2 if rnd.random() < 0.03 else 0
                    beats.append((beat_data(a // 64 + k), resp))
                    exp_beats[rid].append((beat_data(a // 64 + k), k == ln - 1, resp))
                bursts.append(dict(id=rid, beats=beats, sent=0))
        if int(dut.m_axi_rvalid.value) and int(dut.m_axi_rready.value):
            r_hold = None
            if r_beat_kind == "real":
                b = r_burst
                b["sent"] += 1
                if b["sent"] == len(b["beats"]):
                    bursts.remove(b)
        if int(dut.c_rd_rsp_valid.value) and int(dut.c_rd_rsp_ready.value):
            rid = int(dut.c_rd_rsp_id.value)
            got[rid].append((int(dut.c_rd_rsp_data.value), int(dut.c_rd_rsp_last.value)))
            if not exp_beats[rid]:
                bad.append("a forwarded beat for ID %d with nothing outstanding" % rid)
            else:
                d, last, resp = exp_beats[rid].popleft()
                n_err_fwd += resp != 0
                if (d, last) != got[rid][-1]:
                    bad.append("beat of ID %d: data / last differ" % rid)
        # ---- drive for the next cycle (after the rising edge) ----
        await RisingEdge(dut.clk)
        dut.c_rd_req_valid.value = int(bool(reqs) and rnd.random() < 0.7)
        if reqs:
            rid, addr, ln = reqs[0]
            dut.c_rd_req_addr.value = addr
            dut.c_rd_req_len.value = ln
            dut.c_rd_req_id.value = rid
        dut.c_rd_rsp_ready.value = int(rnd.random() < 0.75)
        dut.m_axi_arready.value = int(rnd.random() < 0.6)
        # the write half: random stimulus that has to be ignored
        for n, w in (("c_wr_req_valid", 1), ("c_wr_dat_valid", 1), ("c_wr_dat_last", 1), ("c_wr_rsp_ready", 1),
                     ("m_axi_awready", 1), ("m_axi_wready", 1), ("m_axi_bvalid", 1)):
            getattr(dut, n).value = int(rnd.random() < 0.5)
        dut.c_wr_req_addr.value = rnd.getrandbits(14)
        dut.c_wr_req_len.value = rnd.randint(1, 64)
        dut.c_wr_req_id.value = rnd.randrange(NID)
        dut.c_wr_dat_data.value = rnd.getrandbits(BEAT)
        dut.c_wr_dat_mask.value = rnd.getrandbits(16)
        dut.m_axi_bresp.value = rnd.randrange(4)
        dut.m_axi_bid.value = rnd.randrange(NID)
        # the R channel: hold a beat until accepted, else pick the next one
        r_beat_kind = None
        if r_hold is None and bursts and rnd.random() < 0.8:
            # the first burst of each ID in AR order may deliver; choose one of them
            first = {}
            for b in bursts:
                first.setdefault(b["id"], b)
            r_burst = rnd.choice(list(first.values()))
            k = r_burst["sent"]
            d, resp = r_burst["beats"][k]
            r_hold = (d, resp, r_burst["id"], k == len(r_burst["beats"]) - 1)
            r_beat_kind = "real"
        elif r_hold is None and not bursts and not reqs and not issued and not any(exp_beats.values()) and injected < 3:
            # idle: an R beat for an ID that has nothing outstanding (a response of an earlier reset epoch)
            r_hold = (rnd.getrandbits(BEAT), 0, rnd.randrange(NID), True)
            r_beat_kind = "stray"
            injected += 1
        elif r_hold is not None:
            r_beat_kind = r_prev_kind
        if r_hold is not None:
            d, resp, rid, last = r_hold
            dut.m_axi_rvalid.value = 1
            dut.m_axi_rdata.value = d
            dut.m_axi_rresp.value = resp
            dut.m_axi_rid.value = rid
            dut.m_axi_rlast.value = int(last)
        else:
            dut.m_axi_rvalid.value = 0
        r_prev_kind = r_beat_kind
        if (not reqs and not issued and not bursts and not any(exp_beats.values()) and r_hold is None
                and injected >= 3):
            quiet += 1
            if quiet > 40:
                break
        else:
            quiet = 0
    else:
        bad.append("the run did not finish (%d requests left, %d bursts)" % (len(reqs), len(bursts)))
    n_beats = sum(len(v) for v in got.values())
    if ev_err != n_err_fwd:
        bad.append("ev_rresp_err pulses %d, error beats forwarded %d" % (ev_err, n_err_fwd))
    if ev_unx != injected:
        bad.append("ev_unexpected pulses %d, stray beats injected %d" % (ev_unx, injected))
    dut._log.info("%d requests, %d beats forwarded, %d error beats, %d stray beats, %d cycles",
                  nreq, n_beats, n_err_fwd, injected, cycles)
    for x in bad[:20]:
        dut._log.error(x)
    assert not bad, "%d mismatches" % len(bad)


@pytest.mark.parametrize("slices", [1, 2, 4])
@pytest.mark.parametrize("seed", [1, 7])
def test_axi_read_only(slices, seed):
    platform = REPO / "rtl/platform/alveo_xrt"
    run("tsr_xrt_axi_bridge", __name__, parameters={"SLICES": slices, "READ_ONLY": 1}, seed=seed,
        name="axi_read_only_s%d_seed%d" % (slices, seed),
        extra_sources=[platform / name for name in ("tsr_xrt_slices.sv", "tsr_xrt_axi_bridge.sv")],
        extra_env={"TSR_SEED": str(seed)})

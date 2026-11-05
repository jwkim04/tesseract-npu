"""Write mask and ordering checks on the kernel's AXI data bridge."""

import cocotb
from cocotb.triggers import Timer

from common.runner import REPO, run


@cocotb.test()
async def independent_aw_w_and_b(dut):
    def value(name):
        return int(getattr(dut, name).value)

    def drive(**signals):
        for name, val in signals.items():
            getattr(dut, name).value = val

    async def settle():
        dut.clk.value = 0
        await Timer(5, unit="ns")

    async def edge():
        dut.clk.value = 1
        await Timer(5, unit="ns")
        dut.clk.value = 0

    base = 0x4000
    reset_names = ("rst_n", "rst_axi_n", "rst_rd_n", "rst_wr_n", "rst_rdc_n", "rst_wrc_n")
    drive(**dict.fromkeys(reset_names, 0))
    drive(base=base, c_rd_req_valid=0, c_rd_req_addr=0, c_rd_req_len=0, c_rd_req_id=0,
          c_rd_rsp_ready=1, m_axi_arready=0, m_axi_rvalid=0, m_axi_rdata=0, m_axi_rresp=0,
          m_axi_rlast=0, m_axi_rid=0, c_wr_req_valid=0, c_wr_req_addr=0, c_wr_req_len=0,
          c_wr_req_id=0, c_wr_dat_valid=0, c_wr_dat_data=0, c_wr_dat_last=0, c_wr_dat_mask=0,
          c_wr_rsp_ready=0, m_axi_awready=0, m_axi_wready=0, m_axi_bvalid=0, m_axi_bresp=0, m_axi_bid=0)
    for _ in range(4):
        await settle()
        await edge()
    drive(**{name: (1 << len(getattr(dut, name))) - 1 if name in reset_names[-2:] else 1
             for name in reset_names})

    # W handshakes before AW for the first burst. AW handshakes before W
    # for the second. Alternate WREADY afterwards to stall both mask shapes.
    for data_first, ident, address in [(True, 2, 32), (False, 3, 64)]:
        masks = [0x003C, 0x8000, 0x8421, 0xFFFF]
        data = [int.from_bytes(bytes([0x10 * ident + i]) * 64, "little") for i in range(len(masks))]
        aw_sent = aw_seen = False
        w_sent = w_seen = 0
        held_aw = held_w = None
        aw_cycle = first_w_cycle = None
        for cycle in range(100):
            aw_ready = cycle >= (18 if data_first else 5)
            w_ready = cycle >= (6 if data_first else 18) and cycle % 2 == 0
            wi = min(w_sent, len(data) - 1)
            drive(c_wr_req_valid=int(not aw_sent), c_wr_req_addr=address, c_wr_req_len=len(data),
                  c_wr_req_id=ident, c_wr_dat_valid=int(w_sent < len(data)), c_wr_dat_data=data[wi],
                  c_wr_dat_mask=masks[wi], c_wr_dat_last=int(wi == len(data) - 1),
                  m_axi_awready=int(aw_ready), m_axi_wready=int(w_ready))
            await settle()
            aw = (value("m_axi_awaddr"), value("m_axi_awlen"), value("m_axi_awid"))
            w = (value("m_axi_wdata"), value("m_axi_wstrb"), value("m_axi_wlast"))
            if held_aw is not None:
                assert value("m_axi_awvalid") and aw == held_aw
            if held_w is not None:
                assert value("m_axi_wvalid") and w == held_w
            held_aw = aw if value("m_axi_awvalid") and not aw_ready else None
            held_w = w if value("m_axi_wvalid") and not w_ready else None
            if value("m_axi_awvalid") and aw_ready:
                assert not aw_seen and aw == (base + 64 * address, len(data) - 1, ident)
                assert value("m_axi_awsize") == 6 and value("m_axi_awburst") == 1
                aw_seen, aw_cycle = True, cycle
            if value("m_axi_wvalid") and w_ready:
                assert w_seen < len(data)
                expected_strobe = sum(0xF << (4 * word) for word in range(16)
                                      if masks[w_seen] & (1 << word))
                assert w == (data[w_seen], expected_strobe, int(w_seen == len(data) - 1))
                if first_w_cycle is None:
                    first_w_cycle = cycle
                w_seen += 1
            aw_sent |= bool(value("c_wr_req_valid") and value("c_wr_req_ready"))
            w_sent += bool(value("c_wr_dat_valid") and value("c_wr_dat_ready"))
            assert not value("c_wr_rsp_valid")
            await edge()
            if aw_seen and w_seen == len(data):
                break
        assert aw_sent and aw_seen and w_sent == w_seen == len(data)
        assert (first_w_cycle < aw_cycle) == data_first

    # Both IDs are outstanding. AXI may return different IDs out of order.
    drive(c_wr_req_valid=0, c_wr_dat_valid=0)
    for ident in (3, 2):
        b_sent = False
        held_response = False
        for cycle in range(32):
            drive(m_axi_bvalid=int(not b_sent), m_axi_bid=ident, c_wr_rsp_ready=int(cycle >= 12))
            await settle()
            b_sent |= bool(value("m_axi_bvalid") and value("m_axi_bready"))
            if held_response:
                assert value("c_wr_rsp_valid")
            if value("c_wr_rsp_valid"):
                assert value("c_wr_rsp_id") == ident
                held_response = not value("c_wr_rsp_ready")
                if value("c_wr_rsp_ready"):
                    await edge()
                    break
            await edge()
        else:
            raise AssertionError("missing write response")
        assert b_sent


def test_axi_write_mask():
    platform = REPO / "rtl/platform/alveo_xrt"
    run("tsr_xrt_axi_bridge", __name__, parameters={"SLICES": 2},
        extra_sources=[platform / name for name in ("tsr_xrt_slices.sv", "tsr_xrt_axi_bridge.sv")])

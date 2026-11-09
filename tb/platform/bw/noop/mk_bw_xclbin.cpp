// mk_bw_xclbin.cpp — write metadata-only xclbins of the bandwidth kernel tsr_bw_k4 for XRT's
// no-op shim (XCL_EMULATION_MODE=noop: no card, no driver), so that sw/tools/tsr_bw's XRT path —
// xclbin parsing, kernel / argument lookup and offset checks, clock topology, device-only buffers
// per memory group, xrt::run argument sizes, start / wait — runs on the dev server
// (tb/platform/bw/noop/noop_smoke.sh; docs/bandwidth-kernel.md §6). The section layout follows
// sw/transport/tests/test_xrt_noop.cpp (EMBEDDED_METADATA kernel XML, IP_LAYOUT, MEM_TOPOLOGY with
// four DDR4 banks, CONNECTIVITY) plus CLOCK_FREQ_TOPOLOGY (DATA clock 287 MHz, KERNEL clock 500).
//
//   mk_bw_xclbin <out.xclbin> [good|bad_offset|bad_size|no_arg|range|two_cu]
//
// Build: g++ -std=c++17 -I<XRT root>/include mk_bw_xclbin.cpp -o mk_bw_xclbin

#include <xclbin.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string kernel_xml(const std::string & variant) {
    const unsigned range = variant == "range" ? 0x800 : 0x1000;
    std::string x = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<project name=\"tsr_bw_noop\">\n"
                    "<platform vendor=\"xilinx\" boardid=\"u200\" name=\"noop\" featureRomTime=\"0\">\n"
                    "<version major=\"1\" minor=\"0\"/>\n"
                    "<device name=\"fpga0\" fpgaDevice=\"virtexuplus:xcu200:fsgd2104:-2:e\" addrWidth=\"0\">\n"
                    "<core name=\"OCL_REGION_0\" target=\"hw\" type=\"clc_region\" clockFreq=\"300MHz\" numComputeUnits=\"1\">\n";
    char b[512];
    snprintf(b, sizeof b,
             "<kernel name=\"tsr_bw_k4\" language=\"ip_c\" vlnv=\"tesseract:kernel:tsr_bw_k4:1.0\" preferredWorkGroupSizeMultiple=\"0\" "
             "workGroupSize=\"1\" interrupt=\"true\" hwControlProtocol=\"ap_ctrl_hs\">\n"
             "<port name=\"s_axi_control\" mode=\"slave\" range=\"0x%x\" dataWidth=\"32\" portType=\"addressable\" base=\"0x0\"/>\n",
             range);
    x += b;
    for (unsigned c = 0; c < 4; ++c) {
        snprintf(b, sizeof b,
                 "<port name=\"m_axi_ch%u\" mode=\"master\" range=\"0xFFFFFFFFFFFFFFFF\" dataWidth=\"512\" portType=\"addressable\" base=\"0x0\"/>\n", c);
        x += b;
    }
    int id = 0;
    auto arg = [&](const std::string & name, const std::string & port, unsigned off, unsigned size, int aq, const char * type) {
        snprintf(b, sizeof b,
                 "<arg name=\"%s\" addressQualifier=\"%d\" id=\"%d\" port=\"%s\" size=\"0x%x\" offset=\"0x%x\" hostOffset=\"0x0\" "
                 "hostSize=\"0x%x\" type=\"%s\"/>\n",
                 name.c_str(), aq, id++, port.c_str(), size, off, size, type);
        x += b;
    };
    // the args of rtl/platform/alveo_xrt/bw/kernel_bw_k4.xml (docs/bandwidth-kernel.md §2)
    arg("seed", "s_axi_control", 0x10, 4, 0, "uint");
    for (unsigned c = 0; c < 4; ++c) {
        const unsigned base = 0x20 + 0x20 * c;
        arg("ctrl" + std::to_string(c), "s_axi_control", base, 4, 0, "uint");
        if (variant == "bad_size" && c == 2) arg("bytes2", "s_axi_control", base + 8, 4, 0, "uint");
        else arg("bytes" + std::to_string(c), "s_axi_control", base + 8, 8, 0, "ulong");
        if (variant == "bad_offset" && c == 1) arg("rd_off1", "s_axi_control", base + 0x14, 8, 0, "ulong");
        else arg("rd_off" + std::to_string(c), "s_axi_control", base + 0x10, 8, 0, "ulong");
        if (!(variant == "no_arg" && c == 3)) arg("wr_off" + std::to_string(c), "s_axi_control", base + 0x18, 8, 0, "ulong");
    }
    for (unsigned c = 0; c < 4; ++c) arg("ch" + std::to_string(c), "m_axi_ch" + std::to_string(c), 0xA0 + 8 * c, 8, 1, "void*");
    const int ncu = variant == "two_cu" ? 2 : 1;
    for (int i = 0; i < ncu; ++i) {
        snprintf(b, sizeof b, "<instance name=\"tsr_bw_k4_%d\"><addrRemap base=\"0x%x\" range=\"0x%x\" port=\"s_axi_control\"/></instance>\n",
                 i + 1, 0x1800000 + 0x10000 * i, range);
        x += b;
    }
    x += "</kernel>\n</core>\n</device>\n</platform>\n</project>\n";
    return x;
}

int nargs(const std::string & variant) { return variant == "no_arg" ? 20 : 21; }

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: mk_bw_xclbin <out.xclbin> [good|bad_offset|bad_size|no_arg|range|two_cu]\n");
        return 2;
    }
    const std::string path = argv[1], variant = argc > 2 ? argv[2] : "good";
    const std::string xml = kernel_xml(variant);
    const int ncu = variant == "two_cu" ? 2 : 1;
    std::vector<uint8_t> ipl(sizeof(ip_layout) + sizeof(ip_data) * (ncu - 1));
    {
        ip_layout * l = (ip_layout *) ipl.data();
        memset(l, 0, ipl.size());
        l->m_count = ncu;
        for (int i = 0; i < ncu; ++i) {
            l->m_ip_data[i].m_type = IP_KERNEL;
            l->m_ip_data[i].properties = (AP_CTRL_HS << IP_CONTROL_SHIFT) | IP_INT_ENABLE_MASK;
            l->m_ip_data[i].m_base_address = 0x1800000 + 0x10000 * i;
            snprintf((char *) l->m_ip_data[i].m_name, sizeof(l->m_ip_data[i].m_name), "tsr_bw_k4:tsr_bw_k4_%d", i + 1);
        }
    }
    std::vector<uint8_t> mem(sizeof(mem_topology) + sizeof(mem_data) * 3);
    {
        mem_topology * m = (mem_topology *) mem.data();
        m->m_count = 4;
        for (unsigned c = 0; c < 4; ++c) {
            mem_data & d = m->m_mem_data[c];
            memset(&d, 0, sizeof(d));
            d.m_type = MEM_DDR4;
            d.m_used = 1;
            d.m_size = 256 * 1024;                                // KiB: 256 MiB banks (the noop shim memsets buffers)
            d.m_base_address = 0x4000000000ull + 0x400000000ull * c;
            snprintf((char *) d.m_tag, sizeof(d.m_tag), "DDR[%u]", c);
        }
    }
    const int na = nargs(variant);
    std::vector<uint8_t> con(sizeof(connectivity) + sizeof(connection) * (4 * ncu - 1));
    {
        connectivity * k = (connectivity *) con.data();
        k->m_count = 4 * ncu;
        for (int i = 0; i < ncu; ++i)
            for (int c = 0; c < 4; ++c) k->m_connection[4 * i + c] = connection{na - 4 + c, i, c};
    }
    std::vector<uint8_t> clk(sizeof(clock_freq_topology) + sizeof(clock_freq));
    {
        clock_freq_topology * t = (clock_freq_topology *) clk.data();
        memset(t, 0, clk.size());
        t->m_count = 2;
        t->m_clock_freq[0].m_freq_Mhz = 287;
        t->m_clock_freq[0].m_type = CT_DATA;
        snprintf(t->m_clock_freq[0].m_name, sizeof(t->m_clock_freq[0].m_name), "DATA_CLK");
        t->m_clock_freq[1].m_freq_Mhz = 500;
        t->m_clock_freq[1].m_type = CT_KERNEL;
        snprintf(t->m_clock_freq[1].m_name, sizeof(t->m_clock_freq[1].m_name), "KERNEL_CLK");
    }
    struct sec { axlf_section_kind kind; const char * name; const void * p; size_t n; };
    const sec secs[] = {{EMBEDDED_METADATA, "metadata", xml.data(), xml.size()}, {MEM_TOPOLOGY, "mem", mem.data(), mem.size()},
                        {CONNECTIVITY, "con", con.data(), con.size()},          {IP_LAYOUT, "ip", ipl.data(), ipl.size()},
                        {CLOCK_FREQ_TOPOLOGY, "clk", clk.data(), clk.size()}};
    const size_t nsec = sizeof(secs) / sizeof(secs[0]);
    const size_t hdr = sizeof(axlf) + sizeof(axlf_section_header) * (nsec - 1);
    std::vector<uint8_t> f(hdr);
    std::vector<size_t> offs;
    for (const sec & s : secs) {
        f.resize((f.size() + 7) & ~(size_t) 7);
        offs.push_back(f.size());
        f.insert(f.end(), (const uint8_t *) s.p, (const uint8_t *) s.p + s.n);
    }
    axlf * a = (axlf *) f.data();
    memcpy(a->m_magic, "xclbin2", 8);
    a->m_signature_length = -1;
    memset(a->reserved, 0xFF, sizeof(a->reserved));
    a->m_header.m_length = f.size();
    a->m_header.m_versionMajor = 2;
    a->m_header.m_versionMinor = 1;
    a->m_header.m_mode = XCLBIN_FLAT;
    snprintf((char *) a->m_header.m_platformVBNV, sizeof(a->m_header.m_platformVBNV), "xilinx_u200_noop");
    for (int i = 0; i < 16; ++i) a->m_header.uuid[i] = (unsigned char) (0x37 + 11 * i + variant.size() * 3);
    a->m_header.m_numSections = (uint32_t) nsec;
    for (size_t i = 0; i < nsec; ++i) {
        axlf_section_header & h = a->m_sections[i];
        h.m_sectionKind = secs[i].kind;
        snprintf(h.m_sectionName, sizeof(h.m_sectionName), "%s", secs[i].name);
        h.m_sectionOffset = offs[i];
        h.m_sectionSize = secs[i].n;
    }
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) return 1;
    const bool ok = fwrite(f.data(), 1, f.size(), fp) == f.size();
    return (fclose(fp) == 0 && ok) ? 0 : 1;
}

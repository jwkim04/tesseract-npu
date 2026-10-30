// test_xrt_noop.cpp — ctest transport_xrt_noop (docs/design/06-software-verification.md "SW" §10.2): the
// "xrt" kdev against XRT's own userspace with its no-op shim (XCL_EMULATION_MODE=noop, libxrt_noop):
// no card, no driver (docs/xrt-host.md §6).
//
// Writes metadata-only xclbins of tsr_u200_k4 (EMBEDDED_METADATA kernel XML, IP_LAYOUT,
// MEM_TOPOLOGY with four DDR4 banks, CONNECTIVITY) and checks what XRT itself checks: xclbin
// load, kernel lookup by name, argument discovery by register offset, the memory bank of each
// channel argument from the xclbin (xrt::xclbin::arg::get_mems(), EX §9.1), device-only buffer
// allocation, and the compute unit opened through xrt::ip (exclusive access, EX §2.4) for the
// contract's ap_ctrl_none kernel (what rtl/platform/alveo_xrt/kernel_k*.xml declare) and for EX
// §9.1's fallback form (ap_ctrl_hs without an interrupt). The noop shim implements neither
// register access nor unmanaged DMA, so those calls must fail cleanly (and tsr_transport_open("xrt")
// stops at the first register read, ADAPTER_VERSION). Broken xclbins (s_axi_control range 0x1000,
// a foreign argument, cmd_addr / cmd_count / arg0-7 arguments, no Tesseract kernel) must be
// refused with a reason. Whether XRT refuses a second exclusive open of the compute unit depends
// on the shim (the noop shim's context bookkeeping); it is printed, not asserted (*confirm* on the
// card, P6).
//
//   test-tsr-xrt-noop <XRT root> <scratch dir>

#include "tsr_kdev.h"

#include <xclbin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using namespace tsr;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        g_checks++;                                                         \
        if (!(cond)) {                                                      \
            g_fail++;                                                       \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                   \
            fprintf(stderr, "\n");                                          \
        }                                                                   \
    } while (0)

struct variant {
    std::string kname       = "tsr_u200_k4";
    uint32_t    range       = 0x2000;
    uint32_t    num_ch      = 4;
    bool        foreign_arg = false;
    bool        v1_args     = false;   // run arguments (cmd_addr at 0x10, cmd_count, arg0..7)
    bool        ctrl_none   = true;    // ap_ctrl_none (kernel.xml and IP_LAYOUT), the contract; false: EX §9.1's fallback
    uint64_t    bank_kb     = 64 * 1024;   // 64 MiB banks: the noop shim memsets every buffer
};

// open_xrt_kdev() with stderr captured into *msg (the refusal reason); true if it opened.
static bool open_capture(const std::string & dir, std::string * msg) {
    const std::string path = dir + "/open_stderr.txt";
    FILE * f = fopen(path.c_str(), "w+");
    if (f == nullptr) return false;
    fflush(stderr);
    const int saved = dup(2);
    dup2(fileno(f), 2);
    const bool opened = open_xrt_kdev() != nullptr;
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    msg->clear();
    rewind(f);
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) msg->append(buf, n);
    fclose(f);
    return opened;
}

static std::string kernel_xml(const variant & v) {
    std::string x = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<project name=\"tsr_noop\">\n"
                    "<platform vendor=\"xilinx\" boardid=\"u200\" name=\"noop\" featureRomTime=\"0\">\n"
                    "<version major=\"1\" minor=\"0\"/>\n"
                    "<device name=\"fpga0\" fpgaDevice=\"virtexuplus:xcu200:fsgd2104:-2:e\" addrWidth=\"0\">\n"
                    "<core name=\"OCL_REGION_0\" target=\"hw\" type=\"clc_region\" clockFreq=\"300MHz\" numComputeUnits=\"1\">\n";
    char buf[512];
    snprintf(buf, sizeof(buf),
             "<kernel name=\"%s\" language=\"ip_c\" vlnv=\"tesseract:kernel:%s:1.0\" preferredWorkGroupSizeMultiple=\"0\" "
             "workGroupSize=\"1\" interrupt=\"false\" hwControlProtocol=\"%s\">\n"
             "<port name=\"s_axi_control\" mode=\"slave\" range=\"0x%x\" dataWidth=\"32\" portType=\"addressable\" base=\"0x0\"/>\n",
             v.kname.c_str(), v.kname.c_str(), v.ctrl_none ? "ap_ctrl_none" : "ap_ctrl_hs", v.range);
    x += buf;
    for (uint32_t c = 0; c < v.num_ch; ++c) {
        snprintf(buf, sizeof(buf),
                 "<port name=\"m_axi_ch%u\" mode=\"master\" range=\"0xFFFFFFFFFFFFFFFF\" dataWidth=\"512\" portType=\"addressable\" base=\"0x0\"/>\n", c);
        x += buf;
    }
    auto arg = [&](const char * name, int id, const char * port, uint32_t off, uint32_t size, int aq, const char * type) {
        snprintf(buf, sizeof(buf),
                 "<arg name=\"%s\" addressQualifier=\"%d\" id=\"%d\" port=\"%s\" size=\"0x%x\" offset=\"0x%x\" hostOffset=\"0x0\" "
                 "hostSize=\"0x%x\" type=\"%s\"/>\n",
                 name, aq, id, port, size, off, size, type);
        x += buf;
    };
    int id = 0;
    if (v.v1_args) {
        arg("cmd_addr", id++, "s_axi_control", 0x10, 8, 0, "ulong");
        arg("cmd_count", id++, "s_axi_control", 0x18, 4, 0, "uint");
        for (int i = 0; i < 8; ++i) arg(("arg" + std::to_string(i)).c_str(), id++, "s_axi_control", 0x20 + 4 * i, 4, 0, "uint");
    }
    for (uint32_t c = 0; c < v.num_ch; ++c) {
        arg(("ch" + std::to_string(c)).c_str(), id++, ("m_axi_ch" + std::to_string(c)).c_str(), 0x40 + 8 * c, 8, 1, "void*");
    }
    if (v.foreign_arg) arg("extra", id++, "s_axi_control", 0x1C, 4, 0, "uint");
    snprintf(buf, sizeof(buf), "<instance name=\"%s_1\"><addrRemap base=\"0x1800000\" range=\"0x%x\" port=\"s_axi_control\"/></instance>\n",
             v.kname.c_str(), v.range);
    x += buf;
    x += "</kernel>\n</core>\n</device>\n</platform>\n</project>\n";
    return x;
}

static bool write_xclbin(const std::string & path, const variant & v) {
    // Section payloads.
    const std::string xml = kernel_xml(v);
    std::vector<uint8_t> ipl(sizeof(ip_layout));
    {
        ip_layout * l = (ip_layout *) ipl.data();
        memset(l, 0, sizeof(*l));
        l->m_count = 1;
        l->m_ip_data[0].m_type = IP_KERNEL;
        l->m_ip_data[0].properties = (v.ctrl_none ? AP_CTRL_NONE : AP_CTRL_HS) << IP_CONTROL_SHIFT;
        l->m_ip_data[0].m_base_address = 0x1800000;
        snprintf((char *) l->m_ip_data[0].m_name, sizeof(l->m_ip_data[0].m_name), "%s:%s_1", v.kname.c_str(), v.kname.c_str());
    }
    std::vector<uint8_t> mem(sizeof(mem_topology) + sizeof(mem_data) * (v.num_ch - 1));
    {
        mem_topology * m = (mem_topology *) mem.data();
        m->m_count = (int32_t) v.num_ch;
        for (uint32_t c = 0; c < v.num_ch; ++c) {
            mem_data & d = m->m_mem_data[c];
            memset(&d, 0, sizeof(d));
            d.m_type = MEM_DDR4;
            d.m_used = 1;
            d.m_size = v.bank_kb;
            d.m_base_address = 0x4000000000ull + 0x400000000ull * c;
            snprintf((char *) d.m_tag, sizeof(d.m_tag), "bank%u", c);
        }
    }
    std::vector<uint8_t> con(sizeof(connectivity) + sizeof(connection) * (v.num_ch - 1));
    {
        connectivity * k = (connectivity *) con.data();
        k->m_count = (int32_t) v.num_ch;
        const int32_t first_ch_arg = v.v1_args ? 10 : 0;   // the channel args follow the 10 run args, if any
        for (uint32_t c = 0; c < v.num_ch; ++c) k->m_connection[c] = connection{ first_ch_arg + (int32_t) c, 0, (int32_t) c };
    }
    struct sec { axlf_section_kind kind; const char * name; const void * p; size_t n; };
    const sec secs[] = { { EMBEDDED_METADATA, "metadata", xml.data(), xml.size() }, { MEM_TOPOLOGY, "mem", mem.data(), mem.size() },
                         { CONNECTIVITY, "con", con.data(), con.size() }, { IP_LAYOUT, "ip", ipl.data(), ipl.size() } };
    const size_t nsec = sizeof(secs) / sizeof(secs[0]);
    const size_t hdr = sizeof(axlf) + sizeof(axlf_section_header) * (nsec - 1);
    std::vector<uint8_t> f(hdr);
    for (const sec & s : secs) {
        f.resize((f.size() + 7) & ~(size_t) 7);
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
    for (int i = 0; i < 16; ++i) {
        a->m_header.uuid[i] = (unsigned char) (0x51 + i * 7 + v.range / 0x1000 + v.num_ch + (v.foreign_arg ? 3 : 0) +
                                               (v.v1_args ? 5 : 0) + (v.ctrl_none ? 11 : 0) + v.kname.size());
    }
    a->m_header.m_numSections = (uint32_t) nsec;
    size_t off = hdr;
    for (size_t i = 0; i < nsec; ++i) {
        off = (off + 7) & ~(size_t) 7;
        axlf_section_header & h = a->m_sections[i];
        h.m_sectionKind = secs[i].kind;
        snprintf(h.m_sectionName, sizeof(h.m_sectionName), "%s", secs[i].name);
        h.m_sectionOffset = off;
        h.m_sectionSize = secs[i].n;
        off += secs[i].n;
    }
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = fwrite(f.data(), 1, f.size(), fp) == f.size();
    return fclose(fp) == 0 && ok;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: test-tsr-xrt-noop <XRT root> <scratch dir>\n");
        return 2;
    }
    // Before any XRT call: the emulation mode and the shim location are read once.
    setenv("XCL_EMULATION_MODE", "noop", 1);
    setenv("XILINX_XRT", argv[1], 1);
    const std::string dir = argv[2];
    if (system(("mkdir -p '" + dir + "'").c_str()) != 0) return 2;
    unsetenv("TSR_XRT_CH_GIB");
    unsetenv("TSR_XRT_KERNEL");
    unsetenv("TSR_XRT_DMA");

    variant good;
    const std::string xgood = dir + "/tsr_u200_k4_noop.xclbin";
    CHECK(write_xclbin(xgood, good), "write %s", xgood.c_str());
    setenv("TSR_XCLBIN", xgood.c_str(), 1);
    {
        std::unique_ptr<kdev> d = open_xrt_kdev();
        CHECK(d != nullptr, "open the xrt kdev (ap_ctrl_none through xrt::ip) on the noop shim");
        if (d) {
            printf("%s", d->describe().c_str());
            CHECK(d->num_ch() == 4, "4 channels found by argument offset (%u)", d->num_ch());
            for (uint32_t c = 0; c < d->num_ch(); ++c) {
                CHECK(d->ch_bytes(c) == good.bank_kb * 1024, "ch%u buffer = bank size (%llu)", c, (unsigned long long) d->ch_bytes(c));
                const std::string g = "ch" + std::to_string(c) + ": arg " + std::to_string(c) + ", memory group " + std::to_string(c);
                CHECK(d->describe().find(g) != std::string::npos, "ch%u in the bank of its argument (from the xclbin)", c);
            }
            CHECK(d->describe().find("tsr_u200_k4") != std::string::npos && d->describe().find("xrt::ip") != std::string::npos &&
                  d->describe().find("ap_ctrl_none") != std::string::npos, "describe names the kernel, xrt::ip, ap_ctrl_none");
            CHECK(!d->simulated(), "host time base");
            uint32_t v = 0;
            CHECK(d->reg_read(KREG_CSR_WINDOW, &v) != 0, "noop shim: register read fails cleanly");
            CHECK(d->reg_write(KREG_CSR_WINDOW + TSR_CSR_LOG_CFG, 0) != 0, "noop shim: register write fails cleanly");
            uint8_t b[100] = {};
            CHECK(d->mem_write(0, 3, b, sizeof(b)) != 0, "noop shim: unmanaged DMA fails cleanly (after the staged retry)");
            // A second exclusive open of the same compute unit (another process would get the same answer
            // from the driver on the card).
            std::unique_ptr<kdev> d2 = open_xrt_kdev();
            printf("second xrt::ip open of the compute unit while the first is open: %s (noop shim; *confirm* exclusive refusal on the card)\n",
                   d2 ? "accepted" : "refused");
        }
    }
    CHECK(tsr_transport_open("xrt") == nullptr, "the protocol layer stops at the first register read");
    // EX §9.1's fallback form: ap_ctrl_hs without an interrupt, opened the same way (never started).
    {
        variant hs;
        hs.ctrl_none = false;
        const std::string p = dir + "/tsr_u200_k4_ctrl_hs.xclbin";
        CHECK(write_xclbin(p, hs), "write ap_ctrl_hs");
        setenv("TSR_XCLBIN", p.c_str(), 1);
        std::unique_ptr<kdev> d = open_xrt_kdev();
        CHECK(d != nullptr && d->describe().find("fallback") != std::string::npos, "ap_ctrl_hs fallback opens through xrt::ip");
        setenv("TSR_XCLBIN", xgood.c_str(), 1);
    }

    // Default kernel choice: tsr_u200_k1 when there is no k4.
    variant k1;
    k1.kname = "tsr_u200_k1";
    k1.num_ch = 1;
    const std::string xk1 = dir + "/tsr_u200_k1_noop.xclbin";
    CHECK(write_xclbin(xk1, k1), "write k1");
    setenv("TSR_XCLBIN", xk1.c_str(), 1);
    {
        std::unique_ptr<kdev> d = open_xrt_kdev();
        CHECK(d != nullptr && d->num_ch() == 1, "tsr_u200_k1 picked, 1 channel");
    }
    // TSR_XRT_CH_GIB larger than the bank is refused.
    setenv("TSR_XRT_CH_GIB", "1", 1);
    CHECK(open_xrt_kdev() == nullptr, "TSR_XRT_CH_GIB beyond the bank refused");
    unsetenv("TSR_XRT_CH_GIB");

    // Refusals.
    struct bad { variant v; const char * what; };
    bad bads[4];
    bads[0].v.range = 0x1000;
    bads[0].what = "s_axi_control range 0x1000 refused";
    bads[1].v.foreign_arg = true;
    bads[1].what = "argument outside the contract refused";
    bads[2].v.kname = "other_kernel";
    bads[2].what = "xclbin without a Tesseract kernel refused";
    bads[3].v.v1_args = true;
    bads[3].what = "kernel with cmd_addr / cmd_count / arg0..7 arguments refused";
    for (int i = 0; i < 4; ++i) {
        const std::string p = dir + "/bad" + std::to_string(i) + ".xclbin";
        CHECK(write_xclbin(p, bads[i].v), "write bad %d", i);
        setenv("TSR_XCLBIN", p.c_str(), 1);
        std::string msg;
        CHECK(!open_capture(dir, &msg), "%s", bads[i].what);
        if (bads[i].v.v1_args) CHECK(msg.find("kernel contract") != std::string::npos, "the refusal names the kernel contract: \"%s\"", msg.c_str());
    }
    setenv("TSR_XCLBIN", xgood.c_str(), 1);
    // m2m: the staging buffers are allocated at open, before the channel buffers (the noop shim
    // does not enforce bank capacity, so only the order / success is visible here).
    setenv("TSR_XRT_DMA", "m2m", 1);
    setenv("TSR_XRT_DMA_CHUNK_MIB", "1", 1);
    {
        std::unique_ptr<kdev> d = open_xrt_kdev();
        CHECK(d != nullptr && d->describe().find("staging buffer") != std::string::npos, "m2m: staging buffers allocated at open");
        if (d) printf("%s", d->describe().c_str());
    }
    unsetenv("TSR_XRT_DMA");
    unsetenv("TSR_XRT_DMA_CHUNK_MIB");
    printf("test-tsr-xrt-noop: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

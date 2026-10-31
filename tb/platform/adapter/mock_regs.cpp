// mock_regs.cpp — tb/platform's oracle driver: the mock adapter model (sw/transport/tsr_kdev_mock.cpp,
// TSR_DEVICE=mock; docs/design/01-execution-model.md "EX" §9.2) behind a line protocol, so the RTL
// bench (tb/platform/test_adapter.py) can replay the same register accesses on tsr_xrt_ctrl.
//
//   mock_regs NUM_CH < script        one command per line, one answer line per R:
//     W off val    s_axi_control write (full word)
//     R off        read; prints "off val" (hex)
//     E bits cnt   adapter events: ADAPTER_STATUS |= bits (0..2), error count += cnt (inject_status)
//     B c addr     the device address of channel c's buffer (the model refuses to move with a
//                  latched base that differs, bind_errors; the bench sets it to the bases it binds)
// Bases are 4 KiB aligned device addresses chosen by the script; the model's own default buffer
// addresses are replaced with B lines before MEM_EN.
#include "tsr_kdev_mock.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    tsr::mock_options o;
    o.num_ch = argc > 1 ? (uint32_t) strtoul(argv[1], nullptr, 0) : 4u;
    o.ch_bytes = 1ull << 30;
    tsr::mock_kdev m(o);
    if (!m.ok()) {
        fprintf(stderr, "mock_regs: the mock adapter model did not open\n");
        return 2;
    }
    char line[256];
    while (fgets(line, sizeof(line), stdin)) {
        char op = 0;
        unsigned long long a = 0, b = 0;
        const int n = sscanf(line, " %c %llx %llx", &op, &a, &b);
        if (n < 1) continue;
        if (op == 'W' && n == 3) {
            if (m.reg_write((uint32_t) a, (uint32_t) b) != 0) return 3;
        } else if (op == 'R' && n >= 2) {
            uint32_t v = 0;
            if (m.reg_read((uint32_t) a, &v) != 0) return 3;
            printf("%llx %x\n", a, v);
        } else if (op == 'E' && n == 3) {
            m.inject_status((uint32_t) (a & 7u) | (uint32_t) (b << 16));
        } else if (op == 'B' && n == 3) {
            m.set_ch_addr((uint32_t) a, (uint64_t) b);
        } else {
            fprintf(stderr, "mock_regs: bad line: %s", line);
            return 2;
        }
    }
    printf("bind_errors %" PRIu64 "\n", m.bind_errors);
    return 0;
}

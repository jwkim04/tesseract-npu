// fw_example_vadd: the README's vadd kernel on the simulator, through the transport v2 API.
#include "tsr_fw_paths.h"
#include "tsr_ref.h"
#include "tsr_transport.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int run(const char * sched) {
    setenv("TSR_SIM_SCHED", sched, 1);
    setenv("TSR_SIM_RACE", "fatal", 1);
    constexpr uint32_t n = 1024, bytes = n * 4, beats = bytes / 64;
    constexpr uint32_t A = 0, B = beats, Y = 2 * beats;   // beats on channel 0
    std::vector<float> a(n), b(n), y(n, 0.f);
    uint32_t s = 12345u;
    for (uint32_t i = 0; i < n; ++i) {
        for (float * v : { &a[i], &b[i] }) {
            s = s * 1664525u + 1013904223u;
            *v = (float) ((int32_t) (s >> 8) - (1 << 23)) / (float) (1 << 18);
        }
    }
    tsr_image * img = nullptr;
    if (tsr_image_open(TSR_FW_TEST_IMAGE_fw_example_vadd, &img) != 0) { printf("FAIL image\n"); return 1; }
    tsr_transport * t = tsr_transport_open("sim");
    if (!t || tsr_transport_boot(t, img) != 0) { printf("FAIL open/boot\n"); return 1; }
    const int k = tsr_image_kernel(img, "vadd");
    if (k < 0) { printf("FAIL kernel\n"); return 1; }
    int bad = 0;
    bad += tsr_transport_write(t, 0, (uint64_t) A * 64, a.data(), bytes) != 0;
    bad += tsr_transport_write(t, 0, (uint64_t) B * 64, b.data(), bytes) != 0;
    bad += tsr_transport_write(t, 0, (uint64_t) Y * 64, y.data(), bytes) != 0;
    tsr_launch l{};
    l.type = TSR_DESC_TYPE_LAUNCH;
    l.kernel = (uint16_t) k;
    l.arg[0] = A; l.arg[1] = B; l.arg[2] = Y; l.arg[3] = n;
    tsr_record r;
    if (tsr_transport_call(t, &l, TSR_TIMEOUT_DEFAULT, &r) != 0 || TSR_GET(r.w[0], TSR_LCQ_R0_STATUS) != TSR_LCQ_R0_STATUS_OK) { printf("FAIL call (w0 0x%08x)\n", r.w[0]); bad++; }
    bad += tsr_transport_read(t, 0, (uint64_t) Y * 64, y.data(), bytes) != 0;
    uint32_t mism = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const float e = tsr_fadd(a[i], b[i]);
        if (memcmp(&e, &y[i], 4) != 0) mism++;
    }
    printf("fw_example_vadd [%s]: %u of %u words differ, %d errors\n", sched, mism, n, bad);
    tsr_transport_close(t);
    tsr_image_close(img);
    return (bad || mism) ? 1 : 0;
}

int main() {
    int rc = run("fixed");
    rc |= run("random:1");
    return rc;
}

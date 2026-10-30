// test_fw_selftest_sim.c — fw_selftest_sim: every `selftest` id (docs/v2/06-software-verification.md
// "SW" §7.6, §10.2; kernel sw/ctrl/kernels/selftest.cpp, interface sw/ctrl/ref/selftest_abi.hpp) on
// the production image on tsr_rv.
//
//   test_fw_selftest_sim IMAGE [--scheds N]
// One session per schedule (fixed + 7 random, TSR_SIM_RACE fatal): a fresh simulator boots the image,
// answers a PING, then runs ids 1-10 (id 2 twice) and three misuses, each a launch. Every launch must
// answer DONE OK with RET 0 (the device-side checks of ids 1, 7, 8, 10) or the misuse's code; every
// output is compared with references computed here from the inputs (tsr_ref, tsr_ref_vu, tsr_ref_at;
// for ids 4-5 each op from the device's own inputs, so a difference names the op); id 9's log records
// and event are decoded and checked against the ELF's format strings. The SW §4.7 observables (per-unit
// op streams, RISC-V SPM stores, final written state) and every DONE record must be equal under every
// schedule. Exit 0 / 1, 77 (skipped) without the image.
#undef _GNU_SOURCE
#include "test_sim_obs.h"

#include "fw_selftest_cases.h"   // the cases (W2a: shared with fw_m6_e2e)

static const char * g_image;

// ---------------------------------------------------------------------------
// One session per schedule
// ---------------------------------------------------------------------------
static void run(const char * sched, obs * o, session * s) {
    tsr_config cfg = TSR_CONFIG_U200_V1;
    cfg.num_ch = NCH;
    cfg.t_max = 8;
    cfg.ch_bytes = 1ull << 30;
    cfg.caps = TSR_SIM_CAPS_IMPLEMENTED;
    tsr_sim_opts so;
    memset(&so, 0, sizeof(so));
    so.sched = sched;
    so.race = 2;
    so.threads = 2;
    memset(s, 0, sizeof(*s));
    s->sched = sched;
    s->sim = tsr_sim_create_opts(&cfg, &so);
    CHECK(s->sim != NULL, "[%s] simulator", sched);
    if (s->sim == NULL) return;
    obs_attach(o, s->sim);
    if (fwh_open(&s->h, s->sim, g_image) != 0) {
        CHECK(0, "[%s] boot", sched);
        tsr_sim_destroy(s->sim);
        s->sim = NULL;
        return;
    }
    s->idx = fwh_kernel(&s->h, "selftest");
    CHECK(s->idx >= 0, "no selftest kernel in %s", g_image);
    if (s->idx >= 0) {
        CHECK(fwh_ping(&s->h, 0xC0FFEE09u) == 0, "[%s] PING -> PONG with the ARG0 echo", sched);
        st_run_ids(s);
        const uint32_t st = tsr_sim_csr_read(s->sim, TSR_CSR_STATUS);
        CHECK(!(st & TSR_STATUS_QERR), "[%s] a unit left in error state (STATUS 0x%08x)", sched, st);
    }
    obs_finish(o, s->sim);
    printf("fw_selftest_sim [%s]: %u launches, %llu output regions (%llu lines) compared, MTIME %llu, %llu RV instructions\n",
           sched, s->n_done, (unsigned long long)g_cmp_n, (unsigned long long)g_cmp_lines, (unsigned long long)o->mtime,
           (unsigned long long)o->instret);
    g_cmp_n = g_cmp_lines = 0;
    fwh_close(&s->h);
    tsr_sim_destroy(s->sim);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s IMAGE [--scheds N]\n", argv[0]);
        return 2;
    }
    g_image = argv[1];
    FILE * f = fopen(g_image, "rb");
    if (f == NULL) {
        printf("%s: no image %s: skipped\n", argv[0], g_image);
        return 77;
    }
    fclose(f);
    uint32_t n_sched = 8;
    if (argc > 3 && !strcmp(argv[2], "--scheds")) n_sched = (uint32_t)atoi(argv[3]);
    static session ses[8];
    obs o[8];
    for (uint32_t k = 0; k < n_sched && k < 8u; ++k) {
        char * sched = (char *)malloc(32);
        if (k == 0) snprintf(sched, 32, "fixed");
        else snprintf(sched, 32, "random:%u", 900u + k);
        run(sched, &o[k], &ses[k]);
        if (k > 0) {
            char what[64];
            snprintf(what, sizeof(what), "%s vs fixed", sched);
            obs_equal(&o[0], &o[k], what);
            CHECK(ses[k].n_done == ses[0].n_done && memcmp(ses[k].done, ses[0].done, sizeof(ses[0].done)) == 0,
                  "%s: DONE records differ", what);
        }
    }
    for (uint32_t k = 0; k < n_sched && k < 8u; ++k) obs_free(&o[k]);
    return test_finish("fw_selftest_sim");
}

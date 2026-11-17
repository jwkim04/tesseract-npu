// The host side of the Ising machine example (ising_run.hpp).
#include "ising_run.hpp"

#include <string.h>

#include <algorithm>

#include "tsr_ref.h"

namespace ism {

namespace {

// The DRAM map of EX §11, in beats of 64 bytes (per channel; the regions below 512 MiB are channel 0's,
// except the compute chunks that software may place in the low 512 MiB of the others).
constexpr uint32_t BEATS_MIB = 1u << 14;                 // beats in 1 MiB
constexpr uint32_t OPL_BEAT = 0;                         // 0 - 64 MiB: argument blocks
constexpr uint32_t IO_BEAT = 64 * BEATS_MIB;             // 64 - 320 MiB: host to device tensors
constexpr uint32_t IO_END = 320 * BEATS_MIB;
constexpr uint32_t WEIGHT_BEAT = 512 * BEATS_MIB;        // weights from 512 MiB to the end of the channel
constexpr uint32_t LOW_END = 512 * BEATS_MIB;            // the low 512 MiB of channels 1..3
constexpr uint32_t RUNTIME_LINES = 16;                   // the top SPM lines that the runtime keeps for the argument block

std::string mib(uint64_t beats) { return std::to_string((beats * 64 + (1u << 20) - 1) >> 20) + " MiB"; }

}  // namespace

bool machine::open(tsr_transport * t, int kernel, const problem & prob, const options & opt, std::string * err) {
    const tsr_config * cfg = tsr_transport_config(t);
    t_ = t;
    kernel_ = kernel;
    n_ = prob.n;
    nch_ = cfg->num_ch;
    opt_ = opt;
    auto fail = [&](const std::string & m) { if (err) *err = m; return false; };
    if (opt.t == 0 || opt.t > TSR_T_MAX_ARCH) return fail("replicas (t) must be 1 to 8");
    if (opt.t > cfg->t_max) return fail("this device runs at most " + std::to_string(cfg->t_max) + " replicas (T_MAX), not " + std::to_string(opt.t));
    const uint32_t kmax = cfg->k_max / 32u * 32u;
    pl_ = placement{};
    pl_.kblk = opt.kblk != 0 ? opt.kblk : std::min(n_, kmax);
    if (pl_.kblk > n_) pl_.kblk = n_;
    if (pl_.kblk % 32 != 0 || pl_.kblk > kmax) return fail("kblk must be a multiple of 32 and at most K_MAX (" + std::to_string(kmax) + ")");
    pl_.nblk = (n_ + pl_.kblk - 1) / pl_.kblk;
    if (pl_.nblk > MAX_BLK) return fail("more than " + std::to_string(MAX_BLK) + " column tensors: raise kblk");

    // Steps per launch.
    uint32_t chunk = opt.chunk != 0 ? std::min(opt.chunk, MAX_STEPS) : MAX_STEPS;
    if (opt.snap_every != 0) {
        if (opt.snap_every > MAX_STEPS) return fail("snap_every is above " + std::to_string(MAX_STEPS));
        chunk = std::max(opt.snap_every, chunk / opt.snap_every * opt.snap_every);   // a launch holds whole snapshot periods
    }
    pl_.chunk = chunk;

    // The SPM must hold the buffers of a launch of `chunk` steps (the kernel checks again).
    {
        params pr{};
        pr.magic = MAGIC; pr.version = VERSION; pr.n = n_; pr.t = opt.t; pr.steps = chunk; pr.kblk = pl_.kblk;
        pr.flags = opt.energy ? FLAG_ENERGY : 0u;
        layout lay;
        const uint32_t spm_lines = cfg->spm_bytes / 64u - RUNTIME_LINES;
        const int32_t st = make_layout(pr, spm_lines, cfg->t_max, cfg->k_max, &lay);
        if (st == ST_BAD_SPM) return fail("the buffers of " + std::to_string(opt.t) + " replicas of " + std::to_string(n_) +
                                          " spins do not fit in the scratchpad: lower t or the chunk");
        if (st != ST_OK) return fail("the device refuses these parameters (status " + std::to_string(st) + ")");
    }

    // Device memory: parameter block, state, energy lines, snapshots, weights.
    const uint64_t tl = uint64_t(opt.t) * (n_ / 16u);                 // beats of the t rows of x (or y, or one snapshot)
    const uint64_t e_beats = opt.energy ? chunk : 0u;
    pl_.param = OPL_BEAT;
    pl_.x = IO_BEAT;
    pl_.y = uint32_t(IO_BEAT + tl);
    pl_.energy = uint32_t(IO_BEAT + 2 * tl);
    uint64_t io_used = 2 * tl + e_beats;
    const uint64_t snap_beats = opt.snap_every ? uint64_t(chunk / opt.snap_every) * tl : 0u;
    if (nch_ > 1) {   // snapshots go to the low 512 MiB of channel 1
        pl_.snap_ch = 1;
        pl_.snap = 0;
        if (snap_beats > LOW_END) return fail("the snapshots of one launch need " + mib(snap_beats) + ", channel 1 has " + mib(LOW_END) +
                                               " free: raise snap_every or lower chunk");
    } else {
        pl_.snap_ch = 0;
        pl_.snap = uint32_t(IO_BEAT + io_used);
        io_used += snap_beats;
    }
    if (IO_BEAT + io_used > IO_END) return fail("the state, energies and snapshots of one launch need " + mib(io_used) +
                                                " of the 256 MiB IO region: raise snap_every or lower chunk");

    const uint32_t R = (n_ + nch_ - 1) / nch_;
    const uint64_t row_beats = tsr_row_beats(TSR_FMT_Q8_0, pl_.kblk);
    pl_.w_off = WEIGHT_BEAT;
    const uint64_t stride = uint64_t(R) * row_beats;
    pl_.tensor_stride = uint32_t(stride);
    const uint64_t end_beat = uint64_t(WEIGHT_BEAT) + stride * pl_.nblk;
    if (end_beat * 64 > cfg->ch_bytes || end_beat >= (1ull << cfg->addr_beat_w))
        return fail("the weights need " + mib(stride * pl_.nblk) + " per channel from 512 MiB on, a channel has " +
                    mib(cfg->ch_bytes / 64) + ": the problem is too large for this device");

    // The timeout of a launch: every step streams the weights of one channel at no less than 250 MB/s
    // (the card and the simulator are far faster), plus 100 us of fixed cost.
    timeout_per_step_ns_ = stride * 64u * 4u + 100000u;
    return upload_weights(prob, err);
}

bool machine::upload_weights(const problem & prob, std::string * err) {
    const uint32_t n = n_, kblk = pl_.kblk;
    std::vector<int8_t> row(n);
    std::vector<tsr_block_q8_0> blocks(kblk / 32);
    const uint32_t row_bytes = tsr_row_beats(TSR_FMT_Q8_0, kblk) * 64u;
    std::vector<uint8_t> packed(row_bytes);
    // Consecutive rows on one channel are consecutive in memory: collect them and write in large pieces.
    std::vector<uint8_t> pending;
    uint32_t pend_ch = 0;
    uint64_t pend_off = 0;
    auto flush = [&]() {
        if (pending.empty()) return true;
        const int rc = tsr_transport_write(t_, pend_ch, pend_off, pending.data(), pending.size());
        pending.clear();
        return rc == 0;
    };
    for (uint32_t b = 0; b < pl_.nblk; ++b) {
        const uint32_t col0 = b * kblk, k = (b + 1 < pl_.nblk) ? kblk : n - col0;
        const uint32_t nb = k / 32, rb = tsr_row_beats(TSR_FMT_Q8_0, k) * 64u;
        for (uint32_t r = 0; r < n; ++r) {
            prob.row(r, row.data());
            for (uint32_t j = 0; j < nb; ++j) {
                blocks[j].d = tsr_f32_to_fp16(1.0f);
                memcpy(blocks[j].qs, row.data() + col0 + j * 32, 32);
            }
            tsr_repack_row(TSR_FMT_Q8_0, blocks.data(), k, packed.data());
            uint32_t ch;
            uint64_t boff;
            tsr_row_location(TSR_FMT_Q8_0, n, k, nch_, r, &ch, &boff);
            const uint64_t off = (uint64_t(pl_.w_off) + uint64_t(b) * pl_.tensor_stride + boff) * 64u;
            if (!pending.empty() && (ch != pend_ch || off != pend_off + pending.size() || pending.size() >= (8u << 20))) {
                if (!flush()) { if (err) *err = "writing the weights failed"; return false; }
            }
            if (pending.empty()) { pend_ch = ch; pend_off = off; }
            pending.insert(pending.end(), packed.begin(), packed.begin() + rb);
        }
    }
    if (!flush()) { if (err) *err = "writing the weights failed"; return false; }
    return true;
}

bool machine::write_state(const float * x, const float * y) {
    const size_t bytes = size_t(opt_.t) * n_ * 4;
    return tsr_transport_write(t_, 0, uint64_t(pl_.x) * 64, x, bytes) == 0 && tsr_transport_write(t_, 0, uint64_t(pl_.y) * 64, y, bytes) == 0;
}

bool machine::read_state(std::vector<float> * x, std::vector<float> * y) {
    const size_t cnt = size_t(opt_.t) * n_;
    x->resize(cnt);
    y->resize(cnt);
    return tsr_transport_read(t_, 0, uint64_t(pl_.x) * 64, x->data(), cnt * 4) == 0 &&
           tsr_transport_read(t_, 0, uint64_t(pl_.y) * 64, y->data(), cnt * 4) == 0;
}

params machine::make_params(uint32_t steps, const schedule & sch) const {
    params p{};
    p.magic = MAGIC;
    p.version = VERSION;
    p.n = n_;
    p.t = opt_.t;
    p.steps = steps;
    p.snap_every = opt_.snap_every;
    p.flags = opt_.energy ? FLAG_ENERGY : 0u;
    p.w_off = pl_.w_off;
    p.kblk = pl_.kblk;
    p.tensor_stride = pl_.tensor_stride;
    p.x = (0u << 28) | pl_.x;
    p.y = (0u << 28) | pl_.y;
    p.snap = (pl_.snap_ch << 28) | pl_.snap;
    p.energy = (0u << 28) | pl_.energy;
    p.ca0 = sch.ca0;
    p.da = sch.da;
    p.cf = sch.cf;
    p.cd = sch.cd;
    return p;
}

bool machine::launch(const params & p, tsr_result * res, int32_t * ret, std::string * err) {
    auto fail = [&](const std::string & m) { if (err) *err = m; return false; };
    if (tsr_transport_write(t_, 0, uint64_t(pl_.param) * 64, &p, sizeof(p)) != 0) return fail("writing the parameter block failed");
    tsr_launch l{};
    l.type = TSR_DESC_TYPE_LAUNCH;
    l.kernel = uint16_t(kernel_);
    l.arg[0] = pl_.param;   // channel 0, beat of the block
    tsr_record rec;
    const uint64_t timeout = 10ull * 1000000000ull + uint64_t(p.steps) * timeout_per_step_ns_;
    if (!stop_poll_) {
        const int rc = tsr_transport_call(t_, &l, timeout, &rec);
        if (rc != 0) return fail(std::string("the launch failed: ") + tsr_transport_strerror(rc));
    } else {
        // Launch, then poll for the answer and for the stop request. A launch that outlasts its timeout is cancelled.
        uint32_t seq = 0;
        int rc = tsr_transport_launch(t_, &l, &seq);
        if (rc != 0) return fail(std::string("the launch failed: ") + tsr_transport_strerror(rc));
        const uint64_t t0 = tsr_transport_now_ns(t_);
        bool stopped = false;
        for (;;) {
            rc = tsr_transport_poll(t_, seq, &rec);
            if (rc != 0) break;
            if (!stopped && stop_poll_()) {
                stopped = true;
                tsr_transport_stop(t_, seq);
            }
            if (tsr_transport_now_ns(t_) - t0 > timeout) {
                tsr_transport_cancel(t_, seq);
                tsr_transport_wait(t_, seq, TSR_TIMEOUT_DEFAULT, &rec);
                return fail("the launch did not finish in time and was cancelled");
            }
            tsr_transport_idle(t_, 20000);
        }
        if (rc < 0) return fail(std::string("the launch failed: ") + tsr_transport_strerror(rc));
    }
    tsr_record_decode(&rec, res);
    *ret = res->ret;
    return true;
}

bool machine::run(const schedule & sch0, uint32_t steps, run_result * out, std::string * err) {
    auto fail = [&](const std::string & m) { if (err) *err = m; return false; };
    schedule sch = sch0;
    const uint32_t t = opt_.t, n = n_;
    const uint64_t tl = uint64_t(t) * (n / 16u);
    out->energy.clear();
    out->snapshots.clear();
    out->steps_done = 0;
    out->stopped = false;
    for (uint32_t done = 0; done < steps;) {
        if (stop_poll_ && stop_poll_()) { out->stopped = true; break; }
        const uint32_t cnt = std::min(pl_.chunk, steps - done);
        const params p = make_params(cnt, sch);
        tsr_result res;
        int32_t ret = 0;
        if (!launch(p, &res, &ret, err)) return false;
        uint32_t did = cnt;   // the steps this launch did
        if (res.status == TSR_LCQ_R0_STATUS_STOPPED) {
            if ((ret & ST_STOPPED_FLAG) == 0 || uint32_t(ret & 0xFFFF) > cnt) return fail("a stopped launch returned a bad step count");
            did = uint32_t(ret & 0xFFFF);
            out->stopped = true;
        } else {
            if (res.status != TSR_LCQ_R0_STATUS_OK)
                return fail(std::string("the launch ended with status ") + tsr_status_name(res.status) +
                            " (error code " + std::to_string(res.err_code) + ")");
            if (ret != ST_OK) return fail("the kernel refused the parameter block (status " + std::to_string(ret) + ")");
        }
        if (opt_.energy && did != 0) {
            std::vector<float> lines(size_t(did) * 16);
            if (tsr_transport_read(t_, 0, uint64_t(pl_.energy) * 64, lines.data(), lines.size() * 4) != 0) return fail("reading the energies failed");
            for (uint32_t k = 0; k < did; ++k) out->energy.insert(out->energy.end(), lines.begin() + size_t(k) * 16, lines.begin() + size_t(k) * 16 + t);
        }
        if (opt_.snap_every != 0) {
            const size_t snaps = did / opt_.snap_every;
            const size_t old = out->snapshots.size();
            out->snapshots.resize(old + snaps * size_t(t) * n);
            if (snaps != 0 && tsr_transport_read(t_, pl_.snap_ch, uint64_t(pl_.snap) * 64, out->snapshots.data() + old, snaps * tl * 64) != 0)
                return fail("reading the snapshots failed");
        }
        sch.ca0 = advance_ca(sch.ca0, sch.da, did);
        done += did;
        out->steps_done = done;
        if (out->stopped) break;
    }
    return read_state(&out->x, &out->y) ? true : fail("reading the state failed");
}

}  // namespace ism

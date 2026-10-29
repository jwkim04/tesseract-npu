// tsr-wlayout.cpp — see tsr-wlayout.h
#include "tsr-wlayout.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "tsr_isa.h"
#include "tsr_ref.h"

size_t tsr_dev_blocks_row_bytes(uint32_t fmt, uint32_t K) {
    return tsr_fmt_ggml_block_bytes(fmt) * (size_t) (K / tsr_fmt_block_size(fmt));
}

namespace {

// Rows [r, end) of channel c, starting at row `local` of that channel.
struct store_batch {
    uint32_t c, local, r, end;
};

} // namespace

void tsr_weight_store_rows(uint32_t num_ch, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                           uint32_t r0, uint32_t r1, size_t max_batch_bytes,
                           const tsr_row_blocks_fn & row_blocks, const tsr_dram_write_fn & write,
                           unsigned threads) {
    const uint64_t dev_rb = (uint64_t) tsr_row_beats(fmt, K) * TSR_BEAT_BYTES;
    const uint32_t R = tsr_rows_per_ch(N, num_ch);
    const uint32_t batch = (uint32_t) std::max<uint64_t>(1, max_batch_bytes / dev_rb);
    const size_t scratch_bytes = tsr_dev_blocks_row_bytes(fmt, K);

    std::vector<store_batch> batches;
    for (uint32_t r = r0; r < r1;) {
        const uint32_t c      = r / R;
        const uint32_t ch_end = std::min<uint32_t>((c + 1) * R, N);
        const uint32_t end    = std::min<uint32_t>({ r1, ch_end, r + batch });
        batches.push_back({ c, r - c * R, r, end });
        r = end;
    }
    auto fill = [&](const store_batch & b, std::vector<uint8_t> & out, std::vector<uint8_t> & scratch) {
        out.resize((size_t) (b.end - b.r) * dev_rb);
        for (uint32_t rr = b.r; rr < b.end; ++rr) {
            const void * blk = row_blocks(rr, scratch.data());
            tsr_repack_row(fmt, blk, K, out.data() + (size_t) (rr - b.r) * dev_rb);
        }
    };
    auto put = [&](const store_batch & b, const std::vector<uint8_t> & out) {
        write(b.c, w_off + (uint64_t) b.local * dev_rb, out.data(), out.size());
    };

    const unsigned nw = (unsigned) std::min<size_t>(std::max(threads, 1u), batches.size());
    if (nw <= 1) {
        std::vector<uint8_t> scratch(scratch_bytes), out;
        for (const store_batch & b : batches) {
            fill(b, out, scratch);
            put(b, out);
        }
        return;
    }

    // Workers take batches in order and fill free buffers; the calling thread writes each filled
    // buffer and returns it. nw + 1 buffers: one per worker plus the one being written.
    std::vector<std::vector<uint8_t>> bufs(nw + 1);
    std::vector<size_t> free_bufs;
    for (size_t i = 0; i < bufs.size(); ++i) {
        free_bufs.push_back(i);
    }
    std::deque<std::pair<size_t, size_t>> ready;   // (batch, buffer)
    std::mutex mu;
    std::condition_variable cv;
    size_t next = 0;
    bool stop = false;
    std::exception_ptr err;

    auto fail = [&](std::exception_ptr e) {
        std::lock_guard<std::mutex> lk(mu);
        if (!err) {
            err = e;
        }
        stop = true;
        cv.notify_all();
    };
    auto worker = [&] {
        std::vector<uint8_t> scratch(scratch_bytes);
        for (;;) {
            size_t bi, buf;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || next >= batches.size() || !free_bufs.empty(); });
                if (stop || next >= batches.size()) {
                    return;
                }
                bi = next++;
                buf = free_bufs.back();
                free_bufs.pop_back();
            }
            try {
                fill(batches[bi], bufs[buf], scratch);
            } catch (...) {
                fail(std::current_exception());
                return;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                ready.emplace_back(bi, buf);
            }
            cv.notify_all();
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nw; ++t) {
        pool.emplace_back(worker);
    }
    for (size_t written = 0; written < batches.size(); ++written) {
        std::pair<size_t, size_t> item;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return stop || !ready.empty(); });
            if (stop) {
                break;
            }
            item = ready.front();
            ready.pop_front();
        }
        try {
            put(batches[item.first], bufs[item.second]);
        } catch (...) {
            fail(std::current_exception());
            break;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            free_bufs.push_back(item.second);
        }
        cv.notify_all();
    }
    for (std::thread & t : pool) {
        t.join();
    }
    if (err) {
        std::rethrow_exception(err);
    }
}

void tsr_weight_load_rows(uint32_t num_ch, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                          uint32_t r0, uint32_t r1, size_t max_batch_bytes,
                          const tsr_dram_read_fn & read, const tsr_row_sink_fn & sink) {
    const uint64_t dev_rb = (uint64_t) tsr_row_beats(fmt, K) * TSR_BEAT_BYTES;
    const uint32_t R = tsr_rows_per_ch(N, num_ch);
    const uint32_t batch = (uint32_t) std::max<uint64_t>(1, max_batch_bytes / dev_rb);

    std::vector<uint8_t> blocks(tsr_dev_blocks_row_bytes(fmt, K));
    std::vector<uint8_t> in;

    uint32_t r = r0;
    while (r < r1) {
        const uint32_t c      = r / R;
        const uint32_t local  = r - c * R;
        const uint32_t ch_end = std::min<uint32_t>((c + 1) * R, N);
        const uint32_t end    = std::min<uint32_t>({ r1, ch_end, r + batch });
        in.resize((size_t) (end - r) * dev_rb);
        read(c, w_off + (uint64_t) local * dev_rb, in.data(), in.size());
        for (uint32_t rr = r; rr < end; ++rr) {
            tsr_unrepack_row(fmt, in.data() + (size_t) (rr - r) * dev_rb, K, blocks.data());
            sink(rr, blocks.data());
        }
        r = end;
    }
}

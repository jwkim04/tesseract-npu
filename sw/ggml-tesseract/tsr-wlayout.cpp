// tsr-wlayout.cpp — see tsr-wlayout.h
#include "tsr-wlayout.h"

#include <algorithm>
#include <vector>

#include "tsr_isa.h"
#include "tsr_ref.h"

size_t tsr_dev_blocks_row_bytes(uint32_t fmt, uint32_t K) {
    return tsr_fmt_ggml_block_bytes(fmt) * (size_t) (K / tsr_fmt_block_size(fmt));
}

void tsr_weight_store_rows(uint32_t num_ch, uint32_t fmt, uint32_t N, uint32_t K, uint64_t w_off,
                           uint32_t r0, uint32_t r1, size_t max_batch_bytes,
                           const tsr_row_blocks_fn & row_blocks, const tsr_dram_write_fn & write) {
    const uint64_t dev_rb = (uint64_t) tsr_row_beats(fmt, K) * TSR_BEAT_BYTES;
    const uint32_t R = tsr_rows_per_ch(N, num_ch);
    const uint32_t batch = (uint32_t) std::max<uint64_t>(1, max_batch_bytes / dev_rb);

    std::vector<uint8_t> scratch(tsr_dev_blocks_row_bytes(fmt, K));
    std::vector<uint8_t> out;

    uint32_t r = r0;
    while (r < r1) {
        const uint32_t c      = r / R;
        const uint32_t local  = r - c * R;
        const uint32_t ch_end = std::min<uint32_t>((c + 1) * R, N);
        const uint32_t end    = std::min<uint32_t>({ r1, ch_end, r + batch });
        out.resize((size_t) (end - r) * dev_rb);
        for (uint32_t rr = r; rr < end; ++rr) {
            const void * blk = row_blocks(rr, scratch.data());
            tsr_repack_row(fmt, blk, K, out.data() + (size_t) (rr - r) * dev_rb);
        }
        write(c, w_off + (uint64_t) local * dev_rb, out.data(), out.size());
        r = end;
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

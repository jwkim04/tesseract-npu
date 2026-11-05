// Local intermediate geometry shared by the firmware and functional simulator.
#ifndef TSR_LOCAL_TILE_H
#define TSR_LOCAL_TILE_H

#include <stdint.h>

#define TSR_LOCAL_AT_OWNERS 4u
#define TSR_LOCAL_AT_SLOTS 3u
#define TSR_LOCAL_AT_OWNER_LINES 4480u
#define TSR_LOCAL_AT_MAX_ROWS 224u
#define TSR_LOCAL_AT_MAX_GROUPS 128u
#define TSR_LOCAL_AT_SLOT_LINES (TSR_LOCAL_AT_OWNERS * TSR_LOCAL_AT_OWNER_LINES)

// A score row keeps chunk-relative order. Its stride is padded to four lines
// so every owner has an integral number of line positions per row.
static inline uint32_t tsr_local_at_stride(uint32_t groups) {
    return (groups + TSR_LOCAL_AT_OWNERS - 1u) & ~(TSR_LOCAL_AT_OWNERS - 1u);
}

static inline int tsr_local_at_fits(uint32_t channels, uint32_t rows, uint32_t groups) {
    if (channels != TSR_LOCAL_AT_OWNERS || rows == 0u || rows > TSR_LOCAL_AT_MAX_ROWS ||
        groups == 0u || groups > TSR_LOCAL_AT_MAX_GROUPS)
        return 0;
    return rows * (tsr_local_at_stride(groups) >> 2u) <= TSR_LOCAL_AT_OWNER_LINES;
}

// An owner window is relative to the original packing, never to a smaller N.
// The expert base and token origins are software metadata. The dense opcode
// uses expert zero and identity token order; a routed expert would also carry
// explicit token IDs for gather and combine.
typedef struct {
    uint64_t weight_base, expert_stride;    // beats in each owner's channel
    uint32_t expert, channels, rows, row_beats;
    uint32_t input_token0, output_token0, tokens;
} tsr_dense_tile;

typedef struct {
    uint64_t weight_begin, weight_beats;
    uint32_t output_row0, rows, input_token0, output_token0, tokens;
} tsr_dense_window;

static inline int tsr_local_dense_window(const tsr_dense_tile * tile, uint32_t owner,
                                         uint32_t first, uint32_t max_rows, tsr_dense_window * out) {
    if (owner >= tile->channels || tile->row_beats == 0u ||
        (tile->expert && tile->expert_stride > (UINT64_MAX - tile->weight_base) / tile->expert))
        return 0;
    const uint32_t per_owner = tile->rows / tile->channels + (tile->rows % tile->channels != 0u);
    const uint64_t global_first = (uint64_t)owner * per_owner;
    uint32_t count = global_first < tile->rows ? tile->rows - (uint32_t)global_first : 0u;
    if (count > per_owner) count = per_owner;
    if (first > count) return 0;
    count -= first;
    if (count > max_rows) count = max_rows;
    const uint64_t base = tile->weight_base + tile->expert_stride * tile->expert;
    const uint64_t offset = (uint64_t)first * tile->row_beats;
    const uint64_t beats = (uint64_t)count * tile->row_beats;
    if (offset > UINT64_MAX - base || beats > UINT64_MAX - base - offset) return 0;
    out->weight_begin = base + offset;
    out->weight_beats = beats;
    out->output_row0 = global_first < tile->rows ? (uint32_t)global_first + first : tile->rows;
    out->rows = count;
    out->input_token0 = tile->input_token0;
    out->output_token0 = tile->output_token0;
    out->tokens = tile->tokens;
    return 1;
}

#endif

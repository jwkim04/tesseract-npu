// tsr-kvbuf.h — KV-cache buffers of the ggml-tesseract backend (docs/design/06-software-verification.md
// "SW" §9.3 M7 "KV buffers", SW-17; docs/design/03-attention-kv.md "AT" §4 plane layout, §4.5 region,
// §4.6 correspondence with ggml; docs/design/01-execution-model.md "EX" §11 DRAM map). W3b (P3.5).
//
// A llama.cpp KV cache with the cache offloaded (offload_kqv = true, -ctk q8_0 -ctv q8_0, SW §8 R3)
// is one ggml buffer per cache whose tensors are all `cache_<tag>k_l<n>` / `cache_<tag>v_l<n>`
// (llama.cpp src/llama-kv-cache.cpp:233-237): q8_0, 3D [E, kv_size, n_stream], cleared right after
// allocation (:293). The backend recognizes such a buffer at its first data access (that clear) and
// stores it as KV planes instead of raw bytes:
//   - one plane per (tensor, stream) = per (layer, stream, K/V) (AT §4.5), geometry {E, C, LG} with
//     LG = ceil(kv_size / (16 C)) (tsr_kv_lg_min), K token-major (AT §4.3), V slice-major (§4.4);
//   - planes are allocated symmetrically (the same byte offset on every channel), top-down from the
//     end of the channels (tsr_range_alloc::alloc_top on the WEIGHT heap; EX §11, AT-21), 4 KiB
//     aligned (AT §4.5: recommended, not required);
//   - set / get / memset / clear on byte ranges of the ggml image convert between ggml block_q8_0
//     rows and the planes with AT's C reference (tsr_kv_set_bytes / tsr_kv_get_bytes, a pure byte
//     permutation), so llama.cpp's state save / load is lossless and its files are bit-identical to
//     the CPU's (AT §4.6). Any byte range works: partial blocks and rows, several tensors, streams.
//
// Data path of one access: the touched cells are cut into windows of whole local groups (all
// channels' groups [l0, l1), at most ~8 MiB of plane bytes per window); a window is read from the
// device (K: one beat range per channel; V: one per slice and channel), converted on the host with
// the reference on a window geometry {E, C, l1 - l0} (the cells shifted by 16 C l0: same channel,
// local group l - l0, so the V slice regions stay contiguous) and written back. A window whose
// cells < kv_size are all covered by whole rows of the access is not read (the padding cells of the
// plane beyond kv_size take the last clear value; nothing reads them: FA's n_kv <= kv_size).
//
// The builder (SW §9.3 M7 builder, SET_ROWS_KV / FLASH_ATTN records, SW §7.2 kinds 0x18 / 0x19) takes
// the OPL words of a plane from tsr_kv_plane: base_beat() (SET_ROWS_KV word 8, FLASH_ATTN words 16 /
// 17), geom_word() (SET_ROWS_KV word 9, FLASH_ATTN word 18; K and V of one layer have the same E and
// LG), handle() (the informational cache-handle words: SET_ROWS_KV word 0, FLASH_ATTN words 5 / 6),
// stream (SET_ROWS_KV word 11), and its hazard ranges from ranges() (every channel's plane bytes:
// SET_ROWS_KV writes them, FLASH_ATTN reads K and V). ggml views of a cache tensor (cpy_k / cpy_v
// SET_ROWS dst, get_k / get_v FA operands, src/llama-kv-cache.cpp:1301-1351) resolve by address with
// tsr_kvbuf::resolve (stream and first cell). In ggml-tesseract.cpp the backend's helpers wrap this
// for a ggml tensor (the "W3b kv" sections):
//   static bool tsr_kv_view_of(tsr_device * d, const ggml_tensor * t, tsr_kv_view * out);
//       t (or the tensor it views) lies in a placed KV buffer: plane, stream, first cell.
//   static bool tsr_kv_tensor_supported(tsr_device * d, const ggml_tensor * base,
//                                       tsr_kv_shape * shape = nullptr);
//       supports_op's test (also before placement): base is a KV-cache tensor (tsr_kv_identify) in a
//       Tesseract buffer that is (or will be at its first data access) a KV buffer; its geometry.
//
// No ggml dependency (part of the tsr_lower library; unit test tests/test-tsr-kvbuf.cpp). Not
// thread-safe: the backend calls it with its device mutex held.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tsr-alloc.h"
#include "tsr-opl.h"        // tsr_dram_range, tsr_dram_handle
#include "tsr-wlayout.h"    // tsr_dram_read_fn, tsr_dram_write_fn
#include "tsr_ref_at.h"     // tsr_kv_geom, TSR_KV_K, TSR_KV_V

// ---------------------------------------------------------------------------------------------
// Identification (AT §4.6)
// ---------------------------------------------------------------------------------------------
// "cache_<tag>k_l<n>" -> which = TSR_KV_K, "cache_<tag>v_l<n>" -> TSR_KV_V (tag: any characters,
// llama.cpp's name_tag; n: decimal digits ending the name). False for any other name.
bool tsr_kv_parse_name(const char * name, uint32_t * which, int * layer);

struct tsr_kv_shape {
    uint32_t which = 0;       // TSR_KV_K or TSR_KV_V
    int      layer = -1;      // <n> of the name
    uint32_t E = 0;           // ne[0] = n_head_kv * D: values per cell row
    uint32_t kv_size = 0;     // ne[1]: cells (ggml rows) per stream
    uint32_t n_stream = 0;    // ne[2]
    uint32_t C = 0;           // channels
    uint32_t LG = 0;          // tsr_kv_lg_min(kv_size, C)
    size_t   row_bytes = 0;   // E/32 * 34 (ggml q8_0 row)
    uint64_t plane_bytes = 0; // per channel: tsr_kv_plane_beats * 64
};

// A KV-cache tensor the device can hold (AT §4.1, §4.6, §12): the name above, ggml type q8_0
// (is_q8_0), 3D [E, kv_size, n_stream] with ne[3] = 1 and contiguous q8_0 strides, E % 64 = 0,
// 64 <= E <= K_MAX (k_max 0: not checked) and E/64 <= 511 (the 9-bit E/64 of ATTQK / ATTPV and
// GEOM's field), kv_size >= 1, LG < 2^16 (GEOM's LG field) and the plane under 2^28 beats (a
// handle's beat field). Fills *out (when non-null); *why (when non-null) names the failed rule.
bool tsr_kv_identify(const char * name, bool is_q8_0, const int64_t ne[4], const size_t nb[4], uint32_t num_ch,
                     uint32_t k_max, tsr_kv_shape * out, std::string * why);

// ---------------------------------------------------------------------------------------------
// Planes and buffers
// ---------------------------------------------------------------------------------------------
// The device image of one stream of one KV tensor (AT §4.5: one plane per (layer, stream, K/V)).
struct tsr_kv_plane {
    uint32_t which = 0;         // TSR_KV_K / TSR_KV_V (= SET_ROWS_KV word 7, KVST sub)
    int      layer = -1;
    uint32_t stream = 0;        // SET_ROWS_KV word 11
    uint32_t kv_size = 0;       // ggml rows of the stream (cells 0 .. kv_size-1 are the ggml rows)
    tsr_kv_geom geom = {};      // {E, C, LG}: 16 C LG >= kv_size cells
    uint64_t off = 0;           // byte offset of beat B in every channel (symmetric, 4 KiB aligned)
    uint64_t bytes = 0;         // bytes per channel (tsr_kv_plane_beats(geom) * 64)

    uint32_t base_beat() const { return (uint32_t) (off / TSR_BEAT_BYTES); }      // B
    uint32_t geom_word() const { return tsr::opl::geom_word(geom.E / 64u, geom.LG); }
    uint32_t handle() const { return tsr_dram_handle(0, off); }                    // ch0 beat B
    // [off, off + bytes) on every channel (the hazard ranges of a record that writes / reads it).
    void ranges(std::vector<tsr_dram_range> & out) const;
};

// One KV-cache tensor of a KV buffer.
struct tsr_kv_tensor {
    std::string  name;
    uint64_t     vaddr = 0;     // virtual address of byte 0 (tensor->data)
    size_t       nbytes = 0;    // ggml nbytes = n_stream * kv_size * row_bytes
    tsr_kv_shape shape;
    std::vector<tsr_kv_plane> planes;   // one per stream
};

// Where an address of a KV buffer points (a view's data: view_src->data + view_offs).
struct tsr_kv_view {
    const tsr_kv_tensor * t = nullptr;
    const tsr_kv_plane  * plane = nullptr;   // the stream's plane
    uint32_t stream = 0;
    uint32_t cell0 = 0;         // the cell (row) of the stream the address falls in
    uint32_t byte_in_row = 0;   // offset inside that row (0 for every view the device accepts)
};

class tsr_kvbuf {
public:
    // num_ch: channels; heap: the symmetric WEIGHT heap (planes are taken with alloc_top and given
    // back by release()); read / write: device transfers (channel, byte offset).
    tsr_kvbuf(uint32_t num_ch, tsr_range_alloc * heap, tsr_dram_read_fn read, tsr_dram_write_fn write);
    ~tsr_kvbuf();
    tsr_kvbuf(const tsr_kvbuf &) = delete;
    tsr_kvbuf & operator=(const tsr_kvbuf &) = delete;

    // Before place(): one tensor (its shape from tsr_kv_identify). Tensors must not overlap.
    void add(const std::string & name, uint64_t vaddr, size_t nbytes, const tsr_kv_shape & shape);
    // Allocate every plane, in tensor address order then stream (the first plane is the highest).
    // False with *err (nothing allocated) when the heap has no room. The planes' contents are
    // undefined until clear() (llama.cpp clears every KV buffer after allocating it).
    bool place(std::string * err);
    bool placed() const { return placed_; }
    void release();             // give the planes back (buffer free)

    // Byte ranges of the ggml images, by virtual address: [addr, addr + n) may span tensors but
    // must lie inside tensors (gaps between tensors: abort). fill() writes the byte `value`;
    // clear() fills every plane with `value` (the ggml image of a buffer clear; also the padding).
    void set(uint64_t addr, const void * src, size_t n);
    void get(uint64_t addr, void * dst, size_t n);
    void fill(uint64_t addr, uint8_t value, size_t n);
    void clear(uint8_t value);

    // The tensor / plane / cell of an address (false if it is not inside a tensor of this buffer).
    bool resolve(uint64_t addr, tsr_kv_view * out) const;
    const std::vector<tsr_kv_tensor> & tensors() const { return tensors_; }
    uint64_t device_bytes_per_ch() const;   // sum of the planes' bytes (per channel)

    // Statistics (TSR_STATS / tests): windows converted, of which read from the device first.
    uint64_t st_windows = 0, st_windows_read = 0, st_bytes_in = 0, st_bytes_out = 0;
    // Plane bytes (all channels) per window, default 8 MiB (tests use small windows).
    void set_window_bytes(uint64_t b) { window_bytes_ = b; }

private:
    tsr_kv_tensor * find(uint64_t addr);
    // set (src != nullptr), get (dst != nullptr) or fill (both null) of [rel, rel + n) of one
    // stream's ggml image (rel relative to the stream's first byte).
    void stream_access(const tsr_kv_tensor & t, const tsr_kv_plane & p, uint64_t rel, size_t n, const uint8_t * src,
                       uint8_t * dst, uint8_t value);

    uint32_t num_ch_;
    tsr_range_alloc * heap_;
    tsr_dram_read_fn read_;
    tsr_dram_write_fn write_;
    std::vector<tsr_kv_tensor> tensors_;    // by vaddr
    bool placed_ = false;
    uint8_t clear_value_ = 0;
    uint64_t window_bytes_ = 8ull << 20;
};

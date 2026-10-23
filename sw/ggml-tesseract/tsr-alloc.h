// tsr-alloc.h — first-fit range allocator used for the device DRAM regions
// (IO windows, AUX blobs, the symmetric WEIGHT heap) of the ggml-tesseract backend.
//
// Pure C++17, no ggml dependency (unit-tested in tests/test-tsr-backend-units.cpp).
#pragma once

#include <cstdint>
#include <map>

class tsr_range_alloc {
public:
    tsr_range_alloc() = default;

    // Manage [base, base + size). Every returned offset is a multiple of `align` relative to 0
    // (base itself must be aligned); every allocation is rounded up to `align`.
    void init(uint64_t base, uint64_t size, uint64_t align);

    // First fit. Returns false (and leaves *off untouched) if no free range is large enough.
    bool alloc(uint64_t size, uint64_t * off);

    // Like alloc(), but from the top: the highest free range that fits, allocated at its end.
    // Keeps long-lived small allocations (scratch) away from the bottom-up first-fit ones.
    bool alloc_top(uint64_t size, uint64_t * off);

    // Release an allocation previously returned by alloc(). Aborts on an unknown offset.
    void release(uint64_t off);

    // Size of the allocation at `off` (rounded), or 0 if `off` is not allocated.
    uint64_t size_of(uint64_t off) const;

    uint64_t base()          const { return base_; }
    uint64_t total()         const { return size_; }
    uint64_t used()          const { return used_bytes_; }
    uint64_t free_bytes()    const { return size_ - used_bytes_; }
    uint64_t largest_free()  const;
    size_t   n_allocations() const { return used_.size(); }

    // Consistency check (free + used ranges tile [base, base+size) exactly, no adjacent free
    // ranges). Used by the unit tests.
    bool check() const;

private:
    uint64_t base_  = 0;
    uint64_t size_  = 0;
    uint64_t align_ = 1;
    uint64_t used_bytes_ = 0;
    std::map<uint64_t, uint64_t> free_;   // offset -> size, coalesced
    std::map<uint64_t, uint64_t> used_;   // offset -> size
};

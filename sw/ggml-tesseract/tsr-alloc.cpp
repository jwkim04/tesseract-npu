// tsr-alloc.cpp — see tsr-alloc.h
#include "tsr-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>

static uint64_t tsr_round_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

void tsr_range_alloc::init(uint64_t base, uint64_t size, uint64_t align) {
    if (align == 0 || base % align != 0) {
        fprintf(stderr, "tsr_range_alloc::init: bad alignment (base %llu, align %llu)\n",
                (unsigned long long) base, (unsigned long long) align);
        abort();
    }
    base_  = base;
    size_  = size / align * align;
    align_ = align;
    used_bytes_ = 0;
    free_.clear();
    used_.clear();
    if (size_ > 0) {
        free_[base_] = size_;
    }
}

bool tsr_range_alloc::alloc(uint64_t size, uint64_t * off) {
    const uint64_t need = tsr_round_up(size == 0 ? 1 : size, align_);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->second >= need) {
            const uint64_t o    = it->first;
            const uint64_t rest = it->second - need;
            free_.erase(it);
            if (rest > 0) {
                free_[o + need] = rest;
            }
            used_[o] = need;
            used_bytes_ += need;
            *off = o;
            return true;
        }
    }
    return false;
}

bool tsr_range_alloc::alloc_top(uint64_t size, uint64_t * off) {
    const uint64_t need = tsr_round_up(size == 0 ? 1 : size, align_);
    for (auto it = free_.rbegin(); it != free_.rend(); ++it) {
        if (it->second >= need) {
            const uint64_t fo = it->first;
            const uint64_t fs = it->second;
            const uint64_t o  = fo + fs - need;
            free_.erase(fo);
            if (fs > need) {
                free_[fo] = fs - need;
            }
            used_[o] = need;
            used_bytes_ += need;
            *off = o;
            return true;
        }
    }
    return false;
}

void tsr_range_alloc::release(uint64_t off) {
    auto u = used_.find(off);
    if (u == used_.end()) {
        fprintf(stderr, "tsr_range_alloc::release: offset 0x%llx is not allocated\n", (unsigned long long) off);
        abort();
    }
    uint64_t o = off;
    uint64_t s = u->second;
    used_.erase(u);
    used_bytes_ -= s;

    // coalesce with the following free range
    auto next = free_.lower_bound(o);
    if (next != free_.end() && next->first == o + s) {
        s += next->second;
        next = free_.erase(next);
    }
    // coalesce with the preceding free range
    if (next != free_.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == o) {
            prev->second += s;
            return;
        }
    }
    free_[o] = s;
}

uint64_t tsr_range_alloc::size_of(uint64_t off) const {
    auto u = used_.find(off);
    return u == used_.end() ? 0 : u->second;
}

uint64_t tsr_range_alloc::largest_free() const {
    uint64_t m = 0;
    for (const auto & f : free_) {
        if (f.second > m) {
            m = f.second;
        }
    }
    return m;
}

bool tsr_range_alloc::check() const {
    std::map<uint64_t, std::pair<uint64_t, bool>> all;   // off -> (size, is_free)
    for (const auto & f : free_) all[f.first] = { f.second, true };
    for (const auto & u : used_) {
        if (all.count(u.first)) return false;
        all[u.first] = { u.second, false };
    }
    uint64_t pos = base_;
    bool prev_free = false;
    uint64_t used = 0;
    for (const auto & a : all) {
        if (a.first != pos) return false;
        if (a.second.first == 0 || a.second.first % align_ != 0) return false;
        if (a.second.second && prev_free) return false;   // uncoalesced
        prev_free = a.second.second;
        if (!a.second.second) used += a.second.first;
        pos += a.second.first;
    }
    return pos == base_ + size_ && used == used_bytes_;
}

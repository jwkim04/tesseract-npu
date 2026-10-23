// tsr-lower.cpp — see tsr-lower.h
#include "tsr-lower.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include "tsr_ref.h"

uint32_t tsr_mm_t_max(const tsr_config & cfg) {
    uint32_t t = cfg.t_max;
    t = std::min<uint32_t>(t, TSR_T_MAX_ARCH);
    t = std::min<uint32_t>(t, 15u);            // MVM w5 T field is 4 bits
    return t;
}

uint64_t tsr_mm_lines_per_token(uint32_t fmt, uint32_t N, uint32_t K) {
    if (!tsr_fmt_valid(fmt) || K == 0 || K % 32 != 0) {
        return 0;
    }
    const uint32_t kind = tsr_qa_kind_for_fmt(fmt);
    const uint64_t x  = (uint64_t) K * 4 / TSR_LINE_BYTES;
    const uint64_t qa = tsr_qa_lines(kind, K);
    const uint64_t y  = tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
    return x + qa + y;
}

bool tsr_mm_shape_ok(const tsr_config & cfg, uint32_t fmt, uint64_t N, uint64_t K) {
    if (cfg.beat_bits != 512) {
        return false;                          // v1 software assumes 64-byte beats/lines
    }
    if (!tsr_fmt_valid(fmt)) {
        return false;
    }
    if (N < 1 || N > UINT32_MAX || K < 32 || K > 0xFFFFu || K % 32 != 0) {
        return false;
    }
    if (!tsr_fmt_k_valid(fmt, (uint32_t) K, cfg.k_max)) {
        return false;
    }
    if (tsr_mm_t_max(cfg) < 1) {
        return false;
    }
    const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
    const uint64_t lpt = tsr_mm_lines_per_token(fmt, (uint32_t) N, (uint32_t) K);
    return lpt > 0 && lpt <= spm_lines;
}

bool tsr_mm_plan_tiling(const tsr_config & cfg, uint32_t fmt, uint32_t N, uint32_t K, uint32_t T,
                        tsr_mm_tiling * out) {
    if (T < 1 || !tsr_mm_shape_ok(cfg, fmt, N, K)) {
        return false;
    }
    const uint64_t spm_lines = cfg.spm_bytes / TSR_LINE_BYTES;
    tsr_mm_tiling t = {};
    t.kind     = tsr_qa_kind_for_fmt(fmt);
    t.x_lines  = K * 4 / TSR_LINE_BYTES;
    t.qa_lines = tsr_qa_lines(t.kind, K);
    t.y_lines  = (uint32_t) tsr_ceil_div_u64((uint64_t) N * 4, TSR_LINE_BYTES);
    const uint64_t lpt = (uint64_t) t.x_lines + t.qa_lines + t.y_lines;
    uint64_t tt = std::min<uint64_t>(T, tsr_mm_t_max(cfg));
    tt = std::min<uint64_t>(tt, spm_lines / lpt);
    if (tt < 1) {
        return false;
    }
    t.t_tile  = (uint32_t) tt;
    t.spm_x   = 0;
    t.spm_qa  = t.spm_x  + t.t_tile * t.x_lines;
    t.spm_y   = t.spm_qa + t.t_tile * t.qa_lines;
    t.spm_end = t.spm_y  + t.t_tile * t.y_lines;
    *out = t;
    return true;
}

void tsr_mm_emit(const tsr_config & cfg, const tsr_mm_desc & d,
                 std::vector<tsr_insn> & insns, std::vector<size_t> & cuts) {
    tsr_mm_tiling t;
    if (!tsr_mm_plan_tiling(cfg, d.fmt, d.N, d.K, d.T, &t)) {
        fprintf(stderr, "tsr_mm_emit: MUL_MAT fmt=%u N=%u K=%u T=%u cannot be lowered\n", d.fmt, d.N, d.K, d.T);
        abort();
    }
    const uint64_t beat_lim = 1ull << 32;
    if (d.w_off_beat >= beat_lim ||
        d.x_beat + (uint64_t) d.T * t.x_lines > beat_lim ||
        d.y_beat + (uint64_t) d.T * t.y_lines > beat_lim) {
        fprintf(stderr, "tsr_mm_emit: DRAM beat address out of the 32-bit field\n");
        abort();
    }
    for (uint32_t t0 = 0; t0 < d.T; t0 += t.t_tile) {
        const uint32_t tt = std::min(t.t_tile, d.T - t0);
        insns.push_back(tsr_enc_ld(0, t.spm_x, (uint32_t) (d.x_beat + (uint64_t) t0 * t.x_lines), tt * t.x_lines));
        insns.push_back(tsr_enc_qact(t.kind, t.spm_qa, t.spm_x, d.K, tt, t.x_lines));
        insns.push_back(tsr_enc_mvm(d.fmt, 0, t.spm_y, t.spm_qa, (uint32_t) d.w_off_beat, d.N, d.K, tt, t.y_lines));
        insns.push_back(tsr_enc_st(0, t.spm_y, (uint32_t) (d.y_beat + (uint64_t) t0 * t.y_lines), tt * t.y_lines));
        cuts.push_back(insns.size());
    }
}

static const char * tsr_op_name(uint32_t op) {
    switch (op) {
        case TSR_OP_NOP:   return "NOP";
        case TSR_OP_END:   return "END";
        case TSR_OP_LI:    return "LI";
        case TSR_OP_ADDI:  return "ADDI";
        case TSR_OP_MULI:  return "MULI";
        case TSR_OP_RLD:   return "RLD";
        case TSR_OP_LOOP:  return "LOOP";
        case TSR_OP_WAITM: return "WAITM";
        case TSR_OP_JUMP:  return "JUMP";
        case TSR_OP_LD:    return "LD";
        case TSR_OP_ST:    return "ST";
        case TSR_OP_DQROW: return "DQROW";
        case TSR_OP_QACT:  return "QACT";
        case TSR_OP_MVM:   return "MVM";
        case TSR_OP_VOP:   return "VOP";
        case TSR_OP_RED:   return "RED";
        case TSR_OP_ROPE:  return "ROPE";
        case TSR_OP_TOPK:  return "TOPK";
        case TSR_OP_SOP:   return "SOP";
        case TSR_OP_FENCE: return "FENCE";
        default:           return "???";
    }
}

std::string tsr_insn_str(const tsr_insn & in) {
    char buf[256];
    const uint32_t w0 = in.w[0];
    const uint32_t op = tsr_w0_opcode(w0);
    const uint32_t sub = tsr_w0_sub(w0);
    switch (op) {
        case TSR_OP_LD:
            snprintf(buf, sizeof(buf), "LD    ch%u spm[%u] <- dram[0x%x] x%u lines", sub, in.w[1], in.w[2], in.w[3]);
            break;
        case TSR_OP_ST:
            snprintf(buf, sizeof(buf), "ST    ch%u dram[0x%x] <- spm[%u] x%u lines", sub, in.w[2], in.w[1], in.w[3]);
            break;
        case TSR_OP_QACT:
            snprintf(buf, sizeof(buf), "QACT  %s spm[%u] <- spm[%u] K=%u T=%u src_stride=%u",
                     sub == TSR_QA_A32 ? "A32" : "A256", in.w[1], in.w[2], in.w[3], in.w[4], in.w[5]);
            break;
        case TSR_OP_MVM:
            snprintf(buf, sizeof(buf), "MVM   %s%s spm[%u] <- W[0x%x] x spm[%u] N=%u K=%u T=%u dst_stride=%u",
                     sub == TSR_FMT_Q8_0 ? "Q8_0" : (sub == TSR_FMT_Q4_K ? "Q4_K" : "fmt?"),
                     (tsr_w0_flags(w0) & TSR_MVM_FLAG_ACC) ? ".ACC" : "",
                     in.w[1], in.w[3], in.w[2], in.w[4], in.w[5] & 0xFFFFu, (in.w[5] >> 16) & 0xFu, in.w[6]);
            break;
        default:
            snprintf(buf, sizeof(buf), "%-5s w0=%08x w1=%08x w2=%08x w3=%08x", tsr_op_name(op), w0, in.w[1], in.w[2], in.w[3]);
            break;
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Program cache
// ---------------------------------------------------------------------------

tsr_prog_cache::tsr_prog_cache(size_t max_bytes, size_t max_entry_bytes, size_t max_entries)
    : max_bytes_(max_bytes), max_entry_bytes_(std::min(max_entry_bytes, max_bytes)),
      max_entries_(std::max<size_t>(max_entries, 1)) {}

size_t tsr_prog_cache::cost(const std::string & key, const tsr_prog & p) {
    // instructions + cut points + the key (held twice: map and LRU list) + bookkeeping
    return p.insns.size() * sizeof(tsr_insn) + p.cuts.size() * sizeof(size_t) + 2 * key.size() + 128;
}

bool tsr_prog_cache::cacheable(const std::string & key, const tsr_prog & p) const {
    return cost(key, p) <= max_entry_bytes_;
}

tsr_prog * tsr_prog_cache::find(const std::string & key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
        return nullptr;
    }
    lru_.splice(lru_.begin(), lru_, it->second.lru);
    return &it->second.prog;
}

void tsr_prog_cache::evict_one() {
    const std::string & victim = lru_.back();
    auto it = map_.find(victim);
    bytes_ -= it->second.cost;
    map_.erase(it);
    lru_.pop_back();
    evictions_++;
}

tsr_prog * tsr_prog_cache::insert(const std::string & key, tsr_prog && p) {
    const size_t c = cost(key, p);
    if (c > max_entry_bytes_) {
        fprintf(stderr, "tsr_prog_cache::insert: program of %zu bytes is not cacheable\n", c);
        abort();
    }
    auto old = map_.find(key);
    if (old != map_.end()) {
        bytes_ -= old->second.cost;
        lru_.erase(old->second.lru);
        map_.erase(old);
    }
    while (!lru_.empty() && (bytes_ + c > max_bytes_ || map_.size() + 1 > max_entries_)) {
        evict_one();
    }
    lru_.push_front(key);
    entry & e = map_[key];
    e.prog = std::move(p);
    e.cost = c;
    e.lru  = lru_.begin();
    bytes_ += c;
    return &e.prog;
}

void tsr_prog_cache::clear() {
    map_.clear();
    lru_.clear();
    bytes_ = 0;
}

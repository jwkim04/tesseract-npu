// test_fw_opl_m7.cpp — ctest fw_opl_m7 (W2b infra): the op list M7 kinds' argument words in
// tsr/opl.hpp and the M7 planning glue in tsr/plan.hpp against docs/design/06-software-verification.md
// "SW" §7.2 (kinds 0x11-0x1B, preload entries) and §7.3 (record checks), host C++17 (the backend's
// language: the headers must stay C++17-compatible).
//
//   layout     every argument struct field at its SW §7.2 word index (also static_asserts in opl.hpp),
//              kind_nargs and the slot counts (FLASH_ATTN two slots)
//   roundtrip  per kind 64 random records (and the writer's preload helpers, JOIN folding of more
//              than two predecessors): writer -> blob (CRC) -> reader: kind, slots, deps, n_args and
//              every word equal (args_as), the preload table as added
//   nargs      per M7 kind a record with n_args - 1 and + 1 (add_raw): the reader's record check says
//              BAD_RECORD (as the executor, fw_exec), and check_args too; a FLASH_ATTN record in one
//              slot (its 22 words do not fit): BAD_RECORD
//   validate   per kind the structural rules of opl::check_args (enumerations, flag bits, row periods,
//              softcap, every preload reference's range and alignment, at the boundary and one past)
//   plan       plan_op / check_op dispatch the M7 kinds to their planners (plan_kind gives the same
//              status), BAD_RECORD for a wrong n_args whatever the planner; plan_limits_of; the shared
//              helpers plan_rows (one tile without a divide, the double-buffer rule, SCRATCH) and
//              for_period_segments (row period blocks) against brute force
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "tsr/opl.hpp"
#include "tsr/plan.hpp"

namespace opl = tsr::opl;
namespace plan = tsr::plan;

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...)                                                                   \
    do {                                                                                   \
        ++g_checks;                                                                        \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);                \
            fprintf(stderr, __VA_ARGS__);                                                  \
            fputc('\n', stderr);                                                           \
            if (++g_fail >= 40) {                                                          \
                fprintf(stderr, "too many failures, stopping\n");                          \
                return;                                                                    \
            }                                                                              \
        }                                                                                  \
    } while (0)

struct rng {
    uint64_t s;
    uint32_t u32() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return uint32_t((z ^ (z >> 31)) >> 32);
    }
};

static const uint32_t M7_KINDS[] = {opl::K_ADD, opl::K_MUL, opl::K_SCALE, opl::K_RMS_NORM, opl::K_GLU, opl::K_ROPE,
                                    opl::K_GET_ROWS, opl::K_SET_ROWS_KV, opl::K_FLASH_ATTN, opl::K_CPY, opl::K_UNARY};
static const char * kind_name(uint32_t k) {
    switch (k) {
        case opl::K_ADD: return "ADD";
        case opl::K_MUL: return "MUL";
        case opl::K_SCALE: return "SCALE";
        case opl::K_RMS_NORM: return "RMS_NORM";
        case opl::K_GLU: return "GLU";
        case opl::K_ROPE: return "ROPE";
        case opl::K_GET_ROWS: return "GET_ROWS";
        case opl::K_SET_ROWS_KV: return "SET_ROWS_KV";
        case opl::K_FLASH_ATTN: return "FLASH_ATTN";
        case opl::K_CPY: return "CPY";
        case opl::K_UNARY: return "UNARY";
        default: return "?";
    }
}

// --- a valid record of each kind (structurally: check_args OK with PRE_LINES preload lines) -------
static const uint32_t PRE_LINES = 8;   // the lists below: 512 preload bytes
static std::vector<uint32_t> valid_words(uint32_t kind, rng & r) {
    std::vector<uint32_t> w(opl::kind_nargs(kind));
    for (auto & x : w) x = r.u32();
    switch (kind) {
        case opl::K_ADD:
        case opl::K_MUL:
            w[opl::ew::FLAGS] = r.u32() & opl::ew::FLAGS_KNOWN;
            w[opl::ew::B_PERIOD] = 1u + (r.u32() & 0xFFu);
            break;
        case opl::K_RMS_NORM:
            if (r.u32() & 1u) w[opl::rms::W] = opl::HANDLE_NONE;
            else w[opl::rms::W_PERIOD] = 1u + (r.u32() & 0xFFu);
            break;
        case opl::K_GLU: w[opl::glu::VARIANT] = r.u32() % (opl::glu::VARIANT_MAX + 1u); break;
        case opl::K_ROPE:
            w[opl::rope::MODE] = r.u32() & opl::rope::MODE_KNOWN;
            w[opl::rope::N_TOKENS] = r.u32() % 64u;
            w[opl::rope::POS] = 64u * (r.u32() % 4u);                     // + 4 x 63 <= 512
            break;
        case opl::K_GET_ROWS:
            w[opl::getrows::N_IDS] = r.u32() % 64u;
            w[opl::getrows::IDS] = 4u * (r.u32() % 64u);                   // + 4 x 63 <= 508
            break;
        case opl::K_SET_ROWS_KV:
            w[opl::setrows::WHICH] = r.u32() & 1u;
            w[opl::setrows::N_ROWS] = r.u32() % 32u;
            w[opl::setrows::IDX] = 8u * (r.u32() % 32u);                   // + 8 x 31 <= 504
            break;
        case opl::K_FLASH_ATTN:
            w[opl::fa::SOFTCAP] = 0u;
            w[opl::fa::N_TOKENS] = r.u32() % 16u;
            w[opl::fa::MSUM] = 64u * (r.u32() % 3u);                       // + 32 + 16 x 15 <= 400
            break;
        case opl::K_UNARY: w[opl::unary::FUNC] = r.u32() % (opl::unary::FUNC_MAX + 1u); break;
        default: break;
    }
    return w;
}

// The writer's named helper of `kind` with the struct decoded from the words (so the helpers and the
// structs are exercised, not only add()).
static uint32_t add_named(opl::writer & w, uint32_t kind, const std::vector<uint32_t> & a, const uint32_t * deps,
                          uint32_t nd, uint32_t kflags) {
    switch (kind) {
        case opl::K_ADD: return w.add_add(opl::get_args<opl::ew_args>(a.data()), deps, nd, kflags);
        case opl::K_MUL: return w.add_mul(opl::get_args<opl::ew_args>(a.data()), deps, nd, kflags);
        case opl::K_SCALE: return w.add_scale(opl::get_args<opl::scale_args>(a.data()), deps, nd, kflags);
        case opl::K_RMS_NORM: return w.add_rms_norm(opl::get_args<opl::rms_norm_args>(a.data()), deps, nd, kflags);
        case opl::K_GLU: return w.add_glu(opl::get_args<opl::glu_args>(a.data()), deps, nd, kflags);
        case opl::K_ROPE: return w.add_rope(opl::get_args<opl::rope_args>(a.data()), deps, nd, kflags);
        case opl::K_GET_ROWS: return w.add_get_rows(opl::get_args<opl::get_rows_args>(a.data()), deps, nd, kflags);
        case opl::K_SET_ROWS_KV:
            return w.add_set_rows_kv(opl::get_args<opl::set_rows_kv_args>(a.data()), deps, nd, kflags);
        case opl::K_FLASH_ATTN:
            return w.add_flash_attn(opl::get_args<opl::flash_attn_args>(a.data()), deps, nd, kflags);
        case opl::K_CPY: return w.add_cpy(opl::get_args<opl::cpy_args>(a.data()), deps, nd, kflags);
        case opl::K_UNARY: return w.add_unary(opl::get_args<opl::unary_args>(a.data()), deps, nd, kflags);
        default: return opl::HANDLE_NONE;
    }
}
// args_as of the record's kind == the words.
static bool args_as_equal(const opl::record & r, const std::vector<uint32_t> & a) {
    auto eq = [&](const auto & s) {
        uint32_t w[32] = {};
        opl::put_args(s, w);
        return memcmp(w, a.data(), 4u * a.size()) == 0;
    };
    switch (r.kind) {
        case opl::K_ADD:
        case opl::K_MUL: { opl::ew_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_SCALE: { opl::scale_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_RMS_NORM: { opl::rms_norm_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_GLU: { opl::glu_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_ROPE: { opl::rope_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_GET_ROWS: { opl::get_rows_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_SET_ROWS_KV: { opl::set_rows_kv_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_FLASH_ATTN: { opl::flash_attn_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_CPY: { opl::cpy_args s{}; return r.args_as(&s) && eq(s); }
        case opl::K_UNARY: { opl::unary_args s{}; return r.args_as(&s) && eq(s); }
        default: return false;
    }
}

// --- layout ------------------------------------------------------------------------------------
static void t_layout() {
    const uint32_t want[11] = {10, 10, 8, 10, 9, 14, 7, 12, 22, 6, 7};   // SW §7.2 table
    for (uint32_t i = 0; i < 11; ++i) {
        const uint32_t k = M7_KINDS[i];
        CHECK(opl::kind_nargs(k) == want[i], "%s: n_args %u (SW §7.2: %u)", kind_name(k), opl::kind_nargs(k), want[i]);
        CHECK(opl::slots_for_nargs(want[i]) == (want[i] + 2u > 16u ? 2u : 1u), "%s: slots", kind_name(k));
    }
    // A few spot checks of named words (the full mapping is static_asserted in opl.hpp).
    opl::flash_attn_args f{};
    f.msum = 0x1234u;
    f.flags = 0x55u;
    uint32_t w[22] = {};
    opl::put_args(f, w);
    CHECK(w[12] == 0x1234u && w[21] == 0x55u, "FLASH_ATTN: MSUM word 12, flags word 21");
    opl::rope_args rp{};
    rp.pos = 64u;
    rp.mode = 3u;
    uint32_t wr[14] = {};
    opl::put_args(rp, wr);
    CHECK(wr[4] == 64u && wr[13] == 3u, "ROPE: pos word 4, mode word 13");
    CHECK(opl::geom_word(8u, 24u) == (8u | (24u << 16)) && opl::geom_e64(opl::geom_word(8u, 24u)) == 8u &&
              opl::geom_lg(opl::geom_word(8u, 24u)) == 24u, "GEOM word E/64 | LG << 16");
    CHECK(opl::msum_bytes(0u) == 32u && opl::msum_bytes(3u) == 32u + 48u, "MSUM image size");
}

// --- roundtrip -----------------------------------------------------------------------------------
static void t_roundtrip() {
    rng r{0x5eed0007u};
    for (uint32_t pass = 0; pass < 64u; ++pass) {
        opl::writer w(opl::F_STOP_ON_ERROR, 0xC0FFEE00u + pass);
        // Preload entries through the M7 helpers: sizes as SW §7.2 states them.
        const uint32_t p_pos = w.pre_positions(0x10000040u, 3u);
        const uint32_t p_ids = w.pre_ids(0x10000080u, 5u);
        const uint32_t p_idx = w.pre_kv_idx(0x100000C0u, 2u);
        const uint32_t p_ms = w.pre_msum(0x10000100u, 4u);
        // A plain entry fills the area to PRE_LINES lines (valid_words' preload references assume 512 B).
        const uint32_t p_pad = w.add_preload(0x10000200u, 512u - 320u);
        CHECK(p_pos == 0u && p_ids == 64u && p_idx == 128u && p_ms == 192u && p_pad == 320u,
              "preload offsets %u %u %u %u %u", p_pos, p_ids, p_idx, p_ms, p_pad);
        std::vector<std::vector<uint32_t>> words;
        std::vector<uint32_t> kinds, kfl, idx;
        std::vector<std::vector<uint32_t>> dps;
        for (uint32_t i = 0; i < 11u; ++i) {
            const uint32_t k = M7_KINDS[(i + pass) % 11u];
            const std::vector<uint32_t> a = valid_words(k, r);
            // Predecessors: none, one, two or (folded by JOINs) four earlier ops.
            std::vector<uint32_t> d;
            const uint32_t n_ops = w.n_ops();
            const uint32_t nd = n_ops == 0u ? 0u : r.u32() % 5u;
            for (uint32_t j = 0; j < nd; ++j) d.push_back(r.u32() % n_ops);
            const uint32_t kf = r.u32() & 0xFFu;
            const uint32_t at = add_named(w, k, a, d.data(), uint32_t(d.size()), kf);
            CHECK(at != opl::HANDLE_NONE && w.ok(), "%s: writer refused (%s)", kind_name(k), w.error());
            words.push_back(a);
            kinds.push_back(k);
            kfl.push_back(kf);
            idx.push_back(at);
            dps.push_back(d);
        }
        const std::vector<uint8_t> blob = w.finish();
        CHECK(!blob.empty(), "pass %u: finish", pass);
        opl::reader rd;
        CHECK(rd.open(blob.data(), uint32_t(blob.size()), true) == opl::RET_OK, "pass %u: open", pass);
        CHECK(rd.hdr().n_pre == 5u && rd.hdr().pre_lines == PRE_LINES, "pass %u: preload table", pass);
        CHECK(rd.pre(0).bytes == 12u && rd.pre(1).bytes == 20u && rd.pre(2).bytes == 16u &&
                  rd.pre(3).bytes == opl::msum_bytes(4u) && rd.pre(3).pre_byte == 192u && rd.pre(1).src == 0x10000080u,
              "pass %u: preload entries", pass);
        uint32_t m = 0;
        opl::record rec{};
        for (uint32_t i = 0; i < rd.hdr().n_ops; ++i) {
            const uint32_t e = rd.next(rec);
            CHECK(e == opl::RET_OK, "pass %u op %u: next %08x", pass, i, e);
            if (rec.kind == opl::K_JOIN) continue;
            CHECK(m < kinds.size() && rec.idx == idx[m], "pass %u: op order", pass);
            if (m >= kinds.size()) break;
            CHECK(rec.kind == kinds[m] && rec.kflags == kfl[m] && rec.n_args == words[m].size() &&
                      rec.n_slots == opl::slots_for_nargs(rec.n_args),
                  "pass %u op %u: %s header", pass, i, kind_name(kinds[m]));
            CHECK(args_as_equal(rec, words[m]), "pass %u op %u: %s words", pass, i, kind_name(kinds[m]));
            CHECK(rd.check_args(rec) == opl::RET_OK, "pass %u op %u: %s check_args", pass, i, kind_name(kinds[m]));
            // Deps: for <= 2 distinct predecessors exactly them; more: the op's deps cover them through JOINs.
            std::vector<uint32_t> want = dps[m];
            std::vector<uint32_t> uniq;
            for (uint32_t x : want) {
                bool dup = false;
                for (uint32_t y : uniq) dup = dup || x == y;
                if (!dup) uniq.push_back(x);
            }
            if (uniq.size() <= 2u) {
                const uint32_t d0 = uniq.size() > 0 ? uniq[0] : opl::DEP_NONE, d1 = uniq.size() > 1 ? uniq[1] : opl::DEP_NONE;
                CHECK(rec.dep0 == d0 && rec.dep1 == d1, "pass %u op %u: deps %u %u (want %u %u)", pass, i, rec.dep0,
                      rec.dep1, d0, d1);
            } else {
                CHECK(rec.dep0 != opl::DEP_NONE && rec.dep0 < rec.idx && rec.dep1 < rec.idx, "pass %u op %u: folded deps",
                      pass, i);
            }
            ++m;
        }
        CHECK(m == kinds.size(), "pass %u: %u M7 records read of %zu", pass, m, kinds.size());
        CHECK(rd.next(rec) == opl::RET_BAD_RECORD, "pass %u: past the last op", pass);
    }
    // f32 bits: the writer's helper is the IEEE pattern.
    CHECK(opl::writer::f32_bits(1.0f) == 0x3F800000u && opl::writer::f32_bits(-0.0f) == 0x80000000u, "f32_bits");
}

// --- nargs / slots ---------------------------------------------------------------------------------
static void t_nargs() {
    rng r{77u};
    for (uint32_t k : M7_KINDS) {
        for (int delta = -1; delta <= 1; delta += 2) {
            std::vector<uint32_t> a = valid_words(k, r);
            const uint32_t n = uint32_t(int(a.size()) + delta);
            a.resize(n + 1u, 0u);
            opl::writer w;
            w.add_raw(k, a.data(), n, opl::DEP_NONE, opl::DEP_NONE);
            const std::vector<uint8_t> blob = w.finish();
            opl::reader rd;
            CHECK(rd.open(blob.data(), uint32_t(blob.size()), false) == opl::RET_OK, "%s: open", kind_name(k));
            opl::record rec{};
            CHECK(rd.next(rec) == opl::RET_BAD_RECORD, "%s with %u arguments: BAD_RECORD", kind_name(k), n);
            CHECK(opl::check_args(k, a.data(), n, PRE_LINES) == opl::RET_BAD_RECORD, "%s: check_args with %u arguments",
                  kind_name(k), n);
            const uint32_t st = plan::check_op(plan::U200_K4, k, a.data(), n);
            CHECK(st == opl::RET_BAD_RECORD, "%s: plan check_op with %u arguments: %08x", kind_name(k), n, st);
        }
    }
    // FLASH_ATTN needs two slots: one slot holds 14 argument words.
    {
        std::vector<uint32_t> a = valid_words(opl::K_FLASH_ATTN, r);
        opl::writer w;
        w.add_raw(opl::K_FLASH_ATTN, a.data(), opl::fa::NARGS, opl::DEP_NONE, opl::DEP_NONE, 0, 1);
        const std::vector<uint8_t> blob = w.finish();
        opl::reader rd;
        CHECK(rd.open(blob.data(), uint32_t(blob.size()), false) == opl::RET_OK, "FA one slot: open");
        opl::record rec{};
        CHECK(rd.next(rec) == opl::RET_BAD_RECORD, "FLASH_ATTN in one slot: BAD_RECORD");
    }
    // A record with more slots than it needs is legal (padding slots), and args_as refuses a struct of
    // another layout.
    {
        std::vector<uint32_t> a = valid_words(opl::K_CPY, r);
        opl::writer w;
        w.add_raw(opl::K_CPY, a.data(), opl::cpy::NARGS, opl::DEP_NONE, opl::DEP_NONE, 0, 3);
        const std::vector<uint8_t> blob = w.finish();
        opl::reader rd;
        CHECK(rd.open(blob.data(), uint32_t(blob.size()), false) == opl::RET_OK, "CPY 3 slots: open");
        opl::record rec{};
        CHECK(rd.next(rec) == opl::RET_OK && rec.n_slots == 3u, "CPY in three slots");
        opl::ew_args e{};
        CHECK(!rec.args_as(&e), "args_as with another kind's layout");
    }
}

// --- validate ------------------------------------------------------------------------------------
static uint32_t chk(uint32_t k, const std::vector<uint32_t> & a, uint32_t pre_lines = PRE_LINES) {
    return opl::check_args(k, a.data(), uint32_t(a.size()), pre_lines);
}
static void t_validate() {
    const uint32_t OK = opl::RET_OK, BAD = opl::RET_BAD_RECORD;
    const uint32_t PB = 64u * PRE_LINES;
    rng r{1234u};
    for (uint32_t k : M7_KINDS) CHECK(chk(k, valid_words(k, r)) == OK, "%s: a valid record", kind_name(k));
    for (uint32_t k : {opl::K_ADD, opl::K_MUL}) {
        auto a = valid_words(k, r);
        a[opl::ew::FLAGS] = 2u;
        CHECK(chk(k, a) == BAD, "%s: undefined flag bit 1", kind_name(k));
        a[opl::ew::FLAGS] = opl::ew::FLAG_B_SCALAR;
        CHECK(chk(k, a) == OK, "%s: B_SCALAR", kind_name(k));
        a[opl::ew::B_PERIOD] = 0u;
        CHECK(chk(k, a) == BAD, "%s: b row period 0", kind_name(k));
    }
    {
        auto a = valid_words(opl::K_RMS_NORM, r);
        a[opl::rms::W] = 0x10000000u;
        a[opl::rms::W_PERIOD] = 0u;
        CHECK(chk(opl::K_RMS_NORM, a) == BAD, "RMS_NORM: a weight with period 0");
        a[opl::rms::W] = opl::HANDLE_NONE;
        CHECK(chk(opl::K_RMS_NORM, a) == OK, "RMS_NORM: no weight, period ignored");
    }
    {
        auto a = valid_words(opl::K_GLU, r);
        a[opl::glu::VARIANT] = 3u;
        CHECK(chk(opl::K_GLU, a) == BAD, "GLU: variant 3");
    }
    {
        auto a = valid_words(opl::K_UNARY, r);
        a[opl::unary::FUNC] = 6u;
        CHECK(chk(opl::K_UNARY, a) == BAD, "UNARY: func 6");
        a[opl::unary::FUNC] = opl::unary::SIGMOID;
        CHECK(chk(opl::K_UNARY, a) == OK, "UNARY: SIGMOID");
        a[opl::unary::FUNC] = opl::unary::GELU;
        CHECK(chk(opl::K_UNARY, a) == OK, "UNARY: GELU");
    }
    {
        auto a = valid_words(opl::K_ROPE, r);
        a[opl::rope::MODE] = 4u;
        CHECK(chk(opl::K_ROPE, a) == BAD, "ROPE: mode bit 2");
        a[opl::rope::MODE] = opl::rope::MODE_NEOX | opl::rope::MODE_ATAB;
        a[opl::rope::N_TOKENS] = 16u;
        a[opl::rope::POS] = PB - 64u;
        CHECK(chk(opl::K_ROPE, a) == OK, "ROPE: 16 positions in the last preload line");
        a[opl::rope::N_TOKENS] = 17u;
        CHECK(chk(opl::K_ROPE, a) == BAD, "ROPE: 17 positions past the preload area");
        a[opl::rope::N_TOKENS] = 1u;
        a[opl::rope::POS] = 4u;
        CHECK(chk(opl::K_ROPE, a) == BAD, "ROPE: position offset not line-aligned");
        a[opl::rope::POS] = 0u;
        CHECK(chk(opl::K_ROPE, a, 0u) == BAD, "ROPE: a position without a preload area");
        a[opl::rope::N_TOKENS] = 0u;
        CHECK(chk(opl::K_ROPE, a, 0u) == OK, "ROPE: no tokens, no positions");
        a[opl::rope::N_TOKENS] = 0x40000000u;   // 4 x n wraps 32 bits: must not pass
        CHECK(chk(opl::K_ROPE, a) == BAD, "ROPE: n_tokens whose byte count wraps");
    }
    {
        auto a = valid_words(opl::K_GET_ROWS, r);
        a[opl::getrows::N_IDS] = 1u;
        a[opl::getrows::IDS] = PB - 4u;
        CHECK(chk(opl::K_GET_ROWS, a) == OK, "GET_ROWS: the last word");
        a[opl::getrows::N_IDS] = 2u;
        CHECK(chk(opl::K_GET_ROWS, a) == BAD, "GET_ROWS: one id past the area");
        a[opl::getrows::N_IDS] = 1u;
        a[opl::getrows::IDS] = 2u;
        CHECK(chk(opl::K_GET_ROWS, a) == BAD, "GET_ROWS: ids not 4-aligned");
        a[opl::getrows::IDS] = 0xFFFFFFFCu;
        CHECK(chk(opl::K_GET_ROWS, a) == BAD, "GET_ROWS: offset past 2^32 - 4");
    }
    {
        auto a = valid_words(opl::K_SET_ROWS_KV, r);
        a[opl::setrows::WHICH] = 2u;
        CHECK(chk(opl::K_SET_ROWS_KV, a) == BAD, "SET_ROWS_KV: which 2");
        a[opl::setrows::WHICH] = opl::setrows::WHICH_V;
        a[opl::setrows::N_ROWS] = 1u;
        a[opl::setrows::IDX] = PB - 8u;
        CHECK(chk(opl::K_SET_ROWS_KV, a) == OK, "SET_ROWS_KV: the last index");
        a[opl::setrows::IDX] = PB - 4u;
        CHECK(chk(opl::K_SET_ROWS_KV, a) == BAD, "SET_ROWS_KV: index 4-aligned only");
        a[opl::setrows::IDX] = 0u;
        a[opl::setrows::N_ROWS] = PB / 8u + 1u;
        CHECK(chk(opl::K_SET_ROWS_KV, a) == BAD, "SET_ROWS_KV: too many indices");
    }
    {
        auto a = valid_words(opl::K_FLASH_ATTN, r);
        a[opl::fa::SOFTCAP] = 0x3F800000u;
        CHECK(chk(opl::K_FLASH_ATTN, a) == BAD, "FLASH_ATTN: softcap 1.0");
        a[opl::fa::SOFTCAP] = 0x80000000u;
        CHECK(chk(opl::K_FLASH_ATTN, a) == BAD, "FLASH_ATTN: softcap -0.0 (the word must be 0)");
        a[opl::fa::SOFTCAP] = 0u;
        a[opl::fa::N_TOKENS] = (PB - 32u) / 16u;   // 30 rows: exactly the area
        a[opl::fa::MSUM] = 0u;
        CHECK(chk(opl::K_FLASH_ATTN, a) == OK, "FLASH_ATTN: an MSUM filling the area");
        a[opl::fa::N_TOKENS] += 1u;
        CHECK(chk(opl::K_FLASH_ATTN, a) == BAD, "FLASH_ATTN: MSUM one row past the area");
        a[opl::fa::N_TOKENS] = 1u;
        a[opl::fa::MSUM] = 32u;
        CHECK(chk(opl::K_FLASH_ATTN, a) == BAD, "FLASH_ATTN: MSUM not line-aligned");
    }
    // Kinds without M7 words pass check_args (their checks are elsewhere).
    CHECK(opl::check_args(opl::K_NOP, nullptr, 0u, 0u) == OK && opl::check_args(0x3Fu, nullptr, 7u, 0u) == OK,
          "NOP / undefined kinds");
}

// --- plan ----------------------------------------------------------------------------------------
static void t_plan() {
    // plan_limits_of = the documented derivation.
    const plan::plan_limits L = plan::plan_limits_of(plan::U200_K4);
    CHECK(L.budget == 65536u - 16u - 4096u - 256u - 1024u && L.t_max == 8u && L.k_max == 32768u && L.num_ch == 4u &&
              L.pcap == 9344u && L.vl == 16u && L.caps == (plan::CAPS_VU | plan::CAPS_ATT) && L.spm_lines == 65536u,
          "plan_limits_of(U200_K4)");
    const plan::dev_cfg m6only{65536u, 8u, 32768u};   // a host that fills only the M6 fields
    CHECK(m6only.num_ch == 0u && m6only.pcap == 0u && m6only.caps == 0u, "dev_cfg: M7 fields default to unknown (0)");

    // plan_op / check_op = plan_kind's status for every kind and some records; a planner's status is
    // one of the three codes and scratch is reported only with OK.
    rng r{99u};
    for (uint32_t k : M7_KINDS) {
        int n_ok = 0;
        for (int i = 0; i < 200; ++i) {
            std::vector<uint32_t> a = valid_words(k, r);
            tsr::kind_plan kp{};
            const bool ok = plan::plan_kind(L, k, a.data(), uint32_t(a.size()), &kp);
            const plan::op_plan op = plan::plan_op(plan::U200_K4, k, a.data(), uint32_t(a.size()));
            CHECK(op.status == kp.status && ok == (kp.status == opl::RET_OK), "%s: plan_op %08x vs plan_kind %08x",
                  kind_name(k), op.status, kp.status);
            CHECK(kp.status == opl::RET_OK || kp.status == opl::RET_BAD_RECORD || kp.status == opl::RET_SCRATCH,
                  "%s: status %08x", kind_name(k), kp.status);
            CHECK(ok ? op.scratch_lines == kp.scratch_lines : op.scratch_lines == plan::SCRATCH_UNSUPPORTED,
                  "%s: scratch figure", kind_name(k));
            n_ok += ok;
        }
        printf("fw_opl_m7: %-11s planner: %3d of 200 random structurally valid records plan on the U200\n", kind_name(k),
               n_ok);
    }
    {
        tsr::kind_plan kp{};
        CHECK(!plan::plan_kind(L, opl::K_MUL_MAT, nullptr, 0u, &kp) && kp.status == opl::RET_BAD_RECORD &&
                  kp.scratch_lines == plan::SCRATCH_UNSUPPORTED,
              "plan_kind: not an M7 kind");
    }

    // plan_rows: the rule against brute force over small budgets.
    for (uint32_t budget = 1; budget <= 64u; ++budget)
        for (uint32_t lpr = 1; lpr <= 20u; ++lpr)
            for (uint32_t rows = 0; rows <= 70u; ++rows) {
                plan::row_tiling t{};
                const uint32_t st = plan::plan_rows(rows, lpr, budget, &t);
                if (lpr > budget) {
                    CHECK(st == opl::RET_SCRATCH, "plan_rows(%u, %u, %u): SCRATCH", rows, lpr, budget);
                    continue;
                }
                CHECK(st == opl::RET_OK, "plan_rows(%u, %u, %u): OK", rows, lpr, budget);
                if (rows * lpr <= budget) {
                    CHECK(t.tile_rows == rows && t.n_tiles == (rows ? 1u : 0u), "plan_rows(%u, %u, %u): one tile",
                          rows, lpr, budget);
                } else {
                    const uint32_t tr = (budget / 2u) / lpr ? (budget / 2u) / lpr : 1u;
                    CHECK(t.tile_rows == tr && t.n_tiles == (rows + tr - 1u) / tr && t.tile_rows * lpr <= budget,
                          "plan_rows(%u, %u, %u): %u x %u", rows, lpr, budget, t.tile_rows, t.n_tiles);
                }
            }
    {
        plan::row_tiling t{};
        CHECK(plan::plan_rows(5u, 0u, 100u, &t) == opl::RET_BAD_RECORD, "plan_rows: lpr 0");
        CHECK(plan::plan_rows(70000u, 1u, 0xFFFFFFFFu, &t) == opl::RET_OK && t.tile_rows == 0xFFFFu && t.n_tiles == 2u,
              "plan_rows: the 16-bit rows field");
    }
    // for_period_segments: every row once, in order, b row = r mod p (0 for p <= 1), segments split
    // exactly where r mod p wraps.
    for (uint32_t p = 0; p <= 9u; ++p)
        for (uint32_t r0 = 0; r0 <= 20u; ++r0)
            for (uint32_t n = 0; n <= 20u; ++n) {
                uint32_t next = r0, segs = 0;
                bool ok = true;
                const uint32_t cnt = plan::for_period_segments(r0, n, p, [&](uint32_t row, uint32_t len, uint32_t brow, bool bc) {
                    ok = ok && row == next && len >= 1u && bc == (p <= 1u);
                    for (uint32_t j = 0; j < len && ok; ++j) {
                        const uint32_t want = p <= 1u ? 0u : (row + j) % p;
                        const uint32_t got = bc ? 0u : brow + j;
                        ok = ok && want == got;
                    }
                    next = row + len;
                    ++segs;
                });
                uint32_t want_segs = 0;
                if (n != 0u) {
                    if (p <= 1u) want_segs = 1u;
                    else
                        for (uint32_t row = r0; row < r0 + n; ++row) want_segs += (row == r0 || row % p == 0u) ? 1u : 0u;
                }
                CHECK(ok && next == r0 + n && segs == cnt && cnt == want_segs, "period segments r0 %u n %u p %u: %u (%u)",
                      r0, n, p, cnt, want_segs);
            }
    CHECK(plan::vu_lines(0u) == 0u && plan::vu_lines(1u) == 1u && plan::vu_lines(16u) == 1u && plan::vu_lines(17u) == 2u &&
              plan::vu_lines(896u) == 56u && plan::vu_lines(0xFFFFFFFFu) == 0x10000000u,
          "vu_lines");
}

int main() {
    t_layout();
    t_roundtrip();
    t_nargs();
    t_validate();
    t_plan();
    if (g_fail) {
        fprintf(stderr, "fw_opl_m7: %d FAILED of %ld checks\n", g_fail, g_checks);
        return 1;
    }
    printf("fw_opl_m7: all %ld checks passed\n", g_checks);
    return 0;
}

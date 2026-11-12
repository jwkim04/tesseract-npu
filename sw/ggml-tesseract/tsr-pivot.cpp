// tsr-pivot.cpp — Tesseract Pivot core: the machine model, the validator and the hoist search
// (header: tsr-pivot.h). No ggml, no floating point.
#include "tsr-pivot.h"

#include <algorithm>

namespace {

constexpr uint32_t NONE = TSR_PIVOT_NONE;

inline int64_t max3(int64_t a, int64_t b, int64_t c) { return std::max(a, std::max(b, c)); }

// The clocks of the machine model and the open deferred units (in the order they were opened). A copy of
// it is the state saved at a position of an order.
struct mstate {
    int64_t  t_issue = 0;
    int64_t  t_glue  = 0;
    int64_t  t_qact  = 0;
    int64_t  t_mvm   = 0;
    uint32_t n_open  = 0;
    uint32_t open[TSR_PIVOT_MAX_OPEN] = {};
};

inline bool has_mvm(const tsr_pivot_unit & u) {
    return u.cls == TSR_PIVOT_MVM || u.cls == TSR_PIVOT_MVM_ID || u.cls == TSR_PIVOT_ATTN;
}

inline bool reads(const std::vector<tsr_pivot_unit> & units, uint32_t reader, uint32_t u) {
    if (reader == NONE) return false;
    for (uint32_t p : units[reader].preds) if (p == u) return true;
    return false;
}

// The model's steps. The per-unit `done` and MVM end live in two layers: the base layer, written by the
// evaluation of the current order, and a scratch layer for one trial (a stamp says which entries the trial
// wrote), so that a trial leaves the base layer of the positions before its start as it was.
class machine {
public:
    explicit machine(const std::vector<tsr_pivot_unit> & units)
        : U(units), done_(units.size(), 0), mend_(units.size(), 0), cdone_(units.size(), 0),
          cmend_(units.size(), 0), dst_(units.size(), 0), mst_(units.size(), 0) {}

    void begin_scratch() { scratch_ = true; ++gen_; }
    void end_scratch()   { scratch_ = false; }

    // Process unit u; `next` is the unit after it in the order (NONE at the end).
    void step(mstate & s, uint32_t u, uint32_t next) {
        const tsr_pivot_unit & x = U[u];
        const bool opens = x.cls == TSR_PIVOT_MVM && x.defer && !reads(U, next, u);

        // the open units among the preds are closed, in the order they were opened
        if (s.n_open) {
            uint32_t w = 0;
            for (uint32_t k = 0; k < s.n_open; k++) {
                const uint32_t v = s.open[k];
                bool is_pred = false;
                for (uint32_t p : x.preds) if (p == v) { is_pred = true; break; }
                if (is_pred) close(s, v); else s.open[w++] = v;
            }
            s.n_open = w;
        }
        // a ninth open unit closes the oldest first
        if (opens && s.n_open == TSR_PIVOT_MAX_OPEN) {
            close(s, s.open[0]);
            for (uint32_t k = 1; k < s.n_open; k++) s.open[k - 1] = s.open[k];
            s.n_open--;
        }

        s.t_issue += (x.cls == TSR_PIVOT_MVM && opens) ? x.issue_open : x.issue;
        int64_t ready = 0;
        for (uint32_t p : x.preds) ready = std::max(ready, get_done(p));
        const int64_t t0 = max3(s.t_glue, s.t_issue, ready);

        switch (x.cls) {
        case TSR_PIVOT_GLUE:
            s.t_glue = t0 + x.body;
            set_done(u, s.t_glue);
            break;
        case TSR_PIVOT_MVM: {
            s.t_glue = t0 + x.ld;
            s.t_qact = std::max(s.t_glue, s.t_qact) + x.qact;
            const int64_t mvm_end = std::max(s.t_qact, s.t_mvm) + x.mvm;
            s.t_mvm = mvm_end;
            set_mend(u, mvm_end);
            if (opens) {
                s.open[s.n_open++] = u;
            } else {
                s.t_glue = std::max(s.t_glue, mvm_end) + x.back;
                set_done(u, s.t_glue);
            }
            break;
        }
        case TSR_PIVOT_MVM_ID: {
            s.t_glue  = t0 + x.ids;
            s.t_issue = std::max(s.t_issue, s.t_glue);
            const int64_t mvm_end = std::max(s.t_glue, s.t_mvm) + x.mvm;
            s.t_mvm   = mvm_end;
            s.t_issue = std::max(s.t_issue, mvm_end - x.tail);
            s.t_glue  = mvm_end + x.back;
            set_done(u, s.t_glue);
            break;
        }
        case TSR_PIVOT_BLOCK:
            s.t_glue = s.t_issue = t0 + x.body;
            set_done(u, s.t_glue);
            break;
        case TSR_PIVOT_ATTN: {
            const int64_t mvm_end = std::max(t0 + x.front, s.t_mvm) + x.mvm;
            s.t_mvm  = mvm_end;
            s.t_glue = mvm_end + x.back;
            set_done(u, s.t_glue);
            break;
        }
        }
    }

    // Close the open units (in the order they were opened) and return the latest of the clocks.
    int64_t finish(mstate & s) {
        for (uint32_t k = 0; k < s.n_open; k++) close(s, s.open[k]);
        s.n_open = 0;
        return std::max(std::max(s.t_issue, s.t_glue), std::max(s.t_qact, s.t_mvm));
    }

    // The model's time of the units seq[0 .. len) from the state s0, with `look` the unit that follows
    // seq[len - 1] in the order (NONE at the end of the graph). Writes the scratch layer only.
    int64_t score(const mstate & s0, const uint32_t * seq, size_t len, uint32_t look) {
        begin_scratch();
        mstate s = s0;
        for (size_t k = 0; k < len; k++) step(s, seq[k], k + 1 < len ? seq[k + 1] : look);
        const int64_t t = finish(s);
        end_scratch();
        return t;
    }

private:
    void close(mstate & s, uint32_t v) {
        const tsr_pivot_unit & x = U[v];
        s.t_issue += x.issue_close;
        s.t_glue = max3(s.t_glue, get_mend(v), s.t_issue) + x.back;
        set_done(v, s.t_glue);
    }
    int64_t get_done(uint32_t v) const { return scratch_ && dst_[v] == gen_ ? cdone_[v] : done_[v]; }
    int64_t get_mend(uint32_t v) const { return scratch_ && mst_[v] == gen_ ? cmend_[v] : mend_[v]; }
    void set_done(uint32_t v, int64_t t) { if (scratch_) { cdone_[v] = t; dst_[v] = gen_; } else done_[v] = t; }
    void set_mend(uint32_t v, int64_t t) { if (scratch_) { cmend_[v] = t; mst_[v] = gen_; } else mend_[v] = t; }

    const std::vector<tsr_pivot_unit> & U;
    std::vector<int64_t>  done_, mend_, cdone_, cmend_;
    std::vector<uint32_t> dst_, mst_;
    uint32_t gen_ = 0;
    bool scratch_ = false;
};

}  // namespace

bool tsr_pivot_valid(const std::vector<tsr_pivot_unit> & units, const std::vector<uint32_t> & order, std::string * why) {
    auto fail = [&](const std::string & m) { if (why) *why = m; return false; };
    const size_t n = units.size();
    for (size_t u = 0; u < n; u++)
        for (uint32_t p : units[u].preds)
            if (p >= n) return fail("unit " + std::to_string(u) + " has a pred out of range: " + std::to_string(p));
    if (order.size() != n) return fail("order has " + std::to_string(order.size()) + " entries for " + std::to_string(n) + " units");
    std::vector<uint32_t> pos(n, NONE);
    for (size_t k = 0; k < n; k++) {
        if (order[k] >= n) return fail("order entry " + std::to_string(k) + " is out of range: " + std::to_string(order[k]));
        if (pos[order[k]] != NONE) return fail("unit " + std::to_string(order[k]) + " appears twice");
        pos[order[k]] = (uint32_t) k;
    }
    for (size_t u = 0; u < n; u++)
        for (uint32_t p : units[u].preds)
            if (pos[p] >= pos[u])
                return fail("unit " + std::to_string(u) + " (position " + std::to_string(pos[u]) + ") is not after its pred " +
                            std::to_string(p) + " (position " + std::to_string(pos[p]) + ")");
    return true;
}

int64_t tsr_pivot_eval(const std::vector<tsr_pivot_unit> & units, const std::vector<uint32_t> & order, uint32_t after) {
    machine m(units);
    mstate s;
    for (size_t k = 0; k < order.size(); k++) m.step(s, order[k], k + 1 < order.size() ? order[k + 1] : after);
    return m.finish(s);
}

tsr_pivot_result tsr_pivot_run(const std::vector<tsr_pivot_unit> & units, const tsr_pivot_params & params,
                               const tsr_pivot_score_hook * hook) {
    tsr_pivot_params P = params;
    P.threshold = std::max<int64_t>(1, P.threshold);
    tsr_pivot_result r;
    const uint32_t n = (uint32_t) units.size();
    r.order.resize(n);
    for (uint32_t k = 0; k < n; k++) r.order[k] = k;
    // the original order must itself be valid: every pred is a smaller number
    for (uint32_t u = 0; u < n; u++)
        for (uint32_t p : units[u].preds)
            if (p >= u) {
                r.error = "unit " + std::to_string(u) + " has pred " + std::to_string(p) + ", not before it";
                return r;
            }
    r.ok = true;
    if (n == 0) return r;

    machine m(units);
    std::vector<uint32_t> order = r.order;
    std::vector<mstate>   snap(n + 1);      // snap[k]: the state before position k of `order`
    uint32_t valid_to = 0;                  // snap[0 .. valid_to] and the base layer up to it belong to `order`

    // bring the saved states up to position `to`
    auto extend = [&](uint32_t to) {
        while (valid_to < to) {
            mstate s = snap[valid_to];
            m.step(s, order[valid_to], valid_to + 1 < n ? order[valid_to + 1] : NONE);
            snap[++valid_to] = s;
        }
    };
    extend(n);
    // the model's time of the original order (finish on a copy of the state after the last unit)
    int64_t cur_total = m.score(snap[n], nullptr, 0, NONE);
    r.cycles_before = cur_total;

    std::vector<uint32_t> need_st(n, 0), mv_st(n, 0);
    uint32_t gen = 0;
    std::vector<uint32_t> A, B, seg;
    // every incremental score goes through here, so that a test can compare it with a full evaluation
    auto scored = [&](uint32_t start, const uint32_t * q, size_t len, uint32_t look) {
        const int64_t t = m.score(snap[start], q, len, look);
        if (hook != nullptr) (*hook)(order, start, q, len, look, t);
        return t;
    };

    for (;;) {
        r.passes++;
        bool any = false;
        for (uint32_t i = 1; i < n; i++) {
            const uint32_t u = order[i];
            if (!has_mvm(units[u])) continue;
            const uint32_t lo = i > P.window ? i - P.window : 0;
            extend(i);
            const uint32_t e = (uint32_t) std::min<uint64_t>(n, (uint64_t) i + 1 + P.horizon);
            const uint32_t look = e < n ? order[e] : NONE;
            const int64_t score_id = scored(i, order.data() + i, e - i, look);

            ++gen;
            mv_st[u] = gen;
            for (uint32_t q : units[u].preds) need_st[q] = gen;
            uint32_t count = 1;
            int64_t  best = INT64_MAX;
            uint32_t best_p = NONE;
            for (int64_t pp = (int64_t) i - 1; pp >= (int64_t) lo; --pp) {
                const uint32_t p = (uint32_t) pp, v = order[p];
                if (need_st[v] == gen) {       // an ancestor: it joins the moved set (same order as the position after it)
                    mv_st[v] = gen;
                    for (uint32_t q : units[v].preds) need_st[q] = gen;
                    if (++count > P.max_moved) break;
                    continue;
                }
                // the moved units (positions p + 1 .. i), then the others, then the rest up to the horizon
                A.clear(); B.clear();
                for (uint32_t j = p; j <= i; j++) (mv_st[order[j]] == gen ? A : B).push_back(order[j]);
                seg.assign(order.begin() + (p ? p - 1 : 0), order.begin() + p);
                seg.insert(seg.end(), A.begin(), A.end());
                seg.insert(seg.end(), B.begin(), B.end());
                seg.insert(seg.end(), order.begin() + i + 1, order.begin() + e);
                const int64_t sc = scored(p ? p - 1 : 0, seg.data(), seg.size(), look);
                if (sc < best) { best = sc; best_p = p; }   // descending p: a tie keeps the least movement
            }
            if (best_p == NONE || score_id - best < P.threshold) continue;

            // confirm on the rest of the graph, from the state saved at best_p
            A.clear(); B.clear();
            for (uint32_t j = best_p; j <= i; j++) (mv_st[order[j]] == gen ? A : B).push_back(order[j]);
            const uint32_t start = best_p ? best_p - 1 : 0;
            seg.assign(order.begin() + start, order.begin() + best_p);
            seg.insert(seg.end(), A.begin(), A.end());
            seg.insert(seg.end(), B.begin(), B.end());
            const uint32_t moved = (uint32_t) A.size();
            seg.insert(seg.end(), order.begin() + i + 1, order.end());
            const int64_t total = scored(start, seg.data(), seg.size(), NONE);
            if (cur_total - total < P.threshold) continue;

            std::copy(seg.begin() + (best_p - start), seg.begin() + (best_p - start) + (i - best_p + 1), order.begin() + best_p);
            tsr_pivot_move mv;
            mv.unit = u; mv.from = i; mv.to = best_p; mv.moved = moved; mv.gain = cur_total - total;
            r.moves.push_back(mv);
            cur_total = total;
            valid_to  = std::min(valid_to, start);   // the state at best_p holds the old decision on the unit before it
            any = true;
        }
        if (!any) break;
    }
    r.order = order;
    r.cycles_after = cur_total;
    return r;
}

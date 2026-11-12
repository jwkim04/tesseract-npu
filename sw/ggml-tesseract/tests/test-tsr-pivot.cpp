// test-tsr-pivot.cpp — unit tests of Tesseract Pivot's core (tsr-pivot: the machine model, the validator and the
// hoist search), without ggml and without a device (ctest ggml_tesseract_pivot).
//
// Model: tsr_pivot_eval against times worked out by hand from the rules (each case states its arithmetic):
// the clocks of every class, a deferred unit that stays open and one whose next unit reads it, the closing order,
// the limit of eight open units, the issue costs.
// Validator: a permutation, every unit after its preds.
// Search on hand-built graphs, with the order each must end in (derived from the rules, not from the code):
//   - a BLOCK unit with an independent chain [GLUE, MVM, MVM] after it: the chain goes in front of the BLOCK
//     unit, behind the short MVM unit that feeds it;
//   - sibling MVM units separated by glue chains end up back to back;
//   - a chain whose long half fits a later, longer window goes to the later one, not before the MVM_ID
//     unit that needs the MVM unit first;
//   - nothing to gain: unchanged; order constraints that forbid the best move: only the free part moves;
//   - the threshold (inclusive; below 1 it is taken as 1), the window and the move limit.
// Random DAGs (several parameter sets): the result is a valid order, never slower in the model, the model's
// time of the result is what the search reports, replaying the recorded moves with an independent hoist
// reproduces every gain, every incremental score the search takes equals a full evaluation of the same units, a second run on
// the result moves nothing, and the same input gives the same result.
// A graph of 3000 units: valid, not slower, and the time of the run is printed.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "tsr-pivot.h"

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                            \
        }                                                                        \
    } while (0)

namespace {

using units_t = std::vector<tsr_pivot_unit>;
using order_t = std::vector<uint32_t>;
using preds_t = std::vector<uint32_t>;

// ---------------------------------------------------------------------------------------------
// Builders. issue = false: no control-core time (the arithmetic of the model tests); true: the
// measured record costs (the search tests).
// ---------------------------------------------------------------------------------------------
tsr_pivot_unit glue(int64_t body, preds_t preds, bool issue = true) {
    tsr_pivot_unit u;
    u.cls = TSR_PIVOT_GLUE; u.body = body; u.preds = std::move(preds);
    u.issue = issue ? TSR_PIVOT_ISSUE_SHORT : 0;
    return u;
}
tsr_pivot_unit mvm(int64_t ld, int64_t qact, int64_t mvm_t, int64_t back, preds_t preds, bool defer = true, bool issue = true) {
    tsr_pivot_unit u;
    u.cls = TSR_PIVOT_MVM; u.ld = ld; u.qact = qact; u.mvm = mvm_t; u.back = back; u.defer = defer; u.preds = std::move(preds);
    if (issue) { u.issue = TSR_PIVOT_ISSUE_MUL_MAT; u.issue_open = TSR_PIVOT_ISSUE_BEGIN; u.issue_close = TSR_PIVOT_ISSUE_END; }
    return u;
}
// a weight MUL_MAT of the search tests: LD 200, QACT 100, ST 300
tsr_pivot_unit mm(int64_t mvm_t, preds_t preds) { return mvm(200, 100, mvm_t, 300, std::move(preds)); }
tsr_pivot_unit block(int64_t body, preds_t preds) {
    tsr_pivot_unit u;
    u.cls = TSR_PIVOT_BLOCK; u.body = body; u.preds = std::move(preds);
    return u;
}
tsr_pivot_unit mvm_id(int64_t ids, int64_t mvm_t, int64_t tail, int64_t back, preds_t preds) {
    tsr_pivot_unit u;
    u.cls = TSR_PIVOT_MVM_ID; u.ids = ids; u.mvm = mvm_t; u.tail = tail; u.back = back; u.preds = std::move(preds);
    return u;
}
tsr_pivot_unit attn(int64_t front, int64_t mvm_t, int64_t back, preds_t preds) {
    tsr_pivot_unit u;
    u.cls = TSR_PIVOT_ATTN; u.front = front; u.mvm = mvm_t; u.back = back; u.preds = std::move(preds);
    return u;
}

order_t identity(size_t n) {
    order_t o(n);
    for (size_t k = 0; k < n; k++) o[k] = (uint32_t) k;
    return o;
}

std::string str(const order_t & o) {
    std::string s = "[";
    for (size_t k = 0; k < o.size(); k++) s += (k ? "," : "") + std::to_string(o[k]);
    return s + "]";
}

// The graph with its units renumbered to the positions of `order` (the input of a second run).
units_t relabel(const units_t & units, const order_t & order) {
    std::vector<uint32_t> pos(units.size());
    for (size_t k = 0; k < order.size(); k++) pos[order[k]] = (uint32_t) k;
    units_t out(units.size());
    for (size_t k = 0; k < order.size(); k++) {
        out[k] = units[order[k]];
        for (uint32_t & p : out[k].preds) p = pos[p];
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// The model, by hand
// ---------------------------------------------------------------------------------------------
void test_eval() {
    // GLUE with an issue cost: the unit cannot start before the control core has issued it.
    // t_issue = 700; s = 700; ends at 700 + 1000 = 1700.
    {
        units_t u = { glue(1000, {}) };
        CHECK(tsr_pivot_eval(u, {0}) == 1700);
    }
    // MVM, not deferred (issue 830, ld 100, qact 50, mvm 1000, back 200): t_issue 830; LD to 930; QACT to 980;
    // MVM 980 .. 1980; ST at max(930, 1980) .. 2180.
    {
        units_t u = { mvm(100, 50, 1000, 200, {}, false) };
        CHECK(tsr_pivot_eval(u, {0}) == 2180);
    }
    // MVM_ID (ids 400, mvm 5000, tail 800, back 300): join ends at 400 (t_glue, t_issue); MVM 400 .. 5400;
    // t_issue = 5400 - 800 = 4600; done = t_glue = 5700. A GLUE unit after it (body 100) starts at
    // max(5700, 4600) and ends at 5800.
    {
        units_t u = { mvm_id(400, 5000, 800, 300, {}), glue(100, {}, false) };
        CHECK(tsr_pivot_eval(u, {0, 1}) == 5800);
        units_t v = { mvm_id(400, 5000, 800, 300, {}) };
        CHECK(tsr_pivot_eval(v, {0}) == 5700);
    }
    // ATTN (front 500, mvm 6000, back 300): MVM 500 .. 6500; done = t_glue = 6800.
    {
        units_t u = { attn(500, 6000, 300, {}) };
        CHECK(tsr_pivot_eval(u, {0}) == 6800);
    }
    // BLOCK (body 3000) behind a GLUE (body 500): 500 .. 3500; t_issue and t_glue take 3500.
    {
        units_t u = { glue(500, {}, false), block(3000, {0}) };
        CHECK(tsr_pivot_eval(u, {0, 1}) == 3500);
    }

    // Deferral. A: MVM, deferred (ld 100, qact 50, mvm 1000, back 200); B: GLUE 500, independent;
    // C: GLUE 100 that reads A. No issue costs.
    //   A, B, C: A stays open (B does not read it): LD to 100, QACT to 150, MVM 100 .. 1150. B: 100 .. 600.
    //   C reads A: its ST starts at max(600, 1150) = 1150, ends 1350 (done A); C: 1350 .. 1450.      total 1450.
    //   A, C, B: C is next after A and reads it, so A is not deferred: ST 1150 .. 1350; C 1350 .. 1450;
    //   B 1450 .. 1950.                                                                              total 1950.
    {
        units_t u = { mvm(100, 50, 1000, 200, {}, true, false), glue(500, {}, false), glue(100, {0}, false) };
        CHECK(tsr_pivot_eval(u, {0, 1, 2}) == 1450);
        units_t w = u;
        CHECK(tsr_pivot_eval(w, {0, 2, 1}) == 1950);
        // the same A without defer: the ST follows at once: 100 .. 1150 MVM, ST 1150 .. 1350; B 1350 .. 1850; C 1850 .. 1950
        w[0].defer = false;
        CHECK(tsr_pivot_eval(w, {0, 1, 2}) == 1950);
    }
    // The control core: a deferred unit that stays open costs issue_open, the closing record issue_close; one that is
    // closed by its next unit costs issue. A (defer; issue 830 / 820 / 460, ld 100, qact 0, mvm 100, back 100), B GLUE
    // (body 100, issue 700), no preds.
    //   A, B: A open: t_issue 820, t_glue 920, MVM 920 .. 1020 (QACT 0: 920). B: t_issue 1520, starts 1520, ends 1620.
    //   Close at the end: t_issue 1980; ST at max(1620, 1020, 1980) = 1980 .. 2080. total 2080.
    {
        units_t u = { mvm(100, 0, 100, 100, {}), glue(100, {}) };
        CHECK(tsr_pivot_eval(u, {0, 1}) == 2080);
        // B, A: B: t_issue 700, 700 .. 800. A is last: open (no next): t_issue 1520; s = max(800, 1520) = 1520;
        // LD to 1620; MVM 1620 .. 1720. Close: t_issue 1980; ST at 1980 .. 2080.
        CHECK(tsr_pivot_eval(u, {1, 0}) == 2080);
    }
    // Only the open units that the next unit reads are closed before it.
    // A, B: MVM (ld 10, qact 0, mvm 100, back 50), D: GLUE 5 that reads B only; no issue costs.
    //   A: LD 10; MVM 10 .. 110, open (B does not read it). B: LD 20; MVM 110 .. 210; D, the next unit, reads B, so B
    //   is not deferred: its ST at max(20, 210) = 210 .. 260. D: 260 .. 265 (it does not close A). At the end A
    //   closes: ST at max(265, 110) = 265 .. 315.
    {
        units_t u = { mvm(10, 0, 100, 50, {}, true, false), mvm(10, 0, 100, 50, {}, true, false), glue(5, {1}, false) };
        CHECK(tsr_pivot_eval(u, {0, 1, 2}) == 315);
    }
    // Eight open units. Nine independent MVM units (ld 10, qact 0, mvm 100, back 500), no issue costs, in order.
    // Unit k: LD ends at 10 (k + 1); MVM ends at 110 + 100 k. Units 0 .. 7 stay open (t_glue 80). The ninth (unit 8)
    // is the 9th open: the oldest closes first: ST at max(80, 110) = 110 .. 610; unit 8: LD 610 .. 620, MVM 710 .. 910
    // (the MVM unit is busy to 810: 810 .. 910). End: units 1 .. 8 close in order: 620 + 500 k for k = 1 .. 8 = 4620.
    // (Without the limit the nine would close from 110: 110 + 9 * 500 = 4610.)
    {
        units_t u;
        for (int k = 0; k < 9; k++) u.push_back(mvm(10, 0, 100, 500, {}, true, false));
        CHECK(tsr_pivot_eval(u, identity(9)) == 4620);
    }
}

// ---------------------------------------------------------------------------------------------
// The validator
// ---------------------------------------------------------------------------------------------
void test_valid() {
    units_t u = { glue(10, {}), glue(10, {0}), glue(10, {0, 1}) };
    std::string why;
    CHECK(tsr_pivot_valid(u, {0, 1, 2}, &why));
    CHECK(!tsr_pivot_valid(u, {0, 2, 1}, &why) && why.find("not after its pred") != std::string::npos);
    CHECK(!tsr_pivot_valid(u, {1, 0, 2}, &why));
    CHECK(!tsr_pivot_valid(u, {0, 1, 1}, &why) && why.find("twice") != std::string::npos);
    CHECK(!tsr_pivot_valid(u, {0, 1}, &why) && why.find("entries") != std::string::npos);
    CHECK(!tsr_pivot_valid(u, {0, 1, 3}, &why) && why.find("out of range") != std::string::npos);
    units_t bad = u; bad[1].preds = {7};
    CHECK(!tsr_pivot_valid(bad, {0, 1, 2}, &why) && why.find("pred out of range") != std::string::npos);
    // a graph whose original order is not valid is refused by the search
    units_t fwd = { glue(10, {1}), glue(10, {}) };
    tsr_pivot_result r = tsr_pivot_run(fwd);
    CHECK(!r.ok && !r.error.empty() && r.order == order_t({0, 1}));
    // an empty graph
    r = tsr_pivot_run(units_t());
    CHECK(r.ok && r.order.empty() && r.cycles_before == 0 && r.cycles_after == 0);
}

// ---------------------------------------------------------------------------------------------
// The search on hand-built graphs
// ---------------------------------------------------------------------------------------------
void expect_order(const char * what, const units_t & u, const order_t & want, const tsr_pivot_params & P = tsr_pivot_params()) {
    tsr_pivot_result r = tsr_pivot_run(u, P);
    const bool ok = r.ok && r.order == want;
    if (!ok) fprintf(stderr, "%s: got %s, want %s\n", what, str(r.order).c_str(), str(want).c_str());
    CHECK(ok);
    CHECK(tsr_pivot_valid(u, r.order));
    CHECK(r.cycles_after == tsr_pivot_eval(u, r.order));
    CHECK(r.cycles_before == tsr_pivot_eval(u, identity(u.size())));
    if (want == identity(u.size())) CHECK(r.moves.empty() && r.cycles_after == r.cycles_before);
    else                            CHECK(!r.moves.empty() && r.cycles_after < r.cycles_before);
}

// 0 a: the glue before the router; 1 s: a short MVM that feeds the BLOCK unit (the router's input); 2 R: the
// BLOCK unit (18000 cycles); 3 e: the experts' MVM_ID (9000 on the MVM unit); then an independent chain: 4 g: GLUE, 5 m1 and
// 6 m2: two MVM units that read g (6000 each); 7 c: the combine, reads e, m1 and m2.
// The chain goes in front of R, behind s: its MVM work then runs under R's 18000 cycles. Before s it would hold the
// MVM unit ahead of s and R would wait for s; after R it stays where the MVM unit is busy with e and the glue
// pipeline with c. Of the equal places for m1 and m2 the one that moves least is taken, so the order of m1 and m2 holds.
units_t graph_block(bool constraint_on_block = false, bool m2_after_e = false) {
    units_t u;
    u.push_back(glue(2000, {}));                                    // 0 a
    u.push_back(mm(1500, {0}));                                     // 1 s
    u.push_back(block(18000, {1}));                                 // 2 R
    u.push_back(mvm_id(400, 9000, 700, 300, {2}));                  // 3 e
    u.push_back(glue(1500, constraint_on_block ? preds_t{0, 2} : preds_t{0}));   // 4 g
    u.push_back(mm(6000, {4}));                                     // 5 m1
    u.push_back(mm(6000, m2_after_e ? preds_t{4, 3} : preds_t{4})); // 6 m2
    u.push_back(glue(1500, {3, 5, 6}));                             // 7 c
    return u;
}

void test_block() {
    expect_order("block chain", graph_block(), {0, 1, 4, 5, 6, 2, 3, 7});
    // the chain's glue g has R as a pred: nothing may go in front of R except by moving R too: unchanged
    expect_order("chain after R by constraint", graph_block(true), identity(8));
    // m2 must come after e (an order constraint): only g and m1 move in front of R
    expect_order("m2 after e", graph_block(false, true), {0, 1, 4, 5, 2, 3, 6, 7});
}

// Q, K and V products of one layer with the glue (rope) between them: the products are siblings
// (all read the norm n), each followed by the glue that reads it.
void test_siblings() {
    units_t u;
    u.push_back(glue(1500, {}));          // 0 n
    u.push_back(mm(2000, {0}));           // 1 Q
    u.push_back(glue(3000, {1}));         // 2 rq
    u.push_back(mm(2000, {0}));           // 3 K
    u.push_back(glue(3000, {3}));         // 4 rk
    u.push_back(mm(2000, {0}));           // 5 V
    u.push_back(glue(1500, {2, 4, 5}));   // 6 attention input
    // Each product's MVM work (2000, with LD and QACT 2300 on the way) is shorter than the glue chain behind it (3000). Apart,
    // every chain waits for the ST of its product, which waits for its MVM work. Queued together, the MVM unit runs the
    // three products while the glue pipeline runs LD of the next one and then the chains: only the first wait is left.
    // K goes in front of rq, then V in front of rq as well (in front of rk alone it would still wait for K's
    // chain to queue its LD).
    expect_order("siblings", u, {0, 1, 3, 5, 2, 4, 6});
}

// A chain [m1 -> h -> m2] whose long half m2 (20000 on the MVM unit) fits the long BLOCK unit B2 (24000), not the
// short B1 (5000): the MVM_ID unit e between them needs the MVM unit first, so m2 may not go in front of it, and
// the glue pipeline is busy with e's ST until e ends. m2 (with h) goes in front of B2, behind e. The short half m1
// (40 cycles of LD, QACT, MVM and ST) does not belong behind e: its wait would sit on the glue pipeline in front of B2;
// it goes in front of B1, where the BLOCK unit hides it. The alternatives are worse in the model (stated below).
void test_later_window() {
    units_t u;
    u.push_back(glue(1500, {}));                          // 0 a
    u.push_back(block(5000, {0}));                        // 1 B1
    u.push_back(mvm_id(400, 4000, 700, 300, {1}));        // 2 e
    u.push_back(block(24000, {2}));                       // 3 B2
    u.push_back(mvm(10, 10, 10, 10, {0}));                // 4 m1
    u.push_back(glue(500, {4}));                          // 5 h
    u.push_back(mm(20000, {5}));                          // 6 m2
    u.push_back(glue(1000, {3, 6}));                      // 7 c
    const order_t want = {0, 4, 1, 2, 5, 6, 3, 7};
    expect_order("later window", u, want);
    const int64_t t = tsr_pivot_eval(u, want);
    CHECK(t < tsr_pivot_eval(u, {0, 1, 2, 4, 5, 6, 3, 7}));   // the whole chain in front of B2
    CHECK(t < tsr_pivot_eval(u, {0, 4, 5, 6, 1, 2, 3, 7}));   // the whole chain in front of B1: e waits for m2's MVM work
    CHECK(t < tsr_pivot_eval(u, {0, 1, 4, 5, 6, 2, 3, 7}));   // the whole chain in front of e: the same
    CHECK(t < tsr_pivot_eval(u, {0, 4, 1, 2, 3, 5, 6, 7}));   // m2 stays behind B2
    CHECK(t < tsr_pivot_eval(u, identity(8)));
}

void test_nothing() {
    // a dependent chain: every unit reads the one before
    units_t u;
    u.push_back(glue(1500, {}));
    u.push_back(mm(5000, {0}));
    u.push_back(glue(1500, {1}));
    u.push_back(mm(5000, {2}));
    u.push_back(glue(1500, {3}));
    expect_order("dependent chain", u, identity(5));
    // the siblings already together, then the glue
    units_t v;
    v.push_back(glue(1500, {}));
    v.push_back(mm(5000, {0}));
    v.push_back(mm(5000, {0}));
    v.push_back(mm(5000, {0}));
    v.push_back(glue(1500, {1}));
    v.push_back(glue(1500, {2, 3, 4}));
    expect_order("siblings together", v, identity(6));
    // only glue and BLOCK units: nothing has MVM work
    units_t w = { glue(100, {}), block(5000, {0}), glue(300, {}), block(4000, {2}) };
    expect_order("no MVM work", w, identity(4));
}

// The threshold: BLOCK 1000, an independent MVM unit x (200 on the MVM unit, no LD, QACT, ST), then C (GLUE 100) that
// reads both. No issue costs. [R, x, C]: R 0 .. 1000; x LD 0 (t_glue 1000), MVM 1000 .. 1200; C reads x: ST at 1200,
// C 1200 .. 1300. [x, R, C]: x MVM 0 .. 200 (open); R 0 .. 1000; C closes x at 1000, runs 1000 .. 1100.
// The gain is 200: taken with a threshold of 200 (a gain of at least the threshold), not with 201.
void test_threshold() {
    units_t u = { block(1000, {}), mvm(0, 0, 200, 0, {}, true, false), glue(100, {0, 1}, false) };
    CHECK(tsr_pivot_eval(u, {0, 1, 2}) == 1300);
    CHECK(tsr_pivot_eval(u, {1, 0, 2}) == 1100);
    tsr_pivot_params P;
    expect_order("gain 200, threshold 256", u, identity(3), P);
    P.threshold = 201;
    expect_order("gain 200, threshold 201", u, identity(3), P);
    P.threshold = 200;
    expect_order("gain 200, threshold 200", u, {1, 0, 2}, P);
    tsr_pivot_result r = tsr_pivot_run(u, P);
    CHECK(r.moves.size() == 1 && r.moves[0].unit == 1 && r.moves[0].from == 1 && r.moves[0].to == 0 &&
          r.moves[0].moved == 1 && r.moves[0].gain == 200 && r.cycles_before == 1300 && r.cycles_after == 1100);
    // a huge threshold turns the search off
    P.threshold = int64_t(1) << 40;
    expect_order("huge threshold", graph_block(), identity(8), P);
    // a threshold of 0 or less is taken as 1: every move still gains, so the search ends (zero-gain moves could cycle forever)
    for (int64_t th : { int64_t(0), int64_t(-5) }) {
        P.threshold = th;
        tsr_pivot_result z = tsr_pivot_run(graph_block(), P);
        CHECK(z.ok && z.passes < 100 && tsr_pivot_valid(graph_block(), z.order));
        for (const tsr_pivot_move & m : z.moves) CHECK(m.gain >= 1);
    }
}

// The move limit and the window: the chain of graph_block() needs a move of 3 units from 2 positions back.
void test_limits() {
    tsr_pivot_params P;
    P.max_moved = 1;                          // g, m1: 2 units: not tried
    expect_order("max_moved 1", graph_block(), identity(8), P);
    P.max_moved = 3;
    expect_order("max_moved 3", graph_block(), {0, 1, 4, 5, 6, 2, 3, 7}, P);
    P = tsr_pivot_params();
    P.window = 1;                             // only the position right before the unit
    expect_order("window 1", graph_block(), identity(8), P);
    // a window of 3: m1 (position 5) can reach position 2 (R) with g: 3 positions back
    P.window = 3;
    expect_order("window 3", graph_block(), {0, 1, 4, 5, 6, 2, 3, 7}, P);
}

// ---------------------------------------------------------------------------------------------
// Random DAGs
// ---------------------------------------------------------------------------------------------
units_t random_graph(std::mt19937 & rng, uint32_t n) {
    auto rnd = [&](int64_t lo, int64_t hi) { return lo + (int64_t) (rng() % (uint64_t) (hi - lo + 1)); };
    units_t u;
    for (uint32_t k = 0; k < n; k++) {
        preds_t preds;
        const uint32_t np = (uint32_t) rnd(0, 3);
        for (uint32_t j = 0; j < np && k > 0; j++) {
            const uint32_t back = (uint32_t) rnd(1, std::min<int64_t>(k, rng() % 4 == 0 ? k : 24));
            const uint32_t p = k - back;
            if (std::find(preds.begin(), preds.end(), p) == preds.end()) preds.push_back(p);
        }
        std::sort(preds.begin(), preds.end());
        const int c = (int) (rng() % 100);
        if (c < 40)       u.push_back(glue(rnd(300, 6000), preds));
        else if (c < 80)  u.push_back(mvm(rnd(50, 400), rnd(50, 500), rnd(200, 20000), rnd(100, 600), preds, rng() % 8 != 0));
        else if (c < 88)  u.push_back(mvm_id(400, rnd(2000, 30000), rnd(100, 1500), 300, preds));
        else if (c < 94)  u.push_back(block(rnd(2000, 25000), preds));
        else              u.push_back(attn(500, rnd(1000, 9000), 300, preds));
    }
    return u;
}

// An independent hoist: the unit at position `from` and its ancestors at or after `to`, put at `to`.
order_t naive_hoist(const units_t & u, const order_t & order, uint32_t from, uint32_t to) {
    std::vector<char> in(u.size(), 0);
    in[order[from]] = 1;
    for (int64_t k = from; k >= (int64_t) to; --k) {
        if (!in[order[k]]) continue;
        for (uint32_t p : u[order[k]].preds) in[p] = 1;
    }
    order_t out(order.begin(), order.begin() + to), rest;
    for (uint32_t k = to; k <= from; k++) (in[order[k]] ? out : rest).push_back(order[k]);
    out.insert(out.end(), rest.begin(), rest.end());
    out.insert(out.end(), order.begin() + from + 1, order.end());
    return out;
}

size_t g_hook_calls = 0;

// Every incremental score of the search (the unchanged order's, each candidate's, the confirmation's) against a full evaluation of
// the same units from the start of the graph: the sequence is a permutation of the span it replaces, the lookahead is the unit
// behind that span, and the score is the model's time of the prefix and the sequence.
tsr_pivot_score_hook score_check(const units_t & u, bool * ok) {
    return [&u, ok](const order_t & order, uint32_t start, const uint32_t * seg, size_t len, uint32_t look, int64_t score) {
        g_hook_calls++;
        const size_t n = order.size();
        bool good = start + len <= n && len > 0 && look == (start + len < n ? order[start + len] : TSR_PIVOT_NONE);
        if (good) {
            order_t a(seg, seg + len), b(len);
            std::copy(order.begin() + start, order.begin() + start + len, b.begin());
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            good = a == b;
        }
        if (good) {
            order_t full(order.begin(), order.begin() + start);
            full.insert(full.end(), seg, seg + len);
            good = tsr_pivot_eval(u, full, look) == score;
        }
        if (!good) *ok = false;
    };
}

void check_random(const units_t & u, const tsr_pivot_params & P, const char * tag) {
    bool scores_ok = true;
    const tsr_pivot_score_hook hook = score_check(u, &scores_ok);
    tsr_pivot_result r = tsr_pivot_run(u, P, &hook);
    CHECK(r.ok);
    if (!scores_ok) fprintf(stderr, "%s: an incremental score differs from the full evaluation\n", tag);
    CHECK(scores_ok);
    std::string why;
    const bool valid = tsr_pivot_valid(u, r.order, &why);
    if (!valid) fprintf(stderr, "%s: invalid result: %s\n", tag, why.c_str());
    CHECK(valid);
    CHECK(r.cycles_before == tsr_pivot_eval(u, identity(u.size())));
    CHECK(r.cycles_after == tsr_pivot_eval(u, r.order));
    CHECK(r.cycles_after <= r.cycles_before);
    // the same input, the same result
    tsr_pivot_result r2 = tsr_pivot_run(u, P);
    CHECK(r2.order == r.order && r2.cycles_after == r.cycles_after && r2.moves.size() == r.moves.size() && r2.passes == r.passes);
    // replay the recorded moves with an independent hoist: every gain is the fall of the model's time
    order_t cur = identity(u.size());
    int64_t t = r.cycles_before;
    bool replay_ok = true;
    for (const tsr_pivot_move & m : r.moves) {
        if (m.to >= m.from || m.from >= cur.size() || cur[m.from] != m.unit || m.gain < P.threshold || m.moved > P.max_moved ||
            m.from - m.to > P.window) { replay_ok = false; break; }
        cur = naive_hoist(u, cur, m.from, m.to);
        const int64_t t2 = tsr_pivot_eval(u, cur);
        if (!tsr_pivot_valid(u, cur) || t - t2 != m.gain) { replay_ok = false; break; }
        t = t2;
    }
    if (!replay_ok) fprintf(stderr, "%s: replay of the moves failed\n", tag);
    CHECK(replay_ok);
    CHECK(cur == r.order && t == r.cycles_after);
    // the fixed point: the result, taken as a graph, is left as it is
    units_t v = relabel(u, r.order);
    tsr_pivot_result q = tsr_pivot_run(v, P);
    const bool fixed = q.ok && q.moves.empty() && q.order == identity(u.size()) && q.cycles_after == r.cycles_after;
    if (!fixed) fprintf(stderr, "%s: second run moved %zu units\n", tag, q.moves.size());
    CHECK(fixed);
}

void test_random() {
    std::mt19937 rng(20241112);
    tsr_pivot_params defaults;
    tsr_pivot_params small;
    small.window = 8; small.max_moved = 3; small.threshold = 1; small.horizon = 4;
    tsr_pivot_params zero = small;
    zero.threshold = 0;                       // taken as 1
    tsr_pivot_params wide;
    wide.window = 200; wide.max_moved = 16; wide.threshold = 64; wide.horizon = 100;
    size_t total_moves = 0;
    int improved = 0;
    for (int seed = 0; seed < 120; seed++) {
        const uint32_t n = 2 + (uint32_t) (rng() % 250);
        units_t u = random_graph(rng, n);
        const std::string tag = "random " + std::to_string(seed);
        check_random(u, defaults, tag.c_str());
        check_random(u, small, tag.c_str());
        if (seed % 3 == 0) {
            tsr_pivot_result z = tsr_pivot_run(u, zero), o = tsr_pivot_run(u, small);
            CHECK(z.order == o.order && z.passes == o.passes);   // the same search as with threshold 1
        }
        if (seed % 4 == 0) check_random(u, wide, tag.c_str());
        tsr_pivot_result r = tsr_pivot_run(u, defaults);
        total_moves += r.moves.size();
        improved += r.cycles_after < r.cycles_before;
    }
    // the generator must give the search something to do, or the properties above say nothing
    CHECK(total_moves >= 100);
    CHECK(improved >= 40);
    CHECK(g_hook_calls >= 100000);
    printf("random DAGs: %zu moves in 120 graphs, %d improved\n", total_moves, improved);
}

// ---------------------------------------------------------------------------------------------
// A graph of 3000 units: layers of a decode step (glue, Q K V, attention, output, an MoE block with
// its router, dense FFN products and expert products), 50 units each.
// ---------------------------------------------------------------------------------------------
units_t layered_graph(uint32_t layers) {
    units_t u;
    auto add = [&](tsr_pivot_unit x) { u.push_back(std::move(x)); return (uint32_t) (u.size() - 1); };
    uint32_t prev = add(glue(1800, {}));
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t n1 = add(glue(1700, {prev}));
        uint32_t qkv[3];
        for (int k = 0; k < 3; k++) {
            const uint32_t m = add(mm(4000 + 800 * k, {n1}));
            qkv[k] = add(glue(1500, {m}));
        }
        const uint32_t kv = add(glue(900, {qkv[1], qkv[2]}));
        const uint32_t at = add(attn(500, 6000, 300, {qkv[0], kv}));
        const uint32_t o = add(mm(5000, {at}));
        const uint32_t r1 = add(glue(1600, {prev, o}));
        const uint32_t n2 = add(glue(1700, {r1}));
        const uint32_t router = add(block(6500 + 96 * 128, {n2}));
        uint32_t last = router;
        std::vector<uint32_t> outs;
        for (int k = 0; k < 3; k++) {
            const uint32_t e = add(mvm_id(400, 9000, 700, 300, {last}));
            last = add(glue(800, {e}));
            outs.push_back(last);
        }
        const uint32_t g = add(mm(7000, {n2}));
        const uint32_t up = add(mm(7000, {n2}));
        const uint32_t gu = add(glue(1500, {g, up}));
        const uint32_t dn = add(mm(7000, {gu}));
        // filler: independent glue of the layer
        std::vector<uint32_t> fill;
        for (int k = 0; k < 12; k++) fill.push_back(add(glue(600 + 40 * k, {r1})));
        preds_t join = { dn };
        join.insert(join.end(), outs.begin(), outs.end());
        join.push_back(fill.back());
        prev = add(glue(1800, join));
        for (int k = 0; k < 5; k++) prev = add(glue(300, {prev}));
    }
    return u;
}

void test_big() {
    units_t u = layered_graph(74);
    printf("graph of %zu units\n", u.size());
    CHECK(u.size() >= 2800 && u.size() <= 3300);
    const auto t0 = std::chrono::steady_clock::now();
    tsr_pivot_result r = tsr_pivot_run(u);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tsr_pivot_run on %zu units: %.1f ms, %zu moves, %u passes, model %lld -> %lld cycles\n", u.size(), ms,
           r.moves.size(), r.passes, (long long) r.cycles_before, (long long) r.cycles_after);
    CHECK(r.ok && tsr_pivot_valid(u, r.order));
    CHECK(r.cycles_after <= r.cycles_before);
    CHECK(r.cycles_after == tsr_pivot_eval(u, r.order));
    CHECK(!r.moves.empty());
    CHECK(ms < 1000.0);                           // "well under a second"
    units_t v = relabel(u, r.order);
    tsr_pivot_result q = tsr_pivot_run(v);
    CHECK(q.moves.empty());
}

}  // namespace

int main() {
    test_eval();
    test_valid();
    test_block();
    test_siblings();
    test_later_window();
    test_nothing();
    test_threshold();
    test_limits();
    test_random();
    test_big();
    if (g_fail) {
        fprintf(stderr, "test-tsr-pivot: %d checks failed\n", g_fail);
        return 1;
    }
    printf("test-tsr-pivot: ok\n");
    return 0;
}

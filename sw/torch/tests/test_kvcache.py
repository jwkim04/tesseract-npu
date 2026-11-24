"""Full mode with a static KV cache (Hugging Face ``StaticCache``): the cached attention.

A cached attention runs on the device with the K and V of earlier calls in the device's KV planes (q8_0). The host
cache stays the complete and valid cache, and the device holds a shadow of it; ``tesseract_torch.runtime`` keeps, for
each cache, how many cells of the shadow equal the quantized host rows (``n_dev``) and the ``_version`` of the two
tensors for which that holds. These tests drive every transition of that state and compare the result of the run
under test with the result of a *fresh* run on the same host cache content (a deep copy of the cache: new tensors,
so a new state that is made from the host rows alone), and with the stateless compiled forward of the whole sequence.

Oracles (strongest first): ``TSR_TORCH_VERIFY`` (every region call against the reference executor, and the cells a
call stored against the quantized rows the region returned, and the cells below it against the quantized host rows:
``runtime.q8_0_rows``, torch arithmetic written from the definition); the planes read back from the device against ``spec_model.q8_0_ref_blocks`` (numpy) of the host rows;
the cached run against the stateless run of the same sequence; a fresh run on a copy of the host cache.

Measured on the simulator (relative NMSE = sum((cached - stateless)^2) / sum(stateless^2) of the logits of every
position): 0 (bit for bit) for the tiny model below with a prefill of 5 rows and 7 decode steps. The gate is
NMSE_MAX = 1e-10, a margin of more than 1e4 over the largest value measured anywhere (3e-15: the runtime tests of
cached attention with a given mask).
"""
from __future__ import annotations

import copy
import gc
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import spec_model as sm
import tesseract_torch as tt
from tesseract_torch import backend as tb
from tesseract_torch import runtime as rt

transformers = pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, Qwen2Config, StaticCache  # noqa: E402

NMSE_MAX = 1e-10
L = 48                                    # cells of the test caches


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setenv("TSR_TORCH_MODE", "full")
    monkeypatch.delenv("TSR_TORCH_KV_CACHE", raising=False)
    old = torch._dynamo.config.cache_size_limit
    torch._dynamo.config.cache_size_limit = 64        # the tests recompile for copies of a cache
    yield
    torch._dynamo.config.cache_size_limit = old


def tiny(max_pos=64, layers=2, seed=0):
    """A random Qwen2 of the device's shapes: 4 heads over 2 KV heads of 64, tied head."""
    cfg = Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=max_pos,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval(), cfg


def compiled(m, **kw):
    return tt.compile(m.forward, mode="full", verify=True, **kw)


def ids_of(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)


def step(fn, ids, cache, **kw):
    with torch.no_grad():
        return fn(input_ids=ids, past_key_values=cache, use_cache=True, **kw).logits


def bits(a):
    return a.contiguous().view(torch.int32)


def same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def nmse(y, ref) -> float:
    return sm.nmse(y.detach().double().numpy(), ref.detach().double().numpy())


def stateless(m, ids):
    """The logits of every position of ``ids`` [1, n] from the compiled forward without a cache."""
    with torch.no_grad():
        return compiled(m)(input_ids=ids, use_cache=False).logits


def c():
    return tt.counters()


def no_fallback(calls: int):
    k = c()
    assert k["region_cpu_fallback_calls"] == 0, k
    assert k["region_calls"] == calls and k["verified_regions"] == calls and k["kv_region_calls"] == calls, k
    assert k["rt_graph_launches"] == calls, k          # one launch per region call


def assert_planes_equal_host(cache, n):
    """The cells [0, n) of every layer read back from the device equal numpy's q8_0 of the host rows."""
    dev = tt.device()
    for layer in cache.layers:
        st = rt._kv_by_base[rt.WeightCache.key(layer.keys)]
        assert st.key_v == rt.WeightCache.key(layer.values) and not st.dead()
        kb, vb = dev.kv_read(st.handle, 0, n)
        for got, host in ((kb, layer.keys), (vb, layer.values)):
            want = sm.q8_0_ref_blocks(host[0, :, :n, :].transpose(0, 1).reshape(n, -1).numpy())
            assert got.shape == want.shape and (got.numpy() == want).all(), "device cells differ from the quantized host rows"


def force_cpu(monkeypatch, calls):
    """The next ``calls`` KV preparations (region calls with a cache) are refused: those calls run the saved aten
    subgraph on the CPU, as a call that the device does not take does."""
    left = [calls]
    orig = rt._kv_check_call

    def prep(rg, kvargs, T):
        if left[0] > 0:
            left[0] -= 1
            raise tt.TesseractError(f"tesseract: {rg.name}: a cached attention does not fit the device: forced by the test")
        return orig(rg, kvargs, T)

    monkeypatch.setattr(rt, "_kv_check_call", prep)


# ------------------------------------------------------------------------------------------
# the quantizer of the cache check
# ------------------------------------------------------------------------------------------
def test_q8_0_rows_is_ggml_quantize_row_q8_0_ref():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 192, generator=g) * torch.logspace(-3, 3, 40).unsqueeze(1)
    x[0] = 0.0                                          # d = 0
    x[1, :32] = 1e-30                                   # tiny
    x[2, 5] = 3e4                                       # huge
    x[3] = torch.round(x[3]) * 0.5                      # ties after scaling
    x[4, ::2] = -x[4, ::2]
    got = rt.q8_0_rows(x)
    want = sm.q8_0_ref_blocks(x.numpy())
    assert got.dtype == torch.uint8 and tuple(got.shape) == (40, 192 // 32 * 34)
    assert (got.numpy() == want).all()


# ------------------------------------------------------------------------------------------
# prefill and decode
# ------------------------------------------------------------------------------------------
def test_prefill_and_decode_steps_equal_the_stateless_run():
    m, cfg = tiny()
    ids = ids_of(12)
    fn = compiled(m)
    cache = StaticCache(cfg, max_cache_len=L)
    steps = [(0, 5)] + [(i, i + 1) for i in range(5, 12)]
    outs = []
    for k, (a, b) in enumerate(steps):
        outs.append(step(fn, ids[:, a:b], cache))
        no_fallback(k + 1)                                # a prefill and every decode step: one region call, one launch
    k = c()
    assert k["regions"] == 2 and k["kv_caches"] == 2 and k["rt_kv_creates"] == 2     # a prefill graph and a decode graph
    assert k["kv_resync_rows"] == 0 and k["rt_kv_stores"] == 0, k     # no row is stored again in a plain run
    # verify runs the plan twice (the device and the reference executor): a miss for the prefill and the first decode step
    assert k["rt_graph_plan_misses"] == 2 and k["rt_graph_plan_hits"] == 2 * len(steps) - 2, k
    cached = torch.cat(outs, 1)
    full = stateless(m, ids)
    e = nmse(cached, full)
    print(f"relative NMSE, cached against stateless: {e:.3g}")
    assert e <= NMSE_MAX
    # the planes against numpy's quantizer of the host rows: the host cache holds what the device stored
    assert_planes_equal_host(cache, 12)
    assert float(cache.layers[0].keys[0, :, :12].abs().sum()) > 0
    assert float(cache.layers[0].keys[0, :, 12:].abs().sum()) == 0      # the cells above are untouched


def test_generate_with_a_static_cache():
    """``generate(cache_implementation="static")`` through a compiled forward: the prefill of a fresh cache is
    ``is_causal`` attention over the first cells of the cache (no mask), the decode step has a mask input. Both are
    one region of cached attentions; the tokens are eager's; a second generation of the prompt makes nothing new."""
    m, cfg = tiny()
    ids = ids_of(9)
    am = torch.ones_like(ids)
    kw = dict(attention_mask=am, max_new_tokens=6, min_new_tokens=6, do_sample=False, cache_implementation="static", pad_token_id=0)
    with torch.no_grad():
        ref = m.generate(input_ids=ids, **kw)
    m.forward = compiled(m)
    with torch.no_grad():
        out = m.generate(input_ids=ids, **kw)
    k = c()
    assert torch.equal(out, ref)
    assert k["regions"] == 2 and k["region_calls"] == 6 and k["region_cpu_fallback_calls"] == 0, k
    assert k["rt_graph_launches"] == 6 and k["kv_region_calls"] == 6 and k["kv_resync_rows"] == 0, k
    pre, dec = tt.regions()[-2:]                  # the regions of this test: its prefill graph and its decode graph
    assert [kc.causal0 for kc in pre.region.kv] == [True, True], "the prefill graph: causal over the first cells"
    assert [kc.causal0 for kc in dec.region.kv] == [False, False] and dec.region.kv[0].form == "existing"
    k0 = dict(c())
    with torch.no_grad():
        out2 = m.generate(input_ids=ids, **kw)
    k1 = c()
    assert torch.equal(out2, ref)
    assert k1["uploads"] == k0["uploads"] and k1["rt_graph_plan_misses"] == k0["rt_graph_plan_misses"]
    assert k1["region_calls"] == 12 and k1["region_cpu_fallback_calls"] == 0


def test_second_generation_on_a_new_cache_needs_no_upload_and_no_plan():
    m, cfg = tiny()
    ids = ids_of(9)
    fn = compiled(m)
    runs = []
    k0 = {}
    for rnd in range(2):
        cache = StaticCache(cfg, max_cache_len=L)
        runs.append(torch.cat([step(fn, ids[:, :6], cache)] + [step(fn, ids[:, i:i + 1], cache) for i in range(6, 9)], 1))
        if rnd == 0:
            k0 = dict(c())
            del cache
            gc.collect()
    k1 = c()
    assert same(runs[0], runs[1])
    assert k1["uploads"] == k0["uploads"] and k1["consts_uploaded"] == k0["consts_uploaded"]
    assert k1["rt_graph_plan_misses"] == k0["rt_graph_plan_misses"]
    assert k1["region_cpu_fallback_calls"] == 0


# ------------------------------------------------------------------------------------------
# the transitions of the state of a cache
# ------------------------------------------------------------------------------------------
def _prefilled(cfg, ids, n, fn):
    cache = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :n], cache)
    return cache


def test_a_changed_host_cache_is_stored_again():
    """An earlier row edited in place (K of layer 0, V of layer 1) bumps the versions: the next call stores the
    whole cache of those slots again, and gives what a fresh run on the same host cache gives."""
    m, cfg = tiny()
    ids = ids_of(12)
    fn = compiled(m)
    cache = _prefilled(cfg, ids, 6, fn)
    step(fn, ids[:, 6:7], cache)
    step(fn, ids[:, 7:8], cache)                          # p0 = 8
    untouched = copy.deepcopy(cache)
    cache.layers[0].keys[0, :, 2, :] *= 3.0
    cache.layers[1].values[0, :, 7, :] += 0.5
    fresh = copy.deepcopy(cache)
    r0 = c()["kv_resync_rows"]
    y = step(fn, ids[:, 8:9], cache)
    assert c()["kv_resync_rows"] - r0 == 16, "layer 0 and layer 1 each store their 8 rows again, nothing else"
    y_fresh = step(fn, ids[:, 8:9], fresh)
    y_old = step(fn, ids[:, 8:9], untouched)
    assert same(y, y_fresh)
    assert not same(y, y_old), "the edit must matter (else this test proves nothing)"
    assert_planes_equal_host(cache, 9)
    # nothing is stored again afterwards
    r1 = c()["kv_resync_rows"]
    snap = copy.deepcopy(cache)
    y2 = step(fn, ids[:, 9:10], cache)
    assert c()["kv_resync_rows"] == r1
    assert same(y2, step(fn, ids[:, 9:10], snap))
    assert c()["region_cpu_fallback_calls"] == 0


def test_reset_and_reuse_of_a_cache():
    """``cache.reset()`` zeroes the tensors in place (versions change) and the length: a new prompt on the same
    tensors gives what a new cache gives."""
    m, cfg = tiny()
    fn = compiled(m)
    a, b = ids_of(10, seed=2), ids_of(10, seed=3)
    cache = StaticCache(cfg, max_cache_len=L)
    ya = [step(fn, a[:, :6], cache)] + [step(fn, a[:, i:i + 1], cache) for i in range(6, 10)]
    cache.reset()
    kbefore = c()["kv_caches"]
    yb = [step(fn, b[:, :4], cache)] + [step(fn, b[:, i:i + 1], cache) for i in range(4, 10)]
    new = StaticCache(cfg, max_cache_len=L)
    yn = [step(fn, b[:, :4], new)] + [step(fn, b[:, i:i + 1], new) for i in range(4, 10)]
    for u, v in zip(yb, yn):
        assert same(u, v)
    assert c()["kv_caches"] - kbefore == 2, "the reset cache keeps its state; only the new cache makes states"
    assert nmse(torch.cat(yb, 1), stateless(m, b)) <= NMSE_MAX
    assert nmse(torch.cat(ya, 1), stateless(m, a)) <= NMSE_MAX
    assert_planes_equal_host(cache, 10)
    assert c()["region_cpu_fallback_calls"] == 0


def test_a_call_that_ran_on_the_cpu_in_the_middle(monkeypatch):
    """A region call that the device does not take runs the saved aten subgraph and writes the host rows without
    a device step: the shadow lacks that row, and the next device call stores it."""
    m, cfg = tiny()
    ids = ids_of(12)
    fn = compiled(m)
    cache = _prefilled(cfg, ids, 5, fn)
    step(fn, ids[:, 5:6], cache)                          # device, p0 = 5
    r0 = c()["kv_resync_rows"]
    force_cpu(monkeypatch, 1)
    y_cpu = step(fn, ids[:, 6:7], cache)                  # CPU, p0 = 6
    k = c()
    assert k["region_cpu_fallback_calls"] == 1 and torch.isfinite(y_cpu).all()
    assert k["kv_resync_rows"] == r0, "nothing is stored by the call that fell back"
    fresh = copy.deepcopy(cache)
    y = step(fn, ids[:, 7:8], cache)                      # device, p0 = 7
    assert c()["kv_resync_rows"] - r0 == 2, "row 6 of each of the two layers"
    assert same(y, step(fn, ids[:, 7:8], fresh))
    assert_planes_equal_host(cache, 8)
    r1 = c()["kv_resync_rows"]                            # (the fresh run stored its own rows)
    snap = copy.deepcopy(cache)
    y2 = step(fn, ids[:, 8:9], cache)
    assert c()["kv_resync_rows"] == r1
    assert same(y2, step(fn, ids[:, 8:9], snap))
    assert c()["region_cpu_fallback_calls"] == 1


def test_rewind_below_the_device_length(monkeypatch):
    """The length of the cache set back (the tensors are not touched): cells above the new start are rewritten by
    the call, so the shadow keeps only the cells below it. A call on the CPU at the new start leaves the shadow
    short of the row it wrote."""
    m, cfg = tiny()
    ids = ids_of(14)
    fn = compiled(m)
    cache = _prefilled(cfg, ids, 6, fn)
    for i in range(6, 8):
        step(fn, ids[:, i:i + 1], cache)                  # p0 = 8, n_dev = 8
    for layer in cache.layers:
        layer.cumulative_length.fill_(3)
    snap = copy.deepcopy(cache)
    # (a) on the device: the shadow's length is clamped to 3, nothing is stored again
    r0 = c()["kv_resync_rows"]
    y = step(fn, ids[:, 10:11], cache)
    assert c()["kv_resync_rows"] == r0
    assert same(y, step(fn, ids[:, 10:11], snap))
    assert_planes_equal_host(cache, 4)
    # (b) on the CPU at cell 4, then on the device at cell 5: the CPU call changed host cell 4 but not the shadow's
    force_cpu(monkeypatch, 1)
    r1 = c()["kv_resync_rows"]
    step(fn, ids[:, 11:12], cache)
    assert c()["region_cpu_fallback_calls"] == 1 and c()["kv_resync_rows"] == r1
    snap = copy.deepcopy(cache)
    y = step(fn, ids[:, 12:13], cache)
    assert c()["kv_resync_rows"] - r1 == 2, "cell 4 of both layers"
    assert same(y, step(fn, ids[:, 12:13], snap))
    assert_planes_equal_host(cache, 6)


def test_rewind_then_a_cpu_call_does_not_leave_a_stale_cell(monkeypatch):
    """The state where a wrong clamp gives a silently wrong answer: the shadow is long (8 cells), the length is
    set back to 3, a CPU call rewrites host cell 3, and the device call at cell 4 must not trust the shadow's cell 3."""
    m, cfg = tiny()
    ids = ids_of(14)
    fn = compiled(m)
    cache = _prefilled(cfg, ids, 6, fn)
    for i in range(6, 8):
        step(fn, ids[:, i:i + 1], cache)                  # n_dev = 8
    for layer in cache.layers:
        layer.cumulative_length.fill_(3)
    force_cpu(monkeypatch, 1)
    step(fn, ids[:, 9:10], cache)                         # CPU, writes cell 3 on the host
    snap = copy.deepcopy(cache)
    r0 = c()["kv_resync_rows"]
    y = step(fn, ids[:, 10:11], cache)                    # device at cell 4
    assert c()["kv_resync_rows"] - r0 == 2, "cell 3 of both layers"
    assert same(y, step(fn, ids[:, 10:11], snap))
    assert_planes_equal_host(cache, 5)


def test_two_caches_interleaved():
    m, cfg = tiny()
    fn = compiled(m)
    a, b = ids_of(10, seed=4), ids_of(10, seed=5)
    ca, cb = StaticCache(cfg, max_cache_len=L), StaticCache(cfg, max_cache_len=L)
    snaps, outs = [], []
    plan = [("a", 0, 5), ("b", 0, 7), ("a", 5, 6), ("b", 7, 8), ("a", 6, 7), ("b", 8, 9), ("a", 7, 8), ("b", 9, 10), ("a", 8, 9)]
    for which, s, e in plan:
        cache, ids = (ca, a) if which == "a" else (cb, b)
        snaps.append((copy.deepcopy(cache), ids[:, s:e]))
        outs.append(step(fn, ids[:, s:e], cache))
    k = c()
    assert k["kv_caches"] == 4 and k["kv_resync_rows"] == 0, "each cache has its own state; no row is stored again"
    assert k["region_cpu_fallback_calls"] == 0
    for (snap, tok), y in zip(snaps, outs):
        assert same(step(fn, tok, snap), y), "a step of an interleaved cache equals a fresh run on a copy of it"
    assert_planes_equal_host(ca, 9)
    assert_planes_equal_host(cb, 10)
    la = torch.cat([y for (w, s, e), y in zip(plan, outs) if w == "a"], 1)
    assert nmse(la, stateless(m, a[:, :9])) <= NMSE_MAX


def test_a_dropped_cache_gives_its_planes_back():
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(9, seed=6)
    cache = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :6], cache)
    step(fn, ids[:, 6:7], cache)
    offs = sorted(st.k_off for st in rt._kv_states.values())
    assert len(rt._kv_states) == 2
    del cache
    torch._dynamo.reset()
    gc.collect()
    if rt.kv_sweep() == 0:
        pytest.skip("the dropped cache is still referenced (Dynamo keeps the tensors of a compiled call alive)")
    assert not rt._kv_states and sum(len(v) for v in rt._kv_pool.values()) == 2
    k0 = c()["rt_kv_creates"]
    h0 = c()["kv_pool_hits"]
    cache = StaticCache(cfg, max_cache_len=L)
    y = step(fn, ids[:, :6], cache)
    assert c()["rt_kv_creates"] == k0 and c()["kv_pool_hits"] - h0 == 2
    assert sorted(st.k_off for st in rt._kv_states.values()) == offs, "the cache made again got the same planes"
    # planes that were not cleared: the run still gives the stateless result, and so does the next step
    y2 = step(fn, ids[:, 6:7], cache)
    assert nmse(torch.cat([y, y2], 1), stateless(m, ids[:, :7])) <= NMSE_MAX
    assert_planes_equal_host(cache, 7)


def test_release_weights_frees_the_planes():
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(8)
    cache = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :6], cache)
    assert len(rt._kv_states) == 2
    tt.release_weights()
    assert not rt._kv_states and not any(rt._kv_pool.values()) and not rt._kv_by_base
    # the same cache continues: its state is made again and stored from the host rows
    r0 = c()["kv_resync_rows"]
    snap = copy.deepcopy(cache)
    y = step(fn, ids[:, 6:7], cache)
    assert c()["kv_resync_rows"] - r0 == 12, "6 rows of two layers"
    assert same(y, step(fn, ids[:, 6:7], snap))


# ------------------------------------------------------------------------------------------
# calls the device does not take
# ------------------------------------------------------------------------------------------
def test_slots_that_start_at_different_cells_run_on_the_cpu_and_the_device_continues():
    """The layers of a Hugging Face cache write at their own length: when two lengths differ the slots of the region
    start at different cells, which the device does not take. The call runs the saved aten subgraph, which is
    eager's computation; with the lengths equal again the device stores the host rows that the CPU call wrote."""
    m, cfg = tiny()
    ids = ids_of(12)
    fn = compiled(m)
    cache = _prefilled(cfg, ids, 3, fn)
    step(fn, ids[:, 3:4], cache)                          # p0 = 4, n_dev = 4 in both layers
    cache.layers[1].cumulative_length.fill_(2)
    eager_cache = copy.deepcopy(cache)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = step(fn, ids[:, 4:5], cache)                  # layer 0 writes cell 4, layer 1 writes cell 2
    with torch.no_grad():
        y_eager = m(input_ids=ids[:, 4:5], past_key_values=eager_cache, use_cache=True).logits
    assert c()["region_cpu_fallback_calls"] == 1
    assert nmse(y, y_eager) <= 1e-12, "the saved aten subgraph is eager's computation"
    for layer in cache.layers:
        layer.cumulative_length.fill_(5)
    r0 = c()["kv_resync_rows"]
    snap = copy.deepcopy(cache)
    y2 = step(fn, ids[:, 5:6], cache)                     # device at cell 5
    # layer 0 lacks cell 4 (its shadow has 4 cells): 1 row; layer 1 was rewritten at cell 2 (shadow cut to 2): rows 2 .. 4
    assert c()["kv_resync_rows"] - r0 == 1 + 3
    assert same(y2, step(fn, ids[:, 5:6], snap))
    assert c()["region_cpu_fallback_calls"] == 1


def test_inference_tensors_run_on_the_cpu():
    """A cache made under torch.inference_mode has no version counter: a change of it could not be seen, so the
    device does not take the call (and says to use torch.no_grad)."""
    m, cfg = tiny()
    ids = ids_of(8)
    fn = compiled(m)
    with torch.inference_mode():
        cache = StaticCache(cfg, max_cache_len=L)
        with pytest.warns(RuntimeWarning, match="inference tensor"):
            y = fn(input_ids=ids[:, :6], past_key_values=cache, use_cache=True).logits
        y2 = fn(input_ids=ids[:, 6:7], past_key_values=cache, use_cache=True).logits
        ref_cache = StaticCache(cfg, max_cache_len=L)
        ref = m(input_ids=ids[:, :6], past_key_values=ref_cache, use_cache=True).logits
        ref2 = m(input_ids=ids[:, 6:7], past_key_values=ref_cache, use_cache=True).logits
    k = c()
    assert k["region_cpu_fallback_calls"] == 2 and k["kv_caches"] == 0 and k["rt_kv_creates"] == 0, k
    assert nmse(y, ref) <= 1e-12 and nmse(y2, ref2) <= 1e-12


def test_the_cache_check_of_verify_catches_a_wrong_plane(monkeypatch):
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(8)
    cache = StaticCache(cfg, max_cache_len=L)
    orig = rt.q8_0_rows

    def bad(x):
        r = orig(x).clone()
        r[1, 40] ^= 1                          # row 1, block 1, a quantized value
        return r

    monkeypatch.setattr(rt, "q8_0_rows", bad)
    with pytest.raises(tt.TesseractError, match=r"plane of KV slot 0 differs from the quantized rows: cell 1 \(row 1\), first block 1"):
        step(fn, ids[:, :6], cache)


# ------------------------------------------------------------------------------------------
# a cache length that changes between calls (Dynamo serves it with one graph in which the length is a symbol)
# ------------------------------------------------------------------------------------------
def _pool(cells=None):
    return sorted(x["cells"] for x in tt.kv_caches() if x["name"] == "(pool)" and (cells is None or x["cells"] == cells))


def test_the_cache_length_changes_between_calls():
    """Caches of three lengths in turn through one compiled forward: the second length makes Dynamo recompile with the
    length as a symbol, the third reuses that graph, and the first length again goes through the symbolic graph. Every
    call is a cached attention on the device; the device graph of a region is made once per length it serves; the
    planes of a dead cache are pooled under (E, cells) and the next cache of that length takes them."""
    m, cfg = tiny(max_pos=96)
    fn = compiled(m)
    ids = ids_of(8, seed=21)
    calls = 0
    graphs_before = 0
    results = []
    for n_cells, expect_new_graphs in ((48, 2), (56, 2), (64, 2), (48, 2), (64, 2), (64, 0)):
        cache = StaticCache(cfg, max_cache_len=n_cells)
        pre = step(fn, ids[:, :5], cache)
        outs = [pre] + [step(fn, ids[:, i:i + 1], cache) for i in range(5, 8)]
        calls += 4
        no_fallback(calls)
        k = c()
        # (the last phase has the length of the one before, in a new cache: the two regions keep their graphs)
        assert k["rt_graphs"] - graphs_before == expect_new_graphs, (n_cells, k["rt_graphs"] - graphs_before)
        graphs_before = k["rt_graphs"]
        results.append(torch.cat(outs, 1))
        assert_planes_equal_host(cache, 8)
        assert {x["cells"] for x in tt.kv_caches() if x["alive"]} >= {n_cells}
        if n_cells in (48, 56):          # the cache dies: its planes go to the pool under their own geometry
            del cache, outs, pre
            gc.collect()
            rt.kv_sweep()
            assert _pool(n_cells) == [n_cells, n_cells]
    k = c()
    assert k["kv_region_calls"] == k["region_calls"] == calls and k["region_cpu_fallback_calls"] == 0
    # regions: a prefill and a decode graph for the first length, and the same two for the symbolic one
    assert k["regions"] == 4, k["regions"]
    # the second 48 took the planes of the first, the second 64 those of the (dead) first 64: no kv_create for them
    assert k["kv_pool_hits"] >= 2, k["kv_pool_hits"]
    assert k["rt_kv_creates"] == k["kv_caches"] - k["kv_pool_hits"]
    assert k["kv_resync_rows"] == 0
    full = stateless(m, ids)                         # (a region call of its own: after the counters were read)
    for y in results:
        assert nmse(y, full) <= NMSE_MAX
    assert same(results[0], results[3]) and same(results[2], results[4]) and same(results[2], results[5]), "the same sequence gives the same bits"


def test_a_symbolic_cache_length_is_a_graph_key_and_a_refusal_is_per_length(monkeypatch):
    """The device graph of a symbolic region is made for the cells of the call: the same region with another length
    frees it and makes the graph of the new length; a length that the device refuses at graph_create runs on the CPU
    for that length only."""
    m, cfg = tiny(max_pos=96)
    fn = compiled(m)
    ids = ids_of(8, seed=22)
    for n_cells in (48, 56):                           # the symbolic graph exists from the second length on
        cache = StaticCache(cfg, max_cache_len=n_cells)
        step(fn, ids[:, :5], cache)
        step(fn, ids[:, 5:6], cache)
    dec = tt.regions()[-1]
    assert dec.key[-1] == (56, 56) and dec.graph is not None
    created = []
    real = tt.device()

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def graph_create(self, prog):
            if "kv_slots" not in prog:
                return real.graph_create(prog)
            created.append(list(prog["kv_slots"]))
            if prog["kv_slots"][0] == 64:
                raise tt.TesseractError("tesseract: the device refuses this geometry (forced by the test)")
            return real.graph_create(prog)

    monkeypatch.setattr(rt, "device", lambda: Spy())
    cache = StaticCache(cfg, max_cache_len=64)
    y = step(fn, ids[:, :5], cache)                    # the prefill region: its graph for 64 cells is refused
    step(fn, ids[:, 5:6], cache)                       # the decode region: refused too
    assert created == [[64, 64], [64, 64]] and c()["region_cpu_fallback_calls"] == 2
    # (the CPU runs the attention in float32 on the exact rows; the device on q8_0 ones: close, not equal)
    assert nmse(y, stateless(m, ids)[:, :5]) <= 1e-2
    step(fn, ids[:, 6:7], cache)                       # refused for that key again: not asked twice
    assert created == [[64, 64], [64, 64]] and c()["region_cpu_fallback_calls"] == 3
    cache2 = StaticCache(cfg, max_cache_len=48)
    step(fn, ids[:, :5], cache2)                       # another length: a graph again, on the device
    assert created[-1] == [48, 48] and c()["region_cpu_fallback_calls"] == 3


def test_generate_twice_with_prompts_of_different_length():
    """``generate`` sizes a new static cache from the prompt (and never below the size of the call before): the
    second prompt is longer, so Dynamo recompiles with the cache length as a symbol; the third is shorter and keeps
    the symbolic graph. Every call of all three is a cached attention on the device, and the tokens are eager's."""
    m, cfg = tiny(max_pos=64)
    ref_model = copy.deepcopy(m)
    prompts = [ids_of(9, seed=31), ids_of(14, seed=32), ids_of(7, seed=33)]
    kw = dict(max_new_tokens=6, min_new_tokens=6, do_sample=False, cache_implementation="static", pad_token_id=0)
    with torch.no_grad():
        refs = [ref_model.generate(input_ids=p, attention_mask=torch.ones_like(p), **kw) for p in prompts]
    m.forward = compiled(m)
    calls = 0
    for i, (p, ref) in enumerate(zip(prompts, refs)):
        with torch.no_grad():
            out = m.generate(input_ids=p, attention_mask=torch.ones_like(p), **kw)
        assert torch.equal(out, ref), i
        calls += 6
        k = c()
        assert k["region_calls"] == calls and k["kv_region_calls"] == calls and k["region_cpu_fallback_calls"] == 0, (i, k)
        assert k["rt_graph_launches"] == calls
    assert any(r.region.kv and r.region.kv[0].cells is None for r in tt.regions()), "a symbolic graph was made"
    assert c()["regions"] == 4, "a prefill and a decode graph for the first length, the same two for every other"


# ------------------------------------------------------------------------------------------
# verify checks the shadow before the call (the reference executor reads the history cells from the device itself)
# ------------------------------------------------------------------------------------------
def _stale_setup(which: str, slot: int):
    """A cache whose host rows changed without the version counter seeing it (``.data``): the state machine trusts a
    shadow that is stale. Returns (fn, cache, ids); the next call starts at cell 7."""
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(12)
    cache = _prefilled(cfg, ids, 6, fn)
    step(fn, ids[:, 6:7], cache)
    layer = cache.layers[slot]
    (layer.keys if which == "K" else layer.values).data[0, :, 3, :] += 1.0
    return fn, cache, ids


@pytest.mark.parametrize("which,slot", [("K", 1), ("V", 0)])
def test_verify_catches_a_stale_shadow(monkeypatch, which, slot):
    """Without the check of the shadow the stale cell is invisible: the reference executor reads the history from the
    device, so the device and the reference agree on the wrong answer and verify stays green. The check compares the
    cells [0, p0) of the planes with the quantized host rows."""
    fn, cache, ids = _stale_setup(which, slot)
    seen = []
    orig = rt._kv_check_shadow
    monkeypatch.setattr(rt, "_kv_check_shadow", lambda *a: seen.append(a[-1]))      # the check of the shadow off
    step(fn, ids[:, 7:8], cache)                                                    # the old verify: no complaint
    assert seen == [7] and c()["region_cpu_fallback_calls"] == 0
    monkeypatch.setattr(rt, "_kv_check_shadow", orig)
    fn, cache, ids = _stale_setup(which, slot)
    with pytest.raises(tt.TesseractError) as ei:
        step(fn, ids[:, 7:8], cache)
    msg = str(ei.value)
    assert f"the {which} plane of KV slot {slot} differs from the quantized host rows" in msg, msg
    assert "cell 3 (of the first 7), first block 0" in msg and "device shadow is stale" in msg and "region" in msg, msg
    # the state is not trusted afterwards: the next call stores the host rows again and passes
    r0 = c()["kv_resync_rows"]
    step(fn, ids[:, 7:8], cache)
    assert c()["kv_resync_rows"] - r0 == 14, "both slots store their first 7 rows again"
    assert_planes_equal_host(cache, 8)


def test_the_shadow_check_reads_each_plane_once_and_skips_an_empty_history(monkeypatch):
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(9)
    reads = []
    real = rt.device()

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def kv_read(self, h, cell0, n):
            reads.append((cell0, n))
            return real.kv_read(h, cell0, n)

    monkeypatch.setattr(rt, "device", lambda: Spy())
    cache = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :5], cache)               # p0 = 0: no history; only the check of the new cells reads, once per slot
    assert sorted(reads) == [(0, 5), (0, 5)], reads
    reads.clear()
    step(fn, ids[:, 5:6], cache)              # p0 = 5, two slots: the history [0, 5) once per slot (K and V in one read)
    assert sorted(reads) == [(0, 5), (0, 5), (5, 1), (5, 1)], reads


def test_no_false_alarm_of_the_shadow_check_over_the_flows(monkeypatch):
    """Reset, rewind, a CPU call in the middle, two caches interleaved, a host edit that bumps the version, a second
    prompt on a reset cache: every flow runs with verify, and the shadow check ran over real history in all of them."""
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(14)
    checked = []
    orig = rt._kv_check_shadow
    monkeypatch.setattr(rt, "_kv_check_shadow", lambda rg, dev, st, kvargs, p0: (checked.append(p0), orig(rg, dev, st, kvargs, p0))[1])
    cache = _prefilled(cfg, ids, 6, fn)
    for i in range(6, 9):
        step(fn, ids[:, i:i + 1], cache)                  # plain decode
    for layer in cache.layers:
        layer.cumulative_length.fill_(3)                  # rewind
    step(fn, ids[:, 9:10], cache)
    force_cpu(monkeypatch, 1)
    step(fn, ids[:, 10:11], cache)                        # a CPU call in the middle
    step(fn, ids[:, 11:12], cache)                        # the device continues: the missing row is stored
    other = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :4], other)                           # two caches interleaved
    step(fn, ids[:, 12:13], cache)
    step(fn, ids[:, 4:5], other)
    cache.layers[0].keys[0, :, 1, :] *= 2.0               # an edit the version counter sees
    step(fn, ids[:, 13:14], cache)
    cache.reset()                                         # a new prompt on the same tensors
    step(fn, ids[:, :5], cache)
    step(fn, ids[:, 5:6], cache)
    assert c()["region_cpu_fallback_calls"] == 1
    assert c()["kv_resync_rows"] > 0
    assert len([p for p in checked if p > 0]) >= 9 and max(checked) >= 6, checked


# ------------------------------------------------------------------------------------------
# long prefill, dynamic prefill, options
# ------------------------------------------------------------------------------------------
def test_dynamic_prefill_lengths():
    m, cfg = tiny()
    fn = compiled(m)
    for n, seed in ((5, 8), (7, 9), (6, 10)):
        ids = ids_of(n + 2, seed=seed)
        cache = StaticCache(cfg, max_cache_len=L)
        y = torch.cat([step(fn, ids[:, :n], cache), step(fn, ids[:, n:n + 1], cache), step(fn, ids[:, n + 1:n + 2], cache)], 1)
        assert nmse(y, stateless(m, ids)) <= NMSE_MAX
    assert c()["region_cpu_fallback_calls"] == 0


def test_prefill_above_680_rows_runs_in_chunks():
    m, cfg = tiny(max_pos=1024)
    fn = compiled(m)
    ids = ids_of(702, seed=11)
    cache = StaticCache(cfg, max_cache_len=720)
    y = step(fn, ids[:, :700], cache)
    k = c()
    assert k["region_calls"] == 1 and k["rt_graph_launches"] >= 2 and k["region_cpu_fallback_calls"] == 0, k
    y1 = step(fn, ids[:, 700:701], cache)
    y2 = step(fn, ids[:, 701:702], cache)
    assert c()["region_cpu_fallback_calls"] == 0
    # a prefix of the sequence fits a stateless region: rows 0 .. 599 of the chunked prefill equal it
    short = stateless(m, ids[:, :600])
    assert nmse(y[:, :600], short) <= NMSE_MAX
    # the same rows in other calls (350 + 350 + the steps): the chunking does not change a row
    cache2 = StaticCache(cfg, max_cache_len=720)
    z = torch.cat([step(fn, ids[:, :350], cache2), step(fn, ids[:, 350:700], cache2)], 1)
    assert nmse(y, z) <= NMSE_MAX
    z12 = torch.cat([step(fn, ids[:, 700:701], cache2), step(fn, ids[:, 701:702], cache2)], 1)
    assert nmse(torch.cat([y1, y2], 1), z12) <= NMSE_MAX
    # eager float32 (statistical)
    with torch.no_grad():
        e = m(input_ids=ids[:, :700], past_key_values=StaticCache(cfg, max_cache_len=720), use_cache=True).logits
    assert nmse(y, e) < 1e-2


def test_option_kv_cache_false_keeps_the_attention_on_the_cpu():
    m, cfg = tiny()
    ids = ids_of(8)
    fn = compiled(m, kv_cache=False)
    cache = StaticCache(cfg, max_cache_len=L)
    y = [step(fn, ids[:, :6], cache), step(fn, ids[:, 6:7], cache), step(fn, ids[:, 7:8], cache)]
    k = c()
    assert k["rt_kv_creates"] == 0 and k["kv_caches"] == 0 and k["kv_region_calls"] == 0, k
    assert k["region_cpu_fallback_calls"] == 0 and k["regions"] > 2, "one region per piece between the attentions"
    with torch.no_grad():
        ref_cache = StaticCache(cfg, max_cache_len=L)
        ref = torch.cat([m(input_ids=ids[:, a:b], past_key_values=ref_cache, use_cache=True).logits for a, b in ((0, 6), (6, 7), (7, 8))], 1)
    assert nmse(torch.cat(y, 1), ref) < 5e-4
    # the same model with the cache on the device
    tt.reset_counters()
    torch._dynamo.reset()
    fn2 = compiled(m)
    cache2 = StaticCache(cfg, max_cache_len=L)
    step(fn2, ids[:, :6], cache2)
    assert c()["kv_region_calls"] == 1


def test_attention_on_the_cpu_leaves_the_cache_alone():
    """attention="cpu": the attention and the cache update stay in the aten graph (the epilogue copy_ included); the
    cache is written by torch, not by the rewrite."""
    m, cfg = tiny()
    ids = ids_of(8)
    fn = compiled(m, attention="cpu")
    cache = StaticCache(cfg, max_cache_len=L)
    y = torch.cat([step(fn, ids[:, :6], cache), step(fn, ids[:, 6:7], cache), step(fn, ids[:, 7:8], cache)], 1)
    k = c()
    assert k["rt_kv_creates"] == 0 and k["kv_region_calls"] == 0 and k["region_cpu_fallback_calls"] == 0, k
    with torch.no_grad():
        ref_cache = StaticCache(cfg, max_cache_len=L)
        ref = torch.cat([m(input_ids=ids[:, a:b], past_key_values=ref_cache, use_cache=True).logits for a, b in ((0, 6), (6, 7), (7, 8))], 1)
    assert nmse(y, ref) < 5e-4
    for a, b in zip(cache.layers, ref_cache.layers):
        assert float((a.keys - b.keys).abs().max()) < 0.1          # the weights are q8_0 on the device


def test_option_resolution(monkeypatch):
    assert tb.resolve_kv_cache(None) is True
    monkeypatch.setenv("TSR_TORCH_KV_CACHE", "0")
    assert tb.resolve_kv_cache(None) is False
    assert tb.resolve_kv_cache(True) is True, "the explicit option wins"
    monkeypatch.setenv("TSR_TORCH_KV_CACHE", "1")
    assert tb.resolve_kv_cache(False) is False and tb.resolve_kv_cache(None) is True
    monkeypatch.setenv("TSR_TORCH_KV_CACHE", "maybe")
    with pytest.raises(ValueError, match="TSR_TORCH_KV_CACHE"):
        tb.resolve_kv_cache(None)
    assert tb.Options.from_dict({"kv_cache": False}).kv_cache is False
    # an unknown value of the variable is an error when a full-mode graph is compiled, not a CPU run
    m, cfg = tiny()
    fn = tt.compile(m.forward, mode="full")
    with pytest.raises(Exception, match="TSR_TORCH_KV_CACHE"):
        step(fn, ids_of(4), StaticCache(cfg, max_cache_len=L))
    # the variable switches the cache off
    monkeypatch.setenv("TSR_TORCH_KV_CACHE", "0")
    torch._dynamo.reset()
    tt.reset_counters()
    fn = compiled(m)
    step(fn, ids_of(6), StaticCache(cfg, max_cache_len=L))
    assert c()["kv_region_calls"] == 0 and c()["rt_kv_creates"] == 0


# ------------------------------------------------------------------------------------------
# the state machine alone (no model): one slot, the device stores what a region call would store
# ------------------------------------------------------------------------------------------
class Slot:
    """One cache slot of E = 2 x 64 values and CELLS cells, driven by hand the way a region call drives it:
    ``rt._kv_prepare`` (before the run), the run's stores (``Device.kv_store``) and the two ``kv_rows_write`` nodes."""

    CELLS = 24

    def __init__(self, bumps=(0, 0), causal0=False, like=None, seed=3):
        """like: another slot whose region this one shares (a second cache of the same compiled region)."""
        from types import SimpleNamespace
        if like is None:
            self.kvc = SimpleNamespace(hkv=2, d=64, cells=self.CELLS, causal0=causal0)
            self.rg = rt.RegionRT(SimpleNamespace(kv=[self.kvc], inputs=[], consts=[], weights=[], scalars=[]), None, 0, 0,
                                  "stub region")
            self.rg.kv_bumps = [bumps]
            self.rid = rt.register_region(self.rg)
        else:
            self.kvc, self.rg, self.rid = like.kvc, like.rg, like.rid
        self.owns_region = like is None
        self.K = torch.zeros(1, 2, self.CELLS, 64)
        self.V = torch.zeros(1, 2, self.CELLS, 64)
        self.bumps = bumps
        self.g = torch.Generator().manual_seed(seed)

    def sibling(self, seed=4):
        """A second cache (own K and V) on the region of this slot."""
        return Slot(bumps=self.bumps, like=self, seed=seed)

    def close(self):
        if self.owns_region:
            rt._REGIONS.remove(self.rg)

    def state(self):
        return rt._kv_by_base.get(rt.WeightCache.key(self.K))

    def rows(self, n):
        return torch.randn(1, 2, n, 64, generator=self.g)         # head-major new rows, as the region returns them

    def begin_call(self):
        """``device_region`` numbers every call of a region, on the device or not."""
        with rt._lock:
            return rt._begin_call(self.rid)

    def pending(self, p0, T):
        """What a device step records for the writes that follow it."""
        return rt._KVPending(p0, T, {}, rt._own_call(self.rid))

    def prepare(self, p0, T):
        self.begin_call()
        pos = torch.arange(p0, p0 + T)
        for t in (self.K, self.V):                                 # the wrapper of AOTAutograd bumps the versions
            for _ in range(self.bumps[0] if t is self.K else self.bumps[1]):
                t._version  # noqa: B018
                torch.autograd.graph.increment_version(t)
        return rt._kv_prepare(self.rg, tt.device(), [self.K, self.V, pos], T), pos

    def device_step(self, p0, T, k=None, v=None, write=True):
        """prepare, the run's stores, the writes. Returns the rows."""
        (handles, q0, states), pos = self.prepare(p0, T)
        assert q0 == p0
        k = self.rows(T) if k is None else k
        v = self.rows(T) if v is None else v
        tt.device().kv_store(handles[0], p0, k[0].transpose(0, 1).reshape(T, -1).contiguous(),
                             v[0].transpose(0, 1).reshape(T, -1).contiguous())
        for st in states:
            st.pending = self.pending(p0, T)
        if write:
            self.writes(p0, T, k, v)
        return k, v

    def writes(self, p0, T, k, v, rid=None):
        """The two cache writes (K, then V) at the cells p0 .. p0 + T, as the region id ``rid`` (default: this slot's)."""
        rid = self.rid if rid is None else rid
        pos = torch.arange(p0, p0 + T)
        rt.kv_rows_write(rid, 0, "k", self.K, pos, k)
        rt.kv_rows_write(rid, 0, "v", self.V, pos, v)

    def cpu_step(self, p0, T, pos=None):
        """A call that ran on the CPU: its writes without a device step, at the cells ``pos`` (default p0 .. p0 + T)."""
        self.begin_call()
        pos = torch.arange(p0, p0 + T) if pos is None else pos
        for t, b in ((self.K, self.bumps[0]), (self.V, self.bumps[1])):
            for _ in range(b):
                torch.autograd.graph.increment_version(t)
        k, v = self.rows(T), self.rows(T)
        rt.kv_rows_write(self.rid, 0, "k", self.K, pos, k)
        rt.kv_rows_write(self.rid, 0, "v", self.V, pos, v)

    def check_planes(self, n):
        st = self.state()
        kb, vb = tt.device().kv_read(st.handle, 0, n)
        for got, host in ((kb, self.K), (vb, self.V)):
            want = sm.q8_0_ref_blocks(host[0, :, :n, :].transpose(0, 1).reshape(n, -1).numpy())
            assert (got.numpy() == want).all()


@pytest.fixture
def slot():
    made = []

    def make(**kw):
        s = Slot(**kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
def test_state_machine_follows_every_transition(slot, bumps):
    s = slot(bumps=bumps)
    # a new cache: a state, n_dev = 0, a first call of 4 rows from cell 0
    s.device_step(0, 4)
    st = s.state()
    assert st is not None and st.n_dev == 4 and st.pending is None
    s.check_planes(4)
    # decode: nothing stored again
    s.device_step(4, 1)
    s.device_step(5, 1)
    assert st.n_dev == 6 and c()["kv_resync_rows"] == 0
    s.check_planes(6)
    # a call on the CPU at cell 6: the shadow stays at 6 cells and the version is followed
    s.cpu_step(6, 1)
    assert st.n_dev == 6 and st.pending is None
    s.device_step(7, 1)                                    # stores host row 6, then runs
    assert c()["kv_resync_rows"] == 1 and st.n_dev == 8
    s.check_planes(8)
    # the host cache changed in place (version): everything below p0 is stored again
    s.K[0, :, 1, :] += 1.0
    s.device_step(8, 1)
    assert c()["kv_resync_rows"] == 1 + 8 and st.n_dev == 9
    s.check_planes(9)
    s.V[0, 0, 8, 0] -= 2.0                                  # the last row, V only
    s.device_step(9, 1)
    assert c()["kv_resync_rows"] == 1 + 8 + 9 and st.n_dev == 10
    s.check_planes(10)
    # the length set back: the shadow keeps the cells below the new start, nothing is stored
    s.device_step(3, 2)
    assert c()["kv_resync_rows"] == 1 + 8 + 9 and st.n_dev == 5
    s.check_planes(5)
    # set back, then a CPU call (writes cell 3, which the shadow holds with the old rows), then the device at cell 4
    s.cpu_step(3, 1)
    assert st.n_dev == 3, "the shadow's cell 3 is not the host's any more"
    s.device_step(4, 1)
    assert c()["kv_resync_rows"] == 1 + 8 + 9 + 1 and st.n_dev == 5
    s.check_planes(5)
    # a device step whose writes never ran (an error in between): the shadow is not advanced
    s.device_step(5, 2, write=False)
    assert st.n_dev == 5 and st.pending is not None
    s.device_step(5, 1)                                     # the same cells again, now with the writes
    assert st.n_dev == 6 and st.pending is None
    s.check_planes(6)
    # a change between the device step and its write (the version is not the recorded one): not trusted afterwards
    (handles, q0, states), pos = s.prepare(6, 1)
    k, v = s.rows(1), s.rows(1)
    tt.device().kv_store(handles[0], 6, k[0].transpose(0, 1).reshape(1, -1).contiguous(), v[0].transpose(0, 1).reshape(1, -1).contiguous())
    st.pending = s.pending(6, 1)
    s.K.add_(0.0)                                           # a bump from somewhere else
    rt.kv_rows_write(s.rid, 0, "k", s.K, pos, k)
    rt.kv_rows_write(s.rid, 0, "v", s.V, pos, v)
    assert st.n_dev == 6, "the step is not counted when a write saw an unexpected version"
    before = c()["kv_resync_rows"]
    s.device_step(7, 1)
    assert c()["kv_resync_rows"] - before == 7, "everything below the call is stored again"
    s.check_planes(8)


def test_prepare_refuses_what_the_device_does_not_take(slot):
    s = slot()
    dev = tt.device()

    def refused(match, K=None, V=None, pos=None, T=2):
        K = s.K if K is None else K
        V = s.V if V is None else V
        pos = torch.arange(3, 3 + T) if pos is None else pos
        with pytest.raises(tt.TesseractError, match=match) as ei:
            rt._kv_prepare(s.rg, dev, [K, V, pos], T)
        assert "does not fit" in str(ei.value) and rt.region_placement_error(ei.value)    # the call runs on the CPU

    refused("not consecutive", pos=torch.tensor([3, 5]))
    refused("not consecutive", pos=torch.tensor([4, 3]))
    refused("int64", pos=torch.tensor([3, 4], dtype=torch.int32))
    refused(r"int64 \[2\]", pos=torch.tensor([3, 4, 5]))
    refused(r"cells 23 \.\. 25 do not fit the 24 cells", pos=torch.tensor([23, 24]))
    refused("not a float32 CPU tensor", K=torch.zeros(1, 2, 24, 64, dtype=torch.float64))
    refused(r"is \(1, 2, 24, 32\)", V=torch.zeros(1, 2, 24, 32))
    refused("not contiguous", K=torch.zeros(1, 2, 64, 24).transpose(2, 3))
    with torch.inference_mode():
        inf = torch.zeros(1, 2, 24, 64)
    refused("inference tensor.*torch.no_grad", K=inf)
    s2 = slot()
    s2.kvc.causal0 = True
    with pytest.raises(tt.TesseractError, match="causal over the first 2 cells of the cache and the call starts at cell 3"):
        rt._kv_prepare(s2.rg, dev, [s2.K, s2.V, torch.arange(3, 5)], 2)
    # nothing above made a state or stored a row
    assert not rt._kv_states and c()["kv_resync_rows"] == 0 and c()["rt_kv_creates"] == 0
    # the same tensor as two slots, and slots at different cells
    from types import SimpleNamespace
    two = SimpleNamespace(hkv=2, d=64, cells=24, causal0=False)
    rg2 = rt.RegionRT(SimpleNamespace(kv=[two, two]), None, 0, 0, "stub region 2")
    with pytest.raises(tt.TesseractError, match="one tensor is the cache of two slots"):
        rt._kv_prepare(rg2, dev, [s.K, s.V, torch.arange(0, 2), s.K, torch.zeros(1, 2, 24, 64), torch.arange(0, 2)], 2)
    with pytest.raises(tt.TesseractError, match=r"slots start at different cells \(5 and 3\)"):
        rt._kv_prepare(rg2, dev, [s.K, s.V, torch.arange(3, 5), torch.zeros(1, 2, 24, 64), torch.zeros(1, 2, 24, 64),
                                  torch.arange(5, 7)], 2)


# ------------------------------------------------------------------------------------------
# writes of a call that ran on the CPU: the cells it wrote are not the shadow's
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
def test_a_cpu_call_with_unsorted_positions_leaves_no_stale_cell(slot, bumps):
    """The positions of a call that ran on the CPU need not be sorted (non-consecutive positions are one of the reasons
    it ran there): the shadow is trusted only below the smallest cell that call wrote."""
    s = slot(bumps=bumps)
    s.device_step(0, 6)
    st = s.state()
    s.cpu_step(5, 2, pos=torch.tensor([5, 4]))
    assert st.n_dev == 4 and st.pending is None
    s.device_step(6, 1)                                  # stores the host rows 4 and 5 again
    assert c()["kv_resync_rows"] == 2 and st.n_dev == 7
    s.check_planes(7)


@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
def test_a_cpu_call_of_two_rows_below_the_device_length(slot, bumps):
    s = slot(bumps=bumps)
    s.device_step(0, 6)
    st = s.state()
    s.cpu_step(2, 2)                                     # rewrites the cells 2 and 3 on the host only
    assert st.n_dev == 2
    s.device_step(6, 1)
    assert c()["kv_resync_rows"] == 4 and st.n_dev == 7
    s.check_planes(7)


@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
def test_a_write_of_no_rows_changes_no_cell(slot, bumps):
    s = slot(bumps=bumps)
    s.device_step(0, 4)
    st = s.state()
    s.device_step(4, 1, write=False)                     # a device step whose writes are still to come
    pend = st.pending
    assert st.n_dev == 4 and pend is not None
    empty = torch.zeros(0, dtype=torch.int64)
    s.begin_call()
    for t, b in ((s.K, bumps[0]), (s.V, bumps[1])):
        for _ in range(b):
            torch.autograd.graph.increment_version(t)
    rt.kv_rows_write(s.rid, 0, "k", s.K, empty, s.rows(0))
    rt.kv_rows_write(s.rid, 0, "v", s.V, empty, s.rows(0))
    assert st.n_dev == 4 and st.pending is pend and not pend.seen
    # on a state with no step pending the shadow is left alone, too
    s.device_step(5, 1)
    st = s.state()
    n = st.n_dev
    s.cpu_step(0, 0, pos=empty)
    assert st.n_dev == n and st.pending is None
    s.check_planes(n)


@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
def test_the_writes_of_another_call_do_not_complete_a_device_step(slot, bumps):
    """A device step whose writes never ran (an exception between the region call and the writes) leaves its record. A
    later call of the region at the same cells that ran on the CPU is not that step: the shadow holds the device's rows
    there and the host cache holds the CPU's."""
    s = slot(bumps=bumps)
    s.device_step(0, 5)
    st = s.state()
    s.device_step(5, 2, write=False)
    assert st.n_dev == 5 and st.pending is not None
    s.cpu_step(5, 2)
    assert st.n_dev == 5 and st.pending is None
    s.device_step(7, 1)
    # the wrapper's bump makes the CPU writes' version unexpected: the shadow is stored again from cell 0
    assert c()["kv_resync_rows"] == (2 if bumps == (0, 0) else 7) and st.n_dev == 8
    s.check_planes(8)


def test_a_device_step_is_matched_by_the_writes_of_its_own_call_only(slot):
    s = slot()
    s.device_step(0, 3)
    st = s.state()
    s.device_step(3, 2, write=False)
    rows = s.rows(2), s.rows(2)
    s.begin_call()                                       # another call of the region ...
    s.writes(3, 2, *rows)                                # ... whose writes arrive at the same cells
    assert st.n_dev == 3, "the record of the earlier call is not completed by these writes"
    other = rt.RegionRT(s.rg.region, None, 0, 0, "other stub region")
    other.kv_bumps = [(0, 0)]
    orid = rt.register_region(other)
    try:
        # this thread calls another region after the device step; writes that come with that region's id are not the step's
        k, v = s.device_step(3, 2, write=False)
        with rt._lock:
            rt._begin_call(orid)
        s.writes(3, 2, *rows, rid=orid)
        assert st.n_dev == 3 and st.pending is None
        # the same, but the step's own writes (its own region id) arrive after the other region's call: they complete it
        k, v = s.device_step(3, 2, write=False)
        with rt._lock:
            rt._begin_call(orid)
        s.writes(3, 2, k, v)
        assert st.n_dev == 5 and st.pending is None
        before = c()["kv_resync_rows"]
        s.device_step(5, 1)
        assert c()["kv_resync_rows"] == before, "no row is stored again"
        s.check_planes(6)
    finally:
        rt._REGIONS.remove(other)


def _in(pool, fn, *a, **kw):
    """Run fn on the single worker thread of ``pool`` and return its result."""
    return pool.submit(fn, *a, **kw).result()


@pytest.fixture
def two_threads():
    from concurrent.futures import ThreadPoolExecutor
    a, b, d = ThreadPoolExecutor(1), ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    yield a, b, d
    for p in (a, b, d):
        p.shutdown()


def test_two_caches_on_one_region_from_two_threads_cost_no_store(slot, two_threads):
    """One compiled region serves two caches from two threads: the other thread's region call falls between a thread's
    region call and its cache writes. Each thread's step is still completed by its own writes."""
    ta, tb, _ = two_threads
    A = slot(bumps=(1, 1))
    B = A.sibling()
    _in(ta, A.device_step, 0, 4)
    _in(tb, B.device_step, 0, 6)
    base = c()["kv_resync_rows"]
    for r in range(5):
        pa, pb = 4 + r, 6 + r
        k, v = _in(ta, A.device_step, pa, 1, write=False)     # A: region call, no writes yet
        _in(tb, B.device_step, pb, 1)                           # B: a whole step in between
        _in(ta, A.writes, pa, 1, k, v)                          # A: its writes
    assert c()["kv_resync_rows"] == base, "no row stored again"
    assert A.state().n_dev == 9 and B.state().n_dev == 11
    assert A.state().pending is None and B.state().pending is None
    A.check_planes(9)
    B.check_planes(11)


@pytest.mark.parametrize("bumps", [(0, 0), (1, 1)], ids=["no-wrapper", "wrapper-bumps"])
@pytest.mark.parametrize("own_call", [True, False], ids=["other-thread-called-region", "other-thread-never-called"])
def test_writes_of_another_thread_at_the_cells_of_a_step_do_not_complete_it(slot, two_threads, bumps, own_call):
    ta, tb, tc = two_threads
    A = slot(bumps=bumps)
    B = A.sibling()
    _in(ta, A.device_step, 0, 4)
    _in(tb, B.device_step, 0, 4)
    st = A.state()
    _in(ta, A.device_step, 4, 1, write=False)             # A's region call; its writes do not come
    assert st.pending is not None
    other = tb if own_call else tc                         # tc: a thread that never called the region
    if own_call:
        _in(tb, A.begin_call)                               # a call of B's own, then its writes into A's cache
    _in(other, A.writes, 4, 1, A.rows(1), A.rows(1))
    assert st.n_dev == 4 and st.pending is None
    base = c()["kv_resync_rows"]
    _in(ta, A.device_step, 5, 1)
    assert c()["kv_resync_rows"] > base, "the rows of the other thread are stored again"
    assert st.n_dev == 6
    A.check_planes(6)


def test_a_call_that_the_device_refuses_is_a_call_of_its_own(slot, monkeypatch):
    s = slot(bumps=(0, 0))

    def refuse(*a, **kw):
        raise tt.TesseractError("tsr_rt: graph: modelled refusal: does not fit the device")

    assert rt.region_placement_error(tt.TesseractError("x does not fit y"))
    monkeypatch.setattr(rt, "_device_run", refuse)
    monkeypatch.setattr(rt, "_cpu_run", lambda rg, ext: [])
    s.device_step(0, 5)
    st = s.state()
    s.device_step(5, 2, write=False)                       # the device holds rows 5 and 6; their writes never run
    assert st.pending is not None
    pos = torch.arange(5, 7)
    with pytest.warns(RuntimeWarning, match="runs on the CPU"):
        assert rt.device_region(s.rid, s.K, s.V, pos) == ()  # the call that the device refuses: a call of its own
    assert c()["region_cpu_fallback_calls"] == 1
    s.writes(5, 2, s.rows(2), s.rows(2))                    # its writes: other rows at the cells of the pending step
    assert st.n_dev == 5 and st.pending is None
    before = c()["kv_resync_rows"]
    s.device_step(7, 1)
    assert c()["kv_resync_rows"] - before == 2, "cells 5 and 6 are stored again"
    s.check_planes(8)


# -- the same through compiled modules, with positions the caller gives
class PosAttn(nn.Module):
    """One cached attention with the positions given by the caller (the idiom of a static cache whose update takes
    ``cache_position``)."""

    def __init__(self, cells=32, hid=256, H=4, Hkv=2, D=64):
        super().__init__()
        self.L, self.H, self.Hkv, self.D = cells, H, Hkv, D
        self.q = nn.Linear(hid, H * D)
        self.k = nn.Linear(hid, Hkv * D)
        self.v = nn.Linear(hid, Hkv * D)
        self.o = nn.Linear(H * D, hid, bias=False)
        self.kc = torch.zeros(1, Hkv, cells, D)
        self.vc = torch.zeros(1, Hkv, cells, D)

    def forward(self, x, pos, mask):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.k(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        v = self.v(x).view(B, T, self.Hkv, self.D).transpose(1, 2)
        self.kc.index_copy_(2, pos, k)
        self.vc.index_copy_(2, pos, v)
        g = self.H // self.Hkv
        ke = self.kc[:, :, None].expand(B, self.Hkv, g, self.L, self.D).reshape(B, self.H, self.L, self.D)
        ve = self.vc[:, :, None].expand(B, self.Hkv, g, self.L, self.D).reshape(B, self.H, self.L, self.D)
        a = F.scaled_dot_product_attention(q, ke, ve, attn_mask=mask, scale=self.D ** -0.5)
        return self.o(a.transpose(1, 2).reshape(B, T, -1))


def pos_run(fn, x, pos, cells):
    mask = torch.where(torch.arange(cells)[None, :] <= pos[:, None], 0.0, float("-inf"))[None, None]
    with torch.no_grad():
        return fn(x, pos, mask)


def pos_attn(cells=32, seed=0):
    torch.manual_seed(seed)
    return PosAttn(cells).eval(), torch.randn(1, 16, 256, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("verify", [False, True], ids=["verify-off", "verify-on"])
def test_a_cpu_call_with_unsorted_positions_then_a_device_call_equals_a_fresh_run(monkeypatch, verify):
    monkeypatch.setattr(rt, "_verify", rt._verify)           # restored after the test
    cells = 32
    m, xs = pos_attn(cells)
    fn = tt.compile(m, mode="full", verify=verify)
    for a in (0, 2, 4):                                      # the cells 0 .. 5 on the device, two rows a call
        pos_run(fn, xs[:, a:a + 2], torch.arange(a, a + 2), cells)
    k = c()
    assert k["kv_region_calls"] == 3 and k["region_cpu_fallback_calls"] == 0, k
    pos_run(fn, xs[:, 6:8], torch.tensor([5, 4]), cells)     # not consecutive: the call runs on the CPU
    assert c()["region_cpu_fallback_calls"] == 1
    st = rt._kv_by_base[rt.WeightCache.key(m.kc)]
    assert st.n_dev == 4
    m2 = copy.deepcopy(m)                                    # a fresh run on the same host cache content
    fn2 = tt.compile(m2, mode="full", verify=verify)
    want = pos_run(fn2, xs[:, 8:10], torch.tensor([6, 7]), cells)
    r0 = c()["kv_resync_rows"]
    got = pos_run(fn, xs[:, 8:10], torch.tensor([6, 7]), cells)
    assert c()["kv_resync_rows"] - r0 == 2, "the host rows 4 and 5 were stored again"
    assert same(got, want)
    assert c()["region_cpu_fallback_calls"] == 1


class _Interrupt(dict):
    """``rt._kv_by_base`` that raises KeyboardInterrupt on the n-th ``get`` after ``arm`` (in ``kv_rows_write``, after
    the host write and before the state is updated)."""
    left = -1

    def arm(self, n):
        self.left = n

    def get(self, k, d=None):
        self.left -= 1
        if self.left == 0:
            raise KeyboardInterrupt("interrupted between the region call and the cache writes")
        return super().get(k, d)


@pytest.mark.parametrize("which", [1, 2], ids=["at-the-K-write", "at-the-V-write"])
def test_an_interrupt_between_the_region_call_and_the_cache_writes(monkeypatch, which):
    """The device step ran (its rows are in the planes) and the writes did not all run. The retry of the call runs on
    the CPU (the device refuses it), which writes the same cells with other rows; the next device call equals a fresh
    run on the host cache."""
    monkeypatch.setattr(rt, "_verify", rt._verify)
    cells = 32
    m, xs = pos_attn(cells)
    fn = tt.compile(m, mode="full", verify=True)
    for a in (0, 2):
        pos_run(fn, xs[:, a:a + 2], torch.arange(a, a + 2), cells)
    trap = _Interrupt(rt._kv_by_base)
    monkeypatch.setattr(rt, "_kv_by_base", trap)
    trap.arm(which)
    with pytest.raises(KeyboardInterrupt):
        pos_run(fn, xs[:, 4:6], torch.arange(4, 6), cells)
    monkeypatch.setattr(rt, "_kv_by_base", dict(trap))
    force_cpu(monkeypatch, 1)
    pos_run(fn, xs[:, 4:6], torch.arange(4, 6), cells)       # the retry, on the CPU
    assert c()["region_cpu_fallback_calls"] == 1
    m2 = copy.deepcopy(m)
    want = pos_run(tt.compile(m2, mode="full", verify=True), xs[:, 6:8], torch.arange(6, 8), cells)
    got = pos_run(fn, xs[:, 6:8], torch.arange(6, 8), cells)
    assert same(got, want)
    st = rt._kv_by_base[rt.WeightCache.key(m.kc)]
    assert st.n_dev == 8
    assert_planes_equal_host_of(m, 8)


def assert_planes_equal_host_of(m, n):
    st = rt._kv_by_base[rt.WeightCache.key(m.kc)]
    kb, vb = tt.device().kv_read(st.handle, 0, n)
    for got, host in ((kb, m.kc), (vb, m.vc)):
        want = sm.q8_0_ref_blocks(host[0, :, :n, :].transpose(0, 1).reshape(n, -1).numpy())
        assert (got.numpy() == want).all()


# ------------------------------------------------------------------------------------------
# the checks of verify look at both planes
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("plane", ["K", "V"])
def test_the_cache_check_of_verify_looks_at_both_planes(monkeypatch, plane):
    """A device whose ``kv_read`` returns a flipped byte in one plane: the check of the new cells names that plane."""
    m, cfg = tiny()
    fn = compiled(m)
    real = rt.device()

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def kv_read(self, h, cell0, n):
            kb, vb = real.kv_read(h, cell0, n)
            if plane == "K":
                kb = kb.clone()
                kb[1, 40] ^= 1
            else:
                vb = vb.clone()
                vb[1, 40] ^= 1
            return kb, vb

    monkeypatch.setattr(rt, "device", lambda: Spy())
    with pytest.raises(tt.TesseractError, match=rf"the {plane} plane of KV slot 0 differs from the quantized rows: cell 1 \(row 1\)"):
        step(fn, ids_of(8)[:, :6], StaticCache(cfg, max_cache_len=L))


# ------------------------------------------------------------------------------------------
# a device that is closed and opened again between steps
# ------------------------------------------------------------------------------------------
def test_a_device_reopened_between_steps_gets_new_planes_and_every_row_again():
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(9, seed=4)
    cache = StaticCache(cfg, max_cache_len=L)
    step(fn, ids[:, :6], cache)
    step(fn, ids[:, 6:7], cache)
    serial = tt.device().serial
    tt.close_device()
    assert not rt._kv_states and not rt._kv_by_base and not any(rt._kv_pool.values())
    snap = copy.deepcopy(cache)
    r0 = c()["kv_resync_rows"]
    y = step(fn, ids[:, 7:8], cache)
    assert tt.device().serial != serial
    states = list(rt._kv_states.values())
    assert len(states) == 2 and all(st.serial == tt.device().serial and st.n_dev == 8 for st in states)
    assert c()["kv_resync_rows"] - r0 == 14, "7 rows below the call, two layers"
    assert same(y, step(fn, ids[:, 7:8], snap)), "equal to the stateless run on a copy of the host cache"
    assert_planes_equal_host(cache, 8)
    assert c()["region_cpu_fallback_calls"] == 0


# ------------------------------------------------------------------------------------------
# a slot whose cache length is a symbol: V of another length than K
# ------------------------------------------------------------------------------------------
def test_a_symbolic_slot_called_with_a_v_cache_of_another_length_is_refused(slot):
    """The length of the K tensor is the slot's cells; a V tensor of another length does not fit, and the call runs on
    the CPU. (A compiled model cannot produce this call: the expansion of the cache in front of the attention reshapes
    both tensors to one length, so the unit test of the check is the test.)"""
    from types import SimpleNamespace
    kvc = SimpleNamespace(hkv=2, d=64, cells=None, causal0=False)
    rg = rt.RegionRT(SimpleNamespace(kv=[kvc]), None, 0, 0, "stub region")
    K, pos = torch.zeros(1, 2, 24, 64), torch.arange(0, 2)
    assert rt._kv_check_call(rg, [K, torch.zeros(1, 2, 24, 64), pos], 2) == (0, [24])
    with pytest.raises(tt.TesseractError, match=r"the V cache is \(1, 2, 20, 64\), expected \(1, 2, 24, 64\)") as ei:
        rt._kv_check_call(rg, [K, torch.zeros(1, 2, 20, 64), pos], 2)
    assert "does not fit" in str(ei.value) and rt.region_placement_error(ei.value)
    assert not rt._kv_states and c()["rt_kv_creates"] == 0


# ------------------------------------------------------------------------------------------
# device memory: the pool of free planes is bounded, and a full heap frees what nothing uses
# ------------------------------------------------------------------------------------------
def _pool_geometries():
    return {(x["E"], x["cells"]) for x in tt.kv_caches() if x["name"] == "(pool)"}


def test_the_pool_keeps_at_most_four_geometries_while_generate_grows_the_cache():
    """Every prompt of another length is a cache of another length for ``generate``: the planes of the lengths before it
    are not used again. The pool holds the four most recent geometries; a length that comes again takes its planes."""
    m, cfg = tiny(max_pos=64)
    ref_model = copy.deepcopy(m)
    lengths = [9, 14, 7, 20, 26, 11, 30]                      # cache lengths 13, 18, 11, 24, 30, 15, 34
    kw = dict(max_new_tokens=4, min_new_tokens=4, do_sample=False, cache_implementation="static", pad_token_id=0)
    prompts = [ids_of(n, seed=40 + n) for n in lengths] + [ids_of(30, seed=77)]
    with torch.no_grad():
        refs = [ref_model.generate(input_ids=p, attention_mask=torch.ones_like(p), **kw) for p in prompts]
    m.forward = compiled(m)
    seen = set()
    for i, (p, ref) in enumerate(zip(prompts, refs)):
        hits = c()["kv_pool_hits"]
        with torch.no_grad():
            out = m.generate(input_ids=p, attention_mask=torch.ones_like(p), **kw)
        assert torch.equal(out, ref), i
        seen.add(len(p[0]) + 4)
        assert len(_pool_geometries()) <= 4, (i, tt.kv_caches())
        if i == len(prompts) - 1:                              # the length of the generate before it
            assert c()["kv_pool_hits"] - hits == 2, "the two layers took the planes of the cache of that length"
        gc.collect()
    rt.kv_sweep()
    assert len(seen) >= 7 and 1 <= len(_pool_geometries()) <= 4
    assert c()["region_cpu_fallback_calls"] == 0


def test_the_pool_frees_the_planes_of_the_least_recent_geometry(monkeypatch):
    """Unit test of the bound: five geometries are pooled in turn; the first is freed on the device, and a geometry that
    is taken from or pooled again is the most recent."""
    freed = []

    class Dev:
        serial = 7

        def kv_free(self, h):
            freed.append(h)

    class FakeC:
        @staticmethod
        def is_open():
            return True

        @staticmethod
        def get():
            return Dev()

    monkeypatch.setattr(rt, "_C", FakeC)
    monkeypatch.setattr(rt, "_kv_pool", {})

    def st(E, cells, h):
        return rt._KVState(h, 7, E, cells, (), (), None, None, 0 + h * 8)

    for i, geo in enumerate([(128, 10), (128, 11), (128, 12), (128, 13)]):
        rt._kv_pool_put(st(*geo, i))
    assert list(rt._kv_pool) == [(128, 10), (128, 11), (128, 12), (128, 13)] and not freed
    assert rt._kv_pool_take(128, 10, 7) == (0, 0)                 # taken: the geometry is gone (its list is empty) ...
    assert (128, 10) not in rt._kv_pool
    rt._kv_pool_put(st(128, 11, 40))                               # ... and (128, 11) is now the most recent
    assert list(rt._kv_pool) == [(128, 12), (128, 13), (128, 11)]
    rt._kv_pool_put(st(128, 14, 5))
    rt._kv_pool_put(st(128, 15, 6))                                # the fifth: the least recent goes
    assert freed == [2] and list(rt._kv_pool) == [(128, 13), (128, 11), (128, 14), (128, 15)]
    assert sorted(h for _, h, _ in rt._kv_pool[(128, 11)]) == [1, 40]
    # planes of another device are dropped, not freed
    rt._kv_pool[(128, 13)] = [(0, 99, 3)]
    rt._kv_pool_put(st(128, 16, 8))
    assert freed == [2] and (128, 13) not in rt._kv_pool and len(rt._kv_pool) == 4


class _FullHeap:
    """A device whose WEIGHT heap is full while any KV plane of a dead cache is allocated (a state not swept yet, or the
    pool): ``upload`` and ``graph_create`` fail then, as the runtime fails when the planes fill the heap."""

    def __init__(self, real, upload=True, graph=True, always=False, any_plane=False):
        self.real, self.fail_upload, self.fail_graph, self.always = real, upload, graph, always
        self.any_plane = any_plane        # full while any KV plane is allocated, a dead cache's or not
        self.upload_calls = self.graph_calls = 0

    def __getattr__(self, name):
        return getattr(self.real, name)

    @staticmethod
    def dead_planes():
        return sum(1 for st in rt._kv_states.values() if st.dead()) + sum(len(v) for v in rt._kv_pool.values())

    @staticmethod
    def all_planes():
        return len(rt._kv_states) + sum(len(v) for v in rt._kv_pool.values())

    def upload(self, w):
        self.upload_calls += 1
        if self.fail_upload and (self.all_planes() if self.any_plane else self.dead_planes()):
            raise tt.TesseractError("tsr_rt: WEIGHT heap full: planes of dead caches fill it (modelled)")
        return self.real.upload(w)

    def graph_create(self, prog):
        self.graph_calls += 1
        if self.fail_graph and (self.always or self.dead_planes()):
            raise tt.TesseractError("tsr_rt: graph: the KV planes need 1.0 MiB per channel: does not fit the WEIGHT heap "
                                    "(modelled)")
        return self.real.graph_create(prog)


def _dead_cache():
    """Run a cached model once and let its cache die: its planes are allocated, in a state that is not swept yet."""
    m, cfg = tiny()
    fn = compiled(m)
    cache = StaticCache(cfg, max_cache_len=48)
    step(fn, ids_of(6), cache)
    del cache
    gc.collect()
    return fn


def _stateless_model():
    m2, _ = tiny(seed=5)
    return m2, compiled(m2)


def test_a_weight_upload_that_finds_the_heap_full_frees_the_planes_of_a_dead_cache(monkeypatch):
    fn = _dead_cache()
    spy = _FullHeap(rt.device(), graph=False)
    if not spy.dead_planes():
        pytest.skip("the dropped cache is still referenced (Dynamo keeps the tensors of a compiled call alive)")
    monkeypatch.setattr(rt, "device", lambda: spy)
    m2, fn2 = _stateless_model()
    with torch.no_grad():
        y = fn2(input_ids=ids_of(6), use_cache=False).logits
        ref = m2(input_ids=ids_of(6), use_cache=False).logits
    assert spy.dead_planes() == 0 and not rt._kv_states and not any(rt._kv_pool.values()), "swept and freed"
    k = c()
    assert k["region_cpu_fallback_calls"] == 0 and k["region_calls"] == 1 + 1, k
    assert nmse(y, ref) < 5e-4


def test_a_dead_cache_that_only_a_collection_frees(monkeypatch):
    """The cache of a finished ``generate`` can sit in a reference cycle: its tensors are alive until the cyclic collector
    runs, so only the collection inside ``free_idle_device_memory`` makes its planes free."""
    m, cfg = tiny()
    fn = compiled(m)
    gc.collect()
    gc.disable()
    try:
        cache = StaticCache(cfg, max_cache_len=48)
        step(fn, ids_of(6), cache)
        cache._cycle = cache
        del cache                                           # no gc.collect() here
        spy = _FullHeap(rt.device(), graph=False, any_plane=True)
        assert spy.all_planes() > 0 and rt._kv_states
        assert not any(st.dead() for st in rt._kv_states.values()), "alive until the collector runs"
        monkeypatch.setattr(rt, "device", lambda: spy)
        m2, fn2 = _stateless_model()
        with torch.no_grad():
            y = fn2(input_ids=ids_of(6), use_cache=False).logits
            ref = m2(input_ids=ids_of(6), use_cache=False).logits
        if rt._kv_states and not any(st.dead() for st in rt._kv_states.values()):
            gc.collect()
            if not any(st.dead() for st in rt._kv_states.values()):
                pytest.skip("the dropped cache is still referenced after a collection (Dynamo keeps it alive)")
        assert spy.all_planes() == 0 and not rt._kv_states and not any(rt._kv_pool.values()), "collected, swept and freed"
        k = c()
        assert k["region_cpu_fallback_calls"] == 0 and k["region_calls"] == 1 + 1, k
        assert nmse(y, ref) < 5e-4
    finally:
        gc.enable()


def test_a_program_that_finds_the_heap_full_frees_the_planes_of_a_dead_cache_and_is_made_once(monkeypatch):
    fn = _dead_cache()
    spy = _FullHeap(rt.device(), upload=False)
    if not spy.dead_planes():
        pytest.skip("the dropped cache is still referenced (Dynamo keeps the tensors of a compiled call alive)")
    monkeypatch.setattr(rt, "device", lambda: spy)
    m2, fn2 = _stateless_model()
    with torch.no_grad():
        y = fn2(input_ids=ids_of(6), use_cache=False).logits
    assert spy.graph_calls == 2, "one failure, one retry"
    assert spy.dead_planes() == 0
    k = c()
    assert k["region_cpu_fallback_calls"] == 0 and k["region_calls"] == 2, k
    assert all(rg.failed is None for rg in tt.regions()[-1:])
    assert nmse(y, m2(input_ids=ids_of(6), use_cache=False).logits) < 5e-4


def test_a_program_that_still_does_not_fit_after_the_retry_is_refused_as_before(monkeypatch):
    spy = _FullHeap(rt.device(), always=True)
    monkeypatch.setattr(rt, "device", lambda: spy)
    m2, fn2 = _stateless_model()
    ids = ids_of(6)
    with torch.no_grad():
        y = fn2(input_ids=ids, use_cache=False).logits
        assert spy.graph_calls == 2, "the failure and one retry"
        rg = tt.regions()[-1]
        assert rg.failed is not None and "WEIGHT heap" in rg.failed[1]
        y2 = fn2(input_ids=ids, use_cache=False).logits
    assert spy.graph_calls == 2, "the refusal is remembered for the key: no new try"
    k = c()
    assert k["region_cpu_fallback_calls"] == 2 and k["region_calls"] == 0, k
    assert same(y, y2) and nmse(y, m2(input_ids=ids, use_cache=False).logits) < 1e-10


def test_a_cache_that_does_not_fit_frees_idle_memory_and_retries_once(monkeypatch):
    """kv_create: the first call fails with the heap full, the dead cache's planes are freed, the second succeeds."""
    fn = _dead_cache()
    real = rt.device()
    dead = _FullHeap.dead_planes
    if not dead():
        pytest.skip("the dropped cache is still referenced (Dynamo keeps the tensors of a compiled call alive)")

    class Spy(_FullHeap):
        creates = 0

        def kv_create(self, E, cells):
            Spy.creates += 1
            if dead():
                raise tt.TesseractError("tsr_rt: kv_create: does not fit the WEIGHT heap (modelled)")
            return real.kv_create(E, cells)

    spy = Spy(real, upload=False, graph=False)
    monkeypatch.setattr(rt, "device", lambda: spy)
    m, cfg = tiny(seed=3)
    cache = StaticCache(cfg, max_cache_len=40)
    step(compiled(m), ids_of(6), cache)
    assert Spy.creates == 3, "the failure, the retry, the second layer"
    assert dead() == 0 and len(rt._kv_states) == 2 and not any(rt._kv_pool.values()), "the dead cache's planes are gone"
    assert c()["region_cpu_fallback_calls"] == 0 and c()["kv_region_calls"] == 2, "the dead cache's call and this one"


# ------------------------------------------------------------------------------------------
# the debug log never changes what runs
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("graph_info", ["chunk", "error"])
def test_the_debug_log_does_not_change_what_runs(monkeypatch, capsys, graph_info):
    """TSR_TORCH_DEBUG = 1 asks the device for the plan of the first run of a region. For a prefill that runs in chunks
    that is the plan of the first chunk (the log says so); an error of the question is a line of the log, not a
    placement error: nothing runs on the CPU and the next step stores no row again."""
    monkeypatch.setattr(rt, "_debug", 1)
    if graph_info == "error":
        real = rt.device()

        class Spy:
            def __getattr__(self, name):
                return getattr(real, name)

            def graph_info(self, h, T):
                raise tt.TesseractError("tesseract: graph_info: T does not fit an attn op (forced by the test)")

        monkeypatch.setattr(rt, "device", lambda: Spy())
    m, cfg = tiny(max_pos=1024)
    fn = compiled(m)
    ids = ids_of(702, seed=11)
    cache = StaticCache(cfg, max_cache_len=720)
    step(fn, ids[:, :700], cache)
    k = c()
    assert k["region_cpu_fallback_calls"] == 0 and k["rt_graph_launches"] >= 2, k
    step(fn, ids[:, 700:701], cache)
    k = c()
    assert k["region_cpu_fallback_calls"] == 0 and k["kv_resync_rows"] == 0, k
    err = capsys.readouterr().err
    if graph_info == "chunk":
        assert "first plan: chunk of " in err and " rows of 700:" in err, err
    else:
        assert "no plan information for T=700" in err and "does not fit an attn op" in err, err


# ------------------------------------------------------------------------------------------
# a mask that does not fit a cached attention is a refusal
# ------------------------------------------------------------------------------------------
def test_a_mask_that_does_not_fit_a_cached_attention_is_a_refusal(monkeypatch):
    """The lowering guarantees a [.., T, cells] mask for a cached attention. A call whose mask is not (a row for T > 1
    rows, another length) is refused as a placement error (the call then runs on the CPU), not raised by the runtime."""
    cells = 32
    m, xs = pos_attn(cells)
    fn = tt.compile(m, mode="full", verify=True)
    rec = []
    orig = rt._device_run

    def spy(rg, ins, cs, ws, ss, kvargs=()):
        rec.append((rg, ins, cs, ws, ss, kvargs))
        return orig(rg, ins, cs, ws, ss, kvargs)

    monkeypatch.setattr(rt, "_device_run", spy)
    pos_run(fn, xs[:, 0:2], torch.arange(0, 2), cells)
    rg, ins, cs, ws, ss, kvargs = rec[-1]
    assert rg.mask_slot == {next(sp.value for sp in rg.region.inputs if sp.kind == "mask"): 0}
    base = dict(c())

    def with_mask(mask):
        return [mask if sp.kind == "mask" else x for sp, x in zip(rg.region.inputs, ins)]

    good = next(x for sp, x in zip(rg.region.inputs, ins) if sp.kind == "mask")
    assert tuple(good.shape) == (1, 1, 2, cells)
    for bad in (good[:, :, -1:, :], good[..., :cells - 4], torch.zeros(2, 1, 2, cells), torch.zeros(1, 1, 3, cells)):
        with pytest.raises(tt.TesseractError, match=r"a cached attention does not fit the device: the attention mask of slot 0") as ei:
            with rt._lock:
                orig(rg, with_mask(bad), cs, ws, ss, kvargs)
        assert rt.region_placement_error(ei.value) and "does not fit" in str(ei.value)
    k = c()
    assert k["region_calls"] == base["region_calls"] and k["kv_resync_rows"] == base["kv_resync_rows"]
    with rt._lock:
        orig(rg, with_mask(good), cs, ws, ss, kvargs)               # the right mask is taken
    assert c()["region_calls"] == base["region_calls"] + 1


# ------------------------------------------------------------------------------------------
# logits_to_keep = 1: ``slice(-1, None)`` of the hidden states in front of the head
# ------------------------------------------------------------------------------------------
def test_logits_to_keep_one_row_is_an_identity_for_one_row_and_a_slice_for_several():
    """For a call of one row the slice keeps the whole value (the head joins the region, and the result is the call
    without ``logits_to_keep``, bit for bit); for a prefill of T > 1 rows it drops rows and stays outside the region
    (the last row of the logits is the last row of the full call). The logits are close to eager's, nothing runs on the CPU."""
    m, cfg = tiny()
    fn = compiled(m)
    ids = ids_of(8, seed=12)
    cache, full = StaticCache(cfg, max_cache_len=L), StaticCache(cfg, max_cache_len=L)
    y = step(fn, ids[:, :6], cache, logits_to_keep=1)
    assert y.shape == (1, 1, 256)
    yf = step(fn, ids[:, :6], full)
    assert nmse(y, yf[:, -1:]) <= NMSE_MAX
    with torch.no_grad():
        e = m(input_ids=ids[:, :6], past_key_values=StaticCache(cfg, max_cache_len=L), use_cache=True, logits_to_keep=1).logits
    assert nmse(y, e) < 5e-4
    y1 = step(fn, ids[:, 6:7], cache, logits_to_keep=1)
    y1f = step(fn, ids[:, 6:7], full)
    assert same(y1, y1f)
    k = c()
    assert k["region_cpu_fallback_calls"] == 0 and k["kv_resync_rows"] == 0, k
    assert_planes_equal_host(cache, 7)

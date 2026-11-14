"""The differential fuzzer of full mode (sw/torch/tools/fuzz_full.py) on a fixed seed range, on the simulator.

Two generators: random models with matmuls (eager against matmul mode against full mode, judged with a
relative bound) and matmul-free models (full mode against eager float32, bound 1e-8). Each range holds the
seeds that flagged a defect earlier (they must stay clean) and a run of fresh ones. A flagged seed fails the
test and its message gives the replay command. About one minute in total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fuzz_full as FZ  # noqa: E402

# seeds that flagged a defect before it was fixed: a mutated input or buffer read after the update, a fully masked
# additive row, a scalar through a narrowing cast, a value and its views leaving a region, outputs with other strides
# than eager's, two regions in one graph, a head-major root read through the views of an earlier region, a region that
# reads the value of a region that comes later in the graph (401284: the order of the regions);
# 83933 (eager's output is exactly zero), 85367, 93374 (outputs that are rounding noise), 122772 (an output tiny against the
# scale of the values) and 240091 (strides that PyTorch's compiled graph gives to every backend) were judge defects;
# 121070 .. 121158 return a value together with a view or a copy of it (the storage sharing of the outputs);
# 600xxx, 500030 .. 500128: the models that carry a new generator tag: a fully masked row of finite values (mask:add_pad_finfo,
# mask:add_pad_1e9), expanded keys or values that also leave the model (gqa:exp_out), a buffer or input that is changed
# in place after its last read and returned (bufmut_live), a contiguous head-major copy of a region value that is
# returned (out:hm_copy: 500030, 500122, 500128)
RANDOM_FLAGGED = [30000, 30105, 30218, 30222, 30285, 30453, 30455, 30492, 30553, 1030226, 1030354, 1030645, 1030677,
                  70104, 70140, 70174, 70252, 70287, 1070023, 1070059, 1070166, 1070436, 1070502, 401284,
                  600010, 600058, 600011, 600032, 600013, 600025, 600064, 600002, 600018, 600076,
                  500030, 500122, 500128]
RANDOM_RANGE = range(40000, 40030)
ELEM_FLAGGED = [50089, 50139, 50141, 50161, 50179, 50235, 50275, 50327, 50330, 50356, 50449, 50524, 50525, 50661, 50702,
                50766, 50803, 50898, 50962, 51016, 51035, 51056, 51111, 51136, 51198, 51254, 51264, 51285, 51426, 51518,
                51558, 51561, 51572, 51680, 51695, 51699, 51711, 51801, 51927, 51955, 51969, 51979, 51995, 52109, 52164,
                52189, 52216, 52295, 52350, 52379, 52402, 52414, 52597, 52614, 52624, 52710, 52720, 52738, 52741, 52774,
                52795, 52827, 52926, 52929, 52936, 53058, 80456, 82611, 83933, 85367, 93374,
                121070, 121083, 121136, 121158, 122772, 141663, 240091]
ELEM_RANGE = range(60000, 60150)
# A judge that has gone blind shows as a case count: at most this share of the fresh cases may have an unstable output
# (its values are not judged) or an output that passed by absolute error only (its relative error is not judged)
MAX_UNSTABLE = 0.05
MAX_LOW_ENERGY = 0.05


def _run(kind: str, seeds) -> list:
    bad, recs = [], []
    for seed in seeds:
        rec, flags, _ = FZ.check_seed(seed, kind)
        recs.append(rec)
        if flags:
            bad.append(f"seed {seed} ({kind}): {flags}\n    replay: {FZ.replay_command(seed, kind)}")
    assert not bad, "the fuzzer flagged:\n  " + "\n  ".join(bad)
    return recs


def _shares(recs: list) -> tuple:
    n = max(len(recs), 1)
    return (sum(any(r.get("unstable") or []) for r in recs) / n, sum(bool(r.get("low_energy")) for r in recs) / n)


def test_fuzzer_random_models():
    _run("random", RANDOM_FLAGGED)
    unstable, low = _shares(_run("random", RANDOM_RANGE))
    assert unstable <= MAX_UNSTABLE and low <= MAX_LOW_ENERGY, (unstable, low)


def test_fuzzer_matmul_free_models():
    _run("elem", ELEM_FLAGGED)
    unstable, low = _shares(_run("elem", ELEM_RANGE))
    assert unstable <= MAX_UNSTABLE and low <= MAX_LOW_ENERGY, (unstable, low)


def _out(nmse, anmse=0.0, share=None, **kw):
    """The record of one output as ``compare`` makes it (shape and dtype agree)."""
    d = {"shape_ok": True, "dtype_ok": True, "nmse": nmse, "anmse": anmse, "nonfinite_mismatch": 0, "shape": [1], "dtype": "f",
         "ref_shape": [1], "ref_dtype": "f", "scale": 1.0, "rms_err": 0.0, "rms": 1.0}
    if share is not None:
        d["share"] = share
    d.update(kw)
    return d


def _rec(full, mm=None, attn=False):
    return {"mm": {"calls": [mm if mm is not None else [_out(1e-5)]]},
            "full": {"calls": [full], "regions": [{"ops": {"attn": 1} if attn else {"mul_mat": 1}}]}}


def _kinds(flags):
    return [k for k, _ in flags]


def test_the_judge_flags_a_wrong_result_and_a_crash():
    """The judge itself: a full-mode result far from eager's, and a crash where matmul mode runs."""
    rec = _rec([_out(0.2, 0.2)])
    assert _kinds(FZ.judge_random(rec)) == ["nmse"]
    rec["full"]["calls"] = [{"error": "IndexError: tuple index out of range"}]
    assert _kinds(FZ.judge_random(rec)) == ["crash_full"]
    rec["full"]["calls"] = [[_out(1e-5)]]
    assert FZ.judge_random(rec) == []
    elem = lambda *v: {"outs": [_out(x, x) for x in v], "tags": []}  # noqa: E731
    assert _kinds(FZ.judge_elem(elem(3e-8))) == ["nmse"] and FZ.judge_elem(elem(1e-9, None)) == []


def test_a_nan_and_a_non_finite_mismatch_are_flags():
    """A NaN is not <= any bound; a value that is finite where eager's is not (and the reverse) is a flag; a
    non-finite reference value alone is not, and the finite elements are still judged."""
    import torch

    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    nan = torch.tensor([1.0, float("nan"), 3.0, 4.0])
    for kind, rec in (("elem", {"outs": FZ.compare([nan], [ref]), "tags": []}),
                      ("random", _rec(FZ.compare([nan], [ref])))):
        judge = FZ.judge_elem if kind == "elem" else FZ.judge_random
        assert "nmse" in _kinds(judge(rec)) and "nonfinite" in _kinds(judge(rec)), kind
    # eager overflows (inf), full mode is finite: the narrowing-cast defect
    inf_ref = torch.tensor([1.0, float("inf"), 3.0, 4.0])
    fin = torch.tensor([1.0, 65504.0, 3.0, 4.0])
    d = FZ.compare([fin], [inf_ref])[0]
    assert d["nonfinite_mismatch"] == 1 and d["nmse"] == 0.0 and d["ref_finite"] is False
    assert _kinds(FZ.judge_elem({"outs": [d], "tags": []})) == ["nonfinite"]
    # both overflow in the same place: nothing to flag; a wrong finite element next to it is flagged
    same = FZ.compare([inf_ref.clone()], [inf_ref])
    assert FZ.judge_elem({"outs": same, "tags": []}) == []
    wrong = FZ.compare([torch.tensor([1.0, float("inf"), 3.5, 4.0])], [inf_ref])
    assert _kinds(FZ.judge_elem({"outs": wrong, "tags": []})) == ["nmse"]
    # matmul mode that is not finite itself: full mode is not judged against it
    mm = [_out(float("nan"), float("nan"))]
    assert FZ.judge_random(_rec([_out(float("nan"), float("nan"))], mm)) == []


def test_the_probe_keeps_the_non_finite_values_of_a_mask():
    """The one-ulp probe moves finite values only: the -inf of an additive mask stays, so a fully masked row stays
    masked and the model is not called unstable for it."""
    import torch

    gg = torch.Generator().manual_seed(2)
    x = torch.randn(1, 4, 6, generator=gg)
    mask = torch.zeros(1, 4, 6)
    mask[0, 1] = float("-inf")                      # a fully masked row

    def masked_softmax(t, m):
        return torch.softmax(t + m, -1).nan_to_num(0.0)

    ref = FZ.flat_outs(masked_softmax(x, mask))
    ulp = FZ.one_ulp_nmse(masked_softmax, lambda: [x.clone(), mask.clone()], ref, 4)
    assert ulp is not None and ulp[0] < 1e-10, ulp
    assert FZ.unstable_outputs(ulp, 1, FZ.ELEM_BOUND) == [False]


def test_the_probe_ignores_a_non_finite_reference_and_sees_a_changed_pattern():
    import torch

    x = torch.linspace(0.5, 3.0, 8)[None]

    def overflow(t):                                # eager is inf in one place, finite elsewhere
        return torch.cat([t * 2.0, torch.full((1, 1), float("inf"))], -1)

    ref = FZ.flat_outs(overflow(x))
    ulp = FZ.one_ulp_nmse(overflow, lambda: [x.clone()], ref, 1)
    assert ulp[0] < 1e-10                           # the inf alone makes nothing unstable

    def blow_up(t):                                 # the finite / non-finite pattern follows the input
        return torch.where(t > 1.0, torch.full_like(t, float("inf")), t)

    edge = torch.tensor([[0.5, 1.0, 1.5]])          # 1.0 is on the edge: one ulp up changes the pattern
    ulp = FZ.one_ulp_nmse(blow_up, lambda: [edge.clone()], FZ.flat_outs(blow_up(edge)), 0)
    changed = [FZ.one_ulp_nmse(blow_up, lambda: [edge.clone()], FZ.flat_outs(blow_up(edge)), s) for s in range(8)]
    assert any(u is not None and u[0] == float("inf") for u in changed + [ulp])


def test_unstable_is_per_output():
    """An unstable output is not judged on values; the stable output next to it, and the shapes, are."""
    import torch

    x = 0.1 + 0.8 * torch.rand(1, 4, 8, generator=torch.Generator().manual_seed(5))

    def two(t):
        c = t
        for _ in range(60):                         # chaotic: a one-ulp change grows to order one
            c = 3.9 * c * (1.0 - c)
        return t * 2.0, c

    ref = FZ.flat_outs(two(x))
    ulp = FZ.one_ulp_nmse(lambda t: two(t), lambda: [x.clone()], ref, 3)
    assert FZ.unstable_outputs(ulp, 2, FZ.ELEM_BOUND) == [False, True], ulp
    rec = {"outs": [_out(3e-2, 3e-2), _out(3e-2, 3e-2)], "tags": [], "unstable": [False, True]}
    assert [f for f in FZ.judge_elem(rec)] == [("nmse", FZ.judge_elem(rec)[0][1])] and "out 0" in FZ.judge_elem(rec)[0][1]
    rec["outs"][1] = _out(3e-2, 3e-2, shape_ok=False, shape=[2])
    assert _kinds(FZ.judge_elem(rec)) == ["nmse", "shape_dtype"]
    rec = _rec([_out(0.2, 0.2), _out(0.2, 0.2)], mm=[_out(1e-5), _out(1e-5)])
    rec["unstable"] = [True, False]
    fl = FZ.judge_random(rec)
    assert _kinds(fl) == ["nmse"] and "out 1" in fl[0][1]


def test_an_output_of_low_energy_passes_by_absolute_error():
    """An output whose relative NMSE fails passes when its error is within the absolute tolerance
    (LOW_ENERGY x the largest value of the output and of the inputs) and eager's float32 result is itself
    inaccurate (here: without limit, so that the absolute tolerance alone decides); the case records it."""
    import torch

    def elem(outs):
        return {"outs": outs, "tags": [], "cond": [float("inf")] * len(outs)}

    big = torch.full((8,), 4.0)
    ref = torch.full((8,), 1e-5)                    # rms 1e-5 against values of order 4
    err = 3e-8 * 4.0 * 0.1                          # a tenth of the tolerance sqrt(bound) x LOW_ENERGY x scale
    ok = FZ.compare([ref + err, big], [ref, big], in_scale=4.0)
    rec = elem(ok)
    assert FZ.judge_elem(rec) == [] and rec["low_energy"] == [0]
    assert ok[0]["nmse"] > FZ.ELEM_BOUND            # the relative error alone would flag it
    bad = FZ.compare([ref + 100 * err, big], [ref, big], in_scale=4.0)
    rec = elem(bad)
    assert _kinds(FZ.judge_elem(rec)) == ["nmse"] and rec["low_energy"] == []
    # the scale is the output's own and the inputs': another large output does not weaken the check
    alone = FZ.compare([ref + 100 * err], [ref], in_scale=0.0)
    assert _kinds(FZ.judge_elem(elem(alone))) == ["nmse"]
    beside = FZ.compare([ref + 1e-8, big], [ref, big], in_scale=0.0)       # 1e-8 is far above 3e-13 and far below 1.2e-7
    assert _kinds(FZ.judge_elem(elem(beside))) == ["nmse"]
    assert FZ.judge_elem(elem(FZ.compare([ref + 1e-8, big], [ref, big], in_scale=4.0))) == []
    # an output that is exactly zero with all-zero inputs must match exactly
    z = torch.zeros(8)
    assert FZ.judge_elem(elem(FZ.compare([z], [z]))) == []
    assert _kinds(FZ.judge_elem(elem(FZ.compare([z + 1e-30], [z])))) == ["nmse"]
    # the ratio test of the random generator has the same construction on both sides
    rec = _rec([_out(0.05, 10.0)], [_out(1e-3, 0.5)])
    assert FZ.judge_random(rec) == [] and rec["low_energy"] == [[0, 0]]          # 10 <= 20 x 0.5
    rec = _rec([_out(0.05, 11.0)], [_out(1e-3, 0.5)])
    assert _kinds(FZ.judge_random(rec)) == ["nmse"]
    rec = _rec([_out(0.21, 10.0)], [_out(1e-3, 0.5)])                            # above 10 x the limit of 20 x 1e-3
    assert _kinds(FZ.judge_random(rec)) == ["nmse"]


def test_the_judge_handles_scale_and_sharing():
    import torch

    z, big = torch.zeros(8), torch.full((8,), 3.0)
    assert FZ.input_scale([big, torch.tensor([0.0, float("-inf")]), 2.5]) == 3.0     # a mask does not count
    e = FZ.errors(torch.full((8,), 0.5), z, 3.0)
    assert e["nmse"] == float("inf") and e["scale"] == 3.0
    # the storage sharing of full mode must be matmul mode's
    assert _kinds(FZ.judge_random(_rec([_out(1e-5, share=[0, 1])], [_out(1e-5, share=[0, 0])]))) == ["sharing"]
    assert FZ.judge_random(_rec([_out(1e-5, share=[0, 0])], [_out(1e-5, share=[0, 0])])) == []
    assert _kinds(FZ.judge_elem({"outs": [_out(1e-9)], "tags": [], "share": [0, 1], "share_ref": [0, 0]})) == ["sharing"]
    # a dtype or a shape that differs from eager's, and an output count that differs, are flags
    assert _kinds(FZ.judge_elem({"outs": [_out(1e-9, dtype_ok=False)], "tags": []})) == ["shape_dtype"]
    assert _kinds(FZ.judge_elem({"outs": FZ.compare([torch.zeros(2), torch.zeros(2)], [torch.zeros(2)]), "tags": []})) == ["outputs"]


def test_a_root_read_through_the_views_of_an_earlier_region():
    """Seed 70140 had a region that read the head-major result of an attention of an earlier region (its root node)
    through a transpose and a reshape; the root is rebuilt from the rows. Full mode must keep every graph in regions
    (strict raises otherwise) and match eager."""
    import torch
    import tesseract_torch as tt
    from fuzz_gen import gen

    src, _ = gen(70140)
    ns: dict = {}
    exec(compile(src, "<seed 70140>", "exec"), ns)
    m = ns["M"]().eval()
    args = ns["inputs"](9, 0)
    c = tt.compile(m, mode="full", strict=True, verify=True)
    first = len(tt.regions())
    with torch.no_grad():
        ys, refs = c(*args), m(*args)
    assert len(tt.regions()) - first >= 1
    for y, ref in zip(ys, refs):
        assert y.shape == ref.shape and FZ.nmse(y, ref) < 5e-3


def test_the_one_ulp_probe_tells_a_stable_model_from_an_unstable_one():
    import torch

    x = 0.1 + 0.8 * torch.rand(1, 4, 8, generator=torch.Generator().manual_seed(5))

    def logistic(t):                     # a chaotic map: a one-ulp change grows to order one
        for _ in range(60):
            t = 3.9 * t * (1.0 - t)
        return t

    stable, unstable = (lambda t: t * 2.0), logistic
    for fn, want_big in ((stable, False), (unstable, True)):
        ref = FZ.flat_outs(fn(x))
        worst = FZ.one_ulp_nmse(fn, lambda: [x.clone()], ref, 3)[0]
        assert (worst > FZ.ELEM_BOUND) == want_big, worst


def test_a_small_stable_output_is_judged_by_its_relative_error():
    """An output whose rms is far below 3e-4 x the scale of the inputs has a loose absolute tolerance. It passes by
    absolute error only as far as eager's own float32 result is inaccurate (``cond``: eager float32 against eager
    float64): a defect on an output that float32 computes exactly is flagged by its relative NMSE."""
    import torch

    ref = torch.full((16,), 6e-13)
    scale = 5.4
    for wrong in (ref * 2.0, torch.zeros(16)):
        rec = {"outs": FZ.compare([wrong, ref], [ref, ref], in_scale=scale), "tags": [], "cond": [1e-14, 1e-14]}
        fl = FZ.judge_elem(rec)
        assert _kinds(fl) == ["nmse"] and "out 0" in fl[0][1] and rec["low_energy"] == [], fl
    near = FZ.compare([ref * (1.0 + 2e-3)], [ref], in_scale=scale)      # a relative error of 2e-3: NMSE 4e-6
    assert 3.9e-6 < near[0]["nmse"] < 4.1e-6

    def judged(cond):
        rec = {"outs": near, "tags": []}
        if cond != "absent":
            rec["cond"] = [cond]
        return _kinds(FZ.judge_elem(rec)), rec["low_energy"]

    assert judged(1e-14) == (["nmse"], [])                   # float32 is exact here: the error is a defect
    assert judged(0.0) == (["nmse"], [])
    assert judged(8e-8) == ([], [0])                         # eager itself is off by a fiftieth of that: rounding
    assert judged(2e-8) == (["nmse"], [])                    # 200 times eager's own error: not rounding
    assert judged(float("inf")) == ([], [0])
    # an output that is rounding noise in eager (float64 says so) is still held to a relative NMSE of 1e-4
    for rel, want in ((7e-3, ([], [0])), (1.4e-2, (["nmse"], []))):                 # NMSE 0.5e-4 and 2e-4
        rec = {"outs": FZ.compare([ref * (1.0 + rel)], [ref], in_scale=scale), "tags": [], "cond": [float("inf")]}
        assert (_kinds(FZ.judge_elem(rec)), rec["low_energy"]) == want, rel
    assert judged(None) == (["nmse"], []) and judged("absent") == (["nmse"], [])    # no float64 run: 10 x the bound
    tight = FZ.compare([ref * (1.0 + 2e-4)], [ref], in_scale=scale)     # NMSE 4e-8: within 10 x the bound
    rec = {"outs": tight, "tags": [], "cond": [None]}
    assert FZ.judge_elem(rec) == [] and rec["low_energy"] == [0]
    # an eager output that is exactly zero has no relative error: the absolute one decides
    z = torch.zeros(16)
    rec = {"outs": FZ.compare([z + 4e-10], [z], in_scale=scale), "tags": [], "cond": [1.0]}
    assert FZ.judge_elem(rec) == [] and rec["low_energy"] == [0]
    assert _kinds(FZ.judge_elem({"outs": FZ.compare([z + 4e-6], [z], in_scale=scale), "tags": [], "cond": [1.0]})) == ["nmse"]
    # the random generator: matmul mode is exact, full mode doubles the output
    rec = _rec([_out(1.0, 1e-13, rms=6e-13)], [_out(0.0, 0.0, rms=6e-13)])
    assert _kinds(FZ.judge_random(rec)) == ["nmse"]


def test_the_float64_run_tells_a_cancelling_output_from_a_small_exact_one():
    """``conditioning``: eager float32 against eager float64. 1 + tanh(u) far in the negative tail is rounding noise
    in float32 (the sum cancels); a value that is only scaled down is exact. A cast inside the model stays."""
    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.full((8,), 0.5))

        def forward(self, x):
            u = -7.0 - x.abs() * self.w
            return (1.0 + torch.tanh(u)) * x, x * 1e-9 * self.w, (x * self.w).half().float(), (x > 0).long()

    m = M().eval()
    x = torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = FZ.flat_outs(m(x))
    r64 = FZ.float64_outputs(m, lambda: [x.clone()])
    assert r64 is not None and r64[0].dtype == torch.float64 and r64[2].dtype == torch.float32
    assert m.w.dtype == torch.float32                           # the model itself is not changed
    c = FZ.conditioning(ref, r64)
    assert c[0] > 1e-6 and c[1] < 1e-12 and c[2] is not None and c[3] is None, c
    assert FZ.conditioning(ref, None) == [None] * 4 and FZ.conditioning(ref, r64[:2]) == [None] * 4
    z = torch.zeros(4)
    assert FZ.conditioning([z, z], [z.double(), z.double() + 1e-20]) == [0.0, 1.0]
    assert FZ.conditioning([z + 1.0], [z.double()]) == [float("inf")]
    assert FZ.float64_outputs(lambda t: t, lambda: [x]) is None        # not a module: no float64 run


def test_a_non_finite_mismatch_of_an_unstable_output_is_a_flag():
    """The values of an unstable output are not judged, but a NaN where eager has a finite value is not rounding noise.
    When the probe moved the finite / non-finite pattern itself, the pattern is not stable either."""
    import torch

    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    nan = torch.tensor([1.0, float("nan"), 3.0, 4.0])
    for kind in ("elem", "random"):
        outs = FZ.compare([nan, ref], [ref, ref])
        rec = {"outs": outs, "tags": [], "unstable": [True, False], "ulp_nmse": [1.0, 0.0]} if kind == "elem" else \
            dict(_rec(outs), unstable=[True, False], ulp_nmse=[1.0, 0.0])
        judge = FZ.judge_elem if kind == "elem" else FZ.judge_random
        fl = judge(rec)
        assert _kinds(fl) == ["nonfinite"] and "out 0" in fl[0][1], (kind, fl)
        rec["ulp_nmse"] = [float("inf"), 0.0]                         # the probe moved the pattern: not judged
        assert judge(rec) == [], kind
        rec["unstable"] = [False, False]
        rec["ulp_nmse"] = [0.0, 0.0]
        assert "nonfinite" in _kinds(judge(rec))
    # an unstable output with a finite error that is large is still not judged
    rec = {"outs": FZ.compare([ref * 3.0], [ref]), "tags": [], "unstable": [True], "ulp_nmse": [1.0]}
    assert FZ.judge_elem(rec) == []


def test_the_strides_of_an_output_are_judged():
    """The contiguous head-major copy of eager has other strides than a view of the region's rows; a later .view()
    of that output fails. Dimensions of size 1 have no stride to judge."""
    import torch

    base = torch.randn(1, 5, 4, 8)
    copy = base.transpose(1, 2).contiguous()                       # [1, 4, 5, 8], contiguous
    view = base.transpose(1, 2)                                    # same values, strides of the rows
    rec = {"outs": FZ.compare([view], [copy]), "tags": []}
    fl = FZ.judge_elem(rec)
    assert _kinds(fl) == ["strides"], fl
    assert FZ.judge_elem({"outs": FZ.compare([copy.clone()], [copy]), "tags": []}) == []
    one = torch.randn(1, 6)
    assert FZ.judge_elem({"outs": FZ.compare([one.as_strided((1, 6), (100, 1))], [one]), "tags": []}) == []
    # random generator: matmul mode's layout is the one full mode must have when it differs from eager's
    d = FZ.compare([view], [copy])[0]
    assert _kinds(FZ.judge_random(dict(_rec([d], [d])))) == []
    assert _kinds(FZ.judge_random(dict(_rec([d], [FZ.compare([copy], [copy])[0]])))) == ["strides"]
    # matmul-free generator: the same, with the strides of the model compiled in matmul mode (``mm_stride``)
    assert FZ.judge_elem({"outs": [dict(d, mm_stride=d["stride"])], "tags": []}) == []
    assert _kinds(FZ.judge_elem({"outs": [dict(d, mm_stride=d["ref_stride"])], "tags": []})) == ["strides"]


def test_a_layout_of_the_compiled_graph_is_not_a_stride_flag():
    """Seed 240091 of the matmul-free generator: PyTorch's compiled graph returns the output with other strides than
    eager for every backend. The run records matmul mode's strides and the judge does not flag them."""
    rec, flags, _ = FZ.check_seed(240091, "elem")
    d = rec["outs"][0]
    assert d["stride"] != d["ref_stride"] and d["mm_stride"] == d["stride"] and flags == [], (d, flags)


def test_a_finite_mask_value_is_not_an_activation():
    import torch

    x = torch.full((4,), 2.0)
    fmin = torch.full((4, 4), torch.finfo(torch.float32).min)
    assert FZ.input_scale([x, fmin, torch.full((4, 4), -1e9)]) == 2.0

#!/usr/bin/env python3
"""fuzz_full.py - differential fuzzer of full mode (whole-graph lowering) of the PyTorch backend.

    build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --start 0 --count 50          # seeds 0 .. 49
    build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --gen elem --start 0 --count 50
    build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --replay SEED [--gen elem] [-v]

What it compares. fuzz_gen.py draws a small random model from a seed (linear layers, residuals,
RMS norms, activations, GLU, scalars, rows, views, RoPE, SDPA with masks, CPU-only ops, in-place
updates, a block called twice, ...). For each seed the model runs on the simulator (TSR_DEVICE, sim by
default), always with TSR_TORCH_VERIFY on (every region call is also run by the reference executor and
compared bit for bit), in three ways:

  eager    float32 on the CPU, the reference;
  matmul   torch.compile(backend="tesseract", mode="matmul"): matmuls on the device, the rest on the CPU;
  full     the same with mode="full": regions of ops, one launch per region.

Each runs the same four calls: T1, T1 with another input and another Python float argument, T2 (a new
sequence length: a recompile, or a symbolic one with dynamic=True), and T1 again after some weights,
norm weights or buffers were changed in place. The state of the model (buffers that forward updates) is
restored before each mode. dynamic=True is drawn in 20 % of the cases, torch.inference_mode in a third.

Judge (flags, per output of each call). Every output of both generators gets the same record (``compare``): its
shape and dtype, the count of non-finite mismatches (finite in full mode and not in eager, or the reverse), the plain
relative NMSE over the elements that are finite in eager, and the same error in units of an absolute tolerance.
  gen random   full mode is flagged when its graph lost full mode with a warning ("forming regions failed",
               "partitioning failed"); when an output's NMSE(full, eager) is above max(20 x NMSE(matmul, eager), floor), the
               floor being 2e-4 (5e-3 when a region has attention: 8-bit K, V and Q); when it raises where matmul
               mode does not; when shape, dtype or the number of outputs differ; or when it is non-finite where
               matmul mode is finite, or finite where eager is not. A model that eager itself cannot run is a
               generator defect and is reported as "eager_error", not flagged.
  gen elem     matmul-free models: no 8-bit weights or activations are involved, so full mode must
               match eager to float32 accuracy: NMSE(full, eager) > 1e-8 is flagged (1e-6 for a model with a bf16 or half
               island, where a last-bit difference of the device can move a value across a rounding boundary), as is
               any error, a dtype that differs from eager's, and a non-finite mismatch.
               This has a resolution of about 1e-8, where the random generator cannot see a semantic error
               below about 1e-4 once a matmul is in the graph.
  Both:        the storage-sharing pattern of the returned tensors (which outputs share a storage) of full mode must
               equal matmul mode's (gen random) or eager's (gen elem); a difference is flagged as "sharing".
               An NMSE that is not within its limit is flagged, a NaN included.
  Low energy   An output is first judged by its plain relative NMSE. If that fails, it passes when its absolute error
               is within the absolute tolerance: the rms error is at most sqrt(limit) x LOW_ENERGY x scale, the scale being
               the largest finite absolute value of eager's values of that output and of the float activation inputs of
               the call (a mask does not count; there is no minimum), and eager's own float32 result is no more accurate
               than that: gen elem runs the model in float64 too, and the relative NMSE of full mode may be at most
               COND_RATIO x the NMSE of eager float32 against eager float64 (a sum that cancels, a saturated tanh) and
               at most LOW_ENERGY_NMSE (10 x the bound when the float64 run is not available). An output that is small
               and exact in float32 is so judged by its relative error whatever the scale. An eager output that is
               exactly zero has no relative error: the absolute one decides (with all-zero inputs it must match
               exactly). The random generator applies the construction to both sides of the ratio test, with a relative
               NMSE of at most 10 x the limit.
               The case records these outputs in ``low_energy`` and the summary counts them.
  Strides      The strides of an output (in its dimensions of size above one) must equal eager's, or matmul mode's when that
               differs from eager's the same way (gen elem compiles the model in matmul mode when a stride differs: a
               layout that PyTorch's compiled graph gives to every backend is not full mode's); a difference is flagged
               as "strides" (a later ``.view()`` of the output fails on a layout that eager does not have).
  Unstable     An output of eager that moves by more than the bound when every finite value of the float inputs changes
               by one ulp (an ill-conditioned model, an exp chain) is "unstable": its values (its NMSE) are not judged, since no
               implementation can match them; a non-finite mismatch is, unless the probe moved the finite / non-finite
               pattern itself; every other output of the case is judged in full, and shapes, dtypes, strides, errors, lost
               full mode and storage sharing always are. The probe compares the
               elements that are finite in both runs, and calls an output unstable too when the finite / non-finite
               pattern of the moved run differs; a non-finite eager value alone makes nothing unstable. ``unstable`` and
               ``ulp_nmse`` of a record are lists with one entry per output. The summary prints how many cases had an
               unstable output. The probe of the random generator is taken on the first call and applies to the same
               output of every call.
A case that full mode runs on the CPU with the warning "runs on the CPU for a call that the device does not
take" is correct by design and is not flagged.

Replay a seed with --replay SEED (and --gen elem for the matmul-free generator); -v prints the model source.
Exit code 1 when a case is flagged. The failing seed and its replay command are printed.
A process grows by a few hundred kilobytes per case (PyTorch's compile caches, and the record of every region,
which ``tt.regions()`` keeps): run a long campaign in chunks of a few thousand seeds per process.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("TSR_TORCH_EXT_DIR", str(REPO / "build" / "torch" / "python"))      # a card build names its own
sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO / "sw" / "torch"), os.environ["TSR_TORCH_EXT_DIR"]]

import torch  # noqa: E402

import torch._dynamo  # noqa: E402,F401

import tesseract_torch as tt  # noqa: E402
from tesseract_torch import runtime as RT  # noqa: E402

from fuzz_gen import gen, gen_elem  # noqa: E402

torch._dynamo.config.cache_size_limit = 64

ATTN_FLOOR = 5e-3        # a region with attention (8-bit K, V and Q)
FREE_FLOOR = 2e-4        # matmuls only: 8-bit weights and activations
ELEM_BOUND = 1e-8        # matmul-free models: float32 accuracy
ELEM_ISLAND_BOUND = 1e-6  # with a bf16 / half island: the device's last-bit differences move a value across a rounding boundary
RATIO = 20.0             # full mode may be this many times worse than matmul mode


LOW_ENERGY = 3e-4        # absolute tolerance of an output, as a fraction of the scale of its values (see judge_output)
COND_RATIO = 100.0       # an output judged by absolute error may be this many times further from eager float32 than eager float64 is
LOW_ENERGY_NMSE = 1e-4   # and never further than this relative NMSE (an eager output of rms 0 excepted)
MASK_VALUE = 1e9         # an input with a value this large is a mask (finfo.min, -1e9), not an activation


def _finite_max(ts) -> float:
    """The largest finite absolute value of the float tensors ``ts`` (0 when there is none)."""
    m = 0.0
    for t in ts:
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
            f = t.detach().abs().double()
            f = f[torch.isfinite(f)]
            if f.numel():
                m = max(m, float(f.max()))
    return m


def input_scale(args) -> float:
    """The scale of the activation inputs of a call: the largest finite absolute value of its float tensor
    arguments. A tensor with a non-finite value or a value of 1e9 or more is a mask (0 / -inf, finfo.min, -1e9)
    and does not count."""
    return _finite_max(a for a in args if isinstance(a, torch.Tensor) and a.is_floating_point() and bool(torch.isfinite(a).all())
                       and not bool((a.abs() >= MASK_VALUE).any()))


def errors(y: torch.Tensor, ref: torch.Tensor, in_scale: float = 0.0) -> dict:
    """The errors of y against ref over the elements that are finite in ref (float64). ``nmse`` is the plain
    relative NMSE (0 for an exact match, inf for an error against an all-zero reference), ``anmse`` the mean
    squared error in units of the squared absolute tolerance (LOW_ENERGY x scale)^2, ``scale`` the largest
    finite absolute value of ref and of the activation inputs (``in_scale``). None when no element is finite."""
    y, ref = y.detach().double(), ref.detach().double()
    m = torch.isfinite(ref)
    if not bool(m.any()):
        return {"nmse": None, "anmse": None, "scale": 0.0, "rms_err": None, "rms": None}
    yy, rr = y[m], ref[m]
    mse = float(((yy - rr) ** 2).mean())
    ms = float((rr ** 2).mean())
    scale = max(float(rr.abs().max()), in_scale)
    tol2 = (LOW_ENERGY * scale) ** 2

    def ratio(num, den):
        return num / den if den > 0.0 else (0.0 if num == 0.0 else float("inf"))

    return {"nmse": ratio(mse, ms), "anmse": ratio(mse, tol2), "scale": scale, "rms_err": mse ** 0.5, "rms": ms ** 0.5}


def nmse(y: torch.Tensor, ref: torch.Tensor):
    """The plain relative NMSE of y against ref over the elements that are finite in ref (None when there is none)."""
    return errors(y, ref)["nmse"]


def float64_outputs(model, make_args, ctx=torch.no_grad):
    """Eager's outputs with the float parameters, buffers and inputs of the model in float64 (a cast that the
    model itself makes stays), or None when that does not run. The model is copied: its state is not touched."""
    try:
        m64 = copy.deepcopy(model).double()
        args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in make_args()]
        with warnings.catch_warnings(), ctx():
            warnings.simplefilter("ignore")       # PyTorch warns about a float32 value next to a float64 weight
            return flat_outs(m64(*args))
    except Exception:  # noqa: BLE001
        return None


def conditioning(ref: list, ref64) -> list:
    """Per output: how accurate eager's float32 result is: the plain relative NMSE of eager float32 (``ref``)
    against eager float64 (``ref64``) over the elements that are finite in both. 0 when they are equal, inf when
    the float64 values are all zero there and the float32 ones are not. None when it is not known (no float64
    run, another count or shape, a non-float output, no element finite in both)."""
    if ref64 is None or len(ref64) != len(ref):
        return [None] * len(ref)
    out = []
    for y, r in zip(ref, ref64):
        if not (y.is_floating_point() and r.is_floating_point()) or y.shape != r.shape:
            out.append(None)
            continue
        y, r = y.detach().double(), r.detach().double()
        m = torch.isfinite(y) & torch.isfinite(r)
        if not bool(m.any()):
            out.append(None)
            continue
        mse, ms = float(((y[m] - r[m]) ** 2).mean()), float((r[m] ** 2).mean())
        c = mse / ms if ms > 0.0 else (0.0 if mse == 0.0 else float("inf"))
        out.append(c if c == c else float("inf"))          # inf / inf: float64 values beyond the square range
    return out


def sharing(outs: list) -> list:
    """For each returned tensor, the index of the first one that has its storage."""
    ptrs = [t.untyped_storage().data_ptr() if t.numel() else -(i + 1) for i, t in enumerate(outs)]
    return [ptrs.index(p) for p in ptrs]


def strides_of(t: torch.Tensor) -> list:
    """The strides of a tensor in its dimensions of size above one (a size-1 dimension has no stride to judge)."""
    return [st for sz, st in zip(t.shape, t.stride()) if sz > 1] if t.numel() else []


def flat_outs(o) -> list:
    return [o] if isinstance(o, torch.Tensor) else list(o)


def compare(ys: list, refs: list, in_scale: float = 0.0) -> list:
    """Per output: shape, dtype, the count of non-finite mismatches (finite in one, not in the other) and the
    errors over the elements that are finite in the reference (see ``errors``)."""
    if len(ys) != len(refs):
        return [{"count_mismatch": [len(ys), len(refs)]}]
    res = []
    share = sharing(ys)
    for y, r in zip(ys, refs):
        d = {"shape": list(y.shape), "ref_shape": list(r.shape), "dtype": str(y.dtype), "ref_dtype": str(r.dtype),
             "shape_ok": tuple(y.shape) == tuple(r.shape), "dtype_ok": y.dtype == r.dtype, "share": share,
             "stride": strides_of(y), "ref_stride": strides_of(r)}
        if d["shape_ok"] and y.dtype.is_floating_point and r.dtype.is_floating_point:
            d["ref_finite"] = bool(torch.isfinite(r).all())
            d["nonfinite_mismatch"] = int((torch.isfinite(r) != torch.isfinite(y)).sum())
            d.update(errors(y, r, in_scale))
        elif d["shape_ok"]:
            d["equal"] = bool(torch.equal(y, r))
        res.append(d)
    return res


def _pattern_and_nmse(y: torch.Tensor, ref: torch.Tensor, in_scale: float):
    """(finite patterns equal, plain relative NMSE over the elements finite in both) of a moved run against the reference."""
    same = bool((torch.isfinite(y) == torch.isfinite(ref)).all())
    both = torch.isfinite(y) & torch.isfinite(ref)
    if not bool(both.any()):
        return same, 0.0
    e = errors(y[both], ref[both], in_scale)
    return same, (e["nmse"] if e["nmse"] is not None else 0.0)


def one_ulp_nmse(model, make_args, ref: list, seed: int, ctx=torch.no_grad):
    """The sensitivity of every output of eager to one ulp: one number per output, or None when the probe cannot
    be run (no float input, another count or shape). Every finite value of the float inputs moves by one ulp, up
    or down at random; a non-finite value (the -inf of a mask) stays. The caller restores the state of the model.
    The number is the plain relative NMSE of the moved run against ``ref`` over the elements that are finite in
    both (no floor: an output that is rounding noise changes by order one), and inf when the finite / non-finite
    pattern of the moved run differs from the reference's. A stable output changes by about 1e-14; one that
    amplifies rounding errors (an exp chain, a sum that cancels) by a lot. A non-finite reference value alone
    makes nothing unstable. A non-float output gets None."""
    args = make_args()
    gen = torch.Generator().manual_seed(seed + 31)
    changed = False
    for a in args:
        if isinstance(a, torch.Tensor) and a.is_floating_point() and a.numel():
            up = torch.rand(a.shape, generator=gen) < 0.5
            inf = torch.full_like(a, float("inf"))
            moved = torch.where(up, torch.nextafter(a, inf), torch.nextafter(a, -inf))
            a.copy_(torch.where(torch.isfinite(a), moved, a))
            changed = True
    if not changed:
        return None
    in_scale = input_scale(args)
    with ctx():
        ys = flat_outs(model(*args))
    if len(ys) != len(ref) or any(y.shape != r.shape for y, r in zip(ys, ref)):
        return None
    out = []
    for y, r in zip(ys, ref):
        if y.is_floating_point() and r.is_floating_point():
            same, v = _pattern_and_nmse(y, r, in_scale)
            out.append(v if same else float("inf"))
        else:
            out.append(None)
    return out


def unstable_outputs(ulp, n: int, bound: float) -> list:
    """Per output: is it unstable (its sensitivity to one ulp exceeds ``bound``)?"""
    if ulp is None:
        return [False] * n
    return [v is not None and not (v <= bound) for v in ulp]


def _is_unstable(rec: dict, k: int) -> bool:
    u = rec.get("unstable")
    if isinstance(u, list):
        return k < len(u) and bool(u[k])
    return bool(u)


def _pattern_moved(rec: dict, k: int) -> bool:
    """Did the probe move the finite / non-finite pattern of output k (its value is inf)?"""
    u = rec.get("ulp_nmse")
    return isinstance(u, list) and k < len(u) and u[k] is not None and math.isinf(u[k])


def _stride_flag(d: dict, dm: dict | None, where: str) -> list:
    """A flag when the strides differ from eager's (and from matmul mode's ``dm``, when that differs the same way)."""
    if d.get("stride") is None or d["stride"] == d.get("ref_stride"):
        return []
    if dm is not None and dm.get("stride") == d["stride"]:
        return []
    return [("strides", f"{where}: strides {d['stride']} against eager {d.get('ref_stride')}")]


def judge_output(d: dict, dm: dict | None, floor: float, ratio: float, where: str, values: bool = True, cond=None):
    """The value flags of one output and whether it passed as low energy. ``d`` is the record of full mode,
    ``dm`` matmul mode's record of the same output (None: judged against the bound ``floor`` alone).
    A non-finite mismatch (finite in one of full mode and eager, not in the other) is a flag unless matmul mode
    has one too; it is judged also with ``values`` False (an unstable output: its values are rounding noise, a
    NaN is not). When matmul mode's own NMSE is not finite, the values are not judged. The plain relative NMSE
    passes when it is at most max(ratio x matmul's, floor). If it does not, the output passes as low energy when
    the mean squared error, in units of the squared absolute tolerance ((LOW_ENERGY x scale)^2, the scale being
    the largest finite absolute value of the eager output and of the activation inputs), is within the same
    limit and the relative NMSE is at most COND_RATIO x ``cond`` (the NMSE of eager float32 against eager
    float64, see ``conditioning``: full mode may be as far from eager as eager is from the exact value) and at
    most LOW_ENERGY_NMSE, or at most 10 x the plain limit when ``cond`` is None (an eager output of rms 0 has no
    relative error: the absolute one decides). An NMSE that is not within the limit, NaN included, is a flag."""
    fl, low = [], False
    mism = d.get("nonfinite_mismatch", 0)
    if mism and not (dm and dm.get("nonfinite_mismatch")):
        fl.append(("nonfinite", f"{where}: {mism} non-finite mismatches"))
    if not values:
        return fl, low
    if d.get("nmse") is None:
        return fl, low
    if dm is not None and dm.get("nmse") is not None and not math.isfinite(dm["nmse"]):
        return fl, low            # matmul mode's own values are not finite: there is nothing to compare with

    def limit(key):
        base = dm[key] if dm is not None and dm.get(key) is not None else 0.0
        return max(ratio * base, floor)

    if d["nmse"] <= limit("nmse"):
        return fl, low
    cap = 10.0 * limit("nmse") if cond is None else min(COND_RATIO * cond, LOW_ENERGY_NMSE)
    if d["anmse"] <= limit("anmse") and (not d.get("rms") or d["nmse"] <= cap):
        return fl, True
    fl.append(("nmse", f"{where}: full {d['nmse']:.3g} (absolute {d['anmse']:.3g}, rms error {d['rms_err']:.3g}, "
                       f"rms {d['rms']:.3g}, scale {d['scale']:.3g}) matmul {(dm or {}).get('nmse')} limit {limit('nmse'):.3g}"))
    return fl, low


def _snapshot(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict(keep_vars=True).items()}


def _restore(model, st: dict) -> None:
    with torch.no_grad():
        for k, v in model.state_dict(keep_vars=True).items():
            if not torch.equal(v, st[k]):
                v.copy_(st[k])


def _warnings(wl) -> list:
    return sorted({str(w.message)[:300] for w in wl if "tesseract" in str(w.message)})


def run_random(seed: int, src: str | None = None) -> tuple[dict, str]:
    """One case of the random generator: eager, matmul mode and full mode on four calls (see the docstring)."""
    meta = {"tags": []}
    if src is None:
        src, meta = gen(seed)
    rnd = random.Random(seed * 7919 + 1)
    T1, T2 = rnd.randint(1, 24), rnd.randint(1, 24)
    if T2 == T1:
        T2 = (T1 % 24) + 1
    dynamic = rnd.random() < 0.2
    ctxname = rnd.choice(["no_grad", "no_grad", "inference_mode"])
    ctx = torch.no_grad if ctxname == "no_grad" else torch.inference_mode
    rec = {"seed": seed, "gen": "random", "T": [T1, T1, T2, T1], "dynamic": dynamic, "ctx": ctxname, "tags": meta["tags"]}
    ns: dict = {}
    exec(compile(src, f"<case {seed}>", "exec"), ns)
    model = ns["M"]().eval()
    inputs = ns["inputs"]
    calls = [(T1, 0), (T1, 1), (T2, 2)]
    s0 = _snapshot(model)
    scales = []

    def eager(T, s):
        args = inputs(T, s)
        scales.append(input_scale(args))
        return flat_outs(model(*args))

    try:
        with ctx():
            refs = [eager(T, s) for T, s in calls]
    except Exception as e:  # noqa: BLE001  (a generator defect: eager does not run it)
        rec["eager_error"] = f"{type(e).__name__}: {e}"[:300]
        return rec, src
    _restore(model, s0)
    try:
        ulp = one_ulp_nmse(model, lambda: inputs(T1, 0), refs[0], seed, ctx)
    except Exception:  # noqa: BLE001
        ulp = None
    _restore(model, s0)
    rec["ulp_nmse"] = ulp if ulp is not None else [None] * len(refs[0])
    rec["unstable"] = unstable_outputs(ulp, len(refs[0]), FREE_FLOOR)
    kw = {"dynamic": True} if dynamic else {}
    out = {}
    for mode in ("matmul", "full"):
        torch._dynamo.reset()
        tt.release_weights()
        tt.reset_counters()
        RT._warned.clear()
        first = len(tt.regions())
        res: dict = {"calls": []}
        _restore(model, s0)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            c = tt.compile(model, mode=mode, verify=True, **kw)
            for i, (T, s) in enumerate(calls):
                try:
                    with ctx():
                        res["calls"].append(compare(flat_outs(c(*inputs(T, s))), refs[i], scales[i]))
                except Exception as e:  # noqa: BLE001
                    res["calls"].append({"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]})
            # weights, norm weights and buffers changed in place, the same compiled function
            pool = [p for p in model.parameters() if p.dim() >= 1]
            pool += [b for b in model.buffers() if b.dim() >= 1 and b.dtype == torch.float32]
            if pool:
                sel = random.Random(seed + 17).sample(range(len(pool)), min(len(pool), 3))
                with torch.no_grad():
                    for j in sel:
                        pool[j].mul_(1.1).add_(0.05)
                try:
                    st = _snapshot(model)
                    args4 = inputs(T1, 3)
                    scale4 = input_scale(args4)
                    with ctx():
                        ref4 = flat_outs(model(*args4))
                    _restore(model, st)
                    with ctx():
                        res["calls"].append(compare(flat_outs(c(*inputs(T1, 3))), ref4, scale4))
                except Exception as e:  # noqa: BLE001
                    res["calls"].append({"error": f"{type(e).__name__}: {e}"[:600]})
            res["warnings"] = _warnings(wl)
        cn = tt.counters()
        res["counters"] = {k: cn.get(k, 0) for k in ("graphs", "regions", "region_calls", "region_cpu_fallback_calls",
                                                       "verified_regions", "nodes_device", "nodes_cpu", "device_calls",
                                                       "graphs_eager")}
        res["regions"] = [{"name": r.name, "ops": r.region.op_counts(), "calls": r.calls, "cpu_calls": r.cpu_calls}
                          for r in tt.regions()[first:]]
        out[mode] = res
        del c
    rec["mm"], rec["full"] = out["matmul"], out["full"]
    return rec, src


def _lost(warns: list) -> list:
    """A graph that silently lost full mode (the regions could not be formed, or the graph could not be partitioned)."""
    return [("lost_full_mode", w[:200]) for w in warns if "forming regions failed" in w or "partitioning failed" in w]


def judge_random(rec: dict) -> list:
    """The flags (kind, detail) of a case of the random generator. Sets ``rec["low_energy"]``: the [call, output]
    pairs that passed by absolute error only."""
    fl = []
    rec["low_energy"] = []
    if "eager_error" in rec or "mm" not in rec:
        return fl
    mm, fu = rec["mm"], rec["full"]
    fl += _lost(fu.get("warnings", []))
    has_attn = any("attn" in r["ops"] for r in fu["regions"])
    floor = ATTN_FLOOR if has_attn else FREE_FLOOR
    for i, cf in enumerate(fu["calls"]):
        cm = mm["calls"][i] if i < len(mm["calls"]) else (mm["calls"][0] if mm["calls"] else None)
        if isinstance(cf, dict) and "error" in cf:
            if cm is None or (isinstance(cm, dict) and "error" in cm):
                continue          # matmul mode fails too: not a full-mode defect
            fl.append(("crash_full", f"call {i}: {cf['error'][:200]}"))
            continue
        if isinstance(cm, dict):
            cm = None
        for k, d in enumerate(cf):
            if "count_mismatch" in d:
                fl.append(("outputs", str(d)))
                continue
            if not d["shape_ok"] or not d["dtype_ok"]:
                fl.append(("shape_dtype", f"call {i} out {k}: {d['shape']} {d['dtype']} vs {d['ref_shape']} {d['ref_dtype']}"))
                continue
            dm = cm[k] if cm and k < len(cm) else None
            if dm is not None and dm.get("share") not in (None, d.get("share")) and d.get("share") is not None:
                fl.append(("sharing", f"call {i}: full {d['share']} matmul {dm['share']}"))
            fl += _stride_flag(d, dm, f"call {i} out {k}")
            if _is_unstable(rec, k):
                # rounding noise of this output is not a value to judge; a non-finite value that eager has not is
                if not _pattern_moved(rec, k):
                    f, _ = judge_output(d, dm, floor, RATIO, f"call {i} out {k} (attention {has_attn})", values=False)
                    fl += f
                continue
            if d.get("equal") is False:
                fl.append(("value", f"call {i} out {k}: differs from eager"))
            f, low = judge_output(d, dm, floor, RATIO, f"call {i} out {k} (attention {has_attn})")
            fl += f
            if low:
                rec["low_energy"].append([i, k])
    return fl


def run_elem(seed: int, src: str | None = None) -> tuple[dict, str]:
    """One case of the matmul-free generator: full mode against eager, one call."""
    meta = {"tags": []}
    if src is None:
        src, meta = gen_elem(seed)
    ns: dict = {}
    exec(compile(src, f"<elem {seed}>", "exec"), ns)
    m = ns["M"]().eval()
    st = {k: v.detach().clone() for k, v in m.state_dict().items()}
    T = random.Random(seed).randint(1, 24)
    rec = {"seed": seed, "gen": "elem", "T": T, "tags": meta["tags"]}
    with torch.no_grad():
        try:
            ref = flat_outs(m(*ns["inputs"](T, 0)))
        except Exception as e:  # noqa: BLE001
            rec["eager_error"] = f"{type(e).__name__}: {e}"[:300]
            return rec, src
    rec["share_ref"] = sharing(ref)
    m.load_state_dict(st)
    try:
        ulp = one_ulp_nmse(m, lambda: ns["inputs"](T, 0), ref, seed)
    except Exception:  # noqa: BLE001
        ulp = None
    m.load_state_dict(st)         # eager may have updated a buffer; inputs are made again (it may have changed them)
    bound = ELEM_ISLAND_BOUND if "dtype_island" in rec["tags"] else ELEM_BOUND
    rec["ulp_nmse"] = ulp if ulp is not None else [None] * len(ref)
    rec["unstable"] = unstable_outputs(ulp, len(ref), bound)
    rec["cond"] = conditioning(ref, float64_outputs(m, lambda: ns["inputs"](T, 0)))
    args = ns["inputs"](T, 0)
    in_scale = input_scale(args)
    torch._dynamo.reset()
    tt.release_weights()
    RT._warned.clear()
    first = len(tt.regions())
    with warnings.catch_warnings(record=True) as wl, torch.no_grad():
        warnings.simplefilter("always")
        try:
            y = flat_outs(tt.compile(m, mode="full", verify=True)(*args))
            rec["share"] = sharing(y)
            rec["outs"] = compare(y, ref, in_scale)
            rec["nmse"] = [d.get("nmse") for d in rec["outs"]]
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["warnings"] = _warnings(wl)
    rec["regions"] = [(r.region.op_counts(), r.calls, r.cpu_calls) for r in tt.regions()[first:]]
    if any(d.get("stride") != d.get("ref_stride") for d in rec.get("outs", [])):
        # a layout that eager does not have: PyTorch's compiled graph gives it to every backend (matmul mode, which
        # lowers nothing of a matmul-free model, has it too), or it is full mode's own
        try:
            m.load_state_dict(st)
            torch._dynamo.reset()
            tt.release_weights()
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter("ignore")
                ym = flat_outs(tt.compile(m, mode="matmul", verify=True)(*ns["inputs"](T, 0)))
            if len(ym) == len(rec["outs"]):
                for d, t in zip(rec["outs"], ym):
                    d["mm_stride"] = strides_of(t)
        except Exception:  # noqa: BLE001  (no compiled reference: the difference stays a flag)
            pass
    return rec, src


def judge_elem(rec: dict) -> list:
    """The flags of a case of the matmul-free generator. Sets ``rec["low_energy"]``: the outputs that passed by
    absolute error only."""
    fl = []
    rec["low_energy"] = []
    if "eager_error" in rec:
        return fl
    fl += _lost(rec.get("warnings", []))
    if "error" in rec:
        fl.append(("crash_full", rec["error"]))
    bound = ELEM_ISLAND_BOUND if "dtype_island" in rec.get("tags", []) else ELEM_BOUND
    if rec.get("share") is not None and rec.get("share_ref") is not None and rec["share"] != rec["share_ref"]:
        fl.append(("sharing", f"full {rec['share']} eager {rec['share_ref']}"))
    for k, d in enumerate(rec.get("outs", [])):
        if "count_mismatch" in d:
            fl.append(("outputs", str(d)))
            continue
        if not d["shape_ok"] or not d["dtype_ok"]:
            fl.append(("shape_dtype", f"out {k}: {d['shape']} {d['dtype']} vs {d['ref_shape']} {d['ref_dtype']}"))
            continue
        fl += _stride_flag(d, {"stride": d["mm_stride"]} if "mm_stride" in d else None, f"out {k}")
        if _is_unstable(rec, k):
            if not _pattern_moved(rec, k):
                f, _ = judge_output(d, None, bound, 0.0, f"out {k}", values=False)
                fl += f
            continue
        if d.get("equal") is False:
            fl.append(("value", f"out {k}: differs from eager"))
        cond = rec.get("cond")
        f, low = judge_output(d, None, bound, 0.0, f"out {k}", cond=cond[k] if cond and k < len(cond) else None)
        fl += f
        if low:
            rec["low_energy"].append(k)
    return fl


GENERATORS = {"random": (gen, run_random, judge_random), "elem": (gen_elem, run_elem, judge_elem)}


def check_seed(seed: int, kind: str = "random", src: str | None = None) -> tuple[dict, list, str]:
    """Run one seed; (record, flags, model source). The harness-level failure is a flag too."""
    _, run, judge = GENERATORS[kind]
    try:
        rec, src = run(seed, src)
    except Exception as e:  # noqa: BLE001
        rec, src = {"seed": seed, "gen": kind, "harness_error": f"{type(e).__name__}: {e}"[:500],
                    "tb": traceback.format_exc()[-2000:]}, src or ""
        return rec, [("harness", rec["harness_error"])], src
    return rec, judge(rec), src


def replay_command(seed: int, kind: str) -> str:
    return (f"build/tools/torchvenv/bin/python -I sw/torch/tools/fuzz_full.py --replay {seed}"
            + (" --gen elem" if kind == "elem" else "") + " -v")


class _Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise _Timeout("case timeout")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen", choices=sorted(GENERATORS), default="random", help="random: models with matmuls; elem: matmul-free")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--replay", type=int, metavar="SEED", help="run one seed and print its record")
    ap.add_argument("--src", help="with --replay: a model source file in place of the generated one")
    ap.add_argument("--json", help="append one JSON line per case here")
    ap.add_argument("--threads", type=int, default=2, help="torch threads (the simulator: TSR_SIM_THREADS)")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per case")
    ap.add_argument("-v", action="store_true", help="with --replay: print the model source")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.replay is not None:
        rec, flags, src = check_seed(a.replay, a.gen, open(a.src).read() if a.src else None)
        if a.v:
            print(src)
        print(json.dumps(rec, indent=1, default=str))
        print("FLAGS", flags)
        return 1 if flags else 0
    signal.signal(signal.SIGALRM, _alarm)
    flagged = unstable = low = 0
    log = open(a.json, "a") if a.json else None
    for seed in range(a.start, a.start + a.count):
        t0 = time.time()
        signal.alarm(a.timeout)
        try:
            rec, flags, _ = check_seed(seed, a.gen)
        except _Timeout:
            rec, flags = {"seed": seed, "gen": a.gen, "harness_error": "timeout"}, [("harness", "timeout")]
        finally:
            signal.alarm(0)
        unstable += any(rec.get("unstable") or [])
        low += bool(rec.get("low_energy"))
        rec["time"] = round(time.time() - t0, 2)
        rec["flags"] = flags
        if log:
            log.write(json.dumps(rec, default=str) + "\n")
            log.flush()
        if flags:
            flagged += 1
            print(f"FLAGGED seed {seed} ({a.gen}): {flags}\n  replay: {replay_command(seed, a.gen)}", flush=True)
    print(f"{a.count} case(s) from seed {a.start} ({a.gen}): {flagged} flagged, {unstable} with an unstable output "
          f"(its values not judged), {low} with an output judged by absolute error only")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

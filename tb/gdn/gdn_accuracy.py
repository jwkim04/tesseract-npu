#!/usr/bin/env python3
"""Accuracy study of the GDN state update (docs/design/02-vector-unit.md §7.12.6, VU-29).

Question: how should the two d_k-long dot products of the fused op be accumulated?
  (A) f32 adders in a fixed tree: 8 lane accumulators over the row (lane l sums elements
      l, l+8, l+16, ... in order), then ((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)); products rounded to f32
  (B) exact integer sums of aligned mantissa products (what DSP48E2 accumulators do), one
      conversion to f32 per dot product: the C reference sw/sim/tsr_ref_gdn.c (tsr_ref_gdn)

Everything else is the same in A and B (the algebra of VU §7.12.3: u' = S.(alpha*beta*k),
w = S.q, kappa = k.q, e = beta*v - u', S' = FA(FM(alpha,S), FM(e,k)), o = alpha*w + e*kappa), so the
runs differ only in the dot products. The reference is the plain recurrence in float64:
    u = S k;  e = beta (v - alpha u);  S <- alpha S + e k^T;  o = S q
with the same f32 inputs. A third run, "lit32", is that recurrence in float32 in PyTorch (torch.mv for
both dots, f32 elementwise ops), the way a CPU or GPU kernel computes it.

Parts
  dot     one dot product of a 128-row state against a unit vector (static data of several
          kinds): error of A8 (the tree above), of numpy's f32 matmul, and of B with different
          table and window widths, relative to |S| |k|
  update  the element update in a block frame (S' = alpha S + e k computed as integer products of the
          aligned S and k and converted to f32 per element, "U3") against the f32 update FA(FM, FM) of the
          reference ("U1"), same dots: why VU-30 keeps the f32 update
  system  a head of 128 x 128 over N steps (default 32768) with gating, in several input
          scenarios; relative error of o per step (median, 99th percentile, maximum, and the last
          1000 steps) and of S (Frobenius, and the largest error relative to the row maximum) at
          checkpoints

Usage (the PyTorch environment is build/tools/torchvenv/bin/python; without torch lit32 runs in
numpy f32):
  gdn_accuracy.py dot
  gdn_accuracy.py update [--steps 8192]
  gdn_accuracy.py system [--steps 32768] [--scenarios a,b,...] [--json out.json]
Needs libtsr_ref.so (cmake --build build/dev --target tsr_ref_shared).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdn_ref  # noqa: E402

f32 = np.float32
M = 128           # d_k
R = 128           # d_v


def u32(x):
    return np.ascontiguousarray(x, dtype=f32).view(np.uint32)


def f2u(x) -> int:
    return int(np.float32(x).view(np.uint32))


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return (x / np.sqrt((x * x).sum())).astype(f32)


# ---------------------------------------------------------------------------
# (A) f32 tree
# ---------------------------------------------------------------------------
def dot_a8(S, k, lanes=8):
    r, d = S.shape
    p = (S * k[None, :]).astype(f32).reshape(r, d // lanes, lanes)
    acc = np.zeros((r, lanes), f32)
    for g in range(d // lanes):
        acc = (acc + p[:, g, :]).astype(f32)
    while acc.shape[1] > 1:
        acc = (acc[:, 0::2] + acc[:, 1::2]).astype(f32)
    return acc[:, 0]


# ---------------------------------------------------------------------------
# (B) numpy prototype with a configurable window (for the width sweep; the 26 / 24 bit point is
#     the C reference, tsr_ref_gdn.c, and is checked against it below)
# ---------------------------------------------------------------------------
def _fields(x):
    b = x.view(np.uint32)
    s = (b >> 31).astype(np.int64)
    e = ((b >> 23) & 0xFF).astype(np.int64)
    m = np.where(e == 0, 0, (b & 0x7FFFFF).astype(np.int64) | (1 << 23))
    return s, e, m


def _align(m, e, emax, w, rnd):
    """magnitude window of w bits below the frame maximum (the element at emax keeps its 24-bit
    significand, shifted by w - 24), truncated or rounded half up."""
    sh = np.minimum(emax - e, 62)
    g = w - 24
    if g >= 0:
        t = m << g
        if rnd:
            r = np.where(sh >= 1, ((t >> np.maximum(sh - 1, 0)) + 1) >> 1, t)
            return np.where(sh > 0, r, t)
        return t >> sh
    tot = sh - g
    if rnd:
        return ((m >> np.maximum(tot - 1, 0)) + 1) >> 1
    return m >> tot


def bfp_table(x, w=24):
    s, e, m = _fields(x)
    emax = int(e.max())
    q = _align(m, e, emax, w, True)
    return np.where(s == 1, -q, q), emax - 150 - (w - 24)


def dot_b(S, tab, w=26):
    q, qe = tab
    s, e, m = _fields(S)
    es = e.max(axis=1, keepdims=True)
    sm = _align(m, e, es, w, False)
    sf = np.where(s == 1, -sm, sm)
    acc = (sf * q[None, :]).sum(axis=1)
    return np.ldexp(acc.astype(f32), es[:, 0] - 150 - (w - 24) + qe).astype(f32)


def dot_study():
    rng = np.random.default_rng(5)
    D = M
    kinds = {
        "gauss": lambda: rng.standard_normal((R, D)),
        "heavy": lambda: rng.standard_cauchy((R, D)),
        "log-uniform": lambda: rng.choice([-1, 1], (R, D)) * np.exp(rng.uniform(-10, 10, (R, D))),
        "orthogonal": lambda: None,
    }
    cands = {
        "np f32 matmul": lambda S, k: (S @ k).astype(f32),
        "A8 tree": lambda S, k: dot_a8(S, k),
        "B S26 T24": lambda S, k: dot_b(S, bfp_table(k, 24), 26),
        "B S24 T24": lambda S, k: dot_b(S, bfp_table(k, 24), 24),
        "B S26 T20": lambda S, k: dot_b(S, bfp_table(k, 20), 26),
        "B S26 T18": lambda S, k: dot_b(S, bfp_table(k, 18), 26),
        "B S26 T16": lambda S, k: dot_b(S, bfp_table(k, 16), 26),
    }
    print("dot product error / (|S_row| |k|), 200 trials of 128 rows x 128, max over rows and trials (median of the "
          "per-trial maxima in brackets); one f32 ulp of a result is 2^-24 = 6.0e-08 of its own magnitude")
    print("%-14s" % "" + "".join("%-22s" % n for n in cands))
    for kn, mk in kinds.items():
        res = {n: [] for n in cands}
        for _ in range(200):
            if kn == "orthogonal":
                k = rng.standard_normal(D)
                k /= np.linalg.norm(k)
                S = rng.standard_normal((R, D))
                S = S - np.outer(S @ k, k) + 1e-6 * rng.standard_normal((R, D))
            else:
                S = mk()
                k = rng.standard_normal(D)
                k /= np.linalg.norm(k)
            S, k = S.astype(f32), k.astype(f32)
            ex = S.astype(np.float64) @ k.astype(np.float64)
            scale = np.linalg.norm(S.astype(np.float64), axis=1) * np.linalg.norm(k.astype(np.float64))
            for n, fn in cands.items():
                res[n].append(np.max(np.abs(fn(S, k).astype(np.float64) - ex) / scale))
        print("%-14s" % kn + "".join("%-22s" % ("%.2e (%.2e)" % (max(v), np.median(v))) for v in res.values()))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_scenario(name, rng, d_k):
    """Returns gen(t) -> (k, q, v, alpha, beta): f32 vectors and f32 scalars (Python floats)."""
    mu = unit(rng.standard_normal(d_k))
    outl = rng.choice(d_k, 2, replace=False)

    def gate(lam):
        g = -lam * np.exp(0.5 * rng.standard_normal())
        return float(np.float32(np.exp(g)))

    def beta():
        return float(np.float32(1.0 / (1.0 + np.exp(-rng.standard_normal()))))

    def gen(t):
        qraw = rng.standard_normal(d_k)
        kraw = rng.standard_normal(d_k)
        v = rng.standard_normal(R)
        lam = 1e-1
        if name == "fast":          # decay 0.9 per token
            lam = 1e-1
        elif name == "typical":     # time constant 100 tokens
            lam = 1e-2
        elif name == "slow":        # 10000 tokens
            lam = 1e-4
        elif name == "correlated":  # keys close to one direction, medium decay
            lam = 1e-2
            kraw = mu * 4.0 + kraw * 0.5
        elif name == "outliers":    # two key channels 30x larger, heavy-tailed values
            lam = 1e-2
            kraw[outl] *= 30.0
            v = rng.standard_t(3, R)
        elif name == "large":       # values of magnitude 1e3 (state far from 1)
            lam = 1e-2
            v = v * 1e3
        else:
            raise ValueError(name)
        k = unit(kraw)
        q = (unit(qraw) * f32(1.0 / np.sqrt(d_k))).astype(f32)
        return k, q, v.astype(f32), gate(lam), beta()

    return gen


SCENARIOS = ["fast", "typical", "slow", "correlated", "outliers", "large"]


# ---------------------------------------------------------------------------
# System runs
# ---------------------------------------------------------------------------
def system_run(scn, steps, seed, report_every):
    rng = np.random.default_rng(seed)
    gen = make_scenario(scn, rng, M)
    try:
        import torch
        torch.set_num_threads(1)
        have_torch = True
    except Exception:       # noqa: BLE001
        have_torch = False

    S64 = np.zeros((R, M))
    SA = np.zeros((R, M), f32)
    SB = np.zeros((R, M), np.uint32)
    if have_torch:
        SL = torch.zeros((R, M), dtype=torch.float32)
    else:
        SL = np.zeros((R, M), f32)
    names = ["A8", "B", "lit32"]
    eo = {n: np.zeros(steps) for n in names}
    marks = {}
    for t in range(steps):
        k, q, v, alpha, beta = gen(t)
        a32, b32 = f32(alpha), f32(beta)
        # float64 reference
        k64, q64, v64 = k.astype(np.float64), q.astype(np.float64), v.astype(np.float64)
        u64 = S64 @ k64
        e64 = float(b32) * (v64 - float(a32) * u64)
        S64 = float(a32) * S64 + np.outer(e64, k64)
        o64 = S64 @ q64
        n64 = np.linalg.norm(o64) + 1e-300
        # A: f32 tree dots, same chain
        ab = f32(a32 * b32)
        kp = (ab * k).astype(f32)
        bv = (b32 * v).astype(f32)
        up = dot_a8(SA, kp)
        w = dot_a8(SA, q)
        kappa = dot_a8(k[None, :], q)[0]
        e = (bv - up).astype(f32)
        SA = ((a32 * SA).astype(f32) + np.outer(e, k).astype(f32)).astype(f32)
        oA = ((a32 * w).astype(f32) + (e * kappa).astype(f32)).astype(f32)
        # B: the C reference
        oB = gdn_ref.head_inplace(M, f2u(a32), f2u(b32), u32(k), u32(q), u32(v), SB).view(f32)
        # lit32: the textbook recurrence in float32
        if have_torch:
            kt, qt, vt = torch.from_numpy(k), torch.from_numpy(q), torch.from_numpy(v)
            ut = torch.mv(SL, kt)
            et = torch.tensor(b32) * (vt - torch.tensor(a32) * ut)
            SL = torch.tensor(a32) * SL + torch.outer(et, kt)
            oL = torch.mv(SL, qt).numpy()
        else:
            ut = (SL @ k).astype(f32)
            et = (b32 * (v - (a32 * ut).astype(f32)).astype(f32)).astype(f32)
            SL = ((a32 * SL).astype(f32) + np.outer(et, k).astype(f32)).astype(f32)
            oL = (SL @ q).astype(f32)
        for n, o in (("A8", oA), ("B", oB), ("lit32", oL)):
            eo[n][t] = np.linalg.norm(o.astype(np.float64) - o64) / n64
        if (t + 1) % report_every == 0 or t + 1 == steps:
            rm = np.abs(S64).max(axis=1, keepdims=True) + 1e-300
            nrm = np.linalg.norm(S64)
            Sd = {"A8": SA.astype(np.float64), "B": SB.view(f32).astype(np.float64),
                  "lit32": (SL.numpy() if have_torch else SL).astype(np.float64)}
            marks[t + 1] = {n: (float(np.linalg.norm(Sd[n] - S64) / nrm), float((np.abs(Sd[n] - S64) / rm).max()))
                            for n in names}
    return eo, marks


def system_study(steps, scenarios, seed, jpath):
    out = {}
    t0 = time.time()
    for scn in scenarios:
        eo, marks = system_run(scn, steps, seed, max(1, steps // 4))
        out[scn] = {"o": {n: {"median": float(np.median(e)), "p99": float(np.percentile(e, 99)), "max": float(e.max()),
                              "last1000_median": float(np.median(e[-1000:]))} for n, e in eo.items()},
                    "S": {str(t): m for t, m in marks.items()}}
        print("scenario %-10s (%d steps, %.0f s)" % (scn, steps, time.time() - t0))
        print("   %-6s %-12s %-12s %-12s %-12s | S: %s" % ("", "o median", "o p99", "o max", "o last1000",
                                                          "Frobenius and max |dS|/row max at steps " +
                                                          ", ".join(str(t) for t in marks)))
        for n in eo:
            od = out[scn]["o"][n]
            print("   %-6s %-12.3e %-12.3e %-12.3e %-12.3e | %s" % (
                n, od["median"], od["p99"], od["max"], od["last1000_median"],
                "  ".join("%.2e/%.2e" % marks[t][n] for t in marks)))
        sys.stdout.flush()
    if jpath:
        Path(jpath).write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# The rejected alternative for the element update (VU-30): S' in a block frame
# ---------------------------------------------------------------------------
def _scalar(x):
    b = int(np.float32(x).view(np.uint32))
    return b >> 31, (b >> 23) & 0xFF, ((b & 0x7FFFFF) | 0x800000) if (b >> 23) & 0xFF else 0


def update_u3(S, alpha, e, ktab, w=26, rnd=True):
    """S' = alpha S + e k with S aligned to the row frame (w-bit window) and k to its table frame (24 bit); the
    scalar mantissas of alpha and e are shifted to a common frame (the smaller term loses bits, as a DSP
    implementation without a product shifter must); result converted to f32 per element (RNE)."""
    kq, kexp = ktab
    s, ex, m = _fields(S)
    es = ex.max(axis=1)
    sm = _align(m, ex, es[:, None], w, rnd)
    sf = np.where(s == 1, -sm, sm)
    sa, ea, ma = _scalar(alpha)
    se, ee, me = _fields(e)
    ek = kexp + 150
    f1 = ea + es - (w - 24)
    f2 = ee + ek
    f = np.maximum(f1, f2)
    ma_s = ma >> np.minimum(f - f1, 62)
    me_s = me >> np.minimum(f - f2, 62)
    t = (-1 if sa else 1) * ma_s[:, None] * sf + (np.where(se == 1, -1, 1) * me_s)[:, None] * kq[None, :]
    return np.ldexp(t.astype(f32), (f - 300)[:, None]).astype(f32)


def update_study(steps):
    print("S' error against float64 after %d steps (typical gating, 128 x 128): f32 update (U1) and the block-frame "
          "update (U3) with 26 / 24 bit windows, rounded or truncated; same dot products (B)" % steps)
    for lam, label in ((1e-1, "alpha ~ 0.9"), (1e-3, "alpha ~ 0.999"), (1e-5, "alpha ~ 0.99999")):
        res = {}
        rng = np.random.default_rng(3)
        st = {n: np.zeros((R, M), f32) for n in ("U1", "U3 w26 rn", "U3 w26 tr", "U3 w24 tr")}
        S64 = np.zeros((R, M))
        for t in range(steps):
            k = unit(rng.standard_normal(M))
            q = unit(rng.standard_normal(M))
            v = rng.standard_normal(R).astype(f32)
            a32 = f32(np.exp(-lam * np.exp(0.5 * rng.standard_normal())))
            b32 = f32(1.0 / (1.0 + np.exp(-rng.standard_normal())))
            u64 = S64 @ k.astype(np.float64)
            e64 = float(b32) * (v.astype(np.float64) - float(a32) * u64)
            S64 = float(a32) * S64 + np.outer(e64, k.astype(np.float64))
            kp = (f32(a32 * b32) * k).astype(f32)
            bv = (b32 * v).astype(f32)
            ktab = bfp_table(k, 24)
            for n in st:
                up = dot_b(st[n], bfp_table(kp, 24), 26)
                e = (bv - up).astype(f32)
                if n == "U1":
                    st[n] = ((a32 * st[n]).astype(f32) + np.outer(e, k).astype(f32)).astype(f32)
                else:
                    w = 24 if "w24" in n else 26
                    st[n] = update_u3(st[n], a32, e, ktab, w, "rn" in n)
        nrm = np.linalg.norm(S64)
        print("  %-16s" % label + "  ".join("%s %.2e" % (n, np.linalg.norm(st[n].astype(np.float64) - S64) / nrm) for n in st))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dot")
    up = sub.add_parser("update")
    up.add_argument("--steps", type=int, default=8192)
    sp = sub.add_parser("system")
    sp.add_argument("--steps", type=int, default=32768)
    sp.add_argument("--scenarios", default=",".join(SCENARIOS))
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "dot":
        dot_study()
    elif a.cmd == "update":
        update_study(a.steps)
    else:
        system_study(a.steps, a.scenarios.split(","), a.seed, a.json)


if __name__ == "__main__":
    main()

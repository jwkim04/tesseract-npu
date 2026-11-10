"""Data classes of the GDN benches (docs/design/02-vector-unit.md §7.12.10): one head's k, q, v, S, alpha, beta as
flat lists of f32 bit patterns. Shared by tb/gdn (the engine against the C reference) and tb/vu (the vector unit with
the GDN op against the simulator)."""
from __future__ import annotations

import struct

import numpy as np

KINDS = ["model", "spread", "zero", "tiny", "huge", "bits", "rowinf", "satur", "edge"]


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def head_data(r, kind: str, m: int, rows: int):
    """k, q, v, S (flat lists of f32 bit patterns), alpha, beta for one head of a data class; r is a random.Random."""
    g = np.random.default_rng(r.getrandbits(32))
    f = lambda a: [int(x) for x in np.asarray(a, dtype=np.float32).view(np.uint32)]

    def unit(x):
        return x / np.sqrt((x * x).sum())

    if kind == "model":                       # normalized keys, values of order one, gating
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)) / np.sqrt(m))
        v = f(g.standard_normal(rows)); S = f(g.standard_normal((rows, m)).ravel() * 0.5)
        alpha = f2u(float(np.exp(-g.random() * 0.1))); beta = f2u(float(1 / (1 + np.exp(-g.standard_normal()))))
    elif kind == "spread":                    # a wide exponent spread inside rows
        k = f(g.standard_normal(m) * np.exp2(g.integers(-20, 20, m))); q = f(g.standard_normal(m) * np.exp2(g.integers(-20, 20, m)))
        v = f(g.standard_normal(rows) * np.exp2(g.integers(-10, 10, rows)))
        S = f(g.standard_normal(rows * m) * np.exp2(g.integers(-40, 40, rows * m)))
        alpha = f2u(float(g.random())); beta = f2u(float(g.random()))
    elif kind == "zero":                      # zero state, zero vectors
        k = f(g.standard_normal(m)); q = [0] * m; v = f(np.zeros(rows)); S = [0] * (rows * m)
        alpha = f2u(0.5); beta = f2u(0.25)
    elif kind == "tiny":                      # results near and below 2^-126
        k = f(g.standard_normal(m) * 2.0 ** -60); q = f(g.standard_normal(m) * 2.0 ** -60)
        v = f(g.standard_normal(rows) * 2.0 ** -60); S = f(g.standard_normal(rows * m) * 2.0 ** -60)
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "huge":
        k = f(g.standard_normal(m) * 2.0 ** 60); q = f(g.standard_normal(m) * 2.0 ** 60)
        v = f(g.standard_normal(rows) * 2.0 ** 60); S = f(g.standard_normal(rows * m) * 2.0 ** 60)
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "bits":                      # arbitrary bit patterns: NaN, Inf, subnormals
        k = [r.getrandbits(32) for _ in range(m)]; q = [r.getrandbits(32) for _ in range(m)]
        v = [r.getrandbits(32) for _ in range(rows)]; S = [r.getrandbits(32) for _ in range(rows * m)]
        alpha = r.getrandbits(32); beta = r.getrandbits(32)
    elif kind == "rowinf":                    # one poisoned row
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)))
        v = f(g.standard_normal(rows)); S = f(g.standard_normal(rows * m))
        S[(rows // 2) * m + 3] = 0x7F800000
        alpha = f2u(0.8); beta = f2u(0.5)
    elif kind == "edge":                      # elements 22 .. 28 binades below the row maximum (the window edge)
        k = f(unit(g.standard_normal(m))); q = f(unit(g.standard_normal(m)))
        v = f(g.standard_normal(rows))
        S = []
        for i in range(rows):
            row = [(r.getrandbits(1) << 31) | ((150 - r.randrange(22, 29)) << 23) | r.getrandbits(23) for _ in range(m)]
            row[r.randrange(m)] = (150 << 23) | r.getrandbits(23)
            S += row
        alpha = f2u(0.9); beta = f2u(0.5)
    elif kind == "satur":                     # the largest aligned elements, all one sign
        k = [0x3F7FFFFF] * m; q = [0x3F7FFFFF] * m; v = f(g.standard_normal(rows))
        S = [(200 << 23) | 0x7FFFFF] * (rows * m); alpha = f2u(1.0); beta = f2u(1.0)
    else:
        raise ValueError(kind)
    return k, q, v, S, alpha, beta

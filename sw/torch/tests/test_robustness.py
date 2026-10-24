"""Regression tests of the 2026-10-24 review of sw/torch: one test (or group) per finding.

1. the churn guard is per weight, not per graph node (regional compilation, a compiled helper
   called per layer, HOP bodies shared by every layer);
2. cached handles belong to the device they were uploaded to (Device.close() / _C.close());
3. open_device() options stay usable afterwards (_C.get() returns the open device as is);
4. an unavailable device is an error, and so is a transport error during an upload;
5. closing a device while another thread is inside a device call (subprocess: it crashed);
6. concurrent use of the singleton functions (subprocess: it deadlocked);
7. inference-tensor weights updated in place are uploaded again;
8. matmuls inside higher-order-op bodies (nested_compile_region, cond, while_loop, map).
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch._higher_order_ops.map import map as hop_map
from torch._higher_order_ops.while_loop import while_loop

import tesseract_torch as tt

HERE = Path(__file__).resolve().parent
_C = tt._C


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    iv = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
    return torch.equal(a.contiguous().view(iv[a.dtype]), b.contiguous().view(iv[b.dtype]))


def _randn(*shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class Layer(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return torch.relu(self.fc(x))


def _layers(n, seed=0):
    torch.manual_seed(seed)
    return nn.ModuleList(Layer() for _ in range(n)).eval()


def _run_script(code: str, timeout: float = 180):
    """Run code in a fresh interpreter with this process's module paths."""
    env = dict(os.environ)
    paths = [str(HERE.parent), os.path.dirname(_C.__file__)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                          timeout=timeout)


# ------------------------------------------------------------------------------------------
# 1. churn guard per weight
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["regional", "helper"])
def test_shared_graph_uploads_each_weight_once(form):
    """layer.compile() per block (torch's regional compilation recipe) and a compiled helper called
    per layer reuse one graph, so one matmul node sees every layer's weight. Each weight is uploaded
    once, and every call runs on the device (the node used to go to the CPU after 4 distinct weights)."""
    L = 8
    layers = _layers(L)
    if form == "regional":
        for layer in layers:
            layer.compile(backend="tesseract")

        def step(layer, h):
            return layer(h)
    else:
        step = torch.compile(lambda layer, h: layer(h), backend="tesseract")
    x = _randn(2, 128, seed=1)
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        for _ in range(3):
            h = x
            for layer in layers:
                h = step(layer, h)
        c = tt.counters()
        with tt.reference_linears():
            h_ref = x
            for layer in layers:
                h_ref = Layer.forward(layer, h_ref)
    assert c["graphs"] == 1, "the layers no longer share one graph: this test does not cover the shared node"
    assert (c["device_calls"], c["cpu_fallback_calls"], c["uploads"], c["cache_hits"]) == (3 * L, 0, L, 2 * L), \
        tt.format_counters(c)
    assert not [i for i in w if "consecutive calls" in str(i.message)]
    assert _same_bits(h, h_ref)


def test_shared_graph_upload_failure_is_per_weight(monkeypatch):
    """A weight that cannot be placed moves only that weight to the CPU, not every weight of the node."""
    L = 6
    layers = _layers(L)
    for layer in layers:
        layer.compile(backend="tesseract")
    bad = layers[2].fc.weight
    cache = tt.weight_cache()
    orig = cache.acquire

    def acquire(w, name=""):
        if w.data_ptr() == bad.data_ptr():
            raise tt.TesseractError("tsr_rt: WEIGHT heap full (test)")
        return orig(w, name)

    monkeypatch.setattr(cache, "acquire", acquire)
    x = _randn(2, 128, seed=2)
    with warnings.catch_warnings(record=True) as w, torch.no_grad():
        warnings.simplefilter("always")
        for _ in range(2):
            h = x
            for layer in layers:
                h = layer(h)
        h_ref = x
        for i, layer in enumerate(layers):
            if i == 2:
                h_ref = Layer.forward(layer, h_ref)           # this one on the CPU
            else:
                with tt.reference_linears():
                    h_ref = Layer.forward(layer, h_ref)
    c = tt.counters()
    assert (c["device_calls"], c["cpu_fallback_calls"], c["uploads"]) == (2 * (L - 1), 2, L - 1), tt.format_counters(c)
    assert sum("stays on the CPU" in str(i.message) for i in w) == 1
    assert _same_bits(h, h_ref)


# ------------------------------------------------------------------------------------------
# 2. handles belong to their device
# ------------------------------------------------------------------------------------------
def test_device_close_invalidates_cached_handles():
    """tsr_rt handles restart at 1 on every device. After Device.close() / _C.close() a cached entry
    must not be a hit on the reopened device (it named another weight there), and sweeping a stale
    entry must not free a weight of the new device."""
    A, B, C = _randn(128, 128, seed=3), _randn(128, 128, seed=4), _randn(32, 128, seed=5)
    x = _randn(3, 128, seed=6)
    tt.close_device()                            # a fresh device: A gets handle 1
    with torch.no_grad():
        yA0 = tt.linear(x, A)
        tt.device().close()                      # the binding's close(), not tt.close_device()
        yB0 = tt.linear(x, B)                    # reopened device: B gets handle 1 again
        yA1 = tt.linear(x, A)
        assert _same_bits(yA1, yA0), "A computed with a stale handle"
        assert not _same_bits(yA1, yB0)
        D = _randn(64, 128, seed=7)
        _C.close()                               # the module's close()
        tt.linear(x, D)                          # handle 1 on this device
        _C.close()
        yB1 = tt.linear(x, B)                    # handle 1 on the next one
        del D
        gc.collect()
        tt.linear(x, C)                          # a miss sweeps D's dead entry: must not free(1) here
        assert _same_bits(tt.linear(x, B), yB1)
    assert _same_bits(yB1, yB0)


# ------------------------------------------------------------------------------------------
# 3. open options
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"read_perf": False}, {"io_bytes": 1 << 20}], ids=["read_perf", "io_bytes"])
def test_open_device_options_stay_usable(kw):
    """After open_device(read_perf=False) or open_device(io_bytes=...), tt.linear, tt.device(),
    tt.counters() and torch.compile use that device (they used to raise 'other options', and
    compiled graphs ran on the CPU)."""
    tt.close_device()
    try:
        d = tt.open_device(**kw)
        assert tt.device() is d
        W, x = _randn(64, 128, seed=8), _randn(2, 128, seed=9)
        torch.manual_seed(10)
        m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
        with torch.no_grad():
            y = tt.linear(x, W)
            assert _same_bits(y, tt.linear_ref(x, W))
            yc = torch.compile(m, backend="tesseract")(x)
            with tt.reference_linears():
                assert _same_bits(yc, m(x))
        c = tt.counters()
        assert (c["graphs"], c["nodes_device"], c["device_calls"]) == (1, 2, 3)
        if "read_perf" in kw:
            assert c["rt_perf_cycles"] == 0           # the PERF CSRs are not read
        # an explicit open with different options still refuses
        other = {"read_perf": True} if "read_perf" in kw else {"io_bytes": 1 << 21}
        with pytest.raises(tt.TesseractError, match="other options"):
            tt.open_device(**other)
        assert tt.open_device(**kw) is d and tt.open_device() is d
    finally:
        tt.close_device()


# ------------------------------------------------------------------------------------------
# 4. errors are not turned into CPU runs
# ------------------------------------------------------------------------------------------
def test_unavailable_device_is_an_error(monkeypatch):
    """TSR_DEVICE naming a transport that cannot be opened makes torch.compile(backend="tesseract")
    fail (it used to warn and compute everything on the CPU)."""
    tt.close_device()
    monkeypatch.setenv("TSR_DEVICE", "vkernal")
    torch.manual_seed(11)
    m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    with torch.no_grad(), pytest.raises(Exception, match="failed to open the Tesseract transport"):
        torch.compile(m, backend="tesseract")(_randn(2, 128, seed=12))
    assert not _C.is_open() and tt.counters()["graphs"] == 0


def test_upload_transport_error_raises(monkeypatch):
    """A transport failure during a weight upload raises; only "cannot place" (heap full) falls back."""
    torch.manual_seed(13)
    m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    x = _randn(2, 128, seed=14)
    msg = {"text": "tsr_rt: transport write failed (ch 1, offset 0x40000, 65536 bytes) (test)"}

    def upload(self, w, max_part_rows=0, max_part_k=0):
        raise tt.TesseractError(msg["text"])

    monkeypatch.setattr(_C.Device, "upload", upload)
    with torch.no_grad():
        c = torch.compile(m, backend="tesseract")
        with pytest.raises(tt.TesseractError, match="transport write failed"):
            c(x)
        msg["text"] = "tsr_rt: WEIGHT heap full: (test)"
        with pytest.warns(RuntimeWarning, match="stays on the CPU"):
            y = c(x)
    assert torch.equal(y, m(x)) and tt.counters()["cpu_fallback_calls"] == 2


# ------------------------------------------------------------------------------------------
# 5. close while a call is in flight (subprocess: this crashed with SIGSEGV)
# ------------------------------------------------------------------------------------------
def test_close_during_device_call():
    r = _run_script("""
        import faulthandler, threading, time, torch
        import tesseract_torch as tt
        faulthandler.dump_traceback_later(150, exit=True)
        _C = tt._C
        W = torch.randn(2048, 4096, generator=torch.Generator().manual_seed(0))
        x = torch.randn(64, 4096, generator=torch.Generator().manual_seed(1))

        # (a) a Device of its own, closed while another thread is inside linear()
        d = _C.Device("sim")
        h = d.upload(W)
        y0 = d.linear(h, x)
        started, out = threading.Event(), {"n": 0}
        def worker():
            try:
                for _ in range(100):
                    started.set()
                    out["y"] = d.linear(h, x)
                    out["n"] += 1
            except Exception as e:
                out["err"] = f"{type(e).__name__}: {e}"
        t = threading.Thread(target=worker)
        t.start(); started.wait(); time.sleep(0.05)
        d.close()
        t.join()
        assert d.closed and "the device is closed" in out.get("err", ""), out
        assert out["n"] >= 1 and torch.equal(out["y"], y0)

        # (b) the process-wide device, closed with tt.close_device() during tt.linear in a thread
        with torch.no_grad():
            tt.linear(x[:1], W)
        started, out = threading.Event(), {"n": 0, "closed": 0}
        def worker2():
            with torch.no_grad():
                for _ in range(4):
                    started.set()
                    try:
                        y = tt.linear(x, W)
                    except tt.TesseractError as e:      # the call raced with the close
                        assert "closed" in str(e), e
                        out["closed"] += 1
                        continue
                    assert torch.equal(y, y0)
                    out["n"] += 1
        t = threading.Thread(target=worker2)
        t.start(); started.wait(); time.sleep(0.05)
        tt.close_device()
        t.join()
        assert out["n"] + out["closed"] == 4 and out["n"] >= 3, out
        print("OK", out)
    """)
    assert r.returncode == 0 and "OK" in r.stdout, f"exit {r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"


# ------------------------------------------------------------------------------------------
# 6. concurrent singleton functions (subprocess: this deadlocked)
# ------------------------------------------------------------------------------------------
def test_concurrent_singleton_open():
    r = _run_script("""
        import faulthandler, threading
        import tesseract_torch as tt
        C = tt._C
        faulthandler.dump_traceback_later(30, exit=True)
        for it in range(25):
            C.close()
            go = threading.Barrier(5)
            res = [None] * 5
            fns = [C.get, C.get, C.open, C.get, C.is_open]
            def run(i):
                go.wait()
                res[i] = fns[i]()
            ths = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(5)]
            for t in ths: t.start()
            for t in ths: t.join()
            devs = res[:4]
            assert all(d is devs[0] for d in devs) and not devs[0].closed, res
        faulthandler.cancel_dump_traceback_later()
        print("OK")
    """, timeout=120)
    assert r.returncode == 0 and "OK" in r.stdout, f"exit {r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"


# ------------------------------------------------------------------------------------------
# 7. inference tensors
# ------------------------------------------------------------------------------------------
def test_inference_tensor_weight_updated_in_place():
    """A weight that is an inference tensor (model built under inference_mode) has no version
    counter. An in-place update of it (load_state_dict, mul_) must still reach the device, with a
    warning that such weights are checked by a sample."""
    tt.runtime._warned.discard("inference-tensor")
    x = _randn(2, 128, seed=15)
    with torch.inference_mode():
        m = nn.Linear(128, 64, bias=False).eval()
    assert m.weight.is_inference()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.inference_mode():
            y0 = tt.linear(x, m.weight)
            assert _same_bits(tt.linear(x, m.weight), y0)        # unchanged: a cache hit
            m.load_state_dict({"weight": _randn(64, 128, seed=16)})
            y1 = tt.linear(x, m.weight)
        w1 = m.weight.clone()                                     # a normal tensor with the new values
        with torch.inference_mode():
            m.weight.mul_(2.0)
            y2 = tt.linear(x, m.weight)
        w2 = m.weight.clone()
    with torch.no_grad():
        assert _same_bits(y1, tt.linear_ref(x, w1))
        assert _same_bits(y2, tt.linear_ref(x, w2))
    assert not _same_bits(y1, y0)
    assert any("inference tensor" in str(i.message) for i in w)
    c = tt.counters()
    assert (c["reuploads"], c["cache_hits"] >= 1) == (2, True), tt.format_counters(c)


# ------------------------------------------------------------------------------------------
# 8. matmuls inside higher-order-op bodies
# ------------------------------------------------------------------------------------------
class NestedLayer(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.fc = nn.Linear(d, d, bias=False)

    @torch.compiler.nested_compile_region
    def forward(self, x):
        return torch.relu(self.fc(x))


class NestedNet(nn.Module):
    def __init__(self, L=6):
        super().__init__()
        self.layers = nn.ModuleList(NestedLayer() for _ in range(L))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_nested_compile_region_body_on_device():
    """torch.compiler.nested_compile_region: the layers share one invoke_subgraph body; its matmul goes
    to the device with every layer's weight (it used to be neither offloaded nor reported)."""
    L = 6
    torch.manual_seed(17)
    m = NestedNet(L).eval()
    x = _randn(2, 128, seed=18)
    with torch.no_grad():
        c = torch.compile(m, backend="tesseract")
        y = c(x)
        y2 = c(x)
        with tt.reference_linears():
            y_ref = m(x)
    cnt = tt.counters()
    assert (cnt["graphs"], cnt["nodes_device"], cnt["nodes_cpu"]) == (1, 1, 0), tt.format_counters(cnt)
    assert (cnt["device_calls"], cnt["uploads"], cnt["cpu_fallback_calls"]) == (2 * L, L, 0), tt.format_counters(cnt)
    assert _same_bits(y, y_ref) and _same_bits(y2, y_ref)


class Hops(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128, 128, bias=False)
        self.b = nn.Linear(128, 128)
        self.c = nn.Linear(128, 64, bias=False)

    def forward(self, x, p, n, xs):
        y = torch.cond(p, lambda t: self.a(t), lambda t: torch.relu(self.b(t)), (x,))
        _, h = while_loop(lambda i, t: i < n, lambda i, t: (i + 1, torch.tanh(self.a(t))),
                          (torch.zeros((), dtype=torch.int64), y))
        return h, hop_map(lambda t: self.c(t), xs)


def test_cond_while_loop_map_bodies_on_device():
    torch.manual_seed(19)
    m = Hops().eval()
    x, xs = _randn(2, 128, seed=20), _randn(3, 2, 128, seed=21)
    L = tt.linear_ref
    with torch.no_grad():
        c = torch.compile(m, backend="tesseract")
        for p, n in ((True, 2), (False, 3)):
            h, z = c(x, torch.tensor(p), torch.tensor(n), xs)
            ref = L(x, m.a.weight) if p else torch.relu(L(x, m.b.weight, m.b.bias))
            for _ in range(n):
                ref = torch.tanh(L(ref, m.a.weight))
            assert _same_bits(h, ref), (p, n)
            assert _same_bits(z, torch.stack([L(t, m.c.weight) for t in xs])), (p, n)
    cnt = tt.counters()
    # cond: 2 branch bodies (1 node each), while_loop body: 1, map body: 1
    assert (cnt["graphs"], cnt["nodes_device"], cnt["nodes_cpu"]) == (1, 4, 0), tt.format_counters(cnt)
    assert cnt["device_calls"] == (1 + 2 + 3) + (1 + 3 + 3) and cnt["cpu_fallback_calls"] == 0
    assert cnt["uploads"] == 3


def test_unknown_hop_body_reported_on_cpu(monkeypatch, capfd):
    """Matmuls in the body of a higher-order op the backend does not map stay on the CPU and are
    reported as CPU nodes."""
    monkeypatch.delitem(tt.backend._HOP_OPERANDS, "map_impl")
    torch.manual_seed(22)
    m = Hops().eval()
    x, xs = _randn(2, 128, seed=23), _randn(3, 2, 128, seed=24)
    tt.set_debug(1)
    try:
        with torch.no_grad():
            h, z = torch.compile(m, backend="tesseract")(x, torch.tensor(True), torch.tensor(1), xs)
    finally:
        tt.set_debug(0)
    cnt = tt.counters()
    assert (cnt["nodes_device"], cnt["nodes_cpu"]) == (3, 1), tt.format_counters(cnt)
    err = capfd.readouterr().err
    assert "inside a map_impl body" in err
    with torch.no_grad():
        assert torch.allclose(z, torch.stack([m.c(t) for t in xs]), rtol=0, atol=1e-5)

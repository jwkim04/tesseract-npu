"""Device singleton, weight cache, eager linear and counters of tesseract_torch.

Numerics are the llama.cpp path's (docs/spec.md "Numeric rules", "Load-time conversion"): a float32 / float16 /
bfloat16 weight becomes Q8_0 when it is uploaded (ggml's ``quantize_row_q8_0_ref``, f16 / bf16 widened
exactly first), the activations are quantized on the device by QACT (A32), results are float32.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import sys
import threading
import warnings
from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from ._ext import _C

TesseractError = _C.TesseractError
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
BLOCK = 32  # Q8_0 block: K must be a multiple of it

_lock = threading.RLock()


# ------------------------------------------------------------------------------------------
# debug output (TSR_TORCH_DEBUG)
# ------------------------------------------------------------------------------------------
def _env_debug() -> int:
    v = os.environ.get("TSR_TORCH_DEBUG", "")
    try:
        return int(v) if v else 0
    except ValueError:
        return 1


_debug = _env_debug()
_verify = os.environ.get("TSR_TORCH_VERIFY", "") not in ("", "0")


def debug_level() -> int:
    return _debug


def set_debug(level: int) -> None:
    """Override TSR_TORCH_DEBUG (0 off, 1 partition + counters at exit, 2 also every device call,
    3 also the rewritten aten graphs)."""
    global _debug
    _debug = int(level)


def verify_enabled() -> bool:
    return _verify


def set_verify(on: bool) -> None:
    """Override TSR_TORCH_VERIFY: check every device matmul against the reference model (libtsr_ref on
    the weight rows read back from the device) bit for bit; a mismatch raises TesseractError."""
    global _verify
    _verify = bool(on)


def log(msg: str) -> None:
    print(f"[tesseract_torch] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------
# counters of the Python layer (the runtime's own are in device().counters())
# ------------------------------------------------------------------------------------------
_STAT_KEYS = (
    "graphs",            # AOT graphs partitioned by the backend
    "graphs_eager",      # Dynamo graphs run eagerly because they need autograd (training / requires_grad)
    "nodes_device",      # matmul nodes put on Tesseract
    "nodes_cpu",         # matmul nodes left on the CPU (unsupported)
    "device_calls",      # weight matmuls executed on Tesseract
    "local_scale_calls",  # completed fused device epilogues
    "device_tokens",     # their total T (rows of x)
    "cpu_fallback_calls",  # device nodes that ran on the CPU at run time (weight could not be uploaded)
    "uploads",           # weights uploaded (including re-uploads)
    "reuploads",         # uploads of a cached weight that changed in place (_version, or the sample of an inference tensor)
    "upload_bytes",      # bytes of the host tensors uploaded
    "cache_hits",        # weight lookups served by the cache
    "weights_freed",     # device weights released (dead storage, re-upload, release_weights)
    "verified_calls",    # device matmuls checked against the reference model (TSR_TORCH_VERIFY)
)
_stats = dict.fromkeys(_STAT_KEYS, 0)


def _bump(key: str, n: int = 1) -> None:
    _stats[key] += n


# ------------------------------------------------------------------------------------------
# device singleton
# ------------------------------------------------------------------------------------------
_atexit_done = False


def _register_atexit() -> None:
    global _atexit_done
    if not _atexit_done:
        atexit.register(_shutdown)
        _atexit_done = True


def open_device(kind: str | None = None, io_bytes: int | None = None, read_perf: bool | None = None):
    """Open the process-wide Tesseract device (kind: sim | mock | vkernel | xrt; None = TSR_DEVICE,
    default sim; io_bytes None = 0, all of the IO region; read_perf None = True). Returns the
    ``_tesseract_rt.Device``. If it is already open, it is returned when every option given matches,
    else TesseractError. Without a call, ``device()`` opens it with the defaults on first use."""
    with _lock:
        was_open = _C.is_open()
        d = _C.open(kind, io_bytes, read_perf)
        _register_atexit()
        if not was_open and _debug >= 1:
            log(f"opened {d.description}")
        return d


def device():
    """The process-wide device as it is open (whatever ``open_device`` options it has), else opened
    with the defaults (TSR_DEVICE)."""
    with _lock:
        was_open = _C.is_open()
        d = _C.get()
        if not was_open:
            _register_atexit()
            if _debug >= 1:
                log(f"opened {d.description}")
        return d


def placement_error(err: BaseException) -> bool:
    """True if err says that a weight cannot be placed on the device (WEIGHT heap full, or a type /
    shape the device does not take: ValueError). Transport and execution errors are not."""
    if isinstance(err, ValueError):
        return True
    return isinstance(err, TesseractError) and "heap full" in str(err)


def close_device() -> None:
    """Release every cached weight and close the device (a later call reopens it)."""
    with _lock:
        _cache.forget()
        _C.close()


def _shutdown() -> None:
    try:
        if _debug >= 1 and _C.is_open():
            log("counters at exit:\n" + format_counters())
    finally:
        try:
            close_device()
        except Exception:  # noqa: BLE001 - never fail interpreter shutdown
            pass


# ------------------------------------------------------------------------------------------
# weight cache
# ------------------------------------------------------------------------------------------
def _version_of(t: torch.Tensor):
    try:
        return t._version
    except RuntimeError:  # inference tensors have no version counter: checked by a sample instead
        return None


_SAMPLE = 64   # rows and columns of the sample that stands in for the version of an inference tensor


def _sample(w: torch.Tensor) -> torch.Tensor:
    """Raw bits of up to 64 x 64 elements of w [N, K] on an even grid (first and last row and column
    included)."""
    N, K = int(w.shape[0]), int(w.shape[1])
    rn, cn = min(N, _SAMPLE), min(K, _SAMPLE)
    r = torch.arange(rn) * (N - 1) // max(rn - 1, 1)
    c = torch.arange(cn) * (K - 1) // max(cn - 1, 1)
    with torch.no_grad():
        s = w.detach().index_select(0, r).index_select(1, c)
    return s.view(torch.int16 if s.element_size() == 2 else torch.int32)


@dataclass
class _Entry:
    handle: int
    version: object
    ref: StorageWeakRef
    N: int
    K: int
    dtype: torch.dtype
    name: str
    serial: int                        # the device the handle belongs to (handles are per device)
    sample: torch.Tensor | None = None  # inference tensors (no _version): _sample() at upload


class WeightCache:
    """Device copies of weight tensors, keyed by tensor identity: the storage (its StorageImpl, kept
    from reuse by a weak reference), data pointer, storage offset, shape, stride and dtype. A hit
    also needs the same ``_version``; an in-place update (``w.mul_()``, ``copy_``,
    ``load_state_dict``) bumps it and the next use uploads the weight again. Inference tensors have
    no ``_version``: for them a hit needs the same sample of up to 64 x 64 elements (an in-place
    change outside the sample is not seen; ``release_weights()`` after such changes). A hit also
    needs the device the copy was uploaded to: after that device was closed (``Device.close()``,
    ``_C.close()``, ``close_device()``) the entry is a miss and its handle is never freed on the
    new device. Entries whose storage died are released on the next miss, by ``sweep()`` or with
    ``release_weights()``."""

    def __init__(self) -> None:
        self._map: dict[tuple, _Entry] = {}

    @staticmethod
    def key(w: torch.Tensor) -> tuple:
        st = w.untyped_storage()
        return (st._cdata, st.data_ptr(), w.storage_offset(), tuple(w.shape), tuple(w.stride()), w.dtype)

    def _current(self, e: _Entry, dev, w: torch.Tensor, ver) -> bool:
        if e.serial != dev.serial or e.ref.expired() or e.version != ver:
            return False
        return e.sample is None or torch.equal(e.sample, _sample(w))

    def acquire(self, w: torch.Tensor, name: str = "") -> tuple:
        """(device, handle, uploaded) of weight w [N, K] on the process-wide device: the cached copy if
        it is current (uploaded False), else a new upload (uploaded True). Use the handle with that
        device only."""
        with _lock:
            dev = device()
            k = self.key(w)
            e = self._map.get(k)
            ver = _version_of(w)
            reupload = False
            if e is not None:
                if self._current(e, dev, w, ver):
                    _bump("cache_hits")
                    return dev, e.handle, False
                reupload = e.serial == dev.serial and not e.ref.expired()
                self._release(k, e)
            self.sweep()
            N, K = int(w.shape[0]), int(w.shape[1])
            try:
                h = dev.upload(w.detach())
            except TesseractError as err:
                if "heap full" not in str(err):
                    raise
                self.sweep()
                h = dev.upload(w.detach())
            sample = None
            if ver is None:
                sample = _sample(w)
                _warn_once("inference-tensor",
                           f"tesseract_torch: weight {name or '[%d x %d]' % (N, K)} is an inference tensor (no version "
                           "counter): an in-place update of it is detected only through a sample of up to "
                           f"{_SAMPLE} x {_SAMPLE} elements on each use; call tesseract_torch.release_weights() after "
                           "changing such a weight in place")
            self._map[k] = _Entry(h, ver, StorageWeakRef(w.untyped_storage()), N, K, w.dtype, name, dev.serial, sample)
            _bump("uploads")
            _bump("upload_bytes", w.numel() * w.element_size())
            if reupload:
                _bump("reuploads")
            if _debug >= 1:
                info = dev.info(h)
                log(f"upload {name or 'weight'} [{N} x {K}] {str(w.dtype).replace('torch.', '')} -> {info['fmt']}"
                    f" ({info['n_parts']} x {info['k_parts']} parts, {info['dev_bytes'] / 2**20:.2f} MiB on the device)"
                    + (" (re-upload: the weight changed)" if reupload else ""))
            return dev, h, True

    def handle(self, w: torch.Tensor, name: str = "") -> int:
        """The device handle of weight w [N, K] (uploading it on a miss), on ``device()``."""
        return self.acquire(w, name)[1]

    def find(self, w: torch.Tensor) -> int | None:
        """The device handle of w [N, K] if it is cached and current on the open device, else None."""
        with _lock:
            e = self._map.get(self.key(w))
            if e is None or not _C.is_open():
                return None
            return e.handle if self._current(e, _C.get(), w, _version_of(w)) else None

    def _release(self, k: tuple, e: _Entry) -> None:
        """Drop an entry; free its device copy if it is on the open device (on a closed device it is
        gone already, and its handle may name another weight of the open one)."""
        del self._map[k]
        if _C.is_open():
            d = _C.get()
            if d.serial == e.serial:
                try:
                    d.free(e.handle)
                except ValueError:
                    pass
                _bump("weights_freed")

    def sweep(self) -> int:
        """Release the device copies of weights whose storage is gone, and drop the entries of closed
        devices. Returns how many entries were removed."""
        with _lock:
            serial = _C.get().serial if _C.is_open() else None
            dead = [(k, e) for k, e in self._map.items() if e.ref.expired() or e.serial != serial]
            for k, e in dead:
                self._release(k, e)
            return len(dead)

    def clear(self) -> int:
        """Release every cached weight on the device. Returns how many."""
        with _lock:
            items = list(self._map.items())
            for k, e in items:
                self._release(k, e)
            return len(items)

    def forget(self) -> None:
        """Drop every entry without touching the device (it is being closed)."""
        with _lock:
            self._map.clear()

    def __len__(self) -> int:
        return len(self._map)

    def entries(self) -> list[dict]:
        with _lock:
            return [dict(name=e.name, N=e.N, K=e.K, dtype=e.dtype, handle=e.handle, alive=not e.ref.expired(),
                         device=e.serial) for e in self._map.values()]


_cache = WeightCache()


def weight_cache() -> WeightCache:
    return _cache


def release_weights() -> int:
    """Free every device weight (they are uploaded again on their next use)."""
    return _cache.clear()


# ------------------------------------------------------------------------------------------
# eager linear
# ------------------------------------------------------------------------------------------
def supported_weight(weight: torch.Tensor) -> str | None:
    """None if `weight` [N, K] can go to Tesseract, else the reason."""
    if not isinstance(weight, torch.Tensor):
        return "not a tensor"
    if weight.device.type != "cpu":
        return f"weight on {weight.device}"
    if weight.dim() != 2:
        return f"weight is {weight.dim()}D"
    if weight.dtype not in SUPPORTED_DTYPES:
        return f"weight dtype {str(weight.dtype).replace('torch.', '')} (float32 / float16 / bfloat16 only)"
    N, K = weight.shape
    if K % BLOCK != 0:
        return f"K % 32 != 0 (K = {K})"
    if N < 1:
        return "N = 0"
    return device().unsupported_reason(weight.dtype, N, K)


def _x_f32(x2: torch.Tensor) -> torch.Tensor:
    if x2.dtype != torch.float32:
        return x2.to(torch.float32, memory_format=torch.contiguous_format)
    return x2


def device_linear_2d(handle: int, x2: torch.Tensor, *, ref: bool = False, name: str = "", dev=None) -> torch.Tensor:
    """y [T, N] float32 = x2 [T, K] @ W^T on the device (or with the reference model if ref). dev: the
    device the handle belongs to (default ``device()``)."""
    dev = device() if dev is None else dev
    x2 = _x_f32(x2)
    if ref:
        return dev.linear_ref(handle, x2)
    y = dev.linear(handle, x2)
    if _verify and y.numel() > 0:
        y_ref = dev.linear_ref(handle, x2)
        a, b = y.view(torch.int32), y_ref.view(torch.int32)
        if not torch.equal(a, b):
            bad = (a != b).nonzero()
            t, n = (int(v) for v in bad[0])
            raise TesseractError(
                f"tesseract verify: {name or 'weight'} [N={y.shape[1]}] T={y.shape[0]}: {bad.shape[0]} of {y.numel()} "
                f"outputs differ from the reference model; first y[{t}, {n}] = {y[t, n].item()!r} "
                f"(0x{a[t, n].item() & 0xFFFFFFFF:08x}), reference {y_ref[t, n].item()!r} (0x{b[t, n].item() & 0xFFFFFFFF:08x}) "
                f"on {dev.description}")
        _bump("verified_calls")
    return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, ref: bool = False,
           name: str = "") -> torch.Tensor:
    """``torch.nn.functional.linear(x, weight, bias)`` with the weight matmul on Tesseract.

    weight [N, K] float32 / float16 / bfloat16 with K % 32 == 0 (uploaded once as Q8_0 and cached);
    x [..., K] any float dtype (the leading dims are flattened into T tokens). The product is computed
    in float32 on the device, the bias is added on the CPU in float32, and the result is returned in
    x's dtype. ``ref=True`` computes the same bits on the host with the reference model (libtsr_ref on
    the weight rows read back from the device)."""
    why = supported_weight(weight)
    if why is not None:
        raise ValueError(f"tesseract_torch.linear: {why}")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("tesseract_torch.linear is inference only: call it under torch.no_grad() / inference_mode()")
    N, K = weight.shape
    if x.shape[-1] != K:
        raise ValueError(f"tesseract_torch.linear: x has {x.shape[-1]} features, weight K = {K}")
    lead = x.shape[:-1]
    dev, h, _ = _cache.acquire(weight, name)
    y = device_linear_2d(h, x.reshape(-1, K), ref=ref, name=name, dev=dev)
    if not ref:
        _bump("device_calls")
        _bump("device_tokens", y.shape[0])
    if bias is not None:
        y.add_(bias.to(torch.float32))
    return y.to(x.dtype).reshape(*lead, N)


def linear_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``linear`` computed on the host by the reference model; bit-identical to ``linear`` on a correct
    device."""
    return linear(x, weight, bias, ref=True)


@contextlib.contextmanager
def reference_linears(enabled: bool = True):
    """Eager execution in which every ``torch.nn.Linear`` with a supported weight runs through
    ``linear_ref`` (the reference model of the device). Inside it an eager model computes what the
    same model compiled with backend="tesseract" must produce, bit for bit, when its CPU ops match
    eager's (functional matmuls such as ``x @ W.t()`` are not covered)."""
    if not enabled:
        yield
        return
    orig = torch.nn.Linear.forward

    def forward(self, x):
        if supported_weight(self.weight) is None:
            return linear_ref(x, self.weight, self.bias)
        return orig(self, x)

    torch.nn.Linear.forward = forward
    try:
        yield
    finally:
        torch.nn.Linear.forward = orig


# ------------------------------------------------------------------------------------------
# counters
# ------------------------------------------------------------------------------------------
def counters() -> dict:
    """Backend counters (see _STAT_KEYS) plus the runtime's (prefixed ``rt_``: linears, tokens,
    execs, prog_hits, prog_misses, perf_cycles, perf_mvm_cycles, dev_read_bytes, h2d_bytes,
    h2d_weight_bytes, d2h_bytes) and the cache size."""
    with _lock:
        r = dict(_stats)
        r["cached_weights"] = len(_cache)
        if _C.is_open():
            for k, v in _C.get().counters().items():
                r["rt_" + k] = v
        return r


def reset_counters() -> None:
    with _lock:
        for k in _stats:
            _stats[k] = 0
        if _C.is_open():
            _C.get().reset_counters()


def format_counters(c: dict | None = None) -> str:
    c = counters() if c is None else c
    width = max(len(k) for k in c)
    return "\n".join(f"  {k:<{width}}  {v}" for k, v in c.items())


_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

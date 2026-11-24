"""Device singleton, weight cache, eager linear and counters of tesseract_torch.

Numerics are the llama.cpp path's (docs/spec.md "Numeric rules", "Load-time conversion"): a float32 / float16 /
bfloat16 weight becomes Q8_0 when it is uploaded (ggml's ``quantize_row_q8_0_ref``, f16 / bf16 widened
exactly first), the activations are quantized on the device by QACT (A32), results are float32.
"""
from __future__ import annotations

import atexit
import contextlib
import gc
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
    "regions",           # full mode: device regions formed by the backend (one op list each)
    "region_calls",      # region calls that ran on the device
    "region_tokens",     # their total T (rows)
    "region_cpu_fallback_calls",  # region calls that ran their aten subgraph on the CPU (placement failed)
    "consts_uploaded",   # constants (norm weights, biases, RoPE tables) uploaded, including re-uploads
    "const_reuploads",   # uploads of a cached constant that changed in place
    "verified_regions",  # region calls checked against the reference executor (TSR_TORCH_VERIFY)
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
    "kv_caches",         # KV cache states made (the device KV planes of one host cache: a pair of K and V tensors)
    "kv_pool_hits",      # states whose planes came from the pool of planes of dead caches
    "kv_resync_rows",    # cache rows stored again from the host cache (it changed, or a call ran on the CPU)
    "kv_region_calls",   # region calls on the device that used a KV slot (cached attention)
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
    """Release every cached weight, constant and region graph and close the device (a later call
    reopens it)."""
    with _lock:
        _cache.forget()
        _consts.forget()
        _kv_forget()
        for rg in _REGIONS:
            rg.forget()
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
    return s.view({1: torch.int8, 2: torch.int16}.get(s.element_size(), torch.int32))


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

    _free_method = "free"        # Device method that releases an entry's handle
    _stat_upload = "uploads"     # counters bumped by an upload
    _stat_reupload = "reuploads"
    _stat_bytes = "upload_bytes"

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
        N, K = int(w.shape[0]), int(w.shape[1])
        return self._acquire(w, self.key(w), N, K, lambda dev: dev.upload(w.detach()), name,
                             str(w.dtype).replace("torch.", ""))

    def acquire_blocks(self, blocks: torch.Tensor, fmt: str, K: int, name: str = "") -> tuple:
        """Like ``acquire`` for ggml blocks ('q8_0' or 'q4_k', uint8 [N, row_bytes]): they are stored on the
        device as they are (``Device.upload_blocks``). The key adds the format and K, since the same bytes
        read as another format are another weight."""
        N = int(blocks.shape[0])
        return self._acquire(blocks, self.key(blocks) + (fmt, K), N, K,
                             lambda dev: dev.upload_blocks(fmt, blocks.detach(), N, K), name, fmt)

    def _acquire(self, w: torch.Tensor, k: tuple, N: int, K: int, upload, name: str, what: str) -> tuple:
        with _lock:
            dev = device()
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
            try:
                h = upload(dev)
            except TesseractError as err:
                if "heap full" not in str(err) and "does not fit" not in str(err):
                    raise
                free_idle_device_memory()   # weights and planes of dead objects fill the same heap
                h = upload(dev)
            sample = None
            if ver is None:
                sample = _sample(w)
                _warn_once("inference-tensor",
                           f"tesseract_torch: weight {name or '[%d x %d]' % (N, K)} is an inference tensor (no version "
                           "counter): an in-place update of it is detected only through a sample of up to "
                           f"{_SAMPLE} x {_SAMPLE} elements on each use; call tesseract_torch.release_weights() after "
                           "changing such a weight in place")
            self._map[k] = _Entry(h, ver, StorageWeakRef(w.untyped_storage()), N, K, w.dtype, name, dev.serial, sample)
            _bump(self._stat_upload)
            if self._stat_bytes:
                _bump(self._stat_bytes, w.numel() * w.element_size())
            if reupload:
                _bump(self._stat_reupload)
            if _debug >= 1 and self._free_method == "free":
                info = dev.info(h)
                log(f"upload {name or 'weight'} [{N} x {K}] {what} -> {info['fmt']}"
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
                    getattr(d, self._free_method)(e.handle)
                except ValueError:
                    pass
                if self._free_method == "free":
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
    """Free every device weight, constant and region graph (they are made again on their next use).
    Returns the number of weights."""
    with _lock:
        for rg in _REGIONS:
            rg.release()
        kv_release_all()
        _consts.clear()
        return _cache.clear()


# ------------------------------------------------------------------------------------------
# constants and regions (full mode)
# ------------------------------------------------------------------------------------------
class ConstCache(WeightCache):
    """Device copies of the constant rows of region programs (norm weights, biases, RoPE tables): the
    same identity rules as the weight cache (storage, offset, shape, stride, dtype, ``_version`` or the
    sample of an inference tensor, device serial). The transform ("identity", or "one_plus" for the
    Gemma norm: the row ``1 + w``) is applied on the host before ``Device.const_upload``."""

    _free_method = "const_free"
    _stat_upload = "consts_uploaded"
    _stat_reupload = "const_reuploads"
    _stat_bytes = ""

    def acquire_const(self, t: torch.Tensor, transform: str, n: int, name: str = "") -> tuple:
        """(device, handle, uploaded) of the constant row t (n elements) after ``transform``."""
        if t.numel() != n:
            raise ValueError(f"tesseract_torch: constant {name or '?'} has {t.numel()} elements, the program expects {n}")
        if transform not in ("identity", "one_plus"):
            raise ValueError(f"tesseract_torch: unknown constant transform {transform!r}")

        def upload(dev):
            x = t.detach().reshape(-1).to(torch.float32)
            if transform == "one_plus":
                x = 1.0 + x
            return dev.const_upload(x.contiguous())

        t2 = t.detach().reshape(1, -1)      # _acquire reads the version and samples a 2D view of the same storage
        return self._acquire(t2, self.key(t) + (transform,), 1, n, upload, name, transform)


_consts = ConstCache()


def const_cache() -> ConstCache:
    return _consts


# ------------------------------------------------------------------------------------------
# KV caches (full mode, cached attention)
# ------------------------------------------------------------------------------------------
# A Hugging Face static cache is a pair of host float32 tensors per layer, [1, Hkv, L, D], that the graph updates in
# place. The device keeps a q8_0 shadow of the pair in a KV object (two planes). The host tensors are always the
# complete and valid cache; the shadow is only a copy, and ``_KVState.n_dev`` says how many cells of it equal the
# quantization of the host rows. Whatever changes the host tensors without the device (``zero_()`` at a reset, a row
# edited in place, a call that ran on the CPU) changes their ``_version`` or leaves rows missing, and the next device
# call stores the missing rows again from the host (``_kv_sync``). A change that does not bump the version counter
# (raw writes into the storage, ``.data``) is not seen; TSR_TORCH_VERIFY compares the shadow with the host rows before
# each call and raises when they differ.


# Every call of every region gets a number of its own (``_begin_call``, process wide). Each thread remembers the number
# of its last call of each region (``_own_call``): the cache writes that complete a device step are the writes that the
# thread of that region call runs after it, so the step is matched by that thread's number alone. Another thread's call
# of the same region (a second generation through one compiled model) moves the counter, not this thread's record.
_call_counter = 0
_calls_of_thread = threading.local()


def _begin_call(rid: int) -> int:
    """Number a call of region ``rid`` that begins on this thread (call with ``_lock`` held), on the device or not."""
    global _call_counter
    _call_counter += 1
    mine = getattr(_calls_of_thread, "last", None)
    if mine is None:
        mine = _calls_of_thread.last = {}
    mine[rid] = _call_counter
    return _call_counter


def _own_call(rid: int) -> int | None:
    """The number of this thread's last call of region ``rid``; None when this thread never called it."""
    return getattr(_calls_of_thread, "last", {}).get(rid)


@dataclass
class _KVPending:
    """The device step of a state whose cache writes (``kv_rows_write``) have not all run yet. Only the writes that the
    thread of the region call runs after that very call complete it: ``call`` is the number of that call
    (``_begin_call``, unique over all regions and threads). The writes of any other call at the same cells (a later call
    that ran on the CPU, a call of another thread on the same cache) are not its writes."""
    p0: int
    T: int
    seen: dict          # "k" / "v" -> True when its write was recorded against the version of the step, False when not
    call: int | None


@dataclass
class _KVState:
    """The device shadow of one cache (a K and a V tensor)."""
    handle: int
    serial: int                    # the device the handle belongs to
    E: int
    cells: int
    key_k: tuple
    key_v: tuple
    ref_k: StorageWeakRef
    ref_v: StorageWeakRef
    k_off: int
    n_dev: int = 0                 # cells [0, n_dev) of the planes hold the device's quantization of the host rows
    ver_k: int = -1                # the tensors' _version for which that holds, before the bump that the AOT wrapper
    ver_v: int = -1                # adds at the start of each call (``RegionRT.kv_bumps``); -1: none yet
    pending: _KVPending | None = None
    name: str = ""

    def dead(self) -> bool:
        return self.ref_k.expired() or self.ref_v.expired()


_kv_states: dict = {}      # (key_k, key_v) -> _KVState
_kv_by_base: dict = {}     # key of a base tensor -> its _KVState
_kv_pool: dict = {}        # (E, cells) -> [(k_off, handle, serial)] planes of dead caches, free to reuse; the keys are in
#                            order of use, the least recently used first
# Hugging Face ``generate`` makes its cache as long as the prompt plus the new tokens, so every prompt of another length
# is a cache of another geometry and the planes of the lengths before it are never used again. The pool keeps the planes
# of the most recent geometries only; the planes of the others go back to the device.
_KV_POOL_GEOMETRIES = 4


def _kv_pool_take(E: int, cells: int, serial: int):
    """The free planes of this geometry with the highest offset (a cache made again gets the planes of the one
    before it), or None."""
    lst = _kv_pool.get((E, cells))
    if not lst:
        return None
    lst.sort()
    got = None
    while lst and got is None:
        off, h, ser = lst.pop()
        if ser == serial:
            got = h, off
    del _kv_pool[(E, cells)]
    if lst:
        _kv_pool[(E, cells)] = lst            # the geometry is the most recent one
    return got


def _kv_pool_put(st: "_KVState") -> None:
    """Pool the planes of a state under its geometry (the most recent one); the planes of the least recently used
    geometries beyond ``_KV_POOL_GEOMETRIES`` are freed on the device."""
    geo = (st.E, st.cells)
    lst = _kv_pool.pop(geo, [])
    lst.append((st.k_off, st.handle, st.serial))
    _kv_pool[geo] = lst
    while len(_kv_pool) > _KV_POOL_GEOMETRIES:
        old = _kv_pool.pop(next(iter(_kv_pool)))
        if _C.is_open():
            d = _C.get()
            for _, h, ser in old:
                if ser == d.serial:             # planes of another device are gone with it
                    try:
                        d.kv_free(h)
                    except ValueError:
                        pass


def _kv_drop(st: _KVState, to_pool: bool) -> None:
    """Forget a state; its planes go to the pool (or are freed)."""
    _kv_states.pop((st.key_k, st.key_v), None)
    for k in (st.key_k, st.key_v):
        if _kv_by_base.get(k) is st:
            del _kv_by_base[k]
    if not (_C.is_open() and _C.get().serial == st.serial):
        return                      # the device is gone: so are the planes
    if to_pool:
        _kv_pool_put(st)
    else:
        try:
            _C.get().kv_free(st.handle)
        except ValueError:
            pass


def kv_sweep() -> int:
    """Give the planes of the caches whose tensors died to the pool. Returns how many states."""
    with _lock:
        dead = [st for st in _kv_states.values() if st.dead()]
        for st in dead:
            _kv_drop(st, True)
        return len(dead)


def kv_release_pool() -> int:
    """Free the pooled planes (of dead caches) on the device. Returns how many."""
    with _lock:
        n = 0
        if _C.is_open():
            d = _C.get()
            for lst in _kv_pool.values():
                for _, h, ser in lst:
                    if ser == d.serial:
                        try:
                            d.kv_free(h)
                            n += 1
                        except ValueError:
                            pass
        _kv_pool.clear()
        return n


def free_idle_device_memory() -> None:
    """Free what the device holds for nothing, before a retry of an allocation that found the heap full (call with
    ``_lock`` held): the device copies of weights and constants whose tensors died, the planes of caches whose tensors
    died (a finished ``generate`` leaves its cache to the cyclic garbage collector) and the pooled planes."""
    gc.collect()
    _cache.sweep()
    _consts.sweep()
    kv_sweep()
    kv_release_pool()


def kv_release_all() -> int:
    """Free every KV object (states and pool) on the device. Returns how many states there were."""
    with _lock:
        n = len(_kv_states)
        for st in list(_kv_states.values()):
            _kv_drop(st, False)
        kv_release_pool()
        return n


def _kv_forget() -> None:
    """Drop every state and the pool without touching the device (it is being closed)."""
    _kv_states.clear()
    _kv_by_base.clear()
    _kv_pool.clear()


def kv_caches() -> list[dict]:
    """The KV caches the device holds a shadow of: cells, device cells in use (``n_dev``), alive, pooled planes."""
    with _lock:
        out = [dict(name=st.name, E=st.E, cells=st.cells, n_dev=st.n_dev, alive=not st.dead(), handle=st.handle,
                    k_off=st.k_off) for st in _kv_states.values()]
        out += [dict(name="(pool)", E=E, cells=c, n_dev=0, alive=False, handle=h, k_off=off)
                for (E, c), lst in _kv_pool.items() for off, h, _ in lst]
        return out


def q8_0_rows(x: torch.Tensor) -> torch.Tensor:
    """ggml's ``quantize_row_q8_0_ref`` of the rows of x [n, E] (E % 32 == 0) as block_q8_0 bytes, uint8 [n, E / 32 * 34]
    (fp16 d = amax / 127, q = roundf(x * (1 / d)) half away from zero). Written from the definition with torch
    float32 / float64 arithmetic: it shares no code with the device path or the simulator (the check of
    TSR_TORCH_VERIFY compares the device's cells against it)."""
    n, E = int(x.shape[0]), int(x.shape[1])
    nb = E // 32
    xb = x.detach().to(torch.float32).reshape(n, nb, 32)
    d = xb.abs().amax(-1) / 127.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    x0 = (xb * idv.unsqueeze(-1)).to(torch.float64)
    q = (torch.sign(x0) * torch.floor(x0.abs() + 0.5)).to(torch.int8)
    out = torch.empty((n, nb, 34), dtype=torch.uint8)
    out[:, :, 0:2] = d.to(torch.float16).view(torch.uint8).reshape(n, nb, 2)
    out[:, :, 2:34] = q.view(torch.uint8)
    return out.reshape(n, nb * 34)


def _kv_refuse(rg: "RegionRT", why: str):
    raise TesseractError(f"tesseract: {rg.name}: a cached attention does not fit the device: {why}")


def _kv_rows_host(B: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Cells [a, b) of the head-major cache B [1, Hkv, L, D] as token-major rows [b - a, Hkv * D]."""
    return B[0, :, a:b, :].transpose(0, 1).reshape(b - a, -1)


def _kv_state(rg: "RegionRT", dev, slot: int, K: torch.Tensor, V: torch.Tensor, E: int, cells: int) -> _KVState:
    """The state of the cache (K, V): found, or made (planes from the pool, else ``kv_create``)."""
    kk, kv_ = WeightCache.key(K), WeightCache.key(V)
    st = _kv_states.get((kk, kv_))
    if st is not None and (st.dead() or st.serial != dev.serial or st.E != E or st.cells != cells):
        _kv_drop(st, True)
        st = None
    if st is not None:
        return st
    kv_sweep()
    for stale in [x for x in _kv_states.values() if x.serial != dev.serial]:
        _kv_drop(stale, False)
    got = _kv_pool_take(E, cells, dev.serial)
    if got is not None:
        h, off = got
        _bump("kv_pool_hits")
    else:
        try:
            h = dev.kv_create(E, cells)
        except TesseractError as err:
            if "does not fit" not in str(err) and "heap full" not in str(err):
                raise
            free_idle_device_memory()
            h = dev.kv_create(E, cells)
        except ValueError as err:
            _kv_refuse(rg, f"slot {slot}: {err}")
        off = int(dev.kv_info(h)["k_off"])
    st = _KVState(h, dev.serial, E, cells, kk, kv_, StorageWeakRef(K.untyped_storage()),
                  StorageWeakRef(V.untyped_storage()), off, 0, -1, -1, None, f"{rg.name} slot {slot}")
    _kv_states[(kk, kv_)] = st
    _kv_by_base[kk] = _kv_by_base[kv_] = st
    _bump("kv_caches")
    if _debug >= 2:
        log(f"kv cache for {rg.name} slot {slot}: E = {E}, {cells} cells (planes at offset {off})")
    return st


def _kv_check_call(rg: "RegionRT", kvargs, T: int) -> tuple[int, list]:
    """Before the program of a region with KV slots is made (no device needed): check the call. Returns (p0, cells):
    the first cell of the call and, per slot, the cells of the cache tensors of this call (the compile-time cells of a
    static slot, the length of the K tensor for a slot whose length is a symbol of the graph). A call that the device
    cannot take raises a TesseractError that ``region_placement_error`` recognises (it then runs on the CPU)."""
    slots = rg.region.kv
    p0 = None
    seen = set()
    cells = []
    for s, kvc in enumerate(slots):
        K, V, pos = kvargs[3 * s], kvargs[3 * s + 1], kvargs[3 * s + 2]
        for what, B in (("K", K), ("V", V)):
            if not isinstance(B, torch.Tensor) or B.dtype != torch.float32 or B.device.type != "cpu":
                _kv_refuse(rg, f"slot {s}: the {what} cache is not a float32 CPU tensor")
        if kvc.cells is not None:
            n_cells = kvc.cells
        elif K.dim() == 4:
            n_cells = int(K.shape[2])
        else:
            _kv_refuse(rg, f"slot {s}: the K cache is {tuple(K.shape)}, expected [1, {kvc.hkv}, cells, {kvc.d}]")
        cells.append(n_cells)
        for what, B in (("K", K), ("V", V)):
            if tuple(B.shape) != (1, kvc.hkv, n_cells, kvc.d):
                _kv_refuse(rg, f"slot {s}: the {what} cache is {tuple(B.shape)}, expected {(1, kvc.hkv, n_cells, kvc.d)}")
            if not B.is_contiguous():
                _kv_refuse(rg, f"slot {s}: the {what} cache is not contiguous")
            if _version_of(B) is None:
                _kv_refuse(rg, f"slot {s}: the {what} cache is an inference tensor (no version counter, so a change of it "
                               "cannot be seen); make and call the model under torch.no_grad()")
            k = WeightCache.key(B)
            if k in seen:
                _kv_refuse(rg, f"slot {s}: one tensor is the cache of two slots")
            seen.add(k)
        if not isinstance(pos, torch.Tensor) or pos.dim() != 1 or int(pos.numel()) != T or pos.dtype != torch.int64:
            _kv_refuse(rg, f"slot {s}: the cache positions are not an int64 [{T}] tensor")
        q0 = int(pos[0])
        if q0 < 0 or not torch.equal(pos, torch.arange(q0, q0 + T, dtype=torch.int64)):
            _kv_refuse(rg, f"slot {s}: the cache positions are not consecutive from one cell")
        if p0 is None:
            p0 = q0
        elif q0 != p0:
            _kv_refuse(rg, f"slot {s}: the slots start at different cells ({q0} and {p0})")
        if kvc.causal0 and q0 != 0:
            _kv_refuse(rg, f"slot {s}: the attention is causal over the first {T} cells of the cache and the call starts "
                           f"at cell {q0} (this graph is for a fresh cache)")
        if q0 + T > n_cells:
            _kv_refuse(rg, f"slot {s}: cells {q0} .. {q0 + T} do not fit the {n_cells} cells of the cache")
    return p0, cells


def _kv_sync(rg: "RegionRT", dev, kvargs, p0: int, cells: list) -> tuple[list, list]:
    """After the device graph of the region exists (call with _lock held, after ``_kv_check_call``): find or make the
    state of each slot, store the host rows that the planes lack. Returns (handles, states). A cache of another
    length than the last call is another pair of tensors: it gets its own state and planes (pooled by (E, cells))."""
    slots = rg.region.kv
    states = []
    for s, kvc in enumerate(slots):
        K, V = kvargs[3 * s], kvargs[3 * s + 1]
        st = _kv_state(rg, dev, s, K, V, kvc.hkv * kvc.d, cells[s])
        states.append(st)
    for s, (kvc, st) in enumerate(zip(slots, states)):
        K, V = kvargs[3 * s], kvargs[3 * s + 1]
        vk, vv = _version_of(K), _version_of(V)
        wk, wv = rg.kv_bumps[s]
        if (vk - wk, vv - wv) != (st.ver_k, st.ver_v):
            if _debug >= 2 and st.n_dev:
                log(f"kv {rg.name} slot {s}: versions {(vk, vv)} against the recorded {(st.ver_k, st.ver_v)} "
                    f"+ {(wk, wv)} (n_dev {st.n_dev})")
            st.n_dev = 0              # something changed the host cache: the shadow is not trusted
        if st.n_dev > p0:
            st.n_dev = p0
        if st.n_dev < p0:
            a = st.n_dev
            try:
                dev.kv_store(st.handle, a, _kv_rows_host(K, a, p0), _kv_rows_host(V, a, p0))
            except ValueError as err:
                _kv_refuse(rg, f"slot {s}: {err}")
            _bump("kv_resync_rows", p0 - a)
            if _debug >= 1:
                log(f"kv resync {rg.name} slot {s}: stored the host rows [{a}, {p0})")
            st.n_dev = p0
        st.ver_k, st.ver_v = vk - wk, vv - wv
        st.pending = None
    return [st.handle for st in states], states


def _kv_prepare(rg: "RegionRT", dev, kvargs, T: int) -> tuple[list, int, list]:
    """``_kv_check_call`` and ``_kv_sync`` in one step: (handles, p0, states)."""
    p0, cells = _kv_check_call(rg, kvargs, T)
    kvh, states = _kv_sync(rg, dev, kvargs, p0, cells)
    return kvh, p0, states


def _kv_check_shadow(rg: "RegionRT", dev, states, kvargs, p0: int) -> None:
    """TSR_TORCH_VERIFY, after ``_kv_sync`` and before the region runs: the cells [0, p0) of every slot's K and V
    plane against ggml's quantizer of the host rows [0, p0), byte for byte. Every cell below ``n_dev`` was stored from
    the float32 rows that the host cache holds (a resync) or by the device from rows it returned and the host cache
    took in place, so the check is exact: a difference is a shadow that the state machine wrongly trusts. The reference
    executor reads these cells from the device, so it cannot see this. Each plane is read once."""
    if p0 == 0:
        return
    for s, st in enumerate(states):
        kb, vb = dev.kv_read(st.handle, 0, p0)
        for which, got, B in (("K", kb, kvargs[3 * s]), ("V", vb, kvargs[3 * s + 1])):
            want = q8_0_rows(_kv_rows_host(B, 0, p0))
            if got.shape != want.shape or not torch.equal(got, want):
                bad = (got != want).nonzero()
                t = int(bad[0][0]) if bad.numel() else 0
                blk = int(bad[0][1]) // 34 if bad.numel() else 0
                st.n_dev = 0                     # the shadow is not what the state said: store it again
                raise TesseractError(
                    f"tesseract verify: {rg.name}: the {which} plane of KV slot {s} differs from the quantized host rows "
                    f"before the call: cell {t} (of the first {p0}), first block {blk}: device bytes "
                    f"{got[t, blk * 34:blk * 34 + 34].tolist()}, expected {want[t, blk * 34:blk * 34 + 34].tolist()}; the "
                    f"device shadow is stale (the cache changed without the device seeing it) on {dev.description}")


def _kv_rows_of(rg: "RegionRT", outs, inputs, slot: int, which: str) -> torch.Tensor:
    """The [T, E] rows the region stored into slot ``slot`` (its K or V rows: an output of the region or an input)."""
    r = rg.region
    op = next(o for o in r.ops if o.get("kv") == slot)
    vid = op["k" if which == "K" else "v"]
    for spec, x in zip(r.inputs, inputs):
        if spec.value == vid and spec.kind == "in":
            return x
    vals = sorted({o.value for o in r.outputs})
    return outs[vals.index(vid)]


def _kv_check(rg: "RegionRT", dev, states, p0: int, T: int, outs, inputs) -> None:
    """TSR_TORCH_VERIFY: the cells [p0, p0 + T) of every slot against ggml's quantizer of the rows the region
    returned, byte for byte."""
    for s, st in enumerate(states):
        kb, vb = dev.kv_read(st.handle, p0, T)
        for which in ("K", "V"):
            rows = _kv_rows_of(rg, outs, inputs, s, which)
            want = q8_0_rows(rows)
            got = kb if which == "K" else vb
            if got.shape != want.shape or not torch.equal(got, want):
                bad = (got != want).nonzero()
                t = int(bad[0][0]) if bad.numel() else 0
                blk = int(bad[0][1]) // 34 if bad.numel() else 0
                raise TesseractError(
                    f"tesseract verify: {rg.name}: the {which} plane of KV slot {s} differs from the quantized rows: cell "
                    f"{p0 + t} (row {t}), first block {blk}: device bytes {got[t, blk * 34:blk * 34 + 34].tolist()}, "
                    f"expected {want[t, blk * 34:blk * 34 + 34].tolist()} on {dev.description}")


def kv_rows_write(rid: int, slot: int, which: str, B: torch.Tensor, pos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The cache update of a cached attention: ``B.index_copy_(2, pos, x)`` in place (B [1, Hkv, L, D], x the new
    head-major rows [1, Hkv, T, D]); returns B. The rewritten graph runs it after the region call, in place of the
    functional ``index_copy`` and of the ``copy_`` of the new cache into its input. It also keeps the state of the
    device shadow: when the write follows the device step of this very call, the cells [p0, p0 + T) of the shadow now
    equal the host rows (``n_dev`` advances once K and V are written); "this very call" is the region call that this
    thread made last (``_own_call``). Any other write (a call that ran on the CPU, a call of another thread, or a
    device step whose own writes never ran) leaves the shadow short of the cells it wrote: ``n_dev`` goes down to the
    smallest of them (the positions of a call on the CPU need not be sorted), and the next device call stores the
    rows. A write of no rows changes no cell and leaves ``n_dev`` and the pending step alone."""
    with _lock:
        before = _version_of(B)
        B.index_copy_(2, pos, x)
        after = _version_of(B)
        st = _kv_by_base.get(WeightCache.key(B))
        if st is None or st.dead() or not _C.is_open() or _C.get().serial != st.serial or before is None:
            return B
        attr = "ver_k" if which == "k" else "ver_v"
        if (st.key_k if which == "k" else st.key_v) != WeightCache.key(B):
            return B
        rg = _REGIONS[rid]
        w = rg.kv_bumps[slot][0 if which == "k" else 1]
        # nothing changed the tensor since the shadow was checked: its version is the recorded one plus the bump of the
        # wrapper of this call (a call that ran on the CPU did not record anything; a device call did, minus that bump)
        ok = getattr(st, attr) + w == before
        if ok:
            setattr(st, attr, after)
        if int(pos.numel()) == 0:
            return B
        pend = st.pending
        p0 = int(pos[0])               # a device step's positions are consecutive (``_kv_check_call``)
        T = int(x.shape[2])
        if (pend is not None and pend.call is not None and pend.call == _own_call(rid) and pend.p0 == p0
                and pend.T == T and which not in pend.seen):
            pend.seen[which] = ok
            if len(pend.seen) == 2:
                if all(pend.seen.values()):
                    st.n_dev = p0 + T
                st.pending = None
        else:
            st.pending = None
            st.n_dev = min(st.n_dev, int(pos.min()))
    return B



class _GraphRefused(TesseractError):
    """The device refused a whole region program at graph_create (the per-op probe accepted every op). The
    region runs on the CPU for that key, like a placement failure."""


def region_placement_error(err: BaseException) -> bool:
    """True if err says that a region does not fit the device (T above a limit, a heap or the IO region
    full) or that the device refused its program. Other errors are raised."""
    return isinstance(err, _GraphRefused) or isinstance(err, TesseractError) and ("does not fit" in str(err) or "heap full" in str(err))


class RegionRT:
    """Run-time state of one region: its lowering result (``lower.Region``), the aten subgraph for the CPU
    fallback and the C++ graph with the key (handles and scalars) it was made for."""

    def __init__(self, region, fallback, n_ext: int, gid: int, name: str) -> None:
        self.region, self.fallback, self.n_ext, self.gid, self.name = region, fallback, n_ext, gid, name
        self.id = -1
        self.calls = self.tokens = self.cpu_calls = 0
        self.graph: int | None = None
        self.key: tuple | None = None
        self.failed: tuple | None = None       # (key, reason): the graph does not fit for this key
        self.serial = -1
        self.logged = False
        # per KV slot: how many times the AOTAutograd wrapper bumps the version of the K and V cache input before
        # each call of the compiled graph (it does, once, for an input the graph mutates, so that autograd sees the
        # mutation that the graph does itself); (0, 0) for a cache made in the graph
        self.kv_bumps: list = [(0, 0)] * len(getattr(region, "kv", ()))
        # mask value -> the KV slot of the cached attention that reads it (the lowering gives all readers of one mask the
        # same cells): ``_device_run`` checks the shape of these masks against the cells of the slot
        self.mask_slot: dict = {}
        for o in getattr(region, "ops", ()):
            if o.get("op") == "attn" and o.get("kv", -1) >= 0 and o.get("mask") is not None:
                self.mask_slot.setdefault(o["mask"], o["kv"])

    def _free_graph(self) -> None:
        if self.graph is not None and _C.is_open():
            d = _C.get()
            if d.serial == self.serial:
                try:
                    d.graph_free(self.graph)
                except ValueError:
                    pass
        self.graph = self.key = None

    def release(self) -> None:
        self._free_graph()
        self.failed = None

    def forget(self) -> None:
        self.graph = self.key = self.failed = None

    def describe(self) -> str:
        r = self.region
        return f"{self.name} ({len(r.ops)} ops, {len(r.weights)} weights)"


_REGIONS: list[RegionRT] = []


def register_region(rg: RegionRT) -> int:
    with _lock:
        rg.id = len(_REGIONS)
        _REGIONS.append(rg)
        return rg.id


def release_regions(rgs: list[RegionRT]) -> None:
    """Free the device graphs of regions whose compiled function is gone."""
    with _lock:
        for rg in rgs:
            rg.release()


def regions() -> list[RegionRT]:
    """Every region formed so far (with per-region call, token and CPU-fallback counts)."""
    return list(_REGIONS)


def _first_diff(a: torch.Tensor, b: torch.Tensor):
    ia, ib = a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    bad = (ia != ib).nonzero()
    t, n = (int(v) for v in bad[0])
    return int(bad.shape[0]), t, n, a[t, n].item(), b[t, n].item(), ia[t, n].item() & 0xFFFFFFFF, ib[t, n].item() & 0xFFFFFFFF


def _check_f32(x, what: str, rg: RegionRT, want=torch.float32) -> None:
    if not isinstance(x, torch.Tensor) or x.dtype != want or x.device.type != "cpu":
        got = f"{x.dtype} on {x.device}" if isinstance(x, torch.Tensor) else type(x).__name__
        raise TesseractError(f"tesseract: {rg.name}: {what} must be a {str(want).replace('torch.', '')} CPU tensor, got {got}")


def _device_run(rg: RegionRT, ins, cs, ws, ss, kvargs=()) -> list:
    """The outputs of region ``rg`` on the device (call with _lock held). Raises TesseractError; a
    placement failure or a refused program is recognised by ``region_placement_error``.

    Mask rule: the device takes mask values 0 and -inf. In a float32 mask every value <= -1e9 (Hugging Face
    uses finfo.min or -1e9) is replaced by -inf before the run: exp of a score plus such a value is 0 in
    float32, as with -inf. A row of such values and no larger one (every key masked, as a left-padded
    query row) does not fit the device: eager gives a uniform softmax over it, the device would not. The
    call is refused with a placement error and runs on the CPU; a row that is all -inf is not refused.
    A mask without such a value is passed as it is. Any other non-zero value is a placement error of the
    device and the call runs on the CPU."""
    r = rg.region
    inputs = []
    T = None
    for spec, x in zip(r.inputs, ins):
        if spec.kind == "in":
            _check_f32(x, f"input {spec.node.name}", rg)
            x = x.contiguous()
            if x.dim() != 2:
                raise TesseractError(f"tesseract: {rg.name}: input {spec.node.name} must be [T, n], got {tuple(x.shape)}")
            if T is None:
                T = int(x.shape[0])
        elif spec.kind == "mask":
            if x.dtype not in (torch.bool, torch.float32):
                raise TesseractError(f"tesseract: {rg.name}: the mask must be bool or float32, got {x.dtype}")
            x = x.contiguous()
            if x.dtype == torch.float32:
                big = (x <= -1e9) & ~torch.isneginf(x)
                if bool(big.any()):
                    top = x.amax(-1)
                    if bool(((top <= -1e9) & ~torch.isneginf(top)).any()):
                        raise TesseractError(f"tesseract: {rg.name}: a row of the mask has every key masked with a finite "
                                             "value (<= -1e9); eager gives a uniform softmax there, which the device does "
                                             "not: the call does not fit the device")
                    x = x.masked_fill(big, float("-inf"))
        else:
            x = x.contiguous()
        inputs.append(x)
    if not T:
        raise TesseractError(f"tesseract: {rg.name}: T = {T} does not fit (the device needs T >= 1)")
    dev = device()
    wh = []
    for spec, w in zip(r.weights, ws):
        if spec.kind == "blocks":
            _check_f32(w, f"weight {spec.node.name}", rg, torch.uint8)
            wh.append(_cache.acquire_blocks(w, spec.fmt, spec.K, name=spec.node.name)[1])
        else:
            _check_f32(w, f"weight {spec.node.name}", rg)
            wh.append(_cache.acquire(w, spec.node.name)[1])
    ch = []
    for spec, c in zip(r.consts, cs):
        _check_f32(c, f"constant {spec.node.name}", rg)
        ch.append(_consts.acquire_const(c, spec.transform, spec.n, spec.node.name)[1])
    sv = [float(v) for v in ss]
    key = (dev.serial, tuple(wh), tuple(ch), tuple(sv))
    p0, cells = None, None
    if r.kv:
        # the cells of every slot at this call, from the cache tensors; checked before anything uses their shapes. The
        # region has one device graph at a time: a call with another cache length (a symbolic slot) frees it and makes
        # the graph of that length, as a changed weight handle does
        p0, cells = _kv_check_call(rg, kvargs, T)
        for spec, x in zip(r.inputs, inputs):
            sl = rg.mask_slot.get(spec.value) if spec.kind == "mask" else None
            if sl is not None:
                shp = tuple(x.shape)
                if len(shp) < 2 or shp[-1] != cells[sl] or shp[-2] != T or any(d != 1 for d in shp[:-2]):
                    _kv_refuse(rg, f"the attention mask of slot {sl} is {shp}, expected [T = {T}, {cells[sl]}] "
                                   "(leading dimensions of 1 allowed)")
        key += (tuple(cells),)
    if rg.key != key:
        rg._free_graph()
        if rg.failed is not None and rg.failed[0] == key:
            raise _GraphRefused(rg.failed[1])
        prog = r.to_program(sv, cells)
        for v in prog["values"]:
            if v["kind"] == "const":
                v["handle"] = ch[v.pop("source")]
        for o in prog["ops"]:
            if o["op"] == "mul_mat":
                o["w"] = wh[o["w"]]
        try:
            try:
                rg.graph = dev.graph_create(prog)
            except TesseractError as err:
                if "WEIGHT heap" not in str(err):
                    raise
                free_idle_device_memory()           # planes of dead caches sit in the heap that the program needs
                rg.graph = dev.graph_create(prog)
        except TesseractError as err:
            rg.failed = (key, str(err))
            raise _GraphRefused(str(err)) from err
        rg.key, rg.serial, rg.failed = key, dev.serial, None
    h = rg.graph
    if r.kv:
        kvh, states = _kv_sync(rg, dev, kvargs, p0, cells)
        if _verify:
            _kv_check_shadow(rg, dev, states, kvargs, p0)
        outs = dev.graph_run(h, inputs, kv=kvh, pos0=p0)
    else:
        outs = dev.graph_run(h, inputs)
    if _debug >= 1 and not rg.logged:
        rg.logged = True
        try:                                # the log never changes what runs: an error here is a line of the log
            info = dev.graph_info(h, T)
            kinds = ", ".join(f"{k} x{v}" for k, v in sorted(info["by_kind"].items()))
            rows = int(info.get("T", T))
            first = f"first plan T={T}" if rows == T else f"first plan: chunk of {rows} rows of {T}"
            log(f"{rg.name}: {first}: {info['records']} records in {info['lists']} list(s) ({kinds}), "
                f"{info['io_bytes'] / 2**20:.2f} MiB of IO")
        except (TesseractError, ValueError) as err:
            log(f"{rg.name}: no plan information for T={T}: {err}")
    if _debug >= 2:
        log(f"call {rg.name} T={T}")
    if _verify:
        refs = dev.graph_run_ref(h, inputs, kv=kvh, pos0=p0) if r.kv else dev.graph_run_ref(h, inputs)
        names = {}
        for o in r.outputs:
            names.setdefault(o.value, o.node.name)
        for k, (y, y_ref) in enumerate(zip(outs, refs)):
            if y.shape != y_ref.shape or not torch.equal(y.view(torch.int32), y_ref.view(torch.int32)):
                if y.shape != y_ref.shape:
                    raise TesseractError(f"tesseract verify: {rg.name} output {k}: shape {tuple(y.shape)} against the "
                                         f"reference {tuple(y_ref.shape)}")
                nbad, t, n, vy, vr, by, br = _first_diff(y, y_ref)
                node = names[sorted(names)[k]]
                raise TesseractError(
                    f"tesseract verify: {rg.name} output {k} ({node}) T={T}: {nbad} of {y.numel()} values differ from the "
                    f"reference executor; first y[{t}, {n}] = {vy!r} (0x{by:08x}), reference {vr!r} (0x{br:08x}) on {dev.description}")
        _bump("verified_regions")
        if r.kv:
            _kv_check(rg, dev, states, p0, T, outs, inputs)
    if r.kv:
        for st in states:
            st.pending = _KVPending(p0, T, {}, _own_call(rg.id))
        _bump("kv_region_calls")
    rg.calls += 1
    rg.tokens += T
    _bump("region_calls")
    _bump("region_tokens", T)
    return _per_output(r, outs)


def _share_key(o):
    return (o.value, o.sroot if o.sroot is not None else o.node)


def _per_output(r, outs: list) -> list:
    """The device returns one tensor per distinct out value, in value order. A region has one output per
    node; the outputs of a value that share storage in the aten graph (one ``sroot``) get one tensor object,
    so that their views share storage. A group of the value that eager separates by a copy gets a copy."""
    vals = sorted({o.value for o in r.outputs})
    pos = {v: k for k, v in enumerate(vals)}
    given: dict = {}
    used: set = set()
    res = []
    for o in r.outputs:
        key = _share_key(o)
        if key not in given:
            y = outs[pos[o.value]]
            given[key] = y.clone() if o.value in used else y
            used.add(o.value)
        res.append(given[key])
    return res


def _cpu_run(rg: RegionRT, ext) -> list:
    """The outputs of region ``rg`` from its saved aten subgraph, as the [T, n] rows the device gives (one
    tensor object for the outputs that share storage, as ``_per_output``)."""
    res = rg.fallback(*ext)
    outs = []
    given: dict = {}
    for o, y in zip(rg.region.outputs, res):
        key = _share_key(o)
        if key not in given:
            if o.recipe["transpose12"]:
                y = y.transpose(1, 2)
            given[key] = y.reshape(-1, rg.region.values[o.value]["n"])
        outs.append(given[key])
    return outs


def device_region(rid: int, *args):
    """The device form of one region of the aten graph (full mode). args: the flattened inputs, the
    constants, the weight roots, the scalars (0-dim tensors), for each KV slot the K base, the V base and the
    positions, and the nodes the saved aten subgraph needs.
    Returns the tuple of [T, n] float32 outputs."""
    rg = _REGIONS[rid]
    r = rg.region
    ni, nc, nw, ns, nk = len(r.inputs), len(r.consts), len(r.weights), len(r.scalars), 3 * len(r.kv)
    ins, cs, ws = args[:ni], args[ni:ni + nc], args[ni + nc:ni + nc + nw]
    ss, kvargs = args[ni + nc + nw:ni + nc + nw + ns], args[ni + nc + nw + ns:ni + nc + nw + ns + nk]
    ext = args[ni + nc + nw + ns + nk:]
    with _lock:
        _begin_call(rid)           # every call is numbered, also one that the device refuses and that runs on the CPU
        try:
            return tuple(_device_run(rg, ins, cs, ws, ss, kvargs))
        except TesseractError as err:
            if not region_placement_error(err):
                raise
            rg.cpu_calls += 1
            _bump("region_cpu_fallback_calls")
            _warn_once(f"region:{rid}", f"tesseract: {rg.name} runs on the CPU for a call that the device does not take: {err}")
    return tuple(_cpu_run(rg, ext))


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
        r["cached_consts"] = len(_consts)
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

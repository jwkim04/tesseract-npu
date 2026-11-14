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
                self.sweep()
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


def _device_run(rg: RegionRT, ins, cs, ws, ss) -> list:
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
    if rg.key != key:
        rg._free_graph()
        if rg.failed is not None and rg.failed[0] == key:
            raise _GraphRefused(rg.failed[1])
        prog = r.to_program(sv)
        for v in prog["values"]:
            if v["kind"] == "const":
                v["handle"] = ch[v.pop("source")]
        for o in prog["ops"]:
            if o["op"] == "mul_mat":
                o["w"] = wh[o["w"]]
        try:
            rg.graph = dev.graph_create(prog)
        except TesseractError as err:
            rg.failed = (key, str(err))
            raise _GraphRefused(str(err)) from err
        rg.key, rg.serial, rg.failed = key, dev.serial, None
    h = rg.graph
    outs = dev.graph_run(h, inputs)
    if _debug >= 1 and not rg.logged:
        rg.logged = True
        info = dev.graph_info(h, T)
        kinds = ", ".join(f"{k} x{v}" for k, v in sorted(info["by_kind"].items()))
        log(f"{rg.name}: first plan T={T}: {info['records']} records in {info['lists']} list(s) ({kinds}), "
            f"{info['io_bytes'] / 2**20:.2f} MiB of IO")
    if _debug >= 2:
        log(f"call {rg.name} T={T}")
    if _verify:
        refs = dev.graph_run_ref(h, inputs)
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
    constants, the weight roots, the scalars (0-dim tensors) and the nodes the saved aten subgraph needs.
    Returns the tuple of [T, n] float32 outputs."""
    rg = _REGIONS[rid]
    r = rg.region
    ni, nc, nw, ns = len(r.inputs), len(r.consts), len(r.weights), len(r.scalars)
    ins, cs, ws = args[:ni], args[ni:ni + nc], args[ni + nc:ni + nc + nw]
    ss, ext = args[ni + nc + nw:ni + nc + nw + ns], args[ni + nc + nw + ns:]
    with _lock:
        try:
            return tuple(_device_run(rg, ins, cs, ws, ss))
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

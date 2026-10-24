"""The torch.compile backend "tesseract".

Pipeline: Dynamo graph -> AOTAutograd (inference graph of aten ops; parameters and buffers arrive
as graph inputs, also under ``inline_inbuilt_nn_modules``) -> partition: every weight matmul that
Tesseract can run is replaced by a call of ``device_mm`` -> the rewritten aten graph runs eagerly on
the CPU (no Inductor), with those matmuls on the device.

A matmul node goes to the device if
  - it is ``aten.mm`` / ``aten.addmm`` / ``aten.linear`` / ``aten.matmul`` / ``aten.mv`` (``bmm`` and
    batched weights stay on the CPU),
  - one operand is a weight: a parameter, a registered buffer or a graph constant (``get_attr``),
    possibly through views (``t``, ``transpose``, ``permute``, ``view``, ``slice``, ...), 2D [N, K]
    after orienting it, and the other operand is not,
  - the weight is float32 / float16 / bfloat16 with K % 32 == 0, on the CPU, and the device accepts
    the shape (``unsupported_reason``).
Everything else runs on the CPU. On the device the product is float32 (Q8_0 weight, QACT A32
activations); bias / ``beta * input`` and ``alpha`` are applied on the CPU in float32, and the result
is cast to the original output dtype. Batch / sequence dims are flattened into T.

Matmuls inside the bodies of higher-order ops (``invoke_subgraph`` from
``torch.compiler.nested_compile_region``, ``cond``, ``while_loop``, ``map``) are partitioned the same
way: a body placeholder is a weight if the HOP passes it a weight at every call site (a body shared by
all layers sees every layer's weight). Matmuls in the bodies of other HOPs stay on the CPU and are
reported as CPU nodes.

The run-time state of a device node (uploads in a row, CPU fallback) is kept per weight, because one
node can see many weights: regional compilation (``layer.compile()`` per block) and compiled helpers
called per layer reuse one graph, and a shared HOP body runs once per layer.

Graphs that need autograd (grad mode on and an input requiring grad, i.e. training) are not
partitioned: they run eagerly on the CPU (a warning; ``options={"strict": True}`` raises instead).
A device that cannot be opened, and transport or execution errors, raise ``TesseractError``; only a
weight that cannot be placed (WEIGHT heap full) runs on the CPU.
"""
from __future__ import annotations

import itertools
import re
import threading
from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch.multiprocessing.reductions import StorageWeakRef

from . import runtime as _rt
from .runtime import SUPPORTED_DTYPES, BLOCK, log

aten = torch.ops.aten

# Ops that only re-describe a tensor's memory: a weight seen through them is still the weight
# (its storage, and so the cache key, does not change between calls).
_VIEW_OPS = {
    aten.t.default, aten.transpose.int, aten.permute.default, aten.view.default, aten._unsafe_view.default,
    aten.reshape.default, aten.alias.default, aten.detach.default, aten.slice.Tensor, aten.select.int,
    aten.unsqueeze.default, aten.squeeze.dim, aten.squeeze.default, aten.squeeze.dims, aten.expand.default,
    aten.as_strided.default,
}
# A weight that has to be uploaded on every call of a node (changed in place before each call, or seen
# through an op that copies it) runs on the CPU at that node after this many uploads in a row. Counted
# per weight (its root parameter / buffer), so a node shared by many weights is not affected.
_CHURN_LIMIT = 4
# Matmul-like ops reported in the partition. The first group can be offloaded.
_MM_OPS = {
    aten.mm.default: "mm", aten.addmm.default: "addmm", aten.linear.default: "linear",
    aten.matmul.default: "matmul", aten.mv.default: "mv",
}
_OTHER_MM_OPS = {
    aten.bmm.default: "bmm", aten.baddbmm.default: "baddbmm", aten.addmv.default: "addmv",
    aten.dot.default: "dot", aten.addbmm.default: "addbmm", aten.einsum.default: "einsum",
    aten._scaled_mm.default if hasattr(aten, "_scaled_mm") else None: "_scaled_mm",
}
_OTHER_MM_OPS.pop(None, None)


# ------------------------------------------------------------------------------------------
# options
# ------------------------------------------------------------------------------------------
@dataclass
class Options:
    strict: bool = False        # raise instead of running training graphs / failed partitions eagerly
    debug: int | None = None    # override TSR_TORCH_DEBUG (process-wide)
    verify: bool | None = None  # override TSR_TORCH_VERIFY (process-wide): check every device call vs the reference
    offload: bool = True        # False: partition report only, every matmul stays on the CPU

    @classmethod
    def from_dict(cls, d: dict | None) -> Options:
        o = cls()
        for k, v in (d or {}).items():
            if not hasattr(o, k):
                raise ValueError(f"tesseract backend: unknown option {k!r} (strict, debug, verify, offload)")
            setattr(o, k, v)
        return o


# ------------------------------------------------------------------------------------------
# device op
# ------------------------------------------------------------------------------------------
_state_ids = itertools.count(1)


@dataclass
class _WeightState:
    """Run-time state of one weight (identified by its root parameter / buffer) at one device node."""
    ref: StorageWeakRef           # the root's storage: the state is dropped when it dies
    id: int
    uploads_in_a_row: int = 0     # consecutive calls of the node with this weight that uploaded it
    disabled: str | None = None   # set when this weight runs on the CPU at this node from now on


@dataclass
class MMSpec:
    """One matmul node put on the device."""
    id: int
    op: str                  # mm | addmm | linear | matmul | mv
    side: str                # "right": x @ W^T (weight operand b); "left": W @ x (weight operand a)
    N: int
    K: int
    out_dtype: torch.dtype
    alpha: float = 1.0
    beta: float = 1.0
    weight: str = ""
    node: str = ""
    graph: int = 0
    target: object = None    # the original aten op (CPU fallback)
    kwargs: dict = field(default_factory=dict)
    calls: int = 0           # calls on the device
    tokens: int = 0
    cpu_calls: int = 0       # calls that ran on the CPU (a weight that cannot be placed, or churn)
    _weights: dict = field(default_factory=dict, repr=False)   # WeightCache.key(root) -> _WeightState
    _prune_at: int = field(default=64, repr=False)

    def weight_state(self, root: torch.Tensor) -> _WeightState:
        """The state of weight `root` at this node (call with runtime._lock held)."""
        k = _rt.WeightCache.key(root)
        st = self._weights.get(k)
        if st is None or st.ref.expired():
            if len(self._weights) >= self._prune_at:
                for kk in [kk for kk, v in self._weights.items() if v.ref.expired()]:
                    del self._weights[kk]
                self._prune_at = max(64, 2 * len(self._weights))
            st = _WeightState(StorageWeakRef(root.untyped_storage()), next(_state_ids))
            self._weights[k] = st
        return st

    def disabled(self) -> list[str]:
        """Why each live weight of this node that runs on the CPU from now on does so."""
        with _rt._lock:
            return [v.disabled for v in self._weights.values() if v.disabled is not None and not v.ref.expired()]

    def cpu(self, a, b, bias):
        if self.op == "addmm":
            return self.target(bias, a, b, **self.kwargs)
        if self.op == "linear":
            return self.target(a, b, bias)
        return self.target(a, b)


_SPECS: list[MMSpec] = []
_specs_lock = threading.Lock()
_graph_counter = [0]


def _orient(s: MMSpec, a: torch.Tensor, b: torch.Tensor):
    """(W [N, K], x2 [T, K], finish(y [T, N]) -> out) for the operands of the original op."""
    K, N = s.K, s.N
    if s.side == "right":
        w = b if s.op == "linear" else b.t()
        lead = a.shape[:-1]
        return w, a.reshape(-1, K), (lambda y: y.reshape(*lead, N))
    w = a
    if b.dim() == 1:
        return w, b.reshape(1, K), (lambda y: y.reshape(N))
    return w, b.t(), (lambda y: y.t().contiguous())


def _on_cpu(s: MMSpec, a, b, bias):
    s.cpu_calls += 1
    _rt._bump("cpu_fallback_calls")
    return s.cpu(a, b, bias)


def device_mm(spec_id: int, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None,
              root: torch.Tensor | None = None) -> torch.Tensor:
    """The device form of one partitioned matmul node (its original operands; for addmm bias is
    `input`, for linear the bias; root: the parameter / buffer the weight operand is a view of)."""
    s = _SPECS[spec_id]
    w, x2, finish = _orient(s, a, b)
    if w.dtype not in SUPPORTED_DTYPES or tuple(w.shape) != (s.N, s.K) or w.device.type != "cpu":
        return _on_cpu(s, a, b, bias)
    with _rt._lock:
        st = s.weight_state(root if isinstance(root, torch.Tensor) else w)
        off = st.disabled is not None
    if off:
        return _on_cpu(s, a, b, bias)
    try:
        dev, h, uploaded = _rt._cache.acquire(w, s.weight)
    except (_rt.TesseractError, ValueError) as err:
        if not _rt.placement_error(err):
            raise                     # transport / device errors are errors, not a reason to run on the CPU
        # this weight cannot be placed (e.g. WEIGHT heap full): at this node it runs on the CPU from now on
        with _rt._lock:
            st.disabled = str(err)
        _rt._warn_once(f"upload:{s.id}:{st.id}", f"tesseract: {s.weight} [{s.N} x {s.K}] stays on the CPU: {err}")
        return _on_cpu(s, a, b, bias)
    with _rt._lock:
        st.uploads_in_a_row = st.uploads_in_a_row + 1 if uploaded else 0
        churn = st.uploads_in_a_row >= _CHURN_LIMIT
        if churn:
            st.disabled = (f"weight uploaded on {_CHURN_LIMIT} consecutive calls (changed in place, or copied, "
                           "before each call)")
    if churn:
        _rt._warn_once(f"churn:{s.id}:{st.id}",
                       f"tesseract: {s.weight} [{s.N} x {s.K}] runs on the CPU from now on: {st.disabled}")
    y = _rt.device_linear_2d(h, x2, name=s.weight, dev=dev)
    T = y.shape[0]
    s.calls += 1
    s.tokens += T
    _rt._bump("device_calls")
    _rt._bump("device_tokens", T)
    if _rt.debug_level() >= 2:
        log(f"call {s.node} ({s.weight}) T={T} N={s.N} K={s.K}")
    out = finish(y)
    if s.op == "addmm":
        # addmm(input, a, b, beta, alpha) = beta * input + alpha * (a @ b); input broadcasts to the
        # output, and beta = 0 ignores it (torch semantics: NaN / Inf in input do not propagate)
        if s.alpha != 1:
            out.mul_(s.alpha)
        if s.beta != 0 and bias is not None:
            out.add_(bias.to(torch.float32), alpha=s.beta)
    elif s.op == "linear" and bias is not None:
        out.add_(bias.to(torch.float32))
    if s.out_dtype != torch.float32:
        out = out.to(s.out_dtype)
    return out


# ------------------------------------------------------------------------------------------
# partition
# ------------------------------------------------------------------------------------------
_SRC_RE = [
    (re.compile(r"^L\['[^']*'\]"), ""), (re.compile(r"^G\['([^']*)'\]"), r"\1"),
    (re.compile(r"\._modules\['([^']*)'\]"), r".\1"), (re.compile(r"\._parameters\['([^']*)'\]"), r".\1"),
    (re.compile(r"\._buffers\['([^']*)'\]"), r".\1"), (re.compile(r"\[(\d+)\]"), r".\1"),
]


def _pretty_source(name: str) -> str:
    for rx, rep in _SRC_RE:
        name = rx.sub(rep, name)
    return name.lstrip(".") or name


def _dynamo_weights(gm: fx.GraphModule, example_inputs) -> dict[int, str]:
    """Dynamo placeholder index -> weight name, for parameters and registered buffers."""
    out = {}
    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, n in enumerate(phs):
        ex = example_inputs[i] if i < len(example_inputs) else None
        ga = n.meta.get("grapharg")
        src = ""
        if ga is not None and getattr(ga, "source", None) is not None:
            try:
                src = ga.source.name()
            except Exception:  # noqa: BLE001
                src = ""
        is_param = isinstance(ex, torch.nn.Parameter) or "._parameters[" in src
        is_buffer = "._buffers[" in src
        if is_param or is_buffer:
            out[i] = _pretty_source(src) if src else n.name
    return out


def _aot_weights(aot_gm: fx.GraphModule, dyn_weights: dict[int, str] | None, n_dyn_inputs: int) -> dict[fx.Node, str]:
    """AOT graph nodes that are weights -> name: placeholders described as parameters / buffers or as
    Dynamo inputs that are, and tensor constants (get_attr)."""
    from torch._functorch._aot_autograd import descriptors as D

    out: dict[fx.Node, str] = {}
    phs = [n for n in aot_gm.graph.nodes if n.op == "placeholder"]
    have_desc = all("desc" in n.meta for n in phs)
    for pos, n in enumerate(phs):
        desc = n.meta.get("desc")
        if have_desc:
            if isinstance(desc, (D.ParamAOTInput, D.BufferAOTInput)):
                out[n] = str(desc.target)
            elif isinstance(desc, D.PlainAOTInput) and dyn_weights is not None and desc.idx in dyn_weights:
                out[n] = dyn_weights[desc.idx]
        elif dyn_weights is not None and len(phs) == n_dyn_inputs and pos in dyn_weights:
            out[n] = dyn_weights[pos]   # older torch without descriptors: positional, only if counts agree
    out.update(_attr_weights(aot_gm))
    return out


def _getattr_path(gm: fx.GraphModule, target):
    v = gm
    try:
        for atom in str(target).split("."):
            v = getattr(v, atom)
    except AttributeError:
        return None
    return v


def _attr_weights(gm: fx.GraphModule) -> dict[fx.Node, str]:
    """Tensor constants of a graph (get_attr nodes) -> name."""
    return {n: str(n.target) for n in gm.graph.nodes
            if n.op == "get_attr" and isinstance(_getattr_path(gm, n.target), torch.Tensor)}


def _weight_root(node, weights: dict) -> fx.Node | None:
    seen = 0
    while isinstance(node, fx.Node) and seen < 64:
        if node in weights:
            return node
        if node.op == "call_function" and node.target in _VIEW_OPS and node.args:
            node = node.args[0]
            seen += 1
            continue
        return None
    return None


def _val(node):
    return node.meta.get("val") if isinstance(node, fx.Node) else None


def _concrete(v) -> int | None:
    try:
        return int(v) if not isinstance(v, torch.SymInt) else None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class _Decision:
    node: fx.Node
    op: str
    device: bool
    reason: str = ""
    spec: MMSpec | None = None
    root: fx.Node | None = None   # the weight's root node (device decisions)
    gm: fx.GraphModule | None = None
    where: str = ""               # "" in the top-level graph, else the path of the HOP body ("body.")


def _decide(node: fx.Node, op: str, weights: dict, gid: int) -> _Decision:
    args = node.args
    if op == "addmm":
        bias, a, b = args[0], args[1], args[2]
        kw = dict(node.kwargs)
        beta, alpha = kw.get("beta", 1), kw.get("alpha", 1)
    elif op == "linear":
        a, b = args[0], args[1]
        bias = args[2] if len(args) > 2 else node.kwargs.get("bias")
        kw, alpha, beta = {}, 1, 1
    else:
        a, b, bias, kw, alpha, beta = args[0], args[1], None, {}, 1, 1
    va, vb, vo = _val(a), _val(b), _val(node)
    if not all(isinstance(v, torch.Tensor) for v in (va, vb, vo)):
        return _Decision(node, op, False, "no tensor metadata")
    if not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        return _Decision(node, op, False, "symbolic alpha / beta")
    ra, rb = _weight_root(a, weights), _weight_root(b, weights)
    # which operand is the weight, oriented as W [N, K]
    if rb is not None and ra is None:
        side, wroot = "right", rb
        if op == "linear":
            wshape = tuple(vb.shape)
        else:
            if vb.dim() != 2:
                return _Decision(node, op, False, f"weight operand is {vb.dim()}D")
            wshape = (vb.shape[1], vb.shape[0])
        if op == "mv":
            return _Decision(node, op, False, "mv with the weight as the vector")
        wv, xv = vb, va
    elif ra is not None and rb is None and op != "linear":
        side, wroot = "left", ra
        if va.dim() != 2:
            return _Decision(node, op, False, f"weight operand is {va.dim()}D")
        if vb.dim() not in (1, 2):
            return _Decision(node, op, False, f"weight @ {vb.dim()}D operand (batched)")
        wshape = tuple(va.shape)
        wv, xv = va, vb
    elif ra is not None and rb is not None:
        return _Decision(node, op, False, "both operands are weights")
    else:
        return _Decision(node, op, False, "no weight operand (activation x activation)")
    if len(wshape) != 2:
        return _Decision(node, op, False, f"weight is {len(wshape)}D")
    N, K = _concrete(wshape[0]), _concrete(wshape[1])
    wname = weights[wroot]
    if N is None or K is None:
        return _Decision(node, op, False, f"{wname}: symbolic weight shape")
    if wv.device.type != "cpu" or xv.device.type != "cpu":
        return _Decision(node, op, False, f"{wname}: not on the CPU ({wv.device})")
    if wv.dtype not in SUPPORTED_DTYPES:
        return _Decision(node, op, False, f"{wname}: dtype {str(wv.dtype).replace('torch.', '')} "
                                          "(float32 / float16 / bfloat16 only)")
    if xv.dtype not in SUPPORTED_DTYPES or vo.dtype not in SUPPORTED_DTYPES:
        return _Decision(node, op, False, f"{wname}: activation dtype {xv.dtype} / output {vo.dtype}")
    if K % BLOCK != 0:
        return _Decision(node, op, False, f"{wname}: K % 32 != 0 (K = {K})")
    if N < 1:
        return _Decision(node, op, False, f"{wname}: N = 0")
    why = _rt.device().unsupported_reason(wv.dtype, N, K)
    if why is not None:
        return _Decision(node, op, False, f"{wname}: {why}")
    if op == "addmm" and bias is not None:
        vbias = _val(bias)
        if not isinstance(vbias, torch.Tensor):
            return _Decision(node, op, False, "addmm input without metadata")
    spec = MMSpec(id=-1, op=op, side=side, N=N, K=K, out_dtype=vo.dtype, alpha=alpha, beta=beta, weight=wname,
                  node=node.name, graph=gid, target=node.target, kwargs=kw)
    return _Decision(node, op, True, "", spec, root=wroot)


# Higher-order ops whose bodies are partitioned: args -> (body args, the operand of each body
# placeholder; None where it cannot be a weight, e.g. a loop-carried value or a map slice). The dense
# implementations of these ops call the body GraphModule, so its device_mm calls run as they are.
_HOP_OPERANDS = {
    "invoke_subgraph": lambda a: ([a[0]], list(a[2:])),                        # (body, id, *operands)
    "cond": lambda a: ([a[1], a[2]], list(a[3])),                              # (pred, true, false, operands)
    "while_loop": lambda a: ([a[0], a[1]], [None] * len(a[2]) + list(a[3])),   # (cond, body, carried, additional)
    "map_impl": lambda a: ([a[0]], [None] * len(a[1]) + list(a[2])),           # (body, xs, pos_args)
}
_MAX_HOP_DEPTH = 8


def _body_of(gm: fx.GraphModule, arg):
    if isinstance(arg, fx.Node) and arg.op == "get_attr":
        v = _getattr_path(gm, arg.target)
        if isinstance(v, fx.GraphModule):
            return v, str(arg.target)
    return None


def _hop_bodies(gm: fx.GraphModule, node: fx.Node, bodies: dict) -> None:
    """Record the bodies of HOP node: bodies[id] = [body, name, hop, call sites (operand lists) or None
    if some call site is not understood]."""
    hop = node.target.name() if hasattr(node.target, "name") else str(node.target)
    mapped = None
    fn = _HOP_OPERANDS.get(hop)
    if fn is not None and not node.kwargs:
        try:
            body_args, operands = fn(node.args)
            mapped = [_body_of(gm, a) for a in body_args]
        except (IndexError, TypeError):
            mapped = None
        if mapped is not None and any(m is None for m in mapped):
            mapped = None
    if mapped is None:
        flat = []
        for a in list(node.args) + list(node.kwargs.values()):
            flat.extend(a if isinstance(a, (list, tuple)) else [a])
        for a in flat:
            b = _body_of(gm, a)
            if b is not None:
                bodies.setdefault(id(b[0]), [b[0], b[1], hop, []])[3] = None
        return
    for body, name in mapped:
        e = bodies.setdefault(id(body), [body, name, hop, []])
        if e[3] is not None:
            e[3].append(operands)


def _collect(gm: fx.GraphModule, weights: dict, opts: Options, gid: int, where: str, out: list,
             cpu_reason: str | None = None, depth: int = 0) -> None:
    bodies: dict = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            _hop_bodies(gm, node, bodies)
            continue
        if node.target in _MM_OPS:
            op = _MM_OPS[node.target]
            d = _decide(node, op, weights, gid) if cpu_reason is None else _Decision(node, op, False, cpu_reason)
        elif node.target in _OTHER_MM_OPS:
            d = _Decision(node, _OTHER_MM_OPS[node.target], False, "op not offloaded (batched / other matmul)")
        else:
            continue
        d.gm, d.where = gm, where
        if d.device and not opts.offload:
            d.device, d.reason = False, "offload disabled (options offload=False)"
        out.append(d)
    for body, name, hop, sites in bodies.values():
        sub = f"{where}{name}."
        if cpu_reason is not None or sites is None or depth >= _MAX_HOP_DEPTH:
            _collect(body, {}, opts, gid, sub, out, cpu_reason or f"inside a {hop} body (not offloaded)", depth + 1)
            continue
        bw = _attr_weights(body)
        for i, ph in enumerate(n for n in body.graph.nodes if n.op == "placeholder"):
            names = []
            for ops in sites:
                r = _weight_root(ops[i], weights) if i < len(ops) else None
                if r is None:
                    names = []
                    break
                names.append(weights[r])
            if names:
                uniq = list(dict.fromkeys(names))
                bw[ph] = uniq[0] if len(uniq) == 1 else f"{uniq[0]} .. {uniq[-1]} ({len(uniq)} weights)"
        _collect(body, bw, opts, gid, sub, out, None, depth + 1)


def partition(aot_gm: fx.GraphModule, weights: dict, opts: Options, gid: int) -> list[_Decision]:
    """Rewrite aot_gm (and the bodies of its higher-order ops) in place: device matmuls become
    device_mm calls. Returns the decisions. All decisions are made before a graph is touched, so a
    failure leaves them unchanged."""
    decisions: list[_Decision] = []
    _collect(aot_gm, weights, opts, gid, "", decisions)
    touched = {id(aot_gm): aot_gm}
    for d in decisions:
        if not d.device:
            continue
        node, g = d.node, d.gm
        with _specs_lock:
            d.spec.id = len(_SPECS)
            _SPECS.append(d.spec)
        if d.op == "addmm":
            call_args = (d.spec.id, node.args[1], node.args[2], node.args[0], d.root)
        elif d.op == "linear":
            bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
            call_args = (d.spec.id, node.args[0], node.args[1], bias, d.root)
        else:
            call_args = (d.spec.id, node.args[0], node.args[1], None, d.root)
        with g.graph.inserting_before(node):
            new = g.graph.call_function(device_mm, call_args)
        new.meta = dict(node.meta)
        node.replace_all_uses_with(new)
        g.graph.erase_node(node)
        d.node = new
        d.spec.node = d.where + node.name
        touched[id(g)] = g
    for g in touched.values():
        g.graph.lint()
        g.recompile()
    return decisions


def _report(gid: int, aot_gm: fx.GraphModule, decisions: list[_Decision]) -> None:
    nd = sum(d.device for d in decisions)
    n_nodes = sum(1 for _ in aot_gm.graph.nodes)
    log(f"graph {gid}: {nd} matmul(s) on Tesseract, {len(decisions) - nd} on the CPU (aten graph of {n_nodes} nodes)")
    for d in decisions:
        if d.device:
            s = d.spec
            extra = []
            if s.op == "addmm":
                extra.append("+ input on the CPU" if s.beta != 0 else "beta = 0")
            if s.op == "linear" and len(d.node.args) > 3 and d.node.args[3] is not None:
                extra.append("+ bias on the CPU")
            if s.side == "left":
                extra.append("weight on the left (W @ x)")
            if s.out_dtype != torch.float32:
                extra.append(f"out {str(s.out_dtype).replace('torch.', '')}")
            log(f"  tesseract  {s.node:<16} {d.op:<7} {s.weight}  [N={s.N} K={s.K}]"
                + (f"  ({', '.join(extra)})" if extra else ""))
        else:
            log(f"  cpu        {d.where + d.node.name:<16} {d.op:<7} {d.reason}")


# ------------------------------------------------------------------------------------------
# the backend
# ------------------------------------------------------------------------------------------
def _needs_autograd(example_inputs) -> bool:
    return torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in example_inputs)


def tesseract_backend(gm: fx.GraphModule, example_inputs, options: dict | None = None, mode: str | None = None):
    """torch.compile backend: ``torch.compile(model, backend="tesseract")`` (after ``import
    tesseract_torch``). options: strict (bool), debug (int), verify (bool), offload (bool)."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    opts = Options.from_dict(options)
    if opts.debug is not None:
        _rt.set_debug(opts.debug)
    if opts.verify is not None:
        _rt.set_verify(opts.verify)
    if _needs_autograd(example_inputs):
        msg = ("tesseract backend: this graph needs autograd (grad mode is on and an input requires grad); "
               "Tesseract runs inference only, so it runs eagerly on the CPU. Use torch.no_grad() / "
               "torch.inference_mode() or requires_grad_(False) to put its matmuls on the device.")
        if opts.strict:
            raise RuntimeError(msg)
        _rt._warn_once("autograd", msg)
        _rt._bump("graphs_eager")
        if _rt.debug_level() >= 1:
            log("graph needs autograd: running it eagerly on the CPU")
        return gm.forward
    flat = not any(isinstance(x, (list, tuple, dict)) for x in example_inputs)
    dyn_weights = _dynamo_weights(gm, example_inputs) if flat else None
    n_dyn = len(example_inputs)

    def compiler(aot_gm: fx.GraphModule, aot_inputs):
        # a device that cannot be opened (TSR_DEVICE, no card, no server) is an error, not a CPU run
        _rt.device()
        with _specs_lock:
            gid = _graph_counter[0]
            _graph_counter[0] += 1
        try:
            weights = _aot_weights(aot_gm, dyn_weights, n_dyn)
            decisions = partition(aot_gm, weights, opts, gid)
        except Exception as err:  # noqa: BLE001
            if opts.strict or isinstance(err, _rt.TesseractError):
                raise
            _rt._warn_once(f"partition:{type(err).__name__}",
                           f"tesseract backend: partitioning failed ({type(err).__name__}: {err}); graph runs on the CPU")
            return make_boxed_func(aot_gm.forward)
        _rt._bump("graphs")
        _rt._bump("nodes_device", sum(d.device for d in decisions))
        _rt._bump("nodes_cpu", sum(not d.device for d in decisions))
        if _rt.debug_level() >= 1:
            _report(gid, aot_gm, decisions)
        if _rt.debug_level() >= 3:
            log("rewritten aten graph:\n" + aot_gm.print_readable(print_output=False))
        fn = make_boxed_func(aot_gm.forward)
        fn._tesseract_decisions = decisions
        return fn

    return aot_autograd(fw_compiler=compiler, inference_compiler=compiler,
                        keep_inference_input_mutations=True)(gm, example_inputs)


def specs() -> list[MMSpec]:
    """Every matmul node partitioned to the device so far (with per-node call / token / CPU-call counts;
    ``disabled()`` lists the weights that run on the CPU at that node)."""
    return list(_SPECS)

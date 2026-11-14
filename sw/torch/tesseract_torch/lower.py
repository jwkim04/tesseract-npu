"""Whole-graph lowering of an aten graph to device regions (full mode), device free.

``form_regions(gm, weights, probe, opts)`` takes the aten graph that AOTAutograd gives the backend and
finds the regions that run as one op list of firmware kinds. This module only analyses the graph and
builds plain Python data; the graph surgery and the device calls are done by the caller. Nothing here
imports the C++ extension: what needs the device (``probe``) is injected as a callable.

Concepts
  device value   a float32 [T, n] tensor of token-major rows that lives on the device.
  View           an aten node seen as a device value: the node whose rows it is a view of (``base``), the
                 form of this node (token-major "tm" or head-major "hm", ``[1, H, T, D]``), the row length
                 ``n`` and the heads (layout rules in ``LowerCtx.view_of``).
  Cand           one device op (an IR op) with the aten nodes it replaces.
  Region         a convex set of at least two ops; ``Region.to_program()`` is the program of DESIGN.md §3.

Program placeholders (what ``to_program`` writes for things only known at call time)
  {"kind": "const", "n": n, "rows": 1, "source": i}   i indexes ``Region.consts`` (the runtime uploads that
                                                       parameter / buffer, after the transform, and puts
                                                       the handle in the value)
  {"op": "mul_mat", "w": i, ...}                      i indexes ``Region.weights`` (the runtime resolves
                                                       the weight handle)
  "in" / "pos" / "mask" values are given at every call, in the order of ``Region.inputs``.
"""
from __future__ import annotations

import math
import operator
import struct
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import sympy
import torch
import torch.fx as fx

aten = torch.ops.aten

BLOCK = 32                      # K of a weight matmul is a multiple of this (as in backend.py)
_INF = float("inf")
UNARY_FUNC = {"tanh": 0, "silu": 2, "exp": 3, "gelu": 4, "sigmoid": 5}


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------
def f32(x: float) -> float:
    """x rounded to float32 (as a Python float): every float of a program is exactly a float32."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _targets(*names):
    """aten overloads that exist in this torch (a missing name is skipped)."""
    out = set()
    for n in names:
        o = aten
        try:
            for part in n.split("."):
                o = getattr(o, part)
        except (AttributeError, RuntimeError):
            continue
        out.add(o)
    return out


def _custom(name: str):
    try:
        ns, op = name.split("::")
        return getattr(getattr(torch.ops, ns), op).default
    except (AttributeError, RuntimeError):
        return None


_VIEW_RESHAPE = _targets("view.default", "_unsafe_view.default", "reshape.default", "unsqueeze.default",
                         "squeeze.default", "squeeze.dim", "squeeze.dims", "flatten.using_ints")
_IDENT = _targets("alias.default", "clone.default", "detach.default", "_to_copy.default", "contiguous.default")
_COPY = _targets("clone.default", "contiguous.default", "_to_copy.default")
_UNSAFE_VIEW = _targets("_unsafe_view.default")
_CONVERT = _targets("lift_fresh_copy.default") | _targets("_to_copy.default")
try:
    _CONVERT.add(torch.ops.prims.convert_element_type.default)
except (AttributeError, RuntimeError):
    pass
_SDPA_CPU = _targets("_scaled_dot_product_flash_attention_for_cpu.default")
_SDPA = _targets("scaled_dot_product_attention.default")
_LINEAR_BLOCKS = _custom("tesseract::linear_blocks")


def storage_root(n, stop=None):
    """The node whose storage ``n`` shares: walk up the first argument while the node is a view (a schema that
    returns an alias of its input, and ``_unsafe_view``) and not a copy (``_COPY``) or a write. ``stop`` is a node
    at which the walk ends (the root of a region value)."""
    while (isinstance(n, fx.Node) and n is not stop and n.op == "call_function" and n.args
           and isinstance(n.args[0], fx.Node) and n.target not in _COPY):
        sch = getattr(n.target, "_schema", None)
        if n.target not in _UNSAFE_VIEW and (sch is None or not sch.returns or sch.returns[0].alias_info is None
                                             or sch.is_mutable):
            break
        n = n.args[0]
    return n


def _val(node):
    return node.meta.get("val") if isinstance(node, fx.Node) else None


def _tensor(node):
    v = _val(node)
    return v if isinstance(v, torch.Tensor) else None


def _shape(node):
    v = _tensor(node)
    return None if v is None else tuple(v.shape)


def _x(d):
    """A dimension as a sympy value (int or SymInt)."""
    if isinstance(d, sympy.Basic):
        return d
    if isinstance(d, torch.SymInt):
        return d.node.expr
    return sympy.Integer(int(d))


def _int(d) -> int | None:
    e = _x(d)
    return int(e) if e.is_Integer else None


def _eq(a, b) -> bool:
    return _x(a) == _x(b)


def _prod(dims):
    p = sympy.Integer(1)
    for d in dims:
        p = p * _x(d)
    return p


def _contiguous_strides(shape) -> list:
    out, p = [], sympy.Integer(1)
    for d in reversed(shape):
        out.append(p)
        p = p * _x(d)
    return out[::-1]


def _same_shape(a, b) -> bool:
    return a is not None and b is not None and len(a) == len(b) and all(_eq(p, q) for p, q in zip(a, b))


def _is_f32(node) -> bool:
    v = _tensor(node)
    return v is not None and v.dtype == torch.float32


def _is_call(node, targets) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function" and node.target in targets


def _name(node) -> str:
    return str(node.target)


def _arg(node, i, name, default=None):
    if len(node.args) > i:
        return node.args[i]
    return node.kwargs.get(name, default)


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _lit(node) -> float | None:
    """The value of a Python scalar, or of a ``scalar_tensor`` node of a literal."""
    n = _num(node)
    if n is not None:
        return n
    if _is_call(node, _targets("scalar_tensor.default")):
        return _num(node.args[0])
    return None


# ------------------------------------------------------------------------------------------
# the IR
# ------------------------------------------------------------------------------------------
@dataclass
class View:
    """An aten node seen as a device value (DESIGN.md §5.2).

    base      the aten node whose rows the device value is (a producing op's result, or a region input)
    base_hm   True when ``base`` itself is head-major [1, H, T, D]; its flattening recipe starts with a
              ``transpose(1, 2)``. Otherwise the base is flattened with ``reshape(-1, n)``.
    form      "tm": the node is [..., n] or [..., H, D] token-major; "hm": the node is [1, H, T, D]
    n         row length of the device value (H * D when heads is set)
    heads, hd the logical heads and head size when the node's tail is split (tm) or head-major (hm)
    rep       grouped-KV expansion: the node repeats every device head ``rep`` times (hm only); ``heads``
              is then the expanded head count
    rows      the number of rows T (a sympy value; a SymInt for a dynamic sequence length)
    chain     the aten nodes from this node down to the base (this one first, the base excluded)
    """
    base: fx.Node
    base_hm: bool
    form: str
    n: int
    heads: int | None
    hd: int | None
    rep: int
    rows: Any
    chain: tuple = ()

    def key(self):
        return (self.base, self.form, self.n, self.heads, self.hd, self.rep)

    def same_layout(self, o: "View") -> bool:
        return (self.form, self.n, self.heads, self.hd, self.rep) == (o.form, o.n, o.heads, o.hd, o.rep) \
            and _eq(self.rows, o.rows)


@dataclass(frozen=True)
class ConstSpec:
    """A parameter or buffer used as a constant row of the program. ``transform``: "identity", or
    "one_plus" (the runtime uploads ``1 + w``; the Gemma norm)."""
    node: fx.Node
    transform: str
    n: int


@dataclass(frozen=True)
class WeightSpec:
    """A matmul weight: the parameter / buffer node ``root`` ([N, K], float32), or ``kind == "blocks"``
    for ``tesseract::linear_blocks`` (``root`` is the uint8 blocks tensor, ``fmt`` the ggml format)."""
    node: fx.Node
    N: int
    K: int
    kind: str = "dense"
    fmt: str | None = None


# operand references inside a Cand (resolved to value ids when a program is built)
@dataclass(frozen=True)
class V:
    view: View = field(compare=False)
    key: tuple = ()


@dataclass(frozen=True)
class C:
    spec: ConstSpec


@dataclass(frozen=True)
class P:
    node: fx.Node


@dataclass(frozen=True)
class M:
    node: fx.Node
    form: str            # "bool" (True keeps) or "additive" (0 / -inf)


@dataclass(frozen=True)
class W:
    spec: WeightSpec


@dataclass(frozen=True)
class Scalar:
    """A float of a program that is only known at call time: the 0-dim tensor ``node`` (a graph input or
    buffer; Dynamo passes a Python float as a 0-dim input once it has recompiled for another value).
    ``fn`` is applied to its value: "id", "recip" (1 / x) or "neg" (-x); the result is rounded to float32."""
    node: fx.Node
    fn: str = "id"

    def apply(self, x: float) -> float:
        if self.fn == "recip":
            if x == 0.0:
                raise ValueError("lower: scalar 0 in a division")
            x = 1.0 / x
        elif self.fn == "neg":
            x = -x
        r = f32(x)
        if not math.isfinite(r):
            raise ValueError(f"lower: scalar {x} is not finite in float32")
        return r


def _v(view: View) -> V:
    return V(view, view.key())


@dataclass(eq=False)
class Cand:
    """One device op: its fields (operands as references), the layout of its result, and the aten nodes."""
    kind: str
    root: fx.Node
    fields: dict
    out: View
    owned: list                                  # nodes this op replaces (the root included)
    bypassed: list = field(default_factory=list)  # nodes it reads but does not consume (cos / sin, mask chain)
    chain: list = field(default_factory=list)    # view nodes between its operands and their bases
    addend_of: "Cand | None" = None              # folded: the mul_mat Cand and the add Cand it came from
    parts: tuple = ()
    reason: str = ""

    def operands(self):
        for v in self.fields.values():
            if isinstance(v, V):
                yield v


@dataclass
class Input:
    """A region input: the aten node to pass and the recipe that flattens it to the 2D device form.

    recipe by kind
      "in":   {"transpose12": bool, "reshape": [-1, n]}   x = node; if transpose12: x = x.transpose(1, 2);
                                                           x = x.reshape(-1, n)
      "pos":  {"reshape": [-1]}                            integer positions (int64 / int32), one per row
      "mask": {"form": "bool" | "additive", "reshape": "TxT"}   the mask as a [T, T] tensor (T = rows)
    """
    value: int
    node: fx.Node
    kind: str
    recipe: dict


@dataclass
class Output:
    """A region output: the aten node it replaces and how to rebuild it from the [T, n] result.

    recipe: {"dims": [int | {"input": k, "dim": j}, ...], "transpose12": bool, "contiguous": bool}
      y = result.reshape(dims) (an entry {"input": k, "dim": j} is ``sym_size(inputs[k].node, j)``);
      if transpose12: y = y.transpose(1, 2); if contiguous: y = y.contiguous().
    The result has the strides that ``node`` has in eager (``_out_recipe`` refuses any other layout).
    Several outputs can be views of one value. ``root`` is the root node of the op that made the value and
    ``root_recipe`` rebuilds it from the rows (same form as ``recipe``; None when ``node`` is the root or the
    root cannot be rebuilt): a later region can read the root through views that ``node`` ends.
    ``sroot`` is the node whose storage ``node`` shares in the aten graph (``storage_root``, ending at ``root``):
    the outputs of one value with the same ``sroot`` share storage after the rewrite, those with another do not.
    """
    value: int
    node: fx.Node
    recipe: dict
    root: fx.Node | None = None
    root_recipe: dict | None = None
    sroot: fx.Node | None = None


@dataclass
class Region:
    id: int
    values: list
    ops: list
    inputs: list                      # [Input] in value order (kinds "in", "pos", "mask")
    consts: list                      # [ConstSpec]; value {"kind": "const", "source": i}
    weights: list                     # [WeightSpec]; op {"w": i}
    outputs: list                     # [Output]
    nodes: list                       # aten nodes the region replaces, in graph order
    bypassed: list                    # nodes it reads but does not replace (dead if nothing else uses them)
    rows: Any                         # T as a sympy value
    kv_size: int
    op_nodes: list                    # the aten node each op came from (its root), parallel to ops
    touching: dict = field(default_factory=dict)   # CPU node next to the region -> reason
    scalars: list = field(default_factory=list)    # 0-dim graph inputs the program's floats depend on

    def to_program(self, scalar_values=None) -> dict:
        """The program of DESIGN.md §3. ``scalar_values`` are the current Python floats of
        ``self.scalars`` (the 0-dim tensors, read at call time); a region without scalars needs none."""
        if self.scalars and scalar_values is None:
            raise ValueError("lower: this region has runtime scalars; pass their values")
        sv = dict(zip(self.scalars, scalar_values or ()))
        ops = []
        for o in self.ops:
            d = dict(o)
            for k, v in d.items():
                if isinstance(v, Scalar):
                    d[k] = v.apply(float(sv[v.node]))
            ops.append(d)
        return {"values": [dict(v) for v in self.values], "ops": ops, "kv_size": self.kv_size}

    def op_counts(self) -> dict:
        out: dict = {}
        for o in self.ops:
            out[o["op"]] = out.get(o["op"], 0) + 1
        return out

    def describe(self) -> str:
        c = ", ".join(f"{k} x{v}" for k, v in sorted(self.op_counts().items()))
        return (f"region {self.id}: {len(self.ops)} ops ({c}), {len(self.inputs)} inputs, "
                f"{len(self.outputs)} outputs, {len(self.weights)} weights, {len(self.consts)} consts")


@dataclass
class LowerOpts:
    kv_size: int = 4096        # cells of every KV plane of a region
    min_ops: int = 2           # D2: a region has at least two ops
    attention: str = "device"  # "cpu": scaled_dot_product_attention is not lowered (it stays on the CPU in float32)

    @staticmethod
    def from_dict(d: dict | None) -> "LowerOpts":
        d = dict(d or {})
        return LowerOpts(kv_size=int(d.get("kv_size", 4096)), min_ops=int(d.get("min_ops", 2)),
                         attention=str(d.get("attention", "device")))


# ------------------------------------------------------------------------------------------
# the program builder (resolves the references of Cands to value ids)
# ------------------------------------------------------------------------------------------
class _Builder:
    def __init__(self):
        self.values: list = []
        self.ops: list = []
        self.op_nodes: list = []
        self.inputs: list[Input] = []
        self.consts: list[ConstSpec] = []
        self.weights: list[WeightSpec] = []
        self.scalars: list = []
        self.produced: dict = {}        # root node -> value id
        self._in: dict = {}
        self._const: dict = {}
        self._pos: dict = {}
        self._mask: dict = {}
        self._weight: dict = {}

    def _new(self, v: dict) -> int:
        self.values.append(v)
        return len(self.values) - 1

    def view(self, view: View) -> int:
        if view.base in self.produced:
            return self.produced[view.base]
        k = (view.base, view.base_hm, view.n)
        if k not in self._in:
            vid = self._new({"kind": "in", "n": view.n})
            recipe = {"transpose12": view.base_hm, "reshape": [-1, view.n]}
            self.inputs.append(Input(vid, view.base, "in", recipe))
            self._in[k] = vid
        return self._in[k]

    def const(self, spec: ConstSpec) -> int:
        if spec not in self._const:
            self.consts.append(spec)
            self._const[spec] = self._new({"kind": "const", "n": spec.n, "rows": 1, "source": len(self.consts) - 1})
        return self._const[spec]

    def pos(self, node) -> int:
        if node not in self._pos:
            vid = self._new({"kind": "pos"})
            self.inputs.append(Input(vid, node, "pos", {"reshape": [-1]}))
            self._pos[node] = vid
        return self._pos[node]

    def mask(self, node, form) -> int:
        if (node, form) not in self._mask:
            vid = self._new({"kind": "mask"})
            self.inputs.append(Input(vid, node, "mask", {"form": form, "reshape": "TxT"}))
            self._mask[(node, form)] = vid
        return self._mask[(node, form)]

    def weight(self, spec: WeightSpec) -> int:
        if spec not in self._weight:
            self.weights.append(spec)
            self._weight[spec] = len(self.weights) - 1
        return self._weight[spec]

    def resolve(self, ref):
        if ref is None or isinstance(ref, (int, float, str)):
            return ref
        if isinstance(ref, V):
            return self.view(ref.view)
        if isinstance(ref, C):
            return self.const(ref.spec)
        if isinstance(ref, P):
            return self.pos(ref.node)
        if isinstance(ref, M):
            return self.mask(ref.node, ref.form)
        if isinstance(ref, W):
            return self.weight(ref.spec)
        if isinstance(ref, Scalar):
            if ref.node not in self.scalars:
                self.scalars.append(ref.node)
            return ref
        raise TypeError(ref)

    def emit(self, c: Cand, tag: int):
        op = {"op": c.kind}
        for k, v in c.fields.items():
            op[k] = self.resolve(v)
        dst = self._new({"kind": "tmp", "n": c.out.n, "out": False})
        self.produced[c.root] = dst
        op["dst"] = dst
        op["tag"] = tag
        self.ops.append(op)
        self.op_nodes.append(c.root)


# ------------------------------------------------------------------------------------------
# the lowering context
# ------------------------------------------------------------------------------------------
class LowerCtx:
    """The analysis of one aten graph: layouts, patterns, regions."""

    def __init__(self, gm: fx.GraphModule, weights: dict, probe: Callable | None, opts: LowerOpts):
        self.gm = gm
        self.weights = weights
        self.probe = probe
        self.opts = opts
        self.nodes = list(gm.graph.nodes)
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.decl: dict = {}            # cand root -> declared View
        self._vc: dict = {}
        self.why: dict = {}             # node -> reason it stays on the CPU

    # ---------------------------------------------------------------- reasons
    def note(self, node, text: str):
        self.why.setdefault(node, text)

    # ---------------------------------------------------------------- layout rules (§5.2)
    def base_view(self, node, want=None) -> View | None:
        """``node`` as the owner of its device rows. want: None (token-major, tail [n]), "tm4"
        ([1, T, H, D] with the tail split) or "hm" ([1, H, T, D])."""
        if not _is_f32(node):
            return None
        s = _shape(node)
        if not s:
            return None
        if want is None:
            n = _int(s[-1])
            if n is None or n < 1:
                return None
            return View(node, False, "tm", n, None, None, 1, _prod(s[:-1]))
        if len(s) != 4 or _int(s[0]) != 1:
            return None
        if want == "tm4":
            H, D = _int(s[2]), _int(s[3])
            if H is None or D is None:
                return None
            return View(node, False, "tm", H * D, H, D, 1, _x(s[1]))
        if want == "hm":
            H, D = _int(s[1]), _int(s[3])
            if H is None or D is None:
                return None
            return View(node, True, "hm", H * D, H, D, 1, _x(s[2]))
        return None

    def view_of(self, node, want=None) -> tuple[View | None, str]:
        """The device value that ``node`` is a view of, or (None, reason).

        A chain of view ops keeps the memory order of a token-major [T, n] value when it is made of
        view / _unsafe_view / reshape / unsqueeze / squeeze / alias / clone / contiguous / float32
        _to_copy with the last dimension n unchanged or split / merged as heads ([.., H * D] <->
        [.., H, D]). The head-major form [1, H, T, D] is ``transpose(1, 2)`` of [1, T, H, D];
        ``transpose(1, 2)`` of it returns to rows. Grouped-KV expansion (unsqueeze, expand, clone,
        reshape; or repeat_interleave) of a head-major value is a head-major view with ``rep`` > 1.
        ``cat`` of one tensor with empty ones is that tensor. Any other op ends the chain: the node is
        then its own base (a region input if its producer is not lowered). ``want`` says how to read a
        node that is its own base ("hm": [1, H, T, D]; "tm4": [1, T, H, D]).
        """
        key = (node, want)
        if key not in self._vc:
            self._vc[key] = self._view_of(node, want)
        return self._vc[key]

    def _view_of(self, node, want):
        if not isinstance(node, fx.Node) or node.op == "output":
            return None, "not a tensor value"
        if not _is_f32(node):
            v = _tensor(node)
            return None, f"{'no tensor metadata' if v is None else 'dtype ' + str(v.dtype).replace('torch.', '')} (float32 only)"
        if node in self.decl:
            return self.decl[node], ""
        if node.op == "call_function":
            res = self._chain_step(node, want)
            if res is not None:
                return res, ""
        v = self.base_view(node, want)
        if v is None:
            return None, f"shape {tuple(_shape(node) or ())} is not rows of float32 (want {want})"
        return v, ""

    def _wrap(self, p: View, node, **kw) -> View:
        return replace(p, chain=(node,) + p.chain, **kw)

    def _chain_step(self, node, want) -> View | None:
        t = node.target
        s = _shape(node)
        if t in _IDENT:
            src = node.args[0]
            if _is_f32(src) and _same_shape(_shape(src), s):
                p, _ = self.view_of(src, want)
                if p is not None:
                    return self._wrap(p, node)
            return None
        if t is aten.slice.Tensor:
            src = node.args[0]
            full = (_arg(node, 2, "start", 0) in (0, None) and _arg(node, 4, "step", 1) == 1
                    and _is_f32(src) and _same_shape(_shape(src), s))
            if full:
                p, _ = self.view_of(src, want)
                if p is not None:
                    return self._wrap(p, node)
            return None
        if t in _VIEW_RESHAPE:
            g = self._gqa(node)
            if g is not None:
                return g
            p, _ = self.view_of(node.args[0], None)
            if p is None or p.form != "tm" or p.rep != 1 or not s:
                return None
            n = p.n
            if _int(s[-1]) == n:
                heads, hd, rows = None, None, _prod(s[:-1])
            elif len(s) >= 2 and _int(s[-1]) and _int(s[-2]) and _int(s[-1]) * _int(s[-2]) == n:
                heads, hd, rows = _int(s[-2]), _int(s[-1]), _prod(s[:-2])
            else:
                return None
            if not _eq(rows, p.rows):
                return None
            return self._wrap(p, node, heads=heads, hd=hd)
        if t is aten.transpose.int:
            return self._transpose(node, want)
        if t is aten.cat.default:
            parts = node.args[0]
            full = [x for x in parts if not self._is_empty(x)]
            if len(full) == 1 and _same_shape(_shape(full[0]), s):
                p, _ = self.view_of(full[0], want)
                if p is not None:
                    empties = tuple(x for x in parts if x is not full[0] and isinstance(x, fx.Node))
                    return replace(p, chain=(node,) + empties + p.chain)
            return None
        return None

    @staticmethod
    def _is_empty(x) -> bool:
        s = _shape(x)
        return s is not None and any(_int(d) == 0 for d in s)

    def _transpose(self, node, want) -> View | None:
        src, d0, d1 = node.args[0], node.args[1], node.args[2]
        s_in, s = _shape(src), _shape(node)
        if s_in is None or len(s_in) != 4 or {d0 % 4, d1 % 4} != {1, 2} or _int(s_in[0]) != 1:
            return None
        pw = "hm" if want == "tm" else "tm4"
        p, _ = self.view_of(src, pw)
        if p is None or p.rep != 1:
            return None
        a, b, c = s_in[1], s_in[2], s_in[3]            # src is [1, a, b, c]
        if p.form == "tm" and p.heads is not None and p.heads == _int(b) and p.hd == _int(c) and _eq(p.rows, a):
            return self._wrap(p, node, form="hm", rows=_x(a))
        if p.form == "tm" and p.heads is None and _int(b) == 1 and p.n == _int(c) and _eq(p.rows, a):
            # [1, T, 1, D] of a [1, T, D] value is one head
            return self._wrap(p, node, form="hm", rows=_x(a), heads=1, hd=p.n)
        if p.form == "hm" and p.heads == _int(a) and p.hd == _int(c) and _eq(p.rows, b):
            return self._wrap(p, node, form="tm", rows=_x(b))
        return None

    def _gqa(self, node) -> View | None:
        """[1, Hkv * g, T, D] from a head-major [1, Hkv, T, D] by repeating every head g times."""
        s = _shape(node)
        if s is None or len(s) != 4 or _int(s[0]) != 1:
            return None
        src = node.args[0]
        if _is_call(src, _targets("repeat_interleave.self_int")) and len(src.args) >= 3 and src.args[2] in (1, -3):
            g, y, chain = src.args[1], src.args[0], [node, src]
            s5 = None
        else:
            chain = [node]
            cur = src
            if _is_call(cur, _targets("clone.default", "contiguous.default")):
                chain.append(cur)
                cur = cur.args[0]
            if not _is_call(cur, _targets("expand.default")):
                return None
            s5 = _shape(cur)
            if s5 is None or len(s5) != 5 or _int(s5[0]) != 1 or _int(s5[2]) is None:
                return None
            chain.append(cur)
            un = cur.args[0]
            if _is_call(un, _targets("slice.Tensor")) and _same_shape(_shape(un.args[0]), _shape(un)) \
                    and _arg(un, 2, "start", 0) in (0, None) and _arg(un, 4, "step", 1) == 1:
                chain.append(un)
                un = un.args[0]
            if not (_is_call(un, _targets("unsqueeze.default")) and un.args[1] in (2, -3)):
                return None
            chain.append(un)
            y, g = un.args[0], _int(s5[2])
        sy = _shape(y)
        if sy is None or len(sy) != 4 or not isinstance(g, int) or g < 1:
            return None
        p, _ = self.view_of(y, "hm")
        if p is None or p.form != "hm" or p.rep != 1:
            return None
        Hq = _int(s[1])
        if Hq != p.heads * g or p.hd != _int(s[3]) or not _eq(p.rows, s[2]):
            return None
        return replace(p, form="hm", heads=Hq, rep=g, rows=p.rows, chain=tuple(chain) + p.chain)

    # ---------------------------------------------------------------- operand helpers
    def dev(self, node, want=None, rep_ok=False) -> tuple[View | None, str]:
        v, why = self.view_of(node, want)
        if v is None:
            return None, why
        if v.rep != 1 and not rep_ok:
            return None, "grouped-KV expansion outside attention"
        return v, ""

    def const_row(self, node) -> ConstSpec | None:
        """A parameter / buffer used as a row [n] (possibly through shape-preserving ops), or None."""
        cur = node
        for _ in range(16):
            if cur in self.weights:
                break
            if cur.op == "call_function" and (cur.target in _VIEW_RESHAPE or cur.target in _IDENT):
                if not _is_f32(cur):    # a narrowing and widening copy rounds the values
                    return None
                cur = cur.args[0]
                continue
            return None
        else:
            return None
        if not _is_f32(cur):
            return None
        sr = _shape(cur)
        sn = _shape(node)
        if sr is None or len(sr) != 1 or _int(sr[0]) is None or sn is None:
            return None
        if _int(sn[-1]) != _int(sr[0]) or any(_int(d) != 1 for d in sn[:-1]):
            return None
        return ConstSpec(cur, "identity", _int(sr[0]))

    def scalar(self, node):
        """(value, nodes read) for a Python scalar, a ``scalar_tensor`` literal, or a 0-dim float graph
        input seen through dtype conversions (a ``Scalar``); None for anything else."""
        n = _lit(node)
        if n is not None:
            return n, []
        chain = []
        cur = node
        for _ in range(8):
            t = _tensor(cur) if isinstance(cur, fx.Node) else None
            # a narrowing cast rounds the value: only float32 and float64 pass
            if t is None or t.dim() != 0 or t.dtype not in (torch.float32, torch.float64):
                return None
            if cur.op in ("placeholder", "get_attr"):
                return Scalar(cur), chain
            if cur.op == "call_function" and (cur.target in _CONVERT or cur.target in _IDENT):
                chain.append(cur)
                cur = cur.args[0]
                continue
            return None
        return None

    def _only_in(self, node, allowed) -> bool:
        return all(u in allowed for u in node.users)

    def _out_like(self, root, v: View) -> View:
        """The declared layout of an op result that has the shape and layout of operand view ``v``."""
        return View(root, v.form == "hm", v.form, v.n, v.heads, v.hd, 1, v.rows)

    def _register(self, c: Cand):
        self.decl[c.root] = c.out
        return c

    # ---------------------------------------------------------------- patterns (§5.3)
    def collect(self) -> list[Cand]:
        cands: list[Cand] = []
        for node in self.nodes:
            if node.op != "call_function":
                continue
            c = None
            for fn in (self._m_attn, self._m_mm, self._m_rope, self._m_rms, self._m_glu,
                       self._m_unary, self._m_binary, self._m_scale):
                c = fn(node)
                if c is not None:
                    break
            if c is not None:
                cands.append(self._register(c))
            else:
                self._why_none(node)
        return cands

    _NO_KIND = {
        "native_layer_norm": "layer norm has no firmware kind",
        "layer_norm": "layer norm has no firmware kind",
        "softmax": "softmax has no firmware kind",
        "_softmax": "softmax has no firmware kind",
        "embedding": "embedding lookup has no lowering in this version",
        "bmm": "matmul of two activations has no firmware kind",
        "baddbmm": "matmul of two activations has no firmware kind",
        "erf": "erf has no firmware kind",
    }

    def _why_none(self, node):
        if node in self.why:
            return
        t = node.target
        if t is operator.getitem:
            self.why[node] = "element of a tuple result of a CPU node"
            return
        base = str(t)
        try:
            base = t._schema.name.split("::")[-1]
        except AttributeError:
            pass
        sh = _shape(node)
        if sh and _is_f32(node) and _int(sh[-1]) is None:
            self.why[node] = "row length is symbolic (a row length must be an integer)"
        elif base in self._NO_KIND:
            self.why[node] = self._NO_KIND[base]
        elif base in ("gelu",):
            self.why[node] = "gelu is not the tanh form"
        else:
            self.why[node] = f"no lowering for aten.{base}"

    # -- matmul
    def _weight_t(self, b):
        """The weight node of ``t(W)`` / ``transpose(W, 0, 1)`` / ``permute(W, [1, 0])``, or None."""
        if not _is_call(b, _targets("t.default", "transpose.int", "permute.default")):
            return None
        if b.target is aten.transpose.int and {b.args[1] % 2, b.args[2] % 2} != {0, 1}:
            return None
        if b.target is aten.permute.default and list(b.args[1]) not in ([1, 0], [-1, -2]):
            return None
        w = b.args[0]
        return w if w in self.weights else None

    def _m_mm(self, node):
        t = node.target
        bias = None
        if t is aten.mm.default:
            a, b = node.args[:2]
            op = "mm"
        elif t is aten.addmm.default:
            bias, a, b = node.args[:3]
            op = "addmm"
            if node.kwargs.get("beta", 1) != 1 or node.kwargs.get("alpha", 1) != 1:
                self.note(node, "addmm with alpha / beta other than 1")
                return None
        elif t is aten.matmul.default:
            a, b = node.args[:2]
            op = "matmul"
        elif t is aten.linear.default:
            a, w = node.args[:2]
            bias = _arg(node, 2, "bias")
            b = None
            op = "linear"
        elif _LINEAR_BLOCKS is not None and t is _LINEAR_BLOCKS:
            a, blocks, fmt, K = node.args[:4]
            b = None
            op = "linear_blocks"
        else:
            return None
        why = self._mm_check(node, op)
        if why:
            self.note(node, why)
            return None
        if op == "linear":
            root = w if w in self.weights else None
        elif op == "linear_blocks":
            root = blocks if blocks in self.weights else None
        else:
            root = self._weight_t(b)
        if root is None:
            self.note(node, "no weight operand of the form t(parameter): activation x activation, "
                            "a weight on the left or a weight seen through other views")
            return None
        wv = _tensor(root)
        if op == "linear_blocks":
            N, Kd = _int(_shape(node)[-1]), int(K)
            spec = WeightSpec(root, N, Kd, "blocks", fmt)
        else:
            if wv.dim() != 2 or wv.dtype != torch.float32:
                self.note(node, "weight is not a 2D float32 tensor")
                return None
            N, Kd = _int(wv.shape[0]), _int(wv.shape[1])
            if N is None or Kd is None:
                self.note(node, "symbolic weight shape")
                return None
            if Kd % BLOCK != 0 or N < 1:
                self.note(node, f"K = {Kd} is not a multiple of {BLOCK}" if Kd % BLOCK else "N = 0")
                return None
            if wv.device.type != "cpu":
                self.note(node, f"weight is on {wv.device}")
                return None
            spec = WeightSpec(root, N, Kd)
        if N is None or Kd % BLOCK != 0:
            self.note(node, "bad weight shape")
            return None
        xv, why = self.dev(a)
        if xv is None:
            self.note(node, f"input is not a float32 row value ({why})")
            return None
        if xv.heads is not None or xv.n != Kd or xv.form != "tm":
            self.note(node, f"input rows are not of length K = {Kd}")
            return None
        s = _shape(node)
        if len(s) < 2 or _int(s[-1]) != N or not _eq(_prod(s[:-1]), xv.rows):
            self.note(node, "output shape is not [rows, N]")
            return None
        fields = {"w": W(spec), "x": _v(xv), "addend": None}
        owned, chain, bypass = [node], list(xv.chain), []
        if b is not None and b is not root and all(u is node for u in b.users):
            owned.append(b)                       # the t(W) node dies with the matmul
        if bias is not None:
            cs = self.const_row(bias) if isinstance(bias, fx.Node) else None
            if cs is None or cs.n != N:
                self.note(node, "bias is not a parameter row of length N")
                return None
            fields["addend"] = C(cs)
        out = View(node, False, "tm", N, None, None, 1, xv.rows)
        return Cand("mul_mat", node, fields, out, owned, bypass, chain)

    def _mm_check(self, node, op) -> str:
        if _tensor(node) is None or not _is_f32(node):
            return "output is not float32"
        return ""

    # -- binary, scale, unary
    def _m_binary(self, node):
        t = node.target
        if t is aten.add.Tensor:
            kind = "add"
            if node.kwargs.get("alpha", 1) != 1:
                return None
        elif t is aten.mul.Tensor:
            kind = "mul"
        else:
            return None
        a, b = node.args[:2]
        if self.scalar(a) is not None or self.scalar(b) is not None:
            return None
        if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)) or not _is_f32(node):
            return None
        for x, y in ((a, b), (b, a)):
            cs = self.const_row(y)
            if cs is not None and self.const_row(x) is None:
                xv, why = self.dev(x)
                if xv is None:
                    continue
                if xv.form != "tm" or xv.heads is not None or xv.n != cs.n or not _same_shape(_shape(node), _shape(x)):
                    continue
                f = {"a": _v(xv), "b": C(cs)}
                return Cand(kind, node, f, self._out_like(node, xv), [node], [], list(xv.chain))
        va, wa = self.dev(a)
        vb, wb = self.dev(b)
        if va is None or vb is None:
            return None
        if not (va.same_layout(vb) and _same_shape(_shape(a), _shape(b)) and _same_shape(_shape(a), _shape(node))):
            self.note(node, "operands differ in shape (broadcast) or layout")
            return None
        f = {"a": _v(va), "b": _v(vb)}
        return Cand(kind, node, f, self._out_like(node, va), [node], [], list(va.chain) + list(vb.chain))

    def _m_scale(self, node):
        t = node.target
        if t not in _targets("mul.Tensor", "div.Tensor", "add.Tensor", "sub.Tensor"):
            return None
        a, b = node.args[:2]
        if node.kwargs.get("alpha", 1) != 1:
            return None
        for x, c, swapped in ((a, b, False), (b, a, True)):
            sc_ = self.scalar(c) if isinstance(c, fx.Node) or _num(c) is not None else None
            if sc_ is None or not isinstance(x, fx.Node) or self.scalar(x) is not None:
                continue
            cv, cchain = sc_
            if t is aten.div.Tensor and swapped:
                continue
            if t is aten.sub.Tensor and swapped:
                continue
            xv, why = self.dev(x)
            if xv is None or not _is_f32(node) or not _same_shape(_shape(x), _shape(node)):
                continue
            if xv.rep != 1:
                continue
            lit = not isinstance(cv, Scalar)
            if lit and t is aten.div.Tensor and (cv == 0.0 or not math.isfinite(cv)):
                self.note(node, "division by zero or a non-finite scalar")
                return None
            if t is aten.mul.Tensor:
                s, bb = (f32(cv) if lit else cv), 0.0
            elif t is aten.div.Tensor:
                s, bb = (f32(1.0 / cv) if lit else Scalar(cv.node, "recip")), 0.0
            elif t is aten.add.Tensor:
                s, bb = 1.0, (f32(cv) if lit else cv)
            else:
                s, bb = 1.0, (f32(-cv) if lit else Scalar(cv.node, "neg"))
            if lit and not (math.isfinite(s) and math.isfinite(bb)):
                self.note(node, "scalar not finite in float32")
                return None
            owned = [node] + ([c] if isinstance(c, fx.Node) and len(c.users) == 1 and lit else [])
            f = {"x": _v(xv), "s": s, "b": bb}
            return Cand("scale", node, f, self._out_like(node, xv), owned, [n for n in cchain], list(xv.chain))
        return None

    def _m_unary(self, node):
        t = node.target
        func = None
        if t is aten.silu.default:
            func = "silu"
        elif t is aten.sigmoid.default:
            func = "sigmoid"
        elif t is aten.tanh.default:
            func = "tanh"
        elif t is aten.exp.default:
            func = "exp"
        elif t is aten.gelu.default:
            approx = _arg(node, 1, "approximate", "none")
            if approx != "tanh":
                self.note(node, "gelu approximate='none' (erf) has no firmware kind; only the tanh form")
                return None
            func = "gelu"
        else:
            return None
        xv, why = self.dev(node.args[0])
        if xv is None or not _same_shape(_shape(node.args[0]), _shape(node)):
            return None
        f = {"x": _v(xv), "func": UNARY_FUNC[func]}
        return Cand("unary", node, f, self._out_like(node, xv), [node], [], list(xv.chain))

    def _act(self, g):
        """(variant, input node) when g is silu(x) or gelu(x, approximate='tanh'), else None."""
        if _is_call(g, _targets("silu.default")):
            return 0, g.args[0]
        if _is_call(g, _targets("gelu.default")) and _arg(g, 1, "approximate", "none") == "tanh":
            return 2, g.args[0]
        return None

    def _m_glu(self, node):
        if node.target is not aten.mul.Tensor:
            return None
        a, b = node.args[:2]
        if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
            return None
        for g, u in ((a, b), (b, a)):
            act = self._act(g)
            if act is None or g.users.keys() != {node} or g is u:
                continue
            variant, gin = act
            gv, _ = self.dev(gin)
            uv, _ = self.dev(u)
            if gv is None or uv is None:
                continue
            if not (gv.same_layout(uv) and _same_shape(_shape(gin), _shape(u)) and _same_shape(_shape(gin), _shape(node))):
                self.note(node, "gate and up differ in shape or layout")
                return None
            f = {"gate": _v(gv), "up": _v(uv), "variant": variant}
            return Cand("glu", node, f, self._out_like(node, gv), [node, g], [], list(gv.chain) + list(uv.chain))
        return None

    # -- RMSNorm
    def _rms_core(self, m3):
        """m3 = x * rsqrt(mean(x ** 2, -1, keepdim) + eps): (x node, eps, owned nodes, nodes read) or None."""
        if not _is_call(m3, _targets("mul.Tensor")):
            return None
        for x, r in (m3.args[:2], m3.args[:2][::-1]):
            if not _is_call(r, _targets("rsqrt.default")):
                continue
            ad = r.args[0]
            if not _is_call(ad, _targets("add.Tensor")) or ad.kwargs.get("alpha", 1) != 1:
                continue
            ep = self.scalar(ad.args[1]) if isinstance(ad.args[1], fx.Node) or _num(ad.args[1]) is not None else None
            mean = ad.args[0]
            if ep is None or not _is_call(mean, _targets("mean.dim")):
                continue
            eps, eps_nodes = ep
            dims = mean.args[1]
            keep = _arg(mean, 2, "keepdim", False)
            sx = _shape(x)
            if sx is None or not keep or list(dims) not in ([-1], [len(sx) - 1]):
                continue
            pw = mean.args[0]
            if not _is_call(pw, _targets("pow.Tensor_Scalar")) or _num(pw.args[1]) != 2.0:
                continue
            x2 = pw.args[0]
            if not (x2 is x or self._same_value(x, x2)):
                continue
            chain = [m3, r, ad, mean, pw]
            if not (r.users.keys() == {m3} and ad.users.keys() == {r} and mean.users.keys() == {ad}
                    and pw.users.keys() == {mean}):
                continue
            return x, (f32(eps) if not isinstance(eps, Scalar) else eps), chain, eps_nodes
        return None

    def _same_value(self, a, b) -> bool:
        va, _ = self.view_of(a)
        vb, _ = self.view_of(b)
        return va is not None and vb is not None and va.key() == vb.key() and _same_shape(_shape(a), _shape(b))

    def _m_rms(self, node):
        if node.target is not aten.mul.Tensor:
            return None
        a, b = node.args[:2]
        if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
            return None
        core = None
        wspec = None
        extra_owned, bypass = [], []
        for p, q in ((a, b), (b, a)):
            c = self._rms_core(p)
            if c is None or p.users.keys() != {node}:
                continue
            ws = self._norm_weight(q)
            if ws is not None:
                core, wspec, wnodes = c, ws[0], ws[1]
                extra_owned = [w for w in wnodes if w.users.keys() == {node}]
                bypass = [w for w in wnodes if w not in extra_owned]
                break
        if core is None:
            c = self._rms_core(node)
            if c is None:
                return None
            core = c
        x, eps, chain, eps_nodes = core
        bypass = bypass + eps_nodes
        xv, why = self.dev(x)
        if xv is None:
            self.note(node, f"norm input is not a float32 row value ({why})")
            return None
        if xv.form != "tm" or xv.heads is not None:
            self.note(node, "norm over a split (per head) tail has no kind")
            return None
        if not _same_shape(_shape(node), _shape(x)):
            return None
        if wspec is not None and wspec.n != xv.n:
            self.note(node, "norm weight length differs from the row length")
            return None
        owned = ([node] + chain) if wspec is not None else chain     # chain[0] is the root when unweighted
        f = {"x": _v(xv), "weight": C(wspec) if wspec else None, "eps": eps}
        return Cand("rms_norm", node, f, self._out_like(node, xv), owned + extra_owned, bypass, list(xv.chain))

    def _norm_weight(self, q):
        """(ConstSpec, [nodes]) for the weight of a norm: w, or 1 + w (the Gemma form)."""
        cs = self.const_row(q)
        if cs is not None:
            return cs, []
        if _is_call(q, _targets("add.Tensor")) and q.kwargs.get("alpha", 1) == 1:
            w, one = q.args[:2]
            if _lit(one) == 1.0 and isinstance(w, fx.Node):
                cs = self.const_row(w)
                if cs is not None:
                    return ConstSpec(cs.node, "one_plus", cs.n), [q]
        return None

    # -- RoPE
    def _strip(self, node):
        """Skip float32 identity ops (_to_copy, alias, clone)."""
        while _is_call(node, _IDENT) and _is_f32(node.args[0]) and _is_f32(node):
            node = node.args[0]
        return node

    def _m_rope(self, node):
        if node.target is not aten.add.Tensor or node.kwargs.get("alpha", 1) != 1:
            return None
        A, B = node.args[:2]
        if not (_is_call(A, _targets("mul.Tensor")) and _is_call(B, _targets("mul.Tensor"))):
            return None
        why_last = ""
        for ta, tb in ((A, B), (B, A)):
            r = self._rope_try(node, ta, tb)
            if isinstance(r, Cand):
                return r
            if r:
                why_last = r
        if why_last:
            self.note(node, why_last)
        return None

    def _rope_try(self, root, ta, tb):
        """ta = x * cos, tb = rotate_half(x) * sin. A Cand, a reason string (near miss) or None."""
        for x, cu in (ta.args[:2], ta.args[:2][::-1]):
            if not isinstance(x, fx.Node) or not isinstance(cu, fx.Node):
                continue
            cosi = self._trig(cu, "cos")
            if cosi is None:
                continue
            for rot, su in (tb.args[:2], tb.args[:2][::-1]):
                if not isinstance(rot, fx.Node) or not isinstance(su, fx.Node):
                    continue
                sini = self._trig(su, "sin")
                if sini is None:
                    continue
                if not _is_call(rot, _targets("cat.default")):
                    continue
                res = self._rope_rot(rot, x)
                if res is None:
                    return "rotate_half is not cat([-x[..., D/2:], x[..., :D/2]], -1) of the rotated tensor"
                rot_owned = res
                (cos_f, cos_pos, cos_inv, cos_nodes), (sin_f, sin_pos, sin_inv, sin_nodes) = cosi, sini
                if cos_f != sin_f or cos_pos is not sin_pos or cos_inv.node is not sin_inv.node:
                    return "cos and sin tables differ (factor, positions or frequencies)"
                xv, why = self.dev(x, "hm")
                if xv is None or xv.form != "hm" or xv.rep != 1:
                    return f"rotated tensor is not a head-major [1, H, T, D] value ({why})"
                D = xv.hd
                if cos_inv.n * 2 != D:
                    return "the table has not D / 2 angles (partial rotary is not supported)"
                spos = _shape(cos_pos)
                if spos is None or not _eq(_prod(spos), xv.rows):
                    return "positions do not have T entries"
                owned = [root, ta, tb] + rot_owned
                if not all(o.users.keys() <= set(owned) for o in owned if o is not root):
                    return "rotary interior value is used elsewhere"
                f = {"x": _v(xv), "n_heads": xv.heads, "head_dim": D, "n_dims": D, "table": C(cos_inv),
                     "pos": P(cos_pos), "factor": cos_f}
                out = View(root, True, "hm", xv.n, xv.heads, D, 1, xv.rows)
                return Cand("rope", root, f, out, owned, cos_nodes + sin_nodes, list(xv.chain))
        return None

    def _rope_rot(self, cat, x):
        """cat([neg(x[..., D/2:]), x[..., :D/2]], -1): the owned nodes, or None."""
        parts = cat.args[0]
        dim = _arg(cat, 1, "dim", 0)
        sx = _shape(x)
        if sx is None or len(parts) != 2 or dim not in (-1, len(sx) - 1):
            return None
        ng, lo = parts
        if not _is_call(ng, _targets("neg.default")):
            return None
        hi = ng.args[0]
        if not (_is_call(hi, _targets("slice.Tensor")) and _is_call(lo, _targets("slice.Tensor"))):
            return None
        D = _int(sx[-1])
        if D is None or D % 2:
            return None
        h = D // 2
        for sl, start, end in ((hi, h, (D, 9223372036854775807)), (lo, 0, (h,))):
            src, d = sl.args[0], sl.args[1]
            st = _arg(sl, 2, "start", 0)
            en = _arg(sl, 3, "end", 9223372036854775807)
            step = _arg(sl, 4, "step", 1)
            if not (src is x or self._same_value(src, x)) or d not in (-1, len(sx) - 1) or st != start \
                    or en not in end or step != 1:
                return None
        return [cat, ng, hi, lo]

    def _trig(self, u, fn):
        """unsqueeze(m, 1) with m = cos / sin(cat([f, f], -1)) [* factor]: (factor, pos node, inv_freq
        ConstSpec, [nodes read]) or None."""
        if not (_is_call(u, _targets("unsqueeze.default")) and u.args[1] in (1, -3)):
            return None
        m = self._strip(u.args[0])
        nodes = [u]
        factor = 1.0
        if _is_call(m, _targets("mul.Tensor")):
            fv = self.scalar(m.args[1]) if isinstance(m.args[1], fx.Node) or _num(m.args[1]) is not None else None
            if fv is None or not isinstance(m.args[0], fx.Node):
                return None
            factor = f32(fv[0]) if not isinstance(fv[0], Scalar) else fv[0]
            nodes.append(m)
            nodes.extend(fv[1])
            m = m.args[0]
        m = self._strip(m)
        if not _is_call(m, _targets(f"{fn}.default")):
            return None
        nodes.append(m)
        c = self._strip(m.args[0])
        if not (_is_call(c, _targets("cat.default")) and len(c.args[0]) == 2 and c.args[0][0] is c.args[0][1]
                and _arg(c, 1, "dim", 0) in (-1, len(_shape(c)) - 1)):
            return None
        nodes.append(c)
        f = self._strip(c.args[0][0])
        if not _is_call(f, _targets("mul.Tensor")):
            return None
        nodes.append(f)
        for pf, inv in (f.args[:2], f.args[:2][::-1]):
            if not (isinstance(pf, fx.Node) and isinstance(inv, fx.Node)):
                continue
            cs = self.const_row(inv)
            if cs is None:
                continue
            pos = self._int_source(pf, nodes)
            if pos is not None:
                return factor, pos, cs, nodes
        return None

    def _int_source(self, node, nodes):
        """The integer tensor that float positions come from, through _to_copy / unsqueeze / view."""
        cur = node
        for _ in range(8):
            v = _tensor(cur)
            if v is None:
                return None
            if not v.dtype.is_floating_point:
                return cur if v.dtype in (torch.int32, torch.int64) else None
            if cur.op == "call_function" and (cur.target in _IDENT or cur.target in _VIEW_RESHAPE
                                              or cur.target is aten.expand.default):
                nodes.append(cur)
                cur = cur.args[0]
                continue
            return None
        return None

    # -- attention
    def _m_attn(self, node):
        sd = node
        getitem = False
        if node.target is operator.getitem and node.args[1] == 0 and _is_call(node.args[0], _SDPA_CPU):
            sd, getitem = node.args[0], True
        elif node.target in _SDPA:
            pass
        else:
            return None
        if self.opts.attention == "cpu":
            return self._attn_fail(node, sd, "attention stays on the CPU by the option attention='cpu'")
        if sd.target in _SDPA_CPU:
            q, k, v = sd.args[:3]
            dropout = _arg(sd, 3, "dropout_p", 0.0)
            causal = _arg(sd, 4, "is_causal", False)
            mask = sd.kwargs.get("attn_mask")
            scale = sd.kwargs.get("scale")
            gqa = False
        else:
            q, k, v = sd.args[:3]
            mask = _arg(sd, 3, "attn_mask")
            dropout = _arg(sd, 4, "dropout_p", 0.0)
            causal = _arg(sd, 5, "is_causal", False)
            scale = _arg(sd, 6, "scale")
            gqa = bool(_arg(sd, 7, "enable_gqa", False))
        why = self._attn_check(node, sd, getitem, dropout, causal, mask, gqa)
        if why:
            return self._attn_fail(node, sd, why)
        qv, wq = self.dev(q, "hm")
        kv, wk = self.dev(k, "hm", rep_ok=True)
        vv, wv = self.dev(v, "hm", rep_ok=True)
        for nm, vw, w in (("q", qv, wq), ("k", kv, wk), ("v", vv, wv)):
            if vw is None or vw.form != "hm":
                return self._attn_fail(node, sd, f"{nm} is not a head-major [1, H, T, D] device value ({w})")
        if qv.rep != 1:
            return self._attn_fail(node, sd, "q is a grouped-KV expansion")
        if not (_eq(qv.rows, kv.rows) and _eq(qv.rows, vv.rows)):
            return self._attn_fail(node, sd, "k / v are not fresh rows: their row count differs from q's "
                                             "(cache or cross attention)")
        if kv.hd != qv.hd or vv.hd != qv.hd or kv.heads != vv.heads or kv.rep != vv.rep:
            return self._attn_fail(node, sd, "q, k and v differ in head size or head count")
        n_head, d = qv.heads, qv.hd
        if kv.rep > 1:
            heads_ok = kv.heads == n_head
        else:
            heads_ok = kv.heads == n_head or (gqa and n_head % kv.heads == 0)
        if not heads_ok:
            return self._attn_fail(node, sd, "q and k differ in head count")
        n_kv = kv.heads // kv.rep
        if scale is None:
            sc = f32(1.0 / math.sqrt(d))
        else:
            if _num(scale) is None:
                return self._attn_fail(node, sd, "symbolic attention scale")
            sc = f32(scale)
        mref, bypass, owned_extra = None, [], []
        if mask is not None:
            mr = self._attn_mask(mask, qv)
            if isinstance(mr, str):
                return self._attn_fail(node, sd, mr)
            mref, bypass = mr
        elif not causal:
            return self._attn_fail(node, sd, "attention without a mask and not causal")
        owned = [node] if not getitem else [node, sd]
        for u in (sd.users if getitem else []):
            if u is node:
                continue
            if u.target is operator.getitem and not u.users:
                owned.append(u)
        f = {"q": _v(qv), "k": _v(kv), "v": _v(vv), "n_head": n_head, "n_head_kv": n_kv, "d": d, "scale": sc,
             "mask": mref}
        out = View(node, True, "hm", qv.n, n_head, d, 1, qv.rows)
        chain = list(qv.chain) + list(kv.chain) + list(vv.chain)
        return Cand("attn", node, f, out, owned, bypass, chain)

    def _attn_fail(self, node, sd, why):
        self.note(node, why)
        self.why[sd] = why                  # the call is visited before its getitem: replace the generic reason
        return None

    def _attn_check(self, node, sd, getitem, dropout, causal, mask, gqa) -> str:
        if _num(dropout) != 0.0:
            return "attention with dropout"
        if not isinstance(causal, bool):
            return "symbolic is_causal"
        if causal and mask is not None:
            return "is_causal together with a mask"
        if getitem:
            for u in sd.users:
                if u is not node and not (u.target is operator.getitem and not u.users):
                    return "attention returns more than its output to other nodes"
        s = _shape(node)
        if s is None or len(s) != 4 or _int(s[0]) != 1:
            return "attention with batch other than 1"
        if not _is_f32(node):
            return "attention output is not float32"
        return ""

    def _attn_mask(self, mask, qv: View):
        """(mask reference, bypassed nodes) or a reason."""
        if not isinstance(mask, fx.Node):
            return "attention mask is not a tensor"
        T = qv.rows
        bypass = []
        node, form = mask, None
        if _is_call(mask, _targets("where.self", "where.Scalar")):
            cond, a, b = mask.args[:3]
            if _lit(a) == 0.0 and _lit(b) == -_INF and isinstance(cond, fx.Node) and _tensor(cond) is not None \
                    and _tensor(cond).dtype == torch.bool:
                node, form = cond, "bool"
                bypass = [mask] + [x for x in (a, b) if isinstance(x, fx.Node)]
        if form is None:
            tv = _tensor(mask)
            if tv is None:
                return "attention mask without metadata"
            if tv.dtype == torch.bool:
                form = "bool"
            elif tv.dtype == torch.float32:
                form = "additive"
            else:
                return f"attention mask dtype {tv.dtype}"
        s = _shape(node)
        if s is None or len(s) < 2 or not (_eq(s[-1], T) and _eq(s[-2], T)) or any(_int(d) != 1 for d in s[:-2]):
            return "attention mask is not [T, T] (batch, heads or a cache length in it)"
        return M(node, form), bypass

    # ---------------------------------------------------------------- the fold
    def fold(self, cands: list[Cand]) -> list[Cand]:
        """add(x, mul_mat result) with no other user and no addend -> mul_mat with addend x."""
        by_root = {c.root: c for c in cands}
        dead = set()
        out = []
        for c in cands:
            if c.kind != "add" or c in dead:
                out.append(c)
                continue
            done = False
            for mine, other in (("a", "b"), ("b", "a")):
                mv, ov = c.fields[mine], c.fields[other]
                if not (isinstance(mv, V) and isinstance(ov, V)):
                    continue
                mm = by_root.get(mv.view.base)
                if mm is None or mm.kind != "mul_mat" or mm.fields["addend"] is not None or mm in dead:
                    continue
                if mv.view.heads is not None or mv.view.form != "tm":
                    continue
                if ov.view.base is mm.root or self._depends(ov.view.base, mm.root):
                    continue
                between = set(mv.view.chain)
                if not (self._only_chain_to(mm.root, c.root) and ov.view.same_layout(mv.view)):
                    continue
                fields = dict(mm.fields)
                fields["addend"] = ov
                merged = Cand("mul_mat", c.root, fields, c.out, mm.owned + list(between) + c.owned,
                              mm.bypassed + c.bypassed, mm.chain + c.chain + list(ov.view.chain),
                              parts=(mm, c))
                dead.add(mm)
                out = [x for x in out if x is not mm]
                out.append(merged)
                done = True
                break
            if not done:
                out.append(c)
        return out

    def _only_chain_to(self, root, add_root) -> bool:
        """Every path of users from ``root`` ends at ``add_root`` (the mul_mat result has no other use)."""
        seen = set()
        stack = [root]
        while stack:
            n = stack.pop()
            for u in n.users:
                if u is add_root:
                    continue
                v, _ = self.view_of(u)
                if v is None or v.base is not root or v.rep != 1:
                    return False
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        return True

    def _depends(self, node, target) -> bool:
        """True if ``node`` has ``target`` among its ancestors."""
        lo = self.index[target]
        seen, stack = set(), [node]
        while stack:
            n = stack.pop()
            if n is target:
                return True
            if n in seen or self.index[n] <= lo:
                continue
            seen.add(n)
            stack.extend(n.all_input_nodes)
        return False

    # ---------------------------------------------------------------- ownership and the probe
    def resolve_ownership(self, cands: list[Cand]) -> list[Cand]:
        claimed: set = set()
        kept = []
        for c in sorted(cands, key=lambda c: self.index[c.root], reverse=True):
            if c.root in claimed:
                continue
            kept.append(c)
            claimed.update(o for o in c.owned if o is not c.root)
        kept.sort(key=lambda c: self.index[c.root])
        return kept

    def probe_cands(self, cands: list[Cand]) -> list[Cand]:
        if self.probe is None:
            return cands
        memo: dict = {}
        out = []
        for c in cands:
            why = self._probe_one(c, memo)
            if why and c.parts:
                ok = []
                for p in c.parts:
                    w = self._probe_one(p, memo)
                    if w:
                        self.note(p.root, f"device: {w}")
                    else:
                        ok.append(p)
                out.extend(ok)
                continue
            if why:
                self.note(c.root, f"device: {why}")
                continue
            out.append(c)
        return out

    def _probe_one(self, c: Cand, memo) -> str | None:
        b = _Builder()
        b.emit(c, self.index[c.root])
        ops = [{k: (v.apply(1.0) if isinstance(v, Scalar) else v) for k, v in o.items()} for o in b.ops]
        prog = {"values": b.values, "ops": ops, "kv_size": self.opts.kv_size,
                "weights": [{"N": w.N, "K": w.K, "kind": w.kind, "fmt": w.fmt} for w in b.weights]}
        key = repr([{k: v for k, v in o.items() if k != "tag"} for o in b.ops] + [b.values, prog["weights"]])
        if key not in memo:
            memo[key] = self.probe(prog)
        return memo[key]

    # ---------------------------------------------------------------- regions
    def form(self, cands: list[Cand]) -> list[Region]:
        cands = list(cands)
        stage: dict = {}
        rounds = 0
        while True:
            rounds += 1
            if rounds > 4 * len(cands) + 16:
                raise RuntimeError("lower: cannot separate the regions")
            groups = self._groups(cands, stage)
            viol = self._violations(cands, groups)
            if viol:
                changed = False
                for a, b in viol:
                    if stage.get(b, 0) <= stage.get(a, 0):
                        stage[b] = stage.get(a, 0) + 1
                        changed = True
                if not changed:
                    raise RuntimeError("lower: cannot make the regions convex")
                continue
            cyc = self._cycle_edges(groups)
            if cyc:
                # regions that read each other through nodes outside every region: the cand that reads
                # moves to a later stage, which cuts it from the region of the cand it reads through
                for a, b in cyc:
                    if stage.get(b, 0) <= stage.get(a, 0):
                        stage[b] = stage.get(a, 0) + 1
                        break
                else:
                    raise RuntimeError("lower: the regions read each other")
                continue
            regions, failed = self._build(groups)
            if failed:
                cands = [c for c in cands if c not in failed]
                for c in failed:
                    self.note(c.root, "output shape or layout cannot be rebuilt from the region inputs")
                continue
            return regions

    def _groups(self, cands, stage) -> list[list[Cand]]:
        prod = {c.root: c for c in cands}
        parent = {c: c for c in cands}

        def find(c):
            while parent[c] is not c:
                parent[c] = parent[parent[c]]
                c = parent[c]
            return c

        for c in cands:
            for v in c.operands():
                p = prod.get(v.view.base)
                if p is not None and p is not c and stage.get(p, 0) == stage.get(c, 0) and p.out.n == v.view.n:
                    parent[find(c)] = find(p)
        groups: dict = {}
        for c in cands:
            groups.setdefault(find(c), []).append(c)
        return [sorted(g, key=lambda c: self.index[c.root]) for g in groups.values()]

    def _region_nodes(self, g: list[Cand]) -> dict:
        m = {}
        for c in g:
            for n in c.owned + c.chain:
                m.setdefault(n, c)
            m[c.root] = c
        return m

    def _violations(self, cands, groups) -> list:
        """(a, b): cand b of a region reads, through a node outside the region, a value of cand a."""
        out = []
        for g in groups:
            if len(g) < 2:
                continue
            S = self._region_nodes(g)
            own = {n: c for c in g for n in c.owned}
            own.update({c.root: c for c in g})
            lo = min(self.index[n] for n in S)
            prod = {c.root: c for c in g}
            for b in g:
                seeds = []
                for v in b.operands():
                    if v.view.base not in prod:
                        seeds.append(v.view.base)
                for f in b.fields.values():
                    if isinstance(f, (P, M)):
                        seeds.append(f.node)
                for s in seeds:
                    for a in self._read_through(s, S, own, prod, lo):
                        if a is not b:
                            out.append((a, b))
        return out

    def _read_through(self, start, S, own, prod, lo) -> list:
        """The cands of the region whose values the node ``start`` reads, going back through nodes outside the
        region. A view node in the region that is only the chain of an operand shows the value of its base: it
        counts as the cand that makes the base, and when the base is not made in the region the walk goes on
        through it."""
        seen, stack, hit = set(), [start], []
        while stack:
            n = stack.pop()
            if n in seen or self.index[n] < lo:
                continue
            seen.add(n)
            if n in own:
                hit.append(own[n])
                continue
            if n in S:
                v = self.view_of(n)[0] if n.op == "call_function" else None
                if v is not None and v.base in prod:
                    hit.append(prod[v.base])
                    continue
            stack.extend(n.all_input_nodes)
        return hit

    def _reach(self, owner: dict) -> dict:
        """For every node of the graph, the units (values of ``owner``: node -> unit) it reads, going back
        through nodes outside every unit. A node of ``owner`` reads itself and stops the walk."""
        reach: dict = {}
        for n in sorted(self.index, key=self.index.get):
            if n in owner:
                reach[n] = frozenset((owner[n],))
                continue
            acc: frozenset = frozenset()
            for i in n.all_input_nodes:
                r = reach.get(i)
                if r:
                    acc = acc | r
            reach[n] = acc
        return reach

    def _cycle_edges(self, groups) -> list:
        """(a, b) pairs, cand b of a region reads a value of cand a of another region, through nodes outside
        every region, where the two regions also depend on each other the other way around (a cycle).
        The latest reader comes first."""
        live = [g for g in groups if len(g) >= self.opts.min_ops]
        if len(live) < 2:
            return []
        gid = {c: i for i, g in enumerate(live) for c in g}
        roots = {c.root: c for c in gid}
        owner: dict = {}
        for c in gid:
            for n in c.owned:
                owner.setdefault(n, c)
            owner[c.root] = c
        for c in gid:
            for n in c.chain:
                if n not in owner and n.op == "call_function":
                    v = self.view_of(n)[0]
                    if v is not None and v.base in roots:
                        owner[n] = roots[v.base]
        reach = self._reach(owner)
        edges = []
        for b in gid:
            seeds = [v.view.base for v in b.operands()]
            seeds += [f.node for f in b.fields.values() if isinstance(f, (P, M))]
            for s in seeds:
                for a in reach.get(s, ()):
                    if gid[a] != gid[b]:
                        edges.append((a, b))
        reads: dict = {}
        for a, b in edges:
            reads.setdefault(gid[b], set()).add(gid[a])

        def reaches(src, dst):
            seen, stack = set(), [src]
            while stack:
                x = stack.pop()
                if x == dst:
                    return True
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(reads.get(x, ()))
            return False

        out = [(a, b) for a, b in edges if reaches(gid[a], gid[b])]
        out.sort(key=lambda e: -self.index[e[1].root])
        return out

    def _closure(self, root) -> list:
        """Nodes that are plain views of the device value of ``root`` (the root included)."""
        out = [root]
        seen = {root}
        stack = [root]
        while stack:
            n = stack.pop()
            for u in n.users:
                if u in seen or u.op != "call_function":
                    continue
                v, _ = self.view_of(u)
                if v is not None and v.base is root and v.rep == 1 and v.chain:
                    if v.form == "hm" and u.target in _COPY:    # a copy is contiguous; the rebuilt output is not
                        continue
                    seen.add(u)
                    out.append(u)
                    stack.append(u)
        return out

    def _build(self, groups):
        regions: list[Region] = []
        failed: list[Cand] = []
        covered_by: dict = {}
        live = [g for g in groups if len(g) >= self.opts.min_ops]
        for g in groups:
            if len(g) < self.opts.min_ops:
                for c in g:
                    for n in c.owned:
                        self.note(n, "a region has at least two lowered ops (a lone matmul stays a device_mm)")
        for g in live:
            r, f = self._build_region(g)
            if f:
                failed.extend(f)
            else:
                regions.append(r)
        if failed:
            return [], failed
        regions = self._ordered(regions)
        for i, r in enumerate(regions):
            r.id = i
        return regions, []

    def _ordered(self, regions: list) -> list:
        """The regions in graph order (by their first op), except that a region comes after every region
        whose value it reads, directly or through nodes outside every region. The rewrite replaces the
        regions in this order: the view nodes between a value and a region that reads it go with the region
        that makes the value, and that region can come later in graph order (the reader has an earlier op
        that does not depend on it)."""
        regions = sorted(regions, key=lambda r: self.index[r.op_nodes[0]])
        if len(regions) < 2:
            return regions
        owner = {n: k for k, r in enumerate(regions) for n in list(r.op_nodes) + [o.node for o in r.outputs]}
        reach = self._reach(owner)
        needs = []
        for k, r in enumerate(regions):
            need: set = set()
            for i in r.inputs:
                need |= reach.get(i.node, frozenset())
            need.discard(k)
            needs.append(need)
        out: list = []
        done: set = set()
        while len(out) < len(regions):
            k = next((k for k in range(len(regions)) if k not in done and needs[k] <= done), None)
            if k is None:
                raise RuntimeError("lower: the regions read each other")
            done.add(k)
            out.append(regions[k])
        return out

    def _build_region(self, g: list[Cand]):
        b = _Builder()
        for c in g:
            b.emit(c, self.index[c.root])
        roots = {c.root for c in g}
        S = set()
        for c in g:
            S.update(c.owned)
        clos = {}
        for c in g:
            for n in self._closure(c.root):
                clos[n] = c
        S.update(clos)
        chain_all = set()
        for c in g:
            chain_all.update(c.chain)
        changed = True
        while changed:
            changed = False
            for n in chain_all:
                if n not in S and n.users and all(u in S for u in n.users):
                    S.add(n)
                    changed = True
        outputs: list[Output] = []
        failed: list[Cand] = []
        for n, c in clos.items():
            if all(u in S for u in n.users):
                continue
            vid = b.produced[c.root]
            v = self.view_of(n)[0]
            rec = self._out_recipe(n, v, b)
            if rec is None:
                failed.append(c)
                continue
            root_rec = None if n is c.root else self._out_recipe(c.root, c.out, b)
            outputs.append(Output(vid, n, rec, c.root, root_rec, storage_root(n, c.root)))
            b.values[vid]["out"] = True
        # outputs that share storage in the aten graph are views of one tensor; a member that needs the
        # contiguous copy cannot be one
        members: dict = {}
        for o in outputs:
            members.setdefault((o.value, o.sroot), []).append(o)
        for grp in members.values():
            if len(grp) > 1 and any(o.recipe["contiguous"] for o in grp):
                failed.append(clos[grp[0].node])
        if failed:
            return None, failed
        bypassed = []
        for c in g:
            for n in c.bypassed:
                if n not in S and n not in bypassed:
                    bypassed.append(n)
        nodes = sorted(S, key=lambda n: self.index[n])
        outputs.sort(key=lambda o: (o.value, self.index[o.node]))
        r = Region(-1, b.values, b.ops, b.inputs, b.consts, b.weights, outputs, nodes, bypassed,
                   g[0].out.rows, self.opts.kv_size, b.op_nodes, scalars=list(b.scalars))
        r.touching = self._touching(r)
        return r, []

    def _out_recipe(self, node, v: View, b: _Builder):
        s = _shape(node)
        if s is None or v is None:
            return None
        if v.form == "hm":
            if len(s) != 4:
                return None
            dims_src = [s[0], s[2], s[1], s[3]]        # [1, T, H, D]; the final node is its transpose
            tr = True
        else:
            dims_src = list(s)
            tr = False
        # the strides the rebuilt value has against the strides eager gives the node (size-1 dims do not count)
        strides = _tensor(node).stride()
        if len(strides) != len(s):
            return None
        if tr:
            T_, H_, D_ = (_x(dims_src[1]), _x(dims_src[2]), _x(dims_src[3]))
            rebuilt = [T_ * H_ * D_, D_, H_ * D_, sympy.Integer(1)]
        else:
            rebuilt = _contiguous_strides(s)
        same = lambda want: all(_eq(a, b) for a, b, d in zip(strides, want, s) if _int(d) != 1)  # noqa: E731
        if same(rebuilt):
            contiguous = False
        elif same(_contiguous_strides(s)):
            contiguous = True
        else:
            return None
        dims = []
        for d in dims_src:
            i = _int(d)
            if i is not None:
                dims.append(i)
                continue
            found = None
            for k, inp in enumerate(b.inputs):
                if inp.kind != "in":
                    continue
                sb = _shape(inp.node)
                for j, bd in enumerate(sb):
                    if _eq(bd, d):
                        found = {"input": k, "dim": j}
                        break
                if found:
                    break
            if found is None:
                return None
            dims.append(found)
        return {"dims": dims, "transpose12": tr, "contiguous": contiguous}

    def _touching(self, r: Region) -> dict:
        inside = set(r.nodes)
        out = {}
        for inp in r.inputs:
            if inp.node.op == "call_function" and inp.node not in inside:
                out[inp.node] = None
        for o in r.outputs:
            for u in o.node.users:
                if u.op == "call_function" and u not in inside:
                    out[u] = None
        return out


# ------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------
def form_regions(gm: fx.GraphModule, weights: dict, probe: Callable | None = None,
                 opts: LowerOpts | dict | None = None) -> tuple[list[Region], dict]:
    """Find the device regions of an aten graph.

    weights   {placeholder / get_attr node: name} of the parameters and buffers (as backend._aot_weights)
    probe     probe(program) -> None | reason, asked once per candidate op on a one-op program (the
              program of ``Region.to_program`` plus the key "weights": [{"N", "K", "kind", "fmt"}] that
              the op's "w" indexes). None accepts everything.
    opts      LowerOpts or a dict (kv_size, min_ops, attention)

    Returns (regions, cpu_reasons). A region is convex and has at least ``min_ops`` ops. cpu_reasons maps
    every call_function node that is not in a region to the reason it stays on the CPU.
    """
    if not isinstance(opts, LowerOpts):
        opts = LowerOpts.from_dict(opts)
    ctx = LowerCtx(gm, weights, probe, opts)
    cands = ctx.collect()
    cands = ctx.resolve_ownership(cands)
    cands = ctx.fold(cands)
    cands = ctx.resolve_ownership(cands)
    cands = ctx.probe_cands(cands)
    regions = ctx.form(cands)
    inside = {n for r in regions for n in r.nodes}
    bypassed = {n for r in regions for n in r.bypassed}
    reasons = {}
    for n in ctx.nodes:
        if n.op != "call_function" or n in inside:
            continue
        if n in bypassed:
            reasons[n] = "read by the lowering only (a cos / sin table, a mask): dead after the rewrite unless used elsewhere"
            continue
        reasons[n] = ctx.why.get(n) or ("view or value computed on the CPU" if n.target in _VIEW_RESHAPE
                                        or n.target in _IDENT else f"no lowering for {_name(n)}")
    for r in regions:
        r.touching = {n: reasons.get(n, "") for n in r.touching}
    return regions, reasons

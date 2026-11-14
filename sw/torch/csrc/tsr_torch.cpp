// tsr_torch.cpp — Python module _tesseract_rt: torch bindings of the Tesseract runtime (sw/rt/tsr_rt.h).
//
// Binds tsr::rt::device as _tesseract_rt.Device (weight upload, linear, reference linear, counters)
// plus a process-wide device singleton (open / get / is_open / close). Tensors cross the boundary as
// torch.Tensor (CPU); the device calls run with the GIL released. Numerics are tsr_rt's: f32 / f16 /
// bf16 weights become Q8_0 at upload with ggml's quantize_row_q8_0_ref, activations are quantized
// by QACT on the device, results are f32. The Python package tesseract_torch (sw/torch) builds the
// torch.compile backend on top of this module.
//
// Graphs (whole-graph lowering, full mode): const_upload / graph_unsupported / graph_create / graph_run / graph_run_ref /
// graph_info bind the graph API of tsr_rt.h. A program is the Python dict of docs/torch-backend.md (values, ops, kv_size),
// parsed strictly here.
//
// Errors: tsr::rt::error -> _tesseract_rt.TesseractError (a RuntimeError); std::invalid_argument ->
// ValueError (pybind11's translation).
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "tsr_rt.h"

namespace py = pybind11;
namespace rt = tsr::rt;

namespace {

rt::dtype dtype_of_scalar(at::ScalarType s) {
    switch (s) {
        case at::kFloat:    return rt::dtype::f32;
        case at::kHalf:     return rt::dtype::f16;
        case at::kBFloat16: return rt::dtype::bf16;
        default: break;
    }
    throw std::invalid_argument(std::string("tesseract: weight dtype ") + c10::toString(s) +
                                " is not supported (float32, float16, bfloat16; or ggml blocks via upload_blocks)");
}

rt::dtype dtype_of_name(const std::string & n) {
    if (n == "f32" || n == "float32") return rt::dtype::f32;
    if (n == "f16" || n == "float16") return rt::dtype::f16;
    if (n == "bf16" || n == "bfloat16") return rt::dtype::bf16;
    if (n == "q8_0") return rt::dtype::q8_0;
    if (n == "q4_k" || n == "q4_K") return rt::dtype::q4_k;
    throw std::invalid_argument("tesseract: unknown weight type '" + n + "' (f32, f16, bf16, q8_0, q4_k)");
}

// dtype given as a torch.dtype or a string
rt::dtype dtype_of_obj(const py::object & o) {
    if (py::isinstance<py::str>(o)) return dtype_of_name(o.cast<std::string>());
    return dtype_of_scalar(torch::python::detail::py_object_to_dtype(o));
}

const char * fmt_name(uint32_t fmt) { return fmt == TSR_FMT_Q4_K ? "q4_k" : "q8_0"; }

uint32_t to_u32(int64_t v, const char * what) {
    if (v < 0 || v > (int64_t) UINT32_MAX) throw std::invalid_argument(std::string("tesseract: ") + what + " out of range");
    return (uint32_t) v;
}

void check_cpu(const at::Tensor & t, const char * what) {
    if (!t.defined()) throw std::invalid_argument(std::string("tesseract: ") + what + " is undefined");
    if (t.device().type() != c10::DeviceType::CPU) {
        throw std::invalid_argument(std::string("tesseract: ") + what + " must be a CPU tensor (got " + t.device().str() + ")");
    }
    if (t.is_sparse() || t.is_quantized() || t.is_complex()) {
        throw std::invalid_argument(std::string("tesseract: ") + what + " must be a dense real tensor");
    }
}

// A 2D tensor whose rows are contiguous (stride(1) == 1) and do not overlap; otherwise a contiguous copy.
at::Tensor rows_contiguous(const at::Tensor & t) {
    const int64_t n = t.size(0), k = t.size(1);
    if (k == 0 || n == 0) return t.contiguous();
    if (t.stride(1) == 1 && (n == 1 || t.stride(0) >= k)) return t;
    return t.contiguous();
}


// ---------------------------------------------------------------------------------------------
// The graph program (a Python dict) -> rt::graph_desc, strictly
// ---------------------------------------------------------------------------------------------
[[noreturn]] void bad_prog(const std::string & what) { throw std::invalid_argument("tesseract: program: " + what); }

std::string op_label(size_t i, int tag, const std::string & name) {
    return "op #" + std::to_string(i) + " (tag " + std::to_string(tag) + ", " + name + ")";
}

// One dict of a program: the keys it may have, which of them it must have; reads are checked against the label.
class Fields {
public:
    Fields(const py::dict & d, std::string label, std::initializer_list<const char *> required, std::initializer_list<const char *> optional)
        : d_(d), label_(std::move(label)) {
        for (const char * k : required) allowed_.push_back(k);
        for (const char * k : optional) allowed_.push_back(k);
        for (auto item : d) {
            if (!py::isinstance<py::str>(item.first)) bad_prog(label_ + ": a key that is not a string");
            const std::string key = item.first.cast<std::string>();
            if (std::find(allowed_.begin(), allowed_.end(), key) == allowed_.end()) bad_prog(label_ + ": unknown key '" + key + "'");
        }
        for (const char * k : required) {
            if (!d.contains(k)) bad_prog(label_ + ": missing key '" + k + "'");
        }
    }
    bool has(const char * k) const { return d_.contains(k) && !d_[k].is_none(); }
    int64_t integer(const char * k) const {
        const py::object o = d_[k];
        if (py::isinstance<py::bool_>(o) || !py::isinstance<py::int_>(o)) bad_prog(label_ + ": '" + k + "' must be an int");
        try {
            return o.cast<int64_t>();
        } catch (const py::cast_error &) {
            bad_prog(label_ + ": '" + k + "' is out of range");
        }
    }
    uint32_t u32(const char * k) const {
        const int64_t v = integer(k);
        if (v < 0 || v > (int64_t) UINT32_MAX) bad_prog(label_ + ": '" + k + "' is out of range");
        return (uint32_t) v;
    }
    int32_t id(const char * k) const {
        const int64_t v = integer(k);
        if (v < 0 || v > INT32_MAX) bad_prog(label_ + ": '" + k + "' is not a value id");
        return (int32_t) v;
    }
    int32_t opt_id(const char * k) const { return has(k) ? id(k) : -1; }
    float real(const char * k) const {
        const py::object o = d_[k];
        if (py::isinstance<py::bool_>(o) || !(py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o))) {
            bad_prog(label_ + ": '" + k + "' must be a float");
        }
        const double v = o.cast<double>();
        const float f = (float) v;
        if (!std::isfinite(f) && !std::isinf(v)) bad_prog(label_ + ": '" + k + "' is not a number");
        return f;
    }
    bool boolean(const char * k) const {
        const py::object o = d_[k];
        if (!py::isinstance<py::bool_>(o)) bad_prog(label_ + ": '" + k + "' must be a bool");
        return o.cast<bool>();
    }
    std::string str(const char * k) const {
        const py::object o = d_[k];
        if (!py::isinstance<py::str>(o)) bad_prog(label_ + ": '" + k + "' must be a string");
        return o.cast<std::string>();
    }

private:
    const py::dict & d_;
    std::string label_;
    std::vector<std::string> allowed_;
};

rt::graph_desc parse_program(const py::object & prog) {
    if (!py::isinstance<py::dict>(prog)) bad_prog("must be a dict with 'values', 'ops' and 'kv_size'");
    const py::dict top = py::reinterpret_borrow<py::dict>(prog);
    Fields tf(top, "program", { "values", "ops", "kv_size" }, {});
    rt::graph_desc g;
    g.kv_size = tf.u32("kv_size");
    if (!py::isinstance<py::list>(top["values"]) || !py::isinstance<py::list>(top["ops"])) bad_prog("'values' and 'ops' must be lists");
    const py::list values = py::reinterpret_borrow<py::list>(top["values"]);
    const py::list ops = py::reinterpret_borrow<py::list>(top["ops"]);
    size_t vi = 0;
    for (const py::handle & hv : values) {
        const std::string label = "value #" + std::to_string(vi++);
        if (!py::isinstance<py::dict>(hv)) bad_prog(label + ": must be a dict");
        const py::dict d = py::reinterpret_borrow<py::dict>(hv);
        if (!d.contains("kind")) bad_prog(label + ": missing key 'kind'");
        if (!py::isinstance<py::str>(d["kind"])) bad_prog(label + ": 'kind' must be a string");
        const std::string kind = d["kind"].cast<std::string>();
        rt::graph_value v;
        if (kind == "in") {
            Fields f(d, label + " (in)", { "kind", "n" }, {});
            v.kind = rt::gv::in;
            v.n = f.u32("n");
        } else if (kind == "tmp") {
            Fields f(d, label + " (tmp)", { "kind", "n" }, { "out" });
            v.kind = rt::gv::tmp;
            v.n = f.u32("n");
            v.out = f.has("out") && f.boolean("out");
        } else if (kind == "const") {
            Fields f(d, label + " (const)", { "kind", "n", "handle" }, { "rows" });
            v.kind = rt::gv::konst;
            v.n = f.u32("n");
            v.rows = f.has("rows") ? f.u32("rows") : 1u;
            v.handle = (uint64_t) f.integer("handle");
        } else if (kind == "pos") {
            Fields f(d, label + " (pos)", { "kind" }, {});
            v.kind = rt::gv::pos;
        } else if (kind == "mask") {
            Fields f(d, label + " (mask)", { "kind" }, {});
            v.kind = rt::gv::mask;
        } else {
            bad_prog(label + ": unknown kind '" + kind + "' (in, tmp, const, pos, mask)");
        }
        g.values.push_back(v);
    }
    size_t oi = 0;
    for (const py::handle & ho : ops) {
        const size_t i = oi++;
        if (!py::isinstance<py::dict>(ho)) bad_prog("op #" + std::to_string(i) + ": must be a dict");
        const py::dict d = py::reinterpret_borrow<py::dict>(ho);
        if (!d.contains("op")) bad_prog("op #" + std::to_string(i) + ": missing key 'op'");
        if (!py::isinstance<py::str>(d["op"])) bad_prog("op #" + std::to_string(i) + ": 'op' must be a string");
        const std::string name = d["op"].cast<std::string>();
        int tag = -1;
        if (d.contains("tag") && !d["tag"].is_none()) {
            if (!py::isinstance<py::int_>(d["tag"])) bad_prog("op #" + std::to_string(i) + " (" + name + "): 'tag' must be an int");
            long long t = -2;
            try {
                t = d["tag"].cast<long long>();
            } catch (const py::cast_error &) {
            }
            if (t < -1 || t > 0x7fffffffLL) bad_prog("op #" + std::to_string(i) + " (" + name + "): 'tag' out of range");
            tag = (int) t;
        }
        const std::string label = op_label(i, tag, name);
        rt::graph_op o;
        o.tag = tag;
        if (name == "mul_mat") {
            Fields f(d, label, { "op", "w", "x", "dst" }, { "addend", "tag" });
            o.op = rt::gop::mul_mat;
            o.w = (uint64_t) f.integer("w");
            o.x = f.id("x"); o.dst = f.id("dst"); o.addend = f.opt_id("addend");
        } else if (name == "add" || name == "mul") {
            Fields f(d, label, { "op", "a", "b", "dst" }, { "tag" });
            o.op = name == "add" ? rt::gop::add : rt::gop::mul;
            o.a = f.id("a"); o.b = f.id("b"); o.dst = f.id("dst");
        } else if (name == "scale") {
            Fields f(d, label, { "op", "x", "dst", "s", "b" }, { "tag" });
            o.op = rt::gop::scale;
            o.x = f.id("x"); o.dst = f.id("dst"); o.s = f.real("s"); o.sb = f.real("b");
        } else if (name == "rms_norm") {
            Fields f(d, label, { "op", "x", "dst", "eps" }, { "weight", "tag" });
            o.op = rt::gop::rms_norm;
            o.x = f.id("x"); o.dst = f.id("dst"); o.eps = f.real("eps"); o.weight = f.opt_id("weight");
        } else if (name == "glu") {
            Fields f(d, label, { "op", "gate", "up", "dst", "variant" }, { "tag" });
            o.op = rt::gop::glu;
            o.gate = f.id("gate"); o.up = f.id("up"); o.dst = f.id("dst"); o.variant = f.u32("variant");
        } else if (name == "unary") {
            Fields f(d, label, { "op", "x", "dst", "func" }, { "tag" });
            o.op = rt::gop::unary;
            o.x = f.id("x"); o.dst = f.id("dst"); o.func = f.u32("func");
        } else if (name == "rope") {
            Fields f(d, label, { "op", "x", "dst", "n_heads", "head_dim", "n_dims", "table", "pos", "factor" }, { "tag" });
            o.op = rt::gop::rope;
            o.x = f.id("x"); o.dst = f.id("dst"); o.n_heads = f.u32("n_heads"); o.head_dim = f.u32("head_dim"); o.n_dims = f.u32("n_dims");
            o.table = f.id("table"); o.pos = f.id("pos"); o.factor = f.real("factor");
        } else if (name == "attn") {
            Fields f(d, label, { "op", "q", "k", "v", "dst", "n_head", "n_head_kv", "d", "scale" }, { "mask", "tag" });
            o.op = rt::gop::attn;
            o.q = f.id("q"); o.k = f.id("k"); o.v = f.id("v"); o.dst = f.id("dst"); o.n_head = f.u32("n_head");
            o.n_head_kv = f.u32("n_head_kv"); o.d = f.u32("d"); o.scale = f.real("scale"); o.mask = f.opt_id("mask");
        } else if (name == "cpy") {
            Fields f(d, label, { "op", "x", "dst" }, { "tag" });
            o.op = rt::gop::cpy;
            o.x = f.id("x"); o.dst = f.id("dst");
        } else {
            bad_prog(label + ": unknown op (mul_mat, add, mul, scale, rms_norm, glu, unary, rope, attn, cpy)");
        }
        g.ops.push_back(o);
    }
    return g;
}

// The tensors of one graph_run, converted and kept alive for the call.
struct run_args {
    uint32_t T = 0;
    std::vector<at::Tensor> keep;
    rt::graph_inputs in;
    std::vector<at::Tensor> out;
    rt::graph_outputs go;
};

std::atomic<uint64_t> g_serial{0};

// Py wrapper around one tsr::rt::device.
//
// Lifetime: a method uses the rt::device only through a lease, which holds a reference to it and
// counts as a call in flight, also while the GIL is released. close() detaches the device at once
// (later calls raise "the device is closed"), waits until the calls in flight have returned, and then
// closes the transport. So a close from another thread never destroys the device under a running
// call, and the transport is closed when close() returns. A lease is never held while Python code
// can run (it is taken after the tensors are prepared), so close() cannot wait on its own thread.
class Device {
public:
    Device(std::shared_ptr<rt::device> d, std::string kind, uint64_t io_bytes, bool read_perf)
        : d_(std::move(d)), kind_(std::move(kind)), transport_(d_->transport()), description_(d_->description()),
          io_bytes_(io_bytes), read_perf_(read_perf), serial_(++g_serial) {}

    // Open the transport. Does not touch the GIL: callers release it around this.
    static std::shared_ptr<Device> open(const std::optional<std::string> & kind, uint64_t io_bytes, bool read_perf) {
        std::string k = kind ? *kind : std::string();
        rt::open_options o;
        if (!k.empty()) o.kind = k.c_str();
        o.io_bytes = io_bytes;
        o.read_perf = read_perf;
        auto d = std::make_shared<rt::device>(o);
        return std::make_shared<Device>(std::move(d), std::move(k), io_bytes, read_perf);
    }

    class lease {
    public:
        explicit lease(const Device & o) : o_(o) {
            std::lock_guard<std::mutex> lk(o.m_);
            if (!o.d_) throw rt::error("tesseract: the device is closed");
            d_ = o.d_;
            ++o.inflight_;
        }
        ~lease() {
            d_.reset();   // close() holds its own reference while it waits: never the last one here
            std::lock_guard<std::mutex> lk(o_.m_);
            if (--o_.inflight_ == 0) o_.cv_.notify_all();
        }
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        rt::device * operator->() const { return d_.get(); }

    private:
        const Device & o_;
        std::shared_ptr<rt::device> d_;
    };

    // Called without the GIL (the bindings release it).
    void close_nogil() {
        std::shared_ptr<rt::device> d;
        {
            std::unique_lock<std::mutex> lk(m_);
            d.swap(d_);
            cv_.wait(lk, [this] { return inflight_ == 0; });
        }
        d.reset();   // the last reference: every weight is released and the transport closed here
    }
    bool closed() const {
        std::lock_guard<std::mutex> lk(m_);
        return !d_;
    }

    const std::string & transport() const { return transport_; }
    const std::string & description() const { return description_; }
    const std::string & requested_kind() const { return kind_; }
    uint64_t io_bytes() const { return io_bytes_; }
    bool read_perf() const { return read_perf_; }
    uint64_t serial() const { return serial_; }

    py::dict config() const {
        tsr_config c;
        bool local_scale;
        {
            lease l(*this);
            c = l->config();
            local_scale = l->has_local_scale();
        }
        py::dict r;
        r["num_ch"] = c.num_ch;
        r["beat_bits"] = c.beat_bits;
        r["lanes"] = c.lanes;
        r["t_max"] = c.t_max;
        r["k_max"] = c.k_max;
        r["spm_bytes"] = c.spm_bytes;
        r["vl"] = c.vl;
        r["ch_bytes"] = c.ch_bytes;
        r["caps"] = c.caps;
        r["local_scale"] = local_scale;
        return r;
    }

    py::object unsupported_reason(const py::object & t, int64_t N, int64_t K, int64_t max_part_rows, int64_t max_part_k) const {
        rt::weight_options o;
        o.max_part_rows = to_u32(max_part_rows, "max_part_rows");
        o.max_part_k = to_u32(max_part_k, "max_part_k");
        const rt::dtype dt = dtype_of_obj(t);
        if (N < 1 || N > (int64_t) UINT32_MAX) return py::str("N must be in [1, 2^32)");
        if (K < 1 || K > (int64_t) UINT32_MAX) return py::str("K must be in [1, 2^32)");
        const char * why;
        {
            lease l(*this);
            why = l->unsupported_reason(dt, (uint32_t) N, (uint32_t) K, o);
        }
        if (why == nullptr) return py::none();
        return py::str(why);
    }

    // W [N, K] float32 / float16 / bfloat16 -> Q8_0 on the device. Rows may have any stride.
    uint64_t upload(const at::Tensor & w, int64_t max_part_rows, int64_t max_part_k) {
        check_cpu(w, "weight");
        if (w.dim() != 2) throw std::invalid_argument("tesseract: weight must be 2D [N, K]");
        const rt::dtype dt = dtype_of_scalar(w.scalar_type());
        const at::Tensor wc = rows_contiguous(w);
        rt::weight_options o;
        o.max_part_rows = to_u32(max_part_rows, "max_part_rows");
        o.max_part_k = to_u32(max_part_k, "max_part_k");
        const uint32_t N = to_u32(wc.size(0), "N");
        const uint32_t K = to_u32(wc.size(1), "K");
        const size_t row_stride = N > 1 ? (size_t) wc.stride(0) * wc.element_size() : 0;
        const void * data = wc.data_ptr();
        lease l(*this);
        py::gil_scoped_release nogil;
        return l->upload(dt, data, N, K, row_stride, o);
    }

    // ggml blocks (q8_0: K/32 x 34 B per row; q4_k: K/256 x 144 B per row) stored as they are.
    uint64_t upload_blocks(const std::string & type, const at::Tensor & blocks, int64_t N_, int64_t K_,
                           int64_t max_part_rows, int64_t max_part_k) {
        check_cpu(blocks, "blocks");
        const rt::dtype dt = dtype_of_name(type);
        if (dt != rt::dtype::q8_0 && dt != rt::dtype::q4_k) {
            throw std::invalid_argument("tesseract: upload_blocks takes q8_0 or q4_k; use upload() for float weights");
        }
        if (blocks.scalar_type() != at::kByte) throw std::invalid_argument("tesseract: blocks must be a uint8 tensor");
        const uint32_t N = to_u32(N_, "N"), K = to_u32(K_, "K");
        const size_t rb = rt::dtype_row_bytes(dt, K);
        if (rb == 0) throw std::invalid_argument("tesseract: K is not a multiple of the block size (32 q8_0, 256 q4_k)");
        const at::Tensor bc = blocks.contiguous();
        if ((size_t) bc.numel() != rb * N) {
            throw std::invalid_argument("tesseract: blocks has " + std::to_string(bc.numel()) + " bytes, expected N * row_bytes = " +
                                        std::to_string(rb * N));
        }
        rt::weight_options o;
        o.max_part_rows = to_u32(max_part_rows, "max_part_rows");
        o.max_part_k = to_u32(max_part_k, "max_part_k");
        const void * data = bc.data_ptr();
        lease l(*this);
        py::gil_scoped_release nogil;
        return l->upload(dt, data, N, K, 0, o);
    }

    void free_weight(uint64_t h) {
        lease l(*this);
        py::gil_scoped_release nogil;
        l->free_weight(h);
    }

    rt::weight_info info_of(uint64_t h) const {
        lease l(*this);
        return l->info(h);
    }

    py::dict info(uint64_t h) const {
        const rt::weight_info i = info_of(h);
        py::dict r;
        r["N"] = i.N;
        r["K"] = i.K;
        r["src"] = rt::dtype_name(i.src);
        r["fmt"] = fmt_name(i.fmt);
        r["n_parts"] = i.n_parts;
        r["n_part"] = i.n_part;
        r["k_parts"] = i.k_parts;
        r["k_part"] = i.k_part;
        r["dev_bytes"] = i.dev_bytes;
        return r;
    }

    // rows [r0, r1) as stored, ggml blocks of the device format: uint8 [r1 - r0, row_bytes]
    at::Tensor read_blocks(uint64_t h, int64_t r0, int64_t r1) const {
        const rt::weight_info i = info_of(h);
        if (r1 < 0) r1 = i.N;
        if (r0 < 0 || r0 > r1 || r1 > (int64_t) i.N) throw std::invalid_argument("tesseract: read_blocks row range outside the weight");
        const size_t rb = rt::dtype_row_bytes(i.fmt == TSR_FMT_Q4_K ? rt::dtype::q4_k : rt::dtype::q8_0, i.K);
        at::Tensor out = torch::empty({r1 - r0, (int64_t) rb}, torch::dtype(torch::kUInt8));
        void * dst = out.data_ptr();
        lease l(*this);
        py::gil_scoped_release nogil;
        l->read_blocks(h, (uint32_t) r0, (uint32_t) r1, dst);
        return out;
    }

    at::Tensor linear(uint64_t h, const at::Tensor & x, const std::optional<at::Tensor> & out, bool ref,
                      std::optional<float> scale = std::nullopt) {
        const rt::weight_info wi = info_of(h);
        check_cpu(x, "x");
        if (x.dim() != 2) throw std::invalid_argument("tesseract: x must be 2D [T, K]");
        if (x.scalar_type() != at::kFloat) throw std::invalid_argument("tesseract: x must be float32");
        if (x.size(1) != (int64_t) wi.K) {
            throw std::invalid_argument("tesseract: x has " + std::to_string(x.size(1)) + " columns, the weight K = " + std::to_string(wi.K));
        }
        const at::Tensor xc = rows_contiguous(x);
        const int64_t T = xc.size(0);
        if (T > (int64_t) UINT32_MAX) throw std::invalid_argument("tesseract: too many tokens");
        at::Tensor y;
        if (out.has_value()) {
            y = *out;
            check_cpu(y, "out");
            if (y.scalar_type() != at::kFloat || y.dim() != 2 || y.size(0) != T || y.size(1) != (int64_t) wi.N ||
                y.stride(1) != 1 || (T > 1 && y.stride(0) < (int64_t) wi.N)) {
                throw std::invalid_argument("tesseract: out must be float32 [T, N] with contiguous rows");
            }
        } else {
            y = torch::empty({T, (int64_t) wi.N}, torch::dtype(torch::kFloat));
        }
        if (T == 0) return y;
        const size_t xs = T > 1 ? (size_t) xc.stride(0) : 0;
        const size_t ys = T > 1 ? (size_t) y.stride(0) : 0;
        const float * xp = xc.data_ptr<float>();
        float * yp = y.data_ptr<float>();
        {
            lease l(*this);
            py::gil_scoped_release nogil;
            if (ref) {
                l->linear_ref(h, xp, (uint32_t) T, yp, xs, ys);
            } else if (scale) {
                if (!l->try_linear_scale(h, xp, (uint32_t) T, yp, *scale, xs, ys))
                    return {};
            } else {
                l->linear(h, xp, (uint32_t) T, yp, xs, ys);
            }
        }
        return y;
    }


    // ---- graphs ----

    // f32 [n] or [rows, n] -> constant rows in the AUX region of the device; returns a const handle
    uint64_t const_upload(const at::Tensor & t) {
        check_cpu(t, "const");
        if (t.scalar_type() != at::kFloat) throw std::invalid_argument("tesseract: a const must be a float32 tensor");
        if (t.dim() != 1 && t.dim() != 2) throw std::invalid_argument("tesseract: a const must be 1D [n] or 2D [rows, n]");
        const at::Tensor c = t.dim() == 1 ? t.contiguous().unsqueeze(0) : rows_contiguous(t);
        const uint32_t rows = to_u32(c.size(0), "rows"), n = to_u32(c.size(1), "n");
        if (rows == 0 || n == 0) throw std::invalid_argument("tesseract: a const must not be empty");
        const size_t stride = rows > 1 ? (size_t) c.stride(0) : 0;
        const float * p = c.data_ptr<float>();
        lease l(*this);
        py::gil_scoped_release nogil;
        return l->const_upload(p, rows, n, stride);
    }

    void const_free(uint64_t h) {
        lease l(*this);
        py::gil_scoped_release nogil;
        l->const_free(h);
    }

    // None, or why the device cannot run the program (ValueError if the program is malformed)
    py::object graph_unsupported(const py::object & prog) {
        const rt::graph_desc g = parse_program(prog);
        std::string why;
        bool has = false;
        {
            lease l(*this);
            py::gil_scoped_release nogil;
            const char * r = l->graph_unsupported(g);
            if (r != nullptr) {
                has = true;
                why = r;
            }
        }
        if (!has) return py::none();
        return py::str(why);
    }

    uint64_t graph_create(const py::object & prog) {
        const rt::graph_desc g = parse_program(prog);
        lease l(*this);
        py::gil_scoped_release nogil;
        return l->graph_create(g);
    }

    void graph_free(uint64_t h) {
        lease l(*this);
        py::gil_scoped_release nogil;
        l->graph_free(h);
    }

    py::dict graph_shape(uint64_t h) const {
        rt::graph_shape s;
        {
            lease l(*this);
            s = l->graph_shape_of(h);
        }
        py::list args;
        for (size_t i = 0; i < s.arg_kind.size(); ++i) {
            py::dict a;
            a["kind"] = s.arg_kind[i] == rt::gv::in ? "in" : (s.arg_kind[i] == rt::gv::pos ? "pos" : "mask");
            a["n"] = s.arg_n[i];
            args.append(a);
        }
        py::list outs;
        for (uint32_t n : s.out_n) outs.append(n);
        py::dict r;
        r["args"] = args;
        r["outs"] = outs;
        r["kv_size"] = s.kv_size;
        r["kv_bytes"] = s.kv_bytes;
        return r;
    }

    // The tensors of `inputs` (the in, pos and mask values in value order) as the graph takes them; T = rows of the in tensors.
    run_args graph_args(uint64_t h, const py::list & inputs) const {
        rt::graph_shape s;
        {
            lease l(*this);
            s = l->graph_shape_of(h);
        }
        if (inputs.size() != s.arg_kind.size()) {
            throw std::invalid_argument("tesseract: graph_run: " + std::to_string(inputs.size()) + " inputs given, the graph takes " +
                                        std::to_string(s.arg_kind.size()) + " (its in, pos and mask values in value order)");
        }
        run_args a;
        std::vector<at::Tensor> ts;
        for (size_t i = 0; i < inputs.size(); ++i) ts.push_back(py::cast<at::Tensor>(inputs[i]));
        int64_t T = -1;
        auto agree = [&](int64_t t, const char * what) {
            if (T >= 0 && t != T) {
                throw std::invalid_argument(std::string("tesseract: graph_run: ") + what + " has " + std::to_string(t) + " rows, the other inputs " + std::to_string(T));
            }
            T = t;
        };
        for (size_t i = 0; i < ts.size(); ++i) {
            const std::string name = "input " + std::to_string(i);
            check_cpu(ts[i], name.c_str());
            if (s.arg_kind[i] == rt::gv::in) {
                if (ts[i].dim() != 2 || ts[i].scalar_type() != at::kFloat || ts[i].size(1) != (int64_t) s.arg_n[i]) {
                    throw std::invalid_argument("tesseract: graph_run: " + name + " must be float32 [T, " + std::to_string(s.arg_n[i]) + "]");
                }
                agree(ts[i].size(0), name.c_str());
            }
        }
        if (T < 0) {
            for (size_t i = 0; i < ts.size() && T < 0; ++i) {
                if (s.arg_kind[i] == rt::gv::pos) T = ts[i].numel();
                else if (s.arg_kind[i] == rt::gv::mask && ts[i].dim() >= 2) T = ts[i].size(-1);
            }
        }
        if (T < 1 || T > (int64_t) UINT32_MAX) throw std::invalid_argument("tesseract: graph_run: T must be >= 1");
        a.T = (uint32_t) T;
        for (size_t i = 0; i < ts.size(); ++i) {
            const std::string name = "input " + std::to_string(i);
            rt::graph_arg ga;
            if (s.arg_kind[i] == rt::gv::in) {
                const at::Tensor c = rows_contiguous(ts[i]);
                a.keep.push_back(c);
                ga.data = c.data_ptr<float>();
                ga.stride = T > 1 ? (size_t) c.stride(0) : 0;
            } else if (s.arg_kind[i] == rt::gv::pos) {
                if (ts[i].numel() != T) {
                    throw std::invalid_argument("tesseract: graph_run: " + name + " (positions) must have " + std::to_string(T) + " elements");
                }
                if (ts[i].scalar_type() != at::kInt && ts[i].scalar_type() != at::kLong) {
                    throw std::invalid_argument("tesseract: graph_run: " + name + " (positions) must be int32 or int64");
                }
                const at::Tensor p = ts[i].reshape({-1}).contiguous();
                if (p.scalar_type() == at::kLong) {
                    const int64_t lo = p.min().item<int64_t>(), hi = p.max().item<int64_t>();
                    if (lo < INT32_MIN || hi > INT32_MAX) throw std::invalid_argument("tesseract: graph_run: " + name + " (positions) do not fit int32");
                }
                const at::Tensor c = p.to(at::kInt).contiguous();
                a.keep.push_back(c);
                ga.data = c.data_ptr<int32_t>();
            } else {
                // [.., T, T] with leading dimensions of 1; float32 additive (0 / -inf) or bool (True: visible)
                const at::Tensor & m = ts[i];
                bool shape_ok = m.dim() >= 2 && m.size(-1) == T && m.size(-2) == T;
                for (int64_t d = 0; shape_ok && d < m.dim() - 2; ++d) shape_ok = m.size(d) == 1;
                if (!shape_ok) throw std::invalid_argument("tesseract: graph_run: " + name + " (mask) must be [T, T] (leading dimensions of 1 allowed)");
                at::Tensor f;
                if (m.scalar_type() == at::kBool) {
                    f = torch::zeros({T, T}, torch::dtype(torch::kFloat)).masked_fill(~m.reshape({T, T}), -INFINITY);
                } else if (m.scalar_type() == at::kFloat) {
                    f = m.reshape({T, T}).contiguous();
                } else {
                    throw std::invalid_argument("tesseract: graph_run: " + name + " (mask) must be float32 (0 / -inf) or bool (True: visible)");
                }
                a.keep.push_back(f);
                ga.data = f.data_ptr<float>();
                ga.stride = 0;
            }
            a.in.args.push_back(ga);
        }
        for (uint32_t n : s.out_n) {
            a.out.push_back(torch::empty({T, (int64_t) n}, torch::dtype(torch::kFloat)));
            a.go.outs.push_back({ a.out.back().data_ptr<float>(), 0 });
        }
        return a;
    }

    // The out values of a run: float32 [T, n] tensors in value order
    py::list graph_run(uint64_t h, const py::list & inputs, bool ref) {
        run_args a = graph_args(h, inputs);
        {
            lease l(*this);
            py::gil_scoped_release nogil;
            if (ref) l->graph_run_ref(h, a.T, a.in, a.go);
            else l->graph_run(h, a.T, a.in, a.go);
        }
        py::list r;
        for (const at::Tensor & t : a.out) r.append(t);
        return r;
    }

    py::dict graph_info(uint64_t h, int64_t T) {
        const uint32_t t = to_u32(T, "T");
        rt::graph_plan_info i;
        {
            lease l(*this);
            py::gil_scoped_release nogil;
            i = l->graph_info(h, t);
        }
        py::dict r;
        r["T"] = i.T;
        r["lists"] = i.lists;
        r["records"] = i.records;
        r["joins"] = i.joins;
        r["mul_mat_begin"] = i.mul_mat_begin;
        r["io_bytes"] = i.io_bytes;
        r["preload_lines"] = i.preload_lines;
        r["fa_new"] = i.fa_new;
        py::dict kinds;
        for (uint32_t k = 0; k < 64; ++k) {
            if (i.by_kind[k] != 0) kinds[rt::record_kind_name(k)] = i.by_kind[k];
        }
        r["by_kind"] = kinds;
        return r;
    }

    py::dict counters() const {
        rt::counters c;
        {
            lease l(*this);
            c = l->get_counters();
        }
        py::dict r;
        r["linears"] = c.linears;
        r["tokens"] = c.tokens;
        r["execs"] = c.execs;
        r["prog_hits"] = c.prog_hits;
        r["prog_misses"] = c.prog_misses;
        r["perf_cycles"] = c.perf_cycles;
        r["perf_mvm_cycles"] = c.perf_mvm_cycles;
        r["dev_read_bytes"] = c.dev_read_bytes;
        r["h2d_bytes"] = c.h2d_bytes;
        r["h2d_weight_bytes"] = c.h2d_weight_bytes;
        r["d2h_bytes"] = c.d2h_bytes;
        r["graphs"] = c.graphs;
        r["graph_runs"] = c.graph_runs;
        r["graph_launches"] = c.graph_launches;
        r["graph_records"] = c.graph_records;
        r["graph_plan_hits"] = c.graph_plan_hits;
        r["graph_plan_misses"] = c.graph_plan_misses;
        r["consts"] = c.consts;
        return r;
    }

    void reset_counters() {
        lease l(*this);
        l->reset_counters();
    }

    py::dict memory() const {
        rt::memory_info m;
        {
            lease l(*this);
            m = l->memory();
        }
        py::dict r;
        r["weight_total"] = m.weight_total;
        r["weight_free"] = m.weight_free;
        r["weight_largest_free"] = m.weight_largest_free;
        r["n_weights"] = m.n_weights;
        r["prog_cached"] = m.prog_cached;
        r["prog_cmd_bytes"] = m.prog_cmd_bytes;
        return r;
    }

private:
    mutable std::mutex m_;               // guards d_ and inflight_
    mutable std::condition_variable cv_; // inflight_ dropped to 0
    mutable int inflight_ = 0;           // leases alive
    std::shared_ptr<rt::device> d_;      // null once closed
    std::string kind_;
    std::string transport_;
    std::string description_;
    uint64_t io_bytes_ = 0;
    bool read_perf_ = true;
    uint64_t serial_ = 0;
};

// Process-wide singleton (the torch backend's device). g_mtx is only taken with the GIL released
// (the module functions use py::call_guard<py::gil_scoped_release>): opening a transport can take a
// while, and a thread waiting for g_mtx must not hold the GIL meanwhile.
std::mutex g_mtx;
std::shared_ptr<Device> g_dev;

// The open device, or a new one. An option that is given must match the open device's.
std::shared_ptr<Device> singleton(const std::optional<std::string> & kind, std::optional<uint64_t> io_bytes,
                                  std::optional<bool> read_perf) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_dev && !g_dev->closed()) {
        const bool same = (!kind || kind->empty() || *kind == g_dev->requested_kind() || *kind == g_dev->transport()) &&
                          (!io_bytes || *io_bytes == g_dev->io_bytes()) && (!read_perf || *read_perf == g_dev->read_perf());
        if (!same) {
            throw rt::error("tesseract: the device is already open (" + g_dev->description() +
                            ") with other options; close() it first");
        }
        return g_dev;
    }
    g_dev = Device::open(kind, io_bytes.value_or(0), read_perf.value_or(true));
    return g_dev;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Tesseract NPU runtime (tsr_rt) for PyTorch: weight upload and y = x W^T on the device";
    py::register_exception<rt::error>(m, "TesseractError", PyExc_RuntimeError);

    py::class_<Device, std::shared_ptr<Device>>(m, "Device")
        .def(py::init([](const std::optional<std::string> & kind, uint64_t io_bytes, bool read_perf) {
                 py::gil_scoped_release nogil;
                 return Device::open(kind, io_bytes, read_perf);
             }), py::arg("kind") = py::none(), py::arg("io_bytes") = 0, py::arg("read_perf") = true,
             "Open a device of its own (kind: sim | mock | vkernel | xrt; None: TSR_DEVICE, default sim). "
             "The torch backend uses the singleton (open / get) instead.")
        .def("close", &Device::close_nogil, py::call_guard<py::gil_scoped_release>(),
             "Release every weight and close the transport, after the calls in flight on other threads have "
             "returned; later calls raise TesseractError")
        .def_property_readonly("closed", &Device::closed)
        .def_property_readonly("serial", &Device::serial,
                               "Number of this device in the process (1, 2, ...): weight handles are per device")
        .def_property_readonly("transport", &Device::transport)
        .def_property_readonly("description", &Device::description)
        .def_property_readonly("io_bytes", &Device::io_bytes)
        .def_property_readonly("read_perf", &Device::read_perf)
        .def_property_readonly("config", &Device::config)
        .def("unsupported_reason", &Device::unsupported_reason, py::arg("dtype"), py::arg("N"), py::arg("K"),
             py::arg("max_part_rows") = 0, py::arg("max_part_k") = 0,
             "None if a weight of this dtype (torch.dtype or 'f32'/'f16'/'bf16'/'q8_0'/'q4_k') and shape [N, K] can be "
             "uploaded, else the reason")
        .def("upload", &Device::upload, py::arg("weight"), py::arg("max_part_rows") = 0, py::arg("max_part_k") = 0,
             "Upload W [N, K] (float32 / float16 / bfloat16, CPU, any row stride) as Q8_0 "
             "(quantize_row_q8_0_ref); returns a weight handle (int, never 0)")
        .def("upload_blocks", &Device::upload_blocks, py::arg("type"), py::arg("blocks"), py::arg("N"), py::arg("K"),
             py::arg("max_part_rows") = 0, py::arg("max_part_k") = 0,
             "Upload ggml blocks ('q8_0' or 'q4_k', uint8 tensor of N * row_bytes) as they are; returns a handle")
        .def("free", &Device::free_weight, py::arg("handle"))
        .def("info", &Device::info, py::arg("handle"))
        .def("read_blocks", &Device::read_blocks, py::arg("handle"), py::arg("r0") = 0, py::arg("r1") = -1,
             "Rows [r0, r1) as stored on the device, ggml blocks of the device format: uint8 [rows, row_bytes]")
        .def("linear", [](Device & d, uint64_t h, const at::Tensor & x, const std::optional<at::Tensor> & out) {
                 return d.linear(h, x, out, false);
             }, py::arg("handle"), py::arg("x"), py::arg("out") = py::none(),
             "y [T, N] float32 = x [T, K] float32 @ W^T on the device")
        .def("_try_linear_scale", [](Device & d, uint64_t h, const at::Tensor & x, float scale) -> py::object {
            const auto y = d.linear(h, x, std::nullopt, false, scale);
            if (!y.defined()) return py::none();
            return py::cast(y);
        }, py::arg("h"), py::arg("x"), py::arg("scale"))
        .def("linear_ref", [](Device & d, uint64_t h, const at::Tensor & x, const std::optional<at::Tensor> & out) {
                 return d.linear(h, x, out, true);
             }, py::arg("handle"), py::arg("x"), py::arg("out") = py::none(),
             "The same y on the host with libtsr_ref from the weight rows read back from the device (bit-identical "
             "to linear() on a correct device)")
        .def("const_upload", &Device::const_upload, py::arg("tensor"),
             "Upload a float32 [n] or [rows, n] tensor as constant rows (norm weight, bias row, RoPE table) in the AUX "
             "region; returns a const handle (int, never 0)")
        .def("const_free", &Device::const_free, py::arg("handle"))
        .def("graph_unsupported", &Device::graph_unsupported, py::arg("program"),
             "None if the device can run the program (a dict of values, ops, kv_size), else the reason (it names the op's "
             "tag and kind); ValueError if the program is malformed. A const with handle 0 and a mul_mat with w = 0 are "
             "checked by shape only")
        .def("graph_create", &Device::graph_create, py::arg("program"),
             "Validate the program as graph_unsupported does and allocate its KV planes; returns a graph handle. "
             "ValueError: malformed; TesseractError: the device cannot run it, or the planes do not fit")
        .def("graph_free", &Device::graph_free, py::arg("handle"))
        .def("graph_shape", &Device::graph_shape, py::arg("handle"),
             "{'args': [{'kind': 'in' | 'pos' | 'mask', 'n': int}], 'outs': [n], 'kv_size', 'kv_bytes'}")
        .def("graph_run", [](Device & d, uint64_t h, const py::list & inputs) { return d.graph_run(h, inputs, false); },
             py::arg("handle"), py::arg("inputs"),
             "Run the graph: inputs = the tensors of the in, pos and mask values in value order (in: float32 [T, n]; pos: "
             "int32 / int64 [T]; mask: float32 0 / -inf or bool True = visible, [T, T]); T is the row count of the in "
             "tensors. Returns the out values as float32 [T, n] tensors, in value order. TesseractError whose text "
             "contains 'does not fit': this T cannot be placed")
        .def("graph_run_ref", [](Device & d, uint64_t h, const py::list & inputs) { return d.graph_run(h, inputs, true); },
             py::arg("handle"), py::arg("inputs"),
             "The same result on the host: the plan's op lists executed with the per-kind references on an image of the "
             "device's memory (bit-identical to graph_run on a correct device)")
        .def("graph_info", &Device::graph_info, py::arg("handle"), py::arg("T"),
             "The plan of T (made if needed): lists, records, joins, mul_mat_begin, by_kind {name: count}, io_bytes, "
             "preload_lines, fa_new")
        .def("counters", &Device::counters)
        .def("reset_counters", &Device::reset_counters)
        .def("memory", &Device::memory);

    m.def("open", &singleton, py::arg("kind") = py::none(), py::arg("io_bytes") = py::none(),
          py::arg("read_perf") = py::none(), py::call_guard<py::gil_scoped_release>(),
          "The process-wide device: the open one (each option given must match it, else TesseractError), or a "
          "new one with these options (None: TSR_DEVICE, io_bytes 0, read_perf True)");
    m.def("get", []() { return singleton(std::nullopt, std::nullopt, std::nullopt); },
          py::call_guard<py::gil_scoped_release>(),
          "The process-wide device as it is open, or a new one with the defaults (TSR_DEVICE)");
    m.def("is_open", []() {
        std::lock_guard<std::mutex> lk(g_mtx);
        return g_dev && !g_dev->closed();
    }, py::call_guard<py::gil_scoped_release>());
    m.def("close", []() {
        std::shared_ptr<Device> d;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            d.swap(g_dev);
        }
        if (d) d->close_nogil();
    }, py::call_guard<py::gil_scoped_release>(),
       "Close the process-wide device after its calls in flight (every weight handle becomes invalid)");
    m.def("row_bytes", [](const std::string & type, int64_t K) {
        return rt::dtype_row_bytes(dtype_of_name(type), to_u32(K, "K"));
    }, py::arg("type"), py::arg("K"), "Bytes of one packed row of K elements ('f32', 'f16', 'bf16', 'q8_0', 'q4_k'); 0 if K is not a block multiple");
}

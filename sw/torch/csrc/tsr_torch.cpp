// tsr_torch.cpp — Python module _tesseract_rt: torch bindings of the Tesseract runtime (sw/rt/tsr_rt.h).
//
// Binds tsr::rt::device as _tesseract_rt.Device (weight upload, linear, reference linear, counters)
// plus a process-wide device singleton (open / get / is_open / close). Tensors cross the boundary as
// torch.Tensor (CPU); the device calls run with the GIL released. Numerics are tsr_rt's: f32 / f16 /
// bf16 weights become Q8_0 at upload with ggml's quantize_row_q8_0_ref, activations are quantized
// by QACT on the device, results are f32. The Python package tesseract_torch (sw/torch) builds the
// torch.compile backend on top of this module.
//
// Errors: tsr::rt::error -> _tesseract_rt.TesseractError (a RuntimeError); std::invalid_argument ->
// ValueError (pybind11's translation).
#include <torch/extension.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

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

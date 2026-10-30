// tsr-opl-dev.cpp — see tsr-opl.h: the `graph` kernel on a transport (boot, launch, wait) and the
// text of a failed launch.
#include "tsr-opl.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace opl = tsr::opl;

static uint64_t tsr_opl_now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool tsr_opl_dev::open(tsr_transport * tp, std::string * err) {
    char buf[512];
    const tsr_image * img = tsr_transport_image(tp);
    if (img == nullptr) {
        tsr_image * own = nullptr;
        const int rc = tsr_image_open_default(&own);
        if (rc != 0 || own == nullptr) {
            const char * e = getenv("TSR_IMAGE");
            snprintf(buf, sizeof(buf), "cannot open the firmware image %s (%s)",
                     e && *e ? e : "(the build's tsr-fw.elf; TSR_IMAGE overrides it)", tsr_transport_strerror(rc));
            *err = buf;
            return false;
        }
        const int rb = tsr_transport_boot(tp, own);
        if (rb != 0) {
            snprintf(buf, sizeof(buf), "booting %s on %s failed (%s)", tsr_image_path(own), tsr_transport_name(tp),
                     tsr_transport_strerror(rb));
            *err = buf;
            tsr_image_close(own);
            return false;
        }
        tsr_image_close(own);      // images are reference counted: the transport keeps the booted one
        img = tsr_transport_image(tp);
        if (img == nullptr) {
            *err = "the transport reports no booted image after a successful boot";
            return false;
        }
    }
    k_graph_ = tsr_image_kernel(img, "graph");
    if (k_graph_ < 0) {
        snprintf(buf, sizeof(buf), "the firmware image %s has no `graph` kernel", tsr_image_path(img));
        *err = buf;
        return false;
    }
    image_id_ = tsr_image_id(img);
    image_path_ = tsr_image_path(img) ? tsr_image_path(img) : "";
    cache_.init(0, TSR_REGION_OPL_OFF, TSR_REGION_OPL_SIZE);
    tp_ = tp;
    return true;
}

tsr_opl_run_result tsr_opl_dev::run(const std::vector<uint8_t> & blob, uint32_t launch_flags, const tsr_dram_write_fn & write) {
    tsr_opl_run_result r;
    if (!cache_.place(blob, write, &r.opl_off, &r.cache_hit)) {
        r.rc = TSR_TE_INVAL;
        r.too_big = true;
        r.res.ret = (int32_t) blob.size();
        return r;
    }
    tsr_launch l = {};
    l.type   = TSR_DESC_TYPE_LAUNCH;
    l.flags  = 0;
    l.kernel = (uint16_t) k_graph_;
    l.arg[0] = tsr_dram_handle(cache_.channel(), r.opl_off);
    l.arg[1] = (uint32_t) blob.size();
    l.arg[2] = launch_flags;
    tsr_record rec = {};
    const uint64_t t0 = tsr_opl_now_ns();
    // the transport's timeout (TSR_LAUNCH_TIMEOUT_MS, debug-aware, with DB's recovery ladder, SW §9.1)
    r.rc = tsr_transport_call(tp_, &l, TSR_TIMEOUT_DEFAULT, &rec);
    r.wall_ns = tsr_opl_now_ns() - t0;
    launches++;
    launch_ns += r.wall_ns;
    if (blob.size() >= opl::HDR_BYTES) {
        ops += opl::rd32(blob.data() + 4u * opl::hw::N_OPS);
    }
    if (r.rc == 0) {
        tsr_record_decode(&rec, &r.res);
    }
    return r;
}

const char * tsr_opl_ret_name(uint32_t ret) {
    switch (ret) {
        case opl::RET_BAD_MAGIC:    return "BAD_MAGIC";
        case opl::RET_BAD_VERSION:  return "BAD_VERSION";
        case opl::RET_TOO_MANY_OPS: return "TOO_MANY_OPS";
        case opl::RET_BAD_RECORD:   return "BAD_RECORD";
        case opl::RET_BAD_DEP:      return "BAD_DEP";
        case opl::RET_SCRATCH:      return "SCRATCH";
        case opl::RET_CRC:          return "CRC";
        default:                    return "?";
    }
}

const char * tsr_err_code_name(uint32_t code) {
    switch (code) {
        case TSR_ERR_NONE:        return "none";
        case TSR_ERR_ILLEGAL_OP:  return "ILLEGAL_OP";
        case TSR_ERR_BAD_FMT:     return "BAD_FMT";
        case TSR_ERR_QA_MISMATCH: return "QA_MISMATCH";
        case TSR_ERR_SPM_RANGE:   return "SPM_RANGE";
        case TSR_ERR_DRAM_RANGE:  return "DRAM_RANGE";
        case TSR_ERR_K_RANGE:     return "K_RANGE";
        case TSR_ERR_T_RANGE:     return "T_RANGE";
        case TSR_ERR_ABORTED:     return "ABORTED";
        case TSR_ERR_DEP_FAILED:  return "DEP_FAILED";
        case TSR_ERR_BAD_DEP:     return "BAD_DEP";
        case TSR_ERR_OPERAND:     return "OPERAND";
        default:                  return "?";
    }
}

std::string tsr_opl_describe(const tsr_opl_run_result & r, const char * op_name) {
    char buf[768];
    if (r.rc != 0) {
        if (r.too_big) {
            snprintf(buf, sizeof(buf), "the op list of %d bytes does not fit the OPL region", (int) r.res.ret);
        } else {
            snprintf(buf, sizeof(buf), "graph launch failed in the transport: %s (rc %d; the transport's diagnosis is above)",
                     tsr_transport_strerror(r.rc), r.rc);
        }
        return buf;
    }
    const tsr_result & s = r.res;
    const uint32_t ret = (uint32_t) s.ret;
    if (s.type != TSR_LCQ_R0_TYPE_DONE) {
        snprintf(buf, sizeof(buf), "graph launch answered by a record of type 0x%02x (status %u %s, r2 0x%08x)",
                 s.type, s.status, tsr_status_name(s.status), ret);
    } else if (s.status == TSR_LCQ_R0_STATUS_TILE_ERR) {
        snprintf(buf, sizeof(buf), "TILE_ERR at op %u (%s): error %u %s, ERR_INFO 0x%08x (ticket 0x%08x, w0 0x%08x, push %u)",
                 ret - 1, op_name ? op_name : "?", s.err_code, tsr_err_code_name(s.err_code), s.err_info, s.err_ticket,
                 s.err_w0, s.err_pidx);
    } else if (s.status == TSR_LCQ_R0_STATUS_KERNEL_ERR) {
        snprintf(buf, sizeof(buf), "KERNEL_ERR: the graph executor refused the op list: RET 0x%08x %s", ret, tsr_opl_ret_name(ret));
    } else {
        snprintf(buf, sizeof(buf), "DONE status %u %s, RET 0x%08x, error %u %s, ERR_INFO 0x%08x", s.status,
                 tsr_status_name(s.status), ret, s.err_code, tsr_err_code_name(s.err_code), s.err_info);
    }
    return buf;
}

// tsr_transport.c — transport selection (TSR_DEVICE=sim|mock|vkernel|xrt|<registered>, default
// sim) and the record helpers of SW §9.1 (docs/design/06-software-verification.md; EX §4.4). Every
// kind is the protocol layer over a kdev (tsr_kdev.cpp, SW-18); the sim kind is the sim kdev.

#include "tsr_transport.h"
#include "tsr_transport_impl.h"

#include <stdlib.h>
#include <string.h>

tsr_transport * tsr_transport_open(const char * kind) {
    if (kind == NULL || kind[0] == '\0') {
        kind = getenv("TSR_DEVICE");
        if (kind == NULL || kind[0] == '\0') kind = "sim";
    }
    return tsr_transport_kdev_open(kind);
}

void tsr_record_decode(const tsr_record * rec, tsr_result * out) {
    const uint32_t * r = rec->w;
    memset(out, 0, sizeof(*out));
    out->type = (uint8_t)TSR_GET(r[0], TSR_LCQ_R0_TYPE);
    out->status = (uint8_t)TSR_GET(r[0], TSR_LCQ_R0_STATUS);
    out->err_code = (uint8_t)TSR_GET(r[0], TSR_LCQ_R0_ERR_CODE);
    out->seq = r[TSR_LCQ_DONE_SEQ_WORD];
    out->ret = (int32_t)r[TSR_LCQ_DONE_RET_WORD];
    out->err_ticket = r[TSR_LCQ_DONE_ERR_TICKET_WORD];
    out->err_info = r[TSR_LCQ_DONE_ERR_INFO_WORD];
    out->err_w0 = r[TSR_LCQ_DONE_ERR_W0_WORD];
    out->err_pidx = r[TSR_LCQ_DONE_ERR_PIDX_WORD];
    out->duration = r[TSR_LCQ_DONE_DURATION_WORD];
}

const char * tsr_status_name(uint8_t status) {
    switch (status) {
    case TSR_LCQ_R0_STATUS_OK:         return "OK";
    case TSR_LCQ_R0_STATUS_TILE_ERR:   return "TILE_ERR";
    case TSR_LCQ_R0_STATUS_BAD_DESC:   return "BAD_DESC";
    case TSR_LCQ_R0_STATUS_KERNEL_ERR: return "KERNEL_ERR";
    case TSR_LCQ_R0_STATUS_TRAP:       return "TRAP";
    case TSR_LCQ_R0_STATUS_ABORTED:    return "ABORTED";
    case TSR_LCQ_R0_STATUS_PLAT_ERR:   return "PLAT_ERR";
    case TSR_LCQ_R0_STATUS_STOPPED:    return "STOPPED";
    default:                           return "RESERVED";
    }
}

const char * tsr_transport_strerror(int rc) {
    switch (rc) {
    case 0:              return "ok";
    case TSR_TE_INVAL:   return "invalid argument";
    case TSR_TE_IO:      return "device access failed";
    case TSR_TE_TIMEOUT: return "timeout (device diagnosed and recovered)";
    case TSR_TE_LOST:    return "launch lost in a device reset";
    case TSR_TE_DEVICE:  return "device failure";
    case TSR_TE_IMAGE:   return "image refused";
    case TSR_TE_STATE:   return "wrong device state";
    default:             return "unknown error";
    }
}

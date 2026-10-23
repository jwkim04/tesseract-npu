// tsr_transport.c — transport selection and dispatch (TSR_DEVICE=sim|xrt, default sim).

#include "tsr_transport.h"
#include "tsr_transport_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tsr_transport {
    const tsr_transport_ops * ops;
    void *                    impl;
};

tsr_transport * tsr_transport_open(const char * kind) {
    if (kind == NULL || kind[0] == '\0') {
        kind = getenv("TSR_DEVICE");
        if (kind == NULL || kind[0] == '\0') kind = "sim";
    }
    const tsr_transport_ops * ops = NULL;
    void * impl = NULL;
    if (strcmp(kind, "sim") == 0) {
        impl = tsr_transport_sim_open(&ops);
    } else if (strcmp(kind, "xrt") == 0) {
#ifdef TSR_HAVE_XRT
        impl = tsr_transport_xrt_open(&ops);
#else
        fprintf(stderr, "tsr_transport: the \"xrt\" transport is not built in (configure with -DTSR_WITH_XRT=ON)\n");
        return NULL;
#endif
    } else {
        fprintf(stderr, "tsr_transport: unknown transport \"%s\" (TSR_DEVICE must be sim or xrt)\n", kind);
        return NULL;
    }
    if (impl == NULL) return NULL;
    tsr_transport * t = (tsr_transport *)calloc(1, sizeof(tsr_transport));
    if (t == NULL) {
        ops->close(impl);
        fprintf(stderr, "tsr_transport: out of memory\n");
        return NULL;
    }
    t->ops = ops;
    t->impl = impl;
    return t;
}

void tsr_transport_close(tsr_transport * t) {
    if (t == NULL) return;
    t->ops->close(t->impl);
    free(t);
}

const char * tsr_transport_name(const tsr_transport * t) { return t->ops->name; }

const tsr_config * tsr_transport_config(const tsr_transport * t) { return t->ops->config(t->impl); }

int tsr_transport_write(tsr_transport * t, uint32_t ch, uint64_t off, const void * src, size_t size) {
    return t->ops->write(t->impl, ch, off, src, size);
}

int tsr_transport_read(tsr_transport * t, uint32_t ch, uint64_t off, void * dst, size_t size) {
    return t->ops->read(t->impl, ch, off, dst, size);
}

int tsr_transport_exec(tsr_transport * t, uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8],
                       tsr_exec_result * res) {
    tsr_exec_result local;
    return t->ops->exec(t->impl, cmd_addr_beat, cmd_count, args, res ? res : &local);
}

uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off) { return t->ops->csr_read(t->impl, off); }

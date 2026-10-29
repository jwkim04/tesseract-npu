// tsr_transport.c — transport selection and dispatch (TSR_DEVICE=sim|xrt|mock|vkernel|<registered>,
// default sim).

#include "tsr_transport.h"
#include "tsr_transport_impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tsr_transport {
    const tsr_transport_ops * ops;
    void *                    impl;
};

tsr_transport * tsr_transport_wrap(const tsr_transport_ops * ops, void * impl) {
    if (ops == NULL || impl == NULL) return NULL;
    tsr_transport * t = (tsr_transport *)calloc(1, sizeof(tsr_transport));
    if (t == NULL) {
        fprintf(stderr, "tsr_transport: out of memory\n");
        return NULL;
    }
    t->ops = ops;
    t->impl = impl;
    return t;
}

const tsr_transport_ops * tsr_transport_get_ops(const tsr_transport * t) { return t ? t->ops : NULL; }
void * tsr_transport_get_impl(const tsr_transport * t) { return t ? t->impl : NULL; }

tsr_transport * tsr_transport_open(const char * kind) {
    if (kind == NULL || kind[0] == '\0') {
        kind = getenv("TSR_DEVICE");
        if (kind == NULL || kind[0] == '\0') kind = "sim";
    }
    const tsr_transport_ops * ops = NULL;
    void * impl = NULL;
    if (strcmp(kind, "sim") == 0) {
        impl = tsr_transport_sim_open(&ops);
    } else {
        impl = tsr_transport_kdev_open(kind, &ops);   // xrt, mock, vkernel, registered kdev kinds
    }
    if (impl == NULL) return NULL;
    tsr_transport * t = tsr_transport_wrap(ops, impl);
    if (t == NULL) ops->close(impl);
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

uint32_t tsr_transport_csr_read(tsr_transport * t, uint32_t off) { return t->ops->csr_read(t->impl, off); }

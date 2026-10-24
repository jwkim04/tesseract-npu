// tsr_transport_impl.h — private: the per-implementation operation table behind tsr_transport.
#pragma once

#include "tsr_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsr_transport_ops {
    const char * name;
    void               (*close)(void * impl);
    const tsr_config * (*config)(void * impl);
    int                (*write)(void * impl, uint32_t ch, uint64_t off, const void * src, size_t size);
    int                (*read)(void * impl, uint32_t ch, uint64_t off, void * dst, size_t size);
    int                (*exec)(void * impl, uint64_t cmd_addr_beat, uint32_t cmd_count, const uint32_t args[8],
                               tsr_exec_result * res);
    uint32_t           (*csr_read)(void * impl, uint32_t off);
} tsr_transport_ops;

// Each returns an implementation handle and its ops, or NULL (message on stderr).
void * tsr_transport_sim_open(const tsr_transport_ops ** ops);
// Every other kind ("xrt", "mock", kinds registered with tsr::register_kdev): the kernel-protocol
// transport over a kdev (tsr_kdev.cpp).
void * tsr_transport_kdev_open(const char * kind, const tsr_transport_ops ** ops);

// Wrap an already opened implementation (tsr::transport_open_kdev); NULL on failure (impl untouched).
tsr_transport *           tsr_transport_wrap(const tsr_transport_ops * ops, void * impl);
const tsr_transport_ops * tsr_transport_get_ops(const tsr_transport * t);
void *                    tsr_transport_get_impl(const tsr_transport * t);

#ifdef __cplusplus
}
#endif

// tsr_transport_impl.h — private: the entry from the C dispatcher (tsr_transport.c) into the
// protocol layer (tsr_kdev.cpp), which implements every kind (V2-SW-18, EX §10.2).
#pragma once

#include "tsr_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opens kind ("sim", "mock", "vkernel", "xrt" or a kind registered with tsr::register_kdev):
// the kdev, then the protocol layer's open. NULL on failure (message on stderr).
tsr_transport * tsr_transport_kdev_open(const char * kind);

#ifdef __cplusplus
}
#endif

// embed_auto.cpp — libtsr_dbg: links the embedded debug service into a program (docs/design/05-debug.md
// "DB" DB-17, §9.5). An object of the OBJECT library tsr_dbg_embed: a program that links it gets the
// service registered before main (TSR_GDB / TSR_DEBUGGER then start it at every tsr_transport_open);
// a program that links only tsr_dbg calls tsr_dbg_embed_install() itself.
#include "tsr_dbg_lib.hpp"

namespace {
struct auto_install {
    auto_install() { tsr::dbg::embed_install(); }
} g_tsr_dbg_auto_install;
} // namespace

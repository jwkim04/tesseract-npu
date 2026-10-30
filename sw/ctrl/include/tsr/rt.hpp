// tsr/rt.hpp — firmware SDK runtime interface (docs/design/06-software-verification.md "SW" §6.2-§6.5;
// docs/design/01-execution-model.md "EX" §3.1, §4.3-§4.7; docs/design/05-debug.md "DB" §3.8, §6.3).
//
// What a kernel and the tools need from the runtime:
//   - struct tsr_args, the kernel ABI (SW §6.5), and struct tsr_kernel_desc, the 36-byte entry of
//     the kernel table .tsr_kernels (SW §6.2);
//   - TSR_KERNEL(name, nargs, flags, blk_lines): declares the kernel and emits its table entry;
//   - the running launch (tsr::launch(): descriptor, sequence, first push index, base sequences),
//     RT_INFO (tsr::set_info), kernel events 0xC0-0xFF (tsr::post_event), the soft-stop poll
//     (tsr::rt_stop_requested, which kernel.hpp's stop_requested() calls), the SPM argument range.
// Logging and panic are in tsr/log.hpp.
//
// The kind table .tsr_kinds (SW §6.2, §6.9): TSR_KIND(NAME) beside each op list kind's handler emits
// one 4-byte entry {kind, revision, 0}; the runtime folds the table into HELLO r4[31:16] (the kind
// table signature, tsr::opl::kinds_sig) and the host reads it from the ELF before the boot.
//
// Kernel index rule (EX §3.1, §4.6). The kernel index of a descriptor is the position in
// .tsr_kernels. TSR_KERNEL puts each entry in the input section ".tsr_kernels.<rank>.<name>" and the
// linker script collects them with KEEP(*(SORT_BY_NAME(.tsr_kernels.*))) between __tsr_kernels_start
// and __tsr_kernels_end, so the table is sorted by (rank, name):
//   rank 0 graph, 1 selftest, 2 replay   EX §4.6's base kernels, in SW §7.1's order: in the
//                                        production image (which always has all three) they are
//                                        kernels 0, 1 and 2
//   rank 5 every other kernel            after them, by name (test images: tsr_fw_image extras)
// Tools find a kernel by name (tsr-elf's table dump, the test drivers), never by a hard-coded index
// beyond the three base kernels. Names are at most 23 characters (char name[24], NUL-padded).
//
// Link-time sections and symbols the runtime relies on (sw/ctrl/rt/link.ld, SW §6.2): .vectors at
// RV_BOOT with _start at +0x80 (crt0.S), .tsr_kernels (DTCM, loaded: __tsr_kernels_start / _end),
// .tsr_kinds (DTCM, loaded: __tsr_kinds_start / _end),
// .tsr_id (DTCM, loaded, 8 bytes: __tsr_image_id, written by tsr-elf), .tsr_dbg (DTCM, loaded,
// 64 B: __tsr_dbg), .bss and .tsr_crash (DTCM, not loaded: crt0 zeroes [__zero_start, __zero_end),
// which holds __tsr_crash), .init_array (__init_array_start / _end), the 512 B trap stack
// (__trap_stack_top, mscratch) below the 8 KiB stack (__stack_top = the top of DTCM),
// __global_pointer$, and .tsr_logfmt (VMA 0xF000_0000, not allocated; tsr/log.hpp).
//
// Host build (TSR_FW_HOST, SW §6.8): TSR_KERNEL only declares the kernel and a constexpr
// tsr_kinfo_<name> (nargs, flags, blk_lines, name) for the host shim; tsr::launch() and the event
// sink are host variables the shim fills / sets (below). There is no table and no runtime.
#ifndef TSR_RT_HPP
#define TSR_RT_HPP

#include <stddef.h>
#include <stdint.h>

#include "tsr/mmio.hpp"
#include "tsr/version.h"

// ---------------------------------------------------------------------------------------------
// Versions (SW §6.9): the Tesseract Basis version TSR_BASIS_VERSION and the Basis ABI TSR_SDK_ABI
// come from tsr/version.h only; the link writes both into .note.tsr (words 6 and 0, --defsym), the
// runtime posts them in HELLO (r6 from the link's __tsr_basis_version, r4[15:0]).
// ---------------------------------------------------------------------------------------------
// Kernel table flags (SW §6.2).
#define TSR_KFLAG_ARGBLK 0x01u    // ARG0 is the DRAM handle of an argument block (<= 1 KiB)

// The runtime's SPM argument range: the top 16 lines of the SPM (SW §6.3 R5, §7.3).
#define TSR_ARG_LINES 16u

// ---------------------------------------------------------------------------------------------
// Kernel ABI (SW §6.5)
// ---------------------------------------------------------------------------------------------
struct tsr_args {
    uint32_t a[7];                   // descriptor ARG0..ARG6 (EX §4.3); args beyond the kernel's
                                     // nargs read 0 (the runtime zeroes them)
    const volatile uint8_t * blk;    // SPM-window pointer to the argument block, or nullptr
    uint32_t seq;                    // launch sequence number (LQ_HEAD_SEQ)
    uint32_t flags;                  // bit0 PROF (EX §4.3 w0[8]): emit NODE markers
};
#define TSR_ARGS_FLAG_PROF 0x01u

typedef int32_t (*tsr_kernel_fn)(const tsr_args *);

// Kernel table entry (SW §6.2), 36 bytes, little-endian, in .tsr_kernels sorted by index.
struct tsr_kernel_desc {
    uint32_t name_hash;      // FNV-1a 32 of name (tools only)
#ifdef TSR_FW_HOST
    uint32_t entry;          // the kernel's RV address (host tools read the table from the ELF)
#else
    tsr_kernel_fn entry;     // int32_t (*)(const tsr_args *): 32 bits on RV32
#endif
    uint8_t  nargs;          // 0..7 (descriptor ARG0..ARG6)
    uint8_t  flags;          // bit0 ARGBLK
    uint16_t argblk_lines;   // lines loaded into the SPM argument range when ARGBLK (1..16), else 0
    char     name[24];       // NUL-padded
};
// SW §6.2 calls the entry 32 bytes, but its fields sum to 36: the fields win (link.ld's
// __tsr_kdesc_bytes = 36, tsr-elf), so names keep 23 characters.
static_assert(sizeof(tsr_kernel_desc) == 36, "SW §6.2 kernel table entry: 4 + 4 + 1 + 1 + 2 + 24 bytes");

// Kind table entry (SW §6.2, §6.9), 4 bytes, in .tsr_kinds (unique kinds, any order): an op list kind
// the image provides and its numerics revision (tsr/opl.hpp kind_rev). TSR_KIND emits it.
struct tsr_kind_desc {
    uint8_t  kind;           // 0x00..0x3F (tsr::opl::K_*)
    uint8_t  rev;            // 1..255
    uint16_t rsvd;           // 0
};
static_assert(sizeof(tsr_kind_desc) == 4, "SW §6.2 kind table entry: 1 + 1 + 2 bytes");

namespace tsr {

// FNV-1a 32 (name_hash, SW §6.2).
constexpr uint32_t fnv1a32(const char * s) {
    uint32_t h = 2166136261u;
    for (; *s; ++s) h = (h ^ uint8_t(*s)) * 16777619u;
    return h;
}

// Host view of a kernel's table fields (host build: tsr_kinfo_<name>).
struct kernel_info {
    uint8_t nargs, flags;
    uint16_t argblk_lines;
    const char * name;
};

// The running launch (EX §4.7 R2 / R5). Filled by the runtime before the kernel is called; valid
// during the kernel. DTCM state of the runtime, rebuilt at every launch (SW-4).
struct launch_info {
    uint32_t desc[8];               // the descriptor words copied at R2 (LQ_HEAD_W0..7)
    uint32_t seq;                   // its sequence number (LQ_HEAD_SEQ)
    uint32_t kernel;                // KERNEL (w0[31:16])
    uint32_t flags;                 // TSR_ARGS_FLAG_PROF
    uint32_t pidx0;                 // ISSUE_PIDX at R5: the launch's first push index (launch_pidx)
    uint32_t base[NUM_UNITS];       // ISSUED_SEQ[u] at R5 (base_u, SW §7.7)
    uint32_t stop_seen;             // rt_stop_requested() returned true (R7: STOPPED)
    uint32_t active;                // 1 between LQ_POP and the answer
};

// Runtime panic codes (tsr::panic code, TRAP r4 = mtval; DB §6.3). Kernels use their own codes.
constexpr uint32_t PANIC_RT_EVENT_TYPE  = 0x52540001u;  // post_event: type outside 0xC0..0xFF
constexpr uint32_t PANIC_RT_NOT_IDLE    = 0x52540002u;  // debug build: QSTATUS.QIDLE never rose at R10
constexpr uint32_t PANIC_RT_TABLE       = 0x52540003u;  // kernel table entry with an invalid field

// Log EVENT codes of the runtime (DB §6.1 EVENT payload word 0, marker encoding of DB §8.4 with
// class 15: SDK runtime events).
constexpr uint32_t LOG_EV_DROPPED = 0xF0000001u;        // "dropped n": arg = records dropped

#ifndef TSR_FW_HOST
extern "C" launch_info tsr_rt_launch;                   // runtime.cpp
inline const launch_info & launch() { return tsr_rt_launch; }

// RT_INFO (EX §4.5): the runtime- or kernel-defined progress word, mirrored to host 0x03C.
inline void set_info(uint32_t v) {
    *reinterpret_cast<volatile uint32_t *>(uintptr_t(RV_CTL_BASE + CW_RT_INFO)) = v;
}

// Kernel event (EX §4.4): r0 = type | info[15:0] << 8 | SEQ[7:0] << 24, r1 = SEQ of the running
// launch, r2..r7 = d[0..5]; posted with ANSWER = 0, after waiting (WFI on irq_fast[3]) while
// LCQ_FREE = 0. type must be 0xC0..0xFF (else tsr::panic(PANIC_RT_EVENT_TYPE)). Kernel code only,
// never from an interrupt handler. Its content must obey the timing rule (SW §4.7).
void post_event(uint32_t type, uint32_t info, const uint32_t d[6]);
inline void event(uint32_t type, uint32_t info = 0, uint32_t r2 = 0, uint32_t r3 = 0, uint32_t r4 = 0,
                  uint32_t r5 = 0, uint32_t r6 = 0, uint32_t r7 = 0) {
    const uint32_t d[6] = { r2, r3, r4, r5, r6, r7 };
    post_event(type, info, d);
}

// EX §4.9 soft stop: STOP_VALID and LQ_STOP covers the running launch ((s - q) mod 2^32 < 2^31).
// A true result is remembered (launch_info.stop_seen): R7 then answers STOPPED unless a status of
// higher precedence applies (EX §4.4). kernel.hpp's tsr::stop_requested() is this function.
bool rt_stop_requested();

// SPM size in lines (CFG1 read through the core window at boot) and the first line of the
// argument range [spm_lines() - 16, spm_lines()).
uint32_t spm_lines();
inline uint32_t argblk_line() { return spm_lines() - TSR_ARG_LINES; }

#else   // TSR_FW_HOST

// Host build (SW §6.8): the shim fills host_launch before it calls a kernel; events go to
// host_event_sink (nullptr: dropped). No table, no runtime steps.
inline launch_info host_launch{};
inline void (*host_event_sink)(uint32_t type, uint32_t info, const uint32_t d[6]) = nullptr;
inline uint32_t host_spm_lines = SPM_BYTES_U200 / 64u;
inline uint32_t host_rt_info = 0;
inline uint32_t host_stop = 0;           // the shim's soft-stop request for the running launch

inline const launch_info & launch() { return host_launch; }
inline void set_info(uint32_t v) { host_rt_info = v; }
inline void post_event(uint32_t type, uint32_t info, const uint32_t d[6]) {
    if (host_event_sink) host_event_sink(type, info, d);
}
inline void event(uint32_t type, uint32_t info = 0, uint32_t r2 = 0, uint32_t r3 = 0, uint32_t r4 = 0,
                  uint32_t r5 = 0, uint32_t r6 = 0, uint32_t r7 = 0) {
    const uint32_t d[6] = { r2, r3, r4, r5, r6, r7 };
    post_event(type, info, d);
}
inline bool rt_stop_requested() {
    if (host_stop) host_launch.stop_seen = 1;
    return host_stop != 0;
}
inline uint32_t spm_lines() { return host_spm_lines; }
inline uint32_t argblk_line() { return spm_lines() - TSR_ARG_LINES; }

#endif  // TSR_FW_HOST

}  // namespace tsr

// ---------------------------------------------------------------------------------------------
// TSR_KERNEL(name, nargs, flags, blk_lines) (SW §6.5)
// ---------------------------------------------------------------------------------------------
// Rank of a kernel in the table (see the index rule above): TSR_KRANK_<name> is defined as
// "~, <rank>" for the base kernels; any other name gets the default rank 5.
#define TSR_KRANK_graph    ~, "0"
#define TSR_KRANK_selftest ~, "1"
#define TSR_KRANK_replay   ~, "2"
#define TSR_KRANK_SECOND_(a, b, ...) b
#define TSR_KRANK_PICK_(...) TSR_KRANK_SECOND_(__VA_ARGS__, "5", ~)
#define TSR_KRANK_(name) TSR_KRANK_PICK_(TSR_KRANK_##name)

#define TSR_KERNEL_CHECK_(name, nargs, flags, blk_lines)                                            \
    static_assert((nargs) >= 0 && (nargs) <= 7, "TSR_KERNEL " #name ": nargs 0..7");                  \
    static_assert(((flags) & ~TSR_KFLAG_ARGBLK) == 0, "TSR_KERNEL " #name ": flags");                 \
    static_assert(((flags) & TSR_KFLAG_ARGBLK) ? ((blk_lines) >= 1 && (blk_lines) <= 16 && (nargs) >= 1) \
                                               : (blk_lines) == 0,                                    \
                  "TSR_KERNEL " #name ": ARGBLK needs blk_lines 1..16 and nargs >= 1, else 0");       \
    static_assert(sizeof(#name) <= 24, "TSR_KERNEL " #name ": name longer than 23 characters")

#ifndef TSR_FW_HOST
#define TSR_KERNEL(name, nargs, flags, blk_lines)                                                   \
    extern "C" int32_t name(const tsr_args * args);                                                 \
    TSR_KERNEL_CHECK_(name, nargs, flags, blk_lines);                                               \
    extern "C" __attribute__((used, aligned(4), section(".tsr_kernels." TSR_KRANK_(name) "." #name))) \
    const tsr_kernel_desc tsr_kdesc_##name = { ::tsr::fnv1a32(#name), &name, uint8_t(nargs),        \
                                               uint8_t(flags), uint16_t(blk_lines), #name }
#else
#define TSR_KERNEL(name, nargs, flags, blk_lines)                                                   \
    extern "C" int32_t name(const tsr_args * args);                                                 \
    TSR_KERNEL_CHECK_(name, nargs, flags, blk_lines);                                               \
    inline constexpr ::tsr::kernel_info tsr_kinfo_##name = { uint8_t(nargs), uint8_t(flags),        \
                                                             uint16_t(blk_lines), #name }
#endif

// ---------------------------------------------------------------------------------------------
// TSR_KIND(NAME) (SW §6.2, §6.9)
// ---------------------------------------------------------------------------------------------
// Declares that the image provides op list kind tsr::opl::K_<NAME> at its revision
// tsr::opl::kind_rev(K_<NAME>): one .tsr_kinds entry, written once beside the kind's handler (the
// source that defines it strongly; the executor's NOP and JOIN in graph.cpp). The includer has
// tsr/opl.hpp. A kind declared twice in one image is a link error (the symbol tsr_kind_<NAME>);
// fw_kind_revs checks that the production image declares exactly the kinds it handles. Host build
// (TSR_FW_HOST): a constexpr tsr_kinfo_kind_<NAME> only.
#ifndef TSR_FW_HOST
#define TSR_KIND(NAME)                                                                                  \
    static_assert(::tsr::opl::kind_rev(::tsr::opl::K_##NAME) >= 1u, "TSR_KIND " #NAME ": no revision"); \
    extern "C" __attribute__((used, aligned(4), section(".tsr_kinds")))                                 \
    const tsr_kind_desc tsr_kind_##NAME = {                                                             \
        uint8_t(::tsr::opl::K_##NAME), uint8_t(::tsr::opl::kind_rev(::tsr::opl::K_##NAME)), 0u }
#else
#define TSR_KIND(NAME)                                                                                  \
    static_assert(::tsr::opl::kind_rev(::tsr::opl::K_##NAME) >= 1u, "TSR_KIND " #NAME ": no revision"); \
    inline constexpr tsr_kind_desc tsr_kinfo_kind_##NAME = {                                            \
        uint8_t(::tsr::opl::K_##NAME), uint8_t(::tsr::opl::kind_rev(::tsr::opl::K_##NAME)), 0u }
#endif

#endif  // TSR_RT_HPP

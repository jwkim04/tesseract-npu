/* tsr/version.h — the Tesseract Basis version and the Basis ABI (docs/design/06-software-verification.md
 * "SW" §6.9, SW-54; docs/design/01-execution-model.md "EX" §4.4 HELLO r4 / r6).
 *
 * Plain C, also valid C++ and freestanding: the firmware (tsr/rt.hpp), tsr-elf, the transport and the
 * tools include it, and sw/ctrl/cmake/tsr_version.cmake reads the three version lines and TSR_SDK_ABI
 * with a regex (the link's --defsym __tsr_basis_version / __tsr_sdk_abi, so .note.tsr words 6 and 0
 * carry them). This file is the only source of both numbers. It is not in sw/tables: a software
 * release never regenerates the hardware tables.
 *
 *   Tesseract Basis MAJOR.MINOR.PATCH (SemVer; 0.x until the M7 card gate, then 1.0.0)
 *     MAJOR  from 1.0.0: an ABI break (below); in 0.x an incompatible change bumps MINOR and the ABI
 *     MINOR  new kinds, kernels, kind variants or flags, a planner accepting more shapes, any kind
 *            revision bump (tsr/opl.hpp kind_rev), new host API or tools
 *     PATCH  fixes and performance work that leave every output bit of every kind, every interface
 *            and every planner's acceptance unchanged
 *   Bumped at release with CHANGELOG.md ("(unreleased)" -> the date) and the annotated tag
 *   basis-vX.Y.Z. Encoding (TSR_BASIS_VERSION, .note.tsr word 6, HELLO r6, layout HELLO_R6):
 *   [23:16] MAJOR, [15:8] MINOR, [7:0] PATCH, [31:24] 0.
 *
 *   Basis ABI (TSR_SDK_ABI; .note.tsr word 0, HELLO r4[15:0]): tsr_args, TSR_KERNEL and the kernel
 *   table entry, the .note.tsr layout and the .tsr_kinds entry, the launch protocol as the runtime
 *   implements it (descriptor, record and HELLO layouts, R1-R10), the op list format (header, record
 *   layout, every kind's argument words and their meaning, removing a kind) and a planner accepting
 *   less (SW-20). Bumped by the breaking change itself. ABI 1 is defined as of Basis 0.1.0.
 */
#ifndef TSR_VERSION_H
#define TSR_VERSION_H

#define TSR_BASIS_VERSION_MAJOR 0
#define TSR_BASIS_VERSION_MINOR 1
#define TSR_BASIS_VERSION_PATCH 0

#define TSR_BASIS_VERSION_OF(major, minor, patch) \
    ((((major) & 0xFFu) << 16) | (((minor) & 0xFFu) << 8) | ((patch) & 0xFFu))
#define TSR_BASIS_VERSION \
    TSR_BASIS_VERSION_OF(TSR_BASIS_VERSION_MAJOR, TSR_BASIS_VERSION_MINOR, TSR_BASIS_VERSION_PATCH)
/* "MAJOR.MINOR.PATCH" as a string literal (the tests' expectations; the tools format the code they read). */
#define TSR_BASIS_STR_(x)  #x
#define TSR_BASIS_XSTR_(x) TSR_BASIS_STR_(x)
#define TSR_BASIS_VERSION_STRING                                                                                 \
    TSR_BASIS_XSTR_(TSR_BASIS_VERSION_MAJOR) "." TSR_BASIS_XSTR_(TSR_BASIS_VERSION_MINOR) "." TSR_BASIS_XSTR_( \
        TSR_BASIS_VERSION_PATCH)
#define TSR_BASIS_MAJOR_OF(v) (((v) >> 16) & 0xFFu)
#define TSR_BASIS_MINOR_OF(v) (((v) >> 8) & 0xFFu)
#define TSR_BASIS_PATCH_OF(v) ((v) & 0xFFu)

#define TSR_SDK_ABI 1u

#endif

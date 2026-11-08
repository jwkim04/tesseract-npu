// tsr_elf.h — Tesseract firmware images on the host (SW §6.2, §9.5; EX §3.1-§3.2; P2.1).
//
// A parsed RV32 firmware ELF: its PT_LOAD segments (what the loader writes, EX §3.3), the image id
// (EX §3.2), the kernel table (.tsr_kernels), the kind table (.tsr_kinds, SW §6.9), the image
// requirements and versions (.note.tsr), the log-format table (.tsr_logfmt, SW §6.6) and the size
// report against SW §7.8's budget. Used by the tool
// tsr-elf (stamping after the link, checks, dumps) and by host code that needs the same facts
// (decoders of FMT log records, loaders, tests). C API, implemented in C++17 (tsr_elf.cpp); the
// SHA-256 here is independent of sw/sim/tsr_trace.c's (fw_elf compares the two).
//
// Image id (EX §3.2): SHA-256 over every PT_LOAD segment in ascending physical address, each as
// (u32 paddr LE, u32 padded filesz LE, padded bytes) with the 8 bytes of .tsr_id taken as zero;
// the id is the first 8 digest bytes read as a little-endian uint64. "Padded" is p_filesz rounded
// up to a whole word with zeros, as the host loads it (EX §3.1).
#ifndef TSR_ELF_H
#define TSR_ELF_H

#include <stddef.h>
#include <stdint.h>

#include "tsr/version.h"   // TSR_SDK_ABI, TSR_BASIS_VERSION (SW §6.9)

#ifdef __cplusplus
extern "C" {
#endif

// Formats (SW §6.2, EX §3.1, SW §7.8)
#define TSR_ELF_KDESC_BYTES      36u           // kernel table entry: name_hash, entry, nargs, flags,
                                               // argblk_lines, name[24]
#define TSR_ELF_KNAME_BYTES      24u
#define TSR_ELF_KFLAG_ARGBLK     1u            // flags bit0: ARG0 is an argument-block handle
#define TSR_ELF_MAX_NARGS        7u            // descriptor ARG0..ARG6
#define TSR_ELF_ARGBLK_MAX_LINES 16u
#define TSR_ELF_NOTE_NAME        "TSR"
#define TSR_ELF_NOTE_TYPE        1u
#define TSR_ELF_NOTE_DESC_BYTES  32u
#define TSR_ELF_SDK_ABI          TSR_SDK_ABI   // the Basis ABI this tool checks images against
#define TSR_ELF_KIND_BYTES       4u            // kind table entry: kind, rev, rsvd (u16)
#define TSR_ELF_N_KINDS          64u           // op list kinds 0x00-0x3F (tsr/opl.hpp N_KINDS)
#define TSR_ELF_RESET_OFFSET     0x80u         // reset entry at RV_BOOT + 0x80 (RV_BOOT = 0)
#define TSR_ELF_BUDGET_ITCM      (96u * 1024u) // SW §7.8 image-size targets (ITCM 48 -> 80 KiB: SW-38, M7 kinds; -> 96 KiB: SW-75, MoE + Gated DeltaNet kinds)
#define TSR_ELF_BUDGET_DTCM      (56u * 1024u)

typedef struct tsr_elf tsr_elf;

// One PT_LOAD segment, as the loader writes it.
typedef struct tsr_elf_seg {
    uint32_t        paddr;
    uint32_t        filesz;     // p_filesz
    uint32_t        padded;     // p_filesz rounded up to a word (the words the host writes)
    uint32_t        memsz;      // p_memsz (the rest is .bss, zeroed by crt0)
    uint32_t        flags;      // p_flags (PF_X 1, PF_W 2, PF_R 4)
    const uint8_t * data;       // filesz bytes (inside the image buffer: .tsr_id holds the stamped id)
} tsr_elf_seg;

// One kernel table entry (SW §6.2), decoded.
typedef struct tsr_elf_kernel {
    uint32_t index;             // position in the table = descriptor KERNEL (EX §3.1)
    uint32_t name_hash;         // as stored
    uint32_t entry;
    uint8_t  nargs;
    uint8_t  flags;
    uint16_t argblk_lines;
    char     name[TSR_ELF_KNAME_BYTES + 1];   // NUL-terminated copy
} tsr_elf_kernel;

// .note.tsr descriptor (EX §3.1, SW §6.2, §6.9).
typedef struct tsr_elf_note {
    uint32_t sdk_abi;           // Basis ABI
    uint32_t cap_min;           // Tesseract Core capability the image needs: major << 8 | minor (VERSION[15:0])
    uint32_t itcm_min;          // bytes
    uint32_t dtcm_min;          // bytes
    uint32_t caps_req;          // required CAPS bits
    uint32_t n_kernels;
    uint32_t basis_version;     // Tesseract Basis version: major << 16 | minor << 8 | patch (0: none)
    uint32_t rsv;               // word 7, 0
} tsr_elf_note;

// Memory the checks assume: ITCM / DTCM sizes (RV_CFG; 0 = the U200 defaults of tsr_tables.h)
// and the SW §7.8 budget (0 = the defaults above).
typedef struct tsr_elf_limits {
    uint32_t itcm_bytes, dtcm_bytes;
    uint32_t budget_itcm, budget_dtcm;
} tsr_elf_limits;

// Size report. "used" counts every allocated section in the region (loaded or not), so DTCM
// includes .bss, the crash record, the stack and the trap stack (SW §7.8).
typedef struct tsr_elf_sizes {
    uint32_t itcm_bytes, dtcm_bytes, budget_itcm, budget_dtcm;
    uint32_t itcm_used;         // end of the last ITCM section
    uint32_t dtcm_used;         // DTCM extent from the base minus the largest hole (the free
                                // space below the stacks); alignment padding counts as used
    uint32_t dtcm_loaded;       // DTCM bytes the host writes (padded PT_LOAD file sizes)
    uint32_t dtcm_zeroed;       // DTCM sections not loaded (.bss, .tsr_crash, .stack)
    uint32_t logfmt;            // .tsr_logfmt bytes (in the file only)
    int      over_itcm, over_dtcm;   // budget exceeded
} tsr_elf_sizes;

// --- reading -------------------------------------------------------------------------------------
// Parses an image (a copy of buf is kept). Fails only on what makes the file unusable as an RV32
// image (not ELF32 little-endian EM_RISCV ET_EXEC, truncated headers or segments); the EX §3.1 /
// SW §6.2 rules are tsr_elf_check's. 0, or -1 with a message in err.
int  tsr_elf_parse(const void * buf, size_t n, tsr_elf ** out, char * err, size_t errlen);
int  tsr_elf_read(const char * path, tsr_elf ** out, char * err, size_t errlen);
void tsr_elf_free(tsr_elf * e);

const uint8_t *     tsr_elf_bytes(const tsr_elf * e, size_t * n);
uint32_t            tsr_elf_entry(const tsr_elf * e);
uint32_t            tsr_elf_nseg(const tsr_elf * e);
const tsr_elf_seg * tsr_elf_segment(const tsr_elf * e, uint32_t i);   // ascending paddr
// Value of a symbol of .symtab: 0, or -1 if absent.
int                 tsr_elf_symbol(const tsr_elf * e, const char * name, uint32_t * value);
// A section by name (the first of that name): address, file offset (0 for SHT_NOBITS), size and
// sh_type. 0, or -1 if absent (or the image has no section headers).
int                 tsr_elf_section(const tsr_elf * e, const char * name, uint32_t * addr, uint32_t * off,
                                    uint32_t * size, uint32_t * type);

// --- image id (EX §3.2) --------------------------------------------------------------------------
uint64_t tsr_elf_id(const tsr_elf * e);                         // computed from the segments
int      tsr_elf_stored_id(const tsr_elf * e, uint64_t * id);   // .tsr_id content; -1 without .tsr_id
int      tsr_elf_set_id(tsr_elf * e, uint64_t id);              // into the buffer; -1 without .tsr_id
// Writes the (possibly stamped) buffer to path atomically (temporary file + rename). 0 or -1.
int      tsr_elf_write(const tsr_elf * e, const char * path, char * err, size_t errlen);

// --- kernel table, note, log formats -------------------------------------------------------------
uint32_t               tsr_elf_n_kernels(const tsr_elf * e);    // whole entries in .tsr_kernels
const tsr_elf_kernel * tsr_elf_kernel_at(const tsr_elf * e, uint32_t index);
int                    tsr_elf_kernel_find(const tsr_elf * e, const char * name);   // index or -1
int                    tsr_elf_get_note(const tsr_elf * e, tsr_elf_note * note);    // 0, -1 if absent
// The kind table .tsr_kinds (SW §6.9): rev[k] = the image's revision of kind k, 0 if absent (no
// section: no kinds). Returns the number of kinds, or -1 for a malformed table (size not a multiple
// of 4, a kind >= 0x40, a kind twice, revision 0, rsvd != 0; why says which, NUL-terminated).
int                    tsr_elf_kinds(const tsr_elf * e, uint8_t rev[TSR_ELF_N_KINDS], char * why, size_t whylen);
// The format string of an FMT log record (DB §6.1): the NUL-terminated string at addr inside
// .tsr_logfmt (addresses inside a string are valid: the linker may merge string tails). NULL if
// addr is outside the section or the string is unterminated.
const char * tsr_elf_logfmt(const tsr_elf * e, uint32_t addr);
// The log-format table: the strings of .tsr_logfmt in address order (split at NULs).
uint32_t     tsr_elf_logfmt_count(const tsr_elf * e);
const char * tsr_elf_logfmt_at(const tsr_elf * e, uint32_t i, uint32_t * addr);
// A %s argument: the NUL-terminated string at addr in a loaded section (SW §6.6). NULL if none.
const char * tsr_elf_string(const tsr_elf * e, uint32_t addr);

// --- checks and sizes ----------------------------------------------------------------------------
// Checks EX §3.1 and SW §6.2: segments inside the TCMs, word aligned, not overlapping, vaddr =
// paddr; e_entry and _start at RV_BOOT + 0x80; .tsr_id (8 bytes of file data in a DTCM PT_LOAD);
// the kernel table (whole entries in a loaded DTCM section; names C identifiers of 1..23
// characters, NUL-padded, unique, also by hash; name_hash = FNV-1a 32 of the name; nargs <= 7;
// flags bit0 only; argblk_lines 1..16 with ARGBLK and nargs >= 1, 0 without; entry in an executable
// ITCM segment and equal to the symbol of the same name; sorted by index, i.e. by (rank, name) with
// rank 0 graph, 1 selftest, 2 replay, 5 every other kernel: tsr/rt.hpp's TSR_KERNEL rule);
// the kind table (tsr_elf_kinds's rules; a non-empty table in the file data of a DTCM PT_LOAD);
// .note.tsr (exactly one note "TSR" type 1 with a 32-byte descriptor: sdk_abi = TSR_SDK_ABI, cap_min
// a capability 0x0001..0xFFFF, itcm_min / dtcm_min covering the image and within the TCMs, n_kernels
// = table entries, basis_version a version (nonzero, [31:24] 0), word 7 0); .tsr_logfmt at
// TSR_LOGFMT_VMA, not allocated, its last string terminated.
// Flags: TSR_ELF_CHECK_ID also checks that the stored id equals the computed one (a stamped,
// untampered image); TSR_ELF_CHECK_BASE that EX §4.6's base kernels graph, selftest and replay
// are kernels 0, 1 and 2 (the production image).
// Every problem is one line "error: [tag] text" (budget overruns: "warning: [tag] text") appended
// to report (NUL-terminated, truncated at replen; may be NULL). Returns the number of errors.
#define TSR_ELF_CHECK_ID   1u
#define TSR_ELF_CHECK_BASE 2u
int  tsr_elf_check(const tsr_elf * e, const tsr_elf_limits * lim, unsigned flags, char * report, size_t replen);
// Rank of a kernel name in the table order (0 graph, 1 selftest, 2 replay, 5 other).
uint32_t tsr_elf_kernel_rank(const char * name);
void tsr_elf_get_sizes(const tsr_elf * e, const tsr_elf_limits * lim, tsr_elf_sizes * s);
// The size report as text (sections per region, totals, budget), NUL-terminated.
void tsr_elf_size_report(const tsr_elf * e, const tsr_elf_limits * lim, char * out, size_t outlen);

// --- helpers -------------------------------------------------------------------------------------
uint32_t tsr_elf_fnv1a(const char * s);                         // FNV-1a 32 (kernel name hash)
void     tsr_elf_sha256(const void * data, size_t n, uint8_t out[32]);

#ifdef __cplusplus
}
#endif

#endif

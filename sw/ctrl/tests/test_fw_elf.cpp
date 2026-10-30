// test_fw_elf.cpp — the fw_elf ctests (P2.1; SW §6.2, §9.5, §10.2; EX §3.1-§3.4): tsr-elf and the
// firmware image format.
//
//   test_fw_elf unit <elf dir> <tsr-elf> [<other.elf> ...]
//       SHA-256 and FNV-1a known answers; image-id golden values of synthesized ELFs (computed
//       independently with Python's hashlib) and the same ids from sw/sim/tsr_trace.c's own
//       implementation; the toolchain-built test images of sw/ctrl/tests/elf/elf_img.S linked with
//       the real link.ld (kernel order, note values, log formats, sizes, .tsr_logfmt outside every
//       PT_LOAD, the simulator's ELF reader seeing the same segments); stamping; tamper detection;
//       one patched image per validation rule; the tsr-elf command line. Extra ELFs (e.g. sw/sim's
//       rv_issue firmware) get the two-implementation id comparison.
//   test_fw_elf dump <dir>
//       writes the synthesized ELFs (to recompute the golden ids with another implementation).
//   test_fw_elf image <image.elf> [--base] [--has K | --only K] [--kinds all|K1,K2,...] [--tool T]
//       a built firmware image: every tsr-elf check including the stored id (and, with --base, EX
//       §4.6's base kernels at 0, 1, 2 and no static constructor), SW §7.8's budget as a failure,
//       the id against tsr_trace.c's and the segments against the simulator's ELF reader; --has /
//       --only: the probe kernel K of tests/elf/k_probe.cpp is in the table (after the base kernels /
//       alone); --kinds: .tsr_kinds holds exactly these kinds (all: every kind with a revision) at
//       tsr/opl.hpp kind_rev (SW §6.9); --tool: `tsr-elf kinds` and `note` print the same table,
//       signature and versions.
//   test_fw_elf hello <image.elf>
//       boots the image on the simulator through the host CSR loader (EX §3.3: receive CRC and
//       VERIFY per segment), RV_BOOT = 0, RV_RUN; the first record must be HELLO with r2 / r3 = the
//       id tsr-elf computes (= the stored one), r4[15:0] = the note's sdk_abi, r4[31:16] = the kind
//       table signature of .tsr_kinds (tsr/opl.hpp kinds_sig), r5 = the kernel count and r6 = the
//       note's basis_version (EX §3.4, SW §6.3 R1, §6.9); under the fixed and 7 random schedules,
//       race detector fatal.
// Exit 0 when every check passed, else 1.
#include "tsr_elf.h"

#include "tsr/opl.hpp"   // kind_rev, kind_name, kinds_sig (SW §6.9)
#include "tsr_bus.h"
#include "tsr_isa.h"
#include "tsr_sim.h"
#include "tsr_trace.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>

namespace {

int g_fail = 0, g_checks = 0;

void check_(bool ok, const char * file, int line, const char * fmt, ...) __attribute__((format(printf, 4, 5)));
void check_(bool ok, const char * file, int line, const char * fmt, ...) {
    ++g_checks;
    if (ok) return;
    ++g_fail;
    std::fprintf(stderr, "FAIL %s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}
#define CHECK(c, ...) check_((c), __FILE__, __LINE__, __VA_ARGS__)

using Bytes = std::vector<uint8_t>;

bool read_file(const std::string & p, Bytes * out) {
    FILE * f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    out->clear();
    uint8_t b[65536];
    size_t k;
    while ((k = std::fread(b, 1, sizeof(b), f)) > 0) out->insert(out->end(), b, b + k);
    std::fclose(f);
    return true;
}
bool write_file(const std::string & p, const Bytes & b) {
    FILE * f = std::fopen(p.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    return std::fclose(f) == 0 && ok;
}
bool exists(const std::string & p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
// The capability the images need as tsr-elf note prints it ("MAJOR.MINOR" of VERSION[15:0], tsr_tables.h).
std::string cap_min_str() {
    return std::to_string((TSR_VERSION_VALUE >> 8) & 0xFFu) + "." + std::to_string(TSR_VERSION_VALUE & 0xFFu);
}

std::string slurp(const std::string & p) {
    Bytes b;
    return read_file(p, &b) ? std::string(b.begin(), b.end()) : std::string();
}

void wr16(uint8_t * p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void wr32(uint8_t * p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
uint16_t rd16(const uint8_t * p) { return uint16_t(p[0] | p[1] << 8); }
uint32_t rd32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }

std::string hex(const uint8_t * d, size_t n) {
    std::string s;
    char b[3];
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(b, sizeof(b), "%02x", d[i]);
        s += b;
    }
    return s;
}
std::string hex64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", (unsigned long long)v);
    return b;
}

uint32_t crc32_zlib(const uint8_t * p, size_t n) {   // EX §3.3 LOAD_CRC
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
    }
    return ~c;
}

// An image from bytes; nullptr (and a failed check) if it does not parse.
tsr_elf * parse(const Bytes & b, const char * what) {
    tsr_elf * e = nullptr;
    char err[256];
    const int r = tsr_elf_parse(b.data(), b.size(), &e, err, sizeof(err));
    CHECK(r == 0, "%s: parse: %s", what, err);
    return r == 0 ? e : nullptr;
}

// tsr_elf_check's report and error count.
struct CheckResult {
    int         nerr = -1;
    std::string report;
    bool has(const char * tag) const { return report.find(std::string("[") + tag + "]") != std::string::npos; }
};
CheckResult run_check(const tsr_elf * e, unsigned flags, const tsr_elf_limits * lim = nullptr) {
    CheckResult r;
    std::vector<char> rep(1 << 16);
    r.nerr = tsr_elf_check(e, lim, flags, rep.data(), rep.size());
    r.report = rep.data();
    return r;
}

// The id of sw/sim/tsr_trace.c (an independent implementation of EX §3.2).
uint64_t trace_id(const Bytes & b, const char * what) {
    uint64_t id = 0;
    char err[256] = "";
    CHECK(tsr_trace2_image_id(b.data(), b.size(), &id, err, sizeof(err)) == 0, "%s: tsr_trace2_image_id: %s", what, err);
    return id;
}

// The simulator's ELF reader (tsr_bus.c, what sw/sim's tests load) sees the same segments.
void check_sim_reader(const std::string & path, const tsr_elf * e) {
    tsr_rv_elf r;
    if (tsr_rv_elf_read(path.c_str(), &r) != 0) {
        CHECK(false, "%s: tsr_rv_elf_read", path.c_str());
        return;
    }
    std::vector<std::pair<uint32_t, uint32_t>> a, b;
    for (uint32_t i = 0; i < r.nseg; ++i) a.emplace_back(r.seg[i].paddr, r.seg[i].filesz);
    for (uint32_t i = 0; i < tsr_elf_nseg(e); ++i) b.emplace_back(tsr_elf_segment(e, i)->paddr, tsr_elf_segment(e, i)->filesz);
    std::sort(a.begin(), a.end());
    CHECK(a == b, "%s: the simulator's ELF reader sees other PT_LOAD segments", path.c_str());
    CHECK(r.entry == tsr_elf_entry(e), "%s: entry", path.c_str());
    tsr_rv_elf_free(&r);
}

// ---------------------------------------------------------------------------
// Known answers
// ---------------------------------------------------------------------------
void test_sha256() {
    struct { std::string msg; const char * hex; } kat[] = {   // FIPS 180-4 / NIST CAVP examples
        { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
        { "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
        { std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" },
    };
    for (const auto & k : kat) {
        uint8_t d[32];
        tsr_elf_sha256(k.msg.data(), k.msg.size(), d);
        CHECK(hex(d, 32) == k.hex, "SHA-256 of a %zu-byte message: %s", k.msg.size(), hex(d, 32).c_str());
    }
    // Lengths around the padding edges: bytes i % 251, from Python's hashlib.sha256.
    struct { size_t n; const char * hex; } edge[] = {
        { 55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59" },
        { 56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562" },
        { 63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488" },
        { 64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108" },
        { 65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781" },
        { 119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6" },
        { 120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c" },
        { 128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5" },
    };
    for (const auto & k : edge) {
        Bytes m(k.n);
        for (size_t i = 0; i < k.n; ++i) m[i] = uint8_t(i % 251);
        uint8_t d[32];
        tsr_elf_sha256(m.data(), m.size(), d);
        CHECK(hex(d, 32) == k.hex, "SHA-256 of %zu bytes i %% 251: %s", k.n, hex(d, 32).c_str());
    }
}

void test_fnv() {
    CHECK(tsr_elf_fnv1a("") == 0x811c9dc5u, "FNV-1a('')");
    CHECK(tsr_elf_fnv1a("a") == 0xe40c292cu, "FNV-1a('a')");
    CHECK(tsr_elf_fnv1a("foobar") == 0xbf9cf968u, "FNV-1a('foobar')");
    CHECK(tsr_elf_fnv1a("graph") == 0xb9780647u, "FNV-1a('graph')");
    CHECK(tsr_elf_kernel_rank("graph") == 0 && tsr_elf_kernel_rank("selftest") == 1 && tsr_elf_kernel_rank("replay") == 2 &&
              tsr_elf_kernel_rank("k_x") == 5,
          "kernel ranks");
}

// ---------------------------------------------------------------------------
// Synthesized images: golden ids
// ---------------------------------------------------------------------------
struct SynSeg {
    uint32_t paddr, memsz, flags;
    Bytes    data;
};
struct SynSec {             // a section whose data lies in segment seg at seg_off
    std::string name;
    uint32_t    type, flags, addr, seg, seg_off, size;
};

// ELF32 LE RISC-V ET_EXEC: header, program headers in the given order, segment data, .shstrtab,
// section headers (null, secs, .shstrtab).
Bytes synth(const std::vector<SynSeg> & segs, const std::vector<SynSec> & secs) {
    const uint32_t nph = uint32_t(segs.size());
    Bytes b(52 + 32 * nph, 0);
    std::vector<uint32_t> seg_off;
    for (const SynSeg & s : segs) {
        while (b.size() % 4) b.push_back(0);
        seg_off.push_back(uint32_t(b.size()));
        b.insert(b.end(), s.data.begin(), s.data.end());
    }
    std::string strtab(1, '\0');
    std::vector<uint32_t> name_off;
    for (const SynSec & s : secs) {
        name_off.push_back(uint32_t(strtab.size()));
        strtab += s.name + '\0';
    }
    const uint32_t shstr_name = uint32_t(strtab.size());
    strtab += std::string(".shstrtab") + '\0';
    const uint32_t shstr_off = uint32_t(b.size());
    b.insert(b.end(), strtab.begin(), strtab.end());
    while (b.size() % 4) b.push_back(0);
    const uint32_t nsh = secs.empty() ? 0 : uint32_t(secs.size()) + 2;
    const uint32_t shoff = nsh ? uint32_t(b.size()) : 0;
    b.resize(b.size() + 40 * nsh, 0);
    std::memcpy(b.data(), "\177ELF", 4);
    b[4] = 1; b[5] = 1; b[6] = 1;
    wr16(&b[16], 2);
    wr16(&b[18], 243);
    wr32(&b[20], 1);
    wr32(&b[24], 0x80);
    wr32(&b[28], 52);
    wr32(&b[32], shoff);
    wr16(&b[40], 52);
    wr16(&b[42], 32);
    wr16(&b[44], nph);
    wr16(&b[46], 40);
    wr16(&b[48], nsh);
    wr16(&b[50], nsh ? nsh - 1 : 0);
    for (uint32_t i = 0; i < nph; ++i) {
        uint8_t * ph = &b[52 + 32 * i];
        wr32(ph, 1);
        wr32(ph + 4, seg_off[i]);
        wr32(ph + 8, segs[i].paddr);
        wr32(ph + 12, segs[i].paddr);
        wr32(ph + 16, uint32_t(segs[i].data.size()));
        wr32(ph + 20, segs[i].memsz);
        wr32(ph + 24, segs[i].flags);
        wr32(ph + 28, 4);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        uint8_t * sh = &b[shoff + 40 * (i + 1)];
        wr32(sh, name_off[i]);
        wr32(sh + 4, secs[i].type);
        wr32(sh + 8, secs[i].flags);
        wr32(sh + 12, secs[i].addr);
        wr32(sh + 16, seg_off[secs[i].seg] + secs[i].seg_off);
        wr32(sh + 20, secs[i].size);
        wr32(sh + 32, 1);
    }
    if (nsh) {
        uint8_t * sh = &b[shoff + 40 * (nsh - 1)];
        wr32(sh, shstr_name);
        wr32(sh + 4, 3);
        wr32(sh + 16, shstr_off);
        wr32(sh + 20, uint32_t(strtab.size()));
        wr32(sh + 32, 1);
    }
    return b;
}

Bytes ramp(size_t n, uint8_t start) {
    Bytes b(n);
    for (size_t i = 0; i < n; ++i) b[i] = uint8_t(start + i);
    return b;
}

std::vector<Bytes> synth_cases() {
    std::vector<Bytes> v;
    // S0: one ITCM segment of 6 bytes (padded to 8 in the hash), memsz 8.
    v.push_back(synth({ { 0x0, 8, 5, ramp(6, 1) } }, {}));
    // S1: two segments, program headers in descending address order (the id sorts them).
    v.push_back(synth({ { 0x00100000, 16, 6, ramp(16, 0xA0) }, { 0x0, 8, 5, ramp(8, 0x10) } }, {}));
    // S2: S1 with .tsr_id at 0x0010_0008 (the last 8 DTCM bytes) holding 0x1122334455667788.
    Bytes d2 = ramp(16, 0xA0);
    for (int i = 0; i < 8; ++i) d2[size_t(8 + i)] = uint8_t(0x88 - 0x11 * i);
    v.push_back(synth({ { 0x00100000, 16, 6, d2 }, { 0x0, 8, 5, ramp(8, 0x10) } }, { { ".tsr_id", 1, 3, 0x00100008, 0, 8, 8 } }));
    // S3: S2 with .tsr_id = 0: the same id as S2 (EX §3.2: its bytes are taken as zero).
    Bytes d3 = ramp(16, 0xA0);
    std::fill(d3.begin() + 8, d3.end(), uint8_t(0));
    v.push_back(synth({ { 0x00100000, 16, 6, d3 }, { 0x0, 8, 5, ramp(8, 0x10) } }, { { ".tsr_id", 1, 3, 0x00100008, 0, 8, 8 } }));
    // S4: a 1-byte ITCM segment, a bss-only DTCM segment (filesz 0, memsz 64), a 13-byte DTCM one.
    v.push_back(synth({ { 0x0, 4, 5, { 0x5A } }, { 0x00100100, 64, 6, {} }, { 0x00100000, 16, 6, ramp(13, 0x30) } }, {}));
    return v;
}

// EX §3.2 ids of synth_cases(), computed from the files `test_fw_elf dump` writes with an
// independent Python implementation: hashlib.sha256 over the PT_LOADs sorted by p_paddr, each as
// struct.pack('<II', paddr, padded) + data zero-padded to padded, with the .tsr_id bytes zeroed;
// the id is int.from_bytes(digest[:8], 'little').
const uint64_t k_golden[5] = { 0xa543ab83983512a0ull, 0x03ceb8b1984f1ebcull, 0x9b3edece4cc01099ull, 0x9b3edece4cc01099ull,
                                0x70cdbfdc92632adeull };

void test_synth() {
    const std::vector<Bytes> cases = synth_cases();
    for (size_t i = 0; i < cases.size(); ++i) {
        tsr_elf * e = parse(cases[i], "synth");
        if (!e) continue;
        const uint64_t id = tsr_elf_id(e);
        CHECK(id == trace_id(cases[i], "synth"), "S%zu: tsr-elf and tsr_trace.c ids differ", i);
        CHECK(id == k_golden[i], "S%zu: id %016llx, golden %016llx", i, (unsigned long long)id, (unsigned long long)k_golden[i]);
        tsr_elf_free(e);
    }
    CHECK(k_golden[2] == k_golden[3], "S2 / S3: .tsr_id does not enter the id");
}

// ---------------------------------------------------------------------------
// Toolchain-built images (elf_img.S + link.ld + tsr_note.S)
// ---------------------------------------------------------------------------
// Section header (in the bytes) of the first section named name, or nullptr.
uint8_t * shdr(Bytes & b, const char * name) {
    const uint32_t shoff = rd32(&b[32]), shnum = rd16(&b[48]), shstrndx = rd16(&b[50]);
    const uint32_t stroff = rd32(&b[shoff + 40 * shstrndx + 16]);
    for (uint32_t i = 0; i < shnum; ++i) {
        uint8_t * sh = &b[shoff + 40 * i];
        if (std::strcmp(reinterpret_cast<const char *>(&b[stroff + rd32(sh)]), name) == 0) return sh;
    }
    return nullptr;
}
uint32_t sec_off(Bytes & b, const char * name) {
    uint8_t * sh = shdr(b, name);
    CHECK(sh != nullptr, "section %s", name);
    return sh ? rd32(sh + 16) : 0;
}
// Program header of the PT_LOAD with the given paddr.
uint8_t * phdr(Bytes & b, uint32_t paddr) {
    for (uint32_t i = 0; i < rd16(&b[44]); ++i) {
        uint8_t * ph = &b[rd32(&b[28]) + 32 * i];
        if (rd32(ph) == 1 && rd32(ph + 12) == paddr) return ph;
    }
    CHECK(false, "no PT_LOAD at 0x%08x", paddr);
    return &b[0];
}

struct Img {
    std::string path;
    Bytes       b;
    tsr_elf *   e = nullptr;
};
bool load_img(Img * m, const std::string & path) {
    m->path = path;
    if (!read_file(path, &m->b)) {
        CHECK(false, "missing test image %s", path.c_str());
        return false;
    }
    m->e = parse(m->b, path.c_str());
    return m->e != nullptr;
}

void test_layout(const Img & m) {
    const tsr_elf * e = m.e;
    const CheckResult c0 = run_check(e, TSR_ELF_CHECK_BASE);
    CHECK(c0.nerr == 0 && c0.report.empty(), "unstamped image: checks without the id pass:\n%s", c0.report.c_str());
    const CheckResult c1 = run_check(e, TSR_ELF_CHECK_ID);
    CHECK(c1.nerr == 1 && c1.has("id-mismatch") && c1.report.find("not stamped") != std::string::npos,
          "unstamped image: id check fails as 'not stamped':\n%s", c1.report.c_str());

    // Kernel table: sorted by (rank, name) although elf_img.S emits it in reverse.
    static const char * const names[5] = { "graph", "selftest", "replay", "k_alpha", "k_beta" };
    static const uint8_t nargs[5] = { 3, 5, 3, 3, 0 };
    CHECK(tsr_elf_n_kernels(e) == 5, "5 kernels");
    for (uint32_t i = 0; i < 5 && i < tsr_elf_n_kernels(e); ++i) {
        const tsr_elf_kernel * k = tsr_elf_kernel_at(e, i);
        uint32_t sym = 0;
        CHECK(std::strcmp(k->name, names[i]) == 0, "kernel %u is %s, expected %s", i, k->name, names[i]);
        CHECK(tsr_elf_symbol(e, names[i], &sym) == 0 && sym == k->entry, "kernel %u entry = symbol", i);
        CHECK(k->nargs == nargs[i] && k->name_hash == tsr_elf_fnv1a(names[i]), "kernel %u nargs / hash", i);
        CHECK(tsr_elf_kernel_find(e, names[i]) == int(i), "find %s", names[i]);
    }
    const tsr_elf_kernel * ka = tsr_elf_kernel_at(e, 3);
    CHECK(ka && ka->flags == TSR_ELF_KFLAG_ARGBLK && ka->argblk_lines == 2, "k_alpha ARGBLK, 2 lines");
    CHECK(tsr_elf_kernel_find(e, "nope") == -1, "find absent kernel");

    // Note: link.ld's values.
    tsr_elf_sizes z;
    tsr_elf_get_sizes(e, nullptr, &z);
    tsr_elf_note n;
    CHECK(tsr_elf_get_note(e, &n) == 0, "note present");
    CHECK(n.sdk_abi == TSR_SDK_ABI && n.cap_min == (TSR_VERSION_VALUE & 0xFFFFu) && n.caps_req == 0 && n.n_kernels == 5 &&
              n.basis_version == TSR_BASIS_VERSION && n.rsv == 0,
          "note: sdk_abi %u cap_min 0x%04x caps %u n %u basis 0x%06x", n.sdk_abi, n.cap_min, n.caps_req, n.n_kernels,
          n.basis_version);
    // Kind table (SW §6.9): elf_img.S declares MUL_MAT, NOP, JOIN at revision 1.
    uint8_t rev[TSR_ELF_N_KINDS];
    char why[96];
    CHECK(tsr_elf_kinds(e, rev, why, sizeof(why)) == 3 && rev[0x00] == 1 && rev[0x01] == 1 && rev[0x10] == 1,
          "kinds: NOP, JOIN, MUL_MAT (%s)", why);
    CHECK(tsr::opl::kinds_sig(rev) == 0x512Eu, "kinds signature 0x%04x", tsr::opl::kinds_sig(rev));
    uint8_t none[TSR_ELF_N_KINDS] = {};
    CHECK(tsr::opl::kinds_sig(none) == 0xB51Bu, "empty kind table signature 0x%04x", tsr::opl::kinds_sig(none));
    CHECK(n.itcm_min == z.itcm_used && n.dtcm_min == TSR_DTCM_BYTES, "note: itcm_min %u (used %u), dtcm_min %u", n.itcm_min,
          z.itcm_used, n.dtcm_min);
    uint32_t a = 0, off = 0, size = 0, type = 0;
    CHECK(tsr_elf_section(e, ".note.tsr", &a, &off, &size, &type) == 0 && type == 7 && size == 48, "note: SHT_NOTE, 48 bytes");

    // Memory map (SW §6.2, CC §4.1).
    uint32_t v = 0;
    CHECK(tsr_elf_entry(e) == 0x80 && tsr_elf_symbol(e, "_start", &v) == 0 && v == 0x80, "_start at 0x80");
    CHECK(tsr_elf_symbol(e, "__stack_top", &v) == 0 && v == TSR_RV_DTCM_BASE + TSR_DTCM_BYTES, "stack at the top of DTCM");
    uint32_t sb = 0, tt = 0, tb = 0;
    CHECK(tsr_elf_symbol(e, "__stack_bottom", &sb) == 0 && v - sb == 8192, "8 KiB stack");
    CHECK(tsr_elf_symbol(e, "__trap_stack_top", &tt) == 0 && tsr_elf_symbol(e, "__trap_stack_bottom", &tb) == 0 && tt == sb &&
              tt - tb == 512,
          "512 B trap stack below the stack");
    uint32_t z0 = 0, z1 = 0, b0 = 0, c0a = 0, c1a = 0;
    CHECK(tsr_elf_symbol(e, "__zero_start", &z0) == 0 && tsr_elf_symbol(e, "__zero_end", &z1) == 0 &&
              tsr_elf_symbol(e, "__bss_start", &b0) == 0 && tsr_elf_symbol(e, "__tsr_crash_start", &c0a) == 0 &&
              tsr_elf_symbol(e, "__tsr_crash_end", &c1a) == 0 && z0 == b0 && c1a == z1 && c1a - c0a == 192 && z0 % 4 == 0 &&
              z1 % 4 == 0,
          "zeroed range = .bss + .tsr_crash");
    CHECK(tsr_elf_symbol(e, "__tsr_kernels_start", &v) == 0 && v == TSR_RV_DTCM_BASE, ".tsr_kernels at DTCM base");
    CHECK(tsr_elf_section(e, ".tsr_id", &a, &off, &size, &type) == 0 && size == 8 && a % 8 == 0 && type == 1, ".tsr_id 8 bytes");
    CHECK(tsr_elf_section(e, ".tsr_dbg", &a, &off, &size, &type) == 0 && size == 64 && type == 1, ".tsr_dbg 64 bytes, loaded");
    CHECK(tsr_elf_section(e, ".tsr_crash", &a, &off, &size, &type) == 0 && size == 192 && type == 8, ".tsr_crash NOBITS");
    CHECK(tsr_elf_section(e, ".vectors", &a, &off, &size, &type) == 0 && a == 0 && size >= 0x80, ".vectors at 0");

    // .tsr_logfmt: in the file, at 0xF000_0000, in no PT_LOAD.
    CHECK(tsr_elf_section(e, ".tsr_logfmt", &a, &off, &size, &type) == 0 && a == TSR_LOGFMT_VMA && type == 1 && size == 21,
          ".tsr_logfmt at 0x%08x size %u", a, size);
    for (uint32_t i = 0; i < tsr_elf_nseg(e); ++i) {
        const tsr_elf_seg * g = tsr_elf_segment(e, i);
        CHECK(!(TSR_LOGFMT_VMA >= g->paddr && TSR_LOGFMT_VMA < g->paddr + g->memsz), "no PT_LOAD holds .tsr_logfmt");
        CHECK(g->paddr < TSR_RV_DTCM_BASE ? g->paddr + g->memsz <= TSR_ITCM_BYTES : g->paddr + g->memsz <= TSR_RV_DTCM_BASE + TSR_DTCM_BYTES,
              "segment 0x%08x inside its TCM", g->paddr);
    }
    CHECK(tsr_elf_symbol(e, "fmt_hello", &v) == 0 && v == TSR_LOGFMT_VMA, "fmt_hello at 0xF000_0000");
    CHECK(tsr_elf_logfmt_count(e) == 2, "2 log formats");
    const char * s = tsr_elf_logfmt(e, TSR_LOGFMT_VMA);
    CHECK(s && std::strcmp(s, "hello %d") == 0, "logfmt[0]");
    s = tsr_elf_logfmt(e, TSR_LOGFMT_VMA + 9);
    CHECK(s && std::strcmp(s, "x=%08x\ty=%u") == 0, "logfmt[1]");
    s = tsr_elf_logfmt(e, TSR_LOGFMT_VMA + 2);
    CHECK(s && std::strcmp(s, "llo %d") == 0, "logfmt inside a string (merged tails)");
    CHECK(tsr_elf_logfmt(e, TSR_LOGFMT_VMA + 21) == nullptr && tsr_elf_logfmt(e, 0x100) == nullptr, "logfmt out of range");
    uint32_t la = 0;
    CHECK(tsr_elf_logfmt_at(e, 1, &la) != nullptr && la == TSR_LOGFMT_VMA + 9, "logfmt_at");
    CHECK(tsr_elf_symbol(e, "msg", &v) == 0 && tsr_elf_string(e, v) && std::strcmp(tsr_elf_string(e, v), "a loaded string") == 0,
          "%%s string in .rodata");
    CHECK(tsr_elf_string(e, TSR_LOGFMT_VMA) == nullptr, "%%s strings come from loaded sections only");

    // Sizes: every DTCM section counts, incl. not loaded ones and both stacks (SW §7.8).
    CHECK(z.itcm_used == n.itcm_min && z.itcm_used < 0x200, "ITCM used %u", z.itcm_used);
    CHECK(z.dtcm_zeroed == 1024 + 192 + 8192 + 512, "DTCM zeroed / stack %u", z.dtcm_zeroed);
    // This layout has no hole inside the loaded data or between it and .bss / .tsr_crash: what is
    // used is exactly the loaded words plus the zeroed sections and the stacks.
    CHECK(z.dtcm_used == z.dtcm_zeroed + z.dtcm_loaded, "DTCM used %u = zeroed %u + loaded %u", z.dtcm_used, z.dtcm_zeroed,
          z.dtcm_loaded);
    CHECK(z.logfmt == 21 && !z.over_itcm && !z.over_dtcm && z.budget_itcm == 80 * 1024 && z.budget_dtcm == 56 * 1024, "sizes");
    std::vector<char> rep(1 << 16);
    tsr_elf_size_report(e, nullptr, rep.data(), rep.size());
    const std::string r = rep.data();
    CHECK(r.find("SW §7.8 budget 81920 B: ok") != std::string::npos && r.find("SW §7.8 budget 57344 B: ok") != std::string::npos &&
              r.find(".stack") != std::string::npos && r.find("not stamped") != std::string::npos &&
              r.find("kernels    5") != std::string::npos,
          "size report:\n%s", r.c_str());
    check_sim_reader(m.path, e);
}

void test_stamp_and_tamper(const Img & m, const std::string & tmpdir) {
    const uint64_t id = tsr_elf_id(m.e);
    CHECK(id == trace_id(m.b, "elf_img"), "elf_img: tsr-elf and tsr_trace.c ids agree");
    // Stamp in memory and through a file.
    tsr_elf * e = parse(m.b, "copy");
    if (!e) return;
    CHECK(tsr_elf_set_id(e, id) == 0, "set_id");
    uint64_t st = 0;
    CHECK(tsr_elf_stored_id(e, &st) == 0 && st == id && tsr_elf_id(e) == id, "stamped: stored = computed, id unchanged");
    const std::string p = tmpdir + "/stamped.elf";
    char err[256];
    CHECK(tsr_elf_write(e, p.c_str(), err, sizeof(err)) == 0, "write: %s", err);
    tsr_elf_free(e);
    Bytes sb;
    CHECK(read_file(p, &sb), "read back");
    tsr_elf * s = parse(sb, "stamped");
    if (!s) return;
    const CheckResult c = run_check(s, TSR_ELF_CHECK_ID | TSR_ELF_CHECK_BASE);
    CHECK(c.nerr == 0 && c.report.empty(), "stamped image passes every check:\n%s", c.report.c_str());
    CHECK(trace_id(sb, "stamped") == id, "tsr_trace.c: the stamped image has the same id");
    uint32_t idaddr = 0, idoff = 0;
    CHECK(tsr_elf_section(s, ".tsr_id", &idaddr, &idoff, nullptr, nullptr) == 0 && idoff != 0, ".tsr_id offset");
    uint64_t le = 0;
    for (int i = 0; i < 8; ++i) le |= uint64_t(sb[idoff + uint32_t(i)]) << (8 * i);
    CHECK(le == id, ".tsr_id holds the id little-endian (HELLO r2 = low word, r3 = high word)");
    tsr_elf_free(s);

    // Tampering. Each case: the bytes, whether the computed id must change, whether the stored id
    // still matches (check with TSR_ELF_CHECK_ID passes).
    struct Case { const char * what; Bytes b; bool id_changes; bool still_ok; };
    std::vector<Case> cases;
    auto mk = [&](const char * what, bool chg, bool ok, auto && f) {
        Bytes b = sb;
        f(b);
        cases.push_back({ what, std::move(b), chg, ok });
    };
    mk(".text byte", true, false, [&](Bytes & b) { b[sec_off(b, ".text")] ^= 0x01; });
    mk(".vectors byte", true, false, [&](Bytes & b) { b[sec_off(b, ".vectors") + 3] ^= 0x80; });
    mk(".data byte", true, false, [&](Bytes & b) { b[sec_off(b, ".data")] ^= 0xFF; });
    mk(".tsr_kernels byte", true, false, [&](Bytes & b) { b[sec_off(b, ".tsr_kernels") + 8] ^= 0x01; });
    mk(".tsr_dbg byte", true, false, [&](Bytes & b) { b[sec_off(b, ".tsr_dbg") + 20] ^= 0x01; });
    mk("ITCM segment moved by 4", true, false, [&](Bytes & b) {
        uint8_t * ph = phdr(b, 0);
        wr32(ph + 8, 4);
        wr32(ph + 12, 4);
    });
    mk("DTCM segment shortened", true, false, [&](Bytes & b) {
        uint8_t * ph = phdr(b, TSR_RV_DTCM_BASE);
        wr32(ph + 16, rd32(ph + 16) - 4);
    });
    mk(".tsr_id byte", false, false, [&](Bytes & b) { b[idoff + 5] ^= 0x10; });
    mk(".tsr_logfmt text", false, true, [&](Bytes & b) { b[sec_off(b, ".tsr_logfmt")] = 'H'; });
    mk(".debug_line byte", false, true, [&](Bytes & b) { b[sec_off(b, ".debug_line") + 4] ^= 0x01; });
    mk(".debug_str byte", false, true, [&](Bytes & b) { b[sec_off(b, ".debug_str")] ^= 0x01; });
    mk("bytes appended", false, true, [&](Bytes & b) { b.insert(b.end(), 64, 0xEE); });
    for (const Case & k : cases) {
        tsr_elf * t = parse(k.b, k.what);
        if (!t) continue;
        const uint64_t tid = tsr_elf_id(t);
        CHECK((tid != id) == k.id_changes, "tamper %s: id %s", k.what, k.id_changes ? "must change" : "must not change");
        CHECK(tid == trace_id(k.b, k.what), "tamper %s: tsr_trace.c agrees", k.what);
        const CheckResult r = run_check(t, TSR_ELF_CHECK_ID);
        CHECK(k.still_ok ? r.nerr == 0 : (r.has("id-mismatch") && r.nerr >= 1), "tamper %s: check:\n%s", k.what, r.report.c_str());
        tsr_elf_free(t);
    }
    // The edited format string is what a decoder now reads.
    tsr_elf * t = parse(cases[8].b, "logfmt edit");
    if (t) {
        CHECK(tsr_elf_logfmt(t, TSR_LOGFMT_VMA) && std::strcmp(tsr_elf_logfmt(t, TSR_LOGFMT_VMA), "Hello %d") == 0, "edited logfmt");
        tsr_elf_free(t);
    }
}

// One patched image per validation rule: the expected tag must be reported.
void test_validation(const Img & stamped, const Img & nobase, const Img & big) {
    const Bytes & sb = stamped.b;
    const uint32_t kt = [&] { Bytes b = sb; return sec_off(b, ".tsr_kernels"); }();
    const uint32_t nd = [&] { Bytes b = sb; return sec_off(b, ".note.tsr") + 16; }();   // descriptor
    const uint32_t kk = [&] { Bytes b = sb; return sec_off(b, ".tsr_kinds"); }();       // MUL_MAT, NOP, JOIN
    auto kd = [&](Bytes & b, uint32_t i) { return &b[kt + 36 * i]; };
    auto set_name = [&](Bytes & b, uint32_t i, const char * nm) {
        uint8_t * p = kd(b, i);
        std::memset(p + 12, 0, 24);
        std::memcpy(p + 12, nm, std::strlen(nm));
        wr32(p, tsr_elf_fnv1a(nm));
    };
    struct Case {
        const char *   tag;      // expected tag
        const char *   what;
        Bytes          b;
        tsr_elf_limits lim;
        unsigned       flags;
    };
    std::vector<Case> cases;
    auto mk = [&](const char * tag, const char * what, auto && f, tsr_elf_limits lim = { 0, 0, 0, 0 }, unsigned fl = 0) {
        Bytes b = sb;
        f(b);
        cases.push_back({ tag, what, std::move(b), lim, fl });
    };
    // Kernel table (SW §6.2): entries 0 graph, 1 selftest, 2 replay, 3 k_alpha (ARGBLK, 2 lines), 4 k_beta.
    mk("kt-nargs", "nargs 8", [&](Bytes & b) { kd(b, 1)[8] = 8; });
    mk("kt-hash", "name_hash off by one bit", [&](Bytes & b) { kd(b, 0)[0] ^= 1; });
    mk("kt-argblk", "ARGBLK with 0 lines", [&](Bytes & b) { wr16(kd(b, 3) + 10, 0); });
    mk("kt-argblk", "ARGBLK with 17 lines", [&](Bytes & b) { wr16(kd(b, 3) + 10, 17); });
    mk("kt-argblk", "lines without ARGBLK", [&](Bytes & b) { wr16(kd(b, 0) + 10, 3); });
    mk("kt-argblk", "ARGBLK with nargs 0", [&](Bytes & b) { kd(b, 4)[9] = 1; wr16(kd(b, 4) + 10, 1); });
    mk("kt-flags", "reserved flag bit", [&](Bytes & b) { kd(b, 4)[9] = 2; });
    mk("kt-entry", "entry in DTCM", [&](Bytes & b) { wr32(kd(b, 4) + 4, TSR_RV_DTCM_BASE); });
    mk("kt-entry", "entry is not the kernel's symbol", [&](Bytes & b) { wr32(kd(b, 4) + 4, rd32(kd(b, 3) + 4)); });
    mk("kt-entry", "odd entry", [&](Bytes & b) { wr32(kd(b, 4) + 4, rd32(kd(b, 4) + 4) + 1); });
    mk("kt-name", "name with a '-'", [&](Bytes & b) { set_name(b, 4, "k-beta"); });
    mk("kt-name", "garbage after the NUL", [&](Bytes & b) { kd(b, 4)[12 + 20] = 'x'; });
    mk("kt-name", "24 characters, no NUL", [&](Bytes & b) { set_name(b, 4, "k_beta_123456789abcdefgh"); });
    mk("kt-name", "empty name", [&](Bytes & b) { set_name(b, 4, ""); });
    mk("kt-dup", "two kernels k_alpha", [&](Bytes & b) { set_name(b, 4, "k_alpha"); });
    mk("kt-order", "k_alpha / k_beta swapped", [&](Bytes & b) {
        uint8_t t[36];
        std::memcpy(t, kd(b, 3), 36);
        std::memcpy(kd(b, 3), kd(b, 4), 36);
        std::memcpy(kd(b, 4), t, 36);
    });
    mk("kt-base", "graph / selftest swapped", [&](Bytes & b) {
        uint8_t t[36];
        std::memcpy(t, kd(b, 0), 36);
        std::memcpy(kd(b, 0), kd(b, 1), 36);
        std::memcpy(kd(b, 1), t, 36);
    }, { 0, 0, 0, 0 }, TSR_ELF_CHECK_BASE);
    mk("kt-size", ".tsr_kernels 4 bytes short", [&](Bytes & b) { uint8_t * sh = shdr(b, ".tsr_kernels"); wr32(sh + 20, rd32(sh + 20) - 4); });
    // .note.tsr
    mk("note-abi", "sdk_abi 2", [&](Bytes & b) { wr32(&b[nd], 2); });
    mk("note-cap", "cap_min 0", [&](Bytes & b) { wr32(&b[nd + 4], 0); });
    mk("note-cap", "cap_min 0x10000", [&](Bytes & b) { wr32(&b[nd + 4], 0x10000); });
    mk("note-basis", "basis_version 0 (built before versioning)", [&](Bytes & b) { wr32(&b[nd + 24], 0); });
    mk("note-basis", "basis_version [31:24] != 0", [&](Bytes & b) { wr32(&b[nd + 24], 0x01000100); });
    // .tsr_kinds (SW §6.9)
    mk("kinds", "kind 0x10 twice", [&](Bytes & b) { b[kk + 4] = 0x10; });
    mk("kinds", "kind 0x40", [&](Bytes & b) { b[kk + 8] = 0x40; });
    mk("kinds", "revision 0", [&](Bytes & b) { b[kk + 1] = 0; });
    mk("kinds", "rsvd bits", [&](Bytes & b) { b[kk + 2] = 1; });
    mk("kinds", ".tsr_kinds 2 bytes short", [&](Bytes & b) { uint8_t * sh = shdr(b, ".tsr_kinds"); wr32(sh + 20, rd32(sh + 20) - 2); });
    mk("note-itcm", "itcm_min below the image", [&](Bytes & b) { wr32(&b[nd + 8], 4); });
    mk("note-itcm", "itcm_min beyond ITCM", [&](Bytes & b) { wr32(&b[nd + 8], TSR_ITCM_BYTES + 4); });
    mk("note-dtcm", "dtcm_min below the stack top", [&](Bytes & b) { wr32(&b[nd + 12], 0x8000); });
    mk("note-dtcm", "dtcm_min beyond DTCM", [&](Bytes & b) { wr32(&b[nd + 12], TSR_DTCM_BYTES * 2); });
    mk("note-nkernels", "n_kernels 6", [&](Bytes & b) { wr32(&b[nd + 20], 6); });
    mk("note-rsv", "word 7 set", [&](Bytes & b) { wr32(&b[nd + 28], 1); });
    mk("note-format", "28-byte descriptor", [&](Bytes & b) { wr32(&b[nd - 12], 28); });
    mk("note-missing", "owner TSX", [&](Bytes & b) { b[nd - 2] = 'X'; });
    mk("note-missing", "note type 2", [&](Bytes & b) { wr32(&b[nd - 8], 2); });
    mk("note-alloc", ".note.tsr allocated", [&](Bytes & b) { uint8_t * sh = shdr(b, ".note.tsr"); wr32(sh + 8, 2); });
    // Segments, entry, sections (EX §3.1)
    mk("entry", "e_entry 0x84", [&](Bytes & b) { wr32(&b[24], 0x84); });
    mk("seg-align", "DTCM segment at +2", [&](Bytes & b) {
        uint8_t * ph = phdr(b, TSR_RV_DTCM_BASE);
        wr32(ph + 8, TSR_RV_DTCM_BASE + 2);
        wr32(ph + 12, TSR_RV_DTCM_BASE + 2);
    });
    mk("seg-vaddr", "p_vaddr != p_paddr", [&](Bytes & b) { wr32(phdr(b, 0) + 8, 0x1000); });
    mk("seg-range", "ITCM segment beyond a 128-byte ITCM", [](Bytes &) {}, { 128, 0, 0, 0 });
    mk("seg-range", "stack beyond a 32 KiB DTCM", [](Bytes &) {}, { 0, 32 * 1024, 0, 0 });
    mk("seg-overlap", "ITCM segment grown into the next", [&](Bytes & b) {
        uint8_t * ph = phdr(b, 0);
        wr32(ph + 20, TSR_RV_DTCM_BASE + 0x10);
    }, { 0x00200000, 0, 0, 0 });
    mk("id-missing", ".tsr_id renamed", [&](Bytes & b) { uint8_t * sh = shdr(b, ".tsr_id"); wr32(sh, rd32(sh) + 1); });
    mk("id-size", ".tsr_id 4 bytes", [&](Bytes & b) { wr32(shdr(b, ".tsr_id") + 20, 4); });
    mk("id-load", ".tsr_id address outside the loaded data", [&](Bytes & b) { wr32(shdr(b, ".tsr_id") + 12, TSR_RV_DTCM_BASE + 0x8000); });
    mk("logfmt-alloc", ".tsr_logfmt allocated", [&](Bytes & b) { wr32(shdr(b, ".tsr_logfmt") + 8, 2); });
    mk("logfmt-addr", ".tsr_logfmt at 0xF000_0100", [&](Bytes & b) { wr32(shdr(b, ".tsr_logfmt") + 12, TSR_LOGFMT_VMA + 0x100); });
    mk("logfmt-term", ".tsr_logfmt unterminated", [&](Bytes & b) {
        uint8_t * sh = shdr(b, ".tsr_logfmt");
        b[rd32(sh + 16) + rd32(sh + 20) - 1] = 'x';
    });
    mk("logfmt-nobits", ".tsr_logfmt NOBITS", [&](Bytes & b) { wr32(shdr(b, ".tsr_logfmt") + 4, 8); });
    mk("vectors", ".vectors moved", [&](Bytes & b) { wr32(shdr(b, ".vectors") + 12, 0x10); });
    for (const Case & k : cases) {
        tsr_elf * t = parse(k.b, k.what);
        if (!t) continue;
        const CheckResult r = run_check(t, k.flags | TSR_ELF_CHECK_ID, &k.lim);
        CHECK(r.nerr >= 1 && r.has(k.tag), "%s: expected [%s]:\n%s", k.what, k.tag, r.report.c_str());
        tsr_elf_free(t);
    }
    // The unpatched stamped image passes all of them (also the limits used above, where valid).
    const CheckResult ok = run_check(stamped.e, TSR_ELF_CHECK_ID | TSR_ELF_CHECK_BASE);
    CHECK(ok.nerr == 0, "stamped image:\n%s", ok.report.c_str());
    // Base kernels missing; budget overrun is a warning, not an error.
    const CheckResult nb = run_check(nobase.e, TSR_ELF_CHECK_BASE);
    CHECK(nb.nerr == 3 && nb.has("kt-base"), "NO_BASE image:\n%s", nb.report.c_str());
    const CheckResult nb0 = run_check(nobase.e, 0);
    CHECK(nb0.nerr == 0, "NO_BASE image without the base check:\n%s", nb0.report.c_str());
    const CheckResult bg = run_check(big.e, 0);
    CHECK(bg.nerr == 0 && bg.has("budget-itcm") && bg.report.find("warning: [budget-itcm]") != std::string::npos,
          "BIG_TEXT image: budget warning only:\n%s", bg.report.c_str());
    tsr_elf_sizes z;
    tsr_elf_get_sizes(big.e, nullptr, &z);
    CHECK(z.over_itcm && !z.over_dtcm && z.itcm_used > 82 * 1024, "BIG_TEXT sizes");
    const tsr_elf_limits loose = { 0, 0, 96 * 1024, 0 };
    tsr_elf_get_sizes(big.e, &loose, &z);
    CHECK(!z.over_itcm, "BIG_TEXT within a 96 KiB budget");
}

// ---------------------------------------------------------------------------
// The command line
// ---------------------------------------------------------------------------
struct Run {
    int         rc;
    std::string out;
};
Run run(const std::string & cmd) {
    Run r{ -1, "" };
    FILE * p = popen((cmd + " 2>&1").c_str(), "r");
    if (!p) return r;
    char b[4096];
    size_t k;
    while ((k = std::fread(b, 1, sizeof(b), p)) > 0) r.out.append(b, k);
    const int st = pclose(p);
    r.rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return r;
}

void test_cli(const std::string & tool, const std::string & dir, const std::string & tmp) {
    const std::string img = dir + "/elf_img.elf", big = dir + "/elf_img_big_text.elf";
    const std::string out = tmp + "/cli.elf", rep = tmp + "/cli.size.txt", lf = tmp + "/cli.logfmt.txt";
    Run r = run(tool + " stamp " + img + " -o " + out + " --report " + rep + " --logfmt " + lf);
    CHECK(r.rc == 0 && exists(out) && r.out.find(" 5 kernels, 2 log formats") != std::string::npos, "stamp: rc %d\n%s", r.rc,
          r.out.c_str());
    r = run(tool + " check " + out + " --base");
    CHECK(r.rc == 0 && r.out.find(": ok, id ") != std::string::npos, "check stamped: rc %d\n%s", r.rc, r.out.c_str());
    r = run(tool + " check " + img);
    CHECK(r.rc == 1 && r.out.find("[id-mismatch]") != std::string::npos && r.out.find("not stamped") != std::string::npos,
          "check unstamped: rc %d\n%s", r.rc, r.out.c_str());
    r = run(tool + " check " + img + " --no-id --base");
    CHECK(r.rc == 0, "check unstamped --no-id: rc %d\n%s", r.rc, r.out.c_str());
    Bytes ib;
    if (read_file(img, &ib) && write_file(tmp + "/inplace.elf", ib)) {   // stamping in place
        r = run(tool + " stamp " + tmp + "/inplace.elf -o " + tmp + "/inplace.elf -q");
        CHECK(r.rc == 0 && r.out.empty(), "stamp in place: rc %d\n%s", r.rc, r.out.c_str());
        r = run(tool + " check " + tmp + "/inplace.elf");
        CHECK(r.rc == 0, "stamped in place: rc %d\n%s", r.rc, r.out.c_str());
    }
    tsr_elf * e = nullptr;
    if (read_file(img, &ib)) e = parse(ib, "cli img");
    const std::string id = e ? hex64(tsr_elf_id(e)) : "?";
    r = run(tool + " id " + out);
    CHECK(r.rc == 0 && r.out == id + "\n", "id: %s", r.out.c_str());
    r = run(tool + " id --stored " + out);
    CHECK(r.rc == 0 && r.out == id + "\n", "id --stored: %s", r.out.c_str());
    r = run(tool + " id --stored " + img);
    CHECK(r.rc == 0 && r.out == "0000000000000000\n", "id --stored unstamped: %s", r.out.c_str());
    CHECK(slurp(lf) == "0xf0000000\thello %d\n0xf0000009\tx=%08x\\ty=%u\n", "logfmt table file:\n%s", slurp(lf).c_str());
    r = run(tool + " logfmt " + out);
    CHECK(r.rc == 0 && r.out == slurp(lf), "logfmt command = stamp's table");
    const std::string sr = slurp(rep);
    CHECK(sr.find("image id   " + id + " (stamped)") != std::string::npos && sr.find("budget 81920 B: ok") != std::string::npos,
          "size report file:\n%s", sr.c_str());
    r = run(tool + " size " + out);
    CHECK(r.rc == 0 && r.out == sr, "size command = stamp's report:\n%s", r.out.c_str());
    r = run(tool + " kernels " + out);
    CHECK(r.rc == 0 && r.out.find("0\tgraph\t") == 0 && r.out.find("3\tk_alpha\t") != std::string::npos &&
              r.out.find("flags=0x01\targblk_lines=2") != std::string::npos,
          "kernels:\n%s", r.out.c_str());
    r = run(tool + " note " + out);
    CHECK(r.rc == 0 && r.out.find("sdk_abi=" + std::to_string(TSR_SDK_ABI) + "\n") == 0 &&
              r.out.find("\ncap_min=" + cap_min_str() + "\n") != std::string::npos && r.out.find("n_kernels=5\n") != std::string::npos &&
              r.out.find("\nbasis_version=" TSR_BASIS_VERSION_STRING "\n") != std::string::npos,
          "note:\n%s", r.out.c_str());
    r = run(tool + " kinds " + out);
    CHECK(r.rc == 0 && r.out == "0x00\tNOP\trev=1\n0x01\tJOIN\trev=1\n0x10\tMUL_MAT\trev=1\nsignature=0x512e\n", "kinds:\n%s",
          r.out.c_str());
    r = run(tool + " --version");
    CHECK(r.rc == 0 && r.out == "tsr-elf (Tesseract Basis " TSR_BASIS_VERSION_STRING ", Basis ABI " + std::to_string(TSR_SDK_ABI) + ")\n",
          "--version: %s", r.out.c_str());
    // Segment CRCs are the LOAD_CRC values of EX §3.3.
    r = run(tool + " segments " + out);
    if (e) {
        std::string want;
        Bytes ob;
        tsr_elf * o = read_file(out, &ob) ? parse(ob, "cli out") : nullptr;
        for (uint32_t i = 0; o && i < tsr_elf_nseg(o); ++i) {
            const tsr_elf_seg * g = tsr_elf_segment(o, i);
            Bytes p(g->padded, 0);
            if (g->filesz) std::memcpy(p.data(), g->data, g->filesz);
            char ln[160];
            std::snprintf(ln, sizeof(ln), "0x%08x\tfilesz=%u\tpadded=%u\tmemsz=%u\tflags=%c%c%c\tcrc32=0x%08x\n", g->paddr, g->filesz,
                          g->padded, g->memsz, (g->flags & 4) ? 'r' : '-', (g->flags & 2) ? 'w' : '-', (g->flags & 1) ? 'x' : '-',
                          crc32_zlib(p.data(), p.size()));
            want += ln;
        }
        CHECK(r.rc == 0 && r.out == want, "segments:\n%s\nwant:\n%s", r.out.c_str(), want.c_str());
        tsr_elf_free(o);
    }
    // A failing stamp removes a stale output; a budget overrun warns, and fails with --strict-budget.
    Bytes bad = ib;
    if (!bad.empty()) {
        bad[sec_off(bad, ".tsr_kernels") + 8] = 9;   // graph nargs 9
        CHECK(write_file(tmp + "/bad.elf", bad), "write bad.elf");
    }
    CHECK(write_file(out, { 1, 2, 3 }), "stale output");
    r = run(tool + " stamp " + tmp + "/bad.elf -o " + out);
    CHECK(r.rc == 1 && !exists(out) && r.out.find("[kt-nargs]") != std::string::npos, "stamp of a bad image: rc %d\n%s", r.rc,
          r.out.c_str());
    r = run(tool + " stamp " + big + " -o " + out);
    CHECK(r.rc == 0 && exists(out) && r.out.find("warning: [budget-itcm]") != std::string::npos, "stamp over budget: rc %d\n%s", r.rc,
          r.out.c_str());
    r = run(tool + " stamp " + big + " -o " + out + " --strict-budget");
    CHECK(r.rc == 1 && !exists(out), "stamp --strict-budget over budget: rc %d\n%s", r.rc, r.out.c_str());
    r = run(tool + " size " + big + " --strict-budget");
    CHECK(r.rc == 1 && r.out.find("EXCEEDED") != std::string::npos, "size --strict-budget: rc %d", r.rc);
    r = run(tool + " size " + big + " --strict-budget --budget-itcm 96K");
    CHECK(r.rc == 0, "size with a 96 KiB budget: rc %d\n%s", r.rc, r.out.c_str());
    r = run(tool + " check " + tmp + "/nonexistent.elf");
    CHECK(r.rc == 1 && r.out.find("cannot open") != std::string::npos, "missing file: rc %d", r.rc);
    r = run(tool + " frobnicate " + img);
    CHECK(r.rc == 2, "unknown command: rc %d", r.rc);
    tsr_elf_free(e);
}

// ---------------------------------------------------------------------------
// Robustness: corrupted and truncated images never crash the reader (run under ASan / UBSan by
// fw_elf_asan); whatever parses also goes through every query.
// ---------------------------------------------------------------------------
void test_corrupt(const Bytes & good) {
    uint32_t x = 0x2545F491u;
    auto rnd = [&] { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
    const uint32_t shoff = rd32(&good[32]);
    int parsed = 0;
    for (int it = 0; it < 3000; ++it) {
        Bytes b = good;
        const uint32_t kind = rnd() % 4;
        if (kind == 0) {
            b.resize(rnd() % b.size());                                    // truncated
        } else {
            const int n = 1 + int(rnd() % 6);
            for (int k = 0; k < n; ++k) {
                // headers (ELF, program, section) most of the time, anywhere else sometimes
                size_t at;
                if (kind == 1) at = rnd() % 0x80;
                else if (kind == 2) at = shoff + rnd() % (b.size() - shoff);
                else at = rnd() % b.size();
                b[at] = uint8_t(rnd());
            }
        }
        tsr_elf * e = nullptr;
        char err[256];
        if (tsr_elf_parse(b.data(), b.size(), &e, err, sizeof(err)) != 0) continue;
        ++parsed;
        std::vector<char> rep(1 << 16);
        (void)tsr_elf_check(e, nullptr, TSR_ELF_CHECK_ID | TSR_ELF_CHECK_BASE, rep.data(), rep.size());
        tsr_elf_size_report(e, nullptr, rep.data(), rep.size());
        (void)tsr_elf_id(e);
        tsr_elf_note n;
        (void)tsr_elf_get_note(e, &n);
        uint8_t rv[TSR_ELF_N_KINDS];
        (void)tsr_elf_kinds(e, rv, nullptr, 0);
        for (uint32_t i = 0; i < tsr_elf_logfmt_count(e); ++i) (void)tsr_elf_logfmt_at(e, i, nullptr);
        (void)tsr_elf_logfmt(e, TSR_LOGFMT_VMA + rnd() % 64);
        (void)tsr_elf_string(e, TSR_RV_DTCM_BASE + rnd() % 1024);
        uint32_t v;
        (void)tsr_elf_symbol(e, "_start", &v);
        tsr_elf_free(e);
    }
    CHECK(parsed > 100, "corruption test: %d of 3000 mutated images parsed", parsed);
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------
int mode_unit(const std::string & dir, const std::string & tool, const std::vector<std::string> & extra, const std::string & self) {
    test_sha256();
    test_fnv();
    test_synth();
    const std::string tmp = dir + "/tmp_" + self.substr(self.find_last_of('/') + 1);   // per driver binary
    ::mkdir(tmp.c_str(), 0755);
    Img img, nobase, big;
    if (load_img(&img, dir + "/elf_img.elf") && load_img(&nobase, dir + "/elf_img_no_base.elf") &&
        load_img(&big, dir + "/elf_img_big_text.elf")) {
        test_layout(img);
        test_stamp_and_tamper(img, tmp);
        test_corrupt(img.b);
        Img st;
        if (load_img(&st, tmp + "/stamped.elf")) {
            test_validation(st, nobase, big);
            tsr_elf_free(st.e);
        }
        test_cli(tool, dir, tmp);
    }
    tsr_elf_free(img.e);
    tsr_elf_free(nobase.e);
    tsr_elf_free(big.e);
    for (const std::string & p : extra) {   // two implementations agree on other images
        Bytes b;
        if (!read_file(p, &b)) {
            std::printf("fw_elf: %s not present, not compared\n", p.c_str());
            continue;
        }
        tsr_elf * e = parse(b, p.c_str());
        if (!e) continue;
        CHECK(tsr_elf_id(e) == trace_id(b, p.c_str()), "%s: tsr-elf and tsr_trace.c ids differ", p.c_str());
        check_sim_reader(p, e);
        std::printf("fw_elf: %s id %s (both implementations)\n", p.c_str(), hex64(tsr_elf_id(e)).c_str());
        tsr_elf_free(e);
    }
    return 0;
}

int mode_dump(const std::string & dir) {
    const std::vector<Bytes> cases = synth_cases();
    for (size_t i = 0; i < cases.size(); ++i) {
        const std::string p = dir + "/syn" + std::to_string(i) + ".elf";
        CHECK(write_file(p, cases[i]), "write %s", p.c_str());
    }
    return 0;
}

// has: a kernel that must be in the table after the base kernels, as k_probe.cpp declares it
// (nargs 2, ARGBLK, 4 lines); only: the same, and the only kernel. kinds: "all" (every kind with a
// revision) or kind names "NOP,JOIN,MUL_MAT": .tsr_kinds holds exactly those at kind_rev. tool:
// tsr-elf's `kinds` and `note` agree with the library.
int mode_image(const std::string & path, bool base, const std::string & has, bool only, const std::string & kinds,
               const std::string & tool) {
    Bytes b;
    if (!read_file(path, &b)) {
        CHECK(false, "%s: the image was not built", path.c_str());
        return 1;
    }
    tsr_elf * e = parse(b, path.c_str());
    if (!e) return 1;
    const CheckResult r = run_check(e, TSR_ELF_CHECK_ID | (base ? TSR_ELF_CHECK_BASE : 0u));
    CHECK(r.nerr == 0, "%s: tsr-elf checks:\n%s", path.c_str(), r.report.c_str());
    tsr_elf_sizes z;
    tsr_elf_get_sizes(e, nullptr, &z);
    CHECK(!z.over_itcm && !z.over_dtcm, "%s: SW §7.8 budget: ITCM %u / %u B, DTCM %u / %u B", path.c_str(), z.itcm_used,
          z.budget_itcm, z.dtcm_used, z.budget_dtcm);
    CHECK(tsr_elf_id(e) == trace_id(b, path.c_str()), "%s: tsr-elf and tsr_trace.c ids differ", path.c_str());
    check_sim_reader(path, e);
    if (base) {   // fix W2a: freestanding SDK: no static constructor (crt0 runs .init_array; the SDK has none)
        uint32_t i0 = 0, i1 = 0;
        CHECK(tsr_elf_symbol(e, "__init_array_start", &i0) == 0 && tsr_elf_symbol(e, "__init_array_end", &i1) == 0 && i0 == i1,
              "%s: .init_array is empty (%u bytes of static constructors)", path.c_str(), i1 - i0);
    }
    if (!has.empty()) {
        const int k = tsr_elf_kernel_find(e, has.c_str());
        const tsr_elf_kernel * d = k >= 0 ? tsr_elf_kernel_at(e, uint32_t(k)) : nullptr;
        CHECK(d && d->nargs == 2 && d->flags == TSR_ELF_KFLAG_ARGBLK && d->argblk_lines == 4, "%s: kernel %s as declared", path.c_str(),
              has.c_str());
        CHECK(!base || k >= 3, "%s: %s after the base kernels", path.c_str(), has.c_str());
        CHECK(!only || (k == 0 && tsr_elf_n_kernels(e) == 1), "%s: %s is the only kernel", path.c_str(), has.c_str());
    }
    if (!kinds.empty()) {   // SW §6.9: the kind table
        uint8_t rev[TSR_ELF_N_KINDS];
        char why[96];
        const int nk = tsr_elf_kinds(e, rev, why, sizeof(why));
        CHECK(nk >= 0, "%s: .tsr_kinds: %s", path.c_str(), why);
        std::string want = "," + kinds + ",";
        int n_want = 0;
        for (uint32_t k = 0; k < TSR_ELF_N_KINDS; ++k) {
            const bool in = kinds == "all" ? tsr::opl::kind_rev(k) != 0
                                           : want.find("," + std::string(tsr::opl::kind_name(k)) + ",") != std::string::npos;
            n_want += in;
            CHECK(rev[k] == (in ? tsr::opl::kind_rev(k) : 0u), "%s: kind 0x%02x %s: revision %u in .tsr_kinds, expected %u",
                  path.c_str(), k, tsr::opl::kind_name(k), rev[k], in ? tsr::opl::kind_rev(k) : 0u);
        }
        CHECK(nk == n_want, "%s: %d kinds, expected %d", path.c_str(), nk, n_want);
        if (!tool.empty() && nk >= 0) {
            std::string lines;
            for (uint32_t k = 0; k < TSR_ELF_N_KINDS; ++k) {
                if (!rev[k]) continue;
                char ln[64];
                std::snprintf(ln, sizeof(ln), "0x%02x\t%s\trev=%u\n", k, tsr::opl::kind_name(k), rev[k]);
                lines += ln;
            }
            char sig[32];
            std::snprintf(sig, sizeof(sig), "signature=0x%04x\n", tsr::opl::kinds_sig(rev));
            Run cli = run(tool + " kinds " + path);
            CHECK(cli.rc == 0 && cli.out == lines + sig, "%s: tsr-elf kinds:\n%s", path.c_str(), cli.out.c_str());
            cli = run(tool + " note " + path);
            CHECK(cli.rc == 0 && cli.out.find("\ncap_min=" + cap_min_str() + "\n") != std::string::npos &&
                      cli.out.find("\nbasis_version=" TSR_BASIS_VERSION_STRING "\n") != std::string::npos,
                  "%s: tsr-elf note:\n%s", path.c_str(), cli.out.c_str());
            std::printf("%s: %d kinds, %s", path.c_str(), nk, sig);
        }
    }
    std::vector<char> rep(1 << 16);
    tsr_elf_size_report(e, nullptr, rep.data(), rep.size());
    std::printf("%s", rep.data());
    tsr_elf_free(e);
    return 0;
}

// One boot: cold load from tsr-elf's segments (EX §3.3: padded words, receive CRC, VERIFY),
// RV_BOOT = 0, RV_RUN, until the first record. The record in r; false if none came.
bool hello_boot(const tsr_elf * e, const char * sched, uint32_t r[8]) {
    tsr_config cfg = TSR_CONFIG_U200;
    cfg.num_ch = 2;
    cfg.ch_bytes = 1ull << 30;
    tsr_sim_opts o;
    std::memset(&o, 0, sizeof(o));
    o.sched = sched;
    o.race = 2;   // fatal
    tsr_sim * sim = tsr_sim_create_opts(&cfg, &o);
    CHECK(sim != nullptr, "create the simulator (%s)", sched);
    if (!sim) return false;
    for (uint32_t s = 0; s < tsr_elf_nseg(e); ++s) {
        const tsr_elf_seg * g = tsr_elf_segment(e, s);
        if (g->padded == 0) continue;
        Bytes p(g->padded, 0);
        std::memcpy(p.data(), g->data, g->filesz);
        const uint32_t crc = crc32_zlib(p.data(), p.size());
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        for (uint32_t i = 0; i < g->padded; i += 4) tsr_sim_csr_write(sim, TSR_CSR_LOAD_DATA, rd32(&p[i]));
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc, "segment 0x%08x: receive CRC", g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_ADDR, g->paddr);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_COUNT, g->padded / 4u);
        tsr_sim_csr_write(sim, TSR_CSR_LOAD_CTRL, TSR_LOAD_CTRL_VERIFY);
        for (int i = 0; i < 1000 && (tsr_sim_csr_read(sim, TSR_CSR_LOAD_STATUS) & TSR_LOAD_STATUS_VERIFY_BUSY); ++i)
            tsr_sim_advance(sim, 1000);
        CHECK(tsr_sim_csr_read(sim, TSR_CSR_LOAD_CRC) == crc, "segment 0x%08x: VERIFY CRC", g->paddr);
    }
    const uint32_t st = tsr_sim_csr_read(sim, TSR_CSR_STATUS);
    CHECK(!(st & (TSR_STATUS_LOAD_REFUSED | TSR_STATUS_LOAD_TIMEOUT)), "loader status 0x%08x", st);
    tsr_sim_csr_write(sim, TSR_CSR_RV_BOOT, 0u);
    tsr_sim_csr_write(sim, TSR_CSR_CTRL, TSR_CTRL_RV_RUN);
    for (int it = 0; it < 2000 && tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) == 0; ++it) {
        tsr_sim_advance(sim, 100000);
        const uint32_t rt = tsr_sim_csr_read(sim, TSR_CSR_RT_STATE);
        if (TSR_GET(rt, TSR_RT_STATE_PHASE) == TSR_RT_STATE_PHASE_TRAPPED) {
            CHECK(false, "%s: the firmware trapped: RT_INFO 0x%08x, pc 0x%08x", sched, tsr_sim_csr_read(sim, TSR_CSR_RT_INFO),
                  tsr_sim_rv_pc(sim));
            break;
        }
    }
    const bool got = tsr_sim_csr_read(sim, TSR_CSR_LCQ_WR) >= 1;
    CHECK(got, "%s: a record after boot", sched);
    for (uint32_t j = 0; j < 8; ++j) r[j] = tsr_sim_csr_read(sim, TSR_CSR_LCQ_REC(0, j));
    tsr_sim_destroy(sim);
    return got;
}

// HELLO under the fixed and 7 random schedules with the race detector fatal (SW §4.7, §10.2): the
// schedule-independent words (r0, the id r2 / r3, r4, r5) are the same in every run.
int mode_hello(const std::string & path) {
    Bytes b;
    if (!read_file(path, &b)) {
        CHECK(false, "%s: the image was not built", path.c_str());
        return 1;
    }
    tsr_elf * e = parse(b, path.c_str());
    if (!e) return 1;
    const uint64_t id = tsr_elf_id(e);
    uint64_t stored = 0;
    tsr_elf_note n;
    std::memset(&n, 0, sizeof(n));
    CHECK(tsr_elf_stored_id(e, &stored) == 0 && stored == id, "the image is stamped");
    CHECK(tsr_elf_get_note(e, &n) == 0, "the image has its note");
    uint8_t rev[TSR_ELF_N_KINDS];
    CHECK(tsr_elf_kinds(e, rev, nullptr, 0) >= 0, "the image's kind table is well formed");
    const uint32_t sig = tsr::opl::kinds_sig(rev);
    static const char * const scheds[8] = { "fixed", "random:1", "random:2", "random:3", "random:4", "random:5", "random:6", "random:7" };
    uint32_t first[8] = { 0 };
    for (int k = 0; k < 8; ++k) {
        uint32_t r[8];
        if (!hello_boot(e, scheds[k], r)) continue;
        CHECK(TSR_GET(r[0], TSR_LCQ_R0_TYPE) == TSR_LCQ_R0_TYPE_HELLO, "%s: first record HELLO (r0 0x%08x)", scheds[k], r[0]);
        CHECK(r[2] == uint32_t(id) && r[3] == uint32_t(id >> 32), "%s: HELLO id %08x%08x, tsr-elf %016llx", scheds[k], r[3], r[2],
              (unsigned long long)id);
        CHECK(TSR_GET(r[4], TSR_HELLO_R4_SDK_ABI) == n.sdk_abi, "%s: HELLO Basis ABI %u, note %u", scheds[k],
              TSR_GET(r[4], TSR_HELLO_R4_SDK_ABI), n.sdk_abi);
        CHECK(r[5] == tsr_elf_n_kernels(e) && r[5] == n.n_kernels, "%s: HELLO kernel count %u, table %u", scheds[k], r[5],
              tsr_elf_n_kernels(e));
        // SW §6.9: r4[31:16] the signature of the kind table the host reads from the ELF, r6 the note's
        // Tesseract Basis version.
        CHECK(TSR_GET(r[4], TSR_HELLO_R4_KINDS_SIG) == sig, "%s: HELLO kind table signature 0x%04x, .tsr_kinds 0x%04x", scheds[k],
              TSR_GET(r[4], TSR_HELLO_R4_KINDS_SIG), sig);
        CHECK(r[6] == n.basis_version && r[6] == TSR_BASIS_VERSION, "%s: HELLO r6 0x%08x, note basis_version 0x%08x", scheds[k],
              r[6], n.basis_version);
        if (k == 0) {
            std::memcpy(first, r, sizeof(first));
            std::printf("fw_elf_hello: %s HELLO id %08x%08x, Basis ABI %u, kind table 0x%04x, %u kernels, Tesseract Basis %u.%u.%u\n",
                        path.c_str(), r[3], r[2], TSR_GET(r[4], TSR_HELLO_R4_SDK_ABI), TSR_GET(r[4], TSR_HELLO_R4_KINDS_SIG), r[5],
                        TSR_GET(r[6], TSR_HELLO_R6_MAJOR), TSR_GET(r[6], TSR_HELLO_R6_MINOR), TSR_GET(r[6], TSR_HELLO_R6_PATCH));
        } else {
            CHECK(r[0] == first[0] && r[2] == first[2] && r[3] == first[3] && r[4] == first[4] && r[5] == first[5] &&
                      r[6] == first[6],
                  "%s: HELLO differs from the fixed schedule's", scheds[k]);
        }
    }
    tsr_elf_free(e);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "unit" && argc >= 4) {
        mode_unit(argv[2], argv[3], std::vector<std::string>(argv + 4, argv + argc), argv[0]);
    } else if (mode == "dump" && argc == 3) {
        mode_dump(argv[2]);
    } else if (mode == "image" && argc >= 3) {
        bool base = false, only = false;
        std::string has, kinds, tool;
        for (int i = 3; i < argc; ++i) {
            if (std::strcmp(argv[i], "--base") == 0) base = true;
            else if ((std::strcmp(argv[i], "--has") == 0 || std::strcmp(argv[i], "--only") == 0) && i + 1 < argc) {
                only = std::strcmp(argv[i], "--only") == 0;
                has = argv[++i];
            } else if (std::strcmp(argv[i], "--kinds") == 0 && i + 1 < argc) {
                kinds = argv[++i];
            } else if (std::strcmp(argv[i], "--tool") == 0 && i + 1 < argc) {
                tool = argv[++i];
            }
        }
        mode_image(argv[2], base, has, only, kinds, tool);
    } else if (mode == "hello" && argc == 3) {
        mode_hello(argv[2]);
    } else {
        std::fprintf(stderr, "usage: test_fw_elf unit <elf dir> <tsr-elf> [elf ...] | dump <dir> | image <elf> [--base] [--has K|--only K] "
                     "[--kinds all|K,...] [--tool T] | hello <elf>\n");
        return 2;
    }
    std::printf("fw_elf %s: %d checks, %d failed\n", mode.c_str(), g_checks, g_fail);
    return g_fail ? 1 : 0;
}

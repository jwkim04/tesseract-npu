// tsr_elf.cpp — firmware images on the host (tsr_elf.h; SW §6.2, §9.5; EX §3.1-§3.2; P2.1).
//
// Plain C++17 with the C API of tsr_elf.h. Everything is read from a private copy of the file with
// explicit bounds checks; a malformed file fails tsr_elf_parse with a message, never a crash.
#include "tsr_elf.h"

#include "tsr_tables.h"   // TSR_ITCM_BYTES, TSR_DTCM_BYTES, TSR_RV_DTCM_BASE, TSR_LOGFMT_VMA
#include "tsr/opl.hpp"    // N_KINDS (the kind table, SW §6.9)

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Written for this tool; sw/sim/tsr_trace.c has its own, and fw_elf
// compares the ids both compute.
// ---------------------------------------------------------------------------
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        static const uint32_t iv[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
        std::memcpy(h_, iv, sizeof(iv));
        total_ = 0;
        used_ = 0;
    }
    void update(const void * p, size_t n) {
        if (n == 0) return;
        const uint8_t * b = static_cast<const uint8_t *>(p);
        total_ += n;
        if (used_ > 0) {
            const size_t k = std::min(n, sizeof(blk_) - used_);
            std::memcpy(blk_ + used_, b, k);
            used_ += k; b += k; n -= k;
            if (used_ < sizeof(blk_)) return;
            compress(blk_);
            used_ = 0;
        }
        for (; n >= 64; b += 64, n -= 64) compress(b);
        std::memcpy(blk_, b, n);
        used_ = n;
    }
    void final(uint8_t out[32]) {
        const uint64_t bits = total_ * 8u;
        uint8_t pad[72] = { 0x80 };
        const size_t padn = (used_ < 56 ? 56 - used_ : 120 - used_);
        for (int i = 0; i < 8; ++i) pad[padn + i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        update(pad, padn + 8);
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<uint8_t>(h_[i] >> (24 - 8 * j));
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void compress(const uint8_t * p) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[16];
        for (int t = 0; t < 16; ++t)
            w[t] = uint32_t(p[4 * t]) << 24 | uint32_t(p[4 * t + 1]) << 16 | uint32_t(p[4 * t + 2]) << 8 | p[4 * t + 3];
        uint32_t v[8];
        std::memcpy(v, h_, sizeof(v));
        for (int t = 0; t < 64; ++t) {
            uint32_t wt;
            if (t < 16) {
                wt = w[t];
            } else {   // message schedule in a 16-word ring
                const uint32_t a = w[(t + 1) & 15], b = w[(t + 14) & 15];
                wt = w[t & 15] += (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[(t + 9) & 15] +
                                  (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
            }
            const uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            const uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const uint32_t t1 = v[7] + s1 + ch + K[t] + wt;
            const uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            const uint32_t mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            std::memmove(v + 1, v, 7 * sizeof(uint32_t));
            v[4] += t1;
            v[0] = t1 + s0 + mj;
        }
        for (int i = 0; i < 8; ++i) h_[i] += v[i];
    }

    uint32_t h_[8];
    uint8_t  blk_[64];
    uint64_t total_;
    size_t   used_;
};

// ---------------------------------------------------------------------------
// ELF32 constants (the subset used)
// ---------------------------------------------------------------------------
constexpr uint16_t ET_EXEC_    = 2;
constexpr uint16_t EM_RISCV_   = 243;
constexpr uint32_t PT_LOAD_    = 1;
constexpr uint32_t PF_X_       = 1;
constexpr uint32_t SHT_SYMTAB_ = 2;
constexpr uint32_t SHT_NOTE_   = 7;
constexpr uint32_t SHT_NOBITS_ = 8;
constexpr uint32_t SHF_ALLOC_  = 2;

uint16_t rd16(const uint8_t * p) { return uint16_t(p[0] | p[1] << 8); }
uint32_t rd32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }

void fmt_to(char * err, size_t errlen, const char * fmt, ...) __attribute__((format(printf, 3, 4)));
void fmt_to(char * err, size_t errlen, const char * fmt, ...) {
    if (err == nullptr || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

std::string strf(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char * fmt, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    return b;
}

struct Section {
    std::string name;
    uint32_t    type, flags, addr, off, size;
    bool alloc() const { return (flags & SHF_ALLOC_) != 0; }
    bool has_data() const { return type != SHT_NOBITS_; }
};

struct Seg {
    tsr_elf_seg pub;
    uint32_t    vaddr, off;
};

struct NoteFound {
    uint32_t     namesz, descsz, type;
    tsr_elf_note desc;
    bool         desc_ok;   // descsz = 32
};

bool in_itcm(uint64_t lo, uint64_t hi, uint32_t itcm) { return hi <= itcm && lo <= hi; }
bool in_dtcm(uint64_t lo, uint64_t hi, uint32_t dtcm) {
    return lo >= TSR_RV_DTCM_BASE && hi <= uint64_t(TSR_RV_DTCM_BASE) + dtcm && lo <= hi;
}

tsr_elf_limits norm_limits(const tsr_elf_limits * lim) {
    tsr_elf_limits l = { 0, 0, 0, 0 };
    if (lim) l = *lim;
    if (l.itcm_bytes == 0) l.itcm_bytes = TSR_ITCM_BYTES;
    if (l.dtcm_bytes == 0) l.dtcm_bytes = TSR_DTCM_BYTES;
    if (l.budget_itcm == 0) l.budget_itcm = TSR_ELF_BUDGET_ITCM;
    if (l.budget_dtcm == 0) l.budget_dtcm = TSR_ELF_BUDGET_DTCM;
    return l;
}

bool is_ident_char(char c, bool first) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || (!first && c >= '0' && c <= '9');
}

}  // namespace

struct tsr_elf {
    std::vector<uint8_t>        buf;
    uint32_t                    entry = 0;
    std::vector<Seg>            segs;        // ascending paddr
    std::vector<tsr_elf_seg>    pubsegs;
    std::vector<Section>        secs;
    int                         id_sec = -1, kt_sec = -1, kinds_sec = -1, logfmt_sec = -1;
    std::vector<tsr_elf_kernel> kernels;
    std::vector<NoteFound>      notes;       // every note with owner "TSR"
    std::vector<std::pair<uint32_t, uint32_t>> logfmt;   // (address, file offset) per string
    uint32_t sym_off = 0, sym_num = 0, str_off = 0, str_size = 0;
    bool     has_symtab = false;

    int find_sec(const char * name) const {
        for (size_t i = 0; i < secs.size(); ++i)
            if (secs[i].name == name) return int(i);
        return -1;
    }
    // The PT_LOAD whose file data covers [addr, addr + n) (by virtual address), or -1.
    int seg_with_data(uint32_t addr, uint32_t n) const {
        for (size_t i = 0; i < segs.size(); ++i) {
            const Seg & s = segs[i];
            if (uint64_t(addr) >= s.vaddr && uint64_t(addr) + n <= uint64_t(s.vaddr) + s.pub.filesz) return int(i);
        }
        return -1;
    }
};

extern "C" {

uint32_t tsr_elf_fnv1a(const char * s) {
    uint32_t h = 0x811C9DC5u;
    for (; *s; ++s) {
        h ^= uint8_t(*s);
        h *= 0x01000193u;
    }
    return h;
}

void tsr_elf_sha256(const void * data, size_t n, uint8_t out[32]) {
    Sha256 s;
    s.update(data, n);
    s.final(out);
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
static int parse_into(tsr_elf * e, char * err, size_t en) {
    const uint8_t * b = e->buf.data();
    const size_t    n = e->buf.size();
    if (n < 52 || std::memcmp(b, "\177ELF", 4) != 0) { fmt_to(err, en, "not an ELF file"); return -1; }
    if (b[4] != 1 || b[5] != 1) { fmt_to(err, en, "not an ELF32 little-endian file"); return -1; }
    if (rd16(b + 18) != EM_RISCV_) { fmt_to(err, en, "not a RISC-V ELF (e_machine %u)", rd16(b + 18)); return -1; }
    if (rd16(b + 16) != ET_EXEC_) {
        fmt_to(err, en, "not a static executable (e_type %u; EX §3.1: no relocations)", rd16(b + 16));
        return -1;
    }
    e->entry = rd32(b + 24);
    const uint64_t phoff = rd32(b + 28), shoff = rd32(b + 32);
    const uint32_t phentsize = rd16(b + 42), phnum = rd16(b + 44);
    const uint32_t shentsize = rd16(b + 46), shnum = rd16(b + 48), shstrndx = rd16(b + 50);

    if (phnum > 0 && (phentsize != 32 || phoff + 32ull * phnum > n)) { fmt_to(err, en, "bad program header table"); return -1; }
    for (uint32_t i = 0; i < phnum; ++i) {
        const uint8_t * ph = b + phoff + 32ull * i;
        if (rd32(ph) != PT_LOAD_) continue;
        Seg s{};
        s.off = rd32(ph + 4);
        s.vaddr = rd32(ph + 8);
        s.pub.paddr = rd32(ph + 12);
        s.pub.filesz = rd32(ph + 16);
        s.pub.memsz = rd32(ph + 20);
        s.pub.flags = rd32(ph + 24);
        if (uint64_t(s.off) + s.pub.filesz > n) { fmt_to(err, en, "PT_LOAD %u: file data beyond the end of the file", i); return -1; }
        if (s.pub.filesz > s.pub.memsz) { fmt_to(err, en, "PT_LOAD %u: p_filesz > p_memsz", i); return -1; }
        if (s.pub.filesz > 0xFFFFFFFCu) { fmt_to(err, en, "PT_LOAD %u: p_filesz too large", i); return -1; }
        s.pub.padded = (s.pub.filesz + 3u) & ~3u;
        e->segs.push_back(s);
    }
    std::stable_sort(e->segs.begin(), e->segs.end(), [](const Seg & x, const Seg & y) { return x.pub.paddr < y.pub.paddr; });

    if (shoff != 0 && shnum != 0) {
        if (shentsize != 40 || shoff + 40ull * shnum > n || shstrndx >= shnum) { fmt_to(err, en, "bad section header table"); return -1; }
        const uint8_t * ss = b + shoff + 40ull * shstrndx;
        const uint64_t  stroff = rd32(ss + 16), strsize = rd32(ss + 20);
        if (stroff + strsize > n) { fmt_to(err, en, "bad section name table"); return -1; }
        for (uint32_t i = 0; i < shnum; ++i) {
            const uint8_t * sh = b + shoff + 40ull * i;
            Section s;
            const uint32_t nm = rd32(sh);
            s.type = rd32(sh + 4);
            s.flags = rd32(sh + 8);
            s.addr = rd32(sh + 12);
            s.off = rd32(sh + 16);
            s.size = rd32(sh + 20);
            if (nm < strsize) {
                const char * p = reinterpret_cast<const char *>(b + stroff + nm);
                s.name.assign(p, strnlen(p, size_t(strsize - nm)));
            }
            if (i != 0 && s.has_data() && uint64_t(s.off) + s.size > n) {
                fmt_to(err, en, "section %s: data beyond the end of the file", s.name.c_str());
                return -1;
            }
            if (s.type == SHT_SYMTAB_) {
                const uint32_t link = rd32(sh + 24);
                if (link < shnum && rd32(sh + 36) == 16) {
                    const uint8_t * lk = b + shoff + 40ull * link;
                    const uint64_t  so = rd32(lk + 16), sz = rd32(lk + 20);
                    if (so + sz <= n) {
                        e->has_symtab = true;
                        e->sym_off = s.off;
                        e->sym_num = s.size / 16u;
                        e->str_off = uint32_t(so);
                        e->str_size = uint32_t(sz);
                    }
                }
            }
            e->secs.push_back(s);
        }
    }
    for (const Seg & s : e->segs) {
        tsr_elf_seg p = s.pub;
        p.data = b + s.off;
        e->pubsegs.push_back(p);
    }

    e->id_sec = e->find_sec(".tsr_id");
    e->kt_sec = e->find_sec(".tsr_kernels");
    e->kinds_sec = e->find_sec(".tsr_kinds");
    e->logfmt_sec = e->find_sec(".tsr_logfmt");

    // Kernel table: whole entries (tsr_elf_check reports a partial one).
    if (e->kt_sec >= 0 && e->secs[size_t(e->kt_sec)].has_data()) {
        const Section & k = e->secs[size_t(e->kt_sec)];
        for (uint32_t i = 0; i < k.size / TSR_ELF_KDESC_BYTES; ++i) {
            const uint8_t * p = b + k.off + i * TSR_ELF_KDESC_BYTES;
            tsr_elf_kernel d{};
            d.index = i;
            d.name_hash = rd32(p);
            d.entry = rd32(p + 4);
            d.nargs = p[8];
            d.flags = p[9];
            d.argblk_lines = rd16(p + 10);
            std::memcpy(d.name, p + 12, TSR_ELF_KNAME_BYTES);
            d.name[TSR_ELF_KNAME_BYTES] = '\0';
            e->kernels.push_back(d);
        }
    }

    // Notes with owner "TSR": every SHT_NOTE section, and a section named .note.tsr of any type.
    for (const Section & s : e->secs) {
        if (!(s.type == SHT_NOTE_ || s.name == ".note.tsr") || !s.has_data()) continue;
        uint64_t o = s.off;
        const uint64_t end = uint64_t(s.off) + s.size;
        while (o + 12 <= end) {
            NoteFound f{};
            f.namesz = rd32(b + o);
            f.descsz = rd32(b + o + 4);
            f.type = rd32(b + o + 8);
            const uint64_t name_o = o + 12, desc_o = name_o + ((uint64_t(f.namesz) + 3) & ~3ull);
            const uint64_t next = desc_o + ((uint64_t(f.descsz) + 3) & ~3ull);
            if (next > end) break;                       // truncated note: tsr_elf_check reports it
            if (f.namesz == 4 && std::memcmp(b + name_o, "TSR", 4) == 0) {
                f.desc_ok = f.descsz == TSR_ELF_NOTE_DESC_BYTES;
                if (f.desc_ok) {
                    const uint8_t * d = b + desc_o;
                    f.desc.sdk_abi = rd32(d);
                    f.desc.cap_min = rd32(d + 4);
                    f.desc.itcm_min = rd32(d + 8);
                    f.desc.dtcm_min = rd32(d + 12);
                    f.desc.caps_req = rd32(d + 16);
                    f.desc.n_kernels = rd32(d + 20);
                    f.desc.basis_version = rd32(d + 24);
                    f.desc.rsv = rd32(d + 28);
                }
                e->notes.push_back(f);
            }
            o = next;
        }
    }

    // Log-format table: the strings of .tsr_logfmt.
    if (e->logfmt_sec >= 0 && e->secs[size_t(e->logfmt_sec)].has_data()) {
        const Section & l = e->secs[size_t(e->logfmt_sec)];
        uint32_t i = 0;
        while (i < l.size) {
            if (b[l.off + i] == 0) { ++i; continue; }    // padding between strings
            e->logfmt.emplace_back(l.addr + i, l.off + i);
            while (i < l.size && b[l.off + i] != 0) ++i;
        }
    }
    return 0;
}

int tsr_elf_parse(const void * buf, size_t n, tsr_elf ** out, char * err, size_t errlen) {
    *out = nullptr;
    if (buf == nullptr || n > 0xFFFFFFFFu) { fmt_to(err, errlen, "bad buffer"); return -1; }
    tsr_elf * e = new tsr_elf;
    e->buf.assign(static_cast<const uint8_t *>(buf), static_cast<const uint8_t *>(buf) + n);
    if (parse_into(e, err, errlen) != 0) {
        delete e;
        return -1;
    }
    *out = e;
    return 0;
}

int tsr_elf_read(const char * path, tsr_elf ** out, char * err, size_t errlen) {
    *out = nullptr;
    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) { fmt_to(err, errlen, "%s: cannot open", path); return -1; }
    std::vector<uint8_t> b;
    uint8_t chunk[65536];
    size_t k;
    while ((k = std::fread(chunk, 1, sizeof(chunk), f)) > 0) b.insert(b.end(), chunk, chunk + k);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) { fmt_to(err, errlen, "%s: read error", path); return -1; }
    char e2[256];
    if (tsr_elf_parse(b.data(), b.size(), out, e2, sizeof(e2)) != 0) {
        fmt_to(err, errlen, "%s: %s", path, e2);
        return -1;
    }
    return 0;
}

void tsr_elf_free(tsr_elf * e) { delete e; }

const uint8_t * tsr_elf_bytes(const tsr_elf * e, size_t * n) {
    if (n) *n = e->buf.size();
    return e->buf.data();
}
uint32_t tsr_elf_entry(const tsr_elf * e) { return e->entry; }
uint32_t tsr_elf_nseg(const tsr_elf * e) { return uint32_t(e->pubsegs.size()); }
const tsr_elf_seg * tsr_elf_segment(const tsr_elf * e, uint32_t i) {
    return i < e->pubsegs.size() ? &e->pubsegs[i] : nullptr;
}

int tsr_elf_symbol(const tsr_elf * e, const char * name, uint32_t * value) {
    if (!e->has_symtab) return -1;
    const uint8_t * b = e->buf.data();
    for (uint32_t i = 1; i < e->sym_num; ++i) {
        const uint8_t * s = b + e->sym_off + 16ull * i;
        const uint32_t nm = rd32(s);
        if (nm >= e->str_size) continue;
        const char * p = reinterpret_cast<const char *>(b + e->str_off + nm);
        const size_t L = strnlen(p, e->str_size - nm);
        if (L == std::strlen(name) && std::memcmp(p, name, L) == 0 && rd16(s + 14) != 0) {   // defined
            *value = rd32(s + 4);
            return 0;
        }
    }
    return -1;
}

int tsr_elf_section(const tsr_elf * e, const char * name, uint32_t * addr, uint32_t * off, uint32_t * size,
                    uint32_t * type) {
    const int i = e->find_sec(name);
    if (i < 0) return -1;
    const Section & s = e->secs[size_t(i)];
    if (addr) *addr = s.addr;
    if (off) *off = s.has_data() ? s.off : 0;
    if (size) *size = s.size;
    if (type) *type = s.type;
    return 0;
}

// ---------------------------------------------------------------------------
// Image id (EX §3.2)
// ---------------------------------------------------------------------------
uint64_t tsr_elf_id(const tsr_elf * e) {
    // The bytes of .tsr_id, by address (taken as zero wherever a segment's file data covers them).
    uint64_t id_lo = 1, id_hi = 0;   // empty
    if (e->id_sec >= 0) {
        const Section & s = e->secs[size_t(e->id_sec)];
        id_lo = s.addr;
        id_hi = uint64_t(s.addr) + 8u;
    }
    Sha256 h;
    std::vector<uint8_t> tmp;
    for (const Seg & s : e->segs) {
        uint8_t hdr[8];
        for (int i = 0; i < 4; ++i) {
            hdr[i] = uint8_t(s.pub.paddr >> (8 * i));
            hdr[4 + i] = uint8_t(s.pub.padded >> (8 * i));
        }
        h.update(hdr, 8);
        if (s.pub.padded == 0) continue;
        tmp.assign(s.pub.padded, 0);
        std::memcpy(tmp.data(), e->buf.data() + s.off, s.pub.filesz);
        for (uint64_t a = std::max<uint64_t>(id_lo, s.vaddr); a < id_hi && a < uint64_t(s.vaddr) + s.pub.filesz; ++a)
            tmp[size_t(a - s.vaddr)] = 0;
        h.update(tmp.data(), tmp.size());
    }
    uint8_t d[32];
    h.final(d);
    uint64_t id = 0;
    for (int i = 0; i < 8; ++i) id |= uint64_t(d[i]) << (8 * i);
    return id;
}

static int id_file_offset(const tsr_elf * e, size_t * off) {
    if (e->id_sec < 0) return -1;
    const Section & s = e->secs[size_t(e->id_sec)];
    if (!s.has_data() || s.size != 8) return -1;
    *off = s.off;
    return 0;
}

int tsr_elf_stored_id(const tsr_elf * e, uint64_t * id) {
    size_t off;
    if (id_file_offset(e, &off) != 0) return -1;
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(e->buf[off + size_t(i)]) << (8 * i);
    *id = v;
    return 0;
}

int tsr_elf_set_id(tsr_elf * e, uint64_t id) {
    size_t off;
    if (id_file_offset(e, &off) != 0) return -1;
    for (int i = 0; i < 8; ++i) e->buf[off + size_t(i)] = uint8_t(id >> (8 * i));
    return 0;
}

int tsr_elf_write(const tsr_elf * e, const char * path, char * err, size_t errlen) {
    const std::string tmp = std::string(path) + ".tmp" + std::to_string(long(getpid()));
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { fmt_to(err, errlen, "%s: cannot create", tmp.c_str()); return -1; }
    const bool ok = std::fwrite(e->buf.data(), 1, e->buf.size(), f) == e->buf.size();
    if (std::fclose(f) != 0 || !ok) {
        std::remove(tmp.c_str());
        fmt_to(err, errlen, "%s: write error", tmp.c_str());
        return -1;
    }
    if (std::rename(tmp.c_str(), path) != 0) {
        std::remove(tmp.c_str());
        fmt_to(err, errlen, "%s: cannot rename to %s", tmp.c_str(), path);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel table, note, strings
// ---------------------------------------------------------------------------
uint32_t tsr_elf_n_kernels(const tsr_elf * e) { return uint32_t(e->kernels.size()); }
const tsr_elf_kernel * tsr_elf_kernel_at(const tsr_elf * e, uint32_t index) {
    return index < e->kernels.size() ? &e->kernels[index] : nullptr;
}
int tsr_elf_kernel_find(const tsr_elf * e, const char * name) {
    for (const tsr_elf_kernel & k : e->kernels)
        if (std::strcmp(k.name, name) == 0) return int(k.index);
    return -1;
}

int tsr_elf_get_note(const tsr_elf * e, tsr_elf_note * note) {
    for (const NoteFound & f : e->notes)
        if (f.type == TSR_ELF_NOTE_TYPE && f.desc_ok) {
            *note = f.desc;
            return 0;
        }
    return -1;
}

int tsr_elf_kinds(const tsr_elf * e, uint8_t rev[TSR_ELF_N_KINDS], char * why, size_t whylen) {
    static_assert(TSR_ELF_N_KINDS == tsr::opl::N_KINDS, "kind table size");
    std::memset(rev, 0, TSR_ELF_N_KINDS);
    if (why && whylen) why[0] = '\0';
    if (e->kinds_sec < 0) return 0;                      // no TSR_KIND in the image: no kinds
    const Section & s = e->secs[size_t(e->kinds_sec)];
    if (!s.has_data() || s.size == 0) return 0;
    if (s.size % TSR_ELF_KIND_BYTES != 0) {
        fmt_to(why, whylen, "size %u not a multiple of %u", s.size, TSR_ELF_KIND_BYTES);
        return -1;
    }
    const uint8_t * p = e->buf.data() + s.off;
    int n = 0;
    for (uint32_t i = 0; i < s.size / TSR_ELF_KIND_BYTES; ++i, p += TSR_ELF_KIND_BYTES) {
        const uint32_t kind = p[0], r = p[1], rsvd = rd16(p + 2);
        if (kind >= TSR_ELF_N_KINDS) { fmt_to(why, whylen, "kind 0x%02x out of range", kind); return -1; }
        if (rev[kind] != 0) { fmt_to(why, whylen, "kind 0x%02x twice", kind); return -1; }
        if (r == 0) { fmt_to(why, whylen, "kind 0x%02x revision 0", kind); return -1; }
        if (rsvd != 0) { fmt_to(why, whylen, "kind 0x%02x reserved bits 0x%04x", kind, rsvd); return -1; }
        rev[kind] = uint8_t(r);
        ++n;
    }
    return n;
}

const char * tsr_elf_logfmt(const tsr_elf * e, uint32_t addr) {
    if (e->logfmt_sec < 0) return nullptr;
    const Section & s = e->secs[size_t(e->logfmt_sec)];
    if (!s.has_data() || addr < s.addr || uint64_t(addr) >= uint64_t(s.addr) + s.size) return nullptr;
    const char * p = reinterpret_cast<const char *>(e->buf.data() + s.off + (addr - s.addr));
    const size_t room = size_t(uint64_t(s.addr) + s.size - addr);
    return strnlen(p, room) < room ? p : nullptr;
}

uint32_t tsr_elf_logfmt_count(const tsr_elf * e) { return uint32_t(e->logfmt.size()); }
const char * tsr_elf_logfmt_at(const tsr_elf * e, uint32_t i, uint32_t * addr) {
    if (i >= e->logfmt.size()) return nullptr;
    if (addr) *addr = e->logfmt[i].first;
    return tsr_elf_logfmt(e, e->logfmt[i].first);
}

const char * tsr_elf_string(const tsr_elf * e, uint32_t addr) {
    for (const Section & s : e->secs) {
        if (!s.alloc() || !s.has_data() || s.size == 0 || e->seg_with_data(s.addr, s.size) < 0) continue;
        if (addr < s.addr || uint64_t(addr) >= uint64_t(s.addr) + s.size) continue;
        const char * p = reinterpret_cast<const char *>(e->buf.data() + s.off + (addr - s.addr));
        const size_t room = size_t(uint64_t(s.addr) + s.size - addr);
        return strnlen(p, room) < room ? p : nullptr;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------
void tsr_elf_get_sizes(const tsr_elf * e, const tsr_elf_limits * lim, tsr_elf_sizes * z) {
    const tsr_elf_limits l = norm_limits(lim);
    std::memset(z, 0, sizeof(*z));
    z->itcm_bytes = l.itcm_bytes;
    z->dtcm_bytes = l.dtcm_bytes;
    z->budget_itcm = l.budget_itcm;
    z->budget_dtcm = l.budget_dtcm;
    // Allocated ranges per region: from the section headers (every allocated section, loaded or
    // not), else from the PT_LOADs (memsz).
    std::vector<std::pair<uint64_t, uint64_t>> dr;   // DTCM [lo, hi)
    bool have_alloc_secs = false;
    for (const Section & s : e->secs) {
        if (!s.alloc() || s.size == 0) continue;
        have_alloc_secs = true;
        const uint64_t lo = s.addr, hi = uint64_t(s.addr) + s.size;
        if (in_itcm(lo, hi, 0x00100000u)) {
            z->itcm_used = uint32_t(std::max<uint64_t>(z->itcm_used, hi));
        } else if (in_dtcm(lo, hi, 0x00100000u)) {
            dr.emplace_back(lo, hi);
            if (!s.has_data()) z->dtcm_zeroed += s.size;
        }
    }
    for (const Seg & s : e->segs) {
        const uint64_t lo = s.pub.paddr, hi = uint64_t(s.pub.paddr) + s.pub.memsz;
        if (in_dtcm(lo, hi, 0x00100000u)) {
            z->dtcm_loaded += s.pub.padded;
            if (!have_alloc_secs && hi > lo) {
                dr.emplace_back(lo, hi);
                z->dtcm_zeroed += s.pub.memsz - s.pub.filesz;
            }
        } else if (!have_alloc_secs && in_itcm(lo, hi, 0x00100000u)) {
            z->itcm_used = uint32_t(std::max<uint64_t>(z->itcm_used, hi));
        }
    }
    // DTCM used = the extent from the DTCM base minus the largest hole: alignment padding between
    // sections counts as used, the free space between .bss and the stacks at the top does not.
    if (!dr.empty()) {
        std::sort(dr.begin(), dr.end());
        uint64_t end = TSR_RV_DTCM_BASE, hole = 0;
        for (const auto & r : dr) {
            if (r.first > end) hole = std::max(hole, r.first - end);
            end = std::max(end, r.second);
        }
        z->dtcm_used = uint32_t(end - TSR_RV_DTCM_BASE - hole);
    }
    if (e->logfmt_sec >= 0) z->logfmt = e->secs[size_t(e->logfmt_sec)].size;
    z->over_itcm = z->itcm_used > z->budget_itcm;
    z->over_dtcm = z->dtcm_used > z->budget_dtcm;
}

void tsr_elf_size_report(const tsr_elf * e, const tsr_elf_limits * lim, char * out, size_t outlen) {
    tsr_elf_sizes z;
    tsr_elf_get_sizes(e, lim, &z);
    std::string r;
    uint64_t stored = 0;
    const bool has_id = tsr_elf_stored_id(e, &stored) == 0;
    const uint64_t id = tsr_elf_id(e);
    r += strf("image id   %016llx%s\n", (unsigned long long)id,
              !has_id ? " (no .tsr_id)" : stored == id ? " (stamped)" : stored == 0 ? " (not stamped)" : " (STORED ID DIFFERS)");
    auto region = [&](const char * name, uint32_t used, uint32_t size, uint32_t budget, bool itcm) {
        r += strf("%-10s %7u B of %7u B (%5.1f %%), SW §7.8 budget %u B: %s\n", name, used, size,
                  size ? 100.0 * used / size : 0.0, budget, used > budget ? "EXCEEDED" : "ok");
        for (const Section & s : e->secs) {
            if (!s.alloc() || s.size == 0) continue;
            const uint64_t lo = s.addr, hi = uint64_t(s.addr) + s.size;
            if (itcm ? !in_itcm(lo, hi, 0x00100000u) : !in_dtcm(lo, hi, 0x00100000u)) continue;
            r += strf("  %-14s 0x%08x %7u B%s\n", s.name.c_str(), s.addr, s.size, s.has_data() ? "" : "  (not loaded)");
        }
    };
    region("ITCM", z.itcm_used, z.itcm_bytes, z.budget_itcm, true);
    region("DTCM", z.dtcm_used, z.dtcm_bytes, z.budget_dtcm, false);
    r += strf("DTCM loaded %u B, zeroed by crt0 or stack %u B\n", z.dtcm_loaded, z.dtcm_zeroed);
    if (e->logfmt_sec >= 0)
        r += strf(".tsr_logfmt %u B at 0x%08x, %u strings (file only)\n", z.logfmt, e->secs[size_t(e->logfmt_sec)].addr,
                  tsr_elf_logfmt_count(e));
    r += strf("kernels    %u\n", tsr_elf_n_kernels(e));
    for (const tsr_elf_kernel & k : e->kernels)
        r += strf("  %3u %-24s entry 0x%08x nargs %u flags 0x%02x argblk_lines %u\n", k.index, k.name, k.entry, k.nargs,
                  k.flags, k.argblk_lines);
    uint8_t rev[TSR_ELF_N_KINDS];
    const int nk = tsr_elf_kinds(e, rev, nullptr, 0);
    r += nk < 0 ? std::string("kinds      (malformed .tsr_kinds)\n") : strf("kinds      %d\n", nk);
    for (uint32_t k = 0; nk > 0 && k < TSR_ELF_N_KINDS; ++k)
        if (rev[k]) r += strf("  0x%02x %-24s rev %u\n", k, tsr::opl::kind_name(k), rev[k]);
    tsr_elf_note nt;
    if (tsr_elf_get_note(e, &nt) == 0)
        r += strf("note.tsr   sdk_abi %u cap_min %u.%u itcm_min %u dtcm_min %u caps_req 0x%08x n_kernels %u basis_version %u.%u.%u\n",
                  nt.sdk_abi, nt.cap_min >> 8, nt.cap_min & 0xFFu, nt.itcm_min, nt.dtcm_min, nt.caps_req, nt.n_kernels,
                  (nt.basis_version >> 16) & 0xFFu, (nt.basis_version >> 8) & 0xFFu, nt.basis_version & 0xFFu);
    if (out && outlen) {
        std::strncpy(out, r.c_str(), outlen - 1);
        out[outlen - 1] = '\0';
    }
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------
uint32_t tsr_elf_kernel_rank(const char * name) {
    if (std::strcmp(name, "graph") == 0) return 0;
    if (std::strcmp(name, "selftest") == 0) return 1;
    if (std::strcmp(name, "replay") == 0) return 2;
    return 5;
}

int tsr_elf_check(const tsr_elf * e, const tsr_elf_limits * lim, unsigned flags, char * report, size_t replen) {
    const tsr_elf_limits l = norm_limits(lim);
    std::string rep;
    int nerr = 0;
    auto error = [&](const char * tag, const std::string & msg) {
        rep += strf("error: [%s] ", tag) + msg + "\n";
        ++nerr;
    };
    auto warning = [&](const char * tag, const std::string & msg) { rep += strf("warning: [%s] ", tag) + msg + "\n"; };

    // --- segments (EX §3.1) ---
    if (e->segs.empty()) error("seg-none", "no PT_LOAD segment");
    uint64_t itcm_end = 0, dtcm_end = TSR_RV_DTCM_BASE;
    for (size_t i = 0; i < e->segs.size(); ++i) {
        const Seg & s = e->segs[i];
        const uint64_t lo = s.pub.paddr, hi = uint64_t(s.pub.paddr) + s.pub.memsz;
        if (in_itcm(lo, hi, l.itcm_bytes)) {
            itcm_end = std::max(itcm_end, hi);
        } else if (in_dtcm(lo, hi, l.dtcm_bytes)) {
            dtcm_end = std::max(dtcm_end, hi);
        } else {
            error("seg-range", strf("PT_LOAD [0x%08llx, 0x%08llx) is not inside ITCM [0, 0x%x) or DTCM [0x%x, 0x%x) (EX §3.1)",
                                    (unsigned long long)lo, (unsigned long long)hi, l.itcm_bytes, TSR_RV_DTCM_BASE,
                                    TSR_RV_DTCM_BASE + l.dtcm_bytes));
        }
        if (s.pub.paddr % 4 != 0) error("seg-align", strf("PT_LOAD at 0x%08x is not word aligned (EX §3.1)", s.pub.paddr));
        if (s.vaddr != s.pub.paddr)
            error("seg-vaddr", strf("PT_LOAD at 0x%08x has p_vaddr 0x%08x (a static image has VMA = LMA)", s.pub.paddr, s.vaddr));
        if (i > 0) {
            const Seg & p = e->segs[i - 1];
            if (uint64_t(p.pub.paddr) + ((uint64_t(p.pub.memsz) + 3) & ~3ull) > s.pub.paddr)
                error("seg-overlap", strf("PT_LOAD segments at 0x%08x and 0x%08x overlap (EX §3.1)", p.pub.paddr, s.pub.paddr));
        }
    }
    // Allocated sections must lie inside the TCMs too (also not-loaded ones such as .bss and .stack).
    for (const Section & s : e->secs) {
        if (!s.alloc() || s.size == 0) continue;
        const uint64_t lo = s.addr, hi = uint64_t(s.addr) + s.size;
        if (in_itcm(lo, hi, l.itcm_bytes))
            itcm_end = std::max(itcm_end, hi);
        else if (in_dtcm(lo, hi, l.dtcm_bytes))
            dtcm_end = std::max(dtcm_end, hi);
        else
            error("sec-range", strf("section %s [0x%08llx, 0x%08llx) is allocated outside ITCM and DTCM", s.name.c_str(),
                                    (unsigned long long)lo, (unsigned long long)hi));
    }

    // --- reset entry (EX §3.4, SW §6.2) ---
    if (e->entry != TSR_ELF_RESET_OFFSET)
        error("entry", strf("e_entry is 0x%08x; the core starts at RV_BOOT + 0x80 = 0x%08x", e->entry, TSR_ELF_RESET_OFFSET));
    uint32_t v;
    if (tsr_elf_symbol(e, "_start", &v) == 0 && v != TSR_ELF_RESET_OFFSET)
        error("entry", strf("_start is at 0x%08x, not RV_BOOT + 0x80", v));
    const int vec = e->find_sec(".vectors");
    if (vec >= 0 && (e->secs[size_t(vec)].addr != 0 || e->secs[size_t(vec)].size < 0x80))
        error("vectors", strf(".vectors at 0x%08x size %u: must start at 0 with 32 slots of 4 bytes", e->secs[size_t(vec)].addr,
                              e->secs[size_t(vec)].size));

    // --- .tsr_id (EX §3.2) ---
    if (e->id_sec < 0) {
        error("id-missing", "no .tsr_id section (EX §3.1)");
    } else {
        const Section & s = e->secs[size_t(e->id_sec)];
        if (!s.has_data() || s.size != 8) {
            error("id-size", strf(".tsr_id must hold 8 bytes of file data (type %u, size %u)", s.type, s.size));
        } else {
            const int g = e->seg_with_data(s.addr, 8);
            if (g < 0 || !in_dtcm(s.addr, uint64_t(s.addr) + 8, l.dtcm_bytes))
                error("id-load", strf(".tsr_id at 0x%08x is not in the file data of a DTCM PT_LOAD (EX §3.1)", s.addr));
        }
        uint64_t stored;
        if ((flags & TSR_ELF_CHECK_ID) && tsr_elf_stored_id(e, &stored) == 0) {
            const uint64_t id = tsr_elf_id(e);
            if (stored != id)
                error("id-mismatch", strf(".tsr_id holds %016llx, the image's EX §3.2 id is %016llx (%s)", (unsigned long long)stored,
                                          (unsigned long long)id, stored == 0 ? "not stamped" : "image modified after stamping"));
        }
    }

    // --- kernel table (SW §6.2) ---
    if (e->kt_sec >= 0) {
        const Section & s = e->secs[size_t(e->kt_sec)];
        if (!s.has_data() || !s.alloc()) {
            error("kt-section", ".tsr_kernels is not an allocated section with file data");
        } else {
            if (s.size % TSR_ELF_KDESC_BYTES != 0)
                error("kt-size", strf(".tsr_kernels is %u bytes, not a multiple of the %u-byte entry", s.size, TSR_ELF_KDESC_BYTES));
            if (s.size > 0 && (e->seg_with_data(s.addr, s.size) < 0 || !in_dtcm(s.addr, uint64_t(s.addr) + s.size, l.dtcm_bytes)))
                error("kt-load", strf(".tsr_kernels at 0x%08x is not in the file data of a DTCM PT_LOAD (SW §6.2)", s.addr));
            if (s.addr % 4 != 0) error("kt-align", strf(".tsr_kernels at 0x%08x is not word aligned", s.addr));
        }
    }
    const uint8_t * b = e->buf.data();
    for (size_t i = 0; i < e->kernels.size(); ++i) {
        const tsr_elf_kernel & k = e->kernels[i];
        const Section & s = e->secs[size_t(e->kt_sec)];
        const uint8_t * nm = b + s.off + i * TSR_ELF_KDESC_BYTES + 12;
        size_t L = 0;
        while (L < TSR_ELF_KNAME_BYTES && nm[L] != 0) ++L;
        bool name_ok = L > 0 && L < TSR_ELF_KNAME_BYTES;   // at most 23 characters and a NUL
        for (size_t j = 0; j < L; ++j) name_ok = name_ok && is_ident_char(char(nm[j]), j == 0);
        for (size_t j = L; j < TSR_ELF_KNAME_BYTES; ++j) name_ok = name_ok && nm[j] == 0;
        const std::string who = strf("kernel %u '%s'", k.index, name_ok ? k.name : "?");
        if (!name_ok) error("kt-name", who + ": name is not a NUL-padded C identifier of 1..23 characters");
        if (k.name_hash != tsr_elf_fnv1a(k.name))
            error("kt-hash", who + strf(": name_hash 0x%08x, FNV-1a 32 of the name is 0x%08x", k.name_hash, tsr_elf_fnv1a(k.name)));
        if (k.nargs > TSR_ELF_MAX_NARGS) error("kt-nargs", who + strf(": nargs %u > 7 (ARG0..ARG6)", k.nargs));
        if (k.flags & ~TSR_ELF_KFLAG_ARGBLK) error("kt-flags", who + strf(": reserved flag bits 0x%02x", k.flags));
        if (k.flags & TSR_ELF_KFLAG_ARGBLK) {
            if (k.argblk_lines < 1 || k.argblk_lines > TSR_ELF_ARGBLK_MAX_LINES)
                error("kt-argblk", who + strf(": ARGBLK with argblk_lines %u (1..16)", k.argblk_lines));
            if (k.nargs < 1) error("kt-argblk", who + ": ARGBLK needs ARG0 (nargs >= 1)");
        } else if (k.argblk_lines != 0) {
            error("kt-argblk", who + strf(": argblk_lines %u without ARGBLK", k.argblk_lines));
        }
        bool in_x = false;
        for (const Seg & g : e->segs)
            if ((g.pub.flags & PF_X_) && k.entry >= g.vaddr && uint64_t(k.entry) + 2 <= uint64_t(g.vaddr) + g.pub.filesz &&
                in_itcm(g.pub.paddr, uint64_t(g.pub.paddr) + g.pub.memsz, l.itcm_bytes))
                in_x = true;
        if (!in_x || (k.entry & 1u)) error("kt-entry", who + strf(": entry 0x%08x is not code in an executable ITCM segment", k.entry));
        uint32_t sv;
        if (name_ok && tsr_elf_symbol(e, k.name, &sv) == 0 && sv != k.entry)
            error("kt-entry", who + strf(": entry 0x%08x differs from symbol %s at 0x%08x", k.entry, k.name, sv));
        for (size_t j = 0; j < i; ++j) {
            if (std::strcmp(e->kernels[j].name, k.name) == 0)
                error("kt-dup", who + strf(": same name as kernel %u", unsigned(j)));
            else if (e->kernels[j].name_hash == k.name_hash)
                error("kt-dup", who + strf(": same name_hash as kernel %u '%s'", unsigned(j), e->kernels[j].name));
        }
        // Sorted by index: (rank, name) strictly increasing (link.ld's SORT_BY_NAME over
        // ".tsr_kernels.<rank>.<name>", tsr/rt.hpp).
        if (i > 0) {
            const tsr_elf_kernel & p = e->kernels[i - 1];
            const uint32_t rp = tsr_elf_kernel_rank(p.name), rk = tsr_elf_kernel_rank(k.name);
            if (rp > rk || (rp == rk && std::strcmp(p.name, k.name) >= 0))
                error("kt-order", who + strf(": not sorted after kernel %u '%s' (order: graph, selftest, replay, then by name)",
                                             unsigned(i - 1), p.name));
        }
    }
    // --- kind table (SW §6.2, §6.9) ---
    {
        uint8_t rev[TSR_ELF_N_KINDS];
        char why[96];
        if (tsr_elf_kinds(e, rev, why, sizeof(why)) < 0) error("kinds", strf(".tsr_kinds is malformed: %s", why));
        if (e->kinds_sec >= 0) {
            const Section & s = e->secs[size_t(e->kinds_sec)];
            if (s.size > 0 && (!s.has_data() || !s.alloc() || e->seg_with_data(s.addr, s.size) < 0 ||
                               !in_dtcm(s.addr, uint64_t(s.addr) + s.size, l.dtcm_bytes)))
                error("kinds", strf(".tsr_kinds at 0x%08x is not in the file data of a DTCM PT_LOAD (SW §6.2)", s.addr));
        }
    }
    if (flags & TSR_ELF_CHECK_BASE) {
        static const char * const base[3] = { "graph", "selftest", "replay" };
        for (uint32_t i = 0; i < 3; ++i)
            if (i >= e->kernels.size() || std::strcmp(e->kernels[i].name, base[i]) != 0)
                error("kt-base", strf("kernel %u is '%s', EX §4.6 base kernel '%s' expected", i,
                                      i < e->kernels.size() ? e->kernels[i].name : "(none)", base[i]));
    }

    // --- .note.tsr (EX §3.1, SW §6.2) ---
    const int ns = e->find_sec(".note.tsr");
    if (ns >= 0 && e->secs[size_t(ns)].alloc())
        error("note-alloc", ".note.tsr is allocated (SW §6.2: not loaded)");
    std::vector<const NoteFound *> t1;
    for (const NoteFound & f : e->notes)
        if (f.type == TSR_ELF_NOTE_TYPE) t1.push_back(&f);
    if (t1.empty()) {
        error("note-missing", "no ELF note \"TSR\" type 1 (.note.tsr, SW §6.2)");
    } else if (t1.size() > 1) {
        error("note-dup", strf("%u notes \"TSR\" type 1", unsigned(t1.size())));
    } else if (!t1[0]->desc_ok) {
        error("note-format", strf("note \"TSR\" type 1 has a %u-byte descriptor (32)", t1[0]->descsz));
    } else {
        const tsr_elf_note & n = t1[0]->desc;
        const uint32_t itcm_need = uint32_t(itcm_end), dtcm_need = uint32_t(dtcm_end - TSR_RV_DTCM_BASE);
        if (n.sdk_abi != TSR_ELF_SDK_ABI) error("note-abi", strf("sdk_abi %u (this SDK: Basis ABI %u)", n.sdk_abi, TSR_ELF_SDK_ABI));
        if (n.cap_min == 0 || n.cap_min > 0xFFFFu)
            error("note-cap", strf("cap_min 0x%08x is not a Tesseract Core capability (VERSION[15:0], EX §7.5)", n.cap_min));
        if (n.basis_version == 0 || (n.basis_version >> 24) != 0)
            error("note-basis", strf("basis_version 0x%08x is not a Tesseract Basis version (SW §6.9)", n.basis_version));
        if (n.itcm_min < itcm_need || n.itcm_min > l.itcm_bytes)
            error("note-itcm", strf("itcm_min %u: the image occupies %u B of a %u B ITCM", n.itcm_min, itcm_need, l.itcm_bytes));
        if (n.dtcm_min < dtcm_need || n.dtcm_min > l.dtcm_bytes)
            error("note-dtcm", strf("dtcm_min %u: the image occupies %u B of a %u B DTCM", n.dtcm_min, dtcm_need, l.dtcm_bytes));
        if (n.n_kernels != e->kernels.size())
            error("note-nkernels", strf("n_kernels %u, the kernel table has %u entries", n.n_kernels, unsigned(e->kernels.size())));
        if (n.rsv != 0) error("note-rsv", "reserved note word 7 is not 0");
    }
    for (const Section & s : e->secs) {   // truncated notes
        if (!(s.type == SHT_NOTE_ || s.name == ".note.tsr") || !s.has_data()) continue;
        uint64_t o = s.off;
        const uint64_t end = uint64_t(s.off) + s.size;
        while (o + 12 <= end) {
            const uint64_t nx = o + 12 + ((uint64_t(rd32(b + o)) + 3) & ~3ull) + ((uint64_t(rd32(b + o + 4)) + 3) & ~3ull);
            if (nx > end) { error("note-format", "section " + s.name + ": truncated note"); break; }
            o = nx;
        }
    }

    // --- .tsr_logfmt (SW §6.2, §6.6) ---
    if (e->logfmt_sec >= 0) {
        const Section & s = e->secs[size_t(e->logfmt_sec)];
        if (s.alloc()) error("logfmt-alloc", ".tsr_logfmt is allocated (SW §6.2: not loaded, in no PT_LOAD)");
        if (!s.has_data()) error("logfmt-nobits", ".tsr_logfmt has no file data (the host decoders read the strings from the ELF)");
        if (s.addr != TSR_LOGFMT_VMA && s.size > 0)
            error("logfmt-addr", strf(".tsr_logfmt at 0x%08x, not TSR_LOGFMT_VMA 0x%08x", s.addr, TSR_LOGFMT_VMA));
        for (const Seg & g : e->segs)
            if (s.has_data() && s.size > 0 && uint64_t(s.off) < uint64_t(g.off) + g.pub.filesz && uint64_t(g.off) < uint64_t(s.off) + s.size)
                error("logfmt-alloc", ".tsr_logfmt is inside a PT_LOAD segment");
        if (s.has_data() && s.size > 0 && b[s.off + s.size - 1] != 0)
            error("logfmt-term", ".tsr_logfmt does not end with a NUL");
    }

    // --- SW §7.8 budget (warnings) ---
    tsr_elf_sizes z;
    tsr_elf_get_sizes(e, &l, &z);
    if (z.over_itcm) warning("budget-itcm", strf("ITCM use %u B exceeds the SW §7.8 budget of %u B", z.itcm_used, z.budget_itcm));
    if (z.over_dtcm) warning("budget-dtcm", strf("DTCM use %u B exceeds the SW §7.8 budget of %u B", z.dtcm_used, z.budget_dtcm));

    if (report && replen) {
        std::strncpy(report, rep.c_str(), replen - 1);
        report[replen - 1] = '\0';
    }
    return nerr;
}

}  // extern "C"

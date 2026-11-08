// tsr_image.cpp — firmware images on the host: tsr_image_* of tsr_transport.h (EX §3.1-§3.2, SW §6.2,
// SW §9.1) on top of the tsr-elf library (sw/ctrl/tools/tsr-elf), plus what the protocol layer and
// the trap report need from an image (tsr_image_int.h): the padded PT_LOAD segments with their
// CRC-32 (EX §3.3), .note.tsr, the kernel table, the kind table (SW §6.9), and pc -> function /
// line (DB §6.3: "mepc -> function and line") from .symtab and DWARF .debug_line (versions 2-5; the
// SDK builds with -g, SW §6.1). The versions API of tsr_transport.h (tsr_versions_str) is here too.

#include "tsr_image_int.h"

#include "tsr/opl.hpp"      // kind_name, kinds_sig (SW §6.9)
#include "tsr/version.h"    // TSR_SDK_ABI, TSR_BASIS_VERSION
#include "tsr_elf.h"
#include "tsr_fw_paths.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct tsr_image {
    std::atomic<int>            refs{ 1 };
    std::string                 path;
    tsr_elf *                   elf = nullptr;
    uint64_t                    id = 0;
    tsr::image_note             note;
    std::vector<tsr::image_seg> segs;
    std::vector<std::string>    knames;
    uint8_t                     kind_rev[TSR_ELF_N_KINDS] = {};   // .tsr_kinds (SW §6.9)
    uint32_t                    n_kinds = 0, kinds_sig = 0;

    // Symbolization, built on first use.
    struct sym {
        uint32_t    addr, size;
        std::string name;
    };
    struct row {
        uint32_t addr;
        uint32_t file;     // index into files
        uint32_t line;
        bool     end;      // end_sequence: no code at addr
    };
    mutable std::once_flag      sym_once;
    mutable std::vector<sym>    syms;     // FUNC symbols, by address
    mutable std::vector<row>    rows;     // line rows, sequences in file order
    mutable std::vector<std::string> files;

    ~tsr_image() { tsr_elf_free(elf); }
};

namespace tsr {

uint32_t crc32(const void * data, size_t n) {
    static uint32_t tab[256];
    static std::once_flag once;
    std::call_once(once, [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (TSR_CRC32_POLY & (0u - (c & 1u)));
            tab[i] = c;
        }
    });
    const uint8_t * p = (const uint8_t *) data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = tab[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return ~c;
}

const std::vector<image_seg> & image_segments(const tsr_image * img) { return img->segs; }
image_note image_get_note(const tsr_image * img) { return img->note; }
const tsr_elf * image_elf(const tsr_image * img) { return img->elf; }
uint32_t image_n_kernels(const tsr_image * img) { return (uint32_t) img->knames.size(); }
const char * image_kernel_name(const tsr_image * img, uint32_t index) {
    return index < img->knames.size() ? img->knames[index].c_str() : "";
}
const uint8_t * image_kind_revs(const tsr_image * img) { return img->kind_rev; }
uint32_t image_n_kinds(const tsr_image * img) { return img->n_kinds; }
uint32_t image_kinds_sig(const tsr_image * img) { return img->kinds_sig; }

void image_ref(const tsr_image * img) {
    if (img) const_cast<tsr_image *>(img)->refs.fetch_add(1);
}
void image_unref(const tsr_image * img) {
    if (img && const_cast<tsr_image *>(img)->refs.fetch_sub(1) == 1) delete const_cast<tsr_image *>(img);
}

void image_test_set_id(tsr_image * img, uint64_t id) { img->id = id; }
void image_test_set_note(tsr_image * img, const image_note & note) { img->note = note; }
void image_test_set_n_kernels(tsr_image * img, uint32_t n) {
    while (img->knames.size() < n) img->knames.push_back("test_kernel_" + std::to_string(img->knames.size()));
    img->knames.resize(n);
}
void image_test_set_kinds(tsr_image * img, const uint8_t rev[TSR_ELF_N_KINDS]) {
    img->n_kinds = 0;
    for (uint32_t k = 0; k < TSR_ELF_N_KINDS; ++k) {
        img->kind_rev[k] = rev[k];
        img->n_kinds += rev[k] != 0;
    }
    img->kinds_sig = tsr::opl::kinds_sig(img->kind_rev);
}

// ---------------------------------------------------------------------------
// Symbols and DWARF line tables
// ---------------------------------------------------------------------------
namespace {

struct rd {
    const uint8_t * p;
    const uint8_t * end;
    bool            bad = false;
    bool     has(size_t n) const { return !bad && (size_t) (end - p) >= n; }
    uint64_t u(size_t n) {
        if (!has(n)) {
            bad = true;
            p = end;
            return 0;
        }
        uint64_t v = 0;
        for (size_t i = 0; i < n; ++i) v |= (uint64_t) p[i] << (8 * i);
        p += n;
        return v;
    }
    uint64_t uleb() {
        uint64_t v = 0;
        for (int s = 0; s < 64; s += 7) {
            if (!has(1)) break;
            const uint8_t b = *p++;
            v |= (uint64_t) (b & 0x7F) << s;
            if (!(b & 0x80)) return v;
        }
        bad = true;
        return v;
    }
    int64_t sleb() {
        int64_t v = 0;
        int s = 0;
        uint8_t b = 0;
        do {
            if (!has(1)) {
                bad = true;
                return v;
            }
            b = *p++;
            v |= (int64_t) (b & 0x7F) << s;
            s += 7;
        } while ((b & 0x80) && s < 64);
        if (s < 64 && (b & 0x40)) v |= -((int64_t) 1 << s);
        return v;
    }
    std::string str() {
        const uint8_t * q = p;
        while (q < end && *q) ++q;
        if (q >= end) {
            bad = true;
            p = end;
            return std::string();
        }
        std::string s((const char *) p, (size_t) (q - p));
        p = q + 1;
        return s;
    }
    void skip(size_t n) {
        if (!has(n)) {
            bad = true;
            p = end;
        } else {
            p += n;
        }
    }
};

struct section {
    const uint8_t * p = nullptr;
    size_t          n = 0;
};

section get_section(const tsr_elf * e, const char * name) {
    section s;
    uint32_t addr = 0, off = 0, size = 0, type = 0;
    size_t total = 0;
    const uint8_t * b = tsr_elf_bytes(e, &total);
    if (tsr_elf_section(e, name, &addr, &off, &size, &type) != 0 || type == 8 /* SHT_NOBITS */) return s;
    if ((uint64_t) off + size > total) return s;
    s.p = b + off;
    s.n = size;
    return s;
}

std::string demangle(const std::string & m) {
    if (m.size() < 2 || m[0] != '_' || m[1] != 'Z') return m;
    int st = 0;
    char * d = abi::__cxa_demangle(m.c_str(), nullptr, nullptr, &st);
    if (st != 0 || d == nullptr) return m;
    std::string r(d);
    free(d);
    return r;
}

void load_symbols(const tsr_image * img) {
    const section st = get_section(img->elf, ".symtab"), ss = get_section(img->elf, ".strtab");
    if (st.p == nullptr || ss.p == nullptr) return;
    for (size_t i = 0; i + 16 <= st.n; i += 16) {
        rd r{ st.p + i, st.p + i + 16 };
        const uint32_t name = (uint32_t) r.u(4), value = (uint32_t) r.u(4), size = (uint32_t) r.u(4);
        const uint8_t info = (uint8_t) r.u(1);
        if ((info & 0xF) != 2 /* STT_FUNC */ || name >= ss.n) continue;
        const char * nm = (const char *) ss.p + name;
        const size_t len = strnlen(nm, ss.n - name);
        img->syms.push_back({ value & ~1u, size, demangle(std::string(nm, len)) });
    }
    std::sort(img->syms.begin(), img->syms.end(), [](const tsr_image::sym & a, const tsr_image::sym & b) { return a.addr < b.addr; });
}

// DW_FORM values used in DWARF 5 line headers.
enum : uint64_t {
    F_block = 0x09, F_block1 = 0x0a, F_data1 = 0x0b, F_data2 = 0x05, F_data4 = 0x06, F_data8 = 0x07, F_data16 = 0x1e,
    F_string = 0x08, F_strp = 0x0e, F_line_strp = 0x1f, F_udata = 0x0f, F_sdata = 0x0d, F_strx = 0x1a, F_strx1 = 0x25,
    F_strx2 = 0x26, F_strx4 = 0x28,
};

// One attribute value of a DWARF 5 entry format: a string (path) or a number.
bool form_value(rd & r, uint64_t form, bool dwarf64, const section & line_str, const section & str, std::string * s,
                uint64_t * v) {
    *v = 0;
    auto strp = [&](const section & sec) {
        const uint64_t off = r.u(dwarf64 ? 8 : 4);
        if (sec.p != nullptr && off < sec.n) *s = std::string((const char *) sec.p + off, strnlen((const char *) sec.p + off, sec.n - off));
    };
    switch (form) {
    case F_string:    *s = r.str(); return !r.bad;
    case F_line_strp: strp(line_str); return !r.bad;
    case F_strp:      strp(str); return !r.bad;
    case F_udata:     *v = r.uleb(); return !r.bad;
    case F_sdata:     *v = (uint64_t) r.sleb(); return !r.bad;
    case F_data1:     *v = r.u(1); return !r.bad;
    case F_data2:     *v = r.u(2); return !r.bad;
    case F_data4:     *v = r.u(4); return !r.bad;
    case F_data8:     *v = r.u(8); return !r.bad;
    case F_data16:    r.skip(16); return !r.bad;
    case F_block:     r.skip((size_t) r.uleb()); return !r.bad;
    case F_block1:    r.skip((size_t) r.u(1)); return !r.bad;
    case F_strx: case F_strx1: case F_strx2: case F_strx4:   // needs .debug_str_offsets: not used by GCC here
        if (form == F_strx) r.uleb();
        else r.skip(form == F_strx1 ? 1 : (form == F_strx2 ? 2 : 4));
        return !r.bad;
    default: return false;
    }
}

void load_lines(const tsr_image * img) {
    const section dl = get_section(img->elf, ".debug_line");
    if (dl.p == nullptr) return;
    const section line_str = get_section(img->elf, ".debug_line_str"), str = get_section(img->elf, ".debug_str");
    rd all{ dl.p, dl.p + dl.n };
    while (all.has(4)) {
        uint64_t len = all.u(4);
        bool dwarf64 = false;
        if (len == 0xFFFFFFFFull) {
            len = all.u(8);
            dwarf64 = true;
        }
        if (!all.has((size_t) len)) break;
        rd u{ all.p, all.p + len };
        all.p += len;
        const uint32_t version = (uint32_t) u.u(2);
        if (version < 2 || version > 5) continue;
        uint32_t addr_size = 4;
        if (version >= 5) {
            addr_size = (uint32_t) u.u(1);
            u.u(1);   // segment selector size
        }
        const uint64_t hlen = u.u(dwarf64 ? 8 : 4);
        if (!u.has((size_t) hlen)) continue;
        const uint8_t * prog = u.p + hlen;
        const uint32_t min_inst = (uint32_t) u.u(1);
        if (version >= 4) u.u(1);   // maximum_operations_per_instruction (1: no VLIW)
        const bool default_is_stmt = u.u(1) != 0;
        (void) default_is_stmt;
        const int line_base = (int8_t) u.u(1);
        const uint32_t line_range = (uint32_t) u.u(1);
        const uint32_t opcode_base = (uint32_t) u.u(1);
        std::vector<uint8_t> std_len(opcode_base > 0 ? opcode_base - 1 : 0);
        for (auto & l : std_len) l = (uint8_t) u.u(1);
        if (u.bad || line_range == 0) continue;
        std::vector<std::string> dirs, names;
        std::vector<uint64_t> name_dir;
        if (version >= 5) {
            auto entries = [&](std::vector<std::string> * out, std::vector<uint64_t> * dir_idx) -> bool {
                const uint32_t nf = (uint32_t) u.u(1);
                std::vector<std::pair<uint64_t, uint64_t>> fmt(nf);
                for (auto & f : fmt) {
                    f.first = u.uleb();
                    f.second = u.uleb();
                }
                const uint64_t count = u.uleb();
                for (uint64_t i = 0; i < count && !u.bad; ++i) {
                    std::string path;
                    uint64_t dir = 0;
                    for (auto & f : fmt) {
                        std::string s;
                        uint64_t v = 0;
                        if (!form_value(u, f.second, dwarf64, line_str, str, &s, &v)) return false;
                        if (f.first == 1) path = s;          // DW_LNCT_path
                        else if (f.first == 2) dir = v;      // DW_LNCT_directory_index
                    }
                    out->push_back(path);
                    if (dir_idx) dir_idx->push_back(dir);
                }
                return !u.bad;
            };
            if (!entries(&dirs, nullptr) || !entries(&names, &name_dir)) continue;
        } else {
            dirs.push_back(std::string());   // index 0: the compilation directory
            for (;;) {
                std::string d = u.str();
                if (u.bad || d.empty()) break;
                dirs.push_back(d);
            }
            names.push_back(std::string());  // file index 0 is unused before DWARF 5
            name_dir.push_back(0);
            for (;;) {
                std::string f = u.str();
                if (u.bad || f.empty()) break;
                name_dir.push_back(u.uleb());
                u.uleb();
                u.uleb();
                names.push_back(f);
            }
        }
        if (u.bad) continue;
        const uint32_t file_base = (uint32_t) img->files.size();
        for (size_t i = 0; i < names.size(); ++i) {
            std::string p = names[i];
            const uint64_t d = i < name_dir.size() ? name_dir[i] : 0;
            if (!p.empty() && p[0] != '/' && d < dirs.size() && !dirs[d].empty()) p = dirs[d] + "/" + p;
            img->files.push_back(p);
        }
        // The line-number program.
        rd r{ prog, u.end };
        uint64_t addr = 0;
        uint32_t file = 1, line = 1;
        auto emit = [&](bool end) {
            const uint32_t f = file < names.size() ? file_base + file : UINT32_MAX;
            img->rows.push_back({ (uint32_t) addr, f, line, end });
        };
        while (r.has(1)) {
            const uint32_t op = (uint32_t) r.u(1);
            if (op >= opcode_base) {
                const uint32_t adj = op - opcode_base;
                addr += (uint64_t) (adj / line_range) * min_inst;
                line = (uint32_t) ((int64_t) line + line_base + (int64_t) (adj % line_range));
                emit(false);
            } else if (op == 0) {
                const uint64_t elen = r.uleb();
                if (!r.has((size_t) elen) || elen == 0) break;
                const uint8_t * next = r.p + elen;
                const uint32_t sub = (uint32_t) r.u(1);
                if (sub == 1) {          // DW_LNE_end_sequence
                    emit(true);
                    addr = 0;
                    file = 1;
                    line = 1;
                } else if (sub == 2) {   // DW_LNE_set_address
                    addr = r.u(addr_size);
                }
                r.p = next;              // define_file, set_discriminator, vendor ops: skipped
            } else {
                switch (op) {
                case 1: emit(false); break;                                         // copy
                case 2: addr += r.uleb() * min_inst; break;                         // advance_pc
                case 3: line = (uint32_t) ((int64_t) line + r.sleb()); break;       // advance_line
                case 4: file = (uint32_t) r.uleb(); break;                          // set_file
                case 5: r.uleb(); break;                                            // set_column
                case 6: case 7: case 10: case 11: break;                            // flags
                case 8: addr += (uint64_t) ((255u - opcode_base) / line_range) * min_inst; break;   // const_add_pc
                case 9: addr += r.u(2); break;                                      // fixed_advance_pc
                default:                                                            // unknown standard opcode
                    for (uint32_t k = 0; k < std_len[op - 1]; ++k) r.uleb();
                    break;
                }
            }
        }
    }
}

void build_symbols(const tsr_image * img) {
    std::call_once(img->sym_once, [img] {
        load_symbols(img);
        load_lines(img);
    });
}

// The row covering pc: rows of one sequence ascend; pc is in [row.addr, next.addr).
const tsr_image::row * find_row(const tsr_image * img, uint32_t pc) {
    const tsr_image::row * best = nullptr;
    const auto & rows = img->rows;
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        if (rows[i].end) continue;
        const uint32_t a = rows[i].addr, b = rows[i + 1].addr;
        if (pc >= a && pc < b && (best == nullptr || a >= best->addr)) best = &rows[i];
    }
    return best;
}

} // namespace

std::string image_line(const tsr_image * img, uint32_t pc) {
    if (img == nullptr) return std::string();
    build_symbols(img);
    const tsr_image::row * r = find_row(img, pc);
    if (r == nullptr || r->file >= img->files.size()) return std::string();
    return img->files[r->file] + ":" + std::to_string(r->line);
}

std::string image_symbolize(const tsr_image * img, uint32_t pc) {
    char buf[64];
    snprintf(buf, sizeof(buf), "0x%08x", pc);
    std::string out = buf;
    if (img == nullptr) return out;
    build_symbols(img);
    // The FUNC symbol with the highest address <= pc that covers pc (size 0: up to the next one).
    const tsr_image::sym * best = nullptr;
    for (const auto & s : img->syms) {
        if (s.addr > pc) break;
        if (s.size == 0 || pc < s.addr + s.size) best = &s;
    }
    if (best != nullptr) {
        snprintf(buf, sizeof(buf), "+0x%x", pc - best->addr);
        out += " " + best->name + buf;
    }
    const std::string line = image_line(img, pc);
    if (!line.empty()) out += " (" + line + ")";
    return out;
}

} // namespace tsr

// ---------------------------------------------------------------------------
// C API (EX §10.1, SW §9.1)
// ---------------------------------------------------------------------------
using namespace tsr;

extern "C" int tsr_image_open(const char * elf_path, tsr_image ** out) {
    if (out == nullptr) return TSR_TE_INVAL;
    *out = nullptr;
    if (elf_path == nullptr || elf_path[0] == '\0') {
        fprintf(stderr, "tsr_image: no ELF path\n");
        return TSR_TE_INVAL;
    }
    char err[512] = "";
    tsr_elf * e = nullptr;
    if (tsr_elf_read(elf_path, &e, err, sizeof(err)) != 0) {
        fprintf(stderr, "tsr_image: %s: %s\n", elf_path, err);
        return TSR_TE_IMAGE;
    }
    std::unique_ptr<tsr_image> img(new tsr_image);
    img->path = elf_path;
    img->elf = e;
    // The versions first (SW §6.9), with their own messages: the Basis ABI, a Tesseract Basis
    // version, a well-formed kind table. tsr_elf_check below would refuse them less clearly.
    tsr_elf_note n{};
    if (tsr_elf_get_note(e, &n) != 0) {
        fprintf(stderr, "tsr_image: %s has no .note.tsr\n", elf_path);
        return TSR_TE_IMAGE;
    }
    char host[32];
    snprintf(host, sizeof(host), "%u.%u.%u", TSR_BASIS_VERSION_MAJOR, TSR_BASIS_VERSION_MINOR, TSR_BASIS_VERSION_PATCH);
    if (n.sdk_abi != TSR_SDK_ABI) {
        fprintf(stderr, "tsr_image: %s: Basis ABI %u, this host implements ABI %u (Tesseract Basis %s): rebuild the image with "
                "this tree's SDK\n", elf_path, n.sdk_abi, TSR_SDK_ABI, host);
        return TSR_TE_IMAGE;
    }
    if (n.basis_version == 0) {
        fprintf(stderr, "tsr_image: %s: no Tesseract Basis version in .note.tsr (built before versioning): rebuild the image\n",
                elf_path);
        return TSR_TE_IMAGE;
    }
    char why[96] = "";
    const int nk = tsr_elf_kinds(e, img->kind_rev, why, sizeof(why));
    if (nk < 0) {
        fprintf(stderr, "tsr_image: %s: malformed kind table .tsr_kinds (%s)\n", elf_path, why);
        return TSR_TE_IMAGE;
    }
    img->n_kinds = uint32_t(nk);
    img->kinds_sig = tsr::opl::kinds_sig(img->kind_rev);
    // EX §3.1 / SW §6.2 rules, and a stamped image whose stored id is the computed one (EX §3.2).
    std::vector<char> rep(8192);
    const int nerr = tsr_elf_check(e, nullptr, TSR_ELF_CHECK_ID, rep.data(), rep.size());
    if (nerr != 0) {
        fprintf(stderr, "tsr_image: %s is not a valid firmware image (%d error%s):\n%s", elf_path, nerr, nerr == 1 ? "" : "s",
                rep.data());
        return TSR_TE_IMAGE;
    }
    img->note.sdk_abi = n.sdk_abi;
    img->note.cap_min = n.cap_min;
    img->note.itcm_min = n.itcm_min;
    img->note.dtcm_min = n.dtcm_min;
    img->note.caps_req = n.caps_req;
    img->note.n_kernels = n.n_kernels;
    img->note.basis_version = n.basis_version;
    img->id = tsr_elf_id(e);
    for (uint32_t i = 0; i < tsr_elf_nseg(e); ++i) {
        const tsr_elf_seg * s = tsr_elf_segment(e, i);
        if (s->filesz == 0) continue;   // .bss only: crt0 zeroes it (EX §3.1)
        image_seg g;
        g.paddr = s->paddr;
        g.itcm = s->paddr < TSR_RV_ITCM_LIMIT;   // ITCM at 0, DTCM at 0x0010_0000 (CC §4.1; tsr_elf_check placed them)
        g.bytes.assign(s->padded, 0);
        memcpy(g.bytes.data(), s->data, s->filesz);
        g.crc = crc32(g.bytes.data(), g.bytes.size());
        img->segs.push_back(std::move(g));
    }
    for (uint32_t k = 0; k < tsr_elf_n_kernels(e); ++k) img->knames.push_back(tsr_elf_kernel_at(e, k)->name);
    *out = img.release();
    return 0;
}

extern "C" void tsr_image_close(tsr_image * img) { image_unref(img); }

extern "C" int tsr_image_kernel(const tsr_image * img, const char * name) {
    if (img == nullptr || name == nullptr) return -1;
    return tsr_elf_kernel_find(img->elf, name);
}

extern "C" uint64_t tsr_image_id(const tsr_image * img) { return img ? img->id : 0; }

extern "C" const char * tsr_image_path(const tsr_image * img) { return img ? img->path.c_str() : ""; }

extern "C" int tsr_image_open_default(tsr_image ** out) {
    const char * p = getenv("TSR_IMAGE");
    if (p == nullptr || p[0] == '\0') p = TSR_FW_DEFAULT_IMAGE;
    return tsr_image_open(p, out);
}

// ---------------------------------------------------------------------------
// Versions (SW §6.9)
// ---------------------------------------------------------------------------
extern "C" uint32_t tsr_basis_version(void) { return TSR_BASIS_VERSION; }

extern "C" int tsr_image_versions(const tsr_image * img, tsr_versions * v) {
    if (img == nullptr || v == nullptr) return TSR_TE_INVAL;
    memset(v, 0, sizeof(*v));
    v->core_min = img->note.cap_min;
    v->basis = img->note.basis_version;
    v->abi = img->note.sdk_abi;
    v->kinds_sig = img->kinds_sig;
    v->n_kinds = img->n_kinds;
    memcpy(v->kind_rev, img->kind_rev, sizeof(v->kind_rev));
    v->image_id = img->id;
    return 0;
}

extern "C" int tsr_versions_str(const tsr_versions * v, int verbose, char * buf, size_t n) {
    if (v == nullptr || (buf == nullptr && n != 0)) return TSR_TE_INVAL;
    std::string s;
    char b[96];
    if (v->core != 0) {
        snprintf(b, sizeof(b), "Tesseract Core %u.%u", (v->core >> 8) & 0xFFu, v->core & 0xFFu);
        s += b;
        static const struct { uint32_t bit; const char * name; } caps[] = {
            { TSR_CAPS_VU, "VU" }, { TSR_CAPS_TOPK, "TOPK" }, { TSR_CAPS_ATT, "ATT" }, { TSR_CAPS_DQROW, "DQROW" },
            { TSR_CAPS_VMASK, "VMASK" }, { TSR_CAPS_GDN, "GDN" }, { TSR_CAPS_ATT_LOCAL, "ATT_LOCAL" }, { TSR_CAPS_MVM_LOCAL_SCALE, "MVM_LOCAL_SCALE" }, { TSR_CAPS_MVM_LOCAL_SOFTCAP, "MVM_LOCAL_SOFTCAP" } };
        std::string c;
        for (const auto & x : caps)
            if (v->caps & x.bit) c += (c.empty() ? "" : " ") + std::string(x.name);
        if (!c.empty()) s += " (" + c + ")";
    }
    if (v->basis != 0) {
        snprintf(b, sizeof(b), "%sTesseract Basis %u.%u.%u (ABI %u, %u kinds", s.empty() ? "" : ", ", TSR_BASIS_MAJOR_OF(v->basis),
                 TSR_BASIS_MINOR_OF(v->basis), TSR_BASIS_PATCH_OF(v->basis), v->abi, v->n_kinds);
        s += b;
        s += ")";
        if (verbose) {
            snprintf(b, sizeof(b), ", image %016" PRIx64 ", kinds 0x%04x:", v->image_id, v->kinds_sig);
            s += b;
            for (uint32_t k = 0; k < TSR_ELF_N_KINDS; ++k) {
                if (v->kind_rev[k] == 0) continue;
                snprintf(b, sizeof(b), " %s r%u", tsr::opl::kind_name(k), v->kind_rev[k]);
                s += b;
            }
        }
    }
    if (n != 0) {
        const size_t k = std::min(s.size(), n - 1);
        memcpy(buf, s.data(), k);
        buf[k] = '\0';
    }
    return int(s.size());
}

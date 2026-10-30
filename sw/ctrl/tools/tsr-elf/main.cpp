// tsr-elf — firmware image tool (SW §9.5; SW §6.2, EX §3.1-§3.3; P2.1).
//
//   tsr-elf stamp IN.elf -o OUT.elf [--report F] [--logfmt F] [limits] [--strict-budget] [--base] [-q]
//         after the link: checks IN (tsr_elf_check without the id), computes the EX §3.2 image id,
//         writes it into .tsr_id of OUT, and writes the size report and the log-format table.
//         On any error OUT is removed, so a build never keeps a stale image.
//   tsr-elf check IMAGE [limits] [--strict-budget] [--no-id] [--base]
//         every check, including "the stored id equals the computed one"; exit 1 on errors.
//         --base: graph, selftest and replay must be kernels 0, 1, 2 (EX §4.6; the production image)
//   tsr-elf id IMAGE [--stored]     the computed (or stored) id, 16 hex digits
//   tsr-elf size IMAGE [limits] [--strict-budget]   size report against SW §7.8's budget
//   tsr-elf kernels IMAGE           kernel table
//   tsr-elf note IMAGE              .note.tsr
//   tsr-elf logfmt IMAGE            log-format table: "0xADDR<TAB>string" per line (C escapes)
//   tsr-elf segments IMAGE          PT_LOAD segments as loaded: paddr, filesz, padded, memsz, CRC-32
//                                   (zlib, of the padded bytes: the LOAD_CRC the host expects, EX §3.3)
// limits: --itcm BYTES --dtcm BYTES (default: tsr_tables.h, U200), --budget-itcm / --budget-dtcm
// BYTES (default SW §7.8: 48 KiB / 56 KiB). Numbers take 0x and K suffixes.
// --strict-budget turns a budget overrun into a failure (exit 1).
#include "tsr_elf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: tsr-elf stamp IN.elf -o OUT.elf [--report F] [--logfmt F] [limits] [--strict-budget] [-q]\n"
                 "       tsr-elf check IMAGE [limits] [--strict-budget] [--no-id] [--base]\n"
                 "       tsr-elf id IMAGE [--stored]\n"
                 "       tsr-elf size IMAGE [limits] [--strict-budget]\n"
                 "       tsr-elf kernels|note|logfmt|segments IMAGE\n"
                 "limits: --itcm BYTES --dtcm BYTES --budget-itcm BYTES --budget-dtcm BYTES\n");
}

bool parse_num(const char * s, uint32_t * v) {
    char * end = nullptr;
    unsigned long long x = std::strtoull(s, &end, 0);
    if (end == s) return false;
    if (*end == 'K' || *end == 'k') {
        x *= 1024u;
        ++end;
    }
    if (*end != '\0' || x > 0xFFFFFFFFull) return false;
    *v = uint32_t(x);
    return true;
}

uint32_t crc32_zlib(const uint8_t * p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

std::string escape(const char * s) {
    std::string r;
    for (; *s; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\t') r += "\\t";
        else if (c == '\r') r += "\\r";
        else if (c < 0x20 || c >= 0x7F) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\x%02x", c);
            r += b;
        } else r += char(c);
    }
    return r;
}

bool write_text(const std::string & path, const std::string & text) {
    FILE * f = std::fopen(path.c_str(), "w");
    if (f == nullptr) return false;
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    return std::fclose(f) == 0 && ok;
}

std::string logfmt_table(const tsr_elf * e) {
    std::string r;
    for (uint32_t i = 0; i < tsr_elf_logfmt_count(e); ++i) {
        uint32_t a = 0;
        const char * s = tsr_elf_logfmt_at(e, i, &a);
        char b[16];
        std::snprintf(b, sizeof(b), "0x%08x\t", a);
        r += b;
        r += s ? escape(s) : std::string("<unterminated>");
        r += "\n";
    }
    return r;
}

std::string report_of(const tsr_elf * e, const tsr_elf_limits * l) {
    std::vector<char> b(1 << 20);
    tsr_elf_size_report(e, l, b.data(), b.size());
    return b.data();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    std::string in, out, report_path, logfmt_path;
    tsr_elf_limits lim = { 0, 0, 0, 0 };
    bool strict = false, quiet = false, no_id = false, stored = false, base = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&](uint32_t * v) {
            if (i + 1 >= argc || !parse_num(argv[i + 1], v)) {
                std::fprintf(stderr, "tsr-elf: %s needs a number\n", a.c_str());
                std::exit(2);
            }
            ++i;
        };
        auto str = [&](std::string * v) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tsr-elf: %s needs an argument\n", a.c_str());
                std::exit(2);
            }
            *v = argv[++i];
        };
        if (a == "-o") str(&out);
        else if (a == "--report") str(&report_path);
        else if (a == "--logfmt") str(&logfmt_path);
        else if (a == "--itcm") num(&lim.itcm_bytes);
        else if (a == "--dtcm") num(&lim.dtcm_bytes);
        else if (a == "--budget-itcm") num(&lim.budget_itcm);
        else if (a == "--budget-dtcm") num(&lim.budget_dtcm);
        else if (a == "--strict-budget") strict = true;
        else if (a == "--no-id") no_id = true;
        else if (a == "--base") base = true;
        else if (a == "--stored") stored = true;
        else if (a == "-q") quiet = true;
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "tsr-elf: unknown option %s\n", a.c_str());
            usage();
            return 2;
        } else if (in.empty()) in = a;
        else {
            usage();
            return 2;
        }
    }
    if (in.empty()) {
        usage();
        return 2;
    }

    char err[512];
    tsr_elf * e = nullptr;
    // Never leave a stale image (in-place stamping, OUT = IN, keeps the input on failure).
    if (cmd == "stamp" && !out.empty() && out != in) std::remove(out.c_str());
    if (tsr_elf_read(in.c_str(), &e, err, sizeof(err)) != 0) {
        std::fprintf(stderr, "tsr-elf: %s\n", err);
        return 1;
    }
    std::vector<char> rep(1 << 20);
    int rc = 0;

    if (cmd == "stamp" || cmd == "check") {
        if (cmd == "stamp" && out.empty()) {
            usage();
            tsr_elf_free(e);
            return 2;
        }
        const unsigned cf = (cmd == "check" && !no_id ? TSR_ELF_CHECK_ID : 0u) | (base ? TSR_ELF_CHECK_BASE : 0u);
        const int nerr = tsr_elf_check(e, &lim, cf, rep.data(), rep.size());
        if (rep[0]) std::fprintf(stderr, "%s: %s", in.c_str(), rep.data());
        tsr_elf_sizes z;
        tsr_elf_get_sizes(e, &lim, &z);
        const bool over = z.over_itcm || z.over_dtcm;
        if (nerr > 0 || (strict && over)) {
            std::fprintf(stderr, "tsr-elf %s: %s: %d error(s)%s\n", cmd.c_str(), in.c_str(), nerr,
                         strict && over ? ", SW §7.8 budget exceeded" : "");
            tsr_elf_free(e);
            return 1;
        }
        if (cmd == "stamp") {
            const uint64_t id = tsr_elf_id(e);
            if (tsr_elf_set_id(e, id) != 0 || tsr_elf_id(e) != id) {   // .tsr_id does not enter the id
                std::fprintf(stderr, "tsr-elf stamp: %s: writing .tsr_id changed the id\n", in.c_str());
                tsr_elf_free(e);
                return 1;
            }
            const std::string text = report_of(e, &lim);
            if ((!report_path.empty() && !write_text(report_path, text)) ||
                (!logfmt_path.empty() && !write_text(logfmt_path, logfmt_table(e)))) {
                std::fprintf(stderr, "tsr-elf stamp: cannot write the report / log-format table\n");
                tsr_elf_free(e);
                return 1;
            }
            if (tsr_elf_write(e, out.c_str(), err, sizeof(err)) != 0) {
                std::fprintf(stderr, "tsr-elf stamp: %s\n", err);
                tsr_elf_free(e);
                return 1;
            }
            if (!quiet)
                std::printf("tsr-elf: %s id %016llx, ITCM %u B, DTCM %u B (budget %u / %u), %u kernels, %u log formats\n",
                            out.c_str(), (unsigned long long)id, z.itcm_used, z.dtcm_used, z.budget_itcm, z.budget_dtcm,
                            tsr_elf_n_kernels(e), tsr_elf_logfmt_count(e));
        } else if (!quiet) {
            std::printf("%s: ok, id %016llx\n", in.c_str(), (unsigned long long)tsr_elf_id(e));
        }
    } else if (cmd == "id") {
        uint64_t id = tsr_elf_id(e);
        if (stored && tsr_elf_stored_id(e, &id) != 0) {
            std::fprintf(stderr, "tsr-elf: %s has no .tsr_id\n", in.c_str());
            rc = 1;
        } else {
            std::printf("%016llx\n", (unsigned long long)id);
        }
    } else if (cmd == "size") {
        std::fputs(report_of(e, &lim).c_str(), stdout);
        tsr_elf_sizes z;
        tsr_elf_get_sizes(e, &lim, &z);
        if (z.over_itcm || z.over_dtcm) {
            std::fprintf(stderr, "%s: warning: SW §7.8 budget exceeded (ITCM %u / %u B, DTCM %u / %u B)\n", in.c_str(),
                         z.itcm_used, z.budget_itcm, z.dtcm_used, z.budget_dtcm);
            if (strict) rc = 1;
        }
    } else if (cmd == "kernels") {
        for (uint32_t i = 0; i < tsr_elf_n_kernels(e); ++i) {
            const tsr_elf_kernel * k = tsr_elf_kernel_at(e, i);
            std::printf("%u\t%s\tentry=0x%08x\tnargs=%u\tflags=0x%02x\targblk_lines=%u\thash=0x%08x\n", k->index, k->name,
                        k->entry, k->nargs, k->flags, k->argblk_lines, k->name_hash);
        }
    } else if (cmd == "note") {
        tsr_elf_note n;
        if (tsr_elf_get_note(e, &n) != 0) {
            std::fprintf(stderr, "tsr-elf: %s has no .note.tsr\n", in.c_str());
            rc = 1;
        } else {
            std::printf("sdk_abi=%u\nctl_id_min=%u\nitcm_min=%u\ndtcm_min=%u\ncaps_req=0x%08x\nn_kernels=%u\n", n.sdk_abi,
                        n.ctl_id_min, n.itcm_min, n.dtcm_min, n.caps_req, n.n_kernels);
        }
    } else if (cmd == "logfmt") {
        std::fputs(logfmt_table(e).c_str(), stdout);
    } else if (cmd == "segments") {
        for (uint32_t i = 0; i < tsr_elf_nseg(e); ++i) {
            const tsr_elf_seg * g = tsr_elf_segment(e, i);
            std::vector<uint8_t> p(g->padded, 0);
            if (g->filesz) std::memcpy(p.data(), g->data, g->filesz);
            std::printf("0x%08x\tfilesz=%u\tpadded=%u\tmemsz=%u\tflags=%c%c%c\tcrc32=0x%08x\n", g->paddr, g->filesz, g->padded,
                        g->memsz, (g->flags & 4) ? 'r' : '-', (g->flags & 2) ? 'w' : '-', (g->flags & 1) ? 'x' : '-',
                        crc32_zlib(p.data(), p.size()));
        }
    } else {
        usage();
        rc = 2;
    }
    tsr_elf_free(e);
    return rc;
}

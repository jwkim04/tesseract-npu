// tsr_log.cpp — tsr-log: decode raw device log words with a firmware ELF (docs/design/
// 06-software-verification.md "SW" §9.5; record format docs/design/05-debug.md "DB" §6.1). The text of
// each record is the transport's decoder (sw/transport/tsr_logdec.h, the same lines TSR_LOG prints),
// so a saved log reads exactly like a live one.
//
//   tsr-log [--elf IMAGE | --no-elf] [--hex] [--level N] [-v] [FILE | -]...
//   tsr-log selftest [IMAGE]
//
//   FILE       raw log words: little-endian 32-bit words (what tsr-selftest --log-out writes and
//              tsr_transport_log_drain returns), or with --hex whitespace-separated hex words (0x
//              optional, '#' starts a comment); "-" or no FILE: stdin
//   --elf      the image that wrote them (FMT format strings, %s strings, TRAP symbols and lines);
//              default $TSR_IMAGE, else the build's tsr-fw.elf (tsr_fw_paths.h) when it exists
//   --level N  print only records with LEVEL <= N (0 FATAL .. 5 TRACE)
//   -v         prefix each line with the word index of its header
// Output: one line per record, "[tsr LEVEL source seq LSEQ t MTIME] text", and one "tsr-log: bad
// record" line per malformed record (DB §6.1 checks: SYNC, TYPE, LEVEL, the FMT address inside
// .tsr_logfmt, a record cut short). The transport skips to LOG_WR after a bad header (DB §6.1);
// a saved file may hold several drains, so tsr-log resynchronizes at the next valid header and
// says how many words it skipped. Summary on stderr.
// Exit: 0 every word decoded, 1 malformed records, 2 usage or I/O error.
//
// selftest: builds records of every type against IMAGE's format table (the `selftest` kernel's id 9
// strings, a TEXT, a "dropped n" EVENT, a BLOB, a TRAP whose mepc lies in the `selftest` kernel) and
// three malformed ones, and checks the decoded text and the resynchronization (ctest
// tools_log_selftest).
#include "tsr_elf.h"
#include "tsr_fw_paths.h"
#include "tsr_isa.h"
#include "tsr_logdec.h"
#include "tsr_transport.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

bool read_words(const char * path, bool hex, std::vector<uint32_t> & w, std::string & err) {
    FILE * f = (path == nullptr || strcmp(path, "-") == 0) ? stdin : fopen(path, hex ? "r" : "rb");
    if (f == nullptr) {
        err = std::string(path) + ": " + strerror(errno);
        return false;
    }
    if (hex) {
        char tok[64];
        int c;
        while ((c = fgetc(f)) != EOF) {
            if (c == '#') {
                while (c != EOF && c != '\n') c = fgetc(f);
                continue;
            }
            if (isspace(c) || c == ',') continue;
            size_t n = 0;
            while (c != EOF && !isspace(c) && c != ',' && c != '#' && n + 1 < sizeof(tok)) {
                tok[n++] = (char) c;
                c = fgetc(f);
            }
            tok[n] = 0;
            if (c == '#') ungetc(c, f);
            char * end = nullptr;
            errno = 0;
            const unsigned long long v = strtoull(tok, &end, 16);
            if (errno || end == tok || *end || v > 0xFFFFFFFFull) {
                err = std::string(path ? path : "stdin") + ": \"" + tok + "\" is not a 32-bit hex word";
                if (f != stdin) fclose(f);
                return false;
            }
            w.push_back((uint32_t) v);
        }
    } else {
        uint8_t b[4];
        size_t n;
        while ((n = fread(b, 1, 4, f)) == 4) w.push_back((uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24));
        if (n != 0) {
            err = std::string(path ? path : "stdin") + ": length is not a multiple of 4 bytes";
            if (f != stdin) fclose(f);
            return false;
        }
    }
    if (f != stdin) fclose(f);
    return true;
}

const char * default_image() {
    const char * e = getenv("TSR_IMAGE");
    if (e && *e) return e;
    FILE * f = fopen(TSR_FW_DEFAULT_IMAGE, "rb");
    if (f) {
        fclose(f);
        return TSR_FW_DEFAULT_IMAGE;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Framing (DB §6.1) and decoding
// ---------------------------------------------------------------------------
bool header_ok(uint32_t h) {
    return TSR_GET(h, TSR_LOG_HDR_SYNC) == TSR_LOG_HDR_SYNC_MAGIC && TSR_GET(h, TSR_LOG_HDR_TYPE) <= TSR_LOG_HDR_TYPE_BLOB &&
           TSR_GET(h, TSR_LOG_HDR_LEVEL) <= TSR_LOG_HDR_LEVEL_TRACE;
}

struct decoded {
    size_t      at;      // word index of the header
    bool        ok;      // a record (else a malformed one: line says what)
    uint32_t    level;
    std::string line;
};

// Splits words into records and decodes each one with the transport's decoder. A bad header or a
// record cut short is reported; decoding resumes at the next valid header.
std::vector<decoded> decode_all(const tsr_image * img, const uint32_t * w, size_t n) {
    std::vector<decoded> out;
    size_t i = 0;
    char b[160];
    while (i < n) {
        const uint32_t h = w[i], N = TSR_GET(h, TSR_LOG_HDR_N);
        if (!header_ok(h) || i + 1 + N > n) {
            size_t j = i + 1;
            while (j < n && !header_ok(w[j])) ++j;
            if (!header_ok(h))
                snprintf(b, sizeof(b), "word 0x%08x is not a record header (SYNC 0x%x, TYPE %u, LEVEL %u); %zu word(s) skipped", h,
                         TSR_GET(h, TSR_LOG_HDR_SYNC), TSR_GET(h, TSR_LOG_HDR_TYPE), TSR_GET(h, TSR_LOG_HDR_LEVEL), j - i);
            else
                snprintf(b, sizeof(b), "record 0x%08x of %u payload words cut short (%zu word(s) left)", h, N, n - i - 1);
            out.push_back({ i, false, 0, b });
            if (header_ok(h)) break;   // cut short: the end of the data
            i = j;
            continue;
        }
        bool ok = true;
        std::string l = tsr::log_decoder::format(img, w + i, 1u + N, &ok);
        out.push_back({ i, ok, TSR_GET(h, TSR_LOG_HDR_LEVEL), l });
        i += 1 + N;
    }
    return out;
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------
int g_fail = 0;
void st(bool ok, const char * what, const std::string & got = std::string()) {
    if (!ok) {
        g_fail++;
        fprintf(stderr, "tsr-log selftest FAIL: %s%s%s\n", what, got.empty() ? "" : ": ", got.c_str());
    }
}

uint32_t hdr(uint32_t n, uint32_t level, uint32_t source, uint32_t type, uint32_t lseq) {
    return TSR_PUT(TSR_LOG_HDR_N, n) | TSR_PUT(TSR_LOG_HDR_LEVEL, level) | TSR_PUT(TSR_LOG_HDR_SOURCE, source) |
           TSR_PUT(TSR_LOG_HDR_TYPE, type) | TSR_LOG_HDR_TS | TSR_PUT(TSR_LOG_HDR_LSEQ, lseq) |
           TSR_PUT(TSR_LOG_HDR_SYNC, TSR_LOG_HDR_SYNC_MAGIC);
}

bool has(const std::string & s, const char * sub) { return s.find(sub) != std::string::npos; }

int selftest(const char * image) {
    if (image == nullptr) {
        printf("tsr-log selftest: no image (pass IMAGE or set TSR_IMAGE): skipped\n");
        return 77;
    }
    tsr_elf * e = nullptr;
    char err[256];
    if (tsr_elf_read(image, &e, err, sizeof(err)) != 0) {
        fprintf(stderr, "tsr-log selftest: %s: %s\n", image, err);
        return 2;
    }
    tsr_image * img = nullptr;
    if (tsr_image_open(image, &img) != 0) {
        fprintf(stderr, "tsr-log selftest: tsr_image_open(%s) failed\n", image);
        tsr_elf_free(e);
        return 2;
    }
    uint32_t a_info = 0, a_warn = 0;
    for (uint32_t i = 0; i < tsr_elf_logfmt_count(e); ++i) {
        uint32_t a = 0;
        const char * s = tsr_elf_logfmt_at(e, i, &a);
        if (s && strcmp(s, "selftest 9: seed %x, id %u") == 0) a_info = a;
        if (s && strcmp(s, "selftest 9: %u %x") == 0) a_warn = a;
    }
    st(a_info != 0 && a_warn != 0, "the image's .tsr_logfmt has the selftest kernel's id 9 strings");
    const int kidx = tsr_elf_kernel_find(e, "selftest");
    const tsr_elf_kernel * k = kidx >= 0 ? tsr_elf_kernel_at(e, (uint32_t) kidx) : nullptr;
    st(k != nullptr, "the image has a selftest kernel");
    const uint32_t seed = 0x9D9C0DEu;
    std::vector<uint32_t> w;
    // 0, 1: the two FMT records of selftest id 9 (sw/ctrl/kernels/selftest.cpp), LSEQ 0x12
    w.insert(w.end(), { hdr(4, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 0x12), 1000u, a_info, seed, 9u });
    w.insert(w.end(), { hdr(4, TSR_LOG_HDR_LEVEL_WARN, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 0x12), 1001u, a_warn, ~seed, seed + 1u });
    // 2: TEXT "hello, log" (10 bytes: three words, zero-padded)
    w.insert(w.end(), { hdr(4, TSR_LOG_HDR_LEVEL_DEBUG, TSR_LOG_HDR_SOURCE_RUNTIME, TSR_LOG_HDR_TYPE_TEXT, 0), 1002u, 0x6c6c6568u,
                        0x6c202c6fu, 0x0000676fu });
    // 3: "dropped 3"
    w.insert(w.end(), { hdr(3, TSR_LOG_HDR_LEVEL_WARN, TSR_LOG_HDR_SOURCE_RUNTIME, TSR_LOG_HDR_TYPE_EVENT, 0), 1003u, 0xF0000001u, 3u });
    // 4: BLOB tag 0x1234, words 1, 2, 3
    w.insert(w.end(), { hdr(5, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_DEBUG, TSR_LOG_HDR_TYPE_BLOB, 7), 1004u, 0x1234u, 1u, 2u, 3u });
    // 5: TRAP: illegal instruction at the selftest kernel's entry + 4
    {
        std::vector<uint32_t> c(TSR_CRASH_LOG_WORDS, 0u);
        c[TSR_CRASH_MAGIC_WORD] = TSR_CRASH_MAGIC;
        c[TSR_CRASH_VERSION_WORD] = 1;
        c[TSR_CRASH_MCAUSE_WORD] = 2;
        c[TSR_CRASH_MEPC_WORD] = k ? k->entry + 4 : 0;
        c[TSR_CRASH_MTVAL_WORD] = 0xDEADBEEFu;
        c[TSR_CRASH_SEQ_WORD] = 0x34;
        c[TSR_CRASH_KERNEL_WORD] = (uint32_t) (kidx < 0 ? 0 : kidx);
        c[TSR_CRASH_RT_STATE_WORD] = TSR_RT_STATE_PHASE_KERNEL;
        w.push_back(hdr(1 + TSR_CRASH_LOG_WORDS, TSR_LOG_HDR_LEVEL_FATAL, TSR_LOG_HDR_SOURCE_TRAP, TSR_LOG_HDR_TYPE_TRAP, 0x34));
        w.push_back(1005u);
        w.insert(w.end(), c.begin(), c.end());
    }
    // 6: a word with SYNC 5 (not a header): reported, skipped to the next header
    const size_t bad_at = w.size();
    w.push_back(0x5000BEEFu);
    // 7: a FMT record pointing outside .tsr_logfmt (a firmware log bug, framing intact)
    w.insert(w.end(), { hdr(3, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 0), 1006u, 0x12345678u, 1u });
    // 8: a good record after them
    w.insert(w.end(), { hdr(3, TSR_LOG_HDR_LEVEL_WARN, TSR_LOG_HDR_SOURCE_RUNTIME, TSR_LOG_HDR_TYPE_EVENT, 0), 1007u, 0xF0000001u, 1u });
    // 9: a record cut short at the end
    w.insert(w.end(), { hdr(5, TSR_LOG_HDR_LEVEL_INFO, TSR_LOG_HDR_SOURCE_KERNEL, TSR_LOG_HDR_TYPE_FMT, 0), 1008u, a_info });

    const std::vector<decoded> d = decode_all(img, w.data(), w.size());
    st(d.size() == 10, "10 lines", std::to_string(d.size()));
    char buf[192];
    if (d.size() == 10) {
        snprintf(buf, sizeof(buf), "[tsr INFO kernel seq 18 t 1000] selftest 9: seed %x, id %u", seed, 9u);
        st(d[0].ok && d[0].line == buf, "FMT INFO", d[0].line);
        snprintf(buf, sizeof(buf), "[tsr WARN kernel seq 18 t 1001] selftest 9: %u %x", ~seed, seed + 1u);
        st(d[1].ok && d[1].line == buf, "FMT WARN", d[1].line);
        st(d[2].ok && d[2].line == "[tsr DEBUG runtime seq 0 t 1002] hello, log", "TEXT", d[2].line);
        st(d[3].ok && has(d[3].line, "dropped 3 log record(s)"), "EVENT dropped", d[3].line);
        st(d[4].ok && has(d[4].line, "tag 0x00001234 0x00000001 0x00000002 0x00000003"), "BLOB", d[4].line);
        st(d[5].ok && has(d[5].line, "illegal instruction (mcause 2) at ") && has(d[5].line, "selftest+0x4") &&
               has(d[5].line, "mtval 0xdeadbeef") && has(d[5].line, "launch seq 52"),
           "TRAP with the kernel's symbol", d[5].line);
        st(!d[6].ok && has(d[6].line, "not a record header") && has(d[6].line, "1 word(s) skipped") && d[6].at == bad_at, "bad SYNC",
           d[6].line);
        st(!d[7].ok && has(d[7].line, "outside .tsr_logfmt"), "FMT address check", d[7].line);
        st(d[8].ok && has(d[8].line, "dropped 1 log record(s)"), "resynchronized after the bad records", d[8].line);
        st(!d[9].ok && has(d[9].line, "cut short"), "record cut short at the end", d[9].line);
    }
    // Without an image: FMT shown as the address and its words, no address check.
    const std::vector<decoded> d2 = decode_all(nullptr, w.data(), 10);
    st(d2.size() == 2 && d2[0].ok && has(d2[0].line, "<fmt 0x"), "no image", d2.empty() ? "" : d2[0].line);
    // The --hex reader (written to a temporary file).
    {
        const char * td = getenv("TMPDIR");
        std::string tmpl = std::string(td && *td ? td : "/tmp") + "/tsr-log-selftest-XXXXXX";
        std::vector<char> pb(tmpl.begin(), tmpl.end());
        pb.push_back(0);
        char * path = pb.data();
        const int fd = mkstemp(path);
        FILE * f = fd >= 0 ? fdopen(fd, "w") : nullptr;
        if (f) {
            fprintf(f, "# two words\n0x%08x %08x,\n0x%08x 0x%08x 0x%08x\n", w[0], w[1], w[2], w[3], w[4]);
            fclose(f);
            std::vector<uint32_t> r;
            std::string rerr;
            st(read_words(path, true, r, rerr) && r.size() == 5 && std::equal(r.begin(), r.end(), w.begin()), "--hex reader", rerr);
            unlink(path);
        } else {
            st(false, "temporary file for the --hex reader");
        }
    }
    tsr_image_close(img);
    tsr_elf_free(e);
    printf("tsr-log selftest: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}

void usage() {
    fprintf(stderr, "usage: tsr-log [--elf IMAGE | --no-elf] [--hex] [--level N] [-v] [FILE | -]...\n"
                    "       tsr-log selftest [IMAGE]\n");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "selftest") == 0) return selftest(argc >= 3 ? argv[2] : default_image());
    const char * image = nullptr;
    bool no_elf = false, hex = false, verbose = false;
    int max_level = 15;
    std::vector<const char *> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--elf" && i + 1 < argc) image = argv[++i];
        else if (a == "--no-elf") no_elf = true;
        else if (a == "--hex") hex = true;
        else if (a == "--level" && i + 1 < argc) max_level = atoi(argv[++i]);
        else if (a == "-v") verbose = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a.size() > 1 && a[0] == '-') { usage(); return 2; }
        else files.push_back(argv[i]);
    }
    if (files.empty()) files.push_back("-");
    tsr_image * img = nullptr;
    if (!no_elf) {
        if (image == nullptr) image = default_image();
        if (image != nullptr && tsr_image_open(image, &img) != 0) {
            fprintf(stderr, "tsr-log: %s: not a usable firmware image\n", image);
            return 2;
        }
        if (img == nullptr) fprintf(stderr, "tsr-log: no ELF (--elf, TSR_IMAGE): FMT records are shown as addresses\n");
    }
    size_t n_rec = 0, n_bad = 0, n_words = 0;
    for (const char * f : files) {
        std::vector<uint32_t> w;
        std::string err;
        if (!read_words(f, hex, w, err)) {
            fprintf(stderr, "tsr-log: %s\n", err.c_str());
            if (img) tsr_image_close(img);
            return 2;
        }
        n_words += w.size();
        for (const decoded & d : decode_all(img, w.data(), w.size())) {
            if (!d.ok) {
                printf("tsr-log: bad record at word %zu: %s\n", d.at, d.line.c_str());
                n_bad++;
                continue;
            }
            n_rec++;
            if ((int) d.level > max_level) continue;
            if (verbose) printf("%6zu ", d.at);
            printf("%s\n", d.line.c_str());
        }
    }
    fprintf(stderr, "tsr-log: %zu words, %zu records, %zu malformed%s%s\n", n_words, n_rec, n_bad, img ? ", formats from " : "",
            img ? image : "");
    if (img) tsr_image_close(img);
    return n_bad ? 1 : 0;
}

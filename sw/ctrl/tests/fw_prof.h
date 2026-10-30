// fw_prof.h — RISC-V profile from the tsr_rv retire stream, for the firmware perf harnesses
// (fw_m6_perf in test_fw_m6_e2e.cpp, fw_exec_perf in test_fw_exec.cpp; docs/v2/06-software-
// verification.md "SW" §7.8). Header-only C++17, host side.
//
// Every retired instruction's cost is the RV-cycle delta since the previous retire (tsr_rv_cycles:
// the class cost of SW §4.2 plus its bus latency); deltas above 256 are sleeps / fast-forwarded
// waits and are dropped, as the harnesses' busy counters do. The cost is attributed to
//   - the instruction's PC (a per-PC histogram; the ELF symbol table maps it to a function),
//   - its access class: the control-window register it reads or writes (ISSUE_W*, ISSUE_PUSH,
//     ISSUE_TICKET, ISSUE_CREDIT, DONE_SEQ, ...), the core window, the SPM window (exact /
//     line-buffered alias), DTCM / ITCM loads and stores, or the instruction class when it makes no
//     data access (ALU, taken branch / jump, mul, div / rem).
// fw_prof::diff(big, small) subtracts two snapshots, so a harness that runs a list at two sizes
// reports the per-op cost of each function and class with the launch's fixed cost cancelled.
// TSR_FW_PROF_DUMP=<path> makes report() also write the per-PC histogram ("pc cycles count" per
// line, hex pc) for a per-source-line breakdown with riscv-none-elf-addr2line (fw_prof_lines.py).
#pragma once

#include <cxxabi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tsr_rv.h"

namespace fwprof {

// Access / instruction classes (C_SPIN: an instruction of a wait loop after its first test, see
// spin_filter).
enum cls : int {
    C_ALU = 0, C_BRANCH, C_MUL, C_DIV, C_DTCM_LD, C_DTCM_ST, C_ITCM_LD,
    C_CW_ISSUE_W, C_CW_DEP, C_CW_PUSH, C_CW_TICKET, C_CW_CREDIT, C_CW_PIDX, C_CW_DONE_SEQ, C_CW_ERR,
    C_CW_QSTATUS, C_CW_MARK, C_CW_EX, C_CW_OTHER, C_CORE, C_SPM_LD, C_SPM_ST, C_SPM_LB_LD, C_OTHER_MEM,
    C_SPIN, C_N
};

// Wait loops (credit waits in issue(), DONE_SEQ spins in join(), QSTATUS polls) are waiting, not
// work: the simulator fast-forwards a spin after its 4th identical read (SW §4.4), but the reads
// before that, and the loop instructions around them, retire with small cycle deltas. A read of a
// timing register (ISSUE_CREDIT, DONE_SEQ, QSTATUS) that returns the same value as the previous
// such read, with no other data access in between, is a spin iteration; it and the instructions
// after it are spin until the next other data access. The first test of a wait is work.
struct spin_filter {
    uint32_t last_off = ~0u, last_val = 0;
    bool spinning = false;
    static bool timing_reg(uint32_t off) { return off == 0x030u || (off >= 0x040u && off < 0x080u) || off == 0x144u; }
    bool step(const tsr_rv_retire * r) {
        if (r->mem_size != 0u && !r->trap) {
            const uint32_t a = r->mem_addr;
            if (!r->mem_write && a >= 0x20000000u && a < 0x20001000u && timing_reg(a - 0x20000000u)) {
                const uint32_t off = a - 0x20000000u;
                spinning = off == last_off && r->mem_val == last_val;
                last_off = off;
                last_val = r->mem_val;
                return spinning;
            }
            last_off = ~0u;
            spinning = false;
            return false;
        }
        return spinning;
    }
};
inline const char * cls_name(int c) {
    static const char * const n[C_N] = {
        "ALU / CSR / not-taken branch", "taken branch / jump", "mul", "div / rem", "DTCM load", "DTCM store",
        "ITCM load", "CW ISSUE_W0-W7 store", "CW ISSUE_DEP0/1 store", "CW ISSUE_PUSH", "CW ISSUE_TICKET load",
        "CW ISSUE_CREDIT load", "CW ISSUE_PIDX load", "CW DONE_SEQ load", "CW ERR_* load", "CW QSTATUS load",
        "CW TRACE_MARK store", "CW EX (LQ/LCQ/log/RT_*)", "CW other", "core window load", "SPM exact load",
        "SPM store", "SPM line-buffer load", "other memory", "spin wait (excluded from work)" };
    return (c >= 0 && c < C_N) ? n[c] : "?";
}

inline int classify(const tsr_rv_retire * r) {
    if (r->mem_size != 0u && !r->trap) {
        const uint32_t a = r->mem_addr;
        const bool st = r->mem_write != 0u;
        if (a >= 0x20000000u && a < 0x20001000u) {
            const uint32_t o = a - 0x20000000u;
            if (o < 0x020u) return C_CW_ISSUE_W;
            if (o < 0x028u) return C_CW_DEP;
            if (o == 0x028u) return C_CW_PUSH;
            if (o == 0x02Cu) return C_CW_TICKET;
            if (o == 0x030u) return C_CW_CREDIT;
            if (o == 0x03Cu) return C_CW_PIDX;
            if (o >= 0x040u && o < 0x080u) return C_CW_DONE_SEQ;
            if ((o >= 0x0C0u && o < 0x140u) || (o >= 0x180u && o < 0x1A0u)) return C_CW_ERR;
            if (o == 0x144u) return C_CW_QSTATUS;
            if (o == 0x178u) return C_CW_MARK;
            if (o >= 0x200u && o < 0x300u) return C_CW_EX;
            return C_CW_OTHER;
        }
        if (a >= 0x20001000u && a < 0x20002000u) return C_CORE;
        if (a >= 0x40000000u && a < 0x40400000u) return st ? C_SPM_ST : C_SPM_LD;
        if (a >= 0x50000000u && a < 0x50400000u) return st ? C_SPM_ST : C_SPM_LB_LD;
        if (a >= 0x00100000u && a < 0x00200000u) return st ? C_DTCM_ST : C_DTCM_LD;
        if (a < 0x00100000u) return C_ITCM_LD;
        return C_OTHER_MEM;
    }
    const uint32_t i = r->insn;
    if (r->len == 4u) {
        const uint32_t opc = i & 0x7Fu;
        if (opc == 0x33u && ((i >> 25) & 0x7Fu) == 1u) return ((i >> 12) & 7u) >= 4u ? C_DIV : C_MUL;
    }
    if (r->next_pc != r->pc + r->len) return C_BRANCH;
    return C_ALU;
}

struct sym {
    uint32_t lo, hi;
    std::string name;
    uint32_t bind;   // STB_GLOBAL 1, STB_WEAK 2, STB_LOCAL 0
};

struct stat2 {
    double cyc = 0, n = 0;
};

class prof {
public:
    // The ELF's function symbols (STT_FUNC with a size). Returns false if the file is unreadable.
    bool load_symbols(const char * elf_path) {
        syms_.clear();
        FILE * f = fopen(elf_path, "rb");
        if (!f) return false;
        std::vector<uint8_t> b;
        uint8_t buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) b.insert(b.end(), buf, buf + n);
        fclose(f);
        auto u16 = [&](size_t o) { return o + 2 <= b.size() ? uint32_t(b[o] | b[o + 1] << 8) : 0u; };
        auto u32 = [&](size_t o) {
            return o + 4 <= b.size() ? uint32_t(b[o] | b[o + 1] << 8 | b[o + 2] << 16 | uint32_t(b[o + 3]) << 24) : 0u;
        };
        if (b.size() < 52 || memcmp(b.data(), "\x7f" "ELF", 4) != 0 || b[4] != 1) return false;
        const uint32_t shoff = u32(32), shentsize = u16(46), shnum = u16(48);
        for (uint32_t s = 0; s < shnum; ++s) {
            const size_t sh = size_t(shoff) + size_t(s) * shentsize;
            if (u32(sh + 4) != 2u) continue;            // SHT_SYMTAB
            const uint32_t off = u32(sh + 16), size = u32(sh + 20), link = u32(sh + 24), ent = u32(sh + 36);
            const size_t strsh = size_t(shoff) + size_t(link) * shentsize;
            const uint32_t stroff = u32(strsh + 16);
            for (uint32_t k = 0; ent && k < size / ent; ++k) {
                const size_t e = size_t(off) + size_t(k) * ent;
                const uint32_t name = u32(e), value = u32(e + 4), sz = u32(e + 8);
                const uint32_t info = b[e + 12];
                if ((info & 0xFu) != 2u || sz == 0u) continue;   // STT_FUNC
                const char * nm = reinterpret_cast<const char *>(b.data() + stroff + name);
                int st = 0;
                char * dm = abi::__cxa_demangle(nm, nullptr, nullptr, &st);
                std::string s = (st == 0 && dm) ? dm : nm;
                free(dm);
                syms_.push_back(sym{value & ~1u, (value & ~1u) + sz, s, info >> 4});
            }
        }
        std::sort(syms_.begin(), syms_.end(), [](const sym & a, const sym & b2) { return a.lo < b2.lo; });
        return !syms_.empty();
    }

    // spin: the instruction belongs to a wait loop (spin_filter); it is counted in C_SPIN and
    // not attributed to its PC, so the function and line tables show work only.
    void add(const tsr_rv_retire * r, uint64_t d, bool spin = false) {
        const int c = spin ? C_SPIN : classify(r);
        cls_[c].cyc += double(d);
        cls_[c].n += 1.0;
        if (spin) return;
        pc_[r->pc].cyc += double(d);
        pc_[r->pc].n += 1.0;
    }
    void clear() {
        pc_.clear();
        for (auto & c : cls_) c = stat2{};
    }

    // this - small (per-PC and per-class), into out.
    static prof diff(const prof & big, const prof & small) {
        prof o;
        o.syms_ = big.syms_;
        o.pc_ = big.pc_;
        for (const auto & kv : small.pc_) {
            o.pc_[kv.first].cyc -= kv.second.cyc;
            o.pc_[kv.first].n -= kv.second.n;
        }
        for (int c = 0; c < C_N; ++c) {
            o.cls_[c].cyc = big.cls_[c].cyc - small.cls_[c].cyc;
            o.cls_[c].n = big.cls_[c].n - small.cls_[c].n;
        }
        return o;
    }

    // The binding of the function whose demangled name starts with `prefix` (STB_GLOBAL 1, STB_WEAK
    // 2, ...), or -1 if there is none: a W2b kind handler that is still zz_kind_stubs.cpp's weak stub
    // reads 2 (the harnesses report such a kind as not in the image).
    int binding_of(const char * prefix) const {
        const size_t n = strlen(prefix);
        for (const auto & s : syms_)
            if (s.name.compare(0, n, prefix) == 0) return int(s.bind);
        return -1;
    }

    const char * func_of(uint32_t pc) const {
        auto it = std::upper_bound(syms_.begin(), syms_.end(), pc, [](uint32_t v, const sym & s) { return v < s.lo; });
        if (it == syms_.begin()) return "?";
        --it;
        return pc < it->hi ? it->name.c_str() : "?";
    }

    // Prints the per-function and per-class tables, each cost divided by `per` (e.g. the op count
    // of a difference); top = rows of the function table. With TSR_FW_PROF_DUMP the per-PC
    // histogram is written to that file (tag appended).
    void report(FILE * out, const char * title, double per, int top = 16, const char * tag = "") const {
        double total = 0, tn = 0;
        for (const auto & kv : pc_) { total += kv.second.cyc; tn += kv.second.n; }
        if (per <= 0) per = 1;
        fprintf(out, "%s: %.1f RV cycles, %.1f instructions per unit of work (%.0f units; spin waits: %.1f cycles, "
                "%.1f instructions)\n", title, total / per, tn / per, per, cls_[C_SPIN].cyc / per, cls_[C_SPIN].n / per);
        std::map<std::string, stat2> fn;
        for (const auto & kv : pc_) {
            stat2 & s = fn[func_of(kv.first)];
            s.cyc += kv.second.cyc;
            s.n += kv.second.n;
        }
        std::vector<std::pair<std::string, stat2>> v(fn.begin(), fn.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.cyc > b.second.cyc; });
        fprintf(out, "  %-64s %9s %8s %6s\n", "function (self)", "cycles", "insns", "share");
        int k = 0;
        for (const auto & e : v) {
            if (k++ >= top || (e.second.cyc / per < 0.5 && k > 4)) break;
            std::string nm = e.first;
            if (nm.size() > 64) nm = nm.substr(0, 61) + "...";
            fprintf(out, "  %-64s %9.1f %8.1f %5.1f%%\n", nm.c_str(), e.second.cyc / per, e.second.n / per,
                    total > 0 ? 100.0 * e.second.cyc / total : 0.0);
        }
        fprintf(out, "  %-64s %9s %8s %6s\n", "access / instruction class", "cycles", "insns", "share");
        for (int c = 0; c < C_N; ++c) {
            if (cls_[c].n / per < 0.05 && cls_[c].cyc / per < 0.5) continue;
            fprintf(out, "  %-64s %9.1f %8.1f %5.1f%%\n", cls_name(c), cls_[c].cyc / per, cls_[c].n / per,
                    total > 0 ? 100.0 * cls_[c].cyc / total : 0.0);
        }
        if (const char * p = getenv("TSR_FW_PROF_DUMP")) {
            std::string path = std::string(p) + tag;
            if (FILE * f = fopen(path.c_str(), "w")) {
                fprintf(f, "# %s; per = %.0f\n", title, per);
                std::vector<std::pair<uint32_t, stat2>> pv(pc_.begin(), pc_.end());
                std::sort(pv.begin(), pv.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
                for (const auto & e : pv)
                    if (e.second.n != 0) fprintf(f, "%08x %.0f %.0f\n", e.first, e.second.cyc, e.second.n);
                fclose(f);
                fprintf(out, "  per-PC histogram: %s\n", path.c_str());
            }
        }
    }

    double cycles() const {
        double t = 0;
        for (const auto & kv : pc_) t += kv.second.cyc;
        return t;
    }

private:
    std::vector<sym> syms_;
    std::unordered_map<uint32_t, stat2> pc_;
    stat2 cls_[C_N];
};

}  // namespace fwprof

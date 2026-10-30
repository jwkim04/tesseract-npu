// tsr_image_int.h — private: the firmware image behind tsr_image (tsr_image.cpp) as the protocol
// layer, the log decoder and the tests use it (docs/v2/01-execution-model.md "EX" §3;
// docs/v2/06-software-verification.md "SW" §6.2; docs/v2/05-debug.md "DB" §6.3).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tsr_transport.h"

struct tsr_elf;

namespace tsr {

// One PT_LOAD segment as the loader writes it (EX §3.1, §3.3): the padded words and their CRC-32.
struct image_seg {
    uint32_t             paddr = 0;
    bool                 itcm = false;     // in ITCM [0, ITCM_BYTES); else DTCM (0x0010_0000 +)
    std::vector<uint8_t> bytes;            // p_filesz padded with zeros to a whole word
    uint32_t             crc = 0;          // zlib crc32 of bytes (EX §3.3)
};

// .note.tsr (EX §3.1, SW §6.2): what the image requires of the device.
struct image_note {
    uint32_t sdk_abi = 0, ctl_id_min = 0, itcm_min = 0, dtcm_min = 0, caps_req = 0, n_kernels = 0;
};

const std::vector<image_seg> & image_segments(const tsr_image * img);
image_note                     image_get_note(const tsr_image * img);
const tsr_elf *                image_elf(const tsr_image * img);
uint32_t                       image_n_kernels(const tsr_image * img);
const char *                   image_kernel_name(const tsr_image * img, uint32_t index);   // "" if none

// Reference counting: a transport keeps its booted image alive (tsr_image_close drops one reference).
void image_ref(const tsr_image * img);
void image_unref(const tsr_image * img);

// pc -> "function+0x1c (file.cpp:123)" from .symtab (demangled) and .debug_line (DWARF 2-5), as
// far as the image has them; "0x0000xxxx" if neither knows pc.
std::string image_symbolize(const tsr_image * img, uint32_t pc);
// The DWARF line of pc alone: "file:line" or "" (tests compare it with addr2line).
std::string image_line(const tsr_image * img, uint32_t pc);

// zlib crc32 (reflected 0xEDB88320, init and final XOR 0xFFFFFFFF; EX §3.3).
uint32_t crc32(const void * data, size_t n);

// Test hooks: change what the host expects of the device (an image-id mismatch at HELLO, a
// .note.tsr requirement the device does not meet, a kernel count HELLO does not report: n above
// the table appends placeholder names, below it drops the last ones).
void image_test_set_id(tsr_image * img, uint64_t id);
void image_test_set_note(tsr_image * img, const image_note & note);
void image_test_set_n_kernels(tsr_image * img, uint32_t n);

} // namespace tsr

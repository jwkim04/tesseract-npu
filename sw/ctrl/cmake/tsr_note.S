/* sw/ctrl/cmake/tsr_note.S — the image's ".note.tsr" (EX §3.1, SW §6.2).
 *
 * Linked into every firmware image by sw/ctrl/CMakeLists.txt (not a runtime source). An ELF note,
 * not allocated (no PT_LOAD, so it never enters the image id), owner "TSR", type 1, 32-byte
 * descriptor of eight little-endian words:
 *   0 sdk_abi      Basis ABI (tsr/version.h TSR_SDK_ABI; HELLO r4[15:0] carries the same value)
 *   1 cap_min      the Tesseract Core capability the image needs (VERSION[15:0] encoding, major << 8
 *                  | minor; the build's VERSION_VALUE): same major as the device, device minor >= it
 *   2 itcm_min     ITCM bytes the image occupies (end of .text), compared with RV_CFG[15:0] KiB
 *   3 dtcm_min     DTCM bytes the image needs (the stack ends at the top of the DTCM it was
 *                  linked for, so this is that DTCM size), compared with RV_CFG[31:16] KiB
 *   4 caps_req     CAPS bits the image requires (EX §7.5)
 *   5 n_kernels    entries of .tsr_kernels
 *   6 basis_version  Tesseract Basis version (tsr/version.h TSR_BASIS_VERSION: major << 16 |
 *                  minor << 8 | patch; HELLO r6 carries the same value); 0 = built before versioning
 *   7              reserved, 0
 * Every value is an absolute symbol of sw/ctrl/rt/link.ld or of the build's --defsym (words 0, 1 and
 * 6, sw/ctrl/cmake/tsr_version.cmake), resolved by the static link (the relocations of a
 * non-allocated section are applied like those of DWARF). The host reads the note before loading
 * and refuses an image whose requirements the device does not meet; tsr-elf checks it after linking.
 */
    .section .note.tsr, "", @note
    .p2align 2
    .word   4                       /* n_namesz: "TSR\0" */
    .word   32                      /* n_descsz */
    .word   1                       /* n_type */
    .asciz  "TSR"
    .p2align 2
    .word   __tsr_sdk_abi
    .word   __tsr_cap_min
    .word   __tsr_itcm_min
    .word   __tsr_dtcm_min
    .word   __tsr_caps_req
    .word   __tsr_n_kernels
    .word   __tsr_basis_version
    .word   0

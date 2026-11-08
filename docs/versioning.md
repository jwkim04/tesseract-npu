# Versions

Tesseract has four version numbers. Each answers a separate question and each changes on its own.
The scheme is modelled on CUDA.

| Number | Answers | CUDA counterpart |
| --- | --- | --- |
| Tesseract Core capability, MAJOR.MINOR | What the hardware does. | compute capability |
| Tesseract Basis version, MAJOR.MINOR.PATCH | Which release of the firmware SDK and the host side this is. | toolkit version |
| Basis ABI, one integer | Whether a host and a firmware image speak the same protocol. | driver ABI |
| Kind numerics revision, one integer per op list kind | Whether a kind still produces the same bits. | none |

Everything described here runs on the simulator. No card has this design yet. Normative text:
[SW](design/06-software-verification.md) §6.9 and [EX](design/01-execution-model.md) §4.4 and §7.5.

## Where the numbers live

| Number | Source | In a firmware image | On a running device |
| --- | --- | --- | --- |
| Core capability | `sw/tables/tsr_tables.py` (`ISA_MAJOR`, `ISA_MINOR`) | `.note.tsr` word 1, `cap_min`: the capability the image needs | CSR `VERSION` (0x004) |
| Basis version | `sw/ctrl/include/tsr/version.h` | `.note.tsr` word 6 | HELLO record, r6 |
| Basis ABI | `sw/ctrl/include/tsr/version.h` (`TSR_SDK_ABI`) | `.note.tsr` word 0 | HELLO record, r4[15:0] |
| Kind revisions | `sw/ctrl/include/tsr/opl.hpp` (`kind_rev`) | section `.tsr_kinds` | HELLO record, r4[31:16]: a 16-bit signature of the kind table |

The build writes the Basis version, the ABI and the capability into every image. Each number has
one source.

## Tesseract Core capability

The capability is the number in the `VERSION` register. Bits 15:8 are the major, bits 7:0 the minor,
bits 31:16 are 0. The current design reads `0x0000_0100`, capability 1.0.

Software built for capability M.n runs on a device M.m with m at least n. It does not run on another
major.

The `CAPS` register (0x014) lists the optional features a configuration has: `VU`, `TOPK`, `ATT`,
`DQROW`, `VMASK`, `ATT_LOCAL`, `MVM_LOCAL_SCALE`, `MVM_LOCAL_SOFTCAP`. An optional feature is a `CAPS` bit, never a minor. The sizes of a configuration
(channels, lanes, `T_MAX`, scratchpad) are in `CFG0` to `CFG3`. They are not part of the capability.
Results are defined per configuration.

The number is set in `sw/tables/tsr_tables.py`. The table generator writes it into the C, C++,
SystemVerilog and Python headers, so the RTL, the simulator and the software read one value. The
hardware owner changes it in the same change that alters the RTL and the simulator.

- Major: an incompatible change of the hardware contract. This covers an instruction encoding, a
  CSR, queue or record field that software relies on, a removed opcode, the result bits of any
  tile op for the same inputs, and a smaller control-core ISA.
- Minor: a backward-compatible addition that every configuration of that minor has. Examples are a
  new opcode or unit, a new CSR field, and a control-core ISA extension the compiler may target.

It is shown as `Tesseract Core 1.0 (VU TOPK ATT DQROW)`: the capability, then the `CAPS` features.

## Tesseract Basis version

Basis is the firmware SDK in `sw/ctrl` and the host side that loads and drives it: the runtime, the
kernels, the op list format and its planners, the transport, the ggml backend and the tools. Its
version follows [Semantic Versioning](https://semver.org/).

- Major: an ABI break (next section). Not before 1.0.0.
- Minor: new kinds, kernels, kind variants or flags. A planner that accepts more shapes. Any kind
  revision bump. New host API or tools.
- Patch: fixes and performance work that leave every output bit of every kind, every interface and
  every planner's acceptance unchanged.

The versions stay 0.x until the M7 gate has passed on the card. Then comes 1.0.0. In 0.x an
incompatible change bumps the minor and the ABI. The first release is 0.1.0.

The version is a 32-bit value, `major << 16 | minor << 8 | patch`. Basis 0.1.0 is `0x0000_0100`.

The number in `version.h` changes at release, not with each commit. A build between two releases
carries the number of the last release (0.1.0 until the first release is out), so it is told apart
by its image id and its kind revisions, not by the Basis version alone.

## Basis ABI

The ABI is one integer, `TSR_SDK_ABI`, currently 1. It is defined in `sw/ctrl/include/tsr/version.h`
and nowhere else. It changes when an existing image and an existing host would stop working
together. It covers:

- `tsr_args`, `TSR_KERNEL` and the kernel table entry.
- The layout of `.note.tsr` and of a `.tsr_kinds` entry.
- The launch protocol as the runtime implements it: the descriptor, the record and HELLO layouts,
  and the runtime steps R1 to R10.
- The op list format: the header, the record layout, the argument words of every kind and what they
  mean, and the removal of a kind.
- A planner that accepts less than before. The host and the image share the planners (SW-20), so
  this is a break.

The author of the breaking change raises the ABI in that change, not at release. ABI 1 is defined
as of Basis 0.1.0.

## Kind numerics revision

Every op list kind has a revision from 1 to 255. A revision of 0 means the image does not have the
kind. The revision is raised when any output bit of the kind can change for the same record words,
the same input bytes and the same device configuration (`CFG0` to `CFG3`, `CAPS`) on the same
capability.

A result is therefore reproducible from (capability, configuration, kind, revision).

- No bump when the outputs stay identical: push order, scheduling, scratch use.
- No bump when the change only accepts new inputs, such as a new variant or larger shapes. That is a
  Basis minor.
- A new kind starts at 1. Revisions never decrease and are never reused.

The revisions are `opl::kind_rev(kind)` in `sw/ctrl/include/tsr/opl.hpp`, next to `kind_nargs`. An
image declares the kinds it provides with `TSR_KIND(NAME)` beside the handler. The macro emits a
4-byte `{kind, revision, 0}` entry into `.tsr_kinds`, so the host reads the table from the ELF
before it boots. The runtime hashes the revisions (FNV-1a over a 64-byte array, folded to 16 bits)
and posts the result in HELLO r4. The host computes the same value from the ELF and compares.

The test `fw_kind_revs` enforces the rule. It runs the production image on the simulator over a
fixed corpus of records per kind and compares a digest of the outputs with
`sw/ctrl/tests/kind_revs.txt`.

- A digest that moved at the same revision fails with "bump the revision".
- A revision that moved with the same digest fails with "revert the bump".
- A kind the image handles without declaring it, or declares without a handler, fails.

To bump a revision, change `kind_rev`, run `test_fw_kind_revs tsr-fw.elf
sw/ctrl/tests/kind_revs.txt --update`, and add a `CHANGELOG.md` line such as "ROPE r2: ...".

## Reading the numbers

tsr-selftest prints them in its identity and boot sections:

```sh
build/dev/bin/tsr-selftest --device sim
```

```
  ID 0x54535231  Tesseract Core capability 1.0 (VERSION 0x00000100), CAPS 0x00000069 (VU TOPK ATT DQROW)
  CTL_ID 0x00005d01 (control core, debug module, SPM bridge, log ring, watchdog, SPM alias)  ...
== boot (fw/tsr-fw.elf, id 21df766de1377d19)
  image fw/tsr-fw.elf: id 21df766de1377d19, Tesseract Basis 0.1.0, Basis ABI 1, needs capability 1.0; host Tesseract Basis 0.1.0
  kinds (19, signature 0x64dc): NOP r1 JOIN r1 MUL_MAT r1 ADD r1 MUL r1 SCALE r1 RMS_NORM r1 GLU r1 ROPE r1 GET_ROWS r1 SET_ROWS_KV r1 FLASH_ATTN r1 CPY r1 UNARY r1 MUL_MAT_SCALE r1 MUL_MAT_SOFTCAP r1 SOFTPLUS r1 SSM_CONV r1 GATED_DELTA_NET r1
```

The image id changes with every build. `tsr-selftest --version` prints the Basis version and ABI of
the tool.

tsr-elf reads them from an image without a device:

```sh
build/dev/sw/ctrl/host/tsr-elf --version
build/dev/sw/ctrl/host/tsr-elf note build/dev/fw/tsr-fw.elf     # sdk_abi, cap_min, basis_version, ...
build/dev/sw/ctrl/host/tsr-elf kinds build/dev/fw/tsr-fw.elf    # one line per kind, then the signature
```

```
0x10	MUL_MAT	rev=1
...
signature=0xfa6f
```

llama.cpp lists the device with both versions:

```
$ TSR_DEVICE=sim build/dev/bin/llama-cli --list-devices
Available devices:
  Tesseract0: Tesseract Core 1.0 (VU TOPK ATT DQROW; sim, C=4, LANES=128, T_MAX=1, K_MAX=32768, SPM=4096 KiB, 16 GiB/ch), Tesseract Basis 0.1.0 (ABI 1, 14 kinds, image 21df766de1377d19) (63488 MiB, 63488 MiB free)
```

The text up to the semicolon is the capability and its `CAPS` features. After it come the
configuration, the Basis version of the booted image, its ABI, the number of kinds and the image id.
The memory sizes at the end are added by llama.cpp. With `TSR_DEBUG=2` the backend logs why it
refuses each op, including "kind X is not in the booted image".

klcheck and numcheck record the capability, the Basis version and the kind table of a device run in
their output files, so a comparison says which versions produced the result (`.klc` header and
`report.json` `versions`; numcheck `meta.json` and `report.json` `versions`, with every kind's
revision).

In C and C++, `sw/ctrl/include/tsr/version.h` defines:

- `TSR_BASIS_VERSION_MAJOR`, `TSR_BASIS_VERSION_MINOR`, `TSR_BASIS_VERSION_PATCH`
- `TSR_BASIS_VERSION`, the packed 32-bit value, and `TSR_BASIS_MAJOR_OF(v)`,
  `TSR_BASIS_MINOR_OF(v)`, `TSR_BASIS_PATCH_OF(v)` to unpack one
- `TSR_SDK_ABI`

`tsr/rt.hpp` includes it. Firmware sources also use `tsr::opl::kind_rev(kind)` and
`tsr::opl::kind_name(kind)` from `tsr/opl.hpp`. The generated `tsr/mmio.hpp` and
`sw/sim/include/tsr_tables.h` hold the capability as `ISA_MAJOR`, `ISA_MINOR` and `VERSION_VALUE`
(`TSR_ISA_MAJOR` and so on in C).

A host program asks the transport (`sw/transport/tsr_transport.h`):

```c
tsr_versions v;
char s[256];
tsr_transport_versions(t, &v);          // the device's capability and CAPS, the booted image's versions
tsr_versions_str(&v, 1, s, sizeof s);   // 1: also the image id and every kind with its revision
puts(s);
```

`tsr_image_versions` does the same for an image that is not booted, and `tsr_basis_version()`
returns the version of the host library itself. `v.kind_rev[k]` is the revision of kind `k` in the
booted image.

## Compatibility

A host, a firmware image and a device work together when these checks pass. Each check refuses with
a message that names both numbers.

| Check | Where | Rule |
| --- | --- | --- |
| Device | `tsr_transport_open`, before any write | The `VERSION` major is the one the host supports (1), any minor. `CTL_ID` is not 0, which would mean no control core. The adapter version major and `NUM_CH` match. |
| Image | `tsr_image_open` | `.note.tsr` is present. Its ABI is `TSR_SDK_ABI`. Its Basis version is not 0, which marks an image built before versioning. `.tsr_kinds` is well formed. |
| Image on device | `tsr_transport_boot`, before any load | The image's `cap_min` has the device's major, and a minor not above the device's. `CAPS` has every bit the image requires. The memory sizes fit. The ABI matches. |
| HELLO | every cold and warm boot | The image id, the ABI, the kind table signature, the kernel count and the Basis version match the image file. |
| Backend and image | ggml-tesseract and `sw/rt`, when the device opens | The image's Basis version, major and minor, is not older than the backend's. The patch is ignored. A newer image is accepted. |
| Launch | `supports_op` and the op list builder | Only kinds with a nonzero revision in the booted image are sent. The executor still answers `BAD_RECORD` for a kind it lacks and `BAD_VERSION` for an op list format it does not know. |

The backend rule is the same as CUDA's rule for an insufficient driver. The host and the image
share the planners, so a backend that is newer than its image could send a shape the image refuses.
The transport does not enforce the rule, so captured launch traces of older images still replay.

An image that provides fewer kinds works. The ops it cannot run stay on the CPU. `TSR_KINDS=m6`
narrows the device to MUL_MAT the same way.

Examples of the messages:

```
Tesseract Core capability 2.0 (VERSION 0x00000200): this software supports capability 1.x
CTL_ID = 0: no control core
tsr_image: IMAGE: Basis ABI 2, this host implements ABI 1 (Tesseract Basis 0.1.0): rebuild the image with this tree's SDK
image IMAGE refused: the device does not meet its requirements ... Tesseract Core capability 1.1 (device 1.0);
HELLO does not match the image IMAGE: ... Tesseract Basis 0.2.0, expected 0.1.0;
```

## Identifiers that are not versions

These fields exist, are never shown as a product version, and are not compared with a release.

- `CTL_ID[7:0]` is the control-core revision, currently 1. 0 means there is no control core.
- The adapter version major (`ADAPTER_VERSION[31:16]`) is 1. The debug block's `DBG_ID[7:0]` is 1.
- The kind table signature and the 64-bit image id name a build.
- File formats carry their own numbers: the op list header (1), trace metadata (2), the `.klc`
  header (2), numcheck metadata (2), `DBG_INFO` (1), the crash record (1) and `MSUM` (1). They tell
  readers which layout a file has.

## Kinds at Basis 0.1.0

Op list kinds with their revisions, as declared by `TSR_KIND` in the file that defines the handler.
The argument words of each kind are in [SW](design/06-software-verification.md) §7.2.

| Kind | Name | Revision | Handler | Computes |
| --- | --- | --- | --- | --- |
| 0x00 | NOP | 1 | `graph.cpp` | Nothing. Placeholder that issues no tile ops. |
| 0x01 | JOIN | 1 | `graph.cpp` | Waits for two earlier ops. The builder adds it when an op has more than two predecessors. |
| 0x10 | MUL_MAT | 1 | `k_mulmat.cpp` | Matrix product with Q8_0 or Q4_K weights. An addend (bias or residual) and accumulation into the destination are optional. |
| 0x11 | ADD | 1 | `k_elem.cpp` | Element-wise add. The second operand repeats with a row period, or is one scalar per row. |
| 0x12 | MUL | 1 | `k_elem.cpp` | Element-wise multiply, with the same operand forms as ADD. |
| 0x13 | SCALE | 1 | `k_elem.cpp` | `x * s`, then `+ b` when b is not 0. |
| 0x14 | RMS_NORM | 1 | `k_norm.cpp` | RMS norm, with an optional fused weight multiply. |
| 0x15 | GLU | 1 | `k_glu.cpp` | Gated activation: SWIGLU, GEGLU16 (the fp16 GELU of the ggml CPU backend) or GEGLU (exact tanh form). |
| 0x16 | ROPE | 1 | `k_rope.cpp` | Rotary position embedding, normal or NEOX order, with an optional host-folded angle table. |
| 0x17 | GET_ROWS | 1 | `k_rows.cpp` | Gathers f32 rows by id. |
| 0x18 | SET_ROWS_KV | 1 | `k_kv.cpp` | Quantizes rows to q8_0 and writes them into the KV cache. |
| 0x19 | FLASH_ATTN | 1 | `k_fattn.cpp` | Attention over the KV cache with a mask summary and grouped-query heads. |
| 0x1A | CPY | 1 | `k_rows.cpp` | Copies f32 rows. |
| 0x1B | UNARY | 1 | `k_elem.cpp` | TANH, GELU16, SILU, EXP or GELU (exact tanh form). |
| 0x1C | MUL_MAT_SCALE | 1 | `k_mulmat.cpp` | A MUL_MAT whose result is multiplied by a scalar before it is stored. |
| 0x1D | MUL_MAT_SOFTCAP | 1 | `k_mulmat.cpp` | A MUL_MAT followed by the scale, tanh and scale of a logit soft cap. |
| 0x1E | MOE_ROUTE | 1 | `k_moe.cpp` | Router softmax, top-k selection and weight renormalisation of a MoE layer (SW §7.9, SW-61). |
| 0x1F | MUL_MAT_ID | 1 | `k_moe.cpp` | Matrix product against the experts that the ids of each token select, with an optional per-expert scale (SW-62). |

Kinds 0x20 to 0x3F are free. The MoE kinds took the lowest free numbers; the Gated DeltaNet kinds are numbered
downward from 0x3F. The signature of the table above (18 kinds) is `0xc341`; the 14 kinds up to UNARY have `0xfa6f`.
A table with only NOP, JOIN and MUL_MAT at revision 1 has `0x512e`, and an empty table `0xb51b`.

A kind has one revision for all its variants. Adding a variant of GLU is a Basis minor. Changing
the bits of an existing variant raises the revision of GLU.

## Releases

Each release has a Basis version, a section in `CHANGELOG.md` and an annotated git tag
`basis-vX.Y.Z` on the release commit.

1. Set `TSR_BASIS_VERSION_MAJOR`, `_MINOR` and `_PATCH` in `sw/ctrl/include/tsr/version.h`.
2. In `CHANGELOG.md`, replace "(unreleased)" in the heading of the section by the date.
3. In `README.md`, update the release badge to the new tag.
4. Build, run `ctest -LE slow`, and run `tsr-selftest --device sim`.
5. Commit, then `git tag -a basis-vX.Y.Z -m "Tesseract Basis X.Y.Z"` and push the tag.

Changes made after a release go into a new section headed "Basis X.Y.Z (unreleased)", where X.Y.Z
is the next version. The ABI and the kind revisions are not part of this step. Their authors change
them in the commit that needs it.

The sections of `CHANGELOG.md` give the ABI and the capability they target.

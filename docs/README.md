# Documentation

Read [getting-started.md](getting-started.md) to build and run Tesseract. Read
[design/README.md](design/README.md) for how the design documents fit together and for the two
execution modes, full mode and matmul mode.

When documents disagree, the order of authority is [spec.md](spec.md), the simulator in `sw/sim`,
the design documents in [design/](design/README.md), then the other documents.

## Index

- [getting-started.md](getting-started.md): requirements, build, running a model on the simulator, PyTorch and the card, tests, repository layout.
- [spec.md](spec.md): architecture, memory model, numeric rules, ISA, the ggml backend contract and the verification strategy.
- [design/README.md](design/README.md): the design set, the execution modes, the decision index and the implementation map.
- [design/01-execution-model.md](design/01-execution-model.md): launch protocol, CSR map, opcode and unit tables, error codes, PERF counters, transport API (EX).
- [design/02-vector-unit.md](design/02-vector-unit.md): the vector unit, its numerics and the fused Gated DeltaNet op (VU).
- [design/03-attention-kv.md](design/03-attention-kv.md): attention and the KV cache on the device (AT).
- [design/04-control-core.md](design/04-control-core.md): the RISC-V control core, issue path, clock domains and floorplan (CC).
- [design/05-debug.md](design/05-debug.md): debug module, GDB over PCIe, tile-level debug and the trace writer (DB).
- [design/06-software-verification.md](design/06-software-verification.md): software stack, firmware SDK, simulator, op list, backend and verification (SW).
- [design/07-graph-scheduler.md](design/07-graph-scheduler.md): Tesseract Pivot, the graph scheduler of the ggml backend: units, machine model, costs, search, adapter and guards (PV).
- [platform-u200.md](platform-u200.md): the XRT adapter, the kernels `tsr_u200_k1` and `tsr_u200_k4`, the `v++` flow and floorplan, card bring-up.
- [ctrl-riscv.md](ctrl-riscv.md): design record of the control core, with the decisions and the rejected alternatives.
- [rtl-core.md](rtl-core.md): RTL contract of the core infrastructure: conventions, interfaces, SPM, CSR block, QACT and the fp library.
- [rtl-mvm.md](rtl-mvm.md): RTL of the MVM unit: architecture, numerics, verification and synthesis results.
- [bandwidth-kernel.md](bandwidth-kernel.md): the DDR4 bandwidth kernel `tsr_bw_k4` and the host tool `tsr_bw`.
- [xrt-host.md](xrt-host.md): the host transport, the `xrt` kernel device, the card selftest and bring-up.
- [llama-pin.md](llama-pin.md): the pinned llama.cpp commit and the backend API items checked at it.
- [torch-backend.md](torch-backend.md): the PyTorch backend and the `tsr_rt` runtime.
- [versioning.md](versioning.md): Core capability and release, Basis version, Basis ABI and kind numerics revisions.
- [klcheck.md](klcheck.md): the model-level KL check against the CPU.
- [numcheck.md](numcheck.md): the one-prompt numeric check of every weight matrix product.
- [prof-dbg.md](prof-dbg.md): debugging and profiling tools, the profile report and `tsr-top`.
- [performance.md](performance.md): how to measure, what to report and the results.

Component documentation lives next to the code: [sw/ctrl](../sw/ctrl/README.md) (the Basis kernel
SDK), [sw/ggml-tesseract](../sw/ggml-tesseract/README.md), [sw/transport](../sw/transport/README.md),
[sw/sim](../sw/sim/README.md), [sw/rt](../sw/rt/README.md), [sw/torch](../sw/torch/README.md),
[sw/xrt-driver](../sw/xrt-driver/README.md) and [traces](../traces/README.md). The figure sources
and their style guide are in [fig/src](fig/src/STYLE.md).

## Supported features

- Any model that llama.cpp runs. The ggml ops that the device does not support run on the CPU through the ggml scheduler. The ops that run on the device are listed in the [README](../README.md#operators-and-features) and the op list kinds in [versioning.md](versioning.md).
- Dense decoders: Qwen2.5 and Gemma 4.
- Mixture of experts: Gemma 4 26B-A4B and Qwen3-30B-A3B. The router, the top-k selection and the experts run on the device.
- Gated DeltaNet hybrids: Qwen3.5 and Qwen3.8. The recurrent state lives in the device's cache cells.
- Multi-token prediction: the model's MTP head drafts and the device verifies a window of tokens in one launch, with rollback of the recurrent state.
- Weights: GGUF as shipped. Q8_0 and Q4_K run natively. Every other type is converted to Q8_0 at load.
- Attention: flash attention on the device with a Q8_0 KV cache, head dimensions 64, 128, 256 and 512.
- Execution: one launch per decode token in full mode. A prefill or verification pass takes up to `T_MAX` tokens per weight pass; the release image has `T_MAX` 4.
- PyTorch: `torch.compile(model, backend="tesseract")` runs the weight matmuls on the device. `QuantLinear` takes ggml Q8_0 and Q4_K blocks.
- Debugging and profiling: GDB over PCIe on the control core, `tsr-dbg`, `tsr-prof`, the profile report and `tsr-top`.
- Verification: the C simulator is the bit-exact oracle for the RTL, the firmware and the host stack. `klcheck` and `numcheck` check whole models.

## Known limitations

- Weights and KV cache are limited to 32 GiB on this U200: four DDR4 channels of 8 GiB. The design supports up to 16 GiB per channel (`CH_BYTES`, 64 GiB with four channels), and the transport detects the capacity of the card at open ([xrt-host.md](xrt-host.md) §4.4). A model that needs more than the card holds stops at load with the message `WEIGHT heap full`.
- There is no native path for 5-bit and 6-bit formats, I-quants, Q4_0 or MXFP4. Their weights are converted to Q8_0 at load.
- The KV cache is Q8_0. Other KV types run on the CPU with `-nkvo`. The attention rotation must be off (`LLAMA_ATTN_ROT_DISABLE=1`).
- Attention with ALiBi, logit softcap, sinks or K-shift is not supported on the device. Context shift and defragmentation of the device KV cache are refused.
- Flash attention for a prefill ubatch of more than 680 tokens runs on the CPU.
- A mixture-of-experts router runs on the device only for softmax gating with top-k selection and weights normalized by their sum. Other routers run on the CPU and the experts still run on the device.
- Prefill of a mixture-of-experts layer runs one matrix product per token and expert slot. Tokens are not grouped by expert.
- Gated DeltaNet and multi-token prediction are checked with one sequence per ubatch. Several sequences with snapshots in one ubatch, vision positions and the KDA gate run on the CPU or are not checked.
- A model that combines mixture-of-experts layers and Gated DeltaNet layers (Qwen3.6-35B-A3B) gives the CPU's
  text on the simulator and on the card with part of its layers on the device. Its Q8_0 file does not fit
  this card, and a run with every layer on the card has not been done.
- The host submits the graph of every token, reads the logits and samples. There is no resident decode loop on the device.
- One process owns the card at a time, and one card is used per process. Multi-card operation is not supported.

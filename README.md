# Tesseract

FPGA overlay core that runs llama.cpp's ggml graphs as an instruction stream (Alveo U200 first).
The contract is `docs/spec.md`; llama.cpp pin and verified backend API: `docs/llama-pin.md`.

```
git submodule update --init
third_party/apply-patches.sh          # ggml-tesseract hooks into the pinned llama.cpp
cmake -S . -B build/dev -DCMAKE_BUILD_TYPE=Release
cmake --build build/dev -j4
ctest --test-dir build/dev
TSR_DEVICE=sim build/dev/bin/test-backend-ops -o MUL_MAT -b Tesseract0
```

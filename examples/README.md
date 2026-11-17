# Custom kernel examples

These are custom kernels that use the Tesseract Basis SDK (`sw/ctrl`). Each one is a project of its own: it is
not part of the SDK, it is not in a Basis release or in the Basis tests, and it has no version of its own.
`cmake -S . -B build/dev` builds them unless `-DTSR_WITH_EXAMPLES=OFF`.

| Example | What it is |
| --- | --- |
| [`ising/`](ising) | an Ising machine: simulated bifurcation on a Sherrington-Kirkpatrick spin glass |

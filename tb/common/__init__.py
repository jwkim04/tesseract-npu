"""Shared Tesseract testbench infrastructure (see docs/rtl-m2.md §9).

isa         constants, encoders and struct layouts mirroring rtl/core/tsr_pkg.sv
tsr_ref     ctypes wrapper of libtsr_ref.so (bit-exact reference numerics)
trace       trace directory reader (tsr_trace.h semantics)
mem_model   cocotb per-channel memory model with protocol assertions
mvm_model   golden model of the MVM unit blocks (docs/rtl-m3.md): front beats, cluster, gather
simlib      the ISA simulator through ctypes (content-keyed shared library)
proggen     random programs (ProgGen; m3=True: MVM-oriented, no undefined overlaps)
csr_driver  CSR port driver
replay      trace replay + comparison on a tsr_core DUT (compare() usable on any model)
pycore      Python model of the core (M2 + MVM) over a real memory port (harness proof)
runner      build + run a cocotb test on Verilator with the project defaults
tbutil      clock/reset/seed helpers
"""

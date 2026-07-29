# Kernel implementation units

The CPU kernels are grouped by concrete responsibility:

- `cpu_quant.inc`: scalar conversion, quantization formats, and quantized dot
  products.
- `cpu_matmul.inc`: embedding lookup, dense matrix-vector operations, and
  routed-expert matrix products.

These are implementation fragments included exactly once by `ds4.c`. They
remain specialized for the supported model tensor layouts; this directory is
an ownership boundary, not a generic kernel API.

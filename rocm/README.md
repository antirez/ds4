# ROCm implementation units

This directory owns ROCm runtime code and low-level implementations reused by
model integrations: tensor storage, hipBLASLt matmul, quantization, embedding,
normalization/RoPE, shared-expert, and MoE launch paths.

Model-specific ROCm implementations live under `models/<model>/rocm/`.
`ds4_rocm.cu` includes the shared and model-owned headers into one translation
unit, so the ownership split introduces no dispatch or wrapper layer between a
model and its custom kernels.

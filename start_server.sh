#!/bin/bash
# 1. Clean system cache memory (Requires sudo)
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 2. Ensure no stale processes or locks
pkill -9 -x ds4-server || true
rm -f /tmp/ds4.lock

# 3. Use Zero-Copy UMA Mode (Direct access to RAM)
# For HIP, this is controlled by DS4_HIP_COPY_MODEL. Unsetting it enables zero-copy if possible.
unset DS4_HIP_COPY_MODEL

# 4. Set Prefill Chunk Size (Backend specific)
export DS4_HIP_PREFILL_CHUNK=4096

# 5. Start the optimized ds4-server with ROCm backend
# --rocm enables the ROCm/HIP graph backend.
# To use MTP, add: --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
exec ./ds4-server --rocm --ctx 65536 \
    --warm-weights \
    --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192

#!/bin/bash
# 1. Clean system cache memory (Requires sudo)
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 2. Ensure no stale processes or locks
pkill -9 -x ds4-server || true
rm -f /tmp/ds4.lock

# 3. Use Zero-Copy UMA Mode (Direct access to RAM)
unset DS4_CUDA_COPY_MODEL

# 4. Set Prefill Chunk Size to 4096 (Environment Variable)
export DS4_METAL_PREFILL_CHUNK=4096

# 5. Start the optimized ds4-server with MTP Speculative Decoding
# # --mtp enables multi-token prediction to push TPS past 15+
exec ./ds4-server --cuda --ctx 65536 \
    # --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
    # # --mtp-draft 1 \
    --warm-weights \
    --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192

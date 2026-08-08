#!/usr/bin/env python3
"""
Generate a compact local code-map without any network access.

Usage:
    python3 hy_extract_dspark_context.py ds4.c > /tmp/dspark_map.txt
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "ds4.c")
lines = path.read_text(errors="replace").splitlines()

symbols = {
    "DSpark main loop": "ds4_session_eval_dspark_speculative_argmax",
    "Batched verifier": "metal_graph_verify_suffix_tops_impl",
    "Exact N=2 verifier reference": "metal_graph_verify_decode2_exact",
    "Ordinary one-token decode": "metal_graph_eval_token_raw_swa",
    "Frontier snapshot": "spec_frontier_snapshot",
    "Frontier restore": "spec_frontier_restore",
    "Prefix-1 commit": "spec_frontier_commit_prefix1",
    "Canonical compressor projection/store": "ds4_gpu_matmul_f16_pair_compressor_store_tensor",
    "Generic F16 pair projection": "ds4_gpu_matmul_f16_pair_tensor",
    "Existing exact Q8 pair rows primitive": "ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor",
    "Existing exact Q8 rows primitive": "ds4_gpu_matmul_q8_0_decode_rows_exact_tensor",
    "Existing exact router rows primitive": "ds4_gpu_matmul_f16_router_rows_exact_tensor",
    "Existing exact shared rows helper": "metal_graph_encode_shared_rows_exact",
}

def hits(needle):
    return [i for i, line in enumerate(lines) if needle in line]

for title, sym in symbols.items():
    hs = hits(sym)
    print(f"\n=== {title}: {sym} ===")
    if not hs:
        print("NOT FOUND")
        continue
    for i in hs[:8]:
        lo = max(0, i - 5)
        hi = min(len(lines), i + 11)
        print(f"\n--- lines {lo+1}-{hi} ---")
        for j in range(lo, hi):
            print(f"{j+1:6d}: {lines[j]}")
    if len(hs) > 8:
        print(f"... {len(hs)-8} more hits omitted")

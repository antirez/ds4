#!/usr/bin/env bash
set -euo pipefail

FILE="${1:-ds4.c}"

if [[ ! -f "$FILE" ]]; then
  echo "usage: $0 [path/to/ds4.c]" >&2
  exit 2
fi

symbols=(
  "ds4_session_eval_dspark_speculative_argmax"
  "metal_graph_verify_suffix_tops_impl"
  "metal_graph_verify_suffix_tops"
  "metal_graph_verify_decode2_exact"
  "metal_graph_eval_token_raw_swa"
  "spec_frontier_snapshot"
  "spec_frontier_restore"
  "spec_frontier_commit_prefix1"
  "ds4_gpu_matmul_f16_pair_compressor_store_tensor"
  "ds4_gpu_matmul_f16_pair_tensor"
  "ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor"
  "ds4_gpu_matmul_q8_0_decode_rows_exact_tensor"
  "ds4_gpu_matmul_f16_router_rows_exact_tensor"
  "metal_graph_encode_shared_rows_exact"
)

echo "# ds4 DSpark/exactness symbol map"
echo "# file: $FILE"
echo

for s in "${symbols[@]}"; do
  echo "## $s"
  grep -n -F "$s" "$FILE" || true
  echo
done

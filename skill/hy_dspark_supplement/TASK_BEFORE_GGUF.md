# HY task before DSpark support GGUF is ready

1. Run:
   ```sh
   bash hy_locate_dspark_paths.sh ds4.c
   python3 hy_extract_dspark_context.py ds4.c > /tmp/dspark_context.txt
   ```

2. Read `metal_graph_verify_decode2_exact()` carefully and produce a one-page
   note describing which steps remain canonical/rowwise and which are batched.

3. Trace `metal_graph_verify_suffix_tops_impl()` ->
   `metal_graph_encode_layer_batch()` and produce the stage comparison table
   in `CURRENT_MAIN_REFERENCE.md`.

4. Inspect implementations of all existing `*_decode_rows_exact_tensor`
   functions, especially Metal.  Identify whether they:
   - launch one kernel per row,
   - use a row dimension inside one dispatch,
   - preserve an explicit reduction order,
   - or simply alias a batch kernel known to be exact for that datatype.

5. Find the canonical single-token compressor path around
   `ds4_gpu_matmul_f16_pair_compressor_store_tensor()` and the generic batch
   compressor projection around `ds4_gpu_matmul_f16_pair_tensor()`.

6. Do not implement a DSpark compressor kernel yet.  Write an API proposal and
   identify the exact state/tensor that would be its input/output.

7. Integrate the previously supplied tensor-comparison helper if it builds
   cleanly; otherwise adapt it to local conventions, but keep it behavior-off
   by default.

Deliverables before the GGUF arrives:

- `/tmp/dspark_context.txt`
- exact N=2 vs generic verifier stage table
- proposed probe insertion points
- proposed exact compressor rows API signature
- zero behavior change to normal inference

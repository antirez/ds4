# Testing

## Correctness Tests

```sh
make test
```

Individual checks:

```sh
./ds4_test --server             # API, rendering, streaming, tool calls
./ds4_test --logprob-vectors    # Token bytes vs official vectors
./ds4_test --long-context       # Story fact-recall regression
./ds4_test --tool-call-quality  # DSML tool-call emission
./ds4_test --metal-kernels      # Isolated Metal kernel checks
```

Override paths:

```sh
DS4_TEST_MODEL=/path/to/model.gguf ./ds4_test --logprob-vectors
```

## CUDA Regression

```sh
make
make cuda-regression
```

## CPU Portability

```sh
make cpu
```

The CPU backend is a reference/debug path, not production.

## Speed Tests

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 \
  --step-incr 2048 --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

## Debugging

```sh
./ds4-server --trace /tmp/ds4-trace.txt
```
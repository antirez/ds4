# Contributing

Changes should be tested against the failure mode they can realistically affect. Include commands, machine/backend, model quant, and failures in PR notes.

**No speed regressions** except for important correctness fixes.

## Correctness Tests

```sh
make clean
make
make test
```

## Quality Checks (Quantization)

```sh
make -C gguf-tools quality-score
```

Compare old vs new GGUFs:

```sh
gguf-tools/quality-testing/score_official OLD.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/old.tsv 4096

gguf-tools/quality-testing/score_official NEW.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/new.tsv 4096

python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Lower `avg_nll` is better.

## Speed Tests

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 \
  --step-incr 2048 --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

## PR Guidelines

1. Test on the backend(s) affected
2. Include commands, machine, model quant
3. No speed regressions except for correctness fixes
4. Update docs for user-facing changes
5. Run the test suite before opening
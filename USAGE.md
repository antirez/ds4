# USAGE — comandi utili della campagna M1 Ultra

Raccolta operativa dei comandi usati e validati in questa campagna
(agosto 2026, Mac Studio M1 Ultra 128 GB, Metal). Dettagli e numeri in
`EXPERIMENTS_M1_ULTRA.md`; descrizione PR in `PR_DESCRIPTION.md`;
patch complete in `m1_ultra_patches.diff` (ripristinabili con `git apply`).

## Taratura e default per-device

Su M1/M2 i default si applicano **da soli** all'avvio (riga di log
"defaults applied"). Env utili:

```bash
# comportamento upstream puro (kill-switch di tutti i default)
DS4_NO_DEVICE_DEFAULTS=1 ./ds4 -m ds4flash.gguf --temp 0 -p "..."

# override puntuali (le env esplicite vincono sempre sui default)
DS4_METAL_WIDEN_M3_GATES=0        # disattiva le fusioni M1 (+9.7% quando on)
DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=2 DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS=8  # split M1 (+6.8%)
DS4_NGRAM_SPEC=0                  # spegne la speculazione n-gram
DS4_NGRAM_SPEC=8 DS4_NGRAM_MIN_MATCH=6  # taratura server/agent (thinking): mai peggio di -4%, +13% su edit
DS4_NGRAM_MIN_MATCH=3             # taratura per copia/edit con --nothink (+16%)
```

## Benchmark di velocità (ds4-bench)

```bash
# residente, sweep standard CONTRIBUTING
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128 --csv /tmp/speed.csv

# SSD streaming (MXFP4 145 GB): prima frontiera = transitorio, poi regime ~9-10 t/s
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 ./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf \
  --ssd-streaming --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-incr 2048 --gen-tokens 128 --csv /tmp/streaming.csv

python3 speed-bench/plot_speed.py /tmp/speed.csv --title "M1 Ultra t/s"
```

## Eval (qualità e velocità in situazione reale)

```bash
# gate deterministico anti-deriva (atteso su M1: 622 tok B/B, 240 C/C, 624 70/70, 2048)
./ds4-eval -m ds4flash.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1

# velocità con SSD streaming (t/s per domanda nella riga "(Xs, N tokens)")
# NB: --tokens deve essere > 512 (hard-limit-reply-budget)
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 ./ds4-eval \
  -m gguf/DeepSeek-V4-Flash-MXFP4Experts-...-mxfp4-0731.gguf \
  --ssd-streaming --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

## A/B bit-esatto (per qualunque modifica al decode)

```bash
make metal-decode-schedule-bench
# abortisce se anche un solo logit o token selezionato differisce
./speed-bench/metal_decode_schedule_bench --include-selection \
  --control-first 2 --control-second 8 --candidate-env NOME_ENV_DA_TESTARE
```

## Banco di prova del verificatore speculativo

```bash
# verifica a blocco pieno a ogni ciclo, stream ≡ decode puro (byte-esatto a N≤5):
# misura il costo del verificatore in isolamento
DS4_NGRAM_SPEC=8 DS4_NGRAM_FAKE_PROPOSAL=1 DS4_DSPARK_STATS=1 \
  ./ds4 -m ds4flash.gguf --prompt-file PROMPT --temp 0 --nothink --tokens 128
# statistiche accettazione/verify (anche per l'n-gram): DS4_DSPARK_STATS=1
# log per-ciclo: DS4_NGRAM_SPEC_LOG=1
```

## Suite di regressione (CONTRIBUTING)

```bash
make && make test          # completo
./ds4_test --server        # logica server/rendering
./ds4_test --logprob-vectors   # vettori ufficiali DeepSeek (gate numerico principale)
./ds4_test --metal-kernels     # NB: 29 failure PREESISTENTI su M1 (identiche su HEAD pristino)

# ATTENZIONE make cpu: sovrascrive i binari con le build CPU e il make
# successivo NON li ricollega (sintomo: "backend=cpu" all'avvio, ~6 t/s). Rimedio:
rm ds4 ds4-server ds4-bench ds4-eval ds4-agent && make
```

## Profiling

```bash
# per token: encode CPU / esecuzione GPU / lettura logits
DS4_METAL_GRAPH_TOKEN_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1 ./ds4 ...

# per stage dentro un layer (attn/ffn; sincronizza a ogni confine: solo attribuzione relativa)
DS4_METAL_LAYER_STAGE_PROFILE=1 ./ds4 ...          # =N per un solo layer
DS4_METAL_MOE_STAGE_PROFILE=1 DS4_METAL_MOE_STAGE_PROFILE_LAYER=20 ./ds4 ...
DS4_METAL_ATTN_OUT_STAGE_PROFILE=1 ./ds4 ...

# streaming: cache, pread, percorsi di preparazione
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 ./ds4 ... --ssd-streaming ...
DS4_METAL_STREAM_PREPARE_SUBPROFILE=1 ...          # istogramma percorsi prepare + delta mlock

# oracolo esperti (per studi di prefetch offline)
DS4_MOE_RECORD_SELECTED_IDS=/tmp/oracle.bin ./ds4 ... --ssd-streaming ...
```

## Server e router

```bash
# server locale con la taratura consigliata (i default M1 fanno già tutto;
# esplicitare solo per documentare o per override)
./ds4-server -m ds4flash.gguf --ctx 262144 \
  --kv-disk-dir ~/Projects/.ds4/server-kv --kv-disk-space-mb 26144

# llm-router (Mac Studio): le env vanno nella chiave `env:` del backend,
# MAI come prefisso in `command:` (argv puro, niente shell)
cd ~/Projects/llm-router && .venv/bin/python -m llmrouter --config configs/macstudio/llmrouter.yaml

# benchmark dal MacBook (senza --base-url punta a 127.0.0.1 → Connection refused!)
.venv/bin/python tools/benchmark.py --models deepseek-v4-flash \
  --prompt-tokens 2048,8192,32768 --csv bench.csv \
  --base-url http://192.168.1.235:4000 --api-key sk-llmr0ut3r-l0c4l

# metriche prefill/gen per-richiesta dai log del server
python3 /tmp/parse_ds4_log.py /tmp/llmrouter-ds4.log
```

## Sistema

```bash
# tetto memoria GPU: il default (~96 GiB) è quasi saturo col modello 91 GiB
# e sfora caricando DSpark. Reversibile al riavvio o con =0.
sudo sysctl iogpu.wired_limit_mb=106496   # 104 GiB, ~24 GB al sistema
```

## Patch e backup

```bash
git diff > m1_ultra_patches.diff     # snapshot delle modifiche (mai committate)
git apply m1_ultra_patches.diff      # ripristino su un tree pulito
```

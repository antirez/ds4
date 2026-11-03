# Verifica della correzione del drift Qwen residente

Questo archivio conserva report, manifest, patch, script e log della verifica sul M1 Max locale. `RISULTATI.md` distingue il primo A/B della working tree dal secondo A/B dei sorgenti selezionati nell'indice per il commit. Entrambi comprendono 14 esecuzioni e sette confronti, tutti PASS. Non sono benchmark né una replica del Q2 completo su M4 Max del tester.

## Contenuti e provenienza

- `source-snapshots-manifest.json`: hash e dimensioni di 89 file di main `0aaea5a238fb41a35106a551e73c8409dfb751ac` e di 91 file dell'input precedente alla correzione.
- `pending-input.patch`: soltanto le quattro sorgenti dell'input locale che differivano da Git `575369b19d8754f7eff69b9fdaa9a4c3684b825a`: Makefile, ds4_metal.m, metal/dense.metal e metal/qwen4.metal.
- `prepare-snapshots.py`: ricostruisce questi due input esclusivamente dai blob Git e dalla patch, verificando tutti gli hash del manifest. Non usa il contenuto della working tree o gli assoluti storici registrati nei manifest.
- `fix-ab/results.json`: primo A/B, con il candidato della working tree e gli esperimenti locali preesistenti.
- `release-staged-sources.json` e `release-staged-ab/results.json`: 94 file estratti dall'indice per verificare separatamente il candidato destinato al commit, senza gli esperimenti locali preesistenti.
- `run-fix-ab.py`: runner seriale, con controlli di sorgenti, binari, identità dei modelli, prompt e finitezza dei dump.
- `make-resident-fixture.py`: generatore della fixture residente con 48 layer, pesi originali dei blocchi 0–3 ripetuti, PLE zero e predictor assente. I manifest documentano layout e payload; non è un modello per valutare qualità.
- `correction-only.patch`, test mirati, log e `RISULTATI.md`: correzione isolata ed evidenze numeriche.

Sono esclusi i GGUF, le directory ricostruite `base0aa/`, `before-fix/`, `release-staged/`, `reconstructed-check/`, i backup completi in `before/`, i binari e gli oggetti di build. Il riferimento a `before/` in `RISULTATI.md` descrive soltanto il backup locale originario: il contenuto sorgente dell'input prima del fix si ricostruisce da Git `575369b1` più `pending-input.patch`. Le directory e i percorsi assoluti nei vecchi log descrivono il computer della verifica.

## Ricostruire gli input storici

Eseguire i comandi dalla radice di un clone che contiene i due commit. La destinazione deve essere nuova; il helper rifiuta anche una directory esistente vuota. Non compila né avvia modelli.

```sh
python3 misc/qwen-drift-fix-2026-10-03/prepare-snapshots.py \
  misc/qwen-drift-fix-2026-10-03/replay-sources --repo .
```

Il risultato è `replay-sources/base0aa/` con 89 sorgenti, `replay-sources/before-fix/` con 91 sorgenti e `RECONSTRUCTION.json` con stato PASS. Prima della scrittura, il helper verifica gli hash del manifest e della patch, applica il diff in memoria con contesto esatto e verifica tutte le sorgenti finali. È stato provato su una destinazione temporanea e dal processo principale in `reconstructed-check/`; entrambi i controlli sono riusciti.

## Ripetere i confronti della versione selezionata per il commit

Usare un checkout pulito del commit che contiene la correzione e questo archivio come candidato. I tre prompt sono già versionati in `misc/qwen-drift-2026-10-03/fixtures/`; i loro hash e i conteggi 29, 575 e 5942 sono nei report. Le build precedono tutte le esecuzioni GPU.

```sh
make -C misc/qwen-drift-fix-2026-10-03/replay-sources/base0aa ds4
make -C misc/qwen-drift-fix-2026-10-03/replay-sources/before-fix ds4
make ds4 ds4-server tests/test_qwen4_generation
make test-qwen4-resident-arithmetic test-qwen4-memory
```

Il test arithmetic prova quattro assetti retained/unretained × math-safe off/on, nelle modalità prefill/decode residente e SSD. La reference congela il dispatch di main, usando i kernel correnti. `hc-reference.m` è una verifica separata dello shader HC storico:

```sh
clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
  misc/qwen-drift-fix-2026-10-03/hc-reference.m \
  -o misc/qwen-drift-fix-2026-10-03/replay-hc-reference
misc/qwen-drift-fix-2026-10-03/replay-hc-reference "$PWD"
```

La sorgente Q4 completa deve essere già disponibile in `gguf/Qwen3.8-Flash-Next-Q4.gguf`. Il primo comando seguente prepara soltanto un piano; il secondo scrive una nuova fixture di 7,953,727,808 byte. Il generatore corrente include già le 16 teste PLE corrette, quindi la riparazione dell'header documentata nei risultati storici non va ripetuta su una nuova fixture.

```sh
python3 misc/qwen-drift-fix-2026-10-03/make-resident-fixture.py \
  gguf/Qwen3.8-Flash-Next-Q4.gguf \
  misc/qwen-drift-fix-2026-10-03/replay-resident-fixture.gguf \
  --manifest misc/qwen-drift-fix-2026-10-03/replay-fixture-plan.json
python3 misc/qwen-drift-fix-2026-10-03/make-resident-fixture.py \
  gguf/Qwen3.8-Flash-Next-Q4.gguf \
  misc/qwen-drift-fix-2026-10-03/replay-resident-fixture.gguf \
  --manifest misc/qwen-drift-fix-2026-10-03/replay-fixture-written.json --write
```

Il runner seguente usa main come reference residente, l'input prima del fix come reference SSD e il checkout pulito come candidato. Avvia i processi in sequenza, ciascuno dalla propria directory, così i sorgenti Metal caricati a runtime corrispondono al binario. `replay-ab/` deve essere una nuova destinazione.

```sh
python3 misc/qwen-drift-fix-2026-10-03/run-fix-ab.py \
  --base-dir misc/qwen-drift-fix-2026-10-03/replay-sources/base0aa \
  --before-dir misc/qwen-drift-fix-2026-10-03/replay-sources/before-fix \
  --fixed-dir . \
  --resident-model misc/qwen-drift-fix-2026-10-03/replay-resident-fixture.gguf \
  --ssd-model gguf/Qwen3.8-Flash-Next-Q4.gguf \
  --fixtures misc/qwen-drift-2026-10-03/fixtures \
  --output misc/qwen-drift-fix-2026-10-03/replay-ab
```

Le verifiche one-shot/sessione usano gli stessi token teacher-forced e confrontano il vocabolario completo. Si eseguono separatamente e in sequenza:

```sh
./tests/test_qwen4_generation \
  misc/qwen-drift-fix-2026-10-03/replay-resident-fixture.gguf \
  --chunk 2048 --ctx 16384 --tokens 575
./tests/test_qwen4_generation gguf/Qwen3.8-Flash-Next-Q4.gguf \
  --ssd-streaming --prefill-reference --chunk 2048 --ctx 16384 --tokens 29
```

Il numero passato a `--tokens` di questo test è la lunghezza del prompt sintetico ripetuto; il test aggiunge otto passi decode. Questi prompt sono distinti dai file testuali dell'A/B. Il risultato atteso è nove frontiere full-vocab identiche per ciascuna modalità. Nei report A/B i cinque casi logits devono avere zero float32 differenti e JSON identici; i due casi greedy devono avere gli stessi 64 token e JSON delle logprob identici. I tempi dei processi non dimostrano una variazione di throughput.

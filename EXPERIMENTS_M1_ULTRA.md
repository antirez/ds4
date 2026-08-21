# Campagna di ottimizzazione decode su Mac Studio M1 Ultra (Metal)

Data: 2026-08-21. Modello: DeepSeek V4 Flash q2 imatrix 0731 (91 GB, residente).
Tutte le modifiche sono nel working tree, **non committate**, e **opt-in via env**:
con env di default il comportamento è identico a HEAD (84cc882), tranne il flush
periodico del verificatore esteso ai blocchi senza capture (rollback:
`DS4_METAL_DISABLE_SPEC_VERIFY_FLUSH=1`).

## Risultati (greedy, gen 128-512 token)

| Configurazione | t/s | Note |
|---|---:|---|
| HEAD default (ds4-bench, ctx 2048) | 24.6 | baseline |
| + `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=2` `_SECOND_SPLIT_LAYERS=8` | 26.3 | +6.8%, bit-exact (lo schedule adattivo 2/32 è tarato su M3) |
| + `DS4_METAL_WIDEN_M3_GATES=1` | 28.9 | +9.7%, **bit-exact verificato** con `metal_decode_schedule_bench --candidate-env` |
| + `DS4_NGRAM_SPEC=8` su task di riscrittura codice | **34.0** | +16.2% sopra la base; neutro su prosa; nessun modello di supporto |
| ctx 16384 / 32768 (patch vs default) | 24.7/24.2 vs 24.7/23.4 | mai peggiorativo |
| DSpark (`--mtp ... --dspark`), codice / prosa | 26.7 / 24.4 | **negativo su M1** (vs 29.3/28.8) nonostante 83.65% di accettazione |

Qualità: `ds4-eval` q1-q4 (`--plain --questions 4 --tokens 2048 --temp 0 --seed 1`)
identico tra env patchato e default (622/240/624/2048 token, B/C/70/C, 4/4 PASSED;
la tabella del README non corrisponde su questa macchina già a default).
n-gram N=5: output byte-identico al non speculativo; N≥8 e DSpark: stesso
contratto documentato di DSpark (ordine FP del verifier dopo un blocco accettato).

## Le patch

1. **`ds4_gpu_device_is_m3_class()`** (ds4_metal.m): con `DS4_METAL_WIDEN_M3_GATES=1`
   i 15 gate di fusione legati alla stringa "M3" (rope fuse + packed32 FA reduce,
   KV rope FP8, gathered KV staging, persistent zero mask, shared KV pad,
   compressor pair-proj/APE/ratio4, HC weights4, router weights batch, mask cache)
   si aprono su ogni pre-M5. Ogni fusione conserva il proprio `DS4_METAL_DISABLE_*`.
2. **Speculazione n-gram / prompt-lookup** (ds4.c): `ds4_ngram_propose()` +
   `ds4_session_eval_ngram_speculative_argmax()`; attivazione `DS4_NGRAM_SPEC=N`
   (consigliato 8), `DS4_NGRAM_MIN_MATCH` (default 3), log `DS4_NGRAM_SPEC_LOG`,
   statistiche con `DS4_DSPARK_STATS=1`. Riusa il verificatore DSpark: il confine
   è un array di token. Nessun file di supporto richiesto. I draft vengono
   troncati al primo token di stop/controllo, così un blocco accettato non può
   attraversare un confine di turno (necessario per l'agent, che riusa il KV
   in modo incrementale). Copertura: CLI e server (sessione singola, greedy)
   la usavano già via `ds4_engine_mtp_draft_tokens`; **ds4-agent** ora ha il
   proprio ramo speculativo (`agent_speculative_block_cap` +
   `worker_accept_verified_token` in ds4_agent.c), attivo solo su turni greedy,
   solo fuori dalle stanze DSML e con blocchi ≤8: misurato -18% di tempo turno
   sul task di riscrittura codice. ds4-bench e ds4-eval restano volutamente
   senza speculazione (sono gli strumenti di misura e di riferimento deriva).
3. **Strumentazione**: `ds4_gpu_busy_profile_seconds()` (ds4_metal.m, ds4_gpu.h) e
   campo `verify_gpu_busy` nelle statistiche: separa CPU-encode da GPU-exec nella
   verifica.
4. Cache PSO veloce estesa al layout q2 (`DS4_METAL_ENABLE_Q2_PIPELINE_FAST_LOOKUP`):
   misurata neutra, lasciata per completezza.

## La scoperta centrale: il verificatore è il collo di bottiglia speculativo

Misure su M1 Ultra (stats `verify_*`):

- verifica di 5 righe ≈ 112 ms, di 8 righe ≈ 121-125 ms → **costo quasi fisso**
  ≈ 3× un token di decode (37 ms);
- `verify_layer` = 99.9% del tempo; `verify_gpu_busy` = 96.6% di `verify_layer`
  → è **esecuzione GPU dei kernel batch di prefill** a piccole righe
  (~140 GB/s effettivi contro ~290 del decode), non encode CPU;
- per questo DSpark, pur accettando 2.29 token/ciclo, perde su M1: il commento
  dell'autore a `ds4.c:35360` ("not yet the final hand-written N=2/N=4 decode
  microbatch") è esattamente il pezzo mancante.

**Prossimo passo ad alto valore**: verificatore microbatch N≤8 con kernel derivati
dal decode (pesi letti una volta per N righe). Target ~60 ms/blocco → stima:
DSpark ~33 t/s su codice (da 26.7), n-gram ~37-38 (da 34). `metal_graph_verify_decode2_exact`
è il precedente nel codice ma rilegge i pesi per token (conviene solo a N=2).

## Wave A del verificatore microbatch: risultati e ipotesi smentite

Banco di prova: `DS4_NGRAM_FAKE_PROPOSAL=1` (verifica a blocco pieno ogni ciclo,
1 token committato → stream identico al decode; N≤5 verificato byte-identico).
Curva misurata del verificatore attuale: **verifica(N) ≈ 34 + 10.9·N ms GPU**
(N=2: 55-60, N=5: 100-126, N=8: 121-135 a seconda del percorso di commit).

Interventi provati (codice presente, gate via env):

1. **`mv_ext r1_6/7/8`** (kernel densi, pesi letti 1× invece di 2× a N=6..8):
   **bit-identico ma -25% di throughput di verifica su M1**. Le due passate
   grid.y=2 storiche girano in parallelo su più threadgroup, mentre r1_8
   dimezza i TG e satura i registri (8 accumulatori/thread). Default: OFF,
   opt-in `DS4_METAL_ENABLE_MV_EXT_WIDE` per riprovare su M3/M5.
2. **Soglie MoE tiny 5→8 e down_sum6 4→8** (host-only, i kernel tiny hanno già
   l'indice riga in griglia): stream identico, ~+1%. Default: ON
   (rollback `DS4_METAL_DISABLE_MOE_TINY_WIDE`) — meno dispatch, nessun costo.

**Diagnosi rivista**: a N piccolo la verifica NON è bandwidth-bound (dimezzare
i byte densi o MoE non sposta il muro): è **latency-bound sulle catene di
piccoli dispatch** — ~500 dispatch mono-riga del compressor per blocco
(`ds4.c:28870/29171`, il percorso batch non riceve le fusioni `defer_finalize`
del decode), flash attention mma a 64 threadgroup senza split-K (il decode usa
la variante vec con 2048 TG), maschera transient riempita da CPU per layer,
~86 chiusure di encoder con barrier.

**Wave B eseguita — fusione compressor nel batch** (`metal_graph_batch_comp_fuse_defer`
+ loop unico per-riga in ds4.c, rollback `DS4_METAL_DISABLE_BATCH_COMP_FINALIZE_FUSE`):
**bit-exact certificata** su tre confronti (N=8 fuso ≡ pre-patch, N=5 ≡ decode puro,
rollback ≡), adottata di default, guadagno ~1%. Terza lezione del silicio: nemmeno
il numero di dispatch era il muro (~10 µs l'uno se accodati nello stesso encoder).

Bug trovato sul campo e corretto: il ramo batch non-allineato è raggiunto anche
dai **resumed prefill** (pos0 arbitrario dopo un checkpoint vivo, tipico
dell'agent dopo i tool call); lì la fusione sforava lo staging condiviso →
`gpu layer 2 attention batch encode failed`. Il gate ora richiede
`n_tokens <= 16` (blocchi di verifica), unico regime dove conviene.

**Attribuzione per-stage definitiva** (DS4_METAL_LAYER_STAGE_PROFILE su verifiche
a 8 righe, depurata dall'overhead di sync, ms per verifica):
routed_moe ≈ 47 (40%: **già a ~290 GB/s, bandwidth-saturato** — riducibile solo
leggendo meno byte: dedup dell'unione esperti, overlap misurato ~38% → ~-16 ms);
output_proj ≈ 22 (l'8× di attn_output_a è reale ma la SLC ne assorbe metà);
hc_pre ≈ 13; q_path ≈ 11; indexer_setup ≈ 10; shared ≈ 5; l'attention mma,
principale sospettato della teoria dispatch-latency, è ~0-5 ms: assolta.

**Wave C (prossima, specifica dai dati)**:
1. kernel MoE expert-gathered per la verifica: raggruppare le (riga, esperto)
   per esperto e leggere ogni slab una volta (~-16 ms/verifica);
2. `kernel_dsv4_attn_out_low_q8_0_f32` a N righe (loop token interno, pesi
   una volta) (~-10-15 ms);
3. micro-fix su hc_pre/q_path (residuo ~10-15 ms).
Floor realistico verifica(8): ~60-75 ms → DSpark ~33 t/s e n-gram ~37-38 su
codice. Riferimento: mappa stage-per-stage dell'agente in questa sessione.

## Vicoli ciechi misurati (non ripetere)

- Draft adattivo troncato sui match deboli: -13% (con verifica a costo fisso
  conviene proporre solo blocchi lunghi ad alta confidenza).
- `DS4_METAL_MODEL_UNTRACKED=1`, cache PSO q2, split 1/32 e 2/4: neutri o peggio.
- Flush pipeline nel verificatore: ~+0.5 t/s (il collo è GPU, non encode).
- Da letteratura, per M1 Ultra: co-esecuzione CPU/AMX (banda unificata condivisa),
  ANE come drafter (~55 GB/s), prefetch predittivo esperti da SSD (routing di
  V4 Flash quasi uniforme: 209-229 esperti caldi/layer su 256).

## Raccomandazione di sistema (manuale, reversibile al riavvio)

```sh
sudo sysctl iogpu.wired_limit_mb=106496
```

Il cap di default (~96 GiB) è quasi saturo col modello (91 GiB) e viene sforato
caricando il supporto DSpark (+5.6 GiB); 104 GiB lasciano ~24 GB al sistema.

**Controprova eseguita** con il cap a 112000 MB: DSpark su codice 26.7 → 27.2 t/s
(proposta -5.7% ms, verifica -1.3%), prosa invariata. Il cap non era la causa
della perdita di DSpark: la diagnosi del verificatore GPU-bound è confermata.

## Conferma su workload reali (log ds4-server, default per-device attivi)

| workload | generazione | vs upstream 24.6 t/s |
|---|---|---|
| agent-eval multifile (15 step, tool) | **34.4 t/s** eff. (mediana 38.4, picchi 42.7) | **+40%**, tempo task -25% |
| accuracy hard (25 domande, thinking) | **27.8 t/s** eff. (mediana 27.6) | **+13%** |

Qualità sugli stessi run: 96% accuracy hard, agent task pass pieno (hidden
inclusi, 0 tool call invalide). L'agent ha prefillato solo 3.5k dei 50k token
di prompt grazie al riuso esatto del prefisso KV (replay DSML): i chunk
incrementali (~230 token/step) girano a ~135 t/s — regime da overhead fisso,
non confrontabile col prefill bench a chunk 2048. Parser: /tmp/parse_ds4_log.py.

## Fase 0 streaming SSD (MXFP4 145 GB, ctx 8192, prompt codice)

| Config | gen t/s | hit rate |
|---|---:|---:|
| cache auto (68.6 GiB, 5511 esperti) | **7.27** | 0.80 |
| cache 64 GB | 6.04 | 0.68 |
| cache 48 GB | 5.47 | 0.50 |
| thread pread 18 | 5.89 | 0.76 (nessun guadagno: SSD non queue-limited) |

Anatomia del token (~137 ms): ~34 ms compute + **~44 ms di overhead host
(bind ~23 + prepare_buffer ~21)** + attese miss (~0.61 esperti/layer).
L'overhead host è il 31% ed è il bersaglio n°1 della Fase 1 — il 57% dei
layer-eval è "all resident" eppure paga il bind per selezione. Candidati a
costo zero: knob esistenti `DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE`,
`..._COMPACT_ADDR`, `..._STATIC_DECODE_MAP`.

n-gram sotto streaming: **non provisioned** (verifier_unavailable su ogni
blocco; upstream infatti rifiuta --mtp con --ssd-streaming). Output identico
con/senza (rollback puliti); il gate ora esclude `ssd_streaming`, quindi i
default per-device non attivano macchinario morto lì.

Oracolo esperti registrato (`/tmp/ds4_oracle.bin`, 64 token) per il calcolo
del tetto teorico del prefetch in Fase 2. Nota harness: prompt-file da
tagliare a ~5-6k token per ctx 8192 (il taglio a 32 KB sfora col template).

**Fase 1b streaming — verdetto finale (quarta lezione del silicio)**: l'overhead
host di prepare/mlock (~0.5 ms per miss, individuato con sub-profiling a
istogramma dei percorsi: era la mlock lazy per-slot durante il riempimento
degli slab) **non sta sul percorso critico del token** — gira sui thread
async del loader, già nascosto dall'overlap. La patch bulk-mlock provata è
throughput-neutra (stream byte-identici) ma raddoppia il lavoro mlock
(zero-fill wiring su pagine fresche): ritirata, con nota nel codice. Il collo
reale del transitorio è il riempimento cache da SSD (fisica); il regime vero
del MXFP4 è **~9.5-9.7 t/s** (run lunghi). Diagnostica conservata:
`DS4_METAL_STREAM_PREPARE_SUBPROFILE=1` (sub-timer + istogramma percorsi +
delta mlock). Leve restanti per lo streaming: solo Fase 2 (two-tier,
prefetch da oracolo).

**Chiusura Fase 1 streaming — il regime è al floor fisico**: ~34 ms GPU +
~62 ms SSD (334 MB/token di miss a 5.4 GB/s saturi) ≈ i 101 ms misurati
(9.9 t/s). L'SSD è serializzato per costruzione (i miss di L si conoscono
solo dopo il router di L; i 3 layer hash sono i primi del forward: anticipo
zero). Settimo esperimento: preload pieno (`AUTO_PRELOAD_CAP=6000`) PEGGIORA
(6.16 → 4.50 t/s, hit 0.76 → 0.60: la hotlist globale su routing quasi
uniforme scaccia gli esperti del prompt — il cap upstream 4096 è corretto).
Migliorie implementate in Fase 1: gate n-gram sotto streaming + diagnostica
sub-profile. Ogni ulteriore velocità richiede predizione/two-tier (Fase 2,
sospesa su decisione del progetto).

**Fase 1 streaming (A/B knob esistenti)**: `ENABLE_STREAMING_EXPERT_ADDR_TABLE`,
`COMPACT_ADDR`, `STATIC_DECODE_MAP` e la combo sono tutti **neutri**
(5.93-6.10 vs 6.00 di baseline interna; `prepare_buffer_total` invariato a
~30 ms/token). Prefill streaming su prosa ~5.5k token: **135 t/s** (buono);
gen dopo prompt lungo 3.66 t/s. Metodologia: baseline varia tra batterie con
lo stato page-cache (6.0 vs 7.27 a hit 0.76 vs 0.80) — confrontare solo
dentro la stessa batteria. **Prossimo passo (Fase 1b)**: cache della
preparazione buffer per i layer all-resident (52 chiamate/token × 0.45 ms;
il 57% dei layer-eval rifà una preparazione identica) — stima +25-30%.

## Attenzione: trappola `make cpu`

`make cpu` sovrascrive i binari (`ds4`, `ds4-eval`, ...) con le build CPU-only
e il `make` successivo NON li ricollega (li vede più recenti degli oggetti):
si resta silenziosamente su `backend=cpu` a ~6 t/s. Sintomo: la riga di avvio
dice `backend=cpu`. Rimedio: `rm ds4 ds4-server ds4-bench ds4-eval ds4-agent && make`.
Segnalata anche in PR_DESCRIPTION.md.

## Comandi di riferimento

```sh
# profilo per-token
DS4_METAL_GRAPH_TOKEN_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1 ./ds4 ...
# A/B bit-esatto di qualunque gate
./speed-bench/metal_decode_schedule_bench --include-selection \
  --control-first 2 --control-second 8 --candidate-env DS4_METAL_WIDEN_M3_GATES
# configurazione veloce consigliata su M1
DS4_METAL_WIDEN_M3_GATES=1 \
DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=2 DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS=8 \
DS4_NGRAM_SPEC=8 ./ds4 -m ds4flash.gguf --temp 0 ...
```

## Default automatici per device (nuovo)

Da questa versione i cinque eseguibili applicano da soli la taratura misurata:
all'init Metal, se il device è **M1/M2**, `ds4_gpu_apply_device_default_env()`
(ds4_metal.m) imposta con `setenv(overwrite=0)`:
`WIDEN_M3_GATES=1`, split `2/8`, `NGRAM_SPEC=8`, `NGRAM_MIN_MATCH=6`.
Le env passate esplicitamente al lancio **vincono sempre**;
`DS4_NO_DEVICE_DEFAULTS=1` ripristina il comportamento upstream puro;
`DS4_METAL_WIDEN_M3_GATES=0` ora disattiva (check sensibile al valore).
M3/M4/M5 restano sui default upstream finché non misurati. `--quality`
esclude l'n-gram anche coi default attivi (stesso contratto DSpark).
Verificato: no-env 29.6 t/s con riga di log; kill-switch 24.6; override
puntuale ok; eval gate invariato (622 tok, B/B).

## Taratura n-gram per workload (misure server, gen t/s prefill escluso)

| ds4-server (thinking attivo) | codice nuovo | riscrittura |
|---|---:|---:|
| senza n-gram | 30.2 | 28.8 |
| `DS4_NGRAM_SPEC=8` (min_match=3) | 28.4 | 27.0 |
| `DS4_NGRAM_SPEC=8 DS4_NGRAM_MIN_MATCH=6` | 29.1 | **32.6** |

Regola pratica: col **thinking attivo** (server/agent di default) usare
`DS4_NGRAM_MIN_MATCH=6` — la prosa di ragionamento genera match a 3 token
frequenti ma poco predittivi, e ogni verifica costa ~120 ms. Con `--nothink`
su task di copia/edit il default min_match=3 rende di più (+16% CLI). Il
verificatore microbatch (Wave C) ridurrà il costo delle verifiche e
allenterà questo trade-off precisione/richiamo.

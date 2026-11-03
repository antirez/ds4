# Revisione indipendente della correzione — 3 ottobre 2026

Esito: non emergono problemi bloccanti nell'implementazione dei due guard o
nel test di regressione. La correzione ripristina il dispatch residente del
base `0aaea5a2` e conserva il dispatch SSD presente immediatamente prima
della correzione. Non introduce un'opzione numerica configurabile dall'utente.

Revisione statica e controlli di provenienza; nessuna compilazione, esecuzione
GPU o modifica ai sorgenti di produzione da parte del revisore. Le misure
complete del modello e i risultati dei test sono responsabilità dell'agente
principale e non vengono anticipati qui.

## Implementazione

Il diff di `ds4.c` contro `before/ds4.c` contiene due cambiamenti di condizioni,
più i commenti che ne spiegano lo scopo.

1. `ds4.c:58242–58243`: la policy half/unsplit del prefill è attiva soltanto
   quando sono vere sia `g->ssd_streaming` sia la fase PREFILL.
   Per un grafo residente il valore è sempre falso, così le condizioni F16
   `n_tok > 3 && n_tok <= 64`, Q8 `5 <= n_tok <= 32` e il wrapper F32
   split-K coincidono con `qwen4_gemv_rows` del base. Tutti gli altri predicati,
   vincoli di capacità, fallback e override diagnostici restano identici.
   Per SSD prefill il valore resta vero, esattamente come prima della patch.
   Durante il decode il valore resta falso per entrambi i modi.
2. `ds4.c:58495`: l'attenzione che mantiene la policy del parent durante
   partizioni interne brevi è limitata al prefill SSD. Il residente torna
   alla chiamata per-range del base, compresa la scelta della partial buffer
   in base a `parent_rows <= 2`. Perciò le tre righe dense a posizioni
   2048–2050 prendono l'attenzione scalar quando il parent è residente,
   mentre il parent SSD conserva la policy matrix. Il resto della routine
   e la selezione dense/sparse non cambiano.

Entrambe le condizioni sono sotto guard Apple: la prima nel ramo `#else`
di `#if !defined(__APPLE__)`, la seconda in `#ifdef __APPLE__`. Il codice
CUDA conserva i medesimi entry point e argomenti; non è stata eseguita
validazione GPU CUDA.

Il flag `ssd_streaming` esisteva già nel grafo. La CLI lo assegna prima del
prefill (`ds4.c:59669`) e le sessioni lo copiano dall'engine
(`ds4.c:73553`); `qwen4_graph_reset` non lo azzera. Il cambiamento non
dipende da un nuovo environment flag o da una stringa del dispositivo.
I kernel Metal, la quantizzazione, la cache SSD, il caricamento esperti e
le pipeline del decode non cambiano nel delta della correzione.

## Test di regressione

`tests/test_qwen4_resident_arithmetic.c` contiene un riferimento di dispatch
estratto letteralmente da `0aaea5a2:ds4.c`. Una verifica Python contro Git
conferma che il corpo completo di `resident_base_projection` differisce
soltanto per il nome della funzione. Non viene generato dal dispatcher
attuale sotto test.

Le fixture verificano quattro modi distinti: residente prefill, SSD prefill,
residente decode e SSD decode. Il residente è confrontato con il riferimento
base; SSD prefill con il riferimento indipendente legacy/unsplit selezionato
solo durante la chiamata oracle. Il test elimina gli override che potrebbero
mascherare il cambiamento di policy e tiene la modalità math fast/safe
selezionata dal comando di esecuzione.

La copertura è mirata alle soglie che possono cambiare l'aritmetica:

- HC-down F16 [10240,320] e HC-up F16 [320,10240], T=1/3/8/9/29/64/65.
- Q8 [2560,6144], T=1/3/4/5/8/9/16/17/29/32/33.
- GDN alpha F32 [2560,48], T=1/3/8/9/29/575/1846/2048; copre gli split
  del prompt breve, Roma single-chunk e l'ultima coda del testo lungo.
- Router F32 [2560,512], T=29/575.
- Partizione attention di tre righe a posizione 2048: riferimento per-range
  confrontato con una chiamata matrix indipendente a nove righe che contiene
  le stesse ultime tre query. La fixture esige una differenza tra i due
  riferimenti, quindi non può passare usando un caso numericamente neutro.
  Verifica anche che un parent realmente breve conservi la policy scalar.

I confronti sono bit per bit, con controlli di finitezza, guard sugli output,
immutabilità di input e hash dei pesi. Le fixture vengono rilasciate in
sequenza; non richiedono un GGUF o mapping del modello completo. Per i casi
principali da 29 token il test esige una differenza fra riferimento residente
e SSD, rendendo significativa la prova negativa contro il dispatcher pre-fix.

Il Makefile esegue il test nei quattro incroci fast/safe e command buffer
retained/unretained. Il caso retained usa `env -u DS4_METAL_UNRETAINED`,
correttamente rispetto alla lettura per presenza di quel flag nel backend.
La modifica di `test_qwen4_generation` impedisce l'uso accidentale del
riferimento SSD unsplit con una sessione residente, ma continua a permettere
il confronto residente standard fra one-shot/sessione.

## Limiti delle conclusioni

- L'oracle congela il dispatch storico, non l'intera libreria shader storica:
  entrambi i percorsi usano il backend attuale. Dimostra la correzione della
  precisione e delle riduzioni selezionate, non la completa parità di logits
  con tutti i kernel di `0aaea5a2`.
- Il confronto full-model base/fix resta necessario per i residui delle
  altre modifiche, incluso il mixer HC F16 del commit `028b43f3`.
- La macchina locale è M1 Max e il modello disponibile è Q4. I risultati
  locali non sostituiscono un replay del Q2 residente sul M4 Max del tester.
- Il test sintetico di attenzione esercita direttamente il helper con
  `use_sel=false`; le partizioni sparse selezionate non sono coperte da
  questa specifica fixture. Il guard agisce identicamente sui due valori
  di `use_sel`, mentre la suite esistente e il full-model coprono il flusso
  completo.
- Ripristinare il riferimento residente non implica identità universale
  fra chunk diversi o fra GPU diverse; implica conservare il dispatch
  numerico di main per quel modo di esecuzione.

## Identità dei file revisionati

SHA256 rilevati durante questa revisione:

```
ds4.c
4045c6a98af062ef38e8fd2ded0e85e01fba85e0ca1174f07ae43f8a0b780ff1
tests/test_qwen4_resident_arithmetic.c
604edfcca9d3bcaa0e6d7dd09b4be84554c1313ebb9067849bf24f29c67f2374
tests/test_qwen4_generation.c
1b624026c3ff4ab2c1b2a345b8edb2179744136632ce62427509868fa4b9994a
Makefile
17b06206bf0cf87db2f5e7083a4a7e8e786f75b56690612ba7c58d9c6edcf081
ds4_metal.m
efd5fd3b72f356d9f8c7127d92ba9d9b1c73166a0cdad3746d3e137336b8089b
metal/qwen4.metal
6c4d04d1a647edf48d5d53bd216a0f38a06b80ecf675c7b6f732442e8c3475ae
metal/dense.metal
71f69ce54a6a46ca1d71da509cb2de888171444790930235dc0b460a5c82f00e
```

L'identità delle copie isolate base/pre-fix è documentata separatamente in
`source-snapshots-manifest.json`. Nessun file di produzione o sorgente
congelato dei run A/B è stato modificato dal revisore.

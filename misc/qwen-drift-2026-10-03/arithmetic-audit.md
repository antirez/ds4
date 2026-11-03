# Audit aritmetico del drift Qwen — 3 ottobre 2026

Audit del codice, senza compilazione o esecuzione GPU. Confronto richiesto:
`0aaea5a2` contro `02313128`, Qwen Q2 residente su M4 Max. Il worktree contiene
esperimenti successivi non committati; l'analisi delle differenze usa `git show`
e `git diff` dei due commit, non attribuisce gli esperimenti locali al tester.

## Candidato principale: cambia l'ordine delle somme delle proiezioni GDN

Il merge `b1af94b4` incorpora `2d6a207b`, che distingue esplicitamente prefill
e decode. Nel commit base, `qwen4_gemv_rows()` seleziona la GEMM FP32 anche per
le proiezioni F32 del prompt e permette la divisione della dimensione K.
Nel commit HEAD il medesimo prefill chiama
`ds4_gpu_qwen4_dense_mm_prefill_tensor()`: il parametro `allow_split_k=false`
mantiene `n_split=1`.

Riferimenti nel codice attuale:

- `ds4.c:58238–58253`: scelta della fase e del wrapper prefill.
- `ds4.c:58464–58465`: proiezioni `lin_alpha` e `lin_beta` della GDN.
- `ds4_metal.m:52061–52080`: formula di scelta dei gruppi e degli split.
- `ds4_metal.m:52111–52116`: wrapper prefill che disabilita gli split.
- `metal/qwen4.metal:5069–5099`: ogni split accumula solo un intervallo K,
  partendo da zero.
- `metal/qwen4.metal:5117–5128`: una seconda catena FP32 somma le parziali.

La lettura del solo header del Q4 presente in locale conferma:

```
blk.0.ssm_alpha.weight [2560, 48] F32
blk.0.ssm_beta.weight  [2560, 48] F32
output_hc_up.weight   [320, 10240] F16
```

Il Q2 del tester non è presente in locale. La conversione usa la stessa
politica per i tensori di controllo; verificare i tipi del file Q2 del tester
prima di considerare questa lettura una prova del suo esatto file.

Per le due proiezioni GDN [2560,48], la formula base è:
`threadgroups=ceil(48/32)*ceil(T/32)`, `nk=2560/32=80`,
`n_split=min(ceil(128/threadgroups), nk/4, 64)`.

| Prompt / chunk del tester | T della proiezione | Base | HEAD |
| --- | ---: | ---: | ---: |
| Roma corto, chunk 2048 | 29 | 20 split | 1 |
| Roma, un solo chunk | 575 | 4 split | 1 |
| Promessi sposi, primi due chunk | 2048 | 1 split | 1 |
| Promessi sposi, ultimo chunk | 1846 | 2 split | 1 |

Quindi il caso da 575 token può cambiare già al primo logit anche senza
divisione del prompt in più chunk. La divisione interna di K è distinta dalla
divisione del prompt: l'assenza di una coda del prompt non la esclude.

Le operazioni sono matematicamente equivalenti, ma le somme FP32 non sono
associative. Quattro accumulatori inizializzati a zero e poi sommati non
arrotondano come un unico accumulatore continuo. Alpha e beta entrano nelle
dinamiche ricorrenti della GDN; differenze inizialmente piccole possono cambiare
gli stati e le selezioni degli esperti nelle layer successive. Un token greedy
diverge quando cambia l'ordine dei logits migliori. La grandezza precisa dei
delta riportati dal tester resta da provare con il confronto controfattuale
gestito dall'agente principale: questo audit non misura quei delta.

Il commento del commit `2d6a207b` dichiara che questa modifica corregge una
precedente selezione di aritmetica decode durante il prefill e riporta confronti
contro `e37f1857`, non contro `0aaea5a2`. Non va classificata automaticamente
come bug nuovo né rimossa soltanto per riprodurre un baseline numerico diverso.

## Ulteriore differenza per i prompt brevi

Nel base, tra 9 e 64 token, le proiezioni F16 usano matrici con attivazioni FP32.
HEAD esclude questi kernel nella fase prefill e conserva il percorso di
riferimento con attivazioni arrotondate a half. Inoltre il base poteva usare
la GEMM Q8 FP32 per 5–32 token, mentre HEAD la riserva al decode.
Entrambe le differenze sono indipendenti dalla policy M1/M2/M3/M4 e possono
contribuire al delta del prompt da 29 token. Non spiegano da sole il prompt
da 575 token, per il quale la diversa riduzione F32 sopra è il candidato
più diretto.

## Candidato secondario: aritmetica del mixer HC F16

Il commit `028b43f3` cambia il kernel generico F16 su tutti i dispositivi,
indipendentemente dalle nuove impostazioni di tuning:

```
base: acc += w * silu(lo / hc)
HEAD: x = lo * (1 / hc)
      t = x * w
      u = t * sigmoid(x)
      acc = acc + u
```

HEAD disabilita esplicitamente reassociazione e contrazione nel corpo F16
(`metal/qwen4.metal:244–253`); il kernel a due token invece usa una catena FMA
esplicita (`metal/qwen4.metal:420–425`). L'intento era preservare la sequenza
osservata sul compilatore M1 e uniformare il generico ai kernel prefetch/reuse.
Non c'è una prova nel repository che il compilatore M4 precedente generasse
lo stesso ordine. Un confronto binario dei kernel vecchio e nuovo su M4 è
necessario per attribuirgli un eventuale delta residuo.

Per T>8 la miscelazione HC del tronco normalmente passa attraverso la GEMM
(`ds4.c:58403–58408`), quindi non usa questo loop. Tuttavia la CLI, quando
richiede solo gli ultimi logits, esegue il mixer finale su un'unica riga anche
per un prompt lungo (`ds4.c:59107–59119`). Pertanto HC può contribuire al primo
logit di tutti e tre i prompt. Durante il decode il loop si ripete anche nelle
layer del tronco, con possibilità di amplificazione ricorrente. Una sola
miscelazione finale, senza amplificazione nel tronco, rende HC un candidato
meno diretto dei nuovi split GDN per spiegare errori del primo logit >0,6.

## Altri sospetti controllati

- I flag `HC_STABLE`, `NORM_RSQRT_DISABLE`, `KV_RAW_F32` e `ROPE_EXP2_LOG2`
  conservano gli stessi default tra i commit. I relativi macro non appaiono
  nei kernel Qwen di `metal/qwen4.metal`; agiscono principalmente sui kernel
  DeepSeek e sulle normalizzazioni generiche. Non emerge una loro modifica
  che spieghi il confronto del tester.
- `MATH_SAFE` è off per default in entrambi. HEAD registra il valore in una
  variabile per escludere il prefetch MXFP4 in modalità strict; non cambia
  l'opzione di compilazione predefinita. M4 non usa la policy tensor M5/M6.
- La GEMV Q8 generica cambia caricamenti e riduzione in `metal/dense.metal`;
  mantiene i termini scalari e i due `simd_sum` con gli stessi zeri positivi.
  Un vecchio esperimento con quants `char2` aveva effettivamente causato drift
  tardivo, ma `04c08672` lo rimuove prima di HEAD. Un oracle congelato da
  `e37f1857` verifica l'intera GEMV su M1; l'esecuzione su M4 resta da validare.
- Le modifiche generiche IQ2 stage16 leggono insieme il medesimo header e
  preservano `dl * grid * sign` e la conversione half. L'ispezione non mostra
  un errore di indice/segno. Gli indirizzi SSD specializzati sono eliminati
  dalle pipeline residenti; non sono un responsabile naturale del caso
  residente del tester.

## Perché i test potevano passare

`tests/test_qwen4_generation.c:83–192` confronta one-shot, sessione e un grafo
diretto costruiti dalla stessa sorgente. Con `--prefill-reference`, il grafo
diretto imposta `DENSE_MM_LEGACY=1` e `NO_DENSE_MM_KSPLIT=1`
(`:143–146`): è il riferimento unsplit voluto dal commit di correzione,
non un'esecuzione congelata di `0aaea5a2`. Il test verifica la coerenza del
dispatch attuale ma non può imporre uguaglianza con il precedente dispatch
split-K del base.

Analogamente i test HC prefetch/reuse di `tests/test_qwen4_kernels.c:2261–2293`
e `:3671–3707` usano il kernel generico della stessa HEAD come riferimento.
Poiché `028b43f3` modifica quel generico insieme alle varianti, la loro
uguaglianza non prova uguaglianza con il generico pre-028 su M4. L'oracle Q8
è invece congelato indipendentemente: è un controllo più forte per quel
kernel, ma i risultati locali riportati riguardano M1.

Conclusione dell'audit: il cambiamento esplicito split-K/unsplit delle
proiezioni alpha/beta è il candidato principale con una catena causale
concreta per tutti i prompt; HC F16 è una seconda differenza numerica globale
da isolare per i residui e il decode. I toggle del tuning M1→M4 non
disabilitano nessuno dei due cambiamenti, quindi la loro uguaglianza non
costituisce un controesempio.

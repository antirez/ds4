# Audit del drift: base `0aaea5a` contro head `02313128`

Audit statico del 3 ottobre 2026. Nessuna modifica ai file di produzione,
nessuna compilazione e nessuna esecuzione GPU effettuata da questo audit.

## Evidenza da spiegare

Il tester usa M4 Max 64 GiB, Qwen3.8 Q2 residente, context 16384,
prefill chunk 2048, temperatura zero, `--nothink`. I tre prompt hanno 29,
575 e 5942 token. Rispetto a `0aaea5a`, il primo fronte di logits di head
presenta differenze massime circa 0,29 / 0,62 / 0,95; la generazione greedy
diverge in due casi. L'abilitazione del tuning M4 è numericamente neutra nel
confronto head/patch opt-in e conserva il miglioramento di velocità.

Questi dati escludono che il solo gate di `02313128` introduca il drift
misurato. Non identificano da soli una singola operazione responsabile.

## Due cambiamenti di politica aritmetica già presenti prima del port M4

### `2d6a207b`: proiezioni del prompt

In `0aaea5a`, `qwen4_gemv_rows` sceglie i kernel in base al numero di righe.
Su Apple, senza override diagnostici:

- F16, 4..64 righe: alcune proiezioni usano `ds4_gpu_qwen4_dense_mm_tensor`,
  che tiene gli operandi in FP32, anche se le righe sono un prompt.
- Q8, 5..32 righe: le proiezioni compatibili usano il batch GEMM FP32.
- F32, più di 8 righe: `dense_mm` può spezzare K in più accumulatori,
  poi sommare le matrici parziali.

`2d6a207b` introduce `QWEN4_PROJECTION_PREFILL`. Head esclude il batch F16 e
Q8 dai prompt e passa le proiezioni F32 a una nuova entry point che vieta
lo split-K. Non è una semplice modifica dei carichi o della griglia:
cambiano la precisione del RHS e/o l'associazione delle somme.

File e linee della working tree letta durante l'audit:

- `ds4.c:58242`: riconoscimento del prefill.
- `ds4.c:58244`: esclusione dei batch F16 dal prefill.
- `ds4.c:58248`: entry point F32 prefill.
- `ds4.c:58260`: esclusione del batch Q8 dal prefill.
- `ds4_metal.m:52070`: condizione `allow_split_k`.
- `ds4_metal.m:52111`: entry point prefill, con `allow_split_k=false`.

Il header del Q4 locale è stato letto senza caricare il modello: le matrici
statiche `ssm_alpha.weight` e `ssm_beta.weight` sono F32 `[2560,48]`.
La disposizione attesa per Q2 è la stessa, ma il Q2 del tester non è
disponibile localmente e questa identità non è stata verificata sul suo file.

Per queste matrici, la formula del dispatch base è:

`threadgroups = ceil(48/32) * ceil(T/32)`

`n_split = min(ceil(128/threadgroups), floor((2560/32)/4), 64)`

| Prompt / chunk | T | Gruppi prima dello split | Split base | Split head |
| --- | ---: | ---: | ---: | ---: |
| Roma breve | 29 | 2 | 20 | 1 |
| Roma lungo, singolo chunk | 575 | 36 | 4 | 1 |
| Promessi, primo chunk | 2048 | 128 | 1 | 1 |
| Promessi, secondo chunk | 2048 | 128 | 1 | 1 |
| Promessi, ultimo chunk | 1846 | 116 | 2 | 1 |

Di conseguenza, un singolo chunk da 575 token **non** esclude una differenza
nel prefill: alpha/beta selezionano ancora una diversa catena di somme.
Il prompt da 29 token ha inoltre il cambio FP32/half nelle proiezioni
F16 HC e nei batch Q8. Per 5942 token resta il cambio split-K nell'ultimo
chunk, oltre alla differenza di attenzione descritta sotto.

### `89986c3c`: attenzione alle partizioni interne

La soglia sparse è `(512+1)*4-1 = 2051`. Con chunk 2048, il secondo chunk
contiene tre righe ancora dense alle posizioni assolute 2048..2050 e 2045
righe sparse. In base il dispatch vede tre righe, quindi sceglie l'attenzione
scalare. Head trasporta la politica del parent da 2048 righe e sceglie
`kernel_qwen4_attn_mm` anche per quelle tre righe. La versione MM arrotonda
le query a half e segue la sua propria accumulazione/softmax.

- `ds4.c:58495`: parent prefill con più di 8 righe.
- `ds4_metal.m:49922`: scelta MM tramite `prefill_mm` indipendente dalla
  lunghezza della partizione.

Questo cambiamento è raggiungibile nel prompt da 5942 token. Non è
raggiungibile nei prompt da 29 o 575 token, che terminano prima della soglia.
Non può quindi essere l'unica causa comune ai tre casi.

## Quale riferimento viene preservato

Il messaggio di `2d6a207b` dichiara esplicitamente che il riferimento è
`e37f185`: il vecchio prefill half/unsplit. La convalida M1 SSD riporta
prefill contro quel riferimento da 0,48730135 a 1,90734863e-6, non parità
con `0aaea5a`. `89986c3c` dichiara invece parità con un prefill originale
single-pass: sul caso da 5760 token, la vecchia versione chunked aveva
errore massimo 0,597853661 e la candidata zero.

Quindi queste correzioni rendono il branch coerente con riferimenti precedenti
e con la versione single-pass. Non garantiscono la conservazione dei logits
del `main` `0aaea5a`, che contiene ancora il dispatch per numero di righe.
La differenza osservata dal tester è compatibile con questo cambio esplicito
di politica; l'attribuzione numerica dei delta 0,29 / 0,62 / 0,95 richiede
ablazioni sugli stessi dati e sullo stesso device.

## Mappa per l'isolamento, senza bisect ingenuo del merge

`0aaea5a` è il secondo parent di `b1af94b4`. Non è un antenato di
`2d6a207b` o `89986c3c`: questi commit sono stati scritti prima del secondo
merge di main. Un confronto diretto di quei commit con `0aaea5a` cambia
anche altre modifiche upstream e non isola un singolo intervento.

Ordine utile delle ablazioni sullo **stesso** snapshot head:

1. Ripristinare solo la selezione delle proiezioni base: F16/Q8 batch,
   split-K F32 consentito. Confrontare fronti 29 e 575; nessuna delle due
   traiettorie raggiunge la partizione sparse. Tenere tutto il resto uguale.
2. Sul prompt 575, ripristinare solo split-K delle matrici F32. Questo separa
   l'effetto della riduzione da quello FP32/half del prompt breve.
3. Sul prompt 5942, ripristinare solo la politica base della partizione
   attenzione; confrontare prima il layer 3, riga assoluta 2048, poi i logits.
4. Se resta una differenza, isolare il mixer HC di `028b43f3`, quindi il Q8
   GEMV/reduction. Confrontare il primo stage diverso, prima della selezione
   MoE, per distinguere errore iniziale da amplificazione tramite routing.

Il 3 ottobre è stato esaminato anche lo snapshot diagnostico `head/`, creato
dal coordinatore dai file committati di `02313128`. Rispetto a quel commit,
`ds4.c` cambia soltanto il bool locale `prefill`, aggiungendo
`DS4_DIAG_BASE_PREFILL_DENSE`; `ds4_metal.m` aggiunge
`DS4_DIAG_BASE_PREFILL_SPLIT_ONLY` al controllo split-K e una traccia stderr.
L'isolamento statico è corretto:

- `DENSE` rende identiche al base le tre condizioni di selezione F16/Q8/F32
  del helper, compresa la scelta della wrapper generic che permette split-K.
  Non cambia `g->projection_phase`: l'attenzione conserva la politica head.
- `SPLIT_ONLY` conserva i gate F16/Q8 head e consente soltanto lo split nei
  call site che prima passavano `allow_split_k=false`. I call site generic
  avevano già `true`, quindi restano invariati.
- Entrambi i diagnostici controllano la presenza della variabile: anche
  `=0` li attiva. I controlli richiedono `unset`, non `=0`.

Questo test confronta tre politiche sullo stesso programma head. Non
confronta due interi programmi base/head e non certifica la parità con base:
restano volutamente i kernel Q8, HC e le altre ottimizzazioni head.

Ulteriore candidato indipendente dal tuning: `028b43f3` modifica il mixer
generico F16 da `weight * silu(x)` a `(x * weight) * sigmoid(x)`, con
reassociation e contraction disabilitate; il pair impone invece `fma`.
È stato fissato il comportamento del compilatore M1. In un fronte finale
su una singola riga (`ds4.c:59115`) questo mixer è raggiungibile per tutti
e tre i prompt. Senza test M4 prima/dopo non si può assumere che il vecchio
compilatore M4 producesse esattamente quella stessa sequenza.

## Esclusioni e limiti

- I commit SSD di staging/cache/overlap non sono il percorso del tester
  residente. Le nuove varianti locali del bundle Q4 sono non committate,
  default off e non sono presenti nell'head remoto testato.
- `02313128` cambia i gate legacy M2-M4; il tester ha già isolato la sua
  neutralità numerica. Riattivare/disattivare quel gate non ripristina la
  politica `0aaea5a` dei due commit sopra.
- `04c08672` riporta il dot Q8 all'otto-termine originale e conserva il nuovo
  helper di riduzione. I test congelati M1 passano; ciò non certifica M4.
- Qui non è stato dimostrato che i delta misurati siano un calo di qualità:
  un confronto di logits individua aritmetica differente; una valutazione
  separata stabilisce qualità. La temperatura zero non impedisce a piccole
  differenze di cambiare un argmax o la selezione degli esperti.
- I conteggi sopra sono una prova statica dei dispatch, non risultati di
  un'esecuzione M4. L'hardware locale è M1 Max e il solo GGUF locale è Q4.

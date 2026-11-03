# Audit del runtime: drift Qwen fra 0aaea5a e 02313128

Audit statico, 2026-10-03. Nessuna esecuzione GPU, nessuna modifica al codice di produzione. Il confronto è fra i file nei due commit, non fra le modifiche locali non committate. Hardware del tester: Apple M4 Max, 64 GiB; Q2 residente; `--ctx 16384 --prefill-chunk 2048 --temp 0 --nothink`. Lunghezze dei tre prompt comunicate dal tester: 29, 575, 5942 token. Per la verifica numerica conclusiva servono le misure del processo principale: questo documento identifica percorsi raggiungibili e non attribuisce da solo l'intero errore numerico osservato.

## Differenza centrale: il confronto usa due politiche numeriche di prefill

Nel base **0aaea5a**, `qwen4_gemv_rows` sceglie per numero di righe:

- F16, T fra 4 e 64: kernel Qwen dense MM con input FP32 quando N <= 512 oppure T > 8.
- Q8, T fra 5 e 32: kernel Qwen batch MM con input FP32 quando i buffer contengono almeno 32 righe e K/N sono allineati.
- F32, T > 8: kernel Qwen dense MM con split-K automatico.

Nel branch **02313128**, il commit **2d6a207b** ha aggiunto `projection_phase=PREFILL` ai forward dei prompt. Il prefill disabilita esplicitamente le prime due scorciatoie e usa un nuovo ingresso dense MM che impedisce lo split-K. Gli ingressi decode/MTP conservano la loro politica separata.

Riferimenti nei file attuali: `ds4.c:58244` (politica prefill), `ds4.c:58250` (F16), `ds4.c:58269` (Q8), `ds4.c:59167` (ingresso prefill); `ds4_metal.m:52057` (split-K), `ds4_metal.m:52111` (ingresso prefill senza split).

Queste differenze restano anche disabilitando le ottimizzazioni hardware M2-M4 aggiunte da 02313128. Non sono condizionate dal riconoscimento del device. Quindi la neutralità del flag nel commento del tester non le esclude.

## Precisione e riduzioni

Il kernel `kernel_qwen4_dense_mm` in `metal/qwen4.metal:5049` conserva input e pesi dequantizzati in memoria FP32 e usa `simdgroup_float8x8`. Il kernel Q8 batch di quel file usa anch'esso operandi FP32.

Il prefill F16/Q8 generico di M4 usa matrici half e converte l'input FP32 in half durante lo staging. L'istanza F16 di `metal/dense.metal:3297` usa `simdgroup_half8x8`; la funzione generica Q8 e quella F16 le selezionano dal backend. Su M4 il percorso TensorOps destinato alle generazioni più nuove non interviene.

Per T=29 il cambio di politica modifica quindi **anche la precisione degli operandi**, oltre alla suddivisione della somma. Per le proiezioni F32 più lunghe, la precisione degli operandi resta FP32, ma cambia la catena degli accumuli: il base accumula sottointervalli K e somma i parziali; il branch accumula tutto K in una sola catena.

## Formati delle proiezioni di controllo

L'intestazione del Q4 locale conferma:

| Proiezione | K | N | Formato |
|---|---:|---:|---|
| HC down | 10240 | 320 | F16 |
| HC up | 320 | 10240 | F16 |
| SSM alpha | 2560 | 48 | F32 |
| SSM beta | 2560 | 48 | F32 |
| Router | 2560 | 512 | F32 |

Non è presente un Q2 locale: i suoi byte non sono stati ispezionati. La stessa forma/formato è però fissata dal convertitore comune `gguf-tools/qwen4_pack_to_qwen4exp.py:493`, `:504`, `:505`, `:530`. La conversione IQ2/Q2 di `gguf-tools/qwen4_iq2.py` sostituisce le matrici degli esperti, conservando i tensori di controllo del template. È dunque supportata dal codice la corrispondenza di questi tensori fra le ricette Q2 e Q4; resta da verificare l'identità concreta del file usato dal tester.

## Numero esatto degli split

Formula del base (`ds4_metal.m` al commit 0aaea5a, funzione `ds4_gpu_qwen4_dense_mm_tensor`):

```
tiles = ceil(N / 32)
threadgroups = tiles * ceil(T / 32)
want = threadgroups >= 128 ? 1 : ceil(128 / threadgroups)
splits = min(want, floor(ceil(K / 32) / 4), 64)
```

Per i tensori qui considerati il limite inferiore non serve. Il branch impone sempre uno split per F32 in prefill. F16 e Q8 T=29 vengono invece instradati verso i kernel generici, non verso questa funzione.

| T del forward prefill | SSM alpha/beta: gruppi | Split base -> branch | Router: gruppi | Split base -> branch |
|---:|---:|---:|---:|---:|
| 29 | 2 | 20 -> 1 | 16 | 8 -> 1 |
| 575 | 36 | 4 -> 1 | 288 | 1 -> 1 |
| 2048 | 128 | 1 -> 1 | 1024 | 1 -> 1 |
| 1846 | 116 | 2 -> 1 | 928 | 1 -> 1 |
| 5942 in un solo forward | 372 | 1 -> 1 | 2976 | 1 -> 1 |

HC-down a T=29 seleziona nel base dense MM FP32 con **13 split** (`ceil(128/10)=13`, limite K=80). HC-up a T=29 seleziona dense MM FP32 con **1 split**. Nel branch entrambi usano il percorso F16 generico half. Lo stesso ragionamento si applica alle proiezioni Q8 raggiungibili del trunk a T=29: il base sceglie il batch FP32, il branch il generico half (T=29 supera il limite matvec esteso di 16).

`qwen4_graph_linear` non fonde alpha/beta in questi forward: `qwen4_graph_fused` ammette soltanto T<=2 e l'esatto verify T=3. Quindi gli split della tabella sono realmente raggiungibili, non un calcolo per un ingresso inutilizzato. Le proiezioni alpha/beta compaiono nei 36 layer lineari del trunk.

## Applicazione ai tre prompt

**29 token, singolo chunk.** Le due versioni cambiano già il down HC del primo layer: FP32 e split-K nel base contro half generico nel branch. Cambiano anche HC-up, proiezioni Q8, alpha/beta e router. Non serve una differenza di chunk, un MTP attivo o un'ottimizzazione specifica M4 per avere logits diversi. È il caso migliore per isolare la politica dense con un test A/B.

**575 token, singolo chunk.** F16 e Q8 non rientrano nei piccoli batch del base e restano sui percorsi generici. La differenza raggiungibile nel runtime è lo split-K **4 -> 1** di alpha/beta di tutti i layer lineari. Perturbazioni numeriche in alpha e beta modificano le porte del delta-net e lo stato ricorrente; i successivi layer e il routing MoE possono amplificarle. L'audit non quantifica questa amplificazione né prova che da sola produca l'errore 0,62 riportato.

**5942 token, chunk 2048.** I forward sono 2048+2048+1846. Alpha/beta del tail passano da due split a uno. Esiste inoltre una modifica distinta, commit **89986c3c**: il secondo chunk contiene tre righe ancora dense alle posizioni 2048..2050; il base sceglie per loro attenzione scalar/decode perché la sottopartizione è corta, il branch mantiene l'attenzione matriciale del chunk padre. Questo cambia la riduzione e può contribuire al drift del prompt lungo con chunk multipli.

**5942 token, un solo chunk.** Né alpha/beta né router hanno split-K multipli nel base. La partizione dense/sparse contiene rispettivamente 2051 e 3891 righe, entrambe abbastanza lunghe da scegliere attenzione matriciale nei due commit. Le differenze di runtime sopra non spiegano da sole il drift residuo in questo caso. Restano da isolare i cambiamenti numerici nei kernel, fra cui l'aritmetica HC del mixer finale T=1, applicata a tutti i prompt. La loro indagine appartiene all'audit separato dei kernel.

## Cosa è stato escluso nel runtime

- Non cambiano epsilon del modello, scale dell'attenzione, frequenza base RoPE, pesi caricati o semantica della normalizzazione nei forward del trunk osservati in questo diff.
- Lo skip delle query dell'indexer nel prefisso dense rimuove risultati che il forward non consuma; le chiavi vengono ancora proiettate e aggregate. Per il forward intero lungo le query restano tutte proiettate.
- Il threshold MoE residente è **64** in entrambi i commit. Per 29 token entrambi usano i kernel per token; per 575 e i chunk grandi entrambi usano MM. Il nuovo threshold **8** appartiene allo streaming SSD.
- La preparazione MTP aggiunta non è raggiungibile con `mtp_R == NULL`, quindi non spiega le misure con MTP disattivato. Il suo costo di prefill è una questione distinta dal drift.

## Limiti e interpretazione della compatibilità

Il commit 2d6a207b non dichiara compatibilità numerica con il main 0aaea5a: dichiara il ripristino del prefill del precedente **e37f185**, prima delle ottimizzazioni dei piccoli batch. Il suo messaggio riporta una riduzione del drift rispetto a quella vecchia reference. Poiché main 0aaea5a contiene ormai la politica float-batch/split-K, parità con e37f185 e parità con 0aaea5a sono obiettivi diversi.

Un confronto con 0aaea5a può quindi trovare una differenza intenzionale di politica, non necessariamente una corruzione di memoria. Questo non prova quale politica dia migliore qualità e non giustifica attribuire tutti i token diversi a rumore innocuo. Il risultato necessario è un isolamento numerico sullo stesso modello e hardware, seguito da un confronto con la reference scelta per la release.

I piccoli errori delle proiezioni possono propagarsi nel modello e cambiare top-k o argmax quando due candidati sono vicini. Non si possono dedurre né l'errore finale esatto, né i token divergenti, dalla sola variazione dell'ordine dei floating point. In particolare non è dimostrata staticamente l'entità 0,29/0,62/0,95 del commento.

La riproduzione possibile localmente, Q4 in SSD su M1 Max, deve essere etichettata come isolamento della politica. Non riproduce il Q2 residente su M4 del tester. In particolare T=29 in SSD usa MoE MM, mentre T=29 residente usa MoE per token; un A/B sullo stesso snapshot SSD mantiene questa differenza fissa e isola comunque la modifica dense, ma non identifica da solo l'intero delta del tester.

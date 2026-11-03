# Diagnosi del drift Qwen — 3 ottobre 2026

La principale differenza numerica identificata tra `0aaea5a` e `02313128`
è la politica di prefill introdotta da **`2d6a207b`**. Cambia precisione e
ordine degli accumuli; non è il nuovo gate M2/M3/M4 di `02313128`.
Una prova controfattuale sullo stesso snapshot HEAD riproduce sia differenze
ei logits sia una successiva divergenza dei token greedy.

## Segnalazione e riferimento

[Commento del tester](https://github.com/antirez/ds4/pull/1056#issuecomment-5963229415):
M4 Max, Q2 residente; il gate è neutro rispetto alla patch opt-in. Il confronto
con main `0aaea5a` presenta delta dei primi logits circa 0,29 / 0,62 / 0,95.
I dettagli dei prompt e dei comandi sono nel
[commento precedente](https://github.com/antirez/ds4/pull/1056#issuecomment-5922543210).

`2d6a207b` preservava la reference precedente **`e37f185`**, con prefill
half e senza split. Main `0aaea5a` seleziona invece batch FP32 e split-K
anche per prompt. I test contro la reference e quelli contro main verificano
quindi politiche numeriche diverse.

## Punto nel codice

- `ds4.c:58242`: riconosce la fase PREFILL.
- `ds4.c:58244`: esclude i piccoli batch F16 FP32 dal prefill.
- `ds4.c:58260`: esclude i piccoli batch Q8 FP32 dal prefill.
- `ds4_metal.m:52070`: il controllo `allow_split_k` decide la somma per piani.
- `ds4_metal.m:52111`: wrapper prefill passa `allow_split_k=false`.

Le proiezioni GDN alpha/beta statiche sono F32, 2560 ingressi e 48 uscite.
Le tracce della prova confermano questo cambiamento:

| Righe del chunk | Split in main | Split in HEAD |
| --- | ---: | ---: |
| 29 | 20 | 1 |
| 575 | 4 | 1 |
| 2048 | 1 | 1 |
| 1846, tail di 5942 | 2 | 1 |

Quattro somme parziali e una sola catena non arrotondano allo stesso modo
in FP32. Alpha e beta influenzano le porte e lo stato ricorrente del GDN;
le differenze persistono nei successivi layer e possono cambiare routing
MoE e argmax. Non si possono dedurre i delta finali dalla sola formula:
per questo è stata eseguita l'ablazione sul modello reale.

## Prova isolata eseguita

**Hardware e modello locale:** Apple M1 Max 32 GiB, Qwen Q4, SSD streaming.
Context 16384, chunk 2048, temperatura zero, `--nothink`, MTP disattivato.
Rome29 è `narrami la storia di roma`; book575 e book5942 sono prefissi
costruiti da `promessi_sposi.txt` per ottenere esattamente quei conteggi.
I due prompt lunghi non sono i file esatti del tester.

Snapshot dei soli file committati in `head/`, separato dalla working tree
con esperimenti non committati. Stesso eseguibile e shader per ogni arm;
ambiente DS4 ripulito. Il diagnostico DENSE ripristina soltanto i gate
F16/Q8 e split-K del base; SPLIT_ONLY ripristina solo lo split-K F32.
La fase del graph, l'attenzione, il mixer HC e i kernel di decode restano
quelli di HEAD. I diagnostici sono nel solo snapshot, non in produzione.

### Primo fronte di logits: 10 esecuzioni

248320 valori FP32 finiti confrontati in ogni dump.

| Fixture | Cambiamento isolato | Max delta logits |
| --- | --- | ---: |
| 29 | tutta la vecchia politica dense | 0,18282795 |
| 29 | solo vecchio split-K F32 | 0,08559227 |
| 575 | tutta la vecchia politica dense | 0,41702342 |
| 575 | solo vecchio split-K F32 | 0,41702342 |
| 5942 | solo vecchio split-K F32 nel tail | 0,33132219 |

I controlli HEAD ripetuti per 29 e 575 token sono identici byte per byte.
A 575 token gli arm DENSE e SPLIT_ONLY producono JSON interamente identici:
il cambio degli accumuli F32 è sufficiente per questo delta. A 5942 le
tracce confermano split invariati nei due chunk da 2048, modificati solo
nei 72 dispatch alpha/beta del tail da 1846. L'attenzione resta identica
fra i due arm. Il primo argmax non cambia nelle cinque ablazioni.

### Continuazione greedy: quattro esecuzioni da 64 token

Stesso snapshot, DENSE contro HEAD, senza MTP. Il flag DENSE modifica
il prefill; nei passi ordinari di decode T1 entrambi gli arm selezionano
gli stessi percorsi. Si confrontano token ID e bytes, non solo testo.

| Fixture | Prefisso generato identico | Primo token diverso, contando da 1 |
| --- | ---: | ---: |
| 29 | 58 token | 59: HEAD ` capit`, vecchia politica ` ere` |
| 575 | 19 token | 20: HEAD ` **"`, vecchia politica ` ***` |

Questo collega causalmente l'alterazione del prefill alla divergenza dei
token successivi: i kernel decode non sono stati cambiati fra i due arm.
Per 575 token, il precedente dump iniziale DENSE era identico al controllo
SPLIT_ONLY; la continuazione SPLIT_ONLY non è stata eseguita separatamente.

## Altri cambiamenti da distinguere

**`89986c3c`** cambia l'attenzione delle tre righe dense interne 2048..2050
quando il chunk padre è lungo: main sceglie la versione scalare, HEAD
mantiene quella matriciale con query half. Raggiungibile nel prompt 5942
con chunk2048; non spiega i primi due prompt. Il messaggio del commit
riporta già la precedente localizzazione a layer3, token2048, e la verifica
Q2/M1 contro single-pass. Non è stata misurata la sua quota sul M4 del tester.

**`028b43f3`** impone l'ordine `(x*weight)*sigmoid(x)` senza FMA nel mixer
HC F16 generico. Può contribuire a differenze fra compilatori/device:
il mixer finale usa T1 anche dopo un prompt lungo. È un candidato secondario
non isolato quantitativamente qui; non serve a spiegare le ablazioni dense.

I gate legacy M2/M3/M4, MTP preparation e SSD cache/overlap non spiegano
la comune variazione numerica del prefill qui isolata. Il costo MTP del
commento è un problema prestazionale separato.

## Interpretazione e limiti

È dimostrato localmente che la politica di `2d6a207b` cambia logits e token.
Non è stata riprodotta l'esatta combinazione **M4 Max/Q2 residente**, né
attribuita ogni quota dei suoi delta 0,29/0,62/0,95. Nessun confronto di
qualità stabilisce qui quale reference sia più corretta. Ripristinare
l'aritmetica main annullerebbe parte delle precedenti correzioni di parità
con `e37f185`/single-pass: non sarebbe una correzione numericamente neutra.

Durante l’indagine non sono stati modificati file di produzione.
Il successivo commit archivia soltanto documentazione, diagnostici e prove.
Le prove non sono benchmark di prestazioni.

## Materiale verificabile

- `diagnostic.patch`: sole modifiche diagnostiche rispetto a HEAD.
- `snapshot-manifest.json`, `fixtures.json`: provenienza e token ID dei prompt.
- `dense-ab/results.json`, dump e log: ablazioni dei primi logits.
- `greedy-ab/results.json`, dump top20 e log: 64 passi greedy.
- `run-dense-ab.py`, `run-greedy-ab.py`: runner seriali ripetibili.
- `commit-audit.md`, `runtime-audit.md`, `arithmetic-audit.md`: audit separati.
- `validation-audit.md`: verifica indipendente di provenienza e risultati.

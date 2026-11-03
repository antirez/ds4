# Verifica indipendente delle ablazioni dense

Verifica del 3 ottobre 2026, dopo `dense-ab/results.json: status=COMPLETE`
e `greedy-ab/results.json: status=COMPLETE`: dieci dump del primo fronte
e quattro generazioni greedy da 64 token. Audit CPU e lettura dei file;
questo revisore non
ha compilato, eseguito la GPU o modificato file di produzione.

## Provenienza e isolamento

- Il manifest indica il commit `0231312820fb0ba0e8953606976461644511bcab`.
  Tutti i 103 percorsi del manifest sono stati confrontati byte per byte
  con i blob Git di quel commit. I soli file diversi sono `ds4.c` e
  `ds4_metal.m`, contenenti esclusivamente i diagnostici descritti in
  `commit-audit.md`. I 101 altri file, inclusi gli shader Metal, coincidono.
- Il runner usa sempre lo stesso eseguibile assoluto `head/ds4` e
  `cwd=head`. Questo è significativo: il backend carica e concatena i
  file `metal/*.metal` dalla working directory. Non legge gli shader
  non committati della working tree di produzione.
- Prima delle esecuzioni, il runner rimuove **tutte** le variabili con
  prefisso `DS4_`; poi aggiunge la traccia comune e il diagnostico del
  singolo arm. I record `env` confermano questi insiemi. Non eredita quindi
  override degli shader, della chunk policy o della matematica Metal.
- Il runner memorizza gli SHA-256 di 29 file: eseguibile, due sorgenti host
  e 26 shader. Dopo ogni run verifica che siano immutati. Il revisore ha
  verificato nuovamente tutti questi hash sul risultato finale: coincidono.
  SHA-256 dell'eseguibile:
  `8e96b07704e89b060a66d7131c895632f729d37b9cdd4f254ed1d6e8442ec9b7`.
- Tutti i comandi usano Q4 locale, Metal SSD, ctx 16384, chunk 2048,
  `--nothink`, temperatura zero e `--dump-logits`. Il file GGUF intero non
  è hashato dal runner; il confronto usa lo stesso percorso modello e
  processi seriali. Non è un confronto di due build base/head.

Il diagnostico `DENSE` rende false le tre protezioni nuove per il prefill
nel helper delle proiezioni: F16 batch, Q8 batch, wrapper F32 unsplit.
Il diagnostico `SPLIT_ONLY` conserva i primi due gate head e consente lo
split-K del wrapper prefill F32. Nessuno dei due modifica la politica
di attenzione `89986c3c`, il mixer HC o gli altri kernel head.

Le variabili diagnostiche sono controllate per presenza: il controllo
corretto è rimuoverle, come fa il runner, non impostarle a zero.

## Dati ricalcolati

Il revisore ha riletto tutti i dieci dump, verificato 248320 logits finiti
per dump e ricalcolato gli SHA-256 FP32, il massimo delta, il delta medio
e il numero di pattern FP32 diversi. Tutti coincidono con il report.
`ds4_cli.c` serializza con `%.9g`, sufficiente al round trip dei float32;
la verifica dei bit usa `struct.pack('<f', value)`.

| Fixture | Ablazione contro head | Max delta logits | Pattern FP32 diversi | Argmax |
| --- | --- | ---: | ---: | --- |
| 29 token | politica dense base | 0,18282795 | 248318 | invariato, 8482 |
| 29 token | solo split-K base | 0,08559227 | 248320 | invariato, 8482 |
| 575 token | politica dense base | 0,41702342 | 248320 | invariato, 1919 |
| 575 token | solo split-K base | 0,41702342 | 248320 | invariato, 1919 |
| 5942 token | solo split-K base | 0,33132219 | 248320 | invariato, 1919 |

Il controllo head ripetuto è **byte-identico nell'intero JSON** per
29 e 575 token, oltre a coincidere nei pattern FP32. Nel test da 5942
token non è stata eseguita una ripetizione head; il confronto ha due run.

Per 575 token, `base_dense` e `base_split` sono anch'essi **byte-identici
nell'intero JSON**, non solo uguali nel massimo delta. Il loro SHA-256
dei logits FP32 è
`540d24a50a365976aafe07b56f508734abdeae4b51533fbe509e8d343df0b5ca`.

## Tracce e interpretazione

I log confermano i dispatch calcolati nell'audit statico:

- 29 token: alpha/beta F32 2560→48, 72 dispatch, split 1 in head e 20
  nell'arm split-only. I router F32 2560→512 passano da 1 a 8 split.
  Nell'arm dense-base compaiono inoltre 96 HC-down F16 10240→320
  con 13 split e 96 HC-up F16 320→10240 con un solo split.
- 575 token: alpha/beta, 72 dispatch, da 1 a 4 split; i 48 router
  restano con un solo split. Queste sono le sole variazioni raggiungibili
  della politica dense per questo numero di righe. Perciò l'identità
  dense-base/split-only è coerente con il codice e la traccia: il drift
  osservato in questa fixture deriva dall'ablazione dello split-K F32.
- 5942 token: nei due chunk da 2048 righe sia alpha/beta sia router
  restano con un solo split. Nel chunk finale da 1846 righe i 72
  dispatch alpha/beta passano da 1 a 2 split; i router restano a 1.
  L'attenzione è la stessa nei due arm. Il risultato mostra una differenza
  dei logits anche senza cambiare la partizione di attenzione.

Queste ablazioni dimostrano sul programma locale un meccanismo causale:
la politica prefill/split-K di `2d6a207b` basta a cambiare il primo fronte
di logits, prima della generazione. Per il prompt breve il resto della
politica dense aggiunge una differenza distinta. Non identificano qui
il primo tensore interno diverso o la selezione MoE che amplifica l'errore.

## Generazione greedy: divergenza dei token riprodotta localmente

Successivamente il coordinatore ha eseguito `run-greedy-ab.py`: due arm
head/dense-base per le fixture 29 e 575, 64 token ciascuno. I quattro
comandi usano lo stesso eseguibile e CWD del confronto dense, lo stesso
modello Q4 SSD e gli stessi argomenti ctx/chunk/nothink. Le variabili DS4
ereditate sono rimosse: head non ne riceve nessuna, dense-base riceve
soltanto `DS4_DIAG_BASE_PREFILL_DENSE=1`.

Il revisore ha ricalcolato i quattro SHA-256 dei JSON, tutte le liste di
token e le prime posizioni diverse. Per ogni prefisso comune coincidono
non solo gli ID, ma i record completi `selected` (testo e bytes). Tutti i
risultati sono coerenti con il report:

| Fixture | Token comuni prima della divergenza | Primo token diverso, contando da 1 | Head | Dense-base |
| --- | ---: | ---: | --- | --- |
| 29 token | 58 | 59 | ID 58373, ` capit` | ID 36989, ` ere` |
| 575 token | 19 | 20 | ID 173976, bytes `[32,42,42,34]` | ID 16654, bytes `[32,42,42,42]` |

La divergenza avviene quando l'ordine dei due migliori candidati cambia:

- Roma, step zero-based 58: head ha logits 18,9806519 / 18,9401970 per
  ` capit` / ` ere`; dense-base ha 18,9373436 / 18,9717941. Il vantaggio
  del vincitore passa quindi da 0,0404549 a 0,0344505, con ordine inverso.
- Libro, step zero-based 19: head ha logits 20,8863659 / 20,8012505 per
  ID 173976 / 16654; dense-base ha 20,8767471 / 20,8784676. Nell'arm
  dense-base il vantaggio del vincitore è appena 0,0017205.

Non interviene la casualità del sampler: `run_logprob_dump` usa direttamente
`ds4_session_argmax`, poi `ds4_session_eval`. Non invoca un ciclo MTP.
Quest'ultima API chiama `qwen4_graph_forward_token`, che imposta
`QWEN4_PROJECTION_DECODE`. Nel helper diagnostico, il bool locale
`prefill` è quindi false in entrambi gli arm, indipendentemente dalla
presenza della variabile. **Il diagnostico cambia il percorso del prompt,
non la politica dei kernel T1 della generazione.** Stati GDN/KV differenti
dopo il prompt possono persistere anche mentre gli ID successivi restano
uguali, poi capovolgere un argmax quasi pari. Dopo il primo ID differente
i contesti stessi divergono e non si devono più attribuire ogni differenza
successiva esclusivamente all'aritmetica iniziale.

Gli attuali hash dei 29 file controllati coincidono ancora con quelli
registrati dal confronto dense. Il runner greedy, a differenza di quello
dense, non registra o verifica gli hash dopo ogni singolo run; questa
verifica è stata effettuata dal revisore a posteriori. Non sono state
eseguite ripetizioni aggiuntive delle quattro generazioni.

## Confini dell'attribuzione

- Hardware e modello sono **M1 Max, Q4, SSD**, mentre il tester usa
  **M4 Max, Q2, residente**. Non è stata replicata quella combinazione.
- Le fixture `book575` e `book5942` hanno i conteggi giusti ma sono testi
  costruiti per il test locale. In particolare, il SHA-256 di `book575`
  è `121bb1c1ed88f048408c3c468567111566d80ae42eb47c768f743a1a43d4b046`,
  diverso dal `5da9ad53...` riportato per il Roma575 del tester. I tre
  delta locali non sono la riproduzione dei suoi 0,29 / 0,62 / 0,95.
- Gli arm chiamati `base_dense`/`base_split` ripristinano politiche del
  base nello stesso snapshot head. Non sono l'eseguibile `0aaea5a` e
  non dimostrano parità byte per byte con tutti i logits del base.
- Nei dump locali l'argmax del primo token resta uguale in ogni arm.
  Le quattro generazioni successive riproducono invece una divergenza
  greedy locale entro 64 token per entrambe le fixture corte. Questo
  sostituisce il limite del primo audit, che aveva controllato solo il
  primo fronte; non riproduce le traiettorie esatte M4/Q2 del tester.
- Non sono misure di qualità del modello né benchmark prestazionali:
  un dump di logits e una traccia stderr non valutano la qualità e i
  tempi del runner includono avvio, compilazione shader, caricamento e IO.
- La possibile quota del mixer HC `028b43f3` e della partizione attenzione
  `89986c3c` nel drift esatto del tester resta da isolare sul suo ambiente.

Esito dell'audit: **PASS** per provenienza, isolamento e integrità dei dati;
attribuzione sperimentale valida entro l'ambito locale indicato sopra.

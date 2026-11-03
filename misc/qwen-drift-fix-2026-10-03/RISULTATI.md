# Correzione del drift Qwen residente: risultati locali

La correzione supera i sette confronti A/B locali: sulla fixture residente i logits dopo prompt da 29, 575 e 5942 token sono identici a main; le due continuazioni greedy da 64 token hanno JSON identici. Sul Q4 completo in SSD streaming i logits dei prompt da 29 e 575 token rimangono identici al binario precedente alla correzione. I risultati sono registrati in `fix-ab/results.json` (14 esecuzioni, stato `PASS`).

La stessa serie è stata ripetuta sulla snapshot delle sorgenti selezionate nell'indice per il commit, senza gli esperimenti locali preesistenti: anche `release-staged-ab/results.json` registra 14 esecuzioni e sette confronti PASS. È una verifica separata della versione destinata al commit; il primo report riguarda la working tree originaria.

Le prove sono state eseguite dal processo principale sul M1 Max locale da 32 GiB. Non sono una replica del Q2 residente su M4 Max del tester e non misurano un guadagno di throughput.

## Obiettivo e provenienza

Il tester del [commento GitHub 5963229415](https://github.com/antirez/ds4/pull/1056#issuecomment-5963229415) usa Q2 residente su M4 Max. L'indagine identifica una differenza di politica del runtime: il prefill residente ereditava la politica half/unsplit SSD, mentre main usa piccoli batch FP32 e split-K. Un'altra differenza riguarda l'attenzione delle sottopartizioni brevi al confine dense/sparse.

Reference main congelata: `0aaea5a238fb41a35106a551e73c8409dfb751ac`. Il branch remoto esaminato nel commento è `0231312820fb0ba0e8953606976461644511bcab`. Il punto di partenza effettivo della modifica locale e gli hash dei file sono in `starting-state.json`; il file registra HEAD `575369b19d8754f7eff69b9fdaa9a4c3684b825a`, inclusi i cambiamenti locali preesistenti. Questi ultimi devono rimanere separati dalla correzione.

La modifica produttiva limita a `g->ssd_streaming` la politica di proiezione legacy/unsplit e il mantenimento dell'attenzione del chunk padre. Il prefill residente riprende la politica di main. Decode e predictor MTP mantengono la loro fase `DECODE`; nessuna nuova variante semantica dietro un flag permanente è necessaria.

## Fixture residente locale e limiti

Il Q4 originale è troppo grande per essere residente sul M1 Max locale da 32 GiB. Per provare il normale percorso residente senza modificare il validatore del modello, è stata preparata una fixture con 48 layer alle forme reali, usando alias dei pesi originali dei blocchi 0–3 ripetuti modulo quattro. Tokenizer, embedding, output e 112 payload distinti conservano i byte originali. Tutti i 512 esperti e le larghezze reali rimangono presenti. La tabella PLE usa una riga BF16 zero e il predictor è assente.

La fixture è sintetica e non è il Qwen completo. Serve per il confronto sullo stesso file fra binari, non per valutare qualità, throughput del modello completo o parità Q2/M4. Informazioni e offset sono in `resident-fixture-plan.json`; la scrittura eseguita dal processo principale registra gli hash dei payload in `resident-fixture-written.json`.

Mapping residente: 7,953,727,488 byte (7.4075 GiB). File scritto: 7,953,727,808 byte. Il primo load ha individuato un errore nel numero delle teste PLE della fixture, non nel codice produttivo: 24 invece delle 16 previste da `(3 - 1) * 8`. Il generatore è stato corretto e il processo principale ha riparato il solo header, conservando tutti gli offset e i payload. L'header corretto ha SHA-256 `5892d2e715b86de3f3a36c8686ccab0ccb7f4c04c0ad840214bcfbfbcb21a8f0`. Il caricamento successivo e le dieci esecuzioni residenti A/B sono riusciti.

Il manifest della scrittura conserva lo stato al momento della riparazione (`not yet loaded or GPU-tested`); la prova successiva del caricamento e dell'esecuzione è `fix-ab/results.json`. I modelli sono stati sorvegliati tramite device, inode, dimensione, mtime e ctime. Non è stato calcolato un SHA-256 dell'intero Q4 da 177,280,286,720 byte; i 112 payload copiati nella fixture hanno invece gli hash registrati durante la copia.

## Audit statico dello stato streaming

Lettura del codice produttivo modificato, senza build o GPU da parte dell'agente di audit:

| Percorso | Stato osservato |
|---|---|
| Allocazione del grafo | `qwen4_graph_alloc` azzera il descriptor; il chiamante pubblico assegna poi il modo |
| One-shot normale | `generate_qwen4_metal_argmax` assegna `g->ssd_streaming` prima di reset e prefill |
| Creazione di sessione | Assegna `s->qwen4_graph.ssd_streaming=e->ssd_streaming` prima di reset e sincronizzazione |
| Reset, rewind, snapshot e payload restore | Modificano stato e cache senza azzerare il campo streaming |
| Arena condivisa | È creata soltanto quando l'engine è residente; il campo false è corretto |
| Microbatch nativo | Il controllo di ammissibilità esclude engine SSD; il grafo per riga copia la sessione e imposta `DECODE` |
| MTP cache, draft, chain e verify | Mantengono il campo streaming del grafo e selezionano temporaneamente `DECODE` |
| Test generation con grafo diretto | Assegna esplicitamente il modo streaming dell'engine |

Nessun percorso pubblico ordinario esaminato perde il campo streaming al reset o al restore. Due grafi privati della diagnostica `--first-token-test` / `DS4_QWEN4_GPU` e `DS4_QWEN4_FT_LIST` rimangono implicitamente residenti, come prima: non ricevono un parametro SSD. Non vanno usati per attribuire a SSD un test del normale percorso streaming. L'oracle del nuovo test deve impostare il campo in modo esplicito.

## Reference indipendenti e portata delle verifiche

Le prove usano tre reference diverse. Il test `test_qwen4_resident_arithmetic` contiene il dispatch delle proiezioni copiato da `0aaea5a`, ma chiama i kernel del backend corrente: isola la scelta della precisione e dello split-K, senza congelare gli shader storici. Per l'attenzione usa chiamate indipendenti all'API originale sui singoli intervalli e sul chunk padre. La sua reference SSD conserva la politica half/unsplit e la selezione per il chunk padre.

`hc-reference.m` compila separatamente il corpo dello shader HC generico congelato da `0aaea5a` e gli shader correnti generico, prefetch e reuse, con le stesse opzioni Metal di default. Questa verifica confronta l'aritmetica degli shader, senza utilizzare il dispatch come oracle.

Infine, la serie residente A/B usa due binari completi: main `0aaea5a` nella snapshot `base0aa/` e il binario corretto della working tree, ciascuno eseguito dalla propria directory sorgente. La serie SSD usa la snapshot completa `before-fix/` contro il binario corretto. Nessuna delle due serie è un confronto fra due flag dello stesso binario.

## Test mirati e build

| Verifica | Reference | Stato / risultato |
|---|---|---|
| Build normale ds4 e server | Makefile normale | Make li dichiara aggiornati; timestamp di sorgenti, oggetti e binari verificati dal processo principale |
| Build test resident-arithmetic, generation e memory | Makefile normale | Compilazione/link terminati con exit code 0, osservati nei tool del processo principale |
| Test CPU di pianificazione memoria | Test del repository | PASS, exit code 0 osservato nei tool del processo principale; workspace, staging recurrent/MTP, budget e pianificazione delle query attention OK |
| Proiezioni F16, Q8 e F32 alle forme reali; prefill/decode residente e SSD | Dispatch main congelato e reference SSD, con kernel correnti | PASS nei quattro assetti retained/unretained × math-safe off/on; `resident-arithmetic-final.log` |
| Attenzione dense/sparse: tre righe a posizione 2048, chunk padre lungo e corto | Reference indipendenti per intervallo/chunk padre | PASS nei medesimi quattro assetti; `resident-arithmetic-final.log` |
| Controllo negativo con runtime precedente alla correzione | Stesso test e fixture sintetiche | Fallisce come previsto: HC-down F16, prefill residente T=8, 2418/2560 float differenti, errore massimo 5.7220459e-06; `resident-negative.log` |
| Mixer HC generico, prefetch e reuse | Shader HC main congelato, compilato separatamente | PASS: 71,793 valori confrontati, 0 differenti, errore massimo 0; `hc-reference.log` |
| Q4 completo SSD: prompt da 29 token e 8 passi decode teacher-forced | Politica legacy/unsplit SSD; grafo one-shot e sessione | PASS: 9 frontiere del vocabolario completo identiche bit per bit; `ssd-generation-reference.log` |
| Fixture residente: prompt da 575 token e 8 passi decode teacher-forced, ctx 16384, chunk 2048 | Grafo one-shot e sessione ordinari dello stesso binario corretto | PASS: 9 frontiere del vocabolario completo identiche bit per bit; `resident-generation.log` |
| Load della fixture corretta nei due binari residenti | Stesso file e header | PASS: tutte le esecuzioni residenti hanno return code 0 |

I casi di proiezione coprono HC-down/up F16 alle soglie T=1, 3, 8, 9, 29, 64 e 65; Q8 a T=1, 3, 4, 5, 8, 9, 16, 17, 29, 32 e 33; GDN alpha F32 a T=1, 3, 8, 9, 29, 575, 1846 e 2048; router F32 a T=29 e 575. Il test controlla anche finitezza, guardie, input e pesi immutati. Il controllo negativo mostra che l'oracle rileva la vecchia politica residente anziché limitarsi a confrontare due percorsi equivalenti.

## Confronto dei binari completi

Configurazione comune: `--metal --ctx 16384 --prefill-chunk 2048 --nothink --temp 0`, con `DS4_QWEN4_PREFILL_CHUNK=2048`. La serie SSD aggiunge `--ssd-streaming`. I tre prompt locali, i relativi hash, i comandi, i log e gli hash dei JSON sono registrati in `fix-ab/results.json`.

| Modalità / modello | Prompt | Reference → candidato | Risultato |
|---|---|---|---|
| Residente, fixture con pesi Q4 ripetuti | Rome, 29 token | main `0aaea5a` → fix | 248,320 logits: 0 float32 differenti, errore massimo 0, stesso argmax, JSON identico |
| Residente, stessa fixture | Book, 575 token | main `0aaea5a` → fix | 248,320 logits: 0 float32 differenti, errore massimo 0, stesso argmax, JSON identico |
| Residente, stessa fixture | Book, 5942 token, chunk 2048 | main `0aaea5a` → fix | 248,320 logits: 0 float32 differenti, errore massimo 0, stesso argmax, JSON identico |
| Residente, stessa fixture | Rome, 29 token + 64 greedy | main `0aaea5a` → fix | Tutti i 64 token e i JSON delle logprob identici; nessun primo passo divergente |
| Residente, stessa fixture | Book, 575 token + 64 greedy | main `0aaea5a` → fix | Tutti i 64 token e i JSON delle logprob identici; nessun primo passo divergente |
| SSD, Q4 completo originale | Rome, 29 token | before-fix → fix | 248,320 logits: 0 float32 differenti, errore massimo 0, stesso argmax, JSON identico |
| SSD, Q4 completo originale | Book, 575 token | before-fix → fix | 248,320 logits: 0 float32 differenti, errore massimo 0, stesso argmax, JSON identico |

La continuazione greedy residente confronta token e logprob esportate, non l'intero vettore dei logits a ogni passo. I due test generation teacher-forced separati, residente e SSD, confrontano invece nove vettori completi usando gli stessi token di continuazione. Quello residente verifica one-shot contro sessione nel binario corretto; il confronto storico con main è la serie A/B dei binari completi. La serie residente da 5942 token verifica anche il confine dense/sparse e il chunk finale da 1846 token.

I tempi riportati dal runner includono avvio, inizializzazione, cache del modello, prefill, I/O dei dump e, dove richiesto, decode. Ogni braccio ha una sola esecuzione per caso e l'ordine può favorire la cache calda. Nessuna percentuale di miglioramento o regressione prestazionale viene dedotta da questi tempi.

## Verifica della versione selezionata per il commit

`release-staged-sources.json` registra 94 file estratti dall'indice dopo aver selezionato soltanto la correzione. La snapshot `release-staged/` è stata compilata e confrontata con gli stessi input storici, modelli e prompt del primo A/B. `release-staged-validation.md` documenta la corrispondenza con l'indice; la lettura successiva degli hash conferma tutti i 94 sorgenti e i binari dei tre bracci invariati.

| Verifica separata del candidato selezionato | Risultato |
|---|---|
| Build ds4, server e test | PASS; `release-staged-build.log` |
| Arithmetic nei quattro assetti Metal | PASS; `release-staged-arithmetic.log` |
| Pianificazione memoria CPU | PASS; `release-staged-memory.log` |
| A/B residente contro main: logits 29/575/5942 e due greedy da 64 token | Zero differenze nei logits e JSON identici; continuazioni greedy e logprob JSON identiche |
| A/B SSD Q4 completo contro before-fix: logits 29/575 | Zero differenze nei logits e JSON identici |
| One-shot/sessione residente, 575 token + 8 decode | Nove frontiere complete identiche; `release-staged-resident-generation.log` |
| One-shot/sessione SSD legacy, 29 token + 8 decode | Nove frontiere complete identiche; `release-staged-ssd-generation.log` |

Stato finale della seconda serie: PASS, 14 esecuzioni e sette confronti. SHA-256 di `release-staged-ab/results.json`: `f04451af079f4b77abaac83423f8b842e24cd832c573fbec51ef2d6a4b43526d`. SHA-256 del manifest delle 94 sorgenti: `b428f91cf7333e3e1059e47e8b456f15b38dc9ffef6e551315a5153d47f80e95`. Questa ripetizione non estende la verifica a Q2/M4, MTP o prestazioni.

## Integrità dell'audit

Un controllo successivo in sola lettura ha confermato gli hash di tutti i file sorgente e dei binari registrati dal runner: 88 file per `rootfix`, 87 per `base0aa`, 88 per `before-fix`, senza mismatch. Anche `run-fix-ab.py` e `source-snapshots-manifest.json` corrispondono agli hash del risultato. Il manifest delle snapshot descrive il momento precedente a compilazione ed esecuzione; i risultati successivi sono nei log e nel JSON A/B. Questa integrazione modifica soltanto il rapporto.

La patch circoscritta è in `correction-only.patch`; `starting-state.json` e `fix-source-hashes.json` conservano la provenienza delle modifiche locali preesistenti e della correzione. `before/` è soltanto il backup locale originario e non viene archiviato: `prepare-snapshots.py` ricostruisce le 89 sorgenti base da Git `0aaea5a` e le 91 sorgenti before-fix da Git `575369b1` più `pending-input.patch`, verificando tutti gli SHA del manifest. Il processo principale comunica inoltre `git diff --check` finale PASS. SHA-256 di `fix-ab/results.json`: `b7e364d9155e20804704d76769a777292173081c644d48512626b0d1a630f220`.

## Conclusione

La differenza di dispatch residente è dimostrata dal controllo negativo; la correzione ripristina la parità con main nei casi locali, mantenendo la reference SSD. Il confronto separato degli shader HC non rileva differenze sui 71,793 valori provati sul M1 Max. Rimane da confermare il risultato sul Q2 completo residente del tester e sul suo M4 Max: la fixture Q4 locale, con PLE zero e senza predictor, non verifica quelle condizioni né MTP. Non è stata misurata una variazione di prestazioni.

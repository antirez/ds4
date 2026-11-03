# Decode radicale: verificare più token per caricamento degli esperti

Audit 2026-09-30, repository corrente `b96a12be` + modifiche locali preesistenti. Nessuna compilazione, GPU o inferenza avviata da questo audit. Sono stati letti solo 11,025,109 byte di intestazione del GGUF locale, non i pesi; `gguf-mtp-header.json` registra dimensioni e metadati.

## La prova già attivabile

`--mtp-timing` abilita il predictor embedded del Qwen locale e le verifiche a 2/3 righe. Il file da 177,280,286,720 byte contiene `qwen4exp.nextn_predict_layers=1`, 49 blocchi fisici e 32 tensori `blk.48.*`, compresi `nextn.eh_proj`, normalizzazioni e MoE. Non occorre scaricare un drafter. `--mtp-model` esterno è invece rifiutato con SSD streaming (`ds4.c:71708`).

Il percorso è reale: `ds4_engine_mtp_draft_tokens` (`ds4.c:62641`) restituisce 2 per Qwen embedded; `ds4_session_qwen4_spec_cycle` (`ds4.c:74707`) esegue `[first_token,draft,draft2]` con T=2 o 3, poi confronta il draft con l'argmax del target (`74815`). A temperatura zero conserva la continuazione greedy, salvo l'override diagnostico non esatto `DS4_QWEN4_SPEC_FORCE_ACCEPT`, che deve essere assente. `--mtp-margin` non sostituisce questa verifica nel percorso Qwen. `--mtp-draft` non imposta la profondità Qwen: il controllo reale è `DS4_QWEN4_MTP_DEPTH=2|3|auto`.

La deduplicazione delle letture SSD fra righe esiste già: `ds4_metal.m:51554` legge tutti i `n_tokens*n_slots` id del batch, ne calcola le frequenze/union e passa l'intero batch a `ds4_gpu_stream_expert_cache_prepare_selected_batch` (`51591`). Quindi due token che scelgono lo stesso esperto condividono il payload caricato. Questo non garantisce meno byte per token emesso: i draft rifiutati possono aumentare l'I/O.

### Vecchia misura, da riconfermare

Fonte: `misc/qwen-gain10-v2-2026-09-27/{run.py,mtp-screen.json,mtp-screen-results.json,mtp-screen-rome.log,mtp-screen-rome-control.log}`.

| | plain | embedded MTP |
|---|---:|---:|
| Prefill t/s | 8.40 | 8.33 |
| Decode t/s | 5.52 | 6.34 |
| Totale wall s | 23.225 | 20.874 |
| Expert cache | 1908 / 4.82 GiB | 1908 / 4.82 GiB |
| Miss | 20,470 | 22,470 |
| Letture miss | 51.73 GiB | 56.79 GiB |
| `pread_ms` aggregati | 11,762.977 | 11,581.953 |
| Picco tensori runtime | 6828.47 MiB | 7056.17 MiB |

Entrambi: ctx32768, chunk8192, n100, temp0, nothink, Roma31 token dal file `misc/qwen-cost-analysis-2026-09-27/prompt-rome.txt`, Q4 locale, SSD. Static6.32GiB, graph reserve9.22, staging2.59, totale previsto22.96. Il job aggiungeva soltanto `--mtp-timing` al candidato; nessun override di profondità o trace era registrato. Il runner usava per default `/tmp/ds4-gain10-v2-base`; non salva la sua hash né una fotografia completa dell'ambiente, quindi non si può attribuire retroattivamente ogni flag ambientale.

Output identico SHA256 `67ce82561ce711888230dd6cf7bfd0896a77f3e5586c70a93b6f38589f23f133`. 54 cicli di verifica e 42 primi draft accettati (77.8%). Non sono registrati secondi draft accettati né durata separata di draft/verify/restore; non sono ricostruibili dal solo contatore. La coppia è in ordine MTP→plain, non ABBA. +14.86% è screening, non conferma. MTP ha letto +9.8% byte: non attribuire il guadagno a una riduzione osservata dei byte.

Confondente importante: senza override il plain usa `generate_qwen4_metal_argmax`, mentre MTP forza le sessioni (`ds4_cli.c:1241`). La nuova ABBA deve impostare `DS4_CLI_FORCE_SESSION=1` in entrambe le braccia. Poi confrontare il vincitore anche col normale percorso CLI dell'utente. `DS4_QWEN4_SPEC_TRACE=1` registra accept/accept2/reject2: utile per un pilot diagnostico, non necessario nei run temporizzati. `--mtp-timing` fornisce soltanto i due contatori riportati sopra, non veri tempi delle fasi Qwen.

Per una conferma iniziale: ctx32768/chunk8192/n128/temp0/nothink e cache identica, ordine plain/MTP/MTP/plain ripetuto su Roma e un prompt di codice. Nessuna timeline GPU durante i run temporizzati. Contare token realmente emessi/forward: i due percorsi possono differire nella valutazione dell'ultimo token; n128 limita ma non elimina il problema. Confrontare stdout, t/s, byte SSD, numero di cicli e picco memoria. Nessuna percentuale va sommata a quella di altri bundle.

## Nuova idea 1: prompt lookup senza rete drafter

Non ho trovato un predictor di continuazioni nel codice corrente (`prompt_lookup`, `ngram.*draft`, `draft.*ngram`, `lookup.*speculat` assenti nei sorgenti). I giganteschi n-gram BF16 del Qwen sono feature del modello, non una cache di proposte speculative: non confonderli.

Idea: trovare il suffisso degli ultimi 3–8 token (includendo il prossimo token target già scelto) nel transcript precedente; prendere i successivi 1–2 token di quell'occorrenza come proposta, poi verificare con lo stesso target a 2/3 righe. Quando manca una corrispondenza, fare il decode ordinario. Può essere particolarmente utile per copia/riformattazione di documenti, codice ripetitivo e JSON; non si presume un vantaggio sulla prosa Roma.

È esatto come continuazione greedy se il target verifica ogni proposta. Si elimina il costo della rete MTP, dell'head draft e della sua preparazione sul prompt. Si sfrutta la medesima union degli esperti del batch. Le proposte sbagliate costano I/O e compute ma non alterano il testo. Non usare force-accept o filtri sul margine.

Si può riusare l'infrastruttura esistente, ma non è un semplice flag già presente:

1. Separare la scelta dei draft dal verificatore di `ds4_session_qwen4_spec_cycle`.
2. Allocare gli snapshot di verifica anche senza `mtp_R` (`qwen4_graph_alloc`, `58062`): oggi sono legati al bool MTP.
3. Rendere il verificatore indipendente dai controlli `g->mtp_R`/`mtp_tail_valid` e dalle chiamate successive al predictor. Per un primo prototipo usare solo un draft/T2, che necessita un solo snapshot.
4. Sul reject riusare `qwen4_graph_state_swap`: ripristina i 36 stati GDN, history conv, PLE/history, posizioni mRoPE e hidden tail. Le KV future restano scritte ma i successivi passi le mascherano/riscrivono attraverso `g->pos`; non basta solo troncare il transcript. T3 richiede `snap2` e geometria exact, già presenti.
5. Verificare accettazione, rifiuto e stop-token con confronto logits/greedy al plain, inclusi confini dell'indicizzazione sparse a blocchi4 e rewind.

Non estenderei inizialmente a 8 o 16 token: l'infrastruttura corrente ha snapshot esatti soltanto dopo riga0 e riga1, e molti kernel/array temporanei MTP assumono massimo3. Una verifica lunga con GDN richiederebbe snapshot per prefix o un replay, che può consumare il guadagno.

## Nuova idea 2: scegliere la profondità con il costo SSD misurato

La policy esistente (`ds4.c:74673`) scala a T3 solo dopo 8/8 primi draft accettati e torna a T2 con segnali di rifiuto. Considera l'acceptance, non il tempo del ciclo o i byte dell'union esperti. Il commento stima che circa0.6 acceptance della seconda proposta non ripaghi il ciclo largo: potrebbe non valere per ogni cache/SSD/context.

Un controllore diagnostico può scegliere plain/T2/T3 in base a token **committati per tempo complessivo** di una finestra, separando prefill e warmup. Per T2, con p probabilità d'accettazione, t1 costo plain, t2 costo totale draft+verify+restore, il guadagno teorico è `(1+p)*t1/t2`. Per superare+10% serve `t2/t1 < (1+p)/1.10`. Prompt lookup riduce la componente draft di t2; un SSD lento può favorire batch con union esperti molto sovrapposta. Nessun valore p o t2 è presunto.

Questo controllore non cambia target/accettazione, soltanto quanto lavoro tentare. Evitare flapping, warmup e cache pollution da sonde frequenti: confronti in finestre brevi non sono automaticamente causali. Prima misurare i costi fissi e acceptance con i flag già attivabili, poi introdurre una policy.

## Riferimento esterno

[SpecMoEOff, arXiv2508.21706](https://arxiv.org/abs/2508.21706): propone di allargare il lavoro per esperto con speculative decoding per nascondere offload, con ottimizzazione della configurazione hardware/workload. È supporto del principio, non una previsione di speedup su M1 Max, Qwen locale o SSD. Il lavoro usa anche una verifica attention CPU dedicata; non si trasferisce automaticamente alla nostra combinazione GDN/Metal.

# SSD Qwen Q4: due cambiamenti di architettura esatti

Audit 30 settembre 2026, working tree b96a12be più modifiche già presenti. Nessuna modifica alla produzione, nessun benchmark GPU/modello e nessuna compilazione in questa lane. Le proposte sono da misurare: non sono risultati di throughput.

## Dati che motivano il lavoro

Ogni esperto Q4 corrente contiene gate 921.600 byte, up 921.600 byte, down MXFP4 870.400 byte: totale **2.713.600 byte = 2,587890625 MiB**. I run di 8631 token + 100 generati misurano 65.316 miss, mentre il pilot n=1 aveva 33.179 miss. La differenza è 32.137 miss in 99 forward decode: **324,62 esperti/forward, 840,07 MiB/forward**. Questa sottrazione è valida per i log con medesima cache1908 e medesimo testo; i byte richiesti a pread non equivalgono a traffico fisico SSD. Il 98,4% hit rate aggregato è dominato dai milioni di scelte del prefill e non descrive il decode.

La baseline più recente impiega ~55,16s per 8631 token e ~281ms/token generato. Per +5% throughput servono ~2,63s prefill o ~13,4ms decode; +10% richiede ~5,01s o ~25,5ms. Non sommare tempi pread e sync: si sovrappongono.

## 1. Prefetch completo della layer successiva durante il prefill

Questo è **vero lookahead tra layer**. La versione corrente aspetta il router della layer corrente, prepara la cache selected, e sovrappone soltanto esperti residenti/missing gate-up/down della stessa layer. `qwen4_stream_read_layer` (`ds4_metal.m:51156`) sa leggere una layer intera, ma è sincrona e non orchestra il lookahead. Le vecchie prove row-gate-up-overlap e router/shared-event non implementano questo schema.

Budget per finestra: gate450 MiB + up450 MiB + down425 MiB = **1,2939453125 GiB**. Due finestre costano **2,587890625 GiB**, già comprese nella riserva di `qwen4_streaming_staging_bytes` (`ds4.c:70875`). Oggi questa riserva non implica che le finestre esistano fisicamente: il prototipo aumenterebbe il footprint quando il percorso cache selected non le alloca.

Il routing reale T8192 catturato in `misc/qwen-gain5-2026-09-27/root/routes-xl.bin` usa **20.599 esperti distinti su24.576**: minimo289, massimo499, media429,146 per layer. Per le48 layer:

| Lettura a cache vuota | GiB |
|---|---:|
| Soli esperti effettivamente selezionati |52,0586|
| Ogni esperto di ogni layer |62,1094|
| Extra del prefetch completo |**10,0508 (+19,31%)**|

Con cache hit preesistenti l'extra rispetto all'attuale cresce: il full prefetch rilegge anche i residenti. Quindi non si può presentare il lookahead come lettura gratis, né assumere che tutte le512 scelte siano usate. Sui tail439 o prompt corti la densità è inferiore: primo prototipo solo chunk realmente8192, fallback corrente altrove. Leggere anche layer non eseguite/ultimo predictor sarebbe sbagliato.

Architettura minima proposta:

1. Al confine di chunk, creare due finestre stabili A/B di tre buffer ciascuna, con `(model identity, layer, byte ranges, generation)` e stato EMPTY/READING/READY/FAILED. La read worker possiede solo fd, offset e destinazioni; non tocca cache, globals GPU, ARC della render lane o address table.
2. Avviare layer0 appena noto il chunk. Quando layer0 è pronta, avviare layer1 sull'altra finestra mentre la GPU esegue layer0. Il lancio deve essere sufficientemente presto: prima del prossimo router, non dentro il wrapper che già lo aspetta.
3. Dopo il router nativo invariato, chiamare l'attuale preparazione selected con un provider di payload: un miss prenota esattamente lo stesso slot/cache victim dell'originale, ma le tre load task copiano i byte dalla finestra invece di fare pread. Frequenze, ordine unique, politica eviction e install rimangono uguali; gli hit non vengono copiati. Nessuna seconda lettura SSD degli stessi dati per popolare la cache.
4. Quando tutte le copie selected della layer sono terminate, la finestra può essere riciclata per layer+2: i kernel usano le normali cache entry, non indirizzi temporanei nella finestra. Questo evita dipendenze GPU sulla finestra e lascia la cache calda al decode.
5. Perdere eventualmente il callback gate-up-ready su READY full-payload non cambia i calcoli. Il percorso delle copie può preservare i callback, ma non deve lanciare down prima che il suo payload sia completo. Non riutilizzare il pool pread globale come se fosse rientrante: i suoi task/generation sono globali. Usare un job dedicato o serializzare il pool esplicitamente.

Varianti successive solo se il primo screen è utile: bitmap dei cache hit fotografata in sicurezza per non rileggerli; prefetch di intervalli mancanti coalescenti; selettore per layer basato su densità osservata in precedenti chunk. Le predizioni influenzano solo le letture, mai il router. Non usare la densità della stessa cattura come conoscenza del futuro in un benchmark.

Rischi/lifetime da verificare: fine chunk e ultimo layer, cancellazione, errore corto/EINTR, fd/modello chiuso mentre read è attiva, errore dopo copie parziali, retained/unretained command buffers, worker che non può pubblicare READY prima del payload completo, fallback su allocazione fallita, nessuna eviction di reader inflight. Il fallback in errore deve prima joinare il job e scartare dati incompleti; evitare duplicare il lavoro che è già stato inviato alla GPU.

Ipotesi di guadagno: il prefetch vince se la lettura dei1,294GiB più copie entra sotto il compute precedente e riduce **I/O esposto**, abbastanza da compensare +19,31% byte minimi. A titolo di bilancio, con5GiB/s servono259ms/window e circa2s di I/O extra per il chunk completo. Non è una misura della banda disponibile durante GPU; CPU memcpy e SSD competono con la GPU per la RAM unificata. Il confronto deve riportare tempo host atteso alla finestra, byte prefetch utili/inutili, byte copiati, picco fisico, prefill completo, decode successivo e output bitwise. Questa è la proposta prioritaria per il prefill perché può nascondere un costo molto più grande dei piccoli kernel, ma può anche perdere se le finestre contendono la memoria.

## 2. Workspace elastico tra prefill e decode

**Diverso dalle prove archiviate** in `qwen-gain10-v2-2026-09-27`: quelle conservavano cap8192 e usavano una stima più precisa + alias PLE, da1908 a3009 slot. Questa proposta conserva il prefill8192 e la sua aritmetica, ma al termine riduce i transienti a1/3 righe e trasferisce quel budget alla cache. Non è cambiare `--prefill-chunk1024`, che può cambiare i confini delle somme nei prompt lunghi.

`qwen4_graph_alloc` (`ds4.c:57973`) alloca tutti i transienti con `T=cap_tokens`; i40 campi non vengono ridotti a decode. Gli scratch opzionali backend sono già rilasciati nel wrapper, ma non questi array strutturali.

Per cap8192→3, forme correnti:

- transienti GPU180.604 float words/riga: **5,5096GiB effettivi** liberabili;
- `host_row`:319,883MiB liberabili; `host_pos3`:~0,125MiB;
- la stima conservativa corrente scende di **7,7239GiB** senza cambiare la sua formula;
- in decode ordinario senza MTP, staging massimo due set di10 esperti anziché due layer512:2,5879→0,05054GiB. Questo recupero è solo di riserva quando lo staging grande non è materializzato.

**I5,51GiB di buffer GPU da soli non portano a5968 esperti.** Occorre distinguere allocazioni rilasciate da riserve del planner. I5,5096GiB sono byte allocati dei buffer riducibili, non una misura garantita di diminuzione del physical footprint: pagine non toccate, riferimenti ARC/views e reader devono essere controllati. Sommando host_row/host_pos3, il rilascio nominale diventa5,8221GiB. Il calo7,7239GiB del planner contiene quindi altri1,9018GiB di riserva, inclusa la parte legacy conservativa e scratch opzionale; non sono ulteriori transienti GPU appena liberati.

La cache usa payload2.713.600byte, ma `ds4_gpu_stream_expert_alloc_slab_slot` arrotonda al page size. Su questa macchina16KiB, uno slot occupa**2.719.744byte**,6144byte in più. Il budget attuale1908slot omette così11,180MiB di padding. La tabella riporta un cap prudente che corregge anche quel padding preesistente: `floor((1908*payload + credito)/slot)`. I cap nominali nella colonna accanto usano invece payload non allineati come il planner corrente.

| Credito riconosciuto rispetto al prefill | Natura | Cap con pagine | Cap nominale |
|---|---|---:|---:|
|5,5096GiB dei soli transienti GPU|Allocazioni da rilasciare e verificare|**4078**|4088|
|5,8221GiB GPU + staging host per righe|Allocazioni da rilasciare e verificare|**4202**|4211|
|7,7239GiB del planner cap8192→3|Allocazioni più riserva conservativa|4953|4964|
|10,2613GiB: planner più riduzione staging2,5373GiB|**Proiezione di admission**, include staging non necessariamente allocato|**5954**|5968|

Per il primo screen scegliere4078 o un limite più basso dopo misure reali; oltre quel cap serve giustificare separatamente il credito host e le riserve. Il valore5968 era dunque un tetto aritmetico nominale dell'intero ridisegno del planner, non un risultato né un cap ottenibile dai soli5,51GiB. Le voci static6,32GiB, KV/stato persistente e headroom runtime2GiB rimangono. Vanno inoltre rispettati il tetto esplicito dell'utente, altre sessioni e l'eventuale capmlock.

Implementazione necessaria:

1. Separare formalmente capacità di scheduling prefill e capacità viva scratch; il primo valore resta8192. Separare scratch dal persistent state nel grafo.
2. Al passaggio dopo l'ultimo prefill, sincronizzare tutti i command buffer e read job. Conservare KV/K/V pooled, GDN state/history, PLE history, pos3, logits/MTP tail e snapshot; solo i transienti si riducono.
3. Allocare prima scratch piccolo completo, poi scambiare i campi e liberare il vecchio insieme. Non usare `qwen4_graph_free` che azzera anche stato persistente. R/HC_u possono essere stati scambiati dal decode fuso: ownership va deduplicata per oggetto e testata. Il passaggio deve avvenire quando il vecchio residuo non è più letto; MTP tail persistente rimane valido.
4. Aumentare il budget cache solo dopo il rilascio e sul planner engine globale; cache fault/alloc fallita mantiene il vecchio limite funzionante. Distinguere limite automatico da un limite esplicito chiesto dall’utente: il secondo resta un tetto e non va superato. Per primo prototipo: singola sessione SSD, greedy normale, MTP off; niente modifica CUDA/residente/multisessione.
5. Prima del turno successivo o prefill nuovo, sincronizzare, ridurre cache al limite prefill con regole inflight correnti e liberare i buffer realmente posseduti, ricreare scratch8192, poi iniziare il chunk. Un mero cambio del limite cache non libera necessariamente gli slab: vanno rivisti slot/free-list e slab partial residency. Evitare picco scratch grande+cache grande.

### Invarianti dello slab allocator da preservare

- Il setter pubblico `ds4_gpu_set_streaming_expert_cache_budget` (`ds4_metal.m:4636`) **azzera tutta la cache e le statistiche**. Non usarlo come semplice grow nel mezzo della richiesta: serve un'operazione separata che conservi entry, route hotness, address table, counters e reader.
- La cache è a singola size class:2.713.600byte payload diventano2.719.744byte/slot; gate/up/down condividono lo stessoMTLBuffer agli offset base/base+921600/base+1843200. Il grow non deve cambiare questa classe o riallineare slot già pubblicati.
- Gli slab sono grandi fino a4GiB nominali, con start_slot/count e un indice locale di slot già usati. `mlock` è per slot. Le entry e le free-list contengono ID assoluti: eliminare/compattare slab richiede aggiornarli insieme agli indirizzi GPU soltanto a GPU ferma.
- L'eviction ordinaria `clear_entry_internal(...recycle_slab_slot=1...)` rimette lo slot nella free-list, **senza munlock dello slot e senza rilasciare lo slab**. Diminuire entry_count o budget non recupera quindi i byte fisici necessari al prefill successivo. Uno shrink deve svuotare/rilasciare slab interi oppure migrare entry superstiti verso una regione mantenuta e ricostruire free-list/tabelle; se si usa unlock di slot vuoti, non presentarlo come rilascio certo diMTLBuffer/residency.
- `clear_all` rilascia gli slab ma azzera anche sequenze inflight. Richiede una barriera reale prima della chiamata; assegnare done_seq=cb_seq non dimostra da solo che i reader GPU siano terminati. Per un primo prototipo il reset completo al ritorno al prefill è più semplice, ma il costo di cache fredda fa parte della misura del turno successivo.
- Gli allocation failure riducono la dimensione degli slab; gli mlock failure possono imporre un cap persistente. Grow e rollback devono lasciar funzionare questi fallback, senza descrivere il tetto calcolato come disponibilità assicurata.

La precedente prova `--prefill-chunk1024` aveva mostrato +30% decode Roma con4584 slot; è evidenza che capacità cache può contare, **non una previsione** per questa diversa transizione e prompt8631. Misurare warm-up della cache accresciuta, 100/1000 token, turno successivo e costo di ripristino. Il cambiamento merita una prova ma è più ampio del lookahead ed espone molte ownership da coprire.

## Idea economica scartata: sidecar compresso lossless

`sample_lossless.py` legge solo21.708.800 byte di payload da8 esperti (ID0/511 alle layer0/9/23/47), oltre al parser header; non legge/interpreta il modello intero. Testa libcompression macOS LZ4/LZFSE e zlib1, raw o shuffle byte per tile1024 blocchi. Tutte le decompressioni e inverse shuffle coincidono byte per byte.

| Variante | Riduzione payload | Decode CPU ms/esperto |
|---|---:|---:|
|Raw LZ4|−0,013%|0,046|
|Raw LZFSE|2,526%|3,900|
|Raw zlib1|3,097%|6,437|
|Shuffle LZ4|1,070%|0,096|
|Shuffle LZFSE|4,332%|3,876|
|Shuffle zlib1|4,902%|6,400|

Tempi mediani di7 decode in RAM; per shuffle escludono l'inversa, quindi sono ottimistici. LZ4 incomprimibile può ricopiare direttamente, da cui la grande banda apparente. I campioni non dimostrano una percentuale per l'intero modello, ma non sostengono una compressione generica utile: risparmiare4% dei~0,5ms di trasferimento per esperto a5GiB/s vale~0,02ms contro~3,9ms di decode CPU. Non costruire ora un sidecar completo da decine diGiB. Output machine-readable `lossless-results.json`.

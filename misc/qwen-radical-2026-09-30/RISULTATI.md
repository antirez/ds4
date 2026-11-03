# Idee radicali per Qwen Q4 SSD — 30 settembre 2026

Dispositivo: Apple M1 Max, 32 GiB. Repository `qwen-kernel-opt`, base `b96a12be`
con modifiche locali preesistenti. Questa ricerca non modifica i file di produzione.
Le percentuali sotto sono guadagni di throughput; non vanno sommate fra loro.

## Risultato nuovo: distribuire meglio il lavoro MoE

Il kernel corrente assegna un rettangolo di threadgroup agli esperti e lascia a
ciascun gruppo un ciclo di lavoro di lunghezza variabile. Con routing sbilanciato
alcuni gruppi escono subito, mentre altri elaborano fino a 32 tile in sequenza.
Il prototipo costruisce una lista dei soli tile utili e ordina gli esperti per
numero di token decrescente. Ogni gruppo esegue un tile indipendente. Cambiano
assegnazione e ordine del lavoro, non pesi, aritmetica o ordine delle somme.

| Routing reale di 8192 token | Baseline mediana | Candidato mediano | Throughput MoE |
|---|---:|---:|---:|
| Layer 9 | 271,556 ms | 250,269 ms | **+8,51%** |
| Layer 47 | 248,010 ms | 225,318 ms | **+10,07%** |

Quattro blocchi ABBA/BAAB per confronto, tutti positivi, nessun campione escluso.
I tempi comprendono conversione, mid, down e costruzione CPU della worklist.
Servono circa 11 KiB di descrittori; il fixture riserva 512 KiB in entrambe le
varianti, da ridurre al fabbisogno effettivo nell'integrazione. Pesi sintetici e
routing reale registrato: questi **non sono benchmark del modello completo**.
Confronti numerici bitwise, guardie e input immutabili superati. Il modo safe è
verificato sulla lista compatta in ordine ID; heavy-first cambia soltanto l'ordine
dei gruppi indipendenti e ha superato le proprie verifiche fast.

Il precedente profilo attribuisce circa 12,66 s grossi a questa famiglia su
circa 55 s di prefill. Proiettare il guadagno sull'intera fase suggerisce circa
**2%**, non 8–10%. La combinazione con il precedente bundle da +3,35% potrebbe
raggiungere il 5%, ma richiede un confronto completo: non è ancora dimostrato.
Questo scenario assume che il beneficio sia simile sugli altri layer e non
venga nascosto dall'overlap SSD. Il microbenchmark usa un unico pass con pesi
residenti e indirizzamento diretto; la produzione divide residenti e mancanti.
I descrittori dei due pass devono essere immutabili fino alla completion GPU:
riscrivere una stessa coda condivisa mentre il primo pass è in volo sarebbe una race.

Dettagli e limiti: [moe/README.md](moe/README.md).
Tutti i campioni: [moe/results-summary.json](moe/results-summary.json).

## Cambiamenti con un margine potenzialmente maggiore

### 1. Riutilizzare per la cache la memoria del prefill dopo il prompt

Oggi il decode conserva transienti dimensionati per 8192 token pur elaborando
una o poche righe. Ridurre effettivamente questi buffer al cambio di fase può
liberare circa **5,51 GiB GPU**; i pesi degli esperti possono occupare lo spazio
liberato e ridurre le letture successive. La capacità di prefill resta 8192:
prima del prompt successivo bisogna ridurre la cache e ricreare lo scratch.
Contando il padding degli slot, il solo credito dei buffer GPU potrebbe portare
la cache da **1908 a circa 4078 esperti**. Non va confuso con un tetto maggiore
ottenibile anche rivedendo le riserve conservative del planner. Oggi il setter
del budget svuota la cache, mentre l'eviction non rilascia i suoi slab: entrambi
i comportamenti vanno adattati per rendere reale questa transizione.

È diverso dalle precedenti ottimizzazioni della sola stima di memoria e dagli
alias PLE. Richiede un vero cambio di capacità dei buffer, preservando KV,
stati GDN, cronologia PLE e stato MTP. Nessuna percentuale di velocità è misurata
per questa proposta. Il beneficio dipende dal riuso degli esperti e dal tempo
necessario a scaldare la cache più grande.

### 2. Leggere la layer successiva mentre la GPU elabora quella corrente

L'overlap attuale riguarda principalmente parti della stessa layer. Un pipeline
fra layer può usare due finestre da **1,294 GiB**: una alimenta la cache degli
esperti correnti, l'altra viene riempita dall'SSD per la layer successiva.

Nel routing reale del chunk lungo sono usati mediamente 429 esperti su 512.
Prefetch di ogni esperto richiederebbe almeno **19,31% di byte in più** rispetto
ai soli selezionati a cache vuota. È quindi utile solo se nasconde abbastanza
attesa da compensare le letture aggiuntive e le copie in RAM. Per i prompt corti
il percorso corrente resta una scelta più sensata per il primo prototipo.

Le finestre sono già considerate nella riserva del planner, ma spesso non sono
materializzate: non si tratta di memoria fisica gratis. La cache deve continuare
a ricevere gli stessi esperti negli stessi slot, altrimenti il decode successivo
parte con una cache diversa e il confronto sarebbe incompleto.

### 3. Proporre token dal testo già disponibile, poi verificarli col modello

Per copia, riformattazione, codice ripetitivo e JSON, un suffisso del transcript
può suggerire i successivi uno o due token. Qwen li verifica insieme e accetta
soltanto quelli corretti. Si potrebbe così evitare il costo di una rete drafter
dedicata e riusare gli esperti del batch.

Il verificatore e gli snapshot esistono per MTP, ma vanno separati dalla rete
predictor. Un errore della proposta deve ripristinare anche GDN e PLE, non solo
la posizione KV. Non è una funzionalità già presente e non c'è un guadagno
misurato. È particolarmente dipendente dal tipo di richiesta.

Il principio di verificare più token per caricamento MoE è sostenuto anche da
[SpecMoEOff](https://arxiv.org/abs/2508.21706); i risultati di quel lavoro non
sono una previsione per questo dispositivo. Una variante futura del prefetch
può predire soltanto quali pesi trasferire, lasciando invariato il router:
[SpecPrefetch](https://arxiv.org/abs/2607.24787) studia questa separazione.

Specifiche, budget e punti di intervento:
[io/ARCHITETTURA.md](io/ARCHITETTURA.md),
[decode/IDEAS.md](decode/IDEAS.md).

## Verifica della generazione speculativa già disponibile

Il GGUF locale contiene già il predictor MTP. Il confronto locale usa lo stesso
binario, lo stesso percorso CLI sessione, cache da 1908 esperti, contesto 32768,
chunk 8192, temperatura zero e 128 token. Ordine plain/MTP/MTP/plain su due
prompt. Il runner registra hash delle sorgenti e dell'output, byte richiesti a
pread, accettazione e picco memoria.

| Prompt | Plain decode t/s | MTP decode t/s | Guadagno decode | Primi draft accettati |
|---|---:|---:|---:|---:|
| Roma, 31 token | 4,910 | 5,864 | **+19,44%** | 81,2% |
| Codice Python | 4,969 | 6,174 | **+24,24%** | 96,0% |

Medie armoniche, due run per variante e prompt. Tutti gli otto run completati,
output byte per byte identici per ciascun prompt. Campioni decode Roma:
plain4,92/4,90; MTP5,80/5,93. Codice: plain5,21/4,75; MTP6,25/6,10. Quindi
il beneficio è presente anche confrontando il candidato peggiore con il plain
migliore, ma il numero di ripetizioni resta piccolo. Non è un benchmark di
qualità generale o una garanzia per ogni prompt, contesto e temperatura.

Il denominatore include il tempo di draft, verifica, rollback e stampa. Il
numeratore conta soltanto token realmente emessi. MTP non salta il forward
finale che il plain evita, quindi quel confine non gonfia il suo vantaggio.
La preparazione storica del predictor è completata nel prefill.

Le letture richieste a pread **aumentano**: Roma61,65→67,85GiB, codice79,49→87,29GiB,
circa+10%; i byte non equivalgono a traffico fisico SSD. Il vantaggio non è
una riduzione misurata delle letture. La memoria di picco aumenta di circa
300MiB. Sul prefill Roma il primo plain è più lento degli altri run: non
attribuire a MTP il +12,27% che il riepilogo automatico calcola su quei due
campioni. Sul prefill codice la differenza è appena+0,61%.

Il wall time medio dell'intero processo scende da32,50 a27,55s per Roma e
da33,82 a28,63s per codice: riduzioni del15,24% e15,34%, rispettivamente.

Questa prova isola MTP usando `DS4_CLI_FORCE_SESSION=1` in entrambi i casi.
Raw completi: [model/mtp-abba/results.json](model/mtp-abba/results.json).

### Conferma rispetto al normale comando CLI

Ulteriori quattro run ABBA Roma128, senza forzare il percorso sessione:

| | CLI ordinario | CLI con `--mtp-timing` |
|---|---:|---:|
| Decode, media armonica |5,485 t/s|6,175 t/s|
| Singoli run decode |5,45 / 5,52|6,01 / 6,35|
| Prefill, media armonica |7,640 t/s|7,623 t/s|

**+12,59% decode rispetto al CLI ordinario**, prefill sostanzialmente invariato
(−0,21%). Output identico nei quattro run e alla precedente serie Roma;
capacità cache, miss e byte uguali ai rispettivi gruppi della prova sessione.
Questo è il numero più pertinente al normale comando dell'utente, non +19,44%.
Il wall time medio scende da28,87 a26,34s, una riduzione di circa8,8%.

Non è codice nuovo: si attiva con `--mtp-timing`, già disponibile per il predictor
embedded. In questa prova si usa `--temp 0`; non estendere questi risultati a
tutte le temperature o a contesti lunghi senza misurarli. Tutti i12 run delle
due serie hanno completato e conservato lo stesso output per prompt.
La generazione speculativa è una via concreta oltre l'obiettivo del10% in questo
caso, mentre per le nuove architetture di cache/prefetch non c'è ancora un
risultato completo. Raw: [model/mtp-user-cli-abba/results.json](model/mtp-user-cli-abba/results.json).

Runner: [model/run-mtp.py](model/run-mtp.py).

## Idea scartata con una prova: comprimere lossless gli esperti

Campionati 20,7 MiB di pesi reali da otto esperti, con ricostruzione esatta.
LZFSE con byte shuffle risparmia **4,33%**, ma richiede circa **3,88 ms/esperto**
per la decompressione; zlib arriva al **4,90%**, con **6,40 ms/esperto**.
Il costo eccede largamente il trasferimento risparmiato. Non conviene costruire
un sidecar completo con queste tecniche. La prova riguarda questi campioni e
queste tecniche, non dimostra l'impossibilità di ogni formato lossless.

Raw: [io/lossless-results.json](io/lossless-results.json).

## Ordine di lavoro proposto

1. Integrare in un binario sperimentale lo scheduling MoE con il precedente
   bundle, poi misurare l'intero prefill e il decode successivo.
2. Prototipare il workspace elastico per attaccare il costo SSD del decode.
3. Prototipare il prefetch fra layer per il prefill lungo.
4. Separare draft e verifica per il prompt lookup e una scelta della profondità
   MTP basata sui token realmente emessi per secondo.

Non dichiarare raggiunto un obiettivo del 5% o 10% sull'intera fase sulla base
dei soli microbenchmark o dei budget di memoria.

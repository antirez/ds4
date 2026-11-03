# Come faremo girare veloce DeepSeek V4 sul tuo PC (Spiegato Semplice)

Questo documento spiega in modo chiaro e senza gergo tecnico **qual è il problema**, **come funziona il modello di intelligenza artificiale** che stiamo scaricando e **cosa faremo passo dopo passo** per renderlo fluido e veloce sul tuo computer, senza mai rischiare spegnimenti o blocchi.

---

## 1. Il punto di partenza: gli "ingredienti" del tuo PC

Immagina il tuo computer come la **cucina di un grande ristorante** dove lo Chef (il chip della scheda video **RTX 3090**) deve preparare le parole della risposta una dopo l'altra, il più velocemente possibile.

Per cucinare, lo Chef ha a disposizione **tre spazi** dove tenere il ricettario (il modello **DeepSeek V4**, che è gigantesco e pesa **81 Gigabyte**):

1. **Il Bancone da Lavoro dello Chef (la memoria della scheda video, 24 GB)**:
   * È **velocissimo** (lo Chef allunga la mano e legge all'istante).
   * Però è **piccolo**: tiene solo **24 GB**, mentre il ricettario intero pesa **81 GB**. Quindi tutto il modello lì sopra non ci sta.
2. **La Dispensa in Cucina (la memoria RAM del computer, 64 GB)**:
   * È **molto grande** (64 GB) ed è collegata al bancone dello Chef da un **passavivande super-veloce** (il collegamento PCIe 4.0, che sposta 27 Gigabyte al secondo).
3. **Il Magazzino in Cantina (il disco SSD da 1 Terabyte)**:
   * È **enorme** (ci sta tutto), ma è **molto più lento**: ogni volta che un cameriere deve scendere in cantina a prendere una ricetta, passa un sacco di tempo (è **8 volte più lento** rispetto a prenderla dalla Dispensa in cucina).

---

## 2. Il segreto di DeepSeek V4: non legge mai tutto il ricettario!

Se per scrivere **ogni singola parola** lo Chef dovesse leggere tutte le 81 GB di ricette, sarebbe lentissimo.
Ma **DeepSeek V4** è costruito in modo furbo (si chiama architettura *"a Esperti"*):

* C'è una **parte fissa** del cervello (circa **13 GB**) che serve sempre per ogni parola.
* Poi ci sono **10.240 piccoli "Specialisti"** (circa **68 GB** in totale): uno è esperto di grammatica italiana, uno di codice Python, uno di matematica, uno di storia, ecc.
* **Il trucco**: per scrivere una singola parola, **non servono tutti i 10.240 specialisti, ma ne vengono chiamati solo 240** (meno del 3% del totale, cioè circa **1,6 GB** di dati)!

Quindi:
* La **parte fissa (13 GB)** la mettiamo stabilmente sul **Bancone dello Chef (24 GB)** e non si muove mai più da lì.
* Per ogni parola, dobbiamo solo far arrivare sul bancone quei pochi **Specialisti** che servono in quel momento!

---

## 3. Cosa fa di sbagliato il programma originale sul tuo PC?

Il programma `ds4` scritto da Antirez è pensato per due tipi estremi di computer:
* Computer costosissimi che hanno un Bancone enorme da 128 GB (dove ci sta tutto dentro),
* Oppure computer con pochissima memoria che sono costretti a leggere tutto dal Magazzino in Cantina (il disco SSD).

Sul tuo PC — che invece ha una **combinazione perfetta (24 GB di Bancone + 64 GB di Dispensa = 88 GB totali)** — il programma originale commette **4 sprechi enormi**:

### Spreco 1: Butta via le ricette dalla Dispensa (RAM) e torna sempre in Cantina (SSD)!
Nel codice originale, quando si attiva la modalità per schede video da 24 GB, il programma legge uno Specialista dal disco SSD, lo passa allo Chef e **un istante dopo dà ordine a Linux di cancellarlo dalla Dispensa (RAM)**!
Così, alla parola successiva, invece di prendere gli Specialisti dalla tua **Dispensa da 64 GB** (velocissima), torna a leggerli dal **Magazzino in Cantina (SSD)**, andando **8 volte più lento**!

### Spreco 2: Lascia vuoto un terzo del Bancone dello Chef!
Per eccesso di prudenza, il programma originale lascia **8 GB completamente vuoti** sui 24 GB della tua scheda video.
Quegli 8 GB sprecati sul bancone potrebbero ospitare stabilmente **oltre 1.000 Specialisti tra i più usati**!

### Spreco 3: Lo Chef incrocia le braccia mentre aspetta i fogli!
In teoria, mentre l'aiutante trasporta sul passavivande i 6 Specialisti scelti per quel passaggio, lo Chef potrebbe già cucinare la parte comune della ricetta.
Nel codice originale per schede NVIDIA c'è però un comando di "Stop" che **obbliga l'aiutante ad aspettare che lo Chef abbia finito, e poi obbliga lo Chef a fermarsi e aspettare che l'aiutante abbia finito**, invece di lavorare insieme nello stesso momento!

### Spreco 4: Controlla e ricarica due volte gli stessi fogli!
Per una svista nel codice, ad ogni singolo piano del modello (40 volte per ogni parola!), il programma controlla quali Specialisti servono, li carica sul bancone e un istante dopo **rifà da capo lo stesso controllo** prima di usarli.

---

## 4. Cosa faremo noi, passo dopo passo

Lavoreremo su una **copia separata del codice** (così l'originale di Antirez rimane intatto) e faremo queste modifiche mirate:

### Passo 1: Misuriamo la velocità di partenza ("Baseline")
Appena finisce lo scaricamento del modello, facciamo una prova col programma così com'è adesso e scriviamo su un foglio quante parole al secondo genera (probabilmente circa **1,5 – 2 parole al secondo**, quindi troppo lento).

### Passo 2: Usiamo la tua Dispensa da 64 GB (con un margine di sicurezza!)
Modificheremo il programma per organizzare gli 81 GB del modello in **3 posti precisi**:
1. **Sul Bancone della Scheda Video (24 GB)**:
   * Mettiamo i **13 GB** fissi + circa **8 GB** con i **1.200 Specialisti più richiesti in assoluto** (quelli che servono quasi a ogni frase).
   * Poiché questi 21 GB sono già sul bancone dello Chef, **li cancelliamo dalla Dispensa (RAM)** per non occupare spazio due volte!
2. **Nella Dispensa in Cucina (i tuoi 64 GB di RAM)**:
   * Teniamo pronti circa **48 GB di Specialisti**, lasciando **sempre 12 GB di memoria RAM completamente liberi** per Ubuntu, il browser e i tuoi programmi (così il PC non rallenta e non va mai in blocco!).
   * Quando serve uno di questi Specialisti, passa dal passavivande veloce a **27 GB al secondo** invece che dalla cantina!
3. **In Cantina (Disco SSD)**:
   * Ci restano solo i ~12 GB degli Specialisti più rari (quelli che vengono chiamati pochissimo).

### Passo 3: Facciamo lavorare Chef e Aiutante in contemporanea
Togliamo i "semafori rossi" inutili nel codice: mentre lo Chef calcola la parte fissa della ricetta, il passavivande trasporta già in parallelo gli Specialisti che serviranno un centesimo di secondo dopo, ed eliminiamo il doppio controllo ripetuto.

### Passo 4: Verifica finale e Protezione del PC
* **Protezione Corrente Elettrica (già attiva!)**: Abbiamo limitato la scheda video a **230 Watt** invece di 350 Watt. Perde meno del 4% di velocità, ma scalda pochissimo e **impedisce al PC di spegnersi di colpo** per colpa dell'alimentatore.
* **Protezione Memoria (già attiva!)**: Abbiamo detto a Linux (`swappiness=10`) di dare sempre priorità assoluta ai programmi del computer rispetto al ricettario dell'intelligenza artificiale.
* **Prova finale**: Rimisuriamo la velocità per verificare di aver raggiunto e superato l'obiettivo delle **8 – 12 parole (token) al secondo** e controlliamo che il modello risponda in modo intelligente e preciso.
* **Feedback per Antirez**: Alla fine avremo una tabella chiara ("Prima vs Dopo") da mandare ad Antirez per mostrargli come far volare il suo programma anche sui PC con 64 GB di RAM e una scheda da 24 GB!

---

## 5. Il Glossario dei Termini Tecnici (Per imparare il "dietro le quinte")

Ecco il significato esatto dei termini ingegneristici che usiamo nel progetto, collegati alla nostra metafora della cucina:

### 1. `MoE` (*Mixture of Experts* — Miscela di Esperti) e `Router`
* **Cos'è**: È l'architettura di DeepSeek V4. Invece di avere un unico blocco monolitico di pesi (modello "denso") che viene letto tutto a ogni parola, ogni strato del modello è diviso in **256 piccoli sotto-modelli ("Esperti")**.
* **Il `Router` (Smistatore)**: Prima di ogni strato c'è un piccolissimo circuito chiamato *Router* che guarda la parola corrente e sceglie in meno di un millisecondo **i 6 Esperti migliori su 256** per elaborarla.

### 2. `mmap` (*Memory-Mapped File*) e `Page Cache`
* **Cos'è**: È un meccanismo del sistema operativo Linux per leggere file enormi (come i nostri 81 GB di modello). Invece di caricare tutto il file con un comando di lettura classico, il programma chiede a Linux di "proiettare" il file negli indirizzi della memoria RAM (`mmap`).
* **La `Page Cache` (Cache delle Pagine)**: Quando il programma legge un pezzo del file mappato, Linux ne conserva automaticamente una copia nei 64 GB di RAM libera (divisa in "pagine" da 4 KB o 2 MB). Se quel pezzo serve di nuovo, Linux lo fornisce istantaneamente dalla RAM senza toccare il disco SSD. E se un altro programma del PC ha bisogno di RAM, Linux può cancellare istantaneamente quelle pagine perché tanto l'originale è salvo sul disco!

### 3. `O_DIRECT` e `POSIX_FADV_DONTNEED`
* **`O_DIRECT` (Direct I/O)**: È un flag con cui un programma apre un file dicendo a Linux: *"Ignora completamente la RAM (`Page Cache`), leggi sempre direttamente dal disco SSD"*. È utile solo su computer con pochissima RAM, ma sul tuo PC con 64 GB è un freno a mano tirato!
* **`POSIX_FADV_DONTNEED`**: È una chiamata di sistema (`posix_fadvise`) con cui il programma dice a Linux: *"Questo pezzo di file che ho appena letto non mi servirà più, cancellalo pure subito dalla RAM"*. Nel codice originale di `ds4`, questo comando viene chiamato per errore anche sugli Esperti che invece riservono pochi istanti dopo!

### 4. `DMA` (*Direct Memory Access* — Accesso Diretto alla Memoria)
* **Cos'è**: Normalmente, per copiare dati da un posto all'altro, il processore del PC (la CPU) deve leggere ogni singolo numero e riscriverlo a destinazione. Con il **DMA**, invece, la CPU dà solo un ordine iniziale (*"Copia 6,75 MB da questo indirizzo della RAM a questo indirizzo della scheda video"*) e poi **se ne lava le mani**.
* **Chi fa il lavoro**: Sulla tua RTX 3090 c'è un chip dedicato chiamato **Copy Engine (Motore di Copia DMA)** che preleva autonomamente i dati dalla RAM attraverso le piste della scheda madre (il bus PCIe 4.0) a **27–28 GB al secondo**, mentre la CPU e il cuore di calcolo della scheda video fanno altro!

### 5. `Pinned Host Memory` (Memoria "Fissata / Inchiodata" in RAM)
* **Perché serve per il DMA**: Linux, per gestire la memoria RAM, si riserva il diritto di spostare in ogni istante le pagine di memoria da un indirizzo fisico all'altro o metterle sul disco di scambio (*swap*).
* **Il problema**: Il motore **DMA** della scheda video non capisce le tabelle dinamiche di Linux: ha bisogno che durante il trasferimento i dati stiano ad un **indirizzo fisico fisso che non può muoversi**.
* **La soluzione (`Pinned Memory` / `cudaMallocHost`)**: "Pinnare" (inchiodare) un blocco di RAM significa dire a Linux: *"Vietato spostare o toccare questo blocco di memoria finché non te lo dico io"*.
* **Perché ne usiamo poca (solo 64–256 MB)**: Se "inchiodassimo" 55 GB di RAM su 64 GB, Linux non potrebbe più muoversi e bloccherebbe il PC. Invece, creando una piccola **"zona di carico inchiodata" da 64–256 MB** (*Staging Buffer*), otteniamo la massima velocità del DMA lasciando il resto dei 48 GB libero di essere gestito in sicurezza da Linux!

### 6. `CUDA Stream` (Flussi di Lavoro CUDA) e `Ping-Pong Buffer`
* **`CUDA Stream`**: È una "corsia" di comandi inviati alla scheda video. Dentro una stessa corsia, la scheda video esegue i comandi rigorosamente in fila indiana (prima A, poi B, poi C). Ma se creiamo **due corsie separate** (una **Corsia di Calcolo** e una **Corsia di Trasferimento DMA**), la RTX 3090 può **calcolare sulla prima corsia e contemporaneamente trasferire dati sulla seconda corsia**!
* **`Ping-Pong Buffer` (Doppio Buffer)**: È la tecnica con cui si usano due piccoli vassoi in memoria (`Buffer A` e `Buffer B`). Mentre la scheda video legge e calcola i dati dal `Buffer A` (*Ping*), il motore DMA riempie già il `Buffer B` con i dati del passaggio successivo (*Pong*). Al giro dopo si scambiano i ruoli, azzerando i tempi morti!
* **`cudaDeviceSynchronize()` vs `cudaEvent_t`**: `cudaDeviceSynchronize()` è un "freno d'emergenza globale" che ferma tutta la scheda video e tutte le corsie finché tutto non è fermo (ed è il motivo per cui nel codice attuale le due corsie non vanno in parallelo). `cudaEvent_t` è invece un "semaforo intelligente" che coordina solo il punto esatto in cui due corsie devono scambiarsi i dati.

### 7. Politica di Cache: `LRU` vs `LFU`
Quando lo spazio per gli Esperti sulla scheda video (~1.200 posti su 10.240 totali) è pieno e serve far posto a un Esperto nuovo, chi buttiamo fuori?
* **`LRU` (*Least Recently Used* — Meno Usato di Recente)**: Butta fuori l'Esperto che non viene toccato da più tempo. È semplice, ma ha un difetto: se per una parola strana entra un Esperto rarissimo che non servirà mai più, rischia di buttare fuori un Esperto famosissimo che era fermo solo da due parole!
* **`LFU` (*Least Frequently Used* — Meno Usato in Assoluto / per Frequenza)**: Conta **quante volte** ogni Esperto è stato chiamato. Combinando `LRU + LFU` (quello che faremo noi), gli Esperti "star" (chiamati centinaia di volte) restano protetti sulla scheda video e vengono sostituiti solo gli Esperti davvero occasionali!

---

## 6. Metodo Scientifico di Benchmark (Come misureremo i risultati sul serio)

Hai toccato un punto fondamentale: nel mondo dell'AI locale troppa gente cambia un parametro, lancia una frase a caso, vede passare da `5.1` a `5.6 tok/s` e grida al miracolo. **Quello è rumore statistico, non è scienza.**

In un modello **MoE (a Esperti)** come DeepSeek V4, la velocità dipende tantissimo da **quali Esperti vengono scelti dal Router**:
* Se fai una domanda banale o ripetitiva, il Router chiama sempre gli stessi pochi Esperti $\rightarrow$ sono già tutti in cache sulla scheda video $\rightarrow$ velocità altissima (ma "finta").
* Se chiedi di scrivere codice C complesso o risolvere un problema matematico, il Router salta tra centinaia di Esperti diversi $\rightarrow$ molti più trasferimenti sul bus PCIe $\rightarrow$ velocità reale sotto sforzo.

Per avere numeri **scientificamente inattaccabili** (degni di essere mandati ad Antirez), applicheremo questo protocollo rigoroso a ogni fase:

### A. Campione Multi-Dominio (4 Input diversi fissi)
Non useremo una sola frase, ma una ** batteria di 4 test distinti** (salvati su file per essere identici al 100% in ogni prova):
1. **Dominio 1 — Coding & Architettura (`bench_code.txt`)**: Analisi e scrittura di codice C/CUDA complesso (attiva gli esperti di programmazione e logica).
2. **Dominio 2 — Ragionamento Matematico (`bench_math.txt`)**: Problema algebrico/combinatorio multi-step (attiva gli esperti simbolici e matematici).
3. **Dominio 3 — Prosa & Sintesi in Italiano (`bench_prose.txt`)**: Scrittura argomentativa in lingua italiana (attiva gli esperti linguistici multilingua).
4. **Dominio 4 — Benchmark Strutturato Multi-Frontiera (`./ds4-bench --teacher-forced-decode`)**: Il tool di benchmark interno di `ds4` che forza esattamente la stessa sequenza di token a lunghezze di contesto crescenti (`2048`, `4096`, `8192`).

### B. Determinismo Assoluto (`Temperature = 0`)
Tutte le prove di velocità verranno eseguite con `--temp 0` (scelta deterministica *greedy*) e `--nothink` / `--think` fissati.
In questo modo, **a parità di input, il modello DEVE generare esattamente le stesse identiche parole e chiamare esattamente gli stessi identici Esperti** in tutte le versioni del codice! Se cambiasse anche solo una parola, staremmo confrontando mele con pere.

### C. Separazione "Cold Start" vs "Warm Steady-State", Modello LMM e Intervalli di Confidenza al 95%
Per ogni configurazione (Baseline, Ottimizzazione 1, Ottimizzazione 2, ecc.):
* **Run 0 (Cold Cache — Disco freddo)**: Misurato a parte dopo aver svuotato la cache o appena avviato, per sapere quanto ci mette il "primo avvio a freddo".
* **Run 1..5 (5 Ripetizioni a regime per ciascuno dei 4 Domini = 20 misurazioni per versione)**:
  Tutti i dati verranno salvati in una tabella `benchmarks/results.csv` e analizzati con un **Modello Lineare a Effetti Misti (`LMM` — *Linear Mixed-Effects Model*)**:
  $$\text{Token/s} \sim \text{Versione\_Codice} + (1 \mid \text{Dominio\_Input})$$
  * **Perché l'LMM è la scelta statisticamente perfetta qui?** Perché domande diverse (codice, matematica, prosa) hanno naturalmente velocità leggermente diverse (alcune chiamano esperti più facili, altre più sparsi). L'LMM separa la variabilità dovuta al *tipo di domanda* (effetto casuale `Dominio_Input`) dall'effetto puro della *nostra modifica al codice* (effetto fisso `Versione_Codice`).
  * Per ogni modifica calcoleremo l'**Intervallo di Confidenza al 95% ($\text{IC}_{95\%}$)** e il **$p$-value**: diremo che una modifica migliora le prestazioni **solo se l'intervallo di confidenza al 95% della differenza è interamente sopra lo zero** ($p < 0.05$).

### D. Controllo di Invarianza Matematica (`--dump-logits`)
La regola d'oro dell'ingegneria: **è facile andare più veloce se sbagli i calcoli!**
Per dimostrare che le nostre ottimizzazioni non degradano di una virgola l'intelligenza del modello, salveremo i *logits* (le probabilità numeriche al millesimo calcolate dal modello con `--dump-logits`) nella Baseline e verificheremo che dopo le nostre modifiche i numeri prodotti dalla RTX 3090 siano **matematicamente identici**!


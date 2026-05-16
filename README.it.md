# DwarfStar 4

*Leggi questo in [English](README.md).*

DwarfStar 4 è un piccolo motore di inferenza nativo specifico per **DeepSeek
V4 Flash**. È deliberatamente ristretto: non è un runner GGUF generico né un
wrapper attorno a un altro runtime, è completamente autonomo. Oltre a
eseguire il modello in modo corretto e veloce, l'obiettivo del progetto è
fornire caricamento specifico per DS4, rendering dei prompt, tool calling,
gestione dello stato KV (in RAM e su disco) e API server, tutto pronto per
funzionare con agenti di coding o con la CLI inclusa. Ci sono anche tool per
la generazione di GGUF e imatrix e per il testing di qualità e velocità.

Backend supportati:
* **Metal** è il nostro target principale. A partire dai MacBook con 96GB di RAM.
* **NVIDIA CUDA** con cura particolare per il DGX Spark.
* **AMD ROCm** è supportato solo nel branch [rocm](https://github.com/antirez/ds4/tree/rocm). È tenuto separato dal main perché io (antirez) non ho accesso diretto all'hardware, quindi la community fa rebase del branch quando serve.

Questo progetto non esisterebbe senza **llama.cpp e GGML**, leggete la
sezione ringraziamenti, un grande grazie a Georgi Gerganov e a tutti gli
altri contributor.

## Motivazioni

Perché crediamo che DeepSeek v4 Flash sia un modello speciale che merita un
motore dedicato? Perché confrontandolo con modelli densi più piccoli e
potenti possiamo riportare che:

1. DeepSeek v4 Flash è più veloce grazie ai meno parametri attivi.
2. In modalità thinking, se eviti *max thinking*, produce una sezione di
   pensiero molto più corta di altri modelli, anche 1/5 in molti casi, e
   soprattutto la lunghezza è **proporzionale alla complessità del
   problema**. Questo rende DeepSeek v4 Flash usabile con thinking attivo
   quando altri modelli sono praticamente inutilizzabili nelle stesse
   condizioni.
3. Il modello ha una context window di **1 milione di token**.
4. Essendo grande, conosce più cose se si campiona ai limiti della conoscenza.
   Ad esempio chiedendo di show italiani o di politica si scopre presto che
   284B parametri sono molti più di 27B o 35B.
5. Scrive molto meglio in inglese e italiano. *Sembra* un modello quasi-frontier.
6. La KV cache è incredibilmente compressa, e questo permette inferenza a
   contesto lungo su computer locali e la **persistenza della KV cache su disco**.
7. Funziona bene a 2 bit di quantizzazione, se quantizzato in modo speciale
   (vedi sotto). Questo permette di girarlo su MacBook con 128GB di RAM (e
   molti hanno riportato di farlo funzionare anche con 96GB, persino con
   una context window di 250k!).
8. Ci aspettiamo che DeepSeek rilasci **versioni aggiornate di v4 Flash** in
   futuro, ancora migliori di quella attuale.

Detto questo, alcune cose importanti su questo progetto:

* Il panorama dell'inferenza locale contiene molti progetti eccellenti, ma
  i modelli nuovi escono di continuo e l'attenzione si sposta subito sul
  prossimo. Questo progetto fa una scommessa deliberatamente ristretta: un
  modello alla volta, validazione con vettori ufficiali (logit ottenuti
  dall'implementazione ufficiale), test a contesto lungo e abbastanza
  integrazione con gli agenti per sapere se funziona davvero. Il modello
  esatto può cambiare man mano che il panorama evolve, ma il vincolo
  rimane: inferenza locale credibile su macchine personali di fascia alta
  o Mac Studio, a partire da 96/128GB di memoria.
* Questo software è sviluppato con **forte assistenza di GPT 5.5** e con
  umani che guidano le idee, i test e il debug. Lo diciamo apertamente
  perché ha plasmato il modo in cui è stato costruito. Se non sei d'accordo
  con codice sviluppato dall'AI, questo software non fa per te. Il
  ringraziamento qui sotto è altrettanto importante: tutto questo non
  esisterebbe senza `llama.cpp` e GGML, in larga parte scritti a mano.
* Questa implementazione è basata sull'idea che KV cache compresse come
  quella di DeepSeek v4 e gli SSD veloci dei MacBook moderni dovrebbero
  cambiare la nostra idea che la KV cache appartenga alla RAM. **La KV
  cache è in realtà una cittadina di prima classe del disco.**
* La nostra visione è che l'inferenza locale dovrebbe essere un insieme di
  tre cose che funzionano bene insieme, out of the box: A) motore di
  inferenza con API HTTP + B) GGUF creato apposta per girare bene sotto
  un dato motore e date assunzioni + C) testing e validazione con
  implementazioni di agenti di coding. Questo motore di inferenza gira
  solo con i file GGUF forniti. Viene testato contro logit ottenuti
  ufficialmente a diversi context size. Questo progetto esiste perché
  volevamo far sembrare finito end-to-end un modello locale, non solo
  eseguibile. Per ora però è solo codice in stato alpha, quindi
  probabilmente non ci siamo ancora.
* La graph path ottimizzata punta a **Metal su macOS** e **CUDA su
  Linux**. La CPU path è solo per i check di correttezza e la diagnostica
  di modello/tokenizer. Per build CPU-only su Linux usa `make cpu`, che
  costruisce i normali `./ds4` e `./ds4-server` senza CUDA o Metal. Su
  macOS, **attenzione: le versioni attuali di macOS hanno un bug nella
  virtual memory che fa crashare il kernel** se provi a girare il codice
  CPU. Ricordi? Software sucks. Non è stato possibile fixare l'inferenza
  CPU per evitare il crash, perché ogni volta bisogna riavviare il
  computer, che non è divertente. Aiutaci, se ne hai il coraggio.

## Ringraziamenti a llama.cpp e GGML

`ds4.c` non si linka contro GGML, ma **esiste grazie alla strada aperta dal
progetto llama.cpp e ai kernel, alle quantizzazioni, all'ecosistema GGUF e
alla conoscenza ingegneristica sudata sviluppata lì**.
Siamo grati e debitori a [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
e ai suoi contributor. La loro implementazione, i kernel, i test e le scelte
di design sono stati un riferimento essenziale mentre costruivamo questa
inferenza specifica per DeepSeek V4 Flash. Alcuni pezzi a livello di codice
sono mantenuti o adattati qui sotto licenza MIT: i layout e le tabelle dei
quant GGUF, la logica di quant/dot in CPU e alcuni kernel. Per questo
motivo, e perché siamo genuinamente grati, manteniamo la nota di copyright
degli autori di GGML nel nostro file `LICENSE`.

## Stato

Codice e file GGUF sono da considerare di **qualità alpha** perché
l'inferenza e il model serving sono una materia complicata e tutto questo
esiste solo da pochi giorni. Ci vorranno mesi per raggiungere una forma più
stabile. Cerchiamo comunque di tenere il progetto in uno stato usabile e
stiamo facendo progressi. Se hai problemi, usa `--trace` per loggare le
sessioni e apri issue includendo il trace completo.

## Modifiche su questo fork

Questo è un fork community (`SiNaPsEr0x/ds4`). L'elenco sotto traccia le
modifiche user-visible rispetto a upstream `antirez/ds4`. La modifica più
recente è in cima.

- **Build nativa Windows + CUDA** per `ds4-server`, `ds4-bench` e
  `ds4-eval` tramite MSVC + CUDA Toolkit ≥ 12.8 (sm_120 Blackwell).
  Vedi [Port Windows (CUDA)](#port-windows-cuda) sotto per prerequisiti,
  comandi di build e limitazioni attuali (la CLI `ds4` è rinviata).
- **README italiano** (questo file, [`README.it.md`](README.it.md)) che
  copre tutti i contenuti della versione inglese. I sub-README restano
  solo in inglese.
- **`CLAUDE.md`** con build commands, panoramica dell'architettura e
  vincoli del progetto (no C++ nel main path, no link a GGML/llama.cpp,
  invarianti dell'instance-lock), per istruire le sessioni Claude Code
  sulle regole del progetto.

## Altra documentazione

Se cerchi cose molto specifiche, abbiamo altri sub-README. Altrimenti per
l'uso normale continua a leggere le prossime sezioni.

- [CONTRIBUTING.md](CONTRIBUTING.md): guida ai test di regressione di
  correttezza e velocità per i contributor. **Leggilo prima di mandare una
  pull request.**
- [gguf-tools/README.md](gguf-tools/README.md): generazione GGUF offline,
  raccolta imatrix, tooling di quantizzazione e check di qualità.
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md): come viene
  raccolta e usata l'imatrix per la MoE routed.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  come viene generato il corpus di prompt per la calibrazione.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  come i GGUF locali vengono valutati contro le continuazioni ufficiali di
  DeepSeek V4 Flash.
- [dir-steering/README.md](dir-steering/README.md): dati di steering
  direzionale, generazione dei vettori e uso.
- [speed-bench/README.md](speed-bench/README.md): file CSV dei benchmark e
  generazione dei grafici.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): vettori di
  continuazione ufficiali usati per i check di regressione.

I sub-README sono attualmente disponibili solo in inglese.

## Pesi del modello

Questa implementazione funziona solo con i GGUF di DeepSeek V4 Flash
pubblicati per questo progetto. Non è un loader GGUF generico, e file
DeepSeek/GGUF arbitrari non avranno il layout dei tensori, il mix di
quantizzazioni, i metadati o lo stato MTP opzionale che il motore si
aspetta. Le quantizzazioni a 2 bit fornite qui non sono uno scherzo: si
comportano bene, funzionano sotto agenti di coding, chiamano tool in modo
affidabile. I quant a 2 bit usano una quantizzazione molto asimmetrica:
solo gli expert routed della MoE sono quantizzati, up/gate a `IQ2_XXS`,
down a `Q2_K`. Sono la maggioranza dello spazio occupato dal modello: gli
altri componenti (shared expert, proiezioni, routing) sono lasciati intatti
per garantire la qualità.

Scarica un modello principale. **Preferisci le versioni imatrix.**

```sh
./download_model.sh q2-imatrix   # macchine 96/128 GB RAM, q2 tarato con imatrix
./download_model.sh q4-imatrix   # macchine >= 256 GB RAM, q4 tarato con imatrix
```

I vecchi file GGUF restano disponibili se ti servono specificamente i quant
non-imatrix:

```sh
./download_model.sh q2           # macchine 96/128 GB RAM, legacy non-imatrix
./download_model.sh q4           # macchine >= 256 GB RAM, legacy non-imatrix
```

Lo script scarica da `https://huggingface.co/antirez/deepseek-v4-gguf`,
salva i file sotto `./gguf/`, riprende download parziali con `curl -C -` e
aggiorna `./ds4flash.gguf` per puntare al modello q2-imatrix/q4-imatrix/q2/q4
selezionato. I pesi q2 XXS plain sono prodotti con il solo vettore di
importanza dei pesi, senza imatrix. Le varianti imatrix sono preferibili.
L'autenticazione è opzionale per i download pubblici, ma vengono usati
`--token TOKEN`, `HF_TOKEN` o la cache locale del token Hugging Face quando
presenti.

Se vuoi rigenerare i GGUF o raccogliere una nuova imatrix, vedi
[gguf-tools/README.md](gguf-tools/README.md). Questi tool sono pensati per
il lavoro offline di model-building e possono richiedere molto tempo sui
pesi pieni di DeepSeek V4 Flash.

`./download_model.sh mtp` recupera il GGUF opzionale di supporto al
decoding speculativo. Può essere usato con q2-imatrix, q4-imatrix, q2 e q4,
ma va abilitato esplicitamente con `--mtp`. Il path MTP/speculative
decoding attuale è ancora sperimentale: è correctness-gated e al momento
dà al massimo un leggero speedup, non un guadagno significativo di velocità
di generazione.

Poi compila:

```sh
# POSIX (macOS + Linux): usa il Makefile
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, altre GPU CUDA locali
make cpu              # build CPU-only per diagnostica

# Windows: usa CMake + MSVC + CUDA (build di ds4-server, ds4-bench, ds4-eval)
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
```

La build Windows richiede Visual Studio 2022, CUDA Toolkit ≥ 12.8, CMake ≥
3.24 e Windows 10 build 19041 o più recente; la CLI interattiva `ds4` non è
ancora buildata su Windows. Vedi [Port Windows (CUDA)](#port-windows-cuda)
più in basso per i prerequisiti completi e le limitazioni attuali.

`./ds4flash.gguf` è il path di default del modello usato da entrambi i
binari. Passa `-m` per selezionare un altro GGUF supportato da `./gguf/`.
Esegui `./ds4 --help` e `./ds4-server --help` per la lista completa dei
flag.

## Velocità

Questi sono numeri single-run della CLI Metal con `--ctx 32768`,
`--nothink`, decoding greedy e `-n 256`. Il prompt corto è un normale
prompt di una piccola storia in italiano. I prompt lunghi esercitano il
prefill chunked più il decode a contesto lungo. Q4 richiede la macchina
della classe di memoria più grande, quindi i numeri Q4 di M3 Max sono `N/A`.

| Macchina | Quant | Prompt | Prefill | Generazione |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | corto | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 token | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | corto | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | lungo | N/A | N/A |
| Mac Studio M3 Ultra, 512 GB | q2 | corto | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 token | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | corto | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 token | 448.82 t/s | 26.62 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 token | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)

## Benchmarking

`ds4-bench` misura il throughput istantaneo di prefill e generazione a
frontiere di contesto invece di riportare una media su tutta la run. Carica
il modello una volta, cammina su una sequenza fissa di token verso
frontiere come 2048, 4096, 6144 e usa il prefill incrementale così che
ogni riga misuri solo l'intervallo di token appena aggiunto. Dopo ogni
frontiera salva lo stato KV vivo in memoria, genera una sonda greedy fissa
non-EOS, ripristina lo snapshot di memoria e prosegue con il prefill.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Il file di esempio è un testo di pubblico dominio ripulito da Project
Gutenberg dei *Promessi Sposi* di Alessandro Manzoni (ebook #45334), con
header e footer Gutenberg rimossi: <https://www.gutenberg.org/ebooks/45334>.

Usa `--step-incr N` per una spaziatura lineare diversa o `--step-mul F`
per sweep esponenziali. L'output è CSV con una riga per frontiera:
token/sec dell'ultimo intervallo di prefill, token/sec di generazione a
quella frontiera e `kvcache_bytes`.

## Valutazione di capacità

`ds4-eval` è un piccolo benchmark di integrazione con il modello reale. Non
è un runner da leaderboard e non va riportato come punteggio ufficiale
GPQA, SuperGPQA o AIME: le domande sono un sottoinsieme embedded di 75 item
scelto per rendere utile e visivamente ispezionabile il test di regressione
locale. Il programma carica il GGUF reale, fa il rendering dei chat prompt
DS4, stremma i token campionati in una TUI split-screen, valuta la risposta
finale e stampa un report per domanda con prompt token, token generati,
stato pass/fail, la risposta del modello e quella corretta.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

La run di default usa `--ctx 100000` e `--tokens 16000`, modalità thinking
abilitata e un cutoff `</think>` soft/hard del budget così che il modello
abbia spazio per produrre una risposta visibile. Premi `p` per mettere in
pausa, Su/Giù per ispezionare o selezionare un'altra domanda, e Invio per
eseguire la domanda selezionata. `--plain` disabilita la TUI.

Le 75 domande embedded sono interlacciate come 25 GPQA Diamond, 25
SuperGPQA e 25 AIME 2025. L'ordine è volutamente progressivo: le domande
iniziali sono utili smoke test, mentre quelle finali sono abbastanza
difficili che un buon modello reasoning ne dovrebbe ancora sbagliare
qualcuna.

Per un modello come DeepSeek V4 Flash il set va trattato come una hard
capability regression suite, non come uno unit test pass/fail:

- **GPQA Diamond** porta domande di scienza di livello laurea magistrale a
  risposta multipla. La model card di DeepSeek riporta risultati forti su
  Flash nel GPQA Diamond pieno in modalità thinking, ma i singoli item
  richiedono comunque ragionamento accurato di fisica, chimica o biologia
  e si perdono facilmente con piccole regressioni di prompt/rendering o
  sampling.
- **SuperGPQA** porta conoscenza specialistica ampia e domande di
  trasferimento di dominio. Il numero SuperGPQA sulla model card è molto
  più basso di GPQA Diamond, quindi questi item sono attesi come
  irregolari: alcuni sembrano banali, altri richiedono conoscenza
  professionale di nicchia o interpretazione esatta di una domanda in
  stile esame tradotto.
- **AIME 2025** porta matematica a contest con risposta esatta. Spesso
  sono gli item più impietosi del set: nessun prior a risposta multipla,
  nessun credito parziale e un singolo errore aritmetico o algebrico
  cambia il voto.

In pratica questo significa che `ds4-eval` non deve essere atteso come una
run 75/75. Serve a rispondere a una domanda ingegneristica più utile: dopo
una modifica di kernel, quantizzazione, prompt-rendering, KV-cache o
tool-streaming, DeepSeek V4 Flash continua a risolvere un mix
rappresentativo di problemi di scienza hard, conoscenza ampia e matematica
esatta usando lo stesso percorso di inferenza che gli utenti girano?

## CLI

Prompt one-shot:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

Senza `-p` parte il prompt interattivo:

```sh
./ds4
ds4>
```

La CLI interattiva è una vera chat DS4 multi-turn. Tiene il transcript
chat renderizzato e il checkpoint KV vivo del grafo, quindi ogni turno
estende la conversazione precedente. Comandi utili sono `/help`, `/think`,
`/think-max`, `/nothink`, `/ctx N`, `/read FILE` e `/quit`. Ctrl+C
interrompe la generazione corrente e torna a `ds4>`.

La CLI di default sta in modalità thinking. Usa `/nothink` o `--nothink`
per risposte dirette. `--mtp MTP.gguf --mtp-draft 2` abilita il path MTP
speculativo opzionale; è utile solo per decoding greedy, attualmente usa
un gate di confidenza (`--mtp-margin`) per evitare accept parziali lenti,
e va trattato come un path sperimentale di lieve speedup.

## Server

Avvia un server locale compatibile OpenAI/Anthropic:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Il server tiene un solo checkpoint backend/KV mutabile in memoria, così i
client stateless che ri-mandano una versione più lunga dello stesso prompt
possono riutilizzare il prefix condiviso invece di fare prefill dal token
zero.

Il parsing delle request e i socket girano nei thread client, ma
l'inferenza è serializzata attraverso un singolo graph worker. Il server
attuale non fa batching di request indipendenti; le request concorrenti
attendono il proprio turno sul singolo graph/sessione vivo.

Endpoint supportati:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` accetta i soliti `messages` stile OpenAI,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `seed`, `stream`, `stream_options.include_usage`, `tools` e
`tool_choice`. Gli schemi dei tool sono renderizzati nel formato DSML di
DeepSeek e le tool call DSML generate sono rimappate a tool call OpenAI.

`/v1/responses` accetta `input`, `instructions`, `tools`, `tool_choice`,
`max_output_tokens`, `temperature`, `top_p`, `stream` e `reasoning` stile
OpenAI Responses. È l'endpoint preferito per la Codex CLI. Il server tiene
le continuation Responses legate allo stato vivo quando possibile e può
fare fallback allo stesso rendering DSML e al riuso del prefix KV usato da
chat completions.

`/v1/messages` è l'endpoint Anthropic-compatibile usato dai client tipo
Claude Code. Accetta `system`, `messages`, `tools`, `tool_choice`,
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`
e i controlli di thinking. Gli usi dei tool sono restituiti come blocchi
`tool_use` Anthropic.

Gli endpoint chat, Responses e Anthropic supportano lo streaming SSE. In
modalità thinking, il reasoning viene streammato nella forma nativa
dell'API invece di essere mescolato col testo finale. Lo streaming chat
OpenAI stremma anche le tool call non appena l'invocazione DSML viene
riconosciuta: prima viene mandato l'header del tool, poi i byte dei
parametri vengono inoltrati come delta di
`tool_calls[].function.arguments` mentre la generazione continua.
L'endpoint Anthropic stremma il thinking e il testo in tempo reale, poi
emette blocchi `tool_use` strutturati quando il blocco tool generato è
completo. L'endpoint Responses stremma il lifecycle di eventi Responses
atteso da Codex, inclusi `response.output_text.delta`, eventi di argomenti
function-call ed eventi terminali `response.completed` /
`response.incomplete` / `response.failed`.

### Gestione delle tool call e canonicalizzazione

DeepSeek V4 Flash emette le tool call come [testo DSML](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). I client agente non rimandano lo stesso testo sulla request
successiva: mandano oggetti JSON tool-call normalizzati in stile
OpenAI/Anthropic. **Se il server renderizzasse quegli oggetti anche solo
leggermente in modo diverso, il prefix di byte renderizzato non
matcherebbe più il checkpoint KV vivo** e il turno successivo andrebbe
ricostruito.

La prima linea di difesa è l'exact replay. Ogni tool call riceve un tool
ID API non indovinabile, e il server ricorda `tool id -> blocco DSML
campionato esatto` in una mappa in memoria limitata, backed da radix
tree. Quando il client rimanda quel tool ID, il renderer del prompt usa i
byte DSML esatti campionati dal modello, non una formattazione fresca
approssimata. Questa mappa può anche essere salvata dentro i file della KV
cache, così l'exact replay sopravvive ai restart del server per le
history in cache.

**La canonicalizzazione è solo il path di backup.** Se il blocco DSML
esatto manca, o se l'exact replay è disabilitato con
`--disable-exact-dsml-tool-replay`, il server renderizza una forma DSML
deterministica dall'oggetto JSON. Dopo un turno di tool call confronta lo
stream di token campionati vivo con il prompt che la prossima request del
client renderizzerà. Se serve, riscrive il checkpoint vivo, o fa fallback
a uno snapshot KV su disco più vecchio e rigioca solo il suffix. Questo
tiene la continuation del modello allineata col transcript stateless
dell'API.

Durante la generazione, il server tratta anche la sintassi DSML in modo
diverso dal payload. Quando il modello emette struttura di protocollo
stabile come tag DSML, header di parametri, punteggiatura JSON o marker di
chiusura, il sampling è forzato a `temperature=0` così la tool call resta
parsable. Questa modalità greedy **non** si applica ai payload degli
argomenti: i body dei parametri `string=true` e i valori string JSON,
inclusi i contenuti dei file e il testo di edit, usano le impostazioni di
sampling normali della request. Quella separazione è importante: il
decoding deterministico aiuta la sintassi ma può creare testo ripetuto se
applicato a code o file body lunghi.

Esempio minimo OpenAI:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Uso con client agente

`ds4-server` può essere usato da agenti di coding locali che parlano chat
completion OpenAI-compatibili. Avvia prima il server e non impostare il
context limit del client sopra il `--ctx` con cui hai avviato il server:

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Puoi usare context più grande e cache più grande se vuoi. Il contesto
pieno di 1M token userà più o meno 26GB di memoria (solo l'indexer
compresso sarà circa 22GB), quindi configura un contesto che abbia senso
sul tuo sistema. Con 128GB di RAM gireresti i quant a 2 bit, che sono già
81GB, 26GB in più probabilmente saranno troppi, quindi una context window
di 100~300k token è più saggia. Tuttavia gli utenti hanno riportato di
riuscire a girare i quant 2bit con context da 250k su Mac con appena 96GB
di memoria di sistema: assicurati di killare i processi che usano troppa
memoria, se hai intenzione di farlo ;)

Il limite di output `384000` qui sotto evita i cap dei token perché il
modello è in grado di generare reply molto lunghe altrimenti (fino a 384k
token). Il server si ferma comunque quando la context window configurata è
piena.

Per **opencode**, aggiungi una entry provider e una agent a
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

Per **Pi**, aggiungi un provider a `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Opzionalmente rendilo il modello Pi di default in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

Per **Codex CLI**, usa la wire API Responses:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Poi esegui:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

Per **Claude Code**, usa l'endpoint Anthropic-compatibile. Un wrapper come
questo corrisponde al setup locale `~/bin/claude-ds4`:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code può mandare un prompt iniziale grande, spesso intorno ai 25k
token, prima di iniziare a fare lavoro utile. Tieni abilitato
`--kv-disk-dir`: dopo il primo prefill costoso la disk KV cache permette
alle continuation successive o alle sessioni riavviate di riutilizzare il
prefix salvato invece di processare di nuovo l'intero prompt.

## Modalità di thinking

DeepSeek V4 Flash ha modalità distinte non-thinking, thinking e Think Max.
Il server di default sta in thinking mode. `reasoning_effort=max` richiede
Think Max, ma viene applicato solo quando il context size è abbastanza
grande per la raccomandazione della model card; context più piccoli
ricadono sul thinking normale. `reasoning_effort=xhigh` di OpenAI mappa
comunque al thinking normale, non a Think Max.

Per reply dirette, usa `thinking: {"type":"disabled"}`, `think:false`, o
un alias di modello non-thinking come `deepseek-chat`.

## Disk KV Cache

Le API chat/completion sono stateless: i client agente di solito rimandano
l'intera conversazione ad ogni request. `ds4-server` prova prima il check
economico di prefix di token esatto, poi fa fallback al confronto dei byte
del prompt renderizzato con i byte del checkpoint decodato. Il checkpoint
vivo in memoria copre la sessione corrente; la disk KV cache permette ai
prefix utili di sopravvivere ai cambi di sessione e ai restart del server.

Per ragioni di RAM al momento c'è una sola KV cache viva in memoria.
Quando una sessione nuova non collegata la sostituisce, il checkpoint
vecchio può essere ripreso senza re-processing solo se è stato scritto
nella disk KV cache. In altre parole, la memory cache gestisce la
sessione attiva; la disk cache è il meccanismo di resume per sessioni
diverse.

Abilitalo con:

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

La chiave di cache è lo SHA1 del prefix di byte renderizzato e i file
sono nominati `<sha1>.kv`. Il payload DS4 contiene comunque gli ID token
esatti e lo stato del grafo per quel prefix. Questo è importante per le
chat continuate: il modello può aver generato un token il cui testo
decodato viene poi rimandato dal client come due token canonici di prompt.
Un hit di prefix di byte renderizzato può comunque riusare il checkpoint e
tokenizzare solo il nuovo suffix.
Il file è scritto intenzionalmente con I/O ordinari `read`/`write`, non
con `mmap`, così il ripristino delle entry della cache non aggiunge altre
mappature VM a un processo che già mappa il modello.

Le tool call mantengono anche una mappa di exact-DSML replay limitata
chiavata da tool ID non indovinabili, così l'history JSON del client può
essere renderizzata di nuovo nell'esatto testo campionato. La mappa in
RAM tiene di default fino a 100000 ID; regolala con `--tool-memory-max-ids`.
Usa `--disable-exact-dsml-tool-replay` per disabilitare e fare fallback
al rendering canonico JSON-to-DSML.

Su disco, un file di cache è:

```text
KVC header fisso, 48 byte
u32 rendered_text_bytes
rendered_text_bytes di token text UTF-8-ish
DS4 session payload, payload_bytes dall'header KVC
mappa tool-id opzionale appesa
```

L'header fisso è little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     bit di quant degli expert routed, al momento 2 o 4
5   u8     motivo del save: 0 sconosciuto, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     flag di estensione, bit 0 = mappa tool-id appesa
7   u8     reserved
8   u32    conteggio token cachati
12  u32    hit count
16  u32    context size per cui lo snapshot è stato scritto
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    byte count del DS4 session payload
```

Il rendered text è il testo decodificato dal tokenizer per il prefix di
token cachati. È sia il prefix ispezionabile dall'umano sia l'identità di
lookup: il suo SHA1 è il filename e un file è riutilizzabile solo quando
quei byte sono un prefix del prompt renderizzato in ingresso. Dopo il
load, i token esatti del checkpoint dal payload DS4 restano autoritativi e
solo il suffix di testo in ingresso dopo i byte cachati viene tokenizzato.

La mappa tool-id opzionale è presente solo quando il bit 0 di estensione
dell'header è settato. Le sezioni appese usano un ordine di bit fisso,
così bit di estensione futuri possono aggiungere campi senza ambiguità.
La mappa salva ID tool call API non indovinabili verso l'esatto blocco
DSML campionato dal modello. Vengono salvate solo le mapping il cui
blocco DSML è presente nel rendered text cachato. Questo permette ai
server riavviati di renderizzare history client successiva byte-per-byte
come l'output originale del modello, anche se il client riordina gli
argomenti JSON.

L'attuale sezione di mappa tool-id è:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

Per ogni entry:
0   u32    lunghezza in byte del tool id
4   u32    lunghezza in byte del DSML campionato
8   bytes  tool id
... bytes  blocco DSML campionato esatto
```

La sezione è memoria di replay ausiliaria, non model state. Un cache hit
ripristina prima il session payload, poi carica la mappa se presente.
Prima di renderizzare una request, il server può anche scansionare i file
di cache per i tool ID presenti nell'history del client e caricare solo
quelle mapping, così un exact DSML replay può sopravvivere ai restart del
server anche quando lo snapshot KV matching non è quello effettivamente
usato per l'hit del rendered prefix.

Il DS4 session payload inizia con tredici campi `u32` little-endian:

```text
0   magic = "DSV4"
1   payload version = 1
2   context size salvato
3   prefill chunk size
4   capacità del ring KV raw
5   lunghezza della sliding-window raw
6   capacità KV compressa
7   conteggio token del checkpoint
8   conteggio layer
9   dimensione KV raw/head
10  dimensione head dell'indexer
11  vocabulary size
12  righe raw vive serializzate sotto
```

Poi salva:

- `u32[token_count]` ID token del checkpoint.
- `float32[vocab_size]` logit per il prossimo token dopo quel checkpoint.
- `u32[layer_count]` conteggi di riga della compressed attention.
- `u32[layer_count]` conteggi di riga dell'indexer ratio-4.
- Per ogni layer: le live raw sliding-window KV rows, scritte in ordine di
  posizione logica e non di posizione fisica nel ring.
- Per layer compressi: live compressed KV rows e tensori frontier del
  compressore.
- Per layer compressi ratio-4: live indexer compressed rows e tensori
  frontier dell'indexer.

I logit sono valori IEEE-754 `float32` raw dal buffer host
`ds4_session`. Vengono salvati subito dopo i token di checkpoint così uno
snapshot caricato può campionare o continuare dall'esatta distribuzione
del prossimo token senza eseguire un decode step extra. Logit/stato di
draft MTP non sono persistiti; dopo aver caricato un checkpoint disk lo
stato di draft viene invalidato e ricostruito dalla generazione normale.

Il payload dei tensori è stato KV/sessione DS4-specifico, non un dump
generico di un inference graph. Ci si aspetta che sia portabile solo tra
build compatibili di `ds4.c` per questo layout di modello.

La cache memorizza i checkpoint in quattro momenti:

- `cold`: dopo che un primo prompt lungo raggiunge un prefix stabile,
  prima della generazione.
- `continued`: quando il prefill o la generazione raggiunge la prossima
  frontiera assoluta allineata.
- `evict`: prima che una request non collegata sostituisca la sessione
  viva in memoria.
- `shutdown`: quando il server esce in modo pulito.

I cold save tagliano intenzionalmente un piccolo suffix di token e
allineano sotto a un boundary di prefill chunk. Questo evita i miss
comuni di retokenizzazione al boundary BPE quando una request futura
appende testo allo stesso prompt. I default sono conservativi: salva
prefix di almeno 512 token, fa cold-save di prompt fino a 30000 token,
taglia 32 token di coda e allinea a chunk da 2048 token. Le knob
importanti sono:

I continued save usano lo stesso allineamento e vengono scritti solo
quando il grafo vivo raggiunge naturalmente una frontiera assoluta. Coi
default questo significa circa ogni 10k token, indipendentemente da dove
è atterrato il primo cold checkpoint, così le generazioni lunghe lasciano
indietro punti di restart senza persistere gli ultimi fragili token.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

Di default i checkpoint possono essere riusati tra le varianti routed-expert
a 2 e 4 bit se il prefix renderizzato matcha. Usa
`--kv-cache-reject-different-quant` quando vuoi solo strict reuse stesso
quant.

La directory di cache è disposable. Se il comportamento sembra sospetto,
ferma il server e rimuovila. Puoi investigare cosa c'è in cache con
hexdump dato che i file della kv cache includono il prompt cachato
verbatim.

## Backend

Il backend grafo di default è Metal su macOS e CUDA nelle build CUDA:

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

Su Linux, il `make` plain stampa i target di build disponibili invece di
selezionare implicitamente un target CUDA. Usa `make cuda-spark` per DGX
Spark / GB10. Omette un `nvcc -arch` esplicito perché al momento è il path
più veloce su GB10. Usa `make cuda-generic` per una build CUDA locale
normale, o setta `CUDA_ARCH` esplicitamente quando fai cross-build o
quando ti serve un target noto:

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

C'è anche un path CPU di riferimento/debug:

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

Non trattare il path CPU come target di produzione. La CLI e `ds4-server`
supportano il backend CPU per uso di riferimento/debug e condividono lo
stesso formato di KV session e snapshot di Metal e CUDA, ma l'inferenza
normale dovrebbe usare Metal o CUDA.

## Port Windows (CUDA)

Il port Windows aggiunge un path di build nativo MSVC + CUDA per
`ds4-server`, `ds4-bench` e `ds4-eval`. Non è una build MSYS/MinGW né
WSL: i binari sono prodotti dalla toolchain di Visual Studio, si linkano
contro il CUDA Toolkit ufficiale e usano direttamente le API Win32. Linux
e macOS continuano a usare il Makefile esistente e restano invariati.

### Prerequisiti

- Windows 10 build 19041 (versione 2004) o più nuovo, oppure Windows 11.
- Visual Studio 2022 col workload "Desktop development with C++".
- CUDA Toolkit ≥ 12.8 (richiesto per `sm_120`, l'architettura Blackwell
  usata dalle schede RTX serie 50 e più nuove).
- CMake ≥ 3.24.

### Comando di build

```sh
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
```

I risultanti `ds4-server.exe`, `ds4-bench.exe` e `ds4-eval.exe` vivono
in `build\Release\` e accettano gli stessi flag dei binari POSIX. Il
download dei GGUF passa ancora per `download_model.sh` da una shell
POSIX (Git Bash funziona); il motore in sé carica qualsiasi GGUF
compatibile che caricherebbe la build Makefile.

### Come funziona il port

Il port è un sottile layer di astrazione della piattaforma invece di una
riscrittura chiamata-per-chiamata, così le ~200 chiamate
mutex/socket/file-IO in `ds4_server.c` sono rimaste intatte:

- `ds4_platform.h` è un passthrough su POSIX e su Windows tira dentro
  `winsock2.h` prima di `windows.h`. Definisce alias che permettono al
  resto del codebase di mantenere i nomi dei simboli POSIX: `pthread_*`,
  `mmap`/`munmap`, `poll`, `flock`, `clock_gettime`, `dprintf`, `pread`,
  `ftruncate`, `fseeko`, `opendir`/`readdir`, termios e `/tmp`.
- `ds4_platform_win.c` contiene le implementazioni Windows:
  `mmap` è `CreateFileMapping` + `MapViewOfFile` tracciato in un piccolo
  registry così `munmap` funziona alla maniera POSIX; pthread mappa a
  `_beginthreadex` + `SRWLOCK` + `CONDITION_VARIABLE`; `/dev/urandom`
  diventa `BCryptGenRandom`; la modalità raw della console in `ds4-eval`
  usa `GetConsoleMode`/`SetConsoleMode`; `clock_gettime` usa
  `QueryPerformanceCounter`; `opendir`/`readdir` sono uno shim
  `FindFirstFile`.
- Tutti gli edit Windows-specific in `ds4.c`, `ds4_cuda.cu`,
  `ds4_server.c` e `ds4_eval.c` sono gated da `#ifdef _WIN32`. Il path
  dell'instance lock passa a `%TEMP%\ds4.lock`; il save/load dello
  snapshot di sessione usa `tmpfile()` perché Windows non ha
  `fmemopen`; il `__attribute__((unused))` del backend CUDA è
  sostituito da una macro `_MSC_VER`-aware; `ds4-server` chiama
  `WSAStartup` e `SetConsoleCtrlHandler` in `main` e instrada
  socket close + sync di errno attraverso piccoli helper.

Il path POSIX Makefile su Linux e macOS è invariato byte-per-byte
(verificato con full rebuild). Se una modifica futura Windows-specific
vorrà alterare codice condiviso, dovrà mantenere questa invariante.

### Cosa è incluso al momento

- `ds4-server.exe` — server HTTP completo con gli stessi endpoint
  OpenAI, Anthropic e Responses, lo stesso formato di disk KV cache e
  la stessa mappa di exact-DSML tool replay.
- `ds4-bench.exe` — benchmarking di velocità a frontiere di contesto.
- `ds4-eval.exe` — valutatore di capacità TUI split-screen.

### Cosa è rinviato

- La CLI interattiva `ds4` non è ancora buildata su Windows: dipende da
  `linenoise`, che usa termios POSIX per il line editing. Serve un port
  di `linenoise` alle Console API prima che il REPL possa essere
  spedito su Windows.
- È rinviata anche la regression `ds4_test --server` basata su
  socketpair; le altre test suite buildano e girano.

### Stato

Il port Windows è in alpha, come il resto del progetto. Il graph CUDA,
la KV cache e il formato della disk cache sono lo stesso codice del path
Linux CUDA, quindi la parità di comportamento dovrebbe seguire dal path
Linux CUDA. Per favore segnala issue con un log `--trace` allegato,
come su POSIX.

## Steering

Questo progetto supporta lo steering con direzioni di attivazione a
singolo vettore; vedi la directory `dir-steering` per maggiori
informazioni. Questo segue l'idea base del paper
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
Puoi usarlo per rendere il modello più o meno verboso, meno propenso a
rispondere a domande di programmazione se è un chatbot per il sito di
noleggio auto, e così via, molto più velocemente di un fine-tuning.
È utile anche per ricercatori di cybersecurity che vogliono ridurre la
disponibilità del modello a fornire indicazioni di sicurezza offensiva o
dual-use.

## Test Vector

`tests/test-vectors` contiene vettori di continuation a contesto corto e
lungo catturati dall'API ufficiale di DeepSeek V4 Flash. Le request usano
`deepseek-v4-flash`, decoding greedy, thinking disabilitato e la slice
massima di `top_logprobs` esposta dall'API. I vettori locali vengono
generati con `./ds4 --dump-logprobs` e confrontati per byte di token,
così le regressioni di tokenizer/template o di attention emergono prima
di diventare failure di generazione lunga.

Tutti i test del progetto sono guidati dal runner C:

```sh
make test                  # ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

## Note di debug

Quando una generazione sembra sbagliata, tre piccoli tool di solito
bastano per ottenere una prima risposta:

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` tokenizza la stringa `-p` o `--prompt-file` esattamente
  come scritta, riconosce gli special del protocollo DS4 e poi esce
  prima che l'inferenza parta. Ad esempio, il marker di chiusura tool
  DSML inizia come due token: `</` e `｜DSML｜`.
- `--dump-logprobs` salva una continuation greedy con le alternative
  locali top a ogni step, che aiuta a separare scelte di sampling da
  problemi di logit/modello.
- `ds4-server --trace` scrive i prompt renderizzati, le decisioni di
  cache, il testo generato e gli eventi del tool parser per un'intera
  sessione di agente.

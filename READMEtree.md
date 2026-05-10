# DeepSeek 4 - KV Cache 'Tree' System

Questo documento descrive le modifiche apportate al sistema di gestione della KV cache per implementare un albero di flusso ottimizzato.

## Obiettivo Didattico
Il progetto implementa un sistema di **Longest Prefix Match** per la KV cache caricata da SSD, utilizzando una struttura dati a **Radix Tree** (o Patricia Trie). Questo permette di identificare istantaneamente il punto più avanzato della conversazione che può essere riutilizzato, minimizzando i tempi di inferenza e il calcolo ridondante.

## Modifiche Principali

### 1. Struttura Dati 'Tree'
- È stata integrata la libreria `rax` (una Radix Tree implementazione di Antirez) all'interno di `kv_disk_cache`.
- Ogni snapshot salvato su disco viene indicizzato nell'albero utilizzando la sequenza di token come chiave.
- Questo approccio è concettualmente simile a quello usato in database come **RocksDB** per l'indicizzazione delle chiavi, ma ottimizzato per la memoria RAM del server DeepSeek.

### 2. Formato File KV v2
- Il formato dei file `.kv` è stato aggiornato alla versione `2` per includere gli ID dei token originali.
- Senza gli ID dei token salvati, non sarebbe possibile ricostruire l'albero al riavvio del server senza ricalcolare tutti gli SHA1.

### 3. Algoritmo di Ricerca
- La ricerca del prefisso (`kv_cache_find_prefix`) non scorre più linearmente tutti i file (O(N)), ma interroga l'albero.
- Se il contesto cambia, il sistema identifica il punto di divergenza e carica solo la parte comune della cache.

## Note per lo Studio
Il sistema è predisposto per essere esteso con motori di archiviazione persistente più complessi:
- **Patricia Trie**: L'attuale implementazione `rax` è già un Patricia Trie in memoria.
- **RocksDB/LMDB**: Per una persistenza su larga scala (milioni di snapshot), si potrebbe sostituire l'albero in memoria con un database Key-Value su disco, usando la sequenza di token serializzata come chiave.

## File Modificati
- `ds4_server.c`: Logica di gestione albero, salvataggio token ID e ricerca ottimizzata.
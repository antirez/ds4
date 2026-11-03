# Fixture residente per il confronto numerico con main

Il modello Qwen di produzione non ammette una semplice GGUF a quattro layer. Il validatore seleziona la forma E=2560 e richiede 48 layer trunk senza MTP (`ds4.c:7116–7140`). La forma MINI richiede E=64 e non esercita gli split-K delle proiezioni SSM reali.

Il generatore `make-resident-fixture.py` mantiene quindi tutti i **48 layer** e le forme reali: E=2560, HC=10240, head_dim=256, 512 esperti e intermedi di 640 elementi. I nomi di ciascun layer rimangono distinti, ma il payload della matrice `blk.n.<nome>` punta alla corrispondente matrice originale `blk.(n%4).<nome>`. I quattro blocchi prototipo conservano la sequenza tre GDN più un'attenzione completa. Lo stato ricorrente e le cache di ogni layer rimangono distinti nel runtime.

Tokenizer, embedding, output e matrici di controllo vengono copiati senza cambiare i byte. Il predictor viene rimosso (`nextn_predict_layers=0`). La tabella PLE n-gram è sostituita con una riga BF16 zero di larghezza 160; i 16 head hanno offset 0 e vocabolario 1. La formula del runtime è `(ngram_size - 1) * heads_per_ngram`: unigram non ha una propria tabella. Questa modifica rende la fixture piccola e preserva la geometria delle operazioni PLE, ma non conserva il comportamento del modello originale.

La fixture è dunque **sintetica, con matrici reali ripetute**, utile per confrontare il normale percorso residente dei due binari sullo stesso file. Non è un modello ridotto per generazione di qualità e non è una prova di parità del Q2 completo su M4.

## Pianificazione verificata

Il dry run sul Q4 locale ha prodotto `resident-fixture-plan.json`:

- 1223 nomi di tensori con payload originale; una tabella PLE aggiuntiva.
- 112 payload originali distinti, copiati una sola volta.
- 7.4075 GiB di mapping residente; file finale di 7,953,727,808 byte.
- Sintassi Python e struttura degli alias verificate prima della scrittura.

`parse_tensors` di ds4 controlla dimensioni e limiti di ogni range, ma non vieta che più tensori condividano lo stesso offset (`ds4.c:2797–2850`). Il caricamento residente usa la dimensione del mapping, pari al file compatto; gli alias non creano copie nel file. Alcuni lettori GGUF esterni vietano gli overlap e non possono aprire questa fixture. Il validatore sperimentale deve essere quello di ds4 dei due commit.

L'allineamento della fixture è almeno 16 KiB. La tabella PLE rimane l'ultimo tensore, allineato a pagina, affinché il normale runtime la rimuova dal mapping residente e la legga dal file. Il generatore rifiuta nomi inattesi, forme diverse fra alias e prototipo, file sorgente modificato e output già esistente. Legge/copia blocchi di massimo 16 MiB, registra SHA-256 per ogni payload e pubblica l'output senza sovrascrivere un file esistente.

## Esecuzione da approvare nel processo principale

Il processo principale ha approvato ed eseguito la scrittura. Il comando seguente **scrive circa 8 GB** e serve per ricostruire la fixture in un nuovo percorso:

```sh
python3 misc/qwen-drift-fix-2026-10-03/make-resident-fixture.py \
  gguf/Qwen3.8-Flash-Next-Q4.gguf \
  misc/qwen-drift-fix-2026-10-03/resident-fixture.gguf \
  --manifest misc/qwen-drift-fix-2026-10-03/resident-fixture-written.json \
  --write
```

Prima di inferenza GPU, verificare che entrambi i binari accettino l'intestazione con il loro normale comando `--inspect`, che tutti i range rientrino nei 7.41 GiB e che i payload copiati corrispondano agli hash del manifest. Eseguire i processi in sequenza; non avviare inferenza CPU.

Il primo load ha trovato un errore del generatore: gli array PLE inizialmente contenevano 24 elementi, mentre il runtime ne richiede 16. Il generatore ora deriva e valida questo conteggio dal metadata sorgente. La correzione restringe l'header ma conserva il padding: `output_data_start=11026432`, tutti gli offset dei payload e la dimensione finale rimangono identici. Il processo principale può correggere soltanto l'header con `--repair-header`, usando il manifest scritto originale. Questa modalità verifica il precedente SHA dell'header e l'invarianza del layout, riscrive circa 11 MB e preserva gli hash dei 112 payload. Non richiede una seconda copia degli 8 GB. Il load corretto resta da verificare.

Per il confronto numerico usare gli stessi token fixture, chunk e continuazioni teacher-forced per il main congelato e il codice corretto. Confrontare l'intero vocabolario, non soltanto il testo generato. Coprire T=29, T=575 e un prompt che attraversi la soglia sparse a 2051 con chunk 2048; il prompt di 5942 token è utile ma più lungo. La dimensione del vocabolario e tutti i kernel principali rimangono quelli reali. Riportare il risultato come confronto di **fixture residente Q4 sintetica su M1**, senza estenderlo automaticamente al Q2 residente M4 del tester.

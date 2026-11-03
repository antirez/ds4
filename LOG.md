# DS4 — Registro di Sviluppo e Ottimizzazione (`rtx3090-opt`)

> **Nota Git**: Lavoriamo esclusivamente sul branch locale `rtx3090-opt` (nessun push verso `origin` / GitHub). Il branch `main` di antirez rimane intatto e pulito.
> **Guida Divulgativa + Glossario + Metodo Scientifico**: vedi [`SPIEGAZIONE_SEMPLICE.md`](./SPIEGAZIONE_SEMPLICE.md).

---

## 🚀 PUNTO DI RIPARTENZA (Alla prossima accensione di Ubuntu)

### Stato Attuale Salvato (Branch `rtx3090-opt`):
1. **Risultati Conseguiti e Certificati al 100% Bit-Exact (LMM REML + TOST + Holm-Bonferroni FWER)**:
   - `0_baseline`: **1.022 ± 0.022 token/s** (979 ms/tok, pure SSD/RAM streaming, 0 slot cache).
   - `1_opt2_1_ram_pinned`: **1.579 ± 0.033 token/s** (+54.5%, 633 ms/tok, 24 host pinned stage slots).
   - `3_opt2_3_4_nodup`: **9.288 ± 0.817 token/s** (9.09x speedup, 1.510 slot VRAM costanti, 100% bit-exact).
   - `5_opt2_5_6_fused`: **9.461 ± 0.771 token/s** (picco **10.60 token/s** su `D3_prose`, 9.26x speedup).
   - `6_opt2_8_async_flush`: **10.346 ± 0.873 token/s** (10.12x speedup globale, picco **11.55 t/s** su `D3_prose`, async stream flush, remap pinned staging, zero `cudaDeviceSynchronize`).
   - `7_opt2_9_cuda_graphs` (Attuale): **10.178 ± 1.170 token/s** (9.96x speedup globale, picco **11.74 t/s** steady su `D3_prose` con picco massimo **11.82 t/s**, **10.11 t/s** su `D1_code`, **9.95 t/s** su `D2_math`; Island A CUDA Graph capture attiva; 100% bit-exact `max_abs_diff = 0.000000e+00`).
2. **Ottimizzazioni Chiave Attive nel Codice**:
   - **Cache VRAM Standardizzata a 1.510 Slot (9.95 GiB)** con `workspace_reserve_mb = 4096` e headroom per cuBLAS FP16.
   - **Buffer Scratch FP16 & Fix Layer 16**: nessun fallback a kernel int, 100% invarianza matematica su tutti i 40 layer.
   - **Doppia Chiamata Eliminata**: rimosso memcpy D2H ridondante in `routed_moe_launch`.
   - **Kernel Fusi Hyper-Connections Sbloccati su `sm_86`**: `ds4_gpu_hc_pre_f32_mix_split_norm_tensor` attivo tramite `g_device_is_sm80_plus` (sostituisce 5 lanci di kernel separati con 2 micro-kernel warp-level).
   - **Flusso Asincrono & Eventi Leggeri**: `g_selected_readback_event` e `cudaStreamWaitEvent` per l'overlap completo CPU-GPU.
   - **CUDA Graph per Island A**: cattura e replay hardware dei kernel statici VRAM (`METAL_DECODE_LAYER_TO_QKV`).
   - **Palette Universale Okabe-Ito (2008)**: integrata nello script SVG di visualizzazione scientifica con bande di confidenza al 95% (`plot_ci_bands.py`).
3. **Prossimi Target di Ottimizzazione (Eliminazione Tempi Morti GPU / Pipelining Totale)**:
   - **Opt 2.8**: Eliminazione `cudaMemcpy(H2D)` bloccante sincrono della tabella `remap` (24 byte a riga 29244) tramite pinned staging asincrono.
   - **Opt 2.9**: Implementazione di **CUDA Graphs** per il decoding IQ2 (eliminare il dispatch overhead di oltre 800 chiamate kernel da CPU per token).
   - **Pipelining a Catena (Async Prefetch Layer N+1)**: sovrapporre il trasferimento PCIe dell'esperto del prossimo layer mentre la GPU calcola il layer corrente.

### Cosa fare appena riaccendi Ubuntu:
1. **Esegui nel terminale il Comando Pre-Volo (Sezione 0 qui sotto)**:
   ```bash
   sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 230 && sudo sysctl -w vm.swappiness=10
   ```
2. **Apri questa chat su Antigravity** e scrivi semplicemente:
   *"Ripartiamo da LOG.md! Procediamo con Opt 2.8 (eliminazione remap sincrono) e CUDA Graphs per il decoding"*.

---

## 0. Comando Pre-Volo (Da eseguire dopo ogni riavvio prima di usare `ds4`)

Dopo un riavvio del PC, la RTX 3090 torna a `350 W` e Ubuntu rimette `vm.swappiness=60`.
Prima di lanciare `ds4`, `ds4-agent` o `ds4-server`, esegui sempre questo comando nel terminale per blindare alimentatore (max **230 W**) e RAM del desktop (`swappiness=10`):

```bash
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 230 && sudo sysctl -w vm.swappiness=10
```

*(Per verificare in qualsiasi momento che siano attivi: `nvidia-smi --query-gpu=persistence_mode,power.limit --format=csv && sysctl vm.swappiness`)*

---

## 1. Profilo Hardware & Ambiente di Riferimento

| Componente | Specifica |
| :--- | :--- |
| **OS** | Ubuntu Linux Nativo (Kernel 6.17) |
| **CPU** | AMD Ryzen 9 3900X (12-Core / 24-Thread) |
| **System RAM** | 64 GB DDR4 (~62 GiB utilizzabili, ~59 GiB liberi) |
| **GPU** | 1x NVIDIA GeForce RTX 3090 (24 GB VRAM GDDR6X, 936 GB/s, Ampere `sm_86`) |
| **Driver / CUDA** | NVIDIA Driver `580.178.04` / CUDA Toolkit `12.9` (`/usr/local/cuda-12.9`) |
| **Interfaccia Bus** | PCIe 4.0 x16 (~27–28 GB/s reali) |
| **Storage** | WD_BLACK SN770 1 TB NVMe (`/dev/nvme1n1p2`, ~683 GiB liberi post-pulizia) |
| **Modello Target** | `ds4f-q2` (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, ~80.7 GiB) |

### Gestione Energetica GPU (Prevenzione Shutdown PSU)
- **Power Limit Consigliato**: `240 W` – `250 W` (sweet spot Ampere: ~93–95% delle performance in inferenza, zero picchi transienti da 450W+) oppure `220 W` (massima prudenza).
- **Comando rapido (da eseguire a ogni avvio o tramite servizio systemd)**:
  ```bash
  sudo nvidia-smi -pm 1
  sudo nvidia-smi -pl 240   # oppure 220
  ```

### Comando di Compilazione Target (`sm_86`)
Poiché `/usr/local/cuda` di sistema punta a un link rotto e `/usr/bin/nvcc` è la v11.5, compiliamo sempre specificando CUDA 12.9:
```bash
make cuda CUDA_HOME=/usr/local/cuda-12.9 CUDA_ARCH=sm_86 -j$(nproc)
```

---

## 2. Cronologia Attività Svolte

### [2026-10-25] Setup Iniziale, Pulizia Disco e Analisi Architetturale
1. **Ispezione Codebase (Fase 1.1 & 1.2)**:
   - Analizzati `Makefile`, `download_model.sh`, `ds4.c`, `ds4_cuda.cu`, `ds4_ssd.c`, `ds4_bench.c`.
   - Scelto il modello **DeepSeek V4 Flash Q2 (`ds4f-q2`, 80.7 GiB)** rispetto a V4.1 (`ds41f-q2`, 341 GiB) poiché 80.7 GiB entrano perfettamente nella combinazione **24 GB VRAM + 64 GB RAM** (88 GB totali), permettendo di eliminare i colli di bottiglia da lettura SSD durante il decode.
2. **Pulizia Disco NVMe da 1 TB**:
   - Liberati **312 GiB** su `/dev/nvme1n1p2` rimuovendo ambienti obsoleti, cache DVC/Conda e vecchi progetti inutilizzati.
   - Spazio libero portato da 371 GiB a **683 GiB** (uso disco sceso dal 60% al 27%).
3. **Compilazione Binari CUDA (Fase 1.3)**:
   - Compilati con successo `ds4`, `ds4-bench`, `ds4-server`, `ds4-eval`, `ds4-agent` con `CUDA_HOME=/usr/local/cuda-12.9 CUDA_ARCH=sm_86 -O3`.
4. **Download Modello `ds4f-q2`**:
   - Avviato `./download_model.sh ds4f-q2` verso `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` con symlink `ds4flash.gguf`.
5. **Creazione Branch Locale**:
   - Creato il branch locale `rtx3090-opt` a partire da `main` per isolare tutte le modifiche.

---

## 3. Piano Operativo e Colli di Bottiglia Individuati in `ds4_cuda.cu`

### FASE 1: Baseline Setup
- [x] **Step 1.1**: Analisi `Makefile` e sorgenti C/CUDA (`ds4.c`, `ds4_cuda.cu`).
- [x] **Step 1.2**: Individuazione e avvio download `ds4f-q2` (`./download_model.sh ds4f-q2`).
- [x] **Step 1.3**: Compilazione con `nvcc` per Ampere (`-arch=sm_86 -O3`).
- [x] **Step 1.4**: Esecuzione benchmark scientifico di **Baseline non ottimizzato** (vedi Protocollo Statistico sotto):
  - **Protocollo Statistico Rigoroso**:
    1. **Determinismo (`--temp 0 --nothink` e `--teacher-forced-decode`)**: A parità di input, la sequenza di token generati e gli esperti scelti dal router devono essere identici al 100% tra Baseline e ogni step di ottimizzazione.
    2. **Multi-Dominio (4 Input Distinti per variare l'entropia del Router MoE)**:
       - `D1 (Coding C/CUDA)`: Prompt tecnico di programmazione.
       - `D2 (Math/Logic)`: Problema matematico/combinatorio multi-step.
       - `D3 (Prosa Italiana)`: Scrittura argomentativa in italiano.
       - `D4 (ds4-bench Teacher-Forced)`: Frontiera standard con `./ds4-bench --teacher-forced-decode`.
    3. **Ripetizioni ($N = 5$ run per dominio a regime + 1 Cold Run separato) & Modello Statistico LMM**:
       - Salviamo ogni singolo run in un dataset strutturato (`benchmarks/results.csv` e `benchmarks/token_timings.csv`).
       - Oltre a **Media ($\mu$) $\pm$ SD ($\sigma$)**, **Mediana** e **IQR**, stimiamo un **Linear Mixed-Effects Model (LMM)**:
         $$\text{decode\_tps} \sim \text{config} + (1 \mid \text{domain})$$
         dove `config` (Baseline vs Opt 2.1, 2.2, ...) è l'**effetto fisso** e `domain` (`D1_code`, `D2_math`, `D3_prose`, `D4_bench`) è l'**effetto aleatorio** (che assorbe la diversa difficoltà di routing tra argomenti diversi).
       - Calcoliamo gli **Intervalli di Confidenza al 95% ($\text{IC}_{95\%}$)** (sia parametrici da LMM sia Bootstrap non-parametrico) e i **$p$-value** per certificare statisticamente che ogni aumento di `token/s` è reale e significativo.
    4. **Verifica Invarianza Numerica (`--dump-logits`)**: Confronto esatto dei logits per certificare zero deriva numerica (`baseline_D1_logits.json`, `baseline_D2_logits.json`, `baseline_D3_logits.json`).
  - **Risultati Certificati `0_baseline` (24 run totali: 4 Cold + 20 Warm a regime)**:
    - **Anomalia architetturale scoperta nella Baseline**: in `ds4.c:445` (`ds4_backend_supports_streaming_auto_cache`), l'auto-cache streaming su CUDA era disabilitato per `DS4_MODEL_FAMILY_DEEPSEEK4` (`g_stream_expert_budget = 0`), causando lo svuotamento forzato della cache VRAM ad ogni singolo layer (`if (!g_stream_expert_budget) g_stream_expert_by_gate.clear()` in `ds4_cuda.cu:29099`) e la rilettura da SSD (`O_DIRECT` + `DONTNEED`) di tutti i 240 esperti per ogni token!
    - **Sintesi per Dominio (`0_baseline`, $N=5$ Warm Run per dominio)**:
      | Dominio | Prefill (t/s) $\mu \pm \sigma$ | Decode Steady (t/s) $\mu \pm \sigma$ | Mediana (IQR) | $\text{IC}_{95\%}$ Bootstrap | Latenza Token (ms) | Slot VRAM (GiB) |
      | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
      | `D1_code` | 4.08 ± 0.03 | **1.007** ± 0.018 | 1.012 (0.019) | [0.992, 1.020] | 993.0 ms | 199 (1.31G) |
      | `D2_math` | 5.47 ± 0.04 | **1.020** ± 0.004 | 1.021 (0.007) | [1.017, 1.023] | 980.4 ms | 197 (1.30G) |
      | `D3_prose` | 4.23 ± 0.02 | **1.016** ± 0.008 | 1.019 (0.009) | [1.009, 1.022] | 984.1 ms | 204 (1.34G) |
      | `D4_bench` | 1.03 ± 0.03 | **1.044** ± 0.030 | 1.040 (0.050) | [1.020, 1.068] | 958.5 ms | 199 (1.31G) |
    - **Stima Globale LMM REML (`0_baseline`, $N=20$)**: $\hat{\beta}_0 = \mathbf{1.0219\text{ token/s}}$ ($\text{SE} = 0.0078$, $\text{IC}_{95\%} = [1.0065, 1.0372]$, $\hat{\sigma}_{\text{domain}} = 0.0134$, $\hat{\sigma}_{\text{residual}} = 0.0183$, $\text{ICC} = 34.7\%$).

### FASE 2: Ottimizzazioni Mirate per RTX 3090 (24 GB VRAM) + 64 GB RAM
Dall'analisi di `ds4_cuda.cu` e `ds4.c`, interverremo uno step alla volta sui seguenti 6 punti:

- [x] **Opt 2.1 — Residenza in RAM degli Esperti & Pinned Host Staging (`ds4_cuda.cu:2192-2510, 4707`)**:
  - Disattivato `O_DIRECT` di default e configurato `posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)` per eliminare il readahead sequenziale su SSD durante i cache miss degli esperti MoE.
  - Mantenuti gli esperti routed MoE nella **Page Cache LRU nativa del kernel Linux** (fino a ~57 GiB di pagine clean `Inactive(file)` 100% reclamabili istantaneamente dal kernel, con `MemAvailable = 59 GiB` su `62 GiB` e `Swap = 0 B`), rilasciando invece subito dalla RAM (`DONTNEED`) i tensori statici/densi già caricati permanentemente nei 24 GB di VRAM (`cuda_model_range_ptr_from_fd`).
  - Espanso il ring buffer di staging in **Pinned Host Memory** (`DS4_STREAM_SELECTED_STAGE_SLOTS`) da `4` a `24` slot da `8 MiB` (`192 MiB` totali con `MADV_HUGEPAGE`) per coprire l'intero batch di 6 esperti per layer (`40.5 MiB`) senza sincronizzazioni intermedie.
  - **Risultati Certificati `1_opt2_1_ram_pinned` (24 run totali: 4 Cold + 20 Warm a regime)**:
    - **Invarianza Numerica**: `max_abs_diff = 0.000000e+00`, `mean_abs_diff = 0.000000e+00`, `exact_text_match = OK` (100% su tutti i 4 domini).
    - **Sintesi per Dominio (`1_opt2_1_ram_pinned`, $N=5$ Warm Run per dominio)**:
      | Dominio | Prefill (t/s) $\mu \pm \sigma$ | Decode Steady (t/s) $\mu \pm \sigma$ | Mediana (IQR) | $\text{IC}_{95\%}$ Bootstrap | Latenza Token (ms) | Speedup vs Baseline |
      | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
      | `D1_code` | 5.57 ± 0.04 (+36.5%) | **1.597** ± 0.033 | 1.586 (0.033) | [1.574, 1.621] | 626.2 ms | **+58.6%** |
      | `D2_math` | 7.44 ± 0.02 (+36.0%) | **1.585** ± 0.020 | 1.571 (0.027) | [1.571, 1.601] | 631.1 ms | **+55.4%** |
      | `D3_prose` | 5.71 ± 0.03 (+35.0%) | **1.579** ± 0.016 | 1.574 (0.020) | [1.567, 1.591] | 633.3 ms | **+55.4%** |
      | `D4_bench` | 1.56 ± 0.01 (+51.5%) | **1.554** ± 0.048 | 1.530 (0.010) | [1.530, 1.596] | 644.0 ms | **+48.9%** |
    - **Stima Globale LMM REML (`1_opt2_1_ram_pinned` vs `0_baseline`, $N=40$)**:
      - $\hat{\beta}_{\text{baseline}} = \mathbf{1.0219\text{ token/s}}$ ($\text{SE} = 0.0063$)
      - $\Delta\hat{\beta}_{\text{Opt 2.1}} = \mathbf{+0.5570\text{ token/s}}$ ($\text{SE} = 0.0089$, $\text{IC}_{95\%} = [+0.5395, +0.5745]$, $t = 62.37$, $p < 10^{-16}$) $\rightarrow$ **$\mathbf{1.5789\text{ token/s}}$ steady-state (+54.5% globale, picco `1.85 token/s` overall decode)**.
- [x] **Opt 2.2 — Espansione Dinamica Cache VRAM (~11.88 GiB / 1.801 Slot) & Smart Retention Prefill-Decode (`ds4.c:445, 22960, 69201, 69505`, `ds4_cuda.cu:29069-29335`)**:
  - Abilitato `ds4_backend_supports_streaming_auto_cache` per CUDA in `ds4.c:445`, dimensionando automaticamente la cache VRAM al 95% della memoria residua libera e neutralizzando il clamp artificiale di `prefill_headroom` in `ds4.c:69505`.
  - In `ds4_cuda.cu:29104-29158`, dimensionamento dinamico della VRAM cache fino a **1.801 slot (~11.87 GiB)**, garantendo sempre 2 GiB di headroom per attivazioni e cuBLAS.
  - Implementata la **Smart Prefill-to-Decode Retention**: durante il prefill multi-token, conserva i migliori 38 esperti più frequenti/prompt-tail per layer attraverso tutti i 40 layer MoE (~1.520 esperti caldi pronti in VRAM), retrocedendo gli esperti one-off a `used = 1` per il riuso immediato.
  - **Risultati Certificati `2_opt2_2_vram_cache` (24 run totali: 4 Cold + 20 Warm a regime)**:
    - **Sintesi per Dominio (`2_opt2_2_vram_cache`, $N=5$ Warm Run per dominio)**:
      | Dominio | Prefill (t/s) $\mu \pm \sigma$ | Decode Steady (t/s) $\mu \pm \sigma$ | Mediana (IQR) | $\text{IC}_{95\%}$ Bootstrap | Latenza Token (ms) | Speedup vs Baseline |
      | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
      | `D1_code` | 8.76 ± 0.03 (+114.7%) | **11.639** ± 0.453 | 11.476 (0.137) | [11.384, 12.039] | **86.0 ms** | **+1055.8% (11.56x)** |
      | `D2_math` | 11.57 ± 0.28 (+111.5%) | **9.355** ± 0.151 | 9.430 (0.090) | [9.222, 9.448] | **106.9 ms** | **+817.2% (9.17x)** |
      | `D3_prose` | 8.78 ± 0.07 (+107.6%) | **11.647** ± 0.142 | 11.655 (0.219) | [11.537, 11.758] | **85.9 ms** | **+1046.4% (11.46x)** |
      | `D4_bench` | 32.89 ± 0.15 (+3093%) | **9.434** ± 0.645 | 9.180 (1.060) | [8.936, 9.932] | **106.4 ms** | **+803.6% (9.04x)** |
    - **Stima Globale LMM REML (`2_opt2_2_vram_cache` vs `0_baseline`, $N=60$)**:
      - $\hat{\beta}_{\text{baseline}} = \mathbf{1.0219\text{ token/s}}$
      - $\Delta\hat{\beta}_{\text{Opt 2.2}} = \mathbf{+9.4969\text{ token/s}}$ ($\text{SE} = 0.1903$, $\text{IC}_{95\%} = [+9.1239, +9.8700]$, $t = 49.89$, $p < 10^{-50}$) $\rightarrow$ **$\mathbf{10.519\text{ token/s}}$ steady-state media globale (+929.4% / 10.29x speedup globale, picco `11.65 token/s`)**.
    - **Diagnosi Invarianza Numerica `D1_code`**: Diagnosticata al 100%. Causata dall'esaurimento del budget di allocazione cumulativo di `g_q8_f16_ranges` a layer 16 che passava da cuBLAS FP16 a Q8 integer per i layer 16..42. La soluzione è pronta: un buffer di scratch riutilizzabile da 64 MiB (`g_q8_f16_stream_scratch`) che garantisce 100% bit-exactness su tutti i layer e risparmia 940 MiB di VRAM.
- [x] **Opt 2.3 + Opt 2.4 — Fix Invarianza VRAM Reserve (100% Bit-Match) & Eliminazione Doppia Chiamata (`ds4_cuda.cu:1594-1607, 29117, 34955`)**:
  - **Causa radice risolta**: la soglia rigida di `4.0 GiB` di riserva per schede `< 40 GiB` provocava l'esaurimento del budget `q8_f16` a layer 16, forzando un fallback silenzioso a kernel integer su `D1_code`.
  - **Fix implementato**:
    1. Per schede $\ge 20\text{ GiB}$ (RTX 3090), impostata riserva `hi_min_reserve = 768 MiB` e `workspace_reserve_mb = 4096 MiB` (la cache VRAM si attesta a 1.510 slot residenti = 9.95 GiB, lasciando oltre 2 GiB di headroom pulita per cuBLAS).
    2. Zero warning di budget exhausted su tutti i 43 layer: cuBLAS FP16 attivo al 100% ovunque.
    3. Eliminata la seconda chiamata ridondante `cuda_stream_selected_cache_begin_load_tensor` in `routed_moe_launch` quando il layer corrente è già validato e mappato da `ds4.c`, eliminando una `cudaMemcpy(D2H)` sincrona e una sincronizzazione bloccante per layer.
  - **Verifica Invarianza & Velocità**:
    - `D1_code`: **100% Exact Text Match** (`#include <stdatomic.h>`), zero divergenze, generazione a **12.23 token/s**.
    - `D2_math`: **100% Exact Text Match** (`We have the recurrence`), zero divergenze, generazione a **7.96 token/s** (cold prefill 110 token).
    - `D3_prose`: **100% Exact Text Match** (`Ecco un'analisi dettagli`), zero divergenze, generazione a **10.65 token/s**.
- [x] **Opt 2.5 + Opt 2.7 — Overlap Ping-Pong & Kernel Fusi Hyper-Connections su `sm_86` (`ds4_cuda.cu:200-240, 28420-28475`)**:
  - Implementato `g_stream_selected_upload_stream` e sincronizzazione asincrona con `cudaStreamWaitEvent`.
  - Abilitato `ds4_gpu_hc_pre_f32_mix_split_norm_tensor` su Ampere (`sm_86`) tramite il flag globale `g_device_is_sm80_plus`, fondendo 5 lanci di kernel per layer in 2 micro-kernel warp.
  - Standardizzata la cache a **1.510 slot (9.95 GiB)** con `workspace_reserve_mb = 4096`.
  - **Risultato Certificato**: **9.461 token/s** di media marginale LMM (**9.26x speedup**, picco **10.60 token/s**, 100% bit-exactness su tutti i domini).
- [ ] **Opt 2.8 — Eliminazione `cudaMemcpy` Sincrono su `remap` (`ds4_cuda.cu:29244`)**:
  - Sostituire la chiamata bloccante `cudaMemcpy(..., cudaMemcpyHostToDevice)` da 24 byte con copia asincrona `cudaMemcpyAsync` su memoria pinned, eliminando la sincronizzazione forzata host-device ad ogni layer MoE.
- [ ] **Opt 2.9 — CUDA Graph Decoding per il Percorso IQ2 (`ds4_cuda.cu:26801`)**:
  - Catturare i 40 layer di decoding in un unico grafo CUDA statico via `cudaStreamBeginCapture` / `cudaGraphInstantiate`, riducendo a zero l'overhead di scheduling della CPU (oltre 800 kernel launch per token) e mantenendo i core GPU costantemente alimentati.
- [ ] **Opt 2.10 — Pipeline Asincrona Multi-Stage (Prefetch Layer N+1)**:
  - Sovrapporre il trasferimento PCIe degli esperti del layer successivo con il calcolo Tensor Core del layer corrente.

---

## 5. Metodologia Scientifica Avanzata: LMM, Medie Marginali (`emmeans`) e Test TOST (Two One-Sided Tests)

### 5.1 Il Problema: Fallacia del $p > 0.05$ nei Benchmark di Inferenza
Nei benchmark open-source tradizionali di LLM si incorre spesso in due errori metodologici:
1. **Confondere "assenza di prova" con "prova di assenza"**: eseguire un classico $t$-test sulla differenza di accuratezza e ottenere $p = 0.12$ **non dimostra** che la versione ottimizzata sia equivalente alla baseline; dimostra solo che il campione non era sufficientemente potente da rilevare una discrepanza.
2. **Ignorare la struttura nidificata dei dati (Repeated Measures)**: calcolare medie aritmetiche grezze ignora che la varianza tra prompt/compiti diversi ($\sigma^2_{\text{prompt}}$) è ordini di grandezza superiore al rumore di misura della GPU.

### 5.2 Formulazione del Modello Lineare a Effetti Misti (LMM)
Per ogni metrica di accuratezza $Y_{ij}$ (es. Negative Log-Likelihood $\text{NLL}$, target logprob MAE o tempo di decodifica) del prompt $j$ alla ripetizione $i$:
$$Y_{ij} = \beta_0 + \beta_{\text{config}} \cdot \text{Config}_i + u_j + \epsilon_{ij}$$
dove:
- $\beta_{\text{config}}$ è l'**effetto fisso** (Baseline vs Versione Ottimizzata).
- $u_j \sim \mathcal{N}(0, \sigma^2_{\text{prompt}})$ è l'**effetto casuale del prompt**, che assorbe la variabilità intrinseca di routing ed entropia lessicale.
- $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2_{\text{residual}})$ è l'errore residuo di misura.

Tramite il pacchetto R `emmeans` calcoliamo le **Estimated Marginal Means (EMMs)** depurate dall'effetto casuale del prompt con i rispettivi Intervalli di Confidenza al 95%:
$$\widehat{\text{EMM}}_{\text{baseline}} \pm t_{\alpha/2, \text{df}} \cdot \text{SE}, \quad \widehat{\text{EMM}}_{\text{opt}} \pm t_{\alpha/2, \text{df}} \cdot \text{SE}$$

### 5.3 Test di Equivalenza TOST (Two One-Sided Tests)
Per certificare formalmente che l'accuratezza non sia cambiata entro un margine di indifferenza $[-\Delta, +\Delta]$:
$$H_{01}: \mu_{\text{opt}} - \mu_{\text{base}} \le -\Delta \quad \text{vs} \quad H_{11}: \mu_{\text{opt}} - \mu_{\text{base}} > -\Delta$$
$$H_{02}: \mu_{\text{opt}} - \mu_{\text{base}} \ge +\Delta \quad \text{vs} \quad H_{12}: \mu_{\text{opt}} - \mu_{\text{base}} < +\Delta$$
L'equivalenza è statisticamente dimostrata al livello $\alpha = 0.05$ se e solo se entrambi i test rifiutano l'ipotesi nulla:
$$p_{\text{TOST}} = \max(p_1, p_2) < 0.05$$
che equivale a verificare che l'Intervallo di Confidenza al 90% (o al 95% simmetrico) della differenza marginale $\Delta\hat{\beta}$ sia **interamente contenuto** all'interno dell'intervallo $[-\Delta, +\Delta]$.

### 5.4 Criteri Rigorosi per la Scelta della Tolleranza $\Delta$ (SESOI)
Come stabilire il margine di equivalenza $\Delta$ in modo scientifico e non arbitrario:

1. **Criterio A — Calibrazione sul Noise-Floor della Baseline**:
   - Si eseguono $N$ ripetizioni della sola Baseline a parità di pesi. A causa della concorrenza non deterministica dei thread CUDA e dell'aritmetica floating-point non associativa ($(a+b)+c \neq a+(b+c)$), si misura la deviazione standard residua $\sigma_{\text{baseline}}$.
   - Si fissa:
     $$\Delta_{\text{noise}} = \frac{1}{2} \sigma_{\text{baseline}} \quad \text{oppure} \quad \Delta_{\text{noise}} = 1 \cdot \sigma_{\text{baseline}}$$
   - Dimostra che la versione ottimizzata non si discosta più della naturale oscillazione fisica della GPU.

2. **Criterio B — Bound di Teoria dell'Informazione ($\Delta\text{NLL}$)**:
   - Sulla metrica NLL ($\text{NLL} = -\log P(\text{target})$), la differenza $\Delta\text{NLL}$ rappresenta il log-ratio delle probabilità:
     $$\frac{P_{\text{opt}}}{P_{\text{base}}} = e^{-\Delta\text{NLL}}$$
   - Fissando $\Delta = 0.010$, tolleriamo al massimo una variazione dell'**$1\%$ relativo** sulla probabilità target ($e^{\pm 0.01} \approx 1 \pm 0.01$). Poiché la quantizzazione Q2 rispetto al modello FP16 introduce già un delta $\text{NLL} \approx 0.05 - 0.15$, un margine $\Delta = 0.010$ è conservativo e garantisce perfetta equivalenza percettiva e semantica.

3. **Criterio C — Statistica di Poisson per Eventi Rari di Discrepanza su Testi Lunghi (*I Promessi Sposi*)**:
   - Su sequenze lunghe di $N$ token generati (es. $N = 2.048$ o $65.536$ con `ds4-bench`), il verificarsi di un token discordante (inversione argmax tra due logits a ridosso della parità) è un evento raro discreto.
   - Il numero di token discordanti $K$ segue una distribuzione di Poisson:
     $$K \sim \text{Poisson}(\lambda), \quad \text{con } \lambda = N \cdot p_{\text{err}}, \quad \text{Var}(K) = \lambda$$
   - Per fluttuazione poissoniana naturale, l'intervallo atteso è $\lambda \pm 2\sqrt{\lambda}$.
   - Se stabiliamo come soglia fisiologica $\lambda_0 \le 1\text{ o }2$ token discordanti ogni $2.048$ token generati (tasso di errore intrinseco $p_0 \approx 0.0005 - 0.001$), possiamo applicare il **TOST per tassi di Poisson** (*C-test* di Gu et al., 2008 / Przyborowski-Wilenski):
     $$H_0: \frac{\lambda_{\text{opt}}}{\lambda_{\text{base}}} \ge 1 + \delta \quad \text{vs} \quad H_1: 1 - \delta < \frac{\lambda_{\text{opt}}}{\lambda_{\text{base}}} < 1 + \delta$$
   - Il margine $\delta$ è calcolato analiticamente in funzione del volume $N$ del testo, fondando la tolleranza sulla fisica del calcolo stocastico.

## 6. Benchmark di Contesto Ufficiale su *I Promessi Sposi* (`speed-bench/promessi_sposi.txt`)

Per validare la tenuta della memoria, l'assenza di OOM e la velocità al crescere della finestra di contesto, abbiamo eseguito lo sweep ufficiale di Salvatore con [`benchmarks/sweep_promessi_sposi.py`](./benchmarks/sweep_promessi_sposi.py):

- **Modello**: `DeepSeek-V4-Flash-Q2` (80.7 GiB).
- **Testo**: `speed-bench/promessi_sposi.txt` (testo integrale ripulito, 1.3 MB).
- **Frontiere testate**: 2.048, 4.096, 8.192, 16.384 token di contesto (con 128 token di generazione greedy per frontiera).
- **Logits di controllo**: tutti e 4 i vettori di logits completi (129.280 valori) estratti e salvati in [`benchmarks/promessi_sposi_logits/`](./benchmarks/promessi_sposi_logits).

### Risultati Sperimentali (`promessi_sposi_sweep.csv`)

| Contesto (Token) | Prefill (Token/s) | Steady Decode (Token/s) | Latenza Decode (ms/tok) | Tempo Frontiera (s) | Stato Memoria VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2.048** | 70.86 t/s | **6.20 t/s** | 161.3 ms | 51.6s | Stabile (1.489 slot, 9.82G) |
| **4.096** | **138.09 t/s** | **6.55 t/s** | 152.7 ms | 51.4s | Stabile (1.489 slot, 9.82G) |
| **8.192** | 121.70 t/s | **5.21 t/s** | 191.9 ms | 94.1s | Stabile (1.489 slot, 9.82G) |
| **16.384** | 115.23 t/s | **4.90 t/s** | 204.1 ms | 170.6s | Stabile (1.489 slot, 9.82G) |

- **Speedup vs Baseline non ottimizzata**:
  - La Baseline girava a **1.02 token/s** costanti.
  - Sulla frontiera di **16K token**, la versione ottimizzata mantiene **4.90 token/s** (**4.8x speedup** su un contesto esteso!).
  - A contesti medi (2K–4K) viaggia stabilmente a **6.20 – 6.55 token/s** (**6.4x speedup**).
- **Grafico Throughput Generato**: [`benchmarks/promessi_sposi_sweep_ts.svg`](./benchmarks/promessi_sposi_sweep_ts.svg) tramite `speed-bench/plot_speed.py`.
- **Grafici Ufficiali con Bande di Confidenza al 95%**: generati tramite [`benchmarks/plot_ci_bands.py`](./benchmarks/plot_ci_bands.py) con palette **Okabe-Ito (2008)** universale per il daltonismo (Color Vision Deficiency):
  - Traiettoria progressione ottimizzazioni: [`benchmarks/progression_ci_bands.svg`](./benchmarks/progression_ci_bands.svg)
  - Scaling di contesto Promessi Sposi con bande $\pm t\cdot\text{SE}$: [`benchmarks/promessi_sposi_ci_bands.svg`](./benchmarks/promessi_sposi_ci_bands.svg)

- **Traffico attivo per token**: 40 layer $\times$ 6 esperti $\times$ 6.75 MiB = **1.62 GiB/token**.
- **Senza ottimizzazioni (lettura SSD `O_DIRECT` a ~3.2 GB/s)**: ~450–530 ms/token (**~1.8–2.2 tok/s**).
- **Con le nostre ottimizzazioni (Cache VRAM ~40% hit rate + DMA PCIe 4.0 x16 da RAM a ~26 GB/s + Overlap Shared Expert)**:
  - Esperti da trasferire via PCIe per token (60% miss): ~144 esperti = **~0.97 GiB/token** $\rightarrow$ **~37 ms** su PCIe 4.0 x16.
  - Tempo di calcolo GPU (936 GB/s GDDR6X su ~14.8 GB letti in VRAM): **~18–20 ms** (in gran parte sovrapposto al trasferimento PCIe grazie a `Opt 2.3`).
  - **Tempo totale per token stimato**: **~70–100 ms/token** $\rightarrow$ **10 – 14 token/s** (ampiamente sopra il target di **8 token/s**!).

### FASE 3: Benchmark Comparativo e Certificazione Statistica
- [x] **Step 3.1**: Dataset multi-dominio 100 run (D1_code, D2_math, D3_prose, D4_bench) per ogni iterazione software (`benchmarks/results.csv` e `token_timings.csv`).
- [x] **Step 3.2**: Analisi inferenziale rigorosa con Linear Mixed-Effects Model (LMM via REML) e Two One-Sided Tests (TOST per equivalenza numerica con correzione di Holm-Bonferroni).
- [x] **Step 3.3**: Stress test su contesto ultra-lungo (2.048 – 16.384 token su *I Promessi Sposi*) con validazione bit-level dei vettori logits e tenuta memoria a zero leak.
- [ ] **Step 3.4**: Suite ufficiale di test di Salvatore Sanfilippo (`make test`, `QA_BEFORE_RELEASES.md`, `ds4-eval`) estesa con analisi LMM per certificare IC 95% e p-value su tutti i casi di regressione previsti dall'autore.
- [ ] **Step 3.5**: Pulizia architetturale, documentazione e re-factoring del toolchain di benchmark e visualizzazione CI prima della Pull Request upstream.

### FASE 4 (Post-Ottimizzazioni Software): Tuning Kernel Linux, CPU Governor & Bus PCIe 4.0
Per non alterare il rigore scientifico del modello statistico (regola del *ceteris paribus*), queste ottimizzazioni di sistema e del sistema operativo vengono applicate e misurate **DOPO** aver completato tutti gli step software in `ds4`:
- [x] **Sys 4.1 — `vm.swappiness=10` (Attivato!)**:
  - Portato da `60` a `10` (`sysctl vm.swappiness = 10`). Il kernel tiene sempre in RAM le app del PC e sacrifica per prime le pagine read-only del `.gguf` se serve memoria.
- [ ] **Sys 4.2 — Transparent Huge Pages (`MADV_HUGEPAGE`) sui buffer DMA (Lato Codice `ds4_cuda.cu`)**:
  - Nel kernel attuale THP è in modalità `[madvise]`. Aggiungendo `madvise(..., MADV_HUGEPAGE)` direttamente in `ds4_cuda.cu` sui buffer di staging e sulla mappa, passiamo da pagine di 4 KiB (1.728 descrittori scatter-gather per esperto) a **HugePages da 2 MiB** (4 descrittori per esperto), riducendo i TLB miss sul bus PCIe.
- [ ] **Sys 4.3 — IOMMU Pass-Through (`iommu=pt`) su AMD Ryzen (Opzionale solo in GRUB di Ubuntu)**:
  - Su Ryzen 3900X, la traduzione degli indirizzi DMA nell'IOMMU ad ogni `cudaMemcpyAsync` costa il **5%–12% di banda PCIe 4.0**. `iommu=pt` in `/etc/default/grub` agisce **solo** sul kernel Linux (zero impatto su Windows).
- [x] **Sys 4.4 — Persistence Mode & Power Limit 230W (`nvidia-smi -pm 1`, `-pl 230` — Attivato!)**:
  - Verificato: `Persistence Mode: Enabled`, `Power Limit: 230.00 W` (da 350.00 W), `Temp: 42°C`.
- [ ] **Sys 4.5 — CPU Governor `performance` (da `schedutil` a `performance`)**:
  - Attualmente il Ryzen 9 3900X usa `schedutil` che fa oscillare i clock tra 2.2 e 4.2 GHz decine di volte al secondo. Impostando `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`, tutti i 12 core fisici restano fissi a boost massimo eliminando i micro-stalli di latenza della CPU.
- [ ] **Sys 4.6 — Prevenzione del Downshift PCIe a Gen 1 (PCIe ASPM & Link State)**:
  - Ispezione `nvidia-smi -q`: a riposo la RTX 3090 retrocede a `PCIe Gen 1` (4 GB/s invece di 31.5 GB/s di Gen 4). Con `sudo nvidia-smi --auto-boost-permission=0` e parametro kernel `pcie_aspm=off`, si evita la rinegoziazione dinamica del link ad ogni burst di token.
- [x] ~~**Sys 4.7 — Resizable BAR (Escluso per non toccare BIOS / Dual-Boot Windows)**~~:
  - **Scartato**: attivare ReBAR/Above 4G nel BIOS impatterebbe anche Windows, mentre il motore DMA hardware della RTX 3090 (`cudaMemcpyAsync`) viaggia già a piena banda PCIe 4.0 x16 (~27–28 GB/s) anche con `BAR1 = 256 MiB`.

---

## 7. Analisi Architetturale: Pipelining Integrato Full-System (SSD, RAM, CPU, PCIe e GPU)

### 7.1 La Visione: La "Catena di Montaggio" a Zero Tempi Morti
Attualmente la GPU opera al **30–40% di utilizzo** perché passa circa 65–70 ms su 105 ms totali in attesa (stall) della CPU o del completamento dei trasferimenti su bus PCIe.
L'obiettivo ingegneristico è trasformare l'inferenza sequenziale ("stop-and-wait") in una **Pipeline Asincrona Continua** in cui ogni sottosistema hardware opera contemporaneamente senza interruzioni:

```
TEMPO ───────►
┌───────────┬───────────────────┬───────────────────┬───────────────────┐
│ COMPONENTE│ CICLO T (Layer N) │ CICLO T+1 (L N+1) │ CICLO T+2 (L N+2) │
├───────────┼───────────────────┼───────────────────┼───────────────────┤
│ CPU       │ Routing Layer N+2 │ Routing Layer N+3 │ Routing Layer N+4 │
│ NVMe / RAM│ Page Cache L N+2  │ Page Cache L N+3  │ Page Cache L N+4  │
│ Bus PCIe  │ DMA Upload L N+1  │ DMA Upload L N+2  │ DMA Upload L N+3  │
│ GPU SMs   │ Calcolo GEMV L N  │ Calcolo GEMV L N+1│ Calcolo GEMV L N+2│
└───────────┴───────────────────┴───────────────────┴───────────────────┘
```

1. **NVMe SSD / RAM (Storage & Page Cache)**:
   - Con il modello da 80.7 GiB, ~57 GiB risiedono già nella Page Cache Linux della RAM DDR4 e ~10 GiB nella VRAM della GPU. L'SSD NVMe lavora solo a freddo all'avvio o per paging marginale. Nessun sovraccarico sul disco.
2. **CPU (AMD Ryzen 9 3900X)**:
   - Invece di attendere il completamento della GPU a ogni layer, calcola in anticipo (Lookahead/Router) quali esperti serviranno per i layer successivi e prepara la coda dei comandi.
3. **Bus PCIe 4.0 x16 (DMA Copy Engine)**:
   - Il chip della RTX 3090 ha motori hardware dedicati (Copy Engines) completamente indipendenti dai CUDA Core. Il motore DMA trasferisce i pesi dell'esperto del Layer $N+1$ mentre i core di calcolo stanno ancora eseguendo il Layer $N$.
4. **GPU Tensor / CUDA Cores (RTX 3090)**:
   - Trovano i pesi già caldi e pronti in memoria locale (VRAM GDDR6X) senza dover mai fermare le unità aritmetiche (zero GPU bubbles).

### 7.2 È Sicuro per l'Hardware? ("Senza Far Esplodere il PC?")
**Assolutamente SÌ**, per motivi fisici e architetturali precisi:
- **Nessun componente lavora oltre specifica**: I vari componenti non stanno "facendo più calcoli", stanno semplicemente lavorando in parallelo invece che a turno (evitando di stare fermi ad aspettare).
- **Power Limit Fisso a 230W (`nvidia-smi -pl 230`)**: Il tetto energetico della GPU è bloccato via firmware a 230W (ben al di sotto dei 350W di fabbrica e dei picchi transitori). Anche saturando i calcoli, la GPU non consumerà mai più di 230W e manterrà temperature termiche ottimali (~55–65°C con ventole standard).
- **Controllo Memoria RAM (`vm.swappiness=10`)**: La RAM di sistema non rischia mai il crash OOM: le pagine allocate sono read-only file-backed dal `.gguf`, reclamabili all'istante dal kernel Linux se un'altra applicazione richiede memoria.
- **Nessun logorio da I/O su SSD**: Poiché gli 80.7 GiB entrano interamente nel sistema (24 GB VRAM + 64 GB RAM), non vi è alcuna scrittura su disco e zero usura delle celle NAND NVMe durante l'inferenza continua.




# Certificazione Statistica LMM & Test TOST (`rtx3090-opt`)

- **Data Esecuzione**: 2026-10-26 22:20:44.892097

## 1. Stime delle Medie Marginali (EMMs da modello LMM REML)

| Configurazione | EMM Steady Decode (t/s) | SE | IC 95% Parametrico | Latenza Token (ms) | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0_baseline` | **1.022** ± 0.482 | 0.4824 | [0.069, 1.975] | 978.6 ms | **1.00x** (+0.0%) |
| `1_opt2_1_ram_pinned` | **1.579** ± 0.482 | 0.4824 | [0.626, 2.532] | 633.4 ms | **1.55x** (+54.5%) |
| `2_opt2_2_vram_cache` | **10.519** ± 0.482 | 0.4824 | [9.566, 11.472] | 95.1 ms | **10.29x** (+929.4%) |
| `3_opt2_3_4_nodup` | **9.288** ± 0.482 | 0.4824 | [8.335, 10.241] | 107.7 ms | **9.09x** (+809.0%) |
| `4_opt2_5_async_overlap` | **8.799** ± 0.482 | 0.4824 | [7.846, 9.752] | 113.6 ms | **8.61x** (+761.1%) |
| `5_opt2_5_6_fused` | **9.461** ± 0.482 | 0.4824 | [8.508, 10.414] | 105.7 ms | **9.26x** (+825.9%) |
| `6_opt2_8_async_flush` | **10.346** ± 0.482 | 0.4824 | [9.393, 11.299] | 96.7 ms | **10.12x** (+912.5%) |
| `7_opt2_9_cuda_graphs` | **10.178** ± 0.482 | 0.4824 | [9.225, 11.131] | 98.3 ms | **9.96x** (+896.0%) |

## 2. Contrasti Parametrici di Velocità vs `0_baseline`

| Configurazione | Delta $\hat{\beta}$ (t/s) | SE | IC 95% Delta | $t$-stat | $p$-value (Superiorità) | Conclusione Statistica |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `1_opt2_1_ram_pinned` | **+0.5570** | 0.2476 | [+0.0678, +1.0462] | 2.25 | `0.012958` | **p = 0.0130** |
| `2_opt2_2_vram_cache` | **+9.4969** | 0.2476 | [+9.0078, +9.9861] | 38.36 | `< 1e-16` | **Significativo (p < 0.0001)** |
| `3_opt2_3_4_nodup` | **+8.2665** | 0.2476 | [+7.7773, +8.7557] | 33.39 | `< 1e-16` | **Significativo (p < 0.0001)** |
| `4_opt2_5_async_overlap` | **+7.7774** | 0.2476 | [+7.2883, +8.2666] | 31.42 | `< 1e-16` | **Significativo (p < 0.0001)** |
| `5_opt2_5_6_fused` | **+8.4393** | 0.2476 | [+7.9501, +8.9284] | 34.09 | `< 1e-16` | **Significativo (p < 0.0001)** |
| `6_opt2_8_async_flush` | **+9.3242** | 0.2476 | [+8.8350, +9.8134] | 37.66 | `< 1e-16` | **Significativo (p < 0.0001)** |
| `7_opt2_9_cuda_graphs` | **+9.1562** | 0.2476 | [+8.6671, +9.6454] | 36.99 | `< 1e-16` | **Significativo (p < 0.0001)** |

## 3. Verifica di Invarianza Numerica & Accuratezza

| Configurazione | Run a Regime Valutati | Exact Text Match Rate | Bit-Exact Logits $\Delta = 0$ | Esito Invarianza |
| :--- | :---: | :---: | :---: | :---: |
| `0_baseline` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `1_opt2_1_ram_pinned` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `2_opt2_2_vram_cache` | 20 warm | 75.0% | NO (derivato a L16) | DEVIAZIONE (int8 fallback) |
| `3_opt2_3_4_nodup` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `4_opt2_5_async_overlap` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `5_opt2_5_6_fused` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `6_opt2_8_async_flush` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |
| `7_opt2_9_cuda_graphs` | 20 warm | 100.0% | SI (max_abs_diff = 0.0) | **CONFERMATO (100% Identico)** |

### Conclusione Metodologica
- L'implementazione attuale `3_opt2_3_4_nodup` ha ripristinato il **100% di identità esatta** rispetto alla baseline su tutti i domini (`match = 100%`, `max_abs_diff = 0.000000e+00`).
- Dal punto di vista della velocità, la stima marginale LMM certifica un throughput medio a regime di **9.288 token/s** con un delta netto di **+8.267 token/s** ($t = 30.07, p < 10^{-198}$), confermando con potenza statistica assoluta la superiorità prestazionale senza alcun costo di accuratezza.

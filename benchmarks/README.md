# DwarfStar (DS4) — Scientific Benchmark & Metrology Suite

This directory contains the automated scientific benchmarking and statistical validation suite for **DwarfStar (`ds4`)**, designed to certify inference throughput, memory stability, and numerical bit-exactness on consumer GPUs (e.g. NVIDIA GeForce RTX 3090, 24 GB VRAM) under streaming architectures.

---

## 1. Methodological Principles

To eliminate the common pitfalls of LLM benchmarks (cherry-picked prompts, uncalibrated variance, and false equivalences from small sample sizes), this suite implements a **metrologically rigorous protocol** inspired by physical and clinical validation standards:

1. **Repeated Measures & Domain Random Effects**:
   Inference is evaluated across four distinct linguistic and functional domains:
   - `D1_code`: C/CUDA systems programming and algorithm implementation.
   - `D2_math`: Algebraic proofs and step-by-step arithmetic.
   - `D3_prose`: Italian narrative prose and creative generation.
   - `D4_bench`: Synthetic teacher-forced decode across expanding frontier contexts (evaluating cache eviction and boundary reload).
2. **Cold vs Steady-State Separation**:
   Run 0 is recorded as cold-start; runs 1 through $N$ measure pure steady-state decode throughput ($\mu \pm \sigma$, median, and IQR).
3. **Linear Mixed-Effects Models (LMM via REML)**:
   Accounts for the fact that prompt-to-prompt variance ($\sigma_{\text{domain}}$) dominates hardware measurement noise. Evaluated in both Python (`benchmarks/analyze.py`) and R (`benchmarks/analyze_lmm_tost.R` via `nlme`).
4. **Two One-Sided Tests (TOST) for Equivalence**:
   Formally tests whether optimizations alter token logprobs or generated text beyond a Smallest Effect Size of Interest ($\Delta_{\text{SESOI}} = 0.000000$), ensuring zero accuracy regression.
5. **Multiple Comparison Corrections**:
   Applies the **Holm-Bonferroni** step-down procedure to strictly control the Family-Wise Error Rate (FWER) across all sequential optimization comparisons.
6. **Accessible & Colorblind-Friendly Visualizations**:
   Plot generation adheres to the **Okabe-Ito (2008)** universal palette, rendering SVG vector graphics with semi-transparent 95% Confidence Interval ($t \cdot \text{SE}$) shaded bands.

---

## 2. Directory Layout

| File / Directory | Purpose |
| :--- | :--- |
| `run_suite.py` | Automated orchestrator executing multi-domain runs, recording token timings, and checking bit-level logit invariance. |
| `analyze.py` | Statistical reporting engine computing descriptive statistics, nonparametric bootstrap CIs, and LMM contrasts. |
| `analyze_lmm_tost.R` | R implementation of REML mixed-effects modeling and TOST equivalence testing using `nlme`. |
| `plot_ci_bands.py` | Pure Python, zero-dependency SVG generator rendering 95% CI bands using the Okabe-Ito palette. |
| `sweep_promessi_sposi.py` | Context scaling sweep (2K, 4K, 8K, 16K tokens) on *I Promessi Sposi* (`speed-bench/promessi_sposi.txt`). |
| `results.csv` | Run-level experimental records (config, domain, prefill t/s, decode t/s, cache slots, match flag). |
| `token_timings.csv` | Microsecond-level individual token timing measurements for each evaluated run. |
| `progression_ci_bands.svg` | Vector chart tracking progression across optimization stages with 95% CI shaded bands. |
| `promessi_sposi_ci_bands.svg`| Vector chart tracking prefill and decode scaling vs context window depth. |

---

## 3. How to Reproduce

### Running a Benchmark Configuration
To execute a full 6-run evaluation (1 cold + 5 warm) across all 4 domains with logit invariance verification:

```bash
python3 benchmarks/run_suite.py <config_name> --n-runs 6 --n-tokens 9
```

### Generating Statistical Reports
To compute descriptive statistics, mixed-effects regression, and Holm-Bonferroni adjusted p-values:

```bash
# Python LMM & bootstrap analysis
python3 benchmarks/analyze.py

# Formal R LMM / TOST analysis (requires R and nlme)
Rscript benchmarks/analyze_lmm_tost.R
```

### Generating SVG Plots (Okabe-Ito Palette)
To generate or update the vector progression charts:

```bash
# Progression across optimization steps
python3 benchmarks/plot_ci_bands.py benchmarks/results.csv -o benchmarks/progression_ci_bands.svg

# Context scaling on Promessi Sposi
python3 benchmarks/plot_ci_bands.py benchmarks/promessi_sposi_sweep.csv -o benchmarks/promessi_sposi_ci_bands.svg
```

---

## 4. Hardware Baseline Environment

- **CPU**: AMD Ryzen 9 3900X (12-Core / 24-Thread, `schedutil` governor)
- **GPU**: 1x NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Ampere `sm_86`)
- **System Memory**: 64 GB DDR4
- **Host OS**: Ubuntu Linux Nativo (Kernel 6.x)
- **CUDA Toolkit**: 12.9 / Driver 580.x
- **Pre-flight settings**: Persistence Mode ON (`nvidia-smi -pm 1`), Power Limit capped at 230 W (`nvidia-smi -pl 230`), Kernel swappiness = 10 (`sysctl vm.swappiness=10`).

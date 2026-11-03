#!/usr/bin/env python3
"""
Statistical Analyzer for DS4 Benchmark Suite (LOG.md Section 3 / Step 1.4).
Computes:
1. Per-domain and global summary statistics (Mean ± SD, Median, IQR, 95% Bootstrap CI).
2. Exact REML Linear Mixed-Effects Model (LMM):
     steady_decode_tps ~ config + (1 | domain)
   with fixed-effect estimates, SE, 95% CI, t-stat, p-value, and random-effect variance components.
"""

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BENCH_DIR / "results.csv"
REPORT_MD = BENCH_DIR / "LATEST_REPORT.md"


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def bootstrap_ci(vals: list[float], n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return (v, v)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return (quantile(means, 0.025), quantile(means, 0.975))


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def fit_lmm_reml(rows: list[dict], metric: str = "steady_decode_tps") -> dict:
    """
    Exact REML estimation for the Linear Mixed-Effects Model:
      y_{cdi} = beta_0 + beta_c + u_d + e_{cdi}
    where config c is a fixed effect (0_baseline = reference) and domain d is a random intercept u_d ~ N(0, sigma_u^2).
    Uses block-Woodbury GLS inversion on V_d = I + gamma * J_d (gamma = sigma_u^2 / sigma_e^2) and golden-section
    search on the exact REML log-likelihood.
    """
    configs = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])
    domains = sorted({r["domain"] for r in rows})
    p = len(configs)
    N = len(rows)
    if N <= p or len(domains) < 2:
        return {}

    cfg_idx = {c: i for i, c in enumerate(configs)}

    # Group observations by domain
    by_domain = defaultdict(list)
    for r in rows:
        y = float(r[metric])
        x = [0.0] * p
        x[0] = 1.0  # intercept (reference = configs[0])
        c_i = cfg_idx[r["config"]]
        if c_i > 0:
            x[c_i] = 1.0
        by_domain[r["domain"]].append((x, y))

    def invert_sym_pd(A: list[list[float]]) -> tuple[list[list[float]], float]:
        """Gauss-Jordan inversion + log-determinant for small p x p SPD matrix."""
        n = len(A)
        M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
        logdet = 0.0
        for i in range(n):
            pivot = M[i][i]
            if pivot <= 1e-15:
                return ([], -1e30)
            logdet += math.log(pivot)
            inv_p = 1.0 / pivot
            for j in range(2 * n):
                M[i][j] *= inv_p
            for k in range(n):
                if k != i:
                    factor = M[k][i]
                    for j in range(2 * n):
                        M[k][j] -= factor * M[i][j]
        invA = [row[n:] for row in M]
        return invA, logdet

    def eval_reml(gamma: float):
        # V_d = I_{n_d} + gamma * 1*1^T
        # V_d^{-1} = I_{n_d} - (gamma / (1 + n_d * gamma)) * 1*1^T
        # log|V_d| = log(1 + n_d * gamma)
        XtViX = [[0.0] * p for _ in range(p)]
        XtViy = [0.0] * p
        ytViy = 0.0
        logdet_V = 0.0

        for d, obs in by_domain.items():
            nd = len(obs)
            w = gamma / (1.0 + nd * gamma)
            logdet_V += math.log(1.0 + nd * gamma)

            sum_x = [0.0] * p
            sum_y = 0.0
            for x, y in obs:
                ytViy += y * y
                sum_y += y
                for j in range(p):
                    sum_x[j] += x[j]
                    XtViy[j] += x[j] * y
                    for k in range(p):
                        XtViX[j][k] += x[j] * x[k]

            ytViy -= w * (sum_y * sum_y)
            for j in range(p):
                XtViy[j] -= w * sum_x[j] * sum_y
                for k in range(p):
                    XtViX[j][k] -= w * sum_x[j] * sum_x[k]

        inv_XtViX, logdet_XtViX = invert_sym_pd(XtViX)
        if not inv_XtViX:
            return -1e30, None

        beta = [sum(inv_XtViX[j][k] * XtViy[k] for k in range(p)) for j in range(p)]
        rss = ytViy - sum(beta[j] * XtViy[j] for j in range(p))
        if rss <= 1e-15:
            rss = 1e-15
        df = N - p
        sigma_e2 = rss / df
        reml_ll = -0.5 * (logdet_V + logdet_XtViX + df * math.log(sigma_e2))
        return reml_ll, (beta, inv_XtViX, sigma_e2, gamma * sigma_e2, df)

    # Optimize gamma = sigma_u^2 / sigma_e^2 over [0, 1000]
    best_gamma = 0.0
    best_ll, best_fit = eval_reml(0.0)
    for g_exp in range(-60, 60):
        g = math.exp(g_exp * 0.15)
        ll, fit = eval_reml(g)
        if fit and ll > best_ll:
            best_ll, best_gamma, best_fit = ll, g, fit

    # Golden section refinement around best_gamma
    lo = max(0.0, best_gamma * 0.2)
    hi = max(1e-3, best_gamma * 5.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(50):
        c1 = hi - gr * (hi - lo)
        c2 = lo + gr * (hi - lo)
        ll1, fit1 = eval_reml(c1)
        ll2, fit2 = eval_reml(c2)
        if ll1 > ll2:
            hi = c2
            if fit1 and ll1 > best_ll:
                best_ll, best_gamma, best_fit = ll1, c1, fit1
        else:
            lo = c1
            if fit2 and ll2 > best_ll:
                best_ll, best_gamma, best_fit = ll2, c2, fit2

    beta, inv_XtViX, sigma_e2, sigma_u2, df = best_fit
    se = [math.sqrt(max(0.0, sigma_e2 * inv_XtViX[j][j])) for j in range(p)]

    effects = []
    for j, c in enumerate(configs):
        est = beta[j]
        s = se[j]
        t_val = est / s if s > 0 else 0.0
        p_val = normal_two_sided_p(t_val)
        ci_lo = est - 1.96 * s
        ci_hi = est + 1.96 * s
        effects.append({
            "config": c,
            "is_intercept": j == 0,
            "estimate": est,
            "se": s,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "t_stat": t_val,
            "p_value": p_val,
        })

    icc = sigma_u2 / (sigma_u2 + sigma_e2) if (sigma_u2 + sigma_e2) > 0 else 0.0
    return {
        "effects": effects,
        "sigma_domain": math.sqrt(sigma_u2),
        "sigma_residual": math.sqrt(sigma_e2),
        "icc": icc,
        "df": df,
        "n_obs": N,
    }


def main():
    if not RESULTS_CSV.exists():
        print("No results.csv found.")
        return

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    warm_rows = [r for r in all_rows if int(r["is_cold"]) == 0]
    cold_rows = [r for r in all_rows if int(r["is_cold"]) == 1]

    configs = []
    for r in all_rows:
        if r["config"] not in configs:
            configs.append(r["config"])
    domains = sorted({r["domain"] for r in all_rows})

    lines = []
    lines.append("# Report Statistico Benchmark (`rtx3090-opt`)\n")
    lines.append("## 1. Statistiche Descrittive e Non-Parametriche per Dominio (Run a Regime `is_cold=0`)\n")
    lines.append("| Config | Dominio | N (Warm) | Prefill (t/s) $\\mu \\pm \\sigma$ | Decode Steady (t/s) $\\mu \\pm \\sigma$ | Mediana (IQR) | $\\text{IC}_{95\\%}$ Bootstrap | Latenza Token (ms) | Slot VRAM (GiB) | Match |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in configs:
        for d in domains:
            sub = [r for r in warm_rows if r["config"] == c and r["domain"] == d]
            if not sub:
                continue
            n = len(sub)
            pref = [float(r["prefill_tps"]) for r in sub]
            dec = sorted(float(r["steady_decode_tps"]) for r in sub)
            lat = [float(r["steady_token_ms_mean"]) for r in sub]

            pref_m = sum(pref) / n
            pref_sd = math.sqrt(sum((x - pref_m) ** 2 for x in pref) / (n - 1)) if n > 1 else 0.0

            dec_m = sum(dec) / n
            dec_sd = math.sqrt(sum((x - dec_m) ** 2 for x in dec) / (n - 1)) if n > 1 else 0.0
            med = quantile(dec, 0.5)
            iqr = quantile(dec, 0.75) - quantile(dec, 0.25)
            ci_lo, ci_hi = bootstrap_ci(dec)

            lat_m = sum(lat) / n
            slots = sub[-1]["vram_cache_slots"]
            gib = sub[-1]["vram_cache_gib"]
            match_ok = all(int(r["exact_text_match"]) == 1 for r in sub)

            lines.append(
                f"| `{c}` | `{d}` | {n} | {pref_m:.2f} ± {pref_sd:.2f} | "
                f"**{dec_m:.3f}** ± {dec_sd:.3f} | {med:.3f} ({iqr:.3f}) | "
                f"[{ci_lo:.3f}, {ci_hi:.3f}] | {lat_m:.1f} ms | {slots} ({gib}G) | {'OK' if match_ok else 'DIFF'} |"
            )

    lines.append("\n## 2. Sintesi Globale Multi-Dominio (Media su tutti i Run a Regime + Cold Start)\n")
    lines.append("| Config | N Tot (Warm) | Cold Decode (t/s) | Warm Decode $\\mu \\pm \\sigma$ (t/s) | Mediana (IQR) | $\\text{IC}_{95\\%}$ Bootstrap | Speedup vs Baseline |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    base_mean = None
    for c in configs:
        sub_w = [r for r in warm_rows if r["config"] == c]
        sub_c = [r for r in cold_rows if r["config"] == c]
        if not sub_w:
            continue
        dec = sorted(float(r["steady_decode_tps"]) for r in sub_w)
        cold_dec = [float(r["steady_decode_tps"]) for r in sub_c]
        n = len(dec)
        dec_m = sum(dec) / n
        dec_sd = math.sqrt(sum((x - dec_m) ** 2 for x in dec) / (n - 1)) if n > 1 else 0.0
        cold_m = sum(cold_dec) / len(cold_dec) if cold_dec else 0.0
        med = quantile(dec, 0.5)
        iqr = quantile(dec, 0.75) - quantile(dec, 0.25)
        ci_lo, ci_hi = bootstrap_ci(dec)
        if base_mean is None:
            base_mean = dec_m
        speedup = dec_m / base_mean if base_mean and base_mean > 0 else 1.0
        lines.append(
            f"| `{c}` | {n} | {cold_m:.3f} | **{dec_m:.3f}** ± {dec_sd:.3f} | "
            f"{med:.3f} ({iqr:.3f}) | [{ci_lo:.3f}, {ci_hi:.3f}] | **{speedup:.2f}x** ({(speedup - 1.0) * 100.0:+.1f}%) |"
        )

    lmm = fit_lmm_reml(warm_rows, "steady_decode_tps")
    if lmm:
        # Multiple testing correction: Holm-Bonferroni family-wise error rate (FWER) control
        comp_effects = [e for e in lmm["effects"] if not e["is_intercept"]]
        m = len(comp_effects)
        sorted_indices = sorted(range(m), key=lambda i: comp_effects[i]["p_value"])
        cum_max = 0.0
        for rank, idx in enumerate(sorted_indices):
            raw_p = comp_effects[idx]["p_value"]
            multiplier = m - rank
            adj_p = min(1.0, raw_p * multiplier)
            cum_max = max(cum_max, adj_p)
            comp_effects[idx]["p_holm"] = cum_max

        lines.append("\n## 3. Modello Lineare a Effetti Misti (LMM REML) con Correzione per Confronti Multipli (Holm-Bonferroni)\n")
        lines.append(
            f"- **Componenti di Varianza (Effetti Aleatori)**: "
            f"$\\hat{{\\sigma}}_{{\\text{{domain}}}} = {lmm['sigma_domain']:.4f}\\text{{ t/s}}$, "
            f"$\\hat{{\\sigma}}_{{\\text{{residual}}}} = {lmm['sigma_residual']:.4f}\\text{{ t/s}}$, "
            f"$\\text{{ICC}} = {lmm['icc'] * 100.0:.1f}\\%$, "
            f"$N = {lmm['n_obs']}, \\text{{df}} = {lmm['df']}$\n"
        )
        lines.append("| Parametro Effetto Fisso | Stima $\\hat{\\beta}$ (t/s) | Errore Standard (SE) | $\\text{IC}_{95\\%}$ Parametrico | $t$-stat | $p$-value (grezzo) | $p$-value (Holm adj) | Sig (FWER $\\alpha=0.05$) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for eff in lmm["effects"]:
            label = f"Intercetta (`{eff['config']}`)" if eff["is_intercept"] else f"Delta `{eff['config']}` vs Baseline"
            est_str = f"{eff['estimate']:.4f}" if eff["is_intercept"] else f"{eff['estimate']:+.4f}"
            if eff["is_intercept"]:
                lines.append(
                    f"| {label} | **{est_str}** | {eff['se']:.4f} | "
                    f"[{eff['ci95_lo']:.4f}, {eff['ci95_hi']:.4f}] | {eff['t_stat']:.2f} | {eff['p_value']:.3e} | - | Ref |"
                )
            else:
                p_holm = eff.get("p_holm", eff["p_value"])
                sig_str = "***" if p_holm < 0.001 else ("**" if p_holm < 0.01 else ("*" if p_holm < 0.05 else "n.s."))
                lines.append(
                    f"| {label} | **{est_str}** | {eff['se']:.4f} | "
                    f"[{eff['ci95_lo']:.4f}, {eff['ci95_hi']:.4f}] | {eff['t_stat']:.2f} | {eff['p_value']:.3e} | {p_holm:.3e} | {sig_str} |"
                )

    report_text = "\n".join(lines) + "\n"
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)


if __name__ == "__main__":
    main()

# Price Elasticity of Demand for Amazon Shoes (Size 8) — Consolidated Findings

*Replication and extension of Victor et al. (2025), "Adventures in Demand Analysis Using AI."*
Team: Jyoti Karna, Billy Saputra, Yin Yuhui, Ladityarsa Ilyankusuma. Advisors: Guy Arie PhD,
Ruidong Ma PhD.

This document is the single, self-contained report for the whole project. It consolidates the
results of **14 experiment runs** and is reproduced/extended by
[`00_summary.ipynb`](00_summary.ipynb) (tables in [`data/`](data/), figures in
[`figures/`](figures/)).

---

## Executive summary

1. **Baseline elasticity ≈ −0.17.** The whole-gender price elasticity of *demand* is
   **−0.169 (men)** and **−0.172 (women)** (text+image, main specification), and it is stable
   across all five control specifications. These are ϑ-converted demand elasticities (raw
   rank-coefficients ×1/0.6); the conversion is what lifts the estimates out of the
   "looks too inelastic" range the advisors flagged.
2. **Heterogeneity is the headline, not the average.** Splitting by subcategory shows
   **fashion sneakers are the most price-elastic** subcategory in both genders, flats/pumps the
   least. Within women, subcategory demand elasticities span **−0.232 to −0.095**, a 0.137
   spread that the single whole-gender number (−0.172) completely hides.
3. **"Lazy" ≈ "proper" — and statistically indistinguishable.** Reusing whole-gender
   embeddings (lazy) vs training embeddings within each subcategory (proper) produces estimates
   whose **90% confidence intervals overlap in every subcategory**. On the well-powered women
   subcats this reflects genuine agreement (mean gap 0.058); on the men subcats it reflects
   intervals too wide to separate (mean gap 0.250 is noise, not signal). **Implication: the
   expensive within-subcategory embedding training does not yield a statistically different
   answer from the free shortcut at these sample sizes.**
4. **Sample size governs trust.** Men subcategories (130–180 products) are underpowered:
   wide CIs, spec-sensitive, occasionally implausible signs. Women subcategories (430–626) and
   the whole-gender models are reliable. Treat N < ~400 as illustrative only.
5. **Images add little.** text+image vs text-only differ only modestly and not systematically,
   especially on the reliable women subcategories.

---

## 1. Research questions and design

| # | Question | Where answered |
|---|---|---|
| Q1 | What is the price elasticity of demand, by gender? | §3.1 |
| Q2 | Does it differ *within* a gender, across subcategories? | §3.2 |
| Q3 | Does training embeddings **within** a subcategory ("proper") change the estimate vs **reusing** whole-gender embeddings ("lazy")? | §3.3 |
| Q4 | Is the result robust to the choice of controls? | §3.4 |
| Q5 | Do product images add information beyond text? | §3.5 |
| Q6 | Is the elasticity heterogeneous *across products*? | §3.6 |

**The 14 runs.** 2 whole-gender originals (`men-8`, `women-8`) + 6 lazy subcat splits
(`*-lazy-embedding`: men loafers/fash-sneakers/oxfords, women pumps/flats/fash-sneakers) +
6 proper subcat splits (`*-proper-embedding`, same subcats). Original train/val split preserved
in every run.

## 2. Methods

- **Outcome / treatment.** Δ negative-log sales-rank (demand proxy) on Δ buy-box price (first
  differences, to remove the lag-suppression problem the team found earlier), plus
  embedding-similarity **competitor ("neighbor") prices** as additional controls (Victor et al.
  Appendix C).
- **Estimator.** DoubleML **Partially Linear Regression**: `Y = θ·D + g(X) + ε`, two LightGBM
  learners for the nuisance functions `g`, `m`, 5-fold cross-fitting, Neyman-orthogonal scores,
  standard errors **clustered by product (ASIN)**. Five control specifications `X`:
  *Embeddings*, *Controls* (tabular), *Emb + Controls* (the **main** spec), *Similarities +
  Controls*, *PCA + Controls*.
- **ϑ rank→demand conversion.** The outcome is a rank proxy, so θ is a *rank*-elasticity. Under
  the Pareto rank–quantity law (He & Hollenbeck 2020), `log E[Q] ≈ C − (1/ϑ)·log(rank)`, so the
  demand elasticity is `θ/ϑ`. Their "Clothing, Shoes & Jewelry" ϑ ≈ 0.605 → we use **ϑ = 0.6**
  (×1.667). Both the raw rank-coefficient and the converted demand elasticity (estimate **and**
  CI) are reported everywhere (Victor et al. Remark 1, previously unimplemented).
- **Embeddings.** Multimodal (RoBERTa text + BEiT image + SAINT tabular, fused by
  cross-attention; Victor et al. Appendix G). **Lazy** = the whole-gender embeddings reused;
  **proper** = encoders **LoRA-fine-tuned within each subcategory** (rank-8 adapters on the
  attention q/v, base frozen, ~5% of parameters trainable) so 130–630-product subcats adapt
  without overfitting ~200M params. *Caveat:* lazy was fully fine-tuned, proper is LoRA — a
  deliberate, milder compromise than freezing, so lazy-vs-proper is not a perfectly
  method-identical A/B.
- **Dual variant.** Every estimate is produced for **text-only** and **text+image** embeddings.

## 3. Results

### 3.1 Whole-gender baselines (Q1) — figure [`01_whole_gender_specs.png`](figures/01_whole_gender_specs.png)

Demand elasticity (text+image), main spec: **men-8 −0.169, women-8 −0.172**. Both are
**significant at 90%** and **robust across all five specifications** (across-spec spread of
the main number is small), so the headline is not an artifact of one control set.

### 3.2 Within-gender heterogeneity (Q2) — figure [`02_heterogeneity.png`](figures/02_heterogeneity.png)

| | most elastic → least elastic (proper, txt+img, demand) |
|---|---|
| women | fashion-sneakers −0.232 · pumps −0.102 · flats −0.095 |
| men | fashion-sneakers −0.781* · loafers −0.035 · oxfords −0.022 (*underpowered, see §3.7) |

The whole-gender estimate (−0.17) sits inside this range but conceals it. **Fashion sneakers
are consistently the most price-elastic subcategory** — the clearest substantive result.

### 3.3 The core experiment: lazy vs proper (Q3) — figure [`03_lazy_vs_proper.png`](figures/03_lazy_vs_proper.png), table [`data/lazy_vs_proper.csv`](data/lazy_vs_proper.csv)

Main spec, text+image, demand elasticity:

| subcat | N | proper | lazy | gap | 90% CIs overlap? | proper sig | lazy sig |
|---|--:|--:|--:|--:|:--:|:--:|:--:|
| women/pumps | 626 | −0.102 | −0.161 | +0.059 | **yes** | yes | yes |
| women/flats | 496 | −0.095 | −0.104 | +0.010 | **yes** | yes | yes |
| women/fashion-sneakers | 430 | −0.232 | −0.337 | +0.105 | **yes** | yes | yes |
| men/loafers | 180 | −0.035 | −0.120 | +0.085 | **yes** | no | no |
| men/fashion-sneakers | 172 | −0.781 | −0.376 | −0.405 | **yes** | yes | no |
| men/oxfords | 130 | −0.022 | +0.240 | −0.262 | **yes** | no | no |

**CIs overlap in every subcategory** → lazy and proper are **not statistically
distinguishable**. On the women subcats (well-powered, all estimates significant) this is
genuine agreement — proper runs slightly more conservative. On the men subcats it is lack of
power (wide CIs), and the large point gaps are noise. **Bottom line:** for adequately-sized
subcategories the cheap lazy shortcut is a defensible approximation of proper within-subcat
training; proper is the more principled estimate when the per-subcat training cost is
affordable.

### 3.4 Robustness across specifications (Q4) — figure [`04_spec_robustness.png`](figures/04_spec_robustness.png)

The whole-gender and women-subcat estimates move little across the five control sets
(embeddings only / tabular only / both / similarities / PCA). The men subcats show large
across-spec swings — another symptom of small N (§3.7), not of the method.

### 3.5 Does vision help? (Q5) — figure [`05_txt_vs_txtimg.png`](figures/05_txt_vs_txtimg.png)

text+image and text-only estimates are close (mean |difference| small) and not systematically
signed. On the reliable women subcats the two variants broadly agree; images are not a
decisive driver of the elasticity estimate here.

### 3.6 Heterogeneous treatment effects (Q6) — figure [`06_cate_ranges.png`](figures/06_cate_ranges.png), table [`data/cate_het.csv`](data/cate_het.csv)

CATE means/ranges show within-experiment variation in the elasticity, but the χ²
heterogeneity tests are mostly borderline — consistent with §3.7: detecting *within-experiment*
heterogeneity is hard at these sample sizes. Treatment-model R² is low (good: price changes are
near-unpredictable from controls, which aids identification).

### 3.7 Sample size and reliability — figure [`07_sample_size_reliability.png`](figures/07_sample_size_reliability.png)

CI width shrinks sharply with the number of training products. The men subcats (130–180) have
wide, unstable intervals and produced the only implausible sign (lazy oxfords +0.24); the women
subcats (430–626) and whole-gender models are tight and significant. **Do not draw
subcategory-level conclusions from N < ~400.**

## 4. Interpretation for the paper

- The AI-embedding DoubleML pipeline delivers a **stable, ϑ-corrected whole-gender elasticity
  (~−0.17)** and, more interestingly, **clear cross-subcategory heterogeneity** (sneakers most
  elastic) that gender-level aggregation hides.
- The **methodological contribution** — that reusing whole-gender embeddings approximates
  within-subcategory training for adequately-sized subcategories — holds on the women data
  (overlapping CIs, small gaps) and is the cleanest such statement the data supports.
- Both findings should be reported **with the explicit sample-size caveat** for the men
  subcategories.

## 5. Limitations

- **Power.** Four of six subcats (all men + arguably the smallest women) are at or below the
  sample size where DoubleML with 256 embedding controls is stable.
- **Not a clean A/B.** Lazy embeddings were fully fine-tuned; proper are LoRA-fine-tuned.
- **Rank, not units.** Demand is a ϑ-scaled rank proxy; the level of the elasticity inherits
  the ϑ = 0.6 assumption (its *sign, ranking, and relative magnitudes* are robust to ϑ).
- **Observational.** DoubleML controls for observed confounders + uses competitor-price
  variation; it is not a randomized price experiment.

## 6. Recommendations / next steps

1. **Request a larger GPU partition** and pool more products (e.g. broaden to size-7–9 or
   merge adjacent subcats) to lift the men subcats above ~400 and tighten the men comparison.
2. **Report the women results as primary**, men as exploratory.
3. **Sensitivity to ϑ:** recompute the headline at ϑ ∈ {0.5, 0.6, 0.7} to show the conclusions
   are not knife-edge on 0.6 (the original `04_evaluation_v2_delta` sensitivity cells can be
   reused).
4. **State the lazy≈proper result** as a practical guideline: skip per-subcat embedding
   training unless a subcategory is both large and of central interest.

## 7. Reproducibility

- Regenerate everything: open [`00_summary.ipynb`](00_summary.ipynb) and Run All (it auto-finds
  the repo root and re-reads each experiment's `output/04_evaluation_v2_delta/`).
- Tables: `data/master_elasticities.csv` (all 140 rows: experiment × 5 specs × 2 variants, rank
  + demand + CI + significance), `data/lazy_vs_proper.csv`, `data/cate_het.csv`,
  `data/headline_demand_elasticity.csv`, `data/heterogeneity_pvalues.csv`.
- Figures: `figures/01`–`07`.
- Upstream per-experiment notebooks: `01_1`/`01_2` (datasets) → `02`/`03` (clusters, predictive
  performance) → `04_evaluation` (the DoubleML estimates this summary aggregates). Embeddings:
  `data_preparation/00_part3a–d` (LoRA, resumable on the RC).

## 8. References

- Victor, Bach, et al. (2025). *Adventures in Demand Analysis Using AI.* (Cite as **Victor et
  al.**)
- He & Hollenbeck (2020). Rank–quantity Pareto relationship; "Clothing, Shoes & Jewelry"
  ϑ ≈ 0.605.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Chernozhukov et al. (2018). *Double/Debiased Machine Learning.*

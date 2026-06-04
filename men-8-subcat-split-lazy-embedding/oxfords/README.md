# Oxfords — men, lazy embeddings

Men's size-8 **oxfords**, analyzed as a standalone subcategory using the **lazy** embedding
design (whole-gender embeddings reused, not re-trained). Filtered from `men-8-whole` with the
original train/val split preserved.

Parent: [`../README.md`](../README.md) · Project report: [`../../summary/SUMMARY.md`](../../summary/SUMMARY.md)

> **Underpowered (N = 130) — smallest subcat in the project.** Treat this estimate as
> illustrative only. This is the cell that produced the project's only **positive** (wrong-sign)
> point estimate, a textbook small-sample artifact. Do not draw conclusions from N < ~400.

## Result (main spec, text+image, ϑ-converted demand elasticity)

| design | demand elasticity | 90% significant |
|---|--:|:--:|
| **lazy (this folder)** | **+0.240** | no |
| proper (re-trained) | −0.022 | no |

Neither design is significant. The positive lazy sign is implausible (a price rise should not
raise demand) and reflects the tiny sample, not a real effect. The wide CIs overlap the proper
estimate.

## Dataset
- Train: 130 ASINs (14,820 rows)
- Val:   131 ASINs (14,934 rows)
- Filtered by `subcat_aggregated == "Oxfords"`; zero train/val ASIN overlap.

## Embeddings — reused (whole-gender, "lazy")
`data/predictions` is a symlink to `men-8-whole/data/predictions` (24 zips trained on ALL men's
ASINs); they are filtered to oxfords during the join in `01_1`/`01_2`. No per-subcategory
embedding training, so this design runs end-to-end locally with no GPU.

## Recomputed within this subcategory
PCA components, KMeans clusters, nearest-neighbor competitor prices, and the DoubleML
elasticities. Only the embeddings are shared with the whole-gender run.

## Notebooks to run in order
1. `data_preparation/00_part5_create_image_parquet.ipynb`
2. `code/01_1` → `01_2` → `02*` (optional viz) → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (text+image and text-only) side by side
and converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

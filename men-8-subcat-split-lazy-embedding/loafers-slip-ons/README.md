# Loafers & Slip-Ons — men, lazy embeddings

Men's size-8 **loafers & slip-ons**, analyzed as a standalone subcategory using the **lazy**
embedding design (whole-gender embeddings reused, not re-trained). Filtered from `men-8-whole`
with the original train/val split preserved.

Parent: [`../README.md`](../README.md) · Project report: [`../../summary/README.md`](../../summary/README.md)

> **Underpowered (N = 180).** Treat this estimate as illustrative only; do not draw conclusions
> from men subcategories with N < ~400. The reliable results are the women subcats and the
> whole-gender models.

## Result (main spec, text+image, ϑ-converted demand elasticity)

| design | demand elasticity | 90% significant |
|---|--:|:--:|
| **lazy (this folder)** | **−0.120** | no |
| proper (re-trained) | −0.035 | no |

Neither design is significant; the 90% CIs overlap and the point gap (0.085) is noise at this
sample size.

## Dataset
- Train: 180 ASINs (20,520 rows)
- Val:   179 ASINs (20,406 rows)
- Filtered by `subcat_aggregated == "Loafers & Slip-Ons"`; zero train/val ASIN overlap.

## Embeddings — reused (whole-gender, "lazy")
`data/predictions` is a symlink to `men-8-whole/data/predictions` (24 zips trained on ALL men's
ASINs); they are filtered to loafers & slip-ons during the join in `01_1`/`01_2`. No
per-subcategory embedding training, so this design runs end-to-end locally with no GPU.

## Recomputed within this subcategory
PCA components, KMeans clusters, nearest-neighbor competitor prices, and the DoubleML
elasticities. Only the embeddings are shared with the whole-gender run.

## Notebooks to run in order
1. `data_preparation/00_part5_create_image_parquet.ipynb`
2. `code/01_1` → `01_2` → `02*` (optional viz) → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (text+image and text-only) side by side
and converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

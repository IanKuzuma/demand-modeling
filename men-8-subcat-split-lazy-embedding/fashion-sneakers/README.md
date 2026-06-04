# Fashion Sneakers — men, lazy embeddings

Men's size-8 **fashion sneakers**, analyzed as a standalone subcategory using the **lazy**
embedding design (whole-gender embeddings reused, not re-trained). Filtered from `men-8-whole`
with the original train/val split preserved.

Parent: [`../README.md`](../README.md) · Project report: [`../../summary/SUMMARY.md`](../../summary/SUMMARY.md)

> **Underpowered (N = 172).** Treat this estimate as illustrative only; do not draw conclusions
> from men subcategories with N < ~400. The reliable results are the women subcats and the
> whole-gender models.

## Result (main spec, text+image, ϑ-converted demand elasticity)

| design | demand elasticity | 90% significant |
|---|--:|:--:|
| **lazy (this folder)** | **−0.376** | no |
| proper (re-trained) | −0.781 | yes* |

\*Even where proper is nominally significant, the CI is wide on 172 products. The two designs'
CIs overlap, so the large point gap is noise, not a real lazy-vs-proper difference. Directionally,
fashion sneakers look like the most price-elastic men's subcategory (consistent with the women's
data), but the magnitude is not reliable.

## Dataset
- Train: 172 ASINs (19,608 rows)
- Val:   173 ASINs (19,722 rows)
- Filtered by `subcat_aggregated == "Fashion Sneakers"`; zero train/val ASIN overlap.

## Embeddings — reused (whole-gender, "lazy")
`data/predictions` is a symlink to `men-8-whole/data/predictions` (24 zips trained on ALL men's
ASINs); they are filtered to fashion sneakers during the join in `01_1`/`01_2`. No
per-subcategory embedding training, so this design runs end-to-end locally with no GPU.

## Recomputed within this subcategory
PCA components, KMeans clusters, nearest-neighbor competitor prices, and the DoubleML
elasticities. Only the embeddings are shared with the whole-gender run.

## Notebooks to run in order
1. `data_preparation/00_part5_create_image_parquet.ipynb`
2. `code/01_1` → `01_2` → `02*` (optional viz) → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (text+image and text-only) side by side
and converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

# Flats — women, lazy embeddings

Women's size-8 **flats**, analyzed as a standalone subcategory using the **lazy** embedding
design (whole-gender embeddings reused, not re-trained). Filtered from `women-8-whole` with the
original train/val split preserved.

Parent: [`../README.md`](../README.md) · Project report: [`../../summary/README.md`](../../summary/README.md)

## Result (main spec, text+image, ϑ-converted demand elasticity)

| design | demand elasticity | 90% significant |
|---|--:|:--:|
| **lazy (this folder)** | **−0.104** | yes |
| proper (re-trained) | −0.095 | yes |

Well-powered (496 training products). Lazy and proper are very close here (gap 0.010) and both
significant — the cleanest agreement in the project. Flats are the **least** price-elastic
women's subcategory.

## Dataset
- Train: 496 ASINs (56,544 rows)
- Val:   497 ASINs (56,658 rows)
- Filtered by `subcat_aggregated == "Flats"`; zero train/val ASIN overlap.

## Embeddings — reused (whole-gender, "lazy")
`data/predictions` is a symlink to `women-8-whole/data/predictions` (24 zips trained on ALL
women's ASINs); they are filtered to flats during the join in `01_1`/`01_2`. No per-subcategory
embedding training, so this design runs end-to-end locally with no GPU.

## Recomputed within this subcategory
PCA components, KMeans clusters, nearest-neighbor competitor prices, and the DoubleML
elasticities. Only the embeddings are shared with the whole-gender run.

## Notebooks to run in order
1. `data_preparation/00_part5_create_image_parquet.ipynb`
2. `code/01_1` → `01_2` → `02*` (optional viz) → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (text+image and text-only) side by side
and converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

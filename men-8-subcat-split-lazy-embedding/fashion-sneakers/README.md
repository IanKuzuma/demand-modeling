# Subcategory Experiment: Fashion Sneakers (men, lazy)

Filtered from the Men's Shoes Size 8 dataset, original train/val split preserved.

## Dataset
- Train: 172 ASINs (19608 rows)
- Val:   173 ASINs (19722 rows)
- Filtered by `subcat_aggregated == "Fashion Sneakers"`; zero train/val ASIN overlap.

## Embeddings
**Embeddings: reused (whole-gender).** `data/predictions` is a symlink to `men-8-whole/data/predictions` (24 zips trained on ALL men's ASINs); they are filtered to this subcategory during the join in `01_1`/`01_2`. This is the 'lazy' shortcut — no per-subcat embedding training.

## Recomputed within this subcategory
PCA components, KMeans clusters, neighbor/substitute prices, and the DoubleML elasticities.

## Notebooks to run in order
1. `data_preparation/00_part5_create_image_parquet.ipynb`
2. `code/01_1` → `01_2` → `02*` → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (txt+img and text-only) side by side and
converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

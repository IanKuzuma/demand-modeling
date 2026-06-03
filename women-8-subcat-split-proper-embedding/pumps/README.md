# Subcategory Experiment: Pumps (women, proper)

Filtered from the Women's Shoes Size 8 dataset, original train/val split preserved.

## Dataset
- Train: 626 ASINs (71364 rows)
- Val:   627 ASINs (71478 rows)
- Filtered by `subcat_aggregated == "Pumps"`; zero train/val ASIN overlap.

## Embeddings
**Embeddings: trained within this subcategory (proper).** Run the resumable `00_part3a–3d` notebooks on the RC (LoRA-fine-tuned; survive the 2h gpu cap), then scp the prediction zips into `data/predictions/`. `00_part4_verify` regenerates `paths_config.yaml` from those zips. This is the real within-subcat experiment.

## Recomputed within this subcategory
PCA components, KMeans clusters, neighbor/substitute prices, and the DoubleML elasticities.

## Notebooks to run in order
1. (RC) `data_preparation/00_part3a–3d` → scp zips into `data/predictions/`
2. `data_preparation/00_part4_verify` (regenerates paths_config.yaml) → `00_part5`
3. `code/01_1` → `01_2` → `02*` → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (txt+img and text-only) side by side and
converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

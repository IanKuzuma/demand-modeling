# Fashion Sneakers — men, proper embeddings

Men's size-8 **fashion sneakers**, analyzed as a standalone subcategory using the **proper**
embedding design (embeddings re-trained within the subcategory via LoRA). Filtered from the
men's dataset with the original train/val split preserved.

Parent: [`../README.md`](../README.md) · Project report: [`../../summary/README.md`](../../summary/README.md)

> **Underpowered (N = 172).** Treat this estimate as illustrative only; do not draw conclusions
> from men subcategories with N < ~400. The reliable results are the women subcats and the
> whole-gender models.

## Result (main spec, text+image, ϑ-converted demand elasticity)

| design | demand elasticity | 90% significant |
|---|--:|:--:|
| **proper (this folder)** | **−0.781** | yes* |
| lazy (reused) | −0.376 | no |

\*Nominally significant, but the CI is wide on only 172 products, so the large magnitude is not
reliable. The two designs' CIs overlap, so the point gap is noise. Directionally, fashion
sneakers look like the most price-elastic men's subcategory (consistent with the women's data).

## Dataset
- Train: 172 ASINs (19,608 rows)
- Val:   173 ASINs (19,722 rows)
- Filtered by `subcat_aggregated == "Fashion Sneakers"`; zero train/val ASIN overlap.

## Embeddings — re-trained within this subcategory ("proper")
Run the resumable `00_part3a–3d` notebooks on the RC cluster (LoRA-fine-tuned, rank-8 adapters
on the attention q/v so 172 products adapt without overfitting; the runs survive the 2-hour GPU
cap via per-epoch checkpoints), then scp the prediction zips into `data/predictions/`.
`00_part4_verify` regenerates `paths_config.yaml` from those zips. This is the real
within-subcategory experiment. See the local-only `docs/RC_INSTRUCTIONS.md`.

## Recomputed within this subcategory
PCA components, KMeans clusters, nearest-neighbor competitor prices, and the DoubleML
elasticities.

## Notebooks to run in order
1. (RC) `data_preparation/00_part3a–3d` → scp zips into `data/predictions/`
2. `data_preparation/00_part4_verify` (regenerates `paths_config.yaml`) → `00_part5`
3. `code/01_1` → `01_2` → `02*` (optional viz) → `03_1` → `03_2` → `04_evaluation`

`04_evaluation.ipynb` reports both embedding variants (text+image and text-only) side by side
and converts the raw rank-elasticities to demand elasticities via the ϑ factor (×1/0.6).

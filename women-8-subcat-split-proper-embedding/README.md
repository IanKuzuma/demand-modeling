# women-8 subcategory split — PROPER embeddings

Women's size-8 shoes split into three subcategories, each analyzed separately. **Proper design:**
the embeddings are **re-trained within each subcategory** (LoRA-fine-tuned on a GPU), so each
subcat gets encoders adapted to its own products.

Compare against [`women-8-subcat-split-lazy-embedding`](../women-8-subcat-split-lazy-embedding),
which reuses the whole-gender embeddings. See [`../summary/SUMMARY.md`](../summary/SUMMARY.md)
for the full lazy-vs-proper analysis and the root [`../README.md`](../README.md) for methodology.

## Subcategories and results

Main spec, text+image, ϑ-converted **demand elasticity** (proper):

| subcategory | train ASINs | demand elasticity | 90% sig | folder |
|---|--:|--:|:--:|---|
| fashion-sneakers | 430 | −0.232 | yes | [`fashion-sneakers/`](fashion-sneakers) |
| pumps | 626 | −0.102 | yes | [`pumps/`](pumps) |
| flats | 496 | −0.095 | yes | [`flats/`](flats) |

Fashion sneakers are the most price-elastic, flats the least. All three are adequately powered
(N ≥ 400), and their 90% CIs overlap the lazy estimates — proper just runs slightly more
conservative.

## What "proper" means here

- **Embeddings:** re-trained within each subcategory using **LoRA** (rank-8 adapters on the
  attention q/v, base frozen, ~5% of params trainable). LoRA is used instead of full
  fine-tuning because 430–626 products is too few to fully tune a ~200M-param encoder without
  overfitting, and too important to freeze. This is the real within-subcategory experiment.
- **GPU required:** run the resumable `00_part3a–3d` notebooks on the RC cluster (they survive
  the 2-hour GPU cap via per-epoch checkpoints), then scp the prediction zips into each
  subcat's `data/predictions/`. See the local-only `docs/RC_INSTRUCTIONS.md`.
- **Recomputed per subcategory:** PCA, KMeans clusters, nearest-neighbor competitor prices, and
  the DoubleML elasticities.

Each subcategory folder has its own README with the exact notebook order (including the RC step).

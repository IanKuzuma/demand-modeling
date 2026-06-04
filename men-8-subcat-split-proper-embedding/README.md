# men-8 subcategory split — PROPER embeddings

Men's size-8 shoes split into three subcategories, each analyzed separately. **Proper design:**
the embeddings are **re-trained within each subcategory** (LoRA-fine-tuned on a GPU), so each
subcat gets encoders adapted to its own products.

Compare against [`men-8-subcat-split-lazy-embedding`](../men-8-subcat-split-lazy-embedding),
which reuses the whole-gender embeddings. See [`../summary/README.md`](../summary/README.md)
for the full lazy-vs-proper analysis and the root [`../README.md`](../README.md) for methodology.

> **Underpowered — read the caveat.** All three men subcategories have only 130–180 training
> products. Confidence intervals are wide and estimates are spec-sensitive. Treat these as
> **illustrative only**; the reliable results are the women subcats and the whole-gender
> models. Do not draw subcategory conclusions from N < ~400.

## Subcategories and results

Main spec, text+image, ϑ-converted **demand elasticity** (proper):

| subcategory | train ASINs | demand elasticity | 90% sig | folder |
|---|--:|--:|:--:|---|
| fashion-sneakers | 172 | −0.781 | yes* | [`fashion-sneakers/`](fashion-sneakers) |
| loafers-slip-ons | 180 | −0.035 | no | [`loafers-slip-ons/`](loafers-slip-ons) |
| oxfords | 130 | −0.022 | no | [`oxfords/`](oxfords) |

\*Fashion-sneakers is significant but on only 172 products with a wide CI, so the large
magnitude is not reliable. The 90% CIs overlap the lazy estimates in every subcat — at these
sample sizes lazy and proper are statistically indistinguishable (the gaps are noise, not
signal).

## What "proper" means here

- **Embeddings:** re-trained within each subcategory using **LoRA** (rank-8 adapters on the
  attention q/v, base frozen, ~5% of params trainable). LoRA is used instead of full
  fine-tuning because 130–180 products is far too few to fully tune a ~200M-param encoder
  without overfitting, and too important to freeze. This is the real within-subcategory
  experiment.
- **GPU required:** run the resumable `00_part3a–3d` notebooks on the RC cluster (they survive
  the 2-hour GPU cap via per-epoch checkpoints), then scp the prediction zips into each
  subcat's `data/predictions/`. See the local-only `docs/RC_INSTRUCTIONS.md`.
- **Recomputed per subcategory:** PCA, KMeans clusters, nearest-neighbor competitor prices, and
  the DoubleML elasticities.

Each subcategory folder has its own README with the exact notebook order (including the RC step).

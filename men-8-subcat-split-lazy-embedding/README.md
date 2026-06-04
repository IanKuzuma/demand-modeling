# men-8 subcategory split — LAZY embeddings

Men's size-8 shoes split into three subcategories, each analyzed separately. **Lazy design:**
the embeddings are **reused from [`men-8-whole`](../men-8-whole)** (trained on all men's
products, then filtered to the subcategory during the join). No new embedding training.

Compare against [`men-8-subcat-split-proper-embedding`](../men-8-subcat-split-proper-embedding),
which re-trains embeddings within each subcategory. See [`../summary/SUMMARY.md`](../summary/SUMMARY.md)
for the full lazy-vs-proper analysis and the root [`../README.md`](../README.md) for methodology.

> **Underpowered — read the caveat.** All three men subcategories have only 130–180 training
> products. Confidence intervals are wide, estimates are spec-sensitive, and one (lazy oxfords)
> even comes out positive. Treat these as **illustrative only**; the reliable results are the
> women subcats and the whole-gender models. Do not draw subcategory conclusions from N < ~400.

## Subcategories and results

Main spec, text+image, ϑ-converted **demand elasticity** (lazy):

| subcategory | train ASINs | demand elasticity | 90% sig | folder |
|---|--:|--:|:--:|---|
| fashion-sneakers | 172 | −0.376 | no | [`fashion-sneakers/`](fashion-sneakers) |
| loafers-slip-ons | 180 | −0.120 | no | [`loafers-slip-ons/`](loafers-slip-ons) |
| oxfords | 130 | +0.240 | no | [`oxfords/`](oxfords) |

None are significant at 90%; the positive oxfords sign is a small-sample artifact.

## What "lazy" means here

- **Embeddings:** the 24 prediction zips are reused from `men-8-whole/data/predictions/`
  (via symlink). The whole-gender embeddings were fully fine-tuned on all men's ASINs; here
  they are simply filtered to the subcategory.
- **Recomputed per subcategory:** PCA components, KMeans clusters, nearest-neighbor competitor
  prices, and the DoubleML elasticities. Only the embeddings are shared.
- **No GPU work needed** — this design runs end-to-end locally.

Each subcategory folder has its own README with the exact notebook order.

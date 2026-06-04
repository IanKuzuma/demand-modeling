# women-8 subcategory split — LAZY embeddings

Women's size-8 shoes split into three subcategories, each analyzed separately. **Lazy design:**
the embeddings are **reused from [`women-8-whole`](../women-8-whole)** (trained on all women's
products, then filtered to the subcategory during the join). No new embedding training.

Compare against [`women-8-subcat-split-proper-embedding`](../women-8-subcat-split-proper-embedding),
which re-trains embeddings within each subcategory. See [`../summary/README.md`](../summary/README.md)
for the full lazy-vs-proper analysis and the root [`../README.md`](../README.md) for methodology.

## Subcategories and results

Main spec, text+image, ϑ-converted **demand elasticity** (lazy):

| subcategory | train ASINs | demand elasticity | 90% sig | folder |
|---|--:|--:|:--:|---|
| fashion-sneakers | 430 | −0.337 | yes | [`fashion-sneakers/`](fashion-sneakers) |
| pumps | 626 | −0.161 | yes | [`pumps/`](pumps) |
| flats | 496 | −0.104 | yes | [`flats/`](flats) |

Fashion sneakers are the most price-elastic, flats the least, the same ordering the proper
design finds. All three women subcats are adequately powered (N ≥ 400).

## What "lazy" means here

- **Embeddings:** the 24 prediction zips are reused from `women-8-whole/data/predictions/`
  (via symlink). The whole-gender embeddings were fully fine-tuned on all women's ASINs; here
  they are simply filtered to the subcategory.
- **Recomputed per subcategory:** PCA components, KMeans clusters, nearest-neighbor competitor
  prices, and the DoubleML elasticities. Only the embeddings are shared.
- **No GPU work needed** — this design runs end-to-end locally.

Each subcategory folder has its own README with the exact notebook order.

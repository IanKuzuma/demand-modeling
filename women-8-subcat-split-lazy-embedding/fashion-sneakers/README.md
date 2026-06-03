# Subcategory Experiment: Fashion Sneakers

Filtered from Women's Shoes Size 8 dataset.

## Dataset
- Train: 430 ASINs (49020 rows)
- Val: 430 ASINs (49020 rows)
- Original train/val split preserved (no re-splitting)

## What's reused from women-8
- Prediction zips (24 files) via symlink to women-8-whole/data/predictions/
- Embeddings were trained on ALL women's ASINs, filtered during join

## What's recomputed
- PCA components (within-subcategory variance)
- KMeans clusters (within-subcategory grouping)
- Neighbor distances and weighted substitute prices
- DoubleML elasticity estimates

## Notebooks to run in order
1. data_preparation/00_part5_create_image_parquet.ipynb
2. code/01_1_create_dataset_txt_img.ipynb
3. code/01_2_create_dataset_txt.ipynb
4. code/02_cluster_centroid_products.ipynb (optional visualization)
5. code/03_1_predictive_performance_txt_img.ipynb
6. code/03_2_predictive_performance_txt.ipynb
7. code/04_evaluation.ipynb
8. code/04_evaluation_v2_delta.ipynb

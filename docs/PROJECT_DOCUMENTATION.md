# Demand Modeling Pipeline: Amazon Men's Shoes Size 8

## Replication of Bach et al. (2025) "Adventures in Demand Analysis Using AI"

This document provides a complete, end-to-end explanation of the demand modeling pipeline applied to **Amazon Men's Shoes (Size 8)**. This is a parallel experiment to the Women's Shoes version built by a collaborator.

> **Consolidated results across all experiments** (men/women × whole-gender / lazy / proper subcategory splits, ϑ-converted, dual-variant) are in [`../summary/SUMMARY.md`](../summary/SUMMARY.md) and [`../summary/00_summary.ipynb`](../summary/00_summary.ipynb).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How This Differs from the Women's Version](#2-how-this-differs-from-the-womens-version)
3. [Data Preparation Phase](#3-data-preparation-phase)
4. [Embedding Training Phase (GPU)](#4-embedding-training-phase-gpu)
5. [Verification and Image Processing](#5-verification-and-image-processing)
6. [Analysis Phase](#6-analysis-phase)
7. [Evaluation Phase (DoubleML)](#7-evaluation-phase-doubleml)
8. [Complete File Reference](#8-complete-file-reference)
9. [Subcategory Experiments + Method Updates (May 2026)](#9-subcategory-experiments--method-updates-may-2026)

---

## 1. Project Overview

The goal of this project is to estimate **price elasticity of demand** for Amazon men's shoes using modern AI embeddings. The paper by Bach et al. (2025) proposes using multimodal embeddings (text descriptions, product images, and tabular features) as control variables in a Double Machine Learning (DoubleML) framework to get better causal estimates of how price changes affect sales rank.

**In plain English:** We want to know "if a shoe's price goes up by 1%, how much does its sales rank change?" But simply looking at price vs. sales rank is misleading because many other factors (brand popularity, shoe type, reviews, etc.) affect both. The AI embeddings capture these confounding factors so we can isolate the true causal effect of price on demand.

### Pipeline Summary

```
Raw Data (7,761 ASINs)
  → Filter to Men only (2,278)
  → Clean & forward-fill (2,061)
  → 50/50 train/val split
  → Download product images
  → Train 4 embedding model variants (GPU)
  → Build analysis datasets with features
  → Evaluate predictive performance
  → Estimate price elasticity via DoubleML
```

---

## 2. How This Differs from the Women's Version

| Aspect | Women's (Friend's) | Men's (Ours) |
|--------|-------------------|--------------|
| Gender filter | `gender == 'Women'` | `gender == 'Men'` |
| ASINs after cleaning | 5,147 | 2,061 |
| Top 10 subcategories | Hardcoded (Pumps, Flats, Heeled Sandals, etc.) | Dynamically computed from data |
| Top subcategories | Pumps, Flats, Fashion Sneakers, Loafers & Slip-Ons, Walking, Mules & Clogs, Road Running, Heeled Sandals, Ballet & Dance, Ankle & Bootie | Loafers & Slip-Ons, Fashion Sneakers, Oxfords, Walking, Road Running, Trail Running, Water Shoes, Golf, Shoes, Track & Field & Cross Country |
| Key CSV location | `data_preparation/` folder | `code/` folder |
| Execution environment | Google Colab | Local machine + Northeastern RC cluster (H200 GPUs) |
| Prediction file dates | 2026-03-31 to 2026-04-02 | 2026-04-13 to 2026-04-14 |
| Paths | Absolute Google Drive paths | Relative paths with PROJECT_ROOT detection |

**Key observation:** The men's shoe market has noticeably different characteristics. Where the women's side has categories like Pumps and Heeled Sandals, the men's side is dominated by Loafers, Oxfords, and various athletic shoe categories. This affects the clustering results and potentially the elasticity estimates.

---

## 3. Data Preparation Phase

### 3.1 Notebook: `00_part1_data_preparation.ipynb`

**Purpose:** Load raw Amazon product data, filter to men's shoes only, clean the dataset, and create train/validation splits.

**Input:** `data/amazon_shoes_8_combined.parquet` (all shoes, size 8, all genders)

**Step-by-step:**

1. **Load raw data:** 1,008,930 rows, 7,761 unique ASINs, date range 2025-01-06 to 2026-03-30
   - Gender breakdown: Women (5,472), Men (2,279), Costumes & Accessories (9), Shoe/Jewelry Accessories (1)

2. **Remove dirty ASINs (11 total):**
   - 10 from wrong gender categories (Costumes & Accessories, Shoe/Jewelry & Watch Accessories)
   - 1 wrong size (B0DM7J3W65, Men's size 7 instead of 8)

3. **Filter to Men only:** 7,761 → 2,278 ASINs (296,140 rows)

4. **Drop dead columns:** Removed 9 columns that are no longer needed after filtering (SALES_RANK_original, PRICE_original, BUYBOX_PRICE_original, gender, manufacturer, brand, model, color, size)

5. **Fix data types:** Fill NaN in FBA/FBM offer counts with 0 (263,285 and 288,406 NaN values respectively, because no Keepa offer data means 0 active offers)

6. **Compute Top 10 subcategories dynamically:**
   - Loafers & Slip-Ons: 396 ASINs (17.4%)
   - Fashion Sneakers: 384 (16.9%)
   - Oxfords: 269 (11.8%)
   - Walking: 229 (10.1%)
   - Road Running: 204 (9.0%)
   - Trail Running: 98 (4.3%)
   - Water Shoes: 92 (4.0%)
   - Golf: 75 (3.3%)
   - Shoes: 69 (3.0%)
   - Track & Field & Cross Country: 61 (2.7%)
   - Other: 401 (17.6%)

7. **Forward fill missing values:** Uses Jan/Feb 2025 data as a "seed" to fill March onwards. BUYBOX_PRICE also gets backward-filled since some products gained a Buy Box mid-window.
   - Before fill: SALES_RANK 2.13% NaN, PRICE 5.81%, BUYBOX_PRICE 23.77%
   - After fill: SALES_RANK 1.55%, PRICE 1.76%, BUYBOX_PRICE 2.46%

8. **Trim to analysis window:** Remove Jan/Feb seed rows. 296,140 → 259,692 rows (2025-03-03 to 2026-03-30)

9. **Drop incomplete ASINs (217 total, 9.5%):**
   - SALES_RANK: 53 ASINs still had NaN
   - PRICE: 80 ASINs
   - BUYBOX_PRICE: 56 ASINs (100% missing, never had a Buy Box)
   - RATING: 97 ASINs
   - REVIEW_COUNT: 97 ASINs

10. **Final dataset:** 2,061 ASINs, 234,954 rows, 22 columns, zero NaN

11. **Stratified 50/50 split** (seed=42, stratified by subcat_aggregated):
    - Train: 1,029 ASINs (117,306 rows)
    - Validation: 1,032 ASINs (117,648 rows)
    - Zero ASIN overlap between splits

**Outputs:**
| File | Location | Description |
|------|----------|-------------|
| `train-00000-of-00001.parquet` | `data/amzn_shoes_monthly_diffs_ffill_fixed_splits/` | Training split (117,306 rows, 17 columns) |
| `validation-00000-of-00001.parquet` | Same directory | Validation split (117,648 rows, 17 columns) |
| `main_train_keys.csv` | `code/` | ASIN + date pairs for train split (used by code notebooks) |
| `main_val_keys.csv` | `code/` | ASIN + date pairs for val split |

**Column schema (17 columns):**
ASIN, window, date, SALES_RANK, PRICE, BUYBOX_PRICE, text, RATING, REVIEW_COUNT, subcat, subcat_aggregated, New Offer Count: Current, Count of retrieved live offers: New FBA, Count of retrieved live offers: New FBM, Lightning Deals: Upcoming Deal, Buy Box: Is FBA, image

---

### 3.2 Notebook: `00_part2_download_images.ipynb`

**Purpose:** Download one product image per ASIN from Amazon's CDN.

**Input:** Train and validation parquets (for the image URLs in the `image` column)

**Process:**
- 2,061 unique ASINs, all had image URLs (0 missing)
- Downloads each image, converts to RGB JPEG
- 0.2 second sleep between downloads to be polite to Amazon's servers
- 3 retries per image with 10-second timeout

**Results:** 2,061 of 2,061 downloaded (100% coverage), 0 failures, 471.6 MB total

**Output:** `data/images/{ASIN}.jpg` (one file per product, e.g., `data/images/B0007T5JCI.jpg`)

---

## 4. Embedding Training Phase (GPU)

This is the most computationally expensive phase. Four notebooks train deep learning models to produce embeddings (dense vector representations) that capture product characteristics from text, images, and tabular data.

### What are embeddings?

An embedding is a list of numbers (128 or 256 values) that represents a product in a high-dimensional space. Products that are similar (in text description, visual appearance, or sales characteristics) will have similar embeddings. These embeddings later serve as control variables in the causal analysis.

### Model Architecture (Bach et al. Appendix G)

The multimodal model has three encoders fused via Cross-Attention Blocks:
- **Text encoder:** RoBERTa (cardiffnlp/twitter-roberta-base) processes the product description text
- **Image encoder:** BEiT (microsoft/beit-base-patch16-224) processes the product photo
- **Tabular encoder:** SAINT processes numerical features (RATING, REVIEW_COUNT, offer counts, etc.)

These three representations are fused using all-to-all Cross-Attention, projected to an embedding vector, then two heads predict sales rank (Q) and price (P).

The text-only model uses just RoBERTa, no image or tabular encoder.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Level model epochs | 25 |
| Diff model epochs | 15 |
| Learning rate | 2e-5 |
| Embedding dimensions | 128 and 256 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealing |
| Mixed precision | FP16 (torch.amp) |
| Window | 28 (28-day rolling average) |
| Mod | 4 (every 4th time period) |

### 4.1 Notebook: `00_part3a_train_txtimg_lag1.ipynb`

**Variant:** Text + Image + Tabular, with lag-1 features (previous period's sales rank and price as additional inputs)

**Trains 3 models:**
1. Level model, emb_dim=128 (25 epochs) → best epoch 20, val_loss=0.4017
2. Level model, emb_dim=256 (25 epochs) → best epoch 22, val_loss=0.3603
3. Diff model, emb_dim=128 (15 epochs) → best epoch 14, val_loss=0.2206

**Outputs (6 zip files):** `data/predictions/txtimg/lag1/`

Each zip contains a CSV with columns: index (ASIN), time (date), pred_ml_l (predicted sales rank), pred_ml_m (predicted price), and 128 or 256 embedding columns.

| File | Description |
|------|-------------|
| `train_pred_model-..._dim=128_...zip` | Train predictions + 128-dim embeddings from level model |
| `val_pred_model-..._dim=128_...zip` | Val predictions + 128-dim embeddings from level model |
| `train_pred_model-..._dim=256_...zip` | Train predictions + 256-dim embeddings from level model |
| `val_pred_model-..._dim=256_...zip` | Val predictions + 256-dim embeddings from level model |
| `pred_diff_..._train.zip` | Train predictions from first-difference model |
| `pred_diff_..._val.zip` | Val predictions from first-difference model |

### 4.2 Notebook: `00_part3b_train_txtimg_time_ind.ipynb`

**Variant:** Text + Image + Tabular, time-independent (no lag features)

Same architecture but without lag-1 features. This tests whether the model can predict demand without knowing last period's values.

- Level 128: best epoch 16, val_loss=0.7022
- Level 256: best epoch 11, val_loss=0.6917
- Diff: best epoch 10, val_loss=0.2414

**Outputs (6 zip files):** `data/predictions/txtimg/time_indipendent/`

### 4.3 Notebook: `00_part3c_train_txt_lag1.ipynb`

**Variant:** Text only (RoBERTa), with lag-1 features

Uses only the text encoder, no image or tabular data. Tests how much text alone can capture.

- Level 128: best epoch 23, val_loss=0.3149
- Level 256: best epoch 17, val_loss=0.2980
- Diff: best epoch 12, val_loss=0.2209

**Outputs (6 zip files):** `data/predictions/txt/lag1/`

### 4.4 Notebook: `00_part3d_train_txt_time_ind.ipynb`

**Variant:** Text only (RoBERTa), time-independent

- Level 128: best epoch 11, val_loss=0.8651
- Level 256: best epoch 14, val_loss=0.8576
- Diff: best epoch 2, val_loss=0.2496

**Outputs (6 zip files):** `data/predictions/txt/time_indipendent/`

### Understanding the File Names

Example: `train_pred_model-2026-04-14-embdim=None_lag1_txtimg_mod4-epoch=022-val_combined_loss=0.3603_txtimg_dim=256_proj_emb_step=.zip`

Breaking it down:
- `train_pred_model` → predictions for the training split
- `2026-04-14` → date the model was trained
- `lag1` → uses lag-1 features
- `txtimg` → multimodal (text + image + tabular)
- `mod4` → every 4th time period used
- `epoch=022` → best checkpoint was at epoch 22
- `val_combined_loss=0.3603` → validation loss at best epoch
- `dim=256` → 256-dimensional embeddings

Example: `pred_diff_shoes-diff-model-txtimg_2026-04-14-embdim=768_lag1-epoch=014-val_combined_loss=0.2206_txtimg_lag1_train.zip`

- `pred_diff` → predictions from the first-difference model
- `diff-model` → trained on changes (deltas) rather than levels
- `embdim=768` → internal encoder dimension (not the output embedding size)

---

## 5. Verification and Image Processing

### 5.1 Notebook: `00_part4_verify.ipynb`

**Purpose:** Verify all 24 prediction zip files exist and are correctly named, then generate `paths_config.yaml`.

**Results:** All 24 files found and verified. The config file maps each model variant to its file path, used by the code notebooks to load the right predictions.

**Output:** `code/utils/paths_config.yaml`

### 5.2 Notebook: `00_part5_create_image_parquet.ipynb`

**Purpose:** Package all product images into Parquet shards for efficient loading in the cluster visualization notebook (02).

**Process:** Reads each ASIN's JPEG from `data/images/`, embeds the raw bytes into a Parquet file. Creates 4 shards x 4 parts = 16 files (~25 MB each).

**Why not just use the JPEGs directly?** The HuggingFace `datasets` library (used in notebook 02) expects data in Parquet format. Packaging images this way allows notebook 02 to load product photos alongside embeddings using the same API.

**Output:** 16 parquet files in `data/amzn_shoes_monthly_avg_long_ffill_28_7_images/`
- Format: `train-0000{i}-of-00004-part00{j}.parquet` where i=0..3, j=0..3
- Total: 2,061 ASINs across all shards

---

## 6. Analysis Phase

### 6.1 Notebook: `01_1_create_dataset_txt_img.ipynb`

**Purpose:** Build the final analysis-ready dataset using multimodal (text+image) embeddings.

**Pipeline:**
1. Load cleaned panel data (train + val parquets)
2. Load 256-dim embeddings from the prediction zips
3. Center and normalize embeddings (subtract mean, divide by std)
4. Compute **PCA features** (5 principal components) to reduce embedding dimensionality
5. Compute **cluster similarity features** (cosine similarity to 5 KMeans cluster centroids)
6. Compute **neighbor distances** (5 nearest neighbors per product in embedding space)
7. Compute **weighted substitute prices** (distance-weighted average price of 5 nearest neighbors, lagged one period). This is the BLP-style instrumental variable for DoubleML.
8. Join all prediction outputs (level + diff, time-independent + lag1)
9. Add lag features (Q_t-1, P_bb_t-1, etc.) and first differences (Delta_Q_t, Delta_P_bb_t)

**What is the weighted substitute price?** It's an instrumental variable. The idea: a product's nearest neighbors' prices affect that product's demand (because consumers can switch to similar shoes), but they don't directly cause a specific shoe's price to change. This makes it a valid instrument for isolating the causal effect of price.

**Output:**
| File | Location | Description |
|------|----------|-------------|
| `dataset_txt_only_False_embeddings_True_train.zip` | `data/` | Training dataset with multimodal embeddings |
| `dataset_txt_only_False_embeddings_True_val.zip` | `data/` | Validation dataset with multimodal embeddings |

### 6.2 Notebook: `01_2_create_dataset_txt.ipynb`

**Purpose:** Same as 01_1 but using text-only embeddings (no image features).

**Output:**
| File | Location |
|------|----------|
| `dataset_txt_only_True_embeddings_True_train.zip` | `data/` |
| `dataset_txt_only_True_embeddings_True_val.zip` | `data/` |

### 6.3 Notebooks: `02_cluster_centroid_products.ipynb` and `02_cluster_centroid_products_random20.ipynb`

**Purpose:** Visualize what the embedding clusters look like. Identifies the products closest to each cluster centroid and displays their images and descriptions.

**Process:**
1. Load both multimodal and text-only embeddings
2. Run KMeans clustering (k=5) on both embedding types
3. PCA visualization in 2D and 3D
4. Elbow test (explored k=2 to k=11, k=5 confirmed as paper standard)
5. Find 4 nearest products to each cluster centroid
6. Create image collages for each cluster

**Output:** Visualization PNGs in `output/02_cluster_centroid_products/` and `output/02_cluster_centroid_products_random20/`
- `elbow_test.png` - Shows optimal cluster count
- `pca_2d_clusters.png` - 2D PCA visualization of clusters
- `pca_img_clusters.png` - Image embedding clusters
- `collage_cluster_img_{0-4}.png` - Product photo collages per cluster (multimodal)
- `collage_cluster_txt_{0-4}.png` - Product photo collages per cluster (text-only)

### 6.4 Notebook: `03_1_predictive_performance_txt_img.ipynb`

**Purpose:** Evaluate how well different feature sets predict sales rank (Q) and price (P) using LightGBM and OLS. This shows the value added by embeddings.

**Key results (Test R², multimodal):**

| Feature Set | R² Sales Rank | R² Price |
|------------|---------------|----------|
| OLS (Tabular only) | 30.68% | 30.11% |
| Boosting (Tabular only) | 58.52% | 49.42% |
| OLS (Tabular + PCA) | 55.95% | 66.23% |
| Boosting (Tabular + PCA) | 70.27% | 74.81% |
| Boosting (Tabular + Similarities) | 69.03% | 73.35% |
| Boosting (Tabular + Embeddings) | 70.63% | 76.75% |
| Deep model (time-independent) | 65.17% | 74.58% |
| Deep model (lag1) | 82.59% | 75.08% |

**Interpretation:** Adding embeddings (PCA, similarities, or raw) dramatically improves prediction. The deep model with lag-1 features achieves the best sales rank prediction (82.59% R²). The embeddings capture product characteristics that raw tabular features miss.

### 6.5 Notebook: `03_2_predictive_performance_txt.ipynb`

**Purpose:** Same evaluation but for text-only embeddings.

**Key results (Test R², text-only):**

| Feature Set | R² Sales Rank | R² Price |
|------------|---------------|----------|
| Boosting (Tabular + PCA) | 66.03% | 73.63% |
| Boosting (Tabular + Similarities) | 64.80% | 70.22% |
| Deep model (time-independent) | 57.36% | 66.41% |
| Deep model (lag1) | 86.13% | 73.67% |

**Interpretation:** Text-only embeddings perform competitively with multimodal ones. The text-only deep model with lag1 actually achieves higher R² for sales rank (86.13% vs 82.59%), suggesting product descriptions carry strong predictive signal for men's shoes.

---

## 7. Evaluation Phase (DoubleML)

### 7.1 Notebook: `04_evaluation.ipynb`

This is the final and most important notebook. It estimates the **causal effect of price on demand** using Double Machine Learning.

### What is DoubleML?

DoubleML (Double/Debiased Machine Learning) is a method for causal inference. The core problem: if we just regress sales rank on price, we get a biased estimate because other factors (product quality, brand, etc.) affect both price and sales. DoubleML solves this in two stages:

1. **First stage:** Use ML to predict both the outcome (sales rank) and the treatment (price) from control variables (embeddings, lagged values, etc.). Save the residuals.
2. **Second stage:** Regress the outcome residuals on the treatment residuals. This removes the confounding bias.

The key insight: by using AI embeddings as controls, we capture rich product characteristics that traditional variables might miss.

### OLS Baseline Models

The notebook first runs traditional OLS regressions as baselines:

**Simple OLS (no lag):** Adjusted R² = 0.3345
- Price coefficient: -0.312 (significant, p<0.001)
- Interpretation: Each $1 increase in price is associated with a 0.312 decrease in log sales rank

**OLS with lag (Q_t-1):** Adjusted R² = 0.8831
- Price coefficient: -0.119 (significant, p<0.001)
- Lagged sales rank: 0.896 (highly significant, dominates the model)

**OLS with cluster similarities:** Adjusted R² = 0.8876
- Price coefficient: -0.129 (significant)
- Some clusters show significant effects (cluster 0: -5.231, p=0.048)

**OLS with interactions:** Adjusted R² = 0.892
- Price coefficient: -0.140 (significant)
- Several interaction terms significant

### DoubleML (PLR) Results

The Partially Linear Regression (PLR) model uses LightGBM in the first stage:

| Specification | Elasticity Estimate | 95% CI | Significant? |
|--------------|-------------------|--------|-------------|
| PLR + Embeddings | -0.1043 | [-0.164, -0.045] | Yes |
| PLR + Controls | -0.0930 | [-0.149, -0.037] | Yes |
| PLR + Embeddings + Controls | -0.0947 | [-0.143, -0.047] | Yes |
| PLR + Similarities + Controls | -0.0843 | [-0.130, -0.039] | Yes |

**Main finding: The estimated price elasticity is approximately -0.095.** This means a 1% increase in price leads to roughly a 0.095% worsening (increase) in sales rank. All specifications produce significant negative elasticity estimates, which is consistent with economic theory (higher price → lower demand).

### CATE Analysis (Heterogeneous Effects)

CATE = Conditional Average Treatment Effect. This asks: "Does the price effect vary across different types of products?"

**Overall CATE statistics:**
- Mean: -0.2127
- Std: 0.1254
- Min: -0.6646 (most elastic product)
- Max: 0.2437 (some products show perverse elasticity)

**CATE by cluster:**

| Cluster | n | Mean CATE | Mean Price | Interpretation |
|---------|---|-----------|------------|---------------|
| 0 | 2,384 | -0.124 | $30.94 | Least price-sensitive (budget walking/sneakers) |
| 1 | 3,300 | -0.148 | $42.10 | Moderate sensitivity (loafers, dress shoes) |
| 2 | 1,746 | -0.343 | $87.15 | Highly price-sensitive (premium athletic) |
| 3 | 3,259 | -0.193 | $39.27 | Moderate sensitivity (mixed) |
| 4 | 1,695 | -0.368 | $93.59 | Most price-sensitive (premium performance) |

**Key insight:** Higher-priced product clusters (2 and 4, averaging $87-94) are much more price-elastic than budget clusters (0, averaging $31). This makes economic sense: consumers shopping for premium shoes are more likely to comparison-shop and switch products when prices change.

### Output Visualizations

All saved in `output/04_evaluation/`:

| File | Description |
|------|-------------|
| `linear_ate_estimates.png` | Comparison of OLS elasticity estimates across specifications |
| `plr_ate_estimates.png` | DoubleML PLR elasticity estimates with confidence intervals |
| `forest_plot_dml_elasticity.png` | Forest plot comparing all elasticity specifications |
| `cate_distribution.png` | Histogram of product-level CATE values |
| `cate_scatter.png` | CATE vs. predicted sales rank, colored by magnitude |
| `cate_scatter_per_cluster.png` | CATE scatter plot, colored by cluster membership |
| `gates_elasticity_by_cluster.png` | GATES (Group Average Treatment Effects) by cluster |
| `cluster2_vs4_pca.png` | PCA visualization comparing the two most elastic clusters |
| `cluster2_vs4_subcat.png` | Subcategory distribution in clusters 2 vs 4 |
| `cluster2_vs4_tabular.png` | Tabular feature comparison for clusters 2 vs 4 |
| `cluster_and_level_cate_estimates.png` | Combined cluster + level CATE estimates |
| `cluster_sorted_effects.png` | Clusters ranked by average elasticity |
| `scatter_P_vs_Q_elasticity.png` | Price vs. sales rank colored by elasticity |

---

## 8. Complete File Reference

### Directory Structure

```
demand-modeling-data-men-8/
├── data/
│   ├── amazon_shoes_8_combined.parquet          # Raw input (all genders)
│   ├── amzn_shoes_monthly_diffs_ffill_fixed_splits/
│   │   ├── train-00000-of-00001.parquet         # Clean train split
│   │   └── validation-00000-of-00001.parquet    # Clean val split
│   ├── images/                                   # 2,061 product JPEGs
│   ├── amzn_shoes_monthly_avg_long_ffill_28_7_images/  # 16 image parquet shards
│   ├── predictions/
│   │   ├── txt/lag1/                   (6 zips)  # Text-only, lag1
│   │   ├── txt/time_indipendent/       (6 zips)  # Text-only, time-independent
│   │   ├── txtimg/lag1/                (6 zips)  # Multimodal, lag1
│   │   └── txtimg/time_indipendent/    (6 zips)  # Multimodal, time-independent
│   ├── dataset_txt_only_False_embeddings_True_train.zip  # Analysis dataset (multimodal)
│   ├── dataset_txt_only_False_embeddings_True_val.zip
│   ├── dataset_txt_only_True_embeddings_True_train.zip   # Analysis dataset (text-only)
│   └── dataset_txt_only_True_embeddings_True_val.zip
├── data_preparation/
│   ├── 00_part1_data_preparation.ipynb
│   ├── 00_part2_download_images.ipynb
│   ├── 00_part3a_train_txtimg_lag1.ipynb
│   ├── 00_part3b_train_txtimg_time_ind.ipynb
│   ├── 00_part3c_train_txt_lag1.ipynb
│   ├── 00_part3d_train_txt_time_ind.ipynb
│   ├── 00_part4_verify.ipynb
│   └── 00_part5_create_image_parquet.ipynb
├── code/
│   ├── 01_1_create_dataset_txt_img.ipynb
│   ├── 01_2_create_dataset_txt.ipynb
│   ├── 02_cluster_centroid_products.ipynb
│   ├── 02_cluster_centroid_products_random20.ipynb
│   ├── 03_1_predictive_performance_txt_img.ipynb
│   ├── 03_2_predictive_performance_txt.ipynb
│   ├── 04_evaluation.ipynb
│   ├── main_train_keys.csv
│   ├── main_val_keys.csv
│   └── utils/
│       ├── paths_config.yaml
│       ├── utils_data2.py
│       └── utils_models.py
├── output/
│   ├── 02_cluster_centroid_products/      # Cluster visualization PNGs
│   ├── 02_cluster_centroid_products_random20/
│   ├── 04_evaluation/                     # 13 evaluation plots
│   └── pipeline_flowchart.png             # Pipeline overview diagram
└── PROJECT_DOCUMENTATION.md               # This file
```

### Pipeline Dependency Chain

```
part1 → part2 → part3a ─┐
                 part3b ─┤
                 part3c ─┤→ part4 → part5 → 01_1 ─┐→ 02 ─┐
                 part3d ─┘                  01_2 ─┘      ├→ 03_1 ─┐
                                                         │  03_2 ─┤→ 04
                                                         └────────┘
```

---

## 9. Subcategory Experiments + Method Updates (May 2026)

This section documents work added after the original men-8 / women-8 pipelines: the repo
restructure, the **subcategory experiments** (asking whether elasticity differs *within* a
gender, e.g. Pumps vs Flats), and three method changes the advisors requested. The original
pipeline above is unchanged; this is the consolidated map of what is new and why.

### 9.1 Repository layout (restructured)

The project now lives in `demand_modeling/` with one self-contained folder per experiment.
Each folder has the usual `data/ code/ data_preparation/ output/` layout and runs the same
pipeline; the originals already have trained embeddings, the new folders reuse or retrain them.

```
demand_modeling/
├── men-8/                                     original full pipeline (men)
├── women-8/                                   original full pipeline (women)
├── men-8-subcat-split-lazy-embedding/     LAZY  — symlink men-8 predictions
├── women-8-subcat-split-lazy-embedding/   LAZY  — symlink women-8 predictions
├── men-8-subcat-split-proper-embedding/     PROPER — embeddings trained within subcat (RC)
├── women-8-subcat-split-proper-embedding/   PROPER — embeddings trained within subcat (RC)
└── PROJECT_DOCUMENTATION.md
```

Subcategories (filtered by `subcat_aggregated`, original train/val split preserved, zero ASIN
overlap): **men** = Loafers & Slip-Ons (180 train), Fashion Sneakers (172), Oxfords (130);
**women** = Pumps (626), Flats (496), Fashion Sneakers (430).

### 9.2 The "lazy vs proper" experiment design

The question is whether an embedding model trained on the *whole* gender captures
subcategory-specific demand structure, or whether you must train *within* each subcategory.
We build both, side by side, per gender:

| Variant | Folder suffix | Embeddings | Runs where |
|---|---|---|---|
| **LAZY** | `-lazy-embedding` | reuse the whole-gender embeddings (24 prediction zips symlinked) | local |
| **PROPER** | `-proper-embedding` | retrain the encoders *within each subcategory* | RC (GPU) |

Both keep the original train/val split (no re-splitting → no leakage) and recompute PCA,
clusters, neighbor prices, and the DoubleML elasticities within the subcategory. LAZY is the
cheap shortcut (embeddings already exist); PROPER is the real test of within-subcat adaptation.

### 9.3 Within-subcat embeddings use LoRA (not freeze, not full fine-tune)

The PROPER `00_part3a–3d` notebooks were reworked to fine-tune the encoders with **LoRA**
(Low-Rank Adaptation, Hu et al. 2021; HuggingFace `peft`). The reason is sample size:

- Each subcategory has only ~130–630 training products, but RoBERTa + BEiT carry ~200M params.
- **Full fine-tuning** of all 200M params on 130 products → memorization/overfit.
- **Freezing** the encoders → every subcategory shares identical encoders, so "within-subcat"
  embeddings would be indistinguishable from the whole-dataset ones — the experiment would
  test nothing.
- **LoRA** freezes the base weights and learns small low-rank deltas on the attention
  **query/value** projections (`r=8, α=16, dropout=0.1`), so only ~1% of encoder params train.
  Capacity is matched to sample size while the encoders still genuinely adapt per subcategory.
  Everything downstream (SAINT, cross-attention fusion, projection, prediction heads) trains
  normally. This is also the regularization the advisor asked for. **RC prerequisite:**
  `pip install peft` in the `demand_modeling` conda env.

**Comparability caveat.** The whole-gender (LAZY) embeddings were *fully* fine-tuned while the
within-subcat (PROPER) embeddings are *LoRA*-fine-tuned, so lazy-vs-proper is not a perfectly
method-identical A/B. It is a deliberate, far milder compromise than freezing, and full
fine-tuning on 130–630 products is not viable. We flag this when interpreting results.

### 9.4 The `00_part3` notebooks survive the RC 2-hour GPU cap

RC `gpu-interactive` sessions are capped at ~2 hours. Each part3 notebook was made resumable:
per-epoch checkpoints to `data/checkpoints/` (home filesystem, not `/tmp`), full resume state
(model/optimizer/scheduler/epoch/best), **skip-completed** units whose output zips already
exist, **early stopping** (`PATIENCE`, helps tiny subcats), and a **graceful time-budget stop**
(`WALLCLOCK_MIN`, default 110): after each epoch it checks the clock and, if over budget, saves
state and exits cleanly — never mid-epoch. A long run becomes: launch → auto-stops at the cap →
relaunch → continues, with no lost work. Each notebook trains this gender/variant's level models
(emb_dim 128 and 256) plus the first-difference model → 6 prediction zips per variant.

### 9.5 The ϑ rank→demand conversion (`04_evaluation`)

Our DoubleML outcome is *negative log sales rank*, a **proxy** for demand, so the coefficients
are **rank**-elasticities. He & Hollenbeck (2020) show rank and quantity follow a Pareto law,
`log E[Q] ≈ C − (1/ϑ)·log(rank)`, so a rank-elasticity becomes a **demand (quantity)**
elasticity when multiplied by `1/ϑ`. Their "Clothing, Shoes & Jewelry" category gives
`ϑ ≈ 0.605`, so we use `ϑ = 0.6` → scale by `1/0.6 ≈ 1.667` (this is Victor et al. 2025,
Remark 1, previously unimplemented). Every table and forest plot now reports **both** the raw
rank-coefficient and the converted demand elasticity. This is what addresses the advisor's
"estimates look too inelastic" comment — the converted numbers are ~1.67× larger in magnitude.

### 9.6 `04_evaluation` is now dual-variant

`04_evaluation.ipynb` (which supersedes the old `04_evaluation` and `04_evaluation_v2_delta`)
wraps the whole DoubleML evaluation in `run_variant(txt_only)` and runs it for **both** the
text+image and text-only embeddings, then compares them: a combined elasticity table
(specification × variant, raw and ϑ-converted), an overlaid comparison forest plot, per-variant
plots under `output/04_evaluation/{txtimg,txt}/`, and a CATE / heterogeneity comparison table.
It uses delta (first-difference) outcome/treatment and embedding-similarity competitor prices,
fits five DoubleML PLR specifications, and clusters standard errors by ASIN. Validated on
women-8: the main spec (PLR + Emb + Controls, txt+img) reproduces the prior result, rank coef
**−0.103** → demand elasticity **−0.172**; the text-only variant is weaker (rank −0.076),
indicating the product images carry demand-relevant information.

> Note: the original notebook passed `lr=0.02` to `LGBMRegressor`, which is **not** a LightGBM
> parameter name (it is `learning_rate`) so it was silently ignored and the models ran at the
> default `learning_rate=0.1`. The reworked notebook makes `learning_rate=0.1` explicit so the
> results reproduce and the config is honest.

### 9.7 How to run

- **LAZY (local):** per subcat run `data_preparation/00_part5` → `code/01_1 → 01_2 → 02* →
  03_1 → 03_2 → 04_evaluation`. Embeddings come from the symlinked whole-gender predictions.
- **PROPER (RC then local):** on the RC, run the resumable `00_part3a–3d` (relaunch to resume),
  then scp the prediction zips into `data/predictions/`. Then `00_part4_verify` (regenerates
  `paths_config.yaml` from the new zips) → `00_part5` → `01_1 → 01_2 → 02* → 03_1 → 03_2 →
  04_evaluation`.
- **Originals (men-8 / women-8):** embeddings already exist; just re-run `04_evaluation` for the
  ϑ-corrected, dual-variant numbers.

The payoff: for each gender × subcategory, the LAZY (whole-gender) vs PROPER (within-subcat)
embedding elasticity, each reported as both a rank-coefficient and a ϑ-converted demand
elasticity, for both text-only and text+image.

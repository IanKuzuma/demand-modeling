"""
Generate a pipeline flowchart for the demand modeling project.
Replicates Bach et al. (2025) for Amazon Men's Shoes Size 8.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── Colors ──────────────────────────────────────────────────────────────
C_DATAPREP = "#CCE5FF"       # light blue - data prep
C_DATAPREP_EDGE = "#4A90D9"
C_EMBED = "#D4EDDA"          # light green - embedding training
C_EMBED_EDGE = "#28A745"
C_ANALYSIS = "#FFE8CC"       # light orange - analysis
C_ANALYSIS_EDGE = "#E67E22"
C_EVAL = "#F8D7DA"           # light red - final evaluation
C_EVAL_EDGE = "#DC3545"
C_PHASE_BG_DATA = "#EAF2FB"
C_PHASE_BG_ANALYSIS = "#FFF5EB"
C_ARROW = "#555555"

# ── Figure setup ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 30))
ax.set_xlim(0, 20)
ax.set_ylim(0, 30)
ax.axis("off")
fig.patch.set_facecolor("white")


# ── Helpers ─────────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, title, subtitle, facecolor, edgecolor, fontsize=10):
    """Draw a rounded box. Returns (cx_bot, y_bot, cx_top, y_top)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h * 0.62, title,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color="#1a1a1a", zorder=4,
    )
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.22, subtitle,
            ha="center", va="center",
            fontsize=7.5, color="#555555", zorder=4,
            style="italic",
        )
    return (x + w / 2, y, x + w / 2, y + h)


def arrow(ax, x1, y1, x2, y2):
    """Straight arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5),
        zorder=2,
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYOUT  (y increases upward, 0..30)
#
# 29.0        Title
# 28.5        Data Prep phase label
# 15.2-28.2   Data Prep phase background
# 27.0-28.0   Row 1: 00_part1
# 25.3-26.3   Row 2: 00_part2
# 23.0-24.2   Row 3: 4 embedding boxes
# 21.2-22.2   Row 4: 00_part4
# 19.5-20.5   Row 5: 00_part5
#
# 17.6        Gap / phase transition
# 17.2        Analysis phase label
# 1.5-17.0    Analysis phase background
# 15.5-16.5   Row 6: 01_1, 01_2
# 13.7-14.7   Row 7: 02 cluster viz
# 11.9-12.9   Row 8: 03_1, 03_2
# 9.5-10.7    Row 9: 04 DoubleML
# 8.0-9.0     Legend row
# ══════════════════════════════════════════════════════════════════════════

# ── Title ───────────────────────────────────────────────────────────────
ax.text(
    10, 29.3,
    "Demand Modeling Pipeline: Bach et al. (2025) Replication",
    ha="center", va="center",
    fontsize=18, fontweight="bold", color="#1a1a1a", zorder=5,
)
ax.text(
    10, 28.85,
    "Amazon Men's Shoes, Size 8",
    ha="center", va="center",
    fontsize=14, color="#444444", zorder=5,
)

# ── Phase backgrounds ──────────────────────────────────────────────────
# Data preparation phase
data_bg = FancyBboxPatch(
    (0.3, 18.6), 19.4, 9.8,
    boxstyle="round,pad=0.3",
    facecolor=C_PHASE_BG_DATA, edgecolor="#B0C4DE",
    linewidth=1.5, linestyle="--", zorder=0, alpha=0.5,
)
ax.add_patch(data_bg)
ax.text(
    10, 28.1, "DATA PREPARATION PHASE",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#2C5F8A",
    zorder=1,
)

# Analysis phase
analysis_bg = FancyBboxPatch(
    (0.3, 2.5), 19.4, 14.6,
    boxstyle="round,pad=0.3",
    facecolor=C_PHASE_BG_ANALYSIS, edgecolor="#E0C8A8",
    linewidth=1.5, linestyle="--", zorder=0, alpha=0.5,
)
ax.add_patch(analysis_bg)
ax.text(
    10, 16.8, "ANALYSIS PHASE",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#8B5E3C",
    zorder=1,
)

# ══════════════════════════════════════════════════════════════════════════
# ROW 1: 00_part1
# ══════════════════════════════════════════════════════════════════════════
BW = 7.5   # standard box width
BH = 1.3   # standard box height

r1 = draw_box(
    ax, 6.25, 26.5, BW, BH,
    "00_part1: Data Cleaning & Split",
    "Raw parquet (7,761 ASINs) -> Filter Men -> Clean -> 2,061 ASINs -> 50/50 stratified split",
    C_DATAPREP, C_DATAPREP_EDGE,
)

# Right-side annotation
ax.text(14.1, 27.1, "train/val .parquet + key .csv files",
        fontsize=7, color="#777", style="italic", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# ROW 2: 00_part2
# ══════════════════════════════════════════════════════════════════════════
r2 = draw_box(
    ax, 6.25, 24.6, BW, BH,
    "00_part2: Download Product Images",
    "Download images for all 2,061 ASINs from Amazon",
    C_DATAPREP, C_DATAPREP_EDGE,
)
arrow(ax, r1[0], r1[1], r2[2], r2[3])

ax.text(14.1, 25.2, "images/ directory",
        fontsize=7, color="#777", style="italic", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# ROW 3: Embedding training (4 boxes)
# ══════════════════════════════════════════════════════════════════════════
EW = 4.2
EH = 1.45

e1 = draw_box(
    ax, 0.5, 22.4, EW, EH,
    "00_part3a: txt+img\n(lag1 features)",
    "RoBERTa+BEiT+SAINT -> 6 pred. zips",
    C_EMBED, C_EMBED_EDGE, fontsize=9,
)
e2 = draw_box(
    ax, 5.1, 22.4, EW, EH,
    "00_part3b: txt+img\n(time-independent)",
    "RoBERTa+BEiT+SAINT -> 6 pred. zips",
    C_EMBED, C_EMBED_EDGE, fontsize=9,
)
e3 = draw_box(
    ax, 9.7, 22.4, EW, EH,
    "00_part3c: txt-only\n(lag1 features)",
    "RoBERTa only -> 6 pred. zips",
    C_EMBED, C_EMBED_EDGE, fontsize=9,
)
e4 = draw_box(
    ax, 14.3, 22.4, EW, EH,
    "00_part3d: txt-only\n(time-independent)",
    "RoBERTa only -> 6 pred. zips",
    C_EMBED, C_EMBED_EDGE, fontsize=9,
)

for e in [e1, e2, e3, e4]:
    arrow(ax, r2[0], r2[1], e[2], e[3])

# ══════════════════════════════════════════════════════════════════════════
# ROW 4: 00_part4
# ══════════════════════════════════════════════════════════════════════════
r4 = draw_box(
    ax, 6.25, 20.5, BW, BH,
    "00_part4: Verify Predictions & Config",
    "Check all 24 prediction files -> Generate paths_config.yaml",
    C_DATAPREP, C_DATAPREP_EDGE,
)
for e in [e1, e2, e3, e4]:
    arrow(ax, e[0], e[1], r4[2], r4[3])

ax.text(14.1, 21.1, "paths_config.yaml",
        fontsize=7, color="#777", style="italic", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# ROW 5: 00_part5
# ══════════════════════════════════════════════════════════════════════════
r5 = draw_box(
    ax, 6.25, 19.0, BW, BH,
    "00_part5: Image Parquet Shards",
    "Create 16 image parquet shards for cluster visualization",
    C_DATAPREP, C_DATAPREP_EDGE,
)
arrow(ax, r4[0], r4[1], r5[2], r5[3])

ax.text(14.1, 19.6, "16 image .parquet shards",
        fontsize=7, color="#777", style="italic", zorder=4)

# ── Phase transition visual separator ───────────────────────────────────
# Dashed line between phases
ax.plot([2, 18], [17.9, 17.9], color="#999", lw=1, ls=":", zorder=1)

# ══════════════════════════════════════════════════════════════════════════
# ROW 6: 01_1 and 01_2 (analysis datasets)
# ══════════════════════════════════════════════════════════════════════════
AW = 8.0
AH = 1.3

a1 = draw_box(
    ax, 0.8, 15.2, AW, AH,
    "01_1: Analysis Dataset (txt+img)",
    "PCA, KMeans clustering, neighbors, weighted substitute prices",
    C_ANALYSIS, C_ANALYSIS_EDGE,
)
a2 = draw_box(
    ax, 11.2, 15.2, AW, AH,
    "01_2: Analysis Dataset (txt-only)",
    "PCA, KMeans clustering, neighbors, weighted substitute prices",
    C_ANALYSIS, C_ANALYSIS_EDGE,
)

arrow(ax, r5[0], r5[1], a1[2], a1[3])
arrow(ax, r5[0], r5[1], a2[2], a2[3])

# Output file annotations below the boxes
ax.text(4.8, 14.95, "-> analysis_men_txtimg.parquet",
        fontsize=6.8, color="#666", style="italic", zorder=4)
ax.text(15.2, 14.95, "-> analysis_men_txt.parquet",
        fontsize=6.8, color="#666", style="italic", zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# ROW 7: 02 - Cluster viz (wide box)
# ══════════════════════════════════════════════════════════════════════════
r7 = draw_box(
    ax, 3.0, 13.0, 14.0, AH,
    "02: Cluster Centroid Product Visualization",
    "KMeans centroids, PCA scatter plots, product image collages",
    C_ANALYSIS, C_ANALYSIS_EDGE,
)
arrow(ax, a1[0], a1[1], r7[2], r7[3])
arrow(ax, a2[0], a2[1], r7[2], r7[3])

# ══════════════════════════════════════════════════════════════════════════
# ROW 8: 03_1, 03_2 (predictive performance)
# ══════════════════════════════════════════════════════════════════════════
p1 = draw_box(
    ax, 0.8, 10.8, AW, AH,
    "03_1: Predictive Performance (txt+img)",
    "LightGBM R-squared comparisons across embedding types",
    C_EVAL, C_EVAL_EDGE,
)
p2 = draw_box(
    ax, 11.2, 10.8, AW, AH,
    "03_2: Predictive Performance (txt-only)",
    "LightGBM R-squared comparisons across embedding types",
    C_EVAL, C_EVAL_EDGE,
)

arrow(ax, r7[0], r7[1], p1[2], p1[3])
arrow(ax, r7[0], r7[1], p2[2], p2[3])

# ══════════════════════════════════════════════════════════════════════════
# ROW 9: 04 - DoubleML (final, larger box)
# ══════════════════════════════════════════════════════════════════════════
rf = draw_box(
    ax, 3.5, 8.5, 13.0, 1.5,
    "04: DoubleML Evaluation",
    "Price elasticity estimation, CATE analysis, GATES, policy-relevant treatment effects",
    C_EVAL, C_EVAL_EDGE, fontsize=12,
)
arrow(ax, p1[0], p1[1], rf[2], rf[3])
arrow(ax, p2[0], p2[1], rf[2], rf[3])

# ══════════════════════════════════════════════════════════════════════════
# Bottom: Summary + Legend
# ══════════════════════════════════════════════════════════════════════════

# Summary stats
ax.text(
    10, 7.5,
    "Total: 24 prediction files  |  2 analysis datasets  |  2,061 products  |  6 embedding model variants",
    ha="center", va="center",
    fontsize=9.5, color="#555", zorder=4,
)

# Legend
legend_y = 6.6
legend_x_start = 3.0
legend_items = [
    (C_DATAPREP, C_DATAPREP_EDGE, "Data Preparation"),
    (C_EMBED, C_EMBED_EDGE, "Embedding Training"),
    (C_ANALYSIS, C_ANALYSIS_EDGE, "Analysis"),
    (C_EVAL, C_EVAL_EDGE, "Evaluation"),
]

ax.text(legend_x_start, legend_y + 0.15, "Legend:", fontsize=9.5,
        fontweight="bold", color="#333", zorder=4)

for i, (fc, ec, label) in enumerate(legend_items):
    lx = legend_x_start + 2.0 + i * 3.5
    box = FancyBboxPatch(
        (lx, legend_y - 0.05), 0.6, 0.35,
        boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(lx + 0.8, legend_y + 0.12, label, fontsize=8.5,
            color="#333", va="center", zorder=4)

# ── Embedding training bracket annotation ──────────────────────────────
# A subtle bracket label for the 4 embedding boxes
ax.text(
    10, 23.95,
    "Embedding Training (4 parallel variants, 24 prediction zip files total)",
    ha="center", va="center",
    fontsize=8.5, fontweight="bold", color="#1E7A34",
    zorder=4, alpha=0.8,
)

# ── Save ────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
fig.savefig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_flowchart.png"),
    dpi=110,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()
print("Flowchart saved successfully.")

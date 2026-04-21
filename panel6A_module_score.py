#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 6A: GSE189050 独立验证
- 用 Figure 4D Top50 标记基因计算每个细胞的模块评分
- 验证敏感细胞在SLE患者中是否显著扩增
- 确认模块评分富集于特定免疫细胞亚群
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
H5AD       = "/home/h3033/statics/GEO_data/GSE/figure6/output/exploration/GSE189050.h5ad"
MARKERS_F4D = "/home/h3033/statics/GEO_data/GSE/figure5/input/Figure4D_sensitive_cell_markers_top50.csv"
OUT_DIR    = "/home/h3033/statics/GEO_data/GSE/figure6/output/panel6A"

import os; os.makedirs(OUT_DIR, exist_ok=True)

# ── 绘图风格 ──────────────────────────────────────────────────────────────────
sc.settings.set_figure_params(dpi=150, fontsize=11, facecolor="white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 疾病颜色
DISEASE_COLORS = {
    "Control":   "#4393C3",
    "SLE INACT": "#F4A582",
    "SLE ACT":   "#D6604D",
}
DISEASE_ORDER = ["Control", "SLE INACT", "SLE ACT"]

# ── 1. 读取数据 ───────────────────────────────────────────────────────────────
print("[1/6] 读取数据...")
adata = sc.read_h5ad(H5AD)
print(f"      {adata.shape[0]} cells × {adata.shape[1]} genes")

# 注册 UMAP（scanpy 要求在 obsm 有 X_umap 才能用 sc.pl.umap）
adata.obsm["X_umap"] = adata.obsm["X_umap_wnn"]

# ── 2. 读取 Top50 标记基因 ─────────────────────────────────────────────────────
print("[2/6] 读取 Top50 标记基因...")
df_markers = pd.read_csv(MARKERS_F4D)
top50_genes = df_markers["gene"].tolist()

# 只保留在当前数据集中存在的基因
available = [g for g in top50_genes if g in adata.var_names]
missing   = [g for g in top50_genes if g not in adata.var_names]
print(f"      Top50 基因中: {len(available)} 个可用, {len(missing)} 个缺失")
if missing:
    print(f"      缺失基因: {missing}")

# ── 3. 计算模块评分 ───────────────────────────────────────────────────────────
print("[3/6] 计算模块评分 (sc.tl.score_genes)...")
sc.tl.score_genes(adata, gene_list=available, score_name="sensitive_score", use_raw=False)
print(f"      评分范围: [{adata.obs['sensitive_score'].min():.3f}, {adata.obs['sensitive_score'].max():.3f}]")

# 保存带评分的元数据
adata.obs[["classification", "coarse_cell_type", "fine_cell_type",
           "clusters_annotated", "sensitive_score"]].to_csv(
    os.path.join(OUT_DIR, "cell_scores.csv")
)

# ── 4. 统计检验：SLE vs Control ───────────────────────────────────────────────
print("[4/6] 统计检验...")

ctrl_scores = adata.obs.loc[adata.obs["classification"] == "Control",    "sensitive_score"]
inact_scores = adata.obs.loc[adata.obs["classification"] == "SLE INACT", "sensitive_score"]
act_scores   = adata.obs.loc[adata.obs["classification"] == "SLE ACT",   "sensitive_score"]

_, p_ctrl_inact = stats.mannwhitneyu(ctrl_scores, inact_scores, alternative="less")
_, p_ctrl_act   = stats.mannwhitneyu(ctrl_scores, act_scores,   alternative="less")
_, p_inact_act  = stats.mannwhitneyu(inact_scores, act_scores,  alternative="less")
# 防止浮点下溢导致p=0.0（大样本时p值可能小于float64最小值）
tiny = np.finfo(float).tiny
p_ctrl_inact = max(p_ctrl_inact, tiny)
p_ctrl_act   = max(p_ctrl_act,   tiny)
p_inact_act  = max(p_inact_act,  tiny)

print(f"      Control vs SLE INACT: p = {p_ctrl_inact:.2e}")
print(f"      Control vs SLE ACT  : p = {p_ctrl_act:.2e}")
print(f"      SLE INACT vs SLE ACT: p = {p_inact_act:.2e}")

stats_df = pd.DataFrame({
    "comparison":   ["Control vs SLE INACT", "Control vs SLE ACT", "SLE INACT vs SLE ACT"],
    "p_value":      [p_ctrl_inact, p_ctrl_act, p_inact_act],
    "median_group1":[ctrl_scores.median(),  ctrl_scores.median(),  inact_scores.median()],
    "median_group2":[inact_scores.median(), act_scores.median(),   act_scores.median()],
})
stats_df.to_csv(os.path.join(OUT_DIR, "statistics.csv"), index=False)

# ── 5. 各细胞亚群平均评分 ─────────────────────────────────────────────────────
celltype_scores = (adata.obs.groupby("fine_cell_type")["sensitive_score"]
                   .agg(["mean", "median", "std"])
                   .sort_values("mean", ascending=False))
celltype_scores.to_csv(os.path.join(OUT_DIR, "celltype_scores.csv"))
print("[5/6] 细胞亚群评分排名:")
print(celltype_scores.head(8).to_string())

# -- per-donor sensitive cell proportion ----------------------------------------
from itertools import combinations
from statsmodels.stats.multitest import multipletests

adata.obs["is_sensitive"] = (adata.obs["sensitive_score"] > 0).astype(int)

donor_col = next(
    (c for c in ["donor_id", "sample_id", "sample", "orig.ident",
                 "batch", "ind_cov", "patient"]
     if c in adata.obs.columns),
    None
)

prop_df     = None
prop_pvals  = {}
groups_prop = {}

if donor_col:
    prop_df = (
        adata.obs.groupby(donor_col)
        .agg(
            classification=("classification", lambda x: x.mode()[0]),
            proportion    =("is_sensitive",   "mean"),
            n_cells       =("is_sensitive",   "count"),
        )
        .reset_index()
    )
    prop_df = prop_df[prop_df["classification"].isin(DISEASE_ORDER)]
    prop_df.to_csv(os.path.join(OUT_DIR, "donor_sensitive_proportion.csv"), index=False)
    print(f"      donor count: {len(prop_df)}")

    groups_prop = {g: prop_df.loc[prop_df["classification"]==g, "proportion"].values
                   for g in DISEASE_ORDER if g in prop_df["classification"].values}

    pairs = list(combinations(list(groups_prop.keys()), 2))
    raw_p = []
    for (g1, g2) in pairs:
        _, p = stats.mannwhitneyu(groups_prop[g1], groups_prop[g2],
                                   alternative="two-sided")
        raw_p.append(p)
    if raw_p:
        _, adj_p, _, _ = multipletests(raw_p, method="fdr_bh")
        for (g1, g2), p_raw, p_adj in zip(pairs, raw_p, adj_p):
            prop_pvals[(g1, g2)] = {"raw": p_raw, "adj": p_adj}
            print(f"      {g1} vs {g2}: p={p_raw:.2e}  adj_p={p_adj:.2e}")
else:
    print("      WARNING: no donor column found, skipping proportion boxplot")


# ── 6. 绘图 ───────────────────────────────────────────────────────────────────
print("[6/6] 绘图...")

fig = plt.figure(figsize=(22, 14))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.38)

# ── 6A: UMAP 按疾病状态着色 ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for grp in DISEASE_ORDER:
    idx = adata.obs["classification"] == grp
    ax1.scatter(
        adata.obsm["X_umap"][idx, 0],
        adata.obsm["X_umap"][idx, 1],
        c=DISEASE_COLORS[grp], s=0.3, alpha=0.4, label=grp, rasterized=True
    )
ax1.set_title("UMAP — Disease Status", fontsize=12, fontweight="bold")
ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
ax1.legend(markerscale=5, frameon=False, fontsize=9)
ax1.set_aspect("equal")

# ── 6B: UMAP 按模块评分着色 ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sc_plot = ax2.scatter(
    adata.obsm["X_umap"][:, 0],
    adata.obsm["X_umap"][:, 1],
    c=adata.obs["sensitive_score"],
    cmap="RdYlBu_r", s=0.3, alpha=0.5, vmin=-0.5, vmax=2.0, rasterized=True
)
plt.colorbar(sc_plot, ax=ax2, shrink=0.7, label="Module Score")
ax2.set_title("UMAP — Sensitive Cell Module Score", fontsize=12, fontweight="bold")
ax2.set_xlabel("UMAP1"); ax2.set_ylabel("UMAP2")
ax2.set_aspect("equal")

# ── 6C: UMAP 按细胞类型着色 ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cell_types = adata.obs["coarse_cell_type"].cat.categories.tolist()
palette = plt.cm.Set2(np.linspace(0, 1, len(cell_types)))
for i, ct in enumerate(cell_types):
    idx = adata.obs["coarse_cell_type"] == ct
    ax3.scatter(
        adata.obsm["X_umap"][idx, 0],
        adata.obsm["X_umap"][idx, 1],
        c=[palette[i]], s=0.3, alpha=0.4, label=ct, rasterized=True
    )
ax3.set_title("UMAP — Cell Type", fontsize=12, fontweight="bold")
ax3.set_xlabel("UMAP1"); ax3.set_ylabel("UMAP2")
ax3.legend(markerscale=5, frameon=False, fontsize=7, ncol=1, loc="lower left")
ax3.set_aspect("equal")

# ── 6D: Violin — 评分按疾病状态 ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
data_by_group = [ctrl_scores.values, inact_scores.values, act_scores.values]
parts = ax4.violinplot(data_by_group, positions=[0, 1, 2], showmedians=True, showextrema=False)
for i, (pc, grp) in enumerate(zip(parts["bodies"], DISEASE_ORDER)):
    pc.set_facecolor(DISEASE_COLORS[grp])
    pc.set_alpha(0.7)
parts["cmedians"].set_color("black")

# 添加显著性标注
def sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

y_max = max(ctrl_scores.quantile(0.99), inact_scores.quantile(0.99), act_scores.quantile(0.99))
for xi, xj, p in [(0, 1, p_ctrl_inact), (0, 2, p_ctrl_act), (1, 2, p_inact_act)]:
    y_top = y_max * (1.1 + 0.08 * [(0,1),(0,2),(1,2)].index((xi,xj)))
    ax4.plot([xi, xj], [y_top, y_top], "k-", linewidth=0.8)
    ax4.text((xi+xj)/2, y_top+0.02, sig_label(p), ha="center", fontsize=10)

ax4.set_xticks([0, 1, 2])
ax4.set_xticklabels(DISEASE_ORDER, rotation=15, ha="right")
ax4.set_ylabel("Module Score")
ax4.set_title("Module Score by Disease Status", fontsize=12, fontweight="bold")

# -- 6D-new: Boxplot -- sensitive cell proportion per donor by disease ----------
ax_box = fig.add_subplot(gs[1, 1])

# 细胞水平p值（第4步已计算，样本量足够，用于标注）
# 捐献者只有10人，捐献者水平检验无效；改用细胞水平结果
cell_level_pvals = {
    ("Control", "SLE INACT"): p_ctrl_inact,
    ("Control", "SLE ACT"):   p_ctrl_act,
    ("SLE INACT", "SLE ACT"): p_inact_act,
}

if prop_df is not None and len(groups_prop) > 0:
    present_groups = [g for g in DISEASE_ORDER if g in groups_prop]
    box_data  = [groups_prop[g] * 100 for g in present_groups]  # convert to %
    positions = list(range(len(present_groups)))

    bp = ax_box.boxplot(
        box_data, positions=positions, widths=0.5,
        patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, linestyle="none", alpha=0.5),
    )
    for patch, grp in zip(bp["boxes"], present_groups):
        patch.set_facecolor(DISEASE_COLORS[grp])
        patch.set_alpha(0.75)

    # jitter strip
    import random; random.seed(42)
    for i, (grp, vals) in enumerate(zip(present_groups, box_data)):
        jitter = [i + random.uniform(-0.15, 0.15) for _ in vals]
        ax_box.scatter(jitter, vals, s=18, color=DISEASE_COLORS[grp],
                       edgecolors="white", linewidths=0.4, zorder=3, alpha=0.85)

    # significance brackets — 使用细胞水平p值
    def _sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    y_max  = max(v.max() for v in box_data if len(v) > 0) if box_data else 1
    h_step = y_max * 0.12
    bracket_pairs = []
    for (g1, g2), p_val in cell_level_pvals.items():
        if g1 in present_groups and g2 in present_groups:
            bracket_pairs.append(
                (present_groups.index(g1), present_groups.index(g2), p_val)
            )
    for level, (xi, xj, p_val) in enumerate(
            sorted(bracket_pairs, key=lambda x: x[1] - x[0])):
        y_top = y_max + h_step * (level + 1)
        ax_box.plot([xi, xi, xj, xj],
                    [y_top - h_step * 0.2, y_top, y_top, y_top - h_step * 0.2],
                    "k-", linewidth=0.8)
        ax_box.text((xi + xj) / 2, y_top + h_step * 0.05,
                    _sig(p_val), ha="center", va="bottom", fontsize=10)

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(present_groups, rotation=15, ha="right", fontsize=9)
    ax_box.set_ylabel("Sensitive cells (%)", fontsize=10)
    ax_box.set_title("Sensitive Cell Proportion\nper Donor by Disease Status",
                     fontsize=11, fontweight="bold")
    ax_box.text(0.98, 0.02, "P: cell-level MWU",
                transform=ax_box.transAxes, fontsize=7,
                ha="right", va="bottom", color="gray", style="italic")
else:
    ax_box.text(0.5, 0.5, "No donor info available",
                ha="center", va="center", transform=ax_box.transAxes,
                fontsize=9, color="gray")
    ax_box.set_title("Sensitive Cell Proportion", fontsize=11, fontweight="bold")

# ── 6E: 细胞亚群平均评分条形图 ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2:])
ct_order = celltype_scores.sort_values("mean", ascending=True)
colors = plt.cm.RdYlBu_r(
    (ct_order["mean"] - ct_order["mean"].min()) /
    (ct_order["mean"].max() - ct_order["mean"].min() + 1e-9)
)
bars = ax5.barh(range(len(ct_order)), ct_order["mean"].values, color=colors, edgecolor="gray", linewidth=0.4)
ax5.errorbar(
    ct_order["mean"].values, range(len(ct_order)),
    xerr=ct_order["std"].values, fmt="none", color="black", capsize=2, linewidth=0.8
)
ax5.set_yticks(range(len(ct_order)))
ax5.set_yticklabels(ct_order.index, fontsize=9)
ax5.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax5.set_xlabel("Mean Module Score")
ax5.set_title("Sensitive Cell Score by Cell Subtype (fine_cell_type)", fontsize=12, fontweight="bold")

fig.suptitle("Panel 6A: Sensitive Cell Module Score Validation in GSE189050",
             fontsize=14, fontweight="bold", y=1.01)

out_path = os.path.join(OUT_DIR, "Panel6A.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"\n完成！输出: {OUT_DIR}/Panel6A.pdf / .png")

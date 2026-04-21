#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 6B: GSE186476 空间转录组验证
- 定位核心标记基因（FTL、CTSS、TNFSF13B）在SLE病变组织中的空间分布
- 探究与免疫细胞浸润区域的共定位关系

数据说明:
  GSE186476_RAW.tar 包含:
    - GSM5652700~GSM5652712: Healthy (13例) scRNA-seq 10X MTX格式
    - GSM5652713~GSM5652719: Lupus-Lesional (7例)
    - GSM5652720~GSM5652726: Lupus-NonLesional (7例)
    - GSM6021876: 空间转录组 (lupus-spatial_expr_matrix.h5 + tissue_image.png.gz)

"""

import os
import tarfile
import io
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
RAW_TAR   = "/home/h3033/statics/GEO_data/GSE/figure6/GSE186476/GSE186476_RAW.tar"
DATA_DIR  = "/home/h3033/statics/GEO_data/GSE/figure6/GSE186476/extracted"
OUT_DIR   = "/home/h3033/statics/GEO_data/GSE/figure6/output/panel6B"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# 核心标记基因（scRNA-seq violin图）
CORE_GENES = ["FTL", "CTSS", "TNFSF13B"]

# Top 10 SHAP 基因（空间转录组验证）
SHAP_FILE = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5D_E/shap_importance.csv"
_shap_df   = pd.read_csv(SHAP_FILE)
TOP10_GENES = _shap_df.sort_values("shap_importance", ascending=False)["feature"].head(10).tolist()
print("Top10 SHAP genes:", TOP10_GENES)

# 免疫细胞标志基因（用于共定位分析）
IMMUNE_MARKERS = {
    "Monocyte/Macrophage": ["CD68", "LYZ", "CST3"],
    "B cell":              ["CD19", "MS4A1", "CD79A"],
    "T cell":              ["CD3D", "CD3E", "CD4"],
}

sc.settings.set_figure_params(dpi=150, fontsize=11, facecolor="white")
plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def extract_tar_if_needed(tar_path, out_dir):
    """解压RAW.tar，跳过已存在的文件"""
    # 检验tar文件是否有效
    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getnames()
    except Exception:
        print(f"  [警告] {tar_path} 不是有效的tar文件，请重新从GEO下载。")
        print(f"  下载地址: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186476")
        return False

    already = os.listdir(out_dir)
    if len(already) >= len(members) * 0.8:
        print(f"  已解压 ({len(already)} 个文件)")
        return True

    print(f"  解压中 ({len(members)} 个文件)...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    print(f"  解压完成")
    return True


def load_sample_from_nested_targz(outer_path, condition):
    """
    从嵌套的 *_counts.tar.gz 中直接读取 barcodes/features/matrix，
    不解压到磁盘。
    结构: outer.tar.gz -> SomeDir/{barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz}
    """
    import gzip, scipy.io
    from scipy.sparse import csr_matrix

    try:
        with tarfile.open(outer_path, "r:gz") as tf:
            members = {os.path.basename(m.name): m for m in tf.getmembers() if not m.isdir()}

            b_key = "barcodes.tsv.gz"
            g_key = "features.tsv.gz" if "features.tsv.gz" in members else "genes.tsv.gz"
            m_key = "matrix.mtx.gz"

            if not all(k in members for k in [b_key, g_key, m_key]):
                print(f"    [警告] 缺少必要文件: {list(members.keys())}")
                return None

            with gzip.open(tf.extractfile(members[b_key]), "rt") as f:
                barcodes = [line.strip() for line in f]

            with gzip.open(tf.extractfile(members[g_key]), "rt") as f:
                genes_raw = [line.strip().split("\t") for line in f]
            gene_names = [g[1] if len(g) > 1 else g[0] for g in genes_raw]

            with gzip.open(tf.extractfile(members[m_key]), "rb") as f:
                mat = scipy.io.mmread(io.BytesIO(f.read())).T
            mat = csr_matrix(mat, dtype=np.float32)

        adata = sc.AnnData(X=mat)
        adata.obs_names = barcodes
        adata.var_names = gene_names
        adata.var_names_make_unique()
        adata.obs["condition"] = condition
        return adata

    except Exception as e:
        print(f"    [警告] 加载 {os.path.basename(outer_path)} 失败: {e}")
        return None


# ── 1. 解压数据 ───────────────────────────────────────────────────────────────
print("[1/7] 检查并解压 GSE186476 数据...")
tar_ok = extract_tar_if_needed(RAW_TAR, DATA_DIR)

if not tar_ok:
    print("\n*** 数据文件无效，后续步骤将跳过，请修复数据后重跑 ***")
    import sys; sys.exit(1)

# ── 2. 识别样本文件 ───────────────────────────────────────────────────────────
print("[2/7] 识别样本文件...")

# 文件名格式: GSM{id}_{Condition}_{Sample}_counts.tar.gz
# 条件直接从文件名解析（Healthy / Lupus-Les / Lupus-NonLes）
SAMPLE_META = {}  # filepath -> condition
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith("_counts.tar.gz"):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    # 解析条件：文件名第二段
    parts = fname.split("_")  # ['GSM5652699', 'Healthy', 'D01', 'counts.tar.gz']
    if len(parts) < 3:
        continue
    raw_cond = parts[1]  # 'Healthy' / 'Lupus-Les' / 'Lupus-NonLes'
    if "NonLes" in fname or "NonLes" in raw_cond:
        cond = "Lupus-NonLes"
    elif "Les" in fname:
        cond = "Lupus-Les"
    elif "Healthy" in fname:
        cond = "Healthy"
    else:
        cond = "Unknown"
    sample_id = "_".join(parts[1:-1])  # e.g. 'Healthy_D01'
    SAMPLE_META[fpath] = (sample_id, cond)

print(f"  发现样本: {len(SAMPLE_META)} 个")
cond_counts = pd.Series([v[1] for v in SAMPLE_META.values()]).value_counts()
for cond, cnt in cond_counts.items():
    print(f"    {cond}: {cnt} 个")

# ── 3. 加载 scRNA-seq 样本 ────────────────────────────────────────────────────
print("[3/7] 加载 scRNA-seq 样本...")
adatas = []
for fpath, (sample_id, condition) in SAMPLE_META.items():
    adata_s = load_sample_from_nested_targz(fpath, condition)
    if adata_s is not None:
        adata_s.obs["sample"] = sample_id
        adatas.append(adata_s)
        print(f"    ✓ {sample_id} ({condition}): {adata_s.shape[0]} cells")

if not adatas:
    print("  [错误] 未能加载任何样本")
    import sys; sys.exit(1)

adata_sc = sc.concat(adatas, label="sample", join="outer", fill_value=0)
adata_sc.obs_names_make_unique()
print(f"  合并后: {adata_sc.shape[0]} cells × {adata_sc.shape[1]} genes")

# ── 4. 基本QC与预处理 ─────────────────────────────────────────────────────────
print("[4/7] QC与预处理...")
adata_sc.var["mt"] = adata_sc.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata_sc, qc_vars=["mt"], inplace=True)
adata_sc = adata_sc[
    (adata_sc.obs["n_genes_by_counts"] > 200) &
    (adata_sc.obs["pct_counts_mt"] < 20)
].copy()
print(f"  QC后: {adata_sc.shape[0]} cells")

sc.pp.normalize_total(adata_sc, target_sum=1e4)
sc.pp.log1p(adata_sc)
sc.pp.highly_variable_genes(adata_sc, n_top_genes=2000)
sc.pp.pca(adata_sc)
sc.pp.neighbors(adata_sc, n_pcs=30)
sc.tl.umap(adata_sc)

# ── 5. 计算免疫细胞模块评分 ──────────────────────────────────────────────────
print("[5/7] 计算免疫细胞评分...")
for cell_type, markers in IMMUNE_MARKERS.items():
    avail = [g for g in markers if g in adata_sc.var_names]
    if avail:
        score_key = cell_type.replace("/", "_").replace(" ", "_") + "_score"
        sc.tl.score_genes(adata_sc, gene_list=avail, score_name=score_key)

# ── 6. 空间转录组数据 ─────────────────────────────────────────────────────────
print("[6/7] 加载空间转录组数据...")

spatial_h5   = os.path.join(DATA_DIR, "GSM6021876_lupus-spatial_expr_matrix.h5")
spatial_img_gz = os.path.join(DATA_DIR, "GSM6021876_lupus-spatial_tissue_image.png.gz")
spatial_img  = os.path.join(DATA_DIR, "GSM6021876_lupus-spatial_tissue_image.png")

has_spatial = os.path.exists(spatial_h5)

if has_spatial:
    adata_sp = sc.read_10x_h5(spatial_h5)
    adata_sp.var_names_make_unique()
    print(f"  空间数据: {adata_sp.shape[0]} spots × {adata_sp.shape[1]} genes")

    # 解压组织图像
    if not os.path.exists(spatial_img) and os.path.exists(spatial_img_gz):
        import gzip, shutil
        with gzip.open(spatial_img_gz, "rb") as f_in, open(spatial_img, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # 预处理 + 计算 spots UMAP
    sc.pp.normalize_total(adata_sp, target_sum=1e4)
    sc.pp.log1p(adata_sp)
    sc.pp.highly_variable_genes(adata_sp, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata_sp, n_comps=30)
    sc.pp.neighbors(adata_sp, n_pcs=20)
    sc.tl.umap(adata_sp)
    print(f"  Spots UMAP 计算完成: {adata_sp.obsm['X_umap'].shape}")
else:
    print("  [警告] 未找到空间转录组 h5 文件，跳过空间分析部分")
    has_spatial = False

# ── 7. 绘图 ───────────────────────────────────────────────────────────────────
print("[7/7] 绘图...")

CONDITION_COLORS = {
    "Healthy":       "#4393C3",
    "Lupus-NonLes":  "#F4A582",
    "Lupus-Les":     "#D6604D",
}
COND_ORDER = ["Healthy", "Lupus-NonLes", "Lupus-Les"]

# ── 图1: scRNA-seq UMAP（条件 + 免疫评分 + Top10 SHAP基因）─────────────────
# 布局 3×5：前3格=条件/单核/B细胞评分，后10格=Top10 SHAP基因
fig1 = plt.figure(figsize=(4 * 5, 4 * 3))
gs1  = gridspec.GridSpec(3, 5, figure=fig1, hspace=0.45, wspace=0.35)

# UMAP by condition
ax = fig1.add_subplot(gs1[0, 0])
for cond in COND_ORDER:
    idx = adata_sc.obs["condition"] == cond
    ax.scatter(adata_sc.obsm["X_umap"][idx, 0], adata_sc.obsm["X_umap"][idx, 1],
               c=CONDITION_COLORS[cond], s=0.5, alpha=0.4, label=cond, rasterized=True)
ax.set_title("Condition", fontsize=11, fontweight="bold")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.legend(markerscale=5, frameon=False, fontsize=8)
ax.set_aspect("equal")

# UMAP by monocyte score
ax = fig1.add_subplot(gs1[0, 1])
score_key = "Monocyte_Macrophage_score"
if score_key in adata_sc.obs:
    sc_plot = ax.scatter(adata_sc.obsm["X_umap"][:, 0], adata_sc.obsm["X_umap"][:, 1],
                         c=adata_sc.obs[score_key], cmap="Reds", s=0.5, alpha=0.5, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, shrink=0.7)
ax.set_title("Monocyte/Macrophage Score", fontsize=11, fontweight="bold")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_aspect("equal")

# B cell score
ax = fig1.add_subplot(gs1[0, 2])
score_key = "B_cell_score"
if score_key in adata_sc.obs:
    sc_plot = ax.scatter(adata_sc.obsm["X_umap"][:, 0], adata_sc.obsm["X_umap"][:, 1],
                         c=adata_sc.obs[score_key], cmap="Blues", s=0.5, alpha=0.5, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, shrink=0.7)
ax.set_title("B Cell Score", fontsize=11, fontweight="bold")
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_aspect("equal")

# Top10 SHAP 基因表达 UMAP
# 位置: (0,3),(0,4), (1,0)~(1,4), (2,0)~(2,2)
gene_positions = [(0, 3), (0, 4),
                  (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                  (2, 0), (2, 1), (2, 2)]
cmap_gene = plt.cm.YlOrRd
for (row, col), gene in zip(gene_positions, TOP10_GENES):
    ax = fig1.add_subplot(gs1[row, col])
    if gene in adata_sc.var_names:
        gene_idx = list(adata_sc.var_names).index(gene)
        expr = np.asarray(adata_sc.X[:, gene_idx].todense()).ravel() \
               if hasattr(adata_sc.X, "todense") else adata_sc.X[:, gene_idx]
        sc_plot = ax.scatter(adata_sc.obsm["X_umap"][:, 0], adata_sc.obsm["X_umap"][:, 1],
                             c=expr, cmap=cmap_gene, s=0.5, alpha=0.5, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7, label="log1p expr")
        ax.set_title(f"{gene} Expression", fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"{gene}\n(not found)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{gene}", fontsize=11)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_aspect("equal")

fig1.suptitle("Panel 6B (scRNA-seq): Top 10 SHAP Genes in GSE186476", fontsize=13, fontweight="bold")
fig1.savefig(os.path.join(OUT_DIR, "Panel6B_scrna.pdf"), bbox_inches="tight", dpi=150)
fig1.savefig(os.path.join(OUT_DIR, "Panel6B_scrna.png"), bbox_inches="tight", dpi=150)
plt.close(fig1)

# ── 图2: Top10 SHAP 基因在各条件下的表达分布 ─────────────────────────────────
from scipy.stats import mannwhitneyu
fig2, axes2 = plt.subplots(2, 5, figsize=(5 * 5, 5 * 2))
axes2_flat = axes2.ravel()
for i, gene in enumerate(TOP10_GENES):
    ax = axes2_flat[i]
    if gene not in adata_sc.var_names:
        ax.text(0.5, 0.5, f"{gene} not found", ha="center", va="center", transform=ax.transAxes)
        continue
    gene_idx = list(adata_sc.var_names).index(gene)

    data_by_cond = []
    for cond in COND_ORDER:
        idx = (adata_sc.obs["condition"] == cond).values
        expr = np.asarray(adata_sc.X[idx, gene_idx].todense()).ravel() \
               if hasattr(adata_sc.X, "todense") else adata_sc.X[idx, gene_idx]
        data_by_cond.append(expr)

    parts = ax.violinplot(data_by_cond, positions=range(len(COND_ORDER)),
                          showmedians=True, showextrema=False)
    for pc, cond in zip(parts["bodies"], COND_ORDER):
        pc.set_facecolor(CONDITION_COLORS[cond])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels(COND_ORDER, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("log1p Expression")
    ax.set_title(f"{gene}", fontsize=13, fontweight="bold")

    _, p = mannwhitneyu(data_by_cond[0], data_by_cond[2], alternative="less")
    p = max(p, np.finfo(float).tiny)  # 防止浮点下溢导致p=0.0
    ymax = max(np.quantile(d, 0.99) for d in data_by_cond)
    ax.plot([0, 2], [ymax*1.05, ymax*1.05], "k-", linewidth=0.8)
    label = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(1, ymax*1.07, f"Healthy vs Lupus-Les\n{label} (p={p:.1e})",
            ha="center", fontsize=8)

fig2.suptitle("Panel 6B: Top 10 SHAP Gene Expression Across Conditions (GSE186476)", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "Panel6B_violin.pdf"), bbox_inches="tight", dpi=150)
fig2.savefig(os.path.join(OUT_DIR, "Panel6B_violin.png"), bbox_inches="tight", dpi=150)
plt.close(fig2)

# ── 图3: 空间转录组 spots UMAP（Top10 SHAP 基因表达）────────────────────────
if has_spatial:
    try:
        # 组织图像解压（如需要）
        if not os.path.exists(spatial_img) and os.path.exists(spatial_img_gz):
            import gzip as _gzip, shutil
            with _gzip.open(spatial_img_gz, "rb") as f_in, open(spatial_img, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        umap_xy = adata_sp.obsm["X_umap"]

        # 布局：3行 × 4列 = 12格，第0格组织图，第1格总counts，余下10格=Top10基因
        fig3 = plt.figure(figsize=(4.5 * 4, 4.5 * 3))
        gs3  = gridspec.GridSpec(3, 4, figure=fig3, hspace=0.45, wspace=0.35)

        # ── 组织图像（第0格）────────────────────────────────────────────────
        ax = fig3.add_subplot(gs3[0, 0])
        if os.path.exists(spatial_img):
            tissue_img = plt.imread(spatial_img)
            ax.imshow(tissue_img)
        else:
            ax.text(0.5, 0.5, "No tissue image", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Tissue Image\n(GSM6021876, Lupus)", fontsize=11, fontweight="bold")
        ax.axis("off")

        # ── Spots UMAP: 总 counts（第1格）───────────────────────────────────
        ax = fig3.add_subplot(gs3[0, 1])
        total_counts = np.asarray(adata_sp.X.sum(axis=1)).ravel()
        sc_plot = ax.scatter(umap_xy[:, 0], umap_xy[:, 1],
                             c=total_counts, cmap="viridis",
                             s=5, alpha=0.7, rasterized=True)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7, label="Total counts")
        ax.set_title("Spots UMAP\n(Total counts)", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_aspect("equal")

        # ── Spots UMAP: Top10 SHAP 基因（格2~11）────────────────────────────
        # 位置顺序: row0-col2, row0-col3, row1-col0..col3, row2-col0..col3
        positions = [(0, 2), (0, 3),
                     (1, 0), (1, 1), (1, 2), (1, 3),
                     (2, 0), (2, 1), (2, 2), (2, 3)]
        for (row, col), gene in zip(positions, TOP10_GENES):
            ax = fig3.add_subplot(gs3[row, col])
            if gene in adata_sp.var_names:
                gi   = list(adata_sp.var_names).index(gene)
                expr = np.asarray(adata_sp.X[:, gi].todense()).ravel() \
                       if hasattr(adata_sp.X, "todense") else np.asarray(adata_sp.X[:, gi]).ravel()
                vmax = np.percentile(expr, 99) if expr.max() > 0 else 1
                sc_plot = ax.scatter(umap_xy[:, 0], umap_xy[:, 1],
                                     c=expr, cmap="YlOrRd",
                                     vmin=0, vmax=vmax,
                                     s=5, alpha=0.8, rasterized=True)
                plt.colorbar(sc_plot, ax=ax, shrink=0.7, label="log1p expr")
                ax.set_title(f"Spots UMAP\n{gene}", fontsize=11, fontweight="bold")
            else:
                ax.text(0.5, 0.5, f"{gene}\n(not found)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{gene}", fontsize=11)
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
            ax.set_aspect("equal")

        fig3.suptitle(
            "Panel 6B: Spatial Transcriptomics — Spots UMAP (GSM6021876, Lupus)\n"
            "Top 10 SHAP genes (Figure 5D): " + ", ".join(TOP10_GENES),
            fontsize=12, fontweight="bold"
        )
        fig3.savefig(os.path.join(OUT_DIR, "Panel6B_spatial.pdf"), bbox_inches="tight", dpi=150)
        fig3.savefig(os.path.join(OUT_DIR, "Panel6B_spatial.png"), bbox_inches="tight", dpi=150)
        plt.close(fig3)
        print("  ✓ Spots UMAP 图已保存（Top10 SHAP genes）")
    except Exception as e:
        print(f"  [警告] 空间图绘制失败: {e}")
        import traceback; traceback.print_exc()

print(f"\n完成！输出: {OUT_DIR}/Panel6B_*.pdf / .png")

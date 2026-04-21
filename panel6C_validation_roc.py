#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 6C: 多数据集独立验证 + ROC曲线

两个数据集角色不同：
  GSE254176 (19个样本，全部来自6名SLE患者外周血，无对照)
    → 验证 Top10 SHAP 基因在 SLE 患者中"稳定高表达"
    → 展示：跨样本热图 + 点图（%表达细胞 × 均值）

  GSE162577 (3个样本: SLE-1, SLE-2 + C-1 Control)
    → SLE vs Control 表达差异 + 细胞级 ROC 曲线
    → 注：样本数少(n=3)，ROC在细胞水平计算以保证统计效力
"""

import os, tarfile, io, gzip
import numpy as np
import pandas as pd
import scipy.io, scipy.stats
from scipy.sparse import csr_matrix
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GSE254176_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE254176/GSE254176_RAW.tar"
GSE162577_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE162577/GSE162577_RAW.tar"
SHAP_FILE     = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5D_E/shap_importance.csv"
OUT_DIR       = "/home/h3033/statics/GEO_data/GSE/figure6/output/panel6C"
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.set_figure_params(dpi=150, fontsize=11, facecolor="white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Top10 SHAP 基因（按重要性排序）───────────────────────────────────────────
shap_df     = pd.read_csv(SHAP_FILE)
TOP10_GENES = shap_df.sort_values("shap_importance", ascending=False)["feature"].head(10).tolist()
SHAP_VALS   = shap_df.set_index("feature")["shap_importance"]
print("Top10 SHAP genes:", TOP10_GENES)

GENE_COLORS = {g: c for g, c in zip(TOP10_GENES, plt.cm.tab10(np.linspace(0, 1, 10)))}

# ── GSE162577 分组函数 ────────────────────────────────────────────────────────
def gse162577_condition(prefix):
    """GSM4954813 = C-1 (Control)，其余为 SLE"""
    return "Control" if prefix.startswith("GSM4954813") else "SLE"

# ── 工具函数：从 tar 内存流加载 10X MTX ──────────────────────────────────────
def _read_entry(raw_bytes, as_text=True):
    """
    解析单个文件的原始字节：支持 gzip 和 RAR（名义上叫 .gz 实为 RAR）。
    返回 bytes 或 str（as_text=True）。
    """
    # RAR magic: b'Rar!\x1a\x07' — 用 bsdtar 解压，无需 rarfile 包
    if raw_bytes[:4] == b'Rar!':
        import tempfile, subprocess
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            bsdtar = next(
                p for p in [
                    "/home/h3033/miniconda3/bin/bsdtar",
                    "/usr/bin/bsdtar", "/bin/bsdtar",
                ] if os.path.exists(p)
            )
            result = subprocess.run(
                [bsdtar, "-xOf", tmp_path],
                capture_output=True, check=True
            )
            data = result.stdout
        finally:
            os.unlink(tmp_path)
        # 内部文件可能还是 gzip
        try:
            data = gzip.decompress(data)
        except Exception:
            pass
    else:
        data = gzip.decompress(raw_bytes)
    return data.decode("utf-8") if as_text else data


def load_mtx_from_tar(tar_path, condition_fn, features_col=1):
    """
    直接从 RAW.tar 内存读取，不解压到磁盘。
    支持 .gz（gzip）和名义上叫 .gz 实为 RAR 的文件。
    condition_fn(sample_id) -> condition 字符串
    """
    adatas = []
    with tarfile.open(tar_path) as tf:
        members = {m.name: m for m in tf.getmembers()}
        prefixes = sorted(
            n.replace(".barcodes.tsv.gz", "")
            for n in members if n.endswith(".barcodes.tsv.gz")
        )
        for prefix in prefixes:
            b_key = f"{prefix}.barcodes.tsv.gz"
            g_key = f"{prefix}.genes.tsv.gz"
            if g_key not in members:
                g_key = f"{prefix}.features.tsv.gz"
            m_key = f"{prefix}.matrix.mtx.gz"
            if not all(k in members for k in [b_key, g_key, m_key]):
                continue
            try:
                barcodes_txt = _read_entry(tf.extractfile(members[b_key]).read(), as_text=True)
                barcodes = [line.strip() for line in barcodes_txt.splitlines() if line.strip()]

                genes_txt = _read_entry(tf.extractfile(members[g_key]).read(), as_text=True)
                genes_raw = [line.strip().split("\t") for line in genes_txt.splitlines() if line.strip()]
                gene_names = [
                    g[features_col] if len(g) > features_col else g[0]
                    for g in genes_raw
                ]

                mat_bytes = _read_entry(tf.extractfile(members[m_key]).read(), as_text=False)
                mat = scipy.io.mmread(io.BytesIO(mat_bytes)).T
                mat = csr_matrix(mat, dtype=np.float32)

                adata_s = sc.AnnData(X=mat)
                adata_s.obs_names = barcodes
                adata_s.var_names = gene_names
                adata_s.var_names_make_unique()

                sample_id = prefix.split("_", 1)[1] if "_" in prefix else prefix
                adata_s.obs["sample"]    = sample_id
                adata_s.obs["gsm"]       = prefix.split("_")[0]
                adata_s.obs["condition"] = condition_fn(prefix)
                adatas.append(adata_s)
                print(f"    ✓ {sample_id} [{adata_s.obs['condition'].iloc[0]}]: {adata_s.shape[0]} cells")
            except Exception as e:
                print(f"    [警告] {prefix}: {e}")
                import traceback; traceback.print_exc()
    return adatas


def preprocess(adata, min_genes=200, max_mt=20):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[
        (adata.obs["n_genes_by_counts"] > min_genes) &
        (adata.obs["pct_counts_mt"] < max_mt)
    ].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def get_gene_expr(adata, gene):
    """提取单个基因的表达向量（log1p normalized）"""
    if gene not in adata.var_names:
        return None
    gi = list(adata.var_names).index(gene)
    x  = adata.X[:, gi]
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载并预处理 GSE254176（纯 SLE）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] 加载 GSE254176 (全SLE，19个样本)...")
# 全部标注为 SLE
adatas_254 = load_mtx_from_tar(GSE254176_TAR,
                                condition_fn=lambda _: "SLE",
                                features_col=1)
adata_254 = None
if adatas_254:
    adata_254 = sc.concat(adatas_254, label="sample_idx", join="outer", fill_value=0)
    adata_254.obs_names_make_unique()
    adata_254 = preprocess(adata_254)
    print(f"  合并后: {adata_254.shape[0]} cells × {adata_254.shape[1]} genes")
    # 打印每个样本细胞数
    print("  样本分布:")
    print(adata_254.obs["sample"].value_counts().to_string())
else:
    print("  [错误] GSE254176 加载失败")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 加载并预处理 GSE162577（SLE + Control）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] 加载 GSE162577 (SLE×2 + Control×1)...")
adatas_162 = load_mtx_from_tar(GSE162577_TAR,
                                condition_fn=gse162577_condition,
                                features_col=1)
adata_162 = None
if adatas_162:
    adata_162 = sc.concat(adatas_162, label="sample_idx", join="outer", fill_value=0)
    adata_162.obs_names_make_unique()
    adata_162 = preprocess(adata_162)
    print(f"  合并后: {adata_162.shape[0]} cells × {adata_162.shape[1]} genes")
    print(adata_162.obs.groupby(["sample", "condition"]).size().to_string())
else:
    print("  [错误] GSE162577 加载失败")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GSE254176：跨样本表达统计（稳定高表达验证）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] GSE254176 表达统计...")

dot_records = []  # 用于点图
sample_mean_records = []  # 用于热图

if adata_254 is not None:
    for sample, grp in adata_254.obs.groupby("sample"):
        idx = grp.index
        sub_X = adata_254[idx]
        for gene in TOP10_GENES:
            expr = get_gene_expr(sub_X, gene)
            if expr is None:
                continue
            pct_expr = (expr > 0).mean() * 100
            mean_expr = expr.mean()
            sample_mean_records.append({"sample": sample, "gene": gene,
                                        "mean_expr": mean_expr, "pct_expr": pct_expr})

    # 全数据集水平点图数据
    for gene in TOP10_GENES:
        expr = get_gene_expr(adata_254, gene)
        if expr is None:
            continue
        dot_records.append({
            "gene": gene,
            "mean_expr": expr.mean(),
            "pct_expr":  (expr > 0).mean() * 100,
            "shap":      SHAP_VALS.get(gene, 0),
        })

df_dot_254  = pd.DataFrame(dot_records).set_index("gene")
df_heatmap  = pd.DataFrame(sample_mean_records).pivot(index="sample", columns="gene", values="mean_expr")
df_heatmap  = df_heatmap.reindex(columns=TOP10_GENES)  # 保持SHAP顺序

df_dot_254.to_csv(os.path.join(OUT_DIR, "GSE254176_dot_stats.csv"))
df_heatmap.to_csv(os.path.join(OUT_DIR, "GSE254176_sample_heatmap.csv"))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GSE162577：细胞级 SLE vs Control + ROC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] GSE162577 SLE vs Control + ROC...")

roc_results  = {}
stat_records = []

if adata_162 is not None:
    sle_idx  = adata_162.obs["condition"] == "SLE"
    ctrl_idx = adata_162.obs["condition"] == "Control"
    y_true   = sle_idx.astype(int).values  # 细胞级别标签

    for gene in TOP10_GENES:
        sle_expr  = get_gene_expr(adata_162[sle_idx],  gene)
        ctrl_expr = get_gene_expr(adata_162[ctrl_idx], gene)
        all_expr  = get_gene_expr(adata_162, gene)

        if sle_expr is None or ctrl_expr is None:
            continue

        # Mann-Whitney（细胞级别）
        stat, p = scipy.stats.mannwhitneyu(sle_expr, ctrl_expr, alternative="greater")
        p = max(p, np.finfo(float).tiny)  # prevent float64 underflow (p=0.0 → ~2.2e-308)
        fc = (sle_expr.mean() + 1e-6) / (ctrl_expr.mean() + 1e-6)
        stat_records.append({
            "gene": gene, "p_value": p, "fold_change": fc,
            "sle_mean": sle_expr.mean(), "ctrl_mean": ctrl_expr.mean(),
            "sle_pct":  (sle_expr > 0).mean() * 100,
            "ctrl_pct": (ctrl_expr > 0).mean() * 100,
        })

        # ROC（细胞级别）
        fpr, tpr, _ = roc_curve(y_true, all_expr)
        roc_results[gene] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    # 复合评分 ROC（Top10基因标准化后均值）
    avail = [g for g in TOP10_GENES if g in adata_162.var_names]
    if avail:
        gene_indices = [list(adata_162.var_names).index(g) for g in avail]
        X_sub = adata_162.X[:, gene_indices]
        X_sub = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
        X_scaled  = StandardScaler().fit_transform(X_sub)
        composite = X_scaled.mean(axis=1)
        fpr, tpr, _ = roc_curve(y_true, composite)
        roc_results["Composite"] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    stat_df = pd.DataFrame(stat_records).sort_values("p_value")
    stat_df.to_csv(os.path.join(OUT_DIR, "GSE162577_stats.csv"), index=False)
    print(stat_df.to_string(index=False))

    auc_df = pd.DataFrame([{"gene": g, "auc": v["auc"]} for g, v in roc_results.items()])
    auc_df.to_csv(os.path.join(OUT_DIR, "GSE162577_auc.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 绘图
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] 绘图...")

fig = plt.figure(figsize=(22, 16))
# 布局: 上行 GSE254176(3列) | 下行 GSE162577(3列)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# ─────────────────────────────────────────────────────────────────────────────
# 上行左: GSE254176 样本×基因热图
# ─────────────────────────────────────────────────────────────────────────────
ax_heat = fig.add_subplot(gs[0, 0])
if not df_heatmap.empty:
    im = ax_heat.imshow(
        df_heatmap.values, aspect="auto", cmap="YlOrRd",
        vmin=0, vmax=np.nanpercentile(df_heatmap.values, 95)
    )
    ax_heat.set_xticks(range(len(df_heatmap.columns)))
    ax_heat.set_xticklabels(df_heatmap.columns, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(len(df_heatmap)))
    ax_heat.set_yticklabels(df_heatmap.index, fontsize=7)
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="log1p mean expr")
ax_heat.set_title("GSE254176 (SLE, n=19 samples)\nTop10 SHAP Genes — Cross-sample Expression", fontsize=11, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 上行中: GSE254176 点图（点大小=%表达细胞，颜色=均值）
# ─────────────────────────────────────────────────────────────────────────────
ax_dot = fig.add_subplot(gs[0, 1])
if not df_dot_254.empty:
    genes_ordered = df_dot_254.sort_values("shap", ascending=False).index.tolist()
    y_pos  = np.arange(len(genes_ordered))
    sizes  = df_dot_254.loc[genes_ordered, "pct_expr"].values
    colors = df_dot_254.loc[genes_ordered, "mean_expr"].values
    sc_dot = ax_dot.scatter(
        np.zeros(len(genes_ordered)), y_pos,
        s=sizes * 3,  # 点大小 ∝ %表达细胞
        c=colors, cmap="YlOrRd",
        vmin=0, vmax=colors.max(), edgecolors="gray", linewidths=0.4, zorder=3
    )
    plt.colorbar(sc_dot, ax=ax_dot, shrink=0.6, label="Mean log1p expr")
    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels(genes_ordered, fontsize=9)
    ax_dot.set_xticks([])
    ax_dot.set_xlim(-0.3, 0.5)
    # 在点右侧标注%
    for yi, (gene, pct, mean) in enumerate(zip(
            genes_ordered,
            df_dot_254.loc[genes_ordered, "pct_expr"],
            df_dot_254.loc[genes_ordered, "mean_expr"])):
        ax_dot.text(0.08, yi, f"{pct:.0f}%  μ={mean:.2f}",
                    va="center", fontsize=7.5, color="black")
    # 图例（点大小）
    for sz, label in [(30, "10%"), (90, "30%"), (150, "50%")]:
        ax_dot.scatter([], [], s=sz*3, c="gray", alpha=0.6, label=label)
    ax_dot.legend(title="% Expr cells", frameon=False, fontsize=7, loc="lower right")
ax_dot.set_title("GSE254176 (SLE)\nDot Plot: Expression Fraction & Intensity", fontsize=11, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 上行右: GSE254176 各基因跨样本分布（箱线图）
# ─────────────────────────────────────────────────────────────────────────────
ax_box = fig.add_subplot(gs[0, 2])
if not df_heatmap.empty:
    plot_genes = TOP10_GENES[::-1]  # 从低到高SHAP方向
    box_data = [df_heatmap[g].dropna().values for g in plot_genes if g in df_heatmap.columns]
    plot_genes_avail = [g for g in plot_genes if g in df_heatmap.columns]
    bp = ax_box.boxplot(
        box_data, vert=False, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.5)
    )
    for patch, gene in zip(bp["boxes"], plot_genes_avail):
        patch.set_facecolor(GENE_COLORS.get(gene, "steelblue"))
        patch.set_alpha(0.7)
    # 叠加原始数据点
    for xi, gene in enumerate(plot_genes_avail):
        vals = df_heatmap[gene].dropna().values
        ax_box.scatter(vals, np.random.normal(xi+1, 0.06, len(vals)),
                       s=20, c="black", alpha=0.6, zorder=4)
    ax_box.set_yticks(range(1, len(plot_genes_avail)+1))
    ax_box.set_yticklabels(plot_genes_avail, fontsize=9)
    ax_box.set_xlabel("Mean log1p expr (per sample)")
ax_box.set_title("GSE254176 (SLE)\nCross-sample Consistency (Boxplot)", fontsize=11, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 下行左: GSE162577 SLE vs Control 点图（带显著性）
# ─────────────────────────────────────────────────────────────────────────────
ax_sle = fig.add_subplot(gs[1, 0])
if stat_records and adata_162 is not None:
    stat_df_plot = pd.DataFrame(stat_records).set_index("gene")
    genes_plot   = [g for g in TOP10_GENES if g in stat_df_plot.index]
    y_pos        = np.arange(len(genes_plot))

    sle_means  = stat_df_plot.loc[genes_plot, "sle_mean"].values
    ctrl_means = stat_df_plot.loc[genes_plot, "ctrl_mean"].values

    ax_sle.barh(y_pos - 0.18, sle_means,  0.35, color="#D6604D", alpha=0.8, label="SLE")
    ax_sle.barh(y_pos + 0.18, ctrl_means, 0.35, color="#4393C3", alpha=0.8, label="Control")

    # 显著性标注
    for yi, gene in enumerate(genes_plot):
        p = stat_df_plot.loc[gene, "p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            xmax = max(sle_means[yi], ctrl_means[yi]) + 0.05
            ax_sle.text(xmax, yi, sig, va="center", fontsize=10, color="black")

    ax_sle.set_yticks(y_pos)
    ax_sle.set_yticklabels(genes_plot, fontsize=9)
    ax_sle.set_xlabel("Mean log1p expression")
    ax_sle.legend(frameon=False, fontsize=9)
ax_sle.set_title("GSE162577\nSLE vs Control (cell-level mean)", fontsize=11, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 下行中: ROC 曲线（细胞级别，GSE162577）
# ─────────────────────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 1])
if roc_results:
    for gene, res in roc_results.items():
        if gene == "Composite":
            ax_roc.plot(res["fpr"], res["tpr"], "k-", lw=2.5, zorder=5,
                        label=f"Composite (AUC={res['auc']:.2f})")
        else:
            color = GENE_COLORS.get(gene, "gray")
            ax_roc.plot(res["fpr"], res["tpr"], color=color, lw=1.0, alpha=0.75,
                        label=f"{gene} ({res['auc']:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=1)
ax_roc.set_title("GSE162577\nROC Curves (cell-level, SLE vs Control)", fontsize=11, fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 下行右: AUC 条形图 + SHAP重要性对比
# ─────────────────────────────────────────────────────────────────────────────
ax_auc = fig.add_subplot(gs[1, 2])
if roc_results:
    auc_items = {g: v["auc"] for g, v in roc_results.items() if g != "Composite"}
    auc_sorted = sorted(auc_items.items(), key=lambda x: x[1])
    genes_s = [g for g, _ in auc_sorted]
    auc_s   = [v for _, v in auc_sorted]
    colors_s = [GENE_COLORS.get(g, "gray") for g in genes_s]
    y_pos = np.arange(len(genes_s))

    bars = ax_auc.barh(y_pos, auc_s, color=colors_s, alpha=0.8, edgecolor="gray", linewidth=0.4)
    # 叠加 composite
    if "Composite" in roc_results:
        ax_auc.axvline(roc_results["Composite"]["auc"], color="black", lw=2,
                       linestyle="-", label=f"Composite AUC={roc_results['Composite']['auc']:.2f}")
        ax_auc.legend(frameon=False, fontsize=8)
    ax_auc.axvline(0.5, color="gray", lw=0.8, linestyle="--")
    ax_auc.set_yticks(y_pos)
    ax_auc.set_yticklabels(genes_s, fontsize=9)
    ax_auc.set_xlabel("AUC")
    ax_auc.set_xlim([0, 1])
    # 在条形右侧标注AUC值
    for yi, (g, a) in enumerate(zip(genes_s, auc_s)):
        ax_auc.text(a + 0.01, yi, f"{a:.2f}", va="center", fontsize=8)
ax_auc.set_title("GSE162577\nAUC by Gene", fontsize=11, fontweight="bold")

fig.suptitle(
    "Panel 6C: Independent Validation of Top10 SHAP Genes\n"
    "GSE254176 (SLE expression consistency) | GSE162577 (SLE vs Control + ROC)",
    fontsize=13, fontweight="bold", y=1.01
)

fig.savefig(os.path.join(OUT_DIR, "Panel6C.pdf"), bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(OUT_DIR, "Panel6C.png"), bbox_inches="tight", dpi=150)
plt.close(fig)

print(f"\n完成！输出文件:")
print(f"  图表:        {OUT_DIR}/Panel6C.pdf / .png")
print(f"  GSE254176:   {OUT_DIR}/GSE254176_dot_stats.csv")
print(f"               {OUT_DIR}/GSE254176_sample_heatmap.csv")
print(f"  GSE162577:   {OUT_DIR}/GSE162577_stats.csv")
print(f"               {OUT_DIR}/GSE162577_auc.csv")

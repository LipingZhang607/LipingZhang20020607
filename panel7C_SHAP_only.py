#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 6C - SHAP基因独立验证（单独版本）

专门验证SHAP识别的Top10基因
"""

import os, tarfile, io, gzip
import numpy as np
import pandas as pd
import scipy.io, scipy.stats
from scipy.sparse import csr_matrix
import scanpy as sc
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(__file__))
from figure_config import setup_publication_style, get_figure_size, get_font_size, get_line_width, get_marker_size

setup_publication_style()
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GSE254176_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE254176/GSE254176_RAW.tar"
GSE162577_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE162577/GSE162577_RAW.tar"
SHAP_FILE     = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5D_E/shap_importance.csv"
OUT_DIR       = "/home/h3033/statics/GEO_data/GSE/figure6/output/panel6C_SHAP_only"
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.set_figure_params(dpi=300, fontsize=get_font_size("legend"), facecolor="white")


print("=" * 80)
print("Panel 6C - SHAP基因独立验证")
print("=" * 80)

# ── 加载 SHAP Top10 基因 ──────────────────────────────────────────────────────
shap_df = pd.read_csv(SHAP_FILE)
SHAP_TOP10 = shap_df.sort_values("shap_importance", ascending=False)["feature"].head(10).tolist()
SHAP_VALS = shap_df.set_index("feature")["shap_importance"]
print(f"\nSHAP Top10 genes: {SHAP_TOP10}")

SHAP_COLORS = {g: c for g, c in zip(SHAP_TOP10, plt.cm.Set2(np.linspace(0, 1, 10)))}

def gse162577_condition(prefix):
    """GSM4954813 = C-1 (Control)，其余为 SLE"""
    return "Control" if prefix.startswith("GSM4954813") else "SLE"

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _read_entry(raw_bytes, as_text=True):
    if raw_bytes[:4] == b'Rar!':
        import tempfile, subprocess
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            bsdtar = next(p for p in ["/home/h3033/miniconda3/bin/bsdtar", "/usr/bin/bsdtar", "/bin/bsdtar"] if os.path.exists(p))
            result = subprocess.run([bsdtar, "-xOf", tmp_path], capture_output=True, check=True)
            data = result.stdout
        finally:
            os.unlink(tmp_path)
        try:
            data = gzip.decompress(data)
        except:
            pass
    else:
        data = gzip.decompress(raw_bytes)
    return data.decode("utf-8") if as_text else np.asarray(data)

def load_mtx_from_tar(tar_path, condition_fn, features_col=1):
    adatas = []
    with tarfile.open(tar_path) as tf:
        members = {m.name: m for m in tf.getmembers()}
        prefixes = sorted(n.replace(".barcodes.tsv.gz", "") for n in members if n.endswith(".barcodes.tsv.gz"))
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
                gene_names = [g[features_col] if len(g) > features_col else g[0] for g in genes_raw]

                mat_bytes = _read_entry(tf.extractfile(members[m_key]).read(), as_text=False)
                mat = scipy.io.mmread(io.BytesIO(mat_bytes)).T
                mat = csr_matrix(mat, dtype=np.float32)

                adata_s = sc.AnnData(X=mat)
                adata_s.obs_names = barcodes
                adata_s.var_names = gene_names
                adata_s.var_names_make_unique()

                sample_id = prefix.split("_", 1)[1] if "_" in prefix else prefix
                adata_s.obs["sample"] = sample_id
                adata_s.obs["gsm"] = prefix.split("_")[0]
                adata_s.obs["condition"] = condition_fn(prefix)
                adatas.append(adata_s)
                print(f"    ✓ {sample_id} [{adata_s.obs['condition'].iloc[0]}]: {adata_s.shape[0]} cells")
            except Exception as e:
                print(f"    [警告] {prefix}: {e}")
    return adatas

def preprocess(adata, min_genes=200, max_mt=20):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[(adata.obs["n_genes_by_counts"] > min_genes) & (adata.obs["pct_counts_mt"] < max_mt)].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata

def get_gene_expr(adata, gene):
    if gene not in adata.var_names:
        return None
    gi = list(adata.var_names).index(gene)
    x = adata.X[:, gi]
    return np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] 加载 GSE254176...")
adatas_254 = load_mtx_from_tar(GSE254176_TAR, condition_fn=lambda _: "SLE", features_col=1)
adata_254 = None
if adatas_254:
    adata_254 = sc.concat(adatas_254, label="sample_idx", join="outer", fill_value=0)
    adata_254.obs_names_make_unique()
    adata_254 = preprocess(adata_254)
    print(f"  合并后: {adata_254.shape[0]} cells × {adata_254.shape[1]} genes")

print("\n[2/4] 加载 GSE162577...")
adatas_162 = load_mtx_from_tar(GSE162577_TAR, condition_fn=gse162577_condition, features_col=1)
adata_162 = None
if adatas_162:
    adata_162 = sc.concat(adatas_162, label="sample_idx", join="outer", fill_value=0)
    adata_162.obs_names_make_unique()
    adata_162 = preprocess(adata_162)
    print(f"  合并后: {adata_162.shape[0]} cells × {adata_162.shape[1]} genes")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 验证分析
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] SHAP基因验证分析...")

# GSE254176 表达统计
dot_records = []
sample_mean_records = []
if adata_254 is not None:
    for sample, grp in adata_254.obs.groupby("sample"):
        idx = grp.index
        sub_X = adata_254[idx]
        for gene in SHAP_TOP10:
            expr = get_gene_expr(sub_X, gene)
            if expr is None:
                continue
            sample_mean_records.append({
                "sample": sample, "gene": gene,
                "mean_expr": expr.mean(), "pct_expr": (expr > 0).mean() * 100
            })

    for gene in SHAP_TOP10:
        expr = get_gene_expr(adata_254, gene)
        if expr is None:
            continue
        dot_records.append({
            "gene": gene,
            "mean_expr": expr.mean(),
            "pct_expr": (expr > 0).mean() * 100,
            "importance": SHAP_VALS.get(gene, 0),
        })

dot_df = pd.DataFrame(dot_records).set_index("gene")
heatmap_df = pd.DataFrame(sample_mean_records).pivot(index="sample", columns="gene", values="mean_expr").reindex(columns=SHAP_TOP10)

# GSE162577 SLE vs Control + ROC
roc_results = {}
stat_records = []
if adata_162 is not None:
    sle_idx = adata_162.obs["condition"] == "SLE"
    ctrl_idx = adata_162.obs["condition"] == "Control"
    y_true = sle_idx.astype(int).values

    for gene in SHAP_TOP10:
        sle_expr = get_gene_expr(adata_162[sle_idx], gene)
        ctrl_expr = get_gene_expr(adata_162[ctrl_idx], gene)
        all_expr = get_gene_expr(adata_162, gene)

        if sle_expr is None or ctrl_expr is None:
            continue

        stat, p = scipy.stats.mannwhitneyu(sle_expr, ctrl_expr, alternative="greater")
        p = max(p, np.finfo(float).tiny)
        fc = (sle_expr.mean() + 1e-6) / (ctrl_expr.mean() + 1e-6)
        stat_records.append({
            "gene": gene, "p_value": p, "fold_change": fc,
            "sle_mean": sle_expr.mean(), "ctrl_mean": ctrl_expr.mean(),
            "sle_pct": (sle_expr > 0).mean() * 100,
            "ctrl_pct": (ctrl_expr > 0).mean() * 100,
        })

        fpr, tpr, _ = roc_curve(y_true, all_expr)
        roc_results[gene] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    # Composite score
    avail = [g for g in SHAP_TOP10 if g in adata_162.var_names]
    if avail:
        gene_indices = [list(adata_162.var_names).index(g) for g in avail]
        X_sub = adata_162.X[:, gene_indices]
        X_sub = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
        X_scaled = StandardScaler().fit_transform(X_sub)
        composite = X_scaled.mean(axis=1)
        fpr, tpr, _ = roc_curve(y_true, composite)
        roc_results["Composite_SHAP"] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

stat_df = pd.DataFrame(stat_records).sort_values("p_value")

# 保存结果
dot_df.to_csv(os.path.join(OUT_DIR, "SHAP_GSE254176_dot_stats.csv"))
heatmap_df.to_csv(os.path.join(OUT_DIR, "SHAP_GSE254176_sample_heatmap.csv"))
stat_df.to_csv(os.path.join(OUT_DIR, "SHAP_GSE162577_stats.csv"), index=False)
auc_df = pd.DataFrame([{"gene": g, "auc": v["auc"]} for g, v in roc_results.items()])
auc_df.to_csv(os.path.join(OUT_DIR, "SHAP_GSE162577_auc.csv"), index=False)

print("\n  SHAP基因统计:")
print(stat_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 绘图
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] 生成图表...")

fig = plt.figure(figsize=get_figure_size(n_panels=12, layout="double"))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 热图
ax = fig.add_subplot(gs[0, 0])
if not heatmap_df.empty:
    im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlGnBu",
                   vmin=0, vmax=np.nanpercentile(heatmap_df.values, 95))
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right", fontsize=get_font_size("tick"))
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(heatmap_df.index, fontsize=get_font_size("tick"))
    plt.colorbar(im, ax=ax, shrink=0.6, label="log1p mean expr")
ax.set_title("GSE254176 (SLE, n=19)\nSHAP Genes Cross-sample Expression", fontweight='bold')

# 点图
ax = fig.add_subplot(gs[0, 1])
if not dot_df.empty:
    genes_ordered = dot_df.sort_values("importance", ascending=False).index.tolist()
    y_pos = np.arange(len(genes_ordered))
    sizes = dot_df.loc[genes_ordered, "pct_expr"].values
    colors = dot_df.loc[genes_ordered, "mean_expr"].values
    sc_dot = ax.scatter(np.zeros(len(genes_ordered)), y_pos, s=sizes * 3,
                        c=colors, cmap="YlGnBu", vmin=0, vmax=colors.max(),
                        edgecolors="gray", linewidths=get_marker_size("small"), zorder=3)
    plt.colorbar(sc_dot, ax=ax, shrink=0.6, label="Mean log1p expr")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes_ordered, fontsize=get_font_size("tick"))
    ax.set_xticks([])
    ax.set_xlim(-0.3, 0.5)
    for yi, (gene, pct, mean) in enumerate(zip(genes_ordered,
                                                 dot_df.loc[genes_ordered, "pct_expr"],
                                                 dot_df.loc[genes_ordered, "mean_expr"])):
        ax.text(0.08, yi, f"{pct:.0f}%  μ={mean:.2f}", va="center", fontsize=get_font_size("tick"))
ax.set_title("GSE254176 (SLE)\nSHAP Genes Expression", fontweight='bold')

# ROC曲线
ax = fig.add_subplot(gs[0, 2])
if roc_results:
    for gene, res in roc_results.items():
        if gene.startswith('Composite'):
            ax.plot(res['fpr'], res['tpr'], 'b-', lw=2.5,
                    label=f"{gene} (AUC={res['auc']:.2f})", zorder=10)
        else:
            color = SHAP_COLORS.get(gene, 'gray')
            ax.plot(res['fpr'], res['tpr'], color=color, lw=get_line_width("plot"), alpha=0.75,
                    label=f"{gene} ({res['auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=get_line_width("plot"), alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('GSE162577\nSHAP Genes ROC Curves', fontweight='bold')
    ax.legend(fontsize=get_font_size("legend"), loc='lower right', frameon=False)

# SLE vs Control
ax = fig.add_subplot(gs[1, 0])
if not stat_df.empty:
    genes_plot = [g for g in SHAP_TOP10 if g in stat_df['gene'].values]
    stat_df_plot = stat_df.set_index("gene")
    y_pos = np.arange(len(genes_plot))
    sle_means = stat_df_plot.loc[genes_plot, "sle_mean"].values
    ctrl_means = stat_df_plot.loc[genes_plot, "ctrl_mean"].values
    ax.barh(y_pos - 0.18, sle_means, 0.35, color="#D6604D", alpha=0.8, label="SLE")
    ax.barh(y_pos + 0.18, ctrl_means, 0.35, color="#4393C3", alpha=0.8, label="Control")
    for yi, gene in enumerate(genes_plot):
        p = stat_df_plot.loc[gene, "p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            xmax = max(sle_means[yi], ctrl_means[yi]) + 0.05
            ax.text(xmax, yi, sig, va="center", fontsize=get_font_size("title"))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes_plot, fontsize=get_font_size("tick"))
    ax.set_xlabel("Mean log1p expression")
    ax.legend(frameon=False)
ax.set_title("GSE162577\nSHAP Genes: SLE vs Control", fontweight='bold')

# AUC条形图
ax = fig.add_subplot(gs[1, 1])
if roc_results:
    auc_items = {g: v["auc"] for g, v in roc_results.items() if not g.startswith('Composite')}
    auc_sorted = sorted(auc_items.items(), key=lambda x: x[1])
    genes_s = [g for g, _ in auc_sorted]
    auc_s = [v for _, v in auc_sorted]
    colors_s = [SHAP_COLORS.get(g, "gray") for g in genes_s]
    y_pos = np.arange(len(genes_s))
    ax.barh(y_pos, auc_s, color=colors_s, alpha=0.8, edgecolor="gray", linewidth=0.4)
    if "Composite_SHAP" in roc_results:
        ax.axvline(roc_results["Composite_SHAP"]["auc"], color="blue", lw=2,
                   linestyle="-", label=f"Composite AUC={roc_results['Composite_SHAP']['auc']:.2f}")
        ax.legend(frameon=False, fontsize=get_font_size("tick"))
    ax.axvline(0.5, color="gray", lw=get_line_width("plot"), linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes_s, fontsize=get_font_size("tick"))
    ax.set_xlabel("AUC")
    ax.set_xlim([0, 1])
    for yi, (g, a) in enumerate(zip(genes_s, auc_s)):
        ax.text(a + 0.01, yi, f"{a:.2f}", va="center", fontsize=get_font_size("tick"))
ax.set_title("GSE162577\nSHAP Genes AUC", fontweight='bold')

# 统计摘要
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
if not stat_df.empty:
    n_sig = (stat_df['p_value'] < 0.05).sum()
    n_total = len(stat_df)
    composite_auc = roc_results.get("Composite_SHAP", {}).get("auc", 0)

    summary_text = f"""
SHAP基因验证摘要
{'='*40}

数据集:
  GSE254176: 19个SLE样本
  GSE162577: 2 SLE + 1 Control

基因数: {n_total}

统计显著性 (p<0.05):
  显著基因: {n_sig}/{n_total} ({n_sig/n_total*100:.0f}%)

判别性能:
  Composite AUC: {composite_auc:.3f}

Top 3基因 (by AUC):
"""
    top3 = auc_df[~auc_df['gene'].str.startswith('Composite')].nlargest(3, 'auc')
    for _, row in top3.iterrows():
        summary_text += f"  {row['gene']}: {row['auc']:.3f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=get_font_size("title"), verticalalignment='top', fontfamily='monospace')

fig.suptitle('Panel 6C: SHAP Genes Independent Validation\n'
             'GSE254176 (SLE consistency) | GSE162577 (SLE vs Control + ROC)',
             fontsize=get_font_size("title"), fontweight='bold', y=0.98)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'Panel6C_SHAP_only.pdf'), bbox_inches='tight', dpi=300)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'Panel6C_SHAP_only.png'), bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"\n完成！输出文件:")
print(f"  图表: {OUT_DIR}/Panel6C_SHAP_only.pdf/png")
print(f"  数据: {OUT_DIR}/SHAP_*.csv")
print(f"\nComposite AUC: {roc_results.get('Composite_SHAP', {}).get('auc', 0):.3f}")
print(f"显著基因: {n_sig}/{n_total} ({n_sig/n_total*100:.0f}%)")

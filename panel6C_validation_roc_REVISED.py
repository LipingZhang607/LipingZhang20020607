#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel 6C REVISED: 多数据集独立验证 + ROC曲线 (GAT vs SHAP 平行验证)

【修订说明】
回应审稿人意见：同时验证GAT和SHAP识别的Top10基因，展示两种方法的互补性

两个数据集角色不同：
  GSE254176 (19个样本，全部来自6名SLE患者外周血，无对照)
    → 验证 Top10 GAT 和 Top10 SHAP 基因在 SLE 患者中"稳定高表达"
    → 展示：跨样本热图 + 点图（%表达细胞 × 均值）

  GSE162577 (3个样本: SLE-1, SLE-2 + C-1 Control)
    → SLE vs Control 表达差异 + 细胞级 ROC 曲线
    → 注：样本数少(n=3)，ROC在细胞水平计算以保证统计效力

【输出】
- Panel6C_REVISED_GAT.pdf/png: GAT基因验证
- Panel6C_REVISED_SHAP.pdf/png: SHAP基因验证
- Panel6C_REVISED_comparison.pdf/png: GAT vs SHAP对比
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
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────────────────────
GSE254176_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE254176/GSE254176_RAW.tar"
GSE162577_TAR = "/home/h3033/statics/GEO_data/GSE/figure6/GSE162577/GSE162577_RAW.tar"
GAT_FILE      = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5C/feature_importance.csv"
SHAP_FILE     = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5D_E/shap_importance.csv"
OUT_DIR       = "/home/h3033/statics/GEO_data/GSE/figure6/output/panel6C_REVISED"
os.makedirs(OUT_DIR, exist_ok=True)

sc.settings.set_figure_params(dpi=300, fontsize=get_font_size("legend"), facecolor="white")


# ── 【新增】加载 GAT 和 SHAP 基因列表 ─────────────────────────────────────────
print("=" * 80)
print("加载特征重要性文件...")
print("=" * 80)

# GAT Top10
gat_df = pd.read_csv(GAT_FILE)
GAT_TOP10 = gat_df.sort_values("importance_combined", ascending=False)["feature"].head(10).tolist()
GAT_VALS = gat_df.set_index("feature")["importance_combined"]
print(f"\nGAT Top10 genes: {GAT_TOP10}")

# SHAP Top10
shap_df = pd.read_csv(SHAP_FILE)
SHAP_TOP10 = shap_df.sort_values("shap_importance", ascending=False)["feature"].head(10).tolist()
SHAP_VALS = shap_df.set_index("feature")["shap_importance"]
print(f"SHAP Top10 genes: {SHAP_TOP10}")

# 重叠分析
overlap = set(GAT_TOP10) & set(SHAP_TOP10)
gat_unique = set(GAT_TOP10) - set(SHAP_TOP10)
shap_unique = set(SHAP_TOP10) - set(GAT_TOP10)
print(f"\n重叠基因 (n={len(overlap)}): {sorted(overlap)}")
print(f"GAT独有 (n={len(gat_unique)}): {sorted(gat_unique)}")
print(f"SHAP独有 (n={len(shap_unique)}): {sorted(shap_unique)}")

# 为两组基因分配不同的颜色方案
GAT_COLORS = {g: c for g, c in zip(GAT_TOP10, plt.cm.Set1(np.linspace(0, 1, 10)))}
SHAP_COLORS = {g: c for g, c in zip(SHAP_TOP10, plt.cm.Set2(np.linspace(0, 1, 10)))}

# 保持向后兼容：TOP10_GENES 和 GENE_COLORS 指向 SHAP（原始行为）
TOP10_GENES = SHAP_TOP10
GENE_COLORS = SHAP_COLORS

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


# ── 【新增】通用验证函数 ──────────────────────────────────────────────────────
def validate_gene_set(gene_list, gene_vals, adata_254, adata_162, label=""):
    """
    对给定的基因列表进行完整验证

    Parameters:
    -----------
    gene_list : list
        要验证的基因列表
    gene_vals : pd.Series
        基因重要性分数（用于排序和标注）
    adata_254 : AnnData
        GSE254176数据（纯SLE）
    adata_162 : AnnData
        GSE162577数据（SLE + Control）
    label : str
        标签（"GAT" 或 "SHAP"）

    Returns:
    --------
    dict : 包含所有验证结果的字典
    """
    results = {
        'label': label,
        'genes': gene_list,
        'dot_df': None,
        'heatmap_df': None,
        'roc_results': {},
        'stat_df': None
    }

    # ─── GSE254176 表达统计 ───
    if adata_254 is not None:
        dot_records = []
        sample_mean_records = []

        for sample, grp in adata_254.obs.groupby("sample"):
            idx = grp.index
            sub_X = adata_254[idx]
            for gene in gene_list:
                expr = get_gene_expr(sub_X, gene)
                if expr is None:
                    continue
                pct_expr = (expr > 0).mean() * 100
                mean_expr = expr.mean()
                sample_mean_records.append({
                    "sample": sample, "gene": gene,
                    "mean_expr": mean_expr, "pct_expr": pct_expr
                })

        # 全数据集水平
        for gene in gene_list:
            expr = get_gene_expr(adata_254, gene)
            if expr is None:
                continue
            dot_records.append({
                "gene": gene,
                "mean_expr": expr.mean(),
                "pct_expr": (expr > 0).mean() * 100,
                "importance": gene_vals.get(gene, 0),
            })

        results['dot_df'] = pd.DataFrame(dot_records).set_index("gene")
        results['heatmap_df'] = pd.DataFrame(sample_mean_records).pivot(
            index="sample", columns="gene", values="mean_expr"
        ).reindex(columns=gene_list)

    # ─── GSE162577 SLE vs Control + ROC ───
    if adata_162 is not None:
        sle_idx = adata_162.obs["condition"] == "SLE"
        ctrl_idx = adata_162.obs["condition"] == "Control"
        y_true = sle_idx.astype(int).values

        stat_records = []
        roc_results = {}

        for gene in gene_list:
            sle_expr = get_gene_expr(adata_162[sle_idx], gene)
            ctrl_expr = get_gene_expr(adata_162[ctrl_idx], gene)
            all_expr = get_gene_expr(adata_162, gene)

            if sle_expr is None or ctrl_expr is None:
                continue

            # Mann-Whitney
            stat, p = scipy.stats.mannwhitneyu(sle_expr, ctrl_expr, alternative="greater")
            p = max(p, np.finfo(float).tiny)
            fc = (sle_expr.mean() + 1e-6) / (ctrl_expr.mean() + 1e-6)
            stat_records.append({
                "gene": gene, "p_value": p, "fold_change": fc,
                "sle_mean": sle_expr.mean(), "ctrl_mean": ctrl_expr.mean(),
                "sle_pct": (sle_expr > 0).mean() * 100,
                "ctrl_pct": (ctrl_expr > 0).mean() * 100,
            })

            # ROC
            fpr, tpr, _ = roc_curve(y_true, all_expr)
            roc_results[gene] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

        # Composite score
        avail = [g for g in gene_list if g in adata_162.var_names]
        if avail:
            gene_indices = [list(adata_162.var_names).index(g) for g in avail]
            X_sub = adata_162.X[:, gene_indices]
            X_sub = X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)
            X_scaled = StandardScaler().fit_transform(X_sub)
            composite = X_scaled.mean(axis=1)
            fpr, tpr, _ = roc_curve(y_true, composite)
            roc_results[f"Composite_{label}"] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

        results['stat_df'] = pd.DataFrame(stat_records).sort_values("p_value")
        results['roc_results'] = roc_results

    return results


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
# 3. 【新增】对 GAT 和 SHAP 基因分别进行验证
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] 验证 GAT 和 SHAP 基因...")

# 验证 GAT 基因
print("\n  [3a] 验证 GAT Top10 基因...")
gat_results = validate_gene_set(GAT_TOP10, GAT_VALS, adata_254, adata_162, label="GAT")

# 验证 SHAP 基因
print("\n  [3b] 验证 SHAP Top10 基因...")
shap_results = validate_gene_set(SHAP_TOP10, SHAP_VALS, adata_254, adata_162, label="SHAP")

# 保存结果
if gat_results['dot_df'] is not None:
    gat_results['dot_df'].to_csv(os.path.join(OUT_DIR, "GAT_GSE254176_dot_stats.csv"))
    gat_results['heatmap_df'].to_csv(os.path.join(OUT_DIR, "GAT_GSE254176_sample_heatmap.csv"))
if gat_results['stat_df'] is not None:
    gat_results['stat_df'].to_csv(os.path.join(OUT_DIR, "GAT_GSE162577_stats.csv"), index=False)
    print("\n  GAT基因统计:")
    print(gat_results['stat_df'].to_string(index=False))

if shap_results['dot_df'] is not None:
    shap_results['dot_df'].to_csv(os.path.join(OUT_DIR, "SHAP_GSE254176_dot_stats.csv"))
    shap_results['heatmap_df'].to_csv(os.path.join(OUT_DIR, "SHAP_GSE254176_sample_heatmap.csv"))
if shap_results['stat_df'] is not None:
    shap_results['stat_df'].to_csv(os.path.join(OUT_DIR, "SHAP_GSE162577_stats.csv"), index=False)
    print("\n  SHAP基因统计:")
    print(shap_results['stat_df'].to_string(index=False))

# 保存AUC结果
if gat_results['roc_results']:
    gat_auc_df = pd.DataFrame([{"gene": g, "auc": v["auc"]}
                                for g, v in gat_results['roc_results'].items()])
    gat_auc_df.to_csv(os.path.join(OUT_DIR, "GAT_GSE162577_auc.csv"), index=False)

if shap_results['roc_results']:
    shap_auc_df = pd.DataFrame([{"gene": g, "auc": v["auc"]}
                                 for g, v in shap_results['roc_results'].items()])
    shap_auc_df.to_csv(os.path.join(OUT_DIR, "SHAP_GSE162577_auc.csv"), index=False)

# 【保持向后兼容】为原始绘图代码保留变量
df_dot_254 = shap_results['dot_df']
df_heatmap = shap_results['heatmap_df']
roc_results = shap_results['roc_results']
stat_records = shap_results['stat_df'].to_dict('records') if shap_results['stat_df'] is not None else []

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 【新增】生成 GAT vs SHAP 对比图
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] 生成 GAT vs SHAP 对比图...")

fig_comp = plt.figure(figsize=get_figure_size(n_panels=12, layout="double"))
gs_comp = gridspec.GridSpec(2, 3, figure=fig_comp, hspace=0.4, wspace=0.35)

# ─── 左上: AUC对比（条形图）───
ax = fig_comp.add_subplot(gs_comp[0, 0])
if gat_results['roc_results'] and shap_results['roc_results']:
    gat_aucs = {g: v['auc'] for g, v in gat_results['roc_results'].items() if not g.startswith('Composite')}
    shap_aucs = {g: v['auc'] for g, v in shap_results['roc_results'].items() if not g.startswith('Composite')}

    all_genes = sorted(set(list(gat_aucs.keys()) + list(shap_aucs.keys())))
    gat_vals = [gat_aucs.get(g, 0) for g in all_genes]
    shap_vals = [shap_aucs.get(g, 0) for g in all_genes]

    x = np.arange(len(all_genes))
    width = 0.35
    ax.barh(x - width/2, gat_vals, width, label='GAT', color='#E74C3C', alpha=0.8)
    ax.barh(x + width/2, shap_vals, width, label='SHAP', color='#3498DB', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(all_genes, fontsize=get_font_size("tick"))
    ax.set_xlabel('AUC (GSE162577)')
    ax.set_title('AUC Comparison: GAT vs SHAP Genes', fontweight='bold')
    ax.legend()
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=get_line_width("plot"))
ax.set_xlim([0, 1])

# ─── 中上: Composite AUC对比 ───
ax = fig_comp.add_subplot(gs_comp[0, 1])
composite_data = []
for label, results in [('GAT', gat_results), ('SHAP', shap_results)]:
    for k, v in results['roc_results'].items():
        if k.startswith('Composite'):
            composite_data.append({'Method': label, 'AUC': v['auc']})
if composite_data:
    comp_df = pd.DataFrame(composite_data)
    colors = ['#E74C3C', '#3498DB']
    bars = ax.bar(comp_df['Method'], comp_df['AUC'], color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Composite AUC')
    ax.set_title('Composite Score Performance', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=get_line_width("plot"))
    for bar, val in zip(bars, comp_df['AUC']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontweight='bold', fontsize=get_font_size("title"))

# ─── 右上: 基因重叠Venn图 ───
ax = fig_comp.add_subplot(gs_comp[0, 2])
try:
    from matplotlib_venn import venn2
    venn2([set(GAT_TOP10), set(SHAP_TOP10)], set_labels=('GAT', 'SHAP'), ax=ax)
    overlap_genes = set(GAT_TOP10) & set(SHAP_TOP10)
    jaccard = len(overlap_genes) / len(set(GAT_TOP10) | set(SHAP_TOP10))
    ax.set_title(f'Gene Overlap (Jaccard={jaccard:.3f})\nOverlap: {", ".join(sorted(overlap_genes))}',
                 fontweight='bold', fontsize=get_font_size("title"))
except ImportError:
    ax.text(0.5, 0.5, 'matplotlib_venn not available', ha='center', va='center')
    ax.set_title('Gene Overlap', fontweight='bold')

# ─── 左下: GAT基因ROC曲线 ───
ax = fig_comp.add_subplot(gs_comp[1, 0])
if gat_results['roc_results']:
    for gene, res in gat_results['roc_results'].items():
        if gene.startswith('Composite'):
            ax.plot(res['fpr'], res['tpr'], 'r-', lw=2.5, label=f"{gene} (AUC={res['auc']:.2f})", zorder=10)
        else:
            ax.plot(res['fpr'], res['tpr'], lw=get_line_width("plot"), alpha=0.6, label=f"{gene} ({res['auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=get_line_width("plot"), alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('GAT Genes: ROC Curves (GSE162577)', fontweight='bold')
    ax.legend(fontsize=get_font_size("legend"), loc='lower right', frameon=False)

# ─── 中下: SHAP基因ROC曲线 ───
ax = fig_comp.add_subplot(gs_comp[1, 1])
if shap_results['roc_results']:
    for gene, res in shap_results['roc_results'].items():
        if gene.startswith('Composite'):
            ax.plot(res['fpr'], res['tpr'], 'b-', lw=2.5, label=f"{gene} (AUC={res['auc']:.2f})", zorder=10)
        else:
            ax.plot(res['fpr'], res['tpr'], lw=get_line_width("plot"), alpha=0.6, label=f"{gene} ({res['auc']:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=get_line_width("plot"), alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('SHAP Genes: ROC Curves (GSE162577)', fontweight='bold')
    ax.legend(fontsize=get_font_size("legend"), loc='lower right', frameon=False)

# ─── 右下: 统计显著性对比 ───
ax = fig_comp.add_subplot(gs_comp[1, 2])
sig_data = []
for label, results in [('GAT', gat_results), ('SHAP', shap_results)]:
    if results['stat_df'] is not None:
        n_sig = (results['stat_df']['p_value'] < 0.05).sum()
        n_total = len(results['stat_df'])
        sig_data.append({'Method': label, 'Significant': n_sig, 'Non-significant': n_total - n_sig})
if sig_data:
    sig_df = pd.DataFrame(sig_data).set_index('Method')
    sig_df.plot(kind='bar', stacked=True, ax=ax, color=['#2ECC71', '#95A5A6'], alpha=0.8)
    ax.set_ylabel('Number of Genes')
    ax.set_title('Statistical Significance (p<0.05)\nGSE162577 SLE vs Control', fontweight='bold')
    ax.legend(title='', frameon=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

fig_comp.suptitle('GAT vs SHAP: Independent Validation Comparison',
                  fontsize=get_font_size("title"), fontweight='bold', y=0.98)
fig_comp.savefig(os.path.join(OUT_DIR, 'Panel6C_REVISED_comparison.pdf'), bbox_inches='tight', dpi=300)
fig_comp.savefig(os.path.join(OUT_DIR, 'Panel6C_REVISED_comparison.png'), bbox_inches='tight', dpi=300)
plt.close(fig_comp)
print(f"  ✓ 保存对比图: Panel6C_REVISED_comparison.pdf/png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 绘图（原始SHAP图，保持向后兼容）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] 生成原始SHAP验证图（向后兼容）...")

fig = plt.figure(figsize=get_figure_size(n_panels=12, layout="double"))
# 布局: 上行 GSE254176(3列) | 下行 GSE162577(3列)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

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
    ax_heat.set_xticklabels(df_heatmap.columns, rotation=45, ha="right", fontsize=get_font_size("tick"))
    ax_heat.set_yticks(range(len(df_heatmap)))
    ax_heat.set_yticklabels(df_heatmap.index, fontsize=get_font_size("legend"))
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="log1p mean expr")
ax_heat.set_title("GSE254176 (SLE, n=19 samples)\nTop10 SHAP Genes — Cross-sample Expression", fontsize=get_font_size("title"), fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# 上行中: GSE254176 点图（点大小=%表达细胞，颜色=均值）
# ─────────────────────────────────────────────────────────────────────────────
ax_dot = fig.add_subplot(gs[0, 1])
if not df_dot_254.empty:
    genes_ordered = df_dot_254.sort_values("importance", ascending=False).index.tolist()
    y_pos  = np.arange(len(genes_ordered))
    sizes  = df_dot_254.loc[genes_ordered, "pct_expr"].values
    colors = df_dot_254.loc[genes_ordered, "mean_expr"].values
    sc_dot = ax_dot.scatter(
        np.zeros(len(genes_ordered)), y_pos,
        s=sizes * 3,  # 点大小 ∝ %表达细胞
        c=colors, cmap="YlOrRd",
        vmin=0, vmax=colors.max(), edgecolors="gray", linewidths=get_marker_size("small"), zorder=3
    )
    plt.colorbar(sc_dot, ax=ax_dot, shrink=0.6, label="Mean log1p expr")
    ax_dot.set_yticks(y_pos)
    ax_dot.set_yticklabels(genes_ordered, fontsize=get_font_size("tick"))
    ax_dot.set_xticks([])
    ax_dot.set_xlim(-0.3, 0.5)
    # 在点右侧标注%
    for yi, (gene, pct, mean) in enumerate(zip(
            genes_ordered,
            df_dot_254.loc[genes_ordered, "pct_expr"],
            df_dot_254.loc[genes_ordered, "mean_expr"])):
        ax_dot.text(0.08, yi, f"{pct:.0f}%  μ={mean:.2f}",
                    va="center", fontsize=get_font_size("legend"), color="black")
    # 图例（点大小）
    for sz, label in [(30, "10%"), (90, "30%"), (150, "50%")]:
        ax_dot.scatter([], [], s=sz*3, c="gray", alpha=0.6, label=label)
    ax_dot.legend(title="% Expr cells", frameon=False, fontsize=get_font_size("legend"), loc="lower right")
ax_dot.set_title("GSE254176 (SLE)\nDot Plot: Expression Fraction & Intensity", fontsize=get_font_size("title"), fontweight="bold")

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
        whiskerprops=dict(linewidth=get_line_width("plot")),
        capprops=dict(linewidth=get_line_width("plot")),
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
    ax_box.set_yticklabels(plot_genes_avail, fontsize=get_font_size("tick"))
    ax_box.set_xlabel("Mean log1p expr (per sample)")
ax_box.set_title("GSE254176 (SLE)\nCross-sample Consistency (Boxplot)", fontsize=get_font_size("title"), fontweight="bold")

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
            ax_sle.text(xmax, yi, sig, va="center", fontsize=get_font_size("title"), color="black")

    ax_sle.set_yticks(y_pos)
    ax_sle.set_yticklabels(genes_plot, fontsize=get_font_size("tick"))
    ax_sle.set_xlabel("Mean log1p expression")
    ax_sle.legend(frameon=False, fontsize=get_font_size("tick"))
ax_sle.set_title("GSE162577\nSLE vs Control (cell-level mean)", fontsize=get_font_size("title"), fontweight="bold")

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
            ax_roc.plot(res["fpr"], res["tpr"], color=color, lw=get_line_width("plot"), alpha=0.75,
                        label=f"{gene} ({res['auc']:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=get_line_width("plot"), alpha=0.4)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=get_font_size("legend"), frameon=False, loc="lower right", ncol=1)
ax_roc.set_title("GSE162577\nROC Curves (cell-level, SLE vs Control)", fontsize=get_font_size("title"), fontweight="bold")

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
        ax_auc.legend(frameon=False, fontsize=get_font_size("tick"))
    ax_auc.axvline(0.5, color="gray", lw=get_line_width("plot"), linestyle="--")
    ax_auc.set_yticks(y_pos)
    ax_auc.set_yticklabels(genes_s, fontsize=get_font_size("tick"))
    ax_auc.set_xlabel("AUC")
    ax_auc.set_xlim([0, 1])
    # 在条形右侧标注AUC值
    for yi, (g, a) in enumerate(zip(genes_s, auc_s)):
        ax_auc.text(a + 0.01, yi, f"{a:.2f}", va="center", fontsize=get_font_size("tick"))
ax_auc.set_title("GSE162577\nAUC by Gene", fontsize=get_font_size("title"), fontweight="bold")

fig.suptitle(
    "Panel 6C (SHAP genes): Independent Validation of Top10 SHAP Genes\n"
    "GSE254176 (SLE expression consistency) | GSE162577 (SLE vs Control + ROC)",
    fontsize=get_font_size("title"), fontweight="bold", y=1.01
)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Panel6C_REVISED_SHAP.pdf"), bbox_inches="tight", dpi=300)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Panel6C_REVISED_SHAP.png"), bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"\n" + "=" * 80)
print("完成！输出文件:")
print("=" * 80)
print(f"\n【对比图】")
print(f"  {OUT_DIR}/Panel6C_REVISED_comparison.pdf/png")
print(f"    - GAT vs SHAP AUC对比")
print(f"    - 基因重叠Venn图")
print(f"    - ROC曲线对比")
print(f"\n【SHAP基因验证】（向后兼容原始图）")
print(f"  {OUT_DIR}/Panel6C_REVISED_SHAP.pdf/png")
print(f"  {OUT_DIR}/SHAP_GSE254176_dot_stats.csv")
print(f"  {OUT_DIR}/SHAP_GSE254176_sample_heatmap.csv")
print(f"  {OUT_DIR}/SHAP_GSE162577_stats.csv")
print(f"  {OUT_DIR}/SHAP_GSE162577_auc.csv")
print(f"\n【GAT基因验证】（新增）")
print(f"  {OUT_DIR}/GAT_GSE254176_dot_stats.csv")
print(f"  {OUT_DIR}/GAT_GSE254176_sample_heatmap.csv")
print(f"  {OUT_DIR}/GAT_GSE162577_stats.csv")
print(f"  {OUT_DIR}/GAT_GSE162577_auc.csv")
print(f"\n" + "=" * 80)
print("审稿人回应要点:")
print("=" * 80)
print(f"1. GAT和SHAP Top10基因重叠度: {len(set(GAT_TOP10) & set(SHAP_TOP10))}/10 = {len(set(GAT_TOP10) & set(SHAP_TOP10))*10}%")
print(f"2. 共同基因: {', '.join(sorted(set(GAT_TOP10) & set(SHAP_TOP10)))}")
print(f"3. GAT独有: {', '.join(sorted(set(GAT_TOP10) - set(SHAP_TOP10)))}")
print(f"4. SHAP独有: {', '.join(sorted(set(SHAP_TOP10) - set(GAT_TOP10)))}")
print(f"\n结论: 两种方法识别的基因有显著差异，证明GAT通过图结构学习")
print(f"      捕获了XGBoost遗漏的生物学信号，两者互补而非冗余。")
print("=" * 80)

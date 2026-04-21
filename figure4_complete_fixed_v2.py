#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 Complete Analysis Pipeline - Based on Exploration Results
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import scipy.sparse
import requests
import time
from tqdm import tqdm
from scipy.stats import chi2_contingency
import gseapy as gp
import json

# Set matplotlib font to Arial (required for publication)
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

def _fix_fonts():
    """scanpy 的 sc.pl.* 会重置 rcParams，每次 savefig 前调用此函数恢复。"""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # 强制更新当前图形的所有文本对象
    fig = plt.gcf()
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontname('Arial')

# Settings
sc.settings.verbosity = 3
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
class Config:
    BASE_DIR = Path.home() / "statics/GEO_data/GSE/figure4"
    RAW_DATA_DIR = BASE_DIR / "data/raw"
    PROCESSED_DIR = BASE_DIR / "data/processed"
    FIG_DIR = BASE_DIR / "figs"
    RESULTS_DIR = BASE_DIR / "results"

    RAW_H5AD = RAW_DATA_DIR / "4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
    TARGET_GENES_FILE = BASE_DIR / "imidazoline_SLE_intersection.csv"

    @classmethod
    def setup(cls):
        for d in [cls.PROCESSED_DIR, cls.FIG_DIR, cls.RESULTS_DIR,
                  cls.RESULTS_DIR / 'enrichr_results']:
            d.mkdir(parents=True, exist_ok=True)
        print("✅ Directories created successfully")

Config.setup()

# ==================== 1. Data Loading ====================
print("\n" + "="*80)
print("1. Loading Raw Data")
print("="*80)

adata = sc.read_h5ad(Config.RAW_H5AD)
print(f"Number of cells: {adata.n_obs:,}")
print(f"Number of genes: {adata.n_vars:,}")

# ==================== 2. Gene Name Conversion ====================
print("\n" + "="*80)
print("2. Gene Name Conversion (ENSG → Gene Symbol)")
print("="*80)

# Directly use the built-in feature_name column from h5ad, no API query needed
adata.var['Gene_Symbol'] = adata.var['feature_name'].values
adata.var['original_id'] = adata.var_names
adata.var_names = adata.var['feature_name'].values
adata.var_names_make_unique()

# Clear raw data to avoid gene name mismatch issues
# 但先从 raw counts 计算 QC 指标，避免用 log-normalized X 导致 n_genes_by_counts 为常数
if adata.raw is not None:
    print("📊 从 adata.raw 计算 QC 指标（raw counts）...")
    adata_raw_tmp = adata.raw.to_adata()
    adata_raw_tmp.var_names = adata.var_names  # 同步已转换的基因名
    adata_raw_tmp.var['mt'] = adata_raw_tmp.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_raw_tmp, qc_vars=['mt'],
                                percent_top=None, log1p=False, inplace=True)
    for col in ['n_genes_by_counts','total_counts','pct_counts_mt']:
        if col in adata_raw_tmp.obs.columns:
            adata.obs[col] = adata_raw_tmp.obs[col].values
    del adata_raw_tmp
    adata.raw = None
    print("⚠️  Cleared adata.raw to avoid gene name mismatch")
else:
    print("⚠️  adata.raw 不存在，将用 X 计算 QC（可能不准）")

gene_symbols = list(adata.var_names)

print(f"\nConversion results:")
print(f"  - Total genes: {len(gene_symbols)}")
print(f"  - Successfully mapped: {sum(1 for g in gene_symbols if not g.startswith('ENSG'))}")
print(f"  - Unmapped: {sum(1 for g in gene_symbols if g.startswith('ENSG'))}")

# Check key marker genes
key_genes = ['CD14', 'CD19', 'CD3D', 'CD4', 'CD8A', 'MS4A1', 'NKG7', 'GNLY']
found = [g for g in key_genes if g in adata.var_names]
print(f"\nKey genes found: {len(found)}/{len(key_genes)}")
print(f"Genes found: {found}")

# ==================== 2.5. Quality Control and Filtering ====================
print("\n" + "="*80)
print("2.5. Quality Control and Filtering")
print("="*80)

# QC 指标已在清除 raw 之前计算完毕（n_genes_by_counts / total_counts / pct_counts_mt）
# Identify mitochondrial genes（用于后续过滤逻辑）
adata.var['mt'] = adata.var_names.str.startswith('MT-')
n_mt_genes = adata.var['mt'].sum()
print(f"Number of mitochondrial genes: {n_mt_genes}")

print(f"\nQC metrics summary:")
print(f"  n_genes_by_counts - median: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"  total_counts - median: {adata.obs['total_counts'].median():.2f}")
print(f"  pct_counts_mt - median: {adata.obs['pct_counts_mt'].median():.2f}%")

# Plot QC metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Number of genes per cell
axes[0, 0].hist(adata.obs['n_genes_by_counts'], bins=100, edgecolor='black')
axes[0, 0].set_xlabel('Number of Genes per Cell')
axes[0, 0].set_ylabel('Number of Cells')
axes[0, 0].set_title('Distribution of Genes per Cell')
axes[0, 0].axvline(500, color='red', linestyle='--', label='Threshold=500')
axes[0, 0].axvline(1000, color='orange', linestyle='--', label='Threshold=1000')
axes[0, 0].legend()

# Total counts per cell
axes[0, 1].hist(adata.obs['total_counts'], bins=100, edgecolor='black')
axes[0, 1].set_xlabel('Total Counts per Cell')
axes[0, 1].set_ylabel('Number of Cells')
axes[0, 1].set_title('Distribution of Total Counts per Cell')

# Mitochondrial gene percentage
axes[1, 0].hist(adata.obs['pct_counts_mt'], bins=100, edgecolor='black', range=(0, 30))
axes[1, 0].set_xlabel('Mitochondrial Gene Percentage (%)')
axes[1, 0].set_ylabel('Number of Cells')
axes[1, 0].set_title('Distribution of Mitochondrial Gene %')
axes[1, 0].axvline(5, color='green', linestyle='--', label='Threshold=5%')
axes[1, 0].axvline(10, color='orange', linestyle='--', label='Threshold=10%')
axes[1, 0].axvline(15, color='red', linestyle='--', label='Threshold=15%')
axes[1, 0].legend()

# Scatter: genes vs counts colored by mt%
scatter = axes[1, 1].scatter(adata.obs['total_counts'],
                             adata.obs['n_genes_by_counts'],
                             c=adata.obs['pct_counts_mt'],
                             s=1, alpha=0.3, cmap='viridis',
                             vmin=0, vmax=20)
axes[1, 1].set_xlabel('Total Counts')
axes[1, 1].set_ylabel('Number of Genes')
axes[1, 1].set_title('Genes vs Counts (colored by MT%)')
plt.colorbar(scatter, ax=axes[1, 1], label='MT%')

plt.tight_layout()
_fix_fonts()
plt.savefig(Config.FIG_DIR / 'QC_metrics.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ QC metrics plot saved: QC_metrics.pdf")

# Filter cells based on QC metrics
print(f"\nFiltering cells...")
print(f"Cells before filtering: {adata.n_obs:,}")

# Filter: mitochondrial percentage < 10%
adata = adata[adata.obs['pct_counts_mt'] < 10, :].copy()
print(f"Cells after MT% filtering (<10%): {adata.n_obs:,}")

# Additional filter: remove cells with very low gene counts
adata = adata[adata.obs['n_genes_by_counts'] > 200, :].copy()
print(f"Cells after gene count filtering (>200): {adata.n_obs:,}")

print(f"\n✅ Quality control completed")
print(f"Final cell count: {adata.n_obs:,}")

# ==================== 2.6. Remove RBC Contamination ====================
print("\n" + "="*80)
print("2.6. Remove RBC Contamination")
print("="*80)

rbc_genes = [g for g in ['HBB', 'HBA1', 'HBA2'] if g in adata.var_names]
print(f"Using RBC marker genes: {rbc_genes}")

# Calculate total hemoglobin gene expression per cell
rbc_expr = np.array(adata[:, rbc_genes].X.sum(axis=1)).flatten()
rbc_threshold = np.percentile(rbc_expr, 99)  # Remove top 1% high RBC expression
keep_mask = rbc_expr <= rbc_threshold
n_before = adata.n_obs
adata = adata[keep_mask].copy()
print(f"Cells before RBC filtering: {n_before:,}")
print(f"Cells after RBC filtering: {adata.n_obs:,}  (Removed {n_before - adata.n_obs:,} RBC-contaminated cells)")


print("\n" + "="*80)
print("3. Load Target Gene List")
print("="*80)

target_df = pd.read_csv(Config.TARGET_GENES_FILE)
target_genes = target_df.iloc[:, 0].tolist()
print(f"Total target genes: {len(target_genes)}")

# Find existing targets in the data
existing_targets = [g for g in target_genes if g in adata.var_names]
print(f"Targets found in data: {len(existing_targets)}/{len(target_genes)}")

# ⚠️ Filter targets: keep only genes with actual expression in immune cells
print("\nDiagnostic: Target gene expression in immune cells...")
expr_matrix = adata[:, existing_targets].X
if hasattr(expr_matrix, 'toarray'):
    expr_matrix = expr_matrix.toarray()

pct_expressed = (expr_matrix > 0).mean(axis=0) * 100
mean_expr = np.array(expr_matrix.mean(axis=0)).flatten()

target_expr_df = pd.DataFrame({
    'gene': existing_targets,
    'pct_cells_expressed': pct_expressed,
    'mean_expression': mean_expr
}).sort_values('pct_cells_expressed', ascending=False)

print(f"  - Expressed in >10% cells: {(target_expr_df['pct_cells_expressed'] > 10).sum()}")
print(f"  - Expressed in >5% cells: {(target_expr_df['pct_cells_expressed'] > 5).sum()}")
print(f"  - Expressed in >1% cells: {(target_expr_df['pct_cells_expressed'] > 1).sum()}")

# Filter using threshold (expressed in at least 5% of cells)
MIN_PCT_CELLS = 5.0
filtered_targets = target_expr_df[target_expr_df['pct_cells_expressed'] > MIN_PCT_CELLS]['gene'].tolist()
print(f"\n✅ Filtered target genes: {len(filtered_targets)} (expressed in >{MIN_PCT_CELLS}% cells)")
print(f"Top 10 highly expressed targets: {filtered_targets[:10]}")

# Save diagnostic info
target_expr_df.to_csv(Config.RESULTS_DIR / 'target_gene_expression_diagnosis.csv', index=False)
print(f"Diagnostic file saved: target_gene_expression_diagnosis.csv")

# Use filtered targets
existing_targets = filtered_targets

# ==================== 4. Figure 4A: Target Activity Score ====================
print("\n" + "="*80)
print("4. Figure 4A: Calculate Target Activity Score")
print("="*80)

# Calculate score
sc.tl.score_genes(adata, gene_list=existing_targets,
                  score_name='target_score',
                  ctrl_size=50,
                  use_raw=False)

print(f"Score statistics:")
print(f"  Min: {adata.obs['target_score'].min():.4f}")
print(f"  Max: {adata.obs['target_score'].max():.4f}")
print(f"  Median: {adata.obs['target_score'].median():.4f}")

# Abbreviation → Full name mapping (for all plots)
celltype_names = {
    'T4':    'CD4+ T cell',
    'T8':    'CD8+ T cell',
    'B':     'B cell',
    'cM':    'Classical Monocyte',
    'NK':    'NK cell',
    'ncM':   'Non-classical Monocyte',
    'cDC':   'Classical DC',
    'Prolif':'Proliferating cells',
    'pDC':   'Plasmacytoid DC',
    'PB':    'Plasmablast',
    'Progen':'Progenitor'
}

DISEASE_LABEL = {
    'normal': 'HC',
    'systemic lupus erythematosus': 'SLE',
}
DISEASE_COLOR = {'HC': '#4393C3', 'SLE': '#D6604D'}
adata.obs['disease_label'] = adata.obs['disease'].astype(str).map(DISEASE_LABEL).fillna(adata.obs['disease'].astype(str))

# Plot Figure 4A: 1×3
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel A1: UMAP by target score
sc.pl.umap(adata, color='target_score', ax=axes[0],
           cmap='viridis', title='Target Activity Score (UMAP)',
           show=False)
_fix_fonts()  # Fix fonts after scanpy plot

# Panel A2: Boxplot by major cell types
main_types = ['T4', 'T8', 'B', 'cM', 'NK', 'pDC']
available_types = [t for t in main_types if t in adata.obs['author_cell_type'].unique()]
data_for_violin = [adata.obs.loc[adata.obs['author_cell_type'] == t, 'target_score'].values
                   for t in available_types]
bp = axes[1].boxplot(data_for_violin, patch_artist=True, showfliers=False)
ct_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_types)))
for patch, color in zip(bp['boxes'], ct_colors):
    patch.set_facecolor(color)
axes[1].set_xticks(range(1, len(available_types) + 1))
axes[1].set_xticklabels([celltype_names.get(t, t) for t in available_types], rotation=45, ha='right')
axes[1].set_ylabel('Target Activity Score')
axes[1].set_title('Score by Cell Type')
axes[1].spines[['top', 'right']].set_visible(False)

# Panel A3: HC vs SLE target score — per donor, then show distribution
# Use donor-level mean score to respect independence
from scipy.stats import mannwhitneyu
_donor_score = adata.obs.groupby('donor_id').agg(
    disease_label=('disease_label', 'first'),
    mean_score=('target_score', 'mean')
).reset_index()
hc_score  = _donor_score.loc[_donor_score['disease_label'] == 'HC',  'mean_score'].values
sle_score = _donor_score.loc[_donor_score['disease_label'] == 'SLE', 'mean_score'].values
_, p_hc_sle = mannwhitneyu(sle_score, hc_score, alternative='two-sided')
groups = [hc_score, sle_score]
bp3 = axes[2].boxplot(groups, patch_artist=True, showfliers=False, widths=0.5)
for patch, grp in zip(bp3['boxes'], ['HC', 'SLE']):
    patch.set_facecolor(DISEASE_COLOR[grp])
    patch.set_alpha(0.8)
for i, (grp, vals) in enumerate(zip(['HC', 'SLE'], groups)):
    axes[2].scatter(np.random.normal(i + 1, 0.05, len(vals)), vals,
                    color=DISEASE_COLOR[grp], s=30, alpha=0.7, zorder=3)
y_top = max(np.max(hc_score), np.max(sle_score)) * 1.1
axes[2].plot([1, 2], [y_top, y_top], 'k-', linewidth=0.8)
sig = '***' if p_hc_sle < 0.001 else '**' if p_hc_sle < 0.01 else '*' if p_hc_sle < 0.05 else 'ns'
axes[2].text(1.5, y_top * 1.02, f'{sig}\np={p_hc_sle:.2e}', ha='center', fontsize=9)
axes[2].set_xticks([1, 2])
axes[2].set_xticklabels(['HC', 'SLE'])
axes[2].set_ylabel('Mean Target Activity Score (per donor)')
axes[2].set_title('Target Score: HC vs SLE\n(donor-level, Mann-Whitney)')
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
_fix_fonts()  # Final font fix before saving
plt.savefig(Config.FIG_DIR / 'Figure4A_target_score.pdf', bbox_inches='tight')
plt.savefig(Config.FIG_DIR / 'Figure4A_target_score.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4A saved")

# ==================== 5. Figure 4B: High-score Cell Subpopulation ====================
print("\n" + "="*80)
print("5. Figure 4B: High-score Cell Subpopulation Identification")
print("="*80)

threshold = np.percentile(adata.obs['target_score'], 80)
adata.obs['is_high_score'] = adata.obs['target_score'] > threshold
print(f"Threshold (top 20%): {threshold:.4f}")
print(f"Number of high-score cells: {adata.obs['is_high_score'].sum():,} ({adata.obs['is_high_score'].mean()*100:.1f}%)")

# ── Figure 4_S1: Threshold Justification (donor-level boxplot) ───────────────
print("\nGenerating threshold justification figure (Figure4_S1)...")
from scipy.stats import mannwhitneyu as _mwu

_HC_C, _SLE_C, _THR_C = '#4393C3', '#D6604D', '#2166AC'
_rng = np.random.default_rng(42)

_donor_pct = (adata.obs.groupby(['donor_id', 'disease_label'])['is_high_score']
              .mean().mul(100).reset_index())
_donor_pct.columns = ['donor_id', 'disease_label', 'pct']
_hc_vals  = _donor_pct.loc[_donor_pct['disease_label'] == 'HC',  'pct'].values
_sle_vals = _donor_pct.loc[_donor_pct['disease_label'] == 'SLE', 'pct'].values

fig_s1, ax = plt.subplots(figsize=(4, 5))
for _i, (_vals, _col) in enumerate([(_hc_vals, _HC_C), (_sle_vals, _SLE_C)]):
    ax.scatter(_rng.normal(_i, 0.07, size=len(_vals)), _vals,
               color=_col, alpha=0.65, s=40, zorder=3)
    ax.boxplot(_vals, positions=[_i], widths=0.4,
               patch_artist=True,
               boxprops=dict(facecolor=_col, alpha=0.3),
               medianprops=dict(color='black', lw=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker=''))

_stat, _pval = _mwu(_sle_vals, _hc_vals, alternative='greater')
_y_top = max(_hc_vals.max(), _sle_vals.max()) * 1.06
ax.plot([0, 1], [_y_top, _y_top], 'k-', lw=1)
_sig = '***' if _pval < 0.001 else '**' if _pval < 0.01 else '*' if _pval < 0.05 else 'ns'
ax.text(0.5, _y_top * 1.02, f'p = {_pval:.2e}  {_sig}', ha='center', fontsize=9)
ax.axhline(20, color=_THR_C, lw=1.2, linestyle='--', alpha=0.6, label='Threshold = 20%')
ax.set_xticks([0, 1])
ax.set_xticklabels(['HC', 'SLE'], fontsize=12)
ax.set_ylabel('% Sensitive Cells per Donor\n(Top 20% Target Activity Score)', fontsize=10)
ax.set_title('Sensitive Cell Threshold Justification', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, frameon=False)
ax.spines[['top', 'right']].set_visible(False)

fig_s1.tight_layout()
_fix_fonts()
fig_s1.savefig(Config.FIG_DIR / 'Figure4_S1_threshold_justification.pdf', dpi=300, bbox_inches='tight')
fig_s1.savefig(Config.FIG_DIR / 'Figure4_S1_threshold_justification.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Figure4_S1_threshold_justification saved  (donor MWU p={_pval:.2e})")

# Extract high-score cells
high_score_adata = adata[adata.obs['is_high_score']].copy()

# Re-cluster high-score cells (adata.X is already log1p normalized, no need to repeat)
sc.pp.filter_genes(high_score_adata, min_cells=3)
sc.pp.highly_variable_genes(high_score_adata, n_top_genes=2000)
sc.pp.scale(high_score_adata, max_value=10)
sc.tl.pca(high_score_adata, svd_solver='arpack')
sc.pp.neighbors(high_score_adata, n_pcs=30)
sc.tl.leiden(high_score_adata, resolution=0.5, key_added='high_score_cluster')
sc.tl.umap(high_score_adata)

# Plot Figure 4B: 2×2
fig4b, axes4b = plt.subplots(2, 2, figsize=(14, 12))

# B1 (0,0): Highlight sensitive cells on full UMAP
adata.obs['highlight'] = adata.obs['is_high_score'].map({True: 'Sensitive', False: 'Other'})
sc.pl.umap(adata, color='highlight', ax=axes4b[0, 0],
           palette={'Sensitive': '#E64B35', 'Other': '#DDDDDD'},
           title='Sensitive Cells (Top 20% Score)',
           show=False)
_fix_fonts()  # Fix fonts after scanpy plot

# B2 (0,1): Sensitive cells colored by cell type
high_score_adata.obs['Cell Type'] = high_score_adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(high_score_adata.obs['author_cell_type'].astype(str))
sc.pl.umap(high_score_adata, color='Cell Type', ax=axes4b[0, 1],
           title='Sensitive Cells by Cell Type', show=False)
_fix_fonts()  # Fix fonts after scanpy plot

# B3 (1,0): Full UMAP colored by disease (HC vs SLE)
sc.pl.umap(adata, color='disease_label', ax=axes4b[1, 0],
           palette=DISEASE_COLOR,
           title='All Cells by Disease (HC vs SLE)',
           show=False)
_fix_fonts()  # Fix fonts after scanpy plot

# B4 (1,1): Per-cell-type proportion of sensitive cells in HC vs SLE
# For each major cell type: compute fraction of sensitive cells among HC and SLE separately
_ct_disease = adata.obs.groupby(['author_cell_type', 'disease_label'])['is_high_score'].mean() * 100
_ct_disease = _ct_disease.unstack('disease_label').reindex(available_types).dropna(how='all')
x4 = np.arange(len(_ct_disease))
w4 = 0.35
axes4b[1, 1].bar(x4 - w4/2, _ct_disease.get('HC',  pd.Series(0, index=_ct_disease.index)),
                 w4, label='HC',  color=DISEASE_COLOR['HC'],  alpha=0.85)
axes4b[1, 1].bar(x4 + w4/2, _ct_disease.get('SLE', pd.Series(0, index=_ct_disease.index)),
                 w4, label='SLE', color=DISEASE_COLOR['SLE'], alpha=0.85)
axes4b[1, 1].set_xticks(x4)
axes4b[1, 1].set_xticklabels([celltype_names.get(t, t) for t in _ct_disease.index],
                               rotation=45, ha='right', fontsize=9)
axes4b[1, 1].set_ylabel('Sensitive Cell Fraction (%)')
axes4b[1, 1].set_title('Sensitive Cell Fraction per Cell Type\n(HC vs SLE)')
axes4b[1, 1].legend(frameon=False, fontsize=9)
axes4b[1, 1].spines[['top', 'right']].set_visible(False)

fig4b.suptitle('Figure 4B: Sensitive Cell Subpopulation — HC vs SLE', fontsize=14, fontweight='bold', y=1.01)
fig4b.tight_layout()
_fix_fonts()  # Final font fix before saving
fig4b.savefig(Config.FIG_DIR / 'Figure4B_high_score_subset.pdf', bbox_inches='tight')
fig4b.savefig(Config.FIG_DIR / 'Figure4B_high_score_subset.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4B saved")

# ==================== 6. Figure 4C: Disease Specificity Analysis ====================
print("\n" + "="*80)
print("6. Figure 4C: Disease Specificity Analysis")
print("="*80)

# Statistics by donor (each donor belongs to only one disease, no Cartesian product grouping)
sample_stats = adata.obs.groupby('donor_id').agg(
    disease=('disease', 'first'),
    total_cells=('is_high_score', 'count'),
    high_score_cells=('is_high_score', lambda x: (x == True).sum())
).reset_index()
sample_stats['high_score_percent'] = sample_stats['high_score_cells'] / sample_stats['total_cells'] * 100

print(f"Sample statistics (n={len(sample_stats)} donors):")
print(sample_stats.head())

# Plot boxplot
fig, ax = plt.subplots(figsize=(8, 6))

diseases = ['normal', 'systemic lupus erythematosus']
data_to_plot = []
for d in diseases:
    data = sample_stats[sample_stats['disease'] == d]['high_score_percent'].values
    data_to_plot.append(data)

bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False)
colors = ['#7fbf7f', '#ff7f7f']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add scatter points
for i, data in enumerate(data_to_plot):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(x, data, alpha=0.6, s=30, color='black', zorder=3)

ax.set_xticklabels(['Healthy Control', 'SLE'])
ax.set_ylabel('High-score Cell Proportion (%)')
ax.set_title('High-score Cell Proportion in SLE vs HC')

# Mann-Whitney U test at donor level (correct statistical unit)
from scipy.stats import mannwhitneyu
hc_vals = sample_stats[sample_stats['disease'] == 'normal']['high_score_percent'].values
sle_vals = sample_stats[sample_stats['disease'] == 'systemic lupus erythematosus']['high_score_percent'].values
stat, p_value = mannwhitneyu(sle_vals, hc_vals, alternative='two-sided')
print(f"\nMann-Whitney U test (donor level, n_HC={len(hc_vals)}, n_SLE={len(sle_vals)})")
print(f"p-value: {p_value:.4e}")
ax.text(0.5, 0.95, f'p = {p_value:.2e} (Mann-Whitney, donor-level)', transform=ax.transAxes, ha='center')

plt.tight_layout()
_fix_fonts()
plt.savefig(Config.FIG_DIR / 'Figure4C_disease_specificity.pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Figure 4C saved")

# ==================== 7. Figure 4D: Marker Gene Analysis ====================
print("\n" + "="*80)
print("7. Figure 4D: Marker Gene Analysis")
print("="*80)

# Correct method: Pseudobulk differential analysis (donor-level statistical unit)
print("Computing pseudobulk differential expression (donor-level)...")

# Step 1: Calculate cell-level fold change for pre-filtering
high_cells = adata[adata.obs['is_high_score'] == True]
low_cells = adata[adata.obs['is_high_score'] == False]

if scipy.sparse.issparse(adata.X):
    high_mean = np.array(high_cells.X.mean(axis=0)).flatten()
    low_mean = np.array(low_cells.X.mean(axis=0)).flatten()
else:
    high_mean = high_cells.X.mean(axis=0)
    low_mean = low_cells.X.mean(axis=0)

logFC_cell = high_mean - low_mean

# Pre-filter: only test genes with |logFC| >= 0.25 (reduce computation)
candidate_mask = np.abs(logFC_cell) >= 0.25
candidate_genes = adata.var_names[candidate_mask].tolist()
print(f"Candidate genes (cell-level |logFC| >= 0.25): {len(candidate_genes)}")

# Step 2: Build pseudobulk matrix (aggregate by donor, use mean)
print(f"\nBuilding pseudobulk profiles...")
donors = adata.obs['donor_id'].unique()
n_donors = len(donors)

pseudobulk_high = np.zeros((n_donors, len(candidate_genes)))
pseudobulk_low = np.zeros((n_donors, len(candidate_genes)))
donor_has_high = []
donor_has_low = []

for i, donor in enumerate(donors):
    donor_cells = adata[adata.obs['donor_id'] == donor]

    # High-score cells in this donor
    high_mask = donor_cells.obs['is_high_score'] == True
    if high_mask.sum() > 0:
        donor_high = donor_cells[high_mask]
        if scipy.sparse.issparse(donor_high.X):
            pseudobulk_high[i] = np.array(donor_high[:, candidate_genes].X.mean(axis=0)).flatten()
        else:
            pseudobulk_high[i] = donor_high[:, candidate_genes].X.mean(axis=0)
        donor_has_high.append(True)
    else:
        donor_has_high.append(False)

    # Low-score cells in this donor
    low_mask = donor_cells.obs['is_high_score'] == False
    if low_mask.sum() > 0:
        donor_low = donor_cells[low_mask]
        if scipy.sparse.issparse(donor_low.X):
            pseudobulk_low[i] = np.array(donor_low[:, candidate_genes].X.mean(axis=0)).flatten()
        else:
            pseudobulk_low[i] = donor_low[:, candidate_genes].X.mean(axis=0)
        donor_has_low.append(True)
    else:
        donor_has_low.append(False)

# Only keep donors with both groups
valid_donors = np.array(donor_has_high) & np.array(donor_has_low)
pseudobulk_high = pseudobulk_high[valid_donors]
pseudobulk_low = pseudobulk_low[valid_donors]

print(f"Valid donors (have both high and low cells): {valid_donors.sum()}/{n_donors}")

# Step 3: Paired t-test (paired test, high vs low within same donor)
from scipy.stats import ttest_rel
pvals = []
logFC_pseudobulk = []

for j in range(len(candidate_genes)):
    high_expr = pseudobulk_high[:, j]
    low_expr = pseudobulk_low[:, j]

    # Paired t-test
    try:
        stat, pval = ttest_rel(high_expr, low_expr, alternative='two-sided')
        pvals.append(pval)
    except:
        pvals.append(1.0)

    # LogFC in pseudobulk space
    logFC_pseudobulk.append(high_expr.mean() - low_expr.mean())

# FDR correction
from statsmodels.stats.multitest import fdrcorrection
_, p_adj = fdrcorrection(pvals)

# Build result table
markers = pd.DataFrame({
    'gene': candidate_genes,
    'logFC': logFC_pseudobulk,
    'p_val': pvals,
    'p_val_adj': p_adj
}).sort_values('p_val_adj')

print(f"\nDiagnostic - Markers data distribution:")
print(f"  Total genes tested: {len(markers)}")
print(f"  logFC range: [{markers['logFC'].min():.3f}, {markers['logFC'].max():.3f}]")
print(f"  p_val_adj < 0.05: {(markers['p_val_adj'] < 0.05).sum()}")
print(f"  p_val_adj < 0.01: {(markers['p_val_adj'] < 0.01).sum()}")

# Filter significant genes
significant_markers = markers[
    (markers['logFC'].abs() >= 0.25) &
    (markers['p_val_adj'] < 0.05)
].copy()

print(f"\n✅ Significant marker genes (donor-level, |logFC|>=0.25, FDR<0.05): {len(significant_markers)}")
print(significant_markers.head(20)[['gene', 'logFC', 'p_val_adj']].to_string())

# Save results
significant_markers.to_csv(Config.RESULTS_DIR / 'marker_genes.csv', index=False)

# Plot: volcano + dotplot side by side
if len(significant_markers) >= 5:
    top_genes = significant_markers.head(10)['gene'].tolist()
    adata.obs['Cell Type'] = adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(adata.obs['author_cell_type'].astype(str))

    # ── Volcano plot ──────────────────────────────────────────────────────────
    fig_v, ax_v = plt.subplots(figsize=(7, 6))
    neg_log10_fdr = -np.log10(markers['p_val_adj'].clip(lower=1e-300))
    is_sig = (markers['logFC'].abs() >= 0.25) & (markers['p_val_adj'] < 0.05)
    colors_v = np.where(markers['logFC'] >= 0.25, '#E64B35',
               np.where(markers['logFC'] <= -0.25, '#4DBBD5', '#AAAAAA'))
    ax_v.scatter(markers.loc[~is_sig, 'logFC'], neg_log10_fdr[~is_sig],
                 c='#CCCCCC', s=8, alpha=0.5, rasterized=True, label='NS')
    ax_v.scatter(markers.loc[is_sig & (markers['logFC'] > 0), 'logFC'],
                 neg_log10_fdr[is_sig & (markers['logFC'] > 0)],
                 c='#E64B35', s=12, alpha=0.7, rasterized=True, label='Up (|logFC|≥0.25, FDR<0.05)')
    ax_v.scatter(markers.loc[is_sig & (markers['logFC'] < 0), 'logFC'],
                 neg_log10_fdr[is_sig & (markers['logFC'] < 0)],
                 c='#4DBBD5', s=12, alpha=0.7, rasterized=True, label='Down (|logFC|≥0.25, FDR<0.05)')
    ax_v.axvline( 0.25, color='gray', linestyle='--', linewidth=0.8)
    ax_v.axvline(-0.25, color='gray', linestyle='--', linewidth=0.8)
    ax_v.axhline(-np.log10(0.05), color='gray', linestyle=':', linewidth=0.8)
    # Label top significant genes (use adjustText to avoid overlap)
    from adjustText import adjust_text
    _label_rows = significant_markers.head(20)
    _texts = []
    for _, row in _label_rows.iterrows():
        x = row['logFC']
        y = -np.log10(max(row['p_val_adj'], 1e-300))
        _texts.append(ax_v.text(x, y, row['gene'], fontsize=7))
    adjust_text(
        _texts,
        ax=ax_v,
        arrowprops=dict(arrowstyle='-', color='#888888', lw=0.5),
        expand=(1.5, 1.8),
        force_text=(0.5, 0.8),
        force_points=(0.3, 0.5),
        lim=500,
    )
    ax_v.set_xlabel('logFC (sensitive vs non-sensitive)', fontsize=11)
    ax_v.set_ylabel('-log10(FDR)', fontsize=11)
    ax_v.set_title('Differential Expression: Sensitive Cells\n(pseudobulk, paired t-test, BH correction)',
                   fontsize=11, fontweight='bold')
    ax_v.legend(fontsize=8, frameon=False)
    ax_v.spines[['top', 'right']].set_visible(False)
    fig_v.tight_layout()
    _fix_fonts()
    fig_v.savefig(Config.FIG_DIR / 'Figure4D_volcano.pdf', dpi=300, bbox_inches='tight')
    fig_v.savefig(Config.FIG_DIR / 'Figure4D_volcano.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Figure 4D volcano saved")

    # ── Dotplot ───────────────────────────────────────────────────────────────
    fig_d, ax_d = plt.subplots(figsize=(12, 6))
    sc.pl.dotplot(adata, top_genes, groupby='Cell Type',
                 standard_scale='var', ax=ax_d, show=False,
                 use_raw=False,
                 title='Top Marker Gene Expression by Cell Type\n(|logFC|≥0.25, FDR<0.05)')
    _fix_fonts()  # Fix fonts after scanpy plot
    plt.tight_layout()
    _fix_fonts()  # Final font fix before saving
    plt.savefig(Config.FIG_DIR / 'Figure4D_marker_genes.pdf', bbox_inches='tight')
    plt.savefig(Config.FIG_DIR / 'Figure4D_marker_genes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Figure 4D dotplot saved")

    # ── HC vs SLE pseudobulk DE for top sensitive marker genes ────────────────
    # Shows whether the sensitive-cell markers are also elevated in SLE vs HC at the bulk level
    print("\nComputing HC vs SLE pseudobulk DE for top sensitive marker genes...")
    top_de_genes = significant_markers.head(20)['gene'].tolist()
    top_de_genes = [g for g in top_de_genes if g in adata.var_names]

    _donors_hc_sle = adata.obs[['donor_id', 'disease_label']].drop_duplicates().set_index('donor_id')
    _pb_rows = []
    for donor in _donors_hc_sle.index:
        d_mask = adata.obs['donor_id'] == donor
        d_X = adata[d_mask, top_de_genes].X
        mean_vec = np.array(d_X.mean(axis=0)).flatten() if scipy.sparse.issparse(d_X) else d_X.mean(axis=0)
        _pb_rows.append({'donor_id': donor,
                         'disease_label': _donors_hc_sle.loc[donor, 'disease_label'],
                         **dict(zip(top_de_genes, mean_vec))})
    _pb_df = pd.DataFrame(_pb_rows).set_index('donor_id')

    from scipy.stats import ttest_ind
    _hc_pb  = _pb_df[_pb_df['disease_label'] == 'HC'].drop(columns='disease_label')
    _sle_pb = _pb_df[_pb_df['disease_label'] == 'SLE'].drop(columns='disease_label')

    _hc_sle_results = []
    for g in top_de_genes:
        _, pv = ttest_ind(_sle_pb[g].values, _hc_pb[g].values, equal_var=False)
        lfc   = _sle_pb[g].mean() - _hc_pb[g].mean()
        _hc_sle_results.append({'gene': g, 'logFC_SLE_vs_HC': lfc, 'p_val': pv})
    _hc_sle_df = pd.DataFrame(_hc_sle_results)
    _, _hc_sle_df['FDR'] = fdrcorrection(_hc_sle_df['p_val'].values)
    _hc_sle_df = _hc_sle_df.sort_values('logFC_SLE_vs_HC', ascending=False)
    _hc_sle_df.to_csv(Config.RESULTS_DIR / 'Figure4D_HCvsSLE_top_markers.csv', index=False)

    # Plot: bar chart logFC SLE vs HC for top sensitive markers, color by FDR
    from matplotlib.patches import Patch

    # ── 全局字体 Arial ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':     'Arial',
        'font.sans-serif': ['Arial'],
        'axes.unicode_minus': False,
    })

    _n_genes  = len(_hc_sle_df)
    _row_h    = 0.45                          # 每个基因占的高度(英寸)
    _fig_h    = max(5, _n_genes * _row_h + 1.5)
    fig_hs, ax_hs = plt.subplots(figsize=(8, _fig_h))

    _bar_colors = ['#E64B35' if f < 0.05 else '#AAAAAA' for f in _hc_sle_df['FDR']]
    ax_hs.barh(_hc_sle_df['gene'], _hc_sle_df['logFC_SLE_vs_HC'],
               color=_bar_colors, edgecolor='white', linewidth=0.4, height=0.7)
    ax_hs.axvline(0, color='black', linewidth=0.8)

    # FDR 标注：统一放在 x 轴最右侧，不与 bar 重叠
    _x_annot = _hc_sle_df['logFC_SLE_vs_HC'].abs().max() * 1.15
    for _, row in _hc_sle_df.iterrows():
        ax_hs.text(_x_annot, row['gene'],
                   f"{row['FDR']:.1e}",
                   va='center', ha='left', fontsize=7,
                   fontname='Arial',
                   color='#333333')
    ax_hs.text(_x_annot, _hc_sle_df['gene'].iloc[0],   # 列标题
               'FDR', va='bottom', ha='left', fontsize=7,
               fontname='Arial', fontweight='bold', color='#333333')

    ax_hs.legend(handles=[Patch(color='#E64B35', label='FDR < 0.05'),
                           Patch(color='#AAAAAA', label='FDR ≥ 0.05')],
                 frameon=False, fontsize=9, prop={'family': 'Arial'})
    ax_hs.set_xlabel('logFC (SLE vs HC)', fontsize=11)
    ax_hs.set_ylabel('Gene', fontsize=11)
    ax_hs.set_title('Top Sensitive-Cell Markers: SLE vs HC Expression\n'
                    '(pseudobulk, Welch t-test, BH FDR)',
                    fontsize=11, fontweight='bold')
    ax_hs.tick_params(axis='both', labelsize=9)
    for lbl in ax_hs.get_yticklabels() + ax_hs.get_xticklabels():
        lbl.set_fontname('Arial')
    ax_hs.spines[['top', 'right']].set_visible(False)

    fig_hs.tight_layout()
    _fix_fonts()
    fig_hs.savefig(Config.FIG_DIR / 'Figure4D_HCvsSLE_markers.pdf', dpi=300, bbox_inches='tight')
    fig_hs.savefig(Config.FIG_DIR / 'Figure4D_HCvsSLE_markers.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Figure 4D HC vs SLE marker bar chart saved")

# ==================== 8. Figure 4E: Functional Enrichment Analysis ====================
print("\n" + "="*80)
print("8. Figure 4E: Functional Enrichment Analysis")
print("="*80)

if len(significant_markers) > 0:
    # Use more genes for enrichment (same threshold)
    relaxed_markers = markers[
        (markers['logFC'].abs() >= 0.25) &
        (markers['p_val_adj'] < 0.05)
    ].head(100)  # Max 100 genes

    gene_list = relaxed_markers['gene'].tolist()
    print(f"Number of genes for enrichment: {len(gene_list)}")

    try:
        # ========== Part 1: Enrichr Analysis (GO and KEGG) ==========
        print("\n--- Part 1: Enrichr Analysis ---")
        print("Running GO Biological Process enrichment...")
        go_enrich = gp.enrichr(gene_list=gene_list,
                              gene_sets=['GO_Biological_Process_2021'],
                              organism='Human',
                              outdir=Config.RESULTS_DIR / 'enrichr_results',
                              no_plot=True)

        print("Running KEGG pathway enrichment...")
        kegg_enrich = gp.enrichr(gene_list=gene_list,
                                gene_sets=['KEGG_2021_Human'],
                                organism='Human',
                                outdir=Config.RESULTS_DIR / 'enrichr_results',
                                no_plot=True)

        # Display GO results first (priority)
        if go_enrich.results is not None and len(go_enrich.results) > 0:
            go_results = go_enrich.results[go_enrich.results['Adjusted P-value'] < 0.05].copy()

            print(f"\n✅ GO Biological Process enrichment results (Top 10):")
            print(go_results[['Term', 'P-value', 'Adjusted P-value', 'Genes']].head(10).to_string())

            # Save complete results
            go_results.to_csv(Config.RESULTS_DIR / 'GO_enrichment_results.csv', index=False)

            if len(go_results) > 0:
                # Plot GO bubble plot
                top_go = go_results.head(15).copy()
                top_go['gene_count'] = top_go['Genes'].str.split(';').str.len()
                top_go['-log10_pval'] = -np.log10(top_go['Adjusted P-value'])

                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(top_go['gene_count'],
                                   range(len(top_go)),
                                   s=top_go['-log10_pval'] * 50,
                                   c=top_go['Adjusted P-value'],
                                   cmap='viridis_r',
                                   alpha=0.7,
                                   edgecolors='black',
                                   linewidth=1)

                ax.set_yticks(range(len(top_go)))
                ax.set_yticklabels([t[:60] + '...' if len(t) > 60 else t for t in top_go['Term']], fontsize=9)
                plt.colorbar(scatter, ax=ax, label='Adjusted P-value')
                ax.set_xlabel('Gene Count', fontsize=12)
                ax.set_ylabel('GO Biological Process', fontsize=12)
                ax.set_title('Functional Enrichment of Sensitive Cell Markers (Enrichr)', fontsize=14)
                ax.invert_yaxis()

                plt.tight_layout()
                _fix_fonts()
                plt.savefig(Config.FIG_DIR / 'Figure4E_GO_enrichment.pdf', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✅ Figure 4E (GO Enrichr) saved")

        # Save KEGG results (for reference only)
        if kegg_enrich.results is not None and len(kegg_enrich.results) > 0:
            kegg_results = kegg_enrich.results[kegg_enrich.results['Adjusted P-value'] < 0.05].copy()
            kegg_results.to_csv(Config.RESULTS_DIR / 'KEGG_enrichment_results.csv', index=False)
            print(f"\n⚠️  KEGG enrichment results saved (for reference, may include non-immune pathways)")
            print(f"   File: KEGG_enrichment_results.csv")

        # ========== Part 2: GSEA Analysis ==========
        print("\n--- Part 2: GSEA Analysis ---")
        print("Running Gene Set Enrichment Analysis (GSEA)...")

        # Prepare ranked gene list for GSEA (all genes, ranked by logFC)
        # Use all markers (not just significant ones) to create a ranked list
        all_markers_ranked = markers.sort_values('logFC', ascending=False).copy()

        # Create ranking metric: use logFC * -log10(p_val)
        all_markers_ranked['rank_metric'] = all_markers_ranked['logFC'] * (-np.log10(all_markers_ranked['p_val'] + 1e-10))

        # Create pre-ranked gene list
        rnk = all_markers_ranked[['gene', 'rank_metric']].copy()
        rnk = rnk.sort_values('rank_metric', ascending=False)

        # Save ranked gene list
        rnk.to_csv(Config.RESULTS_DIR / 'ranked_gene_list_for_GSEA.csv', index=False)
        print(f"Ranked gene list saved: ranked_gene_list_for_GSEA.csv")
        print(f"Total genes in ranked list: {len(rnk)}")

        # Run GSEA prerank
        gsea_res = gp.prerank(
            rnk=rnk,
            gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'],
            outdir=Config.RESULTS_DIR / 'gsea_results',
            min_size=15,
            max_size=500,
            permutation_num=1000,
            seed=42,
            verbose=True
        )

        # Extract and save GSEA results
        if gsea_res.res2d is not None and len(gsea_res.res2d) > 0:
            gsea_results = gsea_res.res2d.copy()

            # Convert numeric columns to proper data types
            numeric_cols = ['NES', 'NOM p-val', 'FDR q-val', 'FWER p-val']
            for col in numeric_cols:
                if col in gsea_results.columns:
                    gsea_results[col] = pd.to_numeric(gsea_results[col], errors='coerce')

            # Filter significant results
            gsea_results_sig = gsea_results[gsea_results['FDR q-val'] < 0.25].copy()

            print(f"\n✅ GSEA results summary:")
            print(f"  Total gene sets tested: {len(gsea_results)}")
            print(f"  Significant (FDR < 0.25): {len(gsea_results_sig)}")
            print(f"  Upregulated (NES > 0, FDR < 0.25): {((gsea_results_sig['NES'] > 0)).sum()}")
            print(f"  Downregulated (NES < 0, FDR < 0.25): {((gsea_results_sig['NES'] < 0)).sum()}")

            # Save full GSEA results
            gsea_results.to_csv(Config.RESULTS_DIR / 'GSEA_all_results.csv', index=False)
            gsea_results_sig.to_csv(Config.RESULTS_DIR / 'GSEA_significant_results.csv', index=False)

            print(f"\nTop 10 enriched pathways (by NES):")
            if len(gsea_results_sig) > 0:
                print(gsea_results_sig.nlargest(min(10, len(gsea_results_sig)), 'NES')[['Term', 'NES', 'NOM p-val', 'FDR q-val']].to_string())
            else:
                print("  No significant pathways found")

            # Plot GSEA results
            if len(gsea_results_sig) > 0:
                n_top = min(15, len(gsea_results_sig))
                top_gsea = gsea_results_sig.nlargest(n_top, 'NES').copy()

                fig, ax = plt.subplots(figsize=(12, 8))

                # Color by p-value
                colors = -np.log10(top_gsea['FDR q-val'].astype(float) + 1e-10)

                # Color by p-value
                colors = -np.log10(top_gsea['FDR q-val'].astype(float) + 1e-10)

                bars = ax.barh(range(len(top_gsea)), top_gsea['NES'].astype(float),
                              color=plt.cm.RdYlBu_r(colors / colors.max()),
                              edgecolor='black', linewidth=0.5)

                ax.set_yticks(range(len(top_gsea)))
                ax.set_yticklabels([t[:60] + '...' if len(t) > 60 else t for t in top_gsea['Term']], fontsize=9)
                ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12)
                ax.set_ylabel('Gene Set', fontsize=12)
                ax.set_title('GSEA: Top Enriched Pathways in High-score Cells', fontsize=14)
                ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                          norm=plt.Normalize(vmin=0, vmax=colors.max()))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('-log10(FDR q-val)', fontsize=10)

                plt.tight_layout()
                _fix_fonts()
                plt.savefig(Config.FIG_DIR / 'Figure4E_GSEA_enrichment.pdf', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✅ Figure 4E (GSEA) saved")
            else:
                print("⚠️  No significant pathways to plot")
        else:
            print("⚠️  No significant GSEA results found")

    except Exception as e:
        print(f"Enrichment analysis failed: {e}")
        import traceback
        traceback.print_exc()

# ==================== 9. Save All Results ====================
print("\n" + "="*80)
print("9. Save Processed Data")
print("="*80)

adata.write(Config.PROCESSED_DIR / 'adata_final.h5ad')
high_score_adata.write(Config.PROCESSED_DIR / 'adata_high_score.h5ad')

print(f"✅ Data saved")

# ==================== 9.5. Extract Key Outputs for Figure 4 (for follow-up research) ====================
print("\n" + "="*80)
print("9.5. Extract Key Outputs for Figure 4")
print("="*80)

# 1. Figure 4D: Top 50 sensitive cell marker genes
top50_markers = significant_markers.head(50) if len(significant_markers) >= 50 else significant_markers
top50_markers.to_csv(Config.RESULTS_DIR / 'Figure4D_sensitive_cell_markers_top50.csv', index=False)
print(f"✅ Top 50 sensitive cell marker genes saved: {len(top50_markers)} genes")
print(f"   File: {Config.RESULTS_DIR / 'Figure4D_sensitive_cell_markers_top50.csv'}")

# 2. Figure 4A: Compound target activity scores
target_score_df = pd.DataFrame({
    'cell_barcode': adata.obs_names,
    'target_activity_score': adata.obs['target_score'].values,
    'donor_id': adata.obs['donor_id'].values,
    'disease': adata.obs['disease'].values,
    'cell_type': adata.obs['author_cell_type'].values
})
target_score_df.to_csv(Config.RESULTS_DIR / 'Figure4A_target_activity_scores.csv', index=False)
print(f"✅ Compound target activity scores saved: {len(target_score_df)} cells")
print(f"   File: {Config.RESULTS_DIR / 'Figure4A_target_activity_scores.csv'}")

# 3. Figure 4B: Sensitive cell labels (add sensitive_cell column)
adata.obs['sensitive_cell'] = adata.obs['is_high_score']
sensitive_cells_df = pd.DataFrame({
    'cell_barcode': adata.obs_names,
    'sensitive_cell': adata.obs['sensitive_cell'].values,
    'target_score': adata.obs['target_score'].values,
    'donor_id': adata.obs['donor_id'].values,
    'disease': adata.obs['disease'].values,
    'cell_type': adata.obs['author_cell_type'].values
})
sensitive_cells_df.to_csv(Config.RESULTS_DIR / 'Figure4B_sensitive_cell_labels.csv', index=False)
print(f"✅ Sensitive cell labels saved: {(adata.obs['sensitive_cell'] == True).sum()} sensitive / {len(adata.obs)} total")
print(f"   File: {Config.RESULTS_DIR / 'Figure4B_sensitive_cell_labels.csv'}")

# Re-save complete data with sensitive_cell labels
adata.write(Config.PROCESSED_DIR / 'adata_final_with_sensitive_labels.h5ad')
print(f"✅ Complete data with sensitive cell labels saved")
print(f"   File: {Config.PROCESSED_DIR / 'adata_final_with_sensitive_labels.h5ad'}")

print("\n" + "="*80)
print("Key Output Files Summary:")
print("="*80)
print(f"1. Top 50 sensitive cell marker genes: Figure4D_sensitive_cell_markers_top50.csv")
print(f"2. Compound target activity scores: Figure4A_target_activity_scores.csv")
print(f"3. Sensitive cell labels: Figure4B_sensitive_cell_labels.csv")
print(f"4. Complete data (with labels): adata_final_with_sensitive_labels.h5ad")
print("="*80)

# ==================== 10. Figure 4F: Sensitive Cell Subtype Characterization ====================
print("\n" + "="*80)
print("10. Figure 4F: Sensitive Cell Subtype Characterization")
print("="*80)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# Use human-readable cell type labels
ct_col = 'author_cell_type'
adata.obs['cell_type_label'] = (
    adata.obs[ct_col].astype(str)
    .map(celltype_names)
    .fillna(adata.obs[ct_col].astype(str))
)

sensitive_adata = adata[adata.obs['is_high_score'] == True].copy()
sensitive_obs   = sensitive_adata.obs
all_obs         = adata.obs

# Top marker genes to use for subtype annotation (from 4D pseudobulk DE)
TOP_MARKER_GENES = significant_markers.head(30)['gene'].tolist()
TOP_MARKER_GENES = [g for g in TOP_MARKER_GENES if g in adata.var_names]

# ── A. Composition stats (enrichment ratio) ───────────────────────────────────
sens_counts = sensitive_obs['cell_type_label'].value_counts()
all_counts  = all_obs['cell_type_label'].value_counts()
df_comp = pd.DataFrame({'Sensitive_cells': sens_counts, 'All_cells': all_counts}).fillna(0).astype(int)
df_comp['Sensitive_pct']    = df_comp['Sensitive_cells'] / df_comp['Sensitive_cells'].sum() * 100
df_comp['Background_pct']   = df_comp['All_cells'] / df_comp['All_cells'].sum() * 100
df_comp['Enrichment_ratio'] = (df_comp['Sensitive_pct'] + 1e-3) / (df_comp['Background_pct'] + 1e-3)
df_comp = df_comp.sort_values('Sensitive_pct', ascending=False)

# ── B. Per-cell-type top marker identification within sensitive cells ──────────
# For each cell type: compute mean log1p expression of top marker genes
# → find top-2 genes → create "CellType (GeneA+ GeneB+)" label
print("\nIdentifying top marker genes per cell type in sensitive cells...")
MIN_CELLS = 10  # minimum sensitive cells per cell type to include

celltype_top_genes = {}   # {cell_type: [gene1, gene2]}
celltype_mean_expr = {}   # {cell_type: pd.Series of mean expr per gene}

for ct in df_comp.index:
    ct_mask  = sensitive_obs['cell_type_label'] == ct
    n_cells  = ct_mask.sum()
    if n_cells < MIN_CELLS:
        continue
    ct_adata = sensitive_adata[ct_mask.values]
    avail    = [g for g in TOP_MARKER_GENES if g in ct_adata.var_names]
    if not avail:
        continue
    gi   = [list(ct_adata.var_names).index(g) for g in avail]
    X_ct = ct_adata.X[:, gi]
    if scipy.sparse.issparse(X_ct):
        mean_expr = np.array(X_ct.mean(axis=0)).flatten()
    else:
        mean_expr = X_ct.mean(axis=0)
    mean_series = pd.Series(mean_expr, index=avail).sort_values(ascending=False)
    celltype_mean_expr[ct] = mean_series
    top2 = mean_series.head(2).index.tolist()
    celltype_top_genes[ct] = top2
    print(f"  {ct:35s} (n={n_cells:5d})  top genes: {', '.join(top2)}")

# Build detailed subtype label
def make_detailed_label(ct):
    genes = celltype_top_genes.get(ct)
    if not genes:
        return ct
    return f"{ct} ({'+'.join(genes)}+)"

sensitive_adata.obs['detailed_subtype'] = (
    sensitive_adata.obs['cell_type_label'].apply(make_detailed_label)
)
# propagate back to full adata for UMAP
adata.obs['detailed_subtype'] = 'Non-sensitive'
adata.obs.loc[sensitive_adata.obs_names, 'detailed_subtype'] = sensitive_adata.obs['detailed_subtype'].values

# Save detailed subtype table
df_detail = sensitive_obs[['cell_type_label', 'donor_id', 'disease', 'target_score']].copy()
df_detail['detailed_subtype'] = sensitive_adata.obs['detailed_subtype'].values
df_detail.to_csv(Config.RESULTS_DIR / 'Figure4F_sensitive_detailed_subtypes.csv')
df_comp.to_csv(Config.RESULTS_DIR / 'Figure4F_sensitive_cell_subtype_composition.csv')

# ── C. Heatmap: cell type × top marker genes (sensitive cells only) ───────────
# Build matrix: rows=cell types, cols=top marker genes, values=mean log1p expr
ct_order    = [ct for ct in df_comp.index if ct in celltype_mean_expr]
gene_union  = TOP_MARKER_GENES[:20]  # cap at 20 genes for readability
heatmap_mat = pd.DataFrame(index=ct_order, columns=gene_union, dtype=float)
for ct in ct_order:
    for gene in gene_union:
        heatmap_mat.loc[ct, gene] = celltype_mean_expr[ct].get(gene, 0.0)
heatmap_mat = heatmap_mat.fillna(0).astype(float)

fig, ax = plt.subplots(figsize=(max(12, len(gene_union) * 0.7),
                                max(6,  len(ct_order)   * 0.5)))
im = ax.imshow(heatmap_mat.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(gene_union)))
ax.set_xticklabels(gene_union, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(ct_order)))
ax.set_yticklabels(ct_order, fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.6, label='Mean log1p expression\n(sensitive cells)')
ax.set_title('Top Marker Gene Expression by Cell Type\n(Sensitive Cells Only)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
_fix_fonts()
plt.savefig(Config.FIG_DIR / 'Figure4F_marker_heatmap_per_celltype.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Config.FIG_DIR / 'Figure4F_marker_heatmap_per_celltype.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 4F heatmap saved")

# ── D. Dotplot: % expressing × mean expression (scanpy, sensitive cells) ──────
if len(ct_order) > 0 and len(gene_union) > 0:
    sensitive_adata.obs['cell_type_label'] = pd.Categorical(
        sensitive_adata.obs['cell_type_label'], categories=ct_order
    )
    sc.pl.dotplot(sensitive_adata, gene_union,
                  groupby='cell_type_label',
                  standard_scale='var',
                  title='Marker Gene Expression in Sensitive Cell Subtypes\n(dot size = % expressing, color = scaled mean)',
                  show=False,
                  figsize=(max(14, len(gene_union) * 0.7), max(5, len(ct_order) * 0.45)))
    _fix_fonts()  # Fix fonts after scanpy plot
    plt.savefig(Config.FIG_DIR / 'Figure4F_dotplot_sensitive_subtypes.pdf', bbox_inches='tight')
    plt.savefig(Config.FIG_DIR / 'Figure4F_dotplot_sensitive_subtypes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Figure 4F dotplot saved")

# ── E. Summary 3-panel: composition pie / proportion bar / enrichment bar ─────
palette_ct = sns.color_palette('tab20', n_colors=len(df_comp))
fig, axes  = plt.subplots(1, 3, figsize=(22, 7))

# Pie
ax = axes[0]
wedges, _, autotexts = ax.pie(
    df_comp['Sensitive_pct'],
    labels=None,
    autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
    colors=palette_ct, startangle=90, pctdistance=0.82,
    wedgeprops=dict(linewidth=0.5, edgecolor='white'))
for at in autotexts:
    at.set_fontsize(7)
ax.legend(wedges,
          [make_detailed_label(ct) for ct in df_comp.index],
          loc='center left', bbox_to_anchor=(1.0, 0.5),
          fontsize=6.5, frameon=False, title='Subtype (top genes)', title_fontsize=8)
ax.set_title('Sensitive Cell Subtype\nComposition', fontsize=12, fontweight='bold')

# Proportion bar
ax = axes[1]
x, w = np.arange(len(df_comp)), 0.35
ax.bar(x - w/2, df_comp['Sensitive_pct'],  w, label='Sensitive cells', color='#E64B35', alpha=0.85)
ax.bar(x + w/2, df_comp['Background_pct'], w, label='All cells (BG)',  color='#4DBBD5', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(df_comp.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Proportion (%)', fontsize=10)
ax.set_title('Sensitive vs Background\nCell Type Proportion', fontsize=12, fontweight='bold')
ax.legend(frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# Enrichment bar
ax = axes[2]
log2_enrich   = np.log2(df_comp['Enrichment_ratio'])
colors_enrich = ['#E64B35' if v > 0 else '#4DBBD5' for v in log2_enrich]
labels_enrich = [make_detailed_label(ct) for ct in df_comp.index[::-1]]
ax.barh(labels_enrich, log2_enrich[::-1].values,
        color=colors_enrich[::-1], edgecolor='white', linewidth=0.4)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('log2(Enrichment Ratio)', fontsize=10)
ax.set_title('Cell Type Enrichment in Sensitive Cells\n(annotated with top marker genes)',
             fontsize=12, fontweight='bold')
ax.tick_params(axis='y', labelsize=8)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Figure 4F: Sensitive Cell Subtype Composition & Marker Gene Annotation',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
_fix_fonts()
plt.savefig(Config.FIG_DIR / 'Figure4F_sensitive_cell_subtypes.pdf', dpi=300, bbox_inches='tight')
plt.savefig(Config.FIG_DIR / 'Figure4F_sensitive_cell_subtypes.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 4F composition summary saved")

# ── F. UMAP: detailed subtype labels on sensitive cells ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: full UMAP, highlight sensitive
adata.obs['Sensitive'] = adata.obs['is_high_score'].map({True: 'Sensitive', False: 'Other'})
sc.pl.umap(adata, color='Sensitive',
           palette={'Sensitive': '#E64B35', 'Other': '#DDDDDD'},
           size=3, alpha=0.5, title='Sensitive Cells on UMAP',
           ax=axes[0], show=False)
_fix_fonts()  # Fix fonts after scanpy plot

# Right: sensitive cells only, colored by detailed subtype
sc.pl.umap(sensitive_adata, color='detailed_subtype',
           title='Sensitive Cell Subtypes\n(CellType + top marker genes)',
           size=8, ax=axes[1], show=False)
_fix_fonts()  # Fix fonts after scanpy plot

plt.tight_layout()
_fix_fonts()  # Final font fix before saving
plt.savefig(Config.FIG_DIR / 'Figure4F_sensitive_UMAP_overlay.pdf', bbox_inches='tight')
plt.savefig(Config.FIG_DIR / 'Figure4F_sensitive_UMAP_overlay.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 4F UMAP overlay saved")

print("\n" + "="*80)
print("All figures completed.")
print("="*80)



report = {
    'total_cells': int(adata.n_obs),
    'total_genes': int(adata.n_vars),
    'high_score_cells': int((adata.obs['is_high_score'] == True).sum()),
    'high_score_percent': float((adata.obs['is_high_score'] == True).mean() * 100),
    'marker_genes': len(significant_markers) if len(significant_markers) > 0 else 0,
    'top_markers': significant_markers.head(10)['gene'].tolist() if len(significant_markers) > 0 else []
}

with open(Config.RESULTS_DIR / 'analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("✅ Analysis Complete! All Results Saved")
print("="*80)
print(f"📁 Figure Directory: {Config.FIG_DIR}")
print(f"📁 Results Directory: {Config.RESULTS_DIR}")
print("="*80)
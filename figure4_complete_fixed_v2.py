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
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

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
if adata.raw is not None:
    adata.raw = None
    print("⚠️  Cleared adata.raw to avoid gene name mismatch")

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

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')
n_mt_genes = adata.var['mt'].sum()
print(f"Number of mitochondrial genes: {n_mt_genes}")

# Calculate mitochondrial gene percentage
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

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

# Plot Figure 4A
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

# UMAP score plot
sc.pl.umap(adata, color='target_score', ax=axes[0],
           cmap='viridis', title='Target Activity Score',
           show=False)

# Boxplot - major cell types (use actual abbreviations, display full names on x-axis)
main_types = ['T4', 'T8', 'B', 'cM', 'NK', 'pDC']
available_types = [t for t in main_types if t in adata.obs['author_cell_type'].unique()]

data_for_violin = []
for t in available_types:
    mask = adata.obs['author_cell_type'] == t
    data_for_violin.append(adata.obs.loc[mask, 'target_score'].values)

bp = axes[1].boxplot(data_for_violin, patch_artist=True, showfliers=False)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_types)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes[1].set_xticklabels([celltype_names.get(t, t) for t in available_types], rotation=45, ha='right')
axes[1].set_ylabel('Target Activity Score')
axes[1].set_title('Score Distribution by Major Cell Types')

plt.tight_layout()
plt.savefig(Config.FIG_DIR / 'Figure4A_target_score.pdf', dpi=300, bbox_inches='tight')
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

# Plot Figure 4B
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Highlight high-score cells
adata.obs['highlight'] = adata.obs['is_high_score'].map({True: 'High Score', False: 'Other'})
colors = {'High Score': 'red', 'Other': 'lightgray'}
sc.pl.umap(adata, color='highlight', ax=axes[0],
           palette=colors, title='High-score Cells (Top 20%)',
           show=False)

# Right plot: High-score cells colored by author_cell_type (showing subpopulation origin)
high_score_adata.obs['Cell Type'] = high_score_adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(high_score_adata.obs['author_cell_type'].astype(str))
sc.pl.umap(high_score_adata, color='Cell Type', ax=axes[1],
           title='High-score Cells by Cell Type', show=False)

plt.tight_layout()
plt.savefig(Config.FIG_DIR / 'Figure4B_high_score_subset.pdf', dpi=300, bbox_inches='tight')
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

# Pre-filter: only test genes with logFC > 0.15 (reduce computation)
candidate_mask = logFC_cell > 0.15
candidate_genes = adata.var_names[candidate_mask].tolist()
print(f"Candidate genes (cell-level logFC > 0.15): {len(candidate_genes)}")

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
        stat, pval = ttest_rel(high_expr, low_expr, alternative='greater')
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
    (markers['logFC'] > 0.15) &
    (markers['p_val_adj'] < 0.05)
].copy()

print(f"\n✅ Significant marker genes (donor-level, logFC>0.15, p_adj<0.05): {len(significant_markers)}")
print(significant_markers.head(20)[['gene', 'logFC', 'p_val_adj']].to_string())

# Save results
significant_markers.to_csv(Config.RESULTS_DIR / 'marker_genes.csv', index=False)

# Plot dotplot
if len(significant_markers) >= 10:
    top_genes = significant_markers.head(10)['gene'].tolist()

    # Add full name column, dotplot y-axis shows complete cell type names
    adata.obs['Cell Type'] = adata.obs['author_cell_type'].astype(str).map(celltype_names).fillna(adata.obs['author_cell_type'].astype(str))

    fig, ax = plt.subplots(figsize=(12, 6))
    sc.pl.dotplot(adata, top_genes, groupby='Cell Type',
                 standard_scale='var', ax=ax, show=False,
                 use_raw=False,
                 title='Marker Gene Expression in High-score Cells')

    plt.tight_layout()
    plt.savefig(Config.FIG_DIR / 'Figure4D_marker_genes.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Figure 4D saved")

# ==================== 8. Figure 4E: Functional Enrichment Analysis ====================
print("\n" + "="*80)
print("8. Figure 4E: Functional Enrichment Analysis")
print("="*80)

if len(significant_markers) > 0:
    # Use more genes for enrichment (relax threshold)
    relaxed_markers = markers[
        (markers['logFC'] > 0.2) &
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
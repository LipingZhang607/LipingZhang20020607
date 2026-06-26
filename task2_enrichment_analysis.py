#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务#2: Set D和Set M的功能富集分析对比
使用Enrichr进行GO/KEGG富集分析，对比两个基因集的功能差异
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gseapy as gp
from gseapy.plot import barplot, dotplot
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 路径配置
BASE_DIR = "/home/h3033/statics/GEO_data/GSE"
SET_D_FILE = f"{BASE_DIR}/figure4/imidazoline_SLE_intersection.csv"
SET_M_FILE = f"{BASE_DIR}/figure4/results/Figure4D_sensitive_cell_markers_top50.csv"
OUTPUT_DIR = f"{BASE_DIR}/circular_reasoning_analysis"

print("="*80)
print("任务#2: 功能富集分析对比 (Set D vs Set M)")
print("="*80)

# ==================== 1. 加载基因集 ====================
print("\n[1/5] 加载基因集...")
set_d_df = pd.read_csv(SET_D_FILE)
genes_d = set_d_df.iloc[:, 0].tolist()
print(f"  Set D: {len(genes_d)} genes")

set_m_df = pd.read_csv(SET_M_FILE)
genes_m = set_m_df['gene'].tolist()
print(f"  Set M: {len(genes_m)} genes")

# ==================== 2. Enrichr富集分析 ====================
print("\n[2/5] 进行Enrichr富集分析...")

# 定义要分析的数据库
gene_sets = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023',
    'KEGG_2021_Human',
    'Reactome_2022'
]

enrichment_results = {}

# Set D富集分析
print("\n  分析Set D...")
try:
    enr_d = gp.enrichr(
        gene_list=genes_d,
        gene_sets=gene_sets,
        organism='human',
        outdir=None,
        cutoff=0.05
    )
    enrichment_results['Set_D'] = enr_d.results
    print(f"    ✓ Set D富集完成，发现 {len(enr_d.results)} 个显著通路")
except Exception as e:
    print(f"    ✗ Set D富集失败: {e}")
    enrichment_results['Set_D'] = pd.DataFrame()

# Set M富集分析
print("\n  分析Set M...")
try:
    enr_m = gp.enrichr(
        gene_list=genes_m,
        gene_sets=gene_sets,
        organism='human',
        outdir=None,
        cutoff=0.05
    )
    enrichment_results['Set_M'] = enr_m.results
    print(f"    ✓ Set M富集完成，发现 {len(enr_m.results)} 个显著通路")
except Exception as e:
    print(f"    ✗ Set M富集失败: {e}")
    enrichment_results['Set_M'] = pd.DataFrame()

# 保存完整结果
for name, df in enrichment_results.items():
    if not df.empty:
        df.to_csv(f"{OUTPUT_DIR}/enrichment_{name}_full.csv", index=False)
        print(f"  ✓ {name}完整结果已保存")

# ==================== 3. 提取Top通路 ====================
print("\n[3/5] 提取Top通路...")

def extract_top_pathways(df, n_top=15):
    """提取每个数据库的Top通路"""
    if df.empty:
        return pd.DataFrame()

    # 按数据库分组，取每组的Top通路
    top_pathways = []
    for db in df['Gene_set'].unique():
        db_df = df[df['Gene_set'] == db].copy()
        db_df = db_df.sort_values('Adjusted P-value').head(n_top)
        top_pathways.append(db_df)

    result = pd.concat(top_pathways, ignore_index=True)
    result = result.sort_values('Adjusted P-value').head(n_top)
    return result

top_d = extract_top_pathways(enrichment_results['Set_D'], n_top=20)
top_m = extract_top_pathways(enrichment_results['Set_M'], n_top=20)

print(f"  Set D Top通路: {len(top_d)}")
print(f"  Set M Top通路: {len(top_m)}")

# 保存Top通路
top_d.to_csv(f"{OUTPUT_DIR}/enrichment_Set_D_top20.csv", index=False)
top_m.to_csv(f"{OUTPUT_DIR}/enrichment_Set_M_top20.csv", index=False)

# ==================== 4. 识别独有通路 ====================
print("\n[4/5] 识别独有和共同通路...")

if not top_d.empty and not top_m.empty:
    terms_d = set(top_d['Term'].tolist())
    terms_m = set(top_m['Term'].tolist())

    common_terms = terms_d & terms_m
    unique_d = terms_d - terms_m
    unique_m = terms_m - terms_d

    print(f"  共同通路: {len(common_terms)}")
    print(f"  Set D独有: {len(unique_d)}")
    print(f"  Set M独有: {len(unique_m)}")

    # 保存独有通路
    unique_d_df = top_d[top_d['Term'].isin(unique_d)].copy()
    unique_m_df = top_m[top_m['Term'].isin(unique_m)].copy()

    unique_d_df.to_csv(f"{OUTPUT_DIR}/enrichment_Set_D_unique.csv", index=False)
    unique_m_df.to_csv(f"{OUTPUT_DIR}/enrichment_Set_M_unique.csv", index=False)

    print(f"\n  Set M独有通路示例:")
    for i, row in unique_m_df.head(10).iterrows():
        print(f"    - {row['Term'][:60]}... (p={row['Adjusted P-value']:.2e})")

# ==================== 5. 创建对比可视化 ====================
print("\n[5/5] 创建对比可视化...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# ── 左上: Set D Top10通路 ──
ax1 = fig.add_subplot(gs[0, 0])
if not top_d.empty:
    plot_d = top_d.head(10).copy()
    plot_d['neg_log10_p'] = -np.log10(plot_d['Adjusted P-value'])
    plot_d['Term_short'] = plot_d['Term'].str[:50]

    bars = ax1.barh(range(len(plot_d)), plot_d['neg_log10_p'],
                    color='#E64B35', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(plot_d)))
    ax1.set_yticklabels(plot_d['Term_short'], fontsize=9)
    ax1.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
    ax1.set_title('Set D: Top 10 Enriched Pathways\n(Target Gene Set)',
                  fontsize=12, fontweight='bold')
    ax1.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

# ── 右上: Set M Top10通路 ──
ax2 = fig.add_subplot(gs[0, 1])
if not top_m.empty:
    plot_m = top_m.head(10).copy()
    plot_m['neg_log10_p'] = -np.log10(plot_m['Adjusted P-value'])
    plot_m['Term_short'] = plot_m['Term'].str[:50]

    bars = ax2.barh(range(len(plot_m)), plot_m['neg_log10_p'],
                    color='#4DBBD5', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(plot_m)))
    ax2.set_yticklabels(plot_m['Term_short'], fontsize=9)
    ax2.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
    ax2.set_title('Set M: Top 10 Enriched Pathways\n(Marker Gene Set)',
                  fontsize=12, fontweight='bold')
    ax2.axvline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

# ── 中间: 气泡图对比 ──
ax3 = fig.add_subplot(gs[1, :])
if not top_d.empty and not top_m.empty:
    # 合并两个数据集的Top5通路
    compare_d = top_d.head(5).copy()
    compare_m = top_m.head(5).copy()

    compare_d['Set'] = 'Set D'
    compare_m['Set'] = 'Set M'
    compare_df = pd.concat([compare_d, compare_m], ignore_index=True)

    compare_df['neg_log10_p'] = -np.log10(compare_df['Adjusted P-value'])
    compare_df['Term_short'] = compare_df['Term'].str[:60]

    # 创建气泡图
    for i, (idx, row) in enumerate(compare_df.iterrows()):
        x = i
        y = 0 if row['Set'] == 'Set D' else 1
        size = row['neg_log10_p'] * 50
        color = '#E64B35' if row['Set'] == 'Set D' else '#4DBBD5'
        ax3.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black', linewidth=1)

        # 添加标签
        ax3.text(x, y-0.15 if y==0 else y+0.15, row['Term_short'],
                rotation=45, ha='right', va='top' if y==0 else 'bottom',
                fontsize=8)

    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Set D', 'Set M'], fontsize=12, fontweight='bold')
    ax3.set_xlim(-1, len(compare_df))
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_title('Top 5 Pathways Comparison (Bubble Size = -log10 P-value)',
                  fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xticks([])

# ── 左下: 通路类别分布 ──
ax4 = fig.add_subplot(gs[2, 0])
if not top_d.empty and not top_m.empty:
    # 统计每个数据库的通路数量
    db_counts_d = top_d['Gene_set'].value_counts()
    db_counts_m = top_m['Gene_set'].value_counts()

    all_dbs = sorted(set(db_counts_d.index) | set(db_counts_m.index))
    counts_d = [db_counts_d.get(db, 0) for db in all_dbs]
    counts_m = [db_counts_m.get(db, 0) for db in all_dbs]

    x = np.arange(len(all_dbs))
    width = 0.35

    ax4.bar(x - width/2, counts_d, width, label='Set D', color='#E64B35', alpha=0.7)
    ax4.bar(x + width/2, counts_m, width, label='Set M', color='#4DBBD5', alpha=0.7)

    ax4.set_xticks(x)
    ax4.set_xticklabels([db.replace('_', '\n') for db in all_dbs],
                        fontsize=9, rotation=0)
    ax4.set_ylabel('Number of Pathways', fontsize=11)
    ax4.set_title('Pathway Database Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10, frameon=False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

# ── 右下: 统计摘要 ──
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

summary_text = f"""
Functional Enrichment Comparison Summary
{'='*50}

Set D (Target Genes, n={len(genes_d)}):
  Total significant pathways: {len(enrichment_results['Set_D'])}
  Top 20 pathways: {len(top_d)}
  Unique pathways (vs Set M): {len(unique_d) if not top_d.empty and not top_m.empty else 'N/A'}

Set M (Marker Genes, n={len(genes_m)}):
  Total significant pathways: {len(enrichment_results['Set_M'])}
  Top 20 pathways: {len(top_m)}
  Unique pathways (vs Set D): {len(unique_m) if not top_d.empty and not top_m.empty else 'N/A'}

Common pathways: {len(common_terms) if not top_d.empty and not top_m.empty else 'N/A'}

Key Findings:
"""

if not top_d.empty and not top_m.empty:
    if len(unique_m) > len(unique_d):
        summary_text += f"""
  ✓ Set M enriches {len(unique_m)} unique pathways
  ✓ Provides new functional insights beyond Set D
  ✓ Supports independent biological discovery
"""
    else:
        summary_text += f"""
  ⚠ Set M has fewer unique pathways than Set D
  ⚠ May indicate functional overlap
  ⚠ Requires further investigation
"""
else:
    summary_text += "\n  ⚠ Enrichment analysis incomplete\n"

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Functional Enrichment Analysis: Set D vs Set M',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(f"{OUTPUT_DIR}/Figure_S_enrichment_comparison.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/Figure_S_enrichment_comparison.png", dpi=150, bbox_inches='tight')
print(f"  ✓ 对比图已保存: Figure_S_enrichment_comparison.pdf/png")
plt.close()

print("\n" + "="*80)
print("任务#2 完成！")
print("="*80)
print(f"\n输出文件位置: {OUTPUT_DIR}/")
print("  - enrichment_Set_D_full.csv (完整结果)")
print("  - enrichment_Set_M_full.csv (完整结果)")
print("  - enrichment_Set_D_top20.csv (Top 20通路)")
print("  - enrichment_Set_M_top20.csv (Top 20通路)")
print("  - enrichment_Set_D_unique.csv (独有通路)")
print("  - enrichment_Set_M_unique.csv (独有通路)")
print("  - Figure_S_enrichment_comparison.pdf/png")

if not top_d.empty and not top_m.empty:
    if len(unique_m) > 5:
        print(f"\n✅ 结论: Set M富集出{len(unique_m)}个独有通路，提供了新的功能视角")
    else:
        print(f"\n⚠️  警告: Set M独有通路较少({len(unique_m)}个)，功能差异不明显")

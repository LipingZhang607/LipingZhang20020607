#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务#3: 置换检验验证Set D的特异性
通过1000次随机置换，证明Set D定义的"靶点富集细胞"能产生显著更多的差异表达基因
"""

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 路径配置
BASE_DIR = "/home/h3033/statics/GEO_data/GSE"
SET_D_FILE = f"{BASE_DIR}/figure4/imidazoline_SLE_intersection.csv"
LABELS_FILE = f"{BASE_DIR}/figure4/results/Figure4B_target_enriched_cell_labels.csv"
OUTPUT_DIR = f"{BASE_DIR}/circular_reasoning_analysis"

# 参数配置
N_PERMUTATIONS = 100   # 置换次数（先用100次测试，确认可行后再增加到1000）
TOP_PERCENTILE = 80    # 前20%定义为"富集细胞"
LOGFC_THRESHOLD = 0.25 # logFC阈值
PVAL_THRESHOLD = 0.05  # p值阈值

print("="*80)
print("任务#3: 置换检验验证Set D特异性")
print("="*80)
print(f"\n参数配置:")
print(f"  置换次数: {N_PERMUTATIONS}")
print(f"  富集细胞定义: Top {100-TOP_PERCENTILE}%")
print(f"  差异基因标准: |logFC| >= {LOGFC_THRESHOLD}, p_adj < {PVAL_THRESHOLD}")

# ==================== 1. 加载数据 ====================
print("\n[1/6] 加载数据...")

# 加载Set D
set_d_df = pd.read_csv(SET_D_FILE)
genes_d = set_d_df.iloc[:, 0].tolist()
print(f"  Set D: {len(genes_d)} genes")

# 加载细胞标签（包含真实的靶点富集细胞标签）
labels_df = pd.read_csv(LABELS_FILE)
print(f"  细胞标签: {len(labels_df)} cells")
print(f"  真实靶点富集细胞: {labels_df['sensitive_cell'].sum()} cells")

# 加载单细胞数据（需要表达矩阵进行差异分析）
print("\n  加载单细胞数据（这可能需要几分钟）...")
adata_file = f"{BASE_DIR}/figure4/data/processed/SLE_PBMC_annotated.h5ad"

try:
    adata = sc.read_h5ad(adata_file)
    print(f"  ✓ 加载成功: {adata.shape[0]} cells × {adata.shape[1]} genes")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    print("\n  错误: 无法加载单细胞数据，置换检验无法进行")
    exit(1)

# 确保细胞顺序一致
if 'cell_barcode' in labels_df.columns:
    common_cells = labels_df['cell_barcode'].tolist()
    # 只保留在labels中的细胞
    adata = adata[[c for c in common_cells if c in adata.obs_names], :]
    print(f"  ✓ 细胞顺序已对齐: {adata.shape[0]} cells")

# ==================== 2. 定义差异分析函数 ====================
print("\n[2/6] 定义差异分析函数...")

def perform_differential_analysis(adata, high_score_mask, min_cells=10):
    """
    对"富集细胞" vs "非富集细胞"进行差异表达分析

    参数:
        adata: AnnData对象
        high_score_mask: 布尔数组，标记哪些细胞是"富集细胞"
        min_cells: 基因至少在多少个细胞中表达才纳入分析

    返回:
        n_deg: 差异表达基因数量
    """
    # 创建临时分组标签
    adata.obs['temp_group'] = 'low'
    adata.obs.loc[high_score_mask, 'temp_group'] = 'high'

    # 过滤低表达基因
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 使用Wilcoxon秩和检验（快速且稳健）
    sc.tl.rank_genes_groups(adata, 'temp_group', method='wilcoxon',
                            groups=['high'], reference='low')

    # 提取结果
    result = sc.get.rank_genes_groups_df(adata, group='high')

    # 统计显著差异基因
    n_deg = len(result[
        (result['logfoldchanges'].abs() >= LOGFC_THRESHOLD) &
        (result['pvals_adj'] < PVAL_THRESHOLD)
    ])

    return n_deg

print("  ✓ 差异分析函数已定义")

# ==================== 3. 计算真实Set D的差异基因数 ====================
print("\n[3/6] 计算真实Set D的差异基因数...")

# 获取Set D在数据中存在的基因
existing_genes_d = [g for g in genes_d if g in adata.var_names]
print(f"  Set D中存在于数据的基因: {len(existing_genes_d)}/{len(genes_d)}")

# 计算真实的靶点活性评分
adata.obs['target_score_real'] = adata[:, existing_genes_d].X.mean(axis=1).A1

# 定义真实的"靶点富集细胞"
threshold_real = np.percentile(adata.obs['target_score_real'], TOP_PERCENTILE)
high_score_mask_real = adata.obs['target_score_real'] > threshold_real
print(f"  真实靶点富集细胞: {high_score_mask_real.sum()} cells")

# 进行差异分析
print("  进行差异表达分析...")
adata_copy = adata.copy()
n_deg_real = perform_differential_analysis(adata_copy, high_score_mask_real)
print(f"  ✓ 真实Set D产生的差异基因数: {n_deg_real}")

# ==================== 4. 进行置换检验 ====================
print(f"\n[4/6] 进行{N_PERMUTATIONS}次置换检验...")
print("  （这可能需要10-30分钟，请耐心等待）")

# 获取所有可用基因
all_genes = adata.var_names.tolist()
n_genes_to_sample = len(existing_genes_d)  # 与Set D大小相同

# 存储每次置换的结果
permutation_results = []

for i in tqdm(range(N_PERMUTATIONS), desc="  置换进度"):
    # 随机抽取基因
    random_genes = np.random.choice(all_genes, size=n_genes_to_sample, replace=False)

    # 计算随机基因集的活性评分
    adata.obs[f'target_score_perm'] = adata[:, random_genes].X.mean(axis=1).A1

    # 定义"假想靶点富集细胞"
    threshold_perm = np.percentile(adata.obs[f'target_score_perm'], TOP_PERCENTILE)
    high_score_mask_perm = adata.obs[f'target_score_perm'] > threshold_perm

    # 进行差异分析
    adata_copy = adata.copy()
    n_deg_perm = perform_differential_analysis(adata_copy, high_score_mask_perm)

    permutation_results.append(n_deg_perm)

permutation_results = np.array(permutation_results)
print(f"\n  ✓ 置换检验完成")
print(f"  随机对照的差异基因数: {permutation_results.mean():.1f} ± {permutation_results.std():.1f}")

# ==================== 5. 统计检验 ====================
print("\n[5/6] 统计检验...")

# 计算经验p值
empirical_pval = (permutation_results >= n_deg_real).sum() / N_PERMUTATIONS
print(f"  经验p值: {empirical_pval:.4f}")

# 计算Z分数
z_score = (n_deg_real - permutation_results.mean()) / permutation_results.std()
print(f"  Z分数: {z_score:.2f}")

# 保存结果
results_df = pd.DataFrame({
    'metric': ['n_deg_real', 'n_deg_perm_mean', 'n_deg_perm_std',
               'empirical_pval', 'z_score', 'n_permutations'],
    'value': [n_deg_real, permutation_results.mean(), permutation_results.std(),
              empirical_pval, z_score, N_PERMUTATIONS]
})
results_df.to_csv(f"{OUTPUT_DIR}/permutation_test_results.csv", index=False)
print(f"  ✓ 结果已保存: permutation_test_results.csv")

# 保存完整的置换分布
perm_dist_df = pd.DataFrame({
    'permutation_id': range(N_PERMUTATIONS),
    'n_deg': permutation_results
})
perm_dist_df.to_csv(f"{OUTPUT_DIR}/permutation_distribution.csv", index=False)
print(f"  ✓ 置换分布已保存: permutation_distribution.csv")

# ==================== 6. 创建可视化 ====================
print("\n[6/6] 创建可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 左上: 直方图 ──
ax1 = axes[0, 0]
ax1.hist(permutation_results, bins=50, color='gray', alpha=0.6,
         edgecolor='black', linewidth=0.5, label='Random gene sets')
ax1.axvline(n_deg_real, color='red', linestyle='--', linewidth=2,
            label=f'Set D (n={n_deg_real})')
ax1.axvline(permutation_results.mean(), color='blue', linestyle=':', linewidth=2,
            label=f'Mean random (n={permutation_results.mean():.0f})')
ax1.set_xlabel('Number of DEGs', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Permutation Test: DEG Distribution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── 右上: 累积分布函数 ──
ax2 = axes[0, 1]
sorted_perm = np.sort(permutation_results)
cumulative = np.arange(1, len(sorted_perm) + 1) / len(sorted_perm)
ax2.plot(sorted_perm, cumulative, color='gray', linewidth=2, label='Random gene sets')
ax2.axvline(n_deg_real, color='red', linestyle='--', linewidth=2, label='Set D')
ax2.axhline(1 - empirical_pval, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax2.set_xlabel('Number of DEGs', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.3, linestyle='--')

# ── 左下: 箱线图 + 小提琴图 ──
ax3 = axes[1, 0]
parts = ax3.violinplot([permutation_results], positions=[0], widths=0.7,
                       showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightgray')
    pc.set_alpha(0.6)
ax3.scatter([0], [n_deg_real], color='red', s=200, zorder=10,
           marker='*', edgecolors='black', linewidth=1, label='Set D')
ax3.set_xticks([0])
ax3.set_xticklabels(['Random\nGene Sets'], fontsize=11)
ax3.set_ylabel('Number of DEGs', fontsize=11)
ax3.set_title('Set D vs Random Gene Sets', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, frameon=False)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── 右下: 统计摘要 ──
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
Permutation Test Summary
{'='*50}

Real Set D:
  Number of DEGs: {n_deg_real}

Random Gene Sets (n={N_PERMUTATIONS}):
  Mean DEGs: {permutation_results.mean():.1f}
  Std DEGs: {permutation_results.std():.1f}
  Min DEGs: {permutation_results.min()}
  Max DEGs: {permutation_results.max()}

Statistical Significance:
  Empirical P-value: {empirical_pval:.4f}
  Z-score: {z_score:.2f}

Interpretation:
"""

if empirical_pval < 0.001:
    summary_text += f"""  ✓✓✓ HIGHLY SIGNIFICANT (p < 0.001)
  Set D produces {n_deg_real - permutation_results.mean():.0f} more DEGs
  than random gene sets on average.
  This strongly supports Set D's specificity.
"""
elif empirical_pval < 0.05:
    summary_text += f"""  ✓✓ SIGNIFICANT (p < 0.05)
  Set D produces significantly more DEGs
  than random gene sets.
  This supports Set D's specificity.
"""
else:
    summary_text += f"""  ✗ NOT SIGNIFICANT (p >= 0.05)
  Set D does not produce significantly
  more DEGs than random gene sets.
  This raises concerns about specificity.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Permutation Test: Validating Set D Specificity',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure_S_permutation_test.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/Figure_S_permutation_test.png", dpi=150, bbox_inches='tight')
print(f"  ✓ 可视化已保存: Figure_S_permutation_test.pdf/png")
plt.close()

print("\n" + "="*80)
print("任务#3 完成！")
print("="*80)
print(f"\n输出文件位置: {OUTPUT_DIR}/")
print("  - permutation_test_results.csv (统计结果)")
print("  - permutation_distribution.csv (完整置换分布)")
print("  - Figure_S_permutation_test.pdf/png")

if empirical_pval < 0.001:
    print(f"\n✅ 结论: Set D具有高度特异性 (p={empirical_pval:.4f})")
    print(f"   真实Set D产生{n_deg_real}个差异基因，显著高于随机对照")
elif empirical_pval < 0.05:
    print(f"\n✅ 结论: Set D具有显著特异性 (p={empirical_pval:.4f})")
else:
    print(f"\n⚠️  警告: Set D特异性不显著 (p={empirical_pval:.4f})")
    print("   需要重新评估研究设计")

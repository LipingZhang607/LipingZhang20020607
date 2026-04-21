#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLE PBMC单细胞数据 - 逐步探索分析（修正版：使用feature_name列）
工作目录: ~/statics/GEO_data/GSE/figure4
数据文件: ~/statics/GEO_data/GSE/figure4/data/raw/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置路径
base_dir = Path.home() / "statics/GEO_data/GSE/figure4"
raw_data_dir = base_dir / "data/raw"
processed_data_dir = base_dir / "data/processed"
fig_dir = base_dir / "figs"

# 创建目录
processed_data_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("第1步：加载数据")
print("="*80)

# 加载数据
data_path = raw_data_dir / "4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
print(f"数据路径: {data_path}")
print(f"文件是否存在: {data_path.exists()}")

adata = sc.read_h5ad(data_path)
print(f"\n数据加载成功!")
print(f"细胞数 (观测值): {adata.n_obs}")
print(f"基因数 (变量): {adata.n_vars}")
print(f"稀疏矩阵: {scipy.sparse.issparse(adata.X)}")
print(f"矩阵形状: {adata.X.shape}")

print("\n" + "="*80)
print("第2步：查看数据对象结构 - 重点关注feature_name列")
print("="*80)

print("\n【adata.obs】- 细胞元数据")
print(f"列名: {adata.obs.columns.tolist()}")
print(f"行数: {adata.obs.shape}")
print(f"\n前几行预览:")
print(adata.obs.head(3))

print("\n【adata.var】- 基因元数据")
print(f"列名: {adata.var.columns.tolist()}")
print(f"行数: {adata.var.shape}")

# 特别关注feature_name列
if 'feature_name' in adata.var.columns:
    print("\n✅ 找到feature_name列！")
    print(f"feature_name前10个值:")
    for i, name in enumerate(adata.var['feature_name'].values[:10]):
        print(f"  {i+1:2d}. {name}")
    
    # 检查是否有重复的feature_name
    unique_features = adata.var['feature_name'].nunique()
    total_features = len(adata.var)
    if unique_features < total_features:
        print(f"\n⚠️  警告: feature_name列有重复值!")
        print(f"  总基因数: {total_features}")
        print(f"  唯一feature_name数: {unique_features}")
        print(f"  重复数: {total_features - unique_features}")
        
        # 查看重复的feature_name
        dup_mask = adata.var['feature_name'].duplicated(keep=False)
        dup_features = adata.var[dup_mask]['feature_name'].value_counts()
        print(f"\n重复的feature_name示例:")
        for feat, count in dup_features.head(5).items():
            print(f"  {feat}: {count}次")
    else:
        print(f"\n✅ feature_name列无重复，可以直接使用")
else:
    print("\n❌ 未找到feature_name列！")

print("\n【adata.obsm】- 降维结果存储")
print(f"键名: {list(adata.obsm.keys())}")

print("\n【adata.varm】- 基因相关降维结果")
print(f"键名: {list(adata.varm.keys())}")

print("\n【adata.uns】- 非结构化数据")
print(f"键名: {list(adata.uns.keys())}")

print("\n" + "="*80)
print("第3步：检查数据是否已标准化/对数化")
print("="*80)

# 检查数据矩阵
if scipy.sparse.issparse(adata.X):
    data_sample = adata.X[:1000, :1000].toarray().flatten()
else:
    data_sample = adata.X[:1000, :1000].flatten()

# 移除0值以便更好地查看分布
non_zero = data_sample[data_sample > 0]

print("表达量矩阵统计:")
print(f"  最小值: {data_sample.min():.4f}")
print(f"  最大值: {data_sample.max():.4f}")
print(f"  平均值: {data_sample.mean():.4f}")
print(f"  中位数: {np.median(data_sample):.4f}")
print(f"  非零值比例: {(data_sample > 0).sum() / len(data_sample):.2%}")

# 判断是否log转换过
if data_sample.min() < 0:
    print("\n⚠️  发现负值，数据可能已经过log转换+缩放")
elif data_sample.max() > 50:
    print("\n⚠️  最大值>50，数据可能是原始UMI计数或CPM，未log转换")
elif 5 < data_sample.max() <= 20:
    print("\n⚠️  最大值在5-20之间，数据可能已经过log1p转换")
else:
    print("\n⚠️  无法确定处理状态")

# 可视化表达量分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 直方图
axes[0].hist(non_zero[:10000], bins=50, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('表达量')
axes[0].set_ylabel('频率')
axes[0].set_title('非零表达量分布')

# 箱线图
axes[1].boxplot(non_zero[:10000])
axes[1].set_ylabel('表达量')
axes[1].set_title('表达量箱线图')

# 密度图
sns.kdeplot(non_zero[:10000], ax=axes[2])
axes[2].set_xlabel('表达量')
axes[2].set_ylabel('密度')
axes[2].set_title('表达量密度分布')

plt.tight_layout()
plt.savefig(fig_dir / 'expression_distribution_check.pdf')
plt.show()

print(f"\n表达量分布图已保存: {fig_dir / 'expression_distribution_check.pdf'}")

print("\n" + "="*80)
print("第4步：检查是否有预处理痕迹")
print("="*80)

# 检查是否已有PCA结果
if 'X_pca' in adata.obsm:
    print("✅ 已存在PCA降维结果")
    print(f"  PCA维度: {adata.obsm['X_pca'].shape}")
else:
    print("❌ 未发现PCA降维结果")

# 检查是否已有UMAP结果
if 'X_umap' in adata.obsm:
    print("✅ 已存在UMAP降维结果")
    print(f"  UMAP维度: {adata.obsm['X_umap'].shape}")
else:
    print("❌ 未发现UMAP降维结果")

# 检查是否已聚类
if 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns:
    cluster_col = 'leiden' if 'leiden' in adata.obs.columns else 'louvain'
    print(f"✅ 已存在聚类结果: {cluster_col}")
    print(f"  聚类数: {adata.obs[cluster_col].nunique()}")
else:
    print("❌ 未发现聚类结果")

# 检查是否已细胞类型注释
celltype_keywords = ['cell_type', 'celltype', 'CellType', 'annot', 'cluster_annotation']
found_celltype = [col for col in adata.obs.columns if any(kw in col.lower() for kw in celltype_keywords)]
if found_celltype:
    print(f"✅ 可能已有细胞类型注释: {found_celltype}")
    for col in found_celltype[:3]:
        print(f"  {col}: {adata.obs[col].nunique()} 个类别")
        print(f"    示例: {adata.obs[col].value_counts().head(3).to_dict()}")
else:
    print("❌ 未发现明确的细胞类型注释")

print("\n" + "="*80)
print("第5步：查看基因名称格式 - 使用feature_name列")
print("="*80)

if 'feature_name' in adata.var.columns:
    # 使用feature_name列作为基因名
    gene_names = adata.var['feature_name'].values
    print("使用feature_name列作为基因名:")
    print(f"前20个基因名:")
    for i, gene in enumerate(gene_names[:20]):
        print(f"  {i+1:2d}. {gene}")
    
    # 判断基因名格式
    ensg_count = sum(1 for g in gene_names[:100] if g.startswith('ENSG'))
    symbol_count = sum(1 for g in gene_names[:100] if g[0].isalpha() and not g.startswith('ENSG'))
    
    print(f"\n基因名格式分析 (基于前100个基因):")
    print(f"  ENSG格式: {ensg_count} 个")
    print(f"  Gene Symbol格式: {symbol_count} 个")
    print(f"  其他格式: {100 - ensg_count - symbol_count} 个")
    
    if ensg_count > symbol_count:
        print("⚠️  即使使用feature_name列，仍有较多ENSG ID")
    else:
        print("✅ feature_name列主要是Gene Symbol格式")
    
    # 检查常见的PBMC marker基因
    print("\n检查常见免疫细胞marker基因:")
    common_markers = ['CD3D', 'CD4', 'CD8A', 'CD14', 'CD19', 'NKG7', 'FCGR3A', 'MS4A1', 'CD79A', 'LYZ']
    for marker in common_markers:
        if marker in gene_names:
            # 找到该基因的位置
            idx = np.where(gene_names == marker)[0]
            print(f"  ✅ {marker}: 存在于feature_name列 (位置: {idx[0]})")
        else:
            print(f"  ❌ {marker}: 不存在于feature_name列")
else:
    # 如果没有feature_name列，使用默认的var_names
    print("使用adata.var_names作为基因名:")
    print(f"前20个基因名:")
    for i, gene in enumerate(adata.var_names[:20]):
        print(f"  {i+1:2d}. {gene}")
    
    ensg_count = sum(1 for g in adata.var_names[:100] if g.startswith('ENSG'))
    symbol_count = sum(1 for g in adata.var_names[:100] if g[0].isalpha() and not g.startswith('ENSG'))
    
    print(f"\n基因名格式分析 (基于前100个基因):")
    print(f"  ENSG格式: {ensg_count} 个")
    print(f"  Gene Symbol格式: {symbol_count} 个")
    
    if ensg_count > symbol_count:
        print("⚠️  基因名主要是ENSG ID，需要转换为gene_symbol")

print("\n" + "="*80)
print("第6步：检查是否有样本信息/疾病状态")
print("="*80)

# 查找可能的样本列
sample_keywords = ['sample', 'batch', 'donor', 'subject', 'patient', 'orig.ident']
found_sample_cols = [col for col in adata.obs.columns if any(kw in col.lower() for kw in sample_keywords)]

if found_sample_cols:
    print("✅ 发现可能的样本信息列:")
    for col in found_sample_cols:
        print(f"  {col}: {adata.obs[col].nunique()} 个唯一值")
        print(f"    前5个: {adata.obs[col].dropna().unique()[:5]}")
else:
    print("❌ 未发现明确的样本信息列")
    print("  尝试从细胞barcode中提取...")
    
    # 查看细胞barcode格式
    print(f"\n细胞barcode示例 (前5个):")
    for i, bc in enumerate(adata.obs_names[:5]):
        print(f"  {i+1}. {bc}")
    
    # 尝试提取样本前缀
    if '_' in adata.obs_names[0]:
        samples = [bc.split('_')[0] for bc in adata.obs_names[:100]]
        print(f"\n从barcode中提取的前缀: {set(samples[:10])}")
    elif '-' in adata.obs_names[0]:
        samples = [bc.split('-')[0] for bc in adata.obs_names[:100]]
        print(f"\n从barcode中提取的前缀: {set(samples[:10])}")

# 查找疾病状态
disease_keywords = ['disease', 'condition', 'status', 'group', 'SLE', 'HC', 'healthy', 'lupus']
found_disease_cols = [col for col in adata.obs.columns if any(kw in col.lower() for kw in disease_keywords)]

if found_disease_cols:
    print("\n✅ 发现可能的疾病状态列:")
    for col in found_disease_cols:
        print(f"  {col}: {adata.obs[col].value_counts().to_dict()}")
else:
    print("\n❌ 未发现明确的疾病状态列")

print("\n" + "="*80)
print("第7步：计算QC指标并查看细胞质量")
print("="*80)

# 计算QC指标 - 使用正确的线粒体基因名
if 'feature_name' in adata.var.columns:
    # 使用feature_name列查找线粒体基因
    adata.var['mt'] = adata.var['feature_name'].str.startswith('MT-') | adata.var['feature_name'].str.startswith('mt-')
else:
    # 使用默认的var_names
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')

sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

print("QC指标已添加到adata.obs:")
qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
for col in qc_cols:
    if col in adata.obs.columns:
        print(f"\n{col}:")
        print(f"  最小值: {adata.obs[col].min():.2f}")
        print(f"  最大值: {adata.obs[col].max():.2f}")
        print(f"  中位数: {adata.obs[col].median():.2f}")
        print(f"  平均值: {adata.obs[col].mean():.2f}")

# 可视化QC指标
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基因数分布
axes[0,0].hist(adata.obs['n_genes_by_counts'], bins=50, edgecolor='black', alpha=0.7)
axes[0,0].axvline(500, color='red', linestyle='--', label='500 genes')
axes[0,0].axvline(1000, color='orange', linestyle='--', label='1000 genes')
axes[0,0].axvline(1500, color='green', linestyle='--', label='1500 genes')
axes[0,0].set_xlabel('基因数')
axes[0,0].set_ylabel('细胞数')
axes[0,0].set_title('每个细胞的基因数分布')
axes[0,0].legend()

# UMI数分布
axes[0,1].hist(adata.obs['total_counts'], bins=50, edgecolor='black', alpha=0.7)
axes[0,1].axvline(1000, color='red', linestyle='--', label='1000 UMIs')
axes[0,1].axvline(5000, color='orange', linestyle='--', label='5000 UMIs')
axes[0,1].set_xlabel('UMI计数')
axes[0,1].set_ylabel('细胞数')
axes[0,1].set_title('每个细胞的UMI计数分布')
axes[0,1].legend()

# 线粒体百分比分布
axes[1,0].hist(adata.obs['pct_counts_mt'], bins=50, edgecolor='black', alpha=0.7)
axes[1,0].axvline(5, color='green', linestyle='--', label='5%')
axes[1,0].axvline(10, color='orange', linestyle='--', label='10%')
axes[1,0].axvline(15, color='red', linestyle='--', label='15%')
axes[1,0].set_xlabel('线粒体基因百分比 (%)')
axes[1,0].set_ylabel('细胞数')
axes[1,0].set_title('线粒体基因百分比分布')
axes[1,0].legend()

# 基因数 vs UMI数
scatter = axes[1,1].scatter(adata.obs['total_counts'], 
                           adata.obs['n_genes_by_counts'],
                           c=adata.obs['pct_counts_mt'], 
                           cmap='viridis', 
                           s=5, 
                           alpha=0.5)
axes[1,1].set_xlabel('UMI计数')
axes[1,1].set_ylabel('基因数')
axes[1,1].set_title('基因数 vs UMI数 (颜色=线粒体%)')
plt.colorbar(scatter, ax=axes[1,1])

plt.tight_layout()
plt.savefig(fig_dir / 'QC_metrics_distribution.pdf')
plt.show()

print(f"\nQC图已保存: {fig_dir / 'QC_metrics_distribution.pdf'}")

# 统计各QC阈值的细胞数量
print("\n不同QC阈值下的细胞数量:")
thresholds_genes = [500, 1000, 1500]
thresholds_mt = [5, 10, 15]

for g in thresholds_genes:
    for mt in thresholds_mt:
        n_cells = (adata.obs['n_genes_by_counts'] > g) & (adata.obs['pct_counts_mt'] < mt)
        print(f"  基因>{g}, 线粒体<{mt}%: {n_cells.sum()} 细胞 ({n_cells.sum()/adata.n_obs:.1%})")

print("\n" + "="*80)
print("第8步：查看是否有细胞类型注释")
print("="*80)

# 尝试查找所有可能的注释列
all_obs_cols = adata.obs.columns.tolist()
print("所有obs列名:")
for col in all_obs_cols:
    print(f"  - {col}")

# 如果已有聚类，查看聚类分布
if 'leiden' in adata.obs.columns:
    print(f"\nLeiden聚类分布:")
    leiden_counts = adata.obs['leiden'].value_counts().sort_index()
    for cluster, count in leiden_counts.items():
        print(f"  簇 {cluster}: {count} 细胞 ({count/adata.n_obs:.1%})")

if 'louvain' in adata.obs.columns:
    print(f"\nLouvain聚类分布:")
    louvain_counts = adata.obs['louvain'].value_counts().sort_index()
    for cluster, count in louvain_counts.items():
        print(f"  簇 {cluster}: {count} 细胞 ({count/adata.n_obs:.1%})")

print("\n" + "="*80)
print("第9步：查看marker基因表达情况 - 使用feature_name列")
print("="*80)

# 常见PBMC marker基因
pbmc_markers = {
    'B细胞': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
    'T细胞': ['CD3D', 'CD3E', 'CD3G', 'CD2'],
    'CD4 T细胞': ['CD4', 'IL7R'],
    'CD8 T细胞': ['CD8A', 'CD8B'],
    'NK细胞': ['NKG7', 'GNLY', 'KLRB1', 'NCR1'],
    '单核细胞': ['CD14', 'LYZ', 'FCGR3A', 'CSF1R'],
    'DC细胞': ['FCER1A', 'CLEC10A', 'CD1C'],
    'pDC': ['LILRA4', 'IL3RA', 'TCF4', 'IRF7'],
    '中性粒细胞': ['CSF3R', 'S100A8', 'S100A9'],
    '血小板': ['PPBP', 'PF4']
}

if 'feature_name' in adata.var.columns:
    gene_names = adata.var['feature_name'].values
    print("检查常见免疫细胞marker基因是否存在 (使用feature_name列):")
else:
    gene_names = adata.var_names
    print("检查常见免疫细胞marker基因是否存在 (使用adata.var_names):")

marker_available = {}
for celltype, markers in pbmc_markers.items():
    available = [m for m in markers if m in gene_names]
    marker_available[celltype] = available
    if available:
        print(f"  {celltype}: {available}")
    else:
        print(f"  {celltype}: 无")

# 如果有UMAP，可以快速可视化
if 'X_umap' in adata.obsm:
    print("\n尝试绘制UMAP（使用现有降维）...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 如果有聚类，按聚类着色
    if 'leiden' in adata.obs.columns:
        sc.pl.umap(adata, color='leiden', ax=axes[0], show=False, title='Leiden聚类')
    elif 'louvain' in adata.obs.columns:
        sc.pl.umap(adata, color='louvain', ax=axes[0], show=False, title='Louvain聚类')
    
    # 如果有细胞类型，按细胞类型着色
    if found_celltype:
        sc.pl.umap(adata, color=found_celltype[0], ax=axes[1], show=False, title=f'细胞类型: {found_celltype[0]}')
    
    # 按线粒体百分比着色
    sc.pl.umap(adata, color='pct_counts_mt', ax=axes[2], show=False, title='线粒体百分比', cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'existing_umap_overview.pdf')
    plt.show()
    print(f"UMAP图已保存: {fig_dir / 'existing_umap_overview.pdf'}")

print("\n" + "="*80)
print("第10步：数据摘要报告 - 特别关注feature_name列")
print("="*80)

print("\n=== 数据摘要 ===")
print(f"细胞总数: {adata.n_obs}")
print(f"基因总数: {adata.n_vars}")
print(f"稀疏矩阵: {'是' if scipy.sparse.issparse(adata.X) else '否'}")

print(f"\n=== feature_name列状态 ===")
if 'feature_name' in adata.var.columns:
    print("✅ feature_name列存在")
    print(f"  唯一feature_name数: {adata.var['feature_name'].nunique()}")
    print(f"  总基因数: {len(adata.var)}")
    if adata.var['feature_name'].nunique() < len(adata.var):
        print(f"  ⚠️  有重复值，需要处理")
    else:
        print(f"  ✅ 无重复，可直接使用")
else:
    print("❌ feature_name列不存在")

print(f"\n=== 预处理状态 ===")
print(f"PCA: {'有' if 'X_pca' in adata.obsm else '无'}")
print(f"UMAP: {'有' if 'X_umap' in adata.obsm else '无'}")
print(f"聚类: {'有' if 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns else '无'}")
print(f"细胞类型: {'有' if found_celltype else '无'}")

print(f"\n=== 样本信息 ===")
print(f"样本列: {'有' if found_sample_cols else '无'}")
print(f"疾病状态列: {'有' if found_disease_cols else '无'}")

print(f"\n=== 基因格式 (使用feature_name列) ===")
if 'feature_name' in adata.var.columns:
    ensg_ratio = sum(1 for g in adata.var['feature_name'].values[:100] if g.startswith('ENSG')) / 100
else:
    ensg_ratio = sum(1 for g in adata.var_names[:100] if g.startswith('ENSG')) / 100
print(f"ENSG ID比例: {ensg_ratio:.0%}")
print(f"需要转换: {'是' if ensg_ratio > 0.5 else '否'}")

print(f"\n=== 细胞质量 (中位数) ===")
print(f"基因数/细胞: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"UMI数/细胞: {adata.obs['total_counts'].median():.0f}")
print(f"线粒体%: {adata.obs['pct_counts_mt'].median():.1f}%")

print(f"\n=== 下一步建议 ===")
recommendations = []

if 'feature_name' not in adata.var.columns:
    recommendations.append("1. 需要添加feature_name列作为基因符号")

if ensg_ratio > 0.5:
    recommendations.append("2. 即使有feature_name列，仍有较多ENSG ID，可能需要从外部数据库转换")

if not found_sample_cols:
    recommendations.append("3. 需要提取样本信息: 从细胞barcode或补充metadata")

if not found_disease_cols:
    recommendations.append("4. 需要添加疾病状态: SLE/HC标注")

if 'X_pca' not in adata.obsm:
    recommendations.append("5. 需要进行PCA降维")
    
if 'leiden' not in adata.obs.columns and 'louvain' not in adata.obs.columns:
    recommendations.append("6. 需要进行聚类分析")

if not found_celltype:
    recommendations.append("7. 需要进行细胞类型注释 (CellTypist/SingleR)")

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("✅ 数据已基本处理完成，可以直接进行下游分析")

print("\n" + "="*80)
print("探索完成！下一步请根据以上建议选择相应的处理步骤。")
print("特别注意：请确认feature_name列是否正确包含gene symbols")
print("="*80)

# 保存当前状态，以便后续分析
summary = {
    'data_path': str(data_path),
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'has_feature_name': 'feature_name' in adata.var.columns,
    'feature_name_unique': adata.var['feature_name'].nunique() if 'feature_name' in adata.var.columns else 0,
    'has_pca': 'X_pca' in adata.obsm,
    'has_umap': 'X_umap' in adata.obsm,
    'has_clustering': 'leiden' in adata.obs.columns or 'louvain' in adata.obs.columns,
    'has_celltype': bool(found_celltype),
    'has_sample_info': bool(found_sample_cols),
    'has_disease_info': bool(found_disease_cols),
    'gene_format': 'ENSG' if ensg_ratio > 0.5 else 'symbol',
    'median_genes': adata.obs['n_genes_by_counts'].median(),
    'median_counts': adata.obs['total_counts'].median(),
    'median_mt': adata.obs['pct_counts_mt'].median()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(processed_data_dir / 'data_exploration_summary.csv', index=False)
print(f"\n探索摘要已保存: {processed_data_dir / 'data_exploration_summary.csv'}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5 - Step 1: 数据加载与预处理（修复索引问题）
注意：正确处理整数位置索引和标签索引的转换
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
np.random.seed(42)

# 定义路径
BASE_DIR = "/home/h3033/statics/GEO_data/GSE/figure5"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print("=" * 60)
print("Figure 5 - Step 1: 数据加载与预处理（修复索引问题）")
print("=" * 60)

# 1. 加载单细胞数据
print("\n1. 加载单细胞数据...")
h5ad_path = "/home/h3033/statics/GEO_data/GSE/figure4/data/raw/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
adata = sc.read_h5ad(h5ad_path)
print(f"   数据集形状: {adata.shape}")
print(f"   细胞数: {adata.n_obs}")
print(f"   基因数: {adata.n_vars}")

# 检查基因名存储位置
print("\n   检查基因名存储位置:")
print(f"   adata.var index 前5个: {adata.var.index[:5].tolist()}")
print(f"   adata.var columns: {adata.var.columns.tolist()}")

# 2. 加载敏感细胞标记基因 (Figure 4D top50)
print("\n2. 加载敏感细胞标记基因...")
markers_df = pd.read_csv(os.path.join(INPUT_DIR, "Figure4D_sensitive_cell_markers_top50.csv"))
top50_genes = markers_df['gene'].tolist()
print(f"   加载了 {len(top50_genes)} 个标记基因")
print(f"   前10个基因: {top50_genes[:10]}")

# 使用feature_name列匹配基因
print("\n   使用feature_name列匹配基因...")
if 'feature_name' in adata.var.columns:
    gene_names = adata.var['feature_name'].values
    available_genes = [g for g in top50_genes if g in gene_names]
    print(f"   找到 {len(available_genes)}/{len(top50_genes)} 个匹配基因")
    
    # 获取基因索引
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    gene_indices = [gene_to_idx[g] for g in available_genes]
    
    if len(available_genes) < len(top50_genes):
        missing_genes = set(top50_genes) - set(available_genes)
        print(f"   警告: 缺失 {len(missing_genes)} 个基因: {list(missing_genes)}")
else:
    raise ValueError("feature_name列不存在于adata.var中")

# 3. 加载敏感细胞标签 (Figure 4B)
print("\n3. 加载敏感细胞标签...")
labels_df = pd.read_csv(os.path.join(INPUT_DIR, "Figure4B_sensitive_cell_labels.csv"))
print(f"   标签文件形状: {labels_df.shape}")
print(f"   标签列: {labels_df.columns.tolist()}")

# 检查疾病状态分布
print("\n   疾病状态分布:")
disease_counts = labels_df['disease'].value_counts()
for disease, count in disease_counts.items():
    print(f"     {disease}: {count} 细胞 ({count/len(labels_df)*100:.1f}%)")

# 将标签映射到AnnData对象
adata.obs['cell_barcode'] = adata.obs_names

# 创建从barcode到标签的映射
label_dict = dict(zip(labels_df['cell_barcode'], labels_df['sensitive_cell']))
score_dict = dict(zip(labels_df['cell_barcode'], labels_df['target_score']))
donor_dict = dict(zip(labels_df['cell_barcode'], labels_df['donor_id']))
disease_dict = dict(zip(labels_df['cell_barcode'], labels_df['disease']))
celltype_dict = dict(zip(labels_df['cell_barcode'], labels_df['cell_type']))

# 添加到adata.obs
adata.obs['sensitive_cell'] = adata.obs['cell_barcode'].map(label_dict)
adata.obs['target_score'] = adata.obs['cell_barcode'].map(score_dict)
adata.obs['donor_id'] = adata.obs['cell_barcode'].map(donor_dict)
adata.obs['disease'] = adata.obs['cell_barcode'].map(disease_dict)
adata.obs['cell_type'] = adata.obs['cell_barcode'].map(celltype_dict)

# 删除没有标签的细胞
before_count = adata.n_obs
adata = adata[~adata.obs['sensitive_cell'].isna()].copy()
adata.obs['sensitive_cell'] = adata.obs['sensitive_cell'].astype(bool)
print(f"\n   过滤前细胞数: {before_count}")
print(f"   过滤后细胞数: {adata.n_obs}")
print(f"   敏感细胞比例: {adata.obs['sensitive_cell'].mean():.3f}")

# 过滤后的疾病状态分布
print("\n   过滤后疾病状态分布:")
disease_counts_filtered = adata.obs['disease'].value_counts()
for disease, count in disease_counts_filtered.items():
    print(f"     {disease}: {count} 细胞 ({count/adata.n_obs*100:.1f}%)")

# 4. 提取特征矩阵
print("\n4. 构建特征矩阵...")

# 提取top50基因表达矩阵
X_genes = adata.X[:, gene_indices]
if hasattr(X_genes, 'toarray'):
    X_genes = X_genes.toarray()
elif hasattr(X_genes, 'A'):
    X_genes = X_genes.A

print(f"   基因表达矩阵形状: {X_genes.shape}")

# 只用基因特征，不混入化合物评分
X = X_genes  # shape: (n_cells, len(available_genes))
y = adata.obs['sensitive_cell'].values.astype(int)

# 特征名称
feature_names = available_genes
print(f"   最终特征矩阵形状: {X.shape}")
print(f"   标签形状: {y.shape}")
print(f"   正样本数: {y.sum()}, 负样本数: {len(y)-y.sum()}")

# 5. 按患者ID和疾病状态划分数据集
print("\n5. 按患者ID和疾病状态划分训练/验证/测试集...")

# 获取患者ID和疾病状态
groups = adata.obs['donor_id'].values
disease = adata.obs['disease'].values

unique_groups = np.unique(groups)
unique_diseases = np.unique(disease)
print(f"   总患者数: {len(unique_groups)}")
print(f"   疾病类型: {list(unique_diseases)}")

# 按疾病状态统计患者数
print("\n   各疾病状态的患者数:")
for d in unique_diseases:
    patients_in_disease = np.unique(groups[disease == d])
    print(f"     {d}: {len(patients_in_disease)} 患者, {np.sum(disease == d)} 细胞")

# 首先获取唯一的患者ID及其对应的疾病状态
unique_patients = []
patient_disease = []
for patient in unique_groups:
    # 获取该患者的所有细胞
    patient_cells = disease[groups == patient]
    # 使用众数作为该患者的疾病状态（应该都一样）
    patient_disease_mode = pd.Series(patient_cells).mode()[0]
    unique_patients.append(patient)
    patient_disease.append(patient_disease_mode)

# 将患者按疾病状态分层划分为训练、验证、测试集
patients_df = pd.DataFrame({
    'patient_id': unique_patients,
    'disease': patient_disease
})

print(f"\n   患者级别的疾病分布:")
print(patients_df['disease'].value_counts())

# 首先分出测试集患者
train_val_patients, test_patients = train_test_split(
    patients_df, 
    test_size=0.15, 
    stratify=patients_df['disease'],
    random_state=42
)

# 再从训练验证集中分出验证集患者
train_patients, val_patients = train_test_split(
    train_val_patients,
    test_size=0.15/0.85,  # 15% of 85%
    stratify=train_val_patients['disease'],
    random_state=42
)

print(f"\n   训练集患者数: {len(train_patients)}")
print(f"   验证集患者数: {len(val_patients)}")
print(f"   测试集患者数: {len(test_patients)}")

# 获取对应的细胞索引（布尔索引）
train_mask = np.isin(groups, train_patients['patient_id'].values)
val_mask = np.isin(groups, val_patients['patient_id'].values)
test_mask = np.isin(groups, test_patients['patient_id'].values)

# 转换为整数索引
train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"\n   训练集细胞数: {len(train_idx)}")
print(f"   验证集细胞数: {len(val_idx)}")
print(f"   测试集细胞数: {len(test_idx)}")

# 划分数据
X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

groups_train = groups[train_idx]
groups_val = groups[val_idx]
groups_test = groups[test_idx]

disease_train = disease[train_idx]
disease_val = disease[val_idx]
disease_test = disease[test_idx]

# 检查各数据集的疾病分布
print("\n   各数据集的疾病分布:")
print(f"   训练集:")
for d in unique_diseases:
    count = np.sum(disease_train == d)
    print(f"     {d}: {count} 细胞 ({count/len(disease_train)*100:.1f}%)")

print(f"   验证集:")
for d in unique_diseases:
    count = np.sum(disease_val == d)
    print(f"     {d}: {count} 细胞 ({count/len(disease_val)*100:.1f}%)")

print(f"   测试集:")
for d in unique_diseases:
    count = np.sum(disease_test == d)
    print(f"     {d}: {count} 细胞 ({count/len(disease_test)*100:.1f}%)")

# 6. 保存处理后的数据
print("\n6. 保存处理后的数据...")

# 保存特征和标签
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# 保存分组信息
np.save(os.path.join(OUTPUT_DIR, "groups_train.npy"), groups_train)
np.save(os.path.join(OUTPUT_DIR, "groups_val.npy"), groups_val)
np.save(os.path.join(OUTPUT_DIR, "groups_test.npy"), groups_test)

# 保存疾病状态信息（重要！用于后续分析）
np.save(os.path.join(OUTPUT_DIR, "disease_train.npy"), disease_train)
np.save(os.path.join(OUTPUT_DIR, "disease_val.npy"), disease_val)
np.save(os.path.join(OUTPUT_DIR, "disease_test.npy"), disease_test)

# 保存特征名称
with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), 'wb') as f:
    pickle.dump(feature_names, f)

# 保存基因信息
gene_info = {
    'available_genes': available_genes,
    'gene_indices': gene_indices,
    'all_top50_genes': top50_genes,
    'missing_genes': [g for g in top50_genes if g not in available_genes]
}
with open(os.path.join(OUTPUT_DIR, "gene_info.pkl"), 'wb') as f:
    pickle.dump(gene_info, f)

# 保存数据划分索引
split_info = {
    'train_idx': train_idx.tolist(),  # 转换为列表以便保存
    'val_idx': val_idx.tolist(),
    'test_idx': test_idx.tolist(),
    'train_patients': train_patients['patient_id'].tolist(),
    'val_patients': val_patients['patient_id'].tolist(),
    'test_patients': test_patients['patient_id'].tolist()
}
with open(os.path.join(OUTPUT_DIR, "split_info.pkl"), 'wb') as f:
    pickle.dump(split_info, f)

# 保存细胞元数据（用于后续分析）
cell_metadata = pd.DataFrame({
    'cell_barcode': adata.obs_names[train_idx],
    'donor_id': groups_train,
    'disease': disease_train,
    'cell_type': adata.obs['cell_type'].values[train_idx],
    'sensitive_cell': y_train,
    'target_score': adata.obs['target_score'].values[train_idx],
    'split': 'train'
})
cell_metadata_val = pd.DataFrame({
    'cell_barcode': adata.obs_names[val_idx],
    'donor_id': groups_val,
    'disease': disease_val,
    'cell_type': adata.obs['cell_type'].values[val_idx],
    'sensitive_cell': y_val,
    'target_score': adata.obs['target_score'].values[val_idx],
    'split': 'val'
})
cell_metadata_test = pd.DataFrame({
    'cell_barcode': adata.obs_names[test_idx],
    'donor_id': groups_test,
    'disease': disease_test,
    'cell_type': adata.obs['cell_type'].values[test_idx],
    'sensitive_cell': y_test,
    'target_score': adata.obs['target_score'].values[test_idx],
    'split': 'test'
})

cell_metadata_all = pd.concat([cell_metadata, cell_metadata_val, cell_metadata_test], axis=0)
cell_metadata_all.to_csv(os.path.join(OUTPUT_DIR, "cell_metadata.csv"), index=False)

# 保存AnnData对象（包含所有元数据）- 使用布尔索引而不是整数索引
print("\n   保存AnnData对象...")
adata.obs['split'] = 'unknown'
adata.obs.loc[train_mask, 'split'] = 'train'  # 使用布尔掩码
adata.obs.loc[val_mask, 'split'] = 'val'
adata.obs.loc[test_mask, 'split'] = 'test'

# 验证split列是否正确赋值
print(f"   split列分布:")
print(adata.obs['split'].value_counts())

adata.write(os.path.join(OUTPUT_DIR, "adata_processed.h5ad"))

print("\n数据保存完成!")
print(f"   输出目录: {OUTPUT_DIR}")

# 打印数据集统计信息
print("\n" + "=" * 60)
print("数据集统计信息:")
print(f"   总细胞数: {len(y)}")
print(f"   总特征数: {X.shape[1]}")
print(f"   正样本比例: {y.mean():.3f}")
print(f"\n   训练集: {len(X_train)} 细胞, {len(train_patients)} 患者")
print(f"     正样本比例: {y_train.mean():.3f}")
print(f"     SLE比例: {np.sum(disease_train == 'systemic lupus erythematosus')/len(disease_train)*100:.1f}%")
print(f"     HC比例: {np.sum(disease_train == 'normal')/len(disease_train)*100:.1f}%")
print(f"\n   验证集: {len(X_val)} 细胞, {len(val_patients)} 患者")
print(f"     正样本比例: {y_val.mean():.3f}")
print(f"     SLE比例: {np.sum(disease_val == 'systemic lupus erythematosus')/len(disease_val)*100:.1f}%")
print(f"     HC比例: {np.sum(disease_val == 'normal')/len(disease_val)*100:.1f}%")
print(f"\n   测试集: {len(X_test)} 细胞, {len(test_patients)} 患者")
print(f"     正样本比例: {y_test.mean():.3f}")
print(f"     SLE比例: {np.sum(disease_test == 'systemic lupus erythematosus')/len(disease_test)*100:.1f}%")
print(f"     HC比例: {np.sum(disease_test == 'normal')/len(disease_test)*100:.1f}%")
print("=" * 60)

# 保存一个简单的报告
report = {
    'total_cells': int(len(y)),
    'total_features': int(X.shape[1]),
    'positive_ratio': float(y.mean()),
    'disease_distribution': {
        'SLE': int(np.sum(disease == 'systemic lupus erythematosus')),
        'HC': int(np.sum(disease == 'normal'))
    },
    'train': {
        'cells': int(len(X_train)),
        'patients': int(len(train_patients)),
        'positive_ratio': float(y_train.mean()),
        'SLE_ratio': float(np.sum(disease_train == 'systemic lupus erythematosus')/len(disease_train)),
        'HC_ratio': float(np.sum(disease_train == 'normal')/len(disease_train))
    },
    'val': {
        'cells': int(len(X_val)),
        'patients': int(len(val_patients)),
        'positive_ratio': float(y_val.mean()),
        'SLE_ratio': float(np.sum(disease_val == 'systemic lupus erythematosus')/len(disease_val)),
        'HC_ratio': float(np.sum(disease_val == 'normal')/len(disease_val))
    },
    'test': {
        'cells': int(len(X_test)),
        'patients': int(len(test_patients)),
        'positive_ratio': float(y_test.mean()),
        'SLE_ratio': float(np.sum(disease_test == 'systemic lupus erythematosus')/len(disease_test)),
        'HC_ratio': float(np.sum(disease_test == 'normal')/len(disease_test))
    },
    'available_genes_count': len(available_genes)
}

with open(os.path.join(OUTPUT_DIR, "data_preparation_report.json"), 'w') as f:
    import json
    json.dump(report, f, indent=2)

print("\nStep 1 完成!")
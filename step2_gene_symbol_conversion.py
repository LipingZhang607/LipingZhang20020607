#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2步：基因名转换 - 使用mygene.info API (Python版)
类似 org.Hs.eg.db 的功能
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from pathlib import Path
import warnings
import requests
import time
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 设置路径
base_dir = Path.home() / "statics/GEO_data/GSE/figure4"
raw_data_dir = base_dir / "data/raw"
processed_data_dir = base_dir / "data/processed"
fig_dir = base_dir / "figs"
ref_dir = base_dir / "data/reference"

ref_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("第2步：基因名转换 - 使用mygene.info API")
print("="*80)

# 1. 加载数据
print("\n1. 加载数据...")
data_path = processed_data_dir / "filtered_data.h5ad"

if not data_path.exists():
    print(f"⚠️  未找到过滤后的数据，加载原始数据...")
    raw_path = raw_data_dir / "4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
    adata = sc.read_h5ad(raw_path)
else:
    adata = sc.read_h5ad(data_path)

print(f"细胞数: {adata.n_obs}")
print(f"基因数: {adata.n_vars}")
print(f"基因名示例 (前10个):")
for i, gene in enumerate(adata.var_names[:10]):
    print(f"  {i+1:2d}. {gene}")

# 2. 提取所有ENSG ID
print("\n" + "="*80)
print("2. 提取ENSG ID")
print("="*80)

# 提取ENSG ID（去掉版本号）
ensg_ids = []
original_genes = list(adata.var_names)

for gene in original_genes:
    # 处理可能带版本号的ENSG ID (如 ENSG00000139618.13)
    if gene.startswith('ENSG'):
        base_id = gene.split('.')[0]
        ensg_ids.append(base_id)
    else:
        ensg_ids.append(gene)  # 保留非ENSG格式

print(f"ENSG ID数量: {sum(1 for g in ensg_ids if g.startswith('ENSG'))}")
print(f"非ENSG格式数量: {sum(1 for g in ensg_ids if not g.startswith('ENSG'))}")

# 3. 批量查询mygene.info
print("\n" + "="*80)
print("3. 使用mygene.info API查询基因符号")
print("="*80)

def query_mygene(ensg_list, batch_size=1000):
    """
    批量查询mygene.info API
    类似 org.Hs.eg.db 的功能
    """
    # 过滤出ENSG ID
    ensg_to_query = [g for g in ensg_list if g.startswith('ENSG')]
    
    print(f"需要查询的ENSG ID: {len(ensg_to_query)}")
    
    # 分批查询
    results = {}
    
    for i in range(0, len(ensg_to_query), batch_size):
        batch = ensg_to_query[i:i+batch_size]
        
        # 构建查询
        query_ids = ','.join(batch)
        url = f"http://mygene.info/v3/query?q=ensembl.gene:{query_ids}&fields=ensembl.gene,symbol,name&species=human&size={len(batch)}"
        
        try:
            print(f"  查询批次 {i//batch_size + 1}/{(len(ensg_to_query)-1)//batch_size + 1}...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # 解析结果
                for hit in data.get('hits', []):
                    if 'ensembl' in hit:
                        if isinstance(hit['ensembl'], list):
                            for ens in hit['ensembl']:
                                if ens.get('gene') in batch:
                                    results[ens['gene']] = {
                                        'symbol': hit.get('symbol', ens.get('gene')),
                                        'name': hit.get('name', '')
                                    }
                        else:
                            ens_gene = hit['ensembl'].get('gene')
                            if ens_gene in batch:
                                results[ens_gene] = {
                                    'symbol': hit.get('symbol', ens_gene),
                                    'name': hit.get('name', '')
                                }
            
            # 避免请求过快
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    查询失败: {e}")
            continue
    
    return results

# 先检查是否有缓存的映射文件
mapping_cache = ref_dir / "mygene_mapping.json"
gene_symbol_cache = ref_dir / "gene_symbol_mapping.csv"

if gene_symbol_cache.exists():
    print("找到缓存的映射文件，直接加载...")
    mapping_df = pd.read_csv(gene_symbol_cache)
    mapping_dict = dict(zip(mapping_df['ensembl_id'], mapping_df['gene_symbol']))
    print(f"加载了 {len(mapping_dict)} 个映射")
    
else:
    print("未找到缓存，开始查询mygene.info...")
    
    # 分批查询
    mapping_dict = {}
    failed_queries = []
    
    # 获取所有需要查询的ENSG ID
    ensg_to_query = [g for g in ensg_ids if g.startswith('ENSG')]
    
    # 使用tqdm显示进度
    for i in tqdm(range(0, len(ensg_to_query), 100), desc="查询进度"):
        batch = ensg_to_query[i:i+100]
        
        try:
            # mygene.info 的批量查询接口
            url = "http://mygene.info/v3/query"
            params = {
                'q': ' OR '.join([f'ensembl.gene:{g}' for g in batch]),
                'fields': 'ensembl.gene,symbol',
                'species': 'human',
                'size': len(batch)
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for hit in data.get('hits', []):
                    if 'ensembl' in hit:
                        ens_data = hit['ensembl']
                        if isinstance(ens_data, list):
                            for ens in ens_data:
                                if 'gene' in ens and ens['gene'] in batch:
                                    mapping_dict[ens['gene']] = hit.get('symbol', ens['gene'])
                        else:
                            if 'gene' in ens_data and ens_data['gene'] in batch:
                                mapping_dict[ens_data['gene']] = hit.get('symbol', ens_data['gene'])
            
            time.sleep(0.2)  # 避免请求过快
            
        except Exception as e:
            print(f"批次 {i//100 + 1} 失败: {e}")
            failed_queries.extend(batch)
    
    print(f"\n查询完成:")
    print(f"  成功映射: {len(mapping_dict)}")
    print(f"  失败: {len(failed_queries)}")
    
    # 保存映射到CSV
    mapping_df = pd.DataFrame([
        {'ensembl_id': k, 'gene_symbol': v} 
        for k, v in mapping_dict.items()
    ])
    mapping_df.to_csv(gene_symbol_cache, index=False)
    print(f"映射已保存到: {gene_symbol_cache}")

# 4. 执行基因名转换
print("\n" + "="*80)
print("4. 执行基因名转换")
print("="*80)

# 在adata.var中添加Gene_Symbol列
gene_symbols = []
mapping_status = []

for i, (gene, base_gene) in enumerate(zip(original_genes, ensg_ids)):
    if base_gene.startswith('ENSG') and base_gene in mapping_dict:
        # ENSG ID成功映射到Gene Symbol
        gene_symbols.append(mapping_dict[base_gene])
        mapping_status.append('mapped')
    elif gene.startswith('ENSG'):
        # ENSG ID但未映射成功
        gene_symbols.append(gene)  # 保留原ID
        mapping_status.append('unmapped')
    else:
        # 已经是Gene Symbol格式
        gene_symbols.append(gene)
        mapping_status.append('already_symbol')

# 添加Gene_Symbol列
adata.var['Gene_Symbol'] = gene_symbols
adata.var['mapping_status'] = mapping_status

print(f"转换结果统计:")
status_counts = adata.var['mapping_status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count} ({count/len(adata.var):.1%})")

print(f"\nGene_Symbol列已添加，示例:")
print(adata.var[['Gene_Symbol', 'mapping_status']].head(10))

# 5. 检查重复的Gene_Symbol
print("\n" + "="*80)
print("5. 检查重复的Gene_Symbol")
print("="*80)

duplicate_symbols = adata.var['Gene_Symbol'].duplicated().sum()
print(f"重复的Gene_Symbol数量: {duplicate_symbols}")

if duplicate_symbols > 0:
    print("处理重复的Gene_Symbol...")
    
    # 对于重复的Gene_Symbol，添加后缀
    symbol_counts = {}
    new_symbols = []
    
    for symbol in adata.var['Gene_Symbol']:
        if symbol in symbol_counts:
            symbol_counts[symbol] += 1
            new_symbols.append(f"{symbol}.{symbol_counts[symbol]}")
        else:
            symbol_counts[symbol] = 0
            new_symbols.append(symbol)
    
    adata.var['Gene_Symbol_unique'] = new_symbols
    print(f"已创建唯一Gene_Symbol: {adata.var['Gene_Symbol_unique'].nunique()} 个唯一值")

# 6. 检查重要免疫基因
print("\n" + "="*80)
print("6. 检查重要免疫基因的映射情况")
print("="*80)

# 常见免疫基因列表（用于验证）
immune_genes = [
    'CD14', 'CD19', 'CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B',
    'MS4A1', 'NKG7', 'GNLY', 'FCGR3A', 'LYZ', 'IL7R', 'CCR7',
    'S100A8', 'S100A9', 'LILRA4', 'IRF7', 'TCF4', 'CLEC4C',
    'CD1C', 'FCER1A', 'CST3', 'HLA-DRA', 'HLA-DRB1'
]

print("检查关键免疫基因的映射情况:")
found_genes = []
for gene in immune_genes:
    # 在Gene_Symbol中查找
    if gene in adata.var['Gene_Symbol'].values:
        # 找到该基因的原始ENSG ID
        idx = np.where(adata.var['Gene_Symbol'] == gene)[0][0]
        original = adata.var.index[idx]
        found_genes.append(gene)
        print(f"  ✅ {gene} <- {original}")
    else:
        print(f"  ❌ {gene}")

print(f"\n找到 {len(found_genes)}/{len(immune_genes)} 个免疫基因")

# 7. 可视化映射结果
print("\n" + "="*80)
print("7. 生成映射报告")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 映射状态饼图
status_counts = adata.var['mapping_status'].value_counts()
colors = {'mapped': '#2ecc71', 'unmapped': '#e74c3c', 'already_symbol': '#3498db'}
status_colors = [colors.get(s, '#95a5a6') for s in status_counts.index]
axes[0,0].pie(status_counts.values, labels=status_counts.index, 
              autopct='%1.1f%%', colors=status_colors)
axes[0,0].set_title(f'基因名映射状态 (总{len(adata.var)}基因)')

# 2. 基因名前缀分布（转换前）
prefix_before = pd.Series([g[:6] if len(g) >= 6 else g for g in original_genes[:1000]]).value_counts()
axes[0,1].bar(range(len(prefix_before[:10])), prefix_before.values[:10])
axes[0,1].set_xticks(range(len(prefix_before[:10])))
axes[0,1].set_xticklabels(prefix_before.index[:10], rotation=45, ha='right')
axes[0,1].set_title('转换前基因名前缀 (前1000个)')

# 3. 基因名前缀分布（转换后）
genes_sample = adata.var['Gene_Symbol'][:1000]
prefix_after = pd.Series([g[:4] if len(g) >= 4 else g for g in genes_sample]).value_counts()
axes[1,0].bar(range(len(prefix_after[:10])), prefix_after.values[:10])
axes[1,0].set_xticks(range(len(prefix_after[:10])))
axes[1,0].set_xticklabels(prefix_after.index[:10], rotation=45, ha='right')
axes[1,0].set_title('转换后基因名前缀 (Gene_Symbol)')

# 4. 免疫基因命中率
immune_found = len(found_genes)
immune_total = len(immune_genes)
axes[1,1].bar(['免疫基因命中率'], [immune_found/immune_total])
axes[1,1].set_ylim([0, 1])
axes[1,1].set_ylabel('比例')
axes[1,1].set_title(f'免疫基因命中: {immune_found}/{immune_total}')
axes[1,1].text(0, immune_found/immune_total, f'{immune_found/immune_total:.1%}', 
               ha='center', va='bottom')

plt.tight_layout()
plt.savefig(fig_dir / 'gene_symbol_mapping_report.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"映射报告已保存: {fig_dir / 'gene_symbol_mapping_report.pdf'}")

# 8. 保存数据
print("\n" + "="*80)
print("8. 保存转换后的数据")
print("="*80)

# 保存完整数据（保持原基因名不变，只添加Gene_Symbol列）
converted_path = processed_data_dir / 'data_with_gene_symbol.h5ad'
adata.write(converted_path)
print(f"✅ 数据已保存: {converted_path}")
print(f"   基因名保持原样，Gene_Symbol列已添加")

# 保存映射摘要
mapping_summary = pd.DataFrame({
    'total_genes': [len(adata.var)],
    'mapped': [status_counts.get('mapped', 0)],
    'unmapped': [status_counts.get('unmapped', 0)],
    'already_symbol': [status_counts.get('already_symbol', 0)],
    'mapping_rate': [f"{status_counts.get('mapped', 0)/len(adata.var):.1%}"]
})
mapping_summary.to_csv(processed_data_dir / 'gene_mapping_summary.csv', index=False)

# 保存未映射基因列表
unmapped_genes = adata.var[adata.var['mapping_status'] == 'unmapped'][['Gene_Symbol']]
if len(unmapped_genes) > 0:
    unmapped_genes.to_csv(processed_data_dir / 'unmapped_genes.csv')
    print(f"  未映射基因列表: {processed_data_dir / 'unmapped_genes.csv'}")

# 9. 验证数据完整性
print("\n" + "="*80)
print("9. 验证数据完整性")
print("="*80)

print("检查关键信息:")
print(f"  ✅ 细胞数: {adata.n_obs}")
print(f"  ✅ 基因数: {adata.n_vars}")
print(f"  ✅ Gene_Symbol列存在: {'Gene_Symbol' in adata.var.columns}")
print(f"  ✅ PCA存在: {'X_pca' in adata.obsm}")
print(f"  ✅ UMAP存在: {'X_umap' in adata.obsm}")

# 检查细胞类型注释
celltype_cols = [c for c in adata.obs.columns if 'cell' in c.lower() or 'type' in c.lower()]
if celltype_cols:
    print(f"  ✅ 细胞类型注释列: {celltype_cols}")
    for col in celltype_cols[:2]:
        print(f"      {col}: {adata.obs[col].nunique()} 个类别")
else:
    print(f"  ❌ 未找到细胞类型注释")

print("\n" + "="*80)
print("✅ 基因名转换完成！")
print("="*80)

print("\n数据使用说明:")
print("1. 原基因名保存在adata.var.index")
print("2. Gene_Symbol列已添加到adata.var")
print("3. 后续分析中，你可以选择使用原基因名或Gene_Symbol")
print("\n例如:")
print("  # 使用Gene_Symbol作为基因名")
print("  adata.var_names = adata.var['Gene_Symbol']")
print("\n或者保留原基因名，在分析中引用Gene_Symbol列")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE189050 数据探索 - 修复版v2
尝试多种方法读取Seurat对象
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置路径
BASE_DIR = "/home/h3033/statics/GEO_data/GSE/figure6"
DATA_DIR = os.path.join(BASE_DIR, "data/raw/GSE189050")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/exploration")

# 创建目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("GSE189050 数据探索 - 修复版v2")
print("="*80)

# ------------------------------------------------------------
# 1. 检查文件
# ------------------------------------------------------------
print("\n1. 检查文件...")

rds_file = os.path.join(DATA_DIR, "GSE189050_final_seurat.RDS")

if not os.path.exists(rds_file):
    # 尝试找.gz文件
    gz_file = rds_file + ".gz"
    if os.path.exists(gz_file):
        print(f"发现压缩文件，解压中...")
        import gzip
        with gzip.open(gz_file, 'rb') as f_in:
            with open(rds_file, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✓ 解压完成")
    else:
        print(f"✗ 文件不存在")
        exit(1)

file_size = os.path.getsize(rds_file) / (1024**3)
print(f"文件大小: {file_size:.2f} GB")

# ------------------------------------------------------------
# 2. 尝试用pickle直接读取（Seurat对象本质是R的S3对象，但有时可以用pickle）
# ------------------------------------------------------------
print("\n2. 尝试用pickle读取...")

try:
    with open(rds_file, 'rb') as f:
        obj = pickle.load(f)
    print("✓ pickle读取成功!")
    print(f"对象类型: {type(obj)}")
    
    # 如果是列表或字典，查看结构
    if isinstance(obj, dict):
        print(f"字典键: {list(obj.keys())[:10]}")
    elif hasattr(obj, '__dict__'):
        print(f"属性: {dir(obj)[:20]}")
except Exception as e:
    print(f"✗ pickle读取失败: {e}")

# ------------------------------------------------------------
# 3. 尝试用lomap读取（专门用于RDS的Python库）
# ------------------------------------------------------------
print("\n3. 尝试用lomap读取...")

try:
    import lomap
    print("lomap可用")
    
    # lomap可以读取RDS
    obj = lomap.read_rds(rds_file)
    print(f"✓ lomap读取成功!")
    print(f"对象类型: {type(obj)}")
    
    if isinstance(obj, dict):
        print(f"键: {list(obj.keys())}")
        
        # 尝试找到表达矩阵
        for key in ['counts', 'data', 'scale.data', 'raw.data']:
            if key in obj:
                print(f"找到 {key}, 形状: {obj[key].shape}")
                
                # 创建AnnData
                if hasattr(obj[key], 'toarray'):
                    X = obj[key].toarray()
                else:
                    X = obj[key]
                
                adata = sc.AnnData(X=X.T if X.shape[0] < X.shape[1] else X)
                print(f"✓ 创建AnnData: {adata.shape}")
                break
except ImportError:
    print("lomap未安装，尝试安装...")
    os.system("pip install lomap")
    
# ------------------------------------------------------------
# 4. 如果以上都失败，尝试用rpy2的降级模式
# ------------------------------------------------------------
print("\n4. 尝试用rpy2降级模式...")

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    
    # 设置R选项，避免版本检查
    ro.r('options(Seurat.object.assay.version = "v3")')
    
    # 读取RDS
    ro.r(f'seurat_obj <- readRDS("{rds_file}")')
    
    # 尝试获取数据 - 使用更通用的方法
    try:
        # 方法1: 尝试获取counts
        ro.r('''
            if ("RNA" %in% Assays(seurat_obj)) {
                counts <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
            } else if ("SCT" %in% Assays(seurat_obj)) {
                counts <- GetAssayData(seurat_obj, assay = "SCT", slot = "counts")
            } else {
                counts <- GetAssayData(seurat_obj, slot = "counts")
            }
        ''')
    except:
        # 方法2: 尝试获取data
        ro.r('''
            if ("RNA" %in% Assays(seurat_obj)) {
                data <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
            } else {
                data <- GetAssayData(seurat_obj, slot = "data")
            }
        ''')
    
    # 获取元数据
    ro.r('metadata <- seurat_obj[[]]')
    
    # 转换为pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        if 'counts' in ro.r('ls()'):
            df = ro.conversion.rpy2py(ro.r('counts'))
        else:
            df = ro.conversion.rpy2py(ro.r('data'))
        metadata_df = ro.conversion.rpy2py(ro.r('metadata'))
    
    print(f"✓ rpy2读取成功!")
    print(f"表达矩阵形状: {df.shape}")
    print(f"元数据形状: {metadata_df.shape}")
    
    # 创建AnnData
    if df.shape[0] > df.shape[1]:
        adata = sc.AnnData(X=df.values.T if sparse.issparse(df.values) else df.values.T)
        adata.obs_names = df.columns.astype(str)
        adata.var_names = df.index.astype(str)
    else:
        adata = sc.AnnData(X=df.values if sparse.issparse(df.values) else df.values)
        adata.obs_names = df.index.astype(str)
        adata.var_names = df.columns.astype(str)
    
    # 添加元数据
    for col in metadata_df.columns:
        if len(metadata_df[col]) == adata.n_obs:
            adata.obs[col] = metadata_df[col].values
    
    print(f"✓ AnnData创建成功: {adata.shape}")
    
except Exception as e:
    print(f"✗ rpy2降级模式也失败: {e}")
    adata = None

# ------------------------------------------------------------
# 5. 如果成功读取，保存为h5ad
# ------------------------------------------------------------
if adata is not None:
    print("\n5. 保存为h5ad格式...")
    h5ad_file = os.path.join(OUTPUT_DIR, "GSE189050.h5ad")
    adata.write(h5ad_file)
    print(f"✓ 已保存: {h5ad_file}")
    
    # 显示基本信息
    print(f"\n6. 数据摘要:")
    print(f"   细胞数: {adata.n_obs}")
    print(f"   基因数: {adata.n_vars}")
    print(f"   obs列: {list(adata.obs.columns)}")
    print(f"   obsm键: {list(adata.obsm.keys())}")
    
    # 保存探索报告
    report = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'obs_columns': list(adata.obs.columns),
        'obsm_keys': list(adata.obsm.keys())
    }
    
    import json
    with open(os.path.join(OUTPUT_DIR, "exploration_summary.json"), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ 探索完成! 结果保存在: {OUTPUT_DIR}")
else:
    print("\n❌ 所有方法都失败，建议使用R转换")
    print("\n请运行R脚本转换:")
    print("Rscript code/convert_seurat_to_h5ad.R")
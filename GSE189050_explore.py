#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSE189050 数据结构确认与质控检查
"""

import numpy as np
import pandas as pd
import anndata as ad

H5AD = "/home/h3033/statics/GEO_data/GSE/figure6/output/exploration/GSE189050.h5ad"

adata = ad.read_h5ad(H5AD)
print(f"读取完成: {adata.shape[0]} cells × {adata.shape[1]} genes\n")

# ── 1. 基因名存储位置 ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. 基因名存储位置")
print("=" * 60)
print(f"位置: adata.var_names")
print(f"总基因数: {adata.n_vars}")
print(f"前10个: {list(adata.var_names[:10])}")

# ── 2. 疾病状态列 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 疾病状态列")
print("=" * 60)
print(f"列名: 'classification'")
print(adata.obs["classification"].value_counts().to_string())

# ── 3. 细胞类型列 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 细胞类型列（共3个粒度）")
print("=" * 60)

print("\n[A] coarse_cell_type（粗粒度，8类）")
print(adata.obs["coarse_cell_type"].value_counts().to_string())

print("\n[B] fine_cell_type（细粒度，14类）")
print(adata.obs["fine_cell_type"].value_counts().to_string())

print("\n[C] clusters_annotated（最细，25个cluster）")
print(adata.obs["clusters_annotated"].value_counts().to_string())

# ── 4. UMAP降维结果 ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. UMAP降维结果")
print("=" * 60)
print(f"键名: 'X_umap_wnn'（WNN整合 RNA+CITE）")
print(f"形状: {adata.obsm['X_umap_wnn'].shape}")
umap = adata.obsm["X_umap_wnn"]
print(f"UMAP1 范围: [{umap[:, 0].min():.2f}, {umap[:, 0].max():.2f}]")
print(f"UMAP2 范围: [{umap[:, 1].min():.2f}, {umap[:, 1].max():.2f}]")
print(f"\n其他可用降维: {[k for k in adata.obsm.keys() if k != 'X_umap_wnn']}")

# ── 5. 线粒体基因阈值 & 红细胞污染检查 ────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 线粒体基因与红细胞污染")
print("=" * 60)

# 线粒体基因
mt_genes = [g for g in adata.var_names if g.startswith("MT-")]
print(f"\n[线粒体基因] MT-开头共 {len(mt_genes)} 个: {mt_genes}")

mt = adata.obs["percent_mt"]
print(f"\npercent_mt 分布:")
print(f"  中位数: {mt.median():.2f}%  最大值: {mt.max():.2f}%")
print(f"  >10% 细胞: {(mt > 10).sum()} / {len(mt)} ({(mt > 10).mean()*100:.1f}%)")

# 红细胞污染
hemo = adata.obs["percent_hemo"]
print(f"\npercent_hemo 分布（红细胞血红蛋白基因占比）:")
print(f"  中位数: {hemo.median():.4f}%  最大值: {hemo.max():.4f}%")
print(f"  >1%  细胞: {(hemo > 1).sum()}")
print(f"  >5%  细胞: {(hemo > 5).sum()}")

# 是否有红细胞cluster标注
rbc_keywords = ["erythro", "rbc", "red blood", "hemo", "erythroid"]
for col in ["coarse_cell_type", "fine_cell_type", "clusters_annotated"]:
    vals = adata.obs[col].unique()
    rbc_hits = [v for v in vals if any(k in str(v).lower() for k in rbc_keywords)]
    print(f"\n  {col} 中红细胞相关标签: {rbc_hits if rbc_hits else '无（数据集已预先去除红细胞）'}")

# ── 小结 ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("小结")
print("=" * 60)
print("""
基因名    : adata.var_names
疾病状态  : adata.obs['classification']  (Control / SLE INACT / SLE ACT)
细胞类型  : adata.obs['coarse_cell_type']  粗粒度 8类
            adata.obs['fine_cell_type']    细粒度 14类
            adata.obs['clusters_annotated'] 最细  25类
UMAP      : adata.obsm['X_umap_wnn']  (WNN整合 RNA+CITE)
线粒体    : percent_mt 列已存在；>10%的细胞仅占 2.3%（已在上游QC过滤）
红细胞    : percent_hemo 最大值仅 1.49%，无细胞超过 5% 阈值
            细胞类型标注中亦无红细胞cluster（数据集已预先清除）
""")

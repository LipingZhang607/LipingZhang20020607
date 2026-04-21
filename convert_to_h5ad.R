#!/usr/bin/env Rscript
# GSE189050: Seurat RDS -> 中间文件 (供Python重建h5ad)
# 使用 reticulate 调用系统Python的anndata，绕过SeuratDisk版本不兼容问题

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(reticulate)
})

# 使用系统Python（已有anndata）
use_python("/home/h3033/miniconda3/bin/python3", required = TRUE)

rds_file  <- "/home/h3033/statics/GEO_data/GSE/figure6/data/raw/GSE189050/GSE189050_final_seurat.RDS"
out_dir   <- "/home/h3033/statics/GEO_data/GSE/figure6/output/exploration"
h5ad_file <- file.path(out_dir, "GSE189050.h5ad")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ── 1. 读取 Seurat 对象 ────────────────────────────────────────────────────────
cat("[1/5] 读取 RDS...\n")
seurat_obj <- readRDS(rds_file)
cat("      细胞:", ncol(seurat_obj), " 基因:", nrow(seurat_obj), "\n")
cat("      Assays:", paste(Assays(seurat_obj), collapse=", "), "\n")
cat("      Reductions:", paste(Reductions(seurat_obj), collapse=", "), "\n")

# ── 2. 提取 RNA counts 矩阵 ───────────────────────────────────────────────────
cat("[2/5] 提取 RNA counts 矩阵...\n")
DefaultAssay(seurat_obj) <- "RNA"
# 兼容 Seurat v4/v5
counts_mat <- tryCatch(
  LayerData(seurat_obj, assay = "RNA", layer = "counts"),
  error = function(e) GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
)
cat("      矩阵维度:", nrow(counts_mat), "genes x", ncol(counts_mat), "cells\n")
cat("      稀疏度:", round(1 - nnzero(counts_mat)/length(counts_mat), 4), "\n")

# ── 3. 提取元数据 ─────────────────────────────────────────────────────────────
cat("[3/5] 提取元数据...\n")
metadata <- seurat_obj[[]]
cat("      元数据列数:", ncol(metadata), "\n")
cat("      关键列:", paste(intersect(c("orig.ident","subject_id","classification",
                                       "coarse_cell_type","fine_cell_type",
                                       "clusters_annotated"), colnames(metadata)), collapse=", "), "\n")

# ── 4. 提取降维坐标 ───────────────────────────────────────────────────────────
cat("[4/5] 提取降维坐标...\n")
embeddings <- list()
for (red in Reductions(seurat_obj)) {
  emb <- Embeddings(seurat_obj, reduction = red)
  embeddings[[red]] <- emb
  cat("      -", red, ":", nrow(emb), "x", ncol(emb), "\n")
}

# ── 5. 用Python anndata写h5ad ─────────────────────────────────────────────────
cat("[5/5] 用Python anndata写h5ad...\n")

# 将稀疏矩阵转为 CSC 格式（anndata期望genes x cells -> 转置为cells x genes）
counts_csc <- as(t(counts_mat), "CsparseMatrix")  # cells x genes

# 导入Python模块
ad    <- import("anndata")
scipy <- import("scipy.sparse")
np    <- import("numpy")
pd    <- import("pandas")

# R 的 CsparseMatrix 是 CSC 格式：@p 是列指针，@i 是行索引
# counts_csc 形状: cells x genes (88053 x 27254)
# 用 csc_matrix，anndata 会自动转 CSR 存储
X_csr <- scipy$csc_matrix(
  tuple(
    np$array(counts_csc@x,   dtype = "float32"),
    np$array(counts_csc@i,   dtype = "int32"),
    np$array(counts_csc@p,   dtype = "int32")
  ),
  shape = tuple(as.integer(nrow(counts_csc)), as.integer(ncol(counts_csc)))
)

# 细胞名和基因名
cell_names <- colnames(counts_mat)
gene_names <- rownames(counts_mat)

# 元数据 - 转换为Python dict (只保留字符/数值列，避免因子问题)
meta_clean <- as.data.frame(lapply(metadata, function(x) {
  if (is.factor(x)) as.character(x) else x
}), stringsAsFactors = FALSE)

# 转为Python pandas DataFrame
obs_df <- r_to_py(meta_clean)

# 构建 obsm (降维坐标)
obsm <- list()
for (red_name in names(embeddings)) {
  key <- paste0("X_", red_name)
  obsm[[key]] <- np$array(embeddings[[red_name]], dtype = "float32")
}

# 创建 AnnData
adata <- ad$AnnData(
  X    = X_csr,
  obs  = obs_df,
  obsm = obsm
)
adata$obs_names <- r_to_py(as.list(cell_names))
adata$var_names <- r_to_py(as.list(gene_names))

cat("      AnnData shape:", adata$shape[[1]], "x", adata$shape[[2]], "\n")

# 写入 h5ad
if (file.exists(h5ad_file)) file.remove(h5ad_file)
adata$write_h5ad(h5ad_file)

cat("\n完成!\n")
cat("  h5ad:", h5ad_file, "\n")
size_gb <- file.info(h5ad_file)$size / 1024^3
cat("  大小:", round(size_gb, 2), "GB\n")

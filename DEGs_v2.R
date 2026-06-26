#!/usr/bin/env Rscript
# ==============================================================================
# SLE (Systemic Lupus Erythematosus) Transcriptomics Analysis Pipeline
# ==============================================================================
# Version: 2.0 (Refactored)
# Author: Zhang Liping
# Date: 2026-01-09
#
# Description:
#   Comprehensive transcriptomics analysis for SLE vs Control samples across
#   multiple GSE datasets. Includes data integration, batch correction, DEG
#   analysis, and functional enrichment.
#
# Features:
#   - Modular architecture for better maintainability
#   - Robust error handling and recovery
#   - Support for supplementary data (FPKM/counts)
#   - Detailed logging system
#   - Configuration file support
#   - Handles dual-platform datasets (e.g., GSE110685)
# ==============================================================================

# ==============================================================================
# 0. CONFIGURATION AND SETUP
# ==============================================================================

# Set working directory
# ── Arial 字体全局设置（依赖系统 Cairo）──────────────────────────────
if (capabilities("cairo")) {
  grDevices::X11.options(type = "cairo")
  # 注意：不设 pdf.options(family)，避免影响包内部 pdf(file=NULL) 调用
  # Arial 由各 cairo_pdf(..., family="Arial") 显式指定
}

setwd("~/statics/GEO_data/GSE/figure3")

# Global configuration
CONFIG <- list(
  # Directories
  data_dir = "~/statics/GEO_data/GSE/figure3/GSE数据",
  output_dir = "SLE_Analysis_Results_v2",

  # Analysis parameters
  log_fc_threshold = 0.5,
  adj_p_threshold = 0.05,
  min_samples_per_batch = 3,
  min_expr_percentile = 0.25,
  min_expr_samples_pct = 0.25,

  # Processing options
  use_combat = TRUE,
  use_supplementary_data = TRUE,
  parallel_cores = 4,

  # Output options
  save_intermediate = TRUE,
  create_qc_plots = TRUE,
  verbose = TRUE
)

# ==============================================================================
# 1. PACKAGE LOADING
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("  SLE Transcriptomics Analysis Pipeline v2.0\n")
cat("================================================================================\n")
cat("\n")

cat("Loading required packages...\n")

suppressPackageStartupMessages({
  # Core packages
  library(GEOquery)
  library(Biobase)
  library(limma)
  library(affy)
  library(oligo)
  library(sva)

  # Data manipulation
  library(dplyr)
  library(tibble)
  library(stringr)
  library(tidyr)

  # Visualization
  library(ggplot2)
  library(pheatmap)
  library(RColorBrewer)

  # Enrichment analysis
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(enrichplot)
  library(DOSE)
  library(GSEABase)
  library(GSVA)

  # RNA-seq analysis
  library(DESeq2)
  library(edgeR)

  # File reading
  if (require("readxl", quietly = TRUE)) {
    library(readxl)
  } else {
    cat("WARNING: readxl package not available (needed for Excel files)\n")
  }
})

cat("All packages loaded successfully!\n\n")

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

#' Create timestamped log message
#'
#' @param message Character string to log
#' @param level Log level (INFO, WARNING, ERROR)
#' @param verbose Boolean, print to console
log_message <- function(message, level = "INFO", verbose = CONFIG$verbose) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_entry <- sprintf("[%s] %s: %s", timestamp, level, message)

  if (verbose) {
    cat(log_entry, "\n")
  }

  # Write to log file
  log_file <- file.path(CONFIG$output_dir, "analysis.log")
  write(log_entry, file = log_file, append = TRUE)

  return(invisible(log_entry))
}

#' Safe directory creation
#'
#' @param path Directory path to create
create_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
    log_message(sprintf("Created directory: %s", path))
  }
}

#' Initialize output directories
init_directories <- function() {
  dirs <- c(
    CONFIG$output_dir,
    file.path(CONFIG$output_dir, "QC"),
    file.path(CONFIG$output_dir, "DEGs"),
    file.path(CONFIG$output_dir, "Enrichment"),
    file.path(CONFIG$output_dir, "GSEA"),
    file.path(CONFIG$output_dir, "GSVA"),
    file.path(CONFIG$output_dir, "intermediate")
  )

  lapply(dirs, create_dir)
}

#' Save R object with metadata
#'
#' @param obj R object to save
#' @param filename Filename (without path)
#' @param description Description of the object
save_object <- function(obj, filename, description = "") {
  filepath <- file.path(CONFIG$output_dir, "intermediate", filename)
  saveRDS(obj, filepath)
  log_message(sprintf("Saved: %s - %s", filename, description))
}

#' Load saved R object
#'
#' @param filename Filename (without path)
load_object <- function(filename) {
  filepath <- file.path(CONFIG$output_dir, "intermediate", filename)
  if (file.exists(filepath)) {
    log_message(sprintf("Loading: %s", filename))
    return(readRDS(filepath))
  } else {
    log_message(sprintf("File not found: %s", filename), level = "WARNING")
    return(NULL)
  }
}

# ==============================================================================
# 3. DATASET CONFIGURATION
# ==============================================================================

#' Define all GSE datasets with metadata
get_dataset_info <- function() {
  gse_info <- data.frame(
    GSE_ID = c(
      # GPL570 (Affymetrix HG-U133 Plus 2.0)
      "GSE13887", "GSE185047", "GSE50772", "GSE61635",
      # GPL10558 (Illumina HumanHT-12 V4.0)
      "GSE138458", "GSE65391", "GSE72326", "GSE81622",
      # GPL16791 (Illumina HiSeq 2500)
      "GSE110685-GPL16791", "GSE112087", "GSE72509",
      # GPL1291 (Affymetrix HG-Focus)
      "GSE17755", "GSE20864",
      # GPL96 (Affymetrix HG-U133A)
      "GSE10325",
      # GPL13158 (Affymetrix HT HG-U133+ PM)
      "GSE121239",
      # GPL16699 (Agilent-039494)
      "GSE154851",
      # GPL17077 (Agilent-039494 v2)
      "GSE148601",
      # GPL21290 (Illumina HiSeq 3000)
      "GSE110685-GPL21290",
      # GPL21970 (Illumina HiSeq 4000)
      "GSE99967",
      # GPL6244 (Affymetrix HG Gene 1.0 ST)
      "GSE50635",
      # GPL6884 (Illumina HumanWG-6 v3.0)
      "GSE24706"
    ),
    GPL_ID = c(
      "GPL570", "GPL570", "GPL570", "GPL570",
      "GPL10558", "GPL10558", "GPL10558", "GPL10558",
      "GPL16791", "GPL16791", "GPL16791",
      "GPL1291", "GPL1291",
      "GPL96",
      "GPL13158",
      "GPL16699",
      "GPL17077",
      "GPL21290",
      "GPL21970",
      "GPL6244",
      "GPL6884"
    ),
    Platform_Type = c(
      "microarray", "microarray", "microarray", "microarray",
      "microarray", "microarray", "microarray", "microarray",
      "rnaseq", "rnaseq", "rnaseq",
      "microarray", "microarray",
      "microarray",
      "microarray",
      "microarray",
      "microarray",
      "rnaseq",
      "rnaseq",
      "microarray",
      "microarray"
    ),
    File_Path = c(
      "GSE数据/GPL570/GSE13887_series_matrix.txt.gz",
      "GSE数据/GPL570/GSE185047_series_matrix.txt.gz",
      "GSE数据/GPL570/GSE50772_series_matrix.txt",
      "GSE数据/GPL570/GSE61635_series_matrix.txt.gz",
      "GSE数据/GPL10558/GSE138458_series_matrix.txt",
      "GSE数据/GPL10558/GSE65391_series_matrix.txt.gz",
      "GSE数据/GPL10558/GSE72326_series_matrix.txt",
      "GSE数据/GPL10558/GSE81622_series_matrix.txt",
      "GSE数据/GPL16791/GSE110685_PCarlucci_RNA-Seq_tc_rpkm_combined_submitted.xlsx",
      "GSE数据/GPL16791/GSE112087_counts-matrix-EnsembIDs-GRCh37.p10.txt.gz",
      "GSE数据/GPL16791/GSE72509_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz",
      "GSE数据/GPL1291/GSE17755_series_matrix.txt.gz",
      "GSE数据/GPL1291/GSE20864_series_matrix.txt",
      "GSE数据/GPL96/GSE10325_series_matrix.txt.gz",
      "GSE数据/GPL13158/GSE121239_series_matrix.txt.gz",
      "GSE数据/GPL16699/GSE154851_series_matrix.txt.gz",
      "GSE数据/GPL17077/GSE148601_series_matrix.txt.gz",
      "GSE数据/GPL21290/GSE110685-GPL21290_series_matrix.txt.gz",
      "GSE数据/GPL21970/GSE99967_series_matrix.txt.gz",
      "GSE数据/GPL6244/GSE50635_series_matrix.txt.gz",
      "GSE数据/GPL6884/GSE24706_series_matrix.txt.gz"
    ),
    Data_Type = c(
      "series_matrix", "series_matrix", "series_matrix", "series_matrix",
      "series_matrix", "series_matrix", "series_matrix", "series_matrix",
      "supplementary_xlsx", "supplementary_counts", "supplementary_fpkm",
      "series_matrix", "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix",
      "series_matrix"
    ),
    stringsAsFactors = FALSE
  )

  return(gse_info)
}

# ==============================================================================
# 4. DATA LOADING MODULE
# ==============================================================================

#' Read supplementary counts/FPKM data
#'
#' @param file_path Path to supplementary data file
#' @param gse_id GSE identifier
#' @return Expression matrix
read_supplementary_counts <- function(file_path, gse_id) {
  log_message(sprintf("  Reading supplementary counts/FPKM file: %s", basename(file_path)))

  if (grepl("\\.gz$", file_path)) {
    expr_matrix <- read.delim(gzfile(file_path), row.names = 1, check.names = FALSE)
  } else {
    expr_matrix <- read.delim(file_path, row.names = 1, check.names = FALSE)
  }

  expr_matrix <- as.matrix(expr_matrix)
  log_message(sprintf("  Loaded: %d genes x %d samples", nrow(expr_matrix), ncol(expr_matrix)))

  return(expr_matrix)
}

#' Read supplementary Excel data (GSE110685)
#'
#' @param file_path Path to Excel file
#' @param gse_id GSE identifier
#' @return Expression matrix
read_supplementary_xlsx <- function(file_path, gse_id) {
  log_message(sprintf("  Reading Excel file: %s", basename(file_path)))

  suppressPackageStartupMessages({
    if (!require("readxl", quietly = TRUE)) {
      stop("readxl package required for Excel files")
    }
  })

  expr_data <- read_excel(file_path)

  # Find gene column
  gene_cols <- c("Gene", "GeneSymbol", "Gene.Symbol", "gene", "symbol", "SYMBOL")
  gene_col <- NULL
  for (col in gene_cols) {
    if (col %in% colnames(expr_data)) {
      gene_col <- col
      break
    }
  }

  if (is.null(gene_col)) {
    # Assume first column is gene IDs
    gene_col <- colnames(expr_data)[1]
  }

  rownames(expr_data) <- expr_data[[gene_col]]
  expr_data[[gene_col]] <- NULL

  expr_matrix <- as.matrix(expr_data)
  log_message(sprintf("  Loaded: %d genes x %d samples", nrow(expr_matrix), ncol(expr_matrix)))

  return(expr_matrix)
}

#' Infer sample groups from sample names (RNA-seq data)
#'
#' @param sample_names Vector of sample names
#' @return Vector of group labels (SLE/Control/NA)
infer_groups_from_names <- function(sample_names) {
  group <- rep(NA_character_, length(sample_names))

  # Patterns for SLE
  sle_patterns <- c("SLE", "MON_SLE", "Patient", "Case", "Disease")

  # Patterns for Control
  control_patterns <- c("NORM", "Control", "Healthy", "HC", "Normal", "CTL", "CTRL")

  for (i in seq_along(sample_names)) {
    name <- sample_names[i]

    # Check SLE patterns
    for (pattern in sle_patterns) {
      if (grepl(pattern, name, ignore.case = TRUE)) {
        group[i] <- "SLE"
        break
      }
    }

    # Check Control patterns (only if not already assigned as SLE)
    if (is.na(group[i])) {
      for (pattern in control_patterns) {
        if (grepl(pattern, name, ignore.case = TRUE)) {
          group[i] <- "Control"
          break
        }
      }
    }
  }

  return(group)
}

#' Read single GSE dataset
#'
#' @param gse_id GSE identifier
#' @param file_path Path to series matrix file or supplementary file
#' @param gpl_id GPL platform identifier
#' @param data_type Type of data file (series_matrix, supplementary_counts, supplementary_xlsx, supplementary_fpkm)
#' @return List containing expression matrix and phenotype data
read_gse_dataset <- function(gse_id, file_path, gpl_id, data_type = "series_matrix") {
  log_message(sprintf("Reading %s (Platform: %s, Type: %s)...", gse_id, gpl_id, data_type))

  tryCatch({
    # Check if file exists
    if (!file.exists(file_path)) {
      stop(sprintf("File not found: %s", file_path))
    }

    # Handle different data types
    if (data_type == "series_matrix") {
      # Standard series_matrix format
      gse <- getGEO(filename = file_path, getGPL = FALSE)
      expr_matrix <- exprs(gse)
      pheno_data <- pData(gse)

    } else if (data_type == "supplementary_counts" || data_type == "supplementary_fpkm") {
      # Supplementary counts/FPKM files (GSE112087, GSE72509)
      expr_matrix <- read_supplementary_counts(file_path, gse_id)

      # Try to get phenotype data from series_matrix file
      # Look for series_matrix file in the same directory
      dir_path <- dirname(file_path)
      base_gse <- gsub("-.*$", "", gse_id)  # Remove platform suffix
      matrix_files <- list.files(dir_path, pattern = paste0(base_gse, ".*series_matrix"),
                                full.names = TRUE)

      pheno_data <- NULL
      if (length(matrix_files) > 0) {
        log_message(sprintf("  Found series_matrix file for metadata: %s", basename(matrix_files[1])))
        tryCatch({
          gse_meta <- getGEO(filename = matrix_files[1], getGPL = FALSE)
          pheno_data_full <- pData(gse_meta)

          # Match samples by GSM IDs
          gsm_ids <- colnames(expr_matrix)
          if (all(gsm_ids %in% rownames(pheno_data_full))) {
            pheno_data <- pheno_data_full[gsm_ids, ]
            log_message(sprintf("  Matched %d samples with phenotype data", nrow(pheno_data)))
          }
        }, error = function(e) {
          log_message(sprintf("  Could not load metadata: %s", e$message), level = "WARNING")
        })
      }

      # If no phenotype data found, create minimal data
      if (is.null(pheno_data)) {
        sample_names <- colnames(expr_matrix)
        pheno_data <- data.frame(
          geo_accession = sample_names,
          title = sample_names,
          source_name_ch1 = sample_names,
          row.names = sample_names,
          stringsAsFactors = FALSE
        )

        # Infer groups from sample names
        inferred_groups <- infer_groups_from_names(sample_names)
        pheno_data$inferred_group <- inferred_groups
      }

    } else if (data_type == "supplementary_xlsx") {
      # Excel file (GSE110685)
      expr_matrix <- read_supplementary_xlsx(file_path, gse_id)

      # Try to get phenotype data from series_matrix file
      dir_path <- dirname(file_path)
      base_gse <- gsub("-.*$", "", gse_id)
      matrix_files <- list.files(dir_path, pattern = paste0(base_gse, ".*series_matrix"),
                                full.names = TRUE)

      pheno_data <- NULL
      if (length(matrix_files) > 0) {
        log_message(sprintf("  Found series_matrix file for metadata: %s", basename(matrix_files[1])))
        tryCatch({
          gse_meta <- getGEO(filename = matrix_files[1], getGPL = FALSE)
          pheno_data_full <- pData(gse_meta)

          # For GSE110685, try to match by sample order or find mapping
          # Check if we can match by GSM IDs or sample names
          sample_names <- colnames(expr_matrix)

          # Try matching by GSM pattern
          if (any(grepl("^GSM", rownames(pheno_data_full)))) {
            # Use all samples from series_matrix that match the platform
            pheno_data <- pheno_data_full
            # Subset expression matrix to match available samples
            common_samples <- intersect(colnames(expr_matrix), rownames(pheno_data))
            if (length(common_samples) > 0) {
              expr_matrix <- expr_matrix[, common_samples]
              pheno_data <- pheno_data[common_samples, ]
              log_message(sprintf("  Matched %d samples with phenotype data", nrow(pheno_data)))
            }
          }
        }, error = function(e) {
          log_message(sprintf("  Could not load metadata: %s", e$message), level = "WARNING")
        })
      }

      # If no phenotype data found, create minimal data
      if (is.null(pheno_data) || nrow(pheno_data) == 0) {
        sample_names <- colnames(expr_matrix)
        pheno_data <- data.frame(
          geo_accession = sample_names,
          title = sample_names,
          source_name_ch1 = sample_names,
          row.names = sample_names,
          stringsAsFactors = FALSE
        )

        # Infer groups from sample names
        inferred_groups <- infer_groups_from_names(sample_names)
        pheno_data$inferred_group <- inferred_groups
      }

    } else {
      stop(sprintf("Unknown data type: %s", data_type))
    }

    # Basic validation
    if (ncol(expr_matrix) == 0 || nrow(expr_matrix) == 0) {
      stop("Empty expression matrix")
    }

    log_message(sprintf("  Success! Samples: %d, Features: %d",
                       ncol(expr_matrix), nrow(expr_matrix)))

    return(list(
      expr = expr_matrix,
      pheno = pheno_data,
      gpl = gpl_id,
      data_type = data_type,
      status = "success"
    ))

  }, error = function(e) {
    log_message(sprintf("  Error reading %s: %s", gse_id, e$message),
               level = "ERROR")
    return(list(status = "failed", error = e$message))
  })
}

#' Load all GSE datasets
#'
#' @return List of dataset objects
load_all_datasets <- function() {
  gse_info <- get_dataset_info()
  gse_list <- list()

  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 1: Loading GSE Datasets\n")
  cat("================================================================================\n")
  cat(sprintf("Total datasets to load: %d\n\n", nrow(gse_info)))

  for (i in 1:nrow(gse_info)) {
    gse_id <- gse_info$GSE_ID[i]
    file_path <- gse_info$File_Path[i]
    gpl_id <- gse_info$GPL_ID[i]
    data_type <- gse_info$Data_Type[i]

    cat(sprintf("[%d/%d] ", i, nrow(gse_info)))
    result <- read_gse_dataset(gse_id, file_path, gpl_id, data_type)

    if (result$status == "success") {
      gse_list[[gse_id]] <- result
    }
  }

  log_message(sprintf("Successfully loaded %d/%d datasets",
                     length(gse_list), nrow(gse_info)))

  # Save intermediate result
  if (CONFIG$save_intermediate) {
    save_object(gse_list, "01_raw_data.rds", "Raw loaded datasets")
  }

  return(gse_list)
}

# ==============================================================================
# 5. SAMPLE GROUP EXTRACTION MODULE
# ==============================================================================

#' Extract group information from phenotype data
#'
#' @param pheno_data Phenotype data frame
#' @param gse_id Dataset identifier for special handling
#' @return Character vector of group labels (SLE/Control/NA)
extract_group_info <- function(pheno_data, gse_id = NULL) {
  # First, check if we have inferred groups (from supplementary data)
  if ("inferred_group" %in% colnames(pheno_data)) {
    inferred <- pheno_data$inferred_group
    if (sum(!is.na(inferred)) > 0) {
      n_sle <- sum(inferred == "SLE", na.rm = TRUE)
      n_control <- sum(inferred == "Control", na.rm = TRUE)
      n_unknown <- sum(is.na(inferred))

      log_message("  Using inferred groups from sample names")
      log_message(sprintf("  SLE=%d, Control=%d, Unknown=%d",
                         n_sle, n_control, n_unknown))
      return(inferred)
    }
  }

  # ============================================================================
  # SPECIAL HANDLING FOR PROBLEMATIC DATASETS
  # ============================================================================

  if (!is.null(gse_id)) {
    # GSE154851: Must use title column (Patient vs Control)
    if (gse_id == "GSE154851") {
      if ("title" %in% colnames(pheno_data)) {
        title <- as.character(pheno_data$title)
        group <- ifelse(grepl("^Patient", title), "SLE",
                       ifelse(grepl("^Control", title), "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using title column (GSE154851 special case)")
          log_message(sprintf("  SLE=%d, Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }

    # GSE50772: Use disease status:ch1
    if (gse_id == "GSE50772") {
      if ("disease status:ch1" %in% colnames(pheno_data)) {
        status <- as.character(pheno_data$`disease status:ch1`)
        group <- ifelse(status == "SLE", "SLE",
                       ifelse(status == "Control", "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using disease status:ch1 column (GSE50772)")
          log_message(sprintf("  SLE=%d, Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }

    # GSE13887: BC samples are Blood Controls
    if (gse_id == "GSE13887") {
      if ("title" %in% colnames(pheno_data)) {
        title <- as.character(pheno_data$title)
        group <- ifelse(grepl("^SLE", title), "SLE",
                       ifelse(grepl("^BC", title), "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using title column (GSE13887: BC = Blood Control)")
          log_message(sprintf("  SLE=%d, Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }

    # GSE17755: Must filter out RA and other diseases
    if (gse_id == "GSE17755") {
      if ("disease:ch1" %in% colnames(pheno_data)) {
        disease <- as.character(pheno_data$`disease:ch1`)
        group <- ifelse(disease == "systemic lupus erythematosus", "SLE",
                       ifelse(disease %in% c("healthy individual", "healthy child"), "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        n_excluded <- sum(is.na(group))
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using disease:ch1 column (GSE17755)")
          log_message(sprintf("  SLE=%d, Control=%d, Excluded=%d (RA and other diseases)",
                             n_sle, n_control, n_excluded))
          return(group)
        }
      }
    }

    # GSE24706: Must filter out FDR (first degree relatives)
    if (gse_id == "GSE24706") {
      if ("patient status:ch1" %in% colnames(pheno_data)) {
        status <- as.character(pheno_data$`patient status:ch1`)
        group <- ifelse(status == "SLE patients (SLE)", "SLE",
                       ifelse(status == "healthy control (HC)", "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        n_excluded <- sum(is.na(group))
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using patient status:ch1 column (GSE24706)")
          log_message(sprintf("  SLE=%d, Control=%d, Excluded=%d (FDR relatives)",
                             n_sle, n_control, n_excluded))
          return(group)
        }
      }
    }

    # GSE72326: Must filter out ICU_sepsis and Kidney Disease controls
    if (gse_id == "GSE72326") {
      if ("characteristics_ch1" %in% colnames(pheno_data)) {
        group_col <- as.character(pheno_data$characteristics_ch1)
        # Only keep SLE patients and healthy controls of SLE
        group <- ifelse(group_col == "group: SLE", "SLE",
                       ifelse(group_col == "group: Healthy control of SLE", "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        n_excluded <- sum(is.na(group))
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using characteristics_ch1 column (GSE72326)")
          log_message(sprintf("  SLE=%d, Control=%d, Excluded=%d (other disease controls)",
                             n_sle, n_control, n_excluded))
          return(group)
        }
      }
    }

    # GSE50635: RBP (RNA-binding protein) positive/negative SLE patients
    if (gse_id == "GSE50635") {
      if ("characteristics_ch1.3" %in% colnames(pheno_data)) {
        subject_type <- as.character(pheno_data$characteristics_ch1.3)
        # Both RBP+ and RBP- are SLE patients
        group <- ifelse(subject_type == "subject type: Control", "Control",
                       ifelse(grepl("subject type: Subject RBP", subject_type), "SLE", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using characteristics_ch1.3 column (GSE50635)")
          log_message(sprintf("  SLE=%d (RBP+ and RBP-), Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }

    # GSE110685-GPL16791: Use diagnosis column
    if (gse_id == "GSE110685-GPL16791") {
      if ("characteristics_ch1.1" %in% colnames(pheno_data)) {
        diagnosis <- as.character(pheno_data$characteristics_ch1.1)
        group <- ifelse(diagnosis == "diagnosis: Lupus", "SLE",
                       ifelse(diagnosis == "diagnosis: Healthy Control", "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using characteristics_ch1.1 column (GSE110685)")
          log_message(sprintf("  SLE=%d, Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }

    # GSE72509: Use disease status:ch1 with exact matching
    if (gse_id == "GSE72509") {
      if ("disease status:ch1" %in% colnames(pheno_data)) {
        status <- as.character(pheno_data$`disease status:ch1`)
        group <- ifelse(status == "systemic lupus erythematosus (SLE)", "SLE",
                       ifelse(status == "healthy", "Control", NA))
        n_sle <- sum(group == "SLE", na.rm = TRUE)
        n_control <- sum(group == "Control", na.rm = TRUE)
        if (n_sle > 0 && n_control > 0) {
          log_message("  Using disease status:ch1 column (GSE72509)")
          log_message(sprintf("  SLE=%d, Control=%d", n_sle, n_control))
          return(group)
        }
      }
    }
  }

  # ============================================================================
  # GENERIC EXTRACTION (fallback for other datasets)
  # ============================================================================

  # Priority columns to check
  priority_cols <- c(
    "disease status:ch1",     # Most common
    "patient status:ch1",     # GSE24706
    "disease:ch1",            # GSE17755
    "characteristics_ch1.2",  # Some datasets
    "characteristics_ch1.1",  # Some datasets
    "characteristics_ch1",    # Generic
    "title",                  # Last resort
    "case/control:ch1", "disease state:ch1", "diagnosis"
  )

  # Secondary columns
  other_cols <- c(
    "!Sample_characteristics_ch1", "source_name_ch1",
    "description", "characteristics_ch1.3",
    "characteristics_ch1.4", "characteristics_ch1.5"
  )

  all_cols <- c(priority_cols, other_cols)

  for (col in all_cols) {
    if (col %in% colnames(pheno_data)) {
      info <- as.character(pheno_data[[col]])

      # Define SLE patterns (more specific to avoid matching other diseases)
      sle_pattern <- paste0(
        "\\bSLE\\b|\\bSLE patient|SLE patients|disease: systemic lupus erythematosus|",
        "disease status: SLE|disease state: SLE|",
        "patient status: SLE|from SLE patient|",
        "systemic lupus erythematosus|diagnosis: Lupus|\\bLupus\\b"
      )

      # Define Control patterns
      control_pattern <- paste0(
        "disease status: Control|disease state: Healthy|disease state: control|",
        "patient status: healthy control|healthy control \\(HC\\)|\\bHC\\b(?! )|",
        "disease: healthy individual|disease: healthy child|",
        "diagnosis: Healthy Control|",
        "^Control$|\\bControl\\b(?! )|^HC_|^BC[0-9]+|",
        "healthy individual|healthy child|\\b[Hh]ealthy\\b|\\b[Nn]ormal\\b"
      )

      group <- rep(NA_character_, length(info))

      # Identify SLE samples first (more specific matching)
      sle_matches <- grepl(sle_pattern, info, ignore.case = FALSE, perl = TRUE)
      group[sle_matches] <- "SLE"

      # Identify Control samples (don't overwrite SLE)
      control_matches <- grepl(control_pattern, info, ignore.case = FALSE, perl = TRUE) & !sle_matches
      group[control_matches] <- "Control"

      # Check if we found anything useful
      n_sle <- sum(group == "SLE", na.rm = TRUE)
      n_control <- sum(group == "Control", na.rm = TRUE)
      n_unknown <- sum(is.na(group))

      # Only accept if we found both groups or a reasonable proportion
      if (n_sle > 0 && n_control > 0) {
        log_message(sprintf("  Using column: %s", col))
        log_message(sprintf("  SLE=%d, Control=%d, Unknown=%d",
                           n_sle, n_control, n_unknown))
        return(group)
      }
    }
  }

  # If nothing found with both groups, return all NA
  log_message("  WARNING: Could not identify sample groups", level = "WARNING")
  return(rep(NA_character_, nrow(pheno_data)))
}

#' Extract groups for all datasets
#'
#' @param gse_list List of GSE datasets
#' @return Updated gse_list with group information
extract_all_groups <- function(gse_list) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 2: Extracting Sample Groups\n")
  cat("================================================================================\n\n")

  summary_df <- data.frame()

  for (gse_id in names(gse_list)) {
    cat(sprintf("Processing %s...\n", gse_id))

    pheno_data <- gse_list[[gse_id]]$pheno
    # Pass gse_id to enable special handling for problematic datasets
    group <- extract_group_info(pheno_data, gse_id = gse_id)
    gse_list[[gse_id]]$group <- group

    # Summary statistics
    n_sle <- sum(group == "SLE", na.rm = TRUE)
    n_control <- sum(group == "Control", na.rm = TRUE)
    n_unknown <- sum(is.na(group))

    summary_df <- rbind(summary_df, data.frame(
      Dataset = gse_id,
      SLE = n_sle,
      Control = n_control,
      Unknown = n_unknown,
      Total = length(group),
      SLE_Ratio = ifelse(n_sle + n_control > 0,
                        round(n_sle / (n_sle + n_control), 3), NA)
    ))
  }

  cat("\n")
  cat("Group Summary:\n")
  print(summary_df, row.names = FALSE)

  # Identify problematic datasets
  problematic <- summary_df[
    summary_df$SLE == 0 | summary_df$Control == 0 |
    summary_df$Unknown > 0.5 * summary_df$Total,
  ]

  if (nrow(problematic) > 0) {
    cat("\n")
    log_message("WARNING: Some datasets need manual review:", level = "WARNING")
    print(problematic, row.names = FALSE)
  }

  # Overall statistics
  cat("\n")
  cat("Overall Statistics:\n")
  cat(sprintf("  Total SLE samples: %d\n", sum(summary_df$SLE)))
  cat(sprintf("  Total Control samples: %d\n", sum(summary_df$Control)))
  cat(sprintf("  Total Unknown samples: %d\n", sum(summary_df$Unknown)))

  # Save intermediate result
  if (CONFIG$save_intermediate) {
    save_object(gse_list, "02_grouped_data.rds", "Data with group labels")
    write.csv(summary_df,
             file.path(CONFIG$output_dir, "intermediate", "group_summary.csv"),
             row.names = FALSE)
  }

  return(gse_list)
}

# ==============================================================================
# 6. QUALITY CONTROL AND NORMALIZATION MODULE
# ==============================================================================

#' Perform QC and normalization on single dataset
#'
#' @param gse_id Dataset identifier
#' @param expr_matrix Expression matrix
#' @param group_info Group labels
#' @return Normalized expression matrix
qc_normalize_dataset <- function(gse_id, expr_matrix, group_info) {
  log_message(sprintf("QC and normalization for %s", gse_id))

  # 1. Handle missing values
  na_count <- sum(is.na(expr_matrix))
  if (na_count > 0) {
    log_message(sprintf("  Imputing %d missing values", na_count))
    expr_matrix <- t(apply(expr_matrix, 1, function(x) {
      x[is.na(x)] <- median(x, na.rm = TRUE)
      return(x)
    }))
  }

  # 2. Filter low-expression probes
  expr_threshold <- quantile(expr_matrix, CONFIG$min_expr_percentile, na.rm = TRUE)
  min_samples <- ceiling(ncol(expr_matrix) * CONFIG$min_expr_samples_pct)
  keep_probes <- rowSums(expr_matrix > expr_threshold, na.rm = TRUE) >= min_samples

  n_filtered <- sum(!keep_probes)
  expr_matrix <- expr_matrix[keep_probes, ]
  log_message(sprintf("  Filtered %d low-expression probes, %d remaining",
                     n_filtered, nrow(expr_matrix)))

  # 3. Log transformation if needed
  data_range <- range(expr_matrix, na.rm = TRUE)
  if (data_range[2] > 100) {
    log_message("  Applying log2 transformation")
    expr_matrix <- log2(expr_matrix + 1)
  }

  # 4. Quantile normalization
  log_message("  Performing quantile normalization")
  expr_matrix <- normalizeBetweenArrays(expr_matrix, method = "quantile")

  # 5. Generate QC plots
  if (CONFIG$create_qc_plots) {
    create_qc_plots(gse_id, expr_matrix, group_info)
  }

  return(expr_matrix)
}

#' Create QC plots for a dataset
#'
#' @param gse_id Dataset identifier
#' @param expr_matrix Expression matrix
#' @param group_info Group labels
create_qc_plots <- function(gse_id, expr_matrix, group_info) {
  pdf_file <- file.path(CONFIG$output_dir, "QC", sprintf("%s_QC.pdf", gse_id))
  cairo_pdf(pdf_file, width = 12, height = 8, family = "Arial")

  tryCatch({
    par(mfrow = c(2, 2), mar = c(8, 4, 3, 2))

    # Boxplot
    boxplot(expr_matrix, las = 2,
           main = paste(gse_id, "- Expression Distribution"),
           col = rainbow(min(ncol(expr_matrix), 50)),
           cex.axis = 0.6, outline = FALSE)

    # Density plot
    plot(density(expr_matrix[, 1]),
        main = paste(gse_id, "- Density Plot"),
        xlim = range(expr_matrix, na.rm = TRUE),
        ylim = c(0, max(density(expr_matrix[, 1])$y)))
    for (i in 2:min(ncol(expr_matrix), 20)) {
      lines(density(expr_matrix[, i]), col = i)
    }

    # PCA
    tryCatch({
      gene_var <- apply(expr_matrix, 1, var, na.rm = TRUE)
      valid_genes <- !is.na(gene_var) & gene_var > 1e-10

      if (sum(valid_genes) >= 10) {
        pca_result <- prcomp(t(expr_matrix[valid_genes, ]), scale. = TRUE)
        plot(pca_result$x[, 1:2],
            main = paste(gse_id, "- PCA"),
            xlab = sprintf("PC1 (%.1f%%)",
                          summary(pca_result)$importance[2, 1] * 100),
            ylab = sprintf("PC2 (%.1f%%)",
                          summary(pca_result)$importance[2, 2] * 100),
            col = ifelse(group_info == "SLE", "red", "blue"),
            pch = 19)
        legend("topright", legend = c("SLE", "Control"),
              col = c("red", "blue"), pch = 19)
      } else {
        plot.new()
        text(0.5, 0.5, "PCA not available\nInsufficient variance",
            cex = 1.5, col = "red")
      }
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "PCA Error", cex = 1.5, col = "red")
    })

    # Correlation heatmap
    tryCatch({
      if (ncol(expr_matrix) <= 100) {
        cor_matrix <- cor(expr_matrix, method = "pearson",
                         use = "pairwise.complete.obs")
        heatmap(cor_matrix, main = paste(gse_id, "- Sample Correlation"))
      } else {
        plot.new()
        text(0.5, 0.5, "Too many samples\nfor correlation heatmap",
            cex = 1.5)
      }
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, "Correlation Error", cex = 1.5, col = "red")
    })

  }, error = function(e) {
    log_message(sprintf("Error creating QC plots: %s", e$message),
               level = "ERROR")
  })

  dev.off()
  log_message(sprintf("  QC plots saved: %s", pdf_file))
}

#' QC and normalize all datasets
#'
#' @param gse_list List of GSE datasets
#' @return Updated gse_list with normalized data
qc_normalize_all <- function(gse_list) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 3: Quality Control and Normalization\n")
  cat("================================================================================\n\n")

  for (gse_id in names(gse_list)) {
    cat(sprintf("Processing %s...\n", gse_id))

    expr_matrix <- gse_list[[gse_id]]$expr
    group_info <- gse_list[[gse_id]]$group

    tryCatch({
      expr_normalized <- qc_normalize_dataset(gse_id, expr_matrix, group_info)
      gse_list[[gse_id]]$expr_normalized <- expr_normalized
    }, error = function(e) {
      log_message(sprintf("Failed to process %s: %s", gse_id, e$message),
                 level = "ERROR")
      gse_list[[gse_id]]$expr_normalized <- NULL
    })
  }

  # Save intermediate result
  if (CONFIG$save_intermediate) {
    save_object(gse_list, "03_normalized_data.rds", "QC and normalized data")
  }

  return(gse_list)
}

# ==============================================================================
# 7. PROBE TO GENE CONVERSION MODULE
# ==============================================================================

#' Convert Ensembl IDs or Entrez IDs to gene symbols
#'
#' @param expr_matrix Expression matrix with Ensembl/Entrez IDs as row names
#' @param id_type Type of IDs ("ensembl" or "entrez")
#' @return Expression matrix with gene symbols as row names
convert_rnaseq_ids_to_gene <- function(expr_matrix, id_type = "auto") {
  log_message("  Converting RNA-seq gene IDs to symbols...")

  # Auto-detect ID type
  if (id_type == "auto") {
    sample_ids <- head(rownames(expr_matrix), 100)
    if (any(grepl("^ENSG[0-9]+", sample_ids))) {
      id_type <- "ensembl"
    } else if (any(grepl("^[0-9]+$", sample_ids))) {
      id_type <- "entrez"
    } else {
      log_message("  Assuming gene IDs are already symbols", level = "WARNING")
      return(expr_matrix)
    }
  }

  log_message(sprintf("  Detected ID type: %s", id_type))

  tryCatch({
    if (id_type == "ensembl") {
      # Remove version numbers from Ensembl IDs (e.g., ENSG00000223972.5 -> ENSG00000223972)
      clean_ids <- gsub("\\..*$", "", rownames(expr_matrix))

      genes <- AnnotationDbi::select(
        org.Hs.eg.db,
        keys = clean_ids,
        columns = c("SYMBOL", "ENTREZID"),
        keytype = "ENSEMBL"
      )

    } else if (id_type == "entrez") {
      genes <- AnnotationDbi::select(
        org.Hs.eg.db,
        keys = rownames(expr_matrix),
        columns = c("SYMBOL", "ENTREZID"),
        keytype = "ENTREZID"
      )
    } else {
      stop(sprintf("Unknown ID type: %s", id_type))
    }

    # Filter out genes without symbols
    if (id_type == "ensembl") {
      genes$PROBEID <- genes$ENSEMBL
    } else {
      genes$PROBEID <- genes$ENTREZID
    }

    genes <- genes[!is.na(genes$SYMBOL) &
                  genes$SYMBOL != "" &
                  genes$SYMBOL != "---", ]

    if (nrow(genes) == 0) {
      log_message("  WARNING: No valid gene symbols found", level = "WARNING")
      return(expr_matrix)
    }

    # Create mapping from original IDs to clean IDs
    if (id_type == "ensembl") {
      id_mapping <- data.frame(
        original_id = rownames(expr_matrix),
        clean_id = gsub("\\..*$", "", rownames(expr_matrix)),
        stringsAsFactors = FALSE
      )
      # Match genes to original IDs
      matched_idx <- match(genes$PROBEID, id_mapping$clean_id)
      genes$original_id <- id_mapping$original_id[matched_idx]
      genes <- genes[!is.na(genes$original_id), ]

      # Map to expression matrix using original IDs
      expr_matrix <- expr_matrix[genes$original_id, ]
    } else {
      expr_matrix <- expr_matrix[genes$PROBEID, ]
    }

    rownames(expr_matrix) <- genes$SYMBOL

    # Aggregate multiple IDs per gene (take mean)
    expr_gene <- aggregate(expr_matrix,
                          by = list(rownames(expr_matrix)),
                          FUN = mean)
    rownames(expr_gene) <- expr_gene$Group.1
    expr_gene$Group.1 <- NULL

    log_message(sprintf("  Converted to %d unique genes", nrow(expr_gene)))
    return(as.matrix(expr_gene))

  }, error = function(e) {
    log_message(sprintf("  ERROR: ID conversion failed - %s", e$message),
               level = "ERROR")
    return(expr_matrix)
  })
}

#' Convert probe IDs to gene symbols
#'
#' @param expr_matrix Expression matrix with probe IDs as row names
#' @param gpl_id GPL platform identifier
#' @param gse_id Dataset identifier (for finding local GPL files)
#' @return Expression matrix with gene symbols as row names
convert_probe_to_gene <- function(expr_matrix, gpl_id, gse_id) {
  log_message(sprintf("Converting probes to genes for %s (Platform: %s)",
                     gse_id, gpl_id))

  tryCatch({
    genes <- NULL

    # Try Bioconductor annotation packages first
    if (gpl_id == "GPL570") {
      if (require("hgu133plus2.db", quietly = TRUE)) {
        genes <- AnnotationDbi::select(
          hgu133plus2.db,
          keys = rownames(expr_matrix),
          columns = c("SYMBOL", "ENTREZID"),
          keytype = "PROBEID"
        )
      }
    } else if (gpl_id == "GPL96") {
      if (require("hgu133a.db", quietly = TRUE)) {
        genes <- AnnotationDbi::select(
          hgu133a.db,
          keys = rownames(expr_matrix),
          columns = c("SYMBOL", "ENTREZID"),
          keytype = "PROBEID"
        )
      }
    }

    # If Bioconductor packages not available, use local GPL files
    if (is.null(genes)) {
      gpl_dir <- file.path(CONFIG$data_dir, gpl_id)

      # Find local GPL annotation files
      gpl_files <- list.files(gpl_dir, full.names = TRUE, pattern = gpl_id)
      gpl_files <- gpl_files[grepl("\\.(txt|soft|soft\\.gz)$", gpl_files,
                                  ignore.case = TRUE)]

      if (length(gpl_files) > 0) {
        log_message(sprintf("  Using local GPL file: %s", basename(gpl_files[1])))

        # Read GPL file
        if (grepl("\\.soft(\\.gz)?$", gpl_files[1], ignore.case = TRUE)) {
          gpl <- getGEO(filename = gpl_files[1])
          gpl_table <- Table(gpl)
        } else {
          gpl_table <- read.delim(gpl_files[1], comment.char = "#",
                                 stringsAsFactors = FALSE)
        }

        # Find symbol column
        symbol_cols <- c("Symbol", "Gene Symbol", "Gene.Symbol",
                        "GENE_SYMBOL", "gene_symbol", "GeneSymbol",
                        "gene_assignment")

        symbol_col <- NULL
        for (col in symbol_cols) {
          if (col %in% colnames(gpl_table)) {
            symbol_col <- col
            break
          }
        }

        if (!is.null(symbol_col)) {
          probe_match <- match(rownames(expr_matrix), gpl_table$ID)
          symbols <- gpl_table[[symbol_col]][probe_match]

          # Parse gene_assignment format if needed
          if (symbol_col == "gene_assignment") {
            symbols <- sapply(strsplit(as.character(symbols), " // "),
                            function(x) if (length(x) >= 2) x[2] else NA)
          }

          genes <- data.frame(
            PROBEID = rownames(expr_matrix),
            SYMBOL = symbols,
            ENTREZID = NA,
            stringsAsFactors = FALSE
          )
        }
      } else {
        log_message("  No local GPL file found, downloading...", level = "WARNING")
        gpl <- getGEO(gpl_id, destdir = ".")
        gpl_table <- Table(gpl)

        # Similar processing as above...
      }
    }

    # Filter out probes without gene symbols
    if (!is.null(genes)) {
      genes <- genes[!is.na(genes$SYMBOL) &
                    genes$SYMBOL != "" &
                    genes$SYMBOL != "---", ]

      if (nrow(genes) == 0) {
        log_message("  WARNING: No valid gene symbols found", level = "WARNING")
        return(expr_matrix)
      }

      # Map to expression matrix
      expr_matrix <- expr_matrix[genes$PROBEID, ]
      rownames(expr_matrix) <- genes$SYMBOL

      # Aggregate multiple probes per gene (take mean)
      expr_gene <- aggregate(expr_matrix,
                            by = list(rownames(expr_matrix)),
                            FUN = mean)
      rownames(expr_gene) <- expr_gene$Group.1
      expr_gene$Group.1 <- NULL

      log_message(sprintf("  Converted to %d unique genes", nrow(expr_gene)))
      return(as.matrix(expr_gene))
    } else {
      log_message("  WARNING: Could not convert probes", level = "WARNING")
      return(expr_matrix)
    }

  }, error = function(e) {
    log_message(sprintf("  ERROR: Conversion failed - %s", e$message),
               level = "ERROR")
    return(expr_matrix)
  })
}

#' Convert all datasets
#'
#' @param gse_list List of GSE datasets
#' @return Updated gse_list with gene expression matrices
convert_all_probes <- function(gse_list) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 4: Converting Probe IDs to Gene Symbols\n")
  cat("================================================================================\n\n")

  for (gse_id in names(gse_list)) {
    if (is.null(gse_list[[gse_id]]$expr_normalized)) {
      log_message(sprintf("Skipping %s (no normalized data)", gse_id),
                 level = "WARNING")
      next
    }

    cat(sprintf("Processing %s...\n", gse_id))

    expr_matrix <- gse_list[[gse_id]]$expr_normalized
    gpl_id <- gse_list[[gse_id]]$gpl
    data_type <- gse_list[[gse_id]]$data_type

    # Choose conversion method based on data type
    if (data_type %in% c("supplementary_counts", "supplementary_fpkm", "supplementary_xlsx")) {
      # RNA-seq data - convert Ensembl/Entrez IDs to symbols
      expr_gene <- convert_rnaseq_ids_to_gene(expr_matrix, id_type = "auto")
    } else {
      # Microarray data - convert probe IDs to symbols
      expr_gene <- convert_probe_to_gene(expr_matrix, gpl_id, gse_id)
    }

    gse_list[[gse_id]]$expr_gene <- expr_gene
  }

  # Save intermediate result
  if (CONFIG$save_intermediate) {
    save_object(gse_list, "04_gene_expr.rds",
               "Expression data with gene symbols")
  }

  return(gse_list)
}

# ==============================================================================
# 8. DATA MERGING AND BATCH CORRECTION MODULE
# ==============================================================================

#' Merge datasets and remove batch effects
#'
#' @param gse_list List of GSE datasets
#' @return List containing merged expression matrix and metadata
merge_and_batch_correct <- function(gse_list) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 5: Merging Datasets and Batch Correction\n")
  cat("================================================================================\n\n")

  # Extract gene expression matrices
  expr_list <- lapply(gse_list, function(x) x$expr_gene)
  expr_list <- expr_list[!sapply(expr_list, is.null)]

  if (length(expr_list) == 0) {
    stop("No valid expression matrices found")
  }

  # Find common genes
  common_genes <- Reduce(intersect, lapply(expr_list, rownames))
  log_message(sprintf("Found %d common genes across all datasets",
                     length(common_genes)))

  if (length(common_genes) < 100) {
    log_message("WARNING: Very few common genes found", level = "WARNING")
  }

  # Merge with unique sample names
  log_message("Merging datasets with unique sample identifiers...")
  merged_list <- list()
  sample_mapping <- list()

  for (gse_id in names(expr_list)) {
    mat <- expr_list[[gse_id]][common_genes, ]
    original_names <- colnames(mat)

    # Add dataset prefix to ensure uniqueness
    new_names <- paste0(gse_id, "_", original_names)
    colnames(mat) <- new_names

    # Save mapping
    sample_mapping[[gse_id]] <- data.frame(
      original_id = original_names,
      unique_id = new_names,
      dataset = gse_id,
      stringsAsFactors = FALSE
    )

    merged_list[[gse_id]] <- mat
    log_message(sprintf("  %s: %d samples", gse_id, ncol(mat)))
  }

  # Combine all matrices
  merged_expr <- do.call(cbind, merged_list)
  log_message(sprintf("Merged matrix: %d genes x %d samples",
                     nrow(merged_expr), ncol(merged_expr)))

  # Verify column name uniqueness
  if (any(duplicated(colnames(merged_expr)))) {
    log_message("ERROR: Duplicate column names detected!", level = "ERROR")
    stop("Column name duplication error")
  } else {
    log_message("Column name uniqueness verified")
  }

  # Create batch and group information
  batch <- rep(names(expr_list), sapply(expr_list, ncol))
  group_info <- unlist(lapply(names(expr_list),
                             function(x) gse_list[[x]]$group))

  # Remove samples with unknown group
  known_samples <- !is.na(group_info)
  merged_expr <- merged_expr[, known_samples]
  batch <- batch[known_samples]
  group_info <- group_info[known_samples]

  log_message(sprintf("After removing unknown samples: %d samples (SLE=%d, Control=%d)",
                     ncol(merged_expr),
                     sum(group_info == "SLE"),
                     sum(group_info == "Control")))

  # Batch diagnosis
  cat("\n")
  cat("Batch-Group Distribution:\n")
  batch_table <- table(batch, group_info)
  print(batch_table)

  # Check for problematic batches
  for (b in rownames(batch_table)) {
    sle_n <- batch_table[b, "SLE"]
    ctrl_n <- batch_table[b, "Control"]
    if (sle_n == 0 || ctrl_n == 0) {
      log_message(sprintf("WARNING: Batch %s has only one group type", b),
                 level = "WARNING")
    }
  }

  # Remove small batches
  batch_counts <- table(batch)
  small_batches <- names(batch_counts)[batch_counts < CONFIG$min_samples_per_batch]

  if (length(small_batches) > 0) {
    log_message(sprintf("Removing %d batches with < %d samples",
                       length(small_batches), CONFIG$min_samples_per_batch))

    keep_samples <- !batch %in% small_batches
    merged_expr <- merged_expr[, keep_samples]
    batch <- batch[keep_samples]
    group_info <- group_info[keep_samples]
  }

  # Quality checks before ComBat
  log_message("Performing data quality checks...")

  # Remove NA values
  if (any(is.na(merged_expr))) {
    log_message("Imputing NA values")
    merged_expr[is.na(merged_expr)] <- 0
  }

  # Remove zero-variance genes
  gene_var <- apply(merged_expr, 1, var, na.rm = TRUE)
  valid_genes <- !is.na(gene_var) & gene_var > 1e-10

  if (sum(!valid_genes) > 0) {
    log_message(sprintf("Removing %d zero-variance genes", sum(!valid_genes)))
    merged_expr <- merged_expr[valid_genes, ]
  }

  # Batch correction with ComBat
  combat_expr <- merged_expr

  if (CONFIG$use_combat &&
     length(unique(batch)) >= 2 &&
     ncol(merged_expr) >= 10 &&
     nrow(merged_expr) >= 100) {

    cat("\n")
    log_message("Applying ComBat batch correction...")

    tryCatch({
      mod <- model.matrix(~as.factor(group_info))
      combat_expr <- ComBat(dat = merged_expr,
                           batch = batch,
                           mod = mod,
                           par.prior = TRUE)
      log_message("ComBat correction completed successfully")
    }, error = function(e) {
      log_message(sprintf("ComBat failed: %s. Using uncorrected data.", e$message),
                 level = "WARNING")
      combat_expr <<- merged_expr
    })
  } else {
    log_message("Skipping ComBat (insufficient data or disabled)", level = "WARNING")
  }

  # Create PCA comparison plot
  if (CONFIG$create_qc_plots) {
    create_batch_correction_plot(merged_expr, combat_expr, batch, group_info)
  }

  # Prepare output
  merged_data <- list(
    expr_before = merged_expr,
    expr_after = combat_expr,
    group = group_info,
    batch = batch,
    genes = rownames(combat_expr),
    sample_mapping = do.call(rbind, sample_mapping)
  )

  # Save results
  if (CONFIG$save_intermediate) {
    save_object(merged_data, "05_merged_data.rds", "Merged and batch-corrected data")
  }

  return(merged_data)
}

#' Create batch correction comparison plot
#'
#' @param expr_before Expression before correction
#' @param expr_after Expression after correction
#' @param batch Batch labels
#' @param group Group labels
create_batch_correction_plot <- function(expr_before, expr_after, batch, group) {
  pdf_file <- file.path(CONFIG$output_dir, "QC", "Batch_Correction_PCA.pdf")
  cairo_pdf(pdf_file, width = 14, height = 6, family = "Arial")

  tryCatch({
    par(mfrow = c(1, 2))

    # Remove zero-variance genes for PCA
    gene_var <- apply(expr_before, 1, var, na.rm = TRUE)
    valid_genes <- !is.na(gene_var) & gene_var > 1e-10

    if (sum(valid_genes) < 10) {
      plot.new()
      text(0.5, 0.5, "Insufficient variance for PCA", cex = 1.5, col = "red")
    } else {
      # Prepare colors for batches and shapes for groups
      batch_levels <- unique(batch)
      batch_colors <- rainbow(length(batch_levels))
      names(batch_colors) <- batch_levels
      point_colors <- batch_colors[as.character(batch)]
      point_shapes <- ifelse(group == "SLE", 19, 1)

      # Before correction
      pca_before <- prcomp(t(expr_before[valid_genes, ]), scale. = TRUE)
      plot(pca_before$x[, 1:2],
          col = point_colors,
          pch = point_shapes,
          main = "Before Batch Correction",
          xlab = sprintf("PC1 (%.1f%%)",
                        summary(pca_before)$importance[2, 1] * 100),
          ylab = sprintf("PC2 (%.1f%%)",
                        summary(pca_before)$importance[2, 2] * 100),
          cex = 1.2)

      # Create combined legend showing both datasets and groups
      legend("topright",
             legend = c("Disease Status:", "  SLE", "  Control",
                       "", "Datasets:", paste0("  ", batch_levels)),
             pch = c(NA, 19, 1, NA, NA, rep(15, length(batch_levels))),
             col = c(NA, "black", "black", NA, NA, batch_colors),
             bty = "n", cex = 0.8)

      # After correction
      pca_after <- prcomp(t(expr_after[valid_genes, ]), scale. = TRUE)
      plot(pca_after$x[, 1:2],
          col = ifelse(group == "SLE", "red", "blue"),
          pch = 19,
          main = "After Batch Correction",
          xlab = sprintf("PC1 (%.1f%%)",
                        summary(pca_after)$importance[2, 1] * 100),
          ylab = sprintf("PC2 (%.1f%%)",
                        summary(pca_after)$importance[2, 2] * 100),
          cex = 1.2)
      legend("topright", legend = c("SLE", "Control"),
            col = c("red", "blue"), pch = 19, bty = "n", cex = 0.8)
    }

  }, error = function(e) {
    log_message(sprintf("Error creating batch correction plot: %s", e$message),
               level = "ERROR")
  })

  dev.off()
  log_message(sprintf("Batch correction plot saved: %s", pdf_file))
}

# ==============================================================================
# 9. DIFFERENTIAL EXPRESSION ANALYSIS MODULE
# ==============================================================================

#' Perform differential expression analysis
#'
#' @param merged_data List containing merged expression data
#' @return List with DEG results and statistics
perform_deg_analysis <- function(merged_data) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 6: Differential Expression Analysis\n")
  cat("================================================================================\n\n")

  expr_matrix <- merged_data$expr_after
  group_info <- merged_data$group

  # Build design matrix
  log_message("Building design matrix...")
  design <- model.matrix(~0 + factor(group_info))
  colnames(design) <- c("Control", "SLE")

  # Contrast matrix
  contrast.matrix <- makeContrasts(
    SLE_vs_Control = SLE - Control,
    levels = design
  )

  # Fit linear model
  log_message("Fitting linear model...")
  fit <- lmFit(expr_matrix, design)
  fit2 <- contrasts.fit(fit, contrast.matrix)
  fit2 <- eBayes(fit2)

  # Extract results
  log_message("Extracting differential expression results...")
  deg_results <- topTable(fit2, coef = "SLE_vs_Control",
                         number = Inf, adjust.method = "BH")

  # Diagnostic: logFC distribution
  cat("\n")
  cat("logFC Distribution Diagnostics:\n")
  cat(sprintf("  Range: [%.3f, %.3f]\n",
             min(deg_results$logFC), max(deg_results$logFC)))
  cat(sprintf("  Mean: %.3f\n", mean(deg_results$logFC)))
  cat(sprintf("  Median: %.3f\n", median(deg_results$logFC)))

  # Test multiple thresholds
  cat("\n")
  cat("DEG counts at different thresholds:\n")
  thresholds <- c(0.5, 1.0, 1.5, 2.0)
  for (thresh in thresholds) {
    n_up <- sum(deg_results$logFC > thresh &
               deg_results$adj.P.Val < CONFIG$adj_p_threshold)
    n_down <- sum(deg_results$logFC < -thresh &
                 deg_results$adj.P.Val < CONFIG$adj_p_threshold)
    cat(sprintf("  |logFC| > %.1f: Up=%d, Down=%d, Ratio=%.2f:1\n",
               thresh, n_up, n_down,
               ifelse(n_down > 0, n_up/n_down, Inf)))
  }

  # Add change labels
  deg_results$Change <- ifelse(
    deg_results$adj.P.Val < CONFIG$adj_p_threshold,
    ifelse(deg_results$logFC > CONFIG$log_fc_threshold, "Up",
          ifelse(deg_results$logFC < -CONFIG$log_fc_threshold, "Down", "NS")),
    "NS"
  )

  # Final statistics
  n_up <- sum(deg_results$Change == "Up")
  n_down <- sum(deg_results$Change == "Down")
  n_total <- n_up + n_down

  cat("\n")
  cat(sprintf("Final DEG Statistics (|logFC| > %.1f, adj.P < %.2f):\n",
             CONFIG$log_fc_threshold, CONFIG$adj_p_threshold))
  cat(sprintf("  Upregulated: %d\n", n_up))
  cat(sprintf("  Downregulated: %d\n", n_down))
  cat(sprintf("  Total DEGs: %d\n", n_total))

  # Check for imbalance
  if (n_up == 0 && n_down > 0) {
    log_message("WARNING: Only downregulated genes found!", level = "WARNING")
    log_message("  Possible issues: sample labels reversed, over-correction",
               level = "WARNING")
  } else if (n_down == 0 && n_up > 0) {
    log_message("WARNING: Only upregulated genes found!", level = "WARNING")
    log_message("  Possible issues: sample labels reversed, over-correction",
               level = "WARNING")
  } else if (n_up > 0 && n_down > 0) {
    ratio <- max(n_up, n_down) / min(n_up, n_down)
    if (ratio > 10) {
      log_message(sprintf("WARNING: Severe imbalance (%.1f:1)", ratio),
                 level = "WARNING")
    } else {
      log_message(sprintf("DEG balance looks good (%.1f:1)", ratio))
    }
  }

  # Save results
  out_dir <- file.path(CONFIG$output_dir, "DEGs")
  write.csv(deg_results, file.path(out_dir, "DEG_results_all.csv"),
           row.names = TRUE)

  deg_sig <- deg_results[deg_results$Change != "NS", ]
  write.csv(deg_sig, file.path(out_dir, "DEG_results_significant.csv"),
           row.names = TRUE)

  # Save at different thresholds
  for (thresh in c(0.5, 1.0)) {
    deg_thresh <- deg_results[
      abs(deg_results$logFC) > thresh &
      deg_results$adj.P.Val < CONFIG$adj_p_threshold,
    ]
    write.csv(deg_thresh,
             file.path(out_dir, sprintf("DEG_results_logFC%.1f.csv", thresh)),
             row.names = TRUE)
  }

  # Create visualizations
  if (CONFIG$create_qc_plots) {
    create_deg_plots(deg_results, expr_matrix, group_info)
  }

  return(list(
    all_results = deg_results,
    significant = deg_sig,
    n_up = n_up,
    n_down = n_down,
    n_total = n_total
  ))
}

#' Create DEG visualization plots
#'
#' @param deg_results DEG results table
#' @param expr_matrix Expression matrix
#' @param group_info Group labels
create_deg_plots <- function(deg_results, expr_matrix, group_info) {
  out_dir <- file.path(CONFIG$output_dir, "DEGs")

  # Load imidazoline target genes
  imidazoline_targets <- tryCatch({
    csv_path <- "~/statics/imidazoline_SLE_intersection.csv"
    targets_df <- read.csv(csv_path, stringsAsFactors = FALSE)
    targets_df$Target
  }, error = function(e) {
    log_message("Could not load imidazoline targets for volcano plot", level = "WARNING")
    character(0)
  })

  # 1. Volcano plot with all DEG labels and imidazoline target highlighting
  cairo_pdf(file.path(out_dir, "Volcano_Plot.pdf"), width = 14, height = 12, family = "Arial")

  # Prepare data for plotting
  deg_results$gene_name <- rownames(deg_results)

  # Identify all DEGs and imidazoline targets
  deg_results$is_target <- deg_results$gene_name %in% imidazoline_targets
  deg_results$point_type <- "NS"
  deg_results$point_type[deg_results$Change == "Up"] <- "Up-regulated"
  deg_results$point_type[deg_results$Change == "Down"] <- "Down-regulated"
  deg_results$point_type[deg_results$is_target & deg_results$Change == "Up"] <- "Target (Up)"
  deg_results$point_type[deg_results$is_target & deg_results$Change == "Down"] <- "Target (Down)"

  # Genes to label: all DEGs
  genes_to_label <- deg_results[deg_results$Change != "NS", ]

  # Create plot with enhanced labeling for targets
  p <- ggplot(deg_results, aes(x = logFC, y = -log10(adj.P.Val))) +
    geom_point(aes(color = point_type, size = is_target, alpha = is_target)) +
    scale_color_manual(values = c("Up-regulated" = "red",
                                   "Down-regulated" = "blue",
                                   "Target (Up)" = "darkred",
                                   "Target (Down)" = "darkblue",
                                   "NS" = "grey"),
                      breaks = c("Target (Up)", "Target (Down)",
                                "Up-regulated", "Down-regulated", "NS")) +
    scale_size_manual(values = c("TRUE" = 3, "FALSE" = 1.5), guide = "none") +
    scale_alpha_manual(values = c("TRUE" = 0.9, "FALSE" = 0.4), guide = "none") +
    geom_hline(yintercept = -log10(CONFIG$adj_p_threshold),
              linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_vline(xintercept = c(-CONFIG$log_fc_threshold, CONFIG$log_fc_threshold),
              linetype = "dashed", color = "black", linewidth = 0.5)

  # Add labels using ggrepel for all DEGs
  if (nrow(genes_to_label) > 0) {
    p <- p + ggrepel::geom_text_repel(
      data = genes_to_label,
      aes(label = gene_name,
          color = point_type,
          fontface = ifelse(is_target, "bold", "plain")),
      size = ifelse(genes_to_label$is_target, 3.5, 2.5),
      max.overlaps = Inf,
      box.padding = 0.5,
      point.padding = 0.3,
      segment.color = "grey50",
      segment.size = 0.2,
      show.legend = FALSE
    )
  }

  p <- p +
    labs(title = sprintf("Volcano Plot: SLE vs Control (n=%d DEGs, %d Imidazoline Targets)",
                        nrow(genes_to_label), sum(genes_to_label$is_target)),
        x = "log2(Fold Change)",
        y = "-log10(adjusted P-value)",
        color = "Gene Type") +
    theme_bw() +
    theme(legend.position = "right",
          legend.text = element_text(size = 10),
          legend.title = element_text(size = 11, face = "bold"))

  print(p)
  dev.off()

  log_message(sprintf("Volcano plot created: %d DEGs labeled (%d imidazoline targets)",
                     nrow(genes_to_label), sum(genes_to_label$is_target)))

  # 2. logFC distribution histogram
  cairo_pdf(file.path(out_dir, "logFC_Distribution.pdf"), width = 10, height = 6, family = "Arial")
  hist(deg_results$logFC, breaks = 50,
      main = "log2(Fold Change) Distribution",
      xlab = "log2(Fold Change)",
      col = "lightblue",
      border = "white")
  abline(v = 0, col = "red", lwd = 2, lty = 2)
  abline(v = c(-CONFIG$log_fc_threshold, CONFIG$log_fc_threshold),
        col = "blue", lwd = 2, lty = 2)
  dev.off()

  # 3. Top DEG heatmap with disease group clustering
  cairo_pdf(file.path(out_dir, "Heatmap_Top50_DEGs.pdf"), width = 12, height = 10, family = "Arial")
  top_genes <- rownames(deg_results)[1:min(50, nrow(deg_results))]

  if (all(top_genes %in% rownames(expr_matrix))) {
    tryCatch({
      heatmap_data <- expr_matrix[top_genes, ]

      # Remove genes with zero variance or non-finite values
      gene_var <- apply(heatmap_data, 1, var, na.rm = TRUE)
      valid_genes <- is.finite(gene_var) & gene_var > 0 &
                     !apply(heatmap_data, 1, function(x) any(!is.finite(x)))

      if (sum(valid_genes) == 0) {
        plot.new()
        text(0.5, 0.5, "No valid genes for heatmap\n(all have zero variance or non-finite values)",
             cex = 1.5, col = "red")
      } else {
        heatmap_data <- heatmap_data[valid_genes, , drop = FALSE]

        # CRITICAL: Sort columns by group and add gap between groups
        group_order <- order(group_info)
        heatmap_data <- heatmap_data[, group_order]
        group_info_sorted <- group_info[group_order]

        # Calculate gap position (between Control and SLE)
        n_control <- sum(group_info_sorted == "Control")
        gaps_col <- n_control  # Add gap after Control samples

        annotation_col <- data.frame(
          Group = group_info_sorted,
          row.names = colnames(heatmap_data)
        )

        # Define colors for groups
        ann_colors <- list(
          Group = c(Control = "#4DBBD5", SLE = "#E64B35")
        )

        pheatmap(heatmap_data,
                scale = "row",
                clustering_distance_rows = "euclidean",
                clustering_distance_cols = "euclidean",
                clustering_method = "ward.D2",
                annotation_col = annotation_col,
                annotation_colors = ann_colors,
                show_colnames = FALSE,
                cluster_cols = FALSE,  # No clustering - strict Control vs SLE grouping
                gaps_col = gaps_col,  # Add gap between Control and SLE
                color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
                main = sprintf("Top DEGs (%d genes)\nGrouped by Disease Status", sum(valid_genes)))
      }
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, paste("Error creating DEG heatmap:", e$message),
          cex = 1.5, col = "red")
    })
  } else {
    plot.new()
    text(0.5, 0.5, "Cannot create heatmap\n(gene names mismatch)",
        cex = 1.5, col = "red")
  }
  dev.off()

  log_message("DEG visualization plots created")
}

# ==============================================================================
# 10. FUNCTIONAL ENRICHMENT ANALYSIS MODULE
# ==============================================================================

#' Perform comprehensive enrichment analysis
#'
#' @param deg_results DEG results list
#' @param expr_matrix Expression matrix for GSEA/GSVA
#' @return List with all enrichment results
perform_enrichment_analysis <- function(deg_results, expr_matrix) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 7: Functional Enrichment Analysis\n")
  cat("================================================================================\n\n")

  deg_sig <- deg_results$significant
  deg_all <- deg_results$all_results

  results <- list()

  # Check if we have enough DEGs
  if (nrow(deg_sig) < 10) {
    log_message("WARNING: Too few significant DEGs for enrichment analysis",
               level = "WARNING")
    log_message("Trying relaxed threshold (logFC > 0.5)...", level = "WARNING")

    deg_relaxed <- deg_all[
      abs(deg_all$logFC) > 0.5 &
      deg_all$adj.P.Val < CONFIG$adj_p_threshold,
    ]

    if (nrow(deg_relaxed) >= 10) {
      deg_genes <- rownames(deg_relaxed)
      log_message(sprintf("Using %d genes with relaxed threshold", length(deg_genes)))
    } else {
      log_message("Still too few genes, skipping enrichment", level = "ERROR")
      return(NULL)
    }
  } else {
    deg_genes <- rownames(deg_sig)
  }

  # Convert to Entrez IDs
  log_message("Converting gene symbols to Entrez IDs...")
  gene_entrez <- bitr(deg_genes, fromType = "SYMBOL", toType = "ENTREZID",
                     OrgDb = org.Hs.eg.db)

  if (nrow(gene_entrez) < 5) {
    log_message("ERROR: Too few genes with Entrez IDs", level = "ERROR")
    return(NULL)
  }

  log_message(sprintf("Converted %d/%d genes to Entrez IDs",
                     nrow(gene_entrez), length(deg_genes)))

  # GO Enrichment
  log_message("Running GO enrichment analysis...")
  results$GO <- perform_go_enrichment(gene_entrez$ENTREZID)

  # KEGG Enrichment
  log_message("Running KEGG pathway enrichment...")
  results$KEGG <- perform_kegg_enrichment(gene_entrez$ENTREZID)

  # GSEA
  log_message("Running Gene Set Enrichment Analysis (GSEA)...")
  results$GSEA <- perform_gsea(deg_all, gene_entrez)

  # GSVA (if applicable)
  if (!is.null(expr_matrix) && file.exists("h.all.v2025.1.Hs.symbols.gmt")) {
    log_message("Running Gene Set Variation Analysis (GSVA)...")
    # GSVA analysis would go here
    # Skipped for now to keep code manageable
  }

  return(results)
}

#' GO enrichment analysis
perform_go_enrichment <- function(entrez_ids) {
  out_dir <- file.path(CONFIG$output_dir, "Enrichment")

  go_results <- list()

  for (ont in c("BP", "CC", "MF")) {
    tryCatch({
      go_enrich <- enrichGO(
        gene = entrez_ids,
        OrgDb = org.Hs.eg.db,
        ont = ont,
        pAdjustMethod = "BH",
        pvalueCutoff = 0.05,
        qvalueCutoff = 0.05,
        readable = TRUE
      )

      if (nrow(go_enrich) > 0) {
        go_results[[ont]] <- go_enrich
        write.csv(as.data.frame(go_enrich),
                 file.path(out_dir, sprintf("GO_%s.csv", ont)),
                 row.names = FALSE)
        log_message(sprintf("  GO %s: %d terms", ont, nrow(go_enrich)))
      } else {
        log_message(sprintf("  GO %s: No significant terms", ont),
                   level = "WARNING")
      }
    }, error = function(e) {
      log_message(sprintf("  GO %s failed: %s", ont, e$message),
                 level = "ERROR")
    })
  }

  # Create visualization
  if (length(go_results) > 0) {
    cairo_pdf(file.path(out_dir, "GO_Enrichment.pdf"), width = 12, height = 10, family = "Arial")
    for (ont in names(go_results)) {
      if (nrow(go_results[[ont]]) > 0) {
        print(barplot(go_results[[ont]], showCategory = 20,
                     title = sprintf("GO %s Enrichment", ont)))
      }
    }
    dev.off()
  }

  return(go_results)
}

#' KEGG pathway enrichment
perform_kegg_enrichment <- function(entrez_ids) {
  out_dir <- file.path(CONFIG$output_dir, "Enrichment")

  tryCatch({
    kegg_enrich <- enrichKEGG(
      gene = entrez_ids,
      organism = "hsa",
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05,
      qvalueCutoff = 0.05
    )

    if (nrow(kegg_enrich) > 0) {
      kegg_enrich <- setReadable(kegg_enrich, OrgDb = org.Hs.eg.db,
                                keyType = "ENTREZID")

      write.csv(as.data.frame(kegg_enrich),
               file.path(out_dir, "KEGG_Pathways.csv"),
               row.names = FALSE)

      # Visualization
      cairo_pdf(file.path(out_dir, "KEGG_Enrichment.pdf"), width = 12, height = 10, family = "Arial")
      print(barplot(kegg_enrich, showCategory = 20, title = "KEGG Pathways"))
      print(dotplot(kegg_enrich, showCategory = 20, title = "KEGG Pathways"))
      dev.off()

      log_message(sprintf("  KEGG: %d pathways", nrow(kegg_enrich)))
      return(kegg_enrich)
    } else {
      log_message("  KEGG: No significant pathways", level = "WARNING")
      return(NULL)
    }
  }, error = function(e) {
    log_message(sprintf("  KEGG failed: %s", e$message), level = "ERROR")
    return(NULL)
  })
}

#' GSEA analysis
perform_gsea <- function(deg_all, gene_entrez) {
  out_dir <- file.path(CONFIG$output_dir, "GSEA")

  # Prepare ranked gene list
  gene_list <- deg_all$logFC
  names(gene_list) <- rownames(deg_all)

  # Match with Entrez IDs
  gene_list_entrez <- gene_entrez[!duplicated(gene_entrez$SYMBOL), ]

  # Filter for genes that have both logFC and Entrez ID
  common_genes <- intersect(names(gene_list), gene_list_entrez$SYMBOL)

  if (length(common_genes) == 0) {
    log_message("No common genes found for GSEA", level = "ERROR")
    return(NULL)
  }

  gene_list_final <- gene_list[common_genes]
  entrez_match <- match(common_genes, gene_list_entrez$SYMBOL)
  names(gene_list_final) <- gene_list_entrez$ENTREZID[entrez_match]

  # CRITICAL: Sort in DECREASING order and remove duplicates
  gene_list_final <- sort(gene_list_final, decreasing = TRUE)
  gene_list_final <- gene_list_final[!duplicated(names(gene_list_final))]

  # Verify sorting
  if (!all(diff(gene_list_final) <= 0)) {
    log_message("ERROR: Gene list is not properly sorted in decreasing order", level = "ERROR")
    return(NULL)
  }

  log_message(sprintf("  Prepared gene list: %d genes, range: [%.3f, %.3f]",
                     length(gene_list_final), min(gene_list_final), max(gene_list_final)))

  gsea_results <- list()

  # GSEA GO
  tryCatch({
    gsea_go <- gseGO(
      geneList = gene_list_final,
      OrgDb = org.Hs.eg.db,
      ont = "BP",
      pvalueCutoff = 0.05,
      pAdjustMethod = "BH",
      verbose = FALSE
    )

    if (nrow(gsea_go) > 0) {
      gsea_results$GO <- gsea_go
      write.csv(as.data.frame(gsea_go),
               file.path(out_dir, "GSEA_GO.csv"),
               row.names = FALSE)
      log_message(sprintf("  GSEA GO: %d terms", nrow(gsea_go)))
    } else {
      log_message("  GSEA GO: No significant terms", level = "WARNING")
    }
  }, error = function(e) {
    log_message(sprintf("  GSEA GO failed: %s", e$message), level = "ERROR")
  })

  # GSEA KEGG
  tryCatch({
    gsea_kegg <- gseKEGG(
      geneList = gene_list_final,
      organism = "hsa",
      pvalueCutoff = 0.05,
      pAdjustMethod = "BH",
      verbose = FALSE
    )

    if (nrow(gsea_kegg) > 0) {
      gsea_kegg <- setReadable(gsea_kegg, OrgDb = org.Hs.eg.db,
                              keyType = "ENTREZID")
      gsea_results$KEGG <- gsea_kegg
      write.csv(as.data.frame(gsea_kegg),
               file.path(out_dir, "GSEA_KEGG.csv"),
               row.names = FALSE)
      log_message(sprintf("  GSEA KEGG: %d pathways", nrow(gsea_kegg)))
    } else {
      log_message("  GSEA KEGG: No significant pathways", level = "WARNING")
    }
  }, error = function(e) {
    log_message(sprintf("  GSEA KEGG failed: %s", e$message), level = "ERROR")
  })

  # Create GSEA visualization plots
  if (length(gsea_results) > 0) {
    log_message("Creating GSEA visualization plots...")

    # GSEA GO Plots
    if (!is.null(gsea_results$GO) && nrow(gsea_results$GO) > 0) {
      tryCatch({
        cairo_pdf(file.path(out_dir, "GSEA_GO_Plots.pdf"), width = 14, height = 10, family = "Arial")

        # Dotplot
        if (nrow(gsea_results$GO) >= 1) {
          print(dotplot(gsea_results$GO, showCategory = min(30, nrow(gsea_results$GO)),
                       title = "GSEA GO Biological Process",
                       font.size = 10))
        }

        # Enrichment plots for top 5 pathways
        top_pathways <- head(gsea_results$GO$ID, min(5, nrow(gsea_results$GO)))
        if (length(top_pathways) > 0) {
          print(gseaplot2(gsea_results$GO,
                         geneSetID = top_pathways,
                         title = "GSEA GO: Top Enriched Pathways",
                         pvalue_table = TRUE))
        }

        dev.off()
        log_message("  GSEA GO visualization saved")
      }, error = function(e) {
        log_message(sprintf("  GSEA GO visualization failed: %s", e$message), level = "WARNING")
      })
    }

    # GSEA KEGG Plots
    if (!is.null(gsea_results$KEGG) && nrow(gsea_results$KEGG) > 0) {
      tryCatch({
        cairo_pdf(file.path(out_dir, "GSEA_KEGG_Plots.pdf"), width = 14, height = 10, family = "Arial")

        # Dotplot
        if (nrow(gsea_results$KEGG) >= 1) {
          print(dotplot(gsea_results$KEGG, showCategory = min(30, nrow(gsea_results$KEGG)),
                       title = "GSEA KEGG Pathways",
                       font.size = 10))
        }

        # Enrichment plots for top 5 pathways
        top_pathways <- head(gsea_results$KEGG$ID, min(5, nrow(gsea_results$KEGG)))
        if (length(top_pathways) > 0) {
          print(gseaplot2(gsea_results$KEGG,
                         geneSetID = top_pathways,
                         title = "GSEA KEGG: Top Enriched Pathways",
                         pvalue_table = TRUE))
        }

        dev.off()
        log_message("  GSEA KEGG visualization saved")
      }, error = function(e) {
        log_message(sprintf("  GSEA KEGG visualization failed: %s", e$message), level = "WARNING")
      })
    }

    # Combined comparison plot if both GO and KEGG available
    if (!is.null(gsea_results$GO) && !is.null(gsea_results$KEGG) &&
        nrow(gsea_results$GO) > 0 && nrow(gsea_results$KEGG) > 0) {
      tryCatch({
        cairo_pdf(file.path(out_dir, "GSEA_Combined_Comparison.pdf"), width = 16, height = 10, family = "Arial")
        par(mfrow = c(1, 2))

        # Ridgeplot for GO
        print(ridgeplot(gsea_results$GO, showCategory = 20) +
              labs(title = "GSEA GO: NES Distribution"))

        # Ridgeplot for KEGG
        print(ridgeplot(gsea_results$KEGG, showCategory = 20) +
              labs(title = "GSEA KEGG: NES Distribution"))

        dev.off()
        log_message("  GSEA combined comparison plot saved")
      }, error = function(e) {
        log_message(sprintf("  GSEA combined plot failed: %s", e$message), level = "WARNING")
      })
    }
  }

  return(gsea_results)
}

# ==============================================================================
# 11. IMIDAZOQUINOLINE-SLE TARGET INTERSECTION MODULE
# ==============================================================================

#' Define Imidazoquinoline-SLE intersection targets
#' Updated to read from the latest CSV file: imidazoline_SLE_intersection.csv
get_imidazoquinoline_targets <- function() {
  # Read targets from the updated CSV file
  csv_path <- "~/statics/imidazoline_SLE_intersection.csv"

  tryCatch({
    targets_df <- read.csv(csv_path, stringsAsFactors = FALSE)
    targets <- targets_df$Target
    log_message(sprintf("Successfully loaded %d targets from %s", length(targets), csv_path))
    return(targets)
  }, error = function(e) {
    log_message(sprintf("Error reading CSV file: %s", e$message), level = "ERROR")
    log_message("Falling back to hardcoded targets from previous version", level = "WARNING")
    # Fallback to previous hardcoded targets if CSV read fails
    targets <- c(
      "ABCB11","ABCC1","ABCC2","ABCC4","ABCC9","ABCG2","ABL1","ACADM","ACAT1","ACE",
      "ACHE","ADA","ADAM10","ADAM17","ADK","ADORA1","ADORA2A","ADRA2A","ADRA2B","ADRB1",
      "ADRB2","ADRB3","AGTR1","AGTR2","AKT1","AKT2","ALAD","ALB","ALDH2","ALK","ALOX5",
      "ALOX5AP","ANG","ANXA5","APAF1","APEX1","APP","APRT","AR","ARG1","ATIC","ATR",
      "AVPR2","AXL","BACE1","BCHE","BCL2L1","BMP7","BRD2","BRD4","BTK","C1S","C5AR1",
      "CA2","CAPN1","CASP1","CASP3","CASR","CBS","CCL5","CCNA2","CCR2","CCR3","CCR4",
      "CCR5","CD1A","CD38","CDC42","CDK1","CDK2","CDK5","CFB","CFTR","CHEK1","CHRM1",
      "CHRM3","CHRNA4","CHUK","CNR1","CREBBP","CSK","CSNK2A1","CSNK2B","CTNNA1","CTSB",
      "CTSC","CTSD","CTSF","CTSG","CTSS","CXCR2","CXCR4","CYP11B1","CYP17A1","CYP19A1",
      "CYP1A2","CYP2A6","CYP2C19","CYP2C9","CYP2D6","CYP2E1","CYP3A4","DDX39B","DHFR",
      "DOT1L","DPP4","DRD1","DRD2","DRD3","DRD4","EDNRA","EEA1","EGFR","EIF2AK3","EIF4E",
      "ELANE","ERAP1","ERBB2","ERBB4","ESR1","ESR2","F10","F11","F2","FARS2","FCGRT",
      "FGFR1","FGFR2","FGG","FLT1","FOLH1","FYN","GABRA1","GATM","GBA2","GCDH","GCK",
      "GLRA1","GP1BA","GPI","GRB2","GRIN2B","GSK3B","GSR","GSTM1","GSTP1","HADH","HCK",
      "HDAC2","HDAC6","HDAC8","HEXB","HIF1A","HK1","HMGCR","HMOX1","HPRT1","HRAS",
      "HSD11B1","HSD17B10","HSP90AA1","HSP90AB1","HSPA8","HTR1A","HTR2A","HTR2C","HTR3A",
      "HTT","ICAM1","IDO1","IGF1","IGF1R","IKBKB","IL2","INSR","ISG20","ITGB1","ITK",
      "IVD","JAK1","JAK2","JAK3","KCNA5","KCNH2","KDM1A","KDR","KIF11","KIT","LCK","LCN2",
      "LGALS3","LIMK1","LMNA","LRRK2","LYN","MAN1B1","MAOA","MAOB","MAP2K1","MAPK1",
      "MAPK10","MAPK12","MAPK14","MAPK3","MAPK8","MC4R","MCL1","MDM2","MIF","MME","MMP1",
      "MMP12","MMP13","MMP2","MMP3","MMP7","MMP9","MPO","MTHFD1","NAMPT","NFE2L2","NFKB1",
      "NMNAT1","NOS1","NOS2","NOS3","NQO1","NR1H3","NR1H4","NR3C1","NR3C2","NTRK1","NTRK3",
      "OAT","OPRM1","OTC","P2RX7","PADI4","PAH","PARP1","PDE2A","PDE4D","PDE5A","PEPD","PGR",
      "PHF8","PIK3CD","PIK3CG","PIK3R1","PIM1","PLA2G2A","PLAU","PLG","PMS2","PNP","POLB",
      "PPARA","PPARD","PPARG","PPP3CA","PRKACA","PRKCA","PRKCB","PRKCD","PRKCG","PRKCQ",
      "PRKDC","PROCR","PSEN1","PSEN2","PSMB9","PTGS1","PTGS2","PTK2","PTPN11","PTPN13",
      "PTPN2","PTPRC","RAB11A","RAC1","RAC2","RAF1","RBP4","REN","RET","RHOA","RORA","RXRA",
      "RXRB","S100A9","SAMHD1","SCN3A","SELE","SELP","SERPINA1","SHBG","SIGMAR1","SIRT1",
      "SLC18A3","SLC1A3","SLC2A1","SLC6A2","SLC6A3","SLC6A4","SLC6A5","SLC6A9","SPR","SRC",
      "STAT1","STING1","SYK","TACR1","TAP1","TBXAS1","TEK","TERT","TGFB2","TGFBR1","TGM2",
      "TLR7","TLR8","TNK2","TOP1","TOP2A","TPI1","TRPV1","TTPA","TTR","TYK2","TYMS","TYRO3",
      "UMPS","VDR","WARS1","WAS","XIAP","XPO1","YARS1","ZAP70"
    )
    return(targets)
  })
}

#' Validate expression of imidazoquinoline targets in SLE vs Control
validate_target_expression <- function(merged_data, deg_results) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 8: Imidazoquinoline-SLE Target Validation\n")
  cat("================================================================================\n\n")

  targets <- get_imidazoquinoline_targets()
  log_message(sprintf("Total imidazoquinoline-SLE targets to validate: %d", length(targets)))

  expr_matrix <- merged_data$expr_after
  group_info <- merged_data$group
  deg_all <- deg_results$all_results

  targets_present <- targets[targets %in% rownames(expr_matrix)]
  targets_missing <- targets[!targets %in% rownames(expr_matrix)]

  log_message(sprintf("Targets found in expression data: %d/%d",
                     length(targets_present), length(targets)))

  if (length(targets_missing) > 0) {
    log_message(sprintf("Missing targets: %d", length(targets_missing)), level = "WARNING")
  }

  if (length(targets_present) == 0) {
    log_message("ERROR: No target genes found in expression data", level = "ERROR")
    return(NULL)
  }

  target_expr <- expr_matrix[targets_present, , drop = FALSE]
  target_deg <- deg_all[targets_present, , drop = FALSE]
  target_deg <- target_deg[order(target_deg$adj.P.Val), ]

  target_deg$Status <- ifelse(
    target_deg$adj.P.Val < CONFIG$adj_p_threshold,
    ifelse(target_deg$logFC > CONFIG$log_fc_threshold, "Upregulated",
          ifelse(target_deg$logFC < -CONFIG$log_fc_threshold, "Downregulated",
                 "Not Significant")),
    "Not Significant"
  )

  # Replace NA Status with "Not Significant"
  target_deg$Status[is.na(target_deg$Status)] <- "Not Significant"

  n_up <- sum(target_deg$Status == "Upregulated", na.rm = TRUE)
  n_down <- sum(target_deg$Status == "Downregulated", na.rm = TRUE)
  n_ns <- sum(target_deg$Status == "Not Significant", na.rm = TRUE)

  cat("\n")
  cat("Target Gene Expression Summary:\n")
  cat(sprintf("  Upregulated: %d (%.1f%%)\n", n_up, 100*n_up/nrow(target_deg)))
  cat(sprintf("  Downregulated: %d (%.1f%%)\n", n_down, 100*n_down/nrow(target_deg)))
  cat(sprintf("  Not Significant: %d (%.1f%%)\n", n_ns, 100*n_ns/nrow(target_deg)))

  out_dir <- file.path(CONFIG$output_dir, "Target_Validation")
  create_dir(out_dir)

  write.csv(target_deg, file.path(out_dir, "Target_Genes_DEG_Results.csv"), row.names = TRUE)

  create_target_validation_plots(target_expr, target_deg, group_info, out_dir)

  return(list(
    targets_all = targets,
    targets_present = targets_present,
    targets_missing = targets_missing,
    target_expression = target_expr,
    target_deg = target_deg,
    summary = data.frame(
      Category = c("Upregulated", "Downregulated", "Not Significant"),
      Count = c(n_up, n_down, n_ns),
      Percentage = c(100*n_up/nrow(target_deg), 100*n_down/nrow(target_deg), 100*n_ns/nrow(target_deg))
    )
  ))
}

#' Create target validation plots
create_target_validation_plots <- function(target_expr, target_deg, group_info, out_dir) {
  log_message("Creating target validation plots...")

  # Volcano plot with gene labels for target genes
  cairo_pdf(file.path(out_dir, "Target_Genes_Volcano.pdf"), width = 12, height = 10, family = "Arial")
  target_deg$gene_name <- rownames(target_deg)

  # Select top significant genes for labeling
  top_targets <- target_deg[!is.na(target_deg$Status) & target_deg$Status != "Not Significant", ]
  if (nrow(top_targets) > 15) {
    top_targets <- top_targets[order(top_targets$adj.P.Val), ][1:15, ]
  }

  p <- ggplot(target_deg, aes(x = logFC, y = -log10(adj.P.Val), color = Status)) +
    geom_point(alpha = 0.7, size = 3) +
    geom_text(data = top_targets,
             aes(label = gene_name),
             hjust = -0.1, vjust = 0.5, size = 3.5,
             check_overlap = TRUE, show.legend = FALSE) +
    scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue", "Not Significant" = "grey")) +
    geom_hline(yintercept = -log10(CONFIG$adj_p_threshold), linetype = "dashed", color = "black") +
    geom_vline(xintercept = c(-CONFIG$log_fc_threshold, CONFIG$log_fc_threshold), linetype = "dashed", color = "black") +
    labs(title = "Imidazoquinoline-SLE Target Genes: SLE vs Control", x = "log2(Fold Change)", y = "-log10(adjusted P-value)") +
    theme_bw() + theme(legend.position = "right")
  print(p)
  dev.off()

  # Heatmap for all target genes with disease group clustering
  cairo_pdf(file.path(out_dir, "Target_Genes_Heatmap_All.pdf"), width = 14, height = 16, family = "Arial")
  if (nrow(target_expr) > 0 && ncol(target_expr) > 0) {
    tryCatch({
      # Sort columns by group for better visualization
      group_order <- order(group_info)
      target_expr_sorted <- target_expr[, group_order]
      group_info_sorted <- group_info[group_order]

      # Calculate gap position
      n_control <- sum(group_info_sorted == "Control")
      gaps_col <- n_control

      # Remove genes with zero variance or non-finite values
      gene_var <- apply(target_expr_sorted, 1, var, na.rm = TRUE)
      valid_genes <- is.finite(gene_var) & gene_var > 0 &
                     !apply(target_expr_sorted, 1, function(x) any(!is.finite(x)))

      if (sum(valid_genes) == 0) {
        plot.new()
        text(0.5, 0.5, "No valid genes for heatmap\n(all have zero variance or non-finite values)",
             cex = 1.2, col = "red")
      } else {
        target_expr_sorted <- target_expr_sorted[valid_genes, , drop = FALSE]

        annotation_col <- data.frame(Group = group_info_sorted, row.names = colnames(target_expr_sorted))
        annotation_row <- data.frame(Status = target_deg[rownames(target_expr_sorted), "Status"],
                                     row.names = rownames(target_expr_sorted))

        # Define colors
        ann_colors <- list(
          Group = c(Control = "#4DBBD5", SLE = "#E64B35"),
          Status = c(Upregulated = "red", Downregulated = "blue", "Not Significant" = "grey")
        )

        pheatmap(target_expr_sorted, scale = "row",
                clustering_distance_rows = "correlation",
                clustering_distance_cols = "correlation",
                clustering_method = "ward.D2",
                annotation_col = annotation_col,
                annotation_row = annotation_row,
                annotation_colors = ann_colors,
                show_colnames = FALSE,
                show_rownames = TRUE,
                cluster_cols = FALSE,  # No clustering - strict Control vs SLE grouping
                gaps_col = gaps_col,  # Gap between Control and SLE
                fontsize_row = 6,
                color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
                main = sprintf("Imidazoquinoline-SLE Target Genes (%d genes)\nGrouped by Disease Status", sum(valid_genes)))
      }
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, paste("Error creating heatmap:", e$message), cex = 1.2, col = "red")
    })
  }
  dev.off()

  # Heatmap for significant target genes with disease group clustering
  sig_targets <- rownames(target_deg[!is.na(target_deg$Status) & target_deg$Status != "Not Significant", ])
  # Remove NA values from sig_targets
  sig_targets <- sig_targets[!is.na(sig_targets)]
  if (length(sig_targets) > 0) {
    cairo_pdf(file.path(out_dir, "Target_Genes_Heatmap_Significant.pdf"), width = 14, height = max(8, length(sig_targets) * 0.2), family = "Arial")

    tryCatch({
      # Sort columns by group for better visualization
      sig_expr <- target_expr[sig_targets, , drop = FALSE]
      group_order <- order(group_info)
      sig_expr_sorted <- sig_expr[, group_order]
      group_info_sorted <- group_info[group_order]

      # Calculate gap position
      n_control <- sum(group_info_sorted == "Control")
      gaps_col <- n_control

      # Remove genes with zero variance or non-finite values
      gene_var <- apply(sig_expr_sorted, 1, var, na.rm = TRUE)
      valid_genes <- is.finite(gene_var) & gene_var > 0 &
                     !apply(sig_expr_sorted, 1, function(x) any(!is.finite(x)))

      if (sum(valid_genes) == 0) {
        plot.new()
        text(0.5, 0.5, "No valid significant genes for heatmap\n(all have zero variance or non-finite values)",
             cex = 1.2, col = "red")
      } else {
        sig_expr_sorted <- sig_expr_sorted[valid_genes, , drop = FALSE]
        sig_targets_valid <- rownames(sig_expr_sorted)

        annotation_col <- data.frame(Group = group_info_sorted, row.names = colnames(sig_expr_sorted))
        annotation_row <- data.frame(Status = target_deg[sig_targets_valid, "Status"], row.names = sig_targets_valid)

        # Define colors
        ann_colors <- list(
          Group = c(Control = "#4DBBD5", SLE = "#E64B35"),
          Status = c(Upregulated = "red", Downregulated = "blue")
        )

        # Only cluster if we have more than 1 gene
        if (length(sig_targets_valid) > 1) {
          pheatmap(sig_expr_sorted, scale = "row",
                  clustering_distance_rows = "correlation",
                  clustering_distance_cols = "correlation",
                  clustering_method = "ward.D2",
                  annotation_col = annotation_col,
                  annotation_row = annotation_row,
                  annotation_colors = ann_colors,
                  show_colnames = FALSE,
                  show_rownames = TRUE,
                  cluster_cols = FALSE,  # No clustering - strict Control vs SLE grouping
                  gaps_col = gaps_col,  # Gap between Control and SLE
                  fontsize_row = 8,
                  color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
                  main = sprintf("Significant Target Genes (%d genes)\nGrouped by Disease Status", length(sig_targets_valid)))
        } else {
          # No clustering for single gene
          pheatmap(sig_expr_sorted, scale = "row",
                  cluster_rows = FALSE,
                  cluster_cols = FALSE,
                  annotation_col = annotation_col,
                  annotation_row = annotation_row,
                  annotation_colors = ann_colors,
                  show_colnames = FALSE,
                  show_rownames = TRUE,
                  gaps_col = gaps_col,
                  fontsize_row = 8,
                  color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
                  main = sprintf("Significant Target Gene (%s)\nGrouped by Disease Status", sig_targets_valid[1]))
        }
      }
    }, error = function(e) {
      plot.new()
      text(0.5, 0.5, paste("Error creating significant heatmap:", e$message), cex = 1.2, col = "red")
    })

    dev.off()
  } else {
    # No significant targets, create empty plot
    cairo_pdf(file.path(out_dir, "Target_Genes_Heatmap_Significant.pdf"), width = 14, height = 8, family = "Arial")
    plot.new()
    text(0.5, 0.5, "No significant target genes found", cex = 1.5, col = "orange")
    dev.off()
  }

  # Filter out NA Status and get non-significant targets
  valid_targets <- target_deg[!is.na(target_deg$Status) & target_deg$Status != "Not Significant", ]
  top_targets <- rownames(valid_targets)[1:min(12, nrow(valid_targets))]
  if (length(top_targets) > 0 && !any(is.na(top_targets))) {
    cairo_pdf(file.path(out_dir, "Target_Genes_Boxplots_Top.pdf"), width = 16, height = 12, family = "Arial")
    par(mfrow = c(3, 4), mar = c(8, 4, 3, 2))
    for (gene in top_targets) {
      expr_vals <- as.numeric(target_expr[gene, ])
      boxplot(expr_vals ~ group_info, main = gene, ylab = "Expression", col = c("lightblue", "salmon"), las = 2)
      pval <- target_deg[gene, "adj.P.Val"]
      logfc <- target_deg[gene, "logFC"]
      mtext(sprintf("adj.P=%.2e, logFC=%.2f", pval, logfc), side = 3, line = 0.5, cex = 0.7)
    }
    dev.off()
  }

  log_message("Target validation plots created")
}

# ==============================================================================
# 12. IMMUNE CELL INFILTRATION ANALYSIS MODULE
# ==============================================================================

#' Perform immune cell infiltration analysis
perform_immune_infiltration <- function(merged_data) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 9: Immune Cell Infiltration Analysis\n")
  cat("================================================================================\n\n")

  expr_matrix <- merged_data$expr_after
  group_info <- merged_data$group

  log_message("Preparing expression data for immune infiltration analysis...")

  tryCatch({
    # Refined immune cell signatures with M1/M2 macrophage polarization
    immune_signatures <- list(
      # B cell subsets
      "B_cells" = c("CD19", "CD79A", "MS4A1", "CD79B", "BLK", "PAX5"),
      "Plasma_cells" = c("SDC1", "TNFRSF17", "TNFRSF13B", "MZB1", "XBP1"),

      # T cell subsets
      "T_cells_CD4" = c("CD4", "CD3D", "CD3E", "IL7R", "LCK", "CD40LG"),
      "T_cells_CD8" = c("CD8A", "CD8B", "CD3D", "CD3E", "GZMB", "PRF1"),
      "T_cells_regulatory" = c("FOXP3", "IL2RA", "CTLA4", "IKZF2"),
      "T_cells_follicular_helper" = c("CXCR5", "PDCD1", "BCL6", "IL21"),
      "T_cells_gamma_delta" = c("TRGC1", "TRGC2", "TRDC"),

      # NK cells
      "NK_cells" = c("NCAM1", "NCR1", "NKG7", "KLRD1", "KLRB1", "KLRF1"),

      # Myeloid cells
      "Monocytes" = c("CD14", "CD68", "CSF1R", "FCGR3A", "CD86"),

      # Macrophage polarization states
      "Macrophages_M1" = c("CD68", "CD86", "NOS2", "TNF", "IL1B", "IL6", "CXCL10", "CXCL9", "CCL5"),
      "Macrophages_M2" = c("CD68", "CD163", "MRC1", "MSR1", "ARG1", "IL10", "TGFB1", "CCL18", "CCL22"),

      # Dendritic cells
      "Dendritic_cells_conventional" = c("ITGAX", "CD1C", "HLA-DRA", "HLA-DRB1", "FCER1A"),
      "Dendritic_cells_plasmacytoid" = c("LILRA4", "IL3RA", "CLEC4C", "IRF7", "IRF8"),

      # Neutrophils
      "Neutrophils" = c("FCGR3B", "CSF3R", "FPR1", "CXCR2", "S100A8", "S100A9"),

      # Mast cells
      "Mast_cells" = c("TPSAB1", "TPSB2", "CPA3", "KIT"),

      # Eosinophils
      "Eosinophils" = c("SIGLEC8", "IL5RA", "CCR3", "PRG2"),

      # Basophils
      "Basophils" = c("MS4A2", "GATA2", "CPA3")
    )

    immune_sigs_filtered <- lapply(immune_signatures, function(sig) sig[sig %in% rownames(expr_matrix)])
    immune_sigs_filtered <- immune_sigs_filtered[sapply(immune_sigs_filtered, length) > 0]

    log_message(sprintf("Using %d immune cell signatures", length(immune_sigs_filtered)))

    suppressPackageStartupMessages({
      if (!require("GSVA", quietly = TRUE)) {
        log_message("GSVA package not available, skipping immune analysis", level = "WARNING")
        return(NULL)
      }
    })

    # Check GSVA version and use appropriate function call
    gsva_version <- packageVersion("GSVA")
    log_message(sprintf("Using GSVA version: %s", gsva_version))

    if (gsva_version >= "1.40.0") {
      # New GSVA API (version >= 1.40.0)
      log_message("Using new GSVA API (>=1.40.0)")
      tryCatch({
        # Convert to GeneSetCollection format
        gene_set_list <- GeneSetCollection(lapply(names(immune_sigs_filtered), function(name) {
          GeneSet(immune_sigs_filtered[[name]], setName = name)
        }))

        # Use new gsvaParam interface
        param <- gsvaParam(as.matrix(expr_matrix), gene_set_list, kcdf = "Gaussian")
        gsva_scores <- gsva(param, verbose = FALSE)
      }, error = function(e) {
        log_message(sprintf("New GSVA API failed, trying legacy method: %s", e$message), level = "WARNING")
        # Fallback to legacy method
        gsva_scores <<- gsva(as.matrix(expr_matrix), immune_sigs_filtered,
                            method = "ssgsea", kcdf = "Gaussian", verbose = FALSE)
      })
    } else {
      # Legacy GSVA API (version < 1.40.0)
      log_message("Using legacy GSVA API (<1.40.0)")
      gsva_scores <- gsva(as.matrix(expr_matrix), immune_sigs_filtered,
                         method = "ssgsea", kcdf = "Gaussian", verbose = FALSE)
    }

    log_message(sprintf("Calculated enrichment scores for %d cell types", nrow(gsva_scores)))

    immune_stats <- data.frame()
    for (cell_type in rownames(gsva_scores)) {
      sle_scores <- gsva_scores[cell_type, group_info == "SLE"]
      ctrl_scores <- gsva_scores[cell_type, group_info == "Control"]
      test_result <- t.test(sle_scores, ctrl_scores)
      immune_stats <- rbind(immune_stats, data.frame(
        CellType = cell_type,
        Mean_SLE = mean(sle_scores, na.rm = TRUE),
        Mean_Control = mean(ctrl_scores, na.rm = TRUE),
        logFC = log2(mean(sle_scores, na.rm = TRUE) / mean(ctrl_scores, na.rm = TRUE)),
        P.Value = test_result$p.value,
        adj.P.Val = NA
      ))
    }

    immune_stats$adj.P.Val <- p.adjust(immune_stats$P.Value, method = "BH")
    immune_stats <- immune_stats[order(immune_stats$adj.P.Val), ]

    out_dir <- file.path(CONFIG$output_dir, "Immune_Infiltration")
    create_dir(out_dir)
    write.csv(immune_stats, file.path(out_dir, "Immune_Cell_Differences.csv"), row.names = FALSE)

    create_immune_plots(gsva_scores, immune_stats, group_info, out_dir)

    log_message("Immune infiltration analysis completed")

    return(list(
      gsva_scores = gsva_scores,
      statistics = immune_stats,
      signatures = immune_sigs_filtered,
      output_dir = out_dir
    ))

  }, error = function(e) {
    log_message(sprintf("Immune infiltration analysis failed: %s", e$message), level = "ERROR")
    return(NULL)
  })
}

#' Create immune infiltration plots
create_immune_plots <- function(gsva_scores, immune_stats, group_info, out_dir) {
  log_message("Creating immune infiltration plots...")

  # Immune scores heatmap with disease group clustering
  cairo_pdf(file.path(out_dir, "Immune_Scores_Heatmap.pdf"), width = 16, height = 12, family = "Arial")

  tryCatch({
    # Sort columns by group for better visualization
    group_order <- order(group_info)
    gsva_scores_sorted <- gsva_scores[, group_order]
    group_info_sorted <- group_info[group_order]

    # Calculate gap position
    n_control <- sum(group_info_sorted == "Control")
    gaps_col <- n_control

    # Remove cell types with zero variance or non-finite values
    cell_var <- apply(gsva_scores_sorted, 1, var, na.rm = TRUE)
    valid_cells <- is.finite(cell_var) & cell_var > 0 &
                   !apply(gsva_scores_sorted, 1, function(x) any(!is.finite(x)))

    if (sum(valid_cells) == 0) {
      plot.new()
      text(0.5, 0.5, "No valid cell types for heatmap\n(all have zero variance or non-finite values)",
           cex = 1.2, col = "red")
    } else {
      gsva_scores_sorted <- gsva_scores_sorted[valid_cells, , drop = FALSE]

      # Create cell type categories for row annotation
      cell_categories <- sapply(rownames(gsva_scores_sorted), function(x) {
        if (grepl("Macrophages_M1", x)) return("M1 Macrophage")
        else if (grepl("Macrophages_M2", x)) return("M2 Macrophage")
        else if (grepl("^T_cells", x)) return("T cells")
        else if (grepl("^B_cells|Plasma", x)) return("B cells")
        else if (grepl("NK", x)) return("NK cells")
        else if (grepl("Dendritic", x)) return("Dendritic cells")
        else if (grepl("Mono", x)) return("Myeloid")
        else if (grepl("Neutrophil|Eosinophil|Basophil|Mast", x)) return("Granulocytes")
        else return("Other")
      })

      annotation_col <- data.frame(
        Disease_Group = group_info_sorted,
        row.names = colnames(gsva_scores_sorted)
      )

      annotation_row <- data.frame(
        Cell_Category = cell_categories,
        row.names = rownames(gsva_scores_sorted)
      )

      # Define colors with emphasis on M1/M2 distinction
      ann_colors <- list(
        Disease_Group = c(Control = "#4DBBD5", SLE = "#E64B35"),
        Cell_Category = c(
          "M1 Macrophage" = "#DC0000",
          "M2 Macrophage" = "#00A087",
          "T cells" = "#3C5488",
          "B cells" = "#F39B7F",
          "NK cells" = "#8491B4",
          "Dendritic cells" = "#91D1C2",
          "Myeloid" = "#7E6148",
          "Granulocytes" = "#B09C85",
          "Other" = "#CCCCCC"
        )
      )

      # Create better row labels (remove underscores, capitalize)
      row_labels <- gsub("_", " ", rownames(gsva_scores_sorted))
      row_labels <- gsub("Macrophages M1", "M1 Macrophages", row_labels)
      row_labels <- gsub("Macrophages M2", "M2 Macrophages", row_labels)

      pheatmap(gsva_scores_sorted,
              scale = "row",
              clustering_distance_cols = "correlation",
              clustering_distance_rows = "correlation",
              clustering_method = "ward.D2",
              annotation_col = annotation_col,
              annotation_row = annotation_row,
              annotation_colors = ann_colors,
              labels_row = row_labels,
              show_colnames = FALSE,
              cluster_cols = FALSE,  # No clustering - strict Control vs SLE grouping
              gaps_col = gaps_col,  # Gap between Control and SLE
              fontsize_row = 9,
              color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
              main = sprintf("Immune Cell Infiltration (%d cell types)\nGrouped by Disease Status", sum(valid_cells)))
    }
  }, error = function(e) {
    plot.new()
    text(0.5, 0.5, paste("Error creating immune heatmap:", e$message), cex = 1.2, col = "red")
  })

  dev.off()

  # Boxplots for all immune cell types
  cairo_pdf(file.path(out_dir, "Immune_Scores_Boxplots.pdf"), width = 18, height = 14, family = "Arial")
  n_cells <- nrow(gsva_scores)
  n_cols <- ceiling(sqrt(n_cells))
  n_rows <- ceiling(n_cells / n_cols)
  par(mfrow = c(n_rows, n_cols), mar = c(8, 4, 3, 2))
  for (i in 1:nrow(gsva_scores)) {
    cell_type <- rownames(gsva_scores)[i]
    scores <- as.numeric(gsva_scores[i, ])

    # Special colors for M1/M2 macrophages
    if (grepl("Macrophages_M1", cell_type)) {
      box_colors <- c("#4DBBD5", "#DC0000")  # M1: red for SLE
    } else if (grepl("Macrophages_M2", cell_type)) {
      box_colors <- c("#4DBBD5", "#00A087")  # M2: green for SLE
    } else {
      box_colors <- c("#4DBBD5", "#E64B35")  # Default
    }

    # Clean up cell type name for display
    display_name <- gsub("_", " ", cell_type)
    display_name <- gsub("Macrophages M1", "M1 Macrophages", display_name)
    display_name <- gsub("Macrophages M2", "M2 Macrophages", display_name)

    boxplot(scores ~ group_info, main = display_name,
           ylab = "Enrichment Score",
           col = box_colors, las = 2,
           cex.main = 1.2, font.main = 2)

    pval <- immune_stats[immune_stats$CellType == cell_type, "adj.P.Val"]
    if (length(pval) > 0 && !is.na(pval)) {
      sig_label <- ifelse(pval < 0.001, "***",
                         ifelse(pval < 0.01, "**",
                               ifelse(pval < 0.05, "*", "ns")))
      mtext(sprintf("P=%.2e %s", pval, sig_label),
           side = 3, line = 0.5, cex = 0.7,
           col = ifelse(pval < 0.05, "red", "black"))
    }
  }
  dev.off()

  # Create M1/M2 ratio comparison plot
  m1_idx <- which(grepl("Macrophages_M1", rownames(gsva_scores)))
  m2_idx <- which(grepl("Macrophages_M2", rownames(gsva_scores)))

  if (length(m1_idx) > 0 && length(m2_idx) > 0) {
    cairo_pdf(file.path(out_dir, "Macrophage_M1_M2_Comparison.pdf"), width = 14, height = 10, family = "Arial")

    # M1/M2 ratio
    m1_scores <- as.numeric(gsva_scores[m1_idx, ])
    m2_scores <- as.numeric(gsva_scores[m2_idx, ])
    m1_m2_ratio <- m1_scores / (m2_scores + 1e-6)  # Add small value to avoid division by zero

    par(mfrow = c(2, 2), mar = c(8, 5, 4, 2))

    # Plot 1: M1/M2 ratio boxplot
    boxplot(m1_m2_ratio ~ group_info,
           main = "M1/M2 Macrophage Ratio",
           ylab = "M1/M2 Ratio",
           col = c("#4DBBD5", "#E64B35"),
           las = 2, cex.main = 1.5, font.main = 2)
    ratio_test <- t.test(m1_m2_ratio[group_info == "SLE"],
                        m1_m2_ratio[group_info == "Control"])
    mtext(sprintf("P = %.2e", ratio_test$p.value), side = 3, line = 0.5,
         cex = 1.0, col = ifelse(ratio_test$p.value < 0.05, "red", "black"))

    # Plot 2: M1 scores
    boxplot(m1_scores ~ group_info,
           main = "M1 Macrophages",
           ylab = "Enrichment Score",
           col = c("#4DBBD5", "#DC0000"),
           las = 2, cex.main = 1.5, font.main = 2)
    m1_pval <- immune_stats[grepl("Macrophages_M1", immune_stats$CellType), "adj.P.Val"]
    if (length(m1_pval) > 0) {
      mtext(sprintf("P = %.2e", m1_pval), side = 3, line = 0.5, cex = 1.0,
           col = ifelse(m1_pval < 0.05, "red", "black"))
    }

    # Plot 3: M2 scores
    boxplot(m2_scores ~ group_info,
           main = "M2 Macrophages",
           ylab = "Enrichment Score",
           col = c("#4DBBD5", "#00A087"),
           las = 2, cex.main = 1.5, font.main = 2)
    m2_pval <- immune_stats[grepl("Macrophages_M2", immune_stats$CellType), "adj.P.Val"]
    if (length(m2_pval) > 0) {
      mtext(sprintf("P = %.2e", m2_pval), side = 3, line = 0.5, cex = 1.0,
           col = ifelse(m2_pval < 0.05, "red", "black"))
    }

    # Plot 4: Scatter plot M1 vs M2
    plot(m2_scores, m1_scores,
        col = ifelse(group_info == "SLE", "#E64B35", "#4DBBD5"),
        pch = 19, cex = 1.5,
        xlab = "M2 Macrophage Score",
        ylab = "M1 Macrophage Score",
        main = "M1 vs M2 Macrophage Balance",
        cex.main = 1.5, font.main = 2)
    abline(a = 0, b = 1, lty = 2, col = "gray", lwd = 2)
    legend("topleft", legend = c("Control", "SLE"),
          col = c("#4DBBD5", "#E64B35"), pch = 19, cex = 1.2, bty = "n")

    dev.off()

    log_message("Created M1/M2 macrophage comparison plot")
  }

  # logFC barplot with special highlighting for M1/M2
  cairo_pdf(file.path(out_dir, "Immune_logFC_Barplot.pdf"), width = 14, height = 10, family = "Arial")
  immune_stats_sorted <- immune_stats[order(immune_stats$logFC), ]
  immune_stats_sorted$CellType <- factor(immune_stats_sorted$CellType, levels = immune_stats_sorted$CellType)

  # Clean up cell type names
  immune_stats_sorted$CellType_Display <- gsub("_", " ", as.character(immune_stats_sorted$CellType))
  immune_stats_sorted$CellType_Display <- gsub("Macrophages M1", "M1 Macrophages", immune_stats_sorted$CellType_Display)
  immune_stats_sorted$CellType_Display <- gsub("Macrophages M2", "M2 Macrophages", immune_stats_sorted$CellType_Display)
  immune_stats_sorted$CellType_Display <- factor(immune_stats_sorted$CellType_Display,
                                                 levels = immune_stats_sorted$CellType_Display)

  # Assign colors based on cell type and significance
  immune_stats_sorted$Color <- sapply(seq_len(nrow(immune_stats_sorted)), function(i) {
    cell <- as.character(immune_stats_sorted$CellType[i])
    sig <- immune_stats_sorted$adj.P.Val[i] < 0.05
    if (grepl("Macrophages_M1", cell)) {
      return(ifelse(sig, "#DC0000", "#FFAAAA"))
    } else if (grepl("Macrophages_M2", cell)) {
      return(ifelse(sig, "#00A087", "#AAFFCC"))
    } else {
      return(ifelse(sig, "#E64B35", "#CCCCCC"))
    }
  })

  p <- ggplot(immune_stats_sorted, aes(x = CellType_Display, y = logFC)) +
    geom_bar(stat = "identity", fill = immune_stats_sorted$Color, color = "black", size = 0.3) +
    coord_flip() +
    labs(title = "Immune Cell Infiltration in SLE vs Control",
        subtitle = "M1 Macrophages (Red) | M2 Macrophages (Green) | Other (Orange/Gray)",
        x = "Cell Type", y = "log2(Fold Change: SLE/Control)") +
    theme_bw() +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold")
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.8) +
    # Add significance markers
    geom_text(data = immune_stats_sorted[immune_stats_sorted$adj.P.Val < 0.05, ],
             aes(label = ifelse(adj.P.Val < 0.001, "***",
                               ifelse(adj.P.Val < 0.01, "**", "*")),
                 y = ifelse(logFC > 0, logFC + 0.1, logFC - 0.1)),
             size = 5, fontface = "bold")

  print(p)
  dev.off()

  log_message("Immune infiltration plots created")
}

#' Additional immune infiltration analyses
#' Correlate immune infiltration with DEGs and checkpoints
create_immune_correlation_analyses <- function(immune_results, deg_results, checkpoint_results, merged_data, out_dir) {
  if (is.null(immune_results) || is.null(deg_results)) {
    return(NULL)
  }

  log_message("Creating immune infiltration correlation analyses...")

  gsva_scores <- immune_results$gsva_scores
  deg_sig <- deg_results$significant
  expr_matrix <- merged_data$expr_after
  group_info <- merged_data$group

  # 1. Immune cells vs DEG correlation
  if (nrow(deg_sig) > 0) {
    cairo_pdf(file.path(out_dir, "Immune_DEG_Correlation_Heatmap.pdf"), width = 14, height = 10, family = "Arial")

    # Select top DEGs
    top_degs <- head(rownames(deg_sig), min(50, nrow(deg_sig)))
    top_degs <- top_degs[top_degs %in% rownames(expr_matrix)]

    if (length(top_degs) > 0) {
      deg_expr <- expr_matrix[top_degs, ]

      # Calculate correlations
      cor_matrix <- cor(t(gsva_scores), t(deg_expr), use = "pairwise.complete.obs")

      # Clean row names
      rownames(cor_matrix) <- gsub("_", " ", rownames(cor_matrix))
      rownames(cor_matrix) <- gsub("Macrophages M1", "M1 Macrophages", rownames(cor_matrix))
      rownames(cor_matrix) <- gsub("Macrophages M2", "M2 Macrophages", rownames(cor_matrix))

      pheatmap(cor_matrix,
              color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
              breaks = seq(-1, 1, length.out = 101),
              cluster_rows = TRUE,
              cluster_cols = TRUE,
              show_colnames = TRUE,
              show_rownames = TRUE,
              fontsize_row = 9,
              fontsize_col = 7,
              main = "Correlation: Immune Cell Infiltration vs Top DEGs",
              angle_col = 90)
    }

    dev.off()
  }

  # 2. Immune cells vs Checkpoint genes correlation
  if (!is.null(checkpoint_results)) {
    cairo_pdf(file.path(out_dir, "Immune_Checkpoint_Correlation_Heatmap.pdf"), width = 14, height = 10, family = "Arial")

    checkpoint_expr <- checkpoint_results$expression

    if (ncol(checkpoint_expr) > 0 && nrow(checkpoint_expr) > 0) {
      # Calculate correlations
      cor_matrix <- cor(t(gsva_scores), t(checkpoint_expr), use = "pairwise.complete.obs")

      # Clean row names
      rownames(cor_matrix) <- gsub("_", " ", rownames(cor_matrix))
      rownames(cor_matrix) <- gsub("Macrophages M1", "M1 Macrophages", rownames(cor_matrix))
      rownames(cor_matrix) <- gsub("Macrophages M2", "M2 Macrophages", rownames(cor_matrix))

      pheatmap(cor_matrix,
              color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
              breaks = seq(-1, 1, length.out = 101),
              cluster_rows = TRUE,
              cluster_cols = TRUE,
              show_colnames = TRUE,
              show_rownames = TRUE,
              fontsize_row = 9,
              fontsize_col = 9,
              main = "Correlation: Immune Cell Infiltration vs Checkpoint Genes",
              angle_col = 90)
    }

    dev.off()
  }

  # 3. M1/M2 ratio correlation with key genes
  m1_idx <- which(grepl("Macrophages_M1", rownames(gsva_scores)))
  m2_idx <- which(grepl("Macrophages_M2", rownames(gsva_scores)))

  if (length(m1_idx) > 0 && length(m2_idx) > 0 && nrow(deg_sig) > 0) {
    cairo_pdf(file.path(out_dir, "M1_M2_Ratio_Gene_Correlation.pdf"), width = 14, height = 10, family = "Arial")

    m1_scores <- as.numeric(gsva_scores[m1_idx, ])
    m2_scores <- as.numeric(gsva_scores[m2_idx, ])
    m1_m2_ratio <- m1_scores / (m2_scores + 1e-6)

    # Select top DEGs for correlation
    top_degs <- head(rownames(deg_sig), 20)
    top_degs <- top_degs[top_degs %in% rownames(expr_matrix)]

    par(mfrow = c(4, 5), mar = c(5, 5, 4, 2))

    for (gene in top_degs) {
      gene_expr <- as.numeric(expr_matrix[gene, ])

      plot(m1_m2_ratio, gene_expr,
          pch = 21,
          bg = ifelse(group_info == "SLE", "#E64B35", "#4DBBD5"),
          col = "black",
          cex = 1.5,
          lwd = 1.2,
          xlab = "M1/M2 Ratio",
          ylab = sprintf("%s Expression", gene),
          main = gene,
          cex.main = 1.3)

      # Add regression line
      abline(lm(gene_expr ~ m1_m2_ratio), col = "darkgray", lwd = 2)

      # Add correlation
      cor_val <- cor(m1_m2_ratio, gene_expr, use = "complete.obs")
      cor_test <- cor.test(m1_m2_ratio, gene_expr)

      text(min(m1_m2_ratio, na.rm = TRUE) + 0.1 * diff(range(m1_m2_ratio, na.rm = TRUE)),
          max(gene_expr, na.rm = TRUE) - 0.1 * diff(range(gene_expr, na.rm = TRUE)),
          sprintf("r=%.2f\np=%.2e", cor_val, cor_test$p.value),
          cex = 0.9, adj = 0)
    }

    dev.off()
  }

  # 4. Immune infiltration summary across groups
  cairo_pdf(file.path(out_dir, "Immune_Infiltration_Summary.pdf"), width = 14, height = 10, family = "Arial")

  # Calculate mean infiltration scores for each cell type by group
  control_mean <- apply(gsva_scores[, group_info == "Control"], 1, mean, na.rm = TRUE)
  sle_mean <- apply(gsva_scores[, group_info == "SLE"], 1, mean, na.rm = TRUE)

  immune_summary <- data.frame(
    Cell_Type = rownames(gsva_scores),
    Control = control_mean,
    SLE = sle_mean,
    Diff = sle_mean - control_mean
  )

  immune_summary$Cell_Type <- gsub("_", " ", immune_summary$Cell_Type)
  immune_summary$Cell_Type <- gsub("Macrophages M1", "M1 Macrophages", immune_summary$Cell_Type)
  immune_summary$Cell_Type <- gsub("Macrophages M2", "M2 Macrophages", immune_summary$Cell_Type)

  immune_summary <- immune_summary[order(immune_summary$Diff), ]

  par(mfrow = c(1, 2), mar = c(12, 5, 4, 2))

  # Left: Grouped barplot
  barplot_data <- t(as.matrix(immune_summary[, c("Control", "SLE")]))
  barplot(barplot_data,
         beside = TRUE,
         col = c("#4DBBD5", "#E64B35"),
         names.arg = immune_summary$Cell_Type,
         las = 2,
         ylab = "Mean Enrichment Score",
         main = "Immune Cell Infiltration by Group",
          cex.main = 1.5,
         legend.text = c("Control", "SLE"),
         args.legend = list(x = "topright", bty = "n"))

  # Right: Difference barplot
  barplot(immune_summary$Diff,
         names.arg = immune_summary$Cell_Type,
         col = ifelse(immune_summary$Diff > 0, "#E64B35", "#4DBBD5"),
         border = "black",
         las = 2,
         ylab = "Difference (SLE - Control)",
         main = "Immune Cell Changes in SLE",
         cex.main = 1.5)
  abline(h = 0, lty = 2, lwd = 2)

  dev.off()

  log_message("Immune correlation analyses completed")
}

# ==============================================================================
# 12.5. IMMUNE CHECKPOINT ANALYSIS MODULE
# ==============================================================================

#' Analyze immune checkpoint gene expression
analyze_immune_checkpoints <- function(merged_data) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 9.5: Immune Checkpoint Gene Expression Analysis\n")
  cat("================================================================================\n\n")

  expr_matrix <- merged_data$expr_after
  group_info <- merged_data$group

  log_message("Analyzing immune checkpoint gene expression...")

  tryCatch({
    # Comprehensive immune checkpoint gene list
    checkpoint_genes <- list(
      # Inhibitory checkpoints (main targets for cancer immunotherapy)
      "PD-1/PD-L1 axis" = c("PDCD1", "CD274", "PDCD1LG2"),  # PD-1, PD-L1, PD-L2
      "CTLA-4 axis" = c("CTLA4", "CD80", "CD86"),
      "LAG-3 axis" = c("LAG3", "FGL1"),
      "TIM-3 axis" = c("HAVCR2", "LGALS9", "CEACAM1"),  # TIM-3, Galectin-9
      "TIGIT axis" = c("TIGIT", "PVR", "NECTIN2"),  # CD155, CD112

      # Other inhibitory checkpoints
      "VISTA" = c("VSIR"),
      "BTLA" = c("BTLA", "TNFRSF14"),  # HVEM
      "CD47-SIRPα" = c("CD47", "SIRPA"),
      "B7-H3/B7-H4" = c("CD276", "VTCN1"),  # B7-H3, B7-H4

      # Stimulatory checkpoints
      "CD28 family" = c("CD28", "ICOS", "TNFRSF4", "TNFRSF9", "TNFRSF18"),  # OX40, 4-1BB, GITR
      "CD40" = c("CD40", "CD40LG"),
      "TNFR family" = c("TNFRSF1A", "TNFRSF1B"),  # TNFR1, TNFR2

      # Emerging checkpoints
      "NKG2A/HLA-E" = c("KLRC1", "HLA-E"),  # NKG2A
      "LAIR-1" = c("LAIR1"),
      "Siglecs" = c("SIGLEC7", "SIGLEC9")
    )

    # Flatten gene list
    all_checkpoint_genes <- unique(unlist(checkpoint_genes))

    # Find genes present in expression data
    available_genes <- all_checkpoint_genes[all_checkpoint_genes %in% rownames(expr_matrix)]
    missing_genes <- setdiff(all_checkpoint_genes, available_genes)

    log_message(sprintf("Checkpoint genes found: %d/%d",
                       length(available_genes), length(all_checkpoint_genes)))
    if (length(missing_genes) > 0) {
      log_message(sprintf("Missing genes: %s", paste(head(missing_genes, 10), collapse = ", ")),
                 level = "WARNING")
    }

    if (length(available_genes) == 0) {
      log_message("No checkpoint genes found in expression data", level = "WARNING")
      return(NULL)
    }

    # Extract checkpoint gene expression
    checkpoint_expr <- expr_matrix[available_genes, , drop = FALSE]

    # Perform differential expression analysis
    checkpoint_stats <- data.frame()
    for (gene in available_genes) {
      control_vals <- checkpoint_expr[gene, group_info == "Control"]
      sle_vals <- checkpoint_expr[gene, group_info == "SLE"]

      test_result <- t.test(sle_vals, control_vals)

      checkpoint_stats <- rbind(checkpoint_stats, data.frame(
        Gene = gene,
        Mean_Control = mean(control_vals, na.rm = TRUE),
        Mean_SLE = mean(sle_vals, na.rm = TRUE),
        logFC = log2(mean(sle_vals, na.rm = TRUE) / mean(control_vals, na.rm = TRUE)),
        P.Value = test_result$p.value,
        stringsAsFactors = FALSE
      ))
    }

    checkpoint_stats$adj.P.Val <- p.adjust(checkpoint_stats$P.Value, method = "BH")
    checkpoint_stats$Regulation <- ifelse(checkpoint_stats$adj.P.Val < 0.05,
                                          ifelse(checkpoint_stats$logFC > 0, "Upregulated", "Downregulated"),
                                          "Not Significant")
    checkpoint_stats <- checkpoint_stats[order(checkpoint_stats$P.Value), ]

    log_message(sprintf("Significant checkpoint genes: %d (Up: %d, Down: %d)",
                       sum(checkpoint_stats$adj.P.Val < 0.05),
                       sum(checkpoint_stats$Regulation == "Upregulated"),
                       sum(checkpoint_stats$Regulation == "Downregulated")))

    # Create output directory
    out_dir <- file.path(CONFIG$output_dir, "Immune_Checkpoints")
    create_dir(out_dir)

    # Save results
    write.csv(checkpoint_stats,
             file.path(out_dir, "Checkpoint_Gene_Statistics.csv"),
             row.names = FALSE)

    # Create visualizations
    create_checkpoint_plots(checkpoint_expr, checkpoint_stats, group_info, out_dir)

    log_message("Immune checkpoint analysis completed")

    return(list(
      expression = checkpoint_expr,
      statistics = checkpoint_stats,
      gene_groups = checkpoint_genes
    ))

  }, error = function(e) {
    log_message(sprintf("Immune checkpoint analysis failed: %s", e$message), level = "ERROR")
    return(NULL)
  })
}

#' Create immune checkpoint visualization plots
create_checkpoint_plots <- function(checkpoint_expr, checkpoint_stats, group_info, out_dir) {
  log_message("Creating immune checkpoint visualization plots...")

  # 1. Checkpoint gene heatmap with disease grouping
  cairo_pdf(file.path(out_dir, "Checkpoint_Genes_Heatmap.pdf"), width = 14, height = 12, family = "Arial")

  tryCatch({
    # Sort columns by group
    group_order <- order(group_info)
    checkpoint_expr_sorted <- checkpoint_expr[, group_order]
    group_info_sorted <- group_info[group_order]

    # Calculate gap position
    n_control <- sum(group_info_sorted == "Control")
    gaps_col <- n_control

    # Remove genes with zero variance or non-finite values
    gene_var <- apply(checkpoint_expr_sorted, 1, var, na.rm = TRUE)
    valid_genes <- is.finite(gene_var) & gene_var > 0 &
                   !apply(checkpoint_expr_sorted, 1, function(x) any(!is.finite(x)))

    if (sum(valid_genes) == 0) {
      plot.new()
      text(0.5, 0.5, "No valid checkpoint genes for heatmap", cex = 1.2, col = "red")
    } else {
      checkpoint_expr_sorted <- checkpoint_expr_sorted[valid_genes, , drop = FALSE]
      genes_valid <- rownames(checkpoint_expr_sorted)

      # Assign gene categories
      gene_categories <- sapply(genes_valid, function(g) {
        if (g %in% c("PDCD1", "CD274", "PDCD1LG2")) return("PD-1/PD-L1")
        else if (g %in% c("CTLA4", "CD80", "CD86")) return("CTLA-4")
        else if (g %in% c("LAG3", "FGL1")) return("LAG-3")
        else if (g %in% c("HAVCR2", "LGALS9", "CEACAM1")) return("TIM-3")
        else if (g %in% c("TIGIT", "PVR", "NECTIN2")) return("TIGIT")
        else if (g %in% c("CD28", "ICOS", "TNFRSF4", "TNFRSF9", "TNFRSF18")) return("Stimulatory")
        else return("Other")
      })

      annotation_col <- data.frame(
        Disease_Group = group_info_sorted,
        row.names = colnames(checkpoint_expr_sorted)
      )

      annotation_row <- data.frame(
        Checkpoint_Type = gene_categories,
        Regulation = checkpoint_stats$Regulation[match(genes_valid, checkpoint_stats$Gene)],
        row.names = genes_valid
      )

      ann_colors <- list(
        Disease_Group = c(Control = "#4DBBD5", SLE = "#E64B35"),
        Checkpoint_Type = c(
          "PD-1/PD-L1" = "#E64B35",
          "CTLA-4" = "#4DBBD5",
          "LAG-3" = "#00A087",
          "TIM-3" = "#3C5488",
          "TIGIT" = "#F39B7F",
          "Stimulatory" = "#8491B4",
          "Other" = "#CCCCCC"
        ),
        Regulation = c(
          Upregulated = "red",
          Downregulated = "blue",
          "Not Significant" = "grey"
        )
      )

      pheatmap(checkpoint_expr_sorted,
              scale = "row",
              clustering_distance_rows = "correlation",
              clustering_distance_cols = "correlation",
              clustering_method = "ward.D2",
              annotation_col = annotation_col,
              annotation_row = annotation_row,
              annotation_colors = ann_colors,
              show_colnames = FALSE,
              show_rownames = TRUE,
              cluster_cols = FALSE,  # No clustering - strict Control vs SLE grouping
              gaps_col = gaps_col,
              fontsize_row = 9,
              color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
              main = sprintf("Immune Checkpoint Genes (%d genes)\nGrouped by Disease Status", sum(valid_genes)))
    }
  }, error = function(e) {
    plot.new()
    text(0.5, 0.5, paste("Error creating checkpoint heatmap:", e$message), cex = 1.2, col = "red")
  })

  dev.off()

  # 2. Boxplots for significant checkpoint genes
  sig_checkpoints <- rownames(checkpoint_stats[checkpoint_stats$adj.P.Val < 0.05, ])

  if (length(sig_checkpoints) > 0) {
    cairo_pdf(file.path(out_dir, "Checkpoint_Genes_Boxplots.pdf"), width = 16, height = 12, family = "Arial")

    n_sig <- min(length(sig_checkpoints), 20)  # Max 20 genes
    n_cols <- ceiling(sqrt(n_sig))
    n_rows <- ceiling(n_sig / n_cols)
    par(mfrow = c(n_rows, n_cols), mar = c(8, 5, 4, 2))

    for (i in 1:n_sig) {
      gene <- sig_checkpoints[i]
      expr_vals <- as.numeric(checkpoint_expr[gene, ])

      boxplot(expr_vals ~ group_info,
             main = gene,
             ylab = "Expression Level",
             col = c("#4DBBD5", "#E64B35"),
             las = 2,
             cex.main = 1.5,
             font.main = 2)

      pval <- checkpoint_stats[gene, "adj.P.Val"]
      logFC <- checkpoint_stats[gene, "logFC"]

      sig_label <- ifelse(pval < 0.001, "***",
                         ifelse(pval < 0.01, "**",
                               ifelse(pval < 0.05, "*", "")))

      mtext(sprintf("logFC=%.2f, P=%s %s", logFC,
                   ifelse(pval < 0.001, format(pval, scientific = TRUE, digits = 2),
                         sprintf("%.3f", pval)), sig_label),
           side = 3, line = 0.5, cex = 0.8, col = "red")
    }

    dev.off()
  }

  # 3. Volcano plot for checkpoint genes
  cairo_pdf(file.path(out_dir, "Checkpoint_Genes_Volcano.pdf"), width = 12, height = 10, family = "Arial")

  checkpoint_stats$gene_name <- checkpoint_stats$Gene
  sig_genes <- checkpoint_stats[checkpoint_stats$Regulation != "Not Significant", ]

  p <- ggplot(checkpoint_stats, aes(x = logFC, y = -log10(adj.P.Val), color = Regulation)) +
    geom_point(alpha = 0.8, size = 4) +
    geom_text_repel(data = sig_genes,
                   aes(label = Gene),
                   size = 4,
                   max.overlaps = 20,
                   box.padding = 0.5) +
    scale_color_manual(values = c("Upregulated" = "red",
                                  "Downregulated" = "blue",
                                  "Not Significant" = "grey")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "black") +
    labs(title = "Immune Checkpoint Genes: SLE vs Control",
        x = "log2(Fold Change)",
        y = "-log10(adjusted P-value)") +
    theme_bw() +
    theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
          legend.position = "right",
          axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "bold"))

  print(p)
  dev.off()

  # 4. Barplot for checkpoint gene expression
  cairo_pdf(file.path(out_dir, "Checkpoint_Genes_Barplot.pdf"), width = 14, height = 10, family = "Arial")

  checkpoint_stats_sorted <- checkpoint_stats[order(checkpoint_stats$logFC), ]
  checkpoint_stats_sorted$Gene <- factor(checkpoint_stats_sorted$Gene,
                                         levels = checkpoint_stats_sorted$Gene)

  checkpoint_stats_sorted$Color <- ifelse(checkpoint_stats_sorted$Regulation == "Upregulated", "#E64B35",
                                          ifelse(checkpoint_stats_sorted$Regulation == "Downregulated", "#4DBBD5", "#CCCCCC"))

  p <- ggplot(checkpoint_stats_sorted, aes(x = Gene, y = logFC)) +
    geom_bar(stat = "identity", fill = checkpoint_stats_sorted$Color, color = "black", size = 0.3) +
    coord_flip() +
    labs(title = "Immune Checkpoint Gene Expression in SLE",
        subtitle = "Red: Upregulated | Blue: Downregulated | Grey: Not Significant",
        x = "Checkpoint Gene",
        y = "log2(Fold Change: SLE/Control)") +
    theme_bw() +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold")
    ) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", size = 0.8) +
    geom_text(data = checkpoint_stats_sorted[checkpoint_stats_sorted$Regulation != "Not Significant", ],
             aes(label = ifelse(adj.P.Val < 0.001, "***",
                               ifelse(adj.P.Val < 0.01, "**", "*")),
                 y = ifelse(logFC > 0, logFC + 0.1, logFC - 0.1)),
             size = 5, fontface = "bold")

  print(p)
  dev.off()

  log_message("Immune checkpoint plots created")
}

# ==============================================================================
# 13. WGCNA MODULE-TRAIT CORRELATION MODULE
# ==============================================================================

#' Perform WGCNA module-trait analysis
perform_wgcna_analysis <- function(merged_data, deg_results = NULL) {
  cat("\n")
  cat("================================================================================\n")
  cat("  STEP 10: WGCNA Module-Trait Correlation Analysis\n")
  cat("================================================================================\n\n")

  tryCatch({
    suppressPackageStartupMessages({
      if (!require("WGCNA", quietly = TRUE)) {
        log_message("WGCNA package not installed, skipping WGCNA analysis", level = "WARNING")
        return(NULL)
      }
    })

    expr_matrix <- merged_data$expr_after
    group_info <- merged_data$group

    log_message("Preparing data for WGCNA...")
    datExpr <- t(expr_matrix)

    gsg <- goodSamplesGenes(datExpr, verbose = 3)
    if (!gsg$allOK) {
      log_message("Removing genes/samples with too many missing values...")
      if (sum(!gsg$goodGenes) > 0) datExpr <- datExpr[, gsg$goodGenes]
      if (sum(!gsg$goodSamples) > 0) {
        datExpr <- datExpr[gsg$goodSamples, ]
        group_info <- group_info[gsg$goodSamples]
      }
    }

    log_message(sprintf("WGCNA input: %d samples, %d genes", nrow(datExpr), ncol(datExpr)))

    log_message("Calculating soft-thresholding power...")
    powers <- c(1:10, seq(12, 20, by = 2))
    sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)

    softPower <- sft$powerEstimate
    if (is.na(softPower)) {
      softPower <- 6
      log_message("Using default soft power = 6", level = "WARNING")
    } else {
      log_message(sprintf("Selected soft power: %d", softPower))
    }

    log_message("Constructing co-expression network...")
    net <- blockwiseModules(datExpr, power = softPower, TOMType = "unsigned", minModuleSize = 30,
                           reassignThreshold = 0, mergeCutHeight = 0.25, numericLabels = TRUE,
                           pamRespectsDendro = FALSE, saveTOMs = FALSE, verbose = 3)

    moduleLabels <- net$colors
    moduleColors <- labels2colors(moduleLabels)
    names(moduleColors) <- colnames(datExpr)   # 确保基因名保留（labels2colors不保证）
    log_message(sprintf("Identified %d modules", length(unique(moduleLabels))))

    # Create trait data with both Control and SLE groups
    trait <- data.frame(
      Control = as.numeric(group_info == "Control"),
      SLE = as.numeric(group_info == "SLE")
    )
    rownames(trait) <- rownames(datExpr)

    MEs <- moduleEigengenes(datExpr, moduleColors)$eigengenes
    MEs <- orderMEs(MEs)

    log_message("Calculating module-trait correlations...")
    moduleTraitCor <- cor(MEs, trait, use = "p")
    moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

    out_dir <- file.path(CONFIG$output_dir, "WGCNA")
    create_dir(out_dir)

    # Save soft threshold selection plot
    tryCatch({
      # Load extrafont for proper font embedding
      suppressWarnings({
        if (requireNamespace("extrafont", quietly = TRUE)) {
          library(extrafont)
          loadfonts(device = "pdf", quiet = TRUE)
        }
      })

      cairo_pdf(file.path(out_dir, "Soft_Threshold_Selection.pdf"),
                width = 12, height = 6, family = "Arial")

      par(mfrow = c(1, 2))
      par(mar = c(5, 5, 4, 2))

      # Plot 1: Scale-free topology fit index
      plot(sft$fitIndices[, 1],
           -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2],
           xlab = "Soft Threshold (power)",
           ylab = "Scale Free Topology Model Fit, signed R²",
           type = "n",
           main = "Scale Independence",
           cex.lab = 1.2,
           cex.axis = 1.1,
           cex.main = 1.3)

      text(sft$fitIndices[, 1],
           -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2],
           labels = powers,
           cex = 0.9,
           col = "red")

      # Add threshold line at R^2 = 0.85
      abline(h = 0.85, col = "red", lty = 2)

      # Plot 2: Mean connectivity
      plot(sft$fitIndices[, 1],
           sft$fitIndices[, 5],
           xlab = "Soft Threshold (power)",
           ylab = "Mean Connectivity",
           type = "n",
           main = "Mean Connectivity",
           cex.lab = 1.2,
           cex.axis = 1.1,
           cex.main = 1.3)

      text(sft$fitIndices[, 1],
           sft$fitIndices[, 5],
           labels = powers,
           cex = 0.9,
           col = "red")

      dev.off()

      # Save soft threshold results as CSV
      write.csv(sft$fitIndices, file.path(out_dir, "Soft_Threshold_Results.csv"), row.names = FALSE)

      log_message("Created Soft_Threshold_Selection.pdf")

    }, error = function(e) {
      if (dev.cur() > 1) dev.off()  # Ensure PDF device is closed even on error
      log_message(sprintf("Error creating soft threshold plot: %s", e$message), level = "ERROR")
    })

    modNames <- substring(names(MEs), 3)
    geneModuleMembership <- as.data.frame(cor(datExpr, MEs, use = "p"))
    MMPvalue <- as.data.frame(corPvalueStudent(as.matrix(geneModuleMembership), nrow(datExpr)))
    names(geneModuleMembership) <- paste("MM", modNames, sep = "")
    names(MMPvalue) <- paste("p.MM", modNames, sep = "")

    geneTraitSignificance <- as.data.frame(cor(datExpr, trait, use = "p"))
    GSPvalue <- as.data.frame(corPvalueStudent(as.matrix(geneTraitSignificance), nrow(datExpr)))
    names(geneTraitSignificance) <- paste("GS.", names(trait), sep = "")
    names(GSPvalue) <- paste("p.GS.", names(trait), sep = "")

    gene_info <- data.frame(Gene = colnames(datExpr), Module = moduleColors,
                           geneTraitSignificance, GSPvalue, geneModuleMembership, MMPvalue)
    write.csv(gene_info, file.path(out_dir, "Gene_Module_Membership.csv"), row.names = FALSE)

    module_trait_table <- data.frame(Module = rownames(moduleTraitCor),
                                     Correlation = moduleTraitCor[, 1],
                                     P.Value = moduleTraitPvalue[, 1])
    module_trait_table <- module_trait_table[order(abs(module_trait_table$Correlation), decreasing = TRUE), ]
    write.csv(module_trait_table, file.path(out_dir, "Module_Trait_Correlation.csv"), row.names = FALSE)

    sig_modules <- module_trait_table[module_trait_table$P.Value < 0.05, "Module"]
    if (length(sig_modules) > 0) {
      hub_genes <- data.frame()
      for (mod in sig_modules) {
        mod_genes <- gene_info[gene_info$Module == substring(mod, 3), ]
        mod_genes <- mod_genes[order(abs(mod_genes[[paste0("MM", substring(mod, 3))]]), decreasing = TRUE), ]
        top_hub <- head(mod_genes, 20)
        hub_genes <- rbind(hub_genes, top_hub)
      }
      write.csv(hub_genes, file.path(out_dir, "Hub_Genes_Significant_Modules.csv"), row.names = FALSE)
    }

    # Analyze toxic targets and DEGs in modules
    log_message("Analyzing toxic targets and DEGs in co-expression modules...")
    toxic_targets <- get_imidazoquinoline_targets()
    toxic_in_network <- toxic_targets[toxic_targets %in% gene_info$Gene]
    log_message(sprintf("Found %d toxic targets in WGCNA network", length(toxic_in_network)))

    if (length(toxic_in_network) > 0) {
      toxic_gene_info <- gene_info[gene_info$Gene %in% toxic_in_network, ]
      write.csv(toxic_gene_info, file.path(out_dir, "Toxic_Targets_Module_Membership.csv"), row.names = FALSE)

      # Summarize toxic targets by module
      toxic_module_summary <- as.data.frame(table(toxic_gene_info$Module))
      names(toxic_module_summary) <- c("Module", "Toxic_Target_Count")
      toxic_module_summary <- toxic_module_summary[order(toxic_module_summary$Toxic_Target_Count, decreasing = TRUE), ]
      write.csv(toxic_module_summary, file.path(out_dir, "Toxic_Targets_Module_Summary.csv"), row.names = FALSE)
    }

    if (!is.null(deg_results)) {
      deg_genes <- rownames(deg_results)[deg_results$Change != "NS"]
      deg_in_network <- deg_genes[deg_genes %in% gene_info$Gene]
      log_message(sprintf("Found %d DEGs in WGCNA network", length(deg_in_network)))

      if (length(deg_in_network) > 0) {
        deg_gene_info <- gene_info[gene_info$Gene %in% deg_in_network, ]
        deg_gene_info$Change <- deg_results[deg_gene_info$Gene, "Change"]
        write.csv(deg_gene_info, file.path(out_dir, "DEGs_Module_Membership.csv"), row.names = FALSE)

        # Summarize DEGs by module
        deg_module_summary <- as.data.frame(table(deg_gene_info$Module))
        names(deg_module_summary) <- c("Module", "DEG_Count")
        deg_module_summary <- deg_module_summary[order(deg_module_summary$DEG_Count, decreasing = TRUE), ]
        write.csv(deg_module_summary, file.path(out_dir, "DEGs_Module_Summary.csv"), row.names = FALSE)
      }

      # Find overlap between toxic targets and DEGs
      toxic_deg_overlap <- intersect(toxic_in_network, deg_in_network)
      if (length(toxic_deg_overlap) > 0) {
        log_message(sprintf("Found %d genes that are both toxic targets and DEGs", length(toxic_deg_overlap)))
        overlap_gene_info <- gene_info[gene_info$Gene %in% toxic_deg_overlap, ]
        overlap_gene_info$Change <- deg_results[overlap_gene_info$Gene, "Change"]
        write.csv(overlap_gene_info, file.path(out_dir, "Toxic_DEG_Overlap_Module_Membership.csv"), row.names = FALSE)
      }
    }

    create_wgcna_plots(net, MEs, trait, moduleTraitCor, moduleTraitPvalue, moduleColors, datExpr, out_dir, gene_info, toxic_in_network, deg_in_network)

    log_message("WGCNA analysis completed")

    return(list(net = net, moduleColors = moduleColors, MEs = MEs,
               moduleTraitCor = moduleTraitCor, moduleTraitPvalue = moduleTraitPvalue,
               gene_info = gene_info, module_trait_table = module_trait_table))

  }, error = function(e) {
    log_message(sprintf("WGCNA analysis failed: %s", e$message), level = "ERROR")
    return(NULL)
  })
}

#' Create WGCNA plots (Publication-quality)
create_wgcna_plots <- function(net, MEs, trait, moduleTraitCor, moduleTraitPvalue, moduleColors, datExpr, out_dir, gene_info = NULL, toxic_targets = NULL, deg_genes = NULL) {
  log_message("Creating publication-quality WGCNA plots...")

  # 1. High-quality Gene Dendrogram with Module Colors
  cairo_pdf(file.path(out_dir, "Module_Dendrogram.pdf"), width = 14, height = 8, family = "Arial")
  par(mar = c(2, 4, 2, 2), cex = 0.8)
  plotDendroAndColors(net$dendrograms[[1]],
                     moduleColors[net$blockGenes[[1]]],
                     "Modules",
                     dendroLabels = FALSE,
                     hang = 0.03,
                     addGuide = TRUE,
                     guideHang = 0.05,
                     main = "Gene Co-expression Network: Hierarchical Clustering and Module Assignment",
                     cex.main = 1.4,
                     cex.colorLabels = 1.2,
                     cex.dendroLabels = 0.6,
                     marAll = c(1, 4, 3, 1))
  dev.off()

  # 2. Publication-quality Module-Trait Heatmap
  cairo_pdf(file.path(out_dir, "Module_Trait_Heatmap.pdf"), width = 10, height = 12, family = "Arial")

  # Calculate module sizes for annotation
  module_sizes <- table(moduleColors)
  module_names <- names(MEs)
  module_labels <- sapply(module_names, function(x) {
    color <- gsub("ME", "", x)
    size <- module_sizes[color]
    sprintf("%s\n(n=%d)", color, size)
  })

  # Create text matrix with correlation and p-value
  textMatrix <- matrix("", nrow = nrow(moduleTraitCor), ncol = ncol(moduleTraitCor))
  for (i in 1:nrow(moduleTraitCor)) {
    for (j in 1:ncol(moduleTraitCor)) {
      cor_val <- moduleTraitCor[i, j]
      p_val <- moduleTraitPvalue[i, j]

      # Significance stars
      stars <- ""
      if (p_val < 0.001) stars <- "***"
      else if (p_val < 0.01) stars <- "**"
      else if (p_val < 0.05) stars <- "*"

      textMatrix[i, j] <- sprintf("%.2f\n%s", cor_val, stars)
    }
  }

  par(mar = c(8, 12, 4, 4), cex = 1.0)
  labeledHeatmap(Matrix = moduleTraitCor,
                xLabels = names(trait),
                yLabels = module_labels,
                ySymbols = module_labels,
                colorLabels = FALSE,
                colors = colorRampPalette(c("#2166AC", "#F7F7F7", "#B2182B"))(100),
                textMatrix = textMatrix,
                setStdMargins = FALSE,
                cex.text = 1.1,
                cex.lab.x = 1.3,
                cex.lab.y = 1.0,
                zlim = c(-1, 1),
                main = "Module-Trait Associations in SLE\n(*p<0.05, **p<0.01, ***p<0.001)",
                cex.main = 1.3)
  dev.off()

  # 3. Enhanced Eigengene Network with better visualization
  cairo_pdf(file.path(out_dir, "Eigengene_Network.pdf"), width = 14, height = 7, family = "Arial")
  par(mfrow = c(1, 2), mar = c(1, 1, 4, 1))

  # Left: Dendrogram
  MEDiss <- 1 - cor(MEs, use = "p")
  METree <- hclust(as.dist(MEDiss), method = "average")
  plot(METree,
       main = "Module Eigengene Dendrogram",
       xlab = "", sub = "",
       cex.main = 1.5, cex.lab = 1.2)
  abline(h = 0.25, col = "red", lty = 2, lwd = 2)

  # Right: Heatmap
  plotEigengeneNetworks(MEs,
                       "",
                       marDendro = c(0, 4, 2, 0),
                       marHeatmap = c(4, 6, 2, 2),
                       plotDendrograms = FALSE,
                       xLabelsAngle = 90)
  mtext("Module Eigengene Correlation", side = 3, line = 1, cex = 1.3, font = 2)
  dev.off()

  # 4. Module Gene Count Bar Plot
  cairo_pdf(file.path(out_dir, "Module_Size_Distribution.pdf"), width = 12, height = 8, family = "Arial")
  module_table <- sort(table(moduleColors), decreasing = TRUE)
  module_df <- data.frame(
    Module = names(module_table),
    Count = as.numeric(module_table),
    stringsAsFactors = FALSE
  )
  module_df$Module <- factor(module_df$Module, levels = module_df$Module)

  # Get correlation for coloring
  module_cors <- sapply(module_df$Module, function(m) {
    me_name <- paste0("ME", m)
    if (me_name %in% rownames(moduleTraitCor)) {
      return(moduleTraitCor[me_name, 1])
    } else {
      return(0)
    }
  })

  module_df$Correlation <- module_cors
  module_df$Significant <- sapply(module_df$Module, function(m) {
    me_name <- paste0("ME", m)
    if (me_name %in% rownames(moduleTraitPvalue)) {
      return(moduleTraitPvalue[me_name, 1] < 0.05)
    } else {
      return(FALSE)
    }
  })

  par(mar = c(8, 5, 4, 2), cex = 1.0)
  barplot(module_df$Count,
          names.arg = module_df$Module,
          col = ifelse(module_df$Significant, module_df$Module, "gray90"),
          border = "black",
          las = 2,
          main = "Gene Distribution Across Modules\n(Colored modules are significantly associated with SLE)",
          ylab = "Number of Genes",
          xlab = "",
          cex.main = 1.3,
          cex.lab = 1.2,
          cex.names = 1.0)

  # Add correlation values on top
  text(x = seq_along(module_df$Count),
       y = module_df$Count + max(module_df$Count) * 0.03,
       labels = sprintf("r=%.2f", module_df$Correlation),
       srt = 90, adj = 0, xpd = TRUE, cex = 0.8)
  dev.off()

  # 5. Module-Trait Scatter Plots for Significant Modules
  sig_modules <- rownames(moduleTraitCor)[moduleTraitPvalue[, 1] < 0.05]

  if (length(sig_modules) > 0) {
    cairo_pdf(file.path(out_dir, "Module_Trait_Scatterplots.pdf"), width = 14, height = 10, family = "Arial")

    n_sig <- min(length(sig_modules), 9)  # Max 9 plots
    n_cols <- ceiling(sqrt(n_sig))
    n_rows <- ceiling(n_sig / n_cols)
    par(mfrow = c(n_rows, n_cols), mar = c(5, 5, 4, 2))

    for (i in 1:n_sig) {
      module <- sig_modules[i]
      module_color <- gsub("ME", "", module)
      ME_values <- MEs[, module]
      trait_values <- trait[, 1]

      cor_val <- moduleTraitCor[module, 1]
      p_val <- moduleTraitPvalue[module, 1]

      plot(trait_values, ME_values,
           pch = 21,
           bg = ifelse(trait_values == 1, "#E64B35", "#4DBBD5"),
           col = "black",
           cex = 1.8,
           lwd = 1.5,
           xlab = "Disease Status (0=Control, 1=SLE)",
           ylab = "Module Eigengene",
           main = sprintf("%s Module\n(r=%.2f, p=%.2e)",
                         module_color, cor_val, p_val),
           cex.main = 1.3,
           cex.lab = 1.2,
           cex.axis = 1.1)

      # Add regression line
      abline(lm(ME_values ~ trait_values), col = module_color, lwd = 3)

      # Add grid
      grid(col = "gray80", lty = 2)
    }
    dev.off()
  }

  # 6. Hub Gene Network Visualization for Top Module
  if (length(sig_modules) > 0) {
    tryCatch({
      cairo_pdf(file.path(out_dir, "Top_Module_Hub_Genes.pdf"), width = 12, height = 10, family = "Arial")

      top_module <- sig_modules[which.max(abs(moduleTraitCor[sig_modules, 1]))]
      module_color <- gsub("ME", "", top_module)

      # Get genes in this module
      module_genes <- names(moduleColors)[moduleColors == module_color]

      if (length(module_genes) > 0) {
        # Calculate gene significance and module membership
        GS_matrix <- cor(datExpr[, module_genes], trait, use = "p")
        # Use SLE column (second column) for gene significance
        GS <- as.vector(GS_matrix[, 2])
        names(GS) <- module_genes

        MM <- as.vector(cor(datExpr[, module_genes], MEs[, top_module], use = "p"))
        names(MM) <- module_genes

        # Plot
        plot(MM, GS,
             pch = 21,
             bg = module_color,
             col = "black",
             cex = 1.5,
             lwd = 1.2,
             xlab = sprintf("Module Membership in %s Module", module_color),
             ylab = "Gene Significance for SLE",
             main = sprintf("Hub Genes in %s Module\n(Top module associated with SLE, n=%d genes)",
                           module_color, length(module_genes)),
             cex.main = 1.3,
             cex.lab = 1.2,
             cex.axis = 1.1)

        # Add regression line (not diagonal)
        abline(lm(GS ~ MM), col = module_color, lwd = 2)

        # Add correlation
        cor_GS_MM <- cor(MM, GS, use = "p")
        text(min(MM) + 0.1, max(GS) - 0.05,
             sprintf("cor = %.3f", cor_GS_MM),
             cex = 1.3, font = 2)

        # Label top hub genes
        # Select top genes based on both MM and GS
        hub_score <- abs(MM) * abs(GS)
        top_n <- min(15, length(hub_score))
        top_genes <- names(sort(hub_score, decreasing = TRUE)[1:top_n])

        if (length(top_genes) > 0) {
          hub_idx <- match(top_genes, names(MM))
          text(MM[hub_idx], GS[hub_idx], labels = top_genes,
               pos = 4, cex = 0.7, col = "black", font = 2)
        }

        # Add quadrant lines
        abline(h = 0, v = 0, col = "gray70", lty = 2)

        grid(col = "gray80", lty = 2)
      } else {
        # No genes found in module - create placeholder plot
        plot.new()
        text(0.5, 0.5, sprintf("No genes found in %s module", module_color),
             cex = 1.5, col = "red")
      }

      dev.off()
      log_message(sprintf("Created Top_Module_Hub_Genes.pdf for %s module", module_color))

    }, error = function(e) {
      dev.off()  # Ensure PDF device is closed even on error
      log_message(sprintf("Error creating Top_Module_Hub_Genes plot: %s", e$message), level = "ERROR")
    })
  }

  # 7. Toxic Targets and DEGs Distribution in Modules
  if (!is.null(gene_info) && (!is.null(toxic_targets) || !is.null(deg_genes))) {
    tryCatch({
      cairo_pdf(file.path(out_dir, "Toxic_DEG_Module_Distribution.pdf"), width = 14, height = 10, family = "Arial")
      par(mfrow = c(2, 1), mar = c(5, 5, 4, 2))

      # Plot toxic targets distribution
      if (!is.null(toxic_targets) && length(toxic_targets) > 0) {
        toxic_gene_info <- gene_info[gene_info$Gene %in% toxic_targets, ]
        if (nrow(toxic_gene_info) > 0) {
          toxic_module_counts <- table(toxic_gene_info$Module)
          toxic_module_counts <- sort(toxic_module_counts, decreasing = TRUE)

          barplot(toxic_module_counts,
                  col = names(toxic_module_counts),
                  border = "black",
                  main = sprintf("Toxic Targets Distribution in Modules\n(Total: %d targets in network)", length(toxic_targets)),
                  xlab = "Module",
                  ylab = "Number of Toxic Targets",
                  las = 2,
                  cex.main = 1.3,
                  cex.lab = 1.2,
                  cex.axis = 1.0)
          grid(col = "gray80", lty = 2, lwd = 1)
        }
      }

      # Plot DEGs distribution
      if (!is.null(deg_genes) && length(deg_genes) > 0) {
        deg_gene_info <- gene_info[gene_info$Gene %in% deg_genes, ]
        if (nrow(deg_gene_info) > 0) {
          deg_module_counts <- table(deg_gene_info$Module)
          deg_module_counts <- sort(deg_module_counts, decreasing = TRUE)

          barplot(deg_module_counts,
                  col = names(deg_module_counts),
                  border = "black",
                  main = sprintf("DEGs Distribution in Modules\n(Total: %d DEGs in network)", length(deg_genes)),
                  xlab = "Module",
                  ylab = "Number of DEGs",
                  las = 2,
                  cex.main = 1.3,
                  cex.lab = 1.2,
                  cex.axis = 1.0)
          grid(col = "gray80", lty = 2, lwd = 1)
        }
      }

      dev.off()
      log_message("Created Toxic_DEG_Module_Distribution.pdf")

    }, error = function(e) {
      dev.off()
      log_message(sprintf("Error creating Toxic_DEG_Module_Distribution plot: %s", e$message), level = "ERROR")
    })
  }

  log_message("Publication-quality WGCNA plots created")
}

# ==============================================================================
# 14. SUMMARY REPORT GENERATION
# ==============================================================================

#' Generate comprehensive summary report
generate_summary_report <- function(gse_list, merged_data, deg_results, enrichment_results,
                                   target_validation = NULL, immune_results = NULL, checkpoint_results = NULL, wgcna_results = NULL) {
  cat("\n")
  cat("================================================================================\n")
  cat("  ANALYSIS SUMMARY REPORT\n")
  cat("================================================================================\n\n")

  cat("Dataset Statistics:\n")
  cat(sprintf("  Total datasets loaded: %d\n", length(gse_list)))
  cat(sprintf("  Total samples: %d\n", ncol(merged_data$expr_after)))
  cat(sprintf("    - SLE patients: %d\n", sum(merged_data$group == "SLE")))
  cat(sprintf("    - Controls: %d\n", sum(merged_data$group == "Control")))
  cat(sprintf("  Common genes analyzed: %d\n\n", nrow(merged_data$expr_after)))

  cat("Differential Expression:\n")
  cat(sprintf("  Upregulated genes: %d\n", deg_results$n_up))
  cat(sprintf("  Downregulated genes: %d\n", deg_results$n_down))
  cat(sprintf("  Total DEGs: %d\n\n", deg_results$n_total))

  if (!is.null(enrichment_results)) {
    cat("Functional Enrichment:\n")
    if (!is.null(enrichment_results$GO)) {
      for (ont in names(enrichment_results$GO)) {
        cat(sprintf("  GO %s: %d terms\n", ont, nrow(enrichment_results$GO[[ont]])))
      }
    }
    if (!is.null(enrichment_results$KEGG)) {
      cat(sprintf("  KEGG pathways: %d\n", nrow(enrichment_results$KEGG)))
    }
    if (!is.null(enrichment_results$GSEA)) {
      if (!is.null(enrichment_results$GSEA$GO)) {
        cat(sprintf("  GSEA GO: %d terms\n", nrow(enrichment_results$GSEA$GO)))
      }
      if (!is.null(enrichment_results$GSEA$KEGG)) {
        cat(sprintf("  GSEA KEGG: %d pathways\n", nrow(enrichment_results$GSEA$KEGG)))
      }
    }
    cat("\n")
  }

  if (!is.null(target_validation)) {
    cat("Imidazoquinoline-SLE Target Validation:\n")
    cat(sprintf("  Total targets: %d\n", length(target_validation$targets_all)))
    cat(sprintf("  Targets found in data: %d\n", length(target_validation$targets_present)))
    cat(sprintf("  Upregulated targets: %d\n",
               sum(target_validation$target_deg$Status == "Upregulated")))
    cat(sprintf("  Downregulated targets: %d\n",
               sum(target_validation$target_deg$Status == "Downregulated")))
    cat("\n")
  }

  if (!is.null(immune_results)) {
    cat("Immune Cell Infiltration:\n")
    cat(sprintf("  Cell types analyzed: %d\n", nrow(immune_results$statistics)))
    sig_cells <- immune_results$statistics[immune_results$statistics$adj.P.Val < 0.05, ]
    cat(sprintf("  Significantly different: %d\n", nrow(sig_cells)))
    cat("\n")
  }

  if (!is.null(wgcna_results)) {
    cat("WGCNA Module-Trait Correlation:\n")
    cat(sprintf("  Total modules identified: %d\n",
               length(unique(wgcna_results$moduleColors))))
    sig_mods <- wgcna_results$module_trait_table[
      wgcna_results$module_trait_table$P.Value < 0.05, ]
    cat(sprintf("  Significantly correlated modules: %d\n", nrow(sig_mods)))
    cat("\n")
  }

  cat(sprintf("All results saved to: %s/\n", CONFIG$output_dir))
  cat("\n")

  save.image(file.path(CONFIG$output_dir, "SLE_Complete_Analysis_v2.RData"))
  log_message("Complete analysis workspace saved")
}

# ==============================================================================
# MAIN ANALYSIS FLOW
# ==============================================================================

main <- function() {
  # Initialize
  init_directories()
  log_message("=== Starting SLE Analysis Pipeline v2.0 ===")
  log_message(sprintf("Analysis started at: %s", Sys.time()))

  # Step 1: Load data
  gse_list <- load_all_datasets()

  # Step 2: Extract sample groups
  gse_list <- extract_all_groups(gse_list)

  # Step 3: QC and normalization
  gse_list <- qc_normalize_all(gse_list)

  # Step 4: Probe to gene conversion
  gse_list <- convert_all_probes(gse_list)

  # Step 5: Merge and batch correction
  merged_data <- merge_and_batch_correct(gse_list)

  # Step 6: Differential expression analysis
  deg_results <- perform_deg_analysis(merged_data)

  # Step 7: Functional enrichment analysis
  enrichment_results <- perform_enrichment_analysis(deg_results, merged_data$expr_after)

  # Step 8: Imidazoquinoline-SLE target validation
  target_validation <- validate_target_expression(merged_data, deg_results)

  # Step 9: Immune cell infiltration analysis
  immune_results <- perform_immune_infiltration(merged_data)

  # Step 9.5: Immune checkpoint analysis
  checkpoint_results <- analyze_immune_checkpoints(merged_data)

  # Step 9.6: Immune-DEG-Checkpoint correlation analysis
  if (!is.null(immune_results) && !is.null(immune_results$output_dir)) {
    create_immune_correlation_analyses(immune_results, deg_results, checkpoint_results,
                                      merged_data, immune_results$output_dir)
  }

  # Step 10: WGCNA module-trait correlation
  wgcna_results <- perform_wgcna_analysis(merged_data, deg_results)

  log_message("=== Analysis pipeline completed ===")
  log_message(sprintf("Analysis ended at: %s", Sys.time()))

  # Generate summary report
  generate_summary_report(gse_list, merged_data, deg_results, enrichment_results,
                         target_validation, immune_results, checkpoint_results, wgcna_results)

  return(list(
    gse_list = gse_list,
    merged_data = merged_data,
    deg_results = deg_results,
    enrichment_results = enrichment_results,
    target_validation = target_validation,
    immune_results = immune_results,
    checkpoint_results = checkpoint_results,
    wgcna_results = wgcna_results
  ))
}

# ==============================================================================
# AUTO-RUN: Execute main function
# ==============================================================================

# This script will always run main() when sourced or executed
cat("\n")
cat("################################################################################\n")
cat("# Starting SLE Analysis Pipeline - This may take 30-60 minutes...            #\n")
cat("################################################################################\n\n")

# Run the analysis
analysis_results <- main()

cat("\n")
cat("################################################################################\n")
cat("# Analysis Complete! Check the output directory for results.                 #\n")
cat("################################################################################\n\n")


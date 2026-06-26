#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 5 - Step 3: Feature Importance and Pathway Analysis
==============================================================
This script:
1. Reconstructs graph data from test set
2. Extracts attention weights from trained GAT model
3. Computes feature importance scores
4. Performs pathway enrichment analysis
5. Generates visualizations for Figure 5C

Output files will be saved in: ./output/figure5C/
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Arial font for all plots
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set environment variables to control threading
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'output')          # feature_names, X_test, y_test, etc.
input_dir = os.path.join(base_dir, 'output', 'figure5B')  # model and graph files
output_dir = os.path.join(base_dir, 'output', 'figure5C')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Figure 5 - Step 3: Feature Importance and Pathway Analysis")
print("=" * 60)
print(f"\nData directory:   {data_dir}")
print(f"Model directory:  {input_dir}")
print(f"Output directory: {output_dir}")

# ==================== 1. Load Data ====================
print("\n" + "=" * 60)
print("1. Loading data from figure5B...")
print("=" * 60)

# Load feature names
feature_names_path = os.path.join(data_dir, 'feature_names.pkl')
try:
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Loaded {len(feature_names)} features")
except FileNotFoundError:
    print("✗ Could not find feature_names.pkl")
    sys.exit(1)

# Load gene info (if exists)
gene_info = None
gene_info_path = os.path.join(data_dir, 'gene_info.pkl')
if os.path.exists(gene_info_path):
    with open(gene_info_path, 'rb') as f:
        gene_info = pickle.load(f)
    print(f"✓ Loaded gene info")

# Load test data
X_test_path = os.path.join(data_dir, 'X_test.npy')
y_test_path = os.path.join(data_dir, 'y_test.npy')

try:
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    print(f"✓ Loaded test data: {X_test.shape[0]:,} cells, {X_test.shape[1]} features")
    print(f"  Disease ratio: {y_test.mean():.3f}")
except FileNotFoundError:
    print("✗ Could not find test data files")
    sys.exit(1)

# Load scaler if exists (for reference)
scaler_path = os.path.join(data_dir, 'scaler.pkl')
if os.path.exists(scaler_path):
    scaler = pickle.load(open(scaler_path, 'rb'))
    print(f"✓ Loaded scaler")
else:
    print("  No scaler found, using raw data")

# ==================== 2. Reconstruct Graph Data ====================
print("\n" + "=" * 60)
print("2. Reconstructing KNN graph for test set...")
print("=" * 60)

def build_knn_graph_efficient(X, k=15, batch_size=50000):
    """
    Build KNN graph efficiently using sklearn with batching
    """
    n_samples = X.shape[0]
    print(f"  Building graph for {n_samples:,} cells with k={k}")
    print(f"  Using batch_size={batch_size:,}")
    
    # Normalize data for cosine distance
    from sklearn.preprocessing import normalize
    X_norm = normalize(X, norm='l2')
    
    # Initialize NearestNeighbors
    nn = NearestNeighbors(
        n_neighbors=k+1, 
        metric='cosine', 
        algorithm='brute',  # Brute force is more memory-efficient for large data
        n_jobs=16  # Use multiple threads but not too many
    )
    nn.fit(X_norm)
    
    all_edges = []
    all_weights = []
    
    # Process in batches
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        print(f"  Processing batch {batch_idx+1}/{n_batches}: cells {start_idx:,}-{end_idx:,}")
        
        # Get batch
        batch = X_norm[start_idx:end_idx]
        
        # Find k+1 nearest neighbors (including self)
        distances, indices = nn.kneighbors(batch)
        
        # Build edges for this batch
        batch_edges = []
        batch_weights = []
        
        for i in range(len(batch)):
            global_i = start_idx + i
            for j in range(1, k+1):  # Start from 1 to exclude self
                global_j = indices[i, j]
                dist = distances[i, j]
                
                # Convert cosine distance to similarity weight
                # Cosine distance is in [0,2], convert to weight in [0,1]
                weight = 1.0 - (dist / 2.0)  # Normalize to [0,1]
                
                batch_edges.append([global_i, global_j])
                batch_weights.append(weight)
        
        all_edges.extend(batch_edges)
        all_weights.extend(batch_weights)
        
        print(f"    Batch edges: {len(batch_edges):,}")
    
    # Convert to tensors
    edge_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(all_weights, dtype=torch.float)
    
    print(f"  ✓ Graph built: {edge_index.size(1):,} edges")
    print(f"  Average degree: {edge_index.size(1)/n_samples:.1f}")
    
    return edge_index, edge_weights

# Check for precomputed graph
graph_candidates = [
    os.path.join(input_dir, 'test_graph.pt'),
    os.path.join(input_dir, 'test_graph_faiss.pt'),
    'test_graph.pt',
    'test_graph_faiss.pt'
]

edge_index = None
edge_weights = None

for graph_path in graph_candidates:
    if os.path.exists(graph_path):
        print(f"  Found precomputed graph: {graph_path}")
        try:
            graph_data = torch.load(graph_path)
            if isinstance(graph_data, dict):
                edge_index = graph_data.get('edge_index', graph_data.get('edges'))
                edge_weights = graph_data.get('edge_weights', graph_data.get('weights'))
            elif isinstance(graph_data, (tuple, list)) and len(graph_data) >= 2:
                edge_index, edge_weights = graph_data[0], graph_data[1]
            else:
                edge_index = graph_data
                edge_weights = torch.ones(edge_index.size(1))
            
            if edge_index is not None:
                print(f"  ✓ Loaded graph with {edge_index.size(1):,} edges")
                break
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

if edge_index is None:
    print("  No precomputed graph found, building from scratch...")
    edge_index, edge_weights = build_knn_graph_efficient(X_test, k=15)
    
    # Save for future use
    torch.save({
        'edge_index': edge_index,
        'edge_weights': edge_weights
    }, os.path.join(output_dir, 'test_graph.pt'))
    print(f"  ✓ Graph saved to {output_dir}/test_graph.pt")

# Create PyG data object
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Using device: {device}")

x_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_data = Data(
    x=x_test_tensor,
    edge_index=edge_index,
    edge_attr=edge_weights.unsqueeze(1),
    y=y_test_tensor
)
test_data = test_data.to(device)

print(f"  ✓ PyG data object created:")
print(f"    - Nodes: {test_data.num_nodes:,}")
print(f"    - Edges: {test_data.num_edges:,}")
print(f"    - Features: {test_data.num_features}")

# ==================== 3. Load Trained Model ====================
print("\n" + "=" * 60)
print("3. Loading trained GAT model...")
print("=" * 60)

# Define GAT model class - must match Step2's GATClassifier exactly
class GATClassifier(torch.nn.Module):
    """Graph Attention Network for disease prediction (matches Step2 architecture)"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=4, dropout=0.3):
        super().__init__()

        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             concat=True, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.embeddings = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        if return_attention:
            x1, attn1 = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
            x1 = F.elu(x1)
            x1 = self.dropout(x1)
            x2, attn2 = self.conv2(x1, edge_index, edge_attr, return_attention_weights=True)
            x2 = F.elu(x2)
            return self.fc(x2), (attn1, attn2)
        else:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index, edge_attr)
            x = F.elu(x)
            return self.fc(x)

# Find model file
model_candidates = [
    os.path.join(input_dir, 'gat_model_final.pt'),
    os.path.join(input_dir, 'best_model.pt'),
    'gat_model_final.pt',
    'best_model.pt'
]

model = None
for model_path in model_candidates:
    if os.path.exists(model_path):
        print(f"  Found model: {model_path}")
        try:
            model = GATClassifier(in_channels=50, hidden_channels=64, out_channels=2, heads=4, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"  ✓ Model loaded successfully")
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Parameters: {n_params:,}")
            break
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

if model is None:
    print("  ✗ Could not find trained model file")
    print("  Please ensure the model file exists in:")
    for path in model_candidates:
        print(f"    - {path}")
    sys.exit(1)

# ==================== 4. Extract Attention Weights ====================
print("\n" + "=" * 60)
print("4. Extracting attention weights from model...")
print("=" * 60)

with torch.no_grad():
    # Forward pass with attention weights
    output, (attn1, attn2) = model(test_data.x, test_data.edge_index, test_data.edge_attr, return_attention=True)
    
    # Get predictions
    predictions = F.softmax(output, dim=1)
    pred_classes = predictions.argmax(dim=1).cpu().numpy()
    pred_probs = predictions[:, 1].cpu().numpy()

# Extract attention weights
attn1_edge_index, attn1_weights = attn1
attn2_edge_index, attn2_weights = attn2

print(f"  Layer 1 attention:")
print(f"    - Shape: {attn1_weights.shape}")
print(f"    - Heads: {attn1_weights.size(1)}")
print(f"  Layer 2 attention:")
print(f"    - Shape: {attn2_weights.shape}")

# Calculate node-level attention importance
node_importance = torch.zeros(test_data.x.size(0), device=device)

# Sum attention weights for each node (as target)
for h in range(attn1_weights.size(1)):  # For each attention head
    node_importance.scatter_add_(0, attn1_edge_index[1], attn1_weights[:, h])

# Normalize importance scores
if node_importance.max() > 0:
    node_importance = node_importance / node_importance.max()
node_importance_np = node_importance.cpu().numpy()

print(f"  Node importance scores computed")
print(f"    - Mean: {node_importance_np.mean():.4f}")
print(f"    - Std:  {node_importance_np.std():.4f}")
print(f"    - Max:  {node_importance_np.max():.4f}")

# ==================== 5. Feature Importance Analysis ====================
print("\n" + "=" * 60)
print("5. Computing feature importance scores...")
print("=" * 60)

# Method 1: Attention-based feature importance
# Use top 10% most important nodes
top_node_threshold = np.percentile(node_importance_np, 90)
top_nodes = np.where(node_importance_np >= top_node_threshold)[0]
print(f"  Top 10% important nodes: {len(top_nodes):,}")

if len(top_nodes) > 0:
    # Get feature values for top nodes
    top_node_features = X_test[top_nodes]
    feature_importance_attn = top_node_features.mean(axis=0)
    
    # Normalize
    if feature_importance_attn.max() > 0:
        feature_importance_attn = feature_importance_attn / feature_importance_attn.max()
    print(f"  ✓ Attention-based importance computed")
else:
    feature_importance_attn = np.ones(len(feature_names)) / len(feature_names)

# Method 2: Gradient-based feature importance（在CPU上用小样本计算，避免GPU OOM）
def compute_gradient_importance(model, data, target_class=1, max_samples=5000):
    """Compute feature importance using input gradients on CPU with sampling"""
    import gc

    # 随机采样节点
    n = data.x.size(0)
    idx = torch.randperm(n)[:max_samples]
    print(f"    使用 {len(idx):,} 个节点（共 {n:,}）计算梯度重要度")

    # 把模型和采样数据移到CPU，避免GPU OOM
    cpu_model = model.cpu()
    cpu_model.eval()

    x_sub = data.x[idx].detach().cpu()
    # 对子图只用自环边（无需邻居），近似计算每个节点的局部梯度
    self_loops = torch.arange(len(idx), dtype=torch.long)
    edge_index_sub = torch.stack([self_loops, self_loops], dim=0)
    edge_attr_sub = torch.ones(len(idx), 1)

    x_sub.requires_grad_(True)
    cpu_model.zero_grad()

    output = cpu_model(x_sub, edge_index_sub, edge_attr_sub)
    loss = output[:, target_class].sum()
    loss.backward()

    importance = x_sub.grad.abs().mean(dim=0).numpy()
    if importance.max() > 0:
        importance = importance / importance.max()

    # 把模型移回原设备
    cpu_model.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    return importance

feature_importance_grad = compute_gradient_importance(model, test_data, max_samples=5000)
print(f"  ✓ 梯度重要度计算完成")

# Method 3: Simple correlation with predictions
# (as a lightweight alternative to permutation importance)
correlations = np.array([np.corrcoef(X_test[:, i], pred_probs)[0, 1] for i in range(X_test.shape[1])])
feature_importance_corr = np.abs(correlations)
if feature_importance_corr.max() > 0:
    feature_importance_corr = feature_importance_corr / feature_importance_corr.max()
print(f"  ✓ Correlation-based importance computed")

# Combine importance scores (ensemble)
feature_importance = (
    0.5 * feature_importance_attn + 
    0.3 * feature_importance_grad + 
    0.2 * feature_importance_corr
)
if feature_importance.max() > 0:
    feature_importance = feature_importance / feature_importance.max()

# Create feature importance dataframe
feature_df = pd.DataFrame({
    'feature': feature_names,
    'importance_attention': feature_importance_attn,
    'importance_gradient': feature_importance_grad,
    'importance_correlation': feature_importance_corr,
    'importance_combined': feature_importance
})
feature_df = feature_df.sort_values('importance_combined', ascending=False)

# Save feature importance
feature_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
print(f"\n  Top 20 most important features:")
print(feature_df.head(20).to_string(index=False))

# ==================== 6. Plot Top Features ====================
print("\n" + "=" * 60)
print("6. Generating top features plot...")
print("=" * 60)

# Plot top 30 features
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: Top features bar plot
ax = axes[0]
top_n = 30
top_features = feature_df.head(top_n)
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))

bars = ax.barh(range(top_n), top_features['importance_combined'].values, color=colors)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('Top 30 Most Important Features', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlim(0, 1.05)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['importance_combined'].values)):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=8)

# Right: Importance comparison across methods
ax = axes[1]
methods = ['Attention', 'Gradient', 'Correlation', 'Combined']
top10_features = feature_df.head(10)['feature'].values
importance_methods = feature_df[feature_df['feature'].isin(top10_features)][
    ['importance_attention', 'importance_gradient', 'importance_correlation', 'importance_combined']
].values

x = np.arange(len(top10_features))
width = 0.2
multiplier = 0

for i, method in enumerate(['attention', 'gradient', 'correlation', 'combined']):
    offset = width * multiplier
    rects = ax.bar(x + offset, importance_methods[:, i], width, label=method.capitalize())
    multiplier += 1

ax.set_xlabel('Features', fontsize=11)
ax.set_ylabel('Importance Score', fontsize=11)
ax.set_title('Importance Comparison (Top 10 Features)', fontsize=12, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(top10_features, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure5C_top_features.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Figure5C_top_features.pdf'), bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: Figure5C_top_features.png")

# ── HC vs SLE expression for top GNN features ─────────────────────────────────
print("  Generating HC vs SLE expression comparison for top features...")
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

_top20 = feature_df.head(20)['feature'].values
_feat_idx = [list(feature_names).index(f) for f in _top20 if f in list(feature_names)]
_feat_names = [feature_names[i] for i in _feat_idx]

_hc_mask  = y_test == 0
_sle_mask = y_test == 1

# Mann-Whitney + FDR for each feature
_pvals = []
_lfcs  = []
for i in _feat_idx:
    _h = X_test[_hc_mask, i]
    _s = X_test[_sle_mask, i]
    _, pv = mannwhitneyu(_s, _h, alternative='two-sided')
    _pvals.append(max(pv, np.finfo(float).tiny))
    _lfcs.append(_s.mean() - _h.mean())
_, _fdrs = fdrcorrection(_pvals)

# Violin plot grid: top 20 genes, 4 rows × 5 cols
_ncols, _nrows = 5, 4
fig_hs, axes_hs = plt.subplots(_nrows, _ncols, figsize=(4 * _ncols, 3.5 * _nrows))
axes_hs_flat = axes_hs.ravel()
for _k, (fi, fname, lfc, fdr) in enumerate(zip(_feat_idx, _feat_names, _lfcs, _fdrs)):
    ax = axes_hs_flat[_k]
    _data_hc  = X_test[_hc_mask, fi]
    _data_sle = X_test[_sle_mask, fi]
    vp = ax.violinplot([_data_hc, _data_sle], positions=[0, 1],
                       showmedians=True, showextrema=False)
    vp['bodies'][0].set_facecolor('#4393C3'); vp['bodies'][0].set_alpha(0.75)
    vp['bodies'][1].set_facecolor('#D6604D'); vp['bodies'][1].set_alpha(0.75)
    vp['cmedians'].set_color('black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HC', 'SLE'], fontsize=9)
    ax.set_ylabel('Expression', fontsize=8)
    sig = '***' if fdr < 0.001 else '**' if fdr < 0.01 else '*' if fdr < 0.05 else 'ns'
    ax.set_title(f'{fname}\nlogFC={lfc:+.2f}  FDR={fdr:.2e} {sig}',
                 fontsize=8, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
for _k in range(len(_feat_names), len(axes_hs_flat)):
    axes_hs_flat[_k].set_visible(False)

fig_hs.suptitle('Top GNN Features: HC vs SLE Expression (test set)',
                fontsize=13, fontweight='bold', y=1.01)
fig_hs.tight_layout()
fig_hs.savefig(os.path.join(output_dir, 'Figure5C_top_features_HCvsSLE.pdf'), bbox_inches='tight')
fig_hs.savefig(os.path.join(output_dir, 'Figure5C_top_features_HCvsSLE.pdf'), bbox_inches='tight')
plt.show()
print(f"  ✓ Saved: Figure5C_top_features_HCvsSLE.png")
print("\n" + "=" * 60)
print("7. Performing pathway enrichment analysis...")
print("=" * 60)

# Define gene sets - 针对髓系细胞/单核细胞特征的通路基因集
gene_sets = {
    'Myeloid Cell Identity': [
        'AIF1', 'CD68', 'SPI1', 'MNDA', 'TYROBP', 'FCER1G', 'MS4A6A',
        'CLEC7A', 'CLEC4A', 'FCN1', 'LYZ', 'VCAN', 'CST3', 'LST1',
        'CTSS', 'CTSD', 'CD14', 'CSF1R', 'ITGAM'],
    'Innate Immune Signaling': [
        'TYROBP', 'FCER1G', 'LYN', 'FGR', 'MAPK1', 'SPI1', 'FOS',
        'LY96', 'CARD16', 'MYD88', 'TLR4', 'IRAK1', 'TRAF6', 'NFKB1'],
    'Phagocytosis & Pattern Recognition': [
        'CLEC7A', 'CLEC4A', 'FCN1', 'CYBB', 'LYZ', 'CTSS', 'CTSD',
        'CD68', 'TYROBP', 'FCER1G', 'MSR1', 'MARCO', 'FCGR1A'],
    'Monocyte Inflammatory Response': [
        'AIF1', 'TNFSF13B', 'S100A8', 'S100A9', 'MNDA', 'CYBB',
        'FCN1', 'VCAN', 'CARD16', 'FOS', 'MAPK1', 'SPI1', 'LST1'],
    'Lysosomal & Proteolytic Activity': [
        'CTSS', 'CTSD', 'ASAH1', 'NPC2', 'CSTA', 'LGALS1', 'LGALS3',
        'LYZ', 'AP1S2', 'LAMP1', 'LAMP2', 'CTSB', 'CTSL'],
    'S100 Protein Family': [
        'S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A11', 'S100A12',
        'S100B', 'S100P'],
    'Iron & Heme Metabolism': [
        'FTL', 'FTH1', 'BLVRB', 'HMOX1', 'SLC40A1', 'TFRC', 'HBA1'],
    'Complement System': [
        'CFP', 'FCN1', 'LY96', 'C1QA', 'C1QB', 'C3', 'C5AR1',
        'VSIG4', 'CR1', 'SERPING1'],
    'Oxidative Burst (NOX)': [
        'CYBB', 'FGR', 'LYN', 'NCF1', 'NCF2', 'NCF4', 'RAC2',
        'CYBA', 'MPO'],
    'Monocyte Survival & Apoptosis': [
        'MCL1', 'FOS', 'MAPK1', 'TIMP1', 'BCL2A1', 'BIRC3',
        'TNFRSF1B', 'CASP1', 'CASP4', 'CASP5'],
    'Lipid & Cholesterol Metabolism': [
        'NPC2', 'ASAH1', 'TSPO', 'APOC1', 'APOE', 'LPL',
        'FABP4', 'FABP5', 'CH25H'],
    'Antigen Presentation (MHC II)': [
        'HLA-DRA', 'HLA-DRB1', 'HLA-DQB1', 'HLA-DPA1', 'HLA-DPB1',
        'CD74', 'CTSS', 'CTSD', 'CSTA'],
    'MAPK / AP-1 Signaling': [
        'MAPK1', 'FOS', 'FGR', 'LYN', 'SPI1', 'CARD16',
        'MAP2K1', 'DUSP1', 'DUSP6', 'JUN'],
    'Galectin & Lectin Signaling': [
        'LGALS1', 'LGALS3', 'CLEC7A', 'CLEC4A', 'LY96',
        'SIGLEC1', 'SIGLEC7', 'SIGLEC9'],
    'Parkinson / Neuroinflammation': [
        'LRRK2', 'MAPK1', 'FOS', 'TYROBP', 'AIF1', 'TSPO',
        'HMOX1', 'FTL', 'FTH1'],
}

# Calculate enrichment scores
enrichment_results = []
top_genes = feature_df['feature'].values  # 只有50个基因，全部纳入

print(f"  Testing {len(gene_sets)} pathway gene sets...")

for pathway, genes in gene_sets.items():
    # Find overlapping genes
    overlap = set(top_genes) & set(genes)
    overlap_size = len(overlap)

    if overlap_size >= 2:  # 50个特征中overlap≥2即可
        # Calculate enrichment score based on gene importance
        if overlap_size > 0:
            overlap_importance = feature_df[feature_df['feature'].isin(overlap)]['importance_combined'].mean()
            background_importance = feature_df['importance_combined'].mean()
            
            enrichment_score = overlap_importance / background_importance
            
            # 超几何检验：从全基因组（~20000基因）中随机选50个，
            # 问恰好有k个落在该通路的概率
            # M=全基因组大小, n=通路基因数, N=我们的特征数, k=overlap
            from scipy import stats
            GENOME_SIZE = 20000
            p_value = stats.hypergeom.sf(
                overlap_size - 1,
                GENOME_SIZE,      # M：总体（全基因组）
                len(genes),       # n：通路基因数
                len(top_genes)    # N：我们的特征集大小（50）
            )
            
            enrichment_results.append({
                'pathway': pathway,
                'overlap_genes': ','.join(list(overlap)[:10]),  # First 10 genes
                'overlap_size': overlap_size,
                'enrichment_score': enrichment_score,
                'p_value': p_value,
                '-log10(p_value)': -np.log10(p_value + 1e-10)
            })

if enrichment_results:
    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df = enrichment_df.sort_values('enrichment_score', ascending=False)
    
    # Save enrichment results
    enrichment_df.to_csv(os.path.join(output_dir, 'pathway_enrichment.csv'), index=False)
    
    print(f"\n  Top enriched pathways:")
    for i, row in enrichment_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['pathway']:<20} (score: {row['enrichment_score']:.2f}, overlap: {row['overlap_size']} genes, p={row['p_value']:.2e})")
else:
    print("  No significant pathway enrichment found")
    enrichment_df = pd.DataFrame()

# ==================== 8. Plot Enrichment Analysis ====================
print("\n" + "=" * 60)
print("8. Generating enrichment plots...")
print("=" * 60)

if len(enrichment_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Enrichment score bar plot
    ax = axes[0]
    top_pathways = enrichment_df.head(15).sort_values('enrichment_score', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_pathways)))
    bars = ax.barh(range(len(top_pathways)), top_pathways['enrichment_score'].values, color=colors)
    ax.set_yticks(range(len(top_pathways)))
    ax.set_yticklabels(top_pathways['pathway'].values)
    ax.set_xlabel('Enrichment Score', fontsize=11)
    ax.set_title('Top Enriched Pathways', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Add overlap size labels
    for i, (bar, row) in enumerate(zip(bars, top_pathways.iterrows())):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'n={row[1]["overlap_size"]}', va='center', fontsize=8)
    
    # Right: Dot plot of pathway enrichment
    ax = axes[1]
    top_pathways_dot = enrichment_df.head(15)
    
    # Size by overlap, color by -log10(p)
    scatter = ax.scatter(
        top_pathways_dot['enrichment_score'], 
        range(len(top_pathways_dot)),
        s=top_pathways_dot['overlap_size'] * 20,
        c=top_pathways_dot['-log10(p_value)'].clip(0, 10),
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.set_yticks(range(len(top_pathways_dot)))
    ax.set_yticklabels(top_pathways_dot['pathway'].values)
    ax.set_xlabel('Enrichment Score', fontsize=11)
    ax.set_title('Pathway Enrichment Dot Plot', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(p-value)', fontsize=9)
    
    # Add legend for size
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=5, label='n=5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='n=10'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=11, label='n=15')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure5C_pathway_enrichment.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Figure5C_pathway_enrichment.pdf'), bbox_inches='tight')
    plt.show()
    print(f"  ✓ Saved: Figure5C_pathway_enrichment.png")

# ==================== 9. Generate Summary Report ====================
print("\n" + "=" * 60)
print("9. Generating summary report...")
print("=" * 60)

report = f"""
============================================================
Figure 5C - Feature Importance and Pathway Analysis Summary
============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION
------------------
Number of cells (test set): {len(X_test):,}
Number of features: {len(feature_names)}
Disease prevalence: {y_test.mean():.3f}

GRAPH INFORMATION
-----------------
Number of nodes: {test_data.num_nodes:,}
Number of edges: {test_data.num_edges:,}
Average degree: {test_data.num_edges/test_data.num_nodes:.1f}
KNN parameter k: 15

MODEL INFORMATION
-----------------
Model type: GAT (Graph Attention Network)
Hidden dimension: 64
Attention heads: 4
Total parameters: {n_params:,}

TOP 10 MOST IMPORTANT FEATURES
------------------------------
"""
for i, row in feature_df.head(10).iterrows():
    report += f"{i+1:2d}. {row['feature']:<20} (importance: {row['importance_combined']:.4f})\n"

if len(enrichment_df) > 0:
    report += f"""
TOP ENRICHED PATHWAYS
---------------------
"""
    for i, row in enrichment_df.head(10).iterrows():
        report += f"{i+1:2d}. {row['pathway']:<25} (score: {row['enrichment_score']:.2f}, overlap: {row['overlap_size']} genes, p={row['p_value']:.2e})\n"

report += f"""
OUTPUT FILES
-----------
1. Figure5C_top_features.png - Bar plots of top important features
2. Figure5C_pathway_enrichment.png - Pathway enrichment analysis
3. feature_importance.csv - Complete feature importance scores
4. pathway_enrichment.csv - Pathway enrichment results
5. test_graph.pt - Reconstructed graph for test set
6. summary_report.txt - This summary report

INTERPRETATION
--------------
- Top features represent genes most influential in disease prediction
- Higher importance scores indicate stronger contribution to model decisions
- Enriched pathways highlight biological processes involved in disease
- Pathway enrichment score > 1 indicates over-representation of important genes

============================================================
"""

print(report)

# Save report
with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
    f.write(report)

print(f"\n✓ All results saved to: {output_dir}")
print("✓ Step 3 completed successfully!")
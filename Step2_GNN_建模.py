#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5 - Step 2: 图神经网络(GNN)建模
构建GAT模型，训练并评估敏感细胞分类性能
生成Figure 5B: GAT嵌入UMAP和混淆矩阵
"""

# ============ 重要：设置线程数限制，避免OpenBLAS崩溃 ============
# 服务器CPU核心数远超128，OpenBLAS编译上限是128线程，必须限制！
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'   # ← 必须设为小数，否则段错误
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
# ===============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Configure Arial font for publication
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts

import seaborn as sns
import umap
import pickle
import warnings
from pathlib import Path
import time
import gc

plt.rcParams.update({
    'font.family':     'Arial',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 定义路径
BASE_DIR = "/home/h3033/statics/GEO_data/GSE/figure5"
INPUT_DIR = os.path.join(BASE_DIR, "output")  # Step 1的输出作为输入
OUTPUT_DIR = os.path.join(BASE_DIR, "output/figure5B")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print("=" * 60)
print("Figure 5 - Step 2: 图神经网络(GNN)建模")
print("=" * 60)

# 检查CUDA可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# 1. 加载Step 1保存的数据
print("\n1. 加载预处理数据...")

X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
X_val = np.load(os.path.join(INPUT_DIR, "X_val.npy"))
X_test = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
y_val = np.load(os.path.join(INPUT_DIR, "y_val.npy"))
y_test = np.load(os.path.join(INPUT_DIR, "y_test.npy"))

with open(os.path.join(INPUT_DIR, "feature_names.pkl"), 'rb') as f:
    feature_names = pickle.load(f)

print(f"   训练集: {X_train.shape}, 正样本比例: {y_train.mean():.3f}")
print(f"   验证集: {X_val.shape}, 正样本比例: {y_val.mean():.3f}")
print(f"   测试集: {X_test.shape}, 正样本比例: {y_test.mean():.3f}")
print(f"   特征数: {len(feature_names)}")

# 数据标准化（对GNN很重要）
print("\n2. 数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 保存标准化器供后续使用
with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# 清理内存
print("\n   清理内存...")
n_features = X_train.shape[1]      # ← 保存特征数（删除前）
n_train = len(X_train)              # ← 保存训练集大小
n_val = len(X_val)                  # ← 保存验证集大小
n_test = len(X_test)                # ← 保存测试集大小
del X_train, X_val, X_test
gc.collect()

# 2. 构建KNN图
print("\n3. 构建KNN图...")

def build_knn_graph(X, k=15, metric='cosine', q_batch=2000, r_batch=50000):
    """
    纯PyTorch实现的精确KNN图构建（双批次循环，控制内存峰值）

    原理：对 query 和 reference 同时分批，每次只计算
          (q_batch × r_batch) 的相似度块，维护每个 query 的 topk。
    峰值内存 ≈ q_batch × r_batch × 4 bytes
              = 2000 × 50000 × 4 ≈ 400 MB（GPU显存）

    Args:
        X       : numpy 特征矩阵 (n_nodes, n_features)
        k       : 每个节点的邻居数
        metric  : 'cosine' 或 'euclidean'
        q_batch : 每批查询节点数（调小可降低显存）
        r_batch : 每批参考节点数（调小可降低显存）
    """
    print(f"   PyTorch KNN | {metric} | k={k} | q_batch={q_batch} | r_batch={r_batch}")
    n = len(X)

    # 全量特征常驻 CPU，避免显存不足
    X_cpu = torch.tensor(X, dtype=torch.float32)
    if metric == 'cosine':
        X_cpu = F.normalize(X_cpu, p=2, dim=1)

    all_rows, all_cols, all_weights = [], [], []
    n_q_batches = (n + q_batch - 1) // q_batch

    for qi, q_start in enumerate(range(0, n, q_batch)):
        q_end = min(q_start + q_batch, n)
        q_size = q_end - q_start
        q_vec = X_cpu[q_start:q_end].to(device)   # (q_size, d)

        # 维护当前 query batch 的 topk：初始化为最小值
        topk_vals = torch.full((q_size, k + 1), -1e9, device=device)
        topk_idx  = torch.zeros(q_size, k + 1, dtype=torch.long, device=device)

        for r_start in range(0, n, r_batch):
            r_end = min(r_start + r_batch, n)
            r_vec = X_cpu[r_start:r_end].to(device)   # (r_size, d)

            if metric == 'cosine':
                sim = torch.mm(q_vec, r_vec.t())       # (q_size, r_size)
            else:
                # 欧氏距离转负值（topk largest=True 时越近越好）
                diff = q_vec.unsqueeze(1) - r_vec.unsqueeze(0)
                sim  = -diff.pow(2).sum(-1).sqrt()

            # 全局列索引
            r_global = torch.arange(r_start, r_end, device=device)  # (r_size,)

            # 合并已有 topk 与本批 sim，取新 topk
            cat_vals = torch.cat([topk_vals, sim], dim=1)
            cat_idx  = torch.cat(
                [topk_idx,
                 r_global.unsqueeze(0).expand(q_size, -1)],
                dim=1
            )
            new_vals, rank = torch.topk(cat_vals, k + 1, dim=1, largest=True, sorted=True)
            topk_vals = new_vals
            topk_idx  = torch.gather(cat_idx, 1, rank)

            del r_vec, sim, cat_vals, cat_idx, r_global, rank, new_vals
            torch.cuda.empty_cache()

        # 取第 1: 列（跳过自身，自身一定在 topk[0]）
        src = torch.arange(q_start, q_end, device=device).unsqueeze(1).expand(-1, k)
        dst = topk_idx[:, 1:]

        if metric == 'cosine':
            w = topk_vals[:, 1:].clamp(0, 1)
        else:
            # sim = -dist → dist = -sim → weight = 1/(1+dist)
            w = 1.0 / (1.0 + (-topk_vals[:, 1:]).clamp(min=0))

        all_rows.append(src.reshape(-1).cpu())
        all_cols.append(dst.reshape(-1).cpu())
        all_weights.append(w.reshape(-1).cpu())

        del q_vec, topk_vals, topk_idx, src, dst, w
        torch.cuda.empty_cache()

        if (qi + 1) % max(1, n_q_batches // 10) == 0:
            print(f"   进度: {qi + 1}/{n_q_batches} 批")

    del X_cpu
    gc.collect()

    edge_index  = torch.stack([torch.cat(all_rows), torch.cat(all_cols)], dim=0).long()
    edge_weights = torch.cat(all_weights).float()

    print(f"   构建完成: {edge_index.shape[1]} 条边")
    return edge_index, edge_weights

# 由于数据集很大，我们对训练集构建图，并在验证/测试时使用相同的图结构
print("\n   为训练集构建KNN图...")
edge_index_train, edge_weights_train = build_knn_graph(
    X_train_scaled, k=15, metric='cosine', q_batch=2000, r_batch=50000
)
gc.collect()

# 创建PyG Data对象
train_data = Data(
    x=torch.tensor(X_train_scaled, dtype=torch.float),
    edge_index=edge_index_train,
    edge_attr=edge_weights_train,
    y=torch.tensor(y_train, dtype=torch.long)
)

# 对于验证集和测试集，我们需要构建它们自己的图
print("\n   为验证集构建KNN图...")
edge_index_val, edge_weights_val = build_knn_graph(
    X_val_scaled, k=15, metric='cosine', q_batch=2000, r_batch=50000
)
gc.collect()

print("\n   为测试集构建KNN图...")
edge_index_test, edge_weights_test = build_knn_graph(
    X_test_scaled, k=15, metric='cosine', q_batch=2000, r_batch=50000
)
gc.collect()

val_data = Data(
    x=torch.tensor(X_val_scaled, dtype=torch.float),
    edge_index=edge_index_val,
    edge_attr=edge_weights_val,
    y=torch.tensor(y_val, dtype=torch.long)
)

test_data = Data(
    x=torch.tensor(X_test_scaled, dtype=torch.float),
    edge_index=edge_index_test,
    edge_attr=edge_weights_test,
    y=torch.tensor(y_test, dtype=torch.long)
)

# 清理临时数据（提前计算pos_weight，训练时需要）
pos_weight_val = (len(train_data.y) - train_data.y.sum().item()) / train_data.y.sum().item()
del X_train_scaled, X_val_scaled, X_test_scaled
del edge_index_train, edge_index_val, edge_index_test
del edge_weights_train, edge_weights_val, edge_weights_test
del y_train, y_val, y_test
gc.collect()

# 3. 定义GAT模型
print("\n4. 定义GAT模型...")

class GATClassifier(nn.Module):
    """
    图注意力网络(GAT)用于节点分类
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, 
                 heads=4, dropout=0.3, use_gatv2=True):
        super().__init__()
        
        # 第一层GAT
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads, 
            concat=True, 
            dropout=dropout,
            edge_dim=1  # 使用边权重
        )
        
        # 第二层GAT
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout,
            edge_dim=1
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 存储用于可视化的嵌入
        self.embeddings = None
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 第一层GAT + ELU + Dropout
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 第二层GAT
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        
        # 保存嵌入用于可视化
        self.embeddings = x.detach().cpu().numpy()
        
        # 分类
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1), self.embeddings

# 初始化模型
model = GATClassifier(
    in_channels=n_features,
    hidden_channels=64,
    out_channels=2,
    heads=4,
    dropout=0.3
).to(device)

print(f"\n   模型结构:")
print(f"   输入维度: {n_features}")
print(f"   隐藏层维度: 64")
print(f"   注意力头数: 4")
print(f"   输出类别: 2")
print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 4. 训练模型
print("\n5. 训练GAT模型...")

def train_model(model, train_data, val_data, epochs=100, lr=0.001, weight_decay=5e-4, pos_weight=1.0, node_batch_size=50000):
    """
    Mini-batch 训练GAT模型（节点级子图采样），避免GPU OOM
    """
    # 处理类别不平衡
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=10)

    best_val_acc = 0
    best_model_state = None
    patience = 20
    patience_counter = 0

    train_losses = []
    val_accs = []

    n_train_nodes = train_data.x.shape[0]
    n_val_nodes = val_data.x.shape[0]

    for epoch in range(epochs):
        # ==================== Mini-batch 训练 ====================
        model.train()
        epoch_loss = 0
        n_batches = 0

        # 随机打乱训练节点
        node_indices = torch.randperm(n_train_nodes)

        for batch_start in range(0, n_train_nodes, node_batch_size):
            batch_end = min(batch_start + node_batch_size, n_train_nodes)
            batch_nodes = node_indices[batch_start:batch_end]

            # 构建当前batch的子图
            batch_node_mask = torch.zeros(n_train_nodes, dtype=torch.bool)
            batch_node_mask[batch_nodes] = True

            # 提取只连接到batch节点的边
            src, dst = train_data.edge_index
            edge_mask = batch_node_mask[src] & batch_node_mask[dst]
            batch_edge_index = train_data.edge_index[:, edge_mask]
            batch_edge_attr = train_data.edge_attr[edge_mask]

            # 创建子图
            subgraph_x = train_data.x[batch_nodes].to(device)
            subgraph_y = train_data.y[batch_nodes].to(device)
            subgraph_edge_index = batch_edge_index.to(device)
            subgraph_edge_attr = batch_edge_attr.to(device)

            # 重新索引边
            node_idx_map = torch.full((n_train_nodes,), -1, dtype=torch.long, device=device)
            node_idx_map[batch_nodes] = torch.arange(len(batch_nodes), device=device)
            subgraph_edge_index = node_idx_map[subgraph_edge_index]
            subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index,
                               edge_attr=subgraph_edge_attr, y=subgraph_y)

            optimizer.zero_grad()
            out, _ = model(subgraph)
            loss = criterion(out, subgraph.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # 清理GPU
            del subgraph, subgraph_x, subgraph_y, subgraph_edge_index, subgraph_edge_attr
            torch.cuda.empty_cache()

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # ==================== 验证（分批） ====================
        model.eval()
        val_probs_all = []
        val_preds_all = []

        with torch.no_grad():
            for val_batch_start in range(0, n_val_nodes, node_batch_size):
                val_batch_end = min(val_batch_start + node_batch_size, n_val_nodes)
                val_batch_nodes = torch.arange(val_batch_start, val_batch_end)

                # 构建验证子图
                val_node_mask = torch.zeros(n_val_nodes, dtype=torch.bool)
                val_node_mask[val_batch_nodes] = True

                src, dst = val_data.edge_index
                edge_mask = val_node_mask[src] & val_node_mask[dst]
                val_edge_index = val_data.edge_index[:, edge_mask]
                val_edge_attr = val_data.edge_attr[edge_mask]

                val_subgraph_x = val_data.x[val_batch_nodes].to(device)
                val_subgraph_edge_index = val_edge_index.to(device)
                val_subgraph_edge_attr = val_edge_attr.to(device)

                # 重新索引
                node_idx_map = torch.full((n_val_nodes,), -1, dtype=torch.long, device=device)
                node_idx_map[val_batch_nodes] = torch.arange(len(val_batch_nodes), device=device)
                val_subgraph_edge_index = node_idx_map[val_subgraph_edge_index]

                val_subgraph = Data(x=val_subgraph_x, edge_index=val_subgraph_edge_index,
                                       edge_attr=val_subgraph_edge_attr)

                val_out, _ = model(val_subgraph)
                val_probs_all.append(torch.exp(val_out)[:, 1].cpu())
                val_preds_all.append(val_out.argmax(dim=1).cpu())

                del val_subgraph, val_subgraph_x, val_subgraph_edge_index, val_subgraph_edge_attr
                torch.cuda.empty_cache()

        val_probs = torch.cat(val_probs_all).numpy()
        val_preds = torch.cat(val_preds_all).numpy()
        val_acc = (val_preds == val_data.y.numpy()).mean()
        val_accs.append(val_acc)

        try:
            val_auc = roc_auc_score(val_data.y.numpy(), val_probs)
        except:
            val_auc = 0.0

        scheduler.step(val_acc)

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, os.path.join(CHECKPOINTS_DIR, "best_gat_model.pt"))
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f"   Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if patience_counter >= patience:
            print(f"   早停于 epoch {epoch}")
            break

        gc.collect()

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, train_losses, val_accs

# 开始训练
start_time = time.time()
model, train_losses, val_accs = train_model(
    model, train_data, val_data,
    epochs=100, lr=0.001, pos_weight=pos_weight_val
)
training_time = time.time() - start_time
print(f"\n   训练完成! 用时: {training_time:.2f}秒")

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accs)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.pdf'), bbox_inches='tight')
plt.close()

# 5. 测试集评估（分批）
print("\n6. 测试集评估...")

model.eval()
test_preds_all = []
test_probs_all = []
test_embeddings_all = []
n_test_nodes = test_data.x.shape[0]
node_batch_size = 50000

with torch.no_grad():
    for test_batch_start in range(0, n_test_nodes, node_batch_size):
        test_batch_end = min(test_batch_start + node_batch_size, n_test_nodes)
        test_batch_nodes = torch.arange(test_batch_start, test_batch_end)

        # 构建测试子图
        test_node_mask = torch.zeros(n_test_nodes, dtype=torch.bool)
        test_node_mask[test_batch_nodes] = True

        src, dst = test_data.edge_index
        edge_mask = test_node_mask[src] & test_node_mask[dst]
        test_edge_index = test_data.edge_index[:, edge_mask]
        test_edge_attr = test_data.edge_attr[edge_mask]

        test_subgraph_x = test_data.x[test_batch_nodes].to(device)
        test_subgraph_edge_index = test_edge_index.to(device)
        test_subgraph_edge_attr = test_edge_attr.to(device)

        # 重新索引
        node_idx_map = torch.full((n_test_nodes,), -1, dtype=torch.long, device=device)
        node_idx_map[test_batch_nodes] = torch.arange(len(test_batch_nodes), device=device)
        test_subgraph_edge_index = node_idx_map[test_subgraph_edge_index]
        test_subgraph = Data(x=test_subgraph_x, edge_index=test_subgraph_edge_index,
                                edge_attr=test_subgraph_edge_attr)

        test_out, test_emb = model(test_subgraph)
        test_preds_all.append(test_out.argmax(dim=1).cpu())
        test_probs_all.append(torch.exp(test_out)[:, 1].cpu())
        test_embeddings_all.append(torch.tensor(test_emb, dtype=torch.float32))

        del test_subgraph, test_subgraph_x, test_subgraph_edge_index, test_subgraph_edge_attr
        torch.cuda.empty_cache()

test_pred = torch.cat(test_preds_all).numpy()
test_prob = torch.cat(test_probs_all).numpy()
test_embeddings = torch.cat(test_embeddings_all).numpy()
test_true = test_data.y.numpy()

# 计算各项指标
accuracy = accuracy_score(test_true, test_pred)
precision = precision_score(test_true, test_pred)
recall = recall_score(test_true, test_pred)
f1 = f1_score(test_true, test_pred)
auc = roc_auc_score(test_true, test_prob)

print(f"\n   测试集结果:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   AUC:       {auc:.4f}")

# 保存评估结果
results = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': auc,
    'training_time': training_time
}

with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
    import json
    json.dump(results, f, indent=2)

# 6. 绘制混淆矩阵 (Figure 5B右)
print("\n7. 生成Figure 5B: 混淆矩阵和UMAP...")

cm = confusion_matrix(test_true, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-sensitive', 'Sensitive'],
            yticklabels=['Non-sensitive', 'Sensitive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'GAT Confusion Matrix\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Panel5B_confusion_matrix.pdf'), bbox_inches='tight')
plt.close()

# 7. UMAP降维可视化 (Figure 5B)
print("\n8. 生成GAT嵌入UMAP可视化...")

# 使用UMAP降维
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
emb_2d = reducer.fit_transform(test_embeddings)

# 颜色方案：Non-sensitive=蓝, Sensitive=红, Misclassified=橙
_NS_C  = '#4393C3'   # Non-sensitive
_SEN_C = '#D6604D'   # Sensitive
_MIS_C = '#FF8C00'   # Misclassified

is_wrong = (test_true != test_pred)
n_wrong  = is_wrong.sum()
wrong_rate = n_wrong / len(test_true) * 100

# 3面板图: 真实标签 | 预测标签 | 误分类高亮
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    'GAT Latent Space: Sensitive vs Non-sensitive Cell Classification\n'
    f'(F1={f1:.3f}  AUC={auc:.3f}  Accuracy={accuracy:.3f}  Error rate={wrong_rate:.1f}%)',
    fontsize=13, fontweight='bold'
)

for ax_i, (labels, title) in enumerate([(test_true,  'True Labels'),
                                         (test_pred,  'Predicted Labels'),
                                         (None,       'Misclassified Cells')]):
    ax = axes[ax_i]
    if ax_i < 2:
        # 先画 Non-sensitive，再画 Sensitive（避免覆盖）
        for lbl, col, name in [(0, _NS_C, 'Non-sensitive'), (1, _SEN_C, 'Sensitive')]:
            mask = labels == lbl
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=col, s=4, alpha=0.5, rasterized=True, label=name)
        ax.legend(fontsize=9, markerscale=3, frameon=False,
                  loc='upper right')
    else:
        # 灰色底层 + 橙色误分类点
        ax.scatter(emb_2d[~is_wrong, 0], emb_2d[~is_wrong, 1],
                   c='#CCCCCC', s=3, alpha=0.3, rasterized=True, label='Correct')
        ax.scatter(emb_2d[ is_wrong, 0], emb_2d[ is_wrong, 1],
                   c=_MIS_C,   s=8, alpha=0.8, rasterized=True,
                   label=f'Misclassified (n={n_wrong}, {wrong_rate:.1f}%)')
        ax.legend(fontsize=9, markerscale=2, frameon=False, loc='upper right')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Panel5B_GAT_UMAP.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'Panel5B_GAT_UMAP.pdf'), bbox_inches='tight')
plt.close()
print(f"✅ Panel5B_GAT_UMAP saved  (error rate={wrong_rate:.1f}%)")

# 8. 额外分析：按疾病状态着色
print("\n9. 按疾病状态分析...")

# 加载疾病状态
disease_test = np.load(os.path.join(INPUT_DIR, "disease_test.npy"), allow_pickle=True)

# 创建疾病状态标签（用于可视化）
disease_labels = np.array([1 if d == 'systemic lupus erythematosus' else 0 for d in disease_test])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                      c=disease_labels, cmap='viridis', 
                      s=5, alpha=0.6, rasterized=True)
plt.colorbar(scatter, ticks=[0, 1], label='Disease (0=HC, 1=SLE)')
plt.title('GAT Embeddings - Disease Status')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'GAT_UMAP_by_disease.pdf'), bbox_inches='tight')
plt.close()

# 9. 按细胞类型分析（如果有）
print("\n10. 按细胞类型分析...")

# 尝试加载细胞类型信息
try:
    cell_metadata = pd.read_csv(os.path.join(INPUT_DIR, "cell_metadata.csv"))
    test_cells = cell_metadata[cell_metadata['split'] == 'test']
    
    if len(test_cells) == len(test_true):
        cell_types = test_cells['cell_type'].values
        
        # 获取主要的细胞类型
        unique_types = np.unique(cell_types)
        print(f"   细胞类型: {unique_types}")

        # 为每种细胞类型分配颜色
        from matplotlib.colors import ListedColormap
        import colorsys

        # 生成不同颜色
        n_types = len(unique_types)
        colors = plt.cm.tab20(np.linspace(0, 1, n_types))
        type_to_color = {t: colors[i] for i, t in enumerate(unique_types)}

        # 为每个点分配颜色
        point_colors = [type_to_color[t] for t in cell_types]

        # 细胞类型简写到全名的映射
        celltype_names = {
            'T4':    'CD4+ T cell',
            'T8':    'CD8+ T cell',
            'B':     'B cell',
            'cM':    'Classical Monocyte',
            'NK':    'NK cell',
            'ncM':   'Non-classical Monocyte',
            'cDC':   'Classical DC',
            'Prolif':'Proliferating cells',
            'pDC':   'Plasmacytoid DC',
            'PB':    'Plasmablast',
            'Progen':'Progenitor'
        }

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                              c=point_colors, s=5, alpha=0.6, rasterized=True)

        # 添加图例（使用全名）
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=colors[i], markersize=8,
                                      label=celltype_names.get(t, t))
                          for i, t in enumerate(unique_types)]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.title('GAT Embeddings - Cell Types')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'GAT_UMAP_by_celltype.pdf'), bbox_inches='tight')
        plt.close()
        
except Exception as e:
    print(f"   无法加载细胞类型信息: {e}")

# 10. 保存模型和嵌入
print("\n11. 保存模型和嵌入...")

# 保存整个模型
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "gat_model_final.pt"))

# 保存嵌入用于后续分析
np.save(os.path.join(OUTPUT_DIR, "test_embeddings.npy"), test_embeddings)
np.save(os.path.join(OUTPUT_DIR, "embeddings_2d.npy"), emb_2d)

# 保存预测结果
predictions_df = pd.DataFrame({
    'true_label': test_true,
    'pred_label': test_pred,
    'pred_prob': test_prob,
    'disease': disease_test if 'disease_test' in locals() else None
})
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# 11. 生成性能摘要报告
print("\n12. 生成性能摘要报告...")

summary = f"""
{'='*60}
GAT模型性能摘要
{'='*60}

数据集信息:
  训练集: {n_train} 细胞
  验证集: {n_val} 细胞
  测试集: {n_test} 细胞
  特征数: {n_features}

模型配置:
  模型类型: GAT (图注意力网络)
  隐藏层维度: 64
  注意力头数: 4
  Dropout: 0.3
  优化器: AdamW
  学习率: 0.001

测试集性能:
  Accuracy:  {accuracy:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1 Score:  {f1:.4f}
  AUC:       {auc:.4f}

训练时间: {training_time:.2f} 秒

输出文件:
  - Panel5B_confusion_matrix.png: 混淆矩阵
  - Panel5B_GAT_UMAP.png: GAT嵌入UMAP可视化
  - GAT_UMAP_by_disease.png: 按疾病状态着色
  - GAT_UMAP_by_celltype.png: 按细胞类型着色
  - test_embeddings.npy: 测试集节点嵌入
  - test_predictions.csv: 预测结果
  - gat_model_final.pt: 训练好的模型
{'='*60}
"""

print(summary)

with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), 'w') as f:
    f.write(summary)

print("\nStep 2 完成!")
print(f"输出文件已保存至: {OUTPUT_DIR}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5E - 量子神经网络独立脚本
在 qml_fig5e 环境中运行：
    conda activate qml_fig5e
    python code/Step4E_quantum_standalone.py

依赖：pennylane==0.36.0  autoray==0.6.12  torch  scikit-learn  numpy  scipy
结果保存至 output/figure5D_E/qnn_results.json，供 Step4 主脚本读取。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import pennylane as qml
import pickle

np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = "/home/h3033/statics/GEO_data/GSE/figure5"
INPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/figure5D_E")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Figure 5E - 量子神经网络概念验证")
print(f"PennyLane 版本: {qml.__version__}")
print("=" * 60)

# ==================== 1. 加载数据 ====================
print("\n1. 加载数据...")
X_train = np.load(os.path.join(INPUT_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(INPUT_DIR, "X_test.npy"))
y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(INPUT_DIR, "y_test.npy"))
print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA 降到 n_qubits 维
n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"  PCA 解释方差: {pca.explained_variance_ratio_.sum():.3f}")

# 小样本（量子电路仿真速度慢）
n_train = 200   # 量子仿真逐样本运行，小样本即可
n_test  = 100
idx_tr = np.random.choice(len(X_train_pca), n_train, replace=False)
idx_te = np.random.choice(len(X_test_pca),  n_test,  replace=False)
X_tr = X_train_pca[idx_tr]
y_tr = y_train[idx_tr]
X_te = X_test_pca[idx_te]
y_te = y_test[idx_te]

print(f"  量子训练集: {X_tr.shape}, 测试集: {X_te.shape}")

# ==================== 2. 定义量子电路 ====================
print("\n2. 定义量子神经网络...")

dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 不使用 TorchLayer（0.38.0 批量处理有 bug），改为自定义层逐样本调用
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.01)

    def forward(self, x):
        # x: (batch, n_qubits) — 逐样本送入量子电路
        out = [torch.stack(quantum_circuit(x[i], self.weights))
               for i in range(x.shape[0])]
        return torch.stack(out).float()   # 转为 float32，与经典层一致

class QuantumNN(nn.Module):
    def __init__(self, n_qubits, n_layers=2):
        super().__init__()
        self.qlayer = QuantumLayer(n_qubits, n_layers)
        self.fc1    = nn.Linear(n_qubits, 8)
        self.fc2    = nn.Linear(8, 2)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(0.2)

    def forward(self, x):
        x = self.qlayer(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = QuantumNN(n_qubits, n_layers=2)
n_params = sum(p.numel() for p in model.parameters())
print(f"  量子+经典参数总数: {n_params}")

# ==================== 3. 训练 ====================
print("\n3. 训练量子神经网络（约需几分钟）...")

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_te_t = torch.tensor(X_te, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.long)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

n_epochs   = 50
batch_size = 20   # 每批20个样本，共10批
train_losses, test_accs = [], []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for i in range(0, len(X_tr_t), batch_size):
        bx = X_tr_t[i:i+batch_size]
        by = y_tr_t[i:i+batch_size]
        optimizer.zero_grad()
        out  = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / (len(X_tr_t) // batch_size)
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        pred = model(X_te_t).argmax(dim=1)
        acc  = (pred == y_te_t).float().mean().item()
        test_accs.append(acc)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}")

# 最终评估
model.eval()
with torch.no_grad():
    logits  = model(X_te_t)
    qnn_pred = logits.argmax(dim=1).numpy()
    qnn_prob = torch.softmax(logits, dim=1)[:, 1].numpy()

qnn_acc = accuracy_score(y_te, qnn_pred)
qnn_auc = roc_auc_score(y_te, qnn_prob)
print(f"\n  量子神经网络  Accuracy={qnn_acc:.4f}  AUC={qnn_auc:.4f}")

# ==================== 4. SVM 对比 ====================
print("\n4. 训练 SVM 作为对比...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_tr, y_tr)
svm_pred = svm.predict(X_te)
svm_prob = svm.predict_proba(X_te)[:, 1]
svm_acc  = accuracy_score(y_te, svm_pred)
svm_auc  = roc_auc_score(y_te, svm_prob)
print(f"  SVM (RBF)       Accuracy={svm_acc:.4f}  AUC={svm_auc:.4f}")

# ==================== 5. 绘图 ====================
print("\n5. 生成 Figure 5E 图...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Configure Arial font for publication
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts


plt.rcParams.update({
    'font.family':     'Arial',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：量子电路示意图
ax = axes[0]
for i in range(n_qubits):
    ax.plot([0, 10], [i, i], 'k-', linewidth=1, alpha=0.5)
    ax.text(-0.5, i, f'$q_{i}$', ha='right', va='center', fontsize=12)

# 数据编码层
for i in range(n_qubits):
    ax.add_patch(plt.Rectangle((1, i-0.3), 1, 0.6, facecolor='lightblue', edgecolor='blue', alpha=0.7))
    ax.text(1.5, i, 'RY', ha='center', va='center', fontsize=10, fontweight='bold')

# 变分纠缠层
for layer in range(2):
    xp = 3 + layer * 2.5
    for i in range(n_qubits - 1):
        ax.plot([xp, xp], [i, i+1], 'b-', linewidth=2)
        ax.plot(xp, i,   'o', markersize=8,  color='blue')
        ax.plot(xp, i+1, '+', markersize=12, color='blue', markeredgewidth=2)
    for i in range(n_qubits):
        ax.add_patch(plt.Rectangle((xp+0.5, i-0.3), 1, 0.6, facecolor='lightgreen', edgecolor='green', alpha=0.7))
        ax.text(xp+1, i, 'RY', ha='center', va='center', fontsize=10, fontweight='bold')

# 测量
for i in range(n_qubits):
    ax.plot([9, 9.5], [i, i], 'r-', linewidth=2)
    ax.plot(9.5, i, '^', markersize=10, color='red')
    ax.text(9.8, i, 'Z', ha='left', va='center', fontsize=12, fontweight='bold')

ax.text(1.5, -0.8, 'Data Encoding',    ha='center', fontsize=9, color='blue')
ax.text(5.0, -0.8, 'Entangling Layers', ha='center', fontsize=9, color='green')
ax.text(9.5, -0.8, 'Measure',           ha='center', fontsize=9, color='red')
ax.set_xlim(-1, 11); ax.set_ylim(-1.2, n_qubits)
ax.axis('off')
ax.set_title('Quantum Circuit Architecture\n(4 Qubits, 2 Variational Layers)',
             fontsize=13, fontweight='bold')

# 右：性能对比表
ax = axes[1]
ax.axis('off')
table_data = [
    ['Model',        'Accuracy',        'AUC',            'Parameters'],
    ['Quantum NN',   f'{qnn_acc:.4f}',  f'{qnn_auc:.4f}', f'~{n_params}'],
    ['SVM (RBF)',    f'{svm_acc:.4f}',  f'{svm_auc:.4f}', '~10K'],
]
tbl = ax.table(cellText=table_data, loc='center', cellLoc='center',
               colWidths=[0.22, 0.15, 0.15, 0.18])
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2)
for j in range(4):
    tbl[(0, j)].set_facecolor('#4472C4')
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
tbl[(1, 0)].set_facecolor('#D9E1F2')
tbl[(2, 0)].set_facecolor('#F2F2F2')
ax.set_title(f'Quantum vs Classical Performance\n(Train={n_train}, Test={n_test} cells)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'Panel5E_quantum_comparison.pdf')
plt.savefig(out_path, bbox_inches='tight')
plt.savefig(out_path.replace('.pdf', '.pdf'), bbox_inches='tight')
plt.close()
print(f"  ✓ 保存: {out_path}")

# 训练曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label='Training Loss', linewidth=2)
ax2 = ax.twinx()
ax2.plot(test_accs, color='orange', label='Test Accuracy', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax2.set_ylabel('Accuracy')
ax.set_title('Quantum Neural Network Training Curves')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Panel5E_qnn_training.pdf'), bbox_inches='tight')
plt.close()

# ==================== 6. 保存结果供 Step4 读取 ====================
results = {
    "quantum_available": True,
    "qnn_accuracy":  float(qnn_acc),
    "qnn_auc":       float(qnn_auc),
    "svm_accuracy":  float(svm_acc),
    "svm_auc":       float(svm_auc),
    "n_qubits":      n_qubits,
    "n_layers":      2,
    "n_train":       n_train,
    "n_test":        n_test,
    "n_params":      int(n_params),
    "train_losses":  train_losses,
    "test_accs":     test_accs,
    "pennylane_version": qml.__version__,
}
result_path = os.path.join(OUTPUT_DIR, 'qnn_results.json')
with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ 结果已保存: {result_path}")

print("\n" + "=" * 60)
print("Step 4E 量子部分完成！")
print("现在可以切回 figure5 环境继续运行 Step4 主脚本。")
print("=" * 60)

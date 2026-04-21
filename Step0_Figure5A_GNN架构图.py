#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5A - GAT Architecture (Journal of Advanced Research style)
Font: Arial | Language: English | No overlapping elements
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import numpy as np

# Arial字体设置
available_fonts = [f.name for f in fm.fontManager.ttflist]
plt.rcParams['font.family'] = 'Arial' if 'Arial' in available_fonts else 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42   # 嵌入字体，投稿必须
plt.rcParams['ps.fonttype']  = 42

OUTPUT_DIR = "/home/h3033/statics/GEO_data/GSE/figure5/output/figure5A"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 颜色 ──────────────────────────────────────
C_IN   = '#2E86AB'
C_GAT1 = '#8B2FC9'
C_GAT2 = '#E07C00'
C_FC   = '#C0392B'
C_NODE = '#27AE60'
C_EDGE = '#BBBBBB'
C_TEXT = '#222222'

# ── 画布：两行，用 GridSpec 保证间距 ──────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 1, figure=fig,
              top=0.91, bottom=0.06,
              hspace=0.08,
              height_ratios=[1, 1])
ax_t = fig.add_subplot(gs[0])   # 上：KNN图构建
ax_b = fig.add_subplot(gs[1])   # 下：网络层流程

for ax in (ax_t, ax_b):
    ax.set_facecolor('white')
    ax.axis('off')

ax_t.set_xlim(0, 20);  ax_t.set_ylim(-0.3, 4.2)
ax_b.set_xlim(0, 20);  ax_b.set_ylim(0, 5.0)

# ── 工具函数 ──────────────────────────────────
def rbox(ax, cx, cy, w, h, line1, line2, color, fs=10.5):
    """画圆角矩形，两行文字，内边距充足"""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.12',
                          facecolor=color, edgecolor='white',
                          linewidth=2.0, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + h*0.18, line1,
            ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', zorder=4)
    ax.text(cx, cy - h*0.2, line2,
            ha='center', va='center', fontsize=fs - 2,
            color='white', alpha=0.93, zorder=4,
            linespacing=1.4, style='italic')

def harrow(ax, x1, x2, y, label_up='', label_dn=''):
    """水平箭头，标注分上下两行"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=C_TEXT, lw=1.8))
    if label_up:
        ax.text((x1+x2)/2, y + 0.26, label_up,
                ha='center', va='bottom', fontsize=7.8,
                color='#555555', style='italic')
    if label_dn:
        ax.text((x1+x2)/2, y - 0.26, label_dn,
                ha='center', va='top', fontsize=7.8,
                color='#777777', style='italic')

def dim_pill(ax, cx, cy, text, color):
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold', zorder=5,
            bbox=dict(boxstyle='round,pad=0.32', facecolor=color,
                      edgecolor='white', linewidth=1.2, alpha=0.95))


# ════════════════════════════════════════════════════════════
# 上半区 — KNN Graph Construction
# ════════════════════════════════════════════════════════════
ax_t.text(10, 3.95, 'A   KNN Graph Construction',
          ha='center', va='center', fontsize=12,
          fontweight='bold', color=C_TEXT)

# 左：细胞散点云 (x 0.3–4.2, y 0.4–3.4)
np.random.seed(3)
n = 28
px = np.random.rand(n) * 3.6 + 0.4
py = np.random.rand(n) * 2.8 + 0.4
colors_pts = [C_IN if v > 0.5 else '#E74C3C' for v in np.random.rand(n)]
ax_t.scatter(px, py, s=38, c=colors_pts, alpha=0.75, zorder=2)

ax_t.text(2.2, 0.08, '1,206,594 cells  ·  50-dim expression features',
          ha='center', va='bottom', fontsize=9, color=C_IN, fontweight='bold')

# 图例（散点颜色）
for col, lbl in [(C_IN, 'HC'), ('#E74C3C', 'SLE')]:
    ax_t.scatter([], [], s=40, c=col, label=lbl)
ax_t.legend(loc='upper left', fontsize=8.5, frameon=True,
            edgecolor='#CCCCCC', bbox_to_anchor=(0.0, 0.98))

# 中：箭头
harrow(ax_t, 4.4, 6.7, 2.0,
       label_up='Cosine similarity',
       label_dn='k = 15 neighbors')

# 中：KNN图 (center 10.5, 2.0)
cx_g, cy_g = 8.8, 2.0
angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
r_g = 1.5
nx = cx_g + r_g * np.cos(angles)
ny = cy_g + r_g * np.sin(angles)

for i, j in [(0,1),(0,2),(0,6),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,0),(2,5)]:
    ax_t.plot([nx[i], nx[j]], [ny[i], ny[j]],
              color=C_EDGE, lw=1.4, alpha=0.65, zorder=1)

ax_t.scatter(nx, ny, s=520, c=C_NODE, zorder=2,
             edgecolors='white', linewidths=1.8)
for i in range(7):
    ax_t.text(nx[i], ny[i], str(i+1),
              ha='center', va='center', fontsize=9,
              color='white', fontweight='bold', zorder=3)

ax_t.text(cx_g, -0.05,
          '3,041,265 edges  ·  avg. degree = 15',
          ha='center', va='center', fontsize=9,
          color=C_NODE, fontweight='bold')

# 高亮一条边并标注 edge weight
i0, i1 = 0, 1
ax_t.annotate('', xy=(nx[i1], ny[i1]), xytext=(nx[i0], ny[i0]),
              arrowprops=dict(arrowstyle='->', color=C_GAT1, lw=2.2))
mid_x = (nx[i0]+nx[i1])/2 + 1.0
mid_y = (ny[i0]+ny[i1])/2 + 0.55
ax_t.text(mid_x, mid_y, 'edge weight\n(sim. score)',
          ha='center', va='bottom', fontsize=8,
          color=C_GAT1, fontweight='bold', style='italic')

# 右：属性框 (x 13.0–18.5)
props = [
    ('Graph type',  'KNN (k = 15)'),
    ('Similarity',  'Cosine'),
    ('Edge attr.',  'Similarity score (1-dim)'),
    ('Nodes',       '1,206,594 cells'),
    ('Edges',       '3,041,265'),
]
bx0, bx1, by0, by1 = 13.8, 18.6, 0.35, 3.65
prop_rect = FancyBboxPatch((bx0, by0), bx1-bx0, by1-by0,
                           boxstyle='round,pad=0.12',
                           facecolor='#F4F6FB', edgecolor='#BBBBD0',
                           linewidth=1.5, zorder=2)
ax_t.add_patch(prop_rect)
ax_t.text((bx0+bx1)/2, by1-0.18, 'Graph Properties',
          ha='center', va='center', fontsize=9.5,
          fontweight='bold', color=C_TEXT)
for idx, (k, v) in enumerate(props):
    yy = by1 - 0.52 - idx * 0.52
    ax_t.text(bx0+0.22, yy, k + ':',
              ha='left', va='center', fontsize=8.5,
              color='#555555', fontweight='bold')
    ax_t.text(bx1-0.18, yy, v,
              ha='right', va='center', fontsize=8.5, color=C_TEXT)

# 右向箭头（KNN图 → 属性框）
harrow(ax_t, 10.65, 13.6, 2.0)


# ════════════════════════════════════════════════════════════
# 下半区 — GAT Model Architecture
# ════════════════════════════════════════════════════════════
ax_b.text(10, 4.72, 'B   GAT Model Architecture',
          ha='center', va='center', fontsize=12,
          fontweight='bold', color=C_TEXT)

cy_main = 2.6   # 所有模块中心 y

# ── 坐标（cx） ──────────────────────────────
X_IN   = 1.5
X_G1   = 5.8
X_G2   = 11.2
X_FC   = 15.8
X_OUT  = 18.7

# ── 各模块尺寸 ──────────────────────────────
W_IN, H_IN   = 2.2, 2.2
W_G1, H_G1   = 3.6, 3.2
W_G2, H_G2   = 3.6, 2.8
W_FC, H_FC   = 2.6, 2.4

# ── 模块方块 ──────────────────────────────
rbox(ax_b, X_IN, cy_main, W_IN, H_IN,
     'Input', '50-dim\nexpression', C_IN, fs=11)

rbox(ax_b, X_G1, cy_main, W_G1, H_G1,
     'GAT Conv 1',
     'heads = 4, concat\nedge_dim = 1\n50 → 64 × 4 = 256',
     C_GAT1, fs=10.5)

rbox(ax_b, X_G2, cy_main, W_G2, H_G2,
     'GAT Conv 2',
     'heads = 1, no concat\nedge_dim = 1\n256 → 64',
     C_GAT2, fs=10.5)

rbox(ax_b, X_FC, cy_main, W_FC, H_FC,
     'FC Layer',
     'Linear\n64 → 2\nSoftmax',
     C_FC, fs=10.5)

# ── 输出节点 ──────────────────────────────
for label, col, dy in [('SLE', '#E74C3C', 0.60), ('HC', '#2980B9', -0.60)]:
    c = plt.Circle((X_OUT, cy_main + dy), 0.38,
                   color=col, zorder=3, ec='white', lw=2.0)
    ax_b.add_patch(c)
    ax_b.text(X_OUT, cy_main + dy, label,
              ha='center', va='center', fontsize=9,
              color='white', fontweight='bold', zorder=4)

ax_b.text(X_OUT, cy_main - 1.35,
          'P(disease)',
          ha='center', va='center', fontsize=9, color=C_TEXT)

# ── 箭头 ──────────────────────────────────
harrow(ax_b,
       X_IN + W_IN/2, X_G1 - W_G1/2,
       cy_main, label_up='ELU +\nDropout(0.3)')

harrow(ax_b,
       X_G1 + W_G1/2, X_G2 - W_G2/2,
       cy_main, label_up='ELU +\nDropout(0.3)')

harrow(ax_b,
       X_G2 + W_G2/2, X_FC - W_FC/2,
       cy_main, label_up='ELU')

harrow(ax_b,
       X_FC + W_FC/2, X_OUT - 0.42,
       cy_main, label_up='Softmax')

# ── 维度标注气泡（每层正下方）──────────────
dim_pill(ax_b, X_IN,               0.55, '50',    C_IN)
dim_pill(ax_b, (X_IN+X_G1)/2,      0.55, '50',    C_IN)
dim_pill(ax_b, (X_G1+X_G2)/2,      0.55, '256',   C_GAT1)
dim_pill(ax_b, (X_G2+X_FC)/2,      0.55, '64',    C_GAT2)
dim_pill(ax_b, (X_FC+X_OUT)/2,     0.55, '2',     C_FC)

ax_b.text(10, 0.12,
          '— Feature dimension —',
          ha='center', va='center', fontsize=8.5,
          color='#AAAAAA', style='italic')

# ── 参数统计 ──────────────────────────────
ax_b.text(10, 4.50,
          'Total parameters: 30,914  ·  Dropout: 0.3  ·  Optimizer: Adam  ·  Loss: CrossEntropy',
          ha='center', va='center', fontsize=9, color='#666666')

# ── 图例 ──────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=C_IN,   label='Input layer'),
    mpatches.Patch(facecolor=C_GAT1, label='GAT Conv 1  (4 heads, 256-dim)'),
    mpatches.Patch(facecolor=C_GAT2, label='GAT Conv 2  (1 head,  64-dim)'),
    mpatches.Patch(facecolor=C_FC,   label='FC Classifier'),
]
ax_b.legend(handles=legend_patches, loc='lower center',
            fontsize=8.5, frameon=True, edgecolor='#CCCCCC',
            ncol=4, bbox_to_anchor=(0.5, -0.02))

# ── 总标题 ──────────────────────────────────
fig.text(0.5, 0.96,
         'Figure 5A: Graph Attention Network Architecture for Single-Cell Disease Prediction',
         ha='center', va='top', fontsize=13, fontweight='bold', color=C_TEXT)

# ── 保存 ──────────────────────────────────
out_png = os.path.join(OUTPUT_DIR, 'Panel5A_GAT_architecture.pdf')
out_pdf = os.path.join(OUTPUT_DIR, 'Panel5A_GAT_architecture.pdf')
plt.savefig(out_png, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ {out_png}")
print(f"✓ {out_pdf}")

#!/usr/bin/env python3
"""
批量修改figure5源码：将PNG改为PDF，并配置Arial字体嵌入
"""

import os
import re

# 源码目录
CODE_DIR = "/home/h3033/statics/GEO_data/GSE/figure5/code"

# 需要处理的文件
FILES_TO_PROCESS = [
    "Step0_Figure5A_GNN架构图.py",
    "Step0_Figure5A_数据流程图.py",
    "Step2_GNN_建模.py",
    "Step3_GNN_特征分析.py",
    "Step4_SHAP可解释性分析与量子机器学习概念验证.py",
    "Step4E_quantum_standalone.py"
]

def add_font_config(content):
    """在文件开头添加字体配置"""

    # 检查是否已经有字体配置
    if "matplotlib.rcParams['font.family']" in content or "plt.rcParams['font.family']" in content:
        return content

    # 找到第一个import matplotlib的位置
    import_pattern = r'(import matplotlib\.pyplot as plt|from matplotlib import pyplot as plt)'
    match = re.search(import_pattern, content)

    if match:
        # 在import之后添加字体配置
        font_config = """
# Configure Arial font for publication
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts
"""
        insert_pos = match.end()
        content = content[:insert_pos] + font_config + content[insert_pos:]

    return content

def convert_png_to_pdf(content):
    """将所有savefig的PNG改为PDF"""

    # 替换 .png 为 .pdf
    content = re.sub(
        r"savefig\(([^)]*)'([^']*?)\.png'",
        r"savefig(\1'\2.pdf'",
        content
    )

    content = re.sub(
        r'savefig\(([^)]*)"([^"]*?)\.png"',
        r'savefig(\1"\2.pdf"',
        content
    )

    # 替换 f-string 中的 .png
    content = re.sub(
        r"savefig\(f'([^']*?)\.png'",
        r"savefig(f'\1.pdf'",
        content
    )

    content = re.sub(
        r'savefig\(f"([^"]*?)\.png"',
        r'savefig(f"\1.pdf"',
        content
    )

    # 替换 os.path.join 中的 .png
    content = re.sub(
        r"'([^']*?)\.png'",
        lambda m: f"'{m.group(1)}.pdf'" if '.png' in m.group(0) else m.group(0),
        content
    )

    # 移除 dpi=300 参数（PDF不需要）
    content = re.sub(
        r',\s*dpi=\d+',
        '',
        content
    )

    return content

def process_file(filepath):
    """处理单个文件"""

    print(f"Processing: {os.path.basename(filepath)}")

    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 添加字体配置
    content = add_font_config(content)

    # 转换PNG为PDF
    content = convert_png_to_pdf(content)

    # 如果有修改，写回文件
    if content != original_content:
        # 备份原文件
        backup_path = filepath + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  ✓ Backup created: {os.path.basename(backup_path)}")

        # 写入修改后的内容
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Modified: PNG → PDF, Arial font configured")
    else:
        print(f"  - No changes needed")

    print()

def main():
    print("=" * 70)
    print("Converting PNG to PDF and configuring Arial font")
    print("=" * 70)
    print()

    for filename in FILES_TO_PROCESS:
        filepath = os.path.join(CODE_DIR, filename)

        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"Warning: File not found: {filename}\n")

    print("=" * 70)
    print("Conversion completed!")
    print("=" * 70)
    print("\nBackup files created with .backup extension")
    print("If you need to restore, run: mv file.py.backup file.py")

if __name__ == "__main__":
    main()

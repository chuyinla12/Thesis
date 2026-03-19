
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    install('matplotlib')
    install('numpy')
    import matplotlib.pyplot as plt
    import numpy as np

import re
import os

def abbreviate_method(name):
    s = name.lower()
    if 'no-ne' in s or 'no ne' in s:
        return 'nne'
    if 'no-cluster' in s or 'no cluster' in s:
        return 'ncl'
    if 'no-kl' in s or 'no kl' in s:
        return 'nkl'
    if 'no-sample' in s or 'no sample' in s:
        return 'nsl'
    if 'ours' in s:
        return 'co'
    return name

def parse_md_table(md_content):
    datasets = {}
    current_dataset = None
    lines = md_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('**'):
            # Extract dataset name and remove markup/symbols like ** and colons
            name = line
            name = re.sub(r'\*', '', name)
            name = name.replace('：', '').replace(':', '').strip()
            current_dataset = name
            datasets[current_dataset] = {'methods': [], 'values': []}
        elif line.startswith('|') and '---' not in line and 'ACC' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 4:
                method, acc, nmi, ari = parts
                datasets[current_dataset]['methods'].append(method)
                datasets[current_dataset]['values'].append([float(acc), float(nmi), float(ari)])
    return datasets

def plot_3d_bar(dataset_name, data, output_path):
    methods = data['methods']
    values = np.array(data['values'])
    metrics = ['ACC', 'NMI', 'ARI']
    
    abbrs = [abbreviate_method(m) for m in methods]
    
    # 设置中文字体（优先使用系统自带字体）
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    # SVG 中保留文本为可编辑文字
    plt.rcParams['svg.fonttype'] = 'none'
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    xpos = np.arange(len(methods))
    ypos = np.arange(len(metrics))
    xpos, ypos = np.meshgrid(xpos, ypos, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    dx = 0.6
    dy = 0.6
    dz = values.ravel()

    # 使用更美观的渐变色
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(xpos)))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    for i in range(len(methods)):
        for j in range(len(metrics)):
            val = values[i, j]
            ax.text(i + dx/2, j + dy/2, val + 0.3, f"{val:.2f}", fontsize=8,
                    ha='center', va='bottom', color='black')
    
    # 设置轴标签字体（不使用顶部标题）
    ax.set_xlabel('', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('分数', fontsize=14, fontweight='bold', labelpad=15)
    
    # 设置刻度标签
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(abbrs, fontsize=12, fontweight='bold')
    
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    
    # 设置背景色和网格
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 无图例，直接在 X 轴写缩写
    
    # 调整布局
    plt.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    
    # Sanitize filename for Windows
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', dataset_name)
    plt.savefig(os.path.join(output_path, f'{safe_name}_3d_chart.svg'),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    md_file_path = 'c:\\Users\\Miku12\\Desktop\\GraduationThesis\\Table\\消融实验结果表格.md'
    output_dir = 'c:\\Users\\Miku12\\Desktop\\GraduationThesis\\Table'

    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    all_data = parse_md_table(content)
    
    for name, data in all_data.items():
        plot_3d_bar(name, data, output_dir)
        print(f"Generated chart for {name} and saved to {output_dir}")

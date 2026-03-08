
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
    
    method_labels = [chr(ord('a') + i) for i in range(len(methods))]
    
    # 设置中文字体（优先使用系统自带字体）
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    # SVG 中保留文本为可编辑文字
    plt.rcParams['svg.fonttype'] = 'none'
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    xpos = np.arange(len(method_labels))
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
    
    # 设置轴标签字体（不使用顶部标题）
    ax.set_xlabel('方法', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('指标', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('分数', fontsize=14, fontweight='bold', labelpad=15)
    
    # 设置刻度标签
    ax.set_xticks(np.arange(len(method_labels)))
    ax.set_xticklabels(method_labels, fontsize=12, fontweight='bold')
    
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    
    # 设置背景色和网格
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 创建图例
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.plasma(i/len(methods)), 
                                      edgecolor='black', linewidth=0.5, 
                                      label=f'{label}: {method}') 
                       for i, (label, method) in enumerate(zip(method_labels, methods))]
    
    # 调整图例位置和样式
    legend = ax.legend(handles=legend_elements, title="方法", 
                      bbox_to_anchor=(1.15, 0.9), loc='upper left',
                      fontsize=11, title_fontsize=12, 
                      frameon=True, fancybox=True, shadow=True,
                      borderaxespad=0.5)
    
    # 设置图例背景色
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
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

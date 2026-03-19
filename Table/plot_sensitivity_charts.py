import subprocess
import sys
import re
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors

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
try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    install('scipy')
    from scipy.interpolate import make_interp_spline

def parse_number(s):
    s = re.sub(r'[^\d\.\-]', '', s)
    return float(s) if s else np.nan

def parse_md_sensitivity(md_content):
    datasets = {}
    current_dataset = None
    lines = [ln.strip() for ln in md_content.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#'):
            name = line.lstrip('#').strip()
            current_dataset = name
            datasets[current_dataset] = []
            i += 1
            continue
        if line.startswith('|'):
            header = [p.strip() for p in line.split('|') if p.strip()]
            if i + 1 < len(lines) and '---' in lines[i + 1]:
                table_lines = []
                i += 2
                while i < len(lines) and lines[i].startswith('|'):
                    row = [p.strip() for p in lines[i].split('|') if p.strip()]
                    table_lines.append(row)
                    i += 1
                first = header[0].lower()
                if 'dropout' in first:
                    xs, acc, nmi, ari = [], [], [], []
                    for r in table_lines:
                        xs.append(r[0])
                        acc.append(parse_number(r[1]))
                        nmi.append(parse_number(r[2]))
                        ari.append(parse_number(r[3]))
                    datasets[current_dataset].append({
                        'type': 'dropout',
                        'x': xs,
                        'series': {'ACC': acc, 'NMI': nmi, 'ARI': ari}
                    })
                elif 'degree/ebc' in first:
                    cols = header[1:]
                    rows = []
                    matrix = []
                    for r in table_lines:
                        rows.append(r[0])
                        matrix.append([parse_number(x) for x in r[1:]])
                    datasets[current_dataset].append({
                        'type': 'degree_ebc',
                        'rows': rows,
                        'cols': cols,
                        'matrix': np.array(matrix, dtype=float)
                    })
                elif first == 'cluster':
                    xs, acc, nmi, ari = [], [], [], []
                    for r in table_lines:
                        xs.append(r[0])
                        acc.append(parse_number(r[1]))
                        nmi.append(parse_number(r[2]))
                        ari.append(parse_number(r[3]))
                    datasets[current_dataset].append({
                        'type': 'cluster',
                        'x': xs,
                        'series': {'ACC': acc, 'NMI': nmi, 'ARI': ari}
                    })
                elif first == 'ne':
                    xs, acc, nmi, ari = [], [], [], []
                    for r in table_lines:
                        xs.append(r[0])
                        acc.append(parse_number(r[1]))
                        nmi.append(parse_number(r[2]))
                        ari.append(parse_number(r[3]))
                    datasets[current_dataset].append({
                        'type': 'ne',
                        'x': xs,
                        'series': {'ACC': acc, 'NMI': nmi, 'ARI': ari}
                    })
                elif first == 'kl':
                    xs, acc, nmi, ari = [], [], [], []
                    for r in table_lines:
                        xs.append(r[0])
                        acc.append(parse_number(r[1]))
                        nmi.append(parse_number(r[2]))
                        ari.append(parse_number(r[3]))
                    datasets[current_dataset].append({
                        'type': 'kl',
                        'x': xs,
                        'series': {'ACC': acc, 'NMI': nmi, 'ARI': ari}
                    })
                else:
                    i += 1
                continue
        i += 1
    return datasets

def setup_style():
    plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['svg.fonttype'] = 'none'

def pastel_colors(n):
    cmap = plt.cm.Set2
    if n == 1:
        return [cmap(0.3)]
    return [cmap(i) for i in np.linspace(0.1, 0.8, n)]

def sanitize(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

def to_numeric_ticks(xs):
    vals = []
    for x in xs:
        s = re.sub(r'[^\d\.\-]', '', x)
        vals.append(float(s) if s else np.nan)
    return vals

def soft_cmap(base='YlGnBu', low=0.5, high=0.98):
    base_cmap = plt.colormaps.get_cmap(base)
    arr = base_cmap(np.linspace(low, high, 256))
    return mcolors.LinearSegmentedColormap.from_list(f'{base}_soft', arr)

def lighten(color, factor=0.35):
    r, g, b, a = mcolors.to_rgba(color)
    r = 1 - (1 - r) * (1 - factor)
    g = 1 - (1 - g) * (1 - factor)
    b = 1 - (1 - b) * (1 - factor)
    return (r, g, b, a)

def plot_lines(dataset, label, xs, series, output_dir):
    setup_style()
    metrics = ['ACC', 'NMI', 'ARI']
    colors = pastel_colors(len(metrics))
    x_vals = np.array(to_numeric_ticks(xs), dtype=float)
    x_dense = np.linspace(x_vals[0], x_vals[-1], 300)
    all_vals = np.concatenate([np.array(series[m], dtype=float) for m in metrics])
    y_min = 0.0
    y_max = float(np.nanmax(all_vals)) + 0.8
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.set_ylim(y_min, y_max)
    for idx, m in enumerate(metrics):
        vals = np.array(series[m], dtype=float)
        k = 3 if len(x_vals) > 3 else 1
        spline = make_interp_spline(x_vals, vals, k=k)
        y_dense = spline(x_dense)
        y_dense = np.clip(y_dense, y_min, y_max)
        ax.fill_between(x_dense, y_min, y_dense, color=lighten(colors[idx], 0.3), alpha=0.5)
        ax.plot(x_dense, y_dense, color=colors[idx], linewidth=2.8)
        ax.plot(x_dense, y_dense, color=lighten(colors[idx], 0.6), linewidth=6.0, alpha=0.12)
        peak_i = int(np.argmax(vals))
        peak_x = x_vals[peak_i]
        peak_y = vals[peak_i]
        ax.scatter([peak_x], [peak_y], color=colors[idx], s=34, zorder=5)
        ax.text(peak_x, peak_y, f"{peak_y:.2f}", color=colors[idx], fontsize=9, ha='center', va='bottom')
    ax.set_xlabel(label, fontsize=14, fontweight='bold')
    ax.set_ylabel('', fontsize=14, fontweight='bold')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(xs, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fname = f"{sanitize(dataset)}_{sanitize(label)}_lines.svg"
    plt.savefig(os.path.join(output_dir, fname), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_3d_bar_like_ablation(dataset, label, xs, series, output_dir):
    setup_style()
    metrics = ['ACC', 'NMI', 'ARI']
    values = np.column_stack([series[m] for m in metrics])
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    xpos = np.arange(len(xs))
    ypos = np.arange(len(metrics))
    xpos, ypos = np.meshgrid(xpos, ypos, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = 0.6
    dy = 0.6
    dz = values.ravel()
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(xpos)))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
    for i in range(len(xs)):
        for j in range(len(metrics)):
            val = values[i, j]
            ax.text(i + dx / 2, j + dy / 2, val + 0.3, f"{val:.2f}", fontsize=8,
                    ha='center', va='bottom', color='black')
    ax.set_xlabel(label, fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('分数', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, fontsize=12, fontweight='bold')
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=25, azim=45)
    plt.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    fname = f"{sanitize(dataset)}_{sanitize(label)}_3d_chart.svg"
    plt.savefig(os.path.join(output_dir, fname),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_grouped_bars(dataset, label, xs, series, output_dir):
    setup_style()
    metrics = list(series.keys())
    colors = pastel_colors(len(metrics))
    x_vals = np.arange(len(xs))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x_vals + (idx - (len(metrics)-1)/2)*width, series[m], width=width, color=c, edgecolor='black', linewidth=0.5, label=m)
    ax.set_xlabel(label, fontsize=14, fontweight='bold')
    ax.set_ylabel('分数', fontsize=14, fontweight='bold')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(xs, fontsize=12)
    ax.legend(loc='best', fontsize=12, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f"{sanitize(dataset)}_{sanitize(label)}_bars.svg"
    plt.savefig(os.path.join(output_dir, fname), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_3d_waves(dataset, label, xs, series, output_dir):
    setup_style()
    metrics = ['ACC', 'NMI', 'ARI']
    methods = xs
    colors = pastel_colors(len(metrics))
    x_idx = np.arange(len(methods), dtype=float)
    x_dense = np.linspace(x_idx[0], x_idx[-1], 120)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for idx, m in enumerate(metrics):
        vals = np.array(series[m], dtype=float)
        z_dense = np.interp(x_dense, x_idx, vals)
        ax.plot(x_dense, z_dense, color=colors[idx], linewidth=2.0, label=m)
        ax.fill_between(x_dense, 0, z_dense, color=colors[idx], alpha=0.25)
        peak_idx = int(np.argmax(vals))
        peak_x = x_idx[peak_idx]
        peak_y = vals[peak_idx]
        ax.scatter([peak_x], [peak_y], color=colors[idx], s=35, zorder=5)
        ax.text(peak_x, peak_y, f"{peak_y:.2f}", color=colors[idx],
                fontsize=10, ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel(label, fontsize=14, fontweight='bold')
    ax.set_ylabel('', fontsize=14, fontweight='bold')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(methods, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=12, frameon=True)
    plt.tight_layout()
    fname = f"{sanitize(dataset)}_{sanitize(label)}_3d.svg"
    plt.savefig(os.path.join(output_dir, fname),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_heatmap(dataset, rows, cols, matrix, output_dir):
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        matrix,
        cmap=soft_cmap('YlGnBu', 0.5, 0.98),
        norm=mcolors.PowerNorm(gamma=0.45, vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix))),
        aspect='auto',
        alpha=0.78
    )
    ax.set_xlabel('ebc', fontsize=14, fontweight='bold')
    ax.set_ylabel('degree', fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black', fontsize=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ACC', rotation=90, fontsize=12)
    plt.tight_layout()
    fname = f"{sanitize(dataset)}_degree_ebc_heatmap.svg"
    plt.savefig(os.path.join(output_dir, fname), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()

def plot_degree_ebc_3d(dataset, rows, cols, matrix, output_dir):
    setup_style()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(len(cols))
    y = np.arange(len(rows))
    xpos, ypos = np.meshgrid(x, y, indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = 0.6
    dy = 0.6
    dz = matrix.ravel()
    norm = mcolors.PowerNorm(gamma=0.45, vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    cmap = soft_cmap('YlGnBu', 0.5, 0.98)
    colors_arr = cmap(norm(dz))
    colors_arr[:, 3] = 0.78
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_arr, alpha=0.78, edgecolor='#9EA7B3', linewidth=0.32)
    for i in range(len(cols)):
        for j in range(len(rows)):
            val = float(matrix[j, i])
            ax.text(i + dx/2, j + dy/2, val + 0.2, f"{val:.2f}", fontsize=8, ha='center', va='bottom', color='black')
    ax.set_xlabel('ebc', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('degree', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('分数', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=12, fontweight='bold')
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=12, fontweight='bold')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.view_init(elev=25, azim=45)
    plt.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    fname = f"{sanitize(dataset)}_degree_ebc_3d_chart.svg"
    plt.savefig(os.path.join(output_dir, fname), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
def generate_all(md_path, output_dir):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = parse_md_sensitivity(content)
    for ds, tables in data.items():
        for t in tables:
            if t['type'] == 'dropout':
                plot_3d_bar_like_ablation(ds, 'dropout feat', t['x'], t['series'], output_dir)
                plot_lines(ds, 'dropout feat', t['x'], t['series'], output_dir)
            elif t['type'] == 'degree_ebc':
                plot_degree_ebc_3d(ds, t['rows'], t['cols'], t['matrix'], output_dir)
            elif t['type'] == 'cluster':
                plot_3d_waves(ds, r'$\lambda_{c}$', t['x'], t['series'], output_dir)
            elif t['type'] == 'ne':
                plot_3d_waves(ds, r'$\lambda_{NE}$', t['x'], t['series'], output_dir)
            elif t['type'] == 'kl':
                plot_3d_waves(ds, r'$\lambda_{KL}$', t['x'], t['series'], output_dir)
            print(f"Saved {ds} {t['type']} chart to {output_dir}")

if __name__ == '__main__':
    md_file_path = 'c:\\Users\\Miku12\\Desktop\\GraduationThesis\\Table\\参数敏感性实验表格.md'
    output_dir = 'c:\\Users\\Miku12\\Desktop\\GraduationThesis\\Table'
    generate_all(md_file_path, output_dir)

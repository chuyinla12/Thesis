#!/usr/bin/env python
"""
用已保存的 embedding 文件（npz）画 t-SNE 四子图：
  X（原始特征）
  GCA 视图（GCN 后）
  KNN 视图（GCN 后）
  h_all（融合）
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_4in1(npz_path: str, out_png: str, perplexity: int = 30, seed: int = 42, n_iter: int = 1000, sample_n: int = -1):
    """对四种表示分别做 t-SNE 并各自保存为单图（无标题）"""
    data = np.load(npz_path)
    X = data["X"]
    GCA = data["GCA"]
    KNN = data["KNN"]
    h_all = data["h_all"]
    labels = data["labels"]

    def _tsne(mat):
        """对 mat 做 tsne（参数对齐 tsne_all.py 的风格）"""
        ts = TSNE(
            n_components=2,
            init="pca",
            random_state=seed,
            perplexity=float(perplexity),
            learning_rate="auto",
            n_iter=n_iter,
        )
        emb2d = ts.fit_transform(mat)
        return emb2d

    def _maybe_sample(X, y, sample_n, seed):
        n = X.shape[0]
        if sample_n <= 0 or n <= sample_n:
            return X, y
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(n, size=int(sample_n), replace=False)
        return X[idx], y[idx]

    # 采样
    X_s, y_s = _maybe_sample(X, labels, sample_n, seed)
    GCA_s, _ = _maybe_sample(GCA, labels, sample_n, seed)
    KNN_s, _ = _maybe_sample(KNN, labels, sample_n, seed)
    h_all_s, _ = _maybe_sample(h_all, labels, sample_n, seed)

    print(f"running t-SNE on X (n={X_s.shape[0]})...")
    xy_X = _tsne(X_s)
    print(f"running t-SNE on GCA (n={GCA_s.shape[0]})...")
    xy_GCA = _tsne(GCA_s)
    print(f"running t-SNE on KNN (n={KNN_s.shape[0]})...")
    xy_KNN = _tsne(KNN_s)
    print(f"running t-SNE on h_all (n={h_all_s.shape[0]})...")
    xy_h = _tsne(h_all_s)

    # 颜色/点大小/透明度对齐：tab20 调色板，s=4，alpha=0.8，linewidths=0
    labels_np = y_s.astype(np.int64)
    classes = np.unique(labels_np)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(classes.size)]

    # 输出前缀
    base = os.path.splitext(out_png)[0]
    out_dir = os.path.dirname(base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _save_single(xy, suffix):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for i, c in enumerate(classes):
            mask = labels_np == c
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=4,
                c=[colors[i]],
                alpha=0.8,
                linewidths=0,
                label=str(int(c)),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ["left", "right", "top", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.set_frame_on(False)
        fig.tight_layout()
        out_path = f"{base}_{suffix}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("saved t-SNE plot ->", out_path)

    _save_single(xy_X, "raw")
    _save_single(xy_GCA, "gca")
    _save_single(xy_KNN, "knn")
    _save_single(xy_h, "hall")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="path to amap_embeddings.npz")
    parser.add_argument("--out", default="amap_tsne.svg", help="output svg path")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--sample_n", type=int, default=-1, help="number of samples for t-SNE, -1 for all")
    args = parser.parse_args()
    plot_tsne_4in1(args.npz, args.out, perplexity=args.perplexity, seed=args.seed, n_iter=args.n_iter, sample_n=args.sample_n)


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from pGRACE.dataset import _load_amap_npy, _stratified_sample_indices


def _select_perplexity(n, default_p):
    n = int(n)
    p = int(default_p)
    p = min(p, max(5, n // 10))
    p = max(5, min(p, n - 1)) if n > 1 else 5
    return p


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _scatter_like_final(ax, Z, y, title):
    y_np = np.asarray(y, dtype=np.int64).reshape(-1)
    classes = np.unique(y_np)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(classes.size)]
    for i, c in enumerate(classes):
        m = (y_np == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=4, c=[colors[i]], label=str(int(c)), alpha=0.8, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def _kmeans_acc(X, y, seed):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    C = int(np.max(y)) + 1 if y.size else 0
    if C <= 1:
        return 0.0
    km = KMeans(n_clusters=C, n_init=10, random_state=int(seed))
    pred = km.fit_predict(X)
    D = np.zeros((C, C), dtype=np.int64)
    for pi, ti in zip(pred, y):
        if pi < C and ti < C:
            D[pi, ti] += 1
    row_ind, col_ind = linear_sum_assignment(D.max() - D)
    acc = D[row_ind, col_ind].sum() / max(1, y.size)
    return float(acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=os.path.join("..", "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_n", type=int, default=4000)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--out_dir", type=str, default="vis_results")
    args = ap.parse_args()

    labels, X_unorm, _ = _load_amap_npy(args.data_root)
    X = np.asarray(X_unorm.todense(), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    N = int(X.shape[0])
    n_sample = min(int(args.sample_n), N)
    idx = _stratified_sample_indices(y, n=n_sample, seed=int(args.seed))
    Xs = X[idx]
    ys = y[idx]

    acc_raw = _kmeans_acc(Xs, ys, seed=int(args.seed))
    print("acc_raw", f"{acc_raw:.4f}")

    p = _select_perplexity(len(idx), int(args.perplexity))
    tsne = TSNE(n_components=2, perplexity=p, n_iter=int(args.n_iter), init="pca", learning_rate="auto", random_state=int(args.seed))
    Z = tsne.fit_transform(Xs)

    _ensure_dir(args.out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    _scatter_like_final(ax, Z, ys, "amap Raw Features")
    out_path = os.path.join(args.out_dir, "amap_tsne_raw.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("saved_fig", os.path.abspath(out_path))


if __name__ == "__main__":
    main()


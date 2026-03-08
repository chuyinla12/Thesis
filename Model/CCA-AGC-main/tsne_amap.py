import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import scipy.sparse as sp
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _find_file_recursive(root: str, filename_lower: str):
    filename_lower = str(filename_lower).lower()
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower() == filename_lower:
                return os.path.join(r, f)
    return None


def _load_amap_npy(data_root: str):
    if not os.path.isdir(data_root):
        raise FileNotFoundError(data_root)
    adj_path = _find_file_recursive(data_root, "amap_adj.npy")
    feat_path = _find_file_recursive(data_root, "amap_feat.npy")
    label_path = _find_file_recursive(data_root, "amap_label.npy")
    if adj_path is None or feat_path is None or label_path is None:
        raise FileNotFoundError("amap *_adj/feat/label.npy not found")
    adj = np.load(adj_path, allow_pickle=True)
    if adj.ndim == 0:
        adj = adj.item()
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = np.load(feat_path, allow_pickle=True).astype(np.float32)
    X_unorm = sp.csr_matrix(features)
    labels = np.load(label_path, allow_pickle=True)
    if labels.ndim > 1:
        labels = labels.flatten()
    labels = labels.astype(np.int64)
    return labels, X_unorm, adj


def _stratified_sample_indices(labels_np: np.ndarray, n: int, seed: int):
    labels_np = np.asarray(labels_np, dtype=np.int64).reshape(-1)
    N = int(labels_np.shape[0])
    if n >= N:
        return np.arange(N, dtype=np.int64)
    if n <= 0:
        raise ValueError("n must be positive")
    C = int(labels_np.max()) + 1 if N else 0
    counts = np.bincount(labels_np, minlength=C).astype(np.int64)
    probs = counts / max(1, N)
    ideal = probs * float(n)
    base = np.floor(ideal).astype(np.int64)
    base = np.minimum(base, counts)
    need = int(n - int(base.sum()))
    if need > 0:
        frac = ideal - base
        order = np.argsort(-frac)
        for c in order:
            if need <= 0:
                break
            if base[c] < counts[c]:
                base[c] += 1
                need -= 1
    cur = int(base.sum())
    if cur < n:
        for c in np.argsort(-counts):
            if cur >= n:
                break
            add = min(int(counts[c] - base[c]), int(n - cur))
            if add > 0:
                base[c] += add
                cur += add
    rng = np.random.RandomState(int(seed))
    out = []
    for c in range(C):
        k = int(base[c])
        if k <= 0:
            continue
        idx_c = np.where(labels_np == c)[0]
        if k >= idx_c.size:
            out.append(idx_c.astype(np.int64))
        else:
            out.append(rng.choice(idx_c, size=k, replace=False).astype(np.int64))
    if not out:
        return np.arange(min(n, N), dtype=np.int64)
    idx = np.concatenate(out, axis=0)
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False).astype(np.int64)
    idx = np.unique(idx)
    if idx.size < n:
        remain = np.setdiff1d(np.arange(N, dtype=np.int64), idx, assume_unique=False)
        extra = rng.choice(remain, size=(n - idx.size), replace=False).astype(np.int64)
        idx = np.concatenate([idx, extra], axis=0)
    return np.sort(idx.astype(np.int64))


def _select_perplexity(n, default_p):
    n = int(n)
    p = int(default_p)
    p = min(p, max(5, n // 10))
    p = max(5, min(p, n - 1)) if n > 1 else 5
    return p


def _try_load_pretrain_embeddings(repo_root: str):
    pkl_path = os.path.join(repo_root, "pretrain", "amap_contra.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k in ["z", "emb", "embedding", "feat", "h"]:
                if k in obj:
                    val = obj[k]
                    arr = np.asarray(val)
                    if arr.ndim == 2:
                        return arr.astype(np.float32)
        if isinstance(obj, (list, tuple)):
            for v in obj:
                arr = np.asarray(v)
                if arr.ndim == 2:
                    return arr.astype(np.float32)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=os.path.join("..", "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_n", type=int, default=4000)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--out_prefix", type=str, default=os.path.join("vis_results", "amap_cca_tsne"))
    ap.add_argument("--npz", type=str, default=None, help="optional npz with X/GCA/KNN/h_all/labels for four-view output")
    args = ap.parse_args()

    # If an npz is provided or found, output four separate figures (no titles)
    npz_path = args.npz
    if npz_path is None:
        cand = os.path.join(os.getcwd(), "amap_cca_emb.npz")
        if os.path.exists(cand):
            npz_path = cand
    if npz_path is not None and os.path.exists(npz_path):
        data = np.load(npz_path)
        X = data["X"]
        GCA = data["GCA"]
        KNN = data["KNN"]
        h_all = data["h_all"]
        labels = data["labels"].astype(np.int64)
        N = int(X.shape[0])
        n_sample = min(int(args.sample_n), N)
        idx = _stratified_sample_indices(labels, n=n_sample, seed=int(args.seed))
        Xs = X[idx]
        Gs = GCA[idx]
        Ks = KNN[idx]
        Hs = h_all[idx]
        ys = labels[idx]
        p = _select_perplexity(len(idx), int(args.perplexity))
        def _ts(Xi): return TSNE(n_components=2, perplexity=p, n_iter=int(args.n_iter), init="pca", learning_rate="auto", random_state=int(args.seed)).fit_transform(Xi)
        Z = {
            "raw": _ts(Xs),
            "gca": _ts(Gs),
            "knn": _ts(Ks),
            "hall": _ts(Hs),
        }
        y_np = np.asarray(ys, dtype=np.int64).reshape(-1)
        classes = np.unique(y_np)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(classes.size)]
        base = os.path.splitext(args.out_prefix)[0]
        os.makedirs(os.path.dirname(base), exist_ok=True)
        for key in ["raw", "gca", "knn", "hall"]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
            Zi = Z[key]
            for i, c in enumerate(classes):
                m = (y_np == c)
                ax.scatter(Zi[m, 0], Zi[m, 1], s=4, c=[colors[i]], label=str(int(c)), alpha=0.8, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ["left", "right", "top", "bottom"]:
                ax.spines[s].set_visible(False)
            ax.set_frame_on(False)
            fig.tight_layout()
            out_path = f"{base}_{key}.svg"
            fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print("saved_fig", os.path.abspath(out_path))
        return

    # Fallback: raw (and optional model) as separate figures, no titles
    labels, X_unorm, _ = _load_amap_npy(args.data_root)
    X = np.asarray(X_unorm.todense(), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    N = int(X.shape[0])
    n_sample = min(int(args.sample_n), N)
    idx = _stratified_sample_indices(y, n=n_sample, seed=int(args.seed))
    Xs = X[idx]
    ys = y[idx]

    p = _select_perplexity(len(idx), int(args.perplexity))
    tsne_raw = TSNE(n_components=2, perplexity=p, n_iter=int(args.n_iter), init="pca", learning_rate="auto", random_state=int(args.seed))
    Z_raw = tsne_raw.fit_transform(Xs)

    Z_model = None
    E = _try_load_pretrain_embeddings(os.path.dirname(os.path.abspath(__file__)))
    if E is not None and E.shape[0] == y.shape[0]:
        Es = E[idx]
        tsne_m = TSNE(n_components=2, perplexity=p, n_iter=int(args.n_iter), init="pca", learning_rate="auto", random_state=int(args.seed))
        Z_model = tsne_m.fit_transform(Es.astype(np.float32))

    y_np = np.asarray(ys, dtype=np.int64).reshape(-1)
    classes = np.unique(y_np)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(classes.size)]
    base = os.path.splitext(args.out_prefix)[0]
    os.makedirs(os.path.dirname(base), exist_ok=True)
    # raw
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
    for i, c in enumerate(classes):
        m = (y_np == c)
        ax.scatter(Z_raw[m, 0], Z_raw[m, 1], s=4, c=[colors[i]], label=str(int(c)), alpha=0.8, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["left", "right", "top", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_frame_on(False)
    fig.tight_layout()
    out_path_raw = f"{base}_raw.svg"
    fig.savefig(out_path_raw, format="svg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("saved_fig", os.path.abspath(out_path_raw))
    # model if available
    if Z_model is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=150)
        for i, c in enumerate(classes):
            m = (y_np == c)
            ax.scatter(Z_model[m, 0], Z_model[m, 1], s=4, c=[colors[i]], label=str(int(c)), alpha=0.8, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ["left", "right", "top", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.set_frame_on(False)
        fig.tight_layout()
        out_path_m = f"{base}_model.svg"
        fig.savefig(out_path_m, format="svg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("saved_fig", os.path.abspath(out_path_m))


if __name__ == "__main__":
    main()

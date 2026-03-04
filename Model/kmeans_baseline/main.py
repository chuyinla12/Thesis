import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np


def _add_finalmodel_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    fm = os.path.normpath(os.path.join(here, "..", "FinalModel"))
    if fm not in sys.path:
        sys.path.insert(0, fm)


def _default_extracted_root():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "FinalModel", "data_extracted"))
    if os.path.isdir(cand):
        return cand
    return os.path.normpath(os.path.join(here, "data_extracted"))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cora")
    p.add_argument("--data_root", type=str, default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data")))
    p.add_argument("--extracted_root", type=str, default=_default_extracted_root())

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--k", type=int, default=0)

    p.add_argument("--use_raw_features", type=int, default=0)
    p.add_argument("--l2_normalize", type=int, default=0)
    p.add_argument("--pca_dim", type=int, default=0)
    p.add_argument("--n_init", type=int, default=20)

    p.add_argument("--pubmed_use_small", type=int, default=0)
    p.add_argument("--pubmed_small_n", type=int, default=8000)
    p.add_argument("--pubmed_small_rebuild", type=int, default=0)

    p.add_argument("--amazon_computers_use_small", type=int, default=0)
    p.add_argument("--amazon_computers_small_n", type=int, default=8000)
    p.add_argument("--amazon_computers_small_rebuild", type=int, default=0)

    p.add_argument("--out_dir", type=str, default=os.path.normpath(os.path.join(os.path.dirname(__file__), "runs")))
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def _l2_normalize(x: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _maybe_pca(x: np.ndarray, pca_dim: int, seed: int):
    if int(pca_dim) <= 0:
        return x, None
    pca_dim = int(pca_dim)
    if pca_dim >= x.shape[1]:
        return x, None
    from sklearn.decomposition import PCA

    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=int(seed))
    x2 = pca.fit_transform(x)
    return x2, {"pca_dim": int(pca_dim), "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}


def _run_once(x: np.ndarray, k: int, seed: int, n_init: int):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=int(n_init))
    pred = km.fit_predict(x)
    return pred.astype(np.int64)


def _to_numpy_1d(x):
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy().reshape(-1)
    except Exception:
        pass
    return np.asarray(x).reshape(-1)


def _to_numpy_2d(x):
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = _parse_args()
    _add_finalmodel_to_path()

    from data import load_dataset
    from utils import eva, set_seed

    set_seed(int(args.seed))
    labels, _, features, _, feature_label = load_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        extracted_root=args.extracted_root,
        seed=int(args.seed),
        pubmed_small_n=int(args.pubmed_small_n),
        pubmed_small_rebuild=bool(int(args.pubmed_small_rebuild)),
        pubmed_use_small=bool(int(args.pubmed_use_small)),
        amazon_computers_small_n=int(args.amazon_computers_small_n),
        amazon_computers_small_rebuild=bool(int(args.amazon_computers_small_rebuild)),
        amazon_computers_use_small=bool(int(args.amazon_computers_use_small)),
    )

    y_true = _to_numpy_1d(labels).astype(np.int64)
    x = feature_label if int(args.use_raw_features) else features
    x = _to_numpy_2d(x).astype(np.float32)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    if int(args.l2_normalize):
        x = _l2_normalize(x)

    pca_info: Optional[Dict] = None
    x, pca_info = _maybe_pca(x, pca_dim=int(args.pca_dim), seed=int(args.seed))

    k = int(args.k) if int(args.k) > 0 else int(np.max(y_true) + 1)
    if k <= 1:
        raise ValueError(f"Invalid k={k}, y_true max={int(np.max(y_true))}")

    metrics = []
    for i in range(int(args.repeats)):
        seed_i = int(args.seed) + i
        y_pred = _run_once(x, k=k, seed=seed_i, n_init=int(args.n_init))
        nmi, acc, ari, f1 = eva(y_true, y_pred, epoch=i, visible=True)
        metrics.append({"run": i, "seed": seed_i, "acc": float(acc), "nmi": float(nmi), "ari": float(ari), "f1": float(f1)})

    accs = np.asarray([m["acc"] for m in metrics], dtype=np.float64)
    nmis = np.asarray([m["nmi"] for m in metrics], dtype=np.float64)
    aris = np.asarray([m["ari"] for m in metrics], dtype=np.float64)
    f1s = np.asarray([m["f1"] for m in metrics], dtype=np.float64)
    summary = {
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=0)),
        "nmi_mean": float(nmis.mean()),
        "nmi_std": float(nmis.std(ddof=0)),
        "ari_mean": float(aris.mean()),
        "ari_std": float(aris.std(ddof=0)),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=0)),
    }
    print("summary:", " ".join([f"{k}={v:.4f}" for k, v in summary.items()]))

    tag = str(args.tag).strip()
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, str(args.dataset).lower(), tag)
    payload = {
        "args": vars(args),
        "k": int(k),
        "pca": pca_info,
        "runs": metrics,
        "summary": summary,
    }
    _save_json(os.path.join(out_dir, "kmeans_results.json"), payload)
    print("saved:", os.path.join(out_dir, "kmeans_results.json"))


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data import load_dataset
from models import FinalModel
from utils import get_device, set_seed
from views import build_gca_view, build_knn_adj, make_message_passing_adj, prune_high_ebc_edges, prune_low_degree_edges

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXTRACTED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_extracted")
VIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, "vis_results")

if not os.path.exists(DATA_DIR):
    PARENT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data")
    if os.path.exists(PARENT_DATA_DIR):
        DATA_DIR = PARENT_DATA_DIR

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _list_datasets(runs_dir):
    out = []
    if os.path.isdir(runs_dir):
        for n in os.listdir(runs_dir):
            d = os.path.join(runs_dir, n)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "best.pt")):
                out.append(n)
    return sorted(out)

def _load_ckpt(runs_dir, dataset):
    ckpt_path = os.path.join(runs_dir, str(dataset), "best.pt")
    return torch.load(ckpt_path, map_location="cpu")

def _select_perplexity(n, default_p):
    n = int(n)
    p = int(default_p)
    p = min(p, max(5, n // 10))
    p = max(5, min(p, n - 1)) if n > 1 else 5
    return p

def _prepare_model_and_views(state, labels, features, adj_label, device, seed):
    args = state.get("args", {}) or {}
    class_num = int(labels.max().item()) + 1
    N = int(features.size(0))
    input_dim_x = int(features.size(1))
    hidden_dim = int(args.get("hidden_dim", 256))
    output_dim_g = int(args.get("output_dim_g", 64))
    gcn_dropout = float(args.get("gcn_dropout", 0.1))
    gcn_layers = int(args.get("gcn_layers", 2))
    gcn_impl = str(args.get("gcn_impl", "dense"))
    classifier_hidden = args.get("classifier_hidden", [128, 64])
    model = FinalModel(
        input_dim_x=input_dim_x,
        node_num=N,
        hidden_dim=hidden_dim,
        output_dim_g=output_dim_g,
        class_num=class_num,
        view_num=2,
        gcn_dropout=gcn_dropout,
        gcn_layers=gcn_layers,
        gcn_impl=gcn_impl,
        classifier_hidden=classifier_hidden,
    ).to(device)
    sd = state["model"]
    if isinstance(sd, dict):
        sd = {k: v for k, v in sd.items() if not str(k).startswith("reg_a_enc")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    knn_k = int(args.get("knn_k", 20))
    knn_metric = str(args.get("knn_metric", "cosine"))
    p_low_deg = float(args.get("p_low_deg", 0.0))
    low_deg_score = str(args.get("low_deg_score", "min"))
    p_high_ebc = float(args.get("p_high_ebc", 0.0))
    ebc_approx_k = int(args.get("ebc_approx_k", 256))
    adj_knn = build_knn_adj(features, k=knn_k, metric=knn_metric)
    adj_knn = prune_low_degree_edges(adj_knn, ratio=p_low_deg, score=low_deg_score)
    adj_knn = prune_high_ebc_edges(adj_knn, ratio=p_high_ebc, approx_k=ebc_approx_k, seed=int(seed))
    adj_knn = torch.clamp(adj_knn, 0, 1).to(device)
    adj_knn.fill_diagonal_(0.0)
    adj_knn_mp = make_message_passing_adj(adj_knn)
    gca_drop_edge_p = 0.0
    gca_drop_feat_p = 0.0
    with torch.no_grad():
        x_gca, adj_gca, adj_gca_mp = build_gca_view(features, adj_label, gca_drop_edge_p, gca_drop_feat_p)
        x_gca = x_gca.to(device)
        adj_gca = adj_gca.to(device)
        adj_gca.fill_diagonal_(0.0)
        adj_gca_mp = adj_gca_mp.to(device)
        xs = [x_gca, features]
        adjs_labels = [adj_gca, adj_knn]
        adjs_mp = [adj_gca_mp, adj_knn_mp]
        homo_rate = state.get("homo_rate", [0.5, 0.5])
        if homo_rate is None or len(homo_rate) != 2:
            homo_rate = [0.5, 0.5]
        update_weights = int(args.get("update_weights", 0))
        if update_weights == 1:
            w_raw = torch.tensor([float(homo_rate[0]), float(homo_rate[1])], dtype=torch.float32)
            w_raw = torch.clamp(w_raw, min=float(args.get("weights_min", 0.05)))
            w_raw = w_raw / (w_raw.sum() + 1e-12)
            weights_h = w_raw.to(device)
        else:
            weights_h = (torch.ones(2, device=device) / 2.0).to(device)
        out = model(xs, adjs_mp, adjs_labels, weights_h, homo_rate, compute_x_pred=False, compute_a_logits=False)
        hs = out[1]
        h_all = out[2]
    return hs, h_all

def _tsne_2d(X, seed, perplexity, n_iter):
    tsne = TSNE(n_components=2, init="pca", random_state=int(seed), perplexity=float(perplexity), learning_rate="auto", n_iter=int(n_iter))
    Z = tsne.fit_transform(X)
    return Z

def _scatter(ax, Z, y, title):
    y_np = y.astype(np.int64)
    classes = np.unique(y_np)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(classes.size)]
    for i, c in enumerate(classes):
        m = y_np == c
        ax.scatter(Z[m, 0], Z[m, 1], s=4, c=[colors[i]], label=str(int(c)), alpha=0.8, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def _maybe_sample(X, y, sample_n, seed):
    n = X.shape[0]
    if sample_n <= 0 or n <= sample_n:
        return X, y
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(n, size=int(sample_n), replace=False)
    return X[idx], y[idx]

def run_for_dataset(dataset, data_root, extracted_root, runs_dir, out_dir, sample_n, p_default, n_iter, seed):
    state = _load_ckpt(runs_dir, dataset)
    set_seed(seed)
    device = get_device(True)
    candidates = [dataset]
    dsl = dataset.lower()
    if dsl == "pubmed":
        candidates.append("pubmed-small")
    if dsl in ["amazon_electronics_computers", "amazon-computers", "amazon_computers"]:
        candidates.append("amazon_electronics_computers-small")
    last_err = None
    for ds_opt in candidates:
        try:
            labels, adj, features, adj_label, feature_label = load_dataset(ds_opt, data_root, extracted_root)
            labels = labels.to(device)
            features = features.to(device)
            adj_label = adj_label.to(device).to(torch.float32)
            x_raw = feature_label if torch.is_tensor(feature_label) else features
            hs, h_all = _prepare_model_and_views(state, labels, features, adj_label, device, seed)
            X_raw = x_raw.detach().cpu().numpy()
            y = labels.detach().cpu().numpy()
            E = h_all.detach().cpu().numpy()
            X_raw_s, y_s = _maybe_sample(X_raw, y, sample_n, seed)
            E_s, y_s2 = _maybe_sample(E, y, sample_n, seed)
            p1 = _select_perplexity(X_raw_s.shape[0], p_default)
            p2 = _select_perplexity(E_s.shape[0], p_default)
            Z_raw = _tsne_2d(X_raw_s, seed, p1, n_iter)
            Z_emb = _tsne_2d(E_s, seed, p2, n_iter)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            _scatter(axes[0], Z_raw, y_s, "(a) Raw X")
            _scatter(axes[1], Z_emb, y_s2, "(b) h_all")
            fig.suptitle(str(dataset))
            _ensure_dir(out_dir)
            out_path = os.path.join(out_dir, f"{str(dataset)}_tsne.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print("saved_tsne", os.path.abspath(out_path))
            return out_path
        except Exception as e:
            last_err = e
            continue
    raise last_err

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--runs_dir", type=str, default=RUNS_DIR)
    p.add_argument("--data_root", type=str, default=DATA_DIR)
    p.add_argument("--extracted_root", type=str, default=EXTRACTED_DATA_DIR)
    p.add_argument("--out_dir", type=str, default=VIS_RESULTS_DIR)
    p.add_argument("--sample_n", type=int, default=4000)
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.datasets is None or len(str(args.datasets).strip()) == 0:
        datasets = _list_datasets(args.runs_dir)
    else:
        datasets = [s.strip() for s in str(args.datasets).split(",") if s.strip()]
    for ds in datasets:
        try:
            run_for_dataset(
                dataset=ds,
                data_root=args.data_root,
                extracted_root=args.extracted_root,
                runs_dir=args.runs_dir,
                out_dir=args.out_dir,
                sample_n=int(args.sample_n),
                p_default=int(args.perplexity),
                n_iter=int(args.n_iter),
                seed=int(args.seed),
            )
        except Exception as e:
            print("failed", ds, str(e))

if __name__ == "__main__":
    main()


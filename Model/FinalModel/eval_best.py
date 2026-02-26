import argparse
import os

import numpy as np
import torch
from sklearn.cluster import KMeans

from data import load_dataset
from models import FinalModel
from utils import eva, get_device, set_seed
from views import build_gca_view, build_knn_adj, make_message_passing_adj, prune_high_ebc_edges, prune_low_degree_edges


def _default_save_dir(root, dataset):
    return os.path.join(root, "runs", str(dataset))


def _weights_from_homo(homo_rate, weights_min=0.05):
    w = torch.tensor([float(homo_rate[0]), float(homo_rate[1])], dtype=torch.float32)
    w = torch.clamp(w, min=float(weights_min))
    w = w / (w.sum() + 1e-12)
    return w


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=os.path.join(_default_save_dir(root, "cora"), "best.pt"))
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--extracted_root", type=str, default=None)
    p.add_argument("--cuda", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_mode", type=str, default="kmeans")
    p.add_argument("--eval_embed", type=str, default="h_all")
    p.add_argument("--kmeans_n_init", type=int, default=24)
    p.add_argument("--eval_no_aug", type=int, default=1)
    p.add_argument("--use_saved_snapshot", type=int, default=1)
    args = p.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = state.get("args", {}) or {}
    snapshot = state.get("eval_snapshot", None)
    if int(args.use_saved_snapshot) == 1 and isinstance(snapshot, dict):
        y_true = snapshot.get("y_true", None)
        y_pred = snapshot.get("y_pred", None)
        if torch.is_tensor(y_true) and torch.is_tensor(y_pred):
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            nmi, acc, ari, f1 = eva(y_true_np, y_pred_np, epoch=0, visible=True)
            m = snapshot.get("metrics", {}) if isinstance(snapshot.get("metrics", {}), dict) else {}
            print("ckpt", ckpt_path)
            print("snapshot_epoch", snapshot.get("epoch", state.get("epoch", None)))
            print("snapshot_eval_embed", snapshot.get("eval_embed", None), "snapshot_eval_mode", snapshot.get("eval_mode", None))
            print("acc", float(acc), "nmi", float(nmi), "ari", float(ari), "f1", float(f1))
            print("saved_acc", m.get("acc", None), "saved_nmi", m.get("nmi", None), "saved_ari", m.get("ari", None), "saved_f1", m.get("f1", None))
            return

    dataset = args.dataset if args.dataset is not None else ckpt_args.get("dataset", "cora")
    data_root = args.data_root if args.data_root is not None else ckpt_args.get("data_root", os.path.join(root, "data"))
    extracted_root = args.extracted_root if args.extracted_root is not None else ckpt_args.get("extracted_root", os.path.join(root, "data_extracted"))
    extracted_root = extracted_root if extracted_root is not None else os.path.join(root, "data_extracted")

    set_seed(args.seed)
    labels, adj, features, adj_label, feature_label = load_dataset(dataset, data_root, extracted_root)

    device = get_device(bool(args.cuda))
    labels = labels.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device).to(torch.float32)

    class_num = int(labels.max().item()) + 1
    N = int(features.size(0))
    input_dim_x = int(features.size(1))

    hidden_dim = int(ckpt_args.get("hidden_dim", 256))
    output_dim_g = int(ckpt_args.get("output_dim_g", 64))
    gcn_dropout = float(ckpt_args.get("gcn_dropout", 0.1))
    gcn_layers = int(ckpt_args.get("gcn_layers", 2))
    gcn_impl = str(ckpt_args.get("gcn_impl", "dense"))
    classifier_hidden = ckpt_args.get("classifier_hidden", [128, 64])

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
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    knn_k = int(ckpt_args.get("knn_k", 20))
    knn_metric = str(ckpt_args.get("knn_metric", "cosine"))
    p_low_deg = float(ckpt_args.get("p_low_deg", 0.0))
    low_deg_score = str(ckpt_args.get("low_deg_score", "min"))
    p_high_ebc = float(ckpt_args.get("p_high_ebc", 0.0))
    ebc_approx_k = int(ckpt_args.get("ebc_approx_k", 256))

    adj_knn = build_knn_adj(features, k=knn_k, metric=knn_metric)
    adj_knn = prune_low_degree_edges(adj_knn, ratio=p_low_deg, score=low_deg_score)
    adj_knn = prune_high_ebc_edges(adj_knn, ratio=p_high_ebc, approx_k=ebc_approx_k, seed=int(args.seed))
    adj_knn = torch.clamp(adj_knn, 0, 1).to(device)
    adj_knn.fill_diagonal_(0.0)
    adj_knn_mp = make_message_passing_adj(adj_knn)

    gca_drop_edge_p = float(ckpt_args.get("gca_drop_edge_p", 0.0))
    gca_drop_feat_p = float(ckpt_args.get("gca_drop_feat_p", 0.0))
    if int(args.eval_no_aug) == 1:
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
        update_weights = int(ckpt_args.get("update_weights", 0))
        if update_weights == 1:
            weights_h = _weights_from_homo(homo_rate, weights_min=float(ckpt_args.get("weights_min", 0.05))).to(device)
        else:
            weights_h = (torch.ones(2, device=device) / 2.0).to(device)

        out = model(xs, adjs_mp, adjs_labels, weights_h, homo_rate)
        hs = out[1]
        h_all = out[2]
        cluster_all = out[12]

        ee = str(args.eval_embed).strip().lower()
        if ee == "h0":
            emb = hs[0]
        elif ee == "h1":
            emb = hs[1]
        elif ee == "concat":
            emb = torch.cat([hs[0], hs[1]], dim=1)
        else:
            emb = h_all

        em = emb.detach().cpu().numpy()
        if str(args.eval_mode).strip().lower() == "argmax":
            y_pred = torch.argmax(cluster_all, dim=1).detach().cpu().numpy()
        else:
            km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=int(args.seed))
            y_pred = km.fit_predict(em)

    y_true = labels.detach().cpu().numpy()
    nmi, acc, ari, f1 = eva(y_true, y_pred, epoch=0, visible=True)
    print("ckpt", ckpt_path)
    print("eval_embed", args.eval_embed, "eval_mode", args.eval_mode, "seed", args.seed)
    print("acc", float(acc), "nmi", float(nmi), "ari", float(ari), "f1", float(f1))


if __name__ == "__main__":
    main()

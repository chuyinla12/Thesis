import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm
from sklearn.cluster import KMeans
import torch.nn.functional as F

# 假设你有这些自定义模块，路径需确保正确
from data import load_dataset
from models import FinalModel
from utils import eva, get_device, set_seed
from views import build_gca_view, build_knn_adj, make_message_passing_adj, prune_high_ebc_edges, prune_low_degree_edges

# ======================== 核心路径常量（固定相对路径）========================
# 项目根目录：当前脚本所在的目录（作为所有相对路径的基准）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 固定的子目录定义（全部基于PROJECT_ROOT的相对路径）
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")          # 模型 checkpoint 存储目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")          # 原始数据目录
EXTRACTED_DATA_DIR = os.path.join(PROJECT_ROOT, "data_extracted")  # 提取后的数据目录
VIS_RESULTS_DIR = os.path.join(PROJECT_ROOT, "vis_results")  # 可视化结果保存目录

# 如果本地data目录不存在，尝试使用上级目录的data
if not os.path.exists(DATA_DIR):
    PARENT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data")
    if os.path.exists(PARENT_DATA_DIR):
        DATA_DIR = PARENT_DATA_DIR

def _default_save_dir(dataset):
    """固定相对路径：runs/数据集名"""
    return os.path.join(RUNS_DIR, str(dataset))

def _weights_from_homo(homo_rate, weights_min=0.05):
    w = torch.tensor([float(homo_rate[0]), float(homo_rate[1])], dtype=torch.float32)
    w = torch.clamp(w, min=float(weights_min))
    w = w / (w.sum() + 1e-12)
    return w

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _pick_indices(labels: torch.Tensor, sample_n: int, seed: int):
    n = int(labels.numel())
    sample_n = int(sample_n)
    if sample_n <= 0 or sample_n >= n:
        return torch.arange(n, device=labels.device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    return perm[:sample_n].to(labels.device)

def _save_adj_figure(dataset: str, labels: torch.Tensor, adj_gca: torch.Tensor, adj_knn: torch.Tensor, adj_label: torch.Tensor, out_path: str, sample_n: int, seed: int, sort_by_label: int, dpi: int):
    idx = _pick_indices(labels, sample_n=sample_n, seed=seed)
    y = labels[idx].detach().cpu()
    if int(sort_by_label) == 1:
        order = torch.argsort(y)
        idx = idx[order.to(idx.device)]
        y = y[order]

    A_gca = adj_gca[idx][:, idx].detach().cpu()
    A_knn = adj_knn[idx][:, idx].detach().cpu()
    A_gt = adj_label[idx][:, idx].detach().cpu()
    A_ours = torch.clamp(A_gca + A_knn, 0, 1)

    mats = [
        (A_gca, "(a) GCA"),
        (A_knn, "(b) KNN"),
        (A_ours, "(c) Ours"),
        (A_gt, "(d) Ground truth"),
    ]

    cmap = ListedColormap(["#ffffff", "#4d0010"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.reshape(-1)
    for ax, (A, title) in zip(axes, mats):
        B = (A > 0).numpy().astype(np.uint8)
        n = int(B.shape[0])
        if n >= 400:
            target = 400
            bs = max(1, int(np.ceil(n / target)))
            m = int(np.ceil(n / bs))
            pad = m * bs - n
            if pad > 0:
                Bp = np.pad(B, ((0, pad), (0, pad)), mode="constant", constant_values=0)
            else:
                Bp = B
            dens = Bp.reshape(m, bs, m, bs).sum(axis=(1, 3)).astype(np.float32) / float(bs * bs)
            vmax = float(np.percentile(dens, 99.5))
            vmax = max(vmax, 1e-6)
            ax.imshow(
                dens,
                cmap="Reds",
                interpolation="nearest",
                norm=PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax),
            )
        else:
            ax.imshow(B, cmap=cmap, interpolation="none", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.text(0.5, 0.03, str(dataset), ha="center", va="center", fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

def _cosine_sim_matrix(x: torch.Tensor):
    x = x.to(torch.float32)
    x = F.normalize(x, p=2, dim=1)
    s = torch.mm(x, x.t())
    s = torch.clamp(s, -1.0, 1.0)
    return s

def _save_emb_figure(
    dataset: str,
    labels: torch.Tensor,
    x_raw: torch.Tensor,
    h_gca: torch.Tensor,
    h_knn: torch.Tensor,
    h_all: torch.Tensor,
    out_path: str,
    sample_n: int,
    seed: int,
    sort_by_label: int,
    dpi: int,
):
    idx = _pick_indices(labels, sample_n=sample_n, seed=seed)
    y = labels[idx].detach().cpu()
    if int(sort_by_label) == 1:
        order = torch.argsort(y)
        idx = idx[order.to(idx.device)]
        y = y[order]

    mats = [
        (_cosine_sim_matrix(x_raw[idx][:].detach().cpu()), "(a) X"),
        (_cosine_sim_matrix(h_gca[idx][:].detach().cpu()), "(b) GCA (GCN)"),
        (_cosine_sim_matrix(h_knn[idx][:].detach().cpu()), "(c) KNN (GCN)"),
        (_cosine_sim_matrix(h_all[idx][:].detach().cpu()), "(d) h_all"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.reshape(-1)
    for ax, (S, title) in zip(axes, mats):
        M = ((S + 1.0) * 0.5).numpy().astype(np.float32)
        vmin = float(np.percentile(M, 5))
        vmax = float(np.percentile(M, 99))
        if vmax <= vmin:
            vmin = 0.0
            vmax = 1.0
        ax.imshow(M, cmap="Reds", interpolation="nearest", norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))
        ax.set_title(title)
        ax.axis("off")
    fig.text(0.5, 0.03, str(dataset), ha="center", va="center", fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)

def main():
    # ======================== 命令行参数 ========================
    p = argparse.ArgumentParser()
    # 默认 checkpoint 路径：runs/cora/best.pt（固定相对路径）
    p.add_argument("--ckpt", type=str, default=os.path.join(_default_save_dir("cora"), "best.pt"))
    p.add_argument("--dataset", type=str, default=None)
    # 以下路径参数改为可选，默认使用固定相对路径
    p.add_argument("--data_root", type=str, default=DATA_DIR)
    p.add_argument("--extracted_root", type=str, default=EXTRACTED_DATA_DIR)
    p.add_argument("--fig_dir", type=str, default=VIS_RESULTS_DIR)
    
    p.add_argument("--cuda", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_mode", type=str, default="kmeans")
    p.add_argument("--eval_embed", type=str, default="h_all")
    p.add_argument("--kmeans_n_init", type=int, default=24)
    p.add_argument("--eval_no_aug", type=int, default=1)
    p.add_argument("--use_saved_snapshot", type=int, default=1)
    p.add_argument("--save_fig", type=int, default=0)
    p.add_argument("--save_emb_fig", type=int, default=0)
    p.add_argument("--fig_sample_n", type=int, default=1000)
    p.add_argument("--fig_sort_by_label", type=int, default=1)
    p.add_argument("--fig_dpi", type=int, default=300)
    p.add_argument("--x_source", type=str, default="feature_label")
    args = p.parse_args()

    # ======================== 加载 checkpoint ========================
    ckpt_path = os.path.abspath(args.ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = state.get("args", {}) or {}
    snapshot = state.get("eval_snapshot", None)
    
    # 优先使用 snapshot 中的评估结果（如果存在）
    if (
        int(args.use_saved_snapshot) == 1
        and int(args.save_fig) == 0
        and int(args.save_emb_fig) == 0
        and isinstance(snapshot, dict)
    ):
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

    # ======================== 确定数据集和路径 ========================
    # 数据集名称：命令行参数 > checkpoint 参数 > 默认 cora
    dataset = args.dataset or ckpt_args.get("dataset", "cora")
    # 数据路径：命令行参数（已默认固定路径）> checkpoint 参数 > 固定路径
    data_root = args.data_root or ckpt_args.get("data_root", DATA_DIR)
    extracted_root = args.extracted_root or ckpt_args.get("extracted_root", EXTRACTED_DATA_DIR)
    fig_dir = args.fig_dir or VIS_RESULTS_DIR

    # ======================== 初始化环境和数据 ========================
    set_seed(args.seed)
    labels, adj, features, adj_label, feature_label = load_dataset(dataset, data_root, extracted_root)

    device = get_device(bool(args.cuda))
    labels = labels.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device).to(torch.float32)
    feature_label = feature_label.to(device) if torch.is_tensor(feature_label) else feature_label

    class_num = int(labels.max().item()) + 1
    N = int(features.size(0))
    input_dim_x = int(features.size(1))

    # ======================== 初始化模型 ========================
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

    # ======================== 构建邻接矩阵 ========================
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

    # ======================== 模型推理 ========================
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

        # 选择评估用的嵌入
        ee = str(args.eval_embed).strip().lower()
        if ee == "h0":
            emb = hs[0]
        elif ee == "h1":
            emb = hs[1]
        elif ee == "concat":
            emb = torch.cat([hs[0], hs[1]], dim=1)
        else:
            emb = h_all

        # 聚类评估
        em = emb.detach().cpu().numpy()
        if str(args.eval_mode).strip().lower() == "argmax":
            y_pred = torch.argmax(cluster_all, dim=1).detach().cpu().numpy()
        else:
            km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=int(args.seed))
            y_pred = km.fit_predict(em)

    # ======================== 评估和可视化 ========================
    y_true = labels.detach().cpu().numpy()
    nmi, acc, ari, f1 = eva(y_true, y_pred, epoch=0, visible=True)
    print("ckpt", ckpt_path)
    print("eval_embed", args.eval_embed, "eval_mode", args.eval_mode, "seed", args.seed)
    print("acc", float(acc), "nmi", float(nmi), "ari", float(ari), "f1", float(f1))
    
    # 保存邻接矩阵可视化图
    if int(args.save_fig) == 1:
        out_name = f"{str(dataset)}_eval_{str(args.eval_embed)}_{str(args.eval_mode)}.png"
        out_path = os.path.join(fig_dir, out_name)
        _save_adj_figure(
            dataset=str(dataset),
            labels=labels,
            adj_gca=adj_gca,
            adj_knn=adj_knn,
            adj_label=adj_label,
            out_path=out_path,
            sample_n=int(args.fig_sample_n),
            seed=int(args.seed),
            sort_by_label=int(args.fig_sort_by_label),
            dpi=int(args.fig_dpi),
        )
        print("saved_fig", os.path.abspath(out_path))
    
    # 保存嵌入可视化图
    if int(args.save_emb_fig) == 1:
        xs = str(args.x_source).strip().lower()
        x0 = feature_label if xs in ["feature_label", "raw", "x"] else features
        out_name = f"{str(dataset)}_emb_heatmap.png"
        out_path = os.path.join(fig_dir, out_name)
        _save_emb_figure(
            dataset=str(dataset),
            labels=labels,
            x_raw=x0,
            h_gca=hs[0],
            h_knn=hs[1],
            h_all=h_all,
            out_path=out_path,
            sample_n=int(args.fig_sample_n),
            seed=int(args.seed),
            sort_by_label=int(args.fig_sort_by_label),
            dpi=int(args.fig_dpi),
        )
        print("saved_emb_fig", os.path.abspath(out_path))

if __name__ == "__main__":
    main()
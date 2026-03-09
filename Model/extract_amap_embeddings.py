#!/usr/bin/env python
"""
通用脚本：
1. 加载已保存的 GCA-main 或 CCA-AGC-main checkpoint
2. 抽取 amap 数据集的四种 embedding：
   - X（原始特征）
   - GCA 视图（GCN 后）
   - KNN 视图（GCN 后）
   - h_all（融合）
3. 保存为 .npz 文件，供后续 t-SNE 绘图使用
"""
import argparse
import os
import sys
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# 注意：不要提前把两个 repo 都插入 sys.path，以免 FinalModel 的相对 import 被覆盖

# 通用工具：加载数据
def load_graph_data(dataset: str, data_root: str):
    fm_dir = os.path.join(os.path.dirname(__file__), "FinalModel")
    if fm_dir not in sys.path:
        sys.path.insert(0, fm_dir)
    import FinalModel.data as fm_data
    extracted_root = os.path.join(fm_dir, "data_extracted")
    labels, adj, features, _, feature_label = fm_data.load_dataset(
        dataset=dataset,
        data_root=data_root,
        extracted_root=extracted_root,
    )
    x = feature_label.numpy()
    y = labels.numpy()
    adj_dense = adj.numpy()
    return x, adj_dense, y

# ---------- GCA-main ----------
def load_gca_ckpt(ckpt_path: str, device: str, in_channels: int):
    """返回 encoder（重建后已加载权重）"""
    ckpt = torch.load(ckpt_path, map_location=device)
    from pGRACE.model import Encoder
    # 推断超参
    out_channels = None
    num_layers = 2
    act_name = "relu"
    if "param" in ckpt:
        p = ckpt["param"]
        out_channels = p.get("num_hidden") or p.get("hidden")
        num_layers = int(p.get("num_layers", num_layers))
        act_name = str(p.get("activation", act_name)).lower()
    if out_channels is None:
        for k, v in ckpt["encoder_state_dict"].items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                out_channels = max(out_channels or 0, v.shape[0])
    # 激活函数与层数对齐 ckpt
    if act_name in ["prelu", "p-relu", "p_relu"]:
        activation = torch.nn.PReLU()
    else:
        activation = torch.nn.ReLU()
    encoder = Encoder(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        k=int(num_layers),
        skip=False,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    return encoder

def gca_embed(encoder, x, adj, device):
    """返回 GCN 后的 embedding (N x d) """
    import torch_geometric.utils as tg_utils
    edge_index, _ = tg_utils.dense_to_sparse(torch.FloatTensor(adj))
    edge_index = edge_index.to(device)
    x_t = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        z = encoder(x_t, edge_index)
    return z.cpu().numpy()

# ---------- CCA-AGC-main ----------
def load_cca_ckpt(ckpt_path: str, device: str, in_channels: int):
    """返回 encoder（只用 encoder 计算视图/融合）"""
    ckpt = torch.load(ckpt_path, map_location=device)
    from module import Encoder
    # 重建 encoder（使用保存的超参 + 当前数据维度）
    encoder = Encoder(
        in_channels=in_channels,
        out_channels=ckpt["args"]["out_dim"],
        hidden=ckpt["args"]["hidden"],
        activation=ckpt["args"]["activation"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    return encoder

def cca_embed(encoder, x, adj, device):
    """返回 h (encoder 后) """
    import torch_geometric.utils as tg_utils
    edge_index, _ = tg_utils.dense_to_sparse(torch.FloatTensor(adj))
    edge_index = edge_index.to(device)
    x_t = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        h = encoder(x_t, edge_index)
    return h.cpu().numpy()

# ---------- KNN 视图 ----------
def build_knn_adj(x, k=15, metric="cosine"):
    """返回 kNN 图邻接矩阵 (N x N) """
    sim = cosine_similarity(x)
    # 每行取 top-k
    idx = np.argsort(-sim, axis=1)[:, 1:k+1]
    adj_knn = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        adj_knn[i, idx[i]] = 1.0
    adj_knn = (adj_knn + adj_knn.T) > 0
    return adj_knn.astype(np.float32)

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=["gca", "cca"], required=True,
                        help="which repo checkpoint to load")
    parser.add_argument("--ckpt", required=True,
                        help="path to checkpoint .pth/.pkl")
    parser.add_argument("--dataset", type=str, default="amap")
    parser.add_argument("--data_root", default=r"c:\Users\Miku12\Desktop\GraduationThesis\Model\data")
    parser.add_argument("--out", default="embeddings.npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--knn_k", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x, adj, y = load_graph_data(args.dataset, args.data_root)

    if args.repo == "gca":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GCA-main"))
        encoder = load_gca_ckpt(args.ckpt, device, in_channels=x.shape[1])
        z_gca = gca_embed(encoder, x, adj, device)
        h_all = z_gca
        adj_knn = build_knn_adj(x, k=args.knn_k)
        z_knn = gca_embed(encoder, x, adj_knn, device)
        np.savez(args.out,
                 X=x,
                 GCA=z_gca,
                 KNN=z_knn,
                 h_all=h_all,
                 labels=y)
    else:  # cca
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CCA-AGC-main"))
        encoder = load_cca_ckpt(args.ckpt, device, in_channels=x.shape[1])
        h_gca = cca_embed(encoder, x, adj, device)
        h_all = h_gca
        adj_knn = build_knn_adj(x, k=args.knn_k)
        h_knn = cca_embed(encoder, x, adj_knn, device)
        np.savez(args.out,
                 X=x,
                 GCA=h_gca,
                 KNN=h_knn,
                 h_all=h_all,
                 labels=y)

    print(f"saved embeddings -> {args.out}")


if __name__ == "__main__":
    main()

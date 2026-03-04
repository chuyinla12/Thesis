import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, degree, to_undirected
from torch_scatter import scatter
import networkx as nx


def _row_norm(A: torch.Tensor):
    A = A.to(torch.float32)
    return A / (A.sum(dim=1, keepdim=True) + 1e-12)


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w_mean = w.mean()
    if w_mean == 0:
        w_mean = 1.0
    w = w / w_mean * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = torch.clamp(w, 0, 1)
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w + 1e-12
    w = w.log()
    w_max = w.max()
    w_mean = w.mean()
    if w_max == w_mean:
        return torch.zeros_like(w)
    return (w_max - w) / (w_max - w_mean)


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    edge_weights = torch.nan_to_num(edge_weights, nan=0.0, posinf=0.0, neginf=0.0)
    w_mean = edge_weights.mean()
    if w_mean == 0:
        w_mean = 1.0
    edge_weights = edge_weights / w_mean * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    edge_weights = torch.clamp(edge_weights, 0, 1)
    prob = 1.0 - edge_weights
    prob = torch.clamp(prob, 0, 1)
    sel_mask = torch.bernoulli(prob).to(torch.bool)
    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index, num_nodes=None):
    if edge_index.numel() == 0 or edge_index.size(1) == 0:
        return torch.zeros((0,), device=edge_index.device, dtype=torch.float32)
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1], num_nodes)

    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col + 1e-12)

    s_max = s_col.max()
    s_mean = s_col.mean()
    if s_max == s_mean:
        return torch.zeros_like(s_col)
    return (s_max - s_col) / (s_max - s_mean)


def build_gca_view(x: torch.Tensor, adj_label: torch.Tensor, drop_edge_p: float, drop_feat_p: float):
    N = int(x.size(0))
    A = (adj_label > 0).to(torch.float32)
    edge_index, _ = dense_to_sparse(A)
    drop_weights = degree_drop_weights(edge_index, N)
    edge_index_aug = drop_edge_weighted(edge_index, drop_weights, p=float(drop_edge_p))
    deg = degree(edge_index[1], N)
    feat_w = feature_drop_weights(x, deg)
    x_aug = drop_feature_weighted(x, feat_w, p=float(drop_feat_p))
    adj_aug = to_dense_adj(edge_index_aug, max_num_nodes=N)[0]
    adj_aug = torch.clamp(adj_aug, 0, 1)
    adj_aug.fill_diagonal_(0.0)
    adj_mp = adj_aug.clone()
    adj_mp.fill_diagonal_(1.0)
    return x_aug, adj_aug, adj_mp


def build_knn_adj(x: torch.Tensor, k: int, metric: str = "cosine"):
    metric = str(metric).strip().lower()
    N = int(x.size(0))
    k = max(1, min(int(k), N - 1))
    if metric != "cosine":
        raise ValueError("Only cosine metric is supported currently")
    z = torch.nn.functional.normalize(x, p=2, dim=1)
    sim = torch.mm(z, z.t())
    _, idx = torch.topk(sim, k=k + 1, dim=1)
    mask = torch.zeros((N, N), device=x.device, dtype=torch.bool)
    row = torch.arange(N, device=x.device).unsqueeze(1).expand_as(idx)
    mask[row, idx] = True
    mask.fill_diagonal_(False)
    mask = mask | mask.t()
    adj = mask.to(torch.float32)
    return adj


def prune_low_degree_edges(adj: torch.Tensor, ratio: float, score: str = "min"):
    ratio = float(ratio)
    if ratio <= 0:
        return adj
    A = (adj > 0).to(torch.float32)
    A = torch.triu(A, diagonal=1)
    idx = A.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return adj
    u = idx[:, 0]
    v = idx[:, 1]
    deg = (adj > 0).to(torch.float32).sum(dim=1)
    du = deg[u]
    dv = deg[v]
    score = str(score).strip().lower()
    if score == "sum":
        s = du + dv
    elif score == "avg":
        s = (du + dv) * 0.5
    else:
        s = torch.minimum(du, dv)
    m = int(idx.size(0))
    rm = int(round(m * ratio))
    rm = max(0, min(rm, m))
    if rm <= 0:
        return adj
    _, order = torch.sort(s, descending=False)
    remove = idx[order[:rm]]
    A = (adj > 0).to(torch.bool)
    A[remove[:, 0], remove[:, 1]] = False
    A[remove[:, 1], remove[:, 0]] = False
    A.fill_diagonal_(False)
    return A.to(torch.float32)


def prune_high_ebc_edges(adj: torch.Tensor, ratio: float, approx_k: int = 256, seed: int = 0):
    ratio = float(ratio)
    if ratio <= 0:
        return adj
    A = (adj > 0).to(torch.bool).detach().cpu()
    N = int(A.size(0))
    edges = torch.triu(A, diagonal=1).nonzero(as_tuple=False).numpy()
    if edges.shape[0] == 0:
        return adj
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([(int(u), int(v)) for u, v in edges])
    k = None
    if approx_k is not None:
        k = int(approx_k)
        if k <= 0 or k >= N:
            k = None
    if k is not None:
        bc = nx.edge_betweenness_centrality(G, k=k, seed=int(seed))
    else:
        bc = nx.edge_betweenness_centrality(G)
    m = len(bc)
    rm = int(round(m * ratio))
    rm = max(0, min(rm, m))
    if rm <= 0:
        return adj
    sorted_edges = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)
    to_remove = [e for e, _ in sorted_edges[:rm]]
    G.remove_edges_from(to_remove)
    A2 = nx.to_numpy_array(G, dtype=np.float32)
    np.fill_diagonal(A2, 0.0)
    return torch.from_numpy(A2).to(adj.device)


def make_message_passing_adj(adj_label: torch.Tensor):
    A = (adj_label > 0).to(torch.float32)
    A.fill_diagonal_(1.0)
    return A

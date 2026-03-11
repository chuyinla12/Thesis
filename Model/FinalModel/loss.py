import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(qi, qj, mask, temperature, weight):
    if not isinstance(qi, torch.Tensor):
        qi = torch.tensor(qi, device=mask.device, dtype=torch.float32)
    if not isinstance(qj, torch.Tensor):
        qj = torch.tensor(qj, device=mask.device, dtype=torch.float32)

    if qi.dim() == 1:
        qi = qi.unsqueeze(0)
    if qj.dim() == 1:
        qj = qj.unsqueeze(0)
    if qi.dim() != 2 or qj.dim() != 2:
        raise ValueError(f"Inputs must be 2D [N,C], got qi:{qi.shape}, qj:{qj.shape}")
    if qi.shape[1] != qj.shape[1]:
        raise ValueError(f"Feature dim mismatch: qi:{qi.shape}, qj:{qj.shape}")

    qi = F.normalize(qi, p=2, dim=1)
    qj = F.normalize(qj, p=2, dim=1)

    sim_mm = torch.mm(qi, qi.t())
    sim_mn = torch.mm(qi, qj.t())

    sim_mm = torch.exp(torch.clamp(sim_mm / temperature, -50, 50))
    sim_mn = torch.exp(torch.clamp(sim_mn / temperature, -50, 50))

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    pos_mm = (mask * sim_mm * logits_mask).sum(dim=1)
    pos_mn = (mask * sim_mn).sum(dim=1)
    neg_mm = sim_mm.sum(dim=1)
    neg_mn = sim_mn.sum(dim=1)

    loss_val = (pos_mm + pos_mn) / (neg_mm + neg_mn + 1e-10)
    loss_val = torch.clamp(loss_val, min=1e-10)
    return -weight * torch.mean(torch.log(loss_val))


def contrastive_loss_v1(qi, qj, mask, temperature, weight):
    if not isinstance(qi, torch.Tensor):
        qi = torch.tensor(qi, device=mask.device, dtype=torch.float32)
    if not isinstance(qj, torch.Tensor):
        qj = torch.tensor(qj, device=mask.device, dtype=torch.float32)

    if qi.dim() == 1:
        qi = qi.unsqueeze(0)
    if qj.dim() == 1:
        qj = qj.unsqueeze(0)
    if qi.dim() != 2 or qj.dim() != 2:
        raise ValueError(f"Inputs must be 2D [N,C], got qi:{qi.shape}, qj:{qj.shape}")
    if qi.shape[1] != qj.shape[1]:
        raise ValueError(f"Feature dim mismatch: qi:{qi.shape}, qj:{qj.shape}")

    qi = qi / (torch.linalg.vector_norm(qi, dim=1, keepdim=True) + 1e-6)
    qj = qj / (torch.linalg.vector_norm(qj, dim=1, keepdim=True) + 1e-6)

    logits_mm = torch.mm(qi, qi.t()) / float(temperature)
    logits_mn = torch.mm(qi, qj.t()) / float(temperature)

    logits_mask = torch.ones_like(mask, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    pos_mask = (mask > 0).to(torch.bool)

    pos_mask_mm = pos_mask & logits_mask
    den_mask_mm = logits_mask
    pos_mask_mn = pos_mask
    den_mask_mn = torch.ones_like(pos_mask_mn, dtype=torch.bool)

    neg_inf = -1e9
    log_pos_mm = torch.logsumexp(logits_mm.masked_fill(~pos_mask_mm, neg_inf), dim=1)
    log_den_mm = torch.logsumexp(logits_mm.masked_fill(~den_mask_mm, neg_inf), dim=1)
    log_pos_mn = torch.logsumexp(logits_mn.masked_fill(~pos_mask_mn, neg_inf), dim=1)
    log_den_mn = torch.logsumexp(logits_mn.masked_fill(~den_mask_mn, neg_inf), dim=1)

    log_pos = torch.logaddexp(log_pos_mm, log_pos_mn)
    log_den = torch.logaddexp(log_den_mm, log_den_mn)
    pos_count = pos_mask_mm.to(torch.int64).sum(dim=1) + pos_mask_mn.to(torch.int64).sum(dim=1)
    valid = (pos_count > 0) & torch.isfinite(log_pos) & torch.isfinite(log_den)
    if not torch.any(valid):
        return qi.sum() * 0.0
    loss_val = -(log_pos[valid] - log_den[valid])
    return float(weight) * torch.mean(loss_val)


def sample_level_loss_v1(adjs, q, q_all, temperature, weight=1.0):
    if isinstance(q, list):
        q = torch.stack(q)
    elif isinstance(q, torch.Tensor) and q.dim() == 2:
        q = q.unsqueeze(0)

    device = q.device
    q_all = q_all.to(device)
    adjs = [adj.to(device) if torch.is_tensor(adj) else torch.tensor(adj, device=device) for adj in adjs]

    n_views = q.size(0)
    total = 0.0
    for i in range(n_views):
        current_q = q[i]
        if current_q.dim() != 2:
            current_q = current_q.view(current_q.size(0), -1)
        mask = adjs[i].to(torch.float32)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.clamp(mask, min=0.0)
        mask = mask.clone()
        mask.fill_diagonal_(0.0)
        total = total + contrastive_loss_v1(current_q, q_all, mask, temperature, weight)
    return total / max(n_views, 1)


def sample_level_loss(adjs, q, q_all, temperature, weight=1.0):
    if isinstance(q, list):
        q = torch.stack(q)
    elif isinstance(q, torch.Tensor) and q.dim() == 2:
        q = q.unsqueeze(0)

    device = q.device
    q_all = q_all.to(device)
    adjs = [adj.to(device) if torch.is_tensor(adj) else torch.tensor(adj, device=device) for adj in adjs]

    n_views = q.size(0)
    total = 0.0
    for i in range(n_views):
        current_q = q[i]
        if current_q.dim() != 2:
            current_q = current_q.view(current_q.size(0), -1)
        mask = adjs[i].to(torch.float32)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.clamp(mask, min=0.0)
        mask = mask.clone()
        mask.fill_diagonal_(0.0)
        total = total + contrastive_loss(current_q, q_all, mask, temperature, weight)
    return total / max(n_views, 1)


def cluster_level_loss(q, q_all, temperature, weight=1.0):
    if isinstance(q, list):
        q = torch.stack(q)
    elif q.dim() == 2:
        q = q.unsqueeze(0)

    n_views = q.size(0)
    device = q.device
    n_clusters = q.size(2)
    mask = torch.eye(n_clusters, device=device)

    loss = 0.0
    for i in range(n_views):
        qi_cluster = q[i].t()
        qall_cluster = q_all.t()
        loss = loss + contrastive_loss(qi_cluster, qall_cluster, mask, temperature, weight)
    return loss / max(n_views, 1)


class ClusterLoss(nn.Module):
    def __init__(self, temperature=0.5, ne_weight=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.ne_weight = float(ne_weight)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i = p_i / (p_i.sum() + 1e-10)
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-10)).sum()

        p_j = c_j.sum(0).view(-1)
        p_j = p_j / (p_j.sum() + 1e-10)
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-10)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        N = sim.size(0)

        mask = torch.ones((N, N), dtype=torch.bool, device=c.device)
        mask.fill_diagonal_(0)

        if c_i.size(0) == c_j.size(0):
            class_num = c_i.size(0)
            sim_i_j = torch.diag(sim, class_num)
            sim_j_i = torch.diag(sim, -class_num)
            positives = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        else:
            positives = torch.diag(sim).reshape(N, 1)

        negatives = sim[mask].view(N, -1)

        labels = torch.zeros(N, device=c.device).long()
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / N
        return 0*loss + self.ne_weight * ne_loss


def kl_loss(cluster_q_list, cluster_all):
    total = 0.0
    for qv in cluster_q_list:
        qv = qv + 1e-8
        qa = cluster_all + 1e-8
        total = total + F.kl_div(qv.log(), qa, reduction="batchmean")
    return total / max(len(cluster_q_list), 1)


def sim_loss_func(adj_preds, adj_labels, weight_tensor=None):
    if weight_tensor is None:
        return F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    return F.binary_cross_entropy_with_logits(adj_preds.view(-1), adj_labels.view(-1), weight=weight_tensor)


def swav_approx_loss(h1, h2, lam=1.0, h_fused=None):
    if h1 is None or h2 is None:
        raise ValueError("h1/h2 must not be None")
    if h1.shape != h2.shape:
        raise ValueError(f"h1/h2 shape mismatch: {tuple(h1.shape)} vs {tuple(h2.shape)}")

    h1n = F.normalize(h1, p=2, dim=1)
    h2n = F.normalize(h2, p=2, dim=1)
    align = F.mse_loss(h1n, h2n)

    if h_fused is None:
        hall = torch.cat([h1n, h2n], dim=0)
    else:
        hf = F.normalize(h_fused, p=2, dim=1)
        hall = torch.cat([h1n, h2n, hf], dim=0)

    c = torch.matmul(hall.t(), hall) / (hall.size(0) + 1e-12)
    eye = torch.eye(c.size(0), device=c.device, dtype=c.dtype)
    decor = torch.sum((c - eye) ** 2)
    return align + float(lam) * decor


def swav_decor_loss(vs, cs):
    if vs is None or cs is None:
        raise ValueError("vs/cs must not be None")
    if vs.dim() != 2 or cs.dim() != 2:
        raise ValueError(f"vs/cs must be 2D, got vs:{tuple(vs.shape)} cs:{tuple(cs.shape)}")
    if int(vs.size(0)) != int(cs.size(0)):
        raise ValueError(f"vs/cs N mismatch: vs:{tuple(vs.shape)} cs:{tuple(cs.shape)}")

    H = torch.cat([vs, cs], dim=1)
    Hn = F.normalize(H, p=2, dim=0)
    c = torch.matmul(Hn.t(), Hn) / (Hn.size(0) + 1e-12)
    eye = torch.eye(c.size(0), device=c.device, dtype=c.dtype)
    return torch.sum((c - eye) ** 2)


def _sinkhorn(scores, epsilon=0.05, iters=3):
    if scores.dim() != 2:
        raise ValueError(f"scores must be 2D [B,K], got {tuple(scores.shape)}")
    Q = torch.exp(torch.clamp(scores / float(epsilon), -50, 50)).t()
    B = int(Q.size(1))
    K = int(Q.size(0))
    Q = Q / (Q.sum() + 1e-12)
    for _ in range(int(iters)):
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        Q = Q / float(K)
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        Q = Q / float(B)
    Q = Q * float(B)
    return Q.t()


def swav_classic_loss(h1, h2, prototypes, temp=0.1, epsilon=0.05, sinkhorn_iters=3):
    if h1 is None or h2 is None:
        raise ValueError("h1/h2 must not be None")
    if h1.shape != h2.shape:
        raise ValueError(f"h1/h2 shape mismatch: {tuple(h1.shape)} vs {tuple(h2.shape)}")
    if prototypes is None:
        raise ValueError("prototypes must not be None")
    if prototypes.dim() != 2:
        raise ValueError(f"prototypes must be 2D [K,D], got {tuple(prototypes.shape)}")
    if int(prototypes.size(1)) != int(h1.size(1)):
        raise ValueError(f"prototypes dim mismatch: {tuple(prototypes.shape)} vs h:{tuple(h1.shape)}")

    z1 = F.normalize(h1, p=2, dim=1)
    z2 = F.normalize(h2, p=2, dim=1)
    p = F.normalize(prototypes, p=2, dim=1)

    s1 = torch.matmul(z1, p.t()) / float(temp)
    s2 = torch.matmul(z2, p.t()) / float(temp)

    with torch.no_grad():
        q1 = _sinkhorn(s1, epsilon=epsilon, iters=sinkhorn_iters)
        q2 = _sinkhorn(s2, epsilon=epsilon, iters=sinkhorn_iters)

    l1 = -(q1 * F.log_softmax(s2, dim=1)).sum(dim=1).mean()
    l2 = -(q2 * F.log_softmax(s1, dim=1)).sum(dim=1).mean()
    return 0.5 * (l1 + l2)


def swav_m1_loss(cluster_logits_v0, cluster_logits_v1, cluster_all_logits, temp=0.2, epsilon=0.05, sinkhorn_iters=3):
    def _sinkhorn_m1(scores, n_iters):
        scores = scores - scores.max(dim=1, keepdim=True).values
        scores = torch.clamp(scores, -50, 50)
        Q = torch.exp(scores).t()
        Q = Q / (Q.sum() + 1e-12)
        K, B = Q.shape
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B
        for _ in range(int(n_iters)):
            u = Q.sum(dim=1)
            Q = Q * (r / (u + 1e-12)).unsqueeze(1)
            v = Q.sum(dim=0)
            Q = Q * (c / (v + 1e-12)).unsqueeze(0)
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        return Q.t()

    with torch.no_grad():
        q0 = _sinkhorn_m1(cluster_logits_v0 / float(temp), int(sinkhorn_iters))
        q1 = _sinkhorn_m1(cluster_logits_v1 / float(temp), int(sinkhorn_iters))
    p_all = F.log_softmax(cluster_all_logits / float(temp), dim=1)
    l0 = torch.mean(torch.sum(-q0 * p_all, dim=1))
    l1 = torch.mean(torch.sum(-q1 * p_all, dim=1))
    return 0.5 * (l0 + l1)

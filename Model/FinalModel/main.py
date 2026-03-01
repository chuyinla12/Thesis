import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from utils import set_seed, get_device, ensure_dir, eva, cal_homo_ratio_fast
from data import load_dataset
from views import build_gca_view, build_knn_adj, prune_low_degree_edges, prune_high_ebc_edges, make_message_passing_adj
from models import FinalModel
from loss import sample_level_loss, sample_level_loss_v1, cluster_level_loss, ClusterLoss, kl_loss, sim_loss_func, swav_approx_loss, swav_classic_loss, swav_decor_loss, swav_m1_loss


def build_args():
    p = argparse.ArgumentParser()
    # ---------------- 数据集与路径配置 ----------------
    p.add_argument("--dataset", type=str, default="cora", help="数据集名称 (cora, citeseer, pubmed, acm, amazon_electronics_photo, amazon_electronics_computers)")
    p.add_argument("--data_root", type=str, default=r"../data", help="原始数据存放根目录")
    p.add_argument("--extracted_root", type=str, default=None, help="解压/预处理后的数据目录，默认在 data_extracted 下")
    p.add_argument("--save_dir", type=str, default=None, help="模型保存和日志输出目录")
    
    # ---------------- 训练环境与随机种子 ----------------
    p.add_argument("--seed", type=int, default=4, help="随机种子，用于复现结果")
    p.add_argument("--cuda", type=int, default=1, help="是否使用GPU (1:是, 0:否)")
    p.add_argument("--max_cuda_nodes", type=int, default=10000, help="GPU显存限制下的最大节点数，超过此数将强制使用CPU")
    p.add_argument("--pubmed_use_small", type=int, default=1, help="Pubmed是否使用抽样子图进行训练/评估 (1:是, 0:否)")
    p.add_argument("--pubmed_small_n", type=int, default=8000, help="Pubmed抽样节点数 (按真实标签比例分层抽样)")
    p.add_argument("--pubmed_small_rebuild", type=int, default=1, help="是否强制重新生成 pubmed-small (1:是, 0:否)")

    # ---------------- 训练超参数 ----------------
    p.add_argument("--epochs", type=int, default=400, help="最大训练轮数")
    p.add_argument("--lr", type=float, default=0.0005, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.001, help="优化器的权重衰减 (L2正则)")
    p.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值 (<=0表示关闭)")
    p.add_argument("--eval_interval", type=int, default=10, help="每隔多少轮进行一次评估 (NMI/ACC)")
    p.add_argument("--print_interval", type=int, default=10, help="每隔多少轮打印一次 loss")
    p.add_argument("--eval_mode", type=str, default="Kmeans", help="评估模式: 'Kmeans' (对embedding做聚类) 或 'Argmax' (直接取cluster head最大值)")
    p.add_argument("--eval_embed", type=str, default="h_all", help="用于评估的embedding: 'h_all'(融合), 'h0'(视图0), 'h1'(视图1), 'concat'(拼接)")
    p.add_argument("--kmeans_n_init", type=int, default=20, help="KMeans评估时的n_init参数")
    p.add_argument("--init_kmeans", type=int, default=1, help="是否在训练前用KMeans初始化聚类中心 (1:是, 0:否)")
    
    # ---------------- 伪标签与自监督策略 ----------------
    p.add_argument("--pseudo_interval", type=int, default=10, help="每隔多少轮更新一次伪标签")
    p.add_argument("--classifier_hidden", type=int, nargs="*", default=[128, 64], help="聚类头(MLP)的隐藏层维度列表")
    p.add_argument("--supervise_match", type=int, default=0, help="匹配损失是否使用真实标签 (1:全监督/半监督, 0:自监督伪标签)")
    p.add_argument("--pseudo_source", type=str, default="h_all", help="生成伪标签的来源: 'h_all', 'h0', 'h1', 'concat'")
    p.add_argument("--update_weights", type=int, default=0, help="是否根据视图同质性动态更新融合权重 (1:是, 0:否)")
    p.add_argument("--update_weights_interval", type=int, default=10, help="每隔多少轮更新一次融合权重/同质性 (减少CPU开销)")
    p.add_argument("--weights_momentum", type=float, default=0.9, help="融合权重更新动量 (仅 update_weights=1 生效)")
    p.add_argument("--weights_min", type=float, default=0.05, help="融合权重最小值 (仅 update_weights=1 生效)")
    p.add_argument("--pseudo_warm_start", type=int, default=0, help="伪标签生成是否使用warm start (KMeans)")
    p.add_argument("--early_stop_patience", type=int, default=0, help="早停耐心值 (0表示不启用早停)")
    
    # ---------------- 调试参数 ----------------
    p.add_argument("--debug", type=int, default=0, help="是否开启调试模式，打印梯度和统计信息")
    p.add_argument("--debug_interval", type=int, default=1, help="调试信息打印间隔")
    p.add_argument("--debug_hist_k", type=int, default=7, help="调试时打印Top-K分布的K值")

    # ---------------- 视图1: GCA增强参数 ----------------
    p.add_argument("--gca_drop_edge_p", type=float, default=0.1, help="GCA视图: 删边概率")
    p.add_argument("--gca_drop_feat_p", type=float, default=0.6, help="GCA视图: 特征掩码概率")

    # ---------------- 视图2: KNN与剪枝参数 ----------------
    p.add_argument("--knn_k", type=int, default=20, help="KNN构图的邻居数 K")
    p.add_argument("--knn_metric", type=str, default="cosine", help="KNN距离度量 (目前仅支持 cosine)")
    p.add_argument("--p_low_deg", type=float, default=0.2, help="低度边剪枝比例 (去除噪声边)")
    p.add_argument("--low_deg_score", type=str, default="avg", help="低度边评分方式: 'min', 'avg', 'sum'")
    p.add_argument("--p_high_ebc", type=float, default=0.3, help="高介数边剪枝比例 (强化社区结构)")
    p.add_argument("--ebc_approx_k", type=int, default=256, help="计算介数中心性时的近似节点采样数")

    # ---------------- 模型架构参数 ----------------
    p.add_argument("--hidden_dim", type=int, default=256, help="GCN隐藏层维度")
    p.add_argument("--output_dim_g", type=int, default=64, help="GCN输出/Embedding维度")
    p.add_argument("--gcn_dropout", type=float, default=0.1, help="GCN层的Dropout概率")
    p.add_argument("--gcn_layers", type=int, default=2, help="GCN层数")
    p.add_argument("--gcn_impl", type=str, default="pyg", help="GCN实现: dense / pyg(同Model1)")

    # ---------------- 损失函数权重系数 ----------------
    p.add_argument("--alpha", type=float, default=0.0, help="Loss权重: 匹配损失 (Match Loss)")
    p.add_argument("--beda", type=float, default=0.0, help="Loss权重: 相似度重构损失 (Sim Loss)")
    p.add_argument("--gama", type=float, default=0.0, help="Loss权重: 聚类一致性损失 (KL Soft)")
    p.add_argument("--w_sample", type=float, default=1.0, help="Loss权重: 样本级对比 (Sample-level)")
    p.add_argument("--sample_loss_impl", type=str, default="v2", help="Sample-level 实现版本: v2(当前) / v1(同Model1)")
    p.add_argument("--sample_loss_space", type=str, default="embed", help="Sample-level 输入空间: embed(同Model1) / prob(原Model2)")
    p.add_argument("--w_cluster", type=float, default=1.0, help="Loss权重: 簇级对比 (Cluster-level)")
    p.add_argument("--w_clusterloss", type=float, default=2.5, help="Loss权重: ClusterLoss (InfoNCE + 负熵)")
    p.add_argument("--w_ne", type=float, default=1.0, help="ClusterLoss内负熵正则系数 (仅作用于ne项)")
    p.add_argument("--w_swav", type=float, default=0, help="Loss权重: SwAV简化近似项")
    p.add_argument("--w_kl_g", type=float, default=0.0, help="Loss权重: Student-t分布KL损失 (DEC loss)")
    p.add_argument("--train_centroids", type=int, default=0, help="是否训练聚类中心 (1:是, 0:否)")
    p.add_argument("--w_re_x", type=float, default=0.0, help="Loss权重: 特征重构损失 (BCE)")
    p.add_argument("--w_re_x_mse", type=float, default=0.0, help="Loss权重: 特征重构损失 (MSE)")
    p.add_argument("--w_re_a", type=float, default=0.0, help="Loss权重: 结构重构损失 (BCEWithLogits)")

    # ---------------- SwAV参数 ----------------
    p.add_argument("--swav_variant", type=str, default="classic", help="SwAV版本: approx / classic / decor")
    p.add_argument("--swav_lambda", type=float, default=1, help="SwAV简化近似: decorrelation项系数lambda")
    p.add_argument("--swav_use_fused", type=int, default=0, help="SwAV简化近似: 是否把h_all也并入H_all (1:是,0:否)")
    p.add_argument("--swav_temp", type=float, default=0.1, help="SwAV classic: prototype logits温度系数")
    p.add_argument("--swav_epsilon", type=float, default=0.05, help="SwAV classic: Sinkhorn epsilon")
    p.add_argument("--swav_sinkhorn_iters", type=int, default=3, help="SwAV classic: Sinkhorn迭代次数")
    
    # ---------------- 温度系数 ----------------
    p.add_argument("--tau_sample", type=float, default=0.5, help="Sample-level 对比损失的温度系数")
    p.add_argument("--tau_cluster", type=float, default=0.2, help="Cluster-level 对比损失的温度系数")
    p.add_argument("--tau_clusterloss", type=float, default=0.5, help="ClusterLoss (InfoNCE) 的温度系数")
    p.add_argument("--sim_temp", type=float, default=0.2, help="相似度预测的温度系数 (Scaling)")

    # ---------------- 其他损失权重与参数 ----------------
    p.add_argument("--use_a_pos_weight", type=int, default=0, help="结构重构BCE是否使用正样本加权 (解决稀疏性)")
    p.add_argument("--kl_max", type=float, default=1.0, help="Student-t KL损失的最大权重 (用于Annealing)")
    p.add_argument("--kl_anneal_epochs", type=int, default=100, help="Student-t KL损失权重的退火周期 (线性增长)")
    p.add_argument("--loss_warmup_epochs", type=int, default=0, help="对match/sim/kl-soft/swav做线性warmup的epoch数 (0表示不启用)")
    return p.parse_args()


def calc_loss_kl_g(model, qgs, epoch, kl_max, kl_anneal_epochs):
    l = min(float(kl_max), float(epoch + 1) / float(max(1, int(kl_anneal_epochs))) * float(kl_max))
    pgh = model.target_distribution(qgs[-1].detach())
    loss = F.kl_div((qgs[-1] + 1e-10).log(), pgh, reduction="batchmean")
    for v in range(len(qgs) - 1):
        pg = model.target_distribution(qgs[v].detach())
        loss = loss + F.kl_div((qgs[v] + 1e-10).log(), pg, reduction="batchmean")
        loss = loss + F.kl_div((qgs[v] + 1e-10).log(), pgh, reduction="batchmean")
    return l * loss


def main():
    args = build_args()
    set_seed(args.seed)

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(x.detach().cpu().item())
            except Exception:
                return float("nan")

    def _tensor_stats(name, t):
        if t is None:
            return None
        if not torch.is_tensor(t):
            return None
        with torch.no_grad():
            x = t.detach()
            x0 = x.float()
            is_finite = torch.isfinite(x0)
            num = int(x0.numel())
            fin = int(is_finite.sum().cpu().item()) if num else 0
            nan = int(torch.isnan(x0).sum().cpu().item()) if num else 0
            inf = int(torch.isinf(x0).sum().cpu().item()) if num else 0
            if fin > 0:
                xf = x0[is_finite]
                mn = _safe_float(xf.min())
                mx = _safe_float(xf.max())
                mean = _safe_float(xf.mean())
                std = _safe_float(xf.std(unbiased=False))
            else:
                mn = mx = mean = std = float("nan")
            norm = _safe_float(torch.linalg.vector_norm(x0)) if num else 0.0
            return {"name": name, "shape": tuple(x.shape), "mean": mean, "std": std, "min": mn, "max": mx, "norm": norm, "num": num, "finite": fin, "nan": nan, "inf": inf}

    def _cluster_stats(name, q):
        if q is None or (not torch.is_tensor(q)):
            return None
        with torch.no_grad():
            p = q.detach().to(torch.float32)
            p = torch.clamp(p, 1e-12, 1.0)
            ent = -torch.sum(p * torch.log(p), dim=1)
            ent_mean = _safe_float(ent.mean())
            ent_std = _safe_float(ent.std(unbiased=False))
            maxp, arg = torch.max(p, dim=1)
            maxp_mean = _safe_float(maxp.mean())
            maxp_std = _safe_float(maxp.std(unbiased=False))
            k = int(p.size(1))
            counts = torch.bincount(arg, minlength=k).to(torch.float32)
            frac = counts / (counts.sum() + 1e-12)
            frac_top = torch.topk(frac, k=min(int(args.debug_hist_k), k)).values
            frac_top = [float(v) for v in frac_top.detach().cpu().tolist()]
            uniq = int((counts > 0).sum().cpu().item())
            p_mean = p.mean(dim=0)
            p_mean = p_mean / (p_mean.sum() + 1e-12)
            p_mean_top = torch.topk(p_mean, k=min(int(args.debug_hist_k), k)).values
            p_mean_top = [float(v) for v in p_mean_top.detach().cpu().tolist()]
            p_mean_ent = _safe_float((-p_mean * torch.log(p_mean + 1e-12)).sum())
            return {"name": name, "k": k, "ent_mean": ent_mean, "ent_std": ent_std, "maxp_mean": maxp_mean, "maxp_std": maxp_std, "nonempty": uniq, "top_frac": frac_top, "p_mean_ent": p_mean_ent, "p_mean_top": p_mean_top}

    def _adj_density(name, A):
        if A is None or (not torch.is_tensor(A)):
            return None
        with torch.no_grad():
            x = (A.detach() > 0).to(torch.float32)
            if x.dim() != 2:
                return None
            n = int(x.size(0))
            if n == 0:
                return None
            diag = torch.diag(x)
            off = x.sum() - diag.sum()
            den = off / (n * (n - 1) + 1e-12)
            return {"name": name, "n": n, "density": _safe_float(den), "diag_mean": _safe_float(diag.mean())}

    def _grad_norm(model):
        total = 0.0
        max_g = 0.0
        max_name = ""
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                return {"total": float("nan"), "max": float("nan"), "max_name": n, "nonfinite": True}
            gn = torch.linalg.vector_norm(g).item()
            total += gn * gn
            if gn > max_g:
                max_g = gn
                max_name = n
        return {"total": float(total ** 0.5), "max": float(max_g), "max_name": max_name, "nonfinite": False}

    def _centroid_stats(name, c):
        if c is None or (not torch.is_tensor(c)):
            return None
        with torch.no_grad():
            x = c.detach().to(torch.float32)
            x = F.normalize(x, p=2, dim=1)
            sim = torch.mm(x, x.t())
            k = int(sim.size(0))
            mask = torch.ones((k, k), device=sim.device, dtype=torch.bool)
            mask.fill_diagonal_(False)
            sims = sim[mask]
            return {
                "name": name,
                "k": k,
                "sim_mean": _safe_float(sims.mean()) if sims.numel() else float("nan"),
                "sim_std": _safe_float(sims.std(unbiased=False)) if sims.numel() else float("nan"),
                "sim_min": _safe_float(sims.min()) if sims.numel() else float("nan"),
                "sim_max": _safe_float(sims.max()) if sims.numel() else float("nan"),
            }

    root = os.path.dirname(os.path.abspath(__file__))
    extracted_root = args.extracted_root if args.extracted_root is not None else os.path.join(root, "data_extracted")
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(root, "runs", args.dataset)
    ensure_dir(extracted_root)
    ensure_dir(save_dir)

    labels, adj, features, adj_label, feature_label = load_dataset(
        args.dataset,
        args.data_root,
        extracted_root,
        seed=int(args.seed),
        pubmed_small_n=int(args.pubmed_small_n),
        pubmed_small_rebuild=bool(args.pubmed_small_rebuild),
        pubmed_use_small=bool(args.pubmed_use_small),
    )
    N0 = int(features.size(0))
    device = get_device(bool(args.cuda))
    if device.type == "cuda" and N0 > int(args.max_cuda_nodes):
        device = torch.device("cpu")
    if N0 > int(args.max_cuda_nodes) and args.dataset.lower() == "pubmed":
        raise RuntimeError("pubmed 节点数过大，当前 dense 实现不支持；请后续切换为 sparse 实现")
    labels = labels.to(device)
    features = features.to(device)
    adj_label = adj_label.to(device).to(torch.float32)

    class_num = int(labels.max().item()) + 1
    N = int(features.size(0))
    input_dim_x = int(features.size(1))
    weights_h = torch.ones(2, device=device) / 2.0

    adj_knn = build_knn_adj(features, k=args.knn_k, metric=args.knn_metric)
    adj_knn = prune_low_degree_edges(adj_knn, ratio=args.p_low_deg, score=args.low_deg_score)
    adj_knn = prune_high_ebc_edges(adj_knn, ratio=args.p_high_ebc, approx_k=args.ebc_approx_k, seed=args.seed)
    adj_knn = torch.clamp(adj_knn, 0, 1).to(device)
    adj_knn.fill_diagonal_(0.0)
    adj_knn_mp = make_message_passing_adj(adj_knn)

    model = FinalModel(
        input_dim_x=input_dim_x,
        node_num=N,
        hidden_dim=args.hidden_dim,
        output_dim_g=args.output_dim_g,
        class_num=class_num,
        view_num=2,
        gcn_dropout=args.gcn_dropout,
        gcn_layers=args.gcn_layers,
        gcn_impl=args.gcn_impl,
        classifier_hidden=args.classifier_hidden,
    ).to(device)

    if int(args.train_centroids) == 0:
        for p in model.cluster_layers:
            p.requires_grad_(False)
    excluded = set()
    if int(args.train_centroids) == 0:
        excluded = {id(p) for p in model.cluster_layers}
    params = [p for p in model.parameters() if id(p) not in excluded]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion_cluster = ClusterLoss(temperature=args.tau_clusterloss, ne_weight=args.w_ne).to(device)

    homo_rate = [0.5, 0.5]
    best_nmi = -1.0
    best_acc = -1.0
    best_state = None
    no_improve = 0
    pseudo_labels_views = [None, None]
    pseudo_centers_views = [None, None]
    eval_centers = None
    pos_weight = None
    if int(args.use_a_pos_weight) == 1:
        with torch.no_grad():
            num_pos = float(adj_label.sum().detach().cpu().item())
            num_all = float(adj_label.numel())
            num_neg = max(0.0, num_all - num_pos)
            pos_weight = torch.tensor([num_neg / (num_pos + 1e-12)], device=device, dtype=torch.float32)

    if int(args.init_kmeans) == 1:
        model.eval()
        with torch.no_grad():
            x_gca0, adj_gca0, adj_gca_mp0 = build_gca_view(features, adj_label, args.gca_drop_edge_p, args.gca_drop_feat_p)
            x_gca0 = x_gca0.to(device)
            adj_gca0 = adj_gca0.to(device)
            adj_gca0.fill_diagonal_(0.0)
            adj_gca_mp0 = adj_gca_mp0.to(device)
            xs0 = [x_gca0, features]
            adjs_labels0 = [adj_gca0, adj_knn]
            adjs_mp0 = [adj_gca_mp0, adj_knn_mp]
            out0 = model(xs0, adjs_mp0, adjs_labels0, weights_h, homo_rate)
            hs0 = out0[1]
            h_all0 = out0[2]
            embeds = [hs0[0], hs0[1], h_all0]
            for i, emb in enumerate(embeds):
                km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                y = km.fit_predict(emb.detach().cpu().numpy())
                model.cluster_layers[i].data.copy_(torch.from_numpy(km.cluster_centers_).to(device).to(model.cluster_layers[i].dtype))

    for epoch in range(args.epochs):
        model.train()

        x_gca, adj_gca, adj_gca_mp = build_gca_view(features, adj_label, args.gca_drop_edge_p, args.gca_drop_feat_p)
        x_gca = x_gca.to(device)
        adj_gca = adj_gca.to(device)
        adj_gca.fill_diagonal_(0.0)
        adj_gca_mp = adj_gca_mp.to(device)

        xs = [x_gca, features]
        adjs_labels = [adj_gca, adj_knn]
        adjs_mp = [adj_gca_mp, adj_knn_mp]

        (
            qgs,
            hs,
            h_all,
            Ss,
            zx_norms,
            cluster_logits,
            cluster_all_logits,
            x_preds,
            a_logits_list,
            zxs,
            zas,
            cluster_q,
            cluster_all,
            vs,
            cs,
        ) = model(
            xs,
            adjs_mp,
            adjs_labels,
            weights_h,
            homo_rate,
            compute_x_pred=(float(args.w_re_x) != 0.0 or float(args.w_re_x_mse) != 0.0),
            compute_a_logits=(float(args.w_re_a) != 0.0),
        )

        if int(args.supervise_match) == 0:
            if epoch == 0 or (epoch + 1) % int(args.pseudo_interval) == 0:
                src = str(args.pseudo_source).strip().lower()
                if src == "h_all":
                    emb = h_all
                    X = emb.detach().cpu().numpy()
                    if not np.isfinite(X).all():
                        y = np.argmax(cluster_all.detach().cpu().numpy(), axis=1)
                        eval_centers = None
                    else:
                        use_warm = int(args.pseudo_warm_start) == 1 and eval_centers is not None and hasattr(eval_centers, "shape") and eval_centers.shape == (class_num, X.shape[1])
                        if use_warm:
                            km = KMeans(n_clusters=class_num, init=eval_centers, n_init=1, random_state=args.seed)
                        else:
                            km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                        y = km.fit_predict(X)
                        eval_centers = km.cluster_centers_
                    pseudo = torch.tensor(y, device=device, dtype=torch.long)
                    pseudo_labels_views[0] = pseudo
                    pseudo_labels_views[1] = pseudo
                elif src == "concat":
                    emb = torch.cat([hs[0], hs[1]], dim=1)
                    X = emb.detach().cpu().numpy()
                    if not np.isfinite(X).all():
                        y = np.argmax(cluster_all.detach().cpu().numpy(), axis=1)
                        eval_centers = None
                    else:
                        use_warm = int(args.pseudo_warm_start) == 1 and eval_centers is not None and hasattr(eval_centers, "shape") and eval_centers.shape == (class_num, X.shape[1])
                        if use_warm:
                            km = KMeans(n_clusters=class_num, init=eval_centers, n_init=1, random_state=args.seed)
                        else:
                            km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                        y = km.fit_predict(X)
                        eval_centers = km.cluster_centers_
                    pseudo = torch.tensor(y, device=device, dtype=torch.long)
                    pseudo_labels_views[0] = pseudo
                    pseudo_labels_views[1] = pseudo
                else:
                    for v in range(2):
                        emb = hs[v]
                        X = emb.detach().cpu().numpy()
                        if not np.isfinite(X).all():
                            y = np.argmax(cluster_q[v].detach().cpu().numpy(), axis=1)
                            pseudo_centers_views[v] = None
                            pseudo_labels_views[v] = torch.tensor(y, device=device, dtype=torch.long)
                            continue
                        centers = pseudo_centers_views[v]
                        use_warm = int(args.pseudo_warm_start) == 1 and centers is not None and hasattr(centers, "shape") and centers.shape == (class_num, X.shape[1])
                        if use_warm:
                            km = KMeans(n_clusters=class_num, init=centers, n_init=1, random_state=args.seed)
                        else:
                            km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                        y = km.fit_predict(X)
                        pseudo_centers_views[v] = km.cluster_centers_
                        pseudo_labels_views[v] = torch.tensor(y, device=device, dtype=torch.long)

        contra_loss = 0.0
        loss_sample = 0.0
        loss_cluster = 0.0
        loss_clusterloss = 0.0
        sample_impl = str(args.sample_loss_impl).strip().lower()
        sample_space = str(args.sample_loss_space).strip().lower()
        for v in range(2):
            if sample_space == "prob":
                q_in = cluster_q[v]
                q_all_in = cluster_all
            else:
                q_in = hs[v]
                q_all_in = h_all
            if sample_impl == "v1":
                ls = sample_level_loss_v1([Ss[v]], q_in, q_all_in, temperature=args.tau_sample, weight=args.w_sample)
            else:
                ls = sample_level_loss([Ss[v]], q_in, q_all_in, temperature=args.tau_sample, weight=args.w_sample)
            lc = cluster_level_loss(cluster_q[v], cluster_all, temperature=args.tau_cluster, weight=args.w_cluster)
            lcl = criterion_cluster(cluster_q[v], cluster_all)
            loss_sample = loss_sample + ls
            loss_cluster = loss_cluster + lc
            loss_clusterloss = loss_clusterloss + lcl
            contra_loss = contra_loss + ls + lc + float(args.w_clusterloss) * lcl

        swav_loss = 0.0
        if float(args.w_swav) != 0.0:
            sv = str(args.swav_variant).strip().lower()
            if sv == "m1":
                swav_loss = swav_m1_loss(
                    cluster_logits[0],
                    cluster_logits[1],
                    cluster_all_logits,
                    temp=args.swav_temp,
                    sinkhorn_iters=args.swav_sinkhorn_iters,
                )
            elif sv == "classic":
                proto = model.cluster_layers[-1]
                swav_loss = swav_classic_loss(
                    hs[0],
                    hs[1],
                    proto,
                    temp=args.swav_temp,
                    epsilon=args.swav_epsilon,
                    sinkhorn_iters=args.swav_sinkhorn_iters,
                )
            elif sv == "decor":
                swav_loss = 0.0
                for v in range(2):
                    swav_loss = swav_loss + swav_decor_loss(vs[v], cs[v])
                swav_loss = swav_loss / 2.0
            else:
                hf = h_all if int(args.swav_use_fused) == 1 else None
                swav_loss = swav_approx_loss(hs[0], hs[1], lam=args.swav_lambda, h_fused=hf)

        loss_match = 0.0
        for v in range(2):
            if int(args.supervise_match) == 1:
                loss_match = loss_match + F.nll_loss(F.log_softmax(cluster_logits[v], dim=1), labels)
            else:
                loss_match = loss_match + F.nll_loss(F.log_softmax(cluster_logits[v], dim=1), pseudo_labels_views[v])
        loss_match = loss_match / 2.0

        sim_loss = 0.0
        if float(args.beda) != 0.0:
            adj_sim_pred = torch.mm(F.normalize(h_all, p=2, dim=1), F.normalize(h_all, p=2, dim=1).t())
            sim_logits = adj_sim_pred / float(args.sim_temp)
            for v in range(2):
                sim_loss = sim_loss + sim_loss_func(sim_logits.reshape(-1), adjs_labels[v].reshape(-1))
            sim_loss = sim_loss / 2.0

        kl_div = kl_loss(cluster_q, cluster_all)
        warm = 1.0
        if int(args.loss_warmup_epochs) > 0:
            warm = min(1.0, float(epoch + 1) / float(max(1, int(args.loss_warmup_epochs))))
        alpha_w = float(args.alpha) * warm
        beda_w = float(args.beda) * warm
        gama_w = float(args.gama) * warm
        swav_w = float(args.w_swav) * warm
        total_loss = contra_loss + swav_w * swav_loss + alpha_w * loss_match + beda_w * sim_loss + gama_w * kl_div

        loss_kl_g = calc_loss_kl_g(model, qgs, epoch, args.kl_max, args.kl_anneal_epochs)

        loss_re_x = 0.0
        loss_re_x_mse = 0.0
        loss_re_a = 0.0
        need_re_x = float(args.w_re_x) != 0.0
        need_re_x_mse = float(args.w_re_x_mse) != 0.0
        need_re_a = float(args.w_re_a) != 0.0
        if need_re_x or need_re_x_mse:
            for v in range(2):
                if need_re_x:
                    loss_re_x = loss_re_x + F.binary_cross_entropy(x_preds[v], features)
                if need_re_x_mse:
                    loss_re_x_mse = loss_re_x_mse + F.mse_loss(x_preds[v], features)
            loss_re_x = loss_re_x / 2.0
            loss_re_x_mse = loss_re_x_mse / 2.0
        if need_re_a:
            bce_a = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else torch.nn.BCEWithLogitsLoss()
            for v in range(2):
                loss_re_a = loss_re_a + bce_a(a_logits_list[v].reshape(-1), adj_label.reshape(-1))
            loss_re_a = loss_re_a / 2.0

        loss = args.w_re_x * loss_re_x + args.w_re_x_mse * loss_re_x_mse + args.w_re_a * loss_re_a + args.w_kl_g * loss_kl_g + total_loss

        if (epoch + 1) % int(args.print_interval) == 0 or epoch == 0:
            print(
                "epoch",
                epoch,
                "loss",
                float(loss.detach().cpu().item()),
                "total_loss",
                float(total_loss.detach().cpu().item()),
                "contra_loss",
                float(contra_loss.detach().cpu().item()) if torch.is_tensor(contra_loss) else float(contra_loss),
                "swav",
                float(swav_loss.detach().cpu().item()) if torch.is_tensor(swav_loss) else float(swav_loss),
                "ls_s",
                float(loss_sample.detach().cpu().item()) if torch.is_tensor(loss_sample) else float(loss_sample),
                "ls_c",
                float(loss_cluster.detach().cpu().item()) if torch.is_tensor(loss_cluster) else float(loss_cluster),
                "ls_cl",
                float(loss_clusterloss.detach().cpu().item()) if torch.is_tensor(loss_clusterloss) else float(loss_clusterloss),
                "kl_soft",
                float(kl_div.detach().cpu().item()) if torch.is_tensor(kl_div) else float(kl_div),
                "match",
                float(loss_match.detach().cpu().item()) if torch.is_tensor(loss_match) else float(loss_match),
                "sim",
                float(sim_loss.detach().cpu().item()) if torch.is_tensor(sim_loss) else float(sim_loss),
                "loss_re_x",
                _safe_float(loss_re_x),
                "loss_re_x_mse",
                _safe_float(loss_re_x_mse),
                "loss_re_a",
                _safe_float(loss_re_a),
                "loss_kl_g",
                _safe_float(loss_kl_g),
            )

        optimizer.zero_grad()
        loss.backward()
        if int(args.debug) == 1 and ((epoch % int(args.debug_interval)) == 0):
            gn = _grad_norm(model)
            print("debug", epoch, "grad_total", gn["total"], "grad_max", gn["max"], "grad_max_name", gn["max_name"], "grad_nonfinite", gn["nonfinite"])
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            do_eval = (epoch + 1) % int(args.eval_interval) == 0 or epoch == 0
            eval_mode = str(args.eval_mode).strip().lower()
            if eval_mode == "kmeans" and do_eval:
                ee = str(args.eval_embed).strip().lower()
                if ee == "h0":
                    emb = hs[0]
                elif ee == "h1":
                    emb = hs[1]
                elif ee == "concat":
                    emb = torch.cat([hs[0], hs[1]], dim=1)
                else:
                    emb = h_all
                X = emb.detach().cpu().numpy()
                if not np.isfinite(X).all():
                    y_pred = np.argmax(cluster_all.detach().cpu().numpy(), axis=1)
                    eval_centers = None
                    y_pred_t = torch.from_numpy(y_pred).to(device)
                else:
                    use_warm = int(args.pseudo_warm_start) == 1 and eval_centers is not None and hasattr(eval_centers, "shape") and eval_centers.shape == (class_num, X.shape[1])
                    if use_warm:
                        km = KMeans(n_clusters=class_num, init=eval_centers, n_init=1, random_state=args.seed)
                    else:
                        km = KMeans(n_clusters=class_num, n_init=int(args.kmeans_n_init), random_state=args.seed)
                    y_pred = km.fit_predict(X)
                    eval_centers = km.cluster_centers_
                    y_pred_t = torch.from_numpy(y_pred).to(device)
            else:
                y_pred_t = torch.argmax(cluster_all, dim=1)
                y_pred = y_pred_t.detach().cpu().numpy()

            if do_eval:
                prev_best = float(best_acc)
                nmi, acc, ari, f1 = eva(labels.detach().cpu().numpy(), y_pred, epoch=epoch, visible=True)
                if float(acc) > best_acc:
                    best_acc = float(acc)
                    best_nmi = float(nmi)
                    snapshot = {
                        "eval_mode": str(args.eval_mode),
                        "eval_embed": str(args.eval_embed),
                        "kmeans_n_init": int(args.kmeans_n_init),
                        "seed": int(args.seed),
                        "y_true": labels.detach().cpu().to(torch.long),
                        "y_pred": torch.from_numpy(y_pred).to(torch.long),
                        "embedding": emb.detach().cpu().to(torch.float16) if "emb" in locals() else None,
                        "kmeans_centers": torch.from_numpy(eval_centers).to(torch.float16) if (eval_mode == "kmeans" and do_eval and eval_centers is not None) else None,
                        "metrics": {
                            "acc": float(acc),
                            "nmi": float(nmi),
                            "ari": float(ari),
                            "f1": float(f1),
                        },
                        "rng": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    }
                    best_state = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "best_nmi": best_nmi,
                        "homo_rate": homo_rate,
                        "args": vars(args),
                        "eval_snapshot": snapshot,
                    }
                    torch.save(best_state, os.path.join(save_dir, "best.pt"))
                if float(best_acc) > prev_best:
                    no_improve = 0
                else:
                    no_improve += 1
                if int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
                    break

            if int(args.update_weights) == 1:
                do_update = (epoch + 1) % int(args.update_weights_interval) == 0 or epoch == 0
                if do_update:
                    for v in range(2):
                        r, _ = cal_homo_ratio_fast(adjs_labels[v], y_pred_t, self_loop=True)
                        homo_rate[v] = r
                    w = torch.tensor(homo_rate, device=device, dtype=torch.float32)
                    w = torch.clamp(w, min=float(args.weights_min))
                    w = w / (w.sum() + 1e-12)
                    mom = float(args.weights_momentum)
                    mom = max(0.0, min(0.999, mom))
                    weights_h = mom * weights_h + (1.0 - mom) * w
                    weights_h = weights_h / (weights_h.sum() + 1e-12)

        if int(args.debug) == 1 and ((epoch % int(args.debug_interval)) == 0):
            s = []
            s.append(_adj_density("adj_gca", adjs_labels[0]))
            s.append(_adj_density("adj_knn", adjs_labels[1]))
            s.append(_adj_density("S0", Ss[0]))
            s.append(_adj_density("S1", Ss[1]))
            for item in s:
                if item is not None:
                    print("debug", epoch, item["name"], "n", item["n"], "density", item["density"], "diag_mean", item["diag_mean"])

            print("debug", epoch, "homo_rate", [float(r) for r in homo_rate], "weights_h", [float(v) for v in weights_h.detach().cpu().tolist()])

            ts = [
                _tensor_stats("h_all", h_all),
                _tensor_stats("h0", hs[0]),
                _tensor_stats("h1", hs[1]),
                _tensor_stats("zx0", zx_norms[0]),
                _tensor_stats("zx1", zx_norms[1]),
            ]
            for item in ts:
                if item is not None:
                    print(
                        "debug",
                        epoch,
                        item["name"],
                        "shape",
                        item["shape"],
                        "mean",
                        item["mean"],
                        "std",
                        item["std"],
                        "min",
                        item["min"],
                        "max",
                        item["max"],
                        "norm",
                        item["norm"],
                        "nan",
                        item["nan"],
                        "inf",
                        item["inf"],
                    )

            cs0 = _cluster_stats("cluster_all", cluster_all)
            if cs0 is not None:
                print("debug", epoch, cs0["name"], "k", cs0["k"], "ent_mean", cs0["ent_mean"], "ent_std", cs0["ent_std"], "maxp_mean", cs0["maxp_mean"], "maxp_std", cs0["maxp_std"], "nonempty", cs0["nonempty"], "top_frac", cs0["top_frac"], "p_mean_ent", cs0["p_mean_ent"], "p_mean_top", cs0["p_mean_top"])
            for v in range(2):
                csv = _cluster_stats(f"cluster_q{v}", cluster_q[v])
                if csv is not None:
                    print("debug", epoch, csv["name"], "k", csv["k"], "ent_mean", csv["ent_mean"], "ent_std", csv["ent_std"], "maxp_mean", csv["maxp_mean"], "maxp_std", csv["maxp_std"], "nonempty", csv["nonempty"], "top_frac", csv["top_frac"], "p_mean_ent", csv["p_mean_ent"], "p_mean_top", csv["p_mean_top"])

            qg_all = _cluster_stats("qg_all", qgs[-1])
            if qg_all is not None:
                print("debug", epoch, qg_all["name"], "k", qg_all["k"], "ent_mean", qg_all["ent_mean"], "ent_std", qg_all["ent_std"], "maxp_mean", qg_all["maxp_mean"], "maxp_std", qg_all["maxp_std"], "nonempty", qg_all["nonempty"], "top_frac", qg_all["top_frac"], "p_mean_ent", qg_all["p_mean_ent"], "p_mean_top", qg_all["p_mean_top"])
            for v in range(2):
                qgv = _cluster_stats(f"qg{v}", qgs[v])
                if qgv is not None:
                    print("debug", epoch, qgv["name"], "k", qgv["k"], "ent_mean", qgv["ent_mean"], "ent_std", qgv["ent_std"], "maxp_mean", qgv["maxp_mean"], "maxp_std", qgv["maxp_std"], "nonempty", qgv["nonempty"], "top_frac", qgv["top_frac"], "p_mean_ent", qgv["p_mean_ent"], "p_mean_top", qgv["p_mean_top"])

            for i in range(len(model.cluster_layers)):
                st = _centroid_stats(f"centroid_{i}", model.cluster_layers[i])
                if st is not None:
                    print("debug", epoch, st["name"], "k", st["k"], "sim_mean", st["sim_mean"], "sim_std", st["sim_std"], "sim_min", st["sim_min"], "sim_max", st["sim_max"])

            print(
                "debug",
                epoch,
                "loss_parts",
                "contra",
                _safe_float(contra_loss),
                "match",
                _safe_float(loss_match),
                "sim",
                _safe_float(sim_loss),
                "kl_div",
                _safe_float(kl_div),
                "kl_g",
                _safe_float(loss_kl_g),
                "total",
                _safe_float(total_loss),
                "loss",
                _safe_float(loss),
            )

    if best_state is not None:
        torch.save(best_state, os.path.join(save_dir, "best.pt"))
        print("best_acc", best_state.get("best_acc", None), "best_nmi", best_state.get("best_nmi", None), "epoch", best_state["epoch"])


if __name__ == "__main__":
    main()

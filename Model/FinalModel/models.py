import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except Exception:
    GCNConv = None
    dense_to_sparse = None


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("GCN layers must be >= 1")

        self.layers = nn.ModuleList()
        current_dim = int(in_dim)

        if self.num_layers == 1:
            self.layers.append(nn.Linear(current_dim, int(out_dim), bias=True))
        else:
            # First layer
            self.layers.append(nn.Linear(current_dim, int(hidden_dim), bias=True))
            current_dim = int(hidden_dim)
            # Hidden layers
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Linear(current_dim, int(hidden_dim), bias=True))
            # Output layer
            self.layers.append(nn.Linear(current_dim, int(out_dim), bias=True))

    def _norm_adj(self, adj):
        A = adj.to(torch.float32)
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        return deg_inv_sqrt.view(-1, 1) * A * deg_inv_sqrt.view(1, -1)

    def forward(self, x, adj_mp):
        A = self._norm_adj(adj_mp)
        x = x.to(torch.float32)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h = x
        h_penultimate = h  # To store the output of the second to last layer

        for i, layer in enumerate(self.layers):
            h = torch.matmul(A, h)
            h = layer(h)

            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.normalize(h, p=2, dim=1)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h_penultimate = h
            else:
                h = F.normalize(h, p=2, dim=1)

        return h, h_penultimate


class GCNEncoderPyg(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        if GCNConv is None or dense_to_sparse is None:
            raise RuntimeError("torch_geometric is required for gcn_impl=pyg")
        self.dropout = float(dropout)
        self.conv1 = GCNConv(int(in_dim), int(hidden_dim), add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(int(hidden_dim), int(out_dim), add_self_loops=False, normalize=True)

    def forward(self, x, adj_mp):
        edge_index, _ = dense_to_sparse(adj_mp)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z1 = self.conv1(x, edge_index)
        z1 = F.elu(z1)
        z1 = F.normalize(z1, p=2, dim=1)
        z1 = F.dropout(z1, p=self.dropout, training=self.training)
        z2 = self.conv2(z1, edge_index)
        z2 = F.normalize(z2, p=2, dim=1)
        return z2, z1


class FinalModel(nn.Module):
    def __init__(
        self,
        input_dim_x,
        node_num,
        hidden_dim,
        output_dim_g,
        class_num,
        view_num=2,
        gcn_dropout=0.5,
        gcn_layers=2,
        gcn_impl="dense",
        classifier_hidden=None,
    ):
        super().__init__()
        self.class_num = int(class_num)
        self.view_num = int(view_num)
        self.gcn_impl = str(gcn_impl).strip().lower()
        if self.gcn_impl == "pyg":
            self.gcn = GCNEncoderPyg(input_dim_x, hidden_dim, output_dim_g, dropout=gcn_dropout)
        else:
            self.gcn = GCNEncoder(input_dim_x, hidden_dim, output_dim_g, num_layers=gcn_layers, dropout=gcn_dropout)

        classifier_hidden = [] if classifier_hidden is None else [int(x) for x in classifier_hidden]
        head_layers = []
        in_dim = int(output_dim_g)
        for h in classifier_hidden:
            head_layers.append(nn.Linear(in_dim, int(h)))
            head_layers.append(nn.ReLU())
            in_dim = int(h)
        head_layers.append(nn.Linear(in_dim, self.class_num))
        self.cluster_head = nn.Sequential(*head_layers)
        self.fuse_proj = nn.Linear(int(output_dim_g) * self.view_num, int(output_dim_g), bias=False)
        self.cross_projector = nn.Sequential(nn.Linear(int(output_dim_g), int(output_dim_g)), nn.ReLU())
        self.cluster_projector = nn.Sequential(nn.Linear(int(output_dim_g), self.class_num), nn.Softmax(dim=1))

        node_num = int(node_num)
        self.reg_x_enc = nn.ModuleList(
            [nn.Sequential(nn.Linear(int(output_dim_g), int(hidden_dim)), nn.ELU(), nn.Linear(int(hidden_dim), int(output_dim_g))) for _ in range(self.view_num)]
        )
        self.reg_x_dec = nn.ModuleList(
            [nn.Sequential(nn.Linear(int(output_dim_g), int(hidden_dim)), nn.ELU(), nn.Linear(int(hidden_dim), int(input_dim_x))) for _ in range(self.view_num)]
        )
        self.reg_a_enc = nn.ModuleList(
            [nn.Sequential(nn.Linear(node_num, int(hidden_dim)), nn.ELU(), nn.Linear(int(hidden_dim), int(output_dim_g))) for _ in range(self.view_num)]
        )

        self.cluster_layers = nn.ParameterList(
            [nn.Parameter(torch.empty(self.class_num, output_dim_g)) for _ in range(self.view_num + 1)]
        )
        for p in self.cluster_layers:
            nn.init.xavier_uniform_(p.data)

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layers[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def forward(self, xs, adjs_mp, adjs_labels, weights_h, rs, order=None, compute_x_pred: bool = True, compute_a_logits: bool = True):
        if not torch.is_tensor(weights_h):
            weights_h = torch.tensor(weights_h, device=xs[0].device, dtype=torch.float32)
        weights_h = weights_h.to(xs[0].device).to(torch.float32)

        zx_norms = []
        hs, Ss, qgs, cluster_logits = [], [], [], []
        zxs, zas = [], []
        x_preds, a_logits_list = [], []
        vs, cs = [], []

        for v in range(self.view_num):
            h, zx = self.gcn(xs[v], adjs_mp[v])
            zx_norms.append(zx)
            hs.append(h)

            Ss.append(adjs_labels[v].to(torch.float32))
            cluster_logits.append(self.cluster_head(h))
            qgs.append(self.predict_distribution(h, v))
            vs.append(self.cross_projector(h))
            cs.append(self.cluster_projector(vs[-1]))

            need_zx = bool(compute_x_pred) or bool(compute_a_logits)
            if need_zx:
                z_x = self.reg_x_enc[v](h)
                z_x = F.normalize(z_x, p=2, dim=1)
                zxs.append(z_x)
            else:
                z_x = None
                zxs.append(None)

            if bool(compute_x_pred):
                x_hat = self.reg_x_dec[v](z_x)
                x_preds.append(torch.sigmoid(x_hat))
            else:
                x_preds.append(None)

            if bool(compute_a_logits):
                z_a = self.reg_a_enc[v](adjs_labels[v].to(torch.float32))
                z_a = F.normalize(z_a, p=2, dim=1)
                zas.append(z_a)
                m = torch.mm(z_x.t(), z_x)
                a_logits = torch.mm(torch.mm(z_a, m), z_a.t())
                a_logits_list.append(a_logits)
            else:
                zas.append(None)
                a_logits_list.append(None)

        w_sum = weights_h.sum() + 1e-12
        h_cat = torch.cat([hs[v] * (weights_h[v] / w_sum) for v in range(self.view_num)], dim=1)
        h_all = self.fuse_proj(h_cat)
        h_all = F.normalize(h_all, p=2, dim=-1)

        cluster_all_logits = self.cluster_head(h_all)
        cluster_q = [F.softmax(cluster_logits[v], dim=1) for v in range(self.view_num)] # N*K
        cluster_all = F.softmax(cluster_all_logits, dim=1) # N*K

        qgs.append(self.predict_distribution(h_all, -1))
        x_homos = []
        return (
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
        )

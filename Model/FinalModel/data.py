import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from utils import ensure_dir, extract_tgz, normalize_spadj, normalize_spfeatures


def _find_file(root: str, candidates):
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def _maybe_extract_raw(dataset: str, data_root: str, extracted_root: str, sentinel_name: str):
    ds = dataset.lower()
    ensure_dir(extracted_root)
    out_dir = os.path.join(extracted_root, ds)
    ensure_dir(out_dir)
    found = _find_file_recursive(out_dir, sentinel_name.lower())
    if found is not None:
        return os.path.dirname(found)
    if ds == "pubmed":
        tgz = _find_file(data_root, ["Pubmed-Diabetes.tgz", "pubmed.tgz", "pubmed-diabetes.tgz"])
    else:
        tgz = _find_file(data_root, [f"{ds}.tgz", f"{ds}.tar"])
    if tgz is None:
        raise FileNotFoundError(f"Cannot find archive for {dataset} under {data_root}")
    extract_tgz(tgz, out_dir)
    found = _find_file_recursive(out_dir, sentinel_name.lower())
    if found is None:
        raise FileNotFoundError(f"{sentinel_name} not found after extracting {tgz} into {out_dir}")
    return os.path.dirname(found)


def load_citation_raw(dataset: str, data_root: str, extracted_root: str):
    ds = dataset.lower()
    base = _maybe_extract_raw(ds, data_root, extracted_root, f"{ds}.content")
    content_file = _find_file_recursive(base, f"{ds}.content")
    cites_file = _find_file_recursive(base, f"{ds}.cites")
    if content_file is None or cites_file is None:
        raise FileNotFoundError(f"Missing .content/.cites for {dataset} under {base}")

    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    features_unorm = sp.csr_matrix(idx_features_labels[:, 1:-1].astype(np.float32), dtype=np.float32)
    labels_raw = idx_features_labels[:, -1]

    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(cites_file, dtype=np.dtype(str))
    edges_list = []
    for u, v in edges_unordered:
        if u in idx_map and v in idx_map:
            edges_list.append([idx_map[u], idx_map[v]])
    edges = np.array(edges_list, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features_unorm.shape[0], features_unorm.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()

    adj_norm = adj_noeye + sp.eye(adj_noeye.shape[0], dtype=np.float32, format="csr")
    adj_norm = normalize_spadj(adj_norm)
    features = normalize_spfeatures(features_unorm)

    classes = sorted(list(set(labels_raw)))
    class_map = {c: i for i, c in enumerate(classes)}
    labels = np.array(list(map(class_map.get, labels_raw)), dtype=np.int64)

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(np.array(features.todense()))
    feature_label = torch.FloatTensor(np.array(features_unorm.todense()))
    adj = torch.FloatTensor(np.array(adj_norm.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label, feature_label


def load_pubmed_raw(data_root: str, extracted_root: str):
    base = _maybe_extract_raw("pubmed", data_root, extracted_root, "pubmed-diabetes.node.paper.tab")
    node_file = _find_file_recursive(base, "Pubmed-Diabetes.NODE.paper.tab")
    cites_file = _find_file_recursive(base, "Pubmed-Diabetes.DIRECTED.cites.tab")
    if node_file is None or cites_file is None:
        raise FileNotFoundError(f"Missing pubmed node/cites tab under {base}")

    ids = []
    labels = []
    rows = []
    cols = []
    vals = []
    feat_map = {}
    label_map = {}

    with open(node_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("paper_id"):
                continue
            parts = line.split("\t")
            pid = parts[0]
            if not pid.isdigit():
                continue
            ids.append(pid)
            feats = {}
            lab = None
            for p in parts[1:]:
                if p.startswith("label="):
                    lab = p.split("=", 1)[1]
                elif p.startswith("w-") and "=" in p:
                    k, v = p.split("=", 1)
                    feats[k] = float(v)
            if lab is None:
                lab = "unknown"
            if lab not in label_map:
                label_map[lab] = len(label_map)
            labels.append(label_map[lab])
            i = len(ids) - 1
            for k, v in feats.items():
                if k not in feat_map:
                    feat_map[k] = len(feat_map)
                j = feat_map[k]
                rows.append(i)
                cols.append(j)
                vals.append(v)

    N = len(ids)
    Fdim = len(feat_map)
    X_unorm = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))), shape=(N, Fdim), dtype=np.float32)
    X = normalize_spfeatures(X_unorm)

    idx_map = {pid: i for i, pid in enumerate(ids)}
    edge_u = []
    edge_v = []
    with open(cites_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("paper_id"):
                continue
            parts = line.split("\t")
            papers = [p for p in parts if p.startswith("paper:")]
            if len(papers) < 2:
                continue
            u = papers[0].split("paper:", 1)[1]
            v = papers[1].split("paper:", 1)[1]
            if u in idx_map and v in idx_map:
                edge_u.append(idx_map[u])
                edge_v.append(idx_map[v])
    edges = np.vstack([edge_u, edge_v]).T.astype(np.int64) if len(edge_u) else np.zeros((0, 2), dtype=np.int64)
    adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(N, N), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()
    adj_norm = adj_noeye + sp.eye(N, dtype=np.float32, format="csr")
    adj_norm = normalize_spadj(adj_norm)

    labels = torch.LongTensor(np.asarray(labels, dtype=np.int64))
    features = torch.FloatTensor(np.array(X.todense()))
    feature_label = torch.FloatTensor(np.array(X_unorm.todense()))
    adj = torch.FloatTensor(np.array(adj_norm.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label, feature_label


def _find_file_recursive(root, filename_lower):
    filename_lower = str(filename_lower).lower()
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower() == filename_lower:
                return os.path.join(r, f)
    return None


def load_acm3025(data_root: str):
    mat_path = data_root
    if os.path.isdir(data_root):
        found = _find_file_recursive(data_root, "acm3025.mat")
        if found is None:
            raise FileNotFoundError(f"ACM3025.mat not found under {data_root}")
        mat_path = found
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"ACM3025.mat not found: {mat_path}")

    data = sio.loadmat(mat_path)
    labels_oh = data["label"]
    features_np = data["feature"].astype(np.float32)
    pap = data["PAP"]
    plp = data["PLP"]

    pap = pap.tocsr() if sp.issparse(pap) else sp.csr_matrix(pap)
    plp = plp.tocsr() if sp.issparse(plp) else sp.csr_matrix(plp)

    adj_noeye = (pap + plp).tocsr()
    adj_noeye = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()

    adj = adj_noeye + sp.eye(adj_noeye.shape[0], dtype=np.float32, format="csr")
    adj = normalize_spadj(adj)

    features_unorm = sp.csr_matrix(features_np)
    features = normalize_spfeatures(features_unorm)

    if labels_oh.ndim == 2 and labels_oh.shape[1] > 1:
        labels = np.argmax(labels_oh, axis=1).astype(np.int64)
    else:
        labels = labels_oh.reshape(-1).astype(np.int64)

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(np.array(features.todense()))
    feature_label = torch.FloatTensor(np.array(features_unorm.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label, feature_label


def load_dataset(dataset: str, data_root: str, extracted_root: str):
    ds = dataset.lower()
    if ds in ["cora", "citeseer"]:
        return load_citation_raw(ds, data_root=data_root, extracted_root=extracted_root)
    if ds == "pubmed":
        return load_pubmed_raw(data_root=data_root, extracted_root=extracted_root)
    if ds in ["acm", "acm3025"]:
        return load_acm3025(data_root=data_root)
    raise ValueError(f"Unsupported dataset: {dataset}")

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
    labels_np, X_unorm, adj_noeye = _load_pubmed_sparse(data_root=data_root, extracted_root=extracted_root)
    X = normalize_spfeatures(X_unorm)
    N = int(labels_np.shape[0])
    adj_norm = adj_noeye + sp.eye(N, dtype=np.float32, format="csr")
    adj_norm = normalize_spadj(adj_norm)

    labels = torch.LongTensor(labels_np)
    features = torch.FloatTensor(np.array(X.todense()))
    feature_label = torch.FloatTensor(np.array(X_unorm.todense()))
    adj = torch.FloatTensor(np.array(adj_norm.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label, feature_label


def _load_pubmed_sparse(data_root: str, extracted_root: str):
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
    X_unorm = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(N, Fdim),
        dtype=np.float32,
    )

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
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(N, N),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()
    return np.asarray(labels, dtype=np.int64), X_unorm, adj_noeye


def _stratified_sample_indices(labels_np: np.ndarray, n: int, seed: int):
    labels_np = np.asarray(labels_np, dtype=np.int64).reshape(-1)
    N = int(labels_np.shape[0])
    if n >= N:
        return np.arange(N, dtype=np.int64)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    C = int(labels_np.max()) + 1 if N else 0
    counts = np.bincount(labels_np, minlength=C).astype(np.int64)
    probs = counts / max(1, N)
    ideal = probs * float(n)
    base = np.floor(ideal).astype(np.int64)
    base = np.minimum(base, counts)

    need = int(n - int(base.sum()))
    if need > 0:
        frac = ideal - base
        order = np.argsort(-frac)
        for c in order:
            if need <= 0:
                break
            if base[c] < counts[c]:
                base[c] += 1
                need -= 1

    cur = int(base.sum())
    if cur < n:
        for c in np.argsort(-counts):
            if cur >= n:
                break
            add = min(int(counts[c] - base[c]), int(n - cur))
            if add > 0:
                base[c] += add
                cur += add

    rng = np.random.RandomState(int(seed))
    out = []
    for c in range(C):
        k = int(base[c])
        if k <= 0:
            continue
        idx_c = np.where(labels_np == c)[0]
        if k >= idx_c.size:
            out.append(idx_c.astype(np.int64))
        else:
            out.append(rng.choice(idx_c, size=k, replace=False).astype(np.int64))
    if not out:
        return np.arange(min(n, N), dtype=np.int64)
    idx = np.concatenate(out, axis=0)
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False).astype(np.int64)
    idx = np.unique(idx)
    if idx.size < n:
        remain = np.setdiff1d(np.arange(N, dtype=np.int64), idx, assume_unique=False)
        extra = rng.choice(remain, size=(n - idx.size), replace=False).astype(np.int64)
        idx = np.concatenate([idx, extra], axis=0)
    return np.sort(idx.astype(np.int64))


def build_pubmed_small(data_root: str, extracted_root: str, n: int, seed: int, rebuild: bool = False):
    out_dir = os.path.join(extracted_root, "pubmed-small")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "pubmed-small.npz")
    if (not rebuild) and os.path.exists(out_path):
        return out_path

    labels_np, X_unorm, adj_noeye = _load_pubmed_sparse(data_root=data_root, extracted_root=extracted_root)
    idx = _stratified_sample_indices(labels_np, n=int(n), seed=int(seed))
    X_sub = X_unorm[idx]
    A_sub = adj_noeye[idx][:, idx].tocsr()
    y_sub = labels_np[idx]

    np.savez_compressed(
        out_path,
        adj_data=A_sub.data.astype(np.float32),
        adj_indices=A_sub.indices.astype(np.int32),
        adj_indptr=A_sub.indptr.astype(np.int32),
        adj_shape=np.asarray(A_sub.shape, dtype=np.int64),
        attr_data=X_sub.data.astype(np.float32),
        attr_indices=X_sub.indices.astype(np.int32),
        attr_indptr=X_sub.indptr.astype(np.int32),
        attr_shape=np.asarray(X_sub.shape, dtype=np.int64),
        labels=y_sub.astype(np.int64),
    )
    return out_path


def _load_npz_sparse(data_root: str, filename: str):
    npz_path = filename
    if os.path.isdir(data_root):
        found = _find_file_recursive(data_root, filename.lower())
        if found is None:
            found = _find_file_recursive(data_root, filename)
        if found is None:
            raise FileNotFoundError(f"{filename} not found under {data_root}")
        npz_path = found
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{filename} not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        adj = sp.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=tuple(data["adj_shape"]),
            dtype=np.float32,
        )
        features_unorm = sp.csr_matrix(
            (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
            shape=tuple(data["attr_shape"]),
            dtype=np.float32,
        )
        labels_np = np.asarray(data["labels"]).reshape(-1).astype(np.int64)

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {adj.shape}")
    if adj.shape[0] != labels_np.shape[0]:
        raise ValueError(f"labels length mismatch: N={adj.shape[0]} vs labels={labels_np.shape[0]}")
    if features_unorm.shape[0] != labels_np.shape[0]:
        raise ValueError(f"features rows mismatch: N={labels_np.shape[0]} vs X={features_unorm.shape[0]}")

    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()
    adj_noeye = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)
    return labels_np, features_unorm, adj_noeye


def build_npz_small(
    data_root: str,
    extracted_root: str,
    src_filename: str,
    out_dirname: str,
    out_filename: str,
    n: int,
    seed: int,
    rebuild: bool = False,
):
    out_dir = os.path.join(extracted_root, str(out_dirname))
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, str(out_filename))
    if (not rebuild) and os.path.exists(out_path):
        return out_path

    labels_np, X_unorm, adj_noeye = _load_npz_sparse(data_root=data_root, filename=src_filename)
    idx = _stratified_sample_indices(labels_np, n=int(n), seed=int(seed))
    X_sub = X_unorm[idx]
    A_sub = adj_noeye[idx][:, idx].tocsr()
    y_sub = labels_np[idx]

    np.savez_compressed(
        out_path,
        adj_data=A_sub.data.astype(np.float32),
        adj_indices=A_sub.indices.astype(np.int32),
        adj_indptr=A_sub.indptr.astype(np.int32),
        adj_shape=np.asarray(A_sub.shape, dtype=np.int64),
        attr_data=X_sub.data.astype(np.float32),
        attr_indices=X_sub.indices.astype(np.int32),
        attr_indptr=X_sub.indptr.astype(np.int32),
        attr_shape=np.asarray(X_sub.shape, dtype=np.int64),
        labels=y_sub.astype(np.int64),
    )
    return out_path


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


def load_npz_graph(data_root: str, filename: str):
    npz_path = filename
    if os.path.isdir(data_root):
        found = _find_file_recursive(data_root, filename.lower())
        if found is None:
            found = _find_file_recursive(data_root, filename)
        if found is None:
            raise FileNotFoundError(f"{filename} not found under {data_root}")
        npz_path = found
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{filename} not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        adj = sp.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=tuple(data["adj_shape"]),
            dtype=np.float32,
        )
        features_unorm = sp.csr_matrix(
            (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
            shape=tuple(data["attr_shape"]),
            dtype=np.float32,
        )
        labels_np = np.asarray(data["labels"]).reshape(-1).astype(np.int64)

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {adj.shape}")
    if adj.shape[0] != labels_np.shape[0]:
        raise ValueError(f"labels length mismatch: N={adj.shape[0]} vs labels={labels_np.shape[0]}")
    if features_unorm.shape[0] != labels_np.shape[0]:
        raise ValueError(f"features rows mismatch: N={labels_np.shape[0]} vs X={features_unorm.shape[0]}")

    adj_noeye = adj.tocsr()
    adj_noeye.setdiag(0)
    adj_noeye.eliminate_zeros()
    adj_noeye = adj_noeye + adj_noeye.T.multiply(adj_noeye.T > adj_noeye) - adj_noeye.multiply(adj_noeye.T > adj_noeye)

    adj_norm = adj_noeye + sp.eye(adj_noeye.shape[0], dtype=np.float32, format="csr")
    adj_norm = normalize_spadj(adj_norm)
    features = normalize_spfeatures(features_unorm)

    labels = torch.LongTensor(labels_np)
    features = torch.FloatTensor(np.array(features.todense()))
    feature_label = torch.FloatTensor(np.array(features_unorm.todense()))
    adj = torch.FloatTensor(np.array(adj_norm.todense()))
    adj_label = torch.FloatTensor(np.array(adj_noeye.todense()) != 0)
    return labels, adj, features, adj_label, feature_label


def load_dataset(
    dataset: str,
    data_root: str,
    extracted_root: str,
    seed: int = 1,
    pubmed_small_n: int = 8000,
    pubmed_small_rebuild: bool = False,
    pubmed_use_small: bool = False,
    amazon_computers_small_n: int = 8000,
    amazon_computers_small_rebuild: bool = False,
    amazon_computers_use_small: bool = False,
):
    ds = dataset.lower()
    if ds in ["cora", "citeseer"]:
        return load_citation_raw(ds, data_root=data_root, extracted_root=extracted_root)
    if ds == "pubmed":
        if pubmed_use_small:
            build_pubmed_small(
                data_root=data_root,
                extracted_root=extracted_root,
                n=int(pubmed_small_n),
                seed=int(seed),
                rebuild=bool(pubmed_small_rebuild),
            )
            return load_npz_graph(data_root=extracted_root, filename="pubmed-small.npz")
        return load_pubmed_raw(data_root=data_root, extracted_root=extracted_root)
    if ds in ["pubmed-small", "pubmed_small"]:
        return load_npz_graph(data_root=extracted_root, filename="pubmed-small.npz")
    if ds in ["acm", "acm3025"]:
        return load_acm3025(data_root=data_root)
    if ds in ["amazon_electronics_photo", "amazon-photo", "amazon_photo", "amazon_photo_npz"]:
        return load_npz_graph(data_root=data_root, filename="amazon_electronics_photo.npz")
    if ds in ["amazon_electronics_computers", "amazon-computers", "amazon_computers", "amazon_computers_npz"]:
        if amazon_computers_use_small:
            build_npz_small(
                data_root=data_root,
                extracted_root=extracted_root,
                src_filename="amazon_electronics_computers.npz",
                out_dirname="amazon_electronics_computers-small",
                out_filename="amazon_electronics_computers-small.npz",
                n=int(amazon_computers_small_n),
                seed=int(seed),
                rebuild=bool(amazon_computers_small_rebuild),
            )
            return load_npz_graph(data_root=extracted_root, filename="amazon_electronics_computers-small.npz")
        return load_npz_graph(data_root=data_root, filename="amazon_electronics_computers.npz")
    if ds in [
        "amazon_electronics_computers-small",
        "amazon_electronics_computers_small",
        "amazon-computers-small",
        "amazon_computers-small",
        "amazon_computers_small",
    ]:
        return load_npz_graph(data_root=extracted_root, filename="amazon_electronics_computers-small.npz")
    raise ValueError(f"Unsupported dataset: {dataset}")

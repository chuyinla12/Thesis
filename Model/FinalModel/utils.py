import os
import random
import tarfile
import pickle as pkl
import sys
import numpy as np
import scipy.sparse as sp
import torch
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda: bool = True):
    if cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_tgz(tgz_path: str, extract_path: str):
    ensure_dir(extract_path)
    if tgz_path.endswith(".tgz"):
        tar = tarfile.open(tgz_path, "r:gz")
        tar.extractall(path=extract_path)
        tar.close()
        return
    if tgz_path.endswith(".tar"):
        tar = tarfile.open(tgz_path, "r:")
        tar.extractall(path=extract_path)
        tar.close()
        return
    raise ValueError(f"Unsupported archive type: {tgz_path}")


def parse_index_file(filename: str):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_spfeatures(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_spadj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                continue
            y_pred[ind] = i
            ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        return 0.0, 0.0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0, visible=True):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    if visible:
        print(epoch, ":acc {:.4f}".format(acc), ", nmi {:.4f}".format(nmi), ", ari {:.4f}".format(ari), ", f1 {:.4f}".format(f1))
    return nmi, acc, ari, f1


def cal_homo_ratio(adj: np.ndarray, label: np.ndarray, self_loop: bool = True):
    label = np.asarray(label).astype(np.int64)
    class_num = int(label.max()) + 1 if label.size else 0
    y_onehot = np.eye(class_num)[label] if class_num > 0 else np.zeros((adj.shape[0], 0))
    adj_y = np.matmul(y_onehot, y_onehot.T) if class_num > 0 else np.zeros_like(adj, dtype=np.float64)
    if self_loop:
        adj = adj.copy()
        np.fill_diagonal(adj, 0.0)
    denom = np.sum(adj)
    if denom <= 0:
        return 0.5, 0.0
    homo = float(np.sum(adj_y * adj))
    homo_ratio = homo / float(denom)
    return float(homo_ratio), homo


def cal_homo_ratio_fast(adj: torch.Tensor, label: torch.Tensor, self_loop: bool = True):
    if not torch.is_tensor(adj):
        adj = torch.as_tensor(adj)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)
    A = adj
    y = label.to(torch.long)
    if A.numel() == 0 or A.dim() != 2:
        return 0.5, 0.0
    idx = (A > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return 0.5, 0.0
    if self_loop:
        idx = idx[idx[:, 0] != idx[:, 1]]
        if idx.numel() == 0:
            return 0.5, 0.0
    w = A[idx[:, 0], idx[:, 1]].to(torch.float32)
    denom = w.sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.5, 0.0
    same = (y[idx[:, 0]] == y[idx[:, 1]]).to(torch.float32)
    homo = (w * same).sum()
    homo_ratio = homo / (denom + 1e-12)
    return float(homo_ratio.detach().cpu().item()), float(homo.detach().cpu().item())


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

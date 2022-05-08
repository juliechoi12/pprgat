import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor

from utils import ppr


def csr_by_index(mat, index):
    index_np = index.cpu().numpy()
    index_tuple = [tuple(index_np[:, i]) for i in range(index_np.shape[1])]
    selected = np.array([mat[i] for i in index_tuple])
    return torch.from_numpy(selected)


def select_from_csr(mat, edge_index):
    if isinstance(edge_index, SparseTensor):
        coo = edge_index.coo()
        row, col, _ = [i.cpu().numpy() for i in coo]
    else:
        edge_index_np = edge_index.cpu().numpy()
        row, col = edge_index_np[0], edge_index_np[1]
    mat_a = mat.toarray()
    res = mat_a[row, col]
    return torch.from_numpy(res)


def ppr_matrix_from_adj(edge_index, num_nodes=None, alpha=0.25):
    if isinstance(edge_index, SparseTensor):
        coo = edge_index.coo()
        row, col, _ = [i.cpu().numpy() for i in coo]
    else:
        row, col = [edge_index[i].cpu().numpy() for i in range(2)]
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)))
    ppr_matrix = ppr.topk_ppr_matrix(
        adj_matrix,
        alpha=alpha,
        eps=1e-4,
        idx=np.arange(num_nodes),
        topk=32,
        normalization="sym",
    )
    return ppr_matrix

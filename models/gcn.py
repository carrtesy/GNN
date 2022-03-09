import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.sparse import diags, identity
from scipy.sparse.linalg import inv
from data.utils import spy_sparse2torch_sparse

class GCNLayer(nn.Module):
    def __init__(self, adj, in_features, out_features, norm_fn = "symm"):
        super().__init__()
        normalize_fn = {
            "row": self.row_normalize,
            "col": self.col_normalize,
            "symm": self.symm_normalize
        }
        self.normalized_adj = spy_sparse2torch_sparse(normalize_fn[norm_fn](adj))
        self.embed = nn.Linear(in_features, out_features)

    def forward(self, x):
        embedded = self.embed(x)
        out = self.normalized_adj @ embedded
        return out

    def get_degree_matrix(self, adj):
        '''
        :param adj: scipy adj matrix
        :return: scipy degree matrix
        '''
        D = diags(np.array(np.sum(adj, axis = 1)).reshape(-1), format="csc")
        return D

    def row_normalize(self, adj):
        adj = adj + identity(adj.shape[0])
        D = self.get_degree_matrix(adj)
        norm_adj = inv(D) @ adj
        return norm_adj

    def col_normalize(self, adj):
        adj = adj + identity(adj.shape[0])
        D = self.get_degree_matrix(adj)
        norm_adj = adj @ inv(D)
        return norm_adj

    def symm_normalize(self, adj):
        adj = adj + identity(adj.shape[0])
        D = self.get_degree_matrix(adj)
        sqrtD = np.sqrt(D)
        norm_adj = inv(sqrtD) @ adj @ inv(sqrtD)
        return norm_adj

class GCN(nn.Module):
    def __init__(self, adj, in_features, out_features, hidden_dim = 16):
        super(GCN, self).__init__()

        self.gcn1 = GCNLayer(
            adj,
            in_features=in_features,
            out_features=hidden_dim
        )

        self.gcn2 = GCNLayer(
            adj,
            in_features=hidden_dim,
            out_features=out_features
        )

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.gcn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        out = self.gcn2(x)
        out = self.dropout2(out)
        return out

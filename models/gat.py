import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.sparse import diags, identity
from scipy.sparse.linalg import inv
from data.utils import spy_sparse2torch_sparse

class MultiHeadAttention(nn.Module):
    def __init__(self, num_nodes, num_features, num_attention_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_attention_heads = num_attention_heads
        self.dot_a = nn.ModuleList(
            [nn.Linear(num_features * 2, 1) for i in range(num_attention_heads)]
        )

    def forward(self, x):
        '''
        :param x: embeddings
        :return: embeddings with attention applied
        '''
        matrix_repeat = x.repeat(self.num_nodes, 1).reshape(self.num_nodes, self.num_nodes, self.num_features) # "num_nodes" number of embeddings
        node_repeat = torch.transpose(matrix_repeat, 0, 1) # "num_nodes" copies of embedding vectors for each
        concatenated = torch.cat((node_repeat, matrix_repeat), dim = 2)
        multihead_outputs = torch.stack([self.dot_a[i](concatenated).squeeze(2) for i in range(self.num_attention_heads)])
        multihead_attns = F.softmax(multihead_outputs, dim = 2)
        avg_out = torch.mean(multihead_attns @ x, dim = 0)
        final_out = F.leaky_relu(avg_out, negative_slope=0.2)
        return final_out


class GAT(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, num_nodes):
        super().__init__()
        self.attn1 = MultiHeadAttention(num_nodes=num_nodes, num_features=hidden_dim, num_attention_heads=8)
        self.embed1 = nn.Linear(in_features, hidden_dim)
        self.attn2 = MultiHeadAttention(num_nodes=num_nodes, num_features=out_features, num_attention_heads=8)
        self.embed2 = nn.Linear(hidden_dim, out_features)


    def forward(self, x):
        # first layer
        embedded1 = self.embed1(x)
        out1 = self.attn1(embedded1)
        out1 = F.elu(out1)

        # second layer
        embedded2 = self.embed2(out1)
        final_out = self.attn2(embedded2)
        return final_out

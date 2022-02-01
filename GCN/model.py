import torch
import torch.nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, adj):
        self.adj = adj

    def forward(self, x):

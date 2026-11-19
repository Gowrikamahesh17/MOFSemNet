"""Residual GCN: a GNN that starts out as a plain MLP and can only *learn* to use its neighbours.

Why this exists (Step 5b, fix #5): the Step 5 sweep showed a GNN with no edges scores R2 ~0.84 but
the same GNN on a similarity graph drops to ~0.5, because message passing averages each MOF's own
features with those of neighbours that have different pore sizes. Here every MOF keeps its own
features through a skip connection, and the neighbour term is multiplied by a learnable gate that
starts at 0. At the start of training the model is exactly an MLP; the graph only contributes if
training finds that it helps.

Same call signature as the other models in src/baseline/gnn_training.py, so train_model /
evaluate_model work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCN(nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int):
        super().__init__()
        self.lin_in = nn.Linear(dim_in, dim_h)
        self.conv = GCNConv(dim_h, dim_h)
        self.gate = nn.Parameter(torch.zeros(1))  # 0 => pure MLP at initialisation
        self.lin_out = nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.lin_in(x))
        h = F.dropout(h, p=0.3, training=self.training)
        neighbours = F.relu(self.conv(h, edge_index, edge_weight))
        h = h + self.gate * neighbours
        return self.lin_out(h)

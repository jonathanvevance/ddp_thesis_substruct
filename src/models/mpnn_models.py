"""Python file with MPNN model classes."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
class GCN_2layer(torch.nn.Module):
    def __init__(self, in_features, out_features, mode = 'train'):
        super().__init__()
        self.mode = mode
        self.conv1 = GCNConv(in_features, 16)
        self.conv2 = GCNConv(16, out_features)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = (self.mode == 'train'))
        x = self.conv2(x, edge_index)

        return F.relu(x)


class GAT_2layer(torch.nn.Module):
    def __init__(self, in_features, out_features, mode = 'train'):
        super().__init__()
        in_head, out_head = 8, 1
        self.mode = mode
        self.conv1 = GATv2Conv(in_features, 16, heads = in_head, dropout = 0.6)
        self.conv2 = GATv2Conv(
            16 * in_head, out_features, concat = False, heads = out_head, dropout = 0.6)

    def forward(self, x, edge_index):

        x = F.dropout(x, p=0.6, training = (self.mode == 'train'))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = (self.mode == 'train'))
        x = self.conv2(x, edge_index)

        return F.relu(x)

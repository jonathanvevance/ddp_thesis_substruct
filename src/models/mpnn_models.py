"""Python file with MPNN model classes."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, GATv2Conv

# mpnn without edge attributes
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
class GCN_2layer(torch.nn.Module):
    def __init__(self, in_features, out_features, mode = 'train'):
        super().__init__()
        self.mode = mode
        self.conv1 = GeneralConv(in_features, 256, in_edge_channels = in_features) # in_edge_channels = in_features (HERE)
        self.conv2 = GeneralConv(256, out_features, in_edge_channels = in_features)  # in_edge_channels = in_features (HERE)

        # mpnn without edge attributes
        # self.conv1 = GCNConv(in_features, 256)
        # self.conv2 = GCNConv(256, out_features)

    def forward(self, x, edge_index, edge_attr):

        x = self.conv1(x, edge_index, edge_attr)
        # x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = (self.mode == 'train'))
        x = self.conv2(x, edge_index, edge_attr)
        # x = self.conv2(x, edge_index)

        return F.relu(x)


class GAT_2layer(torch.nn.Module):
    def __init__(self, in_features, out_features, mode = 'train'):
        super().__init__()
        in_head, out_head = 8, 1
        self.mode = mode
        self.conv1 = GATv2Conv(
            in_features, 256, heads = in_head, dropout = 0.6, edge_dim = in_features) # edge_dim = in_features (HERE)
        self.conv2 = GATv2Conv(
            16 * in_head, out_features, concat = False, heads = out_head, dropout = 0.6, edge_dim = in_features) # edge_dim = in_features (HERE)

        # mpnn without edge attributes
        # self.conv1 = GATConv(
        #     in_features, 256, heads = in_head, dropout = 0.6)
        # self.conv2 = GATConv(
        #     16 * in_head, out_features, concat = False, heads = out_head, dropout = 0.6)

    def forward(self, x, edge_index, edge_attr):

        x = F.dropout(x, p=0.6, training = (self.mode == 'train'))
        x = self.conv1(x, edge_index, edge_attr)
        # x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = (self.mode == 'train'))
        x = self.conv2(x, edge_index, edge_attr)
        # x = self.conv2(x, edge_index)

        return F.relu(x)

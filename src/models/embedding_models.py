"""Python file with atomic number, charge, etc embedder."""

import math
import torch
import torch.nn as nn

class FeatureEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim = 128,
        num_unique_atoms = 100,
        num_unique_charges = 13,
        num_unique_bonds = 10,
    ):
        super().__init__()
        self.num_unique_charges = num_unique_charges
        self.atomic_num_embedding = nn.Embedding(num_unique_atoms, embedding_dim)
        self.formal_charge_embedding = nn.Embedding(num_unique_charges, embedding_dim)
        self.bond_num_embedding = nn.Embedding(num_unique_bonds, embedding_dim)
        self.aromaticity_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, graph_x, graph_edge_attr):

        atomic_nums, formal_charges = torch.split(graph_x, 1, dim = 1)
        # IMPORTANT: to get whole number formal charges (used as indices)
        formal_charges = formal_charges + math.ceil(self.num_unique_charges / 2)

        atomic_nums_embedding = self.atomic_num_embedding(atomic_nums).squeeze()
        formal_charges_embedding = self.formal_charge_embedding(formal_charges).squeeze()
        atom_features = torch.cat((atomic_nums_embedding, formal_charges_embedding), dim = 1)

        bond_nums, aromaticity = torch.split(graph_edge_attr, 1, dim = 1)
        bond_nums_embedding = self.bond_num_embedding(bond_nums).squeeze()
        aromaticity_embedding = self.aromaticity_embedding(aromaticity).squeeze()
        edge_features = torch.cat((bond_nums_embedding, aromaticity_embedding), dim = 1)

        return atom_features, edge_features

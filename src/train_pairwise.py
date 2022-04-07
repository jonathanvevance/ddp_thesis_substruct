"""Python file to train pairwise interaction scores model."""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from itertools import chain
from torch_geometric.loader import DataLoader
# TODO: Look at torch_geometric.data.LightningDataset for multi-GPU training

from configs import train_pairwise_cfg as cfg
from data.dataset import reaction_record_dataset
from models.mpnn_models import GCNConv
from models.mlp_models import NeuralNet
from models.mlp_models import ScoringNetwork
from utils.generic import groupby_mean_tensors

RAW_DATASET_PATH = 'data/raw/'


def train():

    # ----- Load dataset, dataloader
    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    train_dataset = reaction_record_dataset(
        dataset_filepath = train_dataset_filepath,
        SUBSTRUCTURE_KEYS = cfg.SUBSTRUCTURE_KEYS,
        mode = 'train',
        sample_pos_fraction = cfg.SAMPLE_POS_FRACTION,
    )
    train_loader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # ----- Load models
    model_mpnn = GCNConv(2, 32) # TODO
    model_aggreg = NeuralNet() # TODO
    model_scoring = ScoringNetwork()

    # ----- Get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_mpnn = model_mpnn.to(device)
    model_aggreg = model_aggreg.to(device)
    model_scoring = model_scoring.to(device)

    # ----- Load training settings
    all_params = chain(
        model_mpnn.parameters(), model_aggreg.parameters(), model_scoring.parameters())
    optimizer = torch.optim.Adam(all_params, lr = cfg.LR, weight_decay = cfg.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.EPOCHS):
        running_loss = 0.0
        for idx, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            atom_mpnn_features = model_mpnn(train_batch.x.float(), train_batch.edge_index)

            # graph.batch -> gives 'node' maps corresponding to reaction
            # Note: we have to further apply a selector-map on them.

            atom_mlp_features = model_aggreg(atom_mpnn_features)
            select_atoms_batch_i = torch.nonzero(train_batch.selector_i, as_tuple=True)
            selected_atom_features_i = atom_mlp_features[select_atoms_batch_i]
            rxn_indices_i = train_batch.batch[select_atoms_batch_i]

            select_atoms_batch_j = torch.nonzero(train_batch.selector_j, as_tuple=True)
            selected_atom_features_j = atom_mlp_features[select_atoms_batch_j]
            rxn_indices_j = train_batch.batch[select_atoms_batch_j]

            assert len(selected_atom_features_i) == len(rxn_indices_i) # True
            assert len(selected_atom_features_j) == len(rxn_indices_j) # True

            substruct_features_i = groupby_mean_tensors(
                selected_atom_features_i, rxn_indices_i
            )
            substruct_features_j = groupby_mean_tensors(
                selected_atom_features_j, rxn_indices_j
            )

            scores = model_scoring(substruct_features_i, substruct_features_j)
            loss = criterion(scores, train_batch.target)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if idx % 2000 == 1999:    # print every 2000 mini-batches
            if idx % 2 == 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss}')
                # running_loss = 0.0

    print('Finished Training')

# load model

# load train settings

# train model

# save results


if __name__ == '__main__':
    train()

"""Python file to train pairwise interaction scores model."""

import os
import torch
from itertools import chain
from torch_geometric.loader import DataLoader

from configs import train_pairwise_cfg as cfg
from data.dataset import reaction_record_dataset
from models.mpnn_models import GCNConv
from models.mlp_models import NeuralNet

RAW_DATASET_PATH = 'data/raw/'


def train():

    # ----- Load dataset, dataloader
    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    train_dataset = reaction_record_dataset(train_dataset_filepath, cfg.SUBSTRUCTURE_KEYS, 'train')
    train_loader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # ----- Load models
    model_mpnn = GCNConv(2, 32) # TODO
    model_scoring = NeuralNet() # TODO

    # ----- Get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_mpnn = model_mpnn.to(device)
    model_scoring = model_scoring.to(device)

    # ----- Load training settings
    all_params = chain(model_mpnn.parameters(), model_scoring.parameters())
    optimizer = torch.optim.Adam(all_params, lr = cfg.LR, weight_decay = cfg.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(2):
        running_loss = 0.0
        for idx, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            atom_mpnn_features = model_mpnn(train_batch.x.float(), train_batch.edge_index)

            # graph.batch -> gives 'node' maps corresponding to reaction
            # Note: we have to further apply a selector-map on them.

            atom_mlp_features = model_scoring(atom_mpnn_features)
            print(atom_mlp_features.shape)
            print('mlp step successful')

            # select BOTH atoms AND DATA.BATCH --> https://stackoverflow.com/questions/60032073/select-specific-rows-of-2d-pytorch-tensor

            # selected_atom_features = atom_features.apply(selector) # TODO
            # interaction_scores = model_scoring(selected_atom_features)

            # loss = criterion(interaction_score, target)
            # loss.backward()
            # optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # if idx % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

    print('Finished Training')

# load model

# load train settings

# train model

# save results


if __name__ == '__main__':
    train()

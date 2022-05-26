"""Python file for pytorch util functions."""

import os
import torch

from models.mpnn_models import GCN_2layer
from models.mpnn_models import GAT_2layer
from models.mlp_models import NeuralNet
from models.mlp_models import ScoringNetwork
from models.embedding_models import FeatureEmbedding

def set_train_mode(models):
    for model in models:
        model.train()

def set_eval_mode(models):
    for model in models:
        model.eval()

def load_models(cfg):

    if cfg.USE_EMBEDDING:
        model_embedding = FeatureEmbedding(
            cfg.EMBEDDING_DIM, cfg.UNIQUE_ATOMS, cfg.UNIQUE_CHARGES, cfg.UNIQUE_BONDS
        )
    else:
        model_embedding = None

    model_feedforward = NeuralNet() if cfg.USE_MLP_FOR_PROCESSING else None
    model_mpnn = GCN_2layer(cfg.MPNN_FEATURES_DIM, 256, 'train')
    model_scoring = ScoringNetwork()

    # if saved model exists, load it
    if cfg.LOAD_MODEL_PATH and os.path.exists(cfg.LOAD_MODEL_PATH):

        model_weights = torch.load(cfg.LOAD_MODEL_PATH)
        model_mpnn.load_state_dict(model_weights['mpnn'])
        model_scoring.load_state_dict(model_weights['scoring'])

        if model_embedding and 'embedding' in model_weights:
            model_embedding.load_state_dict(model_weights['embedding'])

        if model_feedforward and 'feedforward' in model_weights:
            model_feedforward.load_state_dict(model_weights['feedforward'])

    return model_mpnn, model_feedforward, model_scoring, model_embedding

def save_models(cfg, model_mpnn, model_feedforward, model_scoring, model_embedding):

    model_weights_dict = {
        'mpnn': model_mpnn.state_dict(),
        'feedforward': model_feedforward.state_dict() if model_feedforward else None,
        'scoring': model_scoring.state_dict(),
        'embedding': model_embedding.state_dict() if model_embedding else None,
    }
    torch.save(model_weights_dict, cfg.SAVE_MODEL_PATH)

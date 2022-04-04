"""Python file generic util functions."""

import torch

def nested2d_generator(list_A, list_B):
     """Product a stream of 2D coordinates."""
     for a in list_A:
        for b in list_B:
            yield a, b

def groupby_mean_tensors(samples, labels):
    """Groupby mean of tensors with corresponding labels given."""
    # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
    labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
    result = result / labels_count.float().unsqueeze(1)
    return result
"""Python file to make dataset structures."""

import os
from torch.utils.data import DataLoader

from data.dataset import reaction_record_dataset

RAW_DATASET_PATH = 'data/raw/'
SUBSTRUCTURE_KEYS = 'MACCS_FULL'

def prep_dataset():
    """Prepare datasets."""

    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    train_dataset = reaction_record_dataset(train_dataset_filepath, SUBSTRUCTURE_KEYS, 'train') # Disk space required = 150GB approx
    DataLoader(train_dataset)

    test_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'test.txt')
    test_dataset = reaction_record_dataset(test_dataset_filepath, SUBSTRUCTURE_KEYS, 'test')  # Disk space required = 15GB approx
    DataLoader(test_dataset)

    val_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'valid.txt')
    val_dataset = reaction_record_dataset(val_dataset_filepath, SUBSTRUCTURE_KEYS, 'val') # Disk space required = 12GB approx
    DataLoader(val_dataset)


if __name__ == '__main__':
    prep_dataset()

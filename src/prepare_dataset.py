"""Python file to make dataset structures."""

import os

from tqdm import tqdm
from rdkit import Chem

from data.dataset import reactionInfoDataset

RAW_DATASET_PATH = 'data/raw/'
SUBSTRUCTURE_KEYS = 'MACCS_FULL'


#! -----------------------------------------------------------------------------------------------------------
# DATA PART

# get substructure match tuples - CAN DO
# for all non-substruct-atoms, mark bonds - CAN DO
# for all substruct-atoms, mark bonds with atoms outside it - CAN DO

# [targets] for each substructure-pair, are they interacting?
#       a. If (A, B) have common atoms, interaction = 0
#       b. If for any atom in A, the product-bonded atoms includes any atom in B, interaction = 1

# [inputs] Readout for pair of substructure vectors?
#       a. Have multi hot vectors (for atoms) for each substructure match
#       b. Get R1 Readout for substructure representation.     --->> [LATER]
#       c. Get R2 Readout for pair of substructures -> [0, 1]  --->> [LATER]

#! -----------------------------------------------------------------------------------------------------------
# TICKTICK NOTE
# Can we do it WITHOUT dummy atoms?
# - we can leave original graph as it is.
# - we perform node-wise feature extraction as usual using MPNN. 
# - For vector representation of each substructure, we can do separate readout function runs. Then f(v1,v2) =?
# - For NERF, we can "construct" the LHS using atom maps corresponding to pairs of substructures. Again no need for dummy atoms. 
# - Issues
#     - **How to decide interaction(A, B) -> CHECK**
#     - How to decide which all to run? -> LATER (greedy elimination in table, ...)

# Pytorch implementation idea --> use a (multihot vectors) selector matrix to select all vectors for all atoms involved in pairs of substructures. use it maybe with a simple summation readout function.  

# For NERF on pairs, create a new dataset with true inputs (2-stage training). Converting to SMILES might be an easy way. 

#! -----------------------------------------------------------------------------------------------------------

def prep_dataset():
    """Prepare datasets."""

    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    reactionInfoDataset(train_dataset_filepath, SUBSTRUCTURE_KEYS, mode = 'train')

    test_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'test.txt')
    reactionInfoDataset(test_dataset_filepath, SUBSTRUCTURE_KEYS, mode = 'test')

    val_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'valid.txt')
    reactionInfoDataset(val_dataset_filepath, SUBSTRUCTURE_KEYS, mode = 'val')


if __name__ == '__main__':
    prep_dataset()

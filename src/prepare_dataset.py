"""Python file to make dataset structures."""

import os

from data.dataset import reaction_record_dataset

RAW_DATASET_PATH = 'data/raw/'
SUBSTRUCTURE_KEYS = 'MACCS_FULL'
SAVE_EVERY = 10000 # how often to save reactions (to continue running later)


#! -----------------------------------------------------------------------------------------------------------
# DATA PART

# get substructure match tuples - DONE
# for all non-substruct-atoms, mark bonds - DONE
# for all substruct-atoms, mark bonds with atoms outside it - DONE

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
    reaction_record_dataset(train_dataset_filepath, SUBSTRUCTURE_KEYS, 'train', SAVE_EVERY)

    test_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'test.txt')
    reaction_record_dataset(test_dataset_filepath, SUBSTRUCTURE_KEYS, 'test', SAVE_EVERY)

    val_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'valid.txt')
    reaction_record_dataset(val_dataset_filepath, SUBSTRUCTURE_KEYS, 'val', SAVE_EVERY)


if __name__ == '__main__':
    prep_dataset()

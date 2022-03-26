"""Python file to make dataset structures."""

import os

from tqdm import tqdm
from rdkit import Chem

from substructures.get_substructures import get_matching_atomidx_tuples

SUBSTRUCTURE_KEYS = 'MACCS_FULL'

# DATA PART

# get substructure match tuples - CAN DO
# for all non-substruct-atoms, mark bonds - CAN DO
# for all substruct-atoms, mark bonds with atoms outside it - CAN DO

# [targets] for each substructure-pair, are they interacting?
#       a. If (A, B) have common atoms, interaction = 0
#       b. If for any atom in A, the product-bonded atoms includes any atom in B, interaction = 1

# [inputs] Readout for pair of substructure vectors?
#       a. Have multi hot vectors (for atoms) for each substructure match
#       b. Get R1 Readout for substructure representation.    --->> [LATER]
#       c. Get R2 Readout for pair of substructures -> [0, 1] --->> [LATER]


if __name__ == '__main__':

    num_rxns = sum(1 for line in open("data/raw/train.txt", "r"))

    with open("data/raw/train.txt", "r") as train_dataset:
        for reaction in tqdm(train_dataset, total = num_rxns):

            lhs, rhs = reaction.split(">>")
            lhs_mol = Chem.MolFromSmiles(lhs)

            get_matching_atomidx_tuples(lhs_mol, SUBSTRUCTURE_KEYS)
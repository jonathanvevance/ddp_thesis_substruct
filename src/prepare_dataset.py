"""Python file to make dataset structures."""

import os

from substructures.get_substructures import *

SUBSTRUCTURE_KEYS = 'MACCS'

# DATA PART

# get substructure match tuples - CAN DO
# for all non-substruct-atoms, mark bonds - CAN DO
# for all substruct-atoms, mark bonds with atoms outside it - CAN DO

# [targets] for each substructure-pair, are they interacting?
#       a. If (A, B) have common atoms, interaction = 0
#       b. If for any atom in A, the product-bonded atoms includes any atom in B, interaction = 1

# [inputs] Readout for pair of substructure vectors?
#       a. Have multi hot vectors (for atoms) for each substructure match
#       b. Get R1 Readout for substructure representation.
#       c. Get R2 Readout for pair of substructures -> [0, 1]


if __name__ == '__main__':
    print(os.path.exists("data/raw/train.txt"))

    rxn_dataset = []
    for data in rxn_dataset:
        pass
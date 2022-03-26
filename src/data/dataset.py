
from tqdm import tqdm
from rdkit import Chem

from substructures.get_substructures import get_atomid_matches_and_bonds

PROCESSED_DATASET_LOC = ''

class reactionInfoDataset():
    """Class to hold all reaction information."""

    def __init__(self, dataset_filepath, SUBSTRUCTURE_KEYS, mode = 'train'):

        self.mode = mode
        self.data_records = []
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.dataset_filepath = dataset_filepath

    def rename_later(self):

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        with open(self.dataset_filepath, "r") as train_dataset:
            for reaction in tqdm(train_dataset, total = num_rxns):

                lhs, rhs = reaction.split(">>")
                lhs_mol = Chem.MolFromSmiles(lhs)

                get_atomid_matches_and_bonds(lhs_mol, self.SUBSTRUCTURE_KEYS)
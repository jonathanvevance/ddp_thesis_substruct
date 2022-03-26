
from tqdm import tqdm
from rdkit import Chem

from substructures.get_substructures import get_atomid_matches_and_bonds

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record:
    def __init__(self):
        pass

class reaction_record_dataset():
    """Class to hold all reaction information."""

    def __init__(self, dataset_filepath, SUBSTRUCTURE_KEYS, mode='train', SAVE_EVERY=10000):

        self.mode = mode
        self.data_records = []
        self.SAVE_EVERY = SAVE_EVERY
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.dataset_filepath = dataset_filepath

        self.process_reactions()

    def process_reactions(self):

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        # load database-stored dataset
        num_reactions = 0 # find out num reactions processed
        start_from = num_reactions + 1

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction in enumerate(tqdm(train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns)):

                if rxn_num < start_from - 1:
                    continue

                lhs, rhs = reaction.split(">>")
                lhs_mol = Chem.MolFromSmiles(lhs)

                all_atomidx_tuples, all_bonds_for_recon_sets = get_atomid_matches_and_bonds(lhs_mol, self.SUBSTRUCTURE_KEYS)
                reaction = reaction_record(all_atomidx_tuples, all_bonds_for_recon_sets)



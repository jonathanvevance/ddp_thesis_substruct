
from tqdm import tqdm
from rdkit import Chem

from rdkit_helpers.substructures import get_substruct_matches

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record:
    def __init__(self, all_atomidx_tuples, all_bonds_for_recon_sets):
        # for each atom in lhs, RDKit features (list of lists)
        # 
        pass

    def construct_selector_for_match():
        pass

    def construct_selector_for_match_pair():
        pass

class reaction_record_dataset():
    """Class to hold all reaction information."""

    def __init__(self, dataset_filepath, SUBSTRUCTURE_KEYS, mode='train', SAVE_EVERY=10000):

        self.mode = mode
        self.SAVE_EVERY = SAVE_EVERY
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.dataset_filepath = dataset_filepath

        self.pyg_dataset = None # See https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8

        self.process_reactions()

    def process_reactions(self):

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        # https://www.askpython.com/python/examples/save-data-in-python
        start_from = 0 # TODO: find out num of processed reactions saved

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction in enumerate(tqdm(train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns)):

                if rxn_num < start_from:
                    continue

                lhs, rhs = reaction.split(">>")
                lhs_mol = Chem.MolFromSmiles(lhs)

                matching_atomidx_tuples, recon_bonds_per_match = get_substruct_matches(lhs_mol, self.SUBSTRUCTURE_KEYS)
                a = get_atom_feature_vectors(lhs_mol)
                reaction = reaction_record(matching_atomidx_tuples, recon_bonds_per_match)

                # THINGS NEEDED in each reaction_record
                # 1. RDKit feature vectors for each atom (np matrix)
                # 2. 



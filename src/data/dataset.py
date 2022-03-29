
from tqdm import tqdm
from rdkit import Chem

from rdkit_helpers.substructures import get_substruct_matches
from rdkit_helpers.features import get_pyg_dataset_requirements

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

    #! IDEA: for idx -> choose dataset. Then sample one random entry from interaction matrix.
    #! keep a 0 vs 1 balance and enforce that (class imbalance)

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

                lhs_smiles, rhs_smiles = reaction.split(">>")
                lhs_mol = Chem.MolFromSmiles(lhs_smiles)

                substruct_matches = get_substruct_matches(lhs_mol, self.SUBSTRUCTURE_KEYS)
                pyg_requirements = get_pyg_dataset_requirements(lhs_mol)

                reaction = reaction_record(pyg_requirements, substruct_matches)

                # THINGS NEEDED in each reaction_record
                # 1. RDKit feature vectors for each atom (np matrix) - YES
                # 2.



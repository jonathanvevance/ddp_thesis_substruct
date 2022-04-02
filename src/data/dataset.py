"""Python file with dataset classes defined."""

import os
import random
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from utils.generic import nested2d_generator
from rdkit_helpers.generic import get_map_to_id_dict
from rdkit_helpers.substructures import get_substruct_matches
from rdkit_helpers.features import get_pyg_graph_requirements

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record:
    def __init__(self, reaction_smiles, SUBSTRUCTURE_KEYS):

        lhs_smiles, rhs_smiles = reaction_smiles.split(">>")
        self.lhs_mol = Chem.MolFromSmiles(lhs_smiles)
        self.rhs_mol = Chem.MolFromSmiles(rhs_smiles)

        pyg_requirements = get_pyg_graph_requirements(self.lhs_mol)
        self.substruct_matches = get_substruct_matches(self.lhs_mol, SUBSTRUCTURE_KEYS)
        self.map_to_id_dicts = {
            'lhs': get_map_to_id_dict(self.lhs_mol),
            'rhs': get_map_to_id_dict(self.rhs_mol),
        }

        self.pyg_data = Data(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

        self.valid_pairs_substructs = []
        self.save_valid_suibstruct_pairs()

    def save_valid_suibstruct_pairs(self):
        """Records valid substructure pairs."""
        for i in range(len(self.substruct_matches['matches'])):
            for j in range(i + 1, len(self.substruct_matches['matches'])):

                atom_map_tuple_i = self.substruct_matches['matches'][i]
                atom_map_tuple_j = self.substruct_matches['matches'][j]

                # if any common atom, ignore this substructure pair
                if len(set(atom_map_tuple_i).intersection(atom_map_tuple_j)):
                    continue

                self.valid_pairs_substructs.append((i, j))

    def sample_selector_and_target(self):
        """
        Sample random substructure piar. Get multihot selector and target for this pair.

        Returns:
            selector: multi hot selector for this pair.
            int(interacting): 1 or 0 if the pair is interacting.
        """

        random_pair_idx = random.randint(0, len(self.valid_pairs_substructs) - 1)
        i, j = self.valid_pairs_substructs[random_pair_idx]

        atom_map_tuple_i = self.substruct_matches['matches'][i]
        atom_map_tuple_j = self.substruct_matches['matches'][j]

        # ----- multi-hot selectors
        selector = np.zeros(len(self.substruct_matches['matches']))
        all_atom_maps = set(atom_map_tuple_i).union(atom_map_tuple_j)
        selector[list(all_atom_maps)] = 1

        # ----- interaction targets
        interacting = False
        all_atom_pairs_tuples = nested2d_generator(
            atom_map_tuple_i, atom_map_tuple_j
        )

        for atom_map_i, atom_map_j in all_atom_pairs_tuples:

            lhs_id_i = self.map_to_id_dicts['lhs'][atom_map_i]
            lhs_id_j = self.map_to_id_dicts['lhs'][atom_map_j]
            try:
                rhs_id_i = self.map_to_id_dicts['rhs'][atom_map_i]
                rhs_id_j = self.map_to_id_dicts['rhs'][atom_map_j]
            except KeyError:
                continue # atom_map does not exist on RHS

            bond_lhs = self.lhs_mol.GetBondBetweenAtoms(lhs_id_i, lhs_id_j)

            if bond_lhs: # bond already exists on LHS
                continue # hence, bond_rhs does not indicate interaction

            bond_rhs = self.rhs_mol.GetBondBetweenAtoms(rhs_id_i, rhs_id_j)
            if bond_rhs:
                interacting = True
                break

        return selector, int(interacting)


class reaction_record_dataset(Dataset):
    """Class to hold all reaction information."""
    def __init__(
        self,
        dataset_filepath,
        SUBSTRUCTURE_KEYS,
        mode='train',
        transform = None,
        pre_transform = None,
        pre_filter = None,
    ):

        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
        super().__init__(
            None,
            transform,
            pre_transform,
            pre_filter,
        ) # None to skip downloading (see FAQ)

        self.mode = mode
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.dataset_filepath = dataset_filepath
        self.processed_mode_dir = os.path.join(PROCESSED_DATASET_LOC, self.SUBSTRUCTURE_KEYS, self.mode)
        self.processed_filepaths = []

        self.process_reactions()

    def process_reactions(self):
        """Process each reaction in the dataset."""

        if not os.path.exists(self.processed_mode_dir):
            os.makedirs(self.processed_mode_dir)

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction_smiles in enumerate(tqdm(
                train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns
            )):

                if rxn_num == 100: return # TODO: remove

                proccessed_filepath = os.path.join(self.processed_mode_dir, f'rxn_{rxn_num}.pt')
                if os.path.exists(proccessed_filepath):
                    self.processed_filepaths.append(proccessed_filepath)
                    continue

                reaction = reaction_record(reaction_smiles, self.SUBSTRUCTURE_KEYS)

                if self.pre_filter is not None and not self.pre_filter(reaction):
                    continue

                if self.pre_transform is not None:
                    reaction = self.pre_transform(reaction)

                torch.save(reaction, proccessed_filepath)
                self.processed_filepaths.append(proccessed_filepath)

    def len(self):
        """Get length of reaction dataset."""
        return len(self.processed_filepaths)

    def get(self, idx):
        """Get data point for given reaction-idx."""
        processed_filepath = self.processed_filepaths[idx]
        reaction_data = torch.load(processed_filepath) # load graph

        # load substruct-pair and target
        selector, target = reaction_data.sample_selector_and_target()

        print(len(selector))
        return reaction_data.pyg_data, torch.tensor(selector), target

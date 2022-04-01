"""Python file with dataset classes defined."""

import os
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
    def __init__(self, reaction, SUBSTRUCTURE_KEYS):

        lhs_smiles, rhs_smiles = reaction.split(">>")
        lhs_mol = Chem.MolFromSmiles(lhs_smiles)
        rhs_mol = Chem.MolFromSmiles(rhs_smiles)

        substruct_matches = get_substruct_matches(lhs_mol, SUBSTRUCTURE_KEYS)
        pyg_requirements = get_pyg_graph_requirements(lhs_mol)
        map_to_id_dicts = {
            'lhs': get_map_to_id_dict(lhs_mol),
            'rhs': get_map_to_id_dict(rhs_mol),
        }

        self.pyg_data = Data(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

        self.substruct_pair_targets = []
        self.substruct_pair_selectors = []
        self.substruct_pair_recon_bonds = []

        # TODO: make this on the fly
        self.process_substruct_pairs(
            lhs_mol,
            rhs_mol,
            substruct_matches['matches'],
            substruct_matches['bonds'],
            map_to_id_dicts['lhs'],
            map_to_id_dicts['rhs'],
        )

    def process_substruct_pairs(
        self,
        lhs_mol,
        rhs_mol,
        matching_atom_map_tuples,
        recon_bonds_per_match,
        lhs_map_to_id_dicts,
        rhs_map_to_id_dicts,
    ):

        for i in range(len(matching_atom_map_tuples)):
            for j in range(i + 1, len(matching_atom_map_tuples)):

                atom_map_tuple_i = matching_atom_map_tuples[i]
                atom_map_tuple_j = matching_atom_map_tuples[j]

                # if any common atom, ignore this substructure pair
                if len(set(atom_map_tuple_i).intersection(atom_map_tuple_j)):
                    continue

                # ----- multi-hot selectors
                selector = np.zeros(len(matching_atom_map_tuples))
                all_atom_maps = set(atom_map_tuple_i).union(atom_map_tuple_j)
                selector[list(all_atom_maps)] = 1
                self.substruct_pair_selectors.append(selector)

                # ----- interaction targets
                interacting = False
                all_atom_pairs_tuples = nested2d_generator(
                    atom_map_tuple_i, atom_map_tuple_j
                )

                for atom_map_i, atom_map_j in all_atom_pairs_tuples:

                    lhs_id_i = lhs_map_to_id_dicts[atom_map_i]
                    lhs_id_j = lhs_map_to_id_dicts[atom_map_j]
                    try:
                        rhs_id_i = rhs_map_to_id_dicts[atom_map_i]
                        rhs_id_j = rhs_map_to_id_dicts[atom_map_j]
                    except KeyError:
                        continue # atom_map does not exist on RHS

                    bond_lhs = lhs_mol.GetBondBetweenAtoms(lhs_id_i, lhs_id_j)

                    if bond_lhs: # bond already exists on LHS
                        continue # hence, bond_rhs does not indicate interaction

                    bond_rhs = rhs_mol.GetBondBetweenAtoms(rhs_id_i, rhs_id_j)
                    if bond_rhs:
                        interacting = True
                        break

                if interacting:
                    self.substruct_pair_targets.append(1)
                else:
                    self.substruct_pair_targets.append(0)

                # ----- recon bonds for this substructure pair
                self.substruct_pair_recon_bonds.append(
                    recon_bonds_per_match[i].intersection(recon_bonds_per_match[j])
                )


class reaction_record_dataset(Dataset):
    """Class to hold all reaction information."""
    def __init__(
        self,
        dataset_filepath,
        SUBSTRUCTURE_KEYS,
        mode='train',
        SAVE_EVERY=10000,
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
        self.SAVE_EVERY = SAVE_EVERY
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.dataset_filepath = dataset_filepath
        self.processed_mode_dir = os.path.join(PROCESSED_DATASET_LOC, self.mode)
        self.processed_filepaths = []

        self.process_reactions()

    def process_reactions(self):

        if not os.path.exists(self.processed_mode_dir):
            os.makedirs(self.processed_mode_dir)

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction_smiles in enumerate(tqdm(
                train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns
            )):

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
        return len(self.processed_filepaths)

    def get(self, idx):
        # TODO: get a substructure pair target also (return tuple)
        processed_filepath = self.processed_filepaths[idx]
        reaction_data = torch.load(processed_filepath)
        return reaction_data.pyg_data


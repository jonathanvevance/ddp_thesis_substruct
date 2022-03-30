"""Python file with dataset classes defined."""

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data

from utils.generic import nested2d_generator
from rdkit_helpers.generic import get_map_to_id_dict
from rdkit_helpers.substructures import get_substruct_matches
from rdkit_helpers.features import get_pyg_graph_requirements

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record(Data):
    def __init__(self, reaction):

        lhs_smiles, rhs_smiles = reaction.split(">>")
        lhs_mol = Chem.MolFromSmiles(lhs_smiles)
        rhs_mol = Chem.MolFromSmiles(rhs_smiles)

        substruct_matches = get_substruct_matches(lhs_mol, self.SUBSTRUCTURE_KEYS)
        pyg_requirements = get_pyg_graph_requirements(lhs_mol)
        map_to_id_dicts = {
            'lhs': get_map_to_id_dict(lhs_mol),
            'rhs': get_map_to_id_dict(rhs_mol),
        }

        super().__init__(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

        self.substruct_pair_targets = []
        self.substruct_pair_selectors = []
        self.substruct_pair_recon_bonds = []

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

                # if any common atom, ignore this substructure pair
                if len(set(atom_map_tuple_i).intersection(atom_map_tuple_j)):
                    continue

                # ----- multi-hot selectors
                selector = np.zeros(len(matching_atom_map_tuples))
                atom_map_tuple_i = matching_atom_map_tuples[i]
                atom_map_tuple_j = matching_atom_map_tuples[j]
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
                    set(recon_bonds_per_match[i]).intersection(recon_bonds_per_match[j])
                )



class reaction_record_dataset():
    """Class to hold all reaction information."""

    #! IDEA: for idx -> choose dataset. Then sample one random entry from interaction matrix.
    #! keep a 0 vs 1 balance and enforce that (class imbalance)

    def __init__(
        self,
        dataset_filepath,
        SUBSTRUCTURE_KEYS,
        mode='train',
        SAVE_EVERY=10000
    ):

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
            for rxn_num, reaction in enumerate(tqdm(
                train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns
            )):

                if rxn_num < start_from:
                    continue

                reaction = reaction_record(reaction)

                # THINGS NEEDED in each reaction_record
                # 1. RDKit feature vectors for each atom (np matrix) - DONE
                # 2. 



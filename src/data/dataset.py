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

def reaction_filter(reaction):
    return reaction.is_valid()

class reaction_record:
    def __init__(self, reaction_smiles, SUBSTRUCTURE_KEYS):

        lhs_smiles, rhs_smiles = reaction_smiles.split(">>")
        self.lhs_mol = Chem.MolFromSmiles(lhs_smiles)
        self.rhs_mol = Chem.MolFromSmiles(rhs_smiles)

        pyg_requirements = get_pyg_graph_requirements(self.lhs_mol)
        substruct_matches = get_substruct_matches(lhs_smiles, SUBSTRUCTURE_KEYS)

        self.num_atoms = self.lhs_mol.GetNumAtoms()
        self.matches = substruct_matches['matches']
        self.bonds = substruct_matches['bonds']
        self.map_to_molid_dict = substruct_matches['map_to_molid_dict']

        self.map_to_id_dicts = {
            'lhs': get_map_to_id_dict(self.lhs_mol),
            'rhs': get_map_to_id_dict(self.rhs_mol),
        }

        self.pyg_data = Data(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

        self.pos_substructs_and_tgts = []
        self.neg_substructs_and_tgts = []
        self.save_substruct_pairs_and_targets()

    def save_substruct_pairs_and_targets(self):
        """Records valid substructure pairs."""
        for i in range(len(self.matches)):
            for j in range(i + 1, len(self.matches)):

                atom_map_tuple_i = self.matches[i]
                atom_map_tuple_j = self.matches[j]

                molid_i = self.map_to_molid_dict[atom_map_tuple_i[0]]
                molid_j = self.map_to_molid_dict[atom_map_tuple_j[0]]

                if molid_i == molid_j: # no interaction within same molecule
                    continue

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

                if interacting:
                    self.pos_substructs_and_tgts.append((i, j, int(interacting)))
                else:
                    self.neg_substructs_and_tgts.append((i, j, int(interacting)))

    def is_valid(self):
        if len(self.pos_substructs_and_tgts) == 0 or len(self.neg_substructs_and_tgts) == 0:
            return False
        return True

    def sample_selector_and_target(self, sample_pos_fraction):
        """
        Sample random substructure piar. Get multihot selector and target for this pair.

        Returns:
            selector: multi hot selector for this pair.
            int(interacting): 1 or 0 if the pair is interacting.
        """

        if random.uniform(0, 1) < sample_pos_fraction:
            pos_pair_idx = random.randint(0, len(self.pos_substructs_and_tgts) - 1)
            i, j, interacting = self.pos_substructs_and_tgts[pos_pair_idx]
        else:
            neg_pair_idx = random.randint(0, len(self.neg_substructs_and_tgts) - 1)
            i, j, interacting = self.neg_substructs_and_tgts[neg_pair_idx]

        atom_idx_list_i = sorted([(atom_map - 1) for atom_map in self.matches[i]])
        atom_idx_list_j = sorted([(atom_map - 1) for atom_map in self.matches[j]])

        # ----- multi-hot selectors
        selector_i = np.zeros(self.num_atoms)
        selector_j = np.zeros(self.num_atoms)

        selector_i[atom_idx_list_i] = 1
        selector_j[atom_idx_list_j] = 1

        return selector_i, selector_j, int(interacting)


class reaction_record_dataset(Dataset):
    """
    Class to hold all reaction information.

    Implementation notes:
        1.  The get(idx) call samples the reaction given by idx. The information
            required for training is not just the reaction graph but also a
            substructure-pair and the corresponding interaction score.

        2.  So a random substructure-pair is sampled from the 'idx' reaction.
            Corresponding atom-selectors are attached to the 'Data' object (takes
            advantage of torch_geometric.loader.DataLoader batching strategy).

        3.  Hence a single training step on a reaction ONLY trains ONE of the
            possible substructure-pairs.
    """
    def __init__(
        self,
        dataset_filepath,
        SUBSTRUCTURE_KEYS,
        mode='train',
        sample_pos_fraction = 0.5,
        transform = None,
        pre_transform = None,
        pre_filter = reaction_filter,
    ):

        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
        super().__init__(
            None,
            transform,
            pre_transform,
            pre_filter,
        ) # None to skip downloading (see FAQ)

        self.mode = mode
        self.dataset_filepath = dataset_filepath
        self.SUBSTRUCTURE_KEYS = SUBSTRUCTURE_KEYS
        self.sample_pos_fraction = sample_pos_fraction
        self.processed_mode_dir = os.path.join(PROCESSED_DATASET_LOC, self.SUBSTRUCTURE_KEYS, self.mode)
        self.processed_filepaths = []

        self.process_reactions()

    def process_reactions(self):
        """Process each reaction in the dataset."""

        if not os.path.exists(self.processed_mode_dir):
            os.makedirs(self.processed_mode_dir)

        reaction_files = os.listdir(self.processed_mode_dir)
        if len(reaction_files):
            start_from = max(int(reaction_file[4:-3]) for reaction_file in reaction_files)
        else:
            start_from = -1

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction_smiles in enumerate(tqdm(
                train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns
            )):

                if rxn_num == 50000: return # TODO: remove

                processed_filepath = os.path.join(self.processed_mode_dir, f'rxn_{rxn_num}.pt')
                if rxn_num < start_from + 1:
                    if os.path.exists(processed_filepath):
                        self.processed_filepaths.append(processed_filepath)
                    continue

                reaction = reaction_record(reaction_smiles, self.SUBSTRUCTURE_KEYS)

                if self.pre_filter is not None and not self.pre_filter(reaction):
                    continue

                if self.pre_transform is not None:
                    reaction = self.pre_transform(reaction)

                torch.save(reaction, processed_filepath)
                self.processed_filepaths.append(processed_filepath)

    def len(self):
        """Get length of reaction dataset."""
        return len(self.processed_filepaths)

    def get(self, idx):
        """Get data point for given reaction-idx."""
        successful = False
        while not successful:
            try:
                processed_filepath = self.processed_filepaths[idx]
                reaction_data = torch.load(processed_filepath) # load graph
                successful = True
            except: # Try another index
                idx = random.randint(0, len(self.processed_filepaths) - 1)

        # load substruct-pair selectors and target
        selector_i, selector_j, target = \
            reaction_data.sample_selector_and_target(self.sample_pos_fraction)

        # attach selectors and target to Data object
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        reaction_data.pyg_data.selector_i = torch.tensor(selector_i)
        reaction_data.pyg_data.selector_j = torch.tensor(selector_j)
        reaction_data.pyg_data.target = target

        return reaction_data.pyg_data

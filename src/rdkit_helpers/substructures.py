"""Python file with substructure based utilities."""

from rdkit import Chem

from rdkit_helpers.substructure_keys import patterns_dict
from rdkit_helpers.generic import get_atom_maps

def get_substruct_matches(mol, SUBSTRUCTURE_KEYS = 'MACCS_FULL'):
    """
    Get tuples of atom indices corresponding to substructure matches.

    Args:
        mol (RDKit Mol): RDKit Mol object to get substructure matches for
        SUBSTRUCTURE_KEYS (str, optional):
            Chemical substructure keys to use. Defaults to 'MACCS_FULL'.

    Returns:
        all_atomidx_tuples: List of matching tuples of atom indices
    """

    patterns = patterns_dict[SUBSTRUCTURE_KEYS]['SMARTS']
    keys_to_skip = patterns_dict[SUBSTRUCTURE_KEYS]['KEYS_TO_SKIP']
    atom_id_to_atom_map = get_atom_maps(mol)

    matching_atom_map_tuples = [] # tuples of atom ids for each substructure match
    recon_bonds_per_match = [] # bonds to reconstruct, for each match tuple

    for pattern_key, pattern_smarts in patterns:

        if pattern_key in keys_to_skip:
            continue

        pattern_mol = Chem.MolFromSmarts(pattern_smarts)
        match_tuples_list = mol.GetSubstructMatches(pattern_mol)

        for match_tuple in match_tuples_list:

            matched_atom_ids = set(match_tuple)
            bonds_for_recon_this_match = set()

            for bond in mol.GetBonds():
                atom_id_1 = bond.GetBeginAtomIdx()
                atom_id_2 = bond.GetEndAtomIdx()

                if (atom_id_1 in matched_atom_ids) and (atom_id_2 in matched_atom_ids):
                    continue

                atom_map_1 = atom_id_to_atom_map(atom_id_1)
                atom_map_2 = atom_id_to_atom_map(atom_id_2)
                bond_type = mol.GetBondBetweenAtoms(atom_id_1, atom_id_2).GetBondType()
                bonds_for_recon_this_match.append((atom_map_1, atom_map_2, bond_type))

            atom_map_tuple = tuple(map(atom_id_to_atom_map, match_tuple))
            matching_atom_map_tuples.append(atom_map_tuple)
            recon_bonds_per_match.append(bonds_for_recon_this_match)

    substruct_matches = {
        'matches': matching_atom_map_tuples,
        'bonds': recon_bonds_per_match
    }
    return substruct_matches

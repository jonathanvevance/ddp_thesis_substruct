"""Python file with substructure based utilities."""

from rdkit import Chem

from substructures.substructure_keys import patterns_dict

def get_atomid_matches_and_bonds(mol, SUBSTRUCTURE_KEYS = 'MACCS_FULL'):
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

    all_atomidx_tuples = [] # tuples of atom ids for each substructure match
    all_bonds_for_recon_sets = [] # bonds to reconstruct, for each match tuple

    for pattern_key, pattern_smarts in patterns:

        if pattern_key in keys_to_skip:
            continue

        pattern_mol = Chem.MolFromSmarts(pattern_smarts)
        match_tuples_list = mol.GetSubstructMatches(pattern_mol)

        for match_tuple in match_tuples_list:

            matched_atom_ids = set(match_tuple)
            bonds_for_recon_this_match = set()

            for match_atomid in match_tuple:
                match_atom = mol.GetAtomWithIdx(match_atomid)

                for nbr_atom in match_atom.GetNeighbors():

                    if nbr_atom.GetIdx() in matched_atom_ids:
                        continue

                    bond_type = mol.GetBondBetweenAtoms(nbr_atom.GetIdx(), match_atomid).GetBondType()
                    bond_already_added_case_1 = (nbr_atom.GetAtomMapNum(), match_atom.GetAtomMapNum(), bond_type) in bonds_for_recon_this_match
                    bond_already_added_case_2 = (match_atom.GetAtomMapNum(), nbr_atom.GetAtomMapNum(), bond_type) in bonds_for_recon_this_match
                    if bond_already_added_case_1 or bond_already_added_case_2:
                        continue

                    bonds_for_recon_this_match.add((nbr_atom.GetAtomMapNum(), match_atom.GetAtomMapNum(), bond_type))

            all_atomidx_tuples.append(match_tuple)
            all_bonds_for_recon_sets.append(bonds_for_recon_this_match)

    return all_atomidx_tuples, all_bonds_for_recon_sets

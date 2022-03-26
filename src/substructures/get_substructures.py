"""Python file with substructure based utilities."""

from rdkit import Chem

from substructures.substructure_keys import patterns_dict

def get_matching_atomidx_tuples(mol, SUBSTRUCTURE_KEYS = 'MACCS_FULL'):
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

    all_atomidx_tuples = []

    for pattern_key, pattern_smarts in patterns:

        if pattern_key in keys_to_skip:
            continue

        pattern_mol = Chem.MolFromSmarts(pattern_smarts)

        match_tuples_list = mol.GetSubstructMatches(pattern_mol)
        all_atomidx_tuples.extend(match_tuples_list)

    return all_atomidx_tuples


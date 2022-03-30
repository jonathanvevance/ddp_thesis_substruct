"""Python file with generic RDKit helpers."""

def get_atom_maps(mol):
    """Convert atom id to atom map."""
    def atom_id_to_atom_map(idx):
        return mol.GetAtomWithIdx(idx).GetAtomMapNum()
    return atom_id_to_atom_map

def get_map_to_id_dict(mol):
    """Return dictionary with atom map to atom id."""
    mapping = dict()
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        atom_map = atom.GetAtomMapNum()
        mapping[atom_map] = atom_id

    return mapping

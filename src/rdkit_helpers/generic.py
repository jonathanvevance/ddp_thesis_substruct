"""Python file with generic RDKit helpers."""

def get_atom_maps(mol):
    """Convert atom id to atom map."""
    def atom_id_to_atom_map(idx):
        return mol.GetAtomWithIdx(idx).GetAtomMapNum()
    return atom_id_to_atom_map

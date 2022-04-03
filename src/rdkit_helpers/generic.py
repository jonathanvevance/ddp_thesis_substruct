"""Python file with generic RDKit helpers."""

from rdkit import Chem

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

def get_map_to_molid_dict(smiles):
    """Return atom map to molid dictionary."""
    map_to_molid_dict = dict()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles.split('.')]

    for molid, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            map_to_molid_dict[atom.GetAtomMapNum()] = molid

    return map_to_molid_dict

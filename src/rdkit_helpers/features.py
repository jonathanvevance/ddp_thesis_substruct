"""Python file with atom features utilities."""

def get_pyg_graph_requirements(mol):

    num_atom_features = 2
    num_bond_features = 2
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    x = [[None for __ in range(num_atom_features)]
               for __ in range(num_atoms)] # (num_atoms x num_atom_features)

    edge_index = [[None for __ in range(num_bonds)]
                        for __ in range(2)] # (2 x num_bonds)

    edge_attr = [[None for __ in range(num_bond_features)]
                       for __ in range(num_bonds)] # (num_bonds x num_bond_features)

    for atom in mol.GetAtoms():
        atom_idx = atom.GetAtomMapNum() - 1
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        x[atom_idx] = [atomic_num, formal_charge]

    for bond_idx, bond in enumerate(mol.GetBonds()):
        atom_begin = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        atom_end = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        edge_index[0][bond_idx] = atom_begin.GetAtomMapNum() - 1
        edge_index[1][bond_idx] = atom_end.GetAtomMapNum() - 1
        edge_attr[bond_idx][0] = int(bond.GetBondType())
        edge_attr[bond_idx][1] = 1 if str(bond.GetBondType()) == 'AROMATIC' else 0

    pyg_requirements = {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
    }
    return pyg_requirements


from rdkit import Chem
import json

def get_functional_group_patterns():
    functional_groups = {
        'Alcohol': '[#6][OX2H]',
        'Aldehyde': '[CX3H1](=O)[#6,H]',
        'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
        'Arene': '[cX3]1[cX3][cX3][cX3][cX3][cX3]1',
        'Ketone': '[#6][CX3](=O)[#6]',
    }
    return functional_groups, len(functional_groups.keys())
def check_functional_groups(smiles, patterns):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    results = []
    for key, pattern in patterns.items():
        pattern_mol = Chem.MolFromSmarts(pattern)
        if pattern_mol is None or not mol.HasSubstructMatch(pattern_mol):
            pass
            #results.append('none')
        else:
            results.append(str(key))

    return results

def get_match_bond_indices(query, mol, match_atom_indices):
    bond_indices = []
    for query_bond in query.GetBonds():
        atom_index1 = match_atom_indices[query_bond.GetBeginAtomIdx()]
        atom_index2 = match_atom_indices[query_bond.GetEndAtomIdx()]
        bond = mol.GetBondBetweenAtoms(atom_index1, atom_index2)
        if bond:
            bond_indices.append(bond.GetIdx())
    return bond_indices
def find_functional_groups_in_smiles(smiles_list, patterns):
    matches = {}
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        matches[smiles] = {}
        for name, pattern in patterns.items():
            pat = Chem.MolFromSmarts(pattern)
            atom_indices_list = mol.GetSubstructMatches(pat)
            for atom_indices in atom_indices_list:
                bond_indices = get_match_bond_indices(pat, mol, atom_indices)
                matches[smiles].setdefault(name, []).append({'atom_indices': atom_indices, 'bond_indices': bond_indices})
    return matches
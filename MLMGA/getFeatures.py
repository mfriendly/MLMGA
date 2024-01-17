
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import gc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from MLMGA.Featurizer import *
from tqdm import tqdm
import multiprocessing
import json
import traceback
import pickle
import matplotlib.pyplot as plt
degrees = [0,1,2,3,4,5]
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
        return [0] * len(
            patterns
        )  
    return [
        int(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)))
        for pattern in patterns.values()
    ]
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
patterns, num_functional_groups = get_functional_group_patterns()
class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node
    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))
    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)
        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes
    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])
    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])
    def neighbor_list(self, self_ntype, neighbor_ntype):
        # # print("self_ntype, neighbor_ntype", self_ntype, neighbor_ntype)
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]
class FGMolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type
    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node
    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))
    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)
        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)
        self.nodes[ntype] = new_nodes
    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])
    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])
    def neighbor_list(self, self_ntype, neighbor_ntype):
        # # print("self_ntype, neighbor_ntype", self_ntype, neighbor_ntype)
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]
class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]
def calculate_individual_angle(mol, atom_idx):
    central_atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [nbr.GetIdx() for nbr in central_atom.GetNeighbors()]
    angles = {}
    if len(neighbors) > 1:
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                angle = rdMolTransforms.GetAngleDeg(mol.GetConformer(), neighbors[i], atom_idx, neighbors[j])
                angles[(neighbors[i], atom_idx, neighbors[j])] = angle
                ### print("(neighbors[i], atom_idx, neighbors[j])")
                ### print((neighbors[i], atom_idx, neighbors[j]))
                #angles[(neighbors[j], atom_idx, neighbors[i])] = angle
        return angles, (neighbors[i], atom_idx, neighbors[j])
    else:
        return None, None
import json
def get_fg_list_and_indices(task_name, smilesList):
    import pickle, os
    gc.collect()
    def get_functional_group_patterns():
        functional_groups = {
            'Alcohol': '[#6][OX2H]',
            'Aldehyde': '[CX3H1](=O)[#6,H]',
            'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
            'Arene': '[cX3]1[cX3][cX3][cX3][cX3][cX3]1',
            'Ketone': '[#6][CX3](=O)[#6]',
        }
        return functional_groups, len(functional_groups.keys())
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
    patterns, num_functional_groups = get_functional_group_patterns()
    fg_list = list(patterns.keys())
    import pickle
    FG_PATH = f'{task_name}_functional_groups_matches.json'
    FGlist_PATH = f'{task_name}_fg_list.pkl'
    if os.path.exists(FG_PATH):
        with open(FG_PATH, 'r') as f:
            fg_matches = json.load(f)
    else:
        fg_matches = find_functional_groups_in_smiles(smilesList, patterns)
        with open(FG_PATH, 'w') as f:
            json.dump(fg_matches, f, indent=4)
    if os.path.exists(FGlist_PATH):
        with open(FGlist_PATH, 'rb') as f:
            fg_list = pickle.load(f)
    else:
        fg_list = list(patterns.keys())
        with open(FGlist_PATH, 'wb') as f:
            pickle.dump(fg_list, f)
    return fg_matches, fg_list
def extract_fg_indices(args, smiles, fg_list):
    fg_atom_indices = []
    fg_bond_indices = []
    
    fg_matches, _ =  get_fg_list_and_indices(args['task_name'], [smiles])
    if smiles in fg_matches:
        # Iterate over each functional group in the list
        for fg in fg_list:
            # If the functional group is in the SMILES string's entry
            if fg in fg_matches[smiles]:
                # Iterate over each match for the functional group
                for match in fg_matches[smiles][fg]:
                    # Accumulate the atom and bond indices for this functional group
                    fg_atom_indices.extend(match["atom_indices"])
                    fg_bond_indices.extend(match["bond_indices"])
    return fg_atom_indices, fg_bond_indices
def graph_from_smiles(smiles, args=None):
    
    
    
    graph = MolGraph()
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    atoms_by_rd_idx = {}
    angles_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom, args["add_atoms"], args["explicit_H"], args["use_chirality"]), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
        atom_idx = atom.GetIdx()
        
        angles, _ = calculate_individual_angle(mol, atom_idx)
        if angles:
            import itertools
            for angle_pair, angle in angles.items():
                
                
                
                if angle_pair is not None:
                    
                        
                            angle_node = graph.new_node('angle', features=[angle])
                            #left_atom = mol.GetAtomWithIdx(angle_pair[0])
                            #right_atom = mol.GetAtomWithIdx( angle_pair[-1])
                            #new_atom_node_left = graph.new_node('atom', features=atom_features(left_atom, args["add_atoms"], args["explicit_H"], args["use_chirality"]), rdkit_ix=left_atom.GetIdx())
                            #new_atom_node_right = graph.new_node('atom', features=atom_features(right_atom, args["add_atoms"], args
                            ### print("➡ angles_by_rd_idx :", angles_by_rd_idx)
                            new_atom_node.add_neighbors([angle_node])
                            #new_atom_node_left.add_neighbors([angle_node])
                            #new_atom_node_right.add_neighbors([angle_node])
                            angles_by_rd_idx[angle_pair] = angle_node #순서 변경
                            #atoms_by_rd_idx[angle_pair[0]] = new_atom_node_left
                            #atoms_by_rd_idx[angle_pair[-1]] = new_atom_node_right
                    ### print("➡ atoms_by_rd_idx :", atoms_by_rd_idx)
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtom().GetIdx()
        atom2_idx = bond.GetEndAtom().GetIdx()
        atom1_node = atoms_by_rd_idx[atom1_idx]
        atom2_node = atoms_by_rd_idx[atom2_idx]
        new_bond_node = graph.new_node('bond', features=bond_features(bond, mol, args["add_bond_length"], args["use_chirality"]))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors([node for node in graph.nodes['atom']])
    import pickle, json
    fg_list = args['fg_list']
    
    fg_dict = {}
    for fg in fg_list:
        
        fg_atom_indices, _ = extract_fg_indices(args, smiles, [fg])
        fg_atoms = [mol.GetAtomWithIdx(idx) for idx in fg_atom_indices]
        fg_atom_nodes = [atoms_by_rd_idx[atom.GetIdx()]  for atom in fg_atoms]
        
        fg_mol_node = graph.new_node(fg)
        fg_mol_node.add_neighbors(list(fg_atom_nodes ))
        fg_dict[fg] = fg_mol_node
        
        mol_node.add_neighbors([fg_mol_node])
    
    return graph, smiles, mol, atoms_by_rd_idx, mol_node
def fg_graph_from_smiles(fg, smiles, mol, atoms_by_rd_idx, molgraph, mol_node, args=None):
    graph = FGMolGraph()
    
    fg_dict = {}
    
    fg_atom_indices, _ = extract_fg_indices(args, smiles, [fg])
    
    fg_atoms = [mol.GetAtomWithIdx(idx) for idx in fg_atom_indices]
    
    if fg_atoms ==None or len(fg_atoms) == 0:
        
        return None
    else:
        fg_atom_nodes = [atoms_by_rd_idx[atom.GetIdx()]  for atom in fg_atoms]
        fg_mol_node = graph.new_node(fg)
        fg_mol_node.add_neighbors(list(fg_atom_nodes))
        fg_dict[fg] = fg_mol_node
        
        mol_node.add_neighbors([fg_mol_node])
        
        atoms_by_rd_idx = {}
        angles_by_rd_idx = {}
        for atom in fg_atoms:
            
            new_atom_node = graph.new_node('atom', features=atom_features(atom, args["add_atoms"], args["explicit_H"], args["use_chirality"]), rdkit_ix=atom.GetIdx())
            atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
            atom_idx = atom.GetIdx()
            
            angles, _ = calculate_individual_angle(mol, atom_idx)
            if angles:
                import itertools
                for angle_pair, angle in angles.items():
                    
                    
                    
                    if angle_pair is not None:
                        
                            
                                angle_node = graph.new_node('angle', features=[angle])
                                new_atom_node.add_neighbors([angle_node])
                                angles_by_rd_idx[angle_pair] = angle_node 
                                
                                
                        
        for bond in mol.GetBonds():
            try:
                atom1_idx = bond.GetBeginAtom().GetIdx()
                atom2_idx = bond.GetEndAtom().GetIdx()
                atom1_node = atoms_by_rd_idx[atom1_idx]
                atom2_node = atoms_by_rd_idx[atom2_idx]
                new_bond_node = graph.new_node('bond', features=bond_features(bond, mol, args["add_bond_length"], args["use_chirality"]))
                new_bond_node.add_neighbors((atom1_node, atom2_node))
                atom1_node.add_neighbors((atom2_node,))
            except Exception as e:
                
                pass#traceback.print_exc()
        fg_node = graph.new_node('fg')
        fg_node.add_neighbors([node for node in graph.nodes['atom']])
        import pickle, json
        return graph
import numpy as np
def pad_angle_neighbors_data(angle_neighbors_data, min_length=1):
    # 빈 리스트를 제외한 리스트들의 최대 길이를 계산
    non_empty_items = [item for item in angle_neighbors_data if item]
    max_length = max(len(item) for item in non_empty_items) if non_empty_items else min_length
    # 최소 길이보다 작은 경우 최소 길이로 설정
    if max_length < min_length:
        max_length = min_length
    # 패딩된 데이터를 저장할 빈 NumPy 배열 생성
    padded_angle_neighbors_data = np.zeros((len(angle_neighbors_data), max_length), dtype=int)
    # 각 리스트를 패딩하여 저장
    for i, item in enumerate(angle_neighbors_data):
        padded_angle_neighbors_data[i, :len(item)] = item
    return padded_angle_neighbors_data
def array_rep_from_smiles(molgraph):
    arrayrep = {'atom_features': molgraph.feature_array('atom'),
                'bond_features': molgraph.feature_array('bond'),
                'angle_features': molgraph.feature_array('angle'),
                'atom_list': molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix': molgraph.rdkit_ix_array()}
    for degree in degrees:
        atom_neighbors_data = molgraph.neighbor_list(('atom', degree), 'atom')
        bond_neighbors_data = molgraph.neighbor_list(('atom', degree), 'bond')
        angle_neighbors_data = molgraph.neighbor_list(('atom', degree), 'angle')
        # print("angle_neighbors_data", angle_neighbors_data)
        #np.array(atom_neighbors_data, dtype=int)
        # print("np.array(atom_neighbors_data, dtype=int)", np.array(atom_neighbors_data, dtype=int))
        arrayrep[('atom_neighbors', degree)] = np.array(atom_neighbors_data, dtype=int)
        arrayrep[('bond_neighbors', degree)] = np.array(bond_neighbors_data, dtype=int)
        arrayrep[('angle_neighbors', degree)] = np.array(angle_neighbors_data, dtype=int)
        
    return arrayrep
def array_rep_from_smiles_fg(fg, fg_graph):
    
    
    bond_feature = False
    angle_feature =False
    arrayrep = {'atom_features': fg_graph.feature_array('atom')}
    
    try:
        arrayrep['bond_features'] = fg_graph.feature_array('bond')
        bond_feature = True
    except Exception as e:
        pass
        #traceback.print_exc()
    try:
        #if 'angle' in fg_graph.feature_array:
        arrayrep['angle_features'] = fg_graph.feature_array('angle')
        angle_feature = True
    except Exception as e:
        pass
        #traceback.print_exc()
    arrayrep['atom_list'] = fg_graph.neighbor_list('fg', 'atom')
    # print("arrayrep['atom_list']", arrayrep['atom_list'])
    ## print("fg_graph.neighbor_list('fg', 'atom')", fg_graph.neighbor_list('fg', 'atom'))
    #jjj
    arrayrep['rdkit_ix'] = fg_graph.rdkit_ix_array()
    for degree in degrees:
        atom_neighbors_data = fg_graph.neighbor_list(('atom', degree), 'atom')
        # print("atom_neighbors_data", atom_neighbors_data)
        #np.array(atom_neighbors_data, dtype=int)
        # print("np.array(atom_neighbors_data, dtype=int)", np.array(atom_neighbors_data, dtype=int))
        arrayrep[('atom_neighbors', degree)] = np.array(atom_neighbors_data, dtype=int)
        if bond_feature:
            bond_neighbors_data = fg_graph.neighbor_list(('atom', degree), 'bond')
            arrayrep[('bond_neighbors', degree)] = np.array(bond_neighbors_data, dtype=int)
        if angle_feature:
            angle_neighbors_data = fg_graph.neighbor_list(('atom', degree), 'angle')
            if not angle_neighbors_data or all(len(sublist) == 0 for sublist in angle_neighbors_data):
                # Handle the empty sequence case, perhaps by continuing to the next iteration or logging an error
                ## print(f"Empty neighbor data for {angle} at degree {degree}.")
                continue  # Skip this iteration
            # If angle_neighbors_data is not empty, proceed with padding
            max_length = max(len(sublist) for sublist in angle_neighbors_data)
            angle_neighbors_data_padded = [sublist + [-1] * (max_length - len(sublist)) for sublist in angle_neighbors_data]
            arrayrep[(f'{fg}_neighbors', degree)] = np.array(angle_neighbors_data_padded, dtype=int)
            # print("arrayrep", arrayrep)
            #angle_neighbors_data = fg_graph.neighbor_list(('atom', degree), 'angle')
            #arrayrep[('angle_neighbors', degree)] = np.array(angle_neighbors_data, dtype=int)
            ## print("angle_neighbors_data", angle_neighbors_data)
        ## print("arrayrep", arrayrep)
    return arrayrep
def array_rep_from_smiles_sorted_fg_degree(molgraph, arrayrep, fg):
    arrayrep[fg + '_atom_list'] = molgraph.neighbor_list(fg, 'atom')
    for degree in degrees:
        fg_neighbors_data = molgraph.neighbor_list(('atom', degree), fg)
        if not fg_neighbors_data or all(len(sublist) == 0 for sublist in fg_neighbors_data):
            continue  #
        # If fg_neighbors_data is not empty, proceed with padding
        max_length = max(len(sublist) for sublist in fg_neighbors_data)
        fg_neighbors_data_padded = [sublist + [-1] * (max_length - len(sublist)) for sublist in fg_neighbors_data]
        arrayrep[(f'{fg}_neighbors', degree)] = np.array(fg_neighbors_data_padded, dtype=int)
        # print("arrayrep", arrayrep)
    return arrayrep
def process_smiles(smiles, args, fg_list):
    fgs = []
    smiles_processed = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    arrays = {}
    try:
        mol = Chem.MolFromSmiles(smiles_processed)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        molgraph, smiles, mol, atoms_by_rd_idx, molnode = graph_from_smiles(smiles, args)
        fg_d3 = {}
        for fg in fg_list:
            fg_graph = fg_graph_from_smiles(fg, smiles, mol, atoms_by_rd_idx, molgraph, molnode, args)
            if fg_graph is not None:
                fgs.append(fg)
                fg_d3[fg] = fg_graph
        molgraph.sort_nodes_by_degree('atom')
        arrays['mol'] = array_rep_from_smiles(molgraph)
        for fg in fg_list:
            fg_graph = fg_d3[fg]
            if fg_graph is not None:
                fg_graph.sort_nodes_by_degree('atom')
                arrays[fg] = array_rep_from_smiles_fg(fg, fg_graph)
    except Exception as e:
        pass
        #traceback.print_exc()
    return smiles, arrays, fgs

def gen_descriptor_data(smilesList, args):
    smiles_to_fingerprint_array = {}
    fgs={}
    for i, smiles in enumerate(tqdm(smilesList, total=len(smilesList), desc=f"gen_descriptor_data - ({args['task_name']}) - Processing SMILES")):
        # print("smiles", smiles)
        fgs[smiles] = []
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        #try:
        arrays={}
        try:
            # SMILES로부터 RDKit 분자를 생성하고 3D 좌표를 생성
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            # 분자 그래프 및 배열 표현 생성
            molgraph , smiles, mol, atoms_by_rd_idx, molnode= graph_from_smiles(smiles, args)
            fg_d3 = {}
            fg_list = args['fg_list']
            for fg in fg_list:
                
                fg_graph = fg_graph_from_smiles(fg, smiles, mol, atoms_by_rd_idx, molgraph, molnode, args)
                if fg_graph is not None:
                    fgs[smiles].append(fg)
                    
                fg_d3[fg] = fg_graph
            molgraph.sort_nodes_by_degree('atom')
            arrays['mol'] = array_rep_from_smiles(molgraph)
            for fg in fg_list:
                fg_graph = fg_d3[fg]
                if fg_graph is not None:
                    fg_graph.sort_nodes_by_degree('atom')
                    arrays[fg] =  array_rep_from_smiles_fg(fg, fg_graph)
                    
                    import time;time.sleep(0)
            smiles_to_fingerprint_array[smiles] = arrays
            
        except Exception as e:
            
            
            print(f"Error processing {smiles}: {e}")
    with open("f{args['task_name]}_smiles_to_fg_list.json", "w") as f:
        json.dump(fgs, f, indent=4)
    
    return smiles_to_fingerprint_array, fgs
def save_smiles_dicts(smilesList,filename, args):
    
    max_atom_len = 0
    max_bond_len = 0
    max_angle_len = 0
    num_atom_features = 0
    num_bond_features = 0
    num_angle_features = 0
    smiles_to_rdkit_list = {}
    smiles_to_fingerprint_features, _ = gen_descriptor_data(smilesList, args)
    length = len(smiles_to_fingerprint_features)
    
    out_dict={}
    smiles_to_atom_info = {}
    smiles_to_bond_info = {}
    smiles_to_angle_info =  {}
    smiles_to_atom_neighbors =  {}
    smiles_to_bond_neighbors =  {}
    smiles_to_angle_neighbors =  {}
    smiles_to_atom_mask =  {}
    for smiles, val in tqdm(smiles_to_fingerprint_features.items(), total=length,desc=f"0st loop -{args['task_name']}"):
        out_dict[smiles] = {}
        smiles_to_atom_info[smiles] = {}
        smiles_to_bond_info[smiles] = {}
        smiles_to_angle_info[smiles] =  {}
        smiles_to_atom_neighbors[smiles] =  {}
        smiles_to_bond_neighbors[smiles] =  {}
        smiles_to_angle_neighbors[smiles] =  {}
        smiles_to_atom_mask[smiles] =  {}
        max_atom_index_num = max_atom_len
        max_bond_index_num = max_bond_len
        max_angle_index_num = max_angle_len
        max_atom_len += 1
        max_bond_len += 1
        max_angle_len += 1
        for key, arrayrep in tqdm(val.items(), desc=f"1st loop ({args['task_name']})"):
            out_dict[smiles][key] = None
            import time;time.sleep(0)
            try:
                atom_features = arrayrep['atom_features']
                
            except KeyError:
                atom_features = np.empty((100,85))
            try:
                bond_features = arrayrep['bond_features']
            except KeyError:
                default_bond_count = 100  
                num_bond_features = 11   
                bond_features = np.zeros((default_bond_count, num_bond_features))
            try:
                angle_features = arrayrep.get('angle_features', np.empty((0,)))  
                
            except KeyError:
                default_angle_count = 100
                num_angle_features = 1  
                angle_features = np.zeros((default_angle_count, num_angle_features))
            if angle_features is not None:
                angle_len = angle_features.shape[0]
                num_angle_features = np.prod(angle_features.shape[1:])
            rdkit_list = arrayrep['rdkit_ix']
            smiles_to_rdkit_list[smiles] = rdkit_list
            atom_len,num_atom_features = atom_features.shape
            bond_len,num_bond_features = bond_features.shape
            angle_len,num_angle_features = angle_features.shape
            if atom_len > max_atom_len:
                max_atom_len = atom_len
            if bond_len > max_bond_len:
                max_bond_len = bond_len
            if angle_len > max_angle_len:
                max_angle_len = angle_len
        length = len(smiles_to_fingerprint_features)
        for key, arrayrep in tqdm(val.items(), total=length,desc=f"2nd loop({args['task_name']}) "):
            mask = np.zeros((max_atom_len))
            atoms = np.zeros((max_atom_len,num_atom_features))
            bonds = np.zeros((max_bond_len,num_bond_features))
            angles = np.zeros((max_angle_len,num_angle_features))
            atom_neighbors = np.zeros((max_atom_len,len(degrees)))
            bond_neighbors = np.zeros((max_atom_len,len(degrees)))
            angle_neighbors = np.zeros((max_atom_len,len(degrees)))
            atom_neighbors.fill(max_atom_index_num)
            bond_neighbors.fill(max_bond_index_num)
            angle_neighbors.fill(max_angle_index_num)
            for i,feature in enumerate(atom_features):
                
                atoms[i] = feature
            for j,feature in enumerate(bond_features):
                bonds[j] = feature
            for k, feature in enumerate(angle_features):
                angles[k] = feature
            atom_neighbor_count = 0
            bond_neighbor_count = 0
            angle_neighbor_count = 0
            for degree in degrees:
                try:
                    atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
                    if len(atom_neighbors_list) > 0:
                        for i,degree_array in enumerate(atom_neighbors_list):
                            for j,value in enumerate(degree_array):
                                atom_neighbors[atom_neighbor_count,j] = value
                            atom_neighbor_count += 1
                except:
                    pass
                try:
                    bond_neighbors_list = arrayrep[('bond_neighbors', degree)]
                    if len(bond_neighbors_list) > 0:
                        for i,degree_array in enumerate(bond_neighbors_list):
                            for j,value in enumerate(degree_array):
                                bond_neighbors[bond_neighbor_count,j] = value
                            bond_neighbor_count += 1
                except:
                    pass
                try:
                    angle_neighbors_list = arrayrep[('angle_neighbors', degree)]
                    if len(angle_neighbors_list) > 0:
                        for i,degree_array in enumerate(angle_neighbors_list):
                            for j,value in enumerate(degree_array):
                                angle_neighbors[angle_neighbor_count,j] = value
                            angle_neighbor_count += 1
                except:
                    pass
            smiles_to_atom_info[smiles][key] = atoms
            smiles_to_atom_neighbors[smiles][key] = atom_neighbors
            smiles_to_bond_info[smiles][key] = bonds
            smiles_to_bond_neighbors[smiles][key] = bond_neighbors
            smiles_to_angle_info[smiles][key] = angles
            smiles_to_angle_neighbors[smiles][key] = angle_neighbors
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_angle_info': smiles_to_angle_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_angle_neighbors': smiles_to_angle_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list,
        
    }
    
    pickle.dump(feature_dicts,open(filename+'.pickle',"wb"))
    return feature_dicts
import json
import numpy as np
def get_smiles_array(smiles_list, feature_dicts, args):
    
    fg_list = args['fg_list']
    
    keys = fg_list + ['mol']
    
    alld = {}
    for smiles in smiles_list:
        alld[smiles] = {}
        alld[smiles]['x_mask'] = {}
        fgd,fgd2,fgd3 = {},{},{}
        d = {key + f'_{x}_neighbors': [] for key in keys for x in ['atom', 'bond', 'angle']}
        d2 = {key + f'_{x}_features': [] for key in keys for x in ['atom', 'bond', 'angle']}
        d3 = {key + f'_{x}_masks': [] for key in keys for x in ['atom']}
        alld[smiles]['x_neighbor'] = {}
        alld[smiles]['x_info']  = {}
        for fg in keys:
            for x in ['atom', 'bond', 'angle']:
                fgd = feature_dicts[f'smiles_to_{x}_neighbors'].get(smiles, {}).get(fg, None)
                d[fg + f'_{x}_neighbors'] = fgd
                alld[smiles]['x_neighbor']  = d
                fgd2 = feature_dicts[f'smiles_to_{x}_info'].get(smiles, {}).get(fg, None)
                d2[fg + f'_{x}_features'] = fgd2
                alld[smiles]['x_info']  = d2
                if x =='atom':
                    fgd3 = feature_dicts[f'smiles_to_{x}_mask'].get(smiles, {}).get(fg, None)
                    d3[fg + f'_{x}_masks'] = fgd3
                    alld[smiles]['x_mask']  = d3
    outs = []
    outs_name = []
    nei = []
    nei_name = []
    masks = []
    masks_name = []
    
    if True:
        for smiles in smiles_list:
            for k,v in alld[smiles].items():
                
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if 'neighbor' in k and isinstance(v, dict):
                            
                            
                            nei.append(vv)
                            nei_name.append(kk)
                        if 'info' in k and isinstance(v, dict):
                            outs.append(vv)
                            outs_name.append(kk)
                        if 'mask' in k and isinstance(v, dict):
                            masks.append(vv)
                            masks_name.append(kk)
    import torch
    import numpy as np
    device = 'cuda'
    atom_default_shape_feats = (100,85)
    atom_default_shape_neighbors = (100,6 )
    bond_default_shape_feats = (100,11)
    bond_default_shape_neighbors = (100,6 )
    angle_default_shape_feats = (100,1)
    angle_default_shape_neighbors = (100,6 )
    feats = {str(k): (torch.tensor(np.array(v, dtype=np.float32), device=device) if v is not None else torch.zeros(atom_default_shape_feats if 'atom' in k else bond_default_shape_feats if 'bond' in k else angle_default_shape_feats, dtype=torch.float32, device=device)) for k, v in zip(outs_name, outs)}
    neighbors = {str(k): (torch.tensor(np.array(v, dtype=np.float32), device=device) if v is not None else torch.zeros(atom_default_shape_neighbors if 'atom' in k else bond_default_shape_neighbors if 'bond' in k else angle_default_shape_neighbors, dtype=torch.float32, device=device)) for k, v in zip(nei_name, nei)}
    return smiles_list, feats, neighbors
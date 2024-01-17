import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return x + y
class LinearGLU(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, 2*C, 1, 1)
        self.bn = nn.BatchNorm1d(2*C)
        self.dropout = nn.Dropout(args.drpt)
    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)

        out = F.glu(out, dim=1)
        out = self.dropout(out)
        return out
class ConcatFC_simple(nn.Module):
    def __init__(self, C):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Linear(2*C, C)#, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=-1)
        out = self.conv(out)
        out = self.dropout(out)
        return out
class ConcatFC(nn.Module):
    def __init__(self, C):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Linear(3*C, C)#, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, y,z):
        # concat on channels
        out = torch.cat([x, y, z], dim=-1)
        out = self.conv(out)

        #out = F.relu(out)
        out = self.dropout(out)
        return out

class AngleEncoderV1(nn.Module):
    def __init__(self, embedding_dim):
        super(AngleEncoderV1, self).__init__()
        self.embedding = nn.Embedding(360, embedding_dim)
        self.lin = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, angles):
       #radians = angles * math.pi / 180
       # sin_encoding = torch.sin(radians)
       # cos_encoding = torch.cos(radians)
        indices = angles.long()
        embedding_encoding = self.embedding(indices)
        embedding_encoding = self.lin(embedding_encoding)
        embedding_encoding = self.relu(embedding_encoding)
        return _,_,_,embedding_encoding

class AngleEncoderV2(nn.Module):
    def __init__(self, input_dim):
        super(AngleEncoderV2, self).__init__()
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(input_dim, input_dim)

    def forward(self, angles):
        radians = angles * math.pi / 180
        sin_encoding = torch.sin(radians)
        cos_encoding = torch.cos(radians)
        angle_transform = torch.stack([sin_encoding, cos_encoding], dim=-1)
        angle_transform = self.lin1(angle_transform)
        angle_transform = self.relu(angle_transform)
        embedding_encoding = self.lin2(angle_transform)
        return radians, sin_encoding, cos_encoding, embedding_encoding

class AngleEncoderV3(nn.Module):
    def __init__(self, embedding_dim):
        super(AngleEncoderV3, self).__init__()
        self.lin = nn.Linear(2, embedding_dim)

    def forward(self, angles):
        radians = angles * math.pi / 180
        sin_encoding = torch.sin(radians)
        cos_encoding = torch.cos(radians)
        combined_encoding = torch.cat([sin_encoding.unsqueeze(-1), cos_encoding.unsqueeze(-1)], dim=-1)
        embedding_encoding = self.lin(combined_encoding)
        return radians, sin_encoding, cos_encoding, embedding_encoding


class AngleEncoderV4(nn.Module):
    def __init__(self, embedding_dim):
        super(AngleEncoderV4, self).__init__()
        self.lin = nn.Linear(2, embedding_dim)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, angles):
        radians = angles * math.pi / 180
        sin_encoding = torch.sin(radians)
        cos_encoding = torch.cos(radians)
        combined_encoding = torch.cat([sin_encoding.unsqueeze(-1), cos_encoding.unsqueeze(-1)], dim=-1)
        embedding_encoding = self.lin(combined_encoding)
        s = embedding_encoding.shape
       # ## print('s: ', s)
        embedding_encoding = embedding_encoding.squeeze(2)#vi
        embedding_encoding = embedding_encoding.permute(0,2,1)
        embedding_encoding = self.batch_norm(embedding_encoding)
        embedding_encoding = embedding_encoding.permute(0,2,1)
        embedding_encoding = embedding_encoding.view(s[0], s[1], s[2], s[3])
        return radians, sin_encoding, cos_encoding, embedding_encoding

class AngleEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AngleEncoder, self).__init__()
        self.embedding = nn.Embedding(360, embedding_dim)  # 360
        self.lin = nn.Linear(10,1)
    def forward(self, angles):
        #
        radians = angles * math.pi / 180
        #
        sin_encoding = torch.sin(radians)
        cos_encoding = torch.cos(radians)
        indices = angles.long()  #
        embedding_encoding = self.embedding(indices)
        #embedding_encoding = self.lin(embedding_encoding).squeeze(-1) #
        return radians, sin_encoding, cos_encoding, embedding_encoding
import json
def get_class_instance_by_name(class_name, *args, **kwargs):

    cls = globals().get(class_name)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found")
    return cls(*args, **kwargs)

class AngleCategoricalEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AngleCategoricalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.defined_angles = torch.tensor([90, 109.5, 120, 180, 0], dtype=torch.float32)
        self.tolerance = 5.0
        self.embedding = nn.Embedding(len(self.defined_angles), embedding_dim)

    def find_closest_angle_indices(self, angles):
        # Expand dimensions for broadcasting
        angles = angles.unsqueeze(-1)  # Shape: [batch, n_angles, 1]
        defined_angles = self.defined_angles.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, n_defined_angles]

        # Calculate differences and check tolerance
        differences = torch.abs(angles - defined_angles)  # Shape: [batch, n_angles, n_defined_angles]
        within_tolerance = differences <= self.tolerance

        # Find indices of the closest angles
        _, indices = torch.max(within_tolerance, dim=-1)

        return indices

    def forward(self, angles):
        # ## print("angles", angles)
        indices = self.find_closest_angle_indices(angles)
        return self.embedding(indices)
def return_inputs(feats, neighbors, fg):

    fg_feats = {k:v for k,v in feats.items() if fg in k}
    fg_neighbors =  {k:v for k,v in neighbors.items() if fg in k}

    #fg_ns = list(fg_feats.keys()) +list(fg_neighbors.keys())

    inputs = list(fg_feats.values()) +list(fg_neighbors.values())

    return inputs
class Fingerprint(nn.Module):
    __slots__ = (
        'task_name', 'fg_list', 'args', 'batch_size', 'use_functional_groups',
        'input_atom_dim', 'input_bond_dim', 'input_angle_dim',
        'neighbor_fc', 'neighbor_fc_angle', 'neighbor_atom_dim',
        'neighbor_bond_dim', 'neighbor_angle_dim', 'atom_fc', 'angle_fc',
        'bond_fc', 'GRUCell', 'align', 'attend', 'mol_GRUCell', 'mol_align',
        'mol_align_integ', 'mol_attend', 'mol_attend_integ', 'dropout',
        'output1', 'radius', 'T', 'hidden_dim', 'query_A', 'key_A', 'value_A',
        'query_An', 'key_An', 'value_An', 'query_B', 'key_B', 'value_B',
        'feature_attention', 'functional_group_fc', 'attention', 'concat',
        'concat_dual', 'atom_bn', 'bond_bn', 'angle_bn', 'angle_encoder1', 'lin1'
    )

    def __init__(self, args,
                    radius,
                    T,
                    functional_group_dim,
                    num_atom_features,
                    num_bond_features,
                    num_angle_features,
                    num_atom_neighbors,
                    num_bond_neighbors,
                    num_angle_neighbors,
                    fingerprint_dim,
                    output_units_num,
                    p_dropout, num_atom_node
                ):
        super(Fingerprint, self).__init__()
        #self.modules_dict = nn.ModuleDict()

        """for idx, module_params in enumerate(modules_list):
            if module_params["type"] == "linear":
                # Create a linear layer
                self.modules_dict[f"module{idx}"] = nn.Linear(module_params["in_features"], module_params["out_features"])"""

        self.task_name = args['task_name']
        self.fg_list = args['fg_list']

        self.args = args
        self.batch_size = self.args['BATCHSIZE']
        if self.args["FUNCTIONALGROUP"]:
            self.use_functional_groups= True
        # graph attention for atom embedding
        self.input_atom_dim = num_atom_features
        self.input_bond_dim = num_bond_features
        self.input_angle_dim = num_angle_features
        self.neighbor_fc = nn.Linear(3* fingerprint_dim, fingerprint_dim)#self.input_atom_dim +self.input_bond_dim
        self.neighbor_fc_angle = nn.Linear(3* fingerprint_dim, fingerprint_dim)#self.input_atom_dim +self.input_bond_dim
        self.neighbor_atom_dim = num_atom_neighbors
        self.neighbor_bond_dim = num_bond_neighbors
        self.neighbor_angle_dim = num_angle_neighbors
        self.atom_fc = nn.Linear(self.input_atom_dim, fingerprint_dim)
        self.angle_fc = nn.Linear(self.input_angle_dim, fingerprint_dim)
        self.bond_fc = nn.Linear(self.input_bond_dim, fingerprint_dim)
        #self.neighbor_fc = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for _ in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim, 1) for _ in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for _ in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim, 1)
        self.mol_align_integ = nn.Linear(2*fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.mol_attend_integ = nn.Linear(fingerprint_dim*2, fingerprint_dim*2)
        self.dropout = nn.Dropout(p=p_dropout)
        #self.output1 = nn.Linear(fingerprint_dim, 2)
        #self.output1 = nn.Linear(fingerprint_dim, fingerprint_dim)
        output_units_num
        # # print('output_units_num: ', output_units_num)
        self.output1 = nn.Linear(fingerprint_dim, output_units_num)
        self.radius = radius
        self.T = T
        hidden_dim = fingerprint_dim
        self.hidden_dim = fingerprint_dim
        self.query_A = nn.Linear(self.input_atom_dim, hidden_dim)
        self.key_A = nn.Linear(self.input_atom_dim, hidden_dim)
        self.value_A = nn.Linear(self.input_atom_dim, hidden_dim)
        self.query_An = nn.Linear(self.input_angle_dim, hidden_dim)
        self.key_An = nn.Linear(self.input_angle_dim, hidden_dim)
        self.value_An = nn.Linear(self.input_angle_dim, hidden_dim)
        self.query_B = nn.Linear(self.input_bond_dim, hidden_dim)
        ### print("self.input_bond_dim", self.input_bond_dim)
        self.key_B = nn.Linear(self.input_bond_dim, hidden_dim)
        self.value_B = nn.Linear(self.input_bond_dim, hidden_dim)


        self.feature_attention = nn.Linear(3 * fingerprint_dim, 3)
        if self.use_functional_groups:
            if functional_group_dim is None:
                raise ValueError("functional_group_dim must be specified when use_functional_groups is True")
            self.functional_group_fc = nn.Linear(functional_group_dim, fingerprint_dim)
        #self.feature_type_align = nn.Linear(3 * fingerprint_dim, 3)
        #self.feature_type_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        self.attention = nn.Parameter(torch.Tensor(3, 1))
        self.concat = ConcatFC(fingerprint_dim)
        self.concat_dual = ConcatFC_simple(fingerprint_dim)
        self.atom_bn = nn.BatchNorm1d(self.input_atom_dim)
        self.bond_bn = nn.BatchNorm1d(self.input_bond_dim)
        self.angle_bn = nn.BatchNorm1d(self.input_angle_dim)
        embedding_dim = 10

        self.angle_encoder1 = AngleCategoricalEncoder(embedding_dim)
        self.lin1 = nn.Linear(embedding_dim,1)
    def self_attention(self, features, query, key, value, hidden_dim, masks=None):

        Q = query(features)
        K = key(features)
        V = value(features)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return attention_output

    def first_GAT(self, feats, neighbors, fg, radius=1):

        atom_feats, bond_feats, angle_feats, atom_degree_list, bond_degree_list, angle_degree_list = return_inputs(feats, neighbors, fg)

        batch_size, mol_length, num_atom_feat = atom_feats.size()

        bond_neighbor = [bond_feats[i][bond_degree_list[i].long()] for i in range(batch_size)]

        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        bond_neighbor = self.dropout(bond_neighbor)

        bond_neighbor = self.self_attention(bond_neighbor, self.query_B, self.key_B, self.value_B, self.input_bond_dim)#,
        atom_neighbor = [atom_feats[i][atom_degree_list[i].long()] for i in range(batch_size)]

        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        atom_neighbor = self.self_attention(atom_neighbor, self.query_A, self.key_A, self.value_A, self.input_atom_dim)
        angle_neighbor = [angle_feats[i][angle_degree_list[i].long()] for i in range(batch_size)]
        angle_neighbor = torch.stack(angle_neighbor, dim=0)
        angle_neighbor = self.self_attention(angle_neighbor, self.query_An, self.key_An, self.value_An, self.input_angle_dim)#,
        angle_feats = self.angle_encoder1(angle_feats)
        # ## print(f"==>> angle_feats.shape: {angle_feats.shape}")#> angle_feats.shape: torch.Size([4, 123, 1])
        angle_feats = self.lin1(angle_feats).squeeze(-1)

        if False:
            atom_feats = atom_feats.permute(0, 2, 1)
            atom_feats = self.atom_bn(atom_feats)
            atom_feats = atom_feats.permute(0, 2, 1)

            bond_feats = bond_feats.permute(0, 2, 1)
            bond_feats = self.bond_bn(bond_feats)
            bond_feats = bond_feats.permute(0, 2, 1)
            # ## print(f"==>> bond_feats.shape: {bond_feats.shape}")#e([4, 71, 11])
            angle_feats = angle_feats.permute(0, 2, 1)
            angle_feats = self.angle_bn(angle_feats)
            angle_feats = angle_feats.permute(0, 2, 1)

        torch.cuda.empty_cache()
        atom_masks = torch.zeros((batch_size, mol_length ))
        ### print(f"==>> atom_masks.shape: {atom_masks.shape}")
        atom_masks = atom_masks.unsqueeze(2)
        ### print(f"==>> atom_masks.shape: {atom_masks.shape}")
        atom_feature = F.leaky_relu(self.atom_fc(atom_feats))
        angle_feats = F.leaky_relu(self.angle_fc(angle_feats))
        bond_feats = F.leaky_relu(self.bond_fc(bond_feats))
        atom_feature = self.dropout(atom_feature)
        angle_feats = self.dropout(angle_feats)
        bond_feats = self.dropout(bond_feats)
        batch_size, _, _ = atom_feature.size()
        ## mol level
        attend_masks = atom_degree_list.clone()
        attend_masks[attend_masks != mol_length-1] = 1
        attend_masks[attend_masks == mol_length-1] = 0
        attend_masks = attend_masks.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_masks = atom_degree_list.clone()
        softmax_masks[softmax_masks != mol_length-1] = 0
        softmax_masks[softmax_masks == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_masks = softmax_masks.type(torch.cuda.FloatTensor).unsqueeze(-1)
        if self.args["Feature_Fusion"]:

            neighbor_feature = self.concat(atom_neighbor, bond_neighbor, angle_neighbor)#],dim=-1)
            #neighbor_feature = F.leaky_relu(self.neighbor_fc_angle(neighbor_feature))
        else:
            neighbor_feature = torch.cat([atom_neighbor, bond_neighbor, angle_neighbor],dim=-1)
            neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
#  ## print(attention_weight)
        align_score = align_score + softmax_masks
        attention_weight = F.softmax(align_score,-2)
#  ## print(attention_weight)
        attention_weight = attention_weight * attend_masks
#         ## print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
#  ## print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
#  ## print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

        #do nonlinearity
        activated_atom_features = F.relu(atom_feature)
        for d in range(radius-1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_atom_features[i][atom_degree_list[i].long()] for i in range(batch_size)]

            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_atom_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)

            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)

            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
            #  ## print(attention_weight)
            align_score = align_score + softmax_masks
            attention_weight = F.softmax(align_score,-2)
            #  ## print(attention_weight)
            attention_weight = attention_weight * attend_masks
            #  ## print(attention_weight)
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
            #  ## print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
            #  ## print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
            #  atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            # do nonlinearity
            activated_atom_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_atom_features * atom_masks, dim=-2)
        del atom_feats, bond_feats, angle_feats, atom_degree_list, bond_degree_list, angle_degree_list
        del bond_neighbor, atom_neighbor, angle_neighbor
        del attend_masks, align_score, softmax_masks

        return mol_feature, activated_atom_features, atom_masks, mol_length, num_atom_feat

    def second_GAT(self, mol_feature, activated_atom_features, atom_masks, mol_length ):
        activated_features_mol = F.relu(mol_feature)

        mol_softmax_masks = atom_masks.clone()
        mol_softmax_masks[mol_softmax_masks == 0] = -9e8
        mol_softmax_masks[mol_softmax_masks == 1] = 0
        mol_softmax_masks = mol_softmax_masks.type(torch.cuda.FloatTensor)

        for t in range(self.T):

            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(self.batch_size, mol_length, self.hidden_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_atom_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            if len(mol_softmax_masks.shape) == 2:
                mol_softmax_masks = mol_softmax_masks.unsqueeze(-1)
            mol_align_score = mol_align_score + mol_softmax_masks
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_masks
#  ## print(mol_attention_weight.shape,mol_attention_weight)
            activated_atom_features_transform = self.mol_attend(self.dropout(activated_atom_features))
#  aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_atom_features_transform),-2)
#  ## print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)

            activated_features_mol = F.relu(mol_feature)
        #del mol_context, mol_align, mol_align_score, mol_attention_weight, activated_features_transform, mol_context
        torch.cuda.empty_cache()
        return  activated_features_mol

    def integrated_GAT(self, mol_feature, activated_features_mol, activated_features, atom_masks, mol_length ):
        # # print(f"==>> mol_feature.shape: {mol_feature.shape}")#ol_feature.shape: torch.Size([2, 256]
        # # print(f"==>> activated_features_mol.shape: {activated_features_mol.shape}")# activated_features_mol.shape: torch.Size([2, 256])
        # # print(f"==>> activated_features.shape: {activated_features.shape}")#ctivated_features.shape: torch.Size([2, 1056, 256])
        # # print(f"==>> atom_masks.shape: {atom_masks.shape}")
        ## print(f"==>> activated_features.shape: {activated_features.shape}")
        mol_softmax_masks = atom_masks.clone()
        mol_softmax_masks[mol_softmax_masks == 0] = -9e8
        mol_softmax_masks[mol_softmax_masks == 1] = 0
        mol_softmax_masks = mol_softmax_masks.type(torch.cuda.FloatTensor)

        for t in range(self.T):

            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(self.batch_size, mol_length, self.hidden_dim)

            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            # print(f"==>> mol_align.shape: {mol_align.shape}")#
                #==>> mol_align.shape: torch.Size([2, 1056, 512])

            mol_align_score = F.leaky_relu(self.mol_align_integ(mol_align))
            # print(f"==>> mol_align_score.shape: {mol_align_score.shape}")
            ## print(f"==>> _score.shape: {_score.shape}")
            mol_align_score = mol_align_score + mol_softmax_masks
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_masks
#  ## print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend_integ(self.dropout(activated_features))
#  aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
#  ## print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            # print(f"==>> mol_context.shape: {mol_context.shape}")
            # print(f"==>> mol_feature.shape: {mol_feature.shape}")
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)

            activated_features_mol = F.relu(mol_feature)
        try:
            del mol_align, mol_align_score, mol_attention_weight, activated_features_transform, mol_context
        except: pass
        torch.cuda.empty_cache()
        return  activated_features_mol


    def forward(self, feats, neighbors):


        mol_feature, activated_atom_features, mol_atom_masks, mol_length, _  = self.first_GAT(feats, neighbors, 'mol', self.radius)

        activated_features_fg_mol_2=None
        if self.args["FG_Node"] == "MLMGA":
            for fg in self.fg_list:
            # Iterate over the modules based on the one-hot vector
                """for idx, module in enumerate(self.modules_dict.values()):
                    if one_hot_vector[idx] == 1:
                        x = module(x)"""
            #return x
                atom_feats, _, _, _, _, _ = return_inputs(feats, neighbors, fg)

                if len(atom_feats.shape) == 2:
                    ### print("n", fg)
                    continue

                fg_feature, fg_activated_atom_features, fg_atom_masks, fg_mol_length, num_fg_atom_feat= self.first_GAT(feats, neighbors, fg, 1)

                activated_features_fgmol_2 = self.second_GAT(fg_feature, fg_activated_atom_features, fg_atom_masks, fg_mol_length) #
        torch.cuda.empty_cache()

        if activated_features_fg_mol_2 is not None:

            # print(f"==>> activated_features_fg_mol_2.shape: {activated_features_fg_mol_2.shape}")
            if self.args['pool'] == "avg":
                combined_atom_feats =  torch.cat([activated_atom_features, activated_features_fg_mol_2], dim=1)
                # print(f"==>> combined_atom_feats.shape: {combined_atom_feats.shape}")

                fg_mol_length_2 = combined_atom_feats.shape[1]
                # print("fg_mol_length_2", fg_mol_length_2)
                combined_fg_mol_atom_masks_2 = combined_atom_feats.clone()

                pooled_fg_feature = F.avg_pool1d(fg_feature.unsqueeze(1), kernel_size=fg_feature.size(1)).squeeze(1)
                # print(f"==>> pooled_fg_feature.shape: {pooled_fg_feature.shape}")
                combined_fg_mol_feature_2 = torch.cat([mol_feature, pooled_fg_feature], axis = -1)
                # print(f"==>> combined_fg_mol_feature_2.shape: {combined_fg_mol_feature_2.shape}")


                final_activated_features_mol =  self.integrated_GAT(combined_fg_mol_feature_2, activated_features_fgmol_2, combined_atom_feats, combined_fg_mol_atom_masks_2, fg_mol_length_2)

                mol_prediction = self.output1(self.dropout(final_activated_features_mol ))
            elif self.args['pool'] == "max":

                combined_atom_feats =  torch.cat([activated_atom_features, activated_features_fg_mol_2], dim=1)
                # print(f"==>> combined_atom_feats.shape: {combined_atom_feats.shape}")

                fg_mol_length_2 = combined_atom_feats.shape[1]
                # print("fg_mol_length_2", fg_mol_length_2)
                #fg_atom_masks_2 = combined_atom_feats.clone()
                combined_fg_mol_atom_masks_2 = combined_atom_feats.clone()

                pooled_fg_feature = F.max_pool1d(fg_feature.unsqueeze(1), kernel_size=fg_feature.size(1)).squeeze(1)#F.avg_pool1d(fg_feature.unsqueeze(1), kernel_size=fg_feature.size(1)).squeeze(1)
                # print(f"==>> pooled_fg_feature.shape: {pooled_fg_feature.shape}")
                combined_fg_mol_feature_2 = torch.cat([mol_feature, pooled_fg_feature], axis = -1)
                # print(f"==>> combined_fg_mol_feature_2.shape: {combined_fg_mol_feature_2.shape}")

                final_activated_features_mol =  self.integrated_GAT(combined_fg_mol_feature_2, activated_features_fgmol_2, combined_atom_feats, combined_fg_mol_atom_masks_2, fg_mol_length_2)

                mol_prediction = self.output1(self.dropout(final_activated_features_mol ))
            else:
                # print(f"==>> activated_features_fg_mol_2.shape: {activated_features_fg_mol_2.shape}")#[2, 256])
                activated_features_fg_mol_2_exp = activated_features_fg_mol_2.unsqueeze(1)#([2, 1, 256])
                # print(f"==>> activated_features_fg_mol_2_exp.shape: {activated_features_fg_mol_2_exp.shape}")
                # print(f"==>> activated_atom_features.shape: {activated_atom_features.shape}") #([2, 1055, 256])
                #combined_atom_feats =  torch.cat([activated_atom_features, activated_features_fg_mol_2_exp], dim=1)
                combined_atom_feats = self.concat(activated_atom_features, activated_features_fg_mol_2_exp)
                print(f"==>> combined_atom_feats.shape: {combined_atom_feats.shape}")

                fg_mol_length_2 = combined_atom_feats.shape[1]
                # print("fg_mol_length_2", fg_mol_length_2)
                combined_fg_mol_atom_masks_2 = combined_atom_feats.clone()

                combined_fg_mol_feature_2 = torch.cat([mol_feature, fg_feature], axis = -1)
                # print(f"==>> combined_fg_mol_feature_2.shape: {combined_fg_mol_feature_2.shape}")

                final_activated_features_mol =  self.integrated_GAT(combined_fg_mol_feature_2, activated_features_fgmol_2, combined_atom_feats, combined_fg_mol_atom_masks_2, fg_mol_length_2)
                #self.second_GAT(mol_feature, activated_atom_features, mol_atom_masks, mol_length)
                mol_prediction = self.output1(self.dropout(final_activated_features_mol ))

            torch.cuda.empty_cache()
            return mol_prediction
        else:
            mol_feature, activated_atom_features, mol_atom_masks
            # print(f"==>> mol_feature.shape: {mol_feature.shape}")
            # print(f"==>> activated_atom_features.shape: {activated_atom_features.shape}")#activated_atom_features.shape: torch.Size([2, 1059, 256])

            # print(f"==>> mol_atom_masks.shape: {mol_atom_masks.shape}")
            """=>> mol_feature.shape: torch.Size([2, 256])
            ==>> activated_atom_features.shape: torch.Size([2, 1059, 256])
            ==>> mol_atom_masks.shape: torch.Size([2, 1059, 1])"""
            final_activated_features_mol = self.second_GAT(mol_feature, activated_atom_features, mol_atom_masks, mol_length)
            ### print(f"==>> mol_prediction.shape: {mol_prediction.shape}")
            mol_prediction = self.output1(self.dropout(final_activated_features_mol ))

            torch.cuda.empty_cache()
            return mol_prediction
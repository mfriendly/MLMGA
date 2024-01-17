import os, sys
import pickle, json, time, pickle, gc, argparse, csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import set_start_method
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from my_util import *
from chemutils import *
from sklearn.model_selection import train_test_split
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from my_util import set_seed, load_config_from_json
from MLMGA import Fingerprint, save_smiles_dicts, get_smiles_array
import torchmetrics
sys.setrecursionlimit(50000)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
torch.set_default_device(device)
torch.nn.Module.dump_patches = True
sns.set(color_codes=True)
EPOCHS = 200
BATCHSIZE = 32
PATIENCE = 25
PILOT = 1000000000
NWORKERS = 4 
PIN_MEMORY = True
per_task_output_units_num = 2
script_dir = ""
if script_dir not in sys.path:
    sys.path.append(script_dir)
script_name = os.path.basename(__file__).replace(".", "_")
try:
    set_start_method("spawn")
except RuntimeError:
    pass
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
class SMILESDataset(Dataset):
    def __init__(self, dataframe, smiles_list, task_columns, feature_dicts, args):
        self.dataframe = dataframe
        self.smiles_list = smiles_list
        self.task_columns = task_columns
        self.feature_dicts = feature_dicts
        self.args = args
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        targets = []
        for task in self.task_columns:
            targets.append(self.dataframe[task].tolist()[idx])
        smiles, feats, neighbors = get_smiles_array([smiles], self.feature_dicts, self.args)
        smiles = self.smiles_list[idx]
        y_val = [torch.tensor([target], dtype=torch.float) for target in targets]
        y_val = torch.concat(y_val, dim=0)
        return  (feats, neighbors), y_val.long()
def create_collate_fn(typical_shapes):
    def collate_fn(batch):
        if not batch:
            return {}, torch.tensor([])
        keys = fg_list +['mol'] 
        expected_neighbors_keys = [key + f'_{x}_neighbors' for key in keys for x in ['atom', 'bond', 'angle']]
        expected_features_keys = [key + f'_{x}_features' for key in keys for x in ['atom', 'bond', 'angle']]
        smiles_data, targets = zip(*batch)
        collated_feats = {}
        collated_neighbors = {}
        for key in expected_neighbors_keys + expected_features_keys :
            if key not in collated_feats and key not in collated_neighbors:
                default_shape = typical_shapes.get(key, (BATCHSIZE, 256)) # Replace with appropriate default shape
                default_value = torch.zeros(default_shape)
                if '_features' in key:
                    collated_feats[key] = default_value
                elif '_neighbors' in key:
                    collated_neighbors[key] = default_value
        max_lengths_1, max_lengths_2 = {}, {}
        for key in smiles_data[0][0].keys():
            max_lengths_1[key] = max(sample[0][key].size(0) for sample in smiles_data)
        for key in smiles_data[0][1].keys():
            max_lengths_2[key] = max(sample[1][key].size(0) for sample in smiles_data)
        for key in max_lengths_1.keys():
            collated_feats[key] = torch.stack([F.pad(sample[0][key], (0, 0, 0, max(max_lengths_1[key] - sample[0][key].size(0), 0))) for sample in smiles_data], 
                dim=0)
        for key in max_lengths_2.keys():
            collated_neighbors[key] = torch.stack([F.pad(sample[1][key], (0, 0, 0, max(max_lengths_2[key] - sample[1][key].size(0), 0))) for sample in smiles_data], 
                dim=0)
        collated_targets = default_collate(targets)
        return (collated_feats, collated_neighbors), collated_targets
    return collate_fn
def get_data_loaders(train_df, val_df, test_df, tasks, feature_dicts, drop_last=True, args=None):
    train_dataset = SMILESDataset(dataframe=train_df, smiles_list =  train_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    val_dataset = SMILESDataset(dataframe=val_df, smiles_list =  val_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    test_dataset =  SMILESDataset(dataframe=test_df, smiles_list =  test_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    typical_shapes = {}
    for i in range(min(len(train_dataset), 10)):  # Check the first 100 samples
        sample = train_dataset[i]
        for key, tensor in sample[0][0].items():  
            if key not in typical_shapes:
                typical_shapes[key] = []
            typical_shapes[key].append(tensor.shape)
    for key in typical_shapes.keys():
        shapes = typical_shapes[key]
        # Calculate the median shape of each dimension, excluding batch size
        typical_shapes[key] = torch.median(torch.Tensor([s[0] for s in shapes]), 0).values.int().tolist()
    collate_fn_my = create_collate_fn(typical_shapes)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS)
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
class EarlyStopping:
    def __init__(self, TYPE = None, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_ROC_max = -np.Inf
        self.delta = delta
        path = TYPE + path
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_ROC, model):
        score = val_ROC
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_ROC, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_ROC, model)
            self.counter = 0
    def save_checkpoint(self, val_ROC, model):
        if self.verbose:
            self.trace_func(f"Validation ROC increased ({self.val_ROC_max:.6f} --> {val_ROC:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_ROC_max = val_ROC
def train(model, train_loader, optimizer, auroc_metrics):
    torch.cuda.empty_cache()
    model.train()
    model = model.to(device)
    train_losses_list = []
    train_ROC_list = []
    for _, batch in enumerate(train_loader):
        optimizer.zero_grad()  
        targets_list = batch[-1]
        feats, neighbors = batch[0]
        outputs = model(feats, neighbors)
        total_loss = 0
        total_ROC = 0
        for i, target in enumerate(targets_list):
            output = outputs[:, i * per_task_output_units_num : (i + 1) * per_task_output_units_num]
            valid_indices = (target == 0) | (target == 1)
            if valid_indices.sum() == 0:
                continue
            loss = F.cross_entropy(output[valid_indices], target[valid_indices])
            print('CE loss: ', loss)
            total_loss += loss
            y_pred_adjust = F.softmax(output, dim=-1)[:, 1]
            auroc_metrics[i](y_pred_adjust[valid_indices], target[valid_indices])
            auroc_score = auroc_metrics[i].compute()
            total_ROC += auroc_score
        total_loss.backward()
        optimizer.step()
        train_losses_list.append(total_loss.item())
        train_ROC_list.append(total_ROC.item())
    train_loss = np.mean(train_losses_list)
    train_auc = np.mean(train_ROC_list)
    print('Train AUC: ', train_auc)
    print('Train Loss: ', train_loss)
    return train_loss, train_auc
def eval(model, loader,  auroc_metrics):
    torch.cuda.empty_cache()
    model.eval()
    torch.autograd.set_grad_enabled(False)
    model = model.to(device)
    with torch.no_grad():
        eval_losses_list = []
        eval_ROC_list = []
        for _, batch in enumerate(loader):
            targets_list = batch[-1]
            feats, neighbors = batch[0]
            outputs = model(feats, neighbors)
            total_loss = 0
            total_ROC = 0
            for i, target in enumerate(targets_list):
                output = outputs[:, i * per_task_output_units_num : (i + 1) * per_task_output_units_num  ]
                valid_indices = (target == 0) | (target == 1)
                if valid_indices.sum() == 0:
                    continue
                loss = F.cross_entropy(output[valid_indices], target[valid_indices])
                total_loss += loss
                y_pred_adjust = F.softmax(output, dim=-1)[:, 1]
                auroc_metrics[i](y_pred_adjust[valid_indices], target[valid_indices])
                auroc_score = auroc_metrics[i].compute()  
                total_ROC += auroc_score
            eval_losses_list.append(total_loss.item())
            eval_ROC_list.append(total_ROC.item())
    test_loss = np.array(eval_losses_list).mean()
    test_auc = np.array(eval_ROC_list).mean()
    print('test_auc: ', test_auc)
    print('test_loss: ', test_loss)
    del eval_losses_list, eval_ROC_list
    return test_loss, test_auc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Parameters")
    parser.add_argument("--EPOCHS", type=int, default=200, help="Number of epochs")
    parser.add_argument("--SEED", type=int, default=3, help="Random seed")
    parser.add_argument("--BATCHSIZE", type=int, default=BATCHSIZE, help="Batch size")
    parser.add_argument("--ANGLE", type=bool, default=True)
    parser.add_argument("--FUNCTIONALGROUP", type=bool, default=True)
    parser.add_argument("--per_task_output_units_num", type=int, default=1, help="Per task output units number")
    parser.add_argument("--add_atoms", type=bool, default=True, help="Add atoms (default: True)")
    parser.add_argument("--explicit_H", type=bool, default=True, help="Explicit Hydrogen (default: True)")
    parser.add_argument("--use_chirality", type=bool, default=True, help="Use chirality (default: True)")
    parser.add_argument("--add_bond_length", type=bool, default=True, help="Add bond length (default: True)")
    parser.add_argument("--ANGLE_ENCODER", type=str, help="Explicit Hydrogen (default: True)")
    parser.add_argument("--Feature_Fusion", type=str, help="Explicit Hydrogen (default: True)")
    parser.add_argument("--FG_Node", type=str, help="Explicit Hydrogen (default: True)")
    args = vars(parser.parse_args())
    SEED = args["SEED"]
    set_seed(seed=SEED, make_deterministic=True)
    tasks_details = [
        ["clintox", ["FDA_APPROVED", "CT_TOX"], "data/clintox.csv"], 
        #["tox21", ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"], "data/tox21.csv"], 
      #["BBBP", ["BBBP"], "data/BBBP.csv"],  
        #["sider", ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9', 'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17', 'SIDER18', 'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25', 'SIDER26', 'SIDER27'], "data/sider.csv"]
    ]
    if True:#for tuple in [("AngleEncoderV1", True, "MLMGA")]:
        #args["ANGLE_ENCODER"], args["Feature_Fusion"], args["FG_Node"] = tuple
        for task in tasks_details:
            task_name, tasks, raw_filename = task
            args["task_name"] = task_name
            
            PREFIX = task_name + "_" + str(script_name[-6:-3]) + "_" + str(SEED) + "_"
            PREFIX_simple =  task_name + "_" + str(script_name[-6:-3]) + "_" 
            TYPE = task_name
            SAVE = "results_MLMGA/" + task_name
            if not os.path.exists(SAVE):
                os.makedirs(SAVE)
            PREFIX += args["ANGLE_ENCODER"]+ "_" 
            PREFIX +=  str(args["Feature_Fusion"])+ "_" 
            PREFIX += str(args["FG_Node"])+ "_" 
            TYPE = task_name + "_"+"_".join(PREFIX.split("_")[-4:])
            R = SAVE + "/" + PREFIX + "_result.json"
            feature_filename = raw_filename.replace(".csv", "_full_3.pickle")
            filename = raw_filename.replace(".csv", "_full_3")
            prefix_filename = raw_filename.split("/")[-1].replace(".csv", "_full_3")
            try:
                smiles_tasks_df = pd.read_csv(raw_filename, nrows=PILOT)
            except Exception as e:
                smiles_tasks_df = pd.read_csv(raw_filename, names=tasks + ["smiles"], nrows=PILOT)
            smilesList = smiles_tasks_df.smiles.values
            atom_num_dist = []
            remained_smiles = []
            canonical_smiles_list = []
            for smiles in smilesList:
                if smiles is None:
                    continue
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None or len(mol.GetAtoms()) < 3:
                        continue
                    atom_num_dist.append(len(mol.GetAtoms()))
                    remained_smiles.append(smiles)
                    canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                except Exception as e:
                    continue
            smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
            smiles_tasks_df["cano_smiles"] = canonical_smiles_list
            smilesList = [smiles
                for smiles in canonical_smiles_list
                if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 401]  # 101]
            uncovered = [smiles
                for smiles in canonical_smiles_list
                if len(Chem.MolFromSmiles(smiles).GetAtoms()) > 400]  # 100]
            gc.collect()
            import time
            start_time = str(time.ctime()).replace(":", "-").replace(" ", "_")
            import pickle, os
            if os.path.exists(feature_filename):
                feature_dicts = pickle.load(open(feature_filename, "rb"))
                print("loaded feature_dicts from {}".format(feature_filename))
            else:
                feature_dicts = save_smiles_dicts(smilesList, filename, args)
            gc.collect()
       
            patterns, num_functional_groups = get_functional_group_patterns()
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
            args['fg_list']  = fg_list
            fg_dict = {}
            smilesList = feature_dicts["smiles_to_atom_mask"].keys()
            def check_functional_groups(smiles, patterns):
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    return []
                results = []
                for key, pattern in patterns.items():
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol is None or not mol.HasSubstructMatch(pattern_mol):
                        pass
                    else:
                        results.append(str(key))
                
                return results
            for smiles in smilesList:
                fg_dict[str(smiles)] =check_functional_groups(smiles, patterns)
            
            with open("f{args['task_name']}_smiles_to_fg_list.json", "w") as f:
                json.dump(fg_dict, f, indent=4)
            feature_dicts["smiles_to_functional_group_info"] = fg_dict
            remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())]
            uncovered_df = smiles_tasks_df.drop(remained_df.index)
            gc.collect()
            np.set_printoptions(threshold=100)
            weights = []
            for i, task in enumerate(tasks):
                negative_df = remained_df[remained_df[task] == 0][["smiles", task]]
                positive_df = remained_df[remained_df[task] == 1][["smiles", task]]
                weights.append([(positive_df.shape[0] + negative_df.shape[0])
                        / negative_df.shape[0], (positive_df.shape[0] + negative_df.shape[0])
                        / positive_df.shape[0], ])
            test_df = remained_df.sample(frac=1 / 10, random_state=SEED)
            training_data = remained_df.drop(test_df.index)
            val_df = training_data.sample(frac=1 / 9, random_state=SEED)
            train_df = training_data.drop(val_df.index)
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            loaders = get_data_loaders(train_df, val_df, test_df, tasks, feature_dicts, True, args)
            np.set_printoptions(threshold=100)
            cc = canonical_smiles_list[0]
            smiles, feats, neighbors = get_smiles_array([cc], feature_dicts, args)
            num_atom_node = feats['mol_atom_features'].shape[-2]
            num_atom_features = feats['mol_atom_features'].shape[-1]
            num_bond_features = feats['mol_bond_features'].shape[-1]
            num_angle_features = feats['mol_angle_features'].shape[-1]
            num_atom_neighbors = neighbors['mol_atom_neighbors'].shape[-1]
            num_bond_neighbors = neighbors['mol_bond_neighbors'].shape[-1]
            num_angle_neighbors = neighbors['mol_angle_neighbors'].shape[-1]
            del feats, neighbors, smiles
            gc.collect()
            loss_function =nn.CrossEntropyLoss()
 
            try:
                C_PATH = SAVE + "/" + PREFIX_simple + "config.json"
                config = load_config_from_json(C_PATH)
                
            except Exception as e:
                print(e)
                C_PATH = "results_MLMGA/BBBP/BBBP_CLS_config.json"
                config = load_config_from_json(C_PATH)
            fingerprint_dim = config["fingerprint_dim"]
            weight_decay = config["weight_decay"]
            learning_rate = config["learning_rate"]
            radius = config["radius"]
            T = config["T"]
            p_dropout = config["p_dropout"]
            output_units_num = len(tasks) * per_task_output_units_num
            loss_function =     nn.CrossEntropyLoss()
            model = Fingerprint(args,
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
                    p_dropout, num_atom_node).to(device)
            best_path = ("saved_models/model_"
                + prefix_filename
                + "_best.pt")
            try:
                model.load_state_dict(torch.load(TYPE + "checkpoint.pt"), strict =False)
                #best_model = torch.load(best_path)
                #best_model_dict = best_model.state_dict()
                #best_model_wts = copy.deepcopy(best_model_dict)
                #model.load_state_dict(best_model_wts)
                ## print(f"CKPT loaded")# , best_path)
            except Exception as e:
                print("failed to laod CKPT", e)
            model.to(device)
            optimizer = optim.Adam(model.parameters())#, 10**-learning_rate, weight_decay=10**-weight_decay,)
            auroc_metrics = [torchmetrics.AUROC(num_classes=2,  task="binary") for _ in tasks]
            
            best_param = {}
            best_param["roc_epoch"] = 0
            best_param["loss_epoch"] = 0
            best_param["val_ROC"] = 0
            best_param["val_loss"] = 9e8
            early_stopping = EarlyStopping(TYPE = TYPE, patience=PATIENCE, verbose=True)
            LOSS_PATH = SAVE + "/" + PREFIX +"ROCs.csv"
            with open(LOSS_PATH, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_ROC", "val_ROC"])
                best_val_loss = float("inf")
                for epoch in tqdm(range(EPOCHS), total=EPOCHS, desc="Training Progress"):
                
                    train_loss, train_ROC = train(model, loaders['train'], optimizer, auroc_metrics)
                    print('train_loss, train_ROC: ', train_loss, train_ROC)
                    val_loss, val_ROC = eval(model, loaders['val'], auroc_metrics)
                    if val_ROC > best_param["val_ROC"]:
                        best_param["val_epoch"] = epoch
   
                        best_param["val_ROC"] = val_ROC
                        best_path = ("saved_models/model_"
                            + prefix_filename
                            + "_best.pt")
                        torch.save(model, best_path)
                    train_ROC, val_ROC = round(train_ROC, 4), round(val_ROC, 4)
                    writer.writerow([epoch, train_ROC, val_ROC])
                    early_stopping(val_ROC, model)
                    last_epoch = None
                    if early_stopping.early_stop:
                        last_epoch = epoch
                        print("➡ last_epoch :", last_epoch)
                        break
            try:
                model.load_state_dict(torch.load(TYPE + "checkpoint.pt"))
                #best_model = torch.load(best_path)
                #best_model_dict = best_model.state_dict()
                #best_model_wts = copy.deepcopy(best_model_dict)
                #model.load_state_dict(best_model_wts)
                #(best_model.align[0].weight == model.align[0].weight).all()
                test_loss, test_ROC = eval(model, loaders['test'],auroc_metrics)
                print('test_loss, test_ROC: ', test_loss, test_ROC)
                #del best_model_wts
            except Exception as e:
                print(e)
            gc.collect()
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            def serialize_dict(data):
                if isinstance(data, dict):
                    return {k: serialize_dict(v) for k, v in data.items()}
                elif isinstance(data, np.ndarray):
                    return data.tolist()  
                elif isinstance(data, (np.int64, np.float64)):
                    return int(data) if isinstance(data, np.int64) else float(data)
                return data
            results_dict = {
                "SEED": SEED, 
                "EPOCHS": EPOCHS, 
                "BATCHSIZE": BATCHSIZE, 
                "params": params, 
                "PATIENCE": PATIENCE, 
                #"Feature_Fusion": args["Feature_Fusion"], 
                "ANGLE_ENCODER": args["ANGLE_ENCODER"], 
                "Feature_Fusion": args["Feature_Fusion"], 
                "FG_Node": args["FG_Node"], 
                "Use chirality": args["use_chirality"], 
                "FG_Node": args["add_bond_length"], 
                "test_ROC": round(float(test_ROC), 4), 
                #"config_file_path": str(C_PATH), 
                "best_path": str(best_path), 
                "last_epoch": int(last_epoch)
            }
            results_dict = {
                k: int(v) if isinstance(v, np.int64) else v
                for k, v in results_dict.items()
            }
            results_dict = serialize_dict(results_dict)
            with open(R, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(results_dict)
            
            df = pd.read_csv(LOSS_PATH)[['train_ROC','val_ROC']]
            print('df: ', df)
            df = df.iloc[-last_epoch:,:].reset_index(drop=True)
            print('df: ', df)
            df['train_ROC'] = pd.to_numeric(df['train_ROC'], errors='coerce')
            df['val_ROC'] = pd.to_numeric(df['val_ROC'], errors='coerce')
            df[['train_ROC','val_ROC']].plot()
            print('df: ', df)
            plot_filename = LOSS_PATH.replace(".csv","_plot.png") #
            plt.savefig(plot_filename)
            print(f"Loss plot saved as {plot_filename}")
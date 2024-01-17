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
from sklearn.model_selection import train_test_split
from my_util import *
from chemutils import *
from functools import cached_property
import pandas as pd
import json
from torch.utils.data import Dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
from my_util import set_seed, load_config_from_json
from MLMGA import Fingerprint, save_smiles_dicts, get_smiles_array
#import torchmetrics
sys.setrecursionlimit(50000)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
torch.set_default_device(device)
torch.nn.Module.dump_patches = True

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

EPOCHS = 200
BATCHSIZE = 32
PATIENCE = 20
NWORKERS = 0
PIN_MEMORY = False
PILOT = 100000000000
per_task_output_units_num = 1
np.set_printoptions(threshold=100)
script_dir = ""
if script_dir not in sys.path:
    sys.path.append(script_dir)
script_name = os.path.basename(__file__).replace(".", "_")
class EarlyStopping:
    def __init__(self,
        TYPE=None,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        path = TYPE + path
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class SMILESDataset(Dataset):
    def __init__(self, dataframe, smiles_list, task_columns, feature_dicts, args):
        self.dataframe = dataframe
        self.smiles_list = smiles_list
        self.task_columns = task_columns[0]
        self.feature_dicts = feature_dicts
        self.args = args
        self.path = args.get('dataset_path', 'default_dataset_path.csv')  # Example path, modify as needed

    @cached_property
    def dataset(self):
        print("Loading the dataset...")
        return self._load_dataset()

    def _load_dataset(self):
        # Load a big dataset here
        df = pd.read_csv(self.path)
        return df

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        # Additional processing as required
        target = self.dataframe[self.task_columns].tolist()[idx]
        smiles, feats, neighbors = get_smiles_array([smiles], self.feature_dicts, self.args)
        y_val = torch.tensor(target, dtype=torch.float)
        return (feats, neighbors), y_val


class SMILESDatas000et(Dataset):
    def __init__(self, dataframe, smiles_list, task_columns, feature_dicts, args):
        self.dataframe = dataframe
        self.smiles_list = smiles_list
        self.task_columns = task_columns[0]
        self.feature_dicts = feature_dicts
        self.args = args


        # Load JSON data
        def load_json_data(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data

        #self.smiles_to_fglist = load_json_data(f"{args['task_name']}_smiles_to_fg_list.json")
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        #fg_list = self.smiles_to_fglist[smiles]
        target = self.dataframe[self.task_columns].tolist()[idx]
        smiles, feats, neighbors = get_smiles_array([smiles], self.feature_dicts, self.args)
        y_val = torch.tensor(target, dtype=torch.float)
        return  (feats, neighbors), y_val

def create_collate_fn(typical_shapes):
    def collate_fn(batch):
        if not batch:
            return {}, torch.tensor([])
        #global fg_list
        keys = fg_list +['mol']
        expected_neighbors_keys = [key + f'_{x}_neighbors' for key in keys for x in ['atom', 'bond', 'angle']]
        expected_features_keys = [key + f'_{x}_features' for key in keys for x in ['atom', 'bond', 'angle']]
        smiles_data, targets = zip(*batch)
        #fg_list_in_batch = smiles_data[-1]
        collated_feats = {}
        collated_neighbors = {}
        for key in expected_neighbors_keys + expected_features_keys :
            if key not in collated_feats and key not in collated_neighbors:
                #print(f"Warning: Key '{key}' not found. Using default value.")
                default_shape = typical_shapes.get(key, (BATCHSIZE, 256))
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
        #collated_fglist = default_collate(fg_list_in_batch)
        return (collated_feats, collated_neighbors ), collated_targets
    return collate_fn
def get_data_loaders(train_df, val_df, test_df, tasks, feature_dicts, drop_last=True, args=None):
    train_dataset = SMILESDataset(dataframe=train_df, smiles_list =  train_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    val_dataset = SMILESDataset(dataframe=val_df, smiles_list =  val_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    test_dataset =  SMILESDataset(dataframe=test_df, smiles_list =  test_df['cano_smiles'].tolist(), task_columns=tasks, feature_dicts= feature_dicts, args = args)
    typical_shapes = {}
    for i in range(min(len(train_dataset), 10)):
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
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS, generator=torch.Generator(device=device))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS, generator=torch.Generator(device=device))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, drop_last=drop_last, pin_memory=PIN_MEMORY, collate_fn=collate_fn_my, num_workers=NWORKERS, generator=torch.Generator(device=device))
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
def train(model, train_loader, optimizer, loss_function):
    torch.cuda.empty_cache()
    model.train()
    for _, batch in enumerate(train_loader):
        y_val = batch[-1]
        feats= batch[0][0]
        neighbors = batch[0][1]
        y_val = torch.tensor(y_val, dtype=torch.float).to(device)
        gc.collect()
        mol_prediction = model(feats, neighbors)
        gc.collect()
        mol_prediction = mol_prediction.to(device)
        model.zero_grad()
        loss = torch.sqrt(loss_function(mol_prediction, torch.Tensor(y_val).view(-1, 1)))
        loss.backward()
        optimizer.step()
        gc.collect()
    return loss,loss
def eval(model, loader):
    torch.cuda.empty_cache()
    model.eval()
    model.to(device)
    test_RMSE_list=[]
    with torch.no_grad():
        for _, batch in enumerate(loader):
            y_val = batch[-1]
            feats, neighbors= batch[0]
            y_val = torch.tensor(y_val, dtype=torch.float).to(device)
            gc.collect()
            mol_prediction = model(feats, neighbors)
            mol_prediction = mol_prediction.view(-1, 1)
            RMSE = torch.sqrt(F.mse_loss(mol_prediction, y_val.view(-1, 1), reduction="none"))
            rmse_numpy = RMSE.data.squeeze().cpu().numpy()
            if rmse_numpy.ndim == 0:
                test_RMSE_list.append(rmse_numpy.item())
            else:
                test_RMSE_list.extend(rmse_numpy.tolist())
            gc.collect()
        mean_RMSE = np.mean(test_RMSE_list)
    del rmse_numpy, RMSE, test_RMSE_list
    gc.collect()
    return mean_RMSE, mean_RMSE
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
    args['pool'] = 'Concat'
    set_seed(seed=SEED, make_deterministic=True)
    tasks_details = [
        #('ZINC', ["logP", "qed", "SAS"] , "data/ZINC.csv"),
        #('lipophilicity', ['exp'], "data/Lipophilicity.csv"),
      # ('Freesolv',['exp'], "data/Freesolv.csv"), #nan NAN
         #       ('qm8',['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],'data/qm8.csv') ,
        ('solubility', ['measured log solubility in mols per litre'], "data/delaney-processed.csv"),
        #('Malaria Bioactivity', ['Loge EC50'], "data/malaria-processed.csv"),
        #('Photovoltaic efficiency', ['PCE'], "data/cep-processed.csv"),
    ]
    if True:#
    #for tuple in [("AngleEncoderV1", True, "MLMGA")]:
        #args["ANGLE_ENCODER"], args["Feature_Fusion"], args["FG_Node"] = tuple
        for task in tasks_details:

            task_name, tasks, raw_filename = task
            args["task_name"] = task_name
            if task_name =='ZINC':
                PILOT = 5000

            PREFIX = task_name + "_" + str(script_name[-6:-3]) + "_" + str(SEED) + "_"
            PREFIX_simple = task_name + "_" + str(script_name[-6:-3]) + "_"
            TYPE = task_name
            SAVE = "results_MLMGA/" + task_name
            if not os.path.exists(SAVE):
                os.makedirs(SAVE)
            PREFIX += str(args["ANGLE_ENCODER"])+ "_"
            PREFIX +=  str(args["Feature_Fusion"])+ "_"
            PREFIX += str(args["FG_Node"])+ "_"
            PREFIX += str(args['pool'] )+ "_"
            TYPE = task_name + "_"+"_".join(PREFIX.split("_")[-4:])
            R = SAVE + "/" + PREFIX + "_result.json"
            feature_filename = raw_filename.replace(".csv", "_full_3.pickle")

            filename = raw_filename.replace(".csv", "_full_3")
            prefix_filename = raw_filename.split("/")[-1].replace(".csv", "_full_3")
            try:
                smiles_tasks_df = pd.read_csv(raw_filename, nrows =PILOT)
            except Exception as e:
                smiles_tasks_df = pd.read_csv(raw_filename, names=tasks + ["smiles"], nrows =PILOT)
            print("➡ smiles_tasks_df :", smiles_tasks_df.columns)
            smilesList = smiles_tasks_df.smiles.values
            atom_num_dist = []
            remained_smiles = []
            canonical_smiles_list = []
            for smiles in smilesList:
                if smiles is None:
                    continue
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None or len(mol.GetAtoms()) < 3 or len(mol.GetAtoms()) >401:
                        continue
                    atom_num_dist.append(len(mol.GetAtoms()))
                    remained_smiles.append(smiles)
                    canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                except Exception as e:
                    continue
            print("number of successfully processed smiles: ", len(remained_smiles))
            smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
            smiles_tasks_df["cano_smiles"] = canonical_smiles_list
            smilesList = canonical_smiles_list
            torch.cuda.empty_cache()
            patterns, num_functional_groups = get_functional_group_patterns()
            fg_list = list(patterns.keys())
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
            args['fg_list'] = fg_list
            fg_dict = {}
            if os.path.exists(feature_filename):
                feature_dicts = pickle.load(open(feature_filename, "rb"))
                print("loaded feature_dicts from {}".format(feature_filename))
            else:
                print("Saving feature_dicts to {}".format(feature_filename))
                feature_dicts = save_smiles_dicts(smilesList, filename, args)
            smilesList = list(feature_dicts["smiles_to_atom_mask"].keys())
            for smiles in smilesList:

                fg_dict[str(smiles)] =check_functional_groups(smiles, patterns)
            with open(f"{args['task_name']}_smiles_to_fg_list.json", "w") as f:
                json.dump(fg_dict, f, indent=4)

            feature_dicts["smiles_to_functional_group_info"] = fg_dict

            remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())]
            uncovered_df = smiles_tasks_df.drop(remained_df.index)
            gc.collect()
            torch.cuda.empty_cache()
            cc =canonical_smiles_list[0]

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
            train_df, test_df = train_test_split(remained_df, test_size=0.1)
            train_df, val_df = train_test_split(train_df, test_size=1 / 9)
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            loaders = get_data_loaders(train_df, val_df, test_df, tasks, feature_dicts, True, args)

            loss_function = nn.MSELoss()
            if True:
                C_PATH = SAVE + "/" + PREFIX_simple + "config.json"
                config = load_config_from_json(C_PATH)
                fingerprint_dim = config["fingerprint_dim"]
                weight_decay = config["weight_decay"]
                learning_rate = config["learning_rate"]
                radius = config["radius"]
                T = config["T"]
                p_dropout = config["p_dropout"]
                output_units_num = len(tasks) * per_task_output_units_num
                print("SEED", SEED)
                print("★ Beging training!")
                def initialize_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                model = Fingerprint(args, radius, T, num_functional_groups, num_atom_features, num_bond_features, num_angle_features, num_atom_neighbors, num_bond_neighbors, num_angle_neighbors, fingerprint_dim, output_units_num, p_dropout, num_atom_node)
                model = model.to(device)
                #model.apply(initialize_weights)
                best_path = ("saved_models/model_"
                    + prefix_filename
                    + "_best.pt")
                try:
                    model.load_state_dict(torch.load(TYPE + "checkpoint.pt"), strict =False)
                    #best_model = torch.load(best_path)
                    #best_model_dict = best_model.state_dict()
                    #best_model_wts = copy.deepcopy(best_model_dict)
                    #model.load_state_dict(best_model_wts)
                    # print(f"CKPT loaded")# , best_path)
                except Exception as e:
                    print("failed to laod CKPT", e)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay,)
                best_param = {}
                best_param["val_epoch"] = 0
                best_param["val_RMSE"] = 9e8

                early_stopping = EarlyStopping(TYPE=TYPE, patience=PATIENCE, verbose=True)
                LOSS_PATH = SAVE + "/" + PREFIX +"losses.csv"
                with open(LOSS_PATH, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["epoch", "train_RMSE", "val_RMSE"])
                    for epoch in tqdm(range(EPOCHS), total=EPOCHS, desc="Training Progress"):
                        train_RMSE, train_RMSE  =train(model, loaders['train'], optimizer, loss_function)
                        val_RMSE, val_RMSE = eval(model, loaders['val'])
                        if val_RMSE < best_param["val_RMSE"]:
                            best_param["val_epoch"] = epoch
                            best_param["val_RMSE"] = val_RMSE
                            best_path = (       "saved_models/model_"
                                + prefix_filename
                                + "_best.pt")
                            torch.save(model, best_path)
                        writer.writerow([epoch, train_RMSE, val_RMSE])
                        print("➡ epoch, train_RMSE, val_RMSE:", epoch, train_RMSE, val_RMSE, PREFIX)
                        early_stopping(val_RMSE, model)
                        last_epoch = None
                        if early_stopping.early_stop:
                            print("Early stopping")
                            last_epoch = epoch
                            break
            model.load_state_dict(torch.load(TYPE + "checkpoint.pt"), strict = False)
            #best_model = torch.load(best_path)
            #best_model_dict = best_model.state_dict()
            #best_model_wts = copy.deepcopy(best_model_dict)
            #model.load_state_dict(best_model_wts, strict = False)
            #(best_model.align[0].weight == model.align[0].weight).all()
            test_RMSE, test_RMSE = eval(model, loaders['test'])
            # print("test_RMSE, test_RMSE", test_RMSE, test_RMSE)
            gc.collect()
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            def serialize_dict(data):
                if isinstance(data, dict):
                    return {k: serialize_dict(v) for k, v in data.items()}
                elif isinstance(data, np.ndarray):
                    return data.tolist()
                elif isinstance(data, (np.int64, np.float64, np.int32)):
                    return int(data) if isinstance(data, np.int64) else float(data)
                return data
            results_dict = {
                "SEED": SEED, "EPOCHS": EPOCHS, "BATCHSIZE": BATCHSIZE, "params": params, #
                "PATIENCE": PATIENCE, "ANGLE": args["ANGLE"], "FUNCTIONALGROUP": args["FUNCTIONALGROUP"], "Add atoms": args["add_atoms"], "Explicit Hydrogen": args["explicit_H"], "Use chirality": args["use_chirality"], "Add bond length": args["add_bond_length"], "test RMSE": round(float(test_RMSE), 3), "config_file_path": str(C_PATH), "best_path": str(best_path), "last_epoch": int(last_epoch)
            }
            results_dict = {
                k: str(v) if isinstance(v, np.int64) or isinstance(v, np.int32) else v
                for k, v in results_dict.items()
            }
            results_dict = serialize_dict(results_dict)
            with open(R, "w") as f:
                json.dump(results_dict, f, indent=4)
            print(results_dict)
            df = pd.read_csv(LOSS_PATH)[['train_RMSE','val_RMSE']]
            print('df: ', df)
            df = df.iloc[-last_epoch:,:].reset_index(drop=True)
            print('df: ', df)
            df['train_RMSE'] = pd.to_numeric(df['train_RMSE'], errors='coerce')
            df['val_RMSE'] = pd.to_numeric(df['val_RMSE'], errors='coerce')
            df[['train_RMSE','val_RMSE']].plot()

            plot_filename = LOSS_PATH.replace(".csv","_plot.png") #
            plt.savefig(plot_filename)
            print(f"Loss plot saved as {plot_filename}")
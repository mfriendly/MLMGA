import json
from collections import OrderedDict
import torch
import numpy as np
import random
import os
def set_seed(seed=42, make_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
def save_config_as_json(hp_opt, file_path):
    with open(file_path, "w") as file:
        json.dump(dict(hp_opt), file)
def load_config_from_json(file_path):
    with open(file_path, "r") as file:
        return OrderedDict(json.load(file))
def save_as_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)
def load_from_json(filename):
    with open(filename, "r") as file:
        return json.load(file)

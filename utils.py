import torch
import random
import numpy as np

def set_seeds(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.set_default_dtype(torch.float32)

def load_torch_file(model, _path):
    state_dict = torch.load(_path)
    # model.load_torch_file(state_dict)
    model.load_state_dict(state_dict, strict=False)
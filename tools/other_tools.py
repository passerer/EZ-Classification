import os 
import random
import numpy as np 
import torch


def seed_everything(seed = 777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def exists_or_mkdir(path):
    assert type(path) == str
    if not os.path.exists(path):
        os.mkdir(path)
        

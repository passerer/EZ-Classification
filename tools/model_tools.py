from torch import nn
from config import config
import matplotlib.pyplot as plt
import numpy as np

def print_model_parm_nums(model):            
    total = sum([param.nelement() for param in model.parameters()])
    print(' Number of params: {} Million'.format(total / 1e6))   
    
def load_weights(model,pretrain_path):
    try:
        pretrained_dict = torch.load(pretrain_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("load success")
    except:
        print("load failed")

def show_lr(schedule, epoch,base_lr = 0.1):
    index = np.arange(epoch)
    lrs = []
    for i in index:
        scheduler.step()
        lr = scheduler.get_lr()
        lrs.append (lr[0])
    plt.plot(index,lrs)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.show()
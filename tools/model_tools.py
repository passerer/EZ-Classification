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
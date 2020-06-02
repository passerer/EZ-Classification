class Accuracy:
    def __init__(self):
        self.right = 0.
        self.num = 0
    def update(self,input:np.ndarray, target:np.ndarray):
        "Computes accuracy with `target` when `input` is bs * n_classes."
        n = target.shape[0]#batch_size
        input = np.argmax(input,axis=-1)
        targs = target
        self.right +=(input==target).sum()
        self.num += n
    def get(self):
        return  self.right/self.num
    def reset(self):
        self.right =0.
        self.num = 0.
        
class Loss:
    def __init__(self):
        self.loss = 0.
        self.count = 0
    def update(self,input):
        "Computes accuracy with `target` when `input` is bs * n_classes."
        self.loss += input
        self.count += 1
    def get(self):
        return  self.loss/self.count
    def reset(self):
        self.loss =0.
        self.count = 0
        
def evaluate_acc_loss(val_loader,model,all_loss,acc,criterion):
    model.eval()
    pbar = tqdm(total=len(val_loader))
    pbar.set_description("valuation loss and accuracy")
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader):
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target))).to(device)
            output = model(input)
            loss = criterion(output,target)
            all_loss.update(loss.cpu().detach().numpy())
            #measure accuracy and record loss
            acc.update(output.cpu().detach().numpy(),target.cpu().detach().numpy())
            pbar.update(1)
            pbar.set_postfix(loss=all_loss.get(),acc=acc.get())
    pbar.close()
    return acc.get()

def evaluate_loss(val_loader,model,all_loss,criterion):
    model.eval()
    pbar = tqdm(total=len(val_loader))
    pbar.set_description("valuation loss")
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader):
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target))).to(device)
            output = model(input)
            loss = criterion(output,target)
            all_loss.update(loss.cpu().detach().numpy())
            pbar.update(1)
            pbar.set_postfix(loss=all_loss.get())
    pbar.close()
    return all_loss.get()

def evaluate_acc(val_loader,model,acc,criterion):
    model.eval()
    pbar = tqdm(total=len(val_loader))
    pbar.set_description("valuation loss and accuracy")
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader):
            input = Variable(input).to(device)
            target = Variable(torch.from_numpy(np.array(target))).to(device)
            output = model(input)
            #measure accuracy and record loss
            acc.update(output.cpu().detach().numpy(),target.cpu().detach().numpy())
            pbar.update(1)
            pbar.set_postfix(acc=acc.get())
    pbar.close()
    return acc.get()
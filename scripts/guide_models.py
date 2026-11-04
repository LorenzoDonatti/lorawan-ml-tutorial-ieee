"""Inspectable classification and reconstruction networks for the tutorial."""
import copy,time
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

class SmallNetwork(nn.Module):
    def __init__(self,p,task):
        super().__init__()
        widths=[p,64,32,1] if task=='classification' else [p,32,2,32,p]
        layers=[]
        for j,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
            layers.append(nn.Linear(a,b))
            if j<len(widths)-2:layers.append(nn.ReLU())
        self.net=nn.Sequential(*layers)
    def forward(self,x):return self.net(x)

class FittedModel:
    def __init__(self,net,scaler,task):self.net,self.scaler,self.task=net,scaler,task
    def predict(self,X):
        self.net.eval()
        with torch.no_grad():
            z=torch.as_tensor(self.scaler.transform(X),dtype=torch.float32)
            out=self.net(z)
            return torch.sigmoid(out).numpy().ravel() if self.task=='classification' else out.numpy()
    def save(self,path):
        arrays={k:v.detach().numpy() for k,v in self.net.state_dict().items()}
        np.savez_compressed(path,**arrays,x_mean=self.scaler.mean_,x_scale=self.scaler.scale_,task=self.task)

def restore(path):
    a=np.load(path,allow_pickle=False);task=str(a['task']);s=StandardScaler()
    s.mean_,s.scale_=a['x_mean'],a['x_scale'];s.var_=s.scale_**2;s.n_features_in_=len(s.mean_)
    net=SmallNetwork(len(s.mean_),task);net.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in net.state_dict()})
    return FittedModel(net,s,task)

def fit(Xtr,Xva,*,task,seed,ytr=None,yva=None):
    start=time.perf_counter();torch.manual_seed(seed);s=StandardScaler().fit(Xtr)
    x=torch.as_tensor(s.transform(Xtr),dtype=torch.float32);v=torch.as_tensor(s.transform(Xva),dtype=torch.float32)
    target=torch.as_tensor(ytr[:,None],dtype=torch.float32) if task=='classification' else x
    valtarget=torch.as_tensor(yva[:,None],dtype=torch.float32) if task=='classification' else v
    net=SmallNetwork(x.shape[1],task);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    lossfn=nn.BCEWithLogitsLoss() if task=='classification' else nn.MSELoss()
    best=np.inf;best_epoch=0;history=[]
    for epoch in range(1,81):
        net.train();loss_sum=0.
        for indices in torch.randperm(len(x)).split(256):
            opt.zero_grad();loss=lossfn(net(x[indices]),target[indices]);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(indices)
        net.eval()
        with torch.no_grad():value=float(lossfn(net(v),valtarget))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(x),validation_loss=value))
        if value<best-1e-6:best=value;best_epoch=epoch;state=copy.deepcopy(net.state_dict())
        if epoch-best_epoch>=12:break
    net.load_state_dict(state)
    return FittedModel(net,s,task),history,dict(best_epoch=best_epoch,validation_loss=best,
        train_seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in net.parameters()))

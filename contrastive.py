import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch import nn
import matplotlib.pyplot as plt
from scipy import signal
import torch
import copy
import random
import os
import argparse
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pandas as pd
from scipy import signal
from soft_dtw_cuda import SoftDTW   # should be in the system
from sklearn.model_selection import train_test_split


class TemporalContrast(nn.Module):
    def __init__(self, batch_size=32, energy=0.5, use_cuda=False, gamma=0.1):
        super(TemporalContrast, self).__init__()
        self.batch_size = batch_size
        self.energy = energy
        self.sdtw = SoftDTW(use_cuda=False, gamma=gamma)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.mask = torch.eye(self.batch_size*2, dtype=torch.bool, device=self.device)
        
    def computeSimmiarity(self, x):
        n = x.shape[0]
        new_mat = torch.zeros(n,n)
        for i in range(n):
            new_ = torch.unsqueeze(x[i], 0)
            print(new_.shape)
            for j in range(n):
                if i == j:
                    pass
                else:
                    _new = torch.unsqueeze(x[j],0)
                    print(_new.shape)
                    m = self.sdtw(new_, _new) 
                    print('value', m)
                    new_mat[i,j] = m #self.sdtw(new_, _new) 
        return new_mat
    
    def forward(self, xxx, yyy):
        full = torch.concat((xxx, yyy), dim=0)
        sim = self.computeSimmiarity(full)
        sim = sim/self.energy
        
        top_p = torch.diag(sim, self.batch_size)
        bel_p = torch.diag(sim, self.batch_size)
        
        postives = torch.concat((top_p, bel_p), dim=0)  # positives
        loss = -positives + torch.logsumexp(sim, dim=-1)
        loss = loss.mean()
        return loss
    
class FeatureContrast(nn.Module):
    def __init__(self, batch_size=32, energy=0.5):
        super(FeatureContrast, self).__init__()
        self.batch_size = batch_size
        self.energy = energy
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.mask = torch.eye(self.batch_size*2, dtype=torch.bool, device=self.device)
        
    def forward(self, x, y):
        full = torch.concat((x, y), dim=0)
        sim = F.cosine_similarity(full[:,None,:], full[None,:,:], dim=-1)  # check
        sim = sim/self.energy
        
        top_p = torch.diag(sim, self.batch_size)
        bel_p = torch.diag(sim, self.batch_size)
        
        positives = torch.concat((top_p, bel_p), dim=0)  # positives
        loss = -positives + torch.logsumexp(sim, dim=-1)
        loss = loss.mean()
        return loss


class Jitter(nn.Module):
    def __init__(self, mean=0.0, std=0.2, size=(10,2)):
        super(Jitter, self).__init__()
        self.mean=mean
        self.std = std
        
    def forward(self, x):
        size = x.shape
        noise = torch.normal(mean=self.mean, std = self.std, size=size)
        return x+noise
    
class Smooth(nn.Module):
    def __init__(self, window=7, polord=3):
        super(Smooth, self).__init__()
        self.window = window
        self.polord = polord
        
    def forward(self, x):
        xx = copy.deepcopy(x)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[-1]):
                xx[i,:,j] = torch.from_numpy(signal.savgol_filter(x[i,:,j], self.window, self.polord))  # check if it returns tensor
        return xx

class Scale(nn.Module):
    def __init__(self, mean=0, std=0.9):
        super(Scale, self).__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        shp = x.shape 
        if len(shp) == 1:
            size = shp
        elif len(shp) == 2:
            size = (shp[0],1)
        elif len(shp) == 3:
            size = (shp[0],1,1)
        scale =  torch.normal(mean=self.mean, std = self.std, size=size)
        return x*scale
    
class TimePermute(nn.Module):
    def __init__(self, n_segs=7):
        super(TimePermute, self).__init__()
        self.n_segs = n_segs
        
    def rawPermute(self, x):
        seg_len = int(x.shape[0]/self.n_segs)
        inds = list(range(seg_len, x.shape[0], seg_len))
        splits = np.split(x, inds)
        permuted = torch.concat([tensor[torch.randperm(tensor.shape[0])].view(-1) for tensor in splits])
        return permuted
    
    def forward(self, A):
        AA = copy.deepcopy(A)
        for i in range(AA.shape[0]):
            for j in range(AA.shape[-1]):
                AA[i,:,j] = self.rawPermute(AA[i,:,j])
        return AA
    

class InceptionTime(nn.Module):
    def __init__(self, input_dim=13,
                 num_classes=9,
                 num_layers=4,
                 hidden_dims=64,
                 use_bias=False,
                 phase='train',
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
        super(InceptionTime, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        self.phase = phase
        self.inlinear = nn.Linear(input_dim, hidden_dims*4)
        self.num_layers = num_layers
        self.inception_modules_list = [InceptionModule(kernel_size=32, num_filters=hidden_dims*4,
                                                       use_bias=use_bias, device=device) for _ in range(num_layers)]
        self.inception_modules = nn.Sequential(
            *self.inception_modules_list)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims*4,num_classes)

        self.to(device)

    def forward(self,x):
        # N x T x D -> N x D x T
        x = x.transpose(1,2)
        
        x = self.inlinear(x.transpose(1, 2)).transpose(1, 2)
        for i in range(self.num_layers):
            x = self.inception_modules_list[i](x)
        y = self.avgpool(x).squeeze(2)
        y = self.outlinear(y)
        if self.phase != 'train':
            logs = F.log_softmax(y, dim=-1)
            return logs
        else:
            return x, y

class InceptionModule(nn.Module):
    def __init__(self, kernel_size=32,
                 num_filters=128,
                 residual=True,
                 use_bias=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
        super(InceptionModule, self).__init__()

        self.residual = residual
        self.bottleneck = nn.Linear(num_filters, out_features=1, bias=use_bias)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(1, num_filters//4, kernel_size=kernel_size+1, stride=1, bias=use_bias, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s]

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(num_filters, num_filters//4, kernel_size=1, padding=0, bias=use_bias)
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )

        if residual:
            self.residual_relu = nn.ReLU()

        self.to(device)


    def forward(self, input_tensor):
        input_inception = self.bottleneck(input_tensor.transpose(1,2)).transpose(1,2)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1)
        features = self.bn_relu(features)
        if self.residual:
            features = features + input_tensor
            features = self.residual_relu(features)
        return features


def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def validation_split(X, Y, v_ratio=0.2):
    strata = np.unique(Y, return_counts=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=v_ratio, random_state=3, shuffle=True, stratify=Y)
    return (X_train, y_train), (X_test, y_test)


class SatDataset(Dataset):
    def __init__(self, x, y=None, read=False, normalize=True):
        
        self.read = read
        self.normalize = normalize
        
        if self.read:
            self.x = torch.from_numpy(np.load(x)) # .permute(0,2,1)
            if y is not None:
                self.y = np.load(y)
            else:
                self.y = None
        else:
            self.x = x
            self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        xx = self.x[idx]
        if self.y is not None:
            yy = self.y[idx]
        if self.normalize:
            if self.y is None:
                return normalize(xx)
            else:
                return normalize(xx), yy
        else:
            if self.y is None:
                return xx
            else:
                return xx, yy
            

class Engine:
    def __init__(self, lr=0.001,
                 epochs=100,
                 input_dim=13,
                 num_classes=9,
                 num_layers=4,
                 hidden_dims=64,
                 out_path=None,
                 rep_path=None,
                 batch_size=32,
                 energy=0.5,
                 gamma=0.1,
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
        self.device = device # torch.device(device)
        self.temp_loss = SoftDTW(use_cuda=False, gamma=gamma) # TemporalContrast(batch_size=batch_size, energy=energy, use_cuda=False, gamma=gamma)
        self.feat_loss = FeatureContrast(batch_size=batch_size, energy=energy)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.energy = energy
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.transforms = {'jitter':Jitter(mean=0, std=0.2),
                           'smooth':Smooth(window=7, polord=3),
                           'scale':Scale(mean=0.5, std=0.8),
                           'permute': TimePermute(n_segs=7)}
        self.names = ['jitter', 'smooth', 'scale', 'permute']
        self.weight_path = f'{out_path}/weights' # need this to be ok
        self.rep_path = f'{out_path}/report'
        self.weight_name = f'lr_{lr}_nlayer_{num_layers}_hdim_{hidden_dims}.pth'
        self.rep_name = f'lr_{lr}_nlayer_{num_layers}_hdim_{hidden_dims}'
        
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path, exist_ok=True)
        if not os.path.exists(self.rep_path):
            os.makedirs(self.rep_path, exist_ok=True)
        
    def train_contrastive(self, train_loader=None):  # contrastive learning
        model = InceptionTime(input_dim=self.input_dim,
                              num_classes=self.num_classes,
                              num_layers=self.num_layers,
                              hidden_dims=self.hidden_dims,
                              use_bias=False,
                              phase='train',
                              device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        self.tloss = []
        self.floss = []
        self.c_checkpoint = f'{self.weight_path}/contrast_{self.weight_name}'
        for j in range(self.epochs):
            distort = random.sample(self.names, 2)  # ever epoch sample 
            epoch_loss = []
            for data in train_loader:
                x1 = self.transforms[distort[0]](data)  # distorted view 1
                x2 = self.transforms[distort[1]](data)  # distorted view 2
                optimizer.zero_grad()
                t1, f1 = model(x1.float())
                t2, f2 = model(x2.float())
                temp_loss = self.temp_loss(t1, t2).mean()
                feat_loss = self.feat_loss(f1, f2)
                tot_loss = temp_loss + feat_loss
                tot_loss.backward()
                optimizer.step()
                self.tloss.append(temp_loss.item())
                self.floss.append(feat_loss.item())
                print(f'epoch: {j}, temporal loss: {temp_loss.item()}, feature loss: {feat_loss.item()}, total loss: {tot_loss.item()}, contrasters: {distort[0]} --> {distort[1]}', end='\r', flush=True)
            if j %5 == 0:
                torch.save(model.state_dict(), self.c_checkpoint) # save weights ever epoch
        report = {'time_loss':self.tloss, 'feature_loss': self.floss}
        df = pd.DataFrame.from_dict(report)
        df.to_csv(f'{self.rep_path}/{self.rep_name}.csv')
        
    def adaptTrain(self, train_loader=None, valid_loader=None, from_scartch=False, test=False):
        model = InceptionTime(input_dim=self.input_dim,
                              num_classes=self.num_classes,
                              num_layers=self.num_layers,
                              hidden_dims=self.hidden_dims,
                              use_bias=False,
                              phase='adapt',
                              device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        if not from_scartch:
            self.a_checkpoint = f'{self.weight_path}/adapt_{self.weight_name}'
            self.c_checkpoint = f'{self.weight_path}/contrast_{self.weight_name}'
        else:
            self.a_checkpoint = f'{self.weight_path}/train_{self.weight_name}'
            
        if not from_scartch:
            self.model.load_state_dict(torch.load(self.c_checkpoint))
        self.atloss = []
        self.avloss = []
        self.best_weight = None
        bestv = 0
        control = 0
        tolerance = 10
        
        for j in range(self.epochs):
            s_loss = []
            for feat, label in self.train_loader:
                optimizer.zero_grad()
                ouuts = self.model(feat)
                loss = F.nll_loss(outs, label)
                loss.backward()
                optimizer.step()
                s_loss.append(loss.item())
            ms_loss = np.nanmean(s_loss)
            self.atloss.append(ms_loss)
            
            with torch.no_grad():
                vls = []
                for feat, label  in self.valid_loader:
                    out = self.model(feat)
                    loss = F.nll_loss(out, label)
                    vls.append(loss.item())
                vls = np.nanmean(vls)
                self.avloss.append(vls)
                if j == 0:
                    bestv = vls
                else:
                    if vls<=bestv:
                        bstv = vls
                        self.best_weight = self.model.state_dict()
                    else:
                        control+=1
                if control>=tolerance:
                    torch.save(self.best_weight, self.a_checkpoint)
        torch.save(self.best_weight, self.a_checkpoint)
        
    def test(self, test_loader=None, read_weight = False, sup=False, full_scene=None, tempelate=None):
        model = InceptionTime(input_dim=input_dim,
                              num_classes=num_classes,
                              num_layers=num_layers,
                              hidden_dims=hidden_dims,
                              use_bias=False,
                              phase=phase,
                              device=self.device)
        iou = 0
        f1 = 0
        c = 0
        
        
        if read_weight:
            self.model.load_state_dict(torch.load(self.a_checkpoint))
        else:
            self.load_state_dict(self.best_weight)
        with torch.no_grad():
            for feat, label in self.test_loader:
                out = self.model(feat)
                # convert soft probablit towards hard classes
                acc = self.acc_fn(out, label)  # check the order
                iou+=acc['iou']
                f1+=acc['f1s']
                c+=1
        iou = np.nansum(iou)
        f1 = np.nansum(f1)
        
        alls = []
        if full_scene is not None:
            print('Started prediction of ')
            inds = list(range(0, full_scene.shape[0], self.batch_size))
            for i in range(len(inds)):
                ins = full_scene[inds[i]:inds[i+self.batch_size],:,:]  # this shoud be the 3-D image
                outs = model(ins)
                # binarize the output
                alls.append(outs)
        alls = list(torch.concate(alls, dim=0).view(-1)) # may be check the dimension 
        shp = gpd.from_file(tempelate)
        shp['pred'] = alls
        shp.to_file(f'{self.rep_path}/{self.rep_name}.shp')
        with open(f'test_rep_{self.rep_path}/{self.rep_name}.txt', 'w') as txt:
            txt.write(f'iou: {iou} \n')
            txt.write(f'f1: {f1} \n')
            txt.close()
            


def argumentParser():
    parser = argparse.ArgumentParser(description='model agnostic meta learning implementation both for classic and MAML')
    parser.add_argument('--data_root', help='Root folder that contained all tensors folders', type=str)
    parser.add_argument('--out_path', help='path/folder to save weight', type = str)
    parser.add_argument('--log_path', help='Folder to save the reports and logs', type=str)
    parser.add_argument('--lr', help='adaptation learning rate for inner model', default=0.0001, type=float, required=False)
    parser.add_argument('--batch_size', help='batch size to load task datasets', default=100, type=int, required=False)
    parser.add_argument('--epochs', help='number of epochs to train', default=100, type=int, required=False)
    parser.add_argument('--nlayer', help='number of model layers', default=4, type=int, required=False)
    parser.add_argument('--hdim', help='number of hidden layers', default=64, type=int, required=False)
    parser.add_argument('--runner', help='Whether to optimize or train', default='train', type=str, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argumentParser()
    train_sup_pixel = f'{args.data_root}/fused.npy'
    train_feat = f'{args.data_root}/train_fused.npy'
    train_label = f'{args.data_root}/train_labels.npy'
    test_feat = f'{args.data_root}/valid_fused.npy'
    test_label = f'{args.data_root}/valid_labels.npy'
    
    
    u_dataset = SatDataset(x=train_sup_pixel, read=True)
    u_loader = DataLoader(dataset=u_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    xx_train = np.load(train_feat)
    yy_train = np.load(train_label)

    train_u, valid_u = validation_split(xx_train, yy_train) # train test split

    train_dataset = SatDataset(x=train_u[0],y=train_u[1], read=False)
    tran_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    valid_dataset = SatDataset(x=valid_u[0],y=valid_u[1], read=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    test_dataset = SatDataset(x=test_feat,y=test_label, read=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    if args.runner == 'train':
        engine = Engine(lr=args.lr,
                        num_classes=11,
                        input_dim=args.hdim,
                        num_layers=args.nlayer,
                        hidden_dims=args.hdim,
                        out_path=args.out_path,
                        rep_path=args.log_path,
                        device='cpu',
                        batch_size=args.batch_size,
                        energy=0.5,
                        gamma=0.1)
        engine.train_contrastive(u_loader)
        engine.adaptTrain(train_loader=tran_loader, valid_loader=valid_loader, from_scartch=False, test=False)
        engine.test(test_loader=test_loader, read_weight = False, sup=False, full_scene=None, tempelate=None)
        
    elif args.runner == 'optimize':
        lrs = [0.1, 0.01,0.001,0.0001,0.00001]
        nlayers = [4, 8, 16, 32, 64, 128, 256]
        hdims = [4, 8, 16, 32, 64, 128, 256]
        for lr in lrs:
            for hdim in hdims:
                for nlayer in nlayers:
                    engine = Engine(lr=lr,
                                    num_classes=11,
                                    input_dim=hdim,
                                    num_layers=args.nlayer,
                                    hidden_dims=args.hdim,
                                    out_path=args.out_path,
                                    rep_path=args.log_path,
                                    device='cpu',
                                    batch_size=args.batch_size,
                                    energy=0.5,
                                    gamma=0.1)
                    engine.train_contrastive(u_loader)
                    engine.adaptTrain(train_loader=tran_loader, valid_loader=valid_loader, from_scartch=False, test=False)
                    engine.test(test_loader=test_loader, read_weight = False, sup=False, full_scene=None, tempelate=None)
    else:
        raise ValueError(f'Provided {args.runner} is not known!')
